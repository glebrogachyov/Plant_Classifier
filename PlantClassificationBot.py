import telebot
import time
import os
import keras
from keras.models import load_model
import cv2
from PIL import Image
import imageio

import numpy as np
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input

token = "-" #(token removed)
bot = telebot.TeleBot(token)


model = load_model("W://bot/MobileNetV2_model_100ep.h5")
labels = ['Abutilon_theophrasti', 'Acalypha_rhomboidea', 'Acer_negundo', 'Acer_rubrum', 'Acer_saccharinum',
          'Acer_saccharum', 'Ageratina_altissima', 'Ailanthus_altissima', 'Alliaria_petiolata', 'Allium_cernuum',
          'Allium_tricoccum', 'Amaranthus_spinosus', 'Ambrosia_artemisiifolia', 'Ambrosia_trifida',
          'Amelanchier_arborea', 'Amphicarpa_bracteata', 'Apocynum_cannabinum', 'Arabis_laevigata', 'Artemisia_annua',
          'Asarum_canadense', 'Asclepias_syriaca', 'Asimina_triloba', 'Asplenium_platyneuron', 'Aster_cordifolius',
          'Aster_divaricatus', 'Boehmeria_cylindrica', 'Campsis_radicans', 'Carpinus_caroliniana', 'Carya_cordiformis',
          'Carya_glabra', 'Catalpa_bignonioides', 'Celastrus_orbiculatus', 'Celtis_occidentalis',
          'Cephalanthus_occidentalis', 'Cercis_canadensis', 'Chasmanthium_latifolium', 'Chionanthus_virginicus',
          'Circaea_lutetiana', 'Collinsonia_canadensis', 'Commelina_communis', 'Conium_maculatum', 'Corydalis_flavula',
          'Crataegus_crus-galli', 'Cunila_origanoides', 'Cynanchum_laeve', 'Datura_stramonium', 'Daucus_carota',
          'Desmodium_glabellum', 'Dicentra_canadensis', 'Dichanthelium_acuminatum_var._fasciculatum',
          'Dichanthelium_boscii', 'Dichanthelium_clandestinum', 'Dioscorea_villosa', 'Diospyros_virginiana',
          'Dipsacus_sylvestris', 'Dirca_palustris', 'Dryopteris_marginalis', 'Duchesnea_indica', 'Erigenia_bulbosa',
          'Erigeron_annuus', 'Erysimum_cheiranthoides', 'Erythronium_americanum', 'Euonymus_americanus',
          'Eupatorium_coelestinum', 'Eupatorium_purpureum', 'Eupatorium_serotinum', 'Euphorbia_maculata',
          'Fraxinus_americana', 'Galium_aparine', 'Geum_canadense', 'Glechoma_hederacea', 'Gleditsia_triacanthos',
          'Hamamelis_virginiana', 'Heliopsis_helianthoides', 'Hesperis_matronalis', 'Hibiscus_laevis',
          'Houstonia_purpurea', 'Humulus_japonicus', 'Hydrophyllum_canadense', 'Hydrophyllum_virginianum',
          'Hypericum_mutilum', 'Hypericum_prolificum', 'Hypericum_punctatum', 'Ilex_opaca', 'Impatiens_capensis',
          'Ipomoea_lacunosa', 'Ipomoea_pandurata', 'Jeffersonia_diphylla', 'Juglans_nigra', 'Juncus_tenuis',
          'Justicia_americana', 'Lamium_purpureum', 'Laportea_canadensis', 'Lespedeza_procumbens', 'Ligustrum_vulgare',
          'Lindera_benzoin', 'Lindernia_dubia', 'Liriodendron_tulipifera', 'Lonicera_japonica', 'Lonicera_maacki',
          'Lycopus_americanus', 'Lysimachia_nummularia', 'Maclura_pomifera', 'Malus_angustifolia', 'Melilotus_alba',
          'Melilotus_albus', 'Menispermum_canadense', 'Mertensia_virginiana', 'Microstegium_vimineum',
          'Mitchella_repens', 'Mollugo_verticillata', 'Ostrya_virginiana', 'Oxalis_stricta', 'Paronychia_canadensis',
          'Parthenocissus_quinquefolia', 'Passiflora_lutea', 'Penthorum_sedoides', 'Perilla_frutescens',
          'Phacelia_ranunculacea', 'Phlox_divaricata', 'Phytolacca_americana', 'Pilea_pumila', 'Platanus_occidentalis',
          'Podophyllum_peltatum', 'Polygonatum_biflorum', 'Polygonum_cespitosum', 'Polygonum_cuspidatum',
          'Polygonum_lapathifolium', 'Polygonum_perfoliatum', 'Polygonum_punctatum', 'Polygonum_virginianum',
          'Polymnia_uvedalia', 'Polypodium_virginianum', 'Polystichum_acrostichoides', 'Portulaca_oleracea',
          'Potentilla_canadensis', 'Prunus_serotina', 'Ptelea_trifoliata', 'Quercus_alba', 'Quercus_montana',
          'Quercus_rubra', 'Ranunculus_arbortivus', 'Ranunculus_recurvatus', 'Robinia_pseudoacacia', 'Rosa_multiflora',
          'Rubus_flagellaris', 'Rubus_phoenicolasius', 'Rudbeckia_laciniata', 'Rumex_altissimus', 'Sambucus_canadensis',
          'Saponaria_officinalis', 'Saururus_cernuus', 'Saxifraga_virginiensis', 'Scrophularia_marilandica',
          'Sedum_ternatum', 'Senecio_aureus', 'Setaria_faberi', 'Setaria_viridis', 'Sida_spinosa', 'Silene_latifolia',
          'Silene_stellata', 'Smilacina_racemosa', 'Smilax_glauca', 'Smilax_rotundifolia', 'Solanum_carolinense',
          'Solanum_nigrum', 'Solidago_caesia', 'Solidago_flexicaulis', 'Solidago_ulmifolia', 'Staphylea_trifolia',
          'Stellaria_media', 'Stellaria_pubera', 'Symphoricarpos_orbiculatus', 'Tanacetum_vulgare',
          'Teucrium_canadense', 'Tilia_americana', 'Toxicodendron_radicans', 'Tradescantia_virginiana',
          'Trillium_sessile', 'Ulmus_americana', 'Ulmus_rubra', 'Urtica_dioica', 'Uvularia_sessilifolia',
          'Vaccinium_pallidum', 'Vaccinium_stamineum', 'Verbena_hastata', 'Verbena_urticifolia',
          'Verbesina_alternifolia', 'Vernonia_noveboracensis', 'Veronica_hederifolia', 'Viburnum_prunifolium',
          'Viola_sororia', 'Viola_striata', 'Vitis_riparia', 'Xanthium_strumarium', 'Zizia_aurea']


def preprocess(img):
    print("entered preprocessing")
    h, w, ch = img.shape
    sz = max(h, w)
    result = [[[255, 255, 255] for _ in range(sz)] for _ in range(sz)]
    if sz > h:
        for it in range(sz - h):
            for j in range(sz):
                result[h + it][j] = list(img[-1][j])
    else:
        for it in range(h):
            for j in range(sz - w):
                result[it][-j - 1] = list(img[it][-1])
    for it in range(h):
        for j in range(w):
            result[it][j] = list(img[it][j])
    print("resizing done")
    result = np.array(result, dtype='uint8')
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    result = Image.fromarray(result).resize((224, 224))
    fout = "W:/bot/image2.jpg"
    imageio.imwrite(fout, result)


def classify(img_path):
    print("entered classify")
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    prediction = model.predict(img_preprocessed)
    n = 5
    y_preds = np.argsort(prediction, axis=1)[:,-n:]
    res = sorted(y_preds[0])
    resp = []
    links = []
    for i in res:
        resp.append(labels[i])
    for i in range(len(resp)):
        links.append("google.com/search?q=")
        for k in range(len(resp[i])):
            if resp[i][k] == "_":
                links[i] += "+"
            else:
                links[i] += resp[i][k]
    result = ""
    for i in range(len(resp)):
        result += "<a href='https://" + links[i] + "'>" + resp[i] + "</a>"
        if i != len(resp) - 1:
            result += "\n"
    return result


# Начало диалога
@bot.message_handler(commands=['start'])
def start_message(message):
    handle_help(message)


# Основное меню
@bot.message_handler(commands=['help'])  # Обрабатывает команду /help
def handle_help(message):
    bot.send_message(message.chat.id, "Приветствую Вас, " + message.from_user.first_name +
                     "!\nОтправьте в диалог фотографию листа растения."
                     " В ответном сообщении будут указаны 5 наиболее вероятных названий данного вида на латыни.")



# Загрузка файла от пользователя
@bot.message_handler(content_types=["photo"])
def photo(message):
    print("photo received. Processing started.")
    # print('message.photo =', message.photo)
    fileID = message.photo[-1].file_id
    # print('fileID =', fileID)
    file_info = bot.get_file(fileID)
    # print('file.file_path =', file_info.file_path)
    downloaded_file = bot.download_file(file_info.file_path)
    with open("W:/bot/image.jpg", 'wb') as new_file:
        new_file.write(downloaded_file)
    img = cv2.imread("W:/bot/image.jpg")
    preprocess(img)
    print("file preprocessed and saved")
    response = classify("W:/bot/image2.jpg")
    print("classified, responce:\n", response)
    # bot.reply_to(message, response)
    # bot.reply_to(message, text=response, parse_mode=ParseMode.HTML)
    bot.send_message(message.chat.id, text=response, parse_mode="HTML", disable_web_page_preview=True)


# Обработчик всяких непонятных ситуаций када пользователь шлёт всякую дрись
@bot.message_handler(content_types=["sticker", "pinned_message", "document", "audio", 'video', 'video_note', 'voice',
                                    'location', 'contact', 'new_chat_members', 'left_chat_member', 'new_chat_title',
                                    'new_chat_photo', 'delete_chat_photo', 'group_chat_created',
                                    'supergroup_chat_created', "text",
                                    'channel_chat_created', 'migrate_to_chat_id', 'migrate_from_chat_id',
                                    'pinned_message'])
def anthg(message):
    handle_help(message)


tries = 0
while tries < 10:
    try:
        print("trying")
        bot.polling()
    except Exception as E:
        print(E.args)
        print("\ndying...")
        tries += 1
        time.sleep(1)
