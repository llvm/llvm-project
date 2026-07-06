; RUN: llc -mcpu=gfx950 -o - -verify-regalloc %s | FileCheck %s
;
; This test is extremely fragile and chances are it won't test what it
; was meant to pretty quickly.
;
; The gist of it is we want something that triggers the hoistSpillInsideBB
; spilling optimization placement, while spilling a value that is defined
; by a copy inserted by live-range splitting within the prologue of the basic
; block (i.e., where we do our exec masks shenanigans).
; Since the spiller cannot insert the spill code within that prologue, it
; needs to extend the live-range through the prologue region and that test
; is supposed to capture that.
; The verify-regalloc flag makes sure the live-ranges are properly updated.
; Before it would complain about missing live segment.

; CHECK-LABEL: foo:


target triple = "amdgcn-amd-amdhsa"

@global_smem = external addrspace(3) global [0 x i8]

define amdgpu_kernel void @foo(ptr addrspace(1) inreg readonly captures(none) %arg, ptr addrspace(1) inreg readonly %arg1, ptr addrspace(1) inreg readonly %arg2, ptr addrspace(1) inreg %arg3, i32 inreg %arg4, i32 inreg %arg5, i32 inreg %arg6, i32 inreg %arg7, i32 inreg %arg8, i32 inreg %arg9, i32 %arg10, i32 %arg11, i32 %arg12, i32 %arg13, i32 %.frozen, i32 %arg14, i32 %arg15, i32 %arg16, i32 %arg17, i32 %arg18, i32 %arg19, i32 %arg20, i64 %arg21, i32 %arg22, i64 %arg23, i64 %arg24, i32 %arg25, i64 %arg26, i64 %arg27, i64 %arg28, i64 %arg29, i64 %arg30, i32 %arg31, i64 %arg32, i64 %arg33, ptr addrspace(8) %arg34, i32 %.decomposed, i64 %arg35, i32 %arg36, i32 %arg37, i32 %arg38, i32 %arg39, i32 %arg40, i32 %arg41, i32 %arg42, i32 %arg43, i32 %arg44, i32 %arg45, i32 %arg46, i32 %arg47, i32 %arg48, i32 %arg49, i32 %arg50, i32 %arg51, i32 %arg52, i32 %arg53, i32 %arg54, i32 %arg55, i32 %arg56, i32 %arg57, i32 %arg58, i32 %arg59, i32 %arg60, i32 %arg61, i32 %arg62, i32 %arg63, i32 %arg64, i32 %arg65, i32 %arg66, i32 %arg67, i32 %arg68, i32 %arg69, i32 %arg70, i32 %arg71, i32 %arg72, i32 %arg73, i32 %arg74, i32 %arg75, i32 %arg76, i32 %arg77, i32 %arg78, i32 %arg79, i32 %arg80, i32 %arg81, i32 %arg82, i32 %arg83, i32 %arg84, i32 %arg85, i32 %arg86, i32 %arg87, i32 %arg88, i32 %arg89, i32 %arg90, i32 %arg91, i32 %arg92, i32 %arg93, i32 %arg94, i32 %arg95, i32 %arg96, i32 %arg97, i32 %arg98, i32 %arg99, i32 %arg100, i32 %arg101, i32 %arg102, i32 %arg103, i32 %arg104, i32 %arg105, i32 %arg106, i32 %arg107, i32 %arg108, i32 %arg109, i32 %arg110, i32 %arg111, i32 %arg112, i32 %arg113, i32 %arg114, i32 %arg115, i32 %arg116, i32 %arg117, i32 %arg118, i32 %arg119, i32 %arg120, i32 %arg121, i32 %arg122, i32 %arg123, i32 %arg124, i32 %arg125, i32 %arg126, i32 %arg127, i32 %arg128, i32 %arg129, i32 %arg130, i32 %arg131, i32 %arg132, i32 %arg133, i32 %arg134, i32 %arg135, i32 %arg136, i32 %arg137, i32 %arg138, i32 %arg139, i32 %arg140, i32 %arg141, i32 %arg142, i32 %arg143, i32 %arg144, i32 %arg145, i32 %arg146, i64 %arg147, i64 %arg148, i64 %arg149, i64 %arg150, i64 %arg151, i64 %arg152, i64 %arg153, i64 %arg154, i64 %arg155, i64 %arg156, i64 %arg157, i64 %arg158, i64 %arg159, i64 %arg160, i64 %arg161, i64 %arg162, i64 %arg163, i64 %arg164, i64 %arg165, i64 %arg166, i64 %arg167, i64 %arg168, i64 %arg169, i64 %arg170, i64 %arg171, i64 %arg172, i64 %arg173, i64 %arg174, i64 %arg175, i64 %arg176, i64 %arg177, i64 %arg178, i64 %arg179, i64 %arg180, i64 %arg181, i64 %arg182, i64 %arg183, i64 %arg184, i64 %arg185, i64 %arg186, i64 %arg187, i64 %arg188, i64 %arg189, i64 %arg190, i64 %arg191, i64 %arg192, i64 %arg193, i64 %arg194, i64 %arg195, i64 %arg196, i64 %arg197, i64 %arg198, i64 %arg199, i64 %arg200, i64 %arg201, i64 %arg202, i64 %arg203, i64 %arg204, i64 %arg205, i64 %arg206, i64 %arg207, i64 %arg208, i64 %arg209, i64 %arg210, i64 %arg211, i64 %arg212, i64 %arg213, i64 %arg214, i64 %arg215, i64 %arg216, i64 %arg217, i64 %arg218, i64 %arg219, i64 %arg220, i64 %arg221, i64 %arg222, i64 %arg223, i64 %arg224, i64 %arg225, i64 %arg226, i64 %arg227, i64 %arg228, i64 %arg229, i64 %arg230, i64 %arg231, i64 %arg232, i64 %arg233, i64 %arg234, i64 %arg235, i64 %arg236, i64 %arg237, i64 %arg238, i64 %arg239, i64 %arg240, i64 %arg241, i64 %arg242, i64 %arg243, i64 %arg244, i64 %arg245, i64 %arg246, i64 %arg247, i64 %arg248, i64 %arg249, i64 %arg250, i64 %arg251, i64 %arg252, i64 %arg253, i64 %arg254, i64 %arg255, i64 %arg256, i64 %arg257, i64 %arg258, i64 %arg259, i64 %arg260, i64 %arg261, i64 %arg262, i64 %arg263, i64 %arg264, i64 %arg265, i64 %arg266, i64 %arg267, i64 %arg268, i64 %arg269, i64 %arg270, i64 %arg271, i64 %arg272, i64 %arg273, i64 %arg274, i64 %arg275, i64 %arg276, i64 %arg277, i64 %arg278, i64 %arg279, i64 %arg280, i64 %arg281, i64 %arg282, i64 %arg283, i64 %arg284, i64 %arg285, i64 %arg286, i64 %arg287, i64 %arg288, i64 %arg289, i64 %arg290, i64 %arg291, i64 %arg292, i64 %arg293, i64 %arg294, i64 %arg295, i64 %arg296, i64 %arg297, i64 %arg298, i64 %arg299, i64 %arg300, i64 %arg301, i64 %arg302, i64 %arg303, i64 %arg304, i64 %arg305, i64 %arg306, i64 %arg307, i64 %arg308, i64 %arg309, i64 %arg310, i64 %arg311, i64 %arg312, i64 %arg313, i64 %arg314, i64 %arg315, i64 %arg316, i64 %arg317, i64 %arg318, i64 %arg319, i64 %arg320, i64 %arg321, i64 %arg322, i64 %arg323, i64 %arg324, i64 %arg325, i64 %arg326, i64 %arg327, i64 %arg328, i64 %arg329, i64 %arg330, i64 %arg331, i64 %arg332, i64 %arg333, i64 %arg334, i64 %arg335, i64 %arg336, i64 %arg337, i64 %arg338, i64 %arg339, i64 %arg340, i64 %arg341, i64 %arg342, i64 %arg343, i64 %arg344, i64 %arg345, i64 %arg346, i64 %arg347, i64 %arg348, i64 %arg349, i64 %arg350, i64 %arg351, i64 %arg352, i64 %arg353, i64 %arg354, i64 %arg355, i64 %arg356, i64 %arg357, i64 %arg358, i64 %arg359, i64 %arg360, i64 %arg361, i64 %arg362, i64 %arg363, i64 %arg364, i64 %arg365, i64 %arg366, i64 %arg367, i64 %arg368, i64 %arg369, i64 %arg370, i64 %arg371, i64 %arg372, i64 %arg373, i64 %arg374, i64 %arg375, i64 %arg376, i64 %arg377, i64 %arg378, i64 %arg379, i64 %arg380, i64 %arg381, i64 %arg382, i64 %arg383, i64 %arg384, i64 %arg385, i64 %arg386, i64 %arg387, i64 %arg388, i64 %arg389, i64 %arg390, i64 %arg391, i64 %arg392, i64 %arg393, i64 %arg394, i64 %arg395, i64 %arg396, i64 %arg397, i64 %arg398, i64 %arg399, i64 %arg400, i64 %arg401, i64 %arg402, i64 %arg403, i64 %arg404, i64 %arg405, i64 %arg406, i64 %arg407, i64 %arg408, i64 %arg409, i64 %arg410, i64 %arg411, i64 %arg412, i64 %arg413, i64 %arg414, i64 %arg415, i64 %arg416, i64 %arg417, i64 %arg418, i64 %arg419, i64 %arg420, i64 %arg421, i64 %arg422, i64 %arg423, i64 %arg424, i64 %arg425, i64 %arg426, i64 %arg427, i64 %arg428, i64 %arg429, i64 %arg430, i64 %arg431, i64 %arg432, i64 %arg433, i64 %arg434, i64 %arg435, i64 %arg436, i64 %arg437, i64 %arg438, i64 %arg439, i64 %arg440, i64 %arg441, i64 %arg442, i64 %arg443, i64 %arg444, i64 %arg445, i64 %arg446, i64 %arg447, i64 %arg448, i64 %arg449, i64 %arg450, i64 %arg451, i64 %arg452, i64 %arg453, i64 %arg454, i64 %arg455, i64 %arg456, i64 %arg457, i64 %arg458, i64 %arg459, i64 %arg460, i64 %arg461, i64 %arg462, i64 %arg463, i64 %arg464, i64 %arg465, i64 %arg466, i64 %arg467, i64 %arg468, i64 %arg469, i64 %arg470, i64 %arg471, i64 %arg472, i64 %arg473, i64 %arg474, i64 %arg475, i64 %arg476, i64 %arg477, i64 %arg478, i64 %arg479, i64 %arg480, i64 %arg481, i64 %arg482, i64 %arg483, i64 %arg484, i64 %arg485, i64 %arg486, i64 %arg487, i64 %arg488, i64 %arg489, i64 %arg490, i64 %arg491, i64 %arg492, i64 %arg493, i64 %arg494, i64 %arg495, i64 %arg496, i64 %arg497, i64 %arg498, i64 %arg499, i64 %arg500, i64 %arg501, i64 %arg502, i64 %arg503, i64 %arg504, i64 %arg505, i64 %arg506, i64 %arg507, i64 %arg508, i64 %arg509, i64 %arg510, i64 %arg511, i64 %arg512, i64 %arg513, i64 %arg514, i64 %arg515, i64 %arg516, i64 %arg517, i64 %arg518, i64 %arg519, i64 %arg520, i64 %arg521, i64 %arg522, i64 %arg523, i64 %arg524, i64 %arg525, i64 %arg526, i64 %arg527, i64 %arg528, i64 %arg529, i64 %arg530, i64 %arg531, i64 %arg532, i64 %arg533, i64 %arg534, i64 %arg535, i64 %arg536, i64 %arg537, i64 %arg538, i64 %arg539, i64 %arg540, i64 %arg541, i64 %arg542, i64 %arg543, i64 %arg544, i64 %arg545, i64 %arg546, i64 %arg547, i64 %arg548, i64 %arg549, i64 %arg550, i64 %arg551, i64 %arg552, i64 %arg553, i64 %arg554, i64 %arg555, i64 %arg556, i64 %arg557, i64 %arg558, i64 %arg559, i64 %arg560, i64 %arg561, i64 %arg562, i64 %arg563, i64 %arg564, i64 %arg565, i64 %arg566, i64 %arg567, i64 %arg568, i64 %arg569, i64 %arg570, i64 %arg571, i64 %arg572, i64 %arg573, i64 %arg574, i64 %arg575, i64 %arg576, i64 %arg577, i64 %arg578, i64 %arg579, i1 %arg580, i1 %arg581, i1 %arg582, i1 %arg583, i1 %arg584, i1 %arg585, i1 %arg586, i1 %arg587, i1 %arg588, i1 %arg589, i1 %arg590, i1 %arg591, i1 %arg592, i1 %arg593, i1 %arg594, i1 %arg595, i1 %arg596, i1 %arg597, i1 %arg598, i1 %arg599, i1 %arg600, i1 %arg601, i1 %arg602, i1 %arg603, i1 %arg604, i1 %arg605, i1 %arg606, i1 %arg607, i1 %arg608, i1 %arg609, i1 %arg610, i1 %arg611, i1 %arg612, i1 %arg613, i1 %arg614, i1 %arg615, i1 %arg616, i1 %arg617, i1 %arg618, i1 %arg619, i1 %arg620, i1 %arg621, i1 %arg622, i1 %arg623, i1 %arg624, i1 %arg625, i1 %arg626, i1 %arg627, i1 %arg628, i1 %arg629, i1 %arg630, i1 %arg631, i1 %arg632, i1 %arg633, i1 %arg634, i1 %arg635, i1 %arg636, i1 %arg637, i1 %arg638, i1 %arg639, i1 %arg640, i1 %arg641, i1 %arg642, i1 %arg643, i1 %arg644, i1 %arg645, i1 %arg646, i1 %arg647, i1 %arg648, i1 %arg649, i1 %arg650, i1 %arg651, i1 %arg652, i1 %arg653, i1 %arg654, i1 %arg655, i1 %arg656, i1 %arg657, i1 %arg658, i1 %arg659, i1 %arg660, i1 %arg661, i1 %arg662, i1 %arg663, i1 %arg664, i1 %arg665, i1 %arg666, i1 %arg667, i1 %arg668, i1 %arg669, i1 %arg670, i1 %arg671, i1 %arg672, i1 %arg673, i1 %arg674, i1 %arg675, i1 %arg676, i1 %arg677, i1 %arg678, i1 %arg679, i1 %arg680, i1 %arg681, i1 %arg682, i1 %arg683, i1 %arg684, i32 %arg685, i1 %arg686, i32 %arg687, ptr addrspace(8) %arg688, i32 %arg689, i32 %arg690, i1 %arg691, i32 %arg692, i32 %arg693, i1 %arg694, i32 %arg695, i32 %arg696, i1 %arg697, i32 %arg698, i32 %arg699, i32 %arg700, i1 %arg701, i32 %arg702, i32 %arg703, i32 %arg704, i1 %arg705, i32 %arg706, i32 %arg707, i1 %arg708, i32 %arg709, i32 %arg710, i32 %arg711, i1 %arg712, i32 %arg713, i32 %arg714, i32 %arg715, i1 %arg716, i32 %arg717, i32 %arg718, i1 %arg719, i32 %arg720, i32 %arg721, i32 %arg722, i1 %arg723, i32 %arg724, i32 %arg725, i32 %arg726, i1 %arg727, i32 %arg728, i32 %arg729, i32 %arg730, i32 %arg731, i32 %arg732, i32 %arg733, i32 %arg734, i1 %arg735, i32 %arg736, i32 %arg737, i32 %arg738, i1 %arg739, i32 %arg740, i32 %arg741, i32 %arg742, i1 %arg743, i32 %arg744, i32 %arg745, i32 %arg746, i1 %arg747, i32 %arg748, i32 %arg749, i32 %arg750, i1 %arg751, i32 %arg752, i32 %arg753, i32 %arg754, i32 %arg755, i32 %arg756, i1 %arg757, i32 %arg758, i32 %arg759, i32 %arg760, i1 %arg761, i32 %arg762, i32 %arg763, i32 %arg764, i1 %arg765, i32 %arg766, i32 %arg767, i32 %arg768, i32 %arg769, i32 %arg770, i1 %arg771, i1 %arg772, i32 %arg773, i1 %arg774, i1 %arg775, i1 %arg776, i1 %arg777, i1 %arg778, i32 %arg779, i1 %arg780, i32 %arg781, i32 %arg782, i32 %arg783, i1 %arg784, i32 %arg785, i32 %arg786, i32 %arg787, i1 %arg788, i32 %arg789, i32 %arg790, i32 %arg791, i1 %arg792, i32 %arg793, i32 %arg794, i32 %arg795, i1 %arg796, i32 %arg797, i32 %arg798, i32 %arg799, i1 %arg800, i32 %arg801, i32 %arg802, i32 %arg803, i1 %arg804, i32 %arg805, i32 %arg806, i32 %arg807, i1 %arg808, i32 %arg809, i32 %arg810, i32 %arg811, i1 %arg812, i32 %arg813, i32 %arg814, i32 %arg815, i1 %arg816, i32 %arg817, i32 %arg818, i32 %arg819, i1 %arg820, i32 %arg821, i32 %arg822, i32 %arg823, i1 %arg824, i32 %arg825, i32 %arg826, i32 %arg827, i1 %arg828, i32 %arg829, i32 %arg830, i32 %arg831, ptr addrspace(3) %global_smem, i32 %arg832, i16 %arg833, i32 %arg834, ptr addrspace(3) %arg835, i16 %arg836, i16 %arg837, i16 %arg838, i16 %arg839, i16 %arg840, i16 %arg841, i16 %arg842, i32 %arg843, i16 %arg844, i32 %arg845, i16 %arg846, ptr addrspace(3) %arg847, i16 %arg848, i32 %arg849, i16 %arg850, ptr addrspace(3) %arg851, i16 %arg852, i32 %arg853, i16 %arg854, ptr addrspace(3) %arg855, i32 %arg856, i16 %arg857, i16 %arg858, i16 %arg859, i32 %arg860, i32 %arg861, i16 %arg862, ptr addrspace(3) %arg863, i16 %arg864, i16 %arg865, i16 %arg866, i32 %arg867, i16 %arg868, i32 %arg869, i16 %arg870, ptr addrspace(3) %arg871, i16 %arg872, i32 %arg873, i16 %arg874, ptr addrspace(3) %arg875, i16 %arg876, i32 %arg877, i16 %arg878, ptr addrspace(3) %arg879, i16 %arg880, i16 %arg881, i16 %arg882, i32 %arg883, i16 %arg884, ptr addrspace(3) %arg885, i16 %arg886, i16 %arg887, i16 %arg888, i16 %arg889, i32 %arg890, i16 %arg891, ptr addrspace(3) %arg892, i16 %arg893, i32 %arg894, i16 %arg895, ptr addrspace(3) %arg896, i16 %arg897, i32 %arg898, i16 %arg899, ptr addrspace(3) %arg900, i16 %arg901, i32 %arg902, i1 %arg903, i32 %arg904, i32 %arg905, i32 %arg906, i32 %arg907, i32 %arg908, i32 %arg909, i32 %arg910, i32 %arg911, i32 %arg912, i32 %arg913, i32 %arg914, i32 %arg915, i32 %arg916, i32 %arg917, i32 %arg918, i32 %arg919, i32 %arg920, i32 %arg921, i32 %arg922, i32 %arg923, i32 %arg924, i32 %arg925, i32 %arg926, i32 %arg927, i32 %arg928, i32 %arg929, i32 %arg930, i32 %arg931, i32 %arg932, i32 %arg933, i32 %arg934, i32 %arg935, i32 %arg936, i32 %arg937, i32 %arg938, i32 %arg939, i32 %arg940, i32 %arg941, i32 %arg942, i32 %arg943, i32 %arg944, i32 %arg945, i32 %arg946, i32 %arg947, i32 %arg948, i32 %arg949, i32 %arg950, i32 %arg951, i32 %arg952, i32 %arg953, i32 %arg954, i32 %arg955, i32 %arg956, i32 %arg957, i32 %arg958, i32 %arg959, i32 %arg960, i32 %arg961, i32 %arg962, i32 %arg963, i32 %arg964, i32 %arg965, i32 %arg966, i32 %arg967, i32 %arg968, i32 %arg969, i32 %arg970, i32 %arg971, i32 %arg972, i32 %arg973, i32 %arg974, i32 %arg975, i32 %arg976, i32 %arg977, i32 %arg978, i32 %arg979, i32 %arg980, i32 %arg981, i32 %arg982, i32 %arg983, i32 %arg984, i32 %arg985, i32 %arg986, i32 %arg987, i32 %arg988, i32 %arg989, i32 %arg990, i32 %arg991, i32 %arg992, i32 %arg993, i32 %arg994, i32 %arg995, i32 %arg996, i32 %arg997, i32 %arg998, i32 %arg999, float %arg1000, float %arg1001, i32 %arg1002, i32 %arg1003, i32 %arg1004, i32 %.pn3661, i32 %.pn261662, i32 %.pn259663, i32 %.pn257664, i32 %.pn253666, i32 %.pn249668, i32 %.pn247669, i32 %.pn245670, i32 %.pn243671, i32 %.pn241672, i32 %.pn239673, i32 %.pn237674, i32 %.pn233676, i32 %.pn231677, i32 %.pn223681, i32 %.pn105740, i32 %.pn103741, i32 %.pn101742, i32 %.pn99743, i32 %.pn93746, i32 %.pn91747, i32 %.pn89748, i32 %.pn87749, i32 %.pn85750, i32 %.pn83751, i32 %.pn81752, i32 %.pn79753, i32 %.pn77754, i32 %.pn75755, i32 %.pn73756, i32 %.pn71757, i32 %.pn69758, i32 %.pn67759, i32 %.pn65760, i32 %.pn63761, i32 %.pn61762, i32 %.pn59763, i32 %.pn57764, i32 %.pn55765, i32 %.pn53766, i32 %.pn51767, i32 %.pn49768, i32 %.pn41772, i1 %arg1005, i1 %arg1006, i1 %arg1007, i1 %arg1008, i1 %arg1009, i32 %arg1010, i32 %arg1011, i1 %arg1012, i32 %arg1013, i32 %arg1014, i1 %arg1015, i32 %arg1016, i32 %arg1017, i1 %arg1018, i32 %arg1019, i32 %arg1020, i1 %arg1021, i32 %arg1022, i32 %arg1023, i32 %arg1024, i1 %arg1025, i32 %arg1026, i32 %arg1027, i1 %arg1028, i32 %arg1029, i32 %arg1030, i1 %arg1031, i32 %arg1032, i32 %arg1033, i1 %arg1034, i32 %arg1035, i1 %arg1036, i32 %arg1037, i1 %arg1038, i32 %arg1039, i1 %arg1040, i32 %arg1041, i32 %arg1042, i1 %arg1043, i32 %arg1044, i1 %arg1045, i32 %arg1046, i1 %arg1047, i32 %arg1048, i1 %arg1049, i32 %arg1050, i1 %arg1051, i32 %arg1052, i1 %arg1053, i1 %arg1054, i32 %arg1055, i32 %arg1056, i1 %arg1057, i32 %arg1058, i1 %arg1059, i32 %arg1060, i32 %arg1061, float %arg1062, <4 x float> %arg1063, float %arg1064, <4 x float> %arg1065, <4 x float> %arg1066, <4 x float> %arg1067, <4 x float> %arg1068, <4 x float> %arg1069, <4 x float> %arg1070, <4 x float> %arg1071, <4 x float> %arg1072, <4 x float> %arg1073, <4 x float> %arg1074, ptr addrspace(3) %arg1075, i32 %arg1076, i32 %arg1077, ptr addrspace(3) %arg1078, i32 %arg1079, ptr addrspace(3) %arg1080, ptr addrspace(3) %arg1081, i32 %arg1082, ptr addrspace(3) %arg1083, i32 %arg1084, i32 %arg1085, i32 %arg1086, ptr addrspace(3) %arg1087, i32 %arg1088, i32 %arg1089, i32 %arg1090, ptr addrspace(3) %arg1091, ptr addrspace(3) %arg1092, i32 %arg1093, i32 %arg1094, ptr addrspace(3) %arg1095, i32 %arg1096, i32 %arg1097, i32 %arg1098, ptr addrspace(3) %arg1099, ptr addrspace(3) %arg1100, i32 %arg1101, ptr addrspace(3) %arg1102, i32 %arg1103, i16 %arg1104, ptr addrspace(3) %arg1105, i32 %arg1106, i16 %arg1107, i16 %arg1108, i32 %arg1109, ptr addrspace(3) %arg1110, i32 %arg1111, i32 %arg1112, i32 %arg1113, i32 %arg1114, i32 %arg1115, i32 %arg1116, i32 %arg1117, i32 %arg1118, i32 %arg1119, i32 %arg1120, i32 %arg1121, i32 %arg1122, i32 %arg1123, i32 %arg1124, i32 %arg1125, i32 %arg1126, i32 %arg1127, i32 %arg1128, i32 %arg1129, i32 %arg1130, i32 %arg1131, i32 %arg1132, i32 %arg1133, i32 %arg1134, i32 %arg1135, i32 %arg1136, i32 %arg1137, i32 %arg1138, i32 %arg1139, i32 %arg1140, i32 %arg1141, i32 %arg1142, i32 %arg1143, i32 %arg1144, i32 %arg1145, i32 %arg1146, i32 %arg1147, i32 %arg1148, i32 %arg1149, i32 %arg1150, i32 %arg1151, i32 %arg1152, i32 %arg1153, i32 %arg1154, i32 %arg1155, i32 %arg1156, i32 %arg1157, i32 %arg1158, i32 %arg1159, i1 %arg1160, i1 %arg1161, i1 %arg1162, i1 %arg1163, i1 %arg1164, i1 %arg1165, i1 %arg1166, i1 %arg1167, i1 %arg1168, i1 %arg1169, i1 %arg1170, i1 %arg1171, i1 %arg1172, i1 %arg1173, i1 %arg1174, i1 %arg1175, i1 %arg1176, i1 %arg1177, i1 %arg1178, i1 %arg1179, i1 %arg1180, i1 %arg1181, i1 %arg1182, i1 %arg1183, i1 %arg1184, i1 %arg1185, i1 %arg1186, i1 %arg1187, i1 %arg1188, i1 %arg1189, i1 %arg1190, i1 %arg1191, i1 %arg1192, i1 %arg1193, i1 %arg1194, i1 %arg1195, i1 %arg1196, i1 %arg1197, i1 %arg1198, i1 %arg1199, i1 %arg1200, i1 %arg1201, i1 %arg1202, i1 %arg1203, i1 %arg1204, i1 %arg1205, i1 %arg1206, i1 %arg1207, ptr addrspace(8) %arg1208, i32 %arg1209, i1 %arg1210, i32 %arg1211, i1 %arg1212, i32 %arg1213, i1 %arg1214, i32 %arg1215, i1 %arg1216, i32 %arg1217, i1 %arg1218, i32 %arg1219, i1 %arg1220, i32 %arg1221, i1 %arg1222, i32 %arg1223, i1 %arg1224, i32 %arg1225, i1 %arg1226, i32 %arg1227, i1 %arg1228, i32 %arg1229, i1 %arg1230, i32 %arg1231, i1 %arg1232, i32 %arg1233, i1 %arg1234, i32 %arg1235, i1 %arg1236, i32 %arg1237, i1 %arg1238, i32 %arg1239, i1 %arg1240, i32 %arg1241, i1 %arg1242, i32 %arg1243, i1 %arg1244, i32 %arg1245, i1 %arg1246, i32 %arg1247, i1 %arg1248, i32 %arg1249, i1 %arg1250, i32 %arg1251, i1 %arg1252, i32 %arg1253, i1 %arg1254, i32 %arg1255, i1 %arg1256, i32 %arg1257, i1 %arg1258, i32 %arg1259, i1 %arg1260, i32 %arg1261, i1 %arg1262, i32 %arg1263, i1 %arg1264, i32 %arg1265, i1 %arg1266, i32 %arg1267, i1 %arg1268, i32 %arg1269, i1 %arg1270, i32 %arg1271, i1 %arg1272, i32 %arg1273, i1 %arg1274, i32 %arg1275, i1 %arg1276, i32 %arg1277, i1 %arg1278, i32 %arg1279, i1 %arg1280, i32 %arg1281, i1 %arg1282, i32 %arg1283, i1 %arg1284, i32 %arg1285, i1 %arg1286, i32 %arg1287, i1 %arg1288, i32 %arg1289, i1 %arg1290, i32 %arg1291, i1 %arg1292, i32 %arg1293, i1 %arg1294, i32 %arg1295, i1 %arg1296, i32 %arg1297, i1 %arg1298, i32 %arg1299, i32 %.idx, ptr addrspace(3) %arg1300, i16 %arg1301, ptr addrspace(3) %arg1302, i16 %arg1303, i16 %arg1304, ptr addrspace(3) %arg1305, i16 %arg1306, i16 %arg1307, i16 %arg1308, ptr addrspace(3) %arg1309, i16 %arg1310, i16 %arg1311, i32 %arg1312, ptr addrspace(3) %arg1313, i32 %arg1314, i16 %arg1315, ptr addrspace(3) %arg1316, i32 %arg1317, i16 %arg1318, ptr addrspace(3) %arg1319, i16 %arg1320, i32 %arg1321, i16 %arg1322, ptr addrspace(3) %arg1323, i16 %arg1324, ptr addrspace(3) %arg1325, i32 %arg1326, i16 %arg1327, ptr addrspace(3) %arg1328, i16 %arg1329, ptr addrspace(3) %arg1330, i16 %arg1331, i32 %arg1332, i16 %arg1333, i16 %arg1334, i16 %arg1335, ptr addrspace(3) %arg1336, i32 %arg1337, ptr addrspace(3) %arg1338, i32 %arg1339, ptr addrspace(3) %arg1340, i16 %arg1341, i32 %arg1342, i16 %arg1343, i16 %arg1344, i16 %arg1345, i32 %arg1346, i16 %arg1347, i16 %arg1348, i16 %arg1349) #0 {
bb:
  %i = tail call i32 @llvm.amdgcn.workitem.id.x()
  %i1350 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %arg4)
  %i1351 = and i32 %arg4, 448
  %i1352 = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %i1353 = add i32 0, 0
  %i1354 = sdiv i32 0, 0
  %.frozen1 = freeze i32 %arg4
  %i1355 = sdiv i32 %arg4, %arg4
  %i1356 = mul i32 %arg4, %arg4
  %.decomposed2 = sub i32 %arg4, %arg4
  %i1357 = and i32 %arg4, 63
  %i1358 = or disjoint i32 %arg4, %arg17
  %i1359 = and i32 %arg11, 256
  %i1360 = lshr exact i32 %arg18, 0
  %i1361 = or disjoint i32 %arg18, 2
  %i1362 = or disjoint i32 %arg4, 4
  %i1363 = or disjoint i32 %arg4, 6
  %i1364 = or disjoint i32 %arg4, 8
  %i1365 = or disjoint i32 %arg4, 10
  %i1366 = or disjoint i32 %arg4, 12
  %i1367 = or disjoint i32 0, 14
  %i1368 = or disjoint i32 %arg4, 16
  %i1369 = or disjoint i32 %arg4, 18
  %i1370 = or disjoint i32 0, 20
  %i1371 = or disjoint i32 %arg4, 22
  %i1372 = or disjoint i32 %arg4, 24
  %i1373 = or disjoint i32 %arg4, 26
  %i1374 = or disjoint i32 %arg4, 28
  %i1375 = or disjoint i32 0, 30
  %i1376 = or disjoint i32 %arg4, 32
  %i1377 = or disjoint i32 %arg4, 34
  %i1378 = or disjoint i32 0, 36
  %i1379 = or disjoint i32 0, 38
  %i1380 = or disjoint i32 %arg4, 40
  %i1381 = or disjoint i32 %arg4, 42
  %i1382 = or disjoint i32 %arg4, 44
  %i1383 = or disjoint i32 %arg4, 46
  %i1384 = or disjoint i32 %arg4, 48
  %i1385 = or disjoint i32 %arg4, 50
  %i1386 = or disjoint i32 %arg4, 52
  %i1387 = or disjoint i32 %arg4, 56
  %i1388 = or disjoint i32 %arg4, 58
  %i1389 = or disjoint i32 %arg4, 60
  %i1390 = or disjoint i32 %arg4, 62
  %i1391 = or disjoint i32 %arg4, 64
  %i1392 = or disjoint i32 %arg4, 66
  %i1393 = or disjoint i32 %arg4, 68
  %i1394 = or disjoint i32 %arg4, 70
  %i1395 = or disjoint i32 %arg4, 72
  %i1396 = or disjoint i32 %arg4, 74
  %i1397 = or disjoint i32 %arg4, 76
  %i1398 = or disjoint i32 %arg4, 78
  %i1399 = or disjoint i32 %arg4, 80
  %i1400 = or disjoint i32 %arg4, 82
  %i1401 = or disjoint i32 %arg4, 84
  %i1402 = or disjoint i32 %arg4, 86
  %i1403 = or disjoint i32 %arg4, 88
  %i1404 = or disjoint i32 %arg4, 90
  %i1405 = or disjoint i32 %arg4, 92
  %i1406 = or disjoint i32 %arg4, 94
  %i1407 = or disjoint i32 %arg4, 96
  %i1408 = or disjoint i32 %arg4, 98
  %i1409 = or disjoint i32 %arg4, 100
  %i1410 = or disjoint i32 %arg4, 102
  %i1411 = or disjoint i32 %arg4, 104
  %i1412 = or disjoint i32 %arg4, 106
  %i1413 = or disjoint i32 %arg4, 108
  %i1414 = or disjoint i32 %arg4, 110
  %i1415 = or disjoint i32 %arg4, 112
  %i1416 = or disjoint i32 %arg4, 114
  %i1417 = or disjoint i32 0, 116
  %i1418 = or disjoint i32 0, 118
  %i1419 = or disjoint i32 %arg4, 120
  %i1420 = or disjoint i32 %arg4, 122
  %i1421 = or disjoint i32 0, 0
  %i1422 = or disjoint i32 0, 1
  %i1423 = or disjoint i32 %arg4, 0
  %i1424 = or disjoint i32 0, 130
  %i1425 = or disjoint i32 0, 0
  %i1426 = or disjoint i32 0, 1
  %i1427 = or disjoint i32 0, 1
  %i1428 = or disjoint i32 0, 0
  %i1429 = or disjoint i32 0, 0
  %i1430 = or disjoint i32 0, 150
  %i1431 = or disjoint i32 0, 0
  %i1432 = or disjoint i32 %arg4, 154
  %i1433 = or disjoint i32 %arg4, 156
  %i1434 = or disjoint i32 0, 158
  %i1435 = or disjoint i32 %arg4, 160
  %i1436 = or disjoint i32 0, 162
  %i1437 = or disjoint i32 0, 164
  %i1438 = or disjoint i32 %arg4, 166
  %i1439 = or disjoint i32 %arg4, 168
  %i1440 = or disjoint i32 %arg4, 170
  %i1441 = or disjoint i32 %arg4, 172
  %i1442 = or disjoint i32 %arg4, 174
  %i1443 = or disjoint i32 %arg4, 176
  %i1444 = or disjoint i32 %arg4, 178
  %i1445 = or disjoint i32 %arg4, 180
  %i1446 = or disjoint i32 %arg4, 182
  %i1447 = or disjoint i32 %arg4, 184
  %i1448 = or disjoint i32 %arg4, 186
  %i1449 = or disjoint i32 %arg4, 188
  %i1450 = or disjoint i32 %arg4, 190
  %i1451 = or disjoint i32 %arg4, 192
  %i1452 = or disjoint i32 %arg4, 194
  %i1453 = or disjoint i32 %arg4, 196
  %i1454 = or disjoint i32 %arg4, 198
  %i1455 = or disjoint i32 %arg4, 200
  %i1456 = or disjoint i32 %arg4, 202
  %i1457 = or disjoint i32 %arg4, 204
  %i1458 = or disjoint i32 %arg4, 206
  %i1459 = or disjoint i32 %arg4, 208
  %i1460 = or disjoint i32 %arg4, 210
  %i1461 = or disjoint i32 %arg4, 212
  %i1462 = or disjoint i32 %arg4, 214
  %i1463 = or disjoint i32 %arg4, 216
  %i1464 = or disjoint i32 %arg4, 218
  %i1465 = or disjoint i32 %arg4, 1
  %i1466 = or disjoint i32 %arg4, 222
  %i1467 = or disjoint i32 %arg4, 224
  %i1468 = or disjoint i32 %arg4, 226
  %i1469 = or disjoint i32 %arg4, 228
  %i1470 = or disjoint i32 1, 230
  %i1471 = or disjoint i32 %arg4, 232
  %i1472 = or disjoint i32 %arg4, 1
  %i1473 = or disjoint i32 %arg4, 236
  %i1474 = or disjoint i32 %arg4, 238
  %i1475 = or disjoint i32 %arg4, 240
  %i1476 = or disjoint i32 %arg4, 242
  %i1477 = or disjoint i32 %arg4, 244
  %i1478 = or disjoint i32 %arg4, 246
  %i1479 = or disjoint i32 %arg4, 248
  %i1480 = or disjoint i32 1, 250
  %i1481 = or disjoint i32 %arg4, 252
  %i1482 = or disjoint i32 %arg19, 254
  %i1483 = and i32 %arg20, 255
  %i1484 = sext i32 %arg13 to i64
  %i1485 = shl nsw i64 %arg21, 0
  %i1486 = zext nneg i32 %arg4 to i64
  %i1487 = zext nneg i32 %arg4 to i64
  %i1488 = or disjoint i64 %arg21, %arg21
  %i1489 = zext i32 %arg4 to i64
  %i1490 = zext nneg i32 %arg4 to i64
  %i1491 = mul i64 %arg23, 1
  %i1492 = mul nuw nsw i64 %arg24, %arg26
  %i1493 = add i64 %arg27, 1
  %i1494 = add i64 %arg28, %arg29
  %i1495 = trunc i64 %arg30 to i32
  %i1496 = add i32 %arg6, 1
  %i1497 = sdiv i32 %arg31, 256
  %i1498 = icmp sgt i32 %arg31, 255
  %i1499 = sext i32 %arg4 to i64
  %i1500 = icmp slt i64 %arg32, %arg33
  %i1501 = icmp slt i32 %arg25, 1
  %i1502 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %arg, i16 0, i64 2147483646, i32 159744)
  %i1503 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 0, i32 0, i32 0)
  %i1504 = sext i32 %.decomposed to i64
  %i1505 = shl nsw i64 %arg35, 1
  %i1506 = zext nneg i32 %arg7 to i64
  %i1507 = zext nneg i32 %arg4 to i64
  %i1508 = zext nneg i32 %arg4 to i64
  %i1509 = zext nneg i32 %arg4 to i64
  %i1510 = zext nneg i32 %arg4 to i64
  %i1511 = zext nneg i32 %arg4 to i64
  %i1512 = zext nneg i32 %arg4 to i64
  %i1513 = zext nneg i32 %arg4 to i64
  %i1514 = zext nneg i32 %arg4 to i64
  %i1515 = zext nneg i32 %arg4 to i64
  %i1516 = zext nneg i32 %arg4 to i64
  %i1517 = zext nneg i32 %arg4 to i64
  %i1518 = zext nneg i32 %arg4 to i64
  %i1519 = zext nneg i32 %arg4 to i64
  %i1520 = zext nneg i32 %arg4 to i64
  %i1521 = zext nneg i32 %arg4 to i64
  %i1522 = zext nneg i32 %arg4 to i64
  %i1523 = zext nneg i32 %arg4 to i64
  %i1524 = zext nneg i32 %arg4 to i64
  %i1525 = zext nneg i32 %arg4 to i64
  %i1526 = zext nneg i32 %arg4 to i64
  %i1527 = zext nneg i32 %arg4 to i64
  %i1528 = zext nneg i32 %arg4 to i64
  %i1529 = zext nneg i32 %arg4 to i64
  %i1530 = zext nneg i32 %arg4 to i64
  %i1531 = zext nneg i32 %arg4 to i64
  %i1532 = zext nneg i32 %arg4 to i64
  %i1533 = zext nneg i32 %arg4 to i64
  %i1534 = zext nneg i32 %arg4 to i64
  %i1535 = zext nneg i32 %arg4 to i64
  %i1536 = zext nneg i32 %arg4 to i64
  %i1537 = zext nneg i32 %arg4 to i64
  %i1538 = zext nneg i32 %arg4 to i64
  %i1539 = zext nneg i32 %arg4 to i64
  %i1540 = zext nneg i32 %arg4 to i64
  %i1541 = zext nneg i32 %arg4 to i64
  %i1542 = zext nneg i32 %arg4 to i64
  %i1543 = zext nneg i32 %arg4 to i64
  %i1544 = zext nneg i32 %arg4 to i64
  %i1545 = zext nneg i32 %arg4 to i64
  %i1546 = zext nneg i32 %arg4 to i64
  %i1547 = zext nneg i32 %arg4 to i64
  %i1548 = zext nneg i32 %arg4 to i64
  %i1549 = zext nneg i32 %arg4 to i64
  %i1550 = zext nneg i32 %arg4 to i64
  %i1551 = zext nneg i32 %arg4 to i64
  %i1552 = zext nneg i32 %arg4 to i64
  %i1553 = zext nneg i32 %arg4 to i64
  %i1554 = zext nneg i32 %arg4 to i64
  %i1555 = zext nneg i32 %arg4 to i64
  %i1556 = zext nneg i32 %arg4 to i64
  %i1557 = zext nneg i32 %arg4 to i64
  %i1558 = zext nneg i32 %arg4 to i64
  %i1559 = zext nneg i32 %arg4 to i64
  %i1560 = zext nneg i32 %arg4 to i64
  %i1561 = zext nneg i32 %arg4 to i64
  %i1562 = zext nneg i32 %arg4 to i64
  %i1563 = zext nneg i32 %arg4 to i64
  %i1564 = zext nneg i32 %arg4 to i64
  %i1565 = zext nneg i32 0 to i64
  %i1566 = zext nneg i32 1 to i64
  %i1567 = zext nneg i32 %arg4 to i64
  %i1568 = zext nneg i32 %arg4 to i64
  %i1569 = zext nneg i32 %arg4 to i64
  %i1570 = zext nneg i32 %arg4 to i64
  %i1571 = zext nneg i32 %arg4 to i64
  %i1572 = zext nneg i32 %arg4 to i64
  %i1573 = zext nneg i32 %arg4 to i64
  %i1574 = zext nneg i32 %arg4 to i64
  %i1575 = zext nneg i32 %arg4 to i64
  %i1576 = zext nneg i32 %arg4 to i64
  %i1577 = zext nneg i32 %arg4 to i64
  %i1578 = zext nneg i32 %arg4 to i64
  %i1579 = zext nneg i32 %arg4 to i64
  %i1580 = zext nneg i32 %arg4 to i64
  %i1581 = zext nneg i32 %arg4 to i64
  %i1582 = zext nneg i32 %arg4 to i64
  %i1583 = zext nneg i32 %arg4 to i64
  %i1584 = zext nneg i32 %arg4 to i64
  %i1585 = zext nneg i32 %arg4 to i64
  %i1586 = zext nneg i32 %arg4 to i64
  %i1587 = zext nneg i32 %arg4 to i64
  %i1588 = zext nneg i32 %arg4 to i64
  %i1589 = zext nneg i32 %arg4 to i64
  %i1590 = zext nneg i32 %arg4 to i64
  %i1591 = zext nneg i32 %arg4 to i64
  %i1592 = zext nneg i32 %arg4 to i64
  %i1593 = zext nneg i32 %arg4 to i64
  %i1594 = zext nneg i32 %arg4 to i64
  %i1595 = zext nneg i32 %arg4 to i64
  %i1596 = zext nneg i32 %arg4 to i64
  %i1597 = zext nneg i32 %arg4 to i64
  %i1598 = zext nneg i32 %arg4 to i64
  %i1599 = zext nneg i32 %arg4 to i64
  %i1600 = zext nneg i32 %arg4 to i64
  %i1601 = zext nneg i32 %arg4 to i64
  %i1602 = zext nneg i32 %arg4 to i64
  %i1603 = zext nneg i32 %arg4 to i64
  %i1604 = zext nneg i32 %arg4 to i64
  %i1605 = zext nneg i32 %arg4 to i64
  %i1606 = zext nneg i32 %arg4 to i64
  %i1607 = zext nneg i32 %arg4 to i64
  %i1608 = zext nneg i32 %arg4 to i64
  %i1609 = zext nneg i32 %arg4 to i64
  %i1610 = zext nneg i32 %arg4 to i64
  %i1611 = zext nneg i32 %arg4 to i64
  %i1612 = zext nneg i32 %arg4 to i64
  %i1613 = zext nneg i32 %arg4 to i64
  %i1614 = zext nneg i32 %arg4 to i64
  %i1615 = zext nneg i32 %arg4 to i64
  %i1616 = zext nneg i32 %arg4 to i64
  %i1617 = zext nneg i32 %arg4 to i64
  %i1618 = zext nneg i32 %arg4 to i64
  %i1619 = or disjoint i64 %arg21, %arg21
  %i1620 = or disjoint i64 1, %arg21
  %i1621 = or disjoint i64 %arg21, %arg21
  %i1622 = or disjoint i64 %arg21, %arg21
  %i1623 = or disjoint i64 %arg21, %arg21
  %i1624 = or disjoint i64 1, %arg21
  %i1625 = or disjoint i64 %arg21, %arg21
  %i1626 = or disjoint i64 %arg21, %arg21
  %i1627 = or disjoint i64 1, %arg21
  %i1628 = or disjoint i64 %arg21, %arg21
  %i1629 = or disjoint i64 %arg21, %arg21
  %i1630 = or disjoint i64 %arg21, %arg21
  %i1631 = or disjoint i64 %arg21, %arg21
  %i1632 = or disjoint i64 %arg21, %arg21
  %i1633 = or disjoint i64 %arg21, %arg21
  %i1634 = or disjoint i64 %arg21, %arg21
  %i1635 = or disjoint i64 %arg21, %arg21
  %i1636 = or disjoint i64 %arg21, %arg21
  %i1637 = or disjoint i64 %arg21, %arg21
  %i1638 = or disjoint i64 %arg21, %arg21
  %i1639 = or disjoint i64 %arg21, %arg21
  %i1640 = or disjoint i64 %arg21, %arg21
  %i1641 = or disjoint i64 %arg21, %arg21
  %i1642 = or disjoint i64 1, %arg21
  %i1643 = or disjoint i64 1, %arg21
  %i1644 = or disjoint i64 %arg21, %arg21
  %i1645 = or disjoint i64 %arg21, %arg21
  %i1646 = or disjoint i64 %arg21, %arg21
  %i1647 = or disjoint i64 %arg21, %arg21
  %i1648 = or disjoint i64 %arg21, %arg21
  %i1649 = or disjoint i64 %arg21, %arg21
  %i1650 = or disjoint i64 %arg21, %arg21
  %i1651 = or disjoint i64 %arg21, %arg21
  %i1652 = or disjoint i64 %arg21, %arg21
  %i1653 = or disjoint i64 %arg21, %arg21
  %i1654 = or disjoint i64 %arg21, %arg21
  %i1655 = or disjoint i64 %arg21, %arg21
  %i1656 = or disjoint i64 %arg21, %arg21
  %i1657 = or disjoint i64 %arg21, %arg21
  %i1658 = or disjoint i64 %arg21, %arg21
  %i1659 = or disjoint i64 %arg21, %arg21
  %i1660 = or disjoint i64 %arg21, %arg21
  %i1661 = or disjoint i64 %arg21, %arg21
  %i1662 = or disjoint i64 %arg21, %arg21
  %i1663 = or disjoint i64 %arg21, %arg21
  %i1664 = or disjoint i64 %arg21, %arg21
  %i1665 = or disjoint i64 1, %arg21
  %i1666 = or disjoint i64 %arg21, %arg21
  %i1667 = or disjoint i64 %arg21, %arg21
  %i1668 = or disjoint i64 %arg21, %arg21
  %i1669 = or disjoint i64 %arg21, %arg21
  %i1670 = or disjoint i64 %arg21, %arg21
  %i1671 = or disjoint i64 %arg21, %arg21
  %i1672 = or disjoint i64 %arg21, %arg21
  %i1673 = or disjoint i64 %arg21, %arg21
  %i1674 = or disjoint i64 %arg21, %arg21
  %i1675 = or disjoint i64 %arg21, %arg21
  %i1676 = or disjoint i64 %arg21, %arg21
  %i1677 = or disjoint i64 %arg21, %arg21
  %i1678 = or disjoint i64 0, %arg21
  %i1679 = or disjoint i64 0, %arg21
  %i1680 = or disjoint i64 0, 0
  %i1681 = or disjoint i64 1, 1
  %i1682 = or disjoint i64 %arg21, %arg21
  %i1683 = or disjoint i64 %arg21, %arg21
  %i1684 = or disjoint i64 %arg21, %arg21
  %i1685 = or disjoint i64 %arg21, %arg21
  %i1686 = or disjoint i64 %arg21, %arg21
  %i1687 = or disjoint i64 %arg21, %arg21
  %i1688 = or disjoint i64 %arg21, %arg21
  %i1689 = or disjoint i64 %arg21, %arg21
  %i1690 = or disjoint i64 0, %arg21
  %i1691 = or disjoint i64 %arg21, %arg21
  %i1692 = or disjoint i64 %arg21, %arg21
  %i1693 = or disjoint i64 1, %arg21
  %i1694 = or disjoint i64 1, %arg21
  %i1695 = or disjoint i64 1, %arg21
  %i1696 = or disjoint i64 1, %arg21
  %i1697 = or disjoint i64 1, %arg21
  %i1698 = or disjoint i64 1, %arg21
  %i1699 = or disjoint i64 1, %arg21
  %i1700 = or disjoint i64 1, %arg21
  %i1701 = or disjoint i64 1, %arg21
  %i1702 = or disjoint i64 1, %arg21
  %i1703 = or disjoint i64 1, %arg21
  %i1704 = or disjoint i64 0, %arg21
  %i1705 = or disjoint i64 %arg21, %arg21
  %i1706 = or disjoint i64 0, %arg21
  %i1707 = or disjoint i64 %arg21, %arg21
  %i1708 = or disjoint i64 1, %arg21
  %i1709 = or disjoint i64 %arg21, 0
  %i1710 = or disjoint i64 0, %arg21
  %i1711 = or disjoint i64 1, %arg21
  %i1712 = or disjoint i64 %arg21, %arg21
  %i1713 = or disjoint i64 %arg21, 1
  %i1714 = or disjoint i64 1, %arg21
  %i1715 = or disjoint i64 1, %arg21
  %i1716 = or disjoint i64 %arg21, 1
  %i1717 = or disjoint i64 1, %arg21
  %i1718 = or disjoint i64 1, %arg21
  %i1719 = or disjoint i64 1, %arg21
  %i1720 = or disjoint i64 1, %arg21
  %i1721 = or disjoint i64 1, %arg21
  %i1722 = or disjoint i64 0, %arg21
  %i1723 = or disjoint i64 0, %arg21
  %i1724 = or disjoint i64 1, %arg21
  %i1725 = or disjoint i64 1, %arg21
  %i1726 = or disjoint i64 0, %arg21
  %i1727 = or disjoint i64 0, 0
  %i1728 = or disjoint i64 1, %arg21
  %i1729 = or disjoint i64 0, %arg21
  %i1730 = or disjoint i64 0, %arg21
  %i1731 = or disjoint i64 0, %arg21
  %i1732 = or disjoint i64 0, %arg21
  %i1733 = or disjoint i64 0, %arg21
  %i1734 = or disjoint i64 %arg21, %arg21
  %i1735 = zext i32 %arg4 to i64
  %i1736 = mul nsw i64 %arg21, %arg21
  %i1737 = mul nuw nsw i64 %arg21, %arg21
  %i1738 = mul nuw nsw i64 %arg21, %arg21
  %i1739 = mul nuw nsw i64 %arg21, %arg21
  %i1740 = mul nuw nsw i64 %arg21, %arg21
  %i1741 = mul nuw nsw i64 %arg21, %arg21
  %i1742 = mul nuw nsw i64 %arg21, %arg21
  %i1743 = mul nuw nsw i64 %arg21, %arg21
  %i1744 = mul nuw nsw i64 %arg21, %arg21
  %i1745 = mul nuw nsw i64 %arg21, %arg21
  %i1746 = mul nuw nsw i64 %arg21, %arg21
  %i1747 = mul nuw nsw i64 %arg21, %arg21
  %i1748 = mul nuw nsw i64 %arg21, %arg21
  %i1749 = mul nuw nsw i64 %arg21, %arg21
  %i1750 = mul nuw nsw i64 %arg21, %arg21
  %i1751 = mul nuw nsw i64 %arg21, %arg21
  %i1752 = mul nuw nsw i64 %arg21, %arg21
  %i1753 = mul nuw nsw i64 %arg21, %arg21
  %i1754 = mul nuw nsw i64 0, 0
  %i1755 = mul nuw nsw i64 0, 0
  %i1756 = mul nuw nsw i64 %arg21, %arg21
  %i1757 = mul nuw nsw i64 %arg21, 1
  %i1758 = mul nuw nsw i64 0, 0
  %i1759 = mul nuw nsw i64 %arg21, %arg21
  %i1760 = mul nuw nsw i64 %arg21, 1
  %i1761 = mul nuw nsw i64 %arg21, %arg21
  %i1762 = mul nuw nsw i64 %arg21, %arg21
  %i1763 = mul nuw nsw i64 %arg21, %arg21
  %i1764 = mul nuw nsw i64 %arg21, %arg21
  %i1765 = mul nuw nsw i64 %arg21, %arg21
  %i1766 = mul nuw nsw i64 %arg21, %arg21
  %i1767 = mul nuw nsw i64 %arg21, %arg21
  %i1768 = mul nuw nsw i64 %arg21, %arg21
  %i1769 = mul nuw nsw i64 %arg21, %arg21
  %i1770 = mul nuw nsw i64 %arg21, %arg21
  %i1771 = mul nuw nsw i64 %arg21, %arg21
  %i1772 = mul nuw nsw i64 %arg21, %arg21
  %i1773 = mul nuw nsw i64 %arg21, %arg21
  %i1774 = mul nuw nsw i64 %arg21, %arg21
  %i1775 = mul nuw nsw i64 %arg21, %arg21
  %i1776 = mul nuw nsw i64 %arg21, %arg21
  %i1777 = mul nuw nsw i64 %arg21, %arg21
  %i1778 = mul nuw nsw i64 %arg21, %arg21
  %i1779 = mul nuw nsw i64 %arg21, %arg21
  %i1780 = mul nuw nsw i64 %arg21, %arg21
  %i1781 = mul nuw nsw i64 %arg21, %arg21
  %i1782 = mul nuw nsw i64 %arg21, %arg21
  %i1783 = mul nuw nsw i64 %arg21, 1
  %i1784 = mul nuw nsw i64 %arg21, %arg21
  %i1785 = mul nuw nsw i64 %arg21, %arg21
  %i1786 = mul nuw nsw i64 %arg21, %arg21
  %i1787 = mul nuw nsw i64 %arg21, 1
  %i1788 = mul nuw nsw i64 %arg21, %arg21
  %i1789 = mul nuw nsw i64 %arg21, %arg21
  %i1790 = mul nuw nsw i64 %arg21, %arg21
  %i1791 = mul nuw nsw i64 %arg21, 1
  %i1792 = mul nuw nsw i64 1, %arg21
  %i1793 = mul nuw nsw i64 1, %arg21
  %i1794 = mul nuw nsw i64 %arg21, %arg21
  %i1795 = mul nuw nsw i64 %arg21, %arg21
  %i1796 = mul nuw nsw i64 0, 0
  %i1797 = mul nuw nsw i64 %arg21, %arg21
  %i1798 = mul nuw nsw i64 %arg21, %arg21
  %i1799 = mul nuw nsw i64 %arg21, %arg21
  %i1800 = mul nuw nsw i64 %arg21, %arg21
  %i1801 = mul nuw nsw i64 %arg21, %arg21
  %i1802 = mul nuw nsw i64 %arg21, %arg21
  %i1803 = mul nuw nsw i64 %arg21, %arg21
  %i1804 = mul nuw nsw i64 %arg21, %arg21
  %i1805 = mul nuw nsw i64 %arg21, %arg21
  %i1806 = mul nuw nsw i64 %arg21, %arg21
  %i1807 = mul nuw nsw i64 %arg21, %arg21
  %i1808 = mul nuw nsw i64 %arg21, %arg21
  %i1809 = mul nuw nsw i64 %arg21, %arg21
  %i1810 = mul nuw nsw i64 %arg21, %arg21
  %i1811 = mul nuw nsw i64 %arg21, %arg21
  %i1812 = mul nuw nsw i64 %arg21, %arg21
  %i1813 = mul nuw nsw i64 %arg21, %arg21
  %i1814 = mul nuw nsw i64 %arg21, %arg21
  %i1815 = mul nuw nsw i64 %arg21, %arg21
  %i1816 = mul nuw nsw i64 %arg21, %arg21
  %i1817 = mul nuw nsw i64 %arg21, %arg21
  %i1818 = mul nuw nsw i64 %arg21, %arg21
  %i1819 = mul nuw nsw i64 %arg21, %arg21
  %i1820 = mul nuw nsw i64 %arg21, %arg21
  %i1821 = mul nuw nsw i64 %arg21, %arg21
  %i1822 = mul nuw nsw i64 %arg21, %arg21
  %i1823 = mul nuw nsw i64 %arg21, 1
  %i1824 = mul nuw nsw i64 %arg21, %arg21
  %i1825 = mul nuw nsw i64 %arg21, %arg21
  %i1826 = mul nuw nsw i64 %arg21, %arg21
  %i1827 = mul nuw nsw i64 %arg21, %arg21
  %i1828 = mul nuw nsw i64 %arg21, %arg21
  %i1829 = mul nuw nsw i64 %arg21, %arg21
  %i1830 = mul nuw nsw i64 %arg21, %arg21
  %i1831 = mul nuw nsw i64 %arg21, %arg21
  %i1832 = mul nuw nsw i64 %arg21, %arg21
  %i1833 = mul nuw nsw i64 %arg21, 1
  %i1834 = mul nuw nsw i64 %arg21, %arg21
  %i1835 = mul nuw nsw i64 %arg21, %arg21
  %i1836 = mul nuw nsw i64 %arg21, %arg21
  %i1837 = mul nuw nsw i64 %arg21, %arg21
  %i1838 = mul nuw nsw i64 %arg21, %arg21
  %i1839 = mul nuw nsw i64 %arg21, %arg21
  %i1840 = mul nuw nsw i64 %arg21, %arg21
  %i1841 = mul nuw nsw i64 %arg21, %arg21
  %i1842 = or disjoint i64 %arg21, %arg21
  %i1843 = add i64 %arg21, %arg21
  %i1844 = add i64 %arg21, %arg21
  %i1845 = add i64 %arg21, %arg21
  %i1846 = add i64 %arg21, %arg21
  %i1847 = add i64 %arg21, %arg21
  %i1848 = add i64 0, %arg21
  %i1849 = add i64 %arg21, %arg21
  %i1850 = add i64 %arg21, %arg21
  %i1851 = add i64 %arg21, %arg21
  %i1852 = add i64 %arg21, %arg21
  %i1853 = add i64 %arg21, %arg21
  %i1854 = add i64 %arg21, %arg21
  %i1855 = add i64 0, %arg21
  %i1856 = add i64 0, %arg21
  %i1857 = add i64 %arg21, %arg21
  %i1858 = add i64 %arg21, %arg21
  %i1859 = add i64 0, %arg21
  %i1860 = add i64 0, 0
  %i1861 = add i64 0, 0
  %i1862 = add i64 %arg21, %arg21
  %i1863 = add i64 0, %arg21
  %i1864 = add i64 %arg21, 0
  %i1865 = add i64 0, %arg21
  %i1866 = add i64 0, %arg21
  %i1867 = add i64 0, %arg21
  %i1868 = add i64 0, %arg21
  %i1869 = add i64 0, %arg21
  %i1870 = add i64 0, %arg21
  %i1871 = add i64 0, %arg21
  %i1872 = add i64 0, %arg21
  %i1873 = add i64 0, %arg21
  %i1874 = add i64 0, %arg21
  %i1875 = add i64 0, %arg21
  %i1876 = add i64 0, %arg21
  %i1877 = add i64 0, %arg21
  %i1878 = add i64 0, %arg21
  %i1879 = add i64 0, %arg21
  %i1880 = add i64 0, %arg21
  %i1881 = add i64 0, %arg21
  %i1882 = add i64 0, %arg21
  %i1883 = add i64 0, %arg21
  %i1884 = add i64 0, %arg21
  %i1885 = add i64 0, %arg21
  %i1886 = add i64 0, %arg21
  %i1887 = add i64 0, %arg21
  %i1888 = add i64 0, %arg21
  %i1889 = add i64 0, %arg21
  %i1890 = add i64 0, %arg21
  %i1891 = add i64 0, %arg21
  %i1892 = add i64 0, %arg21
  %i1893 = add i64 0, %arg21
  %i1894 = add i64 0, %arg21
  %i1895 = add i64 0, %arg21
  %i1896 = add i64 0, %arg21
  %i1897 = add i64 0, %arg21
  %i1898 = add i64 0, %arg21
  %i1899 = add i64 %arg21, %arg21
  %i1900 = add i64 %arg21, %arg21
  %i1901 = add i64 %arg21, %arg21
  %i1902 = add i64 0, 0
  %i1903 = add i64 1, %arg21
  %i1904 = add i64 1, %arg21
  %i1905 = add i64 %arg21, %arg21
  %i1906 = add i64 %arg21, %arg21
  %i1907 = add i64 %arg21, %arg21
  %i1908 = add i64 %arg21, %arg21
  %i1909 = add i64 %arg21, %arg21
  %i1910 = add i64 %arg21, %arg21
  %i1911 = add i64 %arg21, %arg21
  %i1912 = add i64 %arg21, %arg21
  %i1913 = add i64 %arg21, %arg21
  %i1914 = add i64 %arg21, %arg21
  %i1915 = add i64 %arg21, %arg21
  %i1916 = add i64 %arg21, %arg21
  %i1917 = add i64 %arg21, %arg21
  %i1918 = add i64 %arg21, %arg21
  %i1919 = add i64 %arg21, %arg21
  %i1920 = add i64 %arg21, %arg21
  %i1921 = add i64 %arg21, %arg21
  %i1922 = add i64 %arg21, %arg21
  %i1923 = add i64 %arg21, %arg21
  %i1924 = add i64 %arg21, %arg21
  %i1925 = add i64 %arg21, %arg21
  %i1926 = add i64 1, %arg21
  %i1927 = add i64 0, %arg21
  %i1928 = add i64 0, %arg21
  %i1929 = add i64 0, %arg21
  %i1930 = add i64 %arg21, %arg21
  %i1931 = add i64 %arg21, %arg21
  %i1932 = add i64 %arg21, %arg21
  %i1933 = add i64 %arg21, %arg21
  %i1934 = add i64 %arg21, %arg21
  %i1935 = add i64 %arg21, %arg21
  %i1936 = add i64 0, %arg21
  %i1937 = add i64 1, %arg21
  %i1938 = add i64 %arg21, %arg21
  %i1939 = add i64 %arg21, %arg21
  %i1940 = add i64 %arg21, %arg21
  %i1941 = add i64 %arg21, %arg21
  %i1942 = add i64 %arg21, %arg21
  %i1943 = add i64 0, %arg21
  %i1944 = add i64 0, %arg21
  %i1945 = add i64 0, %arg21
  %i1946 = add i64 0, %arg21
  %i1947 = add i64 0, %arg21
  %i1948 = trunc i64 %arg365 to i32
  %i1949 = trunc i64 %arg366 to i32
  %i1950 = trunc i64 %arg367 to i32
  %i1951 = trunc i64 %arg368 to i32
  %i1952 = trunc i64 %arg369 to i32
  %i1953 = trunc i64 %arg370 to i32
  %i1954 = trunc i64 %arg371 to i32
  %i1955 = trunc i64 %arg372 to i32
  %i1956 = trunc i64 %arg373 to i32
  %i1957 = trunc i64 %arg374 to i32
  %i1958 = trunc i64 %arg155 to i32
  %i1959 = trunc i64 %arg21 to i32
  %i1960 = trunc i64 %arg21 to i32
  %i1961 = trunc i64 %arg21 to i32
  %i1962 = trunc i64 %arg21 to i32
  %i1963 = trunc i64 %arg21 to i32
  %i1964 = trunc i64 %arg21 to i32
  %i1965 = trunc i64 0 to i32
  %i1966 = trunc i64 0 to i32
  %i1967 = trunc i64 %arg21 to i32
  %i1968 = trunc i64 %arg21 to i32
  %i1969 = trunc i64 %arg21 to i32
  %i1970 = trunc i64 %arg21 to i32
  %i1971 = trunc i64 %arg21 to i32
  %i1972 = trunc i64 %arg387 to i32
  %i1973 = trunc i64 %arg388 to i32
  %i1974 = trunc i64 %arg389 to i32
  %i1975 = trunc i64 %arg390 to i32
  %i1976 = trunc i64 %arg23 to i32
  %i1977 = trunc i64 %arg21 to i32
  %i1978 = trunc i64 %arg21 to i32
  %i1979 = trunc i64 %arg394 to i32
  %i1980 = trunc i64 %arg395 to i32
  %i1981 = trunc i64 %arg396 to i32
  %i1982 = trunc i64 %arg397 to i32
  %i1983 = trunc i64 %arg30 to i32
  %i1984 = trunc i64 %arg21 to i32
  %i1985 = trunc i64 %arg21 to i32
  %i1986 = trunc i64 %arg21 to i32
  %i1987 = trunc i64 %arg21 to i32
  %i1988 = trunc i64 %arg21 to i32
  %i1989 = trunc i64 %arg21 to i32
  %i1990 = trunc i64 %arg21 to i32
  %i1991 = trunc i64 %arg21 to i32
  %i1992 = trunc i64 %arg21 to i32
  %i1993 = trunc i64 %arg21 to i32
  %i1994 = trunc i64 %arg21 to i32
  %i1995 = trunc i64 %arg21 to i32
  %i1996 = trunc i64 %arg411 to i32
  %i1997 = trunc i64 %arg410 to i32
  %i1998 = trunc i64 %arg21 to i32
  %i1999 = trunc i64 %arg21 to i32
  %i2000 = trunc i64 %arg21 to i32
  %i2001 = trunc i64 %arg21 to i32
  %i2002 = trunc i64 %arg21 to i32
  %i2003 = trunc i64 %arg21 to i32
  %i2004 = trunc i64 %arg21 to i32
  %i2005 = trunc i64 %arg21 to i32
  %i2006 = trunc i64 %arg21 to i32
  %i2007 = trunc i64 %arg21 to i32
  %i2008 = trunc i64 0 to i32
  %i2009 = trunc i64 %arg21 to i32
  %i2010 = trunc i64 %arg21 to i32
  %i2011 = trunc i64 %arg424 to i32
  %i2012 = trunc i64 %arg425 to i32
  %i2013 = trunc i64 %arg426 to i32
  %i2014 = trunc i64 %arg427 to i32
  %i2015 = trunc i64 %arg428 to i32
  %i2016 = trunc i64 %arg429 to i32
  %i2017 = trunc i64 %arg430 to i32
  %i2018 = trunc i64 %arg431 to i32
  %i2019 = trunc i64 %arg432 to i32
  %i2020 = trunc i64 %arg433 to i32
  %i2021 = trunc i64 %arg434 to i32
  %i2022 = trunc i64 %arg435 to i32
  %i2023 = trunc i64 %arg436 to i32
  %i2024 = trunc i64 %arg437 to i32
  %i2025 = trunc i64 %arg438 to i32
  %i2026 = trunc i64 %arg439 to i32
  %i2027 = trunc i64 %arg179 to i32
  %i2028 = trunc i64 %arg21 to i32
  %i2029 = trunc i64 %arg21 to i32
  %i2030 = trunc i64 %arg21 to i32
  %i2031 = trunc i64 %arg21 to i32
  %i2032 = trunc i64 %arg21 to i32
  %i2033 = trunc i64 %arg21 to i32
  %i2034 = trunc i64 %arg21 to i32
  %i2035 = trunc i64 %arg21 to i32
  %i2036 = trunc i64 %arg21 to i32
  %i2037 = trunc i64 %arg21 to i32
  %i2038 = trunc i64 %arg21 to i32
  %i2039 = trunc i64 %arg21 to i32
  %i2040 = trunc i64 %arg21 to i32
  %i2041 = trunc i64 %arg21 to i32
  %i2042 = trunc i64 %arg455 to i32
  %i2043 = trunc i64 %arg456 to i32
  %i2044 = trunc i64 %arg457 to i32
  %i2045 = trunc i64 %arg458 to i32
  %i2046 = trunc i64 %arg459 to i32
  %i2047 = trunc i64 %arg460 to i32
  %i2048 = trunc i64 %arg461 to i32
  %i2049 = trunc i64 %arg462 to i32
  %i2050 = trunc i64 %arg180 to i32
  %i2051 = trunc i64 %arg21 to i32
  %i2052 = trunc i64 %arg21 to i32
  %i2053 = trunc i64 %arg21 to i32
  %i2054 = sext i32 %arg4 to i64
  %i2055 = icmp slt i64 %arg21, %arg468
  %i2056 = icmp slt i64 %arg469, %arg234
  %i2057 = icmp slt i64 %arg470, %arg172
  %i2058 = icmp slt i64 %arg471, %arg160
  %i2059 = icmp slt i64 %arg472, %arg35
  %i2060 = icmp slt i64 %arg473, %arg232
  %i2061 = icmp slt i64 %arg474, %arg166
  %i2062 = icmp slt i64 %arg475, %arg199
  %i2063 = icmp slt i64 %arg21, %arg21
  %i2064 = icmp slt i64 %arg21, %arg21
  %i2065 = icmp slt i64 %arg21, %arg21
  %i2066 = icmp slt i64 %arg21, %arg21
  %i2067 = icmp slt i64 %arg21, %arg21
  %i2068 = icmp slt i64 %arg21, %arg21
  %i2069 = icmp slt i64 %arg21, %arg21
  %i2070 = icmp slt i64 %arg21, %arg21
  %i2071 = icmp slt i64 %arg21, %arg21
  %i2072 = icmp slt i64 %arg21, %arg21
  %i2073 = icmp slt i64 %arg21, %arg21
  %i2074 = icmp slt i64 %arg21, %arg21
  %i2075 = icmp slt i64 %arg21, %arg21
  %i2076 = icmp slt i64 %arg21, %arg21
  %i2077 = icmp slt i64 %arg21, %arg21
  %i2078 = icmp slt i64 %arg21, %arg21
  %i2079 = icmp slt i64 %arg21, %arg21
  %i2080 = icmp slt i64 %arg21, %arg21
  %i2081 = icmp slt i64 %arg149, %arg21
  %i2082 = icmp slt i64 %arg164, %arg21
  %i2083 = icmp slt i64 %arg177, %arg21
  %i2084 = icmp slt i64 %arg189, %arg21
  %i2085 = icmp slt i64 %arg151, %arg21
  %i2086 = icmp slt i64 %arg21, %arg468
  %i2087 = icmp slt i64 %arg171, %arg21
  %i2088 = icmp slt i64 %arg501, %arg468
  %i2089 = icmp slt i64 %arg502, %arg468
  %i2090 = icmp slt i64 %arg503, %arg468
  %i2091 = icmp slt i64 %arg504, %arg233
  %i2092 = icmp slt i64 %arg505, %arg21
  %i2093 = icmp slt i64 %arg506, %arg239
  %i2094 = icmp slt i64 %arg507, %arg468
  %i2095 = icmp slt i64 %arg508, %arg28
  %i2096 = icmp slt i64 %arg212, %arg21
  %i2097 = icmp slt i64 %arg211, %arg26
  %i2098 = icmp slt i64 %arg21, %arg21
  %i2099 = icmp slt i64 %arg512, %arg184
  %i2100 = icmp slt i64 %arg513, %arg179
  %i2101 = icmp slt i64 %arg514, %arg224
  %i2102 = icmp slt i64 %arg515, %arg468
  %i2103 = icmp slt i64 %arg516, %arg468
  %i2104 = icmp slt i64 %arg517, %arg468
  %i2105 = icmp slt i64 %arg518, %arg468
  %i2106 = icmp slt i64 %arg519, %arg468
  %i2107 = icmp slt i64 %arg520, %arg170
  %i2108 = icmp slt i64 %arg521, %arg158
  %i2109 = icmp slt i64 %arg522, %arg194
  %i2110 = icmp slt i64 %arg523, %arg468
  %i2111 = icmp slt i64 %arg524, %arg468
  %i2112 = icmp slt i64 %arg525, %arg468
  %i2113 = icmp slt i64 %arg526, %arg468
  %i2114 = icmp slt i64 %arg527, %arg468
  %i2115 = icmp slt i64 %arg528, %arg30
  %i2116 = icmp slt i64 0, %arg468
  %i2117 = icmp slt i64 1, %arg184
  %i2118 = icmp slt i64 %arg529, %arg468
  %i2119 = icmp slt i64 0, 0
  %i2120 = icmp slt i64 %arg530, %arg192
  %i2121 = icmp slt i64 %arg531, %arg170
  %i2122 = icmp slt i64 %arg532, %arg35
  %i2123 = icmp slt i64 %arg533, %arg223
  %i2124 = icmp slt i64 %arg534, %arg147
  %i2125 = icmp slt i64 %arg535, %arg170
  %i2126 = icmp slt i64 %arg536, %arg30
  %i2127 = icmp slt i64 %arg537, %arg468
  %i2128 = icmp slt i64 %arg538, %arg185
  %i2129 = icmp slt i64 %arg539, %arg205
  %i2130 = icmp slt i64 %arg540, %arg224
  %i2131 = icmp slt i64 %arg541, %arg468
  %i2132 = icmp slt i64 %arg542, %arg468
  %i2133 = icmp slt i64 %arg543, %arg468
  %i2134 = icmp slt i64 %arg544, %arg468
  %i2135 = icmp slt i64 %arg545, %arg468
  %i2136 = icmp slt i64 %arg546, %arg166
  %i2137 = icmp slt i64 %arg547, %arg228
  %i2138 = icmp slt i64 %arg548, %arg468
  %i2139 = icmp slt i64 %arg549, %arg468
  %i2140 = icmp slt i64 %arg550, %arg468
  %i2141 = icmp slt i64 %arg551, %arg468
  %i2142 = icmp slt i64 %arg552, %arg468
  %i2143 = icmp slt i64 %arg553, %arg468
  %i2144 = icmp slt i64 %arg554, %arg182
  %i2145 = icmp slt i64 %arg555, %arg21
  %i2146 = icmp slt i64 %arg556, %arg26
  %i2147 = icmp slt i64 %arg557, %arg27
  %i2148 = icmp slt i64 %arg558, %arg28
  %i2149 = icmp slt i64 %arg559, %arg28
  %i2150 = icmp slt i64 %arg162, 1
  %i2151 = icmp slt i64 %arg21, %arg21
  %i2152 = icmp slt i64 %arg21, %arg21
  %i2153 = icmp slt i64 %arg21, %arg21
  %i2154 = icmp slt i64 %arg564, %arg156
  %i2155 = icmp slt i64 %arg565, %arg250
  %i2156 = icmp slt i64 %arg566, %arg30
  %i2157 = icmp slt i64 %arg567, %arg185
  %i2158 = icmp slt i64 %arg568, %arg227
  %i2159 = icmp slt i64 %arg569, %arg468
  %i2160 = icmp slt i64 %arg570, %arg154
  %i2161 = icmp slt i64 %arg571, %arg192
  %i2162 = icmp slt i64 %arg572, %arg179
  %i2163 = icmp slt i64 %arg573, %arg468
  %i2164 = icmp slt i64 0, %arg468
  %i2165 = icmp slt i64 %arg574, %arg468
  %i2166 = icmp slt i64 %arg575, %arg468
  %i2167 = icmp slt i64 %arg174, %arg468
  %i2168 = icmp slt i64 %arg157, %arg21
  %i2169 = icmp slt i64 %arg21, %arg21
  %i2170 = icmp slt i64 %arg21, %arg21
  %i2171 = and i1 %arg580, %arg581
  %i2172 = and i1 %arg580, %arg582
  %i2173 = and i1 %arg580, %arg583
  %i2174 = and i1 %arg580, %arg584
  %i2175 = and i1 %arg580, %arg585
  %i2176 = and i1 %arg580, %arg586
  %i2177 = and i1 %arg580, %arg587
  %i2178 = and i1 %arg580, %arg588
  %i2179 = and i1 %arg580, %arg589
  %i2180 = and i1 %arg580, %arg590
  %i2181 = and i1 %arg580, %arg591
  %i2182 = and i1 %arg580, %arg592
  %i2183 = and i1 %arg580, %arg593
  %i2184 = and i1 %arg580, %arg594
  %i2185 = and i1 %arg580, %arg595
  %i2186 = and i1 %arg580, %arg596
  %i2187 = and i1 %arg580, %arg597
  %i2188 = and i1 %arg580, %arg598
  %i2189 = and i1 %arg580, %arg599
  %i2190 = and i1 %arg580, %arg600
  %i2191 = and i1 %arg580, %arg601
  %i2192 = and i1 %arg580, %arg602
  %i2193 = and i1 %arg580, %arg603
  %i2194 = and i1 %arg580, %arg604
  %i2195 = and i1 %arg580, %arg605
  %i2196 = and i1 %arg580, %arg606
  %i2197 = and i1 %arg580, %arg607
  %i2198 = and i1 %arg580, %arg608
  %i2199 = and i1 %arg580, %arg609
  %i2200 = and i1 %arg580, %arg610
  %i2201 = and i1 %i1501, %i2088
  %i2202 = and i1 %i1501, %i2089
  %i2203 = and i1 %i1501, %i2090
  %i2204 = and i1 %arg580, %arg608
  %i2205 = and i1 %arg580, %arg580
  %i2206 = and i1 %arg580, %arg580
  %i2207 = and i1 %i1501, %i2094
  %i2208 = and i1 %arg580, %i2095
  %i2209 = and i1 %arg1028, %arg580
  %i2210 = and i1 %arg580, %arg580
  %i2211 = and i1 %arg580, %arg580
  %i2212 = and i1 %arg580, %arg580
  %i2213 = and i1 %arg580, %arg580
  %i2214 = and i1 %arg580, %arg580
  %i2215 = and i1 %i1501, %i2102
  %i2216 = and i1 %i1501, %i2103
  %i2217 = and i1 %i1501, %i2104
  %i2218 = and i1 %i1501, %i2105
  %i2219 = and i1 %i1501, %i2106
  %i2220 = and i1 %arg580, %arg587
  %i2221 = and i1 %arg580, %arg580
  %i2222 = and i1 %arg580, %arg580
  %i2223 = and i1 %i1501, %i2110
  %i2224 = and i1 %i1501, %i2111
  %i2225 = and i1 %i1501, %i2112
  %i2226 = and i1 %i1501, %i2113
  %i2227 = and i1 %i1501, %i2114
  %i2228 = and i1 %arg580, %i2115
  %i2229 = and i1 %i1501, %i2116
  %i2230 = and i1 %arg580, %arg620
  %i2231 = and i1 %i1501, %i2118
  %i2232 = and i1 false, false
  %i2233 = and i1 %i1501, %i2127
  %i2234 = and i1 %arg580, %arg609
  %i2235 = and i1 %arg580, %arg580
  %i2236 = and i1 %arg580, %arg580
  %i2237 = and i1 %i1501, %i2131
  %i2238 = and i1 %i1501, %i2132
  %i2239 = and i1 %i1501, %i2133
  %i2240 = and i1 %i1501, %i2134
  %i2241 = and i1 %i1501, %i2135
  %i2242 = and i1 %arg580, %arg616
  %i2243 = and i1 %arg580, %arg617
  %i2244 = and i1 %arg580, %i2138
  %i2245 = and i1 %i1501, %i2139
  %i2246 = and i1 %i1501, %i2140
  %i2247 = and i1 %i1501, %i2141
  %i2248 = and i1 %i1501, %i2142
  %i2249 = and i1 %i1501, %i2143
  %i2250 = and i1 %arg580, %arg597
  %i2251 = and i1 %arg580, %arg580
  %i2252 = and i1 %arg580, %arg580
  %i2253 = and i1 true, %arg580
  %i2254 = and i1 %arg580, %arg580
  %i2255 = and i1 false, false
  %i2256 = and i1 %arg580, %arg580
  %i2257 = and i1 true, %arg580
  %i2258 = and i1 %arg580, %arg580
  %i2259 = and i1 true, %arg580
  %i2260 = and i1 %arg580, %arg580
  %i2261 = and i1 %arg580, %arg580
  %i2262 = and i1 %arg580, %arg580
  %i2263 = and i1 %arg580, %arg580
  %i2264 = and i1 %arg580, %arg630
  %i2265 = and i1 %i1501, %i2159
  %i2266 = and i1 %arg580, %arg591
  %i2267 = and i1 %arg580, %arg580
  %i2268 = and i1 %arg580, %arg580
  %i2269 = and i1 %i1501, %i2163
  %i2270 = and i1 %i1501, %i2164
  %i2271 = and i1 %i1501, %i2165
  %i2272 = and i1 %i1501, %i2166
  %i2273 = and i1 %i1501, %i2167
  %i2274 = and i1 %arg580, %arg604
  %i2275 = and i1 %arg580, %arg580
  %i2276 = and i1 %arg580, %arg580
  %i2277 = and i1 %i1498, %i2171
  %i2278 = and i1 %i1498, %i2172
  %i2279 = and i1 %i1498, %arg634
  %i2280 = and i1 %arg635, %i2174
  %i2281 = and i1 %i1498, %i2175
  %i2282 = and i1 %i1498, %i2176
  %i2283 = and i1 %arg595, %arg580
  %i2284 = and i1 %arg580, %arg580
  %i2285 = and i1 %i1498, %i2179
  %i2286 = and i1 %i1498, %i2180
  %i2287 = and i1 %i1498, %i2181
  %i2288 = and i1 %i1498, %i2182
  %i2289 = and i1 %i1498, %i2183
  %i2290 = and i1 %arg601, %arg580
  %i2291 = and i1 %arg580, %arg580
  %i2292 = and i1 %arg580, %arg580
  %i2293 = and i1 %i1498, %i2187
  %i2294 = and i1 %i1498, %i2188
  %i2295 = and i1 %arg611, %arg580
  %i2296 = and i1 %arg580, %arg580
  %i2297 = and i1 %arg580, %arg580
  %i2298 = and i1 %i1498, %i2192
  %i2299 = and i1 %i1498, %i2193
  %i2300 = and i1 %i1498, %i2194
  %i2301 = and i1 %i1498, %i2195
  %i2302 = and i1 %arg613, %arg580
  %i2303 = and i1 %arg580, %arg580
  %i2304 = and i1 %arg580, %arg580
  %i2305 = and i1 %i1498, %i2199
  %i2306 = and i1 %i1498, %i2200
  %i2307 = and i1 %i1498, %i2201
  %i2308 = and i1 %i1498, %i2202
  %i2309 = and i1 %i1498, %i2203
  %i2310 = and i1 %arg627, %arg580
  %i2311 = and i1 %arg580, %arg580
  %i2312 = and i1 %arg580, %arg580
  %i2313 = and i1 %i1498, %i2207
  %i2314 = and i1 %arg584, %i2208
  %i2315 = and i1 %i1498, %i2209
  %i2316 = and i1 %arg589, %arg580
  %i2317 = and i1 %arg580, %arg580
  %i2318 = and i1 %arg580, %arg580
  %i2319 = and i1 %arg580, %arg580
  %i2320 = and i1 %arg580, %arg580
  %i2321 = and i1 %i1498, %i2215
  %i2322 = and i1 %i1498, %i2216
  %i2323 = and i1 %i1498, %i2217
  %i2324 = and i1 %i1498, %i2218
  %i2325 = and i1 %i1498, %i2219
  %i2326 = and i1 %arg625, %arg580
  %i2327 = and i1 %arg580, %arg580
  %i2328 = and i1 %arg580, %arg580
  %i2329 = and i1 %i1498, %i2223
  %i2330 = and i1 %i1498, %i2224
  %i2331 = and i1 %i1498, %i2225
  %i2332 = and i1 %i1498, %i2226
  %i2333 = and i1 %i1498, %i2227
  %i2334 = and i1 %arg635, %i2228
  %i2335 = and i1 %arg584, %i2229
  %i2336 = and i1 %arg616, %arg580
  %i2337 = and i1 %i1498, %i2231
  %i2338 = and i1 false, false
  %i2339 = and i1 %i1498, %i2233
  %i2340 = and i1 %arg630, %arg580
  %i2341 = and i1 %arg580, %arg580
  %i2342 = and i1 %arg580, %arg580
  %i2343 = and i1 %i1498, %i2237
  %i2344 = and i1 %i1498, %i2238
  %i2345 = and i1 %i1498, %i2239
  %i2346 = and i1 %i1498, %i2240
  %i2347 = and i1 %i1498, %i2241
  %i2348 = and i1 %arg595, %arg580
  %i2349 = and i1 %arg580, %arg580
  %i2350 = and i1 %arg635, %i2244
  %i2351 = and i1 %i1498, %i2245
  %i2352 = and i1 %i1498, %i2246
  %i2353 = and i1 %i1498, %i2247
  %i2354 = and i1 %i1498, %i2248
  %i2355 = and i1 %i1498, %i2249
  %i2356 = and i1 %arg621, %arg580
  %i2357 = and i1 true, %arg580
  %i2358 = and i1 %arg580, %arg580
  %i2359 = and i1 %arg580, %arg580
  %i2360 = and i1 %arg580, %arg580
  %i2361 = and i1 false, false
  %i2362 = and i1 %arg580, %arg580
  %i2363 = and i1 %arg580, %arg580
  %i2364 = and i1 %arg580, %arg580
  %i2365 = and i1 %arg580, %arg580
  %i2366 = and i1 %arg580, %arg580
  %i2367 = and i1 %arg580, %arg580
  %i2368 = and i1 %arg580, %arg580
  %i2369 = and i1 %arg580, %arg580
  %i2370 = and i1 %i1498, %i2264
  %i2371 = and i1 %i1498, %i2265
  %i2372 = and i1 %arg602, %arg580
  %i2373 = and i1 %arg580, %arg580
  %i2374 = and i1 %arg580, %arg580
  %i2375 = and i1 %i1498, %i2269
  %i2376 = and i1 %i1498, %i2270
  %i2377 = and i1 %i1498, %i2271
  %i2378 = and i1 %i1498, %i2272
  %i2379 = and i1 %i1498, %i2273
  %i2380 = and i1 %arg613, %arg580
  %i2381 = and i1 %arg580, %arg580
  %i2382 = and i1 %arg580, %arg580
  %i2383 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %arg1, i16 0, i64 2147483646, i32 159744)
  %i2384 = shl i32 %i1948, 1
  %i2385 = select i1 %i2277, i32 %i2384, i32 0
  %i2386 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2385, i32 0, i32 0)
  %i2387 = shl i32 %i1949, 1
  %i2388 = select i1 %i2278, i32 %i2387, i32 0
  %i2389 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2388, i32 0, i32 0)
  %i2390 = shl i32 %i1950, 1
  %i2391 = select i1 %i2279, i32 %i2390, i32 0
  %i2392 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2391, i32 0, i32 0)
  %i2393 = shl i32 %i1951, 1
  %i2394 = select i1 %i2280, i32 %i2393, i32 0
  %i2395 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2394, i32 0, i32 0)
  %i2396 = shl i32 %i1952, 1
  %i2397 = select i1 %i2281, i32 %i2396, i32 0
  %i2398 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %i2397, i32 0, i32 0)
  %i2399 = shl i32 %i1953, 0
  %i2400 = select i1 %i2282, i32 %i2399, i32 0
  %i2401 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %i2400, i32 0, i32 0)
  %i2402 = shl i32 %i1954, 1
  %i2403 = select i1 %arg615, i32 %i2402, i32 0
  %i2404 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %i2403, i32 0, i32 0)
  %i2405 = shl i32 %i1955, 1
  %i2406 = select i1 %arg599, i32 %i2405, i32 0
  %i2407 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %i2406, i32 0, i32 0)
  %i2408 = shl i32 %i1956, 1
  %i2409 = select i1 %i2285, i32 %i2408, i32 0
  %i2410 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %i2409, i32 0, i32 0)
  %i2411 = shl i32 %i1957, 0
  %i2412 = select i1 %i2286, i32 %i2411, i32 0
  %i2413 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %i2412, i32 0, i32 0)
  %i2414 = shl i32 0, 0
  %i2415 = select i1 %i2287, i32 0, i32 1
  %i2416 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %i2415, i32 0, i32 0)
  %i2417 = shl i32 %arg44, 1
  %i2418 = select i1 %i2288, i32 %arg126, i32 0
  %i2419 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %i2418, i32 0, i32 0)
  %i2420 = shl i32 %arg5, 0
  %i2421 = select i1 %i2289, i32 %arg98, i32 0
  %i2422 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %i2421, i32 0, i32 0)
  %i2423 = shl i32 %arg38, 0
  %i2424 = select i1 %arg580, i32 %arg4, i32 0
  %i2425 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i2426 = shl i32 %arg4, 0
  %i2427 = select i1 %arg580, i32 %arg4, i32 0
  %i2428 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i2429 = select i1 %arg580, i32 0, i32 1
  %i2430 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i2431 = shl i32 %arg4, 0
  %i2432 = select i1 %i2293, i32 %arg105, i32 0
  %i2433 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2432, i32 0, i32 0)
  %i2434 = shl i32 %arg42, 0
  %i2435 = select i1 %i2294, i32 %arg74, i32 0
  %i2436 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2435, i32 0, i32 0)
  %i2437 = shl i32 %arg62, 0
  %i2438 = select i1 %arg580, i32 %arg4, i32 0
  %i2439 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i2440 = shl i32 %arg4, 0
  %i2441 = select i1 %arg580, i32 %arg4, i32 0
  %i2442 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i2443 = shl i32 %arg4, 0
  %i2444 = select i1 %arg580, i32 %arg4, i32 0
  %i2445 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i2446 = shl i32 %i1972, 0
  %i2447 = select i1 %i2298, i32 %i2446, i32 0
  %i2448 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2447, i32 0, i32 0)
  %i2449 = shl i32 %i1973, 0
  %i2450 = select i1 %i2299, i32 %i2449, i32 0
  %i2451 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2450, i32 0, i32 0)
  %i2452 = shl i32 %i1974, 0
  %i2453 = select i1 %i2300, i32 %i2452, i32 0
  %i2454 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2453, i32 0, i32 0)
  %i2455 = shl i32 %i1975, 0
  %i2456 = select i1 %i2301, i32 %i2455, i32 0
  %i2457 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2456, i32 0, i32 0)
  %i2458 = shl i32 %.frozen, 0
  %i2459 = select i1 %arg580, i32 %arg4, i32 0
  %i2460 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i2461 = shl i32 %arg4, 0
  %i2462 = select i1 %arg580, i32 %arg4, i32 0
  %i2463 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i2464 = shl i32 %arg4, 0
  %i2465 = select i1 %arg580, i32 %arg4, i32 0
  %i2466 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i2467 = shl i32 %i1979, 0
  %i2468 = select i1 %i2305, i32 %i2467, i32 0
  %i2469 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2468, i32 0, i32 0)
  %i2470 = shl i32 %i1980, 0
  %i2471 = select i1 %i2306, i32 %i2470, i32 0
  %i2472 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2471, i32 0, i32 0)
  %i2473 = shl i32 %i1981, 0
  %i2474 = select i1 %i2307, i32 %i2473, i32 0
  %i2475 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2474, i32 0, i32 0)
  %i2476 = shl i32 %i1982, 0
  %i2477 = select i1 %i2308, i32 %i2476, i32 0
  %i2478 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2477, i32 0, i32 0)
  %i2479 = shl i32 %arg46, 0
  %i2480 = select i1 %i2309, i32 %arg726, i32 0
  %i2481 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2480, i32 0, i32 0)
  %i2482 = shl i32 %arg62, 0
  %i2483 = select i1 %arg580, i32 %arg4, i32 0
  %i2484 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i2485 = shl i32 %arg4, 0
  %i2486 = select i1 %arg580, i32 %arg4, i32 0
  %i2487 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i2488 = shl i32 %arg4, 0
  %i2489 = select i1 %arg580, i32 %arg4, i32 0
  %i2490 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i2491 = shl i32 %arg4, 0
  %i2492 = select i1 %i2313, i32 %arg700, i32 0
  %i2493 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2492, i32 0, i32 0)
  %i2494 = shl i32 %arg45, 0
  %i2495 = select i1 %i2314, i32 %arg17, i32 0
  %i2496 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i2497 = shl i32 %arg4, 0
  %i2498 = select i1 %i2315, i32 %arg707, i32 0
  %i2499 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2498, i32 0, i32 0)
  %i2500 = shl i32 %arg50, 0
  %i2501 = select i1 %arg580, i32 %arg4, i32 0
  %i2502 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %arg737, i32 0, i32 0)
  %i2503 = shl i32 %arg15, 0
  %i2504 = select i1 %arg580, i32 %arg4, i32 0
  %i2505 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %arg713, i32 0, i32 0)
  %i2506 = shl i32 %arg4, 0
  %i2507 = select i1 %arg580, i32 %arg4, i32 0
  %i2508 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i2509 = shl i32 %arg4, 0
  %i2510 = select i1 %arg580, i32 %arg4, i32 0
  %i2511 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i2512 = shl i32 %arg4, 0
  %i2513 = select i1 %arg580, i32 %arg4, i32 0
  %i2514 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i2515 = shl i32 %arg4, 0
  %i2516 = select i1 %i2321, i32 %arg755, i32 0
  %i2517 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2516, i32 0, i32 0)
  %i2518 = shl i32 %i1996, 0
  %i2519 = select i1 %i2322, i32 %i2518, i32 0
  %i2520 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2519, i32 0, i32 0)
  %i2521 = shl i32 %i1997, 0
  %i2522 = select i1 %i2323, i32 %i2521, i32 0
  %i2523 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2522, i32 0, i32 0)
  %i2524 = shl i32 %arg13, 0
  %i2525 = select i1 %i2324, i32 %arg53, i32 0
  %i2526 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2525, i32 0, i32 0)
  %i2527 = shl i32 %arg39, 0
  %i2528 = select i1 %i2325, i32 %arg790, i32 0
  %i2529 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2528, i32 0, i32 0)
  %i2530 = shl i32 %arg56, 0
  %i2531 = select i1 %arg580, i32 %arg4, i32 0
  %i2532 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i2533 = shl i32 %arg4, 0
  %i2534 = select i1 %arg580, i32 %arg4, i32 0
  %i2535 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i2536 = shl i32 %arg4, 0
  %i2537 = select i1 %arg580, i32 %arg4, i32 0
  %i2538 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i2539 = shl i32 %arg4, 0
  %i2540 = select i1 %i2329, i32 %arg769, i32 0
  %i2541 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2540, i32 0, i32 0)
  %i2542 = select i1 %i2330, i32 0, i32 1
  %i2543 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2542, i32 0, i32 0)
  %i2544 = select i1 %i2331, i32 0, i32 1
  %i2545 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2544, i32 0, i32 0)
  %i2546 = select i1 %i2332, i32 0, i32 1
  %i2547 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2546, i32 0, i32 0)
  %i2548 = select i1 %i2333, i32 0, i32 1
  %i2549 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2548, i32 0, i32 0)
  %i2550 = select i1 %i2334, i32 0, i32 1
  %i2551 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %arg60, i32 0, i32 0)
  %i2552 = select i1 %arg580, i32 0, i32 0
  %i2553 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %i2554 = select i1 %arg580, i32 0, i32 1
  %i2555 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i2556 = select i1 %i2337, i32 0, i32 1
  %i2557 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2556, i32 0, i32 0)
  %i2558 = select i1 false, i32 0, i32 0
  %i2559 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %i2560 = select i1 %i2339, i32 0, i32 1
  %i2561 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2560, i32 0, i32 0)
  %i2562 = shl i32 %i2011, 1
  %i2563 = select i1 %arg581, i32 %i2562, i32 0
  %i2564 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %i2563, i32 0, i32 0)
  %i2565 = shl i32 %i2012, 1
  %i2566 = select i1 %arg591, i32 %i2565, i32 0
  %i2567 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %i2566, i32 0, i32 0)
  %i2568 = shl i32 %i2013, 1
  %i2569 = select i1 %arg599, i32 %i2568, i32 0
  %i2570 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %i2569, i32 0, i32 0)
  %i2571 = shl i32 %i2014, 1
  %i2572 = select i1 %i2343, i32 %i2571, i32 0
  %i2573 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2572, i32 0, i32 0)
  %i2574 = shl i32 %i2015, 1
  %i2575 = select i1 %i2344, i32 %i2574, i32 0
  %i2576 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2575, i32 0, i32 0)
  %i2577 = shl i32 %i2016, 1
  %i2578 = select i1 %i2345, i32 %i2577, i32 0
  %i2579 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2578, i32 0, i32 0)
  %i2580 = shl i32 %i2017, 1
  %i2581 = select i1 %i2346, i32 %i2580, i32 0
  %i2582 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2581, i32 0, i32 0)
  %i2583 = shl i32 %i2018, 1
  %i2584 = select i1 %i2347, i32 %i2583, i32 0
  %i2585 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2584, i32 0, i32 0)
  %i2586 = shl i32 %i2019, 1
  %i2587 = select i1 %arg591, i32 %i2586, i32 0
  %i2588 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %i2587, i32 0, i32 0)
  %i2589 = shl i32 %i2020, 1
  %i2590 = select i1 %arg597, i32 %i2589, i32 0
  %i2591 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %i2590, i32 0, i32 0)
  %i2592 = shl i32 %i2021, 1
  %i2593 = select i1 %i2350, i32 %i2592, i32 0
  %i2594 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %i2593, i32 0, i32 0)
  %i2595 = shl i32 %i2022, 1
  %i2596 = select i1 %i2351, i32 %i2595, i32 0
  %i2597 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2596, i32 0, i32 0)
  %i2598 = shl i32 %i2023, 1
  %i2599 = select i1 %i2352, i32 %i2598, i32 0
  %i2600 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2599, i32 0, i32 0)
  %i2601 = shl i32 %i2024, 1
  %i2602 = select i1 %i2353, i32 %i2601, i32 0
  %i2603 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2602, i32 0, i32 0)
  %i2604 = shl i32 %i2025, 1
  %i2605 = select i1 %i2354, i32 %i2604, i32 0
  %i2606 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %i2605, i32 0, i32 0)
  %i2607 = shl i32 %i2026, 1
  %i2608 = select i1 %i2355, i32 %i2607, i32 0
  %i2609 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %i2608, i32 0, i32 0)
  %i2610 = shl i32 %arg47, 1
  %i2611 = select i1 %arg580, i32 %arg4, i32 0
  %i2612 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i2613 = shl i32 %arg4, 1
  %i2614 = select i1 %arg580, i32 %arg4, i32 0
  %i2615 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i2616 = shl i32 %arg4, 1
  %i2617 = select i1 %arg580, i32 %arg4, i32 0
  %i2618 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i2619 = shl i32 %arg4, 1
  %i2620 = select i1 %arg580, i32 %arg4, i32 0
  %i2621 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %arg794, i32 0, i32 0)
  %i2622 = shl i32 %arg14, 1
  %i2623 = select i1 %arg580, i32 %arg4, i32 0
  %i2624 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %arg789, i32 0, i32 0)
  %i2625 = shl i32 0, 0
  %i2626 = select i1 false, i32 0, i32 0
  %i2627 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %i2628 = shl i32 %arg4, 0
  %i2629 = select i1 %arg580, i32 %arg4, i32 0
  %i2630 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %arg750, i32 0, i32 0)
  %i2631 = shl i32 %arg4, 0
  %i2632 = select i1 %arg580, i32 %arg4, i32 0
  %i2633 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %arg703, i32 0, i32 0)
  %i2634 = shl i32 %arg41, 0
  %i2635 = select i1 %arg580, i32 %arg4, i32 0
  %i2636 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i2637 = shl i32 %arg4, 1
  %i2638 = select i1 %arg580, i32 %arg4, i32 0
  %i2639 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i2640 = shl i32 %arg4, 1
  %i2641 = select i1 %arg580, i32 %arg4, i32 0
  %i2642 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i2643 = shl i32 %arg4, 1
  %i2644 = select i1 %arg580, i32 %arg4, i32 0
  %i2645 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %arg822, i32 0, i32 0)
  %i2646 = shl i32 %arg40, 1
  %i2647 = select i1 %arg580, i32 %arg4, i32 0
  %i2648 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %arg826, i32 0, i32 0)
  %i2649 = shl i32 %arg52, 1
  %i2650 = select i1 %arg580, i32 %arg4, i32 0
  %i2651 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %arg830, i32 0, i32 0)
  %i2652 = shl i32 %arg55, 1
  %i2653 = select i1 %i2370, i32 %arg53, i32 0
  %i2654 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %i2653, i32 0, i32 0)
  %i2655 = shl i32 %i2042, 0
  %i2656 = select i1 %i2371, i32 %i2655, i32 0
  %i2657 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %i2656, i32 0, i32 0)
  %i2658 = shl i32 %i2043, 1
  %i2659 = select i1 %arg626, i32 %i2658, i32 0
  %i2660 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %i2659, i32 0, i32 0)
  %i2661 = shl i32 %i2044, 1
  %i2662 = select i1 %arg599, i32 %i2661, i32 0
  %i2663 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %i2662, i32 0, i32 0)
  %i2664 = shl i32 %i2045, 1
  %i2665 = select i1 %arg623, i32 %i2664, i32 0
  %i2666 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %i2665, i32 0, i32 0)
  %i2667 = shl i32 %i2046, 1
  %i2668 = select i1 %i2375, i32 %i2667, i32 0
  %i2669 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %i2668, i32 0, i32 0)
  %i2670 = shl i32 %i2047, 1
  %i2671 = select i1 %i2376, i32 %i2670, i32 0
  %i2672 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %i2671, i32 0, i32 0)
  %i2673 = shl i32 %i2048, 1
  %i2674 = select i1 %i2377, i32 %i2673, i32 0
  %i2675 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %i2674, i32 0, i32 0)
  %i2676 = shl i32 %i2049, 0
  %i2677 = select i1 %i2378, i32 %i2676, i32 0
  %i2678 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %i2677, i32 0, i32 0)
  %i2679 = shl i32 %arg56, 0
  %i2680 = select i1 %i2379, i32 %arg36, i32 0
  %i2681 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %i2680, i32 0, i32 0)
  %i2682 = shl i32 %arg14, 0
  %i2683 = select i1 %arg580, i32 0, i32 0
  %i2684 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 0, i32 0, i32 0)
  %i2685 = shl i32 %arg4, 0
  %i2686 = select i1 %arg580, i32 0, i32 0
  %i2687 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 0, i32 0, i32 0)
  %i2688 = shl i32 %arg4, 0
  %i2689 = select i1 %arg580, i32 0, i32 0
  %i2690 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 0, i32 0, i32 0)
  %i2691 = shl nuw nsw i32 %i1483, 1
  %i2692 = icmp eq i32 %i1359, 0
  %i2693 = select i1 %i2692, i32 0, i32 528
  %i2694 = xor i32 %i2691, %i2693
  %i2695 = xor i32 %arg869, 1056
  %i2696 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i32 262144), i32 %i2695
  store i16 %i1503, ptr addrspace(3) %i2696, align 2
  %i2697 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %i2694
  store i16 %i2386, ptr addrspace(3) %i2697, align 2
  store i16 %i2410, ptr addrspace(3) null, align 2
  %i2698 = or disjoint i32 0, 1
  %i2699 = getelementptr inbounds nuw i8, ptr addrspace(3) %global_smem, i32 %i2698
  store i16 %i2433, ptr addrspace(3) %i2699, align 2
  store i16 %i2448, ptr addrspace(3) null, align 2
  store i16 %i2469, ptr addrspace(3) %global_smem, align 2
  %i2700 = or disjoint i32 1, 0
  store i16 %i2493, ptr addrspace(3) null, align 2
  store i16 %i2517, ptr addrspace(3) %global_smem, align 2
  store i16 %i2541, ptr addrspace(3) null, align 2
  %i2701 = or disjoint i32 %i2694, 65536
  %i2702 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %i2701
  store i16 %i2557, ptr addrspace(3) %i2702, align 2
  %i2703 = or disjoint i32 %arg718, 73728
  %i2704 = or disjoint i32 0, 81920
  store i16 %i2573, ptr addrspace(3) null, align 2
  %i2705 = or disjoint i32 %i2694, 98304
  %i2706 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %i2705
  store i16 %i2597, ptr addrspace(3) %i2706, align 2
  %i2707 = or disjoint i32 %arg53, 106496
  %i2708 = getelementptr inbounds nuw i8, ptr addrspace(3) %global_smem, i32 %i2707
  store i16 %i2621, ptr addrspace(3) null, align 2
  %i2709 = or disjoint i32 0, 114688
  %i2710 = getelementptr inbounds nuw i8, ptr addrspace(3) %global_smem, i32 %i2709
  store i16 %i2645, ptr addrspace(3) %i2710, align 2
  %i2711 = or disjoint i32 255, 122880
  store i16 %i2669, ptr addrspace(3) null, align 2
  %i2712 = getelementptr inbounds nuw i8, ptr addrspace(3) %global_smem, i32 %arg832
  store i16 %i2389, ptr addrspace(3) %i2712, align 2
  %i2713 = or disjoint i32 0, 0
  %i2714 = getelementptr inbounds nuw i8, ptr addrspace(3) %global_smem, i32 %i2713
  store i16 %i2413, ptr addrspace(3) %i2714, align 2
  %i2715 = or disjoint i32 1, 0
  store i16 %i2451, ptr addrspace(3) null, align 2
  store i16 %i2472, ptr addrspace(3) %global_smem, align 2
  store i16 %arg833, ptr addrspace(3) null, align 2
  store i16 %i2520, ptr addrspace(3) %global_smem, align 2
  store i16 %i2543, ptr addrspace(3) null, align 2
  %i2716 = or disjoint i32 %i2695, 65536
  %i2717 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %i2716
  store i16 0, ptr addrspace(3) %i2717, align 2
  %i2718 = or disjoint i32 0, 0
  %i2719 = or disjoint i32 %arg127, 81920
  store i16 %i2576, ptr addrspace(3) null, align 2
  %i2720 = or disjoint i32 %i2695, 98304
  %i2721 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %i2720
  store i16 %i2600, ptr addrspace(3) %i2721, align 2
  %i2722 = or disjoint i32 1, 1
  %i2723 = getelementptr inbounds nuw i8, ptr addrspace(3) %global_smem, i32 %i2722
  store i16 %i2624, ptr addrspace(3) %i2723, align 2
  %i2724 = or disjoint i32 %i2695, 114688
  %i2725 = getelementptr inbounds nuw i8, ptr addrspace(3) %global_smem, i32 %i2724
  store i16 %i2648, ptr addrspace(3) null, align 2
  %i2726 = or disjoint i32 %i2695, 122880
  %i2727 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %i2726
  store i16 %i2672, ptr addrspace(3) %i2727, align 2
  %i2728 = xor i32 %arg790, 2112
  store i16 %i2392, ptr addrspace(3) null, align 2
  %i2729 = or disjoint i32 %i2728, 0
  %i2730 = getelementptr inbounds nuw i8, ptr addrspace(3) %global_smem, i32 %i2729
  store i16 %i2416, ptr addrspace(3) %i2730, align 2
  store i16 %i2454, ptr addrspace(3) null, align 2
  %i2731 = or disjoint i32 1, 0
  %i2732 = getelementptr inbounds nuw i8, ptr addrspace(3) %global_smem, i32 %i2731
  store i16 %i2475, ptr addrspace(3) %i2732, align 2
  store i16 %i2499, ptr addrspace(3) null, align 2
  store i16 %i2523, ptr addrspace(3) %global_smem, align 2
  store i16 %i2545, ptr addrspace(3) null, align 2
  %i2733 = or disjoint i32 0, 81920
  %i2734 = getelementptr inbounds nuw i8, ptr addrspace(3) %global_smem, i32 %i2733
  store i16 0, ptr addrspace(3) %i2734, align 2
  %i2735 = or disjoint i32 0, 0
  %i2736 = getelementptr inbounds nuw i8, ptr addrspace(3) %global_smem, i32 %i2735
  store i16 %i2579, ptr addrspace(3) null, align 2
  %i2737 = or disjoint i32 0, 1
  %i2738 = getelementptr inbounds nuw i8, ptr addrspace(3) %global_smem, i32 %i2737
  store i16 %i2603, ptr addrspace(3) %i2738, align 2
  %i2739 = or disjoint i32 0, 0
  %i2740 = getelementptr inbounds nuw i8, ptr addrspace(3) %global_smem, i32 %i2739
  store i16 0, ptr addrspace(3) null, align 2
  %i2741 = or disjoint i32 0, 0
  %i2742 = getelementptr inbounds nuw i8, ptr addrspace(3) %global_smem, i32 %i2741
  store i16 %i2651, ptr addrspace(3) null, align 2
  %i2743 = or disjoint i32 0, 1
  %i2744 = getelementptr inbounds nuw i8, ptr addrspace(3) %global_smem, i32 %i2743
  store i16 %i2675, ptr addrspace(3) %i2744, align 2
  %i2745 = xor i32 %arg789, 3168
  store i16 %i2395, ptr addrspace(3) null, align 2
  %i2746 = or disjoint i32 0, 0
  %i2747 = getelementptr inbounds nuw i8, ptr addrspace(3) %global_smem, i32 %i2746
  store i16 %i2419, ptr addrspace(3) %i2747, align 2
  store i16 %i2478, ptr addrspace(3) null, align 2
  store i16 %i2502, ptr addrspace(3) %global_smem, align 2
  store i16 %i2526, ptr addrspace(3) null, align 2
  store i16 %i2547, ptr addrspace(3) %global_smem, align 2
  %i2748 = or disjoint i32 %arg823, 73728
  %i2749 = or disjoint i32 %arg19, 90112
  %i2750 = getelementptr inbounds nuw i8, ptr addrspace(3) %global_smem, i32 %i2749
  store i16 %i2582, ptr addrspace(3) null, align 2
  %i2751 = or disjoint i32 %i2745, 98304
  %i2752 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %i2751
  store i16 %i2606, ptr addrspace(3) %i2752, align 2
  %i2753 = or disjoint i32 %arg123, 106496
  %i2754 = getelementptr inbounds nuw i8, ptr addrspace(3) %global_smem, i32 %i2753
  store i16 %i2630, ptr addrspace(3) null, align 2
  %i2755 = or disjoint i32 %i2745, 114688
  %i2756 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %i2755
  store i16 %i2654, ptr addrspace(3) %i2756, align 2
  %i2757 = or disjoint i32 0, 0
  store i16 %i2678, ptr addrspace(3) null, align 2
  %i2758 = xor i32 %arg1146, 0
  %i2759 = getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem, i32 %i2758
  store i16 %i2398, ptr addrspace(3) %i2759, align 2
  store i16 %i2422, ptr addrspace(3) null, align 2
  %i2760 = or disjoint i32 1, 0
  %i2761 = getelementptr inbounds nuw i8, ptr addrspace(3) %global_smem, i32 %i2760
  store i16 %i2436, ptr addrspace(3) %i2761, align 2
  store i16 %i2457, ptr addrspace(3) null, align 2
  %i2762 = or disjoint i32 1, 0
  %i2763 = getelementptr inbounds nuw i8, ptr addrspace(3) %global_smem, i32 %i2762
  store i16 %i2481, ptr addrspace(3) %i2763, align 2
  store i16 %i2505, ptr addrspace(3) null, align 2
  store i16 %i2529, ptr addrspace(3) %global_smem, align 2
  store i16 %i2549, ptr addrspace(3) null, align 2
  %i2764 = or disjoint i32 0, 0
  %i2765 = getelementptr inbounds nuw i8, ptr addrspace(3) %global_smem, i32 %i2764
  store i16 0, ptr addrspace(3) %i2765, align 2
  %i2766 = or disjoint i32 1, 0
  store i16 %i2561, ptr addrspace(3) null, align 2
  %i2767 = or disjoint i32 %arg89, 90112
  %i2768 = getelementptr inbounds nuw i8, ptr addrspace(3) %global_smem, i32 %arg834
  store i16 %i2585, ptr addrspace(3) %i2768, align 2
  %i2769 = or disjoint i32 0, 0
  %i2770 = getelementptr inbounds nuw i8, ptr addrspace(3) %global_smem, i32 %i2769
  store i16 %i2609, ptr addrspace(3) %i2770, align 2
  store i16 %i2633, ptr addrspace(3) null, align 2
  store i16 %i2657, ptr addrspace(3) %global_smem, align 2
  store i16 %i2681, ptr addrspace(3) null, align 2
  %i2771 = xor i32 %arg98, 5280
  %i2772 = getelementptr inbounds nuw i8, ptr addrspace(3) %global_smem, i32 %i2771
  store i16 %i2401, ptr addrspace(3) %arg835, align 2
  store i16 %arg836, ptr addrspace(3) %global_smem, align 2
  %i2773 = or disjoint i32 1, 0
  store i16 %arg837, ptr addrspace(3) null, align 2
  store i16 %arg838, ptr addrspace(3) %global_smem, align 2
  store i16 %arg839, ptr addrspace(3) null, align 2
  store i16 %arg840, ptr addrspace(3) %global_smem, align 2
  store i16 %arg841, ptr addrspace(3) null, align 2
  store i16 %arg842, ptr addrspace(3) %global_smem, align 2
  %i2774 = or disjoint i32 1, 0
  %i2775 = or disjoint i32 %arg93, 73728
  store i16 %arg844, ptr addrspace(3) null, align 2
  %i2776 = or disjoint i32 %arg843, 90112
  %i2777 = getelementptr inbounds nuw i8, ptr addrspace(3) %global_smem, i32 %arg845
  store i16 %arg846, ptr addrspace(3) %arg847, align 2
  %i2778 = or disjoint i32 %arg750, 98304
  store i16 %arg848, ptr addrspace(3) null, align 2
  %i2779 = or disjoint i32 %arg146, 106496
  %i2780 = getelementptr inbounds nuw i8, ptr addrspace(3) %global_smem, i32 %arg849
  store i16 %arg850, ptr addrspace(3) %arg851, align 2
  %i2781 = or disjoint i32 0, 0
  store i16 %arg852, ptr addrspace(3) null, align 2
  %i2782 = or disjoint i32 %arg843, 122880
  %i2783 = getelementptr inbounds nuw i8, ptr addrspace(3) %global_smem, i32 %arg853
  store i16 %arg854, ptr addrspace(3) %arg855, align 2
  %i2784 = xor i32 %arg856, 1
  store i16 %arg857, ptr addrspace(3) null, align 2
  store i16 %arg858, ptr addrspace(3) %global_smem, align 2
  store i16 %arg859, ptr addrspace(3) null, align 2
  %i2785 = or disjoint i32 %arg860, 0
  %i2786 = getelementptr inbounds nuw i8, ptr addrspace(3) %global_smem, i32 %arg861
  store i16 %arg862, ptr addrspace(3) %arg863, align 2
  store i16 %arg864, ptr addrspace(3) %global_smem, align 2
  store i16 %arg865, ptr addrspace(3) null, align 2
  store i16 %arg866, ptr addrspace(3) %global_smem, align 2
  store i16 0, ptr addrspace(3) null, align 2
  %i2787 = or disjoint i32 %arg860, 73728
  %i2788 = getelementptr inbounds nuw i8, ptr addrspace(3) %global_smem, i32 %arg867
  store i16 0, ptr addrspace(3) null, align 2
  %i2789 = or disjoint i32 %arg715, 81920
  store i16 %arg868, ptr addrspace(3) null, align 2
  %i2790 = or disjoint i32 %arg103, 90112
  %i2791 = getelementptr inbounds nuw i8, ptr addrspace(3) %global_smem, i32 %arg869
  store i16 %arg870, ptr addrspace(3) %arg871, align 2
  %i2792 = or disjoint i32 %arg19, 98304
  store i16 %arg872, ptr addrspace(3) null, align 2
  %i2793 = or disjoint i32 0, 0
  %i2794 = getelementptr inbounds nuw i8, ptr addrspace(3) %global_smem, i32 %arg873
  store i16 %arg874, ptr addrspace(3) %arg875, align 2
  %i2795 = or disjoint i32 %arg22, 114688
  store i16 %arg876, ptr addrspace(3) null, align 2
  %i2796 = or disjoint i32 %arg113, 122880
  %i2797 = getelementptr inbounds nuw i8, ptr addrspace(3) %global_smem, i32 %arg877
  store i16 %arg878, ptr addrspace(3) %arg879, align 2
  %i2798 = xor i32 1, 0
  store i16 %arg880, ptr addrspace(3) null, align 2
  store i16 %arg881, ptr addrspace(3) %global_smem, align 2
  store i16 %arg882, ptr addrspace(3) null, align 2
  %i2799 = or disjoint i32 1, 0
  %i2800 = getelementptr inbounds nuw i8, ptr addrspace(3) %global_smem, i32 %arg883
  store i16 %arg884, ptr addrspace(3) %arg885, align 2
  store i16 %arg886, ptr addrspace(3) null, align 2
  store i16 %arg887, ptr addrspace(3) %global_smem, align 2
  store i16 %arg888, ptr addrspace(3) null, align 2
  store i16 %arg889, ptr addrspace(3) %global_smem, align 2
  %i2801 = or disjoint i32 1, 0
  %i2802 = getelementptr inbounds nuw i8, ptr addrspace(3) %global_smem, i32 %arg890
  store i16 %arg891, ptr addrspace(3) %arg892, align 2
  %i2803 = or disjoint i32 0, 0
  store i16 %arg893, ptr addrspace(3) null, align 2
  %i2804 = or disjoint i32 0, 0
  %i2805 = getelementptr inbounds nuw i8, ptr addrspace(3) %global_smem, i32 %arg894
  store i16 %arg895, ptr addrspace(3) %arg896, align 2
  %i2806 = or disjoint i32 0, 0
  store i16 %arg897, ptr addrspace(3) null, align 2
  %i2807 = or disjoint i32 0, 0
  %i2808 = getelementptr inbounds nuw i8, ptr addrspace(3) %global_smem, i32 %arg898
  store i16 %arg899, ptr addrspace(3) %arg900, align 2
  store i16 %arg901, ptr addrspace(3) null, align 2
  %i2809 = add nsw i32 %arg15, 0
  %i2810 = icmp sgt i32 %arg4, 1
  br i1 %arg903, label %.lr.ph, label %.._crit_edge_crit_edge

.._crit_edge_crit_edge:                           ; preds = %bb
  %.pre = and i32 %arg10, 1
  br label %._crit_edge

.lr.ph:                                           ; preds = %bb
  %i2811 = and i32 %arg10, 1
  %smax = tail call i32 @llvm.smax.i32(i32 %arg904, i32 0)
  br label %bb2812

bb2812:                                           ; preds = %bb2812, %.lr.ph
  %.pn17784 = phi i32 [ %arg905, %.lr.ph ], [ %arg906, %bb2812 ]
  %.pn19783 = phi i32 [ %arg907, %.lr.ph ], [ %arg908, %bb2812 ]
  %.pn21782 = phi i32 [ %arg909, %.lr.ph ], [ %arg910, %bb2812 ]
  %.pn23781 = phi i32 [ %arg911, %.lr.ph ], [ %arg912, %bb2812 ]
  %.pn25780 = phi i32 [ %arg913, %.lr.ph ], [ %arg914, %bb2812 ]
  %.pn27779 = phi i32 [ %arg915, %.lr.ph ], [ %arg916, %bb2812 ]
  %.pn31777 = phi i32 [ %arg831, %.lr.ph ], [ %arg917, %bb2812 ]
  %.pn33776 = phi i32 [ %arg827, %.lr.ph ], [ %arg918, %bb2812 ]
  %.pn35775 = phi i32 [ %arg823, %.lr.ph ], [ %arg919, %bb2812 ]
  %.pn37774 = phi i32 [ %arg41, %.lr.ph ], [ %arg920, %bb2812 ]
  %.pn39773 = phi i32 [ %arg92, %.lr.ph ], [ %arg921, %bb2812 ]
  %.pn417723 = phi i32 [ %arg45, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn497684 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn517675 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn537666 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn557657 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn577648 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn597639 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn6176210 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn6376111 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn6576012 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn6775913 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn6975814 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn7175715 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn7375616 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn7575517 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn7775418 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn7975319 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn8175220 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn8375121 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn8575022 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn8774923 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn8974824 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn9174725 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn9374626 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn95745 = phi i32 [ %arg4, %.lr.ph ], [ 0, %bb2812 ]
  %.pn97744 = phi i32 [ 0, %.lr.ph ], [ 0, %bb2812 ]
  %.pn9974327 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn10174228 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn10374129 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn10574030 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn219683 = phi i32 [ 0, %.lr.ph ], [ 0, %bb2812 ]
  %.pn221682 = phi i32 [ 0, %.lr.ph ], [ 0, %bb2812 ]
  %.pn22368131 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn225680 = phi i32 [ 0, %.lr.ph ], [ 0, %bb2812 ]
  %.pn227679 = phi i32 [ 0, %.lr.ph ], [ 0, %bb2812 ]
  %.pn229678 = phi i32 [ 0, %.lr.ph ], [ 0, %bb2812 ]
  %.pn23167732 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn23367633 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn235675 = phi i32 [ 0, %.lr.ph ], [ 0, %bb2812 ]
  %.pn23767434 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn23967335 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn24167236 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn24367137 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn24567038 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn24766939 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn24966840 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn251667 = phi i32 [ 0, %.lr.ph ], [ 0, %bb2812 ]
  %.pn25366641 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn255665 = phi i32 [ 1, %.lr.ph ], [ 1, %bb2812 ]
  %.pn25766442 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn25966343 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %.pn26166244 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %i2813 = phi float [ 0.000000e+00, %.lr.ph ], [ %arg1000, %bb2812 ]
  %i2814 = phi float [ 0.000000e+00, %.lr.ph ], [ %arg1000, %bb2812 ]
  %.pn366145 = phi i32 [ %arg4, %.lr.ph ], [ %arg4, %bb2812 ]
  %i2815 = phi i32 [ 0, %.lr.ph ], [ %arg1004, %bb2812 ]
  %i2816 = add i32 %.pn3661, 1
  %i2817 = add i32 %.pn261662, 1
  %i2818 = add i32 %arg61, 1
  %i2819 = add i32 %.pn257664, 1
  %i2820 = add i32 1, 256
  %i2821 = add i32 %arg48, 1
  %i2822 = add i32 0, 0
  %i2823 = add i32 %.pn249668, 1
  %i2824 = add i32 %arg20, 256
  %i2825 = add i32 %arg4, 1
  %i2826 = add i32 %arg4, 1
  %i2827 = add i32 %arg4, 256
  %i2828 = add i32 %arg4, 1
  %i2829 = add i32 %arg4, 0
  %i2830 = add i32 0, 256
  %i2831 = add i32 %arg4, 1
  %i2832 = add i32 %arg4, 256
  %i2833 = add i32 0, 0
  %i2834 = add i32 0, 0
  %i2835 = add i32 0, 0
  %i2836 = add i32 %arg4, 1
  %i2837 = add i32 0, 0
  %i2838 = add i32 0, 0
  %i2839 = add i32 %.pn105740, 1
  %i2840 = add i32 %.pn103741, 1
  %i2841 = add i32 %.pn101742, 1
  %i2842 = add i32 %.pn99743, 256
  %i2843 = add i32 0, 256
  %i2844 = add i32 0, 256
  %i2845 = add i32 %.pn93746, 1
  %i2846 = add i32 %.pn91747, 1
  %i2847 = add i32 %.pn89748, 1
  %i2848 = add i32 %.pn87749, 1
  %i2849 = add i32 %.pn85750, 1
  %i2850 = add i32 %.pn83751, 1
  %i2851 = add i32 %.pn81752, 1
  %i2852 = add i32 %.pn79753, 1
  %i2853 = add i32 %.pn77754, 1
  %i2854 = add i32 %.pn75755, 1
  %i2855 = add i32 %.pn73756, 1
  %i2856 = add i32 %arg14, 1
  %i2857 = add i32 %arg4, 1
  %i2858 = add i32 %arg942, 1
  %i2859 = add i32 %arg122, 1
  %i2860 = add i32 %.decomposed, 1
  %i2861 = add i32 %arg48, 1
  %i2862 = add i32 %.pn59763, 1
  %i2863 = add i32 %arg720, 1
  %i2864 = add i32 %arg4, 1
  %i2865 = add i32 %arg4, 1
  %i2866 = add i32 %arg4, 1
  %i2867 = add i32 %arg4, 1
  %i2868 = add i32 %arg4, 1
  %i2869 = add i32 %.pn39773, 1
  %i2870 = add i32 %.pn37774, 1
  %i2871 = add i32 %.pn35775, 1
  %i2872 = add i32 %.pn33776, 1
  %i2873 = add i32 %.pn31777, 1
  %i2874 = add i32 %.pn27779, 1
  %i2875 = add i32 %.pn25780, 1
  %i2876 = add i32 %.pn23781, 1
  %i2877 = add i32 %.pn21782, 1
  %i2878 = add i32 %.pn19783, 1
  %i2879 = add i32 %.pn17784, 1
  %i2880 = add nuw nsw i32 %i2815, 1
  %i2881 = shl i32 %i2880, 0
  %i2882 = sub i32 %arg6, %i2881
  %i2883 = icmp slt i32 %i1483, %i2882
  %i2884 = and i1 %i1500, %i2883
  %i2885 = shl i32 %i2816, 1
  %i2886 = select i1 %i2884, i32 %i2885, i32 0
  %i2887 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i1502, i32 %i2886, i32 0, i32 0)
  tail call void @llvm.amdgcn.s.waitcnt(i32 0)
  %i2888 = and i1 %i2055, true
  %i2889 = and i1 %i2056, %i2883
  %i2890 = and i1 %i2057, %i2883
  %i2891 = and i1 %i2058, %i2883
  %i2892 = and i1 %i2059, %i2883
  %i2893 = and i1 %i2060, %i2883
  %i2894 = and i1 %i2061, %i2883
  %i2895 = and i1 %i2062, %i2883
  %i2896 = and i1 %arg589, %arg617
  %i2897 = and i1 %arg580, %arg580
  %i2898 = and i1 %arg580, %arg580
  %i2899 = and i1 %arg580, %arg580
  %i2900 = and i1 %arg580, %arg580
  %i2901 = and i1 %arg580, %arg580
  %i2902 = and i1 %arg580, %arg580
  %i2903 = and i1 %arg580, %arg580
  %i2904 = and i1 %arg580, %arg580
  %i2905 = and i1 %arg580, %arg580
  %i2906 = and i1 %arg580, %arg580
  %i2907 = and i1 %arg580, %arg580
  %i2908 = and i1 %arg580, %arg580
  %i2909 = and i1 %arg580, %arg580
  %i2910 = and i1 %arg580, %arg580
  %i2911 = and i1 %arg580, %arg580
  %i2912 = and i1 %arg580, %arg580
  %i2913 = and i1 %arg580, %arg580
  %i2914 = and i1 %i2081, %i2883
  %i2915 = and i1 %i2082, %i2883
  %i2916 = and i1 %i2083, %i2883
  %i2917 = and i1 %i2084, %i2883
  %i2918 = and i1 %i2085, %i2883
  %i2919 = and i1 %arg663, %i2883
  %i2920 = and i1 %i2088, %i2883
  %i2921 = and i1 %i2089, %i2883
  %i2922 = and i1 %i2090, %i2883
  %i2923 = and i1 %i2091, %i2883
  %i2924 = and i1 %i2092, %i2883
  %i2925 = and i1 %i2093, %i2883
  %i2926 = and i1 %i2094, %i2883
  %i2927 = and i1 %arg617, %i2883
  %i2928 = and i1 %i2097, %i2883
  %i2929 = and i1 %arg611, %arg586
  %i2930 = and i1 %i2099, %i2883
  %i2931 = and i1 %i2100, %i2883
  %i2932 = and i1 %i2101, %i2883
  %i2933 = and i1 %i2102, %i2883
  %i2934 = and i1 %i2104, %i2883
  %i2935 = and i1 %i2105, %i2883
  %i2936 = and i1 %i2106, %i2883
  %i2937 = and i1 %i2107, %i2883
  %i2938 = and i1 %i2108, %i2883
  %i2939 = and i1 %i2109, %i2883
  %i2940 = and i1 %i2110, %i2883
  %i2941 = and i1 %i2111, %i2883
  %i2942 = and i1 %i2120, %i2883
  %i2943 = and i1 %i2121, %i2883
  %i2944 = and i1 %i2122, %i2883
  %i2945 = and i1 %i2123, %i2883
  %i2946 = and i1 %i2124, %i2883
  %i2947 = and i1 %i2125, %i2883
  %i2948 = and i1 %i2126, %i2883
  %i2949 = and i1 %i2127, %i2883
  %i2950 = and i1 %i2128, %i2883
  %i2951 = and i1 %i2129, %i2883
  %i2952 = and i1 %i2130, %i2883
  %i2953 = and i1 %i2131, %i2883
  %i2954 = and i1 %i2132, %i2883
  %i2955 = and i1 %i2133, %i2883
  %i2956 = and i1 %i2134, %i2883
  %i2957 = and i1 %i2135, %i2883
  %i2958 = and i1 %i2136, %i2883
  %i2959 = and i1 %i2137, %i2883
  %i2960 = and i1 %i2138, %i2883
  %i2961 = and i1 %i2139, %i2883
  %i2962 = and i1 %i2140, %i2883
  %i2963 = and i1 %i2141, %i2883
  %i2964 = and i1 %i2142, %i2883
  %i2965 = and i1 %i2143, %i2883
  %i2966 = and i1 %i2144, %i2883
  %i2967 = and i1 %i2145, %i2883
  %i2968 = and i1 %i2146, %i2883
  %i2969 = and i1 %i2147, %i2883
  %i2970 = and i1 %i2148, %i2883
  %i2971 = and i1 %i2149, %i2883
  %i2972 = and i1 %i2150, %i2883
  %i2973 = and i1 %arg583, %arg580
  %i2974 = and i1 %arg580, %arg580
  %i2975 = and i1 %i2154, %i2883
  %i2976 = and i1 %i2155, %i2883
  %i2977 = and i1 %i2156, %i2883
  %i2978 = and i1 %i2157, %i2883
  %i2979 = and i1 %i2158, %i2883
  %i2980 = and i1 %i2160, %i2883
  %i2981 = and i1 %i2161, %i2883
  %i2982 = and i1 %i2162, %i2883
  %i2983 = and i1 %i2163, %i2883
  %i2984 = and i1 %i2164, %i2883
  %i2985 = and i1 %i2165, %i2883
  %i2986 = shl i32 %i2817, 1
  %i2987 = select i1 %i2888, i32 %i2986, i32 0
  %i2988 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2987, i32 0, i32 0)
  %i2989 = shl i32 %arg25, 1
  %i2990 = select i1 %i2889, i32 %arg715, i32 0
  %i2991 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %i2990, i32 0, i32 0)
  %i2992 = shl i32 %i2819, 1
  %i2993 = select i1 %i2890, i32 %i2992, i32 0
  %i2994 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2993, i32 0, i32 0)
  %i2995 = shl i32 %arg869, 1
  %i2996 = select i1 %i2891, i32 %i2995, i32 0
  %i2997 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2996, i32 0, i32 0)
  %i2998 = shl i32 %.pn253666, 1
  %i2999 = select i1 %i2892, i32 %i2998, i32 0
  %i3000 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i2999, i32 0, i32 0)
  %i3001 = shl i32 0, 0
  %i3002 = select i1 %i2893, i32 0, i32 -2147483648
  %i3003 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3002, i32 0, i32 0)
  %i3004 = shl i32 %i2823, 1
  %i3005 = select i1 %i2894, i32 %i3004, i32 0
  %i3006 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3005, i32 0, i32 0)
  %i3007 = shl i32 %arg787, 1
  %i3008 = select i1 %i2895, i32 %i3007, i32 0
  %i3009 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3008, i32 0, i32 0)
  %i3010 = shl i32 %arg18, 1
  %i3011 = select i1 %arg580, i32 %arg4, i32 0
  %i3012 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %arg1011, i32 0, i32 0)
  %i3013 = shl i32 %arg49, 1
  %i3014 = select i1 %arg580, i32 %arg4, i32 0
  %i3015 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %arg1013, i32 0, i32 0)
  %i3016 = shl i32 %arg4, 1
  %i3017 = select i1 %arg580, i32 %arg4, i32 0
  %i3018 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %arg1017, i32 0, i32 0)
  %i3019 = shl i32 %arg15, 1
  %i3020 = select i1 %arg580, i32 %arg4, i32 0
  %i3021 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %arg1020, i32 0, i32 0)
  %i3022 = shl i32 %arg36, 1
  %i3023 = select i1 %arg580, i32 %arg4, i32 0
  %i3024 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %arg1023, i32 0, i32 0)
  %i3025 = shl i32 %arg42, 0
  %i3026 = select i1 %arg580, i32 %arg4, i32 0
  %i3027 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %arg908, i32 0, i32 0)
  %i3028 = shl i32 %arg4, 1
  %i3029 = select i1 %arg580, i32 %arg4, i32 0
  %i3030 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %arg721, i32 0, i32 0)
  %i3031 = shl i32 %arg4, 1
  %i3032 = select i1 %arg580, i32 %arg4, i32 0
  %i3033 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %arg4, i32 0, i32 0)
  %i3034 = shl i32 0, 0
  %i3035 = select i1 %arg580, i32 0, i32 -2147483648
  %i3036 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %arg1035, i32 0, i32 0)
  %i3037 = shl i32 0, 0
  %i3038 = select i1 %arg600, i32 0, i32 -2147483648
  %i3039 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %arg1030, i32 0, i32 0)
  %i3040 = shl i32 0, 0
  %i3041 = select i1 %arg580, i32 0, i32 -2147483648
  %i3042 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %arg1039, i32 0, i32 0)
  %i3043 = shl i32 %.decomposed, 1
  %i3044 = select i1 %arg580, i32 %arg4, i32 0
  %i3045 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %arg1042, i32 0, i32 0)
  %i3046 = shl i32 0, 0
  %i3047 = select i1 %arg581, i32 0, i32 -2147483648
  %i3048 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %arg1044, i32 0, i32 0)
  %i3049 = shl i32 0, 0
  %i3050 = select i1 %arg602, i32 0, i32 -2147483648
  %i3051 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %arg1046, i32 0, i32 0)
  %i3052 = select i1 %arg619, i32 0, i32 -2147483648
  %i3053 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i3054 = select i1 %arg580, i32 0, i32 -2147483648
  %i3055 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i3056 = select i1 %arg580, i32 0, i32 -2147483648
  %i3057 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %arg67, i32 0, i32 0)
  %i3058 = select i1 %arg1053, i32 0, i32 -2147483648
  %i3059 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3058, i32 0, i32 0)
  %i3060 = select i1 %i2914, i32 0, i32 -2147483648
  %i3061 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3060, i32 0, i32 0)
  %i3062 = select i1 %i2915, i32 0, i32 -2147483648
  %i3063 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3062, i32 0, i32 0)
  %i3064 = select i1 %i2916, i32 0, i32 -2147483648
  %i3065 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3064, i32 0, i32 0)
  %i3066 = select i1 %i2917, i32 0, i32 -2147483648
  %i3067 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3066, i32 0, i32 0)
  %i3068 = select i1 %i2918, i32 0, i32 -2147483648
  %i3069 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3068, i32 0, i32 0)
  %i3070 = select i1 %i2919, i32 0, i32 -2147483648
  %i3071 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3070, i32 0, i32 0)
  %i3072 = select i1 %i2920, i32 0, i32 -2147483648
  %i3073 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3072, i32 0, i32 0)
  %i3074 = select i1 %i2921, i32 0, i32 -2147483648
  %i3075 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3074, i32 0, i32 0)
  %i3076 = select i1 %i2922, i32 0, i32 -2147483648
  %i3077 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3076, i32 0, i32 0)
  %i3078 = select i1 %i2923, i32 0, i32 -2147483648
  %i3079 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3078, i32 0, i32 0)
  %i3080 = select i1 %i2924, i32 0, i32 -2147483648
  %i3081 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3080, i32 0, i32 0)
  %i3082 = select i1 %i2925, i32 0, i32 -2147483648
  %i3083 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3082, i32 0, i32 0)
  %i3084 = select i1 %i2926, i32 0, i32 -2147483648
  %i3085 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3084, i32 0, i32 0)
  %i3086 = select i1 %i2927, i32 0, i32 -2147483648
  %i3087 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3086, i32 0, i32 0)
  %i3088 = select i1 %i2928, i32 0, i32 -2147483648
  %i3089 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3088, i32 0, i32 0)
  %i3090 = select i1 %arg608, i32 0, i32 -2147483648
  %i3091 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %arg1055, i32 0, i32 0)
  %i3092 = select i1 %i2930, i32 0, i32 -2147483648
  %i3093 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3092, i32 0, i32 0)
  %i3094 = select i1 %i2931, i32 0, i32 -2147483648
  %i3095 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3094, i32 0, i32 0)
  %i3096 = select i1 %i2932, i32 0, i32 -2147483648
  %i3097 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3096, i32 0, i32 0)
  %i3098 = select i1 %i2933, i32 0, i32 -2147483648
  %i3099 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3098, i32 0, i32 0)
  %i3100 = select i1 %i2934, i32 0, i32 -2147483648
  %i3101 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3100, i32 0, i32 0)
  %i3102 = select i1 %i2935, i32 0, i32 -2147483648
  %i3103 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3102, i32 0, i32 0)
  %i3104 = select i1 %i2936, i32 0, i32 -2147483648
  %i3105 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3104, i32 0, i32 0)
  %i3106 = select i1 %i2937, i32 0, i32 -2147483648
  %i3107 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3106, i32 0, i32 0)
  %i3108 = select i1 %i2938, i32 0, i32 -2147483648
  %i3109 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3108, i32 0, i32 0)
  %i3110 = select i1 %i2939, i32 0, i32 -2147483648
  %i3111 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3110, i32 0, i32 0)
  %i3112 = select i1 %i2940, i32 0, i32 -2147483648
  %i3113 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3112, i32 0, i32 0)
  %i3114 = select i1 %i2941, i32 0, i32 -2147483648
  %i3115 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3114, i32 0, i32 0)
  %i3116 = select i1 %i2942, i32 0, i32 -2147483648
  %i3117 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3116, i32 0, i32 0)
  %i3118 = shl i32 %i2839, 1
  %i3119 = select i1 %i2943, i32 %i3118, i32 0
  %i3120 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3119, i32 0, i32 0)
  %i3121 = shl i32 %i2840, 1
  %i3122 = select i1 %i2944, i32 %i3121, i32 0
  %i3123 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3122, i32 0, i32 0)
  %i3124 = shl i32 %i2841, 1
  %i3125 = select i1 %i2945, i32 %i3124, i32 0
  %i3126 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3125, i32 0, i32 0)
  %i3127 = shl i32 %i2842, 1
  %i3128 = select i1 %i2946, i32 %i3127, i32 0
  %i3129 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3128, i32 0, i32 0)
  %i3130 = shl i32 %arg910, 0
  %i3131 = select i1 %i2947, i32 %i3130, i32 0
  %i3132 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3131, i32 0, i32 0)
  %i3133 = shl i32 %arg31, 1
  %i3134 = select i1 %i2948, i32 %arg4, i32 0
  %i3135 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3134, i32 0, i32 0)
  %i3136 = shl i32 %i2845, 1
  %i3137 = select i1 %i2949, i32 %i3136, i32 0
  %i3138 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3137, i32 0, i32 0)
  %i3139 = shl i32 %i2846, 1
  %i3140 = select i1 %i2950, i32 %i3139, i32 0
  %i3141 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3140, i32 0, i32 0)
  %i3142 = shl i32 %i2847, 1
  %i3143 = select i1 %i2951, i32 %i3142, i32 0
  %i3144 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3143, i32 0, i32 0)
  %i3145 = shl i32 %i2848, 1
  %i3146 = select i1 %i2952, i32 %i3145, i32 0
  %i3147 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3146, i32 0, i32 0)
  %i3148 = shl i32 %i2849, 1
  %i3149 = select i1 %i2953, i32 %i3148, i32 0
  %i3150 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3149, i32 0, i32 0)
  %i3151 = shl i32 %i2850, 1
  %i3152 = select i1 %i2954, i32 %i3151, i32 0
  %i3153 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3152, i32 0, i32 0)
  %i3154 = shl i32 %i2851, 1
  %i3155 = select i1 %i2955, i32 %i3154, i32 0
  %i3156 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3155, i32 0, i32 0)
  %i3157 = shl i32 %i2852, 1
  %i3158 = select i1 %i2956, i32 %i3157, i32 0
  %i3159 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3158, i32 0, i32 0)
  %i3160 = shl i32 %i2853, 1
  %i3161 = select i1 %i2957, i32 %i3160, i32 0
  %i3162 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3161, i32 0, i32 0)
  %i3163 = shl i32 %i2854, 1
  %i3164 = select i1 %i2958, i32 %i3163, i32 0
  %i3165 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3164, i32 0, i32 0)
  %i3166 = shl i32 %i2855, 1
  %i3167 = select i1 %i2959, i32 %i3166, i32 0
  %i3168 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3167, i32 0, i32 0)
  %i3169 = shl i32 %.pn223681, 1
  %i3170 = select i1 %i2960, i32 %i3169, i32 0
  %i3171 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3170, i32 0, i32 0)
  %i3172 = shl i32 %arg46, 1
  %i3173 = select i1 %i2961, i32 %arg1056, i32 0
  %i3174 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %i3173, i32 0, i32 0)
  %i3175 = shl i32 %i2858, 1
  %i3176 = select i1 %i2962, i32 %i3175, i32 0
  %i3177 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3176, i32 0, i32 0)
  %i3178 = shl i32 %i2859, 1
  %i3179 = select i1 %i2963, i32 %i3178, i32 0
  %i3180 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3179, i32 0, i32 0)
  %i3181 = shl i32 %i2860, 1
  %i3182 = select i1 %i2964, i32 %i3181, i32 0
  %i3183 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3182, i32 0, i32 0)
  %i3184 = shl i32 %arg43, 1
  %i3185 = select i1 %i2965, i32 %arg1030, i32 0
  %i3186 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3185, i32 0, i32 0)
  %i3187 = shl i32 %i2862, 1
  %i3188 = select i1 %i2966, i32 %i3187, i32 0
  %i3189 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3188, i32 0, i32 0)
  %i3190 = shl i32 %i2863, 1
  %i3191 = select i1 %i2967, i32 %i3190, i32 0
  %i3192 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3191, i32 0, i32 0)
  %i3193 = shl i32 %i2864, 1
  %i3194 = select i1 %i2968, i32 %i3193, i32 0
  %i3195 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3194, i32 0, i32 0)
  %i3196 = shl i32 %arg144, 1
  %i3197 = select i1 %i2969, i32 %i3196, i32 0
  %i3198 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3197, i32 0, i32 0)
  %i3199 = shl i32 %arg31, 1
  %i3200 = select i1 %i2970, i32 %arg92, i32 0
  %i3201 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3200, i32 0, i32 0)
  %i3202 = shl i32 %arg76, 1
  %i3203 = select i1 %i2971, i32 %i3202, i32 0
  %i3204 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %i3203, i32 0, i32 0)
  %i3205 = select i1 %i2972, i32 0, i32 -2147483648
  %i3206 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3205, i32 0, i32 0)
  %i3207 = select i1 %arg609, i32 0, i32 -2147483648
  %i3208 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %arg1026, i32 0, i32 0)
  %i3209 = shl i32 %arg4, 1
  %i3210 = select i1 %arg580, i32 %arg4, i32 0
  %i3211 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg688, i32 %arg127, i32 0, i32 0)
  %i3212 = shl i32 %i2869, 1
  %i3213 = select i1 %i2975, i32 %i3212, i32 0
  %i3214 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3213, i32 0, i32 0)
  %i3215 = shl i32 %i2870, 1
  %i3216 = select i1 %i2976, i32 %i3215, i32 0
  %i3217 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3216, i32 0, i32 0)
  %i3218 = shl i32 %i2871, 1
  %i3219 = select i1 %i2977, i32 %i3218, i32 0
  %i3220 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3219, i32 0, i32 0)
  %i3221 = shl i32 %i2872, 1
  %i3222 = select i1 %i2978, i32 %i3221, i32 0
  %i3223 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3222, i32 0, i32 0)
  %i3224 = shl i32 %i2873, 1
  %i3225 = select i1 %i2979, i32 %i3224, i32 0
  %i3226 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3225, i32 0, i32 0)
  %i3227 = shl i32 %i2874, 1
  %i3228 = select i1 %i2980, i32 %i3227, i32 0
  %i3229 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %i3228, i32 0, i32 0)
  %i3230 = shl i32 %i2875, 1
  %i3231 = select i1 %i2981, i32 %i3230, i32 0
  %i3232 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3231, i32 0, i32 0)
  %i3233 = shl i32 %i2876, 1
  %i3234 = select i1 %i2982, i32 %i3233, i32 0
  %i3235 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %i3234, i32 0, i32 0)
  %i3236 = shl i32 %i2877, 1
  %i3237 = select i1 %i2983, i32 %i3236, i32 0
  %i3238 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3237, i32 0, i32 0)
  %i3239 = shl i32 %i2878, 1
  %i3240 = select i1 %i2984, i32 %i3239, i32 0
  %i3241 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3240, i32 0, i32 0)
  %i3242 = shl i32 %i2879, 1
  %i3243 = select i1 %i2985, i32 %i3242, i32 0
  %i3244 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i2383, i32 %i3243, i32 0, i32 0)
  %i3245 = insertelement <4 x float> zeroinitializer, float %arg1062, i64 0
  %i3246 = insertelement <4 x float> %arg1063, float %arg1064, i64 0
  %i3247 = insertelement <4 x float> %arg1065, float 0.000000e+00, i64 1
  %i3248 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %arg1066, i32 0, i32 0, i32 0)
  %i3249 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %arg1067, i32 0, i32 0, i32 0)
  %i3250 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %arg1068, i32 0, i32 0, i32 0)
  %i3251 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %arg1069, i32 0, i32 0, i32 0)
  %i3252 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %arg1070, i32 0, i32 0, i32 0)
  %i3253 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %arg1071, i32 0, i32 0, i32 0)
  %i3254 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %arg1072, i32 0, i32 0, i32 0)
  %i3255 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> %arg1073, i32 0, i32 0, i32 0)
  %i3256 = extractelement <4 x float> %arg1074, i64 0
  %i3257 = extractelement <4 x float> %arg1074, i64 0
  %i3258 = getelementptr inbounds nuw i8, ptr addrspace(3) null, i32 %arg832
  store i16 %i2887, ptr addrspace(3) null, align 2
  %.idx655 = shl i32 0, 0
  %i3259 = getelementptr i8, ptr addrspace(3) @global_smem, i32 %.idx655
  %i3260 = getelementptr inbounds nuw i8, ptr addrspace(3) %i3259, i32 %i2694
  store i16 %i2988, ptr addrspace(3) %i3260, align 2
  store i16 %i3012, ptr addrspace(3) null, align 2
  %i3261 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %arg1076
  store i16 %i3036, ptr addrspace(3) %i3261, align 2
  store i16 %i3057, ptr addrspace(3) %arg1075, align 2
  store i16 %i3071, ptr addrspace(3) null, align 2
  %i3262 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %arg1077
  store i16 %i3085, ptr addrspace(3) %arg1078, align 2
  store i16 %i3099, ptr addrspace(3) %arg1075, align 2
  store i16 %i3113, ptr addrspace(3) null, align 2
  %i3263 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %i2701
  store i16 0, ptr addrspace(3) %i3263, align 2
  %i3264 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %i2703
  store i16 0, ptr addrspace(3) %i3264, align 2
  %i3265 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %i2704
  store i16 %i3126, ptr addrspace(3) %i3265, align 2
  store i16 %i3150, ptr addrspace(3) null, align 2
  %i3266 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %i2705
  store i16 %i3174, ptr addrspace(3) %i3266, align 2
  %i3267 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %arg1079
  store i16 %i3198, ptr addrspace(3) %arg1080, align 2
  %i3268 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %i2709
  store i16 %i3217, ptr addrspace(3) %arg1081, align 2
  %i3269 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %arg1082
  store i16 %i3238, ptr addrspace(3) %arg1083, align 2
  store i16 %i2991, ptr addrspace(3) null, align 2
  %i3270 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %i2713
  store i16 %i3015, ptr addrspace(3) %i3270, align 2
  store i16 %i3039, ptr addrspace(3) null, align 2
  store i16 %i3059, ptr addrspace(3) %arg1075, align 2
  store i16 %i3115, ptr addrspace(3) null, align 2
  %i3271 = getelementptr inbounds nuw i8, ptr addrspace(3) %i3259, i32 %i2716
  store i16 0, ptr addrspace(3) %i3271, align 2
  %i3272 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %arg1084
  store i16 0, ptr addrspace(3) null, align 2
  %i3273 = getelementptr inbounds nuw i8, ptr addrspace(3) %i3259, i32 %i2719
  store i16 %i3129, ptr addrspace(3) %i3273, align 2
  store i16 %i3153, ptr addrspace(3) null, align 2
  %i3274 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %i2720
  store i16 %i3177, ptr addrspace(3) %i3274, align 2
  %i3275 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %arg1085
  store i16 %i3201, ptr addrspace(3) null, align 2
  %i3276 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %arg1086
  store i16 %i3220, ptr addrspace(3) %arg1087, align 2
  %i3277 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %i2726
  store i16 %i3241, ptr addrspace(3) %i3277, align 2
  store i16 %i2994, ptr addrspace(3) null, align 2
  %i3278 = getelementptr inbounds nuw i8, ptr addrspace(3) %i3259, i32 %i2729
  store i16 %i3018, ptr addrspace(3) %i3278, align 2
  store i16 %i3042, ptr addrspace(3) %arg1075, align 2
  store i16 %i3061, ptr addrspace(3) null, align 2
  %i3279 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %i2731
  store i16 %i3073, ptr addrspace(3) %i3279, align 2
  store i16 %i3087, ptr addrspace(3) %arg1075, align 2
  store i16 %i3101, ptr addrspace(3) null, align 2
  %i3280 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %arg1088
  store i16 %i3132, ptr addrspace(3) %i3280, align 2
  %i3281 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %arg1089
  store i16 %i3156, ptr addrspace(3) null, align 2
  %i3282 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %arg1090
  store i16 %i3180, ptr addrspace(3) %arg1091, align 2
  %i3283 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %i2739
  store i16 %i3204, ptr addrspace(3) %arg1092, align 2
  %i3284 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %arg1093
  store i16 %i3223, ptr addrspace(3) null, align 2
  %i3285 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %arg1094
  store i16 %i3244, ptr addrspace(3) %arg1095, align 2
  %i3286 = getelementptr inbounds nuw i8, ptr addrspace(3) %i3259, i32 %i2745
  store i16 %i2997, ptr addrspace(3) %i3286, align 2
  store i16 %i3021, ptr addrspace(3) null, align 2
  store i16 %i3045, ptr addrspace(3) %arg1075, align 2
  store i16 %i3075, ptr addrspace(3) null, align 2
  store i16 %i3089, ptr addrspace(3) %arg1075, align 2
  store i16 %i3103, ptr addrspace(3) null, align 2
  store i16 0, ptr addrspace(3) %arg1075, align 2
  %i3287 = getelementptr inbounds nuw i8, ptr addrspace(3) %i3259, i32 %i2748
  store i16 0, ptr addrspace(3) %i3287, align 2
  store i16 %i3135, ptr addrspace(3) null, align 2
  %i3288 = getelementptr inbounds nuw i8, ptr addrspace(3) %i3259, i32 %i2749
  store i16 %i3159, ptr addrspace(3) %i3288, align 2
  %i3289 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %arg1096
  store i16 %i3183, ptr addrspace(3) null, align 2
  %i3290 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %i2753
  store i16 %i3206, ptr addrspace(3) %i3290, align 2
  %i3291 = getelementptr inbounds nuw i8, ptr addrspace(3) %i3259, i32 %i2755
  store i16 %i3226, ptr addrspace(3) %i3291, align 2
  %i3292 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %arg1097
  store i16 0, ptr addrspace(3) null, align 2
  %i3293 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %arg1098
  store i16 %i3000, ptr addrspace(3) %arg1099, align 2
  store i16 %i3024, ptr addrspace(3) null, align 2
  %i3294 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %i2760
  store i16 %i3048, ptr addrspace(3) %arg1100, align 2
  store i16 %i3063, ptr addrspace(3) %arg1075, align 2
  store i16 %i3077, ptr addrspace(3) null, align 2
  store i16 %i3091, ptr addrspace(3) %arg1075, align 2
  store i16 %i3105, ptr addrspace(3) null, align 2
  %i3295 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %arg1101
  store i16 0, ptr addrspace(3) %arg1102, align 2
  %i3296 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %arg1103
  store i16 0, ptr addrspace(3) null, align 2
  store i16 %i3138, ptr addrspace(3) null, align 2
  %i3297 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %i2767
  store i16 %i3162, ptr addrspace(3) %i3297, align 2
  %i3298 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %i2769
  store i16 %i3186, ptr addrspace(3) null, align 2
  %i3299 = getelementptr inbounds nuw i8, ptr addrspace(3) %i3259, i32 %i2771
  store i16 %i3003, ptr addrspace(3) %i3299, align 2
  store i16 %i3027, ptr addrspace(3) null, align 2
  %i3300 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %i2773
  store i16 %i3051, ptr addrspace(3) %i3300, align 2
  store i16 %i3065, ptr addrspace(3) %arg1075, align 2
  store i16 %i3079, ptr addrspace(3) null, align 2
  store i16 %i3093, ptr addrspace(3) %arg1075, align 2
  store i16 %i3107, ptr addrspace(3) null, align 2
  %i3301 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %i2774
  store i16 0, ptr addrspace(3) %i3301, align 2
  %i3302 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %i2775
  store i16 %i3117, ptr addrspace(3) %i3302, align 2
  store i16 %i3141, ptr addrspace(3) null, align 2
  %i3303 = getelementptr inbounds nuw i8, ptr addrspace(3) %i3259, i32 %i2776
  store i16 %i3165, ptr addrspace(3) %i3303, align 2
  %i3304 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %i2778
  store i16 %i3189, ptr addrspace(3) %i3304, align 2
  %i3305 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %i2779
  store i16 %i3208, ptr addrspace(3) %i3305, align 2
  %i3306 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %i2781
  store i16 %i3229, ptr addrspace(3) null, align 2
  %i3307 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %arg853
  store i16 0, ptr addrspace(3) %i3307, align 2
  store i16 %i3006, ptr addrspace(3) null, align 2
  store i16 %i3030, ptr addrspace(3) %arg1075, align 2
  store i16 %arg1104, ptr addrspace(3) null, align 2
  %i3308 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %i2785
  store i16 %i3067, ptr addrspace(3) %i3308, align 2
  store i16 %i3081, ptr addrspace(3) null, align 2
  store i16 %i3095, ptr addrspace(3) %arg1075, align 2
  store i16 %i3109, ptr addrspace(3) null, align 2
  %i3309 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %i2787
  store i16 %i3120, ptr addrspace(3) %i3309, align 2
  %i3310 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %i2789
  store i16 %i3144, ptr addrspace(3) %i3310, align 2
  %i3311 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %i2790
  store i16 %i3168, ptr addrspace(3) %i3311, align 2
  %i3312 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %i2792
  store i16 %i3192, ptr addrspace(3) %arg1105, align 2
  %i3313 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %arg873
  store i16 %i3211, ptr addrspace(3) null, align 2
  %i3314 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %arg1106
  store i16 %i3232, ptr addrspace(3) %i3314, align 2
  %i3315 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %i2796
  store i16 0, ptr addrspace(3) %i3315, align 2
  store i16 %i3009, ptr addrspace(3) null, align 2
  store i16 %arg1107, ptr addrspace(3) %arg1075, align 2
  store i16 %arg1108, ptr addrspace(3) null, align 2
  %i3316 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %i2799
  store i16 %i3069, ptr addrspace(3) %i3316, align 2
  store i16 %i3083, ptr addrspace(3) %arg1075, align 2
  store i16 %i3097, ptr addrspace(3) null, align 2
  store i16 %i3111, ptr addrspace(3) %arg1075, align 2
  store i16 %i3123, ptr addrspace(3) null, align 2
  %i3317 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %arg890
  store i16 %i3147, ptr addrspace(3) %i3317, align 2
  %i3318 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %arg1109
  store i16 %i3171, ptr addrspace(3) %i3318, align 2
  %i3319 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %arg894
  store i16 %i3195, ptr addrspace(3) null, align 2
  %i3320 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %i2806
  store i16 %i3214, ptr addrspace(3) %arg1110, align 2
  %i3321 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1075, i32 %arg898
  store i16 %i3235, ptr addrspace(3) null, align 2
  %exitcond.not = icmp eq i32 %i2880, %smax
  br i1 %exitcond.not, label %._crit_edge.loopexit, label %bb2812

._crit_edge.loopexit:                             ; preds = %bb2812
  br label %._crit_edge

._crit_edge:                                      ; preds = %._crit_edge.loopexit, %.._crit_edge_crit_edge
  %.pre-phi943 = phi i32 [ %.pre, %.._crit_edge_crit_edge ], [ %.decomposed, %._crit_edge.loopexit ]
  %i3322 = xor i32 %.pre-phi943, 64
  %i3323 = or disjoint i32 %i3322, 1
  %i3324 = xor i32 %.pre-phi943, 1
  %i3325 = or disjoint i32 %i3324, 1
  br i1 %i1498, label %bb3326, label %._crit_edge._crit_edge

bb3326:                                           ; preds = %._crit_edge
  %i3327 = getelementptr inbounds nuw i8, ptr addrspace(3) null, i32 %i3325
  %i3328 = load <8 x half>, ptr addrspace(3) %i3327, align 16
  %i3329 = getelementptr inbounds nuw i8, ptr addrspace(3) null, i32 %i3323
  %i3330 = load <8 x half>, ptr addrspace(3) %i3329, align 16
  ret void

._crit_edge._crit_edge:                           ; preds = %._crit_edge
  %i3331 = icmp slt i64 %i1734, %i2054
  %i3332 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %arg2, i16 0, i64 1, i32 1)
  br i1 %arg903, label %.lr.ph923, label %._crit_edge924

.lr.ph923:                                        ; preds = %._crit_edge._crit_edge
  br label %bb3333

bb3333:                                           ; preds = %bb3333, %.lr.ph923
  %i3334 = phi i32 [ 0, %.lr.ph923 ], [ %arg139, %bb3333 ]
  %i3335 = add i32 %i3334, 1
  %.not = icmp slt i32 0, %i1497
  %i3336 = select i1 %.not, i32 %i3335, i32 0
  %i3337 = shl i32 %i3336, 1
  %i3338 = or disjoint i32 %arg45, %arg4
  %i3339 = or disjoint i32 %arg4, %arg22
  %i3340 = or disjoint i32 %i3337, %i1364
  %i3341 = or disjoint i32 %i3337, %arg39
  %i3342 = or disjoint i32 %arg7, %arg4
  %i3343 = or disjoint i32 %arg4, %arg4
  %i3344 = or disjoint i32 %arg4, %arg43
  %i3345 = or disjoint i32 %i3337, %arg46
  %i3346 = or disjoint i32 %i3337, %arg47
  %i3347 = or disjoint i32 %arg51, %arg4
  %i3348 = or disjoint i32 0, %arg4
  %i3349 = or disjoint i32 %i3337, %i1377
  %i3350 = or disjoint i32 %i3337, %arg1213
  %i3351 = or disjoint i32 %i3337, %arg48
  %i3352 = or disjoint i32 %arg4, %arg4
  %i3353 = or disjoint i32 %arg4, %arg58
  %i3354 = or disjoint i32 %i3337, %arg56
  %i3355 = or disjoint i32 %i3337, %arg928
  %i3356 = or disjoint i32 %i3337, %arg130
  %i3357 = or disjoint i32 %arg4, %arg4
  %i3358 = or disjoint i32 %i3337, %arg120
  %i3359 = or disjoint i32 %i3337, %arg766
  %i3360 = or disjoint i32 %i3337, %arg45
  %i3361 = or disjoint i32 %i3337, %arg82
  %i3362 = or disjoint i32 %arg4, %arg4
  %i3363 = or disjoint i32 %i3337, %arg63
  %i3364 = or disjoint i32 %i3337, %arg51
  %i3365 = or disjoint i32 %arg4, %arg4
  %i3366 = or disjoint i32 %i3337, %arg79
  %i3367 = or disjoint i32 %arg54, %arg4
  %i3368 = or disjoint i32 %arg4, %arg4
  %i3369 = or disjoint i32 %i3337, %arg65
  %i3370 = or disjoint i32 %i3337, %arg86
  %i3371 = or disjoint i32 %arg20, %arg4
  %i3372 = or disjoint i32 %arg4, %arg4
  %i3373 = or disjoint i32 %arg4, %arg4
  %i3374 = or disjoint i32 %arg4, %arg4
  %i3375 = or disjoint i32 0, 0
  %i3376 = or disjoint i32 %arg4, %arg4
  %i3377 = or disjoint i32 0, 0
  %i3378 = or disjoint i32 %arg4, 1
  %i3379 = or disjoint i32 %i3337, 1
  %i3380 = or disjoint i32 %arg25, 0
  %i3381 = or disjoint i32 0, 0
  %i3382 = or disjoint i32 %i3337, %arg963
  %i3383 = or disjoint i32 %arg4, 0
  %i3384 = or disjoint i32 %i3337, %arg96
  %i3385 = or disjoint i32 %arg40, %arg4
  %i3386 = or disjoint i32 %arg4, %arg4
  %i3387 = or disjoint i32 %arg4, %arg4
  %i3388 = or disjoint i32 %arg4, %arg4
  %i3389 = or disjoint i32 %arg4, %arg4
  %i3390 = or disjoint i32 %arg4, %arg4
  %i3391 = or disjoint i32 %arg4, %arg4
  %i3392 = or disjoint i32 %arg4, %arg4
  %i3393 = or disjoint i32 %arg4, %arg4
  %i3394 = or disjoint i32 0, 0
  %i3395 = or disjoint i32 %arg4, %arg4
  %i3396 = or disjoint i32 %arg4, %arg4
  %i3397 = or disjoint i32 %arg4, %arg4
  %i3398 = or disjoint i32 %arg4, %arg4
  %i3399 = or disjoint i32 %arg4, %arg4
  %i3400 = or disjoint i32 %arg4, %arg4
  %i3401 = or disjoint i32 %arg4, %arg4
  %i3402 = or disjoint i32 %arg4, %arg4
  %i3403 = or disjoint i32 %arg4, %arg4
  %i3404 = or disjoint i32 %arg4, %arg4
  %i3405 = or disjoint i32 %arg4, %arg4
  %i3406 = or disjoint i32 %arg4, %arg4
  %i3407 = or disjoint i32 %arg4, %arg4
  %i3408 = or disjoint i32 %arg4, %arg4
  %i3409 = or disjoint i32 %arg4, %arg4
  %i3410 = or disjoint i32 %arg4, %arg4
  %i3411 = or disjoint i32 1, 1
  %i3412 = icmp slt i32 %arg4, %arg4
  %i3413 = icmp slt i32 %arg4, %arg6
  %i3414 = icmp slt i32 %i3340, %arg6
  %i3415 = icmp slt i32 %i3341, %arg6
  %i3416 = icmp slt i32 %arg9, %arg4
  %i3417 = icmp slt i32 %arg4, %arg4
  %i3418 = icmp slt i32 %arg4, %arg6
  %i3419 = icmp slt i32 %i3345, %arg6
  %i3420 = icmp slt i32 %i3346, 1
  %i3421 = icmp slt i32 %arg16, %arg4
  %i3422 = icmp slt i32 %arg4, %arg4
  %i3423 = icmp slt i32 %i3349, %arg6
  %i3424 = icmp slt i32 %i3350, %arg6
  %i3425 = icmp slt i32 %i3351, %arg6
  %i3426 = icmp slt i32 %arg12, %arg4
  %i3427 = icmp slt i32 %arg4, %arg6
  %i3428 = icmp slt i32 %i3354, %arg6
  %i3429 = icmp slt i32 %i3355, %arg6
  %i3430 = icmp slt i32 %i3356, %arg6
  %i3431 = icmp slt i32 %arg61, %arg6
  %i3432 = icmp slt i32 %i3358, %arg6
  %i3433 = icmp slt i32 %i3359, %arg6
  %i3434 = icmp slt i32 %i3360, %arg6
  %i3435 = icmp slt i32 %i3361, 1
  %i3436 = icmp slt i32 %arg44, %arg6
  %i3437 = icmp slt i32 %i3363, %arg6
  %i3438 = icmp slt i32 %i3364, %arg6
  %i3439 = icmp slt i32 %arg57, %arg6
  %i3440 = icmp slt i32 %i3366, %arg6
  %i3441 = icmp slt i32 %arg25, %arg4
  %i3442 = icmp slt i32 %arg4, %arg6
  %i3443 = icmp slt i32 %i3369, %arg6
  %i3444 = icmp slt i32 %i3370, %arg6
  %i3445 = icmp slt i32 %arg18, %arg4
  %i3446 = icmp slt i32 %arg4, %arg4
  %i3447 = icmp slt i32 %arg4, %arg4
  %i3448 = icmp slt i32 %arg4, 0
  %i3449 = icmp slt i32 0, 0
  %i3450 = icmp slt i32 %arg4, 0
  %i3451 = icmp slt i32 0, 0
  %i3452 = icmp slt i32 %arg4, 0
  %i3453 = icmp slt i32 %i3379, %arg6
  %i3454 = icmp slt i32 %arg7, 0
  %i3455 = icmp slt i32 0, 0
  %i3456 = icmp slt i32 %i3382, 0
  %i3457 = icmp slt i32 %arg39, %arg6
  %i3458 = icmp slt i32 %i3384, %arg6
  %i3459 = icmp slt i32 %arg54, 0
  %i3460 = icmp slt i32 %arg4, %arg4
  %i3461 = icmp slt i32 %arg4, 0
  %i3462 = icmp slt i32 %arg4, 0
  %i3463 = icmp slt i32 %arg4, %arg4
  %i3464 = icmp slt i32 %arg4, 0
  %i3465 = icmp slt i32 %arg4, %arg4
  %i3466 = icmp slt i32 %arg4, 0
  %i3467 = icmp slt i32 %arg4, 0
  %i3468 = icmp slt i32 0, 0
  %i3469 = icmp slt i32 %arg4, %arg4
  %i3470 = icmp slt i32 %arg4, 0
  %i3471 = icmp slt i32 %arg4, %arg4
  %i3472 = icmp slt i32 %arg4, 0
  %i3473 = icmp slt i32 %arg4, 0
  %i3474 = icmp slt i32 %arg4, 0
  %i3475 = icmp slt i32 %arg4, 0
  %i3476 = icmp slt i32 %arg4, %arg4
  %i3477 = icmp slt i32 %arg4, 0
  %i3478 = icmp slt i32 %arg4, 0
  %i3479 = icmp slt i32 %arg4, 0
  %i3480 = icmp slt i32 %arg4, 0
  %i3481 = icmp slt i32 %arg4, %arg4
  %i3482 = icmp slt i32 %arg4, 0
  %i3483 = icmp slt i32 %arg4, %arg4
  %i3484 = icmp slt i32 %arg4, 0
  %i3485 = icmp slt i32 1, 0
  %i3486 = and i1 %arg580, %arg580
  %i3487 = and i1 %arg580, %arg580
  %i3488 = and i1 %i3331, %i3414
  %i3489 = and i1 %arg1160, %i3415
  %i3490 = and i1 %arg1160, %arg621
  %i3491 = and i1 %arg580, %arg580
  %i3492 = and i1 %arg580, %arg580
  %i3493 = and i1 %arg1160, %i3419
  %i3494 = and i1 %arg1160, %i3420
  %i3495 = and i1 %arg605, %arg580
  %i3496 = and i1 true, %arg580
  %i3497 = and i1 %arg1160, %i3423
  %i3498 = and i1 %arg1160, %i3424
  %i3499 = and i1 %arg1160, %i3425
  %i3500 = and i1 %arg585, %arg580
  %i3501 = and i1 %arg580, %arg580
  %i3502 = and i1 %arg1160, %i3428
  %i3503 = and i1 %arg1160, %i3429
  %i3504 = and i1 %arg1160, %i3430
  %i3505 = and i1 %arg590, %arg580
  %i3506 = and i1 %arg1160, %i3432
  %i3507 = and i1 %arg1160, %i3433
  %i3508 = and i1 %arg1160, %i3434
  %i3509 = and i1 %arg1160, %i3435
  %i3510 = and i1 %arg609, %arg580
  %i3511 = and i1 %arg1160, %i3437
  %i3512 = and i1 %arg1160, %i3438
  %i3513 = and i1 %arg593, %arg580
  %i3514 = and i1 %arg1160, %i3440
  %i3515 = and i1 %arg609, %arg580
  %i3516 = and i1 %arg580, %arg580
  %i3517 = and i1 %arg1160, %i3443
  %i3518 = and i1 %arg1160, %i3444
  %i3519 = and i1 %arg588, %arg580
  %i3520 = and i1 %arg580, %arg580
  %i3521 = and i1 %arg580, %arg580
  %i3522 = and i1 %arg580, %arg580
  %i3523 = and i1 false, false
  %i3524 = and i1 %arg580, %arg580
  %i3525 = and i1 false, false
  %i3526 = and i1 %arg580, %arg580
  %i3527 = and i1 %arg1160, %i3453
  %i3528 = and i1 %arg619, %arg580
  %i3529 = and i1 false, false
  %i3530 = and i1 %arg1160, %i3456
  %i3531 = and i1 %arg626, %arg580
  %i3532 = and i1 %arg1160, %i3458
  %i3533 = and i1 %arg582, %arg580
  %i3534 = and i1 %arg580, %arg580
  %i3535 = and i1 %arg580, %arg580
  %i3536 = and i1 %arg580, %arg580
  %i3537 = and i1 %arg580, %arg580
  %i3538 = and i1 %arg580, %arg580
  %i3539 = and i1 %arg580, %arg580
  %i3540 = and i1 %arg580, %arg580
  %i3541 = and i1 %arg580, %arg580
  %i3542 = and i1 false, false
  %i3543 = and i1 %arg580, %arg580
  %i3544 = and i1 %arg580, %arg580
  %i3545 = and i1 %arg580, %arg580
  %i3546 = and i1 %arg580, %arg580
  %i3547 = and i1 %arg580, %arg580
  %i3548 = and i1 %arg580, %arg580
  %i3549 = and i1 %arg580, %arg580
  %i3550 = and i1 %arg580, %arg580
  %i3551 = and i1 %arg580, %arg580
  %i3552 = and i1 %arg580, %arg580
  %i3553 = and i1 %arg580, %arg580
  %i3554 = and i1 %arg580, %arg580
  %i3555 = and i1 %arg580, %arg580
  %i3556 = and i1 %arg580, %arg580
  %i3557 = and i1 %arg580, %arg580
  %i3558 = and i1 %arg580, %arg580
  %i3559 = and i1 true, true
  %i3560 = select i1 %arg580, i32 0, i32 -2147483648
  %i3561 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i3562 = select i1 %arg580, i32 0, i32 -2147483648
  %i3563 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i3564 = select i1 %i3488, i32 0, i32 -2147483648
  %i3565 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i3332, i32 %i3564, i32 0, i32 0)
  %i3566 = select i1 %i3489, i32 0, i32 -2147483648
  %i3567 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i3332, i32 %i3566, i32 0, i32 0)
  %i3568 = select i1 %i3490, i32 0, i32 -2147483648
  %i3569 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i3332, i32 %i3568, i32 0, i32 0)
  %i3570 = select i1 %arg620, i32 0, i32 -2147483648
  %i3571 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i3572 = select i1 %arg580, i32 0, i32 -2147483648
  %i3573 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i3574 = select i1 %i3493, i32 0, i32 -2147483648
  %i3575 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i3332, i32 %i3574, i32 0, i32 0)
  %i3576 = select i1 %i3494, i32 0, i32 -2147483648
  %i3577 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i3332, i32 %i3576, i32 0, i32 0)
  %i3578 = select i1 %arg621, i32 0, i32 -2147483648
  %i3579 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg1208, i32 %arg22, i32 0, i32 0)
  %i3580 = select i1 %arg580, i32 0, i32 1
  %i3581 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 %arg4, i32 0, i32 0)
  %i3582 = select i1 %i3497, i32 0, i32 -2147483648
  %i3583 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i3332, i32 %i3582, i32 0, i32 0)
  %i3584 = select i1 %i3498, i32 0, i32 -2147483648
  %i3585 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i3332, i32 %i3584, i32 0, i32 0)
  %i3586 = select i1 %i3499, i32 0, i32 -2147483648
  %i3587 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i3332, i32 %i3586, i32 0, i32 0)
  %i3588 = select i1 %arg582, i32 0, i32 1
  %i3589 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i3590 = select i1 %arg580, i32 0, i32 -2147483648
  %i3591 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i3592 = select i1 %i3502, i32 0, i32 -2147483648
  %i3593 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i3332, i32 %i3592, i32 0, i32 0)
  %i3594 = select i1 %i3503, i32 0, i32 -2147483648
  %i3595 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i3332, i32 %i3594, i32 0, i32 0)
  %i3596 = select i1 %i3504, i32 0, i32 -2147483648
  %i3597 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i3332, i32 %i3596, i32 0, i32 0)
  %i3598 = select i1 %arg583, i32 0, i32 -2147483648
  %i3599 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i3600 = select i1 %i3506, i32 0, i32 -2147483648
  %i3601 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i3332, i32 %i3600, i32 0, i32 0)
  %i3602 = select i1 %i3507, i32 0, i32 -2147483648
  %i3603 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i3332, i32 %i3602, i32 0, i32 0)
  %i3604 = select i1 %i3508, i32 0, i32 -2147483648
  %i3605 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i3332, i32 %i3604, i32 0, i32 0)
  %i3606 = select i1 %i3509, i32 0, i32 -2147483648
  %i3607 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i3332, i32 %i3606, i32 0, i32 0)
  %i3608 = select i1 %arg626, i32 0, i32 -2147483648
  %i3609 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i3610 = select i1 %i3511, i32 0, i32 -2147483648
  %i3611 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i3332, i32 %i3610, i32 0, i32 0)
  %i3612 = select i1 %i3512, i32 0, i32 -2147483648
  %i3613 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i3332, i32 %i3612, i32 0, i32 0)
  %i3614 = select i1 %arg582, i32 0, i32 -2147483648
  %i3615 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i3616 = select i1 %i3514, i32 0, i32 -2147483648
  %i3617 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i3332, i32 %i3616, i32 0, i32 0)
  %i3618 = select i1 %arg619, i32 0, i32 -2147483648
  %i3619 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i3620 = select i1 %arg580, i32 0, i32 -2147483648
  %i3621 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i3622 = select i1 %i3517, i32 0, i32 -2147483648
  %i3623 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i3332, i32 %i3622, i32 0, i32 0)
  %i3624 = select i1 %i3518, i32 0, i32 -2147483648
  %i3625 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i3332, i32 %i3624, i32 0, i32 0)
  %i3626 = select i1 %arg583, i32 0, i32 -2147483648
  %i3627 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i3628 = select i1 %arg580, i32 0, i32 1
  %i3629 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i3630 = select i1 %arg580, i32 0, i32 -2147483648
  %i3631 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg1208, i32 %arg101, i32 0, i32 0)
  %i3632 = select i1 %arg580, i32 0, i32 -2147483648
  %i3633 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg1208, i32 %arg56, i32 0, i32 0)
  %i3634 = select i1 false, i32 0, i32 0
  %i3635 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %i3636 = select i1 %arg580, i32 0, i32 -2147483648
  %i3637 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg1208, i32 %arg944, i32 0, i32 0)
  %i3638 = select i1 false, i32 0, i32 0
  %i3639 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %i3640 = select i1 %arg580, i32 0, i32 -2147483648
  %i3641 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg1208, i32 %arg60, i32 0, i32 0)
  %i3642 = select i1 %i3527, i32 0, i32 -2147483648
  %i3643 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i3332, i32 %i3642, i32 0, i32 0)
  %i3644 = select i1 %arg582, i32 0, i32 -2147483648
  %i3645 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i3646 = select i1 false, i32 0, i32 0
  %i3647 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %i3648 = select i1 %i3530, i32 0, i32 -2147483648
  %i3649 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i3332, i32 %i3648, i32 0, i32 0)
  %i3650 = select i1 %arg600, i32 0, i32 -2147483648
  %i3651 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg1208, i32 %arg96, i32 0, i32 0)
  %i3652 = select i1 %i3532, i32 0, i32 -2147483648
  %i3653 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %i3332, i32 %i3652, i32 0, i32 0)
  %i3654 = select i1 %arg616, i32 0, i32 -2147483648
  %i3655 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i3656 = select i1 %arg580, i32 0, i32 -2147483648
  %i3657 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i3658 = select i1 %arg580, i32 0, i32 -2147483648
  %i3659 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg1208, i32 %arg752, i32 0, i32 0)
  %i3660 = select i1 %arg580, i32 0, i32 -2147483648
  %i3661 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg1208, i32 %arg31, i32 0, i32 0)
  %i3662 = select i1 %arg580, i32 0, i32 -2147483648
  %i3663 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i3664 = select i1 %arg580, i32 0, i32 -2147483648
  %i3665 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg1208, i32 %arg107, i32 0, i32 0)
  %i3666 = select i1 %arg580, i32 0, i32 -2147483648
  %i3667 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i3668 = select i1 %arg580, i32 0, i32 -2147483648
  %i3669 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg1208, i32 %arg70, i32 0, i32 0)
  %i3670 = select i1 %arg580, i32 0, i32 1
  %i3671 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg1208, i32 %arg49, i32 0, i32 0)
  %i3672 = select i1 false, i32 0, i32 0
  %i3673 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) null, i32 0, i32 0, i32 0)
  %i3674 = select i1 %arg580, i32 0, i32 -2147483648
  %i3675 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg1208, i32 %arg15, i32 0, i32 0)
  %i3676 = select i1 %arg580, i32 0, i32 -2147483648
  %i3677 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i3678 = select i1 %arg580, i32 0, i32 -2147483648
  %i3679 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i3680 = select i1 %arg580, i32 0, i32 -2147483648
  %i3681 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg1208, i32 %arg95, i32 0, i32 0)
  %i3682 = select i1 %arg580, i32 0, i32 -2147483648
  %i3683 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg1208, i32 %arg4, i32 0, i32 0)
  %i3684 = select i1 %arg580, i32 0, i32 -2147483648
  %i3685 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i3686 = select i1 %arg580, i32 0, i32 -2147483648
  %i3687 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i3688 = select i1 %arg580, i32 0, i32 -2147483648
  %i3689 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg1208, i32 %arg59, i32 0, i32 0)
  %i3690 = select i1 %arg580, i32 0, i32 -2147483648
  %i3691 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg1208, i32 %arg71, i32 0, i32 0)
  %i3692 = select i1 %arg580, i32 0, i32 -2147483648
  %i3693 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg1208, i32 %arg19, i32 0, i32 0)
  %i3694 = select i1 %arg580, i32 0, i32 -2147483648
  %i3695 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i3696 = select i1 %arg580, i32 0, i32 -2147483648
  %i3697 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i3698 = select i1 %arg580, i32 0, i32 -2147483648
  %i3699 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg1208, i32 %arg95, i32 0, i32 0)
  %i3700 = select i1 %arg580, i32 0, i32 -2147483648
  %i3701 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg1208, i32 %arg96, i32 0, i32 0)
  %i3702 = select i1 %arg580, i32 0, i32 -2147483648
  %i3703 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg1208, i32 %arg95, i32 0, i32 0)
  %i3704 = select i1 %arg580, i32 0, i32 -2147483648
  %i3705 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg34, i32 %arg4, i32 0, i32 0)
  %i3706 = select i1 true, i32 0, i32 1
  %i3707 = tail call i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) %arg1208, i32 1, i32 0, i32 0)
  %i3708 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %i3709 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %i3710 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %i3711 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> zeroinitializer, <8 x half> zeroinitializer, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %i3712 = shufflevector <4 x float> zeroinitializer, <4 x float> zeroinitializer, <2 x i32> <i32 0, i32 1>
  %i3713 = fptrunc <2 x float> zeroinitializer to <2 x half>
  %i3714 = shufflevector <2 x half> zeroinitializer, <2 x half> zeroinitializer, <2 x i32> <i32 0, i32 2>
  store <2 x half> zeroinitializer, ptr addrspace(3) null, align 4
  %.idx46 = shl i32 0, 0
  %i3715 = getelementptr i8, ptr addrspace(3) %global_smem, i32 %.idx
  %i3716 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1300, i32 %arg856
  store i16 %arg1301, ptr addrspace(3) %arg1302, align 2
  store i16 %arg1303, ptr addrspace(3) null, align 2
  %i3717 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1300, i32 %arg1076
  store i16 %arg1304, ptr addrspace(3) %arg1305, align 2
  store i16 %arg1306, ptr addrspace(3) %arg1300, align 2
  store i16 %arg1307, ptr addrspace(3) null, align 2
  %i3718 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1300, i32 %arg1077
  store i16 %arg1308, ptr addrspace(3) %arg1309, align 2
  store i16 %arg1310, ptr addrspace(3) %arg1300, align 2
  store i16 %arg1311, ptr addrspace(3) null, align 2
  %i3719 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1300, i32 %arg1312
  store i16 0, ptr addrspace(3) %arg1313, align 2
  %i3720 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1300, i32 %arg1314
  store i16 %arg1315, ptr addrspace(3) %arg1316, align 2
  %i3721 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1300, i32 %arg1317
  store i16 %arg1318, ptr addrspace(3) %arg1319, align 2
  store i16 %arg1320, ptr addrspace(3) null, align 2
  %i3722 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1300, i32 %arg1321
  store i16 %arg1322, ptr addrspace(3) %arg1323, align 2
  %i3723 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1300, i32 %arg1079
  store i16 %arg1324, ptr addrspace(3) %arg1325, align 2
  %i3724 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1300, i32 %arg1326
  store i16 %arg1327, ptr addrspace(3) null, align 2
  %i3725 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1300, i32 %arg1082
  store i16 1, ptr addrspace(3) %arg1328, align 2
  %i3726 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1300, i32 %arg832
  store i16 %arg1329, ptr addrspace(3) %arg1330, align 2
  store i16 %arg1331, ptr addrspace(3) null, align 2
  %i3727 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1300, i32 %arg1332
  store i16 %i3583, ptr addrspace(3) %i3727, align 2
  store i16 %i3593, ptr addrspace(3) %i3715, align 2
  store i16 %i3601, ptr addrspace(3) null, align 2
  store i16 %i3611, ptr addrspace(3) %arg1300, align 2
  store i16 %arg1333, ptr addrspace(3) null, align 2
  store i16 %arg1334, ptr addrspace(3) %arg1300, align 2
  %i3728 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1300, i32 %i2716
  store i16 %i3637, ptr addrspace(3) %i3728, align 2
  store i16 0, ptr addrspace(3) null, align 2
  %i3729 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1300, i32 %i2719
  store i16 %i3659, ptr addrspace(3) %i3729, align 2
  store i16 %i3669, ptr addrspace(3) %arg1300, align 2
  store i16 %arg1335, ptr addrspace(3) null, align 2
  %i3730 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1300, i32 %arg1085
  store i16 %i3689, ptr addrspace(3) %arg1336, align 2
  %i3731 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1300, i32 %i2724
  store i16 %i3699, ptr addrspace(3) %i3731, align 2
  store i16 0, ptr addrspace(3) null, align 2
  %i3732 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1300, i32 %i2748
  store i16 %i3649, ptr addrspace(3) %i3732, align 2
  store i16 %i3661, ptr addrspace(3) null, align 2
  %i3733 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1300, i32 %i2749
  store i16 %i3671, ptr addrspace(3) %i3733, align 2
  %i3734 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1300, i32 %i2751
  store i16 %i3681, ptr addrspace(3) %i3734, align 2
  store i16 %i3691, ptr addrspace(3) null, align 2
  %i3735 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1300, i32 %i2755
  store i16 %i3701, ptr addrspace(3) %i3735, align 2
  store i16 %i3565, ptr addrspace(3) null, align 2
  store i16 %i3575, ptr addrspace(3) %arg1300, align 2
  %i3736 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1300, i32 %arg1337
  store i16 %i3585, ptr addrspace(3) %arg1338, align 2
  store i16 %i3595, ptr addrspace(3) null, align 2
  %i3737 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1300, i32 %arg1339
  store i16 %i3603, ptr addrspace(3) %arg1340, align 2
  store i16 %i3613, ptr addrspace(3) %arg1300, align 2
  store i16 %i3623, ptr addrspace(3) null, align 2
  store i16 %i3631, ptr addrspace(3) %arg1300, align 2
  store i16 %i3641, ptr addrspace(3) null, align 2
  %i3738 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1300, i32 %arg1103
  store i16 %i3651, ptr addrspace(3) %i3738, align 2
  store i16 %arg1341, ptr addrspace(3) %arg1300, align 2
  store i16 0, ptr addrspace(3) null, align 2
  %i3739 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1300, i32 %arg1342
  store i16 %i3683, ptr addrspace(3) null, align 2
  store i16 %i3693, ptr addrspace(3) %arg1300, align 2
  store i16 %i3703, ptr addrspace(3) null, align 2
  %i3740 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1300, i32 %i2771
  store i16 %i3567, ptr addrspace(3) %i3740, align 2
  store i16 %i3577, ptr addrspace(3) null, align 2
  %i3741 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1300, i32 %i2773
  store i16 %i3587, ptr addrspace(3) %i3741, align 2
  store i16 %i3597, ptr addrspace(3) %arg1300, align 2
  store i16 %i3605, ptr addrspace(3) null, align 2
  store i16 %arg1343, ptr addrspace(3) %arg1300, align 2
  store i16 %i3625, ptr addrspace(3) null, align 2
  store i16 %i3633, ptr addrspace(3) %arg1300, align 2
  %i3742 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1300, i32 %i2774
  store i16 %i3643, ptr addrspace(3) %i3742, align 2
  store i16 %i3653, ptr addrspace(3) null, align 2
  store i16 %i3665, ptr addrspace(3) %arg1300, align 2
  %i3743 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1300, i32 %i2776
  store i16 %i3675, ptr addrspace(3) %i3743, align 2
  store i16 %arg1344, ptr addrspace(3) null, align 2
  %i3744 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1300, i32 %arg849
  store i16 %arg1345, ptr addrspace(3) %i3744, align 2
  %i3745 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1300, i32 %arg1346
  store i16 %arg1347, ptr addrspace(3) null, align 2
  %i3746 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1300, i32 %i2782
  store i16 0, ptr addrspace(3) %i3746, align 2
  %i3747 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1300, i32 %i2784
  store i16 %i3569, ptr addrspace(3) %i3747, align 2
  store i16 %i3579, ptr addrspace(3) null, align 2
  store i16 %arg1348, ptr addrspace(3) %arg1300, align 2
  store i16 %i3607, ptr addrspace(3) null, align 2
  store i16 %i3617, ptr addrspace(3) %arg1300, align 2
  %i3748 = getelementptr inbounds nuw i8, ptr addrspace(3) %arg1300, i32 %arg867
  store i16 %arg1349, ptr addrspace(3) null, align 2
  br label %bb3333

._crit_edge924:                                   ; preds = %._crit_edge._crit_edge
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) readnone, i16, i64, i32) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #4

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.waitcnt(i32 immarg) #5

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #6

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half>, <8 x half>, <4 x float>, i32 immarg, i32 immarg, i32 immarg) #7

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #3

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(argmem: read)
declare <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) captures(none)) #8

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.update.dpp.i32(i32, i32, i32 immarg, i32 immarg, i32 immarg, i1 immarg) #9

declare i32 @llvm.smax.i32(i32, i32) #3

attributes #0 = { nofree norecurse nounwind "amdgpu-agpr-alloc"="0" "amdgpu-cluster-dims"="1,1,1" "amdgpu-flat-work-group-size"="1,512" "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-waves-per-eu"="3, 3"  "uniform-work-group-size" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #5 = { nocallback nofree nounwind willreturn }
attributes #6 = { convergent nocallback nofree nounwind willreturn }
attributes #7 = { convergent nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #8 = { convergent nocallback nofree nounwind willreturn memory(argmem: read) }
attributes #9 = { convergent nocallback nofree nounwind willreturn memory(none) }
