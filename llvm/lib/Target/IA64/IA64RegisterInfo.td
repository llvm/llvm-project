//===- IA64RegisterInfo.td - Describe the IA64 Register File ----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file describes the IA64 register file, defining the registers
// themselves, aliases between the registers, and the register classes built
// out of the registers.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Register definitions...
//

class IA64Register<string n> : Register<n> {
    let Namespace = "IA64";
}

// GR - One of 128 32-bit general registers
class GR<bits<7> num, string n> : IA64Register<n> {
    field bits<7> Num = num;
}

// FP - One of 128 82-bit floating-point registers
class FP<bits<7> num, string n> : IA64Register<n> {
    field bits<7> Num = num;
}

// PR - One of 64 1-bit predicate registers
class PR<bits<6> num, string n> : IA64Register<n> {
    field bits<6> Num = num;
}

/* general registers */
def r0 : GR< 0, "r0">, DwarfRegNum<[0]>;
def r1 : GR< 1, "r1">, DwarfRegNum<[1]>;
def r2 : GR< 2, "r2">, DwarfRegNum<[2]>;
def r3 : GR< 3, "r3">, DwarfRegNum<[3]>;
def r4 : GR< 4, "r4">, DwarfRegNum<[4]>;
def r5 : GR< 5, "r5">, DwarfRegNum<[5]>;
def r6 : GR< 6, "r6">, DwarfRegNum<[6]>;
def r7 : GR< 7, "r7">, DwarfRegNum<[7]>;
def r8 : GR< 8, "r8">, DwarfRegNum<[8]>;
def r9 : GR< 9, "r9">, DwarfRegNum<[9]>;
def r10 : GR< 10, "r10">, DwarfRegNum<[10]>;
def r11 : GR< 11, "r11">, DwarfRegNum<[11]>;
def r12 : GR< 12, "r12">, DwarfRegNum<[12]>;
def r13 : GR< 13, "r13">, DwarfRegNum<[13]>;
def r14 : GR< 14, "r14">, DwarfRegNum<[14]>;
def r15 : GR< 15, "r15">, DwarfRegNum<[15]>;
def r16 : GR< 16, "r16">, DwarfRegNum<[16]>;
def r17 : GR< 17, "r17">, DwarfRegNum<[17]>;
def r18 : GR< 18, "r18">, DwarfRegNum<[18]>;
def r19 : GR< 19, "r19">, DwarfRegNum<[19]>;
def r20 : GR< 20, "r20">, DwarfRegNum<[20]>;
def r21 : GR< 21, "r21">, DwarfRegNum<[21]>;
def r22 : GR< 22, "r22">, DwarfRegNum<[22]>;
def r23 : GR< 23, "r23">, DwarfRegNum<[23]>;
def r24 : GR< 24, "r24">, DwarfRegNum<[24]>;
def r25 : GR< 25, "r25">, DwarfRegNum<[25]>;
def r26 : GR< 26, "r26">, DwarfRegNum<[26]>;
def r27 : GR< 27, "r27">, DwarfRegNum<[27]>;
def r28 : GR< 28, "r28">, DwarfRegNum<[28]>;
def r29 : GR< 29, "r29">, DwarfRegNum<[29]>;
def r30 : GR< 30, "r30">, DwarfRegNum<[30]>;
def r31 : GR< 31, "r31">, DwarfRegNum<[31]>;
def r32 : GR< 32, "r32">, DwarfRegNum<[32]>;
def r33 : GR< 33, "r33">, DwarfRegNum<[33]>;
def r34 : GR< 34, "r34">, DwarfRegNum<[34]>;
def r35 : GR< 35, "r35">, DwarfRegNum<[35]>;
def r36 : GR< 36, "r36">, DwarfRegNum<[36]>;
def r37 : GR< 37, "r37">, DwarfRegNum<[37]>;
def r38 : GR< 38, "r38">, DwarfRegNum<[38]>;
def r39 : GR< 39, "r39">, DwarfRegNum<[39]>;
def r40 : GR< 40, "r40">, DwarfRegNum<[40]>;
def r41 : GR< 41, "r41">, DwarfRegNum<[41]>;
def r42 : GR< 42, "r42">, DwarfRegNum<[42]>;
def r43 : GR< 43, "r43">, DwarfRegNum<[43]>;
def r44 : GR< 44, "r44">, DwarfRegNum<[44]>;
def r45 : GR< 45, "r45">, DwarfRegNum<[45]>;
def r46 : GR< 46, "r46">, DwarfRegNum<[46]>;
def r47 : GR< 47, "r47">, DwarfRegNum<[47]>;
def r48 : GR< 48, "r48">, DwarfRegNum<[48]>;
def r49 : GR< 49, "r49">, DwarfRegNum<[49]>;
def r50 : GR< 50, "r50">, DwarfRegNum<[50]>;
def r51 : GR< 51, "r51">, DwarfRegNum<[51]>;
def r52 : GR< 52, "r52">, DwarfRegNum<[52]>;
def r53 : GR< 53, "r53">, DwarfRegNum<[53]>;
def r54 : GR< 54, "r54">, DwarfRegNum<[54]>;
def r55 : GR< 55, "r55">, DwarfRegNum<[55]>;
def r56 : GR< 56, "r56">, DwarfRegNum<[56]>;
def r57 : GR< 57, "r57">, DwarfRegNum<[57]>;
def r58 : GR< 58, "r58">, DwarfRegNum<[58]>;
def r59 : GR< 59, "r59">, DwarfRegNum<[59]>;
def r60 : GR< 60, "r60">, DwarfRegNum<[60]>;
def r61 : GR< 61, "r61">, DwarfRegNum<[61]>;
def r62 : GR< 62, "r62">, DwarfRegNum<[62]>;
def r63 : GR< 63, "r63">, DwarfRegNum<[63]>;
def r64 : GR< 64, "r64">, DwarfRegNum<[64]>;
def r65 : GR< 65, "r65">, DwarfRegNum<[65]>;
def r66 : GR< 66, "r66">, DwarfRegNum<[66]>;
def r67 : GR< 67, "r67">, DwarfRegNum<[67]>;
def r68 : GR< 68, "r68">, DwarfRegNum<[68]>;
def r69 : GR< 69, "r69">, DwarfRegNum<[69]>;
def r70 : GR< 70, "r70">, DwarfRegNum<[70]>;
def r71 : GR< 71, "r71">, DwarfRegNum<[71]>;
def r72 : GR< 72, "r72">, DwarfRegNum<[72]>;
def r73 : GR< 73, "r73">, DwarfRegNum<[73]>;
def r74 : GR< 74, "r74">, DwarfRegNum<[74]>;
def r75 : GR< 75, "r75">, DwarfRegNum<[75]>;
def r76 : GR< 76, "r76">, DwarfRegNum<[76]>;
def r77 : GR< 77, "r77">, DwarfRegNum<[77]>;
def r78 : GR< 78, "r78">, DwarfRegNum<[78]>;
def r79 : GR< 79, "r79">, DwarfRegNum<[79]>;
def r80 : GR< 80, "r80">, DwarfRegNum<[80]>;
def r81 : GR< 81, "r81">, DwarfRegNum<[81]>;
def r82 : GR< 82, "r82">, DwarfRegNum<[82]>;
def r83 : GR< 83, "r83">, DwarfRegNum<[83]>;
def r84 : GR< 84, "r84">, DwarfRegNum<[84]>;
def r85 : GR< 85, "r85">, DwarfRegNum<[85]>;
def r86 : GR< 86, "r86">, DwarfRegNum<[86]>;
def r87 : GR< 87, "r87">, DwarfRegNum<[87]>;
def r88 : GR< 88, "r88">, DwarfRegNum<[88]>;
def r89 : GR< 89, "r89">, DwarfRegNum<[89]>;
def r90 : GR< 90, "r90">, DwarfRegNum<[90]>;
def r91 : GR< 91, "r91">, DwarfRegNum<[91]>;
def r92 : GR< 92, "r92">, DwarfRegNum<[92]>;
def r93 : GR< 93, "r93">, DwarfRegNum<[93]>;
def r94 : GR< 94, "r94">, DwarfRegNum<[94]>;
def r95 : GR< 95, "r95">, DwarfRegNum<[95]>;
def r96 : GR< 96, "r96">, DwarfRegNum<[96]>;
def r97 : GR< 97, "r97">, DwarfRegNum<[97]>;
def r98 : GR< 98, "r98">, DwarfRegNum<[98]>;
def r99 : GR< 99, "r99">, DwarfRegNum<[99]>;
def r100 : GR< 100, "r100">, DwarfRegNum<[100]>;
def r101 : GR< 101, "r101">, DwarfRegNum<[101]>;
def r102 : GR< 102, "r102">, DwarfRegNum<[102]>;
def r103 : GR< 103, "r103">, DwarfRegNum<[103]>;
def r104 : GR< 104, "r104">, DwarfRegNum<[104]>;
def r105 : GR< 105, "r105">, DwarfRegNum<[105]>;
def r106 : GR< 106, "r106">, DwarfRegNum<[106]>;
def r107 : GR< 107, "r107">, DwarfRegNum<[107]>;
def r108 : GR< 108, "r108">, DwarfRegNum<[108]>;
def r109 : GR< 109, "r109">, DwarfRegNum<[109]>;
def r110 : GR< 110, "r110">, DwarfRegNum<[110]>;
def r111 : GR< 111, "r111">, DwarfRegNum<[111]>;
def r112 : GR< 112, "r112">, DwarfRegNum<[112]>;
def r113 : GR< 113, "r113">, DwarfRegNum<[113]>;
def r114 : GR< 114, "r114">, DwarfRegNum<[114]>;
def r115 : GR< 115, "r115">, DwarfRegNum<[115]>;
def r116 : GR< 116, "r116">, DwarfRegNum<[116]>;
def r117 : GR< 117, "r117">, DwarfRegNum<[117]>;
def r118 : GR< 118, "r118">, DwarfRegNum<[118]>;
def r119 : GR< 119, "r119">, DwarfRegNum<[119]>;
def r120 : GR< 120, "r120">, DwarfRegNum<[120]>;
def r121 : GR< 121, "r121">, DwarfRegNum<[121]>;
def r122 : GR< 122, "r122">, DwarfRegNum<[122]>;
def r123 : GR< 123, "r123">, DwarfRegNum<[123]>;
def r124 : GR< 124, "r124">, DwarfRegNum<[124]>;
def r125 : GR< 125, "r125">, DwarfRegNum<[125]>;
def r126 : GR< 126, "r126">, DwarfRegNum<[126]>;
def r127 : GR< 127, "r127">, DwarfRegNum<[127]>;

/* floating-point registers */
def F0 : FP< 0, "f0">, DwarfRegNum<[128]>;
def F1 : FP< 1, "f1">, DwarfRegNum<[129]>;
def F2 : FP< 2, "f2">, DwarfRegNum<[130]>;
def F3 : FP< 3, "f3">, DwarfRegNum<[131]>;
def F4 : FP< 4, "f4">, DwarfRegNum<[132]>;
def F5 : FP< 5, "f5">, DwarfRegNum<[133]>;
def F6 : FP< 6, "f6">, DwarfRegNum<[134]>;
def F7 : FP< 7, "f7">, DwarfRegNum<[135]>;
def F8 : FP< 8, "f8">, DwarfRegNum<[136]>;
def F9 : FP< 9, "f9">, DwarfRegNum<[137]>;
def F10 : FP< 10, "f10">, DwarfRegNum<[138]>;
def F11 : FP< 11, "f11">, DwarfRegNum<[139]>;
def F12 : FP< 12, "f12">, DwarfRegNum<[140]>;
def F13 : FP< 13, "f13">, DwarfRegNum<[141]>;
def F14 : FP< 14, "f14">, DwarfRegNum<[142]>;
def F15 : FP< 15, "f15">, DwarfRegNum<[143]>;
def F16 : FP< 16, "f16">, DwarfRegNum<[144]>;
def F17 : FP< 17, "f17">, DwarfRegNum<[145]>;
def F18 : FP< 18, "f18">, DwarfRegNum<[146]>;
def F19 : FP< 19, "f19">, DwarfRegNum<[147]>;
def F20 : FP< 20, "f20">, DwarfRegNum<[148]>;
def F21 : FP< 21, "f21">, DwarfRegNum<[149]>;
def F22 : FP< 22, "f22">, DwarfRegNum<[150]>;
def F23 : FP< 23, "f23">, DwarfRegNum<[151]>;
def F24 : FP< 24, "f24">, DwarfRegNum<[152]>;
def F25 : FP< 25, "f25">, DwarfRegNum<[153]>;
def F26 : FP< 26, "f26">, DwarfRegNum<[154]>;
def F27 : FP< 27, "f27">, DwarfRegNum<[155]>;
def F28 : FP< 28, "f28">, DwarfRegNum<[156]>;
def F29 : FP< 29, "f29">, DwarfRegNum<[157]>;
def F30 : FP< 30, "f30">, DwarfRegNum<[158]>;
def F31 : FP< 31, "f31">, DwarfRegNum<[159]>;
def F32 : FP< 32, "f32">, DwarfRegNum<[160]>;
def F33 : FP< 33, "f33">, DwarfRegNum<[161]>;
def F34 : FP< 34, "f34">, DwarfRegNum<[162]>;
def F35 : FP< 35, "f35">, DwarfRegNum<[163]>;
def F36 : FP< 36, "f36">, DwarfRegNum<[164]>;
def F37 : FP< 37, "f37">, DwarfRegNum<[165]>;
def F38 : FP< 38, "f38">, DwarfRegNum<[166]>;
def F39 : FP< 39, "f39">, DwarfRegNum<[167]>;
def F40 : FP< 40, "f40">, DwarfRegNum<[168]>;
def F41 : FP< 41, "f41">, DwarfRegNum<[169]>;
def F42 : FP< 42, "f42">, DwarfRegNum<[170]>;
def F43 : FP< 43, "f43">, DwarfRegNum<[171]>;
def F44 : FP< 44, "f44">, DwarfRegNum<[172]>;
def F45 : FP< 45, "f45">, DwarfRegNum<[173]>;
def F46 : FP< 46, "f46">, DwarfRegNum<[174]>;
def F47 : FP< 47, "f47">, DwarfRegNum<[175]>;
def F48 : FP< 48, "f48">, DwarfRegNum<[176]>;
def F49 : FP< 49, "f49">, DwarfRegNum<[177]>;
def F50 : FP< 50, "f50">, DwarfRegNum<[178]>;
def F51 : FP< 51, "f51">, DwarfRegNum<[179]>;
def F52 : FP< 52, "f52">, DwarfRegNum<[180]>;
def F53 : FP< 53, "f53">, DwarfRegNum<[181]>;
def F54 : FP< 54, "f54">, DwarfRegNum<[182]>;
def F55 : FP< 55, "f55">, DwarfRegNum<[183]>;
def F56 : FP< 56, "f56">, DwarfRegNum<[184]>;
def F57 : FP< 57, "f57">, DwarfRegNum<[185]>;
def F58 : FP< 58, "f58">, DwarfRegNum<[186]>;
def F59 : FP< 59, "f59">, DwarfRegNum<[187]>;
def F60 : FP< 60, "f60">, DwarfRegNum<[188]>;
def F61 : FP< 61, "f61">, DwarfRegNum<[189]>;
def F62 : FP< 62, "f62">, DwarfRegNum<[190]>;
def F63 : FP< 63, "f63">, DwarfRegNum<[191]>;
def F64 : FP< 64, "f64">, DwarfRegNum<[192]>;
def F65 : FP< 65, "f65">, DwarfRegNum<[193]>;
def F66 : FP< 66, "f66">, DwarfRegNum<[194]>;
def F67 : FP< 67, "f67">, DwarfRegNum<[195]>;
def F68 : FP< 68, "f68">, DwarfRegNum<[196]>;
def F69 : FP< 69, "f69">, DwarfRegNum<[197]>;
def F70 : FP< 70, "f70">, DwarfRegNum<[198]>;
def F71 : FP< 71, "f71">, DwarfRegNum<[199]>;
def F72 : FP< 72, "f72">, DwarfRegNum<[200]>;
def F73 : FP< 73, "f73">, DwarfRegNum<[201]>;
def F74 : FP< 74, "f74">, DwarfRegNum<[202]>;
def F75 : FP< 75, "f75">, DwarfRegNum<[203]>;
def F76 : FP< 76, "f76">, DwarfRegNum<[204]>;
def F77 : FP< 77, "f77">, DwarfRegNum<[205]>;
def F78 : FP< 78, "f78">, DwarfRegNum<[206]>;
def F79 : FP< 79, "f79">, DwarfRegNum<[207]>;
def F80 : FP< 80, "f80">, DwarfRegNum<[208]>;
def F81 : FP< 81, "f81">, DwarfRegNum<[209]>;
def F82 : FP< 82, "f82">, DwarfRegNum<[210]>;
def F83 : FP< 83, "f83">, DwarfRegNum<[211]>;
def F84 : FP< 84, "f84">, DwarfRegNum<[212]>;
def F85 : FP< 85, "f85">, DwarfRegNum<[213]>;
def F86 : FP< 86, "f86">, DwarfRegNum<[214]>;
def F87 : FP< 87, "f87">, DwarfRegNum<[215]>;
def F88 : FP< 88, "f88">, DwarfRegNum<[216]>;
def F89 : FP< 89, "f89">, DwarfRegNum<[217]>;
def F90 : FP< 90, "f90">, DwarfRegNum<[218]>;
def F91 : FP< 91, "f91">, DwarfRegNum<[219]>;
def F92 : FP< 92, "f92">, DwarfRegNum<[220]>;
def F93 : FP< 93, "f93">, DwarfRegNum<[221]>;
def F94 : FP< 94, "f94">, DwarfRegNum<[222]>;
def F95 : FP< 95, "f95">, DwarfRegNum<[223]>;
def F96 : FP< 96, "f96">, DwarfRegNum<[224]>;
def F97 : FP< 97, "f97">, DwarfRegNum<[225]>;
def F98 : FP< 98, "f98">, DwarfRegNum<[226]>;
def F99 : FP< 99, "f99">, DwarfRegNum<[227]>;
def F100 : FP< 100, "f100">, DwarfRegNum<[228]>;
def F101 : FP< 101, "f101">, DwarfRegNum<[229]>;
def F102 : FP< 102, "f102">, DwarfRegNum<[230]>;
def F103 : FP< 103, "f103">, DwarfRegNum<[231]>;
def F104 : FP< 104, "f104">, DwarfRegNum<[232]>;
def F105 : FP< 105, "f105">, DwarfRegNum<[233]>;
def F106 : FP< 106, "f106">, DwarfRegNum<[234]>;
def F107 : FP< 107, "f107">, DwarfRegNum<[235]>;
def F108 : FP< 108, "f108">, DwarfRegNum<[236]>;
def F109 : FP< 109, "f109">, DwarfRegNum<[237]>;
def F110 : FP< 110, "f110">, DwarfRegNum<[238]>;
def F111 : FP< 111, "f111">, DwarfRegNum<[239]>;
def F112 : FP< 112, "f112">, DwarfRegNum<[240]>;
def F113 : FP< 113, "f113">, DwarfRegNum<[241]>;
def F114 : FP< 114, "f114">, DwarfRegNum<[242]>;
def F115 : FP< 115, "f115">, DwarfRegNum<[243]>;
def F116 : FP< 116, "f116">, DwarfRegNum<[244]>;
def F117 : FP< 117, "f117">, DwarfRegNum<[245]>;
def F118 : FP< 118, "f118">, DwarfRegNum<[246]>;
def F119 : FP< 119, "f119">, DwarfRegNum<[247]>;
def F120 : FP< 120, "f120">, DwarfRegNum<[248]>;
def F121 : FP< 121, "f121">, DwarfRegNum<[249]>;
def F122 : FP< 122, "f122">, DwarfRegNum<[250]>;
def F123 : FP< 123, "f123">, DwarfRegNum<[251]>;
def F124 : FP< 124, "f124">, DwarfRegNum<[252]>;
def F125 : FP< 125, "f125">, DwarfRegNum<[253]>;
def F126 : FP< 126, "f126">, DwarfRegNum<[254]>;
def F127 : FP< 127, "f127">, DwarfRegNum<[255]>;

/* predicate registers */
def p0 : PR< 0, "p0">, DwarfRegNum<[256]>;
def p1 : PR< 1, "p1">, DwarfRegNum<[257]>;
def p2 : PR< 2, "p2">, DwarfRegNum<[258]>;
def p3 : PR< 3, "p3">, DwarfRegNum<[259]>;
def p4 : PR< 4, "p4">, DwarfRegNum<[260]>;
def p5 : PR< 5, "p5">, DwarfRegNum<[261]>;
def p6 : PR< 6, "p6">, DwarfRegNum<[262]>;
def p7 : PR< 7, "p7">, DwarfRegNum<[263]>;
def p8 : PR< 8, "p8">, DwarfRegNum<[264]>;
def p9 : PR< 9, "p9">, DwarfRegNum<[265]>;
def p10 : PR< 10, "p10">, DwarfRegNum<[266]>;
def p11 : PR< 11, "p11">, DwarfRegNum<[267]>;
def p12 : PR< 12, "p12">, DwarfRegNum<[268]>;
def p13 : PR< 13, "p13">, DwarfRegNum<[269]>;
def p14 : PR< 14, "p14">, DwarfRegNum<[270]>;
def p15 : PR< 15, "p15">, DwarfRegNum<[271]>;
def p16 : PR< 16, "p16">, DwarfRegNum<[272]>;
def p17 : PR< 17, "p17">, DwarfRegNum<[273]>;
def p18 : PR< 18, "p18">, DwarfRegNum<[274]>;
def p19 : PR< 19, "p19">, DwarfRegNum<[275]>;
def p20 : PR< 20, "p20">, DwarfRegNum<[276]>;
def p21 : PR< 21, "p21">, DwarfRegNum<[277]>;
def p22 : PR< 22, "p22">, DwarfRegNum<[278]>;
def p23 : PR< 23, "p23">, DwarfRegNum<[279]>;
def p24 : PR< 24, "p24">, DwarfRegNum<[280]>;
def p25 : PR< 25, "p25">, DwarfRegNum<[281]>;
def p26 : PR< 26, "p26">, DwarfRegNum<[282]>;
def p27 : PR< 27, "p27">, DwarfRegNum<[283]>;
def p28 : PR< 28, "p28">, DwarfRegNum<[284]>;
def p29 : PR< 29, "p29">, DwarfRegNum<[285]>;
def p30 : PR< 30, "p30">, DwarfRegNum<[286]>;
def p31 : PR< 31, "p31">, DwarfRegNum<[287]>;
def p32 : PR< 32, "p32">, DwarfRegNum<[288]>;
def p33 : PR< 33, "p33">, DwarfRegNum<[289]>;
def p34 : PR< 34, "p34">, DwarfRegNum<[290]>;
def p35 : PR< 35, "p35">, DwarfRegNum<[291]>;
def p36 : PR< 36, "p36">, DwarfRegNum<[292]>;
def p37 : PR< 37, "p37">, DwarfRegNum<[293]>;
def p38 : PR< 38, "p38">, DwarfRegNum<[294]>;
def p39 : PR< 39, "p39">, DwarfRegNum<[295]>;
def p40 : PR< 40, "p40">, DwarfRegNum<[296]>;
def p41 : PR< 41, "p41">, DwarfRegNum<[297]>;
def p42 : PR< 42, "p42">, DwarfRegNum<[298]>;
def p43 : PR< 43, "p43">, DwarfRegNum<[299]>;
def p44 : PR< 44, "p44">, DwarfRegNum<[300]>;
def p45 : PR< 45, "p45">, DwarfRegNum<[301]>;
def p46 : PR< 46, "p46">, DwarfRegNum<[302]>;
def p47 : PR< 47, "p47">, DwarfRegNum<[303]>;
def p48 : PR< 48, "p48">, DwarfRegNum<[304]>;
def p49 : PR< 49, "p49">, DwarfRegNum<[305]>;
def p50 : PR< 50, "p50">, DwarfRegNum<[306]>;
def p51 : PR< 51, "p51">, DwarfRegNum<[307]>;
def p52 : PR< 52, "p52">, DwarfRegNum<[308]>;
def p53 : PR< 53, "p53">, DwarfRegNum<[309]>;
def p54 : PR< 54, "p54">, DwarfRegNum<[310]>;
def p55 : PR< 55, "p55">, DwarfRegNum<[311]>;
def p56 : PR< 56, "p56">, DwarfRegNum<[312]>;
def p57 : PR< 57, "p57">, DwarfRegNum<[313]>;
def p58 : PR< 58, "p58">, DwarfRegNum<[314]>;
def p59 : PR< 59, "p59">, DwarfRegNum<[315]>;
def p60 : PR< 60, "p60">, DwarfRegNum<[316]>;
def p61 : PR< 61, "p61">, DwarfRegNum<[317]>;
def p62 : PR< 62, "p62">, DwarfRegNum<[318]>;
def p63 : PR< 63, "p63">, DwarfRegNum<[319]>;

// XXX : this is temporary, we'll eventually have the output registers
// in the general purpose register class too?
def out0 : GR<0, "out0">, DwarfRegNum<[120]>;
def out1 : GR<1, "out1">, DwarfRegNum<[121]>;
def out2 : GR<2, "out2">, DwarfRegNum<[122]>;
def out3 : GR<3, "out3">, DwarfRegNum<[123]>;
def out4 : GR<4, "out4">, DwarfRegNum<[124]>;
def out5 : GR<5, "out5">, DwarfRegNum<[125]>;
def out6 : GR<6, "out6">, DwarfRegNum<[126]>;
def out7 : GR<7, "out7">, DwarfRegNum<[127]>;

// application (special) registers:

// "previous function state" application register
def AR_PFS : GR<0, "ar.pfs">, DwarfRegNum<[331]>;

// "return pointer" (this is really branch register b0)
def rp : GR<0, "rp">, DwarfRegNum<[-1]>;

// branch reg 6
def B6 : GR<0, "b6">, DwarfRegNum<[326]>;

//===----------------------------------------------------------------------===//
// Register Class Definitions... now that we have all of the pieces, define the
// top-level register classes.  The order specified in the register list is
// implicitly defined to be the register allocation order.
//

// these are the scratch (+stacked) general registers
// FIXME/XXX  we also reserve a frame pointer (r5)
// FIXME/XXX  we also reserve r2 for spilling/filling predicates
// in IA64RegisterInfo.cpp
// FIXME/XXX  we also reserve r22 for calculating addresses
// in IA64RegisterInfo.cpp

def GR : RegisterClass<"IA64", [i64], 64, 
       [
       
//FIXME!: for both readability and performance, we don't want the out
//        registers to be the first ones allocated

        out7, out6, out5, out4, out3, out2, out1, out0,
        r3,  r8,  r9,  r10, r11, r14, r15,
        r16, r17, r18, r19, r20, r21, r23,
        r24, r25, r26, r27, r28, r29, r30, r31,
        r32, r33, r34, r35, r36, r37, r38, r39,
        r40, r41, r42, r43, r44, r45, r46, r47,
        r48, r49, r50, r51, r52, r53, r54, r55,
        r56, r57, r58, r59, r60, r61, r62, r63,
        r64, r65, r66, r67, r68, r69, r70, r71,
        r72, r73, r74, r75, r76, r77, r78, r79,
        r80, r81, r82, r83, r84, r85, r86, r87,
        r88, r89, r90, r91, r92, r93, r94, r95,
        r96, r97, r98, r99, r100, r101, r102, r103,
        r104, r105, r106, r107, r108, r109, r110, r111,
        r112, r113, r114, r115, r116, r117, r118, r119,
        // last 17 are special (look down)
        r120, r121, r122, r123, r124, r125, r126, r127,
        r0, r1, r2, r5, r12, r13, r22, rp, AR_PFS]>
  {
    let MethodProtos = [{
    iterator allocation_order_begin(const MachineFunction &MF) const;
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    GRClass::iterator
    GRClass::allocation_order_begin(const MachineFunction &MF) const {
        // hide the 8 out? registers appropriately:
        return begin()+(8-(MF.getInfo<IA64FunctionInfo>()->outRegsUsed));
      }

      GRClass::iterator
      GRClass::allocation_order_end(const MachineFunction &MF) const {
        // the 9 special registers r0,r1,r2,r5,r12,r13 etc
        int numReservedRegs=9;
        
        // we also can't allocate registers for use as locals if they're already
        // required as 'out' registers
        numReservedRegs+=MF.getInfo<IA64FunctionInfo>()->outRegsUsed;
        return end()-numReservedRegs; // hide registers appropriately
      }
  }];
}


// these are the scratch (+stacked) FP registers

def FP : RegisterClass<"IA64", [f64], 64, 
       [F6, F7, 
        F8, F9, F10, F11, F12, F13, F14, F15, 
        F32, F33, F34, F35, F36, F37, F38, F39, 
        F40, F41, F42, F43, F44, F45, F46, F47, 
        F48, F49, F50, F51, F52, F53, F54, F55, 
        F56, F57, F58, F59, F60, F61, F62, F63, 
        F64, F65, F66, F67, F68, F69, F70, F71, 
        F72, F73, F74, F75, F76, F77, F78, F79, 
        F80, F81, F82, F83, F84, F85, F86, F87, 
        F88, F89, F90, F91, F92, F93, F94, F95, 
        F96, F97, F98, F99, F100, F101, F102, F103, 
        F104, F105, F106, F107, F108, F109, F110, F111, 
        F112, F113, F114, F115, F116, F117, F118, F119, 
        F120, F121, F122, F123, F124, F125, F126, F127,
        F0, F1]> // these last two are hidden
  {
// the 128s here are to make stf.spill/ldf.fill happy,
// when storing full (82-bit) FP regs to stack slots
// we need to 16-byte align
    let Size=128;
    let Alignment=128;

    let MethodProtos = [{
    iterator allocation_order_begin(const MachineFunction &MF) const;
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    FPClass::iterator
    FPClass::allocation_order_begin(const MachineFunction &MF) const {
      return begin(); // we don't hide any FP regs from the start
    }

    FPClass::iterator
    FPClass::allocation_order_end(const MachineFunction &MF) const {
      return end()-2; // we hide regs F0, F1 from the end 
    }
  }];
}

// these are the predicate registers, p0 (1/TRUE) is not here
def PR : RegisterClass<"IA64", [i1], 64, 

// for now, let's be wimps and only have the scratch predicate regs
 [p6, p7, p8, p9, p10, p11, p12, p13, p14, p15]> {
   let Size = 64;
 }

/*
 [p1, p2, p3, p4, p5, p6, p7,
  p8, p9, p10, p11, p12, p13, p14, p15,
  p16, p17, p18, p19, p20, p21, p22, p23,
  p24, p25, p26, p27, p28, p29, p30, p31,
  p32, p33, p34, p35, p36, p37, p38, p39,
  p40, p41, p42, p43, p44, p45, p46, p47,
  p48, p49, p50, p51, p52, p53, p54, p55,
  p56, p57, p58, p59, p60, p61, p62, p63]>;
  */
