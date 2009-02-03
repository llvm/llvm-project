//===- SPURegisterInfo.td - The Cell SPU Register File -----*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

class SPUReg<string n> : Register<n> {
  let Namespace = "SPU";
}

// The SPU's register are all 128-bits wide, which makes specifying the
// registers relatively easy, if relatively mundane:

class SPUVecReg<bits<7> num, string n> : SPUReg<n> {
  field bits<7> Num = num;
}

def R0 : SPUVecReg<0, "$lr">, DwarfRegNum<[0]>;
def R1 : SPUVecReg<1, "$sp">, DwarfRegNum<[1]>;
def R2 : SPUVecReg<2, "$2">, DwarfRegNum<[2]>;
def R3 : SPUVecReg<3, "$3">, DwarfRegNum<[3]>;
def R4 : SPUVecReg<4, "$4">, DwarfRegNum<[4]>;
def R5 : SPUVecReg<5, "$5">, DwarfRegNum<[5]>;
def R6 : SPUVecReg<6, "$6">, DwarfRegNum<[6]>;
def R7 : SPUVecReg<7, "$7">, DwarfRegNum<[7]>;
def R8 : SPUVecReg<8, "$8">, DwarfRegNum<[8]>;
def R9 : SPUVecReg<9, "$9">, DwarfRegNum<[9]>;
def R10 : SPUVecReg<10, "$10">, DwarfRegNum<[10]>;
def R11 : SPUVecReg<11, "$11">, DwarfRegNum<[11]>;
def R12 : SPUVecReg<12, "$12">, DwarfRegNum<[12]>;
def R13 : SPUVecReg<13, "$13">, DwarfRegNum<[13]>;
def R14 : SPUVecReg<14, "$14">, DwarfRegNum<[14]>;
def R15 : SPUVecReg<15, "$15">, DwarfRegNum<[15]>;
def R16 : SPUVecReg<16, "$16">, DwarfRegNum<[16]>;
def R17 : SPUVecReg<17, "$17">, DwarfRegNum<[17]>;
def R18 : SPUVecReg<18, "$18">, DwarfRegNum<[18]>;
def R19 : SPUVecReg<19, "$19">, DwarfRegNum<[19]>;
def R20 : SPUVecReg<20, "$20">, DwarfRegNum<[20]>;
def R21 : SPUVecReg<21, "$21">, DwarfRegNum<[21]>;
def R22 : SPUVecReg<22, "$22">, DwarfRegNum<[22]>;
def R23 : SPUVecReg<23, "$23">, DwarfRegNum<[23]>;
def R24 : SPUVecReg<24, "$24">, DwarfRegNum<[24]>;
def R25 : SPUVecReg<25, "$25">, DwarfRegNum<[25]>;
def R26 : SPUVecReg<26, "$26">, DwarfRegNum<[26]>;
def R27 : SPUVecReg<27, "$27">, DwarfRegNum<[27]>;
def R28 : SPUVecReg<28, "$28">, DwarfRegNum<[28]>;
def R29 : SPUVecReg<29, "$29">, DwarfRegNum<[29]>;
def R30 : SPUVecReg<30, "$30">, DwarfRegNum<[30]>;
def R31 : SPUVecReg<31, "$31">, DwarfRegNum<[31]>;
def R32 : SPUVecReg<32, "$32">, DwarfRegNum<[32]>;
def R33 : SPUVecReg<33, "$33">, DwarfRegNum<[33]>;
def R34 : SPUVecReg<34, "$34">, DwarfRegNum<[34]>;
def R35 : SPUVecReg<35, "$35">, DwarfRegNum<[35]>;
def R36 : SPUVecReg<36, "$36">, DwarfRegNum<[36]>;
def R37 : SPUVecReg<37, "$37">, DwarfRegNum<[37]>;
def R38 : SPUVecReg<38, "$38">, DwarfRegNum<[38]>;
def R39 : SPUVecReg<39, "$39">, DwarfRegNum<[39]>;
def R40 : SPUVecReg<40, "$40">, DwarfRegNum<[40]>;
def R41 : SPUVecReg<41, "$41">, DwarfRegNum<[41]>;
def R42 : SPUVecReg<42, "$42">, DwarfRegNum<[42]>;
def R43 : SPUVecReg<43, "$43">, DwarfRegNum<[43]>;
def R44 : SPUVecReg<44, "$44">, DwarfRegNum<[44]>;
def R45 : SPUVecReg<45, "$45">, DwarfRegNum<[45]>;
def R46 : SPUVecReg<46, "$46">, DwarfRegNum<[46]>;
def R47 : SPUVecReg<47, "$47">, DwarfRegNum<[47]>;
def R48 : SPUVecReg<48, "$48">, DwarfRegNum<[48]>;
def R49 : SPUVecReg<49, "$49">, DwarfRegNum<[49]>;
def R50 : SPUVecReg<50, "$50">, DwarfRegNum<[50]>;
def R51 : SPUVecReg<51, "$51">, DwarfRegNum<[51]>;
def R52 : SPUVecReg<52, "$52">, DwarfRegNum<[52]>;
def R53 : SPUVecReg<53, "$53">, DwarfRegNum<[53]>;
def R54 : SPUVecReg<54, "$54">, DwarfRegNum<[54]>;
def R55 : SPUVecReg<55, "$55">, DwarfRegNum<[55]>;
def R56 : SPUVecReg<56, "$56">, DwarfRegNum<[56]>;
def R57 : SPUVecReg<57, "$57">, DwarfRegNum<[57]>;
def R58 : SPUVecReg<58, "$58">, DwarfRegNum<[58]>;
def R59 : SPUVecReg<59, "$59">, DwarfRegNum<[59]>;
def R60 : SPUVecReg<60, "$60">, DwarfRegNum<[60]>;
def R61 : SPUVecReg<61, "$61">, DwarfRegNum<[61]>;
def R62 : SPUVecReg<62, "$62">, DwarfRegNum<[62]>;
def R63 : SPUVecReg<63, "$63">, DwarfRegNum<[63]>;
def R64 : SPUVecReg<64, "$64">, DwarfRegNum<[64]>;
def R65 : SPUVecReg<65, "$65">, DwarfRegNum<[65]>;
def R66 : SPUVecReg<66, "$66">, DwarfRegNum<[66]>;
def R67 : SPUVecReg<67, "$67">, DwarfRegNum<[67]>;
def R68 : SPUVecReg<68, "$68">, DwarfRegNum<[68]>;
def R69 : SPUVecReg<69, "$69">, DwarfRegNum<[69]>;
def R70 : SPUVecReg<70, "$70">, DwarfRegNum<[70]>;
def R71 : SPUVecReg<71, "$71">, DwarfRegNum<[71]>;
def R72 : SPUVecReg<72, "$72">, DwarfRegNum<[72]>;
def R73 : SPUVecReg<73, "$73">, DwarfRegNum<[73]>;
def R74 : SPUVecReg<74, "$74">, DwarfRegNum<[74]>;
def R75 : SPUVecReg<75, "$75">, DwarfRegNum<[75]>;
def R76 : SPUVecReg<76, "$76">, DwarfRegNum<[76]>;
def R77 : SPUVecReg<77, "$77">, DwarfRegNum<[77]>;
def R78 : SPUVecReg<78, "$78">, DwarfRegNum<[78]>;
def R79 : SPUVecReg<79, "$79">, DwarfRegNum<[79]>;
def R80 : SPUVecReg<80, "$80">, DwarfRegNum<[80]>;
def R81 : SPUVecReg<81, "$81">, DwarfRegNum<[81]>;
def R82 : SPUVecReg<82, "$82">, DwarfRegNum<[82]>;
def R83 : SPUVecReg<83, "$83">, DwarfRegNum<[83]>;
def R84 : SPUVecReg<84, "$84">, DwarfRegNum<[84]>;
def R85 : SPUVecReg<85, "$85">, DwarfRegNum<[85]>;
def R86 : SPUVecReg<86, "$86">, DwarfRegNum<[86]>;
def R87 : SPUVecReg<87, "$87">, DwarfRegNum<[87]>;
def R88 : SPUVecReg<88, "$88">, DwarfRegNum<[88]>;
def R89 : SPUVecReg<89, "$89">, DwarfRegNum<[89]>;
def R90 : SPUVecReg<90, "$90">, DwarfRegNum<[90]>;
def R91 : SPUVecReg<91, "$91">, DwarfRegNum<[91]>;
def R92 : SPUVecReg<92, "$92">, DwarfRegNum<[92]>;
def R93 : SPUVecReg<93, "$93">, DwarfRegNum<[93]>;
def R94 : SPUVecReg<94, "$94">, DwarfRegNum<[94]>;
def R95 : SPUVecReg<95, "$95">, DwarfRegNum<[95]>;
def R96 : SPUVecReg<96, "$96">, DwarfRegNum<[96]>;
def R97 : SPUVecReg<97, "$97">, DwarfRegNum<[97]>;
def R98 : SPUVecReg<98, "$98">, DwarfRegNum<[98]>;
def R99 : SPUVecReg<99, "$99">, DwarfRegNum<[99]>;
def R100 : SPUVecReg<100, "$100">, DwarfRegNum<[100]>;
def R101 : SPUVecReg<101, "$101">, DwarfRegNum<[101]>;
def R102 : SPUVecReg<102, "$102">, DwarfRegNum<[102]>;
def R103 : SPUVecReg<103, "$103">, DwarfRegNum<[103]>;
def R104 : SPUVecReg<104, "$104">, DwarfRegNum<[104]>;
def R105 : SPUVecReg<105, "$105">, DwarfRegNum<[105]>;
def R106 : SPUVecReg<106, "$106">, DwarfRegNum<[106]>;
def R107 : SPUVecReg<107, "$107">, DwarfRegNum<[107]>;
def R108 : SPUVecReg<108, "$108">, DwarfRegNum<[108]>;
def R109 : SPUVecReg<109, "$109">, DwarfRegNum<[109]>;
def R110 : SPUVecReg<110, "$110">, DwarfRegNum<[110]>;
def R111 : SPUVecReg<111, "$111">, DwarfRegNum<[111]>;
def R112 : SPUVecReg<112, "$112">, DwarfRegNum<[112]>;
def R113 : SPUVecReg<113, "$113">, DwarfRegNum<[113]>;
def R114 : SPUVecReg<114, "$114">, DwarfRegNum<[114]>;
def R115 : SPUVecReg<115, "$115">, DwarfRegNum<[115]>;
def R116 : SPUVecReg<116, "$116">, DwarfRegNum<[116]>;
def R117 : SPUVecReg<117, "$117">, DwarfRegNum<[117]>;
def R118 : SPUVecReg<118, "$118">, DwarfRegNum<[118]>;
def R119 : SPUVecReg<119, "$119">, DwarfRegNum<[119]>;
def R120 : SPUVecReg<120, "$120">, DwarfRegNum<[120]>;
def R121 : SPUVecReg<121, "$121">, DwarfRegNum<[121]>;
def R122 : SPUVecReg<122, "$122">, DwarfRegNum<[122]>;
def R123 : SPUVecReg<123, "$123">, DwarfRegNum<[123]>;
def R124 : SPUVecReg<124, "$124">, DwarfRegNum<[124]>;
def R125 : SPUVecReg<125, "$125">, DwarfRegNum<[125]>;
def R126 : SPUVecReg<126, "$126">, DwarfRegNum<[126]>;
def R127 : SPUVecReg<127, "$127">, DwarfRegNum<[127]>;

/* Need floating point status register here: */
/* def FPCSR : ... */

// The SPU's registers as 128-bit wide entities, and can function as general
// purpose registers, where the operands are in the "preferred slot":
def GPRC : RegisterClass<"SPU", [i128], 128,
 [
   /* volatile register */
   R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16, 
   R17, R18, R19, R20, R21, R22, R23, R24, R25, R26, R27, R28, R29, R30, R31,
   R32, R33, R34, R35, R36, R37, R38, R39, R40, R41, R42, R43, R44, R45, R46,
   R47, R48, R49, R50, R51, R52, R53, R54, R55, R56, R57, R58, R59, R60, R61,
   R62, R63, R64, R65, R66, R67, R68, R69, R70, R71, R72, R73, R74, R75, R76,
   R77, R78, R79,
   /* non-volatile register: take hint from PPC and allocate in reverse order */
   R127, R126, R125, R124, R123, R122, R121, R120, R119, R118, R117, R116, R115,
   R114, R113, R112, R111, R110, R109, R108, R107, R106, R105, R104, R103, R102,
   R101, R100, R99, R98, R97, R96, R95, R94, R93, R92, R91, R90, R89, R88, R87,
   R86, R85, R84, R83, R82, R81, R80, 
   /* environment ptr, SP, LR */ 
   R2, R1, R0 ]>
{
  let MethodProtos = [{
    iterator allocation_order_begin(const MachineFunction &MF) const;
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    GPRCClass::iterator
    GPRCClass::allocation_order_begin(const MachineFunction &MF) const {
      return begin();
    }
    GPRCClass::iterator
    GPRCClass::allocation_order_end(const MachineFunction &MF) const {
      return end()-3;  // don't allocate R2, R1, or R0 (envp, sp, lr)
    }
  }];
}

// The SPU's registers as 64-bit wide (double word integer) "preferred slot":
def R64C : RegisterClass<"SPU", [i64], 128,
 [
   /* volatile register */
   R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16, 
   R17, R18, R19, R20, R21, R22, R23, R24, R25, R26, R27, R28, R29, R30, R31,
   R32, R33, R34, R35, R36, R37, R38, R39, R40, R41, R42, R43, R44, R45, R46,
   R47, R48, R49, R50, R51, R52, R53, R54, R55, R56, R57, R58, R59, R60, R61,
   R62, R63, R64, R65, R66, R67, R68, R69, R70, R71, R72, R73, R74, R75, R76,
   R77, R78, R79,
   /* non-volatile register: take hint from PPC and allocate in reverse order */
   R127, R126, R125, R124, R123, R122, R121, R120, R119, R118, R117, R116, R115,
   R114, R113, R112, R111, R110, R109, R108, R107, R106, R105, R104, R103, R102,
   R101, R100, R99, R98, R97, R96, R95, R94, R93, R92, R91, R90, R89, R88, R87,
   R86, R85, R84, R83, R82, R81, R80, 
   /* environment ptr, SP, LR */ 
   R2, R1, R0 ]>
{
  let MethodProtos = [{
    iterator allocation_order_begin(const MachineFunction &MF) const;
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    R64CClass::iterator
    R64CClass::allocation_order_begin(const MachineFunction &MF) const {
      return begin();
    }
    R64CClass::iterator
    R64CClass::allocation_order_end(const MachineFunction &MF) const {
      return end()-3;  // don't allocate R2, R1, or R0 (envp, sp, lr)
    }
  }];
}

// The SPU's registers as 64-bit wide (double word) FP "preferred slot":
def R64FP : RegisterClass<"SPU", [f64], 128,
 [
   /* volatile register */
   R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16, 
   R17, R18, R19, R20, R21, R22, R23, R24, R25, R26, R27, R28, R29, R30, R31,
   R32, R33, R34, R35, R36, R37, R38, R39, R40, R41, R42, R43, R44, R45, R46,
   R47, R48, R49, R50, R51, R52, R53, R54, R55, R56, R57, R58, R59, R60, R61,
   R62, R63, R64, R65, R66, R67, R68, R69, R70, R71, R72, R73, R74, R75, R76,
   R77, R78, R79,
   /* non-volatile register: take hint from PPC and allocate in reverse order */
   R127, R126, R125, R124, R123, R122, R121, R120, R119, R118, R117, R116, R115,
   R114, R113, R112, R111, R110, R109, R108, R107, R106, R105, R104, R103, R102,
   R101, R100, R99, R98, R97, R96, R95, R94, R93, R92, R91, R90, R89, R88, R87,
   R86, R85, R84, R83, R82, R81, R80, 
   /* environment ptr, SP, LR */ 
   R2, R1, R0 ]>
{
  let MethodProtos = [{
    iterator allocation_order_begin(const MachineFunction &MF) const;
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    R64FPClass::iterator
    R64FPClass::allocation_order_begin(const MachineFunction &MF) const {
      return begin();
    }
    R64FPClass::iterator
    R64FPClass::allocation_order_end(const MachineFunction &MF) const {
      return end()-3;  // don't allocate R2, R1, or R0 (envp, sp, lr)
    }
  }];
}

// The SPU's registers as 32-bit wide (word) "preferred slot":
def R32C : RegisterClass<"SPU", [i32], 128,
 [
   /* volatile register */
   R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16, 
   R17, R18, R19, R20, R21, R22, R23, R24, R25, R26, R27, R28, R29, R30, R31,
   R32, R33, R34, R35, R36, R37, R38, R39, R40, R41, R42, R43, R44, R45, R46,
   R47, R48, R49, R50, R51, R52, R53, R54, R55, R56, R57, R58, R59, R60, R61,
   R62, R63, R64, R65, R66, R67, R68, R69, R70, R71, R72, R73, R74, R75, R76,
   R77, R78, R79,
   /* non-volatile register: take hint from PPC and allocate in reverse order */
   R127, R126, R125, R124, R123, R122, R121, R120, R119, R118, R117, R116, R115,
   R114, R113, R112, R111, R110, R109, R108, R107, R106, R105, R104, R103, R102,
   R101, R100, R99, R98, R97, R96, R95, R94, R93, R92, R91, R90, R89, R88, R87,
   R86, R85, R84, R83, R82, R81, R80, 
   /* environment ptr, SP, LR */ 
   R2, R1, R0 ]>
{
  let MethodProtos = [{
    iterator allocation_order_begin(const MachineFunction &MF) const;
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    R32CClass::iterator
    R32CClass::allocation_order_begin(const MachineFunction &MF) const {
      return begin();
    }
    R32CClass::iterator
    R32CClass::allocation_order_end(const MachineFunction &MF) const {
      return end()-3;  // don't allocate R2, R1, or R0 (envp, sp, lr)
    }
  }];
}

// The SPU's registers as single precision floating point "preferred slot":
def R32FP : RegisterClass<"SPU", [f32], 128,
 [
   /* volatile register */
   R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16, 
   R17, R18, R19, R20, R21, R22, R23, R24, R25, R26, R27, R28, R29, R30, R31,
   R32, R33, R34, R35, R36, R37, R38, R39, R40, R41, R42, R43, R44, R45, R46,
   R47, R48, R49, R50, R51, R52, R53, R54, R55, R56, R57, R58, R59, R60, R61,
   R62, R63, R64, R65, R66, R67, R68, R69, R70, R71, R72, R73, R74, R75, R76,
   R77, R78, R79,
   /* non-volatile register: take hint from PPC and allocate in reverse order */
   R127, R126, R125, R124, R123, R122, R121, R120, R119, R118, R117, R116, R115,
   R114, R113, R112, R111, R110, R109, R108, R107, R106, R105, R104, R103, R102,
   R101, R100, R99, R98, R97, R96, R95, R94, R93, R92, R91, R90, R89, R88, R87,
   R86, R85, R84, R83, R82, R81, R80, 
   /* environment ptr, SP, LR */ 
   R2, R1, R0 ]>
{
  let MethodProtos = [{
    iterator allocation_order_begin(const MachineFunction &MF) const;
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    R32FPClass::iterator
    R32FPClass::allocation_order_begin(const MachineFunction &MF) const {
      return begin();
    }
    R32FPClass::iterator
    R32FPClass::allocation_order_end(const MachineFunction &MF) const {
      return end()-3;  // don't allocate R2, R1, or R0 (envp, sp, lr)
    }
  }];
}

// The SPU's registers as 16-bit wide (halfword) "preferred slot":
def R16C : RegisterClass<"SPU", [i16], 128,
 [
   /* volatile register */
   R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16, 
   R17, R18, R19, R20, R21, R22, R23, R24, R25, R26, R27, R28, R29, R30, R31,
   R32, R33, R34, R35, R36, R37, R38, R39, R40, R41, R42, R43, R44, R45, R46,
   R47, R48, R49, R50, R51, R52, R53, R54, R55, R56, R57, R58, R59, R60, R61,
   R62, R63, R64, R65, R66, R67, R68, R69, R70, R71, R72, R73, R74, R75, R76,
   R77, R78, R79,
   /* non-volatile register: take hint from PPC and allocate in reverse order */
   R127, R126, R125, R124, R123, R122, R121, R120, R119, R118, R117, R116, R115,
   R114, R113, R112, R111, R110, R109, R108, R107, R106, R105, R104, R103, R102,
   R101, R100, R99, R98, R97, R96, R95, R94, R93, R92, R91, R90, R89, R88, R87,
   R86, R85, R84, R83, R82, R81, R80, 
   /* environment ptr, SP, LR */ 
   R2, R1, R0 ]>
{
  let MethodProtos = [{
    iterator allocation_order_begin(const MachineFunction &MF) const;
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    R16CClass::iterator
    R16CClass::allocation_order_begin(const MachineFunction &MF) const {
      return begin();
    }
    R16CClass::iterator
    R16CClass::allocation_order_end(const MachineFunction &MF) const {
      return end()-3;  // don't allocate R2, R1, or R0 (envp, sp, lr)
    }
  }];
}

// The SPU's registers as 8-bit wide (byte) "preferred slot":
def R8C : RegisterClass<"SPU", [i8], 128,
 [
   /* volatile register */
   R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16, 
   R17, R18, R19, R20, R21, R22, R23, R24, R25, R26, R27, R28, R29, R30, R31,
   R32, R33, R34, R35, R36, R37, R38, R39, R40, R41, R42, R43, R44, R45, R46,
   R47, R48, R49, R50, R51, R52, R53, R54, R55, R56, R57, R58, R59, R60, R61,
   R62, R63, R64, R65, R66, R67, R68, R69, R70, R71, R72, R73, R74, R75, R76,
   R77, R78, R79,
   /* non-volatile register: take hint from PPC and allocate in reverse order */
   R127, R126, R125, R124, R123, R122, R121, R120, R119, R118, R117, R116, R115,
   R114, R113, R112, R111, R110, R109, R108, R107, R106, R105, R104, R103, R102,
   R101, R100, R99, R98, R97, R96, R95, R94, R93, R92, R91, R90, R89, R88, R87,
   R86, R85, R84, R83, R82, R81, R80, 
   /* environment ptr, SP, LR */ 
   R2, R1, R0 ]>
{
  let MethodProtos = [{
    iterator allocation_order_begin(const MachineFunction &MF) const;
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    R8CClass::iterator
    R8CClass::allocation_order_begin(const MachineFunction &MF) const {
      return begin();
    }
    R8CClass::iterator
    R8CClass::allocation_order_end(const MachineFunction &MF) const {
      return end()-3;  // don't allocate R2, R1, or R0 (envp, sp, lr)
    }
  }];
}

// The SPU's registers as vector registers:
def VECREG : RegisterClass<"SPU",
                           [v16i8,v8i16,v2i32,v4i32,v4f32,v2i64,v2f64],
                           128,
 [
   /* volatile register */
   R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16, 
   R17, R18, R19, R20, R21, R22, R23, R24, R25, R26, R27, R28, R29, R30, R31,
   R32, R33, R34, R35, R36, R37, R38, R39, R40, R41, R42, R43, R44, R45, R46,
   R47, R48, R49, R50, R51, R52, R53, R54, R55, R56, R57, R58, R59, R60, R61,
   R62, R63, R64, R65, R66, R67, R68, R69, R70, R71, R72, R73, R74, R75, R76,
   R77, R78, R79,
   /* non-volatile register: take hint from PPC and allocate in reverse order */
   R127, R126, R125, R124, R123, R122, R121, R120, R119, R118, R117, R116, R115,
   R114, R113, R112, R111, R110, R109, R108, R107, R106, R105, R104, R103, R102,
   R101, R100, R99, R98, R97, R96, R95, R94, R93, R92, R91, R90, R89, R88, R87,
   R86, R85, R84, R83, R82, R81, R80, 
   /* environment ptr, SP, LR */ 
   R2, R1, R0 ]>
{
  let MethodProtos = [{
    iterator allocation_order_begin(const MachineFunction &MF) const;
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    VECREGClass::iterator
    VECREGClass::allocation_order_begin(const MachineFunction &MF) const {
      return begin();
    }
    VECREGClass::iterator
    VECREGClass::allocation_order_end(const MachineFunction &MF) const {
      return end()-3;  // don't allocate R2, R1, or R0 (envp, sp, lr)
    }
  }];
}
