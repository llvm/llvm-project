//===-- LoongArchSchedule.td - LoongArch Scheduling Definitions ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// Define scheduler resources associated with def operands.

/// BASE instruction - integer instruction
def WriteIALU         : SchedWrite; // 32 or 64-bit integer ALU operations
def WriteIALU32       : SchedWrite; // 32-bit integer ALU operations on LA64
def ReadIALU          : SchedRead;  // 32 or 64-bit integer ALU operations
def ReadIALU32        : SchedRead;  // 32-bit integer ALU operations on LA64
//===----------------------------------------------------------------------===//
// ADD.W  SUB.W                     Sched<[WriteIALU, ReadIALU32, ReadIALU32]>
// ADD.D  SUB.D                     Sched<[WriteIALU, ReadIALU, ReadIALU]>
// ADDI.W                           Sched<[WriteIALU, ReadIALU32]>
// ADDI.D  ADDUI16.D                Sched<[WriteIALU, ReadIALU]>
// AND  ANDN  OR  NOR  ORN  XOR     Sched<[WriteIALU, ReadIALU, ReadIALU]>
// ANDI  XORI  ORI                  Sched<[WriteIALU, ReadIALU]>
// SLT  SLTU                        Sched<[WriteIALU, ReadIALU, ReadIALU]>
// SLTI  SLTUI                      Sched<[WriteIALU, ReadIALU]>
// LU12I.W  LU32I.D                 Sched<[WriteIALU]>
// LU52I.D                          Sched<[WriteIALU, ReadIALU]>
//===----------------------------------------------------------------------===//

def WriteIALSL        : SchedWrite; // 32 or 64-bit integer ALSL operations
def WriteIALSL32      : SchedWrite; // 32-bit integer ALSL operations on LA64
def ReadIALSL         : SchedRead;  // 32 or 64-bit integer ALSL operations
def ReadIALSL32       : SchedRead;  // 32-bit integer ALSL operations on LA64
//===----------------------------------------------------------------------===//
// ALSL.W  ALSL.WU                  Sched<[WriteIALSL, ReadIALSL32, ReadIALSL32]>
// ALSL.D                           Sched<[WriteIALSL, ReadIALSL, ReadIALSL]>
//===----------------------------------------------------------------------===//

def WriteIPCAdd       : SchedWrite; // LoongArch PC ADD instruction operation
def ReadIPCAdd        : SchedRead;  // LoongArch PC ADD instruction operation
//===----------------------------------------------------------------------===//
// PCADDI  PCADDUI12I  PCADDU18I  PCALAU12I   Sched<[WriteIPCAdd]>
//===----------------------------------------------------------------------===//

def WriteIMUL         : SchedWrite; // 32 or 64-bit integer MUL operations
def WriteIMUL32       : SchedWrite; // 32-bit integer MUL operations on LA64
def ReadIMUL          : SchedRead;  // 32 or 64-bit integer MUL operations
def ReadIMUL32        : SchedRead;  // 32-bit integer MUL operations on LA64
//===----------------------------------------------------------------------===//
// MUL.W  MULH.W  MULH.WU           Sched<[WriteIMUL, ReadIMUL32, ReadIMUL32]>
// MUL.D  MULH.D  MULH.DU           Sched<[WriteIMUL, ReadIMUL, ReadIMUL]>
// MULW.D.W  MULW.D.WU              Sched<[WriteIMUL, ReadIMUL32, ReadIMUL32]>
//===----------------------------------------------------------------------===//

def WriteIDIV         : SchedWrite; // 32 or 64-bit integer DIV operations
def WriteIDIV32       : SchedWrite; // 32-bit integer DIV operations on LA64
def ReadIDIV          : SchedRead;  // 32 or 64-bit integer DIV operations
def ReadIDIV32        : SchedRead;  // 32-bit integer DIV operations on LA64
//===----------------------------------------------------------------------===//
// DIV.W  MOD.W  DIV.WU  MOD.WU     Sched<[WriteIDIV, ReadIDIV32, ReadIDIV32]>
// DIV.D  MOD.D  DIV.DU  MOD.DU     Sched<[WriteIDIV, ReadIDIV, ReadIDIV]>
//===----------------------------------------------------------------------===//

def WriteIShift       : SchedWrite; // 32 or 64-bit Shift operations
def WriteIShift32     : SchedWrite; // 32-bit Shift operations on LA64
def ReadIShift        : SchedRead;  // 32 or 64-bit Shift operations
def ReadIShift32      : SchedRead;  // 32-bit Shift operations on LA64
//===----------------------------------------------------------------------===//
// SLL.W  SRL.W  SRA.W  ROTR.W      Sched<[WriteIShift, ReadIShift32, ReadIShift32]>
// SLLI.W SRLI.W SRAI.W ROTRI.W     Sched<[WriteIShift, ReadIShift32]>
// SLL.D  SRL.D  SRA.D  ROTR.D      Sched<[WriteIShift, ReadIShift, ReadIShift]>
// SLLI.D SRLI.D SRAI.D ROTRI.D     Sched<[WriteIShift, ReadIShift]>
//===----------------------------------------------------------------------===//

def WriteIEXT         : SchedWrite; // 32 or 64-bit EXT operations
def ReadIEXT          : SchedRead;  // 32 or 64-bit EXT operations
//===----------------------------------------------------------------------===//
// EXT.W.B  EXT.W.H                 Sched<[WriteIEXT, ReadIEXT]>
//===----------------------------------------------------------------------===//

def WriteICL          : SchedWrite; // 32 or 64-bit CLO/CLZ operations
def WriteICL32        : SchedWrite; // 32-bit CLO/CLZ operations on LA64
def WriteICT          : SchedWrite; // 32 or 64-bit CTO/CTZ operations
def WriteICT32        : SchedWrite; // 32-bit CTO/CTZ operations on LA64
def ReadICL           : SchedRead;  // 32 or 64-bit CLO/CLZ operations
def ReadICL32         : SchedRead;  // 32-bit CLO/CLZ operations on LA64
def ReadICT           : SchedRead;  // 32 or 64-bit CTO/CTZ operations
def ReadICT32         : SchedRead;  // 32-bit CTO/CTZ operations on LA64
//===----------------------------------------------------------------------===//
// CLO.W  CLZ.W                     Sched<[WriteICL, ReadICL32]>
// CTO.W  CTZ.W                     Sched<[WriteICT, ReadICT32]>
// CLO.D  CLZ.D                     Sched<[WriteICL, ReadICL]>
// CTO.D  CTZ.D                     Sched<[WriteICT, ReadICT]>
//===----------------------------------------------------------------------===//

def WriteIBytePick    : SchedWrite; // 32 or 64-bit BytePick operations
def WriteIBytePick32  : SchedWrite; // 32-bit BytePick operations on LA64
def ReadIBytePick     : SchedRead;  // 32 or 64-bit BytePick operations
def ReadIBytePick32   : SchedRead;  // 32-bit BytePick operations on LA64
//===----------------------------------------------------------------------===//
// BYTEPICK.W  Sched<[WriteIBytePick, ReadIBytePick32, ReadIBytePick32]>
// BYTEPICK.D  Sched<[WriteIBytePick, ReadIBytePick, ReadIBytePick]>
//===----------------------------------------------------------------------===//

def WriteIRev         : SchedWrite; // 32 or 64-bit RevB operations
def WriteIRev32       : SchedWrite; // 32-bit RevB operations on LA64
def ReadIRev          : SchedRead;  // 32 or 64-bit RevB operations
def ReadIRev32        : SchedRead;  // 32-bit RevB operations on LA64
//===----------------------------------------------------------------------===//
// REVB.2H                          Sched<[WriteIRev, ReadIRev32]>
// REVB.4H  REVB.2W  REVB.D         Sched<[WriteIRev, ReadIRev]>
// REVH.2W  REVH.D                  Sched<[WriteIRev, ReadIRev]>
//===----------------------------------------------------------------------===//

def WriteIBitRev      : SchedWrite; // 32 or 64-bit BitRev operations
def WriteIBitRev32    : SchedWrite; // 32-bit BitRev operations on LA64
def ReadIBitRev       : SchedRead;  // 32 or 64-bit BitRev operations
def ReadIBitRev32     : SchedRead;  // 32-bit BitRev operations on LA64
//===----------------------------------------------------------------------===//
// BITREV.4B                        Sched<[WriteIBitRev, ReadIBitRev32]>
// BITREV.8B  BITREV.W  BITREV.D    Sched<[WriteIBitRev, ReadIBitRev]>
//===----------------------------------------------------------------------===//

def WriteIBstrins     : SchedWrite; // 32 or 64-bit Bstrins operations
def WriteIBstrins32   : SchedWrite; // 32-bit Bstrins operations on LA64
def ReadIBstrins      : SchedRead;  // 32 or 64-bit Bstrins operations
def ReadIBstrins32    : SchedRead;  // 32-bit Bstrins operations on LA64
//===----------------------------------------------------------------------===//
// BSTRINS.W          Sched<[WriteIBstrins, ReadIBstrins32]>
// BSTRINS.D          Sched<[WriteIBstrins, ReadIBstrins]>
//===----------------------------------------------------------------------===//

def WriteIBstrPick    : SchedWrite; // 32 or 64-bit BstrPick operations
def WriteIBstrPick32  : SchedWrite; // 32-bit BstrPick operations on LA64
def ReadIBstrPick     : SchedRead;  // 32 or 64-bit BstrPick operations
def ReadIBstrPick32   : SchedRead;  // 32-bit BstrPick operations on LA64
//===----------------------------------------------------------------------===//
// BSTRPICK.W         Sched<[WriteIBstrPick, ReadIBstrPick32]>
// BSTRPICK.D         Sched<[WriteIBstrPick, ReadIBstrPick]>
//===----------------------------------------------------------------------===//

def WriteIMASK        : SchedWrite; // 32 or 64-bit MASK operations
def ReadIMASK         : SchedRead;  // 32 or 64-bit MASK operations
//===----------------------------------------------------------------------===//
// MASKEQZ  MASKNEZ   Sched<[WriteIMASK, ReadIMASK]>
//===----------------------------------------------------------------------===//

def WriteIJump        : SchedWrite; // integer Jump
def ReadIJump         : SchedRead;  // integer Jump
//===----------------------------------------------------------------------===//
// BEQ BNE BLT BGE BLTU BGEU        Sched<[WriteIJump, ReadIJump, ReadIJump]>
// BEQZ BNEZ                        Sched<[WriteIJump, ReadIJump]>
// B BL                             Sched<[WriteIJump]>
//===----------------------------------------------------------------------===//

def WriteLD           : SchedWrite; // Load  reg+imm
def WriteST           : SchedWrite; // Store reg+imm
def WriteLDX          : SchedWrite; // Load  reg+reg
def WriteSTX          : SchedWrite; // Store reg+reg
def ReadSTData        : SchedRead;  // Load / Store STData
def ReadMemBase       : SchedRead;  // Load / Store MemBase
def ReadMemOffset     : SchedRead;  // Load / Store MemOffset
//===----------------------------------------------------------------------===//
// LD.B    LD.H    LD.W    LD.D     Sched<[WriteLD, ReadMemBase]>
// LD.BU   LD.HU   LD.WU            Sched<[WriteLD, ReadMemBase]>
// ST.B    ST.H    ST.W    ST.D     Sched<[WriteST, ReadSTData, ReadMemBase]>
//
// def SB : Store_rri<0b000, "sb">, Sched<[WriteSTB, ReadStoreData, ReadMemBase]>;
//
// LDX.B   LDX.H   LDX.W   LDX.D    Sched<[WriteLDX, ReadMemBase, ReadMemOffset]>
// LDX.BU  LDX.HU  LDX.WU           Sched<[WriteLDX, ReadMemBase, ReadMemOffset]>
// STX.B   STX.H   STX.W   STX.D    Sched<[WriteSTX, ReadSTData, ReadMemBase, ReadMemOffset]>
//===----------------------------------------------------------------------===//

def WriteLDPTR        : SchedWrite; // Load ptr  reg+imm
def WriteSTPTR        : SchedWrite; // Store ptr reg+imm
//===----------------------------------------------------------------------===//
// LDPTR.W  LDPTR.D                 Sched<[WriteLDPTR, ReadMemBase]>
// STPTR.W  STPRT.D                 Sched<[WriteSTPTR, ReadSTData, ReadMemBase]>
//===----------------------------------------------------------------------===//

def WriteLDGT         : SchedWrite; // 64-bit LDGT operations
def WriteLDLE         : SchedWrite; // 64-bit LDLE operations
def WriteSTGT         : SchedWrite; // 64-bit STGT operations
def WriteSTLE         : SchedWrite; // 64-bit STLE operations
def ReadMemCmp        : SchedRead;  // LDGT LDLE STGT STLE MemCmp operaitons
//===----------------------------------------------------------------------===//
// LDGT.B  LDGT.H  LDGT.W  LDGT.D   Sched<[WriteLDGT, ReadMemBase, ReadMemCmp]>
// LDLE.B  LDLE.H  LDLE.W  LDLE.D   Sched<[WriteLDLE, ReadMemBase, ReadMemCmp]>
// STGT.B  STGT.H  STGT.W  STGT.D   Sched<[WriteSTGT, ReadSTData, ReadMemBase, ReadMemCmp]>
// STLE.B  STLE.H  STLE.W  STLE.D   Sched<[WriteSTLE, ReadSTData, ReadMemBase, ReadMemCmp]>
//===----------------------------------------------------------------------===//

def WriteCRCB        : SchedWrite; // 64-bit CRC byte operations
def WriteCRCH        : SchedWrite; // 64-bit CRC half operations
def WriteCRCW        : SchedWrite; // 64-bit CRC word operations
def WriteCRCD        : SchedWrite; // 64-bit CRC dword operations
def ReadCRCB         : SchedRead;  // 64-bit CRC byte operations
def ReadCRCH         : SchedRead;  // 64-bit CRC half operations
def ReadCRCW         : SchedRead;  // 64-bit CRC word operations
def ReadCRCD         : SchedRead;  // 64-bit CRC dword operations
//===----------------------------------------------------------------------===//
// CRC.W.B.W  CRCC.W.B.W            Sched<[WriteCRCB, ReadCRCB, ReadCRCB]>
// CRC.W.H.W  CRCC.W.H.W            Sched<[WriteCRCH, ReadCRCH, ReadCRCH]>
// CRC.W.W.W  CRCC.W.W.W            Sched<[WriteCRCW, ReadCRCW, ReadCRCW]>
// CRC.W.D.W  CRCC.W.D.W            Sched<[WriteCRCD, ReadCRCD, ReadCRCD]>
//===----------------------------------------------------------------------===//

/// BASE instruction - floating instruction
def WriteFADD         : SchedWrite; // floating-point addition/subtraction
def ReadFADD          : SchedRead;  // floating-point addition/subtraction
//===----------------------------------------------------------------------===//
// FADD.S  FADD.D  FSUB.S  FSUB.D   Sched<[WriteFADD, ReadFADD, ReadFADD]>
//===----------------------------------------------------------------------===//

def WriteFMUL         : SchedWrite; // floating-point multiply
def ReadFMUL          : SchedRead;  // floating-point multiply
//===----------------------------------------------------------------------===//
// FMUL.S  FMUL.D     Sched<[WriteFMUL, ReadFMUL, ReadFMUL]>
//===----------------------------------------------------------------------===//

def WriteFDIV         : SchedWrite; // floating-point divide
def WriteFDIV32       : SchedWrite; // 32-bit floating-point divide on LA64
def ReadFDIV          : SchedRead;  // floating-point divide
def ReadFDIV32        : SchedRead;  // 32-bit floating-point divide on LA64
//===----------------------------------------------------------------------===//
// FDIV.S             Sched<[WriteFDIV32, ReadFDIV32, ReadFDIV32]>
// FDIV.D             Sched<[WriteFDIV, ReadFDIV, ReadFDIV]>
//===----------------------------------------------------------------------===//

def WriteFMADD        : SchedWrite; // floating-point fused multiply-add
def ReadFMADD         : SchedRead;  // floating-point fused multiply-add
//===----------------------------------------------------------------------===//
// FMADD.S  FMADD.D   Sched<[WriteFMADD, ReadFMADD, ReadFMADD, ReadFMADD]>
// FMSUB.S  FMSUB.D   Sched<[WriteFMADD, ReadFMADD, ReadFMADD, ReadFMADD]>
// FNMADD.S FNMADD.D  Sched<[WriteFMADD, ReadFMADD, ReadFMADD, ReadFMADD]>
// FNMSUB.S FNMSUB.D  Sched<[WriteFMADD, ReadFMADD, ReadFMADD, ReadFMADD]>
//===----------------------------------------------------------------------===//

def WriteFMinMax      : SchedWrite; // floating-point min or max
def ReadFMinMax       : SchedRead;  // floating-point min or max
//===----------------------------------------------------------------------===//
// FMAX.S   FMAX.D    Sched<[WriteFMinMax, ReadFMinMax, ReadFMinMax]>
// FMIN.S   FMIN.D    Sched<[WriteFMinMax, ReadFMinMax, ReadFMinMax]>
// FMAXA.S  FMAXA.D   Sched<[WriteFMinMax, ReadFMinMax, ReadFMinMax]>
// FMINA.S  FMINA.D   Sched<[WriteFMinMax, ReadFMinMax, ReadFMinMax]>
//===----------------------------------------------------------------------===//

def WriteFABS         : SchedWrite; // floating-point abs
def ReadFABS          : SchedRead;  // floating-point abs
//===----------------------------------------------------------------------===//
// FABS.S  FABS.D     Sched<[WriteFABS, ReadFABS]>
//===----------------------------------------------------------------------===//

def WriteFNEG         : SchedWrite; // floating-point neg
def ReadFNEG          : SchedRead;  // floating-point neg
//===----------------------------------------------------------------------===//
// FNEG.S  FNEG.D     Sched<[WriteFNEG, ReadFNEG]>
//===----------------------------------------------------------------------===//

def WriteFSQRT        : SchedWrite; // floating-point sqrt
def WriteFSQRT32      : SchedWrite; // 32-bit floating-point sqrt on LA64
def ReadFSQRT         : SchedRead;  // floating-point sqrt
def ReadFSQRT32       : SchedRead;  // 32-bit floating-point sqrt on LA64
//===----------------------------------------------------------------------===//
// FSQRT.S  FSQRT.D   Sched<[WriteFSQRT, ReadFSQRT]>
//===----------------------------------------------------------------------===//

def WriteFRECIP       : SchedWrite; // floating-point recip
def WriteFRECIP32     : SchedWrite; // 32-bit floating-point recip on LA64
def ReadFRECIP        : SchedRead;  // floating-point recip
def ReadFRECIP32      : SchedRead;  // 32-bit floating-point recip on LA64
//===----------------------------------------------------------------------===//
// FRECIP.S  FRECIP.D     Sched<[WriteFRECIP, ReadFRECIP]>
//===----------------------------------------------------------------------===//

def WriteFRSQRT       : SchedWrite; // floating-point rsqrt
def WriteFRSQRT32     : SchedWrite; // 32-bit floating-point rsqrt on LA64
def ReadFRSQRT        : SchedRead;  // floating-point rsqrt
def ReadFRSQRT32      : SchedRead;  // 32-bit floating-point rsqrt on LA64
//===----------------------------------------------------------------------===//
// FRSQRT.S  FRSQRT.D     Sched<[WriteFRSQRT, ReadFRSQRT]>
//===----------------------------------------------------------------------===//

def WriteFRECIPE      : SchedWrite; // floating-point recipe
def ReadFRECIPE       : SchedRead;  // floating-point recipe
//===----------------------------------------------------------------------===//
// FRECIPE.S  FRECIPE.D   Sched<[WriteFRECIPE, ReadFRECIPE]>
//===----------------------------------------------------------------------===//

def WriteFRSQRTE      : SchedWrite; // floating-point rsqrte
def ReadFRSQRTE       : SchedRead;  // floating-point rsqrte
//===----------------------------------------------------------------------===//
// FRSQRTE.S  FRSQRTE.D   Sched<[WriteFRSQRTE, ReadFRSQRTE]>
//===----------------------------------------------------------------------===//

def WriteFSCALEB      : SchedWrite; // floating-point scaleb
def ReadFSCALEB       : SchedRead;  // floating-point scaleb
//===----------------------------------------------------------------------===//
// FSCALEB.S  FSCALEB.D   Sched<[WriteFSCALEB, ReadFSCALEB, ReadFSCALEB]>
//===----------------------------------------------------------------------===//

def WriteFLOGB        : SchedWrite; // floating-point logb
def ReadFLOGB         : SchedRead;  // floating-point logb
//===----------------------------------------------------------------------===//
// FLOGB.S  FLOGB.D       Sched<[WriteFLOGB, ReadFLOGB]>
//===----------------------------------------------------------------------===//

def WriteFCOPYSIGN    : SchedWrite; // floating-point copysign
def ReadFCOPYSIGN     : SchedRead;  // floating-point copysign
//===----------------------------------------------------------------------===//
// FCOPYSIGN.S  FCOPYSIGN.D   Sched<[WriteFCOPYSIGN, ReadFCOPYSIGN, ReadFCOPYSIGN]>
//===----------------------------------------------------------------------===//

def WriteFCLASS       : SchedWrite; // floating-point class
def ReadFCLASS        : SchedRead;  // floating-point class
//===----------------------------------------------------------------------===//
// FCLASS.S  FCLASS.D         Sched<[WriteFCLASS, ReadFCLASS]>
//===----------------------------------------------------------------------===//

def WriteFCMP         : SchedWrite; // floating-point condition
def ReadFCMP          : SchedRead;  // floating-point condition
//===----------------------------------------------------------------------===//
// FCMP.COND.S  FCMP.COND.D   Sched<[WriteFCMP, ReadFCMP, ReadFCMP]>
//===----------------------------------------------------------------------===//

def WriteFCvtStoD     : SchedWrite; // Conversion single to double
def WriteFCvtDtoS     : SchedWrite; // Conversion double to single
def ReadFCvtStoD      : SchedRead;  // Conversion single to double
def ReadFCvtDtoS      : SchedRead;  // Conversion double to single
//===----------------------------------------------------------------------===//
// FCVT.S.D                   Sched<[WriteFCvtStoD, ReadFCvtStoD]>
// FCVT.D.S                   Sched<[WriteFCvtDtoS, ReadFCvtDtoS]>
//===----------------------------------------------------------------------===//

def WriteFCvtStoW     : SchedWrite; // Conversion single to integer word
def WriteFCvtStoL     : SchedWrite; // Conversion single to integer doubleword
def WriteFCvtDtoW     : SchedWrite; // Conversion double to integer word
def WriteFCvtDtoL     : SchedWrite; // Conversion double to integer doubleword
def WriteFCvtWtoS     : SchedWrite; // Conversion integer word to single
def WriteFCvtWtoD     : SchedWrite; // Conversion integer word to double
def WriteFCvtLtoS     : SchedWrite; // Conversion integer doubleword to single
def WriteFCvtLtoD     : SchedWrite; // Conversion integer doubleword to double
def ReadFCvtStoW      : SchedRead;  // Conversion single to integer word
def ReadFCvtStoL      : SchedRead;  // Conversion single to integer doubleword
def ReadFCvtDtoW      : SchedRead;  // Conversion double to integer word
def ReadFCvtDtoL      : SchedRead;  // Conversion double to integer doubleword
def ReadFCvtWtoS      : SchedRead;  // Conversion integer word to single
def ReadFCvtWtoD      : SchedRead;  // Conversion integer word to double
def ReadFCvtLtoS      : SchedRead;  // Conversion integer doubleword to single
def ReadFCvtLtoD      : SchedRead;  // Conversion integer doubleword to double
//===----------------------------------------------------------------------===//
// FFINT.S.W                        Sched<[WriteFCvtStoW, ReadFCvtStoW]>
// FFINT.S.L                        Sched<[WriteFCvtStoL, ReadFCvtStoL]>
// FFINT.D.W                        Sched<[WriteFCvtDtoW, ReadFCvtDtoW]>
// FFINT.D.L                        Sched<[WriteFCvtDtoL, ReadFCvtDtoL]>
// FTINT.W.S                        Sched<[WriteFCvtWtoS, ReadFCvtWtoS]>
// FTINT.W.D                        Sched<[WriteFCvtWtoD, ReadFCvtWtoD]>
// FTINT.L.S                        Sched<[WriteFCvtLtoS, ReadFCvtLtoS]>
// FTINT.L.D                        Sched<[WriteFCvtLtoD, ReadFCvtLtoD]>
//===----------------------------------------------------------------------===//

def WriteFCvtWtoS_RM  : SchedWrite; // Conversion integer word to single
def WriteFCvtWtoD_RM  : SchedWrite; // Conversion integer word to double
def WriteFCvtLtoS_RM  : SchedWrite; // Conversion integer doubleword to single
def WriteFCvtLtoD_RM  : SchedWrite; // Conversion integer doubleword to double
def ReadFCvtWtoS_RM   : SchedRead;  // Conversion integer word to single
def ReadFCvtWtoD_RM   : SchedRead;  // Conversion integer word to double
def ReadFCvtLtoS_RM   : SchedRead;  // Conversion integer doubleword to single
def ReadFCvtLtoD_RM   : SchedRead;  // Conversion integer doubleword to double
def WriteFCvtWtoS_RP  : SchedWrite; // Conversion integer word to single
def WriteFCvtWtoD_RP  : SchedWrite; // Conversion integer word to double
def WriteFCvtLtoS_RP  : SchedWrite; // Conversion integer doubleword to single
def WriteFCvtLtoD_RP  : SchedWrite; // Conversion integer doubleword to double
def ReadFCvtWtoS_RP   : SchedRead;  // Conversion integer word to single
def ReadFCvtWtoD_RP   : SchedRead;  // Conversion integer word to double
def ReadFCvtLtoS_RP   : SchedRead;  // Conversion integer doubleword to single
def ReadFCvtLtoD_RP   : SchedRead;  // Conversion integer doubleword to double
def WriteFCvtWtoS_RZ  : SchedWrite; // Conversion integer word to single
def WriteFCvtWtoD_RZ  : SchedWrite; // Conversion integer word to double
def WriteFCvtLtoS_RZ  : SchedWrite; // Conversion integer doubleword to single
def WriteFCvtLtoD_RZ  : SchedWrite; // Conversion integer doubleword to double
def ReadFCvtWtoS_RZ   : SchedRead;  // Conversion integer word to single
def ReadFCvtWtoD_RZ   : SchedRead;  // Conversion integer word to double
def ReadFCvtLtoS_RZ   : SchedRead;  // Conversion integer doubleword to single
def ReadFCvtLtoD_RZ   : SchedRead;  // Conversion integer doubleword to double
def WriteFCvtWtoS_RNE : SchedWrite; // Conversion integer word to single
def WriteFCvtWtoD_RNE : SchedWrite; // Conversion integer word to double
def WriteFCvtLtoS_RNE : SchedWrite; // Conversion integer doubleword to single
def WriteFCvtLtoD_RNE : SchedWrite; // Conversion integer doubleword to double
def ReadFCvtWtoS_RNE  : SchedRead;  // Conversion integer word to single
def ReadFCvtWtoD_RNE  : SchedRead;  // Conversion integer word to double
def ReadFCvtLtoS_RNE  : SchedRead;  // Conversion integer doubleword to single
def ReadFCvtLtoD_RNE  : SchedRead;  // Conversion integer doubleword to double
//===----------------------------------------------------------------------===//
// FTINTRM.W.S                      Sched<[WriteFCvtWtoS_RM, ReadFCvtWtoS_RM]>
// FTINTRM.W.D                      Sched<[WriteFCvtWtoD_RM, ReadFCvtWtoD_RM]>
// FTINTRM.L.S                      Sched<[WriteFCvtLtoS_RM, ReadFCvtLtoS_RM]>
// FTINTRM.L.D                      Sched<[WriteFCvtLtoD_RM, ReadFCvtLtoD_RM]>
// FTINTRP.W.S                      Sched<[WriteFCvtWtoS_RP, ReadFCvtWtoS_RP]>
// FTINTRP.W.D                      Sched<[WriteFCvtWtoD_RP, ReadFCvtWtoD_RP]>
// FTINTRP.L.S                      Sched<[WriteFCvtLtoS_RP, ReadFCvtLtoS_RP]>
// FTINTRP.L.D                      Sched<[WriteFCvtLtoD_RP, ReadFCvtLtoD_RP]>
// FTINTRZ.W.S                      Sched<[WriteFCvtWtoS_RZ, ReadFCvtWtoS_RZ]>
// FTINTRZ.W.D                      Sched<[WriteFCvtWtoD_RZ, ReadFCvtWtoD_RZ]>
// FTINTRZ.L.S                      Sched<[WriteFCvtLtoS_RZ, ReadFCvtLtoS_RZ]>
// FTINTRZ.L.D                      Sched<[WriteFCvtLtoD_RZ, ReadFCvtLtoD_RZ]>
// FTINTRNE.W.S                     Sched<[WriteFCvtWtoS_RNE, ReadFCvtWtoS_RNE]>
// FTINTRNE.W.D                     Sched<[WriteFCvtWtoD_RNE, ReadFCvtWtoD_RNE]>
// FTINTRNE.L.S                     Sched<[WriteFCvtLtoS_RNE, ReadFCvtLtoS_RNE]>
// FTINTRNE.L.D                     Sched<[WriteFCvtLtoD_RNE, ReadFCvtLtoD_RNE]>
//===----------------------------------------------------------------------===//

def WriteFCvtFStoIS   : SchedWrite; // Conversion single-floating point to integer single-floating point
def WriteFCvtFDtoID   : SchedWrite; // Conversion double-floating point to integer double-floating point
def ReadFCvtFStoIS    : SchedRead;  // Conversion single-floating point to integer single-floating point
def ReadFCvtFDtoID    : SchedRead;  // Conversion double-floating point to integer double-floating point
//===----------------------------------------------------------------------===//
// FRINT.S                          Sched<[WriteFCvtFStoIS, ReadFCvtFStoIS]>
// FRINT.D                          Sched<[WriteFCvtFDtoID, ReadFCvtFDtoID]>
//===----------------------------------------------------------------------===//

def WriteFMovS        : SchedWrite; // Move single-floating point
def WriteFMovD        : SchedWrite; // Move double-floating point
def ReadFMovS         : SchedRead;  // Move single-floating point
def ReadFMovD         : SchedRead;  // Move double-floating point
//===----------------------------------------------------------------------===//
// FMOV.S                           Sched<[WriteFMovS, ReadFMovS]>
// FMOV.D                           Sched<[WriteFMovD, ReadFMovD]>
//===----------------------------------------------------------------------===//

def WriteFSEL         : SchedWrite; // floating-point select instruction
def ReadFSEL          : SchedRead;  // floating-point select instruction
//===----------------------------------------------------------------------===//
// FSEL                             Sched<[WriteFSEL, ReadFSEL, ReadFSEL, ReadFSEL]>
//===----------------------------------------------------------------------===//

def WriteFMovGRtoFRW  : SchedWrite; // Move GR[31: 0] value to FR[31: 0]
def WriteFMovGRtoFRHW : SchedWrite; // Move GR[63:32] value to FR[63:32]
def WriteFMovGRtoFRD  : SchedWrite; // Move GR[63: 0] value to FR[63: 0]
def ReadFMovGRtoFRW   : SchedRead;  // Move GR[31: 0] value to FR[31: 0]
def ReadFMovGRtoFRHW  : SchedRead;  // Move GR[63:32] value to FR[63:32]
def ReadFMovGRtoFRD   : SchedRead;  // Move GR[63: 0] value to FR[63: 0]
//===----------------------------------------------------------------------===//
// MOVGR2FR.W                       Sched<[WriteFMovGRtoFRW, ReadFMovGRtoFRW]>
// MOVGR2FRH.W                      Sched<[WriteFMovGRtoFRHW, ReadFMovGRtoFRHW]>
// MOVGR2FR.D                       Sched<[WriteFMovGRtoFRD, ReadFMovGRtoFRD]>
//===----------------------------------------------------------------------===//

def WriteFMovFRtoGRS  : SchedWrite; // Move FR[31: 0] value to GR
def WriteFMovFRHtoGRS : SchedWrite; // Move FR[63:32] value to GR
def WriteFMovFRtoGRD  : SchedWrite; // Move FR[63: 0] value to GR
def ReadFMovFRtoGRS   : SchedRead;  // Move FR[31: 0] value to GR
def ReadFMovFRHtoGRS  : SchedRead;  // Move FR[63:32] value to GR
def ReadFMovFRtoGRD   : SchedRead;  // Move FR[63: 0] value to GR
//===----------------------------------------------------------------------===//
// MOVFR2GR.S                       Sched<[WriteFMovFRtoGRS, ReadFMovFRtoGRS]>
// MOVFRH2GR.S                      Sched<[WriteFMovFRHtoGRS, ReadFMovFRHtoGRS]>
// MOVFR2GR.D                       Sched<[WriteFMovFRtoGRD, ReadFMovFRtoGRD]>
//===----------------------------------------------------------------------===//

def WriteFMovGRtoFCSR : SchedWrite; // Move GR value to FCSR
def WriteFMovFCSRtoGR : SchedWrite; // Move FCSR value to GR
def ReadFMovGRtoFCSR  : SchedRead;  // Move GR value to FCSR
def ReadFMovFCSRtoGR  : SchedRead;  // Move FCSR value to GR
//===----------------------------------------------------------------------===//
// MOVGR2FCSR                       Sched<[WriteFMovGRtoFCSR, ReadFMovGRtoFCSR]>
// MOVFCSR2GR                       Sched<[WriteFMovFCSRtoGR, ReadFMovFCSRtoGR]>
//===----------------------------------------------------------------------===//

def WriteFMoveFRtoCF  : SchedWrite; // Move FR value to FCC
def WriteFMoveCFtoFR  : SchedWrite; // Move FCC value to FR
def ReadFMoveFRtoCF   : SchedRead;  // Move FR value to FCC
def ReadFMoveCFtoFR   : SchedRead;  // Move FCC value to FR
//===----------------------------------------------------------------------===//
// MOVFR2CF                         Sched<[WriteFMoveFRtoCF, ReadFMoveFRtoCF]>
// MOVCF2FR                         Sched<[WriteFMoveCFtoFR, ReadFMoveCFtoFR]>
//===----------------------------------------------------------------------===//

def WriteFMoveGRtoCF  : SchedWrite; // Move GR value to FCC
def WriteFMoveCFtoGR  : SchedWrite; // Move FCC value to GR
def ReadFMoveGRtoCF   : SchedRead;  // Move GR value to FCC
def ReadFMoveCFtoGR   : SchedRead;  // Move FCC value to GR
//===----------------------------------------------------------------------===//
// MOVGR2CF                         Sched<[WriteFMoveGRtoCF, ReadFMoveGRtoCF]>
// MOVCF2GR                         Sched<[WriteFMoveCFtoGR, ReadFMoveCFtoGR]>
//===----------------------------------------------------------------------===//

def WriteFJump        : SchedWrite; // floating-point JUMP
def ReadFJump         : SchedRead;  // floating-point JUMP
//===----------------------------------------------------------------------===//
// BCEQZ                            Sched<[WriteFJump, ReadFJump]>
// BCNEZ                            Sched<[WriteFJump, ReadFJump]>
//===----------------------------------------------------------------------===//

def WriteFLD          : SchedWrite; // Load  reg+imm
def WriteFST          : SchedWrite; // Store reg+imm
def WriteFLDX         : SchedWrite; // Load  reg+reg
def WriteFSTX         : SchedWrite; // Store reg+imm
def ReadFSTData       : SchedRead;  // Load / Store FSTData
def ReadFMemBase      : SchedRead;  // Load / Store FMemBase
def ReadFMemOffset    : SchedRead;  // Load / Store FMemOffset
//===----------------------------------------------------------------------===//
// FLD.S   FLD.D      Sched<[WriteFLD, ReadFMemBase]>
// FST.S   FST.D      Sched<[WriteFST, ReadFSTData, ReadFMemBase]>
// FLDX.S  FLDX.D     Sched<[WriteFLDX, ReadFMemBase, ReadFMemOffset]>
// FSTX.S  FSTX.D     Sched<[WriteFSTX, ReadFSTData, ReadFMemBase, ReadFMemOffset]>
//===----------------------------------------------------------------------===//

def WriteFLDGT        : SchedWrite; // 64-bit FLDGT operations
def WriteFLDLE        : SchedWrite; // 64-bit FLDLE operations
def WriteFSTGT        : SchedWrite; // 64-bit FSTGT operations
def WriteFSTLE        : SchedWrite; // 64-bit FSTLE operations
def ReadFMemCmp       : SchedRead;  // FLDGT FLDLE FSTGT FSTLE MemCmp operaitons
//===----------------------------------------------------------------------===//
// FLDGT.S  FLDGT.D   Sched<[WriteFLDGT, ReadFMemBase, ReadFMemCmp]>
// FLDLE.S  FLDLE.D   Sched<[WriteFLDLE, ReadFMemBase, ReadFMemCmp]>
// FSTGT.S  FSTGT.D   Sched<[WriteFSTGT, ReadFSTData, ReadFMemBase, ReadFMemCmp]>
// FSTLE.S  FSTLE.D   Sched<[WriteFSTLE, ReadFSTData, ReadFMemBase, ReadFMemCmp]>
//===----------------------------------------------------------------------===//

// Atomic Instruction is unsupported scheduling.
def WriteAtomic       : SchedWrite; // Atomic write operations
def ReadAtomic        : SchedRead;  // Atomic read  operations

// Pseudo Instruction is unsupported scheduling.
def WritePseudo       : SchedWrite; // Pseudo write operations
def ReadPseudo        : SchedRead;  // Pseudo read  operations

// Others Instruction is unsupported scheduling.
def WriteOthers       : SchedWrite; // Others write operations
def ReadOthers        : SchedRead;  // Others read  operations

// Include the scheduler resources for other instruction extensions.
include "LoongArchScheduleV.td"
