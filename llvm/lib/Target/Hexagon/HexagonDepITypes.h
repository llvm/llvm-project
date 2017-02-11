//===--- HexagonDepITypes.h -----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace llvm {
namespace HexagonII {
enum Type {
  TypeALU32_2op = 0,
  TypeALU32_3op = 1,
  TypeALU32_ADDI = 2,
  TypeALU64 = 3,
  TypeCJ = 4,
  TypeCOPROC_VMEM = 5,
  TypeCR = 7,
  TypeCVI_HIST = 10,
  TypeCVI_VA = 16,
  TypeCVI_VA_DV = 17,
  TypeCVI_VINLANESAT = 18,
  TypeCVI_VM_CUR_LD = 19,
  TypeCVI_VM_LD = 20,
  TypeCVI_VM_NEW_ST = 21,
  TypeCVI_VM_ST = 22,
  TypeCVI_VM_STU = 23,
  TypeCVI_VM_TMP_LD = 24,
  TypeCVI_VM_VP_LDU = 25,
  TypeCVI_VP = 26,
  TypeCVI_VP_VS = 27,
  TypeCVI_VS = 28,
  TypeCVI_VX = 30,
  TypeCVI_VX_DV = 31,
  TypeDUPLEX = 32,
  TypeENDLOOP = 33,
  TypeEXTENDER = 34,
  TypeJ = 35,
  TypeLD = 36,
  TypeM = 37,
  TypeMAPPING = 38,
  TypeNCJ = 39,
  TypePSEUDO = 40,
  TypeST = 41,
  TypeSUBINSN = 42,
  TypeS_2op = 43,
  TypeS_3op = 44,
  TypeV2LDST = 47,
  TypeV4LDST = 48
};
}
}
