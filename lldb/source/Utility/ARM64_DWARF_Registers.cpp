//===-- ARM64_DWARF_Registers.c ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <string.h>

#include "ARM64_DWARF_Registers.h"

using namespace lldb;
using namespace lldb_private;
using namespace arm64_dwarf;

const char *arm64_dwarf::GetRegisterName(unsigned reg_num,
                                         bool altnernate_name) {
  if (altnernate_name) {
    switch (reg_num) {
    case fp:
      return "x29";
    case lr:
      return "x30";
    case sp:
      return "x31";
    default:
      break;
    }
    return nullptr;
  }

  switch (reg_num) {
  case x0:
    return "x0";
  case x1:
    return "x1";
  case x2:
    return "x2";
  case x3:
    return "x3";
  case x4:
    return "x4";
  case x5:
    return "x5";
  case x6:
    return "x6";
  case x7:
    return "x7";
  case x8:
    return "x8";
  case x9:
    return "x9";
  case x10:
    return "x10";
  case x11:
    return "x11";
  case x12:
    return "x12";
  case x13:
    return "x13";
  case x14:
    return "x14";
  case x15:
    return "x15";
  case x16:
    return "x16";
  case x17:
    return "x17";
  case x18:
    return "x18";
  case x19:
    return "x19";
  case x20:
    return "x20";
  case x21:
    return "x21";
  case x22:
    return "x22";
  case x23:
    return "x23";
  case x24:
    return "x24";
  case x25:
    return "x25";
  case x26:
    return "x26";
  case x27:
    return "x27";
  case x28:
    return "x28";
  case fp:
    return "fp";
  case lr:
    return "lr";
  case sp:
    return "sp";
  case pc:
    return "pc";
  case cpsr:
    return "cpsr";
  case v0:
    return "v0";
  case v1:
    return "v1";
  case v2:
    return "v2";
  case v3:
    return "v3";
  case v4:
    return "v4";
  case v5:
    return "v5";
  case v6:
    return "v6";
  case v7:
    return "v7";
  case v8:
    return "v8";
  case v9:
    return "v9";
  case v10:
    return "v10";
  case v11:
    return "v11";
  case v12:
    return "v12";
  case v13:
    return "v13";
  case v14:
    return "v14";
  case v15:
    return "v15";
  case v16:
    return "v16";
  case v17:
    return "v17";
  case v18:
    return "v18";
  case v19:
    return "v19";
  case v20:
    return "v20";
  case v21:
    return "v21";
  case v22:
    return "v22";
  case v23:
    return "v23";
  case v24:
    return "v24";
  case v25:
    return "v25";
  case v26:
    return "v26";
  case v27:
    return "v27";
  case v28:
    return "v28";
  case v29:
    return "v29";
  case v30:
    return "v30";
  case v31:
    return "v31";
  }
  return nullptr;
}

bool arm64_dwarf::GetRegisterInfo(unsigned reg_num, RegisterInfo &reg_info) {
  ::memset(&reg_info, 0, sizeof(RegisterInfo));
  ::memset(reg_info.kinds, LLDB_INVALID_REGNUM, sizeof(reg_info.kinds));

  if (reg_num >= x0 && reg_num <= pc) {
    reg_info.byte_size = 8;
    reg_info.format = eFormatHex;
    reg_info.encoding = eEncodingUint;
  } else if (reg_num >= v0 && reg_num <= v31) {
    reg_info.byte_size = 16;
    reg_info.format = eFormatVectorOfFloat32;
    reg_info.encoding = eEncodingVector;
  } else if (reg_num == cpsr) {
    reg_info.byte_size = 4;
    reg_info.format = eFormatHex;
    reg_info.encoding = eEncodingUint;
  } else {
    return false;
  }

  reg_info.name = arm64_dwarf::GetRegisterName(reg_num, false);
  reg_info.alt_name = arm64_dwarf::GetRegisterName(reg_num, true);
  reg_info.kinds[eRegisterKindDWARF] = reg_num;

  switch (reg_num) {
  case fp:
    reg_info.kinds[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_FP;
    break;
  case lr:
    reg_info.kinds[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_RA;
    break;
  case sp:
    reg_info.kinds[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_SP;
    break;
  case pc:
    reg_info.kinds[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_PC;
    break;
  default:
    break;
  }
  return true;
}
