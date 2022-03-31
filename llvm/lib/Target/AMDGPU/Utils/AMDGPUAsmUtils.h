//===-- AMDGPUAsmUtils.h - AsmParser/InstPrinter common ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPUASMUTILS_H
#define LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPUASMUTILS_H

#include "SIDefines.h"

#include "llvm/ADT/StringRef.h"

namespace llvm {

class StringLiteral;
class MCSubtargetInfo;

namespace AMDGPU {

const int OPR_ID_UNKNOWN = -1;
const int OPR_ID_UNSUPPORTED = -2;

template <class T> struct CustomOperand {
  StringLiteral Name;
  int Encoding = 0;
  bool (*Cond)(T Context) = nullptr;
};

namespace SendMsg { // Symbolic names for the sendmsg(...) syntax.

extern const CustomOperand<const MCSubtargetInfo &> Msg[];
extern const int MSG_SIZE;

extern const char *const OpSysSymbolic[OP_SYS_LAST_];
extern const char *const OpGsSymbolic[OP_GS_LAST_];

} // namespace SendMsg

namespace Hwreg { // Symbolic names for the hwreg(...) syntax.

extern const CustomOperand<const MCSubtargetInfo &> Opr[];
extern const int OPR_SIZE;

} // namespace Hwreg

namespace MTBUFFormat {

extern StringLiteral const DfmtSymbolic[];
extern StringLiteral const NfmtSymbolicGFX10[];
extern StringLiteral const NfmtSymbolicSICI[];
extern StringLiteral const NfmtSymbolicVI[];
extern StringLiteral const UfmtSymbolic[];
extern unsigned const DfmtNfmt2UFmt[];

} // namespace MTBUFFormat

namespace Swizzle { // Symbolic names for the swizzle(...) syntax.

extern const char* const IdSymbolic[];

} // namespace Swizzle

namespace VGPRIndexMode { // Symbolic names for the gpr_idx(...) syntax.

extern const char* const IdSymbolic[];

} // namespace VGPRIndexMode

} // namespace AMDGPU
} // namespace llvm

#endif
