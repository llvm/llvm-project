//===-- AMDGPUAsmUtils.h - AsmParser/InstPrinter common ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPUASMUTILS_H
#define LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPUASMUTILS_H

namespace llvm {
namespace AMDGPU {
namespace SendMsg { // Symbolic names for the sendmsg(...) syntax.

extern const char* const IdSymbolic[];
extern const char* const OpSysSymbolic[];
extern const char* const OpGsSymbolic[];

} // namespace SendMsg

namespace Hwreg { // Symbolic names for the hwreg(...) syntax.

extern const char* const IdSymbolic[];

} // namespace Hwreg

namespace Swizzle { // Symbolic names for the swizzle(...) syntax.

extern const char* const IdSymbolic[];

} // namespace Swizzle

namespace VGPRIndexMode { // Symbolic names for the gpr_idx(...) syntax.

extern const char* const IdSymbolic[];

} // namespace VGPRIndexMode

} // namespace AMDGPU
} // namespace llvm

#endif
