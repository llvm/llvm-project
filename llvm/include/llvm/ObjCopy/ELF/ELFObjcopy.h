//===- ELFObjcopy.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJCOPY_ELF_ELFOBJCOPY_H
#define LLVM_OBJCOPY_ELF_ELFOBJCOPY_H

#include "llvm/ADT/STLFunctionalExtras.h"

namespace llvm {
class Error;
class MemoryBuffer;
class raw_ostream;
class Twine;

namespace object {
class ELFObjectFileBase;
} // end namespace object

namespace objcopy {
struct CommonConfig;
struct ELFConfig;

namespace elf {
/// Apply the transformations described by \p Config and \p ELFConfig to
/// \p In, which must represent an IHex file, and writes the result
/// into \p Out. Warnings can be printed via \p WarningCallback.
/// \returns any Error encountered whilst performing the operation.
Error executeObjcopyOnIHex(const CommonConfig &Config,
                           const ELFConfig &ELFConfig, MemoryBuffer &In,
                           raw_ostream &Out,
                           function_ref<void(const Twine &)> WarningCallback);

/// Apply the transformations described by \p Config and \p ELFConfig to
/// \p In, which is treated as a raw binary input, and writes the result
/// into \p Out. Warnings can be printed via \p WarningCallback.
/// \returns any Error encountered whilst performing the operation.
Error executeObjcopyOnRawBinary(
    const CommonConfig &Config, const ELFConfig &ELFConfig, MemoryBuffer &In,
    raw_ostream &Out, function_ref<void(const Twine &)> WarningCallback);

/// Apply the transformations described by \p Config and \p ELFConfig to
/// \p In and writes the result into \p Out. Warnings can be printed via
/// \p WarningCallback.
/// \returns any Error encountered whilst performing the operation.
Error executeObjcopyOnBinary(const CommonConfig &Config,
                             const ELFConfig &ELFConfig,
                             object::ELFObjectFileBase &In, raw_ostream &Out,
                             function_ref<void(const Twine &)> WarningCallback);

} // end namespace elf
} // end namespace objcopy
} // end namespace llvm

#endif // LLVM_OBJCOPY_ELF_ELFOBJCOPY_H
