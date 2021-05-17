//===- MachOObjcopy.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_OBJCOPY_MACHOOBJCOPY_H
#define LLVM_TOOLS_OBJCOPY_MACHOOBJCOPY_H

namespace llvm {
class Error;
class raw_ostream;

namespace object {
class MachOObjectFile;
class MachOUniversalBinary;
} // end namespace object

namespace objcopy {
struct CopyConfig;
class Buffer;

namespace macho {
Error executeObjcopyOnBinary(const CopyConfig &Config,
                             object::MachOObjectFile &In, raw_ostream &Out);

Error executeObjcopyOnMachOUniversalBinary(
    CopyConfig &Config, const object::MachOUniversalBinary &In,
    raw_ostream &Out);

} // end namespace macho
} // end namespace objcopy
} // end namespace llvm

#endif // LLVM_TOOLS_OBJCOPY_MACHOOBJCOPY_H
