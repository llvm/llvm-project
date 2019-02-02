//===- MachOObjcopy.h -------------------------------------------*- C++ -*-===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_OBJCOPY_MACHOOBJCOPY_H
#define LLVM_TOOLS_OBJCOPY_MACHOOBJCOPY_H

namespace llvm {
class Error;

namespace object {
class MachOObjectFile;
class MachOUniversalBinary;
} // end namespace object

namespace objcopy {
struct CopyConfig;
class Buffer;

namespace macho {
Error executeObjcopyOnBinary(const CopyConfig &Config,
                             object::MachOObjectFile &In, Buffer &Out);
} // end namespace macho
} // end namespace objcopy
} // end namespace llvm

#endif // LLVM_TOOLS_OBJCOPY_MACHOOBJCOPY_H
