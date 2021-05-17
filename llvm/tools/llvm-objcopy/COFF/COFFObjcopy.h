//===- COFFObjcopy.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_OBJCOPY_COFFOBJCOPY_H
#define LLVM_TOOLS_OBJCOPY_COFFOBJCOPY_H

namespace llvm {
class Error;
class raw_ostream;

namespace object {
class COFFObjectFile;
} // end namespace object

namespace objcopy {
struct CopyConfig;

namespace coff {
Error executeObjcopyOnBinary(const CopyConfig &Config,
                             object::COFFObjectFile &In, raw_ostream &Out);

} // end namespace coff
} // end namespace objcopy
} // end namespace llvm

#endif // LLVM_TOOLS_OBJCOPY_COFFOBJCOPY_H
