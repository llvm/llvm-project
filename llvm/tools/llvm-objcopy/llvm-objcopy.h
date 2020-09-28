//===- llvm-objcopy.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_OBJCOPY_OBJCOPY_H
#define LLVM_TOOLS_OBJCOPY_OBJCOPY_H

#include "llvm/Support/Error.h"

namespace llvm {

struct NewArchiveMember;

namespace object {

class Archive;

} // end namespace object

namespace objcopy {
struct CopyConfig;
Expected<std::vector<NewArchiveMember>>
createNewArchiveMembers(CopyConfig &Config, const object::Archive &Ar);

} // end namespace objcopy
} // end namespace llvm

#endif // LLVM_TOOLS_OBJCOPY_OBJCOPY_H
