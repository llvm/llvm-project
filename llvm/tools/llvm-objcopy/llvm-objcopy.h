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
#include "llvm/Support/raw_ostream.h"

namespace llvm {

struct NewArchiveMember;

namespace object {

class Archive;

} // end namespace object

namespace objcopy {
struct CopyConfig;
Expected<std::vector<NewArchiveMember>>
createNewArchiveMembers(CopyConfig &Config, const object::Archive &Ar);

/// A writeToFile helper creates an output stream, based on the specified
/// \p OutputFileName: std::outs for the "-", raw_null_ostream for
/// the "/dev/null", temporary file in the same directory as the final output
/// file for other names. The final output file is atomically replaced with
/// the temporary file after \p Write handler is finished.
Error writeToFile(StringRef OutputFileName,
                  std::function<Error(raw_ostream &)> Write);

} // end namespace objcopy
} // end namespace llvm

#endif // LLVM_TOOLS_OBJCOPY_OBJCOPY_H
