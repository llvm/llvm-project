//===- ObjCopy.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJCOPY_OBJCOPY_H
#define LLVM_OBJCOPY_OBJCOPY_H

#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"

namespace llvm {
class raw_ostream;

namespace object {
class Archive;
class Binary;
} // end namespace object

namespace objcopy {
class MultiFormatConfig;

/// Returns the format name of \p B if it is an ObjectFile, or "" otherwise.
LLVM_ABI StringRef getObjectFormatName(const object::Binary &B);

/// Prints information about the input and output files involved in a copy
/// operation to stdout.
LLVM_ABI void printCopyMessage(StringRef InPath, StringRef InFormatName,
                               StringRef OutPath, StringRef OutFormatName);

/// Applies the transformations described by \p Config to
/// each member in archive \p Ar.
/// Writes a result in a file specified by \p Config.OutputFilename.
/// \returns any Error encountered whilst performing the operation.
LLVM_ABI Error executeObjcopyOnArchive(const MultiFormatConfig &Config,
                                       const object::Archive &Ar);

/// Applies the transformations described by \p Config to \p In and writes
/// the result into \p Out. This function does the dispatch based on the
/// format of the input binary (COFF, ELF, MachO or wasm).
/// \returns any Error encountered whilst performing the operation.
LLVM_ABI Error executeObjcopyOnBinary(const MultiFormatConfig &Config,
                                      object::Binary &In, raw_ostream &Out);

} // end namespace objcopy
} // end namespace llvm

#endif // LLVM_OBJCOPY_OBJCOPY_H
