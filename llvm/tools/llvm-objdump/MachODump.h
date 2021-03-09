//===-- MachODump.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_OBJDUMP_MACHODUMP_H
#define LLVM_TOOLS_LLVM_OBJDUMP_MACHODUMP_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"

namespace llvm {

class Error;
class StringRef;

namespace object {
class MachOObjectFile;
class MachOUniversalBinary;
class ObjectFile;
class RelocationRef;
} // namespace object

namespace objdump {

// MachO specific options
extern cl::OptionCategory MachOCat;
extern cl::opt<bool> Bind;
extern cl::opt<bool> DataInCode;
extern cl::opt<bool> DylibsUsed;
extern cl::opt<bool> DylibId;
extern cl::opt<bool> ExportsTrie;
extern cl::opt<bool> FirstPrivateHeader;
extern cl::opt<bool> FunctionStarts;
extern cl::opt<bool> IndirectSymbols;
extern cl::opt<bool> InfoPlist;
extern cl::opt<bool> LazyBind;
extern cl::opt<bool> LinkOptHints;
extern cl::opt<bool> ObjcMetaData;
extern cl::opt<bool> Rebase;
extern cl::opt<bool> UniversalHeaders;
extern cl::opt<bool> WeakBind;

Error getMachORelocationValueString(const object::MachOObjectFile *Obj,
                                    const object::RelocationRef &RelRef,
                                    llvm::SmallVectorImpl<char> &Result);

void parseInputMachO(StringRef Filename);
void parseInputMachO(object::MachOUniversalBinary *UB);

void printMachOUnwindInfo(const object::MachOObjectFile *O);
void printMachOFileHeader(const object::ObjectFile *O);
void printMachOLoadCommands(const object::ObjectFile *O);

void printExportsTrie(const object::ObjectFile *O);
void printRebaseTable(object::ObjectFile *O);
void printBindTable(object::ObjectFile *O);
void printLazyBindTable(object::ObjectFile *O);
void printWeakBindTable(object::ObjectFile *O);

} // namespace objdump
} // namespace llvm

#endif
