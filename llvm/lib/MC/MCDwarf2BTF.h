//===- MCDwarf2BTF.h ------------------------------------------ *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIB_MC_MCDWARF2BTF_H
#define LLVM_LIB_MC_MCDWARF2BTF_H

#include "llvm/MC/MCDwarf.h"

namespace llvm {

using FileContent = std::pair<std::string, std::vector<std::string>>;

class MCDwarf2BTF {
public:
  static void addFiles(MCObjectStreamer *MCOS, std::string &FileName,
    std::vector<FileContent> &Files);
  static void addLines(MCObjectStreamer *MCOS, StringRef &SectionName,
    std::vector<FileContent> &Files,
    const MCLineSection::MCDwarfLineEntryCollection &LineEntries);
  static void addDwarfLineInfo(MCObjectStreamer *MCOS);
};

}
#endif
