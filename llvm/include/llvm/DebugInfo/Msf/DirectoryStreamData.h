//===- DirectoryStreamData.h ---------------------------------- *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_MSF_DIRECTORYSTREAMDATA_H
#define LLVM_DEBUGINFO_MSF_DIRECTORYSTREAMDATA_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/DebugInfo/Msf/IMsfFile.h"
#include "llvm/DebugInfo/Msf/IMsfStreamData.h"
#include "llvm/Support/Endian.h"

namespace llvm {
namespace msf {
class IMsfFile;

class DirectoryStreamData : public IMsfStreamData {
public:
  DirectoryStreamData(uint32_t Length, ArrayRef<support::ulittle32_t> Blocks)
      : Length(Length), Blocks(Blocks) {}

  virtual uint32_t getLength() { return Length; }
  virtual llvm::ArrayRef<llvm::support::ulittle32_t> getStreamBlocks() {
    return Blocks;
  }

private:
  uint32_t Length;
  ArrayRef<support::ulittle32_t> Blocks;
};
}
}

#endif
