//===-- llvm-objdump.h ----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJDUMP_H
#define LLVM_OBJDUMP_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/MemoryObject.h"

namespace llvm {

extern cl::opt<std::string> TripleName;
extern cl::opt<std::string> ArchName;

// Various helper functions.
void DumpBytes(StringRef bytes);
void DisassembleInputMachO(StringRef Filename);

class StringRefMemoryObject : public MemoryObject {
  virtual void anchor();
  StringRef Bytes;
public:
  StringRefMemoryObject(StringRef bytes) : Bytes(bytes) {}

  uint64_t getBase() const { return 0; }
  uint64_t getExtent() const { return Bytes.size(); }

  int readByte(uint64_t Addr, uint8_t *Byte) const {
    if (Addr >= getExtent())
      return -1;
    *Byte = Bytes[Addr];
    return 0;
  }
};

}

#endif
