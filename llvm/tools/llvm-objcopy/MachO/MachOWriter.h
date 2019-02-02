//===- MachOWriter.h --------------------------------------------*- C++ -*-===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../Buffer.h"
#include "MachOObjcopy.h"
#include "Object.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Object/MachO.h"

namespace llvm {
class Error;

namespace objcopy {
namespace macho {

class MachOWriter {
  Object &O;
  bool Is64Bit;
  bool IsLittleEndian;
  Buffer &B;

  size_t headerSize() const;
  size_t loadCommandsSize() const;
  size_t symTableSize() const;
  size_t strTableSize() const;

  void writeHeader();
  void writeLoadCommands();
  void writeSections();
  void writeSymbolTable();
  void writeStringTable();
  void writeRebaseInfo();
  void writeBindInfo();
  void writeWeakBindInfo();
  void writeLazyBindInfo();
  void writeExportInfo();
  void writeTail();

public:
  MachOWriter(Object &O, bool Is64Bit, bool IsLittleEndian, Buffer &B)
      : O(O), Is64Bit(Is64Bit), IsLittleEndian(IsLittleEndian), B(B) {}

  size_t totalSize() const;
  Error write();
};

} // end namespace macho
} // end namespace objcopy
} // end namespace llvm
