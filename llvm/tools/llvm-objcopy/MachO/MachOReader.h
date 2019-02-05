//===- MachOReader.h --------------------------------------------*- C++ -*-===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MachOObjcopy.h"
#include "Object.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Object/MachO.h"
#include <memory>

namespace llvm {
namespace objcopy {
namespace macho {

// The hierarchy of readers is responsible for parsing different inputs:
// raw binaries and regular MachO object files.
class Reader {
public:
  virtual ~Reader(){};
  virtual std::unique_ptr<Object> create() const = 0;
};

class MachOReader : public Reader {
  const object::MachOObjectFile &MachOObj;

  void readHeader(Object &O) const;
  void readLoadCommands(Object &O) const;
  void readSymbolTable(Object &O) const;
  void readStringTable(Object &O) const;
  void readRebaseInfo(Object &O) const;
  void readBindInfo(Object &O) const;
  void readWeakBindInfo(Object &O) const;
  void readLazyBindInfo(Object &O) const;
  void readExportInfo(Object &O) const;

public:
  explicit MachOReader(const object::MachOObjectFile &Obj) : MachOObj(Obj) {}

  std::unique_ptr<Object> create() const override;
};

} // end namespace macho
} // end namespace objcopy
} // end namespace llvm
