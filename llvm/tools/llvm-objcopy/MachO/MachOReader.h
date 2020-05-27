//===- MachOReader.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
  void setSymbolInRelocationInfo(Object &O) const;
  void readRebaseInfo(Object &O) const;
  void readBindInfo(Object &O) const;
  void readWeakBindInfo(Object &O) const;
  void readLazyBindInfo(Object &O) const;
  void readExportInfo(Object &O) const;
  void readDataInCodeData(Object &O) const;
  void readFunctionStartsData(Object &O) const;
  void readIndirectSymbolTable(Object &O) const;
  void readSwiftVersion(Object &O) const;

public:
  explicit MachOReader(const object::MachOObjectFile &Obj) : MachOObj(Obj) {}

  std::unique_ptr<Object> create() const override;
};

} // end namespace macho
} // end namespace objcopy
} // end namespace llvm
