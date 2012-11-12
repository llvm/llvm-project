//===-- RuntimeDyldMachO.h - Run-time dynamic linker for MC-JIT ---*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// MachO support for MC-JIT runtime dynamic linker.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_RUNTIME_DYLD_MACHO_H
#define LLVM_RUNTIME_DYLD_MACHO_H

#include "llvm/ADT/IndexedMap.h"
#include "llvm/Object/MachOObject.h"
#include "llvm/Support/Format.h"
#include "RuntimeDyldImpl.h"

using namespace llvm;
using namespace llvm::object;


namespace llvm {
class RuntimeDyldMachO : public RuntimeDyldImpl {
protected:
  bool resolveI386Relocation(uint8_t *LocalAddress,
                             uint64_t FinalAddress,
                             uint64_t Value,
                             bool isPCRel,
                             unsigned Type,
                             unsigned Size,
                             int64_t Addend);
  bool resolveX86_64Relocation(uint8_t *LocalAddress,
                               uint64_t FinalAddress,
                               uint64_t Value,
                               bool isPCRel,
                               unsigned Type,
                               unsigned Size,
                               int64_t Addend);
  bool resolveARMRelocation(uint8_t *LocalAddress,
                            uint64_t FinalAddress,
                            uint64_t Value,
                            bool isPCRel,
                            unsigned Type,
                            unsigned Size,
                            int64_t Addend);

  virtual void processRelocationRef(const ObjRelocationInfo &Rel,
                                    ObjectImage &Obj,
                                    ObjSectionToIDMap &ObjSectionToID,
                                    const SymbolTableMap &Symbols,
                                    StubMap &Stubs);

public:
  virtual void resolveRelocation(const SectionEntry &Section,
                                 uint64_t Offset,
                                 uint64_t Value,
                                 uint32_t Type,
                                 int64_t Addend);

  RuntimeDyldMachO(RTDyldMemoryManager *mm) : RuntimeDyldImpl(mm) {}

  bool isCompatibleFormat(const ObjectBuffer *Buffer) const;
};

} // end namespace llvm

#endif
