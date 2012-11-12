//===---- ObjectImage.h - Format independent executuable object image -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares a file format independent ObjectImage class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_OBJECTIMAGE_H
#define LLVM_EXECUTIONENGINE_OBJECTIMAGE_H

#include "llvm/Object/ObjectFile.h"
#include "llvm/ExecutionEngine/ObjectBuffer.h"

namespace llvm {


/// ObjectImage - A container class that represents an ObjectFile that has been
/// or is in the process of being loaded into memory for execution.
class ObjectImage {
  ObjectImage() LLVM_DELETED_FUNCTION;
  ObjectImage(const ObjectImage &other) LLVM_DELETED_FUNCTION;

protected:
  OwningPtr<ObjectBuffer> Buffer;

public:
  ObjectImage(ObjectBuffer *Input) : Buffer(Input) {}
  virtual ~ObjectImage() {}

  virtual object::symbol_iterator begin_symbols() const = 0;
  virtual object::symbol_iterator end_symbols() const = 0;

  virtual object::section_iterator begin_sections() const = 0;
  virtual object::section_iterator end_sections() const  = 0;

  virtual /* Triple::ArchType */ unsigned getArch() const = 0;

  // Subclasses can override these methods to update the image with loaded
  // addresses for sections and common symbols
  virtual void updateSectionAddress(const object::SectionRef &Sec,
                                    uint64_t Addr) = 0;
  virtual void updateSymbolAddress(const object::SymbolRef &Sym,
                                   uint64_t Addr) = 0;

  virtual StringRef getData() const = 0;

  // Subclasses can override these methods to provide JIT debugging support
  virtual void registerWithDebugger() = 0;
  virtual void deregisterWithDebugger() = 0;
};

} // end namespace llvm

#endif // LLVM_RUNTIMEDYLD_OBJECT_IMAGE_H

