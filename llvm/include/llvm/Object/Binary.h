//===- Binary.h - A generic binary file -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Binary class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_BINARY_H
#define LLVM_OBJECT_BINARY_H

#include "llvm/ADT/OwningPtr.h"
#include "llvm/Object/Error.h"

namespace llvm {

class MemoryBuffer;
class StringRef;

namespace object {

class Binary {
private:
  Binary() LLVM_DELETED_FUNCTION;
  Binary(const Binary &other) LLVM_DELETED_FUNCTION;

  unsigned int TypeID;

protected:
  MemoryBuffer *Data;

  Binary(unsigned int Type, MemoryBuffer *Source);

  enum {
    ID_Archive,
    // Object and children.
    ID_StartObjects,
    ID_COFF,
    ID_ELF32L, // ELF 32-bit, little endian
    ID_ELF32B, // ELF 32-bit, big endian
    ID_ELF64L, // ELF 64-bit, little endian
    ID_ELF64B, // ELF 64-bit, big endian
    ID_MachO,
    ID_EndObjects
  };

  static inline unsigned int getELFType(bool isLittleEndian, bool is64Bits) {
    if (isLittleEndian)
      return is64Bits ? ID_ELF64L : ID_ELF32L;
    else
      return is64Bits ? ID_ELF64B : ID_ELF32B;
  }

public:
  virtual ~Binary();

  StringRef getData() const;
  StringRef getFileName() const;

  // Cast methods.
  unsigned int getType() const { return TypeID; }

  // Convenience methods
  bool isObject() const {
    return TypeID > ID_StartObjects && TypeID < ID_EndObjects;
  }

  bool isArchive() const {
    return TypeID == ID_Archive;
  }

  bool isELF() const {
    return TypeID >= ID_ELF32L && TypeID <= ID_ELF64B;
  }

  bool isMachO() const {
    return TypeID == ID_MachO;
  }

  bool isCOFF() const {
    return TypeID == ID_COFF;
  }
};

/// @brief Create a Binary from Source, autodetecting the file type.
///
/// @param Source The data to create the Binary from. Ownership is transferred
///        to Result if successful. If an error is returned, Source is destroyed
///        by createBinary before returning.
/// @param Result A pointer to the resulting Binary if no error occured.
error_code createBinary(MemoryBuffer *Source, OwningPtr<Binary> &Result);

error_code createBinary(StringRef Path, OwningPtr<Binary> &Result);

}
}

#endif
