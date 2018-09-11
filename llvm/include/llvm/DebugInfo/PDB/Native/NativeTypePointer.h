//===- NativeTypePointer.h - info about pointer type ------------------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_NATIVE_NATIVETYPEPOINTER_H
#define LLVM_DEBUGINFO_PDB_NATIVE_NATIVETYPEPOINTER_H

#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/PDB/Native/NativeRawSymbol.h"
#include "llvm/DebugInfo/PDB/Native/NativeSession.h"

namespace llvm {
namespace pdb {

class NativeTypePointer : public NativeRawSymbol {
public:
  NativeTypePointer(NativeSession &Session, SymIndexId Id, codeview::CVType CV);
  NativeTypePointer(NativeSession &Session, SymIndexId Id,
                    codeview::PointerRecord PR);
  ~NativeTypePointer() override;

  void dump(raw_ostream &OS, int Indent) const override;
  std::unique_ptr<NativeRawSymbol> clone() const override;

  bool isConstType() const override;
  uint64_t getLength() const override;
  bool isReference() const override;
  bool isRValueReference() const override;
  bool isPointerToDataMember() const override;
  bool isPointerToMemberFunction() const override;
  uint32_t getTypeId() const override;
  bool isRestrictedType() const override;
  bool isVolatileType() const override;
  bool isUnalignedType() const override;

protected:
  codeview::PointerRecord Record;
};

} // namespace pdb
} // namespace llvm

#endif // LLVM_DEBUGINFO_PDB_NATIVE_NATIVETYPEPOINTER_H