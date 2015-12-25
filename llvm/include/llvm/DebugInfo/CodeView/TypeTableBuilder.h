//===- TypeTableBuilder.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_TYPETABLEBUILDER_H
#define LLVM_DEBUGINFO_CODEVIEW_TYPETABLEBUILDER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
namespace codeview {

class FieldListRecordBuilder;
class MethodListRecordBuilder;
class TypeRecordBuilder;

class TypeTableBuilder {
private:
  TypeTableBuilder(const TypeTableBuilder &) = delete;
  TypeTableBuilder &operator=(const TypeTableBuilder &) = delete;

protected:
  TypeTableBuilder();

public:
  virtual ~TypeTableBuilder();

public:
  TypeIndex writeModifier(const ModifierRecord &Record);
  TypeIndex writeProcedure(const ProcedureRecord &Record);
  TypeIndex writeMemberFunction(const MemberFunctionRecord &Record);
  TypeIndex writeArgumentList(const ArgumentListRecord &Record);
  TypeIndex writeRecord(TypeRecordBuilder &builder);
  TypeIndex writePointer(const PointerRecord &Record);
  TypeIndex writePointerToMember(const PointerToMemberRecord &Record);
  TypeIndex writeArray(const ArrayRecord &Record);
  TypeIndex writeAggregate(const AggregateRecord &Record);
  TypeIndex writeEnum(const EnumRecord &Record);
  TypeIndex writeBitField(const BitFieldRecord &Record);
  TypeIndex writeVirtualTableShape(const VirtualTableShapeRecord &Record);

  TypeIndex writeFieldList(FieldListRecordBuilder &FieldList);
  TypeIndex writeMethodList(MethodListRecordBuilder &MethodList);

private:
  virtual TypeIndex writeRecord(llvm::StringRef record) = 0;
};
}
}

#endif
