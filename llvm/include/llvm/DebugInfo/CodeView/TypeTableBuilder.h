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

#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class StringRef;

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
  TypeIndex writeKnownType(const ModifierRecord &Record);
  TypeIndex writeKnownType(const ProcedureRecord &Record);
  TypeIndex writeKnownType(const MemberFunctionRecord &Record);
  TypeIndex writeKnownType(const ArgListRecord &Record);
  TypeIndex writeKnownType(const PointerRecord &Record);
  TypeIndex writeKnownType(const ArrayRecord &Record);
  TypeIndex writeKnownType(const ClassRecord &Record);
  TypeIndex writeKnownType(const UnionRecord &Record);
  TypeIndex writeKnownType(const EnumRecord &Record);
  TypeIndex writeKnownType(const BitFieldRecord &Record);
  TypeIndex writeKnownType(const VFTableShapeRecord &Record);
  TypeIndex writeKnownType(const StringIdRecord &Record);
  TypeIndex writeKnownType(const VFTableRecord &Record);
  TypeIndex writeKnownType(const UdtSourceLineRecord &Record);
  TypeIndex writeKnownType(const UdtModSourceLineRecord &Record);
  TypeIndex writeKnownType(const FuncIdRecord &Record);
  TypeIndex writeKnownType(const MemberFuncIdRecord &Record);
  TypeIndex writeKnownType(const BuildInfoRecord &Record);
  TypeIndex writeKnownType(const MethodOverloadListRecord &Record);
  TypeIndex writeKnownType(const TypeServer2Record &Record);

  TypeIndex writeFieldList(FieldListRecordBuilder &FieldList);

  TypeIndex writeRecord(TypeRecordBuilder &builder);

  virtual TypeIndex writeRecord(llvm::StringRef record) = 0;
};
}
}

#endif
