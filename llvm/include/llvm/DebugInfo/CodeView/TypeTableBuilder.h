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
  TypeIndex writeModifier(const ModifierRecord &Record);
  TypeIndex writeProcedure(const ProcedureRecord &Record);
  TypeIndex writeMemberFunction(const MemberFunctionRecord &Record);
  TypeIndex writeArgList(const ArgListRecord &Record);
  TypeIndex writePointer(const PointerRecord &Record);
  TypeIndex writeArray(const ArrayRecord &Record);
  TypeIndex writeClass(const ClassRecord &Record);
  TypeIndex writeUnion(const UnionRecord &Record);
  TypeIndex writeEnum(const EnumRecord &Record);
  TypeIndex writeBitField(const BitFieldRecord &Record);
  TypeIndex writeVFTableShape(const VFTableShapeRecord &Record);
  TypeIndex writeStringId(const StringIdRecord &Record);
  TypeIndex writeVFTable(const VFTableRecord &Record);
  TypeIndex writeUdtSourceLine(const UdtSourceLineRecord &Record);
  TypeIndex writeUdtModSourceLine(const UdtModSourceLineRecord &Record);
  TypeIndex writeFuncId(const FuncIdRecord &Record);
  TypeIndex writeMemberFuncId(const MemberFuncIdRecord &Record);
  TypeIndex writeBuildInfo(const BuildInfoRecord &Record);
  TypeIndex writeMethodOverloadList(const MethodOverloadListRecord &Record);
  TypeIndex writeTypeServer2(const TypeServer2Record &Record);

  TypeIndex writeFieldList(FieldListRecordBuilder &FieldList);

  TypeIndex writeRecord(TypeRecordBuilder &builder);

  virtual TypeIndex writeRecord(llvm::StringRef record) = 0;
};
}
}

#endif
