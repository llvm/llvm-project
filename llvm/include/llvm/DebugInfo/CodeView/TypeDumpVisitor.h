//===-- TypeDumpVisitor.h - CodeView type info dumper -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_TYPEDUMPVISITOR_H
#define LLVM_DEBUGINFO_CODEVIEW_TYPEDUMPVISITOR_H

#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/CodeView/CVPacks.h"
#include "llvm/DebugInfo/CodeView/CVRecord.h"
#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/TypeVisitorCallbacks.h"
#include "llvm/Support/Compiler.h"
#include <cstdint>
#include <utility>
#include <vector>

namespace llvm {
class ScopedPrinter;

namespace codeview {
class TypeIndex;
struct CVMemberRecord;
struct MemberAttributes;

class TypeCollection;

/// Dumper for CodeView type streams found in COFF object files and PDB files.
class LLVM_ABI TypeDumpVisitor : public TypeVisitorCallbacks {
public:
  TypeDumpVisitor(TypeCollection &TpiTypes, ScopedPrinter *W,
                  bool PrintRecordBytes)
      : W(W), PrintRecordBytes(PrintRecordBytes), TpiTypes(TpiTypes) {}

  /// When dumping types from an IPI stream in a PDB, a type index may refer to
  /// a type or an item ID. The dumper will lookup the "name" of the index in
  /// the item database if appropriate. If ItemDB is null, it will use TypeDB,
  /// which is correct when dumping types from an object file (/Z7).
  void setIpiTypes(TypeCollection &Types) { IpiTypes = &Types; }

  void printTypeIndex(StringRef FieldName, TypeIndex TI) const;

  void printItemIndex(StringRef FieldName, TypeIndex TI) const;

  /// Action to take on unknown types. By default, they are ignored.
  Error visitUnknownType(CVType &Record) override;
  Error visitUnknownMember(CVMemberRecord &Record) override;

  /// Paired begin/end actions for all types. Receives all record data,
  /// including the fixed-length record prefix.
  Error visitTypeBegin(CVType &Record) override;
  Error visitTypeBegin(CVType &Record, TypeIndex Index) override;
  Error visitTypeEnd(CVType &Record) override;
  Error visitMemberBegin(CVMemberRecord &Record) override;
  Error visitMemberEnd(CVMemberRecord &Record) override;

#define TYPE_RECORD(EnumName, EnumVal, Name)                                   \
  Error visitKnownRecord(CVType &CVR, Name##Record &Record) override;
#define MEMBER_RECORD(EnumName, EnumVal, Name)                                 \
  Error visitKnownMember(CVMemberRecord &CVR, Name##Record &Record) override;
#define TYPE_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#define MEMBER_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#include "llvm/DebugInfo/CodeView/CodeViewTypes.def"

private:
  void printMemberAttributes(MemberAttributes Attrs);
  void printMemberAttributes(MemberAccess Access, MethodKind Kind,
                             MethodOptions Options);

  /// Get the database of indices for the stream that we are dumping. If ItemDB
  /// is set, then we must be dumping an item (IPI) stream. This will also
  /// always get the appropriate DB for printing item names.
  TypeCollection &getSourceTypes() const {
    return IpiTypes ? *IpiTypes : TpiTypes;
  }

  /// The index of the next inline record of a packed object's type stream.
  Expected<TypeIndex> nextCVPackInlineIndex();

  ScopedPrinter *W;

  bool PrintRecordBytes = false;

  TypeCollection &TpiTypes;
  TypeCollection *IpiTypes = nullptr;

  /// The type stream of a packed object starts with CodeView pack records,
  /// which consume no type index; its range map then gives the index of each
  /// inline record.
  bool HasCVPacks = false;
  cvpack::RangeMap CVPackMap;
  /// Once the inline records begin: the runs of inline records of the map, as
  /// (first index, count), and the position of the next inline record.
  bool CVPackRunsReady = false;
  std::vector<std::pair<uint64_t, uint64_t>> CVPackInlineRuns;
  size_t CVPackRun = 0;
  uint64_t CVPackInRun = 0;
};

} // end namespace codeview
} // end namespace llvm

#endif
