//===- SearchableTableEmitter.cpp - Generate efficiently searchable tables -==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits a generic array initialized by specified fields,
// together with companion index tables and lookup functions (binary search,
// currently).
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include <algorithm>
#include <sstream>
#include <string>
#include <vector>
using namespace llvm;

#define DEBUG_TYPE "searchable-table-emitter"

namespace {

class SearchableTableEmitter {
  RecordKeeper &Records;

public:
  SearchableTableEmitter(RecordKeeper &R) : Records(R) {}

  void run(raw_ostream &OS);

private:
  typedef std::pair<Init *, int> SearchTableEntry;

  int getAsInt(BitsInit *B) {
    return cast<IntInit>(B->convertInitializerTo(IntRecTy::get()))->getValue();
  }
  int getInt(Record *R, StringRef Field) {
    return getAsInt(R->getValueAsBitsInit(Field));
  }

  std::string primaryRepresentation(Init *I) {
    if (StringInit *SI = dyn_cast<StringInit>(I))
      return SI->getAsString();
    else if (BitsInit *BI = dyn_cast<BitsInit>(I))
      return "0x" + utohexstr(getAsInt(BI));
    else if (BitInit *BI = dyn_cast<BitInit>(I))
      return BI->getValue() ? "true" : "false";
    else if (CodeInit *CI = dyn_cast<CodeInit>(I)) {
      return CI->getValue();
    }
    PrintFatalError(SMLoc(),
                    "invalid field type, expected: string, bits, bit or code");
  }

  std::string searchRepresentation(Init *I) {
    std::string PrimaryRep = primaryRepresentation(I);
    if (!isa<StringInit>(I))
      return PrimaryRep;
    return StringRef(PrimaryRep).upper();
  }

  std::string searchableFieldType(Init *I) {
    if (isa<StringInit>(I))
      return "const char *";
    else if (BitsInit *BI = dyn_cast<BitsInit>(I)) {
      unsigned NumBits = BI->getNumBits();
      if (NumBits <= 8)
        NumBits = 8;
      else if (NumBits <= 16)
        NumBits = 16;
      else if (NumBits <= 32)
        NumBits = 32;
      else if (NumBits <= 64)
        NumBits = 64;
      else
        PrintFatalError(SMLoc(), "bitfield too large to search");
      return "uint" + utostr(NumBits) + "_t";
    }
    PrintFatalError(SMLoc(), "Unknown type to search by");
  }

  void emitMapping(Record *MappingDesc, raw_ostream &OS);
  void emitMappingEnum(std::vector<Record *> &Items, Record *InstanceClass,
                       raw_ostream &OS);
  void
  emitPrimaryTable(StringRef Name, std::vector<std::string> &FieldNames,
                   std::vector<std::string> &SearchFieldNames,
                   std::vector<std::vector<SearchTableEntry>> &SearchTables,
                   std::vector<Record *> &Items, raw_ostream &OS);
  void emitSearchTable(StringRef Name, StringRef Field,
                       std::vector<SearchTableEntry> &SearchTable,
                       raw_ostream &OS);
  void emitLookupDeclaration(StringRef Name, StringRef Field, Init *I,
                             raw_ostream &OS);
  void emitLookupFunction(StringRef Name, StringRef Field, Init *I,
                          raw_ostream &OS);
};

} // End anonymous namespace.

/// Emit an enum providing symbolic access to some preferred field from
/// C++.
void SearchableTableEmitter::emitMappingEnum(std::vector<Record *> &Items,
                                             Record *InstanceClass,
                                             raw_ostream &OS) {
  std::string EnumNameField = InstanceClass->getValueAsString("EnumNameField");
  std::string EnumValueField;
  if (!InstanceClass->isValueUnset("EnumValueField"))
    EnumValueField = InstanceClass->getValueAsString("EnumValueField");

  OS << "enum " << InstanceClass->getName() << "Values {\n";
  for (auto Item : Items) {
    OS << "  " << Item->getValueAsString(EnumNameField);
    if (EnumValueField != StringRef())
      OS << " = " << getInt(Item, EnumValueField);
    OS << ",\n";
  }
  OS << "};\n\n";
}

void SearchableTableEmitter::emitPrimaryTable(
    StringRef Name, std::vector<std::string> &FieldNames,
    std::vector<std::string> &SearchFieldNames,
    std::vector<std::vector<SearchTableEntry>> &SearchTables,
    std::vector<Record *> &Items, raw_ostream &OS) {
  OS << "const " << Name << " " << Name << "sList[] = {\n";

  for (auto Item : Items) {
    OS << "  { ";
    for (unsigned i = 0; i < FieldNames.size(); ++i) {
      OS << primaryRepresentation(Item->getValueInit(FieldNames[i]));
      if (i != FieldNames.size() - 1)
        OS << ", ";
    }
    OS << "},\n";
  }
  OS << "};\n\n";
}

void SearchableTableEmitter::emitSearchTable(
    StringRef Name, StringRef Field, std::vector<SearchTableEntry> &SearchTable,
    raw_ostream &OS) {
  OS << "const std::pair<" << searchableFieldType(SearchTable[0].first)
     << ", int> " << Name << "sBy" << Field << "[] = {\n";

  if (isa<BitsInit>(SearchTable[0].first)) {
    std::stable_sort(SearchTable.begin(), SearchTable.end(),
                     [this](const SearchTableEntry &LHS,
                            const SearchTableEntry &RHS) {
                       return getAsInt(cast<BitsInit>(LHS.first)) <
                              getAsInt(cast<BitsInit>(RHS.first));
                     });
  } else {
    std::stable_sort(SearchTable.begin(), SearchTable.end(),
                     [this](const SearchTableEntry &LHS,
                            const SearchTableEntry &RHS) {
                       return searchRepresentation(LHS.first) <
                              searchRepresentation(RHS.first);
                     });
  }

  for (auto Entry : SearchTable) {
    OS << "  { " << searchRepresentation(Entry.first) << ", " << Entry.second
       << " },\n";
  }
  OS << "};\n\n";
}

void SearchableTableEmitter::emitLookupFunction(StringRef Name, StringRef Field,
                                                Init *I, raw_ostream &OS) {
  bool IsIntegral = isa<BitsInit>(I);
  std::string FieldType = searchableFieldType(I);
  std::string PairType = "std::pair<" + FieldType + ", int>";

  // const SysRegs *lookupSysRegByName(const char *Name) {
  OS << "const " << Name << " *"
     << "lookup" << Name << "By" << Field;
  OS << "(" << (IsIntegral ? FieldType : "StringRef") << " " << Field
     << ") {\n";

  if (IsIntegral) {
    OS << "  auto CanonicalVal = " << Field << ";\n";
    OS << " " << PairType << " Val = {CanonicalVal, 0};\n";
  } else {
    // Make sure the result is null terminated because it's going via "char *".
    OS << "  std::string CanonicalVal = " << Field << ".upper();\n";
    OS << "  " << PairType << " Val = {CanonicalVal.c_str(), 0};\n";
  }

  OS << "  ArrayRef<" << PairType << "> Table(" << Name << "sBy" << Field
     << ");\n";
  OS << "  auto Idx = std::lower_bound(Table.begin(), Table.end(), Val";

  if (IsIntegral)
    OS << ");\n";
  else {
    OS << ",\n                              ";
    OS << "[](const " << PairType << " &LHS, const " << PairType
       << " &RHS) {\n";
    OS << "    return std::strcmp(LHS.first, RHS.first) < 0;\n";
    OS << "  });\n\n";
  }

  OS << "  if (Idx == Table.end() || CanonicalVal != Idx->first)\n";
  OS << "    return nullptr;\n";

  OS << "  return &" << Name << "sList[Idx->second];\n";
  OS << "}\n\n";
}

void SearchableTableEmitter::emitLookupDeclaration(StringRef Name,
                                                   StringRef Field, Init *I,
                                                   raw_ostream &OS) {
  bool IsIntegral = isa<BitsInit>(I);
  std::string FieldType = searchableFieldType(I);
  OS << "const " << Name << " *"
     << "lookup" << Name << "By" << Field;
  OS << "(" << (IsIntegral ? FieldType : "StringRef") << " " << Field
     << ");\n\n";
}

void SearchableTableEmitter::emitMapping(Record *InstanceClass,
                                         raw_ostream &OS) {
  const std::string &TableName = InstanceClass->getName();
  std::vector<Record *> Items = Records.getAllDerivedDefinitions(TableName);

  // Gather all the records we're going to need for this particular mapping.
  std::vector<std::vector<SearchTableEntry>> SearchTables;
  std::vector<std::string> SearchFieldNames;

  std::vector<std::string> FieldNames;
  for (const RecordVal &Field : InstanceClass->getValues()) {
    std::string FieldName = Field.getName();

    // Skip uninteresting fields: either built-in, special to us, or injected
    // template parameters (if they contain a ':').
    if (FieldName.find(':') != std::string::npos || FieldName == "NAME" ||
        FieldName == "SearchableFields" || FieldName == "EnumNameField" ||
        FieldName == "EnumValueField")
      continue;

    FieldNames.push_back(FieldName);
  }

  for (auto *Field : *InstanceClass->getValueAsListInit("SearchableFields")) {
    SearchTables.emplace_back();
    SearchFieldNames.push_back(Field->getAsUnquotedString());
  }

  int Idx = 0;
  for (Record *Item : Items) {
    for (unsigned i = 0; i < SearchFieldNames.size(); ++i) {
      Init *SearchVal = Item->getValueInit(SearchFieldNames[i]);
      SearchTables[i].emplace_back(SearchVal, Idx);
    }
    ++Idx;
  }

  OS << "#ifdef GET_" << StringRef(TableName).upper() << "_DECL\n";
  OS << "#undef GET_" << StringRef(TableName).upper() << "_DECL\n";

  // Next emit the enum containing the top-level names for use in C++ code if
  // requested
  if (!InstanceClass->isValueUnset("EnumNameField")) {
    emitMappingEnum(Items, InstanceClass, OS);
  }

  // And the declarations for the functions that will perform lookup.
  for (unsigned i = 0; i < SearchFieldNames.size(); ++i)
    emitLookupDeclaration(TableName, SearchFieldNames[i],
                          SearchTables[i][0].first, OS);

  OS << "#endif\n\n";

  OS << "#ifdef GET_" << StringRef(TableName).upper() << "_IMPL\n";
  OS << "#undef GET_" << StringRef(TableName).upper() << "_IMPL\n";

  // The primary data table contains all the fields defined for this map.
  emitPrimaryTable(TableName, FieldNames, SearchFieldNames, SearchTables, Items,
                   OS);

  // Indexes are sorted "{ Thing, PrimaryIdx }" arrays, so that a binary
  // search can be performed by "Thing".
  for (unsigned i = 0; i < SearchTables.size(); ++i) {
    emitSearchTable(TableName, SearchFieldNames[i], SearchTables[i], OS);
    emitLookupFunction(TableName, SearchFieldNames[i], SearchTables[i][0].first,
                       OS);
  }

  OS << "#endif\n";
}

void SearchableTableEmitter::run(raw_ostream &OS) {
  // Tables are defined to be the direct descendents of "SearchableEntry".
  Record *SearchableTable = Records.getClass("SearchableTable");
  for (auto &NameRec : Records.getClasses()) {
    Record *Class = NameRec.second.get();
    if (Class->getSuperClasses().size() != 1 ||
        !Class->isSubClassOf(SearchableTable))
      continue;
    emitMapping(Class, OS);
  }
}

namespace llvm {

void EmitSearchableTables(RecordKeeper &RK, raw_ostream &OS) {
  SearchableTableEmitter(RK).run(OS);
}

} // End llvm namespace.
