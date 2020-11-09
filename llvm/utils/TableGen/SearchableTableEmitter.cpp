//===- SearchableTableEmitter.cpp - Generate efficiently searchable tables -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits a generic array initialized by specified fields,
// together with companion index tables and lookup functions (binary search,
// currently).
//
//===----------------------------------------------------------------------===//

#include "CodeGenIntrinsics.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include <algorithm>
#include <set>
#include <string>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "searchable-table-emitter"

namespace {

struct GenericTable;

int getAsInt(Init *B) {
  return cast<IntInit>(B->convertInitializerTo(IntRecTy::get()))->getValue();
}
int getInt(Record *R, StringRef Field) {
  return getAsInt(R->getValueInit(Field));
}

struct GenericEnum {
  using Entry = std::pair<StringRef, int64_t>;

  std::string Name;
  Record *Class = nullptr;
  std::string PreprocessorGuard;
  std::vector<std::unique_ptr<Entry>> Entries;
  DenseMap<Record *, Entry *> EntryMap;
};

struct GenericField {
  std::string Name;
  RecTy *RecType = nullptr;
  bool IsIntrinsic = false;
  bool IsInstruction = false;
  GenericEnum *Enum = nullptr;

  GenericField(StringRef Name) : Name(std::string(Name)) {}
};

struct SearchIndex {
  std::string Name;
  SMLoc Loc; // Source location of PrimaryKey or Key field definition.
  SmallVector<GenericField, 1> Fields;
  bool EarlyOut = false;
};

struct GenericTable {
  std::string Name;
  ArrayRef<SMLoc> Locs; // Source locations from the Record instance.
  std::string PreprocessorGuard;
  std::string CppTypeName;
  SmallVector<GenericField, 2> Fields;
  std::vector<Record *> Entries;

  std::unique_ptr<SearchIndex> PrimaryKey;
  SmallVector<std::unique_ptr<SearchIndex>, 2> Indices;

  const GenericField *getFieldByName(StringRef Name) const {
    for (const auto &Field : Fields) {
      if (Name == Field.Name)
        return &Field;
    }
    return nullptr;
  }
};

class SearchableTableEmitter {
  RecordKeeper &Records;
  DenseMap<Init *, std::unique_ptr<CodeGenIntrinsic>> Intrinsics;
  std::vector<std::unique_ptr<GenericEnum>> Enums;
  DenseMap<Record *, GenericEnum *> EnumMap;
  std::set<std::string> PreprocessorGuards;

public:
  SearchableTableEmitter(RecordKeeper &R) : Records(R) {}

  void run(raw_ostream &OS);

private:
  typedef std::pair<Init *, int> SearchTableEntry;

  enum TypeContext {
    TypeInStaticStruct,
    TypeInTempStruct,
    TypeInArgument,
  };

  std::string primaryRepresentation(SMLoc Loc, const GenericField &Field,
                                    Init *I) {
    if (StringInit *SI = dyn_cast<StringInit>(I))
      return SI->getAsString();
    else if (BitsInit *BI = dyn_cast<BitsInit>(I))
      return "0x" + utohexstr(getAsInt(BI));
    else if (BitInit *BI = dyn_cast<BitInit>(I))
      return BI->getValue() ? "true" : "false";
    else if (CodeInit *CI = dyn_cast<CodeInit>(I))
      return std::string(CI->getValue());
    else if (Field.IsIntrinsic)
      return "Intrinsic::" + getIntrinsic(I).EnumName;
    else if (Field.IsInstruction)
      return I->getAsString();
    else if (Field.Enum) {
      auto *Entry = Field.Enum->EntryMap[cast<DefInit>(I)->getDef()];
      if (!Entry)
        PrintFatalError(Loc,
                        Twine("Entry for field '") + Field.Name + "' is null");
      return std::string(Entry->first);
    }
    PrintFatalError(Loc, Twine("invalid field type for field '") + Field.Name + 
                             "'; expected: bit, bits, string, or code");
  }

  bool isIntrinsic(Init *I) {
    if (DefInit *DI = dyn_cast<DefInit>(I))
      return DI->getDef()->isSubClassOf("Intrinsic");
    return false;
  }

  CodeGenIntrinsic &getIntrinsic(Init *I) {
    std::unique_ptr<CodeGenIntrinsic> &Intr = Intrinsics[I];
    if (!Intr)
      Intr = std::make_unique<CodeGenIntrinsic>(cast<DefInit>(I)->getDef(),
                                                std::vector<Record *>());
    return *Intr;
  }

  bool compareBy(Record *LHS, Record *RHS, const SearchIndex &Index);

  bool isIntegral(Init *I) {
    return isa<BitsInit>(I) || isa<CodeInit>(I) || isIntrinsic(I);
  }

  std::string searchableFieldType(const GenericTable &Table,
                                  const SearchIndex &Index,
                                  const GenericField &Field, TypeContext Ctx) {
    if (isa<StringRecTy>(Field.RecType)) {
      if (Ctx == TypeInStaticStruct)
        return "const char *";
      if (Ctx == TypeInTempStruct)
        return "std::string";
      return "StringRef";
    } else if (BitsRecTy *BI = dyn_cast<BitsRecTy>(Field.RecType)) {
      unsigned NumBits = BI->getNumBits();
      if (NumBits <= 8)
        return "uint8_t";
      if (NumBits <= 16)
        return "uint16_t";
      if (NumBits <= 32)
        return "uint32_t";
      if (NumBits <= 64)
        return "uint64_t";
      PrintFatalError(Index.Loc, Twine("In table '") + Table.Name + 
                                     "' lookup method '" + Index.Name +
                                     "', key field '" + Field.Name +
                                     "' of type bits is too large");
    } else if (Field.Enum || Field.IsIntrinsic || Field.IsInstruction)
      return "unsigned";
    PrintFatalError(Index.Loc,
                    Twine("In table '") + Table.Name + "' lookup method '" +
                        Index.Name + "', key field '" + Field.Name +
                        "' has invalid type: " + Field.RecType->getAsString());
  }

  void emitGenericTable(const GenericTable &Table, raw_ostream &OS);
  void emitGenericEnum(const GenericEnum &Enum, raw_ostream &OS);
  void emitLookupDeclaration(const GenericTable &Table,
                             const SearchIndex &Index, raw_ostream &OS);
  void emitLookupFunction(const GenericTable &Table, const SearchIndex &Index,
                          bool IsPrimary, raw_ostream &OS);
  void emitIfdef(StringRef Guard, raw_ostream &OS);

  bool parseFieldType(GenericField &Field, Init *II);
  std::unique_ptr<SearchIndex>
  parseSearchIndex(GenericTable &Table, const RecordVal *RecVal, StringRef Name,
                   const std::vector<StringRef> &Key, bool EarlyOut);
  void collectEnumEntries(GenericEnum &Enum, StringRef NameField,
                          StringRef ValueField,
                          const std::vector<Record *> &Items);
  void collectTableEntries(GenericTable &Table,
                           const std::vector<Record *> &Items);
};

} // End anonymous namespace.

// For search indices that consists of a single field whose numeric value is
// known, return that numeric value.
static int64_t getNumericKey(const SearchIndex &Index, Record *Rec) {
  assert(Index.Fields.size() == 1);

  if (Index.Fields[0].Enum) {
    Record *EnumEntry = Rec->getValueAsDef(Index.Fields[0].Name);
    return Index.Fields[0].Enum->EntryMap[EnumEntry]->second;
  }

  return getInt(Rec, Index.Fields[0].Name);
}

/// Less-than style comparison between \p LHS and \p RHS according to the
/// key of \p Index.
bool SearchableTableEmitter::compareBy(Record *LHS, Record *RHS,
                                       const SearchIndex &Index) {
  for (const auto &Field : Index.Fields) {
    Init *LHSI = LHS->getValueInit(Field.Name);
    Init *RHSI = RHS->getValueInit(Field.Name);

    if (isa<BitsRecTy>(Field.RecType) || isa<IntRecTy>(Field.RecType)) {
      int64_t LHSi = getAsInt(LHSI);
      int64_t RHSi = getAsInt(RHSI);
      if (LHSi < RHSi)
        return true;
      if (LHSi > RHSi)
        return false;
    } else if (Field.IsIntrinsic) {
      CodeGenIntrinsic &LHSi = getIntrinsic(LHSI);
      CodeGenIntrinsic &RHSi = getIntrinsic(RHSI);
      if (std::tie(LHSi.TargetPrefix, LHSi.Name) <
          std::tie(RHSi.TargetPrefix, RHSi.Name))
        return true;
      if (std::tie(LHSi.TargetPrefix, LHSi.Name) >
          std::tie(RHSi.TargetPrefix, RHSi.Name))
        return false;
    } else if (Field.IsInstruction) {
      // This does not correctly compare the predefined instructions!
      Record *LHSr = cast<DefInit>(LHSI)->getDef();
      Record *RHSr = cast<DefInit>(RHSI)->getDef();

      bool LHSpseudo = LHSr->getValueAsBit("isPseudo");
      bool RHSpseudo = RHSr->getValueAsBit("isPseudo");
      if (LHSpseudo && !RHSpseudo)
        return true;
      if (!LHSpseudo && RHSpseudo)
        return false;

      int comp = LHSr->getName().compare(RHSr->getName());
      if (comp < 0)
        return true;
      if (comp > 0)
        return false;
    } else if (Field.Enum) {
      auto LHSr = cast<DefInit>(LHSI)->getDef();
      auto RHSr = cast<DefInit>(RHSI)->getDef();
      int64_t LHSv = Field.Enum->EntryMap[LHSr]->second;
      int64_t RHSv = Field.Enum->EntryMap[RHSr]->second;
      if (LHSv < RHSv)
        return true;
      if (LHSv > RHSv)
        return false;
    } else {
      std::string LHSs = primaryRepresentation(Index.Loc, Field, LHSI);
      std::string RHSs = primaryRepresentation(Index.Loc, Field, RHSI);

      if (isa<StringRecTy>(Field.RecType)) {
        LHSs = StringRef(LHSs).upper();
        RHSs = StringRef(RHSs).upper();
      }

      int comp = LHSs.compare(RHSs);
      if (comp < 0)
        return true;
      if (comp > 0)
        return false;
    }
  }
  return false;
}

void SearchableTableEmitter::emitIfdef(StringRef Guard, raw_ostream &OS) {
  OS << "#ifdef " << Guard << "\n";
  PreprocessorGuards.insert(std::string(Guard));
}

/// Emit a generic enum.
void SearchableTableEmitter::emitGenericEnum(const GenericEnum &Enum,
                                             raw_ostream &OS) {
  emitIfdef((Twine("GET_") + Enum.PreprocessorGuard + "_DECL").str(), OS);

  OS << "enum " << Enum.Name << " {\n";
  for (const auto &Entry : Enum.Entries)
    OS << "  " << Entry->first << " = " << Entry->second << ",\n";
  OS << "};\n";

  OS << "#endif\n\n";
}

void SearchableTableEmitter::emitLookupFunction(const GenericTable &Table,
                                                const SearchIndex &Index,
                                                bool IsPrimary,
                                                raw_ostream &OS) {
  OS << "\n";
  emitLookupDeclaration(Table, Index, OS);
  OS << " {\n";

  std::vector<Record *> IndexRowsStorage;
  ArrayRef<Record *> IndexRows;
  StringRef IndexTypeName;
  StringRef IndexName;

  if (IsPrimary) {
    IndexTypeName = Table.CppTypeName;
    IndexName = Table.Name;
    IndexRows = Table.Entries;
  } else {
    OS << "  struct IndexType {\n";
    for (const auto &Field : Index.Fields) {
      OS << "    "
         << searchableFieldType(Table, Index, Field, TypeInStaticStruct) << " "
         << Field.Name << ";\n";
    }
    OS << "    unsigned _index;\n";
    OS << "  };\n";

    OS << "  static const struct IndexType Index[] = {\n";

    std::vector<std::pair<Record *, unsigned>> Entries;
    Entries.reserve(Table.Entries.size());
    for (unsigned i = 0; i < Table.Entries.size(); ++i)
      Entries.emplace_back(Table.Entries[i], i);

    std::stable_sort(Entries.begin(), Entries.end(),
                     [&](const std::pair<Record *, unsigned> &LHS,
                         const std::pair<Record *, unsigned> &RHS) {
                       return compareBy(LHS.first, RHS.first, Index);
                     });

    IndexRowsStorage.reserve(Entries.size());
    for (const auto &Entry : Entries) {
      IndexRowsStorage.push_back(Entry.first);

      OS << "    { ";
      bool NeedComma = false;
      for (const auto &Field : Index.Fields) {
        if (NeedComma)
          OS << ", ";
        NeedComma = true;

        std::string Repr = primaryRepresentation(
            Index.Loc, Field, Entry.first->getValueInit(Field.Name));
        if (isa<StringRecTy>(Field.RecType))
          Repr = StringRef(Repr).upper();
        OS << Repr;
      }
      OS << ", " << Entry.second << " },\n";
    }

    OS << "  };\n\n";

    IndexTypeName = "IndexType";
    IndexName = "Index";
    IndexRows = IndexRowsStorage;
  }

  bool IsContiguous = false;

  if (Index.Fields.size() == 1 &&
      (Index.Fields[0].Enum || isa<BitsRecTy>(Index.Fields[0].RecType))) {
    IsContiguous = true;
    for (unsigned i = 0; i < IndexRows.size(); ++i) {
      if (getNumericKey(Index, IndexRows[i]) != i) {
        IsContiguous = false;
        break;
      }
    }
  }

  if (IsContiguous) {
    OS << "  auto Table = makeArrayRef(" << IndexName << ");\n";
    OS << "  size_t Idx = " << Index.Fields[0].Name << ";\n";
    OS << "  return Idx >= Table.size() ? nullptr : ";
    if (IsPrimary)
      OS << "&Table[Idx]";
    else
      OS << "&" << Table.Name << "[Table[Idx]._index]";
    OS << ";\n";
    OS << "}\n";
    return;
  }

  if (Index.EarlyOut) {
    const GenericField &Field = Index.Fields[0];
    std::string FirstRepr = primaryRepresentation(
        Index.Loc, Field, IndexRows[0]->getValueInit(Field.Name));
    std::string LastRepr = primaryRepresentation(
        Index.Loc, Field, IndexRows.back()->getValueInit(Field.Name));
    OS << "  if ((" << Field.Name << " < " << FirstRepr << ") ||\n";
    OS << "      (" << Field.Name << " > " << LastRepr << "))\n";
    OS << "    return nullptr;\n\n";
  }

  OS << "  struct KeyType {\n";
  for (const auto &Field : Index.Fields) {
    OS << "    " << searchableFieldType(Table, Index, Field, TypeInTempStruct) 
       << " " << Field.Name << ";\n";
  }
  OS << "  };\n";
  OS << "  KeyType Key = {";
  bool NeedComma = false;
  for (const auto &Field : Index.Fields) {
    if (NeedComma)
      OS << ", ";
    NeedComma = true;

    OS << Field.Name;
    if (isa<StringRecTy>(Field.RecType)) {
      OS << ".upper()";
      if (IsPrimary)
        PrintFatalError(Index.Loc, 
                        Twine("In table '") + Table.Name +
                            "', use a secondary lookup method for "
                            "case-insensitive comparison of field '" +
                            Field.Name + "'");
    }
  }
  OS << "};\n";

  OS << "  auto Table = makeArrayRef(" << IndexName << ");\n";
  OS << "  auto Idx = std::lower_bound(Table.begin(), Table.end(), Key,\n";
  OS << "    [](const " << IndexTypeName << " &LHS, const KeyType &RHS) {\n";

  for (const auto &Field : Index.Fields) {
    if (isa<StringRecTy>(Field.RecType)) {
      OS << "      int Cmp" << Field.Name << " = StringRef(LHS." << Field.Name
         << ").compare(RHS." << Field.Name << ");\n";
      OS << "      if (Cmp" << Field.Name << " < 0) return true;\n";
      OS << "      if (Cmp" << Field.Name << " > 0) return false;\n";
    } else if (Field.Enum) {
      // Explicitly cast to unsigned, because the signedness of enums is
      // compiler-dependent.
      OS << "      if ((unsigned)LHS." << Field.Name << " < (unsigned)RHS."
         << Field.Name << ")\n";
      OS << "        return true;\n";
      OS << "      if ((unsigned)LHS." << Field.Name << " > (unsigned)RHS."
         << Field.Name << ")\n";
      OS << "        return false;\n";
    } else {
      OS << "      if (LHS." << Field.Name << " < RHS." << Field.Name << ")\n";
      OS << "        return true;\n";
      OS << "      if (LHS." << Field.Name << " > RHS." << Field.Name << ")\n";
      OS << "        return false;\n";
    }
  }

  OS << "      return false;\n";
  OS << "    });\n\n";

  OS << "  if (Idx == Table.end()";

  for (const auto &Field : Index.Fields)
    OS << " ||\n      Key." << Field.Name << " != Idx->" << Field.Name;
  OS << ")\n    return nullptr;\n";

  if (IsPrimary)
    OS << "  return &*Idx;\n";
  else
    OS << "  return &" << Table.Name << "[Idx->_index];\n";

  OS << "}\n";
}

void SearchableTableEmitter::emitLookupDeclaration(const GenericTable &Table,
                                                   const SearchIndex &Index,
                                                   raw_ostream &OS) {
  OS << "const " << Table.CppTypeName << " *" << Index.Name << "(";

  bool NeedComma = false;
  for (const auto &Field : Index.Fields) {
    if (NeedComma)
      OS << ", ";
    NeedComma = true;

    OS << searchableFieldType(Table, Index, Field, TypeInArgument) << " "
       << Field.Name;
  }
  OS << ")";
}

void SearchableTableEmitter::emitGenericTable(const GenericTable &Table,
                                              raw_ostream &OS) {
  emitIfdef((Twine("GET_") + Table.PreprocessorGuard + "_DECL").str(), OS);

  // Emit the declarations for the functions that will perform lookup.
  if (Table.PrimaryKey) {
    emitLookupDeclaration(Table, *Table.PrimaryKey, OS);
    OS << ";\n";
  }
  for (const auto &Index : Table.Indices) {
    emitLookupDeclaration(Table, *Index, OS);
    OS << ";\n";
  }

  OS << "#endif\n\n";

  emitIfdef((Twine("GET_") + Table.PreprocessorGuard + "_IMPL").str(), OS);

  // The primary data table contains all the fields defined for this map.
  OS << "constexpr " << Table.CppTypeName << " " << Table.Name << "[] = {\n";
  for (unsigned i = 0; i < Table.Entries.size(); ++i) {
    Record *Entry = Table.Entries[i];
    OS << "  { ";

    bool NeedComma = false;
    for (const auto &Field : Table.Fields) {
      if (NeedComma)
        OS << ", ";
      NeedComma = true;

      OS << primaryRepresentation(Table.Locs[0], Field,
                                  Entry->getValueInit(Field.Name));
    }

    OS << " }, // " << i << "\n";
  }
  OS << " };\n";

  // Indexes are sorted "{ Thing, PrimaryIdx }" arrays, so that a binary
  // search can be performed by "Thing".
  if (Table.PrimaryKey)
    emitLookupFunction(Table, *Table.PrimaryKey, true, OS);
  for (const auto &Index : Table.Indices)
    emitLookupFunction(Table, *Index, false, OS);

  OS << "#endif\n\n";
}

bool SearchableTableEmitter::parseFieldType(GenericField &Field, Init *II) {
  if (auto DI = dyn_cast<DefInit>(II)) {
    Record *TypeRec = DI->getDef();
    if (TypeRec->isSubClassOf("GenericEnum")) {
      Field.Enum = EnumMap[TypeRec];
      Field.RecType = RecordRecTy::get(Field.Enum->Class);
      return true;
    }
  }

  return false;
}

std::unique_ptr<SearchIndex> SearchableTableEmitter::parseSearchIndex(
    GenericTable &Table, const RecordVal *KeyRecVal, StringRef Name,
    const std::vector<StringRef> &Key, bool EarlyOut) {
  auto Index = std::make_unique<SearchIndex>();
  Index->Name = std::string(Name);
  Index->Loc = KeyRecVal->getLoc();
  Index->EarlyOut = EarlyOut;

  for (const auto &FieldName : Key) {
    const GenericField *Field = Table.getFieldByName(FieldName);
    if (!Field)
      PrintFatalError(
          KeyRecVal,
          Twine("In table '") + Table.Name +
              "', 'PrimaryKey' or 'Key' refers to nonexistent field '" +
              FieldName + "'");
                      
    Index->Fields.push_back(*Field);
  }

  if (EarlyOut && isa<StringRecTy>(Index->Fields[0].RecType)) {
    PrintFatalError(
        KeyRecVal, Twine("In lookup method '") + Name + "', early-out is not " +
                       "supported for a first key field of type string");
  }

  return Index;
}

void SearchableTableEmitter::collectEnumEntries(
    GenericEnum &Enum, StringRef NameField, StringRef ValueField,
    const std::vector<Record *> &Items) {
  for (auto EntryRec : Items) {
    StringRef Name;
    if (NameField.empty())
      Name = EntryRec->getName();
    else
      Name = EntryRec->getValueAsString(NameField);

    int64_t Value = 0;
    if (!ValueField.empty())
      Value = getInt(EntryRec, ValueField);

    Enum.Entries.push_back(std::make_unique<GenericEnum::Entry>(Name, Value));
    Enum.EntryMap.insert(std::make_pair(EntryRec, Enum.Entries.back().get()));
  }

  if (ValueField.empty()) {
    std::stable_sort(Enum.Entries.begin(), Enum.Entries.end(),
                     [](const std::unique_ptr<GenericEnum::Entry> &LHS,
                        const std::unique_ptr<GenericEnum::Entry> &RHS) {
                       return LHS->first < RHS->first;
                     });

    for (size_t i = 0; i < Enum.Entries.size(); ++i)
      Enum.Entries[i]->second = i;
  }
}

void SearchableTableEmitter::collectTableEntries(
    GenericTable &Table, const std::vector<Record *> &Items) {
  if (Items.empty())
    PrintFatalError(Table.Locs,
                    Twine("Table '") + Table.Name + "' has no entries");

  for (auto EntryRec : Items) {
    for (auto &Field : Table.Fields) {
      auto TI = dyn_cast<TypedInit>(EntryRec->getValueInit(Field.Name));
      if (!TI || !TI->isComplete()) {
        PrintFatalError(EntryRec, Twine("Record '") + EntryRec->getName() +
                                      "' for table '" + Table.Name +
                                      "' is missing field '" + Field.Name +
                                      "'");
      }
      if (!Field.RecType) {
        Field.RecType = TI->getType();
      } else {
        RecTy *Ty = resolveTypes(Field.RecType, TI->getType());
        if (!Ty)
          PrintFatalError(EntryRec->getValue(Field.Name), 
                          Twine("Field '") + Field.Name + "' of table '" +
                          Table.Name + "' entry has incompatible type: " +
                          TI->getType()->getAsString() + " vs. " +
                          Field.RecType->getAsString());
        Field.RecType = Ty;
      }
    }

    Table.Entries.push_back(EntryRec); // Add record to table's record list.
  }

  Record *IntrinsicClass = Records.getClass("Intrinsic");
  Record *InstructionClass = Records.getClass("Instruction");
  for (auto &Field : Table.Fields) {
    if (!Field.RecType)
      PrintFatalError(Twine("Cannot determine type of field '") + Field.Name +
                      "' in table '" + Table.Name + "'. Maybe it is not used?");

    if (auto RecordTy = dyn_cast<RecordRecTy>(Field.RecType)) {
      if (IntrinsicClass && RecordTy->isSubClassOf(IntrinsicClass))
        Field.IsIntrinsic = true;
      else if (InstructionClass && RecordTy->isSubClassOf(InstructionClass))
        Field.IsInstruction = true;
    }
  }
}

void SearchableTableEmitter::run(raw_ostream &OS) {
  // Emit tables in a deterministic order to avoid needless rebuilds.
  SmallVector<std::unique_ptr<GenericTable>, 4> Tables;
  DenseMap<Record *, GenericTable *> TableMap;

  // Collect all definitions first.
  for (auto EnumRec : Records.getAllDerivedDefinitions("GenericEnum")) {
    StringRef NameField;
    if (!EnumRec->isValueUnset("NameField"))
      NameField = EnumRec->getValueAsString("NameField");

    StringRef ValueField;
    if (!EnumRec->isValueUnset("ValueField"))
      ValueField = EnumRec->getValueAsString("ValueField");

    auto Enum = std::make_unique<GenericEnum>();
    Enum->Name = std::string(EnumRec->getName());
    Enum->PreprocessorGuard = std::string(EnumRec->getName());

    StringRef FilterClass = EnumRec->getValueAsString("FilterClass");
    Enum->Class = Records.getClass(FilterClass);
    if (!Enum->Class)
      PrintFatalError(EnumRec->getValue("FilterClass"), 
                      Twine("Enum FilterClass '") + FilterClass +
                          "' does not exist");

    collectEnumEntries(*Enum, NameField, ValueField,
                       Records.getAllDerivedDefinitions(FilterClass));
    EnumMap.insert(std::make_pair(EnumRec, Enum.get()));
    Enums.emplace_back(std::move(Enum));
  }

  for (auto TableRec : Records.getAllDerivedDefinitions("GenericTable")) {
    auto Table = std::make_unique<GenericTable>();
    Table->Name = std::string(TableRec->getName());
    Table->Locs = TableRec->getLoc();
    Table->PreprocessorGuard = std::string(TableRec->getName());
    Table->CppTypeName = std::string(TableRec->getValueAsString("CppTypeName"));

    std::vector<StringRef> Fields = TableRec->getValueAsListOfStrings("Fields");
    for (const auto &FieldName : Fields) {
      Table->Fields.emplace_back(FieldName); // Construct a GenericField.

      if (auto TypeOfVal = TableRec->getValue(("TypeOf_" + FieldName).str())) {
        if (!parseFieldType(Table->Fields.back(), TypeOfVal->getValue())) {
          PrintFatalError(TypeOfVal, 
                          Twine("Table '") + Table->Name +
                              "' has invalid 'TypeOf_" + FieldName +
                              "': " + TypeOfVal->getValue()->getAsString());
        }
      }
    }

    StringRef FilterClass = TableRec->getValueAsString("FilterClass");
    if (!Records.getClass(FilterClass))
      PrintFatalError(TableRec->getValue("FilterClass"), 
                      Twine("Table FilterClass '") +
                          FilterClass + "' does not exist");

    collectTableEntries(*Table, Records.getAllDerivedDefinitions(FilterClass));

    if (!TableRec->isValueUnset("PrimaryKey")) {
      Table->PrimaryKey =
          parseSearchIndex(*Table, TableRec->getValue("PrimaryKey"),
                           TableRec->getValueAsString("PrimaryKeyName"),
                           TableRec->getValueAsListOfStrings("PrimaryKey"),
                           TableRec->getValueAsBit("PrimaryKeyEarlyOut"));

      std::stable_sort(Table->Entries.begin(), Table->Entries.end(),
                       [&](Record *LHS, Record *RHS) {
                         return compareBy(LHS, RHS, *Table->PrimaryKey);
                       });
    }

    TableMap.insert(std::make_pair(TableRec, Table.get()));
    Tables.emplace_back(std::move(Table));
  }

  for (Record *IndexRec : Records.getAllDerivedDefinitions("SearchIndex")) {
    Record *TableRec = IndexRec->getValueAsDef("Table");
    auto It = TableMap.find(TableRec);
    if (It == TableMap.end())
      PrintFatalError(IndexRec->getValue("Table"), 
                      Twine("SearchIndex '") + IndexRec->getName() +
                          "' refers to nonexistent table '" +
                          TableRec->getName());

    GenericTable &Table = *It->second;
    Table.Indices.push_back(
        parseSearchIndex(Table, IndexRec->getValue("Key"), IndexRec->getName(), 
                         IndexRec->getValueAsListOfStrings("Key"),
                         IndexRec->getValueAsBit("EarlyOut")));
  }

  // Translate legacy tables.
  Record *SearchableTable = Records.getClass("SearchableTable");
  for (auto &NameRec : Records.getClasses()) {
    Record *Class = NameRec.second.get();
    if (Class->getSuperClasses().size() != 1 ||
        !Class->isSubClassOf(SearchableTable))
      continue;

    StringRef TableName = Class->getName();
    std::vector<Record *> Items = Records.getAllDerivedDefinitions(TableName);
    if (!Class->isValueUnset("EnumNameField")) {
      StringRef NameField = Class->getValueAsString("EnumNameField");
      StringRef ValueField;
      if (!Class->isValueUnset("EnumValueField"))
        ValueField = Class->getValueAsString("EnumValueField");

      auto Enum = std::make_unique<GenericEnum>();
      Enum->Name = (Twine(Class->getName()) + "Values").str();
      Enum->PreprocessorGuard = Class->getName().upper();
      Enum->Class = Class;

      collectEnumEntries(*Enum, NameField, ValueField, Items);

      Enums.emplace_back(std::move(Enum));
    }

    auto Table = std::make_unique<GenericTable>();
    Table->Name = (Twine(Class->getName()) + "sList").str();
    Table->Locs = Class->getLoc();
    Table->PreprocessorGuard = Class->getName().upper();
    Table->CppTypeName = std::string(Class->getName());

    for (const RecordVal &Field : Class->getValues()) {
      std::string FieldName = std::string(Field.getName());

      // Skip uninteresting fields: either special to us, or injected
      // template parameters (if they contain a ':').
      if (FieldName.find(':') != std::string::npos ||
          FieldName == "SearchableFields" || FieldName == "EnumNameField" ||
          FieldName == "EnumValueField")
        continue;

      Table->Fields.emplace_back(FieldName);
    }

    collectTableEntries(*Table, Items);

    for (const auto &Field :
         Class->getValueAsListOfStrings("SearchableFields")) {
      std::string Name =
          (Twine("lookup") + Table->CppTypeName + "By" + Field).str();
      Table->Indices.push_back(parseSearchIndex(*Table, Class->getValue(Field),
                                                Name, {Field}, false));
    }

    Tables.emplace_back(std::move(Table));
  }

  // Emit everything.
  for (const auto &Enum : Enums)
    emitGenericEnum(*Enum, OS);

  for (const auto &Table : Tables)
    emitGenericTable(*Table, OS);

  // Put all #undefs last, to allow multiple sections guarded by the same
  // define.
  for (const auto &Guard : PreprocessorGuards)
    OS << "#undef " << Guard << "\n";
}

namespace llvm {

void EmitSearchableTables(RecordKeeper &RK, raw_ostream &OS) {
  SearchableTableEmitter(RK).run(OS);
}

} // End llvm namespace.
