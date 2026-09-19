//===- OptionParserEmitter.cpp - Table Driven Command Option Line Parsing -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Common/OptEmitter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/StringToOffsetTable.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <cstring>
#include <map>

using namespace llvm;

static std::string getOptionName(const Record &R) {
  // Use the record name unless EnumName is defined.
  if (isa<UnsetInit>(R.getValueInit("EnumName")))
    return R.getName().str();

  return R.getValueAsString("EnumName").str();
}

// Only pass EmitComment for short strings that cannot contain "*/".
static void writeStrTableOffset(raw_ostream &OS,
                                const StringToOffsetTable &Table,
                                llvm::StringRef Str, bool EmitComment = false) {
  std::optional<unsigned> Offset = Table.GetStringOffset(Str);
  if (!Offset)
    PrintFatalError("string was not added to the option string table: " + Str);
  OS << *Offset;
  if (EmitComment) {
    OS << " /* ";
    OS.write_escaped(Str);
    OS << " */";
  }
}

static raw_ostream &writeCstring(raw_ostream &OS, llvm::StringRef Str) {
  OS << '"';
  OS.write_escaped(Str);
  OS << '"';
  return OS;
}

static StringRef getOptionalString(const Record &R, StringRef Field) {
  return R.getValueAsOptionalString(Field).value_or("");
}

// Offset zero is the empty string and stands for an unset HelpText. A
// HelpText<""> marks an option as deliberately undocumented, so it maps to a
// second empty string that the table does not put at offset zero.
static StringRef getHelpText(const Record &R) {
  std::optional<StringRef> S = R.getValueAsOptionalString("HelpText");
  if (!S)
    return StringRef();
  return S->empty() ? StringRef("\0", 1) : *S;
}

// The string table appends the empty string that terminates the list.
static std::string getAliasArgsBlob(const Record &R) {
  std::string Blob;
  for (StringRef AliasArg : R.getValueAsListOfStrings("AliasArgs")) {
    if (AliasArg.empty())
      PrintFatalError(R.getLoc(), "AliasArgs entries must not be empty");
    Blob += AliasArg;
    Blob += '\0';
  }
  return Blob;
}

static std::string getOptionPrefixedName(const Record &R) {
  std::vector<StringRef> Prefixes = R.getValueAsListOfStrings("Prefixes");
  StringRef Name = R.getValueAsString("Name");

  if (Prefixes.empty())
    return Name.str();

  return (Prefixes[0] + Twine(Name)).str();
}

class MarshallingInfo {
public:
  static constexpr const char *MacroName = "OPTION_WITH_MARSHALLING";
  const Record &R;
  bool ShouldAlwaysEmit = false;
  StringRef MacroPrefix;
  StringRef KeyPath;
  StringRef DefaultValue;
  StringRef NormalizedValuesScope;
  StringRef ImpliedCheck;
  StringRef ImpliedValue;
  StringRef ShouldParse;
  StringRef Normalizer;
  StringRef Denormalizer;
  int TableIndex = -1;
  std::vector<StringRef> Values;
  std::vector<StringRef> NormalizedValues;
  std::string ValueTableName;

  static size_t NextTableIndex;

  static constexpr const char *ValueTablePreamble = R"(
struct SimpleEnumValue {
  const char *Name;
  unsigned Value;
};

struct SimpleEnumValueTable {
  const SimpleEnumValue *Table;
  unsigned Size;
};
)";

  static constexpr const char *ValueTablesDecl =
      "static const SimpleEnumValueTable SimpleEnumValueTables[] = ";

  MarshallingInfo(const Record &R) : R(R) {}

  std::string getMacroName() const {
    return (MacroPrefix + MarshallingInfo::MacroName).str();
  }

  void emit(raw_ostream &OS) const {
    OS << ShouldParse;
    OS << ", ";
    OS << ShouldAlwaysEmit;
    OS << ", ";
    OS << KeyPath;
    OS << ", ";
    emitScopedNormalizedValue(OS, DefaultValue);
    OS << ", ";
    OS << ImpliedCheck;
    OS << ", ";
    emitScopedNormalizedValue(OS, ImpliedValue);
    OS << ", ";
    OS << Normalizer;
    OS << ", ";
    OS << Denormalizer;
    OS << ", ";
    OS << TableIndex;
  }

  std::optional<StringRef> emitValueTable(raw_ostream &OS) const {
    if (TableIndex == -1)
      return {};
    OS << "static const SimpleEnumValue " << ValueTableName << "[] = {\n";
    for (unsigned I = 0, E = Values.size(); I != E; ++I) {
      OS << "{";
      writeCstring(OS, Values[I]);
      OS << ",";
      OS << "static_cast<unsigned>(";
      emitScopedNormalizedValue(OS, NormalizedValues[I]);
      OS << ")},";
    }
    OS << "};\n";
    return StringRef(ValueTableName);
  }

private:
  void emitScopedNormalizedValue(raw_ostream &OS,
                                 StringRef NormalizedValue) const {
    if (!NormalizedValuesScope.empty())
      OS << NormalizedValuesScope << "::";
    OS << NormalizedValue;
  }
};

size_t MarshallingInfo::NextTableIndex = 0;

static MarshallingInfo createMarshallingInfo(const Record &R) {
  assert(!isa<UnsetInit>(R.getValueInit("KeyPath")) &&
         !isa<UnsetInit>(R.getValueInit("DefaultValue")) &&
         "MarshallingInfo must have a provide a keypath, default value and a "
         "value merger");

  MarshallingInfo Ret(R);

  Ret.ShouldAlwaysEmit = R.getValueAsBit("ShouldAlwaysEmit");
  Ret.MacroPrefix = R.getValueAsString("MacroPrefix");
  Ret.KeyPath = R.getValueAsString("KeyPath");
  Ret.DefaultValue = R.getValueAsString("DefaultValue");
  Ret.NormalizedValuesScope = R.getValueAsString("NormalizedValuesScope");
  Ret.ImpliedCheck = R.getValueAsString("ImpliedCheck");
  Ret.ImpliedValue =
      R.getValueAsOptionalString("ImpliedValue").value_or(Ret.DefaultValue);

  Ret.ShouldParse = R.getValueAsString("ShouldParse");
  Ret.Normalizer = R.getValueAsString("Normalizer");
  Ret.Denormalizer = R.getValueAsString("Denormalizer");

  if (!isa<UnsetInit>(R.getValueInit("NormalizedValues"))) {
    assert(!isa<UnsetInit>(R.getValueInit("Values")) &&
           "Cannot provide normalized values for value-less options");
    Ret.TableIndex = MarshallingInfo::NextTableIndex++;
    Ret.NormalizedValues = R.getValueAsListOfStrings("NormalizedValues");
    Ret.Values.reserve(Ret.NormalizedValues.size());
    Ret.ValueTableName = getOptionName(R) + "ValueTable";

    StringRef ValuesStr = R.getValueAsString("Values");
    for (;;) {
      size_t Idx = ValuesStr.find(',');
      if (Idx == StringRef::npos)
        break;
      if (Idx > 0)
        Ret.Values.push_back(ValuesStr.slice(0, Idx));
      ValuesStr = ValuesStr.substr(Idx + 1);
    }
    if (!ValuesStr.empty())
      Ret.Values.push_back(ValuesStr);

    assert(Ret.Values.size() == Ret.NormalizedValues.size() &&
           "The number of normalized values doesn't match the number of "
           "values");
  }

  return Ret;
}

/// OptionParserEmitter - This tablegen backend takes an input .td file
/// describing a list of options and emits a data structure for parsing and
/// working with those options when given an input command line.
static void emitOptionParser(const RecordKeeper &Records, raw_ostream &OS) {
  // Get the option groups and options.
  ArrayRef<const Record *> Groups =
      Records.getAllDerivedDefinitions("OptionGroup");
  std::vector<const Record *> Opts = Records.getAllDerivedDefinitions("Option");
  llvm::sort(Opts, IsOptionRecordsLess);

  std::vector<const Record *> SubCommands =
      Records.getAllDerivedDefinitions("SubCommand");

  emitSourceFileHeader("Option Parsing Definitions", OS);

  // Generate prefix groups.
  using PrefixKeyT = SmallVector<SmallString<2>, 2>;
  using PrefixesT = std::map<PrefixKeyT, unsigned>;
  PrefixesT Prefixes;
  Prefixes.try_emplace(PrefixKeyT(), 0);
  for (const Record &R : llvm::make_pointee_range(Opts)) {
    std::vector<StringRef> RPrefixes = R.getValueAsListOfStrings("Prefixes");
    PrefixKeyT PrefixKey(RPrefixes.begin(), RPrefixes.end());
    Prefixes.try_emplace(PrefixKey, 0);
  }

  // Generate sub command groups.
  using SubCommandKeyT = SmallVector<StringRef, 2>;
  using SubCommandIDsT = std::map<SubCommandKeyT, unsigned>;
  SubCommandIDsT SubCommandIDs;

  auto PrintSubCommandIdsOffset = [&SubCommandIDs, &OS](const Record &R) {
    if (R.getValue("SubCommands") != nullptr) {
      std::vector<const Record *> SubCommands =
          R.getValueAsListOfDefs("SubCommands");
      SubCommandKeyT SubCommandKey;
      for (const auto &SubCommand : SubCommands)
        SubCommandKey.push_back(SubCommand->getName());
      OS << SubCommandIDs[SubCommandKey];
    } else {
      // The option SubCommandIDsOffset (for default top level toolname is 0).
      OS << '0';
    }
  };

  SubCommandIDs.try_emplace(SubCommandKeyT(), 0);
  for (const Record &R : llvm::make_pointee_range(Opts)) {
    std::vector<const Record *> RSubCommands =
        R.getValueAsListOfDefs("SubCommands");
    SubCommandKeyT SubCommandKey;
    for (const auto &SubCommand : RSubCommands)
      SubCommandKey.push_back(SubCommand->getName());
    SubCommandIDs.try_emplace(SubCommandKey, 0);
  }

  llvm::StringToOffsetTable Table;
  for (const auto &[PrefixSet, _] : Prefixes)
    for (const auto &Prefix : PrefixSet)
      Table.GetOrAddStringOffset(Prefix);
  for (const Record &R : llvm::make_pointee_range(Groups)) {
    Table.GetOrAddStringOffset(R.getValueAsString("Name"));
    Table.GetOrAddStringOffset(getHelpText(R));
  }
  for (const Record &R : llvm::make_pointee_range(Opts)) {
    Table.GetOrAddStringOffset(getOptionPrefixedName(R));
    Table.GetOrAddStringOffset(getHelpText(R));
    Table.GetOrAddStringOffset(getOptionalString(R, "MetaVarName"));
    Table.GetOrAddStringOffset(getOptionalString(R, "Values"));
    Table.GetOrAddStringOffset(getAliasArgsBlob(R));
    for (const Record *V : R.getValueAsListOfDefs("HelpTextsForVariants"))
      Table.GetOrAddStringOffset(V->getValueAsString("Text"));
  }

  // Flags and Visibility name enumerators of the including tool. An option
  // inherits its group's.
  auto GetMask = [](const Record &R, StringRef Field) {
    std::string Mask;
    raw_string_ostream MaskOS(Mask);
    ListSeparator Sep(" | ");
    for (const Init *I : *R.getValueAsListInit(Field))
      MaskOS << Sep << cast<DefInit>(I)->getDef()->getName();
    if (const DefInit *DI = dyn_cast<DefInit>(R.getValueInit("Group")))
      for (const Init *I : *DI->getDef()->getValueAsListInit(Field))
        MaskOS << Sep << cast<DefInit>(I)->getDef()->getName();
    return Mask.empty() ? std::string("0") : Mask;
  };

  // IDs are 1-based positions in the table, which lists groups first.
  DenseMap<const Record *, unsigned> OptionID;
  for (const Record &R : llvm::make_pointee_range(Groups))
    OptionID.try_emplace(&R, OptionID.size() + 1);
  for (const Record &R : llvm::make_pointee_range(Opts))
    OptionID.try_emplace(&R, OptionID.size() + 1);
  auto GetRefID = [&](const Record &R, StringRef Field) {
    if (const DefInit *DI = dyn_cast<DefInit>(R.getValueInit(Field)))
      return OptionID.lookup(DI->getDef());
    return 0u;
  };

  // Dump string table.
  OS << "/////////\n";
  OS << "// String table\n\n";
  OS << "#if defined(OPTTABLE_STR_TABLE_CODE) || defined(OPTTABLE_CODE)\n";
  Table.EmitStringTableDef(OS, "OptionStrTable");
  OS << "#undef OPTTABLE_STR_TABLE_CODE\n";
  OS << "#endif // OPTTABLE_STR_TABLE_CODE || OPTTABLE_CODE\n\n";

  OS << "/////////\n";
  OS << "// Tables\n\n";
  OS << "#ifdef OPTTABLE_CODE\n";
  // A function rather than an object: the object needs dynamic relocations.
  OS << "static llvm::opt::OptTable::Tables optionTables() {\n";

  // Dump prefixes.
  OS << "  static constexpr llvm::StringTable::Offset OptionPrefixesTable[] = "
        "{\n";
  {
    // Ensure the first prefix set is always empty.
    assert(!Prefixes.empty() &&
           "We should always emit an empty set of prefixes");
    assert(Prefixes.begin()->first.empty() &&
           "First prefix set should always be empty");
    llvm::ListSeparator Sep(",\n");
    unsigned CurIndex = 0;
    for (auto &[Prefix, PrefixIndex] : Prefixes) {
      // First emit the number of prefix strings in this list of prefixes.
      OS << Sep << "    " << Prefix.size() << " /* prefixes */";
      PrefixIndex = CurIndex;
      assert((CurIndex == 0 || !Prefix.empty()) &&
             "Only first prefix set should be empty!");
      for (const auto &PrefixKey : Prefix)
        OS << ", " << *Table.GetStringOffset(PrefixKey) << " /* '" << PrefixKey
           << "' */";
      CurIndex += Prefix.size() + 1;
    }
  }
  OS << "\n  };\n\n";

  // Dump help text variants. Each option's variants form a run ended by a zero
  // row; offset 0 is the empty run.
  OS << "  static constexpr llvm::opt::OptTable::HelpTextVariant "
        "OptionHelpTextVariantsTable[] = {\n";
  DenseMap<const Record *, unsigned> HelpTextVariantsOffset;
  unsigned NumVariantRows = 1;
  OS << "    {0, 0},\n";
  for (const Record &R : llvm::make_pointee_range(Opts)) {
    std::vector<const Record *> Variants =
        R.getValueAsListOfDefs("HelpTextsForVariants");
    if (Variants.empty())
      continue;
    HelpTextVariantsOffset[&R] = NumVariantRows;
    NumVariantRows += Variants.size() + 1;
    for (const Record *V : Variants) {
      const ListInit *Vis = V->getValueAsListInit("Visibilities");
      if (Vis->empty())
        PrintFatalError(V->getLoc(), "HelpTextVariant needs a visibility");
      OS << "    {";
      ListSeparator Sep(" | ");
      for (const Init *I : *Vis)
        OS << Sep << I->getAsUnquotedString();
      OS << ", ";
      writeStrTableOffset(OS, Table, V->getValueAsString("Text"));
      OS << "},\n";
    }
    OS << "    {0, 0},\n";
  }
  OS << "  };\n\n";

  // Dump subcommands.
  if (!SubCommands.empty()) {
    OS << "  static constexpr llvm::opt::OptTable::SubCommand "
          "OptionSubCommands[] = {\n";
    for (const Record *SubCommand : SubCommands) {
      OS << "    { \"" << SubCommand->getValueAsString("Name") << "\", ";
      OS << "\"" << SubCommand->getValueAsString("HelpText") << "\", ";
      OS << "\"" << SubCommand->getValueAsString("Usage") << "\" },\n";
    }
    OS << "  };\n\n";
  }

  // Dump subcommand IDs.
  OS << "  static constexpr unsigned OptionSubCommandIDsTable[] = {\n";
  {
    // Ensure the first subcommand set is always empty.
    assert(!SubCommandIDs.empty() &&
           "We should always emit an empty set of subcommands");
    assert(SubCommandIDs.begin()->first.empty() &&
           "First subcommand set should always be empty");
    llvm::ListSeparator Sep(",\n");
    unsigned CurIndex = 0;
    for (auto &[SubCommand, SubCommandIndex] : SubCommandIDs) {
      // First emit the number of subcommand strings in this list of
      // subcommands.
      OS << Sep << "    " << SubCommand.size() << " /* subcommands */";
      SubCommandIndex = CurIndex;
      assert((CurIndex == 0 || !SubCommand.empty()) &&
             "Only first subcommand set should be empty!");
      for (const auto &SubCommandKey : SubCommand) {
        auto It = llvm::find_if(SubCommands, [&](const Record *R) {
          return R->getName() == SubCommandKey;
        });
        assert(It != SubCommands.end() && "SubCommand not found");
        OS << ", " << std::distance(SubCommands.begin(), It) << " /* '"
           << SubCommandKey << "' */";
      }
      CurIndex += SubCommand.size() + 1;
    }
  }
  OS << "\n  };\n\n";

  // Dump the option table in OptTable::Info field order.
  OS << "  static constexpr llvm::opt::OptTable::Info OptionInfoTable[] = {\n";
  for (const Record &R : llvm::make_pointee_range(Groups)) {
    OS << "    {";
    writeStrTableOffset(OS, Table, R.getValueAsString("Name"),
                        /*EmitComment=*/true);
    OS << ", ";
    writeStrTableOffset(OS, Table, getHelpText(R));
    OS << ", 0, 0, 0, 0, 0, 0, " << GetRefID(R, "Group") << ", 0, 0, ";
    PrintSubCommandIdsOffset(R);
    OS << ", llvm::opt::Option::GroupClass, 0},\n";
  }
  for (const Record &R : llvm::make_pointee_range(Opts)) {
    OS << "    {";
    writeStrTableOffset(OS, Table, getOptionPrefixedName(R),
                        /*EmitComment=*/true);
    OS << ", ";
    writeStrTableOffset(OS, Table, getHelpText(R));
    OS << ", ";
    writeStrTableOffset(OS, Table, getOptionalString(R, "MetaVarName"));
    OS << ", ";
    writeStrTableOffset(OS, Table, getAliasArgsBlob(R));
    OS << ", ";
    writeStrTableOffset(OS, Table, getOptionalString(R, "Values"));
    OS << ", " << GetMask(R, "Flags") << ", " << GetMask(R, "Visibility");
    std::vector<StringRef> RPrefixes = R.getValueAsListOfStrings("Prefixes");
    OS << ", " << Prefixes[PrefixKeyT(RPrefixes.begin(), RPrefixes.end())];
    OS << ", " << GetRefID(R, "Group") << ", " << GetRefID(R, "Alias");
    OS << ", " << HelpTextVariantsOffset.lookup(&R) << ", ";
    PrintSubCommandIdsOffset(R);
    OS << ", llvm::opt::Option::"
       << R.getValueAsDef("Kind")->getValueAsString("Name") << "Class, "
       << R.getValueAsInt("NumArgs") << "},\n";
  }
  OS << "  };\n\n";

  OS << "  return {OptionStrTable, OptionPrefixesTable, OptionInfoTable,\n";
  OS << "          OptionHelpTextVariantsTable, "
     << (SubCommands.empty() ? "{}" : "OptionSubCommands")
     << ", OptionSubCommandIDsTable};\n";
  OS << "}\n";
  OS << "#undef OPTTABLE_CODE\n";
  OS << "#endif // OPTTABLE_CODE\n\n";

  // Dump ValuesCode.
  OS << "/////////\n";
  OS << "// ValuesCode\n\n";
  OS << "#ifdef OPTTABLE_VALUES_CODE\n";
  std::vector<const Record *> ValuesCodeOpts;
  for (const Record &R : llvm::make_pointee_range(Opts)) {
    // The option values, if any;
    if (!isa<UnsetInit>(R.getValueInit("ValuesCode"))) {
      if (!isa<UnsetInit>(R.getValueInit("Values")))
        PrintFatalError(R.getLoc(), "cannot set both Values and ValuesCode");
      ValuesCodeOpts.push_back(&R);
      OS << "#define VALUES_CODE " << getOptionName(R) << "_Values\n";
      OS << R.getValueAsString("ValuesCode") << "\n";
      OS << "#undef VALUES_CODE\n";
    }
  }
  // A function keeps these strings out of a relocated table. It names OPT_ IDs,
  // so include this block after the option enum; a table that uses a different
  // ID prefix cannot use ValuesCode.
  OS << "static llvm::StringRef getOptionValuesCode(unsigned ID) {\n";
  OS << "  switch (ID) {\n";
  for (const Record *R : ValuesCodeOpts)
    OS << "  case OPT_" << getOptionName(*R) << ": return " << getOptionName(*R)
       << "_Values;\n";
  OS << "  }\n  return {};\n}\n";
  OS << "#undef OPTTABLE_VALUES_CODE\n";
  OS << "#endif // OPTTABLE_VALUES_CODE\n";

  OS << "/////////\n";
  OS << "// Groups\n\n";
  OS << "#ifdef OPTION\n";
  for (const Record &R : llvm::make_pointee_range(Groups)) {
    // Start a single option entry.
    OS << "OPTION(";

    // A zero prefix offset corresponds to an empty set of prefixes.
    OS << "0 /* no prefixes */";

    // The option string offset.
    OS << ", ";
    writeStrTableOffset(OS, Table, R.getValueAsString("Name"),
                        /*EmitComment=*/true);

    // The option identifier name.
    OS << ", " << getOptionName(R);

    // The option kind.
    OS << ", Group";

    // The containing option group (if any).
    OS << ", ";
    if (const DefInit *DI = dyn_cast<DefInit>(R.getValueInit("Group")))
      OS << getOptionName(*DI->getDef());
    else
      OS << "INVALID";

    // The other option arguments (unused for groups).
    OS << ", INVALID, 0, 0, 0, 0";

    // The option help text.
    OS << ", ";
    writeStrTableOffset(OS, Table, getHelpText(R));

    // Groups have no help text variants.
    OS << ", 0";

    // The option meta-variable name (unused).
    OS << ", 0";

    // The option Values (unused for groups).
    OS << ", 0";

    // The option SubCommandIDsOffset.
    OS << ", ";
    PrintSubCommandIdsOffset(R);
    OS << ")\n";
  }
  OS << "\n";

  OS << "//////////\n";
  OS << "// Options\n\n";

  auto WriteOptRecordFields = [&](raw_ostream &OS, const Record &R) {
    // The option prefix;
    std::vector<StringRef> RPrefixes = R.getValueAsListOfStrings("Prefixes");
    OS << Prefixes[PrefixKeyT(RPrefixes.begin(), RPrefixes.end())] << ", ";

    // The option prefixed name.
    writeStrTableOffset(OS, Table, getOptionPrefixedName(R),
                        /*EmitComment=*/true);

    // The option identifier name.
    OS << ", " << getOptionName(R);

    // The option kind.
    OS << ", " << R.getValueAsDef("Kind")->getValueAsString("Name");

    // The containing option group (if any).
    OS << ", ";
    if (const DefInit *DI = dyn_cast<DefInit>(R.getValueInit("Group")))
      OS << getOptionName(*DI->getDef());
    else
      OS << "INVALID";

    // The option alias (if any).
    OS << ", ";
    if (const DefInit *DI = dyn_cast<DefInit>(R.getValueInit("Alias")))
      OS << getOptionName(*DI->getDef());
    else
      OS << "INVALID";

    // The option alias arguments (if any).
    OS << ", ";
    writeStrTableOffset(OS, Table, getAliasArgsBlob(R));

    // "Flags" for the option, such as HelpHidden and Render*
    OS << ", " << GetMask(R, "Flags");

    // Option visibility, for sharing options between drivers.
    OS << ", " << GetMask(R, "Visibility");

    // The option parameter field.
    OS << ", " << R.getValueAsInt("NumArgs");

    // The option help text.
    OS << ", ";
    writeStrTableOffset(OS, Table, getHelpText(R));

    // The option help text variants.
    OS << ", " << HelpTextVariantsOffset.lookup(&R);

    // The option meta-variable name.
    OS << ", ";
    writeStrTableOffset(OS, Table, getOptionalString(R, "MetaVarName"));

    // The option Values. Used for shell autocompletion.
    OS << ", ";
    writeStrTableOffset(OS, Table, getOptionalString(R, "Values"));

    // The option SubCommandIDsOffset.
    OS << ", ";
    PrintSubCommandIdsOffset(R);
  };

  auto IsMarshallingOption = [](const Record &R) {
    return !isa<UnsetInit>(R.getValueInit("KeyPath")) &&
           !R.getValueAsString("KeyPath").empty();
  };

  std::vector<const Record *> OptsWithMarshalling;
  for (const Record &R : llvm::make_pointee_range(Opts)) {
    // Start a single option entry.
    OS << "OPTION(";
    WriteOptRecordFields(OS, R);
    OS << ")\n";
    if (IsMarshallingOption(R))
      OptsWithMarshalling.push_back(&R);
  }
  OS << "#endif // OPTION\n";

  auto CmpMarshallingOpts = [](const Record *const *A, const Record *const *B) {
    unsigned AID = (*A)->getID();
    unsigned BID = (*B)->getID();

    if (AID < BID)
      return -1;
    if (AID > BID)
      return 1;
    return 0;
  };
  // The RecordKeeper stores records (options) in lexicographical order, and we
  // have reordered the options again when generating prefix groups. We need to
  // restore the original definition order of options with marshalling to honor
  // the topology of the dependency graph implied by `DefaultAnyOf`.
  array_pod_sort(OptsWithMarshalling.begin(), OptsWithMarshalling.end(),
                 CmpMarshallingOpts);

  std::vector<MarshallingInfo> MarshallingInfos;
  MarshallingInfos.reserve(OptsWithMarshalling.size());
  for (const auto *R : OptsWithMarshalling)
    MarshallingInfos.push_back(createMarshallingInfo(*R));

  for (const auto &MI : MarshallingInfos) {
    OS << "#ifdef " << MI.getMacroName() << "\n";
    OS << MI.getMacroName() << "(";
    WriteOptRecordFields(OS, MI.R);
    OS << ", ";
    MI.emit(OS);
    OS << ")\n";
    OS << "#endif // " << MI.getMacroName() << "\n";
  }

  OS << "\n";
  OS << "#ifdef SIMPLE_ENUM_VALUE_TABLE";
  OS << "\n";
  OS << MarshallingInfo::ValueTablePreamble;
  std::vector<StringRef> ValueTableNames;
  for (const auto &MI : MarshallingInfos)
    if (auto MaybeValueTableName = MI.emitValueTable(OS))
      ValueTableNames.push_back(*MaybeValueTableName);

  OS << MarshallingInfo::ValueTablesDecl << "{";
  for (auto ValueTableName : ValueTableNames)
    OS << "{" << ValueTableName << ", std::size(" << ValueTableName << ")},\n";
  OS << "};\n";
  OS << "static const unsigned SimpleEnumValueTablesSize = "
        "std::size(SimpleEnumValueTables);\n";

  OS << "#endif // SIMPLE_ENUM_VALUE_TABLE\n";
}

static TableGen::Emitter::Opt X("gen-opt-parser-defs", emitOptionParser,
                                "Generate option definitions");
