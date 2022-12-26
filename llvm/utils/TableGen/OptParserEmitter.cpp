//===- OptParserEmitter.cpp - Table Driven Command Line Parsing -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OptEmitter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <cstring>
#include <map>
#include <memory>

using namespace llvm;

static std::string getOptionName(const Record &R) {
  // Use the record name unless EnumName is defined.
  if (isa<UnsetInit>(R.getValueInit("EnumName")))
    return std::string(R.getName());

  return std::string(R.getValueAsString("EnumName"));
}

static raw_ostream &write_cstring(raw_ostream &OS, llvm::StringRef Str) {
  OS << '"';
  OS.write_escaped(Str);
  OS << '"';
  return OS;
}

static std::string getOptionSpelling(const Record &R, size_t &PrefixLength) {
  std::vector<StringRef> Prefixes = R.getValueAsListOfStrings("Prefixes");
  StringRef Name = R.getValueAsString("Name");

  if (Prefixes.empty()) {
    PrefixLength = 0;
    return Name.str();
  }

  PrefixLength = Prefixes[0].size();
  return (Twine(Prefixes[0]) + Twine(Name)).str();
}

static std::string getOptionSpelling(const Record &R) {
  size_t PrefixLength;
  return getOptionSpelling(R, PrefixLength);
}

static void emitNameUsingSpelling(raw_ostream &OS, const Record &R) {
  size_t PrefixLength;
  OS << "llvm::StringRef(&";
  write_cstring(OS, StringRef(getOptionSpelling(R, PrefixLength)));
  OS << "[" << PrefixLength << "], " << R.getValueAsString("Name").size() << ")";
}

class MarshallingInfo {
public:
  static constexpr const char *MacroName = "OPTION_WITH_MARSHALLING";
  const Record &R;
  bool ShouldAlwaysEmit;
  StringRef MacroPrefix;
  StringRef KeyPath;
  StringRef DefaultValue;
  StringRef NormalizedValuesScope;
  StringRef ImpliedCheck;
  StringRef ImpliedValue;
  StringRef ShouldParse;
  StringRef Normalizer;
  StringRef Denormalizer;
  StringRef ValueMerger;
  StringRef ValueExtractor;
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
    write_cstring(OS, StringRef(getOptionSpelling(R)));
    OS << ", ";
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
    OS << ValueMerger;
    OS << ", ";
    OS << ValueExtractor;
    OS << ", ";
    OS << TableIndex;
  }

  std::optional<StringRef> emitValueTable(raw_ostream &OS) const {
    if (TableIndex == -1)
      return {};
    OS << "static const SimpleEnumValue " << ValueTableName << "[] = {\n";
    for (unsigned I = 0, E = Values.size(); I != E; ++I) {
      OS << "{";
      write_cstring(OS, Values[I]);
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
         !isa<UnsetInit>(R.getValueInit("ValueMerger")) &&
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
  Ret.ValueMerger = R.getValueAsString("ValueMerger");
  Ret.ValueExtractor = R.getValueAsString("ValueExtractor");

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
      ValuesStr = ValuesStr.slice(Idx + 1, StringRef::npos);
    }
    if (!ValuesStr.empty())
      Ret.Values.push_back(ValuesStr);

    assert(Ret.Values.size() == Ret.NormalizedValues.size() &&
           "The number of normalized values doesn't match the number of "
           "values");
  }

  return Ret;
}

/// OptParserEmitter - This tablegen backend takes an input .td file
/// describing a list of options and emits a data structure for parsing and
/// working with those options when given an input command line.
namespace llvm {
void EmitOptParser(RecordKeeper &Records, raw_ostream &OS) {
  // Get the option groups and options.
  const std::vector<Record*> &Groups =
    Records.getAllDerivedDefinitions("OptionGroup");
  std::vector<Record*> Opts = Records.getAllDerivedDefinitions("Option");

  emitSourceFileHeader("Option Parsing Definitions", OS);

  array_pod_sort(Opts.begin(), Opts.end(), CompareOptionRecords);
  // Generate prefix groups.
  typedef SmallVector<SmallString<2>, 2> PrefixKeyT;
  typedef std::map<PrefixKeyT, std::string> PrefixesT;
  PrefixesT Prefixes;
  Prefixes.insert(std::make_pair(PrefixKeyT(), "prefix_0"));
  unsigned CurPrefix = 0;
  for (const Record &R : llvm::make_pointee_range(Opts)) {
    std::vector<StringRef> RPrefixes = R.getValueAsListOfStrings("Prefixes");
    PrefixKeyT PrefixKey(RPrefixes.begin(), RPrefixes.end());
    unsigned NewPrefix = CurPrefix + 1;
    std::string Prefix = (Twine("prefix_") + Twine(NewPrefix)).str();
    if (Prefixes.insert(std::make_pair(PrefixKey, Prefix)).second)
      CurPrefix = NewPrefix;
  }

  // Dump prefixes.

  OS << "/////////\n";
  OS << "// Prefixes\n\n";
  OS << "#ifdef PREFIX\n";
  OS << "#define COMMA ,\n";
  for (const auto &Prefix : Prefixes) {
    OS << "PREFIX(";

    // Prefix name.
    OS << Prefix.second;

    // Prefix values.
    OS << ", {";
    for (const auto &PrefixKey : Prefix.first)
      OS << "\"" << PrefixKey << "\" COMMA ";
    OS << "nullptr})\n";
  }
  OS << "#undef COMMA\n";
  OS << "#endif // PREFIX\n\n";

  OS << "/////////\n";
  OS << "// Groups\n\n";
  OS << "#ifdef OPTION\n";
  for (const Record &R : llvm::make_pointee_range(Groups)) {
    // Start a single option entry.
    OS << "OPTION(";

    // The option prefix;
    OS << "nullptr";

    // The option string.
    OS << ", \"" << R.getValueAsString("Name") << '"';

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
    OS << ", INVALID, nullptr, 0, 0";

    // The option help text.
    if (!isa<UnsetInit>(R.getValueInit("HelpText"))) {
      OS << ",\n";
      OS << "       ";
      write_cstring(OS, R.getValueAsString("HelpText"));
    } else
      OS << ", nullptr";

    // The option meta-variable name (unused).
    OS << ", nullptr";

    // The option Values (unused for groups).
    OS << ", nullptr)\n";
  }
  OS << "\n";

  OS << "//////////\n";
  OS << "// Options\n\n";

  auto WriteOptRecordFields = [&](raw_ostream &OS, const Record &R) {
    // The option prefix;
    std::vector<StringRef> RPrefixes = R.getValueAsListOfStrings("Prefixes");
    OS << Prefixes[PrefixKeyT(RPrefixes.begin(), RPrefixes.end())] << ", ";

    // The option string.
    emitNameUsingSpelling(OS, R);

    // The option identifier name.
    OS << ", " << getOptionName(R);

    // The option kind.
    OS << ", " << R.getValueAsDef("Kind")->getValueAsString("Name");

    // The containing option group (if any).
    OS << ", ";
    const ListInit *GroupFlags = nullptr;
    if (const DefInit *DI = dyn_cast<DefInit>(R.getValueInit("Group"))) {
      GroupFlags = DI->getDef()->getValueAsListInit("Flags");
      OS << getOptionName(*DI->getDef());
    } else
      OS << "INVALID";

    // The option alias (if any).
    OS << ", ";
    if (const DefInit *DI = dyn_cast<DefInit>(R.getValueInit("Alias")))
      OS << getOptionName(*DI->getDef());
    else
      OS << "INVALID";

    // The option alias arguments (if any).
    // Emitted as a \0 separated list in a string, e.g. ["foo", "bar"]
    // would become "foo\0bar\0". Note that the compiler adds an implicit
    // terminating \0 at the end.
    OS << ", ";
    std::vector<StringRef> AliasArgs = R.getValueAsListOfStrings("AliasArgs");
    if (AliasArgs.size() == 0) {
      OS << "nullptr";
    } else {
      OS << "\"";
      for (StringRef AliasArg : AliasArgs)
        OS << AliasArg << "\\0";
      OS << "\"";
    }

    // The option flags.
    OS << ", ";
    int NumFlags = 0;
    const ListInit *LI = R.getValueAsListInit("Flags");
    for (Init *I : *LI)
      OS << (NumFlags++ ? " | " : "") << cast<DefInit>(I)->getDef()->getName();
    if (GroupFlags) {
      for (Init *I : *GroupFlags)
        OS << (NumFlags++ ? " | " : "")
           << cast<DefInit>(I)->getDef()->getName();
    }
    if (NumFlags == 0)
      OS << '0';

    // The option parameter field.
    OS << ", " << R.getValueAsInt("NumArgs");

    // The option help text.
    if (!isa<UnsetInit>(R.getValueInit("HelpText"))) {
      OS << ",\n";
      OS << "       ";
      write_cstring(OS, R.getValueAsString("HelpText"));
    } else
      OS << ", nullptr";

    // The option meta-variable name.
    OS << ", ";
    if (!isa<UnsetInit>(R.getValueInit("MetaVarName")))
      write_cstring(OS, R.getValueAsString("MetaVarName"));
    else
      OS << "nullptr";

    // The option Values. Used for shell autocompletion.
    OS << ", ";
    if (!isa<UnsetInit>(R.getValueInit("Values")))
      write_cstring(OS, R.getValueAsString("Values"));
    else
      OS << "nullptr";
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
  OS << "\n";

  OS << "\n";
  OS << "#ifdef OPTTABLE_ARG_INIT\n";
  OS << "//////////\n";
  OS << "// Option Values\n\n";
  for (const Record &R : llvm::make_pointee_range(Opts)) {
    if (isa<UnsetInit>(R.getValueInit("ValuesCode")))
      continue;
    OS << "{\n";
    OS << "bool ValuesWereAdded;\n";
    OS << R.getValueAsString("ValuesCode");
    OS << "\n";
    for (StringRef Prefix : R.getValueAsListOfStrings("Prefixes")) {
      OS << "ValuesWereAdded = Opt.addValues(";
      std::string S(Prefix);
      S += R.getValueAsString("Name");
      write_cstring(OS, S);
      OS << ", Values);\n";
      OS << "(void)ValuesWereAdded;\n";
      OS << "assert(ValuesWereAdded && \"Couldn't add values to "
            "OptTable!\");\n";
    }
    OS << "}\n";
  }
  OS << "\n";
  OS << "#endif // OPTTABLE_ARG_INIT\n";
}
} // end namespace llvm
