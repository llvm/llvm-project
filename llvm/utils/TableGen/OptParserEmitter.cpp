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
#include <cctype>
#include <cstring>
#include <map>
#include <memory>

using namespace llvm;

static const std::string getOptionName(const Record &R) {
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

static const std::string getOptionSpelling(const Record &R,
                                           size_t &PrefixLength) {
  std::vector<StringRef> Prefixes = R.getValueAsListOfStrings("Prefixes");
  StringRef Name = R.getValueAsString("Name");

  if (Prefixes.empty()) {
    PrefixLength = 0;
    return Name.str();
  }

  PrefixLength = Prefixes[0].size();
  return (Twine(Prefixes[0]) + Twine(Name)).str();
}

static const std::string getOptionSpelling(const Record &R) {
  size_t PrefixLength;
  return getOptionSpelling(R, PrefixLength);
}

static void emitNameUsingSpelling(raw_ostream &OS, const Record &R) {
  size_t PrefixLength;
  OS << "&";
  write_cstring(OS, StringRef(getOptionSpelling(R, PrefixLength)));
  OS << "[" << PrefixLength << "]";
}

class MarshallingInfo {
public:
  using Ptr = std::unique_ptr<MarshallingInfo>;

  const char *MacroName;
  const Record &R;
  bool ShouldAlwaysEmit;
  StringRef KeyPath;
  StringRef DefaultValue;
  StringRef NormalizedValuesScope;
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

  MarshallingInfo(const Record &R)
      : MacroName("OPTION_WITH_MARSHALLING"), R(R) {}
  MarshallingInfo(const char *MacroName, const Record &R)
      : MacroName(MacroName), R(R){};

  virtual ~MarshallingInfo() = default;

  virtual void emit(raw_ostream &OS) const {
    write_cstring(OS, StringRef(getOptionSpelling(R)));
    OS << ", ";
    OS << ShouldAlwaysEmit;
    OS << ", ";
    OS << KeyPath;
    OS << ", ";
    emitScopedNormalizedValue(OS, DefaultValue);
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

  Optional<StringRef> emitValueTable(raw_ostream &OS) const {
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

class MarshallingInfoBooleanFlag : public MarshallingInfo {
public:
  const Record &NegOption;

  MarshallingInfoBooleanFlag(const Record &Option, const Record &NegOption)
      : MarshallingInfo("OPTION_WITH_MARSHALLING_BOOLEAN", Option),
        NegOption(NegOption) {}

  void emit(raw_ostream &OS) const override {
    MarshallingInfo::emit(OS);
    OS << ", ";
    OS << getOptionName(NegOption);
    OS << ", ";
    write_cstring(OS, getOptionSpelling(NegOption));
  }
};

static MarshallingInfo::Ptr createMarshallingInfo(const Record &R) {
  assert(!isa<UnsetInit>(R.getValueInit("KeyPath")) &&
         !isa<UnsetInit>(R.getValueInit("DefaultValue")) &&
         !isa<UnsetInit>(R.getValueInit("ValueMerger")) &&
         "MarshallingInfo must have a provide a keypath, default value and a "
         "value merger");

  MarshallingInfo::Ptr Ret;
  StringRef KindString = R.getValueAsString("MarshallingInfoKind");
  if (KindString == "Default") {
    Ret = std::make_unique<MarshallingInfo>(R);
  } else if (KindString == "BooleanFlag") {
    Ret = std::make_unique<MarshallingInfoBooleanFlag>(
        R, *R.getValueAsDef("NegOption"));
  }

  Ret->ShouldAlwaysEmit = R.getValueAsBit("ShouldAlwaysEmit");
  Ret->KeyPath = R.getValueAsString("KeyPath");
  Ret->DefaultValue = R.getValueAsString("DefaultValue");
  Ret->NormalizedValuesScope = R.getValueAsString("NormalizedValuesScope");

  Ret->Normalizer = R.getValueAsString("Normalizer");
  Ret->Denormalizer = R.getValueAsString("Denormalizer");
  Ret->ValueMerger = R.getValueAsString("ValueMerger");
  Ret->ValueExtractor = R.getValueAsString("ValueExtractor");

  if (!isa<UnsetInit>(R.getValueInit("NormalizedValues"))) {
    assert(!isa<UnsetInit>(R.getValueInit("Values")) &&
           "Cannot provide normalized values for value-less options");
    Ret->TableIndex = MarshallingInfo::NextTableIndex++;
    Ret->NormalizedValues = R.getValueAsListOfStrings("NormalizedValues");
    Ret->Values.reserve(Ret->NormalizedValues.size());
    Ret->ValueTableName = getOptionName(R) + "ValueTable";

    StringRef ValuesStr = R.getValueAsString("Values");
    for (;;) {
      size_t Idx = ValuesStr.find(',');
      if (Idx == StringRef::npos)
        break;
      if (Idx > 0)
        Ret->Values.push_back(ValuesStr.slice(0, Idx));
      ValuesStr = ValuesStr.slice(Idx + 1, StringRef::npos);
    }
    if (!ValuesStr.empty())
      Ret->Values.push_back(ValuesStr);

    assert(Ret->Values.size() == Ret->NormalizedValues.size() &&
           "The number of normalized values doesn't match the number of "
           "values");
  }

  // FIXME: This is a workaround for a bug in older versions of clang (< 3.9)
  //   The constructor that is supposed to allow for Derived to Base
  //   conversion does not work. Remove this if we drop support for such
  //   configurations.
  return MarshallingInfo::Ptr(Ret.release());
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
  for (unsigned i = 0, e = Opts.size(); i != e; ++i) {
    const Record &R = *Opts[i];
    std::vector<StringRef> prf = R.getValueAsListOfStrings("Prefixes");
    PrefixKeyT prfkey(prf.begin(), prf.end());
    unsigned NewPrefix = CurPrefix + 1;
    if (Prefixes.insert(std::make_pair(prfkey, (Twine("prefix_") +
                                              Twine(NewPrefix)).str())).second)
      CurPrefix = NewPrefix;
  }

  // Dump prefixes.

  OS << "/////////\n";
  OS << "// Prefixes\n\n";
  OS << "#ifdef PREFIX\n";
  OS << "#define COMMA ,\n";
  for (PrefixesT::const_iterator I = Prefixes.begin(), E = Prefixes.end();
                                  I != E; ++I) {
    OS << "PREFIX(";

    // Prefix name.
    OS << I->second;

    // Prefix values.
    OS << ", {";
    for (PrefixKeyT::const_iterator PI = I->first.begin(),
                                    PE = I->first.end(); PI != PE; ++PI) {
      OS << "\"" << *PI << "\" COMMA ";
    }
    OS << "nullptr})\n";
  }
  OS << "#undef COMMA\n";
  OS << "#endif // PREFIX\n\n";

  OS << "/////////\n";
  OS << "// Groups\n\n";
  OS << "#ifdef OPTION\n";
  for (unsigned i = 0, e = Groups.size(); i != e; ++i) {
    const Record &R = *Groups[i];

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
    std::vector<StringRef> prf = R.getValueAsListOfStrings("Prefixes");
    OS << Prefixes[PrefixKeyT(prf.begin(), prf.end())] << ", ";

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
      for (size_t i = 0, e = AliasArgs.size(); i != e; ++i)
        OS << AliasArgs[i] << "\\0";
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
  for (unsigned I = 0, E = Opts.size(); I != E; ++I) {
    const Record &R = *Opts[I];

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

  std::vector<MarshallingInfo::Ptr> MarshallingInfos;
  for (const auto *R : OptsWithMarshalling)
    MarshallingInfos.push_back(createMarshallingInfo(*R));

  for (const auto &MI : MarshallingInfos) {
    OS << "#ifdef " << MI->MacroName << "\n";
    OS << MI->MacroName << "(";
    WriteOptRecordFields(OS, MI->R);
    OS << ", ";
    MI->emit(OS);
    OS << ")\n";
    OS << "#endif // " << MI->MacroName << "\n";
  }

  OS << "\n";
  OS << "#ifdef SIMPLE_ENUM_VALUE_TABLE";
  OS << "\n";
  OS << MarshallingInfo::ValueTablePreamble;
  std::vector<StringRef> ValueTableNames;
  for (const auto &MI : MarshallingInfos)
    if (auto MaybeValueTableName = MI->emitValueTable(OS))
      ValueTableNames.push_back(*MaybeValueTableName);

  OS << MarshallingInfo::ValueTablesDecl << "{";
  for (auto ValueTableName : ValueTableNames)
    OS << "{" << ValueTableName << ", sizeof(" << ValueTableName
       << ") / sizeof(SimpleEnumValue)"
       << "},\n";
  OS << "};\n";
  OS << "static const unsigned SimpleEnumValueTablesSize = "
        "sizeof(SimpleEnumValueTables) / sizeof(SimpleEnumValueTable);\n";

  OS << "#endif // SIMPLE_ENUM_VALUE_TABLE\n";
  OS << "\n";

  OS << "\n";
  OS << "#ifdef OPTTABLE_ARG_INIT\n";
  OS << "//////////\n";
  OS << "// Option Values\n\n";
  for (unsigned I = 0, E = Opts.size(); I != E; ++I) {
    const Record &R = *Opts[I];
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
