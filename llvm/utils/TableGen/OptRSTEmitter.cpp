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
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <cctype>
#include <cstring>
#include <map>

using namespace llvm;

/// OptParserEmitter - This tablegen backend takes an input .td file
/// describing a list of options and emits a RST man page.
namespace llvm {
void EmitOptRST(RecordKeeper &Records, raw_ostream &OS) {
  llvm::StringMap<std::vector<Record *>> OptionsByGroup;
  std::vector<Record *> OptionsWithoutGroup;

  // Get the options.
  std::vector<Record *> Opts = Records.getAllDerivedDefinitions("Option");
  array_pod_sort(Opts.begin(), Opts.end(), CompareOptionRecords);

  // Get the option groups.
  const std::vector<Record *> &Groups =
      Records.getAllDerivedDefinitions("OptionGroup");
  for (unsigned i = 0, e = Groups.size(); i != e; ++i) {
    const Record &R = *Groups[i];
    OptionsByGroup.try_emplace(R.getValueAsString("Name"));
  }

  // Map options to their group.
  for (unsigned i = 0, e = Opts.size(); i != e; ++i) {
    const Record &R = *Opts[i];
    if (const DefInit *DI = dyn_cast<DefInit>(R.getValueInit("Group"))) {
      OptionsByGroup[DI->getDef()->getValueAsString("Name")].push_back(Opts[i]);
    } else {
      OptionsByGroup["options"].push_back(Opts[i]);
    }
  }

  // Print options under their group.
  for (const auto &KV : OptionsByGroup) {
    std::string GroupName = KV.getKey().upper();
    OS << GroupName << '\n';
    OS << std::string(GroupName.size(), '-') << '\n';
    OS << '\n';

    for (Record *R : KV.getValue()) {
      OS << ".. option:: ";

      // Print the prefix.
      std::vector<StringRef> Prefixes = R->getValueAsListOfStrings("Prefixes");
      if (!Prefixes.empty())
        OS << Prefixes[0];

      // Print the option name.
      OS << R->getValueAsString("Name");

      // Print the meta-variable.
      if (!isa<UnsetInit>(R->getValueInit("MetaVarName"))) {
        OS << '=';
        OS.write_escaped(R->getValueAsString("MetaVarName"));
      }

      OS << "\n\n";

      // The option help text.
      if (!isa<UnsetInit>(R->getValueInit("HelpText"))) {
        OS << ' ';
        OS.write_escaped(R->getValueAsString("HelpText"));
        OS << "\n\n";
      }
    }
  }
}
} // end namespace llvm
