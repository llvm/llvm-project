//===- TableGen.cpp - Top-Level TableGen implementation for Clang ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These tablegen backends emits LLDB's OptionDefinition values for different
// LLDB commands.
//
//===----------------------------------------------------------------------===//

#include "LLDBTableGenBackends.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/StringMatcher.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <map>
#include <vector>

using namespace llvm;

/// Map of command names to their associated records. Also makes sure our
/// commands are sorted in a deterministic way.
typedef std::map<std::string, std::vector<Record *>> RecordsByCommand;

/// Groups all records by their command.
static RecordsByCommand getCommandList(std::vector<Record *> Options) {
  RecordsByCommand result;
  for (Record *Option : Options)
    result[Option->getValueAsString("Command").str()].push_back(Option);
  return result;
}

static void emitOption(Record *Option, raw_ostream &OS) {
  OS << "  {";

  // List of option groups this option is in.
  std::vector<std::string> GroupsArg;

  if (Option->getValue("Groups")) {
    // The user specified a list of groups.
    auto Groups = Option->getValueAsListOfInts("Groups");
    for (int Group : Groups)
      GroupsArg.push_back("LLDB_OPT_SET_" + std::to_string(Group));
  } else if (Option->getValue("GroupStart")) {
    // The user specified a range of groups (with potentially only one element).
    int GroupStart = Option->getValueAsInt("GroupStart");
    int GroupEnd = Option->getValueAsInt("GroupEnd");
    for (int i = GroupStart; i <= GroupEnd; ++i)
      GroupsArg.push_back("LLDB_OPT_SET_" + std::to_string(i));
  }

  // If we have any groups, we merge them. Otherwise we move this option into
  // the all group.
  if (GroupsArg.empty())
    OS << "LLDB_OPT_SET_ALL";
  else
    OS << llvm::join(GroupsArg.begin(), GroupsArg.end(), " | ");

  OS << ", ";

  // Check if this option is required.
  OS << (Option->getValue("Required") ? "true" : "false");

  // Add the full and short name for this option.
  OS << ", \"" << Option->getValueAsString("FullName") << "\", ";
  OS << '\'' << Option->getValueAsString("ShortName") << "'";

  auto ArgType = Option->getValue("ArgType");
  bool IsOptionalArg = Option->getValue("OptionalArg") != nullptr;

  // Decide if we have either an option, required or no argument for this
  // option.
  OS << ", OptionParser::";
  if (ArgType) {
    if (IsOptionalArg)
      OS << "eOptionalArgument";
    else
      OS << "eRequiredArgument";
  } else
    OS << "eNoArgument";
  OS << ", nullptr, ";

  if (Option->getValue("ArgEnum"))
    OS << Option->getValueAsString("ArgEnum");
  else
    OS << "{}";
  OS << ", ";

  // Read the tab completions we offer for this option (if there are any)
  if (Option->getValue("Completions")) {
    auto Completions = Option->getValueAsListOfStrings("Completions");
    std::vector<std::string> CompletionArgs;
    for (llvm::StringRef Completion : Completions)
      CompletionArgs.push_back("CommandCompletions::e" + Completion.str() +
                               "Completion");

    OS << llvm::join(CompletionArgs.begin(), CompletionArgs.end(), " | ");
  } else {
    OS << "CommandCompletions::eNoCompletion";
  }

  // Add the argument type.
  OS << ", eArgType";
  if (ArgType) {
    OS << ArgType->getValue()->getAsUnquotedString();
  } else
    OS << "None";
  OS << ", ";

  // Add the description if there is any.
  if (auto D = Option->getValue("Description"))
    OS << D->getValue()->getAsString();
  else
    OS << "\"\"";
  OS << "},\n";
}

/// Emits all option initializers to the raw_ostream.
static void emitOptions(std::string Command, std::vector<Record *> Option,
                        raw_ostream &OS) {
  // Generate the macro that the user needs to define before including the
  // *.inc file.
  std::string NeededMacro = "LLDB_OPTIONS_" + Command;
  std::replace(NeededMacro.begin(), NeededMacro.end(), ' ', '_');

  // All options are in one file, so we need put them behind macros and ask the
  // user to define the macro for the options that are needed.
  OS << "// Options for " << Command << "\n";
  OS << "#ifdef " << NeededMacro << "\n";
  for (Record *R : Option)
    emitOption(R, OS);
  // We undefine the macro for the user like Clang's include files are doing it.
  OS << "#undef " << NeededMacro << "\n";
  OS << "#endif // " << Command << " command\n\n";
}

void lldb_private::EmitOptionDefs(RecordKeeper &Records, raw_ostream &OS) {

  std::vector<Record *> Options = Records.getAllDerivedDefinitions("Option");

  emitSourceFileHeader("Options for LLDB command line commands.", OS);

  RecordsByCommand ByCommand = getCommandList(Options);

  for (auto &CommandRecordPair : ByCommand) {
    emitOptions(CommandRecordPair.first, CommandRecordPair.second, OS);
  }
}
