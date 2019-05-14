//===- YAMLRemarkParser.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides utility methods used by clients that want to use the
// parser for remark diagnostics in LLVM.
//
//===----------------------------------------------------------------------===//

#include "YAMLRemarkParser.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Remarks/RemarkParser.h"

using namespace llvm;
using namespace llvm::remarks;

char YAMLParseError::ID = 0;

Error YAMLRemarkParser::parseKey(StringRef &Result, yaml::KeyValueNode &Node) {
  if (auto *Key = dyn_cast<yaml::ScalarNode>(Node.getKey())) {
    Result = Key->getRawValue();
    return Error::success();
  }

  return make_error<YAMLParseError>("key is not a string.", Node);
}

template <typename T>
Error YAMLRemarkParser::parseStr(T &Result, yaml::KeyValueNode &Node) {
  auto *Value = dyn_cast<yaml::ScalarNode>(Node.getValue());
  if (!Value)
    return make_error<YAMLParseError>("expected a value of scalar type.", Node);
  StringRef Tmp;
  if (!StrTab) {
    Tmp = Value->getRawValue();
  } else {
    // If we have a string table, parse it as an unsigned.
    unsigned StrID = 0;
    if (Error E = parseUnsigned(StrID, Node))
      return E;
    if (Expected<StringRef> Str = (*StrTab)[StrID])
      Tmp = *Str;
    else
      return Str.takeError();
  }

  if (Tmp.front() == '\'')
    Tmp = Tmp.drop_front();

  if (Tmp.back() == '\'')
    Tmp = Tmp.drop_back();

  Result = Tmp;

  return Error::success();
}

template <typename T>
Error YAMLRemarkParser::parseUnsigned(T &Result, yaml::KeyValueNode &Node) {
  SmallVector<char, 4> Tmp;
  auto *Value = dyn_cast<yaml::ScalarNode>(Node.getValue());
  if (!Value)
    return make_error<YAMLParseError>("expected a value of scalar type.", Node);
  unsigned UnsignedValue = 0;
  if (Value->getValue(Tmp).getAsInteger(10, UnsignedValue))
    return make_error<YAMLParseError>("expected a value of integer type.",
                                      *Value);
  Result = UnsignedValue;
  return Error::success();
}

Error YAMLRemarkParser::parseType(Type &Result, yaml::MappingNode &Node) {
  auto Type = StringSwitch<remarks::Type>(Node.getRawTag())
                  .Case("!Passed", remarks::Type::Passed)
                  .Case("!Missed", remarks::Type::Missed)
                  .Case("!Analysis", remarks::Type::Analysis)
                  .Case("!AnalysisFPCommute", remarks::Type::AnalysisFPCommute)
                  .Case("!AnalysisAliasing", remarks::Type::AnalysisAliasing)
                  .Case("!Failure", remarks::Type::Failure)
                  .Default(remarks::Type::Unknown);
  if (Type == remarks::Type::Unknown)
    return make_error<YAMLParseError>("expected a remark tag.", Node);
  Result = Type;
  return Error::success();
}

Error YAMLRemarkParser::parseDebugLoc(Optional<RemarkLocation> &Result,
                                      yaml::KeyValueNode &Node) {
  auto *DebugLoc = dyn_cast<yaml::MappingNode>(Node.getValue());
  if (!DebugLoc)
    return make_error<YAMLParseError>("expected a value of mapping type.",
                                      Node);

  Optional<StringRef> File;
  Optional<unsigned> Line;
  Optional<unsigned> Column;

  for (yaml::KeyValueNode &DLNode : *DebugLoc) {
    StringRef KeyName;
    if (Error E = parseKey(KeyName, DLNode))
      return E;
    if (KeyName == "File") {
      if (Error E = parseStr(File, DLNode))
        return E;
    } else if (KeyName == "Column") {
      if (Error E = parseUnsigned(Column, DLNode))
        return E;
    } else if (KeyName == "Line") {
      if (Error E = parseUnsigned(Line, DLNode))
        return E;
    } else {
      return make_error<YAMLParseError>("unknown entry in DebugLoc map.",
                                        DLNode);
    }
  }

  // If any of the debug loc fields is missing, return an error.
  if (!File || !Line || !Column)
    return make_error<YAMLParseError>("DebugLoc node incomplete.", Node);

  Result = RemarkLocation{*File, *Line, *Column};

  return Error::success();
}

Error YAMLRemarkParser::parseRemarkField(yaml::KeyValueNode &RemarkField) {

  StringRef KeyName;
  if (Error E = parseKey(KeyName, RemarkField))
    return E;

  if (KeyName == "Pass") {
    if (Error E = parseStr(State->TheRemark.PassName, RemarkField))
      return E;
  } else if (KeyName == "Name") {
    if (Error E = parseStr(State->TheRemark.RemarkName, RemarkField))
      return E;
  } else if (KeyName == "Function") {
    if (Error E = parseStr(State->TheRemark.FunctionName, RemarkField))
      return E;
  } else if (KeyName == "Hotness") {
    State->TheRemark.Hotness = 0;
    if (Error E = parseUnsigned(*State->TheRemark.Hotness, RemarkField))
      return E;
  } else if (KeyName == "DebugLoc") {
    if (Error E = parseDebugLoc(State->TheRemark.Loc, RemarkField))
      return E;
  } else if (KeyName == "Args") {
    auto *Args = dyn_cast<yaml::SequenceNode>(RemarkField.getValue());
    if (!Args)
      return make_error<YAMLParseError>("wrong value type for key.",
                                        RemarkField);

    for (yaml::Node &Arg : *Args)
      if (Error E = parseArg(State->Args, Arg))
        return E;

    State->TheRemark.Args = State->Args;
  } else {
    return make_error<YAMLParseError>("unknown key.", RemarkField);
  }

  return Error::success();
}

Error YAMLRemarkParser::parseArg(SmallVectorImpl<Argument> &Args,
                                 yaml::Node &Node) {
  auto *ArgMap = dyn_cast<yaml::MappingNode>(&Node);
  if (!ArgMap)
    return make_error<YAMLParseError>("expected a value of mapping type.",
                                      Node);

  StringRef KeyStr;
  StringRef ValueStr;
  Optional<RemarkLocation> Loc;

  for (yaml::KeyValueNode &ArgEntry : *ArgMap)
    if (Error E = parseArgEntry(ArgEntry, KeyStr, ValueStr, Loc))
      return E;

  if (KeyStr.empty())
    return make_error<YAMLParseError>("argument key is missing.", *ArgMap);
  if (ValueStr.empty())
    return make_error<YAMLParseError>("argument value is missing.", *ArgMap);

  Args.push_back(Argument{KeyStr, ValueStr, Loc});

  return Error::success();
}

Error YAMLRemarkParser::parseArgEntry(yaml::KeyValueNode &ArgEntry,
                                      StringRef &KeyStr, StringRef &ValueStr,
                                      Optional<RemarkLocation> &Loc) {
  StringRef KeyName;
  if (Error E = parseKey(KeyName, ArgEntry))
    return E;

  // Try to parse debug locs.
  if (KeyName == "DebugLoc") {
    // Can't have multiple DebugLoc entries per argument.
    if (Loc)
      return make_error<YAMLParseError>(
          "only one DebugLoc entry is allowed per argument.", ArgEntry);

    if (Error E = parseDebugLoc(Loc, ArgEntry))
      return E;
    return Error::success();
  }

  // If we already have a string, error out.
  if (!ValueStr.empty())
    return make_error<YAMLParseError>(
        "only one string entry is allowed per argument.", ArgEntry);

  // Try to parse a string.
  if (Error E = parseStr(ValueStr, ArgEntry))
    return E;

  // Keep the key from the string.
  KeyStr = KeyName;
  return Error::success();
}

Error YAMLRemarkParser::parseYAMLElement(yaml::Document &Remark) {
  // Parsing a new remark, clear the previous one by re-constructing the state
  // in-place in the Optional.
  State.emplace(TmpArgs);

  yaml::Node *YAMLRoot = Remark.getRoot();
  if (!YAMLRoot)
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "not a valid YAML file.");

  auto *Root = dyn_cast<yaml::MappingNode>(YAMLRoot);
  if (!Root)
    return make_error<YAMLParseError>("document root is not of mapping type.",
                                      *YAMLRoot);

  if (Error E = parseType(State->TheRemark.RemarkType, *Root))
    return E;

  for (yaml::KeyValueNode &RemarkField : *Root)
    if (Error E = parseRemarkField(RemarkField))
      return E;

  // If the YAML parsing failed, don't even continue parsing. We might
  // encounter malformed YAML.
  if (Stream.failed())
    return make_error<YAMLParseError>("YAML parsing failed.",
                                      *Remark.getRoot());

  // Check if any of the mandatory fields are missing.
  if (State->TheRemark.RemarkType == Type::Unknown ||
      State->TheRemark.PassName.empty() ||
      State->TheRemark.RemarkName.empty() ||
      State->TheRemark.FunctionName.empty())
    return make_error<YAMLParseError>("Type, Pass, Name or Function missing.",
                                      *Remark.getRoot());

  return Error::success();
}

/// Handle a diagnostic from the YAML stream. Records the error in the
/// YAMLRemarkParser class.
void YAMLRemarkParser::HandleDiagnostic(const SMDiagnostic &Diag, void *Ctx) {
  assert(Ctx && "Expected non-null Ctx in diagnostic handler.");
  auto *Parser = static_cast<YAMLRemarkParser *>(Ctx);
  Diag.print(/*ProgName=*/nullptr, Parser->ErrorStream, /*ShowColors*/ false,
             /*ShowKindLabels*/ true);
}
