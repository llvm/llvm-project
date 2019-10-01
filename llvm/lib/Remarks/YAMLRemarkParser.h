//===-- YAMLRemarkParser.h - Parser for YAML remarks ------------*- C++/-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides the impementation of the YAML remark parser.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_REMARKS_YAML_REMARK_PARSER_H
#define LLVM_REMARKS_YAML_REMARK_PARSER_H

#include "RemarkParserImpl.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Remarks/Remark.h"
#include "llvm/Remarks/RemarkParser.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

namespace llvm {
namespace remarks {
/// Parses and holds the state of the latest parsed remark.
struct YAMLRemarkParser {
  /// Source manager for better error messages.
  SourceMgr SM;
  /// Stream for yaml parsing.
  yaml::Stream Stream;
  /// Storage for the error stream.
  std::string ErrorString;
  /// The error stream.
  raw_string_ostream ErrorStream;
  /// Temporary parsing buffer for the arguments.
  SmallVector<Argument, 8> TmpArgs;
  /// The string table used for parsing strings.
  Optional<ParsedStringTable> StrTab;
  /// The state used by the parser to parse a remark entry. Invalidated with
  /// every call to `parseYAMLElement`.
  struct ParseState {
    /// Temporary parsing buffer for the arguments.
    /// The parser itself is owning this buffer in order to reduce the number of
    /// allocations.
    SmallVectorImpl<Argument> &Args;
    Remark TheRemark;

    ParseState(SmallVectorImpl<Argument> &Args) : Args(Args) {}
    /// Use Args only as a **temporary** buffer.
    ~ParseState() { Args.clear(); }
  };

  /// The current state of the parser. If the parsing didn't start yet, it will
  /// not be containing any value.
  Optional<ParseState> State;

  YAMLRemarkParser(StringRef Buf, Optional<StringRef> StrTabBuf = None)
      : SM(), Stream(Buf, SM), ErrorString(), ErrorStream(ErrorString),
        TmpArgs(), StrTab() {
    SM.setDiagHandler(YAMLRemarkParser::HandleDiagnostic, this);

    if (StrTabBuf)
      StrTab.emplace(*StrTabBuf);
  }

  /// Parse a YAML element.
  Error parseYAMLElement(yaml::Document &Remark);

private:
  /// Parse one key to a string.
  /// otherwise.
  Error parseKey(StringRef &Result, yaml::KeyValueNode &Node);
  /// Parse one value to a string.
  template <typename T> Error parseStr(T &Result, yaml::KeyValueNode &Node);
  /// Parse one value to an unsigned.
  template <typename T>
  Error parseUnsigned(T &Result, yaml::KeyValueNode &Node);
  /// Parse the type of a remark to an enum type.
  Error parseType(Type &Result, yaml::MappingNode &Node);
  /// Parse a debug location.
  Error parseDebugLoc(Optional<RemarkLocation> &Result,
                      yaml::KeyValueNode &Node);
  /// Parse a remark field and update the parsing state.
  Error parseRemarkField(yaml::KeyValueNode &RemarkField);
  /// Parse an argument.
  Error parseArg(SmallVectorImpl<Argument> &TmpArgs, yaml::Node &Node);
  /// Parse an entry from the contents of an argument.
  Error parseArgEntry(yaml::KeyValueNode &ArgEntry, StringRef &KeyStr,
                      StringRef &ValueStr, Optional<RemarkLocation> &Loc);

  /// Handle a diagnostic from the YAML stream. Records the error in the
  /// YAMLRemarkParser class.
  static void HandleDiagnostic(const SMDiagnostic &Diag, void *Ctx);
};

class YAMLParseError : public ErrorInfo<YAMLParseError> {
public:
  static char ID;

  YAMLParseError(StringRef Message, yaml::Node &Node)
      : Message(Message), Node(Node) {}

  void log(raw_ostream &OS) const override { OS << Message; }
  std::error_code convertToErrorCode() const override {
    return inconvertibleErrorCode();
  }

  StringRef getMessage() const { return Message; }
  yaml::Node &getNode() const { return Node; }

private:
  StringRef Message; // No need to hold a full copy of the buffer.
  yaml::Node &Node;
};

/// Regular YAML to Remark parser.
struct YAMLParserImpl : public ParserImpl {
  /// The object parsing the YAML.
  YAMLRemarkParser YAMLParser;
  /// Iterator in the YAML stream.
  yaml::document_iterator YAMLIt;
  /// Set to `true` if we had any errors during parsing.
  bool HasErrors = false;

  YAMLParserImpl(StringRef Buf, Optional<StringRef> StrTabBuf = None)
      : ParserImpl{ParserImpl::Kind::YAML}, YAMLParser(Buf, StrTabBuf),
        YAMLIt(YAMLParser.Stream.begin()), HasErrors(false) {}

  static bool classof(const ParserImpl *PI) {
    return PI->ParserKind == ParserImpl::Kind::YAML;
  }
};
} // end namespace remarks
} // end namespace llvm

#endif /* LLVM_REMARKS_YAML_REMARK_PARSER_H */
