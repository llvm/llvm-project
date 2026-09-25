//===- yaml2obj - Convert YAML to a binary object file --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This program takes a YAML description of an object file and outputs the
// binary equivalent.
//
// This is used for writing tests that require binary files.
//
//===----------------------------------------------------------------------===//

#include "llvm/ObjectYAML/yaml2obj.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ObjectYAML/ObjectYAML.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <system_error>

using namespace llvm;

namespace {
enum ID {
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(...) LLVM_MAKE_OPT_ID(__VA_ARGS__),
#include "Opts.inc"
#undef OPTION
};

using namespace llvm::opt;
#define OPTTABLE_CODE
#include "Opts.inc"

class Yaml2ObjOptTable : public opt::OptTable {
public:
  Yaml2ObjOptTable() : OptTable(optionTables()) { setDashDashParsing(true); }
};
} // namespace

static std::optional<std::string> preprocess(StringRef Buf,
                                             ArrayRef<std::string> D,
                                             yaml::ErrorHandler ErrHandler) {
  DenseMap<StringRef, StringRef> Defines;
  for (StringRef Define : D) {
    StringRef Macro, Definition;
    std::tie(Macro, Definition) = Define.split('=');
    if (!Define.count('=') || Macro.empty()) {
      ErrHandler("invalid syntax for -D: " + Define);
      return {};
    }
    if (!Defines.try_emplace(Macro, Definition).second) {
      ErrHandler("'" + Macro + "'" + " redefined");
      return {};
    }
  }

  std::string Preprocessed;
  while (!Buf.empty()) {
    if (Buf.starts_with("[[")) {
      size_t I = Buf.find_first_of("[]", 2);
      if (Buf.substr(I).starts_with("]]")) {
        StringRef MacroExpr = Buf.substr(2, I - 2);
        StringRef Macro;
        StringRef Default;
        std::tie(Macro, Default) = MacroExpr.split('=');

        // When the -D option is requested, we use the provided value.
        // Otherwise we use a default macro value if present.
        auto It = Defines.find(Macro);
        std::optional<StringRef> Value;
        if (It != Defines.end())
          Value = It->second;
        else if (!Default.empty() || MacroExpr.ends_with("="))
          Value = Default;

        if (Value) {
          Preprocessed += *Value;
          Buf = Buf.substr(I + 2);
          continue;
        }
      }
    }

    Preprocessed += Buf[0];
    Buf = Buf.substr(1);
  }

  return Preprocessed;
}

template <class T>
static void parseIntArg(const opt::InputArgList &Args, int ID, T &Value,
                        yaml::ErrorHandler ErrHandler) {
  if (const opt::Arg *A = Args.getLastArg(ID)) {
    StringRef V(A->getValue());
    if (!to_integer(V, Value, 0)) {
      ErrHandler("expected an integer, but got '" + V + "'");
      exit(1);
    }
  }
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  constexpr StringRef ProgName = "yaml2obj";
  auto ErrHandler = [&](const Twine &Msg) {
    WithColor::error(errs(), ProgName) << Msg << "\n";
  };

  BumpPtrAllocator A;
  StringSaver Saver(A);
  Yaml2ObjOptTable Tbl;
  opt::InputArgList Args =
      Tbl.parseArgs(argc, argv, OPT_UNKNOWN, Saver, [&](StringRef Msg) {
        ErrHandler(Msg);
        exit(1);
      });
  if (Args.hasArg(OPT_help)) {
    Tbl.printHelp(outs(), "yaml2obj [options] <input file>",
                  "Create an object file from a YAML description");
    return 0;
  }
  if (Args.hasArg(OPT_version)) {
    cl::PrintVersionMessage();
    return 0;
  }

  std::vector<std::string> Inputs = Args.getAllArgValues(OPT_INPUT);
  if (Inputs.size() > 1) {
    ErrHandler("too many input files");
    return 1;
  }
  StringRef Input = Inputs.empty() ? StringRef("-") : StringRef(Inputs[0]);
  StringRef OutputFilename = Args.getLastArgValue(OPT_o, "-");
  unsigned DocNum = 1;
  parseIntArg(Args, OPT_docnum_EQ, DocNum, ErrHandler);
  uint64_t MaxSize = 10 * 1024 * 1024;
  parseIntArg(Args, OPT_max_size_EQ, MaxSize, ErrHandler);

  std::error_code EC;
  std::unique_ptr<ToolOutputFile> Out(
      new ToolOutputFile(OutputFilename, EC, sys::fs::OF_None));
  if (EC) {
    ErrHandler("failed to open '" + OutputFilename + "': " + EC.message());
    return 1;
  }

  ErrorOr<std::unique_ptr<MemoryBuffer>> Buf =
      MemoryBuffer::getFileOrSTDIN(Input, /*IsText=*/true);
  if (std::error_code EC = Buf.getError()) {
    WithColor::error(errs(), ProgName) << Input << ": " << EC.message() << '\n';
    return 1;
  }

  std::optional<std::string> Buffer = preprocess(
      Buf.get()->getBuffer(), Args.getAllArgValues(OPT_D), ErrHandler);
  if (!Buffer)
    return 1;

  if (Args.hasArg(OPT_E)) {
    Out->os() << Buffer;
  } else {
    yaml::Input YIn(*Buffer);

    if (!convertYAML(YIn, Out->os(), ErrHandler, DocNum,
                     MaxSize == 0 ? UINT64_MAX : MaxSize))
      return 1;
  }

  Out->keep();
  Out->os().flush();
  return 0;
}
