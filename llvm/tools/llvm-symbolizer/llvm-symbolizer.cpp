//===-- llvm-symbolizer.cpp - Simple addr2line-like symbolizer ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This utility works much like "addr2line". It is able of transforming
// tuples (module name, module offset) to code locations (function name,
// file, line number, column number). It is targeted for compiler-rt tools
// (especially AddressSanitizer and ThreadSanitizer) that can use it
// to symbolize stack traces in their error reports.
//
//===----------------------------------------------------------------------===//

#include "Opts.inc"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/Symbolize/DIPrinter.h"
#include "llvm/DebugInfo/Symbolize/Symbolize.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/COM.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>

using namespace llvm;
using namespace symbolize;

namespace {
enum ID {
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELPTEXT, METAVAR, VALUES)                                      \
  OPT_##ID,
#include "Opts.inc"
#undef OPTION
};

#define PREFIX(NAME, VALUE) const char *const NAME[] = VALUE;
#include "Opts.inc"
#undef PREFIX

static const opt::OptTable::Info InfoTable[] = {
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELPTEXT, METAVAR, VALUES)                                      \
  {                                                                            \
      PREFIX,      NAME,      HELPTEXT,                                        \
      METAVAR,     OPT_##ID,  opt::Option::KIND##Class,                        \
      PARAM,       FLAGS,     OPT_##GROUP,                                     \
      OPT_##ALIAS, ALIASARGS, VALUES},
#include "Opts.inc"
#undef OPTION
};

class SymbolizerOptTable : public opt::OptTable {
public:
  SymbolizerOptTable() : OptTable(InfoTable, true) {}
};
} // namespace

static cl::list<std::string> ClInputAddresses(cl::Positional,
                                              cl::desc("<input addresses>..."),
                                              cl::ZeroOrMore);

template<typename T>
static bool error(Expected<T> &ResOrErr) {
  if (ResOrErr)
    return false;
  logAllUnhandledErrors(ResOrErr.takeError(), errs(),
                        "LLVMSymbolizer: error reading file: ");
  return true;
}

enum class Command {
  Code,
  Data,
  Frame,
};

static bool parseCommand(StringRef BinaryName, bool IsAddr2Line,
                         StringRef InputString, Command &Cmd,
                         std::string &ModuleName, uint64_t &ModuleOffset) {
  const char kDelimiters[] = " \n\r";
  ModuleName = "";
  if (InputString.consume_front("CODE ")) {
    Cmd = Command::Code;
  } else if (InputString.consume_front("DATA ")) {
    Cmd = Command::Data;
  } else if (InputString.consume_front("FRAME ")) {
    Cmd = Command::Frame;
  } else {
    // If no cmd, assume it's CODE.
    Cmd = Command::Code;
  }
  const char *Pos = InputString.data();
  // Skip delimiters and parse input filename (if needed).
  if (BinaryName.empty()) {
    Pos += strspn(Pos, kDelimiters);
    if (*Pos == '"' || *Pos == '\'') {
      char Quote = *Pos;
      Pos++;
      const char *End = strchr(Pos, Quote);
      if (!End)
        return false;
      ModuleName = std::string(Pos, End - Pos);
      Pos = End + 1;
    } else {
      int NameLength = strcspn(Pos, kDelimiters);
      ModuleName = std::string(Pos, NameLength);
      Pos += NameLength;
    }
  } else {
    ModuleName = BinaryName.str();
  }
  // Skip delimiters and parse module offset.
  Pos += strspn(Pos, kDelimiters);
  int OffsetLength = strcspn(Pos, kDelimiters);
  StringRef Offset(Pos, OffsetLength);
  // GNU addr2line assumes the offset is hexadecimal and allows a redundant
  // "0x" or "0X" prefix; do the same for compatibility.
  if (IsAddr2Line)
    Offset.consume_front("0x") || Offset.consume_front("0X");
  return !Offset.getAsInteger(IsAddr2Line ? 16 : 0, ModuleOffset);
}

static void symbolizeInput(const opt::InputArgList &Args, uint64_t AdjustVMA,
                           bool IsAddr2Line, DIPrinter::OutputStyle OutputStyle,
                           StringRef InputString, LLVMSymbolizer &Symbolizer,
                           DIPrinter &Printer) {
  Command Cmd;
  std::string ModuleName;
  uint64_t Offset = 0;
  if (!parseCommand(Args.getLastArgValue(OPT_obj_EQ), IsAddr2Line,
                    StringRef(InputString), Cmd, ModuleName, Offset)) {
    outs() << InputString << "\n";
    return;
  }

  if (Args.hasArg(OPT_addresses)) {
    outs() << "0x";
    outs().write_hex(Offset);
    StringRef Delimiter = Args.hasArg(OPT_pretty_print) ? ": " : "\n";
    outs() << Delimiter;
  }
  Offset -= AdjustVMA;
  if (Cmd == Command::Data) {
    auto ResOrErr = Symbolizer.symbolizeData(
        ModuleName, {Offset, object::SectionedAddress::UndefSection});
    Printer << (error(ResOrErr) ? DIGlobal() : ResOrErr.get());
  } else if (Cmd == Command::Frame) {
    auto ResOrErr = Symbolizer.symbolizeFrame(
        ModuleName, {Offset, object::SectionedAddress::UndefSection});
    if (!error(ResOrErr)) {
      for (DILocal Local : *ResOrErr)
        Printer << Local;
      if (ResOrErr->empty())
        outs() << "??\n";
    }
  } else if (Args.hasFlag(OPT_inlines, OPT_no_inlines, !IsAddr2Line)) {
    auto ResOrErr = Symbolizer.symbolizeInlinedCode(
        ModuleName, {Offset, object::SectionedAddress::UndefSection});
    Printer << (error(ResOrErr) ? DIInliningInfo() : ResOrErr.get());
  } else if (OutputStyle == DIPrinter::OutputStyle::GNU) {
    // With PrintFunctions == FunctionNameKind::LinkageName (default)
    // and UseSymbolTable == true (also default), Symbolizer.symbolizeCode()
    // may override the name of an inlined function with the name of the topmost
    // caller function in the inlining chain. This contradicts the existing
    // behavior of addr2line. Symbolizer.symbolizeInlinedCode() overrides only
    // the topmost function, which suits our needs better.
    auto ResOrErr = Symbolizer.symbolizeInlinedCode(
        ModuleName, {Offset, object::SectionedAddress::UndefSection});
    Printer << (error(ResOrErr) ? DILineInfo() : ResOrErr.get().getFrame(0));
  } else {
    auto ResOrErr = Symbolizer.symbolizeCode(
        ModuleName, {Offset, object::SectionedAddress::UndefSection});
    Printer << (error(ResOrErr) ? DILineInfo() : ResOrErr.get());
  }
  if (OutputStyle == DIPrinter::OutputStyle::LLVM)
    outs() << "\n";
}

static void printHelp(bool IsAddr2Line, const SymbolizerOptTable &Tbl,
                      raw_ostream &OS) {
  StringRef ToolName = IsAddr2Line ? "llvm-addr2line" : "llvm-symbolizer";
  const char HelpText[] = " [options] addresses...";
  Tbl.PrintHelp(OS, (ToolName + HelpText).str().c_str(),
                ToolName.str().c_str());
  // TODO Replace this with OptTable API once it adds extrahelp support.
  OS << "\nPass @FILE as argument to read options from FILE.\n";
}

static opt::InputArgList parseOptions(int Argc, char *Argv[], bool IsAddr2Line,
                                      StringSaver &Saver,
                                      SymbolizerOptTable &Tbl) {
  Tbl.setGroupedShortOptions(true);
  // The environment variable specifies initial options which can be overridden
  // by commnad line options.
  Tbl.setInitialOptionsFromEnvironment(IsAddr2Line ? "LLVM_ADDR2LINE_OPTS"
                                                   : "LLVM_SYMBOLIZER_OPTS");
  bool HasError = false;
  opt::InputArgList Args =
      Tbl.parseArgs(Argc, Argv, OPT_UNKNOWN, Saver, [&](StringRef Msg) {
        errs() << ("error: " + Msg + "\n");
        HasError = true;
      });
  if (HasError)
    exit(1);
  if (Args.hasArg(OPT_help)) {
    printHelp(IsAddr2Line, Tbl, outs());
    exit(0);
  }

  return Args;
}

template <typename T>
static void parseIntArg(const opt::InputArgList &Args, int ID, T &Value) {
  if (const opt::Arg *A = Args.getLastArg(ID)) {
    StringRef V(A->getValue());
    if (!llvm::to_integer(V, Value, 0)) {
      errs() << A->getSpelling() +
                    ": expected a non-negative integer, but got '" + V + "'";
      exit(1);
    }
  } else {
    Value = 0;
  }
}

static FunctionNameKind decideHowToPrintFunctions(const opt::InputArgList &Args,
                                                  bool IsAddr2Line) {
  if (Args.hasArg(OPT_functions))
    return FunctionNameKind::LinkageName;
  if (const opt::Arg *A = Args.getLastArg(OPT_functions_EQ))
    return StringSwitch<FunctionNameKind>(A->getValue())
        .Case("none", FunctionNameKind::None)
        .Case("short", FunctionNameKind::ShortName)
        .Default(FunctionNameKind::LinkageName);
  return IsAddr2Line ? FunctionNameKind::None : FunctionNameKind::LinkageName;
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  sys::InitializeCOMRAII COM(sys::COMThreadingMode::MultiThreaded);

  bool IsAddr2Line = sys::path::stem(argv[0]).contains("addr2line");
  BumpPtrAllocator A;
  StringSaver Saver(A);
  SymbolizerOptTable Tbl;
  opt::InputArgList Args = parseOptions(argc, argv, IsAddr2Line, Saver, Tbl);

  LLVMSymbolizer::Options Opts;
  uint64_t AdjustVMA;
  unsigned SourceContextLines;
  parseIntArg(Args, OPT_adjust_vma_EQ, AdjustVMA);
  if (const opt::Arg *A = Args.getLastArg(OPT_basenames, OPT_relativenames)) {
    Opts.PathStyle =
        A->getOption().matches(OPT_basenames)
            ? DILineInfoSpecifier::FileLineInfoKind::BaseNameOnly
            : DILineInfoSpecifier::FileLineInfoKind::RelativeFilePath;
  } else {
    Opts.PathStyle = DILineInfoSpecifier::FileLineInfoKind::AbsoluteFilePath;
  }
  Opts.DebugFileDirectory = Args.getAllArgValues(OPT_debug_file_directory_EQ);
  Opts.DefaultArch = Args.getLastArgValue(OPT_default_arch_EQ).str();
  Opts.Demangle = Args.hasFlag(OPT_demangle, OPT_no_demangle, !IsAddr2Line);
  Opts.DWPName = Args.getLastArgValue(OPT_dwp_EQ).str();
  Opts.FallbackDebugPath =
      Args.getLastArgValue(OPT_fallback_debug_path_EQ).str();
  Opts.PrintFunctions = decideHowToPrintFunctions(Args, IsAddr2Line);
  parseIntArg(Args, OPT_print_source_context_lines_EQ, SourceContextLines);
  Opts.RelativeAddresses = Args.hasArg(OPT_relative_address);
  Opts.UntagAddresses =
      Args.hasFlag(OPT_untag_addresses, OPT_no_untag_addresses, !IsAddr2Line);
  Opts.UseNativePDBReader = Args.hasArg(OPT_use_native_pdb_reader);
  Opts.UseSymbolTable = true;

  for (const opt::Arg *A : Args.filtered(OPT_dsym_hint_EQ)) {
    StringRef Hint(A->getValue());
    if (sys::path::extension(Hint) == ".dSYM") {
      Opts.DsymHints.emplace_back(Hint);
    } else {
      errs() << "Warning: invalid dSYM hint: \"" << Hint
             << "\" (must have the '.dSYM' extension).\n";
    }
  }

  auto OutputStyle =
      IsAddr2Line ? DIPrinter::OutputStyle::GNU : DIPrinter::OutputStyle::LLVM;
  if (const opt::Arg *A = Args.getLastArg(OPT_output_style_EQ)) {
    OutputStyle = strcmp(A->getValue(), "GNU") == 0
                      ? DIPrinter::OutputStyle::GNU
                      : DIPrinter::OutputStyle::LLVM;
  }

  LLVMSymbolizer Symbolizer(Opts);
  DIPrinter Printer(outs(), Opts.PrintFunctions != FunctionNameKind::None,
                    Args.hasArg(OPT_pretty_print), SourceContextLines,
                    Args.hasArg(OPT_verbose), OutputStyle);

  std::vector<std::string> InputAddresses = Args.getAllArgValues(OPT_INPUT);
  if (InputAddresses.empty()) {
    const int kMaxInputStringLength = 1024;
    char InputString[kMaxInputStringLength];

    while (fgets(InputString, sizeof(InputString), stdin)) {
      // Strip newline characters.
      std::string StrippedInputString(InputString);
      StrippedInputString.erase(
          std::remove_if(StrippedInputString.begin(), StrippedInputString.end(),
                         [](char c) { return c == '\r' || c == '\n'; }),
          StrippedInputString.end());
      symbolizeInput(Args, AdjustVMA, IsAddr2Line, OutputStyle,
                     StrippedInputString, Symbolizer, Printer);
      outs().flush();
    }
  } else {
    for (StringRef Address : InputAddresses)
      symbolizeInput(Args, AdjustVMA, IsAddr2Line, OutputStyle, Address,
                     Symbolizer, Printer);
  }

  return 0;
}
