//===- lldb-test.cpp ------------------------------------------ *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "FormatUtil.h"
#include "SystemInitializerTest.h"

#include "Plugins/SymbolFile/DWARF/SymbolFileDWARF.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/Section.h"
#include "lldb/Initialization/SystemLifetimeManager.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/ClangASTImporter.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/StreamString.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include <thread>

using namespace lldb;
using namespace lldb_private;
using namespace llvm;

namespace opts {
static cl::SubCommand BreakpointSubcommand("breakpoints",
                                           "Test breakpoint resolution");
cl::SubCommand ModuleSubcommand("module-sections",
                                "Display LLDB Module Information");
cl::SubCommand SymbolsSubcommand("symbols", "Dump symbols for an object file");

namespace breakpoint {
static cl::opt<std::string> Target(cl::Positional, cl::desc("<target>"),
                                   cl::Required, cl::sub(BreakpointSubcommand));
static cl::opt<std::string> CommandFile(cl::Positional,
                                        cl::desc("<command-file>"),
                                        cl::init("-"),
                                        cl::sub(BreakpointSubcommand));
static cl::opt<bool> Persistent(
    "persistent",
    cl::desc("Don't automatically remove all breakpoints before each command"),
    cl::sub(BreakpointSubcommand));

static llvm::StringRef plural(uintmax_t value) { return value == 1 ? "" : "s"; }
static void dumpState(const BreakpointList &List, LinePrinter &P);
static std::string substitute(StringRef Cmd);
static void evaluateBreakpoints(Debugger &Dbg);
} // namespace breakpoint

namespace module {
cl::opt<bool> SectionContents("contents",
                              cl::desc("Dump each section's contents"),
                              cl::sub(ModuleSubcommand));
cl::list<std::string> InputFilenames(cl::Positional, cl::desc("<input files>"),
                                     cl::OneOrMore, cl::sub(ModuleSubcommand));
} // namespace module

namespace symbols {
cl::list<std::string> InputFilenames(cl::Positional, cl::desc("<input files>"),
                                     cl::OneOrMore, cl::sub(SymbolsSubcommand));
}
} // namespace opts

static llvm::ManagedStatic<SystemLifetimeManager> DebuggerLifetime;

void opts::breakpoint::dumpState(const BreakpointList &List, LinePrinter &P) {
  P.formatLine("{0} breakpoint{1}", List.GetSize(), plural(List.GetSize()));
  if (List.GetSize() > 0)
    P.formatLine("At least one breakpoint.");
  for (size_t i = 0, e = List.GetSize(); i < e; ++i) {
    BreakpointSP BP = List.GetBreakpointAtIndex(i);
    P.formatLine("Breakpoint ID {0}:", BP->GetID());
    AutoIndent Indent(P, 2);
    P.formatLine("{0} location{1}.", BP->GetNumLocations(),
                 plural(BP->GetNumLocations()));
    if (BP->GetNumLocations() > 0)
      P.formatLine("At least one location.");
    P.formatLine("{0} resolved location{1}.", BP->GetNumResolvedLocations(),
                 plural(BP->GetNumResolvedLocations()));
    if (BP->GetNumResolvedLocations() > 0)
      P.formatLine("At least one resolved location.");
    for (size_t l = 0, le = BP->GetNumLocations(); l < le; ++l) {
      BreakpointLocationSP Loc = BP->GetLocationAtIndex(l);
      P.formatLine("Location ID {0}:", Loc->GetID());
      AutoIndent Indent(P, 2);
      P.formatLine("Enabled: {0}", Loc->IsEnabled());
      P.formatLine("Resolved: {0}", Loc->IsResolved());
      SymbolContext sc;
      Loc->GetAddress().CalculateSymbolContext(&sc);
      lldb_private::StreamString S;
      sc.DumpStopContext(&S, BP->GetTarget().GetProcessSP().get(),
                         Loc->GetAddress(), false, true, false, true, true);
      P.formatLine("Address: {0}", S.GetString());
    }
  }
  P.NewLine();
}

std::string opts::breakpoint::substitute(StringRef Cmd) {
  std::string Result;
  raw_string_ostream OS(Result);
  while (!Cmd.empty()) {
    switch (Cmd[0]) {
    case '%':
      if (Cmd.consume_front("%p") && (Cmd.empty() || !isalnum(Cmd[0]))) {
        OS << sys::path::parent_path(CommandFile);
        break;
      }
      // fall through
    default:
      size_t pos = Cmd.find('%');
      OS << Cmd.substr(0, pos);
      Cmd = Cmd.substr(pos);
      break;
    }
  }
  return std::move(OS.str());
}

void opts::breakpoint::evaluateBreakpoints(Debugger &Dbg) {
  TargetSP Target;
  Status ST =
      Dbg.GetTargetList().CreateTarget(Dbg, breakpoint::Target, /*triple*/ "",
                                       /*get_dependent_modules*/ false,
                                       /*platform_options*/ nullptr, Target);
  if (ST.Fail()) {
    errs() << formatv("Failed to create target '{0}: {1}\n", breakpoint::Target,
                      ST);
    exit(1);
  }

  auto MB = MemoryBuffer::getFileOrSTDIN(CommandFile);
  if (!MB) {
    errs() << formatv("Could not open file '{0}: {1}\n", CommandFile,
                      MB.getError().message());
    exit(1);
  }

  LinePrinter P(4, outs());
  StringRef Rest = (*MB)->getBuffer();
  while (!Rest.empty()) {
    StringRef Line;
    std::tie(Line, Rest) = Rest.split('\n');
    Line = Line.ltrim();
    if (Line.empty() || Line[0] == '#')
      continue;

    if (!Persistent)
      Target->RemoveAllBreakpoints(/*internal_also*/ true);

    std::string Command = substitute(Line);
    P.formatLine("Command: {0}", Command);
    CommandReturnObject Result;
    if (!Dbg.GetCommandInterpreter().HandleCommand(
            Command.c_str(), /*add_to_history*/ eLazyBoolNo, Result)) {
      P.formatLine("Failed: {0}", Result.GetErrorData());
      continue;
    }

    dumpState(Target->GetBreakpointList(/*internal*/ false), P);
  }
}

static void dumpSymbols(Debugger &Dbg) {
  for (const auto &File : opts::symbols::InputFilenames) {
    ModuleSpec Spec{FileSpec(File, false)};
    Spec.GetSymbolFileSpec().SetFile(File, false);

    auto ModulePtr = std::make_shared<lldb_private::Module>(Spec);

    StreamString Stream;
    ModulePtr->ParseAllDebugSymbols();
    ModulePtr->Dump(&Stream);
    llvm::outs() << Stream.GetData() << "\n";
    llvm::outs().flush();
  }
}

static void dumpModules(Debugger &Dbg) {
  LinePrinter Printer(4, llvm::outs());

  for (const auto &File : opts::module::InputFilenames) {
    ModuleSpec Spec{FileSpec(File, false)};

    auto ModulePtr = std::make_shared<lldb_private::Module>(Spec);
    // Fetch symbol vendor before we get the section list to give the symbol
    // vendor a chance to populate it.
    ModulePtr->GetSymbolVendor();
    SectionList *Sections = ModulePtr->GetSectionList();
    if (!Sections) {
      llvm::errs() << "Could not load sections for module " << File << "\n";
      continue;
    }

    size_t Count = Sections->GetNumSections(0);
    Printer.formatLine("Showing {0} sections", Count);
    for (size_t I = 0; I < Count; ++I) {
      AutoIndent Indent(Printer, 2);
      auto S = Sections->GetSectionAtIndex(I);
      assert(S);
      Printer.formatLine("Index: {0}", I);
      Printer.formatLine("Name: {0}", S->GetName().GetStringRef());
      Printer.formatLine("VM size: {0}", S->GetByteSize());
      Printer.formatLine("File size: {0}", S->GetFileSize());

      if (opts::module::SectionContents) {
        DataExtractor Data;
        S->GetSectionData(Data);
        ArrayRef<uint8_t> Bytes = {Data.GetDataStart(), Data.GetDataEnd()};
        Printer.formatBinary("Data: ", Bytes, 0);
      }
      Printer.NewLine();
    }
  }
}

int main(int argc, const char *argv[]) {
  StringRef ToolName = argv[0];
  sys::PrintStackTraceOnErrorSignal(ToolName);
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y;

  cl::ParseCommandLineOptions(argc, argv, "LLDB Testing Utility\n");

  DebuggerLifetime->Initialize(llvm::make_unique<SystemInitializerTest>(),
                               nullptr);

  auto Dbg = lldb_private::Debugger::CreateInstance();

  if (opts::BreakpointSubcommand)
    opts::breakpoint::evaluateBreakpoints(*Dbg);
  if (opts::ModuleSubcommand)
    dumpModules(*Dbg);
  else if (opts::SymbolsSubcommand)
    dumpSymbols(*Dbg);

  DebuggerLifetime->Terminate();
  return 0;
}
