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
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/Section.h"
#include "lldb/Initialization/SystemLifetimeManager.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/ClangASTImporter.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/StreamString.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include <thread>

using namespace lldb;
using namespace lldb_private;
using namespace llvm;

namespace opts {
cl::SubCommand ModuleSubcommand("module-sections",
                                "Display LLDB Module Information");
cl::SubCommand SymbolsSubcommand("symbols", "Dump symbols for an object file");

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
    Spec.GetSymbolFileSpec().SetFile(File, false);

    auto ModulePtr = std::make_shared<lldb_private::Module>(Spec);
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

  if (opts::ModuleSubcommand)
    dumpModules(*Dbg);
  else if (opts::SymbolsSubcommand)
    dumpSymbols(*Dbg);

  DebuggerLifetime->Terminate();
  return 0;
}
