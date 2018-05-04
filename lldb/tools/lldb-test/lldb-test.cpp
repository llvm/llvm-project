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
#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/Symbol/TypeList.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Utility/CleanUp.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/StreamString.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/WithColor.h"
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
static int evaluateBreakpoints(Debugger &Dbg);
} // namespace breakpoint

namespace module {
cl::opt<bool> SectionContents("contents",
                              cl::desc("Dump each section's contents"),
                              cl::sub(ModuleSubcommand));
cl::list<std::string> InputFilenames(cl::Positional, cl::desc("<input files>"),
                                     cl::OneOrMore, cl::sub(ModuleSubcommand));
} // namespace module

namespace symbols {
static cl::list<std::string> InputFilenames(cl::Positional,
                                            cl::desc("<input files>"),
                                            cl::OneOrMore,
                                            cl::sub(SymbolsSubcommand));
enum class FindType {
  None,
  Function,
  Namespace,
  Type,
  Variable,
};
static cl::opt<FindType> Find(
    "find", cl::desc("Choose search type:"),
    cl::values(
        clEnumValN(FindType::None, "none",
                   "No search, just dump the module."),
        clEnumValN(FindType::Function, "function", "Find functions."),
        clEnumValN(FindType::Namespace, "namespace", "Find namespaces."),
        clEnumValN(FindType::Type, "type", "Find types."),
        clEnumValN(FindType::Variable, "variable", "Find global variables.")),
    cl::sub(SymbolsSubcommand));

static cl::opt<std::string> Name("name", cl::desc("Name to find."),
                                 cl::sub(SymbolsSubcommand));
static cl::opt<bool>
    Regex("regex",
          cl::desc("Search using regular expressions (avaliable for variables "
                   "and functions only)."),
          cl::sub(SymbolsSubcommand));
static cl::opt<std::string>
    Context("context",
            cl::desc("Restrict search to the context of the given variable."),
            cl::value_desc("variable"), cl::sub(SymbolsSubcommand));

static cl::list<FunctionNameType> FunctionNameFlags(
    "function-flags", cl::desc("Function search flags:"),
    cl::values(clEnumValN(eFunctionNameTypeAuto, "auto",
                          "Automatically deduce flags based on name."),
               clEnumValN(eFunctionNameTypeFull, "full", "Full function name."),
               clEnumValN(eFunctionNameTypeBase, "base", "Base name."),
               clEnumValN(eFunctionNameTypeMethod, "method", "Method name."),
               clEnumValN(eFunctionNameTypeSelector, "selector",
                          "Selector name.")),
    cl::sub(SymbolsSubcommand));
static FunctionNameType getFunctionNameFlags() {
  FunctionNameType Result = FunctionNameType(0);
  for (FunctionNameType Flag : FunctionNameFlags)
    Result = FunctionNameType(Result | Flag);
  return Result;
}

static Expected<CompilerDeclContext> getDeclContext(SymbolVendor &Vendor);

static Error findFunctions(lldb_private::Module &Module);
static Error findNamespaces(lldb_private::Module &Module);
static Error findTypes(lldb_private::Module &Module);
static Error findVariables(lldb_private::Module &Module);
static Error dumpModule(lldb_private::Module &Module);

static int dumpSymbols(Debugger &Dbg);
}
} // namespace opts

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

int opts::breakpoint::evaluateBreakpoints(Debugger &Dbg) {
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
  int HadErrors = 0;
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
      HadErrors = 1;
      continue;
    }

    dumpState(Target->GetBreakpointList(/*internal*/ false), P);
  }
  return HadErrors;
}

Expected<CompilerDeclContext>
opts::symbols::getDeclContext(SymbolVendor &Vendor) {
  if (Context.empty())
    return CompilerDeclContext();
  VariableList List;
  Vendor.FindGlobalVariables(ConstString(Context), nullptr, false, UINT32_MAX,
                             List);
  if (List.Empty()) {
    return make_error<StringError>("Context search didn't find a match.",
                                   inconvertibleErrorCode());
  }
  if (List.GetSize() > 1) {
    return make_error<StringError>("Context search found multiple matches.",
                                   inconvertibleErrorCode());
  }
  return List.GetVariableAtIndex(0)->GetDeclContext();
}

Error opts::symbols::findFunctions(lldb_private::Module &Module) {
  SymbolVendor &Vendor = *Module.GetSymbolVendor();
  SymbolContextList List;
  if (Regex) {
    RegularExpression RE(Name);
    assert(RE.IsValid());
    Vendor.FindFunctions(RE, true, false, List);
  } else {
    Expected<CompilerDeclContext> ContextOr = getDeclContext(Vendor);
    if (!ContextOr)
      return ContextOr.takeError();
    CompilerDeclContext *ContextPtr =
        ContextOr->IsValid() ? &*ContextOr : nullptr;

    Vendor.FindFunctions(ConstString(Name), ContextPtr, getFunctionNameFlags(),
                          true, false, List);
  }
  outs() << formatv("Found {0} functions:\n", List.GetSize());
  StreamString Stream;
  List.Dump(&Stream, nullptr);
  outs() << Stream.GetData() << "\n";
  return Error::success();
}

Error opts::symbols::findNamespaces(lldb_private::Module &Module) {
  SymbolVendor &Vendor = *Module.GetSymbolVendor();
  Expected<CompilerDeclContext> ContextOr = getDeclContext(Vendor);
  if (!ContextOr)
    return ContextOr.takeError();
  CompilerDeclContext *ContextPtr =
      ContextOr->IsValid() ? &*ContextOr : nullptr;

  SymbolContext SC;
  CompilerDeclContext Result =
      Vendor.FindNamespace(SC, ConstString(Name), ContextPtr);
  if (Result)
    outs() << "Found namespace: "
           << Result.GetScopeQualifiedName().GetStringRef() << "\n";
  else
    outs() << "Namespace not found.\n";
  return Error::success();
}

Error opts::symbols::findTypes(lldb_private::Module &Module) {
  SymbolVendor &Vendor = *Module.GetSymbolVendor();
  Expected<CompilerDeclContext> ContextOr = getDeclContext(Vendor);
  if (!ContextOr)
    return ContextOr.takeError();
  CompilerDeclContext *ContextPtr =
      ContextOr->IsValid() ? &*ContextOr : nullptr;

  SymbolContext SC;
  DenseSet<SymbolFile *> SearchedFiles;
  TypeMap Map;
  Vendor.FindTypes(SC, ConstString(Name), ContextPtr, true, UINT32_MAX,
                    SearchedFiles, Map);

  outs() << formatv("Found {0} types:\n", Map.GetSize());
  StreamString Stream;
  Map.Dump(&Stream, false);
  outs() << Stream.GetData() << "\n";
  return Error::success();
}

Error opts::symbols::findVariables(lldb_private::Module &Module) {
  SymbolVendor &Vendor = *Module.GetSymbolVendor();
  VariableList List;
  if (Regex) {
    RegularExpression RE(Name);
    assert(RE.IsValid());
    Vendor.FindGlobalVariables(RE, false, UINT32_MAX, List);
  } else {
    Expected<CompilerDeclContext> ContextOr = getDeclContext(Vendor);
    if (!ContextOr)
      return ContextOr.takeError();
    CompilerDeclContext *ContextPtr =
        ContextOr->IsValid() ? &*ContextOr : nullptr;

    Vendor.FindGlobalVariables(ConstString(Name), ContextPtr, false, UINT32_MAX,
                               List);
  }
  outs() << formatv("Found {0} variables:\n", List.GetSize());
  StreamString Stream;
  List.Dump(&Stream, false);
  outs() << Stream.GetData() << "\n";
  return Error::success();
}

Error opts::symbols::dumpModule(lldb_private::Module &Module) {
  StreamString Stream;
  Module.ParseAllDebugSymbols();
  Module.Dump(&Stream);
  outs() << Stream.GetData() << "\n";
  return Error::success();
}

int opts::symbols::dumpSymbols(Debugger &Dbg) {
  if (Find != FindType::None && Regex && !Context.empty()) {
    WithColor::error()
        << "Cannot search using both regular expressions and context.\n";
    return 1;
  }
  if ((Find == FindType::Type || Find == FindType::Namespace) && Regex) {
    WithColor::error() << "Cannot search for types and namespaces using "
                          "regular expressions.\n";
    return 1;
  }
  if (Find == FindType::Function && Regex && getFunctionNameFlags() != 0) {
    WithColor::error() << "Cannot search for types using both regular "
                          "expressions and function-flags.\n";
    return 1;
  }
  if (Regex && !RegularExpression(Name).IsValid()) {
    WithColor::error() << "`" << Name
                       << "` is not a valid regular expression.\n";
    return 1;
  }

  Error (*Action)(lldb_private::Module &);
  switch (Find) {
  case FindType::Function:
    Action = findFunctions;
    break;
  case FindType::Namespace:
    Action = findNamespaces;
    break;
  case FindType::Type:
    Action = findTypes;
    break;
  case FindType::Variable:
    Action = findVariables;
    break;
  case FindType::None:
    Action = dumpModule;
    break;
  }

  int HadErrors = 0;
  for (const auto &File : InputFilenames) {
    outs() << "Module: " << File << "\n";
    ModuleSpec Spec{FileSpec(File, false)};
    Spec.GetSymbolFileSpec().SetFile(File, false);

    auto ModulePtr = std::make_shared<lldb_private::Module>(Spec);
    SymbolVendor *Vendor = ModulePtr->GetSymbolVendor();
    if (!Vendor) {
      WithColor::error() << "Module has no symbol vendor.\n";
      HadErrors = 1;
      continue;
    }
    
    if (Error E = Action(*ModulePtr)) {
      WithColor::error() << toString(std::move(E)) << "\n";
      HadErrors = 1;
    }

    outs().flush();
  }
  return HadErrors;
}

static int dumpModules(Debugger &Dbg) {
  LinePrinter Printer(4, llvm::outs());

  int HadErrors = 0;
  for (const auto &File : opts::module::InputFilenames) {
    ModuleSpec Spec{FileSpec(File, false)};

    auto ModulePtr = std::make_shared<lldb_private::Module>(Spec);
    // Fetch symbol vendor before we get the section list to give the symbol
    // vendor a chance to populate it.
    ModulePtr->GetSymbolVendor();
    SectionList *Sections = ModulePtr->GetSectionList();
    if (!Sections) {
      llvm::errs() << "Could not load sections for module " << File << "\n";
      HadErrors = 1;
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
      Printer.formatLine("Type: {0}", S->GetTypeAsCString());
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
  return HadErrors;
}

int main(int argc, const char *argv[]) {
  StringRef ToolName = argv[0];
  sys::PrintStackTraceOnErrorSignal(ToolName);
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y;

  cl::ParseCommandLineOptions(argc, argv, "LLDB Testing Utility\n");

  SystemLifetimeManager DebuggerLifetime;
  DebuggerLifetime.Initialize(llvm::make_unique<SystemInitializerTest>(),
                              nullptr);
  CleanUp TerminateDebugger([&] { DebuggerLifetime.Terminate(); });

  auto Dbg = lldb_private::Debugger::CreateInstance();

  if (opts::BreakpointSubcommand)
    return opts::breakpoint::evaluateBreakpoints(*Dbg);
  if (opts::ModuleSubcommand)
    return dumpModules(*Dbg);
  if (opts::SymbolsSubcommand)
    return opts::symbols::dumpSymbols(*Dbg);

  WithColor::error() << "No command specified.\n";
  return 1;
}
