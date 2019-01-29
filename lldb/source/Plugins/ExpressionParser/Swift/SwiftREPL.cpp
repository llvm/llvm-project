//===-- SwiftREPL.cpp -------------------------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#include "SwiftREPL.h"

#include "SwiftExpressionVariable.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/DataFormatters/ValueObjectPrinter.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SwiftASTContext.h"
#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/AnsiTerminal.h"
#include "lldb/Utility/CleanUp.h"

#include "llvm/Support/raw_ostream.h"
#include "swift/Basic/Version.h"
#include "swift/Frontend/Frontend.h"
#include "swift/IDE/REPLCodeCompletion.h"
#include "swift/IDE/Utils.h"
#include "swift/SIL/SILModule.h"

using namespace lldb;
using namespace lldb_private;

lldb::REPLSP SwiftREPL::CreateInstance(Status &err, lldb::LanguageType language,
                                       Debugger *debugger, Target *target,
                                       const char *repl_options) {
  if (language != eLanguageTypeSwift) {
    // TODO: EnumerateSupportedLanguages should make checking for this
    // unnecessary.
    return nullptr;
  }

  if (!target && !debugger) {
    err.SetErrorString("must have a debugger or a target to create a REPL");
    return nullptr;
  }

  if (target)
    return CreateInstanceFromTarget(err, *target, repl_options);
  else
    return CreateInstanceFromDebugger(err, *debugger, repl_options);
}

lldb::REPLSP SwiftREPL::CreateInstanceFromTarget(Status &err, Target &target,
                                                 const char *repl_options) {
  // Sanity check the target to make sure a REPL would work here.
  if (!target.GetProcessSP() || !target.GetProcessSP()->IsAlive()) {
    err.SetErrorString("can't launch a Swift REPL without a running process");
    return nullptr;
  }

  SymbolContextList sc_list;
  target.GetImages().FindSymbolsWithNameAndType(ConstString("_swift_release"),
                                                eSymbolTypeAny, sc_list);

  if (!sc_list.GetSize()) {
    err.SetErrorString("can't launch a Swift REPL in a process that doesn't "
                       "have the Swift standard library");
    return nullptr;
  }

  // Check that we can get a type system, or we aren't going anywhere:
  TypeSystem *type_system = target.GetScratchTypeSystemForLanguage(
      nullptr, eLanguageTypeSwift, true, repl_options);
  if (!type_system) {
    err.SetErrorString("Could not construct an expression "
                       "context for the REPL.\n");
    return nullptr;
  }

  // Sanity checks succeeded.  Go ahead.
  auto repl_sp = lldb::REPLSP(new SwiftREPL(target));
  repl_sp->SetCompilerOptions(repl_options);
  return repl_sp;
}

lldb::REPLSP SwiftREPL::CreateInstanceFromDebugger(Status &err,
                                                   Debugger &debugger,
                                                   const char *repl_options) {
  const char *bp_name = "repl_main";

  FileSpec repl_executable = HostInfo::GetSupportExeDir();

  if (!repl_executable) {
    err.SetErrorString("unable to locate REPL executable");
    return nullptr;
  }

  repl_executable.GetFilename().SetCString("repl_swift");
  std::string repl_exe_path(repl_executable.GetPath());

  if (!FileSystem::Instance().Exists(repl_executable)) {
    err.SetErrorStringWithFormat("REPL executable does not exist: '%s'",
                                 repl_exe_path.c_str());
    return nullptr;
  }

  llvm::Triple target_triple = HostInfo::GetArchitecture().GetTriple();
  llvm::SmallString<16> os_name;
  llvm::raw_svector_ostream os(os_name);
  // Use the most generic sub-architecture.
  target_triple.setArch(target_triple.getArch());
  // Override the stub's minimum deployment target to the host os version.
  llvm::VersionTuple version = HostInfo::GetOSVersion();
  os << llvm::Triple::getOSTypeName(target_triple.getOS());
  os << version.getAsString();
  target_triple.setOSName(os.str());

  TargetSP target_sp;
  err = debugger.GetTargetList().CreateTarget(
      debugger, repl_exe_path.c_str(), target_triple.getTriple(),
      eLoadDependentsYes, nullptr, target_sp);
  if (!err.Success()) {
    err.SetErrorStringWithFormat("failed to create REPL target: %s",
                                 err.AsCString());
    return nullptr;
  }

  // Limit the breakpoint to our executable module
  ModuleSP exe_module_sp(target_sp->GetExecutableModule());
  if (!exe_module_sp) {
    err.SetErrorString("unable to resolve REPL executable module");
    target_sp->Destroy();
    return nullptr;
  }

  FileSpecList containingModules;
  containingModules.Append(exe_module_sp->GetFileSpec());

  BreakpointSP main_bp_sp = target_sp->CreateBreakpoint(
      &containingModules,    // Limit to these modules
      NULL,                  // Don't limit the breakpoint to any source files
      bp_name,               // Function name
      eFunctionNameTypeAuto, // Name type
      eLanguageTypeUnknown,  // Language
      0,                     // offset
      eLazyBoolYes,          // skip_prologue,
      true,                  // internal
      false);                // request_hardware

  if (main_bp_sp->GetNumLocations() == 0) {
    err.SetErrorStringWithFormat("failed to resolve REPL breakpoint for '%s'",
                                 bp_name);
    return nullptr;
  }

  main_bp_sp->SetBreakpointKind("REPL");
  assert(main_bp_sp->IsInternal()); // We made an internal
                                    // breakpoint above, it better
                                    // say it is internal

  lldb_private::ProcessLaunchInfo launch_info;
  llvm::StringRef target_settings_argv0 = target_sp->GetArg0();

  if (target_sp->GetDisableASLR())
    launch_info.GetFlags().Set(eLaunchFlagDisableASLR);

  if (target_sp->GetDisableSTDIO())
    launch_info.GetFlags().Set(eLaunchFlagDisableSTDIO);

  if (!target_settings_argv0.empty()) {
    launch_info.GetArguments().AppendArgument(target_settings_argv0);
    launch_info.SetExecutableFile(exe_module_sp->GetPlatformFileSpec(), false);
  } else {
    launch_info.SetExecutableFile(exe_module_sp->GetPlatformFileSpec(), true);
  }

  debugger.SetAsyncExecution(false);
  err = target_sp->Launch(launch_info, nullptr);
  debugger.SetAsyncExecution(true);

  if (!err.Success()) {
    err.SetErrorStringWithFormat("failed to launch REPL process: %s",
                                 err.AsCString());
    return nullptr;
  }

  ProcessSP process_sp(target_sp->GetProcessSP());
  if (!process_sp) {
    err.SetErrorString("failed to launch REPL process");
    return nullptr;
  }

  // Start handling process events automatically
  debugger.StartEventHandlerThread();

  // Destroy the process and the event handler thread after a fatal error.
  CleanUp cleanup{[&] {
    process_sp->Destroy(/*force_kill=*/false);
    debugger.StopEventHandlerThread();
  }};

  StateType state = process_sp->GetState();

  if (state != eStateStopped) {
    err.SetErrorString("failed to stop process at REPL breakpoint");
    return nullptr;
  }

  ThreadList &thread_list = process_sp->GetThreadList();
  const uint32_t num_threads = thread_list.GetSize();
  if (num_threads == 0) {
    err.SetErrorString("process is not in valid state (no threads)");
    return nullptr;
  }

  ThreadSP thread_sp = thread_list.GetSelectedThread();
  if (!thread_sp) {
    thread_sp = thread_list.GetThreadAtIndex(0);
    thread_list.SetSelectedThreadByID(thread_sp->GetID());
    assert(thread_sp && "There should be at least one thread");
  }

  thread_sp->SetSelectedFrameByIndex(0);

  REPLSP repl_sp(new SwiftREPL(*target_sp));
  repl_sp->SetCompilerOptions(repl_options);
  target_sp->SetREPL(lldb::eLanguageTypeSwift, repl_sp);

  // Check that we can get a type system, or we aren't
  // going anywhere.  Remember to pass in the repl_options
  // in case they set up framework paths we need, etc.
  TypeSystem *type_system = target_sp->GetScratchTypeSystemForLanguage(
      nullptr, eLanguageTypeSwift, true, repl_options);
  if (!type_system) {
    err.SetErrorString("Could not construct an expression"
                       " context for the REPL.\n");
    return nullptr;
  }

  // Disable the cleanup, since we have a valid repl session now.
  cleanup.disable();

  std::string swift_full_version(swift::version::getSwiftFullVersion());
  printf("Welcome to %s.\nType :help for assistance.\n",
         swift_full_version.c_str());

  return repl_sp;
}

void SwiftREPL::EnumerateSupportedLanguages(
    std::set<lldb::LanguageType> &languages) {
  languages.insert(lldb::eLanguageTypeSwift);
}

void SwiftREPL::Initialize() {
  PluginManager::RegisterPlugin(ConstString("swift"), "The Swift REPL",
                                &CreateInstance, &EnumerateSupportedLanguages);
}

void SwiftREPL::Terminate() {
  PluginManager::UnregisterPlugin(&CreateInstance);
}

SwiftREPL::SwiftREPL(Target &target)
    : REPL(LLVMCastKind::eKindSwift, target), m_swift_ast_sp() {}

SwiftREPL::~SwiftREPL() {}

Status SwiftREPL::DoInitialization() {
  Status error;

  if (!m_compiler_options.empty()) {
    (void)m_target.GetScratchTypeSystemForLanguage(
        &error, eLanguageTypeSwift, true, m_compiler_options.c_str());
  }

  return error;
}

ConstString SwiftREPL::GetSourceFileBasename() {
  static ConstString s_basename("repl.swift");
  return s_basename;
}

bool SwiftREPL::SourceIsComplete(const std::string &source) {
  std::unique_ptr<llvm::MemoryBuffer> source_buffer_ap(
      llvm::MemoryBuffer::getMemBuffer(source));
  swift::ide::SourceCompleteResult result =
      swift::ide::isSourceInputComplete(std::move(source_buffer_ap),
                                        swift::SourceFileKind::REPL);
  return result.IsComplete;
}

static bool GetIdentifier(llvm::StringRef &s, llvm::StringRef &identifier) {
  if (!s.empty()) {
    char ch = s[0];
    if (isalpha(ch) || ch == '_') {
      size_t i;
      for (i = 1; i < s.size(); ++i) {
        ch = s[i];
        if (isalnum(ch) || ch == '_')
          continue;
        else
          break;
      }
      identifier = s.substr(0, i);
      s = s.drop_front(i);
      return true;
    }
  }
  return false;
}

lldb::offset_t SwiftREPL::GetDesiredIndentation(const StringList &lines,
                                                int cursor_position,
                                                int tab_size) {
  // Determine appropriate indentation after the initial N-1 lines
  StringList prior_lines = lines;
  prior_lines.PopBack();
  std::string source_string(prior_lines.CopyList());
  std::unique_ptr<llvm::MemoryBuffer> source_buffer_ap(
      llvm::MemoryBuffer::getMemBuffer(source_string));
  swift::ide::SourceCompleteResult result =
      swift::ide::isSourceInputComplete(std::move(source_buffer_ap),
                                        swift::SourceFileKind::REPL);

  int desired_indent =
      (result.IndentLevel * tab_size) + result.IndentPrefix.length();

  const std::string &last_line = lines[lines.GetSize() - 1];

  // Unindent for an initial closed brace on a line break, or when the brace was
  // just typed
  if (cursor_position == 0 || last_line[cursor_position - 1] == '}') {

    // The brace must be the first non-space character
    const size_t actual_indent = REPL::CalculateActualIndentation(lines);

    if (last_line.length() > actual_indent && last_line[actual_indent] == '}') {
      // Stop searching once a reason to unindent was found
      desired_indent -= tab_size;
      if (desired_indent < 0)
        desired_indent = 0;
      return desired_indent;
    }
  }

  // Unindent for a case clause on a line break, or when the colon was just
  // typed
  if (cursor_position == 0 || last_line[cursor_position - 1] == ':') {
    size_t colon_pos = last_line.find_last_of(':');
    if (colon_pos != std::string::npos) {
      std::string line_to_colon = last_line.substr(0, colon_pos);
      llvm::StringRef line(line_to_colon);
      line = line.ltrim(); // Skip leading spaces
      if (!line.empty()) {
        llvm::StringRef identifier;
        if (GetIdentifier(line, identifier)) {
          line = line.ltrim(); // Skip leading spaces
          // If we have an empty line now, we have a simple label followed by a
          // ':'
          // and if it isn't we have "case" followed by a space, then we handle
          // this as a cast statement
          bool outdent = false;
          if (line.empty())
            outdent = !identifier.equals("case");
          else if (identifier.equals("case")) {
            outdent = true;
          } else {
            line = line.rtrim(); // Skip trailing spaces
            // Check for any identifier followed by an optional paren expression
            // and a colon
            if (line.front() == '(' && line.back() == ')')
              outdent = true;
          }

          if (outdent) {
            // Stop searching once a reason to unindent was found
            desired_indent -= tab_size;
            if (desired_indent < 0)
              desired_indent = 0;
            return desired_indent;
          }
        }
      }
    }
  }

  // Otherwise, only change indentation when creating a new line
  if (cursor_position == 0)
    return desired_indent;

  return LLDB_INVALID_OFFSET;
}

lldb::LanguageType SwiftREPL::GetLanguage() { return eLanguageTypeSwift; }

bool isThrownError(ValueObjectSP valobj_sp) {
  ConstString name = valobj_sp->GetName();
  size_t length = name.GetLength();
  if (length < 3)
    return false;

  const char *name_cstr = name.AsCString();
  if (name_cstr[0] != '$')
    return false;
  if (name_cstr[1] != 'E')
    return false;
  for (size_t index = 2; index < length; index++) {

    char digit = name_cstr[index];
    if (digit < '0' || digit > '9')
      return false;
  }
  return true;
}

bool SwiftREPL::PrintOneVariable(Debugger &debugger, StreamFileSP &output_sp,
                                 ValueObjectSP &valobj_sp,
                                 ExpressionVariable *var) {
  bool is_computed = false;

  if (var) {
    if (lldb::ValueObjectSP valobj_sp = var->GetValueObject()) {
      Flags valobj_type_flags(valobj_sp->GetCompilerType().GetTypeInfo());
      const bool is_swift(valobj_type_flags.AllSet(eTypeIsSwift));
      if ((var->GetName().AsCString("anonymous")[0] != '$') && is_swift) {
        is_computed = llvm::cast<SwiftExpressionVariable>(var)->GetIsComputed();
      } else {
        return false;
      }
    } else {
      return false;
    }
  }

  const bool colorize_out = output_sp->GetFile().GetIsTerminalWithColors();

  bool handled = false;

  Format format = m_format_options.GetFormat();

  bool treat_as_void = (format == eFormatVoid);
  // if we are asked to suppress void, check if this is the empty tuple type,
  // and if so suppress it
  if (!treat_as_void && !debugger.GetNotifyVoid()) {
    const CompilerType &expr_type(valobj_sp->GetCompilerType());
    Flags expr_type_flags(expr_type.GetTypeInfo());
    if (expr_type_flags.AllSet(eTypeIsSwift | eTypeIsTuple)) {
      treat_as_void = (expr_type.GetNumFields() == 0);
    }
  }

  if (!treat_as_void) {
    if (format != eFormatDefault)
      valobj_sp->SetFormat(format);

    DumpValueObjectOptions options;
    options.SetUseDynamicType(lldb::eDynamicCanRunTarget);
    options.SetMaximumPointerDepth(
        {DumpValueObjectOptions::PointerDepth::Mode::Formatters, 1});
    options.SetUseSyntheticValue(true);
    options.SetRevealEmptyAggregates(false);
    options.SetHidePointerValue(true);
    options.SetVariableFormatDisplayLanguage(lldb::eLanguageTypeSwift);
    options.SetDeclPrintingHelper([](ConstString type_name,
                                     ConstString var_name,
                                     const DumpValueObjectOptions &options,
                                     Stream &stream) -> bool {
      if (!type_name || !var_name)
        return false;

      std::string type_name_str(type_name ? type_name.GetCString() : "");
      for (auto iter = type_name_str.find(" *"); iter != std::string::npos;
           iter = type_name_str.find(" *")) {
        type_name_str.erase(iter, 2);
      }
      if (!type_name_str.empty()) {
        stream.Printf("%s: %s =", var_name.GetCString(), type_name_str.c_str());
        return true;
      }

      return false;
    });

    if (is_computed) {
      StringSummaryFormat::Flags flags;
      flags.SetDontShowChildren(true);
      flags.SetDontShowValue(true);
      flags.SetHideItemNames(true);
      flags.SetShowMembersOneLiner(false);
      flags.SetSkipPointers(false);
      flags.SetSkipReferences(false);
      options.SetHideValue(true);
      options.SetShowSummary(true);
      options.SetSummary(lldb::TypeSummaryImplSP(
          new StringSummaryFormat(flags, "<computed property>")));
    }

    if (colorize_out) {
      const char *color = isThrownError(valobj_sp)
                              ? ANSI_ESCAPE1(ANSI_FG_COLOR_RED)
                              : ANSI_ESCAPE1(ANSI_FG_COLOR_CYAN);
      fprintf(output_sp->GetFile().GetStream(), "%s", color);
    }

    valobj_sp->Dump(*output_sp, options);

    if (colorize_out)
      fprintf(output_sp->GetFile().GetStream(), ANSI_ESCAPE1(ANSI_CTRL_NORMAL));

    handled = true;
  }

  return handled;
}

int SwiftREPL::CompleteCode(const std::string &current_code,
                            lldb_private::StringList &matches) {
  //----------------------------------------------------------------------g
  // If we use the target's SwiftASTContext for completion, it reaaallly
  // slows down subsequent expressions. The compiler team doesn't have time
  // to fix this issue currently, so we need to work around it by making
  // our own copy of the AST and using this separate AST for completion.
  //----------------------------------------------------------------------
  Status error;
  if (!m_swift_ast_sp) {
    SwiftASTContext *target_swift_ast = llvm::dyn_cast_or_null<SwiftASTContext>(
        m_target.GetScratchTypeSystemForLanguage(&error, eLanguageTypeSwift));
    if (target_swift_ast)
      m_swift_ast_sp.reset(new SwiftASTContext(*target_swift_ast));
  }
  SwiftASTContext *swift_ast = m_swift_ast_sp.get();

  if (swift_ast) {
    swift::ASTContext *ast = swift_ast->GetASTContext();
    swift::REPLCompletions completions;
    static ConstString g_repl_module_name("repl");
    swift::ModuleDecl *repl_module =
        swift_ast->GetModule(g_repl_module_name, error);
    if (repl_module == NULL) {
      repl_module = swift_ast->CreateModule(g_repl_module_name, error);
      const swift::SourceFile::ImplicitModuleImportKind implicit_import_kind =
          swift::SourceFile::ImplicitModuleImportKind::Stdlib;
      llvm::Optional<unsigned> bufferID;
      swift::SourceFile *repl_source_file = new (*ast)
          swift::SourceFile(*repl_module, swift::SourceFileKind::REPL, bufferID,
                            implicit_import_kind, /*Keep tokens*/false);
      repl_module->addFile(*repl_source_file);
    }
    if (repl_module) {
      swift::SourceFile &repl_source_file =
          repl_module->getMainSourceFile(swift::SourceFileKind::REPL);

      llvm::StringRef current_code_ref(current_code);
      completions.populate(repl_source_file, current_code_ref);
      llvm::StringRef root = completions.getRoot();
      if (!root.empty()) {
        matches.AppendString(root.data(), root.size());
        return 1;
      }
      // Otherwise, advance through the completion state machine.
      const swift::CompletionState completion_state = completions.getState();
      switch (completion_state) {
      case swift::CompletionState::CompletedRoot: {
        // We completed the root. Next step is to display the completion list.
        matches.AppendString(""); // Empty string to indicate no completion,
        // just display other strings that come after it
        llvm::ArrayRef<llvm::StringRef> llvm_matches =
            completions.getCompletionList();
        for (const auto &llvm_match : llvm_matches) {
          if (!llvm_match.empty())
            matches.AppendString(llvm_match.data(), llvm_match.size());
        }
        // Don't include the empty string we appended above or we will display
        // one
        // too many we need to return the magical value of one less than our
        // actual matches.
        // TODO: modify all IOHandlerDelegate::IOHandlerComplete() to use a
        // CompletionMatches
        // class that wraps up the "StringList matches;" along with other smarts
        // so we don't
        // have to return magic values and incorrect sizes.
        return matches.GetSize() - 1;
      } break;

      case swift::CompletionState::DisplayedCompletionList: {
        // Complete the next completion stem in the cycle.
        llvm::StringRef stem = completions.getPreviousStem().InsertableString;
        matches.AppendString(stem.data(), stem.size());
      } break;

      case swift::CompletionState::Empty:
      case swift::CompletionState::Unique:
        // We already provided a definitive completion--nothing else to do.
        break;

      case swift::CompletionState::Invalid:
        llvm_unreachable("got an invalid completion set?!");
      }
    }
  }

  return matches.GetSize();
}
