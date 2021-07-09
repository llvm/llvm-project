//===-- ClangUserExpression.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Config.h"

#include <cstdio>
#if HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#include <cstdlib>
#include <map>
#include <string>

#include "ClangUserExpression.h"

#include "ASTResultSynthesizer.h"
#include "ClangASTMetadata.h"
#include "ClangDiagnostic.h"
#include "ClangExpressionDeclMap.h"
#include "ClangExpressionParser.h"
#include "ClangModulesDeclVendor.h"
#include "ClangPersistentVariables.h"
#include "CppModuleConfiguration.h"

#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Expression/ExpressionSourceCode.h"
#include "lldb/Expression/IRExecutionUnit.h"
#include "lldb/Expression/IRInterpreter.h"
#include "lldb/Expression/Materializer.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Symbol/Block.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/ThreadPlan.h"
#include "lldb/Target/ThreadPlanCallUserExpression.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/StreamString.h"

#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"

#include "llvm/ADT/ScopeExit.h"

using namespace lldb_private;

char ClangUserExpression::ID;

ClangUserExpression::ClangUserExpression(
    ExecutionContextScope &exe_scope, llvm::StringRef expr,
    llvm::StringRef prefix, lldb::LanguageType language,
    ResultType desired_type, const EvaluateExpressionOptions &options,
    ValueObject *ctx_obj)
    : LLVMUserExpression(exe_scope, expr, prefix, language, desired_type,
                         options),
      m_type_system_helper(*m_target_wp.lock(), options.GetExecutionPolicy() ==
                                                    eExecutionPolicyTopLevel),
      m_result_delegate(exe_scope.CalculateTarget()), m_ctx_obj(ctx_obj) {
  switch (m_language) {
  case lldb::eLanguageTypeC_plus_plus:
    m_allow_cxx = true;
    break;
  case lldb::eLanguageTypeObjC:
    m_allow_objc = true;
    break;
  case lldb::eLanguageTypeObjC_plus_plus:
  default:
    m_allow_cxx = true;
    m_allow_objc = true;
    break;
  }
}

ClangUserExpression::~ClangUserExpression() = default;

void ClangUserExpression::ScanContext(ExecutionContext &exe_ctx, Status &err) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));

  LLDB_LOGF(log, "ClangUserExpression::ScanContext()");

  m_target = exe_ctx.GetTargetPtr();

  if (!(m_allow_cxx || m_allow_objc)) {
    LLDB_LOGF(log, "  [CUE::SC] Settings inhibit C++ and Objective-C");
    return;
  }

  StackFrame *frame = exe_ctx.GetFramePtr();
  if (frame == nullptr) {
    LLDB_LOGF(log, "  [CUE::SC] Null stack frame");
    return;
  }

  SymbolContext sym_ctx = frame->GetSymbolContext(lldb::eSymbolContextFunction |
                                                  lldb::eSymbolContextBlock);

  if (!sym_ctx.function) {
    LLDB_LOGF(log, "  [CUE::SC] Null function");
    return;
  }

  // Find the block that defines the function represented by "sym_ctx"
  Block *function_block = sym_ctx.GetFunctionBlock();

  if (!function_block) {
    LLDB_LOGF(log, "  [CUE::SC] Null function block");
    return;
  }

  CompilerDeclContext decl_context = function_block->GetDeclContext();

  if (!decl_context) {
    LLDB_LOGF(log, "  [CUE::SC] Null decl context");
    return;
  }

  if (m_ctx_obj) {
    switch (m_ctx_obj->GetObjectRuntimeLanguage()) {
    case lldb::eLanguageTypeC:
    case lldb::eLanguageTypeC89:
    case lldb::eLanguageTypeC99:
    case lldb::eLanguageTypeC11:
    case lldb::eLanguageTypeC_plus_plus:
    case lldb::eLanguageTypeC_plus_plus_03:
    case lldb::eLanguageTypeC_plus_plus_11:
    case lldb::eLanguageTypeC_plus_plus_14:
      m_in_cplusplus_method = true;
      break;
    case lldb::eLanguageTypeObjC:
    case lldb::eLanguageTypeObjC_plus_plus:
      m_in_objectivec_method = true;
      break;
    default:
      break;
    }
    m_needs_object_ptr = true;
  } else if (clang::CXXMethodDecl *method_decl =
          TypeSystemClang::DeclContextGetAsCXXMethodDecl(decl_context)) {
    if (m_allow_cxx && method_decl->isInstance()) {
      if (m_enforce_valid_object) {
        lldb::VariableListSP variable_list_sp(
            function_block->GetBlockVariableList(true));

        const char *thisErrorString = "Stopped in a C++ method, but 'this' "
                                      "isn't available; pretending we are in a "
                                      "generic context";

        if (!variable_list_sp) {
          err.SetErrorString(thisErrorString);
          return;
        }

        lldb::VariableSP this_var_sp(
            variable_list_sp->FindVariable(ConstString("this")));

        if (!this_var_sp || !this_var_sp->IsInScope(frame) ||
            !this_var_sp->LocationIsValidForFrame(frame)) {
          err.SetErrorString(thisErrorString);
          return;
        }
      }

      m_in_cplusplus_method = true;
      m_needs_object_ptr = true;
    }
  } else if (clang::ObjCMethodDecl *method_decl =
                 TypeSystemClang::DeclContextGetAsObjCMethodDecl(
                     decl_context)) {
    if (m_allow_objc) {
      if (m_enforce_valid_object) {
        lldb::VariableListSP variable_list_sp(
            function_block->GetBlockVariableList(true));

        const char *selfErrorString = "Stopped in an Objective-C method, but "
                                      "'self' isn't available; pretending we "
                                      "are in a generic context";

        if (!variable_list_sp) {
          err.SetErrorString(selfErrorString);
          return;
        }

        lldb::VariableSP self_variable_sp =
            variable_list_sp->FindVariable(ConstString("self"));

        if (!self_variable_sp || !self_variable_sp->IsInScope(frame) ||
            !self_variable_sp->LocationIsValidForFrame(frame)) {
          err.SetErrorString(selfErrorString);
          return;
        }
      }

      m_in_objectivec_method = true;
      m_needs_object_ptr = true;

      if (!method_decl->isInstanceMethod())
        m_in_static_method = true;
    }
  } else if (clang::FunctionDecl *function_decl =
                 TypeSystemClang::DeclContextGetAsFunctionDecl(decl_context)) {
    // We might also have a function that said in the debug information that it
    // captured an object pointer.  The best way to deal with getting to the
    // ivars at present is by pretending that this is a method of a class in
    // whatever runtime the debug info says the object pointer belongs to.  Do
    // that here.

    ClangASTMetadata *metadata =
        TypeSystemClang::DeclContextGetMetaData(decl_context, function_decl);
    if (metadata && metadata->HasObjectPtr()) {
      lldb::LanguageType language = metadata->GetObjectPtrLanguage();
      if (language == lldb::eLanguageTypeC_plus_plus) {
        if (m_enforce_valid_object) {
          lldb::VariableListSP variable_list_sp(
              function_block->GetBlockVariableList(true));

          const char *thisErrorString = "Stopped in a context claiming to "
                                        "capture a C++ object pointer, but "
                                        "'this' isn't available; pretending we "
                                        "are in a generic context";

          if (!variable_list_sp) {
            err.SetErrorString(thisErrorString);
            return;
          }

          lldb::VariableSP this_var_sp(
              variable_list_sp->FindVariable(ConstString("this")));

          if (!this_var_sp || !this_var_sp->IsInScope(frame) ||
              !this_var_sp->LocationIsValidForFrame(frame)) {
            err.SetErrorString(thisErrorString);
            return;
          }
        }

        m_in_cplusplus_method = true;
        m_needs_object_ptr = true;
      } else if (language == lldb::eLanguageTypeObjC) {
        if (m_enforce_valid_object) {
          lldb::VariableListSP variable_list_sp(
              function_block->GetBlockVariableList(true));

          const char *selfErrorString =
              "Stopped in a context claiming to capture an Objective-C object "
              "pointer, but 'self' isn't available; pretending we are in a "
              "generic context";

          if (!variable_list_sp) {
            err.SetErrorString(selfErrorString);
            return;
          }

          lldb::VariableSP self_variable_sp =
              variable_list_sp->FindVariable(ConstString("self"));

          if (!self_variable_sp || !self_variable_sp->IsInScope(frame) ||
              !self_variable_sp->LocationIsValidForFrame(frame)) {
            err.SetErrorString(selfErrorString);
            return;
          }

          Type *self_type = self_variable_sp->GetType();

          if (!self_type) {
            err.SetErrorString(selfErrorString);
            return;
          }

          CompilerType self_clang_type = self_type->GetForwardCompilerType();

          if (!self_clang_type) {
            err.SetErrorString(selfErrorString);
            return;
          }

          if (TypeSystemClang::IsObjCClassType(self_clang_type)) {
            return;
          } else if (TypeSystemClang::IsObjCObjectPointerType(
                         self_clang_type)) {
            m_in_objectivec_method = true;
            m_needs_object_ptr = true;
          } else {
            err.SetErrorString(selfErrorString);
            return;
          }
        } else {
          m_in_objectivec_method = true;
          m_needs_object_ptr = true;
        }
      }
    }
  }
}

// This is a really nasty hack, meant to fix Objective-C expressions of the
// form (int)[myArray count].  Right now, because the type information for
// count is not available, [myArray count] returns id, which can't be directly
// cast to int without causing a clang error.
static void ApplyObjcCastHack(std::string &expr) {
  const std::string from = "(int)[";
  const std::string to = "(int)(long long)[";

  size_t offset;

  while ((offset = expr.find(from)) != expr.npos)
    expr.replace(offset, from.size(), to);
}

bool ClangUserExpression::SetupPersistentState(DiagnosticManager &diagnostic_manager,
                                 ExecutionContext &exe_ctx) {
  if (Target *target = exe_ctx.GetTargetPtr()) {
    if (PersistentExpressionState *persistent_state =
            target->GetPersistentExpressionStateForLanguage(
                lldb::eLanguageTypeC)) {
      m_clang_state = llvm::cast<ClangPersistentVariables>(persistent_state);
      m_result_delegate.RegisterPersistentState(persistent_state);
    } else {
      diagnostic_manager.PutString(
          eDiagnosticSeverityError,
          "couldn't start parsing (no persistent data)");
      return false;
    }
  } else {
    diagnostic_manager.PutString(eDiagnosticSeverityError,
                                 "error: couldn't start parsing (no target)");
    return false;
  }
  return true;
}

static void SetupDeclVendor(ExecutionContext &exe_ctx, Target *target,
                            DiagnosticManager &diagnostic_manager) {
  if (!target->GetEnableAutoImportClangModules())
    return;

  auto *persistent_state = llvm::cast<ClangPersistentVariables>(
      target->GetPersistentExpressionStateForLanguage(lldb::eLanguageTypeC));
  if (!persistent_state)
    return;

  std::shared_ptr<ClangModulesDeclVendor> decl_vendor =
      persistent_state->GetClangModulesDeclVendor();
  if (!decl_vendor)
    return;

  StackFrame *frame = exe_ctx.GetFramePtr();
  if (!frame)
    return;

  Block *block = frame->GetFrameBlock();
  if (!block)
    return;
  SymbolContext sc;

  block->CalculateSymbolContext(&sc);

  if (!sc.comp_unit)
    return;
  StreamString error_stream;

  ClangModulesDeclVendor::ModuleVector modules_for_macros =
      persistent_state->GetHandLoadedClangModules();
  if (decl_vendor->AddModulesForCompileUnit(*sc.comp_unit, modules_for_macros,
                                            error_stream))
    return;

  // Failed to load some modules, so emit the error stream as a diagnostic.
  if (!error_stream.Empty()) {
    // The error stream already contains several Clang diagnostics that might
    // be either errors or warnings, so just print them all as one remark
    // diagnostic to prevent that the message starts with "error: error:".
    diagnostic_manager.PutString(eDiagnosticSeverityRemark,
                                 error_stream.GetString());
    return;
  }

  diagnostic_manager.PutString(eDiagnosticSeverityError,
                               "Unknown error while loading modules needed for "
                               "current compilation unit.");
}

ClangExpressionSourceCode::WrapKind ClangUserExpression::GetWrapKind() const {
  assert(m_options.GetExecutionPolicy() != eExecutionPolicyTopLevel &&
         "Top level expressions aren't wrapped.");
  using Kind = ClangExpressionSourceCode::WrapKind;
  if (m_in_cplusplus_method)
    return Kind::CppMemberFunction;
  else if (m_in_objectivec_method) {
    if (m_in_static_method)
      return Kind::ObjCStaticMethod;
    return Kind::ObjCInstanceMethod;
  }
  // Not in any kind of 'special' function, so just wrap it in a normal C
  // function.
  return Kind::Function;
}

void ClangUserExpression::CreateSourceCode(
    DiagnosticManager &diagnostic_manager, ExecutionContext &exe_ctx,
    std::vector<std::string> modules_to_import, bool for_completion) {

  std::string prefix = m_expr_prefix;

  if (m_options.GetExecutionPolicy() == eExecutionPolicyTopLevel) {
    m_transformed_text = m_expr_text;
  } else {
    m_source_code.reset(ClangExpressionSourceCode::CreateWrapped(
        m_filename, prefix, m_expr_text, GetWrapKind()));

    if (!m_source_code->GetText(m_transformed_text, exe_ctx, !m_ctx_obj,
                                for_completion, modules_to_import)) {
      diagnostic_manager.PutString(eDiagnosticSeverityError,
                                   "couldn't construct expression body");
      return;
    }

    // Find and store the start position of the original code inside the
    // transformed code. We need this later for the code completion.
    std::size_t original_start;
    std::size_t original_end;
    bool found_bounds = m_source_code->GetOriginalBodyBounds(
        m_transformed_text, original_start, original_end);
    if (found_bounds)
      m_user_expression_start_pos = original_start;
  }
}

static bool SupportsCxxModuleImport(lldb::LanguageType language) {
  switch (language) {
  case lldb::eLanguageTypeC_plus_plus:
  case lldb::eLanguageTypeC_plus_plus_03:
  case lldb::eLanguageTypeC_plus_plus_11:
  case lldb::eLanguageTypeC_plus_plus_14:
  case lldb::eLanguageTypeObjC_plus_plus:
    return true;
  default:
    return false;
  }
}

/// Utility method that puts a message into the expression log and
/// returns an invalid module configuration.
static CppModuleConfiguration LogConfigError(const std::string &msg) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));
  LLDB_LOG(log, "[C++ module config] {0}", msg);
  return CppModuleConfiguration();
}

CppModuleConfiguration GetModuleConfig(lldb::LanguageType language,
                                       ExecutionContext &exe_ctx) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));

  // Don't do anything if this is not a C++ module configuration.
  if (!SupportsCxxModuleImport(language))
    return LogConfigError("Language doesn't support C++ modules");

  Target *target = exe_ctx.GetTargetPtr();
  if (!target)
    return LogConfigError("No target");

  StackFrame *frame = exe_ctx.GetFramePtr();
  if (!frame)
    return LogConfigError("No frame");

  Block *block = frame->GetFrameBlock();
  if (!block)
    return LogConfigError("No block");

  SymbolContext sc;
  block->CalculateSymbolContext(&sc);
  if (!sc.comp_unit)
    return LogConfigError("Couldn't calculate symbol context");

  // Build a list of files we need to analyze to build the configuration.
  FileSpecList files;
  for (const FileSpec &f : sc.comp_unit->GetSupportFiles())
    files.AppendIfUnique(f);
  // We also need to look at external modules in the case of -gmodules as they
  // contain the support files for libc++ and the C library.
  llvm::DenseSet<SymbolFile *> visited_symbol_files;
  sc.comp_unit->ForEachExternalModule(
      visited_symbol_files, [&files](Module &module) {
        for (std::size_t i = 0; i < module.GetNumCompileUnits(); ++i) {
          const FileSpecList &support_files =
              module.GetCompileUnitAtIndex(i)->GetSupportFiles();
          for (const FileSpec &f : support_files) {
            files.AppendIfUnique(f);
          }
        }
        return false;
      });

  LLDB_LOG(log, "[C++ module config] Found {0} support files to analyze",
           files.GetSize());
  if (log && log->GetVerbose()) {
    for (const FileSpec &f : files)
      LLDB_LOGV(log, "[C++ module config] Analyzing support file: {0}",
                f.GetPath());
  }

  // Try to create a configuration from the files. If there is no valid
  // configuration possible with the files, this just returns an invalid
  // configuration.
  return CppModuleConfiguration(files);
}

bool ClangUserExpression::PrepareForParsing(
    DiagnosticManager &diagnostic_manager, ExecutionContext &exe_ctx,
    bool for_completion) {
  InstallContext(exe_ctx);

  if (!SetupPersistentState(diagnostic_manager, exe_ctx))
    return false;

  Status err;
  ScanContext(exe_ctx, err);

  if (!err.Success()) {
    diagnostic_manager.PutString(eDiagnosticSeverityWarning, err.AsCString());
  }

  ////////////////////////////////////
  // Generate the expression
  //

  ApplyObjcCastHack(m_expr_text);

  SetupDeclVendor(exe_ctx, m_target, diagnostic_manager);

  m_filename = m_clang_state->GetNextExprFileName();

  if (m_target->GetImportStdModule() == eImportStdModuleTrue)
    SetupCppModuleImports(exe_ctx);

  CreateSourceCode(diagnostic_manager, exe_ctx, m_imported_cpp_modules,
                   for_completion);
  return true;
}

bool ClangUserExpression::TryParse(
    DiagnosticManager &diagnostic_manager, ExecutionContextScope *exe_scope,
    ExecutionContext &exe_ctx, lldb_private::ExecutionPolicy execution_policy,
    bool keep_result_in_memory, bool generate_debug_info) {
  m_materializer_up = std::make_unique<Materializer>();

  ResetDeclMap(exe_ctx, m_result_delegate, keep_result_in_memory);

  auto on_exit = llvm::make_scope_exit([this]() { ResetDeclMap(); });

  if (!DeclMap()->WillParse(exe_ctx, GetMaterializer())) {
    diagnostic_manager.PutString(
        eDiagnosticSeverityError,
        "current process state is unsuitable for expression parsing");
    return false;
  }

  if (m_options.GetExecutionPolicy() == eExecutionPolicyTopLevel) {
    DeclMap()->SetLookupsEnabled(true);
  }

  m_parser = std::make_unique<ClangExpressionParser>(
      exe_scope, *this, generate_debug_info, m_include_directories, m_filename);

  unsigned num_errors = m_parser->Parse(diagnostic_manager);

  // Check here for FixItHints.  If there are any try to apply the fixits and
  // set the fixed text in m_fixed_text before returning an error.
  if (num_errors) {
    if (diagnostic_manager.HasFixIts()) {
      if (m_parser->RewriteExpression(diagnostic_manager)) {
        size_t fixed_start;
        size_t fixed_end;
        m_fixed_text = diagnostic_manager.GetFixedExpression();
        // Retrieve the original expression in case we don't have a top level
        // expression (which has no surrounding source code).
        if (m_source_code && m_source_code->GetOriginalBodyBounds(
                                 m_fixed_text, fixed_start, fixed_end))
          m_fixed_text =
              m_fixed_text.substr(fixed_start, fixed_end - fixed_start);
      }
    }
    return false;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Prepare the output of the parser for execution, evaluating it statically
  // if possible
  //

  {
    Status jit_error = m_parser->PrepareForExecution(
        m_jit_start_addr, m_jit_end_addr, m_execution_unit_sp, exe_ctx,
        m_can_interpret, execution_policy);

    if (!jit_error.Success()) {
      const char *error_cstr = jit_error.AsCString();
      if (error_cstr && error_cstr[0])
        diagnostic_manager.PutString(eDiagnosticSeverityError, error_cstr);
      else
        diagnostic_manager.PutString(eDiagnosticSeverityError,
                                     "expression can't be interpreted or run");
      return false;
    }
  }
  return true;
}

void ClangUserExpression::SetupCppModuleImports(ExecutionContext &exe_ctx) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));

  CppModuleConfiguration module_config = GetModuleConfig(m_language, exe_ctx);
  m_imported_cpp_modules = module_config.GetImportedModules();
  m_include_directories = module_config.GetIncludeDirs();

  LLDB_LOG(log, "List of imported modules in expression: {0}",
           llvm::make_range(m_imported_cpp_modules.begin(),
                            m_imported_cpp_modules.end()));
  LLDB_LOG(log, "List of include directories gathered for modules: {0}",
           llvm::make_range(m_include_directories.begin(),
                            m_include_directories.end()));
}

static bool shouldRetryWithCppModule(Target &target, ExecutionPolicy exe_policy) {
  // Top-level expression don't yet support importing C++ modules.
  if (exe_policy == ExecutionPolicy::eExecutionPolicyTopLevel)
    return false;
  return target.GetImportStdModule() == eImportStdModuleFallback;
}

bool ClangUserExpression::Parse(DiagnosticManager &diagnostic_manager,
                                ExecutionContext &exe_ctx,
                                lldb_private::ExecutionPolicy execution_policy,
                                bool keep_result_in_memory,
                                bool generate_debug_info) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));

  if (!PrepareForParsing(diagnostic_manager, exe_ctx, /*for_completion*/ false))
    return false;

  LLDB_LOGF(log, "Parsing the following code:\n%s", m_transformed_text.c_str());

  ////////////////////////////////////
  // Set up the target and compiler
  //

  Target *target = exe_ctx.GetTargetPtr();

  if (!target) {
    diagnostic_manager.PutString(eDiagnosticSeverityError, "invalid target");
    return false;
  }

  //////////////////////////
  // Parse the expression
  //

  Process *process = exe_ctx.GetProcessPtr();
  ExecutionContextScope *exe_scope = process;

  if (!exe_scope)
    exe_scope = exe_ctx.GetTargetPtr();

  bool parse_success = TryParse(diagnostic_manager, exe_scope, exe_ctx,
                                execution_policy, keep_result_in_memory,
                                generate_debug_info);
  // If the expression failed to parse, check if retrying parsing with a loaded
  // C++ module is possible.
  if (!parse_success && shouldRetryWithCppModule(*target, execution_policy)) {
    // Load the loaded C++ modules.
    SetupCppModuleImports(exe_ctx);
    // If we did load any modules, then retry parsing.
    if (!m_imported_cpp_modules.empty()) {
      // The module imports are injected into the source code wrapper,
      // so recreate those.
      CreateSourceCode(diagnostic_manager, exe_ctx, m_imported_cpp_modules,
                       /*for_completion*/ false);
      // Clear the error diagnostics from the previous parse attempt.
      diagnostic_manager.Clear();
      parse_success = TryParse(diagnostic_manager, exe_scope, exe_ctx,
                               execution_policy, keep_result_in_memory,
                               generate_debug_info);
    }
  }
  if (!parse_success)
    return false;

  if (exe_ctx.GetProcessPtr() && execution_policy == eExecutionPolicyTopLevel) {
    Status static_init_error =
        m_parser->RunStaticInitializers(m_execution_unit_sp, exe_ctx);

    if (!static_init_error.Success()) {
      const char *error_cstr = static_init_error.AsCString();
      if (error_cstr && error_cstr[0])
        diagnostic_manager.Printf(eDiagnosticSeverityError,
                                  "%s\n",
                                  error_cstr);
      else
        diagnostic_manager.PutString(eDiagnosticSeverityError,
                                     "couldn't run static initializers\n");
      return false;
    }
  }

  if (m_execution_unit_sp) {
    bool register_execution_unit = false;

    if (m_options.GetExecutionPolicy() == eExecutionPolicyTopLevel) {
      register_execution_unit = true;
    }

    // If there is more than one external function in the execution unit, it
    // needs to keep living even if it's not top level, because the result
    // could refer to that function.

    if (m_execution_unit_sp->GetJittedFunctions().size() > 1) {
      register_execution_unit = true;
    }

    if (register_execution_unit) {
      if (auto *persistent_state =
              exe_ctx.GetTargetPtr()->GetPersistentExpressionStateForLanguage(
                  m_language))
        persistent_state->RegisterExecutionUnit(m_execution_unit_sp);
    }
  }

  if (generate_debug_info) {
    lldb::ModuleSP jit_module_sp(m_execution_unit_sp->GetJITModule());

    if (jit_module_sp) {
      ConstString const_func_name(FunctionName());
      FileSpec jit_file;
      jit_file.GetFilename() = const_func_name;
      jit_module_sp->SetFileSpecAndObjectName(jit_file, ConstString());
      m_jit_module_wp = jit_module_sp;
      target->GetImages().Append(jit_module_sp);
    }
  }

  if (process && m_jit_start_addr != LLDB_INVALID_ADDRESS)
    m_jit_process_wp = lldb::ProcessWP(process->shared_from_this());
  return true;
}

/// Converts an absolute position inside a given code string into
/// a column/line pair.
///
/// \param[in] abs_pos
///     A absolute position in the code string that we want to convert
///     to a column/line pair.
///
/// \param[in] code
///     A multi-line string usually representing source code.
///
/// \param[out] line
///     The line in the code that contains the given absolute position.
///     The first line in the string is indexed as 1.
///
/// \param[out] column
///     The column in the line that contains the absolute position.
///     The first character in a line is indexed as 0.
static void AbsPosToLineColumnPos(size_t abs_pos, llvm::StringRef code,
                                  unsigned &line, unsigned &column) {
  // Reset to code position to beginning of the file.
  line = 0;
  column = 0;

  assert(abs_pos <= code.size() && "Absolute position outside code string?");

  // We have to walk up to the position and count lines/columns.
  for (std::size_t i = 0; i < abs_pos; ++i) {
    // If we hit a line break, we go back to column 0 and enter a new line.
    // We only handle \n because that's what we internally use to make new
    // lines for our temporary code strings.
    if (code[i] == '\n') {
      ++line;
      column = 0;
      continue;
    }
    ++column;
  }
}

bool ClangUserExpression::Complete(ExecutionContext &exe_ctx,
                                   CompletionRequest &request,
                                   unsigned complete_pos) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));

  // We don't want any visible feedback when completing an expression. Mostly
  // because the results we get from an incomplete invocation are probably not
  // correct.
  DiagnosticManager diagnostic_manager;

  if (!PrepareForParsing(diagnostic_manager, exe_ctx, /*for_completion*/ true))
    return false;

  LLDB_LOGF(log, "Parsing the following code:\n%s", m_transformed_text.c_str());

  //////////////////////////
  // Parse the expression
  //

  m_materializer_up = std::make_unique<Materializer>();

  ResetDeclMap(exe_ctx, m_result_delegate, /*keep result in memory*/ true);

  auto on_exit = llvm::make_scope_exit([this]() { ResetDeclMap(); });

  if (!DeclMap()->WillParse(exe_ctx, GetMaterializer())) {
    diagnostic_manager.PutString(
        eDiagnosticSeverityError,
        "current process state is unsuitable for expression parsing");

    return false;
  }

  if (m_options.GetExecutionPolicy() == eExecutionPolicyTopLevel) {
    DeclMap()->SetLookupsEnabled(true);
  }

  Process *process = exe_ctx.GetProcessPtr();
  ExecutionContextScope *exe_scope = process;

  if (!exe_scope)
    exe_scope = exe_ctx.GetTargetPtr();

  ClangExpressionParser parser(exe_scope, *this, false);

  // We have to find the source code location where the user text is inside
  // the transformed expression code. When creating the transformed text, we
  // already stored the absolute position in the m_transformed_text string. The
  // only thing left to do is to transform it into the line:column format that
  // Clang expects.

  // The line and column of the user expression inside the transformed source
  // code.
  unsigned user_expr_line, user_expr_column;
  if (m_user_expression_start_pos.hasValue())
    AbsPosToLineColumnPos(*m_user_expression_start_pos, m_transformed_text,
                          user_expr_line, user_expr_column);
  else
    return false;

  // The actual column where we have to complete is the start column of the
  // user expression + the offset inside the user code that we were given.
  const unsigned completion_column = user_expr_column + complete_pos;
  parser.Complete(request, user_expr_line, completion_column, complete_pos);

  return true;
}

bool ClangUserExpression::AddArguments(ExecutionContext &exe_ctx,
                                       std::vector<lldb::addr_t> &args,
                                       lldb::addr_t struct_address,
                                       DiagnosticManager &diagnostic_manager) {
  lldb::addr_t object_ptr = LLDB_INVALID_ADDRESS;
  lldb::addr_t cmd_ptr = LLDB_INVALID_ADDRESS;

  if (m_needs_object_ptr) {
    lldb::StackFrameSP frame_sp = exe_ctx.GetFrameSP();
    if (!frame_sp)
      return true;

    ConstString object_name;

    if (m_in_cplusplus_method) {
      object_name.SetCString("this");
    } else if (m_in_objectivec_method) {
      object_name.SetCString("self");
    } else {
      diagnostic_manager.PutString(
          eDiagnosticSeverityError,
          "need object pointer but don't know the language");
      return false;
    }

    Status object_ptr_error;

    if (m_ctx_obj) {
      AddressType address_type;
      object_ptr = m_ctx_obj->GetAddressOf(false, &address_type);
      if (object_ptr == LLDB_INVALID_ADDRESS ||
          address_type != eAddressTypeLoad)
        object_ptr_error.SetErrorString("Can't get context object's "
                                        "debuggee address");
    } else
      object_ptr = GetObjectPointer(frame_sp, object_name, object_ptr_error);

    if (!object_ptr_error.Success()) {
      exe_ctx.GetTargetRef().GetDebugger().GetAsyncOutputStream()->Printf(
          "warning: `%s' is not accessible (substituting 0)\n",
          object_name.AsCString());
      object_ptr = 0;
    }

    if (m_in_objectivec_method) {
      ConstString cmd_name("_cmd");

      cmd_ptr = GetObjectPointer(frame_sp, cmd_name, object_ptr_error);

      if (!object_ptr_error.Success()) {
        diagnostic_manager.Printf(
            eDiagnosticSeverityWarning,
            "couldn't get cmd pointer (substituting NULL): %s",
            object_ptr_error.AsCString());
        cmd_ptr = 0;
      }
    }

    args.push_back(object_ptr);

    if (m_in_objectivec_method)
      args.push_back(cmd_ptr);

    args.push_back(struct_address);
  } else {
    args.push_back(struct_address);
  }
  return true;
}

lldb::ExpressionVariableSP ClangUserExpression::GetResultAfterDematerialization(
    ExecutionContextScope *exe_scope) {
  return m_result_delegate.GetVariable();
}

void ClangUserExpression::ClangUserExpressionHelper::ResetDeclMap(
    ExecutionContext &exe_ctx,
    Materializer::PersistentVariableDelegate &delegate,
    bool keep_result_in_memory,
    ValueObject *ctx_obj) {
  std::shared_ptr<ClangASTImporter> ast_importer;
  auto *state = exe_ctx.GetTargetSP()->GetPersistentExpressionStateForLanguage(
      lldb::eLanguageTypeC);
  if (state) {
    auto *persistent_vars = llvm::cast<ClangPersistentVariables>(state);
    ast_importer = persistent_vars->GetClangASTImporter();
  }
  m_expr_decl_map_up = std::make_unique<ClangExpressionDeclMap>(
      keep_result_in_memory, &delegate, exe_ctx.GetTargetSP(), ast_importer,
      ctx_obj);
}

clang::ASTConsumer *
ClangUserExpression::ClangUserExpressionHelper::ASTTransformer(
    clang::ASTConsumer *passthrough) {
  m_result_synthesizer_up = std::make_unique<ASTResultSynthesizer>(
      passthrough, m_top_level, m_target);

  return m_result_synthesizer_up.get();
}

void ClangUserExpression::ClangUserExpressionHelper::CommitPersistentDecls() {
  if (m_result_synthesizer_up) {
    m_result_synthesizer_up->CommitPersistentDecls();
  }
}

ConstString ClangUserExpression::ResultDelegate::GetName() {
  return m_persistent_state->GetNextPersistentVariableName(false);
}

void ClangUserExpression::ResultDelegate::DidDematerialize(
    lldb::ExpressionVariableSP &variable) {
  m_variable = variable;
}

void ClangUserExpression::ResultDelegate::RegisterPersistentState(
    PersistentExpressionState *persistent_state) {
  m_persistent_state = persistent_state;
}

lldb::ExpressionVariableSP &ClangUserExpression::ResultDelegate::GetVariable() {
  return m_variable;
}
