//===-- ClangUserExpression.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#if HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#include <cstdlib>
#include <map>
#include <string>

#include "ClangUserExpression.h"

#include "ASTResultSynthesizer.h"
#include "ClangDiagnostic.h"
#include "ClangExpressionDeclMap.h"
#include "ClangExpressionParser.h"
#include "ClangModulesDeclVendor.h"
#include "ClangPersistentVariables.h"

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
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/ClangExternalASTSourceCommon.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/ObjectFile.h"
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

using namespace lldb_private;

ClangUserExpression::ClangUserExpression(
    ExecutionContextScope &exe_scope, llvm::StringRef expr,
    llvm::StringRef prefix, lldb::LanguageType language,
    ResultType desired_type, const EvaluateExpressionOptions &options)
    : LLVMUserExpression(exe_scope, expr, prefix, language, desired_type,
                         options),
      m_type_system_helper(*m_target_wp.lock().get(),
                           options.GetExecutionPolicy() ==
                               eExecutionPolicyTopLevel) {
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

ClangUserExpression::~ClangUserExpression() {}

void ClangUserExpression::ScanContext(ExecutionContext &exe_ctx, Status &err) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));

  if (log)
    log->Printf("ClangUserExpression::ScanContext()");

  m_target = exe_ctx.GetTargetPtr();

  if (!(m_allow_cxx || m_allow_objc)) {
    if (log)
      log->Printf("  [CUE::SC] Settings inhibit C++ and Objective-C");
    return;
  }

  StackFrame *frame = exe_ctx.GetFramePtr();
  if (frame == NULL) {
    if (log)
      log->Printf("  [CUE::SC] Null stack frame");
    return;
  }

  SymbolContext sym_ctx = frame->GetSymbolContext(lldb::eSymbolContextFunction |
                                                  lldb::eSymbolContextBlock);

  if (!sym_ctx.function) {
    if (log)
      log->Printf("  [CUE::SC] Null function");
    return;
  }

  // Find the block that defines the function represented by "sym_ctx"
  Block *function_block = sym_ctx.GetFunctionBlock();

  if (!function_block) {
    if (log)
      log->Printf("  [CUE::SC] Null function block");
    return;
  }

  CompilerDeclContext decl_context = function_block->GetDeclContext();

  if (!decl_context) {
    if (log)
      log->Printf("  [CUE::SC] Null decl context");
    return;
  }

  if (clang::CXXMethodDecl *method_decl =
          ClangASTContext::DeclContextGetAsCXXMethodDecl(decl_context)) {
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
                 ClangASTContext::DeclContextGetAsObjCMethodDecl(
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
                 ClangASTContext::DeclContextGetAsFunctionDecl(decl_context)) {
    // We might also have a function that said in the debug information that it
    // captured an
    // object pointer.  The best way to deal with getting to the ivars at
    // present is by pretending
    // that this is a method of a class in whatever runtime the debug info says
    // the object pointer
    // belongs to.  Do that here.

    ClangASTMetadata *metadata =
        ClangASTContext::DeclContextGetMetaData(decl_context, function_decl);
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

          if (ClangASTContext::IsObjCClassType(self_clang_type)) {
            return;
          } else if (ClangASTContext::IsObjCObjectPointerType(
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

// This is a really nasty hack, meant to fix Objective-C expressions of the form
// (int)[myArray count].  Right now, because the type information for count is
// not available, [myArray count] returns id, which can't be directly cast to
// int without causing a clang error.
static void ApplyObjcCastHack(std::string &expr) {
#define OBJC_CAST_HACK_FROM "(int)["
#define OBJC_CAST_HACK_TO "(int)(long long)["

  size_t from_offset;

  while ((from_offset = expr.find(OBJC_CAST_HACK_FROM)) != expr.npos)
    expr.replace(from_offset, sizeof(OBJC_CAST_HACK_FROM) - 1,
                 OBJC_CAST_HACK_TO);

#undef OBJC_CAST_HACK_TO
#undef OBJC_CAST_HACK_FROM
}

bool ClangUserExpression::Parse(DiagnosticManager &diagnostic_manager,
                                ExecutionContext &exe_ctx,
                                lldb_private::ExecutionPolicy execution_policy,
                                bool keep_result_in_memory,
                                bool generate_debug_info) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));

  Status err;

  InstallContext(exe_ctx);

  if (Target *target = exe_ctx.GetTargetPtr()) {
    if (PersistentExpressionState *persistent_state =
            target->GetPersistentExpressionStateForLanguage(
                lldb::eLanguageTypeC)) {
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

  ScanContext(exe_ctx, err);

  if (!err.Success()) {
    diagnostic_manager.PutString(eDiagnosticSeverityWarning, err.AsCString());
  }

  ////////////////////////////////////
  // Generate the expression
  //

  ApplyObjcCastHack(m_expr_text);
  // ApplyUnicharHack(m_expr_text);

  std::string prefix = m_expr_prefix;

  if (ClangModulesDeclVendor *decl_vendor =
          m_target->GetClangModulesDeclVendor()) {
    const ClangModulesDeclVendor::ModuleVector &hand_imported_modules =
        llvm::cast<ClangPersistentVariables>(
            m_target->GetPersistentExpressionStateForLanguage(
                lldb::eLanguageTypeC))
            ->GetHandLoadedClangModules();
    ClangModulesDeclVendor::ModuleVector modules_for_macros;

    for (ClangModulesDeclVendor::ModuleID module : hand_imported_modules) {
      modules_for_macros.push_back(module);
    }

    if (m_target->GetEnableAutoImportClangModules()) {
      if (StackFrame *frame = exe_ctx.GetFramePtr()) {
        if (Block *block = frame->GetFrameBlock()) {
          SymbolContext sc;

          block->CalculateSymbolContext(&sc);

          if (sc.comp_unit) {
            StreamString error_stream;

            decl_vendor->AddModulesForCompileUnit(
                *sc.comp_unit, modules_for_macros, error_stream);
          }
        }
      }
    }
  }

  lldb::LanguageType lang_type = lldb::eLanguageTypeUnknown;

  if (m_options.GetExecutionPolicy() == eExecutionPolicyTopLevel) {
    m_transformed_text = m_expr_text;
  } else {
    std::unique_ptr<ExpressionSourceCode> source_code(
        ExpressionSourceCode::CreateWrapped(prefix.c_str(),
                                            m_expr_text.c_str()));

    if (m_in_cplusplus_method)
      lang_type = lldb::eLanguageTypeC_plus_plus;
    else if (m_in_objectivec_method)
      lang_type = lldb::eLanguageTypeObjC;
    else
      lang_type = lldb::eLanguageTypeC;

    if (!source_code->GetText(m_transformed_text, lang_type, m_in_static_method,
                              exe_ctx)) {
      diagnostic_manager.PutString(eDiagnosticSeverityError,
                                   "couldn't construct expression body");
      return false;
    }
  }

  if (log)
    log->Printf("Parsing the following code:\n%s", m_transformed_text.c_str());

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

  m_materializer_ap.reset(new Materializer());

  ResetDeclMap(exe_ctx, m_result_delegate, keep_result_in_memory);

  class OnExit {
  public:
    typedef std::function<void(void)> Callback;

    OnExit(Callback const &callback) : m_callback(callback) {}

    ~OnExit() { m_callback(); }

  private:
    Callback m_callback;
  };

  OnExit on_exit([this]() { ResetDeclMap(); });

  if (!DeclMap()->WillParse(exe_ctx, m_materializer_ap.get())) {
    diagnostic_manager.PutString(
        eDiagnosticSeverityError,
        "current process state is unsuitable for expression parsing");

    ResetDeclMap(); // We are being careful here in the case of breakpoint
                    // conditions.

    return false;
  }

  if (m_options.GetExecutionPolicy() == eExecutionPolicyTopLevel) {
    DeclMap()->SetLookupsEnabled(true);
  }

  Process *process = exe_ctx.GetProcessPtr();
  ExecutionContextScope *exe_scope = process;

  if (!exe_scope)
    exe_scope = exe_ctx.GetTargetPtr();

  // We use a shared pointer here so we can use the original parser - if it
  // succeeds
  // or the rewrite parser we might make if it fails.  But the parser_sp will
  // never be empty.

  ClangExpressionParser parser(exe_scope, *this, generate_debug_info);

  unsigned num_errors = parser.Parse(diagnostic_manager);

  // Check here for FixItHints.  If there are any try to apply the fixits and
  // set the fixed text in m_fixed_text
  // before returning an error.
  if (num_errors) {
    if (diagnostic_manager.HasFixIts()) {
      if (parser.RewriteExpression(diagnostic_manager)) {
        size_t fixed_start;
        size_t fixed_end;
        const std::string &fixed_expression =
            diagnostic_manager.GetFixedExpression();
        if (ExpressionSourceCode::GetOriginalBodyBounds(
                fixed_expression, lang_type, fixed_start, fixed_end))
          m_fixed_text =
              fixed_expression.substr(fixed_start, fixed_end - fixed_start);
      }
    }

    ResetDeclMap(); // We are being careful here in the case of breakpoint
                    // conditions.

    return false;
  }

  //////////////////////////////////////////////////////////////////////////////////////////
  // Prepare the output of the parser for execution, evaluating it statically if
  // possible
  //

  {
    Status jit_error = parser.PrepareForExecution(
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

  if (exe_ctx.GetProcessPtr() && execution_policy == eExecutionPolicyTopLevel) {
    Status static_init_error =
        parser.RunStaticInitializers(m_execution_unit_sp, exe_ctx);

    if (!static_init_error.Success()) {
      const char *error_cstr = static_init_error.AsCString();
      if (error_cstr && error_cstr[0])
        diagnostic_manager.Printf(eDiagnosticSeverityError,
                                  "couldn't run static initializers: %s\n",
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

    // If there is more than one external function in the execution
    // unit, it needs to keep living even if it's not top level, because
    // the result could refer to that function.

    if (m_execution_unit_sp->GetJittedFunctions().size() > 1) {
      register_execution_unit = true;
    }

    if (register_execution_unit) {
      llvm::cast<PersistentExpressionState>(
          exe_ctx.GetTargetPtr()->GetPersistentExpressionStateForLanguage(
              m_language))
          ->RegisterExecutionUnit(m_execution_unit_sp);
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

  ResetDeclMap(); // Make this go away since we don't need any of its state
                  // after parsing.  This also gets rid of any
                  // ClangASTImporter::Minions.

  if (process && m_jit_start_addr != LLDB_INVALID_ADDRESS)
    m_jit_process_wp = lldb::ProcessWP(process->shared_from_this());
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
    bool keep_result_in_memory) {
  m_expr_decl_map_up.reset(
      new ClangExpressionDeclMap(keep_result_in_memory, &delegate, exe_ctx));
}

clang::ASTConsumer *
ClangUserExpression::ClangUserExpressionHelper::ASTTransformer(
    clang::ASTConsumer *passthrough) {
  m_result_synthesizer_up.reset(
      new ASTResultSynthesizer(passthrough, m_top_level, m_target));

  return m_result_synthesizer_up.get();
}

void ClangUserExpression::ClangUserExpressionHelper::CommitPersistentDecls() {
  if (m_result_synthesizer_up.get()) {
    m_result_synthesizer_up->CommitPersistentDecls();
  }
}

ClangUserExpression::ResultDelegate::ResultDelegate() {}

ConstString ClangUserExpression::ResultDelegate::GetName() {
  return m_persistent_state->GetNextPersistentVariableName();
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
