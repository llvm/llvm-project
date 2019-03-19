//===-- SwiftUserExpression.cpp ---------------------------------*- C++ -*-===//
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

#include <stdio.h>
#if HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#include "SwiftExpressionParser.h"
#include "SwiftREPLMaterializer.h"

#include "lldb/Core/Module.h"
#include "lldb/Expression/DiagnosticManager.h"
#include "lldb/Expression/ExpressionParser.h"
#include "lldb/Expression/ExpressionSourceCode.h"
#include "lldb/Expression/IRExecutionUnit.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/Variable.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/SwiftLanguageRuntime.h"
#include "lldb/Utility/LLDBAssert.h"
#include "lldb/Utility/Log.h"

#include "swift/AST/Type.h"
#include "swift/AST/Types.h"

#include <cstdlib>
#include <map>
#include <string>

#include "SwiftUserExpression.h"

using namespace lldb_private;

SwiftUserExpression::SwiftUserExpression(
    ExecutionContextScope &exe_scope, llvm::StringRef expr,
    llvm::StringRef prefix, lldb::LanguageType language,
    ResultType desired_type, const EvaluateExpressionOptions &options)
    : LLVMUserExpression(exe_scope, expr, prefix, language, desired_type,
                         options),
      m_type_system_helper(*m_target_wp.lock().get()),
      m_result_delegate(exe_scope.CalculateTarget(), *this, false),
      m_error_delegate(exe_scope.CalculateTarget(), *this, true),
      m_persistent_variable_delegate(*this) {
  m_runs_in_playground_or_repl =
      options.GetREPLEnabled() || options.GetPlaygroundTransformEnabled();
}

SwiftUserExpression::~SwiftUserExpression() {}

void SwiftUserExpression::WillStartExecuting() {
  if (auto process = m_jit_process_wp.lock()) {
    if (auto *swift_runtime = process->GetSwiftLanguageRuntime())
      swift_runtime->WillStartExecutingUserExpression(
          m_runs_in_playground_or_repl);
    else
      llvm_unreachable("Can't execute a swift expression without a runtime");
  } else
    llvm_unreachable("Can't execute an expression without a process");
}

void SwiftUserExpression::DidFinishExecuting() {
  if (auto process = m_jit_process_wp.lock()) {
    if (auto swift_runtime = process->GetSwiftLanguageRuntime())
      swift_runtime->DidFinishExecutingUserExpression(
          m_runs_in_playground_or_repl);
    else
      llvm_unreachable("Can't execute a swift expression without a runtime");
  }
}

static CompilerType GetConcreteType(ExecutionContext &exe_ctx,
                                    StackFrame *frame, CompilerType type) {
  auto swift_type = GetSwiftType(type.GetOpaqueQualType());
  StreamString type_name;
  if (SwiftLanguageRuntime::GetAbstractTypeName(type_name, swift_type)) {
    auto *runtime = exe_ctx.GetProcessRef().GetSwiftLanguageRuntime();
    return runtime->GetConcreteType(frame, ConstString(type_name.GetString()));
  }
  return type;
}

void SwiftUserExpression::ScanContext(ExecutionContext &exe_ctx, Status &err) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));

  if (log)
    log->Printf("SwiftUserExpression::ScanContext()");

  m_target = exe_ctx.GetTargetPtr();

  if (!m_target) {
    if (log)
      log->Printf("  [SUE::SC] Null target");
    return;
  }

  StackFrame *frame = exe_ctx.GetFramePtr();
  if (frame == NULL) {
    if (log)
      log->Printf("  [SUE::SC] Null stack frame");
    return;
  }

  SymbolContext sym_ctx = frame->GetSymbolContext(
      lldb::eSymbolContextFunction | lldb::eSymbolContextBlock |
      lldb::eSymbolContextCompUnit | lldb::eSymbolContextSymbol);

  // This stage of the scan is only for Swift, but when we are going
  // to do Swift evaluation we need to do this scan.
  // So be sure to cover both cases:
  // 1) When the language is eLanguageTypeUnknown, to determine if this IS Swift
  // 2) When the language is explicitly set to eLanguageTypeSwift.
  bool frame_is_swift = false;

  if (sym_ctx.comp_unit && (m_language == lldb::eLanguageTypeUnknown ||
                            m_language == lldb::eLanguageTypeSwift)) {
    if (sym_ctx.comp_unit->GetLanguage() == lldb::eLanguageTypeSwift ||
        sym_ctx.comp_unit->GetLanguage() == lldb::eLanguageTypePLI)
      frame_is_swift = true;
  } else if (sym_ctx.symbol && m_language == lldb::eLanguageTypeUnknown) {
    if (sym_ctx.symbol->GetMangled().GuessLanguage() ==
        lldb::eLanguageTypeSwift)
      frame_is_swift = true;
  }

  if (frame_is_swift) {
    m_language_flags &= ~eLanguageFlagIsClass;
    m_language_flags &= ~eLanguageFlagNeedsObjectPointer;

    // Make sure the target's SwiftASTContext has been setup before
    // doing any Swift name lookups.
    if (m_target) {
      auto swift_ast_ctx = m_target->GetScratchSwiftASTContext(err, *frame);
      if (!swift_ast_ctx) {
        if (log)
          log->Printf("  [SUE::SC] NULL Swift AST Context");
        return;
      }

      if (!swift_ast_ctx->GetClangImporter()) {
        if (log)
          log->Printf("  [SUE::SC] Swift AST Context has no Clang importer");
        return;
      }

      if (swift_ast_ctx->HasFatalErrors()) {
        if (log)
          log->Printf("  [SUE::SC] Swift AST Context has fatal errors");
        return;
      }
    }

    if (log)
      log->Printf("  [SUE::SC] Compilation unit is swift");

    Block *function_block = sym_ctx.GetFunctionBlock();

    if (function_block) {
      lldb::VariableListSP variable_list_sp(
          function_block->GetBlockVariableList(true));

      if (variable_list_sp) {
        lldb::VariableSP self_var_sp(
            variable_list_sp->FindVariable(ConstString("self")));

        do {
          if (!self_var_sp)
            break;

          CompilerType self_type;

          lldb::StackFrameSP stack_frame_sp = exe_ctx.GetFrameSP();

          if (stack_frame_sp) {
            // If we have a self variable, but it has no location at
            // the current PC, then we can't use it.  Set the self var
            // back to empty and we'll just pretend we are in a
            // regular frame, which is really the best we can do.
            if (!self_var_sp->LocationIsValidForFrame(stack_frame_sp.get())) {
              self_var_sp.reset();
              break;
            }

            lldb::ValueObjectSP valobj_sp =
                stack_frame_sp->GetValueObjectForFrameVariable(
                    self_var_sp, lldb::eDynamicDontRunTarget);

            if (valobj_sp && valobj_sp->GetError().Success())
              self_type = valobj_sp->GetCompilerType();
          }

          if (!self_type.IsValid()) {
            Type *self_lldb_type = self_var_sp->GetType();

            if (self_lldb_type)
              self_type = self_var_sp->GetType()->GetForwardCompilerType();
          }

          if (!self_type.IsValid()) {
            // If the self_type is invalid at this point, reset it.
            // Code below the phony do/while will assume the existence
            // of this var means something, but it is useless in this
            // condition.
            self_var_sp.reset();
            break;
          }

          // Check to see if we are in a class func of a class (or
          // static func of a struct) and adjust our self_type to
          // point to the instance type.
          m_language_flags |= eLanguageFlagNeedsObjectPointer;

          Flags self_type_flags(self_type.GetTypeInfo());

          if (self_type_flags.AllSet(lldb::eTypeIsSwift |
                                     lldb::eTypeIsMetatype)) {
            self_type = self_type.GetInstanceType();
            self_type_flags = self_type.GetTypeInfo();
            if (self_type_flags.Test(lldb::eTypeIsClass))
              m_language_flags |= eLanguageFlagIsClass;
            m_language_flags |= eLanguageFlagInStaticMethod;
          }

          if (self_type_flags.AllSet(lldb::eTypeIsSwift |
                                     lldb::eTypeInstanceIsPointer)) {
            if (self_type_flags.Test(lldb::eTypeIsClass))
              m_language_flags |= eLanguageFlagIsClass;
          }

          swift::Type object_type = GetSwiftType(self_type);

          if (object_type.getPointer() &&
              (object_type.getPointer() != self_type.GetOpaqueQualType()))
            self_type = CompilerType(self_type.GetTypeSystem(),
                                     object_type.getPointer());

          // Handle weak self.
          if (auto *ref_type = llvm::dyn_cast<swift::ReferenceStorageType>(
                  GetSwiftType(self_type).getPointer())) {
            if (ref_type->getOwnership() == swift::ReferenceOwnership::Weak) {
              m_language_flags |= eLanguageFlagIsClass;
              m_language_flags |= eLanguageFlagIsWeakSelf;
            }
          }

          if (Flags(self_type.GetTypeInfo())
                  .AllSet(lldb::eTypeIsSwift | lldb::eTypeIsStructUnion |
                          lldb::eTypeIsGeneric) &&
              self_type_flags.AllSet(lldb::eTypeIsSwift |
                                     lldb::eTypeIsReference |
                                     lldb::eTypeHasValue)) {
            // We can't extend generic structs when "self" is mutating at the
            // moment.
            m_language_flags &= ~eLanguageFlagNeedsObjectPointer;
            self_var_sp.reset();
            break;
          }

          if (log)
            log->Printf("  [SUE::SC] Containing class name: %s",
                        self_type.GetTypeName().AsCString());
        } while (0);
      }
    }
  }
}

static SwiftPersistentExpressionState *
GetPersistentState(Target *target, ExecutionContext &exe_ctx) {
  auto exe_scope = exe_ctx.GetBestExecutionContextScope();
  if (!exe_scope)
    return nullptr;
  return target->GetSwiftPersistentExpressionState(*exe_scope);
}

bool SwiftUserExpression::Parse(DiagnosticManager &diagnostic_manager,
                                ExecutionContext &exe_ctx,
                                lldb_private::ExecutionPolicy execution_policy,
                                bool keep_result_in_memory,
                                bool generate_debug_info) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));

  Status err;

  InstallContext(exe_ctx);
  Target *target = exe_ctx.GetTargetPtr();
  if (!target) {
    diagnostic_manager.PutString(eDiagnosticSeverityError,
                                 "couldn't start parsing (no target)");
    return false;
  }
  if (auto *persistent_state = GetPersistentState(target, exe_ctx)) {
    persistent_state->AddHandLoadedModule(ConstString("Swift"));
    m_result_delegate.RegisterPersistentState(persistent_state);
    m_error_delegate.RegisterPersistentState(persistent_state);
  } else {
    diagnostic_manager.PutString(eDiagnosticSeverityError,
                                 "couldn't start parsing (no persistent data)");
    return false;
  }

  ScanContext(exe_ctx, err);

  if (!err.Success()) {
    diagnostic_manager.Printf(eDiagnosticSeverityError, "warning: %s\n",
                              err.AsCString());
  }

  StreamString m_transformed_stream;

  //
  // Generate the expression.
  //

  std::string prefix = m_expr_prefix;

  std::unique_ptr<ExpressionSourceCode> source_code(
      ExpressionSourceCode::CreateWrapped(prefix.c_str(), m_expr_text.c_str()));

  const lldb::LanguageType lang_type = lldb::eLanguageTypeSwift;

  m_options.SetLanguage(lang_type);
  uint32_t first_body_line = 0;

  if (!source_code->GetText(m_transformed_text, lang_type, m_language_flags,
                            m_options, exe_ctx, first_body_line)) {
    diagnostic_manager.PutString(eDiagnosticSeverityError,
                                  "couldn't construct expression body");
    return false;
  }

  if (log)
    log->Printf("Parsing the following code:\n%s", m_transformed_text.c_str());

  //
  // Parse the expression.
  //

  if (m_options.GetREPLEnabled())
    m_materializer_up.reset(new SwiftREPLMaterializer());
  else
    m_materializer_up.reset(new Materializer());

  class OnExit {
  public:
    typedef std::function<void(void)> Callback;

    OnExit(Callback const &callback) : m_callback(callback) {}

    ~OnExit() { m_callback(); }

  private:
    Callback m_callback;
  };

  ExecutionContextScope *exe_scope = NULL;

  Process *process = exe_ctx.GetProcessPtr();

  do {
    exe_scope = exe_ctx.GetFramePtr();
    if (exe_scope)
      break;

    exe_scope = process;
    if (exe_scope)
      break;

    exe_scope = exe_ctx.GetTargetPtr();
  } while (0);

  m_parser =
      llvm::make_unique<SwiftExpressionParser>(exe_scope, *this, m_options);

  unsigned error_code = m_parser->Parse(
      diagnostic_manager, first_body_line,
      first_body_line + source_code->GetNumBodyLines());

  if (error_code == 2) {
    m_fixed_text = m_expr_text;
    return false;
  } else if (error_code) {
    // Calculate the fixed expression string at this point:
    if (diagnostic_manager.HasFixIts()) {
      if (m_parser->RewriteExpression(diagnostic_manager)) {
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
    return false;
  }


  // Prepare the output of the parser for execution, evaluating it
  // statically if possible.
  Status jit_error = m_parser->PrepareForExecution(
      m_jit_start_addr, m_jit_end_addr, m_execution_unit_sp, exe_ctx,
      m_can_interpret, execution_policy);

  if (m_execution_unit_sp) {
    if (m_options.GetREPLEnabled()) {
      llvm::cast<SwiftREPLMaterializer>(m_materializer_up.get())
          ->RegisterExecutionUnit(m_execution_unit_sp.get());
    }

    bool register_execution_unit = false;

    if (m_options.GetREPLEnabled()) {
      if (!m_execution_unit_sp->GetJittedFunctions().empty() ||
          !m_execution_unit_sp->GetJittedGlobalVariables().empty()) {
        register_execution_unit = true;
      }
    } else {
      if (m_execution_unit_sp->GetJittedFunctions().size() > 1 ||
          m_execution_unit_sp->GetJittedGlobalVariables().size() > 1) {
        register_execution_unit = true;
      }
    }

    if (register_execution_unit) {
      // We currently key off there being more than one external
      // function in the execution unit to determine whether it needs
      // to live in the process.
      GetPersistentState(exe_ctx.GetTargetPtr(), exe_ctx)
          ->RegisterExecutionUnit(m_execution_unit_sp);
    }
  }

  if (m_options.GetGenerateDebugInfo()) {
    StreamString jit_module_name;
    jit_module_name.Printf("%s%u", FunctionName(),
                           m_options.GetExpressionNumber());
    const char *limit_file = m_options.GetPoundLineFilePath();
    FileSpec limit_file_spec;
    uint32_t limit_start_line = 0;
    uint32_t limit_end_line = 0;
    if (limit_file) {
      limit_file_spec.SetFile(limit_file, FileSpec::Style::native);
      limit_start_line = m_options.GetPoundLineLine();
      limit_end_line = limit_start_line +
                       std::count(m_expr_text.begin(), m_expr_text.end(), '\n');
    }
    m_execution_unit_sp->CreateJITModule(jit_module_name.GetString().data(),
                                         limit_file ? &limit_file_spec : NULL,
                                         limit_start_line, limit_end_line);
  }

  if (jit_error.Success()) {
    if (process && m_jit_start_addr != LLDB_INVALID_ADDRESS)
      m_jit_process_wp = lldb::ProcessWP(process->shared_from_this());
    return true;
  } else {
    const char *error_cstr = jit_error.AsCString();
    if (error_cstr && error_cstr[0])
      diagnostic_manager.PutString(eDiagnosticSeverityError, error_cstr);
    else
      diagnostic_manager.PutString(eDiagnosticSeverityError,
                                    "expression can't be interpreted or run\n");
    return false;
  }
}

bool SwiftUserExpression::AddArguments(ExecutionContext &exe_ctx,
                                       std::vector<lldb::addr_t> &args,
                                       lldb::addr_t struct_address,
                                       DiagnosticManager &diagnostic_manager) {
  lldb::addr_t object_ptr = LLDB_INVALID_ADDRESS;

  if (m_language_flags & eLanguageFlagNeedsObjectPointer) {
    lldb::StackFrameSP frame_sp = exe_ctx.GetFrameSP();
    if (!frame_sp)
      return true;

    ConstString object_name("self");

    Status object_ptr_error;

    object_ptr = GetObjectPointer(frame_sp, object_name, object_ptr_error);

    if (!object_ptr_error.Success()) {
      diagnostic_manager.Printf(
          eDiagnosticSeverityWarning,
          "couldn't get required object pointer (substituting NULL): %s\n",
          object_ptr_error.AsCString());
      object_ptr = 0;
    }

    if (m_options.GetPlaygroundTransformEnabled() ||
        m_options.GetREPLEnabled()) {
      // When calling the playground function we are calling a main
      // function which takes two arguments: argc and argv So we pass
      // two zeroes as arguments.
      args.push_back(0); // argc
      args.push_back(0); // argv
    } else {
      args.push_back(struct_address);
      args.push_back(object_ptr);
    }
  } else {
    args.push_back(struct_address);
  }
  return true;
}

lldb::ExpressionVariableSP SwiftUserExpression::GetResultAfterDematerialization(
    ExecutionContextScope *exe_scope) {
  lldb::ExpressionVariableSP in_result_sp = m_result_delegate.GetVariable();
  lldb::ExpressionVariableSP in_error_sp = m_error_delegate.GetVariable();

  lldb::ExpressionVariableSP result_sp;

  if (in_error_sp) {
    bool error_is_valid = false;

    if (llvm::isa<SwiftASTContext>(
            in_error_sp->GetCompilerType().GetTypeSystem())) {
      lldb::ValueObjectSP val_sp = in_error_sp->GetValueObject();
      if (val_sp) {
        if (exe_scope) {
          lldb::ProcessSP process_sp = exe_scope->CalculateProcess();
          if (process_sp) {
            SwiftLanguageRuntime *swift_runtime =
                process_sp->GetSwiftLanguageRuntime();
            if (swift_runtime)
              error_is_valid = swift_runtime->IsValidErrorValue(*val_sp.get());
          }
        }
      }
    }

    lldb::TargetSP target_sp = exe_scope->CalculateTarget();

    if (target_sp) {
      if (auto *persistent_state =
              target_sp->GetSwiftPersistentExpressionState(*exe_scope)) {
        if (error_is_valid) {
          persistent_state->RemovePersistentVariable(in_result_sp);
          result_sp = in_error_sp;
        } else {
          persistent_state->RemovePersistentVariable(in_error_sp);
          result_sp = in_result_sp;
        }
      }
    }
  } else
    result_sp = in_result_sp;

  return result_sp;
}

SwiftUserExpression::ResultDelegate::ResultDelegate(
    lldb::TargetSP target, SwiftUserExpression &, bool is_error)
    : m_target_sp(target), m_is_error(is_error) {}

ConstString SwiftUserExpression::ResultDelegate::GetName() {
  auto prefix = m_persistent_state->GetPersistentVariablePrefix(m_is_error);
  return m_persistent_state->GetNextPersistentVariableName(*m_target_sp,
                                                           prefix);
}

void SwiftUserExpression::ResultDelegate::DidDematerialize(
    lldb::ExpressionVariableSP &variable) {
  m_variable = variable;
}

void SwiftUserExpression::ResultDelegate::RegisterPersistentState(
    PersistentExpressionState *persistent_state) {
  m_persistent_state = persistent_state;
}

lldb::ExpressionVariableSP &SwiftUserExpression::ResultDelegate::GetVariable() {
  return m_variable;
}

SwiftUserExpression::PersistentVariableDelegate::PersistentVariableDelegate(
    SwiftUserExpression &) {}

ConstString SwiftUserExpression::PersistentVariableDelegate::GetName() {
  return ConstString();
}

void SwiftUserExpression::PersistentVariableDelegate::DidDematerialize(
    lldb::ExpressionVariableSP &variable) {
  if (SwiftExpressionVariable *swift_var =
          llvm::dyn_cast<SwiftExpressionVariable>(variable.get())) {
    swift_var->m_swift_flags &= ~SwiftExpressionVariable::EVSNeedsInit;
  }
}
