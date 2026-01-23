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

#include "SwiftASTManipulator.h"
#include "SwiftExpressionParser.h"
#include "SwiftExpressionSourceCode.h"
#include "SwiftPersistentExpressionState.h"
#include "SwiftREPLMaterializer.h"

#include "Plugins/LanguageRuntime/Swift/SwiftLanguageRuntime.h"
#include "lldb/Core/Debugger.h"
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
#include "lldb/Target/Language.h"
#include "lldb/Utility/LLDBAssert.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/Timer.h"
#include "lldb/ValueObject/ValueObject.h"
#include "lldb/lldb-enumerations.h"

#include "swift/AST/ASTContext.h"
#include "swift/AST/GenericEnvironment.h"
#include "swift/AST/Type.h"
#include "swift/AST/Types.h"
#include "swift/Demangling/Demangler.h"
#include "llvm/BinaryFormat/Dwarf.h"

#include <map>
#include <string>
#if HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#include "SwiftUserExpression.h"

using namespace lldb_private;

char SwiftUserExpression::ID;

SwiftUserExpression::SwiftUserExpression(
    ExecutionContextScope &exe_scope, llvm::StringRef expr,
    llvm::StringRef prefix, SourceLanguage language, ResultType desired_type,
    const EvaluateExpressionOptions &options)
    : LLVMUserExpression(exe_scope, expr, prefix, language, desired_type,
                         options),
      m_type_system_helper(*m_target_wp.lock().get()),
      m_result_delegate(exe_scope.CalculateTarget(), *this, false),
      m_error_delegate(exe_scope.CalculateTarget(), *this, true),
      m_persistent_variable_delegate(*this) {
  if (auto target = exe_scope.CalculateTarget())
    m_debugger_id = target->GetDebugger().GetID();
  m_runs_in_playground_or_repl =
      options.GetREPLEnabled() || options.GetPlaygroundTransformEnabled();
}

SwiftUserExpression::~SwiftUserExpression() {}

void SwiftUserExpression::WillStartExecuting() {
  if (auto process = m_jit_process_wp.lock())
    if (auto *swift_runtime = SwiftLanguageRuntime::Get(process))
      swift_runtime->WillStartExecutingUserExpression(
          m_runs_in_playground_or_repl);
    else
      Debugger::ReportError(
          "Can't execute a swift expression without a runtime",
          m_debugger_id);
  else
    Debugger::ReportError("Can't execute an expression without a process",
                          m_debugger_id);
}

void SwiftUserExpression::DidFinishExecuting() {
  auto process = m_jit_process_wp.lock();
  if (!process) {
    Debugger::ReportError("Could not finish a expression without a process",
                          m_debugger_id);
    return;
  }
  if (!process->IsValid()) {
    // This will cause SwiftLanguageRuntime::Get(process) tp fail.
    Debugger::ReportError("Could not finish swift expression because the "
                          "process is being torn down",
                          m_debugger_id);
    return;
  }
  auto *swift_runtime = SwiftLanguageRuntime::Get(process);
  if (!swift_runtime) {
    Debugger::ReportError("Could not finish swift expression without a runtime",
                          m_debugger_id);
    return;
  }
  swift_runtime->DidFinishExecutingUserExpression(m_runs_in_playground_or_repl);
}

/// Determine whether we have a Swift language symbol context. This handles
/// some special cases, such as when the expression language is unknown, or
/// when we have to guess from a mangled name.
static bool isSwiftLanguageSymbolContext(const SwiftUserExpression &expr,
                                         const SymbolContext &sym_ctx) {
  if (sym_ctx.comp_unit &&
      (!expr.Language() ||
       expr.Language().name == llvm::dwarf::DW_LNAME_Swift)) {
    if (sym_ctx.comp_unit->GetLanguage() == lldb::eLanguageTypeSwift)
      return true;
  } else if (sym_ctx.symbol && !expr.Language()) {
    if (sym_ctx.symbol->GetMangled().GuessLanguage() ==
        lldb::eLanguageTypeSwift)
      return true;
  }
  return false;
}

/// Information about `self` in a frame.
struct SwiftSelfInfo {
  /// Whether `self` is a metatype (i.e. whether we're in a static method).
  bool is_metatype = false;

  /// Adjusted type of `self`. If we're in a static method, this is an instance
  /// type.
  CompilerType type;

  /// Type flags for the adjusted type of `self`.
  Flags type_flags = {};
};

/// Find information about `self` in the frame.
static std::optional<SwiftSelfInfo>
findSwiftSelf(StackFrame &frame, lldb::VariableSP self_var_sp) {
  SwiftSelfInfo info;

  lldb::ValueObjectSP valobj_sp = frame.GetValueObjectForFrameVariable(
      self_var_sp, lldb::eDynamicDontRunTarget);

  // 1) Try finding the type of `self` from its ValueObject.
  if (valobj_sp && valobj_sp->GetError().Success())
    info.type = valobj_sp->GetCompilerType();

  // 2) If (1) fails, try finding the type of `self` from its Variable.
  if (!info.type.IsValid())
    if (Type *self_lldb_type = self_var_sp->GetType())
      info.type = self_lldb_type->GetForwardCompilerType();

  // 3) If (1) and (2) fail, give up.
  if (!info.type.IsValid())
    return {};

  // 4) If `self` is a metatype, get its instance type.
  if (Flags(info.type.GetTypeInfo())
          .AllSet(lldb::eTypeIsSwift | lldb::eTypeIsMetatype)) {
    info.type = TypeSystemSwift::GetInstanceType(info.type, &frame);
    info.is_metatype = true;
  }

  auto ts = info.type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwift>();
  if (!ts)
    return {};
  info.type = ts->GetStaticSelfType(info.type.GetOpaqueQualType());

  info.type_flags = Flags(info.type.GetTypeInfo());

  if (!info.type.IsValid())
    return {};
  return info;
}

void SwiftUserExpression::ScanContext(ExecutionContext &exe_ctx, Status &err) {
  Log *log = GetLog(LLDBLog::Expressions);
  LLDB_LOG(log, "SwiftUserExpression::ScanContext()");

  m_target = exe_ctx.GetTargetPtr();
  if (!m_target) {
    LLDB_LOG(log, "  [SUE::SC] Null target");
    return;
  }

  StackFrame *frame = exe_ctx.GetFramePtr();
  if (!frame) {
    LLDB_LOG(log, "  [SUE::SC] Null stack frame");
    return;
  }

  SymbolContext sym_ctx = frame->GetSymbolContext(
      lldb::eSymbolContextFunction | lldb::eSymbolContextBlock |
      lldb::eSymbolContextCompUnit | lldb::eSymbolContextSymbol);
  bool frame_is_swift = isSwiftLanguageSymbolContext(*this, sym_ctx);
  if (!frame_is_swift) {
    LLDB_LOG(log, "  [SUE::SC] Frame is not swift-y");
    return;
  }

  if (!m_swift_ast_ctx) {
    LLDB_LOG(log, "  [SUE::SC] NULL Swift AST Context");
    return;
  }
  if (!m_swift_ast_ctx->GetClangImporter()) {
    LLDB_LOG(log, "  [SUE::SC] Swift AST Context has no Clang importer");
    return;
  }

  if (m_swift_ast_ctx->HasFatalErrors()) {
    LLDB_LOG(log, "  [SUE::SC] Swift AST Context has fatal errors");
    return;
  }

  LLDB_LOG(log, "  [SUE::SC] Compilation unit is swift");

  Block *innermost_block = sym_ctx.block;
  if (!innermost_block) {
    LLDB_LOG(log, "  [SUE::SC] No block");
    return;
  }

  VariableList variable_list;
  innermost_block->AppendVariables(/*can_create*/
                                   true,
                                   /*get_parent_variables*/ true,
                                   /*stop_if_block_is_inlined_function*/ false,
                                   /*filter*/
                                   [&](Variable *var) {
                                     if (!variable_list.Empty())
                                       return false;
                                     return SwiftLanguageRuntime::IsSelf(*var);
                                   },
                                   &variable_list);

  if (variable_list.Empty()) {
    LLDB_LOG(log, "  [SUE::SC] No self variable");
    return;
  }

  assert(variable_list.GetSize() == 1 && variable_list.GetVariableAtIndex(0) &&
         "Unexpected variable list state");

  lldb::VariableSP self_var_sp = variable_list.GetVariableAtIndex(0);
  // If we have a self variable, but it has no location at the current PC, then
  // we can't use it.  Set the self var back to empty and we'll just pretend we
  // are in a regular frame, which is really the best we can do.
  if (!self_var_sp->LocationIsValidForFrame(frame)) {
    LLDB_LOG(log, "  [SUE::SC] `self` variable location not valid for frame");
    return;
  }

  auto maybe_self_info = findSwiftSelf(*frame, self_var_sp);
  if (!maybe_self_info) {
    LLDB_LOG(log, "  [SUE::SC] Could not determine info about `self`");
    return;
  }

  // Check to see if we are in a class func of a class (or static func of a
  // struct) and adjust our type to point to the instance type.
  SwiftSelfInfo info = *maybe_self_info;

  m_in_static_method = info.is_metatype;

  if (info.type_flags.AllSet(lldb::eTypeIsSwift | lldb::eTypeInstanceIsPointer))
    m_is_class |= info.type_flags.Test(lldb::eTypeIsClass);

  // Handle weak self.

  auto ts = info.type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwift>();
  if (!ts)
    return;

  if (auto ownership_kind = ts->GetNonTriviallyManagedReferenceKind(
          info.type.GetOpaqueQualType()))
    if (*ownership_kind ==
        SwiftASTContext::NonTriviallyManagedReferenceKind::eWeak) {
      m_is_class = true;
      m_is_weak_self = true;
    }

  m_needs_object_ptr = !m_in_static_method;

  LLDB_LOGF(log, "  [SUE::SC] Containing class name: %s",
            info.type.GetTypeName().AsCString());
}

/// Create a \c VariableInfo record for \c variable if there isn't
/// already shadowing inner declaration in \c processed_variables.
static llvm::Error AddVariableInfo(
    lldb::VariableSP variable_sp, lldb::StackFrameSP &stack_frame_sp,
    SwiftASTContextForExpressions &ast_context, SwiftLanguageRuntime *runtime,
    llvm::SmallDenseSet<const char *, 8> &processed_variables,
    llvm::SmallVectorImpl<SwiftASTManipulator::VariableInfo> &local_variables,
    lldb::DynamicValueType use_dynamic,
    lldb::BindGenericTypes bind_generic_types) {
  StringRef name = variable_sp->GetUnqualifiedName().GetStringRef();
  const char *name_cstr = name.data();
  assert(StringRef(name_cstr) == name && "missing null terminator");
  if (name.empty())
    return llvm::Error::success();

  // To support "guard let self = self" the function argument "self"
  // is processed (as the special self argument) even if it is
  // shadowed by a local variable.
  bool is_self = SwiftLanguageRuntime::IsSelf(*variable_sp);
  const char *overridden_name = name_cstr;
  if (is_self)
    overridden_name = "$__lldb_injected_self";

  if (processed_variables.count(overridden_name))
    return llvm::Error::success();

  if (!stack_frame_sp)
    return llvm::Error::success();

  if (!variable_sp || !variable_sp->GetType())
    return llvm::Error::success();

  CompilerType target_type;
  bool should_not_bind_generic_types =
      !SwiftASTManipulator::ShouldBindGenericTypes(bind_generic_types);
  bool is_unbound_pack =
      should_not_bind_generic_types &&
      (variable_sp->GetType()->GetForwardCompilerType().GetTypeInfo() &
       lldb::eTypeIsPack);
  bool is_meaningless_without_dynamic_resolution = false;
  // If we're not binding the generic types, we need to set the self type as an
  // opaque pointer type. This is necessary because we don't bind the generic
  // parameters, and we can't have a type with unbound generics in a non-generic
  // function.
  if (should_not_bind_generic_types && is_self)
    target_type = ast_context.GetBuiltinRawPointerType();
  else if (is_unbound_pack)
    target_type = variable_sp->GetType()->GetForwardCompilerType();
  else {
    CompilerType var_type = SwiftExpressionParser::ResolveVariable(
        variable_sp, stack_frame_sp, runtime, use_dynamic, bind_generic_types);

    is_meaningless_without_dynamic_resolution =
        var_type.IsMeaninglessWithoutDynamicResolution();

    // ImportType would only return a TypeRef type anyway, so it's
    // faster to leave the type in its original context for faster
    // type alias resolution.
    target_type = var_type;
  }

  // If the import failed, give up.
  if (!target_type.IsValid()) {
    // Treat an invalid type for self as a fatal error.
    if (is_self)
      return llvm::createStringError("type for self is invalid");
    return llvm::Error::success();
  }

  // Report a fatal error if self can't be reconstructed as a Swift AST type.
  if (is_self) {
    auto self_ty = ast_context.GetSwiftType(target_type);
    if (!self_ty)
      return llvm::createStringError("type for self cannot be reconstructed: " +
                                     llvm::toString(self_ty.takeError()));
  }

  auto ts = target_type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwift>();
  if (!ts)
    return llvm::createStringError("type for self has no type system");

  // If we couldn't fully realize the type, then we aren't going
  // to get very far making a local out of it, so discard it here.
  Log *log = GetLog(LLDBLog::Types | LLDBLog::Expressions);
  if (is_meaningless_without_dynamic_resolution) {
    if (log)
      log->Printf("Discarding local %s because we couldn't fully realize it, "
                  "our best attempt was: %s.",
                  name_cstr,
                  target_type.GetDisplayTypeName().AsCString("<unknown>"));
    // Not realizing self is a fatal error for an expression and the
    // Swift compiler error alone is not particularly useful.
    if (is_self)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "Discarding local %s because we couldn't fully realize it, "
          "our best attempt was: %s.",
          name_cstr, target_type.GetDisplayTypeName().AsCString("<unknown>"));
    return llvm::Error::success();
  }

  if (log && is_self)
    if (swift::Type swift_type =
            llvm::expectedToStdOptional(ast_context.GetSwiftType(target_type))
                .value_or(swift::Type())) {
      std::string s;
      llvm::raw_string_ostream ss(s);
      swift_type->dump(ss);
      ss.flush();
      log->Printf("Adding injected self: type (%p) context(%p) is: %s",
                  static_cast<void *>(swift_type.getPointer()),
                  static_cast<void *>(*ast_context.GetASTContext()), s.c_str());
    }

  // Stash diagnostics for missing locations.
  Status variable_status;
  if (lldb::ValueObjectSP valobj_sp =
          stack_frame_sp->GetValueObjectForFrameVariable(
              variable_sp, lldb::eNoDynamicValues)) {
    variable_status = valobj_sp->GetError().Clone();
    if (variable_status.Fail() && variable_sp->IsArtificial())
      return llvm::Error::success();
  }

  // A one-off clone of variable_sp with the type replaced by target_type.
  auto patched_variable_sp = std::make_shared<lldb_private::Variable>(
      0, variable_sp->GetName().GetCString(), "",
      std::make_shared<lldb_private::SymbolFileType>(
          *variable_sp->GetType()->GetSymbolFile(),
          variable_sp->GetType()->GetSymbolFile()->MakeType(
              0, variable_sp->GetType()->GetName(), std::nullopt,
              variable_sp->GetType()->GetSymbolContextScope(), LLDB_INVALID_UID,
              Type::eEncodingIsUID, variable_sp->GetType()->GetDeclaration(),
              target_type, lldb_private::Type::ResolveState::Full,
              variable_sp->GetType()->GetPayload())),
      variable_sp->GetScope(), variable_sp->GetSymbolContextScope(),
      variable_sp->GetScopeRange(),
      const_cast<lldb_private::Declaration *>(&variable_sp->GetDeclaration()),
      variable_sp->LocationExpressionList(), variable_sp->IsExternal(),
      variable_sp->IsArtificial(),
      variable_sp->GetLocationIsConstantValueData(),
      variable_sp->IsStaticMember(), variable_sp->IsConstant());
  SwiftASTManipulatorBase::VariableMetadataSP metadata_sp(
      new SwiftASTManipulatorBase::VariableMetadataVariable(
          patched_variable_sp));
  local_variables.emplace_back(
      target_type, ast_context.GetASTContext()->getIdentifier(overridden_name),
      metadata_sp,
      variable_sp->IsConstant() ? swift::VarDecl::Introducer::Let
                                : swift::VarDecl::Introducer::Var,
      false, is_unbound_pack);
  if (variable_status.Fail())
    local_variables.back().SetLookupError(variable_status.takeError());

  processed_variables.insert(overridden_name);
  return llvm::Error::success();
}

/// Collets all the variables visible in the current scope.
static void CollectVariablesInScope(SymbolContext &sc,
                                    lldb::StackFrameSP &stack_frame_sp,
                                    VariableList &variables) {
  Block *block = sc.block;
  Block *top_block = block ? block->GetContainingInlinedBlock() : nullptr;  

  if (!top_block && sc.function)
    top_block = &sc.function->GetBlock(true);

  // The module scoped variables are stored at the CompUnit level, so
  // after we go through the current context, then we have to take one
  // more pass through the variables in the CompUnit.
  bool done = false;
  while (block) {
    // Iterate over all parent contexts *including* the top_block.
    if (block == top_block)
      done = true;
    bool can_create = true;
    bool get_parent_variables = false;
    bool stop_if_block_is_inlined_function = true;

    block->AppendVariables(
        can_create, get_parent_variables, stop_if_block_is_inlined_function,
        [](Variable *) { return true; }, &variables);

    if (done)
      break;
    block = block->GetParent();
  }

  // Also add local copies of globals. This is in many cases redundant
  // work because the globals would also be found in the expression
  // context's Swift module, but it allows a limited form of
  // expression evaluation to work even if the Swift module failed to
  // load, as long as the module isn't necessary to resolve the type
  // or aother symbols in the expression.
  if (sc.comp_unit) {
    lldb::VariableListSP globals_sp = sc.comp_unit->GetVariableList(true);
    if (globals_sp)
      variables.AddVariables(globals_sp.get());
  }
}

/// Create a \c VariableInfo record for each visible variable.
static llvm::Error RegisterAllVariables(
    SymbolContext &sc, lldb::StackFrameSP &stack_frame_sp,
    SwiftASTContextForExpressions &ast_context,
    llvm::SmallVectorImpl<SwiftASTManipulator::VariableInfo> &local_variables,
    lldb::DynamicValueType use_dynamic,
    lldb::BindGenericTypes bind_generic_types) {
  SwiftLanguageRuntime *language_runtime = nullptr;

  if (stack_frame_sp)
    language_runtime =
        SwiftLanguageRuntime::Get(stack_frame_sp->GetThread()->GetProcess());

  VariableList variables;
  CollectVariablesInScope(sc, stack_frame_sp, variables);

  // Proceed from the innermost scope outwards, adding all variables
  // not already shadowed by an inner declaration.
  llvm::SmallDenseSet<const char *, 8> processed_names;
  for (size_t vi = 0, ve = variables.GetSize(); vi != ve; ++vi)
    if (auto error =
            AddVariableInfo({variables.GetVariableAtIndex(vi)}, stack_frame_sp,
                            ast_context, language_runtime, processed_names,
                            local_variables, use_dynamic, bind_generic_types))
      return error;
  return llvm::Error::success();
}

/// Check if we can evaluate the expression as generic.
/// Currently, evaluating expression as a generic has several limitations:
/// - Only self will be evaluated with unbound generics.
/// - The Self type can only have one generic parameter. 
/// - The Self type has to be the outermost type with unbound generics.
static bool CanEvaluateExpressionWithoutBindingGenericParams(
    const llvm::SmallVectorImpl<SwiftASTManipulator::VariableInfo> &variables,
    const std::optional<SwiftLanguageRuntime::GenericSignature> &generic_sig,
    SwiftASTContextForExpressions &scratch_ctx, Block *block,
    StackFrame &stack_frame) {
  // First, find the compiler type of self with the generic parameters not
  // bound.
  auto self_var = SwiftExpressionParser::FindSelfVariable(block);
  if (!self_var) {
    // Freestanding variadic generic functions are also supported.
    if (generic_sig)
      return generic_sig->pack_expansions.size();

    return false;
  }

  lldb::ValueObjectSP self_valobj = stack_frame.GetValueObjectForFrameVariable(
      self_var, lldb::eNoDynamicValues);
  if (!self_valobj)
    return false;

  auto self_type = self_valobj->GetCompilerType();
  if (!self_type)
    return false;

  auto ts = self_type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwift>();
  if (!ts)
    return false;

  auto swift_type =
      llvm::expectedToStdOptional(scratch_ctx.GetSwiftType(self_type))
          .value_or(swift::Type());
  if (!swift_type)
    return false;

  auto *decl = swift_type->getAnyGeneric();
  if (!decl)
    return false;

  auto *env = decl->getGenericEnvironment();
  if (!env)
    return false;
  auto generic_params = env->getGenericParams();

  // If there aren't any generic parameters we can't evaluate the expression as
  // generic.
  if (generic_params.empty())
    return false;

  auto *first_param = generic_params[0];
  // Currently we only support evaluating self as generic if the generic
  // parameter is the first one.
  if (first_param->getDepth() != 0 || first_param->getIndex() != 0)
    return false;

  llvm::SmallVector<const SwiftASTManipulator::VariableInfo *>
      outermost_metadata_vars;
  for (auto &variable : variables)
    if (variable.IsOutermostMetadataPointer())
      outermost_metadata_vars.push_back(&variable);

  // Check that all metadata belong to the outermost type, and check that we do
  // have the metadata pointer available.
  for (auto *generic_param : generic_params) {
    if (generic_param->getDepth() != 0)
      return false;

    std::string var_name;
    llvm::raw_string_ostream s(var_name);
    s << "$τ_0_" << generic_param->getIndex();
    auto found = false;
    for (auto *metadata_var : outermost_metadata_vars) {
      if (metadata_var->GetName().str() == var_name) {
        found = true;
        break;
      }
    }
    if (!found)
      return false;
  }

  return true;
}

SwiftExpressionParser::ParseResult
SwiftUserExpression::GetTextAndSetExpressionParser(
    DiagnosticManager &diagnostic_manager,
    std::unique_ptr<SwiftExpressionSourceCode> &source_code,
    ExecutionContext &exe_ctx, ExecutionContextScope *exe_scope) {
  using ParseResult = SwiftExpressionParser::ParseResult;
  Log *log = GetLog(LLDBLog::Expressions);

  lldb::TargetSP target_sp;
  SymbolContext sc;
  lldb::StackFrameSP stack_frame;
  if (exe_scope) {
    target_sp = exe_scope->CalculateTarget();
    stack_frame = exe_scope->CalculateStackFrame();

    if (stack_frame) {
      sc = stack_frame->GetSymbolContext(lldb::eSymbolContextEverything);
    }
  }

  llvm::SmallVector<SwiftASTManipulator::VariableInfo> local_variables;

  if (llvm::Error error = RegisterAllVariables(
          sc, stack_frame, *m_swift_ast_ctx, local_variables,
          m_options.GetUseDynamic(), m_options.GetBindGenericTypes())) {
    diagnostic_manager.PutString(lldb::eSeverityInfo,
                                 llvm::toString(std::move(error)));
    diagnostic_manager.PutString(
        lldb::eSeverityError,
        "Couldn't realize Swift AST type of self. Hint: using `v` to "
        "directly inspect variables and fields may still work.");
    return ParseResult::retry_bind_generic_params;
  }

  auto ts = m_swift_ast_ctx->GetTypeSystemSwiftTypeRef();
  if (ts && stack_frame) {
    // Extract the generic signature of the context.
    ConstString func_name =
        stack_frame->GetSymbolContext(lldb::eSymbolContextFunction)
            .GetFunctionName(Mangled::ePreferMangled);
    m_generic_signature = SwiftLanguageRuntime::GetGenericSignature(
        func_name.GetStringRef(), *ts);
  }

  if (!SwiftASTManipulator::ShouldBindGenericTypes(
          m_options.GetBindGenericTypes()) &&
      !CanEvaluateExpressionWithoutBindingGenericParams(
          local_variables, m_generic_signature, *m_swift_ast_ctx, sc.block,
          *stack_frame.get())) {
    diagnostic_manager.PutString(
        lldb::eSeverityError,
        "Could not evaluate the expression without binding generic types.");
    return ParseResult::retry_bind_generic_params_not_supported;
  }

  uint32_t first_body_line = 0;
  Status status = source_code->GetText(
      m_transformed_text, m_options.GetLanguage(), m_needs_object_ptr,
      m_in_static_method, m_is_class, m_is_weak_self, m_options,
      m_generic_signature, exe_ctx, first_body_line, local_variables);
  if (status.Fail()) {
    diagnostic_manager.PutString(
        lldb::eSeverityError,
        "couldn't construct expression body: " +
            std::string(status.AsCString("<unknown error>")));
    return ParseResult::unrecoverable_error;
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

  auto *swift_parser =
      new SwiftExpressionParser(exe_scope, *m_swift_ast_ctx, *this,
                                std::move(local_variables), m_options);
  SwiftExpressionParser::ParseResult parse_result =
      swift_parser->Parse(diagnostic_manager, first_body_line,
                          first_body_line + source_code->GetNumBodyLines());
  m_parser.reset(swift_parser);

  return parse_result;
}

/// If `sc` represents a "closure-like" function according to `lang`, and
/// `var_name` can be found in a parent context, create a diagnostic
/// explaining that this variable is available but not captured by the closure.
static std::string
CreateVarInParentScopeDiagnostic(StringRef var_name,
                                 StringRef parent_func_name) {
  return llvm::formatv("A variable named '{0}' existed in function '{1}', but "
                       "it was not captured in the closure.\nHint: the "
                       "variable may be available in a parent frame.",
                       var_name, parent_func_name);
}

/// If `diagnostic_manager` contains a "cannot find <var_name> in scope"
/// diagnostic, attempt to enhance it by showing if `var_name` is used inside a
/// closure, not captured, but defined in a parent scope.
static void EnhanceNotInScopeDiagnostics(DiagnosticManager &diagnostic_manager,
                                         ExecutionContextScope *exe_scope) {
  if (!exe_scope)
    return;
  lldb::StackFrameSP stack_frame = exe_scope->CalculateStackFrame();
  if (!stack_frame)
    return;
  SymbolContext sc =
      stack_frame->GetSymbolContext(lldb::eSymbolContextEverything);
  Language *swift_lang =
      Language::FindPlugin(lldb::LanguageType::eLanguageTypeSwift);
  if (!swift_lang)
    return;

  static const RegularExpression not_in_scope_regex =
      RegularExpression("(.*): cannot find '([^']+)' in scope\n(.*)");
  for (auto &diag : diagnostic_manager.Diagnostics()) {
    if (!diag)
      continue;

    llvm::SmallVector<StringRef, 4> match_groups;

    if (StringRef old_rendered_msg = diag->GetDetail().rendered;
        !not_in_scope_regex.Execute(old_rendered_msg, &match_groups))
      continue;

    StringRef prefix = match_groups[1];
    StringRef var_name = match_groups[2];
    StringRef suffix = match_groups[3];

    Function *parent_func =
        swift_lang->FindParentOfClosureWithVariable(var_name, sc);
    if (!parent_func)
      continue;
    std::string new_message = CreateVarInParentScopeDiagnostic(
        var_name, parent_func->GetDisplayName());

    std::string new_rendered =
        llvm::formatv("{0}: {1}\n{2}", prefix, new_message, suffix);
    const DiagnosticDetail &old_detail = diag->GetDetail();
    diag = std::make_unique<Diagnostic>(
        diag->getKind(), diag->GetCompilerID(),
        DiagnosticDetail{old_detail.source_location, old_detail.severity,
                         std::move(new_message), std::move(new_rendered)});
  }
}

bool SwiftUserExpression::Parse(DiagnosticManager &diagnostic_manager,
                                ExecutionContext &exe_ctx,
                                lldb_private::ExecutionPolicy execution_policy,
                                bool keep_result_in_memory,
                                bool generate_debug_info) {
  Log *log = GetLog(LLDBLog::Expressions);
  Status err;

  auto diagnose = [&](const char *error_msg, const char *detail,
                      lldb::Severity severity) -> bool {
    if (detail)
      LLDB_LOG(log, "{0}: {1}", error_msg, detail);
    else
      LLDB_LOG(log, error_msg);

    diagnostic_manager.PutString(severity, error_msg);
    if (detail)
      diagnostic_manager.AppendMessageToDiagnostic(detail);
    return false;
  };
  auto error = [&](const char *error_msg, const char *detail = nullptr) {
    return diagnose(error_msg, detail, lldb::eSeverityError);
  };
  auto note = [&](const char *error_msg, const char *detail = nullptr) {
    return diagnose(error_msg, detail, lldb::eSeverityInfo);
  };

  InstallContext(exe_ctx);
  Target *target = exe_ctx.GetTargetPtr();
  if (!target)
    return error("couldn't start parsing (no target)");

  StackFrame *frame = exe_ctx.GetFramePtr();
  if (!frame)
    return error("couldn't start parsing - no stack frame");

  ExecutionContextScope *exe_scope =
      m_options.GetREPLEnabled() || m_options.GetPlaygroundTransformEnabled()
          ? static_cast<ExecutionContextScope *>(target)
          : static_cast<ExecutionContextScope *>(frame);

  exe_scope = exe_ctx.GetBestExecutionContextScope();

  auto ts_or_err = target->GetScratchTypeSystemForLanguage(
      lldb::eLanguageTypeSwift, /*create_on_demand=*/true);
  if (!ts_or_err)
    return error("could not create a Swift scratch typesystem",
                 llvm::toString(ts_or_err.takeError()).c_str());
  m_swift_scratch_ctx =
      std::static_pointer_cast<TypeSystemSwiftTypeRefForExpressions>(
          *ts_or_err);
  if (!m_swift_scratch_ctx)
    return error("could not create a Swift scratch typesystem",
                 "unknown error");
  // Notify SwiftASTContext that this is a Playground.
  if (m_options.GetPlaygroundTransformEnabled())
    m_swift_scratch_ctx->SetCompilerOptions(
        m_options.GetREPLEnabled(), m_options.GetPlaygroundTransformEnabled(),
        "");

  // For playgrounds, the target triple should be used for expression
  // evaluation, not the current module. This requires disabling precise
  // compiler invocations.
  SymbolContext sc;
  if (m_options.GetREPLEnabled() || m_options.GetPlaygroundTransformEnabled())
    sc = SymbolContext(target->shared_from_this(),
                       target->GetExecutableModule());
  else
    sc = frame->GetSymbolContext(lldb::eSymbolContextFunction);
  auto swift_ast_ctx = m_swift_scratch_ctx->GetSwiftASTContext(sc);
  if (llvm::dyn_cast_or_null<SwiftASTContextForExpressions>(
          swift_ast_ctx.get()))
    m_swift_ast_ctx =
        std::static_pointer_cast<SwiftASTContextForExpressions>(swift_ast_ctx);

  if (!m_swift_ast_ctx)
    return error("could not initialize Swift compiler",
                 "run \"swift-healthcheck\" for more details");

  if (m_swift_ast_ctx->HasFatalErrors()) {
    m_swift_ast_ctx->PrintDiagnostics(diagnostic_manager);
    return note("Swift AST context is in a fatal error state, run "
                "\"swift-healthcheck\" for more details");
  }
  
  // This may destroy the scratch context.
  auto *persistent_state =
      target->GetPersistentExpressionStateForLanguage(lldb::eLanguageTypeSwift);
  if (!persistent_state)
    return error("could not start parsing (no persistent data)");

  auto module_decl_or_err = m_swift_ast_ctx->ImportStdlib();
  if (!module_decl_or_err)
    return error("could not load Swift Standard Library",
                 llvm::toString(module_decl_or_err.takeError()).c_str());

  m_swift_ast_ctx->AddHandLoadedModule(
      ConstString("Swift"), swift::ImportedModule(&*module_decl_or_err));
  m_result_delegate.RegisterPersistentState(persistent_state);
  m_error_delegate.RegisterPersistentState(persistent_state);
 
  ScanContext(exe_ctx, err);

  if (!err.Success())
    diagnostic_manager.Printf(lldb::eSeverityWarning, "warning: %s\n",
                              err.AsCString());

  StreamString m_transformed_stream;

  //
  // Generate the expression.
  //

  std::string prefix = m_expr_prefix;

  std::unique_ptr<SwiftExpressionSourceCode> source_code(
      SwiftExpressionSourceCode::CreateWrapped(prefix.c_str(), 
                                               m_expr_text.c_str()));

  const lldb::LanguageType lang_type = lldb::eLanguageTypeSwift;

  m_options.SetLanguage(lang_type);
  m_options.SetGenerateDebugInfo(generate_debug_info);

  using ParseResult = SwiftExpressionParser::ParseResult;
  // Use a separate diagnostic manager instead of the main one, the reason we do
  // this is that on retries we would like to ignore diagnostics produced by
  // either the first or second try.
  DiagnosticManager first_try_diagnostic_manager;
  DiagnosticManager second_try_diagnostic_manager;

  bool retry = false;
  bool first_expression_not_supported = false;
  while (true) {
    SwiftExpressionParser::ParseResult parse_result;
    if (!retry) {
      parse_result = GetTextAndSetExpressionParser(
          first_try_diagnostic_manager, source_code, exe_ctx, exe_scope);
      if ((parse_result !=
               SwiftExpressionParser::ParseResult::retry_bind_generic_params &&
           parse_result != SwiftExpressionParser::ParseResult::
                               retry_bind_generic_params_not_supported) ||
          m_options.GetBindGenericTypes() != lldb::eBindAuto)
        // If we're not retrying, just copy the diagnostics over.
        diagnostic_manager.Consume(std::move(first_try_diagnostic_manager));
    } else {
      parse_result = GetTextAndSetExpressionParser(
          second_try_diagnostic_manager, source_code, exe_ctx, exe_scope);

        diagnostic_manager.Consume(std::move(second_try_diagnostic_manager));
        if (parse_result == SwiftExpressionParser::ParseResult::success ||
            first_expression_not_supported)
          // If we succeeded the second time around, copy any diagnostics we
          // produced in the success case over, and ignore the first attempt's
          // failures.
          // If the generic expression evaluator failed because it doesn't
          // support the expression the error message won't be very useful, so
          // print the regular expression evaluator's error instead.
          diagnostic_manager.Consume(std::move(second_try_diagnostic_manager));
        else
          // If we failed though, copy the diagnostics of the first attempt, and
          // silently ignore any errors produced by the retry, as the retry was
          // not what the user asked, and any diagnostics produced by it will
          // most likely confuse the user.
          diagnostic_manager.Consume(std::move(first_try_diagnostic_manager));
    }

    if (parse_result == SwiftExpressionParser::ParseResult::success)
      break;

    switch (parse_result) {
    case ParseResult::retry_bind_generic_params_not_supported:
      first_expression_not_supported = true;
      LLVM_FALLTHROUGH;
    case ParseResult::retry_bind_generic_params:
      // Retry running the expression without binding the generic types if
      // BindGenericTypes was in the auto setting, give up otherwise.
      if (m_options.GetBindGenericTypes() != lldb::eBindAuto) 
        return false;
      // Retry binding generic parameters, this is the only
      // case that will loop.
      m_options.SetBindGenericTypes(lldb::eBind);
      retry = true;
      break;
    
    case ParseResult::retry_fresh_context:
      m_fixed_text = m_expr_text;
      return false;
    case ParseResult::unrecoverable_error:
      // If fixits are enabled, calculate the fixed expression string.
      if (m_options.GetAutoApplyFixIts() && diagnostic_manager.HasFixIts()) {
        if (m_parser->RewriteExpression(diagnostic_manager)) {
          uint32_t fixed_start;
          uint32_t fixed_end;
          const std::string &fixed_expression =
              diagnostic_manager.GetFixedExpression();
          if (SwiftExpressionSourceCode::GetOriginalBodyBounds(
                  fixed_expression, fixed_start, fixed_end))
            m_fixed_text =
                fixed_expression.substr(fixed_start, fixed_end - fixed_start);
        }
      }
      EnhanceNotInScopeDiagnostics(diagnostic_manager, exe_scope);
      return false;
    case ParseResult::success:
      llvm_unreachable("Success case is checked separately before switch!");
    }
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
      if (auto *target = exe_ctx.GetTargetPtr())
        if (auto *state = target->GetPersistentExpressionStateForLanguage(
                lldb::eLanguageTypeSwift))
          state->RegisterExecutionUnit(m_execution_unit_sp);
    }
  }

  StreamString jit_module_name;
  jit_module_name.Printf("%s%u", FunctionName(),
                         m_options.GetExpressionNumber());
  auto module =
      m_execution_unit_sp->CreateJITModule(jit_module_name.GetString().data());

  Process *process = exe_ctx.GetProcessPtr();
  auto *swift_runtime = SwiftLanguageRuntime::Get(process);
  if (module && swift_runtime) {
    ModuleList modules;
    modules.Append(module, false);
    swift_runtime->ModulesDidLoad(modules);
  }

  if (jit_error.Success()) {
    if (process && m_jit_start_addr != LLDB_INVALID_ADDRESS)
      m_jit_process_wp = lldb::ProcessWP(process->shared_from_this());
    return true;
  }

  const char *error_cstr = jit_error.AsCString();
  if (!error_cstr || !error_cstr[0])
    error_cstr = "expression can't be interpreted or run";
  return error(error_cstr);
}

bool SwiftUserExpression::AddArguments(ExecutionContext &exe_ctx,
                                       std::vector<lldb::addr_t> &args,
                                       lldb::addr_t struct_address,
                                       DiagnosticManager &diagnostic_manager) {
  if (m_options.GetPlaygroundTransformEnabled() || m_options.GetREPLEnabled()) {
    // When calling the playground function we are calling a main
    // function which takes two arguments: argc and argv So we pass
    // two zeroes as arguments.
    args.push_back(0); // argc
    args.push_back(0); // argv
    return true;
  }
  args.push_back(struct_address);
  return true;
}

lldb::ExpressionVariableSP SwiftUserExpression::GetResultAfterDematerialization(
    ExecutionContextScope *exe_scope) {
  lldb::ExpressionVariableSP in_result_sp = m_result_delegate.GetVariable();
  lldb::ExpressionVariableSP in_error_sp = m_error_delegate.GetVariable();

  lldb::ExpressionVariableSP result_sp;

  if (in_error_sp) {
    bool error_is_valid = false;

    if (in_error_sp->GetCompilerType().GetTypeSystem()->SupportsLanguage(
            lldb::eLanguageTypeSwift)) {
      lldb::ValueObjectSP val_sp = in_error_sp->GetValueObject();
      if (val_sp) {
        if (exe_scope) {
          lldb::ProcessSP process_sp = exe_scope->CalculateProcess();
          if (process_sp) {
            auto *swift_runtime = SwiftLanguageRuntime::Get(process_sp);
            if (swift_runtime)
              error_is_valid = swift_runtime->IsValidErrorValue(*val_sp.get());
          }
        }
      }
    }

    lldb::TargetSP target_sp = exe_scope->CalculateTarget();

    if (target_sp) {
      if (auto *persistent_state =
              target_sp->GetPersistentExpressionStateForLanguage(
                  lldb::eLanguageTypeSwift)) {
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
  return m_persistent_state->GetNextPersistentVariableName(m_is_error);
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
