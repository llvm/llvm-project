//===-- SwiftExpressionParser.cpp ---------------------------------------*-===//
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

#include "SwiftExpressionParser.h"

#include "SwiftASTManipulator.h"
#include "SwiftExpressionSourceCode.h"
#include "SwiftREPLMaterializer.h"
#include "SwiftSILManipulator.h"
#include "SwiftUserExpression.h"

#include "Plugins/ExpressionParser/Swift/SwiftDiagnostic.h"
#include "Plugins/ExpressionParser/Swift/SwiftExpressionVariable.h"
#include "Plugins/LanguageRuntime/Swift/SwiftLanguageRuntime.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Expression/DiagnosticManager.h"
#include "lldb/Expression/Expression.h"
#include "lldb/Expression/IRExecutionUnit.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Stream.h"
#include "lldb/Utility/Timer.h"

#include "llvm-c/Analysis.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include "clang/Basic/Module.h"
#include "clang/Rewrite/Core/RewriteBuffer.h"

#include "swift/AST/ASTContext.h"
#include "swift/AST/DiagnosticConsumer.h"
#include "swift/AST/DiagnosticEngine.h"
#include "swift/AST/IRGenOptions.h"
#include "swift/AST/IRGenRequests.h"
#include "swift/AST/Import.h"
#include "swift/AST/ASTMangler.h"
#include "swift/AST/Module.h"
#include "swift/AST/ModuleLoader.h"
#include "swift/AST/GenericParamList.h"
#include "swift/AST/GenericEnvironment.h"
#include "swift/Basic/OptimizationMode.h"
#include "swift/Basic/PrimarySpecificPaths.h"
#include "swift/Basic/SourceManager.h"
#include "swift/ClangImporter/ClangImporter.h"
#include "swift/Demangling/Demangle.h"
#include "swift/Frontend/Frontend.h"
#include "swift/Parse/PersistentParserState.h"
#include "swift/SIL/SILDebuggerClient.h"
#include "swift/SIL/SILFunction.h"
#include "swift/SIL/SILModule.h"
#include "swift/SIL/TypeLowering.h"
#include "swift/SILOptimizer/PassManager/Passes.h"
#include "swift/Serialization/SerializedModuleLoader.h"
#include "swift/Subsystems.h"

using namespace lldb_private;
using llvm::make_error;
using llvm::StringError;
using llvm::StringRef;
using llvm::inconvertibleErrorCode;

SwiftExpressionParser::SwiftExpressionParser(
       ExecutionContextScope *exe_scope,
       SwiftASTContextForExpressions &swift_ast_ctx, Expression &expr,
       llvm::SmallVector<SwiftASTManipulator::VariableInfo> &&local_variables,
       const EvaluateExpressionOptions &options) 
    : ExpressionParser(exe_scope, expr, options.GetGenerateDebugInfo()),
      m_expr(expr), m_swift_ast_ctx(swift_ast_ctx), m_exe_scope(exe_scope),
      m_local_variables(std::move(local_variables)),
      m_options(options) {
  assert(expr.Language() == lldb::eLanguageTypeSwift);

  // TODO: This code is copied from ClangExpressionParser.cpp.
  // Factor this out into common code.

  lldb::TargetSP target_sp;
  if (exe_scope) {
    target_sp = exe_scope->CalculateTarget();

    lldb::StackFrameSP stack_frame = exe_scope->CalculateStackFrame();

    if (stack_frame) {
      m_stack_frame_wp = stack_frame;
      m_sc = stack_frame->GetSymbolContext(lldb::eSymbolContextEverything);
    } else {
      m_sc.target_sp = target_sp;
    }
  }
}

static CompilerType ImportType(SwiftASTContextForExpressions &target_context,
                               CompilerType source_type) {
  Status error, mangled_error;
  return target_context.ImportType(source_type, error);
}

namespace {
class LLDBNameLookup : public swift::SILDebuggerClient {
public:
  LLDBNameLookup(swift::SourceFile &source_file,
                 SwiftExpressionParser::SILVariableMap &variable_map,
                 SymbolContext &sc, ExecutionContextScope &exe_scope)
      : SILDebuggerClient(source_file.getASTContext()),
        m_log(GetLog(LLDBLog::Expressions)), m_source_file(source_file),
        m_variable_map(variable_map), m_sc(sc) {
    source_file.getParentModule()->setDebugClient(this);

    if (!m_sc.target_sp)
      return;
    m_persistent_vars =
        m_sc.target_sp->GetSwiftPersistentExpressionState(exe_scope);
  }

  swift::SILValue emitLValueForVariable(swift::VarDecl *var,
                                        swift::SILBuilder &builder) override {
    SwiftSILManipulator manipulator(builder);

    swift::Identifier variable_name = var->getName();
    ConstString variable_const_string(variable_name.get());

    SwiftExpressionParser::SILVariableMap::iterator vi =
        m_variable_map.find(variable_const_string.AsCString());

    if (vi == m_variable_map.end())
      return swift::SILValue();

    return manipulator.emitLValueForVariable(var, vi->second);
  }

  SwiftPersistentExpressionState::SwiftDeclMap &GetStagedDecls() {
    return m_staged_decls;
  }

protected:
  Log *m_log;
  swift::SourceFile &m_source_file;
  SwiftExpressionParser::SILVariableMap &m_variable_map;
  SymbolContext m_sc;
  SwiftPersistentExpressionState *m_persistent_vars = nullptr;

  // Subclasses stage globalized decls in this map. They get copied
  // over to the SwiftPersistentVariable store if the parse succeeds.
  SwiftPersistentExpressionState::SwiftDeclMap m_staged_decls;
};

/// A name lookup class for debugger expr mode.
class LLDBExprNameLookup : public LLDBNameLookup {
public:
  LLDBExprNameLookup(swift::SourceFile &source_file,
                     SwiftExpressionParser::SILVariableMap &variable_map,
                     SymbolContext &sc, ExecutionContextScope &exe_scope)
      : LLDBNameLookup(source_file, variable_map, sc, exe_scope) {}

  bool shouldGlobalize(swift::Identifier Name, swift::DeclKind Kind) override {
    // Extensions have to be globalized, there's no way to mark them
    // as local to the function, since their name is the name of the
    // thing being extended...
    if (Kind == swift::DeclKind::Extension)
      return true;

    // Operators need to be parsed at the global scope regardless of name.
    if (Kind == swift::DeclKind::Func && Name.isOperator())
      return true;

    const char *name_cstr = Name.get();
    if (name_cstr && name_cstr[0] == '$') {
      if (m_log)
        m_log->Printf("[LLDBExprNameLookup::shouldGlobalize] Returning true to "
                      "globalizing %s",
                      name_cstr);
      return true;
    }
    return false;
  }

  void didGlobalize(swift::Decl *decl) override {
    swift::ValueDecl *value_decl = swift::dyn_cast<swift::ValueDecl>(decl);
    if (value_decl) {
      // It seems weird to be asking this again, but some DeclKinds
      // must be moved to the source-file level to be legal.  But we
      // don't want to register them with lldb unless they are of the
      // kind lldb explicitly wants to globalize.
      if (shouldGlobalize(value_decl->getBaseName().getIdentifier(),
                          value_decl->getKind()))
        m_staged_decls.AddDecl(value_decl, false, ConstString());
    }
  }

  bool lookupOverrides(swift::DeclBaseName Name, swift::DeclContext *DC,
                       swift::SourceLoc Loc, bool IsTypeLookup,
                       ResultVector &RV) override {
    static unsigned counter = 0;
    unsigned count = counter++;

    if (m_log) {
      m_log->Printf("[LLDBExprNameLookup::lookupOverrides(%u)] Searching for %s",
                    count, Name.getIdentifier().get());
    }

    return false;
  }

  bool lookupAdditions(swift::DeclBaseName Name, swift::DeclContext *DC,
                       swift::SourceLoc Loc, bool IsTypeLookup,
                       ResultVector &RV) override {
    LLDB_SCOPED_TIMER();
    static unsigned counter = 0;
    unsigned count = counter++;

    StringRef NameStr = Name.getIdentifier().str();

    if (m_log) {
      m_log->Printf("[LLDBExprNameLookup::lookupAdditions (%u)] Searching for %s",
                    count, Name.getIdentifier().str().str().c_str());
    }

    ConstString name_const_str(NameStr);
    std::vector<swift::ValueDecl *> results;

    // First look up the matching decls we've made in this compile.
    // Later, when we look for persistent decls, these staged decls
    // take precedence.

    m_staged_decls.FindMatchingDecls(name_const_str, {}, results);

    // Next look up persistent decls matching this name.  Then, if we
    // aren't looking at a debugger variable, filter out persistent
    // results of the same kind as one found by the ordinary lookup
    // mechanism in the parser.  The problem we are addressing here is
    // the case where the user has entered the REPL while in an
    // ordinary debugging session to play around.  While there, e.g.,
    // they define a class that happens to have the same name as one
    // in the program, then in some other context "expr" will call the
    // class they've defined, not the one in the program itself would
    // use.  Plain "expr" should behave as much like code in the
    // program would, so we want to favor entities of the same
    // DeclKind & name from the program over ones defined in the REPL.
    // For function decls we check the interface type and full name so
    // we don't remove overloads that don't exist in the current
    // scope.
    //
    // Note also, we only do this for the persistent decls.  Anything
    // in the "staged" list has been defined in this expr setting and
    // so is more local than local.
    if (m_persistent_vars) {
      bool is_debugger_variable = !NameStr.empty() && NameStr.front() == '$';

      size_t num_external_results = RV.size();
      if (!is_debugger_variable && num_external_results > 0) {
        std::vector<swift::ValueDecl *> persistent_results;
        m_persistent_vars->GetSwiftPersistentDecls(name_const_str, {},
                                                   persistent_results);

        size_t num_persistent_results = persistent_results.size();
        for (size_t idx = 0; idx < num_persistent_results; idx++) {
          swift::ValueDecl *value_decl = persistent_results[idx];
          if (!value_decl)
            continue;
          swift::DeclName value_decl_name = value_decl->getName();
          swift::DeclKind value_decl_kind = value_decl->getKind();
          swift::CanType value_interface_type =
              value_decl->getInterfaceType()->getCanonicalType();

          bool is_function =
              swift::isa<swift::AbstractFunctionDecl>(value_decl);

          bool skip_it = false;
          for (size_t rv_idx = 0; rv_idx < num_external_results; rv_idx++) {
            if (swift::ValueDecl *rv_decl = RV[rv_idx].getValueDecl()) {
              if (value_decl_kind == rv_decl->getKind()) {
                if (is_function) {
                  swift::DeclName rv_full_name = rv_decl->getName();
                  if (rv_full_name.matchesRef(value_decl_name)) {
                    // If the full names match, make sure the
                    // interface types match:
                    if (rv_decl->getInterfaceType()->getCanonicalType() ==
                        value_interface_type)
                      skip_it = true;
                  }
                } else {
                  skip_it = true;
                }

                if (skip_it)
                  break;
              }
            }
          }
          if (!skip_it)
            results.push_back(value_decl);
        }
      } else {
        m_persistent_vars->GetSwiftPersistentDecls(name_const_str, results,
                                                   results);
      }
    }

    for (size_t idx = 0; idx < results.size(); idx++) {
      swift::ValueDecl *value_decl = results[idx];
      // No import required.
      assert(&DC->getASTContext() == &value_decl->getASTContext());
      RV.push_back(swift::LookupResultEntry(value_decl));
    }

    return results.size() > 0;
  }

  swift::Identifier getPreferredPrivateDiscriminator() override {
    if (m_sc.comp_unit) {
      if (lldb_private::Module *module = m_sc.module_sp.get()) {
        if (lldb_private::SymbolFile *symbol_file =
                module->GetSymbolFile()) {
          std::string private_discriminator_string;
          if (symbol_file->GetCompileOption("-private-discriminator",
                                              private_discriminator_string,
                                              m_sc.comp_unit)) {
            return m_source_file.getASTContext().getIdentifier(
                private_discriminator_string);
          }
        }
      }
    }
    return swift::Identifier();
  }
};

/// A name lookup class for REPL and Playground mode.
class LLDBREPLNameLookup : public LLDBNameLookup {
public:
  LLDBREPLNameLookup(swift::SourceFile &source_file,
                     SwiftExpressionParser::SILVariableMap &variable_map,
                     SymbolContext &sc, ExecutionContextScope &exe_scope)
      : LLDBNameLookup(source_file, variable_map, sc, exe_scope) {}

  bool shouldGlobalize(swift::Identifier Name, swift::DeclKind kind) override {
    return false;
  }

  void didGlobalize(swift::Decl *Decl) override {}

  bool lookupOverrides(swift::DeclBaseName Name, swift::DeclContext *DC,
                       swift::SourceLoc Loc, bool IsTypeLookup,
                       ResultVector &RV) override {
    return false;
  }

  bool lookupAdditions(swift::DeclBaseName Name, swift::DeclContext *DC,
                       swift::SourceLoc Loc, bool IsTypeLookup,
                       ResultVector &RV) override {
    LLDB_SCOPED_TIMER();
    static unsigned counter = 0;
    unsigned count = counter++;

    StringRef NameStr = Name.getIdentifier().str();
    ConstString name_const_str(NameStr);

    if (m_log) {
      m_log->Printf("[LLDBREPLNameLookup::lookupAdditions (%u)] Searching for %s",
                    count, Name.getIdentifier().str().str().c_str());
    }

    // Find decls that come from the current compilation.
    std::vector<swift::ValueDecl *> current_compilation_results;
    for (auto result : RV) {
      auto result_decl = result.getValueDecl();
      auto result_decl_context = result_decl->getDeclContext();
      if (result_decl_context->isChildContextOf(DC) || result_decl_context == DC)
        current_compilation_results.push_back(result_decl);
    }

    // Find persistent decls, excluding decls that are equivalent to
    // decls from the current compilation.  This makes the decls from
    // the current compilation take precedence.
    std::vector<swift::ValueDecl *> persistent_decl_results;
    m_persistent_vars->GetSwiftPersistentDecls(name_const_str,
                                               current_compilation_results,
                                               persistent_decl_results);

    // Append the persistent decls that we found to the result vector.
    for (auto result : persistent_decl_results) {
      // No import required.
      assert(&DC->getASTContext() == &result->getASTContext());
      RV.push_back(swift::LookupResultEntry(result));
    }

    return !persistent_decl_results.empty();
  }

  swift::Identifier getPreferredPrivateDiscriminator() override {
    return swift::Identifier();
  }
};
}; // END Anonymous namespace


/// Returns the Swift type for a ValueObject representing a variable.
/// An invalid CompilerType is returned on error.
static CompilerType GetSwiftTypeForVariableValueObject(
    lldb::ValueObjectSP valobj_sp, lldb::StackFrameSP &stack_frame_sp,
    SwiftLanguageRuntime *runtime, lldb::BindGenericTypes bind_generic_types) {
  LLDB_SCOPED_TIMER();
  // Check that the passed ValueObject is valid.
  if (!valobj_sp || valobj_sp->GetError().Fail())
    return {};
  CompilerType result = valobj_sp->GetCompilerType();
  if (!result)
    return {};
  if (bind_generic_types != lldb::eDontBind)
    result = runtime->BindGenericTypeParameters(*stack_frame_sp, result);
  if (!result)
    return {};
  if (!result.GetTypeSystem()->SupportsLanguage(lldb::eLanguageTypeSwift))
    return {};
  return result;
}

/// Return the type for a local variable. This function is threading a
/// fine line between using dynamic type resolution to resolve generic
/// types and not resolving too much: Objective-C classes can have
/// more specific private implementations that LLDB can resolve, but
/// SwiftASTContext cannot see because there is no header file that
/// would declare them.
CompilerType SwiftExpressionParser::ResolveVariable(
    lldb::VariableSP variable_sp, lldb::StackFrameSP &stack_frame_sp,
    SwiftLanguageRuntime *runtime, lldb::DynamicValueType use_dynamic,
    lldb::BindGenericTypes bind_generic_types) {
  LLDB_SCOPED_TIMER();
  lldb::ValueObjectSP valobj_sp =
      stack_frame_sp->GetValueObjectForFrameVariable(variable_sp,
                                                     lldb::eNoDynamicValues);
  const bool use_dynamic_value = use_dynamic > lldb::eNoDynamicValues;

  CompilerType var_type = GetSwiftTypeForVariableValueObject(
      valobj_sp, stack_frame_sp, runtime, bind_generic_types);

  if (!var_type.IsValid())
    return {};

  // If the type can't be realized and dynamic types are allowed, fall back to
  // the dynamic type. We can only do this when not binding generic types
  // though, as we don't bind the generic parameters in that case.
  if (!SwiftASTContext::IsFullyRealized(var_type) &&
      bind_generic_types != lldb::eDontBind && use_dynamic_value) {
    var_type = GetSwiftTypeForVariableValueObject(
        valobj_sp->GetDynamicValue(use_dynamic), stack_frame_sp, runtime,
        bind_generic_types);
    if (!var_type.IsValid())
      return {};
  }
  return var_type;
}

lldb::VariableSP SwiftExpressionParser::FindSelfVariable(Block *block) {
  if (!block)
    return {};

  Function *function = block->CalculateSymbolContextFunction();

  if (!function)
    return {};

  constexpr bool can_create = true;
  Block &function_block(function->GetBlock(can_create));

  lldb::VariableListSP variable_list_sp(
      function_block.GetBlockVariableList(true));

  if (!variable_list_sp)
    return {};

  return variable_list_sp->FindVariable(ConstString("self"));
}

static void AddRequiredAliases(Block *block, lldb::StackFrameSP &stack_frame_sp,
                               SwiftASTContextForExpressions &swift_ast_context,
                               SwiftASTManipulator &manipulator,
                               lldb::DynamicValueType use_dynamic,
                               lldb::BindGenericTypes bind_generic_types) {
  LLDB_SCOPED_TIMER();

  // First emit the typealias for "$__lldb_context".
  lldb::VariableSP self_var_sp = SwiftExpressionParser::FindSelfVariable(block);

  if (!self_var_sp)
    return;

  auto *swift_runtime =
      SwiftLanguageRuntime::Get(stack_frame_sp->GetThread()->GetProcess());
  CompilerType self_type = SwiftExpressionParser::ResolveVariable(
      self_var_sp, stack_frame_sp, swift_runtime, use_dynamic,
      bind_generic_types);

  if (!self_type.IsValid()) {
    if (Type *type = self_var_sp->GetType()) {
      self_type = type->GetForwardCompilerType();
    }
  }

  if (!self_type.IsValid() ||
      !self_type.GetTypeSystem()->SupportsLanguage(lldb::eLanguageTypeSwift))
    return;

  // Import before getting the unbound version, because the unbound
  // version may not be in the mangled name map.

  CompilerType imported_self_type = ImportType(swift_ast_context, self_type);

  if (!imported_self_type.IsValid())
    return;

  auto *stack_frame = stack_frame_sp.get();
  if (bind_generic_types != lldb::eDontBind) {
    imported_self_type = swift_runtime->BindGenericTypeParameters(
        *stack_frame, imported_self_type);
    if (!imported_self_type)
      return;
  }

  {
    auto swift_type_system =
        imported_self_type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwift>();
    if (!swift_type_system)
      return;

    // This might be a referenced type, in which case we really want to
    // extend the referent:
    imported_self_type = swift_type_system->GetReferentType(
        imported_self_type.GetOpaqueQualType());
    if (!imported_self_type)
      return;
  }

  {
    auto swift_type_system =
        imported_self_type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwift>();
    if (!swift_type_system)
      return;

    // If we are extending a generic class it's going to be a metatype,
    // and we have to grab the instance type:
    imported_self_type = swift_type_system->GetInstanceType(
        imported_self_type.GetOpaqueQualType());
    if (!imported_self_type)
      return;
  }

  Flags imported_self_type_flags(imported_self_type.GetTypeInfo());

  auto swift_self_type = GetSwiftType(imported_self_type);
  if (!swift_self_type) {
    Log *log = GetLog(LLDBLog::Types | LLDBLog::Expressions);
    if (log)
      log->Printf("Couldn't get SwiftASTContext type for self type %s.",
                  imported_self_type.GetDisplayTypeName().AsCString("<unknown>"));
    
    return;
  }

  swift::Type object_type = swift_self_type->getWithoutSpecifierType();

  if (object_type.getPointer() &&
      (object_type.getPointer() != imported_self_type.GetOpaqueQualType()))
    imported_self_type = ToCompilerType(object_type.getPointer());

  // If 'self' is a weak storage type, it must be an optional.  Look
  // through it and unpack the argument of "optional".
  if (swift::WeakStorageType *weak_storage_type =
          GetSwiftType(imported_self_type)->getAs<swift::WeakStorageType>()) {
    swift::Type referent_type = weak_storage_type->getReferentType();
    swift::BoundGenericEnumType *optional_type =
        referent_type->getAs<swift::BoundGenericEnumType>();

    if (!optional_type)
      return;

    swift::Type first_arg_type = optional_type->getGenericArgs()[0];

    swift::ClassType *self_class_type =
        first_arg_type->getAs<swift::ClassType>();

    if (!self_class_type)
      return;

    imported_self_type = ToCompilerType(self_class_type);
  }

  imported_self_type_flags.Reset(imported_self_type.GetTypeInfo());
  if (imported_self_type_flags.AllClear(lldb::eTypeIsGenericTypeParam)) {
    swift::ValueDecl *type_alias_decl = nullptr;

    type_alias_decl = manipulator.MakeGlobalTypealias(
        swift_ast_context.GetASTContext()->getIdentifier("$__lldb_context"),
        imported_self_type);

    if (!type_alias_decl) {
      Log *log = GetLog(LLDBLog::Expressions);
      if (log)
        log->Printf("SEP:AddRequiredAliases: Failed to make the "
                    "$__lldb_context typealias.");
    }
  } else {
    Log *log = GetLog(LLDBLog::Expressions);
    if (log)
      log->Printf("SEP:AddRequiredAliases: Failed to resolve the self "
                  "archetype - could not make the $__lldb_context "
                  "typealias.");
  }
  // Alias the builtin type, since we can't use it directly in source code.
  auto builtin_ptr_t = swift_ast_context.GetBuiltinRawPointerType();
  manipulator.MakeGlobalTypealias(
      swift_ast_context.GetASTContext()->getIdentifier("$__lldb_builtin_ptr_t"),
      builtin_ptr_t, false);
}

static void ResolveSpecialNames(
    SymbolContext &sc, ExecutionContextScope &exe_scope,
    SwiftASTContextForExpressions &ast_context,
    llvm::SmallVectorImpl<swift::Identifier> &special_names,
    llvm::SmallVectorImpl<SwiftASTManipulator::VariableInfo> &local_variables) {
  Log *log = GetLog(LLDBLog::Expressions);
  LLDB_SCOPED_TIMER();
  
  if (!sc.target_sp)
    return;

  auto *persistent_state =
      sc.target_sp->GetSwiftPersistentExpressionState(exe_scope);

  std::set<ConstString> resolved_names;

  for (swift::Identifier &name : special_names) {
    ConstString name_cs = ConstString(name.str());

    if (resolved_names.count(name_cs))
      continue;

    resolved_names.insert(name_cs);

    if (log)
      log->Printf("Resolving special name %s", name_cs.AsCString());

    lldb::ExpressionVariableSP expr_var_sp =
        persistent_state->GetVariable(name_cs);

    if (!expr_var_sp)
      continue;

    CompilerType var_type = expr_var_sp->GetCompilerType();

    if (!var_type.IsValid())
      continue;

    if (!var_type.GetTypeSystem()->SupportsLanguage(lldb::eLanguageTypeSwift))
      continue;

    CompilerType target_type;
    Status error;

    target_type = ast_context.ImportType(var_type, error);

    if (!target_type)
      continue;

    SwiftASTManipulatorBase::VariableMetadataSP metadata_sp(
        new SwiftASTManipulatorBase::VariableMetadataPersistent(expr_var_sp));

    auto introducer = llvm::cast<SwiftExpressionVariable>(expr_var_sp.get())
                       ->GetIsModifiable()
                   ? swift::VarDecl::Introducer::Var
                   : swift::VarDecl::Introducer::Let;
    SwiftASTManipulator::VariableInfo variable_info(
        target_type, ast_context.GetASTContext()->getIdentifier(name.str()),
        metadata_sp, introducer);

    local_variables.push_back(variable_info);
  }
}

/// Initialize the SwiftASTContext and return the wrapped
/// swift::ASTContext when successful.
static swift::ASTContext *
SetupASTContext(SwiftASTContextForExpressions &swift_ast_context,
                DiagnosticManager &diagnostic_manager,
                std::function<bool()> disable_objc_runtime, bool repl,
                bool playground) {
  // Lazily get the clang importer if we can to make sure it exists in
  // case we need it.
  if (!swift_ast_context.GetClangImporter()) {
    std::string swift_error =
        swift_ast_context.GetFatalErrors().AsCString("error: unknown error.");
    diagnostic_manager.PutString(eDiagnosticSeverityError, swift_error);
    diagnostic_manager.PutString(eDiagnosticSeverityRemark,
                                 "Couldn't initialize Swift expression "
                                 "evaluator due to previous errors.");
    return nullptr;
  }

  if (swift_ast_context.HasFatalErrors()) {
    diagnostic_manager.PutString(eDiagnosticSeverityError,
                                 "The AST context is in a fatal error state.");
    return nullptr;
  }

  swift::ASTContext *ast_context = swift_ast_context.GetASTContext();
  if (!ast_context) {
    diagnostic_manager.PutString(
        eDiagnosticSeverityError,
        "Couldn't initialize the AST context.  Please check your settings.");
    return nullptr;
  }

  if (swift_ast_context.HasFatalErrors()) {
    diagnostic_manager.PutString(eDiagnosticSeverityError,
                                 "The AST context is in a fatal error state.");
    return nullptr;
  }

  // TODO: Find a way to get contraint-solver output sent to a stream
  //       so we can log it.
  // swift_ast_context.GetLanguageOptions().DebugConstraintSolver = true;
  swift_ast_context.ClearDiagnostics();

  // No longer part of debugger support, set it separately.
  swift_ast_context.GetLanguageOptions().EnableDollarIdentifiers = true;
  swift_ast_context.GetLanguageOptions().EnableAccessControl =
      (repl || playground);
  swift_ast_context.GetLanguageOptions().EnableTargetOSChecking = false;

  if (disable_objc_runtime())
    swift_ast_context.GetLanguageOptions().EnableObjCInterop = false;

  swift_ast_context.GetLanguageOptions().Playground = repl || playground;
  swift_ast_context.GetIRGenOptions().Playground = repl || playground;

  // For the expression parser and REPL we want to relax the
  // requirement that you put "try" in front of every expression that
  // might throw.
  if (repl || !playground)
    swift_ast_context.GetLanguageOptions().EnableThrowWithoutTry = true;

  swift_ast_context.GetIRGenOptions().OutputKind =
      swift::IRGenOutputKind::Module;
  swift_ast_context.GetIRGenOptions().OptMode =
      swift::OptimizationMode::NoOptimization;
  // Normally we'd like to verify, but unfortunately the verifier's
  // error mode is abort().
  swift_ast_context.GetIRGenOptions().Verify = false;
  swift_ast_context.GetIRGenOptions().ForcePublicLinkage = true;

  swift_ast_context.GetIRGenOptions().DisableRoundTripDebugTypes = true;
  return ast_context;
}

/// Returns the buffer_id for the expression's source code.
static std::pair<unsigned, std::string>
CreateMainFile(SwiftASTContextForExpressions &swift_ast_context,
               StringRef filename, StringRef text,
               const EvaluateExpressionOptions &options) {
  const bool generate_debug_info = options.GetGenerateDebugInfo();
  swift_ast_context.SetGenerateDebugInfo(generate_debug_info
                                           ? swift::IRGenDebugInfoLevel::Normal
                                           : swift::IRGenDebugInfoLevel::None);
  swift::IRGenOptions &ir_gen_options = swift_ast_context.GetIRGenOptions();

  if (generate_debug_info) {
    std::string temp_source_path;
    if (SwiftASTManipulator::SaveExpressionTextToTempFile(text, options, temp_source_path)) {
      auto error_or_buffer_ap =
          llvm::MemoryBuffer::getFile(temp_source_path.c_str());
      if (error_or_buffer_ap.getError() == std::error_condition()) {
        unsigned buffer_id =
            swift_ast_context.GetSourceManager().addNewSourceBuffer(
                std::move(error_or_buffer_ap.get()));

        llvm::SmallString<256> source_dir(temp_source_path);
        llvm::sys::path::remove_filename(source_dir);
        ir_gen_options.DebugCompilationDir = std::string(source_dir);

        return {buffer_id, temp_source_path};
      }
    }
  }

  std::unique_ptr<llvm::MemoryBuffer> expr_buffer(
      llvm::MemoryBuffer::getMemBufferCopy(text, filename));
  unsigned buffer_id = swift_ast_context.GetSourceManager().addNewSourceBuffer(
      std::move(expr_buffer));
  return {buffer_id, filename.str()};
}

/// Attempt to materialize one variable.
static llvm::Optional<SwiftExpressionParser::SILVariableInfo>
MaterializeVariable(SwiftASTManipulatorBase::VariableInfo &variable,
                    SwiftUserExpression &user_expression,
                    Materializer &materializer,
                    SwiftASTManipulator &manipulator,
                    lldb::StackFrameWP &stack_frame_wp,
                    DiagnosticManager &diagnostic_manager, Log *log,
                    bool repl) {
  LLDB_SCOPED_TIMER();
  uint64_t offset = 0;
  bool needs_init = false;

  bool is_result = llvm::isa<SwiftASTManipulatorBase::VariableMetadataResult>(
      variable.m_metadata.get());
  bool is_error = llvm::isa<SwiftASTManipulatorBase::VariableMetadataError>(
      variable.m_metadata.get());

  auto compiler_type = variable.GetType();
  // Add the persistent variable as a typeref compiler type.
  if (auto swift_ast_ctx =
          compiler_type.GetTypeSystem().dyn_cast_or_null<SwiftASTContext>()) {
    // Add the persistent variable as a typeref compiler type, but only if
    // doesn't have archetypes (which can be the case when we're evaluating an
    // expression as generic), since we can't mangle free-standing archetypes.
    if (!swift_ast_ctx->TypeHasArchetype(compiler_type))
      variable.SetType(
          swift_ast_ctx->GetTypeRefType(compiler_type.GetOpaqueQualType()));
  }

  if (is_result || is_error) {
    needs_init = true;

    Status error;

    if (repl) {
      if (!variable.GetType().IsVoidType()) {
        auto &repl_mat = *llvm::cast<SwiftREPLMaterializer>(&materializer);
        offset = repl_mat.AddREPLResultVariable(
            variable.GetType(), variable.GetDecl(),
            is_result ? &user_expression.GetResultDelegate()
                      : &user_expression.GetErrorDelegate(),
            error);
      }
    } else {
      CompilerType actual_type = variable.GetType();
      // Desugar '$lldb_context', etc.
      swift::Type actual_swift_type = GetSwiftType(actual_type);
      if (!actual_swift_type)
        return llvm::None;

      auto transformed_type =
          actual_swift_type.transform([](swift::Type t) -> swift::Type {
            if (auto *aliasTy =
                    swift::dyn_cast<swift::TypeAliasType>(t.getPointer())) {
              if (aliasTy && aliasTy->getDecl()->isDebuggerAlias()) {
                return aliasTy->getSinglyDesugaredType();
              }
            }
            return t;
          });

      if (!transformed_type)
        return llvm::None;

      actual_type =
          ToCompilerType(transformed_type->mapTypeOutOfContext().getPointer());
      auto swift_ast_ctx =
          actual_type.GetTypeSystem().dyn_cast_or_null<SwiftASTContext>();

      actual_type =
          swift_ast_ctx->GetTypeRefType(actual_type.GetOpaqueQualType());

      offset = materializer.AddResultVariable(
          actual_type, false, true,
          is_result ? &user_expression.GetResultDelegate()
                    : &user_expression.GetErrorDelegate(),
          error);
    }

    if (!error.Success()) {
      diagnostic_manager.Printf(
          eDiagnosticSeverityError, "couldn't add %s variable to struct: %s.\n",
          is_result ? "result" : "error", error.AsCString());
      return llvm::None;
    }

    if (log)
      log->Printf("Added %s variable to struct at offset %llu",
                  is_result ? "result" : "error", (unsigned long long)offset);
  } else if (auto *variable_metadata = llvm::dyn_cast<
                 SwiftASTManipulatorBase::VariableMetadataVariable>(
                 variable.m_metadata.get())) {
    Status error;

    offset = materializer.AddVariable(variable_metadata->m_variable_sp, error);

    if (!error.Success()) {
      diagnostic_manager.Printf(eDiagnosticSeverityError,
                                "couldn't add variable to struct: %s.\n",
                                error.AsCString());
      return llvm::None;
    }

    if (log)
      log->Printf("Added variable %s to struct at offset %llu",
                  variable_metadata->m_variable_sp->GetName().AsCString(),
                  (unsigned long long)offset);
  } else if (auto *variable_metadata = llvm::dyn_cast<
                 SwiftASTManipulatorBase::VariableMetadataPersistent>(
                 variable.m_metadata.get())) {
    needs_init = llvm::cast<SwiftExpressionVariable>(
                     variable_metadata->m_persistent_variable_sp.get())
                     ->m_swift_flags &
                 SwiftExpressionVariable::EVSNeedsInit;

    Status error;

    // When trying to materialize variables in the REPL, check whether
    // this is possibly a zero-sized type and call the correct function which
    // correctly handles zero-sized types. Unfortunately we currently have
    // this check scattered in several places in the codebase, we should at
    // some point centralize it.
    lldb::StackFrameSP stack_frame_sp = stack_frame_wp.lock();
    llvm::Optional<uint64_t> size =
        variable.GetType().GetByteSize(stack_frame_sp.get());
    if (repl && size && *size == 0) {
      auto &repl_mat = *llvm::cast<SwiftREPLMaterializer>(&materializer);
      offset = repl_mat.AddREPLResultVariable(
          variable.GetType(), variable.GetDecl(),
          &user_expression.GetPersistentVariableDelegate(), error);
    } else {
      // Transform the variable metadata to a typeref type if necessary.
      auto compiler_type =
          variable_metadata->m_persistent_variable_sp->GetCompilerType();
      if (auto swift_ast_ctx = compiler_type.GetTypeSystem()
                                   .dyn_cast_or_null<SwiftASTContext>()) {
        variable_metadata->m_persistent_variable_sp->SetCompilerType(
            swift_ast_ctx->GetTypeRefType(compiler_type.GetOpaqueQualType()));
      }

      offset = materializer.AddPersistentVariable(
          variable_metadata->m_persistent_variable_sp,
          &user_expression.GetPersistentVariableDelegate(), error);
    }

    if (!error.Success()) {
      diagnostic_manager.Printf(eDiagnosticSeverityError,
                                "couldn't add variable to struct: %s.\n",
                                error.AsCString());
      return llvm::None;
    }

    if (log)
      log->Printf(
          "Added persistent variable %s with flags 0x%llx to "
          "struct at offset %llu",
          variable_metadata->m_persistent_variable_sp->GetName().AsCString(),
          (unsigned long long)
              variable_metadata->m_persistent_variable_sp->m_flags,
          (unsigned long long)offset);
  }

  bool unowned_self = false;
  if (variable.IsSelf()) {
    if (auto swift_ts =
            compiler_type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwift>())

      if (auto kind = swift_ts->GetNonTriviallyManagedReferenceKind(
              compiler_type.GetOpaqueQualType()))
        unowned_self =
            *kind ==
            TypeSystemSwift::NonTriviallyManagedReferenceKind::eUnowned;
    }
  return SwiftExpressionParser::SILVariableInfo(
      variable.GetType(), offset, needs_init, unowned_self);
}

namespace {

/// This error indicates that the error has already been diagnosed.
struct PropagatedError : public llvm::ErrorInfo<PropagatedError> {
  static char ID;

  void log(llvm::raw_ostream &OS) const override { OS << "Propagated"; }
  std::error_code convertToErrorCode() const override {
    return inconvertibleErrorCode();
  }
};

/// This indicates an error in the SwiftASTContext.
struct SwiftASTContextError : public llvm::ErrorInfo<SwiftASTContextError> {
  static char ID;

  void log(llvm::raw_ostream &OS) const override { OS << "SwiftASTContext"; }
  std::error_code convertToErrorCode() const override {
    return inconvertibleErrorCode();
  }
};

/// This indicates an error in the SwiftASTContext.
struct ModuleImportError : public llvm::ErrorInfo<ModuleImportError> {
  static char ID;
  std::string msg;
  bool is_new_dylib;

  ModuleImportError(llvm::Twine message, bool is_new_dylib = false)
      : msg(message.str()), is_new_dylib(is_new_dylib) {}
  void log(llvm::raw_ostream &OS) const override {
    OS << "error while processing module import: ";
    OS << msg;
  }
  std::error_code convertToErrorCode() const override {
    return inconvertibleErrorCode();
  }
};

char PropagatedError::ID = 0;
char SwiftASTContextError::ID = 0;
char ModuleImportError::ID = 0;

/// This holds the result of ParseAndImport.
struct ParsedExpression {
  std::unique_ptr<SwiftASTManipulator> code_manipulator;
  swift::ASTContext &ast_context;
  swift::ModuleDecl &module;
  LLDBNameLookup &external_lookup;
  swift::SourceFile &source_file;
  std::string main_filename;
  unsigned buffer_id;
};

} // namespace

/// Attempt to parse an expression and import all the Swift modules
/// the expression and its context depend on.
static llvm::Expected<ParsedExpression> ParseAndImport(
    SwiftASTContextForExpressions &swift_ast_context, Expression &expr,
    SwiftExpressionParser::SILVariableMap &variable_map, unsigned &buffer_id,
    DiagnosticManager &diagnostic_manager,
    SwiftExpressionParser &swift_expr_parser,
    lldb::StackFrameWP &stack_frame_wp, SymbolContext &sc,
    ExecutionContextScope &exe_scope, 
    llvm::SmallVectorImpl<SwiftASTManipulator::VariableInfo> &local_variables,
    const EvaluateExpressionOptions &options,
    bool repl, bool playground) {
  Log *log = GetLog(LLDBLog::Expressions);
  LLDB_SCOPED_TIMER();

  auto should_disable_objc_runtime = [&]() {
    lldb::StackFrameSP this_frame_sp(stack_frame_wp.lock());
    if (!this_frame_sp)
      return false;
    lldb::ProcessSP process_sp(this_frame_sp->CalculateProcess());
    if (!process_sp)
      return false;
    return !ObjCLanguageRuntime::Get(*process_sp);
  };

  swift::ASTContext *ast_context =
      SetupASTContext(swift_ast_context, diagnostic_manager,
                      should_disable_objc_runtime, repl, playground);
  if (!ast_context)
    return make_error<SwiftASTContextError>();

  // If we are using the playground, hand import the necessary
  // modules.
  //
  // FIXME: We won't have to do this once the playground adds import
  //        statements for the things it needs itself.
  if (playground) {
    auto *persistent_state =
        sc.target_sp->GetSwiftPersistentExpressionState(exe_scope);

    Status error;
    SourceModule module_info;
    module_info.path.emplace_back("Swift");
    swift::ModuleDecl *module = swift_ast_context.GetModule(module_info, error);

    if (error.Fail() || !module) {
      LLDB_LOG(log, "couldn't load Swift Standard Library\n");
      return error.ToError();
    }

    persistent_state->AddHandLoadedModule(ConstString("Swift"),
                                          swift::ImportedModule(module));
  }

  std::string main_filename;
  std::tie(buffer_id, main_filename) = CreateMainFile(
      swift_ast_context, repl ? "<REPL>" : "<EXPR>", expr.Text(), options);

  char expr_name_buf[32];

  snprintf(expr_name_buf, sizeof(expr_name_buf), "__lldb_expr_%u",
           options.GetExpressionNumber());

  // Gather the modules that need to be implicitly imported.
  // The Swift stdlib needs to be imported before the SwiftLanguageRuntime can
  // be used.
  Status implicit_import_error;
  llvm::SmallVector<swift::AttributedImport<swift::ImportedModule>, 16>
      additional_imports;
  lldb::ProcessSP process_sp;
  if (lldb::StackFrameSP this_frame_sp = stack_frame_wp.lock())
    process_sp = this_frame_sp->CalculateProcess();
  swift_ast_context.LoadImplicitModules(sc.target_sp, process_sp, exe_scope);
  if (!SwiftASTContext::GetImplicitImports(swift_ast_context, sc, exe_scope,
                                           process_sp, additional_imports,
                                           implicit_import_error)) {
    const char *msg = implicit_import_error.AsCString();
    if (!msg)
      msg = "error status positive, but import still failed";
    return make_error<ModuleImportError>(msg);
  }

  swift::ImplicitImportInfo importInfo;
  importInfo.StdlibKind = swift::ImplicitStdlibKind::Stdlib;
  for (auto &attributed_import : additional_imports)
    importInfo.AdditionalImports.emplace_back(attributed_import);

  auto module_id = ast_context->getIdentifier(expr_name_buf);
  auto &module = *swift::ModuleDecl::create(module_id, *ast_context,
                                            importInfo);

  swift::SourceFileKind source_file_kind = swift::SourceFileKind::Library;
  if (playground || repl) {
    source_file_kind = swift::SourceFileKind::Main;
  }

  // Create the source file. Note, we disable delayed parsing for the
  // swift expression parser.
  swift::SourceFile *source_file = new (*ast_context)
      swift::SourceFile(module, source_file_kind, buffer_id,
                        swift::SourceFile::ParsingFlags::DisableDelayedBodies);
  module.addFile(*source_file);

  // Swift Modules that rely on shared libraries (not frameworks)
  // don't record the link information in the swiftmodule file, so we
  // can't really make them work without outside information.
  // However, in the REPL you can added -L & -l options to the initial
  // compiler startup, and we should dlopen anything that's been
  // stuffed on there and hope it will be useful later on.
  if (repl) {
    lldb::StackFrameSP this_frame_sp(stack_frame_wp.lock());

    if (this_frame_sp) {
      lldb::ProcessSP process_sp(this_frame_sp->CalculateProcess());
      if (process_sp) {
        Status error;
        swift_ast_context.LoadExtraDylibs(*process_sp.get(), error);
      }
    }
  }

  auto &invocation = swift_ast_context.GetCompilerInvocation();
  invocation.getFrontendOptions().ModuleName = expr_name_buf;
  invocation.getIRGenOptions().ModuleName = expr_name_buf;

  bool enable_bare_slash_regex_literals =
      sc.target_sp->GetSwiftEnableBareSlashRegex();
  invocation.getLangOptions().EnableBareSlashRegexLiterals =
      enable_bare_slash_regex_literals;
  invocation.getLangOptions().EnableExperimentalStringProcessing =
      enable_bare_slash_regex_literals;


  auto should_use_prestable_abi = [&]() {
    lldb::StackFrameSP this_frame_sp(stack_frame_wp.lock());
    if (!this_frame_sp)
      return false;
    lldb::ProcessSP process_sp(this_frame_sp->CalculateProcess());
    if (!process_sp)
      return false;
    auto *runtime = SwiftLanguageRuntime::Get(process_sp);
    return !runtime->IsABIStable();
  };

  invocation.getLangOptions().UseDarwinPreStableABIBit =
      should_use_prestable_abi();

  LLDBNameLookup *external_lookup;
  if (options.GetPlaygroundTransformEnabled() || options.GetREPLEnabled()) {
    external_lookup = new LLDBREPLNameLookup(*source_file, variable_map, sc,
                                             exe_scope);
  } else {
    external_lookup = new LLDBExprNameLookup(*source_file, variable_map, sc,
                                             exe_scope);
  }

  // FIXME: This call is here just so that the we keep the
  //        DebuggerClients alive as long as the Module we are not
  //        inserting them in.
  swift_ast_context.AddDebuggerClient(external_lookup);

  if (swift_ast_context.HasErrors())
    return make_error<SwiftASTContextError>();

  // Resolve the file's imports, including the implicit ones returned from
  // GetImplicitImports.
  swift::performImportResolution(*source_file);

  if (swift_ast_context.HasErrors())
    return make_error<ModuleImportError>(
        swift_ast_context.GetAllErrors().AsCString(
            "Explicit module import error"));

  std::unique_ptr<SwiftASTManipulator> code_manipulator;
  if (repl || !playground) {
    code_manipulator = std::make_unique<SwiftASTManipulator>(
        *source_file, repl, options.GetBindGenericTypes());

    if (!playground) {
      code_manipulator->RewriteResult();
    }
  }

  if (!playground && !repl) {
    lldb::StackFrameSP stack_frame_sp = stack_frame_wp.lock();

    bool local_context_is_swift = true;

    if (sc.block) {
      Function *function = sc.block->CalculateSymbolContextFunction();
      if (function && function->GetLanguage() != lldb::eLanguageTypeSwift)
        local_context_is_swift = false;
    }

    if (local_context_is_swift) {
      AddRequiredAliases(sc.block, stack_frame_sp, swift_ast_context,
                         *code_manipulator, options.GetUseDynamic(),
                         options.GetBindGenericTypes());

    }
    //
    // Register all magic variables.
    llvm::SmallVector<swift::Identifier, 2> special_names;
    llvm::StringRef persistent_var_prefix;
    if (!repl)
      persistent_var_prefix = "$";

    code_manipulator->FindSpecialNames(special_names, persistent_var_prefix);

    ResolveSpecialNames(sc, exe_scope, swift_ast_context, special_names,
                        local_variables);

    if (!code_manipulator->AddExternalVariables(local_variables))
      return make_error<StringError>(inconvertibleErrorCode(),
                                     "Could not add external variables.");

    stack_frame_sp.reset();
  }

  // Cache the source file's imports such that they're accessible to future
  // expression evaluations.
  {
    std::lock_guard<std::recursive_mutex> global_context_locker(
        IRExecutionUnit::GetLLVMGlobalContextMutex());

    Status auto_import_error;
    if (!SwiftASTContext::CacheUserImports(swift_ast_context, sc, exe_scope,
                                           process_sp, *source_file,
                                           auto_import_error)) {
      const char *msg = auto_import_error.AsCString();
      if (!msg) {
        // The import itself succeeded, but the AST context is in a
        // fatal error state. One way this can happen is if the import
        // triggered a dylib import, in which case the context is
        // purposefully poisoned.
        msg = "import may have triggered a dylib import";
      }
      return make_error<ModuleImportError>(msg, /*is_new_dylib=*/true);
    }
  }

  // After the swift code manipulator performed AST transformations,
  // verify that the AST we have in our hands is valid. This is a nop
  // for release builds, but helps catching bug when assertions are
  // turned on.
  swift::verify(*source_file);

  ParsedExpression result = {
    std::move(code_manipulator), *ast_context, module, *external_lookup,
    *source_file, std::move(main_filename), /*buffer_id*/0,
  };
  return std::move(result);
}

bool SwiftExpressionParser::Complete(CompletionRequest &request, unsigned line,
				     unsigned pos, unsigned typed_pos) {
  return false;
}

/// Replaces the call in the entrypoint from the sink function to the trampoline
/// function. This is done at the IR level so we can bypass the swift type
/// system.
static bool
RedirectCallFromSinkToTrampolineFunction(llvm::Module &module,
                                         SwiftASTManipulator &manipulator) {
  Log *log = GetLog(LLDBLog::Expressions);

  swift::Mangle::ASTMangler mangler;
  auto *entrypoint_decl = manipulator.GetEntrypointDecl();
  if (!entrypoint_decl) {
    log->Printf(
        "[RedirectCallFromSinkToTrampolineFunction] Could not set the call: no "
        "entrypoint decl.");
    return false;
  }

  auto *func_decl = manipulator.GetFuncDecl();
  if (!func_decl) {
    log->Printf(
        "[RedirectCallFromSinkToTrampolineFunction] Could not set the call: no "
        "func decl.");
    return false;
  }

  auto *trampoline_func_decl = manipulator.GetTrampolineDecl();
  if (!trampoline_func_decl) {
    log->Printf(
        "[RedirectCallFromSinkToTrampolineFunction] Could not set the call: no "
        "trampoline func decl.");
    return false;
  }

  auto *sink_decl = manipulator.GetSinkDecl();
  if (!sink_decl) {
    log->Printf(
        "[RedirectCallFromSinkToTrampolineFunction] Could not set the call: no "
        "sink decl.");
    return false;
  }

  auto expr_func_name = mangler.mangleEntity(entrypoint_decl);
  auto wrapped_func_name = mangler.mangleEntity(func_decl);
  auto trampoline_func_name = mangler.mangleEntity(trampoline_func_decl);
  auto sink_func_name = mangler.mangleEntity(sink_decl);

  llvm::Function *lldb_expr_func = module.getFunction(expr_func_name);
  llvm::Function *wrapped_func = module.getFunction(wrapped_func_name);
  llvm::Function *trampoline_func = module.getFunction(trampoline_func_name);
  llvm::Function *sink_func = module.getFunction(sink_func_name);

  assert(lldb_expr_func && wrapped_func && trampoline_func && sink_decl);
  if (!lldb_expr_func || !wrapped_func || !trampoline_func || !sink_func) {
    log->Printf(
        "[RedirectCallFromSinkToTrampolineFunction] Could not set the call: "
        "could not find one of the required functions in the IR.");
    return false;
  }

  auto *trampoline_func_type = trampoline_func->getFunctionType();
  auto trampoline_num_params = trampoline_func_type->getNumParams();
  // There should be at least 3 params, the raw pointer, the self type, and at
  // least one pointer to metadata.
  if (trampoline_num_params < 3) {
    log->Printf(
        "[RedirectCallFromSinkToTrampolineFunction] Could not set the call: "
        "trampoline function has %u parameters",
        trampoline_num_params);
    return false;
  }

  auto *sink_func_type = sink_func->getFunctionType();
  auto sink_num_params = sink_func_type->getNumParams();

  if (trampoline_num_params != sink_num_params) {
    log->Printf(
        "[RedirectCallFromSinkToTrampolineFunction] Could not set the call: "
        "trampoline function has %u parameters but sink has %u parameters.",
        trampoline_num_params, sink_num_params);
    return false;
  }

  auto &basic_blocks = lldb_expr_func->getBasicBlockList();
  // The entrypoint function should only have one basic block whith
  // materialization instructions and the call to the sink.
  if (basic_blocks.size() != 1) {
    log->Printf(
        "[RedirectCallFromSinkToTrampolineFunction] Could not set the call: "
        "entrypoint function has %zu basic blocks.",
        basic_blocks.size());
    return false;
  }

  auto &basic_block = basic_blocks.back();
  if (basic_block.getInstList().size() == 0) {
    log->Printf(
        "[RedirectCallFromSinkToTrampolineFunction] Could not set the call: "
        "basic block has no instructions.");
    return false;
  }

  // Find the call to the sink.
  llvm::CallInst *sink_call = nullptr;
  for (auto &I : basic_block.instructionsWithoutDebug()) {
    if (auto *call = llvm::dyn_cast<llvm::CallInst>(&I)) {
      if (call->getCalledFunction() == sink_func) {
        sink_call = call;
        break;
      }
    }
  }

  if (!sink_call) {
    log->Printf(
        "[RedirectCallFromSinkToTrampolineFunction] Could not set the call: "
        "call to sink function not found.");
    return false;
  }

  if (sink_call->arg_size() != sink_num_params) {
    log->Printf(
        "[RedirectCallFromSinkToTrampolineFunction] Could not set the call: "
        "call to sink function has %u arguments.",
        sink_call->arg_size());
    return false;
  }

  // The sink call should have at least three parameters, the pointer to
  // lldb_arg, a pointer to self and a pointer to the trampoline metadata of
  // self.
  llvm::Value *lldb_arg_ptr = sink_call->getArgOperand(0);
  llvm::Value *self_load = sink_call->getArgOperand(1);
  llvm::SmallVector<llvm::Value *> metadata_loads;
  for (size_t i = 2; i < sink_num_params; ++i)
    metadata_loads.emplace_back(sink_call->getArgOperand(i));

  // Delete the sink since we fished out the values we needed.
  sink_call->eraseFromParent();
  sink_func->eraseFromParent();

  // We need to fish the pointer to self, which the load instruction loads.
  llvm::Value *self_opaque_ptr = nullptr;
  if (auto *load = llvm::dyn_cast<llvm::LoadInst>(self_load))
    self_opaque_ptr = load->getPointerOperand();
  if (!self_opaque_ptr) {
    log->Printf(
        "[RedirectCallFromSinkToTrampolineFunction] Could not set the call: "
        "could not find the argument of the load of the self pointer.");
    return false;
  }

  auto &it = basic_block.getInstList().back();
  // Initialize the builder from the last instruction since we want to place the
  // new call there.
  llvm::IRBuilder<> builder(&it);

  llvm::Type *lldb_arg_type = trampoline_func_type->getParamType(1);
  llvm::Type *self_type = trampoline_func_type->getParamType(2);

  // Bitcast the operands to the expected types, since they were type-erased
  // in the call to the sink.
  auto *self_ptr = builder.CreateBitCast(self_opaque_ptr, lldb_arg_type);

  llvm::SmallVector<llvm::Value *> trampoline_call_params;
  trampoline_call_params.push_back(lldb_arg_ptr);
  trampoline_call_params.push_back(self_ptr);
  for (auto &metadata_load : metadata_loads)
    trampoline_call_params.push_back(
        builder.CreateBitCast(metadata_load, self_type));

  // Finally, create the call.
  builder.CreateCall(trampoline_func_type, trampoline_func,
                     trampoline_call_params);
  return true;
}

SwiftExpressionParser::ParseResult
SwiftExpressionParser::Parse(DiagnosticManager &diagnostic_manager,
                             uint32_t first_line, uint32_t last_line) {
  using ParseResult = SwiftExpressionParser::ParseResult;
  Log *log = GetLog(LLDBLog::Expressions);
  LLDB_SCOPED_TIMER();

  SwiftExpressionParser::SILVariableMap variable_map;

  // Helper function to diagnose errors in m_swift_scratch_context.
  unsigned buffer_id = UINT32_MAX;
  auto DiagnoseSwiftASTContextError = [&]() {
    assert(m_swift_ast_ctx.HasErrors() && "error expected");
    m_swift_ast_ctx.PrintDiagnostics(diagnostic_manager, buffer_id, first_line,
                                     last_line);
  };

  // In the case of playgrounds, we turn all rewriting functionality off.
  const bool repl = m_options.GetREPLEnabled();
  const bool playground = m_options.GetPlaygroundTransformEnabled();

  if (!m_exe_scope)
    return ParseResult::unrecoverable_error;

  // Parse the expression and import all nececssary swift modules.
  auto parsed_expr = ParseAndImport(
      m_swift_ast_ctx, m_expr, variable_map, buffer_id, diagnostic_manager,
      *this, m_stack_frame_wp, m_sc, *m_exe_scope, m_local_variables, m_options,
      repl, playground);

  if (!parsed_expr) {
    bool retry = false;
    handleAllErrors(
        parsed_expr.takeError(),
        [&](const ModuleImportError &MIE) {
          diagnostic_manager.PutString(eDiagnosticSeverityError, MIE.message());
          // There are no fallback contexts in REPL and playgrounds.
          if (repl || playground || MIE.is_new_dylib) {
            retry = true;
            return;
          }
          if (!m_sc.target_sp->UseScratchTypesystemPerModule()) {
            // This, together with the fatal error forces
            // a per-module scratch to be instantiated on
            // retry.
            m_sc.target_sp->SetUseScratchTypesystemPerModule(true);
            m_swift_ast_ctx.RaiseFatalError(MIE.message());
            retry = true;
          }
        },
        [&](const SwiftASTContextError &SACE) {
          DiagnoseSwiftASTContextError();
        },
        [&](const StringError &SE) {
          diagnostic_manager.PutString(eDiagnosticSeverityError,
                                       SE.getMessage());
        },
        [](const PropagatedError &P) {});

    // Signal that we want to retry the expression exactly once with a
    // fresh SwiftASTContext initialized with the flags from the
    // current lldb::Module / Swift dylib to avoid header search
    // mismatches.
    if (retry)
      return ParseResult::retry_fresh_context;

    // Unrecoverable error.
    return ParseResult::unrecoverable_error;
  }

  if (log) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    parsed_expr->source_file.dump(ss);
    ss.flush();

    log->Printf("Source file before type checking:");
    log->PutCString(s.c_str());
  }

  swift::bindExtensions(parsed_expr->module);
  swift::performTypeChecking(parsed_expr->source_file);

  if (m_swift_ast_ctx.HasErrors()) {
    DiagnoseSwiftASTContextError();
    return ParseResult::unrecoverable_error;
  }
  if (log) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    parsed_expr->source_file.dump(ss);
    ss.flush();

    log->Printf("Source file after type checking:");
    log->PutCString(s.c_str());
  }

  if (repl) {
    parsed_expr->code_manipulator->MakeDeclarationsPublic();
  }

  Status error;
  if (!playground) {
    parsed_expr->code_manipulator->FixupResultAfterTypeChecking(error);

    if (!error.Success()) {
      diagnostic_manager.PutString(eDiagnosticSeverityError, error.AsCString());
      return ParseResult::unrecoverable_error;
    }
  } else {
    swift::performPlaygroundTransform(
        parsed_expr->source_file,
        m_options.GetPlaygroundTransformHighPerformance());
  }

  // FIXME: We now should have to do the name binding and type
  //        checking again, but there should be only the result
  //        variable to bind up at this point.
  if (log) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    parsed_expr->source_file.dump(ss);
    ss.flush();

    log->Printf("Source file after FixupResult:");
    log->PutCString(s.c_str());
  }

  // Allow variables to be re-used from previous REPL statements.
  if (m_sc.target_sp && (repl || !playground)) {
    Status error;
    auto *persistent_state =
        m_sc.target_sp->GetSwiftPersistentExpressionState(*m_exe_scope);

    llvm::SmallVector<size_t, 1> declaration_indexes;
    parsed_expr->code_manipulator->FindVariableDeclarations(declaration_indexes,
                                                            repl);

    for (size_t declaration_index : declaration_indexes) {
      SwiftASTManipulator::VariableInfo &variable_info =
          parsed_expr->code_manipulator->GetVariableInfo()[declaration_index];

      CompilerType imported_type =
          ImportType(m_swift_ast_ctx, variable_info.GetType());

      if (!imported_type)
        continue;

      lldb::ExpressionVariableSP persistent_variable =
          persistent_state->AddNewlyConstructedVariable(
              new SwiftExpressionVariable(
                  m_sc.target_sp.get(),
                  ConstString(variable_info.GetName().str()), imported_type,
                  m_sc.target_sp->GetArchitecture().GetByteOrder(),
                  m_sc.target_sp->GetArchitecture().GetAddressByteSize()));
      // Detect global resilient variables in a fixed value buffer.
      // Globals without a fixed size are placed in a fixed-size buffer.
      auto *var_decl = variable_info.GetDecl();
      if (var_decl && var_decl->getDeclContext()->isModuleScopeContext())
        if (!m_swift_ast_ctx.IsFixedSize(imported_type))
          persistent_variable->m_flags |=
              ExpressionVariable::EVIsSwiftFixedBuffer;
      if (repl) {
        persistent_variable->m_flags |= ExpressionVariable::EVKeepInTarget;
        persistent_variable->m_flags |=
            ExpressionVariable::EVIsProgramReference;
      } else {
        persistent_variable->m_flags |= ExpressionVariable::EVNeedsAllocation;
        persistent_variable->m_flags |= ExpressionVariable::EVKeepInTarget;
        llvm::cast<SwiftExpressionVariable>(persistent_variable.get())
            ->m_swift_flags |= SwiftExpressionVariable::EVSNeedsInit;
      }

      swift::VarDecl *decl = variable_info.GetDecl();
      if (decl) {
        auto swift_var =
            llvm::cast<SwiftExpressionVariable>(persistent_variable.get());
        swift_var->SetIsModifiable(!decl->isLet());
        swift_var->SetIsComputed(!decl->hasStorage());
      }

      variable_info.m_metadata.reset(
          new SwiftASTManipulatorBase::VariableMetadataPersistent(
              persistent_variable));

      persistent_state->RegisterSwiftPersistentDecl(decl);
    }

    if (repl) {
      llvm::SmallVector<swift::ValueDecl *, 1> non_variables;
      parsed_expr->code_manipulator->FindNonVariableDeclarations(non_variables);

      for (swift::ValueDecl *decl : non_variables) {
        persistent_state->RegisterSwiftPersistentDecl(decl);
      }
    }
  }

  if (!playground && !repl) {
    parsed_expr->code_manipulator->FixCaptures();

    // FIXME: This currently crashes with Assertion failed: (BufferID != -1),
    //        function findBufferContainingLoc, file
    //        llvm/tools/swift/include/swift/Basic/SourceManager.h, line 92.
    //
    // if (log)
    // {
    //     std::string s;
    //     llvm::raw_string_ostream ss(s);
    //     parsed_expr->source_file.dump(ss);
    //     ss.flush();
    //
    //     log->Printf("Source file after capture fixing:");
    //     log->PutCString(s.c_str());
    // }

    if (log) {
      log->Printf("Variables:");

      for (const SwiftASTManipulatorBase::VariableInfo &variable :
           parsed_expr->code_manipulator->GetVariableInfo()) {
        StreamString ss;
        variable.Print(ss);
        log->Printf("  %s", ss.GetData());
      }
    }
  }

  if (repl || !playground)
    if (auto *materializer = m_expr.GetMaterializer())
      for (auto &variable : parsed_expr->code_manipulator->GetVariableInfo()) {
        auto &swift_expr = *static_cast<SwiftUserExpression *>(&m_expr);
        auto var_info = MaterializeVariable(
            variable, swift_expr, *materializer, *parsed_expr->code_manipulator,
            m_stack_frame_wp, diagnostic_manager, log, repl);
        if (!var_info)
          return ParseResult::unrecoverable_error;

        const char *name = ConstString(variable.GetName().get()).GetCString();
        variable_map[name] = *var_info;
      }

  if (log) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    parsed_expr->source_file.dump(ss);
    ss.flush();

    log->Printf("Source file before SILgen:");;
    log->PutCString(s.c_str());
  }
  
  // FIXME: Should share TypeConverter instances
  std::unique_ptr<swift::Lowering::TypeConverter> sil_types(
      new swift::Lowering::TypeConverter(
          *parsed_expr->source_file.getParentModule()));

  std::unique_ptr<swift::SILModule> sil_module = swift::performASTLowering(
      parsed_expr->source_file, *sil_types, m_swift_ast_ctx.GetSILOptions());

  if (log) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    sil_module->print(ss, &parsed_expr->module);
    ss.flush();

    log->Printf("SIL module before linking:");
    log->PutCString(s.c_str());
  }

  if (m_swift_ast_ctx.HasErrors()) {
    DiagnoseSwiftASTContextError();
    return ParseResult::unrecoverable_error;
  }

  if (log) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    sil_module->print(ss, &parsed_expr->module);
    ss.flush();

    log->Printf("Generated SIL module:");
    log->PutCString(s.c_str());
  }

  runSILDiagnosticPasses(*sil_module);
  runSILLoweringPasses(*sil_module);

  if (log) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    sil_module->print(ss, &parsed_expr->module);
    ss.flush();

    log->Printf("SIL module after diagnostic passes:");
    log->PutCString(s.c_str());
  }

  if (m_swift_ast_ctx.HasErrors()) {
    DiagnoseSwiftASTContextError();
    return ParseResult::unrecoverable_error;
  }

  {
    std::lock_guard<std::recursive_mutex> global_context_locker(
        IRExecutionUnit::GetLLVMGlobalContextMutex());

    const auto &IRGenOpts = m_swift_ast_ctx.GetIRGenOptions();

    auto GenModule = swift::performIRGeneration(
        &parsed_expr->module, IRGenOpts, m_swift_ast_ctx.GetTBDGenOptions(),
        std::move(sil_module), "lldb_module",
        swift::PrimarySpecificPaths("", parsed_expr->main_filename),
        llvm::ArrayRef<std::string>());

    if (GenModule) {
      swift::performLLVMOptimizations(IRGenOpts, GenModule.getModule(),
                                      GenModule.getTargetMachine(), nullptr);
    }
    auto ContextAndModule = std::move(GenModule).release();
    m_llvm_context.reset(ContextAndModule.first);
    m_module.reset(ContextAndModule.second);
  }

  if (m_swift_ast_ctx.HasErrors()) {
    DiagnoseSwiftASTContextError();
    return ParseResult::unrecoverable_error;
  }

  if (!m_module) {
    auto &warnings = m_swift_ast_ctx.GetModuleImportWarnings();
    for (StringRef message : warnings) {
      // FIXME: Don't store diagnostics as strings.
      auto severity = eDiagnosticSeverityWarning;
      if (message.consume_front("warning: "))
        severity = eDiagnosticSeverityWarning;
      if (message.consume_front("error: "))
        severity = eDiagnosticSeverityError;
      diagnostic_manager.PutString(severity, message);
    }
    std::string error = "couldn't IRGen expression";
    diagnostic_manager.Printf(
        eDiagnosticSeverityError, "couldn't IRGen expression. %s",
        warnings.empty()
            ? "Please enable the expression log by running \"log enable lldb "
              "expr\", then run the failing expression again, and file a "
              "bugreport with the log output."
            : "Please check the above error messages for possible root causes.");
    return ParseResult::unrecoverable_error;
  }

  if (log) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    m_module->print(ss, NULL);
    ss.flush();

    log->Printf("Generated IR module:");
    log->PutCString(s.c_str());
  }

  if (m_options.GetBindGenericTypes() == lldb::eDontBind &&
      !RedirectCallFromSinkToTrampolineFunction(
          *m_module.get(), *parsed_expr->code_manipulator.get())) {
    diagnostic_manager.Printf(
        eDiagnosticSeverityError,
        "couldn't setup call to the trampoline function. Please enable the "
        "expression log by running \"log enable lldb "
        "expr\", then run the failing expression again, and file a "
        "bugreport with the log output.");
    return ParseResult::unrecoverable_error;
  }

  if (log) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    m_module->print(ss, NULL);
    ss.flush();

    log->Printf("Generated IR module after replacing call to sink:");
    log->PutCString(s.c_str());
  }

  {
    std::lock_guard<std::recursive_mutex> global_context_locker(
        IRExecutionUnit::GetLLVMGlobalContextMutex());

    LLVMVerifyModule((LLVMOpaqueModule *)m_module.get(), LLVMReturnStatusAction,
                     nullptr);
  }

  if (m_swift_ast_ctx.HasErrors())
    return ParseResult::unrecoverable_error;

  // The Parse succeeded!  Now put this module into the context's list
  // of loaded modules, and copy the Decls that were globalized as
  // part of the parse from the staging area in the external lookup
  // object into the SwiftPersistentExpressionState.
  swift::ModuleDecl *module = &parsed_expr->module;
  parsed_expr->ast_context.addLoadedModule(module);
  m_swift_ast_ctx.CacheModule(module);
  if (m_sc.target_sp) {
    auto *persistent_state =
        m_sc.target_sp->GetSwiftPersistentExpressionState(*m_exe_scope);
    persistent_state->CopyInSwiftPersistentDecls(
        parsed_expr->external_lookup.GetStagedDecls());
  }
  return ParseResult::success;
}

static bool FindFunctionInModule(ConstString &mangled_name,
                                 llvm::Module *module, const char *orig_name,
                                 bool exact) {
  LLDB_SCOPED_TIMER();
  swift::Demangle::Context demangle_ctx;
  for (llvm::Module::iterator fi = module->getFunctionList().begin(),
                              fe = module->getFunctionList().end();
       fi != fe; ++fi) {
    if (exact) {
      if (!fi->getName().str().compare(orig_name)) {
        mangled_name.SetCString(fi->getName().str().c_str());
        return true;
      }
    } else {
      if (fi->getName().str().find(orig_name) != std::string::npos) {
        mangled_name.SetCString(fi->getName().str().c_str());
        return true;
      }

      // The new demangling is cannier about compression, so the name
      // may not be in the mangled name plain.  Let's demangle it and
      // see if we can find it in the demangled nodes.
      demangle_ctx.clear();

      auto *node_ptr = SwiftLanguageRuntime::DemangleSymbolAsNode(fi->getName(),
                                                                  demangle_ctx);
      if (node_ptr) {
        if (node_ptr->getKind() != swift::Demangle::Node::Kind::Global)
          continue;
        if (node_ptr->getNumChildren() != 1)
          continue;
        node_ptr = node_ptr->getFirstChild();
        if (node_ptr->getKind() != swift::Demangle::Node::Kind::Function)
          continue;
        size_t num_children = node_ptr->getNumChildren();
        for (size_t i = 0; i < num_children; i++) {
          swift::Demangle::NodePointer child_ptr = node_ptr->getChild(i);
          if (child_ptr->getKind() == swift::Demangle::Node::Kind::Identifier) {
            if (!child_ptr->hasText())
              continue;
            if (child_ptr->getText().contains(orig_name)) {
              mangled_name.SetCString(fi->getName().str().c_str());
              return true;
            }
          }
        }
      }
    }
  }

  return false;
}

Status SwiftExpressionParser::PrepareForExecution(
    lldb::addr_t &func_addr, lldb::addr_t &func_end,
    lldb::IRExecutionUnitSP &execution_unit_sp, ExecutionContext &exe_ctx,
    bool &can_interpret, ExecutionPolicy execution_policy) {
  LLDB_SCOPED_TIMER();
  Status err;
  Log *log = GetLog(LLDBLog::Expressions);

  if (!m_module) {
    err.SetErrorString("Can't prepare a NULL module for execution");
    return err;
  }

  const char *orig_name = nullptr;

  bool exact = false;

  if (m_options.GetPlaygroundTransformEnabled() || m_options.GetREPLEnabled()) {
    orig_name = "main";
    exact = true;
  } else {
    orig_name = "$__lldb_expr";
  }

  ConstString function_name;

  if (!FindFunctionInModule(function_name, m_module.get(), orig_name, exact)) {
    err.SetErrorToGenericError();
    err.SetErrorStringWithFormat("Couldn't find %s() in the module", orig_name);
    return err;
  } else {
    if (log)
      log->Printf("Found function %s for %s", function_name.AsCString(),
                  "$__lldb_expr");
  }

  // Retrieve an appropriate symbol context.
  SymbolContext sc;

  if (lldb::StackFrameSP frame_sp = exe_ctx.GetFrameSP()) {
    sc = frame_sp->GetSymbolContext(lldb::eSymbolContextEverything);
  } else if (lldb::TargetSP target_sp = exe_ctx.GetTargetSP()) {
    sc.target_sp = target_sp;
  }

  std::vector<std::string> features;

  // m_module is handed off here.
  m_execution_unit_sp.reset(
      new IRExecutionUnit(m_llvm_context, m_module, function_name,
                          exe_ctx.GetTargetSP(), sc, features));

  // TODO: figure out some way to work ClangExpressionDeclMap into
  //       this or do the equivalent for Swift.
  m_execution_unit_sp->GetRunnableInfo(err, func_addr, func_end);

  execution_unit_sp = m_execution_unit_sp;
  m_execution_unit_sp.reset();

  return err;
}

bool SwiftExpressionParser::RewriteExpression(
    DiagnosticManager &diagnostic_manager) {
  LLDB_SCOPED_TIMER();
  // There isn't a Swift equivalent to clang::Rewriter, so we'll just
  // use that.
  Log *log = GetLog(LLDBLog::Expressions);
  swift::SourceManager &source_manager =
      m_swift_ast_ctx.GetSourceManager();

  const DiagnosticList &diagnostics = diagnostic_manager.Diagnostics();
  size_t num_diags = diagnostics.size();
  if (num_diags == 0)
    return false;

  clang::RewriteBuffer rewrite_buf;
  llvm::StringRef text_ref(m_expr.Text());
  rewrite_buf.Initialize(text_ref);

  for (const auto &diag : diagnostic_manager.Diagnostics()) {
    const auto *diagnostic = llvm::dyn_cast<SwiftDiagnostic>(diag.get());
    if (!(diagnostic && diagnostic->HasFixIts()))
      continue;

    const SwiftDiagnostic::FixItList &fixits = diagnostic->FixIts();
    std::vector<swift::CharSourceRange> source_ranges;
    for (const swift::DiagnosticInfo::FixIt &fixit : fixits) {
      const swift::CharSourceRange &range = fixit.getRange();
      swift::SourceLoc start_loc = range.getStart();
      if (!start_loc.isValid()) {
        // getLocOffsetInBuffer will assert if you pass it an invalid
        // location, so we have to check that first.
        if (log)
          log->Printf(
              "SwiftExpressionParser::RewriteExpression: ignoring fixit since "
              "it contains an invalid source location: %s.",
              range.str().str().c_str());
        return false;
      }

      // ReplaceText can't handle replacing the same source range more
      // than once, so we have to check that before we proceed:
      if (std::find(source_ranges.begin(), source_ranges.end(), range) !=
          source_ranges.end()) {
        if (log)
          log->Printf(
              "SwiftExpressionParser::RewriteExpression: ignoring fix-it since "
              "source range appears twice: %s.\n",
              range.str().str().c_str());
        return false;
      } else
        source_ranges.push_back(range);

      // ReplaceText will either assert or crash if the start_loc
      // isn't inside the buffer it is said to reside in.  That
      // shouldn't happen, but it doesn't hurt to check before we call
      // ReplaceText.
      auto *Buffer = source_manager.getLLVMSourceMgr().getMemoryBuffer(
          diagnostic->GetBufferID());
      if (!(start_loc.getOpaquePointerValue() >= Buffer->getBuffer().begin() &&
            start_loc.getOpaquePointerValue() <= Buffer->getBuffer().end())) {
        if (log)
          log->Printf(
              "SwiftExpressionParser::RewriteExpression: ignoring fixit since "
              "it contains a source location not in the specified buffer: %s.",
              range.str().str().c_str());
      }

      unsigned offset = source_manager.getLocOffsetInBuffer(
          range.getStart(), diagnostic->GetBufferID());
      rewrite_buf.ReplaceText(offset, range.getByteLength(), fixit.getText());
    }
  }

  std::string fixed_expression;
  llvm::raw_string_ostream out_stream(fixed_expression);

  rewrite_buf.write(out_stream);
  out_stream.flush();
  diagnostic_manager.SetFixedExpression(fixed_expression);

  return true;
}
