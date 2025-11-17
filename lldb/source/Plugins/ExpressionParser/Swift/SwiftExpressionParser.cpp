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

#include "Plugins/TypeSystem/Swift/SwiftASTContext.h"
#include "Plugins/Language/Swift/LogChannelSwift.h"
#include "SwiftASTManipulator.h"
#include "SwiftDiagnostic.h"
#include "SwiftExpressionSourceCode.h"
#include "SwiftExpressionVariable.h"
#include "SwiftPersistentExpressionState.h"
#include "SwiftREPLMaterializer.h"
#include "SwiftSILManipulator.h"
#include "SwiftUserExpression.h"

#include "Plugins/LanguageRuntime/Swift/SwiftLanguageRuntime.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/ModuleSpec.h"
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
#include "lldb/ValueObject/ValueObject.h"

#include "llvm-c/Analysis.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/RewriteBuffer.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"

#include "clang/Basic/Module.h"

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
using namespace lldb;
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
  assert(expr.Language().name == llvm::dwarf::DW_LNAME_Swift);

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

static CompilerDecl GetCompilerDecl(swift::Decl *decl) {
  return {SwiftASTContext::GetSwiftASTContext(&decl->getASTContext()), decl};
}

LLDBNameLookup::LLDBNameLookup(
    swift::SourceFile &source_file,
    SwiftExpressionParser::SILVariableMap &variable_map, SymbolContext &sc,
    ExecutionContextScope &exe_scope)
    : SILDebuggerClient(source_file.getASTContext()),
      m_source_file(source_file), m_variable_map(variable_map), m_sc(sc) {
  source_file.getParentModule()->setDebugClient(this);

  if (!m_sc.target_sp)
    return;
  m_persistent_vars = llvm::cast<SwiftPersistentExpressionState>(
      m_sc.target_sp->GetPersistentExpressionStateForLanguage(
          lldb::eLanguageTypeSwift));
}

swift::SILValue LLDBNameLookup::emitLValueForVariable(
    swift::VarDecl *var, swift::SILBuilder &builder) {
  SwiftSILManipulator manipulator(builder);

  swift::Identifier variable_name = var->getName();
  ConstString variable_const_string(variable_name.get());

  SwiftExpressionParser::SILVariableMap::iterator vi =
      m_variable_map.find(variable_const_string.AsCString());

  if (vi == m_variable_map.end())
    return swift::SILValue();

  return manipulator.emitLValueForVariable(var, vi->second);
}

SwiftPersistentExpressionState::SwiftDeclMap &
LLDBNameLookup::GetStagedDecls() {
  return m_staged_decls;
}

void LLDBNameLookup::RegisterTypeAliases(
    llvm::SmallVectorImpl<swift::TypeAliasDecl *> &type_aliases) {
  m_type_aliases.append(type_aliases.begin(), type_aliases.end());
}

/// A name lookup class for debugger expr mode.
class LLDBExprNameLookup : public LLDBNameLookup {
  const SwiftASTContextForExpressions *m_this_context;

public:
  LLDBExprNameLookup(swift::SourceFile &source_file,
                     SwiftExpressionParser::SILVariableMap &variable_map,
                     SymbolContext &sc, ExecutionContextScope &exe_scope,
                     const SwiftASTContextForExpressions *this_context)
      : LLDBNameLookup(source_file, variable_map, sc, exe_scope),
        m_this_context(this_context) {}

  bool shouldGlobalize(swift::Identifier Name, swift::DeclKind Kind) override {
    // Extensions have to be globalized, there's no way to mark them
    // as local to the function, since their name is the name of the
    // thing being extended...
    if (Kind == swift::DeclKind::Extension)
      return true;

    // Operators need to be parsed at the global scope regardless of name.
    if (Kind == swift::DeclKind::Func && Name.isOperator())
      return true;

    if (Name.str().starts_with("$")) {
      LLDB_LOG(GetLog(LLDBLog::Expressions),
               "[LLDBExprNameLookup::shouldGlobalize] Returning true to "
               "globalizing {0}",
               Name.str());
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
        m_staged_decls.AddDecl(GetCompilerDecl(value_decl), false, {});
    }
  }

  bool lookupOverrides(swift::DeclBaseName Name, swift::DeclContext *DC,
                       swift::SourceLoc Loc, bool IsTypeLookup,
                       ResultVector &RV) override {
    static unsigned counter = 0;
    unsigned count = counter++;

    LLDB_LOG(GetLog(LLDBLog::Expressions),
             "[LLDBExprNameLookup::lookupOverrides({0})] Searching for \"{1}\"",
             count, Name.getIdentifier().get());

    return false;
  }

  bool lookupAdditions(swift::DeclBaseName Name, swift::DeclContext *DC,
                       swift::SourceLoc Loc, bool IsTypeLookup,
                       ResultVector &RV) override {
    assert(SwiftASTContext::GetSwiftASTContext(&DC->getASTContext()) ==
           m_this_context);
    static unsigned counter = 0;
    unsigned count = counter++;

    StringRef NameStr = Name.getIdentifier().str();
    if (NameStr.empty())
      return false;

    LLDB_LOG(
        GetLog(LLDBLog::Expressions),
        "[LLDBExprNameLookup::lookupAdditions ({0})] Searching for \"{1}\"",
        count, NameStr);

    std::vector<CompilerDecl> results;

    for (auto *alias : m_type_aliases) {
      if (alias->getName().str() == NameStr) {
        results.push_back(GetCompilerDecl(alias));
        break;
      }
    }

    // First look up the matching decls we've made in this compile.
    // Later, when we look for persistent decls, these staged decls
    // take precedence.

    m_staged_decls.FindMatchingDecls(NameStr, {}, results);

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
      bool is_debugger_variable = NameStr.front() == '$';

      size_t num_external_results = RV.size();
      if (!is_debugger_variable && num_external_results > 0) {
        std::vector<CompilerDecl> persistent_results;
        m_persistent_vars->GetSwiftPersistentDecls(NameStr, {},
                                                   persistent_results);

        for (CompilerDecl & decl : persistent_results) {
          if (decl.GetTypeSystem() != m_this_context) {
            LLDB_LOG(GetLog(LLDBLog::Expressions),
                     "Ignoring persistent result from other context: {0}",
                     NameStr);
            continue;
          }
          auto *value_decl =
              static_cast<swift::ValueDecl *>(decl.GetOpaqueDecl());
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
            results.push_back(decl);
        }
      } else {
        m_persistent_vars->GetSwiftPersistentDecls(NameStr, results, results);
      }
    }

    for (CompilerDecl &decl : results) {
      auto *value_decl = static_cast<swift::ValueDecl *>(decl.GetOpaqueDecl());
      // No import required.
      assert(&DC->getASTContext() == &value_decl->getASTContext());
      RV.push_back(swift::LookupResultEntry(value_decl));
    }

    return results.size() > 0;
  }

  swift::Identifier getPreferredPrivateDiscriminator() override {
    if (m_sc.comp_unit) {
      if (lldb_private::Module *module = m_sc.module_sp.get()) {
        if (lldb_private::SymbolFile *symbol_file = module->GetSymbolFile()) {
          std::string private_discriminator_string;
          if (symbol_file->GetCompileOption("-private-discriminator",
                                            private_discriminator_string,
                                            m_sc.comp_unit) &&
              !private_discriminator_string.empty()) {
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
  const SwiftASTContextForExpressions *m_this_context;

public:
  LLDBREPLNameLookup(swift::SourceFile &source_file,
                     SwiftExpressionParser::SILVariableMap &variable_map,
                     SymbolContext &sc, ExecutionContextScope &exe_scope,
                     const SwiftASTContextForExpressions *this_context)
      : LLDBNameLookup(source_file, variable_map, sc, exe_scope),
        m_this_context(this_context) {}

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
    assert(SwiftASTContext::GetSwiftASTContext(&DC->getASTContext()) ==
           m_this_context);
    static unsigned counter = 0;
    unsigned count = counter++;

    StringRef NameStr = Name.getIdentifier().str();
    if (NameStr.empty())
      return false;

    LLDB_LOG(
        GetLog(LLDBLog::Expressions),
        "[LLDBREPLNameLookup::lookupAdditions ({0})] Searching for \"{1}\"",
        count, NameStr);

    // Find decls that come from the current compilation.
    std::vector<CompilerDecl> current_compilation_results;
    for (auto result : RV) {
      auto result_decl = result.getValueDecl();
      auto result_decl_context = result_decl->getDeclContext();
      if (result_decl_context->isChildContextOf(DC) || result_decl_context == DC)
        current_compilation_results.push_back(GetCompilerDecl(result_decl));
    }

    // Find persistent decls, excluding decls that are equivalent to
    // decls from the current compilation.  This makes the decls from
    // the current compilation take precedence.
    std::vector<CompilerDecl> persistent_decl_results;
    m_persistent_vars->GetSwiftPersistentDecls(
        NameStr, current_compilation_results, persistent_decl_results);

    // Append the persistent decls that we found to the result vector.
    for (auto result : persistent_decl_results) {
      if (result.GetTypeSystem() != m_this_context) {
        LLDB_LOG(GetLog(LLDBLog::Expressions),
                 "Ignoring persistent result from other context: {0}", NameStr);
        continue;
      }

      // No import required.
      auto *result_decl =
          static_cast<swift::ValueDecl *>(result.GetOpaqueDecl());
      assert(&DC->getASTContext() == &result_decl->getASTContext());
      RV.push_back(swift::LookupResultEntry(result_decl));
    }

    return !persistent_decl_results.empty();
  }

  swift::Identifier getPreferredPrivateDiscriminator() override {
    return swift::Identifier();
  }
};

/// Returns the Swift type for a ValueObject representing a variable.
/// An invalid CompilerType is returned on error.
static CompilerType GetSwiftTypeForVariableValueObject(
    lldb::ValueObjectSP valobj_sp, lldb::StackFrameSP &stack_frame_sp,
    SwiftLanguageRuntime *runtime, lldb::BindGenericTypes bind_generic_types) {
  // Check that the passed ValueObject is valid.
  if (!valobj_sp || valobj_sp->GetError().Fail())
    return {};
  CompilerType result = valobj_sp->GetCompilerType();
  if (!result)
    return {};
  if (SwiftASTManipulator::ShouldBindGenericTypes(bind_generic_types))
    result = llvm::expectedToOptional(
                 runtime->BindGenericTypeParameters(*stack_frame_sp, result))
                 .value_or(CompilerType());
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
  lldb::ValueObjectSP valobj_sp =
      stack_frame_sp->GetValueObjectForFrameVariable(variable_sp,
                                                     lldb::eNoDynamicValues);
  const bool use_dynamic_value = use_dynamic > lldb::eNoDynamicValues;

  CompilerType var_type = GetSwiftTypeForVariableValueObject(
      valobj_sp, stack_frame_sp, runtime, bind_generic_types);

  if (!var_type.IsValid())
    return {};

  auto swift_type_system =
      var_type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwift>();
  if (!swift_type_system)
    return {};

  // If the type can't be realized and dynamic types are allowed, fall back to
  // the dynamic type. We can only do this when not binding generic types
  // though, as we don't bind the generic parameters in that case.
  if (swift_type_system->IsMeaninglessWithoutDynamicResolution(
          var_type.GetOpaqueQualType()) &&
      SwiftASTManipulator::ShouldBindGenericTypes(bind_generic_types) &&
      use_dynamic_value) {
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

/// Adds the type aliases the type-checker needs to type-check the expression.
///
/// - Returns: A `Status` instance that indicates whether the method finished
/// successfully. If the method returns an error status, it contains a string
/// that explain the failure.
static llvm::Error
AddRequiredAliases(Block *block, lldb::StackFrameSP &stack_frame_sp,
                   SwiftASTContextForExpressions &swift_ast_context,
                   SwiftASTManipulator &manipulator,
                   lldb::DynamicValueType use_dynamic,
                   lldb::BindGenericTypes bind_generic_types) {
  // Alias builtin types, since we can't use them directly in source code.
  auto builtin_ptr_t = swift_ast_context.GetBuiltinRawPointerType();
  auto alias = manipulator.MakeTypealias(
      swift_ast_context.GetASTContext()->getIdentifier("$__lldb_builtin_ptr_t"),
      builtin_ptr_t, false);
  if (!alias)
    return alias.takeError();
  auto builtin_int_t = swift_ast_context.GetBuiltinIntType();
  alias = manipulator.MakeTypealias(
      swift_ast_context.GetASTContext()->getIdentifier("$__lldb_builtin_int_t"),
      builtin_int_t, false);
  if (!alias)
    return alias.takeError();

  // First emit the typealias for "$__lldb_context".
  lldb::VariableSP self_var_sp = SwiftExpressionParser::FindSelfVariable(block);

  // If there is no self we don't need to add the "$__lldb_context" alias.
  if (!self_var_sp)
    return llvm::Error::success();

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
    return llvm::createStringError(
        "Unable to add the aliases the expression needs because "
        "self isn't valid.");

  // Import before getting the unbound version, because the unbound
  // version may not be in the mangled name map.

  CompilerType imported_self_type = ImportType(swift_ast_context, self_type);

  if (!imported_self_type.IsValid())
    return llvm::createStringError(
        "Unable to add the aliases the expression needs because the "
        "self type from an import isn't valid.");

  auto *stack_frame = stack_frame_sp.get();
  if (SwiftASTManipulator::ShouldBindGenericTypes(bind_generic_types)) {
    auto bound_type_or_err = swift_runtime->BindGenericTypeParameters(
        *stack_frame, imported_self_type);
    if (!bound_type_or_err)
      return llvm::joinErrors(
          llvm::createStringError(
              "Unable to add the aliases the expression needs because the "
              "Swift expression parser couldn't bind the type parameters for "
              "self."),
          bound_type_or_err.takeError());

    imported_self_type = *bound_type_or_err;
  }

  {
    auto swift_type_system =
        imported_self_type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwift>();
    if (!swift_type_system)
      return llvm::createStringError(
          "Unable to add the aliases the expression needs because "
          "self is not a Swift type.");

    // This might be a referenced type, in which case we really want to
    // extend the referent:
    imported_self_type = swift_type_system->GetReferentType(
        imported_self_type.GetOpaqueQualType());
    if (!imported_self_type)
      return llvm::createStringError(
          "Unable to add the aliases the expression needs because "
          "the Swift expression parser couldn't get the referent "
          "type for self.");
  }

  {
    auto swift_type_system =
        imported_self_type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwift>();
    if (!swift_type_system)
      return llvm::createStringError(
          "Unable to add the aliases the expression needs because "
          "self is not a Swift type.");

    // If we are extending a generic class it's going to be a metatype,
    // and we have to grab the instance type:
    imported_self_type = swift_type_system->GetInstanceType(
        imported_self_type.GetOpaqueQualType(), stack_frame_sp.get());
    if (!imported_self_type)
      return llvm::createStringError(
          "Unable to add the aliases the expression needs because the Swift "
          "expression parser couldn't get the instance type for self.");
  }

  Flags imported_self_type_flags(imported_self_type.GetTypeInfo());

  auto swift_self_type = swift_ast_context.GetSwiftType(imported_self_type);
  if (!swift_self_type) {
    LLDB_LOG(GetLog(LLDBLog::Types | LLDBLog::Expressions),
             "Couldn't get SwiftASTContext type for self type {0}.",
             imported_self_type.GetDisplayTypeName());
    return llvm::createStringError(
        "Unable to add the aliases the expression needs because the Swift "
        "expression parser couldn't get the Swift type for self.");
  }
  if (!swift_self_type.get())
    return llvm::createStringError("null self type");

  swift::Type object_type = swift_self_type.get()->getWithoutSpecifierType();

  if (object_type.getPointer() &&
      (object_type.getPointer() != imported_self_type.GetOpaqueQualType()))
    imported_self_type = ToCompilerType(object_type.getPointer());

  // If 'self' is a weak storage type, it must be an optional.  Look
  // through it and unpack the argument of "optional".
  if (swift::WeakStorageType *weak_storage_type =
          swift_self_type.get()->getAs<swift::WeakStorageType>()) {
    swift::Type referent_type = weak_storage_type->getReferentType();
    swift::BoundGenericEnumType *optional_type =
        referent_type->getAs<swift::BoundGenericEnumType>();

    if (!optional_type || optional_type->getGenericArgs().empty())
      return llvm::createStringError(
          "Unable to add the aliases the expression needs because the Swift "
          "expression parser couldn't get an optional type for self.");

    swift::Type first_arg_type = optional_type->getGenericArgs()[0];

    // In Swift only class types can be weakly captured.
    if (!first_arg_type->is<swift::ClassType>() &&
        !first_arg_type->is<swift::BoundGenericClassType>())
      return llvm::createStringError(
          "Unable to add the aliases the expression needs because "
          "weakly captured type is not a class type.");

    imported_self_type = ToCompilerType(first_arg_type);
  }

  imported_self_type_flags.Reset(imported_self_type.GetTypeInfo());
  if (imported_self_type_flags.Test(lldb::eTypeIsGenericTypeParam)) {
    LLDB_LOG(GetLog(LLDBLog::Expressions),
             "SEP:AddRequiredAliases: Failed to resolve the self archetype - "
             "could not make the $__lldb_context typealias.");
    return llvm::createStringError(
        "Unable to add the aliases the expression needs because the "
        "Swift expression parser couldn't resolve the self archetype.");
  }
  llvm::Expected<swift::ValueDecl *> type_alias_decl =
      manipulator.MakeTypealias(
          swift_ast_context.GetASTContext()->getIdentifier("$__lldb_context"),
          imported_self_type);

  if (!type_alias_decl) {
    LLDB_LOG(GetLog(LLDBLog::Expressions),
             "SEP:AddRequiredAliases: Failed to make the $__lldb_context "
             "typealias.");
    return llvm::createStringError(
        "Unable to add the aliases the expression needs because the "
        "Swift expression parser couldn't create a context type "
        "alias for lldb. " +
        llvm::toString(type_alias_decl.takeError()));
  }

  return llvm::Error::success();
}

static void ResolveSpecialNames(
    SymbolContext &sc, ExecutionContextScope &exe_scope,
    SwiftASTContextForExpressions &ast_context,
    llvm::SmallVectorImpl<swift::Identifier> &special_names,
    llvm::SmallVectorImpl<SwiftASTManipulator::VariableInfo> &local_variables) {
  Log *log = GetLog(LLDBLog::Expressions);
  if (!sc.target_sp)
    return;

  auto *persistent_state =
      sc.target_sp->GetPersistentExpressionStateForLanguage(
          lldb::eLanguageTypeSwift);

  std::set<ConstString> resolved_names;

  for (swift::Identifier &name : special_names) {
    ConstString name_cs = ConstString(name.str());

    if (resolved_names.count(name_cs))
      continue;

    resolved_names.insert(name_cs);

    LLDB_LOG(log, "Resolving special name {0}");

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

ThreadSafeASTContext
SwiftExpressionParser::GetASTContext(DiagnosticManager &diagnostic_manager) {
  llvm::call_once(m_ast_init_once_flag, [&] {
    // Lazily get the clang importer if we can to make sure it exists in
    // case we need it.
    if (!m_swift_ast_ctx.GetClangImporter()) {
      std::string swift_error =
          m_swift_ast_ctx.GetFatalErrors().AsCString("error: unknown error.");
      diagnostic_manager.PutString(eSeverityError, swift_error);
      diagnostic_manager.PutString(eSeverityInfo,
                                   "Couldn't initialize Swift expression "
                                   "evaluator due to previous errors.");
      return;
    }

    if (m_swift_ast_ctx.HasFatalErrors()) {
      diagnostic_manager.PutString(
          eSeverityError, "The AST context is in a fatal error state.");
      return;
    }

    ThreadSafeASTContext ast_context = m_swift_ast_ctx.GetASTContext();
    if (!ast_context) {
      diagnostic_manager.PutString(
          eSeverityError,
          "Couldn't initialize the AST context.  Please check your settings.");
      return;
    }

    if (m_swift_ast_ctx.HasFatalErrors()) {
      diagnostic_manager.PutString(
          eSeverityError, "The AST context is in a fatal error state.");
      return;
    }

    bool repl = m_options.GetREPLEnabled();
    bool playground = m_options.GetPlaygroundTransformEnabled();
    // TODO: Find a way to get contraint-solver output sent to a stream
    //       so we can log it.
    // swift_ast_context.GetLanguageOptions().DebugConstraintSolver = true;

    // No longer part of debugger support, set it separately.
    m_swift_ast_ctx.GetLanguageOptions().EnableDollarIdentifiers = true;
    m_swift_ast_ctx.GetLanguageOptions().EnableAccessControl =
        (repl || playground);
    LLDB_LOG(lldb_private::GetSwiftHealthLog(), "Language option EnableAccessControl = {0}",
             m_swift_ast_ctx.GetLanguageOptions().EnableAccessControl);

    m_swift_ast_ctx.GetLanguageOptions().EnableTargetOSChecking = false;

    auto should_disable_objc_runtime = [&]() {
      if (playground)
        return false;
      lldb::StackFrameSP this_frame_sp(m_stack_frame_wp.lock());
      if (!this_frame_sp)
        return false;
      lldb::ProcessSP process_sp(this_frame_sp->CalculateProcess());
      if (!process_sp)
        return false;
      return !ObjCLanguageRuntime::Get(*process_sp);
    };
    if (should_disable_objc_runtime())
      m_swift_ast_ctx.GetLanguageOptions().EnableObjCInterop = false;
    LLDB_LOG(lldb_private::GetSwiftHealthLog(), "Language option EnableObjCInterop = {0}",
             m_swift_ast_ctx.GetLanguageOptions().EnableObjCInterop);

    m_swift_ast_ctx.GetLanguageOptions().Playground = repl || playground;
    LLDB_LOG(lldb_private::GetSwiftHealthLog(), "Language option Playground = {0}",
             m_swift_ast_ctx.GetLanguageOptions().Playground);

    m_swift_ast_ctx.GetIRGenOptions().Playground = repl || playground;

    // For the expression parser and REPL we want to relax the
    // requirement that you put "try" in front of every expression that
    // might throw.
    if (repl || !playground)
      m_swift_ast_ctx.GetLanguageOptions().EnableThrowWithoutTry = true;
    LLDB_LOG(lldb_private::GetSwiftHealthLog(), "Language option EnableThrowWithoutTry = {0}",
             m_swift_ast_ctx.GetLanguageOptions().EnableThrowWithoutTry);


    m_swift_ast_ctx.GetIRGenOptions().OutputKind =
        swift::IRGenOutputKind::Module;
    m_swift_ast_ctx.GetIRGenOptions().OptMode =
        swift::OptimizationMode::NoOptimization;
    // Normally we'd like to verify, but unfortunately the verifier's
    // error mode is abort().
    m_swift_ast_ctx.GetIRGenOptions().Verify = false;
    m_swift_ast_ctx.GetIRGenOptions().ForcePublicLinkage = true;

    m_swift_ast_ctx.GetIRGenOptions().DisableRoundTripDebugTypes = true;
    m_ast_init_successful = true;
  });
  if (m_ast_init_successful)
    return m_swift_ast_ctx.GetASTContext();
  return ThreadSafeASTContext();
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
static llvm::Expected<SwiftExpressionParser::SILVariableInfo>
MaterializeVariable(SwiftASTManipulatorBase::VariableInfo &variable,
                    SwiftUserExpression &user_expression,
                    Materializer &materializer,
                    SwiftASTManipulator &manipulator,
                    lldb::StackFrameWP &stack_frame_wp,
                    DiagnosticManager &diagnostic_manager, Log *log,
                    bool repl) {
  uint64_t offset = 0;
  bool needs_init = false;

  bool is_result = llvm::isa<SwiftASTManipulatorBase::VariableMetadataResult>(
      variable.GetMetadata());
  bool is_error = llvm::isa<SwiftASTManipulatorBase::VariableMetadataError>(
      variable.GetMetadata());

  auto compiler_type = variable.GetType();
  // Add the persistent variable as a typeref compiler type.
  if (auto ts =
          compiler_type.GetTypeSystem().dyn_cast_or_null<SwiftASTContext>()) {
    // Add the persistent variable as a typeref compiler type, but only if
    // doesn't have archetypes (which can be the case when we're evaluating an
    // expression as generic), since we can't mangle free-standing archetypes.
    if (!manipulator.GetScratchContext().TypeHasArchetype(compiler_type))
      variable.SetType(ts->GetTypeRefType(compiler_type.GetOpaqueQualType()));
  }

  if (is_result || is_error) {
    needs_init = true;

    Status error;
    if (repl) {
      if (!variable.GetType().IsVoidType()) {
        auto &repl_mat = *llvm::cast<SwiftREPLMaterializer>(&materializer);
        assert(variable.GetType()
                   .GetTypeSystem()
                   .isa_and_nonnull<TypeSystemSwiftTypeRef>());
        offset = repl_mat.AddREPLResultVariable(
            variable.GetType(), variable.GetDecl(),
            is_result ? &user_expression.GetResultDelegate()
                      : &user_expression.GetErrorDelegate(),
            error);
      }
    } else {
      CompilerType actual_type = variable.GetType();
      // Desugar '$lldb_context', etc.
      llvm::Expected<swift::Type> actual_swift_type =
          manipulator.GetScratchContext().GetSwiftType(actual_type);
      if (!actual_swift_type)
        return actual_swift_type.takeError();
      if (!actual_swift_type.get())
        return make_error<StringError>(inconvertibleErrorCode(),
                                       "actual_swift_type is a nullptr");

      auto transformed_type =
          actual_swift_type->transformRec([](swift::TypeBase *t)
              -> std::optional<swift::Type> {
            if (auto *aliasTy = swift::dyn_cast<swift::TypeAliasType>(t)) {
              if (aliasTy->getDecl()->isDebuggerAlias()) {
                return aliasTy->getSinglyDesugaredType();
              }
            }
            return std::nullopt;
          });

      if (!transformed_type)
        return make_error<StringError>(inconvertibleErrorCode(),
                                       "transformed type is empty");

      actual_type =
          ToCompilerType(transformed_type->mapTypeOutOfEnvironment().getPointer());
      auto swift_ast_ctx =
          actual_type.GetTypeSystem().dyn_cast_or_null<SwiftASTContext>();
      if (!swift_ast_ctx)
        return make_error<StringError>(inconvertibleErrorCode(),
                                       "no Swift AST context");

      actual_type =
          swift_ast_ctx->GetTypeRefType(actual_type.GetOpaqueQualType());

      offset = materializer.AddResultVariable(
          actual_type, false, true,
          is_result ? &user_expression.GetResultDelegate()
                    : &user_expression.GetErrorDelegate(),
          error);
    }

    if (!error.Success())
      return llvm::createStringError(
          inconvertibleErrorCode(), "couldn't add %s variable to struct: %s.",
          is_result ? "result" : "error", error.AsCString());

    LLDB_LOG(log, "Added {0} variable to struct at offset {1}",
             is_result ? "result" : "error", (unsigned long long)offset);
  } else if (auto *variable_metadata = llvm::dyn_cast<
                 SwiftASTManipulatorBase::VariableMetadataVariable>(
                 variable.GetMetadata())) {
    Status error;

    offset = materializer.AddVariable(variable_metadata->m_variable_sp, error);

    if (!error.Success())
      return llvm::createStringError(inconvertibleErrorCode(),
                                     "couldn't add variable to struct: %s.\n",
                                     error.AsCString());

    LLDB_LOG(log, "Added variable {0} to struct at offset {1}",
             variable_metadata->m_variable_sp->GetName(),
             (unsigned long long)offset);
  } else if (auto *variable_metadata = llvm::dyn_cast<
                 SwiftASTManipulatorBase::VariableMetadataPersistent>(
                 variable.GetMetadata())) {
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
    auto size_or_err =
        variable.GetType().GetByteSize(stack_frame_sp.get());
    if (!size_or_err)
      return size_or_err.takeError();
    uint64_t size = *size_or_err;
    if (repl && size == 0) {
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

    if (!error.Success())
      return llvm::createStringError(inconvertibleErrorCode(),
                                     "couldn't add variable to struct: %s.\n",
                                     error.AsCString());

    LLDB_LOGF(
        log,
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
    OS << msg;
  }
  std::error_code convertToErrorCode() const override {
    return inconvertibleErrorCode();
  }
};

char SwiftASTContextError::ID = 0;
char ModuleImportError::ID = 0;
} // namespace

/// Adds typealiases from the archetypes as they appear in the source code to
/// their bound  type. For example, in the following snippet, when stopped
/// inside f, a typealias is added from T to Int:
/// @code
/// func f<T>(t: T) { ... }
/// f<Int>(t: 5)
/// \endcode
/// \return The vector of newly created typealiases.
static llvm::Expected<llvm::SmallVector<swift::TypeAliasDecl *>>
AddArchetypeTypeAliases(std::unique_ptr<SwiftASTManipulator> &code_manipulator,
                         StackFrame &stack_frame,
                         SwiftASTContextForExpressions &swift_ast_context) {
  Log *log = GetLog(LLDBLog::Expressions);
  llvm::SmallVector<swift::TypeAliasDecl *> type_aliases;
  lldb::ProcessSP process_sp(stack_frame.CalculateProcess());
  if (!process_sp)
    return llvm::createStringError("no process");
  auto *runtime = SwiftLanguageRuntime::Get(process_sp);
  if (!runtime)
    return llvm::createStringError("no runtime");

  auto typeref_typesystem = swift_ast_context.GetTypeSystemSwiftTypeRef();
  if (!typeref_typesystem)
    return llvm::createStringError("no typeref typesystem");

  // Skip this for variadic generic functions.
  ConstString func_name =
      stack_frame.GetSymbolContext(lldb::eSymbolContextFunction)
          .GetFunctionName(Mangled::ePreferMangled);
  if (auto signature = SwiftLanguageRuntime::GetGenericSignature(
          func_name.GetStringRef(), *typeref_typesystem))
    if (signature->pack_expansions.size())
      return llvm::createStringError("[AddArchetypeTypeAliases] Variadic "
                                     "generic functions are not supported.");

  struct MetadataPointerInfo {
    unsigned int depth;
    unsigned int index;
    lldb_private::SwiftASTManipulator::VariableInfo &variable;
  };

  // Collect the metadata pointers that should be typealiased.
  llvm::SmallDenseMap<llvm::StringRef, MetadataPointerInfo>
      visible_metadata_pointers;
  for (auto &variable : code_manipulator->GetVariableInfo()) {
    if (!variable.IsMetadataPointer())
      continue;

    llvm::StringRef type_name;
    if (auto *variable_metadata =
            llvm::dyn_cast<SwiftASTManipulatorBase::VariableMetadataVariable>(
                variable.GetMetadata())) {
      type_name =
          variable_metadata->m_variable_sp->GetType()->GetName().GetStringRef();
    }

    if (type_name.empty())
      continue;

    auto di = ParseSwiftGenericParameter(variable.GetName().str());
    if (!di)
      continue;
    unsigned depth = di->first;
    unsigned index = di->second;

    auto it = visible_metadata_pointers.find(type_name);
    if (it != visible_metadata_pointers.end()) {
      auto &info = it->getSecond();
      // The current metadata pointer shadows the one we registered, replace it.
      if (info.depth < depth) {
        info.depth = depth;
        info.index = index;
        info.variable = variable;
      }
    } else {
      visible_metadata_pointers.insert({type_name, {depth, index, variable}});
    }
  }

  // Create a typealias from name -> type for each visible metadata pointer.
  for (auto &pair : visible_metadata_pointers) {
    llvm::StringRef &type_name = pair.getFirst();
    MetadataPointerInfo &info = pair.getSecond();

    auto flavor = SwiftLanguageRuntime::GetManglingFlavor(type_name);
    auto dependent_type = typeref_typesystem->CreateGenericTypeParamType(
        info.depth, info.index, flavor);
    auto bound_type_or_err =
        runtime->BindGenericTypeParameters(stack_frame, dependent_type);
    if (!bound_type_or_err) {
      LLDB_LOG_ERROR(
          log, bound_type_or_err.takeError(),
          "[AddArchetypeTypeAliases] Could not bind dependent generic param "
          "type {1}: {0}",
          dependent_type.GetMangledTypeName());
      continue;
    }
    auto bound_type = *bound_type_or_err;

    LLDB_LOG(log,
             "[AddArchetypeTypeAliases] Binding dependent generic param "
             "type {0} to {1}",
             dependent_type.GetMangledTypeName(),
             bound_type.GetMangledTypeName());
    auto identifier =
        swift_ast_context.GetASTContext()->getIdentifier(type_name);

    // The following code:
    // class A<T> {
    //   func f<T>(t: T) {
    //     print(1) // break here
    //   }
    // }
    // A<Int>.f<String>(t: "Hello")
    //
    // Will generate the following extension when evaluating the expression:
    // extension A where T == Int { ... }
    // Which is why we need to make the typealias inside the function with the
    // user's code, as it needs to shadow the generic type requirement.
    auto type_alias_decl = code_manipulator->MakeTypealias(
        identifier, bound_type, true, code_manipulator->GetFuncDecl());
    if (type_alias_decl) {
      type_aliases.push_back(*type_alias_decl);
      LLDB_LOG(log,
               "[AddArchetypeTypeAliases] Adding typealias from {0} to "
               "{1}",
               type_name, bound_type.GetMangledTypeName());
    } else
      return type_alias_decl.takeError();
  }
  return type_aliases;
}

llvm::Expected<SwiftExpressionParser::ParsedExpression>
SwiftExpressionParser::ParseAndImport(
    SwiftASTContext::ScopedDiagnostics &expr_diagnostics,
    SwiftExpressionParser::SILVariableMap &variable_map, unsigned &buffer_id,
    DiagnosticManager &diagnostic_manager) {

  Log *log = GetLog(LLDBLog::Expressions);
  bool repl = m_options.GetREPLEnabled();
  bool playground = m_options.GetPlaygroundTransformEnabled();

  // Install a progress meter.
  auto progress_raii = m_swift_ast_ctx.GetModuleImportProgressRAII(
      "Importing modules used in expression");

  // If we are using the playground, hand import the necessary
  // modules.
  //
  // FIXME: We won't have to do this once the playground adds import
  //        statements for the things it needs itself.
  if (playground) {
    auto module_or_err = m_swift_ast_ctx.ImportStdlib();

    if (!module_or_err) {
      LLDB_LOG(log, "couldn't load Swift Standard Library");
      return module_or_err.takeError();
    }

    m_swift_ast_ctx.AddHandLoadedModule(ConstString("Swift"),
                                        swift::ImportedModule(&*module_or_err));
  }

  std::string main_filename;
  std::tie(buffer_id, main_filename) = CreateMainFile(
      m_swift_ast_ctx, repl ? "<REPL>" : "<EXPR>", m_expr.Text(), m_options);

  char expr_name_buf[32];

  snprintf(expr_name_buf, sizeof(expr_name_buf), "__lldb_expr_%u",
           m_options.GetExpressionNumber());

  // Gather the modules that need to be implicitly imported.
  // The Swift stdlib needs to be imported before the SwiftLanguageRuntime can
  // be used.
  Status implicit_import_error;
  llvm::SmallVector<swift::AttributedImport<swift::ImportedModule>, 16>
      additional_imports;
  lldb::ProcessSP process_sp;
  if (lldb::StackFrameSP this_frame_sp = m_stack_frame_wp.lock())
    process_sp = this_frame_sp->CalculateProcess();
  m_swift_ast_ctx.LoadImplicitModules(m_sc.target_sp, process_sp, *m_exe_scope);
  if (!m_swift_ast_ctx.GetImplicitImports(m_sc, process_sp, additional_imports,
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

  swift::ModuleDecl *module = nullptr;
  swift::SourceFile *source_file = nullptr;
  {
    ThreadSafeASTContext ast_context = GetASTContext(diagnostic_manager);
    if (!ast_context)
      return make_error<SwiftASTContextError>();

    auto module_id = ast_context->getIdentifier(expr_name_buf);
    module = swift::ModuleDecl::create(
        module_id, **ast_context, importInfo,
        [&](swift::ModuleDecl *module, auto addFile) {
      swift::SourceFileKind source_file_kind = swift::SourceFileKind::Library;
      if (playground || repl) {
        source_file_kind = swift::SourceFileKind::Main;
      }

      // Create the source file. Note, we disable delayed parsing for the
      // swift expression parser.
      source_file = new (**ast_context) swift::SourceFile(
          *module, source_file_kind, buffer_id,
          swift::SourceFile::ParsingFlags::DisableDelayedBodies);
      addFile(source_file);
    });
  }
  // Swift Modules that rely on shared libraries (not frameworks)
  // don't record the link information in the swiftmodule file, so we
  // can't really make them work without outside information.
  // However, in the REPL you can added -L & -l options to the initial
  // compiler startup, and we should dlopen anything that's been
  // stuffed on there and hope it will be useful later on.
  if (repl) {
    lldb::StackFrameSP this_frame_sp(m_stack_frame_wp.lock());

    if (this_frame_sp) {
      lldb::ProcessSP process_sp(this_frame_sp->CalculateProcess());
      if (process_sp) {
        Status error;
        m_swift_ast_ctx.LoadExtraDylibs(*process_sp.get(), error);
      }
    }
  }

  auto &invocation = m_swift_ast_ctx.GetCompilerInvocation();
  invocation.getFrontendOptions().ModuleName = expr_name_buf;
  invocation.getIRGenOptions().ModuleName = expr_name_buf;

  auto &lang_opts = invocation.getLangOptions();
  bool enable_bare_slash_regex_literals =
      m_sc.target_sp->GetSwiftEnableBareSlashRegex();
  if (enable_bare_slash_regex_literals) {
    lang_opts.enableFeature(swift::Feature::BareSlashRegexLiterals);
  }
  if (uint32_t version = m_expr.Language().version)
    lang_opts.EffectiveLanguageVersion =
        llvm::VersionTuple(version / 100, version % 100);
  if (lang_opts.EffectiveLanguageVersion >= swift::version::Version({6}))
    lang_opts.StrictConcurrencyLevel = swift::StrictConcurrency::Complete;

  auto should_use_prestable_abi = [&]() {
    lldb::StackFrameSP this_frame_sp(m_stack_frame_wp.lock());
    if (!this_frame_sp)
      return false;
    lldb::ProcessSP process_sp(this_frame_sp->CalculateProcess());
    if (!process_sp)
      return false;
    auto *runtime = SwiftLanguageRuntime::Get(process_sp);
    return !runtime->IsABIStable();
  };

  lang_opts.UseDarwinPreStableABIBit = should_use_prestable_abi();

  LLDBNameLookup *external_lookup;
  if (m_options.GetPlaygroundTransformEnabled() || m_options.GetREPLEnabled()) {
    external_lookup = new LLDBREPLNameLookup(*source_file, variable_map, m_sc,
                                             *m_exe_scope, &m_swift_ast_ctx);
  } else {
    external_lookup = new LLDBExprNameLookup(*source_file, variable_map, m_sc,
                                             *m_exe_scope, &m_swift_ast_ctx);
  }

  // FIXME: This call is here just so that the we keep the
  //        DebuggerClients alive as long as the Module we are not
  //        inserting them in.
  m_swift_ast_ctx.AddDebuggerClient(external_lookup);

  if (expr_diagnostics.HasErrors())
    return make_error<SwiftASTContextError>();

  // Resolve the file's imports, including the implicit ones returned from
  // GetImplicitImports.
  swift::performImportResolution(*source_file);

  if (expr_diagnostics.HasErrors()) {
    // FIXME: This could be done more elegantly.
    std::string msg = llvm::toString(expr_diagnostics.GetAllErrors());
    if (StringRef(msg).contains(": could not build module "))
      return make_error<ModuleImportError>(msg);
    return expr_diagnostics.GetAsExpressionError(lldb::eExpressionParseError);
  }

  std::unique_ptr<SwiftASTManipulator> code_manipulator;
  if (repl || !playground) {
    code_manipulator = std::make_unique<SwiftASTManipulator>(
        m_swift_ast_ctx, *source_file, m_sc, repl,
        m_options.GetBindGenericTypes());

    if (!playground) {
      code_manipulator->RewriteResult();
    }
  }

  if (!playground && !repl) {
    lldb::StackFrameSP stack_frame_sp = m_stack_frame_wp.lock();

    bool local_context_is_swift = true;

    if (m_sc.block) {
      Function *function = m_sc.block->CalculateSymbolContextFunction();
      if (function && function->GetLanguage() != lldb::eLanguageTypeSwift)
        local_context_is_swift = false;
    }

    if (local_context_is_swift) {
      llvm::Error error = AddRequiredAliases(
          m_sc.block, stack_frame_sp, m_swift_ast_ctx, *code_manipulator,
          m_options.GetUseDynamic(), m_options.GetBindGenericTypes());
      if (error)
        return error;
    }
    //
    // Register all magic variables.
    llvm::SmallVector<swift::Identifier, 2> special_names;
    llvm::StringRef persistent_var_prefix;
    if (!repl)
      persistent_var_prefix = "$";

    code_manipulator->FindSpecialNames(special_names, persistent_var_prefix);

    ResolveSpecialNames(m_sc, *m_exe_scope, m_swift_ast_ctx, special_names,
                        m_local_variables);

    code_manipulator->AddExternalVariables(m_local_variables);

    auto type_aliases = AddArchetypeTypeAliases(
        code_manipulator, *stack_frame_sp.get(), m_swift_ast_ctx);
    if (!type_aliases)
      diagnostic_manager.PutString(eSeverityWarning,
                                   llvm::toString(type_aliases.takeError()));
    else
      external_lookup->RegisterTypeAliases(*type_aliases);
    stack_frame_sp.reset();
  }

  // Cache the source file's imports such that they're accessible to future
  // expression evaluations.
  {
    std::lock_guard<std::recursive_mutex> global_context_locker(
        IRExecutionUnit::GetLLVMGlobalContextMutex());

    Status auto_import_error;
    if (!m_swift_ast_ctx.CacheUserImports(process_sp, *source_file,
                                          auto_import_error)) {
      const char *msg = auto_import_error.AsCString();
      if (!msg) {
        // The import itself succeeded, but the AST context is in a
        // fatal error state. One way this can happen is if the import
        // triggered a dylib import, in which case the context is
        // purposefully poisoned.
        msg = "import may have triggered a dylib import, "
              "resetting compiler state";
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
      std::move(code_manipulator),
      *module,
      *external_lookup,
      *source_file,
      std::move(main_filename),
      /*buffer_id*/ 0,
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
                                         SwiftASTManipulator &manipulator,
                                         swift::ASTContext &ast_ctx) {
  Log *log = GetLog(LLDBLog::Expressions);

  swift::Mangle::ASTMangler mangler(ast_ctx);
  auto *entrypoint_decl = manipulator.GetEntrypointDecl();
  if (!entrypoint_decl) {
    LLDB_LOG(log, "[RedirectCallFromSinkToTrampolineFunction] Could not set "
                  "the call: no entrypoint decl.");
    return false;
  }

  auto *func_decl = manipulator.GetFuncDecl();
  if (!func_decl) {
    LLDB_LOG(log, "[RedirectCallFromSinkToTrampolineFunction] Could not set "
                  "the call: no func decl.");
    return false;
  }

  auto *trampoline_func_decl = manipulator.GetTrampolineDecl();
  bool have_self = trampoline_func_decl;

  auto *sink_decl = manipulator.GetSinkDecl();
  if (!sink_decl) {
    LLDB_LOG(log, "[RedirectCallFromSinkToTrampolineFunction] Could not set "
                  "the call: no sink decl.");
    return false;
  }

  std::string expr_func_name = mangler.mangleEntity(entrypoint_decl);
  std::string wrapped_func_name = mangler.mangleEntity(func_decl);
  std::string trampoline_func_name;
  if (have_self)
    trampoline_func_name = mangler.mangleEntity(trampoline_func_decl);
  std::string sink_func_name = mangler.mangleEntity(sink_decl);

  llvm::Function *lldb_expr_func = module.getFunction(expr_func_name);
  llvm::Function *wrapped_func = module.getFunction(wrapped_func_name);
  llvm::Function *trampoline_func = nullptr;
  if (have_self)
    trampoline_func = module.getFunction(trampoline_func_name);
  llvm::Function *sink_func = module.getFunction(sink_func_name);
  llvm::Function *callee_func = have_self ? trampoline_func : wrapped_func;

  assert(lldb_expr_func && wrapped_func && callee_func && sink_decl);
  if (!lldb_expr_func || !wrapped_func || !callee_func || !sink_func) {
    LLDB_LOG(log,
             "[RedirectCallFromSinkToTrampolineFunction] Could not set the "
             "call: could not find one of the required functions in the IR.");
    return false;
  }

  auto *callee_func_type = callee_func->getFunctionType();
  auto callee_num_params = callee_func_type->getNumParams();
  // There should be at least 3 params, the raw pointer, the self type, and at
  // least one pointer to metadata.
  if (callee_num_params < (have_self ? 3 : 2)) {
    LLDB_LOG(log,
             "[RedirectCallFromSinkToTrampolineFunction] Could not set the "
             "call: callee function has {0} parameters",
             callee_num_params);
    return false;
  }

  auto *sink_func_type = sink_func->getFunctionType();
  auto sink_num_params = sink_func_type->getNumParams();

  if (callee_num_params != sink_num_params) {
    LLDB_LOG(
        log,
        "[RedirectCallFromSinkToTrampolineFunction] Could not set the call: "
        "callee function has {0} parameters but sink has {0} parameters.",
        callee_num_params, sink_num_params);
    return false;
  }

  // The entrypoint function should only have one basic block whith
  // materialization instructions and the call to the sink.
  if (lldb_expr_func->size() != 1) {
    LLDB_LOG(log,
             "[RedirectCallFromSinkToTrampolineFunction] Could not set the "
             "call: entrypoint function has {0} basic blocks.",
        lldb_expr_func->size());
    return false;
  }

  auto &basic_block = lldb_expr_func->back();
  if (basic_block.size() == 0) {
    LLDB_LOG(log, "[RedirectCallFromSinkToTrampolineFunction] Could not set "
                  "the call: basic block has no instructions.");
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
    LLDB_LOG(log, "[RedirectCallFromSinkToTrampolineFunction] Could not set "
                  "the call: call to sink function not found.");
    return false;
  }

  if (sink_call->arg_size() != sink_num_params) {
    LLDB_LOG(log,
             "[RedirectCallFromSinkToTrampolineFunction] Could not set the "
             "call: call to sink function has {0} arguments.",
             sink_call->arg_size());
    return false;
  }

  // The sink call should have at least three parameters, the pointer to
  // lldb_arg, a pointer to self and a pointer to the trampoline metadata of
  // self.
  llvm::Value *lldb_arg_ptr = sink_call->getArgOperand(0);
  llvm::Value *self_load = sink_call->getArgOperand(1);
  llvm::SmallVector<llvm::Value *> generic_args;
  for (size_t i = (have_self ? 2 : 1); i < sink_num_params; ++i)
    generic_args.emplace_back(sink_call->getArgOperand(i));

  // Delete the sink since we fished out the values we needed.
  sink_call->eraseFromParent();
  sink_func->eraseFromParent();

  // We need to fish the pointer to self, which the load instruction loads.
  llvm::Value *self_opaque_ptr = nullptr;
  if (auto *load = llvm::dyn_cast<llvm::LoadInst>(self_load))
    self_opaque_ptr = load->getPointerOperand();
  if (!self_opaque_ptr) {
    LLDB_LOG(
        log,
        "[RedirectCallFromSinkToTrampolineFunction] Could not set the call: "
        "could not find the argument of the load of the self pointer.");
    return false;
  }

  auto &it = basic_block.back();
  // Initialize the builder from the last instruction since we want to place the
  // new call there.
  llvm::IRBuilder<> builder(&it);

  // Bitcast the operands to the expected types, since they were type-erased
  // in the call to the sink.
  llvm::SmallVector<llvm::Value *> call_params;
  call_params.push_back(lldb_arg_ptr);
  if (have_self) {
    llvm::Type *self_type = callee_func_type->getParamType(1);
    auto *self_ptr = builder.CreateBitCast(self_opaque_ptr, self_type);
    call_params.push_back(self_ptr);
  }
  for (auto &arg : generic_args)
    call_params.push_back(
        arg->getType()->isPointerTy()
            ? builder.CreateBitCast(
                  arg, callee_func_type->getParamType(call_params.size()))
            : arg);

  // Finally, create the call.
  builder.CreateCall(callee_func_type, callee_func, call_params);
  return true;
}

static bool ContainsUB(llvm::Module &module) {
  for (llvm::Function &function : module)
    for (llvm::BasicBlock &bb : function)
      for (llvm::Instruction &i : bb)
        if (auto *memcpy = llvm::dyn_cast<llvm::MemCpyInst>(&i))
          if (llvm::isa<llvm::UndefValue>(memcpy->getOperand(0)) ||
              llvm::isa<llvm::UndefValue>(memcpy->getOperand(1)))
            return true;
  return false;
}

/// Add additional context that is otherwise hard to convey.
static void AnnotateDiagnostics(DiagnosticManager &diagnostic_manager) {
  bool append_missing_library_note = false;
  for (auto &diag : diagnostic_manager.Diagnostics()) {
    if (!diag || diag->GetSeverity() != eSeverityError)
      continue;
    const auto &detail = diag->GetDetail();
    if (StringRef(detail.message).contains("has no member")) {
      append_missing_library_note = true;
      break;
    }
  }
  if (append_missing_library_note)
    diagnostic_manager.AddDiagnostic(
        "If the member name is spelled correctly, this error could mean that "
        "it depends on a (private) type from a module that has not been "
        "imported into the current context.",
        eSeverityInfo, eDiagnosticOriginLLDB);
}

SwiftExpressionParser::ParseResult
SwiftExpressionParser::Parse(DiagnosticManager &diagnostic_manager,
                             uint32_t first_line, uint32_t last_line) {
  SwiftExpressionParser::SILVariableMap variable_map;
  using ParseResult = SwiftExpressionParser::ParseResult;
  Log *log = GetLog(LLDBLog::Expressions);
  // Get a scoped diagnostics consumer for all diagnostics produced by
  // this expression.
  auto expr_diagnostics = m_swift_ast_ctx.getScopedDiagnosticConsumer();
  m_swift_ast_ctx.GetDiagnosticEngine().resetHadAnyError();

  // The result in case parsing the expression fails. If the option is auto
  // expression evaluation should retry by binding the generic types.
  auto parse_result_failure = m_options.GetBindGenericTypes() == lldb::eBindAuto
                                  ? ParseResult::retry_bind_generic_params
                                  : ParseResult::unrecoverable_error;
  // Helper function to diagnose errors in m_swift_scratch_context.
  unsigned buffer_id = UINT32_MAX;
  auto DiagnoseSwiftASTContextError = [&]() {
    assert((expr_diagnostics->HasErrors() ||
            m_swift_ast_ctx.HasClangImporterErrors()) &&
           "error expected");
    expr_diagnostics->PrintDiagnostics(diagnostic_manager, buffer_id,
                                       first_line, last_line);
    AnnotateDiagnostics(diagnostic_manager);
  };

  // In the case of playgrounds, we turn all rewriting functionality off.
  const bool repl = m_options.GetREPLEnabled();
  const bool playground = m_options.GetPlaygroundTransformEnabled();

  if (!m_exe_scope)
    return parse_result_failure;

  // Parse the expression and import all nececssary swift modules.
  auto parsed_expr = ParseAndImport(*expr_diagnostics, variable_map, buffer_id,
                                    diagnostic_manager);

  if (!parsed_expr) {
    bool retry = false;
    handleAllErrors(
        parsed_expr.takeError(),
        [&](const ModuleImportError &MIE) {
          diagnostic_manager.PutString(eSeverityError, MIE.message());
          if (MIE.is_new_dylib) {
            retry = true;
            return;
          }
          // There are no fallback contexts in REPL and playgrounds.
          if (repl || playground)
            return;
          // The fatal error causes a new compiler to be instantiated on retry.
          m_swift_ast_ctx.RaiseFatalError(MIE.message());
        },
        [&](const SwiftASTContextError &SACE) {
          DiagnoseSwiftASTContextError();
        },
        [&](const StringError &SE) {
          diagnostic_manager.PutString(eSeverityError, SE.getMessage());
        },
        [&](const ExpressionError &E) {
          // FIXME: We're losing the DiagnosticOrigin here.
          for (auto &detail : E.GetDetails())
            diagnostic_manager.AddDiagnostic(std::make_unique<Diagnostic>(
                eDiagnosticOriginSwift, 0, detail));
        });

    // Signal that we want to retry the expression exactly once with a
    // fresh SwiftASTContext.
    if (retry) {
      if (auto *persistent_state =
              llvm::cast_or_null<SwiftPersistentExpressionState>(
                  m_sc.target_sp->GetPersistentExpressionStateForLanguage(
                      lldb::eLanguageTypeSwift)))
        persistent_state->Clear();

      return ParseResult::retry_fresh_context;
    }

    // Unrecoverable error.
    return parse_result_failure;
  }

  // If any generics are present, this expression is not parseable.
  if (parsed_expr->code_manipulator)
    m_is_cacheable =
        !llvm::any_of(parsed_expr->code_manipulator->GetVariableInfo(),
                      [](const auto &variable) {
                        return variable.IsMetadataPointer() ||
                               variable.IsPackCount() ||
                               variable.IsUnboundPack();
                      });

  auto dumpModule = [&](const char *msg) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    parsed_expr->source_file.dump(ss);
    ss.flush();
    LLDB_LOG(log, "\n{0}\n\n{1}\n", msg, s);
  };

  swift::bindExtensions(parsed_expr->module);

  if (log)
    dumpModule("Module before type checking:");

  swift::performTypeChecking(parsed_expr->source_file);

  if (log)
    dumpModule("Module after type checking:");

  if (expr_diagnostics->HasErrors()) {
    // Missing debug info for a variable could cause a spurious lookup error.
    for (auto &var : m_local_variables) {
      llvm::Error error = var.TakeLookupError();
      if (!error)
        continue;
      diagnostic_manager.Printf(
          eSeverityError,
          "Missing type debug information for variable \"%s\": %s",
          var.GetName().str().str().c_str(),
          llvm::toString(std::move(error)).c_str());
      return parse_result_failure;
    }
    // Otherwise print the diagnostics from the Swift compiler.
    DiagnoseSwiftASTContextError();
    return parse_result_failure;
  }

  if (repl)
    parsed_expr->code_manipulator->MakeDeclarationsPublic();

  if (!playground) {
    llvm::Error error =
        parsed_expr->code_manipulator->FixupResultAfterTypeChecking();

    if (error) {
      diagnostic_manager.PutString(eSeverityError,
                                   llvm::toString(std::move(error)));
      return parse_result_failure;
    }
  } else {
    swift::performPlaygroundTransform(
        parsed_expr->source_file,
        m_options.GetPlaygroundTransformHighPerformance());
  }

  /// Currently LLDB cannot deal with expressions whose result is a non copyable
  /// type, because there's no easy way to assign $__lldb_result to the result
  /// of the expression.
  if (parsed_expr->code_manipulator &&
      parsed_expr->code_manipulator->IsExpressionResultNonCopyable()) {
    diagnostic_manager.PutString(
        eSeverityError,
        "Cannot evaluate an expression that results in a ~Copyable type");
    return parse_result_failure;
  }

  // FIXME: We now should have to do the name binding and type
  //        checking again, but there should be only the result
  //        variable to bind up at this point.
  if (log) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    parsed_expr->source_file.dump(ss);
    ss.flush();
    LLDB_LOG(log, "Source file after FixupResult:\n{0}", s);
  }

  // Allow variables to be re-used from previous REPL statements.
  if (m_sc.target_sp && (repl || !playground)) {
    Status error;
    auto *persistent_state = llvm::cast<SwiftPersistentExpressionState>(
        m_sc.target_sp->GetPersistentExpressionStateForLanguage(
            lldb::eLanguageTypeSwift));

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

      variable_info.TakeMetadata(
          new SwiftASTManipulatorBase::VariableMetadataPersistent(
              persistent_variable));

      // This is only exercised by the PlaygroundsREPL tests.
      persistent_state->RegisterSwiftPersistentDecl(GetCompilerDecl(decl));
    }

    if (repl) {
      llvm::SmallVector<swift::ValueDecl *, 1> non_variables;
      parsed_expr->code_manipulator->FindNonVariableDeclarations(non_variables);

      for (swift::ValueDecl *decl : non_variables)
        persistent_state->RegisterSwiftPersistentDecl(GetCompilerDecl(decl));
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
    //     LLDB_LOG(log, "Source file after capture fixing:\n{0}", s);
    // }

    if (log) {
      LLDB_LOG(log, "Variables:");

      for (const SwiftASTManipulatorBase::VariableInfo &variable :
           parsed_expr->code_manipulator->GetVariableInfo()) {
        StreamString ss;
        variable.Print(ss);
        LLDB_LOG(log, "  {0}", ss.GetData());
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
        if (!var_info) {
          auto error_string = llvm::toString(var_info.takeError());
          LLDB_LOG(log, "Variable info failzed to materialize with error: {0}",
                   error_string);

          return parse_result_failure;
        }

        const char *name = ConstString(variable.GetName().get()).GetCString();
        variable_map[name] = *var_info;
      }

  if (log) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    parsed_expr->source_file.dump(ss);
    ss.flush();
    LLDB_LOG(log, "Source file before SILgen:\n{0}", s);
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
    LLDB_LOG(log, "SIL module before linking:\n{0}", s);
  }

  if (expr_diagnostics->HasErrors()) {
    DiagnoseSwiftASTContextError();
    return parse_result_failure;
  }

  if (log) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    sil_module->print(ss, &parsed_expr->module);
    ss.flush();
    LLDB_LOG(log, "Generated SIL module:\n{0}", s);
  }

  runSILDiagnosticPasses(*sil_module);
  runSILLoweringPasses(*sil_module);

  if (log) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    sil_module->print(ss, &parsed_expr->module);
    ss.flush();
    LLDB_LOG(log, "SIL module after diagnostic passes:\n{0}", s);
  }

  if (expr_diagnostics->HasErrors()) {
    DiagnoseSwiftASTContextError();
    return parse_result_failure;
  }

  if (m_swift_ast_ctx.GetASTContext()->hadError()) {
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
        /*CAS=*/nullptr, llvm::ArrayRef<std::string>(),
        llvm::ArrayRef<std::string>());

    if (GenModule) {
      swift::performLLVMOptimizations(
          IRGenOpts, m_swift_ast_ctx.GetDiagnosticEngine(), nullptr,
          GenModule.getModule(), GenModule.getTargetMachine(), nullptr);
    }
    auto ContextAndModule = std::move(GenModule).release();
    m_llvm_context.reset(ContextAndModule.first);
    m_module.reset(ContextAndModule.second);
  }

  // In rare cases missing type information results in IRGen lowering
  // values into SILUndef. This may be for a variable that isn't even
  // used in the expression, so just warn about it. This is reported
  // as an out-of-band warning, because LLDB suppresses warnings on
  // successful expressions.
  if (m_sc.target_sp && m_module && ContainsUB(*m_module))
    Debugger::ReportWarning(
        "Expression contains undefined behavior. Expression results may be "
        "incorrect. This may be due to incomplete Swift type information.",
        m_sc.target_sp->GetDebugger().GetID());

  // If IRGen failed without errors, the root cause may be a fatal
  // Clang diagnostic.
  using ErrorKind = SwiftASTContext::ScopedDiagnostics::ErrorKind;
  // GetOptionalErrorKind() returns all diagnostics that occurred to during the
  // lifetime of expr_diagnostics, but there could be earlier ClangImporter
  // errors that still caused the expression to fail.
  std::optional<ErrorKind> error_kind =
      expr_diagnostics->GetOptionalErrorKind();
  if (error_kind == ErrorKind::clang ||
      m_swift_ast_ctx.HasClangImporterErrors()) {
    diagnostic_manager.PutString(
        eSeverityInfo, "couldn't IRGen expression: Clang importer error");
    DiagnoseSwiftASTContextError();
    return parse_result_failure;
  }

  if (error_kind == ErrorKind::swift) {
    diagnostic_manager.PutString(eSeverityInfo,
                                 "couldn't IRGen expression: Swift error");
    DiagnoseSwiftASTContextError();
    return parse_result_failure;
  }

  if (!m_module) {
    diagnostic_manager.Printf(
        eSeverityError,
        "couldn't IRGen expression. Please enable the expression log by "
        "running \"log enable lldb expr\", then run the failing expression "
        "again, and file a bug report with the log output.");
    return parse_result_failure;
  }

  if (log) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    m_module->print(ss, NULL);
    ss.flush();
    LLDB_LOG(log, "Generated IR module:\n{0}", s);
  }

  if (ThreadSafeASTContext ast_ctx = m_swift_ast_ctx.GetASTContext()) {
    if (!SwiftASTManipulator::ShouldBindGenericTypes(
            m_options.GetBindGenericTypes()) &&
        !RedirectCallFromSinkToTrampolineFunction(
            *m_module.get(), *parsed_expr->code_manipulator.get(), **ast_ctx)) {
      diagnostic_manager.Printf(
          eSeverityError,
          "couldn't setup call to the trampoline function. Please enable the "
          "expression log by running \"log enable lldb "
          "expr\", then run the failing expression again, and file a "
          "bugreport with the log output.");
      return parse_result_failure;
    }
  }

  if (log) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    m_module->print(ss, NULL);
    ss.flush();
    LLDB_LOG(log, "Generated IR module after replacing call to sink:\n{0}", s);
  }

  {
    std::lock_guard<std::recursive_mutex> global_context_locker(
        IRExecutionUnit::GetLLVMGlobalContextMutex());

    bool has_errors = LLVMVerifyModule((LLVMOpaqueModule *)m_module.get(),
                                       LLVMReturnStatusAction, nullptr);
    if (has_errors) {
      diagnostic_manager.PutString(eSeverityInfo, "LLVM verification error");
      return parse_result_failure;
    }
  }

  if (expr_diagnostics->HasErrors()) {
    diagnostic_manager.PutString(eSeverityInfo, "post-IRGen error");
    DiagnoseSwiftASTContextError();
    return parse_result_failure;
  }

  // The Parse succeeded!  Now put this module into the context's list
  // of loaded modules, and copy the Decls that were globalized as
  // part of the parse from the staging area in the external lookup
  // object into the SwiftPersistentExpressionState.
  swift::ModuleDecl *module = &parsed_expr->module;
  {
    ThreadSafeASTContext ast_context = GetASTContext(diagnostic_manager);
    ast_context->addLoadedModule(module);
  }
  std::string module_name;
  if (module)
    module_name = module->getName().get();
  m_swift_ast_ctx.CacheModule(module_name, module);
  if (m_sc.target_sp) {
    auto *persistent_state = llvm::cast<SwiftPersistentExpressionState>(
        m_sc.target_sp->GetPersistentExpressionStateForLanguage(
            lldb::eLanguageTypeSwift));
    persistent_state->CopyInSwiftPersistentDecls(
        parsed_expr->external_lookup.GetStagedDecls());
  }
  return ParseResult::success;
}

static bool FindFunctionInModule(ConstString &mangled_name,
                                 llvm::Module *module, const char *orig_name,
                                 bool exact) {
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

Status SwiftExpressionParser::DoPrepareForExecution(
    lldb::addr_t &func_addr, lldb::addr_t &func_end,
    lldb::IRExecutionUnitSP &execution_unit_sp, ExecutionContext &exe_ctx,
    bool &can_interpret, ExecutionPolicy execution_policy) {
  Status err;
  Log *log = GetLog(LLDBLog::Expressions);

  if (!m_module) {
    err = Status::FromErrorString("Can't prepare a NULL module for execution");
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
    err = Status::FromErrorStringWithFormatv(
        "Couldn't find {0}() in the module", orig_name);
    return err;
  } else {
    LLDB_LOG(log, "Found function {0} for {1}", function_name, "$__lldb_expr");
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
  // There isn't a Swift equivalent to clang::Rewriter, so we'll just
  // use that.
  Log *log = GetLog(LLDBLog::Expressions);
  swift::SourceManager &source_manager =
      m_swift_ast_ctx.GetSourceManager();

  const DiagnosticList &diagnostics = diagnostic_manager.Diagnostics();
  size_t num_diags = diagnostics.size();
  if (num_diags == 0)
    return false;

  llvm::RewriteBuffer rewrite_buf;
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
        LLDB_LOG(log,
                 "SwiftExpressionParser::RewriteExpression: ignoring fixit "
                 "since it contains an invalid source location: {0}.",
                 range.str());
        return false;
      }

      // ReplaceText can't handle replacing the same source range more
      // than once, so we have to check that before we proceed:
      if (std::find(source_ranges.begin(), source_ranges.end(), range) !=
          source_ranges.end()) {
        LLDB_LOG(log,
                 "SwiftExpressionParser::RewriteExpression: ignoring fix-it "
                 "since source range appears twice: {0}.",
                 range.str());
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
        LLDB_LOG(
            log,
            "SwiftExpressionParser::RewriteExpression: ignoring fixit since it "
            "contains a source location not in the specified buffer: {0}.",
            range.str());
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
