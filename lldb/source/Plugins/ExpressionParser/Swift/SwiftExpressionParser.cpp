//===-- SwiftExpressionParser.cpp -------------------------------*- C++ -*-===//
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
#include "SwiftREPLMaterializer.h"
#include "SwiftSILManipulator.h"
#include "SwiftUserExpression.h"

#include "Plugins/ExpressionParser/Swift/SwiftDiagnostic.h"
#include "Plugins/ExpressionParser/Swift/SwiftExpressionVariable.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Expression/DiagnosticManager.h"
#include "lldb/Expression/Expression.h"
#include "lldb/Expression/ExpressionSourceCode.h"
#include "lldb/Expression/IRExecutionUnit.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/SwiftLanguageRuntime.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Stream.h"

#include "llvm-c/Analysis.h"
#include "llvm/ADT/ArrayRef.h"
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
#include "swift/AST/DiagnosticEngine.h"
#include "swift/AST/DiagnosticConsumer.h"
#include "swift/AST/IRGenOptions.h"
#include "swift/AST/Module.h"
#include "swift/AST/ModuleLoader.h"
#include "swift/Demangling/Demangle.h"
#include "swift/Basic/PrimarySpecificPaths.h"
#include "swift/Basic/SourceManager.h"
#include "swift/Basic/OptimizationMode.h"
#include "swift/ClangImporter/ClangImporter.h"
#include "swift/Frontend/Frontend.h"
#include "swift/Parse/LocalContext.h"
#include "swift/Parse/PersistentParserState.h"
#include "swift/SIL/SILDebuggerClient.h"
#include "swift/SIL/SILFunction.h"
#include "swift/SIL/SILModule.h"
#include "swift/SILOptimizer/PassManager/Passes.h"
#include "swift/Serialization/SerializedModuleLoader.h"
#include "swift/Subsystems.h"

using namespace lldb_private;
using llvm::make_error;
using llvm::StringError;
using llvm::inconvertibleErrorCode;

SwiftExpressionParser::SwiftExpressionParser(
    ExecutionContextScope *exe_scope, Expression &expr,
    const EvaluateExpressionOptions &options)
    : ExpressionParser(exe_scope, expr, options.GetGenerateDebugInfo()),
      m_expr(expr), m_triple(), m_llvm_context(), m_module(),
      m_execution_unit_sp(), m_sc(), m_exe_scope(exe_scope), m_stack_frame_wp(),
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

  if (target_sp && target_sp->GetArchitecture().IsValid()) {
    std::string triple = target_sp->GetArchitecture().GetTriple().str();

    int dash_count = 0;
    for (size_t i = 0; i < triple.size(); ++i) {
      if (triple[i] == '-')
        dash_count++;
      if (dash_count == 3) {
        triple.resize(i);
        break;
      }
    }

    m_triple = triple;
  } else {
    m_triple = llvm::sys::getDefaultTargetTriple();
  }

  if (target_sp) {
    Status error;
    m_swift_ast_context = llvm::make_unique<SwiftASTContextReader>(
        target_sp->GetScratchSwiftASTContext(error, *exe_scope, true));
  }
}

class VariableMetadataPersistent
    : public SwiftASTManipulatorBase::VariableMetadata {
public:
  VariableMetadataPersistent(lldb::ExpressionVariableSP &persistent_variable_sp)
      : m_persistent_variable_sp(persistent_variable_sp) {}

  static constexpr unsigned Type() { return 'Pers'; }
  virtual unsigned GetType() { return Type(); }
  lldb::ExpressionVariableSP m_persistent_variable_sp;
};

class VariableMetadataVariable
    : public SwiftASTManipulatorBase::VariableMetadata {
public:
  VariableMetadataVariable(lldb::VariableSP &variable_sp)
      : m_variable_sp(variable_sp) {}

  static constexpr unsigned Type() { return 'Vari'; }
  virtual unsigned GetType() { return Type(); }
  lldb::VariableSP m_variable_sp;
};

static CompilerType ImportType(SwiftASTContext &target_context,
                               CompilerType source_type) {
  SwiftASTContext *swift_ast_ctx =
      llvm::dyn_cast_or_null<SwiftASTContext>(source_type.GetTypeSystem());

  if (swift_ast_ctx == nullptr)
    return CompilerType();

  if (swift_ast_ctx == &target_context)
    return source_type;

  Status error, mangled_error;
  CompilerType target_type;

  // First try to get the type by using the mangled name. That will
  // save the mangling step ImportType would have to do:

  ConstString type_name = source_type.GetTypeName();
  ConstString mangled_counterpart;
  bool found_counterpart = type_name.GetMangledCounterpart(mangled_counterpart);
  if (found_counterpart)
    target_type = target_context.GetTypeFromMangledTypename(
        mangled_counterpart.GetCString(), mangled_error);

  if (!target_type.IsValid())
    target_type = target_context.ImportType(source_type, error);

  return target_type;
}

namespace {
class LLDBNameLookup : public swift::SILDebuggerClient {
public:
  LLDBNameLookup(swift::SourceFile &source_file,
                 SwiftExpressionParser::SILVariableMap &variable_map,
                 SymbolContext &sc, ExecutionContextScope &exe_scope)
      : SILDebuggerClient(source_file.getASTContext()),
        m_log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS)),
        m_source_file(source_file), m_variable_map(variable_map), m_sc(sc) {
    source_file.getParentModule()->setDebugClient(this);

    if (!m_sc.target_sp)
      return;
    m_persistent_vars =
        m_sc.target_sp->GetSwiftPersistentExpressionState(exe_scope);
  }

  virtual ~LLDBNameLookup() {}

  virtual swift::SILValue emitLValueForVariable(swift::VarDecl *var,
                                                swift::SILBuilder &builder) {
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

  virtual ~LLDBExprNameLookup() {}

  virtual bool shouldGlobalize(swift::Identifier Name, swift::DeclKind Kind) {
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

  virtual void didGlobalize(swift::Decl *decl) {
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

  virtual bool lookupOverrides(swift::DeclBaseName Name, swift::DeclContext *DC,
                               swift::SourceLoc Loc, bool IsTypeLookup,
                               ResultVector &RV) {
    static unsigned counter = 0;
    unsigned count = counter++;

    if (m_log) {
      m_log->Printf("[LLDBExprNameLookup::lookupOverrides(%u)] Searching for %s",
                    count, Name.getIdentifier().get());
    }

    return false;
  }

  virtual bool lookupAdditions(swift::DeclBaseName Name, swift::DeclContext *DC,
                               swift::SourceLoc Loc, bool IsTypeLookup,
                               ResultVector &RV) {
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
          swift::DeclName value_decl_name = value_decl->getFullName();
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
                  swift::DeclName rv_full_name = rv_decl->getFullName();
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

  virtual swift::Identifier getPreferredPrivateDiscriminator() {
    if (m_sc.comp_unit) {
      if (lldb_private::Module *module = m_sc.module_sp.get()) {
        if (lldb_private::SymbolVendor *symbol_vendor =
                module->GetSymbolVendor()) {
          std::string private_discriminator_string;
          if (symbol_vendor->GetCompileOption("-private-discriminator",
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

  virtual ~LLDBREPLNameLookup() {}

  virtual bool shouldGlobalize(swift::Identifier Name, swift::DeclKind kind) {
    return false;
  }

  virtual void didGlobalize (swift::Decl *Decl) {}

  virtual bool lookupOverrides(swift::DeclBaseName Name, swift::DeclContext *DC,
                               swift::SourceLoc Loc, bool IsTypeLookup,
                               ResultVector &RV) {
    return false;
  }

  virtual bool lookupAdditions(swift::DeclBaseName Name, swift::DeclContext *DC,
                               swift::SourceLoc Loc, bool IsTypeLookup,
                               ResultVector &RV) {
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

  virtual swift::Identifier getPreferredPrivateDiscriminator() {
    return swift::Identifier();
  }
};
}; // END Anonymous namespace

static void
AddRequiredAliases(Block *block, lldb::StackFrameSP &stack_frame_sp,
                   SwiftASTContext &swift_ast_context,
                   SwiftASTManipulator &manipulator,
                   const Expression::SwiftGenericInfo &generic_info) {
  // First emit the typealias for "$__lldb_context".
  if (!block)
    return;

  Function *function = block->CalculateSymbolContextFunction();

  if (!function)
    return;

  constexpr bool can_create = true;
  Block &function_block(function->GetBlock(can_create));

  lldb::VariableListSP variable_list_sp(
      function_block.GetBlockVariableList(true));

  if (!variable_list_sp)
    return;

  lldb::VariableSP self_var_sp(
      variable_list_sp->FindVariable(ConstString("self")));

  if (!self_var_sp)
    return;

  CompilerType self_type;

  if (stack_frame_sp) {
    lldb::ValueObjectSP valobj_sp =
        stack_frame_sp->GetValueObjectForFrameVariable(self_var_sp,
                                                       lldb::eNoDynamicValues);

    if (valobj_sp)
      self_type = valobj_sp->GetCompilerType();
  }

  if (!self_type.IsValid()) {
    if (Type *type = self_var_sp->GetType()) {
      self_type = type->GetForwardCompilerType();
    }
  }

  if (!self_type.IsValid() ||
      !llvm::isa<SwiftASTContext>(self_type.GetTypeSystem()))
    return;

  // Import before getting the unbound version, because the unbound
  // version may not be in the mangled name map.

  CompilerType imported_self_type = ImportType(swift_ast_context, self_type);

  if (!imported_self_type.IsValid())
    return;

  SwiftLanguageRuntime *swift_runtime =
      stack_frame_sp->GetThread()->GetProcess()->GetSwiftLanguageRuntime();
  auto *stack_frame = stack_frame_sp.get();
  imported_self_type =
      swift_runtime->DoArchetypeBindingForType(*stack_frame,
                                               imported_self_type);

  // This might be a referenced type, in which case we really want to
  // extend the referent:
  imported_self_type =
      llvm::cast<SwiftASTContext>(imported_self_type.GetTypeSystem())
          ->GetReferentType(imported_self_type);

  // If we are extending a generic class it's going to be a metatype,
  // and we have to grab the instance type:
  imported_self_type =
      llvm::cast<SwiftASTContext>(imported_self_type.GetTypeSystem())
          ->GetInstanceType(imported_self_type.GetOpaqueQualType());

  Flags imported_self_type_flags(imported_self_type.GetTypeInfo());

  swift::Type object_type =
      GetSwiftType(imported_self_type)->getWithoutSpecifierType();

  if (object_type.getPointer() &&
      (object_type.getPointer() != imported_self_type.GetOpaqueQualType()))
    imported_self_type = CompilerType(imported_self_type.GetTypeSystem(),
                                      object_type.getPointer());

  // If the type of 'self' is a bound generic type, get the unbound version.
  bool is_generic = imported_self_type_flags.AllSet(lldb::eTypeIsSwift |
                                                    lldb::eTypeIsGeneric);
  bool is_bound =
      imported_self_type_flags.AllSet(lldb::eTypeIsSwift | lldb::eTypeIsBound);

  if (is_generic) {
    if (is_bound)
      imported_self_type = imported_self_type.GetUnboundType();
  }

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

    imported_self_type =
        CompilerType(imported_self_type.GetTypeSystem(), self_class_type);
  }

  imported_self_type_flags.Reset(imported_self_type.GetTypeInfo());
  if (imported_self_type_flags.AllClear(lldb::eTypeIsGenericTypeParam)) {
    swift::ValueDecl *type_alias_decl = nullptr;

    type_alias_decl = manipulator.MakeGlobalTypealias(
        swift_ast_context.GetASTContext()->getIdentifier("$__lldb_context"),
        imported_self_type);

    if (!type_alias_decl) {
      Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));
      if (log)
        log->Printf("SEP:AddRequiredAliases: Failed to make the "
                    "$__lldb_context typealias.");
    }
  } else {
    Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));
    if (log)
      log->Printf("SEP:AddRequiredAliases: Failed to resolve the self "
                  "archetype - could not make the $__lldb_context "
                  "typealias.");
  }
}

static void CountLocals(
    SymbolContext &sc, lldb::StackFrameSP &stack_frame_sp,
    SwiftASTContext &ast_context,
    llvm::SmallVectorImpl<SwiftASTManipulator::VariableInfo> &local_variables) {
  // Avoids shadowing.
  std::set<ConstString> counted_names;

  if (!sc.block && !sc.function)
    return;

  Block *block = sc.block;
  Block *top_block = block->GetContainingInlinedBlock();

  if (!top_block)
    top_block = &sc.function->GetBlock(true);

  static ConstString s_self_name("self");

  SwiftLanguageRuntime *language_runtime = nullptr;
  ExecutionContextScope *scope = nullptr;

  if (stack_frame_sp) {
    language_runtime =
        stack_frame_sp->GetThread()->GetProcess()->GetSwiftLanguageRuntime();
    scope = stack_frame_sp.get();
  }

  // The module scoped variables are stored at the CompUnit level, so
  // after we go through the current context, then we have to take one
  // more pass through the variables in the CompUnit.
  bool handling_globals = false;

  while (true) {
    VariableList variables;

    if (!handling_globals) {

      constexpr bool can_create = true;
      constexpr bool get_parent_variables = false;
      constexpr bool stop_if_block_is_inlined_function = true;

      block->AppendVariables(can_create, get_parent_variables,
                             stop_if_block_is_inlined_function,
                             [](Variable *) { return true; }, &variables);
    } else {
      if (sc.comp_unit) {
        lldb::VariableListSP globals_sp = sc.comp_unit->GetVariableList(true);
        if (globals_sp)
          variables.AddVariables(globals_sp.get());
      }
    }

    for (size_t vi = 0, ve = variables.GetSize(); vi != ve; ++vi) {
      lldb::VariableSP variable_sp(variables.GetVariableAtIndex(vi));

      const ConstString &name(variable_sp->GetUnqualifiedName());
      const char *name_cstring = variable_sp->GetUnqualifiedName().GetCString();

      if (name.IsEmpty())
        continue;

      if (counted_names.count(name))
        continue;

      CompilerType var_type;

      if (stack_frame_sp) {
        lldb::ValueObjectSP valobj_sp =
            stack_frame_sp->GetValueObjectForFrameVariable(
                variable_sp, lldb::eNoDynamicValues);

        if (!valobj_sp || valobj_sp->GetError().Fail()) {
          // Ignore the variable if we couldn't find its corresponding
          // value object.  TODO if the expression tries to use an
          // ignored variable, produce a sensible error.
          continue;
        } else {
          var_type = valobj_sp->GetCompilerType();
        }

        if (var_type.IsValid() && !SwiftASTContext::IsFullyRealized(var_type)) {
          lldb::ValueObjectSP dynamic_valobj_sp =
              valobj_sp->GetDynamicValue(lldb::eDynamicDontRunTarget);

          if (!dynamic_valobj_sp || dynamic_valobj_sp->GetError().Fail()) {
            continue;
          }
        }
      }

      if (!var_type.IsValid()) {
        Type *var_lldb_type = variable_sp->GetType();

        if (var_lldb_type)
          var_type = var_lldb_type->GetFullCompilerType();
      }

      if (!var_type.IsValid())
        continue;

      if (!llvm::isa<SwiftASTContext>(var_type.GetTypeSystem()))
        continue;

      Status error;
      CompilerType target_type = ast_context.ImportType(var_type, error);

      // If the import failed, give up.
      if (!target_type.IsValid())
        continue;

      // Make sure to resolve all archetypes in the variable type.
      if (stack_frame_sp) {
        if (language_runtime)
          target_type = language_runtime->DoArchetypeBindingForType(
              *stack_frame_sp, target_type);
      }

      // If we couldn't fully realize the type, then we aren't going
      // to get very far making a local out of it, so discard it here.
      Log *log(lldb_private::GetLogIfAnyCategoriesSet(LIBLLDB_LOG_TYPES |
                                                      LIBLLDB_LOG_EXPRESSIONS));
      if (!SwiftASTContext::IsFullyRealized(target_type)) {
        if (log) {
          log->Printf("Discarding local %s because we couldn't fully realize "
                      "it, our best attempt was: %s.",
                      name_cstring,
                      target_type.GetTypeName().AsCString("<unknown>"));
        }
        continue;
      }

      SwiftASTManipulatorBase::VariableMetadataSP metadata_sp(
          new VariableMetadataVariable(variable_sp));

      const char *overridden_name = name_cstring;

      if (name == s_self_name) {
        overridden_name = ConstString("$__lldb_injected_self").AsCString();
        if (log) {
          swift::Type swift_type = GetSwiftType(target_type);
          if (swift_type) {
            std::string s;
            llvm::raw_string_ostream ss(s);
            swift_type->dump(ss);
            ss.flush();
            log->Printf("Adding injected self: type (%p) context(%p) is: %s",
                        swift_type, ast_context.GetASTContext(), s.c_str());
          }
        }
      }

      SwiftASTManipulator::VariableInfo variable_info(
          target_type,
          ast_context.GetASTContext()->getIdentifier(overridden_name),
          metadata_sp,
          swift::VarDecl::Specifier::Var);

      local_variables.push_back(variable_info);

      counted_names.insert(name);
    }

    if (handling_globals) {
      // Okay, now we're done.
      break;
    } else if (block == top_block) {
      // Now add the containing module block, that's what holds the
      // module globals:
      handling_globals = true;
    } else
      block = block->GetParent();
  }
}

static void ResolveSpecialNames(
    SymbolContext &sc, ExecutionContextScope &exe_scope,
    SwiftASTContext &ast_context,
    llvm::SmallVectorImpl<swift::Identifier> &special_names,
    llvm::SmallVectorImpl<SwiftASTManipulator::VariableInfo> &local_variables) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));

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

    if (!llvm::isa<SwiftASTContext>(var_type.GetTypeSystem()))
      continue;

    CompilerType target_type;
    Status error;

    target_type = ast_context.ImportType(var_type, error);

    if (!target_type)
      continue;

    SwiftASTManipulatorBase::VariableMetadataSP metadata_sp(
        new VariableMetadataPersistent(expr_var_sp));

    auto specifier = llvm::cast<SwiftExpressionVariable>(expr_var_sp.get())
                       ->GetIsModifiable()
                   ? swift::VarDecl::Specifier::Var
                   : swift::VarDecl::Specifier::Let;
    SwiftASTManipulator::VariableInfo variable_info(
        target_type, ast_context.GetASTContext()->getIdentifier(name.str()),
        metadata_sp, specifier);

    local_variables.push_back(variable_info);
  }
}

//----------------------------------------------------------------------
/// Diagnostics are part of the SwiftASTContext and we must enable and
/// disable colorization manually in the SwiftASTContext. We need to
/// ensure that if we modify the setting that we restore it to what it
/// was. This class helps us to do that without having to intrument
/// all returns from a function, like in
/// SwiftExpressionParser::Parse(...).
/// //----------------------------------------------------------------------
class SetColorize {
public:
  SetColorize(SwiftASTContext *swift_ast, bool colorize)
      : m_swift_ast(swift_ast),
        m_saved_colorize(swift_ast->SetColorizeDiagnostics(colorize)) {}

  ~SetColorize() { m_swift_ast->SetColorizeDiagnostics(m_saved_colorize); }

protected:
  SwiftASTContext *m_swift_ast;
  const bool m_saved_colorize;
};

/// Initialize the SwiftASTContext and return the wrapped
/// swift::ASTContext when successful.
static swift::ASTContext *SetupASTContext(
    SwiftASTContext *swift_ast_context, DiagnosticManager &diagnostic_manager,
    std::function<bool()> disable_objc_runtime, bool repl, bool playground) {
  if (!swift_ast_context) {
    diagnostic_manager.PutString(
        eDiagnosticSeverityError,
        "No AST context to parse into.  Please parse with a target.\n");
    return nullptr;
  }

  // Lazily get the clang importer if we can to make sure it exists in
  // case we need it.
  if (!swift_ast_context->GetClangImporter()) {
    std::string swift_error = "Fatal Swift ";
    swift_error +=
        swift_ast_context->GetFatalErrors().AsCString("error: unknown error.");
    diagnostic_manager.PutString(eDiagnosticSeverityError, swift_error);
    diagnostic_manager.PutString(
        eDiagnosticSeverityRemark,
        "Swift expressions require OS X 10.10 / iOS 8 SDKs or later.\n");
    return nullptr;
  }

  if (swift_ast_context->HasFatalErrors()) {
    diagnostic_manager.PutString(eDiagnosticSeverityError,
                                 "The AST context is in a fatal error state.");
    return nullptr;
  }

  swift::ASTContext *ast_context = swift_ast_context->GetASTContext();
  if (!ast_context) {
    diagnostic_manager.PutString(
        eDiagnosticSeverityError,
        "Couldn't initialize the AST context.  Please check your settings.");
    return nullptr;
  }

  if (swift_ast_context->HasFatalErrors()) {
    diagnostic_manager.PutString(eDiagnosticSeverityError,
                                 "The AST context is in a fatal error state.");
    return nullptr;
  }

  // TODO: Find a way to get contraint-solver output sent to a stream
  //       so we can log it.
  // swift_ast_context->GetLanguageOptions().DebugConstraintSolver = true;
  swift_ast_context->ClearDiagnostics();

  // Make a class that will set/restore the colorize setting in the
  // SwiftASTContext for us.
  
  // SetColorize colorize(swift_ast_context,
  // stream.GetFlags().Test(Stream::eANSIColor));
  swift_ast_context->GetLanguageOptions().DebuggerSupport = true;
  // No longer part of debugger support, set it separately.
  swift_ast_context->GetLanguageOptions().EnableDollarIdentifiers = true;
  swift_ast_context->GetLanguageOptions().EnableAccessControl =
      (repl || playground);
  swift_ast_context->GetLanguageOptions().EnableTargetOSChecking = false;

  if (disable_objc_runtime())
    swift_ast_context->GetLanguageOptions().EnableObjCInterop = false;

  swift_ast_context->GetLanguageOptions().Playground = repl || playground;
  swift_ast_context->GetIRGenOptions().Playground = repl || playground;

  // For the expression parser and REPL we want to relax the
  // requirement that you put "try" in front of every expression that
  // might throw.
  if (repl || !playground)
    swift_ast_context->GetLanguageOptions().EnableThrowWithoutTry = true;

  swift_ast_context->GetIRGenOptions().OptMode =
      swift::OptimizationMode::NoOptimization;
  // Normally we'd like to verify, but unfortunately the verifier's
  // error mode is abort().
  swift_ast_context->GetIRGenOptions().Verify = false;

  swift_ast_context->GetIRGenOptions().DisableRoundTripDebugTypes = true;
  return ast_context;
}

/// Returns the buffer_id for the expression's source code.
static std::pair<unsigned, std::string>
CreateMainFile(SwiftASTContext &swift_ast_context, StringRef filename,
               StringRef text, const EvaluateExpressionOptions &options) {
  const bool generate_debug_info = options.GetGenerateDebugInfo();
  swift_ast_context.SetGenerateDebugInfo(generate_debug_info
                                           ? swift::IRGenDebugInfoLevel::Normal
                                           : swift::IRGenDebugInfoLevel::None);
  swift::IRGenOptions &ir_gen_options = swift_ast_context.GetIRGenOptions();

  if (generate_debug_info) {
    std::string temp_source_path;
    if (ExpressionSourceCode::SaveExpressionTextToTempFile(text, options,
                                                           temp_source_path)) {
      auto error_or_buffer_ap =
          llvm::MemoryBuffer::getFile(temp_source_path.c_str());
      if (error_or_buffer_ap.getError() == std::error_condition()) {
        unsigned buffer_id =
            swift_ast_context.GetSourceManager().addNewSourceBuffer(
                std::move(error_or_buffer_ap.get()));

        llvm::SmallString<256> source_dir(temp_source_path);
        llvm::sys::path::remove_filename(source_dir);
        ir_gen_options.DebugCompilationDir = source_dir.str();

        return {buffer_id, temp_source_path};
      }
    }
  }

  std::unique_ptr<llvm::MemoryBuffer> expr_buffer(
      llvm::MemoryBuffer::getMemBufferCopy(text, filename));
  unsigned buffer_id = swift_ast_context.GetSourceManager().addNewSourceBuffer(
      std::move(expr_buffer));
  return {buffer_id, filename};
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
  uint64_t offset = 0;
  bool needs_init = false;

  bool is_result =
      variable.MetadataIs<SwiftASTManipulatorBase::VariableMetadataResult>();
  bool is_error =
      variable.MetadataIs<SwiftASTManipulatorBase::VariableMetadataError>();

  if (is_result || is_error) {
    needs_init = true;

    Status error;

    if (repl) {
      if (swift::Type swift_type = GetSwiftType(variable.GetType())) {
        if (!swift_type->isVoid()) {
          auto &repl_mat = *llvm::cast<SwiftREPLMaterializer>(&materializer);
          if (is_result)
            offset = repl_mat.AddREPLResultVariable(
                variable.GetType(), variable.GetDecl(),
                &user_expression.GetResultDelegate(), error);
          else
            offset = repl_mat.AddREPLResultVariable(
                variable.GetType(), variable.GetDecl(),
                &user_expression.GetErrorDelegate(), error);
        }
      }
    } else {
      CompilerType actual_type(variable.GetType());
      auto orig_swift_type = GetSwiftType(actual_type);
      auto *swift_type = orig_swift_type->mapTypeOutOfContext().getPointer();
      actual_type.SetCompilerType(actual_type.GetTypeSystem(), swift_type);
      lldb::StackFrameSP stack_frame_sp = stack_frame_wp.lock();
      if (swift_type->hasTypeParameter()) {
        if (stack_frame_sp && stack_frame_sp->GetThread() &&
            stack_frame_sp->GetThread()->GetProcess()) {
          SwiftLanguageRuntime *swift_runtime = stack_frame_sp->GetThread()
                                                    ->GetProcess()
                                                    ->GetSwiftLanguageRuntime();
          if (swift_runtime) {
            actual_type = swift_runtime->DoArchetypeBindingForType(
                *stack_frame_sp, actual_type);
          }
        }
      }

      // Desugar '$lldb_context', etc.
      auto transformed_type = GetSwiftType(actual_type).transform(
        [](swift::Type t) -> swift::Type {
          if (auto *aliasTy = swift::dyn_cast<swift::TypeAliasType>(t.getPointer())) {
            if (aliasTy && aliasTy->getDecl()->isDebuggerAlias()) {
              return aliasTy->getSinglyDesugaredType();
            }
          }
          return t;
      });
      actual_type.SetCompilerType(actual_type.GetTypeSystem(),
                                  transformed_type.getPointer());

      if (is_result)
        offset = materializer.AddResultVariable(
            actual_type, false, true, &user_expression.GetResultDelegate(),
            error);
      else
        offset = materializer.AddResultVariable(
            actual_type, false, true, &user_expression.GetErrorDelegate(),
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
  } else if (variable.MetadataIs<VariableMetadataVariable>()) {
    Status error;

    VariableMetadataVariable *variable_metadata =
        static_cast<VariableMetadataVariable *>(variable.m_metadata.get());

    // FIXME: It would be nice if we could do something like
    //        variable_metadata->m_variable_sp->SetType(variable.GetType())
    //        here.
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
  } else if (variable.MetadataIs<VariableMetadataPersistent>()) {
    VariableMetadataPersistent *variable_metadata =
        static_cast<VariableMetadataPersistent *>(variable.m_metadata.get());

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

  return SwiftExpressionParser::SILVariableInfo(variable.GetType(), offset,
                                                needs_init);
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
  std::string Message;

  ModuleImportError(llvm::Twine Message) : Message(Message.str()) {}
  void log(llvm::raw_ostream &OS) const override { OS << "ModuleImport"; }
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
static llvm::Expected<ParsedExpression>
ParseAndImport(SwiftASTContext *swift_ast_context, Expression &expr,
               SwiftExpressionParser::SILVariableMap &variable_map,
               unsigned &buffer_id, DiagnosticManager &diagnostic_manager,
               SwiftExpressionParser &swift_expr_parser,
               lldb::StackFrameWP &stack_frame_wp, SymbolContext &sc,
               ExecutionContextScope &exe_scope,
               const EvaluateExpressionOptions &options, bool repl,
               bool playground) {

  auto should_disable_objc_runtime = [&]() {
    lldb::StackFrameSP this_frame_sp(stack_frame_wp.lock());
    if (!this_frame_sp)
      return false;
    lldb::ProcessSP process_sp(this_frame_sp->CalculateProcess());
    if (!process_sp)
      return false;
    return !process_sp->GetObjCLanguageRuntime();
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
    persistent_state->AddHandLoadedModule(ConstString("Swift"));
  }

  std::string main_filename;
  std::tie(buffer_id, main_filename) = CreateMainFile(
      *swift_ast_context, repl ? "<REPL>" : "<EXPR>", expr.Text(), options);

  char expr_name_buf[32];

  snprintf(expr_name_buf, sizeof(expr_name_buf), "__lldb_expr_%u",
           options.GetExpressionNumber());

  auto module_id = ast_context->getIdentifier(expr_name_buf);
  auto &module = *swift::ModuleDecl::create(module_id, *ast_context);
  const auto implicit_import_kind =
      swift::SourceFile::ImplicitModuleImportKind::Stdlib;

  auto &invocation = swift_ast_context->GetCompilerInvocation();
  invocation.getFrontendOptions().ModuleName = expr_name_buf;
  invocation.getIRGenOptions().ModuleName = expr_name_buf;

  swift::SourceFileKind source_file_kind = swift::SourceFileKind::Library;

  if (playground || repl) {
    source_file_kind = swift::SourceFileKind::Main;
  }

  swift::SourceFile *source_file = new (*ast_context) swift::SourceFile(
      module, source_file_kind, buffer_id, implicit_import_kind,
      /*Keep tokens*/ false);
  module.addFile(*source_file);

  bool done = false;

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
  swift_ast_context->AddDebuggerClient(external_lookup);

  swift::PersistentParserState persistent_state(*ast_context);

  while (!done) {
    // Note, we disable delayed parsing for the swift expression parser.
    swift::parseIntoSourceFile(*source_file, buffer_id, &done, nullptr,
                               &persistent_state, nullptr,
                               /*DelayBodyParsing=*/false);

    if (swift_ast_context->HasErrors())
      return make_error<SwiftASTContextError>();
  }

  if (!done)
    return make_error<llvm::StringError>(
        "Parse did not consume the whole expression.",
        inconvertibleErrorCode());

  std::unique_ptr<SwiftASTManipulator> code_manipulator;
  if (repl || !playground) {
    code_manipulator =
        llvm::make_unique<SwiftASTManipulator>(*source_file, repl);

    if (!playground) {
      code_manipulator->RewriteResult();
    }
  }

  Status auto_import_error;
  if (!SwiftASTContext::PerformAutoImport(*swift_ast_context, sc,
                                          stack_frame_wp, source_file,
                                          auto_import_error))
    return make_error<ModuleImportError>(llvm::Twine("in auto-import:\n") +
                                         auto_import_error.AsCString());

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
        swift_ast_context->LoadExtraDylibs(*process_sp.get(), error);
      }
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

    llvm::SmallVector<SwiftASTManipulator::VariableInfo, 5> local_variables;

    if (local_context_is_swift) {
      AddRequiredAliases(sc.block, stack_frame_sp, *swift_ast_context,
                         *code_manipulator, expr.GetSwiftGenericInfo());

      // Register all local variables so that lookups to them resolve.
      CountLocals(sc, stack_frame_sp, *swift_ast_context, local_variables);
    }

    // Register all magic variables.
    llvm::SmallVector<swift::Identifier, 2> special_names;
    llvm::StringRef persistent_var_prefix;
    if (!repl)
      persistent_var_prefix = "$";

    code_manipulator->FindSpecialNames(special_names, persistent_var_prefix);

    ResolveSpecialNames(sc, exe_scope, *swift_ast_context, special_names,
                        local_variables);

    code_manipulator->AddExternalVariables(local_variables);

    stack_frame_sp.reset();
  }

  swift::performNameBinding(*source_file);

  if (swift_ast_context->HasErrors())
    return make_error<SwiftASTContextError>();

  // Do the auto-importing after Name Binding, that's when the Imports
  // for the source file are figured out.
  {
    std::lock_guard<std::recursive_mutex> global_context_locker(
        IRExecutionUnit::GetLLVMGlobalContextMutex());

    Status auto_import_error;
    if (!SwiftASTContext::PerformUserImport(*swift_ast_context, sc, exe_scope,
                                            stack_frame_wp, *source_file,
                                            auto_import_error)) {
      return make_error<ModuleImportError>(llvm::Twine("in user-import:\n") +
                                           auto_import_error.AsCString());
    }
  }

  // After the swift code manipulator performed AST transformations,
  // verify that the AST we have in our hands is valid. This is a nop
  // for release builds, but helps catching bug when assertions are
  // turned on.
  swift::verify(*source_file);

  ParsedExpression result = {std::move(code_manipulator),
                             *ast_context,
                             module,
                             *external_lookup,
                             *source_file,
                             std::move(main_filename)};
  return std::move(result);
}

unsigned SwiftExpressionParser::Parse(DiagnosticManager &diagnostic_manager,
                                      uint32_t first_line, uint32_t last_line,
                                      uint32_t line_offset) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));

  SwiftExpressionParser::SILVariableMap variable_map;
  auto *swift_ast_ctx = m_swift_ast_context->get();

  // Helper function to diagnose errors in m_swift_ast_context.
  unsigned buffer_id = UINT32_MAX;
  auto DiagnoseSwiftASTContextError = [&]() {
    assert(swift_ast_ctx->HasErrors() && "error expected");
    swift_ast_ctx->PrintDiagnostics(diagnostic_manager, buffer_id,
                                          first_line, last_line, line_offset);
  };

  // In the case of playgrounds, we turn all rewriting functionality off.
  const bool repl = m_options.GetREPLEnabled();
  const bool playground = m_options.GetPlaygroundTransformEnabled();

  if (!m_exe_scope)
    return false;

  // Parse the expression an import all nececssary swift modules.
  auto parsed_expr =
      ParseAndImport(m_swift_ast_context->get(), m_expr, variable_map,
                     buffer_id, diagnostic_manager, *this, m_stack_frame_wp,
                     m_sc, *m_exe_scope, m_options, repl, playground);

  if (!parsed_expr) {
    bool retry = false;
    handleAllErrors(parsed_expr.takeError(),
                    [&](const ModuleImportError &MIE) {
                      if (swift_ast_ctx->GetClangImporter())
                        // Already on backup power.
                        diagnostic_manager.PutString(eDiagnosticSeverityError,
                                                     MIE.Message);
                      else
                        // Discard the shared scratch context and retry.
                        retry = true;
                    },
                    [&](const SwiftASTContextError &SACE) {
                      if (swift_ast_ctx->GetClangImporter())
                        DiagnoseSwiftASTContextError();
                      else
                        // Discard the shared scratch context and retry.
                        retry = true;
                    },
                    [&](const StringError &SE) {
                      diagnostic_manager.PutString(eDiagnosticSeverityError,
                                                   SE.getMessage());
                    },
                    [](const PropagatedError &P) {});

    // Unrecoverable error?
    if (!retry)
      return 1;

    // Signal that we want to retry the expression exactly once with a
    // fresh SwiftASTContext initialized with the flags from the
    // current lldb::Module / Swift dylib to avoid header search
    // mismatches.
    m_sc.target_sp->SetUseScratchTypesystemPerModule(true);
    return 2;
  }

  // Not persistent because we're building source files one at a time.
  swift::TopLevelContext top_level_context;
  swift::OptionSet<swift::TypeCheckingFlags> type_checking_options;

  swift::performTypeChecking(parsed_expr->source_file, top_level_context,
                             type_checking_options);

  if (swift_ast_ctx->HasErrors()) {
    DiagnoseSwiftASTContextError();
    return 1;
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
      return 1;
    }
  } else {
    swift::performPlaygroundTransform(parsed_expr->source_file, true);
    swift::typeCheckExternalDefinitions(parsed_expr->source_file);
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
    SwiftASTContext *scratch_ast_context = m_swift_ast_context->get();

    if (scratch_ast_context) {
      auto *persistent_state =
          m_sc.target_sp->GetSwiftPersistentExpressionState(*m_exe_scope);

      llvm::SmallVector<size_t, 1> declaration_indexes;
      parsed_expr->code_manipulator->FindVariableDeclarations(
          declaration_indexes, repl);

      for (size_t declaration_index : declaration_indexes) {
        SwiftASTManipulator::VariableInfo &variable_info =
            parsed_expr->code_manipulator->GetVariableInfo()[declaration_index];

        CompilerType imported_type =
            ImportType(*scratch_ast_context, variable_info.GetType());

        if (imported_type) {
          lldb::ExpressionVariableSP persistent_variable =
              persistent_state->AddNewlyConstructedVariable(
                  new SwiftExpressionVariable(
                      m_sc.target_sp.get(),
                      ConstString(variable_info.GetName().str()), imported_type,
                      m_sc.target_sp->GetArchitecture().GetByteOrder(),
                      m_sc.target_sp->GetArchitecture().GetAddressByteSize()));

          if (repl) {
            persistent_variable->m_flags |= ExpressionVariable::EVKeepInTarget;
            persistent_variable->m_flags |=
                ExpressionVariable::EVIsProgramReference;
          } else {
            persistent_variable->m_flags |=
                ExpressionVariable::EVNeedsAllocation;
            persistent_variable->m_flags |= ExpressionVariable::EVKeepInTarget;
            llvm::cast<SwiftExpressionVariable>(persistent_variable.get())
                ->m_swift_flags |= SwiftExpressionVariable::EVSNeedsInit;
          }

          swift::VarDecl *decl = variable_info.GetDecl();
          if (decl) {
            if (decl->isLet()) {
              llvm::cast<SwiftExpressionVariable>(persistent_variable.get())
                  ->SetIsModifiable(false);
            }
            if (!decl->hasStorage()) {
              llvm::cast<SwiftExpressionVariable>(persistent_variable.get())
                  ->SetIsComputed(true);
            }
          }

          variable_info.m_metadata.reset(
              new VariableMetadataPersistent(persistent_variable));

          persistent_state->RegisterSwiftPersistentDecl(decl);
        }
      }

      if (repl) {
        llvm::SmallVector<swift::ValueDecl *, 1> non_variables;
        parsed_expr->code_manipulator->FindNonVariableDeclarations(
            non_variables);

        for (swift::ValueDecl *decl : non_variables) {
          persistent_state->RegisterSwiftPersistentDecl(decl);
        }
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
          return 1;

        const char *name = ConstString(variable.GetName().get()).GetCString();
        variable_map[name] = *var_info;
      }

  std::unique_ptr<swift::SILModule> sil_module(swift::performSILGeneration(
      parsed_expr->source_file, swift_ast_ctx->GetSILOptions()));

  if (log) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    const bool verbose = false;
    sil_module->print(ss, verbose, &parsed_expr->module);
    ss.flush();

    log->Printf("SIL module before linking:");
    log->PutCString(s.c_str());
  }

  if (swift_ast_ctx->HasErrors()) {
    DiagnoseSwiftASTContextError();
    return 1;
  }

  if (log) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    const bool verbose = false;
    sil_module->print(ss, verbose, &parsed_expr->module);
    ss.flush();

    log->Printf("Generated SIL module:");
    log->PutCString(s.c_str());
  }

  runSILDiagnosticPasses(*sil_module);

  if (log) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    const bool verbose = false;
    sil_module->print(ss, verbose, &parsed_expr->module);
    ss.flush();

    log->Printf("SIL module after diagnostic passes:");
    log->PutCString(s.c_str());
  }

  if (swift_ast_ctx->HasErrors()) {
    DiagnoseSwiftASTContextError();
    return 1;
  }

  {
    std::lock_guard<std::recursive_mutex> global_context_locker(
        IRExecutionUnit::GetLLVMGlobalContextMutex());

    m_module = swift::performIRGeneration(
        swift_ast_ctx->GetIRGenOptions(), &parsed_expr->module,
        std::move(sil_module), "lldb_module",
        swift::PrimarySpecificPaths("", parsed_expr->main_filename),
        SwiftASTContext::GetGlobalLLVMContext(), llvm::ArrayRef<std::string>());
  }

  if (swift_ast_ctx->HasErrors()) {
    DiagnoseSwiftASTContextError();
    return 1;
  }

  if (!m_module) {
    diagnostic_manager.PutString(
        eDiagnosticSeverityError,
        "Couldn't IRGen expression, no additional error");
    return 1;
  }

  if (log) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    m_module->print(ss, NULL);
    ss.flush();

    log->Printf("Generated IR module:");
    log->PutCString(s.c_str());
  }

  {
    std::lock_guard<std::recursive_mutex> global_context_locker(
        IRExecutionUnit::GetLLVMGlobalContextMutex());

    LLVMVerifyModule((LLVMOpaqueModule *)m_module.get(), LLVMReturnStatusAction,
                     nullptr);
  }

  if (swift_ast_ctx->HasErrors())
    return 1;

  // The Parse succeeded!  Now put this module into the context's list
  // of loaded modules, and copy the Decls that were globalized as
  // part of the parse from the staging area in the external lookup
  // object into the SwiftPersistentExpressionState.
  swift::ModuleDecl *module = &parsed_expr->module;
  parsed_expr->ast_context.LoadedModules.insert({module->getName(), module});
  swift_ast_ctx->CacheModule(module);
  if (m_sc.target_sp) {
    auto *persistent_state =
        m_sc.target_sp->GetSwiftPersistentExpressionState(*m_exe_scope);
    persistent_state->CopyInSwiftPersistentDecls(
        parsed_expr->external_lookup.GetStagedDecls());
  }
  return 0;
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

      swift::Demangle::NodePointer node_ptr =
          demangle_ctx.demangleSymbolAsNode(fi->getName());
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
  Status err;
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));

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

  std::unique_ptr<llvm::LLVMContext> llvm_context_up;
  // m_module is handed off here.
  m_execution_unit_sp.reset(
      new IRExecutionUnit(llvm_context_up, m_module, function_name,
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
  auto *swift_ast_ctx = m_swift_ast_context->get();
  if (!swift_ast_ctx)
    return false;

  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));

  swift::SourceManager &source_manager =
      swift_ast_ctx->GetSourceManager();

  const DiagnosticList &diagnostics = diagnostic_manager.Diagnostics();
  size_t num_diags = diagnostics.size();
  if (num_diags == 0)
    return false;

  clang::RewriteBuffer rewrite_buf;
  llvm::StringRef text_ref(m_expr.Text());
  rewrite_buf.Initialize(text_ref);

  for (const Diagnostic *diag : diagnostic_manager.Diagnostics()) {
    const SwiftDiagnostic *diagnostic = llvm::dyn_cast<SwiftDiagnostic>(diag);
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
