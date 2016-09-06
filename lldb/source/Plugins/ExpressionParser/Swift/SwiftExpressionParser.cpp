//===-- SwiftExpressionParser.cpp -------------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2015 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See http://swift.org/LICENSE.txt for license information
// See http://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#include "SwiftExpressionParser.h"

#include "SwiftASTManipulator.h"
#include "SwiftREPLMaterializer.h"
#include "SwiftSILManipulator.h"
#include "SwiftUserExpression.h"

#include "lldb/Core/Log.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Expression/DiagnosticManager.h"
#include "lldb/Expression/Expression.h"
#include "lldb/Expression/ExpressionSourceCode.h"
#include "lldb/Expression/IRExecutionUnit.h"
#include "Plugins/ExpressionParser/Swift/SwiftDiagnostic.h"
#include "Plugins/ExpressionParser/Swift/SwiftExpressionVariable.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/SwiftLanguageRuntime.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

#include "llvm-c/Analysis.h"
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

#include "swift/Subsystems.h"
#include "swift/AST/ASTContext.h"
#include "swift/AST/DiagnosticEngine.h"
#include "swift/AST/IRGenOptions.h"
#include "swift/AST/Module.h"
#include "swift/AST/ModuleLoader.h"
#include "swift/Basic/Demangle.h"
#include "swift/Basic/SourceManager.h"
#include "swift/ClangImporter/ClangImporter.h"
#include "swift/Frontend/Frontend.h"
#include "swift/Parse/LocalContext.h"
#include "swift/Parse/PersistentParserState.h"
#include "swift/Serialization/SerializedModuleLoader.h"
#include "swift/SIL/SILDebuggerClient.h"
#include "swift/SIL/SILFunction.h"
#include "swift/SIL/SILModule.h"
#include "swift/SILOptimizer/PassManager/Passes.h"
#include "swift/Basic/DiagnosticConsumer.h"

using namespace lldb_private;

SwiftExpressionParser::SwiftExpressionParser (ExecutionContextScope *exe_scope,
                                              Expression &expr,
                                              const EvaluateExpressionOptions &options) :
    ExpressionParser(exe_scope,
                     expr,
                     options.GetGenerateDebugInfo()),
    m_expr (expr),
    m_triple (),
    m_llvm_context (),
    m_module (),
    m_execution_unit_sp (),
    m_swift_ast_context (NULL),
    m_sc (),
    m_stack_frame_wp (),
    m_options (options)
{
    assert (expr.Language() == lldb::eLanguageTypeSwift);
    
    // TODO This code is copied from ClangExpressionParser.cpp.
    // Factor this out into common code.
    
    lldb::TargetSP target_sp;
    if (exe_scope)
    {
        target_sp = exe_scope->CalculateTarget();
        
        lldb::StackFrameSP stack_frame = exe_scope->CalculateStackFrame();
        
        if (stack_frame)
        {
            m_stack_frame_wp = stack_frame;
            m_sc = stack_frame->GetSymbolContext(lldb::eSymbolContextEverything);
        }
        else
        {
            m_sc.target_sp = target_sp;
        }
    }
    
    if (target_sp && target_sp->GetArchitecture().IsValid())
    {
        std::string triple = target_sp->GetArchitecture().GetTriple().str();
        
        int dash_count = 0;
        for (size_t i = 0; i < triple.size(); ++i)
        {
            if (triple[i] == '-')
                dash_count++;
            if (dash_count == 3)
            {
                triple.resize(i);
                break;
            }
        }
        
        m_triple = triple;
    }
    else
    {
        m_triple = llvm::sys::getDefaultTargetTriple();
    }
    
    if (target_sp)
    {
        m_swift_ast_context = llvm::cast_or_null<SwiftASTContext>(target_sp->GetScratchTypeSystemForLanguage(nullptr, lldb::eLanguageTypeSwift));
    }
}

static void
DescribeFileUnit(Stream &s, swift::FileUnit *file_unit)
{
    s.PutCString("kind = ");
    
    switch (file_unit->getKind()) {
        default:
        {
            s.PutCString("<unknown>");
        }
        case swift::FileUnitKind::Source:
        {
            s.PutCString("Source, ");
            if (swift::SourceFile *source_file = llvm::dyn_cast<swift::SourceFile>(file_unit))
            {
                s.Printf("filename = '%s', ", source_file->getFilename().str().c_str());
                s.PutCString("source file kind = ");
                switch (source_file->Kind)
                {
                    case swift::SourceFileKind::Library:    s.PutCString("Library");
                    case swift::SourceFileKind::Main:       s.PutCString("Main");
                    case swift::SourceFileKind::REPL:       s.PutCString("REPL");
                    case swift::SourceFileKind::SIL:        s.PutCString("SIL");
                }
            }
        }
            break;
        case swift::FileUnitKind::Builtin:
        {
            s.PutCString("Builtin");
        }
            break;
        case swift::FileUnitKind::SerializedAST:
        case swift::FileUnitKind::ClangModule:
        {
            s.PutCString("SerializedAST, ");
            swift::LoadedFile *loaded_file = llvm::cast<swift::LoadedFile>(file_unit);
            s.Printf("filename = '%s'", loaded_file->getFilename().str().c_str());
        }
            break;
    };
}

// Gets the full module name from the module passed in.

static void
GetNameFromModule (swift::Module *module, std::string &result)
{
    result.clear();
    if (module)
    {
        const char *name = module->getName().get();
        if (!name)
            return;
        result.append(name);
        const clang::Module *clang_module = module->findUnderlyingClangModule();
        
        // At present, there doesn't seem to be any way to get the full module path from the Swift side.
        if (!clang_module)
            return;
        
        for (const clang::Module *cur_module = clang_module->Parent; cur_module; cur_module = cur_module->Parent)
        {
            if (!cur_module->Name.empty())
            {
                result.insert(0, 1, '.');
                result.insert(0, cur_module->Name);
            }
        }
    }
}

// Largely lifted from swift::performAutoImport, but serves our own nefarious purposes.
bool
SwiftExpressionParser::PerformAutoImport (swift::SourceFile &source_file, bool user_imports, Error &error)
{
    Log *log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

    const std::vector<ConstString> *cu_modules = nullptr;
    
    CompileUnit *compile_unit = m_sc.comp_unit;
    
    if (compile_unit)
        cu_modules = &compile_unit->GetImportedModules();
    
    llvm::SmallVector <swift::Module::ImportedModule, 2> imported_modules;
    llvm::SmallVector <std::pair<swift::Module::ImportedModule, swift::SourceFile::ImportOptions>, 2> additional_imports;
    
    source_file.getImportedModules(imported_modules, swift::Module::ImportFilter::All);
    
    std::set<ConstString> loaded_modules;
    
    
    auto load_one_module = [this, log, &loaded_modules, &imported_modules, &additional_imports, &error] (const ConstString &module_name)
    {
        error.Clear();
        if (loaded_modules.count(module_name))
            return true;
        
        if (log)
            log->Printf("[PerformAutoImport] Importing module %s", module_name.AsCString());

        loaded_modules.insert(module_name);
        
        swift::ModuleDecl *swift_module = nullptr;
        lldb::StackFrameSP this_frame_sp (m_stack_frame_wp.lock());
        
        if (module_name == ConstString(m_swift_ast_context->GetClangImporter()->getImportedHeaderModule()->getName().str()))
            swift_module = m_swift_ast_context->GetClangImporter()->getImportedHeaderModule();
        else if (this_frame_sp)
        {
            lldb::ProcessSP process_sp(this_frame_sp->CalculateProcess());
            if (process_sp)
                swift_module = m_swift_ast_context->FindAndLoadModule (module_name, *process_sp.get(), error);
        }
        else
             swift_module = m_swift_ast_context->GetModule(module_name, error);
        
        if (!swift_module || !error.Success() || m_swift_ast_context->HasFatalErrors())
        {
            if (log)
                log->Printf("[PerformAutoImport] Couldnt import module %s: %s", module_name.AsCString(), error.AsCString());
            
            if (!swift_module || m_swift_ast_context->HasFatalErrors())
            {
                return false;
            }
        }
        
        if (log)
        {
            log->Printf("Importing %s with source files:", module_name.AsCString());
            
            for (swift::FileUnit *file_unit : swift_module->getFiles()) {
                StreamString ss;
                DescribeFileUnit(ss, file_unit);
                log->Printf("  %s", ss.GetData());
            }
        }
        
        additional_imports.push_back(std::make_pair(std::make_pair(swift::Module::AccessPathTy(), swift_module), swift::SourceFile::ImportOptions()));
        imported_modules.push_back(std::make_pair(swift::Module::AccessPathTy(), swift_module));
        
        return true;
    };
    
    if (!user_imports)
    {
        if (!load_one_module(ConstString("Swift")))
            return false;
        
        if (cu_modules)
        {
            for (const ConstString &module_name : *cu_modules)
            {
                if (!load_one_module(module_name))
                    return false;
            }
        }
    }
    else
    {
        llvm::SmallVector<swift::Module::ImportedModule, 2> parsed_imports;
        
        source_file.getImportedModules(parsed_imports, swift::Module::ImportFilter::All);
        
        SwiftPersistentExpressionState *persistent_expression_state = llvm::cast<SwiftPersistentExpressionState>(m_sc.target_sp->GetPersistentExpressionStateForLanguage(lldb::eLanguageTypeSwift));
        
        for (auto module_pair : parsed_imports)
        {
            swift::ModuleDecl *module = module_pair.second;
            if (module)
            {
                std::string module_name;
                GetNameFromModule(module, module_name);
                if (!module_name.empty())
                {
                    ConstString module_const_str(module_name);
                    if (log)
                        log->Printf ("[PerformAutoImport] Performing auto import on found module: %s.\n", module_name.c_str());
                    if (!load_one_module (module_const_str))
                        return false;
                    if (1 /* How do we tell we are in REPL or playground mode? */)
                    {
                        persistent_expression_state->AddHandLoadedModule(module_const_str);
                    }
                }
            }
        }
        
        // Finally get the hand-loaded modules from the SwiftPersistentExpressionState and load them into this context:
        if (!persistent_expression_state->RunOverHandLoadedModules(load_one_module))
            return false;
    }
    
    source_file.addImports(additional_imports);

    return true;
}

class VariableMetadataPersistent : public SwiftASTManipulatorBase::VariableMetadata
{
public:
    VariableMetadataPersistent (lldb::ExpressionVariableSP &persistent_variable_sp) :
        m_persistent_variable_sp(persistent_variable_sp)
    {
    }
    
    static constexpr unsigned Type() { return 'Pers'; }
    virtual unsigned GetType() { return Type(); }
    lldb::ExpressionVariableSP m_persistent_variable_sp;
};

class VariableMetadataVariable : public SwiftASTManipulatorBase::VariableMetadata
{
public:
    VariableMetadataVariable (lldb::VariableSP &variable_sp) :
        m_variable_sp(variable_sp)
    {
    }
    
    static constexpr unsigned Type() { return 'Vari'; }
    virtual unsigned GetType() { return Type(); }
    lldb::VariableSP m_variable_sp;
};

static CompilerType ImportType(SwiftASTContext &target_context, CompilerType source_type)
{
    SwiftASTContext *swift_ast_ctx = llvm::dyn_cast_or_null<SwiftASTContext>(source_type.GetTypeSystem());

    if (swift_ast_ctx == nullptr)
        return CompilerType();
    
    if (swift_ast_ctx == &target_context)
        return source_type;
    
    Error error, mangled_error;
    CompilerType target_type;
    
    // First try to get the type by using the mangled name,
    // That will save the mangling step ImportType would have to do:
    
    ConstString type_name = source_type.GetTypeName();
    ConstString mangled_counterpart;
    bool found_counterpart = type_name.GetMangledCounterpart (mangled_counterpart);
    if (found_counterpart)
        target_type = target_context.GetTypeFromMangledTypename(mangled_counterpart.GetCString(), mangled_error);
    
    if (!target_type.IsValid())
        target_type = target_context.ImportType (source_type, error);
    
    return target_type;
}

namespace
{
class LLDBNameLookup : public swift::SILDebuggerClient
{
public:
    LLDBNameLookup (SwiftExpressionParser &parser,
                    swift::SourceFile &source_file,
                    SwiftExpressionParser::SILVariableMap &variable_map,
                    SymbolContext &sc) :
        SILDebuggerClient(source_file.getASTContext()),
        m_parser (parser),
        m_log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS)),
        m_source_file(source_file),
        m_variable_map(variable_map),
        m_sc(sc)
    {
        source_file.getParentModule()->setDebugClient(this);
        
        if (m_sc.target_sp)
        {
            m_persistent_vars = llvm::cast<SwiftPersistentExpressionState>(m_sc.target_sp->GetPersistentExpressionStateForLanguage(lldb::eLanguageTypeSwift));
        }
    }
    
    virtual ~LLDBNameLookup ()
    {
    }
    
    virtual bool shouldGlobalize (swift::Identifier Name,
                                  swift::DeclKind Kind)
    {
        if (m_parser.GetOptions().GetREPLEnabled())
            return true;
        else
        {
            // Extensions have to be globalized, there's no way to mark them as local to the function, since their
            // name is the name of the thing being extended...
            if (Kind == swift::DeclKind::Extension)
                return true;
            
            // Operators need to be parsed at the global scope regardless of name.
            if (Kind == swift::DeclKind::Func && Name.isOperator())
                return true;
            
            const char *name_cstr = Name.get();
            if (name_cstr && name_cstr[0] == '$')
            {
                if (m_log)
                    m_log->Printf ("[LLDBNameLookup::shouldGlobalize] Returning true to globalizing %s", name_cstr);
                return true;
            }
        }
        return false;
    }
    
    virtual void didGlobalize (swift::Decl *decl)
    {
        swift::ValueDecl *value_decl = swift::dyn_cast<swift::ValueDecl>(decl);
        if (value_decl)
        {
            // It seems weird to be asking this again, but some DeclKinds must be moved to
            // the source-file level to be legal.  But we don't want to register them with
            // lldb unless they are of the kind lldb explicitly wants to globalize.
            if (shouldGlobalize(value_decl->getName(), value_decl->getKind()))
                m_staged_decls.AddDecl(value_decl, false, ConstString());
        }
    }
    
    virtual bool lookupOverrides(swift::Identifier Name,
                                 swift::DeclContext *DC,
                                 swift::SourceLoc Loc,
                                 bool IsTypeLookup,
                                 ResultVector &RV)
    {
        static unsigned counter = 0;
        unsigned count = counter++;
        
        if (m_log)
        {
            m_log->Printf("[LLDBNameLookup::lookupOverrides(%u)] Searching for %s",
                          count,
                          Name.get());
        }
        
        return false;
    }

    virtual bool lookupAdditions(swift::Identifier Name,
                                 swift::DeclContext *DC,
                                 swift::SourceLoc Loc,
                                 bool IsTypeLookup,
                                 ResultVector &RV)
    {
        static unsigned counter = 0;
        unsigned count = counter++;
        
        if (m_log)
        {
            m_log->Printf("[LLDBNameLookup::lookupAdditions (%u)] Searching for %s",
                          count,
                          Name.get());
        }
        
        ConstString name_const_str (Name.str());
        std::vector<swift::ValueDecl *> results;
        
        // First look up the matching Decl's we've made in this compile, then pass that list to the
        // persistent decls, which will only add decls it has that are NOT equivalent to the decls
        // we made locally.
        
        m_staged_decls.FindMatchingDecls(name_const_str, results);
        
        // Next look up persistent decls matching this name.  Then, if we are in the plain expression parser, and we
        // aren't looking at a debugger variable, filter out persistent results of the same kind as one found by the
        // ordinary lookup mechanism in the parser .  The problem
        // we are addressing here is the case where the user has entered the REPL while in an ordinary debugging session
        // to play around.  While there, e.g., they define a class that happens to have the same name as one in the
        // program, then in some other context "expr" will call the class they've defined, not the one in the program
        // itself would use.  Plain "expr" should behave as much like code in the program would, so we want to favor
        // entities of the same DeclKind & name from the program over ones defined in the REPL.  For function decls we
        // check the interface type and full name so we don't remove overloads that don't exist in the current scope.
        //
        // Note also, we only do this for the persistent decls.  Anything in the "staged" list has been defined in this
        // expr setting and so is more local than local.
        
        bool skip_results_with_matching_kind = !(m_parser.GetOptions().GetREPLEnabled()
                                                || m_parser.GetOptions().GetPlaygroundTransformEnabled()
                                                || (!Name.str().empty() && Name.str().front() == '$'));
        
        size_t num_external_results = RV.size();
        if (skip_results_with_matching_kind && num_external_results > 0)
        {
            std::vector<swift::ValueDecl *> persistent_results;
            m_persistent_vars->GetSwiftPersistentDecls(name_const_str, persistent_results);
            
            size_t num_persistent_results = persistent_results.size();
            for (size_t idx = 0; idx < num_persistent_results; idx++)
            {
                swift::ValueDecl *value_decl = persistent_results[idx];
                if (!value_decl)
                    continue;
                swift::DeclName value_decl_name = value_decl->getFullName();
                swift::DeclKind value_decl_kind = value_decl->getKind();
                swift::CanType value_interface_type = value_decl->getInterfaceType()->getCanonicalType();
                
                bool is_function = swift::isa<swift::AbstractFunctionDecl>(value_decl);
                
                bool skip_it = false;
                for (size_t rv_idx = 0; rv_idx < num_external_results; rv_idx++)
                {
                    if (swift::ValueDecl *rv_decl = RV[rv_idx].getValueDecl())
                    {
                        if (value_decl_kind == rv_decl->getKind())
                        {
                            if (is_function)
                            {
                                swift::DeclName rv_full_name = rv_decl->getFullName();
                                if (rv_full_name.matchesRef(value_decl_name))
                                {
                                    // If the full names match, make sure the interface types match:
                                    if (rv_decl->getInterfaceType()->getCanonicalType() == value_interface_type)
                                        skip_it = true;
                                }
                            }
                            else
                            {
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
        }
        else
        {
            m_persistent_vars->GetSwiftPersistentDecls(name_const_str, results);
        }

        for (size_t idx = 0; idx < results.size(); idx++)
        {
            swift::ValueDecl *value_decl = results[idx];
            assert(&DC->getASTContext() == &value_decl->getASTContext()); // no import required
            RV.push_back(swift::UnqualifiedLookupResult(value_decl));
        }
        
        return results.size() > 0;
    }
    
    virtual swift::SILValue emitLValueForVariable(swift::VarDecl *var,
                                                  swift::SILBuilder &builder)
    {
        SwiftSILManipulator manipulator(builder);
        
        swift::Identifier variable_name = var->getName();
        ConstString variable_const_string(variable_name.get());
        
        SwiftExpressionParser::SILVariableMap::iterator vi = m_variable_map.find(variable_const_string.AsCString());
        
        if (vi == m_variable_map.end())
            return swift::SILValue();
        
        return manipulator.emitLValueForVariable(var, vi->second);
    }
    
    SwiftPersistentExpressionState::SwiftDeclMap &
    GetStagedDecls()
    {
        return m_staged_decls;
    }
    
    virtual swift::Identifier
    getPreferredPrivateDiscriminator()
    {
        if (m_sc.comp_unit)
        {
            if (lldb_private::Module *module = m_sc.module_sp.get())
            {
                if (lldb_private::SymbolVendor *symbol_vendor = module->GetSymbolVendor())
                {
                    std::string private_discriminator_string;
                    if (symbol_vendor->GetCompileOption("-private-discriminator", private_discriminator_string, m_sc.comp_unit))
                    {
                        return m_source_file.getASTContext().getIdentifier(private_discriminator_string);
                    }
                }
            }
        }
        
        return swift::Identifier();
    }
    
private:
    SwiftExpressionParser                      &m_parser;
    Log                                        *m_log;
    swift::SourceFile                          &m_source_file;
    SwiftExpressionParser::SILVariableMap      &m_variable_map;
    SymbolContext                              m_sc;
    SwiftPersistentExpressionState            *m_persistent_vars = nullptr;
    SwiftPersistentExpressionState::SwiftDeclMap    m_staged_decls; // We stage the decls we are globalize in this map.
                                                                    // They will get copied over to the SwiftPersistentVariable
                                                                    // store if the parse succeeds.
};
} // END Anonymous namespace

static void
AddRequiredAliases(Block *block,
                   lldb::StackFrameSP &stack_frame_sp,
                   SwiftASTContext &swift_ast_context,
                   SwiftASTManipulator &manipulator,
                   const Expression::SwiftGenericInfo &generic_info)
{
    // First, emit the typealias for "$__lldb_context"
    
    do
    {
        if (!block)
            break;
        
        Function *function = block->CalculateSymbolContextFunction();
        
        if (!function)
            break;

        constexpr bool can_create = true;
        Block &function_block(function->GetBlock(can_create));
        
        lldb::VariableListSP variable_list_sp (function_block.GetBlockVariableList (true));
        
        if (!variable_list_sp)
            break;
        
        lldb::VariableSP self_var_sp (variable_list_sp->FindVariable(ConstString("self")));
        
        if (!self_var_sp)
            break;
        
        CompilerType self_type;
        
        if (stack_frame_sp)
        {
            lldb::ValueObjectSP valobj_sp = stack_frame_sp->GetValueObjectForFrameVariable(self_var_sp, lldb::eNoDynamicValues);
            
            if (valobj_sp)
                self_type = valobj_sp->GetCompilerType();
        }
        
        if (!self_type.IsValid()) {
            if (Type *type = self_var_sp->GetType()) {
                self_type = type->GetForwardCompilerType();
            }
        }
        

        if (!self_type.IsValid() || !llvm::isa<SwiftASTContext>(self_type.GetTypeSystem()))
            break;
        
        // Import before getting the unbound version, because the unbound version may not be in the mangled name map
        
        CompilerType imported_self_type = ImportType(swift_ast_context, self_type);

        if (!imported_self_type.IsValid())
            break;

        // This might be a referenced type, in which case we really want to extend the referent:
        imported_self_type = llvm::cast<SwiftASTContext>(imported_self_type.GetTypeSystem())->GetReferentType(imported_self_type);
        
        // If we are extending a generic class it's going to be a metatype, and we have to grab the instance type:
        imported_self_type = llvm::cast<SwiftASTContext>(imported_self_type.GetTypeSystem())->GetInstanceType(imported_self_type.GetOpaqueQualType());
        
        Flags imported_self_type_flags(imported_self_type.GetTypeInfo());
        
        // Get the instance type:
        if (imported_self_type_flags.AllSet(lldb::eTypeIsSwift | lldb::eTypeIsMetatype))
            imported_self_type = imported_self_type.GetInstanceType();
        
        // If 'self' is the Self archetype, resolve it to the actual metatype it is
        if (SwiftASTContext::IsSelfArchetypeType(imported_self_type))
        {
            SwiftLanguageRuntime *swift_runtime = stack_frame_sp->GetThread()->GetProcess()->GetSwiftLanguageRuntime();
            if (CompilerType concrete_self_type = swift_runtime->GetConcreteType(stack_frame_sp.get(), ConstString("Self")))
            {
                if (SwiftASTContext *concrete_self_type_ast_ctx = llvm::dyn_cast_or_null<SwiftASTContext>(concrete_self_type.GetTypeSystem()))
                {
                    imported_self_type = concrete_self_type_ast_ctx->CreateMetatypeType(concrete_self_type);
                    imported_self_type_flags.Reset(imported_self_type.GetTypeInfo());
                    imported_self_type = ImportType(swift_ast_context, imported_self_type);
                    if (imported_self_type_flags.AllSet(lldb::eTypeIsSwift | lldb::eTypeIsMetatype))
                    {
                        imported_self_type = imported_self_type.GetInstanceType();
                    }
                }
            }
            
        }
        
        swift::Type object_type = swift::Type((swift::TypeBase*)(imported_self_type.GetOpaqueQualType()))->getLValueOrInOutObjectType();
        
        if (object_type.getPointer() && (object_type.getPointer() != imported_self_type.GetOpaqueQualType()))
            imported_self_type = CompilerType(imported_self_type.GetTypeSystem(), object_type.getPointer());
        
        // If the type of 'self' is a bound generic type, get the unbound version

        bool is_generic = imported_self_type_flags.AllSet(lldb::eTypeIsSwift | lldb::eTypeIsGeneric);
        bool is_bound = imported_self_type_flags.AllSet(lldb::eTypeIsSwift | lldb::eTypeIsBound);
        
        if (is_generic)
        {
            if (is_bound)
                imported_self_type = imported_self_type.GetUnboundType();
        }
        
        // if 'self' is a weak storage type, it must be an optional.  Look through it and unpack the argument of "optional".
        
        if (swift::WeakStorageType *weak_storage_type = ((swift::TypeBase*)imported_self_type.GetOpaqueQualType())->getAs<swift::WeakStorageType>())
        {
            swift::Type referent_type = weak_storage_type->getReferentType();
            
            swift::BoundGenericEnumType *optional_type = referent_type->getAs<swift::BoundGenericEnumType>();
            
            if (!optional_type)
            {
                break;
            }
            
            swift::Type first_arg_type = optional_type->getGenericArgs()[0];
            
            swift::ClassType *self_class_type = first_arg_type->getAs<swift::ClassType>();
            
            if (!self_class_type)
            {
                break;
            }
            
            imported_self_type = CompilerType(imported_self_type.GetTypeSystem(), self_class_type);
        }

        imported_self_type_flags.Reset(imported_self_type.GetTypeInfo());
        if (imported_self_type_flags.AllClear(lldb::eTypeIsArchetype))
        {
            swift::ValueDecl *type_alias_decl = nullptr;
        
            type_alias_decl = manipulator.MakeGlobalTypealias(swift_ast_context.GetASTContext()->getIdentifier("$__lldb_context"), imported_self_type);
        
            if (!type_alias_decl)
            {
                Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
                if (log)
                    log->Printf("SEP:AddRequiredAliases: Failed to make the $__lldb_context typealias.");
            }
        }
        else
        {
            Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
            if (log)
                log->Printf("SEP:AddRequiredAliases: Failed to resolve the self archetype - could not make the $__lldb_context typealias.");
        }
        
    } while (0);
    
    // Emit the typedefs
    
    for (const Expression::SwiftGenericInfo::Binding &binding : generic_info.function_bindings)
    {
        CompilerType bound_type = binding.type;
        
        if (!llvm::isa<SwiftASTContext>(bound_type.GetTypeSystem()))
            continue;
        
        CompilerType imported_bound_type = ImportType(swift_ast_context, bound_type);
        
        if (!imported_bound_type.IsValid())
            continue;
        
        std::string alias_name("$__lldb_typeof_generic_");
        alias_name.append(binding.name);
        
        swift::ValueDecl *type_alias_decl = manipulator.MakeGlobalTypealias(swift_ast_context.GetASTContext()->getIdentifier(alias_name), imported_bound_type);
        
        if (!type_alias_decl)
            continue;
    }
}

static void
CountLocals (SymbolContext &sc,
             lldb::StackFrameSP &stack_frame_sp,
             SwiftASTContext &ast_context,
             llvm::SmallVectorImpl<SwiftASTManipulator::VariableInfo> &local_variables)
{
    std::set<ConstString> counted_names; // avoids shadowing
    
    if (!sc.block && !sc.function)
        return;
    
    Block *block = sc.block;
    Block *top_block = block->GetContainingInlinedBlock();
    
    if (!top_block)
        top_block = &sc.function->GetBlock(true);
    
    static ConstString s_self_name("self");
    
    SwiftLanguageRuntime *language_runtime = nullptr;
    ExecutionContextScope *scope = nullptr;
    
    if (stack_frame_sp)
    {
        language_runtime = stack_frame_sp->GetThread()->GetProcess()->GetSwiftLanguageRuntime();
        scope = stack_frame_sp.get();
    }

    // The module scoped variables are stored at the CompUnit level, so after we go through the current context,
    // then we have to take one more pass through the variables in the CompUnit.
    bool handling_globals = false;

    while (true) {
        VariableList variables;

        if (!handling_globals)
        {

            constexpr bool can_create = true;
            constexpr bool get_parent_variables = false;
            constexpr bool stop_if_block_is_inlined_function = true;
            
            block->AppendVariables(can_create,
                                   get_parent_variables,
                                   stop_if_block_is_inlined_function,
                                   [](Variable*) { return true; },
                                   &variables);
        }
        else
        {
            if (sc.comp_unit)
            {
                lldb::VariableListSP globals_sp = sc.comp_unit->GetVariableList(true);
                if (globals_sp)
                    variables.AddVariables(globals_sp.get());
            }
        }

        for (size_t vi = 0, ve = variables.GetSize(); vi != ve; ++vi)
        {
            lldb::VariableSP variable_sp(variables.GetVariableAtIndex(vi));
            
            const ConstString &name(variable_sp->GetName());
            const char *name_cstring = variable_sp->GetName().GetCString();
            
            if (name.IsEmpty())
                continue;
            
            if (counted_names.count(name))
                continue;
            
            CompilerType var_type;
            
            if (stack_frame_sp)
            {
                lldb::ValueObjectSP valobj_sp = stack_frame_sp->GetValueObjectForFrameVariable(variable_sp, lldb::eNoDynamicValues);
                
                if (!valobj_sp || valobj_sp->GetError().Fail())
                {
                    // Ignore the variable if we couldn't find its corresponding value object.
                    // TODO if the expression tries to use an ignored variable, produce a sensible error.
                    continue;
                }
                else
                {
                    var_type = valobj_sp->GetCompilerType();
                }
                
                if (var_type.IsValid() && !SwiftASTContext::IsFullyRealized(var_type))
                {
                    lldb::ValueObjectSP dynamic_valobj_sp = valobj_sp->GetDynamicValue(lldb::eDynamicDontRunTarget);
                    
                    if (!dynamic_valobj_sp || dynamic_valobj_sp->GetError().Fail())
                    {
                        continue;
                    }
                }
            }
            
            if (!var_type.IsValid())
            {
                Type *var_lldb_type = variable_sp->GetType();
                
                if (var_lldb_type)
                    var_type = var_lldb_type->GetFullCompilerType();
            }

            if (!var_type.IsValid())
                continue;
            
            if (!llvm::isa<SwiftASTContext>(var_type.GetTypeSystem()))
                continue;
            
            Error error;
            CompilerType target_type = ast_context.ImportType (var_type, error);
            
            // If the import failed, give up
            
            if (!target_type.IsValid())
                continue;
            
            // Make sure to resolve all archetypes in the variable type.
            
            if (language_runtime && stack_frame_sp)
                target_type = language_runtime->DoArchetypeBindingForType(*stack_frame_sp, target_type, &ast_context);


            // If we couldn't fully realize the type, then we aren't going to get very far making a local out of it,
            // so discard it here.
            Log *log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_TYPES|LIBLLDB_LOG_EXPRESSIONS));
            if (!SwiftASTContext::IsFullyRealized(target_type))
            {
                if (log)
                {
                    log->Printf ("Discarding local %s because we couldn't fully realize it, our best attempt was: %s.",
                                 name_cstring,
                                 target_type.GetTypeName().AsCString("<unknown>"));
                }
                continue;
            }

            SwiftASTManipulatorBase::VariableMetadataSP metadata_sp(new VariableMetadataVariable(variable_sp));
            
            const char *overridden_name = name_cstring;
            
            if (name == s_self_name)
            {
                overridden_name = ConstString("$__lldb_injected_self").AsCString();
                if (log)
                {
                    swift::TypeBase *swift_type = (swift::TypeBase *)target_type.GetOpaqueQualType();
                    if (swift_type)
                    {
                        std::string s;
                        llvm::raw_string_ostream ss(s);
                        swift_type->dump(ss);
                        ss.flush();
                        log->Printf("Adding injected self: type (%p) context(%p) is: %s", swift_type, ast_context.GetASTContext(), s.c_str());
                    }
                }
            }
            
            SwiftASTManipulator::VariableInfo variable_info(target_type, ast_context.GetASTContext()->getIdentifier(overridden_name), metadata_sp);

            local_variables.push_back(variable_info);
            
            counted_names.insert(name);
        }
        
        if (handling_globals)
        {
            // Okay, now we're done...
            break;
        }
        else if (block == top_block)
        {
            // Now add the containing module block, that's what holds the module globals:
            handling_globals = true;
        }
        else
            block = block->GetParent();
    }
}

static void
ResolveSpecialNames (SymbolContext &sc,
                     SwiftASTContext &ast_context,
                     llvm::SmallVectorImpl<swift::Identifier> &special_names,
                     llvm::SmallVectorImpl<SwiftASTManipulator::VariableInfo> &local_variables)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    if (!sc.target_sp)
        return;
    
    SwiftPersistentExpressionState *persistent_state = llvm::cast<SwiftPersistentExpressionState>(sc.target_sp->GetPersistentExpressionStateForLanguage(lldb::eLanguageTypeSwift));
    
    std::set<ConstString> resolved_names;
    
    for (swift::Identifier &name : special_names)
    {
        ConstString name_cs = ConstString(name.str());
        
        if (resolved_names.count(name_cs))
            continue;
        
        resolved_names.insert(name_cs);
        
        if (log)
            log->Printf("Resolving special name %s", name_cs.AsCString());
        
        lldb::ExpressionVariableSP expr_var_sp = persistent_state->GetVariable(name_cs);
        
        if (!expr_var_sp)
            continue;
        
        CompilerType var_type = expr_var_sp->GetCompilerType();
        
        if (!var_type.IsValid())
            continue;
        
        if (!llvm::isa<SwiftASTContext>(var_type.GetTypeSystem()))
            continue;
        
        CompilerType target_type;
        Error error;
        
        target_type = ast_context.ImportType (var_type, error);
        
        if (!target_type)
            continue;
        
        SwiftASTManipulatorBase::VariableMetadataSP metadata_sp(new VariableMetadataPersistent(expr_var_sp));
        
        SwiftASTManipulator::VariableInfo variable_info(target_type, ast_context.GetASTContext()->getIdentifier(name.str()), metadata_sp, !llvm::cast<SwiftExpressionVariable>(expr_var_sp.get())->GetIsModifiable());
        
        local_variables.push_back(variable_info);
    }
}

//----------------------------------------------------------------------
// Diagnostics are part of the ShintASTContext and we must enable and
// disable colorization manually in the ShintASTContext. We need to
// ensure that if we modify the setting that we restore it to what it
// was. This class helps us to do that without having to intrument all
// returns from a function, like in SwiftExpressionParser::Parse(...).
//----------------------------------------------------------------------
class SetColorize
{
public:
    SetColorize (SwiftASTContext *swift_ast, bool colorize) :
        m_swift_ast (swift_ast),
        m_saved_colorize (swift_ast->SetColorizeDiagnostics(colorize))
    {
    }
    
    ~SetColorize()
    {
        m_swift_ast->SetColorizeDiagnostics(m_saved_colorize);
    }
    
protected:
    SwiftASTContext *m_swift_ast;
    const bool m_saved_colorize;
};

unsigned
SwiftExpressionParser::Parse (DiagnosticManager &diagnostic_manager,
                              uint32_t first_line,
                              uint32_t last_line,
                              uint32_t line_offset)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    // In the case of playgrounds, we turn all rewriting functionality off.

    const bool repl = m_options.GetREPLEnabled();
    const bool playground = m_options.GetPlaygroundTransformEnabled();

    if (!m_swift_ast_context)
    {
        diagnostic_manager.PutCString(eDiagnosticSeverityError, "No AST context to parse into.  Please parse with a target.\n");
        return 1;
    }
    
    // Lazily get the clang importer if we can to make sure it exists in case we need it
    if (!m_swift_ast_context->GetClangImporter())
    {
        diagnostic_manager.PutCString(eDiagnosticSeverityError, "Swift expressions require OS X 10.10 / iOS 8 SDKs or later.\n");
        return 1;
    }
    
    if (m_swift_ast_context->HasFatalErrors())
    {
        diagnostic_manager.PutCString(eDiagnosticSeverityError, "The AST context is in a fatal error state.");
        return 1;
    }
    
    swift::ASTContext *ast_context = m_swift_ast_context->GetASTContext();
    
    if (!ast_context)
    {
        diagnostic_manager.PutCString(eDiagnosticSeverityError, "Couldn't initialize the AST context.  Please check your settings.");
        return 1;
    }
    
    if (m_swift_ast_context->HasFatalErrors())
    {
        diagnostic_manager.PutCString(eDiagnosticSeverityError, "The AST context is in a fatal error state.");
        return 1;
    }
    
    // If we are using the playground, hand import the necessary modules.
    // FIXME: We won't have to do this once the playground adds import statements for the things it needs itself.
    if (playground)
    {
        SwiftPersistentExpressionState *persistent_state = llvm::cast<SwiftPersistentExpressionState>(m_sc.target_sp->GetPersistentExpressionStateForLanguage(lldb::eLanguageTypeSwift));
        persistent_state->AddHandLoadedModule(ConstString("Swift"));
    }
    
    // TODO find a way to get contraint-solver output sent to a stream so we can log it
    // m_swift_ast_context->GetLanguageOptions().DebugConstraintSolver = true;
    
    m_swift_ast_context->ClearDiagnostics();
    
    // Make a class that will set/restore the colorize setting in the SwiftASTContext for us
    // SetColorize colorize(m_swift_ast_context, stream.GetFlags().Test(Stream::eANSIColor));
    
    m_swift_ast_context->GetLanguageOptions().DebuggerSupport = true;
    m_swift_ast_context->GetLanguageOptions().EnableDollarIdentifiers = true;  // No longer part of debugger support, set it separately.
    m_swift_ast_context->GetLanguageOptions().EnableAccessControl = (repl || playground);
    m_swift_ast_context->GetLanguageOptions().EnableTargetOSChecking = false;

    {
        lldb::StackFrameSP this_frame_sp (m_stack_frame_wp.lock());
        if (this_frame_sp)
        {
            lldb::ProcessSP process_sp(this_frame_sp->CalculateProcess());
            if (process_sp)
            {
                Error error;
                if (!process_sp->GetObjCLanguageRuntime())
                {
                    m_swift_ast_context->GetLanguageOptions().EnableObjCInterop = false;
                }
            }
        }
    }
    
    if (repl || playground)
    {
        m_swift_ast_context->GetLanguageOptions().Playground = true;
        m_swift_ast_context->GetIRGenOptions().Playground = true;
    }
    else
    {
        m_swift_ast_context->GetLanguageOptions().Playground = true;
        m_swift_ast_context->GetIRGenOptions().Playground = false;
    }

    // For the expression parser and REPL we want to relax the requirement that you put "try" in
    // front of every expression that might throw.
    if (!playground)
    {
        m_swift_ast_context->GetLanguageOptions().EnableThrowWithoutTry = true;
    }
    
    m_swift_ast_context->GetIRGenOptions().Optimize = false;
    m_swift_ast_context->GetIRGenOptions().Verify = false; // normally we'd like to verify, but unfortunately the verifier's error mode is abort().
    
    bool created_main_file = false;

    unsigned buffer_id = 0;

    const bool generate_debug_info = m_options.GetGenerateDebugInfo();
    m_swift_ast_context->SetGenerateDebugInfo(generate_debug_info ? swift::IRGenDebugInfoKind::Normal : swift::IRGenDebugInfoKind::None);
    swift::IRGenOptions &ir_gen_options = m_swift_ast_context->GetIRGenOptions();

    if (generate_debug_info)
    {
        std::string temp_source_path;
        if (ExpressionSourceCode::SaveExpressionTextToTempFile(m_expr.Text(), m_options, temp_source_path))
        {
            auto error_or_buffer_ap = llvm::MemoryBuffer::getFile(temp_source_path.c_str());
            if (error_or_buffer_ap.getError() == std::error_condition())
            {
                buffer_id = m_swift_ast_context->GetSourceManager().addNewSourceBuffer(std::move(error_or_buffer_ap.get()));
                ir_gen_options.MainInputFilename = temp_source_path;
                
                llvm::SmallString<256> source_dir(temp_source_path);
                llvm::sys::path::remove_filename(source_dir);
                ir_gen_options.DebugCompilationDir = source_dir.str();
                
                created_main_file = true;
            }
        }
    }

    if (!created_main_file)
    {
        const char *filename = repl ? "<REPL>" : "<EXPR>";
        ir_gen_options.MainInputFilename = filename;
        std::unique_ptr<llvm::MemoryBuffer> expr_buffer(llvm::MemoryBuffer::getMemBufferCopy(m_expr.Text(), filename));
        buffer_id = m_swift_ast_context->GetSourceManager().addNewSourceBuffer(std::move(expr_buffer));
    }

    char expr_name_buf[32];
    
    snprintf(expr_name_buf, sizeof(expr_name_buf), "__lldb_expr_%u", m_options.GetExpressionNumber());
    
    swift::Identifier module_id (ast_context->getIdentifier(expr_name_buf));
    swift::ModuleDecl *module = swift::Module::create(module_id, *ast_context);
    const swift::SourceFile::ImplicitModuleImportKind implicit_import_kind = swift::SourceFile::ImplicitModuleImportKind::Stdlib;
    
    m_swift_ast_context->GetCompilerInvocation().getFrontendOptions().ModuleName = expr_name_buf;
    m_swift_ast_context->GetCompilerInvocation().getIRGenOptions().ModuleName = expr_name_buf;
    
    swift::SourceFileKind source_file_kind = swift::SourceFileKind::Library;
    
    if (playground || repl)
    {
        source_file_kind = swift::SourceFileKind::Main;
    }
    
    swift::SourceFile *source_file = new (*ast_context) swift::SourceFile(*module, source_file_kind, buffer_id, implicit_import_kind);
    module->addFile(*source_file);
    
    bool done = false;
    
    SILVariableMap variable_map;
    
    LLDBNameLookup *external_lookup = new LLDBNameLookup (*this, *source_file, variable_map, m_sc);

    // FIXME: This call is here just so that the we keep the DebuggerClients alive as long as the Module we are not
    // inserting them in.
    m_swift_ast_context->AddDebuggerClient(external_lookup);

    swift::PersistentParserState persistent_state;
    
    while (!done)
    {
        swift::parseIntoSourceFile(*source_file, buffer_id, &done, nullptr, &persistent_state);

        if (m_swift_ast_context->HasErrors())
        {
            m_swift_ast_context->PrintDiagnostics(diagnostic_manager,
                                                  buffer_id,
                                                  first_line,
                                                  last_line,
                                                  line_offset);
            return 1;
        }
    }    

    // This currently crashes with Assertion failed: (BufferID != -1), function findBufferContainingLoc, file llvm/tools/swift/include/swift/Basic/SourceManager.h, line 92.
//    if (log)
//    {
//        std::string s;
//        llvm::raw_string_ostream ss(s);
//        source_file->dump(ss);
//        ss.flush();
//
//        log->Printf("Source file after parsing:");
//        log->PutCString(s.c_str());
//    }
    
    if (!done)
    {
        diagnostic_manager.PutCString(eDiagnosticSeverityError, "Parse did not consume the whole expression.");
        return 1;
    }
    
    std::unique_ptr<SwiftASTManipulator> code_manipulator;
    
    if (!playground)
    {
        code_manipulator.reset(new SwiftASTManipulator(*source_file, repl));
        
        code_manipulator->RewriteResult();
    }
    
    Error auto_import_error;
    if (!PerformAutoImport (*source_file, false, auto_import_error))
    {
        diagnostic_manager.Printf(eDiagnosticSeverityError, "in auto-import:\n%s", auto_import_error.AsCString());
        return 1;
    }

    // Swift Modules that rely on shared libraries (not frameworks) don't record the link information in the
    // swiftmodule file, so we can't really make them work without outside information.  However, in the REPL you can
    // added -L & -l options to the initial compiler startup, and we should dlopen anything that's been stuffed
    // on there and hope it will be useful later on.

    if (repl)
    {
        lldb::StackFrameSP this_frame_sp (m_stack_frame_wp.lock());
        
        if (this_frame_sp)
        {
            lldb::ProcessSP process_sp(this_frame_sp->CalculateProcess());
            if (process_sp)
            {
                Error error;
                m_swift_ast_context->LoadExtraDylibs(*process_sp.get(), error);
            }
        }
    }
    
    if (!playground && !repl)
    {
        lldb::StackFrameSP stack_frame_sp = m_stack_frame_wp.lock();

        bool local_context_is_swift = true;

        if (m_sc.block)
        {
            Function *function = m_sc.block->CalculateSymbolContextFunction();
            if (function && function->GetLanguage() != lldb::eLanguageTypeSwift)
                local_context_is_swift = false;
        }

        llvm::SmallVector<SwiftASTManipulator::VariableInfo, 5> local_variables;

        if (local_context_is_swift)
        {
            AddRequiredAliases(m_sc.block, stack_frame_sp, *m_swift_ast_context, *code_manipulator, m_expr.GetSwiftGenericInfo());
        
            // Register all local variables so that lookups to them resolve

            CountLocals(m_sc, stack_frame_sp, *m_swift_ast_context, local_variables);
        }
        
        // Register all magic variables
    
        llvm::SmallVector<swift::Identifier, 2> special_names;
        
        llvm::StringRef persistent_var_prefix;
        if (!repl)
            persistent_var_prefix = "$";
    
        code_manipulator->FindSpecialNames(special_names, persistent_var_prefix);
        
        ResolveSpecialNames(m_sc, *m_swift_ast_context, special_names, local_variables);
    
        code_manipulator->AddExternalVariables(local_variables);



        // This currently crashes with Assertion failed: (BufferID != -1), function findBufferContainingLoc, file llvm/tools/swift/include/swift/Basic/SourceManager.h, line 92.
//        if (log)
//        {
//            std::string s;
//            llvm::raw_string_ostream ss(s);
//            source_file->dump(ss);
//            ss.flush();
//        
//            log->Printf("Source file after code manipulation:");
//            log->PutCString(s.c_str());
//        }
    
        stack_frame_sp.reset();
    }
    
    swift::performNameBinding(*source_file);
    
    if (m_swift_ast_context->HasErrors())
    {
        m_swift_ast_context->PrintDiagnostics(diagnostic_manager,
                                              buffer_id,
                                              first_line,
                                              last_line,
                                              line_offset);
        return 1;
    }
    
    // Do the auto-importing after Name Binding, that's when the Imports for the source file are figured out.
    {
        std::lock_guard<std::recursive_mutex> global_context_locker(IRExecutionUnit::GetLLVMGlobalContextMutex());

        Error auto_import_error;
        if (!PerformAutoImport (*source_file, true, auto_import_error))
        {
            diagnostic_manager.Printf(eDiagnosticSeverityError, "in auto-import:\n%s", auto_import_error.AsCString());
            return 1;
        }
    }

    swift::TopLevelContext top_level_context; // not persistent because we're building source files one at a time

    swift::OptionSet<swift::TypeCheckingFlags> type_checking_options;
    
    swift::performTypeChecking(*source_file, top_level_context, type_checking_options);
    
    if (m_swift_ast_context->HasErrors())
    {
        m_swift_ast_context->PrintDiagnostics(diagnostic_manager,
                                              buffer_id,
                                              first_line,
                                              last_line,
                                              line_offset);
        return 1;
    }
    if (log)
    {
        std::string s;
        llvm::raw_string_ostream ss(s);
        source_file->dump(ss);
        ss.flush();
        
        log->Printf("Source file after type checking:");
        log->PutCString(s.c_str());
    }

    if (repl)
    {
        code_manipulator->MakeDeclarationsPublic();
    }

    Error error;
    if (!playground)
    {
        code_manipulator->FixupResultAfterTypeChecking(error);
        
        if (!error.Success())
        {
            diagnostic_manager.PutCString(eDiagnosticSeverityError, error.AsCString());
            return 1;
        }
    }
    else
    {
        swift::performPlaygroundTransform(*source_file, true);
        swift::typeCheckExternalDefinitions(*source_file);
    }
        
    // I think we now have to do the name binding and type checking again, but there should be only the result
    // variable to bind up at this point.
    
    if (log)
    {
        std::string s;
        llvm::raw_string_ostream ss(s);
        source_file->dump(ss);
        ss.flush();
        
        log->Printf("Source file after FixupResult:");
        log->PutCString(s.c_str());
    }

    if (m_sc.target_sp && !playground)
    {
        if (!code_manipulator->CheckPatternBindings()) // Do this first, so we don't pollute the persistent variable namespace
        {
            m_swift_ast_context->PrintDiagnostics(diagnostic_manager,
                                                  buffer_id,
                                                  first_line,
                                                  last_line,
                                                  line_offset);
            return 1;
        }
        
        Error error;
        SwiftASTContext *scratch_ast_context = m_sc.target_sp->GetScratchSwiftASTContext(error);
        
        if (scratch_ast_context)
        {
            SwiftPersistentExpressionState *persistent_state = llvm::dyn_cast<SwiftPersistentExpressionState>(scratch_ast_context->GetPersistentExpressionState());

            llvm::SmallVector<size_t, 1> declaration_indexes;
            code_manipulator->FindVariableDeclarations(declaration_indexes, repl);
            
            for (size_t declaration_index : declaration_indexes)
            {
                SwiftASTManipulator::VariableInfo &variable_info = code_manipulator->GetVariableInfo()[declaration_index];
                
                CompilerType imported_type = ImportType(*scratch_ast_context, variable_info.GetType());
                
                if (imported_type)
                {
                    lldb::ExpressionVariableSP persistent_variable = persistent_state->AddNewlyConstructedVariable(new SwiftExpressionVariable(m_sc.target_sp.get(),
                                                                                                                                               ConstString(variable_info.GetName().str()),
                                                                                                                                               imported_type,
                                                                                                                                               m_sc.target_sp->GetArchitecture().GetByteOrder(),
                                                                                                                                               m_sc.target_sp->GetArchitecture().GetAddressByteSize()));
                                                                                                                   
                    if (repl)
                    {
                        persistent_variable->m_flags |= ExpressionVariable::EVKeepInTarget;
                        persistent_variable->m_flags |= ExpressionVariable::EVIsProgramReference;
                    }
                    else
                    {
                        persistent_variable->m_flags |= ExpressionVariable::EVNeedsAllocation;
                        persistent_variable->m_flags |= ExpressionVariable::EVKeepInTarget;
                        llvm::cast<SwiftExpressionVariable>(persistent_variable.get())->m_swift_flags |= SwiftExpressionVariable::EVSNeedsInit;
                    }
                    
                    swift::VarDecl *decl = variable_info.GetDecl();
                    if (decl)
                    {
                        if (decl->isLet())
                        {
                            llvm::cast<SwiftExpressionVariable>(persistent_variable.get())->SetIsModifiable(false);
                        }
                        if (decl->getStorageKind() == swift::VarDecl::StorageKindTy::Computed)
                        {
                            llvm::cast<SwiftExpressionVariable>(persistent_variable.get())->SetIsComputed(true);
                        }
                    }
                    
                    
                    variable_info.m_metadata.reset(new VariableMetadataPersistent(persistent_variable));
                    
                    persistent_state->RegisterSwiftPersistentDecl(decl);
                }
            }
            
            if (repl)
            {
                llvm::SmallVector<swift::ValueDecl *, 1> non_variables;
                code_manipulator->FindNonVariableDeclarations(non_variables);
                
                for (swift::ValueDecl *decl : non_variables)
                {
                    persistent_state->RegisterSwiftPersistentDecl(decl);
                }
            }
        }
    }
    
    if (!playground && !repl)
    {
        code_manipulator->FixCaptures();
        
        // This currently crashes with Assertion failed: (BufferID != -1), function findBufferContainingLoc, file llvm/tools/swift/include/swift/Basic/SourceManager.h, line 92.
//        if (log)
//        {
//            std::string s;
//            llvm::raw_string_ostream ss(s);
//            source_file->dump(ss);
//            ss.flush();
//            
//            log->Printf("Source file after capture fixing:");
//            log->PutCString(s.c_str());
//        }
        
        if (log)
        {
            log->Printf("Variables:");
            
            for (const SwiftASTManipulatorBase::VariableInfo &variable : code_manipulator->GetVariableInfo())
            {
                StreamString ss;
                variable.Print(ss);
                log->Printf("  %s", ss.GetData());
            }
        }
    }
    
    Materializer *materializer = m_expr.GetMaterializer();
    
    if (materializer && !playground)
    {
        for (SwiftASTManipulatorBase::VariableInfo &variable : code_manipulator->GetVariableInfo())
        {
            uint64_t offset = 0;
            bool needs_init = false;

            bool is_result = variable.MetadataIs<SwiftASTManipulatorBase::VariableMetadataResult>();
            bool is_error = variable.MetadataIs<SwiftASTManipulatorBase::VariableMetadataError>();
            
            SwiftUserExpression *user_expression = static_cast<SwiftUserExpression*>(&m_expr); // this is the only thing that has a materializer
            
            if (is_result || is_error)
            {
                needs_init = true;
                
                Error error;
                
                if (repl)
                {
                    if (swift::TypeBase *swift_type = (swift::TypeBase*)variable.GetType().GetOpaqueQualType())
                    {
                        if (!swift_type->getCanonicalType()->isVoid())
                        {
                            if (is_result)
                                offset = llvm::cast<SwiftREPLMaterializer>(materializer)->AddREPLResultVariable(variable.GetType(),
                                                                                                                variable.GetDecl(),
                                                                                                                &user_expression->GetResultDelegate(),
                                                                                                                error);
                            else
                                offset = llvm::cast<SwiftREPLMaterializer>(materializer)->AddREPLResultVariable(variable.GetType(),
                                                                                                                variable.GetDecl(),
                                                                                                                &user_expression->GetErrorDelegate(),
                                                                                                                error);

                        }
                    }
                }
                else
                {
                    CompilerType actual_type (variable.GetType());
                    if (Flags(actual_type.GetTypeInfo()).AllSet(lldb::eTypeIsSwift | lldb::eTypeIsArchetype))
                    {
                        lldb::StackFrameSP stack_frame_sp = m_stack_frame_wp.lock();
                        if (stack_frame_sp &&
                            stack_frame_sp->GetThread() &&
                            stack_frame_sp->GetThread()->GetProcess())
                        {
                            SwiftLanguageRuntime *swift_runtime = stack_frame_sp->GetThread()->GetProcess()->GetSwiftLanguageRuntime();
                            if (swift_runtime)
                            {
                                actual_type = swift_runtime->GetConcreteType(stack_frame_sp.get(),
                                                                             actual_type.GetTypeName());
                                if (actual_type.IsValid())
                                    variable.SetType(actual_type);
                                else
                                    actual_type = variable.GetType();
                            }
                        }
                    }
                    swift::Type actual_swift_type = swift::Type((swift::TypeBase*)actual_type.GetOpaqueQualType());
                                        
                    swift::Type fixed_type = code_manipulator->FixupResultType(actual_swift_type, user_expression->GetLanguageFlags());

                    if (!fixed_type.isNull())
                    {
                        actual_type = CompilerType(actual_type.GetTypeSystem(), fixed_type.getPointer());
                        variable.SetType(actual_type);
                    }
                                        
                    if (is_result)
                        offset = materializer->AddResultVariable(actual_type,
                                                                 false,
                                                                 true,
                                                                 &user_expression->GetResultDelegate(),
                                                                 error);
                    else
                        offset = materializer->AddResultVariable(actual_type,
                                                                 false,
                                                                 true,
                                                                 &user_expression->GetErrorDelegate(),
                                                                 error);
                }
                
                if (!error.Success())
                {
                    diagnostic_manager.Printf(eDiagnosticSeverityError, "couldn't add %s variable to struct: %s.\n",
                                  is_result ? "result" : "error",
                                  error.AsCString());
                    return 1;
                }
                
                if (log)
                    log->Printf("Added %s variable to struct at offset %llu",
                                is_result ? "result" : "error",
                                (unsigned long long)offset);
            }
            else if (variable.MetadataIs<VariableMetadataVariable>())
            {
                Error error;
                
                VariableMetadataVariable *variable_metadata = static_cast<VariableMetadataVariable*>(variable.m_metadata.get());
                
                offset = materializer->AddVariable(variable_metadata->m_variable_sp, error);
            
                if (!error.Success())
                {
                    diagnostic_manager.Printf(eDiagnosticSeverityError, "couldn't add variable to struct: %s.\n", error.AsCString());
                    return 1;
                }
                
                if (log)
                    log->Printf("Added variable %s to struct at offset %llu", variable_metadata->m_variable_sp->GetName().AsCString(), (unsigned long long)offset);
            }
            else if (variable.MetadataIs<VariableMetadataPersistent>())
            {
                VariableMetadataPersistent *variable_metadata = static_cast<VariableMetadataPersistent*>(variable.m_metadata.get());
                
                needs_init = llvm::cast<SwiftExpressionVariable>(variable_metadata->m_persistent_variable_sp.get())->m_swift_flags & SwiftExpressionVariable::EVSNeedsInit;
                
                Error error;
                
                offset = materializer->AddPersistentVariable(variable_metadata->m_persistent_variable_sp, &user_expression->GetPersistentVariableDelegate(), error);
                
                if (!error.Success())
                {
                    diagnostic_manager.Printf(eDiagnosticSeverityError, "couldn't add variable to struct: %s.\n", error.AsCString());
                    return 1;
                }
                
                if (log)
                    log->Printf("Added persistent variable %s with flags 0x%llx to struct at offset %llu",
                                variable_metadata->m_persistent_variable_sp->GetName().AsCString(),
                                (unsigned long long)variable_metadata->m_persistent_variable_sp->m_flags,
                                (unsigned long long)offset);
            }
            
            variable_map[ConstString(variable.GetName().get()).GetCString()] = SILVariableInfo(variable.GetType(), offset, needs_init);
        }
    }
    
    std::unique_ptr<swift::SILModule> sil_module(swift::performSILGeneration(*source_file,
                                                                             m_swift_ast_context->GetSILOptions()));
    
    if (log)
    {
        std::string s;
        llvm::raw_string_ostream ss(s);
        const bool verbose = false;
        sil_module->print(ss, verbose, module);
        ss.flush();
        
        log->Printf("SIL module before linking:");
        log->PutCString(s.c_str());
    }
    
    swift::performSILLinking(sil_module.get());
    
    if (m_swift_ast_context->HasErrors())
    {
        m_swift_ast_context->PrintDiagnostics(diagnostic_manager,
                                              buffer_id,
                                              first_line,
                                              last_line,
                                              line_offset);
        return 1;
    }
    
    if (log)
    {
        std::string s;
        llvm::raw_string_ostream ss(s);
        const bool verbose = false;
        sil_module->print(ss, verbose, module);
        ss.flush();
        
        log->Printf("Generated SIL module:");
        log->PutCString(s.c_str());
    }
    
    runSILDiagnosticPasses(*sil_module);
    
    if (log)
    {
        std::string s;
        llvm::raw_string_ostream ss(s);
        const bool verbose = false;
        sil_module->print(ss, verbose, module);
        ss.flush();
        
        log->Printf("SIL module after diagnostic passes:");
        log->PutCString(s.c_str());
    }

    
    if (m_swift_ast_context->HasErrors())
    {
        m_swift_ast_context->PrintDiagnostics(diagnostic_manager,
                                              buffer_id,
                                              first_line,
                                              last_line,
                                              line_offset);
        return 1;
    }
    
    {
        std::lock_guard<std::recursive_mutex> global_context_locker(IRExecutionUnit::GetLLVMGlobalContextMutex());

        m_module = swift::performIRGeneration(m_swift_ast_context->GetIRGenOptions(),
                                              module,
                                              sil_module.get(),
                                              "lldb_module",
                                              SwiftASTContext::GetGlobalLLVMContext());
    }
    
    if (m_swift_ast_context->HasErrors())
    {
        m_swift_ast_context->PrintDiagnostics(diagnostic_manager,
                                              buffer_id,
                                              first_line,
                                              last_line,
                                              line_offset);
        return 1;
    }
    
    if (log)
    {
        std::string s;
        llvm::raw_string_ostream ss(s);
        m_module->print(ss, NULL);
        ss.flush();
        
        log->Printf("Generated IR module:");
        log->PutCString(s.c_str());
    }
    
    {
        std::lock_guard<std::recursive_mutex> global_context_locker(IRExecutionUnit::GetLLVMGlobalContextMutex());

        LLVMVerifyModule((LLVMOpaqueModule*)m_module.get(), LLVMReturnStatusAction, nullptr);
    }
    
    bool fail = m_swift_ast_context->HasErrors();
    if (!fail)
    {
        // The Parse succeeded!  Now put this module into the context's list of loaded modules,
        // and copy the Decls that were globalized as part of the parse from the staging area in the
        // external lookup object into the SwiftPersistentExpressionState.
        ast_context->LoadedModules.insert(std::make_pair(module_id, module));
        if (m_swift_ast_context)
            m_swift_ast_context->CacheModule(module);
        if (m_sc.target_sp)
        {
            SwiftPersistentExpressionState *persistent_state = llvm::cast<SwiftPersistentExpressionState>(m_sc.target_sp->GetPersistentExpressionStateForLanguage(lldb::eLanguageTypeSwift));
            persistent_state->CopyInSwiftPersistentDecls (external_lookup->GetStagedDecls());
        }
    }
    return fail ? 1 : 0;
}

static bool FindFunctionInModule (ConstString &mangled_name,
                                  llvm::Module *module,
                                  const char *orig_name,
                                  bool exact)
{
    for (llvm::Module::iterator fi = module->getFunctionList().begin(), fe = module->getFunctionList().end();
         fi != fe;
         ++fi)
    {
        if (exact)
        {
            if (!fi->getName().str().compare(orig_name))
            {
                mangled_name.SetCString(fi->getName().str().c_str());
                return true;
            }
        }
        else
        {
            if (fi->getName().str().find(orig_name) != std::string::npos)
            {
                mangled_name.SetCString(fi->getName().str().c_str());
                return true;
            }
        }
    }
    
    return false;
}

Error
SwiftExpressionParser::PrepareForExecution (lldb::addr_t &func_addr,
                                            lldb::addr_t &func_end,
                                            lldb::IRExecutionUnitSP &execution_unit_sp,
                                            ExecutionContext &exe_ctx,
                                            bool &can_interpret,
                                            ExecutionPolicy execution_policy)
{
    Error err;
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    if (!m_module)
    {
        err.SetErrorString("Can't prepare a NULL module for execution");
        return err;
    }
    
    const char *orig_name = nullptr;
    
    bool exact = false;
    
    if (m_options.GetPlaygroundTransformEnabled() ||
        m_options.GetREPLEnabled())
    {
        orig_name = "main";
        exact = true;
    }
    else
    {
        orig_name = "$__lldb_expr";
    }
    
    ConstString function_name;
        
    if (!FindFunctionInModule(function_name, m_module.get(), orig_name, exact))
    {
        err.SetErrorToGenericError();
        err.SetErrorStringWithFormat("Couldn't find %s() in the module", orig_name);
        return err;
    }
    else
    {
        if (log)
            log->Printf("Found function %s for %s", function_name.AsCString(), "$__lldb_expr");
    }

    // Retrieve an appropriate symbol context.
    SymbolContext sc;

    if (lldb::StackFrameSP frame_sp = exe_ctx.GetFrameSP())
    {
        sc = frame_sp->GetSymbolContext(lldb::eSymbolContextEverything);
    }
    else if (lldb::TargetSP target_sp = exe_ctx.GetTargetSP())
    {
        sc.target_sp = target_sp;
    }

    std::vector<std::string> features;
    
    std::unique_ptr<llvm::LLVMContext> llvm_context_up;
    m_execution_unit_sp.reset(new IRExecutionUnit (llvm_context_up,
                                                   m_module, // handed off here
                                                   function_name,
                                                   exe_ctx.GetTargetSP(),
                                                   sc,
                                                   features));
                                               
    // TODO figure out some way to work ClangExpressionDeclMap into this or do the equivalent
    //   for Swift
    
    m_execution_unit_sp->GetRunnableInfo(err, func_addr, func_end);
    
    execution_unit_sp = m_execution_unit_sp;
    m_execution_unit_sp.reset();
    
    return err;
}

bool
SwiftExpressionParser::RewriteExpression(DiagnosticManager &diagnostic_manager)
{
    // There isn't a Swift equivalent to clang::Rewriter, so we'll just use that...
    if (!m_swift_ast_context)
        return false;
    
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

    swift::SourceManager &source_manager = m_swift_ast_context->GetSourceManager();
                                                                              
    
    const DiagnosticList &diagnostics = diagnostic_manager.Diagnostics();
    size_t num_diags = diagnostics.size();
    if (num_diags == 0)
        return false;
    
    clang::RewriteBuffer rewrite_buf;
    llvm::StringRef text_ref(m_expr.Text());
    rewrite_buf.Initialize(text_ref);
    
    for (const Diagnostic *diag : diagnostic_manager.Diagnostics())
    {
        const SwiftDiagnostic *diagnostic = llvm::dyn_cast<SwiftDiagnostic>(diag);
        if (!(diagnostic && diagnostic->HasFixIts()))
            continue;
            
        const SwiftDiagnostic::FixItList &fixits = diagnostic->FixIts();
        std::vector<swift::CharSourceRange> source_ranges;
        for (const swift::DiagnosticInfo::FixIt &fixit : fixits)
        {
            const swift::CharSourceRange &range = fixit.getRange();
            swift::SourceLoc start_loc = range.getStart();
            if (!start_loc.isValid())
            {
                // getLocOffsetInBuffer will assert if you pass it an invalid location, so we have to check that first.
                if (log)
                    log->Printf("SwiftExpressionParser::RewriteExpression: ignoring fixit since "
                                "it contains an invalid source location: %s.", 
                                range.str().str().c_str());
                return false;
            }
            
            // ReplaceText can't handle replacing the same source range more than once, so we have to check that
            // before we proceed:
            if (std::find(source_ranges.begin(), source_ranges.end(), range) != source_ranges.end())
            {
                if (log)
                    log->Printf("SwiftExpressionParser::RewriteExpression: ignoring fix-it since "
                                "source range appears twice: %s.\n", 
                                range.str().str().c_str());
                return false;
            }
            else
                source_ranges.push_back(range);
            
            // ReplaceText will either assert or crash if the start_loc isn't inside the buffer it is said to
            // reside in.  That shouldn't happen, but it doesn't hurt to check before we call ReplaceText.

            auto *Buffer = source_manager.getLLVMSourceMgr().getMemoryBuffer(diagnostic->GetBufferID());
            if (!(start_loc.getOpaquePointerValue() >= Buffer->getBuffer().begin() &&
                  start_loc.getOpaquePointerValue() <= Buffer->getBuffer().end()))
            {
                if (log)
                    log->Printf("SwiftExpressionParser::RewriteExpression: ignoring fixit since "
                                "it contains a source location not in the specified buffer: %s.", 
                                range.str().str().c_str());
            }

            unsigned offset = source_manager.getLocOffsetInBuffer(range.getStart(),
                                                                  diagnostic->GetBufferID());
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

