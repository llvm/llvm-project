//===-- SymbolFileSymtab.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SymbolFileSymtab_h_
#define liblldb_SymbolFileSymtab_h_

#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/Symtab.h"
#include <vector>

class SymbolFileSymtab : public lldb_private::SymbolFile
{
public:
    //------------------------------------------------------------------
    // Static Functions
    //------------------------------------------------------------------
    static void
    Initialize();

    static void
    Terminate();

    static const char *
    GetPluginNameStatic();

    static const char *
    GetPluginDescriptionStatic();

    static lldb_private::SymbolFile*
    CreateInstance (lldb_private::ObjectFile* obj_file);

    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    SymbolFileSymtab(lldb_private::ObjectFile* obj_file);

    virtual
    ~SymbolFileSymtab();

    virtual uint32_t        CalculateAbilities ();

    //------------------------------------------------------------------
    // Compile Unit function calls
    //------------------------------------------------------------------
    virtual uint32_t
    GetNumCompileUnits();

    virtual lldb::CompUnitSP
    ParseCompileUnitAtIndex(uint32_t index);

    virtual lldb::LanguageType
    ParseCompileUnitLanguage (const lldb_private::SymbolContext& sc);

    virtual size_t
    ParseCompileUnitFunctions (const lldb_private::SymbolContext& sc);

    virtual bool
    ParseCompileUnitLineTable (const lldb_private::SymbolContext& sc);

    virtual bool
    ParseCompileUnitSupportFiles (const lldb_private::SymbolContext& sc, lldb_private::FileSpecList &support_files);

    virtual size_t
    ParseFunctionBlocks (const lldb_private::SymbolContext& sc);

    virtual size_t
    ParseTypes (const lldb_private::SymbolContext& sc);

    virtual size_t
    ParseVariablesForContext (const lldb_private::SymbolContext& sc);

    virtual lldb_private::Type*
    ResolveTypeUID(lldb::user_id_t type_uid);

    virtual lldb::clang_type_t
    ResolveClangOpaqueTypeDefinition (lldb::clang_type_t clang_Type);

    virtual uint32_t
    ResolveSymbolContext (const lldb_private::Address& so_addr, uint32_t resolve_scope, lldb_private::SymbolContext& sc);

    virtual uint32_t
    ResolveSymbolContext (const lldb_private::FileSpec& file_spec, uint32_t line, bool check_inlines, uint32_t resolve_scope, lldb_private::SymbolContextList& sc_list);

    virtual uint32_t
    FindGlobalVariables(const lldb_private::ConstString &name, const lldb_private::ClangNamespaceDecl *namespace_decl, bool append, uint32_t max_matches, lldb_private::VariableList& variables);

    virtual uint32_t
    FindGlobalVariables(const lldb_private::RegularExpression& regex, bool append, uint32_t max_matches, lldb_private::VariableList& variables);

    virtual uint32_t
    FindFunctions(const lldb_private::ConstString &name, const lldb_private::ClangNamespaceDecl *namespace_decl, uint32_t name_type_mask, bool include_inlines, bool append, lldb_private::SymbolContextList& sc_list);

    virtual uint32_t
    FindFunctions(const lldb_private::RegularExpression& regex, bool include_inlines, bool append, lldb_private::SymbolContextList& sc_list);

    virtual uint32_t
    FindTypes (const lldb_private::SymbolContext& sc,const lldb_private::ConstString &name, const lldb_private::ClangNamespaceDecl *namespace_decl, bool append, uint32_t max_matches, lldb_private::TypeList& types);

//  virtual uint32_t
//  FindTypes(const lldb_private::SymbolContext& sc, const lldb_private::RegularExpression& regex, bool append, uint32_t max_matches, lldb_private::TypeList& types);

    virtual lldb_private::ClangNamespaceDecl
    FindNamespace (const lldb_private::SymbolContext& sc, 
                   const lldb_private::ConstString &name, 
                   const lldb_private::ClangNamespaceDecl *parent_namespace_decl);

    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    virtual const char *
    GetPluginName();

    virtual const char *
    GetShortPluginName();

    virtual uint32_t
    GetPluginVersion();

protected:
    typedef std::map<lldb_private::ConstString, lldb::TypeSP> TypeMap;

    lldb_private::Symtab::IndexCollection m_source_indexes;
    lldb_private::Symtab::IndexCollection m_func_indexes;
    lldb_private::Symtab::IndexCollection m_code_indexes;
    lldb_private::Symtab::IndexCollection m_data_indexes;
    lldb_private::Symtab::NameToIndexMap m_objc_class_name_to_index;
    TypeMap m_objc_class_types;

    lldb_private::ClangASTContext &
    GetClangASTContext ();
    
private:
    DISALLOW_COPY_AND_ASSIGN (SymbolFileSymtab);
};


#endif  // liblldb_SymbolFileSymtab_h_
