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

// C Includes
// C++ Includes
#include <vector>

// Other libraries and framework includes
// Project includes
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/Symtab.h"

class SymbolFileSymtab : public lldb_private::SymbolFile
{
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    SymbolFileSymtab(lldb_private::ObjectFile* obj_file);

    ~SymbolFileSymtab() override;

    //------------------------------------------------------------------
    // Static Functions
    //------------------------------------------------------------------
    static void
    Initialize();

    static void
    Terminate();

    static lldb_private::ConstString
    GetPluginNameStatic();

    static const char *
    GetPluginDescriptionStatic();

    static lldb_private::SymbolFile*
    CreateInstance (lldb_private::ObjectFile* obj_file);

    uint32_t
    CalculateAbilities() override;

    //------------------------------------------------------------------
    // Compile Unit function calls
    //------------------------------------------------------------------
    uint32_t
    GetNumCompileUnits() override;

    lldb::CompUnitSP
    ParseCompileUnitAtIndex(uint32_t index) override;

    lldb::LanguageType
    ParseCompileUnitLanguage(const lldb_private::SymbolContext& sc) override;

    size_t
    ParseCompileUnitFunctions(const lldb_private::SymbolContext& sc) override;

    bool
    ParseCompileUnitLineTable(const lldb_private::SymbolContext& sc) override;

    bool
    ParseCompileUnitDebugMacros(const lldb_private::SymbolContext& sc) override;

    bool
    ParseCompileUnitSupportFiles(const lldb_private::SymbolContext& sc,
                                 lldb_private::FileSpecList &support_files) override;
    
    bool
    ParseImportedModules(const lldb_private::SymbolContext &sc,
                         std::vector<lldb_private::ConstString> &imported_modules) override;

    size_t
    ParseFunctionBlocks(const lldb_private::SymbolContext& sc) override;

    size_t
    ParseTypes(const lldb_private::SymbolContext& sc) override;

    size_t
    ParseVariablesForContext(const lldb_private::SymbolContext& sc) override;

    lldb_private::Type*
    ResolveTypeUID(lldb::user_id_t type_uid) override;

    bool
    CompleteType(lldb_private::CompilerType& compiler_type) override;

    uint32_t
    ResolveSymbolContext(const lldb_private::Address& so_addr,
                         uint32_t resolve_scope,
                         lldb_private::SymbolContext& sc) override;

    size_t
    GetTypes(lldb_private::SymbolContextScope *sc_scope,
             uint32_t type_mask,
             lldb_private::TypeList &type_list) override;

    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    lldb_private::ConstString
    GetPluginName() override;

    uint32_t
    GetPluginVersion() override;

protected:
    typedef std::map<lldb_private::ConstString, lldb::TypeSP> TypeMap;

    lldb_private::Symtab::IndexCollection m_source_indexes;
    lldb_private::Symtab::IndexCollection m_func_indexes;
    lldb_private::Symtab::IndexCollection m_code_indexes;
    lldb_private::Symtab::IndexCollection m_data_indexes;
    lldb_private::Symtab::NameToIndexMap m_objc_class_name_to_index;
    TypeMap m_objc_class_types;
    
private:
    DISALLOW_COPY_AND_ASSIGN (SymbolFileSymtab);
};

#endif // liblldb_SymbolFileSymtab_h_
