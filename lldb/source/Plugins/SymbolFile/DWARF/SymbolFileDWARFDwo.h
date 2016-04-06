//===-- SymbolFileDWARFDwo.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef SymbolFileDWARFDwo_SymbolFileDWARFDwo_h_
#define SymbolFileDWARFDwo_SymbolFileDWARFDwo_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "SymbolFileDWARF.h"

class SymbolFileDWARFDwo : public SymbolFileDWARF
{
public:
    SymbolFileDWARFDwo(lldb::ObjectFileSP objfile, DWARFCompileUnit* dwarf_cu);

    ~SymbolFileDWARFDwo() override = default;
    
    lldb::CompUnitSP
    ParseCompileUnit(DWARFCompileUnit* dwarf_cu, uint32_t cu_idx) override;

    DWARFCompileUnit*
    GetCompileUnit();

    DWARFCompileUnit*
    GetDWARFCompileUnit(lldb_private::CompileUnit *comp_unit) override;

    lldb_private::DWARFExpression::LocationListFormat
    GetLocationListFormat() const override;

    lldb_private::TypeSystem*
    GetTypeSystemForLanguage(lldb::LanguageType language) override;

    DWARFDIE
    GetDIE(const DIERef &die_ref) override;

    std::unique_ptr<SymbolFileDWARFDwo>
    GetDwoSymbolFileForCompileUnit(DWARFCompileUnit &dwarf_cu, const DWARFDebugInfoEntry &cu_die) override
    {
        return nullptr;
    }

protected:
    void
    LoadSectionData (lldb::SectionType sect_type, lldb_private::DWARFDataExtractor& data) override;

    DIEToTypePtr&
    GetDIEToType() override;

    DIEToVariableSP&
    GetDIEToVariable() override;
    
    DIEToClangType&
    GetForwardDeclDieToClangType() override;

    ClangTypeToDIE&
    GetForwardDeclClangTypeToDie() override;

    UniqueDWARFASTTypeMap&
    GetUniqueDWARFASTTypeMap() override;

    lldb::TypeSP
    FindDefinitionTypeForDWARFDeclContext (const DWARFDeclContext &die_decl_ctx) override;

    SymbolFileDWARF*
    GetBaseSymbolFile();

    lldb::ObjectFileSP m_obj_file_sp;
    DWARFCompileUnit* m_base_dwarf_cu;
};

#endif // SymbolFileDWARFDwo_SymbolFileDWARFDwo_h_
