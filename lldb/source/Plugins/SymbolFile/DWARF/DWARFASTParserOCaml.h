//===-- DWARFASTParserOCaml.h -----------------------------------*- C++ -*-===//

#ifndef SymbolFileDWARF_DWARFASTParserOCaml_h_
#define SymbolFileDWARF_DWARFASTParserOCaml_h_

#include "DWARFASTParser.h"
#include "DWARFCompileUnit.h"
#include "DWARFDebugInfo.h"
#include "DWARFDIE.h"
#include "DWARFDefines.h"
#include "SymbolFileDWARF.h"

#include "lldb/Symbol/OCamlASTContext.h"

class DWARFDebugInfoEntry;
class DWARFDIECollection;

class DWARFASTParserOCaml : public DWARFASTParser
{
public:
    DWARFASTParserOCaml (lldb_private::OCamlASTContext &ast);

    virtual ~DWARFASTParserOCaml ();

    lldb::TypeSP
    ParseBaseTypeFromDIE(const DWARFDIE &die);

    lldb::TypeSP
    ParseTypeFromDWARF (const lldb_private::SymbolContext& sc,
                        const DWARFDIE &die,
                        lldb_private::Log *log,
                        bool *type_is_new_ptr) override;

    lldb_private::Function *
    ParseFunctionFromDWARF (const lldb_private::SymbolContext& sc,
                            const DWARFDIE &die) override;

    bool
    CompleteTypeFromDWARF (const DWARFDIE &die,
                           lldb_private::Type *type,
                           lldb_private::CompilerType &compiler_type) override { return false; }

    lldb_private::CompilerDecl
    GetDeclForUIDFromDWARF (const DWARFDIE &die) override { return lldb_private::CompilerDecl(); }

    lldb_private::CompilerDeclContext
    GetDeclContextForUIDFromDWARF (const DWARFDIE &die) override;

    lldb_private::CompilerDeclContext
    GetDeclContextContainingUIDFromDWARF (const DWARFDIE &die) override;

    std::vector<DWARFDIE>
    GetDIEForDeclContext (lldb_private::CompilerDeclContext decl_context) override { return {}; }

protected:

    lldb_private::OCamlASTContext &m_ast;
};

#endif  // SymbolFileDWARF_DWARFASTParserOCaml_h_
