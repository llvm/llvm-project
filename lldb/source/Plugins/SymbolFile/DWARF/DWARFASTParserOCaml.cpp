//===-- DWARFASTParserOCaml.cpp ---------------------------------*- C++ -*-===//

#include "DWARFASTParserOCaml.h"

#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/TypeList.h"

using namespace lldb;
using namespace lldb_private;

DWARFASTParserOCaml::DWARFASTParserOCaml (OCamlASTContext &ast) :
    m_ast (ast)
{}

DWARFASTParserOCaml::~DWARFASTParserOCaml () {}

TypeSP
DWARFASTParserOCaml::ParseBaseTypeFromDIE(const DWARFDIE &die)
{
    SymbolFileDWARF *dwarf = die.GetDWARF();
    dwarf->m_die_to_type[die.GetDIE()] = DIE_IS_BEING_PARSED;

    ConstString type_name;
    uint64_t byte_size = 0;

    DWARFAttributes attributes;
    const size_t num_attributes = die.GetAttributes(attributes);
    for (uint32_t i = 0; i < num_attributes; ++i)
    {
        DWARFFormValue form_value;
        dw_attr_t attr = attributes.AttributeAtIndex(i);
        if (attributes.ExtractFormValueAtIndex(i, form_value))
        {
            switch (attr)
            {
                case DW_AT_name:
                    type_name.SetCString(form_value.AsCString());
                    break;
                case DW_AT_byte_size:
                    byte_size = form_value.Unsigned();
                    break;
                case DW_AT_encoding:
                    break;
                default:
                    assert(false && "Unsupported attribute for DW_TAG_base_type");
            }
        }
    }

    Declaration decl;
    CompilerType compiler_type = m_ast.CreateBaseType(type_name, byte_size);
    return std::make_shared<Type>(die.GetID(), dwarf, type_name, byte_size, nullptr, LLDB_INVALID_UID,
            Type::eEncodingIsUID, decl, compiler_type, Type::eResolveStateFull);
}

lldb::TypeSP
DWARFASTParserOCaml::ParseTypeFromDWARF (const SymbolContext& sc,
                                         const DWARFDIE &die,
                                         Log *log,
                                         bool *type_is_new_ptr)
{
    if (type_is_new_ptr)
        *type_is_new_ptr = false;

    if (!die)
        return nullptr;

    SymbolFileDWARF *dwarf = die.GetDWARF();

    Type *type_ptr = dwarf->m_die_to_type.lookup(die.GetDIE());
    if (type_ptr == DIE_IS_BEING_PARSED)
        return nullptr;
    if (type_ptr != nullptr)
        return type_ptr->shared_from_this();

    TypeSP type_sp;
    if (type_is_new_ptr)
        *type_is_new_ptr = true;

    switch (die.Tag())
    {
        case DW_TAG_base_type:
            {
                type_sp = ParseBaseTypeFromDIE(die);
                break;
            }
        case DW_TAG_array_type:
            {
                break;
            }
        case DW_TAG_class_type:
            {
                break;
            }
        case DW_TAG_reference_type:
            {
                break;
            }
    }

    if (!type_sp)
        return nullptr;

    DWARFDIE sc_parent_die = SymbolFileDWARF::GetParentSymbolContextDIE(die);
    dw_tag_t sc_parent_tag = sc_parent_die.Tag();

    SymbolContextScope *symbol_context_scope = nullptr;
    if (sc_parent_tag == DW_TAG_compile_unit)
    {
        symbol_context_scope = sc.comp_unit;
    }
    else if (sc.function != nullptr && sc_parent_die)
    {
        symbol_context_scope = sc.function->GetBlock(true).FindBlockByID(sc_parent_die.GetID());
        if (symbol_context_scope == nullptr)
            symbol_context_scope = sc.function;
    }

    if (symbol_context_scope != nullptr)
        type_sp->SetSymbolContextScope(symbol_context_scope);

    dwarf->GetTypeList()->Insert(type_sp);
    dwarf->m_die_to_type[die.GetDIE()] = type_sp.get();

    return type_sp;
}

Function *
DWARFASTParserOCaml::ParseFunctionFromDWARF (const SymbolContext& sc,
                                             const DWARFDIE &die)
{
    DWARFRangeList func_ranges;
    const char *name = NULL;
    const char *mangled = NULL;
    int decl_file = 0;
    int decl_line = 0;
    int decl_column = 0;
    int call_file = 0;
    int call_line = 0;
    int call_column = 0;
    DWARFExpression frame_base(die.GetCU());

    Log *log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_LANGUAGE));

    if (die)
    {
        SymbolFileDWARF *dwarf = die.GetDWARF();
        if (log)
        {
            dwarf->GetObjectFile()->GetModule()->LogMessage(
                log, "DWARFASTParserOCaml::ParseFunctionFromDWARF (die = 0x%8.8x) %s name = '%s')", die.GetOffset(),
                DW_TAG_value_to_name(die.Tag()), die.GetName());
        }
    }

    assert(die.Tag() == DW_TAG_subprogram);

    if (die.Tag() != DW_TAG_subprogram)
        return NULL;

    if (die.GetDIENamesAndRanges(name, mangled, func_ranges, decl_file, decl_line, decl_column, call_file, call_line,
                                 call_column, &frame_base))
    {
        AddressRange func_range;
        lldb::addr_t lowest_func_addr = func_ranges.GetMinRangeBase(0);
        lldb::addr_t highest_func_addr = func_ranges.GetMaxRangeEnd(0);
        if (lowest_func_addr != LLDB_INVALID_ADDRESS && lowest_func_addr <= highest_func_addr)
        {
            ModuleSP module_sp(die.GetModule());
            func_range.GetBaseAddress().ResolveAddressUsingFileSections(lowest_func_addr, module_sp->GetSectionList());
            if (func_range.GetBaseAddress().IsValid())
                func_range.SetByteSize(highest_func_addr - lowest_func_addr);
        }

        if (func_range.GetBaseAddress().IsValid())
        {
            Mangled func_name;

            func_name.SetValue(ConstString(name), true);

            FunctionSP func_sp;
            std::unique_ptr<Declaration> decl_ap;
            if (decl_file != 0 || decl_line != 0 || decl_column != 0)
                decl_ap.reset(new Declaration(sc.comp_unit->GetSupportFiles().GetFileSpecAtIndex(decl_file), decl_line,
                                              decl_column));

            SymbolFileDWARF *dwarf = die.GetDWARF();
            Type *func_type = dwarf->m_die_to_type.lookup(die.GetDIE());

            assert(func_type == NULL || func_type != DIE_IS_BEING_PARSED);

            if (dwarf->FixupAddress(func_range.GetBaseAddress()))
            {
                const user_id_t func_user_id = die.GetID();
                func_sp.reset(new Function(sc.comp_unit,
                                           func_user_id, // UserID is the DIE offset
                                           func_user_id,
                                           func_name,
                                           func_type,
                                           func_range)); // first address range

                if (func_sp.get() != NULL)
                {
                    if (frame_base.IsValid())
                        func_sp->GetFrameBaseExpression() = frame_base;
                    sc.comp_unit->AddFunction(func_sp);
                    return func_sp.get();
                }
            }
        }
    }

    return NULL;
}

lldb_private::CompilerDeclContext
DWARFASTParserOCaml::GetDeclContextForUIDFromDWARF (const DWARFDIE &die)
{
    return CompilerDeclContext();
}

lldb_private::CompilerDeclContext
DWARFASTParserOCaml::GetDeclContextContainingUIDFromDWARF (const DWARFDIE &die)
{
    return CompilerDeclContext();
}
