//===-- DWARFASTParserGo.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DWARFASTParserGo.h"

#include "DWARFASTParserGo.h"
#include "DWARFCompileUnit.h"
#include "DWARFDebugInfo.h"
#include "DWARFDeclContext.h"
#include "DWARFDefines.h"
#include "DWARFDIE.h"
#include "DWARFDIECollection.h"
#include "SymbolFileDWARF.h"
#include "SymbolFileDWARFDebugMap.h"
#include "UniqueDWARFASTType.h"

#include "clang/Basic/Specifiers.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/Value.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/TypeList.h"

//#define ENABLE_DEBUG_PRINTF // COMMENT OUT THIS LINE PRIOR TO CHECKIN

#ifdef ENABLE_DEBUG_PRINTF
#include <stdio.h>
#define DEBUG_PRINTF(fmt, ...) printf(fmt, __VA_ARGS__)
#else
#define DEBUG_PRINTF(fmt, ...)
#endif

#define DW_AT_go_kind 0x2900
#define DW_AT_go_key 0x2901
#define DW_AT_go_elem 0x2902

using namespace lldb;
using namespace lldb_private;
DWARFASTParserGo::DWARFASTParserGo(GoASTContext &ast)
    : m_ast(ast)
{
}

DWARFASTParserGo::~DWARFASTParserGo()
{
}

TypeSP
DWARFASTParserGo::ParseTypeFromDWARF(const lldb_private::SymbolContext &sc, const DWARFDIE &die, lldb_private::Log *log,
                                     bool *type_is_new_ptr)
{
    TypeSP type_sp;

    if (type_is_new_ptr)
        *type_is_new_ptr = false;

    if (die)
    {
        SymbolFileDWARF *dwarf = die.GetDWARF();
        if (log)
        {
            dwarf->GetObjectFile()->GetModule()->LogMessage(
                log, "DWARFASTParserGo::ParseTypeFromDWARF (die = 0x%8.8x) %s name = '%s')", die.GetOffset(),
                DW_TAG_value_to_name(die.Tag()), die.GetName());
        }

        Type *type_ptr = dwarf->m_die_to_type.lookup(die.GetDIE());
        TypeList *type_list = dwarf->GetTypeList();
        if (type_ptr == NULL)
        {
            if (type_is_new_ptr)
                *type_is_new_ptr = true;

            const dw_tag_t tag = die.Tag();

            bool is_forward_declaration = false;
            DWARFAttributes attributes;
            const char *type_name_cstr = NULL;
            ConstString type_name_const_str;
            Type::ResolveState resolve_state = Type::eResolveStateUnresolved;
            uint64_t byte_size = 0;
            uint64_t go_kind = 0;
            Declaration decl;

            Type::EncodingDataType encoding_data_type = Type::eEncodingIsUID;
            CompilerType compiler_type;
            DWARFFormValue form_value;

            dw_attr_t attr;

            switch (tag)
            {
                case DW_TAG_base_type:
                case DW_TAG_pointer_type:
                case DW_TAG_typedef:
                case DW_TAG_unspecified_type:
                {
                    // Set a bit that lets us know that we are currently parsing this
                    dwarf->m_die_to_type[die.GetDIE()] = DIE_IS_BEING_PARSED;

                    const size_t num_attributes = die.GetAttributes(attributes);
                    lldb::user_id_t encoding_uid = LLDB_INVALID_UID;

                    if (num_attributes > 0)
                    {
                        uint32_t i;
                        for (i = 0; i < num_attributes; ++i)
                        {
                            attr = attributes.AttributeAtIndex(i);
                            if (attributes.ExtractFormValueAtIndex(i, form_value))
                            {
                                switch (attr)
                                {
                                    case DW_AT_name:
                                        type_name_cstr = form_value.AsCString();
                                        if (type_name_cstr)
                                            type_name_const_str.SetCString(type_name_cstr);
                                        break;
                                    case DW_AT_byte_size:
                                        byte_size = form_value.Unsigned();
                                        break;
                                    case DW_AT_encoding:
                                        // = form_value.Unsigned();
                                        break;
                                    case DW_AT_type:
                                        encoding_uid = form_value.Reference();
                                        break;
                                    case DW_AT_go_kind:
                                        go_kind = form_value.Unsigned();
                                        break;
                                    default:
                                        // Do we care about DW_AT_go_key or DW_AT_go_elem?
                                        break;
                                }
                            }
                        }
                    }

                    DEBUG_PRINTF("0x%8.8" PRIx64 ": %s (\"%s\") type => 0x%8.8lx\n", die.GetID(),
                                 DW_TAG_value_to_name(tag), type_name_cstr, encoding_uid);

                    switch (tag)
                    {
                        default:
                            break;

                        case DW_TAG_unspecified_type:
                            resolve_state = Type::eResolveStateFull;
                            compiler_type = m_ast.CreateVoidType(type_name_const_str);
                            break;

                        case DW_TAG_base_type:
                            resolve_state = Type::eResolveStateFull;
                            compiler_type = m_ast.CreateBaseType(go_kind, type_name_const_str, byte_size);
                            break;

                        case DW_TAG_pointer_type:
                            encoding_data_type = Type::eEncodingIsPointerUID;
                            break;
                        case DW_TAG_typedef:
                            encoding_data_type = Type::eEncodingIsTypedefUID;
                            CompilerType impl;
                            Type *type = dwarf->ResolveTypeUID(encoding_uid);
                            if (type)
                            {
                                if (go_kind == 0 && type->GetName() == type_name_const_str)
                                {
                                    // Go emits extra typedefs as a forward declaration. Ignore these.
                                    dwarf->m_die_to_type[die.GetDIE()] = type;
                                    return type->shared_from_this();
                                }
                                impl = type->GetForwardCompilerType();
                                compiler_type = m_ast.CreateTypedefType (go_kind, type_name_const_str, impl);
                            }
                            break;
                    }

                    type_sp.reset(new Type(die.GetID(), dwarf, type_name_const_str, byte_size,
                                           NULL, encoding_uid, encoding_data_type, &decl, compiler_type, resolve_state));

                    dwarf->m_die_to_type[die.GetDIE()] = type_sp.get();
                }
                break;

                case DW_TAG_structure_type:
                {
                    // Set a bit that lets us know that we are currently parsing this
                    dwarf->m_die_to_type[die.GetDIE()] = DIE_IS_BEING_PARSED;
                    bool byte_size_valid = false;

                    const size_t num_attributes = die.GetAttributes(attributes);
                    if (num_attributes > 0)
                    {
                        uint32_t i;
                        for (i = 0; i < num_attributes; ++i)
                        {
                            attr = attributes.AttributeAtIndex(i);
                            if (attributes.ExtractFormValueAtIndex(i, form_value))
                            {
                                switch (attr)
                                {
                                    case DW_AT_name:
                                        type_name_cstr = form_value.AsCString();
                                        type_name_const_str.SetCString(type_name_cstr);
                                        break;

                                    case DW_AT_byte_size:
                                        byte_size = form_value.Unsigned();
                                        byte_size_valid = true;
                                        break;

                                    case DW_AT_go_kind:
                                        go_kind = form_value.Unsigned();
                                        break;

                                    // TODO: Should we use SLICETYPE's DW_AT_go_elem?
                                    default:
                                        break;
                                }
                            }
                        }
                    }

                    // TODO(ribrdb): Do we need this?

                    // UniqueDWARFASTType is large, so don't create a local variables on the
                    // stack, put it on the heap. This function is often called recursively
                    // and clang isn't good and sharing the stack space for variables in different blocks.
                    std::unique_ptr<UniqueDWARFASTType> unique_ast_entry_ap(new UniqueDWARFASTType());

                    // Only try and unique the type if it has a name.
                    if (type_name_const_str &&
                        dwarf->GetUniqueDWARFASTTypeMap().Find(type_name_const_str, die, decl,
                                                               byte_size_valid ? byte_size : -1, *unique_ast_entry_ap))
                    {
                        // We have already parsed this type or from another
                        // compile unit. GCC loves to use the "one definition
                        // rule" which can result in multiple definitions
                        // of the same class over and over in each compile
                        // unit.
                        type_sp = unique_ast_entry_ap->m_type_sp;
                        if (type_sp)
                        {
                            dwarf->m_die_to_type[die.GetDIE()] = type_sp.get();
                            return type_sp;
                        }
                    }

                    DEBUG_PRINTF("0x%8.8" PRIx64 ": %s (\"%s\")\n", die.GetID(),
                                 DW_TAG_value_to_name(tag), type_name_cstr);

                    bool compiler_type_was_created = false;
                    compiler_type.SetCompilerType(&m_ast, dwarf->m_forward_decl_die_to_clang_type.lookup(die.GetDIE()));
                    if (!compiler_type)
                    {
                        compiler_type_was_created = true;
                        compiler_type = m_ast.CreateStructType(go_kind, type_name_const_str, byte_size);
                    }

                    type_sp.reset(new Type(die.GetID(), dwarf, type_name_const_str, byte_size,
                                           NULL, LLDB_INVALID_UID, Type::eEncodingIsUID, &decl, compiler_type,
                                           Type::eResolveStateForward));

                    // Add our type to the unique type map so we don't
                    // end up creating many copies of the same type over
                    // and over in the ASTContext for our module
                    unique_ast_entry_ap->m_type_sp = type_sp;
                    unique_ast_entry_ap->m_die = die;
                    unique_ast_entry_ap->m_declaration = decl;
                    unique_ast_entry_ap->m_byte_size = byte_size;
                    dwarf->GetUniqueDWARFASTTypeMap().Insert(type_name_const_str, *unique_ast_entry_ap);

                    if (!is_forward_declaration)
                    {
                        // Always start the definition for a class type so that
                        // if the class has child classes or types that require
                        // the class to be created for use as their decl contexts
                        // the class will be ready to accept these child definitions.
                        if (die.HasChildren() == false)
                        {
                            // No children for this struct/union/class, lets finish it
                            m_ast.CompleteStructType(compiler_type);
                        }
                        else if (compiler_type_was_created)
                        {
                            // Leave this as a forward declaration until we need
                            // to know the details of the type. lldb_private::Type
                            // will automatically call the SymbolFile virtual function
                            // "SymbolFileDWARF::CompleteType(Type *)"
                            // When the definition needs to be defined.
                            dwarf->m_forward_decl_die_to_clang_type[die.GetDIE()] = compiler_type.GetOpaqueQualType();
                            dwarf->m_forward_decl_clang_type_to_die[compiler_type.GetOpaqueQualType()] = die.GetDIERef();
                            // SetHasExternalStorage (compiler_type.GetOpaqueQualType(), true);
                        }
                    }
                }
                break;

                case DW_TAG_subprogram:
                case DW_TAG_subroutine_type:
                {
                    // Set a bit that lets us know that we are currently parsing this
                    dwarf->m_die_to_type[die.GetDIE()] = DIE_IS_BEING_PARSED;

                    bool is_variadic = false;
                    clang::StorageClass storage = clang::SC_None; //, Extern, Static, PrivateExtern

                    const size_t num_attributes = die.GetAttributes(attributes);
                    if (num_attributes > 0)
                    {
                        uint32_t i;
                        for (i = 0; i < num_attributes; ++i)
                        {
                            attr = attributes.AttributeAtIndex(i);
                            if (attributes.ExtractFormValueAtIndex(i, form_value))
                            {
                                switch (attr)
                                {
                                    case DW_AT_name:
                                        type_name_cstr = form_value.AsCString();
                                        type_name_const_str.SetCString(type_name_cstr);
                                        break;

                                    case DW_AT_external:
                                        if (form_value.Unsigned())
                                        {
                                            if (storage == clang::SC_None)
                                                storage = clang::SC_Extern;
                                            else
                                                storage = clang::SC_PrivateExtern;
                                        }
                                        break;

                                    case DW_AT_high_pc:
                                    case DW_AT_low_pc:
                                        break;
                                }
                            }
                        }
                    }

                    DEBUG_PRINTF("0x%8.8" PRIx64 ": %s (\"%s\")\n", die.GetID(),
                                 DW_TAG_value_to_name(tag), type_name_cstr);

                    std::vector<CompilerType> function_param_types;

                    // Parse the function children for the parameters

                    if (die.HasChildren())
                    {
                        ParseChildParameters(sc, die, is_variadic, function_param_types);
                    }

                    // compiler_type will get the function prototype clang type after this call
                    compiler_type = m_ast.CreateFunctionType(type_name_const_str, function_param_types.data(),
                                                          function_param_types.size(), is_variadic);

                    type_sp.reset(new Type(die.GetID(), dwarf, type_name_const_str, 0, NULL,
                                           LLDB_INVALID_UID, Type::eEncodingIsUID, &decl, compiler_type,
                                           Type::eResolveStateFull));
                    assert(type_sp.get());
                }
                break;

                case DW_TAG_array_type:
                {
                    // Set a bit that lets us know that we are currently parsing this
                    dwarf->m_die_to_type[die.GetDIE()] = DIE_IS_BEING_PARSED;

                    lldb::user_id_t type_die_offset = DW_INVALID_OFFSET;
                    int64_t first_index = 0;
                    uint32_t byte_stride = 0;
                    uint32_t bit_stride = 0;
                    const size_t num_attributes = die.GetAttributes(attributes);

                    if (num_attributes > 0)
                    {
                        uint32_t i;
                        for (i = 0; i < num_attributes; ++i)
                        {
                            attr = attributes.AttributeAtIndex(i);
                            if (attributes.ExtractFormValueAtIndex(i, form_value))
                            {
                                switch (attr)
                                {
                                    case DW_AT_name:
                                        type_name_cstr = form_value.AsCString();
                                        type_name_const_str.SetCString(type_name_cstr);
                                        break;

                                    case DW_AT_type:
                                        type_die_offset = form_value.Reference();
                                        break;
                                    case DW_AT_byte_size:
                                        break; // byte_size = form_value.Unsigned(); break;
                                    case DW_AT_go_kind:
                                        go_kind = form_value.Unsigned();
                                        break;
                                    default:
                                        break;
                                }
                            }
                        }

                        DEBUG_PRINTF("0x%8.8" PRIx64 ": %s (\"%s\")\n", die.GetID(),
                                     DW_TAG_value_to_name(tag), type_name_cstr);

                        Type *element_type = dwarf->ResolveTypeUID(type_die_offset);

                        if (element_type)
                        {
                            std::vector<uint64_t> element_orders;
                            ParseChildArrayInfo(sc, die, first_index, element_orders, byte_stride, bit_stride);
                            if (byte_stride == 0)
                                byte_stride = element_type->GetByteSize();
                            CompilerType array_element_type = element_type->GetForwardCompilerType();
                            if (element_orders.size() > 0)
                            {
                                if (element_orders.size() > 1)
                                    printf("golang: unsupported multi-dimensional array %s\n", type_name_cstr);
                                compiler_type =
                                    m_ast.CreateArrayType(type_name_const_str, array_element_type, element_orders[0]);
                            }
                            else
                            {
                                compiler_type = m_ast.CreateArrayType(type_name_const_str, array_element_type, 0);
                            }
                            type_sp.reset(new Type(die.GetID(), dwarf, type_name_const_str,
                                                   byte_stride, NULL, type_die_offset, Type::eEncodingIsUID, &decl,
                                                   compiler_type, Type::eResolveStateFull));
                            type_sp->SetEncodingType(element_type);
                        }
                    }
                }
                break;

                default:
                    dwarf->GetObjectFile()->GetModule()->ReportError("{0x%8.8x}: unhandled type tag 0x%4.4x (%s), "
                                                                     "please file a bug and attach the file at the "
                                                                     "start of this error message",
                                                                     die.GetOffset(), tag, DW_TAG_value_to_name(tag));
                    break;
            }

            if (type_sp.get())
            {
                DWARFDIE sc_parent_die = SymbolFileDWARF::GetParentSymbolContextDIE(die);
                dw_tag_t sc_parent_tag = sc_parent_die.Tag();

                SymbolContextScope *symbol_context_scope = NULL;
                if (sc_parent_tag == DW_TAG_compile_unit)
                {
                    symbol_context_scope = sc.comp_unit;
                }
                else if (sc.function != NULL && sc_parent_die)
                {
                    symbol_context_scope =
                        sc.function->GetBlock(true).FindBlockByID(sc_parent_die.GetID());
                    if (symbol_context_scope == NULL)
                        symbol_context_scope = sc.function;
                }

                if (symbol_context_scope != NULL)
                {
                    type_sp->SetSymbolContextScope(symbol_context_scope);
                }

                // We are ready to put this type into the uniqued list up at the module level
                type_list->Insert(type_sp);

                dwarf->m_die_to_type[die.GetDIE()] = type_sp.get();
            }
        }
        else if (type_ptr != DIE_IS_BEING_PARSED)
        {
            type_sp = type_ptr->shared_from_this();
        }
    }
    return type_sp;
}

size_t
DWARFASTParserGo::ParseChildParameters(const SymbolContext &sc,

                                       const DWARFDIE &parent_die, bool &is_variadic,
                                       std::vector<CompilerType> &function_param_types)
{
    if (!parent_die)
        return 0;

    size_t arg_idx = 0;
    for (DWARFDIE die = parent_die.GetFirstChild(); die.IsValid(); die = die.GetSibling())
    {

        dw_tag_t tag = die.Tag();
        switch (tag)
        {
            case DW_TAG_formal_parameter:
            {
                DWARFAttributes attributes;
                const size_t num_attributes = die.GetAttributes(attributes);
                if (num_attributes > 0)
                {
                    Declaration decl;
                    DWARFFormValue param_type_die_offset;

                    uint32_t i;
                    for (i = 0; i < num_attributes; ++i)
                    {
                        const dw_attr_t attr = attributes.AttributeAtIndex(i);
                        DWARFFormValue form_value;
                        if (attributes.ExtractFormValueAtIndex(i, form_value))
                        {
                            switch (attr)
                            {
                                case DW_AT_name:
                                    // = form_value.AsCString();
                                    break;
                                case DW_AT_type:
                                    param_type_die_offset = form_value;
                                    break;
                                case DW_AT_location:
                                //                          if (form_value.BlockData())
                                //                          {
                                //                              const DWARFDataExtractor& debug_info_data =
                                //                              debug_info();
                                //                              uint32_t block_length = form_value.Unsigned();
                                //                              DWARFDataExtractor location(debug_info_data,
                                //                              form_value.BlockData() - debug_info_data.GetDataStart(),
                                //                              block_length);
                                //                          }
                                //                          else
                                //                          {
                                //                          }
                                //                          break;
                                default:
                                    break;
                            }
                        }
                    }

                    Type *type = parent_die.ResolveTypeUID(DIERef(param_type_die_offset));
                    if (type)
                    {
                        function_param_types.push_back(type->GetForwardCompilerType());
                    }
                }
                arg_idx++;
            }
            break;

            case DW_TAG_unspecified_parameters:
                is_variadic = true;
                break;

            default:
                break;
        }
    }
    return arg_idx;
}

void
DWARFASTParserGo::ParseChildArrayInfo(const SymbolContext &sc, const DWARFDIE &parent_die, int64_t &first_index,
                                      std::vector<uint64_t> &element_orders, uint32_t &byte_stride,
                                      uint32_t &bit_stride)
{
    if (!parent_die)
        return;

    for (DWARFDIE die = parent_die.GetFirstChild(); die.IsValid(); die = die.GetSibling())
    {
        const dw_tag_t tag = die.Tag();
        switch (tag)
        {
            case DW_TAG_subrange_type:
            {
                DWARFAttributes attributes;
                const size_t num_child_attributes = die.GetAttributes(attributes);
                if (num_child_attributes > 0)
                {
                    uint64_t num_elements = 0;
                    uint32_t i;
                    for (i = 0; i < num_child_attributes; ++i)
                    {
                        const dw_attr_t attr = attributes.AttributeAtIndex(i);
                        DWARFFormValue form_value;
                        if (attributes.ExtractFormValueAtIndex(i, form_value))
                        {
                            switch (attr)
                            {
                                case DW_AT_count:
                                    num_elements = form_value.Unsigned();
                                    break;

                                default:
                                case DW_AT_type:
                                    break;
                            }
                        }
                    }

                    element_orders.push_back(num_elements);
                }
            }
            break;
        }
    }
}

bool
DWARFASTParserGo::CompleteTypeFromDWARF(const DWARFDIE &die, lldb_private::Type *type, CompilerType &compiler_type)
{
    if (!die)
        return false;

    const dw_tag_t tag = die.Tag();

    SymbolFileDWARF *dwarf = die.GetDWARF();
    Log *log = nullptr; // (LogChannelDWARF::GetLogIfAny(DWARF_LOG_DEBUG_INFO|DWARF_LOG_TYPE_COMPLETION));
    if (log)
        dwarf->GetObjectFile()->GetModule()->LogMessageVerboseBacktrace(
            log, "0x%8.8" PRIx64 ": %s '%s' resolving forward declaration...", die.GetID(),
            DW_TAG_value_to_name(tag), type->GetName().AsCString());
    assert(compiler_type);
    DWARFAttributes attributes;

    switch (tag)
    {
        case DW_TAG_structure_type:
        {
            {
                if (die.HasChildren())
                {
                    SymbolContext sc(die.GetLLDBCompileUnit());

                    ParseChildMembers(sc, die, compiler_type);
                }
            }
            m_ast.CompleteStructType(compiler_type);
            return (bool)compiler_type;
        }

        default:
            assert(false && "not a forward go type decl!");
            break;
    }

    return false;
}

size_t
DWARFASTParserGo::ParseChildMembers(const SymbolContext &sc, const DWARFDIE &parent_die, CompilerType &class_compiler_type)
{
    size_t count = 0;
    uint32_t member_idx = 0;

    ModuleSP module_sp = parent_die.GetDWARF()->GetObjectFile()->GetModule();
    GoASTContext *ast = llvm::dyn_cast_or_null<GoASTContext>(class_compiler_type.GetTypeSystem());
    if (ast == nullptr)
        return 0;

    for (DWARFDIE die = parent_die.GetFirstChild(); die.IsValid(); die = die.GetSibling())
    {
        dw_tag_t tag = die.Tag();

        switch (tag)
        {
            case DW_TAG_member:
            {
                DWARFAttributes attributes;
                const size_t num_attributes = die.GetAttributes(attributes);
                if (num_attributes > 0)
                {
                    Declaration decl;
                    const char *name = NULL;

                    DWARFFormValue encoding_uid;
                    uint32_t member_byte_offset = UINT32_MAX;
                    uint32_t i;
                    for (i = 0; i < num_attributes; ++i)
                    {
                        const dw_attr_t attr = attributes.AttributeAtIndex(i);
                        DWARFFormValue form_value;
                        if (attributes.ExtractFormValueAtIndex(i, form_value))
                        {
                            switch (attr)
                            {
                                case DW_AT_name:
                                    name = form_value.AsCString();
                                    break;
                                case DW_AT_type:
                                    encoding_uid = form_value;
                                    break;
                                case DW_AT_data_member_location:
                                    if (form_value.BlockData())
                                    {
                                        Value initialValue(0);
                                        Value memberOffset(0);
                                        const DWARFDataExtractor &debug_info_data =
                                            die.GetDWARF()->get_debug_info_data();
                                        uint32_t block_length = form_value.Unsigned();
                                        uint32_t block_offset = form_value.BlockData() - debug_info_data.GetDataStart();
                                        if (DWARFExpression::Evaluate(NULL, // ExecutionContext *
                                                                      NULL, // ClangExpressionVariableList *
                                                                      NULL, // ClangExpressionDeclMap *
                                                                      NULL, // RegisterContext *
                                                                      module_sp, debug_info_data, die.GetCU(),
                                                                      block_offset, block_length, eRegisterKindDWARF,
                                                                      &initialValue, NULL, memberOffset, NULL))
                                        {
                                            member_byte_offset = memberOffset.ResolveValue(NULL).UInt();
                                        }
                                    }
                                    else
                                    {
                                        // With DWARF 3 and later, if the value is an integer constant,
                                        // this form value is the offset in bytes from the beginning
                                        // of the containing entity.
                                        member_byte_offset = form_value.Unsigned();
                                    }
                                    break;

                                default:
                                    break;
                            }
                        }
                    }

                    Type *member_type = die.ResolveTypeUID(DIERef(encoding_uid));
                    if (member_type)
                    {
                        CompilerType member_go_type = member_type->GetFullCompilerType();
                        ConstString name_const_str(name);
                        m_ast.AddFieldToStruct(class_compiler_type, name_const_str, member_go_type, member_byte_offset);
                    }
                }
                ++member_idx;
            }
            break;

            default:
                break;
        }
    }

    return count;
}

Function *
DWARFASTParserGo::ParseFunctionFromDWARF(const SymbolContext &sc, const DWARFDIE &die)
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

    assert(die.Tag() == DW_TAG_subprogram);

    if (die.Tag() != DW_TAG_subprogram)
        return NULL;

    if (die.GetDIENamesAndRanges(name, mangled, func_ranges, decl_file, decl_line, decl_column, call_file, call_line,
                                 call_column, &frame_base))
    {
        // Union of all ranges in the function DIE (if the function is discontiguous)
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
            func_name.SetValue(ConstString(name), false);

            FunctionSP func_sp;
            std::unique_ptr<Declaration> decl_ap;
            if (decl_file != 0 || decl_line != 0 || decl_column != 0)
                decl_ap.reset(new Declaration(sc.comp_unit->GetSupportFiles().GetFileSpecAtIndex(decl_file), decl_line,
                                              decl_column));

            SymbolFileDWARF *dwarf = die.GetDWARF();
            // Supply the type _only_ if it has already been parsed
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
