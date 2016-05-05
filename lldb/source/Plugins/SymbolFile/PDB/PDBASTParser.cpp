//===-- PDBASTParser.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PDBASTParser.h"

#include "clang/AST/CharUnits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"

#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/ClangUtil.h"
#include "lldb/Symbol/Declaration.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/TypeSystem.h"

#include "llvm/DebugInfo/PDB/PDBSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolData.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeArray.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeBuiltin.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeEnum.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeFunctionArg.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeFunctionSig.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeTypedef.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeUDT.h"

using namespace lldb;
using namespace lldb_private;
using namespace llvm;
using namespace llvm::pdb;

namespace
{
int
TranslateUdtKind(PDB_UdtType pdb_kind)
{
    switch (pdb_kind)
    {
        case PDB_UdtType::Class:
            return clang::TTK_Class;
        case PDB_UdtType::Struct:
            return clang::TTK_Struct;
        case PDB_UdtType::Union:
            return clang::TTK_Union;
        case PDB_UdtType::Interface:
            return clang::TTK_Interface;
    }
    return clang::TTK_Class;
}

lldb::Encoding
TranslateBuiltinEncoding(PDB_BuiltinType type)
{
    switch (type)
    {
        case PDB_BuiltinType::Float:
            return lldb::eEncodingIEEE754;
        case PDB_BuiltinType::Int:
        case PDB_BuiltinType::Long:
        case PDB_BuiltinType::Char:
            return lldb::eEncodingSint;
        case PDB_BuiltinType::Bool:
        case PDB_BuiltinType::UInt:
        case PDB_BuiltinType::ULong:
        case PDB_BuiltinType::HResult:
            return lldb::eEncodingUint;
        default:
            return lldb::eEncodingInvalid;
    }
}
}

PDBASTParser::PDBASTParser(lldb_private::ClangASTContext &ast) : m_ast(ast)
{
}

PDBASTParser::~PDBASTParser()
{
}

// DebugInfoASTParser interface

lldb::TypeSP
PDBASTParser::CreateLLDBTypeFromPDBType(const PDBSymbol &type)
{
    // PDB doesn't maintain enough information to robustly rebuild the entire
    // tree, and this is most problematic when it comes to figure out the
    // right DeclContext to put a type in.  So for now, everything goes in
    // the translation unit decl as a fully qualified type.
    clang::DeclContext *tu_decl_ctx = m_ast.GetTranslationUnitDecl();
    Declaration decl;

    if (auto udt = llvm::dyn_cast<PDBSymbolTypeUDT>(&type))
    {
        AccessType access = lldb::eAccessPublic;
        PDB_UdtType udt_kind = udt->getUdtKind();

        if (udt_kind == PDB_UdtType::Class)
            access = lldb::eAccessPrivate;

        CompilerType clang_type =
            m_ast.CreateRecordType(tu_decl_ctx, access, udt->getName().c_str(), TranslateUdtKind(udt_kind),
                                   lldb::eLanguageTypeC_plus_plus, nullptr);

        m_ast.SetHasExternalStorage(clang_type.GetOpaqueQualType(), true);

        return std::make_shared<Type>(type.getSymIndexId(), m_ast.GetSymbolFile(), ConstString(udt->getName()),
                                      udt->getLength(), nullptr, LLDB_INVALID_UID, Type::eEncodingIsUID, decl,
                                      clang_type, Type::eResolveStateForward);
    }
    else if (auto enum_type = llvm::dyn_cast<PDBSymbolTypeEnum>(&type))
    {
        std::string name = enum_type->getName();
        lldb::Encoding encoding = TranslateBuiltinEncoding(enum_type->getBuiltinType());
        uint64_t bytes = enum_type->getLength();
        CompilerType builtin_type = m_ast.GetBuiltinTypeForEncodingAndBitSize(encoding, bytes * 8);

        CompilerType ast_enum = m_ast.CreateEnumerationType(name.c_str(), tu_decl_ctx, decl, builtin_type);
        auto enum_values = enum_type->findAllChildren<PDBSymbolData>();
        while (auto enum_value = enum_values->getNext())
        {
            if (enum_value->getDataKind() != PDB_DataKind::Constant)
                continue;
            AddEnumValue(ast_enum, *enum_value);
        }

        return std::make_shared<Type>(type.getSymIndexId(), m_ast.GetSymbolFile(), ConstString(name), bytes, nullptr,
                                      LLDB_INVALID_UID, Type::eEncodingIsUID, decl, ast_enum, Type::eResolveStateFull);
    }
    else if (auto type_def = llvm::dyn_cast<PDBSymbolTypeTypedef>(&type))
    {
        Type *target_type = m_ast.GetSymbolFile()->ResolveTypeUID(type_def->getTypeId());
        std::string name = type_def->getName();
        uint64_t bytes = type_def->getLength();
        if (!target_type)
            return nullptr;
        CompilerType target_ast_type = target_type->GetFullCompilerType();
        CompilerDeclContext target_decl_ctx = m_ast.GetSymbolFile()->GetDeclContextForUID(target_type->GetID());
        CompilerType ast_typedef = m_ast.CreateTypedefType(target_ast_type, name.c_str(), target_decl_ctx);
        return std::make_shared<Type>(type_def->getSymIndexId(), m_ast.GetSymbolFile(), ConstString(name), bytes,
                                      nullptr, target_type->GetID(), Type::eEncodingIsTypedefUID, decl, ast_typedef,
                                      Type::eResolveStateFull);
    }
    else if (auto func_sig = llvm::dyn_cast<PDBSymbolTypeFunctionSig>(&type))
    {
        auto arg_enum = func_sig->getArguments();
        uint32_t num_args = arg_enum->getChildCount();
        std::vector<CompilerType> arg_list(num_args);
        while (auto arg = arg_enum->getNext())
        {
            Type *arg_type = m_ast.GetSymbolFile()->ResolveTypeUID(arg->getSymIndexId());
            // If there's some error looking up one of the dependent types of this function signature, bail.
            if (!arg_type)
                return nullptr;
            CompilerType arg_ast_type = arg_type->GetFullCompilerType();
            arg_list.push_back(arg_ast_type);
        }
        auto pdb_return_type = func_sig->getReturnType();
        Type *return_type = m_ast.GetSymbolFile()->ResolveTypeUID(pdb_return_type->getSymIndexId());
        // If there's some error looking up one of the dependent types of this function signature, bail.
        if (!return_type)
            return nullptr;
        CompilerType return_ast_type = return_type->GetFullCompilerType();
        uint32_t type_quals = 0;
        if (func_sig->isConstType())
            type_quals |= clang::Qualifiers::Const;
        if (func_sig->isVolatileType())
            type_quals |= clang::Qualifiers::Volatile;
        CompilerType func_sig_ast_type =
            m_ast.CreateFunctionType(return_ast_type, &arg_list[0], num_args, false, type_quals);

        return std::make_shared<Type>(func_sig->getSymIndexId(), m_ast.GetSymbolFile(), ConstString(), 0, nullptr,
                                      LLDB_INVALID_UID, Type::eEncodingIsUID, decl, func_sig_ast_type,
                                      Type::eResolveStateFull);
    }
    else if (auto array_type = llvm::dyn_cast<PDBSymbolTypeArray>(&type))
    {
        uint32_t num_elements = array_type->getCount();
        uint32_t element_uid = array_type->getElementType()->getSymIndexId();
        uint32_t bytes = array_type->getLength();

        Type *element_type = m_ast.GetSymbolFile()->ResolveTypeUID(element_uid);
        CompilerType element_ast_type = element_type->GetFullCompilerType();
        CompilerType array_ast_type = m_ast.CreateArrayType(element_ast_type, num_elements, false);
        return std::make_shared<Type>(array_type->getSymIndexId(), m_ast.GetSymbolFile(), ConstString(), bytes, nullptr,
                                      LLDB_INVALID_UID, Type::eEncodingIsUID, decl, array_ast_type,
                                      Type::eResolveStateFull);
    }
    return nullptr;
}

bool
PDBASTParser::AddEnumValue(CompilerType enum_type, const PDBSymbolData &enum_value) const
{
    Declaration decl;
    Variant v = enum_value.getValue();
    std::string name = enum_value.getName();
    int64_t raw_value;
    switch (v.Type)
    {
        case PDB_VariantType::Int8:
            raw_value = v.Value.Int8;
            break;
        case PDB_VariantType::Int16:
            raw_value = v.Value.Int16;
            break;
        case PDB_VariantType::Int32:
            raw_value = v.Value.Int32;
            break;
        case PDB_VariantType::Int64:
            raw_value = v.Value.Int64;
            break;
        case PDB_VariantType::UInt8:
            raw_value = v.Value.UInt8;
            break;
        case PDB_VariantType::UInt16:
            raw_value = v.Value.UInt16;
            break;
        case PDB_VariantType::UInt32:
            raw_value = v.Value.UInt32;
            break;
        case PDB_VariantType::UInt64:
            raw_value = v.Value.UInt64;
            break;
        default:
            return false;
    }
    CompilerType underlying_type = m_ast.GetEnumerationIntegerType(enum_type.GetOpaqueQualType());
    uint32_t byte_size = m_ast.getASTContext()->getTypeSize(ClangUtil::GetQualType(underlying_type));
    return m_ast.AddEnumerationValueToEnumerationType(enum_type.GetOpaqueQualType(), underlying_type, decl,
                                                      name.c_str(), raw_value, byte_size * 8);
}
