//===-- ClangASTType.cpp ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/ClangASTType.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Type.h"

#include "clang/Basic/Builtins.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"

#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/raw_ostream.h"

#include "lldb/Core/ConstString.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"

using namespace lldb;
using namespace lldb_private;


ClangASTType::~ClangASTType()
{
}

std::string
ClangASTType::GetTypeNameForQualType (clang::ASTContext *ast, clang::QualType qual_type)
{
    std::string type_name;
    
    clang::PrintingPolicy printing_policy (ast->getPrintingPolicy());
    printing_policy.SuppressTagKeyword = true;
    const clang::TypedefType *typedef_type = qual_type->getAs<clang::TypedefType>();
    if (typedef_type)
    {
        const clang::TypedefNameDecl *typedef_decl = typedef_type->getDecl();
        type_name = typedef_decl->getQualifiedNameAsString(printing_policy);
    }
    else
    {
        type_name = qual_type.getAsString(printing_policy);
    }
    return type_name;
}

std::string
ClangASTType::GetTypeNameForOpaqueQualType (clang::ASTContext *ast, clang_type_t opaque_qual_type)
{
    return GetTypeNameForQualType (ast, clang::QualType::getFromOpaquePtr(opaque_qual_type));
}


ConstString
ClangASTType::GetConstTypeName ()
{
    // TODO: verify if we actually need to complete a type just to get its type name????
    if (!ClangASTContext::GetCompleteType (this->m_ast, this->m_type))
        return ConstString("<invalid>");
    return GetConstTypeName (m_ast, m_type);
}

ConstString
ClangASTType::GetConstQualifiedTypeName ()
{
    // TODO: verify if we actually need to complete a type just to get its fully qualified type name????
    if (!ClangASTContext::GetCompleteType (this->m_ast, this->m_type))
        return ConstString("<invalid>");
    return GetConstQualifiedTypeName (m_ast, m_type);
}

ConstString
ClangASTType::GetConstQualifiedTypeName (clang::ASTContext *ast, clang_type_t clang_type)
{
    if (ast == NULL || clang_type == NULL)
        return ConstString("<invalid>");

    return ConstString (GetTypeNameForQualType (ast, clang::QualType::getFromOpaquePtr(clang_type)).c_str());
}


ConstString
ClangASTType::GetConstTypeName (clang::ASTContext *ast, clang_type_t clang_type)
{
    if (!clang_type)
        return ConstString("<invalid>");
    
    std::string type_name (GetTypeNameForOpaqueQualType(ast, clang_type));
    ConstString const_type_name;
    if (type_name.empty())
        const_type_name.SetCString ("<invalid>");
    else
        const_type_name.SetCString(type_name.c_str());
    return const_type_name;
}

clang_type_t
ClangASTType::GetPointeeType ()
{
    return GetPointeeType (m_type);
}

clang_type_t
ClangASTType::GetPointeeType (clang_type_t clang_type)
{
    if (clang_type)
    {
        clang::QualType qual_type(clang::QualType::getFromOpaquePtr(clang_type));
        
        return qual_type.getTypePtr()->getPointeeType().getAsOpaquePtr();
    }
    return NULL;
}

lldb::clang_type_t
ClangASTType::GetArrayElementType (uint32_t& stride)
{
    return GetArrayElementType(m_ast, m_type, stride);
}

lldb::clang_type_t
ClangASTType::GetArrayElementType (clang::ASTContext* ast,
                                   lldb::clang_type_t opaque_clang_qual_type,
                                   uint32_t& stride)
{
    if (opaque_clang_qual_type)
    {
        clang::QualType qual_type(clang::QualType::getFromOpaquePtr(opaque_clang_qual_type));
        
        lldb::clang_type_t ret_type = qual_type.getTypePtr()->getArrayElementTypeNoTypeQual()->getCanonicalTypeUnqualified().getAsOpaquePtr();
        
        // TODO: the real stride will be >= this value.. find the real one!
        stride = GetTypeByteSize(ast, ret_type);
        
        return ret_type;
        
    }
    return NULL;

}

lldb::clang_type_t
ClangASTType::GetPointerType () const
{
    return GetPointerType (m_ast, m_type);
}

lldb::clang_type_t
ClangASTType::GetPointerType (clang::ASTContext *ast_context,
                              lldb::clang_type_t opaque_clang_qual_type)
{
    if (opaque_clang_qual_type)
    {
        clang::QualType qual_type(clang::QualType::getFromOpaquePtr(opaque_clang_qual_type));
        
        return ast_context->getPointerType(qual_type).getAsOpaquePtr();
    }
    return NULL;
}

lldb::Encoding
ClangASTType::GetEncoding (uint32_t &count)
{
    return GetEncoding(m_type, count);
}


lldb::LanguageType
ClangASTType::GetMinimumLanguage ()
{
    return ClangASTType::GetMinimumLanguage (m_ast,
                                             m_type);
}

lldb::TypeClass
ClangASTType::GetTypeClass (clang::ASTContext *ast_context, lldb::clang_type_t clang_type)
{
    if (clang_type == NULL)
        return lldb::eTypeClassInvalid;

    clang::QualType qual_type(clang::QualType::getFromOpaquePtr(clang_type));
    
    switch (qual_type->getTypeClass())
    {
        case clang::Type::UnaryTransform:           break;
        case clang::Type::FunctionNoProto:          return lldb::eTypeClassFunction;
        case clang::Type::FunctionProto:            return lldb::eTypeClassFunction;
        case clang::Type::IncompleteArray:          return lldb::eTypeClassArray;
        case clang::Type::VariableArray:            return lldb::eTypeClassArray;
        case clang::Type::ConstantArray:            return lldb::eTypeClassArray;
        case clang::Type::DependentSizedArray:      return lldb::eTypeClassArray;
        case clang::Type::DependentSizedExtVector:  return lldb::eTypeClassVector;
        case clang::Type::ExtVector:                return lldb::eTypeClassVector;
        case clang::Type::Vector:                   return lldb::eTypeClassVector;
        case clang::Type::Builtin:                  return lldb::eTypeClassBuiltin;
        case clang::Type::ObjCObjectPointer:        return lldb::eTypeClassObjCObjectPointer;
        case clang::Type::BlockPointer:             return lldb::eTypeClassBlockPointer;
        case clang::Type::Pointer:                  return lldb::eTypeClassPointer;
        case clang::Type::LValueReference:          return lldb::eTypeClassReference;
        case clang::Type::RValueReference:          return lldb::eTypeClassReference;
        case clang::Type::MemberPointer:            return lldb::eTypeClassMemberPointer;
        case clang::Type::Complex:
            if (qual_type->isComplexType())
                return lldb::eTypeClassComplexFloat;
            else
                return lldb::eTypeClassComplexInteger;
        case clang::Type::ObjCObject:               return lldb::eTypeClassObjCObject;
        case clang::Type::ObjCInterface:            return lldb::eTypeClassObjCInterface;
        case clang::Type::Record:
            if (ClangASTContext::GetCompleteType (ast_context, clang_type))
            {
                const clang::RecordType *record_type = llvm::cast<clang::RecordType>(qual_type.getTypePtr());
                const clang::RecordDecl *record_decl = record_type->getDecl();
                if (record_decl->isUnion())
                    return lldb::eTypeClassUnion;
                else if (record_decl->isStruct())
                    return lldb::eTypeClassStruct;
                else
                    return lldb::eTypeClassClass;
            }
            break;
        case clang::Type::Enum:                     return lldb::eTypeClassEnumeration;
        case clang::Type::Typedef:                  return lldb::eTypeClassTypedef;
        case clang::Type::UnresolvedUsing:          break;
        case clang::Type::Paren:                    break;
        case clang::Type::Elaborated:               break;
        case clang::Type::Attributed:               break;
        case clang::Type::TemplateTypeParm:         break;
        case clang::Type::SubstTemplateTypeParm:    break;
        case clang::Type::SubstTemplateTypeParmPack:break;
        case clang::Type::Auto:                     break;
        case clang::Type::InjectedClassName:        break;
        case clang::Type::DependentName:            break;
        case clang::Type::DependentTemplateSpecialization: break;
        case clang::Type::PackExpansion:            break;
            
        case clang::Type::TypeOfExpr:               break;
        case clang::Type::TypeOf:                   break;
        case clang::Type::Decltype:                 break;
        case clang::Type::TemplateSpecialization:   break;
        case clang::Type::Atomic:                   break;
    }
    // We don't know hot to display this type...
    return lldb::eTypeClassOther;

}


lldb::LanguageType
ClangASTType::GetMinimumLanguage (clang::ASTContext *ctx,
                                  lldb::clang_type_t clang_type)
{
    if (clang_type == NULL)
        return lldb::eLanguageTypeC;

    // If the type is a reference, then resolve it to what it refers to first:     
    clang::QualType qual_type (clang::QualType::getFromOpaquePtr(clang_type).getNonReferenceType());
    if (qual_type->isAnyPointerType())
    {
        if (qual_type->isObjCObjectPointerType())
            return lldb::eLanguageTypeObjC;
        
        clang::QualType pointee_type (qual_type->getPointeeType());
        if (pointee_type->getPointeeCXXRecordDecl() != NULL)
            return lldb::eLanguageTypeC_plus_plus;
        if (pointee_type->isObjCObjectOrInterfaceType())
            return lldb::eLanguageTypeObjC;
        if (pointee_type->isObjCClassType())
            return lldb::eLanguageTypeObjC;
        if (pointee_type.getTypePtr() == ctx->ObjCBuiltinIdTy.getTypePtr())
            return lldb::eLanguageTypeObjC;
    }
    else
    {
        if (qual_type->isObjCObjectOrInterfaceType())
            return lldb::eLanguageTypeObjC;
        if (qual_type->getAsCXXRecordDecl())
            return lldb::eLanguageTypeC_plus_plus;
        switch (qual_type->getTypeClass())
        {
        default:
                break;
        case clang::Type::Builtin:
          switch (llvm::cast<clang::BuiltinType>(qual_type)->getKind())
            {
                default:
                case clang::BuiltinType::Void:
                case clang::BuiltinType::Bool:
                case clang::BuiltinType::Char_U:
                case clang::BuiltinType::UChar:
                case clang::BuiltinType::WChar_U:
                case clang::BuiltinType::Char16:
                case clang::BuiltinType::Char32:
                case clang::BuiltinType::UShort:
                case clang::BuiltinType::UInt:
                case clang::BuiltinType::ULong:
                case clang::BuiltinType::ULongLong:
                case clang::BuiltinType::UInt128:
                case clang::BuiltinType::Char_S:
                case clang::BuiltinType::SChar:
                case clang::BuiltinType::WChar_S:
                case clang::BuiltinType::Short:
                case clang::BuiltinType::Int:
                case clang::BuiltinType::Long:
                case clang::BuiltinType::LongLong:
                case clang::BuiltinType::Int128:
                case clang::BuiltinType::Float:
                case clang::BuiltinType::Double:
                case clang::BuiltinType::LongDouble:
                    break;

                case clang::BuiltinType::NullPtr:   
                    return eLanguageTypeC_plus_plus;
                    
                case clang::BuiltinType::ObjCId:
                case clang::BuiltinType::ObjCClass:
                case clang::BuiltinType::ObjCSel:   
                    return eLanguageTypeObjC;

                case clang::BuiltinType::Dependent:
                case clang::BuiltinType::Overload:
                case clang::BuiltinType::BoundMember:
                case clang::BuiltinType::UnknownAny:
                    break;
            }
            break;
        case clang::Type::Typedef:
            return GetMinimumLanguage(ctx,
                                      llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr());
        }
    }
    return lldb::eLanguageTypeC;
}

lldb::Encoding
ClangASTType::GetEncoding (clang_type_t clang_type, uint32_t &count)
{
    count = 1;
    clang::QualType qual_type(clang::QualType::getFromOpaquePtr(clang_type));

    switch (qual_type->getTypeClass())
    {
    case clang::Type::UnaryTransform:
        break;
            
    case clang::Type::FunctionNoProto:
    case clang::Type::FunctionProto:
        break;

    case clang::Type::IncompleteArray:
    case clang::Type::VariableArray:
        break;

    case clang::Type::ConstantArray:
        break;

    case clang::Type::ExtVector:
    case clang::Type::Vector:
        // TODO: Set this to more than one???
        break;

    case clang::Type::Builtin:
        switch (llvm::cast<clang::BuiltinType>(qual_type)->getKind())
        {
        default: assert(0 && "Unknown builtin type!");
        case clang::BuiltinType::Void:
            break;

        case clang::BuiltinType::Bool:
        case clang::BuiltinType::Char_S:
        case clang::BuiltinType::SChar:
        case clang::BuiltinType::WChar_S:
        case clang::BuiltinType::Char16:
        case clang::BuiltinType::Char32:
        case clang::BuiltinType::Short:
        case clang::BuiltinType::Int:
        case clang::BuiltinType::Long:
        case clang::BuiltinType::LongLong:
        case clang::BuiltinType::Int128:        return lldb::eEncodingSint;

        case clang::BuiltinType::Char_U:
        case clang::BuiltinType::UChar:
        case clang::BuiltinType::WChar_U:
        case clang::BuiltinType::UShort:
        case clang::BuiltinType::UInt:
        case clang::BuiltinType::ULong:
        case clang::BuiltinType::ULongLong:
        case clang::BuiltinType::UInt128:       return lldb::eEncodingUint;

        case clang::BuiltinType::Float:
        case clang::BuiltinType::Double:
        case clang::BuiltinType::LongDouble:    return lldb::eEncodingIEEE754;
        
        case clang::BuiltinType::ObjCClass:
        case clang::BuiltinType::ObjCId:
        case clang::BuiltinType::ObjCSel:       return lldb::eEncodingUint;

        case clang::BuiltinType::NullPtr:       return lldb::eEncodingUint;
        }
        break;
    // All pointer types are represented as unsigned integer encodings.
    // We may nee to add a eEncodingPointer if we ever need to know the
    // difference
    case clang::Type::ObjCObjectPointer:
    case clang::Type::BlockPointer:
    case clang::Type::Pointer:
    case clang::Type::LValueReference:
    case clang::Type::RValueReference:
    case clang::Type::MemberPointer:            return lldb::eEncodingUint;
    case clang::Type::Complex:
        {
            lldb::Encoding encoding = lldb::eEncodingIEEE754;
            if (qual_type->isComplexType())
                encoding = lldb::eEncodingIEEE754;
            else
            {
                const clang::ComplexType *complex_type = qual_type->getAsComplexIntegerType();
                if (complex_type)
                    encoding = GetEncoding (complex_type->getElementType().getAsOpaquePtr(), count);
                else 
                    encoding = lldb::eEncodingSint;
            }
            count = 2;
            return encoding;
        }

    case clang::Type::ObjCInterface:            break;
    case clang::Type::Record:                   break;
    case clang::Type::Enum:                     return lldb::eEncodingSint;
    case clang::Type::Typedef:
            return GetEncoding(llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr(), count);
        break;

    case clang::Type::DependentSizedArray:
    case clang::Type::DependentSizedExtVector:
    case clang::Type::UnresolvedUsing:
    case clang::Type::Paren:
    case clang::Type::Elaborated:
    case clang::Type::Attributed:
    case clang::Type::TemplateTypeParm:
    case clang::Type::SubstTemplateTypeParm:
    case clang::Type::SubstTemplateTypeParmPack:
    case clang::Type::Auto:
    case clang::Type::InjectedClassName:
    case clang::Type::DependentName:
    case clang::Type::DependentTemplateSpecialization:
    case clang::Type::PackExpansion:
    case clang::Type::ObjCObject:
            
    case clang::Type::TypeOfExpr:
    case clang::Type::TypeOf:
    case clang::Type::Decltype:
    case clang::Type::TemplateSpecialization:
    case clang::Type::Atomic:
        break;

    }
    count = 0;
    return lldb::eEncodingInvalid;
}

lldb::Format
ClangASTType::GetFormat ()
{
    return GetFormat (m_type);
}

lldb::Format
ClangASTType::GetFormat (clang_type_t clang_type)
{
    if (clang_type == NULL)
        return lldb::eFormatDefault;
        
    clang::QualType qual_type(clang::QualType::getFromOpaquePtr(clang_type));

    switch (qual_type->getTypeClass())
    {
    case clang::Type::UnaryTransform:
        break;
        
    case clang::Type::FunctionNoProto:
    case clang::Type::FunctionProto:
        break;

    case clang::Type::IncompleteArray:
    case clang::Type::VariableArray:
        break;

    case clang::Type::ConstantArray:
        break;

    case clang::Type::ExtVector:
    case clang::Type::Vector:
        break;

    case clang::Type::Builtin:
        switch (llvm::cast<clang::BuiltinType>(qual_type)->getKind())
        {
        //default: assert(0 && "Unknown builtin type!");
        case clang::BuiltinType::UnknownAny:
        case clang::BuiltinType::Void:
        case clang::BuiltinType::BoundMember:
            break;

        case clang::BuiltinType::Bool:          return lldb::eFormatBoolean;
        case clang::BuiltinType::Char_S:
        case clang::BuiltinType::SChar:
        case clang::BuiltinType::WChar_S:
        case clang::BuiltinType::Char_U:
        case clang::BuiltinType::UChar:
        case clang::BuiltinType::WChar_U:       return lldb::eFormatChar;
        case clang::BuiltinType::Char16:        return lldb::eFormatUnicode16;
        case clang::BuiltinType::Char32:        return lldb::eFormatUnicode32;
        case clang::BuiltinType::UShort:        return lldb::eFormatUnsigned;
        case clang::BuiltinType::Short:         return lldb::eFormatDecimal;
        case clang::BuiltinType::UInt:          return lldb::eFormatUnsigned;
        case clang::BuiltinType::Int:           return lldb::eFormatDecimal;
        case clang::BuiltinType::ULong:         return lldb::eFormatUnsigned;
        case clang::BuiltinType::Long:          return lldb::eFormatDecimal;
        case clang::BuiltinType::ULongLong:     return lldb::eFormatUnsigned;
        case clang::BuiltinType::LongLong:      return lldb::eFormatDecimal;
        case clang::BuiltinType::UInt128:       return lldb::eFormatUnsigned;
        case clang::BuiltinType::Int128:        return lldb::eFormatDecimal;
        case clang::BuiltinType::Float:         return lldb::eFormatFloat;
        case clang::BuiltinType::Double:        return lldb::eFormatFloat;
        case clang::BuiltinType::LongDouble:    return lldb::eFormatFloat;
        case clang::BuiltinType::NullPtr:       
        case clang::BuiltinType::Overload:
        case clang::BuiltinType::Dependent:
        case clang::BuiltinType::ObjCId:
        case clang::BuiltinType::ObjCClass:
        case clang::BuiltinType::ObjCSel:       
        case clang::BuiltinType::Half:          
        case clang::BuiltinType::ARCUnbridgedCast:          
        case clang::BuiltinType::PseudoObject:
        case clang::BuiltinType::BuiltinFn:
            return lldb::eFormatHex;
        }
        break;
    case clang::Type::ObjCObjectPointer:        return lldb::eFormatHex;
    case clang::Type::BlockPointer:             return lldb::eFormatHex;
    case clang::Type::Pointer:                  return lldb::eFormatHex;
    case clang::Type::LValueReference:
    case clang::Type::RValueReference:          return lldb::eFormatHex;
    case clang::Type::MemberPointer:            break;
    case clang::Type::Complex:
        {
            if (qual_type->isComplexType())
                return lldb::eFormatComplex;
            else
                return lldb::eFormatComplexInteger;
        }
    case clang::Type::ObjCInterface:            break;
    case clang::Type::Record:                   break;
    case clang::Type::Enum:                     return lldb::eFormatEnum;
    case clang::Type::Typedef:
            return ClangASTType::GetFormat(llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr());

    case clang::Type::Auto:
            return ClangASTType::GetFormat(llvm::cast<clang::AutoType>(qual_type)->desugar().getAsOpaquePtr());
    case clang::Type::DependentSizedArray:
    case clang::Type::DependentSizedExtVector:
    case clang::Type::UnresolvedUsing:
    case clang::Type::Paren:
    case clang::Type::Elaborated:
    case clang::Type::Attributed:
    case clang::Type::TemplateTypeParm:
    case clang::Type::SubstTemplateTypeParm:
    case clang::Type::SubstTemplateTypeParmPack:
    case clang::Type::InjectedClassName:
    case clang::Type::DependentName:
    case clang::Type::DependentTemplateSpecialization:
    case clang::Type::PackExpansion:
    case clang::Type::ObjCObject:
            
    case clang::Type::TypeOfExpr:
    case clang::Type::TypeOf:
    case clang::Type::Decltype:
    case clang::Type::TemplateSpecialization:
    case clang::Type::Atomic:
            break;
    }
    // We don't know hot to display this type...
    return lldb::eFormatBytes;
}


void
ClangASTType::DumpValue
(
    ExecutionContext *exe_ctx,
    Stream *s,
    lldb::Format format,
    const lldb_private::DataExtractor &data,
    uint32_t data_byte_offset,
    size_t data_byte_size,
    uint32_t bitfield_bit_size,
    uint32_t bitfield_bit_offset,
    bool show_types,
    bool show_summary,
    bool verbose,
    uint32_t depth
)
{
    return DumpValue (m_ast, 
                      m_type,
                      exe_ctx,
                      s,
                      format,
                      data,
                      data_byte_offset,
                      data_byte_size,
                      bitfield_bit_size,
                      bitfield_bit_offset,
                      show_types,
                      show_summary,
                      verbose,
                      depth);
}
                      
#define DEPTH_INCREMENT 2
void
ClangASTType::DumpValue
(
    clang::ASTContext *ast_context,
    clang_type_t clang_type,
    ExecutionContext *exe_ctx,
    Stream *s,
    lldb::Format format,
    const lldb_private::DataExtractor &data,
    uint32_t data_byte_offset,
    size_t data_byte_size,
    uint32_t bitfield_bit_size,
    uint32_t bitfield_bit_offset,
    bool show_types,
    bool show_summary,
    bool verbose,
    uint32_t depth
)
{
    clang::QualType qual_type(clang::QualType::getFromOpaquePtr(clang_type));
    switch (qual_type->getTypeClass())
    {
    case clang::Type::Record:
        if (ClangASTContext::GetCompleteType (ast_context, clang_type))
        {
            const clang::RecordType *record_type = llvm::cast<clang::RecordType>(qual_type.getTypePtr());
            const clang::RecordDecl *record_decl = record_type->getDecl();
            assert(record_decl);
            uint32_t field_bit_offset = 0;
            uint32_t field_byte_offset = 0;
            const clang::ASTRecordLayout &record_layout = ast_context->getASTRecordLayout(record_decl);
            uint32_t child_idx = 0;


            const clang::CXXRecordDecl *cxx_record_decl = llvm::dyn_cast<clang::CXXRecordDecl>(record_decl);
            if (cxx_record_decl)
            {
                // We might have base classes to print out first
                clang::CXXRecordDecl::base_class_const_iterator base_class, base_class_end;
                for (base_class = cxx_record_decl->bases_begin(), base_class_end = cxx_record_decl->bases_end();
                     base_class != base_class_end;
                     ++base_class)
                {
                    const clang::CXXRecordDecl *base_class_decl = llvm::cast<clang::CXXRecordDecl>(base_class->getType()->getAs<clang::RecordType>()->getDecl());

                    // Skip empty base classes
                    if (verbose == false && ClangASTContext::RecordHasFields(base_class_decl) == false)
                        continue;

                    if (base_class->isVirtual())
                        field_bit_offset = record_layout.getVBaseClassOffset(base_class_decl).getQuantity() * 8;
                    else
                        field_bit_offset = record_layout.getBaseClassOffset(base_class_decl).getQuantity() * 8;
                    field_byte_offset = field_bit_offset / 8;
                    assert (field_bit_offset % 8 == 0);
                    if (child_idx == 0)
                        s->PutChar('{');
                    else
                        s->PutChar(',');

                    clang::QualType base_class_qual_type = base_class->getType();
                    std::string base_class_type_name(base_class_qual_type.getAsString());

                    // Indent and print the base class type name
                    s->Printf("\n%*s%s ", depth + DEPTH_INCREMENT, "", base_class_type_name.c_str());

                    std::pair<uint64_t, unsigned> base_class_type_info = ast_context->getTypeInfo(base_class_qual_type);

                    // Dump the value of the member
                    DumpValue (ast_context,                        // The clang AST context for this type
                               base_class_qual_type.getAsOpaquePtr(),// The clang type we want to dump
                               exe_ctx,
                               s,                                  // Stream to dump to
                               ClangASTType::GetFormat(base_class_qual_type.getAsOpaquePtr()), // The format with which to display the member
                               data,                               // Data buffer containing all bytes for this type
                               data_byte_offset + field_byte_offset,// Offset into "data" where to grab value from
                               base_class_type_info.first / 8,     // Size of this type in bytes
                               0,                                  // Bitfield bit size
                               0,                                  // Bitfield bit offset
                               show_types,                         // Boolean indicating if we should show the variable types
                               show_summary,                       // Boolean indicating if we should show a summary for the current type
                               verbose,                            // Verbose output?
                               depth + DEPTH_INCREMENT);           // Scope depth for any types that have children
                    
                    ++child_idx;
                }
            }
            uint32_t field_idx = 0;
            clang::RecordDecl::field_iterator field, field_end;
            for (field = record_decl->field_begin(), field_end = record_decl->field_end(); field != field_end; ++field, ++field_idx, ++child_idx)
            {
                // Print the starting squiggly bracket (if this is the
                // first member) or comman (for member 2 and beyong) for
                // the struct/union/class member.
                if (child_idx == 0)
                    s->PutChar('{');
                else
                    s->PutChar(',');

                // Indent
                s->Printf("\n%*s", depth + DEPTH_INCREMENT, "");

                clang::QualType field_type = field->getType();
                // Print the member type if requested
                // Figure out the type byte size (field_type_info.first) and
                // alignment (field_type_info.second) from the AST context.
                std::pair<uint64_t, unsigned> field_type_info = ast_context->getTypeInfo(field_type);
                assert(field_idx < record_layout.getFieldCount());
                // Figure out the field offset within the current struct/union/class type
                field_bit_offset = record_layout.getFieldOffset (field_idx);
                field_byte_offset = field_bit_offset / 8;
                uint32_t field_bitfield_bit_size = 0;
                uint32_t field_bitfield_bit_offset = 0;
                if (ClangASTContext::FieldIsBitfield (ast_context, *field, field_bitfield_bit_size))
                    field_bitfield_bit_offset = field_bit_offset % 8;

                if (show_types)
                {
                    std::string field_type_name(field_type.getAsString());
                    if (field_bitfield_bit_size > 0)
                        s->Printf("(%s:%u) ", field_type_name.c_str(), field_bitfield_bit_size);
                    else
                        s->Printf("(%s) ", field_type_name.c_str());
                }
                // Print the member name and equal sign
                s->Printf("%s = ", field->getNameAsString().c_str());


                // Dump the value of the member
                DumpValue (ast_context,                    // The clang AST context for this type
                           field_type.getAsOpaquePtr(),    // The clang type we want to dump
                           exe_ctx,
                           s,                              // Stream to dump to
                           ClangASTType::GetFormat(field_type.getAsOpaquePtr()),   // The format with which to display the member
                           data,                           // Data buffer containing all bytes for this type
                           data_byte_offset + field_byte_offset,// Offset into "data" where to grab value from
                           field_type_info.first / 8,      // Size of this type in bytes
                           field_bitfield_bit_size,        // Bitfield bit size
                           field_bitfield_bit_offset,      // Bitfield bit offset
                           show_types,                     // Boolean indicating if we should show the variable types
                           show_summary,                   // Boolean indicating if we should show a summary for the current type
                           verbose,                        // Verbose output?
                           depth + DEPTH_INCREMENT);       // Scope depth for any types that have children
            }

            // Indent the trailing squiggly bracket
            if (child_idx > 0)
                s->Printf("\n%*s}", depth, "");
        }
        return;

    case clang::Type::Enum:
        if (ClangASTContext::GetCompleteType (ast_context, clang_type))
        {
            const clang::EnumType *enum_type = llvm::cast<clang::EnumType>(qual_type.getTypePtr());
            const clang::EnumDecl *enum_decl = enum_type->getDecl();
            assert(enum_decl);
            clang::EnumDecl::enumerator_iterator enum_pos, enum_end_pos;
            uint32_t offset = data_byte_offset;
            const int64_t enum_value = data.GetMaxU64Bitfield(&offset, data_byte_size, bitfield_bit_size, bitfield_bit_offset);
            for (enum_pos = enum_decl->enumerator_begin(), enum_end_pos = enum_decl->enumerator_end(); enum_pos != enum_end_pos; ++enum_pos)
            {
                if (enum_pos->getInitVal() == enum_value)
                {
                    s->Printf("%s", enum_pos->getNameAsString().c_str());
                    return;
                }
            }
            // If we have gotten here we didn't get find the enumerator in the
            // enum decl, so just print the integer.
            s->Printf("%lli", enum_value);
        }
        return;

    case clang::Type::ConstantArray:
        {
            const clang::ConstantArrayType *array = llvm::cast<clang::ConstantArrayType>(qual_type.getTypePtr());
            bool is_array_of_characters = false;
            clang::QualType element_qual_type = array->getElementType();

            const clang::Type *canonical_type = element_qual_type->getCanonicalTypeInternal().getTypePtr();
            if (canonical_type)
                is_array_of_characters = canonical_type->isCharType();

            const uint64_t element_count = array->getSize().getLimitedValue();

            std::pair<uint64_t, unsigned> field_type_info = ast_context->getTypeInfo(element_qual_type);

            uint32_t element_idx = 0;
            uint32_t element_offset = 0;
            uint64_t element_byte_size = field_type_info.first / 8;
            uint32_t element_stride = element_byte_size;

            if (is_array_of_characters)
            {
                s->PutChar('"');
                data.Dump(s, data_byte_offset, lldb::eFormatChar, element_byte_size, element_count, UINT32_MAX, LLDB_INVALID_ADDRESS, 0, 0);
                s->PutChar('"');
                return;
            }
            else
            {
                lldb::Format element_format = ClangASTType::GetFormat(element_qual_type.getAsOpaquePtr());

                for (element_idx = 0; element_idx < element_count; ++element_idx)
                {
                    // Print the starting squiggly bracket (if this is the
                    // first member) or comman (for member 2 and beyong) for
                    // the struct/union/class member.
                    if (element_idx == 0)
                        s->PutChar('{');
                    else
                        s->PutChar(',');

                    // Indent and print the index
                    s->Printf("\n%*s[%u] ", depth + DEPTH_INCREMENT, "", element_idx);

                    // Figure out the field offset within the current struct/union/class type
                    element_offset = element_idx * element_stride;

                    // Dump the value of the member
                    DumpValue (ast_context,                    // The clang AST context for this type
                               element_qual_type.getAsOpaquePtr(), // The clang type we want to dump
                               exe_ctx,
                               s,                              // Stream to dump to
                               element_format,                 // The format with which to display the element
                               data,                           // Data buffer containing all bytes for this type
                               data_byte_offset + element_offset,// Offset into "data" where to grab value from
                               element_byte_size,              // Size of this type in bytes
                               0,                              // Bitfield bit size
                               0,                              // Bitfield bit offset
                               show_types,                     // Boolean indicating if we should show the variable types
                               show_summary,                   // Boolean indicating if we should show a summary for the current type
                               verbose,                        // Verbose output?
                               depth + DEPTH_INCREMENT);       // Scope depth for any types that have children
                }

                // Indent the trailing squiggly bracket
                if (element_idx > 0)
                    s->Printf("\n%*s}", depth, "");
            }
        }
        return;

    case clang::Type::Typedef:
        {
            clang::QualType typedef_qual_type = llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType();
            lldb::Format typedef_format = ClangASTType::GetFormat(typedef_qual_type.getAsOpaquePtr());
            std::pair<uint64_t, unsigned> typedef_type_info = ast_context->getTypeInfo(typedef_qual_type);
            uint64_t typedef_byte_size = typedef_type_info.first / 8;

            return DumpValue (ast_context,        // The clang AST context for this type
                              typedef_qual_type.getAsOpaquePtr(), // The clang type we want to dump
                              exe_ctx,
                              s,                  // Stream to dump to
                              typedef_format,     // The format with which to display the element
                              data,               // Data buffer containing all bytes for this type
                              data_byte_offset,   // Offset into "data" where to grab value from
                              typedef_byte_size,  // Size of this type in bytes
                              bitfield_bit_size,  // Bitfield bit size
                              bitfield_bit_offset,// Bitfield bit offset
                              show_types,         // Boolean indicating if we should show the variable types
                              show_summary,       // Boolean indicating if we should show a summary for the current type
                              verbose,            // Verbose output?
                              depth);             // Scope depth for any types that have children
        }
        break;

    default:
        // We are down the a scalar type that we just need to display.
        data.Dump(s, data_byte_offset, format, data_byte_size, 1, UINT32_MAX, LLDB_INVALID_ADDRESS, bitfield_bit_size, bitfield_bit_offset);

        if (show_summary)
            DumpSummary (ast_context, clang_type, exe_ctx, s, data, data_byte_offset, data_byte_size);
        break;
    }
}



bool
ClangASTType::DumpTypeValue (Stream *s,
                             lldb::Format format,
                             const lldb_private::DataExtractor &data,
                             uint32_t byte_offset,
                             size_t byte_size,
                             uint32_t bitfield_bit_size,
                             uint32_t bitfield_bit_offset,
                             ExecutionContextScope *exe_scope)
{
    return DumpTypeValue (m_ast,
                          m_type,
                          s,
                          format,
                          data,
                          byte_offset,
                          byte_size,
                          bitfield_bit_size,
                          bitfield_bit_offset,
                          exe_scope);
}


bool
ClangASTType::DumpTypeValue (clang::ASTContext *ast_context,
                             clang_type_t clang_type,
                             Stream *s,
                             lldb::Format format,
                             const lldb_private::DataExtractor &data,
                             uint32_t byte_offset,
                             size_t byte_size,
                             uint32_t bitfield_bit_size,
                             uint32_t bitfield_bit_offset,
                             ExecutionContextScope *exe_scope)
{
    clang::QualType qual_type(clang::QualType::getFromOpaquePtr(clang_type));
    if (ClangASTContext::IsAggregateType (clang_type))
    {
        return 0;
    }
    else
    {
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
            
        switch (type_class)
        {
        case clang::Type::Typedef:
            {
                clang::QualType typedef_qual_type = llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType();
                if (format == eFormatDefault)
                    format = ClangASTType::GetFormat(typedef_qual_type.getAsOpaquePtr());
                std::pair<uint64_t, unsigned> typedef_type_info = ast_context->getTypeInfo(typedef_qual_type);
                uint64_t typedef_byte_size = typedef_type_info.first / 8;

                return ClangASTType::DumpTypeValue (ast_context,            // The clang AST context for this type
                                                    typedef_qual_type.getAsOpaquePtr(),     // The clang type we want to dump
                                                    s,
                                                    format,                 // The format with which to display the element
                                                    data,                   // Data buffer containing all bytes for this type
                                                    byte_offset,            // Offset into "data" where to grab value from
                                                    typedef_byte_size,      // Size of this type in bytes
                                                    bitfield_bit_size,      // Size in bits of a bitfield value, if zero don't treat as a bitfield
                                                    bitfield_bit_offset,    // Offset in bits of a bitfield value if bitfield_bit_size != 0
                                                    exe_scope);
            }
            break;

        case clang::Type::Enum:
            // If our format is enum or default, show the enumeration value as
            // its enumeration string value, else just display it as requested.
            if ((format == eFormatEnum || format == eFormatDefault) && ClangASTContext::GetCompleteType (ast_context, clang_type))
            {
                const clang::EnumType *enum_type = llvm::cast<clang::EnumType>(qual_type.getTypePtr());
                const clang::EnumDecl *enum_decl = enum_type->getDecl();
                assert(enum_decl);
                clang::EnumDecl::enumerator_iterator enum_pos, enum_end_pos;
                uint32_t offset = byte_offset;
                const int64_t enum_value = data.GetMaxU64Bitfield (&offset, byte_size, bitfield_bit_size, bitfield_bit_offset);
                for (enum_pos = enum_decl->enumerator_begin(), enum_end_pos = enum_decl->enumerator_end(); enum_pos != enum_end_pos; ++enum_pos)
                {
                    if (enum_pos->getInitVal() == enum_value)
                    {
                        s->PutCString (enum_pos->getNameAsString().c_str());
                        return true;
                    }
                }
                // If we have gotten here we didn't get find the enumerator in the
                // enum decl, so just print the integer.
                
                s->Printf("%lli", enum_value);
                return true;
            }
            // format was not enum, just fall through and dump the value as requested....
                
        default:
            // We are down the a scalar type that we just need to display.
            {
                uint32_t item_count = 1;
                // A few formats, we might need to modify our size and count for depending
                // on how we are trying to display the value...
                switch (format)
                {
                    default:
                    case eFormatBoolean:
                    case eFormatBinary:
                    case eFormatComplex:
                    case eFormatCString:         // NULL terminated C strings
                    case eFormatDecimal:
                    case eFormatEnum:
                    case eFormatHex:
                    case eFormatHexUppercase:
                    case eFormatFloat:
                    case eFormatOctal:
                    case eFormatOSType:
                    case eFormatUnsigned:
                    case eFormatPointer:
                    case eFormatVectorOfChar:
                    case eFormatVectorOfSInt8:
                    case eFormatVectorOfUInt8:
                    case eFormatVectorOfSInt16:
                    case eFormatVectorOfUInt16:
                    case eFormatVectorOfSInt32:
                    case eFormatVectorOfUInt32:
                    case eFormatVectorOfSInt64:
                    case eFormatVectorOfUInt64:
                    case eFormatVectorOfFloat32:
                    case eFormatVectorOfFloat64:
                    case eFormatVectorOfUInt128:
                        break;

                    case eFormatChar: 
                    case eFormatCharPrintable:  
                    case eFormatCharArray:
                    case eFormatBytes:
                    case eFormatBytesWithASCII:
                        item_count = byte_size;
                        byte_size = 1; 
                        break;

                    case eFormatUnicode16:
                        item_count = byte_size / 2; 
                        byte_size = 2; 
                        break;

                    case eFormatUnicode32:
                        item_count = byte_size / 4; 
                        byte_size = 4; 
                        break;
                }
                return data.Dump (s,
                                  byte_offset,
                                  format,
                                  byte_size,
                                  item_count,
                                  UINT32_MAX,
                                  LLDB_INVALID_ADDRESS,
                                  bitfield_bit_size,
                                  bitfield_bit_offset,
                                  exe_scope);
            }
            break;
        }
    }
    return 0;
}



void
ClangASTType::DumpSummary
(
    ExecutionContext *exe_ctx,
    Stream *s,
    const lldb_private::DataExtractor &data,
    uint32_t data_byte_offset,
    size_t data_byte_size
)
{
    return DumpSummary (m_ast,
                        m_type,
                        exe_ctx, 
                        s, 
                        data, 
                        data_byte_offset, 
                        data_byte_size);
}

void
ClangASTType::DumpSummary
(
    clang::ASTContext *ast_context,
    clang_type_t clang_type,
    ExecutionContext *exe_ctx,
    Stream *s,
    const lldb_private::DataExtractor &data,
    uint32_t data_byte_offset,
    size_t data_byte_size
)
{
    uint32_t length = 0;
    if (ClangASTContext::IsCStringType (clang_type, length))
    {
        if (exe_ctx)
        {
            Process *process = exe_ctx->GetProcessPtr();
            if (process)
            {
                uint32_t offset = data_byte_offset;
                lldb::addr_t pointer_addresss = data.GetMaxU64(&offset, data_byte_size);
                std::vector<uint8_t> buf;
                if (length > 0)
                    buf.resize (length);
                else
                    buf.resize (256);

                lldb_private::DataExtractor cstr_data(&buf.front(), buf.size(), process->GetByteOrder(), 4);
                buf.back() = '\0';
                size_t bytes_read;
                size_t total_cstr_len = 0;
                Error error;
                while ((bytes_read = process->ReadMemory (pointer_addresss, &buf.front(), buf.size(), error)) > 0)
                {
                    const size_t len = strlen((const char *)&buf.front());
                    if (len == 0)
                        break;
                    if (total_cstr_len == 0)
                        s->PutCString (" \"");
                    cstr_data.Dump(s, 0, lldb::eFormatChar, 1, len, UINT32_MAX, LLDB_INVALID_ADDRESS, 0, 0);
                    total_cstr_len += len;
                    if (len < buf.size())
                        break;
                    pointer_addresss += total_cstr_len;
                }
                if (total_cstr_len > 0)
                    s->PutChar ('"');
            }
        }
    }
}

uint32_t
ClangASTType::GetClangTypeBitWidth ()
{
    return GetClangTypeBitWidth (m_ast, m_type);
}

uint32_t
ClangASTType::GetClangTypeBitWidth (clang::ASTContext *ast_context, clang_type_t clang_type)
{
    if (ClangASTContext::GetCompleteType (ast_context, clang_type))
    {
        clang::QualType qual_type(clang::QualType::getFromOpaquePtr(clang_type));
        return ast_context->getTypeSize (qual_type);
    }
    return 0;
}

size_t
ClangASTType::GetTypeBitAlign ()
{
    return GetTypeBitAlign (m_ast, m_type);
}

size_t
ClangASTType::GetTypeBitAlign (clang::ASTContext *ast_context, clang_type_t clang_type)
{
    if (ClangASTContext::GetCompleteType (ast_context, clang_type))
        return ast_context->getTypeAlign(clang::QualType::getFromOpaquePtr(clang_type));
    return 0;
}


bool
ClangASTType::IsDefined()
{
    return ClangASTType::IsDefined (m_type);
}

bool
ClangASTType::IsDefined (clang_type_t clang_type)
{
    clang::QualType qual_type(clang::QualType::getFromOpaquePtr(clang_type));
    const clang::TagType *tag_type = llvm::dyn_cast<clang::TagType>(qual_type.getTypePtr());
    if (tag_type)
    {
        clang::TagDecl *tag_decl = tag_type->getDecl();
        if (tag_decl)
            return tag_decl->isCompleteDefinition();
        return false;
    }
    else
    {
        const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type);
        if (objc_class_type)
        {
            clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
            if (class_interface_decl)
                return class_interface_decl->getDefinition() != NULL;
            return false;
        }
    }
    return true;
}

bool
ClangASTType::IsConst()
{
    return ClangASTType::IsConst (m_type);
}

bool
ClangASTType::IsConst (lldb::clang_type_t clang_type)
{
    clang::QualType qual_type(clang::QualType::getFromOpaquePtr(clang_type));
    
    return qual_type.isConstQualified();
}

void
ClangASTType::DumpTypeDescription (Stream *s)
{
    return DumpTypeDescription (m_ast, m_type, s);
}

// Dump the full description of a type. For classes this means all of the
// ivars and member functions, for structs/unions all of the members. 
void
ClangASTType::DumpTypeDescription (clang::ASTContext *ast_context, clang_type_t clang_type, Stream *s)
{
    if (clang_type)
    {
        clang::QualType qual_type(clang::QualType::getFromOpaquePtr(clang_type));

        llvm::SmallVector<char, 1024> buf;
        llvm::raw_svector_ostream llvm_ostrm (buf);

        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
        case clang::Type::ObjCObject:
        case clang::Type::ObjCInterface:
            if (ClangASTContext::GetCompleteType (ast_context, clang_type))
            {
                const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type.getTypePtr());
                assert (objc_class_type);
                if (objc_class_type)
                {
                    clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                    if (class_interface_decl)
                    {
                        clang::PrintingPolicy policy = ast_context->getPrintingPolicy();
                        policy.DumpSourceManager = &ast_context->getSourceManager();
                        class_interface_decl->print(llvm_ostrm, policy, s->GetIndentLevel());
                    }
                }
            }
            break;
        
        case clang::Type::Typedef:
            {
                const clang::TypedefType *typedef_type = qual_type->getAs<clang::TypedefType>();
                if (typedef_type)
                {
                    const clang::TypedefNameDecl *typedef_decl = typedef_type->getDecl();
                    std::string clang_typedef_name (typedef_decl->getQualifiedNameAsString());
                    if (!clang_typedef_name.empty())
                    {
                        s->PutCString ("typedef ");
                        s->PutCString (clang_typedef_name.c_str());
                    }
                }
            }
            break;

        case clang::Type::Record:
            if (ClangASTContext::GetCompleteType (ast_context, clang_type))
            {
                const clang::RecordType *record_type = llvm::cast<clang::RecordType>(qual_type.getTypePtr());
                const clang::RecordDecl *record_decl = record_type->getDecl();
                const clang::CXXRecordDecl *cxx_record_decl = llvm::dyn_cast<clang::CXXRecordDecl>(record_decl);

                if (cxx_record_decl)
                    cxx_record_decl->print(llvm_ostrm, ast_context->getPrintingPolicy(), s->GetIndentLevel());
                else
                    record_decl->print(llvm_ostrm, ast_context->getPrintingPolicy(), s->GetIndentLevel());
            }
            break;

        default:
            {
                const clang::TagType *tag_type = llvm::dyn_cast<clang::TagType>(qual_type.getTypePtr());
                if (tag_type)
                {
                    clang::TagDecl *tag_decl = tag_type->getDecl();
                    if (tag_decl)
                        tag_decl->print(llvm_ostrm, 0);
                }
                else
                {
                    std::string clang_type_name(qual_type.getAsString());
                    if (!clang_type_name.empty())
                        s->PutCString (clang_type_name.c_str());
                }
            }
        }
        
        llvm_ostrm.flush();
        if (buf.size() > 0)
        {
            s->Write (buf.data(), buf.size());
        }
    }
}

void
ClangASTType::DumpTypeCode (Stream *s)
{
    DumpTypeCode(m_type, s);
}

void
ClangASTType::DumpTypeCode (void *type, 
                            Stream *s)
{
    clang::QualType qual_type(clang::QualType::getFromOpaquePtr(type));
    s->PutCString(qual_type.getAsString().c_str());
}

bool
ClangASTType::GetValueAsScalar
(
    const lldb_private::DataExtractor &data,
    uint32_t data_byte_offset,
    size_t data_byte_size,
    Scalar &value
)
{
    return GetValueAsScalar (m_ast, 
                             m_type, 
                             data, 
                             data_byte_offset, 
                             data_byte_size, 
                             value);
}

bool
ClangASTType::GetValueAsScalar
(
    clang::ASTContext *ast_context,
    clang_type_t clang_type,
    const lldb_private::DataExtractor &data,
    uint32_t data_byte_offset,
    size_t data_byte_size,
    Scalar &value
)
{
    clang::QualType qual_type(clang::QualType::getFromOpaquePtr(clang_type));

    if (ClangASTContext::IsAggregateType (clang_type))
    {
        return false;   // Aggregate types don't have scalar values
    }
    else
    {
        uint32_t count = 0;
        lldb::Encoding encoding = GetEncoding (clang_type, count);

        if (encoding == lldb::eEncodingInvalid || count != 1)
            return false;

        uint64_t bit_width = ast_context->getTypeSize(qual_type);
        uint32_t byte_size = (bit_width + 7 ) / 8;
        uint32_t offset = data_byte_offset;
        switch (encoding)
        {
        case lldb::eEncodingInvalid:
            break;
        case lldb::eEncodingVector:
            break;
        case lldb::eEncodingUint:
            if (byte_size <= sizeof(unsigned long long))
            {
                uint64_t uval64 = data.GetMaxU64 (&offset, byte_size);
                if (byte_size <= sizeof(unsigned int))
                {
                    value = (unsigned int)uval64;
                    return true;
                }
                else if (byte_size <= sizeof(unsigned long))
                {
                    value = (unsigned long)uval64;
                    return true;
                }
                else if (byte_size <= sizeof(unsigned long long))
                {
                    value = (unsigned long long )uval64;
                    return true;
                }
                else
                    value.Clear();
            }
            break;

        case lldb::eEncodingSint:
            if (byte_size <= sizeof(long long))
            {
                int64_t sval64 = data.GetMaxS64 (&offset, byte_size);
                if (byte_size <= sizeof(int))
                {
                    value = (int)sval64;
                    return true;
                }
                else if (byte_size <= sizeof(long))
                {
                    value = (long)sval64;
                    return true;
                }
                else if (byte_size <= sizeof(long long))
                {
                    value = (long long )sval64;
                    return true;
                }
                else
                    value.Clear();
            }
            break;

        case lldb::eEncodingIEEE754:
            if (byte_size <= sizeof(long double))
            {
                uint32_t u32;
                uint64_t u64;
                if (byte_size == sizeof(float))
                {
                    if (sizeof(float) == sizeof(uint32_t))
                    {
                        u32 = data.GetU32(&offset);
                        value = *((float *)&u32);
                        return true;
                    }
                    else if (sizeof(float) == sizeof(uint64_t))
                    {
                        u64 = data.GetU64(&offset);
                        value = *((float *)&u64);
                        return true;
                    }
                }
                else
                if (byte_size == sizeof(double))
                {
                    if (sizeof(double) == sizeof(uint32_t))
                    {
                        u32 = data.GetU32(&offset);
                        value = *((double *)&u32);
                        return true;
                    }
                    else if (sizeof(double) == sizeof(uint64_t))
                    {
                        u64 = data.GetU64(&offset);
                        value = *((double *)&u64);
                        return true;
                    }
                }
                else
                if (byte_size == sizeof(long double))
                {
                    if (sizeof(long double) == sizeof(uint32_t))
                    {
                        u32 = data.GetU32(&offset);
                        value = *((long double *)&u32);
                        return true;
                    }
                    else if (sizeof(long double) == sizeof(uint64_t))
                    {
                        u64 = data.GetU64(&offset);
                        value = *((long double *)&u64);
                        return true;
                    }
                }
            }
            break;
        }
    }
    return false;
}

bool
ClangASTType::SetValueFromScalar (const Scalar &value, Stream &strm)
{
    return SetValueFromScalar (m_ast, m_type, value, strm);
}

bool
ClangASTType::SetValueFromScalar
(
    clang::ASTContext *ast_context,
    clang_type_t clang_type,
    const Scalar &value,
    Stream &strm
)
{
    clang::QualType qual_type(clang::QualType::getFromOpaquePtr(clang_type));

    // Aggregate types don't have scalar values
    if (!ClangASTContext::IsAggregateType (clang_type))
    {
        strm.GetFlags().Set(Stream::eBinary);
        uint32_t count = 0;
        lldb::Encoding encoding = GetEncoding (clang_type, count);

        if (encoding == lldb::eEncodingInvalid || count != 1)
            return false;

        uint64_t bit_width = ast_context->getTypeSize(qual_type);
        // This function doesn't currently handle non-byte aligned assignments
        if ((bit_width % 8) != 0)
            return false;

        uint32_t byte_size = (bit_width + 7 ) / 8;
        switch (encoding)
        {
        case lldb::eEncodingInvalid:
            break;
        case lldb::eEncodingVector:
            break;
        case lldb::eEncodingUint:
            switch (byte_size)
            {
            case 1: strm.PutHex8(value.UInt()); return true;
            case 2: strm.PutHex16(value.UInt()); return true;
            case 4: strm.PutHex32(value.UInt()); return true;
            case 8: strm.PutHex64(value.ULongLong()); return true;
            default:
                break;
            }
            break;

        case lldb::eEncodingSint:
            switch (byte_size)
            {
            case 1: strm.PutHex8(value.SInt()); return true;
            case 2: strm.PutHex16(value.SInt()); return true;
            case 4: strm.PutHex32(value.SInt()); return true;
            case 8: strm.PutHex64(value.SLongLong()); return true;
            default:
                break;
            }
            break;

        case lldb::eEncodingIEEE754:
            if (byte_size <= sizeof(long double))
            {
                if (byte_size == sizeof(float))
                {
                    strm.PutFloat(value.Float());
                    return true;
                }
                else
                if (byte_size == sizeof(double))
                {
                    strm.PutDouble(value.Double());
                    return true;
                }
                else
                if (byte_size == sizeof(long double))
                {
                    strm.PutDouble(value.LongDouble());
                    return true;
                }
            }
            break;
        }
    }
    return false;
}

bool
ClangASTType::ReadFromMemory
(
    lldb_private::ExecutionContext *exe_ctx,
    lldb::addr_t addr,
    AddressType address_type,
    lldb_private::DataExtractor &data
)
{
    return ReadFromMemory (m_ast,
                           m_type,
                           exe_ctx, 
                           addr,
                           address_type,
                           data);
}

uint32_t
ClangASTType::GetTypeByteSize() const
{
    return GetTypeByteSize(m_ast,
                           m_type);
}

uint32_t
ClangASTType::GetTypeByteSize(
                clang::ASTContext *ast_context,
                lldb::clang_type_t opaque_clang_qual_type)
{
    
    if (ClangASTContext::GetCompleteType (ast_context, opaque_clang_qual_type))
    {
        clang::QualType qual_type(clang::QualType::getFromOpaquePtr(opaque_clang_qual_type));
        
        uint32_t byte_size = (ast_context->getTypeSize (qual_type) + 7) / 8;
        
        if (ClangASTContext::IsObjCClassType(opaque_clang_qual_type))
            byte_size += ast_context->getTypeSize(ast_context->ObjCBuiltinClassTy) / 8; // isa
        
        return byte_size;
    }
    return 0;
}


bool
ClangASTType::ReadFromMemory
(
    clang::ASTContext *ast_context,
    clang_type_t clang_type,
    lldb_private::ExecutionContext *exe_ctx,
    lldb::addr_t addr,
    AddressType address_type,
    lldb_private::DataExtractor &data
)
{
    if (address_type == eAddressTypeFile)
    {
        // Can't convert a file address to anything valid without more
        // context (which Module it came from)
        return false;
    }
    
    if (!ClangASTContext::GetCompleteType(ast_context, clang_type))
        return false;
    
    clang::QualType qual_type(clang::QualType::getFromOpaquePtr(clang_type));
    
    const uint32_t byte_size = (ast_context->getTypeSize (qual_type) + 7) / 8;
    if (data.GetByteSize() < byte_size)
    {
        lldb::DataBufferSP data_sp(new DataBufferHeap (byte_size, '\0'));
        data.SetData(data_sp);
    }

    uint8_t* dst = (uint8_t*)data.PeekData(0, byte_size);
    if (dst != NULL)
    {
        if (address_type == eAddressTypeHost)
        {
            // The address is an address in this process, so just copy it
            memcpy (dst, (uint8_t*)NULL + addr, byte_size);
            return true;
        }
        else
        {
            Process *process = NULL;
            if (exe_ctx)
                process = exe_ctx->GetProcessPtr();
            if (process)
            {
                Error error;
                return process->ReadMemory(addr, dst, byte_size, error) == byte_size;
            }
        }
    }
    return false;
}

bool
ClangASTType::WriteToMemory
(
    lldb_private::ExecutionContext *exe_ctx,
    lldb::addr_t addr,
    AddressType address_type,
    StreamString &new_value
)
{
    return WriteToMemory (m_ast,
                          m_type,
                          exe_ctx,
                          addr,
                          address_type,
                          new_value);
}

bool
ClangASTType::WriteToMemory
(
    clang::ASTContext *ast_context,
    clang_type_t clang_type,
    lldb_private::ExecutionContext *exe_ctx,
    lldb::addr_t addr,
    AddressType address_type,
    StreamString &new_value
)
{
    if (address_type == eAddressTypeFile)
    {
        // Can't convert a file address to anything valid without more
        // context (which Module it came from)
        return false;
    }
    clang::QualType qual_type(clang::QualType::getFromOpaquePtr(clang_type));
    const uint32_t byte_size = (ast_context->getTypeSize (qual_type) + 7) / 8;

    if (byte_size > 0)
    {
        if (address_type == eAddressTypeHost)
        {
            // The address is an address in this process, so just copy it
            memcpy ((void *)addr, new_value.GetData(), byte_size);
            return true;
        }
        else
        {
            Process *process = NULL;
            if (exe_ctx)
                process = exe_ctx->GetProcessPtr();
            if (process)
            {
                Error error;
                return process->WriteMemory(addr, new_value.GetData(), byte_size, error) == byte_size;
            }
        }
    }
    return false;
}


lldb::clang_type_t
ClangASTType::RemoveFastQualifiers (lldb::clang_type_t clang_type)
{
    clang::QualType qual_type(clang::QualType::getFromOpaquePtr(clang_type));
    qual_type.getQualifiers().removeFastQualifiers();
    return qual_type.getAsOpaquePtr();
}

clang::CXXRecordDecl *
ClangASTType::GetAsCXXRecordDecl (lldb::clang_type_t opaque_clang_qual_type)
{
    if (opaque_clang_qual_type)
        return clang::QualType::getFromOpaquePtr(opaque_clang_qual_type)->getAsCXXRecordDecl();
    return NULL;
}

bool
lldb_private::operator == (const lldb_private::ClangASTType &lhs, const lldb_private::ClangASTType &rhs)
{
    return lhs.GetASTContext() == rhs.GetASTContext() && lhs.GetOpaqueQualType() == rhs.GetOpaqueQualType();
}


bool
lldb_private::operator != (const lldb_private::ClangASTType &lhs, const lldb_private::ClangASTType &rhs)
{
    return lhs.GetASTContext() != rhs.GetASTContext() || lhs.GetOpaqueQualType() != rhs.GetOpaqueQualType();
}
