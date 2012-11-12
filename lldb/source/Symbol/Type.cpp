//===-- Type.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Other libraries and framework includes

#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Core/StreamString.h"

#include "lldb/Symbol/ClangASTType.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolContextScope.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/TypeList.h"

#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"

#include "llvm/ADT/StringRef.h"

using namespace lldb;
using namespace lldb_private;

Type *
SymbolFileType::GetType ()
{
    if (!m_type_sp)
    {
        Type *resolved_type = m_symbol_file.ResolveTypeUID (GetID());
        if (resolved_type)
            m_type_sp = resolved_type->shared_from_this();
    }
    return m_type_sp.get();
}


Type::Type
(
    lldb::user_id_t uid,
    SymbolFile* symbol_file,
    const ConstString &name,
    uint32_t byte_size,
    SymbolContextScope *context,
    user_id_t encoding_uid,
    EncodingDataType encoding_uid_type,
    const Declaration& decl,
    clang_type_t clang_type,
    ResolveState clang_type_resolve_state
) :
    UserID (uid),
    m_name (name),
    m_symbol_file (symbol_file),
    m_context (context),
    m_encoding_type (NULL),
    m_encoding_uid (encoding_uid),
    m_encoding_uid_type (encoding_uid_type),
    m_byte_size (byte_size),
    m_decl (decl),
    m_clang_type (clang_type)
{
    m_flags.clang_type_resolve_state = (clang_type ? clang_type_resolve_state : eResolveStateUnresolved);
    m_flags.is_complete_objc_class = false;
}

Type::Type () :
    UserID (0),
    m_name ("<INVALID TYPE>"),
    m_symbol_file (NULL),
    m_context (NULL),
    m_encoding_type (NULL),
    m_encoding_uid (0),
    m_encoding_uid_type (eEncodingInvalid),
    m_byte_size (0),
    m_decl (),
    m_clang_type (NULL)
{
    m_flags.clang_type_resolve_state = eResolveStateUnresolved;
    m_flags.is_complete_objc_class = false;
}


Type::Type (const Type &rhs) :
    UserID (rhs),
    m_name (rhs.m_name),
    m_symbol_file (rhs.m_symbol_file),
    m_context (rhs.m_context),
    m_encoding_type (rhs.m_encoding_type),
    m_encoding_uid (rhs.m_encoding_uid),
    m_encoding_uid_type (rhs.m_encoding_uid_type),
    m_byte_size (rhs.m_byte_size),
    m_decl (rhs.m_decl),
    m_clang_type (rhs.m_clang_type),
    m_flags (rhs.m_flags)
{
}

const Type&
Type::operator= (const Type& rhs)
{
    if (this != &rhs)
    {
    }
    return *this;
}


void
Type::GetDescription (Stream *s, lldb::DescriptionLevel level, bool show_name)
{
    *s << "id = " << (const UserID&)*this;

    // Call the name accessor to make sure we resolve the type name
    if (show_name)
    {
        const ConstString &type_name = GetName();
        if (type_name)
        {
            *s << ", name = \"" << type_name << '"';
            ConstString qualified_type_name (GetQualifiedName());
            if (qualified_type_name != type_name)
            {
                *s << ", qualified = \"" << qualified_type_name << '"';
            }
        }
    }

    // Call the get byte size accesor so we resolve our byte size
    if (GetByteSize())
        s->Printf(", byte-size = %u", m_byte_size);
    bool show_fullpaths = (level == lldb::eDescriptionLevelVerbose);
    m_decl.Dump(s, show_fullpaths);

    if (m_clang_type)
    {
        *s << ", clang_type = \"";
        ClangASTType::DumpTypeDescription (GetClangAST(), m_clang_type, s);
        *s << '"';
    }
    else if (m_encoding_uid != LLDB_INVALID_UID)
    {
        s->Printf(", type_uid = 0x%8.8x", m_encoding_uid);
        switch (m_encoding_uid_type)
        {
        case eEncodingInvalid: break;
        case eEncodingIsUID: s->PutCString(" (unresolved type)"); break;
        case eEncodingIsConstUID: s->PutCString(" (unresolved const type)"); break;
        case eEncodingIsRestrictUID: s->PutCString(" (unresolved restrict type)"); break;
        case eEncodingIsVolatileUID: s->PutCString(" (unresolved volatile type)"); break;
        case eEncodingIsTypedefUID: s->PutCString(" (unresolved typedef)"); break;
        case eEncodingIsPointerUID: s->PutCString(" (unresolved pointer)"); break;
        case eEncodingIsLValueReferenceUID: s->PutCString(" (unresolved L value reference)"); break;
        case eEncodingIsRValueReferenceUID: s->PutCString(" (unresolved R value reference)"); break;
        case eEncodingIsSyntheticUID: s->PutCString(" (synthetic type)"); break;
        }
    }    
}


void
Type::Dump (Stream *s, bool show_context)
{
    s->Printf("%p: ", this);
    s->Indent();
    *s << "Type" << (const UserID&)*this << ' ';
    if (m_name)
        *s << ", name = \"" << m_name << "\"";

    if (m_byte_size != 0)
        s->Printf(", size = %u", m_byte_size);

    if (show_context && m_context != NULL)
    {
        s->PutCString(", context = ( ");
        m_context->DumpSymbolContext(s);
        s->PutCString(" )");
    }

    bool show_fullpaths = false;
    m_decl.Dump (s,show_fullpaths);

    if (m_clang_type)
    {
        *s << ", clang_type = " << m_clang_type << ' ';

        ClangASTType::DumpTypeDescription (GetClangAST(), m_clang_type, s);
    }
    else if (m_encoding_uid != LLDB_INVALID_UID)
    {
        *s << ", type_data = " << (uint64_t)m_encoding_uid;
        switch (m_encoding_uid_type)
        {
        case eEncodingInvalid: break;
        case eEncodingIsUID: s->PutCString(" (unresolved type)"); break;
        case eEncodingIsConstUID: s->PutCString(" (unresolved const type)"); break;
        case eEncodingIsRestrictUID: s->PutCString(" (unresolved restrict type)"); break;
        case eEncodingIsVolatileUID: s->PutCString(" (unresolved volatile type)"); break;
        case eEncodingIsTypedefUID: s->PutCString(" (unresolved typedef)"); break;
        case eEncodingIsPointerUID: s->PutCString(" (unresolved pointer)"); break;
        case eEncodingIsLValueReferenceUID: s->PutCString(" (unresolved L value reference)"); break;
        case eEncodingIsRValueReferenceUID: s->PutCString(" (unresolved R value reference)"); break;
        case eEncodingIsSyntheticUID: s->PutCString(" (synthetic type)"); break;
        }
    }

//
//  if (m_access)
//      s->Printf(", access = %u", m_access);
    s->EOL();
}

const ConstString &
Type::GetName()
{
    if (!m_name)
    {
        if (ResolveClangType(eResolveStateForward))
            m_name = ClangASTType::GetConstTypeName (GetClangASTContext ().getASTContext(), m_clang_type);
    }
    return m_name;
}

void
Type::DumpTypeName(Stream *s)
{
    GetName().Dump(s, "<invalid-type-name>");
}


void
Type::DumpValue
(
    ExecutionContext *exe_ctx,
    Stream *s,
    const DataExtractor &data,
    uint32_t data_byte_offset,
    bool show_types,
    bool show_summary,
    bool verbose,
    lldb::Format format
)
{
    if (ResolveClangType(eResolveStateForward))
    {
        if (show_types)
        {
            s->PutChar('(');
            if (verbose)
                s->Printf("Type{0x%8.8llx} ", GetID());
            DumpTypeName (s);
            s->PutCString(") ");
        }

        ClangASTType::DumpValue (GetClangAST (),
                                               m_clang_type,
                                               exe_ctx,
                                               s,
                                               format == lldb::eFormatDefault ? GetFormat() : format,
                                               data,
                                               data_byte_offset,
                                               GetByteSize(),
                                               0, // Bitfield bit size
                                               0, // Bitfield bit offset
                                               show_types,
                                               show_summary,
                                               verbose,
                                               0);
    }
}

Type *
Type::GetEncodingType ()
{
    if (m_encoding_type == NULL && m_encoding_uid != LLDB_INVALID_UID)
        m_encoding_type = m_symbol_file->ResolveTypeUID(m_encoding_uid);
    return m_encoding_type;
}
    


uint32_t
Type::GetByteSize()
{
    if (m_byte_size == 0)
    {
        switch (m_encoding_uid_type)
        {
        case eEncodingInvalid:
        case eEncodingIsSyntheticUID:
            break;
        case eEncodingIsUID:
        case eEncodingIsConstUID:
        case eEncodingIsRestrictUID:
        case eEncodingIsVolatileUID:
        case eEncodingIsTypedefUID:
            {
                Type *encoding_type = GetEncodingType ();
                if (encoding_type)
                    m_byte_size = encoding_type->GetByteSize();
                if (m_byte_size == 0)
                {
                    uint32_t bit_width = ClangASTType::GetClangTypeBitWidth (GetClangAST(), GetClangLayoutType());
                    m_byte_size = (bit_width + 7 ) / 8;
                }
            }
            break;

        // If we are a pointer or reference, then this is just a pointer size;
        case eEncodingIsPointerUID:
        case eEncodingIsLValueReferenceUID:
        case eEncodingIsRValueReferenceUID:
            m_byte_size = m_symbol_file->GetClangASTContext().GetPointerBitSize() / 8;
            break;
        }
    }
    return m_byte_size;
}


uint32_t
Type::GetNumChildren (bool omit_empty_base_classes)
{
    if (ResolveClangType(eResolveStateForward))
    {
        return ClangASTContext::GetNumChildren (m_symbol_file->GetClangASTContext().getASTContext(),
                                                m_clang_type, 
                                                omit_empty_base_classes);
    }
    return 0;
}

bool
Type::IsAggregateType ()
{
    if (ResolveClangType(eResolveStateForward))
        return ClangASTContext::IsAggregateType (m_clang_type);
    return false;
}

lldb::TypeSP
Type::GetTypedefType()
{
    lldb::TypeSP type_sp;
    if (IsTypedef())
    {
        Type *typedef_type = m_symbol_file->ResolveTypeUID(m_encoding_uid);
        if (typedef_type)
            type_sp = typedef_type->shared_from_this();
    }
    return type_sp;
}



lldb::Format
Type::GetFormat ()
{
    // Make sure we resolve our type if it already hasn't been.
    if (!ResolveClangType(eResolveStateForward))
        return lldb::eFormatInvalid;
    return ClangASTType::GetFormat (m_clang_type);
}



lldb::Encoding
Type::GetEncoding (uint32_t &count)
{
    // Make sure we resolve our type if it already hasn't been.
    if (!ResolveClangType(eResolveStateForward))
        return lldb::eEncodingInvalid;

    return ClangASTType::GetEncoding (m_clang_type, count);
}



bool
Type::DumpValueInMemory
(
    ExecutionContext *exe_ctx,
    Stream *s,
    lldb::addr_t address,
    AddressType address_type,
    bool show_types,
    bool show_summary,
    bool verbose
)
{
    if (address != LLDB_INVALID_ADDRESS)
    {
        DataExtractor data;
        Target *target = NULL;
        if (exe_ctx)
            target = exe_ctx->GetTargetPtr();
        if (target)
            data.SetByteOrder (target->GetArchitecture().GetByteOrder());
        if (ReadFromMemory (exe_ctx, address, address_type, data))
        {
            DumpValue(exe_ctx, s, data, 0, show_types, show_summary, verbose);
            return true;
        }
    }
    return false;
}


bool
Type::ReadFromMemory (ExecutionContext *exe_ctx, lldb::addr_t addr, AddressType address_type, DataExtractor &data)
{
    if (address_type == eAddressTypeFile)
    {
        // Can't convert a file address to anything valid without more
        // context (which Module it came from)
        return false;
    }

    const uint32_t byte_size = GetByteSize();
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
            if (exe_ctx)
            {
                Process *process = exe_ctx->GetProcessPtr();
                if (process)
                {
                    Error error;
                    return exe_ctx->GetProcessPtr()->ReadMemory(addr, dst, byte_size, error) == byte_size;
                }
            }
        }
    }
    return false;
}


bool
Type::WriteToMemory (ExecutionContext *exe_ctx, lldb::addr_t addr, AddressType address_type, DataExtractor &data)
{
    return false;
}


TypeList*
Type::GetTypeList()
{
    return GetSymbolFile()->GetTypeList();
}

const Declaration &
Type::GetDeclaration () const
{
    return m_decl;
}

bool
Type::ResolveClangType (ResolveState clang_type_resolve_state)
{
    Type *encoding_type = NULL;
    if (m_clang_type == NULL)
    {
        encoding_type = GetEncodingType();
        if (encoding_type)
        {
            switch (m_encoding_uid_type)
            {
            case eEncodingIsUID:
                if (encoding_type->ResolveClangType(clang_type_resolve_state))
                {
                    m_clang_type = encoding_type->m_clang_type;
                    m_flags.clang_type_resolve_state = encoding_type->m_flags.clang_type_resolve_state;
                }
                break;

            case eEncodingIsConstUID:
                m_clang_type = ClangASTContext::AddConstModifier (encoding_type->GetClangForwardType());
                break;

            case eEncodingIsRestrictUID:
                m_clang_type = ClangASTContext::AddRestrictModifier (encoding_type->GetClangForwardType());
                break;

            case eEncodingIsVolatileUID:
                m_clang_type = ClangASTContext::AddVolatileModifier (encoding_type->GetClangForwardType());
                break;

            case eEncodingIsTypedefUID:
                m_clang_type = CreateClangTypedefType (this, encoding_type);
                // Clear the name so it can get fully qualified in case the
                // typedef is in a namespace.
                m_name.Clear();
                break;

            case eEncodingIsPointerUID:
                m_clang_type = CreateClangPointerType (encoding_type);
                break;

            case eEncodingIsLValueReferenceUID:
                m_clang_type = CreateClangLValueReferenceType (encoding_type);
                break;

            case eEncodingIsRValueReferenceUID:
                m_clang_type = CreateClangRValueReferenceType (encoding_type);
                break;

            default:
                assert(!"Unhandled encoding_data_type.");
                break;
            }
        }
        else
        {
            // We have no encoding type, return void?
            clang_type_t void_clang_type = GetClangASTContext().GetBuiltInType_void();
            switch (m_encoding_uid_type)
            {
            case eEncodingIsUID:
                m_clang_type = void_clang_type;
                break;

            case eEncodingIsConstUID:
                m_clang_type = ClangASTContext::AddConstModifier (void_clang_type);
                break;

            case eEncodingIsRestrictUID:
                m_clang_type = ClangASTContext::AddRestrictModifier (void_clang_type);
                break;

            case eEncodingIsVolatileUID:
                m_clang_type = ClangASTContext::AddVolatileModifier (void_clang_type);
                break;

            case eEncodingIsTypedefUID:
                m_clang_type = GetClangASTContext().CreateTypedefType (m_name.AsCString(), void_clang_type, NULL);
                break;

            case eEncodingIsPointerUID:
                m_clang_type = GetClangASTContext().CreatePointerType (void_clang_type);
                break;

            case eEncodingIsLValueReferenceUID:
                m_clang_type = GetClangASTContext().CreateLValueReferenceType (void_clang_type);
                break;

            case eEncodingIsRValueReferenceUID:
                m_clang_type = GetClangASTContext().CreateRValueReferenceType (void_clang_type);
                break;

            default:
                assert(!"Unhandled encoding_data_type.");
                break;
            }
        }
    }
    
    // Check if we have a forward reference to a class/struct/union/enum?
    if (m_clang_type && m_flags.clang_type_resolve_state < clang_type_resolve_state)
    {
        m_flags.clang_type_resolve_state = eResolveStateFull;
        if (!ClangASTType::IsDefined (m_clang_type))
        {
            // We have a forward declaration, we need to resolve it to a complete
            // definition.
            m_symbol_file->ResolveClangOpaqueTypeDefinition (m_clang_type);
        }
    }
    
    // If we have an encoding type, then we need to make sure it is 
    // resolved appropriately.
    if (m_encoding_uid != LLDB_INVALID_UID)
    {
        if (encoding_type == NULL)
            encoding_type = GetEncodingType();
        if (encoding_type)
        {
            ResolveState encoding_clang_type_resolve_state = clang_type_resolve_state;
            
            if (clang_type_resolve_state == eResolveStateLayout)
            {
                switch (m_encoding_uid_type)
                {
                case eEncodingIsPointerUID:
                case eEncodingIsLValueReferenceUID:
                case eEncodingIsRValueReferenceUID:
                    encoding_clang_type_resolve_state = eResolveStateForward;
                    break;
                default:
                    break;
                }
            }
            encoding_type->ResolveClangType (encoding_clang_type_resolve_state);
        }
    }
    return m_clang_type != NULL;
}
uint32_t
Type::GetEncodingMask ()
{
    uint32_t encoding_mask = 1u << m_encoding_uid_type;
    Type *encoding_type = GetEncodingType();
    assert (encoding_type != this);
    if (encoding_type)
        encoding_mask |= encoding_type->GetEncodingMask ();
    return encoding_mask;
}

clang_type_t 
Type::GetClangFullType ()
{
    ResolveClangType(eResolveStateFull);
    return m_clang_type;
}

clang_type_t 
Type::GetClangLayoutType ()
{
    ResolveClangType(eResolveStateLayout);
    return m_clang_type;
}

clang_type_t 
Type::GetClangForwardType ()
{
    ResolveClangType (eResolveStateForward);
    return m_clang_type;
}

clang::ASTContext *
Type::GetClangAST ()
{
    return GetClangASTContext().getASTContext();
}

ClangASTContext &
Type::GetClangASTContext ()
{
    return m_symbol_file->GetClangASTContext();
}

int
Type::Compare(const Type &a, const Type &b)
{
    // Just compare the UID values for now...
    lldb::user_id_t a_uid = a.GetID();
    lldb::user_id_t b_uid = b.GetID();
    if (a_uid < b_uid)
        return -1;
    if (a_uid > b_uid)
        return 1;
    return 0;
//  if (a.getQualType() == b.getQualType())
//      return 0;
}


void *
Type::CreateClangPointerType (Type *type)
{
    assert(type);
    return GetClangASTContext().CreatePointerType(type->GetClangForwardType());
}

void *
Type::CreateClangTypedefType (Type *typedef_type, Type *base_type)
{
    assert(typedef_type && base_type);
    return GetClangASTContext().CreateTypedefType (typedef_type->GetName().AsCString(), 
                                                   base_type->GetClangForwardType(), 
                                                   typedef_type->GetSymbolFile()->GetClangDeclContextContainingTypeUID(typedef_type->GetID()));
}

void *
Type::CreateClangLValueReferenceType (Type *type)
{
    assert(type);
    return GetClangASTContext().CreateLValueReferenceType(type->GetClangForwardType());
}

void *
Type::CreateClangRValueReferenceType (Type *type)
{
    assert(type);
    return GetClangASTContext().CreateRValueReferenceType (type->GetClangForwardType());
}

bool
Type::IsRealObjCClass()
{
    // For now we are just skipping ObjC classes that get made by hand from the runtime, because
    // those don't have any information.  We could extend this to only return true for "full 
    // definitions" if we can figure that out.
    
    if (ClangASTContext::IsObjCClassType(m_clang_type) && GetByteSize() != 0)
        return true;
    else
        return false;
}

ConstString
Type::GetQualifiedName ()
{
    ConstString qualified_name (ClangASTType::GetTypeNameForOpaqueQualType (GetClangASTContext ().getASTContext(), GetClangForwardType()).c_str());
    return qualified_name;
}


bool
Type::GetTypeScopeAndBasename (const char* &name_cstr,
                               std::string &scope,
                               std::string &basename,
                               TypeClass &type_class)
{
    // Protect against null c string.
    
    type_class = eTypeClassAny;

    if (name_cstr && name_cstr[0])
    {
        llvm::StringRef name_strref(name_cstr);
        if (name_strref.startswith("struct "))
        {
            name_cstr += 7;
            type_class = eTypeClassStruct;
        }
        else if (name_strref.startswith("class "))
        {
            name_cstr += 6;
            type_class = eTypeClassClass;
        }
        else if (name_strref.startswith("union "))
        {
            name_cstr += 6;
            type_class = eTypeClassUnion;
        }
        else if (name_strref.startswith("enum "))
        {
            name_cstr += 5;
            type_class = eTypeClassEnumeration;
        }
        else if (name_strref.startswith("typedef "))
        {
            name_cstr += 8;
            type_class = eTypeClassTypedef;
        }
        const char *basename_cstr = name_cstr;
        const char* namespace_separator = ::strstr (basename_cstr, "::");
        if (namespace_separator)
        {
            const char* template_arg_char = ::strchr (basename_cstr, '<');
            while (namespace_separator != NULL)
            {
                if (template_arg_char && namespace_separator > template_arg_char) // but namespace'd template arguments are still good to go
                    break;
                basename_cstr = namespace_separator + 2;
                namespace_separator = strstr(basename_cstr, "::");
            }
            if (basename_cstr > name_cstr)
            {
                scope.assign (name_cstr, basename_cstr - name_cstr);
                basename.assign (basename_cstr);
                return true;
            }
        }
    }
    return false;
}




TypeAndOrName::TypeAndOrName () : m_type_sp(), m_type_name()
{

}

TypeAndOrName::TypeAndOrName (TypeSP &in_type_sp) : m_type_sp(in_type_sp)
{
    if (in_type_sp)
        m_type_name = in_type_sp->GetName();
}

TypeAndOrName::TypeAndOrName (const char *in_type_str) : m_type_name(in_type_str)
{
}

TypeAndOrName::TypeAndOrName (const TypeAndOrName &rhs) : m_type_sp (rhs.m_type_sp), m_type_name (rhs.m_type_name)
{

}

TypeAndOrName::TypeAndOrName (ConstString &in_type_const_string) : m_type_name (in_type_const_string)
{
}

TypeAndOrName &
TypeAndOrName::operator= (const TypeAndOrName &rhs)
{
    if (this != &rhs)
    {
        m_type_name = rhs.m_type_name;
        m_type_sp = rhs.m_type_sp;
    }
    return *this;
}

ConstString
TypeAndOrName::GetName () const
{    
    if (m_type_sp)
        return m_type_sp->GetName();
    else
        return m_type_name;
}

void
TypeAndOrName::SetName (const ConstString &type_name)
{
    m_type_name = type_name;
}

void
TypeAndOrName::SetName (const char *type_name_cstr)
{
    m_type_name.SetCString (type_name_cstr);
}

void
TypeAndOrName::SetTypeSP (lldb::TypeSP type_sp)
{
    m_type_sp = type_sp;
    if (type_sp)
        m_type_name = type_sp->GetName();
}

bool
TypeAndOrName::IsEmpty()
{
    if (m_type_name || m_type_sp)
        return false;
    else
        return true;
}

TypeImpl::TypeImpl(const lldb_private::ClangASTType& clang_ast_type) :
    m_clang_ast_type(clang_ast_type.GetASTContext(), clang_ast_type.GetOpaqueQualType()),
    m_type_sp()
{}

TypeImpl::TypeImpl(const lldb::TypeSP& type) :
    m_clang_ast_type(type->GetClangAST(), type->GetClangFullType()),
    m_type_sp(type)
{
}

void
TypeImpl::SetType (const lldb::TypeSP &type_sp)
{
    if (type_sp)
    {
        m_clang_ast_type.SetClangType (type_sp->GetClangAST(), type_sp->GetClangFullType());
        m_type_sp = type_sp;
    }
    else
    {
        m_clang_ast_type.Clear();
        m_type_sp.reset();
    }
}

TypeImpl&
TypeImpl::operator = (const TypeImpl& rhs)
{
    if (*this != rhs)
    {
        m_clang_ast_type = rhs.m_clang_ast_type;
        m_type_sp = rhs.m_type_sp;
    }
    return *this;
}

clang::ASTContext*
TypeImpl::GetASTContext()
{
    if (!IsValid())
        return NULL;
    
    return m_clang_ast_type.GetASTContext();
}

lldb::clang_type_t
TypeImpl::GetOpaqueQualType()
{
    if (!IsValid())
        return NULL;
    
    return m_clang_ast_type.GetOpaqueQualType();
}

bool
TypeImpl::GetDescription (lldb_private::Stream &strm, 
                          lldb::DescriptionLevel description_level)
{
    if (m_clang_ast_type.IsValid())
    {
        ClangASTType::DumpTypeDescription (m_clang_ast_type.GetASTContext(), 
                                           m_clang_ast_type.GetOpaqueQualType(), 
                                           &strm);
    }
    else
    {
        strm.PutCString ("No value");
    }
    return true;
}

