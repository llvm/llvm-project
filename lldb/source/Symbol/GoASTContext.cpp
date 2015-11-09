//===-- GoASTContext.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <mutex>
#include <utility>
#include <vector>

#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/UniqueCStringMap.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/DataFormatters/StringPrinter.h"
#include "lldb/Symbol/CompilerType.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/GoASTContext.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Target.h"

#include "Plugins/ExpressionParser/Go/GoUserExpression.h"
#include "Plugins/SymbolFile/DWARF/DWARFASTParserGo.h"

using namespace lldb;

namespace lldb_private
{
class GoArray;
class GoFunction;
class GoStruct;

class GoType
{
  public:
    enum
    {
        KIND_BOOL = 1,
        KIND_INT = 2,
        KIND_INT8 = 3,
        KIND_INT16 = 4,
        KIND_INT32 = 5,
        KIND_INT64 = 6,
        KIND_UINT = 7,
        KIND_UINT8 = 8,
        KIND_UINT16 = 9,
        KIND_UINT32 = 10,
        KIND_UINT64 = 11,
        KIND_UINTPTR = 12,
        KIND_FLOAT32 = 13,
        KIND_FLOAT64 = 14,
        KIND_COMPLEX64 = 15,
        KIND_COMPLEX128 = 16,
        KIND_ARRAY = 17,
        KIND_CHAN = 18,
        KIND_FUNC = 19,
        KIND_INTERFACE = 20,
        KIND_MAP = 21,
        KIND_PTR = 22,
        KIND_SLICE = 23,
        KIND_STRING = 24,
        KIND_STRUCT = 25,
        KIND_UNSAFEPOINTER = 26,
        KIND_LLDB_VOID, // Extension for LLDB, not used by go runtime.
        KIND_MASK = (1 << 5) - 1,
        KIND_DIRECT_IFACE = 1 << 5
    };
    GoType(int kind, const ConstString &name)
        : m_kind(kind & KIND_MASK)
        , m_name(name)
    {
        if (m_kind == KIND_FUNC)
            m_kind = KIND_FUNC;
    }
    virtual ~GoType() {}

    int
    GetGoKind() const
    {
        return m_kind;
    }
    const ConstString &
    GetName() const
    {
        return m_name;
    }
    virtual CompilerType
    GetElementType() const
    {
        return CompilerType();
    }

    bool
    IsTypedef() const
    {
        switch (m_kind)
        {
            case KIND_CHAN:
            case KIND_MAP:
            case KIND_INTERFACE:
                return true;
            default:
                return false;
        }
    }

    GoArray *GetArray();
    GoFunction *GetFunction();
    GoStruct *GetStruct();

  private:
    int m_kind;
    ConstString m_name;
    GoType(const GoType &) = delete;
    const GoType &operator=(const GoType &) = delete;
};

class GoElem : public GoType
{
  public:
    GoElem(int kind, const ConstString &name, const CompilerType &elem)
        : GoType(kind, name)
        , m_elem(elem)
    {
    }
    virtual CompilerType
    GetElementType() const
    {
        return m_elem;
    }

  private:
    // TODO: should we store this differently?
    CompilerType m_elem;

    GoElem(const GoElem &) = delete;
    const GoElem &operator=(const GoElem &) = delete;
};

class GoArray : public GoElem
{
  public:
    GoArray(const ConstString &name, uint64_t length, const CompilerType &elem)
        : GoElem(KIND_ARRAY, name, elem)
        , m_length(length)
    {
    }

    uint64_t
    GetLength() const
    {
        return m_length;
    }

  private:
    uint64_t m_length;
    GoArray(const GoArray &) = delete;
    const GoArray &operator=(const GoArray &) = delete;
};

class GoFunction : public GoType
{
  public:
    GoFunction(const ConstString &name, bool is_variadic)
        : GoType(KIND_FUNC, name)
        , m_is_variadic(is_variadic)
    {
    }

    bool
    IsVariadic() const
    {
        return m_is_variadic;
    }

  private:
    bool m_is_variadic;
    GoFunction(const GoFunction &) = delete;
    const GoFunction &operator=(const GoFunction &) = delete;
};

class GoStruct : public GoType
{
  public:
    struct Field
    {
        Field(const ConstString &name, const CompilerType &type, uint64_t offset)
            : m_name(name)
            , m_type(type)
            , m_byte_offset(offset)
        {
        }
        ConstString m_name;
        CompilerType m_type;
        uint64_t m_byte_offset;
    };

    GoStruct(int kind, const ConstString &name, int64_t byte_size)
        : GoType(kind == 0 ? KIND_STRUCT : kind, name), m_is_complete(false), m_byte_size(byte_size)
    {
    }

    uint32_t
    GetNumFields() const
    {
        return m_fields.size();
    }

    const Field *
    GetField(uint32_t i) const
    {
        if (i < m_fields.size())
            return &m_fields[i];
        return nullptr;
    }

    void
    AddField(const ConstString &name, const CompilerType &type, uint64_t offset)
    {
        m_fields.push_back(Field(name, type, offset));
    }

    bool
    IsComplete() const
    {
        return m_is_complete;
    }

    void
    SetComplete()
    {
        m_is_complete = true;
    }

    int64_t
    GetByteSize() const
    {
        return m_byte_size;
    }

  private:
    bool m_is_complete;
    int64_t m_byte_size;
    std::vector<Field> m_fields;

    GoStruct(const GoStruct &) = delete;
    const GoStruct &operator=(const GoStruct &) = delete;
};

GoArray *
GoType::GetArray()
{
    if (m_kind == KIND_ARRAY)
    {
        return static_cast<GoArray *>(this);
    }
    return nullptr;
}

GoFunction *
GoType::GetFunction()
{
    if (m_kind == KIND_FUNC)
    {
        return static_cast<GoFunction *>(this);
    }
    return nullptr;
}

GoStruct *
GoType::GetStruct()
{
    switch (m_kind)
    {
        case KIND_STRING:
        case KIND_STRUCT:
        case KIND_SLICE:
            return static_cast<GoStruct *>(this);
    }
    return nullptr;
}
} // namespace lldb_private
using namespace lldb_private;

GoASTContext::GoASTContext()
    : TypeSystem(eKindGo)
    , m_pointer_byte_size(0)
    , m_int_byte_size(0)
    , m_types(new TypeMap)
{
}
GoASTContext::~GoASTContext()
{
}

//------------------------------------------------------------------
// PluginInterface functions
//------------------------------------------------------------------

ConstString
GoASTContext::GetPluginNameStatic()
{
    return ConstString("go");
}

ConstString
GoASTContext::GetPluginName()
{
    return GoASTContext::GetPluginNameStatic();
}

uint32_t
GoASTContext::GetPluginVersion()
{
    return 1;
}

lldb::TypeSystemSP
GoASTContext::CreateInstance (lldb::LanguageType language, Module *module, Target *target)
{
    if (language == eLanguageTypeGo)
    {
        ArchSpec arch;
        std::shared_ptr<GoASTContext> go_ast_sp;
        if (module)
        {
            arch = module->GetArchitecture();
            go_ast_sp = std::shared_ptr<GoASTContext>(new GoASTContext);
        }
        else if (target)
        {
            arch = target->GetArchitecture();
            go_ast_sp = std::shared_ptr<GoASTContextForExpr>(new GoASTContextForExpr(target->shared_from_this()));
        }

        if (arch.IsValid())
        {
            go_ast_sp->SetAddressByteSize(arch.GetAddressByteSize());
            return go_ast_sp;
        }
    }
    return lldb::TypeSystemSP();
}

void
GoASTContext::EnumerateSupportedLanguages(std::set<lldb::LanguageType> &languages_for_types, std::set<lldb::LanguageType> &languages_for_expressions)
{
    static std::vector<lldb::LanguageType> s_supported_languages_for_types({
        lldb::eLanguageTypeGo});
    
    static std::vector<lldb::LanguageType> s_supported_languages_for_expressions({});
    
    languages_for_types.insert(s_supported_languages_for_types.begin(), s_supported_languages_for_types.end());
    languages_for_expressions.insert(s_supported_languages_for_expressions.begin(), s_supported_languages_for_expressions.end());
}


void
GoASTContext::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   "AST context plug-in",
                                   CreateInstance,
                                   EnumerateSupportedLanguages);
}

void
GoASTContext::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}


//----------------------------------------------------------------------
// Tests
//----------------------------------------------------------------------

bool
GoASTContext::IsArrayType(lldb::opaque_compiler_type_t type, CompilerType *element_type, uint64_t *size, bool *is_incomplete)
{
    if (element_type)
        element_type->Clear();
    if (size)
        *size = 0;
    if (is_incomplete)
        *is_incomplete = false;
    GoArray *array = static_cast<GoType *>(type)->GetArray();
    if (array)
    {
        if (size)
            *size = array->GetLength();
        if (element_type)
            *element_type = array->GetElementType();
        return true;
    }
    return false;
}

bool
GoASTContext::IsVectorType(lldb::opaque_compiler_type_t type, CompilerType *element_type, uint64_t *size)
{
    if (element_type)
        element_type->Clear();
    if (size)
        *size = 0;
    return false;
}

bool
GoASTContext::IsAggregateType(lldb::opaque_compiler_type_t type)
{
    int kind = static_cast<GoType *>(type)->GetGoKind();
    if (kind < GoType::KIND_ARRAY)
        return false;
    if (kind == GoType::KIND_PTR)
        return false;
    if (kind == GoType::KIND_CHAN)
        return false;
    if (kind == GoType::KIND_MAP)
        return false;
    if (kind == GoType::KIND_STRING)
        return false;
    if (kind == GoType::KIND_UNSAFEPOINTER)
        return false;
    return true;
}

bool
GoASTContext::IsBeingDefined(lldb::opaque_compiler_type_t type)
{
    return false;
}

bool
GoASTContext::IsCharType(lldb::opaque_compiler_type_t type)
{
    // Go's DWARF doesn't distinguish between rune and int32.
    return false;
}

bool
GoASTContext::IsCompleteType(lldb::opaque_compiler_type_t type)
{
    if (!type)
        return false;
    GoType *t = static_cast<GoType *>(type);
    if (GoStruct *s = t->GetStruct())
        return s->IsComplete();
    if (t->IsTypedef() || t->GetGoKind() == GoType::KIND_PTR)
        return t->GetElementType().IsCompleteType();
    return true;
}

bool
GoASTContext::IsConst(lldb::opaque_compiler_type_t type)
{
    return false;
}

bool
GoASTContext::IsCStringType(lldb::opaque_compiler_type_t type, uint32_t &length)
{
    return false;
}

bool
GoASTContext::IsDefined(lldb::opaque_compiler_type_t type)
{
    return type != nullptr;
}

bool
GoASTContext::IsFloatingPointType(lldb::opaque_compiler_type_t type, uint32_t &count, bool &is_complex)
{
    int kind = static_cast<GoType *>(type)->GetGoKind();
    if (kind >= GoType::KIND_FLOAT32 && kind <= GoType::KIND_COMPLEX128)
    {
        if (kind >= GoType::KIND_COMPLEX64)
        {
            is_complex = true;
            count = 2;
        }
        else
        {
            is_complex = false;
            count = 1;
        }
        return true;
    }
    count = 0;
    is_complex = false;
    return false;
}

bool
GoASTContext::IsFunctionType(lldb::opaque_compiler_type_t type, bool *is_variadic_ptr)
{
    GoFunction *func = static_cast<GoType *>(type)->GetFunction();
    if (func)
    {
        if (is_variadic_ptr)
            *is_variadic_ptr = func->IsVariadic();
        return true;
    }
    if (is_variadic_ptr)
        *is_variadic_ptr = false;
    return false;
}

uint32_t
GoASTContext::IsHomogeneousAggregate(lldb::opaque_compiler_type_t type, CompilerType *base_type_ptr)
{
    return false;
}

size_t
GoASTContext::GetNumberOfFunctionArguments(lldb::opaque_compiler_type_t type)
{
    return 0;
}

CompilerType
GoASTContext::GetFunctionArgumentAtIndex(lldb::opaque_compiler_type_t type, const size_t index)
{
    return CompilerType();
}

bool
GoASTContext::IsFunctionPointerType(lldb::opaque_compiler_type_t type)
{
    return IsFunctionType(type);
}

bool
GoASTContext::IsIntegerType(lldb::opaque_compiler_type_t type, bool &is_signed)
{
    is_signed = false;
    // TODO: Is bool an integer?
    if (type)
    {
        int kind = static_cast<GoType *>(type)->GetGoKind();
        if (kind <= GoType::KIND_UINTPTR)
        {
            is_signed = (kind != GoType::KIND_BOOL) & (kind <= GoType::KIND_INT64);
            return true;
        }
    }
    return false;
}

bool
GoASTContext::IsPolymorphicClass(lldb::opaque_compiler_type_t type)
{
    return false;
}

bool
GoASTContext::IsPossibleDynamicType(lldb::opaque_compiler_type_t type,
                                    CompilerType *target_type, // Can pass NULL
                                    bool check_cplusplus, bool check_objc)
{
    if (target_type)
        target_type->Clear();
    if (type)
        return static_cast<GoType *>(type)->GetGoKind() == GoType::KIND_INTERFACE;
    return false;
}

bool
GoASTContext::IsRuntimeGeneratedType(lldb::opaque_compiler_type_t type)
{
    return false;
}

bool
GoASTContext::IsPointerType(lldb::opaque_compiler_type_t type, CompilerType *pointee_type)
{
    if (!type)
        return false;
    GoType *t = static_cast<GoType *>(type);
    if (pointee_type)
    {
        *pointee_type = t->GetElementType();
    }
    switch (t->GetGoKind())
    {
        case GoType::KIND_PTR:
        case GoType::KIND_UNSAFEPOINTER:
        case GoType::KIND_CHAN:
        case GoType::KIND_MAP:
            // TODO: is function a pointer?
            return true;
        default:
            return false;
    }
}

bool
GoASTContext::IsPointerOrReferenceType(lldb::opaque_compiler_type_t type, CompilerType *pointee_type)
{
    return IsPointerType(type, pointee_type);
}

bool
GoASTContext::IsReferenceType(lldb::opaque_compiler_type_t type, CompilerType *pointee_type, bool *is_rvalue)
{
    return false;
}

bool
GoASTContext::IsScalarType(lldb::opaque_compiler_type_t type)
{
    return !IsAggregateType(type);
}

bool
GoASTContext::IsTypedefType(lldb::opaque_compiler_type_t type)
{
    if (type)
        return static_cast<GoType *>(type)->IsTypedef();
    return false;
}

bool
GoASTContext::IsVoidType(lldb::opaque_compiler_type_t type)
{
    if (!type)
        return false;
    return static_cast<GoType *>(type)->GetGoKind() == GoType::KIND_LLDB_VOID;
}

bool
GoASTContext::SupportsLanguage (lldb::LanguageType language)
{
    return language == eLanguageTypeGo;
}

//----------------------------------------------------------------------
// Type Completion
//----------------------------------------------------------------------

bool
GoASTContext::GetCompleteType(lldb::opaque_compiler_type_t type)
{
    if (!type)
        return false;
    GoType *t = static_cast<GoType *>(type);
    if (t->IsTypedef() || t->GetGoKind() == GoType::KIND_PTR || t->GetArray())
        return t->GetElementType().GetCompleteType();
    if (GoStruct *s = t->GetStruct())
    {
        if (s->IsComplete())
            return true;
        CompilerType compiler_type(this, s);
        SymbolFile *symbols = GetSymbolFile();
        return symbols && symbols->CompleteType(compiler_type);
    }
    return true;
}

//----------------------------------------------------------------------
// AST related queries
//----------------------------------------------------------------------

uint32_t
GoASTContext::GetPointerByteSize()
{
    return m_pointer_byte_size;
}

//----------------------------------------------------------------------
// Accessors
//----------------------------------------------------------------------

ConstString
GoASTContext::GetTypeName(lldb::opaque_compiler_type_t type)
{
    if (type)
        return static_cast<GoType *>(type)->GetName();
    return ConstString();
}

uint32_t
GoASTContext::GetTypeInfo(lldb::opaque_compiler_type_t type, CompilerType *pointee_or_element_compiler_type)
{
    if (pointee_or_element_compiler_type)
        pointee_or_element_compiler_type->Clear();
    if (!type)
        return 0;
    GoType *t = static_cast<GoType *>(type);
    if (pointee_or_element_compiler_type)
        *pointee_or_element_compiler_type = t->GetElementType();
    int kind = t->GetGoKind();
    if (kind == GoType::KIND_ARRAY)
        return eTypeHasChildren | eTypeIsArray;
    if (kind < GoType::KIND_ARRAY)
    {
        uint32_t builtin_type_flags = eTypeIsBuiltIn | eTypeHasValue;
        if (kind < GoType::KIND_FLOAT32)
        {
            builtin_type_flags |= eTypeIsInteger | eTypeIsScalar;
            if (kind >= GoType::KIND_INT && kind <= GoType::KIND_INT64)
                builtin_type_flags |= eTypeIsSigned;
        }
        else
        {
            builtin_type_flags |= eTypeIsFloat;
            if (kind < GoType::KIND_COMPLEX64)
                builtin_type_flags |= eTypeIsComplex;
            else
                builtin_type_flags |= eTypeIsScalar;
        }
        return builtin_type_flags;
    }
    if (kind == GoType::KIND_STRING)
        return eTypeHasValue | eTypeIsBuiltIn;
    if (kind == GoType::KIND_FUNC)
        return eTypeIsFuncPrototype | eTypeHasValue;
    if (IsPointerType(type))
        return eTypeIsPointer | eTypeHasValue | eTypeHasChildren;
    if (kind == GoType::KIND_LLDB_VOID)
        return 0;
    return eTypeHasChildren | eTypeIsStructUnion;
}

lldb::TypeClass
GoASTContext::GetTypeClass(lldb::opaque_compiler_type_t type)
{
    if (!type)
        return eTypeClassInvalid;
    int kind = static_cast<GoType *>(type)->GetGoKind();
    if (kind == GoType::KIND_FUNC)
        return eTypeClassFunction;
    if (IsPointerType(type))
        return eTypeClassPointer;
    if (kind < GoType::KIND_COMPLEX64)
        return eTypeClassBuiltin;
    if (kind <= GoType::KIND_COMPLEX128)
        return eTypeClassComplexFloat;
    if (kind == GoType::KIND_LLDB_VOID)
        return eTypeClassInvalid;
    return eTypeClassStruct;
}

lldb::BasicType
GoASTContext::GetBasicTypeEnumeration(lldb::opaque_compiler_type_t type)
{
    ConstString name = GetTypeName(type);
    if (name)
    {
        typedef UniqueCStringMap<lldb::BasicType> TypeNameToBasicTypeMap;
        static TypeNameToBasicTypeMap g_type_map;
        static std::once_flag g_once_flag;
        std::call_once(g_once_flag, [](){
            // "void"
            g_type_map.Append(ConstString("void").GetCString(), eBasicTypeVoid);
            // "int"
            g_type_map.Append(ConstString("int").GetCString(), eBasicTypeInt);
            g_type_map.Append(ConstString("uint").GetCString(), eBasicTypeUnsignedInt);
            
            // Miscellaneous
            g_type_map.Append(ConstString("bool").GetCString(), eBasicTypeBool);

            // Others. Should these map to C types?
            g_type_map.Append(ConstString("byte").GetCString(), eBasicTypeOther);
            g_type_map.Append(ConstString("uint8").GetCString(), eBasicTypeOther);
            g_type_map.Append(ConstString("uint16").GetCString(), eBasicTypeOther);
            g_type_map.Append(ConstString("uint32").GetCString(), eBasicTypeOther);
            g_type_map.Append(ConstString("uint64").GetCString(), eBasicTypeOther);
            g_type_map.Append(ConstString("int8").GetCString(), eBasicTypeOther);
            g_type_map.Append(ConstString("int16").GetCString(), eBasicTypeOther);
            g_type_map.Append(ConstString("int32").GetCString(), eBasicTypeOther);
            g_type_map.Append(ConstString("int64").GetCString(), eBasicTypeOther);
            g_type_map.Append(ConstString("float32").GetCString(), eBasicTypeOther);
            g_type_map.Append(ConstString("float64").GetCString(), eBasicTypeOther);
            g_type_map.Append(ConstString("uintptr").GetCString(), eBasicTypeOther);

            g_type_map.Sort();
        });
        
        return g_type_map.Find(name.GetCString(), eBasicTypeInvalid);
    }
    return eBasicTypeInvalid;
}

lldb::LanguageType
GoASTContext::GetMinimumLanguage(lldb::opaque_compiler_type_t type)
{
    return lldb::eLanguageTypeGo;
}

unsigned
GoASTContext::GetTypeQualifiers(lldb::opaque_compiler_type_t type)
{
    return 0;
}

//----------------------------------------------------------------------
// Creating related types
//----------------------------------------------------------------------

CompilerType
GoASTContext::GetArrayElementType(lldb::opaque_compiler_type_t type, uint64_t *stride)
{
    GoArray *array = static_cast<GoType *>(type)->GetArray();
    if (array)
    {
        if (stride)
        {
            *stride = array->GetElementType().GetByteSize(nullptr);
        }
        return array->GetElementType();
    }
    return CompilerType();
}

CompilerType
GoASTContext::GetCanonicalType(lldb::opaque_compiler_type_t type)
{
    GoType *t = static_cast<GoType *>(type);
    if (t->IsTypedef())
        return t->GetElementType();
    return CompilerType(this, type);
}

CompilerType
GoASTContext::GetFullyUnqualifiedType(lldb::opaque_compiler_type_t type)
{
    return CompilerType(this, type);
}

// Returns -1 if this isn't a function of if the function doesn't have a prototype
// Returns a value >= 0 if there is a prototype.
int
GoASTContext::GetFunctionArgumentCount(lldb::opaque_compiler_type_t type)
{
    return GetNumberOfFunctionArguments(type);
}

CompilerType
GoASTContext::GetFunctionArgumentTypeAtIndex(lldb::opaque_compiler_type_t type, size_t idx)
{
    return GetFunctionArgumentAtIndex(type, idx);
}

CompilerType
GoASTContext::GetFunctionReturnType(lldb::opaque_compiler_type_t type)
{
    CompilerType result;
    if (type)
    {
        GoType *t = static_cast<GoType *>(type);
        if (t->GetGoKind() == GoType::KIND_FUNC)
            result = t->GetElementType();
    }
    return result;
}

size_t
GoASTContext::GetNumMemberFunctions(lldb::opaque_compiler_type_t type)
{
    return 0;
}

TypeMemberFunctionImpl
GoASTContext::GetMemberFunctionAtIndex(lldb::opaque_compiler_type_t type, size_t idx)
{
    return TypeMemberFunctionImpl();
}

CompilerType
GoASTContext::GetNonReferenceType(lldb::opaque_compiler_type_t type)
{
    return CompilerType(this, type);
}

CompilerType
GoASTContext::GetPointeeType(lldb::opaque_compiler_type_t type)
{
    if (!type)
        return CompilerType();
    return static_cast<GoType *>(type)->GetElementType();
}

CompilerType
GoASTContext::GetPointerType(lldb::opaque_compiler_type_t type)
{
    if (!type)
        return CompilerType();
    ConstString type_name = GetTypeName(type);
    ConstString pointer_name(std::string("*") + type_name.GetCString());
    GoType *pointer = (*m_types)[pointer_name].get();
    if (pointer == nullptr)
    {
        pointer = new GoElem(GoType::KIND_PTR, pointer_name, CompilerType(this, type));
        (*m_types)[pointer_name].reset(pointer);
    }
    return CompilerType(this, pointer);
}

// If the current object represents a typedef type, get the underlying type
CompilerType
GoASTContext::GetTypedefedType(lldb::opaque_compiler_type_t type)
{
    if (IsTypedefType(type))
        return static_cast<GoType *>(type)->GetElementType();
    return CompilerType();
}

//----------------------------------------------------------------------
// Create related types using the current type's AST
//----------------------------------------------------------------------
CompilerType
GoASTContext::GetBasicTypeFromAST(lldb::BasicType basic_type)
{
    return CompilerType();
}

CompilerType
GoASTContext::GetBuiltinTypeForEncodingAndBitSize (lldb::Encoding encoding,
                                                   size_t bit_size)
{
    return CompilerType();
}


//----------------------------------------------------------------------
// Exploring the type
//----------------------------------------------------------------------

uint64_t
GoASTContext::GetBitSize(lldb::opaque_compiler_type_t type, ExecutionContextScope *exe_scope)
{
    if (!type)
        return 0;
    if (!GetCompleteType(type))
        return 0;
    GoType *t = static_cast<GoType *>(type);
    GoArray *array = nullptr;
    switch (t->GetGoKind())
    {
        case GoType::KIND_BOOL:
        case GoType::KIND_INT8:
        case GoType::KIND_UINT8:
            return 8;
        case GoType::KIND_INT16:
        case GoType::KIND_UINT16:
            return 16;
        case GoType::KIND_INT32:
        case GoType::KIND_UINT32:
        case GoType::KIND_FLOAT32:
            return 32;
        case GoType::KIND_INT64:
        case GoType::KIND_UINT64:
        case GoType::KIND_FLOAT64:
        case GoType::KIND_COMPLEX64:
            return 64;
        case GoType::KIND_COMPLEX128:
            return 128;
        case GoType::KIND_INT:
        case GoType::KIND_UINT:
            return m_int_byte_size * 8;
        case GoType::KIND_UINTPTR:
        case GoType::KIND_FUNC: // I assume this is a pointer?
        case GoType::KIND_CHAN:
        case GoType::KIND_PTR:
        case GoType::KIND_UNSAFEPOINTER:
        case GoType::KIND_MAP:
            return m_pointer_byte_size * 8;
        case GoType::KIND_ARRAY:
            array = t->GetArray();
            return array->GetLength() * array->GetElementType().GetBitSize(exe_scope);
        case GoType::KIND_INTERFACE:
            return t->GetElementType().GetBitSize(exe_scope);
        case GoType::KIND_SLICE:
        case GoType::KIND_STRING:
        case GoType::KIND_STRUCT:
            return t->GetStruct()->GetByteSize() * 8;
        default:
            assert(false);
    }
    return 0;
}

lldb::Encoding
GoASTContext::GetEncoding(lldb::opaque_compiler_type_t type, uint64_t &count)
{
    count = 1;
    bool is_signed;
    if (IsIntegerType(type, is_signed))
        return is_signed ? lldb::eEncodingSint : eEncodingUint;
    bool is_complex;
    uint32_t complex_count;
    if (IsFloatingPointType(type, complex_count, is_complex))
    {
        count = complex_count;
        return eEncodingIEEE754;
    }
    if (IsPointerType(type))
        return eEncodingUint;
    return eEncodingInvalid;
}

lldb::Format
GoASTContext::GetFormat(lldb::opaque_compiler_type_t type)
{
    if (!type)
        return eFormatDefault;
    switch (static_cast<GoType *>(type)->GetGoKind())
    {
        case GoType::KIND_BOOL:
            return eFormatBoolean;
        case GoType::KIND_INT:
        case GoType::KIND_INT8:
        case GoType::KIND_INT16:
        case GoType::KIND_INT32:
        case GoType::KIND_INT64:
            return eFormatDecimal;
        case GoType::KIND_UINT:
        case GoType::KIND_UINT8:
        case GoType::KIND_UINT16:
        case GoType::KIND_UINT32:
        case GoType::KIND_UINT64:
            return eFormatUnsigned;
        case GoType::KIND_FLOAT32:
        case GoType::KIND_FLOAT64:
            return eFormatFloat;
        case GoType::KIND_COMPLEX64:
        case GoType::KIND_COMPLEX128:
            return eFormatComplexFloat;
        case GoType::KIND_UINTPTR:
        case GoType::KIND_CHAN:
        case GoType::KIND_PTR:
        case GoType::KIND_MAP:
        case GoType::KIND_UNSAFEPOINTER:
            return eFormatHex;
        case GoType::KIND_STRING:
            return eFormatCString;
        case GoType::KIND_ARRAY:
        case GoType::KIND_INTERFACE:
        case GoType::KIND_SLICE:
        case GoType::KIND_STRUCT:
        default:
            // Don't know how to display this.
            return eFormatBytes;
    }
}

size_t
GoASTContext::GetTypeBitAlign(lldb::opaque_compiler_type_t type)
{
    return 0;
}

uint32_t
GoASTContext::GetNumChildren(lldb::opaque_compiler_type_t type, bool omit_empty_base_classes)
{
    if (!type || !GetCompleteType(type))
        return 0;
    GoType *t = static_cast<GoType *>(type);
    if (t->GetGoKind() == GoType::KIND_PTR)
    {
        CompilerType elem = t->GetElementType();
        if (elem.IsAggregateType())
            return elem.GetNumChildren(omit_empty_base_classes);
        return 1;
    }
    else if (GoArray *array = t->GetArray())
    {
        return array->GetLength();
    }
    else if (t->IsTypedef())
    {
        return t->GetElementType().GetNumChildren(omit_empty_base_classes);
    }

    return GetNumFields(type);
}

uint32_t
GoASTContext::GetNumFields(lldb::opaque_compiler_type_t type)
{
    if (!type || !GetCompleteType(type))
        return 0;
    GoType *t = static_cast<GoType *>(type);
    if (t->IsTypedef())
        return t->GetElementType().GetNumFields();
    GoStruct *s = t->GetStruct();
    if (s)
        return s->GetNumFields();
    return 0;
}

CompilerType
GoASTContext::GetFieldAtIndex(lldb::opaque_compiler_type_t type, size_t idx, std::string &name, uint64_t *bit_offset_ptr,
                              uint32_t *bitfield_bit_size_ptr, bool *is_bitfield_ptr)
{
    if (bit_offset_ptr)
        *bit_offset_ptr = 0;
    if (bitfield_bit_size_ptr)
        *bitfield_bit_size_ptr = 0;
    if (is_bitfield_ptr)
        *is_bitfield_ptr = false;

    if (!type || !GetCompleteType(type))
        return CompilerType();

    GoType *t = static_cast<GoType *>(type);
    if (t->IsTypedef())
        return t->GetElementType().GetFieldAtIndex(idx, name, bit_offset_ptr, bitfield_bit_size_ptr, is_bitfield_ptr);

    GoStruct *s = t->GetStruct();
    if (s)
    {
        const auto *field = s->GetField(idx);
        if (field)
        {
            name = field->m_name.GetStringRef();
            if (bit_offset_ptr)
                *bit_offset_ptr = field->m_byte_offset * 8;
            return field->m_type;
        }
    }
    return CompilerType();
}

CompilerType
GoASTContext::GetChildCompilerTypeAtIndex(lldb::opaque_compiler_type_t type, ExecutionContext *exe_ctx, size_t idx, bool transparent_pointers,
                                          bool omit_empty_base_classes, bool ignore_array_bounds, std::string &child_name,
                                          uint32_t &child_byte_size, int32_t &child_byte_offset,
                                          uint32_t &child_bitfield_bit_size, uint32_t &child_bitfield_bit_offset,
                                          bool &child_is_base_class, bool &child_is_deref_of_parent, ValueObject *valobj, uint64_t &language_flags)
{
    child_name.clear();
    child_byte_size = 0;
    child_byte_offset = 0;
    child_bitfield_bit_size = 0;
    child_bitfield_bit_offset = 0;
    child_is_base_class = false;
    child_is_deref_of_parent = false;
    language_flags = 0;

    if (!type || !GetCompleteType(type))
        return CompilerType();

    GoType *t = static_cast<GoType *>(type);
    if (t->GetStruct())
    {
        uint64_t bit_offset;
        CompilerType ret = GetFieldAtIndex(type, idx, child_name, &bit_offset, nullptr, nullptr);
        child_byte_size = ret.GetByteSize(exe_ctx ? exe_ctx->GetBestExecutionContextScope() : nullptr);
        child_byte_offset = bit_offset / 8;
        return ret;
    }
    else if (t->GetGoKind() == GoType::KIND_PTR)
    {
        CompilerType pointee = t->GetElementType();
        if (!pointee.IsValid() || pointee.IsVoidType())
            return CompilerType();
        if (transparent_pointers && pointee.IsAggregateType())
        {
            bool tmp_child_is_deref_of_parent = false;
            return pointee.GetChildCompilerTypeAtIndex(exe_ctx, idx, transparent_pointers, omit_empty_base_classes,
                                                    ignore_array_bounds, child_name, child_byte_size, child_byte_offset,
                                                    child_bitfield_bit_size, child_bitfield_bit_offset,
                                                       child_is_base_class, tmp_child_is_deref_of_parent, valobj, language_flags);
        }
        else
        {
            child_is_deref_of_parent = true;
            const char *parent_name = valobj ? valobj->GetName().GetCString() : NULL;
            if (parent_name)
            {
                child_name.assign(1, '*');
                child_name += parent_name;
            }

            // We have a pointer to an simple type
            if (idx == 0 && pointee.GetCompleteType())
            {
                child_byte_size = pointee.GetByteSize(exe_ctx ? exe_ctx->GetBestExecutionContextScope() : NULL);
                child_byte_offset = 0;
                return pointee;
            }
        }
    }
    else if (GoArray *a = t->GetArray())
    {
        if (ignore_array_bounds || idx < a->GetLength())
        {
            CompilerType element_type = a->GetElementType();
            if (element_type.GetCompleteType())
            {
                char element_name[64];
                ::snprintf(element_name, sizeof(element_name), "[%zu]", idx);
                child_name.assign(element_name);
                child_byte_size = element_type.GetByteSize(exe_ctx ? exe_ctx->GetBestExecutionContextScope() : NULL);
                child_byte_offset = (int32_t)idx * (int32_t)child_byte_size;
                return element_type;
            }
        }
    }
    else if (t->IsTypedef())
    {
        return t->GetElementType().GetChildCompilerTypeAtIndex(
            exe_ctx, idx, transparent_pointers, omit_empty_base_classes, ignore_array_bounds, child_name,
            child_byte_size, child_byte_offset, child_bitfield_bit_size, child_bitfield_bit_offset, child_is_base_class,
            child_is_deref_of_parent, valobj, language_flags);
    }
    return CompilerType();
}

// Lookup a child given a name. This function will match base class names
// and member member names in "clang_type" only, not descendants.
uint32_t
GoASTContext::GetIndexOfChildWithName(lldb::opaque_compiler_type_t type, const char *name, bool omit_empty_base_classes)
{
    if (!type || !GetCompleteType(type))
        return UINT_MAX;

    GoType *t = static_cast<GoType *>(type);
    GoStruct *s = t->GetStruct();
    if (s)
    {
        for (uint32_t i = 0; i < s->GetNumFields(); ++i)
        {
            const GoStruct::Field *f = s->GetField(i);
            if (f->m_name.GetStringRef() == name)
                return i;
        }
    }
    else if (t->GetGoKind() == GoType::KIND_PTR || t->IsTypedef())
    {
        return t->GetElementType().GetIndexOfChildWithName(name, omit_empty_base_classes);
    }
    return UINT_MAX;
}

// Lookup a child member given a name. This function will match member names
// only and will descend into "clang_type" children in search for the first
// member in this class, or any base class that matches "name".
// TODO: Return all matches for a given name by returning a vector<vector<uint32_t>>
// so we catch all names that match a given child name, not just the first.
size_t
GoASTContext::GetIndexOfChildMemberWithName(lldb::opaque_compiler_type_t type, const char *name, bool omit_empty_base_classes,
                                            std::vector<uint32_t> &child_indexes)
{
    uint32_t index = GetIndexOfChildWithName(type, name, omit_empty_base_classes);
    if (index == UINT_MAX)
        return 0;
    child_indexes.push_back(index);
    return 1;
}

// Converts "s" to a floating point value and place resulting floating
// point bytes in the "dst" buffer.
size_t
GoASTContext::ConvertStringToFloatValue(lldb::opaque_compiler_type_t type, const char *s, uint8_t *dst, size_t dst_size)
{
    assert(false);
    return 0;
}
//----------------------------------------------------------------------
// Dumping types
//----------------------------------------------------------------------
void
GoASTContext::DumpValue(lldb::opaque_compiler_type_t type, ExecutionContext *exe_ctx, Stream *s, lldb::Format format,
                        const DataExtractor &data, lldb::offset_t data_offset, size_t data_byte_size,
                        uint32_t bitfield_bit_size, uint32_t bitfield_bit_offset, bool show_types, bool show_summary,
                        bool verbose, uint32_t depth)
{
    assert(false);
}

bool
GoASTContext::DumpTypeValue(lldb::opaque_compiler_type_t type, Stream *s, lldb::Format format, const DataExtractor &data,
                            lldb::offset_t byte_offset, size_t byte_size, uint32_t bitfield_bit_size,
                            uint32_t bitfield_bit_offset, ExecutionContextScope *exe_scope)
{
    if (!type)
        return false;
    if (IsAggregateType(type))
    {
        return false;
    }
    else
    {
        GoType *t = static_cast<GoType *>(type);
        if (t->IsTypedef())
        {
            CompilerType typedef_compiler_type = t->GetElementType();
            if (format == eFormatDefault)
                format = typedef_compiler_type.GetFormat();
            uint64_t typedef_byte_size = typedef_compiler_type.GetByteSize(exe_scope);

            return typedef_compiler_type.DumpTypeValue(
                s,
                format,              // The format with which to display the element
                data,                // Data buffer containing all bytes for this type
                byte_offset,         // Offset into "data" where to grab value from
                typedef_byte_size,   // Size of this type in bytes
                bitfield_bit_size,   // Size in bits of a bitfield value, if zero don't treat as a bitfield
                bitfield_bit_offset, // Offset in bits of a bitfield value if bitfield_bit_size != 0
                exe_scope);
        }

        uint32_t item_count = 1;
        // A few formats, we might need to modify our size and count for depending
        // on how we are trying to display the value...
        switch (format)
        {
            default:
            case eFormatBoolean:
            case eFormatBinary:
            case eFormatComplex:
            case eFormatCString: // NULL terminated C strings
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
        return data.Dump(s, byte_offset, format, byte_size, item_count, UINT32_MAX, LLDB_INVALID_ADDRESS,
                         bitfield_bit_size, bitfield_bit_offset, exe_scope);
    }
    return 0;
}

void
GoASTContext::DumpSummary(lldb::opaque_compiler_type_t type, ExecutionContext *exe_ctx, Stream *s, const DataExtractor &data,
                          lldb::offset_t data_offset, size_t data_byte_size)
{
    assert(false);
}

void
GoASTContext::DumpTypeDescription(lldb::opaque_compiler_type_t type)
{
    assert(false);
} // Dump to stdout

void
GoASTContext::DumpTypeDescription(lldb::opaque_compiler_type_t type, Stream *s)
{
    assert(false);
}

CompilerType
GoASTContext::CreateArrayType(const ConstString &name, const CompilerType &element_type, uint64_t length)
{
    GoType *type = new GoArray(name, length, element_type);
    (*m_types)[name].reset(type);
    return CompilerType(this, type);
}

CompilerType
GoASTContext::CreateBaseType(int go_kind, const lldb_private::ConstString &name, uint64_t byte_size)
{
    if (go_kind == GoType::KIND_UINT || go_kind == GoType::KIND_INT)
        m_int_byte_size = byte_size;
    GoType *type = new GoType(go_kind, name);
    (*m_types)[name].reset(type);
    return CompilerType(this, type);
}

CompilerType
GoASTContext::CreateTypedefType(int kind, const ConstString &name, CompilerType impl)
{
    GoType *type = new GoElem(kind, name, impl);
    (*m_types)[name].reset(type);
    return CompilerType(this, type);
}

CompilerType
GoASTContext::CreateVoidType(const lldb_private::ConstString &name)
{
    GoType *type = new GoType(GoType::KIND_LLDB_VOID, name);
    (*m_types)[name].reset(type);
    return CompilerType(this, type);
}

CompilerType
GoASTContext::CreateStructType(int kind, const lldb_private::ConstString &name, uint32_t byte_size)
{
    GoType *type = new GoStruct(kind, name, byte_size);
    (*m_types)[name].reset(type);
    return CompilerType(this, type);
}

void
GoASTContext::AddFieldToStruct(const lldb_private::CompilerType &struct_type, const lldb_private::ConstString &name,
                               const lldb_private::CompilerType &field_type, uint32_t byte_offset)
{
    if (!struct_type)
        return;
    GoASTContext *ast = llvm::dyn_cast_or_null<GoASTContext>(struct_type.GetTypeSystem());
    if (!ast)
        return;
    GoType *type = static_cast<GoType *>(struct_type.GetOpaqueQualType());
    if (GoStruct *s = type->GetStruct())
        s->AddField(name, field_type, byte_offset);
}

void
GoASTContext::CompleteStructType(const lldb_private::CompilerType &struct_type)
{
    if (!struct_type)
        return;
    GoASTContext *ast = llvm::dyn_cast_or_null<GoASTContext>(struct_type.GetTypeSystem());
    if (!ast)
        return;
    GoType *type = static_cast<GoType *>(struct_type.GetOpaqueQualType());
    if (GoStruct *s = type->GetStruct())
        s->SetComplete();
}

CompilerType
GoASTContext::CreateFunctionType(const lldb_private::ConstString &name, CompilerType *params, size_t params_count,
                                 bool is_variadic)
{
    GoType *type = new GoFunction(name, is_variadic);
    (*m_types)[name].reset(type);
    return CompilerType(this, type);
}

bool
GoASTContext::IsGoString(const lldb_private::CompilerType &type)
{
    if (!type.IsValid() || !llvm::dyn_cast_or_null<GoASTContext>(type.GetTypeSystem()))
        return false;
    return GoType::KIND_STRING == static_cast<GoType *>(type.GetOpaqueQualType())->GetGoKind();
}

bool
GoASTContext::IsGoSlice(const lldb_private::CompilerType &type)
{
    if (!type.IsValid() || !llvm::dyn_cast_or_null<GoASTContext>(type.GetTypeSystem()))
        return false;
    return GoType::KIND_SLICE == static_cast<GoType *>(type.GetOpaqueQualType())->GetGoKind();
}

bool
GoASTContext::IsGoInterface(const lldb_private::CompilerType &type)
{
    if (!type.IsValid() || !llvm::dyn_cast_or_null<GoASTContext>(type.GetTypeSystem()))
        return false;
    return GoType::KIND_INTERFACE == static_cast<GoType *>(type.GetOpaqueQualType())->GetGoKind();
}

bool
GoASTContext::IsPointerKind(uint8_t kind)
{
    return (kind & GoType::KIND_MASK) == GoType::KIND_PTR;
}

bool
GoASTContext::IsDirectIface(uint8_t kind)
{
    return (kind & GoType::KIND_DIRECT_IFACE) == GoType::KIND_DIRECT_IFACE;
}

DWARFASTParser *
GoASTContext::GetDWARFParser()
{
    if (!m_dwarf_ast_parser_ap)
        m_dwarf_ast_parser_ap.reset(new DWARFASTParserGo(*this));
    return m_dwarf_ast_parser_ap.get();
}

UserExpression *
GoASTContextForExpr::GetUserExpression(const char *expr, const char *expr_prefix, lldb::LanguageType language,
                                       Expression::ResultType desired_type, const EvaluateExpressionOptions &options)
{
    TargetSP target = m_target_wp.lock();
    if (target)
        return new GoUserExpression(*target, expr, expr_prefix, language, desired_type, options);
    return nullptr;
}
