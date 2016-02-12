//===-- NSDictionary.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
// C++ Includes
#include <mutex>

// Other libraries and framework includes
#include "clang/AST/DeclCXX.h"

// Project includes
#include "NSDictionary.h"

#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/Host/Endian.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Target/Language.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

std::map<ConstString, CXXFunctionSummaryFormat::Callback>&
NSDictionary_Additionals::GetAdditionalSummaries ()
{
    static std::map<ConstString, CXXFunctionSummaryFormat::Callback> g_map;
    return g_map;
}

std::map<ConstString, CXXSyntheticChildren::CreateFrontEndCallback>&
NSDictionary_Additionals::GetAdditionalSynthetics ()
{
    static std::map<ConstString, CXXSyntheticChildren::CreateFrontEndCallback> g_map;
    return g_map;
}

static CompilerType
GetLLDBNSPairType (TargetSP target_sp)
{
    CompilerType compiler_type;
    
    ClangASTContext *target_ast_context = target_sp->GetScratchClangASTContext();
    
    if (target_ast_context)
    {
        ConstString g___lldb_autogen_nspair("__lldb_autogen_nspair");
        
        compiler_type = target_ast_context->GetTypeForIdentifier<clang::CXXRecordDecl>(g___lldb_autogen_nspair);
        
        if (!compiler_type)
        {
            compiler_type = target_ast_context->CreateRecordType(NULL, lldb::eAccessPublic, g___lldb_autogen_nspair.GetCString(), clang::TTK_Struct, lldb::eLanguageTypeC);
            
            if (compiler_type)
            {
                ClangASTContext::StartTagDeclarationDefinition(compiler_type);
                CompilerType id_compiler_type = target_ast_context->GetBasicType (eBasicTypeObjCID);
                ClangASTContext::AddFieldToRecordType(compiler_type, "key", id_compiler_type, lldb::eAccessPublic, 0);
                ClangASTContext::AddFieldToRecordType(compiler_type, "value", id_compiler_type, lldb::eAccessPublic, 0);
                ClangASTContext::CompleteTagDeclarationDefinition(compiler_type);
            }
        }
    }
    return compiler_type;
}

namespace lldb_private {
    namespace  formatters {
        class NSDictionaryISyntheticFrontEnd : public SyntheticChildrenFrontEnd
        {
        public:
            NSDictionaryISyntheticFrontEnd (lldb::ValueObjectSP valobj_sp);

            ~NSDictionaryISyntheticFrontEnd() override;

            size_t
            CalculateNumChildren() override;
            
            lldb::ValueObjectSP
            GetChildAtIndex(size_t idx) override;
            
            bool
            Update() override;
            
            bool
            MightHaveChildren() override;
            
            size_t
            GetIndexOfChildWithName(const ConstString &name) override;

        private:
            struct DataDescriptor_32
            {
                uint32_t _used : 26;
                uint32_t _szidx : 6;
            };

            struct DataDescriptor_64
            {
                uint64_t _used : 58;
                uint32_t _szidx : 6;
            };

            struct DictionaryItemDescriptor
            {
                lldb::addr_t key_ptr;
                lldb::addr_t val_ptr;
                lldb::ValueObjectSP valobj_sp;
            };

            ExecutionContextRef m_exe_ctx_ref;
            uint8_t m_ptr_size;
            lldb::ByteOrder m_order;
            DataDescriptor_32 *m_data_32;
            DataDescriptor_64 *m_data_64;
            lldb::addr_t m_data_ptr;
            CompilerType m_pair_type;
            std::vector<DictionaryItemDescriptor> m_children;
        };
        
        class NSDictionaryMSyntheticFrontEnd : public SyntheticChildrenFrontEnd
        {
        public:
            NSDictionaryMSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp);

            ~NSDictionaryMSyntheticFrontEnd() override;

            size_t
            CalculateNumChildren() override;
            
            lldb::ValueObjectSP
            GetChildAtIndex(size_t idx) override;
            
            bool
            Update() override;
            
            bool
            MightHaveChildren() override;
            
            size_t
            GetIndexOfChildWithName(const ConstString &name) override;

        private:
            struct DataDescriptor_32
            {
                uint32_t _used : 26;
                uint32_t _kvo : 1;
                uint32_t _size;
                uint32_t _mutations;
                uint32_t _objs_addr;
                uint32_t _keys_addr;
            };

            struct DataDescriptor_64
            {
                uint64_t _used : 58;
                uint32_t _kvo : 1;
                uint64_t _size;
                uint64_t _mutations;
                uint64_t _objs_addr;
                uint64_t _keys_addr;
            };

            struct DictionaryItemDescriptor
            {
                lldb::addr_t key_ptr;
                lldb::addr_t val_ptr;
                lldb::ValueObjectSP valobj_sp;
            };

            ExecutionContextRef m_exe_ctx_ref;
            uint8_t m_ptr_size;
            lldb::ByteOrder m_order;
            DataDescriptor_32 *m_data_32;
            DataDescriptor_64 *m_data_64;
            CompilerType m_pair_type;
            std::vector<DictionaryItemDescriptor> m_children;
        };
    } // namespace formatters
} // namespace lldb_private

template<bool name_entries>
bool
lldb_private::formatters::NSDictionarySummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options)
{
    static ConstString g_TypeHint("NSDictionary");
    ProcessSP process_sp = valobj.GetProcessSP();
    if (!process_sp)
        return false;
    
    ObjCLanguageRuntime* runtime = (ObjCLanguageRuntime*)process_sp->GetLanguageRuntime(lldb::eLanguageTypeObjC);
    
    if (!runtime)
        return false;
    
    ObjCLanguageRuntime::ClassDescriptorSP descriptor(runtime->GetClassDescriptor(valobj));
    
    if (!descriptor.get() || !descriptor->IsValid())
        return false;
    
    uint32_t ptr_size = process_sp->GetAddressByteSize();
    bool is_64bit = (ptr_size == 8);
    
    lldb::addr_t valobj_addr = valobj.GetValueAsUnsigned(0);
    
    if (!valobj_addr)
        return false;
    
    uint64_t value = 0;
    
    ConstString class_name_cs = descriptor->GetClassName();
    const char* class_name = class_name_cs.GetCString();
    
    if (!class_name || !*class_name)
        return false;
    
    if (!strcmp(class_name,"__NSDictionaryI"))
    {
        Error error;
        value = process_sp->ReadUnsignedIntegerFromMemory(valobj_addr + ptr_size, ptr_size, 0, error);
        if (error.Fail())
            return false;
        value &= (is_64bit ? ~0xFC00000000000000UL : ~0xFC000000U);
    }
    else if (!strcmp(class_name,"__NSDictionaryM"))
    {
        Error error;
        value = process_sp->ReadUnsignedIntegerFromMemory(valobj_addr + ptr_size, ptr_size, 0, error);
        if (error.Fail())
            return false;
        value &= (is_64bit ? ~0xFC00000000000000UL : ~0xFC000000U);
    }
    /*else if (!strcmp(class_name,"__NSCFDictionary"))
     {
     Error error;
     value = process_sp->ReadUnsignedIntegerFromMemory(valobj_addr + (is_64bit ? 20 : 12), 4, 0, error);
     if (error.Fail())
     return false;
     if (is_64bit)
     value &= ~0x0f1f000000000000UL;
     }*/
    else
    {
        auto& map(NSDictionary_Additionals::GetAdditionalSummaries());
        auto iter = map.find(class_name_cs), end = map.end();
        if (iter != end)
            return iter->second(valobj, stream, options);
        else
            return false;
    }
    
    std::string prefix,suffix;
    if (Language* language = Language::FindPlugin(options.GetLanguage()))
    {
        if (!language->GetFormatterPrefixSuffix(valobj, g_TypeHint, prefix, suffix))
        {
            prefix.clear();
            suffix.clear();
        }
    }
    
    stream.Printf("%s%" PRIu64 " %s%s%s",
                  prefix.c_str(),
                  value,
                  "key/value pair",
                  value == 1 ? "" : "s",
                  suffix.c_str());
    return true;
}

SyntheticChildrenFrontEnd* lldb_private::formatters::NSDictionarySyntheticFrontEndCreator (CXXSyntheticChildren* synth, lldb::ValueObjectSP valobj_sp)
{
    lldb::ProcessSP process_sp (valobj_sp->GetProcessSP());
    if (!process_sp)
        return NULL;
    ObjCLanguageRuntime *runtime = (ObjCLanguageRuntime*)process_sp->GetLanguageRuntime(lldb::eLanguageTypeObjC);
    if (!runtime)
        return NULL;
    
    CompilerType valobj_type(valobj_sp->GetCompilerType());
    Flags flags(valobj_type.GetTypeInfo());
    
    if (flags.IsClear(eTypeIsPointer))
    {
        Error error;
        valobj_sp = valobj_sp->AddressOf(error);
        if (error.Fail() || !valobj_sp)
            return NULL;
    }
    
    ObjCLanguageRuntime::ClassDescriptorSP descriptor(runtime->GetClassDescriptor(*valobj_sp.get()));
    
    if (!descriptor.get() || !descriptor->IsValid())
        return NULL;
    
    ConstString class_name_cs = descriptor->GetClassName();
    const char* class_name = class_name_cs.GetCString();
    
    if (!class_name || !*class_name)
        return NULL;
    
    if (!strcmp(class_name,"__NSDictionaryI"))
    {
        return (new NSDictionaryISyntheticFrontEnd(valobj_sp));
    }
    else if (!strcmp(class_name,"__NSDictionaryM"))
    {
        return (new NSDictionaryMSyntheticFrontEnd(valobj_sp));
    }
    else
    {
        auto& map(NSDictionary_Additionals::GetAdditionalSynthetics());
        auto iter = map.find(class_name_cs), end = map.end();
        if (iter != end)
            return iter->second(synth, valobj_sp);
    }
    
    return nullptr;
}

lldb_private::formatters::NSDictionaryISyntheticFrontEnd::NSDictionaryISyntheticFrontEnd (lldb::ValueObjectSP valobj_sp) :
SyntheticChildrenFrontEnd(*valobj_sp.get()),
m_exe_ctx_ref(),
m_ptr_size(8),
m_order(lldb::eByteOrderInvalid),
m_data_32(NULL),
m_data_64(NULL),
m_pair_type()
{
}

lldb_private::formatters::NSDictionaryISyntheticFrontEnd::~NSDictionaryISyntheticFrontEnd ()
{
    delete m_data_32;
    m_data_32 = NULL;
    delete m_data_64;
    m_data_64 = NULL;
}

size_t
lldb_private::formatters::NSDictionaryISyntheticFrontEnd::GetIndexOfChildWithName (const ConstString &name)
{
    const char* item_name = name.GetCString();
    uint32_t idx = ExtractIndexFromString(item_name);
    if (idx < UINT32_MAX && idx >= CalculateNumChildren())
        return UINT32_MAX;
    return idx;
}

size_t
lldb_private::formatters::NSDictionaryISyntheticFrontEnd::CalculateNumChildren ()
{
    if (!m_data_32 && !m_data_64)
        return 0;
    return (m_data_32 ? m_data_32->_used : m_data_64->_used);
}

bool
lldb_private::formatters::NSDictionaryISyntheticFrontEnd::Update()
{
    m_children.clear();
    delete m_data_32;
    m_data_32 = NULL;
    delete m_data_64;
    m_data_64 = NULL;
    m_ptr_size = 0;
    ValueObjectSP valobj_sp = m_backend.GetSP();
    if (!valobj_sp)
        return false;
    m_exe_ctx_ref = valobj_sp->GetExecutionContextRef();
    Error error;
    error.Clear();
    lldb::ProcessSP process_sp(valobj_sp->GetProcessSP());
    if (!process_sp)
        return false;
    m_ptr_size = process_sp->GetAddressByteSize();
    m_order = process_sp->GetByteOrder();
    uint64_t data_location = valobj_sp->GetValueAsUnsigned(0) + m_ptr_size;
    if (m_ptr_size == 4)
    {
        m_data_32 = new DataDescriptor_32();
        process_sp->ReadMemory (data_location, m_data_32, sizeof(DataDescriptor_32), error);
    }
    else
    {
        m_data_64 = new DataDescriptor_64();
        process_sp->ReadMemory (data_location, m_data_64, sizeof(DataDescriptor_64), error);
    }
    if (error.Fail())
        return false;
    m_data_ptr = data_location + m_ptr_size;
    return false;
}

bool
lldb_private::formatters::NSDictionaryISyntheticFrontEnd::MightHaveChildren ()
{
    return true;
}

lldb::ValueObjectSP
lldb_private::formatters::NSDictionaryISyntheticFrontEnd::GetChildAtIndex (size_t idx)
{
    uint32_t num_children = CalculateNumChildren();
    
    if (idx >= num_children)
        return lldb::ValueObjectSP();
    
    if (m_children.empty())
    {
        // do the scan phase
        lldb::addr_t key_at_idx = 0, val_at_idx = 0;
        
        uint32_t tries = 0;
        uint32_t test_idx = 0;
        
        while(tries < num_children)
        {
            key_at_idx = m_data_ptr + (2*test_idx * m_ptr_size);
            val_at_idx = key_at_idx + m_ptr_size;
            ProcessSP process_sp = m_exe_ctx_ref.GetProcessSP();
            if (!process_sp)
                return lldb::ValueObjectSP();
            Error error;
            key_at_idx = process_sp->ReadPointerFromMemory(key_at_idx, error);
            if (error.Fail())
                return lldb::ValueObjectSP();
            val_at_idx = process_sp->ReadPointerFromMemory(val_at_idx, error);
            if (error.Fail())
                return lldb::ValueObjectSP();
            
            test_idx++;
            
            if (!key_at_idx || !val_at_idx)
                continue;
            tries++;
            
            DictionaryItemDescriptor descriptor = {key_at_idx,val_at_idx,lldb::ValueObjectSP()};
            
            m_children.push_back(descriptor);
        }
    }
    
    if (idx >= m_children.size()) // should never happen
        return lldb::ValueObjectSP();
    
    DictionaryItemDescriptor &dict_item = m_children[idx];
    if (!dict_item.valobj_sp)
    {
        if (!m_pair_type.IsValid())
        {
            TargetSP target_sp(m_backend.GetTargetSP());
            if (!target_sp)
                return ValueObjectSP();
            m_pair_type = GetLLDBNSPairType(target_sp);
        }
        if (!m_pair_type.IsValid())
            return ValueObjectSP();
        
        DataBufferSP buffer_sp(new DataBufferHeap(2*m_ptr_size,0));
        
        if (m_ptr_size == 8)
        {
            uint64_t *data_ptr = (uint64_t *)buffer_sp->GetBytes();
            *data_ptr = dict_item.key_ptr;
            *(data_ptr+1) = dict_item.val_ptr;
        }
        else
        {
            uint32_t *data_ptr = (uint32_t *)buffer_sp->GetBytes();
            *data_ptr = dict_item.key_ptr;
            *(data_ptr+1) = dict_item.val_ptr;
        }
        
        StreamString idx_name;
        idx_name.Printf("[%" PRIu64 "]", (uint64_t)idx);
        DataExtractor data(buffer_sp, m_order, m_ptr_size);
        dict_item.valobj_sp = CreateValueObjectFromData(idx_name.GetData(),
                                                        data,
                                                        m_exe_ctx_ref,
                                                        m_pair_type);
    }
    return dict_item.valobj_sp;
}

lldb_private::formatters::NSDictionaryMSyntheticFrontEnd::NSDictionaryMSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp) :
SyntheticChildrenFrontEnd(*valobj_sp.get()),
m_exe_ctx_ref(),
m_ptr_size(8),
m_order(lldb::eByteOrderInvalid),
m_data_32(NULL),
m_data_64(NULL),
m_pair_type()
{
}

lldb_private::formatters::NSDictionaryMSyntheticFrontEnd::~NSDictionaryMSyntheticFrontEnd ()
{
    delete m_data_32;
    m_data_32 = NULL;
    delete m_data_64;
    m_data_64 = NULL;
}

size_t
lldb_private::formatters::NSDictionaryMSyntheticFrontEnd::GetIndexOfChildWithName (const ConstString &name)
{
    const char* item_name = name.GetCString();
    uint32_t idx = ExtractIndexFromString(item_name);
    if (idx < UINT32_MAX && idx >= CalculateNumChildren())
        return UINT32_MAX;
    return idx;
}

size_t
lldb_private::formatters::NSDictionaryMSyntheticFrontEnd::CalculateNumChildren ()
{
    if (!m_data_32 && !m_data_64)
        return 0;
    return (m_data_32 ? m_data_32->_used : m_data_64->_used);
}

bool
lldb_private::formatters::NSDictionaryMSyntheticFrontEnd::Update()
{
    m_children.clear();
    ValueObjectSP valobj_sp = m_backend.GetSP();
    m_ptr_size = 0;
    delete m_data_32;
    m_data_32 = NULL;
    delete m_data_64;
    m_data_64 = NULL;
    if (!valobj_sp)
        return false;
    m_exe_ctx_ref = valobj_sp->GetExecutionContextRef();
    Error error;
    error.Clear();
    lldb::ProcessSP process_sp(valobj_sp->GetProcessSP());
    if (!process_sp)
        return false;
    m_ptr_size = process_sp->GetAddressByteSize();
    m_order = process_sp->GetByteOrder();
    uint64_t data_location = valobj_sp->GetValueAsUnsigned(0) + m_ptr_size;
    if (m_ptr_size == 4)
    {
        m_data_32 = new DataDescriptor_32();
        process_sp->ReadMemory (data_location, m_data_32, sizeof(DataDescriptor_32), error);
    }
    else
    {
        m_data_64 = new DataDescriptor_64();
        process_sp->ReadMemory (data_location, m_data_64, sizeof(DataDescriptor_64), error);
    }
    if (error.Fail())
        return false;
    return false;
}

bool
lldb_private::formatters::NSDictionaryMSyntheticFrontEnd::MightHaveChildren ()
{
    return true;
}

lldb::ValueObjectSP
lldb_private::formatters::NSDictionaryMSyntheticFrontEnd::GetChildAtIndex (size_t idx)
{
    lldb::addr_t m_keys_ptr = (m_data_32 ? m_data_32->_keys_addr : m_data_64->_keys_addr);
    lldb::addr_t m_values_ptr = (m_data_32 ? m_data_32->_objs_addr : m_data_64->_objs_addr);
    
    uint32_t num_children = CalculateNumChildren();
    
    if (idx >= num_children)
        return lldb::ValueObjectSP();
    
    if (m_children.empty())
    {
        // do the scan phase
        lldb::addr_t key_at_idx = 0, val_at_idx = 0;
        
        uint32_t tries = 0;
        uint32_t test_idx = 0;
        
        while(tries < num_children)
        {
            key_at_idx = m_keys_ptr + (test_idx * m_ptr_size);
            val_at_idx = m_values_ptr + (test_idx * m_ptr_size);;
            ProcessSP process_sp = m_exe_ctx_ref.GetProcessSP();
            if (!process_sp)
                return lldb::ValueObjectSP();
            Error error;
            key_at_idx = process_sp->ReadPointerFromMemory(key_at_idx, error);
            if (error.Fail())
                return lldb::ValueObjectSP();
            val_at_idx = process_sp->ReadPointerFromMemory(val_at_idx, error);
            if (error.Fail())
                return lldb::ValueObjectSP();
            
            test_idx++;
            
            if (!key_at_idx || !val_at_idx)
                continue;
            tries++;
            
            DictionaryItemDescriptor descriptor = {key_at_idx,val_at_idx,lldb::ValueObjectSP()};
            
            m_children.push_back(descriptor);
        }
    }
    
    if (idx >= m_children.size()) // should never happen
        return lldb::ValueObjectSP();
    
    DictionaryItemDescriptor &dict_item = m_children[idx];
    if (!dict_item.valobj_sp)
    {
        if (!m_pair_type.IsValid())
        {
            TargetSP target_sp(m_backend.GetTargetSP());
            if (!target_sp)
                return ValueObjectSP();
            m_pair_type = GetLLDBNSPairType(target_sp);
        }
        if (!m_pair_type.IsValid())
            return ValueObjectSP();
        
        DataBufferSP buffer_sp(new DataBufferHeap(2*m_ptr_size,0));
        
        if (m_ptr_size == 8)
        {
            uint64_t *data_ptr = (uint64_t *)buffer_sp->GetBytes();
            *data_ptr = dict_item.key_ptr;
            *(data_ptr+1) = dict_item.val_ptr;
        }
        else
        {
            uint32_t *data_ptr = (uint32_t *)buffer_sp->GetBytes();
            *data_ptr = dict_item.key_ptr;
            *(data_ptr+1) = dict_item.val_ptr;
        }
        
        StreamString idx_name;
        idx_name.Printf("[%" PRIu64 "]", (uint64_t)idx);
        DataExtractor data(buffer_sp, m_order, m_ptr_size);
        dict_item.valobj_sp = CreateValueObjectFromData(idx_name.GetData(),
                                                        data,
                                                        m_exe_ctx_ref,
                                                        m_pair_type);
    }
    return dict_item.valobj_sp;
}

template bool
lldb_private::formatters::NSDictionarySummaryProvider<true> (ValueObject&, Stream&, const TypeSummaryOptions&);

template bool
lldb_private::formatters::NSDictionarySummaryProvider<false> (ValueObject&, Stream&, const TypeSummaryOptions&);
