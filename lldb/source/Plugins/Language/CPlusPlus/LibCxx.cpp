//===-- LibCxx.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "LibCxx.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/FormatEntity.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/DataFormatters/StringPrinter.h"
#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/DataFormatters/VectorIterator.h"
#include "lldb/Host/Endian.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

bool
lldb_private::formatters::LibcxxSmartPointerSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options)
{
    ValueObjectSP valobj_sp(valobj.GetNonSyntheticValue());
    if (!valobj_sp)
        return false;
    ValueObjectSP ptr_sp(valobj_sp->GetChildMemberWithName(ConstString("__ptr_"), true));
    ValueObjectSP count_sp(valobj_sp->GetChildAtNamePath( {ConstString("__cntrl_"),ConstString("__shared_owners_")} ));
    ValueObjectSP weakcount_sp(valobj_sp->GetChildAtNamePath( {ConstString("__cntrl_"),ConstString("__shared_weak_owners_")} ));
    
    if (!ptr_sp)
        return false;
    
    if (ptr_sp->GetValueAsUnsigned(0) == 0)
    {
        stream.Printf("nullptr");
        return true;
    }
    else
    {
        bool print_pointee = false;
        Error error;
        ValueObjectSP pointee_sp = ptr_sp->Dereference(error);
        if (pointee_sp && error.Success())
        {
            if (pointee_sp->DumpPrintableRepresentation(stream,
                                                        ValueObject::eValueObjectRepresentationStyleSummary,
                                                        lldb::eFormatInvalid,
                                                        ValueObject::ePrintableRepresentationSpecialCasesDisable,
                                                        false))
                print_pointee = true;
        }
        if (!print_pointee)
            stream.Printf("ptr = 0x%" PRIx64, ptr_sp->GetValueAsUnsigned(0));
    }
    
    if (count_sp)
        stream.Printf(" strong=%" PRIu64, 1+count_sp->GetValueAsUnsigned(0));

    if (weakcount_sp)
        stream.Printf(" weak=%" PRIu64, 1+weakcount_sp->GetValueAsUnsigned(0));
    
    return true;
}

lldb_private::formatters::LibcxxVectorBoolSyntheticFrontEnd::LibcxxVectorBoolSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp) :
    SyntheticChildrenFrontEnd(*valobj_sp),
    m_bool_type(),
    m_exe_ctx_ref(),
    m_count(0),
    m_base_data_address(0),
    m_children()
{
    if (valobj_sp)
    {
        Update();
        m_bool_type = valobj_sp->GetCompilerType().GetBasicTypeFromAST(lldb::eBasicTypeBool);
    }
}

size_t
lldb_private::formatters::LibcxxVectorBoolSyntheticFrontEnd::CalculateNumChildren ()
{
    return m_count;
}

lldb::ValueObjectSP
lldb_private::formatters::LibcxxVectorBoolSyntheticFrontEnd::GetChildAtIndex (size_t idx)
{
    auto iter = m_children.find(idx),
        end = m_children.end();
    if (iter != end)
        return iter->second;
    if (idx >= m_count)
        return ValueObjectSP();
    if (m_base_data_address == 0 || m_count == 0)
        return ValueObjectSP();
    if (!m_bool_type)
        return ValueObjectSP();
    size_t byte_idx = (idx >> 3); // divide by 8 to get byte index
    size_t bit_index = (idx & 7); // efficient idx % 8 for bit index
    lldb::addr_t byte_location = m_base_data_address + byte_idx;
    ProcessSP process_sp(m_exe_ctx_ref.GetProcessSP());
    if (!process_sp)
        return ValueObjectSP();
    uint8_t byte = 0;
    uint8_t mask = 0;
    Error err;
    size_t bytes_read = process_sp->ReadMemory(byte_location, &byte, 1, err);
    if (err.Fail() || bytes_read == 0)
        return ValueObjectSP();
    switch (bit_index)
    {
        case 0:
            mask = 1; break;
        case 1:
            mask = 2; break;
        case 2:
            mask = 4; break;
        case 3:
            mask = 8; break;
        case 4:
            mask = 16; break;
        case 5:
            mask = 32; break;
        case 6:
            mask = 64; break;
        case 7:
            mask = 128; break;
        default:
            return ValueObjectSP();
    }
    bool bit_set = ((byte & mask) != 0);
    DataBufferSP buffer_sp(new DataBufferHeap(m_bool_type.GetByteSize(nullptr), 0));
    if (bit_set && buffer_sp && buffer_sp->GetBytes())
        *(buffer_sp->GetBytes()) = 1; // regardless of endianness, anything non-zero is true
    StreamString name; name.Printf("[%" PRIu64 "]", (uint64_t)idx);
    ValueObjectSP retval_sp(CreateValueObjectFromData(name.GetData(), DataExtractor(buffer_sp, process_sp->GetByteOrder(), process_sp->GetAddressByteSize()), m_exe_ctx_ref, m_bool_type));
    if (retval_sp)
        m_children[idx] = retval_sp;
    return retval_sp;
}

/*(std::__1::vector<std::__1::allocator<bool> >) vBool = {
 __begin_ = 0x00000001001000e0
 __size_ = 56
 __cap_alloc_ = {
 std::__1::__libcpp_compressed_pair_imp<unsigned long, std::__1::allocator<unsigned long> > = {
 __first_ = 1
 }
 }
 }*/

bool
lldb_private::formatters::LibcxxVectorBoolSyntheticFrontEnd::Update()
{
    m_children.clear();
    ValueObjectSP valobj_sp = m_backend.GetSP();
    if (!valobj_sp)
        return false;
    m_exe_ctx_ref = valobj_sp->GetExecutionContextRef();
    ValueObjectSP size_sp(valobj_sp->GetChildMemberWithName(ConstString("__size_"), true));
    if (!size_sp)
        return false;
    m_count = size_sp->GetValueAsUnsigned(0);
    if (!m_count)
        return true;
    ValueObjectSP begin_sp(valobj_sp->GetChildMemberWithName(ConstString("__begin_"), true));
    if (!begin_sp)
    {
        m_count = 0;
        return false;
    }
    m_base_data_address = begin_sp->GetValueAsUnsigned(0);
    if (!m_base_data_address)
    {
        m_count = 0;
        return false;
    }
    return false;
}

bool
lldb_private::formatters::LibcxxVectorBoolSyntheticFrontEnd::MightHaveChildren ()
{
    return true;
}

size_t
lldb_private::formatters::LibcxxVectorBoolSyntheticFrontEnd::GetIndexOfChildWithName (const ConstString &name)
{
    if (!m_count || !m_base_data_address)
        return UINT32_MAX;
    const char* item_name = name.GetCString();
    uint32_t idx = ExtractIndexFromString(item_name);
    if (idx < UINT32_MAX && idx >= CalculateNumChildren())
        return UINT32_MAX;
    return idx;
}

lldb_private::formatters::LibcxxVectorBoolSyntheticFrontEnd::~LibcxxVectorBoolSyntheticFrontEnd() = default;

SyntheticChildrenFrontEnd*
lldb_private::formatters::LibcxxVectorBoolSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP valobj_sp)
{
    return (valobj_sp ? new LibcxxVectorBoolSyntheticFrontEnd(valobj_sp) : nullptr);
}

/*
 (lldb) fr var ibeg --raw --ptr-depth 1
 (std::__1::__map_iterator<std::__1::__tree_iterator<std::__1::pair<int, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > >, std::__1::__tree_node<std::__1::pair<int, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > >, void *> *, long> >) ibeg = {
 __i_ = {
 __ptr_ = 0x0000000100103870 {
 std::__1::__tree_node_base<void *> = {
 std::__1::__tree_end_node<std::__1::__tree_node_base<void *> *> = {
 __left_ = 0x0000000000000000
 }
 __right_ = 0x0000000000000000
 __parent_ = 0x00000001001038b0
 __is_black_ = true
 }
 __value_ = {
 first = 0
 second = { std::string }
 */

lldb_private::formatters::LibCxxMapIteratorSyntheticFrontEnd::LibCxxMapIteratorSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp) :
    SyntheticChildrenFrontEnd(*valobj_sp),
    m_pair_ptr()
{
    if (valobj_sp)
        Update();
}

bool
lldb_private::formatters::LibCxxMapIteratorSyntheticFrontEnd::Update()
{
    ValueObjectSP valobj_sp = m_backend.GetSP();
    if (!valobj_sp)
        return false;
    
    TargetSP target_sp(valobj_sp->GetTargetSP());
    
    if (!target_sp)
        return false;

    if (!valobj_sp)
        return false;

    // this must be a ValueObject* because it is a child of the ValueObject we are producing children for
    // it if were a ValueObjectSP, we would end up with a loop (iterator -> synthetic -> child -> parent == iterator)
    // and that would in turn leak memory by never allowing the ValueObjects to die and free their memory
    m_pair_ptr = valobj_sp->GetValueForExpressionPath(".__i_.__ptr_->__value_",
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      ValueObject::GetValueForExpressionPathOptions().DontCheckDotVsArrowSyntax().SetSyntheticChildrenTraversal(ValueObject::GetValueForExpressionPathOptions::SyntheticChildrenTraversal::None),
                                                      nullptr).get();
    
    return false;
}

size_t
lldb_private::formatters::LibCxxMapIteratorSyntheticFrontEnd::CalculateNumChildren ()
{
    return 2;
}

lldb::ValueObjectSP
lldb_private::formatters::LibCxxMapIteratorSyntheticFrontEnd::GetChildAtIndex (size_t idx)
{
    if (!m_pair_ptr)
        return lldb::ValueObjectSP();
    return m_pair_ptr->GetChildAtIndex(idx, true);
}

bool
lldb_private::formatters::LibCxxMapIteratorSyntheticFrontEnd::MightHaveChildren ()
{
    return true;
}

size_t
lldb_private::formatters::LibCxxMapIteratorSyntheticFrontEnd::GetIndexOfChildWithName (const ConstString &name)
{
    if (name == ConstString("first"))
        return 0;
    if (name == ConstString("second"))
        return 1;
    return UINT32_MAX;
}

lldb_private::formatters::LibCxxMapIteratorSyntheticFrontEnd::~LibCxxMapIteratorSyntheticFrontEnd ()
{
    // this will be deleted when its parent dies (since it's a child object)
    //delete m_pair_ptr;
}

SyntheticChildrenFrontEnd*
lldb_private::formatters::LibCxxMapIteratorSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP valobj_sp)
{
    return (valobj_sp ? new LibCxxMapIteratorSyntheticFrontEnd(valobj_sp) : nullptr);
}

/*
 (lldb) fr var ibeg --raw --ptr-depth 1 -T
 (std::__1::__wrap_iter<int *>) ibeg = {
 (std::__1::__wrap_iter<int *>::iterator_type) __i = 0x00000001001037a0 {
 (int) *__i = 1
 }
 }
*/

SyntheticChildrenFrontEnd*
lldb_private::formatters::LibCxxVectorIteratorSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP valobj_sp)
{
    static ConstString g_item_name;
    if (!g_item_name)
        g_item_name.SetCString("__i");
    return (valobj_sp ? new VectorIteratorSyntheticFrontEnd(valobj_sp, g_item_name) : nullptr);
}

lldb_private::formatters::LibcxxSharedPtrSyntheticFrontEnd::LibcxxSharedPtrSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp) :
    SyntheticChildrenFrontEnd(*valobj_sp),
    m_cntrl(nullptr),
    m_count_sp(),
    m_weak_count_sp(),
    m_ptr_size(0),
    m_byte_order(lldb::eByteOrderInvalid)
{
    if (valobj_sp)
        Update();
}

size_t
lldb_private::formatters::LibcxxSharedPtrSyntheticFrontEnd::CalculateNumChildren ()
{
    return (m_cntrl ? 1 : 0);
}

lldb::ValueObjectSP
lldb_private::formatters::LibcxxSharedPtrSyntheticFrontEnd::GetChildAtIndex (size_t idx)
{
    if (!m_cntrl)
        return lldb::ValueObjectSP();

    ValueObjectSP valobj_sp = m_backend.GetSP();
    if (!valobj_sp)
        return lldb::ValueObjectSP();

    if (idx == 0)
        return valobj_sp->GetChildMemberWithName(ConstString("__ptr_"), true);

    if (idx > 2)
        return lldb::ValueObjectSP();

    if (idx == 1)
    {
        if (!m_count_sp)
        {
            ValueObjectSP shared_owners_sp(m_cntrl->GetChildMemberWithName(ConstString("__shared_owners_"),true));
            if (!shared_owners_sp)
                return lldb::ValueObjectSP();
            uint64_t count = 1 + shared_owners_sp->GetValueAsUnsigned(0);
            DataExtractor data(&count, 8, m_byte_order, m_ptr_size);
            m_count_sp = CreateValueObjectFromData("count", data, valobj_sp->GetExecutionContextRef(), shared_owners_sp->GetCompilerType());
        }
        return m_count_sp;
    }
    else /* if (idx == 2) */
    {
        if (!m_weak_count_sp)
        {
            ValueObjectSP shared_weak_owners_sp(m_cntrl->GetChildMemberWithName(ConstString("__shared_weak_owners_"),true));
            if (!shared_weak_owners_sp)
                return lldb::ValueObjectSP();
            uint64_t count = 1 + shared_weak_owners_sp->GetValueAsUnsigned(0);
            DataExtractor data(&count, 8, m_byte_order, m_ptr_size);
            m_weak_count_sp = CreateValueObjectFromData("count", data, valobj_sp->GetExecutionContextRef(), shared_weak_owners_sp->GetCompilerType());
        }
        return m_weak_count_sp;
    }
}

bool
lldb_private::formatters::LibcxxSharedPtrSyntheticFrontEnd::Update()
{
    m_count_sp.reset();
    m_weak_count_sp.reset();
    m_cntrl = nullptr;
    
    ValueObjectSP valobj_sp = m_backend.GetSP();
    if (!valobj_sp)
        return false;
    
    TargetSP target_sp(valobj_sp->GetTargetSP());
    if (!target_sp)
        return false;
    
    m_byte_order = target_sp->GetArchitecture().GetByteOrder();
    m_ptr_size = target_sp->GetArchitecture().GetAddressByteSize();
    
    lldb::ValueObjectSP cntrl_sp(valobj_sp->GetChildMemberWithName(ConstString("__cntrl_"),true));
    
    m_cntrl = cntrl_sp.get(); // need to store the raw pointer to avoid a circular dependency
    return false;
}

bool
lldb_private::formatters::LibcxxSharedPtrSyntheticFrontEnd::MightHaveChildren ()
{
    return true;
}

size_t
lldb_private::formatters::LibcxxSharedPtrSyntheticFrontEnd::GetIndexOfChildWithName (const ConstString &name)
{
    if (name == ConstString("__ptr_"))
        return 0;
    if (name == ConstString("count"))
        return 1;
    if (name == ConstString("weak_count"))
        return 2;
    return UINT32_MAX;
}

lldb_private::formatters::LibcxxSharedPtrSyntheticFrontEnd::~LibcxxSharedPtrSyntheticFrontEnd() = default;

SyntheticChildrenFrontEnd*
lldb_private::formatters::LibcxxSharedPtrSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP valobj_sp)
{
    return (valobj_sp ? new LibcxxSharedPtrSyntheticFrontEnd(valobj_sp) : nullptr);
}

bool
lldb_private::formatters::LibcxxContainerSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options)
{
    if (valobj.IsPointerType())
    {
        uint64_t value = valobj.GetValueAsUnsigned(0);
        if (!value)
            return false;
        stream.Printf("0x%016" PRIx64 " ", value);
    }
    return FormatEntity::FormatStringRef("size=${svar%#}", stream, nullptr, nullptr, nullptr, &valobj, false, false);
}

// the field layout in a libc++ string (cap, side, data or data, size, cap)
enum LibcxxStringLayoutMode
{
    eLibcxxStringLayoutModeCSD = 0,
    eLibcxxStringLayoutModeDSC = 1,
    eLibcxxStringLayoutModeInvalid = 0xffff
};

// this function abstracts away the layout and mode details of a libc++ string
// and returns the address of the data and the size ready for callers to consume
static bool
ExtractLibcxxStringInfo (ValueObject& valobj,
                         ValueObjectSP &location_sp,
                         uint64_t& size)
{
    ValueObjectSP D(valobj.GetChildAtIndexPath({0,0,0,0}));
    if (!D)
        return false;
    
    ValueObjectSP layout_decider(D->GetChildAtIndexPath({0,0}));
    
    // this child should exist
    if (!layout_decider)
        return false;
    
    ConstString g_data_name("__data_");
    ConstString g_size_name("__size_");
    bool short_mode = false; // this means the string is in short-mode and the data is stored inline
    LibcxxStringLayoutMode layout = (layout_decider->GetName() == g_data_name) ? eLibcxxStringLayoutModeDSC : eLibcxxStringLayoutModeCSD;
    uint64_t size_mode_value = 0;
    
    if (layout == eLibcxxStringLayoutModeDSC)
    {
        ValueObjectSP size_mode(D->GetChildAtIndexPath({1,1,0}));
        if (!size_mode)
            return false;
        
        if (size_mode->GetName() != g_size_name)
        {
            // we are hitting the padding structure, move along
            size_mode = D->GetChildAtIndexPath({1,1,1});
            if (!size_mode)
                return false;
        }
        
        size_mode_value = (size_mode->GetValueAsUnsigned(0));
        short_mode = ((size_mode_value & 0x80) == 0);
    }
    else
    {
        ValueObjectSP size_mode(D->GetChildAtIndexPath({1,0,0}));
        if (!size_mode)
            return false;
        
        size_mode_value = (size_mode->GetValueAsUnsigned(0));
        short_mode = ((size_mode_value & 1) == 0);
    }
    
    if (short_mode)
    {
        ValueObjectSP s(D->GetChildAtIndex(1, true));
        if (!s)
            return false;
        location_sp = s->GetChildAtIndex((layout == eLibcxxStringLayoutModeDSC) ? 0 : 1, true);
        size = (layout == eLibcxxStringLayoutModeDSC) ? size_mode_value : ((size_mode_value >> 1) % 256);
        return (location_sp.get() != nullptr);
    }
    else
    {
        ValueObjectSP l(D->GetChildAtIndex(0, true));
        if (!l)
            return false;
        // we can use the layout_decider object as the data pointer
        location_sp = (layout == eLibcxxStringLayoutModeDSC) ? layout_decider : l->GetChildAtIndex(2, true);
        ValueObjectSP size_vo(l->GetChildAtIndex(1, true));
        if (!size_vo || !location_sp)
            return false;
        size = size_vo->GetValueAsUnsigned(0);
        return true;
    }
}

bool
lldb_private::formatters::LibcxxWStringSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& summary_options)
{
    uint64_t size = 0;
    ValueObjectSP location_sp;
    if (!ExtractLibcxxStringInfo(valobj, location_sp, size))
        return false;
    if (size == 0)
    {
        stream.Printf("L\"\"");
        return true;
    }
    if (!location_sp)
        return false;
    
    DataExtractor extractor;
    
    StringPrinter::ReadBufferAndDumpToStreamOptions options(valobj);
    
    if (summary_options.GetCapping() == TypeSummaryCapping::eTypeSummaryCapped)
    {
        const auto max_size = valobj.GetTargetSP()->GetMaximumSizeOfStringSummary();
        if (size > max_size)
        {
            size = max_size;
            options.SetIsTruncated(true);
        }
    }
    location_sp->GetPointeeData(extractor, 0, size);
    
    // std::wstring::size() is measured in 'characters', not bytes
    auto wchar_t_size = valobj.GetTargetSP()->GetScratchClangASTContext()->GetBasicType(lldb::eBasicTypeWChar).GetByteSize(nullptr);
    
    options.SetData(extractor);
    options.SetStream(&stream);
    options.SetPrefixToken("L");
    options.SetQuote('"');
    options.SetSourceSize(size);
    options.SetBinaryZeroIsTerminator(false);
    
    switch (wchar_t_size)
    {
        case 1:
            StringPrinter::ReadBufferAndDumpToStream<lldb_private::formatters::StringPrinter::StringElementType::UTF8>(options);
            break;
            
        case 2:
            lldb_private::formatters::StringPrinter::ReadBufferAndDumpToStream<lldb_private::formatters::StringPrinter::StringElementType::UTF16>(options);
            break;
            
        case 4:
            lldb_private::formatters::StringPrinter::ReadBufferAndDumpToStream<lldb_private::formatters::StringPrinter::StringElementType::UTF32>(options);
            break;
            
        default:
            stream.Printf("size for wchar_t is not valid");
            return true;
    }
    
    return true;
}

bool
lldb_private::formatters::LibcxxStringSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& summary_options)
{
    uint64_t size = 0;
    ValueObjectSP location_sp;
    
    if (!ExtractLibcxxStringInfo(valobj, location_sp, size))
        return false;
    
    if (size == 0)
    {
        stream.Printf("\"\"");
        return true;
    }
    
    if (!location_sp)
        return false;
    
    StringPrinter::ReadBufferAndDumpToStreamOptions options(valobj);
    
    DataExtractor extractor;
    if (summary_options.GetCapping() == TypeSummaryCapping::eTypeSummaryCapped)
    {
        const auto max_size = valobj.GetTargetSP()->GetMaximumSizeOfStringSummary();
        if (size > max_size)
        {
            size = max_size;
            options.SetIsTruncated(true);
        }
    }
    location_sp->GetPointeeData(extractor, 0, size);
    
    options.SetData(extractor);
    options.SetStream(&stream);
    options.SetPrefixToken(nullptr);
    options.SetQuote('"');
    options.SetSourceSize(size);
    options.SetBinaryZeroIsTerminator(false);
    StringPrinter::ReadBufferAndDumpToStream<StringPrinter::StringElementType::ASCII>(options);
    
    return true;
}
