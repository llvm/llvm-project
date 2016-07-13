//===-- LibCxxList.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "LibCxx.h"

#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/Host/Endian.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

namespace {

    class ListEntry
    {
    public:
        ListEntry() = default;
        ListEntry (ValueObjectSP entry_sp) : m_entry_sp(entry_sp) {}
        ListEntry(const ListEntry& rhs) = default;
        ListEntry (ValueObject* entry) : m_entry_sp(entry ? entry->GetSP() : ValueObjectSP()) {}

        ListEntry
        next ()
        {
            if (!m_entry_sp)
                return ListEntry();
            return ListEntry(m_entry_sp->GetChildAtIndexPath({0,1}));
        }

        ListEntry
        prev ()
        {
            if (!m_entry_sp)
                return ListEntry();
            return ListEntry(m_entry_sp->GetChildAtIndexPath({0,0}));
        }

        uint64_t
        value () const
        {
            if (!m_entry_sp)
                return 0;
            return m_entry_sp->GetValueAsUnsigned(0);
        }

        bool
        null()
        {
            return (value() == 0);
        }

        explicit operator bool ()
        {
            return GetEntry() && !null();
        }

        ValueObjectSP
        GetEntry ()
        {
            return m_entry_sp;
        }

        void
        SetEntry (ValueObjectSP entry)
        {
            m_entry_sp = entry;
        }

        bool
        operator == (const ListEntry& rhs) const
        {
            return value() == rhs.value();
        }

        bool
        operator != (const ListEntry& rhs) const
        {
            return !(*this == rhs);
        }

    private:
        ValueObjectSP m_entry_sp;
    };
    
    class ListIterator
    {
    public:
        ListIterator() = default;
        ListIterator (ListEntry entry) : m_entry(entry) {}
        ListIterator (ValueObjectSP entry) : m_entry(entry) {}
        ListIterator(const ListIterator& rhs) = default;
        ListIterator (ValueObject* entry) : m_entry(entry) {}
        
        ValueObjectSP
        value ()
        {
            return m_entry.GetEntry();
        }
        
        ValueObjectSP
        advance (size_t count)
        {
            if (count == 0)
                return m_entry.GetEntry();
            if (count == 1)
            {
                next ();
                return m_entry.GetEntry();
            }
            while (count > 0)
            {
                next ();
                count--;
                if (m_entry.null())
                    return lldb::ValueObjectSP();
            }
            return m_entry.GetEntry();
        }
        
        bool
        operator == (const ListIterator& rhs) const
        {
            return (rhs.m_entry == m_entry);
        }
        
    protected:
        void
        next ()
        {
            m_entry = m_entry.next();
        }
        
        void
        prev ()
        {
            m_entry = m_entry.prev();
        }
        
    private:
        ListEntry m_entry;
    };

} // end anonymous namespace

namespace lldb_private {
    namespace formatters {
        class LibcxxStdListSyntheticFrontEnd : public SyntheticChildrenFrontEnd
        {
        public:
            LibcxxStdListSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp);

            ~LibcxxStdListSyntheticFrontEnd() override = default;

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
            bool
            HasLoop(size_t count);
            
            size_t m_list_capping_size;
            static const bool g_use_loop_detect = true;

            size_t m_loop_detected; // The number of elements that have had loop detection run over them.
            ListEntry m_slow_runner; // Used for loop detection
            ListEntry m_fast_runner; // Used for loop detection

            lldb::addr_t m_node_address;
            ValueObject* m_head;
            ValueObject* m_tail;
            CompilerType m_element_type;
            size_t m_count;
            std::map<size_t, ListIterator> m_iterators;
        };
    } // namespace formatters
} // namespace lldb_private

lldb_private::formatters::LibcxxStdListSyntheticFrontEnd::LibcxxStdListSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp) :
    SyntheticChildrenFrontEnd(*valobj_sp),
    m_list_capping_size(0),
    m_loop_detected(0),
    m_node_address(),
    m_head(nullptr),
    m_tail(nullptr),
    m_element_type(),
    m_count(UINT32_MAX),
    m_iterators()
{
    if (valobj_sp)
        Update();
}

bool
lldb_private::formatters::LibcxxStdListSyntheticFrontEnd::HasLoop(size_t count)
{
    if (!g_use_loop_detect)
        return false;
    // don't bother checking for a loop if we won't actually need to jump nodes
    if (m_count < 2)
        return false;

    if (m_loop_detected == 0)
    {
        // This is the first time we are being run (after the last update). Set up the loop
        // invariant for the first element.
        m_slow_runner = ListEntry(m_head).next();
        m_fast_runner = m_slow_runner.next();
        m_loop_detected = 1;
    }

    // Loop invariant:
    // Loop detection has been run over the first m_loop_detected elements. If m_slow_runner ==
    // m_fast_runner then the loop has been detected after m_loop_detected elements.
    const size_t steps_to_run = std::min(count,m_count);
    while (m_loop_detected < steps_to_run
            && m_slow_runner
            && m_fast_runner
            && m_slow_runner != m_fast_runner) {

        m_slow_runner = m_slow_runner.next();
        m_fast_runner = m_fast_runner.next().next();
        m_loop_detected++;
    }
    if (count <= m_loop_detected)
        return false; // No loop in the first m_loop_detected elements.
    if (!m_slow_runner || !m_fast_runner)
        return false; // Reached the end of the list. Definitely no loops.
    return m_slow_runner == m_fast_runner;
}

size_t
lldb_private::formatters::LibcxxStdListSyntheticFrontEnd::CalculateNumChildren ()
{
    if (m_count != UINT32_MAX)
        return m_count;
    if (!m_head || !m_tail || m_node_address == 0)
        return 0;
    ValueObjectSP size_alloc(m_backend.GetChildMemberWithName(ConstString("__size_alloc_"), true));
    if (size_alloc)
    {
        ValueObjectSP first(size_alloc->GetChildMemberWithName(ConstString("__first_"), true));
        if (first)
        {
            m_count = first->GetValueAsUnsigned(UINT32_MAX);
        }
    }
    if (m_count != UINT32_MAX)
    {
        return m_count;
    }
    else
    {
        uint64_t next_val = m_head->GetValueAsUnsigned(0);
        uint64_t prev_val = m_tail->GetValueAsUnsigned(0);
        if (next_val == 0 || prev_val == 0)
            return 0;
        if (next_val == m_node_address)
            return 0;
        if (next_val == prev_val)
            return 1;
        uint64_t size = 2;
        ListEntry current(m_head);
        while (current.next() && current.next().value() != m_node_address)
        {
            size++;
            current = current.next();
            if (size > m_list_capping_size)
                break;
        }
        return m_count = (size-1);
    }
}

lldb::ValueObjectSP
lldb_private::formatters::LibcxxStdListSyntheticFrontEnd::GetChildAtIndex (size_t idx)
{
    if (idx >= CalculateNumChildren())
        return lldb::ValueObjectSP();
    
    if (!m_head || !m_tail || m_node_address == 0)
        return lldb::ValueObjectSP();
    
    if (HasLoop(idx+1))
        return lldb::ValueObjectSP();
    
    size_t actual_advance = idx;
    
    ListIterator current(m_head);
    if (idx > 0)
    {
        auto cached_iterator = m_iterators.find(idx-1);
        if (cached_iterator != m_iterators.end())
        {
            current = cached_iterator->second;
            actual_advance = 1;
        }
    }
    
    ValueObjectSP current_sp(current.advance(actual_advance));
    if (!current_sp)
        return lldb::ValueObjectSP();
    
    m_iterators[idx] = current;
    
    current_sp = current_sp->GetChildAtIndex(1, true); // get the __value_ child
    if (!current_sp)
        return lldb::ValueObjectSP();
    // we need to copy current_sp into a new object otherwise we will end up with all items named __value_
    DataExtractor data;
    Error error;
    current_sp->GetData(data, error);
    if (error.Fail())
        return lldb::ValueObjectSP();
    
    StreamString name;
    name.Printf("[%" PRIu64 "]", (uint64_t)idx);
    return CreateValueObjectFromData(name.GetData(),
                                     data,
                                     m_backend.GetExecutionContextRef(),
                                     m_element_type);
}

bool
lldb_private::formatters::LibcxxStdListSyntheticFrontEnd::Update()
{
    m_iterators.clear();
    m_head = m_tail = nullptr;
    m_node_address = 0;
    m_count = UINT32_MAX;
    m_loop_detected = 0;
    m_slow_runner.SetEntry(nullptr);
    m_fast_runner.SetEntry(nullptr);

    Error err;
    ValueObjectSP backend_addr(m_backend.AddressOf(err));
    m_list_capping_size = 0;
    if (m_backend.GetTargetSP())
        m_list_capping_size = m_backend.GetTargetSP()->GetMaximumNumberOfChildrenToDisplay();
    if (m_list_capping_size == 0)
        m_list_capping_size = 255;
    if (err.Fail() || !backend_addr)
        return false;
    m_node_address = backend_addr->GetValueAsUnsigned(0);
    if (!m_node_address || m_node_address == LLDB_INVALID_ADDRESS)
        return false;
    ValueObjectSP impl_sp(m_backend.GetChildMemberWithName(ConstString("__end_"),true));
    if (!impl_sp)
        return false;
    CompilerType list_type = m_backend.GetCompilerType();
    if (list_type.IsReferenceType())
        list_type = list_type.GetNonReferenceType();

    if (list_type.GetNumTemplateArguments() == 0)
        return false;
    lldb::TemplateArgumentKind kind;
    m_element_type = list_type.GetTemplateArgument(0, kind);
    m_head = impl_sp->GetChildMemberWithName(ConstString("__next_"), true).get();
    m_tail = impl_sp->GetChildMemberWithName(ConstString("__prev_"), true).get();
    return false;
}

bool
lldb_private::formatters::LibcxxStdListSyntheticFrontEnd::MightHaveChildren ()
{
    return true;
}

size_t
lldb_private::formatters::LibcxxStdListSyntheticFrontEnd::GetIndexOfChildWithName (const ConstString &name)
{
    return ExtractIndexFromString(name.GetCString());
}

SyntheticChildrenFrontEnd*
lldb_private::formatters::LibcxxStdListSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP valobj_sp)
{
    return (valobj_sp ? new LibcxxStdListSyntheticFrontEnd(valobj_sp) : nullptr);
}
