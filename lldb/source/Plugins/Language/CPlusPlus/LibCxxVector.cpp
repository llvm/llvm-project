//===-- LibCxxVector.cpp ----------------------------------------*- C++ -*-===//
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

#include "lldb/Core/ConstString.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/DataFormatters/FormattersHelpers.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

namespace lldb_private {
    namespace formatters {
        class LibcxxStdVectorSyntheticFrontEnd : public SyntheticChildrenFrontEnd
        {
        public:
            LibcxxStdVectorSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp);

            ~LibcxxStdVectorSyntheticFrontEnd() override;

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
            ValueObject* m_start;
            ValueObject* m_finish;
            CompilerType m_element_type;
            uint32_t m_element_size;
        };
    } // namespace formatters
} // namespace lldb_private

lldb_private::formatters::LibcxxStdVectorSyntheticFrontEnd::LibcxxStdVectorSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp) :
    SyntheticChildrenFrontEnd(*valobj_sp),
    m_start(nullptr),
    m_finish(nullptr),
    m_element_type(),
    m_element_size(0)
{
    if (valobj_sp)
        Update();
}

lldb_private::formatters::LibcxxStdVectorSyntheticFrontEnd::~LibcxxStdVectorSyntheticFrontEnd()
{
    // these need to stay around because they are child objects who will follow their parent's life cycle
    // delete m_start;
    // delete m_finish;
}

size_t
lldb_private::formatters::LibcxxStdVectorSyntheticFrontEnd::CalculateNumChildren ()
{
    if (!m_start || !m_finish)
        return 0;
    uint64_t start_val = m_start->GetValueAsUnsigned(0);
    uint64_t finish_val = m_finish->GetValueAsUnsigned(0);
    
    if (start_val == 0 || finish_val == 0)
        return 0;
    
    if (start_val >= finish_val)
        return 0;
    
    size_t num_children = (finish_val - start_val);
    if (num_children % m_element_size)
        return 0;
    return num_children/m_element_size;
}

lldb::ValueObjectSP
lldb_private::formatters::LibcxxStdVectorSyntheticFrontEnd::GetChildAtIndex (size_t idx)
{
    if (!m_start || !m_finish)
        return lldb::ValueObjectSP();
    
    uint64_t offset = idx * m_element_size;
    offset = offset + m_start->GetValueAsUnsigned(0);
    StreamString name;
    name.Printf("[%" PRIu64 "]", (uint64_t)idx);
    return CreateValueObjectFromAddress(name.GetData(),
                                        offset,
                                        m_backend.GetExecutionContextRef(),
                                        m_element_type);
}

bool
lldb_private::formatters::LibcxxStdVectorSyntheticFrontEnd::Update()
{
    m_start = m_finish = nullptr;
    ValueObjectSP data_type_finder_sp(m_backend.GetChildMemberWithName(ConstString("__end_cap_"),true));
    if (!data_type_finder_sp)
        return false;
    data_type_finder_sp = data_type_finder_sp->GetChildMemberWithName(ConstString("__first_"),true);
    if (!data_type_finder_sp)
        return false;
    m_element_type = data_type_finder_sp->GetCompilerType().GetPointeeType();
    m_element_size = m_element_type.GetByteSize(nullptr);
    
    if (m_element_size > 0)
    {
        // store raw pointers or end up with a circular dependency
        m_start = m_backend.GetChildMemberWithName(ConstString("__begin_"),true).get();
        m_finish = m_backend.GetChildMemberWithName(ConstString("__end_"),true).get();
    }
    return false;
}

bool
lldb_private::formatters::LibcxxStdVectorSyntheticFrontEnd::MightHaveChildren ()
{
    return true;
}

size_t
lldb_private::formatters::LibcxxStdVectorSyntheticFrontEnd::GetIndexOfChildWithName (const ConstString &name)
{
    if (!m_start || !m_finish)
        return UINT32_MAX;
    return ExtractIndexFromString(name.GetCString());
}

lldb_private::SyntheticChildrenFrontEnd*
lldb_private::formatters::LibcxxStdVectorSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP valobj_sp)
{
    return (valobj_sp ? new LibcxxStdVectorSyntheticFrontEnd(valobj_sp) : nullptr);
}
