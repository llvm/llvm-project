//===-- RecordingMemoryManager.cpp ------------------------------*- C++ -*-===//
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
#include "llvm/ExecutionEngine/ExecutionEngine.h"
// Project includes
#include "lldb/Expression/RecordingMemoryManager.h"

using namespace lldb_private;

RecordingMemoryManager::RecordingMemoryManager () :
    llvm::JITMemoryManager(),
    m_default_mm_ap (llvm::JITMemoryManager::CreateDefaultMemManager()),
    m_log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS))
{
}

RecordingMemoryManager::~RecordingMemoryManager ()
{
}

void
RecordingMemoryManager::setMemoryWritable ()
{
    m_default_mm_ap->setMemoryWritable();
}

void
RecordingMemoryManager::setMemoryExecutable ()
{
    m_default_mm_ap->setMemoryExecutable();
}


uint8_t *
RecordingMemoryManager::startFunctionBody(const llvm::Function *F,
                     uintptr_t &ActualSize)
{
    return m_default_mm_ap->startFunctionBody(F, ActualSize);
}

uint8_t *
RecordingMemoryManager::allocateStub(const llvm::GlobalValue* F, unsigned StubSize,
                unsigned Alignment)
{
    uint8_t *return_value = m_default_mm_ap->allocateStub(F, StubSize, Alignment);
    
    Allocation allocation;
    allocation.m_size = StubSize;
    allocation.m_alignment = Alignment;
    allocation.m_local_start = (uintptr_t)return_value;

    if (m_log)
    {
        m_log->Printf("RecordingMemoryManager::allocateStub (F=%p, StubSize=%u, Alignment=%u) = %p",
                      F, StubSize, Alignment, return_value);
        allocation.dump(m_log);
    }
    
    m_allocations.push_back(allocation);
    
    return return_value;
}

void
RecordingMemoryManager::endFunctionBody(const llvm::Function *F, uint8_t *FunctionStart,
               uint8_t *FunctionEnd)
{
    m_default_mm_ap->endFunctionBody(F, FunctionStart, FunctionEnd);
}

uint8_t *
RecordingMemoryManager::allocateSpace(intptr_t Size, unsigned Alignment)
{
    uint8_t *return_value = m_default_mm_ap->allocateSpace(Size, Alignment);
    
    Allocation allocation;
    allocation.m_size = Size;
    allocation.m_alignment = Alignment;
    allocation.m_local_start = (uintptr_t)return_value;
    
    if (m_log)
    {
        m_log->Printf("RecordingMemoryManager::allocateSpace(Size=%llu, Alignment=%u) = %p",
                      (uint64_t)Size, Alignment, return_value);
        allocation.dump(m_log);
    }
    
    m_allocations.push_back(allocation);
    
    return return_value;
}

uint8_t *
RecordingMemoryManager::allocateCodeSection(uintptr_t Size, unsigned Alignment, unsigned SectionID)
{
    uint8_t *return_value = m_default_mm_ap->allocateCodeSection(Size, Alignment, SectionID);
    
    Allocation allocation;
    allocation.m_size = Size;
    allocation.m_alignment = Alignment;
    allocation.m_local_start = (uintptr_t)return_value;
    allocation.m_section_id = SectionID;
    allocation.m_executable = true;
    
    if (m_log)
    {
        m_log->Printf("RecordingMemoryManager::allocateCodeSection(Size=0x%llx, Alignment=%u, SectionID=%u) = %p",
                      (uint64_t)Size, Alignment, SectionID, return_value);
        allocation.dump(m_log);
    }
    
    m_allocations.push_back(allocation);
    
    return return_value;
}

uint8_t *
RecordingMemoryManager::allocateDataSection(uintptr_t Size, unsigned Alignment, unsigned SectionID)
{
    uint8_t *return_value = m_default_mm_ap->allocateDataSection(Size, Alignment, SectionID);
    
    Allocation allocation;
    allocation.m_size = Size;
    allocation.m_alignment = Alignment;
    allocation.m_local_start = (uintptr_t)return_value;
    allocation.m_section_id = SectionID;
    
    if (m_log)
    {
        m_log->Printf("RecordingMemoryManager::allocateDataSection(Size=0x%llx, Alignment=%u, SectionID=%u) = %p",
                      (uint64_t)Size, Alignment, SectionID, return_value);
        allocation.dump(m_log);
    }
    
    m_allocations.push_back(allocation);
    
    return return_value; 
}

uint8_t *
RecordingMemoryManager::allocateGlobal(uintptr_t Size, unsigned Alignment)
{
    uint8_t *return_value = m_default_mm_ap->allocateGlobal(Size, Alignment);
    
    Allocation allocation;
    allocation.m_size = Size;
    allocation.m_alignment = Alignment;
    allocation.m_local_start = (uintptr_t)return_value;
    
    if (m_log)
    {
        m_log->Printf("RecordingMemoryManager::allocateGlobal(Size=0x%llx, Alignment=%u) = %p",
                      (uint64_t)Size, Alignment, return_value);
        allocation.dump(m_log);
    }
    
    m_allocations.push_back(allocation);

    return return_value;
}

void
RecordingMemoryManager::deallocateFunctionBody(void *Body)
{
    m_default_mm_ap->deallocateFunctionBody(Body);
}

uint8_t*
RecordingMemoryManager::startExceptionTable(const llvm::Function* F,
                       uintptr_t &ActualSize)
{
    return m_default_mm_ap->startExceptionTable(F, ActualSize);
}

void
RecordingMemoryManager::endExceptionTable(const llvm::Function *F, uint8_t *TableStart,
                 uint8_t *TableEnd, uint8_t* FrameRegister)
{
    m_default_mm_ap->endExceptionTable(F, TableStart, TableEnd, FrameRegister);
}

void
RecordingMemoryManager::deallocateExceptionTable(void *ET)
{
    m_default_mm_ap->deallocateExceptionTable (ET);
}

lldb::addr_t
RecordingMemoryManager::GetRemoteAddressForLocal (lldb::addr_t local_address)
{
    for (AllocationList::iterator ai = m_allocations.begin(), ae = m_allocations.end();
         ai != ae;
         ++ai)
    {
        if (local_address >= ai->m_local_start &&
            local_address < ai->m_local_start + ai->m_size)
            return ai->m_remote_start + (local_address - ai->m_local_start);
    }

    return LLDB_INVALID_ADDRESS;
}

RecordingMemoryManager::AddrRange
RecordingMemoryManager::GetRemoteRangeForLocal (lldb::addr_t local_address)
{
    for (AllocationList::iterator ai = m_allocations.begin(), ae = m_allocations.end();
         ai != ae;
         ++ai)
    {
        if (local_address >= ai->m_local_start &&
            local_address < ai->m_local_start + ai->m_size)
            return AddrRange(ai->m_remote_start, ai->m_size);
    }
    
    return AddrRange (0, 0);
}

bool
RecordingMemoryManager::CommitAllocations (Process &process)
{
    bool ret = true;
    
    for (AllocationList::iterator ai = m_allocations.begin(), ae = m_allocations.end();
         ai != ae;
         ++ai)
    {
        if (ai->m_allocated)
            continue;
        
        lldb_private::Error err;
        
        size_t allocation_size = (ai->m_size ? ai->m_size : 1) + ai->m_alignment - 1;
        
        if (allocation_size == 0)
            allocation_size = 1;
        
        ai->m_remote_allocation = process.AllocateMemory(
            allocation_size,
            ai->m_executable ? (lldb::ePermissionsReadable | lldb::ePermissionsExecutable) 
                             : (lldb::ePermissionsReadable | lldb::ePermissionsWritable), 
            err);
        
        uint64_t mask = ai->m_alignment - 1;
        
        ai->m_remote_start = (ai->m_remote_allocation + mask) & (~mask);
        
        if (!err.Success())
        {
            ret = false;
            break;
        }
        
        ai->m_allocated = true;
        
        if (m_log)
        {
            m_log->Printf("RecordingMemoryManager::CommitAllocations() committed an allocation");
            ai->dump(m_log);
        }
    }
    
    if (!ret)
    {
        for (AllocationList::iterator ai = m_allocations.end(), ae = m_allocations.end();
             ai != ae;
             ++ai)
        {
            if (ai->m_allocated)
                process.DeallocateMemory(ai->m_remote_start);
        }
    }
    
    return ret;
}

void
RecordingMemoryManager::ReportAllocations (llvm::ExecutionEngine &engine)
{
    for (AllocationList::iterator ai = m_allocations.begin(), ae = m_allocations.end();
         ai != ae;
         ++ai)
    {
        if (!ai->m_allocated)
            continue;
        
        engine.mapSectionAddress((void*)ai->m_local_start, ai->m_remote_start);
    }
}

bool
RecordingMemoryManager::WriteData (Process &process)
{    
    for (AllocationList::iterator ai = m_allocations.begin(), ae = m_allocations.end();
         ai != ae;
         ++ai)
    {
        if (!ai->m_allocated)
            return false;
        
        lldb_private::Error err;
        
        if (process.WriteMemory(ai->m_remote_start, 
                                (void*)ai->m_local_start, 
                                ai->m_size, 
                                err) != ai->m_size ||
            !err.Success())
            return false;
        
        if (m_log)
        {
            m_log->Printf("RecordingMemoryManager::CommitAllocations() wrote an allocation");
            ai->dump(m_log);
        }
    }
    
    return true;
}

void 
RecordingMemoryManager::Allocation::dump (lldb::LogSP log)
{
    if (!log)
        return;
    
    log->Printf("[0x%llx+0x%llx]->0x%llx (alignment %d, section ID %d)",
                (unsigned long long)m_local_start,
                (unsigned long long)m_size,
                (unsigned long long)m_remote_start,
                (unsigned)m_alignment,
                (unsigned)m_section_id);
}
