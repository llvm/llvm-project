//===-- Materializer.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Log.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Core/ValueObjectVariable.h"
#include "lldb/Expression/ClangExpressionVariable.h"
#include "lldb/Expression/Materializer.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/Variable.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"

using namespace lldb_private;

uint32_t
Materializer::AddStructMember (Entity &entity)
{
    uint32_t size = entity.GetSize();
    uint32_t alignment = entity.GetAlignment();
    
    uint32_t ret;
    
    if (m_current_offset == 0)
        m_struct_alignment = alignment;
    
    if (m_current_offset % alignment)
        m_current_offset += (alignment - (m_current_offset % alignment));
    
    ret = m_current_offset;
    
    m_current_offset += size;
    
    return ret;
}

void
Materializer::Entity::SetSizeAndAlignmentFromType (ClangASTType &type)
{
    m_size = type.GetTypeByteSize();
    
    uint32_t bit_alignment = type.GetTypeBitAlign();
    
    if (bit_alignment % 8)
    {
        bit_alignment += 8;
        bit_alignment &= ~((uint32_t)0x111u);
    }
    
    m_alignment = bit_alignment / 8;
}

class EntityPersistentVariable : public Materializer::Entity
{
public:
    EntityPersistentVariable (lldb::ClangExpressionVariableSP &persistent_variable_sp) :
        Entity(),
        m_persistent_variable_sp(persistent_variable_sp)
    {
        // Hard-coding to maximum size of a pointer since persistent variables are materialized by reference
        m_size = 8;
        m_alignment = 8;
    }
    
    void MakeAllocation (IRMemoryMap &map, Error &err)
    {
        Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

        // Allocate a spare memory area to store the persistent variable's contents.
        
        Error allocate_error;
        
        lldb::addr_t mem = map.Malloc(m_persistent_variable_sp->GetByteSize(),
                                      8,
                                      lldb::ePermissionsReadable | lldb::ePermissionsWritable,
                                      IRMemoryMap::eAllocationPolicyMirror,
                                      allocate_error);
        
        if (!allocate_error.Success())
        {
            err.SetErrorToGenericError();
            err.SetErrorStringWithFormat("Couldn't allocate a memory area to store %s: %s", m_persistent_variable_sp->GetName().GetCString(), allocate_error.AsCString());
            return;
        }
        
        if (log)
            log->Printf("Allocated %s (0x%" PRIx64 ") sucessfully", m_persistent_variable_sp->GetName().GetCString(), mem);
        
        // Put the location of the spare memory into the live data of the ValueObject.
                
        m_persistent_variable_sp->m_live_sp = ValueObjectConstResult::Create (map.GetBestExecutionContextScope(),
                                                                              m_persistent_variable_sp->GetTypeFromUser().GetASTContext(),
                                                                              m_persistent_variable_sp->GetTypeFromUser().GetOpaqueQualType(),
                                                                              m_persistent_variable_sp->GetName(),
                                                                              mem,
                                                                              eAddressTypeLoad,
                                                                              m_persistent_variable_sp->GetByteSize());
        
        // Clear the flag if the variable will never be deallocated.
        
        if (m_persistent_variable_sp->m_flags & ClangExpressionVariable::EVKeepInTarget)
            m_persistent_variable_sp->m_flags &= ~ClangExpressionVariable::EVNeedsAllocation;
        
        // Write the contents of the variable to the area.
        
        Error write_error;
        
        map.WriteMemory (mem,
                         m_persistent_variable_sp->GetValueBytes(),
                         m_persistent_variable_sp->GetByteSize(),
                         write_error);
        
        if (!write_error.Success())
        {
            err.SetErrorToGenericError();
            err.SetErrorStringWithFormat ("Couldn't write %s to the target: %s", m_persistent_variable_sp->GetName().AsCString(),
                                          write_error.AsCString());
            return;
        }
    }
    
    void DestroyAllocation (IRMemoryMap &map, Error &err)
    {
        Error deallocate_error;
        
        map.Free((lldb::addr_t)m_persistent_variable_sp->m_live_sp->GetValue().GetScalar().ULongLong(), deallocate_error);
        
        m_persistent_variable_sp->m_live_sp.reset();
            
        if (!deallocate_error.Success())
        {
            err.SetErrorToGenericError();
            err.SetErrorStringWithFormat ("Couldn't deallocate memory for %s: %s", m_persistent_variable_sp->GetName().GetCString(), deallocate_error.AsCString());
        }
    }
    
    void Materialize (lldb::StackFrameSP &frame_sp, IRMemoryMap &map, lldb::addr_t process_address, Error &err)
    {
        Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

        if (log)
        {
            log->Printf("EntityPersistentVariable::Materialize [process_address = 0x%llx, m_name = %s, m_flags = 0x%hx]",
                        (uint64_t)process_address,
                        m_persistent_variable_sp->GetName().AsCString(),
                        m_persistent_variable_sp->m_flags);
        }
        
        if (m_persistent_variable_sp->m_flags & ClangExpressionVariable::EVNeedsAllocation)
        {
            MakeAllocation(map, err);
            m_persistent_variable_sp->m_flags |= ClangExpressionVariable::EVIsLLDBAllocated;
            
            if (!err.Success())
                return;
        }
        
        if ((m_persistent_variable_sp->m_flags & ClangExpressionVariable::EVIsProgramReference && m_persistent_variable_sp->m_live_sp) ||
            m_persistent_variable_sp->m_flags & ClangExpressionVariable::EVIsLLDBAllocated)
        {
            Error write_error;
                        
            map.WriteScalarToMemory(process_address + m_offset,
                                    m_persistent_variable_sp->m_live_sp->GetValue().GetScalar(),
                                    map.GetAddressByteSize(),
                                    write_error);
            
            if (!write_error.Success())
            {
                err.SetErrorToGenericError();
                err.SetErrorStringWithFormat("Couldn't write the location of %s to memory: %s", m_persistent_variable_sp->GetName().AsCString(), write_error.AsCString());
            }
        }
        else
        {
            err.SetErrorToGenericError();
            err.SetErrorStringWithFormat("No materialization happened for persistent variable %s", m_persistent_variable_sp->GetName().AsCString());
            return;
        }
    }
    
    void Dematerialize (lldb::StackFrameSP &frame_sp, IRMemoryMap &map, lldb::addr_t process_address,
                                lldb::addr_t frame_top, lldb::addr_t frame_bottom, Error &err)
    {
        Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
        
        if (log)
        {
            log->Printf("EntityPersistentVariable::Dematerialize [process_address = 0x%llx, m_name = %s, m_flags = 0x%hx]",
                        (uint64_t)process_address,
                        m_persistent_variable_sp->GetName().AsCString(),
                        m_persistent_variable_sp->m_flags);
        }
        
        if ((m_persistent_variable_sp->m_flags & ClangExpressionVariable::EVIsLLDBAllocated) ||
            (m_persistent_variable_sp->m_flags & ClangExpressionVariable::EVIsProgramReference))
        {
            if (m_persistent_variable_sp->m_flags & ClangExpressionVariable::EVIsProgramReference &&
                !m_persistent_variable_sp->m_live_sp)
            {
                // If the reference comes from the program, then the ClangExpressionVariable's
                // live variable data hasn't been set up yet.  Do this now.
                
                lldb::addr_t location;
                Error read_error;
                
                map.ReadPointerFromMemory(&location, process_address + m_offset, read_error);
                                
                if (!read_error.Success())
                {
                    err.SetErrorToGenericError();
                    err.SetErrorStringWithFormat("Couldn't read the address of program-allocated variable %s: %s", m_persistent_variable_sp->GetName().GetCString(), read_error.AsCString());
                    return;
                }
                                
                m_persistent_variable_sp->m_live_sp = ValueObjectConstResult::Create (map.GetBestExecutionContextScope (),
                                                                                      m_persistent_variable_sp->GetTypeFromUser().GetASTContext(),
                                                                                      m_persistent_variable_sp->GetTypeFromUser().GetOpaqueQualType(),
                                                                                      m_persistent_variable_sp->GetName(),
                                                                                      location,
                                                                                      eAddressTypeLoad,
                                                                                      m_persistent_variable_sp->GetByteSize());
                
                if (frame_top != LLDB_INVALID_ADDRESS &&
                    frame_bottom != LLDB_INVALID_ADDRESS &&
                    location >= frame_bottom &&
                    location <= frame_top)
                {
                    // If the variable is resident in the stack frame created by the expression,
                    // then it cannot be relied upon to stay around.  We treat it as needing
                    // reallocation.
                    m_persistent_variable_sp->m_flags |= ClangExpressionVariable::EVIsLLDBAllocated;
                    m_persistent_variable_sp->m_flags |= ClangExpressionVariable::EVNeedsAllocation;
                    m_persistent_variable_sp->m_flags |= ClangExpressionVariable::EVNeedsFreezeDry;
                    m_persistent_variable_sp->m_flags &= ~ClangExpressionVariable::EVIsProgramReference;
                }
            }
            
            lldb::addr_t mem = m_persistent_variable_sp->m_live_sp->GetValue().GetScalar().ULongLong();
            
            if (!m_persistent_variable_sp->m_live_sp)
            {
                err.SetErrorToGenericError();
                err.SetErrorStringWithFormat("Couldn't find the memory area used to store %s", m_persistent_variable_sp->GetName().GetCString());
                return;
            }
            
            if (m_persistent_variable_sp->m_live_sp->GetValue().GetValueAddressType() != eAddressTypeLoad)
            {
                err.SetErrorToGenericError();
                err.SetErrorStringWithFormat("The address of the memory area for %s is in an incorrect format", m_persistent_variable_sp->GetName().GetCString());
                return;
            }
            
            if (m_persistent_variable_sp->m_flags & ClangExpressionVariable::EVNeedsFreezeDry ||
                m_persistent_variable_sp->m_flags & ClangExpressionVariable::EVKeepInTarget)
            {                
                if (log)
                    log->Printf("Dematerializing %s from 0x%" PRIx64 " (size = %llu)", m_persistent_variable_sp->GetName().GetCString(), (uint64_t)mem, (unsigned long long)m_persistent_variable_sp->GetByteSize());
                
                // Read the contents of the spare memory area
                
                m_persistent_variable_sp->ValueUpdated ();
                
                Error read_error;
                
                map.ReadMemory(m_persistent_variable_sp->GetValueBytes(),
                               mem,
                               m_persistent_variable_sp->GetByteSize(),
                               read_error);
                
                if (!read_error.Success())
                {
                    err.SetErrorToGenericError();
                    err.SetErrorStringWithFormat ("Couldn't read the contents of %s from memory: %s", m_persistent_variable_sp->GetName().GetCString(), read_error.AsCString());
                    return;
                }
                    
                m_persistent_variable_sp->m_flags &= ~ClangExpressionVariable::EVNeedsFreezeDry;
            }
        }
        else
        {
            err.SetErrorToGenericError();
            err.SetErrorStringWithFormat("No dematerialization happened for persistent variable %s", m_persistent_variable_sp->GetName().AsCString());
            return;
        }
        
        lldb::ProcessSP process_sp = map.GetBestExecutionContextScope()->CalculateProcess();
        if (!process_sp ||
            !process_sp->CanJIT())
        {
            // Allocations are not persistent so persistent variables cannot stay materialized.
            
            m_persistent_variable_sp->m_flags |= ClangExpressionVariable::EVNeedsAllocation;

            DestroyAllocation(map, err);
            if (!err.Success())
                return;
        }
        else if (m_persistent_variable_sp->m_flags & ClangExpressionVariable::EVNeedsAllocation &&
                 !(m_persistent_variable_sp->m_flags & ClangExpressionVariable::EVKeepInTarget))
        {
            DestroyAllocation(map, err);
            if (!err.Success())
                return;
        }
    }
    
    void DumpToLog (IRMemoryMap &map, lldb::addr_t process_address, Log *log)
    {
        StreamString dump_stream;
        
        Error err;
        
        dump_stream.Printf("0x%llx: EntityPersistentVariable (%s)\n", (unsigned long long)process_address + m_offset, m_persistent_variable_sp->GetName().AsCString());
        
        {
            dump_stream.Printf("Pointer:\n");
            
            DataBufferHeap data (m_size, 0);
            
            map.ReadMemory(data.GetBytes(), process_address + m_offset, m_size, err);
            
            if (!err.Success())
            {
                dump_stream.Printf("  <could not be read>\n");
            }
            else
            {
                DataExtractor extractor (data.GetBytes(), data.GetByteSize(), map.GetByteOrder(), map.GetAddressByteSize());
                
                extractor.DumpHexBytes(&dump_stream, data.GetBytes(), data.GetByteSize(), 16, process_address + m_offset);
                
                dump_stream.PutChar('\n');
            }
        }
        
        {
            dump_stream.Printf("Target:\n");
            
            lldb::addr_t target_address;
            
            map.ReadPointerFromMemory (&target_address, process_address + m_offset, err);
                        
            if (!err.Success())
            {
                dump_stream.Printf("  <could not be read>\n");
            }
            else
            {
                DataBufferHeap data (m_persistent_variable_sp->GetByteSize(), 0);
            
                map.ReadMemory(data.GetBytes(), target_address, m_persistent_variable_sp->GetByteSize(), err);
                
                if (!err.Success())
                {
                    dump_stream.Printf("  <could not be read>\n");
                }
                else
                {
                    DataExtractor extractor (data.GetBytes(), data.GetByteSize(), map.GetByteOrder(), map.GetAddressByteSize());
                    
                    extractor.DumpHexBytes(&dump_stream, data.GetBytes(), data.GetByteSize(), 16, target_address);
                    
                    dump_stream.PutChar('\n');
                }
            }
        }
        
        log->PutCString(dump_stream.GetData());
    }
    
    void Wipe (IRMemoryMap &map, lldb::addr_t process_address)
    {
    }
private:
    lldb::ClangExpressionVariableSP m_persistent_variable_sp;
};

uint32_t
Materializer::AddPersistentVariable (lldb::ClangExpressionVariableSP &persistent_variable_sp, Error &err)
{
    EntityVector::iterator iter = m_entities.insert(m_entities.end(), EntityUP());
    iter->reset (new EntityPersistentVariable (persistent_variable_sp));
    uint32_t ret = AddStructMember(**iter);
    (*iter)->SetOffset(ret);
    return ret;
}

class EntityVariable : public Materializer::Entity
{
public:
    EntityVariable (lldb::VariableSP &variable_sp) :
        Entity(),
        m_variable_sp(variable_sp),
        m_is_reference(false),
        m_temporary_allocation(LLDB_INVALID_ADDRESS),
        m_temporary_allocation_size(0)
    {
        // Hard-coding to maximum size of a pointer since all variables are materialized by reference
        m_size = 8;
        m_alignment = 8;
        m_is_reference = ClangASTContext::IsReferenceType(m_variable_sp->GetType()->GetClangForwardType());
    }
    
    void Materialize (lldb::StackFrameSP &frame_sp, IRMemoryMap &map, lldb::addr_t process_address, Error &err)
    {
        Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
        
        if (log)
        {
            log->Printf("EntityVariable::Materialize [process_address = 0x%llx, m_variable_sp = %s]",
                        (uint64_t)process_address,
                        m_variable_sp->GetName().AsCString());
        }
        
        ExecutionContextScope *scope = frame_sp.get();
        
        if (!scope)
            scope = map.GetBestExecutionContextScope();
        
        lldb::ValueObjectSP valobj_sp = ValueObjectVariable::Create(scope, m_variable_sp);
        
        if (!valobj_sp)
        {
            err.SetErrorToGenericError();
            err.SetErrorStringWithFormat("Couldn't get a value object for variable %s", m_variable_sp->GetName().AsCString());
            return;
        }
        
        if (m_is_reference)
        {
            DataExtractor valobj_extractor;
            valobj_sp->GetData(valobj_extractor);
            lldb::offset_t offset = 0;
            lldb::addr_t reference_addr = valobj_extractor.GetAddress(&offset);
            
            Error write_error;
            map.WritePointerToMemory(process_address + m_offset, reference_addr, write_error);
            
            if (!write_error.Success())
            {
                err.SetErrorToGenericError();
                err.SetErrorStringWithFormat("Couldn't write the contents of reference variable %s to memory: %s", m_variable_sp->GetName().AsCString(), write_error.AsCString());
                return;
            }
        }
        else
        {
            Error get_address_error;
            lldb::ValueObjectSP addr_of_valobj_sp = valobj_sp->AddressOf(get_address_error);
            if (get_address_error.Success())
            {
                DataExtractor valobj_extractor;
                addr_of_valobj_sp->GetData(valobj_extractor);
                lldb::offset_t offset = 0;
                lldb::addr_t addr_of_valobj_addr = valobj_extractor.GetAddress(&offset);
                
                Error write_error;
                map.WritePointerToMemory(process_address + m_offset, addr_of_valobj_addr, write_error);
                
                if (!write_error.Success())
                {
                    err.SetErrorToGenericError();
                    err.SetErrorStringWithFormat("Couldn't write the address of variable %s to memory: %s", m_variable_sp->GetName().AsCString(), write_error.AsCString());
                    return;
                }
            }
            else
            {
                DataExtractor data;
                valobj_sp->GetData(data);
                
                if (m_temporary_allocation != LLDB_INVALID_ADDRESS)
                {
                    err.SetErrorToGenericError();
                    err.SetErrorStringWithFormat("Trying to create a temporary region for %s but one exists", m_variable_sp->GetName().AsCString());
                    return;
                }
                
                if (data.GetByteSize() != m_variable_sp->GetType()->GetByteSize())
                {
                    err.SetErrorToGenericError();
                    err.SetErrorStringWithFormat("Size of variable %s disagrees with the ValueObject's size", m_variable_sp->GetName().AsCString());
                    return;
                }
                
                size_t bit_align = ClangASTType::GetTypeBitAlign(m_variable_sp->GetType()->GetClangAST(), m_variable_sp->GetType()->GetClangLayoutType());
                size_t byte_align = (bit_align + 7) / 8;
                
                Error alloc_error;
                
                m_temporary_allocation = map.Malloc(data.GetByteSize(), byte_align, lldb::ePermissionsReadable | lldb::ePermissionsWritable, IRMemoryMap::eAllocationPolicyMirror, alloc_error);
                m_temporary_allocation_size = data.GetByteSize();
                
                if (!alloc_error.Success())
                {
                    err.SetErrorToGenericError();
                    err.SetErrorStringWithFormat("Couldn't allocate a temporary region for %s: %s", m_variable_sp->GetName().AsCString(), alloc_error.AsCString());
                    return;
                }
                
                Error write_error;
                
                map.WriteMemory(m_temporary_allocation, data.GetDataStart(), data.GetByteSize(), write_error);
                
                if (!write_error.Success())
                {
                    err.SetErrorToGenericError();
                    err.SetErrorStringWithFormat("Couldn't write to the temporary region for %s: %s", m_variable_sp->GetName().AsCString(), write_error.AsCString());
                    return;
                }
                
                Error pointer_write_error;
                
                map.WritePointerToMemory(process_address + m_offset, m_temporary_allocation, pointer_write_error);
                
                if (!pointer_write_error.Success())
                {
                    err.SetErrorToGenericError();
                    err.SetErrorStringWithFormat("Couldn't write the address of the temporary region for %s: %s", m_variable_sp->GetName().AsCString(), pointer_write_error.AsCString());
                }
            }
        }
    }
    
    void Dematerialize (lldb::StackFrameSP &frame_sp, IRMemoryMap &map, lldb::addr_t process_address,
                        lldb::addr_t frame_top, lldb::addr_t frame_bottom, Error &err)
    {
        Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

        if (log)
        {
            log->Printf("EntityVariable::Dematerialize [process_address = 0x%llx, m_variable_sp = %s]",
                        (uint64_t)process_address,
                        m_variable_sp->GetName().AsCString());
        }
        
        if (m_temporary_allocation != LLDB_INVALID_ADDRESS)
        {
            ExecutionContextScope *scope = frame_sp.get();
            
            if (!scope)
                scope = map.GetBestExecutionContextScope();
            
            lldb::ValueObjectSP valobj_sp = ValueObjectVariable::Create(scope, m_variable_sp);
            
            if (!valobj_sp)
            {
                err.SetErrorToGenericError();
                err.SetErrorStringWithFormat("Couldn't get a value object for variable %s", m_variable_sp->GetName().AsCString());
                return;
            }
            
            lldb_private::DataExtractor data;
            
            Error extract_error;
            
            map.GetMemoryData(data, m_temporary_allocation, valobj_sp->GetByteSize(), extract_error);
            
            if (!extract_error.Success())
            {
                err.SetErrorToGenericError();
                err.SetErrorStringWithFormat("Couldn't get the data for variable %s", m_variable_sp->GetName().AsCString());
                return;
            }
            
            Error set_error;
            
            valobj_sp->SetData(data, set_error);
            
            if (!set_error.Success())
            {
                err.SetErrorToGenericError();
                err.SetErrorStringWithFormat("Couldn't write the new contents of %s back into the variable", m_variable_sp->GetName().AsCString());
                return;
            }
            
            Error free_error;
            
            map.Free(m_temporary_allocation, free_error);
            
            if (!free_error.Success())
            {
                err.SetErrorToGenericError();
                err.SetErrorStringWithFormat("Couldn't free the temporary region for %s: %s", m_variable_sp->GetName().AsCString(), free_error.AsCString());
                return;
            }
            
            m_temporary_allocation = LLDB_INVALID_ADDRESS;
            m_temporary_allocation_size = 0;
        }
    }
    
    void DumpToLog (IRMemoryMap &map, lldb::addr_t process_address, Log *log)
    {
        StreamString dump_stream;
        
        dump_stream.Printf("0x%llx: EntityVariable\n", (unsigned long long)process_address + m_offset);
        
        Error err;
                
        lldb::addr_t ptr = LLDB_INVALID_ADDRESS;
        
        {
            dump_stream.Printf("Pointer:\n");
            
            DataBufferHeap data (m_size, 0);
            
            map.ReadMemory(data.GetBytes(), process_address + m_offset, m_size, err);
            
            if (!err.Success())
            {
                dump_stream.Printf("  <could not be read>\n");
            }
            else
            {
                DataExtractor extractor (data.GetBytes(), data.GetByteSize(), map.GetByteOrder(), map.GetAddressByteSize());
                
                extractor.DumpHexBytes(&dump_stream, data.GetBytes(), data.GetByteSize(), 16, process_address + m_offset);
                
                lldb::offset_t offset;
                
                ptr = extractor.GetPointer(&offset);
                
                dump_stream.PutChar('\n');
            }
        }
        
        if (m_temporary_allocation == LLDB_INVALID_ADDRESS)
        {
            dump_stream.Printf("Points to process memory:\n");
        }
        else
        {
            dump_stream.Printf("Temporary allocation:\n");
        }
        
        if (ptr == LLDB_INVALID_ADDRESS)
        {
            dump_stream.Printf("  <could not be be found>\n");
        }
        else
        {
            DataBufferHeap data (m_temporary_allocation_size, 0);
            
            map.ReadMemory(data.GetBytes(), m_temporary_allocation, m_temporary_allocation_size, err);
            
            if (!err.Success())
            {
                dump_stream.Printf("  <could not be read>\n");
            }
            else
            {
                DataExtractor extractor (data.GetBytes(), data.GetByteSize(), map.GetByteOrder(), map.GetAddressByteSize());
                
                extractor.DumpHexBytes(&dump_stream, data.GetBytes(), data.GetByteSize(), 16, process_address + m_offset);
                
                dump_stream.PutChar('\n');
            }
        }
        
        log->PutCString(dump_stream.GetData());
    }
    
    void Wipe (IRMemoryMap &map, lldb::addr_t process_address)
    {
        if (m_temporary_allocation != LLDB_INVALID_ADDRESS)
        {
            Error free_error;
            
            map.Free(m_temporary_allocation, free_error);
            
            m_temporary_allocation = LLDB_INVALID_ADDRESS;
            m_temporary_allocation_size = 0;
        }

    }
private:
    lldb::VariableSP    m_variable_sp;
    bool                m_is_reference;
    lldb::addr_t        m_temporary_allocation;
    size_t              m_temporary_allocation_size;
};

uint32_t
Materializer::AddVariable (lldb::VariableSP &variable_sp, Error &err)
{
    EntityVector::iterator iter = m_entities.insert(m_entities.end(), EntityUP());
    iter->reset (new EntityVariable (variable_sp));
    uint32_t ret = AddStructMember(**iter);
    (*iter)->SetOffset(ret);
    return ret;
}

class EntityResultVariable : public Materializer::Entity
{
public:
    EntityResultVariable (const TypeFromUser &type, bool is_program_reference, bool keep_in_memory) :
        Entity(),
        m_type(type),
        m_is_program_reference(is_program_reference),
        m_keep_in_memory(keep_in_memory),
        m_temporary_allocation(LLDB_INVALID_ADDRESS),
        m_temporary_allocation_size(0)
    {
        // Hard-coding to maximum size of a pointer since all results are materialized by reference
        m_size = 8;
        m_alignment = 8;
    }
    
    void Materialize (lldb::StackFrameSP &frame_sp, IRMemoryMap &map, lldb::addr_t process_address, Error &err)
    {
        if (!m_is_program_reference)
        {
            if (m_temporary_allocation != LLDB_INVALID_ADDRESS)
            {
                err.SetErrorToGenericError();
                err.SetErrorString("Trying to create a temporary region for the result but one exists");
                return;
            }
            
            size_t byte_size = m_type.GetTypeByteSize();
            size_t bit_align = m_type.GetTypeBitAlign();
            size_t byte_align = (bit_align + 7) / 8;
            
            Error alloc_error;
            
            m_temporary_allocation = map.Malloc(byte_size, byte_align, lldb::ePermissionsReadable | lldb::ePermissionsWritable, IRMemoryMap::eAllocationPolicyMirror, alloc_error);
            m_temporary_allocation_size = byte_size;
            
            if (!alloc_error.Success())
            {
                err.SetErrorToGenericError();
                err.SetErrorStringWithFormat("Couldn't allocate a temporary region for the result: %s", alloc_error.AsCString());
                return;
            }
                        
            Error pointer_write_error;
            
            map.WritePointerToMemory(process_address + m_offset, m_temporary_allocation, pointer_write_error);
            
            if (!pointer_write_error.Success())
            {
                err.SetErrorToGenericError();
                err.SetErrorStringWithFormat("Couldn't write the address of the temporary region for the result: %s", pointer_write_error.AsCString());
            }
        }
    }
    
    void Dematerialize (lldb::StackFrameSP &frame_sp, IRMemoryMap &map, lldb::addr_t process_address,
                        lldb::addr_t frame_top, lldb::addr_t frame_bottom, Error &err)
    {
        err.SetErrorToGenericError();
        err.SetErrorString("Tried to detmaterialize a result variable with the normal Dematerialize method");
    }
    
    void Dematerialize (lldb::ClangExpressionVariableSP &result_variable_sp,
                        lldb::StackFrameSP &frame_sp, IRMemoryMap &map, lldb::addr_t process_address,
                        lldb::addr_t frame_top, lldb::addr_t frame_bottom, Error &err)
    {
        err.Clear();
        
        ExecutionContextScope *exe_scope = map.GetBestExecutionContextScope();
        
        if (!exe_scope)
        {
            err.SetErrorToGenericError();
            err.SetErrorString("Couldn't dematerialize a result variable: invalid execution context scope");
            return;
        }
        
        lldb::addr_t address;
        Error read_error;
        
        map.ReadPointerFromMemory (&address, process_address + m_offset, read_error);
        
        if (!read_error.Success())
        {
            err.SetErrorToGenericError();
            err.SetErrorString("Couldn't dematerialize a result variable: couldn't read its address");
            return;
        }
        
        lldb::TargetSP target_sp = exe_scope->CalculateTarget();
        
        if (!target_sp)
        {
            err.SetErrorToGenericError();
            err.SetErrorString("Couldn't dematerialize a result variable: no target");
            return;
        }
        
        ConstString name = target_sp->GetPersistentVariables().GetNextPersistentVariableName();
        
        lldb::ClangExpressionVariableSP ret;
        
        ret = target_sp->GetPersistentVariables().CreateVariable(exe_scope,
                                                                 name,
                                                                 m_type,
                                                                 map.GetByteOrder(),
                                                                 map.GetAddressByteSize());
        
        if (!ret)
        {
            err.SetErrorToGenericError();
            err.SetErrorStringWithFormat("Couldn't dematerialize a result variable: failed to make persistent variable %s", name.AsCString());
            return;
        }
        
        lldb::ProcessSP process_sp = map.GetBestExecutionContextScope()->CalculateProcess();
        
        bool can_persist = (m_is_program_reference && process_sp && process_sp->CanJIT() && !(address >= frame_bottom && address < frame_top));

        if (can_persist && m_keep_in_memory)
        {
            ret->m_live_sp = ValueObjectConstResult::Create(exe_scope,
                                                            m_type.GetASTContext(),
                                                            m_type.GetOpaqueQualType(),
                                                            name,
                                                            address,
                                                            eAddressTypeLoad,
                                                            ret->GetByteSize());
        }
        
        ret->ValueUpdated();
        
        const size_t pvar_byte_size = ret->GetByteSize();
        uint8_t *pvar_data = ret->GetValueBytes();
                
        map.ReadMemory(pvar_data, address, pvar_byte_size, read_error);
        
        if (!read_error.Success())
        {
            err.SetErrorToGenericError();
            err.SetErrorString("Couldn't dematerialize a result variable: couldn't read its memory");
            return;
        }
                
        result_variable_sp = ret;
        
        if (!can_persist || !m_keep_in_memory)
        {
            ret->m_flags |= ClangExpressionVariable::EVNeedsAllocation;
            
            if (m_temporary_allocation != LLDB_INVALID_ADDRESS)
            {
                Error free_error;
                map.Free(m_temporary_allocation, free_error);
            }
        }
        else
        {
            ret->m_flags |= ClangExpressionVariable::EVIsLLDBAllocated;
        }
        
        m_temporary_allocation = LLDB_INVALID_ADDRESS;
        m_temporary_allocation_size = 0;
    }
    
    void DumpToLog (IRMemoryMap &map, lldb::addr_t process_address, Log *log)
    {
        StreamString dump_stream;
                
        dump_stream.Printf("0x%llx: EntityResultVariable\n", (unsigned long long)process_address + m_offset);
        
        Error err;
        
        lldb::addr_t ptr = LLDB_INVALID_ADDRESS;
        
        {
            dump_stream.Printf("Pointer:\n");
            
            DataBufferHeap data (m_size, 0);
            
            map.ReadMemory(data.GetBytes(), process_address + m_offset, m_size, err);
            
            if (!err.Success())
            {
                dump_stream.Printf("  <could not be read>\n");
            }
            else
            {
                DataExtractor extractor (data.GetBytes(), data.GetByteSize(), map.GetByteOrder(), map.GetAddressByteSize());
                
                extractor.DumpHexBytes(&dump_stream, data.GetBytes(), data.GetByteSize(), 16, process_address + m_offset);
                
                lldb::offset_t offset;
                
                ptr = extractor.GetPointer(&offset);
                
                dump_stream.PutChar('\n');
            }
        }
                
        if (m_temporary_allocation == LLDB_INVALID_ADDRESS)
        {
            dump_stream.Printf("Points to process memory:\n");
        }
        else
        {
            dump_stream.Printf("Temporary allocation:\n");
        }
        
        if (ptr == LLDB_INVALID_ADDRESS)
        {
            dump_stream.Printf("  <could not be be found>\n");
        }
        else
        {
            DataBufferHeap data (m_temporary_allocation_size, 0);
            
            map.ReadMemory(data.GetBytes(), m_temporary_allocation, m_temporary_allocation_size, err);
            
            if (!err.Success())
            {
                dump_stream.Printf("  <could not be read>\n");
            }
            else
            {
                DataExtractor extractor (data.GetBytes(), data.GetByteSize(), map.GetByteOrder(), map.GetAddressByteSize());
                
                extractor.DumpHexBytes(&dump_stream, data.GetBytes(), data.GetByteSize(), 16, process_address + m_offset);
                
                dump_stream.PutChar('\n');
            }
        }
        
        log->PutCString(dump_stream.GetData());
    }
    
    void Wipe (IRMemoryMap &map, lldb::addr_t process_address)
    {
        if (!m_keep_in_memory && m_temporary_allocation != LLDB_INVALID_ADDRESS)
        {
            Error free_error;
            
            map.Free(m_temporary_allocation, free_error);
        }
        
        m_temporary_allocation = LLDB_INVALID_ADDRESS;
        m_temporary_allocation_size = 0;
    }
private:
    TypeFromUser    m_type;
    bool            m_is_program_reference;
    bool            m_keep_in_memory;
    
    lldb::addr_t    m_temporary_allocation;
    size_t          m_temporary_allocation_size;
};

uint32_t
Materializer::AddResultVariable (const TypeFromUser &type, bool is_program_reference, bool keep_in_memory, Error &err)
{
    EntityVector::iterator iter = m_entities.insert(m_entities.end(), EntityUP());
    iter->reset (new EntityResultVariable (type, is_program_reference, keep_in_memory));
    uint32_t ret = AddStructMember(**iter);
    (*iter)->SetOffset(ret);
    m_result_entity = iter->get();
    return ret;
}

class EntitySymbol : public Materializer::Entity
{
public:
    EntitySymbol (const Symbol &symbol) :
        Entity(),
        m_symbol(symbol)
    {
        // Hard-coding to maximum size of a symbol
        m_size = 8;
        m_alignment = 8;
    }
    
    void Materialize (lldb::StackFrameSP &frame_sp, IRMemoryMap &map, lldb::addr_t process_address, Error &err)
    {
        Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

        if (log)
        {
            log->Printf("EntitySymbol::Materialize [process_address = 0x%llx, m_symbol = %s]",
                        (uint64_t)process_address,
                        m_symbol.GetName().AsCString());
        }
        
        Address &sym_address = m_symbol.GetAddress();

        ExecutionContextScope *exe_scope = map.GetBestExecutionContextScope();
        
        lldb::TargetSP target_sp;
        
        if (exe_scope)
            target_sp = map.GetBestExecutionContextScope()->CalculateTarget();
        
        if (!target_sp)
        {
            err.SetErrorToGenericError();
            err.SetErrorStringWithFormat("Couldn't resolve symbol %s because there is no target", m_symbol.GetName().AsCString());
            return;
        }
        
        lldb::addr_t resolved_address = sym_address.GetLoadAddress(target_sp.get());
        
        if (resolved_address == LLDB_INVALID_ADDRESS)
            resolved_address = sym_address.GetFileAddress();
        
        Error pointer_write_error;
        
        map.WritePointerToMemory(process_address + m_offset, resolved_address, pointer_write_error);
        
        if (!pointer_write_error.Success())
        {
            err.SetErrorToGenericError();
            err.SetErrorStringWithFormat("Couldn't write the address of symbol %s: %s", m_symbol.GetName().AsCString(), pointer_write_error.AsCString());
            return;
        }
    }
    
    void Dematerialize (lldb::StackFrameSP &frame_sp, IRMemoryMap &map, lldb::addr_t process_address,
                                lldb::addr_t frame_top, lldb::addr_t frame_bottom, Error &err)
    {
        Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

        if (log)
        {
            log->Printf("EntitySymbol::Dematerialize [process_address = 0x%llx, m_symbol = %s]",
                        (uint64_t)process_address,
                        m_symbol.GetName().AsCString());
        }
        
        // no work needs to be done
    }
    
    void DumpToLog (IRMemoryMap &map, lldb::addr_t process_address, Log *log)
    {
        StreamString dump_stream;
        
        Error err;
        
        dump_stream.Printf("0x%llx: EntitySymbol (%s)\n", (unsigned long long)process_address + m_offset, m_symbol.GetName().AsCString());
        
        {
            dump_stream.Printf("Pointer:\n");
            
            DataBufferHeap data (m_size, 0);
            
            map.ReadMemory(data.GetBytes(), process_address + m_offset, m_size, err);
            
            if (!err.Success())
            {
                dump_stream.Printf("  <could not be read>\n");
            }
            else
            {
                DataExtractor extractor (data.GetBytes(), data.GetByteSize(), map.GetByteOrder(), map.GetAddressByteSize());
                
                extractor.DumpHexBytes(&dump_stream, data.GetBytes(), data.GetByteSize(), 16, process_address + m_offset);
                
                dump_stream.PutChar('\n');
            }
        }
        
        log->PutCString(dump_stream.GetData());
    }
    
    void Wipe (IRMemoryMap &map, lldb::addr_t process_address)
    {
    }
private:
    Symbol m_symbol;
};

uint32_t
Materializer::AddSymbol (const Symbol &symbol_sp, Error &err)
{
    EntityVector::iterator iter = m_entities.insert(m_entities.end(), EntityUP());
    iter->reset (new EntitySymbol (symbol_sp));
    uint32_t ret = AddStructMember(**iter);
    (*iter)->SetOffset(ret);
    return ret;
}

class EntityRegister : public Materializer::Entity
{
public:
    EntityRegister (const RegisterInfo &register_info) :
        Entity(),
        m_register_info(register_info)
    {
        // Hard-coding alignment conservatively
        m_size = m_register_info.byte_size;
        m_alignment = m_register_info.byte_size;
    }
    
    void Materialize (lldb::StackFrameSP &frame_sp, IRMemoryMap &map, lldb::addr_t process_address, Error &err)
    {
        Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
        
        if (log)
        {
            log->Printf("EntityRegister::Materialize [process_address = 0x%llx, m_register_info = %s]",
                        (uint64_t)process_address,
                        m_register_info.name);
        }

        RegisterValue reg_value;
        
        if (!frame_sp.get())
        {
            err.SetErrorToGenericError();
            err.SetErrorStringWithFormat("Couldn't materialize register %s without a stack frame", m_register_info.name);
            return;
        }
        
        lldb::RegisterContextSP reg_context_sp = frame_sp->GetRegisterContext();
        
        if (!reg_context_sp->ReadRegister(&m_register_info, reg_value))
        {
            err.SetErrorToGenericError();
            err.SetErrorStringWithFormat("Couldn't read the value of register %s", m_register_info.name);
            return;
        }
        
        DataExtractor register_data;
        
        if (!reg_value.GetData(register_data))
        {
            err.SetErrorToGenericError();
            err.SetErrorStringWithFormat("Couldn't get the data for register %s", m_register_info.name);
            return;
        }
        
        if (register_data.GetByteSize() != m_register_info.byte_size)
        {
            err.SetErrorToGenericError();
            err.SetErrorStringWithFormat("Data for register %s had size %llu but we expected %llu", m_register_info.name, (unsigned long long)register_data.GetByteSize(), (unsigned long long)m_register_info.byte_size);
            return;
        }
        
        Error write_error;
        
        map.WriteMemory(process_address + m_offset, register_data.GetDataStart(), register_data.GetByteSize(), write_error);
        
        if (!write_error.Success())
        {
            err.SetErrorToGenericError();
            err.SetErrorStringWithFormat("Couldn't write the contents of register %s: %s", m_register_info.name, write_error.AsCString());
            return;
        }
    }
    
    void Dematerialize (lldb::StackFrameSP &frame_sp, IRMemoryMap &map, lldb::addr_t process_address,
                                lldb::addr_t frame_top, lldb::addr_t frame_bottom, Error &err)
    {
        Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
        
        if (log)
        {
            log->Printf("EntityRegister::Dematerialize [process_address = 0x%llx, m_register_info = %s]",
                        (uint64_t)process_address,
                        m_register_info.name);
        }
        
        Error extract_error;
        
        DataExtractor register_data;
        
        if (!frame_sp.get())
        {
            err.SetErrorToGenericError();
            err.SetErrorStringWithFormat("Couldn't dematerialize register %s without a stack frame", m_register_info.name);
            return;
        }
        
        lldb::RegisterContextSP reg_context_sp = frame_sp->GetRegisterContext();
        
        map.GetMemoryData(register_data, process_address + m_offset, m_register_info.byte_size, extract_error);
        
        if (!extract_error.Success())
        {
            err.SetErrorToGenericError();
            err.SetErrorStringWithFormat("Couldn't get the data for register %s: %s", m_register_info.name, extract_error.AsCString());
            return;
        }
        
        RegisterValue register_value (const_cast<uint8_t*>(register_data.GetDataStart()), register_data.GetByteSize(), register_data.GetByteOrder());
        
        if (!reg_context_sp->WriteRegister(&m_register_info, register_value))
        {
            err.SetErrorToGenericError();
            err.SetErrorStringWithFormat("Couldn't write the value of register %s", m_register_info.name);
            return;
        }
    }
    
    void DumpToLog (IRMemoryMap &map, lldb::addr_t process_address, Log *log)
    {
        StreamString dump_stream;
        
        Error err;
        
        dump_stream.Printf("0x%llx: EntityRegister (%s)\n", (unsigned long long)process_address + m_offset, m_register_info.name);
        
        {
            dump_stream.Printf("Value:\n");
            
            DataBufferHeap data (m_size, 0);
            
            map.ReadMemory(data.GetBytes(), process_address + m_offset, m_size, err);
            
            if (!err.Success())
            {
                dump_stream.Printf("  <could not be read>\n");
            }
            else
            {
                DataExtractor extractor (data.GetBytes(), data.GetByteSize(), map.GetByteOrder(), map.GetAddressByteSize());
                
                extractor.DumpHexBytes(&dump_stream, data.GetBytes(), data.GetByteSize(), 16, process_address + m_offset);
                
                dump_stream.PutChar('\n');
            }
        }
        
        log->PutCString(dump_stream.GetData());
    }
    
    void Wipe (IRMemoryMap &map, lldb::addr_t process_address)
    {
    }
private:
    RegisterInfo m_register_info;
};

uint32_t
Materializer::AddRegister (const RegisterInfo &register_info, Error &err)
{
    EntityVector::iterator iter = m_entities.insert(m_entities.end(), EntityUP());
    iter->reset (new EntityRegister (register_info));
    uint32_t ret = AddStructMember(**iter);
    (*iter)->SetOffset(ret);
    return ret;
}

Materializer::Materializer () :
    m_dematerializer_wp(),
    m_result_entity(NULL),
    m_current_offset(0),
    m_struct_alignment(8)
{
}

Materializer::~Materializer ()
{
    DematerializerSP dematerializer_sp = m_dematerializer_wp.lock();
    
    if (dematerializer_sp)
        dematerializer_sp->Wipe();
}

Materializer::DematerializerSP
Materializer::Materialize (lldb::StackFrameSP &frame_sp, IRMemoryMap &map, lldb::addr_t process_address, Error &error)
{
    ExecutionContextScope *exe_scope = frame_sp.get();
        
    if (!exe_scope)
        exe_scope = map.GetBestExecutionContextScope();
    
    DematerializerSP dematerializer_sp = m_dematerializer_wp.lock();
    
    if (dematerializer_sp)
    {
        error.SetErrorToGenericError();
        error.SetErrorString("Couldn't materialize: already materialized");
    }
    
    DematerializerSP ret(new Dematerializer(*this, frame_sp, map, process_address));
    
    if (!exe_scope)
    {
        error.SetErrorToGenericError();
        error.SetErrorString("Couldn't materialize: target doesn't exist");
    }
    
    for (EntityUP &entity_up : m_entities)
    {
        entity_up->Materialize(frame_sp, map, process_address, error);
        
        if (!error.Success())
            return DematerializerSP();
    }
    
    if (Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS))
    {
        log->Printf("Materializer::Materialize (frame_sp = %p, process_address = 0x%llx) materialized:", frame_sp.get(), process_address);
        for (EntityUP &entity_up : m_entities)
            entity_up->DumpToLog(map, process_address, log);
    }
    
    m_dematerializer_wp = ret;
    
    return ret;
}

void
Materializer::Dematerializer::Dematerialize (Error &error, lldb::ClangExpressionVariableSP &result_sp, lldb::addr_t frame_bottom, lldb::addr_t frame_top)
{
    lldb::StackFrameSP frame_sp = m_frame_wp.lock();
    
    ExecutionContextScope *exe_scope = m_map->GetBestExecutionContextScope();
    
    if (!IsValid())
    {
        error.SetErrorToGenericError();
        error.SetErrorString("Couldn't dematerialize: invalid dematerializer");
    }
    
    if (!exe_scope)
    {
        error.SetErrorToGenericError();
        error.SetErrorString("Couldn't dematerialize: target is gone");
    }
    else
    {
        if (Log *log =lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS))
        {
            log->Printf("Materializer::Dematerialize (frame_sp = %p, process_address = 0x%llx) about to dematerialize:", frame_sp.get(), m_process_address);
            for (EntityUP &entity_up : m_materializer->m_entities)
                entity_up->DumpToLog(*m_map, m_process_address, log);
        }
        
        for (EntityUP &entity_up : m_materializer->m_entities)
        {
            if (entity_up.get() == m_materializer->m_result_entity)
            {
                static_cast<EntityResultVariable*>(m_materializer->m_result_entity)->Dematerialize (result_sp, frame_sp, *m_map, m_process_address, frame_top, frame_bottom, error);
            }
            else
            {
                entity_up->Dematerialize (frame_sp, *m_map, m_process_address, frame_top, frame_bottom, error);
            }
                
            if (!error.Success())
                break;
        }
    }
    
    Wipe();
}

void
Materializer::Dematerializer::Wipe ()
{
    if (!IsValid())
        return;
    
    for (EntityUP &entity_up : m_materializer->m_entities)
    {
        entity_up->Wipe (*m_map, m_process_address);
    }

    m_materializer = NULL;
    m_map = NULL;
    m_process_address = LLDB_INVALID_ADDRESS;
}
