//===-- GDBRemoteRegisterContext.cpp ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "GDBRemoteRegisterContext.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/Utils.h"
// Project includes
#include "Utility/StringExtractorGDBRemote.h"
#include "ProcessGDBRemote.h"
#include "ProcessGDBRemoteLog.h"
#include "ThreadGDBRemote.h"
#include "Utility/ARM_DWARF_Registers.h"
#include "Utility/ARM_ehframe_Registers.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_gdb_remote;

//----------------------------------------------------------------------
// GDBRemoteRegisterContext constructor
//----------------------------------------------------------------------
GDBRemoteRegisterContext::GDBRemoteRegisterContext
(
    ThreadGDBRemote &thread,
    uint32_t concrete_frame_idx,
    GDBRemoteDynamicRegisterInfo &reg_info,
    bool read_all_at_once
) :
    RegisterContext (thread, concrete_frame_idx),
    m_reg_info (reg_info),
    m_reg_valid (),
    m_reg_data (),
    m_read_all_at_once (read_all_at_once)
{
    // Resize our vector of bools to contain one bool for every register.
    // We will use these boolean values to know when a register value
    // is valid in m_reg_data.
    m_reg_valid.resize (reg_info.GetNumRegisters());

    // Make a heap based buffer that is big enough to store all registers
    DataBufferSP reg_data_sp(new DataBufferHeap (reg_info.GetRegisterDataByteSize(), 0));
    m_reg_data.SetData (reg_data_sp);
    m_reg_data.SetByteOrder(thread.GetProcess()->GetByteOrder());
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
GDBRemoteRegisterContext::~GDBRemoteRegisterContext()
{
}

void
GDBRemoteRegisterContext::InvalidateAllRegisters ()
{
    SetAllRegisterValid (false);
}

void
GDBRemoteRegisterContext::SetAllRegisterValid (bool b)
{
    std::vector<bool>::iterator pos, end = m_reg_valid.end();
    for (pos = m_reg_valid.begin(); pos != end; ++pos)
        *pos = b;
}

size_t
GDBRemoteRegisterContext::GetRegisterCount ()
{
    return m_reg_info.GetNumRegisters ();
}

const RegisterInfo *
GDBRemoteRegisterContext::GetRegisterInfoAtIndex (size_t reg)
{
    return m_reg_info.GetRegisterInfoAtIndex (reg);
}

size_t
GDBRemoteRegisterContext::GetRegisterSetCount ()
{
    return m_reg_info.GetNumRegisterSets ();
}



const RegisterSet *
GDBRemoteRegisterContext::GetRegisterSet (size_t reg_set)
{
    return m_reg_info.GetRegisterSet (reg_set);
}



bool
GDBRemoteRegisterContext::ReadRegister (const RegisterInfo *reg_info, RegisterValue &value)
{
    // Read the register
    if (ReadRegisterBytes (reg_info, m_reg_data))
    {
        const bool partial_data_ok = false;
        Error error (value.SetValueFromData(reg_info, m_reg_data, reg_info->byte_offset, partial_data_ok));
        return error.Success();
    }
    return false;
}

bool
GDBRemoteRegisterContext::PrivateSetRegisterValue (uint32_t reg, StringExtractor &response)
{
    const RegisterInfo *reg_info = GetRegisterInfoAtIndex (reg);
    if (reg_info == NULL)
        return false;

    // Invalidate if needed
    InvalidateIfNeeded(false);

    const uint32_t reg_byte_size = reg_info->byte_size;
    const size_t bytes_copied = response.GetHexBytes (const_cast<uint8_t*>(m_reg_data.PeekData(reg_info->byte_offset, reg_byte_size)), reg_byte_size, '\xcc');
    bool success = bytes_copied == reg_byte_size;
    if (success)
    {
        SetRegisterIsValid(reg, true);
    }
    else if (bytes_copied > 0)
    {
        // Only set register is valid to false if we copied some bytes, else
        // leave it as it was.
        SetRegisterIsValid(reg, false);
    }
    return success;
}

bool
GDBRemoteRegisterContext::PrivateSetRegisterValue (uint32_t reg, uint64_t new_reg_val)
{
    const RegisterInfo *reg_info = GetRegisterInfoAtIndex (reg);
    if (reg_info == NULL)
        return false;

    // Early in process startup, we can get a thread that has an invalid byte order
    // because the process hasn't been completely set up yet (see the ctor where the
    // byte order is setfrom the process).  If that's the case, we can't set the
    // value here.
    if (m_reg_data.GetByteOrder() == eByteOrderInvalid)
    {
        return false;
    }

    // Invalidate if needed
    InvalidateIfNeeded (false);

    DataBufferSP buffer_sp (new DataBufferHeap (&new_reg_val, sizeof (new_reg_val)));
    DataExtractor data (buffer_sp, endian::InlHostByteOrder(), sizeof (void*));

    // If our register context and our register info disagree, which should never happen, don't
    // overwrite past the end of the buffer.
    if (m_reg_data.GetByteSize() < reg_info->byte_offset + reg_info->byte_size)
        return false;

    // Grab a pointer to where we are going to put this register
    uint8_t *dst = const_cast<uint8_t*>(m_reg_data.PeekData(reg_info->byte_offset, reg_info->byte_size));

    if (dst == NULL)
        return false;


    if (data.CopyByteOrderedData (0,                            // src offset
                                  reg_info->byte_size,          // src length
                                  dst,                          // dst
                                  reg_info->byte_size,          // dst length
                                  m_reg_data.GetByteOrder()))   // dst byte order
    {
        SetRegisterIsValid (reg, true);
        return true;
    }
    return false;
}

// Helper function for GDBRemoteRegisterContext::ReadRegisterBytes().
bool
GDBRemoteRegisterContext::GetPrimordialRegister(const RegisterInfo *reg_info,
                                                GDBRemoteCommunicationClient &gdb_comm)
{
    const uint32_t lldb_reg = reg_info->kinds[eRegisterKindLLDB];
    const uint32_t remote_reg = reg_info->kinds[eRegisterKindProcessPlugin];
    StringExtractorGDBRemote response;
    if (gdb_comm.ReadRegister(m_thread.GetProtocolID(), remote_reg, response))
        return PrivateSetRegisterValue (lldb_reg, response);
    return false;
}

bool
GDBRemoteRegisterContext::ReadRegisterBytes (const RegisterInfo *reg_info, DataExtractor &data)
{
    ExecutionContext exe_ctx (CalculateThread());

    Process *process = exe_ctx.GetProcessPtr();
    Thread *thread = exe_ctx.GetThreadPtr();
    if (process == NULL || thread == NULL)
        return false;

    GDBRemoteCommunicationClient &gdb_comm (((ProcessGDBRemote *)process)->GetGDBRemote());

    InvalidateIfNeeded(false);

    const uint32_t reg = reg_info->kinds[eRegisterKindLLDB];

    if (!GetRegisterIsValid(reg))
    {
        if (m_read_all_at_once)
        {
            StringExtractorGDBRemote response;
            if (!gdb_comm.ReadAllRegisters(m_thread.GetProtocolID(), response))
                return false;
            if (response.IsNormalResponse())
                if (response.GetHexBytes(const_cast<void *>(reinterpret_cast<const void *>(m_reg_data.GetDataStart())),
                                         m_reg_data.GetByteSize(), '\xcc') == m_reg_data.GetByteSize())
                    SetAllRegisterValid (true);
        }
        else if (reg_info->value_regs)
        {
            // Process this composite register request by delegating to the constituent
            // primordial registers.
            
            // Index of the primordial register.
            bool success = true;
            for (uint32_t idx = 0; success; ++idx)
            {
                const uint32_t prim_reg = reg_info->value_regs[idx];
                if (prim_reg == LLDB_INVALID_REGNUM)
                    break;
                // We have a valid primordial register as our constituent.
                // Grab the corresponding register info.
                const RegisterInfo *prim_reg_info = GetRegisterInfoAtIndex(prim_reg);
                if (prim_reg_info == NULL)
                    success = false;
                else
                {
                    // Read the containing register if it hasn't already been read
                    if (!GetRegisterIsValid(prim_reg))
                        success = GetPrimordialRegister(prim_reg_info, gdb_comm);
                }
            }

            if (success)
            {
                // If we reach this point, all primordial register requests have succeeded.
                // Validate this composite register.
                SetRegisterIsValid (reg_info, true);
            }
        }
        else
        {
            // Get each register individually
            GetPrimordialRegister(reg_info, gdb_comm);
        }

        // Make sure we got a valid register value after reading it
        if (!GetRegisterIsValid(reg))
            return false;
    }

    if (&data != &m_reg_data)
    {
#if defined (LLDB_CONFIGURATION_DEBUG)
        assert (m_reg_data.GetByteSize() >= reg_info->byte_offset + reg_info->byte_size);
#endif  
        // If our register context and our register info disagree, which should never happen, don't
        // read past the end of the buffer.
        if (m_reg_data.GetByteSize() < reg_info->byte_offset + reg_info->byte_size)
            return false;

        // If we aren't extracting into our own buffer (which
        // only happens when this function is called from
        // ReadRegisterValue(uint32_t, Scalar&)) then
        // we transfer bytes from our buffer into the data
        // buffer that was passed in

        data.SetByteOrder (m_reg_data.GetByteOrder());
        data.SetData (m_reg_data, reg_info->byte_offset, reg_info->byte_size);
    }
    return true;
}

bool
GDBRemoteRegisterContext::WriteRegister (const RegisterInfo *reg_info,
                                         const RegisterValue &value)
{
    DataExtractor data;
    if (value.GetData (data))
        return WriteRegisterBytes (reg_info, data, 0);
    return false;
}

// Helper function for GDBRemoteRegisterContext::WriteRegisterBytes().
bool
GDBRemoteRegisterContext::SetPrimordialRegister(const RegisterInfo *reg_info,
                                                GDBRemoteCommunicationClient &gdb_comm)
{
    StreamString packet;
    StringExtractorGDBRemote response;
    const uint32_t reg = reg_info->kinds[eRegisterKindLLDB];
    packet.Printf ("P%x=", reg_info->kinds[eRegisterKindProcessPlugin]);
    packet.PutBytesAsRawHex8 (m_reg_data.PeekData(reg_info->byte_offset, reg_info->byte_size),
                              reg_info->byte_size,
                              endian::InlHostByteOrder(),
                              endian::InlHostByteOrder());

    if (gdb_comm.GetThreadSuffixSupported())
        packet.Printf (";thread:%4.4" PRIx64 ";", m_thread.GetProtocolID());

    // Invalidate just this register
    SetRegisterIsValid(reg, false);
    if (gdb_comm.SendPacketAndWaitForResponse(packet.GetString().c_str(),
                                              packet.GetString().size(),
                                              response,
                                              false) == GDBRemoteCommunication::PacketResult::Success)
    {
        if (response.IsOKResponse())
            return true;
    }
    return false;
}

void
GDBRemoteRegisterContext::SyncThreadState(Process *process)
{
    // NB.  We assume our caller has locked the sequence mutex.
    
    GDBRemoteCommunicationClient &gdb_comm (((ProcessGDBRemote *) process)->GetGDBRemote());
    if (!gdb_comm.GetSyncThreadStateSupported())
        return;

    StreamString packet;
    StringExtractorGDBRemote response;
    packet.Printf ("QSyncThreadState:%4.4" PRIx64 ";", m_thread.GetProtocolID());
    if (gdb_comm.SendPacketAndWaitForResponse(packet.GetString().c_str(),
                                              packet.GetString().size(),
                                              response,
                                              false) == GDBRemoteCommunication::PacketResult::Success)
    {
        if (response.IsOKResponse())
            InvalidateAllRegisters();
    }
}

bool
GDBRemoteRegisterContext::WriteRegisterBytes (const RegisterInfo *reg_info, DataExtractor &data, uint32_t data_offset)
{
    ExecutionContext exe_ctx (CalculateThread());

    Process *process = exe_ctx.GetProcessPtr();
    Thread *thread = exe_ctx.GetThreadPtr();
    if (process == NULL || thread == NULL)
        return false;

    GDBRemoteCommunicationClient &gdb_comm (((ProcessGDBRemote *)process)->GetGDBRemote());
// FIXME: This check isn't right because IsRunning checks the Public state, but this
// is work you need to do - for instance in ShouldStop & friends - before the public
// state has been changed.
//    if (gdb_comm.IsRunning())
//        return false;


#if defined (LLDB_CONFIGURATION_DEBUG)
    assert (m_reg_data.GetByteSize() >= reg_info->byte_offset + reg_info->byte_size);
#endif

    // If our register context and our register info disagree, which should never happen, don't
    // overwrite past the end of the buffer.
    if (m_reg_data.GetByteSize() < reg_info->byte_offset + reg_info->byte_size)
        return false;

    // Grab a pointer to where we are going to put this register
    uint8_t *dst = const_cast<uint8_t*>(m_reg_data.PeekData(reg_info->byte_offset, reg_info->byte_size));

    if (dst == NULL)
        return false;


    if (data.CopyByteOrderedData (data_offset,                  // src offset
                                  reg_info->byte_size,          // src length
                                  dst,                          // dst
                                  reg_info->byte_size,          // dst length
                                  m_reg_data.GetByteOrder()))   // dst byte order
    {
        Mutex::Locker locker;
        if (gdb_comm.GetSequenceMutex (locker, "Didn't get sequence mutex for write register."))
        {
            const bool thread_suffix_supported = gdb_comm.GetThreadSuffixSupported();
            ProcessSP process_sp (m_thread.GetProcess());
            if (thread_suffix_supported || static_cast<ProcessGDBRemote *>(process_sp.get())->GetGDBRemote().SetCurrentThread(m_thread.GetProtocolID()))
            {
                StreamString packet;
                StringExtractorGDBRemote response;
                
                if (m_read_all_at_once)
                {
                    // Set all registers in one packet
                    packet.PutChar ('G');
                    packet.PutBytesAsRawHex8 (m_reg_data.GetDataStart(),
                                              m_reg_data.GetByteSize(),
                                              endian::InlHostByteOrder(),
                                              endian::InlHostByteOrder());

                    if (thread_suffix_supported)
                        packet.Printf (";thread:%4.4" PRIx64 ";", m_thread.GetProtocolID());

                    // Invalidate all register values
                    InvalidateIfNeeded (true);

                    if (gdb_comm.SendPacketAndWaitForResponse(packet.GetString().c_str(),
                                                              packet.GetString().size(),
                                                              response,
                                                              false) == GDBRemoteCommunication::PacketResult::Success)
                    {
                        SetAllRegisterValid (false);
                        if (response.IsOKResponse())
                        {
                            return true;
                        }
                    }
                }
                else
                {
                    bool success = true;

                    if (reg_info->value_regs)
                    {
                        // This register is part of another register. In this case we read the actual
                        // register data for any "value_regs", and once all that data is read, we will
                        // have enough data in our register context bytes for the value of this register
                        
                        // Invalidate this composite register first.
                        
                        for (uint32_t idx = 0; success; ++idx)
                        {
                            const uint32_t reg = reg_info->value_regs[idx];
                            if (reg == LLDB_INVALID_REGNUM)
                                break;
                            // We have a valid primordial register as our constituent.
                            // Grab the corresponding register info.
                            const RegisterInfo *value_reg_info = GetRegisterInfoAtIndex(reg);
                            if (value_reg_info == NULL)
                                success = false;
                            else
                                success = SetPrimordialRegister(value_reg_info, gdb_comm);
                        }
                    }
                    else
                    {
                        // This is an actual register, write it
                        success = SetPrimordialRegister(reg_info, gdb_comm);
                    }

                    // Check if writing this register will invalidate any other register values?
                    // If so, invalidate them
                    if (reg_info->invalidate_regs)
                    {
                        for (uint32_t idx = 0, reg = reg_info->invalidate_regs[0];
                             reg != LLDB_INVALID_REGNUM;
                             reg = reg_info->invalidate_regs[++idx])
                        {
                            SetRegisterIsValid(reg, false);
                        }
                    }
                    
                    return success;
                }
            }
        }
        else
        {
            Log *log (ProcessGDBRemoteLog::GetLogIfAnyCategoryIsSet (GDBR_LOG_THREAD | GDBR_LOG_PACKETS));
            if (log)
            {
                if (log->GetVerbose())
                {
                    StreamString strm;
                    gdb_comm.DumpHistory(strm);
                    log->Printf("error: failed to get packet sequence mutex, not sending write register for \"%s\":\n%s", reg_info->name, strm.GetData());
                }
                else
                    log->Printf("error: failed to get packet sequence mutex, not sending write register for \"%s\"", reg_info->name);
            }
        }
    }
    return false;
}

bool
GDBRemoteRegisterContext::ReadAllRegisterValues (RegisterCheckpoint &reg_checkpoint)
{
    ExecutionContext exe_ctx (CalculateThread());
    
    Process *process = exe_ctx.GetProcessPtr();
    Thread *thread = exe_ctx.GetThreadPtr();
    if (process == NULL || thread == NULL)
        return false;
    
    GDBRemoteCommunicationClient &gdb_comm (((ProcessGDBRemote *)process)->GetGDBRemote());

    uint32_t save_id = 0;
    if (gdb_comm.SaveRegisterState(thread->GetProtocolID(), save_id))
    {
        reg_checkpoint.SetID(save_id);
        reg_checkpoint.GetData().reset();
        return true;
    }
    else
    {
        reg_checkpoint.SetID(0); // Invalid save ID is zero
        return ReadAllRegisterValues(reg_checkpoint.GetData());
    }
}

bool
GDBRemoteRegisterContext::WriteAllRegisterValues (const RegisterCheckpoint &reg_checkpoint)
{
    uint32_t save_id = reg_checkpoint.GetID();
    if (save_id != 0)
    {
        ExecutionContext exe_ctx (CalculateThread());
        
        Process *process = exe_ctx.GetProcessPtr();
        Thread *thread = exe_ctx.GetThreadPtr();
        if (process == NULL || thread == NULL)
            return false;
        
        GDBRemoteCommunicationClient &gdb_comm (((ProcessGDBRemote *)process)->GetGDBRemote());
        
        return gdb_comm.RestoreRegisterState(m_thread.GetProtocolID(), save_id);
    }
    else
    {
        return WriteAllRegisterValues(reg_checkpoint.GetData());
    }
}

bool
GDBRemoteRegisterContext::ReadAllRegisterValues (lldb::DataBufferSP &data_sp)
{
    ExecutionContext exe_ctx (CalculateThread());

    Process *process = exe_ctx.GetProcessPtr();
    Thread *thread = exe_ctx.GetThreadPtr();
    if (process == NULL || thread == NULL)
        return false;

    GDBRemoteCommunicationClient &gdb_comm (((ProcessGDBRemote *)process)->GetGDBRemote());

    StringExtractorGDBRemote response;

    const bool use_g_packet = gdb_comm.AvoidGPackets ((ProcessGDBRemote *)process) == false;

    Mutex::Locker locker;
    if (gdb_comm.GetSequenceMutex (locker, "Didn't get sequence mutex for read all registers."))
    {
        SyncThreadState(process);
        
        char packet[32];
        const bool thread_suffix_supported = gdb_comm.GetThreadSuffixSupported();
        ProcessSP process_sp (m_thread.GetProcess());
        if (thread_suffix_supported || static_cast<ProcessGDBRemote *>(process_sp.get())->GetGDBRemote().SetCurrentThread(m_thread.GetProtocolID()))
        {
            int packet_len = 0;
            if (thread_suffix_supported)
                packet_len = ::snprintf (packet, sizeof(packet), "g;thread:%4.4" PRIx64, m_thread.GetProtocolID());
            else
                packet_len = ::snprintf (packet, sizeof(packet), "g");
            assert (packet_len < ((int)sizeof(packet) - 1));

            if (use_g_packet && gdb_comm.SendPacketAndWaitForResponse(packet, packet_len, response, false) == GDBRemoteCommunication::PacketResult::Success)
            {
                int packet_len = 0;
                if (thread_suffix_supported)
                    packet_len = ::snprintf (packet, sizeof(packet), "g;thread:%4.4" PRIx64, m_thread.GetProtocolID());
                else
                    packet_len = ::snprintf (packet, sizeof(packet), "g");
                assert (packet_len < ((int)sizeof(packet) - 1));
    
                if (gdb_comm.SendPacketAndWaitForResponse(packet, packet_len, response, false) == GDBRemoteCommunication::PacketResult::Success)
                {
                    if (response.IsErrorResponse())
                        return false;
    
                    std::string &response_str = response.GetStringRef();
                    if (isxdigit(response_str[0]))
                    {
                        response_str.insert(0, 1, 'G');
                        if (thread_suffix_supported)
                        {
                            char thread_id_cstr[64];
                            ::snprintf (thread_id_cstr, sizeof(thread_id_cstr), ";thread:%4.4" PRIx64 ";", m_thread.GetProtocolID());
                            response_str.append (thread_id_cstr);
                        }
                        data_sp.reset (new DataBufferHeap (response_str.c_str(), response_str.size()));
                        return true;
                    }
                }
            }
            else
            {
                // For the use_g_packet == false case, we're going to read each register 
                // individually and store them as binary data in a buffer instead of as ascii
                // characters.
                const RegisterInfo *reg_info;

                // data_sp will take ownership of this DataBufferHeap pointer soon.
                DataBufferSP reg_ctx(new DataBufferHeap(m_reg_info.GetRegisterDataByteSize(), 0));

                for (uint32_t i = 0; (reg_info = GetRegisterInfoAtIndex (i)) != NULL; i++)
                {
                    if (reg_info->value_regs) // skip registers that are slices of real registers
                        continue;
                    ReadRegisterBytes (reg_info, m_reg_data);
                    // ReadRegisterBytes saves the contents of the register in to the m_reg_data buffer
                }
                memcpy (reg_ctx->GetBytes(), m_reg_data.GetDataStart(), m_reg_info.GetRegisterDataByteSize());

                data_sp = reg_ctx;
                return true;
            }
        }
    }
    else
    {

        Log *log (ProcessGDBRemoteLog::GetLogIfAnyCategoryIsSet (GDBR_LOG_THREAD | GDBR_LOG_PACKETS));
        if (log)
        {
            if (log->GetVerbose())
            {
                StreamString strm;
                gdb_comm.DumpHistory(strm);
                log->Printf("error: failed to get packet sequence mutex, not sending read all registers:\n%s", strm.GetData());
            }
            else
                log->Printf("error: failed to get packet sequence mutex, not sending read all registers");
        }
    }

    data_sp.reset();
    return false;
}

bool
GDBRemoteRegisterContext::WriteAllRegisterValues (const lldb::DataBufferSP &data_sp)
{
    if (!data_sp || data_sp->GetBytes() == NULL || data_sp->GetByteSize() == 0)
        return false;

    ExecutionContext exe_ctx (CalculateThread());

    Process *process = exe_ctx.GetProcessPtr();
    Thread *thread = exe_ctx.GetThreadPtr();
    if (process == NULL || thread == NULL)
        return false;

    GDBRemoteCommunicationClient &gdb_comm (((ProcessGDBRemote *)process)->GetGDBRemote());

    const bool use_g_packet = gdb_comm.AvoidGPackets ((ProcessGDBRemote *)process) == false;

    StringExtractorGDBRemote response;
    Mutex::Locker locker;
    if (gdb_comm.GetSequenceMutex (locker, "Didn't get sequence mutex for write all registers."))
    {
        const bool thread_suffix_supported = gdb_comm.GetThreadSuffixSupported();
        ProcessSP process_sp (m_thread.GetProcess());
        if (thread_suffix_supported || static_cast<ProcessGDBRemote *>(process_sp.get())->GetGDBRemote().SetCurrentThread(m_thread.GetProtocolID()))
        {
            // The data_sp contains the entire G response packet including the
            // G, and if the thread suffix is supported, it has the thread suffix
            // as well.
            const char *G_packet = (const char *)data_sp->GetBytes();
            size_t G_packet_len = data_sp->GetByteSize();
            if (use_g_packet
                && gdb_comm.SendPacketAndWaitForResponse (G_packet,
                                                          G_packet_len,
                                                          response,
                                                          false) == GDBRemoteCommunication::PacketResult::Success)
            {
                // The data_sp contains the entire G response packet including the
                // G, and if the thread suffix is supported, it has the thread suffix
                // as well.
                const char *G_packet = (const char *)data_sp->GetBytes();
                size_t G_packet_len = data_sp->GetByteSize();
                if (gdb_comm.SendPacketAndWaitForResponse (G_packet,
                                                           G_packet_len,
                                                           response,
                                                           false) == GDBRemoteCommunication::PacketResult::Success)
                {
                    if (response.IsOKResponse())
                        return true;
                    else if (response.IsErrorResponse())
                    {
                        uint32_t num_restored = 0;
                        // We need to manually go through all of the registers and
                        // restore them manually
    
                        response.GetStringRef().assign (G_packet, G_packet_len);
                        response.SetFilePos(1); // Skip the leading 'G'

                        // G_packet_len is hex-ascii characters plus prefix 'G' plus suffix thread specifier.
                        // This means buffer will be a little more than 2x larger than necessary but we resize
                        // it down once we've extracted all hex ascii chars from the packet.
                        DataBufferHeap buffer (G_packet_len, 0);

                        const uint32_t bytes_extracted = response.GetHexBytes (buffer.GetBytes(),
                                                                               buffer.GetByteSize(),
                                                                               '\xcc');

                        DataExtractor restore_data (buffer.GetBytes(),
                                                    buffer.GetByteSize(),
                                                    m_reg_data.GetByteOrder(),
                                                    m_reg_data.GetAddressByteSize());

                        if (bytes_extracted < restore_data.GetByteSize())
                            restore_data.SetData(restore_data.GetDataStart(), bytes_extracted, m_reg_data.GetByteOrder());
    
                        const RegisterInfo *reg_info;

                        // The g packet contents may either include the slice registers (registers defined in
                        // terms of other registers, e.g. eax is a subset of rax) or not.  The slice registers 
                        // should NOT be in the g packet, but some implementations may incorrectly include them.
                        // 
                        // If the slice registers are included in the packet, we must step over the slice registers 
                        // when parsing the packet -- relying on the RegisterInfo byte_offset field would be incorrect.
                        // If the slice registers are not included, then using the byte_offset values into the
                        // data buffer is the best way to find individual register values.

                        uint64_t size_including_slice_registers = 0;
                        uint64_t size_not_including_slice_registers = 0;
                        uint64_t size_by_highest_offset = 0;

                        for (uint32_t reg_idx=0; (reg_info = GetRegisterInfoAtIndex (reg_idx)) != NULL; ++reg_idx)
                        {
                            size_including_slice_registers += reg_info->byte_size;
                            if (reg_info->value_regs == NULL)
                                size_not_including_slice_registers += reg_info->byte_size;
                            if (reg_info->byte_offset >= size_by_highest_offset)
                                size_by_highest_offset = reg_info->byte_offset + reg_info->byte_size;
                        }

                        bool use_byte_offset_into_buffer;
                        if (size_by_highest_offset == restore_data.GetByteSize())
                        {
                            // The size of the packet agrees with the highest offset: + size in the register file
                            use_byte_offset_into_buffer = true;
                        }
                        else if (size_not_including_slice_registers == restore_data.GetByteSize())
                        {
                            // The size of the packet is the same as concatenating all of the registers sequentially,
                            // skipping the slice registers
                            use_byte_offset_into_buffer = true;
                        }
                        else if (size_including_slice_registers == restore_data.GetByteSize())
                        {
                            // The slice registers are present in the packet (when they shouldn't be).
                            // Don't try to use the RegisterInfo byte_offset into the restore_data, it will
                            // point to the wrong place.
                            use_byte_offset_into_buffer = false;
                        }
                        else {
                            // None of our expected sizes match the actual g packet data we're looking at.
                            // The most conservative approach here is to use the running total byte offset.
                            use_byte_offset_into_buffer = false;
                        }

                        // In case our register definitions don't include the correct offsets,
                        // keep track of the size of each reg & compute offset based on that.
                        uint32_t running_byte_offset = 0;
                        for (uint32_t reg_idx=0; (reg_info = GetRegisterInfoAtIndex (reg_idx)) != NULL; ++reg_idx, running_byte_offset += reg_info->byte_size)
                        {
                            // Skip composite aka slice registers (e.g. eax is a slice of rax).
                            if (reg_info->value_regs)
                                continue;

                            const uint32_t reg = reg_info->kinds[eRegisterKindLLDB];

                            uint32_t register_offset;
                            if (use_byte_offset_into_buffer)
                            {
                                register_offset = reg_info->byte_offset;
                            }
                            else
                            {
                                register_offset = running_byte_offset;
                            }

                            // Only write down the registers that need to be written
                            // if we are going to be doing registers individually.
                            bool write_reg = true;
                            const uint32_t reg_byte_size = reg_info->byte_size;
    
                            const char *restore_src = (const char *)restore_data.PeekData(register_offset, reg_byte_size);
                            if (restore_src)
                            {
                                StreamString packet;
                                packet.Printf ("P%x=", reg_info->kinds[eRegisterKindProcessPlugin]);
                                packet.PutBytesAsRawHex8 (restore_src,
                                                          reg_byte_size,
                                                          endian::InlHostByteOrder(),
                                                          endian::InlHostByteOrder());

                                if (thread_suffix_supported)
                                    packet.Printf (";thread:%4.4" PRIx64 ";", m_thread.GetProtocolID());

                                SetRegisterIsValid(reg, false);
                                if (gdb_comm.SendPacketAndWaitForResponse(packet.GetString().c_str(),
                                                                          packet.GetString().size(),
                                                                          response,
                                                                          false) == GDBRemoteCommunication::PacketResult::Success)
                                {
                                    const char *current_src = (const char *)m_reg_data.PeekData(register_offset, reg_byte_size);
                                    if (current_src)
                                        write_reg = memcmp (current_src, restore_src, reg_byte_size) != 0;
                                }
    
                                if (write_reg)
                                {
                                    StreamString packet;
                                    packet.Printf ("P%x=", reg_info->kinds[eRegisterKindProcessPlugin]);
                                    packet.PutBytesAsRawHex8 (restore_src,
                                                              reg_byte_size,
                                                              endian::InlHostByteOrder(),
                                                              endian::InlHostByteOrder());
    
                                    if (thread_suffix_supported)
                                        packet.Printf (";thread:%4.4" PRIx64 ";", m_thread.GetProtocolID());
    
                                    SetRegisterIsValid(reg, false);
                                    if (gdb_comm.SendPacketAndWaitForResponse(packet.GetString().c_str(),
                                                                              packet.GetString().size(),
                                                                              response,
                                                                              false) == GDBRemoteCommunication::PacketResult::Success)
                                    {
                                        if (response.IsOKResponse())
                                            ++num_restored;
                                    }
                                }
                            }
                        }
                        return num_restored > 0;
                    }
                }
            }
            else
            {
                // For the use_g_packet == false case, we're going to write each register 
                // individually.  The data buffer is binary data in this case, instead of 
                // ascii characters.

                bool arm64_debugserver = false;
                if (m_thread.GetProcess().get())
                {
                    const ArchSpec &arch = m_thread.GetProcess()->GetTarget().GetArchitecture();
                    if (arch.IsValid()
                        && arch.GetMachine() == llvm::Triple::aarch64
                        && arch.GetTriple().getVendor() == llvm::Triple::Apple
                        && arch.GetTriple().getOS() == llvm::Triple::IOS)
                    {
                        arm64_debugserver = true;
                    }
                }
                uint32_t num_restored = 0;
                const RegisterInfo *reg_info;
                for (uint32_t i = 0; (reg_info = GetRegisterInfoAtIndex (i)) != NULL; i++)
                {
                    if (reg_info->value_regs) // skip registers that are slices of real registers
                        continue;
                    // Skip the fpsr and fpcr floating point status/control register writing to
                    // work around a bug in an older version of debugserver that would lead to
                    // register context corruption when writing fpsr/fpcr.
                    if (arm64_debugserver &&
                        (strcmp (reg_info->name, "fpsr") == 0 || strcmp (reg_info->name, "fpcr") == 0))
                    {
                        continue;
                    }
                    StreamString packet;
                    packet.Printf ("P%x=", reg_info->kinds[eRegisterKindProcessPlugin]);
                    packet.PutBytesAsRawHex8 (data_sp->GetBytes() + reg_info->byte_offset, reg_info->byte_size, endian::InlHostByteOrder(), endian::InlHostByteOrder());
                    if (thread_suffix_supported)
                        packet.Printf (";thread:%4.4" PRIx64 ";", m_thread.GetProtocolID());

                    SetRegisterIsValid(reg_info, false);
                    if (gdb_comm.SendPacketAndWaitForResponse(packet.GetString().c_str(),
                                                              packet.GetString().size(),
                                                              response,
                                                              false) == GDBRemoteCommunication::PacketResult::Success)
                    {
                        if (response.IsOKResponse())
                            ++num_restored;
                    }
                }
                return num_restored > 0;
            }
        }
    }
    else
    {
        Log *log (ProcessGDBRemoteLog::GetLogIfAnyCategoryIsSet (GDBR_LOG_THREAD | GDBR_LOG_PACKETS));
        if (log)
        {
            if (log->GetVerbose())
            {
                StreamString strm;
                gdb_comm.DumpHistory(strm);
                log->Printf("error: failed to get packet sequence mutex, not sending write all registers:\n%s", strm.GetData());
            }
            else
                log->Printf("error: failed to get packet sequence mutex, not sending write all registers");
        }
    }
    return false;
}


uint32_t
GDBRemoteRegisterContext::ConvertRegisterKindToRegisterNumber (lldb::RegisterKind kind, uint32_t num)
{
    return m_reg_info.ConvertRegisterKindToRegisterNumber (kind, num);
}


void
GDBRemoteDynamicRegisterInfo::HardcodeARMRegisters(bool from_scratch)
{
    // For Advanced SIMD and VFP register mapping.
    static uint32_t g_d0_regs[] =  { 26, 27, LLDB_INVALID_REGNUM }; // (s0, s1)
    static uint32_t g_d1_regs[] =  { 28, 29, LLDB_INVALID_REGNUM }; // (s2, s3)
    static uint32_t g_d2_regs[] =  { 30, 31, LLDB_INVALID_REGNUM }; // (s4, s5)
    static uint32_t g_d3_regs[] =  { 32, 33, LLDB_INVALID_REGNUM }; // (s6, s7)
    static uint32_t g_d4_regs[] =  { 34, 35, LLDB_INVALID_REGNUM }; // (s8, s9)
    static uint32_t g_d5_regs[] =  { 36, 37, LLDB_INVALID_REGNUM }; // (s10, s11)
    static uint32_t g_d6_regs[] =  { 38, 39, LLDB_INVALID_REGNUM }; // (s12, s13)
    static uint32_t g_d7_regs[] =  { 40, 41, LLDB_INVALID_REGNUM }; // (s14, s15)
    static uint32_t g_d8_regs[] =  { 42, 43, LLDB_INVALID_REGNUM }; // (s16, s17)
    static uint32_t g_d9_regs[] =  { 44, 45, LLDB_INVALID_REGNUM }; // (s18, s19)
    static uint32_t g_d10_regs[] = { 46, 47, LLDB_INVALID_REGNUM }; // (s20, s21)
    static uint32_t g_d11_regs[] = { 48, 49, LLDB_INVALID_REGNUM }; // (s22, s23)
    static uint32_t g_d12_regs[] = { 50, 51, LLDB_INVALID_REGNUM }; // (s24, s25)
    static uint32_t g_d13_regs[] = { 52, 53, LLDB_INVALID_REGNUM }; // (s26, s27)
    static uint32_t g_d14_regs[] = { 54, 55, LLDB_INVALID_REGNUM }; // (s28, s29)
    static uint32_t g_d15_regs[] = { 56, 57, LLDB_INVALID_REGNUM }; // (s30, s31)
    static uint32_t g_q0_regs[] =  { 26, 27, 28, 29, LLDB_INVALID_REGNUM }; // (d0, d1) -> (s0, s1, s2, s3)
    static uint32_t g_q1_regs[] =  { 30, 31, 32, 33, LLDB_INVALID_REGNUM }; // (d2, d3) -> (s4, s5, s6, s7)
    static uint32_t g_q2_regs[] =  { 34, 35, 36, 37, LLDB_INVALID_REGNUM }; // (d4, d5) -> (s8, s9, s10, s11)
    static uint32_t g_q3_regs[] =  { 38, 39, 40, 41, LLDB_INVALID_REGNUM }; // (d6, d7) -> (s12, s13, s14, s15)
    static uint32_t g_q4_regs[] =  { 42, 43, 44, 45, LLDB_INVALID_REGNUM }; // (d8, d9) -> (s16, s17, s18, s19)
    static uint32_t g_q5_regs[] =  { 46, 47, 48, 49, LLDB_INVALID_REGNUM }; // (d10, d11) -> (s20, s21, s22, s23)
    static uint32_t g_q6_regs[] =  { 50, 51, 52, 53, LLDB_INVALID_REGNUM }; // (d12, d13) -> (s24, s25, s26, s27)
    static uint32_t g_q7_regs[] =  { 54, 55, 56, 57, LLDB_INVALID_REGNUM }; // (d14, d15) -> (s28, s29, s30, s31)
    static uint32_t g_q8_regs[] =  { 59, 60, LLDB_INVALID_REGNUM }; // (d16, d17)
    static uint32_t g_q9_regs[] =  { 61, 62, LLDB_INVALID_REGNUM }; // (d18, d19)
    static uint32_t g_q10_regs[] = { 63, 64, LLDB_INVALID_REGNUM }; // (d20, d21)
    static uint32_t g_q11_regs[] = { 65, 66, LLDB_INVALID_REGNUM }; // (d22, d23)
    static uint32_t g_q12_regs[] = { 67, 68, LLDB_INVALID_REGNUM }; // (d24, d25)
    static uint32_t g_q13_regs[] = { 69, 70, LLDB_INVALID_REGNUM }; // (d26, d27)
    static uint32_t g_q14_regs[] = { 71, 72, LLDB_INVALID_REGNUM }; // (d28, d29)
    static uint32_t g_q15_regs[] = { 73, 74, LLDB_INVALID_REGNUM }; // (d30, d31)

    // This is our array of composite registers, with each element coming from the above register mappings.
    static uint32_t *g_composites[] = {
        g_d0_regs, g_d1_regs,  g_d2_regs,  g_d3_regs,  g_d4_regs,  g_d5_regs,  g_d6_regs,  g_d7_regs,
        g_d8_regs, g_d9_regs, g_d10_regs, g_d11_regs, g_d12_regs, g_d13_regs, g_d14_regs, g_d15_regs,
        g_q0_regs, g_q1_regs,  g_q2_regs,  g_q3_regs,  g_q4_regs,  g_q5_regs,  g_q6_regs,  g_q7_regs,
        g_q8_regs, g_q9_regs, g_q10_regs, g_q11_regs, g_q12_regs, g_q13_regs, g_q14_regs, g_q15_regs
    };

    static RegisterInfo g_register_infos[] = {
//   NAME    ALT    SZ  OFF  ENCODING          FORMAT          EH_FRAME             DWARF                GENERIC                 PROCESS PLUGIN  LLDB      VALUE REGS    INVALIDATE REGS
//   ======  ====== === ===  =============     ============    ===================  ===================  ======================  =============   ====      ==========    ===============
    { "r0", "arg1",   4,   0, eEncodingUint,    eFormatHex,   { ehframe_r0,          dwarf_r0,            LLDB_REGNUM_GENERIC_ARG1,0,               0 },        NULL,              NULL},
    { "r1", "arg2",   4,   0, eEncodingUint,    eFormatHex,   { ehframe_r1,          dwarf_r1,            LLDB_REGNUM_GENERIC_ARG2,1,               1 },        NULL,              NULL},
    { "r2", "arg3",   4,   0, eEncodingUint,    eFormatHex,   { ehframe_r2,          dwarf_r2,            LLDB_REGNUM_GENERIC_ARG3,2,               2 },        NULL,              NULL},
    { "r3", "arg4",   4,   0, eEncodingUint,    eFormatHex,   { ehframe_r3,          dwarf_r3,            LLDB_REGNUM_GENERIC_ARG4,3,               3 },        NULL,              NULL},
    { "r4",   NULL,   4,   0, eEncodingUint,    eFormatHex,   { ehframe_r4,          dwarf_r4,            LLDB_INVALID_REGNUM,     4,               4 },        NULL,              NULL},
    { "r5",   NULL,   4,   0, eEncodingUint,    eFormatHex,   { ehframe_r5,          dwarf_r5,            LLDB_INVALID_REGNUM,     5,               5 },        NULL,              NULL},
    { "r6",   NULL,   4,   0, eEncodingUint,    eFormatHex,   { ehframe_r6,          dwarf_r6,            LLDB_INVALID_REGNUM,     6,               6 },        NULL,              NULL},
    { "r7",   "fp",   4,   0, eEncodingUint,    eFormatHex,   { ehframe_r7,          dwarf_r7,            LLDB_REGNUM_GENERIC_FP,  7,               7 },        NULL,              NULL},
    { "r8",   NULL,   4,   0, eEncodingUint,    eFormatHex,   { ehframe_r8,          dwarf_r8,            LLDB_INVALID_REGNUM,     8,               8 },        NULL,              NULL},
    { "r9",   NULL,   4,   0, eEncodingUint,    eFormatHex,   { ehframe_r9,          dwarf_r9,            LLDB_INVALID_REGNUM,     9,               9 },        NULL,              NULL},
    { "r10",  NULL,   4,   0, eEncodingUint,    eFormatHex,   { ehframe_r10,         dwarf_r10,           LLDB_INVALID_REGNUM,    10,              10 },        NULL,              NULL},
    { "r11",  NULL,   4,   0, eEncodingUint,    eFormatHex,   { ehframe_r11,         dwarf_r11,           LLDB_INVALID_REGNUM,    11,              11 },        NULL,              NULL},
    { "r12",  NULL,   4,   0, eEncodingUint,    eFormatHex,   { ehframe_r12,         dwarf_r12,           LLDB_INVALID_REGNUM,    12,              12 },        NULL,              NULL},
    { "sp",   "r13",  4,   0, eEncodingUint,    eFormatHex,   { ehframe_sp,          dwarf_sp,            LLDB_REGNUM_GENERIC_SP, 13,              13 },        NULL,              NULL},
    { "lr",   "r14",  4,   0, eEncodingUint,    eFormatHex,   { ehframe_lr,          dwarf_lr,            LLDB_REGNUM_GENERIC_RA, 14,              14 },        NULL,              NULL},
    { "pc",   "r15",  4,   0, eEncodingUint,    eFormatHex,   { ehframe_pc,          dwarf_pc,            LLDB_REGNUM_GENERIC_PC, 15,              15 },        NULL,              NULL},
    { "f0",   NULL,  12,   0, eEncodingUint,    eFormatHex,   { LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,    16,              16 },        NULL,              NULL},
    { "f1",   NULL,  12,   0, eEncodingUint,    eFormatHex,   { LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,    17,              17 },        NULL,              NULL},
    { "f2",   NULL,  12,   0, eEncodingUint,    eFormatHex,   { LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,    18,              18 },        NULL,              NULL},
    { "f3",   NULL,  12,   0, eEncodingUint,    eFormatHex,   { LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,    19,              19 },        NULL,              NULL},
    { "f4",   NULL,  12,   0, eEncodingUint,    eFormatHex,   { LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,    20,              20 },        NULL,              NULL},
    { "f5",   NULL,  12,   0, eEncodingUint,    eFormatHex,   { LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,    21,              21 },        NULL,              NULL},
    { "f6",   NULL,  12,   0, eEncodingUint,    eFormatHex,   { LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,    22,              22 },        NULL,              NULL},
    { "f7",   NULL,  12,   0, eEncodingUint,    eFormatHex,   { LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,    23,              23 },        NULL,              NULL},
    { "fps",  NULL,   4,   0, eEncodingUint,    eFormatHex,   { LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,    24,              24 },        NULL,              NULL},
    { "cpsr","flags", 4,   0, eEncodingUint,    eFormatHex,   { ehframe_cpsr,        dwarf_cpsr,          LLDB_INVALID_REGNUM,    25,              25 },        NULL,              NULL},
    { "s0",   NULL,   4,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_s0,            LLDB_INVALID_REGNUM,    26,              26 },        NULL,              NULL},
    { "s1",   NULL,   4,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_s1,            LLDB_INVALID_REGNUM,    27,              27 },        NULL,              NULL},
    { "s2",   NULL,   4,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_s2,            LLDB_INVALID_REGNUM,    28,              28 },        NULL,              NULL},
    { "s3",   NULL,   4,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_s3,            LLDB_INVALID_REGNUM,    29,              29 },        NULL,              NULL},
    { "s4",   NULL,   4,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_s4,            LLDB_INVALID_REGNUM,    30,              30 },        NULL,              NULL},
    { "s5",   NULL,   4,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_s5,            LLDB_INVALID_REGNUM,    31,              31 },        NULL,              NULL},
    { "s6",   NULL,   4,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_s6,            LLDB_INVALID_REGNUM,    32,              32 },        NULL,              NULL},
    { "s7",   NULL,   4,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_s7,            LLDB_INVALID_REGNUM,    33,              33 },        NULL,              NULL},
    { "s8",   NULL,   4,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_s8,            LLDB_INVALID_REGNUM,    34,              34 },        NULL,              NULL},
    { "s9",   NULL,   4,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_s9,            LLDB_INVALID_REGNUM,    35,              35 },        NULL,              NULL},
    { "s10",  NULL,   4,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_s10,           LLDB_INVALID_REGNUM,    36,              36 },        NULL,              NULL},
    { "s11",  NULL,   4,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_s11,           LLDB_INVALID_REGNUM,    37,              37 },        NULL,              NULL},
    { "s12",  NULL,   4,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_s12,           LLDB_INVALID_REGNUM,    38,              38 },        NULL,              NULL},
    { "s13",  NULL,   4,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_s13,           LLDB_INVALID_REGNUM,    39,              39 },        NULL,              NULL},
    { "s14",  NULL,   4,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_s14,           LLDB_INVALID_REGNUM,    40,              40 },        NULL,              NULL},
    { "s15",  NULL,   4,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_s15,           LLDB_INVALID_REGNUM,    41,              41 },        NULL,              NULL},
    { "s16",  NULL,   4,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_s16,           LLDB_INVALID_REGNUM,    42,              42 },        NULL,              NULL},
    { "s17",  NULL,   4,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_s17,           LLDB_INVALID_REGNUM,    43,              43 },        NULL,              NULL},
    { "s18",  NULL,   4,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_s18,           LLDB_INVALID_REGNUM,    44,              44 },        NULL,              NULL},
    { "s19",  NULL,   4,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_s19,           LLDB_INVALID_REGNUM,    45,              45 },        NULL,              NULL},
    { "s20",  NULL,   4,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_s20,           LLDB_INVALID_REGNUM,    46,              46 },        NULL,              NULL},
    { "s21",  NULL,   4,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_s21,           LLDB_INVALID_REGNUM,    47,              47 },        NULL,              NULL},
    { "s22",  NULL,   4,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_s22,           LLDB_INVALID_REGNUM,    48,              48 },        NULL,              NULL},
    { "s23",  NULL,   4,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_s23,           LLDB_INVALID_REGNUM,    49,              49 },        NULL,              NULL},
    { "s24",  NULL,   4,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_s24,           LLDB_INVALID_REGNUM,    50,              50 },        NULL,              NULL},
    { "s25",  NULL,   4,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_s25,           LLDB_INVALID_REGNUM,    51,              51 },        NULL,              NULL},
    { "s26",  NULL,   4,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_s26,           LLDB_INVALID_REGNUM,    52,              52 },        NULL,              NULL},
    { "s27",  NULL,   4,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_s27,           LLDB_INVALID_REGNUM,    53,              53 },        NULL,              NULL},
    { "s28",  NULL,   4,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_s28,           LLDB_INVALID_REGNUM,    54,              54 },        NULL,              NULL},
    { "s29",  NULL,   4,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_s29,           LLDB_INVALID_REGNUM,    55,              55 },        NULL,              NULL},
    { "s30",  NULL,   4,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_s30,           LLDB_INVALID_REGNUM,    56,              56 },        NULL,              NULL},
    { "s31",  NULL,   4,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_s31,           LLDB_INVALID_REGNUM,    57,              57 },        NULL,              NULL},
    { "fpscr",NULL,   4,   0, eEncodingUint,    eFormatHex,   { LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,    58,              58 },        NULL,              NULL},
    { "d16",  NULL,   8,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_d16,           LLDB_INVALID_REGNUM,    59,              59 },        NULL,              NULL},
    { "d17",  NULL,   8,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_d17,           LLDB_INVALID_REGNUM,    60,              60 },        NULL,              NULL},
    { "d18",  NULL,   8,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_d18,           LLDB_INVALID_REGNUM,    61,              61 },        NULL,              NULL},
    { "d19",  NULL,   8,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_d19,           LLDB_INVALID_REGNUM,    62,              62 },        NULL,              NULL},
    { "d20",  NULL,   8,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_d20,           LLDB_INVALID_REGNUM,    63,              63 },        NULL,              NULL},
    { "d21",  NULL,   8,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_d21,           LLDB_INVALID_REGNUM,    64,              64 },        NULL,              NULL},
    { "d22",  NULL,   8,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_d22,           LLDB_INVALID_REGNUM,    65,              65 },        NULL,              NULL},
    { "d23",  NULL,   8,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_d23,           LLDB_INVALID_REGNUM,    66,              66 },        NULL,              NULL},
    { "d24",  NULL,   8,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_d24,           LLDB_INVALID_REGNUM,    67,              67 },        NULL,              NULL},
    { "d25",  NULL,   8,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_d25,           LLDB_INVALID_REGNUM,    68,              68 },        NULL,              NULL},
    { "d26",  NULL,   8,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_d26,           LLDB_INVALID_REGNUM,    69,              69 },        NULL,              NULL},
    { "d27",  NULL,   8,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_d27,           LLDB_INVALID_REGNUM,    70,              70 },        NULL,              NULL},
    { "d28",  NULL,   8,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_d28,           LLDB_INVALID_REGNUM,    71,              71 },        NULL,              NULL},
    { "d29",  NULL,   8,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_d29,           LLDB_INVALID_REGNUM,    72,              72 },        NULL,              NULL},
    { "d30",  NULL,   8,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_d30,           LLDB_INVALID_REGNUM,    73,              73 },        NULL,              NULL},
    { "d31",  NULL,   8,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_d31,           LLDB_INVALID_REGNUM,    74,              74 },        NULL,              NULL},
    { "d0",   NULL,   8,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_d0,            LLDB_INVALID_REGNUM,    75,              75 },   g_d0_regs,              NULL},
    { "d1",   NULL,   8,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_d1,            LLDB_INVALID_REGNUM,    76,              76 },   g_d1_regs,              NULL},
    { "d2",   NULL,   8,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_d2,            LLDB_INVALID_REGNUM,    77,              77 },   g_d2_regs,              NULL},
    { "d3",   NULL,   8,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_d3,            LLDB_INVALID_REGNUM,    78,              78 },   g_d3_regs,              NULL},
    { "d4",   NULL,   8,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_d4,            LLDB_INVALID_REGNUM,    79,              79 },   g_d4_regs,              NULL},
    { "d5",   NULL,   8,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_d5,            LLDB_INVALID_REGNUM,    80,              80 },   g_d5_regs,              NULL},
    { "d6",   NULL,   8,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_d6,            LLDB_INVALID_REGNUM,    81,              81 },   g_d6_regs,              NULL},
    { "d7",   NULL,   8,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_d7,            LLDB_INVALID_REGNUM,    82,              82 },   g_d7_regs,              NULL},
    { "d8",   NULL,   8,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_d8,            LLDB_INVALID_REGNUM,    83,              83 },   g_d8_regs,              NULL},
    { "d9",   NULL,   8,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_d9,            LLDB_INVALID_REGNUM,    84,              84 },   g_d9_regs,              NULL},
    { "d10",  NULL,   8,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_d10,           LLDB_INVALID_REGNUM,    85,              85 },  g_d10_regs,              NULL},
    { "d11",  NULL,   8,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_d11,           LLDB_INVALID_REGNUM,    86,              86 },  g_d11_regs,              NULL},
    { "d12",  NULL,   8,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_d12,           LLDB_INVALID_REGNUM,    87,              87 },  g_d12_regs,              NULL},
    { "d13",  NULL,   8,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_d13,           LLDB_INVALID_REGNUM,    88,              88 },  g_d13_regs,              NULL},
    { "d14",  NULL,   8,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_d14,           LLDB_INVALID_REGNUM,    89,              89 },  g_d14_regs,              NULL},
    { "d15",  NULL,   8,   0, eEncodingIEEE754, eFormatFloat, { LLDB_INVALID_REGNUM, dwarf_d15,           LLDB_INVALID_REGNUM,    90,              90 },  g_d15_regs,              NULL},
    { "q0",   NULL,   16,  0, eEncodingVector,  eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM, dwarf_q0,    LLDB_INVALID_REGNUM,    91,              91 },   g_q0_regs,              NULL},
    { "q1",   NULL,   16,  0, eEncodingVector,  eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM, dwarf_q1,    LLDB_INVALID_REGNUM,    92,              92 },   g_q1_regs,              NULL},
    { "q2",   NULL,   16,  0, eEncodingVector,  eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM, dwarf_q2,    LLDB_INVALID_REGNUM,    93,              93 },   g_q2_regs,              NULL},
    { "q3",   NULL,   16,  0, eEncodingVector,  eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM, dwarf_q3,    LLDB_INVALID_REGNUM,    94,              94 },   g_q3_regs,              NULL},
    { "q4",   NULL,   16,  0, eEncodingVector,  eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM, dwarf_q4,    LLDB_INVALID_REGNUM,    95,              95 },   g_q4_regs,              NULL},
    { "q5",   NULL,   16,  0, eEncodingVector,  eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM, dwarf_q5,    LLDB_INVALID_REGNUM,    96,              96 },   g_q5_regs,              NULL},
    { "q6",   NULL,   16,  0, eEncodingVector,  eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM, dwarf_q6,    LLDB_INVALID_REGNUM,    97,              97 },   g_q6_regs,              NULL},
    { "q7",   NULL,   16,  0, eEncodingVector,  eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM, dwarf_q7,    LLDB_INVALID_REGNUM,    98,              98 },   g_q7_regs,              NULL},
    { "q8",   NULL,   16,  0, eEncodingVector,  eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM, dwarf_q8,    LLDB_INVALID_REGNUM,    99,              99 },   g_q8_regs,              NULL},
    { "q9",   NULL,   16,  0, eEncodingVector,  eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM, dwarf_q9,    LLDB_INVALID_REGNUM,   100,             100 },   g_q9_regs,              NULL},
    { "q10",  NULL,   16,  0, eEncodingVector,  eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM, dwarf_q10,   LLDB_INVALID_REGNUM,   101,             101 },  g_q10_regs,              NULL},
    { "q11",  NULL,   16,  0, eEncodingVector,  eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM, dwarf_q11,   LLDB_INVALID_REGNUM,   102,             102 },  g_q11_regs,              NULL},
    { "q12",  NULL,   16,  0, eEncodingVector,  eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM, dwarf_q12,   LLDB_INVALID_REGNUM,   103,             103 },  g_q12_regs,              NULL},
    { "q13",  NULL,   16,  0, eEncodingVector,  eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM, dwarf_q13,   LLDB_INVALID_REGNUM,   104,             104 },  g_q13_regs,              NULL},
    { "q14",  NULL,   16,  0, eEncodingVector,  eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM, dwarf_q14,   LLDB_INVALID_REGNUM,   105,             105 },  g_q14_regs,              NULL},
    { "q15",  NULL,   16,  0, eEncodingVector,  eFormatVectorOfUInt8, { LLDB_INVALID_REGNUM, dwarf_q15,   LLDB_INVALID_REGNUM,   106,             106 },  g_q15_regs,              NULL}
    };

    static const uint32_t num_registers = llvm::array_lengthof(g_register_infos);
    static ConstString gpr_reg_set ("General Purpose Registers");
    static ConstString sfp_reg_set ("Software Floating Point Registers");
    static ConstString vfp_reg_set ("Floating Point Registers");
    size_t i;
    if (from_scratch)
    {
        // Calculate the offsets of the registers
        // Note that the layout of the "composite" registers (d0-d15 and q0-q15) which comes after the
        // "primordial" registers is important.  This enables us to calculate the offset of the composite
        // register by using the offset of its first primordial register.  For example, to calculate the
        // offset of q0, use s0's offset.
        if (g_register_infos[2].byte_offset == 0)
        {
            uint32_t byte_offset = 0;
            for (i=0; i<num_registers; ++i)
            {
                // For primordial registers, increment the byte_offset by the byte_size to arrive at the
                // byte_offset for the next register.  Otherwise, we have a composite register whose
                // offset can be calculated by consulting the offset of its first primordial register.
                if (!g_register_infos[i].value_regs)
                {
                    g_register_infos[i].byte_offset = byte_offset;
                    byte_offset += g_register_infos[i].byte_size;
                }
                else
                {
                    const uint32_t first_primordial_reg = g_register_infos[i].value_regs[0];
                    g_register_infos[i].byte_offset = g_register_infos[first_primordial_reg].byte_offset;
                }
            }
        }
        for (i=0; i<num_registers; ++i)
        {
            ConstString name;
            ConstString alt_name;
            if (g_register_infos[i].name && g_register_infos[i].name[0])
                name.SetCString(g_register_infos[i].name);
            if (g_register_infos[i].alt_name && g_register_infos[i].alt_name[0])
                alt_name.SetCString(g_register_infos[i].alt_name);

            if (i <= 15 || i == 25)
                AddRegister (g_register_infos[i], name, alt_name, gpr_reg_set);
            else if (i <= 24)
                AddRegister (g_register_infos[i], name, alt_name, sfp_reg_set);
            else
                AddRegister (g_register_infos[i], name, alt_name, vfp_reg_set);
        }
    }
    else
    {
        // Add composite registers to our primordial registers, then.
        const size_t num_composites = llvm::array_lengthof(g_composites);
        const size_t num_dynamic_regs = GetNumRegisters();
        const size_t num_common_regs = num_registers - num_composites;
        RegisterInfo *g_comp_register_infos = g_register_infos + num_common_regs;

        // First we need to validate that all registers that we already have match the non composite regs.
        // If so, then we can add the registers, else we need to bail
        bool match = true;
        if (num_dynamic_regs == num_common_regs)
        {
            for (i=0; match && i<num_dynamic_regs; ++i)
            {
                // Make sure all register names match
                if (m_regs[i].name && g_register_infos[i].name)
                {
                    if (strcmp(m_regs[i].name, g_register_infos[i].name))
                    {
                        match = false;
                        break;
                    }
                }
                
                // Make sure all register byte sizes match
                if (m_regs[i].byte_size != g_register_infos[i].byte_size)
                {
                    match = false;
                    break;
                }
            }
        }
        else
        {
            // Wrong number of registers.
            match = false;
        }
        // If "match" is true, then we can add extra registers.
        if (match)
        {
            for (i=0; i<num_composites; ++i)
            {
                ConstString name;
                ConstString alt_name;
                const uint32_t first_primordial_reg = g_comp_register_infos[i].value_regs[0];
                const char *reg_name = g_register_infos[first_primordial_reg].name;
                if (reg_name && reg_name[0])
                {
                    for (uint32_t j = 0; j < num_dynamic_regs; ++j)
                    {
                        const RegisterInfo *reg_info = GetRegisterInfoAtIndex(j);
                        // Find a matching primordial register info entry.
                        if (reg_info && reg_info->name && ::strcasecmp(reg_info->name, reg_name) == 0)
                        {
                            // The name matches the existing primordial entry.
                            // Find and assign the offset, and then add this composite register entry.
                            g_comp_register_infos[i].byte_offset = reg_info->byte_offset;
                            name.SetCString(g_comp_register_infos[i].name);
                            AddRegister(g_comp_register_infos[i], name, alt_name, vfp_reg_set);
                        }
                    }
                }
            }
        }
    }
}
