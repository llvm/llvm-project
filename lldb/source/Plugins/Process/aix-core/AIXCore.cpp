//===-- AIXCore.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstring>

#include "lldb/Core/Section.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Stream.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/State.h"

#include "AIXCore.h"

using namespace AIXCORE;
using namespace lldb;
using namespace lldb_private;

AIXCore64Header::AIXCore64Header() { memset(this, 0, sizeof(AIXCore64Header)); }
AIXCore32Header::AIXCore32Header() { memset(this, 0, sizeof(AIXCore32Header)); }


bool AIXCore32Header::ParseRegisterContext(lldb_private::DataExtractor &data,
                lldb::offset_t *offset) {
    *offset += 20; // skip till curid in mstsave32
    Fault.context.excp_type = data.GetU32(offset);
    Fault.context.pc = data.GetU32(offset);
    Fault.context.msr = data.GetU32(offset);
    Fault.context.cr = data.GetU32(offset);
    Fault.context.lr = data.GetU32(offset);
    Fault.context.ctr = data.GetU32(offset);
    Fault.context.xer = data.GetU32(offset);
    // need to skip 0-39 U32s after this upto gpr
    /* *offset += 8; // mq, tid
    Fault.context.fpscr = data.GetU32(offset);
    Fault.context.fpeu = data.GetU8(offset);
    Fault.context.fpinfo = data.GetU8(offset);
    Fault.context.fpscr24_31 = data.GetU8(offset);
    */
    // Skipping unneeded data
    for(int i = 0; i < 40; i++)
        data.GetU32(offset);
    for(int i = 0; i < 32; i++) {
        Fault.context.gpr[i] = data.GetU32(offset);
    }
    for(int i = 0; i < 32; i++)
        Fault.context.fpr[i] = data.GetU32(offset);
    return true;
}

bool AIXCore32Header::ParseThreadContext(lldb_private::DataExtractor &data,
                lldb::offset_t *offset) {
    lldb::offset_t offset_to_regctx = *offset; 
    offset_to_regctx += sizeof(thrdsinfo64);
    Fault.thread.ti_tid = data.GetU32(offset);
    Fault.thread.ti_pid = data.GetU32(offset);
    int ret = ParseRegisterContext(data, &offset_to_regctx);
    return true;
}
 
bool AIXCore32Header::ParseUserData(lldb_private::DataExtractor &data,
                lldb::offset_t *offset) {
    User.process.pi_pid = data.GetU32(offset); 
    User.process.pi_ppid = data.GetU32(offset); 
    User.process.pi_sid = data.GetU32(offset); 
    User.process.pi_pgrp = data.GetU32(offset); 
    User.process.pi_uid = data.GetU32(offset); 
    User.process.pi_suid = data.GetU32(offset);
    *offset += 728; 

    ByteOrder byteorder = data.GetByteOrder();
    size_t size = 33;
    data.ExtractBytes(*offset, size, byteorder, User.process.pi_comm);
    offset += size;

    return true;
}

bool AIXCore32Header::ParseCoreHeader(lldb_private::DataExtractor &data,
                            lldb::offset_t *offset) {
    Log *log = GetLog(LLDBLog::Process);

    SignalNum = data.GetU8(offset);  
    Flag = data.GetU8(offset);  
    Entries = data.GetU16(offset);  
    Version = data.GetU32(offset);
    FDInfo = data.GetU64(offset);

    LoaderOffset = data.GetU64(offset);
    LoaderSize = data.GetU64(offset);
    NumberOfThreads = data.GetU32(offset);
    Reserved0 = data.GetU32(offset);
    ThreadContextOffset = data.GetU64(offset);
    NumSegRegion = data.GetU64(offset);
    SegRegionOffset = data.GetU64(offset);
    StackOffset = data.GetU64(offset);
    StackBaseAddr = data.GetU64(offset);
    StackSize = data.GetU64(offset);
    DataRegionOffset = data.GetU64(offset);
    DataBaseAddr = data.GetU64(offset);
    DataSize = data.GetU64(offset);

    *offset += 104;
    lldb::offset_t offset_to_user = (*offset + sizeof(mstsave32) +
            sizeof(thrdsinfo64));
    int ret = 0;
    ret = ParseThreadContext(data, offset);
    ret = ParseUserData(data, &offset_to_user);
    
    lldb::offset_t offset_to_threads = ThreadContextOffset;
    for (int i = 0; i < NumberOfThreads; i++) {
        offset_to_threads = ThreadContextOffset + ((sizeof(mstsave32) +
                                                    sizeof(thrdsinfo64)) * i);
        LLDB_LOGF(log, "Multi-threaded parsing, offset %x\n", offset_to_threads);
        AIXCore32Header temp_header;
        ThreadContext32 thread;
        temp_header.ParseThreadContext(data, &offset_to_threads);
        memcpy(&thread, &temp_header.Fault, sizeof(ThreadContext64));
        threads.push_back(std::move(thread));
    }

    return true;
}

bool AIXCore64Header::ParseRegisterContext(lldb_private::DataExtractor &data,
                lldb::offset_t *offset) {
    // The data is arranged in this order in this coredump file
    // so we have to fetch in this exact order. But need to change
    // the context structure order according to Infos_ppc64
    for(int i = 0; i < 32; i++)
        Fault.context.gpr[i] = data.GetU64(offset);
    Fault.context.msr = data.GetU64(offset); 
    Fault.context.pc = data.GetU64(offset); 
    Fault.context.lr = data.GetU64(offset); 
    Fault.context.ctr = data.GetU64(offset); 
    Fault.context.cr = data.GetU32(offset); 
    Fault.context.xer = data.GetU32(offset); 
    Fault.context.fpscr = data.GetU32(offset); 
    Fault.context.fpscrx = data.GetU32(offset); 
    Fault.context.except[0] = data.GetU64(offset); 
    for(int i = 0; i < 32; i++)
        Fault.context.fpr[i] = data.GetU64(offset);
    Fault.context.fpeu = data.GetU8(offset); 
    Fault.context.fpinfo = data.GetU8(offset); 
    Fault.context.fpscr24_31 = data.GetU8(offset); 
    Fault.context.pad[0] = data.GetU8(offset); 
    Fault.context.excp_type = data.GetU32(offset); 

    return true;
}
bool AIXCore64Header::ParseThreadContext(lldb_private::DataExtractor &data,
                lldb::offset_t *offset) {
    lldb::offset_t offset_to_regctx = *offset; 
    offset_to_regctx += sizeof(thrdentry64);
    Fault.thread.ti_tid = data.GetU64(offset);
    Fault.thread.ti_pid = data.GetU32(offset);
    int ret = ParseRegisterContext(data, &offset_to_regctx);
    return true;
}
 
bool AIXCore64Header::ParseUserData(lldb_private::DataExtractor &data,
                lldb::offset_t *offset) {
    User.process.pi_pid = data.GetU32(offset); 
    User.process.pi_ppid = data.GetU32(offset); 
    User.process.pi_sid = data.GetU32(offset); 
    User.process.pi_pgrp = data.GetU32(offset); 
    User.process.pi_uid = data.GetU32(offset); 
    User.process.pi_suid = data.GetU32(offset); 

    *offset += 76;

    ByteOrder byteorder = data.GetByteOrder();
    size_t size = 33;
    data.ExtractBytes(*offset, size, byteorder, User.process.pi_comm);
    offset += size;

    return true;
}

bool AIXCore64Header::ParseCoreHeader(lldb_private::DataExtractor &data,
                            lldb::offset_t *offset) {
    Log *log = GetLog(LLDBLog::Process);

    SignalNum = data.GetU8(offset);  
    Flag = data.GetU8(offset);  
    Entries = data.GetU16(offset);  
    Version = data.GetU32(offset);
    FDInfo = data.GetU64(offset);

    LoaderOffset = data.GetU64(offset);
    LoaderSize = data.GetU64(offset);
    NumberOfThreads = data.GetU32(offset);
    Reserved0 = data.GetU32(offset);
    ThreadContextOffset = data.GetU64(offset);
    NumSegRegion = data.GetU64(offset);
    SegRegionOffset = data.GetU64(offset);
    StackOffset = data.GetU64(offset);
    StackBaseAddr = data.GetU64(offset);
    StackSize = data.GetU64(offset);
    DataRegionOffset = data.GetU64(offset);
    DataBaseAddr = data.GetU64(offset);
    DataSize = data.GetU64(offset);

    *offset += 104;
    // This offset calculation is due to the difference between
    // AIX register size and LLDB register variables order.
    // __context64 does not match RegContext
    // To be modified
    lldb::offset_t offset_to_user = (*offset + sizeof(__context64) +
            sizeof(thrdentry64));
    int ret = 0;
    ret = ParseThreadContext(data, offset);
    ret = ParseUserData(data, &offset_to_user);

    lldb::offset_t offset_to_threads = ThreadContextOffset;
    for (int i = 0; i < NumberOfThreads; i++) {
        offset_to_threads = ThreadContextOffset + ((sizeof(__context64) +
                                                    sizeof(thrdentry64)) * i);
        LLDB_LOGF(log, "Multi-threaded parsing, offset %x\n", offset_to_threads);
        AIXCore64Header temp_header;
        ThreadContext64 thread;
        temp_header.ParseThreadContext(data, &offset_to_threads);
        memcpy(&thread, &temp_header.Fault, sizeof(ThreadContext64));
        threads.push_back(std::move(thread));
    }

    return ret;

}
        
