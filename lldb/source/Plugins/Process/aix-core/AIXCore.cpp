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
    lldb::offset_t offset_to_user = (*offset + sizeof(ThreadContext64));
    int ret = 0;
    ret = ParseThreadContext(data, offset);
    ret = ParseUserData(data, &offset_to_user);

    return true;

}
        
