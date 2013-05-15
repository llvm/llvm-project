//===-- MachVMMemory.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Created by Greg Clayton on 6/26/07.
//
//===----------------------------------------------------------------------===//

#ifndef __MachVMMemory_h__
#define __MachVMMemory_h__

#include "DNBDefs.h"
#include "DNBError.h"
#include <mach/mach.h>

class MachVMMemory
{
public:
    enum { kInvalidPageSize = ~0 };
    MachVMMemory();
    ~MachVMMemory();
    nub_size_t Read(task_t task, nub_addr_t address, void *data, nub_size_t data_count);
    nub_size_t Write(task_t task, nub_addr_t address, const void *data, nub_size_t data_count);
    nub_size_t PageSize(task_t task);
    nub_bool_t GetMemoryRegionInfo(task_t task, nub_addr_t address, DNBRegionInfo *region_info);
    nub_bool_t GetMemoryProfile(DNBProfileDataScanType scanType, task_t task, struct task_basic_info ti, cpu_type_t cputype, nub_process_t pid, vm_statistics_data_t &vm_stats, uint64_t &physical_memory, mach_vm_size_t &rprvt, mach_vm_size_t &rsize, mach_vm_size_t &vprvt, mach_vm_size_t &vsize, mach_vm_size_t &dirty_size);

protected:
    nub_size_t  MaxBytesLeftInPage(task_t task, nub_addr_t addr, nub_size_t count);

    uint64_t    GetStolenPages(task_t task);
    void        GetRegionSizes(task_t task, mach_vm_size_t &rsize, mach_vm_size_t &dirty_size);
    void        GetMemorySizes(task_t task, cpu_type_t cputype, nub_process_t pid, mach_vm_size_t &rprvt, mach_vm_size_t &vprvt);


    nub_size_t  WriteRegion(task_t task, const nub_addr_t address, const void *data, const nub_size_t data_count);

    vm_size_t   m_page_size;
    DNBError    m_err;
};


#endif //    #ifndef __MachVMMemory_h__
