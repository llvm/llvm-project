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

class MachVMMemory {
public:
  MachVMMemory();
  ~MachVMMemory();
  nub_size_t Read(task_t task, nub_addr_t address, void *data,
                  nub_size_t data_count);
  nub_size_t Write(task_t task, nub_addr_t address, const void *data,
                   nub_size_t data_count);
  nub_size_t PageSize(task_t task);
  nub_bool_t GetMemoryRegionInfo(task_t task, nub_addr_t address,
                                 DNBRegionInfo *region_info);
  nub_bool_t GetMemoryProfile(DNBProfileDataScanType scanType, task_t task,
                              struct task_basic_info ti, cpu_type_t cputype,
                              nub_process_t pid, vm_statistics64_data_t &vminfo,
                              uint64_t &physical_memory, uint64_t &anonymous,
                              uint64_t &phys_footprint, uint64_t &memory_cap);

protected:
  nub_size_t MaxBytesLeftInPage(task_t task, nub_addr_t addr, nub_size_t count);

  nub_size_t WriteRegion(task_t task, const nub_addr_t address,
                         const void *data, const nub_size_t data_count);

  vm_size_t m_page_size;
  DNBError m_err;
};

#endif //    #ifndef __MachVMMemory_h__
