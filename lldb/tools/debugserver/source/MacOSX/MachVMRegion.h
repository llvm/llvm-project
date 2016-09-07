//===-- MachVMRegion.h ------------------------------------------*- C++ -*-===//
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

#ifndef __MachVMRegion_h__
#define __MachVMRegion_h__

#include "DNBDefs.h"
#include "DNBError.h"
#include <mach/mach.h>

class MachVMRegion {
public:
  MachVMRegion(task_t task);
  ~MachVMRegion();

  void Clear();
  mach_vm_address_t StartAddress() const { return m_start; }
  mach_vm_address_t EndAddress() const { return m_start + m_size; }
  mach_vm_size_t GetByteSize() const { return m_size; }
  mach_vm_address_t BytesRemaining(mach_vm_address_t addr) const {
    if (ContainsAddress(addr))
      return m_size - (addr - m_start);
    else
      return 0;
  }
  bool ContainsAddress(mach_vm_address_t addr) const {
    return addr >= StartAddress() && addr < EndAddress();
  }

  bool SetProtections(mach_vm_address_t addr, mach_vm_size_t size,
                      vm_prot_t prot);
  bool RestoreProtections();
  bool GetRegionForAddress(nub_addr_t addr);

  uint32_t GetDNBPermissions() const;

  const DNBError &GetError() { return m_err; }

protected:
#if defined(VM_REGION_SUBMAP_SHORT_INFO_COUNT_64)
  typedef vm_region_submap_short_info_data_64_t RegionInfo;
  enum { kRegionInfoSize = VM_REGION_SUBMAP_SHORT_INFO_COUNT_64 };
#else
  typedef vm_region_submap_info_data_64_t RegionInfo;
  enum { kRegionInfoSize = VM_REGION_SUBMAP_INFO_COUNT_64 };
#endif

  task_t m_task;
  mach_vm_address_t m_addr;
  DNBError m_err;
  mach_vm_address_t m_start;
  mach_vm_size_t m_size;
  natural_t m_depth;
  RegionInfo m_data;
  vm_prot_t m_curr_protection; // The current, possibly modified protections.
                               // Original value is saved in m_data.protections.
  mach_vm_address_t
      m_protection_addr; // The start address at which protections were changed
  mach_vm_size_t
      m_protection_size; // The size of memory that had its protections changed
};

#endif // #ifndef __MachVMRegion_h__
