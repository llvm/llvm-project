//===-- MachVMMemory.cpp ----------------------------------------*- C++ -*-===//
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

#include "MachVMMemory.h"
#include "DNBLog.h"
#include "MachVMRegion.h"
#include <dlfcn.h>
#include <mach/mach_vm.h>
#include <mach/shared_region.h>
#include <sys/sysctl.h>

static const vm_size_t kInvalidPageSize = ~0;

MachVMMemory::MachVMMemory() : m_page_size(kInvalidPageSize), m_err(0) {}

MachVMMemory::~MachVMMemory() {}

nub_size_t MachVMMemory::PageSize(task_t task) {
  if (m_page_size == kInvalidPageSize) {
#if defined(TASK_VM_INFO) && TASK_VM_INFO >= 22
    if (task != TASK_NULL) {
      kern_return_t kr;
      mach_msg_type_number_t info_count = TASK_VM_INFO_COUNT;
      task_vm_info_data_t vm_info;
      kr = task_info(task, TASK_VM_INFO, (task_info_t)&vm_info, &info_count);
      if (kr == KERN_SUCCESS) {
        DNBLogThreadedIf(
            LOG_TASK,
            "MachVMMemory::PageSize task_info returned page size of 0x%x",
            (int)vm_info.page_size);
        m_page_size = vm_info.page_size;
        return m_page_size;
      } else {
        DNBLogThreadedIf(LOG_TASK, "MachVMMemory::PageSize task_info call "
                                   "failed to get page size, TASK_VM_INFO %d, "
                                   "TASK_VM_INFO_COUNT %d, kern return %d",
                         TASK_VM_INFO, TASK_VM_INFO_COUNT, kr);
      }
    }
#endif
    m_err = ::host_page_size(::mach_host_self(), &m_page_size);
    if (m_err.Fail())
      m_page_size = 0;
  }
  return m_page_size;
}

nub_size_t MachVMMemory::MaxBytesLeftInPage(task_t task, nub_addr_t addr,
                                            nub_size_t count) {
  const nub_size_t page_size = PageSize(task);
  if (page_size > 0) {
    nub_size_t page_offset = (addr % page_size);
    nub_size_t bytes_left_in_page = page_size - page_offset;
    if (count > bytes_left_in_page)
      count = bytes_left_in_page;
  }
  return count;
}

nub_bool_t MachVMMemory::GetMemoryRegionInfo(task_t task, nub_addr_t address,
                                             DNBRegionInfo *region_info) {
  MachVMRegion vmRegion(task);

  if (vmRegion.GetRegionForAddress(address)) {
    region_info->addr = vmRegion.StartAddress();
    region_info->size = vmRegion.GetByteSize();
    region_info->permissions = vmRegion.GetDNBPermissions();
  } else {
    region_info->addr = address;
    region_info->size = 0;
    if (vmRegion.GetError().Success()) {
      // vmRegion.GetRegionForAddress() return false, indicating that "address"
      // wasn't in a valid region, but the "vmRegion" info was successfully
      // read from the task which means the info describes the next valid
      // region from which we can infer the size of this invalid region
      mach_vm_address_t start_addr = vmRegion.StartAddress();
      if (address < start_addr)
        region_info->size = start_addr - address;
    }
    // If we can't get any info about the size from the next region it means
    // we asked about an address that was past all mappings, so the size
    // of this region will take up all remaining address space.
    if (region_info->size == 0)
      region_info->size = INVALID_NUB_ADDRESS - region_info->addr;

    // Not readable, writeable or executable
    region_info->permissions = 0;
  }
  return true;
}

// For integrated graphics chip, this makes the accounting info for 'wired'
// memory more like top.
uint64_t MachVMMemory::GetStolenPages(task_t task) {
  static uint64_t stolenPages = 0;
  static bool calculated = false;
  if (calculated)
    return stolenPages;

  static int mib_reserved[CTL_MAXNAME];
  static int mib_unusable[CTL_MAXNAME];
  static int mib_other[CTL_MAXNAME];
  static size_t mib_reserved_len = 0;
  static size_t mib_unusable_len = 0;
  static size_t mib_other_len = 0;
  int r;

  /* This can be used for testing: */
  // tsamp->pages_stolen = (256 * 1024 * 1024ULL) / tsamp->pagesize;

  if (0 == mib_reserved_len) {
    mib_reserved_len = CTL_MAXNAME;

    r = sysctlnametomib("machdep.memmap.Reserved", mib_reserved,
                        &mib_reserved_len);

    if (-1 == r) {
      mib_reserved_len = 0;
      return 0;
    }

    mib_unusable_len = CTL_MAXNAME;

    r = sysctlnametomib("machdep.memmap.Unusable", mib_unusable,
                        &mib_unusable_len);

    if (-1 == r) {
      mib_reserved_len = 0;
      return 0;
    }

    mib_other_len = CTL_MAXNAME;

    r = sysctlnametomib("machdep.memmap.Other", mib_other, &mib_other_len);

    if (-1 == r) {
      mib_reserved_len = 0;
      return 0;
    }
  }

  if (mib_reserved_len > 0 && mib_unusable_len > 0 && mib_other_len > 0) {
    uint64_t reserved = 0, unusable = 0, other = 0;
    size_t reserved_len;
    size_t unusable_len;
    size_t other_len;

    reserved_len = sizeof(reserved);
    unusable_len = sizeof(unusable);
    other_len = sizeof(other);

    /* These are all declared as QUAD/uint64_t sysctls in the kernel. */

    if (sysctl(mib_reserved, static_cast<u_int>(mib_reserved_len), &reserved,
               &reserved_len, NULL, 0)) {
      return 0;
    }

    if (sysctl(mib_unusable, static_cast<u_int>(mib_unusable_len), &unusable,
               &unusable_len, NULL, 0)) {
      return 0;
    }

    if (sysctl(mib_other, static_cast<u_int>(mib_other_len), &other, &other_len,
               NULL, 0)) {
      return 0;
    }

    if (reserved_len == sizeof(reserved) && unusable_len == sizeof(unusable) &&
        other_len == sizeof(other)) {
      uint64_t stolen = reserved + unusable + other;
      uint64_t mb128 = 128 * 1024 * 1024ULL;

      if (stolen >= mb128) {
        stolen = (stolen & ~((128 * 1024 * 1024ULL) - 1)); // rounding down
        stolenPages = stolen / PageSize(task);
      }
    }
  }

  calculated = true;
  return stolenPages;
}

static uint64_t GetPhysicalMemory() {
  // This doesn't change often at all. No need to poll each time.
  static uint64_t physical_memory = 0;
  static bool calculated = false;
  if (calculated)
    return physical_memory;

  size_t len = sizeof(physical_memory);
  sysctlbyname("hw.memsize", &physical_memory, &len, NULL, 0);

  calculated = true;
  return physical_memory;
}

// rsize and dirty_size is not adjusted for dyld shared cache and multiple
// __LINKEDIT segment, as in vmmap. In practice, dirty_size doesn't differ much
// but rsize may. There is performance penalty for the adjustment. Right now,
// only use the dirty_size.
void MachVMMemory::GetRegionSizes(task_t task, mach_vm_size_t &rsize,
                                  mach_vm_size_t &dirty_size) {
#if defined(TASK_VM_INFO) && TASK_VM_INFO >= 22

  task_vm_info_data_t vm_info;
  mach_msg_type_number_t info_count;
  kern_return_t kr;

  info_count = TASK_VM_INFO_COUNT;
  kr = task_info(task, TASK_VM_INFO_PURGEABLE, (task_info_t)&vm_info,
                 &info_count);
  if (kr == KERN_SUCCESS)
    dirty_size = vm_info.internal;
#endif
}

// Test whether the virtual address is within the architecture's shared region.
static bool InSharedRegion(mach_vm_address_t addr, cpu_type_t type) {
  mach_vm_address_t base = 0, size = 0;

  switch (type) {
#if defined(CPU_TYPE_ARM64) && defined(SHARED_REGION_BASE_ARM64)
  case CPU_TYPE_ARM64:
    base = SHARED_REGION_BASE_ARM64;
    size = SHARED_REGION_SIZE_ARM64;
    break;
#endif

  case CPU_TYPE_ARM:
    base = SHARED_REGION_BASE_ARM;
    size = SHARED_REGION_SIZE_ARM;
    break;

  case CPU_TYPE_X86_64:
    base = SHARED_REGION_BASE_X86_64;
    size = SHARED_REGION_SIZE_X86_64;
    break;

  case CPU_TYPE_I386:
    base = SHARED_REGION_BASE_I386;
    size = SHARED_REGION_SIZE_I386;
    break;

  default: {
    // Log error abut unknown CPU type
    break;
  }
  }

  return (addr >= base && addr < (base + size));
}

void MachVMMemory::GetMemorySizes(task_t task, cpu_type_t cputype,
                                  nub_process_t pid, mach_vm_size_t &rprvt,
                                  mach_vm_size_t &vprvt) {
  // Collecting some other info cheaply but not reporting for now.
  mach_vm_size_t empty = 0;
  mach_vm_size_t fw_private = 0;

  mach_vm_size_t aliased = 0;
  bool global_shared_text_data_mapped = false;
  vm_size_t pagesize = PageSize(task);

  for (mach_vm_address_t addr = 0, size = 0;; addr += size) {
    vm_region_top_info_data_t info;
    mach_msg_type_number_t count = VM_REGION_TOP_INFO_COUNT;
    mach_port_t object_name;

    kern_return_t kr =
        mach_vm_region(task, &addr, &size, VM_REGION_TOP_INFO,
                       (vm_region_info_t)&info, &count, &object_name);
    if (kr != KERN_SUCCESS)
      break;

    if (InSharedRegion(addr, cputype)) {
      // Private Shared
      fw_private += info.private_pages_resident * pagesize;

      // Check if this process has the globally shared text and data regions
      // mapped in.  If so, set global_shared_text_data_mapped to TRUE and avoid
      // checking again.
      if (global_shared_text_data_mapped == FALSE &&
          info.share_mode == SM_EMPTY) {
        vm_region_basic_info_data_64_t b_info;
        mach_vm_address_t b_addr = addr;
        mach_vm_size_t b_size = size;
        count = VM_REGION_BASIC_INFO_COUNT_64;

        kr = mach_vm_region(task, &b_addr, &b_size, VM_REGION_BASIC_INFO,
                            (vm_region_info_t)&b_info, &count, &object_name);
        if (kr != KERN_SUCCESS)
          break;

        if (b_info.reserved) {
          global_shared_text_data_mapped = TRUE;
        }
      }

      // Short circuit the loop if this isn't a shared private region, since
      // that's the only region type we care about within the current address
      // range.
      if (info.share_mode != SM_PRIVATE) {
        continue;
      }
    }

    // Update counters according to the region type.
    if (info.share_mode == SM_COW && info.ref_count == 1) {
      // Treat single reference SM_COW as SM_PRIVATE
      info.share_mode = SM_PRIVATE;
    }

    switch (info.share_mode) {
    case SM_LARGE_PAGE:
    // Treat SM_LARGE_PAGE the same as SM_PRIVATE
    // since they are not shareable and are wired.
    case SM_PRIVATE:
      rprvt += info.private_pages_resident * pagesize;
      rprvt += info.shared_pages_resident * pagesize;
      vprvt += size;
      break;

    case SM_EMPTY:
      empty += size;
      break;

    case SM_COW:
    case SM_SHARED: {
      if (pid == 0) {
        // Treat kernel_task specially
        if (info.share_mode == SM_COW) {
          rprvt += info.private_pages_resident * pagesize;
          vprvt += size;
        }
        break;
      }

      if (info.share_mode == SM_COW) {
        rprvt += info.private_pages_resident * pagesize;
        vprvt += info.private_pages_resident * pagesize;
      }
      break;
    }
    default:
      // log that something is really bad.
      break;
    }
  }

  rprvt += aliased;
}

static void GetPurgeableAndAnonymous(task_t task, uint64_t &purgeable,
                                     uint64_t &anonymous) {
#if defined(TASK_VM_INFO) && TASK_VM_INFO >= 22

  kern_return_t kr;
  mach_msg_type_number_t info_count;
  task_vm_info_data_t vm_info;

  info_count = TASK_VM_INFO_COUNT;
  kr = task_info(task, TASK_VM_INFO_PURGEABLE, (task_info_t)&vm_info,
                 &info_count);
  if (kr == KERN_SUCCESS) {
    purgeable = vm_info.purgeable_volatile_resident;
    anonymous =
        vm_info.internal + vm_info.compressed - vm_info.purgeable_volatile_pmap;
  }

#endif
}

#if defined(HOST_VM_INFO64_COUNT)
nub_bool_t MachVMMemory::GetMemoryProfile(
    DNBProfileDataScanType scanType, task_t task, struct task_basic_info ti,
    cpu_type_t cputype, nub_process_t pid, vm_statistics64_data_t &vminfo,
    uint64_t &physical_memory, mach_vm_size_t &rprvt, mach_vm_size_t &rsize,
    mach_vm_size_t &vprvt, mach_vm_size_t &vsize, mach_vm_size_t &dirty_size,
    mach_vm_size_t &purgeable, mach_vm_size_t &anonymous)
#else
nub_bool_t MachVMMemory::GetMemoryProfile(
    DNBProfileDataScanType scanType, task_t task, struct task_basic_info ti,
    cpu_type_t cputype, nub_process_t pid, vm_statistics_data_t &vminfo,
    uint64_t &physical_memory, mach_vm_size_t &rprvt, mach_vm_size_t &rsize,
    mach_vm_size_t &vprvt, mach_vm_size_t &vsize, mach_vm_size_t &dirty_size,
    mach_vm_size_t &purgeable, mach_vm_size_t &anonymous)
#endif
{
  if (scanType & eProfileHostMemory)
    physical_memory = GetPhysicalMemory();

  if (scanType & eProfileMemory) {
    static mach_port_t localHost = mach_host_self();
#if defined(HOST_VM_INFO64_COUNT)
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    host_statistics64(localHost, HOST_VM_INFO64, (host_info64_t)&vminfo,
                      &count);
#else
    mach_msg_type_number_t count = HOST_VM_INFO_COUNT;
    host_statistics(localHost, HOST_VM_INFO, (host_info_t)&vminfo, &count);
    vminfo.wire_count += GetStolenPages(task);
#endif

    /* We are no longer reporting these. Let's not waste time.
    GetMemorySizes(task, cputype, pid, rprvt, vprvt);
    rsize = ti.resident_size;
    vsize = ti.virtual_size;

    if (scanType & eProfileMemoryDirtyPage)
    {
        // This uses vmmap strategy. We don't use the returned rsize for now. We
    prefer to match top's version since that's what we do for the rest of the
    metrics.
        GetRegionSizes(task, rsize, dirty_size);
    }
    */

    if (scanType & eProfileMemoryAnonymous) {
      GetPurgeableAndAnonymous(task, purgeable, anonymous);
    }
  }

  return true;
}

nub_size_t MachVMMemory::Read(task_t task, nub_addr_t address, void *data,
                              nub_size_t data_count) {
  if (data == NULL || data_count == 0)
    return 0;

  nub_size_t total_bytes_read = 0;
  nub_addr_t curr_addr = address;
  uint8_t *curr_data = (uint8_t *)data;
  while (total_bytes_read < data_count) {
    mach_vm_size_t curr_size =
        MaxBytesLeftInPage(task, curr_addr, data_count - total_bytes_read);
    mach_msg_type_number_t curr_bytes_read = 0;
    vm_offset_t vm_memory = 0;
    m_err = ::mach_vm_read(task, curr_addr, curr_size, &vm_memory,
                           &curr_bytes_read);

    if (DNBLogCheckLogBit(LOG_MEMORY))
      m_err.LogThreaded("::mach_vm_read ( task = 0x%4.4x, addr = 0x%8.8llx, "
                        "size = %llu, data => %8.8p, dataCnt => %i )",
                        task, (uint64_t)curr_addr, (uint64_t)curr_size,
                        vm_memory, curr_bytes_read);

    if (m_err.Success()) {
      if (curr_bytes_read != curr_size) {
        if (DNBLogCheckLogBit(LOG_MEMORY))
          m_err.LogThreaded(
              "::mach_vm_read ( task = 0x%4.4x, addr = 0x%8.8llx, size = %llu, "
              "data => %8.8p, dataCnt=>%i ) only read %u of %llu bytes",
              task, (uint64_t)curr_addr, (uint64_t)curr_size, vm_memory,
              curr_bytes_read, curr_bytes_read, (uint64_t)curr_size);
      }
      ::memcpy(curr_data, (void *)vm_memory, curr_bytes_read);
      ::vm_deallocate(mach_task_self(), vm_memory, curr_bytes_read);
      total_bytes_read += curr_bytes_read;
      curr_addr += curr_bytes_read;
      curr_data += curr_bytes_read;
    } else {
      break;
    }
  }
  return total_bytes_read;
}

nub_size_t MachVMMemory::Write(task_t task, nub_addr_t address,
                               const void *data, nub_size_t data_count) {
  MachVMRegion vmRegion(task);

  nub_size_t total_bytes_written = 0;
  nub_addr_t curr_addr = address;
  const uint8_t *curr_data = (const uint8_t *)data;

  while (total_bytes_written < data_count) {
    if (vmRegion.GetRegionForAddress(curr_addr)) {
      mach_vm_size_t curr_data_count = data_count - total_bytes_written;
      mach_vm_size_t region_bytes_left = vmRegion.BytesRemaining(curr_addr);
      if (region_bytes_left == 0) {
        break;
      }
      if (curr_data_count > region_bytes_left)
        curr_data_count = region_bytes_left;

      if (vmRegion.SetProtections(curr_addr, curr_data_count,
                                  VM_PROT_READ | VM_PROT_WRITE)) {
        nub_size_t bytes_written =
            WriteRegion(task, curr_addr, curr_data, curr_data_count);
        if (bytes_written <= 0) {
          // Status should have already be posted by WriteRegion...
          break;
        } else {
          total_bytes_written += bytes_written;
          curr_addr += bytes_written;
          curr_data += bytes_written;
        }
      } else {
        DNBLogThreadedIf(
            LOG_MEMORY_PROTECTIONS, "Failed to set read/write protections on "
                                    "region for address: [0x%8.8llx-0x%8.8llx)",
            (uint64_t)curr_addr, (uint64_t)(curr_addr + curr_data_count));
        break;
      }
    } else {
      DNBLogThreadedIf(LOG_MEMORY_PROTECTIONS,
                       "Failed to get region for address: 0x%8.8llx",
                       (uint64_t)address);
      break;
    }
  }

  return total_bytes_written;
}

nub_size_t MachVMMemory::WriteRegion(task_t task, const nub_addr_t address,
                                     const void *data,
                                     const nub_size_t data_count) {
  if (data == NULL || data_count == 0)
    return 0;

  nub_size_t total_bytes_written = 0;
  nub_addr_t curr_addr = address;
  const uint8_t *curr_data = (const uint8_t *)data;
  while (total_bytes_written < data_count) {
    mach_msg_type_number_t curr_data_count =
        static_cast<mach_msg_type_number_t>(MaxBytesLeftInPage(
            task, curr_addr, data_count - total_bytes_written));
    m_err =
        ::mach_vm_write(task, curr_addr, (pointer_t)curr_data, curr_data_count);
    if (DNBLogCheckLogBit(LOG_MEMORY) || m_err.Fail())
      m_err.LogThreaded("::mach_vm_write ( task = 0x%4.4x, addr = 0x%8.8llx, "
                        "data = %8.8p, dataCnt = %u )",
                        task, (uint64_t)curr_addr, curr_data, curr_data_count);

#if !defined(__i386__) && !defined(__x86_64__)
    vm_machine_attribute_val_t mattr_value = MATTR_VAL_CACHE_FLUSH;

    m_err = ::vm_machine_attribute(task, curr_addr, curr_data_count,
                                   MATTR_CACHE, &mattr_value);
    if (DNBLogCheckLogBit(LOG_MEMORY) || m_err.Fail())
      m_err.LogThreaded("::vm_machine_attribute ( task = 0x%4.4x, addr = "
                        "0x%8.8llx, size = %u, attr = MATTR_CACHE, mattr_value "
                        "=> MATTR_VAL_CACHE_FLUSH )",
                        task, (uint64_t)curr_addr, curr_data_count);
#endif

    if (m_err.Success()) {
      total_bytes_written += curr_data_count;
      curr_addr += curr_data_count;
      curr_data += curr_data_count;
    } else {
      break;
    }
  }
  return total_bytes_written;
}
