/*===--------------------------------------------------------------------------
 *              ATMI (Asynchronous Task and Memory Interface)
 *
 * This file is distributed under the MIT License. See LICENSE.txt for details.
 *===------------------------------------------------------------------------*/
#ifndef SRC_RUNTIME_INCLUDE_DATA_H_
#define SRC_RUNTIME_INCLUDE_DATA_H_
#include "atmi.h"
#include <hsa.h>
#include <map>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
// we maintain our own mapping of device addr to a user specified data object
// in order to work around a (possibly historic) bug in ROCr's
// hsa_amd_pointer_info_set_userdata for variable symbols
// this is expected to be temporary

namespace core {
// Internal representation of any data that is created and managed by ATMI.
// Data can be located on any device memory or host memory.
class ATLData {
public:
  ATLData(void *ptr, size_t size, atmi_mem_place_t place)
      : ptr_(ptr), size_(size), place_(place) {}

  void *ptr() const { return ptr_; }
  size_t size() const { return size_; }
  atmi_mem_place_t place() const { return place_; }

private:
  void *ptr_;
  size_t size_;
  atmi_mem_place_t place_;
};

//---
struct ATLMemoryRange {
  const void *base_pointer;
  const void *end_pointer;
  ATLMemoryRange(const void *bp, size_t size_bytes)
      : base_pointer(bp),
        end_pointer(reinterpret_cast<const unsigned char *>(bp) + size_bytes -
                    1) {}
};

// Functor to compare ranges:
struct ATLMemoryRangeCompare {
  // Return true is LHS range is less than RHS - used to order the ranges
  bool operator()(const ATLMemoryRange &lhs, const ATLMemoryRange &rhs) const {
    return lhs.end_pointer < rhs.base_pointer;
  }
};

//-------------------------------------------------------------------------------------------------
// This structure tracks information for each pointer.
// Uses memory-range-based lookups - so pointers that exist anywhere in the
// range of hostPtr + size
// will find the associated ATLPointerInfo.
// The insertions and lookups use a self-balancing binary tree and should
// support O(logN) lookup speed.
// The structure is thread-safe - writers obtain a mutex before modifying the
// tree.  Multiple simulatenous readers are supported.
class ATLPointerTracker {
  typedef std::map<ATLMemoryRange, ATLData *, ATLMemoryRangeCompare>
      MapTrackerType;

public:
  void insert(void *pointer, ATLData *data);
  void remove(void *pointer);
  ATLData *find(const void *pointer);

private:
  MapTrackerType tracker_;
  std::mutex mutex_;
};

extern ATLPointerTracker g_data_map; // Track all am pointer allocations.

enum class Direction { ATMI_H2D, ATMI_D2H, ATMI_D2D, ATMI_H2H };

} // namespace core
#endif // SRC_RUNTIME_INCLUDE_DATA_H_
