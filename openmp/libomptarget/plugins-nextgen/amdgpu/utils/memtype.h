#ifndef __MEMTYPE_H__
#define __MEMTYPE_H__

#include <cstdint>
#include <map>
#include <math.h>

// uncomment to disable assert()
// #define NDEBUG
#include <cassert>

// Virtual memory configuration on Linux x86_64
// for AMDGPU based systems
namespace AMDGPU_X86_64_SystemConfiguration {
const uint64_t max_addressable_byte = 0x00007fffffffffff;
// 4KB
const uint64_t page_size = 4 * 1024;
} // namespace AMDGPU_X86_64_SystemConfiguration

// Bit field table to track single memory page type
class AMDGPUMemTypeBitFieldTable {
private:
  // set \arg idx bit to 1
  inline void set(uint64_t &tab_loc, const uint64_t idx) {
    tab_loc |= 1UL << idx;
  }

  // test if \arg idx bit is set to 1
  inline bool isSet(const uint64_t tab_loc, const uint64_t idx) const {
    return ((1UL << idx) == (tab_loc & (1UL << idx)));
  }

  // return table index for page pointed to by \arg ptr
  inline uint64_t calc_page_index(uintptr_t ptr) const {
    return ptr >> log2page_size;
  }

public:
  AMDGPUMemTypeBitFieldTable(uint64_t mem_size, uint64_t page_size) {
    assert(mem_size % page_size == 0);
    num_pages = mem_size / page_size;
    log2page_size = log2l(page_size);

    log2_pages_per_block = log2l(pages_per_block);
    assert((num_pages % 2) == 0);
    uint64_t tab_size = num_pages >> log2_pages_per_block;
    tab = (uint64_t *)calloc(tab_size, sizeof(uint64_t));
  }

  // Set all pages touched by address in the range [base, base+size-1]
  // \arg base : pointer to first byte of the memory area whose
  // type should become of the tracked type
  // \arg size : size in bytes of the memory area whose type
  // should become of the tracked type
  // \ret if any of the pages was already set
  inline bool insert(const uintptr_t base, size_t size) {
    uint64_t page_start = calc_page_index(base);
    uint64_t page_end = calc_page_index(base + size - 1);
    uint64_t blockId = page_start >> log2_pages_per_block;
    uint64_t blockOffset = page_start & (pages_per_block - 1);
    for (uint64_t i = page_start; i <= page_end; i++) {
      blockId = i >> log2_pages_per_block;
      blockOffset = i & (pages_per_block - 1);
      set(tab[blockId], blockOffset);
    }
    return false;
  }

  // Test if all pages in the range [base, base+size-1]
  // are of the tracked memory type.
  // \arg base : pointer to first byte of the memory area whose
  // type should become of the tracked type
  // \arg size : number of bytes of the memory area whose type
  // should become of the tracked type
  // \ret true if any of the pages was set; false otherwise
  bool contains(const uintptr_t base, size_t size) const {
    uint64_t page_start = calc_page_index(base);
    uint64_t page_end = calc_page_index(base + size - 1);
    for (uint64_t i = page_start; i <= page_end; i++) {
      uint64_t blockId = i >> log2_pages_per_block;
      uint64_t blockOffset = i & (pages_per_block - 1);
      if (!isSet(tab[blockId], blockOffset))
        return false;
    }
    return true;
  }

private:
  uint64_t num_pages;

  // leading zero's for page size
  // used to calculate index in table
  uint64_t log2page_size;

  // number of pages tracked in a single table entry
  // (uint64_t: one bit per page)
  const int pages_per_block = 64;
  int log2_pages_per_block;

  // the actual table that given a page index
  // contains whether the page belongs to the tracked
  // memory type. For any bit:
  // 0 = page is *not* of tracked type
  // 1 = page is of tracked type
  uint64_t *tab;
};

#endif //__MEMTYPE_H__
