/*===--------------------------------------------------------------------------
 *              ATMI (Asynchronous Task and Memory Interface)
 *
 * This file is distributed under the MIT License. See LICENSE.txt for details.
 *===------------------------------------------------------------------------*/
#include "machine.h"
#include "atmi_runtime.h"
#include "internal.h"
#include <cassert>
#include <hsa.h>
#include <hsa_ext_amd.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

extern ATLMachine g_atl_machine;

void ATLProcessor::addMemory(const ATLMemory &mem) {
  for (auto &mem_obj : memories_) {
    // if the memory already exists, then just return
    if (mem.memory().handle == mem_obj.memory().handle)
      return;
  }
  memories_.push_back(mem);
}

const std::vector<ATLMemory> &ATLProcessor::memories() const {
  return memories_;
}

template <> std::vector<ATLCPUProcessor> &ATLMachine::processors() {
  return cpu_processors_;
}

template <> std::vector<ATLGPUProcessor> &ATLMachine::processors() {
  return gpu_processors_;
}

hsa_amd_memory_pool_t get_memory_pool(const ATLProcessor &proc,
                                      const int mem_id) {
  hsa_amd_memory_pool_t pool;
  const std::vector<ATLMemory> &mems = proc.memories();
  assert(mems.size() && mem_id >= 0 && mem_id < mems.size() &&
         "Invalid memory pools for this processor");
  pool = mems[mem_id].memory();
  return pool;
}

template <> void ATLMachine::addProcessor(const ATLCPUProcessor &p) {
  cpu_processors_.push_back(p);
}

template <> void ATLMachine::addProcessor(const ATLGPUProcessor &p) {
  gpu_processors_.push_back(p);
}
