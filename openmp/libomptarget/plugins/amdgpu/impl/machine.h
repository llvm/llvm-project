/*===--------------------------------------------------------------------------
 *              ATMI (Asynchronous Task and Memory Interface)
 *
 * This file is distributed under the MIT License. See LICENSE.txt for details.
 *===------------------------------------------------------------------------*/
#ifndef SRC_RUNTIME_INCLUDE_MACHINE_H_
#define SRC_RUNTIME_INCLUDE_MACHINE_H_
#include "atmi.h"
#include "internal.h"
#include <hsa.h>
#include <hsa_ext_amd.h>
#include <vector>

class ATLMemory;

class ATLProcessor {
public:
  explicit ATLProcessor(hsa_agent_t agent,
                        atmi_devtype_t type = ATMI_DEVTYPE_ALL)
      : agent_(agent), type_(type) {
    memories_.clear();
  }
  void addMemory(const ATLMemory &p);
  hsa_agent_t agent() const { return agent_; }
  const std::vector<ATLMemory> &memories() const;
  atmi_devtype_t type() const { return type_; }

protected:
  hsa_agent_t agent_;
  atmi_devtype_t type_;
  std::vector<ATLMemory> memories_;
};

class ATLCPUProcessor : public ATLProcessor {
public:
  explicit ATLCPUProcessor(hsa_agent_t agent)
      : ATLProcessor(agent, ATMI_DEVTYPE_CPU) {}
};

class ATLGPUProcessor : public ATLProcessor {
public:
  explicit ATLGPUProcessor(hsa_agent_t agent,
                           atmi_devtype_t type = ATMI_DEVTYPE_dGPU)
      : ATLProcessor(agent, type) {}
};

class ATLMemory {
public:
  ATLMemory(hsa_amd_memory_pool_t pool, ATLProcessor p, atmi_memtype_t t)
      : memory_pool_(pool), processor_(p), type_(t) {}
  hsa_amd_memory_pool_t memory() const { return memory_pool_; }

  atmi_memtype_t type() const { return type_; }

private:
  hsa_amd_memory_pool_t memory_pool_;
  ATLProcessor processor_;
  atmi_memtype_t type_;
};

class ATLMachine {
public:
  ATLMachine() {
    cpu_processors_.clear();
    gpu_processors_.clear();
  }
  template <typename T> void addProcessor(const T &p);
  template <typename T> std::vector<T> &processors();
  template <typename T> size_t processorCount() {
    return processors<T>().size();
  }

private:
  std::vector<ATLCPUProcessor> cpu_processors_;
  std::vector<ATLGPUProcessor> gpu_processors_;
};

hsa_amd_memory_pool_t get_memory_pool(const ATLProcessor &proc,
                                      const int mem_id);

extern ATLMachine g_atl_machine;
template <typename T> T &get_processor(int dev_id) {
  if (dev_id == -1) {
    // user is asking runtime to pick a device
    // best device of this type? pick 0 for now
    dev_id = 0;
  }
  return g_atl_machine.processors<T>()[dev_id];
}

#endif // SRC_RUNTIME_INCLUDE_MACHINE_H_
