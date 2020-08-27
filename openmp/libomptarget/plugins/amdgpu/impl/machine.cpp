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
extern hsa_region_t atl_cpu_kernarg_region;

void *ATLMemory::alloc(size_t sz) {
  void *ret;
  hsa_status_t err = hsa_amd_memory_pool_allocate(memory_pool_, sz, 0, &ret);
  ErrorCheck(Allocate from memory pool, err);
  return ret;
}

void ATLMemory::free(void *ptr) {
  hsa_status_t err = hsa_amd_memory_pool_free(ptr);
  ErrorCheck(Allocate from memory pool, err);
}

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

int cu_mask_parser(char *gpu_workers, uint64_t *cu_masks, int count) {
  int cu_mask_enable = 0;

  if (gpu_workers) {
    char *pch, *token;

    // skip num_of_workers
    token = strtok_r(gpu_workers, ":", &pch);
    // printf("num_queues: %s\n", token);

    int qid = 0;
    token = strtok_r(NULL, ";", &pch);

    // parse each queue
    while (token != NULL && qid < count) {
      // printf("qid: %d %s\n", qid, pch);
      char *pch2, *token2;
      cu_mask_enable = 1;
      token2 = strtok_r(token, ",", &pch2);
      // fprintf(stderr, "qid: %d cu:", qid);
      while (token2 != NULL) {
        char *pch3, *token3;
        token3 = strtok_r(token2, "-", &pch3);
        int offset = atoi(token3);
        token3 = strtok_r(NULL, "-", &pch3);
        int num_cus = token3 ? atoi(token3) - offset + 1 : 1;
        token2 = strtok_r(NULL, ",", &pch2);

        // fprintf(stderr, "%d-%d ", offset, num_cus);

        for (int i = 0; i < num_cus; i++) {
          cu_masks[qid] |= (uint64_t)1 << offset;
          offset++;
        }
      }

      // fprintf(stderr, "mask: %lx\n", cu_masks[qid]);

      token = strtok_r(NULL, ";", &pch);
      qid++;
    }
  }

  return cu_mask_enable;
}

void callbackQueue(hsa_status_t status, hsa_queue_t *source, void *data) {
  if (status != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "[%s:%d] GPU error in queue %p %d\n", __FILE__, __LINE__,
            source, status);
    abort();
  }
}

void ATLGPUProcessor::createQueues(const int count) {
  char *gpu_workers = getenv("ATMI_DEVICE_GPU_WORKERS");

  int *num_cus = reinterpret_cast<int *>(calloc(count, sizeof(int)));
  uint64_t *cu_masks =
      reinterpret_cast<uint64_t *>(calloc(count, sizeof(uint64_t)));

  int cu_mask_enable = 0;

  if (gpu_workers)
    cu_mask_enable = cu_mask_parser(gpu_workers, cu_masks, count);

  hsa_status_t err;
  /* Query the maximum size of the queue.  */
  uint32_t queue_size = 0;
  err = hsa_agent_get_info(agent_, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
  ErrorCheck(Querying the agent maximum queue size, err);
  if (queue_size > core::Runtime::getInstance().getMaxQueueSize()) {
    queue_size = core::Runtime::getInstance().getMaxQueueSize();
  }
  /* printf("The maximum queue size is %u.\n", (unsigned int) queue_size);  */

  /* Create queues for each device. */
  int qid;
  for (qid = 0; qid < count; qid++) {
    hsa_queue_t *this_Q;
    err =
        hsa_queue_create(agent_, queue_size, HSA_QUEUE_TYPE_MULTI,
                         callbackQueue, NULL, UINT32_MAX, UINT32_MAX, &this_Q);
    ErrorCheck(Creating the queue, err);
    err = hsa_amd_profiling_set_profiler_enabled(this_Q, 1);
    ErrorCheck(Enabling profiling support, err);

    if (cu_mask_enable) {
      if (!cu_masks[qid]) {
        cu_masks[qid] = -1;
        fprintf(stderr, "Warning: queue[%d]: cu mask is 0x0\n", qid);
      }

      uint32_t *this_cu_mask_v = reinterpret_cast<uint32_t *>(&cu_masks[qid]);
      hsa_status_t ret = hsa_amd_queue_cu_set_mask(this_Q, 64, this_cu_mask_v);

      if (ret != HSA_STATUS_SUCCESS)
        fprintf(stderr, "Error: hsa_amd_queue_cu_set_mask\n");
    }

    queues_.push_back(this_Q);

    DEBUG_PRINT("Queue[%d]: %p\n", qid, this_Q);
  }

  free(cu_masks);
  free(num_cus);
}

void ATLCPUProcessor::createQueues(const int) {}

void ATLProcessor::destroyQueues() {
  for (auto queue : queues_) {
    hsa_status_t err = hsa_queue_destroy(queue);
    ErrorCheck(Destroying the queue, err);
  }
}

int ATLProcessor::num_cus() const {
  hsa_status_t err;
  /* Query the number of compute units.  */
  uint32_t num_cus = 0;
  err = hsa_agent_get_info(
      agent_, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT,
      &num_cus);
  ErrorCheck(Querying the agent number of compute units, err);

  return num_cus;
}
