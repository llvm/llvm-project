/*===--------------------------------------------------------------------------
 *              ATMI (Asynchronous Task and Memory Interface)
 *
 * This file is distributed under the MIT License. See LICENSE.txt for details.
 *===------------------------------------------------------------------------*/

/* This file contains logic for CPU tasking in ATMI */
#include <assert.h>
#include <climits>
#include <iostream>
#include <thread>
#include "internal.h"
#include "kernel.h"
#include "machine.h"
#include "realtimer.h"
#include "task.h"

using core::ComputeTaskImpl;
using core::CPUKernelImpl;
using core::create_header;
using core::get_nanosecs;
using core::getTaskImpl;
using core::Kernel;
using core::lock;
using core::packet_store_release;
using core::TaskImpl;
using core::unlock;

extern struct timespec context_init_time;
extern atmi_machine_t g_atmi_machine;

struct pthreadComparator {
  typedef union {
    pthread_t pth;
    unsigned char b[sizeof(pthread_t)];
  } pthcmp_t;

  bool operator()(const pthread_t &left, const pthread_t &right) const {
    /*
     * Compare two pthread handles in a way that imposes a repeatable but
     * arbitrary ordering on them.
     * I.e. given the same set of pthread_t handles the ordering should be the
     * same each time but the order has no particular meaning other than that.
     * E.g.
     * the ordering does not imply the thread start sequence, or any other
     * relationship between threads.
     *
     * Return values are:
     * false: left is greater than right
     * true: left is less than or equal to right
     */
    DEBUG_PRINT("Comparing %lu %lu\n", left, right);
    int i;
    pthcmp_t L, R;
    L.pth = left;
    R.pth = right;
    for (i = 0; i < sizeof(pthread_t); i++) {
      if (L.b[i] > R.b[i]) {
        DEBUG_PRINT("False because %d > %d\n", L.b[i], R.b[i]);
        return false;
      } else if (L.b[i] < R.b[i]) {
        DEBUG_PRINT("True because %d > %d\n", L.b[i], R.b[i]);
        return true;
      }
    }
    return false;
  }
};
static std::map<pthread_t, hsa_agent_dispatch_packet_t *, pthreadComparator>
    TaskPacketMap;
static pthread_mutex_t mutex_task_packet_map;

hsa_agent_dispatch_packet_t *get_task_packet() {
  hsa_agent_dispatch_packet_t *packet = NULL;
  lock(&mutex_task_packet_map);
  packet = TaskPacketMap[pthread_self()];
  unlock(&mutex_task_packet_map);
  return packet;
}

void set_task_packet(hsa_agent_dispatch_packet_t *packet) {
  lock(&mutex_task_packet_map);
  TaskPacketMap[pthread_self()] = packet;
  unlock(&mutex_task_packet_map);
}

hsa_queue_t *get_cpu_queue(int cpu_id, int tid) {
  atmi_place_t place = ATMI_PLACE_CPU(0, cpu_id);
  ATLCPUProcessor &proc = get_processor<ATLCPUProcessor>(place);
  return proc.getQueueAt(tid);
}

thread_agent_t *get_cpu_q_agent(int cpu_id, int tid) {
  atmi_place_t place = ATMI_PLACE_CPU(0, cpu_id);
  ATLCPUProcessor &proc = get_processor<ATLCPUProcessor>(place);
  return proc.getThreadAgentAt(tid);
}

hsa_signal_t *get_worker_sig(hsa_queue_t *queue) {
  hsa_signal_t *ret = NULL;
  for (int cpu = 0; cpu < g_atmi_machine.device_count_by_type[ATMI_DEVTYPE_CPU];
       cpu++) {
    atmi_place_t place = ATMI_PLACE_CPU(0, cpu);
    ATLCPUProcessor &proc = get_processor<ATLCPUProcessor>(place);
    ret = proc.get_worker_sig(queue);
    if (ret != NULL) {
      break;
    }
  }
  return ret;
}

void signal_worker(hsa_queue_t *queue, int signal) {
  DEBUG_PRINT("Signaling work %d\n", signal);
  hsa_signal_t *worker_sig = get_worker_sig(queue);
  if (!worker_sig) DEBUG_PRINT("Signal is NULL!\n");
  hsa_signal_store_release(*worker_sig, signal);
}

void signal_worker_id(int cpu_id, int tid, int signal) {
  DEBUG_PRINT("Signaling work %d\n", signal);
  atmi_place_t place = ATMI_PLACE_CPU(0, cpu_id);
  ATLCPUProcessor &proc = get_processor<ATLCPUProcessor>(place);
  thread_agent_t *agent = proc.getThreadAgentAt(tid);
  hsa_signal_store_release(agent->worker_sig, signal);
}

uint8_t get_packet_type(uint16_t header) {
  // FIXME: The width of packet type is 8 bits. Change to below line if width
  // changes
  // return (header >> HSA_PACKET_HEADER_TYPE) & ((1 <<
  // HSA_PACKET_HEADER_WIDTH_TYPE) - 1);
  return (header >> HSA_PACKET_HEADER_TYPE) & 0xFF;
}

int process_packet(thread_agent_t *agent) {
  hsa_queue_t *queue = agent->queue;
  int id = agent->id;
  DEBUG_PRINT("Processing Packet from CPU Queue\n");

  struct timespec start_time, end_time;
  uint64_t start_time_ns;
  uint64_t end_time_ns;
  uint64_t read_index = hsa_queue_load_read_index_acquire(queue);
  assert(read_index == 0);
  hsa_signal_t doorbell = queue->doorbell_signal;
  /* FIXME: Handle queue overflows */
  while (read_index < queue->size) {
    DEBUG_PRINT("Read Index: %" PRIu64 " Queue Size: %" PRIu32 "\n", read_index,
                queue->size);
    hsa_signal_value_t doorbell_value = INT_MAX;
    agent->timer.start();
    while ((doorbell_value = hsa_signal_wait_acquire(
                doorbell, HSA_SIGNAL_CONDITION_GTE, read_index, UINT64_MAX,
                ATMI_WAIT_STATE)) < (hsa_signal_value_t)read_index) {
    }
    if (doorbell_value == INT_MAX) break;
    hsa_agent_dispatch_packet_t *packets =
        reinterpret_cast<hsa_agent_dispatch_packet_t *>(queue->base_address);
    hsa_agent_dispatch_packet_t *packet = packets + read_index % queue->size;

    int i;
    DEBUG_PRINT("Processing CPU task with header: %d\n",
                get_packet_type(packet->header));
    // wait til the packet is ready to be dispatched
    while (get_packet_type(packet->header) == HSA_PACKET_TYPE_VENDOR_SPECIFIC) {
    }
    switch (get_packet_type(packet->header)) {
      case HSA_PACKET_TYPE_BARRIER_OR: {
        ;
        hsa_barrier_or_packet_t *barrier_or =
            reinterpret_cast<hsa_barrier_or_packet_t *>(packet);
        DEBUG_PRINT("Executing OR barrier\n");
        for (i = 0; i < 5; ++i) {
          if (barrier_or->dep_signal[i].handle != 0) {
            hsa_signal_wait_acquire(barrier_or->dep_signal[i],
                                    HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
                                    HSA_WAIT_STATE_BLOCKED);
            DEBUG_PRINT("OR Signal %d completed...breaking loop\n", i);
            break;
          }
        }
        packet_store_release(reinterpret_cast<uint32_t *>(barrier_or),
                             create_header(HSA_PACKET_TYPE_INVALID, 0),
                             HSA_PACKET_TYPE_BARRIER_OR);
      } break;
      case HSA_PACKET_TYPE_BARRIER_AND: {
        ;
        hsa_barrier_and_packet_t *barrier =
            reinterpret_cast<hsa_barrier_and_packet_t *>(packet);
        DEBUG_PRINT("Executing AND barrier\n");
        for (i = 0; i < 5; ++i) {
          if (barrier->dep_signal[i].handle != 0) {
            DEBUG_PRINT("Waiting for signal handle: %" PRIu64 "\n",
                        barrier->dep_signal[i].handle);
            hsa_signal_wait_acquire(barrier->dep_signal[i],
                                    HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
                                    HSA_WAIT_STATE_BLOCKED);
            DEBUG_PRINT("AND Signal %d completed...\n", i);
          }
        }
        packet_store_release(reinterpret_cast<uint32_t *>(barrier),
                             create_header(HSA_PACKET_TYPE_INVALID, 0),
                             HSA_PACKET_TYPE_BARRIER_AND);
      } break;
      case HSA_PACKET_TYPE_AGENT_DISPATCH: {
        ;
        DEBUG_PRINT("%lu --> %p\n", pthread_self(), packet);
        set_task_packet(packet);

        TaskImpl *task = getTaskImpl(packet->arg[0]);
        if (task->profilable_ == ATMI_TRUE) {
          clock_gettime(CLOCK_MONOTONIC_RAW, &start_time);
          start_time_ns = get_nanosecs(context_init_time, start_time);
        }
        DEBUG_PRINT("{{{ Thread[%lu] --> ID[%lu]\n", pthread_self(), task->id_);
        Kernel *kernel = reinterpret_cast<Kernel *>(packet->arg[2]);
        int kernel_id = packet->type;
        CPUKernelImpl *kernel_impl =
            dynamic_cast<CPUKernelImpl *>(kernel->impls()[kernel_id]);
        std::vector<void *> kernel_args;
        void *kernel_args_region = reinterpret_cast<void *>(packet->arg[1]);
        uint64_t num_params = kernel->num_args();
        char *thisKernargAddress = reinterpret_cast<char *>(kernel_args_region);
        for (int i = 0; i < kernel->num_args(); i++) {
          kernel_args.push_back(reinterpret_cast<void *>(thisKernargAddress));
          thisKernargAddress += kernel->arg_sizes()[i];
        }
        switch (num_params) {
          case 0: {
            ;
            void (*function0)(void) = (void (*)(void))kernel_impl->function();
            DEBUG_PRINT("Func Ptr: %p Args: NONE\n", function0);
            function0();
          } break;
          case 1: {
            ;
            void (*function1)(ARG_TYPE) =
                (void (*)(ARG_TYPE))kernel_impl->function();
            DEBUG_PRINT("Args: %p\n", kernel_args[0]);
            function1(kernel_args[0]);
          } break;
          case 2: {
            ;
            void (*function2)(ARG_TYPE, ARG_TYPE) =
                (void (*)(ARG_TYPE, ARG_TYPE))kernel_impl->function();
            DEBUG_PRINT("Args: %p %p\n", kernel_args[0], kernel_args[1]);
            function2(kernel_args[0], kernel_args[1]);
          } break;
          case 3: {
            ;
            void (*function3)(ARG_TYPE REPEAT2(ARG_TYPE)) =
                (void (*)(ARG_TYPE REPEAT2(ARG_TYPE)))kernel_impl->function();
            DEBUG_PRINT("Args: %p %p %p\n", kernel_args[0], kernel_args[1],
                        kernel_args[2]);
            function3(kernel_args[0], kernel_args[1], kernel_args[2]);
          } break;
          case 4: {
            ;
            void (*function4)(ARG_TYPE REPEAT2(ARG_TYPE) REPEAT(ARG_TYPE)) =
                (void (*)(ARG_TYPE REPEAT2(ARG_TYPE)
                              REPEAT(ARG_TYPE)))kernel_impl->function();
            function4(kernel_args[0], kernel_args[1], kernel_args[2],
                      kernel_args[3]);
          } break;
          case 5: {
            ;
            void (*function5)(ARG_TYPE REPEAT4(ARG_TYPE)) =
                (void (*)(ARG_TYPE REPEAT4(ARG_TYPE)))kernel_impl->function();
            function5(kernel_args[0], kernel_args[1], kernel_args[2],
                      kernel_args[3], kernel_args[4]);
          } break;
          case 6: {
            ;
            void (*function6)(ARG_TYPE REPEAT4(ARG_TYPE) REPEAT(ARG_TYPE)) =
                (void (*)(ARG_TYPE REPEAT4(ARG_TYPE)
                              REPEAT(ARG_TYPE)))kernel_impl->function();
            function6(kernel_args[0], kernel_args[1], kernel_args[2],
                      kernel_args[3], kernel_args[4], kernel_args[5]);
          } break;
          case 7: {
            ;
            void (*function7)(ARG_TYPE REPEAT4(ARG_TYPE) REPEAT2(ARG_TYPE)) =
                (void (*)(ARG_TYPE REPEAT4(ARG_TYPE)
                              REPEAT2(ARG_TYPE)))kernel_impl->function();
            function7(kernel_args[0], kernel_args[1], kernel_args[2],
                      kernel_args[3], kernel_args[4], kernel_args[5],
                      kernel_args[6]);
          } break;
          case 8: {
            ;
            void (*function8)(ARG_TYPE REPEAT4(ARG_TYPE) REPEAT2(ARG_TYPE)
                                  REPEAT(ARG_TYPE)) =
                (void (*)(ARG_TYPE REPEAT4(ARG_TYPE) REPEAT2(ARG_TYPE)
                              REPEAT(ARG_TYPE)))kernel_impl->function();
            function8(kernel_args[0], kernel_args[1], kernel_args[2],
                      kernel_args[3], kernel_args[4], kernel_args[5],
                      kernel_args[6], kernel_args[7]);
          } break;
          case 9: {
            ;
            void (*function9)(ARG_TYPE REPEAT8(ARG_TYPE)) =
                (void (*)(ARG_TYPE REPEAT8(ARG_TYPE)))kernel_impl->function();
            function9(kernel_args[0], kernel_args[1], kernel_args[2],
                      kernel_args[3], kernel_args[4], kernel_args[5],
                      kernel_args[6], kernel_args[7], kernel_args[8]);
          } break;
          case 10: {
            ;
            void (*function10)(ARG_TYPE REPEAT8(ARG_TYPE) REPEAT(ARG_TYPE)) =
                (void (*)(ARG_TYPE REPEAT8(ARG_TYPE)
                              REPEAT(ARG_TYPE)))kernel_impl->function();
            function10(kernel_args[0], kernel_args[1], kernel_args[2],
                       kernel_args[3], kernel_args[4], kernel_args[5],
                       kernel_args[6], kernel_args[7], kernel_args[8],
                       kernel_args[9]);
          } break;
          case 11: {
            ;
            void (*function11)(ARG_TYPE REPEAT8(ARG_TYPE) REPEAT2(ARG_TYPE)) =
                (void (*)(ARG_TYPE REPEAT8(ARG_TYPE)
                              REPEAT2(ARG_TYPE)))kernel_impl->function();
            function11(kernel_args[0], kernel_args[1], kernel_args[2],
                       kernel_args[3], kernel_args[4], kernel_args[5],
                       kernel_args[6], kernel_args[7], kernel_args[8],
                       kernel_args[9], kernel_args[10]);
          } break;
          case 12: {
            ;
            void (*function12)(ARG_TYPE REPEAT8(ARG_TYPE) REPEAT2(ARG_TYPE)
                                   REPEAT(ARG_TYPE)) =
                (void (*)(ARG_TYPE REPEAT8(ARG_TYPE) REPEAT2(ARG_TYPE)
                              REPEAT(ARG_TYPE)))kernel_impl->function();
            function12(kernel_args[0], kernel_args[1], kernel_args[2],
                       kernel_args[3], kernel_args[4], kernel_args[5],
                       kernel_args[6], kernel_args[7], kernel_args[8],
                       kernel_args[9], kernel_args[10], kernel_args[11]);
          } break;
          case 13: {
            ;
            void (*function13)(ARG_TYPE REPEAT8(ARG_TYPE) REPEAT4(ARG_TYPE)) =
                (void (*)(ARG_TYPE REPEAT8(ARG_TYPE)
                              REPEAT4(ARG_TYPE)))kernel_impl->function();
            function13(kernel_args[0], kernel_args[1], kernel_args[2],
                       kernel_args[3], kernel_args[4], kernel_args[5],
                       kernel_args[6], kernel_args[7], kernel_args[8],
                       kernel_args[9], kernel_args[10], kernel_args[11],
                       kernel_args[12]);
          } break;
          case 14: {
            ;
            void (*function14)(ARG_TYPE REPEAT8(ARG_TYPE) REPEAT4(ARG_TYPE)
                                   REPEAT(ARG_TYPE)) =
                (void (*)(ARG_TYPE REPEAT8(ARG_TYPE) REPEAT4(ARG_TYPE)
                              REPEAT(ARG_TYPE)))kernel_impl->function();
            function14(kernel_args[0], kernel_args[1], kernel_args[2],
                       kernel_args[3], kernel_args[4], kernel_args[5],
                       kernel_args[6], kernel_args[7], kernel_args[8],
                       kernel_args[9], kernel_args[10], kernel_args[11],
                       kernel_args[12], kernel_args[13]);
          } break;
          case 15: {
            ;
            void (*function15)(ARG_TYPE REPEAT8(ARG_TYPE) REPEAT4(ARG_TYPE)
                                   REPEAT2(ARG_TYPE)) =
                (void (*)(ARG_TYPE REPEAT8(ARG_TYPE) REPEAT4(ARG_TYPE)
                              REPEAT2(ARG_TYPE)))kernel_impl->function();
            function15(kernel_args[0], kernel_args[1], kernel_args[2],
                       kernel_args[3], kernel_args[4], kernel_args[5],
                       kernel_args[6], kernel_args[7], kernel_args[8],
                       kernel_args[9], kernel_args[10], kernel_args[11],
                       kernel_args[12], kernel_args[13], kernel_args[14]);
          } break;
          case 16: {
            ;
            void (*function16)(ARG_TYPE REPEAT8(ARG_TYPE) REPEAT4(ARG_TYPE)
                                   REPEAT2(ARG_TYPE) REPEAT(ARG_TYPE)) =
                (void (*)(ARG_TYPE REPEAT8(ARG_TYPE) REPEAT4(ARG_TYPE) REPEAT2(
                    ARG_TYPE) REPEAT(ARG_TYPE)))kernel_impl->function();
            function16(kernel_args[0], kernel_args[1], kernel_args[2],
                       kernel_args[3], kernel_args[4], kernel_args[5],
                       kernel_args[6], kernel_args[7], kernel_args[8],
                       kernel_args[9], kernel_args[10], kernel_args[11],
                       kernel_args[12], kernel_args[13], kernel_args[14],
                       kernel_args[15]);
          } break;
          case 17: {
            ;
            void (*function17)(ARG_TYPE REPEAT16(ARG_TYPE)) =
                (void (*)(ARG_TYPE REPEAT16(ARG_TYPE)))kernel_impl->function();
            function17(kernel_args[0], kernel_args[1], kernel_args[2],
                       kernel_args[3], kernel_args[4], kernel_args[5],
                       kernel_args[6], kernel_args[7], kernel_args[8],
                       kernel_args[9], kernel_args[10], kernel_args[11],
                       kernel_args[12], kernel_args[13], kernel_args[14],
                       kernel_args[15], kernel_args[16]);
          } break;
          case 18: {
            ;
            void (*function18)(ARG_TYPE REPEAT16(ARG_TYPE) REPEAT(ARG_TYPE)) =
                (void (*)(ARG_TYPE REPEAT16(ARG_TYPE)
                              REPEAT(ARG_TYPE)))kernel_impl->function();
            function18(kernel_args[0], kernel_args[1], kernel_args[2],
                       kernel_args[3], kernel_args[4], kernel_args[5],
                       kernel_args[6], kernel_args[7], kernel_args[8],
                       kernel_args[9], kernel_args[10], kernel_args[11],
                       kernel_args[12], kernel_args[13], kernel_args[14],
                       kernel_args[15], kernel_args[16], kernel_args[17]);
          } break;
          case 19: {
            ;
            void (*function19)(ARG_TYPE REPEAT16(ARG_TYPE) REPEAT2(ARG_TYPE)) =
                (void (*)(ARG_TYPE REPEAT16(ARG_TYPE)
                              REPEAT2(ARG_TYPE)))kernel_impl->function();
            function19(kernel_args[0], kernel_args[1], kernel_args[2],
                       kernel_args[3], kernel_args[4], kernel_args[5],
                       kernel_args[6], kernel_args[7], kernel_args[8],
                       kernel_args[9], kernel_args[10], kernel_args[11],
                       kernel_args[12], kernel_args[13], kernel_args[14],
                       kernel_args[15], kernel_args[16], kernel_args[17],
                       kernel_args[18]);
          } break;
          case 20: {
            ;
            void (*function20)(ARG_TYPE REPEAT16(ARG_TYPE) REPEAT2(ARG_TYPE)
                                   REPEAT(ARG_TYPE)) =
                (void (*)(ARG_TYPE REPEAT16(ARG_TYPE) REPEAT2(ARG_TYPE)
                              REPEAT(ARG_TYPE)))kernel_impl->function();
            function20(kernel_args[0], kernel_args[1], kernel_args[2],
                       kernel_args[3], kernel_args[4], kernel_args[5],
                       kernel_args[6], kernel_args[7], kernel_args[8],
                       kernel_args[9], kernel_args[10], kernel_args[11],
                       kernel_args[12], kernel_args[13], kernel_args[14],
                       kernel_args[15], kernel_args[16], kernel_args[17],
                       kernel_args[18], kernel_args[19]);
          } break;
          case 37: {
            ;
            void (*function37)(ARG_TYPE REPEAT16(ARG_TYPE) REPEAT16(ARG_TYPE)
                                   REPEAT4(ARG_TYPE)) =
                (void (*)(ARG_TYPE REPEAT16(ARG_TYPE) REPEAT16(ARG_TYPE)
                              REPEAT4(ARG_TYPE)))kernel_impl->function();
            function37(kernel_args[0], kernel_args[1], kernel_args[2],
                       kernel_args[3], kernel_args[4], kernel_args[5],
                       kernel_args[6], kernel_args[7], kernel_args[8],
                       kernel_args[9], kernel_args[10], kernel_args[11],
                       kernel_args[12], kernel_args[13], kernel_args[14],
                       kernel_args[15], kernel_args[16], kernel_args[17],
                       kernel_args[18], kernel_args[19], kernel_args[20],
                       kernel_args[21], kernel_args[22], kernel_args[23],
                       kernel_args[24], kernel_args[25], kernel_args[26],
                       kernel_args[27], kernel_args[28], kernel_args[29],
                       kernel_args[30], kernel_args[31], kernel_args[32],
                       kernel_args[33], kernel_args[34], kernel_args[35],
                       kernel_args[36]);
          } break;
          default:

            DEBUG_PRINT("Too many function arguments: %" PRIu64 "\n",
                        num_params);
            check(Too many function arguments,
                  HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS);
            break;
        }
        // reset task packet map so that other tasks will not be able to query
        // for
        // thread IDs and sizes
        set_task_packet(NULL);

        DEBUG_PRINT("Signaling from CPU task: %" PRIu64 "\n",
                    packet->completion_signal.handle);
        packet_store_release(reinterpret_cast<uint32_t *>(packet),
                             create_header(HSA_PACKET_TYPE_INVALID, 0),
                             packet->type);
        kernel_args.clear();
        DEBUG_PRINT("End Thread[%lu] --> ID[%lu] }}}\n", pthread_self(),
                    task->id_);
        if (task->profilable_ == ATMI_TRUE) {
          clock_gettime(CLOCK_MONOTONIC_RAW, &end_time);
          end_time_ns = get_nanosecs(context_init_time, end_time);
          if (task->atmi_task_) {
            task->atmi_task_->profile.start_time = start_time_ns;
            task->atmi_task_->profile.end_time = end_time_ns;
            task->atmi_task_->profile.dispatch_time = start_time_ns;
            task->atmi_task_->profile.ready_time = start_time_ns;
            DEBUG_PRINT("Task %p timing info (%" PRIu64 ", %" PRIu64 ")\n",
                        task->atmi_task_, start_time_ns, end_time_ns);
          }
        }
      } break;
    }
    if (packet->completion_signal.handle != 0) {
      hsa_signal_subtract_release(packet->completion_signal, 1);
    }
    read_index++;
    hsa_queue_store_read_index_release(queue, read_index);
    agent->timer.stop();
  }

  DEBUG_PRINT("Finished executing agent dispatch\n");

  // Finishing this task may free up more tasks, so issue the wakeup command
  // DEBUG_PRINT("Signaling more work\n");
  // hsa_signal_store_release(worker_sig[id], PROCESS_PKT);
  return 0;
}

void *agent_worker(void *agent_args) {
  thread_agent_t *agent = reinterpret_cast<thread_agent_t *>(agent_args);

  unsigned num_cpus = std::thread::hardware_concurrency();
  // pin this thread to the core number as its agent ID...
  // ...BUT from the highest core number
  // thread: 0 1 2 3
  // core  : 3 2 1 0
  // rationale: bind the main thread to core 0.
  //            bind the callback thread to core 1.
  //            bind the CPU agent threads to rest of the cores
  int set_core = (num_cpus - 1 - (1 * agent->id)) % num_cpus;
  DEBUG_PRINT("Setting on CPU core: %d / %d\n", set_core, num_cpus);
  set_thread_affinity(set_core);

  hsa_signal_value_t sig_value = IDLE;
  while (sig_value == IDLE) {
    DEBUG_PRINT("Worker thread sleeping\n");
    sig_value =
        hsa_signal_wait_acquire(agent->worker_sig, HSA_SIGNAL_CONDITION_LT,
                                IDLE, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
    DEBUG_PRINT("Worker thread waking up\n");

    if (sig_value == FINISH) {
      DEBUG_PRINT("Worker thread received the EXIT SIGNAL\n");
      break;
    }

    if (PROCESS_PKT ==
        hsa_signal_cas_acq_rel(agent->worker_sig, PROCESS_PKT, IDLE)) {
      if (!process_packet(agent)) continue;
    }
    sig_value = IDLE;
  }

  return NULL;
}

void cpu_agent_init(int cpu_id, const size_t num_queues) {
  static bool initialized = false;
  hsa_status_t err;
  uint32_t i;
  atmi_place_t place = ATMI_PLACE_CPU(0, cpu_id);
  ATLCPUProcessor &proc = get_processor<ATLCPUProcessor>(place);
  proc.createQueues(num_queues);
  if (!initialized) pthread_mutex_init(&mutex_task_packet_map, NULL);
  initialized = true;
}

/* FIXME: When and who should call this cleanup funtion? */
void agent_fini() {
  DEBUG_PRINT("SIGNALING EXIT\n");

  /* wait for the other threads */
  for (int cpu = 0; cpu < g_atmi_machine.device_count_by_type[ATMI_DEVTYPE_CPU];
       cpu++) {
    atmi_place_t place = ATMI_PLACE_CPU(0, cpu);
    ATLCPUProcessor &proc = get_processor<ATLCPUProcessor>(place);
    const std::vector<thread_agent_t *> &agents = proc.thread_agents();
    uint32_t i;
    for (i = 0; i < agents.size(); i++) {
      thread_agent_t *agent = agents[i];
      DEBUG_PRINT("Setting doorbell[%d] to INT_MAX\n", i);
      hsa_signal_store_release(agent->queue->doorbell_signal, INT_MAX);
      hsa_signal_store_release(agent->worker_sig, FINISH);
      pthread_join(agent->thread, NULL);
      std::string str(std::string("CPU[" + std::to_string(i) + "] Timer"));
      agent->timer.bufPrint(std::cout, str);
    }
  }
  DEBUG_PRINT("agent_fini completed\n");
}

TaskImpl *get_cur_thread_task_impl() {
  hsa_agent_dispatch_packet_t *packet = get_task_packet();
  if (!packet) {
    DEBUG_PRINT(
        "WARNING! Cannot query thread diagnostics outside an ATMI CPU task\n");
  }
  DEBUG_PRINT("(Get) %lu --> %p\n", pthread_self(), packet);
  TaskImpl *task = NULL;
  if (packet) task = getTaskImpl(packet->arg[0]);
  return task;
}

atmi_task_handle_t get_atmi_task_handle() {
  TaskImpl *task = get_cur_thread_task_impl();
  if (task) {
    DEBUG_PRINT("Task ID: %lu\n", task->id_);
    return task->id_;
  } else {
    DEBUG_PRINT("Task ID: NULL\n");
    return ATMI_NULL_TASK_HANDLE;
  }
}

unsigned long get_global_size(unsigned int dim) {
  TaskImpl *task = get_cur_thread_task_impl();
  if (task) {
    if (dim >= 0 && dim < 3)
      return dynamic_cast<ComputeTaskImpl *>(task)->gridDim_[dim];
    else
      return 1;
  } else {
    return 0;
  }
}

unsigned long get_local_size(unsigned int dim) {
  /*TaskImpl *task = get_cur_thread_task_impl();
  if(dim >=0 && dim < 3)
      return task->groupDim_[dim];
  else
  */
  // TODO(ashwinma): Current CPU task model is to have a single thread per
  // "workgroup"
  // because i clearly do not see the necessity to have a tree based hierarchy
  // per CPU socket. Should revisit if we get a compelling case otherwise.
  return 1;
}

unsigned long get_num_groups(unsigned int dim) {
  // return ((get_global_size(dim)-1)/(dim)*get_local_size(dim))+1;
  // TODO(ashwinma): simplify return because local dims are hardcoded to 1
  return get_global_size(dim);
}

unsigned long get_local_id(unsigned int dim) {
  // TODO(ashwinma): Current CPU task model is to have a single thread per
  // "workgroup"
  // because i clearly do not see the necessity to have a tree based hierarchy
  // per CPU socket. Should revisit if we get a compelling case otherwise.
  return 0;
}

unsigned long get_global_id(unsigned int dim) {
  hsa_agent_dispatch_packet_t *packet = get_task_packet();
  if (!packet) {
    DEBUG_PRINT(
        "WARNING! Cannot query thread diagnostics outside an ATMI CPU task\n");
  }

  DEBUG_PRINT("(Get) %lu --> %p\n", pthread_self(), packet);
  if (packet && dim >= 0 && dim < 3) {
    unsigned long flat_id = packet->arg[3];
    unsigned long x_dim = get_global_size(0);
    unsigned long y_dim = get_global_size(1);
    unsigned long id = 0;
    unsigned long rel_id = 0;
    if (dim == 0) {
      // rel_id = flat_id % (x_dim * y_dim);
      // id = rel_id / y_dim;
      rel_id = flat_id / y_dim;
      id = rel_id % x_dim;
    } else if (dim == 1) {
      id = flat_id % y_dim;
    } else if (dim == 2) {
      id = flat_id / (x_dim * y_dim);
    }
    return id;
  } else {
    return 0;
  }
}

unsigned long get_group_id(unsigned int dim) { return get_global_id(dim); }
