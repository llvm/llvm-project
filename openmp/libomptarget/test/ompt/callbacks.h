#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

// Tool related code below
#include <omp-tools.h>

// For EMI callbacks
ompt_id_t next_op_id = 0x8000000000000001;

// OMPT callbacks

// Synchronous callbacks
static void on_ompt_callback_device_initialize(int device_num, const char *type,
                                               ompt_device_t *device,
                                               ompt_function_lookup_t lookup,
                                               const char *documentation) {
  printf("Callback Init: device_num=%d type=%s device=%p lookup=%p doc=%p\n",
         device_num, type, device, lookup, documentation);
}

static void on_ompt_callback_device_finalize(int device_num) {
  printf("Callback Fini: device_num=%d\n", device_num);
}

static void on_ompt_callback_device_load(int device_num, const char *filename,
                                         int64_t offset_in_file,
                                         void *vma_in_file, size_t bytes,
                                         void *host_addr, void *device_addr,
                                         uint64_t module_id) {
  printf("Callback Load: device_num:%d module_id:%lu filename:%s host_adddr:%p "
         "device_addr:%p bytes:%lu\n",
         device_num, module_id, filename, host_addr, device_addr, bytes);
}

static void on_ompt_callback_target_data_op(
    ompt_id_t target_id, ompt_id_t host_op_id, ompt_target_data_op_t optype,
    void *src_addr, int src_device_num, void *dest_addr, int dest_device_num,
    size_t bytes, const void *codeptr_ra) {
  assert(codeptr_ra != 0 && "Unexpected null codeptr");
  printf("  Callback DataOp: target_id=%lu host_op_id=%lu optype=%d src=%p "
         "src_device_num=%d "
         "dest=%p dest_device_num=%d bytes=%lu code=%p\n",
         target_id, host_op_id, optype, src_addr, src_device_num, dest_addr,
         dest_device_num, bytes, codeptr_ra);
}

static void on_ompt_callback_target(ompt_target_t kind,
                                    ompt_scope_endpoint_t endpoint,
                                    int device_num, ompt_data_t *task_data,
                                    ompt_id_t target_id,
                                    const void *codeptr_ra) {
  assert(codeptr_ra != 0 && "Unexpected null codeptr");
  printf("Callback Target: target_id=%lu kind=%d endpoint=%d device_num=%d "
         "code=%p\n",
         target_id, kind, endpoint, device_num, codeptr_ra);
}

static void on_ompt_callback_target_submit(ompt_id_t target_id,
                                           ompt_id_t host_op_id,
                                           unsigned int requested_num_teams) {
  printf("  Callback Submit: target_id=%lu host_op_id=%lu req_num_teams=%d\n",
         target_id, host_op_id, requested_num_teams);
}

static void on_ompt_callback_target_map(ompt_id_t target_id,
                                        unsigned int nitems, void **host_addr,
                                        void **device_addr, size_t *bytes,
                                        unsigned int *mapping_flags,
                                        const void *codeptr_ra) {
  printf("Target map callback is unimplemented\n");
  abort();
}

static void on_ompt_callback_target_data_op_emi(
    ompt_scope_endpoint_t endpoint, ompt_data_t *target_task_data,
    ompt_data_t *target_data, ompt_id_t *host_op_id,
    ompt_target_data_op_t optype, void *src_addr, int src_device_num,
    void *dest_addr, int dest_device_num, size_t bytes,
    const void *codeptr_ra) {
  assert(codeptr_ra != 0 && "Unexpected null codeptr");
  if (endpoint == ompt_scope_begin)
    *host_op_id = next_op_id++;
  printf("  Callback DataOp EMI: endpoint=%d optype=%d target_task_data=%p "
         "(0x%lx) target_data=%p (0x%lx) host_op_id=%p (0x%lx) src=%p "
         "src_device_num=%d "
         "dest=%p dest_device_num=%d bytes=%lu code=%p\n",
         endpoint, optype, target_task_data, target_task_data->value,
         target_data, target_data->value, host_op_id, *host_op_id, src_addr,
         src_device_num, dest_addr, dest_device_num, bytes, codeptr_ra);
}

static void on_ompt_callback_target_emi(ompt_target_t kind,
                                        ompt_scope_endpoint_t endpoint,
                                        int device_num, ompt_data_t *task_data,
                                        ompt_data_t *target_task_data,
                                        ompt_data_t *target_data,
                                        const void *codeptr_ra) {
  assert(codeptr_ra != 0 && "Unexpected null codeptr");
  if (endpoint == ompt_scope_begin)
    target_data->value = next_op_id++;
  printf("Callback Target EMI: kind=%d endpoint=%d device_num=%d task_data=%p "
         "(0x%lx) target_task_data=%p (0x%lx) target_data=%p (0x%lx) code=%p\n",
         kind, endpoint, device_num, task_data, task_data->value,
         target_task_data, target_task_data->value, target_data,
         target_data->value, codeptr_ra);
}

static void on_ompt_callback_target_submit_emi(
    ompt_scope_endpoint_t endpoint, ompt_data_t *target_data,
    ompt_id_t *host_op_id, unsigned int requested_num_teams) {
  printf("  Callback Submit EMI: endpoint=%d  req_num_teams=%d target_data=%p "
         "(0x%lx) host_op_id=%p (0x%lx)\n",
         endpoint, requested_num_teams, target_data, target_data->value,
         host_op_id, *host_op_id);
}

static void on_ompt_callback_target_map_emi(ompt_data_t *target_data,
                                            unsigned int nitems,
                                            void **host_addr,
                                            void **device_addr, size_t *bytes,
                                            unsigned int *mapping_flags,
                                            const void *codeptr_ra) {
  printf("Target map emi callback is unimplemented\n");
  abort();
}
