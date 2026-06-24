#include <assert.h>

// Tool related code below
#include <omp-tools.h>

// From openmp/runtime/test/ompt/callback.h
#define register_ompt_callback_t(name, type)                                   \
  do {                                                                         \
    type f_##name = &on_##name;                                                \
    if (ompt_set_callback(name, (ompt_callback_t)f_##name) == ompt_set_never)  \
      printf("0: Could not register callback '" #name "'\n");                  \
  } while (0)

#define register_ompt_callback(name) register_ompt_callback_t(name, name##_t)

// OMPT entry point handles
static ompt_set_callback_t ompt_set_callback = 0;
static ompt_get_record_type_t ompt_get_record_type_fn = 0;
static ompt_get_device_time_t ompt_get_device_time_fn = 0;

// OMPT callbacks

// Synchronous callbacks
static void on_ompt_callback_device_initialize
(
  int device_num,
  const char *type,
  ompt_device_t *device,
  ompt_function_lookup_t lookup,
  const char *documentation
 ) {
  printf("Callback Init: device_num=%d type=%s device=%p lookup=%p doc=%p\n",
	 device_num, type, device, lookup, documentation);

  ompt_get_record_type_fn = (ompt_get_record_type_t) lookup("ompt_get_record_type");
  ompt_get_device_time_fn = (ompt_get_device_time_t) lookup("ompt_get_device_time");

  if (ompt_get_record_type_fn) {
    ompt_buffer_cursor_t buf;
    // FIXME: For now, we pass a NULL buffer
    ompt_record_t rec_type = ompt_get_record_type_fn(NULL, buf);
    printf("Record Type: %s\n", ((rec_type == ompt_record_ompt) ? "OMPT" : "Unknown"));
  } else {
    printf("Could not determine Record Type");
  }

  if (ompt_get_device_time_fn) {
    uint64_t time = ompt_get_device_time_fn(NULL);
    printf("The device time can be queried: %lu\n", time);
  } else {
    printf("Could not determine the device time \n");
  }
}

static void on_ompt_callback_device_finalize
(
  int device_num
 ) {
  printf("Callback Fini: device_num=%d\n", device_num);
}

static void on_ompt_callback_device_load
    (
     int device_num,
     const char *filename,
     int64_t offset_in_file,
     void *vma_in_file,
     size_t bytes,
     void *host_addr,
     void *device_addr,
     uint64_t module_id
     ) {
  printf("Callback Load: device_num:%d module_id:%lu filename:%s host_adddr:%p device_addr:%p bytes:%lu\n",
	 device_num, module_id, filename, host_addr, device_addr, bytes);
}

static void on_ompt_callback_target_data_op
    (
     ompt_id_t target_id,
     ompt_id_t host_op_id,
     ompt_target_data_op_t optype,
     void *src_addr,
     int src_device_num,
     void *dest_addr,
     int dest_device_num,
     size_t bytes,
     const void *codeptr_ra
     ) {
  assert(codeptr_ra != 0);
  // Both src and dest must not be null
  assert(src_addr != 0 || dest_addr != 0);
  printf("  Callback DataOp: target_id=%lu host_op_id=%lu optype=%d src=%p src_device_num=%d "
	 "dest=%p dest_device_num=%d bytes=%lu code=%p\n",
	 target_id, host_op_id, optype, src_addr, src_device_num,
	 dest_addr, dest_device_num, bytes, codeptr_ra);
}

static void on_ompt_callback_target
    (
     ompt_target_t kind,
     ompt_scope_endpoint_t endpoint,
     int device_num,
     ompt_data_t *task_data,
     ompt_id_t target_id,
     const void *codeptr_ra
     ) {
  assert(codeptr_ra != 0);
  printf("Callback Target: target_id=%lu kind=%d endpoint=%d device_num=%d code=%p\n",
	 target_id, kind, endpoint, device_num, codeptr_ra);
}

static void on_ompt_callback_target_submit
    (
     ompt_id_t target_id,
     ompt_id_t host_op_id,
     unsigned int requested_num_teams
     ) {
  printf("  Callback Submit: target_id=%lu host_op_id=%lu req_num_teams=%d\n",
     target_id, host_op_id, requested_num_teams);
}

// Init functions
int ompt_initialize(
  ompt_function_lookup_t lookup,
  int initial_device_num,
  ompt_data_t *tool_data)
{
  ompt_set_callback = (ompt_set_callback_t) lookup("ompt_set_callback");

  if (!ompt_set_callback) return 0; // failed
  
  register_ompt_callback(ompt_callback_device_initialize);
  register_ompt_callback(ompt_callback_device_finalize);
  register_ompt_callback(ompt_callback_device_load);
  register_ompt_callback(ompt_callback_target_data_op);
  register_ompt_callback(ompt_callback_target);
  register_ompt_callback(ompt_callback_target_submit);



  return 1; //success
}

void ompt_finalize(ompt_data_t *tool_data)
{
}

#ifdef __cplusplus
extern "C" {
#endif
ompt_start_tool_result_t *ompt_start_tool(
  unsigned int omp_version,
  const char *runtime_version)
{
  static ompt_start_tool_result_t ompt_start_tool_result = {&ompt_initialize,&ompt_finalize, 0};
  return &ompt_start_tool_result;
}
#ifdef __cplusplus
}
#endif
