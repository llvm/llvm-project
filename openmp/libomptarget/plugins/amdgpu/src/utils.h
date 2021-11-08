#ifndef OPENMP_LIBOMPTARGET_AMDGPU_UTILS_H
#define OPENMP_LIBOMPTARGET_AMDGPU_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif
hsa_status_t impl_memcpy_no_signal(void *dest, void *src, size_t size,
                                   int host2Device);
hsa_status_t host_malloc(void **mem, size_t size);

hsa_status_t device_malloc(void **mem, size_t size, int device_id);
hsa_status_t impl_free(void *mem);

hsa_status_t ftn_assign_wrapper(void *arg0, void *arg1, void *arg2, void *arg3,
                                void *arg4);
#ifdef __cplusplus
}
#endif

#endif
