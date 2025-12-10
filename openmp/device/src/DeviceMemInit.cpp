extern "C" {
void __ockl_dm_init_v1(unsigned long hp, unsigned long sp, unsigned int hb,
                       unsigned int nis);

/// Device memory initialization kernel
__attribute__((amdgpu_kernel, amdgpu_flat_work_group_size(256, 256),
               amdgpu_max_num_work_groups(1), visibility("protected"))) void
__omp_dm_init_kernel(unsigned long heap_ptr, unsigned long slab_ptr) {

  unsigned int HEAP_BYTES = 1;
  unsigned int NUM_SLABS = 256;

  // Use 256 * 2MB = 512MB for GPU memory allocation.
  __ockl_dm_init_v1(heap_ptr, slab_ptr, HEAP_BYTES, NUM_SLABS);
}
}
