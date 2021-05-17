// RUN: %libomptarget-compile-generic && env LIBOMPTARGET_DEBUG=1 %libomptarget-run-generic 2>&1 | %fcheck-generic -allow-empty -check-prefix=DEBUG
// REQUIRES: libomptarget-debug

#include <cstdio>
#include <cstdlib>
#include <cassert>

// Data structure definitions copied from OpenMP RTL.
struct __tgt_target_non_contig {
  int64_t offset;
  int64_t width;
  int64_t stride;
};

enum tgt_map_type {
  OMP_TGT_MAPTYPE_NON_CONTIG      = 0x100000000000
};

// OpenMP RTL interfaces
#ifdef __cplusplus
extern "C" {
#endif
void __tgt_target_data_update(int64_t device_id, int32_t arg_num,
                              void **args_base, void **args, int64_t *arg_sizes,
                              int64_t *arg_types);
#ifdef __cplusplus
}
#endif

int main() {
  // case 1
  // int arr[3][4][5][6];
  // #pragma omp target update to(arr[0:2][1:3][1:2][:])
  // set up descriptor
  __tgt_target_non_contig non_contig[5] = {
      {0, 2, 480}, {1, 3, 120}, {1, 2, 24}, {0, 6, 4}, {0, 1, 4}};
  int64_t size = 4, type = OMP_TGT_MAPTYPE_NON_CONTIG;

  void *base;
  void *begin = &non_contig;
  int64_t *sizes = &size;
  int64_t *types = &type;

  // The below diagram is the visualization of the non-contiguous transfer after
  // optimization. Note that each element represent the innermost dimension
  // (unit size = 24) since the stride * count of last dimension is equal to the
  // stride of second last dimension.
  //
  // OOOOO OOOOO OOOOO
  // OXXOO OXXOO OOOOO
  // OXXOO OXXOO OOOOO
  // OXXOO OXXOO OOOOO
  __tgt_target_data_update(/*device_id*/ -1, /*arg_num*/ 1, &base, &begin,
                           sizes, types);
  // DEBUG: offset 144
  // DEBUG: offset 264
  // DEBUG: offset 384
  // DEBUG: offset 624
  // DEBUG: offset 744
  // DEBUG: offset 864

  // case 2
  // double darr[3][4][5];
  // #pragma omp target update to(darr[0:2:2][2:2][:2:2])
  // set up descriptor
  __tgt_target_non_contig non_contig_2[4] = {
      {0, 2, 320}, {2, 2, 40}, {0, 2, 16}, {0, 1, 8}};
  int64_t size_2 = 4, type_2 = OMP_TGT_MAPTYPE_NON_CONTIG;

  void *base_2;
  void *begin_2 = &non_contig_2;
  int64_t *sizes_2 = &size_2;
  int64_t *types_2 = &type_2;

  // The below diagram is the visualization of the non-contiguous transfer after
  // optimization. Note that each element represent the innermost dimension
  // (unit size = 24) since the stride * count of last dimension is equal to the
  // stride of second last dimension.
  //
  // OOOOO OOOOO OOOOO
  // OOOOO OOOOO OOOOO
  // XOXOO OOOOO XOXOO
  // XOXOO OOOOO XOXOO
  __tgt_target_data_update(/*device_id*/ -1, /*arg_num*/ 1, &base_2, &begin_2,
                           sizes_2, types_2);
  // DEBUG: offset 80
  // DEBUG: offset 96
  // DEBUG: offset 120
  // DEBUG: offset 136
  // DEBUG: offset 400
  // DEBUG: offset 416
  // DEBUG: offset 440
  // DEBUG: offset 456
  return 0;
}

