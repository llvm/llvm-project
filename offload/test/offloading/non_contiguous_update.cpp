// RUN: %libomptarget-compile-generic && env LIBOMPTARGET_DEBUG=1 %libomptarget-run-generic 2>&1 | %fcheck-generic -allow-empty -check-prefix=DEBUG
// REQUIRES: libomptarget-debug

#include <cassert>
#include <cstdio>
#include <cstdlib>

// Data structure definitions copied from OpenMP RTL.
struct __tgt_target_non_contig {
  int64_t Offset;
  int64_t Count;
  int64_t Stride;
};

enum tgt_map_type { OMP_TGT_MAPTYPE_NON_CONTIG = 0x100000000000 };

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
  {
  // case 1
  // int arr[3][4][5][6];
  // #pragma omp target update to(arr[0:2][1:3][1:2][:])
  // set up descriptor
  __tgt_target_non_contig non_contig[5] = {
    {0, 2, 480}, {120, 3, 120}, {24, 2, 24}, {0, 6, 4}, {0, 4, 1}};
  int64_t size = sizeof(non_contig) / sizeof(non_contig[0]), type = OMP_TGT_MAPTYPE_NON_CONTIG;

  void *base;
  void *begin = &non_contig;
  int64_t *sizes = &size;
  int64_t *types = &type;

  // The below diagram is the visualization of the non-contiguous transfer after
  // optimization. Note that each element represent the merged innermost
  // dimension (unit size = 24) since the stride * count of last dimension is
  // equal to the stride of second last dimension.
  //
  // OOOOO OOOOO OOOOO
  // OXXOO OXXOO OOOOO
  // OXXOO OXXOO OOOOO
  // OXXOO OXXOO OOOOO
  __tgt_target_data_update(/*device_id*/ -1, /*arg_num*/ 1, &base, &begin,
                           sizes, types);
  // DEBUG: offset 144 len 48
  // DEBUG: offset 264 len 48
  // DEBUG: offset 384 len 48
  // DEBUG: offset 624 len 48
  // DEBUG: offset 744 len 48
  // DEBUG: offset 864 len 48
  }

  {
  // case 2
  // double darr[3][4][5];
  // #pragma omp target update to(darr[0:2:2][2:2][:2:2])
  // set up descriptor
  __tgt_target_non_contig non_contig[4] = {
    {0, 2, 320}, {80, 2, 40}, {0, 2, 16}, {0, 8, 1}};
  int64_t size = sizeof(non_contig) / sizeof(non_contig[0]), type = OMP_TGT_MAPTYPE_NON_CONTIG;

  void *base;
  void *begin = &non_contig;
  int64_t *sizes = &size;
  int64_t *types = &type;

  // The below diagram is the visualization of the non-contiguous transfer after
  // optimization. Note that each element represent the innermost dimension
  // (unit size = 8).
  //
  // OOOOO OOOOO OOOOO
  // OOOOO OOOOO OOOOO
  // XOXOO OOOOO XOXOO
  // XOXOO OOOOO XOXOO
  __tgt_target_data_update(/*device_id*/ -1, /*arg_num*/ 1, &base, &begin,
                           sizes, types);
  // DEBUG: offset 80 len 8
  // DEBUG: offset 96 len 8
  // DEBUG: offset 120 len 8
  // DEBUG: offset 136 len 8
  // DEBUG: offset 400 len 8
  // DEBUG: offset 416 len 8
  // DEBUG: offset 440 len 8
  // DEBUG: offset 456 len 8
  }

  {
  // case 3
  // int darr[6][6];
  // #pragma omp target update to(darr[1:2:2][2:3])
  // set up descriptor
  __tgt_target_non_contig non_contig[3] = {
    {24, 2, 48}, {8, 3, 4}, {0, 4, 1}};
  int64_t size = sizeof(non_contig) / sizeof(non_contig[0]), type = OMP_TGT_MAPTYPE_NON_CONTIG;

  void *base;
  void *begin = &non_contig;
  int64_t *sizes = &size;
  int64_t *types = &type;

  // The below diagram is the visualization of the non-contiguous transfer after
  // optimization. Note that each element represent the merged innermost
  // dimension (unit size = 12).
  //
  // OOOOOO
  // OOXXXO
  // OOOOOO
  // OOXXXO
  // OOOOOO
  // OOOOOO
  __tgt_target_data_update(/*device_id*/ -1, /*arg_num*/ 1, &base, &begin,
                           sizes, types);
  // DEBUG: offset 24 len 12
  // DEBUG: offset 72 len 12

  }

  return 0;
}
