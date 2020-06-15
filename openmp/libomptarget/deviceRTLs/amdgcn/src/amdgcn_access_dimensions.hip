#include "amdgcn_interface.h"
#include "target_impl.h"

// Subset of AQL kernel dispatch package format
// Bits         Field Name              Description
// 47:32        workgroup_size_x        x dimension of work-group
// 63:48        workgroup_size_y        y dimension of work-group
// 79:64        workgroup_size_z        z dimension of work-group
// 127:96       grid_size_x             x dimension of grid
// 159:128      grid_size_y             y dimension of grid
// 191:160      grid_size_z             z dimension of grid

namespace {

DEVICE char *p(uint32_t offset) {
  return (char *)__builtin_amdgcn_dispatch_ptr() + offset;
}


DEVICE uint16_t workgroup_size_x() { return __builtin_amdgcn_workgroup_size_x(); }
DEVICE uint16_t workgroup_size_y() { return __builtin_amdgcn_workgroup_size_y(); }
DEVICE uint16_t workgroup_size_z() { return __builtin_amdgcn_workgroup_size_z(); }
DEVICE uint32_t grid_size_x() { return *(uint32_t *)p(96 / 8); }
DEVICE uint32_t grid_size_y() { return *(uint32_t *)p(128 / 8); }
DEVICE uint32_t grid_size_z() { return *(uint32_t *)p(160 / 8); }

DEVICE uint32_t get_grid_dim(uint32_t n, uint16_t d) {
  uint32_t q = n / d;
  return q + (n > q * d);
}
DEVICE uint32_t get_workgroup_dim(uint32_t group_id, uint32_t grid_size,
                              uint16_t group_size) {
  uint32_t r = grid_size - group_id * group_size;
  return (r < group_size) ? r : group_size;
}

} // namespace

EXTERN uint32_t __kmpc_amd_grid_dim_x() {
  return get_grid_dim(grid_size_x(), workgroup_size_x());
}
EXTERN uint32_t __kmpc_amd_grid_dim_y() {
  return get_grid_dim(grid_size_y(), workgroup_size_y());
}
EXTERN uint32_t __kmpc_amd_grid_dim_z() {
  return get_grid_dim(grid_size_z(), workgroup_size_z());
}

EXTERN uint32_t __kmpc_amd_workgroup_dim_x() {
  return get_workgroup_dim(__builtin_amdgcn_workgroup_id_x(), grid_size_x(),
                       workgroup_size_x());
}

EXTERN uint32_t __kmpc_amd_workgroup_dim_y() {
  return get_workgroup_dim(__builtin_amdgcn_workgroup_id_y(), grid_size_y(),
                       workgroup_size_y());
}
EXTERN uint32_t __kmpc_amd_workgroup_dim_z() {
  return get_workgroup_dim(__builtin_amdgcn_workgroup_id_z(), grid_size_z(),
                       workgroup_size_z());
}
