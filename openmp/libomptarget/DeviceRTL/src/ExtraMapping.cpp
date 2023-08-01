#include "Interface.h"
#include "Mapping.h"

#pragma omp declare target

#include "llvm/Frontend/OpenMP/OMPGridValues.h"

using namespace ompx::mapping;

namespace ompx::mapping {
namespace impl {

/// AMDGCN Implementation
///
///{
uint32_t __kmpc_impl_smid();
uint32_t getGenericModeMainThreadId();

#pragma omp begin declare variant match(device = {arch(amdgcn)})

// Partially derived fom hcc_detail/device_functions.h

// HW_ID Register bit structure
// WAVE_ID     3:0     Wave buffer slot number. 0-9.
// SIMD_ID     5:4     SIMD which the wave is assigned to within the CU.
// PIPE_ID     7:6     Pipeline from which the wave was dispatched.
// CU_ID       11:8    Compute Unit the wave is assigned to.
// SH_ID       12      Shader Array (within an SE) the wave is assigned to.
// SE_ID       14:13   Shader Engine the wave is assigned to.
// TG_ID       19:16   Thread-group ID
// VM_ID       23:20   Virtual Memory ID
// QUEUE_ID    26:24   Queue from which this wave was dispatched.
// STATE_ID    29:27   State ID (graphics only, not compute).
// ME_ID       31:30   Micro-engine ID.

enum {
  HW_ID = 4, // specify that the hardware register to read is HW_ID

  HW_ID_CU_ID_SIZE = 4,   // size of CU_ID field in bits
  HW_ID_CU_ID_OFFSET = 8, // offset of CU_ID from start of register

  HW_ID_SE_ID_SIZE = 2,    // sizeof SE_ID field in bits
  HW_ID_SE_ID_OFFSET = 13, // offset of SE_ID from start of register
};

// The s_getreg_b32 instruction, exposed as an intrinsic, takes a 16 bit
// immediate and returns a 32 bit value.
// The encoding of the immediate parameter is:
// ID           5:0     Which register to read from
// OFFSET       10:6    Range: 0..31
// WIDTH        15:11   Range: 1..32

// The asm equivalent is s_getreg_b32 %0, hwreg(HW_REG_HW_ID, Offset, Width)
// where hwreg forms a 16 bit immediate encoded by the assembler thus:
// uint64_t encodeHwreg(uint64_t Id, uint64_t Offset, uint64_t Width) {
//   return (Id << 0_) | (Offset << 6) | ((Width - 1) << 11);
// }
#define ENCODE_HWREG(WIDTH, OFF, REG) (REG | (OFF << 6) | ((WIDTH - 1) << 11))

// Note: The results can be changed by a context switch
// Return value in [0 2^SE_ID_SIZE * 2^CU_ID_SIZE), which is an upper
// bound on how many compute units are available. Some values in this
// range may never be returned if there are fewer than 2^CU_ID_SIZE CUs.

static uint32_t __kmpc_impl_smid() {
  uint32_t cu_id = __builtin_amdgcn_s_getreg(
      ENCODE_HWREG(HW_ID_CU_ID_SIZE, HW_ID_CU_ID_OFFSET, HW_ID));
  uint32_t se_id = __builtin_amdgcn_s_getreg(
      ENCODE_HWREG(HW_ID_SE_ID_SIZE, HW_ID_SE_ID_OFFSET, HW_ID));
  return (se_id << HW_ID_CU_ID_SIZE) + cu_id;
}

static uint32_t getGenericModeMainThreadId() {
  unsigned Mask =
      llvm::omp::getAMDGPUGridValues<__AMDGCN_WAVEFRONT_SIZE>().GV_Warp_Size -
      1;
  return (__kmpc_get_hardware_num_threads_in_block() - 1) & (~Mask);
}

#pragma omp end declare variant
///}

/// NVPTX Implementation
///
///{

#pragma omp begin declare variant match(                                       \
    device = {arch(nvptx, nvptx64)}, implementation = {extension(match_any)})

static uint32_t __kmpc_impl_smid() { return 0; }

static uint32_t getGenericModeMainThreadId() {
  unsigned Mask = mapping::getWarpSize() - 1;
  return (__kmpc_get_hardware_num_threads_in_block() - 1) & (~Mask);
}

#pragma omp end declare variant
///}

} // namespace impl
} // namespace ompx::mapping

extern "C" {

/// Extra API calls for ROCm

int omp_ext_get_warp_id() {
  int rc = ompx::mapping::getWarpIdInBlock();
  return rc;
}

int omp_ext_get_lane_id() {
  int rc = ompx::mapping::getThreadIdInWarp();
  return rc;
}

int omp_ext_get_smid() {
  int rc = impl::__kmpc_impl_smid();
  return rc;
}

int omp_ext_is_spmd_mode() {
  int rc = __kmpc_is_spmd_exec_mode();
  return rc != 0;
}

// The following extra call only works for generic mode
int omp_ext_get_master_thread_id() {
  // thread 0 is main thread in SPMD mode
  if (ompx::mapping::isSPMDMode())
    return 0;

  return impl::getGenericModeMainThreadId();
}

unsigned long long omp_ext_get_active_threads_mask() {
  unsigned long long rc = ompx::mapping::activemask();
  return rc;
}

} // end extern "C"

#pragma omp end declare target
