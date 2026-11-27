// clang-format off
// RUN: %libomptarget-compilexx-generic %omptest-flags && %libomptarget-run-generic | %fcheck-generic
// REQUIRES: omptest
// REQUIRES: gpu
// clang-format on

// A rectangular copy is reported to tools as a single
// ompt_target_data_transfer_rect operation covering the whole region,
// whether it is done by one liboffload call (2D/3D) or row by row (4D).

#include "OmptTester.h"

#include <omp.h>

using namespace omptest;

namespace {
constexpr size_t ES = sizeof(int);

// Copy \p NumDims dimensions of \p Volume ints starting at offset 0 of
// buffers whose extents are \p Dims.
void copyRect(void *Dst, const void *Src, int NumDims, const size_t *Volume,
              const size_t *Dims, int DstDev, int SrcDev) {
  size_t Offsets[4] = {0, 0, 0, 0};
  ASSERT_EQ(omp_target_memcpy_rect(Dst, Src, ES, NumDims, Volume, Offsets,
                                   Offsets, Dims, Dims, DstDev, SrcDev),
            0);
}
} // namespace

TEST(OmptestMemcpyRect, SingleEventPerCopy) {
  int Dev = omp_get_default_device();
  int Host = omp_get_initial_device();

  // 3D takes the single-call path, 4D the row-by-row one.
  const size_t Dims3[3] = {4, 4, 4}, Vol3[3] = {2, 3, 4};
  const size_t Dims4[4] = {2, 3, 4, 5}, Vol4[4] = {2, 2, 3, 4};
  const size_t Bytes3 = 2 * 3 * 4 * ES, Bytes4 = 2 * 2 * 3 * 4 * ES;

  int HostBuf[4 * 4 * 4 + 120] = {};
  // Initialize the device first so that its setup events are not observed.
  void *DevBuf = omp_target_alloc(sizeof(HostBuf), Dev);
  omp_target_free(DevBuf, Dev);

  // Strict mode: any event beyond these, such as a transfer for an individual
  // row, fails the test.
  OMPT_ASSERT_SEQUENCE_MODE_STRICT()
  OMPT_ASSERT_SEQUENCE(TargetDataOp, ompt_target_data_alloc, sizeof(HostBuf))
  OMPT_ASSERT_SEQUENCE(TargetDataOp, ompt_target_data_alloc, sizeof(HostBuf))
  // Each copy is exactly one rect event, reported in this order.
  OMPT_ASSERT_SEQUENCE(TargetDataOp, ompt_target_data_transfer_rect, Bytes3)
  OMPT_ASSERT_SEQUENCE(TargetDataOp, ompt_target_data_transfer_rect, Bytes3)
  OMPT_ASSERT_SEQUENCE(TargetDataOp, ompt_target_data_transfer_rect, Bytes3)
  OMPT_ASSERT_SEQUENCE(TargetDataOp, ompt_target_data_transfer_rect, Bytes4)
  OMPT_ASSERT_SEQUENCE(TargetDataOp, ompt_target_data_transfer_rect, Bytes4)
  OMPT_ASSERT_SEQUENCE(TargetDataOp, ompt_target_data_delete, 0)
  OMPT_ASSERT_SEQUENCE(TargetDataOp, ompt_target_data_delete, 0)

  DevBuf = omp_target_alloc(sizeof(HostBuf), Dev);
  void *DevBuf2 = omp_target_alloc(sizeof(HostBuf), Dev);
  ASSERT_NE(DevBuf, nullptr);
  ASSERT_NE(DevBuf2, nullptr);

  copyRect(DevBuf, HostBuf, 3, Vol3, Dims3, Dev, Host);  // H2D, 3D
  copyRect(DevBuf2, DevBuf, 3, Vol3, Dims3, Dev, Dev);   // D2D, 3D
  copyRect(HostBuf, DevBuf2, 3, Vol3, Dims3, Host, Dev); // D2H, 3D
  copyRect(DevBuf, HostBuf, 4, Vol4, Dims4, Dev, Host);  // H2D, 4D
  copyRect(HostBuf, DevBuf, 4, Vol4, Dims4, Host, Dev);  // D2H, 4D

  omp_target_free(DevBuf, Dev);
  omp_target_free(DevBuf2, Dev);
}

OMPTEST_TESTSUITE_MAIN()

// Guards against a vacuous pass.
// CHECK: [  PASSED  ] 1 test.
