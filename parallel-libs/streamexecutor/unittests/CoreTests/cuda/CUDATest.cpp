//===-- CUDATest.cpp - Tests for CUDA platform ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the unit tests for CUDA platform code.
///
//===----------------------------------------------------------------------===//

#include "streamexecutor/StreamExecutor.h"

#include "gtest/gtest.h"

namespace {

namespace compilergen {
using SaxpyKernel =
    streamexecutor::Kernel<float, streamexecutor::GlobalDeviceMemory<float>,
                           streamexecutor::GlobalDeviceMemory<float>>;

const char *SaxpyPTX = R"(
  .version 4.3
  .target sm_20
  .address_size 64

  .visible .entry saxpy(.param .f32 A, .param .u64 X, .param .u64 Y) {
    .reg .f32 %AValue;
    .reg .f32 %XValue;
    .reg .f32 %YValue;
    .reg .f32 %Result;

    .reg .b64 %XBaseAddrGeneric;
    .reg .b64 %YBaseAddrGeneric;
    .reg .b64 %XBaseAddrGlobal;
    .reg .b64 %YBaseAddrGlobal;
    .reg .b64 %XAddr;
    .reg .b64 %YAddr;
    .reg .b64 %ThreadByteOffset;

    .reg .b32 %TID;

    ld.param.f32 %AValue, [A];
    ld.param.u64 %XBaseAddrGeneric, [X];
    ld.param.u64 %YBaseAddrGeneric, [Y];
    cvta.to.global.u64 %XBaseAddrGlobal, %XBaseAddrGeneric;
    cvta.to.global.u64 %YBaseAddrGlobal, %YBaseAddrGeneric;
    mov.u32 %TID, %tid.x;
    mul.wide.u32 %ThreadByteOffset, %TID, 4;
    add.s64 %XAddr, %ThreadByteOffset, %XBaseAddrGlobal;
    add.s64 %YAddr, %ThreadByteOffset, %YBaseAddrGlobal;
    ld.global.f32 %XValue, [%XAddr];
    ld.global.f32 %YValue, [%YAddr];
    fma.rn.f32 %Result, %AValue, %XValue, %YValue;
    st.global.f32 [%XAddr], %Result;
    ret;
  }
)";

static streamexecutor::MultiKernelLoaderSpec SaxpyLoaderSpec = []() {
  streamexecutor::MultiKernelLoaderSpec Spec;
  Spec.addCUDAPTXInMemory("saxpy", {{{2, 0}, SaxpyPTX}});
  return Spec;
}();

using SwapPairsKernel =
    streamexecutor::Kernel<streamexecutor::SharedDeviceMemory<int>,
                           streamexecutor::GlobalDeviceMemory<int>, int>;

const char *SwapPairsPTX = R"(
  .version 4.3
  .target sm_20
  .address_size 64

  .extern .shared .align 4 .b8 SwapSpace[];

  .visible .entry SwapPairs(.param .u64 InOut, .param .u32 InOutSize) {
    .reg .b64 %InOutGeneric;
    .reg .b32 %InOutSizeValue;

    .reg .b32 %LocalIndex;
    .reg .b32 %PartnerIndex;
    .reg .b32 %ThreadsPerBlock;
    .reg .b32 %BlockIndex;
    .reg .b32 %GlobalIndex;

    .reg .b32 %GlobalIndexBound;
    .reg .pred %GlobalIndexTooHigh;

    .reg .b64 %InOutGlobal;
    .reg .b64 %GlobalByteOffset;
    .reg .b64 %GlobalAddress;

    .reg .b32 %InitialValue;
    .reg .b32 %SwappedValue;

    .reg .b64 %SharedBaseAddr;
    .reg .b64 %LocalWriteByteOffset;
    .reg .b64 %LocalReadByteOffset;
    .reg .b64 %SharedWriteAddr;
    .reg .b64 %SharedReadAddr;

    ld.param.u64 %InOutGeneric, [InOut];
    ld.param.u32 %InOutSizeValue, [InOutSize];
    mov.u32 %LocalIndex, %tid.x;
    mov.u32 %ThreadsPerBlock, %ntid.x;
    mov.u32 %BlockIndex, %ctaid.x;
    mad.lo.s32 %GlobalIndex, %ThreadsPerBlock, %BlockIndex, %LocalIndex;
    and.b32 %GlobalIndexBound, %InOutSizeValue, -2;
    setp.ge.s32 %GlobalIndexTooHigh, %GlobalIndex, %GlobalIndexBound;
    @%GlobalIndexTooHigh bra END;

    cvta.to.global.u64 %InOutGlobal, %InOutGeneric;
    mul.wide.s32 %GlobalByteOffset, %GlobalIndex, 4;
    add.s64 %GlobalAddress, %InOutGlobal, %GlobalByteOffset;
    ld.global.u32 %InitialValue, [%GlobalAddress];
    mul.wide.s32 %LocalWriteByteOffset, %LocalIndex, 4;
    mov.u64 %SharedBaseAddr, SwapSpace;
    add.s64 %SharedWriteAddr, %SharedBaseAddr, %LocalWriteByteOffset;
    st.shared.u32 [%SharedWriteAddr], %InitialValue;
    bar.sync 0;
    xor.b32 %PartnerIndex, %LocalIndex, 1;
    mul.wide.s32 %LocalReadByteOffset, %PartnerIndex, 4;
    add.s64 %SharedReadAddr, %SharedBaseAddr, %LocalReadByteOffset;
    ld.shared.u32 %SwappedValue, [%SharedReadAddr];
    st.global.u32 [%GlobalAddress], %SwappedValue;

  END:
    ret;
  }
)";

static streamexecutor::MultiKernelLoaderSpec SwapPairsLoaderSpec = []() {
  streamexecutor::MultiKernelLoaderSpec Spec;
  Spec.addCUDAPTXInMemory("SwapPairs", {{{2, 0}, SwapPairsPTX}});
  return Spec;
}();
} // namespace compilergen

namespace se = ::streamexecutor;
namespace cg = ::compilergen;

class CUDATest : public ::testing::Test {
public:
  CUDATest()
      : Platform(getOrDie(se::PlatformManager::getPlatformByName("CUDA"))),
        Device(getOrDie(Platform->getDevice(0))),
        Stream(getOrDie(Device.createStream())) {}

  se::Platform *Platform;
  se::Device Device;
  se::Stream Stream;
};

TEST_F(CUDATest, Saxpy) {
  float A = 42.0f;
  std::vector<float> HostX = {0, 1, 2, 3};
  std::vector<float> HostY = {4, 5, 6, 7};
  size_t ArraySize = HostX.size();

  cg::SaxpyKernel Kernel =
      getOrDie(Device.createKernel<cg::SaxpyKernel>(cg::SaxpyLoaderSpec));

  se::RegisteredHostMemory<float> RegisteredX =
      getOrDie(Device.registerHostMemory<float>(HostX));
  se::RegisteredHostMemory<float> RegisteredY =
      getOrDie(Device.registerHostMemory<float>(HostY));

  se::GlobalDeviceMemory<float> X =
      getOrDie(Device.allocateDeviceMemory<float>(ArraySize));
  se::GlobalDeviceMemory<float> Y =
      getOrDie(Device.allocateDeviceMemory<float>(ArraySize));

  Stream.thenCopyH2D(RegisteredX, X)
      .thenCopyH2D(RegisteredY, Y)
      .thenLaunch(ArraySize, 1, Kernel, A, X, Y)
      .thenCopyD2H(X, RegisteredX);
  se::dieIfError(Stream.blockHostUntilDone());

  std::vector<float> ExpectedX = {4, 47, 90, 133};
  EXPECT_EQ(ExpectedX, HostX);
}

TEST_F(CUDATest, DynamicSharedMemory) {
  std::vector<int> HostPairs = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::vector<int> HostResult(HostPairs.size(), 0);
  int ArraySize = HostPairs.size();

  cg::SwapPairsKernel Kernel = getOrDie(
      Device.createKernel<cg::SwapPairsKernel>(cg::SwapPairsLoaderSpec));

  se::RegisteredHostMemory<int> RegisteredPairs =
      getOrDie(Device.registerHostMemory<int>(HostPairs));
  se::RegisteredHostMemory<int> RegisteredResult =
      getOrDie(Device.registerHostMemory<int>(HostResult));

  se::GlobalDeviceMemory<int> Pairs =
      getOrDie(Device.allocateDeviceMemory<int>(ArraySize));
  auto SharedMemory =
      se::SharedDeviceMemory<int>::makeFromElementCount(ArraySize);

  Stream.thenCopyH2D(RegisteredPairs, Pairs)
      .thenLaunch(ArraySize, 1, Kernel, SharedMemory, Pairs, ArraySize)
      .thenCopyD2H(Pairs, RegisteredResult);
  se::dieIfError(Stream.blockHostUntilDone());

  std::vector<int> ExpectedPairs = {1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10};
  EXPECT_EQ(ExpectedPairs, HostResult);
}

} // namespace
