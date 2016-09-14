//===-- KernelSpecTest.cpp - Tests for KernelSpec -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the unit tests for the code in KernelSpec.
///
//===----------------------------------------------------------------------===//

#include "streamexecutor/KernelSpec.h"

#include "gtest/gtest.h"

namespace {

namespace se = ::streamexecutor;

TEST(CUDAPTXInMemorySpec, NoCode) {
  se::CUDAPTXInMemorySpec Spec("KernelName", {});
  EXPECT_EQ("KernelName", Spec.getKernelName());
  EXPECT_EQ(nullptr, Spec.getCode(1, 0));
}

TEST(CUDAPTXInMemorySpec, SingleComputeCapability) {
  const char *PTXCodeString = "Dummy PTX code";
  se::CUDAPTXInMemorySpec Spec("KernelName", {{{1, 0}, PTXCodeString}});
  EXPECT_EQ("KernelName", Spec.getKernelName());
  EXPECT_EQ(nullptr, Spec.getCode(0, 5));
  EXPECT_EQ(PTXCodeString, Spec.getCode(1, 0));
  EXPECT_EQ(PTXCodeString, Spec.getCode(2, 0));
}

TEST(CUDAPTXInMemorySpec, TwoComputeCapabilities) {
  const char *PTXCodeString10 = "Dummy PTX code 10";
  const char *PTXCodeString30 = "Dummy PTX code 30";
  se::CUDAPTXInMemorySpec Spec(
      "KernelName", {{{1, 0}, PTXCodeString10}, {{3, 0}, PTXCodeString30}});
  EXPECT_EQ("KernelName", Spec.getKernelName());
  EXPECT_EQ(nullptr, Spec.getCode(0, 5));
  EXPECT_EQ(PTXCodeString10, Spec.getCode(1, 0));
  EXPECT_EQ(PTXCodeString30, Spec.getCode(3, 0));
  EXPECT_EQ(PTXCodeString10, Spec.getCode(2, 0));
}

TEST(CUDAFatbinInMemorySpec, BasicUsage) {
  const char *FatbinBytes = "Dummy fatbin bytes";
  se::CUDAFatbinInMemorySpec Spec("KernelName", FatbinBytes);
  EXPECT_EQ("KernelName", Spec.getKernelName());
  EXPECT_EQ(FatbinBytes, Spec.getBytes());
}

TEST(OpenCLTextInMemorySpec, BasicUsage) {
  const char *OpenCLText = "Dummy OpenCL text";
  se::OpenCLTextInMemorySpec Spec("KernelName", OpenCLText);
  EXPECT_EQ("KernelName", Spec.getKernelName());
  EXPECT_EQ(OpenCLText, Spec.getText());
}

TEST(MultiKernelLoaderSpec, NoCode) {
  se::MultiKernelLoaderSpec MultiSpec;
  EXPECT_FALSE(MultiSpec.hasCUDAPTXInMemory());
  EXPECT_FALSE(MultiSpec.hasCUDAFatbinInMemory());
  EXPECT_FALSE(MultiSpec.hasOpenCLTextInMemory());

  EXPECT_DEBUG_DEATH(MultiSpec.getCUDAPTXInMemory(),
                     "getting spec that is not present");
  EXPECT_DEBUG_DEATH(MultiSpec.getCUDAFatbinInMemory(),
                     "getting spec that is not present");
  EXPECT_DEBUG_DEATH(MultiSpec.getOpenCLTextInMemory(),
                     "getting spec that is not present");
}

TEST(MultiKernelLoaderSpec, Registration) {
  se::MultiKernelLoaderSpec MultiSpec;
  const char *KernelName = "KernelName";
  const char *PTXCodeString = "Dummy PTX code";
  const char *FatbinBytes = "Dummy fatbin bytes";
  const char *OpenCLText = "Dummy OpenCL text";

  MultiSpec.addCUDAPTXInMemory(KernelName, {{{1, 0}, PTXCodeString}})
      .addCUDAFatbinInMemory(KernelName, FatbinBytes)
      .addOpenCLTextInMemory(KernelName, OpenCLText);

  EXPECT_TRUE(MultiSpec.hasCUDAPTXInMemory());
  EXPECT_TRUE(MultiSpec.hasCUDAFatbinInMemory());
  EXPECT_TRUE(MultiSpec.hasOpenCLTextInMemory());

  EXPECT_EQ(KernelName, MultiSpec.getCUDAPTXInMemory().getKernelName());
  EXPECT_EQ(nullptr, MultiSpec.getCUDAPTXInMemory().getCode(0, 5));
  EXPECT_EQ(PTXCodeString, MultiSpec.getCUDAPTXInMemory().getCode(1, 0));
  EXPECT_EQ(PTXCodeString, MultiSpec.getCUDAPTXInMemory().getCode(2, 0));

  EXPECT_EQ(KernelName, MultiSpec.getCUDAFatbinInMemory().getKernelName());
  EXPECT_EQ(FatbinBytes, MultiSpec.getCUDAFatbinInMemory().getBytes());

  EXPECT_EQ(KernelName, MultiSpec.getOpenCLTextInMemory().getKernelName());
  EXPECT_EQ(OpenCLText, MultiSpec.getOpenCLTextInMemory().getText());
}

TEST(MultiKernelLoaderSpec, RegisterTwice) {
  se::MultiKernelLoaderSpec MultiSpec;
  const char *KernelName = "KernelName";
  const char *FatbinBytes = "Dummy fatbin bytes";

  MultiSpec.addCUDAFatbinInMemory(KernelName, FatbinBytes);

  EXPECT_DEBUG_DEATH(MultiSpec.addCUDAFatbinInMemory(KernelName, FatbinBytes),
                     "illegal loader spec overwrite");
}

TEST(MultiKernelLoaderSpec, ConflictingKernelNames) {
  se::MultiKernelLoaderSpec MultiSpec;
  const char *KernelNameA = "KernelName";
  std::string KernelNameB = KernelNameA;
  const char *PTXCodeString = "Dummy PTX code";
  const char *FatbinBytes = "Dummy fatbin bytes";

  // Check that names don't conflict if they are equivalent strings in different
  // locations.
  MultiSpec.addCUDAPTXInMemory(KernelNameA, {{{1, 0}, PTXCodeString}})
      .addCUDAFatbinInMemory(KernelNameB, FatbinBytes);

  const char *OtherKernelName = "OtherKernelName";
  const char *OpenCLText = "Dummy OpenCL text";
  EXPECT_DEBUG_DEATH(
      MultiSpec.addOpenCLTextInMemory(OtherKernelName, OpenCLText),
      "different kernel names in one MultiKernelLoaderSpec");
}

} // namespace
