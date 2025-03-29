//===----------- Triple.cpp - Triple unit tests ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/TargetParser/Triple.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/VersionTuple.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(TripleTest, BasicParsing) {
  Triple T;

  T = Triple("");
  EXPECT_EQ("", T.getArchName().str());
  EXPECT_EQ("", T.getVendorName().str());
  EXPECT_EQ("", T.getOSName().str());
  EXPECT_EQ("", T.getEnvironmentName().str());

  T = Triple("-");
  EXPECT_EQ("", T.getArchName().str());
  EXPECT_EQ("", T.getVendorName().str());
  EXPECT_EQ("", T.getOSName().str());
  EXPECT_EQ("", T.getEnvironmentName().str());

  T = Triple("--");
  EXPECT_EQ("", T.getArchName().str());
  EXPECT_EQ("", T.getVendorName().str());
  EXPECT_EQ("", T.getOSName().str());
  EXPECT_EQ("", T.getEnvironmentName().str());

  T = Triple("---");
  EXPECT_EQ("", T.getArchName().str());
  EXPECT_EQ("", T.getVendorName().str());
  EXPECT_EQ("", T.getOSName().str());
  EXPECT_EQ("", T.getEnvironmentName().str());

  T = Triple("----");
  EXPECT_EQ("", T.getArchName().str());
  EXPECT_EQ("", T.getVendorName().str());
  EXPECT_EQ("", T.getOSName().str());
  EXPECT_EQ("-", T.getEnvironmentName().str());

  T = Triple("a");
  EXPECT_EQ("a", T.getArchName().str());
  EXPECT_EQ("", T.getVendorName().str());
  EXPECT_EQ("", T.getOSName().str());
  EXPECT_EQ("", T.getEnvironmentName().str());

  T = Triple("a-b");
  EXPECT_EQ("a", T.getArchName().str());
  EXPECT_EQ("b", T.getVendorName().str());
  EXPECT_EQ("", T.getOSName().str());
  EXPECT_EQ("", T.getEnvironmentName().str());

  T = Triple("a-b-c");
  EXPECT_EQ("a", T.getArchName().str());
  EXPECT_EQ("b", T.getVendorName().str());
  EXPECT_EQ("c", T.getOSName().str());
  EXPECT_EQ("", T.getEnvironmentName().str());

  T = Triple("a-b-c-d");
  EXPECT_EQ("a", T.getArchName().str());
  EXPECT_EQ("b", T.getVendorName().str());
  EXPECT_EQ("c", T.getOSName().str());
  EXPECT_EQ("d", T.getEnvironmentName().str());
}

TEST(TripleTest, ParsedIDs) {
  Triple T;

  T = Triple("i386-apple-darwin");
  EXPECT_EQ(Triple::x86, T.getArch());
  EXPECT_EQ(Triple::Apple, T.getVendor());
  EXPECT_EQ(Triple::Darwin, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("i386-pc-elfiamcu");
  EXPECT_EQ(Triple::x86, T.getArch());
  EXPECT_EQ(Triple::PC, T.getVendor());
  EXPECT_EQ(Triple::ELFIAMCU, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("i386-pc-hurd-gnu");
  EXPECT_EQ(Triple::x86, T.getArch());
  EXPECT_EQ(Triple::PC, T.getVendor());
  EXPECT_EQ(Triple::Hurd, T.getOS());
  EXPECT_EQ(Triple::GNU, T.getEnvironment());

  T = Triple("i686-pc-linux-gnu");
  EXPECT_EQ(Triple::x86, T.getArch());
  EXPECT_EQ(Triple::PC, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::GNU, T.getEnvironment());
  EXPECT_FALSE(T.isTime64ABI());

  T = Triple("x86_64-pc-linux-gnu");
  EXPECT_EQ(Triple::x86_64, T.getArch());
  EXPECT_EQ(Triple::PC, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::GNU, T.getEnvironment());

  T = Triple("x86_64-pc-linux-musl");
  EXPECT_EQ(Triple::x86_64, T.getArch());
  EXPECT_EQ(Triple::PC, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::Musl, T.getEnvironment());

  T = Triple("x86_64-pc-linux-muslx32");
  EXPECT_EQ(Triple::x86_64, T.getArch());
  EXPECT_EQ(Triple::PC, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::MuslX32, T.getEnvironment());

  T = Triple("x86_64-pc-hurd-gnu");
  EXPECT_EQ(Triple::x86_64, T.getArch());
  EXPECT_EQ(Triple::PC, T.getVendor());
  EXPECT_EQ(Triple::Hurd, T.getOS());
  EXPECT_EQ(Triple::GNU, T.getEnvironment());

  T = Triple("arm-unknown-linux-android16");
  EXPECT_EQ(Triple::arm, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::Android, T.getEnvironment());

  T = Triple("aarch64-unknown-linux-android21");
  EXPECT_EQ(Triple::aarch64, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::Android, T.getEnvironment());

  // PS4 has two spellings for the vendor.
  T = Triple("x86_64-scei-ps4");
  EXPECT_EQ(Triple::x86_64, T.getArch());
  EXPECT_EQ(Triple::SCEI, T.getVendor());
  EXPECT_EQ(Triple::PS4, T.getOS());

  T = Triple("x86_64-sie-ps4");
  EXPECT_EQ(Triple::x86_64, T.getArch());
  EXPECT_EQ(Triple::SCEI, T.getVendor());
  EXPECT_EQ(Triple::PS4, T.getOS());

  T = Triple("x86_64-sie-ps5");
  EXPECT_EQ(Triple::x86_64, T.getArch());
  EXPECT_EQ(Triple::SCEI, T.getVendor());
  EXPECT_EQ(Triple::PS5, T.getOS());

  T = Triple("powerpc-ibm-aix");
  EXPECT_EQ(Triple::ppc, T.getArch());
  EXPECT_EQ(Triple::IBM, T.getVendor());
  EXPECT_EQ(Triple::AIX, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("powerpc64-ibm-aix");
  EXPECT_EQ(Triple::ppc64, T.getArch());
  EXPECT_EQ(Triple::IBM, T.getVendor());
  EXPECT_EQ(Triple::AIX, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("powerpc-dunno-notsure");
  EXPECT_EQ(Triple::ppc, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("powerpcspe-unknown-freebsd");
  EXPECT_EQ(Triple::ppc, T.getArch());
  EXPECT_EQ(Triple::PPCSubArch_spe, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::FreeBSD, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("s390x-ibm-zos");
  EXPECT_EQ(Triple::systemz, T.getArch());
  EXPECT_EQ(Triple::IBM, T.getVendor());
  EXPECT_EQ(Triple::ZOS, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("systemz-ibm-zos");
  EXPECT_EQ(Triple::systemz, T.getArch());
  EXPECT_EQ(Triple::IBM, T.getVendor());
  EXPECT_EQ(Triple::ZOS, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("arm-none-none-eabi");
  EXPECT_EQ(Triple::arm, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());
  EXPECT_EQ(Triple::EABI, T.getEnvironment());
  EXPECT_FALSE(T.isHardFloatABI());

  T = Triple("arm-none-none-eabihf");
  EXPECT_EQ(Triple::arm, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());
  EXPECT_EQ(Triple::EABIHF, T.getEnvironment());
  EXPECT_TRUE(T.isHardFloatABI());

  T = Triple("arm-none-linux-musleabi");
  EXPECT_EQ(Triple::arm, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::MuslEABI, T.getEnvironment());
  EXPECT_FALSE(T.isHardFloatABI());

  T = Triple("arm-none-linux-musleabihf");
  EXPECT_EQ(Triple::arm, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::MuslEABIHF, T.getEnvironment());
  EXPECT_TRUE(T.isHardFloatABI());

  T = Triple("armv6hl-none-linux-gnueabi");
  EXPECT_EQ(Triple::arm, T.getArch());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::GNUEABI, T.getEnvironment());
  EXPECT_FALSE(T.isTime64ABI());
  EXPECT_FALSE(T.isHardFloatABI());

  T = Triple("armv7hl-none-linux-gnueabi");
  EXPECT_EQ(Triple::arm, T.getArch());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::GNUEABI, T.getEnvironment());
  EXPECT_FALSE(T.isTime64ABI());
  EXPECT_FALSE(T.isHardFloatABI());

  T = Triple("armv7hl-none-linux-gnueabihf");
  EXPECT_EQ(Triple::arm, T.getArch());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::GNUEABIHF, T.getEnvironment());
  EXPECT_FALSE(T.isTime64ABI());
  EXPECT_TRUE(T.isHardFloatABI());

  T = Triple("amdil-unknown-unknown");
  EXPECT_EQ(Triple::amdil, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());

  T = Triple("amdil64-unknown-unknown");
  EXPECT_EQ(Triple::amdil64, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());

  T = Triple("hsail-unknown-unknown");
  EXPECT_EQ(Triple::hsail, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());

  T = Triple("hsail64-unknown-unknown");
  EXPECT_EQ(Triple::hsail64, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());

  T = Triple("sparcel-unknown-unknown");
  EXPECT_EQ(Triple::sparcel, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());

  T = Triple("spir-unknown-unknown");
  EXPECT_EQ(Triple::spir, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());

  T = Triple("spir64-unknown-unknown");
  EXPECT_EQ(Triple::spir64, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());

  T = Triple("spirv32-unknown-unknown");
  EXPECT_EQ(Triple::spirv32, T.getArch());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());

  T = Triple("spirv32v1.0-unknown-unknown");
  EXPECT_EQ(Triple::spirv32, T.getArch());
  EXPECT_EQ(Triple::SPIRVSubArch_v10, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());

  T = Triple("spirv32v1.1-unknown-unknown");
  EXPECT_EQ(Triple::spirv32, T.getArch());
  EXPECT_EQ(Triple::SPIRVSubArch_v11, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());

  T = Triple("spirv32v1.2-unknown-unknown");
  EXPECT_EQ(Triple::spirv32, T.getArch());
  EXPECT_EQ(Triple::SPIRVSubArch_v12, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());

  T = Triple("spirv32v1.3-unknown-unknown");
  EXPECT_EQ(Triple::spirv32, T.getArch());
  EXPECT_EQ(Triple::SPIRVSubArch_v13, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());

  T = Triple("spirv32v1.4-unknown-unknown");
  EXPECT_EQ(Triple::spirv32, T.getArch());
  EXPECT_EQ(Triple::SPIRVSubArch_v14, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());

  T = Triple("spirv32v1.5-unknown-unknown");
  EXPECT_EQ(Triple::spirv32, T.getArch());
  EXPECT_EQ(Triple::SPIRVSubArch_v15, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());

  T = Triple("spirv64-unknown-unknown");
  EXPECT_EQ(Triple::spirv64, T.getArch());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());

  T = Triple("spirv64v1.0-unknown-unknown");
  EXPECT_EQ(Triple::spirv64, T.getArch());
  EXPECT_EQ(Triple::SPIRVSubArch_v10, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());

  T = Triple("spirv64v1.1-unknown-unknown");
  EXPECT_EQ(Triple::spirv64, T.getArch());
  EXPECT_EQ(Triple::SPIRVSubArch_v11, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());

  T = Triple("spirv64v1.2-unknown-unknown");
  EXPECT_EQ(Triple::spirv64, T.getArch());
  EXPECT_EQ(Triple::SPIRVSubArch_v12, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());

  T = Triple("spirv64v1.3-unknown-unknown");
  EXPECT_EQ(Triple::spirv64, T.getArch());
  EXPECT_EQ(Triple::SPIRVSubArch_v13, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());

  T = Triple("spirv64v1.4-unknown-unknown");
  EXPECT_EQ(Triple::spirv64, T.getArch());
  EXPECT_EQ(Triple::SPIRVSubArch_v14, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());

  T = Triple("spirv64v1.5-unknown-unknown");
  EXPECT_EQ(Triple::spirv64, T.getArch());
  EXPECT_EQ(Triple::SPIRVSubArch_v15, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());

  T = Triple("spirv-unknown-vulkan-pixel");
  EXPECT_EQ(Triple::spirv, T.getArch());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Vulkan, T.getOS());
  EXPECT_EQ(VersionTuple(1, 2), T.getVulkanVersion());
  EXPECT_EQ(Triple::Pixel, T.getEnvironment());

  T = Triple("spirv-unknown-vulkan-vertex");
  EXPECT_EQ(Triple::spirv, T.getArch());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Vulkan, T.getOS());
  EXPECT_EQ(VersionTuple(1, 2), T.getVulkanVersion());
  EXPECT_EQ(Triple::Vertex, T.getEnvironment());

  T = Triple("spirv-unknown-vulkan-geometry");
  EXPECT_EQ(Triple::spirv, T.getArch());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Vulkan, T.getOS());
  EXPECT_EQ(VersionTuple(1, 2), T.getVulkanVersion());
  EXPECT_EQ(Triple::Geometry, T.getEnvironment());

  T = Triple("spirv-unknown-vulkan-library");
  EXPECT_EQ(Triple::spirv, T.getArch());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Vulkan, T.getOS());
  EXPECT_EQ(VersionTuple(1, 2), T.getVulkanVersion());
  EXPECT_EQ(Triple::Library, T.getEnvironment());

  T = Triple("spirv-unknown-vulkan-raygeneration");
  EXPECT_EQ(Triple::spirv, T.getArch());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Vulkan, T.getOS());
  EXPECT_EQ(VersionTuple(1, 2), T.getVulkanVersion());
  EXPECT_EQ(Triple::RayGeneration, T.getEnvironment());

  T = Triple("spirv-unknown-vulkan-intersection");
  EXPECT_EQ(Triple::spirv, T.getArch());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Vulkan, T.getOS());
  EXPECT_EQ(VersionTuple(1, 2), T.getVulkanVersion());
  EXPECT_EQ(Triple::Intersection, T.getEnvironment());

  T = Triple("spirv-unknown-vulkan-anyhit");
  EXPECT_EQ(Triple::spirv, T.getArch());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Vulkan, T.getOS());
  EXPECT_EQ(VersionTuple(1, 2), T.getVulkanVersion());
  EXPECT_EQ(Triple::AnyHit, T.getEnvironment());

  T = Triple("spirv-unknown-vulkan-closesthit");
  EXPECT_EQ(Triple::spirv, T.getArch());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Vulkan, T.getOS());
  EXPECT_EQ(VersionTuple(1, 2), T.getVulkanVersion());
  EXPECT_EQ(Triple::ClosestHit, T.getEnvironment());

  T = Triple("spirv-unknown-vulkan-miss");
  EXPECT_EQ(Triple::spirv, T.getArch());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Vulkan, T.getOS());
  EXPECT_EQ(VersionTuple(1, 2), T.getVulkanVersion());
  EXPECT_EQ(Triple::Miss, T.getEnvironment());

  T = Triple("spirv-unknown-vulkan-callable");
  EXPECT_EQ(Triple::spirv, T.getArch());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Vulkan, T.getOS());
  EXPECT_EQ(VersionTuple(1, 2), T.getVulkanVersion());
  EXPECT_EQ(Triple::Callable, T.getEnvironment());

  T = Triple("spirv-unknown-vulkan-mesh");
  EXPECT_EQ(Triple::spirv, T.getArch());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Vulkan, T.getOS());
  EXPECT_EQ(VersionTuple(1, 2), T.getVulkanVersion());
  EXPECT_EQ(Triple::Mesh, T.getEnvironment());

  T = Triple("spirv-unknown-vulkan-amplification");
  EXPECT_EQ(Triple::spirv, T.getArch());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Vulkan, T.getOS());
  EXPECT_EQ(VersionTuple(1, 2), T.getVulkanVersion());
  EXPECT_EQ(Triple::Amplification, T.getEnvironment());

  T = Triple("spirv1.5-unknown-vulkan1.2-compute");
  EXPECT_EQ(Triple::spirv, T.getArch());
  EXPECT_EQ(Triple::SPIRVSubArch_v15, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Vulkan, T.getOS());
  EXPECT_EQ(VersionTuple(1, 2), T.getVulkanVersion());
  EXPECT_EQ(Triple::Compute, T.getEnvironment());

  T = Triple("spirv1.6-unknown-vulkan1.3-compute");
  EXPECT_EQ(Triple::spirv, T.getArch());
  EXPECT_EQ(Triple::SPIRVSubArch_v16, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Vulkan, T.getOS());
  EXPECT_EQ(VersionTuple(1, 3), T.getVulkanVersion());
  EXPECT_EQ(Triple::Compute, T.getEnvironment());

  T = Triple("dxilv1.0--shadermodel6.0-pixel");
  EXPECT_EQ(Triple::dxil, T.getArch());
  EXPECT_EQ(Triple::DXILSubArch_v1_0, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::ShaderModel, T.getOS());
  EXPECT_EQ(VersionTuple(1, 0), T.getDXILVersion());
  EXPECT_EQ(Triple::Pixel, T.getEnvironment());

  T = Triple("dxilv1.1--shadermodel6.1-vertex");
  EXPECT_EQ(Triple::dxil, T.getArch());
  EXPECT_EQ(Triple::DXILSubArch_v1_1, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::ShaderModel, T.getOS());
  EXPECT_EQ(VersionTuple(1, 1), T.getDXILVersion());
  EXPECT_EQ(Triple::Vertex, T.getEnvironment());

  T = Triple("dxilv1.2--shadermodel6.2-geometry");
  EXPECT_EQ(Triple::dxil, T.getArch());
  EXPECT_EQ(Triple::DXILSubArch_v1_2, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::ShaderModel, T.getOS());
  EXPECT_EQ(VersionTuple(1, 2), T.getDXILVersion());
  EXPECT_EQ(Triple::Geometry, T.getEnvironment());

  T = Triple("dxilv1.3--shadermodel6.3-library");
  EXPECT_EQ(Triple::dxil, T.getArch());
  EXPECT_EQ(Triple::DXILSubArch_v1_3, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::ShaderModel, T.getOS());
  EXPECT_EQ(VersionTuple(1, 3), T.getDXILVersion());
  EXPECT_EQ(Triple::Library, T.getEnvironment());

  T = Triple("dxilv1.4--shadermodel6.4-hull");
  EXPECT_EQ(Triple::dxil, T.getArch());
  EXPECT_EQ(Triple::DXILSubArch_v1_4, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::ShaderModel, T.getOS());
  EXPECT_EQ(VersionTuple(1, 4), T.getDXILVersion());
  EXPECT_EQ(Triple::Hull, T.getEnvironment());

  T = Triple("dxilv1.5--shadermodel6.5-domain");
  EXPECT_EQ(Triple::dxil, T.getArch());
  EXPECT_EQ(Triple::DXILSubArch_v1_5, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::ShaderModel, T.getOS());
  EXPECT_EQ(VersionTuple(1, 5), T.getDXILVersion());
  EXPECT_EQ(Triple::Domain, T.getEnvironment());

  T = Triple("dxilv1.6--shadermodel6.6-compute");
  EXPECT_EQ(Triple::dxil, T.getArch());
  EXPECT_EQ(Triple::DXILSubArch_v1_6, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::ShaderModel, T.getOS());
  EXPECT_EQ(VersionTuple(1, 6), T.getDXILVersion());
  EXPECT_EQ(Triple::Compute, T.getEnvironment());

  T = Triple("dxilv1.7-unknown-shadermodel6.7-mesh");
  EXPECT_EQ(Triple::dxil, T.getArch());
  EXPECT_EQ(Triple::DXILSubArch_v1_7, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::ShaderModel, T.getOS());
  EXPECT_EQ(VersionTuple(1, 7), T.getDXILVersion());
  EXPECT_EQ(Triple::Mesh, T.getEnvironment());

  T = Triple("dxilv1.8-unknown-shadermodel6.8-amplification");
  EXPECT_EQ(Triple::dxil, T.getArch());
  EXPECT_EQ(Triple::DXILSubArch_v1_8, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::ShaderModel, T.getOS());
  EXPECT_EQ(VersionTuple(1, 8), T.getDXILVersion());
  EXPECT_EQ(Triple::Amplification, T.getEnvironment());

  T = Triple("dxilv1.8-unknown-shadermodel6.15-library");
  EXPECT_EQ(Triple::dxil, T.getArch());
  EXPECT_EQ(Triple::DXILSubArch_v1_8, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::ShaderModel, T.getOS());
  EXPECT_EQ(VersionTuple(1, 8), T.getDXILVersion());

  T = Triple("x86_64-unknown-fuchsia");
  EXPECT_EQ(Triple::x86_64, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Fuchsia, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("x86_64-unknown-hermit");
  EXPECT_EQ(Triple::x86_64, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::HermitCore, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("x86_64-unknown-uefi");
  EXPECT_EQ(Triple::x86_64, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UEFI, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());
  EXPECT_EQ(Triple::COFF, T.getObjectFormat());

  T = Triple("wasm32-unknown-unknown");
  EXPECT_EQ(Triple::wasm32, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("wasm32-unknown-wasi");
  EXPECT_EQ(Triple::wasm32, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::WASI, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("wasm64-unknown-unknown");
  EXPECT_EQ(Triple::wasm64, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("wasm64-unknown-wasi");
  EXPECT_EQ(Triple::wasm64, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::WASI, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("avr-unknown-unknown");
  EXPECT_EQ(Triple::avr, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("avr");
  EXPECT_EQ(Triple::avr, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("lanai-unknown-unknown");
  EXPECT_EQ(Triple::lanai, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("lanai");
  EXPECT_EQ(Triple::lanai, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("amdgcn-mesa-mesa3d");
  EXPECT_EQ(Triple::amdgcn, T.getArch());
  EXPECT_EQ(Triple::Mesa, T.getVendor());
  EXPECT_EQ(Triple::Mesa3D, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("amdgcn-amd-amdhsa");
  EXPECT_EQ(Triple::amdgcn, T.getArch());
  EXPECT_EQ(Triple::AMD, T.getVendor());
  EXPECT_EQ(Triple::AMDHSA, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("amdgcn-amd-amdpal");
  EXPECT_EQ(Triple::amdgcn, T.getArch());
  EXPECT_EQ(Triple::AMD, T.getVendor());
  EXPECT_EQ(Triple::AMDPAL, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("ve-unknown-linux");
  EXPECT_EQ(Triple::ve, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("csky-unknown-unknown");
  EXPECT_EQ(Triple::csky, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("csky-unknown-linux");
  EXPECT_EQ(Triple::csky, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("loongarch32-unknown-unknown");
  EXPECT_EQ(Triple::loongarch32, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("loongarch32-unknown-linux-gnu");
  EXPECT_EQ(Triple::loongarch32, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::GNU, T.getEnvironment());

  T = Triple("loongarch32-unknown-linux-gnuf32");
  EXPECT_EQ(Triple::loongarch32, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::GNUF32, T.getEnvironment());

  T = Triple("loongarch32-unknown-linux-gnuf64");
  EXPECT_EQ(Triple::loongarch32, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::GNUF64, T.getEnvironment());

  T = Triple("loongarch32-unknown-linux-gnusf");
  EXPECT_EQ(Triple::loongarch32, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::GNUSF, T.getEnvironment());

  T = Triple("loongarch32-unknown-linux-musl");
  EXPECT_EQ(Triple::loongarch32, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::Musl, T.getEnvironment());

  T = Triple("loongarch32-unknown-linux-muslf32");
  EXPECT_EQ(Triple::loongarch32, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::MuslF32, T.getEnvironment());

  T = Triple("loongarch32-unknown-linux-muslsf");
  EXPECT_EQ(Triple::loongarch32, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::MuslSF, T.getEnvironment());

  T = Triple("loongarch64-unknown-linux");
  EXPECT_EQ(Triple::loongarch64, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("loongarch64-unknown-linux-gnu");
  EXPECT_EQ(Triple::loongarch64, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::GNU, T.getEnvironment());

  T = Triple("loongarch64-unknown-linux-gnuf32");
  EXPECT_EQ(Triple::loongarch64, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::GNUF32, T.getEnvironment());

  T = Triple("loongarch64-unknown-linux-gnuf64");
  EXPECT_EQ(Triple::loongarch64, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::GNUF64, T.getEnvironment());

  T = Triple("loongarch64-unknown-linux-gnusf");
  EXPECT_EQ(Triple::loongarch64, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::GNUSF, T.getEnvironment());

  T = Triple("loongarch64-unknown-linux-musl");
  EXPECT_EQ(Triple::loongarch64, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::Musl, T.getEnvironment());

  T = Triple("loongarch64-unknown-linux-muslf32");
  EXPECT_EQ(Triple::loongarch64, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::MuslF32, T.getEnvironment());

  T = Triple("loongarch64-unknown-linux-muslsf");
  EXPECT_EQ(Triple::loongarch64, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::MuslSF, T.getEnvironment());

  T = Triple("riscv32-unknown-unknown");
  EXPECT_EQ(Triple::riscv32, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("riscv64-unknown-linux");
  EXPECT_EQ(Triple::riscv64, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("riscv64-unknown-freebsd");
  EXPECT_EQ(Triple::riscv64, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::FreeBSD, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("riscv64-suse-linux");
  EXPECT_EQ(Triple::riscv64, T.getArch());
  EXPECT_EQ(Triple::SUSE, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("armv7hl-suse-linux-gnueabi");
  EXPECT_EQ(Triple::arm, T.getArch());
  EXPECT_EQ(Triple::SUSE, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::GNUEABI, T.getEnvironment());

  T = Triple("i586-pc-haiku");
  EXPECT_EQ(Triple::x86, T.getArch());
  EXPECT_EQ(Triple::PC, T.getVendor());
  EXPECT_EQ(Triple::Haiku, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("x86_64-unknown-haiku");
  EXPECT_EQ(Triple::x86_64, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Haiku, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("mips-mti-linux-gnu");
  EXPECT_EQ(Triple::mips, T.getArch());
  EXPECT_EQ(Triple::MipsTechnologies, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::GNU, T.getEnvironment());

  T = Triple("mipsel-img-linux-gnu");
  EXPECT_EQ(Triple::mipsel, T.getArch());
  EXPECT_EQ(Triple::ImaginationTechnologies, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::GNU, T.getEnvironment());

  T = Triple("mips64-mti-linux-gnu");
  EXPECT_EQ(Triple::mips64, T.getArch());
  EXPECT_EQ(Triple::MipsTechnologies, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::GNU, T.getEnvironment());

  T = Triple("mips64el-img-linux-gnu");
  EXPECT_EQ(Triple::mips64el, T.getArch());
  EXPECT_EQ(Triple::ImaginationTechnologies, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::GNU, T.getEnvironment());

  T = Triple("mips64el-img-linux-gnuabin32");
  EXPECT_EQ(Triple::mips64el, T.getArch());
  EXPECT_EQ(Triple::ImaginationTechnologies, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::GNUABIN32, T.getEnvironment());

  T = Triple("mips64el-unknown-linux-gnuabi64");
  EXPECT_EQ(Triple::mips64el, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::GNUABI64, T.getEnvironment());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());
  T = Triple("mips64el");
  EXPECT_EQ(Triple::mips64el, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::GNUABI64, T.getEnvironment());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());

  T = Triple("mips64-unknown-linux-gnuabi64");
  EXPECT_EQ(Triple::mips64, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::GNUABI64, T.getEnvironment());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());
  T = Triple("mips64");
  EXPECT_EQ(Triple::mips64, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::GNUABI64, T.getEnvironment());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());

  T = Triple("mipsisa64r6el-unknown-linux-gnuabi64");
  EXPECT_EQ(Triple::mips64el, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::GNUABI64, T.getEnvironment());
  EXPECT_EQ(Triple::MipsSubArch_r6, T.getSubArch());
  T = Triple("mips64r6el");
  EXPECT_EQ(Triple::mips64el, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::GNUABI64, T.getEnvironment());
  EXPECT_EQ(Triple::MipsSubArch_r6, T.getSubArch());
  T = Triple("mipsisa64r6el");
  EXPECT_EQ(Triple::mips64el, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::GNUABI64, T.getEnvironment());
  EXPECT_EQ(Triple::MipsSubArch_r6, T.getSubArch());

  T = Triple("mipsisa64r6-unknown-linux-gnuabi64");
  EXPECT_EQ(Triple::mips64, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::GNUABI64, T.getEnvironment());
  EXPECT_EQ(Triple::MipsSubArch_r6, T.getSubArch());
  T = Triple("mips64r6");
  EXPECT_EQ(Triple::mips64, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::GNUABI64, T.getEnvironment());
  EXPECT_EQ(Triple::MipsSubArch_r6, T.getSubArch());
  T = Triple("mipsisa64r6");
  EXPECT_EQ(Triple::mips64, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::GNUABI64, T.getEnvironment());
  EXPECT_EQ(Triple::MipsSubArch_r6, T.getSubArch());

  T = Triple("mips64el-unknown-linux-gnuabin32");
  EXPECT_EQ(Triple::mips64el, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::GNUABIN32, T.getEnvironment());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());
  T = Triple("mipsn32el");
  EXPECT_EQ(Triple::mips64el, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::GNUABIN32, T.getEnvironment());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());

  T = Triple("mips64-unknown-linux-gnuabin32");
  EXPECT_EQ(Triple::mips64, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::GNUABIN32, T.getEnvironment());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());
  T = Triple("mipsn32");
  EXPECT_EQ(Triple::mips64, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::GNUABIN32, T.getEnvironment());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());

  T = Triple("mipsisa64r6el-unknown-linux-gnuabin32");
  EXPECT_EQ(Triple::mips64el, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::GNUABIN32, T.getEnvironment());
  EXPECT_EQ(Triple::MipsSubArch_r6, T.getSubArch());
  T = Triple("mipsn32r6el");
  EXPECT_EQ(Triple::mips64el, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::GNUABIN32, T.getEnvironment());
  EXPECT_EQ(Triple::MipsSubArch_r6, T.getSubArch());

  T = Triple("mipsisa64r6-unknown-linux-gnuabin32");
  EXPECT_EQ(Triple::mips64, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::GNUABIN32, T.getEnvironment());
  EXPECT_EQ(Triple::MipsSubArch_r6, T.getSubArch());
  T = Triple("mipsn32r6");
  EXPECT_EQ(Triple::mips64, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::GNUABIN32, T.getEnvironment());
  EXPECT_EQ(Triple::MipsSubArch_r6, T.getSubArch());

  T = Triple("mipsel-unknown-linux-gnu");
  EXPECT_EQ(Triple::mipsel, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::GNU, T.getEnvironment());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());
  T = Triple("mipsel");
  EXPECT_EQ(Triple::mipsel, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::GNU, T.getEnvironment());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());

  T = Triple("mips-unknown-linux-gnu");
  EXPECT_EQ(Triple::mips, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::GNU, T.getEnvironment());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());
  T = Triple("mips");
  EXPECT_EQ(Triple::mips, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::GNU, T.getEnvironment());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());

  T = Triple("mipsisa32r6el-unknown-linux-gnu");
  EXPECT_EQ(Triple::mipsel, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::GNU, T.getEnvironment());
  EXPECT_EQ(Triple::MipsSubArch_r6, T.getSubArch());
  T = Triple("mipsr6el");
  EXPECT_EQ(Triple::mipsel, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::MipsSubArch_r6, T.getSubArch());
  T = Triple("mipsisa32r6el");
  EXPECT_EQ(Triple::mipsel, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::GNU, T.getEnvironment());
  EXPECT_EQ(Triple::MipsSubArch_r6, T.getSubArch());

  T = Triple("mipsisa32r6-unknown-linux-gnu");
  EXPECT_EQ(Triple::mips, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::GNU, T.getEnvironment());
  EXPECT_EQ(Triple::MipsSubArch_r6, T.getSubArch());
  T = Triple("mipsr6");
  EXPECT_EQ(Triple::mips, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::GNU, T.getEnvironment());
  EXPECT_EQ(Triple::MipsSubArch_r6, T.getSubArch());
  T = Triple("mipsisa32r6");
  EXPECT_EQ(Triple::mips, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::GNU, T.getEnvironment());
  EXPECT_EQ(Triple::MipsSubArch_r6, T.getSubArch());

  T = Triple("mips64el-unknown-linux-muslabi64");
  EXPECT_EQ(Triple::mips64el, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::MuslABI64, T.getEnvironment());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());

  T = Triple("mips64-unknown-linux-muslabi64");
  EXPECT_EQ(Triple::mips64, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::MuslABI64, T.getEnvironment());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());

  T = Triple("mipsisa64r6el-unknown-linux-muslabi64");
  EXPECT_EQ(Triple::mips64el, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::MuslABI64, T.getEnvironment());
  EXPECT_EQ(Triple::MipsSubArch_r6, T.getSubArch());

  T = Triple("mipsisa64r6-unknown-linux-muslabi64");
  EXPECT_EQ(Triple::mips64, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::MuslABI64, T.getEnvironment());
  EXPECT_EQ(Triple::MipsSubArch_r6, T.getSubArch());

  T = Triple("mips64el-unknown-linux-muslabin32");
  EXPECT_EQ(Triple::mips64el, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::MuslABIN32, T.getEnvironment());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());

  T = Triple("mips64-unknown-linux-muslabin32");
  EXPECT_EQ(Triple::mips64, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::MuslABIN32, T.getEnvironment());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());

  T = Triple("mipsisa64r6el-unknown-linux-muslabin32");
  EXPECT_EQ(Triple::mips64el, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::MuslABIN32, T.getEnvironment());
  EXPECT_EQ(Triple::MipsSubArch_r6, T.getSubArch());

  T = Triple("mipsisa64r6-unknown-linux-muslabin32");
  EXPECT_EQ(Triple::mips64, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::MuslABIN32, T.getEnvironment());
  EXPECT_EQ(Triple::MipsSubArch_r6, T.getSubArch());

  T = Triple("mipsel-unknown-linux-musl");
  EXPECT_EQ(Triple::mipsel, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::Musl, T.getEnvironment());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());

  T = Triple("mips-unknown-linux-musl");
  EXPECT_EQ(Triple::mips, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::Musl, T.getEnvironment());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());

  T = Triple("mipsisa32r6el-unknown-linux-musl");
  EXPECT_EQ(Triple::mipsel, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::Musl, T.getEnvironment());
  EXPECT_EQ(Triple::MipsSubArch_r6, T.getSubArch());

  T = Triple("mipsisa32r6-unknown-linux-musl");
  EXPECT_EQ(Triple::mips, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::Musl, T.getEnvironment());
  EXPECT_EQ(Triple::MipsSubArch_r6, T.getSubArch());

  T = Triple("arm-oe-linux-gnueabi");
  EXPECT_EQ(Triple::arm, T.getArch());
  EXPECT_EQ(Triple::OpenEmbedded, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::GNUEABI, T.getEnvironment());

  T = Triple("aarch64-oe-linux");
  EXPECT_EQ(Triple::aarch64, T.getArch());
  EXPECT_EQ(Triple::OpenEmbedded, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());
  EXPECT_TRUE(T.isArch64Bit());

  T = Triple("arm64_32-apple-ios");
  EXPECT_EQ(Triple::aarch64_32, T.getArch());
  EXPECT_EQ(Triple::IOS, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());
  EXPECT_TRUE(T.isArch32Bit());

  T = Triple("dxil-unknown-shadermodel-pixel");
  EXPECT_EQ(Triple::dxil, T.getArch());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::ShaderModel, T.getOS());
  EXPECT_EQ(Triple::Pixel, T.getEnvironment());
  EXPECT_FALSE(T.supportsCOMDAT());

  T = Triple("dxil-unknown-shadermodel-vertex");
  EXPECT_EQ(Triple::dxil, T.getArch());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::ShaderModel, T.getOS());
  EXPECT_EQ(Triple::Vertex, T.getEnvironment());
  EXPECT_FALSE(T.supportsCOMDAT());

  T = Triple("dxil-unknown-shadermodel-geometry");
  EXPECT_EQ(Triple::dxil, T.getArch());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::ShaderModel, T.getOS());
  EXPECT_EQ(Triple::Geometry, T.getEnvironment());
  EXPECT_FALSE(T.supportsCOMDAT());

  T = Triple("dxil-unknown-shadermodel-hull");
  EXPECT_EQ(Triple::dxil, T.getArch());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::ShaderModel, T.getOS());
  EXPECT_EQ(Triple::Hull, T.getEnvironment());
  EXPECT_FALSE(T.supportsCOMDAT());

  T = Triple("dxil-unknown-shadermodel-domain");
  EXPECT_EQ(Triple::dxil, T.getArch());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::ShaderModel, T.getOS());
  EXPECT_EQ(Triple::Domain, T.getEnvironment());
  EXPECT_FALSE(T.supportsCOMDAT());

  T = Triple("dxil-unknown-shadermodel-compute");
  EXPECT_EQ(Triple::dxil, T.getArch());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::ShaderModel, T.getOS());
  EXPECT_EQ(Triple::Compute, T.getEnvironment());
  EXPECT_FALSE(T.supportsCOMDAT());

  T = Triple("dxil-unknown-shadermodel-library");
  EXPECT_EQ(Triple::dxil, T.getArch());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::ShaderModel, T.getOS());
  EXPECT_EQ(Triple::Library, T.getEnvironment());
  EXPECT_FALSE(T.supportsCOMDAT());

  T = Triple("dxil-unknown-shadermodel-raygeneration");
  EXPECT_EQ(Triple::dxil, T.getArch());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::ShaderModel, T.getOS());
  EXPECT_EQ(Triple::RayGeneration, T.getEnvironment());
  EXPECT_FALSE(T.supportsCOMDAT());

  T = Triple("dxil-unknown-shadermodel-intersection");
  EXPECT_EQ(Triple::dxil, T.getArch());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::ShaderModel, T.getOS());
  EXPECT_EQ(Triple::Intersection, T.getEnvironment());
  EXPECT_FALSE(T.supportsCOMDAT());

  T = Triple("dxil-unknown-shadermodel-anyhit");
  EXPECT_EQ(Triple::dxil, T.getArch());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::ShaderModel, T.getOS());
  EXPECT_EQ(Triple::AnyHit, T.getEnvironment());
  EXPECT_FALSE(T.supportsCOMDAT());

  T = Triple("dxil-unknown-shadermodel-closesthit");
  EXPECT_EQ(Triple::dxil, T.getArch());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::ShaderModel, T.getOS());
  EXPECT_EQ(Triple::ClosestHit, T.getEnvironment());
  EXPECT_FALSE(T.supportsCOMDAT());

  T = Triple("dxil-unknown-shadermodel-miss");
  EXPECT_EQ(Triple::dxil, T.getArch());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::ShaderModel, T.getOS());
  EXPECT_EQ(Triple::Miss, T.getEnvironment());
  EXPECT_FALSE(T.supportsCOMDAT());

  T = Triple("dxil-unknown-shadermodel-callable");
  EXPECT_EQ(Triple::dxil, T.getArch());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::ShaderModel, T.getOS());
  EXPECT_EQ(Triple::Callable, T.getEnvironment());
  EXPECT_FALSE(T.supportsCOMDAT());

  T = Triple("dxil-unknown-shadermodel-mesh");
  EXPECT_EQ(Triple::dxil, T.getArch());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::ShaderModel, T.getOS());
  EXPECT_EQ(Triple::Mesh, T.getEnvironment());
  EXPECT_FALSE(T.supportsCOMDAT());

  T = Triple("dxil-unknown-shadermodel-amplification");
  EXPECT_EQ(Triple::dxil, T.getArch());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::ShaderModel, T.getOS());
  EXPECT_EQ(Triple::Amplification, T.getEnvironment());
  EXPECT_FALSE(T.supportsCOMDAT());

  T = Triple("dxilv1.0-unknown-unknown");
  EXPECT_EQ(Triple::dxil, T.getArch());
  EXPECT_EQ(Triple::DXILSubArch_v1_0, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());

  T = Triple("dxilv1.1-unknown-unknown");
  EXPECT_EQ(Triple::dxil, T.getArch());
  EXPECT_EQ(Triple::DXILSubArch_v1_1, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());

  T = Triple("dxilv1.2-unknown-unknown");
  EXPECT_EQ(Triple::dxil, T.getArch());
  EXPECT_EQ(Triple::DXILSubArch_v1_2, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());

  T = Triple("dxilv1.3-unknown-unknown");
  EXPECT_EQ(Triple::dxil, T.getArch());
  EXPECT_EQ(Triple::DXILSubArch_v1_3, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());

  T = Triple("dxilv1.4-unknown-unknown");
  EXPECT_EQ(Triple::dxil, T.getArch());
  EXPECT_EQ(Triple::DXILSubArch_v1_4, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());

  T = Triple("dxilv1.5-unknown-unknown");
  EXPECT_EQ(Triple::dxil, T.getArch());
  EXPECT_EQ(Triple::DXILSubArch_v1_5, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());

  T = Triple("dxilv1.6-unknown-unknown");
  EXPECT_EQ(Triple::dxil, T.getArch());
  EXPECT_EQ(Triple::DXILSubArch_v1_6, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());

  T = Triple("dxilv1.7-unknown-unknown");
  EXPECT_EQ(Triple::dxil, T.getArch());
  EXPECT_EQ(Triple::DXILSubArch_v1_7, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());

  T = Triple("dxilv1.8-unknown-unknown");
  EXPECT_EQ(Triple::dxil, T.getArch());
  EXPECT_EQ(Triple::DXILSubArch_v1_8, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());

  // Check specification of unknown SubArch results in
  // unknown architecture.
  T = Triple("dxilv1.999-unknown-unknown");
  EXPECT_EQ(Triple::UnknownArch, T.getArch());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());

  T = Triple("dxil-unknown-unknown");
  EXPECT_EQ(Triple::dxil, T.getArch());
  EXPECT_EQ(Triple::NoSubArch, T.getSubArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());

  T = Triple("xtensa");
  EXPECT_EQ(Triple::xtensa, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("xtensa-unknown-unknown");
  EXPECT_EQ(Triple::xtensa, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("arm-unknown-linux-ohos");
  EXPECT_EQ(Triple::arm, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::OpenHOS, T.getEnvironment());

  T = Triple("arm-unknown-liteos");
  EXPECT_EQ(Triple::arm, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::LiteOS, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("x86_64-pc-serenity");
  EXPECT_EQ(Triple::x86_64, T.getArch());
  EXPECT_EQ(Triple::PC, T.getVendor());
  EXPECT_EQ(Triple::Serenity, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("aarch64-pc-serenity");
  EXPECT_EQ(Triple::aarch64, T.getArch());
  EXPECT_EQ(Triple::PC, T.getVendor());
  EXPECT_EQ(Triple::Serenity, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("aarch64-unknown-linux-pauthtest");
  EXPECT_EQ(Triple::aarch64, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::PAuthTest, T.getEnvironment());

  // Gentoo time64 triples
  T = Triple("i686-pc-linux-gnut64");
  EXPECT_EQ(Triple::x86, T.getArch());
  EXPECT_EQ(Triple::PC, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::GNUT64, T.getEnvironment());
  EXPECT_TRUE(T.isTime64ABI());

  T = Triple("armv5tel-softfloat-linux-gnueabit64");
  EXPECT_EQ(Triple::arm, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::GNUEABIT64, T.getEnvironment());
  EXPECT_TRUE(T.isTime64ABI());

  T = Triple("armv7a-unknown-linux-gnueabihft64");
  EXPECT_EQ(Triple::arm, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::GNUEABIHFT64, T.getEnvironment());
  EXPECT_TRUE(T.isTime64ABI());
  EXPECT_TRUE(T.isHardFloatABI());

  T = Triple("x86_64-pc-linux-llvm");
  EXPECT_EQ(Triple::x86_64, T.getArch());
  EXPECT_EQ(Triple::PC, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::LLVM, T.getEnvironment());

  T = Triple("spirv64-intel-unknown");
  EXPECT_EQ(Triple::spirv64, T.getArch());
  EXPECT_EQ(Triple::Intel, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("huh");
  EXPECT_EQ(Triple::UnknownArch, T.getArch());
}

static std::string Join(StringRef A, StringRef B, StringRef C) {
  std::string Str = std::string(A);
  Str += '-';
  Str += B;
  Str += '-';
  Str += C;
  return Str;
}

static std::string Join(StringRef A, StringRef B, StringRef C, StringRef D) {
  std::string Str = std::string(A);
  Str += '-';
  Str += B;
  Str += '-';
  Str += C;
  Str += '-';
  Str += D;
  return Str;
}

TEST(TripleTest, Normalization) {

  EXPECT_EQ("unknown", Triple::normalize(""));
  EXPECT_EQ("unknown-unknown", Triple::normalize("-"));
  EXPECT_EQ("unknown-unknown-unknown", Triple::normalize("--"));
  EXPECT_EQ("unknown-unknown-unknown-unknown", Triple::normalize("---"));
  EXPECT_EQ("unknown-unknown-unknown-unknown-unknown",
            Triple::normalize("----"));

  EXPECT_EQ("a", Triple::normalize("a"));
  EXPECT_EQ("a-b", Triple::normalize("a-b"));
  EXPECT_EQ("a-b-c", Triple::normalize("a-b-c"));
  EXPECT_EQ("a-b-c-d", Triple::normalize("a-b-c-d"));

  EXPECT_EQ("i386-b-c", Triple::normalize("i386-b-c"));
  EXPECT_EQ("i386-a-c", Triple::normalize("a-i386-c"));
  EXPECT_EQ("i386-a-b", Triple::normalize("a-b-i386"));
  EXPECT_EQ("i386-a-b-c", Triple::normalize("a-b-c-i386"));

  EXPECT_EQ("a-pc-c", Triple::normalize("a-pc-c"));
  EXPECT_EQ("unknown-pc-b-c", Triple::normalize("pc-b-c"));
  EXPECT_EQ("a-pc-b", Triple::normalize("a-b-pc"));
  EXPECT_EQ("a-pc-b-c", Triple::normalize("a-b-c-pc"));

  EXPECT_EQ("a-b-linux", Triple::normalize("a-b-linux"));
  EXPECT_EQ("unknown-unknown-linux-b-c", Triple::normalize("linux-b-c"));
  EXPECT_EQ("a-unknown-linux-c", Triple::normalize("a-linux-c"));

  EXPECT_EQ("i386-pc-a", Triple::normalize("a-pc-i386"));
  EXPECT_EQ("i386-pc-unknown", Triple::normalize("-pc-i386"));
  EXPECT_EQ("unknown-pc-linux-c", Triple::normalize("linux-pc-c"));
  EXPECT_EQ("unknown-pc-linux", Triple::normalize("linux-pc-"));

  EXPECT_EQ("i386", Triple::normalize("i386"));
  EXPECT_EQ("unknown-pc", Triple::normalize("pc"));
  EXPECT_EQ("unknown-unknown-linux", Triple::normalize("linux"));

  EXPECT_EQ("x86_64-unknown-linux-gnu", Triple::normalize("x86_64-gnu-linux"));

  EXPECT_EQ("a-unknown-unknown",
            Triple::normalize("a", Triple::CanonicalForm::THREE_IDENT));
  EXPECT_EQ("a-b-unknown",
            Triple::normalize("a-b", Triple::CanonicalForm::THREE_IDENT));
  EXPECT_EQ("a-b-c",
            Triple::normalize("a-b-c", Triple::CanonicalForm::THREE_IDENT));
  EXPECT_EQ("a-b-c",
            Triple::normalize("a-b-c-d", Triple::CanonicalForm::THREE_IDENT));
  EXPECT_EQ("a-b-c",
            Triple::normalize("a-b-c-d-e", Triple::CanonicalForm::THREE_IDENT));

  EXPECT_EQ("a-unknown-unknown-unknown",
            Triple::normalize("a", Triple::CanonicalForm::FOUR_IDENT));
  EXPECT_EQ("a-b-unknown-unknown",
            Triple::normalize("a-b", Triple::CanonicalForm::FOUR_IDENT));
  EXPECT_EQ("a-b-c-unknown",
            Triple::normalize("a-b-c", Triple::CanonicalForm::FOUR_IDENT));
  EXPECT_EQ("a-b-c-d",
            Triple::normalize("a-b-c-d", Triple::CanonicalForm::FOUR_IDENT));
  EXPECT_EQ("a-b-c-d",
            Triple::normalize("a-b-c-d-e", Triple::CanonicalForm::FOUR_IDENT));

  EXPECT_EQ("a-unknown-unknown-unknown-unknown",
            Triple::normalize("a", Triple::CanonicalForm::FIVE_IDENT));
  EXPECT_EQ("a-b-unknown-unknown-unknown",
            Triple::normalize("a-b", Triple::CanonicalForm::FIVE_IDENT));
  EXPECT_EQ("a-b-c-unknown-unknown",
            Triple::normalize("a-b-c", Triple::CanonicalForm::FIVE_IDENT));
  EXPECT_EQ("a-b-c-d-unknown",
            Triple::normalize("a-b-c-d", Triple::CanonicalForm::FIVE_IDENT));
  EXPECT_EQ("a-b-c-d-e",
            Triple::normalize("a-b-c-d-e", Triple::CanonicalForm::FIVE_IDENT));

  EXPECT_EQ("i386-b-c-unknown",
            Triple::normalize("i386-b-c", Triple::CanonicalForm::FOUR_IDENT));
  EXPECT_EQ("i386-b-c-unknown-unknown",
            Triple::normalize("i386-b-c", Triple::CanonicalForm::FIVE_IDENT));

  EXPECT_EQ("i386-a-c-unknown",
            Triple::normalize("a-i386-c", Triple::CanonicalForm::FOUR_IDENT));
  EXPECT_EQ("i386-a-c-unknown-unknown",
            Triple::normalize("a-i386-c", Triple::CanonicalForm::FIVE_IDENT));

  EXPECT_EQ("i386-a-b-unknown",
            Triple::normalize("a-b-i386", Triple::CanonicalForm::FOUR_IDENT));
  EXPECT_EQ("i386-a-b-c",
            Triple::normalize("a-b-c-i386", Triple::CanonicalForm::FOUR_IDENT));

  EXPECT_EQ("a-pc-c-unknown",
            Triple::normalize("a-pc-c", Triple::CanonicalForm::FOUR_IDENT));
  EXPECT_EQ("unknown-pc-b-c",
            Triple::normalize("pc-b-c", Triple::CanonicalForm::FOUR_IDENT));
  EXPECT_EQ("a-pc-b-unknown",
            Triple::normalize("a-b-pc", Triple::CanonicalForm::FOUR_IDENT));
  EXPECT_EQ("a-pc-b-c",
            Triple::normalize("a-b-c-pc", Triple::CanonicalForm::FOUR_IDENT));

  EXPECT_EQ("a-b-linux-unknown",
            Triple::normalize("a-b-linux", Triple::CanonicalForm::FOUR_IDENT));
  // We lose `-c` here as expected.
  EXPECT_EQ("unknown-unknown-linux-b",
            Triple::normalize("linux-b-c", Triple::CanonicalForm::FOUR_IDENT));
  EXPECT_EQ("a-unknown-linux-c",
            Triple::normalize("a-linux-c", Triple::CanonicalForm::FOUR_IDENT));

  EXPECT_EQ("i386-pc-a-unknown",
            Triple::normalize("a-pc-i386", Triple::CanonicalForm::FOUR_IDENT));
  EXPECT_EQ("i386-pc-unknown-unknown",
            Triple::normalize("-pc-i386", Triple::CanonicalForm::FOUR_IDENT));
  EXPECT_EQ("unknown-pc-linux-c",
            Triple::normalize("linux-pc-c", Triple::CanonicalForm::FOUR_IDENT));
  EXPECT_EQ("unknown-pc-linux-unknown",
            Triple::normalize("linux-pc-", Triple::CanonicalForm::FOUR_IDENT));

  EXPECT_EQ("i386-unknown-unknown-unknown",
            Triple::normalize("i386", Triple::CanonicalForm::FOUR_IDENT));
  EXPECT_EQ("unknown-pc-unknown-unknown",
            Triple::normalize("pc", Triple::CanonicalForm::FOUR_IDENT));
  EXPECT_EQ("unknown-unknown-linux-unknown",
            Triple::normalize("linux", Triple::CanonicalForm::FOUR_IDENT));

  EXPECT_EQ(
      "x86_64-unknown-linux-gnu",
      Triple::normalize("x86_64-gnu-linux", Triple::CanonicalForm::FOUR_IDENT));

  EXPECT_EQ("i386-a-b-unknown-unknown",
            Triple::normalize("a-b-i386", Triple::CanonicalForm::FIVE_IDENT));
  EXPECT_EQ("i386-a-b-c-unknown",
            Triple::normalize("a-b-c-i386", Triple::CanonicalForm::FIVE_IDENT));

  EXPECT_EQ("a-pc-c-unknown-unknown",
            Triple::normalize("a-pc-c", Triple::CanonicalForm::FIVE_IDENT));
  EXPECT_EQ("unknown-pc-b-c-unknown",
            Triple::normalize("pc-b-c", Triple::CanonicalForm::FIVE_IDENT));
  EXPECT_EQ("a-pc-b-unknown-unknown",
            Triple::normalize("a-b-pc", Triple::CanonicalForm::FIVE_IDENT));
  EXPECT_EQ("a-pc-b-c-unknown",
            Triple::normalize("a-b-c-pc", Triple::CanonicalForm::FIVE_IDENT));

  EXPECT_EQ("a-b-linux-unknown-unknown",
            Triple::normalize("a-b-linux", Triple::CanonicalForm::FIVE_IDENT));
  EXPECT_EQ("unknown-unknown-linux-b-c",
            Triple::normalize("linux-b-c", Triple::CanonicalForm::FIVE_IDENT));
  EXPECT_EQ("a-unknown-linux-c-unknown",
            Triple::normalize("a-linux-c", Triple::CanonicalForm::FIVE_IDENT));

  EXPECT_EQ("i386-pc-a-unknown-unknown",
            Triple::normalize("a-pc-i386", Triple::CanonicalForm::FIVE_IDENT));
  EXPECT_EQ("i386-pc-unknown-unknown-unknown",
            Triple::normalize("-pc-i386", Triple::CanonicalForm::FIVE_IDENT));
  EXPECT_EQ("unknown-pc-linux-c-unknown",
            Triple::normalize("linux-pc-c", Triple::CanonicalForm::FIVE_IDENT));
  EXPECT_EQ("unknown-pc-linux-unknown-unknown",
            Triple::normalize("linux-pc-", Triple::CanonicalForm::FIVE_IDENT));

  EXPECT_EQ("i386-unknown-unknown-unknown-unknown",
            Triple::normalize("i386", Triple::CanonicalForm::FIVE_IDENT));
  EXPECT_EQ("unknown-pc-unknown-unknown-unknown",
            Triple::normalize("pc", Triple::CanonicalForm::FIVE_IDENT));
  EXPECT_EQ("unknown-unknown-linux-unknown-unknown",
            Triple::normalize("linux", Triple::CanonicalForm::FIVE_IDENT));

  EXPECT_EQ(
      "x86_64-unknown-linux-gnu-unknown",
      Triple::normalize("x86_64-gnu-linux", Triple::CanonicalForm::FIVE_IDENT));

  // Check that normalizing a permutated set of valid components returns a
  // triple with the unpermuted components.
  //
  // We don't check every possible combination. For the set of architectures A,
  // vendors V, operating systems O, and environments E, that would require |A|
  // * |V| * |O| * |E| * 4! tests. Instead we check every option for any given
  // slot and make sure it gets normalized to the correct position from every
  // permutation. This should cover the core logic while being a tractable
  // number of tests at (|A| + |V| + |O| + |E|) * 4!.
  auto FirstArchType = Triple::ArchType(Triple::UnknownArch + 1);
  auto FirstVendorType = Triple::VendorType(Triple::UnknownVendor + 1);
  auto FirstOSType = Triple::OSType(Triple::UnknownOS + 1);
  auto FirstEnvType = Triple::EnvironmentType(Triple::UnknownEnvironment + 1);
  StringRef InitialC[] = {Triple::getArchTypeName(FirstArchType),
                          Triple::getVendorTypeName(FirstVendorType),
                          Triple::getOSTypeName(FirstOSType),
                          Triple::getEnvironmentTypeName(FirstEnvType)};
  for (int Arch = FirstArchType; Arch <= Triple::LastArchType; ++Arch) {
    StringRef C[] = {InitialC[0], InitialC[1], InitialC[2], InitialC[3]};
    C[0] = Triple::getArchTypeName(Triple::ArchType(Arch));
    std::string E = Join(C[0], C[1], C[2]);
    int I[] = {0, 1, 2};
    do {
      EXPECT_EQ(E, Triple::normalize(Join(C[I[0]], C[I[1]], C[I[2]])));
    } while (std::next_permutation(std::begin(I), std::end(I)));
    std::string F = Join(C[0], C[1], C[2], C[3]);
    int J[] = {0, 1, 2, 3};
    do {
      EXPECT_EQ(F, Triple::normalize(Join(C[J[0]], C[J[1]], C[J[2]], C[J[3]])));
    } while (std::next_permutation(std::begin(J), std::end(J)));
  }
  for (int Vendor = FirstVendorType; Vendor <= Triple::LastVendorType;
       ++Vendor) {
    StringRef C[] = {InitialC[0], InitialC[1], InitialC[2], InitialC[3]};
    C[1] = Triple::getVendorTypeName(Triple::VendorType(Vendor));
    std::string E = Join(C[0], C[1], C[2]);
    int I[] = {0, 1, 2};
    do {
      EXPECT_EQ(E, Triple::normalize(Join(C[I[0]], C[I[1]], C[I[2]])));
    } while (std::next_permutation(std::begin(I), std::end(I)));
    std::string F = Join(C[0], C[1], C[2], C[3]);
    int J[] = {0, 1, 2, 3};
    do {
      EXPECT_EQ(F, Triple::normalize(Join(C[J[0]], C[J[1]], C[J[2]], C[J[3]])));
    } while (std::next_permutation(std::begin(J), std::end(J)));
  }
  for (int OS = FirstOSType; OS <= Triple::LastOSType; ++OS) {
    if (OS == Triple::Win32)
      continue;
    StringRef C[] = {InitialC[0], InitialC[1], InitialC[2], InitialC[3]};
    C[2] = Triple::getOSTypeName(Triple::OSType(OS));
    std::string E = Join(C[0], C[1], C[2]);
    int I[] = {0, 1, 2};
    do {
      EXPECT_EQ(E, Triple::normalize(Join(C[I[0]], C[I[1]], C[I[2]])));
    } while (std::next_permutation(std::begin(I), std::end(I)));
    std::string F = Join(C[0], C[1], C[2], C[3]);
    int J[] = {0, 1, 2, 3};
    do {
      EXPECT_EQ(F, Triple::normalize(Join(C[J[0]], C[J[1]], C[J[2]], C[J[3]])));
    } while (std::next_permutation(std::begin(J), std::end(J)));
  }
  for (int Env = FirstEnvType; Env <= Triple::LastEnvironmentType; ++Env) {
    StringRef C[] = {InitialC[0], InitialC[1], InitialC[2], InitialC[3]};
    C[3] = Triple::getEnvironmentTypeName(Triple::EnvironmentType(Env));
    std::string F = Join(C[0], C[1], C[2], C[3]);
    int J[] = {0, 1, 2, 3};
    do {
      EXPECT_EQ(F, Triple::normalize(Join(C[J[0]], C[J[1]], C[J[2]], C[J[3]])));
    } while (std::next_permutation(std::begin(J), std::end(J)));
  }

  // Various real-world funky triples.  The value returned by GCC's config.sub
  // is given in the comment.
  EXPECT_EQ("i386-unknown-windows-gnu",
            Triple::normalize("i386-mingw32")); // i386-pc-mingw32
  EXPECT_EQ("x86_64-unknown-linux-gnu",
            Triple::normalize("x86_64-linux-gnu")); // x86_64-pc-linux-gnu
  EXPECT_EQ("i486-unknown-linux-gnu",
            Triple::normalize("i486-linux-gnu")); // i486-pc-linux-gnu
  EXPECT_EQ("i386-redhat-linux",
            Triple::normalize("i386-redhat-linux")); // i386-redhat-linux-gnu
  EXPECT_EQ("i686-unknown-linux",
            Triple::normalize("i686-linux")); // i686-pc-linux-gnu
  EXPECT_EQ("arm-unknown-none-eabi",
            Triple::normalize("arm-none-eabi")); // arm-none-eabi
  EXPECT_EQ("ve-unknown-linux",
            Triple::normalize("ve-linux")); // ve-linux
  EXPECT_EQ("wasm32-unknown-wasi",
            Triple::normalize("wasm32-wasi")); // wasm32-unknown-wasi
  EXPECT_EQ("wasm64-unknown-wasi",
            Triple::normalize("wasm64-wasi")); // wasm64-unknown-wasi
}

TEST(TripleTest, MutateName) {
  Triple T;
  EXPECT_EQ(Triple::UnknownArch, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T.setArchName("i386");
  EXPECT_EQ(Triple::x86, T.getArch());
  EXPECT_EQ("i386--", T.getTriple());

  T.setVendorName("pc");
  EXPECT_EQ(Triple::x86, T.getArch());
  EXPECT_EQ(Triple::PC, T.getVendor());
  EXPECT_EQ("i386-pc-", T.getTriple());

  T.setOSName("linux");
  EXPECT_EQ(Triple::x86, T.getArch());
  EXPECT_EQ(Triple::PC, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ("i386-pc-linux", T.getTriple());

  T.setEnvironmentName("gnu");
  EXPECT_EQ(Triple::x86, T.getArch());
  EXPECT_EQ(Triple::PC, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ("i386-pc-linux-gnu", T.getTriple());

  T.setOSName("freebsd");
  EXPECT_EQ(Triple::x86, T.getArch());
  EXPECT_EQ(Triple::PC, T.getVendor());
  EXPECT_EQ(Triple::FreeBSD, T.getOS());
  EXPECT_EQ("i386-pc-freebsd-gnu", T.getTriple());

  T.setOSAndEnvironmentName("darwin");
  EXPECT_EQ(Triple::x86, T.getArch());
  EXPECT_EQ(Triple::PC, T.getVendor());
  EXPECT_EQ(Triple::Darwin, T.getOS());
  EXPECT_EQ("i386-pc-darwin", T.getTriple());
}

TEST(TripleTest, BitWidthChecks) {
  Triple T;
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_FALSE(T.isArch32Bit());
  EXPECT_FALSE(T.isArch64Bit());
  EXPECT_EQ(T.getArchPointerBitWidth(), 0U);

  T.setArch(Triple::arm);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_TRUE(T.isArch32Bit());
  EXPECT_FALSE(T.isArch64Bit());
  EXPECT_EQ(T.getArchPointerBitWidth(), 32U);

  T.setArch(Triple::hexagon);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_TRUE(T.isArch32Bit());
  EXPECT_FALSE(T.isArch64Bit());
  EXPECT_EQ(T.getArchPointerBitWidth(), 32U);

  T.setArch(Triple::mips);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_TRUE(T.isArch32Bit());
  EXPECT_FALSE(T.isArch64Bit());
  EXPECT_EQ(T.getArchPointerBitWidth(), 32U);

  T.setArch(Triple::mips64);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_FALSE(T.isArch32Bit());
  EXPECT_TRUE(T.isArch64Bit());
  EXPECT_EQ(T.getArchPointerBitWidth(), 64U);

  T.setArch(Triple::msp430);
  EXPECT_TRUE(T.isArch16Bit());
  EXPECT_FALSE(T.isArch32Bit());
  EXPECT_FALSE(T.isArch64Bit());
  EXPECT_EQ(T.getArchPointerBitWidth(), 16U);

  T.setArch(Triple::ppc);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_TRUE(T.isArch32Bit());
  EXPECT_FALSE(T.isArch64Bit());
  EXPECT_EQ(T.getArchPointerBitWidth(), 32U);

  T.setArch(Triple::ppc64);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_FALSE(T.isArch32Bit());
  EXPECT_TRUE(T.isArch64Bit());
  EXPECT_EQ(T.getArchPointerBitWidth(), 64U);

  T.setArch(Triple::x86);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_TRUE(T.isArch32Bit());
  EXPECT_FALSE(T.isArch64Bit());
  EXPECT_EQ(T.getArchPointerBitWidth(), 32U);

  T.setArch(Triple::x86_64);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_FALSE(T.isArch32Bit());
  EXPECT_TRUE(T.isArch64Bit());
  EXPECT_EQ(T.getArchPointerBitWidth(), 64U);

  T.setArch(Triple::amdil);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_TRUE(T.isArch32Bit());
  EXPECT_FALSE(T.isArch64Bit());
  EXPECT_EQ(T.getArchPointerBitWidth(), 32U);

  T.setArch(Triple::amdil64);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_FALSE(T.isArch32Bit());
  EXPECT_TRUE(T.isArch64Bit());
  EXPECT_EQ(T.getArchPointerBitWidth(), 64U);

  T.setArch(Triple::hsail);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_TRUE(T.isArch32Bit());
  EXPECT_FALSE(T.isArch64Bit());
  EXPECT_EQ(T.getArchPointerBitWidth(), 32U);

  T.setArch(Triple::hsail64);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_FALSE(T.isArch32Bit());
  EXPECT_TRUE(T.isArch64Bit());
  EXPECT_EQ(T.getArchPointerBitWidth(), 64U);

  T.setArch(Triple::spir);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_TRUE(T.isArch32Bit());
  EXPECT_FALSE(T.isArch64Bit());
  EXPECT_EQ(T.getArchPointerBitWidth(), 32U);

  T.setArch(Triple::spir64);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_FALSE(T.isArch32Bit());
  EXPECT_TRUE(T.isArch64Bit());
  EXPECT_EQ(T.getArchPointerBitWidth(), 64U);

  T.setArch(Triple::spirv);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_FALSE(T.isArch32Bit());
  EXPECT_TRUE(T.isArch64Bit());
  EXPECT_TRUE(T.isSPIRV());

  T.setArch(Triple::spirv32);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_TRUE(T.isArch32Bit());
  EXPECT_FALSE(T.isArch64Bit());
  EXPECT_TRUE(T.isSPIRV());

  T.setArch(Triple::spirv64);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_FALSE(T.isArch32Bit());
  EXPECT_TRUE(T.isArch64Bit());
  EXPECT_TRUE(T.isSPIRV());

  T.setArch(Triple::sparc);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_TRUE(T.isArch32Bit());
  EXPECT_FALSE(T.isArch64Bit());

  T.setArch(Triple::sparcel);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_TRUE(T.isArch32Bit());
  EXPECT_FALSE(T.isArch64Bit());

  T.setArch(Triple::sparcv9);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_FALSE(T.isArch32Bit());
  EXPECT_TRUE(T.isArch64Bit());

  T.setArch(Triple::wasm32);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_TRUE(T.isArch32Bit());
  EXPECT_FALSE(T.isArch64Bit());

  T.setArch(Triple::wasm64);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_FALSE(T.isArch32Bit());
  EXPECT_TRUE(T.isArch64Bit());

  T.setArch(Triple::avr);
  EXPECT_TRUE(T.isArch16Bit());
  EXPECT_FALSE(T.isArch32Bit());
  EXPECT_FALSE(T.isArch64Bit());

  T.setArch(Triple::lanai);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_TRUE(T.isArch32Bit());
  EXPECT_FALSE(T.isArch64Bit());

  T.setArch(Triple::riscv32);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_TRUE(T.isArch32Bit());
  EXPECT_FALSE(T.isArch64Bit());
  EXPECT_TRUE(T.isRISCV());

  T.setArch(Triple::riscv64);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_FALSE(T.isArch32Bit());
  EXPECT_TRUE(T.isArch64Bit());
  EXPECT_TRUE(T.isRISCV());

  T.setArch(Triple::csky);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_TRUE(T.isArch32Bit());
  EXPECT_FALSE(T.isArch64Bit());
  EXPECT_TRUE(T.isCSKY());

  T.setArch(Triple::loongarch32);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_TRUE(T.isArch32Bit());
  EXPECT_FALSE(T.isArch64Bit());
  EXPECT_TRUE(T.isLoongArch());
  EXPECT_TRUE(T.isLoongArch32());

  T.setArch(Triple::loongarch64);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_FALSE(T.isArch32Bit());
  EXPECT_TRUE(T.isArch64Bit());
  EXPECT_TRUE(T.isLoongArch());
  EXPECT_TRUE(T.isLoongArch64());

  T.setArch(Triple::dxil);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_TRUE(T.isArch32Bit());
  EXPECT_FALSE(T.isArch64Bit());
  EXPECT_TRUE(T.isDXIL());

  T.setArch(Triple::xtensa);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_TRUE(T.isArch32Bit());
  EXPECT_FALSE(T.isArch64Bit());
}

TEST(TripleTest, BitWidthArchVariants) {
  Triple T;
  EXPECT_EQ(Triple::UnknownArch, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::UnknownArch, T.get64BitArchVariant().getArch());

  T.setArch(Triple::UnknownArch);
  EXPECT_EQ(Triple::UnknownArch, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::UnknownArch, T.get64BitArchVariant().getArch());

  T.setArch(Triple::mips);
  EXPECT_EQ(Triple::mips, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::NoSubArch, T.get32BitArchVariant().getSubArch());
  EXPECT_EQ(Triple::mips64, T.get64BitArchVariant().getArch());
  EXPECT_EQ(Triple::NoSubArch, T.get64BitArchVariant().getSubArch());

  T.setArch(Triple::mips, Triple::MipsSubArch_r6);
  EXPECT_EQ(Triple::mips, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::MipsSubArch_r6, T.get32BitArchVariant().getSubArch());
  EXPECT_EQ(Triple::mips64, T.get64BitArchVariant().getArch());
  EXPECT_EQ(Triple::MipsSubArch_r6, T.get64BitArchVariant().getSubArch());

  T.setArch(Triple::mipsel);
  EXPECT_EQ(Triple::mipsel, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::NoSubArch, T.get32BitArchVariant().getSubArch());
  EXPECT_EQ(Triple::mips64el, T.get64BitArchVariant().getArch());
  EXPECT_EQ(Triple::NoSubArch, T.get64BitArchVariant().getSubArch());

  T.setArch(Triple::mipsel, Triple::MipsSubArch_r6);
  EXPECT_EQ(Triple::mipsel, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::MipsSubArch_r6, T.get32BitArchVariant().getSubArch());
  EXPECT_EQ(Triple::mips64el, T.get64BitArchVariant().getArch());
  EXPECT_EQ(Triple::MipsSubArch_r6, T.get64BitArchVariant().getSubArch());

  T.setArch(Triple::ppc);
  EXPECT_EQ(Triple::ppc, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::ppc64, T.get64BitArchVariant().getArch());

  T.setArch(Triple::nvptx);
  EXPECT_EQ(Triple::nvptx, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::nvptx64, T.get64BitArchVariant().getArch());

  T.setArch(Triple::sparc);
  EXPECT_EQ(Triple::sparc, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::sparcv9, T.get64BitArchVariant().getArch());

  T.setArch(Triple::x86);
  EXPECT_EQ(Triple::x86, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::x86_64, T.get64BitArchVariant().getArch());

  T.setArch(Triple::mips64);
  EXPECT_EQ(Triple::mips, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::NoSubArch, T.get32BitArchVariant().getSubArch());
  EXPECT_EQ(Triple::mips64, T.get64BitArchVariant().getArch());
  EXPECT_EQ(Triple::NoSubArch, T.get64BitArchVariant().getSubArch());

  T.setArch(Triple::mips64, Triple::MipsSubArch_r6);
  EXPECT_EQ(Triple::mips, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::MipsSubArch_r6, T.get32BitArchVariant().getSubArch());
  EXPECT_EQ(Triple::mips64, T.get64BitArchVariant().getArch());
  EXPECT_EQ(Triple::MipsSubArch_r6, T.get64BitArchVariant().getSubArch());

  T.setArch(Triple::mips64el);
  EXPECT_EQ(Triple::mipsel, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::NoSubArch, T.get32BitArchVariant().getSubArch());
  EXPECT_EQ(Triple::mips64el, T.get64BitArchVariant().getArch());
  EXPECT_EQ(Triple::NoSubArch, T.get64BitArchVariant().getSubArch());

  T.setArch(Triple::mips64el, Triple::MipsSubArch_r6);
  EXPECT_EQ(Triple::mipsel, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::MipsSubArch_r6, T.get32BitArchVariant().getSubArch());
  EXPECT_EQ(Triple::mips64el, T.get64BitArchVariant().getArch());
  EXPECT_EQ(Triple::MipsSubArch_r6, T.get64BitArchVariant().getSubArch());

  T.setArch(Triple::ppc64);
  EXPECT_EQ(Triple::ppc, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::ppc64, T.get64BitArchVariant().getArch());

  T.setArch(Triple::nvptx64);
  EXPECT_EQ(Triple::nvptx, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::nvptx64, T.get64BitArchVariant().getArch());

  T.setArch(Triple::sparcv9);
  EXPECT_EQ(Triple::sparc, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::sparcv9, T.get64BitArchVariant().getArch());

  T.setArch(Triple::x86_64);
  EXPECT_EQ(Triple::x86, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::x86_64, T.get64BitArchVariant().getArch());

  T.setArch(Triple::amdil);
  EXPECT_EQ(Triple::amdil, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::amdil64, T.get64BitArchVariant().getArch());

  T.setArch(Triple::amdil64);
  EXPECT_EQ(Triple::amdil, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::amdil64, T.get64BitArchVariant().getArch());

  T.setArch(Triple::hsail);
  EXPECT_EQ(Triple::hsail, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::hsail64, T.get64BitArchVariant().getArch());

  T.setArch(Triple::hsail64);
  EXPECT_EQ(Triple::hsail, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::hsail64, T.get64BitArchVariant().getArch());

  T.setArch(Triple::spir);
  EXPECT_EQ(Triple::spir, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::spir64, T.get64BitArchVariant().getArch());

  T.setArch(Triple::spir64);
  EXPECT_EQ(Triple::spir, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::spir64, T.get64BitArchVariant().getArch());

  T.setArch(Triple::spirv);
  EXPECT_EQ(Triple::spirv32, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::spirv64, T.get64BitArchVariant().getArch());

  T.setArch(Triple::spirv32);
  EXPECT_EQ(Triple::spirv32, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::spirv64, T.get64BitArchVariant().getArch());

  T.setArch(Triple::spirv64);
  EXPECT_EQ(Triple::spirv32, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::spirv64, T.get64BitArchVariant().getArch());

  T.setArch(Triple::wasm32);
  EXPECT_EQ(Triple::wasm32, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::wasm64, T.get64BitArchVariant().getArch());

  T.setArch(Triple::wasm64);
  EXPECT_EQ(Triple::wasm32, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::wasm64, T.get64BitArchVariant().getArch());

  T.setArch(Triple::riscv32);
  EXPECT_EQ(Triple::riscv32, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::riscv64, T.get64BitArchVariant().getArch());

  T.setArch(Triple::riscv64);
  EXPECT_EQ(Triple::riscv32, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::riscv64, T.get64BitArchVariant().getArch());

  T.setArch(Triple::csky);
  EXPECT_EQ(Triple::csky, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::UnknownArch, T.get64BitArchVariant().getArch());

  T.setArch(Triple::loongarch32);
  EXPECT_EQ(Triple::loongarch32, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::loongarch64, T.get64BitArchVariant().getArch());

  T.setArch(Triple::loongarch64);
  EXPECT_EQ(Triple::loongarch32, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::loongarch64, T.get64BitArchVariant().getArch());

  T.setArch(Triple::thumbeb);
  EXPECT_EQ(Triple::thumbeb, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::aarch64_be, T.get64BitArchVariant().getArch());

  T.setArch(Triple::thumb);
  EXPECT_EQ(Triple::thumb, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::aarch64, T.get64BitArchVariant().getArch());

  T.setArch(Triple::aarch64);
  EXPECT_EQ(Triple::arm, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::aarch64, T.get64BitArchVariant().getArch());

  T.setArch(Triple::aarch64_be);
  EXPECT_EQ(Triple::armeb, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::aarch64_be, T.get64BitArchVariant().getArch());

  T.setArch(Triple::renderscript32);
  EXPECT_EQ(Triple::renderscript32, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::renderscript64, T.get64BitArchVariant().getArch());

  T.setArch(Triple::renderscript64);
  EXPECT_EQ(Triple::renderscript32, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::renderscript64, T.get64BitArchVariant().getArch());

  T.setArch(Triple::armeb);
  EXPECT_EQ(Triple::armeb, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::aarch64_be, T.get64BitArchVariant().getArch());

  T.setArch(Triple::arm);
  EXPECT_EQ(Triple::arm, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::aarch64, T.get64BitArchVariant().getArch());

  T.setArch(Triple::systemz);
  EXPECT_EQ(Triple::UnknownArch, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::systemz, T.get64BitArchVariant().getArch());

  T.setArch(Triple::xcore);
  EXPECT_EQ(Triple::xcore, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::UnknownArch, T.get64BitArchVariant().getArch());

  T.setArch(Triple::dxil);
  EXPECT_EQ(Triple::dxil, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::UnknownArch, T.get64BitArchVariant().getArch());

  T.setArch(Triple::xtensa);
  EXPECT_EQ(Triple::xtensa, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::UnknownArch, T.get64BitArchVariant().getArch());
}

TEST(TripleTest, EndianArchVariants) {
  Triple T;
  EXPECT_EQ(Triple::UnknownArch, T.getBigEndianArchVariant().getArch());
  EXPECT_EQ(Triple::UnknownArch, T.getLittleEndianArchVariant().getArch());

  T.setArch(Triple::UnknownArch);
  EXPECT_EQ(Triple::UnknownArch, T.getBigEndianArchVariant().getArch());
  EXPECT_EQ(Triple::UnknownArch, T.getLittleEndianArchVariant().getArch());

  T.setArch(Triple::aarch64_be);
  EXPECT_EQ(Triple::aarch64_be, T.getBigEndianArchVariant().getArch());
  EXPECT_EQ(Triple::aarch64, T.getLittleEndianArchVariant().getArch());

  T.setArch(Triple::aarch64);
  EXPECT_EQ(Triple::aarch64_be, T.getBigEndianArchVariant().getArch());
  EXPECT_EQ(Triple::aarch64, T.getLittleEndianArchVariant().getArch());

  T.setArch(Triple::armeb);
  EXPECT_EQ(Triple::armeb, T.getBigEndianArchVariant().getArch());
  EXPECT_EQ(Triple::UnknownArch, T.getLittleEndianArchVariant().getArch());

  T.setArch(Triple::arm);
  EXPECT_EQ(Triple::UnknownArch, T.getBigEndianArchVariant().getArch());
  EXPECT_EQ(Triple::arm, T.getLittleEndianArchVariant().getArch());
  T = Triple("arm");
  EXPECT_TRUE(T.isLittleEndian());
  T = Triple("thumb");
  EXPECT_TRUE(T.isLittleEndian());
  T = Triple("armeb");
  EXPECT_FALSE(T.isLittleEndian());
  T = Triple("thumbeb");
  EXPECT_FALSE(T.isLittleEndian());

  T.setArch(Triple::bpfeb);
  EXPECT_EQ(Triple::bpfeb, T.getBigEndianArchVariant().getArch());
  EXPECT_EQ(Triple::bpfel, T.getLittleEndianArchVariant().getArch());

  T.setArch(Triple::bpfel);
  EXPECT_EQ(Triple::bpfeb, T.getBigEndianArchVariant().getArch());
  EXPECT_EQ(Triple::bpfel, T.getLittleEndianArchVariant().getArch());

  T.setArch(Triple::mips64);
  EXPECT_EQ(Triple::mips64, T.getBigEndianArchVariant().getArch());
  EXPECT_EQ(Triple::NoSubArch, T.getBigEndianArchVariant().getSubArch());
  EXPECT_EQ(Triple::mips64el, T.getLittleEndianArchVariant().getArch());
  EXPECT_EQ(Triple::NoSubArch, T.getLittleEndianArchVariant().getSubArch());

  T.setArch(Triple::mips64, Triple::MipsSubArch_r6);
  EXPECT_EQ(Triple::mips64, T.getBigEndianArchVariant().getArch());
  EXPECT_EQ(Triple::MipsSubArch_r6, T.getBigEndianArchVariant().getSubArch());
  EXPECT_EQ(Triple::mips64el, T.getLittleEndianArchVariant().getArch());
  EXPECT_EQ(Triple::MipsSubArch_r6,
            T.getLittleEndianArchVariant().getSubArch());

  T.setArch(Triple::mips64el);
  EXPECT_EQ(Triple::mips64, T.getBigEndianArchVariant().getArch());
  EXPECT_EQ(Triple::NoSubArch, T.getBigEndianArchVariant().getSubArch());
  EXPECT_EQ(Triple::mips64el, T.getLittleEndianArchVariant().getArch());
  EXPECT_EQ(Triple::NoSubArch, T.getLittleEndianArchVariant().getSubArch());

  T.setArch(Triple::mips64el, Triple::MipsSubArch_r6);
  EXPECT_EQ(Triple::mips64, T.getBigEndianArchVariant().getArch());
  EXPECT_EQ(Triple::MipsSubArch_r6, T.getBigEndianArchVariant().getSubArch());
  EXPECT_EQ(Triple::mips64el, T.getLittleEndianArchVariant().getArch());
  EXPECT_EQ(Triple::MipsSubArch_r6,
            T.getLittleEndianArchVariant().getSubArch());

  T.setArch(Triple::mips);
  EXPECT_EQ(Triple::mips, T.getBigEndianArchVariant().getArch());
  EXPECT_EQ(Triple::NoSubArch, T.getBigEndianArchVariant().getSubArch());
  EXPECT_EQ(Triple::mipsel, T.getLittleEndianArchVariant().getArch());
  EXPECT_EQ(Triple::NoSubArch, T.getLittleEndianArchVariant().getSubArch());

  T.setArch(Triple::mips, Triple::MipsSubArch_r6);
  EXPECT_EQ(Triple::mips, T.getBigEndianArchVariant().getArch());
  EXPECT_EQ(Triple::MipsSubArch_r6, T.getBigEndianArchVariant().getSubArch());
  EXPECT_EQ(Triple::mipsel, T.getLittleEndianArchVariant().getArch());
  EXPECT_EQ(Triple::MipsSubArch_r6,
            T.getLittleEndianArchVariant().getSubArch());

  T.setArch(Triple::mipsel);
  EXPECT_EQ(Triple::mips, T.getBigEndianArchVariant().getArch());
  EXPECT_EQ(Triple::NoSubArch, T.getBigEndianArchVariant().getSubArch());
  EXPECT_EQ(Triple::mipsel, T.getLittleEndianArchVariant().getArch());
  EXPECT_EQ(Triple::NoSubArch, T.getLittleEndianArchVariant().getSubArch());

  T.setArch(Triple::mipsel, Triple::MipsSubArch_r6);
  EXPECT_EQ(Triple::mips, T.getBigEndianArchVariant().getArch());
  EXPECT_EQ(Triple::MipsSubArch_r6, T.getBigEndianArchVariant().getSubArch());
  EXPECT_EQ(Triple::mipsel, T.getLittleEndianArchVariant().getArch());
  EXPECT_EQ(Triple::MipsSubArch_r6,
            T.getLittleEndianArchVariant().getSubArch());

  T.setArch(Triple::ppc);
  EXPECT_EQ(Triple::ppc, T.getBigEndianArchVariant().getArch());
  EXPECT_EQ(Triple::ppcle, T.getLittleEndianArchVariant().getArch());

  T.setArch(Triple::ppc64);
  EXPECT_EQ(Triple::ppc64, T.getBigEndianArchVariant().getArch());
  EXPECT_EQ(Triple::ppc64le, T.getLittleEndianArchVariant().getArch());

  T.setArch(Triple::ppc64le);
  EXPECT_EQ(Triple::ppc64, T.getBigEndianArchVariant().getArch());
  EXPECT_EQ(Triple::ppc64le, T.getLittleEndianArchVariant().getArch());

  T.setArch(Triple::sparc);
  EXPECT_EQ(Triple::sparc, T.getBigEndianArchVariant().getArch());
  EXPECT_EQ(Triple::sparcel, T.getLittleEndianArchVariant().getArch());

  T.setArch(Triple::sparcel);
  EXPECT_EQ(Triple::sparc, T.getBigEndianArchVariant().getArch());
  EXPECT_EQ(Triple::sparcel, T.getLittleEndianArchVariant().getArch());

  T.setArch(Triple::thumb);
  EXPECT_EQ(Triple::UnknownArch, T.getBigEndianArchVariant().getArch());
  EXPECT_EQ(Triple::thumb, T.getLittleEndianArchVariant().getArch());

  T.setArch(Triple::thumbeb);
  EXPECT_EQ(Triple::thumbeb, T.getBigEndianArchVariant().getArch());
  EXPECT_EQ(Triple::UnknownArch, T.getLittleEndianArchVariant().getArch());

  T.setArch(Triple::lanai);
  EXPECT_EQ(Triple::lanai, T.getBigEndianArchVariant().getArch());
  EXPECT_EQ(Triple::UnknownArch, T.getLittleEndianArchVariant().getArch());

  T.setArch(Triple::tcele);
  EXPECT_EQ(Triple::tce, T.getBigEndianArchVariant().getArch());
  EXPECT_EQ(Triple::tcele, T.getLittleEndianArchVariant().getArch());

  T.setArch(Triple::tce);
  EXPECT_EQ(Triple::tce, T.getBigEndianArchVariant().getArch());
  EXPECT_EQ(Triple::tcele, T.getLittleEndianArchVariant().getArch());

  T.setArch(Triple::csky);
  EXPECT_EQ(Triple::UnknownArch, T.getBigEndianArchVariant().getArch());
  EXPECT_EQ(Triple::csky, T.getLittleEndianArchVariant().getArch());

  T.setArch(Triple::loongarch32);
  EXPECT_TRUE(T.isLittleEndian());
  EXPECT_EQ(Triple::UnknownArch, T.getBigEndianArchVariant().getArch());
  EXPECT_EQ(Triple::loongarch32, T.getLittleEndianArchVariant().getArch());

  T.setArch(Triple::loongarch64);
  EXPECT_TRUE(T.isLittleEndian());
  EXPECT_EQ(Triple::UnknownArch, T.getBigEndianArchVariant().getArch());
  EXPECT_EQ(Triple::loongarch64, T.getLittleEndianArchVariant().getArch());

  T.setArch(Triple::dxil);
  EXPECT_TRUE(T.isLittleEndian());
  EXPECT_EQ(Triple::UnknownArch, T.getBigEndianArchVariant().getArch());
  EXPECT_EQ(Triple::dxil, T.getLittleEndianArchVariant().getArch());
}

TEST(TripleTest, XROS) {
  Triple T;
  VersionTuple Version;

  T = Triple("arm64-apple-xros");
  EXPECT_TRUE(T.isXROS());
  EXPECT_TRUE(T.isOSDarwin());
  EXPECT_FALSE(T.isiOS());
  EXPECT_FALSE(T.isMacOSX());
  EXPECT_FALSE(T.isSimulatorEnvironment());
  EXPECT_EQ(T.getOSName(), "xros");
  Version = T.getOSVersion();
  EXPECT_EQ(VersionTuple(0), Version);

  T = Triple("arm64-apple-visionos1.2");
  EXPECT_TRUE(T.isXROS());
  EXPECT_TRUE(T.isOSDarwin());
  EXPECT_FALSE(T.isiOS());
  EXPECT_FALSE(T.isMacOSX());
  EXPECT_FALSE(T.isSimulatorEnvironment());
  EXPECT_EQ(T.getOSName(), "visionos1.2");
  Version = T.getOSVersion();
  EXPECT_EQ(VersionTuple(1, 2), Version);

  T = Triple("arm64-apple-xros1-simulator");
  EXPECT_TRUE(T.isXROS());
  EXPECT_TRUE(T.isOSDarwin());
  EXPECT_FALSE(T.isiOS());
  EXPECT_FALSE(T.isMacOSX());
  EXPECT_TRUE(T.isSimulatorEnvironment());
  Version = T.getOSVersion();
  EXPECT_EQ(VersionTuple(1), Version);
  Version = T.getiOSVersion();
  EXPECT_EQ(VersionTuple(17), Version);
}

TEST(TripleTest, getOSVersion) {
  Triple T;
  VersionTuple Version;

  T = Triple("i386-apple-darwin9");
  EXPECT_TRUE(T.isMacOSX());
  EXPECT_FALSE(T.isiOS());
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_TRUE(T.isArch32Bit());
  EXPECT_FALSE(T.isArch64Bit());
  T.getMacOSXVersion(Version);
  EXPECT_EQ(VersionTuple(10, 5), Version);
  Version = T.getiOSVersion();
  EXPECT_EQ(VersionTuple(5), Version);

  T = Triple("x86_64-apple-darwin9");
  EXPECT_TRUE(T.isMacOSX());
  EXPECT_FALSE(T.isiOS());
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_FALSE(T.isArch32Bit());
  EXPECT_TRUE(T.isArch64Bit());
  T.getMacOSXVersion(Version);
  EXPECT_EQ(VersionTuple(10, 5), Version);
  Version = T.getiOSVersion();
  EXPECT_EQ(VersionTuple(5), Version);

  T = Triple("x86_64-apple-macosx");
  EXPECT_TRUE(T.isMacOSX());
  EXPECT_FALSE(T.isiOS());
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_FALSE(T.isArch32Bit());
  EXPECT_TRUE(T.isArch64Bit());
  T.getMacOSXVersion(Version);
  EXPECT_EQ(VersionTuple(10, 4), Version);
  Version = T.getiOSVersion();
  EXPECT_EQ(VersionTuple(5), Version);

  T = Triple("x86_64-apple-macosx10.7");
  EXPECT_TRUE(T.isMacOSX());
  EXPECT_FALSE(T.isiOS());
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_FALSE(T.isArch32Bit());
  EXPECT_TRUE(T.isArch64Bit());
  T.getMacOSXVersion(Version);
  EXPECT_EQ(VersionTuple(10, 7), Version);
  Version = T.getiOSVersion();
  EXPECT_EQ(VersionTuple(5), Version);

  T = Triple("x86_64-apple-macos11.0");
  EXPECT_TRUE(T.isMacOSX());
  EXPECT_FALSE(T.isiOS());
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_FALSE(T.isArch32Bit());
  EXPECT_TRUE(T.isArch64Bit());
  T.getMacOSXVersion(Version);
  EXPECT_EQ(VersionTuple(11, 0), Version);

  T = Triple("arm64-apple-macosx11.5.8");
  EXPECT_TRUE(T.isMacOSX());
  EXPECT_FALSE(T.isiOS());
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_FALSE(T.isArch32Bit());
  EXPECT_TRUE(T.isArch64Bit());
  T.getMacOSXVersion(Version);
  EXPECT_EQ(VersionTuple(11, 5, 8), Version);

  // 10.16 forms a valid triple, even though it's not
  // a version of a macOS.
  T = Triple("x86_64-apple-macos10.16");
  EXPECT_TRUE(T.isMacOSX());
  T.getMacOSXVersion(Version);
  EXPECT_EQ(VersionTuple(10, 16), Version);

  T = Triple("x86_64-apple-darwin20");
  EXPECT_TRUE(T.isMacOSX());
  T.getMacOSXVersion(Version);
  EXPECT_EQ(VersionTuple(11), Version);

  // For darwin triples on macOS 11, only compare the major version.
  T = Triple("x86_64-apple-darwin20.2");
  EXPECT_TRUE(T.isMacOSX());
  T.getMacOSXVersion(Version);
  EXPECT_EQ(VersionTuple(11), Version);

  T = Triple("armv7-apple-ios");
  EXPECT_FALSE(T.isMacOSX());
  EXPECT_TRUE(T.isiOS());
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_TRUE(T.isArch32Bit());
  EXPECT_FALSE(T.isArch64Bit());
  T.getMacOSXVersion(Version);
  EXPECT_EQ(VersionTuple(10, 4), Version);
  Version = T.getiOSVersion();
  EXPECT_EQ(VersionTuple(5), Version);

  T = Triple("armv7-apple-ios7.0");
  EXPECT_FALSE(T.isMacOSX());
  EXPECT_TRUE(T.isiOS());
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_TRUE(T.isArch32Bit());
  EXPECT_FALSE(T.isArch64Bit());
  T.getMacOSXVersion(Version);
  EXPECT_EQ(VersionTuple(10, 4), Version);
  Version = T.getiOSVersion();
  EXPECT_EQ(VersionTuple(7, 0), Version);
  EXPECT_FALSE(T.isSimulatorEnvironment());

  T = Triple("x86_64-apple-ios10.3-simulator");
  EXPECT_TRUE(T.isiOS());
  Version = T.getiOSVersion();
  EXPECT_EQ(VersionTuple(10, 3), Version);
  EXPECT_TRUE(T.isSimulatorEnvironment());
  EXPECT_FALSE(T.isMacCatalystEnvironment());

  T = Triple("x86_64-apple-ios13.0-macabi");
  EXPECT_TRUE(T.isiOS());
  Version = T.getiOSVersion();
  EXPECT_EQ(VersionTuple(13, 0), Version);
  EXPECT_TRUE(T.getEnvironment() == Triple::MacABI);
  EXPECT_TRUE(T.isMacCatalystEnvironment());
  EXPECT_FALSE(T.isSimulatorEnvironment());

  T = Triple("x86_64-apple-driverkit20.1.0");
  EXPECT_TRUE(T.isDriverKit());
  EXPECT_TRUE(T.isOSDarwin());
  EXPECT_FALSE(T.isMacOSX());
  EXPECT_FALSE(T.isiOS());
  Version = T.getDriverKitVersion();
  EXPECT_EQ(VersionTuple(20, 1), Version);

  T = Triple("x86_64-apple-driverkit20");
  Version = T.getDriverKitVersion();
  EXPECT_EQ(VersionTuple(20, 0), Version);

  // DriverKit version should default to 19.0.
  T = Triple("x86_64-apple-driverkit");
  Version = T.getDriverKitVersion();
  EXPECT_EQ(VersionTuple(19, 0), Version);

  T = Triple("dxil-unknown-shadermodel6.6-pixel");
  EXPECT_EQ(Triple::dxil, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::ShaderModel, T.getOS());
  Version = T.getOSVersion();
  EXPECT_EQ(VersionTuple(6, 6), Version);
  EXPECT_EQ(Triple::Pixel, T.getEnvironment());

  T = Triple("dxil-unknown-shadermodel6.0-pixel");
  EXPECT_EQ(Triple::dxil, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::ShaderModel, T.getOS());
  Version = T.getOSVersion();
  EXPECT_EQ(VersionTuple(6, 0), Version);
  EXPECT_EQ(Triple::Pixel, T.getEnvironment());
}

TEST(TripleTest, getEnvironmentVersion) {
  Triple T;
  VersionTuple Version;

  T = Triple("arm-unknown-linux-android16");
  EXPECT_TRUE(T.isAndroid());
  Version = T.getEnvironmentVersion();
  EXPECT_EQ(VersionTuple(16), Version);
  EXPECT_EQ(Triple::Android, T.getEnvironment());

  T = Triple("aarch64-unknown-linux-android21");
  EXPECT_TRUE(T.isAndroid());
  Version = T.getEnvironmentVersion();
  EXPECT_EQ(VersionTuple(21), Version);
  EXPECT_EQ(Triple::Android, T.getEnvironment());
}

TEST(TripleTest, isMacOSVersionLT) {
  Triple T = Triple("x86_64-apple-macos11");
  EXPECT_TRUE(T.isMacOSXVersionLT(11, 1, 0));
  EXPECT_FALSE(T.isMacOSXVersionLT(10, 15, 0));

  T = Triple("x86_64-apple-darwin20");
  EXPECT_TRUE(T.isMacOSXVersionLT(11, 1, 0));
  EXPECT_FALSE(T.isMacOSXVersionLT(11, 0, 0));
  EXPECT_FALSE(T.isMacOSXVersionLT(10, 15, 0));
}

TEST(TripleTest, CanonicalizeOSVersion) {
  EXPECT_EQ(VersionTuple(10, 15, 4),
            Triple::getCanonicalVersionForOS(Triple::MacOSX,
                                             VersionTuple(10, 15, 4)));
  EXPECT_EQ(VersionTuple(11, 0), Triple::getCanonicalVersionForOS(
                                     Triple::MacOSX, VersionTuple(10, 16)));
  EXPECT_EQ(VersionTuple(20),
            Triple::getCanonicalVersionForOS(Triple::Darwin, VersionTuple(20)));
}

TEST(TripleTest, FileFormat) {
  EXPECT_EQ(Triple::ELF, Triple("i686-unknown-linux-gnu").getObjectFormat());
  EXPECT_EQ(Triple::ELF, Triple("i686-unknown-freebsd").getObjectFormat());
  EXPECT_EQ(Triple::ELF, Triple("i686-unknown-netbsd").getObjectFormat());
  EXPECT_EQ(Triple::ELF, Triple("i686--win32-elf").getObjectFormat());
  EXPECT_EQ(Triple::ELF, Triple("i686---elf").getObjectFormat());

  EXPECT_EQ(Triple::MachO, Triple("i686-apple-macosx").getObjectFormat());
  EXPECT_EQ(Triple::MachO, Triple("i686-apple-ios").getObjectFormat());
  EXPECT_EQ(Triple::MachO, Triple("i686---macho").getObjectFormat());
  EXPECT_EQ(Triple::MachO, Triple("powerpc-apple-macosx").getObjectFormat());

  EXPECT_EQ(Triple::COFF, Triple("i686--win32").getObjectFormat());

  EXPECT_EQ(Triple::ELF, Triple("i686-pc-windows-msvc-elf").getObjectFormat());
  EXPECT_EQ(Triple::ELF, Triple("i686-pc-cygwin-elf").getObjectFormat());

  EXPECT_EQ(Triple::ELF, Triple("systemz-ibm-linux").getObjectFormat());
  EXPECT_EQ(Triple::ELF, Triple("systemz-ibm-unknown").getObjectFormat());

  EXPECT_EQ(Triple::GOFF, Triple("s390x-ibm-zos").getObjectFormat());
  EXPECT_EQ(Triple::GOFF, Triple("systemz-ibm-zos").getObjectFormat());
  EXPECT_EQ(Triple::GOFF, Triple("s390x-ibm-zos-goff").getObjectFormat());
  EXPECT_EQ(Triple::GOFF, Triple("s390x-unknown-zos-goff").getObjectFormat());
  EXPECT_EQ(Triple::GOFF, Triple("s390x---goff").getObjectFormat());

  EXPECT_EQ(Triple::Wasm, Triple("wasm32-unknown-unknown").getObjectFormat());
  EXPECT_EQ(Triple::Wasm, Triple("wasm64-unknown-unknown").getObjectFormat());
  EXPECT_EQ(Triple::Wasm, Triple("wasm32-wasi").getObjectFormat());
  EXPECT_EQ(Triple::Wasm, Triple("wasm64-wasi").getObjectFormat());
  EXPECT_EQ(Triple::Wasm, Triple("wasm32-unknown-wasi").getObjectFormat());
  EXPECT_EQ(Triple::Wasm, Triple("wasm64-unknown-wasi").getObjectFormat());

  EXPECT_EQ(Triple::Wasm,
            Triple("wasm32-unknown-unknown-wasm").getObjectFormat());
  EXPECT_EQ(Triple::Wasm,
            Triple("wasm64-unknown-unknown-wasm").getObjectFormat());
  EXPECT_EQ(Triple::Wasm,
            Triple("wasm32-wasi-wasm").getObjectFormat());
  EXPECT_EQ(Triple::Wasm,
            Triple("wasm64-wasi-wasm").getObjectFormat());
  EXPECT_EQ(Triple::Wasm,
            Triple("wasm32-unknown-wasi-wasm").getObjectFormat());
  EXPECT_EQ(Triple::Wasm,
            Triple("wasm64-unknown-wasi-wasm").getObjectFormat());

  EXPECT_EQ(Triple::XCOFF, Triple("powerpc-ibm-aix").getObjectFormat());
  EXPECT_EQ(Triple::XCOFF, Triple("powerpc64-ibm-aix").getObjectFormat());
  EXPECT_EQ(Triple::XCOFF, Triple("powerpc---xcoff").getObjectFormat());
  EXPECT_EQ(Triple::XCOFF, Triple("powerpc64---xcoff").getObjectFormat());

  EXPECT_EQ(Triple::ELF, Triple("csky-unknown-unknown").getObjectFormat());
  EXPECT_EQ(Triple::ELF, Triple("csky-unknown-linux").getObjectFormat());

  EXPECT_EQ(Triple::SPIRV, Triple("spirv-unknown-unknown").getObjectFormat());
  EXPECT_EQ(Triple::SPIRV, Triple("spirv32-unknown-unknown").getObjectFormat());
  EXPECT_EQ(Triple::SPIRV, Triple("spirv64-unknown-unknown").getObjectFormat());

  EXPECT_EQ(Triple::ELF,
            Triple("loongarch32-unknown-unknown").getObjectFormat());
  EXPECT_EQ(Triple::ELF, Triple("loongarch64-unknown-linux").getObjectFormat());

  Triple MSVCNormalized(Triple::normalize("i686-pc-windows-msvc-elf"));
  EXPECT_EQ(Triple::ELF, MSVCNormalized.getObjectFormat());

  Triple GNUWindowsNormalized(Triple::normalize("i686-pc-windows-gnu-elf"));
  EXPECT_EQ(Triple::ELF, GNUWindowsNormalized.getObjectFormat());

  Triple CygnusNormalised(Triple::normalize("i686-pc-windows-cygnus-elf"));
  EXPECT_EQ(Triple::ELF, CygnusNormalised.getObjectFormat());

  Triple CygwinNormalized(Triple::normalize("i686-pc-cygwin-elf"));
  EXPECT_EQ(Triple::ELF, CygwinNormalized.getObjectFormat());

  EXPECT_EQ(Triple::DXContainer,
            Triple("dxil-unknown-shadermodel").getObjectFormat());

  Triple T = Triple("");
  T.setObjectFormat(Triple::ELF);
  EXPECT_EQ(Triple::ELF, T.getObjectFormat());
  EXPECT_EQ("elf", Triple::getObjectFormatTypeName(T.getObjectFormat()));

  T.setObjectFormat(Triple::MachO);
  EXPECT_EQ(Triple::MachO, T.getObjectFormat());
  EXPECT_EQ("macho", Triple::getObjectFormatTypeName(T.getObjectFormat()));

  T.setObjectFormat(Triple::XCOFF);
  EXPECT_EQ(Triple::XCOFF, T.getObjectFormat());
  EXPECT_EQ("xcoff", Triple::getObjectFormatTypeName(T.getObjectFormat()));

  T.setObjectFormat(Triple::GOFF);
  EXPECT_EQ(Triple::GOFF, T.getObjectFormat());
  EXPECT_EQ("goff", Triple::getObjectFormatTypeName(T.getObjectFormat()));

  T.setObjectFormat(Triple::SPIRV);
  EXPECT_EQ(Triple::SPIRV, T.getObjectFormat());
  EXPECT_EQ("spirv", Triple::getObjectFormatTypeName(T.getObjectFormat()));

  EXPECT_EQ(Triple::ELF, Triple("amdgcn-apple-macosx").getObjectFormat());
  EXPECT_EQ(Triple::ELF, Triple("r600-apple-macosx").getObjectFormat());
  EXPECT_EQ(Triple::SPIRV, Triple("spirv-apple-macosx").getObjectFormat());
  EXPECT_EQ(Triple::SPIRV, Triple("spirv32-apple-macosx").getObjectFormat());
  EXPECT_EQ(Triple::SPIRV, Triple("spirv64-apple-macosx").getObjectFormat());
  EXPECT_EQ(Triple::DXContainer, Triple("dxil-apple-macosx").getObjectFormat());
}

TEST(TripleTest, NormalizeWindows) {
  EXPECT_EQ("i686-pc-windows-msvc", Triple::normalize("i686-pc-win32"));
  EXPECT_EQ("i686-unknown-windows-msvc", Triple::normalize("i686-win32"));
  EXPECT_EQ("i686-pc-windows-gnu", Triple::normalize("i686-pc-mingw32"));
  EXPECT_EQ("i686-unknown-windows-gnu", Triple::normalize("i686-mingw32"));
  EXPECT_EQ("i686-pc-windows-gnu", Triple::normalize("i686-pc-mingw32-w64"));
  EXPECT_EQ("i686-unknown-windows-gnu", Triple::normalize("i686-mingw32-w64"));
  EXPECT_EQ("i686-pc-windows-cygnus", Triple::normalize("i686-pc-cygwin"));
  EXPECT_EQ("i686-unknown-windows-cygnus", Triple::normalize("i686-cygwin"));

  EXPECT_EQ("x86_64-pc-windows-msvc", Triple::normalize("x86_64-pc-win32"));
  EXPECT_EQ("x86_64-unknown-windows-msvc", Triple::normalize("x86_64-win32"));
  EXPECT_EQ("x86_64-pc-windows-gnu", Triple::normalize("x86_64-pc-mingw32"));
  EXPECT_EQ("x86_64-unknown-windows-gnu", Triple::normalize("x86_64-mingw32"));
  EXPECT_EQ("x86_64-pc-windows-gnu",
            Triple::normalize("x86_64-pc-mingw32-w64"));
  EXPECT_EQ("x86_64-unknown-windows-gnu",
            Triple::normalize("x86_64-mingw32-w64"));

  EXPECT_EQ("i686-pc-windows-elf", Triple::normalize("i686-pc-win32-elf"));
  EXPECT_EQ("i686-unknown-windows-elf", Triple::normalize("i686-win32-elf"));
  EXPECT_EQ("i686-pc-windows-macho", Triple::normalize("i686-pc-win32-macho"));
  EXPECT_EQ("i686-unknown-windows-macho",
            Triple::normalize("i686-win32-macho"));

  EXPECT_EQ("x86_64-pc-windows-elf", Triple::normalize("x86_64-pc-win32-elf"));
  EXPECT_EQ("x86_64-unknown-windows-elf",
            Triple::normalize("x86_64-win32-elf"));
  EXPECT_EQ("x86_64-pc-windows-macho",
            Triple::normalize("x86_64-pc-win32-macho"));
  EXPECT_EQ("x86_64-unknown-windows-macho",
            Triple::normalize("x86_64-win32-macho"));

  EXPECT_EQ("i686-pc-windows-cygnus",
            Triple::normalize("i686-pc-windows-cygnus"));
  EXPECT_EQ("i686-pc-windows-gnu", Triple::normalize("i686-pc-windows-gnu"));
  EXPECT_EQ("i686-pc-windows-itanium",
            Triple::normalize("i686-pc-windows-itanium"));
  EXPECT_EQ("i686-pc-windows-msvc", Triple::normalize("i686-pc-windows-msvc"));

  EXPECT_EQ("i686-pc-windows-elf",
            Triple::normalize("i686-pc-windows-elf-elf"));

  EXPECT_TRUE(Triple("x86_64-pc-win32").isWindowsMSVCEnvironment());

  EXPECT_TRUE(Triple(Triple::normalize("mipsel-windows-msvccoff")).isOSBinFormatCOFF());
  EXPECT_TRUE(Triple(Triple::normalize("mipsel-windows-msvc")).isOSBinFormatCOFF());
  EXPECT_TRUE(Triple(Triple::normalize("mipsel-windows-gnu")).isOSBinFormatCOFF());
}

TEST(TripleTest, NormalizeAndroid) {
  EXPECT_EQ("arm-unknown-linux-android16",
            Triple::normalize("arm-linux-androideabi16"));
  EXPECT_EQ("armv7a-unknown-linux-android",
            Triple::normalize("armv7a-linux-androideabi"));
  EXPECT_EQ("aarch64-unknown-linux-android21",
            Triple::normalize("aarch64-linux-android21"));
}

TEST(TripleTest, NormalizeARM) {
  EXPECT_EQ("armv6-unknown-netbsd-eabi",
            Triple::normalize("armv6-netbsd-eabi"));
  EXPECT_EQ("armv7-unknown-netbsd-eabi",
            Triple::normalize("armv7-netbsd-eabi"));
  EXPECT_EQ("armv6eb-unknown-netbsd-eabi",
            Triple::normalize("armv6eb-netbsd-eabi"));
  EXPECT_EQ("armv7eb-unknown-netbsd-eabi",
            Triple::normalize("armv7eb-netbsd-eabi"));
  EXPECT_EQ("armv6-unknown-netbsd-eabihf",
            Triple::normalize("armv6-netbsd-eabihf"));
  EXPECT_EQ("armv7-unknown-netbsd-eabihf",
            Triple::normalize("armv7-netbsd-eabihf"));
  EXPECT_EQ("armv6eb-unknown-netbsd-eabihf",
            Triple::normalize("armv6eb-netbsd-eabihf"));
  EXPECT_EQ("armv7eb-unknown-netbsd-eabihf",
            Triple::normalize("armv7eb-netbsd-eabihf"));

  EXPECT_EQ("armv7-suse-linux-gnueabihf",
            Triple::normalize("armv7-suse-linux-gnueabi"));

  Triple T;
  T = Triple("armv6--netbsd-eabi");
  EXPECT_EQ(Triple::arm, T.getArch());
  T = Triple("armv6eb--netbsd-eabi");
  EXPECT_EQ(Triple::armeb, T.getArch());
  T = Triple("armv7-suse-linux-gnueabihf");
  EXPECT_EQ(Triple::GNUEABIHF, T.getEnvironment());
}

TEST(TripleTest, ParseARMArch) {
  // ARM
  {
    Triple T = Triple("arm");
    EXPECT_EQ(Triple::arm, T.getArch());
  }
  {
    Triple T = Triple("armeb");
    EXPECT_EQ(Triple::armeb, T.getArch());
  }
  // THUMB
  {
    Triple T = Triple("thumb");
    EXPECT_EQ(Triple::thumb, T.getArch());
  }
  {
    Triple T = Triple("thumbeb");
    EXPECT_EQ(Triple::thumbeb, T.getArch());
  }
  // AARCH64
  {
    Triple T = Triple("arm64");
    EXPECT_EQ(Triple::aarch64, T.getArch());
  }
  {
    Triple T = Triple("arm64_32");
    EXPECT_EQ(Triple::aarch64_32, T.getArch());
  }
  {
    Triple T = Triple("aarch64");
    EXPECT_EQ(Triple::aarch64, T.getArch());
  }
  {
    Triple T = Triple("aarch64_be");
    EXPECT_EQ(Triple::aarch64_be, T.getArch());
  }
  {
    Triple T = Triple("arm64e");
    EXPECT_EQ(Triple::aarch64, T.getArch());
    EXPECT_EQ(Triple::AArch64SubArch_arm64e, T.getSubArch());
  }
  {
    Triple T = Triple("arm64ec");
    EXPECT_EQ(Triple::aarch64, T.getArch());
    EXPECT_EQ(Triple::AArch64SubArch_arm64ec, T.getSubArch());
  }
  {
    Triple T;
    T.setArch(Triple::aarch64, Triple::AArch64SubArch_arm64ec);
    EXPECT_EQ("arm64ec", T.getArchName());
  }
}

TEST(TripleTest, isArmT32) {
  // Not isArmT32
  {
    Triple T = Triple("thumbv6m");
    EXPECT_FALSE(T.isArmT32());
  }
  {
    Triple T = Triple("armv8m.base");
    EXPECT_FALSE(T.isArmT32());
  }
  {
    Triple T = Triple("armv7s");
    EXPECT_FALSE(T.isArmT32());
  }
  {
    Triple T = Triple("armv7k");
    EXPECT_FALSE(T.isArmT32());
  }
  {
    Triple T = Triple("armv7ve");
    EXPECT_FALSE(T.isArmT32());
  }
  {
    Triple T = Triple("armv6");
    EXPECT_FALSE(T.isArmT32());
  }
  {
    Triple T = Triple("armv6m");
    EXPECT_FALSE(T.isArmT32());
  }
  {
    Triple T = Triple("armv6k");
    EXPECT_FALSE(T.isArmT32());
  }
  {
    Triple T = Triple("armv6t2");
    EXPECT_FALSE(T.isArmT32());
  }
  {
    Triple T = Triple("armv5");
    EXPECT_FALSE(T.isArmT32());
  }
  {
    Triple T = Triple("armv5te");
    EXPECT_FALSE(T.isArmT32());
  }
  {
    Triple T = Triple("armv4t");
    EXPECT_FALSE(T.isArmT32());
  }

  // isArmT32
  {
    Triple T = Triple("arm");
    EXPECT_TRUE(T.isArmT32());
  }
  {
    Triple T = Triple("armv7m");
    EXPECT_TRUE(T.isArmT32());
  }
  {
    Triple T = Triple("armv7em");
    EXPECT_TRUE(T.isArmT32());
  }
  {
    Triple T = Triple("armv8m.main");
    EXPECT_TRUE(T.isArmT32());
  }
  {
    Triple T = Triple("armv8.1m.main");
    EXPECT_TRUE(T.isArmT32());
  }
}

TEST(TripleTest, isArmMClass) {
  // not M-class
  {
    Triple T = Triple("armv7s");
    EXPECT_FALSE(T.isArmMClass());
  }
  {
    Triple T = Triple("armv7k");
    EXPECT_FALSE(T.isArmMClass());
  }
  {
    Triple T = Triple("armv7ve");
    EXPECT_FALSE(T.isArmMClass());
  }
  {
    Triple T = Triple("armv6");
    EXPECT_FALSE(T.isArmMClass());
  }
  {
    Triple T = Triple("armv6k");
    EXPECT_FALSE(T.isArmMClass());
  }
  {
    Triple T = Triple("armv6t2");
    EXPECT_FALSE(T.isArmMClass());
  }
  {
    Triple T = Triple("armv5");
    EXPECT_FALSE(T.isArmMClass());
  }
  {
    Triple T = Triple("armv5te");
    EXPECT_FALSE(T.isArmMClass());
  }
  {
    Triple T = Triple("armv4t");
    EXPECT_FALSE(T.isArmMClass());
  }
  {
    Triple T = Triple("arm");
    EXPECT_FALSE(T.isArmMClass());
  }

  // is M-class
  {
    Triple T = Triple("armv6m");
    EXPECT_TRUE(T.isArmMClass());
  }
  {
    Triple T = Triple("armv7m");
    EXPECT_TRUE(T.isArmMClass());
  }
  {
    Triple T = Triple("armv7em");
    EXPECT_TRUE(T.isArmMClass());
  }
  {
    Triple T = Triple("armv8m.base");
    EXPECT_TRUE(T.isArmMClass());
  }
  {
    Triple T = Triple("armv8m.main");
    EXPECT_TRUE(T.isArmMClass());
  }
  {
    Triple T = Triple("armv8.1m.main");
    EXPECT_TRUE(T.isArmMClass());
  }
}

TEST(TripleTest, DXILNormaizeWithVersion) {
  EXPECT_EQ("dxilv1.0-unknown-shadermodel6.0",
            Triple::normalize("dxilv1.0--shadermodel6.0"));
  EXPECT_EQ("dxilv1.0-unknown-shadermodel6.0",
            Triple::normalize("dxil--shadermodel6.0"));
  EXPECT_EQ("dxilv1.1-unknown-shadermodel6.1-library",
            Triple::normalize("dxil-shadermodel6.1-unknown-library"));
  EXPECT_EQ("dxilv1.8-unknown-shadermodel6.x-unknown",
            Triple::normalize("dxil-unknown-shadermodel6.x-unknown"));
  EXPECT_EQ("dxilv1.8-unknown-shadermodel6.x-unknown",
            Triple::normalize("dxil-unknown-shadermodel6.x-unknown"));
  EXPECT_EQ("dxil-unknown-unknown-unknown", Triple::normalize("dxil---"));
  EXPECT_EQ("dxilv1.0-pc-shadermodel5.0-compute",
            Triple::normalize("dxil-shadermodel5.0-pc-compute"));
}

TEST(TripleTest, isCompatibleWith) {
  struct {
    const char *A;
    const char *B;
    bool Result;
  } Cases[] = {
      {"armv7-linux-gnueabihf", "thumbv7-linux-gnueabihf", true},
      {"armv4-none-unknown-eabi", "thumbv6-unknown-linux-gnueabihf", false},
      {"x86_64-apple-macosx10.9.0", "x86_64-apple-macosx10.10.0", true},
      {"x86_64-apple-macosx10.9.0", "i386-apple-macosx10.9.0", false},
      {"x86_64-apple-macosx10.9.0", "x86_64h-apple-macosx10.9.0", true},
      {"x86_64-unknown-linux-gnu", "x86_64-unknown-linux-gnu", true},
      {"x86_64-unknown-linux-gnu", "i386-unknown-linux-gnu", false},
      {"x86_64-unknown-linux-gnu", "x86_64h-unknown-linux-gnu", true},
      {"x86_64-pc-windows-gnu", "x86_64-pc-windows-msvc", false},
      {"x86_64-pc-windows-msvc", "x86_64-pc-windows-msvc-elf", false},
      {"i686-w64-windows-gnu", "i386-w64-windows-gnu", true},
      {"x86_64-w64-windows-gnu", "x86_64-pc-windows-gnu", true},
      {"armv7-w64-windows-gnu", "thumbv7-pc-windows-gnu", true},
  };

  auto DoTest = [](const char *A, const char *B,
                   bool Result) -> testing::AssertionResult {
    if (Triple(A).isCompatibleWith(Triple(B)) != Result) {
      return testing::AssertionFailure()
             << llvm::formatv("Triple {0} and {1} were expected to be {2}", A,
                              B, Result ? "compatible" : "incompatible");
    }
    return testing::AssertionSuccess();
  };
  for (const auto &C : Cases) {
    EXPECT_TRUE(DoTest(C.A, C.B, C.Result));
    // Test that the comparison is commutative.
    EXPECT_TRUE(DoTest(C.B, C.A, C.Result));
  }
}
} // end anonymous namespace
