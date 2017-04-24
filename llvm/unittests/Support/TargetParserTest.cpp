//===----------- TargetParser.cpp - Target Parser -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ARMBuildAttributes.h"
#include "llvm/Support/TargetParser.h"
#include "gtest/gtest.h"
#include <string>

using namespace llvm;

namespace {
const char *ARMArch[] = {
    "armv2",     "armv2a",       "armv3",       "armv3m",       "armv4",
    "armv4t",    "armv5",        "armv5t",      "armv5e",       "armv5te",
    "armv5tej",  "armv6",        "armv6j",      "armv6k",       "armv6hl",
    "armv6t2",   "armv6kz",      "armv6z",      "armv6zk",      "armv6-m",
    "armv6m",    "armv6sm",      "armv6s-m",    "armv7-a",      "armv7",
    "armv7a",    "armv7ve",      "armv7hl",     "armv7l",       "armv7-r",
    "armv7r",    "armv7-m",      "armv7m",      "armv7k",       "armv7s",
    "armv7e-m",  "armv7em",      "armv8-a",     "armv8",        "armv8a",
    "armv8.1-a", "armv8.1a",     "armv8.2-a",   "armv8.2a",     "armv8-r",
    "armv8r",    "armv8-m.base", "armv8m.base", "armv8-m.main", "armv8m.main",
    "iwmmxt",    "iwmmxt2",      "xscale"};

bool testARMCPU(StringRef CPUName, StringRef ExpectedArch,
                StringRef ExpectedFPU, unsigned ExpectedFlags,
                StringRef CPUAttr) {
  unsigned ArchKind = ARM::parseCPUArch(CPUName);
  bool pass = ARM::getArchName(ArchKind).equals(ExpectedArch);
  unsigned FPUKind = ARM::getDefaultFPU(CPUName, ArchKind);
  pass &= ARM::getFPUName(FPUKind).equals(ExpectedFPU);

  unsigned ExtKind = ARM::getDefaultExtensions(CPUName, ArchKind);
  if (ExtKind > 1 && (ExtKind & ARM::AEK_NONE))
    pass &= ((ExtKind ^ ARM::AEK_NONE) == ExpectedFlags);
  else
    pass &= (ExtKind == ExpectedFlags);

  pass &= ARM::getCPUAttr(ArchKind).equals(CPUAttr);

  return pass;
}

TEST(TargetParserTest, testARMCPU) {
  EXPECT_TRUE(testARMCPU("invalid", "invalid", "invalid",
                         ARM::AEK_NONE, ""));
  EXPECT_TRUE(testARMCPU("generic", "invalid", "none",
                         ARM::AEK_NONE, ""));

  EXPECT_TRUE(testARMCPU("arm2", "armv2", "none",
                         ARM::AEK_NONE, "2"));
  EXPECT_TRUE(testARMCPU("arm3", "armv2a", "none",
                         ARM::AEK_NONE, "2A"));
  EXPECT_TRUE(testARMCPU("arm6", "armv3", "none",
                         ARM::AEK_NONE, "3"));
  EXPECT_TRUE(testARMCPU("arm7m", "armv3m", "none",
                         ARM::AEK_NONE, "3M"));
  EXPECT_TRUE(testARMCPU("arm8", "armv4", "none",
                         ARM::AEK_NONE, "4"));
  EXPECT_TRUE(testARMCPU("arm810", "armv4", "none",
                         ARM::AEK_NONE, "4"));
  EXPECT_TRUE(testARMCPU("strongarm", "armv4", "none",
                         ARM::AEK_NONE, "4"));
  EXPECT_TRUE(testARMCPU("strongarm110", "armv4", "none",
                         ARM::AEK_NONE, "4"));
  EXPECT_TRUE(testARMCPU("strongarm1100", "armv4", "none",
                         ARM::AEK_NONE, "4"));
  EXPECT_TRUE(testARMCPU("strongarm1110", "armv4", "none",
                         ARM::AEK_NONE, "4"));
  EXPECT_TRUE(testARMCPU("arm7tdmi", "armv4t", "none",
                         ARM::AEK_NONE, "4T"));
  EXPECT_TRUE(testARMCPU("arm7tdmi-s", "armv4t", "none",
                         ARM::AEK_NONE, "4T"));
  EXPECT_TRUE(testARMCPU("arm710t", "armv4t", "none",
                         ARM::AEK_NONE, "4T"));
  EXPECT_TRUE(testARMCPU("arm720t", "armv4t", "none",
                         ARM::AEK_NONE, "4T"));
  EXPECT_TRUE(testARMCPU("arm9", "armv4t", "none",
                         ARM::AEK_NONE, "4T"));
  EXPECT_TRUE(testARMCPU("arm9tdmi", "armv4t", "none",
                         ARM::AEK_NONE, "4T"));
  EXPECT_TRUE(testARMCPU("arm920", "armv4t", "none",
                         ARM::AEK_NONE, "4T"));
  EXPECT_TRUE(testARMCPU("arm920t", "armv4t", "none",
                         ARM::AEK_NONE, "4T"));
  EXPECT_TRUE(testARMCPU("arm922t", "armv4t", "none",
                         ARM::AEK_NONE, "4T"));
  EXPECT_TRUE(testARMCPU("arm9312", "armv4t", "none",
                         ARM::AEK_NONE, "4T"));
  EXPECT_TRUE(testARMCPU("arm940t", "armv4t", "none",
                         ARM::AEK_NONE, "4T"));
  EXPECT_TRUE(testARMCPU("ep9312", "armv4t", "none",
                         ARM::AEK_NONE, "4T"));
  EXPECT_TRUE(testARMCPU("arm10tdmi", "armv5t", "none",
                         ARM::AEK_NONE, "5T"));
  EXPECT_TRUE(testARMCPU("arm1020t", "armv5t", "none",
                         ARM::AEK_NONE, "5T"));
  EXPECT_TRUE(testARMCPU("arm9e", "armv5te", "none",
                         ARM::AEK_DSP, "5TE"));
  EXPECT_TRUE(testARMCPU("arm946e-s", "armv5te", "none",
                         ARM::AEK_DSP, "5TE"));
  EXPECT_TRUE(testARMCPU("arm966e-s", "armv5te", "none",
                         ARM::AEK_DSP, "5TE"));
  EXPECT_TRUE(testARMCPU("arm968e-s", "armv5te", "none",
                         ARM::AEK_DSP, "5TE"));
  EXPECT_TRUE(testARMCPU("arm10e", "armv5te", "none",
                         ARM::AEK_DSP, "5TE"));
  EXPECT_TRUE(testARMCPU("arm1020e", "armv5te", "none",
                         ARM::AEK_DSP, "5TE"));
  EXPECT_TRUE(testARMCPU("arm1022e", "armv5te", "none",
                         ARM::AEK_DSP, "5TE"));
  EXPECT_TRUE(testARMCPU("arm926ej-s", "armv5tej", "none",
                         ARM::AEK_DSP, "5TEJ"));
  EXPECT_TRUE(testARMCPU("arm1136j-s", "armv6", "none",
                         ARM::AEK_DSP, "6"));
  EXPECT_TRUE(testARMCPU("arm1136jf-s", "armv6", "vfpv2",
                         ARM::AEK_DSP, "6"));
  EXPECT_TRUE(testARMCPU("arm1136jz-s", "armv6", "none",
                         ARM::AEK_DSP, "6"));
  EXPECT_TRUE(testARMCPU("arm1176j-s", "armv6k", "none",
                         ARM::AEK_DSP, "6K"));
  EXPECT_TRUE(testARMCPU("arm1176jz-s", "armv6kz", "none",
                         ARM::AEK_SEC | ARM::AEK_DSP, "6KZ"));
  EXPECT_TRUE(testARMCPU("mpcore", "armv6k", "vfpv2",
                         ARM::AEK_DSP, "6K"));
  EXPECT_TRUE(testARMCPU("mpcorenovfp", "armv6k", "none",
                         ARM::AEK_DSP, "6K"));
  EXPECT_TRUE(testARMCPU("arm1176jzf-s", "armv6kz", "vfpv2",
                         ARM::AEK_SEC | ARM::AEK_DSP, "6KZ"));
  EXPECT_TRUE(testARMCPU("arm1156t2-s", "armv6t2", "none",
                         ARM::AEK_DSP, "6T2"));
  EXPECT_TRUE(testARMCPU("arm1156t2f-s", "armv6t2", "vfpv2",
                         ARM::AEK_DSP, "6T2"));
  EXPECT_TRUE(testARMCPU("cortex-m0", "armv6-m", "none",
                         ARM::AEK_NONE, "6-M"));
  EXPECT_TRUE(testARMCPU("cortex-m0plus", "armv6-m", "none",
                         ARM::AEK_NONE, "6-M"));
  EXPECT_TRUE(testARMCPU("cortex-m1", "armv6-m", "none",
                         ARM::AEK_NONE, "6-M"));
  EXPECT_TRUE(testARMCPU("sc000", "armv6-m", "none",
                         ARM::AEK_NONE, "6-M"));
  EXPECT_TRUE(testARMCPU("cortex-a5", "armv7-a", "neon-vfpv4",
                         ARM::AEK_MP | ARM::AEK_SEC | ARM::AEK_DSP, "7-A"));
  EXPECT_TRUE(testARMCPU("cortex-a7", "armv7-a", "neon-vfpv4",
                         ARM::AEK_HWDIVTHUMB | ARM::AEK_HWDIVARM | ARM::AEK_MP |
                             ARM::AEK_SEC | ARM::AEK_VIRT | ARM::AEK_DSP,
                         "7-A"));
  EXPECT_TRUE(testARMCPU("cortex-a8", "armv7-a", "neon",
                         ARM::AEK_SEC | ARM::AEK_DSP, "7-A"));
  EXPECT_TRUE(testARMCPU("cortex-a9", "armv7-a", "neon-fp16",
                         ARM::AEK_MP | ARM::AEK_SEC | ARM::AEK_DSP, "7-A"));
  EXPECT_TRUE(testARMCPU("cortex-a12", "armv7-a", "neon-vfpv4",
                         ARM::AEK_SEC | ARM::AEK_MP | ARM::AEK_VIRT |
                             ARM::AEK_HWDIVARM | ARM::AEK_HWDIVTHUMB |
                             ARM::AEK_DSP,
                         "7-A"));
  EXPECT_TRUE(testARMCPU("cortex-a15", "armv7-a", "neon-vfpv4",
                         ARM::AEK_SEC | ARM::AEK_MP | ARM::AEK_VIRT |
                             ARM::AEK_HWDIVARM | ARM::AEK_HWDIVTHUMB |
                             ARM::AEK_DSP,
                         "7-A"));
  EXPECT_TRUE(testARMCPU("cortex-a17", "armv7-a", "neon-vfpv4",
                         ARM::AEK_SEC | ARM::AEK_MP | ARM::AEK_VIRT |
                             ARM::AEK_HWDIVARM | ARM::AEK_HWDIVTHUMB |
                             ARM::AEK_DSP,
                         "7-A"));
  EXPECT_TRUE(testARMCPU("krait", "armv7-a", "neon-vfpv4",
                         ARM::AEK_HWDIVARM | ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP,
                         "7-A"));
  EXPECT_TRUE(testARMCPU("cortex-r4", "armv7-r", "none",
                         ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP, "7-R"));
  EXPECT_TRUE(testARMCPU("cortex-r4f", "armv7-r", "vfpv3-d16",
                         ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP, "7-R"));
  EXPECT_TRUE(testARMCPU("cortex-r5", "armv7-r", "vfpv3-d16",
                         ARM::AEK_MP | ARM::AEK_HWDIVARM | ARM::AEK_HWDIVTHUMB |
                             ARM::AEK_DSP,
                         "7-R"));
  EXPECT_TRUE(testARMCPU("cortex-r7", "armv7-r", "vfpv3-d16-fp16",
                         ARM::AEK_MP | ARM::AEK_HWDIVARM | ARM::AEK_HWDIVTHUMB |
                             ARM::AEK_DSP,
                         "7-R"));
  EXPECT_TRUE(testARMCPU("cortex-r8", "armv7-r", "vfpv3-d16-fp16",
                         ARM::AEK_MP | ARM::AEK_HWDIVARM | ARM::AEK_HWDIVTHUMB |
                             ARM::AEK_DSP,
                         "7-R"));
  EXPECT_TRUE(testARMCPU("cortex-r52", "armv8-r", "neon-fp-armv8",
                         ARM::AEK_CRC | ARM::AEK_MP | ARM::AEK_VIRT |
                             ARM::AEK_HWDIVARM | ARM::AEK_HWDIVTHUMB |
                             ARM::AEK_DSP,
                         "8-R"));
  EXPECT_TRUE(
      testARMCPU("sc300", "armv7-m", "none", ARM::AEK_HWDIVTHUMB, "7-M"));
  EXPECT_TRUE(
      testARMCPU("cortex-m3", "armv7-m", "none", ARM::AEK_HWDIVTHUMB, "7-M"));
  EXPECT_TRUE(testARMCPU("cortex-m4", "armv7e-m", "fpv4-sp-d16",
                         ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP, "7E-M"));
  EXPECT_TRUE(testARMCPU("cortex-m7", "armv7e-m", "fpv5-d16",
                         ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP, "7E-M"));
  EXPECT_TRUE(testARMCPU("cortex-a32", "armv8-a", "crypto-neon-fp-armv8",
                         ARM::AEK_CRC | ARM::AEK_SEC | ARM::AEK_MP |
                             ARM::AEK_VIRT | ARM::AEK_HWDIVARM |
                             ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP,
                         "8-A"));
  EXPECT_TRUE(testARMCPU("cortex-a35", "armv8-a", "crypto-neon-fp-armv8",
                         ARM::AEK_CRC | ARM::AEK_SEC | ARM::AEK_MP |
                             ARM::AEK_VIRT | ARM::AEK_HWDIVARM |
                             ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP,
                         "8-A"));
  EXPECT_TRUE(testARMCPU("cortex-a53", "armv8-a", "crypto-neon-fp-armv8",
                         ARM::AEK_CRC | ARM::AEK_SEC | ARM::AEK_MP |
                             ARM::AEK_VIRT | ARM::AEK_HWDIVARM |
                             ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP,
                         "8-A"));
  EXPECT_TRUE(testARMCPU("cortex-a57", "armv8-a", "crypto-neon-fp-armv8",
                         ARM::AEK_CRC | ARM::AEK_SEC | ARM::AEK_MP |
                             ARM::AEK_VIRT | ARM::AEK_HWDIVARM |
                             ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP,
                         "8-A"));
  EXPECT_TRUE(testARMCPU("cortex-a72", "armv8-a", "crypto-neon-fp-armv8",
                         ARM::AEK_CRC | ARM::AEK_SEC | ARM::AEK_MP |
                             ARM::AEK_VIRT | ARM::AEK_HWDIVARM |
                             ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP,
                         "8-A"));
  EXPECT_TRUE(testARMCPU("cortex-a73", "armv8-a", "crypto-neon-fp-armv8",
                         ARM::AEK_CRC | ARM::AEK_SEC | ARM::AEK_MP |
                             ARM::AEK_VIRT | ARM::AEK_HWDIVARM |
                             ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP,
                         "8-A"));
  EXPECT_TRUE(testARMCPU("cyclone", "armv8-a", "crypto-neon-fp-armv8",
                         ARM::AEK_CRC | ARM::AEK_SEC | ARM::AEK_MP |
                             ARM::AEK_VIRT | ARM::AEK_HWDIVARM |
                             ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP,
                         "8-A"));
  EXPECT_TRUE(testARMCPU("exynos-m1", "armv8-a", "crypto-neon-fp-armv8",
                         ARM::AEK_CRC | ARM::AEK_SEC | ARM::AEK_MP |
                             ARM::AEK_VIRT | ARM::AEK_HWDIVARM |
                             ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP,
                         "8-A"));
  EXPECT_TRUE(testARMCPU("exynos-m2", "armv8-a", "crypto-neon-fp-armv8",
                         ARM::AEK_CRC | ARM::AEK_SEC | ARM::AEK_MP |
                             ARM::AEK_VIRT | ARM::AEK_HWDIVARM |
                             ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP,
                         "8-A"));
  EXPECT_TRUE(testARMCPU("exynos-m3", "armv8-a", "crypto-neon-fp-armv8",
                         ARM::AEK_CRC | ARM::AEK_SEC | ARM::AEK_MP |
                             ARM::AEK_VIRT | ARM::AEK_HWDIVARM |
                             ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP,
                         "8-A"));
  EXPECT_TRUE(testARMCPU("cortex-m23", "armv8-m.base", "none",
                         ARM::AEK_HWDIVTHUMB, "8-M.Baseline"));
  EXPECT_TRUE(testARMCPU("cortex-m33", "armv8-m.main", "fpv5-sp-d16",
                         ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP, "8-M.Mainline"));
  EXPECT_TRUE(testARMCPU("iwmmxt", "iwmmxt", "none",
                         ARM::AEK_NONE, "iwmmxt"));
  EXPECT_TRUE(testARMCPU("xscale", "xscale", "none",
                         ARM::AEK_NONE, "xscale"));
  EXPECT_TRUE(testARMCPU("swift", "armv7s", "neon-vfpv4",
                         ARM::AEK_HWDIVARM | ARM::AEK_HWDIVTHUMB | ARM::AEK_DSP,
                         "7-S"));
}

bool testARMArch(StringRef Arch, StringRef DefaultCPU, StringRef SubArch,
                 unsigned ArchAttr) {
  unsigned ArchKind = ARM::parseArch(Arch);
  return (ArchKind != ARM::AK_INVALID) &
         ARM::getDefaultCPU(Arch).equals(DefaultCPU) &
         ARM::getSubArch(ArchKind).equals(SubArch) &
         (ARM::getArchAttr(ArchKind) == ArchAttr);
}

TEST(TargetParserTest, testARMArch) {
  EXPECT_TRUE(
      testARMArch("armv2", "arm2", "v2",
                          ARMBuildAttrs::CPUArch::Pre_v4));
  EXPECT_TRUE(
      testARMArch("armv2a", "arm3", "v2a",
                          ARMBuildAttrs::CPUArch::Pre_v4));
  EXPECT_TRUE(
      testARMArch("armv3", "arm6", "v3",
                          ARMBuildAttrs::CPUArch::Pre_v4));
  EXPECT_TRUE(
      testARMArch("armv3m", "arm7m", "v3m",
                          ARMBuildAttrs::CPUArch::Pre_v4));
  EXPECT_TRUE(
      testARMArch("armv4", "strongarm", "v4",
                          ARMBuildAttrs::CPUArch::v4));
  EXPECT_TRUE(
      testARMArch("armv4t", "arm7tdmi", "v4t",
                          ARMBuildAttrs::CPUArch::v4T));
  EXPECT_TRUE(
      testARMArch("armv5t", "arm10tdmi", "v5",
                          ARMBuildAttrs::CPUArch::v5T));
  EXPECT_TRUE(
      testARMArch("armv5te", "arm1022e", "v5e",
                          ARMBuildAttrs::CPUArch::v5TE));
  EXPECT_TRUE(
      testARMArch("armv5tej", "arm926ej-s", "v5e",
                          ARMBuildAttrs::CPUArch::v5TEJ));
  EXPECT_TRUE(
      testARMArch("armv6", "arm1136jf-s", "v6",
                          ARMBuildAttrs::CPUArch::v6));
  EXPECT_TRUE(
      testARMArch("armv6k", "arm1176j-s", "v6k",
                          ARMBuildAttrs::CPUArch::v6K));
  EXPECT_TRUE(
      testARMArch("armv6t2", "arm1156t2-s", "v6t2",
                          ARMBuildAttrs::CPUArch::v6T2));
  EXPECT_TRUE(
      testARMArch("armv6kz", "arm1176jzf-s", "v6kz",
                          ARMBuildAttrs::CPUArch::v6KZ));
  EXPECT_TRUE(
      testARMArch("armv6-m", "cortex-m0", "v6m",
                          ARMBuildAttrs::CPUArch::v6_M));
  EXPECT_TRUE(
      testARMArch("armv7-a", "cortex-a8", "v7",
                          ARMBuildAttrs::CPUArch::v7));
  EXPECT_TRUE(
      testARMArch("armv7ve", "generic", "v7ve",
                          ARMBuildAttrs::CPUArch::v7));
  EXPECT_TRUE(
      testARMArch("armv7-r", "cortex-r4", "v7r",
                          ARMBuildAttrs::CPUArch::v7));
  EXPECT_TRUE(
      testARMArch("armv7-m", "cortex-m3", "v7m",
                          ARMBuildAttrs::CPUArch::v7));
  EXPECT_TRUE(
      testARMArch("armv7e-m", "cortex-m4", "v7em",
                          ARMBuildAttrs::CPUArch::v7E_M));
  EXPECT_TRUE(
      testARMArch("armv8-a", "cortex-a53", "v8",
                          ARMBuildAttrs::CPUArch::v8_A));
  EXPECT_TRUE(
      testARMArch("armv8.1-a", "generic", "v8.1a",
                          ARMBuildAttrs::CPUArch::v8_A));
  EXPECT_TRUE(
      testARMArch("armv8.2-a", "generic", "v8.2a",
                          ARMBuildAttrs::CPUArch::v8_A));
  EXPECT_TRUE(
      testARMArch("armv8-r", "cortex-r52", "v8r",
                          ARMBuildAttrs::CPUArch::v8_R));
  EXPECT_TRUE(
      testARMArch("armv8-m.base", "generic", "v8m.base",
                          ARMBuildAttrs::CPUArch::v8_M_Base));
  EXPECT_TRUE(
      testARMArch("armv8-m.main", "generic", "v8m.main",
                          ARMBuildAttrs::CPUArch::v8_M_Main));
  EXPECT_TRUE(
      testARMArch("iwmmxt", "iwmmxt", "",
                          ARMBuildAttrs::CPUArch::v5TE));
  EXPECT_TRUE(
      testARMArch("iwmmxt2", "generic", "",
                          ARMBuildAttrs::CPUArch::v5TE));
  EXPECT_TRUE(
      testARMArch("xscale", "xscale", "v5e",
                          ARMBuildAttrs::CPUArch::v5TE));
  EXPECT_TRUE(
      testARMArch("armv7s", "swift", "v7s",
                          ARMBuildAttrs::CPUArch::v7));
  EXPECT_TRUE(
      testARMArch("armv7k", "generic", "v7k",
                          ARMBuildAttrs::CPUArch::v7));
}

bool testARMExtension(StringRef CPUName, unsigned ArchKind, StringRef ArchExt) {
  return ARM::getDefaultExtensions(CPUName, ArchKind) &
         ARM::parseArchExt(ArchExt);
}

TEST(TargetParserTest, testARMExtension) {
  EXPECT_FALSE(testARMExtension("arm2", 0, "thumb"));
  EXPECT_FALSE(testARMExtension("arm3", 0, "thumb"));
  EXPECT_FALSE(testARMExtension("arm6", 0, "thumb"));
  EXPECT_FALSE(testARMExtension("arm7m", 0, "thumb"));
  EXPECT_FALSE(testARMExtension("strongarm", 0, "dsp"));
  EXPECT_FALSE(testARMExtension("arm7tdmi", 0, "dsp"));
  EXPECT_FALSE(testARMExtension("arm10tdmi", 0, "simd"));
  EXPECT_FALSE(testARMExtension("arm1022e", 0, "simd"));
  EXPECT_FALSE(testARMExtension("arm926ej-s", 0, "simd"));
  EXPECT_FALSE(testARMExtension("arm1136jf-s", 0, "crypto"));
  EXPECT_FALSE(testARMExtension("arm1176j-s", 0, "crypto"));
  EXPECT_FALSE(testARMExtension("arm1156t2-s", 0, "crypto"));
  EXPECT_FALSE(testARMExtension("arm1176jzf-s", 0, "crypto"));
  EXPECT_FALSE(testARMExtension("cortex-m0", 0, "crypto"));
  EXPECT_FALSE(testARMExtension("cortex-a8", 0, "crypto"));
  EXPECT_FALSE(testARMExtension("cortex-r4", 0, "crypto"));
  EXPECT_FALSE(testARMExtension("cortex-m3", 0, "crypto"));
  EXPECT_FALSE(testARMExtension("cortex-a53", 0, "ras"));
  EXPECT_FALSE(testARMExtension("cortex-r52", 0, "ras"));
  EXPECT_FALSE(testARMExtension("iwmmxt", 0, "crc"));
  EXPECT_FALSE(testARMExtension("xscale", 0, "crc"));
  EXPECT_FALSE(testARMExtension("swift", 0, "crc"));

  EXPECT_FALSE(testARMExtension("generic", ARM::AK_ARMV2, "thumb"));
  EXPECT_FALSE(testARMExtension("generic", ARM::AK_ARMV2A, "thumb"));
  EXPECT_FALSE(testARMExtension("generic", ARM::AK_ARMV3, "thumb"));
  EXPECT_FALSE(testARMExtension("generic", ARM::AK_ARMV3M, "thumb"));
  EXPECT_FALSE(testARMExtension("generic", ARM::AK_ARMV4, "dsp"));
  EXPECT_FALSE(testARMExtension("generic", ARM::AK_ARMV4T, "dsp"));
  EXPECT_FALSE(testARMExtension("generic", ARM::AK_ARMV5T, "simd"));
  EXPECT_FALSE(testARMExtension("generic", ARM::AK_ARMV5TE, "simd"));
  EXPECT_FALSE(testARMExtension("generic", ARM::AK_ARMV5TEJ, "simd"));
  EXPECT_FALSE(testARMExtension("generic", ARM::AK_ARMV6, "crypto"));
  EXPECT_FALSE(testARMExtension("generic", ARM::AK_ARMV6K, "crypto"));
  EXPECT_FALSE(testARMExtension("generic", ARM::AK_ARMV6T2, "crypto"));
  EXPECT_FALSE(testARMExtension("generic", ARM::AK_ARMV6KZ, "crypto"));
  EXPECT_FALSE(testARMExtension("generic", ARM::AK_ARMV6M, "crypto"));
  EXPECT_FALSE(testARMExtension("generic", ARM::AK_ARMV7A, "crypto"));
  EXPECT_FALSE(testARMExtension("generic", ARM::AK_ARMV7R, "crypto"));
  EXPECT_FALSE(testARMExtension("generic", ARM::AK_ARMV7M, "crypto"));
  EXPECT_FALSE(testARMExtension("generic", ARM::AK_ARMV7EM, "crypto"));
  EXPECT_FALSE(testARMExtension("generic", ARM::AK_ARMV8A, "ras"));
  EXPECT_FALSE(testARMExtension("generic", ARM::AK_ARMV8_1A, "ras"));
  EXPECT_FALSE(testARMExtension("generic", ARM::AK_ARMV8_2A, "spe"));
  EXPECT_FALSE(testARMExtension("generic", ARM::AK_ARMV8R, "ras"));
  EXPECT_FALSE(testARMExtension("generic", ARM::AK_ARMV8MBaseline, "crc"));
  EXPECT_FALSE(testARMExtension("generic", ARM::AK_ARMV8MMainline, "crc"));
  EXPECT_FALSE(testARMExtension("generic", ARM::AK_IWMMXT, "crc"));
  EXPECT_FALSE(testARMExtension("generic", ARM::AK_IWMMXT2, "crc"));
  EXPECT_FALSE(testARMExtension("generic", ARM::AK_XSCALE, "crc"));
  EXPECT_FALSE(testARMExtension("generic", ARM::AK_ARMV7S, "crypto"));
  EXPECT_FALSE(testARMExtension("generic", ARM::AK_ARMV7K, "crypto"));
}

TEST(TargetParserTest, ARMFPUVersion) {
  for (ARM::FPUKind FK = static_cast<ARM::FPUKind>(0);
       FK <= ARM::FPUKind::FK_LAST;
       FK = static_cast<ARM::FPUKind>(static_cast<unsigned>(FK) + 1))
    if (FK == ARM::FK_LAST)
      EXPECT_EQ(0U, ARM::getFPUVersion(FK));
    else
      EXPECT_LE(0U, ARM::getFPUVersion(FK));
}

TEST(TargetParserTest, ARMFPUNeonSupportLevel) {
  for (ARM::FPUKind FK = static_cast<ARM::FPUKind>(0);
       FK <= ARM::FPUKind::FK_LAST;
       FK = static_cast<ARM::FPUKind>(static_cast<unsigned>(FK) + 1))
    if (FK == ARM::FK_LAST)
      EXPECT_EQ(0U, ARM::getFPUNeonSupportLevel(FK));
    else
      EXPECT_LE(0U, ARM::getFPUNeonSupportLevel(FK));
}

TEST(TargetParserTest, ARMFPURestriction) {
  for (ARM::FPUKind FK = static_cast<ARM::FPUKind>(0);
       FK <= ARM::FPUKind::FK_LAST;
       FK = static_cast<ARM::FPUKind>(static_cast<unsigned>(FK) + 1))
    if (FK == ARM::FK_LAST)
      EXPECT_EQ(0U, ARM::getFPURestriction(FK));
    else
      EXPECT_LE(0U, ARM::getFPURestriction(FK));
}

TEST(TargetParserTest, ARMExtensionFeatures) {
  std::vector<StringRef> Features;
  unsigned Extensions = ARM::AEK_CRC | ARM::AEK_CRYPTO | ARM::AEK_DSP |
                        ARM::AEK_HWDIVARM | ARM::AEK_HWDIVTHUMB | ARM::AEK_MP |
                        ARM::AEK_SEC | ARM::AEK_VIRT | ARM::AEK_RAS;

  for (unsigned i = 0; i <= Extensions; i++)
    EXPECT_TRUE(i == 0 ? !ARM::getExtensionFeatures(i, Features)
                       : ARM::getExtensionFeatures(i, Features));
}

TEST(TargetParserTest, ARMFPUFeatures) {
  std::vector<StringRef> Features;
  for (ARM::FPUKind FK = static_cast<ARM::FPUKind>(0);
       FK <= ARM::FPUKind::FK_LAST;
       FK = static_cast<ARM::FPUKind>(static_cast<unsigned>(FK) + 1))
    EXPECT_TRUE((FK == ARM::FK_INVALID || FK >= ARM::FK_LAST)
                    ? !ARM::getFPUFeatures(FK, Features)
                    : ARM::getFPUFeatures(FK, Features));
}

TEST(TargetParserTest, ARMArchExtFeature) {
  const char *ArchExt[][4] = {{"crc", "nocrc", "+crc", "-crc"},
                              {"crypto", "nocrypto", "+crypto", "-crypto"},
                              {"dsp", "nodsp", "+dsp", "-dsp"},
                              {"fp", "nofp", nullptr, nullptr},
                              {"idiv", "noidiv", nullptr, nullptr},
                              {"mp", "nomp", nullptr, nullptr},
                              {"simd", "nosimd", nullptr, nullptr},
                              {"sec", "nosec", nullptr, nullptr},
                              {"virt", "novirt", nullptr, nullptr},
                              {"fp16", "nofp16", "+fullfp16", "-fullfp16"},
                              {"ras", "noras", "+ras", "-ras"},
                              {"os", "noos", nullptr, nullptr},
                              {"iwmmxt", "noiwmmxt", nullptr, nullptr},
                              {"iwmmxt2", "noiwmmxt2", nullptr, nullptr},
                              {"maverick", "maverick", nullptr, nullptr},
                              {"xscale", "noxscale", nullptr, nullptr}};

  for (unsigned i = 0; i < array_lengthof(ArchExt); i++) {
    EXPECT_EQ(StringRef(ArchExt[i][2]), ARM::getArchExtFeature(ArchExt[i][0]));
    EXPECT_EQ(StringRef(ArchExt[i][3]), ARM::getArchExtFeature(ArchExt[i][1]));
  }
}

TEST(TargetParserTest, ARMparseHWDiv) {
  const char *hwdiv[] = {"thumb", "arm", "arm,thumb", "thumb,arm"};

  for (unsigned i = 0; i < array_lengthof(hwdiv); i++)
    EXPECT_NE(ARM::AEK_INVALID, ARM::parseHWDiv((StringRef)hwdiv[i]));
}

TEST(TargetParserTest, ARMparseArchEndianAndISA) {
  const char *Arch[] = {
      "v2",     "v2a",   "v3",     "v3m",   "v4",   "v4t",  "v5",    "v5t",
      "v5e",    "v5te",  "v5tej",  "v6",    "v6j",  "v6k",  "v6hl",  "v6t2",
      "v6kz",   "v6z",   "v6zk",   "v6-m",  "v6m",  "v6sm", "v6s-m", "v7-a",
      "v7",     "v7a",   "v7ve",   "v7hl",  "v7l",  "v7-r", "v7r",   "v7-m",
      "v7m",    "v7k",   "v7s",    "v7e-m", "v7em", "v8-a", "v8",    "v8a",
      "v8.1-a", "v8.1a", "v8.2-a", "v8.2a", "v8-r"};

  for (unsigned i = 0; i < array_lengthof(Arch); i++) {
    std::string arm_1 = "armeb" + (std::string)(Arch[i]);
    std::string arm_2 = "arm" + (std::string)(Arch[i]) + "eb";
    std::string arm_3 = "arm" + (std::string)(Arch[i]);
    std::string thumb_1 = "thumbeb" + (std::string)(Arch[i]);
    std::string thumb_2 = "thumb" + (std::string)(Arch[i]) + "eb";
    std::string thumb_3 = "thumb" + (std::string)(Arch[i]);

    EXPECT_EQ(ARM::EK_BIG, ARM::parseArchEndian(arm_1));
    EXPECT_EQ(ARM::EK_BIG, ARM::parseArchEndian(arm_2));
    EXPECT_EQ(ARM::EK_LITTLE, ARM::parseArchEndian(arm_3));

    EXPECT_EQ(ARM::IK_ARM, ARM::parseArchISA(arm_1));
    EXPECT_EQ(ARM::IK_ARM, ARM::parseArchISA(arm_2));
    EXPECT_EQ(ARM::IK_ARM, ARM::parseArchISA(arm_3));
    if (i >= 4) {
      EXPECT_EQ(ARM::EK_BIG, ARM::parseArchEndian(thumb_1));
      EXPECT_EQ(ARM::EK_BIG, ARM::parseArchEndian(thumb_2));
      EXPECT_EQ(ARM::EK_LITTLE, ARM::parseArchEndian(thumb_3));

      EXPECT_EQ(ARM::IK_THUMB, ARM::parseArchISA(thumb_1));
      EXPECT_EQ(ARM::IK_THUMB, ARM::parseArchISA(thumb_2));
      EXPECT_EQ(ARM::IK_THUMB, ARM::parseArchISA(thumb_3));
    }
  }

  EXPECT_EQ(ARM::EK_LITTLE, ARM::parseArchEndian("aarch64"));
  EXPECT_EQ(ARM::EK_BIG, ARM::parseArchEndian("aarch64_be"));

  EXPECT_EQ(ARM::IK_AARCH64, ARM::parseArchISA("aarch64"));
  EXPECT_EQ(ARM::IK_AARCH64, ARM::parseArchISA("aarch64_be"));
  EXPECT_EQ(ARM::IK_AARCH64, ARM::parseArchISA("arm64"));
  EXPECT_EQ(ARM::IK_AARCH64, ARM::parseArchISA("arm64_be"));
}

TEST(TargetParserTest, ARMparseArchProfile) {
  for (unsigned i = 0; i < array_lengthof(ARMArch); i++) {
    switch (ARM::parseArch(ARMArch[i])) {
    case ARM::AK_ARMV6M:
    case ARM::AK_ARMV7M:
    case ARM::AK_ARMV7EM:
    case ARM::AK_ARMV8MMainline:
    case ARM::AK_ARMV8MBaseline:
      EXPECT_EQ(ARM::PK_M, ARM::parseArchProfile(ARMArch[i]));
      continue;
    case ARM::AK_ARMV7R:
    case ARM::AK_ARMV8R:
      EXPECT_EQ(ARM::PK_R, ARM::parseArchProfile(ARMArch[i]));
      continue;
    case ARM::AK_ARMV7A:
    case ARM::AK_ARMV7VE:
    case ARM::AK_ARMV7K:
    case ARM::AK_ARMV8A:
    case ARM::AK_ARMV8_1A:
    case ARM::AK_ARMV8_2A:
      EXPECT_EQ(ARM::PK_A, ARM::parseArchProfile(ARMArch[i]));
      continue;
    }
    EXPECT_EQ(ARM::PK_INVALID, ARM::parseArchProfile(ARMArch[i]));
  }
}

TEST(TargetParserTest, ARMparseArchVersion) {
  for (unsigned i = 0; i < array_lengthof(ARMArch); i++)
    if (((std::string)ARMArch[i]).substr(0, 4) == "armv")
      EXPECT_EQ((ARMArch[i][4] - 48u), ARM::parseArchVersion(ARMArch[i]));
    else
      EXPECT_EQ(5u, ARM::parseArchVersion(ARMArch[i]));
}

bool testAArch64CPU(StringRef CPUName, StringRef ExpectedArch,
                    StringRef ExpectedFPU, unsigned ExpectedFlags,
                    StringRef CPUAttr) {
  unsigned ArchKind = AArch64::parseCPUArch(CPUName);
  bool pass = AArch64::getArchName(ArchKind).equals(ExpectedArch);
  unsigned FPUKind = AArch64::getDefaultFPU(CPUName, ArchKind);
  pass &= AArch64::getFPUName(FPUKind).equals(ExpectedFPU);

  unsigned ExtKind = AArch64::getDefaultExtensions(CPUName, ArchKind);
  if (ExtKind > 1 && (ExtKind & AArch64::AEK_NONE))
    pass &= ((ExtKind ^ AArch64::AEK_NONE) == ExpectedFlags);
  else
    pass &= (ExtKind == ExpectedFlags);

  pass &= AArch64::getCPUAttr(ArchKind).equals(CPUAttr);

  return pass;
}

TEST(TargetParserTest, testAArch64CPU) {
  EXPECT_TRUE(testAArch64CPU(
      "invalid", "invalid", "invalid",
      AArch64::AEK_INVALID, ""));
  EXPECT_TRUE(testAArch64CPU(
      "generic", "invalid", "none",
      AArch64::AEK_NONE, ""));

  EXPECT_TRUE(testAArch64CPU(
      "cortex-a35", "armv8-a", "crypto-neon-fp-armv8",
      AArch64::AEK_CRC | AArch64::AEK_CRYPTO | AArch64::AEK_SIMD, "8-A"));
  EXPECT_TRUE(testAArch64CPU(
      "cortex-a53", "armv8-a", "crypto-neon-fp-armv8",
      AArch64::AEK_CRC | AArch64::AEK_CRYPTO | AArch64::AEK_SIMD, "8-A"));
  EXPECT_TRUE(testAArch64CPU(
      "cortex-a57", "armv8-a", "crypto-neon-fp-armv8",
      AArch64::AEK_CRC | AArch64::AEK_CRYPTO | AArch64::AEK_SIMD, "8-A"));
  EXPECT_TRUE(testAArch64CPU(
      "cortex-a72", "armv8-a", "crypto-neon-fp-armv8",
      AArch64::AEK_CRC | AArch64::AEK_CRYPTO | AArch64::AEK_SIMD, "8-A"));
  EXPECT_TRUE(testAArch64CPU(
      "cortex-a73", "armv8-a", "crypto-neon-fp-armv8",
      AArch64::AEK_CRC | AArch64::AEK_CRYPTO | AArch64::AEK_SIMD, "8-A"));
  EXPECT_TRUE(testAArch64CPU(
      "cyclone", "armv8-a", "crypto-neon-fp-armv8",
      AArch64::AEK_CRYPTO | AArch64::AEK_SIMD, "8-A"));
  EXPECT_TRUE(testAArch64CPU(
      "exynos-m1", "armv8-a", "crypto-neon-fp-armv8",
      AArch64::AEK_CRC | AArch64::AEK_CRYPTO | AArch64::AEK_SIMD, "8-A"));
  EXPECT_TRUE(testAArch64CPU(
      "exynos-m2", "armv8-a", "crypto-neon-fp-armv8",
      AArch64::AEK_CRC | AArch64::AEK_CRYPTO | AArch64::AEK_SIMD, "8-A"));
  EXPECT_TRUE(testAArch64CPU(
      "exynos-m3", "armv8-a", "crypto-neon-fp-armv8",
      AArch64::AEK_CRC | AArch64::AEK_CRYPTO | AArch64::AEK_SIMD, "8-A"));
  EXPECT_TRUE(testAArch64CPU(
      "falkor", "armv8-a", "crypto-neon-fp-armv8",
      AArch64::AEK_CRC | AArch64::AEK_CRYPTO | AArch64::AEK_SIMD, "8-A"));
  EXPECT_TRUE(testAArch64CPU(
      "kryo", "armv8-a", "crypto-neon-fp-armv8",
      AArch64::AEK_CRC | AArch64::AEK_CRYPTO | AArch64::AEK_SIMD, "8-A"));
  EXPECT_TRUE(testAArch64CPU(
      "thunderx2t99", "armv8.1-a", "crypto-neon-fp-armv8",
      AArch64::AEK_CRC | AArch64::AEK_CRYPTO | AArch64::AEK_LSE |
      AArch64::AEK_SIMD, "8.1-A"));
  EXPECT_TRUE(testAArch64CPU(
      "thunderx", "armv8-a", "crypto-neon-fp-armv8",
      AArch64::AEK_CRC | AArch64::AEK_CRYPTO | AArch64::AEK_SIMD |
      AArch64::AEK_FP | AArch64::AEK_PROFILE,
      "8-A"));
  EXPECT_TRUE(testAArch64CPU(
      "thunderxt81", "armv8-a", "crypto-neon-fp-armv8",
      AArch64::AEK_CRC | AArch64::AEK_CRYPTO | AArch64::AEK_SIMD |
      AArch64::AEK_FP | AArch64::AEK_PROFILE,
      "8-A"));
  EXPECT_TRUE(testAArch64CPU(
      "thunderxt83", "armv8-a", "crypto-neon-fp-armv8",
      AArch64::AEK_CRC | AArch64::AEK_CRYPTO | AArch64::AEK_SIMD |
      AArch64::AEK_FP | AArch64::AEK_PROFILE,
      "8-A"));
  EXPECT_TRUE(testAArch64CPU(
      "thunderxt88", "armv8-a", "crypto-neon-fp-armv8",
      AArch64::AEK_CRC | AArch64::AEK_CRYPTO | AArch64::AEK_SIMD |
      AArch64::AEK_FP | AArch64::AEK_PROFILE,
      "8-A"));
}

bool testAArch64Arch(StringRef Arch, StringRef DefaultCPU, StringRef SubArch,
                     unsigned ArchAttr) {
  unsigned ArchKind = AArch64::parseArch(Arch);
  return (ArchKind != static_cast<unsigned>(AArch64::ArchKind::AK_INVALID)) &
         AArch64::getDefaultCPU(Arch).equals(DefaultCPU) &
         AArch64::getSubArch(ArchKind).equals(SubArch) &
         (AArch64::getArchAttr(ArchKind) == ArchAttr);
}

TEST(TargetParserTest, testAArch64Arch) {
  EXPECT_TRUE(testAArch64Arch("armv8-a", "cortex-a53", "v8",
                              ARMBuildAttrs::CPUArch::v8_A));
  EXPECT_TRUE(testAArch64Arch("armv8.1-a", "generic", "v8.1a",
                              ARMBuildAttrs::CPUArch::v8_A));
  EXPECT_TRUE(testAArch64Arch("armv8.2-a", "generic", "v8.2a",
                              ARMBuildAttrs::CPUArch::v8_A));
}

bool testAArch64Extension(StringRef CPUName, unsigned ArchKind,
                          StringRef ArchExt) {
  return AArch64::getDefaultExtensions(CPUName, ArchKind) &
         AArch64::parseArchExt(ArchExt);
}

TEST(TargetParserTest, testAArch64Extension) {
  EXPECT_FALSE(testAArch64Extension("cortex-a35", 0, "ras"));
  EXPECT_FALSE(testAArch64Extension("cortex-a53", 0, "ras"));
  EXPECT_FALSE(testAArch64Extension("cortex-a57", 0, "ras"));
  EXPECT_FALSE(testAArch64Extension("cortex-a72", 0, "ras"));
  EXPECT_FALSE(testAArch64Extension("cortex-a73", 0, "ras"));
  EXPECT_FALSE(testAArch64Extension("cyclone", 0, "ras"));
  EXPECT_FALSE(testAArch64Extension("exynos-m1", 0, "ras"));
  EXPECT_FALSE(testAArch64Extension("kryo", 0, "ras"));
  EXPECT_FALSE(testAArch64Extension("thunderx2t99", 0, "ras"));
  EXPECT_FALSE(testAArch64Extension("thunderx", 0, "lse"));
  EXPECT_FALSE(testAArch64Extension("thunderxt81", 0, "lse"));
  EXPECT_FALSE(testAArch64Extension("thunderxt83", 0, "lse"));
  EXPECT_FALSE(testAArch64Extension("thunderxt88", 0, "lse"));

  EXPECT_FALSE(testAArch64Extension(
      "generic", static_cast<unsigned>(AArch64::ArchKind::AK_ARMV8A), "ras"));
  EXPECT_FALSE(testAArch64Extension(
      "generic", static_cast<unsigned>(AArch64::ArchKind::AK_ARMV8_1A), "ras"));
  EXPECT_FALSE(testAArch64Extension(
      "generic", static_cast<unsigned>(AArch64::ArchKind::AK_ARMV8_2A), "spe"));
}

TEST(TargetParserTest, AArch64ExtensionFeatures) {
  std::vector<StringRef> Features;
  unsigned Extensions = AArch64::AEK_CRC | AArch64::AEK_CRYPTO |
                        AArch64::AEK_FP | AArch64::AEK_SIMD |
                        AArch64::AEK_FP16 | AArch64::AEK_PROFILE |
                        AArch64::AEK_RAS;

  for (unsigned i = 0; i <= Extensions; i++)
    EXPECT_TRUE(i == 0 ? !AArch64::getExtensionFeatures(i, Features)
                       : AArch64::getExtensionFeatures(i, Features));
}

TEST(TargetParserTest, AArch64ArchFeatures) {
  std::vector<StringRef> Features;

  for (unsigned AK = 0; AK < static_cast<unsigned>(AArch64::ArchKind::AK_LAST);
       AK++)
    EXPECT_TRUE((AK == static_cast<unsigned>(AArch64::ArchKind::AK_INVALID) ||
                 AK == static_cast<unsigned>(AArch64::ArchKind::AK_LAST))
                    ? !AArch64::getArchFeatures(AK, Features)
                    : AArch64::getArchFeatures(AK, Features));
}

TEST(TargetParserTest, AArch64ArchExtFeature) {
  const char *ArchExt[][4] = {{"crc", "nocrc", "+crc", "-crc"},
                              {"crypto", "nocrypto", "+crypto", "-crypto"},
                              {"fp", "nofp", "+fp-armv8", "-fp-armv8"},
                              {"simd", "nosimd", "+neon", "-neon"},
                              {"fp16", "nofp16", "+fullfp16", "-fullfp16"},
                              {"profile", "noprofile", "+spe", "-spe"},
                              {"ras", "noras", "+ras", "-ras"}};

  for (unsigned i = 0; i < array_lengthof(ArchExt); i++) {
    EXPECT_EQ(StringRef(ArchExt[i][2]),
              AArch64::getArchExtFeature(ArchExt[i][0]));
    EXPECT_EQ(StringRef(ArchExt[i][3]),
              AArch64::getArchExtFeature(ArchExt[i][1]));
  }
}
}
