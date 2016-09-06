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
static const unsigned kAArch64ArchExtKinds[] = {
#define AARCH64_ARCH_EXT_NAME(NAME, ID, FEATURE, NEGFEATURE) ID,
#include "llvm/Support/AArch64TargetParser.def"
#undef AARCH64_ARCH_EXT_NAME
};

template <typename T> struct ArchNames {
  const char *Name;
  unsigned DefaultFPU;
  unsigned ArchBaseExtensions;
  T ID;
  ARMBuildAttrs::CPUArch ArchAttr;
};
ArchNames<AArch64::ArchKind> kAArch64ARCHNames[] = {
#define AARCH64_ARCH(NAME, ID, CPU_ATTR, SUB_ARCH, ARCH_ATTR, ARCH_FPU,        \
                     ARCH_BASE_EXT)                                            \
  {NAME, ARM::ARCH_FPU, ARCH_BASE_EXT, AArch64::ArchKind::ID, ARCH_ATTR},
#include "llvm/Support/AArch64TargetParser.def"
#undef AARCH64_ARCH
};
ArchNames<ARM::ArchKind> kARMARCHNames[] = {
#define ARM_ARCH(NAME, ID, CPU_ATTR, SUB_ARCH, ARCH_ATTR, ARCH_FPU,            \
                 ARCH_BASE_EXT)                                                \
  {NAME, ARM::ARCH_FPU, ARCH_BASE_EXT, ARM::ID, ARCH_ATTR},
#include "llvm/Support/ARMTargetParser.def"
#undef ARM_ARCH
};

template <typename T> struct CpuNames {
  const char *Name;
  T ID;
  unsigned DefaultFPU;
  unsigned DefaultExt;
};
CpuNames<AArch64::ArchKind> kAArch64CPUNames[] = {
#define AARCH64_CPU_NAME(NAME, ID, DEFAULT_FPU, IS_DEFAULT, DEFAULT_EXT)       \
  {NAME, AArch64::ArchKind::ID, ARM::DEFAULT_FPU, DEFAULT_EXT},
#include "llvm/Support/AArch64TargetParser.def"
#undef AARCH64_CPU_NAME
};
CpuNames<ARM::ArchKind> kARMCPUNames[] = {
#define ARM_CPU_NAME(NAME, ID, DEFAULT_FPU, IS_DEFAULT, DEFAULT_EXT)           \
  {NAME, ARM::ID, ARM::DEFAULT_FPU, DEFAULT_EXT},
#include "llvm/Support/ARMTargetParser.def"
#undef ARM_CPU_NAME
};

const char *ARMArch[] = {
    "armv2",        "armv2a",      "armv3",    "armv3m",       "armv4",
    "armv4t",       "armv5",       "armv5t",   "armv5e",       "armv5te",
    "armv5tej",     "armv6",       "armv6j",   "armv6k",       "armv6hl",
    "armv6t2",      "armv6kz",     "armv6z",   "armv6zk",      "armv6-m",
    "armv6m",       "armv6sm",     "armv6s-m", "armv7-a",      "armv7",
    "armv7a",       "armv7hl",     "armv7l",   "armv7-r",      "armv7r",
    "armv7-m",      "armv7m",      "armv7k",   "armv7s",       "armv7e-m",
    "armv7em",      "armv8-a",     "armv8",    "armv8a",       "armv8.1-a",
    "armv8.1a",     "armv8.2-a",   "armv8.2a", "armv8-m.base", "armv8m.base",
    "armv8-m.main", "armv8m.main", "iwmmxt",   "iwmmxt2",      "xscale"};

template <typename T, size_t N>
bool contains(const T (&array)[N], const T element) {
  return std::find(std::begin(array), std::end(array), element) !=
         std::end(array);
}

template <size_t N>
bool contains(const char *(&array)[N], const char *element) {
  return std::find_if(std::begin(array), std::end(array), [&](const char *S) {
           return ::strcmp(S, element) == 0;
         }) != std::end(array);
}

TEST(TargetParserTest, ARMArchName) {
  for (ARM::ArchKind AK = static_cast<ARM::ArchKind>(0);
       AK <= ARM::ArchKind::AK_LAST;
       AK = static_cast<ARM::ArchKind>(static_cast<unsigned>(AK) + 1))
    EXPECT_TRUE(AK == ARM::AK_LAST ? ARM::getArchName(AK).empty()
                                   : !ARM::getArchName(AK).empty());
}

TEST(TargetParserTest, ARMCPUAttr) {
  for (ARM::ArchKind AK = static_cast<ARM::ArchKind>(0);
       AK <= ARM::ArchKind::AK_LAST;
       AK = static_cast<ARM::ArchKind>(static_cast<unsigned>(AK) + 1))
    EXPECT_TRUE((AK == ARM::AK_INVALID || AK == ARM::AK_LAST)
                    ? ARM::getCPUAttr(AK).empty()
                    : !ARM::getCPUAttr(AK).empty());
}

TEST(TargetParserTest, ARMSubArch) {
  for (ARM::ArchKind AK = static_cast<ARM::ArchKind>(0);
       AK <= ARM::ArchKind::AK_LAST;
       AK = static_cast<ARM::ArchKind>(static_cast<unsigned>(AK) + 1))
    EXPECT_TRUE((AK == ARM::AK_INVALID || AK == ARM::AK_IWMMXT ||
                 AK == ARM::AK_IWMMXT2 || AK == ARM::AK_LAST)
                    ? ARM::getSubArch(AK).empty()
                    : !ARM::getSubArch(AK).empty());
}

TEST(TargetParserTest, ARMFPUName) {
  for (ARM::FPUKind FK = static_cast<ARM::FPUKind>(0);
       FK <= ARM::FPUKind::FK_LAST;
       FK = static_cast<ARM::FPUKind>(static_cast<unsigned>(FK) + 1))
    EXPECT_TRUE(FK == ARM::FK_LAST ? ARM::getFPUName(FK).empty()
                                   : !ARM::getFPUName(FK).empty());
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

TEST(TargetParserTest, ARMDefaultFPU) {
  for (ARM::ArchKind AK = static_cast<ARM::ArchKind>(0);
       AK < ARM::ArchKind::AK_LAST;
       AK = static_cast<ARM::ArchKind>(static_cast<unsigned>(AK) + 1))
    EXPECT_EQ(kARMARCHNames[AK].DefaultFPU,
              ARM::getDefaultFPU(StringRef("generic"), AK));

  for (const auto &ARMCPUName : kARMCPUNames)
    EXPECT_EQ(ARMCPUName.DefaultFPU, ARM::getDefaultFPU(ARMCPUName.Name, 0));
}

TEST(TargetParserTest, ARMDefaultExtensions) {
  for (ARM::ArchKind AK = static_cast<ARM::ArchKind>(0);
       AK < ARM::ArchKind::AK_LAST;
       AK = static_cast<ARM::ArchKind>(static_cast<unsigned>(AK) + 1))
    EXPECT_EQ(kARMARCHNames[AK].ArchBaseExtensions,
              ARM::getDefaultExtensions(StringRef("generic"), AK));

  for (const auto &ARMCPUName : kARMCPUNames) {
    unsigned DefaultExt =
        kARMARCHNames[ARMCPUName.ID].ArchBaseExtensions | ARMCPUName.DefaultExt;
    EXPECT_EQ(DefaultExt, ARM::getDefaultExtensions(ARMCPUName.Name, 0));
  }
}

TEST(TargetParserTest, ARMExtensionFeatures) {
  std::vector<const char *> Features;
  unsigned Extensions = ARM::AEK_CRC | ARM::AEK_CRYPTO | ARM::AEK_DSP |
                        ARM::AEK_HWDIVARM | ARM::AEK_HWDIV | ARM::AEK_MP |
                        ARM::AEK_SEC | ARM::AEK_VIRT | ARM::AEK_RAS;

  for (unsigned i = 0; i <= Extensions; i++)
    EXPECT_TRUE(i == 0 ? !ARM::getExtensionFeatures(i, Features)
                       : ARM::getExtensionFeatures(i, Features));
}

TEST(TargetParserTest, ARMFPUFeatures) {
  std::vector<const char *> Features;
  for (ARM::FPUKind FK = static_cast<ARM::FPUKind>(0);
       FK <= ARM::FPUKind::FK_LAST;
       FK = static_cast<ARM::FPUKind>(static_cast<unsigned>(FK) + 1))
    EXPECT_TRUE((FK == ARM::FK_INVALID || FK >= ARM::FK_LAST)
                    ? !ARM::getFPUFeatures(FK, Features)
                    : ARM::getFPUFeatures(FK, Features));
}

TEST(TargetParserTest, ARMArchAttr) {
  for (ARM::ArchKind AK = static_cast<ARM::ArchKind>(0);
       AK <= ARM::ArchKind::AK_LAST;
       AK = static_cast<ARM::ArchKind>(static_cast<unsigned>(AK) + 1))
    EXPECT_TRUE(AK == ARM::AK_LAST
                    ? (ARMBuildAttrs::CPUArch::Pre_v4 == ARM::getArchAttr(AK))
                    : (kARMARCHNames[AK].ArchAttr == ARM::getArchAttr(AK)));
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
    EXPECT_STREQ(ArchExt[i][2], ARM::getArchExtFeature(ArchExt[i][0]));
    EXPECT_STREQ(ArchExt[i][3], ARM::getArchExtFeature(ArchExt[i][1]));
  }
}

TEST(TargetParserTest, ARMDefaultCPU) {
  for (unsigned i = 0; i < array_lengthof(ARMArch); i++)
    EXPECT_FALSE(ARM::getDefaultCPU(ARMArch[i]).empty());
}

TEST(TargetParserTest, ARMparseHWDiv) {
  const char *hwdiv[] = {"thumb", "arm", "arm,thumb", "thumb,arm"};

  for (unsigned i = 0; i < array_lengthof(hwdiv); i++)
    EXPECT_NE(ARM::AEK_INVALID, ARM::parseHWDiv((StringRef)hwdiv[i]));
}

TEST(TargetParserTest, ARMparseFPU) {
  const char *FPU[] = {"vfp",
                       "vfpv2",
                       "vfp2",
                       "vfpv3",
                       "vfp3",
                       "vfpv3-fp16",
                       "vfpv3-d16",
                       "vfp3-d16",
                       "vfpv3-d16-fp16",
                       "vfpv3xd",
                       "vfpv3xd-fp16",
                       "vfpv4",
                       "vfp4",
                       "vfpv4-d16",
                       "vfp4-d16",
                       "fp4-dp-d16",
                       "fpv4-dp-d16",
                       "fpv4-sp-d16",
                       "fp4-sp-d16",
                       "vfpv4-sp-d16",
                       "fpv5-d16",
                       "fp5-dp-d16",
                       "fpv5-dp-d16",
                       "fpv5-sp-d16",
                       "fp5-sp-d16",
                       "fp-armv8",
                       "neon",
                       "neon-vfpv3",
                       "neon-fp16",
                       "neon-vfpv4",
                       "neon-fp-armv8",
                       "crypto-neon-fp-armv8",
                       "softvfp"};

  for (unsigned i = 0; i < array_lengthof(FPU); i++)
    EXPECT_NE(ARM::FK_INVALID, ARM::parseFPU((StringRef)FPU[i]));
}

TEST(TargetParserTest, ARMparseArch) {
  for (unsigned i = 0; i < array_lengthof(ARMArch); i++)
    EXPECT_NE(ARM::AEK_INVALID, ARM::parseArch(ARMArch[i]));
}

TEST(TargetParserTest, ARMparseArchExt) {
  const char *ArchExt[] = {"none",     "crc",   "crypto", "dsp",    "fp",
                           "idiv",     "mp",    "simd",   "sec",    "virt",
                           "fp16",     "ras",   "os",     "iwmmxt", "iwmmxt2",
                           "maverick", "xscale"};

  for (unsigned i = 0; i < array_lengthof(ArchExt); i++)
    EXPECT_NE(ARM::AEK_INVALID, ARM::parseArchExt(ArchExt[i]));
}

TEST(TargetParserTest, ARMparseCPUArch) {
  const char *CPU[] = {
      "arm2",          "arm3",          "arm6",        "arm7m",
      "arm8",          "arm810",        "strongarm",   "strongarm110",
      "strongarm1100", "strongarm1110", "arm7tdmi",    "arm7tdmi-s",
      "arm710t",       "arm720t",       "arm9",        "arm9tdmi",
      "arm920",        "arm920t",       "arm922t",     "arm9312",
      "arm940t",       "ep9312",        "arm10tdmi",   "arm1020t",
      "arm9e",         "arm946e-s",     "arm966e-s",   "arm968e-s",
      "arm10e",        "arm1020e",      "arm1022e",    "arm926ej-s",
      "arm1136j-s",    "arm1136jf-s",   "arm1136jz-s", "arm1176j-s",
      "arm1176jz-s",   "mpcore",        "mpcorenovfp", "arm1176jzf-s",
      "arm1156t2-s",   "arm1156t2f-s",  "cortex-m0",   "cortex-m0plus",
      "cortex-m1",     "sc000",         "cortex-a5",   "cortex-a7",
      "cortex-a8",     "cortex-a9",     "cortex-a12",  "cortex-a15",
      "cortex-a17",    "krait",         "cortex-r4",   "cortex-r4f",
      "cortex-r5",     "cortex-r7",     "cortex-r8",   "sc300",
      "cortex-m3",     "cortex-m4",     "cortex-m7",   "cortex-a32",
      "cortex-a35",    "cortex-a53",    "cortex-a57",  "cortex-a72",
      "cortex-a73",    "cyclone",       "exynos-m1",   "exynos-m2",   
      "iwmmxt",        "xscale",        "swift"};

  for (const auto &ARMCPUName : kARMCPUNames) {
    if (contains(CPU, ARMCPUName.Name))
      EXPECT_NE(ARM::AK_INVALID, ARM::parseCPUArch(ARMCPUName.Name));
    else
      EXPECT_EQ(ARM::AK_INVALID, ARM::parseCPUArch(ARMCPUName.Name));
  }
}

TEST(TargetParserTest, ARMparseArchEndianAndISA) {
  const char *Arch[] = {
      "v2",    "v2a",    "v3",    "v3m",  "v4",   "v4t",  "v5",    "v5t",
      "v5e",   "v5te",   "v5tej", "v6",   "v6j",  "v6k",  "v6hl",  "v6t2",
      "v6kz",  "v6z",    "v6zk",  "v6-m", "v6m",  "v6sm", "v6s-m", "v7-a",
      "v7",    "v7a",    "v7hl",  "v7l",  "v7-r", "v7r",  "v7-m",  "v7m",
      "v7k",   "v7s",    "v7e-m", "v7em", "v8-a", "v8",   "v8a",   "v8.1-a",
      "v8.1a", "v8.2-a", "v8.2a"};

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
      EXPECT_EQ(ARM::PK_R, ARM::parseArchProfile(ARMArch[i]));
      continue;
    case ARM::AK_ARMV7A:
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

TEST(TargetParserTest, AArch64DefaultFPU) {
  for (unsigned AK = 0; AK < static_cast<unsigned>(AArch64::ArchKind::AK_LAST);
       AK++)
    EXPECT_EQ(kAArch64ARCHNames[AK].DefaultFPU,
              AArch64::getDefaultFPU(StringRef("generic"), AK));

  for (const auto &AArch64CPUName : kAArch64CPUNames)
    EXPECT_EQ(AArch64CPUName.DefaultFPU,
              AArch64::getDefaultFPU(AArch64CPUName.Name,
                                     static_cast<unsigned>(AArch64CPUName.ID)));
}

TEST(TargetParserTest, AArch64DefaultExt) {
  for (unsigned AK = 0; AK < static_cast<unsigned>(AArch64::ArchKind::AK_LAST);
       AK++)
    EXPECT_EQ(kAArch64ARCHNames[AK].ArchBaseExtensions,
              AArch64::getDefaultExtensions(StringRef("generic"), AK));

  for (const auto &AArch64CPUName : kAArch64CPUNames)
    EXPECT_EQ(
        AArch64CPUName.DefaultExt,
        AArch64::getDefaultExtensions(
            AArch64CPUName.Name, static_cast<unsigned>(AArch64CPUName.ID)));
}

TEST(TargetParserTest, AArch64ExtensionFeatures) {
  std::vector<const char *> Features;
  unsigned Extensions = AArch64::AEK_CRC | AArch64::AEK_CRYPTO |
                        AArch64::AEK_FP | AArch64::AEK_SIMD |
                        AArch64::AEK_FP16 | AArch64::AEK_PROFILE |
                        AArch64::AEK_RAS;

  for (unsigned i = 0; i <= Extensions; i++)
    EXPECT_TRUE(i == 0 ? !AArch64::getExtensionFeatures(i, Features)
                       : AArch64::getExtensionFeatures(i, Features));
}

TEST(TargetParserTest, AArch64ArchFeatures) {
  std::vector<const char *> Features;

  for (unsigned AK = 0; AK < static_cast<unsigned>(AArch64::ArchKind::AK_LAST);
       AK++)
    EXPECT_TRUE((AK == static_cast<unsigned>(AArch64::ArchKind::AK_INVALID) ||
                 AK == static_cast<unsigned>(AArch64::ArchKind::AK_LAST))
                    ? !AArch64::getArchFeatures(AK, Features)
                    : AArch64::getArchFeatures(AK, Features));
}

TEST(TargetParserTest, AArch64ArchName) {
  for (unsigned AK = 0; AK < static_cast<unsigned>(AArch64::ArchKind::AK_LAST);
       AK++)
    EXPECT_TRUE(AK == static_cast<unsigned>(AArch64::ArchKind::AK_LAST)
                    ? AArch64::getArchName(AK).empty()
                    : !AArch64::getArchName(AK).empty());
}

TEST(TargetParserTest, AArch64CPUAttr) {
  for (unsigned AK = 0; AK < static_cast<unsigned>(AArch64::ArchKind::AK_LAST);
       AK++)
    EXPECT_TRUE((AK == static_cast<unsigned>(AArch64::ArchKind::AK_INVALID) ||
                 AK == static_cast<unsigned>(AArch64::ArchKind::AK_LAST))
                    ? AArch64::getCPUAttr(AK).empty()
                    : !AArch64::getCPUAttr(AK).empty());
}

TEST(TargetParserTest, AArch64SubArch) {
  for (unsigned AK = 0; AK < static_cast<unsigned>(AArch64::ArchKind::AK_LAST);
       AK++)
    EXPECT_TRUE((AK == static_cast<unsigned>(AArch64::ArchKind::AK_INVALID) ||
                 AK == static_cast<unsigned>(AArch64::ArchKind::AK_LAST))
                    ? AArch64::getSubArch(AK).empty()
                    : !AArch64::getSubArch(AK).empty());
}

TEST(TargetParserTest, AArch64ArchAttr) {
  for (unsigned AK = 0; AK < static_cast<unsigned>(AArch64::ArchKind::AK_LAST);
       AK++)
    EXPECT_TRUE(
        AK == static_cast<unsigned>(AArch64::ArchKind::AK_LAST)
            ? (ARMBuildAttrs::CPUArch::v8_A == AArch64::getArchAttr(AK))
            : (kAArch64ARCHNames[AK].ArchAttr == AArch64::getArchAttr(AK)));
}

TEST(TargetParserTest, AArch64ArchExtName) {
  for (AArch64::ArchExtKind AEK = static_cast<AArch64::ArchExtKind>(0);
       AEK <= AArch64::ArchExtKind::AEK_RAS;
       AEK = static_cast<AArch64::ArchExtKind>(static_cast<unsigned>(AEK) + 1))
    EXPECT_TRUE(contains(kAArch64ArchExtKinds, static_cast<unsigned>(AEK))
                    ? !AArch64::getArchExtName(AEK).empty()
                    : AArch64::getArchExtName(AEK).empty());
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
    EXPECT_STREQ(ArchExt[i][2], AArch64::getArchExtFeature(ArchExt[i][0]));
    EXPECT_STREQ(ArchExt[i][3], AArch64::getArchExtFeature(ArchExt[i][1]));
  }
}

TEST(TargetParserTest, AArch64DefaultCPU) {
  const char *Arch[] = {"armv8a",    "armv8-a",  "armv8",    "armv8.1a",
                        "armv8.1-a", "armv8.2a", "armv8.2-a"};

  for (unsigned i = 0; i < array_lengthof(Arch); i++)
    EXPECT_FALSE(AArch64::getDefaultCPU(Arch[i]).empty());
}

TEST(TargetParserTest, AArch64parseArch) {
  const char *Arch[] = {"armv8",     "armv8a",   "armv8-a",  "armv8.1a",
                        "armv8.1-a", "armv8.2a", "armv8.2-a"};

  for (unsigned i = 0; i < array_lengthof(Arch); i++)
    EXPECT_NE(static_cast<unsigned>(AArch64::ArchKind::AK_INVALID),
              AArch64::parseArch(Arch[i]));
  EXPECT_EQ(static_cast<unsigned>(AArch64::ArchKind::AK_INVALID),
            AArch64::parseArch("aarch64"));
  EXPECT_EQ(static_cast<unsigned>(AArch64::ArchKind::AK_INVALID),
            AArch64::parseArch("arm64"));
}

TEST(TargetParserTest, AArch64parseArchExt) {
  const char *ArchExt[] = {"none", "crc",  "crypto",  "fp",
                           "simd", "fp16", "profile", "ras"};

  for (unsigned i = 0; i < array_lengthof(ArchExt); i++)
    EXPECT_NE(AArch64::AEK_INVALID, AArch64::parseArchExt(ArchExt[i]));
}

TEST(TargetParserTest, AArch64parseCPUArch) {
  const char *CPU[] = {"cortex-a35", "cortex-a53", "cortex-a57",
                       "cortex-a72", "cortex-a73", "cyclone",
                       "exynos-m1",  "exynos-m2",  "kryo",
                       "vulcan"};

  for (const auto &AArch64CPUName : kAArch64CPUNames)
    EXPECT_TRUE(contains(CPU, AArch64CPUName.Name)
                    ? (static_cast<unsigned>(AArch64::ArchKind::AK_INVALID) !=
                       AArch64::parseCPUArch(AArch64CPUName.Name))
                    : (static_cast<unsigned>(AArch64::ArchKind::AK_INVALID) ==
                       AArch64::parseCPUArch(AArch64CPUName.Name)));
}
}
