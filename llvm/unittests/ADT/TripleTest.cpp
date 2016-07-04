//===----------- Triple.cpp - Triple unit tests ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/ADT/Triple.h"

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

  T = Triple("powerpc-bgp-linux");
  EXPECT_EQ(Triple::ppc, T.getArch());
  EXPECT_EQ(Triple::BGP, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("powerpc-bgp-cnk");
  EXPECT_EQ(Triple::ppc, T.getArch());
  EXPECT_EQ(Triple::BGP, T.getVendor());
  EXPECT_EQ(Triple::CNK, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("powerpc64-bgq-linux");
  EXPECT_EQ(Triple::ppc64, T.getArch());
  EXPECT_EQ(Triple::BGQ, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

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

  T = Triple("arm-none-none-eabi");
  EXPECT_EQ(Triple::arm, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());
  EXPECT_EQ(Triple::EABI, T.getEnvironment());

  T = Triple("arm-none-linux-musleabi");
  EXPECT_EQ(Triple::arm, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::MuslEABI, T.getEnvironment());

  T = Triple("armv6hl-none-linux-gnueabi");
  EXPECT_EQ(Triple::arm, T.getArch());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::GNUEABI, T.getEnvironment());

  T = Triple("armv7hl-none-linux-gnueabi");
  EXPECT_EQ(Triple::arm, T.getArch());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::GNUEABI, T.getEnvironment());

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

  T = Triple("x86_64-unknown-cloudabi");
  EXPECT_EQ(Triple::x86_64, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::CloudABI, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("wasm32-unknown-unknown");
  EXPECT_EQ(Triple::wasm32, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("wasm64-unknown-unknown");
  EXPECT_EQ(Triple::wasm64, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());
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

  T = Triple("huh");
  EXPECT_EQ(Triple::UnknownArch, T.getArch());
}

static std::string Join(StringRef A, StringRef B, StringRef C) {
  std::string Str = A; Str += '-'; Str += B; Str += '-'; Str += C;
  return Str;
}

static std::string Join(StringRef A, StringRef B, StringRef C, StringRef D) {
  std::string Str = A; Str += '-'; Str += B; Str += '-'; Str += C; Str += '-';
  Str += D; return Str;
}

TEST(TripleTest, Normalization) {

  EXPECT_EQ("", Triple::normalize(""));
  EXPECT_EQ("-", Triple::normalize("-"));
  EXPECT_EQ("--", Triple::normalize("--"));
  EXPECT_EQ("---", Triple::normalize("---"));
  EXPECT_EQ("----", Triple::normalize("----"));

  EXPECT_EQ("a", Triple::normalize("a"));
  EXPECT_EQ("a-b", Triple::normalize("a-b"));
  EXPECT_EQ("a-b-c", Triple::normalize("a-b-c"));
  EXPECT_EQ("a-b-c-d", Triple::normalize("a-b-c-d"));

  EXPECT_EQ("i386-b-c", Triple::normalize("i386-b-c"));
  EXPECT_EQ("i386-a-c", Triple::normalize("a-i386-c"));
  EXPECT_EQ("i386-a-b", Triple::normalize("a-b-i386"));
  EXPECT_EQ("i386-a-b-c", Triple::normalize("a-b-c-i386"));

  EXPECT_EQ("a-pc-c", Triple::normalize("a-pc-c"));
  EXPECT_EQ("-pc-b-c", Triple::normalize("pc-b-c"));
  EXPECT_EQ("a-pc-b", Triple::normalize("a-b-pc"));
  EXPECT_EQ("a-pc-b-c", Triple::normalize("a-b-c-pc"));

  EXPECT_EQ("a-b-linux", Triple::normalize("a-b-linux"));
  EXPECT_EQ("--linux-b-c", Triple::normalize("linux-b-c"));
  EXPECT_EQ("a--linux-c", Triple::normalize("a-linux-c"));

  EXPECT_EQ("i386-pc-a", Triple::normalize("a-pc-i386"));
  EXPECT_EQ("i386-pc-", Triple::normalize("-pc-i386"));
  EXPECT_EQ("-pc-linux-c", Triple::normalize("linux-pc-c"));
  EXPECT_EQ("-pc-linux", Triple::normalize("linux-pc-"));

  EXPECT_EQ("i386", Triple::normalize("i386"));
  EXPECT_EQ("-pc", Triple::normalize("pc"));
  EXPECT_EQ("--linux", Triple::normalize("linux"));

  EXPECT_EQ("x86_64--linux-gnu", Triple::normalize("x86_64-gnu-linux"));

  // Check that normalizing a permutated set of valid components returns a
  // triple with the unpermuted components.
  StringRef C[4];
  for (int Arch = 1+Triple::UnknownArch; Arch <= Triple::LastArchType; ++Arch) {
    C[0] = Triple::getArchTypeName(Triple::ArchType(Arch));
    for (int Vendor = 1+Triple::UnknownVendor; Vendor <= Triple::LastVendorType;
         ++Vendor) {
      C[1] = Triple::getVendorTypeName(Triple::VendorType(Vendor));
      for (int OS = 1+Triple::UnknownOS; OS <= Triple::LastOSType; ++OS) {
        if (OS == Triple::Win32)
          continue;

        C[2] = Triple::getOSTypeName(Triple::OSType(OS));

        std::string E = Join(C[0], C[1], C[2]);
        EXPECT_EQ(E, Triple::normalize(Join(C[0], C[1], C[2])));

        EXPECT_EQ(E, Triple::normalize(Join(C[0], C[2], C[1])));
        EXPECT_EQ(E, Triple::normalize(Join(C[1], C[2], C[0])));
        EXPECT_EQ(E, Triple::normalize(Join(C[1], C[0], C[2])));
        EXPECT_EQ(E, Triple::normalize(Join(C[2], C[0], C[1])));
        EXPECT_EQ(E, Triple::normalize(Join(C[2], C[1], C[0])));

        for (int Env = 1 + Triple::UnknownEnvironment; Env <= Triple::LastEnvironmentType;
             ++Env) {
          C[3] = Triple::getEnvironmentTypeName(Triple::EnvironmentType(Env));

          std::string F = Join(C[0], C[1], C[2], C[3]);
          EXPECT_EQ(F, Triple::normalize(Join(C[0], C[1], C[2], C[3])));

          EXPECT_EQ(F, Triple::normalize(Join(C[0], C[1], C[3], C[2])));
          EXPECT_EQ(F, Triple::normalize(Join(C[0], C[2], C[3], C[1])));
          EXPECT_EQ(F, Triple::normalize(Join(C[0], C[2], C[1], C[3])));
          EXPECT_EQ(F, Triple::normalize(Join(C[0], C[3], C[1], C[2])));
          EXPECT_EQ(F, Triple::normalize(Join(C[0], C[3], C[2], C[1])));
          EXPECT_EQ(F, Triple::normalize(Join(C[1], C[2], C[3], C[0])));
          EXPECT_EQ(F, Triple::normalize(Join(C[1], C[2], C[0], C[3])));
          EXPECT_EQ(F, Triple::normalize(Join(C[1], C[3], C[0], C[2])));
          EXPECT_EQ(F, Triple::normalize(Join(C[1], C[3], C[2], C[0])));
          EXPECT_EQ(F, Triple::normalize(Join(C[1], C[0], C[2], C[3])));
          EXPECT_EQ(F, Triple::normalize(Join(C[1], C[0], C[3], C[2])));
          EXPECT_EQ(F, Triple::normalize(Join(C[2], C[3], C[0], C[1])));
          EXPECT_EQ(F, Triple::normalize(Join(C[2], C[3], C[1], C[0])));
          EXPECT_EQ(F, Triple::normalize(Join(C[2], C[0], C[1], C[3])));
          EXPECT_EQ(F, Triple::normalize(Join(C[2], C[0], C[3], C[1])));
          EXPECT_EQ(F, Triple::normalize(Join(C[2], C[1], C[3], C[0])));
          EXPECT_EQ(F, Triple::normalize(Join(C[2], C[1], C[0], C[3])));
          EXPECT_EQ(F, Triple::normalize(Join(C[3], C[0], C[1], C[2])));
          EXPECT_EQ(F, Triple::normalize(Join(C[3], C[0], C[2], C[1])));
          EXPECT_EQ(F, Triple::normalize(Join(C[3], C[1], C[2], C[0])));
          EXPECT_EQ(F, Triple::normalize(Join(C[3], C[1], C[0], C[2])));
          EXPECT_EQ(F, Triple::normalize(Join(C[3], C[2], C[0], C[1])));
          EXPECT_EQ(F, Triple::normalize(Join(C[3], C[2], C[1], C[0])));
        }
      }
    }
  }

  // Various real-world funky triples.  The value returned by GCC's config.sub
  // is given in the comment.
  EXPECT_EQ("i386--windows-gnu", Triple::normalize("i386-mingw32")); // i386-pc-mingw32
  EXPECT_EQ("x86_64--linux-gnu", Triple::normalize("x86_64-linux-gnu")); // x86_64-pc-linux-gnu
  EXPECT_EQ("i486--linux-gnu", Triple::normalize("i486-linux-gnu")); // i486-pc-linux-gnu
  EXPECT_EQ("i386-redhat-linux", Triple::normalize("i386-redhat-linux")); // i386-redhat-linux-gnu
  EXPECT_EQ("i686--linux", Triple::normalize("i686-linux")); // i686-pc-linux-gnu
  EXPECT_EQ("arm-none--eabi", Triple::normalize("arm-none-eabi")); // arm-none-eabi
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

  T.setEnvironmentName("amdopencl");
  EXPECT_EQ(Triple::AMDOpenCL, T.getEnvironment());
}

TEST(TripleTest, BitWidthPredicates) {
  Triple T;
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_FALSE(T.isArch32Bit());
  EXPECT_FALSE(T.isArch64Bit());

  T.setArch(Triple::arm);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_TRUE(T.isArch32Bit());
  EXPECT_FALSE(T.isArch64Bit());

  T.setArch(Triple::hexagon);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_TRUE(T.isArch32Bit());
  EXPECT_FALSE(T.isArch64Bit());

  T.setArch(Triple::mips);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_TRUE(T.isArch32Bit());
  EXPECT_FALSE(T.isArch64Bit());

  T.setArch(Triple::mips64);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_FALSE(T.isArch32Bit());
  EXPECT_TRUE(T.isArch64Bit());

  T.setArch(Triple::msp430);
  EXPECT_TRUE(T.isArch16Bit());
  EXPECT_FALSE(T.isArch32Bit());
  EXPECT_FALSE(T.isArch64Bit());

  T.setArch(Triple::ppc);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_TRUE(T.isArch32Bit());
  EXPECT_FALSE(T.isArch64Bit());

  T.setArch(Triple::ppc64);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_FALSE(T.isArch32Bit());
  EXPECT_TRUE(T.isArch64Bit());

  T.setArch(Triple::x86);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_TRUE(T.isArch32Bit());
  EXPECT_FALSE(T.isArch64Bit());

  T.setArch(Triple::x86_64);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_FALSE(T.isArch32Bit());
  EXPECT_TRUE(T.isArch64Bit());

  T.setArch(Triple::amdil);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_TRUE(T.isArch32Bit());
  EXPECT_FALSE(T.isArch64Bit());

  T.setArch(Triple::amdil64);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_FALSE(T.isArch32Bit());
  EXPECT_TRUE(T.isArch64Bit());

  T.setArch(Triple::hsail);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_TRUE(T.isArch32Bit());
  EXPECT_FALSE(T.isArch64Bit());

  T.setArch(Triple::hsail64);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_FALSE(T.isArch32Bit());
  EXPECT_TRUE(T.isArch64Bit());

  T.setArch(Triple::spir);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_TRUE(T.isArch32Bit());
  EXPECT_FALSE(T.isArch64Bit());

  T.setArch(Triple::spir64);
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_FALSE(T.isArch32Bit());
  EXPECT_TRUE(T.isArch64Bit());

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
  EXPECT_EQ(Triple::mips64, T.get64BitArchVariant().getArch());

  T.setArch(Triple::mipsel);
  EXPECT_EQ(Triple::mipsel, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::mips64el, T.get64BitArchVariant().getArch());

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
  EXPECT_EQ(Triple::mips64, T.get64BitArchVariant().getArch());

  T.setArch(Triple::mips64el);
  EXPECT_EQ(Triple::mipsel, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::mips64el, T.get64BitArchVariant().getArch());

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

  T.setArch(Triple::wasm32);
  EXPECT_EQ(Triple::wasm32, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::wasm64, T.get64BitArchVariant().getArch());

  T.setArch(Triple::wasm64);
  EXPECT_EQ(Triple::wasm32, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::wasm64, T.get64BitArchVariant().getArch());
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
  EXPECT_EQ(Triple::mips64el, T.getLittleEndianArchVariant().getArch());

  T.setArch(Triple::mips64el);
  EXPECT_EQ(Triple::mips64, T.getBigEndianArchVariant().getArch());
  EXPECT_EQ(Triple::mips64el, T.getLittleEndianArchVariant().getArch());

  T.setArch(Triple::mips);
  EXPECT_EQ(Triple::mips, T.getBigEndianArchVariant().getArch());
  EXPECT_EQ(Triple::mipsel, T.getLittleEndianArchVariant().getArch());

  T.setArch(Triple::mipsel);
  EXPECT_EQ(Triple::mips, T.getBigEndianArchVariant().getArch());
  EXPECT_EQ(Triple::mipsel, T.getLittleEndianArchVariant().getArch());

  T.setArch(Triple::ppc);
  EXPECT_EQ(Triple::ppc, T.getBigEndianArchVariant().getArch());
  EXPECT_EQ(Triple::UnknownArch, T.getLittleEndianArchVariant().getArch());

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
}

TEST(TripleTest, getOSVersion) {
  Triple T;
  unsigned Major, Minor, Micro;

  T = Triple("i386-apple-darwin9");
  EXPECT_TRUE(T.isMacOSX());
  EXPECT_FALSE(T.isiOS());
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_TRUE(T.isArch32Bit());
  EXPECT_FALSE(T.isArch64Bit());
  T.getMacOSXVersion(Major, Minor, Micro);
  EXPECT_EQ((unsigned)10, Major);
  EXPECT_EQ((unsigned)5, Minor);
  EXPECT_EQ((unsigned)0, Micro);
  T.getiOSVersion(Major, Minor, Micro);
  EXPECT_EQ((unsigned)5, Major);
  EXPECT_EQ((unsigned)0, Minor);
  EXPECT_EQ((unsigned)0, Micro);

  T = Triple("x86_64-apple-darwin9");
  EXPECT_TRUE(T.isMacOSX());
  EXPECT_FALSE(T.isiOS());
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_FALSE(T.isArch32Bit());
  EXPECT_TRUE(T.isArch64Bit());
  T.getMacOSXVersion(Major, Minor, Micro);
  EXPECT_EQ((unsigned)10, Major);
  EXPECT_EQ((unsigned)5, Minor);
  EXPECT_EQ((unsigned)0, Micro);
  T.getiOSVersion(Major, Minor, Micro);
  EXPECT_EQ((unsigned)5, Major);
  EXPECT_EQ((unsigned)0, Minor);
  EXPECT_EQ((unsigned)0, Micro);

  T = Triple("x86_64-apple-macosx");
  EXPECT_TRUE(T.isMacOSX());
  EXPECT_FALSE(T.isiOS());
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_FALSE(T.isArch32Bit());
  EXPECT_TRUE(T.isArch64Bit());
  T.getMacOSXVersion(Major, Minor, Micro);
  EXPECT_EQ((unsigned)10, Major);
  EXPECT_EQ((unsigned)4, Minor);
  EXPECT_EQ((unsigned)0, Micro);
  T.getiOSVersion(Major, Minor, Micro);
  EXPECT_EQ((unsigned)5, Major);
  EXPECT_EQ((unsigned)0, Minor);
  EXPECT_EQ((unsigned)0, Micro);

  T = Triple("x86_64-apple-macosx10.7");
  EXPECT_TRUE(T.isMacOSX());
  EXPECT_FALSE(T.isiOS());
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_FALSE(T.isArch32Bit());
  EXPECT_TRUE(T.isArch64Bit());
  T.getMacOSXVersion(Major, Minor, Micro);
  EXPECT_EQ((unsigned)10, Major);
  EXPECT_EQ((unsigned)7, Minor);
  EXPECT_EQ((unsigned)0, Micro);
  T.getiOSVersion(Major, Minor, Micro);
  EXPECT_EQ((unsigned)5, Major);
  EXPECT_EQ((unsigned)0, Minor);
  EXPECT_EQ((unsigned)0, Micro);

  T = Triple("armv7-apple-ios");
  EXPECT_FALSE(T.isMacOSX());
  EXPECT_TRUE(T.isiOS());
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_TRUE(T.isArch32Bit());
  EXPECT_FALSE(T.isArch64Bit());
  T.getMacOSXVersion(Major, Minor, Micro);
  EXPECT_EQ((unsigned)10, Major);
  EXPECT_EQ((unsigned)4, Minor);
  EXPECT_EQ((unsigned)0, Micro);
  T.getiOSVersion(Major, Minor, Micro);
  EXPECT_EQ((unsigned)5, Major);
  EXPECT_EQ((unsigned)0, Minor);
  EXPECT_EQ((unsigned)0, Micro);

  T = Triple("armv7-apple-ios7.0");
  EXPECT_FALSE(T.isMacOSX());
  EXPECT_TRUE(T.isiOS());
  EXPECT_FALSE(T.isArch16Bit());
  EXPECT_TRUE(T.isArch32Bit());
  EXPECT_FALSE(T.isArch64Bit());
  T.getMacOSXVersion(Major, Minor, Micro);
  EXPECT_EQ((unsigned)10, Major);
  EXPECT_EQ((unsigned)4, Minor);
  EXPECT_EQ((unsigned)0, Micro);
  T.getiOSVersion(Major, Minor, Micro);
  EXPECT_EQ((unsigned)7, Major);
  EXPECT_EQ((unsigned)0, Minor);
  EXPECT_EQ((unsigned)0, Micro);
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

  EXPECT_EQ(Triple::COFF, Triple("i686--win32").getObjectFormat());

  EXPECT_EQ(Triple::ELF, Triple("i686-pc-windows-msvc-elf").getObjectFormat());
  EXPECT_EQ(Triple::ELF, Triple("i686-pc-cygwin-elf").getObjectFormat());

  Triple MSVCNormalized(Triple::normalize("i686-pc-windows-msvc-elf"));
  EXPECT_EQ(Triple::ELF, MSVCNormalized.getObjectFormat());

  Triple GNUWindowsNormalized(Triple::normalize("i686-pc-windows-gnu-elf"));
  EXPECT_EQ(Triple::ELF, GNUWindowsNormalized.getObjectFormat());

  Triple CygnusNormalised(Triple::normalize("i686-pc-windows-cygnus-elf"));
  EXPECT_EQ(Triple::ELF, CygnusNormalised.getObjectFormat());

  Triple CygwinNormalized(Triple::normalize("i686-pc-cygwin-elf"));
  EXPECT_EQ(Triple::ELF, CygwinNormalized.getObjectFormat());

  Triple T = Triple("");
  T.setObjectFormat(Triple::ELF);
  EXPECT_EQ(Triple::ELF, T.getObjectFormat());
}

TEST(TripleTest, NormalizeWindows) {
  EXPECT_EQ("i686-pc-windows-msvc", Triple::normalize("i686-pc-win32"));
  EXPECT_EQ("i686--windows-msvc", Triple::normalize("i686-win32"));
  EXPECT_EQ("i686-pc-windows-gnu", Triple::normalize("i686-pc-mingw32"));
  EXPECT_EQ("i686--windows-gnu", Triple::normalize("i686-mingw32"));
  EXPECT_EQ("i686-pc-windows-gnu", Triple::normalize("i686-pc-mingw32-w64"));
  EXPECT_EQ("i686--windows-gnu", Triple::normalize("i686-mingw32-w64"));
  EXPECT_EQ("i686-pc-windows-cygnus", Triple::normalize("i686-pc-cygwin"));
  EXPECT_EQ("i686--windows-cygnus", Triple::normalize("i686-cygwin"));

  EXPECT_EQ("x86_64-pc-windows-msvc", Triple::normalize("x86_64-pc-win32"));
  EXPECT_EQ("x86_64--windows-msvc", Triple::normalize("x86_64-win32"));
  EXPECT_EQ("x86_64-pc-windows-gnu", Triple::normalize("x86_64-pc-mingw32"));
  EXPECT_EQ("x86_64--windows-gnu", Triple::normalize("x86_64-mingw32"));
  EXPECT_EQ("x86_64-pc-windows-gnu", Triple::normalize("x86_64-pc-mingw32-w64"));
  EXPECT_EQ("x86_64--windows-gnu", Triple::normalize("x86_64-mingw32-w64"));

  EXPECT_EQ("i686-pc-windows-elf", Triple::normalize("i686-pc-win32-elf"));
  EXPECT_EQ("i686--windows-elf", Triple::normalize("i686-win32-elf"));
  EXPECT_EQ("i686-pc-windows-macho", Triple::normalize("i686-pc-win32-macho"));
  EXPECT_EQ("i686--windows-macho", Triple::normalize("i686-win32-macho"));

  EXPECT_EQ("x86_64-pc-windows-elf", Triple::normalize("x86_64-pc-win32-elf"));
  EXPECT_EQ("x86_64--windows-elf", Triple::normalize("x86_64-win32-elf"));
  EXPECT_EQ("x86_64-pc-windows-macho", Triple::normalize("x86_64-pc-win32-macho"));
  EXPECT_EQ("x86_64--windows-macho", Triple::normalize("x86_64-win32-macho"));

  EXPECT_EQ("i686-pc-windows-cygnus",
            Triple::normalize("i686-pc-windows-cygnus"));
  EXPECT_EQ("i686-pc-windows-gnu", Triple::normalize("i686-pc-windows-gnu"));
  EXPECT_EQ("i686-pc-windows-itanium", Triple::normalize("i686-pc-windows-itanium"));
  EXPECT_EQ("i686-pc-windows-msvc", Triple::normalize("i686-pc-windows-msvc"));

  EXPECT_EQ("i686-pc-windows-elf", Triple::normalize("i686-pc-windows-elf-elf"));
}

TEST(TripleTest, getARMCPUForArch) {
  // Standard ARM Architectures.
  {
    llvm::Triple Triple("armv4-unknown-eabi");
    EXPECT_EQ("strongarm", Triple.getARMCPUForArch());
  }
  {
    llvm::Triple Triple("armv4t-unknown-eabi");
    EXPECT_EQ("arm7tdmi", Triple.getARMCPUForArch());
  }
  {
    llvm::Triple Triple("armv5-unknown-eabi");
    EXPECT_EQ("arm10tdmi", Triple.getARMCPUForArch());
  }
  {
    llvm::Triple Triple("armv5t-unknown-eabi");
    EXPECT_EQ("arm10tdmi", Triple.getARMCPUForArch());
  }
  {
    llvm::Triple Triple("armv5e-unknown-eabi");
    EXPECT_EQ("arm1022e", Triple.getARMCPUForArch());
  }
  {
    llvm::Triple Triple("armv5tej-unknown-eabi");
    EXPECT_EQ("arm926ej-s", Triple.getARMCPUForArch());
  }
  {
    llvm::Triple Triple("armv6-unknown-eabi");
    EXPECT_EQ("arm1136jf-s", Triple.getARMCPUForArch());
  }
  {
    llvm::Triple Triple("armv6j-unknown-eabi");
    EXPECT_EQ("arm1136jf-s", Triple.getARMCPUForArch());
  }
  {
    llvm::Triple Triple("armv6k-unknown-eabi");
    EXPECT_EQ("arm1176j-s", Triple.getARMCPUForArch());
  }
  {
    llvm::Triple Triple("armv6kz-unknown-eabi");
    EXPECT_EQ("arm1176jzf-s", Triple.getARMCPUForArch());
  }
  {
    llvm::Triple Triple("armv6zk-unknown-eabi");
    EXPECT_EQ("arm1176jzf-s", Triple.getARMCPUForArch());
  }
  {
    llvm::Triple Triple("armv6t2-unknown-eabi");
    EXPECT_EQ("arm1156t2-s", Triple.getARMCPUForArch());
  }
  {
    llvm::Triple Triple("armv6m-unknown-eabi");
    EXPECT_EQ("cortex-m0", Triple.getARMCPUForArch());
  }
  {
    llvm::Triple Triple("armv7-unknown-eabi");
    EXPECT_EQ("cortex-a8", Triple.getARMCPUForArch());
  }
  {
    llvm::Triple Triple("armv7a-unknown-eabi");
    EXPECT_EQ("cortex-a8", Triple.getARMCPUForArch());
  }
  {
    llvm::Triple Triple("armv7m-unknown-eabi");
    EXPECT_EQ("cortex-m3", Triple.getARMCPUForArch());
  }
  {
    llvm::Triple Triple("armv7r-unknown-eabi");
    EXPECT_EQ("cortex-r4", Triple.getARMCPUForArch());
  }
  {
    llvm::Triple Triple("armv7r-unknown-eabi");
    EXPECT_EQ("cortex-r4", Triple.getARMCPUForArch());
  }
  {
    llvm::Triple Triple("armv7r-unknown-eabi");
    EXPECT_EQ("cortex-r4", Triple.getARMCPUForArch());
  }
  {
    llvm::Triple Triple("armv7r-unknown-eabi");
    EXPECT_EQ("cortex-r4", Triple.getARMCPUForArch());
  }
  {
    llvm::Triple Triple("armv8a-unknown-eabi");
    EXPECT_EQ("cortex-a53", Triple.getARMCPUForArch());
  }
  {
    llvm::Triple Triple("armv8.1a-unknown-eabi");
    EXPECT_EQ("generic", Triple.getARMCPUForArch());
  }
  // Non-synonym names, using -march style, not default arch.
  {
    llvm::Triple Triple("arm");
    EXPECT_EQ("cortex-a8", Triple.getARMCPUForArch("armv7-a"));
  }
  {
    llvm::Triple Triple("arm");
    EXPECT_EQ("cortex-m3", Triple.getARMCPUForArch("armv7-m"));
  }
  {
    llvm::Triple Triple("arm");
    EXPECT_EQ("cortex-a53", Triple.getARMCPUForArch("armv8"));
  }
  {
    llvm::Triple Triple("arm");
    EXPECT_EQ("cortex-a53", Triple.getARMCPUForArch("armv8-a"));
  }
  // Platform specific defaults.
  {
    llvm::Triple Triple("arm--nacl");
    EXPECT_EQ("cortex-a8", Triple.getARMCPUForArch());
  }
  {
    llvm::Triple Triple("armv6-unknown-freebsd");
    EXPECT_EQ("arm1176jzf-s", Triple.getARMCPUForArch());
  }
  {
    llvm::Triple Triple("thumbv6-unknown-freebsd");
    EXPECT_EQ("arm1176jzf-s", Triple.getARMCPUForArch());
  }
  {
    llvm::Triple Triple("armebv6-unknown-freebsd");
    EXPECT_EQ("arm1176jzf-s", Triple.getARMCPUForArch());
  }
  {
    llvm::Triple Triple("arm--win32");
    EXPECT_EQ("cortex-a9", Triple.getARMCPUForArch());
  }
  // Some alternative architectures
  {
    llvm::Triple Triple("xscale-unknown-eabi");
    EXPECT_EQ("xscale", Triple.getARMCPUForArch());
  }
  {
    llvm::Triple Triple("iwmmxt-unknown-eabi");
    EXPECT_EQ("iwmmxt", Triple.getARMCPUForArch());
  }
  {
    llvm::Triple Triple("armv7s-apple-ios7");
    EXPECT_EQ("swift", Triple.getARMCPUForArch());
  }
  {
    llvm::Triple Triple("armv7k-apple-ios9");
    EXPECT_EQ("cortex-a7", Triple.getARMCPUForArch());
  }
  {
    llvm::Triple Triple("armv7k-apple-watchos3");
    EXPECT_EQ("cortex-a7", Triple.getARMCPUForArch());
  }
  {
    llvm::Triple Triple("armv7k-apple-tvos9");
    EXPECT_EQ("cortex-a7", Triple.getARMCPUForArch());
  }
  {
    llvm::Triple Triple("armv7em-apple-ios7");
    EXPECT_EQ("cortex-m4", Triple.getARMCPUForArch());
  }
  {
    llvm::Triple Triple("armv7l-linux-gnueabihf");
    EXPECT_EQ("cortex-a8", Triple.getARMCPUForArch());
  }
  {
    llvm::Triple Triple("armv6sm-apple-ios7");
    EXPECT_EQ("cortex-m0", Triple.getARMCPUForArch());
  }
  // armeb is permitted, but armebeb is not
  {
    llvm::Triple Triple("armeb-none-eabi");
    EXPECT_EQ("arm7tdmi", Triple.getARMCPUForArch());
  }
  {
    llvm::Triple Triple("armebeb-none-eabi");
    EXPECT_EQ("", Triple.getARMCPUForArch());
  }
  {
    llvm::Triple Triple("armebv6eb-none-eabi");
    EXPECT_EQ("", Triple.getARMCPUForArch());
  }
  // armebv6 and armv6eb are permitted, but armebv6eb is not
  {
    llvm::Triple Triple("armebv6-non-eabi");
    EXPECT_EQ("arm1136jf-s", Triple.getARMCPUForArch());
  }
  {
    llvm::Triple Triple("armv6eb-none-eabi");
    EXPECT_EQ("arm1136jf-s", Triple.getARMCPUForArch());
  }
  // xscaleeb is permitted, but armebxscale is not
  {
    llvm::Triple Triple("xscaleeb-none-eabi");
    EXPECT_EQ("xscale", Triple.getARMCPUForArch());
  }
  {
    llvm::Triple Triple("armebxscale-none-eabi");
    EXPECT_EQ("", Triple.getARMCPUForArch());
  }
}

TEST(TripleTest, NormalizeARM) {
  EXPECT_EQ("armv6--netbsd-eabi", Triple::normalize("armv6-netbsd-eabi"));
  EXPECT_EQ("armv7--netbsd-eabi", Triple::normalize("armv7-netbsd-eabi"));
  EXPECT_EQ("armv6eb--netbsd-eabi", Triple::normalize("armv6eb-netbsd-eabi"));
  EXPECT_EQ("armv7eb--netbsd-eabi", Triple::normalize("armv7eb-netbsd-eabi"));
  EXPECT_EQ("armv6--netbsd-eabihf", Triple::normalize("armv6-netbsd-eabihf"));
  EXPECT_EQ("armv7--netbsd-eabihf", Triple::normalize("armv7-netbsd-eabihf"));
  EXPECT_EQ("armv6eb--netbsd-eabihf", Triple::normalize("armv6eb-netbsd-eabihf"));
  EXPECT_EQ("armv7eb--netbsd-eabihf", Triple::normalize("armv7eb-netbsd-eabihf"));

  Triple T;
  T = Triple("armv6--netbsd-eabi");
  EXPECT_EQ(Triple::arm, T.getArch());
  T = Triple("armv6eb--netbsd-eabi");
  EXPECT_EQ(Triple::armeb, T.getArch());
}

TEST(TripleTest, ParseARMArch) {
  // ARM
  {
    Triple T = Triple("arm");
    EXPECT_EQ(Triple::arm, T.getArch());
  }
  {
    Triple T = Triple("armv6t2");
    EXPECT_EQ(Triple::arm, T.getArch());
  }
  {
    Triple T = Triple("armv8");
    EXPECT_EQ(Triple::arm, T.getArch());
  }
  {
    Triple T = Triple("armeb");
    EXPECT_EQ(Triple::armeb, T.getArch());
  }
  {
    Triple T = Triple("armv5eb");
    EXPECT_EQ(Triple::armeb, T.getArch());
  }
  {
    Triple T = Triple("armebv7m");
    EXPECT_EQ(Triple::armeb, T.getArch());
  }
  {
    Triple T = Triple("armv7eb");
    EXPECT_EQ(Triple::armeb, T.getArch());
  }
  // THUMB
  {
    Triple T = Triple("thumb");
    EXPECT_EQ(Triple::thumb, T.getArch());
  }
  {
    Triple T = Triple("thumbv7a");
    EXPECT_EQ(Triple::thumb, T.getArch());
  }
  {
    Triple T = Triple("thumbeb");
    EXPECT_EQ(Triple::thumbeb, T.getArch());
  }
  {
    Triple T = Triple("thumbv4teb");
    EXPECT_EQ(Triple::thumbeb, T.getArch());
  }
  {
    Triple T = Triple("thumbebv7");
    EXPECT_EQ(Triple::thumbeb, T.getArch());
  }
  {
    Triple T = Triple("armv6m");
    EXPECT_EQ(Triple::thumb, T.getArch());
  }
  {
    Triple T = Triple("thumbv2");
    EXPECT_EQ(Triple::UnknownArch, T.getArch());
  }
  {
    Triple T = Triple("thumbebv6eb");
    EXPECT_EQ(Triple::UnknownArch, T.getArch());
  }
  // AARCH64
  {
    Triple T = Triple("arm64");
    EXPECT_EQ(Triple::aarch64, T.getArch());
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
    Triple T = Triple("aarch64be");
    EXPECT_EQ(Triple::UnknownArch, T.getArch());
  }
  {
    Triple T = Triple("arm64be");
    EXPECT_EQ(Triple::UnknownArch, T.getArch());
  }
}
} // end anonymous namespace
