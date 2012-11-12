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

  T = Triple("x86_64-pc-linux-gnu");
  EXPECT_EQ(Triple::x86_64, T.getArch());
  EXPECT_EQ(Triple::PC, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::GNU, T.getEnvironment());

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
  for (int Arch = 1+Triple::UnknownArch; Arch <= Triple::amdil; ++Arch) {
    C[0] = Triple::getArchTypeName(Triple::ArchType(Arch));
    for (int Vendor = 1+Triple::UnknownVendor; Vendor <= Triple::PC;
         ++Vendor) {
      C[1] = Triple::getVendorTypeName(Triple::VendorType(Vendor));
      for (int OS = 1+Triple::UnknownOS; OS <= Triple::Minix; ++OS) {
        C[2] = Triple::getOSTypeName(Triple::OSType(OS));

        std::string E = Join(C[0], C[1], C[2]);
        EXPECT_EQ(E, Triple::normalize(Join(C[0], C[1], C[2])));

        EXPECT_EQ(E, Triple::normalize(Join(C[0], C[2], C[1])));
        EXPECT_EQ(E, Triple::normalize(Join(C[1], C[2], C[0])));
        EXPECT_EQ(E, Triple::normalize(Join(C[1], C[0], C[2])));
        EXPECT_EQ(E, Triple::normalize(Join(C[2], C[0], C[1])));
        EXPECT_EQ(E, Triple::normalize(Join(C[2], C[1], C[0])));

        for (int Env = 1+Triple::UnknownEnvironment; Env <= Triple::MachO;
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
  EXPECT_EQ("i386--mingw32", Triple::normalize("i386-mingw32")); // i386-pc-mingw32
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
}

TEST(TripleTest, BitWidthArchVariants) {
  Triple T;
  EXPECT_EQ(Triple::UnknownArch, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::UnknownArch, T.get64BitArchVariant().getArch());

  T.setArch(Triple::UnknownArch);
  EXPECT_EQ(Triple::UnknownArch, T.get32BitArchVariant().getArch());
  EXPECT_EQ(Triple::UnknownArch, T.get64BitArchVariant().getArch());

  T.setArch(Triple::arm);
  EXPECT_EQ(Triple::arm, T.get32BitArchVariant().getArch());
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
}

TEST(TripleTest, getOSVersion) {
  Triple T;
  unsigned Major, Minor, Micro;

  T = Triple("i386-apple-darwin9");
  T.getMacOSXVersion(Major, Minor, Micro);
  EXPECT_EQ((unsigned)10, Major);
  EXPECT_EQ((unsigned)5, Minor);
  EXPECT_EQ((unsigned)0, Micro);
  T.getiOSVersion(Major, Minor, Micro);
  EXPECT_EQ((unsigned)3, Major);
  EXPECT_EQ((unsigned)0, Minor);
  EXPECT_EQ((unsigned)0, Micro);

  T = Triple("x86_64-apple-darwin9");
  T.getMacOSXVersion(Major, Minor, Micro);
  EXPECT_EQ((unsigned)10, Major);
  EXPECT_EQ((unsigned)5, Minor);
  EXPECT_EQ((unsigned)0, Micro);
  T.getiOSVersion(Major, Minor, Micro);
  EXPECT_EQ((unsigned)3, Major);
  EXPECT_EQ((unsigned)0, Minor);
  EXPECT_EQ((unsigned)0, Micro);

  T = Triple("x86_64-apple-macosx");
  T.getMacOSXVersion(Major, Minor, Micro);
  EXPECT_EQ((unsigned)10, Major);
  EXPECT_EQ((unsigned)4, Minor);
  EXPECT_EQ((unsigned)0, Micro);
  T.getiOSVersion(Major, Minor, Micro);
  EXPECT_EQ((unsigned)3, Major);
  EXPECT_EQ((unsigned)0, Minor);
  EXPECT_EQ((unsigned)0, Micro);

  T = Triple("x86_64-apple-macosx10.7");
  T.getMacOSXVersion(Major, Minor, Micro);
  EXPECT_EQ((unsigned)10, Major);
  EXPECT_EQ((unsigned)7, Minor);
  EXPECT_EQ((unsigned)0, Micro);
  T.getiOSVersion(Major, Minor, Micro);
  EXPECT_EQ((unsigned)3, Major);
  EXPECT_EQ((unsigned)0, Minor);
  EXPECT_EQ((unsigned)0, Micro);

  T = Triple("armv7-apple-ios");
  T.getMacOSXVersion(Major, Minor, Micro);
  EXPECT_EQ((unsigned)10, Major);
  EXPECT_EQ((unsigned)4, Minor);
  EXPECT_EQ((unsigned)0, Micro);
  T.getiOSVersion(Major, Minor, Micro);
  EXPECT_EQ((unsigned)3, Major);
  EXPECT_EQ((unsigned)0, Minor);
  EXPECT_EQ((unsigned)0, Micro);

  T = Triple("armv7-apple-ios5.0");
  T.getMacOSXVersion(Major, Minor, Micro);
  EXPECT_EQ((unsigned)10, Major);
  EXPECT_EQ((unsigned)4, Minor);
  EXPECT_EQ((unsigned)0, Micro);
  T.getiOSVersion(Major, Minor, Micro);
  EXPECT_EQ((unsigned)5, Major);
  EXPECT_EQ((unsigned)0, Minor);
  EXPECT_EQ((unsigned)0, Micro);
}

}
