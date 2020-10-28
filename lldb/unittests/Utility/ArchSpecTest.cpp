//===-- ArchSpecTest.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Utility/ArchSpec.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Support/YAMLParser.h"

using namespace lldb;
using namespace lldb_private;

TEST(ArchSpecTest, TestParseMachCPUDashSubtypeTripleSimple) {

  // Success conditions.  Valid cpu/subtype combinations using both - and .
  ArchSpec AS;
  EXPECT_TRUE(ParseMachCPUDashSubtypeTriple("12-10", AS));
  EXPECT_EQ(12u, AS.GetMachOCPUType());
  EXPECT_EQ(10u, AS.GetMachOCPUSubType());

  AS = ArchSpec();
  EXPECT_TRUE(ParseMachCPUDashSubtypeTriple("12-15", AS));
  EXPECT_EQ(12u, AS.GetMachOCPUType());
  EXPECT_EQ(15u, AS.GetMachOCPUSubType());

  AS = ArchSpec();
  EXPECT_TRUE(ParseMachCPUDashSubtypeTriple("12.15", AS));
  EXPECT_EQ(12u, AS.GetMachOCPUType());
  EXPECT_EQ(15u, AS.GetMachOCPUSubType());

  // Failure conditions.

  // Valid string, unknown cpu/subtype.
  AS = ArchSpec();
  EXPECT_TRUE(ParseMachCPUDashSubtypeTriple("13.11", AS));
  EXPECT_EQ(0u, AS.GetMachOCPUType());
  EXPECT_EQ(0u, AS.GetMachOCPUSubType());

  // Missing / invalid cpu or subtype
  AS = ArchSpec();
  EXPECT_FALSE(ParseMachCPUDashSubtypeTriple("13", AS));

  AS = ArchSpec();
  EXPECT_FALSE(ParseMachCPUDashSubtypeTriple("13.A", AS));

  AS = ArchSpec();
  EXPECT_FALSE(ParseMachCPUDashSubtypeTriple("A.13", AS));

  // Empty string.
  AS = ArchSpec();
  EXPECT_FALSE(ParseMachCPUDashSubtypeTriple("", AS));
}

TEST(ArchSpecTest, TestParseMachCPUDashSubtypeTripleExtra) {
  ArchSpec AS;
  EXPECT_TRUE(ParseMachCPUDashSubtypeTriple("12-15-vendor-os", AS));
  EXPECT_EQ(12u, AS.GetMachOCPUType());
  EXPECT_EQ(15u, AS.GetMachOCPUSubType());
  EXPECT_EQ("vendor", AS.GetTriple().getVendorName());
  EXPECT_EQ("os", AS.GetTriple().getOSName());

  AS = ArchSpec();
  EXPECT_TRUE(ParseMachCPUDashSubtypeTriple("12-10-vendor-os-name", AS));
  EXPECT_EQ(12u, AS.GetMachOCPUType());
  EXPECT_EQ(10u, AS.GetMachOCPUSubType());
  EXPECT_EQ("vendor", AS.GetTriple().getVendorName());
  EXPECT_EQ("os", AS.GetTriple().getOSName());

  AS = ArchSpec();
  EXPECT_TRUE(ParseMachCPUDashSubtypeTriple("12-15-vendor.os-name", AS));
  EXPECT_EQ(12u, AS.GetMachOCPUType());
  EXPECT_EQ(15u, AS.GetMachOCPUSubType());
  EXPECT_EQ("vendor.os", AS.GetTriple().getVendorName());
  EXPECT_EQ("name", AS.GetTriple().getOSName());

  // These there should parse correctly, but the vendor / OS should be defaulted
  // since they are unrecognized.
  AS = ArchSpec();
  EXPECT_TRUE(ParseMachCPUDashSubtypeTriple("12-10-vendor", AS));
  EXPECT_EQ(12u, AS.GetMachOCPUType());
  EXPECT_EQ(10u, AS.GetMachOCPUSubType());
  EXPECT_EQ("apple", AS.GetTriple().getVendorName());
  EXPECT_EQ("", AS.GetTriple().getOSName());

  AS = ArchSpec();
  EXPECT_FALSE(ParseMachCPUDashSubtypeTriple("12.10.10", AS));

  AS = ArchSpec();
  EXPECT_FALSE(ParseMachCPUDashSubtypeTriple("12-10.10", AS));
}

TEST(ArchSpecTest, TestSetTriple) {
  ArchSpec AS;

  // Various flavors of valid triples.
  EXPECT_TRUE(AS.SetTriple("12-10-apple-darwin"));
  EXPECT_EQ(uint32_t(llvm::MachO::CPU_TYPE_ARM), AS.GetMachOCPUType());
  EXPECT_EQ(10u, AS.GetMachOCPUSubType());
  EXPECT_TRUE(llvm::StringRef(AS.GetTriple().str())
                  .consume_front("armv7f-apple-darwin"));
  EXPECT_EQ(ArchSpec::eCore_arm_armv7f, AS.GetCore());

  AS = ArchSpec();
  EXPECT_TRUE(AS.SetTriple("18.100-apple-darwin"));
  EXPECT_EQ(uint32_t(llvm::MachO::CPU_TYPE_POWERPC), AS.GetMachOCPUType());
  EXPECT_EQ(100u, AS.GetMachOCPUSubType());
  EXPECT_TRUE(llvm::StringRef(AS.GetTriple().str())
                  .consume_front("powerpc-apple-darwin"));
  EXPECT_EQ(ArchSpec::eCore_ppc_ppc970, AS.GetCore());

  AS = ArchSpec();
  EXPECT_TRUE(AS.SetTriple("i686-pc-windows"));
  EXPECT_EQ(llvm::Triple::x86, AS.GetTriple().getArch());
  EXPECT_EQ(llvm::Triple::PC, AS.GetTriple().getVendor());
  EXPECT_EQ(llvm::Triple::Win32, AS.GetTriple().getOS());
  EXPECT_TRUE(
      llvm::StringRef(AS.GetTriple().str()).consume_front("i686-pc-windows"));
  EXPECT_STREQ("i686", AS.GetArchitectureName());
  EXPECT_EQ(ArchSpec::eCore_x86_32_i686, AS.GetCore());

  // Various flavors of invalid triples.
  AS = ArchSpec();
  EXPECT_FALSE(AS.SetTriple("unknown-unknown-unknown"));

  AS = ArchSpec();
  EXPECT_FALSE(AS.SetTriple("unknown"));

  AS = ArchSpec();
  EXPECT_FALSE(AS.SetTriple(""));
}

TEST(ArchSpecTest, MergeFrom) {
  {
    ArchSpec A;
    ArchSpec B("x86_64-pc-linux");

    EXPECT_FALSE(A.IsValid());
    ASSERT_TRUE(B.IsValid());
    EXPECT_EQ(llvm::Triple::ArchType::x86_64, B.GetTriple().getArch());
    EXPECT_EQ(llvm::Triple::VendorType::PC, B.GetTriple().getVendor());
    EXPECT_EQ(llvm::Triple::OSType::Linux, B.GetTriple().getOS());
    EXPECT_EQ(ArchSpec::eCore_x86_64_x86_64, B.GetCore());

    A.MergeFrom(B);
    ASSERT_TRUE(A.IsValid());
    EXPECT_EQ(llvm::Triple::ArchType::x86_64, A.GetTriple().getArch());
    EXPECT_EQ(llvm::Triple::VendorType::PC, A.GetTriple().getVendor());
    EXPECT_EQ(llvm::Triple::OSType::Linux, A.GetTriple().getOS());
    EXPECT_EQ(ArchSpec::eCore_x86_64_x86_64, A.GetCore());
  }
  {
    ArchSpec A("aarch64");
    ArchSpec B("aarch64--linux-android");

    ArchSpec C("arm64_32");
    ArchSpec D("arm64_32--watchos");

    EXPECT_TRUE(A.IsValid());
    EXPECT_TRUE(B.IsValid());
    EXPECT_TRUE(C.IsValid());
    EXPECT_TRUE(D.IsValid());

    EXPECT_EQ(llvm::Triple::ArchType::aarch64, B.GetTriple().getArch());
    EXPECT_EQ(llvm::Triple::VendorType::UnknownVendor,
              B.GetTriple().getVendor());
    EXPECT_EQ(llvm::Triple::OSType::Linux, B.GetTriple().getOS());
    EXPECT_EQ(llvm::Triple::EnvironmentType::Android,
              B.GetTriple().getEnvironment());

    A.MergeFrom(B);
    EXPECT_EQ(llvm::Triple::ArchType::aarch64, A.GetTriple().getArch());
    EXPECT_EQ(llvm::Triple::VendorType::UnknownVendor,
              A.GetTriple().getVendor());
    EXPECT_EQ(llvm::Triple::OSType::Linux, A.GetTriple().getOS());
    EXPECT_EQ(llvm::Triple::EnvironmentType::Android,
              A.GetTriple().getEnvironment());

    EXPECT_EQ(llvm::Triple::ArchType::aarch64_32, D.GetTriple().getArch());
    EXPECT_EQ(llvm::Triple::VendorType::UnknownVendor,
              D.GetTriple().getVendor());
    EXPECT_EQ(llvm::Triple::OSType::WatchOS, D.GetTriple().getOS());

    C.MergeFrom(D);
    EXPECT_EQ(llvm::Triple::ArchType::aarch64_32, C.GetTriple().getArch());
    EXPECT_EQ(llvm::Triple::VendorType::UnknownVendor,
              C.GetTriple().getVendor());
    EXPECT_EQ(llvm::Triple::OSType::WatchOS, C.GetTriple().getOS());
  }
  {
    ArchSpec A, B;
    A.SetArchitecture(eArchTypeELF, llvm::ELF::EM_ARM,
                      LLDB_INVALID_CPUTYPE, llvm::ELF::ELFOSABI_NONE);
    B.SetArchitecture(eArchTypeELF, llvm::ELF::EM_ARM,
                      LLDB_INVALID_CPUTYPE, llvm::ELF::ELFOSABI_LINUX);

    EXPECT_TRUE(A.IsValid());
    EXPECT_TRUE(B.IsValid());

    EXPECT_EQ(llvm::Triple::ArchType::arm, B.GetTriple().getArch());
    EXPECT_EQ(llvm::Triple::VendorType::UnknownVendor,
              B.GetTriple().getVendor());
    EXPECT_EQ(llvm::Triple::OSType::Linux, B.GetTriple().getOS());
    EXPECT_EQ(llvm::Triple::EnvironmentType::UnknownEnvironment,
              B.GetTriple().getEnvironment());

    A.MergeFrom(B);
    EXPECT_EQ(llvm::Triple::ArchType::arm, A.GetTriple().getArch());
    EXPECT_EQ(llvm::Triple::VendorType::UnknownVendor,
              A.GetTriple().getVendor());
    EXPECT_EQ(llvm::Triple::OSType::Linux, A.GetTriple().getOS());
    EXPECT_EQ(llvm::Triple::EnvironmentType::UnknownEnvironment,
              A.GetTriple().getEnvironment());
  }
  {
    ArchSpec A("arm--linux-eabihf");
    ArchSpec B("armv8l--linux-gnueabihf");

    EXPECT_TRUE(A.IsValid());
    EXPECT_TRUE(B.IsValid());

    EXPECT_EQ(llvm::Triple::ArchType::arm, A.GetTriple().getArch());
    EXPECT_EQ(llvm::Triple::ArchType::arm, B.GetTriple().getArch());

    EXPECT_EQ(ArchSpec::eCore_arm_generic, A.GetCore());
    EXPECT_EQ(ArchSpec::eCore_arm_armv8l, B.GetCore());

    EXPECT_EQ(llvm::Triple::VendorType::UnknownVendor,
              A.GetTriple().getVendor());
    EXPECT_EQ(llvm::Triple::VendorType::UnknownVendor,
              B.GetTriple().getVendor());

    EXPECT_EQ(llvm::Triple::OSType::Linux, A.GetTriple().getOS());
    EXPECT_EQ(llvm::Triple::OSType::Linux, B.GetTriple().getOS());

    EXPECT_EQ(llvm::Triple::EnvironmentType::EABIHF,
              A.GetTriple().getEnvironment());
    EXPECT_EQ(llvm::Triple::EnvironmentType::GNUEABIHF,
              B.GetTriple().getEnvironment());

    A.MergeFrom(B);
    EXPECT_EQ(llvm::Triple::ArchType::arm, A.GetTriple().getArch());
    EXPECT_EQ(ArchSpec::eCore_arm_armv8l, A.GetCore());
    EXPECT_EQ(llvm::Triple::VendorType::UnknownVendor,
              A.GetTriple().getVendor());
    EXPECT_EQ(llvm::Triple::OSType::Linux, A.GetTriple().getOS());
    EXPECT_EQ(llvm::Triple::EnvironmentType::EABIHF,
              A.GetTriple().getEnvironment());
  }
}

TEST(ArchSpecTest, MergeFromMachOUnknown) {
  class MyArchSpec : public ArchSpec {
  public:
    MyArchSpec() {
      this->SetTriple("unknown-mach-64");
      this->m_core = ArchSpec::eCore_uknownMach64;
      this->m_byte_order = eByteOrderLittle;
      this->m_flags = 0;
    }
  };

  MyArchSpec A;
  ASSERT_TRUE(A.IsValid());
  MyArchSpec B;
  ASSERT_TRUE(B.IsValid());
  A.MergeFrom(B);
  ASSERT_EQ(A.GetCore(), ArchSpec::eCore_uknownMach64);
}

TEST(ArchSpecTest, Compatibility) {
  {
    ArchSpec A("x86_64-apple-macosx10.12");
    ArchSpec B("x86_64-apple-macosx10.12");
    ASSERT_TRUE(A.IsExactMatch(B));
    ASSERT_TRUE(A.IsCompatibleMatch(B));
  }
  {
    // The version information is auxiliary to support availability but
    // doesn't affect compatibility.
    ArchSpec A("x86_64-apple-macosx10.11");
    ArchSpec B("x86_64-apple-macosx10.12");
    ASSERT_TRUE(A.IsExactMatch(B));
    ASSERT_TRUE(A.IsCompatibleMatch(B));
  }
  {
    ArchSpec A("x86_64-apple-macosx10.13");
    ArchSpec B("x86_64h-apple-macosx10.13");
    ASSERT_FALSE(A.IsExactMatch(B));
    ASSERT_TRUE(A.IsCompatibleMatch(B));
  }
  {
    ArchSpec A("x86_64-apple-macosx");
    ArchSpec B("x86_64-apple-ios-simulator");
    ASSERT_FALSE(A.IsExactMatch(B));
    ASSERT_FALSE(A.IsCompatibleMatch(B));
  }
  {
    ArchSpec A("x86_64-*-*");
    ArchSpec B("x86_64-apple-ios-simulator");
    ASSERT_FALSE(A.IsExactMatch(B));
    ASSERT_FALSE(A.IsCompatibleMatch(B));
  }
  {
    ArchSpec A("arm64-apple-ios");
    ArchSpec B("arm64-apple-ios-simulator");
    ASSERT_FALSE(A.IsExactMatch(B));
    ASSERT_FALSE(A.IsCompatibleMatch(B));
    ASSERT_FALSE(B.IsCompatibleMatch(A));
    ASSERT_FALSE(B.IsCompatibleMatch(A));
  }
  {
    ArchSpec A("arm64-*-*");
    ArchSpec B("arm64-apple-ios");
    ASSERT_FALSE(A.IsExactMatch(B));
    // FIXME: This looks unintuitive and we should investigate whether
    // this is the desired behavior.
    ASSERT_FALSE(A.IsCompatibleMatch(B));
  }
  {
    ArchSpec A("x86_64-*-*");
    ArchSpec B("x86_64-apple-ios-simulator");
    ASSERT_FALSE(A.IsExactMatch(B));
    // FIXME: See above, though the extra environment complicates things.
    ASSERT_FALSE(A.IsCompatibleMatch(B));
  }
  {
    ArchSpec A("x86_64");
    ArchSpec B("x86_64-apple-macosx10.14");
    // FIXME: The exact match also looks unintuitive.
    ASSERT_TRUE(A.IsExactMatch(B));
    ASSERT_TRUE(A.IsCompatibleMatch(B));
  }
  {
    ArchSpec A("x86_64");
    ArchSpec B("x86_64-apple-ios12.0.0-macabi");
    // FIXME: The exact match also looks unintuitive.
    ASSERT_TRUE(A.IsExactMatch(B));
    ASSERT_TRUE(A.IsCompatibleMatch(B));
  }
  {
    ArchSpec A("x86_64-apple-ios12.0.0");
    ArchSpec B("x86_64-apple-ios12.0.0-macabi");
    ASSERT_FALSE(A.IsExactMatch(B));
    ASSERT_FALSE(A.IsCompatibleMatch(B));
  }
  {
    ArchSpec A("x86_64-apple-macosx10.14.2");
    ArchSpec B("x86_64-apple-ios12.0.0-macabi");
    ASSERT_FALSE(A.IsExactMatch(B));
    ASSERT_TRUE(A.IsCompatibleMatch(B));
  }
  {
    ArchSpec A("x86_64-apple-macosx10.14.2");
    ArchSpec B("x86_64-apple-ios12.0.0-macabi");
    // ios-macabi wins.
    A.MergeFrom(B);
    ASSERT_TRUE(A.IsExactMatch(B));
  }
  {
    ArchSpec A("x86_64-apple-macosx10.14.2");
    ArchSpec B("x86_64-apple-ios12.0.0-macabi");
    ArchSpec C(B);
    // ios-macabi wins.
    B.MergeFrom(A);
    ASSERT_TRUE(B.IsExactMatch(C));
  }
}

TEST(ArchSpecTest, OperatorBool) {
  EXPECT_FALSE(ArchSpec());
  EXPECT_TRUE(ArchSpec("x86_64-pc-linux"));
}

TEST(ArchSpecTest, TripleComponentsWereSpecified) {
  {
    ArchSpec A("");
    ArchSpec B("-");
    ArchSpec C("--");
    ArchSpec D("---");

    ASSERT_FALSE(A.TripleVendorWasSpecified());
    ASSERT_FALSE(A.TripleOSWasSpecified());
    ASSERT_FALSE(A.TripleEnvironmentWasSpecified());

    ASSERT_FALSE(B.TripleVendorWasSpecified());
    ASSERT_FALSE(B.TripleOSWasSpecified());
    ASSERT_FALSE(B.TripleEnvironmentWasSpecified());

    ASSERT_FALSE(C.TripleVendorWasSpecified());
    ASSERT_FALSE(C.TripleOSWasSpecified());
    ASSERT_FALSE(C.TripleEnvironmentWasSpecified());

    ASSERT_FALSE(D.TripleVendorWasSpecified());
    ASSERT_FALSE(D.TripleOSWasSpecified());
    ASSERT_FALSE(D.TripleEnvironmentWasSpecified());
  }
  {
    // TODO: llvm::Triple::normalize treats the missing components from these
    // triples as specified unknown components instead of unspecified
    // components. We need to either change the behavior in llvm or work around
    // this in lldb.
    ArchSpec A("armv7");
    ArchSpec B("armv7-");
    ArchSpec C("armv7--");
    ArchSpec D("armv7---");

    ASSERT_FALSE(A.TripleVendorWasSpecified());
    ASSERT_FALSE(A.TripleOSWasSpecified());
    ASSERT_FALSE(A.TripleEnvironmentWasSpecified());

    ASSERT_TRUE(B.TripleVendorWasSpecified());
    ASSERT_FALSE(B.TripleOSWasSpecified());
    ASSERT_FALSE(B.TripleEnvironmentWasSpecified());

    ASSERT_TRUE(C.TripleVendorWasSpecified());
    ASSERT_TRUE(C.TripleOSWasSpecified());
    ASSERT_FALSE(C.TripleEnvironmentWasSpecified());

    ASSERT_TRUE(D.TripleVendorWasSpecified());
    ASSERT_TRUE(D.TripleOSWasSpecified());
    ASSERT_TRUE(D.TripleEnvironmentWasSpecified());
  }
  {
    ArchSpec A("x86_64-unknown");
    ArchSpec B("powerpc-unknown-linux");
    ArchSpec C("i386-pc-windows-msvc");
    ArchSpec D("aarch64-unknown-linux-android");

    ASSERT_TRUE(A.TripleVendorWasSpecified());
    ASSERT_FALSE(A.TripleOSWasSpecified());
    ASSERT_FALSE(A.TripleEnvironmentWasSpecified());

    ASSERT_TRUE(B.TripleVendorWasSpecified());
    ASSERT_TRUE(B.TripleOSWasSpecified());
    ASSERT_FALSE(B.TripleEnvironmentWasSpecified());

    ASSERT_TRUE(C.TripleVendorWasSpecified());
    ASSERT_TRUE(C.TripleOSWasSpecified());
    ASSERT_TRUE(C.TripleEnvironmentWasSpecified());

    ASSERT_TRUE(D.TripleVendorWasSpecified());
    ASSERT_TRUE(D.TripleOSWasSpecified());
    ASSERT_TRUE(D.TripleEnvironmentWasSpecified());
  }
}

TEST(ArchSpecTest, YAML) {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);

  // Serialize.
  llvm::yaml::Output yout(os);
  std::vector<ArchSpec> archs = {ArchSpec("x86_64-pc-linux"),
                                 ArchSpec("x86_64-apple-macosx10.12"),
                                 ArchSpec("i686-pc-windows")};
  yout << archs;
  os.flush();

  // Deserialize.
  std::vector<ArchSpec> deserialized;
  llvm::yaml::Input yin(buffer);
  yin >> deserialized;

  EXPECT_EQ(archs, deserialized);
}
