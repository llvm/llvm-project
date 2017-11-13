//===-- ArchSpecTest.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Utility/ArchSpec.h"
#include "llvm/BinaryFormat/MachO.h"

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
