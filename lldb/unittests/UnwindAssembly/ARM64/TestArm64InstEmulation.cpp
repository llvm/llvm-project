//===-- TestArm64InstEmulation.cpp ----------------------------------------===//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include <optional>
#include <vector>

#include "Plugins/UnwindAssembly/InstEmulation/UnwindAssemblyInstEmulation.h"

#include "lldb/Core/Address.h"
#include "lldb/Core/AddressRange.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Target/UnwindAssembly.h"
#include "lldb/Utility/ArchSpec.h"

#include "Plugins/Disassembler/LLVMC/DisassemblerLLVMC.h"
#include "Plugins/Instruction/ARM64/EmulateInstructionARM64.h"
#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "Plugins/Platform/Linux/PlatformLinux.h"
#include "Plugins/Process/Utility/lldb-arm64-register-enums.h"
#include "Plugins/SymbolFile/Symtab/SymbolFileSymtab.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Target/Target.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/TargetSelect.h"

using namespace lldb;
using namespace lldb_private;

class TestArm64InstEmulation : public testing::Test {
public:
  static void SetUpTestCase();
  static void TearDownTestCase();

  //  virtual void SetUp() override { }
  //  virtual void TearDown() override { }

protected:
};

void TestArm64InstEmulation::SetUpTestCase() {
  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllDisassemblers();
  FileSystem::Initialize();
  HostInfo::Initialize();
  DisassemblerLLVMC::Initialize();
  EmulateInstructionARM64::Initialize();
  ObjectFileELF::Initialize();
  SymbolFileSymtab::Initialize();
  platform_linux::PlatformLinux::Initialize();
  // Creating a Debugger requires a host platform. Set one explicitly so this
  // does not depend on which host the test is built for.
  ArchSpec linux_arm64("aarch64-pc-linux");
  Platform::SetHostPlatform(
      platform_linux::PlatformLinux::CreateInstance(true, &linux_arm64));
}

void TestArm64InstEmulation::TearDownTestCase() {
  platform_linux::PlatformLinux::Terminate();
  SymbolFileSymtab::Terminate();
  ObjectFileELF::Terminate();
  EmulateInstructionARM64::Terminate();
  DisassemblerLLVMC::Terminate();
  HostInfo::Terminate();
  FileSystem::Terminate();
}

TEST_F(TestArm64InstEmulation, TestSimpleDarwinFunction) {
  ArchSpec arch("arm64-apple-ios10");
  std::unique_ptr<UnwindAssemblyInstEmulation> engine(
      static_cast<UnwindAssemblyInstEmulation *>(
          UnwindAssemblyInstEmulation::CreateInstance(arch)));
  ASSERT_NE(nullptr, engine);

  const UnwindPlan::Row *row;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  UnwindPlan::Row::AbstractRegisterLocation regloc;

  // 'int main() { }' compiled for arm64-apple-ios with clang
  uint8_t data[] = {
      0xfd, 0x7b, 0xbf, 0xa9, // 0xa9bf7bfd :  stp x29, x30, [sp, #-0x10]!
      0xfd, 0x03, 0x00, 0x91, // 0x910003fd :  mov x29, sp
      0xff, 0x43, 0x00, 0xd1, // 0xd10043ff :  sub sp, sp, #0x10

      0xbf, 0x03, 0x00, 0x91, // 0x910003bf :  mov sp, x29
      0xfd, 0x7b, 0xc1, 0xa8, // 0xa8c17bfd :  ldp x29, x30, [sp], #16
      0xc0, 0x03, 0x5f, 0xd6, // 0xd65f03c0 :  ret
  };

  // UnwindPlan we expect:

  // row[0]:    0: CFA=sp +0 => fp= <same> lr= <same>
  // row[1]:    4: CFA=sp+16 => fp=[CFA-16] lr=[CFA-8]
  // row[2]:    8: CFA=fp+16 => fp=[CFA-16] lr=[CFA-8]
  // row[2]:   16: CFA=sp+16 => fp=[CFA-16] lr=[CFA-8]
  // row[3]:   20: CFA=sp +0 => fp= <same> lr= <same>

  sample_range = AddressRange(0x1000, sizeof(data));

  EXPECT_TRUE(engine->GetNonCallSiteUnwindPlanFromAssembly(
      sample_range, data, sizeof(data), /*target=*/nullptr, unwind_plan));

  // CFA=sp +0 => fp= <same> lr= <same>
  row = unwind_plan.GetRowForFunctionOffset(0);
  EXPECT_EQ(0, row->GetOffset());
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_sp_arm64);
  EXPECT_TRUE(row->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row->GetCFAValue().GetOffset());

  EXPECT_TRUE(row->GetRegisterInfo(gpr_fp_arm64, regloc));
  EXPECT_TRUE(regloc.IsSame());

  EXPECT_TRUE(row->GetRegisterInfo(gpr_lr_arm64, regloc));
  EXPECT_TRUE(regloc.IsSame());

  // CFA=sp+16 => fp=[CFA-16] lr=[CFA-8]
  row = unwind_plan.GetRowForFunctionOffset(4);
  EXPECT_EQ(4, row->GetOffset());
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_sp_arm64);
  EXPECT_TRUE(row->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row->GetCFAValue().GetOffset());

  EXPECT_TRUE(row->GetRegisterInfo(gpr_fp_arm64, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-16, regloc.GetOffset());

  EXPECT_TRUE(row->GetRegisterInfo(gpr_lr_arm64, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());

  // CFA=fp+16 => fp=[CFA-16] lr=[CFA-8]
  row = unwind_plan.GetRowForFunctionOffset(8);
  EXPECT_EQ(8, row->GetOffset());
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_fp_arm64);
  EXPECT_TRUE(row->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row->GetCFAValue().GetOffset());

  EXPECT_TRUE(row->GetRegisterInfo(gpr_fp_arm64, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-16, regloc.GetOffset());

  EXPECT_TRUE(row->GetRegisterInfo(gpr_lr_arm64, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());

  // CFA=sp+16 => fp=[CFA-16] lr=[CFA-8]
  row = unwind_plan.GetRowForFunctionOffset(16);
  EXPECT_EQ(16, row->GetOffset());
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_sp_arm64);
  EXPECT_TRUE(row->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row->GetCFAValue().GetOffset());

  EXPECT_TRUE(row->GetRegisterInfo(gpr_fp_arm64, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-16, regloc.GetOffset());

  EXPECT_TRUE(row->GetRegisterInfo(gpr_lr_arm64, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());

  // CFA=sp +0 => fp= <same> lr= <same>
  row = unwind_plan.GetRowForFunctionOffset(20);
  EXPECT_EQ(20, row->GetOffset());
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_sp_arm64);
  EXPECT_TRUE(row->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row->GetCFAValue().GetOffset());

  EXPECT_TRUE(row->GetRegisterInfo(gpr_fp_arm64, regloc));
  EXPECT_TRUE(regloc.IsSame());

  EXPECT_TRUE(row->GetRegisterInfo(gpr_lr_arm64, regloc));
  EXPECT_TRUE(regloc.IsSame());
}

TEST_F(TestArm64InstEmulation, TestMediumDarwinFunction) {
  ArchSpec arch("arm64-apple-ios10");
  std::unique_ptr<UnwindAssemblyInstEmulation> engine(
      static_cast<UnwindAssemblyInstEmulation *>(
          UnwindAssemblyInstEmulation::CreateInstance(arch)));
  ASSERT_NE(nullptr, engine);

  const UnwindPlan::Row *row;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  UnwindPlan::Row::AbstractRegisterLocation regloc;

  // disassembly of -[NSPlaceholderString initWithBytes:length:encoding:]
  // from Foundation for iOS.
  uint8_t data[] = {
      0xf6, 0x57, 0xbd, 0xa9, // 0:  0xa9bd57f6 stp x22, x21, [sp, #-48]!
      0xf4, 0x4f, 0x01, 0xa9, // 4:  0xa9014ff4 stp x20, x19, [sp, #16]
      0xfd, 0x7b, 0x02, 0xa9, // 8:  0xa9027bfd stp x29, x30, [sp, #32]
      0xfd, 0x83, 0x00, 0x91, // 12: 0x910083fd add x29, sp, #32
      0xff, 0x43, 0x00, 0xd1, // 16: 0xd10043ff sub sp, sp, #16

      // [... function body ...]
      0x1f, 0x20, 0x03, 0xd5, // 20: 0xd503201f nop

      0xbf, 0x83, 0x00, 0xd1, // 24: 0xd10083bf sub sp, x29, #32
      0xfd, 0x7b, 0x42, 0xa9, // 28: 0xa9427bfd ldp x29, x30, [sp, #32]
      0xf4, 0x4f, 0x41, 0xa9, // 32: 0xa9414ff4 ldp x20, x19, [sp, #16]
      0xf6, 0x57, 0xc3, 0xa8, // 36: 0xa8c357f6 ldp x22, x21, [sp], #48
      0x01, 0x16, 0x09, 0x14, // 40: 0x14091601 b   0x18f640524 ; symbol stub
                              // for: CFStringCreateWithBytes
  };

  // UnwindPlan we expect:
  //  0: CFA=sp +0 =>
  //  4: CFA=sp+48 => x21=[CFA-40] x22=[CFA-48]
  //  8: CFA=sp+48 => x19=[CFA-24] x20=[CFA-32] x21=[CFA-40] x22=[CFA-48]
  // 12: CFA=sp+48 => x19=[CFA-24] x20=[CFA-32] x21=[CFA-40] x22=[CFA-48]
  // fp=[CFA-16] lr=[CFA-8]
  // 16: CFA=fp+16 => x19=[CFA-24] x20=[CFA-32] x21=[CFA-40] x22=[CFA-48]
  // fp=[CFA-16] lr=[CFA-8]

  // [... function body ...]

  // 28: CFA=sp+48 => x19=[CFA-24] x20=[CFA-32] x21=[CFA-40] x22=[CFA-48]
  // fp=[CFA-16] lr=[CFA-8]
  // 32: CFA=sp+48 => x19=[CFA-24] x20=[CFA-32] x21=[CFA-40] x22=[CFA-48] fp=
  // <same> lr= <same>
  // 36: CFA=sp+48 => x19= <same> x20= <same> x21=[CFA-40] x22=[CFA-48] fp=
  // <same> lr= <same>
  // 40: CFA=sp +0 => x19= <same> x20= <same> x21= <same> x22= <same> fp= <same>
  // lr= <same>

  sample_range = AddressRange(0x1000, sizeof(data));

  EXPECT_TRUE(engine->GetNonCallSiteUnwindPlanFromAssembly(
      sample_range, data, sizeof(data), /*target=*/nullptr, unwind_plan));

  // 0: CFA=sp +0 =>
  row = unwind_plan.GetRowForFunctionOffset(0);
  EXPECT_EQ(0, row->GetOffset());
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_sp_arm64);
  EXPECT_TRUE(row->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row->GetCFAValue().GetOffset());

  // 4: CFA=sp+48 => x21=[CFA-40] x22=[CFA-48]
  row = unwind_plan.GetRowForFunctionOffset(4);
  EXPECT_EQ(4, row->GetOffset());
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_sp_arm64);
  EXPECT_EQ(48, row->GetCFAValue().GetOffset());

  EXPECT_TRUE(row->GetRegisterInfo(gpr_x21_arm64, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-40, regloc.GetOffset());

  EXPECT_TRUE(row->GetRegisterInfo(gpr_x22_arm64, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-48, regloc.GetOffset());

  // 8: CFA=sp+48 => x19=[CFA-24] x20=[CFA-32] x21=[CFA-40] x22=[CFA-48]
  row = unwind_plan.GetRowForFunctionOffset(8);
  EXPECT_EQ(8, row->GetOffset());
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_sp_arm64);
  EXPECT_EQ(48, row->GetCFAValue().GetOffset());

  EXPECT_TRUE(row->GetRegisterInfo(gpr_x19_arm64, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-24, regloc.GetOffset());

  EXPECT_TRUE(row->GetRegisterInfo(gpr_x20_arm64, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-32, regloc.GetOffset());

  // 12: CFA=sp+48 => x19=[CFA-24] x20=[CFA-32] x21=[CFA-40] x22=[CFA-48]
  // fp=[CFA-16] lr=[CFA-8]
  row = unwind_plan.GetRowForFunctionOffset(12);
  EXPECT_EQ(12, row->GetOffset());
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_sp_arm64);
  EXPECT_EQ(48, row->GetCFAValue().GetOffset());

  EXPECT_TRUE(row->GetRegisterInfo(gpr_fp_arm64, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-16, regloc.GetOffset());

  EXPECT_TRUE(row->GetRegisterInfo(gpr_lr_arm64, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());

  // 16: CFA=fp+16 => x19=[CFA-24] x20=[CFA-32] x21=[CFA-40] x22=[CFA-48]
  // fp=[CFA-16] lr=[CFA-8]
  row = unwind_plan.GetRowForFunctionOffset(16);
  EXPECT_EQ(16, row->GetOffset());
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_fp_arm64);
  EXPECT_TRUE(row->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row->GetCFAValue().GetOffset());

  // 28: CFA=sp+48 => x19=[CFA-24] x20=[CFA-32] x21=[CFA-40] x22=[CFA-48]
  // fp=[CFA-16] lr=[CFA-8]
  row = unwind_plan.GetRowForFunctionOffset(28);
  EXPECT_EQ(28, row->GetOffset());
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_sp_arm64);
  EXPECT_TRUE(row->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(48, row->GetCFAValue().GetOffset());

  // 32: CFA=sp+48 => x19=[CFA-24] x20=[CFA-32] x21=[CFA-40] x22=[CFA-48] fp=
  // <same> lr= <same>
  row = unwind_plan.GetRowForFunctionOffset(32);
  EXPECT_EQ(32, row->GetOffset());

  // I'd prefer if these restored registers were cleared entirely instead of set
  // to IsSame...
  EXPECT_TRUE(row->GetRegisterInfo(gpr_fp_arm64, regloc));
  EXPECT_TRUE(regloc.IsSame());

  EXPECT_TRUE(row->GetRegisterInfo(gpr_lr_arm64, regloc));
  EXPECT_TRUE(regloc.IsSame());

  // 36: CFA=sp+48 => x19= <same> x20= <same> x21=[CFA-40] x22=[CFA-48] fp=
  // <same> lr= <same>
  row = unwind_plan.GetRowForFunctionOffset(36);
  EXPECT_EQ(36, row->GetOffset());

  EXPECT_TRUE(row->GetRegisterInfo(gpr_x19_arm64, regloc));
  EXPECT_TRUE(regloc.IsSame());

  EXPECT_TRUE(row->GetRegisterInfo(gpr_x20_arm64, regloc));
  EXPECT_TRUE(regloc.IsSame());

  // 40: CFA=sp +0 => x19= <same> x20= <same> x21= <same> x22= <same> fp= <same>
  // lr= <same>
  row = unwind_plan.GetRowForFunctionOffset(40);
  EXPECT_EQ(40, row->GetOffset());
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_sp_arm64);
  EXPECT_TRUE(row->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row->GetCFAValue().GetOffset());

  EXPECT_TRUE(row->GetRegisterInfo(gpr_x21_arm64, regloc));
  EXPECT_TRUE(regloc.IsSame());

  EXPECT_TRUE(row->GetRegisterInfo(gpr_x22_arm64, regloc));
  EXPECT_TRUE(regloc.IsSame());
}

TEST_F(TestArm64InstEmulation, TestFramelessThreeEpilogueFunction) {
  ArchSpec arch("arm64-apple-ios10");
  std::unique_ptr<UnwindAssemblyInstEmulation> engine(
      static_cast<UnwindAssemblyInstEmulation *>(
          UnwindAssemblyInstEmulation::CreateInstance(arch)));
  ASSERT_NE(nullptr, engine);

  const UnwindPlan::Row *row;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  UnwindPlan::Row::AbstractRegisterLocation regloc;

  // disassembly of JSC::ARM64LogicalImmediate::findBitRange<16u>
  // from JavaScriptcore for iOS.
  uint8_t data[] = {
      0x08, 0x3c, 0x0f, 0x53, //  0: 0x530f3c08 ubfx   w8, w0, #15, #1
      0x68, 0x00, 0x00, 0x39, //  4: 0x39000068 strb   w8, [x3]
      0x08, 0x3c, 0x40, 0xd2, //  8: 0xd2403c08 eor    x8, x0, #0xffff
      0x1f, 0x00, 0x71, 0xf2, // 12: 0xf271001f tst    x0, #0x8000

      // [...]

      0x3f, 0x01, 0x0c, 0xeb, // 16: 0xeb0c013f cmp    x9, x12
      0x81, 0x00, 0x00, 0x54, // 20: 0x54000081 b.ne +34
      0x5f, 0x00, 0x00, 0xb9, // 24: 0xb900005f str    wzr, [x2]
      0xe0, 0x03, 0x00, 0x32, // 28: 0x320003e0 orr    w0, wzr, #0x1
      0xc0, 0x03, 0x5f, 0xd6, // 32: 0xd65f03c0 ret
      0x89, 0x01, 0x09, 0xca, // 36: 0xca090189 eor    x9, x12, x9

      // [...]

      0x08, 0x05, 0x00, 0x11, // 40: 0x11000508 add    w8, w8, #0x1
      0x48, 0x00, 0x00, 0xb9, // 44: 0xb9000048 str    w8, [x2]
      0xe0, 0x03, 0x00, 0x32, // 48: 0x320003e0 orr    w0, wzr, #0x1
      0xc0, 0x03, 0x5f, 0xd6, // 52: 0xd65f03c0 ret
      0x00, 0x00, 0x80, 0x52, // 56: 0x52800000 mov    w0, #0x0
      0xc0, 0x03, 0x5f, 0xd6, // 60: 0xd65f03c0 ret

  };

  // UnwindPlan we expect:
  //  0: CFA=sp +0 =>
  // (possibly with additional rows at offsets 36 and 56 saying the same thing)

  sample_range = AddressRange(0x1000, sizeof(data));

  EXPECT_TRUE(engine->GetNonCallSiteUnwindPlanFromAssembly(
      sample_range, data, sizeof(data), /*target=*/nullptr, unwind_plan));

  // 0: CFA=sp +0 =>
  row = unwind_plan.GetRowForFunctionOffset(0);
  EXPECT_EQ(0, row->GetOffset());
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_sp_arm64);
  EXPECT_TRUE(row->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row->GetCFAValue().GetOffset());

  row = unwind_plan.GetRowForFunctionOffset(32);
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_sp_arm64);
  EXPECT_TRUE(row->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row->GetCFAValue().GetOffset());

  EXPECT_FALSE(row->GetRegisterInfo(gpr_x19_arm64, regloc));
  EXPECT_FALSE(row->GetRegisterInfo(gpr_x20_arm64, regloc));
  EXPECT_FALSE(row->GetRegisterInfo(gpr_x21_arm64, regloc));
  EXPECT_FALSE(row->GetRegisterInfo(gpr_x22_arm64, regloc));
  EXPECT_FALSE(row->GetRegisterInfo(gpr_x23_arm64, regloc));
  EXPECT_FALSE(row->GetRegisterInfo(gpr_x24_arm64, regloc));
  EXPECT_FALSE(row->GetRegisterInfo(gpr_x25_arm64, regloc));
  EXPECT_FALSE(row->GetRegisterInfo(gpr_x26_arm64, regloc));
  EXPECT_FALSE(row->GetRegisterInfo(gpr_x27_arm64, regloc));
  EXPECT_FALSE(row->GetRegisterInfo(gpr_x28_arm64, regloc));

  EXPECT_TRUE(row->GetRegisterInfo(gpr_fp_arm64, regloc));
  EXPECT_TRUE(regloc.IsSame());

  EXPECT_TRUE(row->GetRegisterInfo(gpr_lr_arm64, regloc));
  EXPECT_TRUE(regloc.IsSame());

  row = unwind_plan.GetRowForFunctionOffset(36);
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_sp_arm64);
  EXPECT_TRUE(row->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row->GetCFAValue().GetOffset());

  row = unwind_plan.GetRowForFunctionOffset(52);
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_sp_arm64);
  EXPECT_TRUE(row->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row->GetCFAValue().GetOffset());

  row = unwind_plan.GetRowForFunctionOffset(56);
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_sp_arm64);
  EXPECT_TRUE(row->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row->GetCFAValue().GetOffset());

  row = unwind_plan.GetRowForFunctionOffset(60);
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_sp_arm64);
  EXPECT_TRUE(row->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row->GetCFAValue().GetOffset());
}

TEST_F(TestArm64InstEmulation, TestRegisterSavedTwice) {
  ArchSpec arch("arm64-apple-ios10");
  std::unique_ptr<UnwindAssemblyInstEmulation> engine(
      static_cast<UnwindAssemblyInstEmulation *>(
          UnwindAssemblyInstEmulation::CreateInstance(arch)));
  ASSERT_NE(nullptr, engine);

  const UnwindPlan::Row *row;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  UnwindPlan::Row::AbstractRegisterLocation regloc;

  // disassembly of mach_msg_sever_once from libsystem_kernel.dylib for iOS.
  uint8_t data[] = {

      0xfc, 0x6f, 0xba, 0xa9, //  0: 0xa9ba6ffc stp  x28, x27, [sp, #-0x60]!
      0xfa, 0x67, 0x01, 0xa9, //  4: 0xa90167fa stp  x26, x25, [sp, #0x10]
      0xf8, 0x5f, 0x02, 0xa9, //  8: 0xa9025ff8 stp  x24, x23, [sp, #0x20]
      0xf6, 0x57, 0x03, 0xa9, // 12: 0xa90357f6 stp  x22, x21, [sp, #0x30]
      0xf4, 0x4f, 0x04, 0xa9, // 16: 0xa9044ff4 stp  x20, x19, [sp, #0x40]
      0xfd, 0x7b, 0x05, 0xa9, // 20: 0xa9057bfd stp  x29, x30, [sp, #0x50]
      0xfd, 0x43, 0x01, 0x91, // 24: 0x910143fd add  x29, sp, #0x50
      0xff, 0xc3, 0x00, 0xd1, // 28: 0xd100c3ff sub  sp, sp, #0x30

      // mid-function, store x20 & x24 on the stack at a different location.
      // this should not show up in the unwind plan; caller's values are not
      // being saved to stack.
      0xf8, 0x53, 0x01, 0xa9, // 32: 0xa90153f8 stp    x24, x20, [sp, #0x10]

      // mid-function, copy x20 and x19 off of the stack -- but not from
      // their original locations.  unwind plan should ignore this.
      0xf4, 0x4f, 0x41, 0xa9, // 36: 0xa9414ff4 ldp  x20, x19, [sp, #0x10]

      // epilogue
      0xbf, 0x43, 0x01, 0xd1, // 40: 0xd10143bf sub  sp, x29, #0x50
      0xfd, 0x7b, 0x45, 0xa9, // 44: 0xa9457bfd ldp  x29, x30, [sp, #0x50]
      0xf4, 0x4f, 0x44, 0xa9, // 48: 0xa9444ff4 ldp  x20, x19, [sp, #0x40]
      0xf6, 0x57, 0x43, 0xa9, // 52: 0xa94357f6 ldp  x22, x21, [sp, #0x30]
      0xf8, 0x5f, 0x42, 0xa9, // 56: 0xa9425ff8 ldp  x24, x23, [sp, #0x20]
      0xfa, 0x67, 0x41, 0xa9, // 60: 0xa94167fa ldp  x26, x25, [sp, #0x10]
      0xfc, 0x6f, 0xc6, 0xa8, // 64: 0xa8c66ffc ldp  x28, x27, [sp], #0x60
      0xc0, 0x03, 0x5f, 0xd6, // 68: 0xd65f03c0 ret
  };

  // UnwindPlan we expect:
  //   0: CFA=sp +0 =>
  //   4: CFA=sp+96 => x27=[CFA-88] x28=[CFA-96]
  //   8: CFA=sp+96 => x25=[CFA-72] x26=[CFA-80] x27=[CFA-88] x28=[CFA-96]
  //  12: CFA=sp+96 => x23=[CFA-56] x24=[CFA-64] x25=[CFA-72] x26=[CFA-80]
  //  x27=[CFA-88] x28=[CFA-96]
  //  16: CFA=sp+96 => x21=[CFA-40] x22=[CFA-48] x23=[CFA-56] x24=[CFA-64]
  //  x25=[CFA-72] x26=[CFA-80] x27=[CFA-88] x28=[CFA-96]
  //  20: CFA=sp+96 => x19=[CFA-24] x20=[CFA-32] x21=[CFA-40] x22=[CFA-48]
  //  x23=[CFA-56] x24=[CFA-64] x25=[CFA-72] x26=[CFA-80] x27=[CFA-88]
  //  x28=[CFA-96]
  //  24: CFA=sp+96 => x19=[CFA-24] x20=[CFA-32] x21=[CFA-40] x22=[CFA-48]
  //  x23=[CFA-56] x24=[CFA-64] x25=[CFA-72] x26=[CFA-80] x27=[CFA-88]
  //  x28=[CFA-96] fp=[CFA-16] lr=[CFA-8]
  //  28: CFA=fp+16 => x19=[CFA-24] x20=[CFA-32] x21=[CFA-40] x22=[CFA-48]
  //  x23=[CFA-56] x24=[CFA-64] x25=[CFA-72] x26=[CFA-80] x27=[CFA-88]
  //  x28=[CFA-96] fp=[CFA-16] lr=[CFA-8]

  //  44: CFA=sp+96 => x19=[CFA-24] x20=[CFA-32] x21=[CFA-40] x22=[CFA-48]
  //  x23=[CFA-56] x24=[CFA-64] x25=[CFA-72] x26=[CFA-80] x27=[CFA-88]
  //  x28=[CFA-96] fp=[CFA-16] lr=[CFA-8]
  //  48: CFA=sp+96 => x19=[CFA-24] x20=[CFA-32] x21=[CFA-40] x22=[CFA-48]
  //  x23=[CFA-56] x24=[CFA-64] x25=[CFA-72] x26=[CFA-80] x27=[CFA-88]
  //  x28=[CFA-96]
  //  52: CFA=sp+96 => x21=[CFA-40] x22=[CFA-48] x23=[CFA-56] x24=[CFA-64]
  //  x25=[CFA-72] x26=[CFA-80] x27=[CFA-88] x28=[CFA-96]
  //  56: CFA=sp+96 => x23=[CFA-56] x24=[CFA-64] x25=[CFA-72] x26=[CFA-80]
  //  x27=[CFA-88] x28=[CFA-96]
  //  60: CFA=sp+96 =>  x25=[CFA-72] x26=[CFA-80] x27=[CFA-88] x28=[CFA-96]
  //  64: CFA=sp+96 =>  x27=[CFA-88] x28=[CFA-96]
  //  68: CFA=sp +0 =>

  sample_range = AddressRange(0x1000, sizeof(data));

  EXPECT_TRUE(engine->GetNonCallSiteUnwindPlanFromAssembly(
      sample_range, data, sizeof(data), /*target=*/nullptr, unwind_plan));

  row = unwind_plan.GetRowForFunctionOffset(36);
  EXPECT_EQ(28, row->GetOffset());
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_fp_arm64);
  EXPECT_TRUE(row->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row->GetCFAValue().GetOffset());

  EXPECT_TRUE(row->GetRegisterInfo(gpr_x20_arm64, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-32, regloc.GetOffset());

  row = unwind_plan.GetRowForFunctionOffset(40);
  EXPECT_EQ(28, row->GetOffset());
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_fp_arm64);
  EXPECT_TRUE(row->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row->GetCFAValue().GetOffset());

  EXPECT_TRUE(row->GetRegisterInfo(gpr_x20_arm64, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-32, regloc.GetOffset());
}

TEST_F(TestArm64InstEmulation, TestRegisterDoubleSpills) {
  ArchSpec arch("arm64-apple-ios10");
  std::unique_ptr<UnwindAssemblyInstEmulation> engine(
      static_cast<UnwindAssemblyInstEmulation *>(
          UnwindAssemblyInstEmulation::CreateInstance(arch)));
  ASSERT_NE(nullptr, engine);

  const UnwindPlan::Row *row;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  UnwindPlan::Row::AbstractRegisterLocation regloc;

  // this file built with clang for iOS arch arm64 optimization -Os
  // #include <stdio.h>
  // double foo(double in) {
  // double arr[32];
  // for (int i = 0; i < 32; i++)
  //   arr[i] = in + i;
  // for (int i = 2; i < 30; i++)
  //   arr[i] = ((((arr[i - 1] * arr[i - 2] * 0.2) + (0.7 * arr[i])) /
  //   ((((arr[i] * 0.73) + 0.65) * (arr[i - 1] + 0.2)) - ((arr[i + 1] + (arr[i]
  //   * 0.32) + 0.52) / 0.3) + (0.531 * arr[i - 2]))) + ((arr[i - 1] + 5) /
  //   ((arr[i + 2] + 0.4) / arr[i])) + (arr[5] * (0.17 + arr[7] * arr[i])) +
  //   ((i > 5 ? (arr[i - 3]) : arr[i - 1]) * 0.263) + (((arr[i - 2] + arr[i -
  //   1]) * 0.3252) + 3.56) - (arr[i + 1] * 0.852311)) * ((arr[i] * 85234.1345)
  //   + (77342.451324 / (arr[i - 2] + arr[i - 1] - 73425341.33455))) + (arr[i]
  //   * 875712013.55) - (arr[i - 1] * 0.5555) - ((arr[i] * (arr[i + 1] +
  //   17342834.44) / 8688200123.555)) + (arr[i - 2] + 8888.888);
  // return arr[16];
  //}
  // int main(int argc, char **argv) { printf("%g\n", foo(argc)); }

  // so function foo() uses enough registers that it spills the callee-saved
  // floating point registers.
  uint8_t data[] = {
      // prologue
      0xef, 0x3b, 0xba, 0x6d, //  0: 0x6dba3bef   stp    d15, d14, [sp, #-0x60]!
      0xed, 0x33, 0x01, 0x6d, //  4: 0x6d0133ed   stp    d13, d12, [sp, #0x10]
      0xeb, 0x2b, 0x02, 0x6d, //  8: 0x6d022beb   stp    d11, d10, [sp, #0x20]
      0xe9, 0x23, 0x03, 0x6d, // 12: 0x6d0323e9   stp    d9, d8, [sp, #0x30]
      0xfc, 0x6f, 0x04, 0xa9, // 16: 0xa9046ffc   stp    x28, x27, [sp, #0x40]
      0xfd, 0x7b, 0x05, 0xa9, // 20: 0xa9057bfd   stp    x29, x30, [sp, #0x50]
      0xfd, 0x43, 0x01, 0x91, // 24: 0x910143fd   add    x29, sp, #0x50
      0xff, 0x43, 0x04, 0xd1, // 28: 0xd10443ff   sub    sp, sp, #0x110

      // epilogue
      0xbf, 0x43, 0x01, 0xd1, // 32: 0xd10143bf   sub    sp, x29, #0x50
      0xfd, 0x7b, 0x45, 0xa9, // 36: 0xa9457bfd   ldp    x29, x30, [sp, #0x50]
      0xfc, 0x6f, 0x44, 0xa9, // 40: 0xa9446ffc   ldp    x28, x27, [sp, #0x40]
      0xe9, 0x23, 0x43, 0x6d, // 44: 0x6d4323e9   ldp    d9, d8, [sp, #0x30]
      0xeb, 0x2b, 0x42, 0x6d, // 48: 0x6d422beb   ldp    d11, d10, [sp, #0x20]
      0xed, 0x33, 0x41, 0x6d, // 52: 0x6d4133ed   ldp    d13, d12, [sp, #0x10]
      0xef, 0x3b, 0xc6, 0x6c, // 56: 0x6cc63bef   ldp    d15, d14, [sp], #0x60
      0xc0, 0x03, 0x5f, 0xd6, // 60: 0xd65f03c0   ret
  };

  // UnwindPlan we expect:
  //   0: CFA=sp +0 =>
  //   4: CFA=sp+96 => d14=[CFA-88] d15=[CFA-96]
  //   8: CFA=sp+96 => d12=[CFA-72] d13=[CFA-80] d14=[CFA-88] d15=[CFA-96]
  //  12: CFA=sp+96 => d10=[CFA-56] d11=[CFA-64] d12=[CFA-72] d13=[CFA-80]
  //  d14=[CFA-88] d15=[CFA-96]
  //  16: CFA=sp+96 => d8=[CFA-40] d9=[CFA-48] d10=[CFA-56] d11=[CFA-64]
  //  d12=[CFA-72] d13=[CFA-80] d14=[CFA-88] d15=[CFA-96]
  //  20: CFA=sp+96 => x27=[CFA-24] x28=[CFA-32] d8=[CFA-40] d9=[CFA-48]
  //  d10=[CFA-56] d11=[CFA-64] d12=[CFA-72] d13=[CFA-80] d14=[CFA-88]
  //  d15=[CFA-96]
  //  24: CFA=sp+96 => x27=[CFA-24] x28=[CFA-32] fp=[CFA-16] lr=[CFA-8]
  //  d8=[CFA-40] d9=[CFA-48] d10=[CFA-56] d11=[CFA-64] d12=[CFA-72]
  //  d13=[CFA-80] d14=[CFA-88] d15=[CFA-96]
  //  28: CFA=fp+16 => x27=[CFA-24] x28=[CFA-32] fp=[CFA-16] lr=[CFA-8]
  //  d8=[CFA-40] d9=[CFA-48] d10=[CFA-56] d11=[CFA-64] d12=[CFA-72]
  //  d13=[CFA-80] d14=[CFA-88] d15=[CFA-96]
  //  36: CFA=sp+96 => x27=[CFA-24] x28=[CFA-32] fp=[CFA-16] lr=[CFA-8]
  //  d8=[CFA-40] d9=[CFA-48] d10=[CFA-56] d11=[CFA-64] d12=[CFA-72]
  //  d13=[CFA-80] d14=[CFA-88] d15=[CFA-96]
  //  40: CFA=sp+96 => x27=[CFA-24] x28=[CFA-32] d8=[CFA-40] d9=[CFA-48]
  //  d10=[CFA-56] d11=[CFA-64] d12=[CFA-72] d13=[CFA-80] d14=[CFA-88]
  //  d15=[CFA-96]
  //  44: CFA=sp+96 => d8=[CFA-40] d9=[CFA-48] d10=[CFA-56] d11=[CFA-64]
  //  d12=[CFA-72] d13=[CFA-80] d14=[CFA-88] d15=[CFA-96]
  //  48: CFA=sp+96 => d10=[CFA-56] d11=[CFA-64] d12=[CFA-72] d13=[CFA-80]
  //  d14=[CFA-88] d15=[CFA-96]
  //  52: CFA=sp+96 => d12=[CFA-72] d13=[CFA-80] d14=[CFA-88] d15=[CFA-96]
  //  56: CFA=sp+96 => d14=[CFA-88] d15=[CFA-96]
  //  60: CFA=sp +0 =>

  sample_range = AddressRange(0x1000, sizeof(data));

  EXPECT_TRUE(engine->GetNonCallSiteUnwindPlanFromAssembly(
      sample_range, data, sizeof(data), /*target=*/nullptr, unwind_plan));

  //  28: CFA=fp+16 => x27=[CFA-24] x28=[CFA-32] fp=[CFA-16] lr=[CFA-8]
  //  d8=[CFA-40] d9=[CFA-48] d10=[CFA-56] d11=[CFA-64] d12=[CFA-72]
  //  d13=[CFA-80] d14=[CFA-88] d15=[CFA-96]
  row = unwind_plan.GetRowForFunctionOffset(28);
  EXPECT_EQ(28, row->GetOffset());
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_fp_arm64);
  EXPECT_TRUE(row->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row->GetCFAValue().GetOffset());

  EXPECT_TRUE(row->GetRegisterInfo(fpu_d15_arm64, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-96, regloc.GetOffset());

  EXPECT_TRUE(row->GetRegisterInfo(fpu_d14_arm64, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-88, regloc.GetOffset());

  EXPECT_TRUE(row->GetRegisterInfo(fpu_d13_arm64, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-80, regloc.GetOffset());

  EXPECT_TRUE(row->GetRegisterInfo(fpu_d12_arm64, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-72, regloc.GetOffset());

  EXPECT_TRUE(row->GetRegisterInfo(fpu_d11_arm64, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-64, regloc.GetOffset());

  EXPECT_TRUE(row->GetRegisterInfo(fpu_d10_arm64, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-56, regloc.GetOffset());

  EXPECT_TRUE(row->GetRegisterInfo(fpu_d9_arm64, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-48, regloc.GetOffset());

  EXPECT_TRUE(row->GetRegisterInfo(fpu_d8_arm64, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-40, regloc.GetOffset());

  //  60: CFA=sp +0 =>
  row = unwind_plan.GetRowForFunctionOffset(60);
  EXPECT_EQ(60, row->GetOffset());
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_sp_arm64);
  EXPECT_TRUE(row->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row->GetCFAValue().GetOffset());

  if (row->GetRegisterInfo(fpu_d8_arm64, regloc)) {
    EXPECT_TRUE(regloc.IsSame());
  }
  if (row->GetRegisterInfo(fpu_d9_arm64, regloc)) {
    EXPECT_TRUE(regloc.IsSame());
  }
  if (row->GetRegisterInfo(fpu_d10_arm64, regloc)) {
    EXPECT_TRUE(regloc.IsSame());
  }
  if (row->GetRegisterInfo(fpu_d11_arm64, regloc)) {
    EXPECT_TRUE(regloc.IsSame());
  }
  if (row->GetRegisterInfo(fpu_d12_arm64, regloc)) {
    EXPECT_TRUE(regloc.IsSame());
  }
  if (row->GetRegisterInfo(fpu_d13_arm64, regloc)) {
    EXPECT_TRUE(regloc.IsSame());
  }
  if (row->GetRegisterInfo(fpu_d14_arm64, regloc)) {
    EXPECT_TRUE(regloc.IsSame());
  }
  if (row->GetRegisterInfo(fpu_d15_arm64, regloc)) {
    EXPECT_TRUE(regloc.IsSame());
  }
  if (row->GetRegisterInfo(gpr_x27_arm64, regloc)) {
    EXPECT_TRUE(regloc.IsSame());
  }
  if (row->GetRegisterInfo(gpr_x28_arm64, regloc)) {
    EXPECT_TRUE(regloc.IsSame());
  }
}

TEST_F(TestArm64InstEmulation, TestCFARegisterTrackedAcrossJumps) {
  ArchSpec arch("arm64-apple-ios10");
  std::unique_ptr<UnwindAssemblyInstEmulation> engine(
      static_cast<UnwindAssemblyInstEmulation *>(
          UnwindAssemblyInstEmulation::CreateInstance(arch)));
  ASSERT_NE(nullptr, engine);

  const UnwindPlan::Row *row;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  UnwindPlan::Row::AbstractRegisterLocation regloc;

  uint8_t data[] = {
      // prologue
      0xf4, 0x4f, 0xbe, 0xa9, //  0: 0xa9be4ff4 stp x20, x19, [sp, #-0x20]!
      0xfd, 0x7b, 0x01, 0xa9, //  4: 0xa9017bfd stp x29, x30, [sp, #0x10]
      0xfd, 0x43, 0x00, 0x91, //  8: 0x910043fd add x29, sp, #0x10
      0xff, 0x43, 0x00, 0xd1, // 12: 0xd10043ff sub sp, sp, #0x10
      // conditional branch over a mid-function epilogue
      0xeb, 0x00, 0x00, 0x54, // 16: 0x540000eb b.lt <+44>
      // mid-function epilogue
      0x1f, 0x20, 0x03, 0xd5, // 20: 0xd503201f   nop
      0xe0, 0x03, 0x13, 0xaa, // 24: 0xaa1303e0   mov    x0, x19
      0xbf, 0x43, 0x00, 0xd1, // 28: 0xd10043bf   sub    sp, x29, #0x10
      0xfd, 0x7b, 0x41, 0xa9, // 32: 0xa9417bfd   ldp    x29, x30, [sp, #0x10]
      0xf4, 0x4f, 0xc2, 0xa8, // 36: 0xa8c24ff4   ldp    x20, x19, [sp], #0x20
      0xc0, 0x03, 0x5f, 0xd6, // 40: 0xd65f03c0   ret
      // unwind state restored, we're using a frame pointer, let's change the
      // stack pointer and see no change in how the CFA is computed
      0x1f, 0x20, 0x03, 0xd5, // 44: 0xd503201f   nop
      0xff, 0x43, 0x00, 0xd1, // 48: 0xd10043ff   sub    sp, sp, #0x10
      0x1f, 0x20, 0x03, 0xd5, // 52: 0xd503201f   nop
      // final epilogue
      0xe0, 0x03, 0x13, 0xaa, // 56: 0xaa1303e0   mov    x0, x19
      0xbf, 0x43, 0x00, 0xd1, // 60: 0xd10043bf   sub    sp, x29, #0x10
      0xfd, 0x7b, 0x41, 0xa9, // 64: 0xa9417bfd   ldp    x29, x30, [sp, #0x10]
      0xf4, 0x4f, 0xc2, 0xa8, // 68: 0xa8c24ff4   ldp    x20, x19, [sp], #0x20
      0xc0, 0x03, 0x5f, 0xd6, // 72: 0xd65f03c0   ret

      0x1f, 0x20, 0x03, 0xd5, // 52: 0xd503201f   nop
  };

  // UnwindPlan we expect:
  // row[0]:    0: CFA=sp +0 =>
  // row[1]:    4: CFA=sp+32 => x19=[CFA-24] x20=[CFA-32]
  // row[2]:    8: CFA=sp+32 => x19=[CFA-24] x20=[CFA-32] fp=[CFA-16] lr=[CFA-8]
  // row[3]:   12: CFA=fp+16 => x19=[CFA-24] x20=[CFA-32] fp=[CFA-16] lr=[CFA-8]
  // row[4]:   32: CFA=sp+32 => x19=[CFA-24] x20=[CFA-32] fp=[CFA-16] lr=[CFA-8]
  // row[5]:   36: CFA=sp+32 => x19=[CFA-24] x20=[CFA-32] fp= <same> lr= <same>
  // row[6]:   40: CFA=sp +0 => x19= <same> x20= <same> fp= <same> lr= <same> 
  // row[7]:   44: CFA=fp+16 => x19=[CFA-24] x20=[CFA-32] fp=[CFA-16] lr=[CFA-8] 
  // row[8]:   64: CFA=sp+32 => x19=[CFA-24] x20=[CFA-32] fp=[CFA-16] lr=[CFA-8] 
  // row[9]:   68: CFA=sp+32 => x19=[CFA-24] x20=[CFA-32] fp= <same> lr= <same> 
  // row[10]:  72: CFA=sp +0 => x19= <same> x20= <same> fp= <same> lr= <same> 

  // The specific bug we're looking for is this incorrect CFA definition, 
  // where the InstEmulation is using the $sp value mixed in with $fp, 
  // it looks like this:
  //
  // row[7]:   44: CFA=fp+16 => x19=[CFA-24] x20=[CFA-32] fp=[CFA-16] lr=[CFA-8]
  // row[8]:   52: CFA=fp+64 => x19=[CFA-24] x20=[CFA-32] fp=[CFA-16] lr=[CFA-8]
  // row[9]:   68: CFA=fp+64 => x19=[CFA-24] x20=[CFA-32] fp= <same> lr= <same>
 
  sample_range = AddressRange(0x1000, sizeof(data));

  EXPECT_TRUE(engine->GetNonCallSiteUnwindPlanFromAssembly(
      sample_range, data, sizeof(data), /*target=*/nullptr, unwind_plan));

  // Confirm CFA at mid-func epilogue 'ret' is $sp+0
  row = unwind_plan.GetRowForFunctionOffset(40);
  EXPECT_EQ(40, row->GetOffset());
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_sp_arm64);
  EXPECT_TRUE(row->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row->GetCFAValue().GetOffset());

  // After the 'ret', confirm we're back to the correct CFA of $fp+16
  row = unwind_plan.GetRowForFunctionOffset(44);
  EXPECT_EQ(44, row->GetOffset());
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_fp_arm64);
  EXPECT_TRUE(row->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row->GetCFAValue().GetOffset());

  // Confirm that we have no additional UnwindPlan rows before the 
  // real epilogue -- we still get the Row at offset 44.
  row = unwind_plan.GetRowForFunctionOffset(60);
  EXPECT_EQ(44, row->GetOffset());
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_fp_arm64);
  EXPECT_TRUE(row->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row->GetCFAValue().GetOffset());

  // And in the epilogue, confirm that we start by switching back to 
  // defining the CFA in terms of $sp.
  row = unwind_plan.GetRowForFunctionOffset(64);
  EXPECT_EQ(64, row->GetOffset());
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_sp_arm64);
  EXPECT_TRUE(row->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(32, row->GetCFAValue().GetOffset());
}

TEST_F(TestArm64InstEmulation, TestCFAResetToSP) {
  ArchSpec arch("arm64-apple-ios15");
  std::unique_ptr<UnwindAssemblyInstEmulation> engine(
      static_cast<UnwindAssemblyInstEmulation *>(
          UnwindAssemblyInstEmulation::CreateInstance(arch)));
  ASSERT_NE(nullptr, engine);

  const UnwindPlan::Row *row;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  UnwindPlan::Row::AbstractRegisterLocation regloc;

  // The called_from_nodebug() from TestStepNoDebug.py
  // Most of the previous unit tests have $sp being set as
  // $fp plus an offset, and the unwinder recognizes that
  // as a CFA change.  This codegen overwrites $fp and we
  // need to know that CFA is now in terms of $sp.
  uint8_t data[] = {
      // prologue
      0xff, 0x83, 0x00, 0xd1, //  0: 0xd10083ff sub sp, sp, #0x20
      0xfd, 0x7b, 0x01, 0xa9, //  4: 0xa9017bfd stp x29, x30, [sp, #0x10]
      0xfd, 0x43, 0x00, 0x91, //  8: 0x910043fd add x29, sp, #0x10

      // epilogue
      0xfd, 0x7b, 0x41, 0xa9, // 12: 0xa9417bfd ldp x29, x30, [sp, #0x10]
      0xff, 0x83, 0x00, 0x91, // 16: 0x910083ff add sp, sp, #0x20
      0xc0, 0x03, 0x5f, 0xd6, // 20: 0xd65f03c0 ret
  };

  // UnwindPlan we expect:
  // row[0]:    0: CFA=sp +0 =>
  // row[1]:    4: CFA=sp+32 =>
  // row[2]:    8: CFA=sp+32 => fp=[CFA-16] lr=[CFA-8]
  // row[3]:   12: CFA=fp+16 => fp=[CFA-16] lr=[CFA-8]
  // row[4]:   16: CFA=sp+32 => x0= <same> fp= <same> lr= <same>
  // row[5]:   20: CFA=sp +0 => x0= <same> fp= <same> lr= <same>

  // The specific issue we're testing for is after the
  // ldp x29, x30, [sp, #0x10]
  // when $fp and $lr have been restored to the original values,
  // the CFA is now set in terms of the stack pointer.  If it is
  // left as being in terms of the frame pointer, $fp now has the
  // caller function's $fp value and our StackID will be wrong etc.

  sample_range = AddressRange(0x1000, sizeof(data));

  EXPECT_TRUE(engine->GetNonCallSiteUnwindPlanFromAssembly(
      sample_range, data, sizeof(data), /*target=*/nullptr, unwind_plan));

  // Confirm CFA before epilogue instructions is in terms of $fp
  row = unwind_plan.GetRowForFunctionOffset(12);
  EXPECT_EQ(12, row->GetOffset());
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_fp_arm64);
  EXPECT_TRUE(row->GetCFAValue().IsRegisterPlusOffset() == true);

  // Confirm that after restoring $fp to caller's value, CFA is now in
  // terms of $sp
  row = unwind_plan.GetRowForFunctionOffset(16);
  EXPECT_EQ(16, row->GetOffset());
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_sp_arm64);
  EXPECT_TRUE(row->GetCFAValue().IsRegisterPlusOffset() == true);
}

TEST_F(TestArm64InstEmulation, TestPrologueStartsWithStrD8) {
  ArchSpec arch("aarch64");
  std::unique_ptr<UnwindAssemblyInstEmulation> engine(
      static_cast<UnwindAssemblyInstEmulation *>(
          UnwindAssemblyInstEmulation::CreateInstance(arch)));
  ASSERT_NE(nullptr, engine);

  const UnwindPlan::Row *row;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  UnwindPlan::Row::AbstractRegisterLocation regloc;

  // The sample function is built with 'clang --target aarch64 -O1':
  //
  //   int bar(float x);
  //   int foo(float x) {
  //     return bar(x) + bar(x);
  //   }
  //
  // The function uses one floating point register and spills it with
  // 'str d8, [sp, #-0x20]!'.

  // clang-format off
  uint8_t data[] = {
      // prologue
      0xe8, 0x0f, 0x1e, 0xfc, //  0: fc1e0fe8    str  d8, [sp, #-0x20]!
      0xfd, 0xfb, 0x00, 0xa9, //  4: a900fbfd    stp  x29, x30, [sp, #0x8]
      0xf3, 0x0f, 0x00, 0xf9, //  8: f9000ff3    str  x19, [sp, #0x18]
      0xfd, 0x23, 0x00, 0x91, // 12: 910023fd    add  x29, sp, #0x8

      // epilogue
      0xfd, 0xfb, 0x40, 0xa9, // 16: a940fbfd    ldp  x29, x30, [sp, #0x8]
      0xf3, 0x0f, 0x40, 0xf9, // 20: f9400ff3    ldr  x19, [sp, #0x18]
      0xe8, 0x07, 0x42, 0xfc, // 24: fc4207e8    ldr  d8, [sp], #0x20
      0xc0, 0x03, 0x5f, 0xd6, // 28: d65f03c0    ret
  };
  // clang-format on

  // UnwindPlan we expect:
  //   0: CFA=sp +0 =>
  //   4: CFA=sp+32 => d8=[CFA-32]
  //   8: CFA=sp+32 => fp=[CFA-24] lr=[CFA-16] d8=[CFA-32]
  //  12: CFA=sp+32 => x19=[CFA-8] fp=[CFA-24] lr=[CFA-16] d8=[CFA-32]
  //  16: CFA=fp+24 => x19=[CFA-8] fp=[CFA-24] lr=[CFA-16] d8=[CFA-32]
  //  20: CFA=sp+32 => x19=[CFA-8] fp=<same> lr=<same> d8=[CFA-32]
  //  24: CFA=sp+32 => x19=<same> fp=<same> lr=<same> d8=[CFA-32]
  //  28: CFA=sp +0 => x19=<same> fp=<same> lr=<same> d8=<same>

  sample_range = AddressRange(0x1000, sizeof(data));

  EXPECT_TRUE(engine->GetNonCallSiteUnwindPlanFromAssembly(
      sample_range, data, sizeof(data), /*target=*/nullptr, unwind_plan));

  //   4: CFA=sp+32 => d8=[CFA-32]
  row = unwind_plan.GetRowForFunctionOffset(4);
  EXPECT_EQ(4, row->GetOffset());
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_sp_arm64);
  EXPECT_TRUE(row->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(32, row->GetCFAValue().GetOffset());

  EXPECT_TRUE(row->GetRegisterInfo(fpu_d8_arm64, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-32, regloc.GetOffset());

  //  16: CFA=fp+24 => x19=[CFA-8] fp=[CFA-24] lr=[CFA-16] d8=[CFA-32]
  row = unwind_plan.GetRowForFunctionOffset(16);
  EXPECT_EQ(16, row->GetOffset());
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_fp_arm64);
  EXPECT_TRUE(row->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(24, row->GetCFAValue().GetOffset());

  EXPECT_TRUE(row->GetRegisterInfo(gpr_x19_arm64, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());

  EXPECT_TRUE(row->GetRegisterInfo(gpr_fp_arm64, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-24, regloc.GetOffset());

  EXPECT_TRUE(row->GetRegisterInfo(gpr_lr_arm64, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-16, regloc.GetOffset());

  EXPECT_TRUE(row->GetRegisterInfo(fpu_d8_arm64, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-32, regloc.GetOffset());

  //  28: CFA=sp +0 => x19=<same> fp=<same> lr=<same> d8=<same>
  row = unwind_plan.GetRowForFunctionOffset(28);
  EXPECT_EQ(28, row->GetOffset());
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_sp_arm64);
  EXPECT_TRUE(row->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row->GetCFAValue().GetOffset());

  if (row->GetRegisterInfo(gpr_x19_arm64, regloc)) {
    EXPECT_TRUE(regloc.IsSame());
  }
  if (row->GetRegisterInfo(gpr_fp_arm64, regloc)) {
    EXPECT_TRUE(regloc.IsSame());
  }
  if (row->GetRegisterInfo(gpr_lr_arm64, regloc)) {
    EXPECT_TRUE(regloc.IsSame());
  }
  if (row->GetRegisterInfo(fpu_d8_arm64, regloc)) {
    EXPECT_TRUE(regloc.IsSame());
  }
}

TEST_F(TestArm64InstEmulation, TestMidFunctionEpilogueAndBackwardsJump) {
  ArchSpec arch("arm64-apple-ios15");
  std::unique_ptr<UnwindAssemblyInstEmulation> engine(
      static_cast<UnwindAssemblyInstEmulation *>(
          UnwindAssemblyInstEmulation::CreateInstance(arch)));
  ASSERT_NE(nullptr, engine);

  const UnwindPlan::Row *row;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  UnwindPlan::Row::AbstractRegisterLocation regloc;

  // clang-format off
  uint8_t data[] = {
      0xff, 0xc3, 0x00, 0xd1, // <+0>:  sub    sp, sp, #0x30
      0xfd, 0x7b, 0x02, 0xa9, // <+4>:  stp    x29, x30, [sp, #0x20]
      0xfd, 0x83, 0x00, 0x91, // <+8>:  add    x29, sp, #0x20
      0x1f, 0x04, 0x00, 0xf1, // <+12>: cmp    x0, #0x1
      0x21, 0x01, 0x00, 0x54, // <+16>: b.ne   ; <+52> DO_SOMETHING_AND_GOTO_AFTER_EPILOGUE
      0xfd, 0x7b, 0x42, 0xa9, // <+20>: ldp    x29, x30, [sp, #0x20]
      0xff, 0xc3, 0x00, 0x91, // <+24>: add    sp, sp, #0x30
      0xc0, 0x03, 0x5f, 0xd6, // <+28>: ret
      // AFTER_EPILOGUE
      0x37, 0x00, 0x80, 0xd2, // <+32>: mov    x23, #0x1
      0xf6, 0x5f, 0x41, 0xa9, // <+36>: ldp    x22, x23, [sp, #0x10]
      0xfd, 0x7b, 0x42, 0xa9, // <+40>: ldp    x29, x30, [sp, #0x20]
      0xff, 0xc3, 0x00, 0x91, // <+44>: add    sp, sp, #0x30
      0xc0, 0x03, 0x5f, 0xd6, // <+48>: ret
      // DO_SOMETHING_AND_GOTO_AFTER_EPILOGUE
      0xf6, 0x5f, 0x01, 0xa9, // <+52>: stp    x22, x23, [sp, #0x10]
      0x36, 0x00, 0x80, 0xd2, // <+56>: mov    x22, #0x1
      0x37, 0x00, 0x80, 0xd2, // <+60>: mov    x23, #0x1
      0xf8, 0xff, 0xff, 0x17, // <+64>: b      ; <+32> AFTER_EPILOGUE
  };

  // UnwindPlan we expect:
  // row[0]:    0: CFA=sp +0 =>
  // row[1]:    4: CFA=sp+48 =>
  // row[2]:    8: CFA=sp+16 => fp=[CFA-16] lr=[CFA-8]
  // row[3]:   12: CFA=fp+16 => fp=[CFA-16] lr=[CFA-8]
  // row[4]:   24: CFA=sp+48 => fp=<same>   lr=<same>
  //
  // This must come from +56
  // row[5]:   32: CFA=fp+16 => fp=[CFA-16] lr=[CFA-8] x22=[CFA-32], x23=[CFA-24]
  // row[6]:   40: CFA=fp+16 => fp=[CFA-16] lr=[CFA-8] x22=same,     x23 = same
  // row[6]:   44: CFA=sp+48 => fp=same     lr=same    x22=same,     x23 = same
  // row[6]:   48: CFA=sp0   => fp=same     lr=same    x22=same,     x23 = same
  //
  // row[x]:   52: CFA=fp+16 => fp=[CFA-16] lr=[CFA-8]
  // row[x]:   56: CFA=fp+16 => fp=[CFA-16] lr=[CFA-8] x22=[CFA-32], x23=[CFA-24]
  // clang-format on

  sample_range = AddressRange(0x1000, sizeof(data));

  EXPECT_TRUE(engine->GetNonCallSiteUnwindPlanFromAssembly(
      sample_range, data, sizeof(data), /*target=*/nullptr, unwind_plan));

  // At the end of prologue (+12), CFA = fp + 16.
  // <+0>:  sub    sp, sp, #0x30
  // <+4>:  stp    x29, x30, [sp, #0x20]
  // <+8>:  add    x29, sp, #0x20
  row = unwind_plan.GetRowForFunctionOffset(12);
  EXPECT_EQ(12, row->GetOffset());
  EXPECT_TRUE(row->GetCFAValue().IsRegisterPlusOffset());
  EXPECT_EQ(row->GetCFAValue().GetRegisterNumber(), gpr_fp_arm64);
  EXPECT_EQ(row->GetCFAValue().GetOffset(), 16);

  // +16 and +20 are the same as +12.
  // <+12>: cmp    x0, #0x1
  // <+16>: b.ne   ; <+52> DO_SOMETHING_AND_GOTO_AFTER_EPILOGUE
  EXPECT_EQ(12, unwind_plan.GetRowForFunctionOffset(16)->GetOffset());
  EXPECT_EQ(12, unwind_plan.GetRowForFunctionOffset(20)->GetOffset());

  // After restoring $fp to caller's value, CFA = $sp + 48
  // <+20>: ldp    x29, x30, [sp, #0x20]
  row = unwind_plan.GetRowForFunctionOffset(24);
  EXPECT_EQ(24, row->GetOffset());
  EXPECT_TRUE(row->GetCFAValue().IsRegisterPlusOffset());
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_sp_arm64);
  EXPECT_EQ(row->GetCFAValue().GetOffset(), 48);

  // $sp has been restored
  // <+24>: add    sp, sp, #0x30
  row = unwind_plan.GetRowForFunctionOffset(28);
  EXPECT_EQ(28, row->GetOffset());
  EXPECT_TRUE(row->GetCFAValue().IsRegisterPlusOffset());
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_sp_arm64);
  EXPECT_EQ(row->GetCFAValue().GetOffset(), 0);

  // Row for offset +32 should not inherit the state of the `ret` instruction
  // in +28. Instead, it should inherit the state of the branch in +64.
  // Check for register x22, which is available in row +64.
  // <+28>: ret
  // <+32>: mov    x23, #0x1
  row = unwind_plan.GetRowForFunctionOffset(32);
  EXPECT_EQ(32, row->GetOffset());
  {
    UnwindPlan::Row::AbstractRegisterLocation loc;
    EXPECT_TRUE(row->GetRegisterInfo(gpr_x22_arm64, loc));
    EXPECT_TRUE(loc.IsAtCFAPlusOffset());
    EXPECT_EQ(loc.GetOffset(), -32);
  }

  // Check that the state of this branch
  // <+16>: b.ne   ; <+52> DO_SOMETHING_AND_GOTO_AFTER_EPILOGUE
  // was forwarded to the branch target:
  // <+52>: stp    x22, x23, [sp, #0x10]
  row = unwind_plan.GetRowForFunctionOffset(52);
  EXPECT_EQ(52, row->GetOffset());
  EXPECT_TRUE(row->GetCFAValue().IsRegisterPlusOffset());
  EXPECT_EQ(row->GetCFAValue().GetRegisterNumber(), gpr_fp_arm64);
  EXPECT_EQ(row->GetCFAValue().GetOffset(), 16);

  row = unwind_plan.GetRowForFunctionOffset(64);
  {
    UnwindPlan::Row::AbstractRegisterLocation loc;
    EXPECT_TRUE(row->GetRegisterInfo(gpr_x22_arm64, loc));
    EXPECT_TRUE(loc.IsAtCFAPlusOffset());
    EXPECT_EQ(loc.GetOffset(), -32);
  }
}

namespace {
/// `caller` occupies the first kCallerSize bytes of .text and
/// OUTLINED_FUNCTION_TEST follows immediately after it.
constexpr size_t kCallerSize = 16;

/// An ELF module holding a `caller` function at 0x1000 immediately followed by
/// an OUTLINED_FUNCTION_TEST helper, plus a Target able to read it.
struct OutlinedFunctionFixture {
  std::optional<TestFile> file;
  ModuleSP module_sp;
  DebuggerSP debugger_sp;
  TargetSP target_sp;
  Address caller_addr;
};

/// \param text the contents of .text: the kCallerSize bytes of `caller`,
/// followed by the body of OUTLINED_FUNCTION_TEST.
std::optional<OutlinedFunctionFixture>
MakeOutlinedFunctionFixture(llvm::ArrayRef<uint8_t> text) {
  OutlinedFunctionFixture fixture;

  llvm::Expected<TestFile> file = TestFile::fromYaml(
      llvm::formatv(R"(
--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  Type:            ET_EXEC
  Machine:         EM_AARCH64
Sections:
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    Address:         0x1000
    AddressAlign:    0x4
    Content:         {0}
Symbols:
  - Name:            caller
    Type:            STT_FUNC
    Section:         .text
    Value:           0x1000
    Size:            {1}
  - Name:            OUTLINED_FUNCTION_TEST
    Type:            STT_FUNC
    Section:         .text
    Value:           0x1010
    Size:            {2}
...
)",
                    llvm::toHex(text), kCallerSize, text.size() - kCallerSize)
          .str());
  if (!file)
    return std::nullopt;
  fixture.file = std::move(*file);

  fixture.module_sp = std::make_shared<Module>(fixture.file->moduleSpec());
  fixture.debugger_sp = Debugger::CreateInstance();
  if (!fixture.module_sp || !fixture.debugger_sp)
    return std::nullopt;

  PlatformSP platform_sp;
  fixture.debugger_sp->GetTargetList().CreateTarget(
      *fixture.debugger_sp, "", fixture.module_sp->GetArchitecture(),
      eLoadDependentsNo, platform_sp, fixture.target_sp);
  if (!fixture.target_sp)
    return std::nullopt;

  if (!fixture.module_sp->ResolveFileAddress(0x1000, fixture.caller_addr))
    return std::nullopt;

  return fixture;
}
} // namespace

// Test that the assembly-scan unwind plan generator can follow outlined
// functions.
TEST_F(TestArm64InstEmulation, TestOutlinedPrologueIsFollowed) {
  // clang-format off
  uint8_t text[] = {
      // 0x1000 <caller>
      0xfd, 0x7b, 0xbf, 0xa9, // 0xa9bf7bfd : stp x29, x30, [sp, #-0x10]!
      0x03, 0x00, 0x00, 0x94, // 0x94000003 : bl  OUTLINED_FUNCTION_TEST
      0x1f, 0x20, 0x03, 0xd5, // 0xd503201f : nop
      0xc0, 0x03, 0x5f, 0xd6, // 0xd65f03c0 : ret

      // 0x1010 <OUTLINED_FUNCTION_TEST>
      0xf4, 0x4f, 0xbe, 0xa9, // 0xa9be4ff4 : stp x20, x19, [sp, #-0x20]!
      0xf6, 0x57, 0x01, 0xa9, // 0xa90157f6 : stp x22, x21, [sp, #0x10]
      0xfd, 0x83, 0x00, 0x91, // 0x910083fd : add x29, sp, #0x20
      0xc0, 0x03, 0x5f, 0xd6, // 0xd65f03c0 : ret
  };
  // clang-format on

  auto fixture = MakeOutlinedFunctionFixture(text);
  ASSERT_TRUE(fixture.has_value());

  std::unique_ptr<UnwindAssemblyInstEmulation> engine(
      static_cast<UnwindAssemblyInstEmulation *>(
          UnwindAssemblyInstEmulation::CreateInstance(
              fixture->module_sp->GetArchitecture())));
  ASSERT_NE(nullptr, engine);

  AddressRange sample_range(fixture->caller_addr, kCallerSize);

  UnwindPlan unwind_plan(eRegisterKindLLDB);
  EXPECT_TRUE(engine->GetNonCallSiteUnwindPlanFromAssembly(
      sample_range, text, kCallerSize, fixture->target_sp.get(), unwind_plan));

  UnwindPlan::Row::AbstractRegisterLocation regloc;

  // Before the call only the caller's own inline save is known.
  const UnwindPlan::Row *row = unwind_plan.GetRowForFunctionOffset(4);
  ASSERT_NE(nullptr, row);
  EXPECT_EQ(4, row->GetOffset());
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_sp_arm64);
  EXPECT_EQ(16, row->GetCFAValue().GetOffset());
  EXPECT_FALSE(row->GetRegisterInfo(gpr_x19_arm64, regloc));

  // After the call, everything in the outlined function should be in the plan.
  row = unwind_plan.GetRowForFunctionOffset(8);
  ASSERT_NE(nullptr, row);
  EXPECT_EQ(8, row->GetOffset());
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_fp_arm64);
  EXPECT_TRUE(row->GetCFAValue().IsRegisterPlusOffset());
  EXPECT_EQ(16, row->GetCFAValue().GetOffset());

  for (auto [reg, offset] :
       {std::pair(gpr_x19_arm64, -40), std::pair(gpr_x20_arm64, -48),
        std::pair(gpr_x21_arm64, -24), std::pair(gpr_x22_arm64, -32),
        std::pair(gpr_fp_arm64, -16), std::pair(gpr_lr_arm64, -8)}) {
    SCOPED_TRACE(reg);
    EXPECT_TRUE(row->GetRegisterInfo(reg, regloc));
    EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
    EXPECT_EQ(offset, regloc.GetOffset());
  }
}

// Test that an outlined function with control flow is not used when creating
// the unwind plan.
TEST_F(TestArm64InstEmulation, TestBranchingOutlinedFunctionIsNotFollowed) {
  // clang-format off
  uint8_t text[] = {
      // 0x1000 <caller>
      0xfd, 0x7b, 0xbf, 0xa9, // 0xa9bf7bfd : stp x29, x30, [sp, #-0x10]!
      0x03, 0x00, 0x00, 0x94, // 0x94000003 : bl  OUTLINED_FUNCTION_TEST
      0x1f, 0x20, 0x03, 0xd5, // 0xd503201f : nop
      0xc0, 0x03, 0x5f, 0xd6, // 0xd65f03c0 : ret

      // 0x1010 <OUTLINED_FUNCTION_TEST>
      0xf4, 0x4f, 0xbe, 0xa9, // 0xa9be4ff4 : stp x20, x19, [sp, #-0x20]!
      0x40, 0x00, 0x00, 0xb4, // 0xb4000040 : cbz x0, #8 <- the branch
      0xf6, 0x57, 0x01, 0xa9, // 0xa90157f6 : stp x22, x21, [sp, #0x10]
      0xfd, 0x83, 0x00, 0x91, // 0x910083fd : add x29, sp, #0x20
      0xc0, 0x03, 0x5f, 0xd6, // 0xd65f03c0 : ret
  };
  // clang-format on

  auto fixture = MakeOutlinedFunctionFixture(text);
  ASSERT_TRUE(fixture.has_value());

  std::unique_ptr<UnwindAssemblyInstEmulation> engine(
      static_cast<UnwindAssemblyInstEmulation *>(
          UnwindAssemblyInstEmulation::CreateInstance(
              fixture->module_sp->GetArchitecture())));
  ASSERT_NE(nullptr, engine);

  AddressRange sample_range(fixture->caller_addr, kCallerSize);

  UnwindPlan unwind_plan(eRegisterKindLLDB);
  EXPECT_TRUE(engine->GetNonCallSiteUnwindPlanFromAssembly(
      sample_range, text, kCallerSize, fixture->target_sp.get(), unwind_plan));

  // The call adds no row of its own: the state after it is still the one
  // established at offset 4 by the caller's own save.
  UnwindPlan::Row::AbstractRegisterLocation regloc;
  const UnwindPlan::Row *row = unwind_plan.GetRowForFunctionOffset(8);
  ASSERT_NE(nullptr, row);
  EXPECT_EQ(4, row->GetOffset());
  EXPECT_TRUE(row->GetCFAValue().GetRegisterNumber() == gpr_sp_arm64);
  EXPECT_EQ(16, row->GetCFAValue().GetOffset());
  for (uint32_t reg :
       {gpr_x19_arm64, gpr_x20_arm64, gpr_x21_arm64, gpr_x22_arm64}) {
    SCOPED_TRACE(reg);
    EXPECT_FALSE(row->GetRegisterInfo(reg, regloc));
  }
}
