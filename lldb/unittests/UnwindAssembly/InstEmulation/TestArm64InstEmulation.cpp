//===-- TestArm64InstEmulation.cpp ------------------------------------*- C++
//-*-===//

//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include <vector>

#include "Plugins/UnwindAssembly/InstEmulation/UnwindAssemblyInstEmulation.h"
#include "Utility/ARM64_DWARF_Registers.h"

#include "lldb/Core/Address.h"
#include "lldb/Core/AddressRange.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Target/UnwindAssembly.h"

#include "Plugins/Disassembler/llvm/DisassemblerLLVMC.h"
#include "Plugins/Instruction/ARM64/EmulateInstructionARM64.h"
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
  DisassemblerLLVMC::Initialize();
  EmulateInstructionARM64::Initialize();
}

void TestArm64InstEmulation::TearDownTestCase() {
  DisassemblerLLVMC::Terminate();
  EmulateInstructionARM64::Terminate();
}

TEST_F(TestArm64InstEmulation, TestSimpleDarwinFunction) {
  ArchSpec arch("arm64-apple-ios10", nullptr);
  UnwindAssemblyInstEmulation *engine =
      static_cast<UnwindAssemblyInstEmulation *>(
          UnwindAssemblyInstEmulation::CreateInstance(arch));
  ASSERT_NE(nullptr, engine);

  UnwindPlan::RowSP row_sp;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  UnwindPlan::Row::RegisterLocation regloc;

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

  // row[0]:    0: CFA=sp +0 =>
  // row[1]:    4: CFA=sp+16 => fp=[CFA-16] lr=[CFA-8]
  // row[2]:    8: CFA=fp+16 => fp=[CFA-16] lr=[CFA-8]
  // row[2]:   16: CFA=sp+16 => fp=[CFA-16] lr=[CFA-8]
  // row[3]:   20: CFA=sp +0 => fp= <same> lr= <same>

  sample_range = AddressRange(0x1000, sizeof(data));

  EXPECT_TRUE(engine->GetNonCallSiteUnwindPlanFromAssembly(
      sample_range, data, sizeof(data), unwind_plan));

  // CFA=sp +0
  row_sp = unwind_plan.GetRowForFunctionOffset(0);
  EXPECT_EQ(0, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == arm64_dwarf::sp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row_sp->GetCFAValue().GetOffset());

  // CFA=sp+16 => fp=[CFA-16] lr=[CFA-8]
  row_sp = unwind_plan.GetRowForFunctionOffset(4);
  EXPECT_EQ(4, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == arm64_dwarf::sp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::fp, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-16, regloc.GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::lr, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());

  // CFA=fp+16 => fp=[CFA-16] lr=[CFA-8]
  row_sp = unwind_plan.GetRowForFunctionOffset(8);
  EXPECT_EQ(8, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == arm64_dwarf::fp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::fp, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-16, regloc.GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::lr, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());

  // CFA=sp+16 => fp=[CFA-16] lr=[CFA-8]
  row_sp = unwind_plan.GetRowForFunctionOffset(16);
  EXPECT_EQ(16, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == arm64_dwarf::sp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::fp, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-16, regloc.GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::lr, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());

  // CFA=sp +0 => fp= <same> lr= <same>
  row_sp = unwind_plan.GetRowForFunctionOffset(20);
  EXPECT_EQ(20, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == arm64_dwarf::sp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row_sp->GetCFAValue().GetOffset());
}

TEST_F(TestArm64InstEmulation, TestMediumDarwinFunction) {
  ArchSpec arch("arm64-apple-ios10", nullptr);
  UnwindAssemblyInstEmulation *engine =
      static_cast<UnwindAssemblyInstEmulation *>(
          UnwindAssemblyInstEmulation::CreateInstance(arch));
  ASSERT_NE(nullptr, engine);

  UnwindPlan::RowSP row_sp;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  UnwindPlan::Row::RegisterLocation regloc;

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
      sample_range, data, sizeof(data), unwind_plan));

  // 0: CFA=sp +0 =>
  row_sp = unwind_plan.GetRowForFunctionOffset(0);
  EXPECT_EQ(0, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == arm64_dwarf::sp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row_sp->GetCFAValue().GetOffset());

  // 4: CFA=sp+48 => x21=[CFA-40] x22=[CFA-48]
  row_sp = unwind_plan.GetRowForFunctionOffset(4);
  EXPECT_EQ(4, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == arm64_dwarf::sp);
  EXPECT_EQ(48, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::x21, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-40, regloc.GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::x22, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-48, regloc.GetOffset());

  // 8: CFA=sp+48 => x19=[CFA-24] x20=[CFA-32] x21=[CFA-40] x22=[CFA-48]
  row_sp = unwind_plan.GetRowForFunctionOffset(8);
  EXPECT_EQ(8, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == arm64_dwarf::sp);
  EXPECT_EQ(48, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::x19, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-24, regloc.GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::x20, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-32, regloc.GetOffset());

  // 12: CFA=sp+48 => x19=[CFA-24] x20=[CFA-32] x21=[CFA-40] x22=[CFA-48]
  // fp=[CFA-16] lr=[CFA-8]
  row_sp = unwind_plan.GetRowForFunctionOffset(12);
  EXPECT_EQ(12, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == arm64_dwarf::sp);
  EXPECT_EQ(48, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::fp, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-16, regloc.GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::lr, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());

  // 16: CFA=fp+16 => x19=[CFA-24] x20=[CFA-32] x21=[CFA-40] x22=[CFA-48]
  // fp=[CFA-16] lr=[CFA-8]
  row_sp = unwind_plan.GetRowForFunctionOffset(16);
  EXPECT_EQ(16, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == arm64_dwarf::fp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row_sp->GetCFAValue().GetOffset());

  // 28: CFA=sp+48 => x19=[CFA-24] x20=[CFA-32] x21=[CFA-40] x22=[CFA-48]
  // fp=[CFA-16] lr=[CFA-8]
  row_sp = unwind_plan.GetRowForFunctionOffset(28);
  EXPECT_EQ(28, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == arm64_dwarf::sp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(48, row_sp->GetCFAValue().GetOffset());

  // 32: CFA=sp+48 => x19=[CFA-24] x20=[CFA-32] x21=[CFA-40] x22=[CFA-48] fp=
  // <same> lr= <same>
  row_sp = unwind_plan.GetRowForFunctionOffset(32);
  EXPECT_EQ(32, row_sp->GetOffset());

  // I'd prefer if these restored registers were cleared entirely instead of set
  // to IsSame...
  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::fp, regloc));
  EXPECT_TRUE(regloc.IsSame());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::lr, regloc));
  EXPECT_TRUE(regloc.IsSame());

  // 36: CFA=sp+48 => x19= <same> x20= <same> x21=[CFA-40] x22=[CFA-48] fp=
  // <same> lr= <same>
  row_sp = unwind_plan.GetRowForFunctionOffset(36);
  EXPECT_EQ(36, row_sp->GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::x19, regloc));
  EXPECT_TRUE(regloc.IsSame());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::x20, regloc));
  EXPECT_TRUE(regloc.IsSame());

  // 40: CFA=sp +0 => x19= <same> x20= <same> x21= <same> x22= <same> fp= <same>
  // lr= <same>
  row_sp = unwind_plan.GetRowForFunctionOffset(40);
  EXPECT_EQ(40, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == arm64_dwarf::sp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::x21, regloc));
  EXPECT_TRUE(regloc.IsSame());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::x22, regloc));
  EXPECT_TRUE(regloc.IsSame());
}

TEST_F(TestArm64InstEmulation, TestFramelessThreeEpilogueFunction) {
  ArchSpec arch("arm64-apple-ios10", nullptr);
  UnwindAssemblyInstEmulation *engine =
      static_cast<UnwindAssemblyInstEmulation *>(
          UnwindAssemblyInstEmulation::CreateInstance(arch));
  ASSERT_NE(nullptr, engine);

  UnwindPlan::RowSP row_sp;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  UnwindPlan::Row::RegisterLocation regloc;

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
      sample_range, data, sizeof(data), unwind_plan));

  // 0: CFA=sp +0 =>
  row_sp = unwind_plan.GetRowForFunctionOffset(0);
  EXPECT_EQ(0, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == arm64_dwarf::sp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row_sp->GetCFAValue().GetOffset());

  row_sp = unwind_plan.GetRowForFunctionOffset(32);
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == arm64_dwarf::sp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row_sp->GetCFAValue().GetOffset());

  EXPECT_FALSE(row_sp->GetRegisterInfo(arm64_dwarf::x19, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(arm64_dwarf::x20, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(arm64_dwarf::x21, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(arm64_dwarf::x22, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(arm64_dwarf::x23, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(arm64_dwarf::x24, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(arm64_dwarf::x25, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(arm64_dwarf::x26, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(arm64_dwarf::x27, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(arm64_dwarf::x28, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(arm64_dwarf::fp, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(arm64_dwarf::lr, regloc));

  row_sp = unwind_plan.GetRowForFunctionOffset(36);
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == arm64_dwarf::sp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row_sp->GetCFAValue().GetOffset());

  row_sp = unwind_plan.GetRowForFunctionOffset(52);
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == arm64_dwarf::sp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row_sp->GetCFAValue().GetOffset());

  row_sp = unwind_plan.GetRowForFunctionOffset(56);
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == arm64_dwarf::sp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row_sp->GetCFAValue().GetOffset());

  row_sp = unwind_plan.GetRowForFunctionOffset(60);
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == arm64_dwarf::sp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(0, row_sp->GetCFAValue().GetOffset());
}

TEST_F(TestArm64InstEmulation, TestRegisterSavedTwice) {
  ArchSpec arch("arm64-apple-ios10", nullptr);
  UnwindAssemblyInstEmulation *engine =
      static_cast<UnwindAssemblyInstEmulation *>(
          UnwindAssemblyInstEmulation::CreateInstance(arch));
  ASSERT_NE(nullptr, engine);

  UnwindPlan::RowSP row_sp;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  UnwindPlan::Row::RegisterLocation regloc;

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
      sample_range, data, sizeof(data), unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(36);
  EXPECT_EQ(28, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == arm64_dwarf::fp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::x20, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-32, regloc.GetOffset());

  row_sp = unwind_plan.GetRowForFunctionOffset(40);
  EXPECT_EQ(28, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == arm64_dwarf::fp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(arm64_dwarf::x20, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-32, regloc.GetOffset());
}
