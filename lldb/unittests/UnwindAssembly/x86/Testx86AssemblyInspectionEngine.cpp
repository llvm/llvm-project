//===-- Testx86AssemblyInspectionEngine.cpp ---------------------------*- C++
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

#include "Plugins/UnwindAssembly/x86/x86AssemblyInspectionEngine.h"
#include "lldb/Core/Address.h"
#include "lldb/Core/AddressRange.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Symbol/UnwindPlan.h"

#include "llvm/Support/TargetSelect.h"

using namespace lldb;
using namespace lldb_private;

class Testx86AssemblyInspectionEngine : public testing::Test {
public:
  //  static void SetUpTestCase() { }

  //  static void TearDownTestCase() { }

  //  virtual void SetUp() override { }

  //  virtual void TearDown() override { }

protected:
};

// only defining the register names / numbers that the unwinder is actually
// using today

// names should match the constants below.  These will be the eRegisterKindLLDB
// register numbers.

const char *x86_64_reg_names[] = {"rax", "rbx", "rcx", "rdx", "rsp", "rbp",
                                  "rsi", "rdi", "r8",  "r9",  "r10", "r11",
                                  "r12", "r13", "r14", "r15", "rip"};

enum x86_64_regs {
  k_rax = 0,
  k_rbx = 1,
  k_rcx = 2,
  k_rdx = 3,
  k_rsp = 4,
  k_rbp = 5,
  k_rsi = 6,
  k_rdi = 7,
  k_r8 = 8,
  k_r9 = 9,
  k_r10 = 10,
  k_r11 = 11,
  k_r12 = 12,
  k_r13 = 13,
  k_r14 = 14,
  k_r15 = 15,
  k_rip = 16
};

// names should match the constants below.  These will be the eRegisterKindLLDB
// register numbers.

const char *i386_reg_names[] = {"eax", "ecx", "edx", "ebx", "esp",
                                "ebp", "esi", "edi", "eip"};

enum i386_regs {
  k_eax = 0,
  k_ecx = 1,
  k_edx = 2,
  k_ebx = 3,
  k_esp = 4,
  k_ebp = 5,
  k_esi = 6,
  k_edi = 7,
  k_eip = 8
};

std::unique_ptr<x86AssemblyInspectionEngine> Getx86_64Inspector() {

  ArchSpec arch("x86_64-apple-macosx", nullptr);
  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllDisassemblers();
  std::unique_ptr<x86AssemblyInspectionEngine> engine(
      new x86AssemblyInspectionEngine(arch));

  std::vector<x86AssemblyInspectionEngine::lldb_reg_info> lldb_regnums;
  int i = 0;
  for (const auto &name : x86_64_reg_names) {
    x86AssemblyInspectionEngine::lldb_reg_info ri;
    ri.name = name;
    ri.lldb_regnum = i++;
    lldb_regnums.push_back(ri);
  }

  engine->Initialize(lldb_regnums);
  return engine;
}

std::unique_ptr<x86AssemblyInspectionEngine> Geti386Inspector() {

  ArchSpec arch("i386-apple-macosx", nullptr);
  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllDisassemblers();
  std::unique_ptr<x86AssemblyInspectionEngine> engine(
      new x86AssemblyInspectionEngine(arch));

  std::vector<x86AssemblyInspectionEngine::lldb_reg_info> lldb_regnums;
  int i = 0;
  for (const auto &name : i386_reg_names) {
    x86AssemblyInspectionEngine::lldb_reg_info ri;
    ri.name = name;
    ri.lldb_regnum = i++;
    lldb_regnums.push_back(ri);
  }

  engine->Initialize(lldb_regnums);
  return engine;
}

TEST_F(Testx86AssemblyInspectionEngine, TestSimple64bitFrameFunction) {
  std::unique_ptr<x86AssemblyInspectionEngine> engine = Getx86_64Inspector();

  // 'int main() { }' compiled for x86_64-apple-macosx with clang
  uint8_t data[] = {
      0x55,             // offset 0 -- pushq %rbp
      0x48, 0x89, 0xe5, // offset 1 -- movq %rsp, %rbp
      0x31, 0xc0,       // offset 4 -- xorl %eax, %eax
      0x5d,             // offset 6 -- popq %rbp
      0xc3              // offset 7 -- retq
  };

  AddressRange sample_range(0x1000, sizeof(data));

  UnwindPlan unwind_plan(eRegisterKindLLDB);
  EXPECT_TRUE(engine->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  // Expect four unwind rows:
  // 0: CFA=rsp +8 => rsp=CFA+0 rip=[CFA-8]
  // 1: CFA=rsp+16 => rbp=[CFA-16] rsp=CFA+0 rip=[CFA-8]
  // 4: CFA=rbp+16 => rbp=[CFA-16] rsp=CFA+0 rip=[CFA-8]
  // 7: CFA=rsp +8 => rsp=CFA+0 rip=[CFA-8]

  EXPECT_TRUE(unwind_plan.GetInitialCFARegister() == k_rsp);
  EXPECT_TRUE(unwind_plan.GetUnwindPlanValidAtAllInstructions() ==
              eLazyBoolYes);
  EXPECT_TRUE(unwind_plan.GetSourcedFromCompiler() == eLazyBoolNo);

  UnwindPlan::Row::RegisterLocation regloc;

  // 0: CFA=rsp +8 => rsp=CFA+0 rip=[CFA-8]
  UnwindPlan::RowSP row_sp = unwind_plan.GetRowForFunctionOffset(0);
  EXPECT_EQ(0, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(8, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_rip, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());

  // 1: CFA=rsp+16 => rbp=[CFA-16] rsp=CFA+0 rip=[CFA-8]
  row_sp = unwind_plan.GetRowForFunctionOffset(1);
  EXPECT_EQ(1, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_rip, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());

  // 4: CFA=rbp+16 => rbp=[CFA-16] rsp=CFA+0 rip=[CFA-8]
  row_sp = unwind_plan.GetRowForFunctionOffset(4);
  EXPECT_EQ(4, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rbp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_rip, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());

  // 7: CFA=rsp +8 => rsp=CFA+0 rip=[CFA-8]
  row_sp = unwind_plan.GetRowForFunctionOffset(7);
  EXPECT_EQ(7, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(8, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_rip, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());
}

TEST_F(Testx86AssemblyInspectionEngine, TestSimple32bitFrameFunction) {
  std::unique_ptr<x86AssemblyInspectionEngine> engine = Geti386Inspector();

  // 'int main() { }' compiled for i386-apple-macosx with clang
  uint8_t data[] = {
      0x55,       // offset 0 -- pushl %ebp
      0x89, 0xe5, // offset 1 -- movl %esp, %ebp
      0x31, 0xc0, // offset 3 -- xorl %eax, %eax
      0x5d,       // offset 5 -- popl %ebp
      0xc3        // offset 6 -- retl
  };

  AddressRange sample_range(0x1000, sizeof(data));

  UnwindPlan unwind_plan(eRegisterKindLLDB);
  EXPECT_TRUE(engine->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  // Expect four unwind rows:
  // 0: CFA=esp +4 => esp=CFA+0 eip=[CFA-4]
  // 1: CFA=esp +8 => ebp=[CFA-8] esp=CFA+0 eip=[CFA-4]
  // 3: CFA=ebp +8 => ebp=[CFA-8] esp=CFA+0 eip=[CFA-4]
  // 6: CFA=esp +4 => esp=CFA+0 eip=[CFA-4]

  EXPECT_TRUE(unwind_plan.GetInitialCFARegister() == k_esp);
  EXPECT_TRUE(unwind_plan.GetUnwindPlanValidAtAllInstructions() ==
              eLazyBoolYes);
  EXPECT_TRUE(unwind_plan.GetSourcedFromCompiler() == eLazyBoolNo);

  UnwindPlan::Row::RegisterLocation regloc;

  // offset 0 -- pushl %ebp
  UnwindPlan::RowSP row_sp = unwind_plan.GetRowForFunctionOffset(0);
  EXPECT_EQ(0, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_esp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(4, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_eip, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_TRUE(regloc.GetOffset() == -4);

  // 1: CFA=esp +8 => ebp=[CFA-8] esp=CFA+0 eip=[CFA-4]
  row_sp = unwind_plan.GetRowForFunctionOffset(1);
  EXPECT_EQ(1, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_esp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(8, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_eip, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-4, regloc.GetOffset());

  // 3: CFA=ebp +8 => ebp=[CFA-8] esp=CFA+0 eip=[CFA-4]
  row_sp = unwind_plan.GetRowForFunctionOffset(3);
  EXPECT_EQ(3, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_ebp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(8, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_eip, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-4, regloc.GetOffset());

  // 6: CFA=esp +4 => esp=CFA+0 eip=[CFA-4]
  row_sp = unwind_plan.GetRowForFunctionOffset(6);
  EXPECT_EQ(6, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_esp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(4, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_eip, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-4, regloc.GetOffset());
}

TEST_F(Testx86AssemblyInspectionEngine, Test64bitFramelessBigStackFrame) {
  std::unique_ptr<x86AssemblyInspectionEngine> engine = Getx86_64Inspector();

  // this source file:
  //
  // #include <stdio.h>
  // int main (int argc, char **argv)
  // {
  //
  //     const int arrsize = 60;
  //     int buf[arrsize * arrsize];
  //     int accum = argc;
  //     for (int i = 0; i < arrsize; i++)
  //         for (int j = 0; j < arrsize; j++)
  //         {
  //             if (i > 0 && j > 0)
  //             {
  //                 int n = buf[(i-1) * (j-1)] * 2;
  //                 int m = buf[(i-1) * (j-1)] / 2;
  //                 int j = buf[(i-1) * (j-1)] + 2;
  //                 int k = buf[(i-1) * (j-1)] - 2;
  //                 printf ("%d ", n + m + j + k);
  //                 buf[(i-1) * (j-1)] += n - m + j - k;
  //             }
  //             buf[i*j] = accum++;
  //         }
  //
  //     return buf[(arrsize * arrsize) - 2] + printf ("%d\n", buf[(arrsize *
  //     arrsize) - 3]);
  // }
  //
  // compiled 'clang -fomit-frame-pointer -Os' for x86_64-apple-macosx

  uint8_t data[] = {
      0x55,       // offset 0  -- pushq %rbp
      0x41, 0x57, // offset 1  -- pushq %r15
      0x41, 0x56, // offset 3  -- pushq %r14
      0x41, 0x55, // offset 5  -- pushq %r13
      0x41, 0x54, // offset 7  -- pushq %r12
      0x53,       // offset 9  -- pushq %rbx
      0x48, 0x81, 0xec, 0x68, 0x38, 0x00,
      0x00, // offset 10 -- subq $0x3868, %rsp

      // ....

      0x48, 0x81, 0xc4, 0x68, 0x38, 0x00,
      0x00,                        // offset 17 -- addq $0x3868, %rsp
      0x5b,                        // offset 24 -- popq %rbx
      0x41, 0x5c,                  // offset 25 -- popq %r12
      0x41, 0x5d,                  // offset 27 -- popq %r13
      0x41, 0x5e,                  // offset 29 -- popq %r14
      0x41, 0x5f,                  // offset 31 -- popq %r15
      0x5d,                        // offset 33 -- popq %rbp
      0xc3,                        // offset 34 -- retq
      0xe8, 0x12, 0x34, 0x56, 0x78 // offset 35 -- callq whatever
  };

  AddressRange sample_range(0x1000, sizeof(data));

  UnwindPlan unwind_plan(eRegisterKindLLDB);
  EXPECT_TRUE(engine->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  // Unwind rules should look like
  // 0: CFA=rsp +8 => rsp=CFA+0 rip=[CFA-8]
  // 1: CFA=rsp+16 => rbp=[CFA-16] rsp=CFA+0 rip=[CFA-8]
  // 3: CFA=rsp+24 => rbp=[CFA-16] rsp=CFA+0 r15=[CFA-24] rip=[CFA-8]
  // 5: CFA=rsp+32 => rbp=[CFA-16] rsp=CFA+0 r14=[CFA-32] r15=[CFA-24]
  // rip=[CFA-8
  // 7: CFA=rsp+40 => rbp=[CFA-16] rsp=CFA+0 r13=[CFA-40] r14=[CFA-32]
  // r15=[CFA-24] rip=[CFA-8]
  // 9: CFA=rsp+48 => rbp=[CFA-16] rsp=CFA+0 r12=[CFA-48] r13=[CFA-40]
  // r14=[CFA-32] r15=[CFA-24] rip=[CFA-8]
  // 10: CFA=rsp+56 => rbx=[CFA-56] rbp=[CFA-16] rsp=CFA+0 r12=[CFA-48]
  // r13=[CFA-40] r14=[CFA-32] r15=[CFA-24] rip=[CFA-8]
  // 17: CFA=rsp+14496 => rbx=[CFA-56] rbp=[CFA-16] rsp=CFA+0 r12=[CFA-48]
  // r13=[CFA-40] r14=[CFA-32] r15=[CFA-24] rip=[CFA-8]

  // 24: CFA=rsp+56 => rbx=[CFA-56] rbp=[CFA-16] rsp=CFA+0 r12=[CFA-48]
  // r13=[CFA-40] r14=[CFA-32] r15=[CFA-24] rip=[CFA-8]
  // 25: CFA=rsp+48 => rbp=[CFA-16] rsp=CFA+0 r12=[CFA-48] r13=[CFA-40]
  // r14=[CFA-32] r15=[CFA-24] rip=[CFA-8]
  // 27: CFA=rsp+40 => rbp=[CFA-16] rsp=CFA+0 r13=[CFA-40] r14=[CFA-32]
  // r15=[CFA-24] rip=[CFA-8]
  // 29: CFA=rsp+32 => rbp=[CFA-16] rsp=CFA+0 r14=[CFA-32] r15=[CFA-24]
  // rip=[CFA-8]
  // 31: CFA=rsp+24 => rbp=[CFA-16] rsp=CFA+0 r15=[CFA-24] rip=[CFA-8]
  // 33: CFA=rsp+16 => rbp=[CFA-16] rsp=CFA+0 rip=[CFA-8]
  // 34: CFA=rsp +8 => rsp=CFA+0 rip=[CFA-8]

  UnwindPlan::Row::RegisterLocation regloc;

  // grab the Row for when the prologue has finished executing:
  // 17: CFA=rsp+14496 => rbx=[CFA-56] rbp=[CFA-16] rsp=CFA+0 r12=[CFA-48]
  // r13=[CFA-40] r14=[CFA-32] r15=[CFA-24] rip=[CFA-8]

  UnwindPlan::RowSP row_sp = unwind_plan.GetRowForFunctionOffset(17);

  EXPECT_EQ(17, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(14496, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_rip, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_rbp, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-16, regloc.GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_r15, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-24, regloc.GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_r14, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-32, regloc.GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_r13, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-40, regloc.GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_r12, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-48, regloc.GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_rbx, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-56, regloc.GetOffset());

  // grab the Row for when the epilogue has finished executing:
  // 34: CFA=rsp +8 => rsp=CFA+0 rip=[CFA-8]

  row_sp = unwind_plan.GetRowForFunctionOffset(34);

  EXPECT_EQ(34, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(8, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_rip, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());

  // these could be set to IsSame and be valid -- meaning that the
  // register value is the same as the caller's -- but I'd rather
  // they not be mentioned at all.
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_rbp, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_r15, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_r14, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_r13, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_r12, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_rbx, regloc));
}

