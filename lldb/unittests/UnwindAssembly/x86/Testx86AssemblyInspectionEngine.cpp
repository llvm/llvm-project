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
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/StreamString.h"

#include "llvm/Support/TargetSelect.h"

using namespace lldb;
using namespace lldb_private;

class Testx86AssemblyInspectionEngine : public testing::Test {
public:
  static void SetUpTestCase();

  //  static void TearDownTestCase() { }

  //  virtual void SetUp() override { }

  //  virtual void TearDown() override { }

protected:
};

void Testx86AssemblyInspectionEngine::SetUpTestCase() {
  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllDisassemblers();
}

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

  ArchSpec arch("x86_64-apple-macosx");
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

  ArchSpec arch("i386-apple-macosx");
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

namespace lldb_private {
static std::ostream &operator<<(std::ostream &OS,
                                const UnwindPlan::Row::FAValue &CFA) {
  StreamString S;
  CFA.Dump(S, nullptr, nullptr);
  return OS << S.GetData();
}
} // namespace lldb_private

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
  EXPECT_EQ(0ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(8, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_rip, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());

  // 1: CFA=rsp+16 => rbp=[CFA-16] rsp=CFA+0 rip=[CFA-8]
  row_sp = unwind_plan.GetRowForFunctionOffset(1);
  EXPECT_EQ(1ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_rip, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());

  // 4: CFA=rbp+16 => rbp=[CFA-16] rsp=CFA+0 rip=[CFA-8]
  row_sp = unwind_plan.GetRowForFunctionOffset(4);
  EXPECT_EQ(4ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rbp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_rip, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());

  // 7: CFA=rsp +8 => rsp=CFA+0 rip=[CFA-8]
  row_sp = unwind_plan.GetRowForFunctionOffset(7);
  EXPECT_EQ(7ull, row_sp->GetOffset());
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
  EXPECT_EQ(0ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_esp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(4, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_eip, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_TRUE(regloc.GetOffset() == -4);

  // 1: CFA=esp +8 => ebp=[CFA-8] esp=CFA+0 eip=[CFA-4]
  row_sp = unwind_plan.GetRowForFunctionOffset(1);
  EXPECT_EQ(1ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_esp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(8, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_eip, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-4, regloc.GetOffset());

  // 3: CFA=ebp +8 => ebp=[CFA-8] esp=CFA+0 eip=[CFA-4]
  row_sp = unwind_plan.GetRowForFunctionOffset(3);
  EXPECT_EQ(3ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_ebp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(8, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_eip, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-4, regloc.GetOffset());

  // 6: CFA=esp +4 => esp=CFA+0 eip=[CFA-4]
  row_sp = unwind_plan.GetRowForFunctionOffset(6);
  EXPECT_EQ(6ull, row_sp->GetOffset());
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

  EXPECT_EQ(17ull, row_sp->GetOffset());
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

  EXPECT_EQ(34ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(8, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_rip, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());

  // these could be set to IsSame and be valid -- meaning that the
  // register value is the same as the caller's -- but I'd rather
  // they not be mentioned at all.

  EXPECT_FALSE(row_sp->GetRegisterInfo(k_rax, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_rbx, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_rcx, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_rdx, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_rbp, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_rsi, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_rdi, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_r8, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_r9, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_r10, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_r11, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_r12, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_r13, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_r14, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_r15, regloc));
}

TEST_F(Testx86AssemblyInspectionEngine, Test32bitFramelessBigStackFrame) {
  std::unique_ptr<x86AssemblyInspectionEngine> engine = Geti386Inspector();

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
  // compiled 'clang -arch i386 -fomit-frame-pointer -Os' for i386-apple-macosx

  // simplified assembly version of the above function, which is used as the
  // input
  // data:
  //
  // 	.section	__TEXT,__text,regular,pure_instructions
  // 	.macosx_version_min 10, 12
  // 	.globl	_main
  // 	.align	4, 0x90
  // _main:                                  ## @main
  // ## BB#0:
  // 	pushl %ebp
  // 	pushl %ebx
  // 	pushl %edi
  // 	pushl %esi
  // L0$pb:
  // 	subl $0x386c, %esp
  //     calll L1
  // L1:
  //     popl %ecx
  //     movl %ecx, 0x8(%esp)
  //     subl $0x8, %esp
  //     pushl %eax
  //     pushl 0x20(%esp)
  //     calll _puts
  //     addl $0x10, %esp
  //     incl %ebx
  //     addl $0x386c, %esp
  //     popl %esi
  //     popl %edi
  //     popl %ebx
  //     popl %ebp
  //     retl
  //
  // 	.section	__TEXT,__cstring,cstring_literals
  // L_.str:                                 ## @.str
  // 	.asciz	"HI"
  //
  //
  // .subsections_via_symbols

  uint8_t data[] = {
      0x55,
      // offset 0 -- pushl %ebp

      0x53,
      // offset 1 -- pushl %ebx

      0x57,
      // offset 2 -- pushl %edi

      0x56,
      // offset 3 -- pushl %esi

      0x81, 0xec, 0x6c, 0x38, 0x00, 0x00,
      // offset 4 -- subl $0x386c, %esp

      0xe8, 0x00, 0x00, 0x00, 0x00,
      // offset 10 -- calll 0
      // call the next instruction, to put the pc on the stack

      0x59,
      // offset 15 -- popl %ecx
      // pop the saved pc address into ecx

      0x89, 0x4c, 0x24, 0x08,
      // offset 16 -- movl %ecx, 0x8(%esp)

      // ....

      0x83, 0xec, 0x08,
      // offset 20 -- subl $0x8, %esp

      0x50,
      // offset 23 -- pushl %eax

      0xff, 0x74, 0x24, 0x20,
      // offset 24 -- pushl 0x20(%esp)

      0xe8, 0x8c, 0x00, 0x00, 0x00,
      // offset 28 -- calll puts

      0x83, 0xc4, 0x10,
      // offset 33 -- addl $0x10, %esp
      // get esp back to the value it was before the
      // alignment & argument saves for the puts call

      0x43,
      // offset 36 -- incl %ebx

      // ....

      0x81, 0xc4, 0x6c, 0x38, 0x00, 0x00,
      // offset 37 -- addl $0x386c, %esp

      0x5e,
      // offset 43 -- popl %esi

      0x5f,
      // offset 44 -- popl %edi

      0x5b,
      // offset 45 -- popl %ebx

      0x5d,
      // offset 46 -- popl %ebp

      0xc3,
      // offset 47 -- retl

      0xe8, 0x12, 0x34, 0x56, 0x78,
      // offset 48 -- calll __stack_chk_fail
  };

  AddressRange sample_range(0x1000, sizeof(data));

  UnwindPlan unwind_plan(eRegisterKindLLDB);
  EXPECT_TRUE(engine->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  // Unwind rules should look like
  //
  //   0: CFA=esp +4 => esp=CFA+0 eip=[CFA-4]
  //   1: CFA=esp +8 => ebp=[CFA-8] esp=CFA+0 eip=[CFA-4]
  //   2: CFA=esp+12 => ebx=[CFA-12] ebp=[CFA-8] esp=CFA+0 eip=[CFA-4]
  //   3: CFA=esp+16 => ebx=[CFA-12] edi=[CFA-16] ebp=[CFA-8] esp=CFA+0
  //   eip=[CFA-4]
  //   4: CFA=esp+20 => ebx=[CFA-12] edi=[CFA-16] esi=[CFA-20] ebp=[CFA-8]
  //   esp=CFA+0 eip=[CFA-4]
  //  10: CFA=esp+14464 => ebx=[CFA-12] edi=[CFA-16] esi=[CFA-20] ebp=[CFA-8]
  //  esp=CFA+0 eip=[CFA-4]
  //  15: CFA=esp+14468 => ebx=[CFA-12] edi=[CFA-16] esi=[CFA-20] ebp=[CFA-8]
  //  esp=CFA+0 eip=[CFA-4]
  //  16: CFA=esp+14464 => ebx=[CFA-12] edi=[CFA-16] esi=[CFA-20] ebp=[CFA-8]
  //  esp=CFA+0 eip=[CFA-4]
  //
  //  ....
  //
  //  23: CFA=esp+14472 => ebx=[CFA-12] edi=[CFA-16] esi=[CFA-20] ebp=[CFA-8]
  //  esp=CFA+0 eip=[CFA-4]
  //  24: CFA=esp+14476 => ebx=[CFA-12] edi=[CFA-16] esi=[CFA-20] ebp=[CFA-8]
  //  esp=CFA+0 eip=[CFA-4]
  //  28: CFA=esp+14480 => ebx=[CFA-12] edi=[CFA-16] esi=[CFA-20] ebp=[CFA-8]
  //  esp=CFA+0 eip=[CFA-4]
  //  36: CFA=esp+14464 => ebx=[CFA-12] edi=[CFA-16] esi=[CFA-20] ebp=[CFA-8]
  //  esp=CFA+0 eip=[CFA-4]
  //
  //  .....
  //
  //  37: CFA=esp+14464 => ebx=[CFA-12] edi=[CFA-16] esi=[CFA-20] ebp=[CFA-8]
  //  esp=CFA+0 eip=[CFA-4]
  //  43: CFA=esp+20 => ebx=[CFA-12] edi=[CFA-16] esi=[CFA-20] ebp=[CFA-8]
  //  esp=CFA+0 eip=[CFA-4]
  //  44: CFA=esp+16 => ebx=[CFA-12] edi=[CFA-16] ebp=[CFA-8] esp=CFA+0
  //  eip=[CFA-4]
  //  45: CFA=esp+12 => ebx=[CFA-12] ebp=[CFA-8] esp=CFA+0 eip=[CFA-4]
  //  46: CFA=esp +8 => ebp=[CFA-8] esp=CFA+0 eip=[CFA-4]
  //  47: CFA=esp +4 => esp=CFA+0 eip=[CFA-4]
  //  48: CFA=esp+14480 => ebx=[CFA-12] edi=[CFA-16] esi=[CFA-20] ebp=[CFA-8]
  //  esp=CFA+0 eip=[CFA-4]

  UnwindPlan::Row::RegisterLocation regloc;
  UnwindPlan::RowSP row_sp;

  // Check that we get the CFA correct for the pic base setup sequence

  // CFA=esp+14464 => ebx=[CFA-12] edi=[CFA-16] esi=[CFA-20] ebp=[CFA-8]
  // esp=CFA+0 eip=[CFA-4]
  row_sp = unwind_plan.GetRowForFunctionOffset(10);
  EXPECT_EQ(10ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_esp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(14464, row_sp->GetCFAValue().GetOffset());

  // 15: CFA=esp+14468 => ebx=[CFA-12] edi=[CFA-16] esi=[CFA-20] ebp=[CFA-8]
  // esp=CFA+0 eip=[CFA-4]
  row_sp = unwind_plan.GetRowForFunctionOffset(15);
  EXPECT_EQ(15ull, row_sp->GetOffset());
  EXPECT_EQ(14468, row_sp->GetCFAValue().GetOffset());

  // 16: CFA=esp+14464 => ebx=[CFA-12] edi=[CFA-16] esi=[CFA-20] ebp=[CFA-8]
  // esp=CFA+0 eip=[CFA-4]
  row_sp = unwind_plan.GetRowForFunctionOffset(16);
  EXPECT_EQ(16ull, row_sp->GetOffset());
  EXPECT_EQ(14464, row_sp->GetCFAValue().GetOffset());

  // Check that the row for offset 16 has the registers saved that we expect

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_eip, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-4, regloc.GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_ebp, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_ebx, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-12, regloc.GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_edi, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-16, regloc.GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_esi, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-20, regloc.GetOffset());

  //
  // Check the pushing & popping around the call printf instruction

  // 23: CFA=esp+14472 => ebx=[CFA-12] edi=[CFA-16] esi=[CFA-20] ebp=[CFA-8]
  // esp=CFA+0 eip=[CFA-4]
  row_sp = unwind_plan.GetRowForFunctionOffset(23);
  EXPECT_EQ(23ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_esp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(14472, row_sp->GetCFAValue().GetOffset());

  // 24: CFA=esp+14476 => ebx=[CFA-12] edi=[CFA-16] esi=[CFA-20] ebp=[CFA-8]
  // esp=CFA+0 eip=[CFA-4]
  row_sp = unwind_plan.GetRowForFunctionOffset(24);
  EXPECT_EQ(24ull, row_sp->GetOffset());
  EXPECT_EQ(14476, row_sp->GetCFAValue().GetOffset());

  // 28: CFA=esp+14480 => ebx=[CFA-12] edi=[CFA-16] esi=[CFA-20] ebp=[CFA-8]
  // esp=CFA+0 eip=[CFA-4]
  row_sp = unwind_plan.GetRowForFunctionOffset(28);
  EXPECT_EQ(28ull, row_sp->GetOffset());
  EXPECT_EQ(14480, row_sp->GetCFAValue().GetOffset());

  // 36: CFA=esp+14464 => ebx=[CFA-12] edi=[CFA-16] esi=[CFA-20] ebp=[CFA-8]
  // esp=CFA+0 eip=[CFA-4]
  row_sp = unwind_plan.GetRowForFunctionOffset(36);
  EXPECT_EQ(36ull, row_sp->GetOffset());
  EXPECT_EQ(14464, row_sp->GetCFAValue().GetOffset());

  // Check that the epilogue gets us back to the original unwind state

  //  47: CFA=esp +4 => esp=CFA+0 eip=[CFA-4]
  row_sp = unwind_plan.GetRowForFunctionOffset(47);
  EXPECT_EQ(47ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_esp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(4, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_eip, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-4, regloc.GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_esp, regloc));
  EXPECT_TRUE(regloc.IsCFAPlusOffset());
  EXPECT_EQ(0, regloc.GetOffset());

  // Check that no unexpected registers were saved

  EXPECT_FALSE(row_sp->GetRegisterInfo(k_eax, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_ebx, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_ecx, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_edx, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_esi, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_edi, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_ebp, regloc));
}

TEST_F(Testx86AssemblyInspectionEngine, Test64bitFramelessSmallStackFrame) {
  std::unique_ptr<x86AssemblyInspectionEngine> engine = Getx86_64Inspector();

  // this source file:
  // #include <stdio.h>
  // int main () {
  //    puts ("HI");
  // }
  //
  // compiled 'clang -fomit-frame-pointer' for x86_64-apple-macosx

  uint8_t data[] = {
      0x50,
      // offset 0  -- pushq %rax

      0x48, 0x8d, 0x3d, 0x32, 0x00, 0x00, 0x00,
      // offset 1 -- leaq 0x32(%rip), %rdi ; "HI"

      0xe8, 0x0b, 0x00, 0x00, 0x00,
      // offset 8 -- callq 0x100000f58 ; puts

      0x31, 0xc9,
      // offset 13 -- xorl %ecx, %ecx

      0x89, 0x44, 0x24, 0x04,
      // offset 15 -- movl %eax, 0x4(%rsp)

      0x89, 0xc8,
      // offset 19 -- movl %ecx, %eax

      0x59,
      // offset 21 -- popq %rcx

      0xc3
      // offset 22 -- retq
  };

  AddressRange sample_range(0x1000, sizeof(data));

  UnwindPlan unwind_plan(eRegisterKindLLDB);
  EXPECT_TRUE(engine->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  // Unwind rules should look like
  //     0: CFA=rsp +8 => rsp=CFA+0 rip=[CFA-8]
  //     1: CFA=rsp+16 => rsp=CFA+0 rip=[CFA-8]
  //    22: CFA=rsp +8 => rsp=CFA+0 rip=[CFA-8]

  UnwindPlan::Row::RegisterLocation regloc;

  // grab the Row for when the prologue has finished executing:
  //     1: CFA=rsp+16 => rsp=CFA+0 rip=[CFA-8]

  UnwindPlan::RowSP row_sp = unwind_plan.GetRowForFunctionOffset(13);

  EXPECT_EQ(1ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_rip, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());

  // none of these were spilled

  EXPECT_FALSE(row_sp->GetRegisterInfo(k_rax, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_rbx, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_rcx, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_rdx, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_rbp, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_rsi, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_rdi, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_r8, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_r9, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_r10, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_r11, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_r12, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_r13, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_r14, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_r15, regloc));

  // grab the Row for when the epilogue has finished executing:
  //     22: CFA=rsp +8 => rsp=CFA+0 rip=[CFA-8]

  row_sp = unwind_plan.GetRowForFunctionOffset(22);

  EXPECT_EQ(22ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(8, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_rip, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());
}

TEST_F(Testx86AssemblyInspectionEngine, Test32bitFramelessSmallStackFrame) {
  std::unique_ptr<x86AssemblyInspectionEngine> engine = Geti386Inspector();

  // this source file:
  // #include <stdio.h>
  // int main () {
  //    puts ("HI");
  // }
  //
  // compiled 'clang -arch i386 -fomit-frame-pointer' for i386-apple-macosx

  uint8_t data[] = {
      0x83, 0xec, 0x0c,
      // offset 0 -- subl $0xc, %esp

      0xe8, 0x00, 0x00, 0x00, 0x00,
      // offset 3 -- calll 0 {call the next instruction, to put the pc on
      // the stack}

      0x58,
      // offset 8 -- popl %eax {pop the saved pc value off stack, into eax}

      0x8d, 0x80, 0x3a, 0x00, 0x00, 0x00,
      // offset 9 -- leal 0x3a(%eax),%eax

      0x89, 0x04, 0x24,
      // offset 15 -- movl %eax, (%esp)

      0xe8, 0x0d, 0x00, 0x00, 0x00,
      // offset 18 -- calll 0x1f94 (puts)

      0x31, 0xc9,
      // offset 23 -- xorl %ecx, %ecx

      0x89, 0x44, 0x24, 0x08,
      // offset 25 -- movl %eax, 0x8(%esp)

      0x89, 0xc8,
      // offset 29 -- movl %ecx, %eax

      0x83, 0xc4, 0x0c,
      // offset 31 -- addl $0xc, %esp

      0xc3
      // offset 34 -- retl
  };

  AddressRange sample_range(0x1000, sizeof(data));

  UnwindPlan unwind_plan(eRegisterKindLLDB);
  EXPECT_TRUE(engine->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  // Unwind rules should look like
  // row[0]:    0: CFA=esp +4 => esp=CFA+0 eip=[CFA-4]
  // row[1]:    3: CFA=esp+16 => esp=CFA+0 eip=[CFA-4]
  // row[2]:    8: CFA=esp+20 => esp=CFA+0 eip=[CFA-4]
  // row[3]:    9: CFA=esp+16 => esp=CFA+0 eip=[CFA-4]
  // row[4]:   34: CFA=esp +4 => esp=CFA+0 eip=[CFA-4]

  UnwindPlan::Row::RegisterLocation regloc;

  // Check unwind state before we set up the picbase register
  //      3: CFA=esp+16 => esp=CFA+0 eip=[CFA-4]

  UnwindPlan::RowSP row_sp = unwind_plan.GetRowForFunctionOffset(3);

  EXPECT_EQ(3ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row_sp->GetCFAValue().GetOffset());

  // Check unwind state after we call the next instruction
  // 8: CFA=esp+20 => esp=CFA+0 eip=[CFA-4]

  row_sp = unwind_plan.GetRowForFunctionOffset(8);
  EXPECT_EQ(8ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(20, row_sp->GetCFAValue().GetOffset());

  // Check unwind state after we pop the pic base value off the stack
  // row[3]:    9: CFA=esp+16 => esp=CFA+0 eip=[CFA-4]

  row_sp = unwind_plan.GetRowForFunctionOffset(9);
  EXPECT_EQ(9ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row_sp->GetCFAValue().GetOffset());

  // Check that no unexpected registers were saved

  EXPECT_FALSE(row_sp->GetRegisterInfo(k_eax, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_ebx, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_ecx, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_edx, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_esi, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_edi, regloc));
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_ebp, regloc));

  // verify that we get back to the original unwind state before the ret
  //  34: CFA=esp +4 => esp=CFA+0 eip=[CFA-4]

  row_sp = unwind_plan.GetRowForFunctionOffset(34);
  EXPECT_EQ(34ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(4, row_sp->GetCFAValue().GetOffset());
}

TEST_F(Testx86AssemblyInspectionEngine, TestPushRBP) {
  UnwindPlan::Row::RegisterLocation regloc;
  UnwindPlan::RowSP row_sp;

  uint8_t data[] = {
      0x55, // pushq %rbp
      0x90  // nop
  };

  AddressRange sample_range(0x1000, sizeof(data));
  UnwindPlan unwind_plan(eRegisterKindLLDB);

  std::unique_ptr<x86AssemblyInspectionEngine> engine64 = Getx86_64Inspector();
  EXPECT_TRUE(engine64->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(1);

  EXPECT_EQ(1ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_rbp, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-16, regloc.GetOffset());

  std::unique_ptr<x86AssemblyInspectionEngine> engine32 = Geti386Inspector();
  EXPECT_TRUE(engine32->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(1);

  EXPECT_EQ(1ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(8, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_rbp, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());
}

TEST_F(Testx86AssemblyInspectionEngine, TestPushImm) {
  UnwindPlan::Row::RegisterLocation regloc;
  UnwindPlan::RowSP row_sp;

  uint8_t data[] = {
      0x68, 0xff, 0xff, 0x01, 0x69, // pushq $0x6901ffff
      0x6a, 0x7d,                   // pushl $0x7d
      0x90                          // nop
  };

  AddressRange sample_range(0x1000, sizeof(data));
  UnwindPlan unwind_plan(eRegisterKindLLDB);

  std::unique_ptr<x86AssemblyInspectionEngine> engine64 = Getx86_64Inspector();
  EXPECT_TRUE(engine64->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(5);
  EXPECT_EQ(5ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row_sp->GetCFAValue().GetOffset());

  row_sp = unwind_plan.GetRowForFunctionOffset(7);
  EXPECT_EQ(7ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(24, row_sp->GetCFAValue().GetOffset());

  std::unique_ptr<x86AssemblyInspectionEngine> engine32 = Geti386Inspector();
  EXPECT_TRUE(engine32->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(5);
  EXPECT_EQ(5ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(8, row_sp->GetCFAValue().GetOffset());

  row_sp = unwind_plan.GetRowForFunctionOffset(7);
  EXPECT_EQ(7ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(12, row_sp->GetCFAValue().GetOffset());
}

// We treat 'pushq $0' / 'pushl $0' specially - this shows up
// in the first function called in a new thread and it needs to
// put a 0 as the saved pc.  We pretend it didn't change the CFA.
TEST_F(Testx86AssemblyInspectionEngine, TestPush0) {
  UnwindPlan::Row::RegisterLocation regloc;
  UnwindPlan::RowSP row_sp;

  uint8_t data[] = {
      0x6a, 0x00, // pushq $0
      0x90        // nop
  };

  AddressRange sample_range(0x1000, sizeof(data));
  UnwindPlan unwind_plan(eRegisterKindLLDB);

  std::unique_ptr<x86AssemblyInspectionEngine> engine64 = Getx86_64Inspector();
  EXPECT_TRUE(engine64->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(2);

  // We're verifying that no row was created for the 'pushq $0'
  EXPECT_EQ(0ull, row_sp->GetOffset());

  std::unique_ptr<x86AssemblyInspectionEngine> engine32 = Geti386Inspector();
  EXPECT_TRUE(engine32->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(2);

  // We're verifying that no row was created for the 'pushq $0'
  EXPECT_EQ(0ull, row_sp->GetOffset());
}

TEST_F(Testx86AssemblyInspectionEngine, TestPushExtended) {
  UnwindPlan::Row::RegisterLocation regloc;
  UnwindPlan::RowSP row_sp;

  uint8_t data[] = {
      0xff, 0x74, 0x24, 0x20,             // pushl 0x20(%esp)
      0xff, 0xb6, 0xce, 0x01, 0xf0, 0x00, // pushl  0xf001ce(%esi)
      0xff, 0x30,                         // pushl  (%eax)
      0x90                                // nop
  };

  AddressRange sample_range(0x1000, sizeof(data));
  UnwindPlan unwind_plan(eRegisterKindLLDB);

  std::unique_ptr<x86AssemblyInspectionEngine> engine64 = Getx86_64Inspector();
  EXPECT_TRUE(engine64->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(4);

  EXPECT_EQ(4ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row_sp->GetCFAValue().GetOffset());

  std::unique_ptr<x86AssemblyInspectionEngine> engine32 = Geti386Inspector();
  EXPECT_TRUE(engine32->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(4);
  EXPECT_EQ(4ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(8, row_sp->GetCFAValue().GetOffset());

  row_sp = unwind_plan.GetRowForFunctionOffset(10);
  EXPECT_EQ(10ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(12, row_sp->GetCFAValue().GetOffset());

  row_sp = unwind_plan.GetRowForFunctionOffset(12);
  EXPECT_EQ(12ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row_sp->GetCFAValue().GetOffset());
}

TEST_F(Testx86AssemblyInspectionEngine, TestPushR15) {
  UnwindPlan::Row::RegisterLocation regloc;
  UnwindPlan::RowSP row_sp;

  uint8_t data[] = {
      0x41, 0x57, // pushq %r15
      0x90        // nop
  };

  AddressRange sample_range(0x1000, sizeof(data));
  UnwindPlan unwind_plan(eRegisterKindLLDB);

  std::unique_ptr<x86AssemblyInspectionEngine> engine64 = Getx86_64Inspector();
  EXPECT_TRUE(engine64->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(2);

  EXPECT_EQ(2ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_r15, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-16, regloc.GetOffset());
}

TEST_F(Testx86AssemblyInspectionEngine, TestPushR14) {
  UnwindPlan::Row::RegisterLocation regloc;
  UnwindPlan::RowSP row_sp;

  uint8_t data[] = {
      0x41, 0x56, // pushq %r14
      0x90        // nop
  };

  AddressRange sample_range(0x1000, sizeof(data));
  UnwindPlan unwind_plan(eRegisterKindLLDB);

  std::unique_ptr<x86AssemblyInspectionEngine> engine64 = Getx86_64Inspector();
  EXPECT_TRUE(engine64->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(2);

  EXPECT_EQ(2ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_r14, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-16, regloc.GetOffset());
}

TEST_F(Testx86AssemblyInspectionEngine, TestPushR13) {
  UnwindPlan::Row::RegisterLocation regloc;
  UnwindPlan::RowSP row_sp;

  uint8_t data[] = {
      0x41, 0x55, // pushq %r13
      0x90        // nop
  };

  AddressRange sample_range(0x1000, sizeof(data));
  UnwindPlan unwind_plan(eRegisterKindLLDB);

  std::unique_ptr<x86AssemblyInspectionEngine> engine64 = Getx86_64Inspector();
  EXPECT_TRUE(engine64->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(2);

  EXPECT_EQ(2ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_r13, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-16, regloc.GetOffset());
}

TEST_F(Testx86AssemblyInspectionEngine, TestPushR12) {
  UnwindPlan::Row::RegisterLocation regloc;
  UnwindPlan::RowSP row_sp;

  uint8_t data[] = {
      0x41, 0x54, // pushq %r13
      0x90        // nop
  };

  AddressRange sample_range(0x1000, sizeof(data));
  UnwindPlan unwind_plan(eRegisterKindLLDB);

  std::unique_ptr<x86AssemblyInspectionEngine> engine64 = Getx86_64Inspector();
  EXPECT_TRUE(engine64->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(2);

  EXPECT_EQ(2ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_r12, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-16, regloc.GetOffset());
}

TEST_F(Testx86AssemblyInspectionEngine, TestPushRBX) {
  UnwindPlan::Row::RegisterLocation regloc;
  UnwindPlan::RowSP row_sp;

  uint8_t data[] = {
      0x53, // pushq %rbx
      0x90  // nop
  };

  AddressRange sample_range(0x1000, sizeof(data));
  UnwindPlan unwind_plan(eRegisterKindLLDB);

  std::unique_ptr<x86AssemblyInspectionEngine> engine64 = Getx86_64Inspector();
  EXPECT_TRUE(engine64->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(1);

  EXPECT_EQ(1ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(16, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_rbx, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-16, regloc.GetOffset());
}

// The ABI is hardcoded in x86AssemblyInspectionEngine such that
// eax, ecx, edx are all considered volatile and push/pops of them are
// not tracked (except to keep track of stack pointer movement)
TEST_F(Testx86AssemblyInspectionEngine, TestPushEAX) {
  UnwindPlan::Row::RegisterLocation regloc;
  UnwindPlan::RowSP row_sp;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  std::unique_ptr<x86AssemblyInspectionEngine> engine32 = Geti386Inspector();

  uint8_t data[] = {
      0x50, // pushl %eax
      0x90  // nop
  };

  sample_range = AddressRange(0x1000, sizeof(data));

  EXPECT_TRUE(engine32->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(1);
  EXPECT_EQ(1ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_esp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(8, row_sp->GetCFAValue().GetOffset());

  EXPECT_FALSE(row_sp->GetRegisterInfo(k_eax, regloc));
}

// The ABI is hardcoded in x86AssemblyInspectionEngine such that
// eax, ecx, edx are all considered volatile and push/pops of them are
// not tracked (except to keep track of stack pointer movement)
TEST_F(Testx86AssemblyInspectionEngine, TestPushECX) {
  UnwindPlan::Row::RegisterLocation regloc;
  UnwindPlan::RowSP row_sp;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  std::unique_ptr<x86AssemblyInspectionEngine> engine32 = Geti386Inspector();

  uint8_t data[] = {
      0x51, // pushl %ecx
      0x90  // nop
  };

  sample_range = AddressRange(0x1000, sizeof(data));

  EXPECT_TRUE(engine32->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(1);
  EXPECT_EQ(1ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_esp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(8, row_sp->GetCFAValue().GetOffset());

  EXPECT_FALSE(row_sp->GetRegisterInfo(k_ecx, regloc));
}

// The ABI is hardcoded in x86AssemblyInspectionEngine such that
// eax, ecx, edx are all considered volatile and push/pops of them are
// not tracked (except to keep track of stack pointer movement)
TEST_F(Testx86AssemblyInspectionEngine, TestPushEDX) {
  UnwindPlan::Row::RegisterLocation regloc;
  UnwindPlan::RowSP row_sp;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  std::unique_ptr<x86AssemblyInspectionEngine> engine32 = Geti386Inspector();

  uint8_t data[] = {
      0x52, // pushl %edx
      0x90  // nop
  };

  sample_range = AddressRange(0x1000, sizeof(data));

  EXPECT_TRUE(engine32->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(1);
  EXPECT_EQ(1ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_esp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(8, row_sp->GetCFAValue().GetOffset());

  EXPECT_FALSE(row_sp->GetRegisterInfo(k_edx, regloc));
}

TEST_F(Testx86AssemblyInspectionEngine, TestPushEBX) {
  UnwindPlan::Row::RegisterLocation regloc;
  UnwindPlan::RowSP row_sp;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  std::unique_ptr<x86AssemblyInspectionEngine> engine32 = Geti386Inspector();

  uint8_t data[] = {
      0x53, // pushl %ebx
      0x90  // nop
  };

  sample_range = AddressRange(0x1000, sizeof(data));

  EXPECT_TRUE(engine32->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(1);
  EXPECT_EQ(1ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_esp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(8, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_ebx, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());
}

TEST_F(Testx86AssemblyInspectionEngine, TestPushEBP) {
  UnwindPlan::Row::RegisterLocation regloc;
  UnwindPlan::RowSP row_sp;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  std::unique_ptr<x86AssemblyInspectionEngine> engine32 = Geti386Inspector();

  uint8_t data[] = {
      0x55, // pushl %ebp
      0x90  // nop
  };

  sample_range = AddressRange(0x1000, sizeof(data));

  EXPECT_TRUE(engine32->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(1);
  EXPECT_EQ(1ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_esp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(8, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_ebp, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());
}

TEST_F(Testx86AssemblyInspectionEngine, TestPushESI) {
  UnwindPlan::Row::RegisterLocation regloc;
  UnwindPlan::RowSP row_sp;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  std::unique_ptr<x86AssemblyInspectionEngine> engine32 = Geti386Inspector();

  uint8_t data[] = {
      0x56, // pushl %esi
      0x90  // nop
  };

  sample_range = AddressRange(0x1000, sizeof(data));

  EXPECT_TRUE(engine32->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(1);
  EXPECT_EQ(1ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_esp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(8, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_esi, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());
}

TEST_F(Testx86AssemblyInspectionEngine, TestPushEDI) {
  UnwindPlan::Row::RegisterLocation regloc;
  UnwindPlan::RowSP row_sp;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  std::unique_ptr<x86AssemblyInspectionEngine> engine32 = Geti386Inspector();

  uint8_t data[] = {
      0x57, // pushl %edi
      0x90  // nop
  };

  sample_range = AddressRange(0x1000, sizeof(data));

  EXPECT_TRUE(engine32->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(1);
  EXPECT_EQ(1ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_esp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(8, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_edi, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());
}

TEST_F(Testx86AssemblyInspectionEngine, TestMovRSPtoRBP) {
  UnwindPlan::Row::RegisterLocation regloc;
  UnwindPlan::RowSP row_sp;

  uint8_t data64_1[] = {
      0x48, 0x8b, 0xec, // movq %rsp, %rbp
      0x90              // nop
  };

  AddressRange sample_range(0x1000, sizeof(data64_1));
  UnwindPlan unwind_plan(eRegisterKindLLDB);

  std::unique_ptr<x86AssemblyInspectionEngine> engine64 = Getx86_64Inspector();
  EXPECT_TRUE(engine64->GetNonCallSiteUnwindPlanFromAssembly(
      data64_1, sizeof(data64_1), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(3);

  EXPECT_EQ(3ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rbp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(8, row_sp->GetCFAValue().GetOffset());

  uint8_t data64_2[] = {
      0x48, 0x89, 0xe5, // movq %rsp, %rbp
      0x90              // nop
  };

  sample_range = AddressRange(0x1000, sizeof(data64_2));
  unwind_plan.Clear();
  EXPECT_TRUE(engine64->GetNonCallSiteUnwindPlanFromAssembly(
      data64_2, sizeof(data64_2), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(3);
  EXPECT_EQ(3ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rbp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(8, row_sp->GetCFAValue().GetOffset());

  uint8_t data32_1[] = {
      0x8b, 0xec, // movl %rsp, %rbp
      0x90        // nop
  };

  sample_range = AddressRange(0x1000, sizeof(data32_1));
  unwind_plan.Clear();
  EXPECT_TRUE(engine64->GetNonCallSiteUnwindPlanFromAssembly(
      data32_1, sizeof(data32_1), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(2);
  EXPECT_EQ(2ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_ebp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(8, row_sp->GetCFAValue().GetOffset());

  uint8_t data32_2[] = {
      0x89, 0xe5, // movl %rsp, %rbp
      0x90        // nop
  };

  sample_range = AddressRange(0x1000, sizeof(data32_2));
  unwind_plan.Clear();
  EXPECT_TRUE(engine64->GetNonCallSiteUnwindPlanFromAssembly(
      data32_2, sizeof(data32_2), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(2);
  EXPECT_EQ(2ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_ebp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(8, row_sp->GetCFAValue().GetOffset());
}

TEST_F(Testx86AssemblyInspectionEngine, TestSubRSP) {
  UnwindPlan::Row::RegisterLocation regloc;
  UnwindPlan::RowSP row_sp;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  std::unique_ptr<x86AssemblyInspectionEngine> engine64 = Getx86_64Inspector();

  uint8_t data1[] = {
      0x48, 0x81, 0xec, 0x00, 0x01, 0x00, 0x00, // subq $0x100, $rsp
      0x90                                      // nop
  };

  sample_range = AddressRange(0x1000, sizeof(data1));

  EXPECT_TRUE(engine64->GetNonCallSiteUnwindPlanFromAssembly(
      data1, sizeof(data1), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(7);
  EXPECT_EQ(7ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(264, row_sp->GetCFAValue().GetOffset());

  uint8_t data2[] = {
      0x48, 0x83, 0xec, 0x10, // subq $0x10, %rsp
      0x90                    // nop
  };

  sample_range = AddressRange(0x1000, sizeof(data2));

  EXPECT_TRUE(engine64->GetNonCallSiteUnwindPlanFromAssembly(
      data2, sizeof(data2), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(4);
  EXPECT_EQ(4ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(24, row_sp->GetCFAValue().GetOffset());
}

TEST_F(Testx86AssemblyInspectionEngine, TestSubESP) {
  UnwindPlan::Row::RegisterLocation regloc;
  UnwindPlan::RowSP row_sp;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  std::unique_ptr<x86AssemblyInspectionEngine> engine32 = Geti386Inspector();

  uint8_t data1[] = {
      0x81, 0xec, 0x00, 0x01, 0x00, 0x00, // subl $0x100, %esp
      0x90                                // nop
  };

  sample_range = AddressRange(0x1000, sizeof(data1));

  EXPECT_TRUE(engine32->GetNonCallSiteUnwindPlanFromAssembly(
      data1, sizeof(data1), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(6);
  EXPECT_EQ(6ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(260, row_sp->GetCFAValue().GetOffset());

  uint8_t data2[] = {
      0x83, 0xec, 0x10, // subq $0x10, %esp
      0x90              // nop
  };

  sample_range = AddressRange(0x1000, sizeof(data2));

  EXPECT_TRUE(engine32->GetNonCallSiteUnwindPlanFromAssembly(
      data2, sizeof(data2), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(3);
  EXPECT_EQ(3ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(20, row_sp->GetCFAValue().GetOffset());
}

TEST_F(Testx86AssemblyInspectionEngine, TestAddRSP) {
  UnwindPlan::Row::RegisterLocation regloc;
  UnwindPlan::RowSP row_sp;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  std::unique_ptr<x86AssemblyInspectionEngine> engine64 = Getx86_64Inspector();

  uint8_t data1[] = {
      0x48, 0x81, 0xc4, 0x00, 0x01, 0x00, 0x00, // addq $0x100, %rsp
      0x90                                      // nop
  };

  sample_range = AddressRange(0x1000, sizeof(data1));

  EXPECT_TRUE(engine64->GetNonCallSiteUnwindPlanFromAssembly(
      data1, sizeof(data1), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(7);
  EXPECT_EQ(7ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(8 - 256, row_sp->GetCFAValue().GetOffset());

  uint8_t data2[] = {
      0x48, 0x83, 0xc4, 0x10, // addq $0x10, %rsp
      0x90                    // nop
  };

  sample_range = AddressRange(0x1000, sizeof(data2));

  EXPECT_TRUE(engine64->GetNonCallSiteUnwindPlanFromAssembly(
      data2, sizeof(data2), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(4);
  EXPECT_EQ(4ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(8 - 16, row_sp->GetCFAValue().GetOffset());
}

TEST_F(Testx86AssemblyInspectionEngine, TestAddESP) {
  UnwindPlan::Row::RegisterLocation regloc;
  UnwindPlan::RowSP row_sp;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  std::unique_ptr<x86AssemblyInspectionEngine> engine32 = Geti386Inspector();

  uint8_t data1[] = {
      0x81, 0xc4, 0x00, 0x01, 0x00, 0x00, // addl $0x100, %esp
      0x90                                // nop
  };

  sample_range = AddressRange(0x1000, sizeof(data1));

  EXPECT_TRUE(engine32->GetNonCallSiteUnwindPlanFromAssembly(
      data1, sizeof(data1), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(6);
  EXPECT_EQ(6ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(4 - 256, row_sp->GetCFAValue().GetOffset());

  uint8_t data2[] = {
      0x83, 0xc4, 0x10, // addq $0x10, %esp
      0x90              // nop
  };

  sample_range = AddressRange(0x1000, sizeof(data2));

  EXPECT_TRUE(engine32->GetNonCallSiteUnwindPlanFromAssembly(
      data2, sizeof(data2), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(3);
  EXPECT_EQ(3ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(4 - 16, row_sp->GetCFAValue().GetOffset());
}

// FIXME add test for lea_rsp_pattern_p

TEST_F(Testx86AssemblyInspectionEngine, TestPopRBX) {
  UnwindPlan::Row::RegisterLocation regloc;
  UnwindPlan::RowSP row_sp;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  std::unique_ptr<x86AssemblyInspectionEngine> engine = Getx86_64Inspector();

  uint8_t data[] = {
      0x53, // pushq %rbx
      0x5b, // popq %rbx
      0x90  // nop
  };

  sample_range = AddressRange(0x1000, sizeof(data));

  EXPECT_TRUE(engine->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(2);
  EXPECT_EQ(2ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(8, row_sp->GetCFAValue().GetOffset());
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_rbx, regloc));
}

TEST_F(Testx86AssemblyInspectionEngine, TestPopRBP) {
  UnwindPlan::Row::RegisterLocation regloc;
  UnwindPlan::RowSP row_sp;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  std::unique_ptr<x86AssemblyInspectionEngine> engine = Getx86_64Inspector();

  uint8_t data[] = {
      0x55, // pushq %rbp
      0x5d, // popq %rbp
      0x90  // nop
  };

  sample_range = AddressRange(0x1000, sizeof(data));

  EXPECT_TRUE(engine->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(2);
  EXPECT_EQ(2ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(8, row_sp->GetCFAValue().GetOffset());
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_rbp, regloc));
}

TEST_F(Testx86AssemblyInspectionEngine, TestPopR12) {
  UnwindPlan::Row::RegisterLocation regloc;
  UnwindPlan::RowSP row_sp;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  std::unique_ptr<x86AssemblyInspectionEngine> engine = Getx86_64Inspector();

  uint8_t data[] = {
      0x41, 0x54, // pushq %r12
      0x41, 0x5c, // popq %r12
      0x90        // nop
  };

  sample_range = AddressRange(0x1000, sizeof(data));

  EXPECT_TRUE(engine->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(4);
  EXPECT_EQ(4ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(8, row_sp->GetCFAValue().GetOffset());
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_r12, regloc));
}

TEST_F(Testx86AssemblyInspectionEngine, TestPopR13) {
  UnwindPlan::Row::RegisterLocation regloc;
  UnwindPlan::RowSP row_sp;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  std::unique_ptr<x86AssemblyInspectionEngine> engine = Getx86_64Inspector();

  uint8_t data[] = {
      0x41, 0x55, // pushq %r13
      0x41, 0x5d, // popq %r13
      0x90        // nop
  };

  sample_range = AddressRange(0x1000, sizeof(data));

  EXPECT_TRUE(engine->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(4);
  EXPECT_EQ(4ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(8, row_sp->GetCFAValue().GetOffset());
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_r13, regloc));
}

TEST_F(Testx86AssemblyInspectionEngine, TestPopR14) {
  UnwindPlan::Row::RegisterLocation regloc;
  UnwindPlan::RowSP row_sp;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  std::unique_ptr<x86AssemblyInspectionEngine> engine = Getx86_64Inspector();

  uint8_t data[] = {
      0x41, 0x56, // pushq %r14
      0x41, 0x5e, // popq %r14
      0x90        // nop
  };

  sample_range = AddressRange(0x1000, sizeof(data));

  EXPECT_TRUE(engine->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(4);
  EXPECT_EQ(4ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(8, row_sp->GetCFAValue().GetOffset());
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_r14, regloc));
}

TEST_F(Testx86AssemblyInspectionEngine, TestPopR15) {
  UnwindPlan::Row::RegisterLocation regloc;
  UnwindPlan::RowSP row_sp;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  std::unique_ptr<x86AssemblyInspectionEngine> engine = Getx86_64Inspector();

  uint8_t data[] = {
      0x41, 0x57, // pushq %r15
      0x41, 0x5f, // popq %r15
      0x90        // nop
  };

  sample_range = AddressRange(0x1000, sizeof(data));

  EXPECT_TRUE(engine->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(4);
  EXPECT_EQ(4ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(8, row_sp->GetCFAValue().GetOffset());
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_r15, regloc));
}

TEST_F(Testx86AssemblyInspectionEngine, TestPopEBX) {
  UnwindPlan::Row::RegisterLocation regloc;
  UnwindPlan::RowSP row_sp;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  std::unique_ptr<x86AssemblyInspectionEngine> engine = Geti386Inspector();

  uint8_t data[] = {
      0x53, // pushl %ebx
      0x5b, // popl %ebx
      0x90  // nop
  };

  sample_range = AddressRange(0x1000, sizeof(data));

  EXPECT_TRUE(engine->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(2);
  EXPECT_EQ(2ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(4, row_sp->GetCFAValue().GetOffset());
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_ebx, regloc));
}

TEST_F(Testx86AssemblyInspectionEngine, TestPopEBP) {
  UnwindPlan::Row::RegisterLocation regloc;
  UnwindPlan::RowSP row_sp;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  std::unique_ptr<x86AssemblyInspectionEngine> engine = Geti386Inspector();

  uint8_t data[] = {
      0x55, // pushl %ebp
      0x5d, // popl %ebp
      0x90  // nop
  };

  sample_range = AddressRange(0x1000, sizeof(data));

  EXPECT_TRUE(engine->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(2);
  EXPECT_EQ(2ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(4, row_sp->GetCFAValue().GetOffset());
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_ebp, regloc));
}

TEST_F(Testx86AssemblyInspectionEngine, TestPopESI) {
  UnwindPlan::Row::RegisterLocation regloc;
  UnwindPlan::RowSP row_sp;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  std::unique_ptr<x86AssemblyInspectionEngine> engine = Geti386Inspector();

  uint8_t data[] = {
      0x56, // pushl %esi
      0x5e, // popl %esi
      0x90  // nop
  };

  sample_range = AddressRange(0x1000, sizeof(data));

  EXPECT_TRUE(engine->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(2);
  EXPECT_EQ(2ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(4, row_sp->GetCFAValue().GetOffset());
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_esi, regloc));
}

TEST_F(Testx86AssemblyInspectionEngine, TestPopEDI) {
  UnwindPlan::Row::RegisterLocation regloc;
  UnwindPlan::RowSP row_sp;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  std::unique_ptr<x86AssemblyInspectionEngine> engine = Geti386Inspector();

  uint8_t data[] = {
      0x57, // pushl %edi
      0x5f, // popl %edi
      0x90  // nop
  };

  sample_range = AddressRange(0x1000, sizeof(data));

  EXPECT_TRUE(engine->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(2);
  EXPECT_EQ(2ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(4, row_sp->GetCFAValue().GetOffset());
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_edi, regloc));
}

// We don't track these registers, but make sure the CFA address is updated
// if we're defining the CFA in term of esp.
TEST_F(Testx86AssemblyInspectionEngine, Testi386IgnoredRegisters) {
  UnwindPlan::Row::RegisterLocation regloc;
  UnwindPlan::RowSP row_sp;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  std::unique_ptr<x86AssemblyInspectionEngine> engine = Geti386Inspector();

  uint8_t data[] = {
      0x0e, // push cs
      0x16, // push ss
      0x1e, // push ds
      0x06, // push es

      0x07, // pop es
      0x1f, // pop ds
      0x17, // pop ss

      0x90 // nop
  };

  sample_range = AddressRange(0x1000, sizeof(data));

  EXPECT_TRUE(engine->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(4);
  EXPECT_EQ(4ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(20, row_sp->GetCFAValue().GetOffset());

  row_sp = unwind_plan.GetRowForFunctionOffset(7);
  EXPECT_EQ(7ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(8, row_sp->GetCFAValue().GetOffset());
}

TEST_F(Testx86AssemblyInspectionEngine, TestLEAVE) {
  UnwindPlan::Row::RegisterLocation regloc;
  UnwindPlan::RowSP row_sp;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  std::unique_ptr<x86AssemblyInspectionEngine> engine64 = Getx86_64Inspector();
  std::unique_ptr<x86AssemblyInspectionEngine> engine32 = Geti386Inspector();

  uint8_t data[] = {
      0x55, // push %rbp/ebp
      0xc9, // leave
      0x90  // nop
  };

  sample_range = AddressRange(0x1000, sizeof(data));

  EXPECT_TRUE(engine64->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(2);
  EXPECT_EQ(2ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(8, row_sp->GetCFAValue().GetOffset());
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_rbp, regloc));

  EXPECT_TRUE(engine32->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(2);
  EXPECT_EQ(2ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(4, row_sp->GetCFAValue().GetOffset());
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_ebp, regloc));
}

// In i386, which lacks pc-relative addressing, a common code sequence
// is to call the next instruction (i.e. call imm32, value of 0) which
// pushes the addr of the next insn on the stack, and then pop that value
// into a register (the "pic base" register).
TEST_F(Testx86AssemblyInspectionEngine, TestCALLNextInsn) {
  UnwindPlan::Row::RegisterLocation regloc;
  UnwindPlan::RowSP row_sp;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  std::unique_ptr<x86AssemblyInspectionEngine> engine32 = Geti386Inspector();

  uint8_t data[] = {
      0xe8, 0x00, 0x00, 0x00, 0x00, // call 0
      0x90                          // nop
  };

  sample_range = AddressRange(0x1000, sizeof(data));

  EXPECT_TRUE(engine32->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(5);
  EXPECT_EQ(5ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(8, row_sp->GetCFAValue().GetOffset());
  EXPECT_FALSE(row_sp->GetRegisterInfo(k_ebp, regloc));
}

TEST_F(Testx86AssemblyInspectionEngine, TestSpillRegToStackViaMOVx86_64) {
  UnwindPlan::Row::RegisterLocation regloc;
  UnwindPlan::RowSP row_sp;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  std::unique_ptr<x86AssemblyInspectionEngine> engine64 = Getx86_64Inspector();

  uint8_t data[] = {
      0x55,                                     // pushq %rbp
      0x48, 0x89, 0xe5,                         // movq %rsp, %rbp
      0x4c, 0x89, 0x75, 0xc0,                   // movq   %r14, -0x40(%rbp)
      0x4c, 0x89, 0xbd, 0x28, 0xfa, 0xff, 0xff, // movq   %r15, -0x5d8(%rbp)
      0x48, 0x89, 0x5d, 0xb8,                   // movq %rbx, -0x48(%rbp)
      0x90                                      // nop
  };

  sample_range = AddressRange(0x1000, sizeof(data));

  EXPECT_TRUE(engine64->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(19);
  EXPECT_EQ(19ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rbp);
  EXPECT_EQ(16, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_r14, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-80, regloc.GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_r15, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-1512, regloc.GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_rbx, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-88, regloc.GetOffset());
}

TEST_F(Testx86AssemblyInspectionEngine, TestSpillRegToStackViaMOVi386) {
  UnwindPlan::Row::RegisterLocation regloc;
  UnwindPlan::RowSP row_sp;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  std::unique_ptr<x86AssemblyInspectionEngine> engine32 = Geti386Inspector();

  uint8_t data[] = {
      0x55,                               // pushl %ebp
      0x89, 0xe5,                         // movl %esp, %ebp
      0x89, 0x9d, 0xb0, 0xfe, 0xff, 0xff, // movl %ebx, -0x150(%ebp)
      0x89, 0x75, 0xe0,                   // movl %esi, -0x20(%ebp)
      0x90                                // nop
  };

  sample_range = AddressRange(0x1000, sizeof(data));

  EXPECT_TRUE(engine32->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(12);
  EXPECT_EQ(12ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rbp);
  EXPECT_EQ(8, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_ebx, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-344, regloc.GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_esi, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-40, regloc.GetOffset());
}

TEST_F(Testx86AssemblyInspectionEngine, TestSimplex86_64Augmented) {
  UnwindPlan::Row::RegisterLocation regloc;
  UnwindPlan::RowSP row_sp;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  std::unique_ptr<x86AssemblyInspectionEngine> engine64 = Getx86_64Inspector();

  uint8_t data[] = {
      0x55,             // pushq %rbp
      0x48, 0x89, 0xe5, // movq %rsp, %rbp

      // x86AssemblyInspectionEngine::AugmentUnwindPlanFromCallSite
      // has a bug where it can't augment a function that is just
      // prologue+epilogue - it needs at least one other instruction
      // in between.
      0x90, // nop

      0x5d, // popq %rbp
      0xc3  // retq
  };

  sample_range = AddressRange(0x1000, sizeof(data));

  unwind_plan.SetSourceName("unit testing hand-created unwind plan");
  unwind_plan.SetPlanValidAddressRange(sample_range);
  unwind_plan.SetRegisterKind(eRegisterKindLLDB);

  row_sp.reset(new UnwindPlan::Row);

  // Describe offset 0
  row_sp->SetOffset(0);
  row_sp->GetCFAValue().SetIsRegisterPlusOffset(k_rsp, 8);

  regloc.SetAtCFAPlusOffset(-8);
  row_sp->SetRegisterInfo(k_rip, regloc);

  unwind_plan.AppendRow(row_sp);

  // Allocate a new Row, populate it with the existing Row contents.
  UnwindPlan::Row *new_row = new UnwindPlan::Row;
  *new_row = *row_sp.get();
  row_sp.reset(new_row);

  // Describe offset 1
  row_sp->SetOffset(1);
  row_sp->GetCFAValue().SetIsRegisterPlusOffset(k_rsp, 16);
  regloc.SetAtCFAPlusOffset(-16);
  row_sp->SetRegisterInfo(k_rbp, regloc);
  unwind_plan.AppendRow(row_sp);

  // Allocate a new Row, populate it with the existing Row contents.
  new_row = new UnwindPlan::Row;
  *new_row = *row_sp.get();
  row_sp.reset(new_row);

  // Describe offset 4
  row_sp->SetOffset(4);
  row_sp->GetCFAValue().SetIsRegisterPlusOffset(k_rbp, 16);
  unwind_plan.AppendRow(row_sp);

  RegisterContextSP reg_ctx_sp;
  EXPECT_TRUE(engine64->AugmentUnwindPlanFromCallSite(
      data, sizeof(data), sample_range, unwind_plan, reg_ctx_sp));

  row_sp = unwind_plan.GetRowForFunctionOffset(6);
  EXPECT_EQ(6ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_EQ(8, row_sp->GetCFAValue().GetOffset());

  // x86AssemblyInspectionEngine::AugmentUnwindPlanFromCallSite
  // doesn't track register restores (pop'ing a reg value back from
  // the stack) - it was just written to make stepping work correctly.
  // Technically we should be able to do the following test, but it
  // won't work today - the unwind plan will still say that the caller's
  // rbp is on the stack.
  // EXPECT_FALSE(row_sp->GetRegisterInfo(k_rbp, regloc));
}

TEST_F(Testx86AssemblyInspectionEngine, TestSimplei386ugmented) {
  UnwindPlan::Row::RegisterLocation regloc;
  UnwindPlan::RowSP row_sp;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  std::unique_ptr<x86AssemblyInspectionEngine> engine32 = Geti386Inspector();

  uint8_t data[] = {
      0x55,       // pushl %ebp
      0x89, 0xe5, // movl %esp, %ebp

      // x86AssemblyInspectionEngine::AugmentUnwindPlanFromCallSite
      // has a bug where it can't augment a function that is just
      // prologue+epilogue - it needs at least one other instruction
      // in between.
      0x90, // nop

      0x5d, // popl %ebp
      0xc3  // retl
  };

  sample_range = AddressRange(0x1000, sizeof(data));

  unwind_plan.SetSourceName("unit testing hand-created unwind plan");
  unwind_plan.SetPlanValidAddressRange(sample_range);
  unwind_plan.SetRegisterKind(eRegisterKindLLDB);

  row_sp.reset(new UnwindPlan::Row);

  // Describe offset 0
  row_sp->SetOffset(0);
  row_sp->GetCFAValue().SetIsRegisterPlusOffset(k_esp, 4);

  regloc.SetAtCFAPlusOffset(-4);
  row_sp->SetRegisterInfo(k_eip, regloc);

  unwind_plan.AppendRow(row_sp);

  // Allocate a new Row, populate it with the existing Row contents.
  UnwindPlan::Row *new_row = new UnwindPlan::Row;
  *new_row = *row_sp.get();
  row_sp.reset(new_row);

  // Describe offset 1
  row_sp->SetOffset(1);
  row_sp->GetCFAValue().SetIsRegisterPlusOffset(k_esp, 8);
  regloc.SetAtCFAPlusOffset(-8);
  row_sp->SetRegisterInfo(k_ebp, regloc);
  unwind_plan.AppendRow(row_sp);

  // Allocate a new Row, populate it with the existing Row contents.
  new_row = new UnwindPlan::Row;
  *new_row = *row_sp.get();
  row_sp.reset(new_row);

  // Describe offset 3
  row_sp->SetOffset(3);
  row_sp->GetCFAValue().SetIsRegisterPlusOffset(k_ebp, 8);
  unwind_plan.AppendRow(row_sp);

  RegisterContextSP reg_ctx_sp;
  EXPECT_TRUE(engine32->AugmentUnwindPlanFromCallSite(
      data, sizeof(data), sample_range, unwind_plan, reg_ctx_sp));

  row_sp = unwind_plan.GetRowForFunctionOffset(5);
  EXPECT_EQ(5ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_esp);
  EXPECT_EQ(4, row_sp->GetCFAValue().GetOffset());

  // x86AssemblyInspectionEngine::AugmentUnwindPlanFromCallSite
  // doesn't track register restores (pop'ing a reg value back from
  // the stack) - it was just written to make stepping work correctly.
  // Technically we should be able to do the following test, but it
  // won't work today - the unwind plan will still say that the caller's
  // ebp is on the stack.
  // EXPECT_FALSE(row_sp->GetRegisterInfo(k_ebp, regloc));
}

// Check that the i386 disassembler disassembles past an opcode that
// is only valid in 32-bit mode (non-long mode), and the x86_64 disassembler
// stops
// disassembling at that point (long-mode).
TEST_F(Testx86AssemblyInspectionEngine, Test32BitOnlyInstruction) {
  UnwindPlan::Row::RegisterLocation regloc;
  UnwindPlan::RowSP row_sp;
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  std::unique_ptr<x86AssemblyInspectionEngine> engine32 = Geti386Inspector();
  std::unique_ptr<x86AssemblyInspectionEngine> engine64 = Getx86_64Inspector();

  uint8_t data[] = {
      0x43, // incl $ebx --- an invalid opcode in 64-bit mode
      0x55, // pushl %ebp
      0x90  // nop
  };

  sample_range = AddressRange(0x1000, sizeof(data));

  EXPECT_TRUE(engine32->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(2);
  EXPECT_EQ(2ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_esp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(8, row_sp->GetCFAValue().GetOffset());

  EXPECT_TRUE(row_sp->GetRegisterInfo(k_ebp, regloc));
  EXPECT_TRUE(regloc.IsAtCFAPlusOffset());
  EXPECT_EQ(-8, regloc.GetOffset());

  unwind_plan.Clear();

  EXPECT_TRUE(engine64->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  row_sp = unwind_plan.GetRowForFunctionOffset(2);
  EXPECT_EQ(0ull, row_sp->GetOffset());
  EXPECT_TRUE(row_sp->GetCFAValue().GetRegisterNumber() == k_rsp);
  EXPECT_TRUE(row_sp->GetCFAValue().IsRegisterPlusOffset() == true);
  EXPECT_EQ(8, row_sp->GetCFAValue().GetOffset());

  EXPECT_FALSE(row_sp->GetRegisterInfo(k_rbp, regloc));
}

TEST_F(Testx86AssemblyInspectionEngine, TestStackRealign8BitDisp_i386) {
  std::unique_ptr<x86AssemblyInspectionEngine> engine = Geti386Inspector();

  uint8_t data[] = {
      0x55,             // pushl %ebp
      0x89, 0xe5,       // movl %esp, %ebp
      0x53,             // pushl %ebx
      0x83, 0xe4, 0xf0, // andl $-16, %esp
      0x83, 0xec, 0x10, // subl $16, %esp
      0x8d, 0x65, 0xfc, // leal -4(%ebp), %esp
      0x5b,             // popl %ebx
      0x5d,             // popl %ebp
      0xc3,             // retl
  };

  AddressRange sample_range(0x1000, sizeof(data));
  UnwindPlan plan(eRegisterKindLLDB);
  ASSERT_TRUE(engine->GetNonCallSiteUnwindPlanFromAssembly(data, sizeof(data),
                                                           sample_range, plan));

  UnwindPlan::Row::FAValue esp_plus_4, esp_plus_8, ebp_plus_8;
  esp_plus_4.SetIsRegisterPlusOffset(k_esp, 4);
  esp_plus_8.SetIsRegisterPlusOffset(k_esp, 8);
  ebp_plus_8.SetIsRegisterPlusOffset(k_ebp, 8);

  EXPECT_EQ(esp_plus_4, plan.GetRowForFunctionOffset(0)->GetCFAValue());
  EXPECT_EQ(esp_plus_8, plan.GetRowForFunctionOffset(1)->GetCFAValue());
  for (size_t i = 3; i < sizeof(data) - 2; ++i)
    EXPECT_EQ(ebp_plus_8, plan.GetRowForFunctionOffset(i)->GetCFAValue())
        << "i: " << i;
  EXPECT_EQ(esp_plus_4,
            plan.GetRowForFunctionOffset(sizeof(data) - 1)->GetCFAValue());
}

TEST_F(Testx86AssemblyInspectionEngine, TestStackRealign32BitDisp_x86_64) {
  std::unique_ptr<x86AssemblyInspectionEngine> engine = Getx86_64Inspector();

  uint8_t data[] = {
      0x55,                                     // pushq %rbp
      0x48, 0x89, 0xe5,                         // movq %rsp, %rbp
      0x53,                                     // pushl %rbx
      0x48, 0x83, 0xe4, 0xf0,                   // andq $-16, %rsp
      0x48, 0x81, 0xec, 0x00, 0x01, 0x00, 0x00, // subq $256, %rsp
      0x48, 0x8d, 0x65, 0xf8,                   // leaq -8(%rbp), %rsp
      0x5b,                                     // popq %rbx
      0x5d,                                     // popq %rbp
      0xc3,                                     // retq
  };

  AddressRange sample_range(0x1000, sizeof(data));
  UnwindPlan plan(eRegisterKindLLDB);
  ASSERT_TRUE(engine->GetNonCallSiteUnwindPlanFromAssembly(data, sizeof(data),
                                                           sample_range, plan));

  UnwindPlan::Row::FAValue rsp_plus_8, rsp_plus_16, rbp_plus_16;
  rsp_plus_8.SetIsRegisterPlusOffset(k_rsp, 8);
  rsp_plus_16.SetIsRegisterPlusOffset(k_rsp, 16);
  rbp_plus_16.SetIsRegisterPlusOffset(k_rbp, 16);

  EXPECT_EQ(rsp_plus_8, plan.GetRowForFunctionOffset(0)->GetCFAValue());
  EXPECT_EQ(rsp_plus_16, plan.GetRowForFunctionOffset(1)->GetCFAValue());
  for (size_t i = 4; i < sizeof(data) - 2; ++i)
    EXPECT_EQ(rbp_plus_16, plan.GetRowForFunctionOffset(i)->GetCFAValue())
        << "i: " << i;
  EXPECT_EQ(rsp_plus_8,
            plan.GetRowForFunctionOffset(sizeof(data) - 1)->GetCFAValue());
}

TEST_F(Testx86AssemblyInspectionEngine, TestStackRealignMSVC_i386) {
  std::unique_ptr<x86AssemblyInspectionEngine> engine = Geti386Inspector();

  uint8_t data[] = {
      0x53,                               // offset 00 -- pushl %ebx
      0x8b, 0xdc,                         // offset 01 -- movl %esp, %ebx
      0x83, 0xec, 0x08,                   // offset 03 -- subl $8, %esp
      0x81, 0xe4, 0x00, 0xff, 0xff, 0xff, // offset 06 -- andl $-256, %esp
      0x83, 0xc4, 0x04,                   // offset 12 -- addl $4, %esp
      0x55,                               // offset 15 -- pushl %ebp
      0x8b, 0xec,                         // offset 16 -- movl %esp, %ebp
      0x81, 0xec, 0x00, 0x02, 0x00, 0x00, // offset 18 -- subl $512, %esp
      0x89, 0x7d, 0xfc,                   // offset 24 -- movl %edi, -4(%ebp)
      0x8b, 0xe5,                         // offset 27 -- movl %ebp, %esp
      0x5d,                               // offset 29 -- popl %ebp
      0x8b, 0xe3,                         // offset 30 -- movl %ebx, %esp
      0x5b,                               // offset 32 -- popl %ebx
      0xc3                                // offset 33 -- retl
  };

  AddressRange sample_range(0x1000, sizeof(data));
  UnwindPlan plan(eRegisterKindLLDB);
  ASSERT_TRUE(engine->GetNonCallSiteUnwindPlanFromAssembly(data, sizeof(data),
                                                           sample_range, plan));

  UnwindPlan::Row::FAValue esp_minus_4, esp_plus_0, esp_plus_4, esp_plus_8,
      ebx_plus_8, ebp_plus_0;
  esp_minus_4.SetIsRegisterPlusOffset(k_esp, -4);
  esp_plus_0.SetIsRegisterPlusOffset(k_esp, 0);
  esp_plus_4.SetIsRegisterPlusOffset(k_esp, 4);
  esp_plus_8.SetIsRegisterPlusOffset(k_esp, 8);
  ebx_plus_8.SetIsRegisterPlusOffset(k_ebx, 8);
  ebp_plus_0.SetIsRegisterPlusOffset(k_ebp, 0);

  // Test CFA
  EXPECT_EQ(esp_plus_4, plan.GetRowForFunctionOffset(0)->GetCFAValue());
  EXPECT_EQ(esp_plus_8, plan.GetRowForFunctionOffset(1)->GetCFAValue());
  for (size_t i = 3; i < 33; ++i)
    EXPECT_EQ(ebx_plus_8, plan.GetRowForFunctionOffset(i)->GetCFAValue())
        << "i: " << i;
  EXPECT_EQ(esp_plus_4, plan.GetRowForFunctionOffset(33)->GetCFAValue());

  // Test AFA
  EXPECT_EQ(esp_plus_0, plan.GetRowForFunctionOffset(12)->GetAFAValue());
  EXPECT_EQ(esp_minus_4, plan.GetRowForFunctionOffset(15)->GetAFAValue());
  EXPECT_EQ(esp_plus_0, plan.GetRowForFunctionOffset(16)->GetAFAValue());
  for (size_t i = 18; i < 30; ++i)
    EXPECT_EQ(ebp_plus_0, plan.GetRowForFunctionOffset(i)->GetAFAValue())
        << "i: " << i;
  EXPECT_EQ(esp_minus_4, plan.GetRowForFunctionOffset(30)->GetAFAValue());

  // Test saved register
  UnwindPlan::Row::RegisterLocation reg_loc;
  EXPECT_TRUE(
      plan.GetRowForFunctionOffset(27)->GetRegisterInfo(k_edi, reg_loc));
  EXPECT_TRUE(reg_loc.IsAtAFAPlusOffset());
  EXPECT_EQ(-4, reg_loc.GetOffset());
}

// Give the disassembler random bytes to test that it doesn't exceed
// the bounds of the array when run under clang's address sanitizer.
TEST_F(Testx86AssemblyInspectionEngine, TestDisassemblyJunkBytes) {
  AddressRange sample_range;
  UnwindPlan unwind_plan(eRegisterKindLLDB);
  std::unique_ptr<x86AssemblyInspectionEngine> engine32 = Geti386Inspector();
  std::unique_ptr<x86AssemblyInspectionEngine> engine64 = Getx86_64Inspector();

  uint8_t data[] = {
      0x10, 0x10, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
      0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 };

  sample_range = AddressRange(0x1000, sizeof(data));

  EXPECT_TRUE(engine32->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

  unwind_plan.Clear();

  EXPECT_TRUE(engine64->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));

}

