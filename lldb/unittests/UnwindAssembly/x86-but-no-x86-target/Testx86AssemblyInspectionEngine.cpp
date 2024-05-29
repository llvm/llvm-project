//===-- Testx86AssemblyInspectionEngine.cpp -------------------------------===//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "Plugins/UnwindAssembly/x86/x86AssemblyInspectionEngine.h"
#include "lldb/Core/AddressRange.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Utility/ArchSpec.h"

#include "llvm/Support/TargetSelect.h"

#include <memory>
#include <vector>

using namespace lldb;
using namespace lldb_private;

class Testx86AssemblyInspectionEngine : public testing::Test {
public:
  static void SetUpTestCase();
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
  EXPECT_FALSE(engine->GetNonCallSiteUnwindPlanFromAssembly(
      data, sizeof(data), sample_range, unwind_plan));
}
