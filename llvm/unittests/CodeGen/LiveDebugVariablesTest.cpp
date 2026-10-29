//===- LiveDebugVariablesTest.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/LiveDebugVariables.h"
#include "CodeGenTestBase.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/Config/Targets.h"
#include "llvm/Support/TargetSelect.h"

using namespace llvm;

namespace {

class LiveDebugVariablesTest : public CodeGenTestBase {
public:
  static void SetUpTestCase() {
#if LLVM_HAS_X86_TARGET
    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86Target();
    LLVMInitializeX86TargetMC();
#endif
  }

  void SetUp() override { setUpImpl("x86_64--", "", ""); }
};

TEST_F(LiveDebugVariablesTest, PHIsAcrossRepeatedSplitsAndShrink) {
  ASSERT_TRUE(parseMIR(R"MIR(
--- |
  define void @test() !dbg !4 { ret void }
  !llvm.dbg.cu = !{!0}
  !llvm.module.flags = !{!5}
  !0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "llvm", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
  !1 = !DIFile(filename: "test.c", directory: "/")
  !2 = !DISubroutineType(types: !3)
  !3 = !{}
  !4 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 1, type: !2, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
  !5 = !{i32 2, !"Debug Info Version", i32 3}
...
---
name: test
tracksRegLiveness: true
body: |
  bb.0:
    liveins: $eax, $ebx, $ecx
    %0:gr32 = COPY $eax
    %1:gr32 = COPY $ebx
    %2:gr32 = COPY $ecx
    RET64
  bb.1:
    RET64
  bb.2:
    RET64
  bb.3:
    RET64
  bb.4:
    RET64
...
)MIR"));
  MachineFunction &MF = getMF("test");
  LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);
  VirtRegMap &VRM = MFAM.getResult<VirtRegMapAnalysis>(MF);
  Register Old = Register::index2VirtReg(0);
  Register First = Register::index2VirtReg(1);
  Register Second = Register::index2VirtReg(2);
  SmallVector<MCRegister, 3> PhysRegs;
  for (const MachineInstr &MI : MF.front())
    if (MI.isCopy())
      PhysRegs.push_back(MI.getOperand(1).getReg().asMCReg());
  ASSERT_EQ(PhysRegs.size(), 3u);
  VRM.assignVirt2Phys(Old, PhysRegs[0]);
  VRM.assignVirt2Phys(First, PhysRegs[1]);
  VRM.assignVirt2Phys(Second, PhysRegs[2]);

  // Model five PHIs coalesced into Old, then separated during allocation.
  LiveInterval &OldLI = LIS.getInterval(Old);
  OldLI.clear();
  for (MachineBasicBlock &MBB : MF) {
    SlotIndex Start = LIS.getMBBStartIdx(&MBB);
    VNInfo *VNI = OldLI.getNextValue(Start, LIS.getVNInfoAllocator());
    OldLI.addSegment({Start, LIS.getMBBEndIdx(&MBB), VNI});
    MF.DebugPHIPositions.try_emplace(MBB.getNumber() + 1, &MBB, Old, 0);
  }
  LiveDebugVariables LDV;
  LDV.analyze(MF, &LIS);

  for (Register Reg : {Old, First, Second})
    LIS.getInterval(Reg).clear();
  for (MachineBasicBlock &MBB : MF) {
    unsigned Block = MBB.getNumber();
    if (Block == 3)
      continue;
    Register Reg = Block == 0 ? First : Block == 1 ? Second : Old;
    LiveInterval &LI = LIS.getInterval(Reg);
    SlotIndex Start = LIS.getMBBStartIdx(&MBB);
    VNInfo *VNI = LI.getNextValue(Start, LIS.getVNInfoAllocator());
    LI.addSegment({Start, LIS.getMBBEndIdx(&MBB), VNI});
  }

  LDV.splitRegister(Old, {First}, LIS);
  LDV.splitRegister(Old, {Second}, LIS);
  LDV.shrinkRegister(Old);
  LDV.emitDebugValues(&VRM);

  SmallVector<std::pair<unsigned, Register>, 4> PHIs;
  for (const MachineBasicBlock &MBB : MF)
    for (const MachineInstr &MI : MBB)
      if (MI.isDebugPHI()) {
        EXPECT_EQ(MI.getOperand(1).getImm(), MBB.getNumber() + 1);
        PHIs.emplace_back(MI.getOperand(1).getImm(), MI.getOperand(0).getReg());
      }
  const SmallVector<std::pair<unsigned, Register>, 4> Expected = {
      {1, PhysRegs[1]}, {2, PhysRegs[2]}, {3, PhysRegs[0]}, {5, PhysRegs[0]}};
  EXPECT_EQ(PHIs, Expected);
}

} // namespace
