//===---- llvm/unittest/CodeGen/DAGCombinerTest.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "gtest/gtest.h"

using namespace llvm;

// computeDemandedBitsFromUses walks the use chain of an SRL upward and
// returns the union of bits demanded by all downstream users.  These tests
// verify the demand computation directly against constructed DAG nodes,
// without running the full DAGCombiner pass (which requires
// FunctionLoweringInfo and other infrastructure not available in this
// lightweight fixture).
//
// The fixture uses x86-64 rather than AArch64 because this test suite is
// compiled with X86 always enabled; SelectionDAGTestBase uses AArch64 and
// would skip on builds that omit it.
class DAGCombinerTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    InitializeAllTargets();
    InitializeAllTargetMCs();
  }

  void SetUp() override {
    StringRef Assembly = "define i64 @f(i64 %x) { ret i64 %x }";

    Triple TargetTriple("x86_64-unknown-linux-gnu");
    std::string Error;
    const Target *T = TargetRegistry::lookupTarget("", TargetTriple, Error);
    if (!T)
      GTEST_SKIP();

    TargetOptions Options;
    TM = std::unique_ptr<TargetMachine>(T->createTargetMachine(
        TargetTriple, "x86-64", "", Options, std::nullopt, std::nullopt,
        CodeGenOptLevel::Default));
    if (!TM)
      GTEST_SKIP();

    SMDiagnostic SMError;
    M = parseAssemblyString(Assembly, SMError, Context);
    ASSERT_TRUE(M && "Could not parse module");
    M->setDataLayout(TM->createDataLayout());

    F = M->getFunction("f");
    ASSERT_TRUE(F && "Could not find function f");

    // MMI and ORE are stored as members: SelectionDAG stores pointers to
    // them and they must remain alive for the duration of the test.
    MMI = std::make_unique<MachineModuleInfo>(TM.get());
    MF = std::make_unique<MachineFunction>(*F, *TM, *TM->getSubtargetImpl(*F),
                                           MMI->getContext(), 0);
    DAG = std::make_unique<SelectionDAG>(*TM, CodeGenOptLevel::Default);
    ASSERT_TRUE(DAG);
    ORE = std::make_unique<OptimizationRemarkEmitter>(F);
    DAG->init(*MF, *ORE, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
              *MMI, nullptr);
  }

  LLVMContext Context;
  std::unique_ptr<TargetMachine> TM;
  std::unique_ptr<Module> M;
  Function *F = nullptr;
  std::unique_ptr<MachineModuleInfo> MMI;
  std::unique_ptr<OptimizationRemarkEmitter> ORE;
  std::unique_ptr<MachineFunction> MF;
  std::unique_ptr<SelectionDAG> DAG;
};

// Simple case: downstream AND with 255 anchors demand to bits[0:7].
// computeDemandedBitsFromUses for the SRL should return exactly bits[0:7].
//
// DAG:  and(srl(x, 8), 255)
//                ^
//         we query here
TEST_F(DAGCombinerTest, SimpleDownstreamAndAnchorsDemand) {
  SDLoc DL;
  MVT VT = MVT::i64;

  SDValue X = DAG->getCopyFromReg(DAG->getEntryNode(), DL,
                                  Register::index2VirtReg(1), VT);
  SDValue Srl = DAG->getNode(ISD::SRL, DL, VT, X, DAG->getConstant(8, DL, VT));
  // Downstream consumer: only bits[0:7] demanded.
  DAG->getNode(ISD::AND, DL, VT, Srl, DAG->getConstant(255, DL, VT));

  APInt Demand = computeDemandedBitsFromUses(Srl);
  EXPECT_EQ(Demand, APInt::getLowBitsSet(64, 8));
}

// When no downstream AND limits the demand, the function should return all-ones
// (conservative — every bit of the SRL might be used).
//
// DAG:  xor(srl(x, 8), y)   — XOR passes demand through unchanged
TEST_F(DAGCombinerTest, XorUserWithNoAnchorReturnsAllOnes) {
  SDLoc DL;
  MVT VT = MVT::i64;

  SDValue X = DAG->getCopyFromReg(DAG->getEntryNode(), DL,
                                  Register::index2VirtReg(1), VT);
  SDValue Y = DAG->getCopyFromReg(DAG->getEntryNode(), DL,
                                  Register::index2VirtReg(2), VT);
  SDValue Srl = DAG->getNode(ISD::SRL, DL, VT, X, DAG->getConstant(8, DL, VT));
  // XOR with no further narrowing consumer: all bits demanded.
  DAG->getNode(ISD::XOR, DL, VT, Srl, Y);

  APInt Demand = computeDemandedBitsFromUses(Srl);
  EXPECT_TRUE(Demand.isAllOnes());
}

// Multi-hop demand chain through two XOR / SRL hops:
//
// DAG:  and(xor(srl(xor(srl(x, 8), x), 4), xor(...)), 15)
//                    ^
//              we query here (the inner SRL)
//
// The final and(..., 15) anchors demand to bits[0:3].  Working backward:
//   xor passes demand through         →  bits[0:3] demanded from srl(., 4)
//   srl(., 4) back-propagates         →  bits[4:7] demanded from xor(srl,x)
//   xor(srl,x) also used directly     →  bits[0:3] | bits[4:7] = bits[0:7]
//   inner srl (shift 8) receives      →  bits[0:7] demanded from its result
//
// So computeDemandedBitsFromUses for the inner SRL should return bits[0:7].
TEST_F(DAGCombinerTest, MultiHopXorSrlChainPropagatesDemand) {
  SDLoc DL;
  MVT VT = MVT::i64;

  SDValue X = DAG->getCopyFromReg(DAG->getEntryNode(), DL,
                                  Register::index2VirtReg(1), VT);

  // Inner SRL — the node under test.
  SDValue InnerSrl =
      DAG->getNode(ISD::SRL, DL, VT, X, DAG->getConstant(8, DL, VT));
  SDValue Xor1 = DAG->getNode(ISD::XOR, DL, VT, InnerSrl, X);
  SDValue OuterSrl =
      DAG->getNode(ISD::SRL, DL, VT, Xor1, DAG->getConstant(4, DL, VT));
  SDValue Xor2 = DAG->getNode(ISD::XOR, DL, VT, OuterSrl, Xor1);
  // Anchor: only bits[0:3] demanded from the chain.
  DAG->getNode(ISD::AND, DL, VT, Xor2, DAG->getConstant(15, DL, VT));

  APInt Demand = computeDemandedBitsFromUses(InnerSrl);
  EXPECT_EQ(Demand, APInt::getLowBitsSet(64, 8));
}
