//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A node is found in CSEMap only if AddNodeIDCustom reproduces the key its
// creator looked it up by. Each test builds a node twice and expects one node.
//
//===----------------------------------------------------------------------===//

#include "SelectionDAGTestBase.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/MC/MCContext.h"

namespace llvm {

class SelectionDAGCSETest : public SelectionDAGTestBase {
protected:
  SDLoc Loc;

  SDValue getChain() { return DAG->getEntryNode(); }

  MachineMemOperand *getMMO(uint64_t Size = 4,
                            AtomicOrdering Order = AtomicOrdering::NotAtomic) {
    return MF->getMachineMemOperand(
        MachinePointerInfo(),
        MachineMemOperand::MOLoad | MachineMemOperand::MOStore, Size, Align(4),
        MMOMetadata(), SyncScope::System, Order);
  }
};

TEST_F(SelectionDAGCSETest, AtomicRMW) {
  SDValue Ptr = DAG->CreateStackTemporary(MVT::i32);
  SDValue Val = DAG->getConstant(1, Loc, MVT::i32);
  MachineMemOperand *MMO = getMMO(4, AtomicOrdering::SequentiallyConsistent);
  auto Atomic = [&] {
    return DAG
        ->getAtomic(ISD::ATOMIC_LOAD_UINC_WRAP, Loc, MVT::i32, getChain(), Ptr,
                    Val, MMO)
        .getNode();
  };
  // FIXME: AddNodeIDCustom's atomic opcode list is a second copy of
  // AtomicSDNode::classof's, stale since ATOMIC_LOAD_FADD.
  EXPECT_NE(Atomic(), Atomic());
}

TEST_F(SelectionDAGCSETest, DeactivationSymbol) {
  // FIXME: AddNodeIDCustom has no DEACTIVATION_SYMBOL case.
  EXPECT_NE(DAG->getDeactivationSymbol(G).getNode(),
            DAG->getDeactivationSymbol(G).getNode());
  EXPECT_NE(DAG->getDeactivationSymbol(G).getNode(),
            DAG->getDeactivationSymbol(AliasedG).getNode());
}

TEST_F(SelectionDAGCSETest, EHLabel) {
  MCSymbol *Sym = MF->getContext().createTempSymbol();
  MCSymbol *Other = MF->getContext().createTempSymbol();
  // FIXME: AddNodeIDCustom has no EH_LABEL case.
  EXPECT_NE(DAG->getEHLabel(Loc, getChain(), Sym).getNode(),
            DAG->getEHLabel(Loc, getChain(), Sym).getNode());
  EXPECT_NE(DAG->getEHLabel(Loc, getChain(), Sym).getNode(),
            DAG->getEHLabel(Loc, getChain(), Other).getNode());
}

TEST_F(SelectionDAGCSETest, ExtStridedLoadVP) {
  SDValue Ptr = DAG->CreateStackTemporary(MVT::v2i64);
  SDValue Stride = DAG->getConstant(8, Loc, MVT::i64);
  SDValue Mask = DAG->getConstant(1, Loc, MVT::v2i1);
  SDValue EVL = DAG->getConstant(2, Loc, MVT::i32);
  MachineMemOperand *MMO = getMMO(8);
  auto Load = [&] {
    return DAG
        ->getExtStridedLoadVP(ISD::EXTLOAD, Loc, MVT::v2i64, getChain(), Ptr,
                              Stride, Mask, EVL, MVT::v2i32, MMO)
        .getNode();
  };
  // FIXME: getStridedLoadVP keys on the result type where the case uses the
  // memory type, so only an extending load misses.
  EXPECT_NE(Load(), Load());
}

TEST_F(SelectionDAGCSETest, FPEnvMem) {
  SDValue Ptr = DAG->CreateStackTemporary(MVT::i32);
  MachineMemOperand *MMO = getMMO();
  // FIXME: AddNodeIDCustom has no GET_FPENV_MEM or SET_FPENV_MEM case.
  EXPECT_NE(DAG->getGetFPEnv(getChain(), Loc, Ptr, MVT::i32, MMO).getNode(),
            DAG->getGetFPEnv(getChain(), Loc, Ptr, MVT::i32, MMO).getNode());
  EXPECT_NE(DAG->getSetFPEnv(getChain(), Loc, Ptr, MVT::i32, MMO).getNode(),
            DAG->getSetFPEnv(getChain(), Loc, Ptr, MVT::i32, MMO).getNode());
}

TEST_F(SelectionDAGCSETest, LifetimeNode) {
  SDValue FIPtr = DAG->CreateStackTemporary(MVT::i32);
  int FI = cast<FrameIndexSDNode>(FIPtr.getNode())->getIndex();
  // FIXME: getLifetimeNode keys on a frame index operand 1 already carries.
  EXPECT_NE(DAG->getLifetimeNode(true, Loc, getChain(), FI).getNode(),
            DAG->getLifetimeNode(true, Loc, getChain(), FI).getNode());
}

TEST_F(SelectionDAGCSETest, MorphMemIntrinsicToMachineNode) {
  SDValue Ptr = DAG->CreateStackTemporary(MVT::i32);
  SDVTList VTs = DAG->getVTList(MVT::i32, MVT::Other);
  SDValue Ops[] = {getChain(), Ptr};
  SDNode *N = DAG->getMemIntrinsicNode(ISD::INTRINSIC_W_CHAIN, Loc, VTs, Ops,
                                       MVT::i32, getMMO())
                  .getNode();
  unsigned Opc = TargetOpcode::COPY;
  ASSERT_EQ(DAG->MorphNodeTo(N, ~Opc, VTs, Ops), N);
  // FIXME: MorphNodeTo's lookup omits AddNodeIDCustom, so the morphed node is
  // stored under a key getMachineNode does not build.
  EXPECT_NE(DAG->getMachineNode(Opc, Loc, VTs, Ops), N);
}

TEST_F(SelectionDAGCSETest, PseudoProbeNode) {
  // FIXME: getPseudoProbeNode drops the attributes its case profiles.
  EXPECT_NE(DAG->getPseudoProbeNode(Loc, getChain(), 1234, 5, 7).getNode(),
            DAG->getPseudoProbeNode(Loc, getChain(), 1234, 5, 7).getNode());
  EXPECT_NE(DAG->getPseudoProbeNode(Loc, getChain(), 1234, 5, 7).getNode(),
            DAG->getPseudoProbeNode(Loc, getChain(), 1234, 5, 8).getNode());
}

} // namespace llvm
