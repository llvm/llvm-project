//===-- WebAssemblyCFGStackify.cpp - CFG Stackification -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements a CFG stacking pass.
///
/// This pass inserts BLOCK and LOOP markers to mark the start of scopes, since
/// scope boundaries serve as the labels for WebAssembly's control transfers.
///
/// This is sufficient to convert arbitrary CFGs into a form that works on
/// WebAssembly, provided that all loops are single-entry.
///
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssembly.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "WebAssemblySubtarget.h"
#include "WebAssemblyUtilities.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-cfg-stackify"

namespace {
class WebAssemblyCFGStackify final : public MachineFunctionPass {
  StringRef getPassName() const override { return "WebAssembly CFG Stackify"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<MachineDominatorTree>();
    AU.addPreserved<MachineDominatorTree>();
    AU.addRequired<MachineLoopInfo>();
    AU.addPreserved<MachineLoopInfo>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

public:
  static char ID; // Pass identification, replacement for typeid
  WebAssemblyCFGStackify() : MachineFunctionPass(ID) {}
};
} // end anonymous namespace

char WebAssemblyCFGStackify::ID = 0;
INITIALIZE_PASS(WebAssemblyCFGStackify, DEBUG_TYPE,
                "Insert BLOCK and LOOP markers for WebAssembly scopes",
                false, false)

FunctionPass *llvm::createWebAssemblyCFGStackify() {
  return new WebAssemblyCFGStackify();
}

/// Test whether Pred has any terminators explicitly branching to MBB, as
/// opposed to falling through. Note that it's possible (eg. in unoptimized
/// code) for a branch instruction to both branch to a block and fallthrough
/// to it, so we check the actual branch operands to see if there are any
/// explicit mentions.
static bool ExplicitlyBranchesTo(MachineBasicBlock *Pred,
                                 MachineBasicBlock *MBB) {
  for (MachineInstr &MI : Pred->terminators())
    for (MachineOperand &MO : MI.explicit_operands())
      if (MO.isMBB() && MO.getMBB() == MBB)
        return true;
  return false;
}

/// Insert a BLOCK marker for branches to MBB (if needed).
static void PlaceBlockMarker(
    MachineBasicBlock &MBB, MachineFunction &MF,
    SmallVectorImpl<MachineBasicBlock *> &ScopeTops,
    DenseMap<const MachineInstr *, MachineInstr *> &BlockTops,
    DenseMap<const MachineInstr *, MachineInstr *> &LoopTops,
    const WebAssemblyInstrInfo &TII,
    const MachineLoopInfo &MLI,
    MachineDominatorTree &MDT,
    WebAssemblyFunctionInfo &MFI) {
  // First compute the nearest common dominator of all forward non-fallthrough
  // predecessors so that we minimize the time that the BLOCK is on the stack,
  // which reduces overall stack height.
  MachineBasicBlock *Header = nullptr;
  bool IsBranchedTo = false;
  int MBBNumber = MBB.getNumber();
  for (MachineBasicBlock *Pred : MBB.predecessors())
    if (Pred->getNumber() < MBBNumber) {
      Header = Header ? MDT.findNearestCommonDominator(Header, Pred) : Pred;
      if (ExplicitlyBranchesTo(Pred, &MBB))
        IsBranchedTo = true;
    }
  if (!Header)
    return;
  if (!IsBranchedTo)
    return;

  assert(&MBB != &MF.front() && "Header blocks shouldn't have predecessors");
  MachineBasicBlock *LayoutPred = &*std::prev(MachineFunction::iterator(&MBB));

  // If the nearest common dominator is inside a more deeply nested context,
  // walk out to the nearest scope which isn't more deeply nested.
  for (MachineFunction::iterator I(LayoutPred), E(Header); I != E; --I) {
    if (MachineBasicBlock *ScopeTop = ScopeTops[I->getNumber()]) {
      if (ScopeTop->getNumber() > Header->getNumber()) {
        // Skip over an intervening scope.
        I = std::next(MachineFunction::iterator(ScopeTop));
      } else {
        // We found a scope level at an appropriate depth.
        Header = ScopeTop;
        break;
      }
    }
  }

  // Decide where in Header to put the BLOCK.
  MachineBasicBlock::iterator InsertPos;
  MachineLoop *HeaderLoop = MLI.getLoopFor(Header);
  if (HeaderLoop && MBB.getNumber() > LoopBottom(HeaderLoop)->getNumber()) {
    // Header is the header of a loop that does not lexically contain MBB, so
    // the BLOCK needs to be above the LOOP, after any END constructs.
    InsertPos = Header->begin();
    while (InsertPos->getOpcode() == WebAssembly::END_BLOCK ||
           InsertPos->getOpcode() == WebAssembly::END_LOOP)
      ++InsertPos;
  } else {
    // Otherwise, insert the BLOCK as late in Header as we can, but before the
    // beginning of the local expression tree and any nested BLOCKs.
    InsertPos = Header->getFirstTerminator();
    while (InsertPos != Header->begin() &&
           WebAssembly::isChild(*std::prev(InsertPos), MFI) &&
           std::prev(InsertPos)->getOpcode() != WebAssembly::LOOP &&
           std::prev(InsertPos)->getOpcode() != WebAssembly::END_BLOCK &&
           std::prev(InsertPos)->getOpcode() != WebAssembly::END_LOOP)
      --InsertPos;
  }

  // Add the BLOCK.
  MachineInstr *Begin = BuildMI(*Header, InsertPos, MBB.findDebugLoc(InsertPos),
                                TII.get(WebAssembly::BLOCK))
                            .addImm(int64_t(WebAssembly::ExprType::Void));

  // Mark the end of the block.
  InsertPos = MBB.begin();
  while (InsertPos != MBB.end() &&
         InsertPos->getOpcode() == WebAssembly::END_LOOP &&
         LoopTops[&*InsertPos]->getParent()->getNumber() >= Header->getNumber())
    ++InsertPos;
  MachineInstr *End = BuildMI(MBB, InsertPos, MBB.findPrevDebugLoc(InsertPos),
                              TII.get(WebAssembly::END_BLOCK));
  BlockTops[End] = Begin;

  // Track the farthest-spanning scope that ends at this point.
  int Number = MBB.getNumber();
  if (!ScopeTops[Number] ||
      ScopeTops[Number]->getNumber() > Header->getNumber())
    ScopeTops[Number] = Header;
}

/// Insert a LOOP marker for a loop starting at MBB (if it's a loop header).
static void PlaceLoopMarker(
    MachineBasicBlock &MBB, MachineFunction &MF,
    SmallVectorImpl<MachineBasicBlock *> &ScopeTops,
    DenseMap<const MachineInstr *, MachineInstr *> &LoopTops,
    const WebAssemblyInstrInfo &TII, const MachineLoopInfo &MLI) {
  MachineLoop *Loop = MLI.getLoopFor(&MBB);
  if (!Loop || Loop->getHeader() != &MBB)
    return;

  // The operand of a LOOP is the first block after the loop. If the loop is the
  // bottom of the function, insert a dummy block at the end.
  MachineBasicBlock *Bottom = LoopBottom(Loop);
  auto Iter = std::next(MachineFunction::iterator(Bottom));
  if (Iter == MF.end()) {
    MachineBasicBlock *Label = MF.CreateMachineBasicBlock();
    // Give it a fake predecessor so that AsmPrinter prints its label.
    Label->addSuccessor(Label);
    MF.push_back(Label);
    Iter = std::next(MachineFunction::iterator(Bottom));
  }
  MachineBasicBlock *AfterLoop = &*Iter;

  // Mark the beginning of the loop (after the end of any existing loop that
  // ends here).
  auto InsertPos = MBB.begin();
  while (InsertPos != MBB.end() &&
         InsertPos->getOpcode() == WebAssembly::END_LOOP)
    ++InsertPos;
  MachineInstr *Begin = BuildMI(MBB, InsertPos, MBB.findDebugLoc(InsertPos),
                                TII.get(WebAssembly::LOOP))
                            .addImm(int64_t(WebAssembly::ExprType::Void));

  // Mark the end of the loop (using arbitrary debug location that branched
  // to the loop end as its location).
  DebugLoc EndDL = (*AfterLoop->pred_rbegin())->findBranchDebugLoc();
  MachineInstr *End = BuildMI(*AfterLoop, AfterLoop->begin(), EndDL,
                              TII.get(WebAssembly::END_LOOP));
  LoopTops[End] = Begin;

  assert((!ScopeTops[AfterLoop->getNumber()] ||
          ScopeTops[AfterLoop->getNumber()]->getNumber() < MBB.getNumber()) &&
         "With block sorting the outermost loop for a block should be first.");
  if (!ScopeTops[AfterLoop->getNumber()])
    ScopeTops[AfterLoop->getNumber()] = &MBB;
}

static unsigned
GetDepth(const SmallVectorImpl<const MachineBasicBlock *> &Stack,
         const MachineBasicBlock *MBB) {
  unsigned Depth = 0;
  for (auto X : reverse(Stack)) {
    if (X == MBB)
      break;
    ++Depth;
  }
  assert(Depth < Stack.size() && "Branch destination should be in scope");
  return Depth;
}

/// In normal assembly languages, when the end of a function is unreachable,
/// because the function ends in an infinite loop or a noreturn call or similar,
/// it isn't necessary to worry about the function return type at the end of
/// the function, because it's never reached. However, in WebAssembly, blocks
/// that end at the function end need to have a return type signature that
/// matches the function signature, even though it's unreachable. This function
/// checks for such cases and fixes up the signatures.
static void FixEndsAtEndOfFunction(
    MachineFunction &MF,
    const WebAssemblyFunctionInfo &MFI,
    DenseMap<const MachineInstr *, MachineInstr *> &BlockTops,
    DenseMap<const MachineInstr *, MachineInstr *> &LoopTops) {
  assert(MFI.getResults().size() <= 1);

  if (MFI.getResults().empty())
    return;

  WebAssembly::ExprType retType;
  switch (MFI.getResults().front().SimpleTy) {
  case MVT::i32: retType = WebAssembly::ExprType::I32; break;
  case MVT::i64: retType = WebAssembly::ExprType::I64; break;
  case MVT::f32: retType = WebAssembly::ExprType::F32; break;
  case MVT::f64: retType = WebAssembly::ExprType::F64; break;
  case MVT::v16i8: retType = WebAssembly::ExprType::I8x16; break;
  case MVT::v8i16: retType = WebAssembly::ExprType::I16x8; break;
  case MVT::v4i32: retType = WebAssembly::ExprType::I32x4; break;
  case MVT::v4f32: retType = WebAssembly::ExprType::F32x4; break;
  case MVT::ExceptRef: retType = WebAssembly::ExprType::ExceptRef; break;
  default: llvm_unreachable("unexpected return type");
  }

  for (MachineBasicBlock &MBB : reverse(MF)) {
    for (MachineInstr &MI : reverse(MBB)) {
      if (MI.isPosition() || MI.isDebugValue())
        continue;
      if (MI.getOpcode() == WebAssembly::END_BLOCK) {
        BlockTops[&MI]->getOperand(0).setImm(int32_t(retType));
        continue;
      }
      if (MI.getOpcode() == WebAssembly::END_LOOP) {
        LoopTops[&MI]->getOperand(0).setImm(int32_t(retType));
        continue;
      }
      // Something other than an `end`. We're done.
      return;
    }
  }
}

// WebAssembly functions end with an end instruction, as if the function body
// were a block.
static void AppendEndToFunction(
    MachineFunction &MF,
    const WebAssemblyInstrInfo &TII) {
  BuildMI(MF.back(), MF.back().end(),
          MF.back().findPrevDebugLoc(MF.back().end()),
          TII.get(WebAssembly::END_FUNCTION));
}

/// Insert LOOP and BLOCK markers at appropriate places.
static void PlaceMarkers(MachineFunction &MF, const MachineLoopInfo &MLI,
                         const WebAssemblyInstrInfo &TII,
                         MachineDominatorTree &MDT,
                         WebAssemblyFunctionInfo &MFI) {
  // For each block whose label represents the end of a scope, record the block
  // which holds the beginning of the scope. This will allow us to quickly skip
  // over scoped regions when walking blocks. We allocate one more than the
  // number of blocks in the function to accommodate for the possible fake block
  // we may insert at the end.
  SmallVector<MachineBasicBlock *, 8> ScopeTops(MF.getNumBlockIDs() + 1);

  // For each LOOP_END, the corresponding LOOP.
  DenseMap<const MachineInstr *, MachineInstr *> LoopTops;

  // For each END_BLOCK, the corresponding BLOCK.
  DenseMap<const MachineInstr *, MachineInstr *> BlockTops;

  for (auto &MBB : MF) {
    // Place the LOOP for MBB if MBB is the header of a loop.
    PlaceLoopMarker(MBB, MF, ScopeTops, LoopTops, TII, MLI);

    // Place the BLOCK for MBB if MBB is branched to from above.
    PlaceBlockMarker(MBB, MF, ScopeTops, BlockTops, LoopTops, TII, MLI, MDT, MFI);
  }

  // Now rewrite references to basic blocks to be depth immediates.
  SmallVector<const MachineBasicBlock *, 8> Stack;
  for (auto &MBB : reverse(MF)) {
    for (auto &MI : reverse(MBB)) {
      switch (MI.getOpcode()) {
      case WebAssembly::BLOCK:
        assert(ScopeTops[Stack.back()->getNumber()]->getNumber() <= MBB.getNumber() &&
               "Block should be balanced");
        Stack.pop_back();
        break;
      case WebAssembly::LOOP:
        assert(Stack.back() == &MBB && "Loop top should be balanced");
        Stack.pop_back();
        break;
      case WebAssembly::END_BLOCK:
        Stack.push_back(&MBB);
        break;
      case WebAssembly::END_LOOP:
        Stack.push_back(LoopTops[&MI]->getParent());
        break;
      default:
        if (MI.isTerminator()) {
          // Rewrite MBB operands to be depth immediates.
          SmallVector<MachineOperand, 4> Ops(MI.operands());
          while (MI.getNumOperands() > 0)
            MI.RemoveOperand(MI.getNumOperands() - 1);
          for (auto MO : Ops) {
            if (MO.isMBB())
              MO = MachineOperand::CreateImm(GetDepth(Stack, MO.getMBB()));
            MI.addOperand(MF, MO);
          }
        }
        break;
      }
    }
  }
  assert(Stack.empty() && "Control flow should be balanced");

  // Fix up block/loop signatures at the end of the function to conform to
  // WebAssembly's rules.
  FixEndsAtEndOfFunction(MF, MFI, BlockTops, LoopTops);

  // Add an end instruction at the end of the function body.
  if (!MF.getSubtarget<WebAssemblySubtarget>()
        .getTargetTriple().isOSBinFormatELF())
    AppendEndToFunction(MF, TII);
}

bool WebAssemblyCFGStackify::runOnMachineFunction(MachineFunction &MF) {
  DEBUG(dbgs() << "********** CFG Stackifying **********\n"
                  "********** Function: "
               << MF.getName() << '\n');

  const auto &MLI = getAnalysis<MachineLoopInfo>();
  auto &MDT = getAnalysis<MachineDominatorTree>();
  // Liveness is not tracked for VALUE_STACK physreg.
  const auto &TII = *MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();
  WebAssemblyFunctionInfo &MFI = *MF.getInfo<WebAssemblyFunctionInfo>();
  MF.getRegInfo().invalidateLiveness();

  // Place the BLOCK and LOOP markers to indicate the beginnings of scopes.
  PlaceMarkers(MF, MLI, TII, MDT, MFI);

  return true;
}
