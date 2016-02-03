//===--- BlockGenerators.cpp - Generate code for statements -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the BlockGenerator and VectorBlockGenerator classes,
// which generate sequential code and vectorized code for a polyhedral
// statement, respectively.
//
//===----------------------------------------------------------------------===//

#include "polly/CodeGen/BlockGenerators.h"
#include "polly/CodeGen/CodeGeneration.h"
#include "polly/CodeGen/IslExprBuilder.h"
#include "polly/CodeGen/RuntimeDebugBuilder.h"
#include "polly/Options.h"
#include "polly/ScopInfo.h"
#include "polly/Support/GICHelper.h"
#include "polly/Support/SCEVValidator.h"
#include "polly/Support/ScopHelper.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "isl/aff.h"
#include "isl/ast.h"
#include "isl/ast_build.h"
#include "isl/set.h"
#include <deque>

using namespace llvm;
using namespace polly;

static cl::opt<bool> Aligned("enable-polly-aligned",
                             cl::desc("Assumed aligned memory accesses."),
                             cl::Hidden, cl::init(false), cl::ZeroOrMore,
                             cl::cat(PollyCategory));

static cl::opt<bool> DebugPrinting(
    "polly-codegen-add-debug-printing",
    cl::desc("Add printf calls that show the values loaded/stored."),
    cl::Hidden, cl::init(false), cl::ZeroOrMore, cl::cat(PollyCategory));

BlockGenerator::BlockGenerator(PollyIRBuilder &B, LoopInfo &LI,
                               ScalarEvolution &SE, DominatorTree &DT,
                               ScalarAllocaMapTy &ScalarMap,
                               ScalarAllocaMapTy &PHIOpMap,
                               EscapeUsersAllocaMapTy &EscapeMap,
                               ValueMapT &GlobalMap,
                               IslExprBuilder *ExprBuilder)
    : Builder(B), LI(LI), SE(SE), ExprBuilder(ExprBuilder), DT(DT),
      EntryBB(nullptr), PHIOpMap(PHIOpMap), ScalarMap(ScalarMap),
      EscapeMap(EscapeMap), GlobalMap(GlobalMap) {}

Value *BlockGenerator::trySynthesizeNewValue(ScopStmt &Stmt, Value *Old,
                                             ValueMapT &BBMap,
                                             LoopToScevMapT &LTS,
                                             Loop *L) const {
  if (!SE.isSCEVable(Old->getType()))
    return nullptr;

  const SCEV *Scev = SE.getSCEVAtScope(Old, L);
  if (!Scev)
    return nullptr;

  if (isa<SCEVCouldNotCompute>(Scev))
    return nullptr;

  const SCEV *NewScev = apply(Scev, LTS, SE);
  ValueMapT VTV;
  VTV.insert(BBMap.begin(), BBMap.end());
  VTV.insert(GlobalMap.begin(), GlobalMap.end());

  Scop &S = *Stmt.getParent();
  const DataLayout &DL =
      S.getRegion().getEntry()->getParent()->getParent()->getDataLayout();
  auto IP = Builder.GetInsertPoint();

  assert(IP != Builder.GetInsertBlock()->end() &&
         "Only instructions can be insert points for SCEVExpander");
  Value *Expanded =
      expandCodeFor(S, SE, DL, "polly", NewScev, Old->getType(), &*IP, &VTV);

  BBMap[Old] = Expanded;
  return Expanded;
}

Value *BlockGenerator::getNewValue(ScopStmt &Stmt, Value *Old, ValueMapT &BBMap,
                                   LoopToScevMapT &LTS, Loop *L) const {
  // Constants that do not reference any named value can always remain
  // unchanged. Handle them early to avoid expensive map lookups. We do not take
  // the fast-path for external constants which are referenced through globals
  // as these may need to be rewritten when distributing code accross different
  // LLVM modules.
  if (isa<Constant>(Old) && !isa<GlobalValue>(Old))
    return Old;

  // Inline asm is like a constant to us.
  if (isa<InlineAsm>(Old))
    return Old;

  if (Value *New = GlobalMap.lookup(Old)) {
    if (Value *NewRemapped = GlobalMap.lookup(New))
      New = NewRemapped;
    if (Old->getType()->getScalarSizeInBits() <
        New->getType()->getScalarSizeInBits())
      New = Builder.CreateTruncOrBitCast(New, Old->getType());

    return New;
  }

  if (Value *New = BBMap.lookup(Old))
    return New;

  if (Value *New = trySynthesizeNewValue(Stmt, Old, BBMap, LTS, L))
    return New;

  // A scop-constant value defined by a global or a function parameter.
  if (isa<GlobalValue>(Old) || isa<Argument>(Old))
    return Old;

  // A scop-constant value defined by an instruction executed outside the scop.
  if (const Instruction *Inst = dyn_cast<Instruction>(Old))
    if (!Stmt.getParent()->getRegion().contains(Inst->getParent()))
      return Old;

  // The scalar dependence is neither available nor SCEVCodegenable.
  llvm_unreachable("Unexpected scalar dependence in region!");
  return nullptr;
}

void BlockGenerator::copyInstScalar(ScopStmt &Stmt, Instruction *Inst,
                                    ValueMapT &BBMap, LoopToScevMapT &LTS) {
  // We do not generate debug intrinsics as we did not investigate how to
  // copy them correctly. At the current state, they just crash the code
  // generation as the meta-data operands are not correctly copied.
  if (isa<DbgInfoIntrinsic>(Inst))
    return;

  Instruction *NewInst = Inst->clone();

  // Replace old operands with the new ones.
  for (Value *OldOperand : Inst->operands()) {
    Value *NewOperand =
        getNewValue(Stmt, OldOperand, BBMap, LTS, getLoopForInst(Inst));

    if (!NewOperand) {
      assert(!isa<StoreInst>(NewInst) &&
             "Store instructions are always needed!");
      delete NewInst;
      return;
    }

    NewInst->replaceUsesOfWith(OldOperand, NewOperand);
  }

  Builder.Insert(NewInst);
  BBMap[Inst] = NewInst;

  if (!NewInst->getType()->isVoidTy())
    NewInst->setName("p_" + Inst->getName());
}

Value *
BlockGenerator::generateLocationAccessed(ScopStmt &Stmt, MemAccInst Inst,
                                         ValueMapT &BBMap, LoopToScevMapT &LTS,
                                         isl_id_to_ast_expr *NewAccesses) {
  const MemoryAccess &MA = Stmt.getArrayAccessFor(Inst);

  isl_ast_expr *AccessExpr = isl_id_to_ast_expr_get(NewAccesses, MA.getId());

  if (AccessExpr) {
    AccessExpr = isl_ast_expr_address_of(AccessExpr);
    auto Address = ExprBuilder->create(AccessExpr);

    // Cast the address of this memory access to a pointer type that has the
    // same element type as the original access, but uses the address space of
    // the newly generated pointer.
    auto OldPtrTy = MA.getAccessValue()->getType()->getPointerTo();
    auto NewPtrTy = Address->getType();
    OldPtrTy = PointerType::get(OldPtrTy->getElementType(),
                                NewPtrTy->getPointerAddressSpace());

    if (OldPtrTy != NewPtrTy) {
      assert(OldPtrTy->getPointerElementType()->getPrimitiveSizeInBits() ==
                 NewPtrTy->getPointerElementType()->getPrimitiveSizeInBits() &&
             "Pointer types to elements with different size found");
      Address = Builder.CreateBitOrPointerCast(Address, OldPtrTy);
    }
    return Address;
  }

  return getNewValue(Stmt, Inst.getPointerOperand(), BBMap, LTS,
                     getLoopForInst(Inst));
}

Loop *BlockGenerator::getLoopForInst(const llvm::Instruction *Inst) {
  return LI.getLoopFor(Inst->getParent());
}

Value *BlockGenerator::generateScalarLoad(ScopStmt &Stmt, LoadInst *Load,
                                          ValueMapT &BBMap, LoopToScevMapT &LTS,
                                          isl_id_to_ast_expr *NewAccesses) {
  if (Value *PreloadLoad = GlobalMap.lookup(Load))
    return PreloadLoad;

  Value *NewPointer =
      generateLocationAccessed(Stmt, Load, BBMap, LTS, NewAccesses);
  Value *ScalarLoad = Builder.CreateAlignedLoad(
      NewPointer, Load->getAlignment(), Load->getName() + "_p_scalar_");

  if (DebugPrinting)
    RuntimeDebugBuilder::createCPUPrinter(Builder, "Load from ", NewPointer,
                                          ": ", ScalarLoad, "\n");

  return ScalarLoad;
}

void BlockGenerator::generateScalarStore(ScopStmt &Stmt, StoreInst *Store,
                                         ValueMapT &BBMap, LoopToScevMapT &LTS,
                                         isl_id_to_ast_expr *NewAccesses) {
  Value *NewPointer =
      generateLocationAccessed(Stmt, Store, BBMap, LTS, NewAccesses);
  Value *ValueOperand = getNewValue(Stmt, Store->getValueOperand(), BBMap, LTS,
                                    getLoopForInst(Store));

  if (DebugPrinting)
    RuntimeDebugBuilder::createCPUPrinter(Builder, "Store to  ", NewPointer,
                                          ": ", ValueOperand, "\n");

  Builder.CreateAlignedStore(ValueOperand, NewPointer, Store->getAlignment());
}

bool BlockGenerator::canSyntheziseInStmt(ScopStmt &Stmt, Instruction *Inst) {
  Loop *L = getLoopForInst(Inst);
  return (Stmt.isBlockStmt() || !Stmt.getRegion()->contains(L)) &&
         canSynthesize(Inst, &LI, &SE, &Stmt.getParent()->getRegion());
}

void BlockGenerator::copyInstruction(ScopStmt &Stmt, Instruction *Inst,
                                     ValueMapT &BBMap, LoopToScevMapT &LTS,
                                     isl_id_to_ast_expr *NewAccesses) {
  // Terminator instructions control the control flow. They are explicitly
  // expressed in the clast and do not need to be copied.
  if (Inst->isTerminator())
    return;

  // Synthesizable statements will be generated on-demand.
  if (canSyntheziseInStmt(Stmt, Inst))
    return;

  if (auto *Load = dyn_cast<LoadInst>(Inst)) {
    Value *NewLoad = generateScalarLoad(Stmt, Load, BBMap, LTS, NewAccesses);
    // Compute NewLoad before its insertion in BBMap to make the insertion
    // deterministic.
    BBMap[Load] = NewLoad;
    return;
  }

  if (auto *Store = dyn_cast<StoreInst>(Inst)) {
    generateScalarStore(Stmt, Store, BBMap, LTS, NewAccesses);
    return;
  }

  if (auto *PHI = dyn_cast<PHINode>(Inst)) {
    copyPHIInstruction(Stmt, PHI, BBMap, LTS);
    return;
  }

  // Skip some special intrinsics for which we do not adjust the semantics to
  // the new schedule. All others are handled like every other instruction.
  if (isIgnoredIntrinsic(Inst))
    return;

  copyInstScalar(Stmt, Inst, BBMap, LTS);
}

void BlockGenerator::copyStmt(ScopStmt &Stmt, LoopToScevMapT &LTS,
                              isl_id_to_ast_expr *NewAccesses) {
  assert(Stmt.isBlockStmt() &&
         "Only block statements can be copied by the block generator");

  ValueMapT BBMap;

  BasicBlock *BB = Stmt.getBasicBlock();
  copyBB(Stmt, BB, BBMap, LTS, NewAccesses);
}

BasicBlock *BlockGenerator::splitBB(BasicBlock *BB) {
  BasicBlock *CopyBB = SplitBlock(Builder.GetInsertBlock(),
                                  &*Builder.GetInsertPoint(), &DT, &LI);
  CopyBB->setName("polly.stmt." + BB->getName());
  return CopyBB;
}

BasicBlock *BlockGenerator::copyBB(ScopStmt &Stmt, BasicBlock *BB,
                                   ValueMapT &BBMap, LoopToScevMapT &LTS,
                                   isl_id_to_ast_expr *NewAccesses) {
  BasicBlock *CopyBB = splitBB(BB);
  Builder.SetInsertPoint(&CopyBB->front());
  generateScalarLoads(Stmt, BBMap);

  copyBB(Stmt, BB, CopyBB, BBMap, LTS, NewAccesses);

  // After a basic block was copied store all scalars that escape this block in
  // their alloca.
  generateScalarStores(Stmt, LTS, BBMap);
  return CopyBB;
}

void BlockGenerator::copyBB(ScopStmt &Stmt, BasicBlock *BB, BasicBlock *CopyBB,
                            ValueMapT &BBMap, LoopToScevMapT &LTS,
                            isl_id_to_ast_expr *NewAccesses) {
  EntryBB = &CopyBB->getParent()->getEntryBlock();

  for (Instruction &Inst : *BB)
    copyInstruction(Stmt, &Inst, BBMap, LTS, NewAccesses);
}

Value *BlockGenerator::getOrCreateAlloca(Value *ScalarBase,
                                         ScalarAllocaMapTy &Map,
                                         const char *NameExt) {
  // If no alloca was found create one and insert it in the entry block.
  if (!Map.count(ScalarBase)) {
    auto *Ty = ScalarBase->getType();
    auto NewAddr = new AllocaInst(Ty, ScalarBase->getName() + NameExt);
    EntryBB = &Builder.GetInsertBlock()->getParent()->getEntryBlock();
    NewAddr->insertBefore(&*EntryBB->getFirstInsertionPt());
    Map[ScalarBase] = NewAddr;
  }

  auto Addr = Map[ScalarBase];

  if (auto NewAddr = GlobalMap.lookup(Addr))
    return NewAddr;

  return Addr;
}

Value *BlockGenerator::getOrCreateAlloca(const MemoryAccess &Access) {
  if (Access.isPHIKind())
    return getOrCreatePHIAlloca(Access.getBaseAddr());
  else
    return getOrCreateScalarAlloca(Access.getBaseAddr());
}

Value *BlockGenerator::getOrCreateAlloca(const ScopArrayInfo *Array) {
  if (Array->isPHIKind())
    return getOrCreatePHIAlloca(Array->getBasePtr());
  else
    return getOrCreateScalarAlloca(Array->getBasePtr());
}

Value *BlockGenerator::getOrCreateScalarAlloca(Value *ScalarBase) {
  return getOrCreateAlloca(ScalarBase, ScalarMap, ".s2a");
}

Value *BlockGenerator::getOrCreatePHIAlloca(Value *ScalarBase) {
  return getOrCreateAlloca(ScalarBase, PHIOpMap, ".phiops");
}

void BlockGenerator::handleOutsideUsers(const Region &R, Instruction *Inst,
                                        Value *Address) {
  // If there are escape users we get the alloca for this instruction and put it
  // in the EscapeMap for later finalization. Lastly, if the instruction was
  // copied multiple times we already did this and can exit.
  if (EscapeMap.count(Inst))
    return;

  EscapeUserVectorTy EscapeUsers;
  for (User *U : Inst->users()) {

    // Non-instruction user will never escape.
    Instruction *UI = dyn_cast<Instruction>(U);
    if (!UI)
      continue;

    if (R.contains(UI))
      continue;

    EscapeUsers.push_back(UI);
  }

  // Exit if no escape uses were found.
  if (EscapeUsers.empty())
    return;

  // Get or create an escape alloca for this instruction.
  auto *ScalarAddr = Address ? Address : getOrCreateScalarAlloca(Inst);

  // Remember that this instruction has escape uses and the escape alloca.
  EscapeMap[Inst] = std::make_pair(ScalarAddr, std::move(EscapeUsers));
}

void BlockGenerator::generateScalarLoads(ScopStmt &Stmt, ValueMapT &BBMap) {
  for (MemoryAccess *MA : Stmt) {
    if (MA->isArrayKind() || MA->isWrite())
      continue;

    auto *Address = getOrCreateAlloca(*MA);
    BBMap[MA->getBaseAddr()] =
        Builder.CreateLoad(Address, Address->getName() + ".reload");
  }
}

void BlockGenerator::generateScalarStores(ScopStmt &Stmt, LoopToScevMapT &LTS,
                                          ValueMapT &BBMap) {
  Loop *L = LI.getLoopFor(Stmt.getBasicBlock());

  assert(Stmt.isBlockStmt() && "Region statements need to use the "
                               "generateScalarStores() function in the "
                               "RegionGenerator");

  for (MemoryAccess *MA : Stmt) {
    if (MA->isArrayKind() || MA->isRead())
      continue;

    Value *Val = MA->getAccessValue();
    if (MA->isAnyPHIKind()) {
      assert(MA->getIncoming().size() >= 1 &&
             "Block statements have exactly one exiting block, or multiple but "
             "with same incoming block and value");
      assert(std::all_of(MA->getIncoming().begin(), MA->getIncoming().end(),
                         [&](std::pair<BasicBlock *, Value *> p) -> bool {
                           return p.first == Stmt.getBasicBlock();
                         }) &&
             "Incoming block must be statement's block");
      Val = MA->getIncoming()[0].second;
    }
    auto *Address = getOrCreateAlloca(*MA);

    Val = getNewValue(Stmt, Val, BBMap, LTS, L);
    Builder.CreateStore(Val, Address);
  }
}

void BlockGenerator::createScalarInitialization(Scop &S) {
  Region &R = S.getRegion();
  BasicBlock *ExitBB = R.getExit();

  // The split block __just before__ the region and optimized region.
  BasicBlock *SplitBB = R.getEnteringBlock();
  BranchInst *SplitBBTerm = cast<BranchInst>(SplitBB->getTerminator());
  assert(SplitBBTerm->getNumSuccessors() == 2 && "Bad region entering block!");

  // Get the start block of the __optimized__ region.
  BasicBlock *StartBB = SplitBBTerm->getSuccessor(0);
  if (StartBB == R.getEntry())
    StartBB = SplitBBTerm->getSuccessor(1);

  Builder.SetInsertPoint(StartBB->getTerminator());

  for (auto &Pair : S.arrays()) {
    auto &Array = Pair.second;
    if (Array->getNumberOfDimensions() != 0)
      continue;
    if (Array->isPHIKind()) {
      // For PHI nodes, the only values we need to store are the ones that
      // reach the PHI node from outside the region. In general there should
      // only be one such incoming edge and this edge should enter through
      // 'SplitBB'.
      auto PHI = cast<PHINode>(Array->getBasePtr());

      for (auto BI = PHI->block_begin(), BE = PHI->block_end(); BI != BE; BI++)
        if (!R.contains(*BI) && *BI != SplitBB)
          llvm_unreachable("Incoming edges from outside the scop should always "
                           "come from SplitBB");

      int Idx = PHI->getBasicBlockIndex(SplitBB);
      if (Idx < 0)
        continue;

      Value *ScalarValue = PHI->getIncomingValue(Idx);

      Builder.CreateStore(ScalarValue, getOrCreatePHIAlloca(PHI));
      continue;
    }

    auto *Inst = dyn_cast<Instruction>(Array->getBasePtr());

    if (Inst && R.contains(Inst))
      continue;

    // PHI nodes that are not marked as such in their SAI object are either exit
    // PHI nodes we model as common scalars but without initialization, or
    // incoming phi nodes that need to be initialized. Check if the first is the
    // case for Inst and do not create and initialize memory if so.
    if (auto *PHI = dyn_cast_or_null<PHINode>(Inst))
      if (!S.hasSingleExitEdge() && PHI->getBasicBlockIndex(ExitBB) >= 0)
        continue;

    Builder.CreateStore(Array->getBasePtr(),
                        getOrCreateScalarAlloca(Array->getBasePtr()));
  }
}

void BlockGenerator::createScalarFinalization(Region &R) {
  // The exit block of the __unoptimized__ region.
  BasicBlock *ExitBB = R.getExitingBlock();
  // The merge block __just after__ the region and the optimized region.
  BasicBlock *MergeBB = R.getExit();

  // The exit block of the __optimized__ region.
  BasicBlock *OptExitBB = *(pred_begin(MergeBB));
  if (OptExitBB == ExitBB)
    OptExitBB = *(++pred_begin(MergeBB));

  Builder.SetInsertPoint(OptExitBB->getTerminator());
  for (const auto &EscapeMapping : EscapeMap) {
    // Extract the escaping instruction and the escaping users as well as the
    // alloca the instruction was demoted to.
    Instruction *EscapeInst = EscapeMapping.getFirst();
    const auto &EscapeMappingValue = EscapeMapping.getSecond();
    const EscapeUserVectorTy &EscapeUsers = EscapeMappingValue.second;
    Value *ScalarAddr = EscapeMappingValue.first;

    // Reload the demoted instruction in the optimized version of the SCoP.
    Value *EscapeInstReload =
        Builder.CreateLoad(ScalarAddr, EscapeInst->getName() + ".final_reload");
    EscapeInstReload =
        Builder.CreateBitOrPointerCast(EscapeInstReload, EscapeInst->getType());

    // Create the merge PHI that merges the optimized and unoptimized version.
    PHINode *MergePHI = PHINode::Create(EscapeInst->getType(), 2,
                                        EscapeInst->getName() + ".merge");
    MergePHI->insertBefore(&*MergeBB->getFirstInsertionPt());

    // Add the respective values to the merge PHI.
    MergePHI->addIncoming(EscapeInstReload, OptExitBB);
    MergePHI->addIncoming(EscapeInst, ExitBB);

    // The information of scalar evolution about the escaping instruction needs
    // to be revoked so the new merged instruction will be used.
    if (SE.isSCEVable(EscapeInst->getType()))
      SE.forgetValue(EscapeInst);

    // Replace all uses of the demoted instruction with the merge PHI.
    for (Instruction *EUser : EscapeUsers)
      EUser->replaceUsesOfWith(EscapeInst, MergePHI);
  }
}

void BlockGenerator::findOutsideUsers(Scop &S) {
  auto &R = S.getRegion();
  for (auto &Pair : S.arrays()) {
    auto &Array = Pair.second;

    if (Array->getNumberOfDimensions() != 0)
      continue;

    if (Array->isPHIKind())
      continue;

    auto *Inst = dyn_cast<Instruction>(Array->getBasePtr());

    if (!Inst)
      continue;

    // Scop invariant hoisting moves some of the base pointers out of the scop.
    // We can ignore these, as the invariant load hoisting already registers the
    // relevant outside users.
    if (!R.contains(Inst))
      continue;

    handleOutsideUsers(R, Inst, nullptr);
  }
}

void BlockGenerator::createExitPHINodeMerges(Scop &S) {
  if (S.hasSingleExitEdge())
    return;

  Region &R = S.getRegion();

  auto *ExitBB = R.getExitingBlock();
  auto *MergeBB = R.getExit();
  auto *AfterMergeBB = MergeBB->getSingleSuccessor();
  BasicBlock *OptExitBB = *(pred_begin(MergeBB));
  if (OptExitBB == ExitBB)
    OptExitBB = *(++pred_begin(MergeBB));

  Builder.SetInsertPoint(OptExitBB->getTerminator());

  for (auto &Pair : S.arrays()) {
    auto &SAI = Pair.second;
    auto *Val = SAI->getBasePtr();

    PHINode *PHI = dyn_cast<PHINode>(Val);
    if (!PHI)
      continue;

    if (PHI->getParent() != AfterMergeBB)
      continue;

    std::string Name = PHI->getName();
    Value *ScalarAddr = getOrCreateScalarAlloca(PHI);
    Value *Reload = Builder.CreateLoad(ScalarAddr, Name + ".ph.final_reload");
    Reload = Builder.CreateBitOrPointerCast(Reload, PHI->getType());
    Value *OriginalValue = PHI->getIncomingValueForBlock(MergeBB);
    auto *MergePHI = PHINode::Create(PHI->getType(), 2, Name + ".ph.merge");
    MergePHI->insertBefore(&*MergeBB->getFirstInsertionPt());
    MergePHI->addIncoming(Reload, OptExitBB);
    MergePHI->addIncoming(OriginalValue, ExitBB);
    int Idx = PHI->getBasicBlockIndex(MergeBB);
    PHI->setIncomingValue(Idx, MergePHI);
  }
}

void BlockGenerator::finalizeSCoP(Scop &S) {
  findOutsideUsers(S);
  createScalarInitialization(S);
  createExitPHINodeMerges(S);
  createScalarFinalization(S.getRegion());
}

VectorBlockGenerator::VectorBlockGenerator(BlockGenerator &BlockGen,
                                           std::vector<LoopToScevMapT> &VLTS,
                                           isl_map *Schedule)
    : BlockGenerator(BlockGen), VLTS(VLTS), Schedule(Schedule) {
  assert(Schedule && "No statement domain provided");
}

Value *VectorBlockGenerator::getVectorValue(ScopStmt &Stmt, Value *Old,
                                            ValueMapT &VectorMap,
                                            VectorValueMapT &ScalarMaps,
                                            Loop *L) {
  if (Value *NewValue = VectorMap.lookup(Old))
    return NewValue;

  int Width = getVectorWidth();

  Value *Vector = UndefValue::get(VectorType::get(Old->getType(), Width));

  for (int Lane = 0; Lane < Width; Lane++)
    Vector = Builder.CreateInsertElement(
        Vector, getNewValue(Stmt, Old, ScalarMaps[Lane], VLTS[Lane], L),
        Builder.getInt32(Lane));

  VectorMap[Old] = Vector;

  return Vector;
}

Type *VectorBlockGenerator::getVectorPtrTy(const Value *Val, int Width) {
  PointerType *PointerTy = dyn_cast<PointerType>(Val->getType());
  assert(PointerTy && "PointerType expected");

  Type *ScalarType = PointerTy->getElementType();
  VectorType *VectorType = VectorType::get(ScalarType, Width);

  return PointerType::getUnqual(VectorType);
}

Value *VectorBlockGenerator::generateStrideOneLoad(
    ScopStmt &Stmt, LoadInst *Load, VectorValueMapT &ScalarMaps,
    __isl_keep isl_id_to_ast_expr *NewAccesses, bool NegativeStride = false) {
  unsigned VectorWidth = getVectorWidth();
  auto *Pointer = Load->getPointerOperand();
  Type *VectorPtrType = getVectorPtrTy(Pointer, VectorWidth);
  unsigned Offset = NegativeStride ? VectorWidth - 1 : 0;

  Value *NewPointer = nullptr;
  NewPointer = generateLocationAccessed(Stmt, Load, ScalarMaps[Offset],
                                        VLTS[Offset], NewAccesses);
  Value *VectorPtr =
      Builder.CreateBitCast(NewPointer, VectorPtrType, "vector_ptr");
  LoadInst *VecLoad =
      Builder.CreateLoad(VectorPtr, Load->getName() + "_p_vec_full");
  if (!Aligned)
    VecLoad->setAlignment(8);

  if (NegativeStride) {
    SmallVector<Constant *, 16> Indices;
    for (int i = VectorWidth - 1; i >= 0; i--)
      Indices.push_back(ConstantInt::get(Builder.getInt32Ty(), i));
    Constant *SV = llvm::ConstantVector::get(Indices);
    Value *RevVecLoad = Builder.CreateShuffleVector(
        VecLoad, VecLoad, SV, Load->getName() + "_reverse");
    return RevVecLoad;
  }

  return VecLoad;
}

Value *VectorBlockGenerator::generateStrideZeroLoad(
    ScopStmt &Stmt, LoadInst *Load, ValueMapT &BBMap,
    __isl_keep isl_id_to_ast_expr *NewAccesses) {
  auto *Pointer = Load->getPointerOperand();
  Type *VectorPtrType = getVectorPtrTy(Pointer, 1);
  Value *NewPointer =
      generateLocationAccessed(Stmt, Load, BBMap, VLTS[0], NewAccesses);
  Value *VectorPtr = Builder.CreateBitCast(NewPointer, VectorPtrType,
                                           Load->getName() + "_p_vec_p");
  LoadInst *ScalarLoad =
      Builder.CreateLoad(VectorPtr, Load->getName() + "_p_splat_one");

  if (!Aligned)
    ScalarLoad->setAlignment(8);

  Constant *SplatVector = Constant::getNullValue(
      VectorType::get(Builder.getInt32Ty(), getVectorWidth()));

  Value *VectorLoad = Builder.CreateShuffleVector(
      ScalarLoad, ScalarLoad, SplatVector, Load->getName() + "_p_splat");
  return VectorLoad;
}

Value *VectorBlockGenerator::generateUnknownStrideLoad(
    ScopStmt &Stmt, LoadInst *Load, VectorValueMapT &ScalarMaps,
    __isl_keep isl_id_to_ast_expr *NewAccesses) {
  int VectorWidth = getVectorWidth();
  auto *Pointer = Load->getPointerOperand();
  VectorType *VectorType = VectorType::get(
      dyn_cast<PointerType>(Pointer->getType())->getElementType(), VectorWidth);

  Value *Vector = UndefValue::get(VectorType);

  for (int i = 0; i < VectorWidth; i++) {
    Value *NewPointer = generateLocationAccessed(Stmt, Load, ScalarMaps[i],
                                                 VLTS[i], NewAccesses);
    Value *ScalarLoad =
        Builder.CreateLoad(NewPointer, Load->getName() + "_p_scalar_");
    Vector = Builder.CreateInsertElement(
        Vector, ScalarLoad, Builder.getInt32(i), Load->getName() + "_p_vec_");
  }

  return Vector;
}

void VectorBlockGenerator::generateLoad(
    ScopStmt &Stmt, LoadInst *Load, ValueMapT &VectorMap,
    VectorValueMapT &ScalarMaps, __isl_keep isl_id_to_ast_expr *NewAccesses) {
  if (Value *PreloadLoad = GlobalMap.lookup(Load)) {
    VectorMap[Load] = Builder.CreateVectorSplat(getVectorWidth(), PreloadLoad,
                                                Load->getName() + "_p");
    return;
  }

  if (!VectorType::isValidElementType(Load->getType())) {
    for (int i = 0; i < getVectorWidth(); i++)
      ScalarMaps[i][Load] =
          generateScalarLoad(Stmt, Load, ScalarMaps[i], VLTS[i], NewAccesses);
    return;
  }

  const MemoryAccess &Access = Stmt.getArrayAccessFor(Load);

  // Make sure we have scalar values available to access the pointer to
  // the data location.
  extractScalarValues(Load, VectorMap, ScalarMaps);

  Value *NewLoad;
  if (Access.isStrideZero(isl_map_copy(Schedule)))
    NewLoad = generateStrideZeroLoad(Stmt, Load, ScalarMaps[0], NewAccesses);
  else if (Access.isStrideOne(isl_map_copy(Schedule)))
    NewLoad = generateStrideOneLoad(Stmt, Load, ScalarMaps, NewAccesses);
  else if (Access.isStrideX(isl_map_copy(Schedule), -1))
    NewLoad = generateStrideOneLoad(Stmt, Load, ScalarMaps, NewAccesses, true);
  else
    NewLoad = generateUnknownStrideLoad(Stmt, Load, ScalarMaps, NewAccesses);

  VectorMap[Load] = NewLoad;
}

void VectorBlockGenerator::copyUnaryInst(ScopStmt &Stmt, UnaryInstruction *Inst,
                                         ValueMapT &VectorMap,
                                         VectorValueMapT &ScalarMaps) {
  int VectorWidth = getVectorWidth();
  Value *NewOperand = getVectorValue(Stmt, Inst->getOperand(0), VectorMap,
                                     ScalarMaps, getLoopForInst(Inst));

  assert(isa<CastInst>(Inst) && "Can not generate vector code for instruction");

  const CastInst *Cast = dyn_cast<CastInst>(Inst);
  VectorType *DestType = VectorType::get(Inst->getType(), VectorWidth);
  VectorMap[Inst] = Builder.CreateCast(Cast->getOpcode(), NewOperand, DestType);
}

void VectorBlockGenerator::copyBinaryInst(ScopStmt &Stmt, BinaryOperator *Inst,
                                          ValueMapT &VectorMap,
                                          VectorValueMapT &ScalarMaps) {
  Loop *L = getLoopForInst(Inst);
  Value *OpZero = Inst->getOperand(0);
  Value *OpOne = Inst->getOperand(1);

  Value *NewOpZero, *NewOpOne;
  NewOpZero = getVectorValue(Stmt, OpZero, VectorMap, ScalarMaps, L);
  NewOpOne = getVectorValue(Stmt, OpOne, VectorMap, ScalarMaps, L);

  Value *NewInst = Builder.CreateBinOp(Inst->getOpcode(), NewOpZero, NewOpOne,
                                       Inst->getName() + "p_vec");
  VectorMap[Inst] = NewInst;
}

void VectorBlockGenerator::copyStore(
    ScopStmt &Stmt, StoreInst *Store, ValueMapT &VectorMap,
    VectorValueMapT &ScalarMaps, __isl_keep isl_id_to_ast_expr *NewAccesses) {
  const MemoryAccess &Access = Stmt.getArrayAccessFor(Store);

  auto *Pointer = Store->getPointerOperand();
  Value *Vector = getVectorValue(Stmt, Store->getValueOperand(), VectorMap,
                                 ScalarMaps, getLoopForInst(Store));

  // Make sure we have scalar values available to access the pointer to
  // the data location.
  extractScalarValues(Store, VectorMap, ScalarMaps);

  if (Access.isStrideOne(isl_map_copy(Schedule))) {
    Type *VectorPtrType = getVectorPtrTy(Pointer, getVectorWidth());
    Value *NewPointer = generateLocationAccessed(Stmt, Store, ScalarMaps[0],
                                                 VLTS[0], NewAccesses);

    Value *VectorPtr =
        Builder.CreateBitCast(NewPointer, VectorPtrType, "vector_ptr");
    StoreInst *Store = Builder.CreateStore(Vector, VectorPtr);

    if (!Aligned)
      Store->setAlignment(8);
  } else {
    for (unsigned i = 0; i < ScalarMaps.size(); i++) {
      Value *Scalar = Builder.CreateExtractElement(Vector, Builder.getInt32(i));
      Value *NewPointer = generateLocationAccessed(Stmt, Store, ScalarMaps[i],
                                                   VLTS[i], NewAccesses);
      Builder.CreateStore(Scalar, NewPointer);
    }
  }
}

bool VectorBlockGenerator::hasVectorOperands(const Instruction *Inst,
                                             ValueMapT &VectorMap) {
  for (Value *Operand : Inst->operands())
    if (VectorMap.count(Operand))
      return true;
  return false;
}

bool VectorBlockGenerator::extractScalarValues(const Instruction *Inst,
                                               ValueMapT &VectorMap,
                                               VectorValueMapT &ScalarMaps) {
  bool HasVectorOperand = false;
  int VectorWidth = getVectorWidth();

  for (Value *Operand : Inst->operands()) {
    ValueMapT::iterator VecOp = VectorMap.find(Operand);

    if (VecOp == VectorMap.end())
      continue;

    HasVectorOperand = true;
    Value *NewVector = VecOp->second;

    for (int i = 0; i < VectorWidth; ++i) {
      ValueMapT &SM = ScalarMaps[i];

      // If there is one scalar extracted, all scalar elements should have
      // already been extracted by the code here. So no need to check for the
      // existance of all of them.
      if (SM.count(Operand))
        break;

      SM[Operand] =
          Builder.CreateExtractElement(NewVector, Builder.getInt32(i));
    }
  }

  return HasVectorOperand;
}

void VectorBlockGenerator::copyInstScalarized(
    ScopStmt &Stmt, Instruction *Inst, ValueMapT &VectorMap,
    VectorValueMapT &ScalarMaps, __isl_keep isl_id_to_ast_expr *NewAccesses) {
  bool HasVectorOperand;
  int VectorWidth = getVectorWidth();

  HasVectorOperand = extractScalarValues(Inst, VectorMap, ScalarMaps);

  for (int VectorLane = 0; VectorLane < getVectorWidth(); VectorLane++)
    BlockGenerator::copyInstruction(Stmt, Inst, ScalarMaps[VectorLane],
                                    VLTS[VectorLane], NewAccesses);

  if (!VectorType::isValidElementType(Inst->getType()) || !HasVectorOperand)
    return;

  // Make the result available as vector value.
  VectorType *VectorType = VectorType::get(Inst->getType(), VectorWidth);
  Value *Vector = UndefValue::get(VectorType);

  for (int i = 0; i < VectorWidth; i++)
    Vector = Builder.CreateInsertElement(Vector, ScalarMaps[i][Inst],
                                         Builder.getInt32(i));

  VectorMap[Inst] = Vector;
}

int VectorBlockGenerator::getVectorWidth() { return VLTS.size(); }

void VectorBlockGenerator::copyInstruction(
    ScopStmt &Stmt, Instruction *Inst, ValueMapT &VectorMap,
    VectorValueMapT &ScalarMaps, __isl_keep isl_id_to_ast_expr *NewAccesses) {
  // Terminator instructions control the control flow. They are explicitly
  // expressed in the clast and do not need to be copied.
  if (Inst->isTerminator())
    return;

  if (canSyntheziseInStmt(Stmt, Inst))
    return;

  if (auto *Load = dyn_cast<LoadInst>(Inst)) {
    generateLoad(Stmt, Load, VectorMap, ScalarMaps, NewAccesses);
    return;
  }

  if (hasVectorOperands(Inst, VectorMap)) {
    if (auto *Store = dyn_cast<StoreInst>(Inst)) {
      copyStore(Stmt, Store, VectorMap, ScalarMaps, NewAccesses);
      return;
    }

    if (auto *Unary = dyn_cast<UnaryInstruction>(Inst)) {
      copyUnaryInst(Stmt, Unary, VectorMap, ScalarMaps);
      return;
    }

    if (auto *Binary = dyn_cast<BinaryOperator>(Inst)) {
      copyBinaryInst(Stmt, Binary, VectorMap, ScalarMaps);
      return;
    }

    // Falltrough: We generate scalar instructions, if we don't know how to
    // generate vector code.
  }

  copyInstScalarized(Stmt, Inst, VectorMap, ScalarMaps, NewAccesses);
}

void VectorBlockGenerator::generateScalarVectorLoads(
    ScopStmt &Stmt, ValueMapT &VectorBlockMap) {
  for (MemoryAccess *MA : Stmt) {
    if (MA->isArrayKind() || MA->isWrite())
      continue;

    auto *Address = getOrCreateAlloca(*MA);
    Type *VectorPtrType = getVectorPtrTy(Address, 1);
    Value *VectorPtr = Builder.CreateBitCast(Address, VectorPtrType,
                                             Address->getName() + "_p_vec_p");
    auto *Val = Builder.CreateLoad(VectorPtr, Address->getName() + ".reload");
    Constant *SplatVector = Constant::getNullValue(
        VectorType::get(Builder.getInt32Ty(), getVectorWidth()));

    Value *VectorVal = Builder.CreateShuffleVector(
        Val, Val, SplatVector, Address->getName() + "_p_splat");
    VectorBlockMap[MA->getBaseAddr()] = VectorVal;
    VectorVal->dump();
  }
}

void VectorBlockGenerator::verifyNoScalarStores(ScopStmt &Stmt) {
  for (MemoryAccess *MA : Stmt) {
    if (MA->isArrayKind() || MA->isRead())
      continue;

    llvm_unreachable("Scalar stores not expected in vector loop");
  }
}

void VectorBlockGenerator::copyStmt(
    ScopStmt &Stmt, __isl_keep isl_id_to_ast_expr *NewAccesses) {
  assert(Stmt.isBlockStmt() && "TODO: Only block statements can be copied by "
                               "the vector block generator");

  BasicBlock *BB = Stmt.getBasicBlock();
  BasicBlock *CopyBB = SplitBlock(Builder.GetInsertBlock(),
                                  &*Builder.GetInsertPoint(), &DT, &LI);
  CopyBB->setName("polly.stmt." + BB->getName());
  Builder.SetInsertPoint(&CopyBB->front());

  // Create two maps that store the mapping from the original instructions of
  // the old basic block to their copies in the new basic block. Those maps
  // are basic block local.
  //
  // As vector code generation is supported there is one map for scalar values
  // and one for vector values.
  //
  // In case we just do scalar code generation, the vectorMap is not used and
  // the scalarMap has just one dimension, which contains the mapping.
  //
  // In case vector code generation is done, an instruction may either appear
  // in the vector map once (as it is calculating >vectorwidth< values at a
  // time. Or (if the values are calculated using scalar operations), it
  // appears once in every dimension of the scalarMap.
  VectorValueMapT ScalarBlockMap(getVectorWidth());
  ValueMapT VectorBlockMap;

  generateScalarVectorLoads(Stmt, VectorBlockMap);

  for (Instruction &Inst : *BB)
    copyInstruction(Stmt, &Inst, VectorBlockMap, ScalarBlockMap, NewAccesses);

  verifyNoScalarStores(Stmt);
}

BasicBlock *RegionGenerator::repairDominance(BasicBlock *BB,
                                             BasicBlock *BBCopy) {

  BasicBlock *BBIDom = DT.getNode(BB)->getIDom()->getBlock();
  BasicBlock *BBCopyIDom = BlockMap.lookup(BBIDom);

  if (BBCopyIDom)
    DT.changeImmediateDominator(BBCopy, BBCopyIDom);

  return BBCopyIDom;
}

void RegionGenerator::copyStmt(ScopStmt &Stmt, LoopToScevMapT &LTS,
                               isl_id_to_ast_expr *IdToAstExp) {
  assert(Stmt.isRegionStmt() &&
         "Only region statements can be copied by the region generator");

  Scop *S = Stmt.getParent();

  // Forget all old mappings.
  BlockMap.clear();
  RegionMaps.clear();
  IncompletePHINodeMap.clear();

  // Collection of all values related to this subregion.
  ValueMapT ValueMap;

  // The region represented by the statement.
  Region *R = Stmt.getRegion();

  // Create a dedicated entry for the region where we can reload all demoted
  // inputs.
  BasicBlock *EntryBB = R->getEntry();
  BasicBlock *EntryBBCopy = SplitBlock(Builder.GetInsertBlock(),
                                       &*Builder.GetInsertPoint(), &DT, &LI);
  EntryBBCopy->setName("polly.stmt." + EntryBB->getName() + ".entry");
  Builder.SetInsertPoint(&EntryBBCopy->front());

  ValueMapT &EntryBBMap = RegionMaps[EntryBBCopy];
  generateScalarLoads(Stmt, EntryBBMap);

  for (auto PI = pred_begin(EntryBB), PE = pred_end(EntryBB); PI != PE; ++PI)
    if (!R->contains(*PI))
      BlockMap[*PI] = EntryBBCopy;

  // Determine the original exit block of this subregion. If it the exit block
  // is also the scop's exit, it it has been changed to polly.merge_new_and_old.
  // We move one block back to find the original block. This only happens if the
  // scop required simplification.
  // If the whole scop consists of only this non-affine region, then they share
  // the same Region object, such that we cannot change the exit of one and not
  // the other.
  BasicBlock *ExitBB = R->getExit();
  if (!S->hasSingleExitEdge() && ExitBB == S->getRegion().getExit())
    ExitBB = *(++pred_begin(ExitBB));

  // Iterate over all blocks in the region in a breadth-first search.
  std::deque<BasicBlock *> Blocks;
  SmallPtrSet<BasicBlock *, 8> SeenBlocks;
  Blocks.push_back(EntryBB);
  SeenBlocks.insert(EntryBB);

  while (!Blocks.empty()) {
    BasicBlock *BB = Blocks.front();
    Blocks.pop_front();

    // First split the block and update dominance information.
    BasicBlock *BBCopy = splitBB(BB);
    BasicBlock *BBCopyIDom = repairDominance(BB, BBCopy);

    // In order to remap PHI nodes we store also basic block mappings.
    BlockMap[BB] = BBCopy;

    // Get the mapping for this block and initialize it with either the scalar
    // loads from the generated entering block (which dominates all blocks of
    // this subregion) or the maps of the immediate dominator, if part of the
    // subregion. The latter necessarily includes the former.
    ValueMapT *InitBBMap;
    if (BBCopyIDom) {
      assert(RegionMaps.count(BBCopyIDom));
      InitBBMap = &RegionMaps[BBCopyIDom];
    } else
      InitBBMap = &EntryBBMap;
    auto Inserted = RegionMaps.insert(std::make_pair(BBCopy, *InitBBMap));
    ValueMapT &RegionMap = Inserted.first->second;

    // Copy the block with the BlockGenerator.
    Builder.SetInsertPoint(&BBCopy->front());
    copyBB(Stmt, BB, BBCopy, RegionMap, LTS, IdToAstExp);

    // In order to remap PHI nodes we store also basic block mappings.
    BlockMap[BB] = BBCopy;

    // Add values to incomplete PHI nodes waiting for this block to be copied.
    for (const PHINodePairTy &PHINodePair : IncompletePHINodeMap[BB])
      addOperandToPHI(Stmt, PHINodePair.first, PHINodePair.second, BB, LTS);
    IncompletePHINodeMap[BB].clear();

    // And continue with new successors inside the region.
    for (auto SI = succ_begin(BB), SE = succ_end(BB); SI != SE; SI++)
      if (R->contains(*SI) && SeenBlocks.insert(*SI).second)
        Blocks.push_back(*SI);

    // Remember value in case it is visible after this subregion.
    if (DT.dominates(BB, ExitBB))
      ValueMap.insert(RegionMap.begin(), RegionMap.end());
  }

  // Now create a new dedicated region exit block and add it to the region map.
  BasicBlock *ExitBBCopy = SplitBlock(Builder.GetInsertBlock(),
                                      &*Builder.GetInsertPoint(), &DT, &LI);
  ExitBBCopy->setName("polly.stmt." + R->getExit()->getName() + ".exit");
  BlockMap[R->getExit()] = ExitBBCopy;

  if (ExitBB == R->getExit())
    repairDominance(ExitBB, ExitBBCopy);
  else
    DT.changeImmediateDominator(ExitBBCopy, BlockMap.lookup(ExitBB));

  // As the block generator doesn't handle control flow we need to add the
  // region control flow by hand after all blocks have been copied.
  for (BasicBlock *BB : SeenBlocks) {

    BasicBlock *BBCopy = BlockMap[BB];
    TerminatorInst *TI = BB->getTerminator();
    if (isa<UnreachableInst>(TI)) {
      while (!BBCopy->empty())
        BBCopy->begin()->eraseFromParent();
      new UnreachableInst(BBCopy->getContext(), BBCopy);
      continue;
    }

    Instruction *BICopy = BBCopy->getTerminator();

    ValueMapT &RegionMap = RegionMaps[BBCopy];
    RegionMap.insert(BlockMap.begin(), BlockMap.end());

    Builder.SetInsertPoint(BICopy);
    copyInstScalar(Stmt, TI, RegionMap, LTS);
    BICopy->eraseFromParent();
  }

  // Add counting PHI nodes to all loops in the region that can be used as
  // replacement for SCEVs refering to the old loop.
  for (BasicBlock *BB : SeenBlocks) {
    Loop *L = LI.getLoopFor(BB);
    if (L == nullptr || L->getHeader() != BB || !R->contains(L))
      continue;

    BasicBlock *BBCopy = BlockMap[BB];
    Value *NullVal = Builder.getInt32(0);
    PHINode *LoopPHI =
        PHINode::Create(Builder.getInt32Ty(), 2, "polly.subregion.iv");
    Instruction *LoopPHIInc = BinaryOperator::CreateAdd(
        LoopPHI, Builder.getInt32(1), "polly.subregion.iv.inc");
    LoopPHI->insertBefore(&BBCopy->front());
    LoopPHIInc->insertBefore(BBCopy->getTerminator());

    for (auto *PredBB : make_range(pred_begin(BB), pred_end(BB))) {
      if (!R->contains(PredBB))
        continue;
      if (L->contains(PredBB))
        LoopPHI->addIncoming(LoopPHIInc, BlockMap[PredBB]);
      else
        LoopPHI->addIncoming(NullVal, BlockMap[PredBB]);
    }

    for (auto *PredBBCopy : make_range(pred_begin(BBCopy), pred_end(BBCopy)))
      if (LoopPHI->getBasicBlockIndex(PredBBCopy) < 0)
        LoopPHI->addIncoming(NullVal, PredBBCopy);

    LTS[L] = SE.getUnknown(LoopPHI);
  }

  // Continue generating code in the exit block.
  Builder.SetInsertPoint(&*ExitBBCopy->getFirstInsertionPt());

  // Write values visible to other statements.
  generateScalarStores(Stmt, LTS, ValueMap);
  BlockMap.clear();
  RegionMaps.clear();
  IncompletePHINodeMap.clear();
}

PHINode *RegionGenerator::buildExitPHI(MemoryAccess *MA, LoopToScevMapT &LTS,
                                       ValueMapT &BBMap, Loop *L) {
  ScopStmt *Stmt = MA->getStatement();
  Region *SubR = Stmt->getRegion();
  auto Incoming = MA->getIncoming();

  PollyIRBuilder::InsertPointGuard IPGuard(Builder);
  PHINode *OrigPHI = cast<PHINode>(MA->getAccessInstruction());
  BasicBlock *NewSubregionExit = Builder.GetInsertBlock();

  // This can happen if the subregion is simplified after the ScopStmts
  // have been created; simplification happens as part of CodeGeneration.
  if (OrigPHI->getParent() != SubR->getExit()) {
    BasicBlock *FormerExit = SubR->getExitingBlock();
    if (FormerExit)
      NewSubregionExit = BlockMap.lookup(FormerExit);
  }

  PHINode *NewPHI = PHINode::Create(OrigPHI->getType(), Incoming.size(),
                                    "polly." + OrigPHI->getName(),
                                    NewSubregionExit->getFirstNonPHI());

  // Add the incoming values to the PHI.
  for (auto &Pair : Incoming) {
    BasicBlock *OrigIncomingBlock = Pair.first;
    BasicBlock *NewIncomingBlock = BlockMap.lookup(OrigIncomingBlock);
    Builder.SetInsertPoint(NewIncomingBlock->getTerminator());
    assert(RegionMaps.count(NewIncomingBlock));
    ValueMapT *LocalBBMap = &RegionMaps[NewIncomingBlock];

    Value *OrigIncomingValue = Pair.second;
    Value *NewIncomingValue =
        getNewValue(*Stmt, OrigIncomingValue, *LocalBBMap, LTS, L);
    NewPHI->addIncoming(NewIncomingValue, NewIncomingBlock);
  }

  return NewPHI;
}

Value *RegionGenerator::getExitScalar(MemoryAccess *MA, LoopToScevMapT &LTS,
                                      ValueMapT &BBMap) {
  ScopStmt *Stmt = MA->getStatement();

  // TODO: Add some test cases that ensure this is really the right choice.
  Loop *L = LI.getLoopFor(Stmt->getRegion()->getExit());

  if (MA->isAnyPHIKind()) {
    auto Incoming = MA->getIncoming();
    assert(!Incoming.empty() &&
           "PHI WRITEs must have originate from at least one incoming block");

    // If there is only one incoming value, we do not need to create a PHI.
    if (Incoming.size() == 1) {
      Value *OldVal = Incoming[0].second;
      return getNewValue(*Stmt, OldVal, BBMap, LTS, L);
    }

    return buildExitPHI(MA, LTS, BBMap, L);
  }

  // MK_Value accesses leaving the subregion must dominate the exit block; just
  // pass the copied value
  Value *OldVal = MA->getAccessValue();
  return getNewValue(*Stmt, OldVal, BBMap, LTS, L);
}

void RegionGenerator::generateScalarStores(ScopStmt &Stmt, LoopToScevMapT &LTS,
                                           ValueMapT &BBMap) {
  assert(Stmt.getRegion() &&
         "Block statements need to use the generateScalarStores() "
         "function in the BlockGenerator");

  for (MemoryAccess *MA : Stmt) {
    if (MA->isArrayKind() || MA->isRead())
      continue;

    Value *NewVal = getExitScalar(MA, LTS, BBMap);
    Value *Address = getOrCreateAlloca(*MA);
    Builder.CreateStore(NewVal, Address);
  }
}

void RegionGenerator::addOperandToPHI(ScopStmt &Stmt, const PHINode *PHI,
                                      PHINode *PHICopy, BasicBlock *IncomingBB,
                                      LoopToScevMapT &LTS) {
  Region *StmtR = Stmt.getRegion();

  // If the incoming block was not yet copied mark this PHI as incomplete.
  // Once the block will be copied the incoming value will be added.
  BasicBlock *BBCopy = BlockMap[IncomingBB];
  if (!BBCopy) {
    assert(StmtR->contains(IncomingBB) &&
           "Bad incoming block for PHI in non-affine region");
    IncompletePHINodeMap[IncomingBB].push_back(std::make_pair(PHI, PHICopy));
    return;
  }

  Value *OpCopy = nullptr;
  if (StmtR->contains(IncomingBB)) {
    assert(RegionMaps.count(BBCopy) &&
           "Incoming PHI block did not have a BBMap");
    ValueMapT &BBCopyMap = RegionMaps[BBCopy];

    Value *Op = PHI->getIncomingValueForBlock(IncomingBB);

    BasicBlock *OldBlock = Builder.GetInsertBlock();
    auto OldIP = Builder.GetInsertPoint();
    Builder.SetInsertPoint(BBCopy->getTerminator());
    OpCopy = getNewValue(Stmt, Op, BBCopyMap, LTS, getLoopForInst(PHI));
    Builder.SetInsertPoint(OldBlock, OldIP);
  } else {

    if (PHICopy->getBasicBlockIndex(BBCopy) >= 0)
      return;

    Value *PHIOpAddr = getOrCreatePHIAlloca(const_cast<PHINode *>(PHI));
    OpCopy = new LoadInst(PHIOpAddr, PHIOpAddr->getName() + ".reload",
                          BlockMap[IncomingBB]->getTerminator());
  }

  assert(OpCopy && "Incoming PHI value was not copied properly");
  assert(BBCopy && "Incoming PHI block was not copied properly");
  PHICopy->addIncoming(OpCopy, BBCopy);
}

Value *RegionGenerator::copyPHIInstruction(ScopStmt &Stmt, PHINode *PHI,
                                           ValueMapT &BBMap,
                                           LoopToScevMapT &LTS) {
  unsigned NumIncoming = PHI->getNumIncomingValues();
  PHINode *PHICopy =
      Builder.CreatePHI(PHI->getType(), NumIncoming, "polly." + PHI->getName());
  PHICopy->moveBefore(PHICopy->getParent()->getFirstNonPHI());
  BBMap[PHI] = PHICopy;

  for (unsigned u = 0; u < NumIncoming; u++)
    addOperandToPHI(Stmt, PHI, PHICopy, PHI->getIncomingBlock(u), LTS);
  return PHICopy;
}
