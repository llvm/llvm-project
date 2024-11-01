//===-- IRMutator.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/FuzzMutate/IRMutator.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/FuzzMutate/Operations.h"
#include "llvm/FuzzMutate/Random.h"
#include "llvm/FuzzMutate/RandomIRBuilder.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Scalar/DCE.h"

using namespace llvm;

static void createEmptyFunction(Module &M) {
  // TODO: Some arguments and a return value would probably be more interesting.
  LLVMContext &Context = M.getContext();
  Function *F = Function::Create(FunctionType::get(Type::getVoidTy(Context), {},
                                                   /*isVarArg=*/false),
                                 GlobalValue::ExternalLinkage, "f", &M);
  BasicBlock *BB = BasicBlock::Create(Context, "BB", F);
  ReturnInst::Create(Context, BB);
}

void IRMutationStrategy::mutate(Module &M, RandomIRBuilder &IB) {
  auto RS = makeSampler<Function *>(IB.Rand);
  for (Function &F : M)
    if (!F.isDeclaration())
      RS.sample(&F, /*Weight=*/1);

  if (RS.isEmpty())
    createEmptyFunction(M);
  else
    mutate(*RS.getSelection(), IB);
}

void IRMutationStrategy::mutate(Function &F, RandomIRBuilder &IB) {
  mutate(*makeSampler(IB.Rand, make_pointer_range(F)).getSelection(), IB);
}

void IRMutationStrategy::mutate(BasicBlock &BB, RandomIRBuilder &IB) {
  mutate(*makeSampler(IB.Rand, make_pointer_range(BB)).getSelection(), IB);
}

void IRMutator::mutateModule(Module &M, int Seed, size_t CurSize,
                             size_t MaxSize) {
  std::vector<Type *> Types;
  for (const auto &Getter : AllowedTypes)
    Types.push_back(Getter(M.getContext()));
  RandomIRBuilder IB(Seed, Types);

  auto RS = makeSampler<IRMutationStrategy *>(IB.Rand);
  for (const auto &Strategy : Strategies)
    RS.sample(Strategy.get(),
              Strategy->getWeight(CurSize, MaxSize, RS.totalWeight()));
  auto Strategy = RS.getSelection();

  Strategy->mutate(M, IB);
}

static void eliminateDeadCode(Function &F) {
  FunctionPassManager FPM;
  FPM.addPass(DCEPass());
  FunctionAnalysisManager FAM;
  FAM.registerPass([&] { return TargetLibraryAnalysis(); });
  FAM.registerPass([&] { return PassInstrumentationAnalysis(); });
  FPM.run(F, FAM);
}

void InjectorIRStrategy::mutate(Function &F, RandomIRBuilder &IB) {
  IRMutationStrategy::mutate(F, IB);
  eliminateDeadCode(F);
}

std::vector<fuzzerop::OpDescriptor> InjectorIRStrategy::getDefaultOps() {
  std::vector<fuzzerop::OpDescriptor> Ops;
  describeFuzzerIntOps(Ops);
  describeFuzzerFloatOps(Ops);
  describeFuzzerControlFlowOps(Ops);
  describeFuzzerPointerOps(Ops);
  describeFuzzerAggregateOps(Ops);
  describeFuzzerVectorOps(Ops);
  return Ops;
}

Optional<fuzzerop::OpDescriptor>
InjectorIRStrategy::chooseOperation(Value *Src, RandomIRBuilder &IB) {
  auto OpMatchesPred = [&Src](fuzzerop::OpDescriptor &Op) {
    return Op.SourcePreds[0].matches({}, Src);
  };
  auto RS = makeSampler(IB.Rand, make_filter_range(Operations, OpMatchesPred));
  if (RS.isEmpty())
    return std::nullopt;
  return *RS;
}

void InjectorIRStrategy::mutate(BasicBlock &BB, RandomIRBuilder &IB) {
  SmallVector<Instruction *, 32> Insts;
  for (auto I = BB.getFirstInsertionPt(), E = BB.end(); I != E; ++I)
    Insts.push_back(&*I);
  if (Insts.size() < 1)
    return;

  // Choose an insertion point for our new instruction.
  size_t IP = uniform<size_t>(IB.Rand, 0, Insts.size() - 1);

  auto InstsBefore = makeArrayRef(Insts).slice(0, IP);
  auto InstsAfter = makeArrayRef(Insts).slice(IP);

  // Choose a source, which will be used to constrain the operation selection.
  SmallVector<Value *, 2> Srcs;
  Srcs.push_back(IB.findOrCreateSource(BB, InstsBefore));

  // Choose an operation that's constrained to be valid for the type of the
  // source, collect any other sources it needs, and then build it.
  auto OpDesc = chooseOperation(Srcs[0], IB);
  // Bail if no operation was found
  if (!OpDesc)
    return;

  for (const auto &Pred : makeArrayRef(OpDesc->SourcePreds).slice(1))
    Srcs.push_back(IB.findOrCreateSource(BB, InstsBefore, Srcs, Pred));

  if (Value *Op = OpDesc->BuilderFunc(Srcs, Insts[IP])) {
    // Find a sink and wire up the results of the operation.
    IB.connectToSink(BB, InstsAfter, Op);
  }
}

uint64_t InstDeleterIRStrategy::getWeight(size_t CurrentSize, size_t MaxSize,
                                          uint64_t CurrentWeight) {
  // If we have less than 200 bytes, panic and try to always delete.
  if (CurrentSize > MaxSize - 200)
    return CurrentWeight ? CurrentWeight * 100 : 1;
  // Draw a line starting from when we only have 1k left and increasing linearly
  // to double the current weight.
  int64_t Line = (-2 * static_cast<int64_t>(CurrentWeight)) *
                 (static_cast<int64_t>(MaxSize) -
                  static_cast<int64_t>(CurrentSize) - 1000) /
                 1000;
  // Clamp negative weights to zero.
  if (Line < 0)
    return 0;
  return Line;
}

void InstDeleterIRStrategy::mutate(Function &F, RandomIRBuilder &IB) {
  auto RS = makeSampler<Instruction *>(IB.Rand);
  for (Instruction &Inst : instructions(F)) {
    // TODO: We can't handle these instructions.
    if (Inst.isTerminator() || Inst.isEHPad() || Inst.isSwiftError() ||
        isa<PHINode>(Inst))
      continue;

    RS.sample(&Inst, /*Weight=*/1);
  }
  if (RS.isEmpty())
    return;

  // Delete the instruction.
  mutate(*RS.getSelection(), IB);
  // Clean up any dead code that's left over after removing the instruction.
  eliminateDeadCode(F);
}

void InstDeleterIRStrategy::mutate(Instruction &Inst, RandomIRBuilder &IB) {
  assert(!Inst.isTerminator() && "Deleting terminators invalidates CFG");

  if (Inst.getType()->isVoidTy()) {
    // Instructions with void type (ie, store) have no uses to worry about. Just
    // erase it and move on.
    Inst.eraseFromParent();
    return;
  }

  // Otherwise we need to find some other value with the right type to keep the
  // users happy.
  auto Pred = fuzzerop::onlyType(Inst.getType());
  auto RS = makeSampler<Value *>(IB.Rand);
  SmallVector<Instruction *, 32> InstsBefore;
  BasicBlock *BB = Inst.getParent();
  for (auto I = BB->getFirstInsertionPt(), E = Inst.getIterator(); I != E;
       ++I) {
    if (Pred.matches({}, &*I))
      RS.sample(&*I, /*Weight=*/1);
    InstsBefore.push_back(&*I);
  }
  if (!RS)
    RS.sample(IB.newSource(*BB, InstsBefore, {}, Pred), /*Weight=*/1);

  Inst.replaceAllUsesWith(RS.getSelection());
  Inst.eraseFromParent();
}

void InstModificationIRStrategy::mutate(Instruction &Inst,
                                        RandomIRBuilder &IB) {
  SmallVector<std::function<void()>, 8> Modifications;
  CmpInst *CI = nullptr;
  GetElementPtrInst *GEP = nullptr;
  switch (Inst.getOpcode()) {
  default:
    break;
  case Instruction::Add:
  case Instruction::Mul:
  case Instruction::Sub:
  case Instruction::Shl:
    Modifications.push_back([&Inst]() { Inst.setHasNoSignedWrap(true); });
    Modifications.push_back([&Inst]() { Inst.setHasNoSignedWrap(false); });
    Modifications.push_back([&Inst]() { Inst.setHasNoUnsignedWrap(true); });
    Modifications.push_back([&Inst]() { Inst.setHasNoUnsignedWrap(false); });

    break;
  case Instruction::ICmp:
    CI = cast<ICmpInst>(&Inst);
    Modifications.push_back([CI]() { CI->setPredicate(CmpInst::ICMP_EQ); });
    Modifications.push_back([CI]() { CI->setPredicate(CmpInst::ICMP_NE); });
    Modifications.push_back([CI]() { CI->setPredicate(CmpInst::ICMP_UGT); });
    Modifications.push_back([CI]() { CI->setPredicate(CmpInst::ICMP_UGE); });
    Modifications.push_back([CI]() { CI->setPredicate(CmpInst::ICMP_ULT); });
    Modifications.push_back([CI]() { CI->setPredicate(CmpInst::ICMP_ULE); });
    Modifications.push_back([CI]() { CI->setPredicate(CmpInst::ICMP_SGT); });
    Modifications.push_back([CI]() { CI->setPredicate(CmpInst::ICMP_SGE); });
    Modifications.push_back([CI]() { CI->setPredicate(CmpInst::ICMP_SLT); });
    Modifications.push_back([CI]() { CI->setPredicate(CmpInst::ICMP_SLE); });
    break;
  case Instruction::GetElementPtr:
    GEP = cast<GetElementPtrInst>(&Inst);
    Modifications.push_back([GEP]() { GEP->setIsInBounds(true); });
    Modifications.push_back([GEP]() { GEP->setIsInBounds(false); });
    break;
  }

  // Randomly switch operands of instructions
  std::pair<int, int> NoneItem({-1, -1}), ShuffleItems(NoneItem);
  switch (Inst.getOpcode()) {
  case Instruction::SDiv:
  case Instruction::UDiv:
  case Instruction::SRem:
  case Instruction::URem:
  case Instruction::FDiv:
  case Instruction::FRem: {
    // Verify that the after shuffle the second operand is not
    // constant 0.
    Value *Operand = Inst.getOperand(0);
    if (Constant *C = dyn_cast<Constant>(Operand)) {
      if (!C->isZeroValue()) {
        ShuffleItems = {0, 1};
      }
    }
    break;
  }
  case Instruction::Select:
    ShuffleItems = {1, 2};
    break;
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
  case Instruction::FAdd:
  case Instruction::FSub:
  case Instruction::FMul:
  case Instruction::ICmp:
  case Instruction::FCmp:
  case Instruction::ShuffleVector:
    ShuffleItems = {0, 1};
    break;
  }
  if (ShuffleItems != NoneItem) {
    Modifications.push_back([&Inst, &ShuffleItems]() {
      Value *Op0 = Inst.getOperand(ShuffleItems.first);
      Inst.setOperand(ShuffleItems.first, Inst.getOperand(ShuffleItems.second));
      Inst.setOperand(ShuffleItems.second, Op0);
    });
  }

  auto RS = makeSampler(IB.Rand, Modifications);
  if (RS)
    RS.getSelection()();
}

void InsertPHIStrategy::mutate(BasicBlock &BB, RandomIRBuilder &IB) {
  // Can't insert PHI node to entry node.
  if (&BB == &BB.getParent()->getEntryBlock())
    return;
  Type *Ty = IB.randomType();
  PHINode *PHI = PHINode::Create(Ty, llvm::pred_size(&BB), "", &BB.front());

  // Use a map to make sure the same incoming basic block has the same value.
  DenseMap<BasicBlock *, Value *> IncomingValues;
  for (BasicBlock *Pred : predecessors(&BB)) {
    Value *Src = IncomingValues[Pred];
    // If `Pred` is not in the map yet, we'll get a nullptr.
    if (!Src) {
      SmallVector<Instruction *, 32> Insts;
      for (auto I = Pred->begin(); I != Pred->end(); ++I)
        Insts.push_back(&*I);
      // There is no need to inform IB what previously used values are if we are
      // using `onlyType`
      Src = IB.findOrCreateSource(*Pred, Insts, {}, fuzzerop::onlyType(Ty));
      IncomingValues[Pred] = Src;
    }
    PHI->addIncoming(Src, Pred);
  }
  SmallVector<Instruction *, 32> InstsAfter;
  for (auto I = BB.getFirstInsertionPt(), E = BB.end(); I != E; ++I)
    InstsAfter.push_back(&*I);
  IB.connectToSink(BB, InstsAfter, PHI);
}

void SinkInstructionStrategy::mutate(Function &F, RandomIRBuilder &IB) {
  for (BasicBlock &BB : F) {
    this->mutate(BB, IB);
  }
}
void SinkInstructionStrategy::mutate(BasicBlock &BB, RandomIRBuilder &IB) {
  SmallVector<Instruction *, 32> Insts;
  for (auto I = BB.getFirstInsertionPt(), E = BB.end(); I != E; ++I)
    Insts.push_back(&*I);
  if (Insts.size() < 1)
    return;
  // Choose an Instruction to mutate.
  uint64_t Idx = uniform<uint64_t>(IB.Rand, 0, Insts.size() - 1);
  Instruction *Inst = Insts[Idx];
  // `Idx + 1` so we don't sink to ourselves.
  auto InstsAfter = makeArrayRef(Insts).slice(Idx + 1);
  LLVMContext &C = BB.getParent()->getParent()->getContext();
  // Don't sink terminators, void function calls, etc.
  if (Inst->getType() != Type::getVoidTy(C))
    // Find a new sink and wire up the results of the operation.
    IB.connectToSink(BB, InstsAfter, Inst);
}

void ShuffleBlockStrategy::mutate(BasicBlock &BB, RandomIRBuilder &IB) {

  SmallPtrSet<Instruction *, 8> AliveInsts;
  for (auto &I : make_early_inc_range(make_range(
           BB.getFirstInsertionPt(), BB.getTerminator()->getIterator()))) {
    // First gather all instructions that can be shuffled. Don't take
    // terminator.
    AliveInsts.insert(&I);
    // Then remove these instructions from the block
    I.removeFromParent();
  }

  // Shuffle these instructions using topological sort.
  // Returns true if all current instruction's dependencies in this block have
  // been shuffled. If so, this instruction can be shuffled too.
  auto hasAliveParent = [&AliveInsts](Instruction *I) {
    for (Value *O : I->operands()) {
      Instruction *P = dyn_cast<Instruction>(O);
      if (P && AliveInsts.count(P))
        return true;
    }
    return false;
  };
  // Get all alive instructions that depend on the current instruction.
  auto getAliveChildren = [&AliveInsts](Instruction *I) {
    SmallPtrSet<Instruction *, 4> Children;
    for (Value *U : I->users()) {
      Instruction *P = dyn_cast<Instruction>(U);
      if (P && AliveInsts.count(P))
        Children.insert(P);
    }
    return Children;
  };
  SmallPtrSet<Instruction *, 8> Roots;
  SmallVector<Instruction *, 8> Insts;
  for (Instruction *I : AliveInsts) {
    if (!hasAliveParent(I))
      Roots.insert(I);
  }
  // Topological sort by randomly selecting a node without a parent, or root.
  while (!Roots.empty()) {
    auto RS = makeSampler<Instruction *>(IB.Rand);
    for (Instruction *Root : Roots)
      RS.sample(Root, 1);
    Instruction *Root = RS.getSelection();
    Roots.erase(Root);
    AliveInsts.erase(Root);
    Insts.push_back(Root);
    for (Instruction *Child : getAliveChildren(Root)) {
      if (!hasAliveParent(Child)) {
        Roots.insert(Child);
      }
    }
  }

  Instruction *Terminator = BB.getTerminator();
  // Then put instructions back.
  for (Instruction *I : Insts) {
    I->insertBefore(Terminator);
  }
}

std::unique_ptr<Module> llvm::parseModule(const uint8_t *Data, size_t Size,
                                          LLVMContext &Context) {

  if (Size <= 1)
    // We get bogus data given an empty corpus - just create a new module.
    return std::make_unique<Module>("M", Context);

  auto Buffer = MemoryBuffer::getMemBuffer(
      StringRef(reinterpret_cast<const char *>(Data), Size), "Fuzzer input",
      /*RequiresNullTerminator=*/false);

  SMDiagnostic Err;
  auto M = parseBitcodeFile(Buffer->getMemBufferRef(), Context);
  if (Error E = M.takeError()) {
    errs() << toString(std::move(E)) << "\n";
    return nullptr;
  }
  return std::move(M.get());
}

size_t llvm::writeModule(const Module &M, uint8_t *Dest, size_t MaxSize) {
  std::string Buf;
  {
    raw_string_ostream OS(Buf);
    WriteBitcodeToFile(M, OS);
  }
  if (Buf.size() > MaxSize)
    return 0;
  memcpy(Dest, Buf.data(), Buf.size());
  return Buf.size();
}

std::unique_ptr<Module> llvm::parseAndVerify(const uint8_t *Data, size_t Size,
                                             LLVMContext &Context) {
  auto M = parseModule(Data, Size, Context);
  if (!M || verifyModule(*M, &errs()))
    return nullptr;

  return M;
}
