//===-- IRMutator.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/FuzzMutate/IRMutator.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/FuzzMutate/Operations.h"
#include "llvm/FuzzMutate/Random.h"
#include "llvm/FuzzMutate/RandomIRBuilder.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
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
    return None;
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
    if (Inst.isTerminator() || Inst.isEHPad() ||
        Inst.isSwiftError() || isa<PHINode>(Inst))
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
    Modifications.push_back([&Inst]() { Inst.setHasNoSignedWrap(true); }),
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

  auto RS = makeSampler(IB.Rand, Modifications);
  if (RS)
    RS.getSelection()();
}
