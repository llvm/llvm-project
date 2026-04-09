//===-- StructuralHash.cpp - IR Hashing -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/StructuralHash.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace {

// Basic hashing mechanism to detect structural change to the IR, used to verify
// pass return status consistency with actual change. In addition to being used
// by the MergeFunctions pass.

class StructuralHashImpl {
  stable_hash Hash = 4;

  bool DetailedHash;

  // This random value acts as a block header, as otherwise the partition of
  // opcodes into BBs wouldn't affect the hash, only the order of the opcodes.
  static constexpr stable_hash BlockHeaderHash = 45798;
  static constexpr stable_hash FunctionHeaderHash = 0x62642d6b6b2d6b72;
  static constexpr stable_hash GlobalHeaderHash = 23456;

  /// IgnoreOp is a function that returns true if the operand should be ignored.
  IgnoreOperandFunc IgnoreOp = nullptr;
  /// A mapping from instruction indices to instruction pointers.
  /// The index represents the position of an instruction based on the order in
  /// which it is first encountered.
  std::unique_ptr<IndexInstrMap> IndexInstruction = nullptr;
  /// A mapping from pairs of instruction indices and operand indices
  /// to the hashes of the operands.
  std::unique_ptr<IndexOperandHashMapType> IndexOperandHashMap = nullptr;

  /// Assign a unique ID to each Value in the order they are first seen.
  DenseMap<const Value *, int> ValueToId;

  static stable_hash hashType(Type *ValueType) {
    SmallVector<stable_hash> Hashes;
    Hashes.emplace_back(ValueType->getTypeID());
    if (ValueType->isIntegerTy())
      Hashes.emplace_back(ValueType->getIntegerBitWidth());
    return stable_hash_combine(Hashes);
  }

public:
  StructuralHashImpl() = delete;
  explicit StructuralHashImpl(bool DetailedHash,
                              IgnoreOperandFunc IgnoreOp = nullptr)
      : DetailedHash(DetailedHash), IgnoreOp(IgnoreOp) {
    if (IgnoreOp) {
      IndexInstruction = std::make_unique<IndexInstrMap>();
      IndexOperandHashMap = std::make_unique<IndexOperandHashMapType>();
    }
  }

  static stable_hash hashAPInt(const APInt &I) {
    SmallVector<stable_hash> Hashes;
    Hashes.emplace_back(I.getBitWidth());
    auto RawVals = ArrayRef<uint64_t>(I.getRawData(), I.getNumWords());
    Hashes.append(RawVals.begin(), RawVals.end());
    return stable_hash_combine(Hashes);
  }

  static stable_hash hashAPFloat(const APFloat &F) {
    return hashAPInt(F.bitcastToAPInt());
  }

  static stable_hash hashGlobalVariable(const GlobalVariable &GVar) {
    if (!GVar.hasInitializer())
      return hashGlobalValue(&GVar);

    // Hash the contents of a string.
    if (GVar.getName().starts_with(".str")) {
      auto *C = GVar.getInitializer();
      if (const auto *Seq = dyn_cast<ConstantDataSequential>(C))
        if (Seq->isString())
          return stable_hash_name(Seq->getAsString());
    }

    // Hash structural contents of Objective-C metadata in specific sections.
    // This can be extended to other metadata if needed.
    static constexpr const char *SectionNames[] = {
        "__cfstring",      "__cstring",      "__objc_classrefs",
        "__objc_methname", "__objc_selrefs",
    };
    if (GVar.hasSection()) {
      StringRef SectionName = GVar.getSection();
      for (const char *Name : SectionNames)
        if (SectionName.contains(Name))
          return hashConstant(GVar.getInitializer());
    }

    return hashGlobalValue(&GVar);
  }

  static stable_hash hashGlobalValue(const GlobalValue *GV) {
    if (!GV->hasName())
      return 0;
    return stable_hash_name(GV->getName());
  }

  // Compute a hash for a Constant. This function is logically similar to
  // FunctionComparator::cmpConstants() in FunctionComparator.cpp, but here
  // we're interested in computing a hash rather than comparing two Constants.
  // Some of the logic is simplified, e.g, we don't expand GEPOperator.
  static stable_hash hashConstant(const Constant *C) {
    SmallVector<stable_hash> Hashes;

    Type *Ty = C->getType();
    Hashes.emplace_back(hashType(Ty));

    if (C->isNullValue()) {
      Hashes.emplace_back(static_cast<stable_hash>('N'));
      return stable_hash_combine(Hashes);
    }

    if (auto *GVar = dyn_cast<GlobalVariable>(C)) {
      Hashes.emplace_back(hashGlobalVariable(*GVar));
      return stable_hash_combine(Hashes);
    }

    if (auto *G = dyn_cast<GlobalValue>(C)) {
      Hashes.emplace_back(hashGlobalValue(G));
      return stable_hash_combine(Hashes);
    }

    if (const auto *Seq = dyn_cast<ConstantDataSequential>(C)) {
      if (Seq->isString()) {
        Hashes.emplace_back(stable_hash_name(Seq->getAsString()));
        return stable_hash_combine(Hashes);
      }
    }

    switch (C->getValueID()) {
    case Value::ConstantIntVal: {
      const APInt &Int = cast<ConstantInt>(C)->getValue();
      Hashes.emplace_back(hashAPInt(Int));
      return stable_hash_combine(Hashes);
    }
    case Value::ConstantFPVal: {
      const APFloat &APF = cast<ConstantFP>(C)->getValueAPF();
      Hashes.emplace_back(hashAPFloat(APF));
      return stable_hash_combine(Hashes);
    }
    case Value::ConstantArrayVal:
    case Value::ConstantStructVal:
    case Value::ConstantVectorVal:
    case Value::ConstantExprVal: {
      for (const auto &Op : C->operands()) {
        auto H = hashConstant(cast<Constant>(Op));
        Hashes.emplace_back(H);
      }
      return stable_hash_combine(Hashes);
    }
    case Value::BlockAddressVal: {
      const BlockAddress *BA = cast<BlockAddress>(C);
      auto H = hashGlobalValue(BA->getFunction());
      Hashes.emplace_back(H);
      return stable_hash_combine(Hashes);
    }
    case Value::DSOLocalEquivalentVal: {
      const auto *Equiv = cast<DSOLocalEquivalent>(C);
      auto H = hashGlobalValue(Equiv->getGlobalValue());
      Hashes.emplace_back(H);
      return stable_hash_combine(Hashes);
    }
    default:
      // Skip other types of constants for simplicity.
      return stable_hash_combine(Hashes);
    }
  }

  stable_hash hashValue(Value *V) {
    // Check constant and return its hash.
    Constant *C = dyn_cast<Constant>(V);
    if (C)
      return hashConstant(C);

    // Hash argument number.
    SmallVector<stable_hash> Hashes;
    if (Argument *Arg = dyn_cast<Argument>(V))
      Hashes.emplace_back(Arg->getArgNo());

    // Get an index (an insertion order) for the non-constant value.
    auto [It, WasInserted] = ValueToId.try_emplace(V, ValueToId.size());
    Hashes.emplace_back(It->second);

    return stable_hash_combine(Hashes);
  }

  stable_hash hashOperand(Value *Operand) {
    SmallVector<stable_hash> Hashes;
    Hashes.emplace_back(hashType(Operand->getType()));
    Hashes.emplace_back(hashValue(Operand));
    return stable_hash_combine(Hashes);
  }

  stable_hash hashInstruction(const Instruction &Inst) {
    SmallVector<stable_hash> Hashes;
    Hashes.emplace_back(Inst.getOpcode());

    if (!DetailedHash)
      return stable_hash_combine(Hashes);

    Hashes.emplace_back(hashType(Inst.getType()));

    // Handle additional properties of specific instructions that cause
    // semantic differences in the IR.
    if (const auto *ComparisonInstruction = dyn_cast<CmpInst>(&Inst))
      Hashes.emplace_back(ComparisonInstruction->getPredicate());

    unsigned InstIdx = 0;
    if (IndexInstruction) {
      InstIdx = IndexInstruction->size();
      IndexInstruction->try_emplace(InstIdx, const_cast<Instruction *>(&Inst));
    }

    for (const auto [OpndIdx, Op] : enumerate(Inst.operands())) {
      auto OpndHash = hashOperand(Op);
      if (IgnoreOp && IgnoreOp(&Inst, OpndIdx)) {
        assert(IndexOperandHashMap);
        IndexOperandHashMap->try_emplace({InstIdx, OpndIdx}, OpndHash);
      } else
        Hashes.emplace_back(OpndHash);
    }

    return stable_hash_combine(Hashes);
  }

  // A function hash is calculated by considering only the number of arguments
  // and whether a function is varargs, the order of basic blocks (given by the
  // successors of each basic block in depth first order), and the order of
  // opcodes of each instruction within each of these basic blocks. This mirrors
  // the strategy FunctionComparator::compare() uses to compare functions by
  // walking the BBs in depth first order and comparing each instruction in
  // sequence. Because this hash currently does not look at the operands, it is
  // insensitive to things such as the target of calls and the constants used in
  // the function, which makes it useful when possibly merging functions which
  // are the same modulo constants and call targets.
  //
  // Note that different users of StructuralHash will want different behavior
  // out of it (i.e., MergeFunctions will want something different from PM
  // expensive checks for pass modification status). When modifying this
  // function, most changes should be gated behind an option and enabled
  // selectively.
  void update(const Function &F) {
    // Declarations don't affect analyses.
    if (F.isDeclaration())
      return;

    SmallVector<stable_hash> Hashes;
    Hashes.emplace_back(Hash);
    Hashes.emplace_back(FunctionHeaderHash);

    Hashes.emplace_back(F.isVarArg());
    Hashes.emplace_back(F.arg_size());

    SmallVector<const BasicBlock *, 8> BBs;
    SmallPtrSet<const BasicBlock *, 16> VisitedBBs;

    // Walk the blocks in the same order as
    // FunctionComparator::cmpBasicBlocks(), accumulating the hash of the
    // function "structure." (BB and opcode sequence)
    BBs.push_back(&F.getEntryBlock());
    VisitedBBs.insert(BBs[0]);
    while (!BBs.empty()) {
      const BasicBlock *BB = BBs.pop_back_val();

      Hashes.emplace_back(BlockHeaderHash);
      for (auto &Inst : *BB)
        Hashes.emplace_back(hashInstruction(Inst));

      for (const BasicBlock *Succ : successors(BB))
        if (VisitedBBs.insert(Succ).second)
          BBs.push_back(Succ);
    }

    // Update the combined hash in place.
    Hash = stable_hash_combine(Hashes);
  }

  void update(const GlobalVariable &GV) {
    // Declarations and used/compiler.used don't affect analyses.
    // Since there are several `llvm.*` metadata, like `llvm.embedded.object`,
    // we ignore anything with the `.llvm` prefix
    if (GV.isDeclaration() || GV.getName().starts_with("llvm."))
      return;
    SmallVector<stable_hash> Hashes;
    Hashes.emplace_back(Hash);
    Hashes.emplace_back(GlobalHeaderHash);
    Hashes.emplace_back(GV.getValueType()->getTypeID());

    // Update the combined hash in place.
    Hash = stable_hash_combine(Hashes);
  }

  void update(const Module &M) {
    for (const GlobalVariable &GV : M.globals())
      update(GV);
    for (const Function &F : M)
      update(F);
  }

  uint64_t getHash() const { return Hash; }

  std::unique_ptr<IndexInstrMap> getIndexInstrMap() {
    return std::move(IndexInstruction);
  }

  std::unique_ptr<IndexOperandHashMapType> getIndexPairOpndHashMap() {
    return std::move(IndexOperandHashMap);
  }
};

} // namespace

stable_hash llvm::StructuralHash(const Function &F, bool DetailedHash) {
  StructuralHashImpl H(DetailedHash);
  H.update(F);
  return H.getHash();
}

stable_hash llvm::StructuralHash(const GlobalVariable &GVar) {
  return StructuralHashImpl::hashGlobalVariable(GVar);
}

stable_hash llvm::StructuralHash(const Module &M, bool DetailedHash) {
  StructuralHashImpl H(DetailedHash);
  H.update(M);
  return H.getHash();
}

FunctionHashInfo
llvm::StructuralHashWithDifferences(const Function &F,
                                    IgnoreOperandFunc IgnoreOp) {
  StructuralHashImpl H(/*DetailedHash=*/true, IgnoreOp);
  H.update(F);
  return FunctionHashInfo(H.getHash(), H.getIndexInstrMap(),
                          H.getIndexPairOpndHashMap());
}
