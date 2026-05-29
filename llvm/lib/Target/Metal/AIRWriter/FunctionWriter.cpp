//===- FunctionWriter.cpp - AIR function-block writer -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Emits the FUNCTION_BLOCK section of the AIR bitcode for one Function:
// instruction stream, block layout, and the per-function constant table.
// Pointer instructions are emitted with their reconstructed pointee types
// from the enclosing PointeeTypeMap, not the opaque LLVM pointer type.
//
//===----------------------------------------------------------------------===//

#include "BitcodeEncoding.h"
#include "MetadataWriter.h"
#include "MetalConstraints.h"
#include "ValueEnumerator.h"
#include "llvm/Bitcode/LLVMBitCodes.h"
#include "llvm/Bitstream/BitstreamWriter.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

namespace llvm {
namespace metal {

// Forward declaration
void emitConstantsBlock(BitstreamWriter &W, ValueEnumerator &E,
                        ArrayRef<const Constant *> Constants,
                        unsigned CodeSize);

void emitFunctionBlock(BitstreamWriter &W, const Function &F,
                       ValueEnumerator &E, const MetadataEnumerator &MD) {
  W.EnterSubblock(bitc::FUNCTION_BLOCK_ID, 5);

  // Build local value ID map
  DenseMap<const Value *, unsigned> LocalMap;
  unsigned NextID = E.globalValues.size() + E.moduleConstants.size();

  for (auto &Arg : F.args())
    LocalMap[&Arg] = NextID++;

  // Collect function-level constants - include constants even if they're
  // also module constants. Metal v1 bitcode requires function-level
  // constant entries; referencing module constants directly from function
  // instructions causes GPU JIT materializeAll failures.
  SmallVector<const Constant *, 32> FuncConsts;
  for (auto &BB : F)
    for (auto &I : BB)
      for (auto &Op : I.operands())
        if (auto *C = dyn_cast<Constant>(Op))
          if (!isa<GlobalValue>(C) && !LocalMap.count(C) &&
              !E.globalValueMap.count(C)) {
            LocalMap[C] = NextID++;
            FuncConsts.push_back(C);
          }

  // Instruction results
  for (auto &BB : F)
    for (auto &I : BB)
      if (!I.getType()->isVoidTy())
        LocalMap[&I] = NextID++;

  auto GetAbsID = [&](const Value *V) -> unsigned {
    auto It = LocalMap.find(V);
    if (It != LocalMap.end())
      return It->second;
    if (E.globalValueMap.count(V))
      return E.globalIdx(V);
    if (auto *C = dyn_cast<Constant>(V))
      if (E.hasModuleConst(C))
        return E.moduleConstIdx(C);
    return 0;
  };

  // Relative value IDs (current instruction ID minus referenced value ID)
  unsigned CurInstID = E.globalValues.size() + E.moduleConstants.size() +
                       F.arg_size() + FuncConsts.size();
  auto GetID = [&](const Value *V) -> unsigned {
    return CurInstID - GetAbsID(V);
  };

  // DECLAREBLOCKS
  unsigned NumBBs = 0;
  for (auto &BB : F)
    (void)BB, NumBBs++;
  SmallVector<uint64_t, 1> DV = {NumBBs};
  W.EmitRecord(bitc::FUNC_CODE_DECLAREBLOCKS, DV);

  // Function constants
  emitConstantsBlock(W, E, FuncConsts, 5);

  // BB index helper
  SmallVector<const BasicBlock *, 8> BBList;
  for (auto &BB : F)
    BBList.push_back(&BB);
  auto BBIdx = [&](const BasicBlock *BB) -> unsigned {
    for (unsigned I = 0; I < BBList.size(); I++)
      if (BBList[I] == BB)
        return I;
    return 0;
  };

  // Emit instructions. Track (InstrIdx, Inst*) for any inst with attached MD;
  // InstrIdx is 0-based across the emitted stream so the reader (which builds
  // an InstructionList in parallel) can index it via Record[0].
  SmallVector<std::pair<unsigned, const Instruction *>, 8> Attached;
  unsigned EmittedIdx = 0;
  for (auto &BB : F) {
    for (auto &I : BB) {
      SmallVector<uint64_t, 16> V;

      if (auto *BO = dyn_cast<BinaryOperator>(&I)) {
        V.push_back(GetID(BO->getOperand(0)));
        V.push_back(GetID(BO->getOperand(1)));
        V.push_back(encodeBinop(BO->getOpcode()));
        if (BO->getType()->isFPOrFPVectorTy())
          V.push_back(0);
        W.EmitRecord(bitc::FUNC_CODE_INST_BINOP, V);
      } else if (auto *CI = dyn_cast<CastInst>(&I)) {
        V.push_back(GetID(CI->getOperand(0)));
        // For casts producing pointers, use PTM-inferred pointee
        // (Metal v1 needs correct typed pointer per value usage)
        if (CI->getType()->isPointerTy()) {
          V.push_back(E.ptrTypeIdxForValue(CI));
        } else {
          V.push_back(E.typeIdx(CI->getDestTy()));
        }
        V.push_back(encodeCast(CI->getOpcode()));
        W.EmitRecord(bitc::FUNC_CODE_INST_CAST, V);
      } else if (auto *LI = dyn_cast<LoadInst>(&I)) {
        V.push_back(GetID(LI->getPointerOperand()));
        // For loads producing pointer types, use per-value pointee
        // (same rationale as PHI - avoid single-pointee-per-AS mismatch)
        if (LI->getType()->isPointerTy())
          V.push_back(E.ptrTypeIdxForValue(LI));
        else
          V.push_back(E.typeIdx(LI->getType()));
        V.push_back(Log2_32(LI->getAlign().value()) + 1);
        V.push_back(LI->isVolatile() ? 1 : 0);
        W.EmitRecord(bitc::FUNC_CODE_INST_LOAD, V);
      } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
        V.push_back(GetID(SI->getPointerOperand()));
        V.push_back(GetID(SI->getValueOperand()));
        V.push_back(Log2_32(SI->getAlign().value()) + 1);
        V.push_back(SI->isVolatile() ? 1 : 0);
        W.EmitRecord(bitc::FUNC_CODE_INST_STORE, V);
      } else if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
        V.push_back(GEP->isInBounds() ? 1 : 0);
        // Metal GPU JIT requires GEP source type to match pointer's pointee
        // type. For device (AS 1) pointers collapsed to float*, remap i32
        // GEP source type to float (same 4-byte stride) - but ONLY when
        // all terminal users (following GEP chains) consume float. If any
        // terminal user is a non-float load/store/atomic, keep i32.
        Type *GepSrcTy = GEP->getSourceElementType();
        if (GEP->getPointerAddressSpace() == metal::AS::Device &&
            GepSrcTy->isIntegerTy(32)) {
          // Walk GEP chains to find terminal (non-GEP) users
          bool AllTerminalFloat = true;
          SmallVector<const GetElementPtrInst *, 8> Worklist;
          Worklist.push_back(GEP);
          while (!Worklist.empty() && AllTerminalFloat) {
            auto *G = Worklist.pop_back_val();
            for (auto *U : G->users()) {
              if (auto *SubGEP = dyn_cast<GetElementPtrInst>(U)) {
                Worklist.push_back(SubGEP);
              } else if (auto *LI = dyn_cast<LoadInst>(U)) {
                if (!LI->getType()->isFloatTy()) {
                  AllTerminalFloat = false;
                  break;
                }
              } else if (auto *SI = dyn_cast<StoreInst>(U)) {
                if (!SI->getValueOperand()->getType()->isFloatTy()) {
                  AllTerminalFloat = false;
                  break;
                }
              } else {
                AllTerminalFloat = false;
                break;
              }
            }
          }
          if (AllTerminalFloat)
            GepSrcTy = Type::getFloatTy(F.getContext());
        }
        V.push_back(E.typeIdx(GepSrcTy));
        for (auto &Op : GEP->operands())
          V.push_back(GetID(Op));
        W.EmitRecord(bitc::FUNC_CODE_INST_GEP, V);
      } else if (auto *EE = dyn_cast<ExtractElementInst>(&I)) {
        V.push_back(GetID(EE->getVectorOperand()));
        V.push_back(GetID(EE->getIndexOperand()));
        W.EmitRecord(bitc::FUNC_CODE_INST_EXTRACTELT, V);
      } else if (auto *IE = dyn_cast<InsertElementInst>(&I)) {
        V.push_back(GetID(IE->getOperand(0)));
        V.push_back(GetID(IE->getOperand(1)));
        V.push_back(GetID(IE->getOperand(2)));
        W.EmitRecord(bitc::FUNC_CODE_INST_INSERTELT, V);
      } else if (auto *SV = dyn_cast<ShuffleVectorInst>(&I)) {
        V.push_back(GetID(SV->getOperand(0)));
        V.push_back(GetID(SV->getOperand(1)));
        V.push_back(GetID(SV->getShuffleMaskForBitcode()));
        W.EmitRecord(bitc::FUNC_CODE_INST_SHUFFLEVEC, V);
      } else if (auto *IV = dyn_cast<InsertValueInst>(&I)) {
        V.push_back(GetID(IV->getAggregateOperand()));
        V.push_back(GetID(IV->getInsertedValueOperand()));
        for (unsigned Idx : IV->getIndices())
          V.push_back(Idx);
        W.EmitRecord(bitc::FUNC_CODE_INST_INSERTVAL, V);
      } else if (auto *EV = dyn_cast<ExtractValueInst>(&I)) {
        V.push_back(GetID(EV->getAggregateOperand()));
        for (unsigned Idx : EV->getIndices())
          V.push_back(Idx);
        W.EmitRecord(bitc::FUNC_CODE_INST_EXTRACTVAL, V);
      } else if (auto *Sel = dyn_cast<SelectInst>(&I)) {
        V.push_back(GetID(Sel->getTrueValue()));
        V.push_back(GetID(Sel->getFalseValue()));
        V.push_back(GetID(Sel->getCondition()));
        // Use SELECT (code 5) for scalar selects - Metal GPU JIT uses this
        // encoding (3 operands: trueVal, falseVal, cond). VSELECT (code 29)
        // has a different format (5 operands) and causes materializeAll.
        W.EmitRecord(bitc::FUNC_CODE_INST_SELECT, V);
      } else if (auto *Cmp = dyn_cast<CmpInst>(&I)) {
        V.push_back(GetID(Cmp->getOperand(0)));
        V.push_back(GetID(Cmp->getOperand(1)));
        V.push_back(Cmp->getPredicate());
        W.EmitRecord(bitc::FUNC_CODE_INST_CMP2, V);
      } else if (auto *PN = dyn_cast<PHINode>(&I)) {
        // For pointer-typed PHIs, use per-value pointee type from PTM
        // to avoid mismatch when different AS1 params have different
        // pointee types (e.g., half* vs float*). The generic typeIdx()
        // returns a single pointee per address space, which is wrong
        // when the PHI's incoming values have a different pointee.
        if (PN->getType()->isPointerTy())
          V.push_back(E.ptrTypeIdxForValue(PN));
        else
          V.push_back(E.typeIdx(PN->getType()));
        for (unsigned J = 0; J < PN->getNumIncomingValues(); J++) {
          // PHI uses signed relative IDs (back-edge values have higher absID
          // than current, producing negative relative ID = forward reference)
          int64_t RelID =
              (int64_t)CurInstID - (int64_t)GetAbsID(PN->getIncomingValue(J));
          // Signed VBR: positive n → 2n, negative n → (-2n)+1
          uint64_t Encoded = (RelID >= 0) ? ((uint64_t)RelID << 1)
                                          : ((uint64_t)(-RelID) << 1) | 1;
          V.push_back(Encoded);
          V.push_back(BBIdx(PN->getIncomingBlock(J)));
        }
        W.EmitRecord(bitc::FUNC_CODE_INST_PHI, V);
      } else if (auto *UBI = dyn_cast<UncondBrInst>(&I)) {
        V.push_back(BBIdx(UBI->getSuccessor(0)));
        W.EmitRecord(bitc::FUNC_CODE_INST_BR, V);
      } else if (auto *CBI = dyn_cast<CondBrInst>(&I)) {
        V.push_back(BBIdx(CBI->getSuccessor(0)));
        V.push_back(BBIdx(CBI->getSuccessor(1)));
        V.push_back(GetID(CBI->getCondition()));
        W.EmitRecord(bitc::FUNC_CODE_INST_BR, V);
      } else if (auto *RI = dyn_cast<ReturnInst>(&I)) {
        if (RI->getReturnValue())
          V.push_back(GetID(RI->getReturnValue()));
        W.EmitRecord(bitc::FUNC_CODE_INST_RET, V);
      } else if (isa<UnreachableInst>(&I)) {
        W.EmitRecord(bitc::FUNC_CODE_INST_UNREACHABLE, V);
      } else if (auto *CI = dyn_cast<CallInst>(&I)) {
        // CALL: [paramattr, cc_flags, fnty, fnid, ...args, [sentinel]]
        // MMA load calls need paramattr=1 (nocapture+readonly on ptr param)
        bool IsMMALoadCall = false;
        if (auto *Callee = CI->getCalledFunction())
          IsMMALoadCall =
              Callee->getName().starts_with("air.simdgroup_matrix_8x8_load");
        V.push_back(IsMMALoadCall ? 1 : 0);

        // Detect MMA intrinsics that need operand bundle encoding
        bool IsMMAWithBundles = false;
        if (auto *Callee = CI->getCalledFunction()) {
          StringRef Name = Callee->getName();
          IsMMAWithBundles =
              Name.starts_with("air.simdgroup_matrix_8x8_load") ||
              Name.starts_with("air.simdgroup_matrix_8x8_multiply_accumulate");
        }

        uint64_t Flags = 0;
        if (CI->isTailCall())
          Flags |= 1;
        if (CI->getCallingConv() != CallingConv::C)
          Flags |= (uint64_t)CI->getCallingConv() << 1;
        Flags |= (1 << 15); // explicit function type
        if (IsMMAWithBundles)
          Flags |= (1 << 17); // operand bundles
        V.push_back(Flags);

        if (IsMMAWithBundles) {
          // MMA bundle encoding: sentinel 254, then function type, then callee
          V.push_back(254);
          V.push_back(E.typeIdx(CI->getFunctionType()));
        } else {
          V.push_back(E.typeIdx(CI->getFunctionType()));
        }

        V.push_back(GetID(CI->getCalledOperand()));
        for (unsigned J = 0; J < CI->arg_size(); J++)
          V.push_back(GetID(CI->getArgOperand(J)));
        W.EmitRecord(bitc::FUNC_CODE_INST_CALL, V);
      } else if (auto *AI = dyn_cast<AllocaInst>(&I)) {
        // For event storage allocas (alloca ptr addrspace(3)), the
        // allocated type must be event_t*3, not the default float*3.
        Type *AllocTy = AI->getAllocatedType();
        if (AllocTy->isPointerTy() && AllocTy->getPointerAddressSpace() == 3) {
          if (auto *EvTy =
                  StructType::getTypeByName(AI->getContext(), "event_t"))
            V.push_back(
                E.ptrTypeIdx(PointerType::get(AI->getContext(), 3), EvTy));
          else
            V.push_back(E.typeIdx(AllocTy));
        } else {
          V.push_back(E.typeIdx(AllocTy));
        }
        V.push_back(E.typeIdx(AI->getArraySize()->getType()));
        V.push_back(GetAbsID(AI->getArraySize()));
        V.push_back((1 << 6) | (Log2_32(AI->getAlign().value()) + 1));
        W.EmitRecord(bitc::FUNC_CODE_INST_ALLOCA, V);
      }

      if (I.hasMetadataOtherThanDebugLoc())
        Attached.push_back({EmittedIdx, &I});
      EmittedIdx++;

      if (!I.getType()->isVoidTy())
        CurInstID++;
    }
  }

  // E2b: per-instruction metadata attachments (alias.scope, noalias, tbaa).
  if (!Attached.empty()) {
    W.EnterSubblock(bitc::METADATA_ATTACHMENT_ID, 3);
    SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
    SmallVector<uint64_t, 16> Record;
    for (auto &P : Attached) {
      MDs.clear();
      P.second->getAllMetadataOtherThanDebugLoc(MDs);
      Record.clear();
      Record.push_back(P.first);
      for (auto &Att : MDs) {
        Record.push_back(Att.first);
        Record.push_back(MD.globalMDID(Att.second));
      }
      W.EmitRecord(bitc::METADATA_ATTACHMENT, Record);
    }
    W.ExitBlock();
  }

  W.ExitBlock();
}

} // namespace metal
} // namespace llvm
