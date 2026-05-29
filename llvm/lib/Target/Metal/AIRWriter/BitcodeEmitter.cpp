//===- BitcodeEmitter.cpp - Metal v1 bitcode emitter ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Top-level orchestrator for Metal v1 bitcode emission. Delegates to
// ValueEnumerator, TypeTableWriter, ConstantsWriter, MetadataWriter,
// and FunctionWriter.
//
//===----------------------------------------------------------------------===//

#include "BitcodeEmitter.h"
#include "BitcodeEncoding.h"
#include "MetadataWriter.h"
#include "MetalConstraints.h"
#include "ValueEnumerator.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/Bitcode/LLVMBitCodes.h"
#include "llvm/Bitstream/BitstreamWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace llvm {
namespace metal {

// Lower all ConstantExpr operands in instructions to real instructions.
// Metal's GPU JIT doesn't handle constant expression records in bitcode,
// so they must be materialized as instructions before serialization.
//
// For byte-stride GEPs on threadgroup float globals (e.g.,
// gep i8, @tg_global, i64 byte_offset where @tg_global is [N x float]),
// converts to float-element GEPs (gep float, @base, i64 float_index)
// because Metal v1 typed-pointer bitcode requires GEP source type to match
// the pointer's pointee type.
static void lowerConstantExprs(Module &M) {
  auto &Ctx = M.getContext();
  Type *FloatTy = Type::getFloatTy(Ctx);
  Type *I64Ty = Type::getInt64Ty(Ctx);
  unsigned FloatSize = M.getDataLayout().getTypeAllocSize(FloatTy);

  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    SmallVector<std::pair<Instruction *, unsigned>, 32> Worklist;
    bool Changed = true;
    while (Changed) {
      Changed = false;
      Worklist.clear();
      for (auto &BB : F)
        for (auto &I : BB)
          for (unsigned J = 0; J < I.getNumOperands(); J++)
            if (isa<ConstantExpr>(I.getOperand(J)))
              Worklist.push_back({&I, J});
      for (auto &[I, OpIdx] : Worklist) {
        auto *CE = cast<ConstantExpr>(I->getOperand(OpIdx));
        Instruction *NewI = CE->getAsInstruction();

        // Convert byte-stride GEPs on TG float globals to float-element GEPs.
        if (auto *GEP = dyn_cast<GetElementPtrInst>(NewI)) {
          if (GEP->getSourceElementType()->isIntegerTy(8) &&
              GEP->getPointerAddressSpace() == metal::AS::Threadgroup &&
              GEP->getNumIndices() == 1) {
            // Check if base is a TG global with float array element type
            Value *Base = GEP->getPointerOperand();
            GlobalVariable *GV = dyn_cast<GlobalVariable>(Base);
            if (GV) {
              Type *ElemTy = nullptr;
              if (auto *AT = dyn_cast<ArrayType>(GV->getValueType()))
                ElemTy = AT->getElementType();
              if (ElemTy && ElemTy->isFloatTy()) {
                Value *ByteIdx = GEP->idx_begin()->get();
                if (auto *CI = dyn_cast<ConstantInt>(ByteIdx)) {
                  uint64_t ByteOff = CI->getZExtValue();
                  if (ByteOff % FloatSize == 0) {
                    // Find or create a base float* from the GV.
                    // Look for an existing gep [N x float], @GV, 0, 0 in the
                    // function's entry block.
                    Value *FloatBase = nullptr;
                    for (auto *U : GV->users()) {
                      auto *BaseGEP = dyn_cast<GetElementPtrInst>(U);
                      if (!BaseGEP || !BaseGEP->getParent())
                        continue;
                      if (BaseGEP->getFunction() == &F &&
                          BaseGEP->getSourceElementType() ==
                              GV->getValueType() &&
                          BaseGEP->getNumIndices() == 2) {
                        FloatBase = BaseGEP;
                        break;
                      }
                    }
                    if (!FloatBase) {
                      // Create base GEP: gep [N x float], @GV, 0, 0
                      auto *NewBaseGEP = GetElementPtrInst::CreateInBounds(
                          GV->getValueType(), GV,
                          {ConstantInt::get(I64Ty, 0),
                           ConstantInt::get(I64Ty, 0)});
                      NewBaseGEP->insertBefore(
                          F.getEntryBlock().getFirstInsertionPt());
                      FloatBase = NewBaseGEP;
                    }
                    // Create: gep float, %base, i64 (byteOff/4)
                    auto *FloatGEP = GetElementPtrInst::CreateInBounds(
                        ElemTy, FloatBase,
                        {ConstantInt::get(I64Ty, ByteOff / FloatSize)});
                    FloatGEP->insertBefore(I->getIterator());
                    I->setOperand(OpIdx, FloatGEP);
                    NewI->deleteValue(); // discard the byte GEP
                    Changed = true;
                    continue;
                  }
                }
              }
            }
          }
        }

        NewI->insertBefore(I->getIterator());
        I->setOperand(OpIdx, NewI);
        Changed = true;
      }
    }

    // Identity ptr-to-ptr bitcasts (same opaque type, different typed pointer
    // semantics) are kept - the FunctionWriter handles them by emitting a
    // bitcast to the correct typed pointer type inferred from PTM/usage.
  }
}

// Fix GEP source type / pointer pointee mismatches for Metal typed bitcode.
//
// Metal v1 typed-pointer bitcode requires the GEP source element type to match
// the pointer's pointee type. When a GEP uses a different element type than the
// pointer (e.g., `gep half, float*3 %ptr`), the GPU JIT rejects it.
//
// For TG (AS3) pointers where the buffer is float-typed (from MMA merge) but
// accessed with half/i8 GEPs (from pipelined loads), we insert identity
// bitcasts before the GEP. The bitcast creates a new pointer value that the
// PTM can type as half* instead of float*, making the GEP consistent:
// %bc = bitcast float*3 %ptr to float*3 (identity in opaque-ptr IR)
// %p = gep half, float*3 %bc, i32 %idx
// Then PTM sets %bc → half, so typed bitcode sees: gep half, half*3 %bc, idx
//
// For device (AS1) pointers with i8 GEPs (from async copy byte offsets),
// convert to float-stride GEPs since all device pointers are float*.
static void fixGEPTypeMismatches(Module &M, PointeeTypeMap &PTM) {
  bool HasMMA = false;
  for (auto &F : M)
    if (F.getName().starts_with("air.simdgroup_matrix_8x8_"))
      HasMMA = true;
  if (!HasMMA)
    return;

  auto &Ctx = M.getContext();
  Type *FloatTy = Type::getFloatTy(Ctx);

  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    SmallVector<GetElementPtrInst *, 8> ToFix;
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
          Type *SrcTy = GEP->getSourceElementType();
          if (SrcTy == FloatTy)
            continue;
          if (GEP->getNumIndices() != 1)
            continue;
          if (!SrcTy->isIntegerTy() && !SrcTy->isHalfTy() &&
              !SrcTy->isBFloatTy())
            continue;
          unsigned AS = GEP->getPointerAddressSpace();
          if (AS != metal::AS::Device && AS != metal::AS::Threadgroup)
            continue;
          ToFix.push_back(GEP);
        }

    for (auto *GEP : ToFix) {
      Type *SrcTy = GEP->getSourceElementType();
      unsigned SrcSize = SrcTy->getPrimitiveSizeInBits();
      Value *Ptr = GEP->getPointerOperand();

      // Same-size types (e.g., i32 vs float, both 4 bytes):
      // Just change the GEP source element type to float. The stride is
      // identical so the pointer arithmetic doesn't change.
      if (SrcSize == 32) {
        GEP->setSourceElementType(FloatTy);
        GEP->setResultElementType(FloatTy);
        continue;
      }

      // Different-size types (half=16bit, i8=8bit vs float=32bit):
      // Insert identity bitcast before GEP to create a new pointer value
      // with the correct PTM entry. The bitcast is a no-op in opaque-ptr IR
      // but gives the serializer a different typed pointer for the GEP source.
      auto *BC = CastInst::Create(Instruction::BitCast, Ptr, Ptr->getType(), "",
                                  GEP->getIterator());
      GEP->setOperand(0, BC);
      PTM.set(BC, SrcTy);
    }
  }
}

// Fix air.arg_type_name / air.arg_type_size in kernel metadata to match
// actual parameter pointee types from PTM. The transform pipeline may set
// all buffer type names to "float" even when the actual type is bfloat/char.
// Metal GPU JIT validates these metadata entries against the bitcode types.
static void fixKernelArgMetadata(Module &M, const PointeeTypeMap &PTM) {
  auto &Ctx = M.getContext();
  auto *AirKernel = M.getNamedMetadata("air.kernel");
  if (!AirKernel)
    return;

  for (unsigned K = 0; K < AirKernel->getNumOperands(); K++) {
    auto *KernelMD = AirKernel->getOperand(K);
    if (KernelMD->getNumOperands() < 3)
      continue;
    // KernelMD: {fn, attrs, argDescs}
    auto *ArgDescs = dyn_cast_or_null<MDNode>(KernelMD->getOperand(2));
    if (!ArgDescs)
      continue;

    auto *FnVAM = dyn_cast_or_null<ValueAsMetadata>(KernelMD->getOperand(0));
    if (!FnVAM)
      continue;
    auto *Fn = dyn_cast<Function>(FnVAM->getValue());
    if (!Fn)
      continue;

    for (unsigned A = 0; A < ArgDescs->getNumOperands(); A++) {
      auto *ArgMD = dyn_cast_or_null<MDNode>(ArgDescs->getOperand(A));
      if (!ArgMD || ArgMD->getNumOperands() < 2)
        continue;

      // Check if this is a buffer arg (has "air.buffer" string)
      bool IsBuffer = false;
      for (unsigned I = 1; I < ArgMD->getNumOperands(); I++)
        if (auto *S = dyn_cast_or_null<MDString>(ArgMD->getOperand(I)))
          if (S->getString() == "air.buffer") {
            IsBuffer = true;
            break;
          }
      if (!IsBuffer)
        continue;

      // Get the arg index from the first operand
      auto *IdxVAM = dyn_cast_or_null<ValueAsMetadata>(ArgMD->getOperand(0));
      if (!IdxVAM)
        continue;
      auto *IdxCI = dyn_cast<ConstantInt>(IdxVAM->getValue());
      if (!IdxCI)
        continue;
      unsigned ArgIdx = IdxCI->getZExtValue();
      if (ArgIdx >= Fn->arg_size())
        continue;

      // Infer pointee type from PTM, following through bitcasts
      Argument *Arg = Fn->getArg(ArgIdx);
      if (!Arg->getType()->isPointerTy())
        continue;
      Type *Pointee = nullptr;
      if (auto *Ty = PTM.get(Arg))
        Pointee = Ty;
      if (!Pointee)
        Pointee = PointeeTypeMap::inferFromUsage(Arg);
      // Follow through bitcasts if inference failed on the arg directly
      if (!Pointee || Pointee->isFloatTy()) {
        for (auto *U : Arg->users()) {
          if (auto *BC = dyn_cast<BitCastInst>(U)) {
            if (auto *Ty = PTM.get(BC)) {
              if (!Ty->isFloatTy()) {
                Pointee = Ty;
                break;
              }
            }
            Type *BcTy = PointeeTypeMap::inferFromUsage(BC);
            if (BcTy && !BcTy->isFloatTy()) {
              Pointee = BcTy;
              break;
            }
          }
        }
      }
      if (!Pointee)
        continue;

      // Determine correct type name, size, and alignment
      StringRef TypeName;
      unsigned TypeSize = 0, TypeAlign = 0;
      if (Pointee->isBFloatTy()) {
        TypeName = "bfloat";
        TypeSize = 2;
        TypeAlign = 2;
      } else if (Pointee->isFloatTy()) {
        TypeName = "float";
        TypeSize = 4;
        TypeAlign = 4;
      } else if (Pointee->isHalfTy()) {
        TypeName = "half";
        TypeSize = 2;
        TypeAlign = 2;
      } else if (Pointee->isIntegerTy(8)) {
        TypeName = "char";
        TypeSize = 1;
        TypeAlign = 1;
      } else if (Pointee->isIntegerTy(16)) {
        TypeName = "short";
        TypeSize = 2;
        TypeAlign = 2;
      } else if (Pointee->isIntegerTy(32)) {
        TypeName = "int";
        TypeSize = 4;
        TypeAlign = 4;
      } else {
        continue; // Unknown type, don't change
      }

      // Rebuild the metadata node with corrected values
      SmallVector<Metadata *, 16> NewOps;
      for (unsigned I = 0; I < ArgMD->getNumOperands(); I++) {
        Metadata *Op = ArgMD->getOperand(I);
        if (I + 1 < ArgMD->getNumOperands()) {
          if (auto *PrevS = dyn_cast_or_null<MDString>(ArgMD->getOperand(I))) {
            if (PrevS->getString() == "air.arg_type_name" &&
                I + 1 < ArgMD->getNumOperands()) {
              NewOps.push_back(Op);
              NewOps.push_back(MDString::get(Ctx, TypeName));
              I++; // skip original type name
              continue;
            }
            if (PrevS->getString() == "air.arg_type_size" &&
                I + 1 < ArgMD->getNumOperands()) {
              NewOps.push_back(Op);
              NewOps.push_back(ValueAsMetadata::get(
                  ConstantInt::get(Type::getInt32Ty(Ctx), TypeSize)));
              I++; // skip original size
              continue;
            }
            if (PrevS->getString() == "air.arg_type_align_size" &&
                I + 1 < ArgMD->getNumOperands()) {
              NewOps.push_back(Op);
              NewOps.push_back(ValueAsMetadata::get(
                  ConstantInt::get(Type::getInt32Ty(Ctx), TypeAlign)));
              I++; // skip original align
              continue;
            }
          }
        }
        NewOps.push_back(Op);
      }
      auto *NewArgMD = MDNode::get(Ctx, NewOps);
      ArgDescs->replaceOperandWith(A, NewArgMD);
    }
  }
}

// Forward declarations (defined in separate .cpp files)
void emitTypeBlock(BitstreamWriter &W, ValueEnumerator &E);
void emitConstantsBlock(BitstreamWriter &W, ValueEnumerator &E,
                        ArrayRef<const Constant *> Constants,
                        unsigned CodeSize);
void emitMetadataKindBlock(BitstreamWriter &W);
void emitMetadataBlock(BitstreamWriter &W, Module &M, ValueEnumerator &E,
                       MetadataEnumerator &MD);
void emitOperandBundleTagsBlock(BitstreamWriter &W);
void emitSinglethreadBlock(BitstreamWriter &W);
void emitFunctionBlock(BitstreamWriter &W, const Function &F,
                       ValueEnumerator &E, const MetadataEnumerator &MD);

// Remove truly redundant ptr-to-ptr bitcasts where the PTM has the SAME
// pointee type on both sides. Bitcasts that serve as typed pointer
// transitions (where PTM records different types) must be kept.
// Only removes bitcasts where BOTH sides have the same PTM entry.
static void removeRedundantBitcasts(Module &M, PointeeTypeMap &PTM) {
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    SmallVector<BitCastInst *, 16> ToRemove;
    for (auto &BB : F) {
      for (auto &I : BB) {
        auto *BC = dyn_cast<BitCastInst>(&I);
        if (!BC || BC->getSrcTy() != BC->getDestTy())
          continue;
        // Only remove if BOTH sides have the same PTM-recorded type.
        // If either side has no PTM entry, keep the bitcast (it may serve
        // as a type transition that the serializer needs).
        Type *SrcPT = PTM.get(BC->getOperand(0));
        Type *DstPT = PTM.get(BC);
        if (!SrcPT || !DstPT)
          continue; // Unknown - keep
        if (SrcPT != DstPT)
          continue; // Different - keep
        ToRemove.push_back(BC);
      }
    }
    for (auto *BC : ToRemove) {
      PTM.remove(BC);
      BC->replaceAllUsesWith(BC->getOperand(0));
      BC->eraseFromParent();
    }
  }
}

std::vector<uint8_t> emitMetalBitcode(Module &M, PointeeTypeMap &PTM) {
  SmallVector<char, 0> Buf;
  BitstreamWriter W(Buf);

  // BC magic
  W.Emit('B', 8);
  W.Emit('C', 8);
  W.Emit(0xC0, 8);
  W.Emit(0xDE, 8);

  // IDENTIFICATION
  W.EnterSubblock(bitc::IDENTIFICATION_BLOCK_ID, 5);
  emitString(W, bitc::IDENTIFICATION_CODE_STRING, "MetalIR");
  {
    SmallVector<uint64_t, 1> V = {0};
    W.EmitRecord(bitc::IDENTIFICATION_CODE_EPOCH, V);
  }
  W.ExitBlock();

  // Pre-serialization IR fixups (these helpers refine the PTM in place).
  removeRedundantBitcasts(M, PTM);
  fixGEPTypeMismatches(M, PTM);

  // Lower ConstantExpr operands to real instructions before enumeration.
  lowerConstantExprs(M);

  // Fix kernel argument metadata to match actual pointee types.
  fixKernelArgMetadata(M, PTM);

  ValueEnumerator E(M, PTM);

  // MODULE_BLOCK (CodeSize=4)
  W.EnterSubblock(bitc::MODULE_BLOCK_ID, 4);

  {
    SmallVector<uint64_t, 1> V = {1};
    W.EmitRecord(bitc::MODULE_CODE_VERSION, V);
  }

  // PARAMATTR blocks BEFORE TYPE_BLOCK (Metal requires this order).
  //
  // E2c — general path: walk every Function, collect its parameter
  // AttributeSets and emit one PARAMATTR_GRP per unique (paramIdx, AS) tuple
  // plus one PARAMATTR list per function. Group ID 1 is reserved for the
  // legacy MMA-load nocapture+readonly entry, which lives on the call site
  // (the CallInst's first paramattr operand), not the function declaration.
  struct GroupKey {
    unsigned ListIdx;
    AttributeSet AS;
    bool operator==(const GroupKey &O) const {
      return ListIdx == O.ListIdx && AS == O.AS;
    }
  };
  struct GroupKeyInfo {
    static GroupKey getEmptyKey() {
      return {~0u, DenseMapInfo<AttributeSet>::getEmptyKey()};
    }
    static GroupKey getTombstoneKey() {
      return {~0u - 1, DenseMapInfo<AttributeSet>::getTombstoneKey()};
    }
    static unsigned getHashValue(const GroupKey &K) {
      return hash_combine(K.ListIdx,
                          DenseMapInfo<AttributeSet>::getHashValue(K.AS));
    }
    static bool isEqual(const GroupKey &A, const GroupKey &B) {
      return A.ListIdx == B.ListIdx && A.AS == B.AS;
    }
  };

  DenseMap<GroupKey, unsigned, GroupKeyInfo> GroupID;
  SmallVector<GroupKey, 8> GroupOrder;
  auto getGroupID = [&](unsigned ListIdx, AttributeSet AS) -> unsigned {
    GroupKey K{ListIdx, AS};
    auto It = GroupID.find(K);
    if (It != GroupID.end())
      return It->second;
    unsigned ID = GroupID.size() + 1;
    GroupID[K] = ID;
    GroupOrder.push_back(K);
    return ID;
  };

  DenseMap<const Function *, unsigned> FnAttrListID;
  SmallVector<SmallVector<unsigned, 4>, 8> AttrLists;

  bool HasMMALoad = false;
  for (auto &F : M) {
    if (F.getName().starts_with("air.simdgroup_matrix_8x8_load")) {
      HasMMALoad = true;
      break;
    }
  }
  if (HasMMALoad) {
    // Reserve group ID 1 + list ID 1 for the MMA call-site paramattr (the
    // FunctionWriter unconditionally threads `paramattr=1` into those calls).
    GroupKey K{/*ListIdx=*/1, AttributeSet()};
    GroupID[K] = 1;
    GroupOrder.push_back(K);
  }

  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    AttributeList AL = F.getAttributes();
    SmallVector<unsigned, 4> GroupIDs;
    for (unsigned i = 0; i < F.arg_size(); i++) {
      AttributeSet AS = AL.getParamAttrs(i);
      if (!AS.hasAttributes())
        continue;
      GroupIDs.push_back(getGroupID(i + 1, AS));
    }
    AttributeSet RetAS = AL.getRetAttrs();
    if (RetAS.hasAttributes())
      GroupIDs.push_back(getGroupID(0, RetAS));
    if (GroupIDs.empty())
      continue;
    unsigned ListID = AttrLists.size() + 1;
    AttrLists.push_back(std::move(GroupIDs));
    FnAttrListID[&F] = ListID;
  }

  if (!GroupID.empty()) {
    W.EnterSubblock(bitc::PARAMATTR_GROUP_BLOCK_ID, 4);
    for (auto &K : GroupOrder) {
      unsigned ID = GroupID.lookup(K);
      SmallVector<uint64_t, 16> Grp;
      Grp.push_back(ID);
      Grp.push_back(K.ListIdx);
      if (HasMMALoad && ID == 1 && K.AS.getNumAttributes() == 0) {
        // Legacy MMA call-site group: param 1 nocapture + readonly.
        Grp.push_back(0);
        Grp.push_back(11);
        Grp.push_back(0);
        Grp.push_back(21);
      } else {
        for (Attribute Attr : K.AS) {
          if (Attr.isStringAttribute()) {
            StringRef Key = Attr.getKindAsString();
            StringRef Val = Attr.getValueAsString();
            // 3 = string-key-only, 4 = key + value (matches upstream
            // writeAttributeGroupTable encoding).
            Grp.push_back(Val.empty() ? 3 : 4);
            for (char C : Key)
              Grp.push_back((unsigned char)C);
            Grp.push_back(0);
            if (!Val.empty()) {
              for (char C : Val)
                Grp.push_back((unsigned char)C);
              Grp.push_back(0);
            }
          } else if (Attr.isEnumAttribute()) {
            // For the small set we currently emit (NoAlias / NoCapture /
            // ReadOnly), the LLVM AttrKind enum value matches the bitcode
            // attr-kind encoding for v1, same assumption the legacy
            // hardcoded path made.
            Grp.push_back(0);
            Grp.push_back((uint64_t)Attr.getKindAsEnum());
          } else if (Attr.isIntAttribute()) {
            Grp.push_back(1);
            Grp.push_back((uint64_t)Attr.getKindAsEnum());
            Grp.push_back(Attr.getValueAsInt());
          }
        }
      }
      W.EmitRecord(bitc::PARAMATTR_GRP_CODE_ENTRY, Grp);
    }
    W.ExitBlock();

    W.EnterSubblock(bitc::PARAMATTR_BLOCK_ID, 4);
    if (HasMMALoad) {
      SmallVector<uint64_t, 2> List;
      List.push_back(1);
      W.EmitRecord(2, List);
    }
    for (auto &AL : AttrLists) {
      SmallVector<uint64_t, 8> List;
      for (unsigned ID : AL)
        List.push_back(ID);
      W.EmitRecord(2, List);
    }
    W.ExitBlock();
  }

  emitTypeBlock(W, E);

  // Emit target triple - Metal GPU JIT expects it for proper codegen.
  // Use module value if set, otherwise default Metal AIR triple.
  {
    std::string T = M.getTargetTriple().str();
    if (T.empty() || T == "air")
      T = "air64_v28-apple-macosx26.0.0";
    emitString(W, bitc::MODULE_CODE_TRIPLE, T);
  }
  // Emit data layout - Metal GPU JIT uses this for type size/alignment.
  {
    auto DLStr = M.getDataLayoutStr();
    if (!DLStr.empty()) {
      emitString(W, bitc::MODULE_CODE_DATALAYOUT, DLStr);
    } else {
      emitString(W, bitc::MODULE_CODE_DATALAYOUT,
                 "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64"
                 "-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32"
                 "-v48:64:64-v64:64:64-v96:128:128-v128:128:128"
                 "-v192:256:256-v256:256:256-v512:512:512"
                 "-v1024:1024:1024-n8:16:32");
    }
  }

  if (!M.getSourceFileName().empty())
    emitString(W, bitc::MODULE_CODE_SOURCE_FILENAME, M.getSourceFileName());

  // GLOBALVAR and FUNCTION records - emit in globalValues order
  // (globals first, then functions, matching value ID assignment)
  for (auto *V : E.globalValues) {
    if (auto *G = dyn_cast<GlobalVariable>(V)) {
      SmallVector<uint64_t, 14> Ops;
      Ops.push_back(E.globalPtrTypeIdx(G)); // ptr-to-valueType
      Ops.push_back(G->isConstant() ? 1 : 0);
      Ops.push_back(
          G->hasInitializer() ? E.moduleConstIdx(G->getInitializer()) + 1 : 0);
      Ops.push_back(encodeLinkage(G->getLinkage()));
      Ops.push_back(G->getAlign() ? Log2_32(G->getAlign()->value()) + 1 : 0);
      for (int J = 0; J < 3; J++)
        Ops.push_back(0);
      Ops.push_back(G->hasGlobalUnnamedAddr() ? 1 : 0);
      Ops.push_back(G->isExternallyInitialized() ? 1 : 0);
      Ops.push_back(0);
      Ops.push_back(0);
      Ops.push_back(G->getAddressSpace());
      Ops.push_back(0);
      W.EmitRecord(bitc::MODULE_CODE_GLOBALVAR, Ops);
    } else if (auto *Fn = dyn_cast<Function>(V)) {
      SmallVector<uint64_t, 17> Ops;
      Ops.push_back(E.typeIdx(Fn->getFunctionType()));
      Ops.push_back(Fn->getCallingConv());
      Ops.push_back(Fn->isDeclaration() ? 1 : 0);
      Ops.push_back(encodeLinkage(Fn->getLinkage()));
      // paramattr: attribute-list ID for this function (0 = none).
      // Per-function list IDs start at 2 when HasMMALoad reserves list 1
      // for the MMA call site, else at 1.
      unsigned ListID = 0;
      auto It = FnAttrListID.find(Fn);
      if (It != FnAttrListID.end())
        ListID = It->second + (HasMMALoad ? 1 : 0);
      Ops.push_back(ListID);
      Ops.push_back(0); // align
      for (int J = 0; J < 10; J++)
        Ops.push_back(0);
      Ops.push_back(Fn->getAddressSpace());
      W.EmitRecord(bitc::MODULE_CODE_FUNCTION, Ops);
    }
  }

  emitConstantsBlock(W, E, E.moduleConstants, 5);
  emitMetadataKindBlock(W);

  // Share one MetadataEnumerator between the module-level METADATA_BLOCK
  // (where the nodes are emitted) and per-function attachment blocks
  // (where they are referenced by ID).
  MetadataEnumerator MDEnum;
  MDEnum.collect(M, E);
  emitMetadataBlock(W, M, E, MDEnum);
  emitOperandBundleTagsBlock(W);
  emitSinglethreadBlock(W);

  for (auto *V : E.globalValues)
    if (auto *F = dyn_cast<Function>(V))
      if (!F->isDeclaration())
        emitFunctionBlock(W, *F, E, MDEnum);

  // VALUE_SYMTAB
  W.EnterSubblock(bitc::VALUE_SYMTAB_BLOCK_ID, 4);
  for (unsigned I = 0; I < E.globalValues.size(); I++) {
    if (!E.globalValues[I]->hasName())
      continue;
    SmallVector<uint64_t, 32> NV;
    NV.push_back(I);
    for (char C : E.globalValues[I]->getName())
      NV.push_back((uint64_t)(unsigned char)C);
    W.EmitRecord(bitc::VST_CODE_ENTRY, NV);
  }
  W.ExitBlock();

  W.ExitBlock(); // MODULE_BLOCK

  return std::vector<uint8_t>(Buf.begin(), Buf.end());
}

} // namespace metal
} // namespace llvm
