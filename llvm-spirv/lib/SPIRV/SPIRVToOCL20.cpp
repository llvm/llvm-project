//===- SPIRVToOCL20.cpp - Transform SPIR-V builtins to OCL20 builtins------===//
//
//                     The LLVM/SPIRV Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (c) 2014 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimers in the documentation
// and/or other materials provided with the distribution.
// Neither the names of Advanced Micro Devices, Inc., nor the names of its
// contributors may be used to endorse or promote products derived from this
// Software without specific prior written permission.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.
//
//===----------------------------------------------------------------------===//
//
// This file implements transform SPIR-V builtins to OCL 2.0 builtins.
//
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "spvtocl20"

#include "OCLUtil.h"
#include "SPIRVInternal.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/PassSupport.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#include <cstring>

using namespace llvm;
using namespace SPIRV;
using namespace OCLUtil;

namespace SPIRV {

static cl::opt<std::string>
    MangledAtomicTypeNamePrefix("spirv-atomic-prefix",
                                cl::desc("Mangled atomic type name prefix"),
                                cl::init("U7_Atomic"));

class SPIRVToOCL20 : public ModulePass, public InstVisitor<SPIRVToOCL20> {
public:
  SPIRVToOCL20() : ModulePass(ID), M(nullptr), Ctx(nullptr) {
    initializeSPIRVToOCL20Pass(*PassRegistry::getPassRegistry());
  }
  bool runOnModule(Module &M) override;

  void visitCallInst(CallInst &CI);

  // SPIR-V reader should translate vector casts into OCL built-ins because
  // such conversions are not defined neither by OpenCL C/C++ nor
  // by SPIR 1.2/2.0 standards. So, it is safer to convert such casts into
  // appropriate calls to conversion built-ins defined by the standards.
  void visitCastInst(CastInst &CI);

  /// Transform __spirv_ImageQuerySize[Lod] into vector of the same lenght
  /// containing {[get_image_width | get_image_dim], get_image_array_size}
  /// for all images except image1d_t which is always converted into
  /// get_image_width returning scalar result.
  void visitCallSPRIVImageQuerySize(CallInst *CI);

  /// Transform __spirv_Atomic* to atomic_*.
  ///   __spirv_Atomic*(atomic_op, scope, sema, ops, ...) =>
  ///      atomic_*(atomic_op, ops, ..., order(sema), map(scope))
  void visitCallSPIRVAtomicBuiltin(CallInst *CI, Op OC);

  /// Transform __spirv_Group* to {work_group|sub_group}_*.
  ///
  /// Special handling of work_group_broadcast.
  ///   __spirv_GroupBroadcast(a, vec3(x, y, z))
  ///     =>
  ///   work_group_broadcast(a, x, y, z)
  ///
  /// Transform OpenCL group builtin function names from group_
  /// to workgroup_ and sub_group_.
  /// Insert group operation part: reduce_/inclusive_scan_/exclusive_scan_
  /// Transform the operation part:
  ///    fadd/iadd/sadd => add
  ///    fmax/smax => max
  ///    fmin/smin => min
  /// Keep umax/umin unchanged.
  void visitCallSPIRVGroupBuiltin(CallInst *CI, Op OC);

  /// Transform __spirv_MemoryBarrier to atomic_work_item_fence.
  ///   __spirv_MemoryBarrier(scope, sema) =>
  ///       atomic_work_item_fence(flag(sema), order(sema), map(scope))
  void visitCallSPIRVMemoryBarrier(CallInst *CI);

  /// Transform __spirv_{PipeOpName} to OCL pipe builtin functions.
  void visitCallSPIRVPipeBuiltin(CallInst *CI, Op OC);

  /// Transform __spirv_* builtins to OCL 2.0 builtins.
  /// No change with arguments.
  void visitCallSPIRVBuiltin(CallInst *CI, Op OC);

  /// Translate mangled atomic type name: "atomic_" =>
  ///   MangledAtomicTypeNamePrefix
  void translateMangledAtomicTypeName();

  /// Get prefix work_/sub_ for OCL group builtin functions.
  /// Assuming the first argument of \param CI is a constant integer for
  /// workgroup/subgroup scope enums.
  std::string getGroupBuiltinPrefix(CallInst *CI);

  static char ID;

private:
  Module *M;
  LLVMContext *Ctx;
};

char SPIRVToOCL20::ID = 0;

bool SPIRVToOCL20::runOnModule(Module &Module) {
  M = &Module;
  Ctx = &M->getContext();
  visit(*M);

  translateMangledAtomicTypeName();

  eraseUselessFunctions(&Module);

  LLVM_DEBUG(dbgs() << "After SPIRVToOCL20:\n" << *M);

  std::string Err;
  raw_string_ostream ErrorOS(Err);
  if (verifyModule(*M, &ErrorOS)) {
    LLVM_DEBUG(errs() << "Fails to verify module: " << ErrorOS.str());
  }
  return true;
}

void SPIRVToOCL20::visitCallInst(CallInst &CI) {
  LLVM_DEBUG(dbgs() << "[visistCallInst] " << CI << '\n');
  auto F = CI.getCalledFunction();
  if (!F)
    return;

  auto MangledName = F->getName();
  std::string DemangledName;
  Op OC = OpNop;
  if (!oclIsBuiltin(MangledName, &DemangledName) ||
      (OC = getSPIRVFuncOC(DemangledName)) == OpNop)
    return;
  LLVM_DEBUG(dbgs() << "DemangledName = " << DemangledName.c_str() << '\n'
                    << "OpCode = " << OC << '\n');

  if (OC == OpImageQuerySize || OC == OpImageQuerySizeLod) {
    visitCallSPRIVImageQuerySize(&CI);
    return;
  }
  if (OC == OpMemoryBarrier) {
    visitCallSPIRVMemoryBarrier(&CI);
    return;
  }
  if (isAtomicOpCode(OC)) {
    visitCallSPIRVAtomicBuiltin(&CI, OC);
    return;
  }
  if (isGroupOpCode(OC)) {
    visitCallSPIRVGroupBuiltin(&CI, OC);
    return;
  }
  if (isPipeOpCode(OC)) {
    visitCallSPIRVPipeBuiltin(&CI, OC);
    return;
  }
  if (OCLSPIRVBuiltinMap::rfind(OC))
    visitCallSPIRVBuiltin(&CI, OC);
}

void SPIRVToOCL20::visitCallSPIRVMemoryBarrier(CallInst *CI) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  mutateCallInstOCL(M, CI,
                    [=](CallInst *, std::vector<Value *> &Args) {
                      auto GetArg = [=](unsigned I) {
                        return cast<ConstantInt>(Args[I])->getZExtValue();
                      };
                      auto MScope = static_cast<Scope>(GetArg(0));
                      auto Sema = mapSPIRVMemSemanticToOCL(GetArg(1));
                      Args.resize(3);
                      Args[0] = getInt32(M, Sema.first);
                      Args[1] = getInt32(M, Sema.second);
                      Args[2] = getInt32(M, rmap<OCLScopeKind>(MScope));
                      return kOCLBuiltinName::AtomicWorkItemFence;
                    },
                    &Attrs);
}

void SPIRVToOCL20::visitCallSPRIVImageQuerySize(CallInst *CI) {
  Function *Func = CI->getCalledFunction();
  // Get image type
  Type *ArgTy = Func->getFunctionType()->getParamType(0);
  assert(ArgTy->isPointerTy() &&
         "argument must be a pointer to opaque structure");
  StructType *ImgTy = cast<StructType>(ArgTy->getPointerElementType());
  assert(ImgTy->isOpaque() && "image type must be an opaque structure");
  StringRef ImgTyName = ImgTy->getName();
  assert(ImgTyName.startswith("opencl.image") && "not an OCL image type");

  unsigned ImgDim = 0;
  bool ImgArray = false;

  if (ImgTyName.startswith("opencl.image1d")) {
    ImgDim = 1;
  } else if (ImgTyName.startswith("opencl.image2d")) {
    ImgDim = 2;
  } else if (ImgTyName.startswith("opencl.image3d")) {
    ImgDim = 3;
  }
  assert(ImgDim != 0 && "unexpected image dimensionality");

  if (ImgTyName.count("_array_") != 0) {
    ImgArray = true;
  }

  AttributeList Attributes = CI->getCalledFunction()->getAttributes();
  BuiltinFuncMangleInfo Mangle;
  Type *Int32Ty = Type::getInt32Ty(*Ctx);
  Instruction *GetImageSize = nullptr;

  if (ImgDim == 1) {
    // OpImageQuerySize from non-arrayed 1d image is always translated
    // into get_image_width returning scalar argument
    GetImageSize = addCallInst(M, kOCLBuiltinName::GetImageWidth, Int32Ty,
                               CI->getArgOperand(0), &Attributes, CI, &Mangle,
                               CI->getName(), false);
    // The width of integer type returning by OpImageQuerySize[Lod] may
    // differ from i32
    if (CI->getType()->getScalarType() != Int32Ty) {
      GetImageSize = CastInst::CreateIntegerCast(GetImageSize,
                                                 CI->getType()->getScalarType(),
                                                 false, CI->getName(), CI);
    }
  } else {
    assert((ImgDim == 2 || ImgDim == 3) && "invalid image type");
    assert(CI->getType()->isVectorTy() &&
           "this code can handle vector result type only");
    // get_image_dim returns int2 and int4 for 2d and 3d images respecitvely.
    const unsigned ImgDimRetEls = ImgDim == 2 ? 2 : 4;
    VectorType *RetTy = VectorType::get(Int32Ty, ImgDimRetEls);
    GetImageSize = addCallInst(M, kOCLBuiltinName::GetImageDim, RetTy,
                               CI->getArgOperand(0), &Attributes, CI, &Mangle,
                               CI->getName(), false);
    // The width of integer type returning by OpImageQuerySize[Lod] may
    // differ from i32
    if (CI->getType()->getScalarType() != Int32Ty) {
      GetImageSize = CastInst::CreateIntegerCast(
          GetImageSize,
          VectorType::get(CI->getType()->getScalarType(),
                          GetImageSize->getType()->getVectorNumElements()),
          false, CI->getName(), CI);
    }
  }

  if (ImgArray || ImgDim == 3) {
    assert(
        CI->getType()->isVectorTy() &&
        "OpImageQuerySize[Lod] must return vector for arrayed and 3d images");
    const unsigned ImgQuerySizeRetEls = CI->getType()->getVectorNumElements();

    if (ImgDim == 1) {
      // get_image_width returns scalar result while OpImageQuerySize
      // for image1d_array_t returns <2 x i32> vector.
      assert(ImgQuerySizeRetEls == 2 &&
             "OpImageQuerySize[Lod] must return <2 x iN> vector type");
      GetImageSize = InsertElementInst::Create(
          UndefValue::get(CI->getType()), GetImageSize,
          ConstantInt::get(Int32Ty, 0), CI->getName(), CI);
    } else {
      // get_image_dim and OpImageQuerySize returns different vector
      // types for arrayed and 3d images.
      SmallVector<Constant *, 4> MaskEls;
      for (unsigned Idx = 0; Idx < ImgQuerySizeRetEls; ++Idx)
        MaskEls.push_back(ConstantInt::get(Int32Ty, Idx));
      Constant *Mask = ConstantVector::get(MaskEls);

      GetImageSize = new ShuffleVectorInst(
          GetImageSize, UndefValue::get(GetImageSize->getType()), Mask,
          CI->getName(), CI);
    }
  }

  if (ImgArray) {
    assert((ImgDim == 1 || ImgDim == 2) && "invalid image array type");
    // Insert get_image_array_size to the last position of the resulting vector.
    Type *SizeTy =
        Type::getIntNTy(*Ctx, M->getDataLayout().getPointerSizeInBits(0));
    Instruction *GetImageArraySize = addCallInst(
        M, kOCLBuiltinName::GetImageArraySize, SizeTy, CI->getArgOperand(0),
        &Attributes, CI, &Mangle, CI->getName(), false);
    // The width of integer type returning by OpImageQuerySize[Lod] may
    // differ from size_t which is returned by get_image_array_size
    if (GetImageArraySize->getType() != CI->getType()->getScalarType()) {
      GetImageArraySize = CastInst::CreateIntegerCast(
          GetImageArraySize, CI->getType()->getScalarType(), false,
          CI->getName(), CI);
    }
    GetImageSize = InsertElementInst::Create(
        GetImageSize, GetImageArraySize,
        ConstantInt::get(Int32Ty, CI->getType()->getVectorNumElements() - 1),
        CI->getName(), CI);
  }

  assert(GetImageSize && "must not be null");
  CI->replaceAllUsesWith(GetImageSize);
  CI->eraseFromParent();
}

void SPIRVToOCL20::visitCallSPIRVAtomicBuiltin(CallInst *CI, Op OC) {
  assert(CI->getCalledFunction() && "Unexpected indirect call");
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  Instruction *PInsertBefore = CI;

  mutateCallInstOCL(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args, Type *&RetTy) {
        auto Ptr = findFirstPtr(Args);
        auto Name = OCLSPIRVBuiltinMap::rmap(OC);
        auto NumOrder = getAtomicBuiltinNumMemoryOrderArgs(Name);
        auto ScopeIdx = Ptr + 1;
        auto OrderIdx = Ptr + 2;
        if (OC == OpAtomicIIncrement || OC == OpAtomicIDecrement) {
          // Since OpenCL 1.2 atomic_inc and atomic_dec builtins don't have,
          // memory scope and memory order syntax, and OpenCL 2.0 doesn't have
          // such builtins, therefore we translate these instructions to
          // atomic_fetch_add_explicit and atomic_fetch_sub_explicit OpenCL 2.0
          // builtins with "operand" argument = 1.
          Name = OCLSPIRVBuiltinMap::rmap(
              OC == OpAtomicIIncrement ? OpAtomicIAdd : OpAtomicISub);
          Type *ValueTy =
              cast<PointerType>(Args[Ptr]->getType())->getElementType();
          assert(ValueTy->isIntegerTy());
          Args.push_back(llvm::ConstantInt::get(ValueTy, 1));
        }
        Args[ScopeIdx] =
            mapUInt(M, cast<ConstantInt>(Args[ScopeIdx]), [](unsigned I) {
              return rmap<OCLScopeKind>(static_cast<Scope>(I));
            });
        for (size_t I = 0; I < NumOrder; ++I)
          Args[OrderIdx + I] =
              mapUInt(M, cast<ConstantInt>(Args[OrderIdx + I]),
                      [](unsigned Ord) { return mapSPIRVMemOrderToOCL(Ord); });
        std::swap(Args[ScopeIdx], Args.back());
        if (OC == OpAtomicCompareExchange ||
            OC == OpAtomicCompareExchangeWeak) {
          // OpAtomicCompareExchange[Weak] semantics is different from
          // atomic_compare_exchange_[strong|weak] semantics as well as
          // arguments order.
          // OCL built-ins returns boolean value and stores a new/original
          // value by pointer passed as 2nd argument (aka expected) while SPIR-V
          // instructions returns this new/original value as a resulting value.
          AllocaInst *PExpected =
              new AllocaInst(CI->getType(), 0, "expected",
                             &(*PInsertBefore->getParent()
                                    ->getParent()
                                    ->getEntryBlock()
                                    .getFirstInsertionPt()));
          PExpected->setAlignment(CI->getType()->getScalarSizeInBits() / 8);
          new StoreInst(Args[1], PExpected, PInsertBefore);
          Args[1] = PExpected;
          std::swap(Args[3], Args[4]);
          std::swap(Args[2], Args[3]);
          RetTy = Type::getInt1Ty(*Ctx);
        }
        return Name;
      },
      [=](CallInst *CI) -> Instruction * {
        if (OC == OpAtomicCompareExchange ||
            OC == OpAtomicCompareExchangeWeak) {
          // OCL built-ins atomic_compare_exchange_[strong|weak] return boolean
          // value. So, to obtain the same value as SPIR-V instruction is
          // returning it has to be loaded from the memory where 'expected'
          // value is stored. This memory must contain the needed value after a
          // call to OCL built-in is completed.
          LoadInst *POriginal =
              new LoadInst(CI->getArgOperand(1), "original", PInsertBefore);
          return POriginal;
        }
        // For other built-ins the return values match.
        return CI;
      },
      &Attrs);
}

void SPIRVToOCL20::visitCallSPIRVBuiltin(CallInst *CI, Op OC) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  mutateCallInstOCL(M, CI,
                    [=](CallInst *, std::vector<Value *> &Args) {
                      return OCLSPIRVBuiltinMap::rmap(OC);
                    },
                    &Attrs);
}

void SPIRVToOCL20::visitCallSPIRVGroupBuiltin(CallInst *CI, Op OC) {
  auto DemangledName = OCLSPIRVBuiltinMap::rmap(OC);
  assert(DemangledName.find(kSPIRVName::GroupPrefix) == 0);

  std::string Prefix = getGroupBuiltinPrefix(CI);

  bool HasGroupOperation = hasGroupOperation(OC);
  if (!HasGroupOperation) {
    DemangledName = Prefix + DemangledName;
  } else {
    auto GO = getArgAs<spv::GroupOperation>(CI, 1);
    StringRef Op = DemangledName;
    Op = Op.drop_front(strlen(kSPIRVName::GroupPrefix));
    bool Unsigned = Op.front() == 'u';
    if (!Unsigned)
      Op = Op.drop_front(1);
    DemangledName = Prefix + kSPIRVName::GroupPrefix +
                    SPIRSPIRVGroupOperationMap::rmap(GO) + '_' + Op.str();
  }
  assert(CI->getCalledFunction() && "Unexpected indirect call");
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  mutateCallInstOCL(M, CI,
                    [=](CallInst *, std::vector<Value *> &Args) {
                      Args.erase(Args.begin(),
                                 Args.begin() + (HasGroupOperation ? 2 : 1));
                      if (OC == OpGroupBroadcast)
                        expandVector(CI, Args, 1);
                      return DemangledName;
                    },
                    &Attrs);
}

void SPIRVToOCL20::visitCallSPIRVPipeBuiltin(CallInst *CI, Op OC) {
  auto DemangledName = OCLSPIRVBuiltinMap::rmap(OC);
  bool HasScope = DemangledName.find(kSPIRVName::GroupPrefix) == 0;
  if (HasScope)
    DemangledName = getGroupBuiltinPrefix(CI) + DemangledName;

  assert(CI->getCalledFunction() && "Unexpected indirect call");
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  mutateCallInstOCL(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        if (HasScope)
          Args.erase(Args.begin(), Args.begin() + 1);

        if (!(OC == OpReadPipe || OC == OpWritePipe ||
              OC == OpReservedReadPipe || OC == OpReservedWritePipe))
          return DemangledName;

        auto &P = Args[Args.size() - 3];
        auto T = P->getType();
        assert(isa<PointerType>(T));
        auto ET = T->getPointerElementType();
        if (!ET->isIntegerTy(8) ||
            T->getPointerAddressSpace() != SPIRAS_Generic) {
          auto NewTy = PointerType::getInt8PtrTy(*Ctx, SPIRAS_Generic);
          P = CastInst::CreatePointerBitCastOrAddrSpaceCast(P, NewTy, "", CI);
        }
        return DemangledName;
      },
      &Attrs);
}

void SPIRVToOCL20::translateMangledAtomicTypeName() {
  for (auto &I : M->functions()) {
    if (!I.hasName())
      continue;
    std::string MangledName = I.getName();
    std::string DemangledName;
    if (!oclIsBuiltin(MangledName, &DemangledName) ||
        DemangledName.find(kOCLBuiltinName::AtomPrefix) != 0)
      continue;
    auto Loc = MangledName.find(kOCLBuiltinName::AtomPrefix);
    Loc = MangledName.find(kMangledName::AtomicPrefixInternal, Loc);
    MangledName.replace(Loc, strlen(kMangledName::AtomicPrefixInternal),
                        MangledAtomicTypeNamePrefix);
    I.setName(MangledName);
  }
}

std::string SPIRVToOCL20::getGroupBuiltinPrefix(CallInst *CI) {
  std::string Prefix;
  auto ES = getArgAsScope(CI, 0);
  switch (ES) {
  case ScopeWorkgroup:
    Prefix = kOCLBuiltinName::WorkPrefix;
    break;
  case ScopeSubgroup:
    Prefix = kOCLBuiltinName::SubPrefix;
    break;
  default:
    llvm_unreachable("Invalid execution scope");
  }
  return Prefix;
}

void SPIRVToOCL20::visitCastInst(CastInst &Cast) {
  if (!isa<ZExtInst>(Cast) && !isa<SExtInst>(Cast) && !isa<TruncInst>(Cast) &&
      !isa<FPTruncInst>(Cast) && !isa<FPExtInst>(Cast) &&
      !isa<FPToUIInst>(Cast) && !isa<FPToSIInst>(Cast) &&
      !isa<UIToFPInst>(Cast) && !isa<SIToFPInst>(Cast))
    return;

  Type const *SrcTy = Cast.getSrcTy();
  Type *DstVecTy = Cast.getDestTy();
  // Leave scalar casts as is. Skip boolean vector casts becase there
  // are no suitable OCL built-ins.
  if (!DstVecTy->isVectorTy() || SrcTy->getScalarSizeInBits() == 1 ||
      DstVecTy->getScalarSizeInBits() == 1)
    return;

  // Assemble built-in name -> convert_gentypeN
  std::string CastBuiltInName(kOCLBuiltinName::ConvertPrefix);
  // Check if this is 'floating point -> unsigned integer' cast
  CastBuiltInName += mapLLVMTypeToOCLType(DstVecTy, !isa<FPToUIInst>(Cast));

  // Replace LLVM conversion instruction with call to conversion built-in
  BuiltinFuncMangleInfo Mangle;
  // It does matter if the source is unsigned integer or not. SExt is for
  // signed source, ZExt and UIToFPInst are for unsigned source.
  if (isa<ZExtInst>(Cast) || isa<UIToFPInst>(Cast))
    Mangle.addUnsignedArg(0);

  AttributeList Attributes;
  CallInst *Call =
      addCallInst(M, CastBuiltInName, DstVecTy, Cast.getOperand(0), &Attributes,
                  &Cast, &Mangle, Cast.getName(), false);
  Cast.replaceAllUsesWith(Call);
  Cast.eraseFromParent();
}

} // namespace SPIRV

INITIALIZE_PASS(SPIRVToOCL20, "spvtoocl20",
                "Translate SPIR-V builtins to OCL 2.0 builtins", false, false)

ModulePass *llvm::createSPIRVToOCL20() { return new SPIRVToOCL20(); }
