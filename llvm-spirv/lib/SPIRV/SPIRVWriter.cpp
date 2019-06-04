//===- SPIRVWriter.cpp - Converts LLVM to SPIR-V ----------------*- C++ -*-===//
//
//                     The LLVM/SPIR-V Translator
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
/// \file
///
/// This file implements conversion of LLVM intermediate language to SPIR-V
/// binary.
///
//===----------------------------------------------------------------------===//

#include "SPIRVWriter.h"
#include "LLVMToSPIRVDbgTran.h"
#include "SPIRVBasicBlock.h"
#include "SPIRVEntry.h"
#include "SPIRVEnum.h"
#include "SPIRVExtInst.h"
#include "SPIRVFunction.h"
#include "SPIRVInstruction.h"
#include "SPIRVInternal.h"
#include "SPIRVMDWalker.h"
#include "SPIRVModule.h"
#include "SPIRVType.h"
#include "SPIRVUtil.h"
#include "SPIRVValue.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/PassSupport.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Transforms/Utils.h" // loop-simplify pass

#include <cstdlib>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <set>
#include <sstream>
#include <vector>

#define DEBUG_TYPE "spirv"

using namespace llvm;
using namespace SPIRV;
using namespace OCLUtil;

namespace SPIRV {

cl::opt<bool> SPIRVMemToReg("spirv-mem2reg", cl::init(true),
                            cl::desc("LLVM/SPIR-V translation enable mem2reg"));

cl::opt<bool> SPIRVNoDerefAttr(
    "spirv-no-deref-attr", cl::init(false),
    cl::desc("Do not translate 'dereferenceable' LLVM attribute to SPIR-V"));

static void foreachKernelArgMD(
    MDNode *MD, SPIRVFunction *BF,
    std::function<void(const std::string &Str, SPIRVFunctionParameter *BA)>
        Func) {
  for (unsigned I = 0, E = MD->getNumOperands(); I != E; ++I) {
    SPIRVFunctionParameter *BA = BF->getArgument(I);
    Func(getMDOperandAsString(MD, I), BA);
  }
}

/// Information for translating OCL builtin.
struct OCLBuiltinSPIRVTransInfo {
  std::string UniqName;
  /// Postprocessor of operands
  std::function<void(std::vector<SPIRVWord> &)> PostProc;
  OCLBuiltinSPIRVTransInfo() {
    PostProc = [](std::vector<SPIRVWord> &) {};
  }
};

LLVMToSPIRV::LLVMToSPIRV(SPIRVModule *SMod)
    : ModulePass(ID), M(nullptr), Ctx(nullptr), BM(SMod), SrcLang(0),
      SrcLangVer(0) {
  DbgTran = make_unique<LLVMToSPIRVDbgTran>(nullptr, SMod, this);
}

bool LLVMToSPIRV::runOnModule(Module &Mod) {
  M = &Mod;
  CG = make_unique<CallGraph>(Mod);
  Ctx = &M->getContext();
  DbgTran->setModule(M);
  assert(BM && "SPIR-V module not initialized");
  translate();
  return true;
}

SPIRVValue *LLVMToSPIRV::getTranslatedValue(const Value *V) const {
  auto Loc = ValueMap.find(V);
  if (Loc != ValueMap.end())
    return Loc->second;
  return nullptr;
}

bool LLVMToSPIRV::oclIsKernel(Function *F) {
  if (F->getCallingConv() == CallingConv::SPIR_KERNEL)
    return true;
  return false;
}

bool LLVMToSPIRV::isBuiltinTransToInst(Function *F) {
  std::string DemangledName;
  if (!oclIsBuiltin(F->getName(), &DemangledName) &&
      !isDecoratedSPIRVFunc(F, &DemangledName))
    return false;
  SPIRVDBG(spvdbgs() << "CallInst: demangled name: " << DemangledName << '\n');
  return getSPIRVFuncOC(DemangledName) != OpNop;
}

bool LLVMToSPIRV::isBuiltinTransToExtInst(Function *F,
                                          SPIRVExtInstSetKind *ExtSet,
                                          SPIRVWord *ExtOp,
                                          SmallVectorImpl<std::string> *Dec) {
  std::string OrigName = F->getName();
  std::string DemangledName;
  if (!oclIsBuiltin(OrigName, &DemangledName))
    return false;
  LLVM_DEBUG(dbgs() << "[oclIsBuiltinTransToExtInst] CallInst: demangled name: "
                    << DemangledName << '\n');
  StringRef S = DemangledName;
  if (!S.startswith(kSPIRVName::Prefix))
    return false;
  S = S.drop_front(strlen(kSPIRVName::Prefix));
  auto Loc = S.find(kSPIRVPostfix::Divider);
  auto ExtSetName = S.substr(0, Loc);
  SPIRVExtInstSetKind Set = SPIRVEIS_Count;
  if (!SPIRVExtSetShortNameMap::rfind(ExtSetName, &Set))
    return false;
  assert((Set == SPIRVEIS_OpenCL || Set == SPIRVEIS_Debug) &&
         "Unsupported extended instruction set");

  auto ExtOpName = S.substr(Loc + 1);
  auto Splited = ExtOpName.split(kSPIRVPostfix::ExtDivider);
  OCLExtOpKind EOC;
  if (!OCLExtOpMap::rfind(Splited.first, &EOC))
    return false;

  if (ExtSet)
    *ExtSet = Set;
  if (ExtOp)
    *ExtOp = EOC;
  if (Dec) {
    SmallVector<StringRef, 2> P;
    Splited.second.split(P, kSPIRVPostfix::Divider);
    for (auto &I : P)
      Dec->push_back(I.str());
  }
  return true;
}

/// Decode SPIR-V type name in the format spirv.{TypeName}._{Postfixes}
/// where Postfixes are strings separated by underscores.
/// \return TypeName.
/// \param Ops contains the integers decoded from postfixes.
static std::string decodeSPIRVTypeName(StringRef Name,
                                       SmallVectorImpl<std::string> &Strs) {
  SmallVector<StringRef, 4> SubStrs;
  const char Delim[] = {kSPIRVTypeName::Delimiter, 0};
  Name.split(SubStrs, Delim, -1, true);
  assert(SubStrs.size() >= 2 && "Invalid SPIRV type name");
  assert(SubStrs[0] == kSPIRVTypeName::Prefix && "Invalid prefix");
  assert((SubStrs.size() == 2 || !SubStrs[2].empty()) && "Invalid postfix");

  if (SubStrs.size() > 2) {
    const char PostDelim[] = {kSPIRVTypeName::PostfixDelim, 0};
    SmallVector<StringRef, 4> Postfixes;
    SubStrs[2].split(Postfixes, PostDelim, -1, true);
    assert(Postfixes.size() > 1 && Postfixes[0].empty() && "Invalid postfix");
    for (unsigned I = 1, E = Postfixes.size(); I != E; ++I)
      Strs.push_back(std::string(Postfixes[I]).c_str());
  }
  return SubStrs[1].str();
}

static bool recursiveType(const StructType *ST, const Type *Ty) {
  SmallPtrSet<const StructType *, 4> Seen;

  std::function<bool(const Type *Ty)> Run = [&](const Type *Ty) {
    if (!isa<CompositeType>(Ty) && !Ty->isPointerTy())
      return false;

    if (auto *StructTy = dyn_cast<StructType>(Ty)) {
      if (StructTy == ST)
        return true;

      if (Seen.count(StructTy))
        return false;

      Seen.insert(StructTy);

      return find_if(StructTy->element_begin(), StructTy->element_end(), Run) !=
             StructTy->element_end();
    }

    if (auto *PtrTy = dyn_cast<PointerType>(Ty))
      return Run(PtrTy->getPointerElementType());

    if (auto *ArrayTy = dyn_cast<ArrayType>(Ty))
      return Run(ArrayTy->getArrayElementType());

    return false;
  };

  return Run(Ty);
}

SPIRVType *LLVMToSPIRV::transType(Type *T) {
  LLVMToSPIRVTypeMap::iterator Loc = TypeMap.find(T);
  if (Loc != TypeMap.end())
    return Loc->second;

  SPIRVDBG(dbgs() << "[transType] " << *T << '\n');
  if (T->isVoidTy())
    return mapType(T, BM->addVoidType());

  if (T->isIntegerTy(1))
    return mapType(T, BM->addBoolType());

  if (T->isIntegerTy())
    return mapType(T, BM->addIntegerType(T->getIntegerBitWidth()));

  if (T->isFloatingPointTy())
    return mapType(T, BM->addFloatType(T->getPrimitiveSizeInBits()));

  // A pointer to image or pipe type in LLVM is translated to a SPIRV
  // sampler or pipe type.
  if (T->isPointerTy()) {
    auto ET = T->getPointerElementType();
    assert(!ET->isFunctionTy() && "Function pointer type is not allowed");
    auto ST = dyn_cast<StructType>(ET);
    auto AddrSpc = T->getPointerAddressSpace();
    if (ST && !ST->isSized()) {
      Op OpCode;
      StringRef STName = ST->getName();
      // Workaround for non-conformant SPIR binary
      if (STName == "struct._event_t") {
        STName = kSPR2TypeName::Event;
        ST->setName(STName);
      }
      if (STName.startswith(kSPR2TypeName::PipeRO) ||
          STName.startswith(kSPR2TypeName::PipeWO)) {
        auto PipeT = BM->addPipeType();
        PipeT->setPipeAcessQualifier(STName.startswith(kSPR2TypeName::PipeRO)
                                         ? AccessQualifierReadOnly
                                         : AccessQualifierWriteOnly);
        return mapType(T, PipeT);
      }
      if (STName.startswith(kSPR2TypeName::ImagePrefix)) {
        assert(AddrSpc == SPIRAS_Global);
        auto SPIRVImageTy = getSPIRVImageTypeFromOCL(M, T);
        return mapType(T, transType(SPIRVImageTy));
      }
      if (STName == kSPR2TypeName::Sampler)
        return mapType(T, transType(getSamplerType(M)));
      if (STName.startswith(kSPIRVTypeName::PrefixAndDelim))
        return transSPIRVOpaqueType(T);

      if (STName.startswith(kOCLSubgroupsAVCIntel::TypePrefix))
        return mapType(T,
                       BM->addSubgroupAvcINTELType(
                           OCLSubgroupINTELTypeOpCodeMap::map(ST->getName())));

      if (OCLOpaqueTypeOpCodeMap::find(STName, &OpCode)) {
        switch (OpCode) {
        default:
          return mapType(T, BM->addOpaqueGenericType(OpCode));
        case OpTypeDeviceEvent:
          return mapType(T, BM->addDeviceEventType());
        case OpTypeQueue:
          return mapType(T, BM->addQueueType());
        }
      }

      if (isPointerToOpaqueStructType(T)) {
        return mapType(
            T, BM->addPointerType(SPIRSPIRVAddrSpaceMap::map(
                                      static_cast<SPIRAddressSpace>(AddrSpc)),
                                  transType(ET)));
      }
    } else {
      return mapType(
          T, BM->addPointerType(SPIRSPIRVAddrSpaceMap::map(
                                    static_cast<SPIRAddressSpace>(AddrSpc)),
                                transType(ET)));
    }
  }

  if (T->isVectorTy())
    return mapType(T, BM->addVectorType(transType(T->getVectorElementType()),
                                        T->getVectorNumElements()));

  if (T->isArrayTy()) {
    // SPIR-V 1.3 s3.32.6: Length is the number of elements in the array.
    //                     It must be at least 1.
    if (T->getArrayNumElements() < 1) {
      std::string Str;
      llvm::raw_string_ostream OS(Str);
      OS << *T;
      SPIRVCK(T->getArrayNumElements() >= 1, InvalidArraySize, OS.str());
    }
    return mapType(T, BM->addArrayType(
                          transType(T->getArrayElementType()),
                          static_cast<SPIRVConstant *>(transValue(
                              ConstantInt::get(getSizetType(),
                                               T->getArrayNumElements(), false),
                              nullptr))));
  }

  if (T->isStructTy() && !T->isSized()) {
    auto ST = dyn_cast<StructType>(T);
    (void)ST; // Silence warning
    assert(!ST->getName().startswith(kSPR2TypeName::PipeRO));
    assert(!ST->getName().startswith(kSPR2TypeName::PipeWO));
    assert(!ST->getName().startswith(kSPR2TypeName::ImagePrefix));
    return mapType(T, BM->addOpaqueType(T->getStructName()));
  }

  if (auto ST = dyn_cast<StructType>(T)) {
    assert(ST->isSized());

    std::string Name;
    if (ST->hasName())
      Name = ST->getName();

    if (Name == getSPIRVTypeName(kSPIRVTypeName::ConstantSampler))
      return transType(getSamplerType(M));
    if (Name == getSPIRVTypeName(kSPIRVTypeName::ConstantPipeStorage))
      return transType(getPipeStorageType(M));

    auto *Struct = BM->openStructType(T->getStructNumElements(), Name);
    mapType(T, Struct);

    SmallVector<unsigned, 4> ForwardRefs;

    for (unsigned I = 0, E = T->getStructNumElements(); I != E; ++I) {
      auto *ElemTy = ST->getElementType(I);
      if ((isa<CompositeType>(ElemTy) || isa<PointerType>(ElemTy)) &&
          recursiveType(ST, ElemTy))
        ForwardRefs.push_back(I);
      else
        Struct->setMemberType(I, transType(ST->getElementType(I)));
    }

    BM->closeStructType(Struct, ST->isPacked());

    for (auto I : ForwardRefs)
      Struct->setMemberType(I, transType(ST->getElementType(I)));

    return Struct;
  }

  if (FunctionType *FT = dyn_cast<FunctionType>(T)) {
    SPIRVType *RT = transType(FT->getReturnType());
    std::vector<SPIRVType *> PT;
    for (FunctionType::param_iterator I = FT->param_begin(),
                                      E = FT->param_end();
         I != E; ++I)
      PT.push_back(transType(*I));
    return mapType(T, BM->addFunctionType(RT, PT));
  }

  llvm_unreachable("Not implemented!");
  return 0;
}

SPIRVType *LLVMToSPIRV::transSPIRVOpaqueType(Type *T) {
  auto ET = T->getPointerElementType();
  auto ST = cast<StructType>(ET);
  auto STName = ST->getStructName();
  assert(STName.startswith(kSPIRVTypeName::PrefixAndDelim) &&
         "Invalid SPIR-V opaque type name");
  SmallVector<std::string, 8> Postfixes;
  auto TN = decodeSPIRVTypeName(STName, Postfixes);
  if (TN == kSPIRVTypeName::Pipe) {
    assert(T->getPointerAddressSpace() == SPIRAS_Global);
    assert(Postfixes.size() == 1 && "Invalid pipe type ops");
    auto PipeT = BM->addPipeType();
    PipeT->setPipeAcessQualifier(
        static_cast<spv::AccessQualifier>(atoi(Postfixes[0].c_str())));
    return mapType(T, PipeT);
  } else if (TN == kSPIRVTypeName::Image) {
    assert(T->getPointerAddressSpace() == SPIRAS_Global);
    // The sampled type needs to be translated through LLVM type to guarantee
    // uniqueness.
    auto SampledT = transType(
        getLLVMTypeForSPIRVImageSampledTypePostfix(Postfixes[0], *Ctx));
    SmallVector<int, 7> Ops;
    for (unsigned I = 1; I < 8; ++I)
      Ops.push_back(atoi(Postfixes[I].c_str()));
    SPIRVTypeImageDescriptor Desc(static_cast<SPIRVImageDimKind>(Ops[0]),
                                  Ops[1], Ops[2], Ops[3], Ops[4], Ops[5]);
    return mapType(T,
                   BM->addImageType(SampledT, Desc,
                                    static_cast<spv::AccessQualifier>(Ops[6])));
  } else if (TN == kSPIRVTypeName::SampledImg) {
    return mapType(
        T, BM->addSampledImageType(static_cast<SPIRVTypeImage *>(
               transType(getSPIRVTypeByChangeBaseTypeName(
                   M, T, kSPIRVTypeName::SampledImg, kSPIRVTypeName::Image)))));
  } else if (TN == kSPIRVTypeName::VmeImageINTEL) {
    // This type is the same as SampledImageType, but consumed by Subgroup AVC
    // Intel extension instructions.
    return mapType(
        T,
        BM->addVmeImageINTELType(static_cast<SPIRVTypeImage *>(
            transType(getSPIRVTypeByChangeBaseTypeName(
                M, T, kSPIRVTypeName::VmeImageINTEL, kSPIRVTypeName::Image)))));
  } else if (TN == kSPIRVTypeName::Sampler)
    return mapType(T, BM->addSamplerType());
  else if (TN == kSPIRVTypeName::DeviceEvent)
    return mapType(T, BM->addDeviceEventType());
  else if (TN == kSPIRVTypeName::Queue)
    return mapType(T, BM->addQueueType());
  else if (TN == kSPIRVTypeName::PipeStorage)
    return mapType(T, BM->addPipeStorageType());
  else
    return mapType(T,
                   BM->addOpaqueGenericType(SPIRVOpaqueTypeOpCodeMap::map(TN)));
}

SPIRVFunction *LLVMToSPIRV::transFunctionDecl(Function *F) {
  if (auto BF = getTranslatedValue(F))
    return static_cast<SPIRVFunction *>(BF);

  if (F->isIntrinsic()) {
    // We should not translate LLVM intrinsics as a function
    assert(none_of(F->user_begin(), F->user_end(),
                   [this](User *U) { return getTranslatedValue(U); }) &&
           "LLVM intrinsics shouldn't be called in SPIRV");
    return nullptr;
  }

  SPIRVTypeFunction *BFT = static_cast<SPIRVTypeFunction *>(
      transType(getAnalysis<OCLTypeToSPIRV>().getAdaptedType(F)));
  SPIRVFunction *BF =
      static_cast<SPIRVFunction *>(mapValue(F, BM->addFunction(BFT)));
  BF->setFunctionControlMask(transFunctionControlMask(F));
  if (F->hasName())
    BM->setName(BF, F->getName());
  if (oclIsKernel(F))
    BM->addEntryPoint(ExecutionModelKernel, BF->getId());
  else if (F->getLinkage() != GlobalValue::InternalLinkage)
    BF->setLinkageType(transLinkageType(F));
  auto Attrs = F->getAttributes();
  for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end(); I != E;
       ++I) {
    auto ArgNo = I->getArgNo();
    SPIRVFunctionParameter *BA = BF->getArgument(ArgNo);
    if (I->hasName())
      BM->setName(BA, I->getName());
    if (I->hasByValAttr())
      BA->addAttr(FunctionParameterAttributeByVal);
    if (I->hasNoAliasAttr())
      BA->addAttr(FunctionParameterAttributeNoAlias);
    if (I->hasNoCaptureAttr())
      BA->addAttr(FunctionParameterAttributeNoCapture);
    if (I->hasStructRetAttr())
      BA->addAttr(FunctionParameterAttributeSret);
    if (Attrs.hasAttribute(ArgNo + 1, Attribute::ZExt))
      BA->addAttr(FunctionParameterAttributeZext);
    if (Attrs.hasAttribute(ArgNo + 1, Attribute::SExt))
      BA->addAttr(FunctionParameterAttributeSext);
    if (!SPIRVNoDerefAttr &&
        Attrs.hasAttribute(ArgNo + 1, Attribute::Dereferenceable))
      BA->addDecorate(DecorationMaxByteOffset,
                      Attrs.getAttribute(ArgNo + 1, Attribute::Dereferenceable)
                          .getDereferenceableBytes());
  }
  if (Attrs.hasAttribute(AttributeList::ReturnIndex, Attribute::ZExt))
    BF->addDecorate(DecorationFuncParamAttr, FunctionParameterAttributeZext);
  if (Attrs.hasAttribute(AttributeList::ReturnIndex, Attribute::SExt))
    BF->addDecorate(DecorationFuncParamAttr, FunctionParameterAttributeSext);
  SPIRVDBG(dbgs() << "[transFunction] " << *F << " => ";
           spvdbgs() << *BF << '\n';)
  return BF;
}

#define _SPIRV_OPL(x) OpLogical##x

#define _SPIRV_OPB(x) OpBitwise##x

SPIRVValue *LLVMToSPIRV::transConstant(Value *V) {
  if (auto CPNull = dyn_cast<ConstantPointerNull>(V))
    return BM->addNullConstant(
        bcast<SPIRVTypePointer>(transType(CPNull->getType())));

  if (auto CAZero = dyn_cast<ConstantAggregateZero>(V)) {
    Type *AggType = CAZero->getType();
    if (const StructType *ST = dyn_cast<StructType>(AggType))
      if (ST->hasName() &&
          ST->getName() == getSPIRVTypeName(kSPIRVTypeName::ConstantSampler))
        return BM->addSamplerConstant(transType(AggType), 0, 0, 0);

    return BM->addNullConstant(transType(AggType));
  }

  if (auto ConstI = dyn_cast<ConstantInt>(V))
    return BM->addConstant(transType(V->getType()), ConstI->getZExtValue());

  if (auto ConstFP = dyn_cast<ConstantFP>(V)) {
    auto BT = static_cast<SPIRVType *>(transType(V->getType()));
    return BM->addConstant(
        BT, ConstFP->getValueAPF().bitcastToAPInt().getZExtValue());
  }

  if (auto ConstDA = dyn_cast<ConstantDataArray>(V)) {
    std::vector<SPIRVValue *> BV;
    for (unsigned I = 0, E = ConstDA->getNumElements(); I != E; ++I)
      BV.push_back(transValue(ConstDA->getElementAsConstant(I), nullptr));
    return BM->addCompositeConstant(transType(V->getType()), BV);
  }

  if (auto ConstA = dyn_cast<ConstantArray>(V)) {
    std::vector<SPIRVValue *> BV;
    for (auto I = ConstA->op_begin(), E = ConstA->op_end(); I != E; ++I)
      BV.push_back(transValue(*I, nullptr));
    return BM->addCompositeConstant(transType(V->getType()), BV);
  }

  if (auto ConstDV = dyn_cast<ConstantDataVector>(V)) {
    std::vector<SPIRVValue *> BV;
    for (unsigned I = 0, E = ConstDV->getNumElements(); I != E; ++I)
      BV.push_back(transValue(ConstDV->getElementAsConstant(I), nullptr));
    return BM->addCompositeConstant(transType(V->getType()), BV);
  }

  if (auto ConstV = dyn_cast<ConstantVector>(V)) {
    std::vector<SPIRVValue *> BV;
    for (auto I = ConstV->op_begin(), E = ConstV->op_end(); I != E; ++I)
      BV.push_back(transValue(*I, nullptr));
    return BM->addCompositeConstant(transType(V->getType()), BV);
  }

  if (const auto *ConstV = dyn_cast<ConstantStruct>(V)) {
    StringRef StructName;
    if (ConstV->getType()->hasName())
      StructName = ConstV->getType()->getName();
    if (StructName == getSPIRVTypeName(kSPIRVTypeName::ConstantSampler)) {
      assert(ConstV->getNumOperands() == 3);
      SPIRVWord AddrMode =
                    ConstV->getOperand(0)->getUniqueInteger().getZExtValue(),
                Normalized =
                    ConstV->getOperand(1)->getUniqueInteger().getZExtValue(),
                FilterMode =
                    ConstV->getOperand(2)->getUniqueInteger().getZExtValue();
      assert(AddrMode < 5 && "Invalid addressing mode");
      assert(Normalized < 2 && "Invalid value of normalized coords");
      assert(FilterMode < 2 && "Invalid filter mode");
      SPIRVType *SamplerTy = transType(ConstV->getType());
      return BM->addSamplerConstant(SamplerTy, AddrMode, Normalized,
                                    FilterMode);
    }
    if (StructName == getSPIRVTypeName(kSPIRVTypeName::ConstantPipeStorage)) {
      assert(ConstV->getNumOperands() == 3);
      SPIRVWord PacketSize =
                    ConstV->getOperand(0)->getUniqueInteger().getZExtValue(),
                PacketAlign =
                    ConstV->getOperand(1)->getUniqueInteger().getZExtValue(),
                Capacity =
                    ConstV->getOperand(2)->getUniqueInteger().getZExtValue();
      assert(PacketAlign >= 1 && "Invalid packet alignment");
      assert(PacketSize >= PacketAlign && PacketSize % PacketAlign == 0 &&
             "Invalid packet size and/or alignment.");
      SPIRVType *PipeStorageTy = transType(ConstV->getType());
      return BM->addPipeStorageConstant(PipeStorageTy, PacketSize, PacketAlign,
                                        Capacity);
    }
    std::vector<SPIRVValue *> BV;
    for (auto I = ConstV->op_begin(), E = ConstV->op_end(); I != E; ++I)
      BV.push_back(transValue(*I, nullptr));
    return BM->addCompositeConstant(transType(V->getType()), BV);
  }

  if (auto ConstUE = dyn_cast<ConstantExpr>(V)) {
    auto Inst = ConstUE->getAsInstruction();
    SPIRVDBG(dbgs() << "ConstantExpr: " << *ConstUE << '\n';
             dbgs() << "Instruction: " << *Inst << '\n';)
    auto BI = transValue(Inst, nullptr, false);
    Inst->dropAllReferences();
    return BI;
  }

  if (isa<UndefValue>(V)) {
    return BM->addUndef(transType(V->getType()));
  }

  return nullptr;
}

SPIRVValue *LLVMToSPIRV::transValue(Value *V, SPIRVBasicBlock *BB,
                                    bool CreateForward) {
  LLVMToSPIRVValueMap::iterator Loc = ValueMap.find(V);
  if (Loc != ValueMap.end() && (!Loc->second->isForward() || CreateForward))
    return Loc->second;

  SPIRVDBG(dbgs() << "[transValue] " << *V << '\n');
  assert((!isa<Instruction>(V) || isa<GetElementPtrInst>(V) ||
          isa<CastInst>(V) || BB) &&
         "Invalid SPIRV BB");

  auto BV = transValueWithoutDecoration(V, BB, CreateForward);
  if (!BV || !transDecoration(V, BV))
    return nullptr;
  std::string Name = V->getName();
  if (!Name.empty()) // Don't erase the name, which BM might already have
    BM->setName(BV, Name);
  return BV;
}

SPIRVInstruction *LLVMToSPIRV::transBinaryInst(BinaryOperator *B,
                                               SPIRVBasicBlock *BB) {
  unsigned LLVMOC = B->getOpcode();
  auto Op0 = transValue(B->getOperand(0), BB);
  SPIRVInstruction *BI = BM->addBinaryInst(
      transBoolOpCode(Op0, OpCodeMap::map(LLVMOC)), transType(B->getType()),
      Op0, transValue(B->getOperand(1), BB), BB);
  checkFpContract(B, BB);
  return BI;
}

SPIRVInstruction *LLVMToSPIRV::transCmpInst(CmpInst *Cmp, SPIRVBasicBlock *BB) {
  auto Op0 = transValue(Cmp->getOperand(0), BB);
  SPIRVInstruction *BI = BM->addCmpInst(
      transBoolOpCode(Op0, CmpMap::map(Cmp->getPredicate())),
      transType(Cmp->getType()), Op0, transValue(Cmp->getOperand(1), BB), BB);
  return BI;
}

SPIRV::SPIRVInstruction *LLVMToSPIRV::transUnaryInst(UnaryInstruction *U,
                                                     SPIRVBasicBlock *BB) {
  Op BOC = OpNop;
  if (auto Cast = dyn_cast<AddrSpaceCastInst>(U)) {
    if (Cast->getDestTy()->getPointerAddressSpace() == SPIRAS_Generic) {
      assert(Cast->getSrcTy()->getPointerAddressSpace() != SPIRAS_Constant &&
             "Casts from constant address space to generic are illegal");
      BOC = OpPtrCastToGeneric;
    } else {
      assert(Cast->getDestTy()->getPointerAddressSpace() != SPIRAS_Constant &&
             "Casts from generic address space to constant are illegal");
      assert(Cast->getSrcTy()->getPointerAddressSpace() == SPIRAS_Generic);
      BOC = OpGenericCastToPtr;
    }
  } else {
    auto OpCode = U->getOpcode();
    BOC = OpCodeMap::map(OpCode);
  }

  auto Op = transValue(U->getOperand(0), BB);
  return BM->addUnaryInst(transBoolOpCode(Op, BOC), transType(U->getType()), Op,
                          BB);
}

/// An instruction may use an instruction from another BB which has not been
/// translated. SPIRVForward should be created as place holder for these
/// instructions and replaced later by the real instructions.
/// Use CreateForward = true to indicate such situation.
SPIRVValue *LLVMToSPIRV::transValueWithoutDecoration(Value *V,
                                                     SPIRVBasicBlock *BB,
                                                     bool CreateForward) {
  if (auto LBB = dyn_cast<BasicBlock>(V)) {
    auto BF =
        static_cast<SPIRVFunction *>(getTranslatedValue(LBB->getParent()));
    assert(BF && "Function not translated");
    BB = static_cast<SPIRVBasicBlock *>(mapValue(V, BM->addBasicBlock(BF)));
    BM->setName(BB, LBB->getName());
    return BB;
  }

  if (auto F = dyn_cast<Function>(V))
    return transFunctionDecl(F);

  if (auto GV = dyn_cast<GlobalVariable>(V)) {
    llvm::PointerType *Ty = GV->getType();
    // Though variables with common linkage type are initialized by 0,
    // they can be represented in SPIR-V as uninitialized variables with
    // 'Export' linkage type, just as tentative definitions look in C
    llvm::Value *Init = GV->hasInitializer() && !GV->hasCommonLinkage()
                            ? GV->getInitializer()
                            : nullptr;
    StructType *ST = Init ? dyn_cast<StructType>(Init->getType()) : nullptr;
    if (ST && ST->hasName() && isSPIRVConstantName(ST->getName())) {
      auto BV = transConstant(Init);
      assert(BV);
      return mapValue(V, BV);
    } else if (ConstantExpr *ConstUE = dyn_cast_or_null<ConstantExpr>(Init)) {
      Instruction *Inst = ConstUE->getAsInstruction();
      if (isSpecialTypeInitializer(Inst)) {
        Init = Inst->getOperand(0);
        Ty = static_cast<PointerType *>(Init->getType());
      }
      Inst->dropAllReferences();
    }
    // Translate initializer first.
    SPIRVValue *BVarInit =
        (Init && !isa<UndefValue>(Init)) ? transValue(Init, nullptr) : nullptr;

    auto BVar = static_cast<SPIRVVariable *>(BM->addVariable(
        transType(Ty), GV->isConstant(), transLinkageType(GV), BVarInit,
        GV->getName(),
        SPIRSPIRVAddrSpaceMap::map(
            static_cast<SPIRAddressSpace>(Ty->getAddressSpace())),
        nullptr));
    mapValue(V, BVar);
    spv::BuiltIn Builtin = spv::BuiltInPosition;
    if (!GV->hasName() || !getSPIRVBuiltin(GV->getName().str(), Builtin))
      return BVar;
    BVar->setBuiltin(Builtin);
    return BVar;
  }

  if (isa<Constant>(V)) {
    auto BV = transConstant(V);
    assert(BV);
    return mapValue(V, BV);
  }

  if (auto Arg = dyn_cast<Argument>(V)) {
    unsigned ArgNo = Arg->getArgNo();
    SPIRVFunction *BF = BB->getParent();
    // assert(BF->existArgument(ArgNo));
    return mapValue(V, BF->getArgument(ArgNo));
  }

  if (CreateForward)
    return mapValue(V, BM->addForward(transType(V->getType())));

  if (StoreInst *ST = dyn_cast<StoreInst>(V)) {
    std::vector<SPIRVWord> MemoryAccess(1, 0);
    if (ST->isVolatile())
      MemoryAccess[0] |= MemoryAccessVolatileMask;
    if (ST->getAlignment()) {
      MemoryAccess[0] |= MemoryAccessAlignedMask;
      MemoryAccess.push_back(ST->getAlignment());
    }
    if (ST->getMetadata(LLVMContext::MD_nontemporal))
      MemoryAccess[0] |= MemoryAccessNontemporalMask;
    if (MemoryAccess.front() == 0)
      MemoryAccess.clear();
    return mapValue(V, BM->addStoreInst(transValue(ST->getPointerOperand(), BB),
                                        transValue(ST->getValueOperand(), BB),
                                        MemoryAccess, BB));
  }

  if (LoadInst *LD = dyn_cast<LoadInst>(V)) {
    std::vector<SPIRVWord> MemoryAccess(1, 0);
    if (LD->isVolatile())
      MemoryAccess[0] |= MemoryAccessVolatileMask;
    if (LD->getAlignment()) {
      MemoryAccess[0] |= MemoryAccessAlignedMask;
      MemoryAccess.push_back(LD->getAlignment());
    }
    if (LD->getMetadata(LLVMContext::MD_nontemporal))
      MemoryAccess[0] |= MemoryAccessNontemporalMask;
    if (MemoryAccess.front() == 0)
      MemoryAccess.clear();
    return mapValue(V, BM->addLoadInst(transValue(LD->getPointerOperand(), BB),
                                       MemoryAccess, BB));
  }

  if (BinaryOperator *B = dyn_cast<BinaryOperator>(V)) {
    SPIRVInstruction *BI = transBinaryInst(B, BB);
    return mapValue(V, BI);
  }

  if (dyn_cast<UnreachableInst>(V))
    return mapValue(V, BM->addUnreachableInst(BB));

  if (auto RI = dyn_cast<ReturnInst>(V)) {
    if (auto RV = RI->getReturnValue())
      return mapValue(V, BM->addReturnValueInst(transValue(RV, BB), BB));
    return mapValue(V, BM->addReturnInst(BB));
  }

  if (CmpInst *Cmp = dyn_cast<CmpInst>(V)) {
    SPIRVInstruction *BI = transCmpInst(Cmp, BB);
    return mapValue(V, BI);
  }

  if (SelectInst *Sel = dyn_cast<SelectInst>(V))
    return mapValue(V, BM->addSelectInst(transValue(Sel->getCondition(), BB),
                                         transValue(Sel->getTrueValue(), BB),
                                         transValue(Sel->getFalseValue(), BB),
                                         BB));

  if (AllocaInst *Alc = dyn_cast<AllocaInst>(V))
    return mapValue(
        V, BM->addVariable(transType(Alc->getType()), false,
                           SPIRVLinkageTypeKind::LinkageTypeInternal, nullptr,
                           Alc->getName(), StorageClassFunction, BB));

  if (auto *Switch = dyn_cast<SwitchInst>(V)) {
    std::vector<SPIRVSwitch::PairTy> Pairs;
    auto Select = transValue(Switch->getCondition(), BB);

    for (auto I = Switch->case_begin(), E = Switch->case_end(); I != E; ++I) {
      SPIRVSwitch::LiteralTy Lit;
      uint64_t CaseValue = I->getCaseValue()->getZExtValue();

      Lit.push_back(CaseValue);
      assert(Select->getType()->getBitWidth() <= 64 &&
             "unexpected selector bitwidth");
      if (Select->getType()->getBitWidth() == 64)
        Lit.push_back(CaseValue >> 32);

      Pairs.push_back(
          std::make_pair(Lit, static_cast<SPIRVBasicBlock *>(
                                  transValue(I->getCaseSuccessor(), nullptr))));
    }

    return mapValue(
        V, BM->addSwitchInst(Select,
                             static_cast<SPIRVBasicBlock *>(
                                 transValue(Switch->getDefaultDest(), nullptr)),
                             Pairs, BB));
  }

  if (BranchInst *Branch = dyn_cast<BranchInst>(V)) {
    SPIRVLabel *SuccessorTrue =
        static_cast<SPIRVLabel *>(transValue(Branch->getSuccessor(0), BB));

    /// Clang attaches !llvm.loop metadata to "latch" BB. This kind of blocks
    /// has an edge directed to the loop header. Thus latch BB matching to
    /// "Continue Target" per the SPIR-V spec. This statement is true only after
    /// applying the loop-simplify pass to the LLVM module.
    /// For "for" and "while" loops latch BB is terminated by an
    /// unconditional branch. Also for this kind of loops "Merge Block" can
    /// be found as block targeted by false edge of the "Header" BB.
    /// For "do while" loop the latch is terminated by a conditional branch
    /// with true edge going to the header and the false edge going out of
    /// the loop, which corresponds to a "Merge Block" per the SPIR-V spec.
    std::vector<SPIRVWord> Parameters;
    spv::LoopControlMask LoopControl = getLoopControl(Branch, Parameters);

    if (Branch->isUnconditional()) {
      // For "for" and "while" loops llvm.loop metadata is attached to
      // an unconditional branch instruction.
      if (LoopControl != spv::LoopControlMaskNone) {
        // SuccessorTrue is the loop header BB.
        const SPIRVInstruction *Term = SuccessorTrue->getTerminateInstr();
        if (Term && Term->getOpCode() == OpBranchConditional) {
          const auto *Br = static_cast<const SPIRVBranchConditional *>(Term);
          BM->addLoopMergeInst(Br->getFalseLabel()->getId(), // Merge Block
                               BB->getId(),                  // Continue Target
                               LoopControl, Parameters, SuccessorTrue);
        }
      }
      return mapValue(V, BM->addBranchInst(SuccessorTrue, BB));
    }
    // For "do-while" loops llvm.loop metadata is attached to a conditional
    // branch instructions
    SPIRVLabel *SuccessorFalse =
        static_cast<SPIRVLabel *>(transValue(Branch->getSuccessor(1), BB));
    if (LoopControl != spv::LoopControlMaskNone)
      // SuccessorTrue is the loop header BB.
      BM->addLoopMergeInst(SuccessorFalse->getId(), // Merge Block
                           BB->getId(),             // Continue Target
                           LoopControl, Parameters, SuccessorTrue);
    return mapValue(
        V, BM->addBranchConditionalInst(transValue(Branch->getCondition(), BB),
                                        SuccessorTrue, SuccessorFalse, BB));
  }

  if (auto Phi = dyn_cast<PHINode>(V)) {
    std::vector<SPIRVValue *> IncomingPairs;
    for (size_t I = 0, E = Phi->getNumIncomingValues(); I != E; ++I) {
      IncomingPairs.push_back(transValue(Phi->getIncomingValue(I), BB));
      IncomingPairs.push_back(transValue(Phi->getIncomingBlock(I), nullptr));
    }
    return mapValue(
        V, BM->addPhiInst(transType(Phi->getType()), IncomingPairs, BB));
  }

  if (auto Ext = dyn_cast<ExtractValueInst>(V)) {
    return mapValue(V, BM->addCompositeExtractInst(
                           transType(Ext->getType()),
                           transValue(Ext->getAggregateOperand(), BB),
                           Ext->getIndices(), BB));
  }

  if (auto Ins = dyn_cast<InsertValueInst>(V)) {
    return mapValue(V, BM->addCompositeInsertInst(
                           transValue(Ins->getInsertedValueOperand(), BB),
                           transValue(Ins->getAggregateOperand(), BB),
                           Ins->getIndices(), BB));
  }

  if (UnaryInstruction *U = dyn_cast<UnaryInstruction>(V)) {
    if (isSpecialTypeInitializer(U))
      return mapValue(V, transValue(U->getOperand(0), BB));
    return mapValue(V, transUnaryInst(U, BB));
  }

  if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(V)) {
    std::vector<SPIRVValue *> Indices;
    for (unsigned I = 0, E = GEP->getNumIndices(); I != E; ++I)
      Indices.push_back(transValue(GEP->getOperand(I + 1), BB));
    return mapValue(
        V, BM->addPtrAccessChainInst(transType(GEP->getType()),
                                     transValue(GEP->getPointerOperand(), BB),
                                     Indices, BB, GEP->isInBounds()));
  }

  if (auto Ext = dyn_cast<ExtractElementInst>(V)) {
    auto Index = Ext->getIndexOperand();
    if (auto Const = dyn_cast<ConstantInt>(Index))
      return mapValue(V, BM->addCompositeExtractInst(
                             transType(Ext->getType()),
                             transValue(Ext->getVectorOperand(), BB),
                             std::vector<SPIRVWord>(1, Const->getZExtValue()),
                             BB));
    else
      return mapValue(V, BM->addVectorExtractDynamicInst(
                             transValue(Ext->getVectorOperand(), BB),
                             transValue(Index, BB), BB));
  }

  if (auto Ins = dyn_cast<InsertElementInst>(V)) {
    auto Index = Ins->getOperand(2);
    if (auto Const = dyn_cast<ConstantInt>(Index))
      return mapValue(V, BM->addCompositeInsertInst(
                             transValue(Ins->getOperand(1), BB),
                             transValue(Ins->getOperand(0), BB),
                             std::vector<SPIRVWord>(1, Const->getZExtValue()),
                             BB));
    else
      return mapValue(
          V, BM->addVectorInsertDynamicInst(transValue(Ins->getOperand(0), BB),
                                            transValue(Ins->getOperand(1), BB),
                                            transValue(Index, BB), BB));
  }

  if (auto SF = dyn_cast<ShuffleVectorInst>(V)) {
    std::vector<SPIRVWord> Comp;
    for (auto &I : SF->getShuffleMask())
      Comp.push_back(I);
    return mapValue(V, BM->addVectorShuffleInst(
                           transType(SF->getType()),
                           transValue(SF->getOperand(0), BB),
                           transValue(SF->getOperand(1), BB), Comp, BB));
  }

  if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(V)) {
    SPIRVValue *BV = transIntrinsicInst(II, BB);
    return BV ? mapValue(V, BV) : nullptr;
  }

  if (CallInst *CI = dyn_cast<CallInst>(V))
    return mapValue(V, transCallInst(CI, BB));

  llvm_unreachable("Not implemented");
  return nullptr;
}

SPIRVType *LLVMToSPIRV::mapType(Type *T, SPIRVType *BT) {
  TypeMap[T] = BT;
  SPIRVDBG(dbgs() << "[mapType] " << *T << " => "; spvdbgs() << *BT << '\n');
  return BT;
}

SPIRVValue *LLVMToSPIRV::mapValue(Value *V, SPIRVValue *BV) {
  auto Loc = ValueMap.find(V);
  if (Loc != ValueMap.end()) {
    if (Loc->second == BV)
      return BV;
    assert(Loc->second->isForward() &&
           "LLVM Value is mapped to different SPIRV Values");
    auto Forward = static_cast<SPIRVForward *>(Loc->second);
    BM->replaceForward(Forward, BV);
  }
  ValueMap[V] = BV;
  SPIRVDBG(dbgs() << "[mapValue] " << *V << " => "; spvdbgs() << BV << "\n");
  return BV;
}

bool LLVMToSPIRV::transDecoration(Value *V, SPIRVValue *BV) {
  if (!transAlign(V, BV))
    return false;
  if ((isa<AtomicCmpXchgInst>(V) && cast<AtomicCmpXchgInst>(V)->isVolatile()) ||
      (isa<AtomicRMWInst>(V) && cast<AtomicRMWInst>(V)->isVolatile()))
    BV->setVolatile(true);

  if (auto BVO = dyn_cast_or_null<OverflowingBinaryOperator>(V)) {
    if (BVO->hasNoSignedWrap()) {
      BV->setNoSignedWrap(true);
    }
    if (BVO->hasNoUnsignedWrap()) {
      BV->setNoUnsignedWrap(true);
    }
  }

  return true;
}

bool LLVMToSPIRV::transAlign(Value *V, SPIRVValue *BV) {
  if (auto AL = dyn_cast<AllocaInst>(V)) {
    BM->setAlignment(BV, AL->getAlignment());
    return true;
  }
  if (auto GV = dyn_cast<GlobalVariable>(V)) {
    BM->setAlignment(BV, GV->getAlignment());
    return true;
  }
  return true;
}

/// Do this after source language is set.
bool LLVMToSPIRV::transBuiltinSet() {
  SPIRVId EISId;
  if (!BM->importBuiltinSet("OpenCL.std", &EISId))
    return false;
  if (SPIRVMDWalker(*M).getNamedMD("llvm.dbg.cu")) {
    if (!BM->importBuiltinSet("SPIRV.debug", &EISId))
      return false;
  }
  return true;
}

/// Translate sampler* spcv.cast(i32 arg) or
/// sampler* __translate_sampler_initializer(i32 arg)
/// Three cases are possible:
///   arg = ConstantInt x -> SPIRVConstantSampler
///   arg = i32 argument -> transValue(arg)
///   arg = load from sampler -> look through load
SPIRVValue *LLVMToSPIRV::oclTransSpvcCastSampler(CallInst *CI,
                                                 SPIRVBasicBlock *BB) {
  assert(CI->getCalledFunction() && "Unexpected indirect call");
  llvm::Function *F = CI->getCalledFunction();
  auto FT = F->getFunctionType();
  auto RT = FT->getReturnType();
  assert(FT->getNumParams() == 1);
  assert((isSPIRVType(RT, kSPIRVTypeName::Sampler) ||
          isPointerToOpaqueStructType(RT, kSPR2TypeName::Sampler)) &&
         FT->getParamType(0)->isIntegerTy() && "Invalid sampler type");
  auto Arg = CI->getArgOperand(0);

  auto GetSamplerConstant = [&](uint64_t SamplerValue) {
    auto AddrMode = (SamplerValue & 0xE) >> 1;
    auto Param = SamplerValue & 0x1;
    auto Filter = SamplerValue ? ((SamplerValue & 0x30) >> 4) - 1 : 0;
    auto BV = BM->addSamplerConstant(transType(RT), AddrMode, Param, Filter);
    return BV;
  };

  if (auto Const = dyn_cast<ConstantInt>(Arg)) {
    // Sampler is declared as a kernel scope constant
    return GetSamplerConstant(Const->getZExtValue());
  } else if (auto Load = dyn_cast<LoadInst>(Arg)) {
    // If value of the sampler is loaded from a global constant, use its
    // initializer for initialization of the sampler.
    auto Op = Load->getPointerOperand();
    assert(isa<GlobalVariable>(Op) && "Unknown sampler pattern!");
    auto GV = cast<GlobalVariable>(Op);
    assert(GV->isConstant() ||
           GV->getType()->getPointerAddressSpace() == SPIRAS_Constant);
    auto Initializer = GV->getInitializer();
    assert(isa<ConstantInt>(Initializer) && "sampler not constant int?");
    return GetSamplerConstant(cast<ConstantInt>(Initializer)->getZExtValue());
  }
  // Sampler is a function argument
  auto BV = transValue(Arg, BB);
  assert(BV && BV->getType() == transType(RT));
  return BV;
}

std::vector<std::pair<Decoration, std::string>>
parseAnnotations(StringRef AnnotatedCode) {
  std::vector<std::pair<Decoration, std::string>> Decorates;

  size_t OpenBracketNum = AnnotatedCode.count('{');
  size_t CloseBracketNum = AnnotatedCode.count('}');
  if (OpenBracketNum != CloseBracketNum)
    return {};

  for (size_t I = 0; I < OpenBracketNum; ++I) {
    size_t From = AnnotatedCode.find('{');
    size_t To = AnnotatedCode.find('}', From);
    StringRef AnnotatedDecoration = AnnotatedCode.substr(From + 1, To - 1);
    std::pair<StringRef, StringRef> D = AnnotatedDecoration.split(':');

    StringRef F = D.first, S = D.second;
    StringRef Value;
    Decoration Dec;
    if (F == "pump") {
      Dec = llvm::StringSwitch<Decoration>(S)
                .Case("1", DecorationSinglepumpINTEL)
                .Case("2", DecorationDoublepumpINTEL);
    } else if (F == "register") {
      Dec = DecorationRegisterINTEL;
    } else {
      Dec = llvm::StringSwitch<Decoration>(F)
                .Case("memory", DecorationMemoryINTEL)
                .Case("numbanks", DecorationNumbanksINTEL)
                .Case("bankwidth", DecorationBankwidthINTEL)
                .Case("max_concurrency", DecorationMaxconcurrencyINTEL)
                .Default(DecorationUserSemantic);
      if (Dec == DecorationUserSemantic)
        Value = AnnotatedCode.substr(From, To + 1);
      else
        Value = S;
    }

    Decorates.push_back({Dec, Value});
    AnnotatedCode = AnnotatedCode.drop_front(To + 1);
  }
  return Decorates;
}

void addIntelFPGADecorations(
    SPIRVEntry *E,
    std::vector<std::pair<Decoration, std::string>> &Decorations) {
  for (const auto &I : Decorations) {
    switch (I.first) {
    case DecorationMemoryINTEL:
      E->addDecorate(new SPIRVDecorateMemoryINTELAttr(E, I.second));
      break;
    case DecorationUserSemantic:
      E->addDecorate(new SPIRVDecorateUserSemanticAttr(E, I.second));
      break;
    case DecorationRegisterINTEL:
    case DecorationSinglepumpINTEL:
    case DecorationDoublepumpINTEL:
      assert(I.second.empty());
      E->addDecorate(I.first);
      break;
    // The rest of IntelFPGA decorations:
    // DecorationNumbanksINTEL
    // DecorationBankwidthINTEL
    // DecorationMaxconcurrencyINTEL
    default:
      SPIRVWord Result = 0;
      StringRef(I.second).getAsInteger(10, Result);
      E->addDecorate(I.first, Result);
      break;
    }
  }
}

void addIntelFPGADecorationsForStructMember(
    SPIRVEntry *E, SPIRVWord MemberNumber,
    std::vector<std::pair<Decoration, std::string>> &Decorations) {
  for (const auto &I : Decorations) {
    switch (I.first) {
    case DecorationMemoryINTEL:
      E->addMemberDecorate(
          new SPIRVMemberDecorateMemoryINTELAttr(E, MemberNumber, I.second));
      break;
    case DecorationUserSemantic:
      E->addMemberDecorate(
          new SPIRVMemberDecorateUserSemanticAttr(E, MemberNumber, I.second));
      break;
    case DecorationRegisterINTEL:
    case DecorationSinglepumpINTEL:
    case DecorationDoublepumpINTEL:
      assert(I.second.empty());
      E->addMemberDecorate(MemberNumber, I.first);
      break;
    // The rest of IntelFPGA decorations:
    // DecorationNumbanksINTEL
    // DecorationBankwidthINTEL
    // DecorationMaxconcurrencyINTEL
    default:
      SPIRVWord Result = 0;
      StringRef(I.second).getAsInteger(10, Result);
      E->addMemberDecorate(MemberNumber, I.first, Result);
      break;
    }
  }
}

SPIRVValue *LLVMToSPIRV::transIntrinsicInst(IntrinsicInst *II,
                                            SPIRVBasicBlock *BB) {
  auto GetMemoryAccess = [](MemIntrinsic *MI) -> std::vector<SPIRVWord> {
    std::vector<SPIRVWord> MemoryAccess(1, MemoryAccessMaskNone);
    if (SPIRVWord AlignVal = MI->getDestAlignment()) {
      MemoryAccess[0] |= MemoryAccessAlignedMask;
      if (auto MTI = dyn_cast<MemTransferInst>(MI)) {
        SPIRVWord SourceAlignVal = MTI->getSourceAlignment();
        assert(SourceAlignVal && "Missed Source alignment!");

        // In a case when alignment of source differs from dest one
        // least value is guaranteed anyway.
        AlignVal = std::min(AlignVal, SourceAlignVal);
      }
      MemoryAccess.push_back(AlignVal);
    }
    if (MI->isVolatile())
      MemoryAccess[0] |= MemoryAccessVolatileMask;
    return MemoryAccess;
  };

  switch (II->getIntrinsicID()) {
  case Intrinsic::fmuladd: {
    // For llvm.fmuladd.* fusion is not guaranteed. If a fused multiply-add
    // is required the corresponding llvm.fma.* intrinsic function should be
    // used instead.
    BB->getParent()->setContractedFMulAddFound();
    SPIRVType *Ty = transType(II->getType());
    SPIRVValue *Mul =
        BM->addBinaryInst(OpFMul, Ty, transValue(II->getArgOperand(0), BB),
                          transValue(II->getArgOperand(1), BB), BB);
    return BM->addBinaryInst(OpFAdd, Ty, Mul,
                             transValue(II->getArgOperand(2), BB), BB);
  }
  case Intrinsic::memset: {
    // Generally memset can't be translated with current version of SPIRV spec.
    // But in most cases it turns out that memset is emited by Clang to do
    // zero-initializtion in default constructors.
    // The code below handles only cases with val = 0 and constant len.
    MemSetInst *MSI = cast<MemSetInst>(II);
    Value *Val = MSI->getValue();
    if (!isa<Constant>(Val)) {
      assert(!"Can't translate llvm.memset with non-const `value` argument");
      return nullptr;
    }
    if (!cast<Constant>(Val)->isZeroValue()) {
      assert(!"Can't translate llvm.memset with non-zero `value` argument");
      return nullptr;
    }
    Value *Len = MSI->getLength();
    if (!isa<ConstantInt>(Len)) {
      assert(!"Can't translate llvm.memset with non-const `length` argument");
      return nullptr;
    }
    uint64_t NumElements = static_cast<ConstantInt *>(Len)->getZExtValue();
    auto *AT = ArrayType::get(Val->getType(), NumElements);
    SPIRVTypeArray *CompositeTy = static_cast<SPIRVTypeArray *>(transType(AT));
    SPIRVValue *Init = BM->addNullConstant(CompositeTy);
    SPIRVType *VarTy = transType(PointerType::get(AT, SPIRV::SPIRAS_Constant));
    SPIRVValue *Var =
        BM->addVariable(VarTy, /*isConstant*/ true, spv::LinkageTypeInternal,
                        Init, "", StorageClassUniformConstant, nullptr);
    SPIRVType *SourceTy =
        transType(PointerType::get(Val->getType(), SPIRV::SPIRAS_Constant));
    SPIRVValue *Source = BM->addUnaryInst(OpBitcast, SourceTy, Var, BB);
    SPIRVValue *Target = transValue(MSI->getRawDest(), BB);
    return BM->addCopyMemorySizedInst(Target, Source, CompositeTy->getLength(),
                                      GetMemoryAccess(MSI), BB);
  } break;
  case Intrinsic::memcpy:
    return BM->addCopyMemorySizedInst(
        transValue(II->getOperand(0), BB), transValue(II->getOperand(1), BB),
        transValue(II->getOperand(2), BB),
        GetMemoryAccess(cast<MemIntrinsic>(II)), BB);
  case Intrinsic::lifetime_start:
  case Intrinsic::lifetime_end: {
    Op OC = (II->getIntrinsicID() == Intrinsic::lifetime_start)
                ? OpLifetimeStart
                : OpLifetimeStop;
    int64_t Size = dyn_cast<ConstantInt>(II->getOperand(0))->getSExtValue();
    if (Size == -1)
      Size = 0;
    return BM->addLifetimeInst(OC, transValue(II->getOperand(1), BB), Size, BB);
  }
  // We don't want to mix translation of regular code and debug info, because
  // it creates a mess, therefore translation of debug intrinsics is
  // postponed until LLVMToSPIRVDbgTran::finalizeDebug...() methods.
  case Intrinsic::dbg_declare:
    return DbgTran->createDebugDeclarePlaceholder(cast<DbgDeclareInst>(II), BB);
  case Intrinsic::dbg_value:
    return DbgTran->createDebugValuePlaceholder(cast<DbgValueInst>(II), BB);
  case Intrinsic::var_annotation: {
    SPIRVValue *SV;
    if (auto *BI = dyn_cast<BitCastInst>(II->getArgOperand(0))) {
      SV = transValue(BI->getOperand(0), BB);
    } else {
      SV = transValue(II->getOperand(0), BB);
    }

    GetElementPtrInst *GEP = cast<GetElementPtrInst>(II->getArgOperand(1));
    Constant *C = cast<Constant>(GEP->getOperand(0));
    StringRef AnnotationString =
        cast<ConstantDataArray>(C->getOperand(0))->getAsString();

    std::vector<std::pair<Decoration, std::string>> Decorations =
        parseAnnotations(AnnotationString);
    if (Decorations.empty()) {
      SV->addDecorate(new SPIRVDecorateUserSemanticAttr(
          SV, AnnotationString.substr(0, AnnotationString.size() - 1)));
    } else {
      addIntelFPGADecorations(SV, Decorations);
    }
    return SV;
  }
  case Intrinsic::ptr_annotation: {
    GetElementPtrInst *GI;
    if (auto *BI = dyn_cast<BitCastInst>(II->getArgOperand(0))) {
      GI = dyn_cast<GetElementPtrInst>(BI->getOperand(0));
    } else {
      GI = dyn_cast<GetElementPtrInst>(II->getOperand(0));
    }
    SPIRVType *Ty = transType(GI->getSourceElementType());

    SPIRVWord MemberNumber =
        dyn_cast<ConstantInt>(GI->getOperand(2))->getZExtValue();

    GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(II->getArgOperand(1));
    Constant *C = dyn_cast<Constant>(GEP->getOperand(0));
    StringRef AnnotationString =
        dyn_cast<ConstantDataArray>(C->getOperand(0))->getAsString();

    std::vector<std::pair<Decoration, std::string>> Decorations =
        parseAnnotations(AnnotationString);

    if (Decorations.empty()) {
      Ty->addMemberDecorate(new SPIRVMemberDecorateUserSemanticAttr(
          Ty, MemberNumber,
          AnnotationString.substr(0, AnnotationString.size() - 1)));
    } else {
      addIntelFPGADecorationsForStructMember(Ty, MemberNumber, Decorations);
    }

    II->replaceAllUsesWith(II->getOperand(0));
    return 0;
  }
  default:
    // LLVM intrinsic functions shouldn't get to SPIRV, because they
    // would have no definition there.
    BM->getErrorLog().checkError(false, SPIRVEC_InvalidFunctionCall,
                                 II->getCalledValue()->getName().str(), "",
                                 __FILE__, __LINE__);
  }
  return nullptr;
}

SPIRVValue *LLVMToSPIRV::transCallInst(CallInst *CI, SPIRVBasicBlock *BB) {
  SPIRVExtInstSetKind ExtSetKind = SPIRVEIS_Count;
  SPIRVWord ExtOp = SPIRVWORD_MAX;
  llvm::Function *F = CI->getCalledFunction();
  auto MangledName = F->getName();
  std::string DemangledName;

  if (MangledName.startswith(SPCV_CAST) || MangledName == SAMPLER_INIT)
    return oclTransSpvcCastSampler(CI, BB);

  if (oclIsBuiltin(MangledName, &DemangledName) ||
      isDecoratedSPIRVFunc(F, &DemangledName))
    if (auto BV = transBuiltinToInst(DemangledName, MangledName, CI, BB))
      return BV;

  SmallVector<std::string, 2> Dec;
  if (isBuiltinTransToExtInst(CI->getCalledFunction(), &ExtSetKind, &ExtOp,
                              &Dec))
    return addDecorations(
        BM->addExtInst(
            transType(CI->getType()), BM->getExtInstSetId(ExtSetKind), ExtOp,
            transArguments(CI, BB,
                           SPIRVEntry::createUnique(ExtSetKind, ExtOp).get()),
            BB),
        Dec);

  return BM->addCallInst(
      transFunctionDecl(CI->getCalledFunction()),
      transArguments(CI, BB, SPIRVEntry::createUnique(OpFunctionCall).get()),
      BB);
}

bool LLVMToSPIRV::transAddressingMode() {
  Triple TargetTriple(M->getTargetTriple());

  SPIRVCKRT(isSupportedTriple(TargetTriple), InvalidTargetTriple,
            "Actual target triple is " + M->getTargetTriple());

  if (TargetTriple.isArch32Bit())
    BM->setAddressingModel(AddressingModelPhysical32);
  else
    BM->setAddressingModel(AddressingModelPhysical64);
  // Physical addressing model requires Addresses capability
  BM->addCapability(CapabilityAddresses);
  return true;
}
std::vector<SPIRVValue *>
LLVMToSPIRV::transValue(const std::vector<Value *> &Args, SPIRVBasicBlock *BB) {
  std::vector<SPIRVValue *> BArgs;
  for (auto &I : Args)
    BArgs.push_back(transValue(I, BB));
  return BArgs;
}

std::vector<SPIRVValue *> LLVMToSPIRV::transArguments(CallInst *CI,
                                                      SPIRVBasicBlock *BB) {
  return transValue(getArguments(CI), BB);
}

std::vector<SPIRVWord> LLVMToSPIRV::transValue(const std::vector<Value *> &Args,
                                               SPIRVBasicBlock *BB,
                                               SPIRVEntry *Entry) {
  std::vector<SPIRVWord> Operands;
  for (size_t I = 0, E = Args.size(); I != E; ++I) {
    Operands.push_back(Entry->isOperandLiteral(I)
                           ? cast<ConstantInt>(Args[I])->getZExtValue()
                           : transValue(Args[I], BB)->getId());
  }
  return Operands;
}

std::vector<SPIRVWord> LLVMToSPIRV::transArguments(CallInst *CI,
                                                   SPIRVBasicBlock *BB,
                                                   SPIRVEntry *Entry) {
  return transValue(getArguments(CI), BB, Entry);
}

SPIRVWord LLVMToSPIRV::transFunctionControlMask(CallInst *CI) {
  SPIRVWord FCM = 0;
  SPIRSPIRVFuncCtlMaskMap::foreach (
      [&](Attribute::AttrKind Attr, SPIRVFunctionControlMaskKind Mask) {
        if (CI->hasFnAttr(Attr))
          FCM |= Mask;
      });
  return FCM;
}

SPIRVWord LLVMToSPIRV::transFunctionControlMask(Function *F) {
  SPIRVWord FCM = 0;
  SPIRSPIRVFuncCtlMaskMap::foreach (
      [&](Attribute::AttrKind Attr, SPIRVFunctionControlMaskKind Mask) {
        if (F->hasFnAttribute(Attr))
          FCM |= Mask;
      });
  return FCM;
}

bool LLVMToSPIRV::transGlobalVariables() {
  for (auto I = M->global_begin(), E = M->global_end(); I != E; ++I) {
    if (!transValue(&(*I), nullptr))
      return false;
  }
  return true;
}

bool LLVMToSPIRV::isAnyFunctionReachableFromFunction(
    const Function *FS,
    const std::unordered_set<const Function *> Funcs) const {
  std::unordered_set<const Function *> Done;
  std::unordered_set<const Function *> ToDo;
  ToDo.insert(FS);

  while (!ToDo.empty()) {
    auto It = ToDo.begin();
    const Function *F = *It;

    if (Funcs.find(F) != Funcs.end())
      return true;

    ToDo.erase(It);
    Done.insert(F);

    const CallGraphNode *FN = (*CG)[F];
    for (unsigned I = 0; I < FN->size(); ++I) {
      const CallGraphNode *NN = (*FN)[I];
      const Function *NNF = NN->getFunction();
      if (!NNF)
        continue;
      if (Done.find(NNF) == Done.end()) {
        ToDo.insert(NNF);
      }
    }
  }

  return false;
}

void LLVMToSPIRV::collectInputOutputVariables(SPIRVFunction *SF, Function *F) {
  for (auto &GV : M->globals()) {
    const auto AS = GV.getAddressSpace();
    if (AS != SPIRAS_Input && AS != SPIRAS_Output)
      continue;

    std::unordered_set<const Function *> Funcs;

    for (const auto &U : GV.uses()) {
      const Instruction *Inst = dyn_cast<Instruction>(U.getUser());
      if (!Inst)
        continue;
      Funcs.insert(Inst->getFunction());
    }

    if (isAnyFunctionReachableFromFunction(F, Funcs)) {
      SF->addVariable(ValueMap[&GV]);
    }
  }
}

void LLVMToSPIRV::mutateFuncArgType(
    const std::map<unsigned, Type *> &ChangedType, Function *F) {
  for (auto &I : ChangedType) {
    for (auto UI = F->user_begin(), UE = F->user_end(); UI != UE; ++UI) {
      auto Call = dyn_cast<CallInst>(*UI);
      if (!Call)
        continue;
      auto Arg = Call->getArgOperand(I.first);
      auto OrigTy = Arg->getType();
      if (OrigTy == I.second)
        continue;
      SPIRVDBG(dbgs() << "[mutate arg type] " << *Call << ", " << *Arg << '\n');
      auto CastF = M->getOrInsertFunction(SPCV_CAST, I.second, OrigTy);
      std::vector<Value *> Args;
      Args.push_back(Arg);
      auto Cast = CallInst::Create(CastF, Args, "", Call);
      Call->replaceUsesOfWith(Arg, Cast);
      SPIRVDBG(dbgs() << "[mutate arg type] -> " << *Cast << '\n');
    }
  }
}

void LLVMToSPIRV::transFunction(Function *I) {
  SPIRVFunction *BF = transFunctionDecl(I);
  // Creating all basic blocks before creating any instruction.
  for (auto &FI : *I) {
    transValue(&FI, nullptr);
  }
  for (auto &FI : *I) {
    SPIRVBasicBlock *BB =
        static_cast<SPIRVBasicBlock *>(transValue(&FI, nullptr));
    for (auto &BI : FI) {
      transValue(&BI, BB, false);
    }
  }

  if (BF->getModule()->isEntryPoint(spv::ExecutionModelKernel, BF->getId()) &&
      BF->shouldFPContractBeDisabled()) {
    BF->addExecutionMode(BF->getModule()->add(
        new SPIRVExecutionMode(BF, spv::ExecutionModeContractionOff)));
  }
  if (BF->getModule()->isEntryPoint(spv::ExecutionModelKernel, BF->getId())) {
    collectInputOutputVariables(BF, I);
  }
}

bool LLVMToSPIRV::translate() {
  BM->setGeneratorVer(KTranslatorVer);

  if (!transSourceLanguage())
    return false;
  if (!transExtension())
    return false;
  if (!transBuiltinSet())
    return false;
  if (!transAddressingMode())
    return false;
  if (!transGlobalVariables())
    return false;

  for (auto &F : *M) {
    auto FT = F.getFunctionType();
    std::map<unsigned, Type *> ChangedType;
    oclGetMutatedArgumentTypesByBuiltin(FT, ChangedType, &F);
    mutateFuncArgType(ChangedType, &F);
  }

  // SPIR-V logical layout requires all function declarations go before
  // function definitions.
  std::vector<Function *> Decls, Defs;
  for (auto &F : *M) {
    if (isBuiltinTransToInst(&F) || isBuiltinTransToExtInst(&F) ||
        F.getName().startswith(SPCV_CAST) ||
        F.getName().startswith(LLVM_MEMCPY) ||
        F.getName().startswith(SAMPLER_INIT))
      continue;
    if (F.isDeclaration())
      Decls.push_back(&F);
    else
      Defs.push_back(&F);
  }
  for (auto I : Decls)
    transFunctionDecl(I);
  for (auto I : Defs)
    transFunction(I);

  if (!transOCLKernelMetadata())
    return false;
  if (!transExecutionMode())
    return false;

  BM->optimizeDecorates();
  BM->resolveUnknownStructFields();
  BM->createForwardPointers();
  DbgTran->transDebugMetadata();
  return true;
}

llvm::IntegerType *LLVMToSPIRV::getSizetType() {
  return IntegerType::getIntNTy(M->getContext(),
                                M->getDataLayout().getPointerSizeInBits());
}

void LLVMToSPIRV::oclGetMutatedArgumentTypesByBuiltin(
    llvm::FunctionType *FT, std::map<unsigned, Type *> &ChangedType,
    Function *F) {
  auto Name = F->getName();
  std::string Demangled;
  if (!oclIsBuiltin(Name, &Demangled))
    return;
  if (Demangled.find(kSPIRVName::SampledImage) == std::string::npos)
    return;
  if (FT->getParamType(1)->isIntegerTy())
    ChangedType[1] = getSamplerType(F->getParent());
}

SPIRVInstruction *
LLVMToSPIRV::transBuiltinToInst(const std::string &DemangledName,
                                const std::string &MangledName, CallInst *CI,
                                SPIRVBasicBlock *BB) {
  SmallVector<std::string, 2> Dec;
  auto OC = getSPIRVFuncOC(DemangledName, &Dec);

  if (OC == OpNop)
    return nullptr;

  auto Inst = transBuiltinToInstWithoutDecoration(OC, CI, BB);
  addDecorations(Inst, Dec);
  return Inst;
}

bool LLVMToSPIRV::transExecutionMode() {
  if (auto NMD = SPIRVMDWalker(*M).getNamedMD(kSPIRVMD::ExecutionMode)) {
    while (!NMD.atEnd()) {
      unsigned EMode = ~0U;
      Function *F = nullptr;
      auto N = NMD.nextOp(); /* execution mode MDNode */
      N.get(F).get(EMode);

      SPIRVFunction *BF = static_cast<SPIRVFunction *>(getTranslatedValue(F));
      assert(BF && "Invalid kernel function");
      if (!BF)
        return false;

      switch (EMode) {
      case spv::ExecutionModeContractionOff:
      case spv::ExecutionModeInitializer:
      case spv::ExecutionModeFinalizer:
        BF->addExecutionMode(BM->add(
            new SPIRVExecutionMode(BF, static_cast<ExecutionMode>(EMode))));
        break;
      case spv::ExecutionModeLocalSize:
      case spv::ExecutionModeLocalSizeHint: {
        unsigned X, Y, Z;
        N.get(X).get(Y).get(Z);
        BF->addExecutionMode(BM->add(new SPIRVExecutionMode(
            BF, static_cast<ExecutionMode>(EMode), X, Y, Z)));
      } break;
      case spv::ExecutionModeVecTypeHint:
      case spv::ExecutionModeSubgroupSize:
      case spv::ExecutionModeSubgroupsPerWorkgroup: {
        unsigned X;
        N.get(X);
        BF->addExecutionMode(BM->add(
            new SPIRVExecutionMode(BF, static_cast<ExecutionMode>(EMode), X)));
      } break;
      default:
        llvm_unreachable("invalid execution mode");
      }
    }
  }
  return true;
}

bool LLVMToSPIRV::transOCLKernelMetadata() {
  for (auto &F : *M) {
    if (F.getCallingConv() != CallingConv::SPIR_KERNEL)
      continue;

    SPIRVFunction *BF = static_cast<SPIRVFunction *>(getTranslatedValue(&F));
    assert(BF && "Kernel function should be translated first");

    // Create 'OpString' as a workaround to store information about
    // *orignal* (typedef'ed, unsigned integers) type names of kernel arguments.
    // OpString "kernel_arg_type.%kernel_name%.typename0,typename1,..."
    if (auto *KernelArgType = F.getMetadata(SPIR_MD_KERNEL_ARG_TYPE)) {
      std::string KernelArgTypesStr =
          std::string(SPIR_MD_KERNEL_ARG_TYPE) + "." + F.getName().str() + ".";
      for (const auto &TyOp : KernelArgType->operands())
        KernelArgTypesStr += cast<MDString>(TyOp)->getString().str() + ",";
      BM->getString(KernelArgTypesStr);
    }

    if (auto *KernelArgTypeQual = F.getMetadata(SPIR_MD_KERNEL_ARG_TYPE_QUAL)) {
      foreachKernelArgMD(
          KernelArgTypeQual, BF,
          [](const std::string &Str, SPIRVFunctionParameter *BA) {
            if (Str.find("volatile") != std::string::npos)
              BA->addDecorate(new SPIRVDecorate(DecorationVolatile, BA));
            if (Str.find("restrict") != std::string::npos)
              BA->addDecorate(
                  new SPIRVDecorate(DecorationFuncParamAttr, BA,
                                    FunctionParameterAttributeNoAlias));
            if (Str.find("const") != std::string::npos)
              BA->addDecorate(
                  new SPIRVDecorate(DecorationFuncParamAttr, BA,
                                    FunctionParameterAttributeNoWrite));
          });
    }
    if (auto *KernelArgName = F.getMetadata(SPIR_MD_KERNEL_ARG_NAME)) {
      foreachKernelArgMD(
          KernelArgName, BF,
          [=](const std::string &Str, SPIRVFunctionParameter *BA) {
            BM->setName(BA, Str);
          });
    }
  }
  return true;
}

bool LLVMToSPIRV::transSourceLanguage() {
  auto Src = getSPIRVSource(M);
  SrcLang = std::get<0>(Src);
  SrcLangVer = std::get<1>(Src);
  BM->setSourceLanguage(static_cast<SourceLanguage>(SrcLang), SrcLangVer);
  return true;
}

bool LLVMToSPIRV::transExtension() {
  if (auto N = SPIRVMDWalker(*M).getNamedMD(kSPIRVMD::Extension)) {
    while (!N.atEnd()) {
      std::string S;
      N.nextOp().get(S);
      assert(!S.empty() && "Invalid extension");
      BM->getExtension().insert(S);
    }
  }
  if (auto N = SPIRVMDWalker(*M).getNamedMD(kSPIRVMD::SourceExtension)) {
    while (!N.atEnd()) {
      std::string S;
      N.nextOp().get(S);
      assert(!S.empty() && "Invalid extension");
      BM->getSourceExtension().insert(S);
    }
  }
  for (auto &I :
       map<SPIRVCapabilityKind>(rmap<OclExt::Kind>(BM->getExtension())))
    BM->addCapability(I);

  return true;
}

void LLVMToSPIRV::dumpUsers(Value *V) {
  SPIRVDBG(dbgs() << "Users of " << *V << " :\n");
  for (auto UI = V->user_begin(), UE = V->user_end(); UI != UE; ++UI)
    SPIRVDBG(dbgs() << "  " << **UI << '\n');
}

Op LLVMToSPIRV::transBoolOpCode(SPIRVValue *Opn, Op OC) {
  if (!Opn->getType()->isTypeVectorOrScalarBool())
    return OC;
  IntBoolOpMap::find(OC, &OC);
  return OC;
}

SPIRVInstruction *
LLVMToSPIRV::transBuiltinToInstWithoutDecoration(Op OC, CallInst *CI,
                                                 SPIRVBasicBlock *BB) {
  if (isGroupOpCode(OC))
    BM->addCapability(CapabilityGroups);
  switch (OC) {
  case OpControlBarrier: {
    auto BArgs = transValue(getArguments(CI), BB);
    return BM->addControlBarrierInst(BArgs[0], BArgs[1], BArgs[2], BB);
  } break;
  case OpGroupAsyncCopy: {
    auto BArgs = transValue(getArguments(CI), BB);
    return BM->addAsyncGroupCopy(BArgs[0], BArgs[1], BArgs[2], BArgs[3],
                                 BArgs[4], BArgs[5], BB);
  } break;
  case OpSelect: {
    auto BArgs = transValue(getArguments(CI), BB);
    return BM->addSelectInst(BArgs[0], BArgs[1], BArgs[2], BB);
  }
  default: {
    if (isCvtOpCode(OC) && OC != OpGenericCastToPtrExplicit) {
      return BM->addUnaryInst(OC, transType(CI->getType()),
                              transValue(CI->getArgOperand(0), BB), BB);
    } else if (isCmpOpCode(OC)) {
      assert(CI && CI->getNumArgOperands() == 2 && "Invalid call inst");
      auto ResultTy = CI->getType();
      Type *BoolTy = IntegerType::getInt1Ty(M->getContext());
      auto IsVector = ResultTy->isVectorTy();
      if (IsVector)
        BoolTy = VectorType::get(BoolTy, ResultTy->getVectorNumElements());
      auto BBT = transType(BoolTy);
      auto Cmp = BM->addCmpInst(OC, BBT, transValue(CI->getArgOperand(0), BB),
                                transValue(CI->getArgOperand(1), BB), BB);
      // OpenCL C and OpenCL C++ built-ins may have different return type
      if (ResultTy == BoolTy)
        return Cmp;
      assert(IsVector || (!IsVector && ResultTy->isIntegerTy(32)));
      auto Zero = transValue(Constant::getNullValue(ResultTy), BB);
      auto One = transValue(
          IsVector ? Constant::getAllOnesValue(ResultTy) : getInt32(M, 1), BB);
      return BM->addSelectInst(Cmp, One, Zero, BB);
    } else if (isBinaryOpCode(OC)) {
      assert(CI && CI->getNumArgOperands() == 2 && "Invalid call inst");
      return BM->addBinaryInst(OC, transType(CI->getType()),
                               transValue(CI->getArgOperand(0), BB),
                               transValue(CI->getArgOperand(1), BB), BB);
    } else if (CI->getNumArgOperands() == 1 && !CI->getType()->isVoidTy() &&
               !hasExecScope(OC) && !isAtomicOpCode(OC)) {
      return BM->addUnaryInst(OC, transType(CI->getType()),
                              transValue(CI->getArgOperand(0), BB), BB);
    } else {
      auto Args = getArguments(CI);
      SPIRVType *SPRetTy = nullptr;
      Type *RetTy = CI->getType();
      auto F = CI->getCalledFunction();
      if (!RetTy->isVoidTy()) {
        SPRetTy = transType(RetTy);
      } else if (Args.size() > 0 && F->arg_begin()->hasStructRetAttr()) {
        SPRetTy = transType(F->arg_begin()->getType()->getPointerElementType());
        Args.erase(Args.begin());
      }
      auto SPI = BM->addInstTemplate(OC, BB, SPRetTy);
      std::vector<SPIRVWord> SPArgs;
      for (size_t I = 0, E = Args.size(); I != E; ++I) {
        assert((!isFunctionPointerType(Args[I]->getType()) ||
                isa<Function>(Args[I])) &&
               "Invalid function pointer argument");
        SPArgs.push_back(SPI->isOperandLiteral(I)
                             ? cast<ConstantInt>(Args[I])->getZExtValue()
                             : transValue(Args[I], BB)->getId());
      }
      SPI->setOpWordsAndValidate(SPArgs);
      if (!SPRetTy || !SPRetTy->isTypeStruct())
        return SPI;
      std::vector<SPIRVWord> Mem;
      SPIRVDBG(spvdbgs() << *SPI << '\n');
      return BM->addStoreInst(transValue(CI->getArgOperand(0), BB), SPI, Mem,
                              BB);
    }
  }
  }
  return nullptr;
}

SPIRVId LLVMToSPIRV::addInt32(int I) {
  return transValue(getInt32(M, I), nullptr, false)->getId();
}

SPIRV::SPIRVLinkageTypeKind
LLVMToSPIRV::transLinkageType(const GlobalValue *GV) {
  if (GV->isDeclarationForLinker())
    return SPIRVLinkageTypeKind::LinkageTypeImport;
  if (GV->hasInternalLinkage() || GV->hasPrivateLinkage())
    return SPIRVLinkageTypeKind::LinkageTypeInternal;
  return SPIRVLinkageTypeKind::LinkageTypeExport;
}

} // namespace SPIRV

char LLVMToSPIRV::ID = 0;

INITIALIZE_PASS_BEGIN(LLVMToSPIRV, "llvmtospv", "Translate LLVM to SPIR-V",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(OCLTypeToSPIRV)
INITIALIZE_PASS_END(LLVMToSPIRV, "llvmtospv", "Translate LLVM to SPIR-V", false,
                    false)

ModulePass *llvm::createLLVMToSPIRV(SPIRVModule *SMod) {
  return new LLVMToSPIRV(SMod);
}

void addPassesForSPIRV(legacy::PassManager &PassMgr) {
  if (SPIRVMemToReg)
    PassMgr.add(createPromoteMemoryToRegisterPass());
  PassMgr.add(createPreprocessMetadata());
  PassMgr.add(createOCL21ToSPIRV());
  PassMgr.add(createSPIRVLowerSPIRBlocks());
  PassMgr.add(createOCLTypeToSPIRV());
  PassMgr.add(createSPIRVLowerOCLBlocks());
  PassMgr.add(createOCL20ToSPIRV());
  PassMgr.add(createSPIRVRegularizeLLVM());
  PassMgr.add(createSPIRVLowerConstExpr());
  PassMgr.add(createSPIRVLowerBool());
  PassMgr.add(createSPIRVLowerMemmove());
}

bool llvm::writeSpirv(Module *M, std::ostream &OS, std::string &ErrMsg) {
  std::unique_ptr<SPIRVModule> BM(SPIRVModule::createSPIRVModule());
  legacy::PassManager PassMgr;
  addPassesForSPIRV(PassMgr);
  if (hasLoopUnrollMetadata(M))
    PassMgr.add(createLoopSimplifyPass());
  PassMgr.add(createLLVMToSPIRV(BM.get()));
  PassMgr.run(*M);

  if (BM->getError(ErrMsg) != SPIRVEC_Success)
    return false;
  OS << *BM;
  return true;
}

bool llvm::regularizeLlvmForSpirv(Module *M, std::string &ErrMsg) {
  std::unique_ptr<SPIRVModule> BM(SPIRVModule::createSPIRVModule());
  legacy::PassManager PassMgr;
  addPassesForSPIRV(PassMgr);
  PassMgr.run(*M);
  return true;
}
