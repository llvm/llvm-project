//===- NVPTXUtilities.cpp - Utility Functions -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains miscellaneous utility functions
//
//===----------------------------------------------------------------------===//

#include "NVPTXUtilities.h"
#include "NVPTX.h"
#include "NVPTXTargetMachine.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Mutex.h"
#include <cstring>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace llvm {

namespace {
typedef std::map<std::string, std::vector<unsigned>> key_val_pair_t;
typedef std::map<const GlobalValue *, key_val_pair_t> global_val_annot_t;

struct AnnotationCache {
  sys::Mutex Lock;
  std::map<const Module *, global_val_annot_t> Cache;
};

AnnotationCache &getAnnotationCache() {
  static AnnotationCache AC;
  return AC;
}
} // anonymous namespace

void clearAnnotationCache(const Module *Mod) {
  auto &AC = getAnnotationCache();
  std::lock_guard<sys::Mutex> Guard(AC.Lock);
  AC.Cache.erase(Mod);
}

static void readIntVecFromMDNode(const MDNode *MetadataNode,
                                 std::vector<unsigned> &Vec) {
  for (unsigned i = 0, e = MetadataNode->getNumOperands(); i != e; ++i) {
    ConstantInt *Val =
        mdconst::extract<ConstantInt>(MetadataNode->getOperand(i));
    Vec.push_back(Val->getZExtValue());
  }
}

static void cacheAnnotationFromMD(const MDNode *MetadataNode,
                                  key_val_pair_t &retval) {
  auto &AC = getAnnotationCache();
  std::lock_guard<sys::Mutex> Guard(AC.Lock);
  assert(MetadataNode && "Invalid mdnode for annotation");
  assert((MetadataNode->getNumOperands() % 2) == 1 &&
         "Invalid number of operands");
  // start index = 1, to skip the global variable key
  // increment = 2, to skip the value for each property-value pairs
  for (unsigned i = 1, e = MetadataNode->getNumOperands(); i != e; i += 2) {
    // property
    const MDString *prop = dyn_cast<MDString>(MetadataNode->getOperand(i));
    assert(prop && "Annotation property not a string");
    std::string Key = prop->getString().str();

    // value
    if (ConstantInt *Val = mdconst::dyn_extract<ConstantInt>(
            MetadataNode->getOperand(i + 1))) {
      retval[Key].push_back(Val->getZExtValue());
    } else if (MDNode *VecMd =
                   dyn_cast<MDNode>(MetadataNode->getOperand(i + 1))) {
      // note: only "grid_constant" annotations support vector MDNodes.
      // assert: there can only exist one unique key value pair of
      // the form (string key, MDNode node). Operands of such a node
      // shall always be unsigned ints.
      auto [It, Inserted] = retval.try_emplace(Key);
      if (Inserted) {
        readIntVecFromMDNode(VecMd, It->second);
        continue;
      }
    } else {
      llvm_unreachable("Value operand not a constant int or an mdnode");
    }
  }
}

static void cacheAnnotationFromMD(const Module *m, const GlobalValue *gv) {
  auto &AC = getAnnotationCache();
  std::lock_guard<sys::Mutex> Guard(AC.Lock);
  NamedMDNode *NMD = m->getNamedMetadata("nvvm.annotations");
  if (!NMD)
    return;
  key_val_pair_t tmp;
  for (unsigned i = 0, e = NMD->getNumOperands(); i != e; ++i) {
    const MDNode *elem = NMD->getOperand(i);

    GlobalValue *entity =
        mdconst::dyn_extract_or_null<GlobalValue>(elem->getOperand(0));
    // entity may be null due to DCE
    if (!entity)
      continue;
    if (entity != gv)
      continue;

    // accumulate annotations for entity in tmp
    cacheAnnotationFromMD(elem, tmp);
  }

  if (tmp.empty()) // no annotations for this gv
    return;

  AC.Cache[m][gv] = std::move(tmp);
}

static std::optional<unsigned> findOneNVVMAnnotation(const GlobalValue *gv,
                                                     const std::string &prop) {
  auto &AC = getAnnotationCache();
  std::lock_guard<sys::Mutex> Guard(AC.Lock);
  const Module *m = gv->getParent();
  if (AC.Cache.find(m) == AC.Cache.end())
    cacheAnnotationFromMD(m, gv);
  else if (AC.Cache[m].find(gv) == AC.Cache[m].end())
    cacheAnnotationFromMD(m, gv);
  if (AC.Cache[m][gv].find(prop) == AC.Cache[m][gv].end())
    return std::nullopt;
  return AC.Cache[m][gv][prop][0];
}

static bool findAllNVVMAnnotation(const GlobalValue *gv,
                                  const std::string &prop,
                                  std::vector<unsigned> &retval) {
  auto &AC = getAnnotationCache();
  std::lock_guard<sys::Mutex> Guard(AC.Lock);
  const Module *m = gv->getParent();
  if (AC.Cache.find(m) == AC.Cache.end())
    cacheAnnotationFromMD(m, gv);
  else if (AC.Cache[m].find(gv) == AC.Cache[m].end())
    cacheAnnotationFromMD(m, gv);
  if (AC.Cache[m][gv].find(prop) == AC.Cache[m][gv].end())
    return false;
  retval = AC.Cache[m][gv][prop];
  return true;
}

static bool globalHasNVVMAnnotation(const Value &V, const std::string &Prop) {
  if (const auto *GV = dyn_cast<GlobalValue>(&V))
    if (const auto Annot = findOneNVVMAnnotation(GV, Prop)) {
      assert((*Annot == 1) && "Unexpected annotation on a symbol");
      return true;
    }

  return false;
}

static bool argHasNVVMAnnotation(const Value &Val,
                                 const std::string &Annotation,
                                 const bool StartArgIndexAtOne = false) {
  if (const Argument *Arg = dyn_cast<Argument>(&Val)) {
    const Function *Func = Arg->getParent();
    std::vector<unsigned> Annot;
    if (findAllNVVMAnnotation(Func, Annotation, Annot)) {
      const unsigned BaseOffset = StartArgIndexAtOne ? 1 : 0;
      if (is_contained(Annot, BaseOffset + Arg->getArgNo())) {
        return true;
      }
    }
  }
  return false;
}

static std::optional<unsigned> getFnAttrParsedInt(const Function &F,
                                                  StringRef Attr) {
  return F.hasFnAttribute(Attr)
             ? std::optional(F.getFnAttributeAsParsedInteger(Attr))
             : std::nullopt;
}

bool isParamGridConstant(const Value &V) {
  if (const Argument *Arg = dyn_cast<Argument>(&V)) {
    // "grid_constant" counts argument indices starting from 1
    if (Arg->hasByValAttr() &&
        argHasNVVMAnnotation(*Arg, "grid_constant",
                             /*StartArgIndexAtOne*/ true)) {
      assert(isKernelFunction(*Arg->getParent()) &&
             "only kernel arguments can be grid_constant");
      return true;
    }
  }
  return false;
}

bool isTexture(const Value &V) { return globalHasNVVMAnnotation(V, "texture"); }

bool isSurface(const Value &V) { return globalHasNVVMAnnotation(V, "surface"); }

bool isSampler(const Value &V) {
  const char *AnnotationName = "sampler";

  return globalHasNVVMAnnotation(V, AnnotationName) ||
         argHasNVVMAnnotation(V, AnnotationName);
}

bool isImageReadOnly(const Value &V) {
  return argHasNVVMAnnotation(V, "rdoimage");
}

bool isImageWriteOnly(const Value &V) {
  return argHasNVVMAnnotation(V, "wroimage");
}

bool isImageReadWrite(const Value &V) {
  return argHasNVVMAnnotation(V, "rdwrimage");
}

bool isImage(const Value &V) {
  return isImageReadOnly(V) || isImageWriteOnly(V) || isImageReadWrite(V);
}

bool isManaged(const Value &V) { return globalHasNVVMAnnotation(V, "managed"); }

StringRef getTextureName(const Value &V) {
  assert(V.hasName() && "Found texture variable with no name");
  return V.getName();
}

StringRef getSurfaceName(const Value &V) {
  assert(V.hasName() && "Found surface variable with no name");
  return V.getName();
}

StringRef getSamplerName(const Value &V) {
  assert(V.hasName() && "Found sampler variable with no name");
  return V.getName();
}

std::optional<unsigned> getMaxNTIDx(const Function &F) {
  return findOneNVVMAnnotation(&F, "maxntidx");
}

std::optional<unsigned> getMaxNTIDy(const Function &F) {
  return findOneNVVMAnnotation(&F, "maxntidy");
}

std::optional<unsigned> getMaxNTIDz(const Function &F) {
  return findOneNVVMAnnotation(&F, "maxntidz");
}

std::optional<unsigned> getMaxNTID(const Function &F) {
  // Note: The semantics here are a bit strange. The PTX ISA states the
  // following (11.4.2. Performance-Tuning Directives: .maxntid):
  //
  //  Note that this directive guarantees that the total number of threads does
  //  not exceed the maximum, but does not guarantee that the limit in any
  //  particular dimension is not exceeded.
  std::optional<unsigned> MaxNTIDx = getMaxNTIDx(F);
  std::optional<unsigned> MaxNTIDy = getMaxNTIDy(F);
  std::optional<unsigned> MaxNTIDz = getMaxNTIDz(F);
  if (MaxNTIDx || MaxNTIDy || MaxNTIDz)
    return MaxNTIDx.value_or(1) * MaxNTIDy.value_or(1) * MaxNTIDz.value_or(1);
  return std::nullopt;
}

std::optional<unsigned> getClusterDimx(const Function &F) {
  return findOneNVVMAnnotation(&F, "cluster_dim_x");
}

std::optional<unsigned> getClusterDimy(const Function &F) {
  return findOneNVVMAnnotation(&F, "cluster_dim_y");
}

std::optional<unsigned> getClusterDimz(const Function &F) {
  return findOneNVVMAnnotation(&F, "cluster_dim_z");
}

std::optional<unsigned> getMaxClusterRank(const Function &F) {
  return getFnAttrParsedInt(F, "nvvm.maxclusterrank");
}

std::optional<unsigned> getReqNTIDx(const Function &F) {
  return findOneNVVMAnnotation(&F, "reqntidx");
}

std::optional<unsigned> getReqNTIDy(const Function &F) {
  return findOneNVVMAnnotation(&F, "reqntidy");
}

std::optional<unsigned> getReqNTIDz(const Function &F) {
  return findOneNVVMAnnotation(&F, "reqntidz");
}

std::optional<unsigned> getReqNTID(const Function &F) {
  // Note: The semantics here are a bit strange. See getMaxNTID.
  std::optional<unsigned> ReqNTIDx = getReqNTIDx(F);
  std::optional<unsigned> ReqNTIDy = getReqNTIDy(F);
  std::optional<unsigned> ReqNTIDz = getReqNTIDz(F);
  if (ReqNTIDx || ReqNTIDy || ReqNTIDz)
    return ReqNTIDx.value_or(1) * ReqNTIDy.value_or(1) * ReqNTIDz.value_or(1);
  return std::nullopt;
}

std::optional<unsigned> getMinCTASm(const Function &F) {
  return getFnAttrParsedInt(F, "nvvm.minctasm");
}

std::optional<unsigned> getMaxNReg(const Function &F) {
  return getFnAttrParsedInt(F, "nvvm.maxnreg");
}

MaybeAlign getAlign(const Function &F, unsigned Index) {
  // First check the alignstack metadata
  if (MaybeAlign StackAlign =
          F.getAttributes().getAttributes(Index).getStackAlignment())
    return StackAlign;

  // check the legacy nvvm metadata only for the return value since llvm does
  // not support stackalign attribute for this.
  if (Index == 0) {
    std::vector<unsigned> Vs;
    if (findAllNVVMAnnotation(&F, "align", Vs))
      for (unsigned V : Vs)
        if ((V >> 16) == Index)
          return Align(V & 0xFFFF);
  }

  return std::nullopt;
}

MaybeAlign getAlign(const CallInst &I, unsigned Index) {
  // First check the alignstack metadata
  if (MaybeAlign StackAlign =
          I.getAttributes().getAttributes(Index).getStackAlignment())
    return StackAlign;

  // If that is missing, check the legacy nvvm metadata
  if (MDNode *alignNode = I.getMetadata("callalign")) {
    for (int i = 0, n = alignNode->getNumOperands(); i < n; i++) {
      if (const ConstantInt *CI =
              mdconst::dyn_extract<ConstantInt>(alignNode->getOperand(i))) {
        unsigned V = CI->getZExtValue();
        if ((V >> 16) == Index)
          return Align(V & 0xFFFF);
        if ((V >> 16) > Index)
          return std::nullopt;
      }
    }
  }
  return std::nullopt;
}

Function *getMaybeBitcastedCallee(const CallBase *CB) {
  return dyn_cast<Function>(CB->getCalledOperand()->stripPointerCasts());
}

bool shouldEmitPTXNoReturn(const Value *V, const TargetMachine &TM) {
  const auto &ST =
      *static_cast<const NVPTXTargetMachine &>(TM).getSubtargetImpl();
  if (!ST.hasNoReturn())
    return false;

  assert((isa<Function>(V) || isa<CallInst>(V)) &&
         "Expect either a call instruction or a function");

  if (const CallInst *CallI = dyn_cast<CallInst>(V))
    return CallI->doesNotReturn() &&
           CallI->getFunctionType()->getReturnType()->isVoidTy();

  const Function *F = cast<Function>(V);
  return F->doesNotReturn() &&
         F->getFunctionType()->getReturnType()->isVoidTy() &&
         !isKernelFunction(*F);
}

bool Isv2x16VT(EVT VT) {
  return (VT == MVT::v2f16 || VT == MVT::v2bf16 || VT == MVT::v2i16);
}

} // namespace llvm
