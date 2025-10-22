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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/ModRef.h"
#include "llvm/Support/Mutex.h"
#include <cstdint>
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
    } else {
      llvm_unreachable("Value operand not a constant int");
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
  auto ACIt = AC.Cache.find(m);
  if (ACIt == AC.Cache.end())
    cacheAnnotationFromMD(m, gv);
  else if (ACIt->second.find(gv) == ACIt->second.end())
    cacheAnnotationFromMD(m, gv);
  // Look up AC.Cache[m][gv] again because cacheAnnotationFromMD may have
  // inserted the entry.
  auto &KVP = AC.Cache[m][gv];
  auto It = KVP.find(prop);
  if (It == KVP.end())
    return std::nullopt;
  return It->second[0];
}

static bool findAllNVVMAnnotation(const GlobalValue *gv,
                                  const std::string &prop,
                                  std::vector<unsigned> &retval) {
  auto &AC = getAnnotationCache();
  std::lock_guard<sys::Mutex> Guard(AC.Lock);
  const Module *m = gv->getParent();
  auto ACIt = AC.Cache.find(m);
  if (ACIt == AC.Cache.end())
    cacheAnnotationFromMD(m, gv);
  else if (ACIt->second.find(gv) == ACIt->second.end())
    cacheAnnotationFromMD(m, gv);
  // Look up AC.Cache[m][gv] again because cacheAnnotationFromMD may have
  // inserted the entry.
  auto &KVP = AC.Cache[m][gv];
  auto It = KVP.find(prop);
  if (It == KVP.end())
    return false;
  retval = It->second;
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
                                 const std::string &Annotation) {
  if (const Argument *Arg = dyn_cast<Argument>(&Val)) {
    const Function *Func = Arg->getParent();
    std::vector<unsigned> Annot;
    if (findAllNVVMAnnotation(Func, Annotation, Annot)) {
      if (is_contained(Annot, Arg->getArgNo()))
        return true;
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

static SmallVector<unsigned, 3> getFnAttrParsedVector(const Function &F,
                                                      StringRef Attr) {
  SmallVector<unsigned, 3> V;
  auto &Ctx = F.getContext();

  if (F.hasFnAttribute(Attr)) {
    // We expect the attribute value to be of the form "x[,y[,z]]", where x, y,
    // and z are unsigned values.
    StringRef S = F.getFnAttribute(Attr).getValueAsString();
    for (unsigned I = 0; I < 3 && !S.empty(); I++) {
      auto [First, Rest] = S.split(",");
      unsigned IntVal;
      if (First.trim().getAsInteger(0, IntVal))
        Ctx.emitError("can't parse integer attribute " + First + " in " + Attr);

      V.push_back(IntVal);
      S = Rest;
    }
  }
  return V;
}

static std::optional<uint64_t> getVectorProduct(ArrayRef<unsigned> V) {
  if (V.empty())
    return std::nullopt;

  return std::accumulate(V.begin(), V.end(), 1, std::multiplies<uint64_t>{});
}

bool isParamGridConstant(const Argument &Arg) {
  assert(isKernelFunction(*Arg.getParent()) &&
         "only kernel arguments can be grid_constant");

  if (!Arg.hasByValAttr())
    return false;

  // Lowering an argument as a grid_constant violates the byval semantics (and
  // the C++ API) by reusing the same memory location for the argument across
  // multiple threads. If an argument doesn't read memory and its address is not
  // captured (its address is not compared with any value), then the tweak of
  // the C++ API and byval semantics is unobservable by the program and we can
  // lower the arg as a grid_constant.
  if (Arg.onlyReadsMemory()) {
    const auto CI = Arg.getAttributes().getCaptureInfo();
    if (!capturesAddress(CI) && !capturesFullProvenance(CI))
      return true;
  }

  // "grid_constant" counts argument indices starting from 1
  if (Arg.hasAttribute("nvvm.grid_constant"))
    return true;

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

SmallVector<unsigned, 3> getMaxNTID(const Function &F) {
  return getFnAttrParsedVector(F, "nvvm.maxntid");
}

SmallVector<unsigned, 3> getReqNTID(const Function &F) {
  return getFnAttrParsedVector(F, "nvvm.reqntid");
}

SmallVector<unsigned, 3> getClusterDim(const Function &F) {
  return getFnAttrParsedVector(F, "nvvm.cluster_dim");
}

std::optional<uint64_t> getOverallMaxNTID(const Function &F) {
  // Note: The semantics here are a bit strange. The PTX ISA states the
  // following (11.4.2. Performance-Tuning Directives: .maxntid):
  //
  //  Note that this directive guarantees that the total number of threads does
  //  not exceed the maximum, but does not guarantee that the limit in any
  //  particular dimension is not exceeded.
  const auto MaxNTID = getMaxNTID(F);
  return getVectorProduct(MaxNTID);
}

std::optional<uint64_t> getOverallReqNTID(const Function &F) {
  // Note: The semantics here are a bit strange. See getMaxNTID.
  const auto ReqNTID = getReqNTID(F);
  return getVectorProduct(ReqNTID);
}

std::optional<uint64_t> getOverallClusterRank(const Function &F) {
  // maxclusterrank and cluster_dim are mutually exclusive.
  if (const auto ClusterRank = getMaxClusterRank(F))
    return ClusterRank;

  // Note: The semantics here are a bit strange. See getMaxNTID.
  const auto ClusterDim = getClusterDim(F);
  return getVectorProduct(ClusterDim);
}

std::optional<unsigned> getMaxClusterRank(const Function &F) {
  return getFnAttrParsedInt(F, "nvvm.maxclusterrank");
}

std::optional<unsigned> getMinCTASm(const Function &F) {
  return getFnAttrParsedInt(F, "nvvm.minctasm");
}

std::optional<unsigned> getMaxNReg(const Function &F) {
  return getFnAttrParsedInt(F, "nvvm.maxnreg");
}

bool hasBlocksAreClusters(const Function &F) {
  return F.hasFnAttribute("nvvm.blocksareclusters");
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

} // namespace llvm
