//===- NVVMProperties.cpp - NVVM annotation utilities ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains NVVM attribute and metadata query utilities.
//
//===----------------------------------------------------------------------===//

#include "NVVMProperties.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ModRef.h"
#include "llvm/Support/Mutex.h"
#include <functional>
#include <map>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>

namespace llvm {

namespace {
using AnnotationValues = std::map<std::string, std::vector<unsigned>>;
using AnnotationMap = std::map<const GlobalValue *, AnnotationValues>;

struct AnnotationCache {
  sys::Mutex Lock;
  std::map<const Module *, AnnotationMap> Cache;
};

AnnotationCache &getAnnotationCache() {
  static AnnotationCache AC;
  return AC;
}
} // namespace

void clearAnnotationCache(const Module *Mod) {
  auto &AC = getAnnotationCache();
  std::lock_guard<sys::Mutex> Guard(AC.Lock);
  AC.Cache.erase(Mod);
}

static void cacheAnnotationFromMD(const MDNode *MetadataNode,
                                  AnnotationValues &RetVal) {
  auto &AC = getAnnotationCache();
  std::lock_guard<sys::Mutex> Guard(AC.Lock);
  assert(MetadataNode && "Invalid mdnode for annotation");
  assert((MetadataNode->getNumOperands() % 2) == 1 &&
         "Invalid number of operands");
  // start index = 1, to skip the global variable key
  // increment = 2, to skip the value for each property-value pairs
  for (unsigned I = 1, E = MetadataNode->getNumOperands(); I != E; I += 2) {
    const MDString *Prop = dyn_cast<MDString>(MetadataNode->getOperand(I));
    assert(Prop && "Annotation property not a string");
    std::string Key = Prop->getString().str();

    if (ConstantInt *Val = mdconst::dyn_extract<ConstantInt>(
            MetadataNode->getOperand(I + 1))) {
      RetVal[Key].push_back(Val->getZExtValue());
    } else {
      llvm_unreachable("Value operand not a constant int");
    }
  }
}

static void cacheAnnotationFromMD(const Module *M, const GlobalValue *GV) {
  auto &AC = getAnnotationCache();
  std::lock_guard<sys::Mutex> Guard(AC.Lock);
  NamedMDNode *NMD = M->getNamedMetadata("nvvm.annotations");
  if (!NMD)
    return;

  AnnotationValues Tmp;
  for (unsigned I = 0, E = NMD->getNumOperands(); I != E; ++I) {
    const MDNode *Elem = NMD->getOperand(I);
    GlobalValue *Entity =
        mdconst::dyn_extract_or_null<GlobalValue>(Elem->getOperand(0));
    // entity may be null due to DCE
    if (!Entity || Entity != GV)
      continue;

    cacheAnnotationFromMD(Elem, Tmp);
  }

  if (Tmp.empty())
    return;

  AC.Cache[M][GV] = std::move(Tmp);
}

static std::optional<unsigned> findOneNVVMAnnotation(const GlobalValue *GV,
                                                     const std::string &Prop) {
  auto &AC = getAnnotationCache();
  std::lock_guard<sys::Mutex> Guard(AC.Lock);
  const Module *M = GV->getParent();
  auto ACIt = AC.Cache.find(M);
  if (ACIt == AC.Cache.end())
    cacheAnnotationFromMD(M, GV);
  else if (ACIt->second.find(GV) == ACIt->second.end())
    cacheAnnotationFromMD(M, GV);

  auto &KVP = AC.Cache[M][GV];
  auto It = KVP.find(Prop);
  if (It == KVP.end())
    return std::nullopt;
  return It->second[0];
}

static bool findAllNVVMAnnotation(const GlobalValue *GV,
                                  const std::string &Prop,
                                  std::vector<unsigned> &RetVal) {
  auto &AC = getAnnotationCache();
  std::lock_guard<sys::Mutex> Guard(AC.Lock);
  const Module *M = GV->getParent();
  auto ACIt = AC.Cache.find(M);
  if (ACIt == AC.Cache.end())
    cacheAnnotationFromMD(M, GV);
  else if (ACIt->second.find(GV) == ACIt->second.end())
    cacheAnnotationFromMD(M, GV);

  auto &KVP = AC.Cache[M][GV];
  auto It = KVP.find(Prop);
  if (It == KVP.end())
    return false;
  RetVal = It->second;
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
  if (const auto *Arg = dyn_cast<Argument>(&Val)) {
    std::vector<unsigned> Annot;
    if (findAllNVVMAnnotation(Arg->getParent(), Annotation, Annot) &&
        is_contained(Annot, Arg->getArgNo()))
      return true;
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

  return std::accumulate(V.begin(), V.end(), uint64_t(1),
                         std::multiplies<uint64_t>{});
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
  return getVectorProduct(getMaxNTID(F));
}

std::optional<uint64_t> getOverallReqNTID(const Function &F) {
  // Note: The semantics here are a bit strange. See getOverallMaxNTID.
  return getVectorProduct(getReqNTID(F));
}

std::optional<uint64_t> getOverallClusterRank(const Function &F) {
  // maxclusterrank and cluster_dim are mutually exclusive.
  if (const auto ClusterRank = getMaxClusterRank(F))
    return ClusterRank;

  // Note: The semantics here are a bit strange. See getOverallMaxNTID.
  return getVectorProduct(getClusterDim(F));
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
  return Arg.hasAttribute("nvvm.grid_constant");
}

MaybeAlign getAlign(const CallInst &I, unsigned Index) {
  // First check the alignstack metadata.
  if (MaybeAlign StackAlign =
          I.getAttributes().getAttributes(Index).getStackAlignment())
    return StackAlign;

  // If that is missing, check the legacy nvvm metadata.
  if (MDNode *AlignNode = I.getMetadata("callalign")) {
    for (int I = 0, N = AlignNode->getNumOperands(); I < N; I++) {
      if (const auto *CI =
              mdconst::dyn_extract<ConstantInt>(AlignNode->getOperand(I))) {
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

} // namespace llvm
