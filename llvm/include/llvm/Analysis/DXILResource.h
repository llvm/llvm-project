//===- DXILResource.h - Representations of DXIL resources -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DXILRESOURCE_H
#define LLVM_ANALYSIS_DXILRESOURCE_H

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/DXILABI.h"

namespace llvm {
class CallInst;
class LLVMContext;
class MDTuple;
class Value;

namespace dxil {

class ResourceInfo {
public:
  struct ResourceBinding {
    uint32_t RecordID;
    uint32_t Space;
    uint32_t LowerBound;
    uint32_t Size;

    bool operator==(const ResourceBinding &RHS) const {
      return std::tie(RecordID, Space, LowerBound, Size) ==
             std::tie(RHS.RecordID, RHS.Space, RHS.LowerBound, RHS.Size);
    }
    bool operator!=(const ResourceBinding &RHS) const {
      return !(*this == RHS);
    }
    bool operator<(const ResourceBinding &RHS) const {
      return std::tie(RecordID, Space, LowerBound, Size) <
             std::tie(RHS.RecordID, RHS.Space, RHS.LowerBound, RHS.Size);
    }
  };

  struct UAVInfo {
    bool GloballyCoherent;
    bool HasCounter;
    bool IsROV;

    bool operator==(const UAVInfo &RHS) const {
      return std::tie(GloballyCoherent, HasCounter, IsROV) ==
             std::tie(RHS.GloballyCoherent, RHS.HasCounter, RHS.IsROV);
    }
    bool operator!=(const UAVInfo &RHS) const { return !(*this == RHS); }
    bool operator<(const UAVInfo &RHS) const {
      return std::tie(GloballyCoherent, HasCounter, IsROV) <
             std::tie(RHS.GloballyCoherent, RHS.HasCounter, RHS.IsROV);
    }
  };

  struct StructInfo {
    uint32_t Stride;
    // Note: we store an integer here rather than using `MaybeAlign` because in
    // GCC 7 MaybeAlign isn't trivial so having one in this union would delete
    // our move constructor.
    // See https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p0602r4.html
    uint32_t AlignLog2;

    bool operator==(const StructInfo &RHS) const {
      return std::tie(Stride, AlignLog2) == std::tie(RHS.Stride, RHS.AlignLog2);
    }
    bool operator!=(const StructInfo &RHS) const { return !(*this == RHS); }
    bool operator<(const StructInfo &RHS) const {
      return std::tie(Stride, AlignLog2) < std::tie(RHS.Stride, RHS.AlignLog2);
    }
  };

  struct TypedInfo {
    dxil::ElementType ElementTy;
    uint32_t ElementCount;

    bool operator==(const TypedInfo &RHS) const {
      return std::tie(ElementTy, ElementCount) ==
             std::tie(RHS.ElementTy, RHS.ElementCount);
    }
    bool operator!=(const TypedInfo &RHS) const { return !(*this == RHS); }
    bool operator<(const TypedInfo &RHS) const {
      return std::tie(ElementTy, ElementCount) <
             std::tie(RHS.ElementTy, RHS.ElementCount);
    }
  };

  struct MSInfo {
    uint32_t Count;

    bool operator==(const MSInfo &RHS) const { return Count == RHS.Count; }
    bool operator!=(const MSInfo &RHS) const { return !(*this == RHS); }
    bool operator<(const MSInfo &RHS) const { return Count < RHS.Count; }
  };

  struct FeedbackInfo {
    dxil::SamplerFeedbackType Type;

    bool operator==(const FeedbackInfo &RHS) const { return Type == RHS.Type; }
    bool operator!=(const FeedbackInfo &RHS) const { return !(*this == RHS); }
    bool operator<(const FeedbackInfo &RHS) const { return Type < RHS.Type; }
  };

private:
  // Universal properties.
  Value *Symbol;
  StringRef Name;

  dxil::ResourceClass RC;
  dxil::ResourceKind Kind;

  ResourceBinding Binding = {};

  // Resource class dependent properties.
  // CBuffer, Sampler, and RawBuffer end here.
  union {
    UAVInfo UAVFlags;            // UAV
    uint32_t CBufferSize;        // CBuffer
    dxil::SamplerType SamplerTy; // Sampler
  };

  // Resource kind dependent properties.
  union {
    StructInfo Struct;     // StructuredBuffer
    TypedInfo Typed;       // All SRV/UAV except Raw/StructuredBuffer
    FeedbackInfo Feedback; // FeedbackTexture
  };

  MSInfo MultiSample;

public:
  ResourceInfo(dxil::ResourceClass RC, dxil::ResourceKind Kind, Value *Symbol,
               StringRef Name)
      : Symbol(Symbol), Name(Name), RC(RC), Kind(Kind) {}

  // Conditions to check before accessing union members.
  bool isUAV() const;
  bool isCBuffer() const;
  bool isSampler() const;
  bool isStruct() const;
  bool isTyped() const;
  bool isFeedback() const;
  bool isMultiSample() const;

  void bind(uint32_t RecordID, uint32_t Space, uint32_t LowerBound,
            uint32_t Size) {
    Binding.RecordID = RecordID;
    Binding.Space = Space;
    Binding.LowerBound = LowerBound;
    Binding.Size = Size;
  }
  const ResourceBinding &getBinding() const { return Binding; }
  void setUAV(bool GloballyCoherent, bool HasCounter, bool IsROV) {
    assert(isUAV() && "Not a UAV");
    UAVFlags.GloballyCoherent = GloballyCoherent;
    UAVFlags.HasCounter = HasCounter;
    UAVFlags.IsROV = IsROV;
  }
  const UAVInfo &getUAV() const {
    assert(isUAV() && "Not a UAV");
    return UAVFlags;
  }
  void setCBuffer(uint32_t Size) {
    assert(isCBuffer() && "Not a CBuffer");
    CBufferSize = Size;
  }
  void setSampler(dxil::SamplerType Ty) { SamplerTy = Ty; }
  void setStruct(uint32_t Stride, MaybeAlign Alignment) {
    assert(isStruct() && "Not a Struct");
    Struct.Stride = Stride;
    Struct.AlignLog2 = Alignment ? Log2(*Alignment) : 0;
  }
  void setTyped(dxil::ElementType ElementTy, uint32_t ElementCount) {
    assert(isTyped() && "Not Typed");
    Typed.ElementTy = ElementTy;
    Typed.ElementCount = ElementCount;
  }
  const TypedInfo &getTyped() const {
    assert(isTyped() && "Not typed");
    return Typed;
  }
  void setFeedback(dxil::SamplerFeedbackType Type) {
    assert(isFeedback() && "Not Feedback");
    Feedback.Type = Type;
  }
  void setMultiSample(uint32_t Count) {
    assert(isMultiSample() && "Not MultiSampled");
    MultiSample.Count = Count;
  }
  const MSInfo &getMultiSample() const {
    assert(isMultiSample() && "Not MultiSampled");
    return MultiSample;
  }

  StringRef getName() const { return Name; }
  dxil::ResourceClass getResourceClass() const { return RC; }
  dxil::ResourceKind getResourceKind() const { return Kind; }

  bool operator==(const ResourceInfo &RHS) const;
  bool operator!=(const ResourceInfo &RHS) const { return !(*this == RHS); }
  bool operator<(const ResourceInfo &RHS) const;

  static ResourceInfo SRV(Value *Symbol, StringRef Name,
                          dxil::ElementType ElementTy, uint32_t ElementCount,
                          dxil::ResourceKind Kind);
  static ResourceInfo RawBuffer(Value *Symbol, StringRef Name);
  static ResourceInfo StructuredBuffer(Value *Symbol, StringRef Name,
                                       uint32_t Stride, MaybeAlign Alignment);
  static ResourceInfo Texture2DMS(Value *Symbol, StringRef Name,
                                  dxil::ElementType ElementTy,
                                  uint32_t ElementCount, uint32_t SampleCount);
  static ResourceInfo Texture2DMSArray(Value *Symbol, StringRef Name,
                                       dxil::ElementType ElementTy,
                                       uint32_t ElementCount,
                                       uint32_t SampleCount);

  static ResourceInfo UAV(Value *Symbol, StringRef Name,
                          dxil::ElementType ElementTy, uint32_t ElementCount,
                          bool GloballyCoherent, bool IsROV,
                          dxil::ResourceKind Kind);
  static ResourceInfo RWRawBuffer(Value *Symbol, StringRef Name,
                                  bool GloballyCoherent, bool IsROV);
  static ResourceInfo RWStructuredBuffer(Value *Symbol, StringRef Name,
                                         uint32_t Stride, MaybeAlign Alignment,
                                         bool GloballyCoherent, bool IsROV,
                                         bool HasCounter);
  static ResourceInfo RWTexture2DMS(Value *Symbol, StringRef Name,
                                    dxil::ElementType ElementTy,
                                    uint32_t ElementCount, uint32_t SampleCount,
                                    bool GloballyCoherent);
  static ResourceInfo RWTexture2DMSArray(Value *Symbol, StringRef Name,
                                         dxil::ElementType ElementTy,
                                         uint32_t ElementCount,
                                         uint32_t SampleCount,
                                         bool GloballyCoherent);
  static ResourceInfo FeedbackTexture2D(Value *Symbol, StringRef Name,
                                        dxil::SamplerFeedbackType FeedbackTy);
  static ResourceInfo
  FeedbackTexture2DArray(Value *Symbol, StringRef Name,
                         dxil::SamplerFeedbackType FeedbackTy);

  static ResourceInfo CBuffer(Value *Symbol, StringRef Name, uint32_t Size);

  static ResourceInfo Sampler(Value *Symbol, StringRef Name,
                              dxil::SamplerType SamplerTy);

  MDTuple *getAsMetadata(LLVMContext &Ctx) const;

  std::pair<uint32_t, uint32_t> getAnnotateProps() const;

  void print(raw_ostream &OS) const;
};

} // namespace dxil

class DXILResourceMap {
  SmallVector<dxil::ResourceInfo> Resources;
  DenseMap<CallInst *, unsigned> CallMap;
  unsigned FirstUAV = 0;
  unsigned FirstCBuffer = 0;
  unsigned FirstSampler = 0;

public:
  using iterator = SmallVector<dxil::ResourceInfo>::iterator;
  using const_iterator = SmallVector<dxil::ResourceInfo>::const_iterator;

  DXILResourceMap(
      SmallVectorImpl<std::pair<CallInst *, dxil::ResourceInfo>> &&CIToRI);

  iterator begin() { return Resources.begin(); }
  const_iterator begin() const { return Resources.begin(); }
  iterator end() { return Resources.end(); }
  const_iterator end() const { return Resources.end(); }

  bool empty() const { return Resources.empty(); }

  iterator find(const CallInst *Key) {
    auto Pos = CallMap.find(Key);
    return Pos == CallMap.end() ? Resources.end()
                                : (Resources.begin() + Pos->second);
  }

  const_iterator find(const CallInst *Key) const {
    auto Pos = CallMap.find(Key);
    return Pos == CallMap.end() ? Resources.end()
                                : (Resources.begin() + Pos->second);
  }

  iterator srv_begin() { return begin(); }
  const_iterator srv_begin() const { return begin(); }
  iterator srv_end() { return begin() + FirstUAV; }
  const_iterator srv_end() const { return begin() + FirstUAV; }
  iterator_range<iterator> srvs() { return make_range(srv_begin(), srv_end()); }
  iterator_range<const_iterator> srvs() const {
    return make_range(srv_begin(), srv_end());
  }

  iterator uav_begin() { return begin() + FirstUAV; }
  const_iterator uav_begin() const { return begin() + FirstUAV; }
  iterator uav_end() { return begin() + FirstCBuffer; }
  const_iterator uav_end() const { return begin() + FirstCBuffer; }
  iterator_range<iterator> uavs() { return make_range(uav_begin(), uav_end()); }
  iterator_range<const_iterator> uavs() const {
    return make_range(uav_begin(), uav_end());
  }

  iterator cbuffer_begin() { return begin() + FirstCBuffer; }
  const_iterator cbuffer_begin() const { return begin() + FirstCBuffer; }
  iterator cbuffer_end() { return begin() + FirstSampler; }
  const_iterator cbuffer_end() const { return begin() + FirstSampler; }
  iterator_range<iterator> cbuffers() {
    return make_range(cbuffer_begin(), cbuffer_end());
  }
  iterator_range<const_iterator> cbuffers() const {
    return make_range(cbuffer_begin(), cbuffer_end());
  }

  iterator sampler_begin() { return begin() + FirstSampler; }
  const_iterator sampler_begin() const { return begin() + FirstSampler; }
  iterator sampler_end() { return end(); }
  const_iterator sampler_end() const { return end(); }
  iterator_range<iterator> samplers() {
    return make_range(sampler_begin(), sampler_end());
  }
  iterator_range<const_iterator> samplers() const {
    return make_range(sampler_begin(), sampler_end());
  }

  void print(raw_ostream &OS) const;
};

class DXILResourceAnalysis : public AnalysisInfoMixin<DXILResourceAnalysis> {
  friend AnalysisInfoMixin<DXILResourceAnalysis>;

  static AnalysisKey Key;

public:
  using Result = DXILResourceMap;

  /// Gather resource info for the module \c M.
  DXILResourceMap run(Module &M, ModuleAnalysisManager &AM);
};

/// Printer pass for the \c DXILResourceAnalysis results.
class DXILResourcePrinterPass : public PassInfoMixin<DXILResourcePrinterPass> {
  raw_ostream &OS;

public:
  explicit DXILResourcePrinterPass(raw_ostream &OS) : OS(OS) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

  static bool isRequired() { return true; }
};

class DXILResourceWrapperPass : public ModulePass {
  std::unique_ptr<DXILResourceMap> ResourceMap;

public:
  static char ID; // Class identification, replacement for typeinfo

  DXILResourceWrapperPass();
  ~DXILResourceWrapperPass() override;

  const DXILResourceMap &getResourceMap() const { return *ResourceMap; }
  DXILResourceMap &getResourceMap() { return *ResourceMap; }

  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnModule(Module &M) override;
  void releaseMemory() override;

  void print(raw_ostream &OS, const Module *M) const override;
  void dump() const;
};

ModulePass *createDXILResourceWrapperPassPass();

} // namespace llvm

#endif // LLVM_ANALYSIS_DXILRESOURCE_H
