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
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/DXILABI.h"

namespace llvm {
class CallInst;
class MDTuple;
class TargetExtType;

namespace dxil {

class ResourceInfo {
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
  };

  struct TypedInfo {
    dxil::ElementType ElementTy;
    uint32_t ElementCount;

    bool operator==(const TypedInfo &RHS) const {
      return std::tie(ElementTy, ElementCount) ==
             std::tie(RHS.ElementTy, RHS.ElementCount);
    }
    bool operator!=(const TypedInfo &RHS) const { return !(*this == RHS); }
  };

  struct MSInfo {
    uint32_t Count;

    bool operator==(const MSInfo &RHS) const { return Count == RHS.Count; }
    bool operator!=(const MSInfo &RHS) const { return !(*this == RHS); }
  };

  struct FeedbackInfo {
    dxil::SamplerFeedbackType Type;

    bool operator==(const FeedbackInfo &RHS) const { return Type == RHS.Type; }
    bool operator!=(const FeedbackInfo &RHS) const { return !(*this == RHS); }
  };

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
  void setUAV(bool GloballyCoherent, bool HasCounter, bool IsROV) {
    assert(isUAV() && "Not a UAV");
    UAVFlags.GloballyCoherent = GloballyCoherent;
    UAVFlags.HasCounter = HasCounter;
    UAVFlags.IsROV = IsROV;
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
  void setFeedback(dxil::SamplerFeedbackType Type) {
    assert(isFeedback() && "Not Feedback");
    Feedback.Type = Type;
  }
  void setMultiSample(uint32_t Count) {
    assert(isMultiSample() && "Not MultiSampled");
    MultiSample.Count = Count;
  }

  bool operator==(const ResourceInfo &RHS) const;

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

  ResourceBinding getBinding() const { return Binding; }
  std::pair<uint32_t, uint32_t> getAnnotateProps() const;

  void print(raw_ostream &OS) const;
};

} // namespace dxil

using DXILResourceMap = MapVector<CallInst *, dxil::ResourceInfo>;

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
