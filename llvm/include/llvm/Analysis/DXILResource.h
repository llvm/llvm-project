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
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/DXILABI.h"

namespace llvm {
class CallInst;
class DataLayout;
class LLVMContext;
class MDTuple;
class Value;

class DXILResourceTypeMap;

namespace dxil {

/// The dx.RawBuffer target extension type
///
/// `target("dx.RawBuffer", Type, IsWriteable, IsROV)`
class RawBufferExtType : public TargetExtType {
public:
  RawBufferExtType() = delete;
  RawBufferExtType(const RawBufferExtType &) = delete;
  RawBufferExtType &operator=(const RawBufferExtType &) = delete;

  bool isStructured() const {
    // TODO: We need to be more prescriptive here, but since there's some debate
    // over whether byte address buffer should have a void type or an i8 type,
    // accept either for now.
    Type *Ty = getTypeParameter(0);
    return !Ty->isVoidTy() && !Ty->isIntegerTy(8);
  }

  Type *getResourceType() const {
    return isStructured() ? getTypeParameter(0) : nullptr;
  }
  bool isWriteable() const { return getIntParameter(0); }
  bool isROV() const { return getIntParameter(1); }

  static bool classof(const TargetExtType *T) {
    return T->getName() == "dx.RawBuffer";
  }
  static bool classof(const Type *T) {
    return isa<TargetExtType>(T) && classof(cast<TargetExtType>(T));
  }
};

/// The dx.TypedBuffer target extension type
///
/// `target("dx.TypedBuffer", Type, IsWriteable, IsROV, IsSigned)`
class TypedBufferExtType : public TargetExtType {
public:
  TypedBufferExtType() = delete;
  TypedBufferExtType(const TypedBufferExtType &) = delete;
  TypedBufferExtType &operator=(const TypedBufferExtType &) = delete;

  Type *getResourceType() const { return getTypeParameter(0); }
  bool isWriteable() const { return getIntParameter(0); }
  bool isROV() const { return getIntParameter(1); }
  bool isSigned() const { return getIntParameter(2); }

  static bool classof(const TargetExtType *T) {
    return T->getName() == "dx.TypedBuffer";
  }
  static bool classof(const Type *T) {
    return isa<TargetExtType>(T) && classof(cast<TargetExtType>(T));
  }
};

/// The dx.Texture target extension type
///
/// `target("dx.Texture", Type, IsWriteable, IsROV, IsSigned, Dimension)`
class TextureExtType : public TargetExtType {
public:
  TextureExtType() = delete;
  TextureExtType(const TextureExtType &) = delete;
  TextureExtType &operator=(const TextureExtType &) = delete;

  Type *getResourceType() const { return getTypeParameter(0); }
  bool isWriteable() const { return getIntParameter(0); }
  bool isROV() const { return getIntParameter(1); }
  bool isSigned() const { return getIntParameter(2); }
  dxil::ResourceKind getDimension() const {
    return static_cast<dxil::ResourceKind>(getIntParameter(3));
  }

  static bool classof(const TargetExtType *T) {
    return T->getName() == "dx.Texture";
  }
  static bool classof(const Type *T) {
    return isa<TargetExtType>(T) && classof(cast<TargetExtType>(T));
  }
};

/// The dx.MSTexture target extension type
///
/// `target("dx.MSTexture", Type, IsWriteable, Samples, IsSigned, Dimension)`
class MSTextureExtType : public TargetExtType {
public:
  MSTextureExtType() = delete;
  MSTextureExtType(const MSTextureExtType &) = delete;
  MSTextureExtType &operator=(const MSTextureExtType &) = delete;

  Type *getResourceType() const { return getTypeParameter(0); }
  bool isWriteable() const { return getIntParameter(0); }
  uint32_t getSampleCount() const { return getIntParameter(1); }
  bool isSigned() const { return getIntParameter(2); }
  dxil::ResourceKind getDimension() const {
    return static_cast<dxil::ResourceKind>(getIntParameter(3));
  }

  static bool classof(const TargetExtType *T) {
    return T->getName() == "dx.MSTexture";
  }
  static bool classof(const Type *T) {
    return isa<TargetExtType>(T) && classof(cast<TargetExtType>(T));
  }
};

/// The dx.FeedbackTexture target extension type
///
/// `target("dx.FeedbackTexture", FeedbackType, Dimension)`
class FeedbackTextureExtType : public TargetExtType {
public:
  FeedbackTextureExtType() = delete;
  FeedbackTextureExtType(const FeedbackTextureExtType &) = delete;
  FeedbackTextureExtType &operator=(const FeedbackTextureExtType &) = delete;

  dxil::SamplerFeedbackType getFeedbackType() const {
    return static_cast<dxil::SamplerFeedbackType>(getIntParameter(0));
  }
  dxil::ResourceKind getDimension() const {
    return static_cast<dxil::ResourceKind>(getIntParameter(1));
  }

  static bool classof(const TargetExtType *T) {
    return T->getName() == "dx.FeedbackTexture";
  }
  static bool classof(const Type *T) {
    return isa<TargetExtType>(T) && classof(cast<TargetExtType>(T));
  }
};

/// The dx.CBuffer target extension type
///
/// `target("dx.CBuffer", <Type>, ...)`
class CBufferExtType : public TargetExtType {
public:
  CBufferExtType() = delete;
  CBufferExtType(const CBufferExtType &) = delete;
  CBufferExtType &operator=(const CBufferExtType &) = delete;

  Type *getResourceType() const { return getTypeParameter(0); }
  uint32_t getCBufferSize() const { return getIntParameter(0); }

  static bool classof(const TargetExtType *T) {
    return T->getName() == "dx.CBuffer";
  }
  static bool classof(const Type *T) {
    return isa<TargetExtType>(T) && classof(cast<TargetExtType>(T));
  }
};

/// The dx.Sampler target extension type
///
/// `target("dx.Sampler", SamplerType)`
class SamplerExtType : public TargetExtType {
public:
  SamplerExtType() = delete;
  SamplerExtType(const SamplerExtType &) = delete;
  SamplerExtType &operator=(const SamplerExtType &) = delete;

  dxil::SamplerType getSamplerType() const {
    return static_cast<dxil::SamplerType>(getIntParameter(0));
  }

  static bool classof(const TargetExtType *T) {
    return T->getName() == "dx.Sampler";
  }
  static bool classof(const Type *T) {
    return isa<TargetExtType>(T) && classof(cast<TargetExtType>(T));
  }
};

//===----------------------------------------------------------------------===//

class ResourceTypeInfo {
public:
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

private:
  TargetExtType *HandleTy;

  // GloballyCoherent and HasCounter aren't really part of the type and need to
  // be determined by analysis, so they're just provided directly by the
  // DXILResourceTypeMap when we construct these.
  bool GloballyCoherent;
  bool HasCounter;

  dxil::ResourceClass RC;
  dxil::ResourceKind Kind;

public:
  ResourceTypeInfo(TargetExtType *HandleTy, const dxil::ResourceClass RC,
                   const dxil::ResourceKind Kind, bool GloballyCoherent = false,
                   bool HasCounter = false);
  ResourceTypeInfo(TargetExtType *HandleTy, bool GloballyCoherent = false,
                   bool HasCounter = false)
      : ResourceTypeInfo(HandleTy, {}, dxil::ResourceKind::Invalid,
                         GloballyCoherent, HasCounter) {}

  TargetExtType *getHandleTy() const { return HandleTy; }
  StructType *createElementStruct();

  // Conditions to check before accessing specific views.
  bool isUAV() const;
  bool isCBuffer() const;
  bool isSampler() const;
  bool isStruct() const;
  bool isTyped() const;
  bool isFeedback() const;
  bool isMultiSample() const;

  // Views into the type.
  UAVInfo getUAV() const;
  uint32_t getCBufferSize(const DataLayout &DL) const;
  dxil::SamplerType getSamplerType() const;
  StructInfo getStruct(const DataLayout &DL) const;
  TypedInfo getTyped() const;
  dxil::SamplerFeedbackType getFeedbackType() const;
  uint32_t getMultiSampleCount() const;

  dxil::ResourceClass getResourceClass() const { return RC; }
  dxil::ResourceKind getResourceKind() const { return Kind; }

  void setGloballyCoherent(bool V) { GloballyCoherent = V; }
  void setHasCounter(bool V) { HasCounter = V; }

  bool operator==(const ResourceTypeInfo &RHS) const;
  bool operator!=(const ResourceTypeInfo &RHS) const { return !(*this == RHS); }
  bool operator<(const ResourceTypeInfo &RHS) const;

  void print(raw_ostream &OS, const DataLayout &DL) const;
};

//===----------------------------------------------------------------------===//

class ResourceBindingInfo {
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

private:
  ResourceBinding Binding;
  TargetExtType *HandleTy;
  GlobalVariable *Symbol = nullptr;

public:
  ResourceBindingInfo(uint32_t RecordID, uint32_t Space, uint32_t LowerBound,
                      uint32_t Size, TargetExtType *HandleTy,
                      GlobalVariable *Symbol = nullptr)
      : Binding{RecordID, Space, LowerBound, Size}, HandleTy(HandleTy),
        Symbol(Symbol) {}

  void setBindingID(unsigned ID) { Binding.RecordID = ID; }

  const ResourceBinding &getBinding() const { return Binding; }
  TargetExtType *getHandleTy() const { return HandleTy; }
  const StringRef getName() const { return Symbol ? Symbol->getName() : ""; }

  bool hasSymbol() const { return Symbol; }
  GlobalVariable *createSymbol(Module &M, StructType *Ty, StringRef Name = "");
  MDTuple *getAsMetadata(Module &M, dxil::ResourceTypeInfo &RTI) const;

  std::pair<uint32_t, uint32_t>
  getAnnotateProps(Module &M, dxil::ResourceTypeInfo &RTI) const;

  bool operator==(const ResourceBindingInfo &RHS) const {
    return std::tie(Binding, HandleTy, Symbol) ==
           std::tie(RHS.Binding, RHS.HandleTy, RHS.Symbol);
  }
  bool operator!=(const ResourceBindingInfo &RHS) const {
    return !(*this == RHS);
  }
  bool operator<(const ResourceBindingInfo &RHS) const {
    return Binding < RHS.Binding;
  }

  void print(raw_ostream &OS, dxil::ResourceTypeInfo &RTI,
             const DataLayout &DL) const;
};

} // namespace dxil

//===----------------------------------------------------------------------===//

class DXILResourceTypeMap {
  DenseMap<TargetExtType *, dxil::ResourceTypeInfo> Infos;

public:
  bool invalidate(Module &M, const PreservedAnalyses &PA,
                  ModuleAnalysisManager::Invalidator &Inv);

  dxil::ResourceTypeInfo &operator[](TargetExtType *Ty) {
    auto It = Infos.find(Ty);
    if (It != Infos.end())
      return It->second;
    auto [NewIt, Inserted] = Infos.try_emplace(Ty, Ty);
    return NewIt->second;
  }
};

class DXILResourceTypeAnalysis
    : public AnalysisInfoMixin<DXILResourceTypeAnalysis> {
  friend AnalysisInfoMixin<DXILResourceTypeAnalysis>;

  static AnalysisKey Key;

public:
  using Result = DXILResourceTypeMap;

  DXILResourceTypeMap run(Module &M, ModuleAnalysisManager &AM) {
    // Running the pass just generates an empty map, which will be filled when
    // users of the pass query the results.
    return Result();
  }
};

class DXILResourceTypeWrapperPass : public ImmutablePass {
  DXILResourceTypeMap DRTM;

  virtual void anchor();

public:
  static char ID;
  DXILResourceTypeWrapperPass();

  DXILResourceTypeMap &getResourceTypeMap() { return DRTM; }
  const DXILResourceTypeMap &getResourceTypeMap() const { return DRTM; }
};

ModulePass *createDXILResourceTypeWrapperPassPass();

//===----------------------------------------------------------------------===//

class DXILBindingMap {
  SmallVector<dxil::ResourceBindingInfo> Infos;
  DenseMap<CallInst *, unsigned> CallMap;
  unsigned FirstUAV = 0;
  unsigned FirstCBuffer = 0;
  unsigned FirstSampler = 0;

  /// Populate the map given the resource binding calls in the given module.
  void populate(Module &M, DXILResourceTypeMap &DRTM);

public:
  using iterator = SmallVector<dxil::ResourceBindingInfo>::iterator;
  using const_iterator = SmallVector<dxil::ResourceBindingInfo>::const_iterator;

  iterator begin() { return Infos.begin(); }
  const_iterator begin() const { return Infos.begin(); }
  iterator end() { return Infos.end(); }
  const_iterator end() const { return Infos.end(); }

  bool empty() const { return Infos.empty(); }

  iterator find(const CallInst *Key) {
    auto Pos = CallMap.find(Key);
    return Pos == CallMap.end() ? Infos.end() : (Infos.begin() + Pos->second);
  }

  const_iterator find(const CallInst *Key) const {
    auto Pos = CallMap.find(Key);
    return Pos == CallMap.end() ? Infos.end() : (Infos.begin() + Pos->second);
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

  void print(raw_ostream &OS, DXILResourceTypeMap &DRTM,
             const DataLayout &DL) const;

  friend class DXILResourceBindingAnalysis;
  friend class DXILResourceBindingWrapperPass;
};

class DXILResourceBindingAnalysis
    : public AnalysisInfoMixin<DXILResourceBindingAnalysis> {
  friend AnalysisInfoMixin<DXILResourceBindingAnalysis>;

  static AnalysisKey Key;

public:
  using Result = DXILBindingMap;

  /// Gather resource info for the module \c M.
  DXILBindingMap run(Module &M, ModuleAnalysisManager &AM);
};

/// Printer pass for the \c DXILResourceBindingAnalysis results.
class DXILResourceBindingPrinterPass
    : public PassInfoMixin<DXILResourceBindingPrinterPass> {
  raw_ostream &OS;

public:
  explicit DXILResourceBindingPrinterPass(raw_ostream &OS) : OS(OS) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

  static bool isRequired() { return true; }
};

class DXILResourceBindingWrapperPass : public ModulePass {
  std::unique_ptr<DXILBindingMap> Map;
  DXILResourceTypeMap *DRTM;

public:
  static char ID; // Class identification, replacement for typeinfo

  DXILResourceBindingWrapperPass();
  ~DXILResourceBindingWrapperPass() override;

  const DXILBindingMap &getBindingMap() const { return *Map; }
  DXILBindingMap &getBindingMap() { return *Map; }

  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnModule(Module &M) override;
  void releaseMemory() override;

  void print(raw_ostream &OS, const Module *M) const override;
  void dump() const;
};

ModulePass *createDXILResourceBindingWrapperPassPass();

} // namespace llvm

#endif // LLVM_ANALYSIS_DXILRESOURCE_H
