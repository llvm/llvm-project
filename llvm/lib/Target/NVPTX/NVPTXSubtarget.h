//=====-- NVPTXSubtarget.h - Define Subtarget for the NVPTX ---*- C++ -*--====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the NVPTX specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NVPTX_NVPTXSUBTARGET_H
#define LLVM_LIB_TARGET_NVPTX_NVPTXSUBTARGET_H

#include "NVPTX.h"
#include "NVPTXFrameLowering.h"
#include "NVPTXISelLowering.h"
#include "NVPTXInstrInfo.h"
#include "NVPTXRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/NVVMIntrinsicUtils.h"
#include "llvm/Support/NVPTXAddrSpace.h"
#include "llvm/TargetParser/NVPTXTargetParser.h"

#define GET_SUBTARGETINFO_HEADER
#include "NVPTXGenSubtargetInfo.inc"

namespace llvm {

class NVPTXSubtarget : public NVPTXGenSubtargetInfo {
  virtual void anchor();

  // PTX version x.y is represented as 10*x+y, e.g. 3.1 == 31
  unsigned PTXVersion;

  NVPTX::GPUKind Arch = NVPTX::GK_NONE;

  // Set by every architecture feature. Their bits are the point, so this is
  // only here because a subtarget feature must name a field.
  bool HasArchitecture = false;

  NVPTXInstrInfo InstrInfo;
  NVPTXTargetLowering TLInfo;
  std::unique_ptr<const SelectionDAGTargetInfo> TSInfo;

  // NVPTX does not have any call stack frame, but need a NVPTX specific
  // FrameLowering class because TargetFrameLowering is abstract.
  NVPTXFrameLowering FrameLowering;

public:
  /// This constructor initializes the data members to match that
  /// of the specified module.
  ///
  NVPTXSubtarget(const Triple &TT, StringRef CPU, StringRef FS,
                 const NVPTXTargetMachine &TM);

  ~NVPTXSubtarget() override;

  const TargetFrameLowering *getFrameLowering() const override {
    return &FrameLowering;
  }
  const NVPTXInstrInfo *getInstrInfo() const override { return &InstrInfo; }
  const NVPTXRegisterInfo *getRegisterInfo() const override {
    return &InstrInfo.getRegisterInfo();
  }
  const NVPTXTargetLowering *getTargetLowering() const override {
    return &TLInfo;
  }

  const SelectionDAGTargetInfo *getSelectionDAGInfo() const override {
    return TSInfo.get();
  }

  // True when any of `Features` is enabled.
  bool hasAnyFeature(ArrayRef<unsigned> Features) const {
    return llvm::any_of(Features, [this](unsigned F) { return hasFeature(F); });
  }

  bool has256BitVectorLoadStore(unsigned AS) const {
    return hasFeature(NVPTX::SM100) && hasFeature(NVPTX::PTX88) &&
           AS == NVPTXAS::ADDRESS_SPACE_GLOBAL;
  }
  bool hasUsedBytesMaskPragma() const {
    return hasFeature(NVPTX::SM50) && hasFeature(NVPTX::PTX83);
  }
  bool hasAtomAddF64() const { return hasFeature(NVPTX::SM60); }
  bool hasAtomScope() const { return hasFeature(NVPTX::SM60); }
  bool hasAtomBitwise64() const { return hasFeature(NVPTX::SM32); }
  bool hasAtomMinMax64() const { return hasFeature(NVPTX::SM32); }
  bool hasAtomCas16() const {
    return hasFeature(NVPTX::SM70) && hasFeature(NVPTX::PTX63);
  }
  bool hasAtomSwap128() const {
    return hasFeature(NVPTX::SM90) && hasFeature(NVPTX::PTX83);
  }
  bool hasClusters() const { return hasFeature(NVPTX::SM90); }
  bool hasLDG() const { return hasFeature(NVPTX::SM32); }
  bool hasHWROT32() const { return hasFeature(NVPTX::SM32); }
  bool hasBrx() const {
    return hasFeature(NVPTX::SM30) && hasFeature(NVPTX::PTX60);
  }
  bool hasFP16Math() const { return hasFeature(NVPTX::SM53); }
  bool hasBF16Math() const { return hasFeature(NVPTX::SM80); }
  bool allowFP16Math() const;
  bool hasMaskOperator() const { return hasFeature(NVPTX::PTX71); }
  bool hasNoReturn() const {
    return hasFeature(NVPTX::SM30) && hasFeature(NVPTX::PTX64);
  }
  // Does SM & PTX support memory orderings (weak and atomic: relaxed, acquire,
  // release, acq_rel, sc) ?
  bool hasMemoryOrdering() const { return hasFeature(NVPTX::SM70); }
  // Does SM & PTX support .acquire and .release qualifiers for fence?
  bool hasSplitAcquireAndReleaseFences() const {
    return hasFeature(NVPTX::SM90) && hasFeature(NVPTX::PTX86);
  }
  // Does SM & PTX support atomic relaxed MMIO operations ?
  bool hasRelaxedMMIO() const {
    return hasFeature(NVPTX::SM70) && hasFeature(NVPTX::PTX82);
  }
  bool hasDotInstructions() const { return hasFeature(NVPTX::SM61); }
  // Cache hint SM/PTX version requirements
  bool hasL1EvictionHint() const {
    return hasFeature(NVPTX::SM70) && hasFeature(NVPTX::PTX74);
  }
  bool hasL2EvictionHint() const {
    return hasFeature(NVPTX::SM100) && hasFeature(NVPTX::PTX88);
  }
  bool hasL2Prefetch64B() const {
    return hasFeature(NVPTX::SM75) && hasFeature(NVPTX::PTX74);
  }
  bool hasL2Prefetch128B() const {
    return hasFeature(NVPTX::SM75) && hasFeature(NVPTX::PTX74);
  }
  bool hasL2Prefetch256B() const {
    return hasFeature(NVPTX::SM80) && hasFeature(NVPTX::PTX74);
  }
  bool hasL2CacheHint() const {
    return hasFeature(NVPTX::SM80) && hasFeature(NVPTX::PTX74);
  }

  // Checks following instructions support:
  // - tcgen05.ld/st
  // - tcgen05.alloc/dealloc/relinquish
  // - tcgen05.cp
  // - tcgen05.fence/wait
  // - tcgen05.commit
  // - tcgen05.mma
  bool hasTcgen05InstSupport() const {
    return hasAnyFeature({NVPTX::SM100f, NVPTX::SM110f});
  }

  // f32x2 instructions in Blackwell family
  bool hasF32x2Instructions() const;

  bool hasTensormapReplaceSupport() const {
    return hasAnyFeature({NVPTX::SM100f, NVPTX::SM110f, NVPTX::SM120f}) ||
           (hasFeature(NVPTX::PTX83) && hasAnyFeature({NVPTX::SM90a}));
  }

  bool hasTensormapReplaceElemtypeSupport(unsigned ElemType) const {
    if (ElemType >= static_cast<unsigned>(nvvm::TensormapElemType::B4x16))
      return (hasFeature(NVPTX::PTX88) &&
              hasAnyFeature({NVPTX::SM100f, NVPTX::SM110f, NVPTX::SM120f})) ||
             (hasFeature(NVPTX::PTX87) &&
              hasAnyFeature({NVPTX::SM100a, NVPTX::SM110a, NVPTX::SM120a}));

    return hasTensormapReplaceSupport();
  }

  bool hasTensormapReplaceSwizzleModeSupport(unsigned SwizzleMode) const {
    if (SwizzleMode ==
        static_cast<unsigned>(nvvm::TensormapSwizzleMode::SWIZZLE_96B))
      return hasAnyFeature({NVPTX::SM103a});

    return hasTensormapReplaceSupport();
  }

  // Prior to CUDA 12.3 ptxas did not recognize that the trap instruction
  // terminates a basic block. Instead, it would assume that control flow
  // continued to the next instruction. The next instruction could be in the
  // block that's lexically below it. This would lead to a phantom CFG edges
  // being created within ptxas. This issue was fixed in CUDA 12.3. Thus, when
  // PTX ISA versions 8.3+ we can confidently say that the bug will not be
  // present.
  bool hasPTXASUnreachableBug() const { return !hasFeature(NVPTX::PTX83); }
  bool hasCvtaParam() const {
    return hasFeature(NVPTX::SM70) && hasFeature(NVPTX::PTX77);
  }
  // The compute capability as a number, for __CUDA_ARCH__. This is the one
  // place an architecture needs to be a number, and it is not an identity:
  // sm_100, sm_100f and sm_100a all report 100.
  unsigned getSmVersion() const { return NVPTX::getSmVersion(Arch) / 10; }

  // Whether -mcpu named a target at all, as opposed to falling back to the
  // default architecture.
  bool hasTargetName() const { return !getCPU().empty(); }

  // The architecture's name, which is what `.target` is emitted from.
  StringRef getTargetName() const { return NVPTX::getArchName(Arch); }

  bool hasNativeBF16Support(unsigned Opcode) const;

  // Get maximum value of required alignments among the supported data types.
  // From the PTX ISA doc, section 8.2.3:
  //  The memory consistency model relates operations executed on memory
  //  locations with scalar data-types, which have a maximum size and alignment
  //  of 64 bits. Memory operations with a vector data-type are modelled as a
  //  set of equivalent memory operations with a scalar data-type, executed in
  //  an unspecified order on the elements in the vector.
  unsigned getMaxRequiredAlignment() const { return 8; }
  // Get the smallest cmpxchg word size that the hardware supports.
  unsigned getMinCmpXchgSizeInBits() const { return 32; }

  unsigned getPTXVersion() const { return PTXVersion; }

  NVPTXSubtarget &initializeSubtargetDependencies(StringRef CPU, StringRef FS);
  void ParseSubtargetFeatures(StringRef CPU, StringRef TuneCPU, StringRef FS);
};

} // namespace llvm

#endif
