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
#include "llvm/Support/NVPTXAddrSpace.h"
#include <string>

#define GET_SUBTARGETINFO_HEADER
#include "NVPTXGenSubtargetInfo.inc"

namespace llvm {

class NVPTXSubtarget : public NVPTXGenSubtargetInfo {
  virtual void anchor();
  std::string TargetName;

  // PTX version x.y is represented as 10*x+y, e.g. 3.1 == 31
  unsigned PTXVersion;

  // Full SM version x.y is represented as 100*x+10*y+feature, e.g. 3.1 == 310
  // sm_90a == 901
  unsigned int FullSmVersion;

  // SM version x.y is represented as 10*x+y, e.g. 3.1 == 31. Derived from
  // FullSmVersion.
  unsigned int SmVersion;

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
  NVPTXSubtarget(const Triple &TT, const std::string &CPU,
                 const std::string &FS, const NVPTXTargetMachine &TM);

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

  const SelectionDAGTargetInfo *getSelectionDAGInfo() const override;

  bool has256BitVectorLoadStore(unsigned AS) const {
    return SmVersion >= 100 && PTXVersion >= 88 &&
           AS == NVPTXAS::ADDRESS_SPACE_GLOBAL;
  }
  bool hasAtomAddF64() const { return SmVersion >= 60; }
  bool hasAtomScope() const { return SmVersion >= 60; }
  bool hasAtomBitwise64() const { return SmVersion >= 32; }
  bool hasAtomMinMax64() const { return SmVersion >= 32; }
  bool hasAtomCas16() const { return SmVersion >= 70 && PTXVersion >= 63; }
  bool hasAtomSwap128() const { return SmVersion >= 90 && PTXVersion >= 83; }
  bool hasClusters() const { return SmVersion >= 90 && PTXVersion >= 78; }
  bool hasLDG() const { return SmVersion >= 32; }
  bool hasHWROT32() const { return SmVersion >= 32; }
  bool hasFP16Math() const { return SmVersion >= 53; }
  bool hasBF16Math() const { return SmVersion >= 80; }
  bool allowFP16Math() const;
  bool hasMaskOperator() const { return PTXVersion >= 71; }
  bool hasNoReturn() const { return SmVersion >= 30 && PTXVersion >= 64; }
  // Does SM & PTX support memory orderings (weak and atomic: relaxed, acquire,
  // release, acq_rel, sc) ?
  bool hasMemoryOrdering() const { return SmVersion >= 70 && PTXVersion >= 60; }
  // Does SM & PTX support .acquire and .release qualifiers for fence?
  bool hasSplitAcquireAndReleaseFences() const {
    return SmVersion >= 90 && PTXVersion >= 86;
  }
  // Does SM & PTX support atomic relaxed MMIO operations ?
  bool hasRelaxedMMIO() const { return SmVersion >= 70 && PTXVersion >= 82; }
  bool hasDotInstructions() const {
    return SmVersion >= 61 && PTXVersion >= 50;
  }
  // Tcgen05 instructions in Blackwell family
  bool hasTcgen05Instructions() const {
    bool HasTcgen05 = false;
    switch (FullSmVersion) {
    default:
      break;
    case 1003: // sm_100a
    case 1013: // sm_101a
      HasTcgen05 = true;
      break;
    }

    return HasTcgen05 && PTXVersion >= 86;
  }
  // f32x2 instructions in Blackwell family
  bool hasF32x2Instructions() const;

  // TMA G2S copy with cta_group::1/2 support
  bool hasCpAsyncBulkTensorCTAGroupSupport() const {
    // TODO: Update/tidy-up after the family-conditional support arrives
    switch (FullSmVersion) {
    case 1003:
    case 1013:
      return PTXVersion >= 86;
    case 1033:
      return PTXVersion >= 88;
    default:
      return false;
    }
  }

  // Prior to CUDA 12.3 ptxas did not recognize that the trap instruction
  // terminates a basic block. Instead, it would assume that control flow
  // continued to the next instruction. The next instruction could be in the
  // block that's lexically below it. This would lead to a phantom CFG edges
  // being created within ptxas. This issue was fixed in CUDA 12.3. Thus, when
  // PTX ISA versions 8.3+ we can confidently say that the bug will not be
  // present.
  bool hasPTXASUnreachableBug() const { return PTXVersion < 83; }
  bool hasCvtaParam() const { return SmVersion >= 70 && PTXVersion >= 77; }
  unsigned int getFullSmVersion() const { return FullSmVersion; }
  unsigned int getSmVersion() const { return getFullSmVersion() / 10; }
  // GPUs with "a" suffix have architecture-accelerated features that are
  // supported on the specified architecture only, hence such targets do not
  // follow the onion layer model. hasArchAccelFeatures() allows distinguishing
  // such GPU variants from the base GPU architecture.
  // - false represents non-accelerated architecture.
  // - true represents architecture-accelerated variant.
  bool hasArchAccelFeatures() const {
    return (getFullSmVersion() & 1) && PTXVersion >= 80;
  }
  // GPUs with 'f' suffix have architecture-accelerated features which are
  // portable across all future architectures under same SM major. For example,
  // sm_100f features will work for sm_10X*f*/sm_10X*a* future architectures.
  // - false represents non-family-specific architecture.
  // - true represents family-specific variant.
  bool hasFamilySpecificFeatures() const {
    return getFullSmVersion() % 10 == 2 ? PTXVersion >= 88
                                        : hasArchAccelFeatures();
  }
  // If the user did not provide a target we default to the `sm_30` target.
  std::string getTargetName() const {
    return TargetName.empty() ? "sm_30" : TargetName;
  }
  bool hasTargetName() const { return !TargetName.empty(); }

  bool hasNativeBF16Support(int Opcode) const;

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

  void failIfClustersUnsupported(std::string const &FailureMessage) const;
};

} // End llvm namespace

#endif
