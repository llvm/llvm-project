//===-- SPUTargetMachine.h - Define TargetMachine for Cell SPU --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the CellSPU-specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef SPU_TARGETMACHINE_H
#define SPU_TARGETMACHINE_H

#include "SPUSubtarget.h"
#include "SPUInstrInfo.h"
#include "SPUISelLowering.h"
#include "SPUSelectionDAGInfo.h"
#include "SPUFrameLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetTransformImpl.h"
#include "llvm/DataLayout.h"

namespace llvm {

/// SPUTargetMachine
///
class SPUTargetMachine : public LLVMTargetMachine {
  SPUSubtarget        Subtarget;
  const DataLayout    DL;
  SPUInstrInfo        InstrInfo;
  SPUFrameLowering    FrameLowering;
  SPUTargetLowering   TLInfo;
  SPUSelectionDAGInfo TSInfo;
  InstrItineraryData  InstrItins;
  ScalarTargetTransformImpl STTI;
  VectorTargetTransformImpl VTTI;
public:
  SPUTargetMachine(const Target &T, StringRef TT,
                   StringRef CPU, StringRef FS, const TargetOptions &Options,
                   Reloc::Model RM, CodeModel::Model CM,
                   CodeGenOpt::Level OL);

  /// Return the subtarget implementation object
  virtual const SPUSubtarget     *getSubtargetImpl() const {
    return &Subtarget;
  }
  virtual const SPUInstrInfo     *getInstrInfo() const {
    return &InstrInfo;
  }
  virtual const SPUFrameLowering *getFrameLowering() const {
    return &FrameLowering;
  }
  /*!
    \note Cell SPU does not support JIT today. It could support JIT at some
    point.
   */
  virtual       TargetJITInfo    *getJITInfo() {
    return NULL;
  }

  virtual const SPUTargetLowering *getTargetLowering() const {
   return &TLInfo;
  }

  virtual const SPUSelectionDAGInfo* getSelectionDAGInfo() const {
    return &TSInfo;
  }

  virtual const SPURegisterInfo *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }

  virtual const DataLayout *getDataLayout() const {
    return &DL;
  }

  virtual const InstrItineraryData *getInstrItineraryData() const {
    return &InstrItins;
  }
  virtual const ScalarTargetTransformInfo *getScalarTargetTransformInfo()const {
    return &STTI;
  }
  virtual const VectorTargetTransformInfo *getVectorTargetTransformInfo()const {
    return &VTTI;
  }

  // Pass Pipeline Configuration
  virtual TargetPassConfig *createPassConfig(PassManagerBase &PM);
};

} // end namespace llvm

#endif
