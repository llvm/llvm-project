//=-- HexagonTargetMachine.h - Define TargetMachine for Hexagon ---*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Hexagon specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef HexagonTARGETMACHINE_H
#define HexagonTARGETMACHINE_H

#include "HexagonInstrInfo.h"
#include "HexagonSubtarget.h"
#include "HexagonISelLowering.h"
#include "HexagonSelectionDAGInfo.h"
#include "HexagonFrameLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetTransformImpl.h"

namespace llvm {

class Module;

class HexagonTargetMachine : public LLVMTargetMachine {
  const DataLayout DL;       // Calculates type size & alignment.
  HexagonSubtarget Subtarget;
  HexagonInstrInfo InstrInfo;
  HexagonTargetLowering TLInfo;
  HexagonSelectionDAGInfo TSInfo;
  HexagonFrameLowering FrameLowering;
  const InstrItineraryData* InstrItins;
  ScalarTargetTransformImpl STTI;
  VectorTargetTransformImpl VTTI;

public:
  HexagonTargetMachine(const Target &T, StringRef TT,StringRef CPU,
                       StringRef FS, const TargetOptions &Options,
                       Reloc::Model RM, CodeModel::Model CM,
                       CodeGenOpt::Level OL);

  virtual const HexagonInstrInfo *getInstrInfo() const {
    return &InstrInfo;
  }
  virtual const HexagonSubtarget *getSubtargetImpl() const {
    return &Subtarget;
  }
  virtual const HexagonRegisterInfo *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }

  virtual const InstrItineraryData* getInstrItineraryData() const {
    return InstrItins;
  }


  virtual const HexagonTargetLowering* getTargetLowering() const {
    return &TLInfo;
  }

  virtual const HexagonFrameLowering* getFrameLowering() const {
    return &FrameLowering;
  }

  virtual const HexagonSelectionDAGInfo* getSelectionDAGInfo() const {
    return &TSInfo;
  }

  virtual const ScalarTargetTransformInfo *getScalarTargetTransformInfo()const {
    return &STTI;
  }

  virtual const VectorTargetTransformInfo *getVectorTargetTransformInfo()const {
    return &VTTI;
  }

  virtual const DataLayout       *getDataLayout() const { return &DL; }
  static unsigned getModuleMatchQuality(const Module &M);

  // Pass Pipeline Configuration.
  virtual bool addPassesForOptimizations(PassManagerBase &PM);
  virtual TargetPassConfig *createPassConfig(PassManagerBase &PM);
};

extern bool flag_aligned_memcpy;

} // end namespace llvm

#endif
