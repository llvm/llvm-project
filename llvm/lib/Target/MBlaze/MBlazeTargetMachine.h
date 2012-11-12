//===-- MBlazeTargetMachine.h - Define TargetMachine for MBlaze -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the MBlaze specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef MBLAZE_TARGETMACHINE_H
#define MBLAZE_TARGETMACHINE_H

#include "MBlazeSubtarget.h"
#include "MBlazeInstrInfo.h"
#include "MBlazeISelLowering.h"
#include "MBlazeSelectionDAGInfo.h"
#include "MBlazeIntrinsicInfo.h"
#include "MBlazeFrameLowering.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetTransformImpl.h"

namespace llvm {
  class formatted_raw_ostream;

  class MBlazeTargetMachine : public LLVMTargetMachine {
    MBlazeSubtarget        Subtarget;
    const DataLayout       DL; // Calculates type size & alignment
    MBlazeInstrInfo        InstrInfo;
    MBlazeFrameLowering    FrameLowering;
    MBlazeTargetLowering   TLInfo;
    MBlazeSelectionDAGInfo TSInfo;
    MBlazeIntrinsicInfo    IntrinsicInfo;
    InstrItineraryData     InstrItins;
    ScalarTargetTransformImpl STTI;
    VectorTargetTransformImpl VTTI;

  public:
    MBlazeTargetMachine(const Target &T, StringRef TT,
                        StringRef CPU, StringRef FS,
                        const TargetOptions &Options,
                        Reloc::Model RM, CodeModel::Model CM,
                        CodeGenOpt::Level OL);

    virtual const MBlazeInstrInfo *getInstrInfo() const
    { return &InstrInfo; }

    virtual const InstrItineraryData *getInstrItineraryData() const
    {  return &InstrItins; }

    virtual const TargetFrameLowering *getFrameLowering() const
    { return &FrameLowering; }

    virtual const MBlazeSubtarget *getSubtargetImpl() const
    { return &Subtarget; }

    virtual const DataLayout *getDataLayout() const
    { return &DL;}

    virtual const MBlazeRegisterInfo *getRegisterInfo() const
    { return &InstrInfo.getRegisterInfo(); }

    virtual const MBlazeTargetLowering *getTargetLowering() const
    { return &TLInfo; }

    virtual const MBlazeSelectionDAGInfo* getSelectionDAGInfo() const
    { return &TSInfo; }

    const TargetIntrinsicInfo *getIntrinsicInfo() const
    { return &IntrinsicInfo; }

    virtual const ScalarTargetTransformInfo *getScalarTargetTransformInfo()const
    { return &STTI; }
    virtual const VectorTargetTransformInfo *getVectorTargetTransformInfo()const
    { return &VTTI; }

    // Pass Pipeline Configuration
    virtual TargetPassConfig *createPassConfig(PassManagerBase &PM);
  };
} // End llvm namespace

#endif
