//===-- AMDGPUHSATargetObjectFile.cpp - AMDGPU Object Files ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUHSATargetObjectFile.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/Support/ELF.h"

using namespace llvm;

void AMDGPUHSATargetObjectFile::Initialize(MCContext &Ctx,
                                           const TargetMachine &TM){
  TargetLoweringObjectFileELF::Initialize(Ctx, TM);
  InitializeELF(TM.Options.UseInitArray);

  TextSection = AMDGPU::getHSATextSection(Ctx);

}

MCSection *AMDGPUHSATargetObjectFile::SelectSectionForGlobal(
                                        const GlobalValue *GV, SectionKind Kind,
                                        Mangler &Mang,
                                        const TargetMachine &TM) const {
  if (Kind.isText() && !GV->hasComdat())
    return getTextSection();

  return TargetLoweringObjectFileELF::SelectSectionForGlobal(GV, Kind, Mang, TM);
}
