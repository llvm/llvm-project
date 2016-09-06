//===-- AMDGPUHSATargetObjectFile.cpp - AMDGPU Object Files ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUTargetObjectFile.h"
#include "AMDGPU.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/Support/ELF.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
// Generic Object File
//===----------------------------------------------------------------------===//

MCSection *AMDGPUTargetObjectFile::SelectSectionForGlobal(const GlobalValue *GV,
                                                          SectionKind Kind,
                                                          Mangler &Mang,
                                                const TargetMachine &TM) const {
  if (Kind.isReadOnly() && AMDGPU::isReadOnlySegment(GV))
    return TextSection;

  return TargetLoweringObjectFileELF::SelectSectionForGlobal(GV, Kind, Mang, TM);
}
