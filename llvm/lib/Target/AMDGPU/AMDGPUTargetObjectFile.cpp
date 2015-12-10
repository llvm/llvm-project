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

//===----------------------------------------------------------------------===//
// HSA Object File
//===----------------------------------------------------------------------===//


void AMDGPUHSATargetObjectFile::Initialize(MCContext &Ctx,
                                           const TargetMachine &TM){
  TargetLoweringObjectFileELF::Initialize(Ctx, TM);
  InitializeELF(TM.Options.UseInitArray);

  TextSection = AMDGPU::getHSATextSection(Ctx);

  DataGlobalAgentSection = AMDGPU::getHSADataGlobalAgentSection(Ctx);
  DataGlobalProgramSection = AMDGPU::getHSADataGlobalProgramSection(Ctx);

  RodataReadonlyAgentSection = AMDGPU::getHSARodataReadonlyAgentSection(Ctx);
}

bool AMDGPUHSATargetObjectFile::isAgentAllocationSection(
    const char *SectionName) const {
  return cast<MCSectionELF>(DataGlobalAgentSection)
      ->getSectionName()
      .equals(SectionName);
}

bool AMDGPUHSATargetObjectFile::isAgentAllocation(const GlobalValue *GV) const {
  // Read-only segments can only have agent allocation.
  return AMDGPU::isReadOnlySegment(GV) ||
         (AMDGPU::isGlobalSegment(GV) && GV->hasSection() &&
          isAgentAllocationSection(GV->getSection()));
}

bool AMDGPUHSATargetObjectFile::isProgramAllocation(
    const GlobalValue *GV) const {
  // The default for global segments is program allocation.
  return AMDGPU::isGlobalSegment(GV) && !isAgentAllocation(GV);
}

MCSection *AMDGPUHSATargetObjectFile::SelectSectionForGlobal(
                                        const GlobalValue *GV, SectionKind Kind,
                                        Mangler &Mang,
                                        const TargetMachine &TM) const {
  if (Kind.isText() && !GV->hasComdat())
    return getTextSection();

  if (AMDGPU::isGlobalSegment(GV)) {
    if (isAgentAllocation(GV))
      return DataGlobalAgentSection;

    if (isProgramAllocation(GV))
      return DataGlobalProgramSection;
  }

  if (Kind.isReadOnly() && AMDGPU::isReadOnlySegment(GV))
    return RodataReadonlyAgentSection;

  return TargetLoweringObjectFileELF::SelectSectionForGlobal(GV, Kind, Mang, TM);
}
