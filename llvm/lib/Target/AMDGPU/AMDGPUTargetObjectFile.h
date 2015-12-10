//===-- AMDGPUTargetObjectFile.h - AMDGPU  Object Info ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file declares the AMDGPU-specific subclass of
/// TargetLoweringObjectFile.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUTARGETOBJECTFILE_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUTARGETOBJECTFILE_H

#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class AMDGPUTargetObjectFile : public TargetLoweringObjectFileELF {
  public:
    MCSection *SelectSectionForGlobal(const GlobalValue *GV, SectionKind Kind,
                                      Mangler &Mang,
                                      const TargetMachine &TM) const override;
};

class AMDGPUHSATargetObjectFile final : public AMDGPUTargetObjectFile {
private:
  MCSection *DataGlobalAgentSection;
  MCSection *DataGlobalProgramSection;
  MCSection *RodataReadonlyAgentSection;

  bool isAgentAllocationSection(const char *SectionName) const;
  bool isAgentAllocation(const GlobalValue *GV) const;
  bool isProgramAllocation(const GlobalValue *GV) const;

public:
  void Initialize(MCContext &Ctx, const TargetMachine &TM) override;

  MCSection *SelectSectionForGlobal(const GlobalValue *GV, SectionKind Kind,
                                    Mangler &Mang,
                                    const TargetMachine &TM) const override;
};

} // end namespace llvm

#endif
