//===-- RISCVTargetParserDef.td - RISC-V TargetParser ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Target/TargetSubtarget.td"
include "llvm/TableGen/SearchableTable.td"

include "RISCVFeatures.td"
include "RISCVProfiles.td"

// The TargetParser tables do not use the scheduling models, which depend on
// the instruction definitions. Define placeholders for the models referenced
// by the processor definitions.
def Andes45Model : SchedMachineModel;
def GenericOOOModel : SchedMachineModel;
def MIPSP8700Model : SchedMachineModel;
def RocketModel : SchedMachineModel;
def SiFive7Model : SchedMachineModel;
def SiFiveP400Model : SchedMachineModel;
def SiFiveP500Model : SchedMachineModel;
def SiFiveP600Model : SchedMachineModel;
def SiFiveP800Model : SchedMachineModel;
def SiFiveX100Model : SchedMachineModel;
def SiFiveX390Model : SchedMachineModel;
def SpacemitX100Model : SchedMachineModel;
def SpacemitX60Model : SchedMachineModel;
def SyntacoreSCR1Model : SchedMachineModel;
def SyntacoreSCR3RV32Model : SchedMachineModel;
def SyntacoreSCR3RV64Model : SchedMachineModel;
def SyntacoreSCR4RV32Model : SchedMachineModel;
def SyntacoreSCR4RV64Model : SchedMachineModel;
def SyntacoreSCR5RV32Model : SchedMachineModel;
def SyntacoreSCR5RV64Model : SchedMachineModel;
def SyntacoreSCR7Model : SchedMachineModel;
def TTAscalonXGModel : SchedMachineModel;
def TTAscalonXModel : SchedMachineModel;
def XiangShanKunMingHuModel : SchedMachineModel;
def XiangShanNanHuModel : SchedMachineModel;

include "RISCVProcessors.td"
