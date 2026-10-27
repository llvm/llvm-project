//===-- AArch64TargetParserDef.td - AArch64 TargetParser ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Target/TargetSubtarget.td"

include "AArch64Features.td"
include "AArch64FMV.td"

// The TargetParser tables do not use the scheduling models, which depend on
// the instruction definitions. Define placeholders for the models referenced
// by the processor definitions.
def A64FXModel : SchedMachineModel;
def Ampere1BModel : SchedMachineModel;
def Ampere1Model : SchedMachineModel;
def C1NanoModel : SchedMachineModel;
def C1PremiumModel : SchedMachineModel;
def C1UltraModel : SchedMachineModel;
def CortexA320Model : SchedMachineModel;
def CortexA510Model : SchedMachineModel;
def CortexA53Model : SchedMachineModel;
def CortexA55Model : SchedMachineModel;
def CortexA57Model : SchedMachineModel;
def CycloneModel : SchedMachineModel;
def ExynosM3Model : SchedMachineModel;
def ExynosM4Model : SchedMachineModel;
def ExynosM5Model : SchedMachineModel;
def FalkorModel : SchedMachineModel;
def HIP12Model : SchedMachineModel;
def KryoModel : SchedMachineModel;
def NeoverseN1Model : SchedMachineModel;
def NeoverseN2Model : SchedMachineModel;
def NeoverseN3Model : SchedMachineModel;
def NeoverseV1Model : SchedMachineModel;
def NeoverseV2Model : SchedMachineModel;
def NeoverseV3AEModel : SchedMachineModel;
def NeoverseV3Model : SchedMachineModel;
def OlympusModel : SchedMachineModel;
def OryonModel : SchedMachineModel;
def ThunderX2T99Model : SchedMachineModel;
def ThunderX3T110Model : SchedMachineModel;
def ThunderXT8XModel : SchedMachineModel;
def TSV110Model : SchedMachineModel;

include "AArch64Processors.td"
