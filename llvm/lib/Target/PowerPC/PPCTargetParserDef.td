//===-- PPCTargetParserDef.td - PowerPC TargetParser -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Target/TargetSubtarget.td"

include "PPCFeatures.td"

// The TargetParser tables do not use the itineraries or scheduling models,
// which depend on the instruction definitions. Define placeholders for the
// ones referenced by the processor definitions.
def G3Itineraries : ProcessorItineraries<[], [], []>;
def G4Itineraries : ProcessorItineraries<[], [], []>;
def G4PlusItineraries : ProcessorItineraries<[], [], []>;
def G5Model : SchedMachineModel;
def P10Model : SchedMachineModel;
def P7Model : SchedMachineModel;
def P8Model : SchedMachineModel;
def P9Model : SchedMachineModel;
def PPC440Model : SchedMachineModel;
def PPCA2Model : SchedMachineModel;
def PPCE500Model : SchedMachineModel;
def PPCE500mcModel : SchedMachineModel;
def PPCE5500Model : SchedMachineModel;

include "PPCProcessors.td"

class Target;
def PPC : Target;
