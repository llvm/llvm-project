//===-- SparcTargetParserDef.td - SPARC TargetParser -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Target/TargetSubtarget.td"

include "SparcFeatures.td"

// The TargetParser tables do not use the itineraries or scheduling models,
// which depend on the instruction definitions. Define placeholders for the
// ones referenced by the processor definitions.
def LEON2Itineraries : ProcessorItineraries<[], [], []>;
def LEON3Itineraries : ProcessorItineraries<[], [], []>;
def LEON4Itineraries : ProcessorItineraries<[], [], []>;
def LEON5Model : SchedMachineModel;

include "SparcProcessors.td"

class Target;
def Sparc : Target;
