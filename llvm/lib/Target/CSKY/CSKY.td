//===-- CSKY.td - Describe the CSKY Target Machine ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// CSKY subtarget features and instruction predicates.
//===----------------------------------------------------------------------===//

// Atomic Support
def FeatureExtendLrw : SubtargetFeature<"elrw", "HasExtendLrw", "true",
                                        "Use the extend LRW instruction">;
def HasExtendLrw : Predicate<"Subtarget->hasExtendLrw()">,
                   AssemblerPredicate<(all_of FeatureExtendLrw),
                   "Use the extend LRW instruction">;

def FeatureDoloop : SubtargetFeature<"doloop", "HasDoloop", "true",
                                     "Enable doloop instructions">;
def HasDoloop : Predicate<"Subtarget->hasDoloop()">,
                AssemblerPredicate<(all_of FeatureDoloop),
                "Enable doloop instructions">;

def HasE1
    : SubtargetFeature<"e1", "HasE1", "true", "Support CSKY e1 instructions",
                       [FeatureExtendLrw]>;
def iHasE1 : Predicate<"Subtarget->hasE1()">,
             AssemblerPredicate<(all_of HasE1),
             "Support CSKY e1 instructions">;

def HasE2
    : SubtargetFeature<"e2", "HasE2", "true", "Support CSKY e2 instructions",
                       [HasE1]>;
def iHasE2 : Predicate<"Subtarget->hasE2()">,
             AssemblerPredicate<(all_of HasE2),
             "Support CSKY e2 instructions">;

def Has2E3 : SubtargetFeature<"2e3", "Has2E3", "true",
                              "Support CSKY 2e3 instructions", [HasE2]>;
def iHas2E3 : Predicate<"Subtarget->has2E3()">,
              AssemblerPredicate<(all_of Has2E3),
              "Support CSKY 2e3 instructions">;

def Has3E3r1 : SubtargetFeature<"3e3r1", "Has3E3r1", "true",
                                "Support CSKY 3e3r1 instructions">;
def iHas3E3r1 : Predicate<"Subtarget->has3E3r1()">,
                AssemblerPredicate<(all_of Has3E3r1),
                "Support CSKY 3e3r1 instructions">;

def Has3r2E3r3
    : SubtargetFeature<"3e3r3", "Has3r2E3r3", "true",
                       "Support CSKY 3e3r3 instructions", [FeatureDoloop]>;
def iHas3r2E3r3 : Predicate<"Subtarget->has3r2E3r3()">,
                  AssemblerPredicate<(all_of Has3r2E3r3),
                  "Support CSKY 3e3r3 instructions">;

def Has3E7 : SubtargetFeature<"3e7", "Has3E7", "true",
                              "Support CSKY 3e7 instructions", [Has2E3]>;
def iHas3E7 : Predicate<"Subtarget->has3E7()">,
              AssemblerPredicate<(all_of Has3E7),
              "Support CSKY 3e7 instructions">;

def HasMP1E2 : SubtargetFeature<"mp1e2", "HasMP1E2", "true",
                                "Support CSKY mp1e2 instructions", [Has3E7]>;
def iHasMP1E2 : Predicate<"Subtarget->hasMP1E2()">,
                AssemblerPredicate<(all_of HasMP1E2),
                "Support CSKY mp1e2 instructions">;

def Has7E10 : SubtargetFeature<"7e10", "Has7E10", "true",
                               "Support CSKY 7e10 instructions", [Has3E7]>;
def iHas7E10 : Predicate<"Subtarget->has7E10()">,
               AssemblerPredicate<(all_of Has7E10),
               "Support CSKY 7e10 instructions">;

def Has10E60 : SubtargetFeature<"10e60", "Has10E60", "true",
                               "Support CSKY 10e60 instructions", [Has7E10]>;
def iHas10E60 : Predicate<"Subtarget->has10E60()">,
               AssemblerPredicate<(all_of Has10E60),
               "Support CSKY 10e60 instructions">;

//===----------------------------------------------------------------------===//
// Registers, calling conventions, instruction descriptions.
//===----------------------------------------------------------------------===//

include "CSKYRegisterInfo.td"
include "CSKYInstrInfo.td"

//===----------------------------------------------------------------------===//
// CSKY processors supported.
//===----------------------------------------------------------------------===//

def : ProcessorModel<"generic", NoSchedModel, []>;

//===----------------------------------------------------------------------===//
// Define the CSKY target.
//===----------------------------------------------------------------------===//

def CSKYInstrInfo : InstrInfo;


def CSKYAsmParser : AsmParser {
  let ShouldEmitMatchRegisterAltName = 1;
  let AllowDuplicateRegisterNames = 1;
}

def CSKYAsmWriter : AsmWriter {
  int PassSubtarget = 1;
}

def CSKY : Target {
  let InstructionSet = CSKYInstrInfo;
  let AssemblyParsers = [CSKYAsmParser];
  let AssemblyWriters = [CSKYAsmWriter];
  let AllowRegisterRenaming = 1;
}
