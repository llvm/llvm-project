//===- NVPTX.td - Describe the NVPTX Target Machine -----------*- tblgen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This is the top level entry point for the NVPTX target.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Target-independent interfaces
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// PTX ISA versions.
//===----------------------------------------------------------------------===//

class FeaturePTX<int version, list<SubtargetFeature> implies>:
   SubtargetFeature<"ptx" # version, "PTXVersion",
                    "" # version,
                    "Use PTX version " # version, implies>,
   Predicate<"Subtarget->hasFeature(NVPTX::" # NAME # ")">;

defvar PTXVersions = [32, 40, 41, 42, 43, 50, 60, 61, 62, 63, 64, 65, 70, 71,
                      72, 73, 74, 75, 76, 77, 78, 80, 81, 82, 83, 84, 85, 86,
                      87, 88, 90, 91, 92, 93, 94];

foreach version = PTXVersions in
  def PTX # version : FeaturePTX<version,
                                 !foreach(earlier,
                                      !filter(v, PTXVersions, !lt(v, version)),
                                      !cast<SubtargetFeature>("PTX" # earlier))>;

//===----------------------------------------------------------------------===//
// Target architectures.
//
// Architectures are named sm_XYz, where 'X' is the generation, 'Y' the version
// within the generation, and 'z' an optional feature suffix. Together they form
// a lattice, spelled out below as `Implies` edges: sm_A implies sm_B exactly
// when PTX written against sm_B is accepted for sm_A. Every edge is listed
// rather than leaning on `Implies` being transitive, which costs nothing
// because a feature's implied set is a fixed-size bitset either way. Asking
// whether a target may use what sm_B introduced is then a single bit test.
//
// The three tiers below are the three portability guarantees the PTX ISA
// defines for a `.target`; see the .target directive in the PTX ISA manual.
//
//  * Base architectures are portable: what they introduce is supported on all
//    later architectures, including ones in other families. This is the "onion
//    layer" model, and it makes them totally ordered by their number, so sm_120
//    implies sm_110, which implies sm_107, and so on.
//
//  * A family-specific variant sm_XYf is portable within its family only, so
//    sm_103f implies sm_100f. A family is the set of architectures sharing an
//    SM major version. There are no edges between families, which is what makes
//    sm_120f and sm_100f incomparable.
//
//  * An architecture-specific variant sm_XYa is not portable at all, so
//    sm_103a implies sm_103f and nothing implies sm_103a.
//
// There is deliberately no numeric encoding of an architecture. Asking whether a
// target may use something is a bit test against the node that introduced it,
// which is what the partial order above requires; a number would only invite the
// ordered comparisons it cannot answer.
//===----------------------------------------------------------------------===//

class FeatureSM<list<SubtargetFeature> implies>:
   SubtargetFeature<"sm_"# !substr(NAME, 2), "HasArchitecture", "true",
                    "Target SM " # !substr(NAME, 2), implies>,
   Predicate<"Subtarget->hasFeature(NVPTX::" # NAME # ")"> {
  assert !eq(!substr(NAME, 0, 2), "SM"),
         "architecture record " # NAME # " must be named SM<version>";
}

class Proc<FeatureSM SM>
 : Processor<SM.Name, NoItineraries, [SM]>;

multiclass ProcSM<list<SubtargetFeature> implies> {
  def NAME : FeatureSM<implies>;
  def : Proc<!cast<FeatureSM>(NAME)>;
}

// Base architectures, in increasing order. Each implies the features of every
// earlier one.
defvar SMVersions = [20, 21, 30, 32, 35, 37, 50, 52, 53, 60,
                     61, 62, 70, 72, 75, 80, 86, 87, 88, 89,
                     90, 100, 103, 107, 110, 120, 121];

foreach sm = SMVersions in
  defm SM#sm : ProcSM<!foreach(earlier, !filter(v, SMVersions, !lt(v, sm)),
                               !cast<SubtargetFeature>("SM"#earlier))>;

// Defines the family-specific ('f') and architecture-specific ('a') variants of
// the base architecture this is named after.
multiclass FeatureSMVariants<list<SubtargetFeature> EarlierInFamily> {
  defm f : ProcSM<[!cast<SubtargetFeature>(NAME)] # EarlierInFamily>;
  defm a : ProcSM<[!cast<SubtargetFeature>(NAME # "f")]>;
}

// The families, each listed in increasing order, mirroring the architecture
// family table in the PTX ISA manual.
defvar SMFamilies = [[100, 103, 107],  // sm_10x
                     [110],            // sm_11x
                     [120, 121]];      // sm_12x

foreach Family = SMFamilies in
  foreach sm = Family in
    defm SM#sm : FeatureSMVariants<
        !foreach(earlier, !filter(v, Family, !lt(v, sm)),
                 !cast<SubtargetFeature>("SM" # earlier # "f"))>;

// sm_90 predates family-specific variants, so sm_90a is the only member of the
// sm_9x family and stands in for its 'f' variant, which does not exist.
defm SM90a : ProcSM<[SM90]>;

// sm_101 was the name PTX 8.6 through 8.8 used for what PTX 9.0 renamed to
// sm_110. The two names describe the same hardware, so they share a node rather
// than duplicating one.
def : ProcessorAlias<"sm_101", "sm_110">;
def : ProcessorAlias<"sm_101f", "sm_110f">;
def : ProcessorAlias<"sm_101a", "sm_110a">;

include "NVPTXRegisterInfo.td"
include "NVPTXInstrInfo.td"

def Is64Bit : Predicate<"Subtarget->getTargetTriple().getArch() == Triple::nvptx64">;
def NVPTX64 : HwMode<[Is64Bit]>;

def nvptx_ptr_rc : RegClassByHwMode<
  [DefaultMode, NVPTX64],
  [B32, B64]>;

defm : RemapAllTargetPseudoPointerOperands<nvptx_ptr_rc>;

def NVPTXInstrInfo : InstrInfo {
}

def NVPTXAsmWriter : AsmWriter {
  int PassSubtarget = 1;
}

def NVPTX : Target {
  let InstructionSet = NVPTXInstrInfo;
  let AssemblyWriters = [NVPTXAsmWriter];
}
