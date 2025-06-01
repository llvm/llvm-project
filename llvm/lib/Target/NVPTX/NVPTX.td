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

include "NVPTXRegisterInfo.td"
include "NVPTXInstrInfo.td"

//===----------------------------------------------------------------------===//
// Subtarget Features.
// - We use the SM version number instead of explicit feature table.
// - Need at least one feature to avoid generating zero sized array by
//   TableGen in NVPTXGenSubtarget.inc.
//===----------------------------------------------------------------------===//

class FeatureSM<string sm, int value>:
   SubtargetFeature<"sm_"# sm, "FullSmVersion",
                    "" # value,
                    "Target SM " # sm>;

class FeaturePTX<int version>:
   SubtargetFeature<"ptx"# version, "PTXVersion",
                    "" # version,
                    "Use PTX version " # version>;
//
// NVPTX Architecture Hierarchy and Ordering:
// 
// Family: 2/3/5/6/7/8/9/10/12 (Follows Onion model, older family is compatible with newer family)
// Arch: 2*/3*/5*/6*/7*/8*/9*/10*/12*
//
// Family-specific: F*f : F*f > F* =>
// + The plain base architecture is compatible with the family-specific architecture
//   (e.g. sm_100 compatible with >= sm_100*f*)
// + The family-specific architecture is compatible with future family-specific
//   architectures within the same family (e.g. sm_100f compatible with >= sm_10X*f*
//   but not with sm_12X*f*)
//
//    Family and SM Target Definition:
//    +----------------+--------------------------------------------------------+
//    | Family         | Target SM architectures included                       |
//    +----------------+--------------------------------------------------------+
//    | sm_10x family  | sm_100f, sm_103f, future targets in sm_10x family      |
//    | sm_101 family  | sm_101f (exception)                                    |
//    | sm_12x family  | sm_120f, sm_121f, future targets in sm_12x family      |
//    +----------------+--------------------------------------------------------+
//
// Architecture-specific: F*a : F*a > F*f > F* =>
// + The plain base architecture is compatible with the architecture-specific architecture
//   (e.g. sm_100 compatible with >= sm_100*a*)
// + The family-specific architecture is compatible with the architecture-specific architecture
//   (e.g. sm_100f compatible with >= sm_100*a*)
// + The architecture-specific architecture is incompatible with any other architecture
//   (e.g. sm_100a is only compatible with sm_100*a*)
//
// Encoding: Arch * 1000 + 'f' * 10 + 'a' * 1 (where 'a' ⇒ 'f')
// 
// This encoding allows simple implementation of the partial ordering of the architectures.
//  + Compare Family and Arch by dividing FullSMVersion by 1000 and 100 respectively before the comparison.
//  + Compare within the family by comparing FullSMVersion, given both belongs to the same family.
//  + Detect 'a' variants by checking FullSMVersion % 10.
//
foreach sm = [20, 21, 30, 32, 35, 37, 50, 52, 53,
              60, 61, 62, 70, 72, 75, 80, 86, 87,
              89, 90, 100, 101, 103, 120, 121] in {
  // Base SM version (e.g. FullSMVersion for sm_100 is 10000)
  def SM#sm : FeatureSM<""#sm, !mul(sm, 100)>;

  // Family-specific targets which are compatible within same family
  // (e.g. FullSMVersion for sm_100f is 10010)
  if !ge(sm, 100) then {
    def SM#sm#f : FeatureSM<""#sm#"f", !add(!mul(sm, 100), 10)>;
  }

  // Architecture-specific targets which are incompatible across architectures
  // (e.g. FullSMVersion for sm_100a is 10011)
  if !ge(sm, 90) then {
    def SM#sm#a : FeatureSM<""#sm#"a", !add(!mul(sm, 100), 11)>;
  }
}

foreach version = [32, 40, 41, 42, 43, 50, 60, 61, 62, 63, 64, 65,
                   70, 71, 72, 73, 74, 75, 76, 77, 78,
                   80, 81, 82, 83, 84, 85, 86, 87, 88] in
  def PTX#version: FeaturePTX<version>;

//===----------------------------------------------------------------------===//
// NVPTX supported processors.
//===----------------------------------------------------------------------===//

class Proc<string Name, list<SubtargetFeature> Features>
 : Processor<Name, NoItineraries, Features>;

def : Proc<"sm_20",   [SM20, PTX32]>;
def : Proc<"sm_21",   [SM21, PTX32]>;
def : Proc<"sm_30",   [SM30]>;
def : Proc<"sm_32",   [SM32, PTX40]>;
def : Proc<"sm_35",   [SM35, PTX32]>;
def : Proc<"sm_37",   [SM37, PTX41]>;
def : Proc<"sm_50",   [SM50, PTX40]>;
def : Proc<"sm_52",   [SM52, PTX41]>;
def : Proc<"sm_53",   [SM53, PTX42]>;
def : Proc<"sm_60",   [SM60, PTX50]>;
def : Proc<"sm_61",   [SM61, PTX50]>;
def : Proc<"sm_62",   [SM62, PTX50]>;
def : Proc<"sm_70",   [SM70, PTX60]>;
def : Proc<"sm_72",   [SM72, PTX61]>;
def : Proc<"sm_75",   [SM75, PTX63]>;
def : Proc<"sm_80",   [SM80, PTX70]>;
def : Proc<"sm_86",   [SM86, PTX71]>;
def : Proc<"sm_87",   [SM87, PTX74]>;
def : Proc<"sm_89",   [SM89, PTX78]>;
def : Proc<"sm_90",   [SM90, PTX78]>;
def : Proc<"sm_90a",  [SM90a, PTX80]>;
def : Proc<"sm_100",  [SM100, PTX86]>;
def : Proc<"sm_100a", [SM100a, PTX86]>;
def : Proc<"sm_100f", [SM100f, PTX88]>;
def : Proc<"sm_101",  [SM101, PTX86]>;
def : Proc<"sm_101a", [SM101a, PTX86]>;
def : Proc<"sm_101f", [SM101f, PTX88]>;
def : Proc<"sm_103",  [SM103, PTX88]>;
def : Proc<"sm_103a", [SM103a, PTX88]>;
def : Proc<"sm_103f", [SM103f, PTX88]>;
def : Proc<"sm_120",  [SM120, PTX87]>;
def : Proc<"sm_120a", [SM120a, PTX87]>;
def : Proc<"sm_120f", [SM120f, PTX88]>;
def : Proc<"sm_121",  [SM121, PTX88]>;
def : Proc<"sm_121a", [SM121a, PTX88]>;
def : Proc<"sm_121f", [SM121f, PTX88]>;

def NVPTXInstrInfo : InstrInfo {
}

def NVPTX : Target {
  let InstructionSet = NVPTXInstrInfo;
}
