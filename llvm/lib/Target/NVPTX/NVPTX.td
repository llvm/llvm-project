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
// NVPTX Architecture Hierarchy and Ordering:
//
// GPU architectures: sm_2Y/sm_3Y/sm_5Y/sm_6Y/sm_7Y/sm_8Y/sm_9Y/sm_10Y/sm_12Y
// ('Y' represents version within the architecture)
// The architectures have name of form sm_XYz where 'X' represent the generation
// number, 'Y' represents the version within the architecture, and 'z' represents
// the optional feature suffix.
// If X1Y1 <= X2Y2, then GPU capabilities of sm_X1Y1 are included in sm_X2Y2.
// For example, take sm_90 (9 represents 'X', 0 represents 'Y', and no feature
// suffix) and sm_103 architectures (10 represents 'X', 3 represents 'Y', and no
// feature suffix). Since 90 <= 103, sm_90 is compatible with sm_103.
//
// The family-specific variants have 'f' feature suffix and they follow
// following order:
// sm_X{Y2}f > sm_X{Y1}f iff Y2 > Y1
// sm_XY{f} > sm_{XY}{}
//
// For example, take sm_100f (10 represents 'X', 0 represents 'Y', and 'f'
// represents 'z') and sm_103f (10 represents 'X', 3 represents 'Y', and 'f'
// represents 'z') architecture variants. Since Y1 < Y2, sm_100f is compatible with
// sm_103f. Similarly based on the second rule, sm_90 is compatible with sm_103f.
//
// Some counter examples, take sm_100f and sm_120f (12 represents 'X', 0
// represents 'Y', and 'f' represents 'z') architecture variants. Since both
// belongs to different family i.e. X1 != X2, sm_100f is not compatible with
// sm_120f.
//
// The architecture-specific variants have 'a' feature suffix and they follow
// following order:
// sm_XY{a} > sm_XY{f} > sm_{XY}{}
//
// For example, take sm_103a (10 represents 'X', 3 represents 'Y', and 'a'
// represents 'z'), sm_103f, and sm_103 architecture variants. The sm_103 is
// compatible with sm_103a and sm_103f, and sm_103f is compatible with sm_103a.
//
// Encoding := Arch * 10 + 2 (for 'f') + 1 (for 'a')
// Arch := X * 10 + Y
//
// For example, sm_103a is encoded as 1033 (103 * 10 + 2 + 1) and sm_103f is
// encoded as 1032 (103 * 10 + 2).
//
// This encoding allows simple partial ordering of the architectures.
//  + Compare Family and Arch by dividing FullSMVersion by 100 and 10
//    respectively before the comparison.
//  + Compare within the family by comparing FullSMVersion, given both belongs to
//    the same family.
//  + Detect 'a' variants by checking FullSMVersion & 1.
foreach sm = [20, 21, 30, 32, 35, 37, 50, 52, 53,
              60, 61, 62, 70, 72, 75, 80, 86, 87,
              89, 90, 100, 101, 103, 120, 121] in {
  // Base SM version (e.g. FullSMVersion for sm_100 is 1000)
  def SM#sm : FeatureSM<""#sm, !mul(sm, 10)>;

  // Family-specific targets which are compatible within same family
  // (e.g. FullSMVersion for sm_100f is 1002)
  if !ge(sm, 100) then
    def SM#sm#f : FeatureSM<""#sm#"f", !add(!mul(sm, 10), 2)>;

  // Architecture-specific targets which are incompatible across architectures
  // (e.g. FullSMVersion for sm_100a is 1003)
  if !ge(sm, 90) then
    def SM#sm#a : FeatureSM<""#sm#"a", !add(!mul(sm, 10), 3)>;
}

foreach version = [32, 40, 41, 42, 43, 50, 60, 61, 62, 63, 64, 65, 70, 71, 72,
                   73, 74, 75, 76, 77, 78, 80, 81, 82, 83, 84, 85, 86, 87, 88,
                   90] in
  def PTX#version : FeaturePTX<version>;

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
