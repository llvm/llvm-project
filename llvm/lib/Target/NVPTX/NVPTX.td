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
// Encoding := Arch * 10 + ArchSuffixOffset
// Arch := X * 10 + Y
// ArchSuffixOffset := 0 (base), 2 ('f'), or 3 ('a')
//
// For example, sm_103a is encoded as 1033 (103 * 10 + 3) and sm_103f is
// encoded as 1032 (103 * 10 + 2).
//
// This encoding allows simple partial ordering of the architectures.
//  + Compare Family and Arch by dividing FullSMVersion by 100 and 10
//    respectively before the comparison.
//  + Compare within the family by comparing FullSMVersion, given both belongs to
//    the same family.
//  + Detect 'a' variants by checking FullSMVersion & 1.
class Proc<FeatureSM SM>
 : Processor<SM.Name, NoItineraries, [SM]>;

foreach sm = [20, 21, 30, 32, 35, 37, 50, 52, 53, 60,
              61, 62, 70, 72, 75, 80, 86, 87, 88, 89,
              90, 100, 101, 103, 110, 120, 121] in {
  // Base SM version (e.g. FullSMVersion for sm_100 is 1000)
  def SM#sm : FeatureSM<""#sm, !mul(sm, 10)>;
  def : Proc<!cast<FeatureSM>("SM"#sm)>;

  // Family-specific variants, compatible within same family (e.g. sm_100f = 1002)
  if !ge(sm, 100) then {
    def SM#sm#f : FeatureSM<""#sm#"f", !add(!mul(sm, 10), 2)>;
    def : Proc<!cast<FeatureSM>("SM"#sm#"f")>;
  }

  // Architecture-specific variants, incompatible across architectures (e.g. sm_100a = 1003)
  if !ge(sm, 90) then {
    def SM#sm#a : FeatureSM<""#sm#"a", !add(!mul(sm, 10), 3)>;
    def : Proc<!cast<FeatureSM>("SM"#sm#"a")>;
  }
}

foreach version = [32, 40, 41, 42, 43, 50, 60, 61, 62, 63, 64, 65, 70, 71, 72,
                   73, 74, 75, 76, 77, 78, 80, 81, 82, 83, 84, 85, 86, 87, 88,
                   90, 91] in
  def PTX#version : FeaturePTX<version>;

def Is64Bit : Predicate<"Subtarget->getTargetTriple().getArch() == Triple::nvptx64">;
def NVPTX64 : HwMode<[Is64Bit]>;

def nvptx_ptr_rc : RegClassByHwMode<
  [DefaultMode, NVPTX64],
  [B32, B64]>;

defm : RemapAllTargetPseudoPointerOperands<nvptx_ptr_rc>;

def NVPTXInstrInfo : InstrInfo {
}

def NVPTX : Target {
  let InstructionSet = NVPTXInstrInfo;
}
