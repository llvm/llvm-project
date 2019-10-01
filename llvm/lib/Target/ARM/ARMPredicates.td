//===-- ARMPredicates.td - ARM Instruction Predicates ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

def HasV4T           : Predicate<"Subtarget->hasV4TOps()">,
                                 AssemblerPredicate<"HasV4TOps", "armv4t">;
def NoV4T            : Predicate<"!Subtarget->hasV4TOps()">;
def HasV5T           : Predicate<"Subtarget->hasV5TOps()">,
                                 AssemblerPredicate<"HasV5TOps", "armv5t">;
def NoV5T            : Predicate<"!Subtarget->hasV5TOps()">;
def HasV5TE          : Predicate<"Subtarget->hasV5TEOps()">,
                                 AssemblerPredicate<"HasV5TEOps", "armv5te">;
def HasV6            : Predicate<"Subtarget->hasV6Ops()">,
                                 AssemblerPredicate<"HasV6Ops", "armv6">;
def NoV6             : Predicate<"!Subtarget->hasV6Ops()">;
def HasV6M           : Predicate<"Subtarget->hasV6MOps()">,
                                 AssemblerPredicate<"HasV6MOps",
                                                    "armv6m or armv6t2">;
def HasV8MBaseline   : Predicate<"Subtarget->hasV8MBaselineOps()">,
                                 AssemblerPredicate<"HasV8MBaselineOps",
                                                    "armv8m.base">;
def HasV8MMainline   : Predicate<"Subtarget->hasV8MMainlineOps()">,
                                 AssemblerPredicate<"HasV8MMainlineOps",
                                                    "armv8m.main">;
def HasV8_1MMainline : Predicate<"Subtarget->hasV8_1MMainlineOps()">,
                                 AssemblerPredicate<"HasV8_1MMainlineOps",
                                                    "armv8.1m.main">;
def HasMVEInt        : Predicate<"Subtarget->hasMVEIntegerOps()">,
                                 AssemblerPredicate<"HasMVEIntegerOps",
                                                    "mve">;
def HasMVEFloat      : Predicate<"Subtarget->hasMVEFloatOps()">,
                                 AssemblerPredicate<"HasMVEFloatOps",
                                                    "mve.fp">;
def HasFPRegs        : Predicate<"Subtarget->hasFPRegs()">,
                                 AssemblerPredicate<"FeatureFPRegs",
                                                    "fp registers">;
def HasFPRegs16      : Predicate<"Subtarget->hasFPRegs16()">,
                                 AssemblerPredicate<"FeatureFPRegs16",
                                                    "16-bit fp registers">;
def HasFPRegs64      : Predicate<"Subtarget->hasFPRegs64()">,
                                 AssemblerPredicate<"FeatureFPRegs64",
                                                    "64-bit fp registers">;
def HasFPRegsV8_1M   : Predicate<"Subtarget->hasFPRegs() && Subtarget->hasV8_1MMainlineOps()">,
                                 AssemblerPredicate<"FeatureFPRegs,HasV8_1MMainlineOps",
                                                    "armv8.1m.main with FP or MVE">;
def HasV6T2          : Predicate<"Subtarget->hasV6T2Ops()">,
                                 AssemblerPredicate<"HasV6T2Ops", "armv6t2">;
def NoV6T2           : Predicate<"!Subtarget->hasV6T2Ops()">;
def HasV6K           : Predicate<"Subtarget->hasV6KOps()">,
                                 AssemblerPredicate<"HasV6KOps", "armv6k">;
def NoV6K            : Predicate<"!Subtarget->hasV6KOps()">;
def HasV7            : Predicate<"Subtarget->hasV7Ops()">,
                                 AssemblerPredicate<"HasV7Ops", "armv7">;
def HasV8            : Predicate<"Subtarget->hasV8Ops()">,
                                 AssemblerPredicate<"HasV8Ops", "armv8">;
def PreV8            : Predicate<"!Subtarget->hasV8Ops()">,
                                 AssemblerPredicate<"!HasV8Ops", "armv7 or earlier">;
def HasV8_1a         : Predicate<"Subtarget->hasV8_1aOps()">,
                                 AssemblerPredicate<"HasV8_1aOps", "armv8.1a">;
def HasV8_2a         : Predicate<"Subtarget->hasV8_2aOps()">,
                                 AssemblerPredicate<"HasV8_2aOps", "armv8.2a">;
def HasV8_3a         : Predicate<"Subtarget->hasV8_3aOps()">,
                                 AssemblerPredicate<"HasV8_3aOps", "armv8.3a">;
def HasV8_4a         : Predicate<"Subtarget->hasV8_4aOps()">,
                                 AssemblerPredicate<"HasV8_4aOps", "armv8.4a">;
def HasV8_5a         : Predicate<"Subtarget->hasV8_5aOps()">,
                                 AssemblerPredicate<"HasV8_5aOps", "armv8.5a">;
def NoVFP            : Predicate<"!Subtarget->hasVFP2Base()">;
def HasVFP2          : Predicate<"Subtarget->hasVFP2Base()">,
                                 AssemblerPredicate<"FeatureVFP2_SP", "VFP2">;
def HasVFP3          : Predicate<"Subtarget->hasVFP3Base()">,
                                 AssemblerPredicate<"FeatureVFP3_D16_SP", "VFP3">;
def HasVFP4          : Predicate<"Subtarget->hasVFP4Base()">,
                                 AssemblerPredicate<"FeatureVFP4_D16_SP", "VFP4">;
def HasDPVFP         : Predicate<"Subtarget->hasFP64()">,
                                 AssemblerPredicate<"FeatureFP64",
                                                    "double precision VFP">;
def HasFPARMv8       : Predicate<"Subtarget->hasFPARMv8Base()">,
                                 AssemblerPredicate<"FeatureFPARMv8_D16_SP", "FPARMv8">;
def HasNEON          : Predicate<"Subtarget->hasNEON()">,
                                 AssemblerPredicate<"FeatureNEON", "NEON">;
def HasSHA2          : Predicate<"Subtarget->hasSHA2()">,
                                 AssemblerPredicate<"FeatureSHA2", "sha2">;
def HasAES           : Predicate<"Subtarget->hasAES()">,
                                 AssemblerPredicate<"FeatureAES", "aes">;
def HasCrypto        : Predicate<"Subtarget->hasCrypto()">,
                                 AssemblerPredicate<"FeatureCrypto", "crypto">;
def HasDotProd       : Predicate<"Subtarget->hasDotProd()">,
                                 AssemblerPredicate<"FeatureDotProd", "dotprod">;
def HasCRC           : Predicate<"Subtarget->hasCRC()">,
                                 AssemblerPredicate<"FeatureCRC", "crc">;
def HasRAS           : Predicate<"Subtarget->hasRAS()">,
                                 AssemblerPredicate<"FeatureRAS", "ras">;
def HasLOB           : Predicate<"Subtarget->hasLOB()">,
                                 AssemblerPredicate<"FeatureLOB", "lob">;
def HasFP16          : Predicate<"Subtarget->hasFP16()">,
                                 AssemblerPredicate<"FeatureFP16","half-float conversions">;
def HasFullFP16      : Predicate<"Subtarget->hasFullFP16()">,
                                 AssemblerPredicate<"FeatureFullFP16","full half-float">;
def HasFP16FML       : Predicate<"Subtarget->hasFP16FML()">,
                                 AssemblerPredicate<"FeatureFP16FML","full half-float fml">;
def HasDivideInThumb : Predicate<"Subtarget->hasDivideInThumbMode()">,
                                 AssemblerPredicate<"FeatureHWDivThumb", "divide in THUMB">;
def HasDivideInARM   : Predicate<"Subtarget->hasDivideInARMMode()">,
                                 AssemblerPredicate<"FeatureHWDivARM", "divide in ARM">;
def HasDSP           : Predicate<"Subtarget->hasDSP()">,
                                 AssemblerPredicate<"FeatureDSP", "dsp">;
def HasDB            : Predicate<"Subtarget->hasDataBarrier()">,
                                 AssemblerPredicate<"FeatureDB",
                                                    "data-barriers">;
def HasDFB           : Predicate<"Subtarget->hasFullDataBarrier()">,
                                 AssemblerPredicate<"FeatureDFB",
                                                    "full-data-barrier">;
def HasV7Clrex  : Predicate<"Subtarget->hasV7Clrex()">,
                            AssemblerPredicate<"FeatureV7Clrex",
                                               "v7 clrex">;
def HasAcquireRelease : Predicate<"Subtarget->hasAcquireRelease()">,
                                  AssemblerPredicate<"FeatureAcquireRelease",
                                                     "acquire/release">;
def HasMP            : Predicate<"Subtarget->hasMPExtension()">,
                                 AssemblerPredicate<"FeatureMP",
                                                    "mp-extensions">;
def HasVirtualization: Predicate<"false">,
                                 AssemblerPredicate<"FeatureVirtualization",
                                                   "virtualization-extensions">;
def HasTrustZone     : Predicate<"Subtarget->hasTrustZone()">,
                                 AssemblerPredicate<"FeatureTrustZone",
                                                    "TrustZone">;
def Has8MSecExt      : Predicate<"Subtarget->has8MSecExt()">,
                                 AssemblerPredicate<"Feature8MSecExt",
                                                    "ARMv8-M Security Extensions">;
def HasZCZ           : Predicate<"Subtarget->hasZeroCycleZeroing()">;
def UseNEONForFP     : Predicate<"Subtarget->useNEONForSinglePrecisionFP()">;
def DontUseNEONForFP : Predicate<"!Subtarget->useNEONForSinglePrecisionFP()">;
def IsThumb          : Predicate<"Subtarget->isThumb()">,
                                 AssemblerPredicate<"ModeThumb", "thumb">;
def IsThumb1Only     : Predicate<"Subtarget->isThumb1Only()">;
def IsThumb2         : Predicate<"Subtarget->isThumb2()">,
                                 AssemblerPredicate<"ModeThumb,FeatureThumb2",
                                                    "thumb2">;
def IsMClass         : Predicate<"Subtarget->isMClass()">,
                                 AssemblerPredicate<"FeatureMClass", "armv*m">;
def IsNotMClass      : Predicate<"!Subtarget->isMClass()">,
                                 AssemblerPredicate<"!FeatureMClass",
                                                    "!armv*m">;
def IsARM            : Predicate<"!Subtarget->isThumb()">,
                                 AssemblerPredicate<"!ModeThumb", "arm-mode">;
def IsMachO          : Predicate<"Subtarget->isTargetMachO()">;
def IsNotMachO       : Predicate<"!Subtarget->isTargetMachO()">;
def IsNaCl           : Predicate<"Subtarget->isTargetNaCl()">;
def IsWindows        : Predicate<"Subtarget->isTargetWindows()">;
def IsNotWindows     : Predicate<"!Subtarget->isTargetWindows()">;
def IsReadTPHard     : Predicate<"Subtarget->isReadTPHard()">;
def IsReadTPSoft     : Predicate<"!Subtarget->isReadTPHard()">;
def UseNaClTrap      : Predicate<"Subtarget->useNaClTrap()">,
                                 AssemblerPredicate<"FeatureNaClTrap", "NaCl">;
def DontUseNaClTrap  : Predicate<"!Subtarget->useNaClTrap()">;

def UseNegativeImmediates :
  Predicate<"false">,
            AssemblerPredicate<"!FeatureNoNegativeImmediates",
                               "NegativeImmediates">;

// FIXME: Eventually this will be just "hasV6T2Ops".
let RecomputePerFunction = 1 in {
  def UseMovt          : Predicate<"Subtarget->useMovt()">;
  def DontUseMovt      : Predicate<"!Subtarget->useMovt()">;
  def UseMovtInPic     : Predicate<"Subtarget->useMovt() && Subtarget->allowPositionIndependentMovt()">;
  def DontUseMovtInPic : Predicate<"!Subtarget->useMovt() || !Subtarget->allowPositionIndependentMovt()">;

  def UseFPVMLx: Predicate<"((Subtarget->useFPVMLx() &&"
                           "  TM.Options.AllowFPOpFusion != FPOpFusion::Fast) ||"
                           "Subtarget->hasMinSize())">;
}
def UseMulOps        : Predicate<"Subtarget->useMulOps()">;

// Prefer fused MAC for fp mul + add over fp VMLA / VMLS if they are available.
// But only select them if more precision in FP computation is allowed, and when
// they are not slower than a mul + add sequence.
// Do not use them for Darwin platforms.
def UseFusedMAC      : Predicate<"(TM.Options.AllowFPOpFusion =="
                                 " FPOpFusion::Fast && "
                                 " Subtarget->hasVFP4Base()) && "
                                 "!Subtarget->isTargetDarwin() &&"
                                 "Subtarget->useFPVMLx()">;

def HasFastVGETLNi32 : Predicate<"!Subtarget->hasSlowVGETLNi32()">;
def HasSlowVGETLNi32 : Predicate<"Subtarget->hasSlowVGETLNi32()">;

def HasFastVDUP32 : Predicate<"!Subtarget->hasSlowVDUP32()">;
def HasSlowVDUP32 : Predicate<"Subtarget->hasSlowVDUP32()">;

def UseVMOVSR : Predicate<"Subtarget->preferVMOVSR() ||"
                          "!Subtarget->useNEONForSinglePrecisionFP()">;
def DontUseVMOVSR : Predicate<"!Subtarget->preferVMOVSR() &&"
                              "Subtarget->useNEONForSinglePrecisionFP()">;

let RecomputePerFunction = 1 in {
  def IsLE             : Predicate<"MF->getDataLayout().isLittleEndian()">;
  def IsBE             : Predicate<"MF->getDataLayout().isBigEndian()">;
}

def GenExecuteOnly : Predicate<"Subtarget->genExecuteOnly()">;

// Armv8.5-A extensions
def HasSB            : Predicate<"Subtarget->hasSB()">,
                       AssemblerPredicate<"FeatureSB", "sb">;
