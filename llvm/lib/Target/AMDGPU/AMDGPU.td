//===-- AMDGPU.td - AMDGPU Tablegen files --------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===------------------------------------------------------------===//
// Subtarget Features (device properties)
//===------------------------------------------------------------===//

def FeatureFP64 : SubtargetFeature<"fp64",
  "FP64",
  "true",
  "Enable double precision operations"
>;

def FeatureFastFMAF32 : SubtargetFeature<"fast-fmaf",
  "FastFMAF32",
  "true",
  "Assuming f32 fma is at least as fast as mul + add"
>;

def HalfRate64Ops : SubtargetFeature<"half-rate-64-ops",
  "HalfRate64Ops",
  "true",
  "Most fp64 instructions are half rate instead of quarter"
>;

def FeatureR600ALUInst : SubtargetFeature<"R600ALUInst",
  "R600ALUInst",
  "false",
  "Older version of ALU instructions encoding"
>;

def FeatureVertexCache : SubtargetFeature<"HasVertexCache",
  "HasVertexCache",
  "true",
  "Specify use of dedicated vertex cache"
>;

def FeatureCaymanISA : SubtargetFeature<"caymanISA",
  "CaymanISA",
  "true",
  "Use Cayman ISA"
>;

def FeatureCFALUBug : SubtargetFeature<"cfalubug",
  "CFALUBug",
  "true",
  "GPU has CF_ALU bug"
>;

def FeatureFlatAddressSpace : SubtargetFeature<"flat-address-space",
  "FlatAddressSpace",
  "true",
  "Support flat address space"
>;

def FeatureXNACK : SubtargetFeature<"xnack",
  "EnableXNACK",
  "true",
  "Enable XNACK support"
>;

def FeatureSGPRInitBug : SubtargetFeature<"sgpr-init-bug",
  "SGPRInitBug",
  "true",
  "VI SGPR initilization bug requiring a fixed SGPR allocation size"
>;

class SubtargetFeatureFetchLimit <string Value> :
                          SubtargetFeature <"fetch"#Value,
  "TexVTXClauseSize",
  Value,
  "Limit the maximum number of fetches in a clause to "#Value
>;

def FeatureFetchLimit8 : SubtargetFeatureFetchLimit <"8">;
def FeatureFetchLimit16 : SubtargetFeatureFetchLimit <"16">;

class SubtargetFeatureWavefrontSize <int Value> : SubtargetFeature<
  "wavefrontsize"#Value,
  "WavefrontSize",
  !cast<string>(Value),
  "The number of threads per wavefront"
>;

def FeatureWavefrontSize16 : SubtargetFeatureWavefrontSize<16>;
def FeatureWavefrontSize32 : SubtargetFeatureWavefrontSize<32>;
def FeatureWavefrontSize64 : SubtargetFeatureWavefrontSize<64>;

class SubtargetFeatureLDSBankCount <int Value> : SubtargetFeature <
  "ldsbankcount"#Value,
  "LDSBankCount",
  !cast<string>(Value),
  "The number of LDS banks per compute unit."
>;

def FeatureLDSBankCount16 : SubtargetFeatureLDSBankCount<16>;
def FeatureLDSBankCount32 : SubtargetFeatureLDSBankCount<32>;

class SubtargetFeatureISAVersion <int Major, int Minor, int Stepping>
                                 : SubtargetFeature <
  "isaver"#Major#"."#Minor#"."#Stepping,
  "IsaVersion",
  "ISAVersion"#Major#"_"#Minor#"_"#Stepping,
  "Instruction set version number"
>;

def FeatureISAVersion7_0_0 : SubtargetFeatureISAVersion <7,0,0>;
def FeatureISAVersion7_0_1 : SubtargetFeatureISAVersion <7,0,1>;
def FeatureISAVersion8_0_0 : SubtargetFeatureISAVersion <8,0,0>;
def FeatureISAVersion8_0_1 : SubtargetFeatureISAVersion <8,0,1>;
def FeatureISAVersion8_0_3 : SubtargetFeatureISAVersion <8,0,3>;

class SubtargetFeatureLocalMemorySize <int Value> : SubtargetFeature<
  "localmemorysize"#Value,
  "LocalMemorySize",
  !cast<string>(Value),
  "The size of local memory in bytes"
>;

def FeatureGCN : SubtargetFeature<"gcn",
  "IsGCN",
  "true",
  "GCN or newer GPU"
>;

def FeatureGCN1Encoding : SubtargetFeature<"gcn1-encoding",
  "GCN1Encoding",
  "true",
  "Encoding format for SI and CI"
>;

def FeatureGCN3Encoding : SubtargetFeature<"gcn3-encoding",
  "GCN3Encoding",
  "true",
  "Encoding format for VI"
>;

def FeatureCIInsts : SubtargetFeature<"ci-insts",
  "CIInsts",
  "true",
  "Additional intstructions for CI+"
>;

//===------------------------------------------------------------===//
// Subtarget Features (options and debugging)
//===------------------------------------------------------------===//

// Some instructions do not support denormals despite this flag. Using
// fp32 denormals also causes instructions to run at the double
// precision rate for the device.
def FeatureFP32Denormals : SubtargetFeature<"fp32-denormals",
  "FP32Denormals",
  "true",
  "Enable single precision denormal handling"
>;

def FeatureFP64Denormals : SubtargetFeature<"fp64-denormals",
  "FP64Denormals",
  "true",
  "Enable double precision denormal handling",
  [FeatureFP64]
>;

def FeatureFPExceptions : SubtargetFeature<"fp-exceptions",
  "FPExceptions",
  "true",
  "Enable floating point exceptions"
>;

class FeatureMaxPrivateElementSize<int size> : SubtargetFeature<
  "max-private-element-size-"#size,
  "MaxPrivateElementSize",
  !cast<string>(size),
  "Maximum private access size may be "#size
>;

def FeatureMaxPrivateElementSize4 : FeatureMaxPrivateElementSize<4>;
def FeatureMaxPrivateElementSize8 : FeatureMaxPrivateElementSize<8>;
def FeatureMaxPrivateElementSize16 : FeatureMaxPrivateElementSize<16>;


def FeatureEnableHugeScratchBuffer : SubtargetFeature<
  "huge-scratch-buffer",
  "EnableHugeScratchBuffer",
  "true",
  "Enable scratch buffer sizes greater than 128 GB"
>;

def FeatureVGPRSpilling : SubtargetFeature<"vgpr-spilling",
  "EnableVGPRSpilling",
  "true",
  "Enable spilling of VGPRs to scratch memory"
>;

def FeatureDumpCode : SubtargetFeature <"DumpCode",
  "DumpCode",
  "true",
  "Dump MachineInstrs in the CodeEmitter"
>;

def FeatureDumpCodeLower : SubtargetFeature <"dumpcode",
  "DumpCode",
  "true",
  "Dump MachineInstrs in the CodeEmitter"
>;

def FeatureIRStructurizer : SubtargetFeature <"disable-irstructurizer",
  "EnableIRStructurizer",
  "false",
  "Disable IR Structurizer"
>;

def FeaturePromoteAlloca : SubtargetFeature <"promote-alloca",
  "EnablePromoteAlloca",
  "true",
  "Enable promote alloca pass"
>;

// XXX - This should probably be removed once enabled by default
def FeatureEnableLoadStoreOpt : SubtargetFeature <"load-store-opt",
  "EnableLoadStoreOpt",
  "true",
  "Enable SI load/store optimizer pass"
>;

// Performance debugging feature. Allow using DS instruction immediate
// offsets even if the base pointer can't be proven to be base. On SI,
// base pointer values that won't give the same result as a 16-bit add
// are not safe to fold, but this will override the conservative test
// for the base pointer.
def FeatureEnableUnsafeDSOffsetFolding : SubtargetFeature <
  "unsafe-ds-offset-folding",
  "EnableUnsafeDSOffsetFolding",
  "true",
  "Force using DS instruction immediate offsets on SI"
>;

def FeatureIfCvt : SubtargetFeature <"disable-ifcvt",
  "EnableIfCvt",
  "false",
  "Disable the if conversion pass"
>;

def FeatureEnableSIScheduler : SubtargetFeature<"si-scheduler",
  "EnableSIScheduler",
  "true",
  "Enable SI Machine Scheduler"
>;

def FeatureFlatForGlobal : SubtargetFeature<"flat-for-global",
  "FlatForGlobal",
  "true",
  "Force to generate flat instruction for global"
>;

// Dummy feature used to disable assembler instructions.
def FeatureDisable : SubtargetFeature<"",
  "FeatureDisable","true",
  "Dummy feature to disable assembler instructions"
>;

class SubtargetFeatureGeneration <string Value,
                                  list<SubtargetFeature> Implies> :
        SubtargetFeature <Value, "Gen", "AMDGPUSubtarget::"#Value,
                          Value#" GPU generation", Implies>;

def FeatureLocalMemorySize0 : SubtargetFeatureLocalMemorySize<0>;
def FeatureLocalMemorySize32768 : SubtargetFeatureLocalMemorySize<32768>;
def FeatureLocalMemorySize65536 : SubtargetFeatureLocalMemorySize<65536>;

def FeatureR600 : SubtargetFeatureGeneration<"R600",
  [FeatureR600ALUInst, FeatureFetchLimit8, FeatureLocalMemorySize0]
>;

def FeatureR700 : SubtargetFeatureGeneration<"R700",
  [FeatureFetchLimit16, FeatureLocalMemorySize0]
>;

def FeatureEvergreen : SubtargetFeatureGeneration<"EVERGREEN",
  [FeatureFetchLimit16, FeatureLocalMemorySize32768]
>;

def FeatureNorthernIslands : SubtargetFeatureGeneration<"NORTHERN_ISLANDS",
  [FeatureFetchLimit16, FeatureWavefrontSize64,
   FeatureLocalMemorySize32768]
>;

def FeatureSouthernIslands : SubtargetFeatureGeneration<"SOUTHERN_ISLANDS",
  [FeatureFP64, FeatureLocalMemorySize32768,
  FeatureWavefrontSize64, FeatureGCN, FeatureGCN1Encoding,
  FeatureLDSBankCount32]
>;

def FeatureSeaIslands : SubtargetFeatureGeneration<"SEA_ISLANDS",
  [FeatureFP64, FeatureLocalMemorySize65536,
  FeatureWavefrontSize64, FeatureGCN, FeatureFlatAddressSpace,
  FeatureGCN1Encoding, FeatureCIInsts]
>;

def FeatureVolcanicIslands : SubtargetFeatureGeneration<"VOLCANIC_ISLANDS",
  [FeatureFP64, FeatureLocalMemorySize65536,
   FeatureWavefrontSize64, FeatureFlatAddressSpace, FeatureGCN,
   FeatureGCN3Encoding, FeatureCIInsts]
>;

//===----------------------------------------------------------------------===//

def AMDGPUInstrInfo : InstrInfo {
  let guessInstructionProperties = 1;
  let noNamedPositionallyEncodedOperands = 1;
}

def AMDGPUAsmParser : AsmParser {
  // Some of the R600 registers have the same name, so this crashes.
  // For example T0_XYZW and T0_XY both have the asm name T0.
  let ShouldEmitMatchRegisterName = 0;
}

def AMDGPU : Target {
  // Pull in Instruction Info:
  let InstructionSet = AMDGPUInstrInfo;
  let AssemblyParsers = [AMDGPUAsmParser];
}

// Dummy Instruction itineraries for pseudo instructions
def ALU_NULL : FuncUnit;
def NullALU : InstrItinClass;

//===----------------------------------------------------------------------===//
// Predicate helper class
//===----------------------------------------------------------------------===//

def TruePredicate : Predicate<"true">;

def isSICI : Predicate<
  "Subtarget->getGeneration() == AMDGPUSubtarget::SOUTHERN_ISLANDS ||"
  "Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS"
>, AssemblerPredicate<"FeatureGCN1Encoding">;

def isVI : Predicate <
  "Subtarget->getGeneration() >= AMDGPUSubtarget::VOLCANIC_ISLANDS">,
  AssemblerPredicate<"FeatureGCN3Encoding">;

def isCIVI : Predicate <
  "Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS || "
  "Subtarget->getGeneration() == AMDGPUSubtarget::VOLCANIC_ISLANDS"
>, AssemblerPredicate<"FeatureCIInsts">;

def HasFlatAddressSpace : Predicate<"Subtarget->hasFlatAddressSpace()">;

class PredicateControl {
  Predicate SubtargetPredicate;
  Predicate SIAssemblerPredicate = isSICI;
  Predicate VIAssemblerPredicate = isVI;
  list<Predicate> AssemblerPredicates = [];
  Predicate AssemblerPredicate = TruePredicate;
  list<Predicate> OtherPredicates = [];
  list<Predicate> Predicates = !listconcat([SubtargetPredicate, AssemblerPredicate],
                                            AssemblerPredicates,
                                            OtherPredicates);
}

// Include AMDGPU TD files
include "R600Schedule.td"
include "SISchedule.td"
include "Processors.td"
include "AMDGPUInstrInfo.td"
include "AMDGPUIntrinsics.td"
include "AMDGPURegisterInfo.td"
include "AMDGPUInstructions.td"
include "AMDGPUCallingConv.td"
