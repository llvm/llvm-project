//===-- AMDIL.td - AMDIL Tablegen files --*- tablegen -*-------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// Subtarget Features
//===----------------------------------------------------------------------===//

// Debugging Features

def FeatureDumpCode : SubtargetFeature <"DumpCode",
        "DumpCode",
        "true",
        "Dump MachineInstrs in the CodeEmitter">;

def FeatureIRStructurizer : SubtargetFeature <"disable-irstructurizer",
        "EnableIRStructurizer",
        "false",
        "Disable IR Structurizer">;

def FeaturePromoteAlloca : SubtargetFeature <"promote-alloca",
        "EnablePromoteAlloca",
        "true",
        "Enable promote alloca pass">;

// Target features

def FeatureIfCvt : SubtargetFeature <"disable-ifcvt",
        "EnableIfCvt",
        "false",
        "Disable the if conversion pass">;

def FeatureFP64 : SubtargetFeature<"fp64",
        "FP64",
        "true",
        "Enable double precision operations">;

def FeatureFP64Denormals : SubtargetFeature<"fp64-denormals",
        "FP64Denormals",
        "true",
        "Enable double precision denormal handling",
        [FeatureFP64]>;

// Some instructions do not support denormals despite this flag. Using
// fp32 denormals also causes instructions to run at the double
// precision rate for the device.
def FeatureFP32Denormals : SubtargetFeature<"fp32-denormals",
        "FP32Denormals",
        "true",
        "Enable single precision denormal handling">;

def Feature64BitPtr : SubtargetFeature<"64BitPtr",
        "Is64bit",
        "true",
        "Specify if 64-bit addressing should be used">;

def FeatureR600ALUInst : SubtargetFeature<"R600ALUInst",
        "R600ALUInst",
        "false",
        "Older version of ALU instructions encoding">;

def FeatureVertexCache : SubtargetFeature<"HasVertexCache",
        "HasVertexCache",
        "true",
        "Specify use of dedicated vertex cache">;

def FeatureCaymanISA : SubtargetFeature<"caymanISA",
        "CaymanISA",
        "true",
        "Use Cayman ISA">;

def FeatureCFALUBug : SubtargetFeature<"cfalubug",
        "CFALUBug",
        "true",
        "GPU has CF_ALU bug">;

// XXX - This should probably be removed once enabled by default
def FeatureEnableLoadStoreOpt : SubtargetFeature <"load-store-opt",
        "EnableLoadStoreOpt",
        "true",
        "Enable SI load/store optimizer pass">;

def FeatureFlatAddressSpace : SubtargetFeature<"flat-address-space",
        "FlatAddressSpace",
        "true",
        "Support flat address space">;

def FeatureVGPRSpilling : SubtargetFeature<"vgpr-spilling",
        "EnableVGPRSpilling",
        "true",
        "Enable spilling of VGPRs to scratch memory">;

def FeatureSGPRInitBug : SubtargetFeature<"sgpr-init-bug",
        "SGPRInitBug",
        "true",
        "VI SGPR initilization bug requiring a fixed SGPR allocation size">;

class SubtargetFeatureFetchLimit <string Value> :
                          SubtargetFeature <"fetch"#Value,
        "TexVTXClauseSize",
        Value,
        "Limit the maximum number of fetches in a clause to "#Value>;

def FeatureFetchLimit8 : SubtargetFeatureFetchLimit <"8">;
def FeatureFetchLimit16 : SubtargetFeatureFetchLimit <"16">;

class SubtargetFeatureWavefrontSize <int Value> : SubtargetFeature<
        "wavefrontsize"#Value,
        "WavefrontSize",
        !cast<string>(Value),
        "The number of threads per wavefront">;

def FeatureWavefrontSize16 : SubtargetFeatureWavefrontSize<16>;
def FeatureWavefrontSize32 : SubtargetFeatureWavefrontSize<32>;
def FeatureWavefrontSize64 : SubtargetFeatureWavefrontSize<64>;

class SubtargetFeatureLocalMemorySize <int Value> : SubtargetFeature<
        "localmemorysize"#Value,
        "LocalMemorySize",
        !cast<string>(Value),
        "The size of local memory in bytes">;

class SubtargetFeatureGeneration <string Value,
                                  list<SubtargetFeature> Implies> :
        SubtargetFeature <Value, "Gen", "AMDGPUSubtarget::"#Value,
                          Value#" GPU generation", Implies>;

def FeatureLocalMemorySize0 : SubtargetFeatureLocalMemorySize<0>;
def FeatureLocalMemorySize32768 : SubtargetFeatureLocalMemorySize<32768>;
def FeatureLocalMemorySize65536 : SubtargetFeatureLocalMemorySize<65536>;

def FeatureR600 : SubtargetFeatureGeneration<"R600",
        [FeatureR600ALUInst, FeatureFetchLimit8, FeatureLocalMemorySize0]>;

def FeatureR700 : SubtargetFeatureGeneration<"R700",
        [FeatureFetchLimit16, FeatureLocalMemorySize0]>;

def FeatureEvergreen : SubtargetFeatureGeneration<"EVERGREEN",
        [FeatureFetchLimit16, FeatureLocalMemorySize32768]>;

def FeatureNorthernIslands : SubtargetFeatureGeneration<"NORTHERN_ISLANDS",
        [FeatureFetchLimit16, FeatureWavefrontSize64,
         FeatureLocalMemorySize32768]
>;

def FeatureSouthernIslands : SubtargetFeatureGeneration<"SOUTHERN_ISLANDS",
        [Feature64BitPtr, FeatureFP64, FeatureLocalMemorySize32768,
         FeatureWavefrontSize64]>;

def FeatureSeaIslands : SubtargetFeatureGeneration<"SEA_ISLANDS",
        [Feature64BitPtr, FeatureFP64, FeatureLocalMemorySize65536,
         FeatureWavefrontSize64, FeatureFlatAddressSpace]>;

def FeatureVolcanicIslands : SubtargetFeatureGeneration<"VOLCANIC_ISLANDS",
        [Feature64BitPtr, FeatureFP64, FeatureLocalMemorySize65536,
         FeatureWavefrontSize64, FeatureFlatAddressSpace]>;

//===----------------------------------------------------------------------===//

def AMDGPUInstrInfo : InstrInfo {
  let guessInstructionProperties = 1;
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

class PredicateControl {
  Predicate SubtargetPredicate;
  list<Predicate> OtherPredicates = [];
  list<Predicate> Predicates = !listconcat([SubtargetPredicate],
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
