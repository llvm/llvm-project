//===-- Next32.td - Describe the Next32 Target Machine -----*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a target description file for the Next32Silicon architecture,
// referred to here as the "NS" architecture.
//
//===----------------------------------------------------------------------===//

// Get the target-independent interfaces which we are implementing...
//
include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//
include "Next32RegisterInfo.td"
include "Next32CallingConv.td"
include "Next32InstrInfo.td"

// This feature will be enabled for "next32gen1" and "next32gen2" once we have
// support for 64-bit shifts in codegraph.
def FeatureLongShift : SubtargetFeature<"shift64", "HasLongShift", "true",
                                        "Support 64-bit shift instructions">;

def FeatureLEA
    : SubtargetFeature<"lea", "HasLEA", "true", "Support LEA instruction">;

def FeaturePrefetch : SubtargetFeature<"prefetch", "HasPrefetch", "true",
                                       "Support Prefetch instruction">;

def FeatureGen1
    : SubtargetFeature<"Gen1", "IsGen1", "true", "Enable Gen1 instructions">;

def FeatureGen2
    : SubtargetFeature<"Gen2", "IsGen2", "true", "Enable Gen2 instructions">;

def FeatureVectorInst : SubtargetFeature<"vector-inst", "HasVectorInst", "true",
                                         "Enable vector instructions">;

def FeatureAtomicFAddFSub
    : SubtargetFeature<"atomicfaddfsub", "HasAtomicFAddFSub", "true",
                       "Support Atomic FAdd and FSub">;

class Proc<string Name, list<SubtargetFeature> Features>
    : Processor<Name, NoItineraries, Features>;

def : Proc<"next32gen1", [FeatureGen1, FeatureLEA, FeaturePrefetch]>;
def : Proc<"next32gen2", [
  FeatureGen2, FeatureLEA, FeaturePrefetch, FeatureVectorInst,
  FeatureAtomicFAddFSub
]>;

def Next32InstrInfo : InstrInfo;

def Next32InstPrinter : AsmWriter {
  string AsmWriterClassName = "InstPrinter";
  bit isMCAsmWriter = 1;
}

def Next32AsmParserVariant : AsmParserVariant {
  int Variant = 0;
  string Name = "NS";
  string BreakCharacters = ".";
}

def Next32 : Target {
  let InstructionSet = Next32InstrInfo;
  let AssemblyWriters = [Next32InstPrinter];
  let AssemblyParserVariants = [Next32AsmParserVariant];
}
