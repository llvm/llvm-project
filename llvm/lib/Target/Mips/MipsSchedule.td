//===-- MipsSchedule.td - Mips Scheduling Definitions ------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Functional units across Mips chips sets. Based on GCC/Mips backend files.
//===----------------------------------------------------------------------===//
def ALU     : FuncUnit;
def IMULDIV : FuncUnit;

//===----------------------------------------------------------------------===//
// Instruction Itinerary classes used for Mips
//===----------------------------------------------------------------------===//
def IIAlu              : InstrItinClass;
def IILoad             : InstrItinClass;
def IIStore            : InstrItinClass;
def IIXfer             : InstrItinClass;
def IIBranch           : InstrItinClass;
def IIHiLo             : InstrItinClass;
def IIImul             : InstrItinClass;
def IIIdiv             : InstrItinClass;
def IIFcvt             : InstrItinClass;
def IIFmove            : InstrItinClass;
def IIFcmp             : InstrItinClass;
def IIFadd             : InstrItinClass;
def IIFmulSingle       : InstrItinClass;
def IIFmulDouble       : InstrItinClass;
def IIFdivSingle       : InstrItinClass;
def IIFdivDouble       : InstrItinClass;
def IIFsqrtSingle      : InstrItinClass;
def IIFsqrtDouble      : InstrItinClass;
def IIFrecipFsqrtStep  : InstrItinClass;
def IIPseudo           : InstrItinClass;

//===----------------------------------------------------------------------===//
// Mips Generic instruction itineraries.
//===----------------------------------------------------------------------===//
def MipsGenericItineraries : ProcessorItineraries<[ALU, IMULDIV], [], [
  InstrItinData<IIAlu              , [InstrStage<1,  [ALU]>]>,
  InstrItinData<IILoad             , [InstrStage<3,  [ALU]>]>,
  InstrItinData<IIStore            , [InstrStage<1,  [ALU]>]>,
  InstrItinData<IIXfer             , [InstrStage<2,  [ALU]>]>,
  InstrItinData<IIBranch           , [InstrStage<1,  [ALU]>]>,
  InstrItinData<IIHiLo             , [InstrStage<1,  [IMULDIV]>]>,
  InstrItinData<IIImul             , [InstrStage<17, [IMULDIV]>]>,
  InstrItinData<IIIdiv             , [InstrStage<38, [IMULDIV]>]>,
  InstrItinData<IIFcvt             , [InstrStage<1,  [ALU]>]>,
  InstrItinData<IIFmove            , [InstrStage<2,  [ALU]>]>,
  InstrItinData<IIFcmp             , [InstrStage<3,  [ALU]>]>,
  InstrItinData<IIFadd             , [InstrStage<4,  [ALU]>]>,
  InstrItinData<IIFmulSingle       , [InstrStage<7,  [ALU]>]>,
  InstrItinData<IIFmulDouble       , [InstrStage<8,  [ALU]>]>,
  InstrItinData<IIFdivSingle       , [InstrStage<23, [ALU]>]>,
  InstrItinData<IIFdivDouble       , [InstrStage<36, [ALU]>]>,
  InstrItinData<IIFsqrtSingle      , [InstrStage<54, [ALU]>]>,
  InstrItinData<IIFsqrtDouble      , [InstrStage<12, [ALU]>]>,
  InstrItinData<IIFrecipFsqrtStep  , [InstrStage<5,  [ALU]>]>
]>;
