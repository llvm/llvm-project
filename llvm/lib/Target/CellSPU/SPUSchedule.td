//===- SPUSchedule.td - Cell Scheduling Definitions --------*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Even pipeline:

def EVEN_UNIT : FuncUnit;       // Even execution unit: (PC & 0x7 == 000)
def ODD_UNIT  : FuncUnit;       // Odd execution unit:  (PC & 0x7 == 100)

//===----------------------------------------------------------------------===//
// Instruction Itinerary classes used for Cell SPU
//===----------------------------------------------------------------------===//

def LoadStore    : InstrItinClass;              // ODD_UNIT
def BranchHints  : InstrItinClass;              // ODD_UNIT
def BranchResolv : InstrItinClass;              // ODD_UNIT
def ChanOpSPR    : InstrItinClass;              // ODD_UNIT
def ShuffleOp    : InstrItinClass;              // ODD_UNIT
def SelectOp     : InstrItinClass;              // ODD_UNIT
def GatherOp     : InstrItinClass;              // ODD_UNIT
def LoadNOP      : InstrItinClass;              // ODD_UNIT
def ExecNOP      : InstrItinClass;              // EVEN_UNIT
def SPrecFP      : InstrItinClass;              // EVEN_UNIT
def DPrecFP      : InstrItinClass;              // EVEN_UNIT
def FPInt        : InstrItinClass;              // EVEN_UNIT (FP<->integer)
def ByteOp       : InstrItinClass;              // EVEN_UNIT
def IntegerOp    : InstrItinClass;              // EVEN_UNIT
def IntegerMulDiv: InstrItinClass;              // EVEN_UNIT
def RotateShift  : InstrItinClass;              // EVEN_UNIT
def ImmLoad      : InstrItinClass;              // EVEN_UNIT

/* Note: The itinerary for the Cell SPU is somewhat contrived... */
def SPUItineraries : ProcessorItineraries<[
  InstrItinData<LoadStore   , [InstrStage<6,  [ODD_UNIT]>]>,
  InstrItinData<BranchHints , [InstrStage<6,  [ODD_UNIT]>]>,
  InstrItinData<BranchResolv, [InstrStage<4,  [ODD_UNIT]>]>,
  InstrItinData<ChanOpSPR   , [InstrStage<6,  [ODD_UNIT]>]>,
  InstrItinData<ShuffleOp   , [InstrStage<4,  [ODD_UNIT]>]>,
  InstrItinData<SelectOp    , [InstrStage<4,  [ODD_UNIT]>]>,
  InstrItinData<GatherOp    , [InstrStage<4,  [ODD_UNIT]>]>,
  InstrItinData<LoadNOP     , [InstrStage<1,  [ODD_UNIT]>]>,
  InstrItinData<ExecNOP     , [InstrStage<1,  [EVEN_UNIT]>]>,
  InstrItinData<SPrecFP     , [InstrStage<6,  [EVEN_UNIT]>]>,
  InstrItinData<DPrecFP     , [InstrStage<13, [EVEN_UNIT]>]>,
  InstrItinData<FPInt       , [InstrStage<2,  [EVEN_UNIT]>]>,
  InstrItinData<ByteOp      , [InstrStage<4,  [EVEN_UNIT]>]>,
  InstrItinData<IntegerOp   , [InstrStage<2,  [EVEN_UNIT]>]>,
  InstrItinData<RotateShift , [InstrStage<4,  [EVEN_UNIT]>]>,
  InstrItinData<IntegerMulDiv,[InstrStage<7,  [EVEN_UNIT]>]>,
  InstrItinData<ImmLoad     , [InstrStage<2,  [EVEN_UNIT]>]>
  ]>;
