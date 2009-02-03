//===- AlphaSchedule.td - Alpha Scheduling Definitions -----*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//

//This is table 2-2 from the 21264 compiler writers guide
//modified some

//Pipelines

def L0   : FuncUnit;
def L1   : FuncUnit;
def FST0 : FuncUnit;
def FST1 : FuncUnit;
def U0   : FuncUnit;
def U1   : FuncUnit;
def FA   : FuncUnit;
def FM   : FuncUnit;

def s_ild   : InstrItinClass;
def s_fld   : InstrItinClass;
def s_ist   : InstrItinClass;
def s_fst   : InstrItinClass;
def s_lda   : InstrItinClass;
def s_rpcc  : InstrItinClass;
def s_rx    : InstrItinClass;
def s_mxpr  : InstrItinClass;
def s_icbr  : InstrItinClass;
def s_ubr   : InstrItinClass;
def s_jsr   : InstrItinClass;
def s_iadd  : InstrItinClass;
def s_ilog  : InstrItinClass;
def s_ishf  : InstrItinClass;
def s_cmov  : InstrItinClass;
def s_imul  : InstrItinClass;
def s_imisc : InstrItinClass;
def s_fbr   : InstrItinClass;
def s_fadd  : InstrItinClass;
def s_fmul  : InstrItinClass;
def s_fcmov : InstrItinClass;
def s_fdivt : InstrItinClass;
def s_fdivs : InstrItinClass;
def s_fsqrts: InstrItinClass;
def s_fsqrtt: InstrItinClass;
def s_ftoi  : InstrItinClass;
def s_itof  : InstrItinClass;
def s_pseudo : InstrItinClass;

//Table 2­4 Instruction Class Latency in Cycles
//modified some

def Alpha21264Itineraries : ProcessorItineraries<[
  InstrItinData<s_ild    , [InstrStage<3, [L0, L1]>]>,
  InstrItinData<s_fld    , [InstrStage<4, [L0, L1]>]>,
  InstrItinData<s_ist    , [InstrStage<0, [L0, L1]>]>,
  InstrItinData<s_fst    , [InstrStage<0, [FST0, FST1, L0, L1]>]>,
  InstrItinData<s_lda    , [InstrStage<1, [L0, L1, U0, U1]>]>,
  InstrItinData<s_rpcc   , [InstrStage<1, [L1]>]>,
  InstrItinData<s_rx     , [InstrStage<1, [L1]>]>,
  InstrItinData<s_mxpr   , [InstrStage<1, [L0, L1]>]>,
  InstrItinData<s_icbr   , [InstrStage<0, [U0, U1]>]>,
  InstrItinData<s_ubr    , [InstrStage<3, [U0, U1]>]>,
  InstrItinData<s_jsr    , [InstrStage<3, [L0]>]>,
  InstrItinData<s_iadd   , [InstrStage<1, [L0, U0, L1, U1]>]>,
  InstrItinData<s_ilog   , [InstrStage<1, [L0, U0, L1, U1]>]>,
  InstrItinData<s_ishf   , [InstrStage<1, [U0, U1]>]>,
  InstrItinData<s_cmov   , [InstrStage<1, [L0, U0, L1, U1]>]>,
  InstrItinData<s_imul   , [InstrStage<7, [U1]>]>,
  InstrItinData<s_imisc  , [InstrStage<3, [U0]>]>,
  InstrItinData<s_fbr    , [InstrStage<0, [FA]>]>,
  InstrItinData<s_fadd   , [InstrStage<6, [FA]>]>,
  InstrItinData<s_fmul   , [InstrStage<6, [FM]>]>,
  InstrItinData<s_fcmov  , [InstrStage<6, [FA]>]>,
  InstrItinData<s_fdivs  , [InstrStage<12, [FA]>]>,
  InstrItinData<s_fdivt  , [InstrStage<15, [FA]>]>,
  InstrItinData<s_fsqrts , [InstrStage<18, [FA]>]>,
  InstrItinData<s_fsqrtt , [InstrStage<33, [FA]>]>,
  InstrItinData<s_ftoi   , [InstrStage<3, [FST0, FST1, L0, L1]>]>,
  InstrItinData<s_itof   , [InstrStage<4, [L0, L1]>]>
]>;
