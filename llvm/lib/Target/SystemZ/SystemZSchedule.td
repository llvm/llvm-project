//==-- SystemZSchedule.td - SystemZ Scheduling Definitions ----*- tblgen -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Scheduler resources

// These resources are used to express decoder grouping rules.  The number of
// decoder slots needed by an instructions is normally one, but there are
// exceptions.
def NormalGr   : SchedWrite;
def Cracked    : SchedWrite;
def GroupAlone : SchedWrite;
def BeginGroup : SchedWrite;
def EndGroup   : SchedWrite;

// A SchedWrite added to other SchedWrites to make LSU latency parameterizable.
def LSULatency : SchedWrite;

// Operand WriteLatencies.
def WLat1 : SchedWrite;
def WLat2 : SchedWrite;
def WLat3 : SchedWrite;
def WLat4 : SchedWrite;
def WLat5 : SchedWrite;
def WLat6 : SchedWrite;
def WLat7 : SchedWrite;
def WLat8 : SchedWrite;
def WLat9 : SchedWrite;
def WLat10 : SchedWrite;
def WLat11 : SchedWrite;
def WLat12 : SchedWrite;
def WLat15 : SchedWrite;
def WLat16 : SchedWrite;
def WLat20 : SchedWrite;
def WLat26 : SchedWrite;
def WLat30 : SchedWrite;

def WLat1LSU : WriteSequence<[WLat1, LSULatency]>;
def WLat2LSU : WriteSequence<[WLat2, LSULatency]>;
def WLat3LSU : WriteSequence<[WLat3, LSULatency]>;
def WLat4LSU : WriteSequence<[WLat4, LSULatency]>;
def WLat6LSU : WriteSequence<[WLat6, LSULatency]>;
def WLat5LSU : WriteSequence<[WLat5, LSULatency]>;
def WLat7LSU : WriteSequence<[WLat7, LSULatency]>;
def WLat8LSU : WriteSequence<[WLat8, LSULatency]>;
def WLat11LSU : WriteSequence<[WLat11, LSULatency]>;
def WLat16LSU : WriteSequence<[WLat16, LSULatency]>;

// ReadAdvances, used for the register operand next to a memory operand,
// modelling that the register operand is needed later than the address
// operands.
def RegReadAdv : SchedRead;

// Fixed-point units
def FXa         : SchedWrite;
def FXa2        : SchedWrite;
def FXa3        : SchedWrite;
def FXa4        : SchedWrite;
def FXb         : SchedWrite;
def FXb2        : SchedWrite;
def FXb3        : SchedWrite;
def FXb4        : SchedWrite;
def FXb5        : SchedWrite;
def FXU         : SchedWrite;
def FXU2        : SchedWrite;
def FXU3        : SchedWrite;
def FXU4        : SchedWrite;
def FXU5        : SchedWrite;
def FXU6        : SchedWrite;

// Load/store unit
def LSU         : SchedWrite;
def LSU2        : SchedWrite;
def LSU3        : SchedWrite;
def LSU4        : SchedWrite;
def LSU5        : SchedWrite;

// Floating point unit (zEC12 and earlier)
def FPU  : SchedWrite;
def FPU2 : SchedWrite;
def FPU4 : SchedWrite;
def DFU  : SchedWrite;
def DFU2 : SchedWrite;
def DFU4 : SchedWrite;

// Vector sub units (z13 and later)
def VecBF     : SchedWrite;
def VecBF2    : SchedWrite;
def VecBF4    : SchedWrite;
def VecDF     : SchedWrite;
def VecDF2    : SchedWrite;
def VecDF4    : SchedWrite;
def VecDFX    : SchedWrite;
def VecDFX2   : SchedWrite;
def VecDFX4   : SchedWrite;
def VecFPd    : SchedWrite; // Blocking BFP div/sqrt unit.
def VecMul    : SchedWrite;
def VecStr    : SchedWrite;
def VecXsPm   : SchedWrite;
def VecXsPm2  : SchedWrite;

// Virtual branching unit
def VBU       : SchedWrite;

// Millicode
def MCD       : SchedWrite;

include "SystemZScheduleZ14.td"
include "SystemZScheduleZ13.td"
include "SystemZScheduleZEC12.td"
include "SystemZScheduleZ196.td"

