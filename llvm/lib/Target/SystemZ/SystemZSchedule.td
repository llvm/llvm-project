//==-- SystemZSchedule.td - SystemZ Scheduling Definitions ----*- tblgen -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Scheduler resources
// Resources ending with a '2' use that resource for 2 cycles. An instruction
// using two such resources use the mapped unit for 4 cycles, and 2 is added
// to the total number of uops of the sched class.

// These three resources are used to express decoder grouping rules.
// The number of decoder slots needed by an instructions is normally
// one. For a cracked instruction (BeginGroup && !EndGroup) it is
// two. Expanded instructions (BeginGroup && EndGroup) group alone.
def GroupAlone : SchedWrite;
def BeginGroup : SchedWrite;
def EndGroup   : SchedWrite;

// Latencies, to make code a bit neater. If more than one resource is
// used for an instruction, the greatest latency (not the sum) will be
// output by Tablegen. Therefore, in such cases one of these resources
// is needed.
def Lat2 : SchedWrite;
def Lat3 : SchedWrite;
def Lat4 : SchedWrite;
def Lat5 : SchedWrite;
def Lat6 : SchedWrite;
def Lat7 : SchedWrite;
def Lat8 : SchedWrite;
def Lat9 : SchedWrite;
def Lat10 : SchedWrite;
def Lat11 : SchedWrite;
def Lat12 : SchedWrite;
def Lat15 : SchedWrite;
def Lat20 : SchedWrite;
def Lat30 : SchedWrite;

// Fixed-point
def FXa         : SchedWrite;
def FXa2        : SchedWrite;
def FXb         : SchedWrite;
def FXU         : SchedWrite;

// Load/store unit
def LSU         : SchedWrite;

// Model a return without latency, otherwise if-converter will model
// extra cost and abort (currently there is an assert that checks that
// all instructions have at least one uop).
def LSU_lat1    : SchedWrite;

// Floating point unit (zEC12 and earlier)
def FPU  : SchedWrite;
def FPU2 : SchedWrite;
def DFU  : SchedWrite;
def DFU2 : SchedWrite;

// Vector sub units (z13)
def VecBF     : SchedWrite;
def VecBF2    : SchedWrite;
def VecDF     : SchedWrite;
def VecDF2    : SchedWrite;
def VecDFX    : SchedWrite;
def VecDFX2   : SchedWrite;
def VecFPd    : SchedWrite; // Blocking BFP div/sqrt unit.
def VecMul    : SchedWrite;
def VecStr    : SchedWrite;
def VecXsPm   : SchedWrite;

// Virtual branching unit
def VBU         : SchedWrite;


include "SystemZScheduleZ13.td"
include "SystemZScheduleZEC12.td"
include "SystemZScheduleZ196.td"

