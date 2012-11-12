//===-- MBlazeSchedule.td - MBlaze Scheduling Definitions --*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MBlaze functional units.
//===----------------------------------------------------------------------===//
def IF : FuncUnit;
def ID : FuncUnit;
def EX : FuncUnit;
def MA : FuncUnit;
def WB : FuncUnit;

//===----------------------------------------------------------------------===//
// Instruction Itinerary classes used for MBlaze
//===----------------------------------------------------------------------===//
def IIC_ALU    : InstrItinClass;
def IIC_ALUm   : InstrItinClass;
def IIC_ALUd   : InstrItinClass;
def IIC_SHT    : InstrItinClass;
def IIC_FSLg   : InstrItinClass;
def IIC_FSLp   : InstrItinClass;
def IIC_MEMs   : InstrItinClass;
def IIC_MEMl   : InstrItinClass;
def IIC_FPU    : InstrItinClass;
def IIC_FPUd   : InstrItinClass;
def IIC_FPUf   : InstrItinClass;
def IIC_FPUi   : InstrItinClass;
def IIC_FPUs   : InstrItinClass;
def IIC_FPUc   : InstrItinClass;
def IIC_BR     : InstrItinClass;
def IIC_BRc    : InstrItinClass;
def IIC_BRl    : InstrItinClass;
def IIC_WDC    : InstrItinClass;
def IIC_Pseudo : InstrItinClass;

//===----------------------------------------------------------------------===//
// MBlaze instruction itineraries for three stage pipeline.
//===----------------------------------------------------------------------===//
include "MBlazeSchedule3.td"

//===----------------------------------------------------------------------===//
// MBlaze instruction itineraries for five stage pipeline.
//===----------------------------------------------------------------------===//
include "MBlazeSchedule5.td"
