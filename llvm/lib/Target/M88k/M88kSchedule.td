//===-- M88kSchedule.td - M88k Scheduling Model ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

def "IU" : SchedWrite;    // Integer Unit
def "DU" : SchedWrite;    // Data Unit
def "FPadd" : SchedWrite; // Floating Point Add Unit
def "FPmul" : SchedWrite; // Floating Point Mul Unit

def M88100SchedModel : SchedMachineModel {
  // The M88100 can issue 1 instructions.
  let IssueWidth = 1;

  // In-order CPU.
  let MicroOpBufferSize = 0;

  int LoadLatency = 3; // Assumes no wait cycle on the bus.

  // Not finished yet.
  let CompleteModel = 0;
  let NoModel = 1;
}

let SchedModel = M88100SchedModel in {
  // The integer unit: integer arithmetic, logical and bitfield instructions.
  def M88100IntU : ProcResource<1> { let BufferSize = 0; }

  def M88100IntUWrite : SchedWriteRes<[M88100IntU]>;

  def : InstRW<[M88100IntUWrite], (instregex "^(ADD(U)?|CMP|SUB(U)?|CLR|EXT(U)?|FF(0|1)|MAK|ROT|SET|AND|MASK|OR|XOR)")>;

  // The data unit: load, store and exchange instructions.
  // The unit has 3 stages.
  def M88100DataU : ProcResource<1> { let BufferSize = 0; }

  def M88100DataUWrite : SchedWriteRes<[M88100DataU]> {
    let Latency = 3;
  }

  def : InstRW<[M88100DataUWrite], (instregex "^(LD|ST|XMEM)")>;

  // The FP add unit: addition/subtraction/compare/divide instructions.
  // The unit has 5 stages.
  def M88100FPAddU : ProcResource<1> { let BufferSize = 0; }
  def M88100FPAddWrite : SchedWriteRes<[M88100FPAddU]> {
    let Latency = 5;
  }
  def M88100Div : SchedWriteRes<[M88100FPAddU]> {
    let Latency = 38;
    let ResourceCycles = [33];
  }
  def M88100DivS : SchedWriteRes<[M88100FPAddU]> {
    let Latency = 30;
    let ResourceCycles = [25];
  }
  def M88100DivD : SchedWriteRes<[M88100FPAddU]> {
    let Latency = 60;
    let ResourceCycles = [54];
  }
  def : InstRW<[M88100FPAddWrite], (instregex "^(FADD|FSUB|FCMP|FLT|INT|NINT|TRNC)")>;
  def : InstRW<[M88100Div], (instregex "^DIV(U)?")>;
  def : InstRW<[M88100DivS], (instregex "^FDIVgs")>;
  def : InstRW<[M88100DivD], (instregex "^FDIVgd")>;

  // The FP mul unit: multiplication instructions.
  // The unit has 6 stages.
  def M88100FPMulU : ProcResource<1> { let BufferSize = 0; }
  def M88100Mul : SchedWriteRes<[M88100FPMulU]> {
    let Latency = 4;
  }
  def M88100MulS : SchedWriteRes<[M88100FPMulU]> {
    let Latency = 6;
  }
  def M88100MulD : SchedWriteRes<[M88100FPMulU]> {
    let Latency = 7;
  }
  def : InstRW<[M88100Mul], (instregex "^MUL")>;
  def : InstRW<[M88100MulS], (instregex "^FMULgs")>;
  def : InstRW<[M88100MulD], (instregex "^FMULgd")>;
}

def M88110SchedModel : SchedMachineModel {
  // The M88110 can issue 2 instructsions.
  let IssueWidth = 2;

  // In-order CPU.
  let MicroOpBufferSize = 0;

  list<Predicate> UnsupportedFeatures = [MC88110];

  // Not finished yet.
  let CompleteModel = 0;
  let NoModel = 1;
}

def ALU : ProcResource<2> {let BufferSize = 0; }
def BF :  ProcResource<1> {let BufferSize = 0; }
//def LS : ProcResource<1> { let BufferSize = 0; }

def ReadALU   : SchedRead;
def ReadBF    : SchedRead;
def WriteALU  : SchedWrite;
def WriteBF   : SchedWrite;
//def WriteLS   : SchedWrite;

let SchedModel = M88110SchedModel in {
  def : WriteRes<WriteALU, [ALU]>   { let Latency = 1; }
  def : WriteRes<WriteBF, [BF]>     { let Latency = 1; }
}
