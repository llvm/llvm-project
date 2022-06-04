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
  // The M88100 can issue 1 instructsions.
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

  def : InstRW<[M88100IntUWrite], (instregex "(ADD(U)?|SUB(U)?|CLR|MAK|EXT(U)?|AND|OR|XOR)")>;

  // The data unit: load, store and exchange instructions.
  def M88100DataU : ProcResource<1> { let BufferSize = 0; }

  def M88100DataUWrite : SchedWriteRes<[M88100DataU]> {
    let Latency = 3;
  }

  def : InstRW<[M88100DataUWrite], (instregex "(LD|ST|XMEM)")>;
}

def M88110SchedModel : SchedMachineModel {
  // The M88110 can issue 2 instructsions.
  let IssueWidth = 2;

  // In-order CPU.
  let MicroOpBufferSize = 0;

  list<Predicate> UnsupportedFeatures = [IsMC881100];

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
