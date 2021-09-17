//===-- M88kSchedule.td - M88k Scheduling Model ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

def M88kSchedModel : SchedMachineModel {
  // The M88110 can issue 2 instructsions.
  let IssueWidth = 2;

  // In-order CPU.
  let MicroOpBufferSize = 0;

  //let LoadLatency = ?;
  //let MispredictPenalty = ?;

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

let SchedModel = M88kSchedModel in {
  def : WriteRes<WriteALU, [ALU]>   { let Latency = 1; }
  def : WriteRes<WriteBF, [BF]>     { let Latency = 1; }
}
