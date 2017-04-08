//==- AArch64SchedFalkor.td - Falkor Scheduling Definitions -*- tablegen -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the machine model for Qualcomm Falkor to support
// instruction scheduling and other instruction cost heuristics.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Define the SchedMachineModel and provide basic properties for coarse grained
// instruction cost model.

def FalkorModel : SchedMachineModel {
  let IssueWidth = 8;          // 8 uops are dispatched per cycle.
  let MicroOpBufferSize = 128; // Out-of-order with temporary unified issue buffer.
  let LoopMicroOpBufferSize = 16;
  let LoadLatency = 3;         // Optimistic load latency.
  let MispredictPenalty = 11;  // Minimum branch misprediction penalty.
  let CompleteModel = 1;
}

//===----------------------------------------------------------------------===//
// Define each kind of processor resource and number available on Falkor.

let SchedModel = FalkorModel in {

  def FalkorUnitB    : ProcResource<1>; // Branch
  def FalkorUnitLD   : ProcResource<1>; // Load pipe
  def FalkorUnitSD   : ProcResource<1>; // Store data
  def FalkorUnitST   : ProcResource<1>; // Store pipe
  def FalkorUnitX    : ProcResource<1>; // Complex arithmetic
  def FalkorUnitY    : ProcResource<1>; // Simple arithmetic
  def FalkorUnitZ    : ProcResource<1>; // Simple arithmetic

  def FalkorUnitVSD  : ProcResource<1>; // Vector store data
  def FalkorUnitVX   : ProcResource<1>; // Vector X-pipe
  def FalkorUnitVY   : ProcResource<1>; // Vector Y-pipe

  def FalkorUnitGTOV : ProcResource<1>; // Scalar to Vector
  def FalkorUnitVTOG : ProcResource<1>; // Vector to Scalar

  // Define the resource groups.
  def FalkorUnitXY   : ProcResGroup<[FalkorUnitX, FalkorUnitY]>;
  def FalkorUnitXYZ  : ProcResGroup<[FalkorUnitX, FalkorUnitY, FalkorUnitZ]>;
  def FalkorUnitXYZB : ProcResGroup<[FalkorUnitX, FalkorUnitY, FalkorUnitZ,
                                     FalkorUnitB]>;
  def FalkorUnitZB   : ProcResGroup<[FalkorUnitZ, FalkorUnitB]>;
  def FalkorUnitVXVY : ProcResGroup<[FalkorUnitVX, FalkorUnitVY]>;

}

//===----------------------------------------------------------------------===//
// Map the target-defined scheduler read/write resources and latency for
// Falkor.

let SchedModel = FalkorModel in {

def : WriteRes<WriteImm,   [FalkorUnitXYZ]> { let Latency = 1; }
def : WriteRes<WriteI,     [FalkorUnitXYZ]> { let Latency = 1; }
def : WriteRes<WriteISReg, [FalkorUnitVXVY, FalkorUnitVXVY]>
      { let Latency = 1; let NumMicroOps = 2; }
def : WriteRes<WriteIEReg, [FalkorUnitXYZ, FalkorUnitXYZ]>
      { let Latency = 2; let NumMicroOps = 2; }
def : WriteRes<WriteExtr,  [FalkorUnitXYZ, FalkorUnitXYZ]>
      { let Latency = 2; let NumMicroOps = 2; }
def : WriteRes<WriteIS,    [FalkorUnitXYZ]> { let Latency = 1; }
def : WriteRes<WriteID32,  [FalkorUnitX, FalkorUnitZ]>
      { let Latency = 8; let NumMicroOps = 2; }
def : WriteRes<WriteID64,  [FalkorUnitX, FalkorUnitZ]>
      { let Latency = 16; let NumMicroOps = 2; }
def : WriteRes<WriteIM32,  [FalkorUnitX]> { let Latency = 4; }
def : WriteRes<WriteIM64,  [FalkorUnitX]> { let Latency = 5; }
def : WriteRes<WriteBr,    [FalkorUnitB]> { let Latency = 1; }
def : WriteRes<WriteBrReg, [FalkorUnitB]> { let Latency = 1; }
def : WriteRes<WriteLD,    [FalkorUnitLD]> { let Latency = 3; }
def : WriteRes<WriteST,    [FalkorUnitLD, FalkorUnitST, FalkorUnitSD]>
      { let Latency = 3; let NumMicroOps = 3; }
def : WriteRes<WriteSTP,   [FalkorUnitST, FalkorUnitSD]>
      { let Latency = 0; let NumMicroOps = 2; }
def : WriteRes<WriteAdr,   [FalkorUnitXYZ]> { let Latency = 5; }
def : WriteRes<WriteLDIdx, [FalkorUnitLD]> { let Latency = 5; }
def : WriteRes<WriteSTIdx, [FalkorUnitLD, FalkorUnitST, FalkorUnitSD]>
      { let Latency = 4; let NumMicroOps = 3; }
def : WriteRes<WriteF,     [FalkorUnitVXVY, FalkorUnitVXVY]>
      { let Latency = 3; let NumMicroOps = 2; }
def : WriteRes<WriteFCmp,  [FalkorUnitVXVY]> { let Latency = 2; }
def : WriteRes<WriteFCvt,  [FalkorUnitVXVY]> { let Latency = 4; }
def : WriteRes<WriteFCopy, [FalkorUnitVXVY]> { let Latency = 4; }
def : WriteRes<WriteFImm,  [FalkorUnitVXVY]> { let Latency = 4; }
def : WriteRes<WriteFMul,  [FalkorUnitVXVY, FalkorUnitVXVY]>
      { let Latency = 6; let NumMicroOps = 2; }
def : WriteRes<WriteFDiv,  [FalkorUnitVXVY, FalkorUnitVXVY]>
      { let Latency = 12; let NumMicroOps = 2; } // Fragent -1 / NoRSV +1
def : WriteRes<WriteV,     [FalkorUnitVXVY]> { let Latency = 6; }
def : WriteRes<WriteVLD,   [FalkorUnitLD]> { let Latency = 3; }
def : WriteRes<WriteVST,   [FalkorUnitST, FalkorUnitVSD]>
      { let Latency = 0; let NumMicroOps = 2; }

def : WriteRes<WriteSys,     []> { let Latency = 1; }
def : WriteRes<WriteBarrier, []> { let Latency = 1; }
def : WriteRes<WriteHint,    []> { let Latency = 1; }

def : WriteRes<WriteLDHi,    []> { let Latency = 3; }

def : WriteRes<WriteAtomic, []> { let Unsupported = 1; }

// No forwarding logic is modelled yet.
def : ReadAdvance<ReadI,       0>;
def : ReadAdvance<ReadISReg,   0>;
def : ReadAdvance<ReadIEReg,   0>;
def : ReadAdvance<ReadIM,      0>;
def : ReadAdvance<ReadIMA,     0>;
def : ReadAdvance<ReadID,      0>;
def : ReadAdvance<ReadExtrHi,  0>;
def : ReadAdvance<ReadAdrBase, 0>;
def : ReadAdvance<ReadVLD,     0>;

// Detailed Refinements
// -----------------------------------------------------------------------------
include "AArch64SchedFalkorDetails.td"

}
