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
  let IssueWidth = 4;          // 4-wide issue for expanded uops.
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

  // Define the resource groups.
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
      { let Latency = 8; let NumMicroOps = 1; } // Fragent -1
def : WriteRes<WriteID64,  [FalkorUnitX, FalkorUnitZ]>
      { let Latency = 8; let NumMicroOps = 1; } // Fragent -1
def : WriteRes<WriteIM32,  [FalkorUnitX]> { let Latency = 4; }
def : WriteRes<WriteIM64,  [FalkorUnitX]> { let Latency = 4; }
def : WriteRes<WriteBr,    [FalkorUnitB]> { let Latency = 1; }
def : WriteRes<WriteBrReg, [FalkorUnitB]> { let Latency = 1; }
def : WriteRes<WriteLD,    [FalkorUnitLD]> { let Latency = 3; }
def : WriteRes<WriteST,    [FalkorUnitLD, FalkorUnitST, FalkorUnitSD]>
      { let Latency = 3; let NumMicroOps = 3; }
def : WriteRes<WriteSTP,   [FalkorUnitST, FalkorUnitSD]>
      { let Latency = 3; let NumMicroOps = 2; }
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
def : WriteRes<WriteVST,   [FalkorUnitST]> { let Latency = 4; }

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

//===----------------------------------------------------------------------===//
// Specialize the coarse model by associating instruction groups with the
// subtarget-defined types. As the modeled is refined, this will override most
// of the above SchedWriteRes and SchedAlias mappings.

// -----------------------------------------------------------------------------
// Miscellaneous
// -----------------------------------------------------------------------------

def : InstRW<[WriteI], (instrs COPY)>;

// -----------------------------------------------------------------------------
// Vector Loads
// -----------------------------------------------------------------------------
def : InstRW<[WriteVLD], (instregex "LD1i(8|16|32|64)$")>;
def : InstRW<[WriteVLD], (instregex "LD1Rv(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[WriteVLD], (instregex "LD1Onev(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[WriteVLD], (instregex "LD1Twov(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[WriteVLD], (instregex "LD1Threev(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[WriteVLD], (instregex "LD1Fourv(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[WriteVLD, WriteAdr], (instregex "LD1i(8|16|32|64)_POST$")>;
def : InstRW<[WriteVLD, WriteAdr], (instregex "LD1Rv(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteVLD, WriteAdr], (instregex "LD1Onev(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteVLD, WriteAdr], (instregex "LD1Twov(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteVLD, WriteAdr], (instregex "LD1Threev(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteVLD, WriteAdr], (instregex "LD1Fourv(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;

def : InstRW<[WriteVLD], (instregex "LD2i(8|16|32|64)$")>;
def : InstRW<[WriteVLD], (instregex "LD2Rv(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[WriteVLD], (instregex "LD2Twov(8b|4h|2s)$")>;
def : InstRW<[WriteVLD], (instregex "LD2Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteVLD, WriteAdr], (instregex "LD2i(8|16|32|64)(_POST)?$")>;
def : InstRW<[WriteVLD, WriteAdr], (instregex "LD2Rv(8b|4h|2s|1d|16b|8h|4s|2d)(_POST)?$")>;
def : InstRW<[WriteVLD, WriteAdr], (instregex "LD2Twov(8b|4h|2s)(_POST)?$")>;
def : InstRW<[WriteVLD, WriteAdr], (instregex "LD2Twov(16b|8h|4s|2d)(_POST)?$")>;

def : InstRW<[WriteVLD], (instregex "LD3i(8|16|32|64)$")>;
def : InstRW<[WriteVLD], (instregex "LD3Rv(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[WriteVLD], (instregex "LD3Threev(8b|4h|2s|1d|16b|8h|4s)$")>;
def : InstRW<[WriteVLD], (instregex "LD3Threev(2d)$")>;
def : InstRW<[WriteVLD, WriteAdr], (instregex "LD3i(8|16|32|64)_POST$")>;
def : InstRW<[WriteVLD, WriteAdr], (instregex "LD3Rv(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteVLD, WriteAdr], (instregex "LD3Threev(8b|4h|2s|1d|16b|8h|4s)_POST$")>;
def : InstRW<[WriteVLD, WriteAdr], (instregex "LD3Threev(2d)_POST$")>;

def : InstRW<[WriteVLD], (instregex "LD4i(8|16|32|64)$")>;
def : InstRW<[WriteVLD], (instregex "LD4Rv(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[WriteVLD], (instregex "LD4Fourv(8b|4h|2s|1d|16b|8h|4s)$")>;
def : InstRW<[WriteVLD], (instregex "LD4Fourv(2d)$")>;
def : InstRW<[WriteVLD, WriteAdr], (instregex "LD4i(8|16|32|64)_POST$")>;
def : InstRW<[WriteVLD, WriteAdr], (instregex "LD4Rv(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteVLD, WriteAdr], (instregex "LD4Fourv(8b|4h|2s|1d|16b|8h|4s)_POST$")>;
def : InstRW<[WriteVLD, WriteAdr], (instregex "LD4Fourv(2d)_POST$")>;

// -----------------------------------------------------------------------------
// Vector Stores
// -----------------------------------------------------------------------------
def : InstRW<[WriteVST], (instregex "ST1i(8|16|32|64)$")>;
def : InstRW<[WriteVST], (instregex "ST1Onev(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[WriteVST], (instregex "ST1Twov(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[WriteVST], (instregex "ST1Threev(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[WriteVST], (instregex "ST1Fourv(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[WriteVST, WriteAdr], (instregex "ST1i(8|16|32|64)_POST$")>;
def : InstRW<[WriteVST, WriteAdr], (instregex "ST1Onev(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteVST, WriteAdr], (instregex "ST1Twov(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteVST, WriteAdr], (instregex "ST1Threev(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteVST, WriteAdr], (instregex "ST1Fourv(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;

def : InstRW<[WriteVST], (instregex "ST2i(8|16|32|64)$")>;
def : InstRW<[WriteVST], (instregex "ST2Twov(8b|4h|2s)$")>;
def : InstRW<[WriteVST], (instregex "ST2Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteVST, WriteAdr], (instregex "ST2i(8|16|32|64)_POST$")>;
def : InstRW<[WriteVST, WriteAdr], (instregex "ST2Twov(8b|4h|2s)_POST$")>;
def : InstRW<[WriteVST, WriteAdr], (instregex "ST2Twov(16b|8h|4s|2d)_POST$")>;

def : InstRW<[WriteVST], (instregex "ST3i(8|16|32|64)$")>;
def : InstRW<[WriteVST], (instregex "ST3Threev(8b|4h|2s|1d|16b|8h|4s)$")>;
def : InstRW<[WriteVST], (instregex "ST3Threev(2d)$")>;
def : InstRW<[WriteVST, WriteAdr], (instregex "ST3i(8|16|32|64)_POST$")>;
def : InstRW<[WriteVST, WriteAdr], (instregex "ST3Threev(8b|4h|2s|1d|16b|8h|4s)_POST$")>;
def : InstRW<[WriteVST, WriteAdr], (instregex "ST3Threev(2d)_POST$")>;

def : InstRW<[WriteVST], (instregex "ST4i(8|16|32|64)$")>;
def : InstRW<[WriteVST], (instregex "ST4Fourv(8b|4h|2s|1d|16b|8h|4s)$")>;
def : InstRW<[WriteVST], (instregex "ST4Fourv(2d)$")>;
def : InstRW<[WriteVST, WriteAdr], (instregex "ST4i(8|16|32|64)_POST$")>;
def : InstRW<[WriteVST, WriteAdr], (instregex "ST4Fourv(8b|4h|2s|1d|16b|8h|4s)_POST$")>;
def : InstRW<[WriteVST, WriteAdr], (instregex "ST4Fourv(2d)_POST$")>;

}
