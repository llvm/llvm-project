//==- RISCVSchedVeeREH1.td - VeeR EH1 Scheduling Definitions -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// VeeR EH1 (formerly Western Digital SweRV EH1):
// https://github.com/chipsalliance/Cores-VeeR-EH1
//
// VeeR EH1 is a 9-stage (F1, F2, Align, Decode, EX1-EX4, WB), 2-wide in-order
// RV32IMC core. The two issue slots (i0, i1) each have a primary ALU in EX1
// and a secondary ALU in EX4. Loads, stores and multiplies are shared between
// both slots, so only one LSU op and one multiply can issue per cycle.
//
// An ALU op whose source is produced by a load or a multiply that is still in
// EX1/EX2 does not stall at decode. It executes in the EX4 secondary ALU, and
// its result is not available to non-ALU consumers (load/store address,
// multiply) until it reaches EX4. The model gives load and multiply results
// their full latency for ALU consumers, which matches the measured cost of
// load -> ALU -> address and multiply -> ALU -> address chains.
//
// A branch can issue in the same cycle as the load or multiply it depends on:
// it executes in the EX4 secondary ALU instead. There it resolves later, which
// raises its mispredict penalty from about 5 to about 8 cycles. Store data is
// read late too, but only ALU and load results are forwarded to it, a store of
// a multiply result waits for the full multiply latency.
//
// Divide and remainder use a radix-2 iterative divider with early outs for
// small quotients. They stall decode until they finish, so nothing overlaps
// with them.

def VeeREH1Model : SchedMachineModel {
  let MicroOpBufferSize = 0;
  let IssueWidth = 2;
  let LoadLatency = 3;
  let MispredictPenalty = 5;
  let CompleteModel = 0;
  let UnsupportedFeatures = [HasStdExtZbkb, HasStdExtZbkc, HasStdExtZbkx,
                             HasStdExtZknd, HasStdExtZkne, HasStdExtZknh,
                             HasStdExtZksed, HasStdExtZksh, HasStdExtZkr,
                             HasVInstructions];
}

let SchedModel = VeeREH1Model in {

// IssueWidth limits issue to the two slots. These are the functional units the
// slots use: one ALU per slot, and one LSU, multiplier and divider shared by
// both. A load or multiply pairs with an ALU op, and a load with a multiply,
// but two loads or two multiplies do not pair.
let BufferSize = 0 in {
def VeeREH1ALU0 : ProcResource<1>;
def VeeREH1ALU1 : ProcResource<1>;
def VeeREH1ALU : ProcResGroup<[VeeREH1ALU0, VeeREH1ALU1]>;
def VeeREH1LSU : ProcResource<1>;
def VeeREH1MUL : ProcResource<1>;
def VeeREH1DIV : ProcResource<1>;
}

// Branching. Branches execute in either ALU. No instruction can issue in i1
// after a JAL or JALR in i0.
def : WriteRes<WriteJmp, [VeeREH1ALU]>;
let EndGroup = true in {
def : WriteRes<WriteJal, [VeeREH1ALU]>;
def : WriteRes<WriteJalr, [VeeREH1ALU]>;
}

// Integer arithmetic and logic.
def : WriteRes<WriteIALU32, [VeeREH1ALU]>;
def : WriteRes<WriteIALU, [VeeREH1ALU]>;
def : WriteRes<WriteShiftImm32, [VeeREH1ALU]>;
def : WriteRes<WriteShiftImm, [VeeREH1ALU]>;
def : WriteRes<WriteShiftReg32, [VeeREH1ALU]>;
def : WriteRes<WriteShiftReg, [VeeREH1ALU]>;

// Results are written back at a fixed stage, so a short-latency instruction
// never waits for an older load, multiply or divide to write back.
let RetireOOO = true in {

// Integer multiplication: 3-stage pipelined multiplier, one per cycle.
let Latency = 3 in {
def : WriteRes<WriteIMul, [VeeREH1MUL]>;
def : WriteRes<WriteIMul32, [VeeREH1MUL]>;
}

// Integer division/remainder. Latency is for a full 32-bit quotient, small
// quotients finish early. Decode stalls until the divider finishes, so no
// other instruction overlaps with it. Measured on the RTL: after a worst-case
// divide the next instruction issues 38 cycles later (39 after a remainder).
// A second divide can issue a little earlier, after 35 (36), but the model
// cannot express that without releasing the other units early too, so every
// unit is held for the full latency. This deliberately approximates the decode
// stall for both the machine scheduler and llvm-mca, which therefore reports
// pressure on every unit while a divide runs.
defvar VeeREH1DivResources = [VeeREH1DIV, VeeREH1ALU0, VeeREH1ALU1,
                              VeeREH1LSU, VeeREH1MUL];
let BeginGroup = true, EndGroup = true in {
let Latency = 38, ReleaseAtCycles = [38, 38, 38, 38, 38] in {
def : WriteRes<WriteIDiv32, VeeREH1DivResources>;
def : WriteRes<WriteIDiv, VeeREH1DivResources>;
}
let Latency = 39, ReleaseAtCycles = [39, 39, 39, 39, 39] in {
def : WriteRes<WriteIRem32, VeeREH1DivResources>;
def : WriteRes<WriteIRem, VeeREH1DivResources>;
}
}

// Loads from the DCCM. A load result reaches ALU ops after 3 cycles.
let Latency = 3 in {
def : WriteRes<WriteLDB, [VeeREH1LSU]>;
def : WriteRes<WriteLDH, [VeeREH1LSU]>;
def : WriteRes<WriteLDW, [VeeREH1LSU]>;
def : WriteRes<WriteLDD, [VeeREH1LSU]>;
}

} // RetireOOO = true

def : WriteRes<WriteSTB, [VeeREH1LSU]>;
def : WriteRes<WriteSTH, [VeeREH1LSU]>;
def : WriteRes<WriteSTW, [VeeREH1LSU]>;
def : WriteRes<WriteSTD, [VeeREH1LSU]>;

// CSR accesses issue alone in i0.
let BeginGroup = true, EndGroup = true in
def : WriteRes<WriteCSR, [VeeREH1ALU0]>;
def : WriteRes<WriteNop, []>;

def : InstRW<[WriteIALU], (instrs COPY)>;
// c.j is an unconditional jump like jal, so nothing issues in i1 after it.
def : InstRW<[WriteJal], (instrs C_J)>;

//===----------------------------------------------------------------------===//
// Bypasses

defvar VeeREH1LoadWrites = [WriteLDB, WriteLDH, WriteLDW, WriteLDD];
defvar VeeREH1LateWrites = !listconcat(VeeREH1LoadWrites,
                                       [WriteIMul, WriteIMul32]);
defvar VeeREH1ALUWrites = [WriteIALU, WriteIALU32, WriteShiftImm,
                           WriteShiftImm32, WriteShiftReg, WriteShiftReg32];

// A branch can issue in the same cycle as a load or multiply it depends on.
def : ReadAdvance<ReadJmp, 3, VeeREH1LateWrites>;
def : ReadAdvance<ReadJalr, 0>;
def : ReadAdvance<ReadCSR, 0>;
// Store data is read late. An i0 ALU op can feed the store data of an i1 store
// in the same cycle, and a load result is forwarded to the store data of a
// store in the next cycle. A multiply result is not forwarded, so a store of
// it waits for the full multiply latency (measured: 2 stall cycles).
def : ReadAdvance<ReadStoreData, 1,
                  !listconcat(VeeREH1ALUWrites, VeeREH1LoadWrites),
                  !listconcat(!listsplat(0, !size(VeeREH1ALUWrites)),
                              !listsplat(1, !size(VeeREH1LoadWrites)))>;
// A load result reaches the address generator and the multiplier one cycle
// before it reaches the ALUs (measured load-to-load-address latency: 2).
def : ReadAdvance<ReadMemBase, 1, VeeREH1LoadWrites>;
def : ReadAdvance<ReadIMul, 1, VeeREH1LoadWrites>;
def : ReadAdvance<ReadIMul32, 1, VeeREH1LoadWrites>;
def : ReadAdvance<ReadIALU, 0>;
def : ReadAdvance<ReadIALU32, 0>;
def : ReadAdvance<ReadShiftImm, 0>;
def : ReadAdvance<ReadShiftImm32, 0>;
def : ReadAdvance<ReadShiftReg, 0>;
def : ReadAdvance<ReadShiftReg32, 0>;
def : ReadAdvance<ReadIDiv, 0>;
def : ReadAdvance<ReadIDiv32, 0>;
def : ReadAdvance<ReadIRem, 0>;
def : ReadAdvance<ReadIRem32, 0>;

//===----------------------------------------------------------------------===//
// Unsupported extensions
defm : UnsupportedSchedA;
defm : UnsupportedSchedF;
defm : UnsupportedSchedSFB;
defm : UnsupportedSchedV;
defm : UnsupportedSchedZabha;
defm : UnsupportedSchedZba;
defm : UnsupportedSchedZbb;
defm : UnsupportedSchedZbc;
defm : UnsupportedSchedZbs;
defm : UnsupportedSchedZbkb;
defm : UnsupportedSchedZbkx;
defm : UnsupportedSchedZfa;
defm : UnsupportedSchedZvk;
defm : UnsupportedSchedXsf;
}
