//===-- RISCVScheduleB.td - RISCV Scheduling Definitions B -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// Define scheduler resources associated with def operands.

// Zba extension
def WriteSHXADD      : SchedWrite; // sh1add/sh2add/sh3add
def WriteSHXADD32    : SchedWrite; // sh1add.uw/sh2add.uw/sh3add.uw

// Zbb extension
def WriteRotateImm   : SchedWrite;
def WriteRotateImm32 : SchedWrite;
def WriteRotateReg   : SchedWrite;
def WriteRotateReg32 : SchedWrite;
def WriteCLZ         : SchedWrite;
def WriteCLZ32       : SchedWrite;
def WriteCTZ         : SchedWrite;
def WriteCTZ32       : SchedWrite;
def WriteCPOP        : SchedWrite;
def WriteCPOP32      : SchedWrite;
def WriteREV8        : SchedWrite;
def WriteORCB        : SchedWrite;

// Zbc extension
def WriteCLMUL       : SchedWrite; // CLMUL/CLMULR/CLMULH

// Zbs extension
def WriteSingleBit   : SchedWrite; // BCLR/BSET/BINV/BEXT
def WriteSingleBitImm: SchedWrite; // BCLRI/BSETI/BINVI/BEXTI

// Zbf extension
def WriteBFP         : SchedWrite; // BFP
def WriteBFP32       : SchedWrite; // BFPW

/// Define scheduler resources associated with use operands.

// Zba extension
def ReadSHXADD      : SchedRead; // sh1add/sh2add/sh3add
def ReadSHXADD32    : SchedRead; // sh1add.uw/sh2add.uw/sh3add.uw

// Zbb extension
def ReadRotateImm   : SchedRead;
def ReadRotateImm32 : SchedRead;
def ReadRotateReg   : SchedRead;
def ReadRotateReg32 : SchedRead;
def ReadCLZ         : SchedRead;
def ReadCLZ32       : SchedRead;
def ReadCTZ         : SchedRead;
def ReadCTZ32       : SchedRead;
def ReadCPOP        : SchedRead;
def ReadCPOP32      : SchedRead;
def ReadREV8        : SchedRead;
def ReadORCB        : SchedRead;

// Zbc extension
def ReadCLMUL       : SchedRead; // CLMUL/CLMULR/CLMULH

// Zbs extension
def ReadSingleBit   : SchedRead; // BCLR/BSET/BINV/BEXT
def ReadSingleBitImm: SchedRead; // BCLRI/BSETI/BINVI/BEXTI

// Zbf extension
def ReadBFP         : SchedRead; // BFP
def ReadBFP32       : SchedRead; // BFPW

/// Define default scheduler resources for B.

multiclass UnsupportedSchedZba {
let Unsupported = true in {
def : WriteRes<WriteSHXADD, []>;
def : WriteRes<WriteSHXADD32, []>;

def : ReadAdvance<ReadSHXADD, 0>;
def : ReadAdvance<ReadSHXADD32, 0>;
}
}

multiclass UnsupportedSchedZbb {
let Unsupported = true in {
def : WriteRes<WriteRotateImm, []>;
def : WriteRes<WriteRotateImm32, []>;
def : WriteRes<WriteRotateReg, []>;
def : WriteRes<WriteRotateReg32, []>;
def : WriteRes<WriteCLZ, []>;
def : WriteRes<WriteCLZ32, []>;
def : WriteRes<WriteCTZ, []>;
def : WriteRes<WriteCTZ32, []>;
def : WriteRes<WriteCPOP, []>;
def : WriteRes<WriteCPOP32, []>;
def : WriteRes<WriteREV8, []>;
def : WriteRes<WriteORCB, []>;

def : ReadAdvance<ReadRotateImm, 0>;
def : ReadAdvance<ReadRotateImm32, 0>;
def : ReadAdvance<ReadRotateReg, 0>;
def : ReadAdvance<ReadRotateReg32, 0>;
def : ReadAdvance<ReadCLZ, 0>;
def : ReadAdvance<ReadCLZ32, 0>;
def : ReadAdvance<ReadCTZ, 0>;
def : ReadAdvance<ReadCTZ32, 0>;
def : ReadAdvance<ReadCPOP, 0>;
def : ReadAdvance<ReadCPOP32, 0>;
def : ReadAdvance<ReadREV8, 0>;
def : ReadAdvance<ReadORCB, 0>;
}
}

multiclass UnsupportedSchedZbc {
let Unsupported = true in {
def : WriteRes<WriteCLMUL, []>;

def : ReadAdvance<ReadCLMUL, 0>;
}
}

multiclass UnsupportedSchedZbs {
let Unsupported = true in {
def : WriteRes<WriteSingleBit, []>;
def : WriteRes<WriteSingleBitImm, []>;

def : ReadAdvance<ReadSingleBit, 0>;
def : ReadAdvance<ReadSingleBitImm, 0>;
}
}

multiclass UnsupportedSchedZbf {
let Unsupported = true in {
def : WriteRes<WriteBFP, []>;
def : WriteRes<WriteBFP32, []>;

def : ReadAdvance<ReadBFP, 0>;
def : ReadAdvance<ReadBFP32, 0>;
}
}
