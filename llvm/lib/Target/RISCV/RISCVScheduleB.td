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

// Zbe extension
def WriteDecompress  : SchedWrite; // bdecompress
def WriteCompress    : SchedWrite; // bcompress
def WriteDecompress32: SchedWrite; // bdecompressw
def WriteCompress32  : SchedWrite; // bcompressw

// Zbf extension
def WriteBFP         : SchedWrite; // BFP
def WriteBFP32       : SchedWrite; // BFPW

// Zbm extension
def WriteBMatrix     : SchedWrite; // bmator/bmatxor/bmatflip

// Zbp extension
def WriteORC         : SchedWrite; // gorc
def WriteREV         : SchedWrite; // grev
def WriteORC32       : SchedWrite; // gorcw
def WriteREV32       : SchedWrite; // grevw
def WriteREVImm      : SchedWrite; // grevi
def WriteORCImm      : SchedWrite; // gorci
def WriteREVImm32    : SchedWrite; // greviw
def WriteORCImm32    : SchedWrite; // gorciw
def WriteSHFL        : SchedWrite; // shfl
def WriteUNSHFL      : SchedWrite; // unshfl
def WriteSHFL32      : SchedWrite; // shflw
def WriteUNSHFL32    : SchedWrite; // unshflw
def WriteSHFLImm     : SchedWrite; // shfli
def WriteUNSHFLImm   : SchedWrite; // unshfli
def WriteXPERMH      : SchedWrite; // xperm.h
def WriteXPERMW      : SchedWrite; // xperm.w
def WritePACK        : SchedWrite; // pack/packh
def WritePACK32      : SchedWrite; // packw
def WritePACKU       : SchedWrite; // packu
def WritePACKU32     : SchedWrite; // packuw

// Zbr extension
def WriteCRCB        : SchedWrite; // crc32.b
def WriteCRCH        : SchedWrite; // crc32.h
def WriteCRCW        : SchedWrite; // crc32.w
def WriteCRCD        : SchedWrite; // crc32.d
def WriteCRCCB       : SchedWrite; // crc32c.b
def WriteCRCCH       : SchedWrite; // crc32c.h
def WriteCRCCW       : SchedWrite; // crc32c.w
def WriteCRCCD       : SchedWrite; // crc32c.d

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

// Zbe extension
def ReadDecompress  : SchedRead; // bdecompress
def ReadCompress    : SchedRead; // bcompress
def ReadDecompress32: SchedRead; // bdecompressw
def ReadCompress32  : SchedRead; // bcompressw

// Zbf extension
def ReadBFP         : SchedRead; // BFP
def ReadBFP32       : SchedRead; // BFPW

// Zbm extension
def ReadBMatrix     : SchedRead; // bmator/bmatxor/bmatflip

// Zbp extension
def ReadORC         : SchedRead; // gorc
def ReadREV         : SchedRead; // grev
def ReadORC32       : SchedRead; // gorcw
def ReadREV32       : SchedRead; // grevw
def ReadREVImm      : SchedRead; // grevi
def ReadORCImm      : SchedRead; // groci
def ReadREVImm32    : SchedRead; // greviw
def ReadORCImm32    : SchedRead; // gorciw
def ReadSHFL        : SchedRead; // shfl
def ReadUNSHFL      : SchedRead; // unshfl
def ReadSHFL32      : SchedRead; // shflw
def ReadUNSHFL32    : SchedRead; // unshflw
def ReadSHFLImm     : SchedRead; // shfli
def ReadUNSHFLImm   : SchedRead; // unshfli
def ReadXPERMH      : SchedRead; // xperm.h
def ReadXPERMW      : SchedRead; // xperm.w
def ReadPACK        : SchedRead; // pack/packh
def ReadPACK32      : SchedRead; // packw
def ReadPACKU       : SchedRead; // packu
def ReadPACKU32     : SchedRead; // packuw

// Zbr extension
def ReadCRCB        : SchedRead; // crc32.b
def ReadCRCH        : SchedRead; // crc32.h
def ReadCRCW        : SchedRead; // crc32.w
def ReadCRCD        : SchedRead; // crc32.d
def ReadCRCCB       : SchedRead; // crc32c.b
def ReadCRCCH       : SchedRead; // crc32c.h
def ReadCRCCW       : SchedRead; // crc32c.w
def ReadCRCCD       : SchedRead; // crc32c.d

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

multiclass UnsupportedSchedZbe {
let Unsupported = true in {
def : WriteRes<WriteDecompress, []>;
def : WriteRes<WriteCompress, []>;
def : WriteRes<WriteDecompress32, []>;
def : WriteRes<WriteCompress32, []>;

def : ReadAdvance<ReadDecompress, 0>;
def : ReadAdvance<ReadCompress, 0>;
def : ReadAdvance<ReadDecompress32, 0>;
def : ReadAdvance<ReadCompress32, 0>;
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

multiclass UnsupportedSchedZbm {
let Unsupported = true in {
def : WriteRes<WriteBMatrix, []>;

def : ReadAdvance<ReadBMatrix, 0>;
}
}

multiclass UnsupportedSchedZbp {
let Unsupported = true in {
def : WriteRes<WriteORC, []>;
def : WriteRes<WriteREV, []>;
def : WriteRes<WriteORC32, []>;
def : WriteRes<WriteREV32, []>;
def : WriteRes<WriteREVImm, []>;
def : WriteRes<WriteORCImm, []>;
def : WriteRes<WriteREVImm32, []>;
def : WriteRes<WriteORCImm32, []>;
def : WriteRes<WriteSHFL, []>;
def : WriteRes<WriteUNSHFL, []>;
def : WriteRes<WriteSHFL32, []>;
def : WriteRes<WriteUNSHFL32, []>;
def : WriteRes<WriteSHFLImm, []>;
def : WriteRes<WriteUNSHFLImm, []>;
def : WriteRes<WriteXPERMH, []>;
def : WriteRes<WriteXPERMW, []>;
def : WriteRes<WritePACK, []>;
def : WriteRes<WritePACK32, []>;
def : WriteRes<WritePACKU, []>;
def : WriteRes<WritePACKU32, []>;

def : ReadAdvance<ReadORC, 0>;
def : ReadAdvance<ReadREV, 0>;
def : ReadAdvance<ReadORC32, 0>;
def : ReadAdvance<ReadREV32, 0>;
def : ReadAdvance<ReadREVImm, 0>;
def : ReadAdvance<ReadORCImm, 0>;
def : ReadAdvance<ReadREVImm32, 0>;
def : ReadAdvance<ReadORCImm32, 0>;
def : ReadAdvance<ReadSHFL, 0>;
def : ReadAdvance<ReadUNSHFL, 0>;
def : ReadAdvance<ReadSHFL32, 0>;
def : ReadAdvance<ReadUNSHFL32, 0>;
def : ReadAdvance<ReadSHFLImm, 0>;
def : ReadAdvance<ReadUNSHFLImm, 0>;
def : ReadAdvance<ReadXPERMH, 0>;
def : ReadAdvance<ReadXPERMW, 0>;
def : ReadAdvance<ReadPACK, 0>;
def : ReadAdvance<ReadPACK32, 0>;
def : ReadAdvance<ReadPACKU, 0>;
def : ReadAdvance<ReadPACKU32, 0>;
}
}

multiclass UnsupportedSchedZbr {
let Unsupported = true in {
def : WriteRes<WriteCRCB, []>;
def : WriteRes<WriteCRCH, []>;
def : WriteRes<WriteCRCW, []>;
def : WriteRes<WriteCRCD, []>;
def : WriteRes<WriteCRCCB, []>;
def : WriteRes<WriteCRCCH, []>;
def : WriteRes<WriteCRCCW, []>;
def : WriteRes<WriteCRCCD, []>;

def : ReadAdvance<ReadCRCB, 0>;
def : ReadAdvance<ReadCRCH, 0>;
def : ReadAdvance<ReadCRCW, 0>;
def : ReadAdvance<ReadCRCD, 0>;
def : ReadAdvance<ReadCRCCB, 0>;
def : ReadAdvance<ReadCRCCH, 0>;
def : ReadAdvance<ReadCRCCW, 0>;
def : ReadAdvance<ReadCRCCD, 0>;
}
}
