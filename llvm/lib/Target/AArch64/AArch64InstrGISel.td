//=----- AArch64InstrGISel.td - AArch64 GISel target pseudos -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AArch64 GlobalISel target pseudo instruction definitions. This is kept
// separately from the other tablegen files for organizational purposes, but
// share the same infrastructure.
//
//===----------------------------------------------------------------------===//


class AArch64GenericInstruction : GenericInstruction {
  let Namespace = "AArch64";
}

// A pseudo to represent a relocatable add instruction as part of address
// computation.
def G_ADD_LOW : AArch64GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type1:$src, type2:$imm);
  let hasSideEffects = 0;
}

// Pseudo for a rev16 instruction. Produced post-legalization from
// G_SHUFFLE_VECTORs with appropriate masks.
def G_REV16 : AArch64GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src);
  let hasSideEffects = 0;
}

// Pseudo for a rev32 instruction. Produced post-legalization from
// G_SHUFFLE_VECTORs with appropriate masks.
def G_REV32 : AArch64GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src);
  let hasSideEffects = 0;
}

// Pseudo for a rev64 instruction. Produced post-legalization from
// G_SHUFFLE_VECTORs with appropriate masks.
def G_REV64 : AArch64GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src);
  let hasSideEffects = 0;
}

// Represents an uzp1 instruction. Produced post-legalization from
// G_SHUFFLE_VECTORs with appropriate masks.
def G_UZP1 : AArch64GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$v1, type0:$v2);
  let hasSideEffects = 0;
}

// Represents an uzp2 instruction. Produced post-legalization from
// G_SHUFFLE_VECTORs with appropriate masks.
def G_UZP2 : AArch64GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$v1, type0:$v2);
  let hasSideEffects = 0;
}

// Represents a zip1 instruction. Produced post-legalization from
// G_SHUFFLE_VECTORs with appropriate masks.
def G_ZIP1 : AArch64GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$v1, type0:$v2);
  let hasSideEffects = 0;
}

// Represents a zip2 instruction. Produced post-legalization from
// G_SHUFFLE_VECTORs with appropriate masks.
def G_ZIP2 : AArch64GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$v1, type0:$v2);
  let hasSideEffects = 0;
}

// Represents a dup instruction. Produced post-legalization from
// G_SHUFFLE_VECTORs with appropriate masks.
def G_DUP: AArch64GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type1:$lane);
  let hasSideEffects = 0;
}

// Represents a lane duplicate operation.
def G_DUPLANE8 : AArch64GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src, type1:$lane);
  let hasSideEffects = 0;
}
def G_DUPLANE16 : AArch64GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src, type1:$lane);
  let hasSideEffects = 0;
}
def G_DUPLANE32 : AArch64GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src, type1:$lane);
  let hasSideEffects = 0;
}
def G_DUPLANE64 : AArch64GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src, type1:$lane);
  let hasSideEffects = 0;
}

// Represents a trn1 instruction. Produced post-legalization from
// G_SHUFFLE_VECTORs with appropriate masks.
def G_TRN1 : AArch64GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$v1, type0:$v2);
  let hasSideEffects = 0;
}

// Represents a trn2 instruction. Produced post-legalization from
// G_SHUFFLE_VECTORs with appropriate masks.
def G_TRN2 : AArch64GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$v1, type0:$v2);
  let hasSideEffects = 0;
}

// Represents an ext instruction. Produced post-legalization from
// G_SHUFFLE_VECTORs with appropriate masks.
def G_EXT: AArch64GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$v1, type0:$v2, untyped_imm_0:$imm);
}

// Represents a vector G_ASHR with an immediate.
def G_VASHR : AArch64GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, untyped_imm_0:$imm);
}

// Represents a vector G_LSHR with an immediate.
def G_VLSHR : AArch64GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, untyped_imm_0:$imm);
}

// Represents an integer to FP conversion on the FPR bank.
def G_SITOF : AArch64GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src);
}
def G_UITOF : AArch64GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src);
}

def : GINodeEquiv<G_REV16, AArch64rev16>;
def : GINodeEquiv<G_REV32, AArch64rev32>;
def : GINodeEquiv<G_REV64, AArch64rev64>;
def : GINodeEquiv<G_UZP1, AArch64uzp1>;
def : GINodeEquiv<G_UZP2, AArch64uzp2>;
def : GINodeEquiv<G_ZIP1, AArch64zip1>;
def : GINodeEquiv<G_ZIP2, AArch64zip2>;
def : GINodeEquiv<G_DUP, AArch64dup>;
def : GINodeEquiv<G_DUPLANE8, AArch64duplane8>;
def : GINodeEquiv<G_DUPLANE16, AArch64duplane16>;
def : GINodeEquiv<G_DUPLANE32, AArch64duplane32>;
def : GINodeEquiv<G_DUPLANE64, AArch64duplane64>;
def : GINodeEquiv<G_TRN1, AArch64trn1>;
def : GINodeEquiv<G_TRN2, AArch64trn2>;
def : GINodeEquiv<G_EXT, AArch64ext>;
def : GINodeEquiv<G_VASHR, AArch64vashr>;
def : GINodeEquiv<G_VLSHR, AArch64vlshr>;
def : GINodeEquiv<G_SITOF, AArch64sitof>;
def : GINodeEquiv<G_UITOF, AArch64uitof>;

def : GINodeEquiv<G_EXTRACT_VECTOR_ELT, vector_extract>;

// These are patterns that we only use for GlobalISel via the importer.
def : Pat<(f32 (fadd (vector_extract (v2f32 FPR64:$Rn), (i64 0)),
                     (vector_extract (v2f32 FPR64:$Rn), (i64 1)))),
           (f32 (FADDPv2i32p (v2f32 FPR64:$Rn)))>;
