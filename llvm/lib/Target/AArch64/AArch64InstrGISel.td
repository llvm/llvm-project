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
  let hasSideEffects = 0;
}

// Represents a vector G_ASHR with an immediate.
def G_VASHR : AArch64GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, untyped_imm_0:$imm);
  let hasSideEffects = 0;
}

// Represents a vector G_LSHR with an immediate.
def G_VLSHR : AArch64GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, untyped_imm_0:$imm);
  let hasSideEffects = 0;
}

// Represents an integer to FP conversion on the FPR bank.
def G_SITOF : AArch64GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src);
  let hasSideEffects = 0;
}
def G_UITOF : AArch64GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src);
  let hasSideEffects = 0;
}

def G_FCMEQ : AArch64GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type1:$src2);
  let hasSideEffects = 0;
}

def G_FCMGE : AArch64GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type1:$src2);
  let hasSideEffects = 0;
}

def G_FCMGT : AArch64GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type1:$src2);
  let hasSideEffects = 0;
}

def G_FCMEQZ : AArch64GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src);
  let hasSideEffects = 0;
}

def G_FCMGEZ : AArch64GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src);
  let hasSideEffects = 0;
}

def G_FCMGTZ : AArch64GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src);
  let hasSideEffects = 0;
}

def G_FCMLEZ : AArch64GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src);
  let hasSideEffects = 0;
}

def G_FCMLTZ : AArch64GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src);
  let hasSideEffects = 0;
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

def : GINodeEquiv<G_FCMEQ, AArch64fcmeq>;
def : GINodeEquiv<G_FCMGE, AArch64fcmge>;
def : GINodeEquiv<G_FCMGT, AArch64fcmgt>;

def : GINodeEquiv<G_FCMEQZ, AArch64fcmeqz>;
def : GINodeEquiv<G_FCMGEZ, AArch64fcmgez>;
def : GINodeEquiv<G_FCMGTZ, AArch64fcmgtz>;
def : GINodeEquiv<G_FCMLEZ, AArch64fcmlez>;
def : GINodeEquiv<G_FCMLTZ, AArch64fcmltz>;

def : GINodeEquiv<G_EXTRACT_VECTOR_ELT, vector_extract>;

// These are patterns that we only use for GlobalISel via the importer.
def : Pat<(f32 (fadd (vector_extract (v2f32 FPR64:$Rn), (i64 0)),
                     (vector_extract (v2f32 FPR64:$Rn), (i64 1)))),
           (f32 (FADDPv2i32p (v2f32 FPR64:$Rn)))>;

let Predicates = [HasNEON] in {
  def : Pat<(v2f64 (sint_to_fp v2i32:$src)),
            (SCVTFv2f64 (SSHLLv2i32_shift V64:$src, 0))>;
  def : Pat<(v2f64 (uint_to_fp v2i32:$src)),
            (UCVTFv2f64 (USHLLv2i32_shift V64:$src, 0))>;
  def : Pat<(v2f32 (sint_to_fp v2i64:$src)),
            (FCVTNv2i32 (SCVTFv2f64 V128:$src))>;
  def : Pat<(v2f32 (uint_to_fp v2i64:$src)),
            (FCVTNv2i32 (UCVTFv2f64 V128:$src))>;

  def : Pat<(v2i64 (fp_to_sint v2f32:$src)),
            (FCVTZSv2f64 (FCVTLv2i32 V64:$src))>;
  def : Pat<(v2i64 (fp_to_uint v2f32:$src)),
            (FCVTZUv2f64 (FCVTLv2i32 V64:$src))>;
  def : Pat<(v2i32 (fp_to_sint v2f64:$src)),
            (XTNv2i32 (FCVTZSv2f64 V128:$src))>;
  def : Pat<(v2i32 (fp_to_uint v2f64:$src)),
            (XTNv2i32 (FCVTZUv2f64 V128:$src))>;

}

let Predicates = [HasNoLSE] in {
def : Pat<(atomic_cmp_swap_8 GPR64:$addr, GPR32:$desired, GPR32:$new),
          (CMP_SWAP_8 GPR64:$addr, GPR32:$desired, GPR32:$new)>;

def : Pat<(atomic_cmp_swap_16 GPR64:$addr, GPR32:$desired, GPR32:$new),
          (CMP_SWAP_16 GPR64:$addr, GPR32:$desired, GPR32:$new)>;

def : Pat<(atomic_cmp_swap_32 GPR64:$addr, GPR32:$desired, GPR32:$new),
          (CMP_SWAP_32 GPR64:$addr, GPR32:$desired, GPR32:$new)>;

def : Pat<(atomic_cmp_swap_64 GPR64:$addr, GPR64:$desired, GPR64:$new),
          (CMP_SWAP_64 GPR64:$addr, GPR64:$desired, GPR64:$new)>;
}

def : Pat<(int_aarch64_stlxp GPR64:$lo, GPR64:$hi, GPR64:$addr),
          (STLXPX GPR64:$lo, GPR64:$hi, GPR64:$addr)>;
def : Pat<(int_aarch64_stxp GPR64:$lo, GPR64:$hi, GPR64:$addr),
          (STXPX GPR64:$lo, GPR64:$hi, GPR64:$addr)>;
