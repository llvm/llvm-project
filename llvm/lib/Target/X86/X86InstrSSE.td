//===-- X86InstrSSE.td - SSE Instruction Set ---------------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the X86 SSE instruction set, defining the instructions,
// and properties of the instructions which are needed for code generation,
// machine code emission, and analysis.
//
//===----------------------------------------------------------------------===//

class OpndItins<InstrItinClass arg_rr, InstrItinClass arg_rm> {
  InstrItinClass rr = arg_rr;
  InstrItinClass rm = arg_rm;
  // InstrSchedModel info.
  X86FoldableSchedWrite Sched = WriteFAdd;
}

class SizeItins<OpndItins arg_s, OpndItins arg_d> {
  OpndItins s = arg_s;
  OpndItins d = arg_d;
}


class ShiftOpndItins<InstrItinClass arg_rr, InstrItinClass arg_rm,
  InstrItinClass arg_ri> {
  InstrItinClass rr = arg_rr;
  InstrItinClass rm = arg_rm;
  InstrItinClass ri = arg_ri;
}


// scalar
let Sched = WriteFAdd in {
def SSE_ALU_F32S : OpndItins<
  IIC_SSE_ALU_F32S_RR, IIC_SSE_ALU_F32S_RM
>;

def SSE_ALU_F64S : OpndItins<
  IIC_SSE_ALU_F64S_RR, IIC_SSE_ALU_F64S_RM
>;
}

def SSE_ALU_ITINS_S : SizeItins<
  SSE_ALU_F32S, SSE_ALU_F64S
>;

let Sched = WriteFMul in {
def SSE_MUL_F32S : OpndItins<
  IIC_SSE_MUL_F32S_RR, IIC_SSE_MUL_F64S_RM
>;

def SSE_MUL_F64S : OpndItins<
  IIC_SSE_MUL_F64S_RR, IIC_SSE_MUL_F64S_RM
>;
}

def SSE_MUL_ITINS_S : SizeItins<
  SSE_MUL_F32S, SSE_MUL_F64S
>;

let Sched = WriteFDiv in {
def SSE_DIV_F32S : OpndItins<
  IIC_SSE_DIV_F32S_RR, IIC_SSE_DIV_F64S_RM
>;

def SSE_DIV_F64S : OpndItins<
  IIC_SSE_DIV_F64S_RR, IIC_SSE_DIV_F64S_RM
>;
}

def SSE_DIV_ITINS_S : SizeItins<
  SSE_DIV_F32S, SSE_DIV_F64S
>;

// parallel
let Sched = WriteFAdd in {
def SSE_ALU_F32P : OpndItins<
  IIC_SSE_ALU_F32P_RR, IIC_SSE_ALU_F32P_RM
>;

def SSE_ALU_F64P : OpndItins<
  IIC_SSE_ALU_F64P_RR, IIC_SSE_ALU_F64P_RM
>;
}

def SSE_ALU_ITINS_P : SizeItins<
  SSE_ALU_F32P, SSE_ALU_F64P
>;

let Sched = WriteFMul in {
def SSE_MUL_F32P : OpndItins<
  IIC_SSE_MUL_F32P_RR, IIC_SSE_MUL_F64P_RM
>;

def SSE_MUL_F64P : OpndItins<
  IIC_SSE_MUL_F64P_RR, IIC_SSE_MUL_F64P_RM
>;
}

def SSE_MUL_ITINS_P : SizeItins<
  SSE_MUL_F32P, SSE_MUL_F64P
>;

let Sched = WriteFDiv in {
def SSE_DIV_F32P : OpndItins<
  IIC_SSE_DIV_F32P_RR, IIC_SSE_DIV_F64P_RM
>;

def SSE_DIV_F64P : OpndItins<
  IIC_SSE_DIV_F64P_RR, IIC_SSE_DIV_F64P_RM
>;
}

def SSE_DIV_ITINS_P : SizeItins<
  SSE_DIV_F32P, SSE_DIV_F64P
>;

def SSE_BIT_ITINS_P : OpndItins<
  IIC_SSE_BIT_P_RR, IIC_SSE_BIT_P_RM
>;

let Sched = WriteVecALU in {
def SSE_INTALU_ITINS_P : OpndItins<
  IIC_SSE_INTALU_P_RR, IIC_SSE_INTALU_P_RM
>;

def SSE_INTALUQ_ITINS_P : OpndItins<
  IIC_SSE_INTALUQ_P_RR, IIC_SSE_INTALUQ_P_RM
>;
}

let Sched = WriteVecIMul in
def SSE_INTMUL_ITINS_P : OpndItins<
  IIC_SSE_INTMUL_P_RR, IIC_SSE_INTMUL_P_RM
>;

def SSE_INTSHIFT_ITINS_P : ShiftOpndItins<
  IIC_SSE_INTSH_P_RR, IIC_SSE_INTSH_P_RM, IIC_SSE_INTSH_P_RI
>;

def SSE_MOVA_ITINS : OpndItins<
  IIC_SSE_MOVA_P_RR, IIC_SSE_MOVA_P_RM
>;

def SSE_MOVU_ITINS : OpndItins<
  IIC_SSE_MOVU_P_RR, IIC_SSE_MOVU_P_RM
>;

def SSE_DPPD_ITINS : OpndItins<
  IIC_SSE_DPPD_RR, IIC_SSE_DPPD_RM
>;

def SSE_DPPS_ITINS : OpndItins<
  IIC_SSE_DPPS_RR, IIC_SSE_DPPD_RM
>;

def DEFAULT_ITINS : OpndItins<
  IIC_ALU_NONMEM, IIC_ALU_MEM
>;

def SSE_EXTRACT_ITINS : OpndItins<
  IIC_SSE_EXTRACTPS_RR, IIC_SSE_EXTRACTPS_RM
>;

def SSE_INSERT_ITINS : OpndItins<
  IIC_SSE_INSERTPS_RR, IIC_SSE_INSERTPS_RM
>;

def SSE_MPSADBW_ITINS : OpndItins<
  IIC_SSE_MPSADBW_RR, IIC_SSE_MPSADBW_RM
>;

def SSE_PMULLD_ITINS : OpndItins<
  IIC_SSE_PMULLD_RR, IIC_SSE_PMULLD_RM
>;

//===----------------------------------------------------------------------===//
// SSE 1 & 2 Instructions Classes
//===----------------------------------------------------------------------===//

/// sse12_fp_scalar - SSE 1 & 2 scalar instructions class
multiclass sse12_fp_scalar<bits<8> opc, string OpcodeStr, SDNode OpNode,
                           RegisterClass RC, X86MemOperand x86memop,
                           OpndItins itins,
                           bit Is2Addr = 1> {
  let isCommutable = 1 in {
    def rr : SI<opc, MRMSrcReg, (outs RC:$dst), (ins RC:$src1, RC:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set RC:$dst, (OpNode RC:$src1, RC:$src2))], itins.rr>,
       Sched<[itins.Sched]>;
  }
  def rm : SI<opc, MRMSrcMem, (outs RC:$dst), (ins RC:$src1, x86memop:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set RC:$dst, (OpNode RC:$src1, (load addr:$src2)))], itins.rm>,
       Sched<[itins.Sched.Folded, ReadAfterLd]>;
}

/// sse12_fp_scalar_int - SSE 1 & 2 scalar instructions intrinsics class
multiclass sse12_fp_scalar_int<bits<8> opc, string OpcodeStr, RegisterClass RC,
                             string asm, string SSEVer, string FPSizeStr,
                             Operand memopr, ComplexPattern mem_cpat,
                             OpndItins itins,
                             bit Is2Addr = 1> {
  def rr_Int : SI<opc, MRMSrcReg, (outs RC:$dst), (ins RC:$src1, RC:$src2),
       !if(Is2Addr,
           !strconcat(asm, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(asm, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set RC:$dst, (!cast<Intrinsic>(
                 !strconcat("int_x86_sse", SSEVer, "_", OpcodeStr, FPSizeStr))
             RC:$src1, RC:$src2))], itins.rr>,
       Sched<[itins.Sched]>;
  def rm_Int : SI<opc, MRMSrcMem, (outs RC:$dst), (ins RC:$src1, memopr:$src2),
       !if(Is2Addr,
           !strconcat(asm, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(asm, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set RC:$dst, (!cast<Intrinsic>(!strconcat("int_x86_sse",
                                          SSEVer, "_", OpcodeStr, FPSizeStr))
             RC:$src1, mem_cpat:$src2))], itins.rm>,
       Sched<[itins.Sched.Folded, ReadAfterLd]>;
}

/// sse12_fp_packed - SSE 1 & 2 packed instructions class
multiclass sse12_fp_packed<bits<8> opc, string OpcodeStr, SDNode OpNode,
                           RegisterClass RC, ValueType vt,
                           X86MemOperand x86memop, PatFrag mem_frag,
                           Domain d, OpndItins itins, bit Is2Addr = 1> {
  let isCommutable = 1 in
    def rr : PI<opc, MRMSrcReg, (outs RC:$dst), (ins RC:$src1, RC:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set RC:$dst, (vt (OpNode RC:$src1, RC:$src2)))], itins.rr, d>,
       Sched<[itins.Sched]>;
  let mayLoad = 1 in
    def rm : PI<opc, MRMSrcMem, (outs RC:$dst), (ins RC:$src1, x86memop:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set RC:$dst, (OpNode RC:$src1, (mem_frag addr:$src2)))],
          itins.rm, d>,
       Sched<[itins.Sched.Folded, ReadAfterLd]>;
}

/// sse12_fp_packed_logical_rm - SSE 1 & 2 packed instructions class
multiclass sse12_fp_packed_logical_rm<bits<8> opc, RegisterClass RC, Domain d,
                                      string OpcodeStr, X86MemOperand x86memop,
                                      list<dag> pat_rr, list<dag> pat_rm,
                                      bit Is2Addr = 1> {
  let isCommutable = 1, hasSideEffects = 0 in
    def rr : PI<opc, MRMSrcReg, (outs RC:$dst), (ins RC:$src1, RC:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       pat_rr, NoItinerary, d>,
       Sched<[WriteVecLogic]>;
  def rm : PI<opc, MRMSrcMem, (outs RC:$dst), (ins RC:$src1, x86memop:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       pat_rm, NoItinerary, d>,
       Sched<[WriteVecLogicLd, ReadAfterLd]>;
}

//===----------------------------------------------------------------------===//
//  Non-instruction patterns
//===----------------------------------------------------------------------===//

// A vector extract of the first f32/f64 position is a subregister copy
def : Pat<(f32 (vector_extract (v4f32 VR128:$src), (iPTR 0))),
          (COPY_TO_REGCLASS (v4f32 VR128:$src), FR32)>;
def : Pat<(f64 (vector_extract (v2f64 VR128:$src), (iPTR 0))),
          (COPY_TO_REGCLASS (v2f64 VR128:$src), FR64)>;

// A 128-bit subvector extract from the first 256-bit vector position
// is a subregister copy that needs no instruction.
def : Pat<(v4i32 (extract_subvector (v8i32 VR256:$src), (iPTR 0))),
          (v4i32 (EXTRACT_SUBREG (v8i32 VR256:$src), sub_xmm))>;
def : Pat<(v4f32 (extract_subvector (v8f32 VR256:$src), (iPTR 0))),
          (v4f32 (EXTRACT_SUBREG (v8f32 VR256:$src), sub_xmm))>;

def : Pat<(v2i64 (extract_subvector (v4i64 VR256:$src), (iPTR 0))),
          (v2i64 (EXTRACT_SUBREG (v4i64 VR256:$src), sub_xmm))>;
def : Pat<(v2f64 (extract_subvector (v4f64 VR256:$src), (iPTR 0))),
          (v2f64 (EXTRACT_SUBREG (v4f64 VR256:$src), sub_xmm))>;

def : Pat<(v8i16 (extract_subvector (v16i16 VR256:$src), (iPTR 0))),
          (v8i16 (EXTRACT_SUBREG (v16i16 VR256:$src), sub_xmm))>;
def : Pat<(v16i8 (extract_subvector (v32i8 VR256:$src), (iPTR 0))),
          (v16i8 (EXTRACT_SUBREG (v32i8 VR256:$src), sub_xmm))>;

// A 128-bit subvector insert to the first 256-bit vector position
// is a subregister copy that needs no instruction.
let AddedComplexity = 25 in { // to give priority over vinsertf128rm
def : Pat<(insert_subvector undef, (v2i64 VR128:$src), (iPTR 0)),
          (INSERT_SUBREG (v4i64 (IMPLICIT_DEF)), VR128:$src, sub_xmm)>;
def : Pat<(insert_subvector undef, (v2f64 VR128:$src), (iPTR 0)),
          (INSERT_SUBREG (v4f64 (IMPLICIT_DEF)), VR128:$src, sub_xmm)>;
def : Pat<(insert_subvector undef, (v4i32 VR128:$src), (iPTR 0)),
          (INSERT_SUBREG (v8i32 (IMPLICIT_DEF)), VR128:$src, sub_xmm)>;
def : Pat<(insert_subvector undef, (v4f32 VR128:$src), (iPTR 0)),
          (INSERT_SUBREG (v8f32 (IMPLICIT_DEF)), VR128:$src, sub_xmm)>;
def : Pat<(insert_subvector undef, (v8i16 VR128:$src), (iPTR 0)),
          (INSERT_SUBREG (v16i16 (IMPLICIT_DEF)), VR128:$src, sub_xmm)>;
def : Pat<(insert_subvector undef, (v16i8 VR128:$src), (iPTR 0)),
          (INSERT_SUBREG (v32i8 (IMPLICIT_DEF)), VR128:$src, sub_xmm)>;
}

// Implicitly promote a 32-bit scalar to a vector.
def : Pat<(v4f32 (scalar_to_vector FR32:$src)),
          (COPY_TO_REGCLASS FR32:$src, VR128)>;
def : Pat<(v8f32 (scalar_to_vector FR32:$src)),
          (COPY_TO_REGCLASS FR32:$src, VR128)>;
// Implicitly promote a 64-bit scalar to a vector.
def : Pat<(v2f64 (scalar_to_vector FR64:$src)),
          (COPY_TO_REGCLASS FR64:$src, VR128)>;
def : Pat<(v4f64 (scalar_to_vector FR64:$src)),
          (COPY_TO_REGCLASS FR64:$src, VR128)>;

// Bitcasts between 128-bit vector types. Return the original type since
// no instruction is needed for the conversion
let Predicates = [HasSSE2] in {
  def : Pat<(v2i64 (bitconvert (v4i32 VR128:$src))), (v2i64 VR128:$src)>;
  def : Pat<(v2i64 (bitconvert (v8i16 VR128:$src))), (v2i64 VR128:$src)>;
  def : Pat<(v2i64 (bitconvert (v16i8 VR128:$src))), (v2i64 VR128:$src)>;
  def : Pat<(v2i64 (bitconvert (v2f64 VR128:$src))), (v2i64 VR128:$src)>;
  def : Pat<(v2i64 (bitconvert (v4f32 VR128:$src))), (v2i64 VR128:$src)>;
  def : Pat<(v4i32 (bitconvert (v2i64 VR128:$src))), (v4i32 VR128:$src)>;
  def : Pat<(v4i32 (bitconvert (v8i16 VR128:$src))), (v4i32 VR128:$src)>;
  def : Pat<(v4i32 (bitconvert (v16i8 VR128:$src))), (v4i32 VR128:$src)>;
  def : Pat<(v4i32 (bitconvert (v2f64 VR128:$src))), (v4i32 VR128:$src)>;
  def : Pat<(v4i32 (bitconvert (v4f32 VR128:$src))), (v4i32 VR128:$src)>;
  def : Pat<(v8i16 (bitconvert (v2i64 VR128:$src))), (v8i16 VR128:$src)>;
  def : Pat<(v8i16 (bitconvert (v4i32 VR128:$src))), (v8i16 VR128:$src)>;
  def : Pat<(v8i16 (bitconvert (v16i8 VR128:$src))), (v8i16 VR128:$src)>;
  def : Pat<(v8i16 (bitconvert (v2f64 VR128:$src))), (v8i16 VR128:$src)>;
  def : Pat<(v8i16 (bitconvert (v4f32 VR128:$src))), (v8i16 VR128:$src)>;
  def : Pat<(v16i8 (bitconvert (v2i64 VR128:$src))), (v16i8 VR128:$src)>;
  def : Pat<(v16i8 (bitconvert (v4i32 VR128:$src))), (v16i8 VR128:$src)>;
  def : Pat<(v16i8 (bitconvert (v8i16 VR128:$src))), (v16i8 VR128:$src)>;
  def : Pat<(v16i8 (bitconvert (v2f64 VR128:$src))), (v16i8 VR128:$src)>;
  def : Pat<(v16i8 (bitconvert (v4f32 VR128:$src))), (v16i8 VR128:$src)>;
  def : Pat<(v4f32 (bitconvert (v2i64 VR128:$src))), (v4f32 VR128:$src)>;
  def : Pat<(v4f32 (bitconvert (v4i32 VR128:$src))), (v4f32 VR128:$src)>;
  def : Pat<(v4f32 (bitconvert (v8i16 VR128:$src))), (v4f32 VR128:$src)>;
  def : Pat<(v4f32 (bitconvert (v16i8 VR128:$src))), (v4f32 VR128:$src)>;
  def : Pat<(v4f32 (bitconvert (v2f64 VR128:$src))), (v4f32 VR128:$src)>;
  def : Pat<(v2f64 (bitconvert (v2i64 VR128:$src))), (v2f64 VR128:$src)>;
  def : Pat<(v2f64 (bitconvert (v4i32 VR128:$src))), (v2f64 VR128:$src)>;
  def : Pat<(v2f64 (bitconvert (v8i16 VR128:$src))), (v2f64 VR128:$src)>;
  def : Pat<(v2f64 (bitconvert (v16i8 VR128:$src))), (v2f64 VR128:$src)>;
  def : Pat<(v2f64 (bitconvert (v4f32 VR128:$src))), (v2f64 VR128:$src)>;
}

// Bitcasts between 256-bit vector types. Return the original type since
// no instruction is needed for the conversion
let Predicates = [HasAVX] in {
  def : Pat<(v4f64  (bitconvert (v8f32 VR256:$src))),  (v4f64 VR256:$src)>;
  def : Pat<(v4f64  (bitconvert (v8i32 VR256:$src))),  (v4f64 VR256:$src)>;
  def : Pat<(v4f64  (bitconvert (v4i64 VR256:$src))),  (v4f64 VR256:$src)>;
  def : Pat<(v4f64  (bitconvert (v16i16 VR256:$src))), (v4f64 VR256:$src)>;
  def : Pat<(v4f64  (bitconvert (v32i8 VR256:$src))),  (v4f64 VR256:$src)>;
  def : Pat<(v8f32  (bitconvert (v8i32 VR256:$src))),  (v8f32 VR256:$src)>;
  def : Pat<(v8f32  (bitconvert (v4i64 VR256:$src))),  (v8f32 VR256:$src)>;
  def : Pat<(v8f32  (bitconvert (v4f64 VR256:$src))),  (v8f32 VR256:$src)>;
  def : Pat<(v8f32  (bitconvert (v32i8 VR256:$src))),  (v8f32 VR256:$src)>;
  def : Pat<(v8f32  (bitconvert (v16i16 VR256:$src))), (v8f32 VR256:$src)>;
  def : Pat<(v4i64  (bitconvert (v8f32 VR256:$src))),  (v4i64 VR256:$src)>;
  def : Pat<(v4i64  (bitconvert (v8i32 VR256:$src))),  (v4i64 VR256:$src)>;
  def : Pat<(v4i64  (bitconvert (v4f64 VR256:$src))),  (v4i64 VR256:$src)>;
  def : Pat<(v4i64  (bitconvert (v32i8 VR256:$src))),  (v4i64 VR256:$src)>;
  def : Pat<(v4i64  (bitconvert (v16i16 VR256:$src))), (v4i64 VR256:$src)>;
  def : Pat<(v32i8  (bitconvert (v4f64 VR256:$src))),  (v32i8 VR256:$src)>;
  def : Pat<(v32i8  (bitconvert (v4i64 VR256:$src))),  (v32i8 VR256:$src)>;
  def : Pat<(v32i8  (bitconvert (v8f32 VR256:$src))),  (v32i8 VR256:$src)>;
  def : Pat<(v32i8  (bitconvert (v8i32 VR256:$src))),  (v32i8 VR256:$src)>;
  def : Pat<(v32i8  (bitconvert (v16i16 VR256:$src))), (v32i8 VR256:$src)>;
  def : Pat<(v8i32  (bitconvert (v32i8 VR256:$src))),  (v8i32 VR256:$src)>;
  def : Pat<(v8i32  (bitconvert (v16i16 VR256:$src))), (v8i32 VR256:$src)>;
  def : Pat<(v8i32  (bitconvert (v8f32 VR256:$src))),  (v8i32 VR256:$src)>;
  def : Pat<(v8i32  (bitconvert (v4i64 VR256:$src))),  (v8i32 VR256:$src)>;
  def : Pat<(v8i32  (bitconvert (v4f64 VR256:$src))),  (v8i32 VR256:$src)>;
  def : Pat<(v16i16 (bitconvert (v8f32 VR256:$src))),  (v16i16 VR256:$src)>;
  def : Pat<(v16i16 (bitconvert (v8i32 VR256:$src))),  (v16i16 VR256:$src)>;
  def : Pat<(v16i16 (bitconvert (v4i64 VR256:$src))),  (v16i16 VR256:$src)>;
  def : Pat<(v16i16 (bitconvert (v4f64 VR256:$src))),  (v16i16 VR256:$src)>;
  def : Pat<(v16i16 (bitconvert (v32i8 VR256:$src))),  (v16i16 VR256:$src)>;
}

// Alias instructions that map fld0 to xorps for sse or vxorps for avx.
// This is expanded by ExpandPostRAPseudos.
let isReMaterializable = 1, isAsCheapAsAMove = 1, canFoldAsLoad = 1,
    isPseudo = 1, SchedRW = [WriteZero] in {
  def FsFLD0SS : I<0, Pseudo, (outs FR32:$dst), (ins), "",
                   [(set FR32:$dst, fp32imm0)]>, Requires<[HasSSE1]>;
  def FsFLD0SD : I<0, Pseudo, (outs FR64:$dst), (ins), "",
                   [(set FR64:$dst, fpimm0)]>, Requires<[HasSSE2]>;
}

//===----------------------------------------------------------------------===//
// AVX & SSE - Zero/One Vectors
//===----------------------------------------------------------------------===//

// Alias instruction that maps zero vector to pxor / xorp* for sse.
// This is expanded by ExpandPostRAPseudos to an xorps / vxorps, and then
// swizzled by ExecutionDepsFix to pxor.
// We set canFoldAsLoad because this can be converted to a constant-pool
// load of an all-zeros value if folding it would be beneficial.
let isReMaterializable = 1, isAsCheapAsAMove = 1, canFoldAsLoad = 1,
    isPseudo = 1, SchedRW = [WriteZero] in {
def V_SET0 : I<0, Pseudo, (outs VR128:$dst), (ins), "",
               [(set VR128:$dst, (v4f32 immAllZerosV))]>;
}

def : Pat<(v2f64 immAllZerosV), (V_SET0)>;
def : Pat<(v4i32 immAllZerosV), (V_SET0)>;
def : Pat<(v2i64 immAllZerosV), (V_SET0)>;
def : Pat<(v8i16 immAllZerosV), (V_SET0)>;
def : Pat<(v16i8 immAllZerosV), (V_SET0)>;


// The same as done above but for AVX.  The 256-bit AVX1 ISA doesn't support PI,
// and doesn't need it because on sandy bridge the register is set to zero
// at the rename stage without using any execution unit, so SET0PSY
// and SET0PDY can be used for vector int instructions without penalty
let isReMaterializable = 1, isAsCheapAsAMove = 1, canFoldAsLoad = 1,
    isPseudo = 1, Predicates = [HasAVX], SchedRW = [WriteZero] in {
def AVX_SET0 : I<0, Pseudo, (outs VR256:$dst), (ins), "",
                 [(set VR256:$dst, (v8f32 immAllZerosV))]>;
}

let Predicates = [HasAVX] in
  def : Pat<(v4f64 immAllZerosV), (AVX_SET0)>;

let Predicates = [HasAVX2] in {
  def : Pat<(v4i64 immAllZerosV), (AVX_SET0)>;
  def : Pat<(v8i32 immAllZerosV), (AVX_SET0)>;
  def : Pat<(v16i16 immAllZerosV), (AVX_SET0)>;
  def : Pat<(v32i8 immAllZerosV), (AVX_SET0)>;
}

// AVX1 has no support for 256-bit integer instructions, but since the 128-bit
// VPXOR instruction writes zero to its upper part, it's safe build zeros.
let Predicates = [HasAVX1Only] in {
def : Pat<(v32i8 immAllZerosV), (SUBREG_TO_REG (i8 0), (V_SET0), sub_xmm)>;
def : Pat<(bc_v32i8 (v8f32 immAllZerosV)),
          (SUBREG_TO_REG (i8 0), (V_SET0), sub_xmm)>;

def : Pat<(v16i16 immAllZerosV), (SUBREG_TO_REG (i16 0), (V_SET0), sub_xmm)>;
def : Pat<(bc_v16i16 (v8f32 immAllZerosV)),
          (SUBREG_TO_REG (i16 0), (V_SET0), sub_xmm)>;

def : Pat<(v8i32 immAllZerosV), (SUBREG_TO_REG (i32 0), (V_SET0), sub_xmm)>;
def : Pat<(bc_v8i32 (v8f32 immAllZerosV)),
          (SUBREG_TO_REG (i32 0), (V_SET0), sub_xmm)>;

def : Pat<(v4i64 immAllZerosV), (SUBREG_TO_REG (i64 0), (V_SET0), sub_xmm)>;
def : Pat<(bc_v4i64 (v8f32 immAllZerosV)),
          (SUBREG_TO_REG (i64 0), (V_SET0), sub_xmm)>;
}

// We set canFoldAsLoad because this can be converted to a constant-pool
// load of an all-ones value if folding it would be beneficial.
let isReMaterializable = 1, isAsCheapAsAMove = 1, canFoldAsLoad = 1,
    isPseudo = 1, SchedRW = [WriteZero] in {
  def V_SETALLONES : I<0, Pseudo, (outs VR128:$dst), (ins), "",
                       [(set VR128:$dst, (v4i32 immAllOnesV))]>;
  let Predicates = [HasAVX2] in
  def AVX2_SETALLONES : I<0, Pseudo, (outs VR256:$dst), (ins), "",
                          [(set VR256:$dst, (v8i32 immAllOnesV))]>;
}


//===----------------------------------------------------------------------===//
// SSE 1 & 2 - Move FP Scalar Instructions
//
// Move Instructions. Register-to-register movss/movsd is not used for FR32/64
// register copies because it's a partial register update; Register-to-register
// movss/movsd is not modeled as an INSERT_SUBREG because INSERT_SUBREG requires
// that the insert be implementable in terms of a copy, and just mentioned, we
// don't use movss/movsd for copies.
//===----------------------------------------------------------------------===//

multiclass sse12_move_rr<RegisterClass RC, SDNode OpNode, ValueType vt,
                         X86MemOperand x86memop, string base_opc,
                         string asm_opr> {
  def rr : SI<0x10, MRMSrcReg, (outs VR128:$dst),
              (ins VR128:$src1, RC:$src2),
              !strconcat(base_opc, asm_opr),
              [(set VR128:$dst, (vt (OpNode VR128:$src1,
                                 (scalar_to_vector RC:$src2))))],
              IIC_SSE_MOV_S_RR>, Sched<[WriteMove]>;

  // For the disassembler
  let isCodeGenOnly = 1, hasSideEffects = 0 in
  def rr_REV : SI<0x11, MRMDestReg, (outs VR128:$dst),
                  (ins VR128:$src1, RC:$src2),
                  !strconcat(base_opc, asm_opr),
                  [], IIC_SSE_MOV_S_RR>, Sched<[WriteMove]>;
}

multiclass sse12_move<RegisterClass RC, SDNode OpNode, ValueType vt,
                      X86MemOperand x86memop, string OpcodeStr> {
  // AVX
  defm V#NAME : sse12_move_rr<RC, OpNode, vt, x86memop, OpcodeStr,
                              "\t{$src2, $src1, $dst|$dst, $src1, $src2}">,
                              VEX_4V, VEX_LIG;

  def V#NAME#mr : SI<0x11, MRMDestMem, (outs), (ins x86memop:$dst, RC:$src),
                     !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                     [(store RC:$src, addr:$dst)], IIC_SSE_MOV_S_MR>,
                     VEX, VEX_LIG, Sched<[WriteStore]>;
  // SSE1 & 2
  let Constraints = "$src1 = $dst" in {
    defm NAME : sse12_move_rr<RC, OpNode, vt, x86memop, OpcodeStr,
                              "\t{$src2, $dst|$dst, $src2}">;
  }

  def NAME#mr   : SI<0x11, MRMDestMem, (outs), (ins x86memop:$dst, RC:$src),
                     !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                     [(store RC:$src, addr:$dst)], IIC_SSE_MOV_S_MR>,
                  Sched<[WriteStore]>;
}

// Loading from memory automatically zeroing upper bits.
multiclass sse12_move_rm<RegisterClass RC, X86MemOperand x86memop,
                         PatFrag mem_pat, string OpcodeStr> {
  def V#NAME#rm : SI<0x10, MRMSrcMem, (outs RC:$dst), (ins x86memop:$src),
                     !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                     [(set RC:$dst, (mem_pat addr:$src))],
                     IIC_SSE_MOV_S_RM>, VEX, VEX_LIG, Sched<[WriteLoad]>;
  def NAME#rm   : SI<0x10, MRMSrcMem, (outs RC:$dst), (ins x86memop:$src),
                     !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                     [(set RC:$dst, (mem_pat addr:$src))],
                     IIC_SSE_MOV_S_RM>, Sched<[WriteLoad]>;
}

defm MOVSS : sse12_move<FR32, X86Movss, v4f32, f32mem, "movss">, XS;
defm MOVSD : sse12_move<FR64, X86Movsd, v2f64, f64mem, "movsd">, XD;

let canFoldAsLoad = 1, isReMaterializable = 1 in {
  defm MOVSS : sse12_move_rm<FR32, f32mem, loadf32, "movss">, XS;

  let AddedComplexity = 20 in
    defm MOVSD : sse12_move_rm<FR64, f64mem, loadf64, "movsd">, XD;
}

// Patterns
let Predicates = [UseAVX] in {
  let AddedComplexity = 15 in {
  // Move scalar to XMM zero-extended, zeroing a VR128 then do a
  // MOVS{S,D} to the lower bits.
  def : Pat<(v4f32 (X86vzmovl (v4f32 (scalar_to_vector FR32:$src)))),
            (VMOVSSrr (v4f32 (V_SET0)), FR32:$src)>;
  def : Pat<(v4f32 (X86vzmovl (v4f32 VR128:$src))),
            (VMOVSSrr (v4f32 (V_SET0)), (COPY_TO_REGCLASS VR128:$src, FR32))>;
  def : Pat<(v4i32 (X86vzmovl (v4i32 VR128:$src))),
            (VMOVSSrr (v4i32 (V_SET0)), (COPY_TO_REGCLASS VR128:$src, FR32))>;
  def : Pat<(v2f64 (X86vzmovl (v2f64 (scalar_to_vector FR64:$src)))),
            (VMOVSDrr (v2f64 (V_SET0)), FR64:$src)>;

  // Move low f32 and clear high bits.
  def : Pat<(v8f32 (X86vzmovl (v8f32 VR256:$src))),
            (SUBREG_TO_REG (i32 0),
             (VMOVSSrr (v4f32 (V_SET0)),
                       (EXTRACT_SUBREG (v8f32 VR256:$src), sub_xmm)), sub_xmm)>;
  def : Pat<(v8i32 (X86vzmovl (v8i32 VR256:$src))),
            (SUBREG_TO_REG (i32 0),
             (VMOVSSrr (v4i32 (V_SET0)),
                       (EXTRACT_SUBREG (v8i32 VR256:$src), sub_xmm)), sub_xmm)>;
  }

  let AddedComplexity = 20 in {
  // MOVSSrm zeros the high parts of the register; represent this
  // with SUBREG_TO_REG. The AVX versions also write: DST[255:128] <- 0
  def : Pat<(v4f32 (X86vzmovl (v4f32 (scalar_to_vector (loadf32 addr:$src))))),
            (COPY_TO_REGCLASS (VMOVSSrm addr:$src), VR128)>;
  def : Pat<(v4f32 (scalar_to_vector (loadf32 addr:$src))),
            (COPY_TO_REGCLASS (VMOVSSrm addr:$src), VR128)>;
  def : Pat<(v4f32 (X86vzmovl (loadv4f32 addr:$src))),
            (COPY_TO_REGCLASS (VMOVSSrm addr:$src), VR128)>;

  // MOVSDrm zeros the high parts of the register; represent this
  // with SUBREG_TO_REG. The AVX versions also write: DST[255:128] <- 0
  def : Pat<(v2f64 (X86vzmovl (v2f64 (scalar_to_vector (loadf64 addr:$src))))),
            (COPY_TO_REGCLASS (VMOVSDrm addr:$src), VR128)>;
  def : Pat<(v2f64 (scalar_to_vector (loadf64 addr:$src))),
            (COPY_TO_REGCLASS (VMOVSDrm addr:$src), VR128)>;
  def : Pat<(v2f64 (X86vzmovl (loadv2f64 addr:$src))),
            (COPY_TO_REGCLASS (VMOVSDrm addr:$src), VR128)>;
  def : Pat<(v2f64 (X86vzmovl (bc_v2f64 (loadv4f32 addr:$src)))),
            (COPY_TO_REGCLASS (VMOVSDrm addr:$src), VR128)>;
  def : Pat<(v2f64 (X86vzload addr:$src)),
            (COPY_TO_REGCLASS (VMOVSDrm addr:$src), VR128)>;

  // Represent the same patterns above but in the form they appear for
  // 256-bit types
  def : Pat<(v8i32 (X86vzmovl (insert_subvector undef,
                   (v4i32 (scalar_to_vector (loadi32 addr:$src))), (iPTR 0)))),
            (SUBREG_TO_REG (i32 0), (VMOVSSrm addr:$src), sub_xmm)>;
  def : Pat<(v8f32 (X86vzmovl (insert_subvector undef,
                   (v4f32 (scalar_to_vector (loadf32 addr:$src))), (iPTR 0)))),
            (SUBREG_TO_REG (i32 0), (VMOVSSrm addr:$src), sub_xmm)>;
  def : Pat<(v4f64 (X86vzmovl (insert_subvector undef,
                   (v2f64 (scalar_to_vector (loadf64 addr:$src))), (iPTR 0)))),
            (SUBREG_TO_REG (i32 0), (VMOVSDrm addr:$src), sub_xmm)>;
  }
  def : Pat<(v8f32 (X86vzmovl (insert_subvector undef,
                   (v4f32 (scalar_to_vector FR32:$src)), (iPTR 0)))),
            (SUBREG_TO_REG (i32 0),
                           (v4f32 (VMOVSSrr (v4f32 (V_SET0)), FR32:$src)),
                           sub_xmm)>;
  def : Pat<(v4f64 (X86vzmovl (insert_subvector undef,
                   (v2f64 (scalar_to_vector FR64:$src)), (iPTR 0)))),
            (SUBREG_TO_REG (i64 0),
                           (v2f64 (VMOVSDrr (v2f64 (V_SET0)), FR64:$src)),
                           sub_xmm)>;
  def : Pat<(v4i64 (X86vzmovl (insert_subvector undef,
                   (v2i64 (scalar_to_vector (loadi64 addr:$src))), (iPTR 0)))),
            (SUBREG_TO_REG (i64 0), (VMOVSDrm addr:$src), sub_xmm)>;

  // Move low f64 and clear high bits.
  def : Pat<(v4f64 (X86vzmovl (v4f64 VR256:$src))),
            (SUBREG_TO_REG (i32 0),
             (VMOVSDrr (v2f64 (V_SET0)),
                       (EXTRACT_SUBREG (v4f64 VR256:$src), sub_xmm)), sub_xmm)>;

  def : Pat<(v4i64 (X86vzmovl (v4i64 VR256:$src))),
            (SUBREG_TO_REG (i32 0),
             (VMOVSDrr (v2i64 (V_SET0)),
                       (EXTRACT_SUBREG (v4i64 VR256:$src), sub_xmm)), sub_xmm)>;

  // Extract and store.
  def : Pat<(store (f32 (vector_extract (v4f32 VR128:$src), (iPTR 0))),
                   addr:$dst),
            (VMOVSSmr addr:$dst, (COPY_TO_REGCLASS (v4f32 VR128:$src), FR32))>;
  def : Pat<(store (f64 (vector_extract (v2f64 VR128:$src), (iPTR 0))),
                   addr:$dst),
            (VMOVSDmr addr:$dst, (COPY_TO_REGCLASS (v2f64 VR128:$src), FR64))>;

  // Shuffle with VMOVSS
  def : Pat<(v4i32 (X86Movss VR128:$src1, VR128:$src2)),
            (VMOVSSrr (v4i32 VR128:$src1),
                      (COPY_TO_REGCLASS (v4i32 VR128:$src2), FR32))>;
  def : Pat<(v4f32 (X86Movss VR128:$src1, VR128:$src2)),
            (VMOVSSrr (v4f32 VR128:$src1),
                      (COPY_TO_REGCLASS (v4f32 VR128:$src2), FR32))>;

  // 256-bit variants
  def : Pat<(v8i32 (X86Movss VR256:$src1, VR256:$src2)),
            (SUBREG_TO_REG (i32 0),
              (VMOVSSrr (EXTRACT_SUBREG (v8i32 VR256:$src1), sub_xmm),
                        (EXTRACT_SUBREG (v8i32 VR256:$src2), sub_xmm)),
              sub_xmm)>;
  def : Pat<(v8f32 (X86Movss VR256:$src1, VR256:$src2)),
            (SUBREG_TO_REG (i32 0),
              (VMOVSSrr (EXTRACT_SUBREG (v8f32 VR256:$src1), sub_xmm),
                        (EXTRACT_SUBREG (v8f32 VR256:$src2), sub_xmm)),
              sub_xmm)>;

  // Shuffle with VMOVSD
  def : Pat<(v2i64 (X86Movsd VR128:$src1, VR128:$src2)),
            (VMOVSDrr VR128:$src1, (COPY_TO_REGCLASS VR128:$src2, FR64))>;
  def : Pat<(v2f64 (X86Movsd VR128:$src1, VR128:$src2)),
            (VMOVSDrr VR128:$src1, (COPY_TO_REGCLASS VR128:$src2, FR64))>;
  def : Pat<(v4f32 (X86Movsd VR128:$src1, VR128:$src2)),
            (VMOVSDrr VR128:$src1, (COPY_TO_REGCLASS VR128:$src2, FR64))>;
  def : Pat<(v4i32 (X86Movsd VR128:$src1, VR128:$src2)),
            (VMOVSDrr VR128:$src1, (COPY_TO_REGCLASS VR128:$src2, FR64))>;

  // 256-bit variants
  def : Pat<(v4i64 (X86Movsd VR256:$src1, VR256:$src2)),
            (SUBREG_TO_REG (i32 0),
              (VMOVSDrr (EXTRACT_SUBREG (v4i64 VR256:$src1), sub_xmm),
                        (EXTRACT_SUBREG (v4i64 VR256:$src2), sub_xmm)),
              sub_xmm)>;
  def : Pat<(v4f64 (X86Movsd VR256:$src1, VR256:$src2)),
            (SUBREG_TO_REG (i32 0),
              (VMOVSDrr (EXTRACT_SUBREG (v4f64 VR256:$src1), sub_xmm),
                        (EXTRACT_SUBREG (v4f64 VR256:$src2), sub_xmm)),
              sub_xmm)>;


  // FIXME: Instead of a X86Movlps there should be a X86Movsd here, the problem
  // is during lowering, where it's not possible to recognize the fold cause
  // it has two uses through a bitcast. One use disappears at isel time and the
  // fold opportunity reappears.
  def : Pat<(v2f64 (X86Movlpd VR128:$src1, VR128:$src2)),
            (VMOVSDrr VR128:$src1, (COPY_TO_REGCLASS VR128:$src2, FR64))>;
  def : Pat<(v2i64 (X86Movlpd VR128:$src1, VR128:$src2)),
            (VMOVSDrr VR128:$src1, (COPY_TO_REGCLASS VR128:$src2, FR64))>;
  def : Pat<(v4f32 (X86Movlps VR128:$src1, VR128:$src2)),
            (VMOVSDrr VR128:$src1, (COPY_TO_REGCLASS VR128:$src2, FR64))>;
  def : Pat<(v4i32 (X86Movlps VR128:$src1, VR128:$src2)),
            (VMOVSDrr VR128:$src1, (COPY_TO_REGCLASS VR128:$src2, FR64))>;
}

let Predicates = [UseSSE1] in {
  let AddedComplexity = 15 in {
  // Move scalar to XMM zero-extended, zeroing a VR128 then do a
  // MOVSS to the lower bits.
  def : Pat<(v4f32 (X86vzmovl (v4f32 (scalar_to_vector FR32:$src)))),
            (MOVSSrr (v4f32 (V_SET0)), FR32:$src)>;
  def : Pat<(v4f32 (X86vzmovl (v4f32 VR128:$src))),
            (MOVSSrr (v4f32 (V_SET0)), (COPY_TO_REGCLASS VR128:$src, FR32))>;
  def : Pat<(v4i32 (X86vzmovl (v4i32 VR128:$src))),
            (MOVSSrr (v4i32 (V_SET0)), (COPY_TO_REGCLASS VR128:$src, FR32))>;
  }

  let AddedComplexity = 20 in {
  // MOVSSrm already zeros the high parts of the register.
  def : Pat<(v4f32 (X86vzmovl (v4f32 (scalar_to_vector (loadf32 addr:$src))))),
            (COPY_TO_REGCLASS (MOVSSrm addr:$src), VR128)>;
  def : Pat<(v4f32 (scalar_to_vector (loadf32 addr:$src))),
            (COPY_TO_REGCLASS (MOVSSrm addr:$src), VR128)>;
  def : Pat<(v4f32 (X86vzmovl (loadv4f32 addr:$src))),
            (COPY_TO_REGCLASS (MOVSSrm addr:$src), VR128)>;
  }

  // Extract and store.
  def : Pat<(store (f32 (vector_extract (v4f32 VR128:$src), (iPTR 0))),
                   addr:$dst),
            (MOVSSmr addr:$dst, (COPY_TO_REGCLASS VR128:$src, FR32))>;

  // Shuffle with MOVSS
  def : Pat<(v4i32 (X86Movss VR128:$src1, VR128:$src2)),
            (MOVSSrr VR128:$src1, (COPY_TO_REGCLASS VR128:$src2, FR32))>;
  def : Pat<(v4f32 (X86Movss VR128:$src1, VR128:$src2)),
            (MOVSSrr VR128:$src1, (COPY_TO_REGCLASS VR128:$src2, FR32))>;
}

let Predicates = [UseSSE2] in {
  let AddedComplexity = 15 in {
  // Move scalar to XMM zero-extended, zeroing a VR128 then do a
  // MOVSD to the lower bits.
  def : Pat<(v2f64 (X86vzmovl (v2f64 (scalar_to_vector FR64:$src)))),
            (MOVSDrr (v2f64 (V_SET0)), FR64:$src)>;
  }

  let AddedComplexity = 20 in {
  // MOVSDrm already zeros the high parts of the register.
  def : Pat<(v2f64 (X86vzmovl (v2f64 (scalar_to_vector (loadf64 addr:$src))))),
            (COPY_TO_REGCLASS (MOVSDrm addr:$src), VR128)>;
  def : Pat<(v2f64 (scalar_to_vector (loadf64 addr:$src))),
            (COPY_TO_REGCLASS (MOVSDrm addr:$src), VR128)>;
  def : Pat<(v2f64 (X86vzmovl (loadv2f64 addr:$src))),
            (COPY_TO_REGCLASS (MOVSDrm addr:$src), VR128)>;
  def : Pat<(v2f64 (X86vzmovl (bc_v2f64 (loadv4f32 addr:$src)))),
            (COPY_TO_REGCLASS (MOVSDrm addr:$src), VR128)>;
  def : Pat<(v2f64 (X86vzload addr:$src)),
            (COPY_TO_REGCLASS (MOVSDrm addr:$src), VR128)>;
  }

  // Extract and store.
  def : Pat<(store (f64 (vector_extract (v2f64 VR128:$src), (iPTR 0))),
                   addr:$dst),
            (MOVSDmr addr:$dst, (COPY_TO_REGCLASS VR128:$src, FR64))>;

  // Shuffle with MOVSD
  def : Pat<(v2i64 (X86Movsd VR128:$src1, VR128:$src2)),
            (MOVSDrr VR128:$src1, (COPY_TO_REGCLASS VR128:$src2, FR64))>;
  def : Pat<(v2f64 (X86Movsd VR128:$src1, VR128:$src2)),
            (MOVSDrr VR128:$src1, (COPY_TO_REGCLASS VR128:$src2, FR64))>;
  def : Pat<(v4f32 (X86Movsd VR128:$src1, VR128:$src2)),
            (MOVSDrr VR128:$src1, (COPY_TO_REGCLASS VR128:$src2, FR64))>;
  def : Pat<(v4i32 (X86Movsd VR128:$src1, VR128:$src2)),
            (MOVSDrr VR128:$src1, (COPY_TO_REGCLASS VR128:$src2, FR64))>;

  // FIXME: Instead of a X86Movlps there should be a X86Movsd here, the problem
  // is during lowering, where it's not possible to recognize the fold cause
  // it has two uses through a bitcast. One use disappears at isel time and the
  // fold opportunity reappears.
  def : Pat<(v2f64 (X86Movlpd VR128:$src1, VR128:$src2)),
            (MOVSDrr VR128:$src1, (COPY_TO_REGCLASS VR128:$src2, FR64))>;
  def : Pat<(v2i64 (X86Movlpd VR128:$src1, VR128:$src2)),
            (MOVSDrr VR128:$src1, (COPY_TO_REGCLASS VR128:$src2, FR64))>;
  def : Pat<(v4f32 (X86Movlps VR128:$src1, VR128:$src2)),
            (MOVSDrr VR128:$src1, (COPY_TO_REGCLASS VR128:$src2, FR64))>;
  def : Pat<(v4i32 (X86Movlps VR128:$src1, VR128:$src2)),
            (MOVSDrr VR128:$src1, (COPY_TO_REGCLASS VR128:$src2, FR64))>;
}

//===----------------------------------------------------------------------===//
// SSE 1 & 2 - Move Aligned/Unaligned FP Instructions
//===----------------------------------------------------------------------===//

multiclass sse12_mov_packed<bits<8> opc, RegisterClass RC,
                            X86MemOperand x86memop, PatFrag ld_frag,
                            string asm, Domain d,
                            OpndItins itins,
                            bit IsReMaterializable = 1> {
let neverHasSideEffects = 1 in
  def rr : PI<opc, MRMSrcReg, (outs RC:$dst), (ins RC:$src),
              !strconcat(asm, "\t{$src, $dst|$dst, $src}"), [], itins.rr, d>,
           Sched<[WriteMove]>;
let canFoldAsLoad = 1, isReMaterializable = IsReMaterializable in
  def rm : PI<opc, MRMSrcMem, (outs RC:$dst), (ins x86memop:$src),
              !strconcat(asm, "\t{$src, $dst|$dst, $src}"),
                   [(set RC:$dst, (ld_frag addr:$src))], itins.rm, d>,
           Sched<[WriteLoad]>;
}

defm VMOVAPS : sse12_mov_packed<0x28, VR128, f128mem, alignedloadv4f32,
                              "movaps", SSEPackedSingle, SSE_MOVA_ITINS>,
                              TB, VEX;
defm VMOVAPD : sse12_mov_packed<0x28, VR128, f128mem, alignedloadv2f64,
                              "movapd", SSEPackedDouble, SSE_MOVA_ITINS>,
                              TB, OpSize, VEX;
defm VMOVUPS : sse12_mov_packed<0x10, VR128, f128mem, loadv4f32,
                              "movups", SSEPackedSingle, SSE_MOVU_ITINS>,
                              TB, VEX;
defm VMOVUPD : sse12_mov_packed<0x10, VR128, f128mem, loadv2f64,
                              "movupd", SSEPackedDouble, SSE_MOVU_ITINS, 0>,
                              TB, OpSize, VEX;

defm VMOVAPSY : sse12_mov_packed<0x28, VR256, f256mem, alignedloadv8f32,
                              "movaps", SSEPackedSingle, SSE_MOVA_ITINS>,
                              TB, VEX, VEX_L;
defm VMOVAPDY : sse12_mov_packed<0x28, VR256, f256mem, alignedloadv4f64,
                              "movapd", SSEPackedDouble, SSE_MOVA_ITINS>,
                              TB, OpSize, VEX, VEX_L;
defm VMOVUPSY : sse12_mov_packed<0x10, VR256, f256mem, loadv8f32,
                              "movups", SSEPackedSingle, SSE_MOVU_ITINS>,
                              TB, VEX, VEX_L;
defm VMOVUPDY : sse12_mov_packed<0x10, VR256, f256mem, loadv4f64,
                              "movupd", SSEPackedDouble, SSE_MOVU_ITINS, 0>,
                              TB, OpSize, VEX, VEX_L;
defm MOVAPS : sse12_mov_packed<0x28, VR128, f128mem, alignedloadv4f32,
                              "movaps", SSEPackedSingle, SSE_MOVA_ITINS>,
                              TB;
defm MOVAPD : sse12_mov_packed<0x28, VR128, f128mem, alignedloadv2f64,
                              "movapd", SSEPackedDouble, SSE_MOVA_ITINS>,
                              TB, OpSize;
defm MOVUPS : sse12_mov_packed<0x10, VR128, f128mem, loadv4f32,
                              "movups", SSEPackedSingle, SSE_MOVU_ITINS>,
                              TB;
defm MOVUPD : sse12_mov_packed<0x10, VR128, f128mem, loadv2f64,
                              "movupd", SSEPackedDouble, SSE_MOVU_ITINS, 0>,
                              TB, OpSize;

let SchedRW = [WriteStore] in {
def VMOVAPSmr : VPSI<0x29, MRMDestMem, (outs), (ins f128mem:$dst, VR128:$src),
                   "movaps\t{$src, $dst|$dst, $src}",
                   [(alignedstore (v4f32 VR128:$src), addr:$dst)],
                   IIC_SSE_MOVA_P_MR>, VEX;
def VMOVAPDmr : VPDI<0x29, MRMDestMem, (outs), (ins f128mem:$dst, VR128:$src),
                   "movapd\t{$src, $dst|$dst, $src}",
                   [(alignedstore (v2f64 VR128:$src), addr:$dst)],
                   IIC_SSE_MOVA_P_MR>, VEX;
def VMOVUPSmr : VPSI<0x11, MRMDestMem, (outs), (ins f128mem:$dst, VR128:$src),
                   "movups\t{$src, $dst|$dst, $src}",
                   [(store (v4f32 VR128:$src), addr:$dst)],
                   IIC_SSE_MOVU_P_MR>, VEX;
def VMOVUPDmr : VPDI<0x11, MRMDestMem, (outs), (ins f128mem:$dst, VR128:$src),
                   "movupd\t{$src, $dst|$dst, $src}",
                   [(store (v2f64 VR128:$src), addr:$dst)],
                   IIC_SSE_MOVU_P_MR>, VEX;
def VMOVAPSYmr : VPSI<0x29, MRMDestMem, (outs), (ins f256mem:$dst, VR256:$src),
                   "movaps\t{$src, $dst|$dst, $src}",
                   [(alignedstore256 (v8f32 VR256:$src), addr:$dst)],
                   IIC_SSE_MOVA_P_MR>, VEX, VEX_L;
def VMOVAPDYmr : VPDI<0x29, MRMDestMem, (outs), (ins f256mem:$dst, VR256:$src),
                   "movapd\t{$src, $dst|$dst, $src}",
                   [(alignedstore256 (v4f64 VR256:$src), addr:$dst)],
                   IIC_SSE_MOVA_P_MR>, VEX, VEX_L;
def VMOVUPSYmr : VPSI<0x11, MRMDestMem, (outs), (ins f256mem:$dst, VR256:$src),
                   "movups\t{$src, $dst|$dst, $src}",
                   [(store (v8f32 VR256:$src), addr:$dst)],
                   IIC_SSE_MOVU_P_MR>, VEX, VEX_L;
def VMOVUPDYmr : VPDI<0x11, MRMDestMem, (outs), (ins f256mem:$dst, VR256:$src),
                   "movupd\t{$src, $dst|$dst, $src}",
                   [(store (v4f64 VR256:$src), addr:$dst)],
                   IIC_SSE_MOVU_P_MR>, VEX, VEX_L;
} // SchedRW

// For disassembler
let isCodeGenOnly = 1, hasSideEffects = 0, SchedRW = [WriteMove] in {
  def VMOVAPSrr_REV : VPSI<0x29, MRMDestReg, (outs VR128:$dst),
                          (ins VR128:$src),
                          "movaps\t{$src, $dst|$dst, $src}", [],
                          IIC_SSE_MOVA_P_RR>, VEX;
  def VMOVAPDrr_REV : VPDI<0x29, MRMDestReg, (outs VR128:$dst),
                           (ins VR128:$src),
                           "movapd\t{$src, $dst|$dst, $src}", [],
                           IIC_SSE_MOVA_P_RR>, VEX;
  def VMOVUPSrr_REV : VPSI<0x11, MRMDestReg, (outs VR128:$dst),
                           (ins VR128:$src),
                           "movups\t{$src, $dst|$dst, $src}", [],
                           IIC_SSE_MOVU_P_RR>, VEX;
  def VMOVUPDrr_REV : VPDI<0x11, MRMDestReg, (outs VR128:$dst),
                           (ins VR128:$src),
                           "movupd\t{$src, $dst|$dst, $src}", [],
                           IIC_SSE_MOVU_P_RR>, VEX;
  def VMOVAPSYrr_REV : VPSI<0x29, MRMDestReg, (outs VR256:$dst),
                            (ins VR256:$src),
                            "movaps\t{$src, $dst|$dst, $src}", [],
                            IIC_SSE_MOVA_P_RR>, VEX, VEX_L;
  def VMOVAPDYrr_REV : VPDI<0x29, MRMDestReg, (outs VR256:$dst),
                            (ins VR256:$src),
                            "movapd\t{$src, $dst|$dst, $src}", [],
                            IIC_SSE_MOVA_P_RR>, VEX, VEX_L;
  def VMOVUPSYrr_REV : VPSI<0x11, MRMDestReg, (outs VR256:$dst),
                            (ins VR256:$src),
                            "movups\t{$src, $dst|$dst, $src}", [],
                            IIC_SSE_MOVU_P_RR>, VEX, VEX_L;
  def VMOVUPDYrr_REV : VPDI<0x11, MRMDestReg, (outs VR256:$dst),
                            (ins VR256:$src),
                            "movupd\t{$src, $dst|$dst, $src}", [],
                            IIC_SSE_MOVU_P_RR>, VEX, VEX_L;
}

let Predicates = [HasAVX] in {
def : Pat<(v8i32 (X86vzmovl
                  (insert_subvector undef, (v4i32 VR128:$src), (iPTR 0)))),
          (SUBREG_TO_REG (i32 0), (VMOVAPSrr VR128:$src), sub_xmm)>;
def : Pat<(v4i64 (X86vzmovl
                  (insert_subvector undef, (v2i64 VR128:$src), (iPTR 0)))),
          (SUBREG_TO_REG (i32 0), (VMOVAPSrr VR128:$src), sub_xmm)>;
def : Pat<(v8f32 (X86vzmovl
                  (insert_subvector undef, (v4f32 VR128:$src), (iPTR 0)))),
          (SUBREG_TO_REG (i32 0), (VMOVAPSrr VR128:$src), sub_xmm)>;
def : Pat<(v4f64 (X86vzmovl
                  (insert_subvector undef, (v2f64 VR128:$src), (iPTR 0)))),
          (SUBREG_TO_REG (i32 0), (VMOVAPSrr VR128:$src), sub_xmm)>;
}


def : Pat<(int_x86_avx_storeu_ps_256 addr:$dst, VR256:$src),
          (VMOVUPSYmr addr:$dst, VR256:$src)>;
def : Pat<(int_x86_avx_storeu_pd_256 addr:$dst, VR256:$src),
          (VMOVUPDYmr addr:$dst, VR256:$src)>;

let SchedRW = [WriteStore] in {
def MOVAPSmr : PSI<0x29, MRMDestMem, (outs), (ins f128mem:$dst, VR128:$src),
                   "movaps\t{$src, $dst|$dst, $src}",
                   [(alignedstore (v4f32 VR128:$src), addr:$dst)],
                   IIC_SSE_MOVA_P_MR>;
def MOVAPDmr : PDI<0x29, MRMDestMem, (outs), (ins f128mem:$dst, VR128:$src),
                   "movapd\t{$src, $dst|$dst, $src}",
                   [(alignedstore (v2f64 VR128:$src), addr:$dst)],
                   IIC_SSE_MOVA_P_MR>;
def MOVUPSmr : PSI<0x11, MRMDestMem, (outs), (ins f128mem:$dst, VR128:$src),
                   "movups\t{$src, $dst|$dst, $src}",
                   [(store (v4f32 VR128:$src), addr:$dst)],
                   IIC_SSE_MOVU_P_MR>;
def MOVUPDmr : PDI<0x11, MRMDestMem, (outs), (ins f128mem:$dst, VR128:$src),
                   "movupd\t{$src, $dst|$dst, $src}",
                   [(store (v2f64 VR128:$src), addr:$dst)],
                   IIC_SSE_MOVU_P_MR>;
} // SchedRW

// For disassembler
let isCodeGenOnly = 1, hasSideEffects = 0, SchedRW = [WriteMove] in {
  def MOVAPSrr_REV : PSI<0x29, MRMDestReg, (outs VR128:$dst), (ins VR128:$src),
                         "movaps\t{$src, $dst|$dst, $src}", [],
                         IIC_SSE_MOVA_P_RR>;
  def MOVAPDrr_REV : PDI<0x29, MRMDestReg, (outs VR128:$dst), (ins VR128:$src),
                         "movapd\t{$src, $dst|$dst, $src}", [],
                         IIC_SSE_MOVA_P_RR>;
  def MOVUPSrr_REV : PSI<0x11, MRMDestReg, (outs VR128:$dst), (ins VR128:$src),
                         "movups\t{$src, $dst|$dst, $src}", [],
                         IIC_SSE_MOVU_P_RR>;
  def MOVUPDrr_REV : PDI<0x11, MRMDestReg, (outs VR128:$dst), (ins VR128:$src),
                         "movupd\t{$src, $dst|$dst, $src}", [],
                         IIC_SSE_MOVU_P_RR>;
}

let Predicates = [HasAVX] in {
  def : Pat<(int_x86_sse_storeu_ps addr:$dst, VR128:$src),
            (VMOVUPSmr addr:$dst, VR128:$src)>;
  def : Pat<(int_x86_sse2_storeu_pd addr:$dst, VR128:$src),
            (VMOVUPDmr addr:$dst, VR128:$src)>;
}

let Predicates = [UseSSE1] in
  def : Pat<(int_x86_sse_storeu_ps addr:$dst, VR128:$src),
            (MOVUPSmr addr:$dst, VR128:$src)>;
let Predicates = [UseSSE2] in
  def : Pat<(int_x86_sse2_storeu_pd addr:$dst, VR128:$src),
            (MOVUPDmr addr:$dst, VR128:$src)>;

// Use vmovaps/vmovups for AVX integer load/store.
let Predicates = [HasAVX] in {
  // 128-bit load/store
  def : Pat<(alignedloadv2i64 addr:$src),
            (VMOVAPSrm addr:$src)>;
  def : Pat<(loadv2i64 addr:$src),
            (VMOVUPSrm addr:$src)>;

  def : Pat<(alignedstore (v2i64 VR128:$src), addr:$dst),
            (VMOVAPSmr addr:$dst, VR128:$src)>;
  def : Pat<(alignedstore (v4i32 VR128:$src), addr:$dst),
            (VMOVAPSmr addr:$dst, VR128:$src)>;
  def : Pat<(alignedstore (v8i16 VR128:$src), addr:$dst),
            (VMOVAPSmr addr:$dst, VR128:$src)>;
  def : Pat<(alignedstore (v16i8 VR128:$src), addr:$dst),
            (VMOVAPSmr addr:$dst, VR128:$src)>;
  def : Pat<(store (v2i64 VR128:$src), addr:$dst),
            (VMOVUPSmr addr:$dst, VR128:$src)>;
  def : Pat<(store (v4i32 VR128:$src), addr:$dst),
            (VMOVUPSmr addr:$dst, VR128:$src)>;
  def : Pat<(store (v8i16 VR128:$src), addr:$dst),
            (VMOVUPSmr addr:$dst, VR128:$src)>;
  def : Pat<(store (v16i8 VR128:$src), addr:$dst),
            (VMOVUPSmr addr:$dst, VR128:$src)>;

  // 256-bit load/store
  def : Pat<(alignedloadv4i64 addr:$src),
            (VMOVAPSYrm addr:$src)>;
  def : Pat<(loadv4i64 addr:$src),
            (VMOVUPSYrm addr:$src)>;
  def : Pat<(alignedstore256 (v4i64 VR256:$src), addr:$dst),
            (VMOVAPSYmr addr:$dst, VR256:$src)>;
  def : Pat<(alignedstore256 (v8i32 VR256:$src), addr:$dst),
            (VMOVAPSYmr addr:$dst, VR256:$src)>;
  def : Pat<(alignedstore256 (v16i16 VR256:$src), addr:$dst),
            (VMOVAPSYmr addr:$dst, VR256:$src)>;
  def : Pat<(alignedstore256 (v32i8 VR256:$src), addr:$dst),
            (VMOVAPSYmr addr:$dst, VR256:$src)>;
  def : Pat<(store (v4i64 VR256:$src), addr:$dst),
            (VMOVUPSYmr addr:$dst, VR256:$src)>;
  def : Pat<(store (v8i32 VR256:$src), addr:$dst),
            (VMOVUPSYmr addr:$dst, VR256:$src)>;
  def : Pat<(store (v16i16 VR256:$src), addr:$dst),
            (VMOVUPSYmr addr:$dst, VR256:$src)>;
  def : Pat<(store (v32i8 VR256:$src), addr:$dst),
            (VMOVUPSYmr addr:$dst, VR256:$src)>;

  // Special patterns for storing subvector extracts of lower 128-bits
  // Its cheaper to just use VMOVAPS/VMOVUPS instead of VEXTRACTF128mr
  def : Pat<(alignedstore (v2f64 (extract_subvector
                                  (v4f64 VR256:$src), (iPTR 0))), addr:$dst),
            (VMOVAPDmr addr:$dst, (v2f64 (EXTRACT_SUBREG VR256:$src,sub_xmm)))>;
  def : Pat<(alignedstore (v4f32 (extract_subvector
                                  (v8f32 VR256:$src), (iPTR 0))), addr:$dst),
            (VMOVAPSmr addr:$dst, (v4f32 (EXTRACT_SUBREG VR256:$src,sub_xmm)))>;
  def : Pat<(alignedstore (v2i64 (extract_subvector
                                  (v4i64 VR256:$src), (iPTR 0))), addr:$dst),
            (VMOVAPDmr addr:$dst, (v2i64 (EXTRACT_SUBREG VR256:$src,sub_xmm)))>;
  def : Pat<(alignedstore (v4i32 (extract_subvector
                                  (v8i32 VR256:$src), (iPTR 0))), addr:$dst),
            (VMOVAPSmr addr:$dst, (v4i32 (EXTRACT_SUBREG VR256:$src,sub_xmm)))>;
  def : Pat<(alignedstore (v8i16 (extract_subvector
                                  (v16i16 VR256:$src), (iPTR 0))), addr:$dst),
            (VMOVAPSmr addr:$dst, (v8i16 (EXTRACT_SUBREG VR256:$src,sub_xmm)))>;
  def : Pat<(alignedstore (v16i8 (extract_subvector
                                  (v32i8 VR256:$src), (iPTR 0))), addr:$dst),
            (VMOVAPSmr addr:$dst, (v16i8 (EXTRACT_SUBREG VR256:$src,sub_xmm)))>;

  def : Pat<(store (v2f64 (extract_subvector
                           (v4f64 VR256:$src), (iPTR 0))), addr:$dst),
            (VMOVUPDmr addr:$dst, (v2f64 (EXTRACT_SUBREG VR256:$src,sub_xmm)))>;
  def : Pat<(store (v4f32 (extract_subvector
                           (v8f32 VR256:$src), (iPTR 0))), addr:$dst),
            (VMOVUPSmr addr:$dst, (v4f32 (EXTRACT_SUBREG VR256:$src,sub_xmm)))>;
  def : Pat<(store (v2i64 (extract_subvector
                           (v4i64 VR256:$src), (iPTR 0))), addr:$dst),
            (VMOVUPDmr addr:$dst, (v2i64 (EXTRACT_SUBREG VR256:$src,sub_xmm)))>;
  def : Pat<(store (v4i32 (extract_subvector
                           (v8i32 VR256:$src), (iPTR 0))), addr:$dst),
            (VMOVUPSmr addr:$dst, (v4i32 (EXTRACT_SUBREG VR256:$src,sub_xmm)))>;
  def : Pat<(store (v8i16 (extract_subvector
                           (v16i16 VR256:$src), (iPTR 0))), addr:$dst),
            (VMOVUPSmr addr:$dst, (v8i16 (EXTRACT_SUBREG VR256:$src,sub_xmm)))>;
  def : Pat<(store (v16i8 (extract_subvector
                           (v32i8 VR256:$src), (iPTR 0))), addr:$dst),
            (VMOVUPSmr addr:$dst, (v16i8 (EXTRACT_SUBREG VR256:$src,sub_xmm)))>;
}

// Use movaps / movups for SSE integer load / store (one byte shorter).
// The instructions selected below are then converted to MOVDQA/MOVDQU
// during the SSE domain pass.
let Predicates = [UseSSE1] in {
  def : Pat<(alignedloadv2i64 addr:$src),
            (MOVAPSrm addr:$src)>;
  def : Pat<(loadv2i64 addr:$src),
            (MOVUPSrm addr:$src)>;

  def : Pat<(alignedstore (v2i64 VR128:$src), addr:$dst),
            (MOVAPSmr addr:$dst, VR128:$src)>;
  def : Pat<(alignedstore (v4i32 VR128:$src), addr:$dst),
            (MOVAPSmr addr:$dst, VR128:$src)>;
  def : Pat<(alignedstore (v8i16 VR128:$src), addr:$dst),
            (MOVAPSmr addr:$dst, VR128:$src)>;
  def : Pat<(alignedstore (v16i8 VR128:$src), addr:$dst),
            (MOVAPSmr addr:$dst, VR128:$src)>;
  def : Pat<(store (v2i64 VR128:$src), addr:$dst),
            (MOVUPSmr addr:$dst, VR128:$src)>;
  def : Pat<(store (v4i32 VR128:$src), addr:$dst),
            (MOVUPSmr addr:$dst, VR128:$src)>;
  def : Pat<(store (v8i16 VR128:$src), addr:$dst),
            (MOVUPSmr addr:$dst, VR128:$src)>;
  def : Pat<(store (v16i8 VR128:$src), addr:$dst),
            (MOVUPSmr addr:$dst, VR128:$src)>;
}

// Alias instruction to load FR32 or FR64 from f128mem using movaps. Upper
// bits are disregarded. FIXME: Set encoding to pseudo!
let canFoldAsLoad = 1, isReMaterializable = 1, SchedRW = [WriteLoad] in {
let isCodeGenOnly = 1 in {
  def FsVMOVAPSrm : VPSI<0x28, MRMSrcMem, (outs FR32:$dst), (ins f128mem:$src),
                         "movaps\t{$src, $dst|$dst, $src}",
                         [(set FR32:$dst, (alignedloadfsf32 addr:$src))],
                         IIC_SSE_MOVA_P_RM>, VEX;
  def FsVMOVAPDrm : VPDI<0x28, MRMSrcMem, (outs FR64:$dst), (ins f128mem:$src),
                         "movapd\t{$src, $dst|$dst, $src}",
                         [(set FR64:$dst, (alignedloadfsf64 addr:$src))],
                         IIC_SSE_MOVA_P_RM>, VEX;
  def FsMOVAPSrm : PSI<0x28, MRMSrcMem, (outs FR32:$dst), (ins f128mem:$src),
                       "movaps\t{$src, $dst|$dst, $src}",
                       [(set FR32:$dst, (alignedloadfsf32 addr:$src))],
                       IIC_SSE_MOVA_P_RM>;
  def FsMOVAPDrm : PDI<0x28, MRMSrcMem, (outs FR64:$dst), (ins f128mem:$src),
                       "movapd\t{$src, $dst|$dst, $src}",
                       [(set FR64:$dst, (alignedloadfsf64 addr:$src))],
                       IIC_SSE_MOVA_P_RM>;
}
}

//===----------------------------------------------------------------------===//
// SSE 1 & 2 - Move Low packed FP Instructions
//===----------------------------------------------------------------------===//

multiclass sse12_mov_hilo_packed_base<bits<8>opc, SDNode psnode, SDNode pdnode,
                                      string base_opc, string asm_opr,
                                      InstrItinClass itin> {
  def PSrm : PI<opc, MRMSrcMem,
         (outs VR128:$dst), (ins VR128:$src1, f64mem:$src2),
         !strconcat(base_opc, "s", asm_opr),
     [(set VR128:$dst,
       (psnode VR128:$src1,
              (bc_v4f32 (v2f64 (scalar_to_vector (loadf64 addr:$src2))))))],
              itin, SSEPackedSingle>, TB,
     Sched<[WriteShuffleLd, ReadAfterLd]>;

  def PDrm : PI<opc, MRMSrcMem,
         (outs VR128:$dst), (ins VR128:$src1, f64mem:$src2),
         !strconcat(base_opc, "d", asm_opr),
     [(set VR128:$dst, (v2f64 (pdnode VR128:$src1,
                              (scalar_to_vector (loadf64 addr:$src2)))))],
              itin, SSEPackedDouble>, TB, OpSize,
     Sched<[WriteShuffleLd, ReadAfterLd]>;

}

multiclass sse12_mov_hilo_packed<bits<8>opc, SDNode psnode, SDNode pdnode,
                                 string base_opc, InstrItinClass itin> {
  defm V#NAME : sse12_mov_hilo_packed_base<opc, psnode, pdnode, base_opc,
                                    "\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                                    itin>, VEX_4V;

let Constraints = "$src1 = $dst" in
  defm NAME : sse12_mov_hilo_packed_base<opc, psnode, pdnode, base_opc,
                                    "\t{$src2, $dst|$dst, $src2}",
                                    itin>;
}

let AddedComplexity = 20 in {
  defm MOVL : sse12_mov_hilo_packed<0x12, X86Movlps, X86Movlpd, "movlp",
                                    IIC_SSE_MOV_LH>;
}

let SchedRW = [WriteStore] in {
def VMOVLPSmr : VPSI<0x13, MRMDestMem, (outs), (ins f64mem:$dst, VR128:$src),
                   "movlps\t{$src, $dst|$dst, $src}",
                   [(store (f64 (vector_extract (bc_v2f64 (v4f32 VR128:$src)),
                                 (iPTR 0))), addr:$dst)],
                                 IIC_SSE_MOV_LH>, VEX;
def VMOVLPDmr : VPDI<0x13, MRMDestMem, (outs), (ins f64mem:$dst, VR128:$src),
                   "movlpd\t{$src, $dst|$dst, $src}",
                   [(store (f64 (vector_extract (v2f64 VR128:$src),
                                 (iPTR 0))), addr:$dst)],
                                 IIC_SSE_MOV_LH>, VEX;
def MOVLPSmr : PSI<0x13, MRMDestMem, (outs), (ins f64mem:$dst, VR128:$src),
                   "movlps\t{$src, $dst|$dst, $src}",
                   [(store (f64 (vector_extract (bc_v2f64 (v4f32 VR128:$src)),
                                 (iPTR 0))), addr:$dst)],
                                 IIC_SSE_MOV_LH>;
def MOVLPDmr : PDI<0x13, MRMDestMem, (outs), (ins f64mem:$dst, VR128:$src),
                   "movlpd\t{$src, $dst|$dst, $src}",
                   [(store (f64 (vector_extract (v2f64 VR128:$src),
                                 (iPTR 0))), addr:$dst)],
                                 IIC_SSE_MOV_LH>;
} // SchedRW

let Predicates = [HasAVX] in {
  // Shuffle with VMOVLPS
  def : Pat<(v4f32 (X86Movlps VR128:$src1, (load addr:$src2))),
            (VMOVLPSrm VR128:$src1, addr:$src2)>;
  def : Pat<(v4i32 (X86Movlps VR128:$src1, (load addr:$src2))),
            (VMOVLPSrm VR128:$src1, addr:$src2)>;

  // Shuffle with VMOVLPD
  def : Pat<(v2f64 (X86Movlpd VR128:$src1, (load addr:$src2))),
            (VMOVLPDrm VR128:$src1, addr:$src2)>;
  def : Pat<(v2i64 (X86Movlpd VR128:$src1, (load addr:$src2))),
            (VMOVLPDrm VR128:$src1, addr:$src2)>;

  // Store patterns
  def : Pat<(store (v4f32 (X86Movlps (load addr:$src1), VR128:$src2)),
                   addr:$src1),
            (VMOVLPSmr addr:$src1, VR128:$src2)>;
  def : Pat<(store (v4i32 (X86Movlps
                   (bc_v4i32 (loadv2i64 addr:$src1)), VR128:$src2)), addr:$src1),
            (VMOVLPSmr addr:$src1, VR128:$src2)>;
  def : Pat<(store (v2f64 (X86Movlpd (load addr:$src1), VR128:$src2)),
                   addr:$src1),
            (VMOVLPDmr addr:$src1, VR128:$src2)>;
  def : Pat<(store (v2i64 (X86Movlpd (load addr:$src1), VR128:$src2)),
                   addr:$src1),
            (VMOVLPDmr addr:$src1, VR128:$src2)>;
}

let Predicates = [UseSSE1] in {
  // (store (vector_shuffle (load addr), v2, <4, 5, 2, 3>), addr) using MOVLPS
  def : Pat<(store (i64 (vector_extract (bc_v2i64 (v4f32 VR128:$src2)),
                                 (iPTR 0))), addr:$src1),
            (MOVLPSmr addr:$src1, VR128:$src2)>;

  // Shuffle with MOVLPS
  def : Pat<(v4f32 (X86Movlps VR128:$src1, (load addr:$src2))),
            (MOVLPSrm VR128:$src1, addr:$src2)>;
  def : Pat<(v4i32 (X86Movlps VR128:$src1, (load addr:$src2))),
            (MOVLPSrm VR128:$src1, addr:$src2)>;
  def : Pat<(X86Movlps VR128:$src1,
                      (bc_v4f32 (v2i64 (scalar_to_vector (loadi64 addr:$src2))))),
            (MOVLPSrm VR128:$src1, addr:$src2)>;

  // Store patterns
  def : Pat<(store (v4f32 (X86Movlps (load addr:$src1), VR128:$src2)),
                                      addr:$src1),
            (MOVLPSmr addr:$src1, VR128:$src2)>;
  def : Pat<(store (v4i32 (X86Movlps
                   (bc_v4i32 (loadv2i64 addr:$src1)), VR128:$src2)),
                              addr:$src1),
            (MOVLPSmr addr:$src1, VR128:$src2)>;
}

let Predicates = [UseSSE2] in {
  // Shuffle with MOVLPD
  def : Pat<(v2f64 (X86Movlpd VR128:$src1, (load addr:$src2))),
            (MOVLPDrm VR128:$src1, addr:$src2)>;
  def : Pat<(v2i64 (X86Movlpd VR128:$src1, (load addr:$src2))),
            (MOVLPDrm VR128:$src1, addr:$src2)>;

  // Store patterns
  def : Pat<(store (v2f64 (X86Movlpd (load addr:$src1), VR128:$src2)),
                           addr:$src1),
            (MOVLPDmr addr:$src1, VR128:$src2)>;
  def : Pat<(store (v2i64 (X86Movlpd (load addr:$src1), VR128:$src2)),
                           addr:$src1),
            (MOVLPDmr addr:$src1, VR128:$src2)>;
}

//===----------------------------------------------------------------------===//
// SSE 1 & 2 - Move Hi packed FP Instructions
//===----------------------------------------------------------------------===//

let AddedComplexity = 20 in {
  defm MOVH : sse12_mov_hilo_packed<0x16, X86Movlhps, X86Movlhpd, "movhp",
                                    IIC_SSE_MOV_LH>;
}

let SchedRW = [WriteStore] in {
// v2f64 extract element 1 is always custom lowered to unpack high to low
// and extract element 0 so the non-store version isn't too horrible.
def VMOVHPSmr : VPSI<0x17, MRMDestMem, (outs), (ins f64mem:$dst, VR128:$src),
                   "movhps\t{$src, $dst|$dst, $src}",
                   [(store (f64 (vector_extract
                                 (X86Unpckh (bc_v2f64 (v4f32 VR128:$src)),
                                            (bc_v2f64 (v4f32 VR128:$src))),
                                 (iPTR 0))), addr:$dst)], IIC_SSE_MOV_LH>, VEX;
def VMOVHPDmr : VPDI<0x17, MRMDestMem, (outs), (ins f64mem:$dst, VR128:$src),
                   "movhpd\t{$src, $dst|$dst, $src}",
                   [(store (f64 (vector_extract
                                 (v2f64 (X86Unpckh VR128:$src, VR128:$src)),
                                 (iPTR 0))), addr:$dst)], IIC_SSE_MOV_LH>, VEX;
def MOVHPSmr : PSI<0x17, MRMDestMem, (outs), (ins f64mem:$dst, VR128:$src),
                   "movhps\t{$src, $dst|$dst, $src}",
                   [(store (f64 (vector_extract
                                 (X86Unpckh (bc_v2f64 (v4f32 VR128:$src)),
                                            (bc_v2f64 (v4f32 VR128:$src))),
                                 (iPTR 0))), addr:$dst)], IIC_SSE_MOV_LH>;
def MOVHPDmr : PDI<0x17, MRMDestMem, (outs), (ins f64mem:$dst, VR128:$src),
                   "movhpd\t{$src, $dst|$dst, $src}",
                   [(store (f64 (vector_extract
                                 (v2f64 (X86Unpckh VR128:$src, VR128:$src)),
                                 (iPTR 0))), addr:$dst)], IIC_SSE_MOV_LH>;
} // SchedRW

let Predicates = [HasAVX] in {
  // VMOVHPS patterns
  def : Pat<(X86Movlhps VR128:$src1,
                 (bc_v4f32 (v2i64 (scalar_to_vector (loadi64 addr:$src2))))),
            (VMOVHPSrm VR128:$src1, addr:$src2)>;
  def : Pat<(X86Movlhps VR128:$src1,
                 (bc_v4i32 (v2i64 (X86vzload addr:$src2)))),
            (VMOVHPSrm VR128:$src1, addr:$src2)>;

  // FIXME: Instead of X86Unpckl, there should be a X86Movlhpd here, the problem
  // is during lowering, where it's not possible to recognize the load fold
  // cause it has two uses through a bitcast. One use disappears at isel time
  // and the fold opportunity reappears.
  def : Pat<(v2f64 (X86Unpckl VR128:$src1,
                      (scalar_to_vector (loadf64 addr:$src2)))),
            (VMOVHPDrm VR128:$src1, addr:$src2)>;
}

let Predicates = [UseSSE1] in {
  // MOVHPS patterns
  def : Pat<(X86Movlhps VR128:$src1,
                 (bc_v4f32 (v2i64 (scalar_to_vector (loadi64 addr:$src2))))),
            (MOVHPSrm VR128:$src1, addr:$src2)>;
  def : Pat<(X86Movlhps VR128:$src1,
                 (bc_v4f32 (v2i64 (X86vzload addr:$src2)))),
            (MOVHPSrm VR128:$src1, addr:$src2)>;
}

let Predicates = [UseSSE2] in {
  // FIXME: Instead of X86Unpckl, there should be a X86Movlhpd here, the problem
  // is during lowering, where it's not possible to recognize the load fold
  // cause it has two uses through a bitcast. One use disappears at isel time
  // and the fold opportunity reappears.
  def : Pat<(v2f64 (X86Unpckl VR128:$src1,
                      (scalar_to_vector (loadf64 addr:$src2)))),
            (MOVHPDrm VR128:$src1, addr:$src2)>;
}

//===----------------------------------------------------------------------===//
// SSE 1 & 2 - Move Low to High and High to Low packed FP Instructions
//===----------------------------------------------------------------------===//

let AddedComplexity = 20, Predicates = [UseAVX] in {
  def VMOVLHPSrr : VPSI<0x16, MRMSrcReg, (outs VR128:$dst),
                                       (ins VR128:$src1, VR128:$src2),
                      "movlhps\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                      [(set VR128:$dst,
                        (v4f32 (X86Movlhps VR128:$src1, VR128:$src2)))],
                        IIC_SSE_MOV_LH>,
                      VEX_4V, Sched<[WriteShuffle]>;
  def VMOVHLPSrr : VPSI<0x12, MRMSrcReg, (outs VR128:$dst),
                                       (ins VR128:$src1, VR128:$src2),
                      "movhlps\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                      [(set VR128:$dst,
                        (v4f32 (X86Movhlps VR128:$src1, VR128:$src2)))],
                        IIC_SSE_MOV_LH>,
                      VEX_4V, Sched<[WriteShuffle]>;
}
let Constraints = "$src1 = $dst", AddedComplexity = 20 in {
  def MOVLHPSrr : PSI<0x16, MRMSrcReg, (outs VR128:$dst),
                                       (ins VR128:$src1, VR128:$src2),
                      "movlhps\t{$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst,
                        (v4f32 (X86Movlhps VR128:$src1, VR128:$src2)))],
                        IIC_SSE_MOV_LH>, Sched<[WriteShuffle]>;
  def MOVHLPSrr : PSI<0x12, MRMSrcReg, (outs VR128:$dst),
                                       (ins VR128:$src1, VR128:$src2),
                      "movhlps\t{$src2, $dst|$dst, $src2}",
                      [(set VR128:$dst,
                        (v4f32 (X86Movhlps VR128:$src1, VR128:$src2)))],
                        IIC_SSE_MOV_LH>, Sched<[WriteShuffle]>;
}

let Predicates = [UseAVX] in {
  // MOVLHPS patterns
  def : Pat<(v4i32 (X86Movlhps VR128:$src1, VR128:$src2)),
            (VMOVLHPSrr VR128:$src1, VR128:$src2)>;
  def : Pat<(v2i64 (X86Movlhps VR128:$src1, VR128:$src2)),
            (VMOVLHPSrr (v2i64 VR128:$src1), VR128:$src2)>;

  // MOVHLPS patterns
  def : Pat<(v4i32 (X86Movhlps VR128:$src1, VR128:$src2)),
            (VMOVHLPSrr VR128:$src1, VR128:$src2)>;
}

let Predicates = [UseSSE1] in {
  // MOVLHPS patterns
  def : Pat<(v4i32 (X86Movlhps VR128:$src1, VR128:$src2)),
            (MOVLHPSrr VR128:$src1, VR128:$src2)>;
  def : Pat<(v2i64 (X86Movlhps VR128:$src1, VR128:$src2)),
            (MOVLHPSrr (v2i64 VR128:$src1), VR128:$src2)>;

  // MOVHLPS patterns
  def : Pat<(v4i32 (X86Movhlps VR128:$src1, VR128:$src2)),
            (MOVHLPSrr VR128:$src1, VR128:$src2)>;
}

//===----------------------------------------------------------------------===//
// SSE 1 & 2 - Conversion Instructions
//===----------------------------------------------------------------------===//

def SSE_CVT_PD : OpndItins<
  IIC_SSE_CVT_PD_RR, IIC_SSE_CVT_PD_RM
>;

let Sched = WriteCvtI2F in
def SSE_CVT_PS : OpndItins<
  IIC_SSE_CVT_PS_RR, IIC_SSE_CVT_PS_RM
>;

let Sched = WriteCvtI2F in
def SSE_CVT_Scalar : OpndItins<
  IIC_SSE_CVT_Scalar_RR, IIC_SSE_CVT_Scalar_RM
>;

let Sched = WriteCvtF2I in
def SSE_CVT_SS2SI_32 : OpndItins<
  IIC_SSE_CVT_SS2SI32_RR, IIC_SSE_CVT_SS2SI32_RM
>;

let Sched = WriteCvtF2I in
def SSE_CVT_SS2SI_64 : OpndItins<
  IIC_SSE_CVT_SS2SI64_RR, IIC_SSE_CVT_SS2SI64_RM
>;

let Sched = WriteCvtF2I in
def SSE_CVT_SD2SI : OpndItins<
  IIC_SSE_CVT_SD2SI_RR, IIC_SSE_CVT_SD2SI_RM
>;

multiclass sse12_cvt_s<bits<8> opc, RegisterClass SrcRC, RegisterClass DstRC,
                     SDNode OpNode, X86MemOperand x86memop, PatFrag ld_frag,
                     string asm, OpndItins itins> {
  def rr : SI<opc, MRMSrcReg, (outs DstRC:$dst), (ins SrcRC:$src), asm,
                        [(set DstRC:$dst, (OpNode SrcRC:$src))],
                        itins.rr>, Sched<[itins.Sched]>;
  def rm : SI<opc, MRMSrcMem, (outs DstRC:$dst), (ins x86memop:$src), asm,
                        [(set DstRC:$dst, (OpNode (ld_frag addr:$src)))],
                        itins.rm>, Sched<[itins.Sched.Folded]>;
}

multiclass sse12_cvt_p<bits<8> opc, RegisterClass SrcRC, RegisterClass DstRC,
                       X86MemOperand x86memop, string asm, Domain d,
                       OpndItins itins> {
let neverHasSideEffects = 1 in {
  def rr : I<opc, MRMSrcReg, (outs DstRC:$dst), (ins SrcRC:$src), asm,
             [], itins.rr, d>, Sched<[itins.Sched]>;
  let mayLoad = 1 in
  def rm : I<opc, MRMSrcMem, (outs DstRC:$dst), (ins x86memop:$src), asm,
             [], itins.rm, d>, Sched<[itins.Sched.Folded]>;
}
}

multiclass sse12_vcvt_avx<bits<8> opc, RegisterClass SrcRC, RegisterClass DstRC,
                          X86MemOperand x86memop, string asm> {
let neverHasSideEffects = 1, Predicates = [UseAVX] in {
  def rr : SI<opc, MRMSrcReg, (outs DstRC:$dst), (ins DstRC:$src1, SrcRC:$src),
              !strconcat(asm,"\t{$src, $src1, $dst|$dst, $src1, $src}"), []>,
           Sched<[WriteCvtI2F]>;
  let mayLoad = 1 in
  def rm : SI<opc, MRMSrcMem, (outs DstRC:$dst),
              (ins DstRC:$src1, x86memop:$src),
              !strconcat(asm,"\t{$src, $src1, $dst|$dst, $src1, $src}"), []>,
           Sched<[WriteCvtI2FLd, ReadAfterLd]>;
} // neverHasSideEffects = 1
}

let Predicates = [UseAVX] in {
defm VCVTTSS2SI   : sse12_cvt_s<0x2C, FR32, GR32, fp_to_sint, f32mem, loadf32,
                                "cvttss2si\t{$src, $dst|$dst, $src}",
                                SSE_CVT_SS2SI_32>,
                                XS, VEX, VEX_LIG;
defm VCVTTSS2SI64 : sse12_cvt_s<0x2C, FR32, GR64, fp_to_sint, f32mem, loadf32,
                                "cvttss2si\t{$src, $dst|$dst, $src}",
                                SSE_CVT_SS2SI_64>,
                                XS, VEX, VEX_W, VEX_LIG;
defm VCVTTSD2SI   : sse12_cvt_s<0x2C, FR64, GR32, fp_to_sint, f64mem, loadf64,
                                "cvttsd2si\t{$src, $dst|$dst, $src}",
                                SSE_CVT_SD2SI>,
                                XD, VEX, VEX_LIG;
defm VCVTTSD2SI64 : sse12_cvt_s<0x2C, FR64, GR64, fp_to_sint, f64mem, loadf64,
                                "cvttsd2si\t{$src, $dst|$dst, $src}",
                                SSE_CVT_SD2SI>,
                                XD, VEX, VEX_W, VEX_LIG;

def : InstAlias<"vcvttss2si{l}\t{$src, $dst|$dst, $src}",
                (VCVTTSS2SIrr GR32:$dst, FR32:$src), 0>;
def : InstAlias<"vcvttss2si{l}\t{$src, $dst|$dst, $src}",
                (VCVTTSS2SIrm GR32:$dst, f32mem:$src), 0>;
def : InstAlias<"vcvttsd2si{l}\t{$src, $dst|$dst, $src}",
                (VCVTTSD2SIrr GR32:$dst, FR64:$src), 0>;
def : InstAlias<"vcvttsd2si{l}\t{$src, $dst|$dst, $src}",
                (VCVTTSD2SIrm GR32:$dst, f64mem:$src), 0>;
def : InstAlias<"vcvttss2si{q}\t{$src, $dst|$dst, $src}",
                (VCVTTSS2SI64rr GR64:$dst, FR32:$src), 0>;
def : InstAlias<"vcvttss2si{q}\t{$src, $dst|$dst, $src}",
                (VCVTTSS2SI64rm GR64:$dst, f32mem:$src), 0>;
def : InstAlias<"vcvttsd2si{q}\t{$src, $dst|$dst, $src}",
                (VCVTTSD2SI64rr GR64:$dst, FR64:$src), 0>;
def : InstAlias<"vcvttsd2si{q}\t{$src, $dst|$dst, $src}",
                (VCVTTSD2SI64rm GR64:$dst, f64mem:$src), 0>;
}
// The assembler can recognize rr 64-bit instructions by seeing a rxx
// register, but the same isn't true when only using memory operands,
// provide other assembly "l" and "q" forms to address this explicitly
// where appropriate to do so.
defm VCVTSI2SS   : sse12_vcvt_avx<0x2A, GR32, FR32, i32mem, "cvtsi2ss{l}">,
                                  XS, VEX_4V, VEX_LIG;
defm VCVTSI2SS64 : sse12_vcvt_avx<0x2A, GR64, FR32, i64mem, "cvtsi2ss{q}">,
                                  XS, VEX_4V, VEX_W, VEX_LIG;
defm VCVTSI2SD   : sse12_vcvt_avx<0x2A, GR32, FR64, i32mem, "cvtsi2sd{l}">,
                                  XD, VEX_4V, VEX_LIG;
defm VCVTSI2SD64 : sse12_vcvt_avx<0x2A, GR64, FR64, i64mem, "cvtsi2sd{q}">,
                                  XD, VEX_4V, VEX_W, VEX_LIG;

let Predicates = [UseAVX] in {
  def : InstAlias<"vcvtsi2ss\t{$src, $src1, $dst|$dst, $src1, $src}",
                (VCVTSI2SSrm FR64:$dst, FR64:$src1, i32mem:$src)>;
  def : InstAlias<"vcvtsi2sd\t{$src, $src1, $dst|$dst, $src1, $src}",
                (VCVTSI2SDrm FR64:$dst, FR64:$src1, i32mem:$src)>;

  def : Pat<(f32 (sint_to_fp (loadi32 addr:$src))),
            (VCVTSI2SSrm (f32 (IMPLICIT_DEF)), addr:$src)>;
  def : Pat<(f32 (sint_to_fp (loadi64 addr:$src))),
            (VCVTSI2SS64rm (f32 (IMPLICIT_DEF)), addr:$src)>;
  def : Pat<(f64 (sint_to_fp (loadi32 addr:$src))),
            (VCVTSI2SDrm (f64 (IMPLICIT_DEF)), addr:$src)>;
  def : Pat<(f64 (sint_to_fp (loadi64 addr:$src))),
            (VCVTSI2SD64rm (f64 (IMPLICIT_DEF)), addr:$src)>;

  def : Pat<(f32 (sint_to_fp GR32:$src)),
            (VCVTSI2SSrr (f32 (IMPLICIT_DEF)), GR32:$src)>;
  def : Pat<(f32 (sint_to_fp GR64:$src)),
            (VCVTSI2SS64rr (f32 (IMPLICIT_DEF)), GR64:$src)>;
  def : Pat<(f64 (sint_to_fp GR32:$src)),
            (VCVTSI2SDrr (f64 (IMPLICIT_DEF)), GR32:$src)>;
  def : Pat<(f64 (sint_to_fp GR64:$src)),
            (VCVTSI2SD64rr (f64 (IMPLICIT_DEF)), GR64:$src)>;
}

defm CVTTSS2SI : sse12_cvt_s<0x2C, FR32, GR32, fp_to_sint, f32mem, loadf32,
                      "cvttss2si\t{$src, $dst|$dst, $src}",
                      SSE_CVT_SS2SI_32>, XS;
defm CVTTSS2SI64 : sse12_cvt_s<0x2C, FR32, GR64, fp_to_sint, f32mem, loadf32,
                      "cvttss2si\t{$src, $dst|$dst, $src}",
                      SSE_CVT_SS2SI_64>, XS, REX_W;
defm CVTTSD2SI : sse12_cvt_s<0x2C, FR64, GR32, fp_to_sint, f64mem, loadf64,
                      "cvttsd2si\t{$src, $dst|$dst, $src}",
                      SSE_CVT_SD2SI>, XD;
defm CVTTSD2SI64 : sse12_cvt_s<0x2C, FR64, GR64, fp_to_sint, f64mem, loadf64,
                      "cvttsd2si\t{$src, $dst|$dst, $src}",
                      SSE_CVT_SD2SI>, XD, REX_W;
defm CVTSI2SS  : sse12_cvt_s<0x2A, GR32, FR32, sint_to_fp, i32mem, loadi32,
                      "cvtsi2ss{l}\t{$src, $dst|$dst, $src}",
                      SSE_CVT_Scalar>, XS;
defm CVTSI2SS64 : sse12_cvt_s<0x2A, GR64, FR32, sint_to_fp, i64mem, loadi64,
                      "cvtsi2ss{q}\t{$src, $dst|$dst, $src}",
                      SSE_CVT_Scalar>, XS, REX_W;
defm CVTSI2SD  : sse12_cvt_s<0x2A, GR32, FR64, sint_to_fp, i32mem, loadi32,
                      "cvtsi2sd{l}\t{$src, $dst|$dst, $src}",
                      SSE_CVT_Scalar>, XD;
defm CVTSI2SD64 : sse12_cvt_s<0x2A, GR64, FR64, sint_to_fp, i64mem, loadi64,
                      "cvtsi2sd{q}\t{$src, $dst|$dst, $src}",
                      SSE_CVT_Scalar>, XD, REX_W;

def : InstAlias<"cvttss2si{l}\t{$src, $dst|$dst, $src}",
                (CVTTSS2SIrr GR32:$dst, FR32:$src), 0>;
def : InstAlias<"cvttss2si{l}\t{$src, $dst|$dst, $src}",
                (CVTTSS2SIrm GR32:$dst, f32mem:$src), 0>;
def : InstAlias<"cvttsd2si{l}\t{$src, $dst|$dst, $src}",
                (CVTTSD2SIrr GR32:$dst, FR64:$src), 0>;
def : InstAlias<"cvttsd2si{l}\t{$src, $dst|$dst, $src}",
                (CVTTSD2SIrm GR32:$dst, f64mem:$src), 0>;
def : InstAlias<"cvttss2si{q}\t{$src, $dst|$dst, $src}",
                (CVTTSS2SI64rr GR64:$dst, FR32:$src), 0>;
def : InstAlias<"cvttss2si{q}\t{$src, $dst|$dst, $src}",
                (CVTTSS2SI64rm GR64:$dst, f32mem:$src), 0>;
def : InstAlias<"cvttsd2si{q}\t{$src, $dst|$dst, $src}",
                (CVTTSD2SI64rr GR64:$dst, FR64:$src), 0>;
def : InstAlias<"cvttsd2si{q}\t{$src, $dst|$dst, $src}",
                (CVTTSD2SI64rm GR64:$dst, f64mem:$src), 0>;

def : InstAlias<"cvtsi2ss\t{$src, $dst|$dst, $src}",
                (CVTSI2SSrm FR64:$dst, i32mem:$src)>;
def : InstAlias<"cvtsi2sd\t{$src, $dst|$dst, $src}",
                (CVTSI2SDrm FR64:$dst, i32mem:$src)>;

// Conversion Instructions Intrinsics - Match intrinsics which expect MM
// and/or XMM operand(s).

multiclass sse12_cvt_sint<bits<8> opc, RegisterClass SrcRC, RegisterClass DstRC,
                         Intrinsic Int, Operand memop, ComplexPattern mem_cpat,
                         string asm, OpndItins itins> {
  def rr : SI<opc, MRMSrcReg, (outs DstRC:$dst), (ins SrcRC:$src),
              !strconcat(asm, "\t{$src, $dst|$dst, $src}"),
              [(set DstRC:$dst, (Int SrcRC:$src))], itins.rr>,
           Sched<[itins.Sched]>;
  def rm : SI<opc, MRMSrcMem, (outs DstRC:$dst), (ins memop:$src),
              !strconcat(asm, "\t{$src, $dst|$dst, $src}"),
              [(set DstRC:$dst, (Int mem_cpat:$src))], itins.rm>,
           Sched<[itins.Sched.Folded]>;
}

multiclass sse12_cvt_sint_3addr<bits<8> opc, RegisterClass SrcRC,
                    RegisterClass DstRC, Intrinsic Int, X86MemOperand x86memop,
                    PatFrag ld_frag, string asm, OpndItins itins,
                    bit Is2Addr = 1> {
  def rr : SI<opc, MRMSrcReg, (outs DstRC:$dst), (ins DstRC:$src1, SrcRC:$src2),
              !if(Is2Addr,
                  !strconcat(asm, "\t{$src2, $dst|$dst, $src2}"),
                  !strconcat(asm, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
              [(set DstRC:$dst, (Int DstRC:$src1, SrcRC:$src2))],
              itins.rr>, Sched<[itins.Sched]>;
  def rm : SI<opc, MRMSrcMem, (outs DstRC:$dst),
              (ins DstRC:$src1, x86memop:$src2),
              !if(Is2Addr,
                  !strconcat(asm, "\t{$src2, $dst|$dst, $src2}"),
                  !strconcat(asm, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
              [(set DstRC:$dst, (Int DstRC:$src1, (ld_frag addr:$src2)))],
              itins.rm>, Sched<[itins.Sched.Folded, ReadAfterLd]>;
}

let Predicates = [UseAVX] in {
defm VCVTSD2SI : sse12_cvt_sint<0x2D, VR128, GR32,
                  int_x86_sse2_cvtsd2si, sdmem, sse_load_f64, "cvtsd2si",
                  SSE_CVT_SD2SI>, XD, VEX, VEX_LIG;
defm VCVTSD2SI64 : sse12_cvt_sint<0x2D, VR128, GR64,
                    int_x86_sse2_cvtsd2si64, sdmem, sse_load_f64, "cvtsd2si",
                    SSE_CVT_SD2SI>, XD, VEX, VEX_W, VEX_LIG;
}
defm CVTSD2SI : sse12_cvt_sint<0x2D, VR128, GR32, int_x86_sse2_cvtsd2si,
                 sdmem, sse_load_f64, "cvtsd2si", SSE_CVT_SD2SI>, XD;
defm CVTSD2SI64 : sse12_cvt_sint<0x2D, VR128, GR64, int_x86_sse2_cvtsd2si64,
                   sdmem, sse_load_f64, "cvtsd2si", SSE_CVT_SD2SI>, XD, REX_W;


let Predicates = [UseAVX] in {
defm Int_VCVTSI2SS : sse12_cvt_sint_3addr<0x2A, GR32, VR128,
          int_x86_sse_cvtsi2ss, i32mem, loadi32, "cvtsi2ss{l}",
          SSE_CVT_Scalar, 0>, XS, VEX_4V;
defm Int_VCVTSI2SS64 : sse12_cvt_sint_3addr<0x2A, GR64, VR128,
          int_x86_sse_cvtsi642ss, i64mem, loadi64, "cvtsi2ss{q}",
          SSE_CVT_Scalar, 0>, XS, VEX_4V,
          VEX_W;
defm Int_VCVTSI2SD : sse12_cvt_sint_3addr<0x2A, GR32, VR128,
          int_x86_sse2_cvtsi2sd, i32mem, loadi32, "cvtsi2sd{l}",
          SSE_CVT_Scalar, 0>, XD, VEX_4V;
defm Int_VCVTSI2SD64 : sse12_cvt_sint_3addr<0x2A, GR64, VR128,
          int_x86_sse2_cvtsi642sd, i64mem, loadi64, "cvtsi2sd{q}",
          SSE_CVT_Scalar, 0>, XD,
          VEX_4V, VEX_W;
}
let Constraints = "$src1 = $dst" in {
  defm Int_CVTSI2SS : sse12_cvt_sint_3addr<0x2A, GR32, VR128,
                        int_x86_sse_cvtsi2ss, i32mem, loadi32,
                        "cvtsi2ss{l}", SSE_CVT_Scalar>, XS;
  defm Int_CVTSI2SS64 : sse12_cvt_sint_3addr<0x2A, GR64, VR128,
                        int_x86_sse_cvtsi642ss, i64mem, loadi64,
                        "cvtsi2ss{q}", SSE_CVT_Scalar>, XS, REX_W;
  defm Int_CVTSI2SD : sse12_cvt_sint_3addr<0x2A, GR32, VR128,
                        int_x86_sse2_cvtsi2sd, i32mem, loadi32,
                        "cvtsi2sd{l}", SSE_CVT_Scalar>, XD;
  defm Int_CVTSI2SD64 : sse12_cvt_sint_3addr<0x2A, GR64, VR128,
                        int_x86_sse2_cvtsi642sd, i64mem, loadi64,
                        "cvtsi2sd{q}", SSE_CVT_Scalar>, XD, REX_W;
}

/// SSE 1 Only

// Aliases for intrinsics
let Predicates = [UseAVX] in {
defm Int_VCVTTSS2SI : sse12_cvt_sint<0x2C, VR128, GR32, int_x86_sse_cvttss2si,
                                    ssmem, sse_load_f32, "cvttss2si",
                                    SSE_CVT_SS2SI_32>, XS, VEX;
defm Int_VCVTTSS2SI64 : sse12_cvt_sint<0x2C, VR128, GR64,
                                   int_x86_sse_cvttss2si64, ssmem, sse_load_f32,
                                   "cvttss2si", SSE_CVT_SS2SI_64>,
                                   XS, VEX, VEX_W;
defm Int_VCVTTSD2SI : sse12_cvt_sint<0x2C, VR128, GR32, int_x86_sse2_cvttsd2si,
                                    sdmem, sse_load_f64, "cvttsd2si",
                                    SSE_CVT_SD2SI>, XD, VEX;
defm Int_VCVTTSD2SI64 : sse12_cvt_sint<0x2C, VR128, GR64,
                                  int_x86_sse2_cvttsd2si64, sdmem, sse_load_f64,
                                  "cvttsd2si", SSE_CVT_SD2SI>,
                                  XD, VEX, VEX_W;
}
defm Int_CVTTSS2SI : sse12_cvt_sint<0x2C, VR128, GR32, int_x86_sse_cvttss2si,
                                    ssmem, sse_load_f32, "cvttss2si",
                                    SSE_CVT_SS2SI_32>, XS;
defm Int_CVTTSS2SI64 : sse12_cvt_sint<0x2C, VR128, GR64,
                                   int_x86_sse_cvttss2si64, ssmem, sse_load_f32,
                                   "cvttss2si", SSE_CVT_SS2SI_64>, XS, REX_W;
defm Int_CVTTSD2SI : sse12_cvt_sint<0x2C, VR128, GR32, int_x86_sse2_cvttsd2si,
                                    sdmem, sse_load_f64, "cvttsd2si",
                                    SSE_CVT_SD2SI>, XD;
defm Int_CVTTSD2SI64 : sse12_cvt_sint<0x2C, VR128, GR64,
                                  int_x86_sse2_cvttsd2si64, sdmem, sse_load_f64,
                                  "cvttsd2si", SSE_CVT_SD2SI>, XD, REX_W;

let Predicates = [UseAVX] in {
defm VCVTSS2SI   : sse12_cvt_sint<0x2D, VR128, GR32, int_x86_sse_cvtss2si,
                                  ssmem, sse_load_f32, "cvtss2si",
                                  SSE_CVT_SS2SI_32>, XS, VEX, VEX_LIG;
defm VCVTSS2SI64 : sse12_cvt_sint<0x2D, VR128, GR64, int_x86_sse_cvtss2si64,
                                  ssmem, sse_load_f32, "cvtss2si",
                                  SSE_CVT_SS2SI_64>, XS, VEX, VEX_W, VEX_LIG;
}
defm CVTSS2SI : sse12_cvt_sint<0x2D, VR128, GR32, int_x86_sse_cvtss2si,
                               ssmem, sse_load_f32, "cvtss2si",
                               SSE_CVT_SS2SI_32>, XS;
defm CVTSS2SI64 : sse12_cvt_sint<0x2D, VR128, GR64, int_x86_sse_cvtss2si64,
                                 ssmem, sse_load_f32, "cvtss2si",
                                 SSE_CVT_SS2SI_64>, XS, REX_W;

defm VCVTDQ2PS   : sse12_cvt_p<0x5B, VR128, VR128, i128mem,
                               "vcvtdq2ps\t{$src, $dst|$dst, $src}",
                               SSEPackedSingle, SSE_CVT_PS>,
                               TB, VEX, Requires<[HasAVX]>;
defm VCVTDQ2PSY  : sse12_cvt_p<0x5B, VR256, VR256, i256mem,
                               "vcvtdq2ps\t{$src, $dst|$dst, $src}",
                               SSEPackedSingle, SSE_CVT_PS>,
                               TB, VEX, VEX_L, Requires<[HasAVX]>;

defm CVTDQ2PS : sse12_cvt_p<0x5B, VR128, VR128, i128mem,
                            "cvtdq2ps\t{$src, $dst|$dst, $src}",
                            SSEPackedSingle, SSE_CVT_PS>,
                            TB, Requires<[UseSSE2]>;

let Predicates = [UseAVX] in {
def : InstAlias<"vcvtss2si{l}\t{$src, $dst|$dst, $src}",
                (VCVTSS2SIrr GR32:$dst, VR128:$src), 0>;
def : InstAlias<"vcvtss2si{l}\t{$src, $dst|$dst, $src}",
                (VCVTSS2SIrm GR32:$dst, ssmem:$src), 0>;
def : InstAlias<"vcvtsd2si{l}\t{$src, $dst|$dst, $src}",
                (VCVTSD2SIrr GR32:$dst, VR128:$src), 0>;
def : InstAlias<"vcvtsd2si{l}\t{$src, $dst|$dst, $src}",
                (VCVTSD2SIrm GR32:$dst, sdmem:$src), 0>;
def : InstAlias<"vcvtss2si{q}\t{$src, $dst|$dst, $src}",
                (VCVTSS2SI64rr GR64:$dst, VR128:$src), 0>;
def : InstAlias<"vcvtss2si{q}\t{$src, $dst|$dst, $src}",
                (VCVTSS2SI64rm GR64:$dst, ssmem:$src), 0>;
def : InstAlias<"vcvtsd2si{q}\t{$src, $dst|$dst, $src}",
                (VCVTSD2SI64rr GR64:$dst, VR128:$src), 0>;
def : InstAlias<"vcvtsd2si{q}\t{$src, $dst|$dst, $src}",
                (VCVTSD2SI64rm GR64:$dst, sdmem:$src), 0>;
}

def : InstAlias<"cvtss2si{l}\t{$src, $dst|$dst, $src}",
                (CVTSS2SIrr GR32:$dst, VR128:$src), 0>;
def : InstAlias<"cvtss2si{l}\t{$src, $dst|$dst, $src}",
                (CVTSS2SIrm GR32:$dst, ssmem:$src), 0>;
def : InstAlias<"cvtsd2si{l}\t{$src, $dst|$dst, $src}",
                (CVTSD2SIrr GR32:$dst, VR128:$src), 0>;
def : InstAlias<"cvtsd2si{l}\t{$src, $dst|$dst, $src}",
                (CVTSD2SIrm GR32:$dst, sdmem:$src), 0>;
def : InstAlias<"cvtss2si{q}\t{$src, $dst|$dst, $src}",
                (CVTSS2SI64rr GR64:$dst, VR128:$src), 0>;
def : InstAlias<"cvtss2si{q}\t{$src, $dst|$dst, $src}",
                (CVTSS2SI64rm GR64:$dst, ssmem:$src), 0>;
def : InstAlias<"cvtsd2si{q}\t{$src, $dst|$dst, $src}",
                (CVTSD2SI64rr GR64:$dst, VR128:$src), 0>;
def : InstAlias<"cvtsd2si{q}\t{$src, $dst|$dst, $src}",
                (CVTSD2SI64rm GR64:$dst, sdmem:$src)>;

/// SSE 2 Only

// Convert scalar double to scalar single
let neverHasSideEffects = 1, Predicates = [UseAVX] in {
def VCVTSD2SSrr  : VSDI<0x5A, MRMSrcReg, (outs FR32:$dst),
                       (ins FR64:$src1, FR64:$src2),
                      "cvtsd2ss\t{$src2, $src1, $dst|$dst, $src1, $src2}", [],
                      IIC_SSE_CVT_Scalar_RR>, VEX_4V, VEX_LIG,
                      Sched<[WriteCvtF2F]>;
let mayLoad = 1 in
def VCVTSD2SSrm  : I<0x5A, MRMSrcMem, (outs FR32:$dst),
                       (ins FR64:$src1, f64mem:$src2),
                      "vcvtsd2ss\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                      [], IIC_SSE_CVT_Scalar_RM>,
                      XD, Requires<[HasAVX, OptForSize]>, VEX_4V, VEX_LIG,
                      Sched<[WriteCvtF2FLd, ReadAfterLd]>;
}

def : Pat<(f32 (fround FR64:$src)), (VCVTSD2SSrr FR64:$src, FR64:$src)>,
          Requires<[UseAVX]>;

def CVTSD2SSrr  : SDI<0x5A, MRMSrcReg, (outs FR32:$dst), (ins FR64:$src),
                      "cvtsd2ss\t{$src, $dst|$dst, $src}",
                      [(set FR32:$dst, (fround FR64:$src))],
                      IIC_SSE_CVT_Scalar_RR>, Sched<[WriteCvtF2F]>;
def CVTSD2SSrm  : I<0x5A, MRMSrcMem, (outs FR32:$dst), (ins f64mem:$src),
                      "cvtsd2ss\t{$src, $dst|$dst, $src}",
                      [(set FR32:$dst, (fround (loadf64 addr:$src)))],
                      IIC_SSE_CVT_Scalar_RM>,
                      XD,
                  Requires<[UseSSE2, OptForSize]>, Sched<[WriteCvtF2FLd]>;

def Int_VCVTSD2SSrr: I<0x5A, MRMSrcReg,
                       (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                       "vcvtsd2ss\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                       [(set VR128:$dst,
                         (int_x86_sse2_cvtsd2ss VR128:$src1, VR128:$src2))],
                       IIC_SSE_CVT_Scalar_RR>, XD, VEX_4V, Requires<[UseAVX]>,
                       Sched<[WriteCvtF2F]>;
def Int_VCVTSD2SSrm: I<0x5A, MRMSrcReg,
                       (outs VR128:$dst), (ins VR128:$src1, sdmem:$src2),
                       "vcvtsd2ss\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                       [(set VR128:$dst, (int_x86_sse2_cvtsd2ss
                                          VR128:$src1, sse_load_f64:$src2))],
                       IIC_SSE_CVT_Scalar_RM>, XD, VEX_4V, Requires<[UseAVX]>,
                       Sched<[WriteCvtF2FLd, ReadAfterLd]>;

let Constraints = "$src1 = $dst" in {
def Int_CVTSD2SSrr: I<0x5A, MRMSrcReg,
                       (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                       "cvtsd2ss\t{$src2, $dst|$dst, $src2}",
                       [(set VR128:$dst,
                         (int_x86_sse2_cvtsd2ss VR128:$src1, VR128:$src2))],
                       IIC_SSE_CVT_Scalar_RR>, XD, Requires<[UseSSE2]>,
                       Sched<[WriteCvtF2F]>;
def Int_CVTSD2SSrm: I<0x5A, MRMSrcReg,
                       (outs VR128:$dst), (ins VR128:$src1, sdmem:$src2),
                       "cvtsd2ss\t{$src2, $dst|$dst, $src2}",
                       [(set VR128:$dst, (int_x86_sse2_cvtsd2ss
                                          VR128:$src1, sse_load_f64:$src2))],
                       IIC_SSE_CVT_Scalar_RM>, XD, Requires<[UseSSE2]>,
                       Sched<[WriteCvtF2FLd, ReadAfterLd]>;
}

// Convert scalar single to scalar double
// SSE2 instructions with XS prefix
let neverHasSideEffects = 1, Predicates = [UseAVX] in {
def VCVTSS2SDrr : I<0x5A, MRMSrcReg, (outs FR64:$dst),
                    (ins FR32:$src1, FR32:$src2),
                    "vcvtss2sd\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                    [], IIC_SSE_CVT_Scalar_RR>,
                    XS, Requires<[HasAVX]>, VEX_4V, VEX_LIG,
                    Sched<[WriteCvtF2F]>;
let mayLoad = 1 in
def VCVTSS2SDrm : I<0x5A, MRMSrcMem, (outs FR64:$dst),
                    (ins FR32:$src1, f32mem:$src2),
                    "vcvtss2sd\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                    [], IIC_SSE_CVT_Scalar_RM>,
                    XS, VEX_4V, VEX_LIG, Requires<[HasAVX, OptForSize]>,
                    Sched<[WriteCvtF2FLd, ReadAfterLd]>;
}

def : Pat<(f64 (fextend FR32:$src)),
    (VCVTSS2SDrr FR32:$src, FR32:$src)>, Requires<[UseAVX]>;
def : Pat<(fextend (loadf32 addr:$src)),
    (VCVTSS2SDrm (f32 (IMPLICIT_DEF)), addr:$src)>, Requires<[UseAVX]>;

def : Pat<(extloadf32 addr:$src),
    (VCVTSS2SDrm (f32 (IMPLICIT_DEF)), addr:$src)>,
    Requires<[UseAVX, OptForSize]>;
def : Pat<(extloadf32 addr:$src),
    (VCVTSS2SDrr (f32 (IMPLICIT_DEF)), (VMOVSSrm addr:$src))>,
    Requires<[UseAVX, OptForSpeed]>;

def CVTSS2SDrr : I<0x5A, MRMSrcReg, (outs FR64:$dst), (ins FR32:$src),
                   "cvtss2sd\t{$src, $dst|$dst, $src}",
                   [(set FR64:$dst, (fextend FR32:$src))],
                   IIC_SSE_CVT_Scalar_RR>, XS,
                 Requires<[UseSSE2]>, Sched<[WriteCvtF2F]>;
def CVTSS2SDrm : I<0x5A, MRMSrcMem, (outs FR64:$dst), (ins f32mem:$src),
                   "cvtss2sd\t{$src, $dst|$dst, $src}",
                   [(set FR64:$dst, (extloadf32 addr:$src))],
                   IIC_SSE_CVT_Scalar_RM>, XS,
                 Requires<[UseSSE2, OptForSize]>, Sched<[WriteCvtF2FLd]>;

// extload f32 -> f64.  This matches load+fextend because we have a hack in
// the isel (PreprocessForFPConvert) that can introduce loads after dag
// combine.
// Since these loads aren't folded into the fextend, we have to match it
// explicitly here.
def : Pat<(fextend (loadf32 addr:$src)),
          (CVTSS2SDrm addr:$src)>, Requires<[UseSSE2]>;
def : Pat<(extloadf32 addr:$src),
          (CVTSS2SDrr (MOVSSrm addr:$src))>, Requires<[UseSSE2, OptForSpeed]>;

def Int_VCVTSS2SDrr: I<0x5A, MRMSrcReg,
                      (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                    "vcvtss2sd\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                    [(set VR128:$dst,
                      (int_x86_sse2_cvtss2sd VR128:$src1, VR128:$src2))],
                    IIC_SSE_CVT_Scalar_RR>, XS, VEX_4V, Requires<[UseAVX]>,
                    Sched<[WriteCvtF2F]>;
def Int_VCVTSS2SDrm: I<0x5A, MRMSrcMem,
                      (outs VR128:$dst), (ins VR128:$src1, ssmem:$src2),
                    "vcvtss2sd\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                    [(set VR128:$dst,
                      (int_x86_sse2_cvtss2sd VR128:$src1, sse_load_f32:$src2))],
                    IIC_SSE_CVT_Scalar_RM>, XS, VEX_4V, Requires<[UseAVX]>,
                    Sched<[WriteCvtF2FLd, ReadAfterLd]>;
let Constraints = "$src1 = $dst" in { // SSE2 instructions with XS prefix
def Int_CVTSS2SDrr: I<0x5A, MRMSrcReg,
                      (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
                    "cvtss2sd\t{$src2, $dst|$dst, $src2}",
                    [(set VR128:$dst,
                      (int_x86_sse2_cvtss2sd VR128:$src1, VR128:$src2))],
                    IIC_SSE_CVT_Scalar_RR>, XS, Requires<[UseSSE2]>,
                    Sched<[WriteCvtF2F]>;
def Int_CVTSS2SDrm: I<0x5A, MRMSrcMem,
                      (outs VR128:$dst), (ins VR128:$src1, ssmem:$src2),
                    "cvtss2sd\t{$src2, $dst|$dst, $src2}",
                    [(set VR128:$dst,
                      (int_x86_sse2_cvtss2sd VR128:$src1, sse_load_f32:$src2))],
                    IIC_SSE_CVT_Scalar_RM>, XS, Requires<[UseSSE2]>,
                    Sched<[WriteCvtF2FLd, ReadAfterLd]>;
}

// Convert packed single/double fp to doubleword
def VCVTPS2DQrr : VPDI<0x5B, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                       "cvtps2dq\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_cvtps2dq VR128:$src))],
                       IIC_SSE_CVT_PS_RR>, VEX, Sched<[WriteCvtF2I]>;
def VCVTPS2DQrm : VPDI<0x5B, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                       "cvtps2dq\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst,
                         (int_x86_sse2_cvtps2dq (loadv4f32 addr:$src)))],
                       IIC_SSE_CVT_PS_RM>, VEX, Sched<[WriteCvtF2ILd]>;
def VCVTPS2DQYrr : VPDI<0x5B, MRMSrcReg, (outs VR256:$dst), (ins VR256:$src),
                        "cvtps2dq\t{$src, $dst|$dst, $src}",
                        [(set VR256:$dst,
                          (int_x86_avx_cvt_ps2dq_256 VR256:$src))],
                        IIC_SSE_CVT_PS_RR>, VEX, VEX_L, Sched<[WriteCvtF2I]>;
def VCVTPS2DQYrm : VPDI<0x5B, MRMSrcMem, (outs VR256:$dst), (ins f256mem:$src),
                        "cvtps2dq\t{$src, $dst|$dst, $src}",
                        [(set VR256:$dst,
                          (int_x86_avx_cvt_ps2dq_256 (loadv8f32 addr:$src)))],
                        IIC_SSE_CVT_PS_RM>, VEX, VEX_L, Sched<[WriteCvtF2ILd]>;
def CVTPS2DQrr : PDI<0x5B, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                     "cvtps2dq\t{$src, $dst|$dst, $src}",
                     [(set VR128:$dst, (int_x86_sse2_cvtps2dq VR128:$src))],
                     IIC_SSE_CVT_PS_RR>, Sched<[WriteCvtF2I]>;
def CVTPS2DQrm : PDI<0x5B, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                     "cvtps2dq\t{$src, $dst|$dst, $src}",
                     [(set VR128:$dst,
                       (int_x86_sse2_cvtps2dq (memopv4f32 addr:$src)))],
                     IIC_SSE_CVT_PS_RM>, Sched<[WriteCvtF2ILd]>;


// Convert Packed Double FP to Packed DW Integers
let Predicates = [HasAVX] in {
// The assembler can recognize rr 256-bit instructions by seeing a ymm
// register, but the same isn't true when using memory operands instead.
// Provide other assembly rr and rm forms to address this explicitly.
def VCVTPD2DQrr  : SDI<0xE6, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                       "vcvtpd2dq\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_cvtpd2dq VR128:$src))]>,
                       VEX, Sched<[WriteCvtF2I]>;

// XMM only
def : InstAlias<"vcvtpd2dqx\t{$src, $dst|$dst, $src}",
                (VCVTPD2DQrr VR128:$dst, VR128:$src)>;
def VCVTPD2DQXrm : SDI<0xE6, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                       "vcvtpd2dqx\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst,
                         (int_x86_sse2_cvtpd2dq (loadv2f64 addr:$src)))]>, VEX,
                       Sched<[WriteCvtF2ILd]>;

// YMM only
def VCVTPD2DQYrr : SDI<0xE6, MRMSrcReg, (outs VR128:$dst), (ins VR256:$src),
                       "vcvtpd2dq{y}\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst,
                         (int_x86_avx_cvt_pd2dq_256 VR256:$src))]>, VEX, VEX_L,
                       Sched<[WriteCvtF2I]>;
def VCVTPD2DQYrm : SDI<0xE6, MRMSrcMem, (outs VR128:$dst), (ins f256mem:$src),
                       "vcvtpd2dq{y}\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst,
                         (int_x86_avx_cvt_pd2dq_256 (loadv4f64 addr:$src)))]>,
                       VEX, VEX_L, Sched<[WriteCvtF2ILd]>;
def : InstAlias<"vcvtpd2dq\t{$src, $dst|$dst, $src}",
                (VCVTPD2DQYrr VR128:$dst, VR256:$src)>;
}

def CVTPD2DQrm  : SDI<0xE6, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                      "cvtpd2dq\t{$src, $dst|$dst, $src}",
                      [(set VR128:$dst,
                        (int_x86_sse2_cvtpd2dq (memopv2f64 addr:$src)))],
                      IIC_SSE_CVT_PD_RM>, Sched<[WriteCvtF2ILd]>;
def CVTPD2DQrr  : SDI<0xE6, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                      "cvtpd2dq\t{$src, $dst|$dst, $src}",
                      [(set VR128:$dst, (int_x86_sse2_cvtpd2dq VR128:$src))],
                      IIC_SSE_CVT_PD_RR>, Sched<[WriteCvtF2I]>;

// Convert with truncation packed single/double fp to doubleword
// SSE2 packed instructions with XS prefix
def VCVTTPS2DQrr : VS2SI<0x5B, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                         "cvttps2dq\t{$src, $dst|$dst, $src}",
                         [(set VR128:$dst,
                           (int_x86_sse2_cvttps2dq VR128:$src))],
                         IIC_SSE_CVT_PS_RR>, VEX, Sched<[WriteCvtF2I]>;
def VCVTTPS2DQrm : VS2SI<0x5B, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                         "cvttps2dq\t{$src, $dst|$dst, $src}",
                         [(set VR128:$dst, (int_x86_sse2_cvttps2dq
                                            (loadv4f32 addr:$src)))],
                         IIC_SSE_CVT_PS_RM>, VEX, Sched<[WriteCvtF2ILd]>;
def VCVTTPS2DQYrr : VS2SI<0x5B, MRMSrcReg, (outs VR256:$dst), (ins VR256:$src),
                          "cvttps2dq\t{$src, $dst|$dst, $src}",
                          [(set VR256:$dst,
                            (int_x86_avx_cvtt_ps2dq_256 VR256:$src))],
                          IIC_SSE_CVT_PS_RR>, VEX, VEX_L, Sched<[WriteCvtF2I]>;
def VCVTTPS2DQYrm : VS2SI<0x5B, MRMSrcMem, (outs VR256:$dst), (ins f256mem:$src),
                          "cvttps2dq\t{$src, $dst|$dst, $src}",
                          [(set VR256:$dst, (int_x86_avx_cvtt_ps2dq_256
                                             (loadv8f32 addr:$src)))],
                          IIC_SSE_CVT_PS_RM>, VEX, VEX_L,
                          Sched<[WriteCvtF2ILd]>;

def CVTTPS2DQrr : S2SI<0x5B, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                       "cvttps2dq\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_cvttps2dq VR128:$src))],
                       IIC_SSE_CVT_PS_RR>, Sched<[WriteCvtF2I]>;
def CVTTPS2DQrm : S2SI<0x5B, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                       "cvttps2dq\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst,
                         (int_x86_sse2_cvttps2dq (memopv4f32 addr:$src)))],
                       IIC_SSE_CVT_PS_RM>, Sched<[WriteCvtF2ILd]>;

let Predicates = [HasAVX] in {
  def : Pat<(v4f32 (sint_to_fp (v4i32 VR128:$src))),
            (VCVTDQ2PSrr VR128:$src)>;
  def : Pat<(v4f32 (sint_to_fp (bc_v4i32 (loadv2i64 addr:$src)))),
            (VCVTDQ2PSrm addr:$src)>;

  def : Pat<(int_x86_sse2_cvtdq2ps VR128:$src),
            (VCVTDQ2PSrr VR128:$src)>;
  def : Pat<(int_x86_sse2_cvtdq2ps (bc_v4i32 (loadv2i64 addr:$src))),
            (VCVTDQ2PSrm addr:$src)>;

  def : Pat<(v4i32 (fp_to_sint (v4f32 VR128:$src))),
            (VCVTTPS2DQrr VR128:$src)>;
  def : Pat<(v4i32 (fp_to_sint (loadv4f32 addr:$src))),
            (VCVTTPS2DQrm addr:$src)>;

  def : Pat<(v8f32 (sint_to_fp (v8i32 VR256:$src))),
            (VCVTDQ2PSYrr VR256:$src)>;
  def : Pat<(v8f32 (sint_to_fp (bc_v8i32 (loadv4i64 addr:$src)))),
            (VCVTDQ2PSYrm addr:$src)>;

  def : Pat<(v8i32 (fp_to_sint (v8f32 VR256:$src))),
            (VCVTTPS2DQYrr VR256:$src)>;
  def : Pat<(v8i32 (fp_to_sint (loadv8f32 addr:$src))),
            (VCVTTPS2DQYrm addr:$src)>;
}

let Predicates = [UseSSE2] in {
  def : Pat<(v4f32 (sint_to_fp (v4i32 VR128:$src))),
            (CVTDQ2PSrr VR128:$src)>;
  def : Pat<(v4f32 (sint_to_fp (bc_v4i32 (memopv2i64 addr:$src)))),
            (CVTDQ2PSrm addr:$src)>;

  def : Pat<(int_x86_sse2_cvtdq2ps VR128:$src),
            (CVTDQ2PSrr VR128:$src)>;
  def : Pat<(int_x86_sse2_cvtdq2ps (bc_v4i32 (memopv2i64 addr:$src))),
            (CVTDQ2PSrm addr:$src)>;

  def : Pat<(v4i32 (fp_to_sint (v4f32 VR128:$src))),
            (CVTTPS2DQrr VR128:$src)>;
  def : Pat<(v4i32 (fp_to_sint (memopv4f32 addr:$src))),
            (CVTTPS2DQrm addr:$src)>;
}

def VCVTTPD2DQrr : VPDI<0xE6, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                        "cvttpd2dq\t{$src, $dst|$dst, $src}",
                        [(set VR128:$dst,
                              (int_x86_sse2_cvttpd2dq VR128:$src))],
                              IIC_SSE_CVT_PD_RR>, VEX, Sched<[WriteCvtF2I]>;

// The assembler can recognize rr 256-bit instructions by seeing a ymm
// register, but the same isn't true when using memory operands instead.
// Provide other assembly rr and rm forms to address this explicitly.

// XMM only
def : InstAlias<"vcvttpd2dqx\t{$src, $dst|$dst, $src}",
                (VCVTTPD2DQrr VR128:$dst, VR128:$src)>;
def VCVTTPD2DQXrm : VPDI<0xE6, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                         "cvttpd2dqx\t{$src, $dst|$dst, $src}",
                         [(set VR128:$dst, (int_x86_sse2_cvttpd2dq
                                            (loadv2f64 addr:$src)))],
                         IIC_SSE_CVT_PD_RM>, VEX, Sched<[WriteCvtF2ILd]>;

// YMM only
def VCVTTPD2DQYrr : VPDI<0xE6, MRMSrcReg, (outs VR128:$dst), (ins VR256:$src),
                         "cvttpd2dq{y}\t{$src, $dst|$dst, $src}",
                         [(set VR128:$dst,
                           (int_x86_avx_cvtt_pd2dq_256 VR256:$src))],
                         IIC_SSE_CVT_PD_RR>, VEX, VEX_L, Sched<[WriteCvtF2I]>;
def VCVTTPD2DQYrm : VPDI<0xE6, MRMSrcMem, (outs VR128:$dst), (ins f256mem:$src),
                         "cvttpd2dq{y}\t{$src, $dst|$dst, $src}",
                         [(set VR128:$dst,
                          (int_x86_avx_cvtt_pd2dq_256 (loadv4f64 addr:$src)))],
                         IIC_SSE_CVT_PD_RM>, VEX, VEX_L, Sched<[WriteCvtF2ILd]>;
def : InstAlias<"vcvttpd2dq\t{$src, $dst|$dst, $src}",
                (VCVTTPD2DQYrr VR128:$dst, VR256:$src)>;

let Predicates = [HasAVX] in {
  def : Pat<(v4i32 (fp_to_sint (v4f64 VR256:$src))),
            (VCVTTPD2DQYrr VR256:$src)>;
  def : Pat<(v4i32 (fp_to_sint (loadv4f64 addr:$src))),
            (VCVTTPD2DQYrm addr:$src)>;
} // Predicates = [HasAVX]

def CVTTPD2DQrr : PDI<0xE6, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                      "cvttpd2dq\t{$src, $dst|$dst, $src}",
                      [(set VR128:$dst, (int_x86_sse2_cvttpd2dq VR128:$src))],
                      IIC_SSE_CVT_PD_RR>, Sched<[WriteCvtF2I]>;
def CVTTPD2DQrm : PDI<0xE6, MRMSrcMem, (outs VR128:$dst),(ins f128mem:$src),
                      "cvttpd2dq\t{$src, $dst|$dst, $src}",
                      [(set VR128:$dst, (int_x86_sse2_cvttpd2dq
                                        (memopv2f64 addr:$src)))],
                                        IIC_SSE_CVT_PD_RM>,
                      Sched<[WriteCvtF2ILd]>;

// Convert packed single to packed double
let Predicates = [HasAVX] in {
                  // SSE2 instructions without OpSize prefix
def VCVTPS2PDrr : I<0x5A, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                     "vcvtps2pd\t{$src, $dst|$dst, $src}",
                     [(set VR128:$dst, (int_x86_sse2_cvtps2pd VR128:$src))],
                     IIC_SSE_CVT_PD_RR>, TB, VEX, Sched<[WriteCvtF2F]>;
def VCVTPS2PDrm : I<0x5A, MRMSrcMem, (outs VR128:$dst), (ins f64mem:$src),
                    "vcvtps2pd\t{$src, $dst|$dst, $src}",
                    [(set VR128:$dst, (v2f64 (extloadv2f32 addr:$src)))],
                    IIC_SSE_CVT_PD_RM>, TB, VEX, Sched<[WriteCvtF2FLd]>;
def VCVTPS2PDYrr : I<0x5A, MRMSrcReg, (outs VR256:$dst), (ins VR128:$src),
                     "vcvtps2pd\t{$src, $dst|$dst, $src}",
                     [(set VR256:$dst,
                       (int_x86_avx_cvt_ps2_pd_256 VR128:$src))],
                     IIC_SSE_CVT_PD_RR>, TB, VEX, VEX_L, Sched<[WriteCvtF2F]>;
def VCVTPS2PDYrm : I<0x5A, MRMSrcMem, (outs VR256:$dst), (ins f128mem:$src),
                     "vcvtps2pd\t{$src, $dst|$dst, $src}",
                     [(set VR256:$dst,
                       (int_x86_avx_cvt_ps2_pd_256 (loadv4f32 addr:$src)))],
                     IIC_SSE_CVT_PD_RM>, TB, VEX, VEX_L, Sched<[WriteCvtF2FLd]>;
}

let Predicates = [UseSSE2] in {
def CVTPS2PDrr : I<0x5A, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                       "cvtps2pd\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_cvtps2pd VR128:$src))],
                       IIC_SSE_CVT_PD_RR>, TB, Sched<[WriteCvtF2F]>;
def CVTPS2PDrm : I<0x5A, MRMSrcMem, (outs VR128:$dst), (ins f64mem:$src),
                   "cvtps2pd\t{$src, $dst|$dst, $src}",
                   [(set VR128:$dst, (v2f64 (extloadv2f32 addr:$src)))],
                   IIC_SSE_CVT_PD_RM>, TB, Sched<[WriteCvtF2FLd]>;
}

// Convert Packed DW Integers to Packed Double FP
let Predicates = [HasAVX] in {
let neverHasSideEffects = 1, mayLoad = 1 in
def VCVTDQ2PDrm  : S2SI<0xE6, MRMSrcMem, (outs VR128:$dst), (ins i64mem:$src),
                     "vcvtdq2pd\t{$src, $dst|$dst, $src}",
                     []>, VEX, Sched<[WriteCvtI2FLd]>;
def VCVTDQ2PDrr  : S2SI<0xE6, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                     "vcvtdq2pd\t{$src, $dst|$dst, $src}",
                     [(set VR128:$dst,
                       (int_x86_sse2_cvtdq2pd VR128:$src))]>, VEX,
                   Sched<[WriteCvtI2F]>;
def VCVTDQ2PDYrm  : S2SI<0xE6, MRMSrcMem, (outs VR256:$dst), (ins i128mem:$src),
                     "vcvtdq2pd\t{$src, $dst|$dst, $src}",
                     [(set VR256:$dst,
                       (int_x86_avx_cvtdq2_pd_256
                        (bitconvert (loadv2i64 addr:$src))))]>, VEX, VEX_L,
                    Sched<[WriteCvtI2FLd]>;
def VCVTDQ2PDYrr  : S2SI<0xE6, MRMSrcReg, (outs VR256:$dst), (ins VR128:$src),
                     "vcvtdq2pd\t{$src, $dst|$dst, $src}",
                     [(set VR256:$dst,
                       (int_x86_avx_cvtdq2_pd_256 VR128:$src))]>, VEX, VEX_L,
                    Sched<[WriteCvtI2F]>;
}

let neverHasSideEffects = 1, mayLoad = 1 in
def CVTDQ2PDrm  : S2SI<0xE6, MRMSrcMem, (outs VR128:$dst), (ins i64mem:$src),
                       "cvtdq2pd\t{$src, $dst|$dst, $src}", [],
                       IIC_SSE_CVT_PD_RR>, Sched<[WriteCvtI2FLd]>;
def CVTDQ2PDrr  : S2SI<0xE6, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                       "cvtdq2pd\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_cvtdq2pd VR128:$src))],
                       IIC_SSE_CVT_PD_RM>, Sched<[WriteCvtI2F]>;

// AVX 256-bit register conversion intrinsics
let Predicates = [HasAVX] in {
  def : Pat<(v4f64 (sint_to_fp (v4i32 VR128:$src))),
            (VCVTDQ2PDYrr VR128:$src)>;
  def : Pat<(v4f64 (sint_to_fp (bc_v4i32 (loadv2i64 addr:$src)))),
            (VCVTDQ2PDYrm addr:$src)>;
} // Predicates = [HasAVX]

// Convert packed double to packed single
// The assembler can recognize rr 256-bit instructions by seeing a ymm
// register, but the same isn't true when using memory operands instead.
// Provide other assembly rr and rm forms to address this explicitly.
def VCVTPD2PSrr : VPDI<0x5A, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                       "cvtpd2ps\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse2_cvtpd2ps VR128:$src))],
                       IIC_SSE_CVT_PD_RR>, VEX, Sched<[WriteCvtF2F]>;

// XMM only
def : InstAlias<"vcvtpd2psx\t{$src, $dst|$dst, $src}",
                (VCVTPD2PSrr VR128:$dst, VR128:$src)>;
def VCVTPD2PSXrm : VPDI<0x5A, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                        "cvtpd2psx\t{$src, $dst|$dst, $src}",
                        [(set VR128:$dst,
                          (int_x86_sse2_cvtpd2ps (loadv2f64 addr:$src)))],
                        IIC_SSE_CVT_PD_RM>, VEX, Sched<[WriteCvtF2FLd]>;

// YMM only
def VCVTPD2PSYrr : VPDI<0x5A, MRMSrcReg, (outs VR128:$dst), (ins VR256:$src),
                        "cvtpd2ps{y}\t{$src, $dst|$dst, $src}",
                        [(set VR128:$dst,
                          (int_x86_avx_cvt_pd2_ps_256 VR256:$src))],
                        IIC_SSE_CVT_PD_RR>, VEX, VEX_L, Sched<[WriteCvtF2F]>;
def VCVTPD2PSYrm : VPDI<0x5A, MRMSrcMem, (outs VR128:$dst), (ins f256mem:$src),
                        "cvtpd2ps{y}\t{$src, $dst|$dst, $src}",
                        [(set VR128:$dst,
                          (int_x86_avx_cvt_pd2_ps_256 (loadv4f64 addr:$src)))],
                        IIC_SSE_CVT_PD_RM>, VEX, VEX_L, Sched<[WriteCvtF2FLd]>;
def : InstAlias<"vcvtpd2ps\t{$src, $dst|$dst, $src}",
                (VCVTPD2PSYrr VR128:$dst, VR256:$src)>;

def CVTPD2PSrr : PDI<0x5A, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                     "cvtpd2ps\t{$src, $dst|$dst, $src}",
                     [(set VR128:$dst, (int_x86_sse2_cvtpd2ps VR128:$src))],
                     IIC_SSE_CVT_PD_RR>, Sched<[WriteCvtF2F]>;
def CVTPD2PSrm : PDI<0x5A, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                     "cvtpd2ps\t{$src, $dst|$dst, $src}",
                     [(set VR128:$dst,
                       (int_x86_sse2_cvtpd2ps (memopv2f64 addr:$src)))],
                     IIC_SSE_CVT_PD_RM>, Sched<[WriteCvtF2FLd]>;


// AVX 256-bit register conversion intrinsics
// FIXME: Migrate SSE conversion intrinsics matching to use patterns as below
// whenever possible to avoid declaring two versions of each one.
let Predicates = [HasAVX] in {
  def : Pat<(int_x86_avx_cvtdq2_ps_256 VR256:$src),
            (VCVTDQ2PSYrr VR256:$src)>;
  def : Pat<(int_x86_avx_cvtdq2_ps_256 (bitconvert (loadv4i64 addr:$src))),
            (VCVTDQ2PSYrm addr:$src)>;

  // Match fround and fextend for 128/256-bit conversions
  def : Pat<(v4f32 (X86vfpround (v2f64 VR128:$src))),
            (VCVTPD2PSrr VR128:$src)>;
  def : Pat<(v4f32 (X86vfpround (loadv2f64 addr:$src))),
            (VCVTPD2PSXrm addr:$src)>;
  def : Pat<(v4f32 (fround (v4f64 VR256:$src))),
            (VCVTPD2PSYrr VR256:$src)>;
  def : Pat<(v4f32 (fround (loadv4f64 addr:$src))),
            (VCVTPD2PSYrm addr:$src)>;

  def : Pat<(v2f64 (X86vfpext (v4f32 VR128:$src))),
            (VCVTPS2PDrr VR128:$src)>;
  def : Pat<(v4f64 (fextend (v4f32 VR128:$src))),
            (VCVTPS2PDYrr VR128:$src)>;
  def : Pat<(v4f64 (extloadv4f32 addr:$src)),
            (VCVTPS2PDYrm addr:$src)>;
}

let Predicates = [UseSSE2] in {
  // Match fround and fextend for 128 conversions
  def : Pat<(v4f32 (X86vfpround (v2f64 VR128:$src))),
            (CVTPD2PSrr VR128:$src)>;
  def : Pat<(v4f32 (X86vfpround (memopv2f64 addr:$src))),
            (CVTPD2PSrm addr:$src)>;

  def : Pat<(v2f64 (X86vfpext (v4f32 VR128:$src))),
            (CVTPS2PDrr VR128:$src)>;
}

//===----------------------------------------------------------------------===//
// SSE 1 & 2 - Compare Instructions
//===----------------------------------------------------------------------===//

// sse12_cmp_scalar - sse 1 & 2 compare scalar instructions
multiclass sse12_cmp_scalar<RegisterClass RC, X86MemOperand x86memop,
                            Operand CC, SDNode OpNode, ValueType VT,
                            PatFrag ld_frag, string asm, string asm_alt,
                            OpndItins itins> {
  def rr : SIi8<0xC2, MRMSrcReg,
                (outs RC:$dst), (ins RC:$src1, RC:$src2, CC:$cc), asm,
                [(set RC:$dst, (OpNode (VT RC:$src1), RC:$src2, imm:$cc))],
                itins.rr>, Sched<[itins.Sched]>;
  def rm : SIi8<0xC2, MRMSrcMem,
                (outs RC:$dst), (ins RC:$src1, x86memop:$src2, CC:$cc), asm,
                [(set RC:$dst, (OpNode (VT RC:$src1),
                                         (ld_frag addr:$src2), imm:$cc))],
                                         itins.rm>,
           Sched<[itins.Sched.Folded, ReadAfterLd]>;

  // Accept explicit immediate argument form instead of comparison code.
  let neverHasSideEffects = 1 in {
    def rr_alt : SIi8<0xC2, MRMSrcReg, (outs RC:$dst),
                      (ins RC:$src1, RC:$src2, i8imm:$cc), asm_alt, [],
                      IIC_SSE_ALU_F32S_RR>, Sched<[itins.Sched]>;
    let mayLoad = 1 in
    def rm_alt : SIi8<0xC2, MRMSrcMem, (outs RC:$dst),
                      (ins RC:$src1, x86memop:$src2, i8imm:$cc), asm_alt, [],
                      IIC_SSE_ALU_F32S_RM>,
                      Sched<[itins.Sched.Folded, ReadAfterLd]>;
  }
}

defm VCMPSS : sse12_cmp_scalar<FR32, f32mem, AVXCC, X86cmpss, f32, loadf32,
                 "cmp${cc}ss\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                 "cmpss\t{$cc, $src2, $src1, $dst|$dst, $src1, $src2, $cc}",
                 SSE_ALU_F32S>,
                 XS, VEX_4V, VEX_LIG;
defm VCMPSD : sse12_cmp_scalar<FR64, f64mem, AVXCC, X86cmpsd, f64, loadf64,
                 "cmp${cc}sd\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                 "cmpsd\t{$cc, $src2, $src1, $dst|$dst, $src1, $src2, $cc}",
                 SSE_ALU_F32S>, // same latency as 32 bit compare
                 XD, VEX_4V, VEX_LIG;

let Constraints = "$src1 = $dst" in {
  defm CMPSS : sse12_cmp_scalar<FR32, f32mem, SSECC, X86cmpss, f32, loadf32,
                  "cmp${cc}ss\t{$src2, $dst|$dst, $src2}",
                  "cmpss\t{$cc, $src2, $dst|$dst, $src2, $cc}", SSE_ALU_F32S>,
                  XS;
  defm CMPSD : sse12_cmp_scalar<FR64, f64mem, SSECC, X86cmpsd, f64, loadf64,
                  "cmp${cc}sd\t{$src2, $dst|$dst, $src2}",
                  "cmpsd\t{$cc, $src2, $dst|$dst, $src2, $cc}",
                  SSE_ALU_F64S>,
                  XD;
}

multiclass sse12_cmp_scalar_int<X86MemOperand x86memop, Operand CC,
                         Intrinsic Int, string asm, OpndItins itins> {
  def rr : SIi8<0xC2, MRMSrcReg, (outs VR128:$dst),
                      (ins VR128:$src1, VR128:$src, CC:$cc), asm,
                        [(set VR128:$dst, (Int VR128:$src1,
                                               VR128:$src, imm:$cc))],
                                               itins.rr>,
           Sched<[itins.Sched]>;
  def rm : SIi8<0xC2, MRMSrcMem, (outs VR128:$dst),
                      (ins VR128:$src1, x86memop:$src, CC:$cc), asm,
                        [(set VR128:$dst, (Int VR128:$src1,
                                               (load addr:$src), imm:$cc))],
                                               itins.rm>,
           Sched<[itins.Sched.Folded, ReadAfterLd]>;
}

// Aliases to match intrinsics which expect XMM operand(s).
defm Int_VCMPSS  : sse12_cmp_scalar_int<f32mem, AVXCC, int_x86_sse_cmp_ss,
                     "cmp${cc}ss\t{$src, $src1, $dst|$dst, $src1, $src}",
                     SSE_ALU_F32S>,
                     XS, VEX_4V;
defm Int_VCMPSD  : sse12_cmp_scalar_int<f64mem, AVXCC, int_x86_sse2_cmp_sd,
                     "cmp${cc}sd\t{$src, $src1, $dst|$dst, $src1, $src}",
                     SSE_ALU_F32S>, // same latency as f32
                     XD, VEX_4V;
let Constraints = "$src1 = $dst" in {
  defm Int_CMPSS  : sse12_cmp_scalar_int<f32mem, SSECC, int_x86_sse_cmp_ss,
                       "cmp${cc}ss\t{$src, $dst|$dst, $src}",
                       SSE_ALU_F32S>, XS;
  defm Int_CMPSD  : sse12_cmp_scalar_int<f64mem, SSECC, int_x86_sse2_cmp_sd,
                       "cmp${cc}sd\t{$src, $dst|$dst, $src}",
                       SSE_ALU_F64S>,
                       XD;
}


// sse12_ord_cmp - Unordered/Ordered scalar fp compare and set EFLAGS
multiclass sse12_ord_cmp<bits<8> opc, RegisterClass RC, SDNode OpNode,
                            ValueType vt, X86MemOperand x86memop,
                            PatFrag ld_frag, string OpcodeStr> {
  def rr: SI<opc, MRMSrcReg, (outs), (ins RC:$src1, RC:$src2),
                     !strconcat(OpcodeStr, "\t{$src2, $src1|$src1, $src2}"),
                     [(set EFLAGS, (OpNode (vt RC:$src1), RC:$src2))],
                     IIC_SSE_COMIS_RR>,
          Sched<[WriteFAdd]>;
  def rm: SI<opc, MRMSrcMem, (outs), (ins RC:$src1, x86memop:$src2),
                     !strconcat(OpcodeStr, "\t{$src2, $src1|$src1, $src2}"),
                     [(set EFLAGS, (OpNode (vt RC:$src1),
                                           (ld_frag addr:$src2)))],
                                           IIC_SSE_COMIS_RM>,
          Sched<[WriteFAddLd, ReadAfterLd]>;
}

let Defs = [EFLAGS] in {
  defm VUCOMISS : sse12_ord_cmp<0x2E, FR32, X86cmp, f32, f32mem, loadf32,
                                  "ucomiss">, TB, VEX, VEX_LIG;
  defm VUCOMISD : sse12_ord_cmp<0x2E, FR64, X86cmp, f64, f64mem, loadf64,
                                  "ucomisd">, TB, OpSize, VEX, VEX_LIG;
  let Pattern = []<dag> in {
    defm VCOMISS  : sse12_ord_cmp<0x2F, VR128, undef, v4f32, f128mem, load,
                                    "comiss">, TB, VEX, VEX_LIG;
    defm VCOMISD  : sse12_ord_cmp<0x2F, VR128, undef, v2f64, f128mem, load,
                                    "comisd">, TB, OpSize, VEX, VEX_LIG;
  }

  defm Int_VUCOMISS  : sse12_ord_cmp<0x2E, VR128, X86ucomi, v4f32, f128mem,
                            load, "ucomiss">, TB, VEX;
  defm Int_VUCOMISD  : sse12_ord_cmp<0x2E, VR128, X86ucomi, v2f64, f128mem,
                            load, "ucomisd">, TB, OpSize, VEX;

  defm Int_VCOMISS  : sse12_ord_cmp<0x2F, VR128, X86comi, v4f32, f128mem,
                            load, "comiss">, TB, VEX;
  defm Int_VCOMISD  : sse12_ord_cmp<0x2F, VR128, X86comi, v2f64, f128mem,
                            load, "comisd">, TB, OpSize, VEX;
  defm UCOMISS  : sse12_ord_cmp<0x2E, FR32, X86cmp, f32, f32mem, loadf32,
                                  "ucomiss">, TB;
  defm UCOMISD  : sse12_ord_cmp<0x2E, FR64, X86cmp, f64, f64mem, loadf64,
                                  "ucomisd">, TB, OpSize;

  let Pattern = []<dag> in {
    defm COMISS  : sse12_ord_cmp<0x2F, VR128, undef, v4f32, f128mem, load,
                                    "comiss">, TB;
    defm COMISD  : sse12_ord_cmp<0x2F, VR128, undef, v2f64, f128mem, load,
                                    "comisd">, TB, OpSize;
  }

  defm Int_UCOMISS  : sse12_ord_cmp<0x2E, VR128, X86ucomi, v4f32, f128mem,
                              load, "ucomiss">, TB;
  defm Int_UCOMISD  : sse12_ord_cmp<0x2E, VR128, X86ucomi, v2f64, f128mem,
                              load, "ucomisd">, TB, OpSize;

  defm Int_COMISS  : sse12_ord_cmp<0x2F, VR128, X86comi, v4f32, f128mem, load,
                                  "comiss">, TB;
  defm Int_COMISD  : sse12_ord_cmp<0x2F, VR128, X86comi, v2f64, f128mem, load,
                                  "comisd">, TB, OpSize;
} // Defs = [EFLAGS]

// sse12_cmp_packed - sse 1 & 2 compare packed instructions
multiclass sse12_cmp_packed<RegisterClass RC, X86MemOperand x86memop,
                            Operand CC, Intrinsic Int, string asm,
                            string asm_alt, Domain d,
                            OpndItins itins = SSE_ALU_F32P> {
  def rri : PIi8<0xC2, MRMSrcReg,
             (outs RC:$dst), (ins RC:$src1, RC:$src2, CC:$cc), asm,
             [(set RC:$dst, (Int RC:$src1, RC:$src2, imm:$cc))],
             itins.rr, d>,
            Sched<[WriteFAdd]>;
  def rmi : PIi8<0xC2, MRMSrcMem,
             (outs RC:$dst), (ins RC:$src1, x86memop:$src2, CC:$cc), asm,
             [(set RC:$dst, (Int RC:$src1, (memop addr:$src2), imm:$cc))],
             itins.rm, d>,
            Sched<[WriteFAddLd, ReadAfterLd]>;

  // Accept explicit immediate argument form instead of comparison code.
  let neverHasSideEffects = 1 in {
    def rri_alt : PIi8<0xC2, MRMSrcReg,
               (outs RC:$dst), (ins RC:$src1, RC:$src2, i8imm:$cc),
               asm_alt, [], itins.rr, d>, Sched<[WriteFAdd]>;
    def rmi_alt : PIi8<0xC2, MRMSrcMem,
               (outs RC:$dst), (ins RC:$src1, x86memop:$src2, i8imm:$cc),
               asm_alt, [], itins.rm, d>,
               Sched<[WriteFAddLd, ReadAfterLd]>;
  }
}

defm VCMPPS : sse12_cmp_packed<VR128, f128mem, AVXCC, int_x86_sse_cmp_ps,
               "cmp${cc}ps\t{$src2, $src1, $dst|$dst, $src1, $src2}",
               "cmpps\t{$cc, $src2, $src1, $dst|$dst, $src1, $src2, $cc}",
               SSEPackedSingle>, TB, VEX_4V;
defm VCMPPD : sse12_cmp_packed<VR128, f128mem, AVXCC, int_x86_sse2_cmp_pd,
               "cmp${cc}pd\t{$src2, $src1, $dst|$dst, $src1, $src2}",
               "cmppd\t{$cc, $src2, $src1, $dst|$dst, $src1, $src2, $cc}",
               SSEPackedDouble>, TB, OpSize, VEX_4V;
defm VCMPPSY : sse12_cmp_packed<VR256, f256mem, AVXCC, int_x86_avx_cmp_ps_256,
               "cmp${cc}ps\t{$src2, $src1, $dst|$dst, $src1, $src2}",
               "cmpps\t{$cc, $src2, $src1, $dst|$dst, $src1, $src2, $cc}",
               SSEPackedSingle>, TB, VEX_4V, VEX_L;
defm VCMPPDY : sse12_cmp_packed<VR256, f256mem, AVXCC, int_x86_avx_cmp_pd_256,
               "cmp${cc}pd\t{$src2, $src1, $dst|$dst, $src1, $src2}",
               "cmppd\t{$cc, $src2, $src1, $dst|$dst, $src1, $src2, $cc}",
               SSEPackedDouble>, TB, OpSize, VEX_4V, VEX_L;
let Constraints = "$src1 = $dst" in {
  defm CMPPS : sse12_cmp_packed<VR128, f128mem, SSECC, int_x86_sse_cmp_ps,
                 "cmp${cc}ps\t{$src2, $dst|$dst, $src2}",
                 "cmpps\t{$cc, $src2, $dst|$dst, $src2, $cc}",
                 SSEPackedSingle, SSE_ALU_F32P>, TB;
  defm CMPPD : sse12_cmp_packed<VR128, f128mem, SSECC, int_x86_sse2_cmp_pd,
                 "cmp${cc}pd\t{$src2, $dst|$dst, $src2}",
                 "cmppd\t{$cc, $src2, $dst|$dst, $src2, $cc}",
                 SSEPackedDouble, SSE_ALU_F64P>, TB, OpSize;
}

let Predicates = [HasAVX] in {
def : Pat<(v4i32 (X86cmpp (v4f32 VR128:$src1), VR128:$src2, imm:$cc)),
          (VCMPPSrri (v4f32 VR128:$src1), (v4f32 VR128:$src2), imm:$cc)>;
def : Pat<(v4i32 (X86cmpp (v4f32 VR128:$src1), (memop addr:$src2), imm:$cc)),
          (VCMPPSrmi (v4f32 VR128:$src1), addr:$src2, imm:$cc)>;
def : Pat<(v2i64 (X86cmpp (v2f64 VR128:$src1), VR128:$src2, imm:$cc)),
          (VCMPPDrri VR128:$src1, VR128:$src2, imm:$cc)>;
def : Pat<(v2i64 (X86cmpp (v2f64 VR128:$src1), (memop addr:$src2), imm:$cc)),
          (VCMPPDrmi VR128:$src1, addr:$src2, imm:$cc)>;

def : Pat<(v8i32 (X86cmpp (v8f32 VR256:$src1), VR256:$src2, imm:$cc)),
          (VCMPPSYrri (v8f32 VR256:$src1), (v8f32 VR256:$src2), imm:$cc)>;
def : Pat<(v8i32 (X86cmpp (v8f32 VR256:$src1), (memop addr:$src2), imm:$cc)),
          (VCMPPSYrmi (v8f32 VR256:$src1), addr:$src2, imm:$cc)>;
def : Pat<(v4i64 (X86cmpp (v4f64 VR256:$src1), VR256:$src2, imm:$cc)),
          (VCMPPDYrri VR256:$src1, VR256:$src2, imm:$cc)>;
def : Pat<(v4i64 (X86cmpp (v4f64 VR256:$src1), (memop addr:$src2), imm:$cc)),
          (VCMPPDYrmi VR256:$src1, addr:$src2, imm:$cc)>;
}

let Predicates = [UseSSE1] in {
def : Pat<(v4i32 (X86cmpp (v4f32 VR128:$src1), VR128:$src2, imm:$cc)),
          (CMPPSrri (v4f32 VR128:$src1), (v4f32 VR128:$src2), imm:$cc)>;
def : Pat<(v4i32 (X86cmpp (v4f32 VR128:$src1), (memop addr:$src2), imm:$cc)),
          (CMPPSrmi (v4f32 VR128:$src1), addr:$src2, imm:$cc)>;
}

let Predicates = [UseSSE2] in {
def : Pat<(v2i64 (X86cmpp (v2f64 VR128:$src1), VR128:$src2, imm:$cc)),
          (CMPPDrri VR128:$src1, VR128:$src2, imm:$cc)>;
def : Pat<(v2i64 (X86cmpp (v2f64 VR128:$src1), (memop addr:$src2), imm:$cc)),
          (CMPPDrmi VR128:$src1, addr:$src2, imm:$cc)>;
}

//===----------------------------------------------------------------------===//
// SSE 1 & 2 - Shuffle Instructions
//===----------------------------------------------------------------------===//

/// sse12_shuffle - sse 1 & 2 shuffle instructions
multiclass sse12_shuffle<RegisterClass RC, X86MemOperand x86memop,
                         ValueType vt, string asm, PatFrag mem_frag,
                         Domain d, bit IsConvertibleToThreeAddress = 0> {
  def rmi : PIi8<0xC6, MRMSrcMem, (outs RC:$dst),
                   (ins RC:$src1, x86memop:$src2, i8imm:$src3), asm,
                   [(set RC:$dst, (vt (X86Shufp RC:$src1, (mem_frag addr:$src2),
                                       (i8 imm:$src3))))], IIC_SSE_SHUFP, d>,
            Sched<[WriteShuffleLd, ReadAfterLd]>;
  let isConvertibleToThreeAddress = IsConvertibleToThreeAddress in
    def rri : PIi8<0xC6, MRMSrcReg, (outs RC:$dst),
                   (ins RC:$src1, RC:$src2, i8imm:$src3), asm,
                   [(set RC:$dst, (vt (X86Shufp RC:$src1, RC:$src2,
                                       (i8 imm:$src3))))], IIC_SSE_SHUFP, d>,
              Sched<[WriteShuffle]>;
}

defm VSHUFPS  : sse12_shuffle<VR128, f128mem, v4f32,
           "shufps\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
           loadv4f32, SSEPackedSingle>, TB, VEX_4V;
defm VSHUFPSY : sse12_shuffle<VR256, f256mem, v8f32,
           "shufps\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
           loadv8f32, SSEPackedSingle>, TB, VEX_4V, VEX_L;
defm VSHUFPD  : sse12_shuffle<VR128, f128mem, v2f64,
           "shufpd\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
           loadv2f64, SSEPackedDouble>, TB, OpSize, VEX_4V;
defm VSHUFPDY : sse12_shuffle<VR256, f256mem, v4f64,
           "shufpd\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
           loadv4f64, SSEPackedDouble>, TB, OpSize, VEX_4V, VEX_L;

let Constraints = "$src1 = $dst" in {
  defm SHUFPS : sse12_shuffle<VR128, f128mem, v4f32,
                    "shufps\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                    memopv4f32, SSEPackedSingle, 1 /* cvt to pshufd */>,
                    TB;
  defm SHUFPD : sse12_shuffle<VR128, f128mem, v2f64,
                    "shufpd\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                    memopv2f64, SSEPackedDouble, 1 /* cvt to pshufd */>,
                    TB, OpSize;
}

let Predicates = [HasAVX] in {
  def : Pat<(v4i32 (X86Shufp VR128:$src1,
                       (bc_v4i32 (loadv2i64 addr:$src2)), (i8 imm:$imm))),
            (VSHUFPSrmi VR128:$src1, addr:$src2, imm:$imm)>;
  def : Pat<(v4i32 (X86Shufp VR128:$src1, VR128:$src2, (i8 imm:$imm))),
            (VSHUFPSrri VR128:$src1, VR128:$src2, imm:$imm)>;

  def : Pat<(v2i64 (X86Shufp VR128:$src1,
                       (loadv2i64 addr:$src2), (i8 imm:$imm))),
            (VSHUFPDrmi VR128:$src1, addr:$src2, imm:$imm)>;
  def : Pat<(v2i64 (X86Shufp VR128:$src1, VR128:$src2, (i8 imm:$imm))),
            (VSHUFPDrri VR128:$src1, VR128:$src2, imm:$imm)>;

  // 256-bit patterns
  def : Pat<(v8i32 (X86Shufp VR256:$src1, VR256:$src2, (i8 imm:$imm))),
            (VSHUFPSYrri VR256:$src1, VR256:$src2, imm:$imm)>;
  def : Pat<(v8i32 (X86Shufp VR256:$src1,
                      (bc_v8i32 (loadv4i64 addr:$src2)), (i8 imm:$imm))),
            (VSHUFPSYrmi VR256:$src1, addr:$src2, imm:$imm)>;

  def : Pat<(v4i64 (X86Shufp VR256:$src1, VR256:$src2, (i8 imm:$imm))),
            (VSHUFPDYrri VR256:$src1, VR256:$src2, imm:$imm)>;
  def : Pat<(v4i64 (X86Shufp VR256:$src1,
                              (loadv4i64 addr:$src2), (i8 imm:$imm))),
            (VSHUFPDYrmi VR256:$src1, addr:$src2, imm:$imm)>;
}

let Predicates = [UseSSE1] in {
  def : Pat<(v4i32 (X86Shufp VR128:$src1,
                       (bc_v4i32 (memopv2i64 addr:$src2)), (i8 imm:$imm))),
            (SHUFPSrmi VR128:$src1, addr:$src2, imm:$imm)>;
  def : Pat<(v4i32 (X86Shufp VR128:$src1, VR128:$src2, (i8 imm:$imm))),
            (SHUFPSrri VR128:$src1, VR128:$src2, imm:$imm)>;
}

let Predicates = [UseSSE2] in {
  // Generic SHUFPD patterns
  def : Pat<(v2i64 (X86Shufp VR128:$src1,
                       (memopv2i64 addr:$src2), (i8 imm:$imm))),
            (SHUFPDrmi VR128:$src1, addr:$src2, imm:$imm)>;
  def : Pat<(v2i64 (X86Shufp VR128:$src1, VR128:$src2, (i8 imm:$imm))),
            (SHUFPDrri VR128:$src1, VR128:$src2, imm:$imm)>;
}

//===----------------------------------------------------------------------===//
// SSE 1 & 2 - Unpack Instructions
//===----------------------------------------------------------------------===//

/// sse12_unpack_interleave - sse 1 & 2 unpack and interleave
multiclass sse12_unpack_interleave<bits<8> opc, SDNode OpNode, ValueType vt,
                                   PatFrag mem_frag, RegisterClass RC,
                                   X86MemOperand x86memop, string asm,
                                   Domain d> {
    def rr : PI<opc, MRMSrcReg,
                (outs RC:$dst), (ins RC:$src1, RC:$src2),
                asm, [(set RC:$dst,
                           (vt (OpNode RC:$src1, RC:$src2)))],
                           IIC_SSE_UNPCK, d>, Sched<[WriteShuffle]>;
    def rm : PI<opc, MRMSrcMem,
                (outs RC:$dst), (ins RC:$src1, x86memop:$src2),
                asm, [(set RC:$dst,
                           (vt (OpNode RC:$src1,
                                       (mem_frag addr:$src2))))],
                                       IIC_SSE_UNPCK, d>,
             Sched<[WriteShuffleLd, ReadAfterLd]>;
}

defm VUNPCKHPS: sse12_unpack_interleave<0x15, X86Unpckh, v4f32, loadv4f32,
      VR128, f128mem, "unpckhps\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                     SSEPackedSingle>, TB, VEX_4V;
defm VUNPCKHPD: sse12_unpack_interleave<0x15, X86Unpckh, v2f64, loadv2f64,
      VR128, f128mem, "unpckhpd\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                     SSEPackedDouble>, TB, OpSize, VEX_4V;
defm VUNPCKLPS: sse12_unpack_interleave<0x14, X86Unpckl, v4f32, loadv4f32,
      VR128, f128mem, "unpcklps\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                     SSEPackedSingle>, TB, VEX_4V;
defm VUNPCKLPD: sse12_unpack_interleave<0x14, X86Unpckl, v2f64, loadv2f64,
      VR128, f128mem, "unpcklpd\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                     SSEPackedDouble>, TB, OpSize, VEX_4V;

defm VUNPCKHPSY: sse12_unpack_interleave<0x15, X86Unpckh, v8f32, loadv8f32,
      VR256, f256mem, "unpckhps\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                     SSEPackedSingle>, TB, VEX_4V, VEX_L;
defm VUNPCKHPDY: sse12_unpack_interleave<0x15, X86Unpckh, v4f64, loadv4f64,
      VR256, f256mem, "unpckhpd\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                     SSEPackedDouble>, TB, OpSize, VEX_4V, VEX_L;
defm VUNPCKLPSY: sse12_unpack_interleave<0x14, X86Unpckl, v8f32, loadv8f32,
      VR256, f256mem, "unpcklps\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                     SSEPackedSingle>, TB, VEX_4V, VEX_L;
defm VUNPCKLPDY: sse12_unpack_interleave<0x14, X86Unpckl, v4f64, loadv4f64,
      VR256, f256mem, "unpcklpd\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                     SSEPackedDouble>, TB, OpSize, VEX_4V, VEX_L;

let Constraints = "$src1 = $dst" in {
  defm UNPCKHPS: sse12_unpack_interleave<0x15, X86Unpckh, v4f32, memopv4f32,
        VR128, f128mem, "unpckhps\t{$src2, $dst|$dst, $src2}",
                       SSEPackedSingle>, TB;
  defm UNPCKHPD: sse12_unpack_interleave<0x15, X86Unpckh, v2f64, memopv2f64,
        VR128, f128mem, "unpckhpd\t{$src2, $dst|$dst, $src2}",
                       SSEPackedDouble>, TB, OpSize;
  defm UNPCKLPS: sse12_unpack_interleave<0x14, X86Unpckl, v4f32, memopv4f32,
        VR128, f128mem, "unpcklps\t{$src2, $dst|$dst, $src2}",
                       SSEPackedSingle>, TB;
  defm UNPCKLPD: sse12_unpack_interleave<0x14, X86Unpckl, v2f64, memopv2f64,
        VR128, f128mem, "unpcklpd\t{$src2, $dst|$dst, $src2}",
                       SSEPackedDouble>, TB, OpSize;
} // Constraints = "$src1 = $dst"

let Predicates = [HasAVX1Only] in {
  def : Pat<(v8i32 (X86Unpckl VR256:$src1, (bc_v8i32 (loadv4i64 addr:$src2)))),
            (VUNPCKLPSYrm VR256:$src1, addr:$src2)>;
  def : Pat<(v8i32 (X86Unpckl VR256:$src1, VR256:$src2)),
            (VUNPCKLPSYrr VR256:$src1, VR256:$src2)>;
  def : Pat<(v8i32 (X86Unpckh VR256:$src1, (bc_v8i32 (loadv4i64 addr:$src2)))),
            (VUNPCKHPSYrm VR256:$src1, addr:$src2)>;
  def : Pat<(v8i32 (X86Unpckh VR256:$src1, VR256:$src2)),
            (VUNPCKHPSYrr VR256:$src1, VR256:$src2)>;

  def : Pat<(v4i64 (X86Unpckl VR256:$src1, (loadv4i64 addr:$src2))),
            (VUNPCKLPDYrm VR256:$src1, addr:$src2)>;
  def : Pat<(v4i64 (X86Unpckl VR256:$src1, VR256:$src2)),
            (VUNPCKLPDYrr VR256:$src1, VR256:$src2)>;
  def : Pat<(v4i64 (X86Unpckh VR256:$src1, (loadv4i64 addr:$src2))),
            (VUNPCKHPDYrm VR256:$src1, addr:$src2)>;
  def : Pat<(v4i64 (X86Unpckh VR256:$src1, VR256:$src2)),
            (VUNPCKHPDYrr VR256:$src1, VR256:$src2)>;
}

let Predicates = [HasAVX] in {
  // FIXME: Instead of X86Movddup, there should be a X86Unpckl here, the
  // problem is during lowering, where it's not possible to recognize the load
  // fold cause it has two uses through a bitcast. One use disappears at isel
  // time and the fold opportunity reappears.
  def : Pat<(v2f64 (X86Movddup VR128:$src)),
            (VUNPCKLPDrr VR128:$src, VR128:$src)>;
}

let Predicates = [UseSSE2] in {
  // FIXME: Instead of X86Movddup, there should be a X86Unpckl here, the
  // problem is during lowering, where it's not possible to recognize the load
  // fold cause it has two uses through a bitcast. One use disappears at isel
  // time and the fold opportunity reappears.
  def : Pat<(v2f64 (X86Movddup VR128:$src)),
            (UNPCKLPDrr VR128:$src, VR128:$src)>;
}

//===----------------------------------------------------------------------===//
// SSE 1 & 2 - Extract Floating-Point Sign mask
//===----------------------------------------------------------------------===//

/// sse12_extr_sign_mask - sse 1 & 2 unpack and interleave
multiclass sse12_extr_sign_mask<RegisterClass RC, Intrinsic Int, string asm,
                                Domain d> {
  def rr : PI<0x50, MRMSrcReg, (outs GR32orGR64:$dst), (ins RC:$src),
              !strconcat(asm, "\t{$src, $dst|$dst, $src}"),
              [(set GR32orGR64:$dst, (Int RC:$src))], IIC_SSE_MOVMSK, d>,
              Sched<[WriteVecLogic]>;
}

let Predicates = [HasAVX] in {
  defm VMOVMSKPS : sse12_extr_sign_mask<VR128, int_x86_sse_movmsk_ps,
                                        "movmskps", SSEPackedSingle>, TB, VEX;
  defm VMOVMSKPD : sse12_extr_sign_mask<VR128, int_x86_sse2_movmsk_pd,
                                        "movmskpd", SSEPackedDouble>, TB,
                                        OpSize, VEX;
  defm VMOVMSKPSY : sse12_extr_sign_mask<VR256, int_x86_avx_movmsk_ps_256,
                                        "movmskps", SSEPackedSingle>, TB,
                                        VEX, VEX_L;
  defm VMOVMSKPDY : sse12_extr_sign_mask<VR256, int_x86_avx_movmsk_pd_256,
                                        "movmskpd", SSEPackedDouble>, TB,
                                        OpSize, VEX, VEX_L;

  def : Pat<(i32 (X86fgetsign FR32:$src)),
            (VMOVMSKPSrr (COPY_TO_REGCLASS FR32:$src, VR128))>;
  def : Pat<(i64 (X86fgetsign FR32:$src)),
            (SUBREG_TO_REG (i64 0),
             (VMOVMSKPSrr (COPY_TO_REGCLASS FR32:$src, VR128)), sub_32bit)>;
  def : Pat<(i32 (X86fgetsign FR64:$src)),
            (VMOVMSKPDrr (COPY_TO_REGCLASS FR64:$src, VR128))>;
  def : Pat<(i64 (X86fgetsign FR64:$src)),
            (SUBREG_TO_REG (i64 0),
             (VMOVMSKPDrr (COPY_TO_REGCLASS FR64:$src, VR128)), sub_32bit)>;
}

defm MOVMSKPS : sse12_extr_sign_mask<VR128, int_x86_sse_movmsk_ps, "movmskps",
                                     SSEPackedSingle>, TB;
defm MOVMSKPD : sse12_extr_sign_mask<VR128, int_x86_sse2_movmsk_pd, "movmskpd",
                                     SSEPackedDouble>, TB, OpSize;

def : Pat<(i32 (X86fgetsign FR32:$src)),
          (MOVMSKPSrr (COPY_TO_REGCLASS FR32:$src, VR128))>,
      Requires<[UseSSE1]>;
def : Pat<(i64 (X86fgetsign FR32:$src)),
          (SUBREG_TO_REG (i64 0),
           (MOVMSKPSrr (COPY_TO_REGCLASS FR32:$src, VR128)), sub_32bit)>,
      Requires<[UseSSE1]>;
def : Pat<(i32 (X86fgetsign FR64:$src)),
          (MOVMSKPDrr (COPY_TO_REGCLASS FR64:$src, VR128))>,
      Requires<[UseSSE2]>;
def : Pat<(i64 (X86fgetsign FR64:$src)),
          (SUBREG_TO_REG (i64 0),
           (MOVMSKPDrr (COPY_TO_REGCLASS FR64:$src, VR128)), sub_32bit)>,
      Requires<[UseSSE2]>;

//===---------------------------------------------------------------------===//
// SSE2 - Packed Integer Logical Instructions
//===---------------------------------------------------------------------===//

let ExeDomain = SSEPackedInt in { // SSE integer instructions

/// PDI_binop_rm - Simple SSE2 binary operator.
multiclass PDI_binop_rm<bits<8> opc, string OpcodeStr, SDNode OpNode,
                        ValueType OpVT, RegisterClass RC, PatFrag memop_frag,
                        X86MemOperand x86memop, OpndItins itins,
                        bit IsCommutable, bit Is2Addr> {
  let isCommutable = IsCommutable in
  def rr : PDI<opc, MRMSrcReg, (outs RC:$dst),
       (ins RC:$src1, RC:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set RC:$dst, (OpVT (OpNode RC:$src1, RC:$src2)))], itins.rr>,
       Sched<[itins.Sched]>;
  def rm : PDI<opc, MRMSrcMem, (outs RC:$dst),
       (ins RC:$src1, x86memop:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set RC:$dst, (OpVT (OpNode RC:$src1,
                                     (bitconvert (memop_frag addr:$src2)))))],
                                     itins.rm>,
       Sched<[itins.Sched.Folded, ReadAfterLd]>;
}
} // ExeDomain = SSEPackedInt

multiclass PDI_binop_all<bits<8> opc, string OpcodeStr, SDNode Opcode,
                         ValueType OpVT128, ValueType OpVT256,
                         OpndItins itins, bit IsCommutable = 0> {
let Predicates = [HasAVX] in
  defm V#NAME : PDI_binop_rm<opc, !strconcat("v", OpcodeStr), Opcode, OpVT128,
                    VR128, loadv2i64, i128mem, itins, IsCommutable, 0>, VEX_4V;

let Constraints = "$src1 = $dst" in
  defm NAME : PDI_binop_rm<opc, OpcodeStr, Opcode, OpVT128, VR128,
                           memopv2i64, i128mem, itins, IsCommutable, 1>;

let Predicates = [HasAVX2] in
  defm V#NAME#Y : PDI_binop_rm<opc, !strconcat("v", OpcodeStr), Opcode,
                               OpVT256, VR256, loadv4i64, i256mem, itins,
                               IsCommutable, 0>, VEX_4V, VEX_L;
}

// These are ordered here for pattern ordering requirements with the fp versions

defm PAND  : PDI_binop_all<0xDB, "pand", and, v2i64, v4i64, SSE_BIT_ITINS_P, 1>;
defm POR   : PDI_binop_all<0xEB, "por", or, v2i64, v4i64, SSE_BIT_ITINS_P, 1>;
defm PXOR  : PDI_binop_all<0xEF, "pxor", xor, v2i64, v4i64, SSE_BIT_ITINS_P, 1>;
defm PANDN : PDI_binop_all<0xDF, "pandn", X86andnp, v2i64, v4i64,
                           SSE_BIT_ITINS_P, 0>;

//===----------------------------------------------------------------------===//
// SSE 1 & 2 - Logical Instructions
//===----------------------------------------------------------------------===//

/// sse12_fp_alias_pack_logical - SSE 1 & 2 aliased packed FP logical ops
///
multiclass sse12_fp_alias_pack_logical<bits<8> opc, string OpcodeStr,
                                       SDNode OpNode, OpndItins itins> {
  defm V#NAME#PS : sse12_fp_packed<opc, !strconcat(OpcodeStr, "ps"), OpNode,
              FR32, f32, f128mem, memopfsf32, SSEPackedSingle, itins, 0>,
              TB, VEX_4V;

  defm V#NAME#PD : sse12_fp_packed<opc, !strconcat(OpcodeStr, "pd"), OpNode,
        FR64, f64, f128mem, memopfsf64, SSEPackedDouble, itins, 0>,
        TB, OpSize, VEX_4V;

  let Constraints = "$src1 = $dst" in {
    defm PS : sse12_fp_packed<opc, !strconcat(OpcodeStr, "ps"), OpNode, FR32,
                f32, f128mem, memopfsf32, SSEPackedSingle, itins>,
                TB;

    defm PD : sse12_fp_packed<opc, !strconcat(OpcodeStr, "pd"), OpNode, FR64,
                f64, f128mem, memopfsf64, SSEPackedDouble, itins>,
                TB, OpSize;
  }
}

// Alias bitwise logical operations using SSE logical ops on packed FP values.
let isCodeGenOnly = 1 in {
  defm FsAND  : sse12_fp_alias_pack_logical<0x54, "and", X86fand,
                SSE_BIT_ITINS_P>;
  defm FsOR   : sse12_fp_alias_pack_logical<0x56, "or", X86for,
                SSE_BIT_ITINS_P>;
  defm FsXOR  : sse12_fp_alias_pack_logical<0x57, "xor", X86fxor,
                SSE_BIT_ITINS_P>;

  let isCommutable = 0 in
    defm FsANDN : sse12_fp_alias_pack_logical<0x55, "andn", X86fandn,
                  SSE_BIT_ITINS_P>;
}

/// sse12_fp_packed_logical - SSE 1 & 2 packed FP logical ops
///
multiclass sse12_fp_packed_logical<bits<8> opc, string OpcodeStr,
                                   SDNode OpNode> {
  defm V#NAME#PSY : sse12_fp_packed_logical_rm<opc, VR256, SSEPackedSingle,
        !strconcat(OpcodeStr, "ps"), f256mem,
        [(set VR256:$dst, (v4i64 (OpNode VR256:$src1, VR256:$src2)))],
        [(set VR256:$dst, (OpNode (bc_v4i64 (v8f32 VR256:$src1)),
                           (loadv4i64 addr:$src2)))], 0>, TB, VEX_4V, VEX_L;

  defm V#NAME#PDY : sse12_fp_packed_logical_rm<opc, VR256, SSEPackedDouble,
        !strconcat(OpcodeStr, "pd"), f256mem,
        [(set VR256:$dst, (OpNode (bc_v4i64 (v4f64 VR256:$src1)),
                                  (bc_v4i64 (v4f64 VR256:$src2))))],
        [(set VR256:$dst, (OpNode (bc_v4i64 (v4f64 VR256:$src1)),
                                  (loadv4i64 addr:$src2)))], 0>,
                                  TB, OpSize, VEX_4V, VEX_L;

  // In AVX no need to add a pattern for 128-bit logical rr ps, because they
  // are all promoted to v2i64, and the patterns are covered by the int
  // version. This is needed in SSE only, because v2i64 isn't supported on
  // SSE1, but only on SSE2.
  defm V#NAME#PS : sse12_fp_packed_logical_rm<opc, VR128, SSEPackedSingle,
       !strconcat(OpcodeStr, "ps"), f128mem, [],
       [(set VR128:$dst, (OpNode (bc_v2i64 (v4f32 VR128:$src1)),
                                 (loadv2i64 addr:$src2)))], 0>, TB, VEX_4V;

  defm V#NAME#PD : sse12_fp_packed_logical_rm<opc, VR128, SSEPackedDouble,
       !strconcat(OpcodeStr, "pd"), f128mem,
       [(set VR128:$dst, (OpNode (bc_v2i64 (v2f64 VR128:$src1)),
                                 (bc_v2i64 (v2f64 VR128:$src2))))],
       [(set VR128:$dst, (OpNode (bc_v2i64 (v2f64 VR128:$src1)),
                                 (loadv2i64 addr:$src2)))], 0>,
                                                 TB, OpSize, VEX_4V;

  let Constraints = "$src1 = $dst" in {
    defm PS : sse12_fp_packed_logical_rm<opc, VR128, SSEPackedSingle,
         !strconcat(OpcodeStr, "ps"), f128mem,
         [(set VR128:$dst, (v2i64 (OpNode VR128:$src1, VR128:$src2)))],
         [(set VR128:$dst, (OpNode (bc_v2i64 (v4f32 VR128:$src1)),
                                   (memopv2i64 addr:$src2)))]>, TB;

    defm PD : sse12_fp_packed_logical_rm<opc, VR128, SSEPackedDouble,
         !strconcat(OpcodeStr, "pd"), f128mem,
         [(set VR128:$dst, (OpNode (bc_v2i64 (v2f64 VR128:$src1)),
                                   (bc_v2i64 (v2f64 VR128:$src2))))],
         [(set VR128:$dst, (OpNode (bc_v2i64 (v2f64 VR128:$src1)),
                                   (memopv2i64 addr:$src2)))]>, TB, OpSize;
  }
}

defm AND  : sse12_fp_packed_logical<0x54, "and", and>;
defm OR   : sse12_fp_packed_logical<0x56, "or", or>;
defm XOR  : sse12_fp_packed_logical<0x57, "xor", xor>;
let isCommutable = 0 in
  defm ANDN : sse12_fp_packed_logical<0x55, "andn", X86andnp>;

//===----------------------------------------------------------------------===//
// SSE 1 & 2 - Arithmetic Instructions
//===----------------------------------------------------------------------===//

/// basic_sse12_fp_binop_xxx - SSE 1 & 2 binops come in both scalar and
/// vector forms.
///
/// In addition, we also have a special variant of the scalar form here to
/// represent the associated intrinsic operation.  This form is unlike the
/// plain scalar form, in that it takes an entire vector (instead of a scalar)
/// and leaves the top elements unmodified (therefore these cannot be commuted).
///
/// These three forms can each be reg+reg or reg+mem.
///

/// FIXME: once all 256-bit intrinsics are matched, cleanup and refactor those
/// classes below
multiclass basic_sse12_fp_binop_p<bits<8> opc, string OpcodeStr,
                                  SDNode OpNode, SizeItins itins> {
  defm V#NAME#PS : sse12_fp_packed<opc, !strconcat(OpcodeStr, "ps"), OpNode,
                               VR128, v4f32, f128mem, loadv4f32,
                               SSEPackedSingle, itins.s, 0>, TB, VEX_4V;
  defm V#NAME#PD : sse12_fp_packed<opc, !strconcat(OpcodeStr, "pd"), OpNode,
                               VR128, v2f64, f128mem, loadv2f64,
                               SSEPackedDouble, itins.d, 0>, TB, OpSize, VEX_4V;

  defm V#NAME#PSY : sse12_fp_packed<opc, !strconcat(OpcodeStr, "ps"),
                        OpNode, VR256, v8f32, f256mem, loadv8f32,
                        SSEPackedSingle, itins.s, 0>, TB, VEX_4V, VEX_L;
  defm V#NAME#PDY : sse12_fp_packed<opc, !strconcat(OpcodeStr, "pd"),
                        OpNode, VR256, v4f64, f256mem, loadv4f64,
                        SSEPackedDouble, itins.d, 0>, TB, OpSize, VEX_4V, VEX_L;

  let Constraints = "$src1 = $dst" in {
    defm PS : sse12_fp_packed<opc, !strconcat(OpcodeStr, "ps"), OpNode, VR128,
                              v4f32, f128mem, memopv4f32, SSEPackedSingle,
                              itins.s>, TB;
    defm PD : sse12_fp_packed<opc, !strconcat(OpcodeStr, "pd"), OpNode, VR128,
                              v2f64, f128mem, memopv2f64, SSEPackedDouble,
                              itins.d>, TB, OpSize;
  }
}

multiclass basic_sse12_fp_binop_s<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                  SizeItins itins> {
  defm V#NAME#SS : sse12_fp_scalar<opc, !strconcat(OpcodeStr, "ss"),
                         OpNode, FR32, f32mem, itins.s, 0>, XS, VEX_4V, VEX_LIG;
  defm V#NAME#SD : sse12_fp_scalar<opc, !strconcat(OpcodeStr, "sd"),
                         OpNode, FR64, f64mem, itins.d, 0>, XD, VEX_4V, VEX_LIG;

  let Constraints = "$src1 = $dst" in {
    defm SS : sse12_fp_scalar<opc, !strconcat(OpcodeStr, "ss"),
                              OpNode, FR32, f32mem, itins.s>, XS;
    defm SD : sse12_fp_scalar<opc, !strconcat(OpcodeStr, "sd"),
                              OpNode, FR64, f64mem, itins.d>, XD;
  }
}

multiclass basic_sse12_fp_binop_s_int<bits<8> opc, string OpcodeStr,
                                      SizeItins itins> {
  defm V#NAME#SS : sse12_fp_scalar_int<opc, OpcodeStr, VR128,
                   !strconcat(OpcodeStr, "ss"), "", "_ss", ssmem, sse_load_f32,
                   itins.s, 0>, XS, VEX_4V, VEX_LIG;
  defm V#NAME#SD : sse12_fp_scalar_int<opc, OpcodeStr, VR128,
                   !strconcat(OpcodeStr, "sd"), "2", "_sd", sdmem, sse_load_f64,
                   itins.d, 0>, XD, VEX_4V, VEX_LIG;

  let Constraints = "$src1 = $dst" in {
    defm SS : sse12_fp_scalar_int<opc, OpcodeStr, VR128,
                   !strconcat(OpcodeStr, "ss"), "", "_ss", ssmem, sse_load_f32,
                   itins.s>, XS;
    defm SD : sse12_fp_scalar_int<opc, OpcodeStr, VR128,
                   !strconcat(OpcodeStr, "sd"), "2", "_sd", sdmem, sse_load_f64,
                   itins.d>, XD;
  }
}

// Binary Arithmetic instructions
defm ADD : basic_sse12_fp_binop_p<0x58, "add", fadd, SSE_ALU_ITINS_P>,
           basic_sse12_fp_binop_s<0x58, "add", fadd, SSE_ALU_ITINS_S>,
           basic_sse12_fp_binop_s_int<0x58, "add", SSE_ALU_ITINS_S>;
defm MUL : basic_sse12_fp_binop_p<0x59, "mul", fmul, SSE_MUL_ITINS_P>,
           basic_sse12_fp_binop_s<0x59, "mul", fmul, SSE_MUL_ITINS_S>,
           basic_sse12_fp_binop_s_int<0x59, "mul", SSE_MUL_ITINS_S>;
let isCommutable = 0 in {
  defm SUB : basic_sse12_fp_binop_p<0x5C, "sub", fsub, SSE_ALU_ITINS_P>,
             basic_sse12_fp_binop_s<0x5C, "sub", fsub, SSE_ALU_ITINS_S>,
             basic_sse12_fp_binop_s_int<0x5C, "sub", SSE_ALU_ITINS_S>;
  defm DIV : basic_sse12_fp_binop_p<0x5E, "div", fdiv, SSE_DIV_ITINS_P>,
             basic_sse12_fp_binop_s<0x5E, "div", fdiv, SSE_DIV_ITINS_S>,
             basic_sse12_fp_binop_s_int<0x5E, "div", SSE_DIV_ITINS_S>;
  defm MAX : basic_sse12_fp_binop_p<0x5F, "max", X86fmax, SSE_ALU_ITINS_P>,
             basic_sse12_fp_binop_s<0x5F, "max", X86fmax, SSE_ALU_ITINS_S>,
             basic_sse12_fp_binop_s_int<0x5F, "max", SSE_ALU_ITINS_S>;
  defm MIN : basic_sse12_fp_binop_p<0x5D, "min", X86fmin, SSE_ALU_ITINS_P>,
             basic_sse12_fp_binop_s<0x5D, "min", X86fmin, SSE_ALU_ITINS_S>,
             basic_sse12_fp_binop_s_int<0x5D, "min", SSE_ALU_ITINS_S>;
}

let isCodeGenOnly = 1 in {
  defm MAXC: basic_sse12_fp_binop_p<0x5F, "max", X86fmaxc, SSE_ALU_ITINS_P>,
             basic_sse12_fp_binop_s<0x5F, "max", X86fmaxc, SSE_ALU_ITINS_S>;
  defm MINC: basic_sse12_fp_binop_p<0x5D, "min", X86fminc, SSE_ALU_ITINS_P>,
             basic_sse12_fp_binop_s<0x5D, "min", X86fminc, SSE_ALU_ITINS_S>;
}

/// Unop Arithmetic
/// In addition, we also have a special variant of the scalar form here to
/// represent the associated intrinsic operation.  This form is unlike the
/// plain scalar form, in that it takes an entire vector (instead of a
/// scalar) and leaves the top elements undefined.
///
/// And, we have a special variant form for a full-vector intrinsic form.

let Sched = WriteFSqrt in {
def SSE_SQRTPS : OpndItins<
  IIC_SSE_SQRTPS_RR, IIC_SSE_SQRTPS_RM
>;

def SSE_SQRTSS : OpndItins<
  IIC_SSE_SQRTSS_RR, IIC_SSE_SQRTSS_RM
>;

def SSE_SQRTPD : OpndItins<
  IIC_SSE_SQRTPD_RR, IIC_SSE_SQRTPD_RM
>;

def SSE_SQRTSD : OpndItins<
  IIC_SSE_SQRTSD_RR, IIC_SSE_SQRTSD_RM
>;
}

let Sched = WriteFRcp in {
def SSE_RCPP : OpndItins<
  IIC_SSE_RCPP_RR, IIC_SSE_RCPP_RM
>;

def SSE_RCPS : OpndItins<
  IIC_SSE_RCPS_RR, IIC_SSE_RCPS_RM
>;
}

/// sse1_fp_unop_s - SSE1 unops in scalar form.
multiclass sse1_fp_unop_s<bits<8> opc, string OpcodeStr,
                          SDNode OpNode, Intrinsic F32Int, OpndItins itins> {
let Predicates = [HasAVX], hasSideEffects = 0 in {
  def V#NAME#SSr : SSI<opc, MRMSrcReg, (outs FR32:$dst),
                      (ins FR32:$src1, FR32:$src2),
                      !strconcat("v", OpcodeStr,
                                 "ss\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                      []>, VEX_4V, VEX_LIG, Sched<[itins.Sched]>;
  let mayLoad = 1 in {
  def V#NAME#SSm : SSI<opc, MRMSrcMem, (outs FR32:$dst),
                      (ins FR32:$src1,f32mem:$src2),
                      !strconcat("v", OpcodeStr,
                                 "ss\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                      []>, VEX_4V, VEX_LIG,
                   Sched<[itins.Sched.Folded, ReadAfterLd]>;
  def V#NAME#SSm_Int : SSI<opc, MRMSrcMem, (outs VR128:$dst),
                      (ins VR128:$src1, ssmem:$src2),
                      !strconcat("v", OpcodeStr,
                                 "ss\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                      []>, VEX_4V, VEX_LIG,
                      Sched<[itins.Sched.Folded, ReadAfterLd]>;
  }
}

  def SSr : SSI<opc, MRMSrcReg, (outs FR32:$dst), (ins FR32:$src),
                !strconcat(OpcodeStr, "ss\t{$src, $dst|$dst, $src}"),
                [(set FR32:$dst, (OpNode FR32:$src))]>, Sched<[itins.Sched]>;
  // For scalar unary operations, fold a load into the operation
  // only in OptForSize mode. It eliminates an instruction, but it also
  // eliminates a whole-register clobber (the load), so it introduces a
  // partial register update condition.
  def SSm : I<opc, MRMSrcMem, (outs FR32:$dst), (ins f32mem:$src),
                !strconcat(OpcodeStr, "ss\t{$src, $dst|$dst, $src}"),
                [(set FR32:$dst, (OpNode (load addr:$src)))], itins.rm>, XS,
            Requires<[UseSSE1, OptForSize]>, Sched<[itins.Sched.Folded]>;
  def SSr_Int : SSI<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                    !strconcat(OpcodeStr, "ss\t{$src, $dst|$dst, $src}"),
                    [(set VR128:$dst, (F32Int VR128:$src))], itins.rr>,
                Sched<[itins.Sched]>;
  def SSm_Int : SSI<opc, MRMSrcMem, (outs VR128:$dst), (ins ssmem:$src),
                    !strconcat(OpcodeStr, "ss\t{$src, $dst|$dst, $src}"),
                    [(set VR128:$dst, (F32Int sse_load_f32:$src))], itins.rm>,
                Sched<[itins.Sched.Folded]>;
}

/// sse1_fp_unop_s_rw - SSE1 unops where vector form has a read-write operand.
multiclass sse1_fp_unop_rw<bits<8> opc, string OpcodeStr, SDNode OpNode,
                           OpndItins itins> {
let Predicates = [HasAVX], hasSideEffects = 0 in {
  def V#NAME#SSr : SSI<opc, MRMSrcReg, (outs FR32:$dst),
                       (ins FR32:$src1, FR32:$src2),
                       !strconcat("v", OpcodeStr,
                           "ss\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                []>, VEX_4V, VEX_LIG, Sched<[itins.Sched]>;
  let mayLoad = 1 in {
  def V#NAME#SSm : SSI<opc, MRMSrcMem, (outs FR32:$dst),
                      (ins FR32:$src1,f32mem:$src2),
                      !strconcat("v", OpcodeStr,
                                 "ss\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                      []>, VEX_4V, VEX_LIG,
                   Sched<[itins.Sched.Folded, ReadAfterLd]>;
  def V#NAME#SSm_Int : SSI<opc, MRMSrcMem, (outs VR128:$dst),
                      (ins VR128:$src1, ssmem:$src2),
                      !strconcat("v", OpcodeStr,
                                 "ss\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                      []>, VEX_4V, VEX_LIG,
                      Sched<[itins.Sched.Folded, ReadAfterLd]>;
  }
}

  def SSr : SSI<opc, MRMSrcReg, (outs FR32:$dst), (ins FR32:$src),
                !strconcat(OpcodeStr, "ss\t{$src, $dst|$dst, $src}"),
                [(set FR32:$dst, (OpNode FR32:$src))]>, Sched<[itins.Sched]>;
  // For scalar unary operations, fold a load into the operation
  // only in OptForSize mode. It eliminates an instruction, but it also
  // eliminates a whole-register clobber (the load), so it introduces a
  // partial register update condition.
  def SSm : I<opc, MRMSrcMem, (outs FR32:$dst), (ins f32mem:$src),
                !strconcat(OpcodeStr, "ss\t{$src, $dst|$dst, $src}"),
                [(set FR32:$dst, (OpNode (load addr:$src)))], itins.rm>, XS,
            Requires<[UseSSE1, OptForSize]>, Sched<[itins.Sched.Folded]>;
  let Constraints = "$src1 = $dst" in {
    def SSr_Int : SSI<opc, MRMSrcReg, (outs VR128:$dst),
                      (ins VR128:$src1, VR128:$src2),
                      !strconcat(OpcodeStr, "ss\t{$src2, $dst|$dst, $src2}"),
                      [], itins.rr>, Sched<[itins.Sched]>;
    let mayLoad = 1, hasSideEffects = 0 in
    def SSm_Int : SSI<opc, MRMSrcMem, (outs VR128:$dst),
                      (ins VR128:$src1, ssmem:$src2),
                      !strconcat(OpcodeStr, "ss\t{$src2, $dst|$dst, $src2}"),
                      [], itins.rm>, Sched<[itins.Sched.Folded, ReadAfterLd]>;
  }
}

/// sse1_fp_unop_p - SSE1 unops in packed form.
multiclass sse1_fp_unop_p<bits<8> opc, string OpcodeStr, SDNode OpNode,
                          OpndItins itins> {
let Predicates = [HasAVX] in {
  def V#NAME#PSr : PSI<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                       !strconcat("v", OpcodeStr,
                                  "ps\t{$src, $dst|$dst, $src}"),
                       [(set VR128:$dst, (v4f32 (OpNode VR128:$src)))],
                       itins.rr>, VEX, Sched<[itins.Sched]>;
  def V#NAME#PSm : PSI<opc, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                       !strconcat("v", OpcodeStr,
                                  "ps\t{$src, $dst|$dst, $src}"),
                       [(set VR128:$dst, (OpNode (loadv4f32 addr:$src)))],
                       itins.rm>, VEX, Sched<[itins.Sched.Folded]>;
  def V#NAME#PSYr : PSI<opc, MRMSrcReg, (outs VR256:$dst), (ins VR256:$src),
                        !strconcat("v", OpcodeStr,
                                   "ps\t{$src, $dst|$dst, $src}"),
                        [(set VR256:$dst, (v8f32 (OpNode VR256:$src)))],
                        itins.rr>, VEX, VEX_L, Sched<[itins.Sched]>;
  def V#NAME#PSYm : PSI<opc, MRMSrcMem, (outs VR256:$dst), (ins f256mem:$src),
                        !strconcat("v", OpcodeStr,
                                   "ps\t{$src, $dst|$dst, $src}"),
                        [(set VR256:$dst, (OpNode (loadv8f32 addr:$src)))],
                        itins.rm>, VEX, VEX_L, Sched<[itins.Sched.Folded]>;
}

  def PSr : PSI<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                !strconcat(OpcodeStr, "ps\t{$src, $dst|$dst, $src}"),
                [(set VR128:$dst, (v4f32 (OpNode VR128:$src)))], itins.rr>,
            Sched<[itins.Sched]>;
  def PSm : PSI<opc, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                !strconcat(OpcodeStr, "ps\t{$src, $dst|$dst, $src}"),
                [(set VR128:$dst, (OpNode (memopv4f32 addr:$src)))], itins.rm>,
            Sched<[itins.Sched.Folded]>;
}

/// sse1_fp_unop_p_int - SSE1 intrinsics unops in packed forms.
multiclass sse1_fp_unop_p_int<bits<8> opc, string OpcodeStr,
                              Intrinsic V4F32Int, Intrinsic V8F32Int,
                              OpndItins itins> {
let Predicates = [HasAVX] in {
  def V#NAME#PSr_Int : PSI<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                           !strconcat("v", OpcodeStr,
                                      "ps\t{$src, $dst|$dst, $src}"),
                           [(set VR128:$dst, (V4F32Int VR128:$src))],
                           itins.rr>, VEX, Sched<[itins.Sched]>;
  def V#NAME#PSm_Int : PSI<opc, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                          !strconcat("v", OpcodeStr,
                          "ps\t{$src, $dst|$dst, $src}"),
                          [(set VR128:$dst, (V4F32Int (loadv4f32 addr:$src)))],
                          itins.rm>, VEX, Sched<[itins.Sched.Folded]>;
  def V#NAME#PSYr_Int : PSI<opc, MRMSrcReg, (outs VR256:$dst), (ins VR256:$src),
                            !strconcat("v", OpcodeStr,
                                       "ps\t{$src, $dst|$dst, $src}"),
                            [(set VR256:$dst, (V8F32Int VR256:$src))],
                            itins.rr>, VEX, VEX_L, Sched<[itins.Sched]>;
  def V#NAME#PSYm_Int : PSI<opc, MRMSrcMem, (outs VR256:$dst),
                          (ins f256mem:$src),
                          !strconcat("v", OpcodeStr,
                                    "ps\t{$src, $dst|$dst, $src}"),
                          [(set VR256:$dst, (V8F32Int (loadv8f32 addr:$src)))],
                          itins.rm>, VEX, VEX_L, Sched<[itins.Sched.Folded]>;
}

  def PSr_Int : PSI<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                    !strconcat(OpcodeStr, "ps\t{$src, $dst|$dst, $src}"),
                    [(set VR128:$dst, (V4F32Int VR128:$src))],
                    itins.rr>, Sched<[itins.Sched]>;
  def PSm_Int : PSI<opc, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                    !strconcat(OpcodeStr, "ps\t{$src, $dst|$dst, $src}"),
                    [(set VR128:$dst, (V4F32Int (memopv4f32 addr:$src)))],
                    itins.rm>, Sched<[itins.Sched.Folded]>;
}

/// sse2_fp_unop_s - SSE2 unops in scalar form.
multiclass sse2_fp_unop_s<bits<8> opc, string OpcodeStr,
                          SDNode OpNode, Intrinsic F64Int, OpndItins itins> {
let Predicates = [HasAVX], hasSideEffects = 0 in {
  def V#NAME#SDr : SDI<opc, MRMSrcReg, (outs FR64:$dst),
                      (ins FR64:$src1, FR64:$src2),
                      !strconcat("v", OpcodeStr,
                                 "sd\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                      []>, VEX_4V, VEX_LIG, Sched<[itins.Sched]>;
  let mayLoad = 1 in {
  def V#NAME#SDm : SDI<opc, MRMSrcMem, (outs FR64:$dst),
                      (ins FR64:$src1,f64mem:$src2),
                      !strconcat("v", OpcodeStr,
                                 "sd\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                      []>, VEX_4V, VEX_LIG,
                   Sched<[itins.Sched.Folded, ReadAfterLd]>;
  def V#NAME#SDm_Int : SDI<opc, MRMSrcMem, (outs VR128:$dst),
                      (ins VR128:$src1, sdmem:$src2),
                      !strconcat("v", OpcodeStr,
                                 "sd\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                      []>, VEX_4V, VEX_LIG,
                      Sched<[itins.Sched.Folded, ReadAfterLd]>;
  }
}

  def SDr : SDI<opc, MRMSrcReg, (outs FR64:$dst), (ins FR64:$src),
                !strconcat(OpcodeStr, "sd\t{$src, $dst|$dst, $src}"),
                [(set FR64:$dst, (OpNode FR64:$src))], itins.rr>,
            Sched<[itins.Sched]>;
  // See the comments in sse1_fp_unop_s for why this is OptForSize.
  def SDm : I<opc, MRMSrcMem, (outs FR64:$dst), (ins f64mem:$src),
                !strconcat(OpcodeStr, "sd\t{$src, $dst|$dst, $src}"),
                [(set FR64:$dst, (OpNode (load addr:$src)))], itins.rm>, XD,
            Requires<[UseSSE2, OptForSize]>, Sched<[itins.Sched.Folded]>;
  def SDr_Int : SDI<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                    !strconcat(OpcodeStr, "sd\t{$src, $dst|$dst, $src}"),
                    [(set VR128:$dst, (F64Int VR128:$src))], itins.rr>,
                Sched<[itins.Sched]>;
  def SDm_Int : SDI<opc, MRMSrcMem, (outs VR128:$dst), (ins sdmem:$src),
                    !strconcat(OpcodeStr, "sd\t{$src, $dst|$dst, $src}"),
                    [(set VR128:$dst, (F64Int sse_load_f64:$src))], itins.rm>,
                Sched<[itins.Sched.Folded]>;
}

/// sse2_fp_unop_p - SSE2 unops in vector forms.
multiclass sse2_fp_unop_p<bits<8> opc, string OpcodeStr,
                          SDNode OpNode, OpndItins itins> {
let Predicates = [HasAVX] in {
  def V#NAME#PDr : PDI<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                       !strconcat("v", OpcodeStr,
                                  "pd\t{$src, $dst|$dst, $src}"),
                       [(set VR128:$dst, (v2f64 (OpNode VR128:$src)))],
                       itins.rr>, VEX, Sched<[itins.Sched]>;
  def V#NAME#PDm : PDI<opc, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                       !strconcat("v", OpcodeStr,
                                  "pd\t{$src, $dst|$dst, $src}"),
                       [(set VR128:$dst, (OpNode (loadv2f64 addr:$src)))],
                       itins.rm>, VEX, Sched<[itins.Sched.Folded]>;
  def V#NAME#PDYr : PDI<opc, MRMSrcReg, (outs VR256:$dst), (ins VR256:$src),
                        !strconcat("v", OpcodeStr,
                                   "pd\t{$src, $dst|$dst, $src}"),
                        [(set VR256:$dst, (v4f64 (OpNode VR256:$src)))],
                        itins.rr>, VEX, VEX_L, Sched<[itins.Sched]>;
  def V#NAME#PDYm : PDI<opc, MRMSrcMem, (outs VR256:$dst), (ins f256mem:$src),
                        !strconcat("v", OpcodeStr,
                                   "pd\t{$src, $dst|$dst, $src}"),
                        [(set VR256:$dst, (OpNode (loadv4f64 addr:$src)))],
                        itins.rm>, VEX, VEX_L, Sched<[itins.Sched.Folded]>;
}

  def PDr : PDI<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
              !strconcat(OpcodeStr, "pd\t{$src, $dst|$dst, $src}"),
              [(set VR128:$dst, (v2f64 (OpNode VR128:$src)))], itins.rr>,
            Sched<[itins.Sched]>;
  def PDm : PDI<opc, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
                !strconcat(OpcodeStr, "pd\t{$src, $dst|$dst, $src}"),
                [(set VR128:$dst, (OpNode (memopv2f64 addr:$src)))], itins.rm>,
            Sched<[itins.Sched.Folded]>;
}

// Square root.
defm SQRT  : sse1_fp_unop_s<0x51, "sqrt",  fsqrt, int_x86_sse_sqrt_ss,
                            SSE_SQRTSS>,
             sse1_fp_unop_p<0x51, "sqrt", fsqrt, SSE_SQRTPS>,
             sse2_fp_unop_s<0x51, "sqrt",  fsqrt, int_x86_sse2_sqrt_sd,
                            SSE_SQRTSD>,
             sse2_fp_unop_p<0x51, "sqrt", fsqrt, SSE_SQRTPD>;

// Reciprocal approximations. Note that these typically require refinement
// in order to obtain suitable precision.
defm RSQRT : sse1_fp_unop_rw<0x52, "rsqrt", X86frsqrt, SSE_SQRTSS>,
             sse1_fp_unop_p<0x52, "rsqrt", X86frsqrt, SSE_SQRTPS>,
             sse1_fp_unop_p_int<0x52, "rsqrt", int_x86_sse_rsqrt_ps,
                                int_x86_avx_rsqrt_ps_256, SSE_SQRTPS>;
defm RCP   : sse1_fp_unop_rw<0x53, "rcp", X86frcp, SSE_RCPS>,
             sse1_fp_unop_p<0x53, "rcp", X86frcp, SSE_RCPP>,
             sse1_fp_unop_p_int<0x53, "rcp", int_x86_sse_rcp_ps,
                                int_x86_avx_rcp_ps_256, SSE_RCPP>;

let Predicates = [UseAVX] in {
  def : Pat<(f32 (fsqrt FR32:$src)),
            (VSQRTSSr (f32 (IMPLICIT_DEF)), FR32:$src)>, Requires<[HasAVX]>;
  def : Pat<(f32 (fsqrt (load addr:$src))),
            (VSQRTSSm (f32 (IMPLICIT_DEF)), addr:$src)>,
            Requires<[HasAVX, OptForSize]>;
  def : Pat<(f64 (fsqrt FR64:$src)),
            (VSQRTSDr (f64 (IMPLICIT_DEF)), FR64:$src)>, Requires<[HasAVX]>;
  def : Pat<(f64 (fsqrt (load addr:$src))),
            (VSQRTSDm (f64 (IMPLICIT_DEF)), addr:$src)>,
            Requires<[HasAVX, OptForSize]>;

  def : Pat<(f32 (X86frsqrt FR32:$src)),
            (VRSQRTSSr (f32 (IMPLICIT_DEF)), FR32:$src)>, Requires<[HasAVX]>;
  def : Pat<(f32 (X86frsqrt (load addr:$src))),
            (VRSQRTSSm (f32 (IMPLICIT_DEF)), addr:$src)>,
            Requires<[HasAVX, OptForSize]>;

  def : Pat<(f32 (X86frcp FR32:$src)),
            (VRCPSSr (f32 (IMPLICIT_DEF)), FR32:$src)>, Requires<[HasAVX]>;
  def : Pat<(f32 (X86frcp (load addr:$src))),
            (VRCPSSm (f32 (IMPLICIT_DEF)), addr:$src)>,
            Requires<[HasAVX, OptForSize]>;
}
let Predicates = [UseAVX] in {
  def : Pat<(int_x86_sse_sqrt_ss VR128:$src),
            (COPY_TO_REGCLASS (VSQRTSSr (f32 (IMPLICIT_DEF)),
                                        (COPY_TO_REGCLASS VR128:$src, FR32)),
                              VR128)>;
  def : Pat<(int_x86_sse_sqrt_ss sse_load_f32:$src),
            (VSQRTSSm_Int (v4f32 (IMPLICIT_DEF)), sse_load_f32:$src)>;

  def : Pat<(int_x86_sse2_sqrt_sd VR128:$src),
            (COPY_TO_REGCLASS (VSQRTSDr (f64 (IMPLICIT_DEF)),
                                        (COPY_TO_REGCLASS VR128:$src, FR64)),
                              VR128)>;
  def : Pat<(int_x86_sse2_sqrt_sd sse_load_f64:$src),
            (VSQRTSDm_Int (v2f64 (IMPLICIT_DEF)), sse_load_f64:$src)>;
}

let Predicates = [HasAVX] in {
  def : Pat<(int_x86_sse_rsqrt_ss VR128:$src),
            (COPY_TO_REGCLASS (VRSQRTSSr (f32 (IMPLICIT_DEF)),
                                         (COPY_TO_REGCLASS VR128:$src, FR32)),
                              VR128)>;
  def : Pat<(int_x86_sse_rsqrt_ss sse_load_f32:$src),
            (VRSQRTSSm_Int (v4f32 (IMPLICIT_DEF)), sse_load_f32:$src)>;

  def : Pat<(int_x86_sse_rcp_ss VR128:$src),
            (COPY_TO_REGCLASS (VRCPSSr (f32 (IMPLICIT_DEF)),
                                       (COPY_TO_REGCLASS VR128:$src, FR32)),
                              VR128)>;
  def : Pat<(int_x86_sse_rcp_ss sse_load_f32:$src),
            (VRCPSSm_Int (v4f32 (IMPLICIT_DEF)), sse_load_f32:$src)>;
}

// Reciprocal approximations. Note that these typically require refinement
// in order to obtain suitable precision.
let Predicates = [UseSSE1] in {
  def : Pat<(int_x86_sse_rsqrt_ss VR128:$src),
            (RSQRTSSr_Int VR128:$src, VR128:$src)>;
  def : Pat<(int_x86_sse_rcp_ss VR128:$src),
            (RCPSSr_Int VR128:$src, VR128:$src)>;
}

// There is no f64 version of the reciprocal approximation instructions.

//===----------------------------------------------------------------------===//
// SSE 1 & 2 - Non-temporal stores
//===----------------------------------------------------------------------===//

let AddedComplexity = 400 in { // Prefer non-temporal versions
let SchedRW = [WriteStore] in {
def VMOVNTPSmr : VPSI<0x2B, MRMDestMem, (outs),
                     (ins f128mem:$dst, VR128:$src),
                     "movntps\t{$src, $dst|$dst, $src}",
                     [(alignednontemporalstore (v4f32 VR128:$src),
                                               addr:$dst)],
                                               IIC_SSE_MOVNT>, VEX;
def VMOVNTPDmr : VPDI<0x2B, MRMDestMem, (outs),
                     (ins f128mem:$dst, VR128:$src),
                     "movntpd\t{$src, $dst|$dst, $src}",
                     [(alignednontemporalstore (v2f64 VR128:$src),
                                               addr:$dst)],
                                               IIC_SSE_MOVNT>, VEX;

let ExeDomain = SSEPackedInt in
def VMOVNTDQmr    : VPDI<0xE7, MRMDestMem, (outs),
                         (ins f128mem:$dst, VR128:$src),
                         "movntdq\t{$src, $dst|$dst, $src}",
                         [(alignednontemporalstore (v2i64 VR128:$src),
                                                   addr:$dst)],
                                                   IIC_SSE_MOVNT>, VEX;

def VMOVNTPSYmr : VPSI<0x2B, MRMDestMem, (outs),
                     (ins f256mem:$dst, VR256:$src),
                     "movntps\t{$src, $dst|$dst, $src}",
                     [(alignednontemporalstore (v8f32 VR256:$src),
                                               addr:$dst)],
                                               IIC_SSE_MOVNT>, VEX, VEX_L;
def VMOVNTPDYmr : VPDI<0x2B, MRMDestMem, (outs),
                     (ins f256mem:$dst, VR256:$src),
                     "movntpd\t{$src, $dst|$dst, $src}",
                     [(alignednontemporalstore (v4f64 VR256:$src),
                                               addr:$dst)],
                                               IIC_SSE_MOVNT>, VEX, VEX_L;
let ExeDomain = SSEPackedInt in
def VMOVNTDQYmr : VPDI<0xE7, MRMDestMem, (outs),
                    (ins f256mem:$dst, VR256:$src),
                    "movntdq\t{$src, $dst|$dst, $src}",
                    [(alignednontemporalstore (v4i64 VR256:$src),
                                              addr:$dst)],
                                              IIC_SSE_MOVNT>, VEX, VEX_L;

def MOVNTPSmr : PSI<0x2B, MRMDestMem, (outs), (ins f128mem:$dst, VR128:$src),
                    "movntps\t{$src, $dst|$dst, $src}",
                    [(alignednontemporalstore (v4f32 VR128:$src), addr:$dst)],
                    IIC_SSE_MOVNT>;
def MOVNTPDmr : PDI<0x2B, MRMDestMem, (outs), (ins f128mem:$dst, VR128:$src),
                    "movntpd\t{$src, $dst|$dst, $src}",
                    [(alignednontemporalstore(v2f64 VR128:$src), addr:$dst)],
                    IIC_SSE_MOVNT>;

let ExeDomain = SSEPackedInt in
def MOVNTDQmr : PDI<0xE7, MRMDestMem, (outs), (ins f128mem:$dst, VR128:$src),
                    "movntdq\t{$src, $dst|$dst, $src}",
                    [(alignednontemporalstore (v2i64 VR128:$src), addr:$dst)],
                    IIC_SSE_MOVNT>;

// There is no AVX form for instructions below this point
def MOVNTImr : I<0xC3, MRMDestMem, (outs), (ins i32mem:$dst, GR32:$src),
                 "movnti{l}\t{$src, $dst|$dst, $src}",
                 [(nontemporalstore (i32 GR32:$src), addr:$dst)],
                 IIC_SSE_MOVNT>,
               TB, Requires<[HasSSE2]>;
def MOVNTI_64mr : RI<0xC3, MRMDestMem, (outs), (ins i64mem:$dst, GR64:$src),
                     "movnti{q}\t{$src, $dst|$dst, $src}",
                     [(nontemporalstore (i64 GR64:$src), addr:$dst)],
                     IIC_SSE_MOVNT>,
                  TB, Requires<[HasSSE2]>;
} // SchedRW = [WriteStore]

def : Pat<(alignednontemporalstore (v2i64 VR128:$src), addr:$dst),
          (VMOVNTDQmr addr:$dst, VR128:$src)>, Requires<[HasAVX]>;

def : Pat<(alignednontemporalstore (v2i64 VR128:$src), addr:$dst),
          (MOVNTDQmr addr:$dst, VR128:$src)>, Requires<[UseSSE2]>;
} // AddedComplexity

//===----------------------------------------------------------------------===//
// SSE 1 & 2 - Prefetch and memory fence
//===----------------------------------------------------------------------===//

// Prefetch intrinsic.
let Predicates = [HasSSE1], SchedRW = [WriteLoad] in {
def PREFETCHT0   : I<0x18, MRM1m, (outs), (ins i8mem:$src),
    "prefetcht0\t$src", [(prefetch addr:$src, imm, (i32 3), (i32 1))],
    IIC_SSE_PREFETCH>, TB;
def PREFETCHT1   : I<0x18, MRM2m, (outs), (ins i8mem:$src),
    "prefetcht1\t$src", [(prefetch addr:$src, imm, (i32 2), (i32 1))],
    IIC_SSE_PREFETCH>, TB;
def PREFETCHT2   : I<0x18, MRM3m, (outs), (ins i8mem:$src),
    "prefetcht2\t$src", [(prefetch addr:$src, imm, (i32 1), (i32 1))],
    IIC_SSE_PREFETCH>, TB;
def PREFETCHNTA  : I<0x18, MRM0m, (outs), (ins i8mem:$src),
    "prefetchnta\t$src", [(prefetch addr:$src, imm, (i32 0), (i32 1))],
    IIC_SSE_PREFETCH>, TB;
}

// FIXME: How should these memory instructions be modeled?
let SchedRW = [WriteLoad] in {
// Flush cache
def CLFLUSH : I<0xAE, MRM7m, (outs), (ins i8mem:$src),
               "clflush\t$src", [(int_x86_sse2_clflush addr:$src)],
               IIC_SSE_PREFETCH>, TB, Requires<[HasSSE2]>;

// Pause. This "instruction" is encoded as "rep; nop", so even though it
// was introduced with SSE2, it's backward compatible.
def PAUSE : I<0x90, RawFrm, (outs), (ins), "pause", [], IIC_SSE_PAUSE>, REP;

// Load, store, and memory fence
def SFENCE : I<0xAE, MRM_F8, (outs), (ins),
               "sfence", [(int_x86_sse_sfence)], IIC_SSE_SFENCE>,
               TB, Requires<[HasSSE1]>;
def LFENCE : I<0xAE, MRM_E8, (outs), (ins),
               "lfence", [(int_x86_sse2_lfence)], IIC_SSE_LFENCE>,
               TB, Requires<[HasSSE2]>;
def MFENCE : I<0xAE, MRM_F0, (outs), (ins),
               "mfence", [(int_x86_sse2_mfence)], IIC_SSE_MFENCE>,
               TB, Requires<[HasSSE2]>;
} // SchedRW

def : Pat<(X86SFence), (SFENCE)>;
def : Pat<(X86LFence), (LFENCE)>;
def : Pat<(X86MFence), (MFENCE)>;

//===----------------------------------------------------------------------===//
// SSE 1 & 2 - Load/Store XCSR register
//===----------------------------------------------------------------------===//

def VLDMXCSR : VPSI<0xAE, MRM2m, (outs), (ins i32mem:$src),
                  "ldmxcsr\t$src", [(int_x86_sse_ldmxcsr addr:$src)],
                  IIC_SSE_LDMXCSR>, VEX, Sched<[WriteLoad]>;
def VSTMXCSR : VPSI<0xAE, MRM3m, (outs), (ins i32mem:$dst),
                  "stmxcsr\t$dst", [(int_x86_sse_stmxcsr addr:$dst)],
                  IIC_SSE_STMXCSR>, VEX, Sched<[WriteStore]>;

def LDMXCSR : PSI<0xAE, MRM2m, (outs), (ins i32mem:$src),
                  "ldmxcsr\t$src", [(int_x86_sse_ldmxcsr addr:$src)],
                  IIC_SSE_LDMXCSR>, Sched<[WriteLoad]>;
def STMXCSR : PSI<0xAE, MRM3m, (outs), (ins i32mem:$dst),
                  "stmxcsr\t$dst", [(int_x86_sse_stmxcsr addr:$dst)],
                  IIC_SSE_STMXCSR>, Sched<[WriteStore]>;

//===---------------------------------------------------------------------===//
// SSE2 - Move Aligned/Unaligned Packed Integer Instructions
//===---------------------------------------------------------------------===//

let ExeDomain = SSEPackedInt in { // SSE integer instructions

let neverHasSideEffects = 1, SchedRW = [WriteMove] in {
def VMOVDQArr  : VPDI<0x6F, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                    "movdqa\t{$src, $dst|$dst, $src}", [], IIC_SSE_MOVA_P_RR>,
                    VEX;
def VMOVDQAYrr : VPDI<0x6F, MRMSrcReg, (outs VR256:$dst), (ins VR256:$src),
                    "movdqa\t{$src, $dst|$dst, $src}", [], IIC_SSE_MOVA_P_RR>,
                    VEX, VEX_L;
def VMOVDQUrr  : VSSI<0x6F, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                    "movdqu\t{$src, $dst|$dst, $src}", [], IIC_SSE_MOVU_P_RR>,
                    VEX;
def VMOVDQUYrr : VSSI<0x6F, MRMSrcReg, (outs VR256:$dst), (ins VR256:$src),
                    "movdqu\t{$src, $dst|$dst, $src}", [], IIC_SSE_MOVU_P_RR>,
                    VEX, VEX_L;
}

// For Disassembler
let isCodeGenOnly = 1, hasSideEffects = 0, SchedRW = [WriteMove] in {
def VMOVDQArr_REV  : VPDI<0x7F, MRMDestReg, (outs VR128:$dst), (ins VR128:$src),
                        "movdqa\t{$src, $dst|$dst, $src}", [],
                        IIC_SSE_MOVA_P_RR>,
                        VEX;
def VMOVDQAYrr_REV : VPDI<0x7F, MRMDestReg, (outs VR256:$dst), (ins VR256:$src),
                        "movdqa\t{$src, $dst|$dst, $src}", [],
                        IIC_SSE_MOVA_P_RR>, VEX, VEX_L;
def VMOVDQUrr_REV  : VSSI<0x7F, MRMDestReg, (outs VR128:$dst), (ins VR128:$src),
                        "movdqu\t{$src, $dst|$dst, $src}", [],
                        IIC_SSE_MOVU_P_RR>,
                        VEX;
def VMOVDQUYrr_REV : VSSI<0x7F, MRMDestReg, (outs VR256:$dst), (ins VR256:$src),
                        "movdqu\t{$src, $dst|$dst, $src}", [],
                        IIC_SSE_MOVU_P_RR>, VEX, VEX_L;
}

let canFoldAsLoad = 1, mayLoad = 1, isReMaterializable = 1,
    neverHasSideEffects = 1, SchedRW = [WriteLoad] in {
def VMOVDQArm  : VPDI<0x6F, MRMSrcMem, (outs VR128:$dst), (ins i128mem:$src),
                   "movdqa\t{$src, $dst|$dst, $src}", [], IIC_SSE_MOVA_P_RM>,
                   VEX;
def VMOVDQAYrm : VPDI<0x6F, MRMSrcMem, (outs VR256:$dst), (ins i256mem:$src),
                   "movdqa\t{$src, $dst|$dst, $src}", [], IIC_SSE_MOVA_P_RM>,
                   VEX, VEX_L;
let Predicates = [HasAVX] in {
  def VMOVDQUrm  : I<0x6F, MRMSrcMem, (outs VR128:$dst), (ins i128mem:$src),
                    "vmovdqu\t{$src, $dst|$dst, $src}",[], IIC_SSE_MOVU_P_RM>,
                    XS, VEX;
  def VMOVDQUYrm : I<0x6F, MRMSrcMem, (outs VR256:$dst), (ins i256mem:$src),
                    "vmovdqu\t{$src, $dst|$dst, $src}",[], IIC_SSE_MOVU_P_RM>,
                    XS, VEX, VEX_L;
}
}

let mayStore = 1, neverHasSideEffects = 1, SchedRW = [WriteStore] in {
def VMOVDQAmr  : VPDI<0x7F, MRMDestMem, (outs),
                     (ins i128mem:$dst, VR128:$src),
                     "movdqa\t{$src, $dst|$dst, $src}", [], IIC_SSE_MOVA_P_MR>,
                     VEX;
def VMOVDQAYmr : VPDI<0x7F, MRMDestMem, (outs),
                     (ins i256mem:$dst, VR256:$src),
                     "movdqa\t{$src, $dst|$dst, $src}", [], IIC_SSE_MOVA_P_MR>,
                     VEX, VEX_L;
let Predicates = [HasAVX] in {
def VMOVDQUmr  : I<0x7F, MRMDestMem, (outs), (ins i128mem:$dst, VR128:$src),
                  "vmovdqu\t{$src, $dst|$dst, $src}",[], IIC_SSE_MOVU_P_MR>,
                  XS, VEX;
def VMOVDQUYmr : I<0x7F, MRMDestMem, (outs), (ins i256mem:$dst, VR256:$src),
                  "vmovdqu\t{$src, $dst|$dst, $src}",[], IIC_SSE_MOVU_P_MR>,
                  XS, VEX, VEX_L;
}
}

let SchedRW = [WriteMove] in {
let neverHasSideEffects = 1 in
def MOVDQArr : PDI<0x6F, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                   "movdqa\t{$src, $dst|$dst, $src}", [], IIC_SSE_MOVA_P_RR>;

def MOVDQUrr :   I<0x6F, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                   "movdqu\t{$src, $dst|$dst, $src}",
                   [], IIC_SSE_MOVU_P_RR>, XS, Requires<[UseSSE2]>;

// For Disassembler
let isCodeGenOnly = 1, hasSideEffects = 0 in {
def MOVDQArr_REV : PDI<0x7F, MRMDestReg, (outs VR128:$dst), (ins VR128:$src),
                       "movdqa\t{$src, $dst|$dst, $src}", [],
                       IIC_SSE_MOVA_P_RR>;

def MOVDQUrr_REV :   I<0x7F, MRMDestReg, (outs VR128:$dst), (ins VR128:$src),
                       "movdqu\t{$src, $dst|$dst, $src}",
                       [], IIC_SSE_MOVU_P_RR>, XS, Requires<[UseSSE2]>;
}
} // SchedRW

let canFoldAsLoad = 1, mayLoad = 1, isReMaterializable = 1,
    neverHasSideEffects = 1, SchedRW = [WriteLoad] in {
def MOVDQArm : PDI<0x6F, MRMSrcMem, (outs VR128:$dst), (ins i128mem:$src),
                   "movdqa\t{$src, $dst|$dst, $src}",
                   [/*(set VR128:$dst, (alignedloadv2i64 addr:$src))*/],
                   IIC_SSE_MOVA_P_RM>;
def MOVDQUrm :   I<0x6F, MRMSrcMem, (outs VR128:$dst), (ins i128mem:$src),
                   "movdqu\t{$src, $dst|$dst, $src}",
                   [/*(set VR128:$dst, (loadv2i64 addr:$src))*/],
                   IIC_SSE_MOVU_P_RM>,
                 XS, Requires<[UseSSE2]>;
}

let mayStore = 1, neverHasSideEffects = 1, SchedRW = [WriteStore] in {
def MOVDQAmr : PDI<0x7F, MRMDestMem, (outs), (ins i128mem:$dst, VR128:$src),
                   "movdqa\t{$src, $dst|$dst, $src}",
                   [/*(alignedstore (v2i64 VR128:$src), addr:$dst)*/],
                   IIC_SSE_MOVA_P_MR>;
def MOVDQUmr :   I<0x7F, MRMDestMem, (outs), (ins i128mem:$dst, VR128:$src),
                   "movdqu\t{$src, $dst|$dst, $src}",
                   [/*(store (v2i64 VR128:$src), addr:$dst)*/],
                   IIC_SSE_MOVU_P_MR>,
                 XS, Requires<[UseSSE2]>;
}

} // ExeDomain = SSEPackedInt

let Predicates = [HasAVX] in {
  def : Pat<(int_x86_sse2_storeu_dq addr:$dst, VR128:$src),
            (VMOVDQUmr addr:$dst, VR128:$src)>;
  def : Pat<(int_x86_avx_storeu_dq_256 addr:$dst, VR256:$src),
            (VMOVDQUYmr addr:$dst, VR256:$src)>;
}
let Predicates = [UseSSE2] in
def : Pat<(int_x86_sse2_storeu_dq addr:$dst, VR128:$src),
          (MOVDQUmr addr:$dst, VR128:$src)>;

//===---------------------------------------------------------------------===//
// SSE2 - Packed Integer Arithmetic Instructions
//===---------------------------------------------------------------------===//

let Sched = WriteVecIMul in
def SSE_PMADD : OpndItins<
  IIC_SSE_PMADD, IIC_SSE_PMADD
>;

let ExeDomain = SSEPackedInt in { // SSE integer instructions

multiclass PDI_binop_rm_int<bits<8> opc, string OpcodeStr, Intrinsic IntId,
                            RegisterClass RC, PatFrag memop_frag,
                            X86MemOperand x86memop,
                            OpndItins itins,
                            bit IsCommutable = 0,
                            bit Is2Addr = 1> {
  let isCommutable = IsCommutable in
  def rr : PDI<opc, MRMSrcReg, (outs RC:$dst),
       (ins RC:$src1, RC:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set RC:$dst, (IntId RC:$src1, RC:$src2))], itins.rr>,
      Sched<[itins.Sched]>;
  def rm : PDI<opc, MRMSrcMem, (outs RC:$dst),
       (ins RC:$src1, x86memop:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set RC:$dst, (IntId RC:$src1, (bitconvert (memop_frag addr:$src2))))],
       itins.rm>, Sched<[itins.Sched.Folded, ReadAfterLd]>;
}

multiclass PDI_binop_all_int<bits<8> opc, string OpcodeStr, Intrinsic IntId128,
                             Intrinsic IntId256, OpndItins itins,
                             bit IsCommutable = 0> {
let Predicates = [HasAVX] in
  defm V#NAME : PDI_binop_rm_int<opc, !strconcat("v", OpcodeStr), IntId128,
                                 VR128, loadv2i64, i128mem, itins,
                                 IsCommutable, 0>, VEX_4V;

let Constraints = "$src1 = $dst" in
  defm NAME : PDI_binop_rm_int<opc, OpcodeStr, IntId128, VR128, memopv2i64,
                               i128mem, itins, IsCommutable, 1>;

let Predicates = [HasAVX2] in
  defm V#NAME#Y : PDI_binop_rm_int<opc, !strconcat("v", OpcodeStr), IntId256,
                                   VR256, loadv4i64, i256mem, itins,
                                   IsCommutable, 0>, VEX_4V, VEX_L;
}

multiclass PDI_binop_rmi<bits<8> opc, bits<8> opc2, Format ImmForm,
                         string OpcodeStr, SDNode OpNode,
                         SDNode OpNode2, RegisterClass RC,
                         ValueType DstVT, ValueType SrcVT, PatFrag bc_frag,
                         ShiftOpndItins itins,
                         bit Is2Addr = 1> {
  // src2 is always 128-bit
  def rr : PDI<opc, MRMSrcReg, (outs RC:$dst),
       (ins RC:$src1, VR128:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set RC:$dst, (DstVT (OpNode RC:$src1, (SrcVT VR128:$src2))))],
        itins.rr>, Sched<[WriteVecShift]>;
  def rm : PDI<opc, MRMSrcMem, (outs RC:$dst),
       (ins RC:$src1, i128mem:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set RC:$dst, (DstVT (OpNode RC:$src1,
                       (bc_frag (memopv2i64 addr:$src2)))))], itins.rm>,
      Sched<[WriteVecShiftLd, ReadAfterLd]>;
  def ri : PDIi8<opc2, ImmForm, (outs RC:$dst),
       (ins RC:$src1, i8imm:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set RC:$dst, (DstVT (OpNode2 RC:$src1, (i8 imm:$src2))))], itins.ri>,
       Sched<[WriteVecShift]>;
}

/// PDI_binop_rm2 - Simple SSE2 binary operator with different src and dst types
multiclass PDI_binop_rm2<bits<8> opc, string OpcodeStr, SDNode OpNode,
                         ValueType DstVT, ValueType SrcVT, RegisterClass RC,
                         PatFrag memop_frag, X86MemOperand x86memop,
                         OpndItins itins,
                         bit IsCommutable = 0, bit Is2Addr = 1> {
  let isCommutable = IsCommutable in
  def rr : PDI<opc, MRMSrcReg, (outs RC:$dst),
       (ins RC:$src1, RC:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set RC:$dst, (DstVT (OpNode (SrcVT RC:$src1), RC:$src2)))]>,
       Sched<[itins.Sched]>;
  def rm : PDI<opc, MRMSrcMem, (outs RC:$dst),
       (ins RC:$src1, x86memop:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set RC:$dst, (DstVT (OpNode (SrcVT RC:$src1),
                                     (bitconvert (memop_frag addr:$src2)))))]>,
       Sched<[itins.Sched.Folded, ReadAfterLd]>;
}
} // ExeDomain = SSEPackedInt

defm PADDB   : PDI_binop_all<0xFC, "paddb", add, v16i8, v32i8,
                             SSE_INTALU_ITINS_P, 1>;
defm PADDW   : PDI_binop_all<0xFD, "paddw", add, v8i16, v16i16,
                             SSE_INTALU_ITINS_P, 1>;
defm PADDD   : PDI_binop_all<0xFE, "paddd", add, v4i32, v8i32,
                             SSE_INTALU_ITINS_P, 1>;
defm PADDQ   : PDI_binop_all<0xD4, "paddq", add, v2i64, v4i64,
                             SSE_INTALUQ_ITINS_P, 1>;
defm PMULLW  : PDI_binop_all<0xD5, "pmullw", mul, v8i16, v16i16,
                             SSE_INTMUL_ITINS_P, 1>;
defm PSUBB   : PDI_binop_all<0xF8, "psubb", sub, v16i8, v32i8,
                             SSE_INTALU_ITINS_P, 0>;
defm PSUBW   : PDI_binop_all<0xF9, "psubw", sub, v8i16, v16i16,
                             SSE_INTALU_ITINS_P, 0>;
defm PSUBD   : PDI_binop_all<0xFA, "psubd", sub, v4i32, v8i32,
                             SSE_INTALU_ITINS_P, 0>;
defm PSUBQ   : PDI_binop_all<0xFB, "psubq", sub, v2i64, v4i64,
                             SSE_INTALUQ_ITINS_P, 0>;
defm PSUBUSB : PDI_binop_all<0xD8, "psubusb", X86subus, v16i8, v32i8,
                             SSE_INTALU_ITINS_P, 0>;
defm PSUBUSW : PDI_binop_all<0xD9, "psubusw", X86subus, v8i16, v16i16,
                             SSE_INTALU_ITINS_P, 0>;
defm PMINUB  : PDI_binop_all<0xDA, "pminub", X86umin, v16i8, v32i8,
                             SSE_INTALU_ITINS_P, 1>;
defm PMINSW  : PDI_binop_all<0xEA, "pminsw", X86smin, v8i16, v16i16,
                             SSE_INTALU_ITINS_P, 1>;
defm PMAXUB  : PDI_binop_all<0xDE, "pmaxub", X86umax, v16i8, v32i8,
                             SSE_INTALU_ITINS_P, 1>;
defm PMAXSW  : PDI_binop_all<0xEE, "pmaxsw", X86smax, v8i16, v16i16,
                             SSE_INTALU_ITINS_P, 1>;

// Intrinsic forms
defm PSUBSB  : PDI_binop_all_int<0xE8, "psubsb", int_x86_sse2_psubs_b,
                                 int_x86_avx2_psubs_b, SSE_INTALU_ITINS_P, 0>;
defm PSUBSW  : PDI_binop_all_int<0xE9, "psubsw" , int_x86_sse2_psubs_w,
                                 int_x86_avx2_psubs_w, SSE_INTALU_ITINS_P, 0>;
defm PADDSB  : PDI_binop_all_int<0xEC, "paddsb" , int_x86_sse2_padds_b,
                                 int_x86_avx2_padds_b, SSE_INTALU_ITINS_P, 1>;
defm PADDSW  : PDI_binop_all_int<0xED, "paddsw" , int_x86_sse2_padds_w,
                                 int_x86_avx2_padds_w, SSE_INTALU_ITINS_P, 1>;
defm PADDUSB : PDI_binop_all_int<0xDC, "paddusb", int_x86_sse2_paddus_b,
                                 int_x86_avx2_paddus_b, SSE_INTALU_ITINS_P, 1>;
defm PADDUSW : PDI_binop_all_int<0xDD, "paddusw", int_x86_sse2_paddus_w,
                                 int_x86_avx2_paddus_w, SSE_INTALU_ITINS_P, 1>;
defm PMULHUW : PDI_binop_all_int<0xE4, "pmulhuw", int_x86_sse2_pmulhu_w,
                                 int_x86_avx2_pmulhu_w, SSE_INTMUL_ITINS_P, 1>;
defm PMULHW  : PDI_binop_all_int<0xE5, "pmulhw" , int_x86_sse2_pmulh_w,
                                 int_x86_avx2_pmulh_w, SSE_INTMUL_ITINS_P, 1>;
defm PMADDWD : PDI_binop_all_int<0xF5, "pmaddwd", int_x86_sse2_pmadd_wd,
                                 int_x86_avx2_pmadd_wd, SSE_PMADD, 1>;
defm PAVGB   : PDI_binop_all_int<0xE0, "pavgb", int_x86_sse2_pavg_b,
                                 int_x86_avx2_pavg_b, SSE_INTALU_ITINS_P, 1>;
defm PAVGW   : PDI_binop_all_int<0xE3, "pavgw", int_x86_sse2_pavg_w,
                                 int_x86_avx2_pavg_w, SSE_INTALU_ITINS_P, 1>;
defm PSADBW  : PDI_binop_all_int<0xF6, "psadbw", int_x86_sse2_psad_bw,
                                 int_x86_avx2_psad_bw, SSE_PMADD, 1>;

let Predicates = [HasAVX] in
defm VPMULUDQ : PDI_binop_rm2<0xF4, "vpmuludq", X86pmuludq, v2i64, v4i32, VR128,
                              loadv2i64, i128mem, SSE_INTMUL_ITINS_P, 1, 0>,
                              VEX_4V;
let Predicates = [HasAVX2] in
defm VPMULUDQY : PDI_binop_rm2<0xF4, "vpmuludq", X86pmuludq, v4i64, v8i32,
                               VR256, loadv4i64, i256mem,
                               SSE_INTMUL_ITINS_P, 1, 0>, VEX_4V, VEX_L;
let Constraints = "$src1 = $dst" in
defm PMULUDQ : PDI_binop_rm2<0xF4, "pmuludq", X86pmuludq, v2i64, v4i32, VR128,
                             memopv2i64, i128mem, SSE_INTMUL_ITINS_P, 1>;

//===---------------------------------------------------------------------===//
// SSE2 - Packed Integer Logical Instructions
//===---------------------------------------------------------------------===//

let Predicates = [HasAVX] in {
defm VPSLLW : PDI_binop_rmi<0xF1, 0x71, MRM6r, "vpsllw", X86vshl, X86vshli,
                            VR128, v8i16, v8i16, bc_v8i16,
                            SSE_INTSHIFT_ITINS_P, 0>, VEX_4V;
defm VPSLLD : PDI_binop_rmi<0xF2, 0x72, MRM6r, "vpslld", X86vshl, X86vshli,
                            VR128, v4i32, v4i32, bc_v4i32,
                            SSE_INTSHIFT_ITINS_P, 0>, VEX_4V;
defm VPSLLQ : PDI_binop_rmi<0xF3, 0x73, MRM6r, "vpsllq", X86vshl, X86vshli,
                            VR128, v2i64, v2i64, bc_v2i64,
                            SSE_INTSHIFT_ITINS_P, 0>, VEX_4V;

defm VPSRLW : PDI_binop_rmi<0xD1, 0x71, MRM2r, "vpsrlw", X86vsrl, X86vsrli,
                            VR128, v8i16, v8i16, bc_v8i16,
                            SSE_INTSHIFT_ITINS_P, 0>, VEX_4V;
defm VPSRLD : PDI_binop_rmi<0xD2, 0x72, MRM2r, "vpsrld", X86vsrl, X86vsrli,
                            VR128, v4i32, v4i32, bc_v4i32,
                            SSE_INTSHIFT_ITINS_P, 0>, VEX_4V;
defm VPSRLQ : PDI_binop_rmi<0xD3, 0x73, MRM2r, "vpsrlq", X86vsrl, X86vsrli,
                            VR128, v2i64, v2i64, bc_v2i64,
                            SSE_INTSHIFT_ITINS_P, 0>, VEX_4V;

defm VPSRAW : PDI_binop_rmi<0xE1, 0x71, MRM4r, "vpsraw", X86vsra, X86vsrai,
                            VR128, v8i16, v8i16, bc_v8i16,
                            SSE_INTSHIFT_ITINS_P, 0>, VEX_4V;
defm VPSRAD : PDI_binop_rmi<0xE2, 0x72, MRM4r, "vpsrad", X86vsra, X86vsrai,
                            VR128, v4i32, v4i32, bc_v4i32,
                            SSE_INTSHIFT_ITINS_P, 0>, VEX_4V;

let ExeDomain = SSEPackedInt, SchedRW = [WriteVecShift] in {
  // 128-bit logical shifts.
  def VPSLLDQri : PDIi8<0x73, MRM7r,
                    (outs VR128:$dst), (ins VR128:$src1, i32i8imm:$src2),
                    "vpslldq\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                    [(set VR128:$dst,
                      (int_x86_sse2_psll_dq_bs VR128:$src1, imm:$src2))]>,
                    VEX_4V;
  def VPSRLDQri : PDIi8<0x73, MRM3r,
                    (outs VR128:$dst), (ins VR128:$src1, i32i8imm:$src2),
                    "vpsrldq\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                    [(set VR128:$dst,
                      (int_x86_sse2_psrl_dq_bs VR128:$src1, imm:$src2))]>,
                    VEX_4V;
  // PSRADQri doesn't exist in SSE[1-3].
}
} // Predicates = [HasAVX]

let Predicates = [HasAVX2] in {
defm VPSLLWY : PDI_binop_rmi<0xF1, 0x71, MRM6r, "vpsllw", X86vshl, X86vshli,
                             VR256, v16i16, v8i16, bc_v8i16,
                             SSE_INTSHIFT_ITINS_P, 0>, VEX_4V, VEX_L;
defm VPSLLDY : PDI_binop_rmi<0xF2, 0x72, MRM6r, "vpslld", X86vshl, X86vshli,
                             VR256, v8i32, v4i32, bc_v4i32,
                             SSE_INTSHIFT_ITINS_P, 0>, VEX_4V, VEX_L;
defm VPSLLQY : PDI_binop_rmi<0xF3, 0x73, MRM6r, "vpsllq", X86vshl, X86vshli,
                             VR256, v4i64, v2i64, bc_v2i64,
                             SSE_INTSHIFT_ITINS_P, 0>, VEX_4V, VEX_L;

defm VPSRLWY : PDI_binop_rmi<0xD1, 0x71, MRM2r, "vpsrlw", X86vsrl, X86vsrli,
                             VR256, v16i16, v8i16, bc_v8i16,
                             SSE_INTSHIFT_ITINS_P, 0>, VEX_4V, VEX_L;
defm VPSRLDY : PDI_binop_rmi<0xD2, 0x72, MRM2r, "vpsrld", X86vsrl, X86vsrli,
                             VR256, v8i32, v4i32, bc_v4i32,
                             SSE_INTSHIFT_ITINS_P, 0>, VEX_4V, VEX_L;
defm VPSRLQY : PDI_binop_rmi<0xD3, 0x73, MRM2r, "vpsrlq", X86vsrl, X86vsrli,
                             VR256, v4i64, v2i64, bc_v2i64,
                             SSE_INTSHIFT_ITINS_P, 0>, VEX_4V, VEX_L;

defm VPSRAWY : PDI_binop_rmi<0xE1, 0x71, MRM4r, "vpsraw", X86vsra, X86vsrai,
                             VR256, v16i16, v8i16, bc_v8i16,
                             SSE_INTSHIFT_ITINS_P, 0>, VEX_4V, VEX_L;
defm VPSRADY : PDI_binop_rmi<0xE2, 0x72, MRM4r, "vpsrad", X86vsra, X86vsrai,
                             VR256, v8i32, v4i32, bc_v4i32,
                             SSE_INTSHIFT_ITINS_P, 0>, VEX_4V, VEX_L;

let ExeDomain = SSEPackedInt, SchedRW = [WriteVecShift] in {
  // 256-bit logical shifts.
  def VPSLLDQYri : PDIi8<0x73, MRM7r,
                    (outs VR256:$dst), (ins VR256:$src1, i32i8imm:$src2),
                    "vpslldq\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                    [(set VR256:$dst,
                      (int_x86_avx2_psll_dq_bs VR256:$src1, imm:$src2))]>,
                    VEX_4V, VEX_L;
  def VPSRLDQYri : PDIi8<0x73, MRM3r,
                    (outs VR256:$dst), (ins VR256:$src1, i32i8imm:$src2),
                    "vpsrldq\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                    [(set VR256:$dst,
                      (int_x86_avx2_psrl_dq_bs VR256:$src1, imm:$src2))]>,
                    VEX_4V, VEX_L;
  // PSRADQYri doesn't exist in SSE[1-3].
}
} // Predicates = [HasAVX2]

let Constraints = "$src1 = $dst" in {
defm PSLLW : PDI_binop_rmi<0xF1, 0x71, MRM6r, "psllw", X86vshl, X86vshli,
                           VR128, v8i16, v8i16, bc_v8i16,
                           SSE_INTSHIFT_ITINS_P>;
defm PSLLD : PDI_binop_rmi<0xF2, 0x72, MRM6r, "pslld", X86vshl, X86vshli,
                           VR128, v4i32, v4i32, bc_v4i32,
                           SSE_INTSHIFT_ITINS_P>;
defm PSLLQ : PDI_binop_rmi<0xF3, 0x73, MRM6r, "psllq", X86vshl, X86vshli,
                           VR128, v2i64, v2i64, bc_v2i64,
                           SSE_INTSHIFT_ITINS_P>;

defm PSRLW : PDI_binop_rmi<0xD1, 0x71, MRM2r, "psrlw", X86vsrl, X86vsrli,
                           VR128, v8i16, v8i16, bc_v8i16,
                           SSE_INTSHIFT_ITINS_P>;
defm PSRLD : PDI_binop_rmi<0xD2, 0x72, MRM2r, "psrld", X86vsrl, X86vsrli,
                           VR128, v4i32, v4i32, bc_v4i32,
                           SSE_INTSHIFT_ITINS_P>;
defm PSRLQ : PDI_binop_rmi<0xD3, 0x73, MRM2r, "psrlq", X86vsrl, X86vsrli,
                           VR128, v2i64, v2i64, bc_v2i64,
                           SSE_INTSHIFT_ITINS_P>;

defm PSRAW : PDI_binop_rmi<0xE1, 0x71, MRM4r, "psraw", X86vsra, X86vsrai,
                           VR128, v8i16, v8i16, bc_v8i16,
                           SSE_INTSHIFT_ITINS_P>;
defm PSRAD : PDI_binop_rmi<0xE2, 0x72, MRM4r, "psrad", X86vsra, X86vsrai,
                           VR128, v4i32, v4i32, bc_v4i32,
                           SSE_INTSHIFT_ITINS_P>;

let ExeDomain = SSEPackedInt, SchedRW = [WriteVecShift] in {
  // 128-bit logical shifts.
  def PSLLDQri : PDIi8<0x73, MRM7r,
                       (outs VR128:$dst), (ins VR128:$src1, i32i8imm:$src2),
                       "pslldq\t{$src2, $dst|$dst, $src2}",
                       [(set VR128:$dst,
                         (int_x86_sse2_psll_dq_bs VR128:$src1, imm:$src2))],
                         IIC_SSE_INTSHDQ_P_RI>;
  def PSRLDQri : PDIi8<0x73, MRM3r,
                       (outs VR128:$dst), (ins VR128:$src1, i32i8imm:$src2),
                       "psrldq\t{$src2, $dst|$dst, $src2}",
                       [(set VR128:$dst,
                         (int_x86_sse2_psrl_dq_bs VR128:$src1, imm:$src2))],
                         IIC_SSE_INTSHDQ_P_RI>;
  // PSRADQri doesn't exist in SSE[1-3].
}
} // Constraints = "$src1 = $dst"

let Predicates = [HasAVX] in {
  def : Pat<(int_x86_sse2_psll_dq VR128:$src1, imm:$src2),
            (VPSLLDQri VR128:$src1, (BYTE_imm imm:$src2))>;
  def : Pat<(int_x86_sse2_psrl_dq VR128:$src1, imm:$src2),
            (VPSRLDQri VR128:$src1, (BYTE_imm imm:$src2))>;
  def : Pat<(v2f64 (X86fsrl VR128:$src1, i32immSExt8:$src2)),
            (VPSRLDQri VR128:$src1, (BYTE_imm imm:$src2))>;

  // Shift up / down and insert zero's.
  def : Pat<(v2i64 (X86vshldq VR128:$src, (i8 imm:$amt))),
            (VPSLLDQri VR128:$src, (BYTE_imm imm:$amt))>;
  def : Pat<(v2i64 (X86vshrdq VR128:$src, (i8 imm:$amt))),
            (VPSRLDQri VR128:$src, (BYTE_imm imm:$amt))>;
}

let Predicates = [HasAVX2] in {
  def : Pat<(int_x86_avx2_psll_dq VR256:$src1, imm:$src2),
            (VPSLLDQYri VR256:$src1, (BYTE_imm imm:$src2))>;
  def : Pat<(int_x86_avx2_psrl_dq VR256:$src1, imm:$src2),
            (VPSRLDQYri VR256:$src1, (BYTE_imm imm:$src2))>;
}

let Predicates = [UseSSE2] in {
  def : Pat<(int_x86_sse2_psll_dq VR128:$src1, imm:$src2),
            (PSLLDQri VR128:$src1, (BYTE_imm imm:$src2))>;
  def : Pat<(int_x86_sse2_psrl_dq VR128:$src1, imm:$src2),
            (PSRLDQri VR128:$src1, (BYTE_imm imm:$src2))>;
  def : Pat<(v2f64 (X86fsrl VR128:$src1, i32immSExt8:$src2)),
            (PSRLDQri VR128:$src1, (BYTE_imm imm:$src2))>;

  // Shift up / down and insert zero's.
  def : Pat<(v2i64 (X86vshldq VR128:$src, (i8 imm:$amt))),
            (PSLLDQri VR128:$src, (BYTE_imm imm:$amt))>;
  def : Pat<(v2i64 (X86vshrdq VR128:$src, (i8 imm:$amt))),
            (PSRLDQri VR128:$src, (BYTE_imm imm:$amt))>;
}

//===---------------------------------------------------------------------===//
// SSE2 - Packed Integer Comparison Instructions
//===---------------------------------------------------------------------===//

defm PCMPEQB : PDI_binop_all<0x74, "pcmpeqb", X86pcmpeq, v16i8, v32i8,
                             SSE_INTALU_ITINS_P, 1>;
defm PCMPEQW : PDI_binop_all<0x75, "pcmpeqw", X86pcmpeq, v8i16, v16i16,
                             SSE_INTALU_ITINS_P, 1>;
defm PCMPEQD : PDI_binop_all<0x76, "pcmpeqd", X86pcmpeq, v4i32, v8i32,
                             SSE_INTALU_ITINS_P, 1>;
defm PCMPGTB : PDI_binop_all<0x64, "pcmpgtb", X86pcmpgt, v16i8, v32i8,
                             SSE_INTALU_ITINS_P, 0>;
defm PCMPGTW : PDI_binop_all<0x65, "pcmpgtw", X86pcmpgt, v8i16, v16i16,
                             SSE_INTALU_ITINS_P, 0>;
defm PCMPGTD : PDI_binop_all<0x66, "pcmpgtd", X86pcmpgt, v4i32, v8i32,
                             SSE_INTALU_ITINS_P, 0>;

//===---------------------------------------------------------------------===//
// SSE2 - Packed Integer Pack Instructions
//===---------------------------------------------------------------------===//

defm PACKSSWB : PDI_binop_all_int<0x63, "packsswb", int_x86_sse2_packsswb_128,
                                  int_x86_avx2_packsswb, SSE_INTALU_ITINS_P, 0>;
defm PACKSSDW : PDI_binop_all_int<0x6B, "packssdw", int_x86_sse2_packssdw_128,
                                  int_x86_avx2_packssdw, SSE_INTALU_ITINS_P, 0>;
defm PACKUSWB : PDI_binop_all_int<0x67, "packuswb", int_x86_sse2_packuswb_128,
                                  int_x86_avx2_packuswb, SSE_INTALU_ITINS_P, 0>;

//===---------------------------------------------------------------------===//
// SSE2 - Packed Integer Shuffle Instructions
//===---------------------------------------------------------------------===//

let ExeDomain = SSEPackedInt in {
multiclass sse2_pshuffle<string OpcodeStr, ValueType vt128, ValueType vt256,
                         SDNode OpNode> {
let Predicates = [HasAVX] in {
  def V#NAME#ri : Ii8<0x70, MRMSrcReg, (outs VR128:$dst),
                      (ins VR128:$src1, i8imm:$src2),
                      !strconcat("v", OpcodeStr,
                                 "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                      [(set VR128:$dst,
                        (vt128 (OpNode VR128:$src1, (i8 imm:$src2))))],
                      IIC_SSE_PSHUF_RI>, VEX, Sched<[WriteShuffle]>;
  def V#NAME#mi : Ii8<0x70, MRMSrcMem, (outs VR128:$dst),
                      (ins i128mem:$src1, i8imm:$src2),
                      !strconcat("v", OpcodeStr,
                                 "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                     [(set VR128:$dst,
                       (vt128 (OpNode (bitconvert (loadv2i64 addr:$src1)),
                        (i8 imm:$src2))))], IIC_SSE_PSHUF_MI>, VEX,
                  Sched<[WriteShuffleLd]>;
}

let Predicates = [HasAVX2] in {
  def V#NAME#Yri : Ii8<0x70, MRMSrcReg, (outs VR256:$dst),
                       (ins VR256:$src1, i8imm:$src2),
                       !strconcat("v", OpcodeStr,
                                  "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                       [(set VR256:$dst,
                         (vt256 (OpNode VR256:$src1, (i8 imm:$src2))))],
                       IIC_SSE_PSHUF_RI>, VEX, VEX_L, Sched<[WriteShuffle]>;
  def V#NAME#Ymi : Ii8<0x70, MRMSrcMem, (outs VR256:$dst),
                       (ins i256mem:$src1, i8imm:$src2),
                       !strconcat("v", OpcodeStr,
                                  "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                      [(set VR256:$dst,
                        (vt256 (OpNode (bitconvert (loadv4i64 addr:$src1)),
                         (i8 imm:$src2))))], IIC_SSE_PSHUF_MI>, VEX, VEX_L,
                   Sched<[WriteShuffleLd]>;
}

let Predicates = [UseSSE2] in {
  def ri : Ii8<0x70, MRMSrcReg,
               (outs VR128:$dst), (ins VR128:$src1, i8imm:$src2),
               !strconcat(OpcodeStr,
                          "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                [(set VR128:$dst,
                  (vt128 (OpNode VR128:$src1, (i8 imm:$src2))))],
                IIC_SSE_PSHUF_RI>, Sched<[WriteShuffle]>;
  def mi : Ii8<0x70, MRMSrcMem,
               (outs VR128:$dst), (ins i128mem:$src1, i8imm:$src2),
               !strconcat(OpcodeStr,
                          "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                [(set VR128:$dst,
                  (vt128 (OpNode (bitconvert (memopv2i64 addr:$src1)),
                          (i8 imm:$src2))))], IIC_SSE_PSHUF_MI>,
           Sched<[WriteShuffleLd]>;
}
}
} // ExeDomain = SSEPackedInt

defm PSHUFD  : sse2_pshuffle<"pshufd", v4i32, v8i32, X86PShufd>, TB, OpSize;
defm PSHUFHW : sse2_pshuffle<"pshufhw", v8i16, v16i16, X86PShufhw>, XS;
defm PSHUFLW : sse2_pshuffle<"pshuflw", v8i16, v16i16, X86PShuflw>, XD;

let Predicates = [HasAVX] in {
  def : Pat<(v4f32 (X86PShufd (loadv4f32 addr:$src1), (i8 imm:$imm))),
            (VPSHUFDmi addr:$src1, imm:$imm)>;
  def : Pat<(v4f32 (X86PShufd VR128:$src1, (i8 imm:$imm))),
            (VPSHUFDri VR128:$src1, imm:$imm)>;
}

let Predicates = [UseSSE2] in {
  def : Pat<(v4f32 (X86PShufd (memopv4f32 addr:$src1), (i8 imm:$imm))),
            (PSHUFDmi addr:$src1, imm:$imm)>;
  def : Pat<(v4f32 (X86PShufd VR128:$src1, (i8 imm:$imm))),
            (PSHUFDri VR128:$src1, imm:$imm)>;
}

//===---------------------------------------------------------------------===//
// SSE2 - Packed Integer Unpack Instructions
//===---------------------------------------------------------------------===//

let ExeDomain = SSEPackedInt in {
multiclass sse2_unpack<bits<8> opc, string OpcodeStr, ValueType vt,
                       SDNode OpNode, PatFrag bc_frag, bit Is2Addr = 1> {
  def rr : PDI<opc, MRMSrcReg,
      (outs VR128:$dst), (ins VR128:$src1, VR128:$src2),
      !if(Is2Addr,
          !strconcat(OpcodeStr,"\t{$src2, $dst|$dst, $src2}"),
          !strconcat(OpcodeStr,"\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
      [(set VR128:$dst, (vt (OpNode VR128:$src1, VR128:$src2)))],
      IIC_SSE_UNPCK>, Sched<[WriteShuffle]>;
  def rm : PDI<opc, MRMSrcMem,
      (outs VR128:$dst), (ins VR128:$src1, i128mem:$src2),
      !if(Is2Addr,
          !strconcat(OpcodeStr,"\t{$src2, $dst|$dst, $src2}"),
          !strconcat(OpcodeStr,"\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
      [(set VR128:$dst, (OpNode VR128:$src1,
                                  (bc_frag (memopv2i64
                                               addr:$src2))))],
                                               IIC_SSE_UNPCK>,
      Sched<[WriteShuffleLd, ReadAfterLd]>;
}

multiclass sse2_unpack_y<bits<8> opc, string OpcodeStr, ValueType vt,
                         SDNode OpNode, PatFrag bc_frag> {
  def Yrr : PDI<opc, MRMSrcReg,
      (outs VR256:$dst), (ins VR256:$src1, VR256:$src2),
      !strconcat(OpcodeStr,"\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
      [(set VR256:$dst, (vt (OpNode VR256:$src1, VR256:$src2)))]>,
      Sched<[WriteShuffle]>;
  def Yrm : PDI<opc, MRMSrcMem,
      (outs VR256:$dst), (ins VR256:$src1, i256mem:$src2),
      !strconcat(OpcodeStr,"\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
      [(set VR256:$dst, (OpNode VR256:$src1,
                                  (bc_frag (memopv4i64 addr:$src2))))]>,
      Sched<[WriteShuffleLd, ReadAfterLd]>;
}

let Predicates = [HasAVX] in {
  defm VPUNPCKLBW  : sse2_unpack<0x60, "vpunpcklbw", v16i8, X86Unpckl,
                                 bc_v16i8, 0>, VEX_4V;
  defm VPUNPCKLWD  : sse2_unpack<0x61, "vpunpcklwd", v8i16, X86Unpckl,
                                 bc_v8i16, 0>, VEX_4V;
  defm VPUNPCKLDQ  : sse2_unpack<0x62, "vpunpckldq", v4i32, X86Unpckl,
                                 bc_v4i32, 0>, VEX_4V;
  defm VPUNPCKLQDQ : sse2_unpack<0x6C, "vpunpcklqdq", v2i64, X86Unpckl,
                                 bc_v2i64, 0>, VEX_4V;

  defm VPUNPCKHBW  : sse2_unpack<0x68, "vpunpckhbw", v16i8, X86Unpckh,
                                 bc_v16i8, 0>, VEX_4V;
  defm VPUNPCKHWD  : sse2_unpack<0x69, "vpunpckhwd", v8i16, X86Unpckh,
                                 bc_v8i16, 0>, VEX_4V;
  defm VPUNPCKHDQ  : sse2_unpack<0x6A, "vpunpckhdq", v4i32, X86Unpckh,
                                 bc_v4i32, 0>, VEX_4V;
  defm VPUNPCKHQDQ : sse2_unpack<0x6D, "vpunpckhqdq", v2i64, X86Unpckh,
                                 bc_v2i64, 0>, VEX_4V;
}

let Predicates = [HasAVX2] in {
  defm VPUNPCKLBW  : sse2_unpack_y<0x60, "vpunpcklbw", v32i8, X86Unpckl,
                                   bc_v32i8>, VEX_4V, VEX_L;
  defm VPUNPCKLWD  : sse2_unpack_y<0x61, "vpunpcklwd", v16i16, X86Unpckl,
                                   bc_v16i16>, VEX_4V, VEX_L;
  defm VPUNPCKLDQ  : sse2_unpack_y<0x62, "vpunpckldq", v8i32, X86Unpckl,
                                   bc_v8i32>, VEX_4V, VEX_L;
  defm VPUNPCKLQDQ : sse2_unpack_y<0x6C, "vpunpcklqdq", v4i64, X86Unpckl,
                                   bc_v4i64>, VEX_4V, VEX_L;

  defm VPUNPCKHBW  : sse2_unpack_y<0x68, "vpunpckhbw", v32i8, X86Unpckh,
                                   bc_v32i8>, VEX_4V, VEX_L;
  defm VPUNPCKHWD  : sse2_unpack_y<0x69, "vpunpckhwd", v16i16, X86Unpckh,
                                   bc_v16i16>, VEX_4V, VEX_L;
  defm VPUNPCKHDQ  : sse2_unpack_y<0x6A, "vpunpckhdq", v8i32, X86Unpckh,
                                   bc_v8i32>, VEX_4V, VEX_L;
  defm VPUNPCKHQDQ : sse2_unpack_y<0x6D, "vpunpckhqdq", v4i64, X86Unpckh,
                                   bc_v4i64>, VEX_4V, VEX_L;
}

let Constraints = "$src1 = $dst" in {
  defm PUNPCKLBW  : sse2_unpack<0x60, "punpcklbw", v16i8, X86Unpckl,
                                bc_v16i8>;
  defm PUNPCKLWD  : sse2_unpack<0x61, "punpcklwd", v8i16, X86Unpckl,
                                bc_v8i16>;
  defm PUNPCKLDQ  : sse2_unpack<0x62, "punpckldq", v4i32, X86Unpckl,
                                bc_v4i32>;
  defm PUNPCKLQDQ : sse2_unpack<0x6C, "punpcklqdq", v2i64, X86Unpckl,
                                bc_v2i64>;

  defm PUNPCKHBW  : sse2_unpack<0x68, "punpckhbw", v16i8, X86Unpckh,
                                bc_v16i8>;
  defm PUNPCKHWD  : sse2_unpack<0x69, "punpckhwd", v8i16, X86Unpckh,
                                bc_v8i16>;
  defm PUNPCKHDQ  : sse2_unpack<0x6A, "punpckhdq", v4i32, X86Unpckh,
                                bc_v4i32>;
  defm PUNPCKHQDQ : sse2_unpack<0x6D, "punpckhqdq", v2i64, X86Unpckh,
                                bc_v2i64>;
}
} // ExeDomain = SSEPackedInt

//===---------------------------------------------------------------------===//
// SSE2 - Packed Integer Extract and Insert
//===---------------------------------------------------------------------===//

let ExeDomain = SSEPackedInt in {
multiclass sse2_pinsrw<bit Is2Addr = 1> {
  def rri : Ii8<0xC4, MRMSrcReg,
       (outs VR128:$dst), (ins VR128:$src1,
        GR32orGR64:$src2, i32i8imm:$src3),
       !if(Is2Addr,
           "pinsrw\t{$src3, $src2, $dst|$dst, $src2, $src3}",
           "vpinsrw\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
       [(set VR128:$dst,
         (X86pinsrw VR128:$src1, GR32orGR64:$src2, imm:$src3))],
       IIC_SSE_PINSRW>, Sched<[WriteShuffle]>;
  def rmi : Ii8<0xC4, MRMSrcMem,
                       (outs VR128:$dst), (ins VR128:$src1,
                        i16mem:$src2, i32i8imm:$src3),
       !if(Is2Addr,
           "pinsrw\t{$src3, $src2, $dst|$dst, $src2, $src3}",
           "vpinsrw\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
       [(set VR128:$dst,
         (X86pinsrw VR128:$src1, (extloadi16 addr:$src2),
                    imm:$src3))], IIC_SSE_PINSRW>,
       Sched<[WriteShuffleLd, ReadAfterLd]>;
}

// Extract
let Predicates = [HasAVX] in
def VPEXTRWri : Ii8<0xC5, MRMSrcReg,
                    (outs GR32orGR64:$dst), (ins VR128:$src1, i32i8imm:$src2),
                    "vpextrw\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                    [(set GR32orGR64:$dst, (X86pextrw (v8i16 VR128:$src1),
                                            imm:$src2))]>, TB, OpSize, VEX,
                Sched<[WriteShuffle]>;
def PEXTRWri : PDIi8<0xC5, MRMSrcReg,
                    (outs GR32orGR64:$dst), (ins VR128:$src1, i32i8imm:$src2),
                    "pextrw\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                    [(set GR32orGR64:$dst, (X86pextrw (v8i16 VR128:$src1),
                                            imm:$src2))], IIC_SSE_PEXTRW>,
               Sched<[WriteShuffleLd, ReadAfterLd]>;

// Insert
let Predicates = [HasAVX] in
defm VPINSRW : sse2_pinsrw<0>, TB, OpSize, VEX_4V;

let Predicates = [UseSSE2], Constraints = "$src1 = $dst" in
defm PINSRW : sse2_pinsrw, TB, OpSize;

} // ExeDomain = SSEPackedInt

//===---------------------------------------------------------------------===//
// SSE2 - Packed Mask Creation
//===---------------------------------------------------------------------===//

let ExeDomain = SSEPackedInt, SchedRW = [WriteVecLogic] in {

def VPMOVMSKBrr  : VPDI<0xD7, MRMSrcReg, (outs GR32orGR64:$dst),
           (ins VR128:$src),
           "pmovmskb\t{$src, $dst|$dst, $src}",
           [(set GR32orGR64:$dst, (int_x86_sse2_pmovmskb_128 VR128:$src))],
           IIC_SSE_MOVMSK>, VEX;

let Predicates = [HasAVX2] in {
def VPMOVMSKBYrr  : VPDI<0xD7, MRMSrcReg, (outs GR32orGR64:$dst),
           (ins VR256:$src),
           "pmovmskb\t{$src, $dst|$dst, $src}",
           [(set GR32orGR64:$dst, (int_x86_avx2_pmovmskb VR256:$src))]>,
           VEX, VEX_L;
}

def PMOVMSKBrr : PDI<0xD7, MRMSrcReg, (outs GR32orGR64:$dst), (ins VR128:$src),
           "pmovmskb\t{$src, $dst|$dst, $src}",
           [(set GR32orGR64:$dst, (int_x86_sse2_pmovmskb_128 VR128:$src))],
           IIC_SSE_MOVMSK>;

} // ExeDomain = SSEPackedInt

//===---------------------------------------------------------------------===//
// SSE2 - Conditional Store
//===---------------------------------------------------------------------===//

let ExeDomain = SSEPackedInt, SchedRW = [WriteStore] in {

let Uses = [EDI], Predicates = [HasAVX,In32BitMode] in
def VMASKMOVDQU : VPDI<0xF7, MRMSrcReg, (outs),
           (ins VR128:$src, VR128:$mask),
           "maskmovdqu\t{$mask, $src|$src, $mask}",
           [(int_x86_sse2_maskmov_dqu VR128:$src, VR128:$mask, EDI)],
           IIC_SSE_MASKMOV>, VEX;
let Uses = [RDI], Predicates = [HasAVX,In64BitMode] in
def VMASKMOVDQU64 : VPDI<0xF7, MRMSrcReg, (outs),
           (ins VR128:$src, VR128:$mask),
           "maskmovdqu\t{$mask, $src|$src, $mask}",
           [(int_x86_sse2_maskmov_dqu VR128:$src, VR128:$mask, RDI)],
           IIC_SSE_MASKMOV>, VEX;

let Uses = [EDI], Predicates = [UseSSE2,In32BitMode] in
def MASKMOVDQU : PDI<0xF7, MRMSrcReg, (outs), (ins VR128:$src, VR128:$mask),
           "maskmovdqu\t{$mask, $src|$src, $mask}",
           [(int_x86_sse2_maskmov_dqu VR128:$src, VR128:$mask, EDI)],
           IIC_SSE_MASKMOV>;
let Uses = [RDI], Predicates = [UseSSE2,In64BitMode] in
def MASKMOVDQU64 : PDI<0xF7, MRMSrcReg, (outs), (ins VR128:$src, VR128:$mask),
           "maskmovdqu\t{$mask, $src|$src, $mask}",
           [(int_x86_sse2_maskmov_dqu VR128:$src, VR128:$mask, RDI)],
           IIC_SSE_MASKMOV>;

} // ExeDomain = SSEPackedInt

//===---------------------------------------------------------------------===//
// SSE2 - Move Doubleword
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//
// Move Int Doubleword to Packed Double Int
//
def VMOVDI2PDIrr : VS2I<0x6E, MRMSrcReg, (outs VR128:$dst), (ins GR32:$src),
                      "movd\t{$src, $dst|$dst, $src}",
                      [(set VR128:$dst,
                        (v4i32 (scalar_to_vector GR32:$src)))], IIC_SSE_MOVDQ>,
                        VEX, Sched<[WriteMove]>;
def VMOVDI2PDIrm : VS2I<0x6E, MRMSrcMem, (outs VR128:$dst), (ins i32mem:$src),
                      "movd\t{$src, $dst|$dst, $src}",
                      [(set VR128:$dst,
                        (v4i32 (scalar_to_vector (loadi32 addr:$src))))],
                        IIC_SSE_MOVDQ>,
                      VEX, Sched<[WriteLoad]>;
def VMOV64toPQIrr : VRS2I<0x6E, MRMSrcReg, (outs VR128:$dst), (ins GR64:$src),
                        "movq\t{$src, $dst|$dst, $src}",
                        [(set VR128:$dst,
                          (v2i64 (scalar_to_vector GR64:$src)))],
                          IIC_SSE_MOVDQ>, VEX, Sched<[WriteMove]>;
let isCodeGenOnly = 1 in
def VMOV64toSDrr : VRS2I<0x6E, MRMSrcReg, (outs FR64:$dst), (ins GR64:$src),
                       "movq\t{$src, $dst|$dst, $src}",
                       [(set FR64:$dst, (bitconvert GR64:$src))],
                       IIC_SSE_MOVDQ>, VEX, Sched<[WriteMove]>;

def MOVDI2PDIrr : S2I<0x6E, MRMSrcReg, (outs VR128:$dst), (ins GR32:$src),
                      "movd\t{$src, $dst|$dst, $src}",
                      [(set VR128:$dst,
                        (v4i32 (scalar_to_vector GR32:$src)))], IIC_SSE_MOVDQ>,
                  Sched<[WriteMove]>;
def MOVDI2PDIrm : S2I<0x6E, MRMSrcMem, (outs VR128:$dst), (ins i32mem:$src),
                      "movd\t{$src, $dst|$dst, $src}",
                      [(set VR128:$dst,
                        (v4i32 (scalar_to_vector (loadi32 addr:$src))))],
                        IIC_SSE_MOVDQ>, Sched<[WriteLoad]>;
def MOV64toPQIrr : RS2I<0x6E, MRMSrcReg, (outs VR128:$dst), (ins GR64:$src),
                        "mov{d|q}\t{$src, $dst|$dst, $src}",
                        [(set VR128:$dst,
                          (v2i64 (scalar_to_vector GR64:$src)))],
                          IIC_SSE_MOVDQ>, Sched<[WriteMove]>;
let isCodeGenOnly = 1 in
def MOV64toSDrr : RS2I<0x6E, MRMSrcReg, (outs FR64:$dst), (ins GR64:$src),
                       "mov{d|q}\t{$src, $dst|$dst, $src}",
                       [(set FR64:$dst, (bitconvert GR64:$src))],
                       IIC_SSE_MOVDQ>, Sched<[WriteMove]>;

//===---------------------------------------------------------------------===//
// Move Int Doubleword to Single Scalar
//
let isCodeGenOnly = 1 in {
  def VMOVDI2SSrr  : VS2I<0x6E, MRMSrcReg, (outs FR32:$dst), (ins GR32:$src),
                        "movd\t{$src, $dst|$dst, $src}",
                        [(set FR32:$dst, (bitconvert GR32:$src))],
                        IIC_SSE_MOVDQ>, VEX, Sched<[WriteMove]>;

  def VMOVDI2SSrm  : VS2I<0x6E, MRMSrcMem, (outs FR32:$dst), (ins i32mem:$src),
                        "movd\t{$src, $dst|$dst, $src}",
                        [(set FR32:$dst, (bitconvert (loadi32 addr:$src)))],
                        IIC_SSE_MOVDQ>,
                        VEX, Sched<[WriteLoad]>;
  def MOVDI2SSrr  : S2I<0x6E, MRMSrcReg, (outs FR32:$dst), (ins GR32:$src),
                        "movd\t{$src, $dst|$dst, $src}",
                        [(set FR32:$dst, (bitconvert GR32:$src))],
                        IIC_SSE_MOVDQ>, Sched<[WriteMove]>;

  def MOVDI2SSrm  : S2I<0x6E, MRMSrcMem, (outs FR32:$dst), (ins i32mem:$src),
                        "movd\t{$src, $dst|$dst, $src}",
                        [(set FR32:$dst, (bitconvert (loadi32 addr:$src)))],
                        IIC_SSE_MOVDQ>, Sched<[WriteLoad]>;
}

//===---------------------------------------------------------------------===//
// Move Packed Doubleword Int to Packed Double Int
//
def VMOVPDI2DIrr  : VS2I<0x7E, MRMDestReg, (outs GR32:$dst), (ins VR128:$src),
                       "movd\t{$src, $dst|$dst, $src}",
                       [(set GR32:$dst, (vector_extract (v4i32 VR128:$src),
                                        (iPTR 0)))], IIC_SSE_MOVD_ToGP>, VEX,
                    Sched<[WriteMove]>;
def VMOVPDI2DImr  : VS2I<0x7E, MRMDestMem, (outs),
                       (ins i32mem:$dst, VR128:$src),
                       "movd\t{$src, $dst|$dst, $src}",
                       [(store (i32 (vector_extract (v4i32 VR128:$src),
                                     (iPTR 0))), addr:$dst)], IIC_SSE_MOVDQ>,
                                     VEX, Sched<[WriteLoad]>;
def MOVPDI2DIrr  : S2I<0x7E, MRMDestReg, (outs GR32:$dst), (ins VR128:$src),
                       "movd\t{$src, $dst|$dst, $src}",
                       [(set GR32:$dst, (vector_extract (v4i32 VR128:$src),
                                        (iPTR 0)))], IIC_SSE_MOVD_ToGP>,
                   Sched<[WriteMove]>;
def MOVPDI2DImr  : S2I<0x7E, MRMDestMem, (outs), (ins i32mem:$dst, VR128:$src),
                       "movd\t{$src, $dst|$dst, $src}",
                       [(store (i32 (vector_extract (v4i32 VR128:$src),
                                     (iPTR 0))), addr:$dst)],
                                     IIC_SSE_MOVDQ>, Sched<[WriteLoad]>;

def : Pat<(v8i32 (X86Vinsert (v8i32 immAllZerosV), GR32:$src2, (iPTR 0))),
        (SUBREG_TO_REG (i32 0), (VMOVDI2PDIrr GR32:$src2), sub_xmm)>;

def : Pat<(v4i64 (X86Vinsert (bc_v4i64 (v8i32 immAllZerosV)), GR64:$src2, (iPTR 0))),
        (SUBREG_TO_REG (i32 0), (VMOV64toPQIrr GR64:$src2), sub_xmm)>;

def : Pat<(v8i32 (X86Vinsert undef, GR32:$src2, (iPTR 0))),
        (SUBREG_TO_REG (i32 0), (VMOVDI2PDIrr GR32:$src2), sub_xmm)>;

def : Pat<(v4i64 (X86Vinsert undef, GR64:$src2, (iPTR 0))),
        (SUBREG_TO_REG (i32 0), (VMOV64toPQIrr GR64:$src2), sub_xmm)>;

//===---------------------------------------------------------------------===//
// Move Packed Doubleword Int first element to Doubleword Int
//
let SchedRW = [WriteMove] in {
def VMOVPQIto64rr : VRS2I<0x7E, MRMDestReg, (outs GR64:$dst), (ins VR128:$src),
                          "movq\t{$src, $dst|$dst, $src}",
                          [(set GR64:$dst, (vector_extract (v2i64 VR128:$src),
                                                           (iPTR 0)))],
                                                           IIC_SSE_MOVD_ToGP>,
                      VEX;

def MOVPQIto64rr : RS2I<0x7E, MRMDestReg, (outs GR64:$dst), (ins VR128:$src),
                        "mov{d|q}\t{$src, $dst|$dst, $src}",
                        [(set GR64:$dst, (vector_extract (v2i64 VR128:$src),
                                                         (iPTR 0)))],
                                                         IIC_SSE_MOVD_ToGP>;
} //SchedRW

//===---------------------------------------------------------------------===//
// Bitcast FR64 <-> GR64
//
let isCodeGenOnly = 1 in {
  let Predicates = [UseAVX] in
  def VMOV64toSDrm : VS2SI<0x7E, MRMSrcMem, (outs FR64:$dst), (ins i64mem:$src),
                          "movq\t{$src, $dst|$dst, $src}",
                          [(set FR64:$dst, (bitconvert (loadi64 addr:$src)))]>,
                          VEX, Sched<[WriteLoad]>;
  def VMOVSDto64rr : VRS2I<0x7E, MRMDestReg, (outs GR64:$dst), (ins FR64:$src),
                           "movq\t{$src, $dst|$dst, $src}",
                           [(set GR64:$dst, (bitconvert FR64:$src))],
                           IIC_SSE_MOVDQ>, VEX, Sched<[WriteMove]>;
  def VMOVSDto64mr : VRS2I<0x7E, MRMDestMem, (outs), (ins i64mem:$dst, FR64:$src),
                           "movq\t{$src, $dst|$dst, $src}",
                           [(store (i64 (bitconvert FR64:$src)), addr:$dst)],
                           IIC_SSE_MOVDQ>, VEX, Sched<[WriteStore]>;

  def MOV64toSDrm : S2SI<0x7E, MRMSrcMem, (outs FR64:$dst), (ins i64mem:$src),
                         "movq\t{$src, $dst|$dst, $src}",
                         [(set FR64:$dst, (bitconvert (loadi64 addr:$src)))],
                         IIC_SSE_MOVDQ>, Sched<[WriteLoad]>;
  def MOVSDto64rr : RS2I<0x7E, MRMDestReg, (outs GR64:$dst), (ins FR64:$src),
                         "mov{d|q}\t{$src, $dst|$dst, $src}",
                         [(set GR64:$dst, (bitconvert FR64:$src))],
                         IIC_SSE_MOVD_ToGP>, Sched<[WriteMove]>;
  def MOVSDto64mr : RS2I<0x7E, MRMDestMem, (outs), (ins i64mem:$dst, FR64:$src),
                         "movq\t{$src, $dst|$dst, $src}",
                         [(store (i64 (bitconvert FR64:$src)), addr:$dst)],
                         IIC_SSE_MOVDQ>, Sched<[WriteStore]>;
}

//===---------------------------------------------------------------------===//
// Move Scalar Single to Double Int
//
let isCodeGenOnly = 1 in {
  def VMOVSS2DIrr  : VS2I<0x7E, MRMDestReg, (outs GR32:$dst), (ins FR32:$src),
                        "movd\t{$src, $dst|$dst, $src}",
                        [(set GR32:$dst, (bitconvert FR32:$src))],
                        IIC_SSE_MOVD_ToGP>, VEX, Sched<[WriteMove]>;
  def VMOVSS2DImr  : VS2I<0x7E, MRMDestMem, (outs), (ins i32mem:$dst, FR32:$src),
                        "movd\t{$src, $dst|$dst, $src}",
                        [(store (i32 (bitconvert FR32:$src)), addr:$dst)],
                        IIC_SSE_MOVDQ>, VEX, Sched<[WriteStore]>;
  def MOVSS2DIrr  : S2I<0x7E, MRMDestReg, (outs GR32:$dst), (ins FR32:$src),
                        "movd\t{$src, $dst|$dst, $src}",
                        [(set GR32:$dst, (bitconvert FR32:$src))],
                        IIC_SSE_MOVD_ToGP>, Sched<[WriteMove]>;
  def MOVSS2DImr  : S2I<0x7E, MRMDestMem, (outs), (ins i32mem:$dst, FR32:$src),
                        "movd\t{$src, $dst|$dst, $src}",
                        [(store (i32 (bitconvert FR32:$src)), addr:$dst)],
                        IIC_SSE_MOVDQ>, Sched<[WriteStore]>;
}

//===---------------------------------------------------------------------===//
// Patterns and instructions to describe movd/movq to XMM register zero-extends
//
let isCodeGenOnly = 1, SchedRW = [WriteMove] in {
let AddedComplexity = 15 in {
def VMOVZQI2PQIrr : VS2I<0x6E, MRMSrcReg, (outs VR128:$dst), (ins GR64:$src),
                       "movq\t{$src, $dst|$dst, $src}", // X86-64 only
                       [(set VR128:$dst, (v2i64 (X86vzmovl
                                      (v2i64 (scalar_to_vector GR64:$src)))))],
                                      IIC_SSE_MOVDQ>,
                                      VEX, VEX_W;
def MOVZQI2PQIrr : RS2I<0x6E, MRMSrcReg, (outs VR128:$dst), (ins GR64:$src),
                       "mov{d|q}\t{$src, $dst|$dst, $src}", // X86-64 only
                       [(set VR128:$dst, (v2i64 (X86vzmovl
                                      (v2i64 (scalar_to_vector GR64:$src)))))],
                                      IIC_SSE_MOVDQ>;
}
} // isCodeGenOnly, SchedRW

let Predicates = [UseAVX] in {
  let AddedComplexity = 15 in
    def : Pat<(v4i32 (X86vzmovl (v4i32 (scalar_to_vector GR32:$src)))),
              (VMOVDI2PDIrr GR32:$src)>;

  // AVX 128-bit movd/movq instruction write zeros in the high 128-bit part.
  let AddedComplexity = 20 in {
    def : Pat<(v4i32 (X86vzmovl (v4i32 (scalar_to_vector (loadi32 addr:$src))))),
              (VMOVDI2PDIrm addr:$src)>;
    def : Pat<(v4i32 (X86vzmovl (bc_v4i32 (loadv4f32 addr:$src)))),
              (VMOVDI2PDIrm addr:$src)>;
    def : Pat<(v4i32 (X86vzmovl (bc_v4i32 (loadv2i64 addr:$src)))),
              (VMOVDI2PDIrm addr:$src)>;
  }
  // Use regular 128-bit instructions to match 256-bit scalar_to_vec+zext.
  def : Pat<(v8i32 (X86vzmovl (insert_subvector undef,
                               (v4i32 (scalar_to_vector GR32:$src)),(iPTR 0)))),
            (SUBREG_TO_REG (i32 0), (VMOVDI2PDIrr GR32:$src), sub_xmm)>;
  def : Pat<(v4i64 (X86vzmovl (insert_subvector undef,
                               (v2i64 (scalar_to_vector GR64:$src)),(iPTR 0)))),
            (SUBREG_TO_REG (i64 0), (VMOVZQI2PQIrr GR64:$src), sub_xmm)>;
}

let Predicates = [UseSSE2] in {
  let AddedComplexity = 15 in
    def : Pat<(v4i32 (X86vzmovl (v4i32 (scalar_to_vector GR32:$src)))),
              (MOVDI2PDIrr GR32:$src)>;

  let AddedComplexity = 20 in {
    def : Pat<(v4i32 (X86vzmovl (v4i32 (scalar_to_vector (loadi32 addr:$src))))),
              (MOVDI2PDIrm addr:$src)>;
    def : Pat<(v4i32 (X86vzmovl (bc_v4i32 (loadv4f32 addr:$src)))),
              (MOVDI2PDIrm addr:$src)>;
    def : Pat<(v4i32 (X86vzmovl (bc_v4i32 (loadv2i64 addr:$src)))),
              (MOVDI2PDIrm addr:$src)>;
  }
}

// These are the correct encodings of the instructions so that we know how to
// read correct assembly, even though we continue to emit the wrong ones for
// compatibility with Darwin's buggy assembler.
def : InstAlias<"movq\t{$src, $dst|$dst, $src}",
                (MOV64toPQIrr VR128:$dst, GR64:$src), 0>;
def : InstAlias<"movq\t{$src, $dst|$dst, $src}",
                (MOVPQIto64rr GR64:$dst, VR128:$src), 0>;
// Allow "vmovd" but print "vmovq" since we don't need compatibility for AVX.
def : InstAlias<"vmovd\t{$src, $dst|$dst, $src}",
                (VMOV64toPQIrr VR128:$dst, GR64:$src), 0>;
def : InstAlias<"vmovd\t{$src, $dst|$dst, $src}",
                (VMOVPQIto64rr GR64:$dst, VR128:$src), 0>;

//===---------------------------------------------------------------------===//
// SSE2 - Move Quadword
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//
// Move Quadword Int to Packed Quadword Int
//

let SchedRW = [WriteLoad] in {
def VMOVQI2PQIrm : I<0x7E, MRMSrcMem, (outs VR128:$dst), (ins i64mem:$src),
                    "vmovq\t{$src, $dst|$dst, $src}",
                    [(set VR128:$dst,
                      (v2i64 (scalar_to_vector (loadi64 addr:$src))))]>, XS,
                    VEX, Requires<[UseAVX]>;
def MOVQI2PQIrm : I<0x7E, MRMSrcMem, (outs VR128:$dst), (ins i64mem:$src),
                    "movq\t{$src, $dst|$dst, $src}",
                    [(set VR128:$dst,
                      (v2i64 (scalar_to_vector (loadi64 addr:$src))))],
                      IIC_SSE_MOVDQ>, XS,
                    Requires<[UseSSE2]>; // SSE2 instruction with XS Prefix
} // SchedRW

//===---------------------------------------------------------------------===//
// Move Packed Quadword Int to Quadword Int
//
let SchedRW = [WriteStore] in {
def VMOVPQI2QImr : VS2I<0xD6, MRMDestMem, (outs), (ins i64mem:$dst, VR128:$src),
                      "movq\t{$src, $dst|$dst, $src}",
                      [(store (i64 (vector_extract (v2i64 VR128:$src),
                                    (iPTR 0))), addr:$dst)],
                                    IIC_SSE_MOVDQ>, VEX;
def MOVPQI2QImr : S2I<0xD6, MRMDestMem, (outs), (ins i64mem:$dst, VR128:$src),
                      "movq\t{$src, $dst|$dst, $src}",
                      [(store (i64 (vector_extract (v2i64 VR128:$src),
                                    (iPTR 0))), addr:$dst)],
                                    IIC_SSE_MOVDQ>;
} // SchedRW

//===---------------------------------------------------------------------===//
// Store / copy lower 64-bits of a XMM register.
//
def VMOVLQ128mr : VS2I<0xD6, MRMDestMem, (outs), (ins i64mem:$dst, VR128:$src),
                     "movq\t{$src, $dst|$dst, $src}",
                     [(int_x86_sse2_storel_dq addr:$dst, VR128:$src)]>, VEX,
                  Sched<[WriteStore]>;
def MOVLQ128mr : S2I<0xD6, MRMDestMem, (outs), (ins i64mem:$dst, VR128:$src),
                     "movq\t{$src, $dst|$dst, $src}",
                     [(int_x86_sse2_storel_dq addr:$dst, VR128:$src)],
                     IIC_SSE_MOVDQ>, Sched<[WriteStore]>;

let isCodeGenOnly = 1, AddedComplexity = 20 in {
def VMOVZQI2PQIrm : I<0x7E, MRMSrcMem, (outs VR128:$dst), (ins i64mem:$src),
                     "vmovq\t{$src, $dst|$dst, $src}",
                     [(set VR128:$dst,
                       (v2i64 (X86vzmovl (v2i64 (scalar_to_vector
                                                 (loadi64 addr:$src))))))],
                                                 IIC_SSE_MOVDQ>,
                     XS, VEX, Requires<[UseAVX]>, Sched<[WriteLoad]>;

def MOVZQI2PQIrm : I<0x7E, MRMSrcMem, (outs VR128:$dst), (ins i64mem:$src),
                     "movq\t{$src, $dst|$dst, $src}",
                     [(set VR128:$dst,
                       (v2i64 (X86vzmovl (v2i64 (scalar_to_vector
                                                 (loadi64 addr:$src))))))],
                                                 IIC_SSE_MOVDQ>,
                     XS, Requires<[UseSSE2]>, Sched<[WriteLoad]>;
}

let Predicates = [UseAVX], AddedComplexity = 20 in {
  def : Pat<(v2i64 (X86vzmovl (bc_v2i64 (loadv4f32 addr:$src)))),
            (VMOVZQI2PQIrm addr:$src)>;
  def : Pat<(v2i64 (X86vzload addr:$src)),
            (VMOVZQI2PQIrm addr:$src)>;
}

let Predicates = [UseSSE2], AddedComplexity = 20 in {
  def : Pat<(v2i64 (X86vzmovl (bc_v2i64 (loadv4f32 addr:$src)))),
            (MOVZQI2PQIrm addr:$src)>;
  def : Pat<(v2i64 (X86vzload addr:$src)), (MOVZQI2PQIrm addr:$src)>;
}

let Predicates = [HasAVX] in {
def : Pat<(v4i64 (alignedX86vzload addr:$src)),
          (SUBREG_TO_REG (i32 0), (VMOVAPSrm addr:$src), sub_xmm)>;
def : Pat<(v4i64 (X86vzload addr:$src)),
          (SUBREG_TO_REG (i32 0), (VMOVUPSrm addr:$src), sub_xmm)>;
}

//===---------------------------------------------------------------------===//
// Moving from XMM to XMM and clear upper 64 bits. Note, there is a bug in
// IA32 document. movq xmm1, xmm2 does clear the high bits.
//
let SchedRW = [WriteVecLogic] in {
let AddedComplexity = 15 in
def VMOVZPQILo2PQIrr : I<0x7E, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                        "vmovq\t{$src, $dst|$dst, $src}",
                    [(set VR128:$dst, (v2i64 (X86vzmovl (v2i64 VR128:$src))))],
                    IIC_SSE_MOVQ_RR>,
                      XS, VEX, Requires<[UseAVX]>;
let AddedComplexity = 15 in
def MOVZPQILo2PQIrr : I<0x7E, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                        "movq\t{$src, $dst|$dst, $src}",
                    [(set VR128:$dst, (v2i64 (X86vzmovl (v2i64 VR128:$src))))],
                    IIC_SSE_MOVQ_RR>,
                      XS, Requires<[UseSSE2]>;
} // SchedRW

let isCodeGenOnly = 1, SchedRW = [WriteVecLogicLd] in {
let AddedComplexity = 20 in
def VMOVZPQILo2PQIrm : I<0x7E, MRMSrcMem, (outs VR128:$dst), (ins i128mem:$src),
                        "vmovq\t{$src, $dst|$dst, $src}",
                    [(set VR128:$dst, (v2i64 (X86vzmovl
                                             (loadv2i64 addr:$src))))],
                                             IIC_SSE_MOVDQ>,
                      XS, VEX, Requires<[UseAVX]>;
let AddedComplexity = 20 in {
def MOVZPQILo2PQIrm : I<0x7E, MRMSrcMem, (outs VR128:$dst), (ins i128mem:$src),
                        "movq\t{$src, $dst|$dst, $src}",
                    [(set VR128:$dst, (v2i64 (X86vzmovl
                                             (loadv2i64 addr:$src))))],
                                             IIC_SSE_MOVDQ>,
                      XS, Requires<[UseSSE2]>;
}
} // isCodeGenOnly, SchedRW

let AddedComplexity = 20 in {
  let Predicates = [UseAVX] in {
    def : Pat<(v2f64 (X86vzmovl (v2f64 VR128:$src))),
              (VMOVZPQILo2PQIrr VR128:$src)>;
  }
  let Predicates = [UseSSE2] in {
    def : Pat<(v2f64 (X86vzmovl (v2f64 VR128:$src))),
              (MOVZPQILo2PQIrr VR128:$src)>;
  }
}

//===---------------------------------------------------------------------===//
// SSE3 - Replicate Single FP - MOVSHDUP and MOVSLDUP
//===---------------------------------------------------------------------===//
multiclass sse3_replicate_sfp<bits<8> op, SDNode OpNode, string OpcodeStr,
                              ValueType vt, RegisterClass RC, PatFrag mem_frag,
                              X86MemOperand x86memop> {
def rr : S3SI<op, MRMSrcReg, (outs RC:$dst), (ins RC:$src),
                    !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                      [(set RC:$dst, (vt (OpNode RC:$src)))],
                      IIC_SSE_MOV_LH>, Sched<[WriteShuffle]>;
def rm : S3SI<op, MRMSrcMem, (outs RC:$dst), (ins x86memop:$src),
                    !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                      [(set RC:$dst, (OpNode (mem_frag addr:$src)))],
                      IIC_SSE_MOV_LH>, Sched<[WriteShuffleLd]>;
}

let Predicates = [HasAVX] in {
  defm VMOVSHDUP  : sse3_replicate_sfp<0x16, X86Movshdup, "vmovshdup",
                                       v4f32, VR128, loadv4f32, f128mem>, VEX;
  defm VMOVSLDUP  : sse3_replicate_sfp<0x12, X86Movsldup, "vmovsldup",
                                       v4f32, VR128, loadv4f32, f128mem>, VEX;
  defm VMOVSHDUPY : sse3_replicate_sfp<0x16, X86Movshdup, "vmovshdup",
                                 v8f32, VR256, loadv8f32, f256mem>, VEX, VEX_L;
  defm VMOVSLDUPY : sse3_replicate_sfp<0x12, X86Movsldup, "vmovsldup",
                                 v8f32, VR256, loadv8f32, f256mem>, VEX, VEX_L;
}
defm MOVSHDUP : sse3_replicate_sfp<0x16, X86Movshdup, "movshdup", v4f32, VR128,
                                   memopv4f32, f128mem>;
defm MOVSLDUP : sse3_replicate_sfp<0x12, X86Movsldup, "movsldup", v4f32, VR128,
                                   memopv4f32, f128mem>;

let Predicates = [HasAVX] in {
  def : Pat<(v4i32 (X86Movshdup VR128:$src)),
            (VMOVSHDUPrr VR128:$src)>;
  def : Pat<(v4i32 (X86Movshdup (bc_v4i32 (loadv2i64 addr:$src)))),
            (VMOVSHDUPrm addr:$src)>;
  def : Pat<(v4i32 (X86Movsldup VR128:$src)),
            (VMOVSLDUPrr VR128:$src)>;
  def : Pat<(v4i32 (X86Movsldup (bc_v4i32 (loadv2i64 addr:$src)))),
            (VMOVSLDUPrm addr:$src)>;
  def : Pat<(v8i32 (X86Movshdup VR256:$src)),
            (VMOVSHDUPYrr VR256:$src)>;
  def : Pat<(v8i32 (X86Movshdup (bc_v8i32 (loadv4i64 addr:$src)))),
            (VMOVSHDUPYrm addr:$src)>;
  def : Pat<(v8i32 (X86Movsldup VR256:$src)),
            (VMOVSLDUPYrr VR256:$src)>;
  def : Pat<(v8i32 (X86Movsldup (bc_v8i32 (loadv4i64 addr:$src)))),
            (VMOVSLDUPYrm addr:$src)>;
}

let Predicates = [UseSSE3] in {
  def : Pat<(v4i32 (X86Movshdup VR128:$src)),
            (MOVSHDUPrr VR128:$src)>;
  def : Pat<(v4i32 (X86Movshdup (bc_v4i32 (memopv2i64 addr:$src)))),
            (MOVSHDUPrm addr:$src)>;
  def : Pat<(v4i32 (X86Movsldup VR128:$src)),
            (MOVSLDUPrr VR128:$src)>;
  def : Pat<(v4i32 (X86Movsldup (bc_v4i32 (memopv2i64 addr:$src)))),
            (MOVSLDUPrm addr:$src)>;
}

//===---------------------------------------------------------------------===//
// SSE3 - Replicate Double FP - MOVDDUP
//===---------------------------------------------------------------------===//

multiclass sse3_replicate_dfp<string OpcodeStr> {
let neverHasSideEffects = 1 in
def rr  : S3DI<0x12, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                    !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                    [], IIC_SSE_MOV_LH>, Sched<[WriteShuffle]>;
def rm  : S3DI<0x12, MRMSrcMem, (outs VR128:$dst), (ins f64mem:$src),
                    !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                    [(set VR128:$dst,
                      (v2f64 (X86Movddup
                              (scalar_to_vector (loadf64 addr:$src)))))],
                              IIC_SSE_MOV_LH>, Sched<[WriteShuffleLd]>;
}

// FIXME: Merge with above classe when there're patterns for the ymm version
multiclass sse3_replicate_dfp_y<string OpcodeStr> {
def rr  : S3DI<0x12, MRMSrcReg, (outs VR256:$dst), (ins VR256:$src),
                    !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                    [(set VR256:$dst, (v4f64 (X86Movddup VR256:$src)))]>,
                    Sched<[WriteShuffle]>;
def rm  : S3DI<0x12, MRMSrcMem, (outs VR256:$dst), (ins f256mem:$src),
                    !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                    [(set VR256:$dst,
                      (v4f64 (X86Movddup
                              (scalar_to_vector (loadf64 addr:$src)))))]>,
                    Sched<[WriteShuffleLd]>;
}

let Predicates = [HasAVX] in {
  defm VMOVDDUP  : sse3_replicate_dfp<"vmovddup">, VEX;
  defm VMOVDDUPY : sse3_replicate_dfp_y<"vmovddup">, VEX, VEX_L;
}

defm MOVDDUP : sse3_replicate_dfp<"movddup">;

let Predicates = [HasAVX] in {
  def : Pat<(X86Movddup (loadv2f64 addr:$src)),
            (VMOVDDUPrm addr:$src)>, Requires<[HasAVX]>;
  def : Pat<(X86Movddup (bc_v2f64 (loadv4f32 addr:$src))),
            (VMOVDDUPrm addr:$src)>, Requires<[HasAVX]>;
  def : Pat<(X86Movddup (bc_v2f64 (loadv2i64 addr:$src))),
            (VMOVDDUPrm addr:$src)>, Requires<[HasAVX]>;
  def : Pat<(X86Movddup (bc_v2f64
                             (v2i64 (scalar_to_vector (loadi64 addr:$src))))),
            (VMOVDDUPrm addr:$src)>, Requires<[HasAVX]>;

  // 256-bit version
  def : Pat<(X86Movddup (loadv4f64 addr:$src)),
            (VMOVDDUPYrm addr:$src)>;
  def : Pat<(X86Movddup (loadv4i64 addr:$src)),
            (VMOVDDUPYrm addr:$src)>;
  def : Pat<(X86Movddup (v4i64 (scalar_to_vector (loadi64 addr:$src)))),
            (VMOVDDUPYrm addr:$src)>;
  def : Pat<(X86Movddup (v4i64 VR256:$src)),
            (VMOVDDUPYrr VR256:$src)>;
}

let Predicates = [UseSSE3] in {
  def : Pat<(X86Movddup (memopv2f64 addr:$src)),
            (MOVDDUPrm addr:$src)>;
  def : Pat<(X86Movddup (bc_v2f64 (memopv4f32 addr:$src))),
            (MOVDDUPrm addr:$src)>;
  def : Pat<(X86Movddup (bc_v2f64 (memopv2i64 addr:$src))),
            (MOVDDUPrm addr:$src)>;
  def : Pat<(X86Movddup (bc_v2f64
                             (v2i64 (scalar_to_vector (loadi64 addr:$src))))),
            (MOVDDUPrm addr:$src)>;
}

//===---------------------------------------------------------------------===//
// SSE3 - Move Unaligned Integer
//===---------------------------------------------------------------------===//

let SchedRW = [WriteLoad] in {
let Predicates = [HasAVX] in {
  def VLDDQUrm : S3DI<0xF0, MRMSrcMem, (outs VR128:$dst), (ins i128mem:$src),
                   "vlddqu\t{$src, $dst|$dst, $src}",
                   [(set VR128:$dst, (int_x86_sse3_ldu_dq addr:$src))]>, VEX;
  def VLDDQUYrm : S3DI<0xF0, MRMSrcMem, (outs VR256:$dst), (ins i256mem:$src),
                   "vlddqu\t{$src, $dst|$dst, $src}",
                   [(set VR256:$dst, (int_x86_avx_ldu_dq_256 addr:$src))]>,
                   VEX, VEX_L;
}
def LDDQUrm : S3DI<0xF0, MRMSrcMem, (outs VR128:$dst), (ins i128mem:$src),
                   "lddqu\t{$src, $dst|$dst, $src}",
                   [(set VR128:$dst, (int_x86_sse3_ldu_dq addr:$src))],
                   IIC_SSE_LDDQU>;
}

//===---------------------------------------------------------------------===//
// SSE3 - Arithmetic
//===---------------------------------------------------------------------===//

multiclass sse3_addsub<Intrinsic Int, string OpcodeStr, RegisterClass RC,
                       X86MemOperand x86memop, OpndItins itins,
                       bit Is2Addr = 1> {
  def rr : I<0xD0, MRMSrcReg,
       (outs RC:$dst), (ins RC:$src1, RC:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set RC:$dst, (Int RC:$src1, RC:$src2))], itins.rr>,
       Sched<[itins.Sched]>;
  def rm : I<0xD0, MRMSrcMem,
       (outs RC:$dst), (ins RC:$src1, x86memop:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set RC:$dst, (Int RC:$src1, (memop addr:$src2)))], itins.rr>,
       Sched<[itins.Sched.Folded, ReadAfterLd]>;
}

let Predicates = [HasAVX] in {
  let ExeDomain = SSEPackedSingle in {
    defm VADDSUBPS : sse3_addsub<int_x86_sse3_addsub_ps, "vaddsubps", VR128,
                                 f128mem, SSE_ALU_F32P, 0>, TB, XD, VEX_4V;
    defm VADDSUBPSY : sse3_addsub<int_x86_avx_addsub_ps_256, "vaddsubps", VR256,
                               f256mem, SSE_ALU_F32P, 0>, TB, XD, VEX_4V, VEX_L;
  }
  let ExeDomain = SSEPackedDouble in {
    defm VADDSUBPD : sse3_addsub<int_x86_sse3_addsub_pd, "vaddsubpd", VR128,
                                 f128mem, SSE_ALU_F64P, 0>, TB, OpSize, VEX_4V;
    defm VADDSUBPDY : sse3_addsub<int_x86_avx_addsub_pd_256, "vaddsubpd", VR256,
                           f256mem, SSE_ALU_F64P, 0>, TB, OpSize, VEX_4V, VEX_L;
  }
}
let Constraints = "$src1 = $dst", Predicates = [UseSSE3] in {
  let ExeDomain = SSEPackedSingle in
  defm ADDSUBPS : sse3_addsub<int_x86_sse3_addsub_ps, "addsubps", VR128,
                              f128mem, SSE_ALU_F32P>, TB, XD;
  let ExeDomain = SSEPackedDouble in
  defm ADDSUBPD : sse3_addsub<int_x86_sse3_addsub_pd, "addsubpd", VR128,
                              f128mem, SSE_ALU_F64P>, TB, OpSize;
}

//===---------------------------------------------------------------------===//
// SSE3 Instructions
//===---------------------------------------------------------------------===//

// Horizontal ops
multiclass S3D_Int<bits<8> o, string OpcodeStr, ValueType vt, RegisterClass RC,
                   X86MemOperand x86memop, SDNode OpNode, bit Is2Addr = 1> {
  def rr : S3DI<o, MRMSrcReg, (outs RC:$dst), (ins RC:$src1, RC:$src2),
       !if(Is2Addr,
         !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
         !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
      [(set RC:$dst, (vt (OpNode RC:$src1, RC:$src2)))], IIC_SSE_HADDSUB_RR>,
      Sched<[WriteFAdd]>;

  def rm : S3DI<o, MRMSrcMem, (outs RC:$dst), (ins RC:$src1, x86memop:$src2),
       !if(Is2Addr,
         !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
         !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
      [(set RC:$dst, (vt (OpNode RC:$src1, (memop addr:$src2))))],
        IIC_SSE_HADDSUB_RM>, Sched<[WriteFAddLd, ReadAfterLd]>;
}
multiclass S3_Int<bits<8> o, string OpcodeStr, ValueType vt, RegisterClass RC,
                  X86MemOperand x86memop, SDNode OpNode, bit Is2Addr = 1> {
  def rr : S3I<o, MRMSrcReg, (outs RC:$dst), (ins RC:$src1, RC:$src2),
       !if(Is2Addr,
         !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
         !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
      [(set RC:$dst, (vt (OpNode RC:$src1, RC:$src2)))], IIC_SSE_HADDSUB_RR>,
      Sched<[WriteFAdd]>;

  def rm : S3I<o, MRMSrcMem, (outs RC:$dst), (ins RC:$src1, x86memop:$src2),
       !if(Is2Addr,
         !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
         !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
      [(set RC:$dst, (vt (OpNode RC:$src1, (memop addr:$src2))))],
        IIC_SSE_HADDSUB_RM>, Sched<[WriteFAddLd, ReadAfterLd]>;
}

let Predicates = [HasAVX] in {
  let ExeDomain = SSEPackedSingle in {
    defm VHADDPS  : S3D_Int<0x7C, "vhaddps", v4f32, VR128, f128mem,
                            X86fhadd, 0>, VEX_4V;
    defm VHSUBPS  : S3D_Int<0x7D, "vhsubps", v4f32, VR128, f128mem,
                            X86fhsub, 0>, VEX_4V;
    defm VHADDPSY : S3D_Int<0x7C, "vhaddps", v8f32, VR256, f256mem,
                            X86fhadd, 0>, VEX_4V, VEX_L;
    defm VHSUBPSY : S3D_Int<0x7D, "vhsubps", v8f32, VR256, f256mem,
                            X86fhsub, 0>, VEX_4V, VEX_L;
  }
  let ExeDomain = SSEPackedDouble in {
    defm VHADDPD  : S3_Int <0x7C, "vhaddpd", v2f64, VR128, f128mem,
                            X86fhadd, 0>, VEX_4V;
    defm VHSUBPD  : S3_Int <0x7D, "vhsubpd", v2f64, VR128, f128mem,
                            X86fhsub, 0>, VEX_4V;
    defm VHADDPDY : S3_Int <0x7C, "vhaddpd", v4f64, VR256, f256mem,
                            X86fhadd, 0>, VEX_4V, VEX_L;
    defm VHSUBPDY : S3_Int <0x7D, "vhsubpd", v4f64, VR256, f256mem,
                            X86fhsub, 0>, VEX_4V, VEX_L;
  }
}

let Constraints = "$src1 = $dst" in {
  let ExeDomain = SSEPackedSingle in {
    defm HADDPS : S3D_Int<0x7C, "haddps", v4f32, VR128, f128mem, X86fhadd>;
    defm HSUBPS : S3D_Int<0x7D, "hsubps", v4f32, VR128, f128mem, X86fhsub>;
  }
  let ExeDomain = SSEPackedDouble in {
    defm HADDPD : S3_Int<0x7C, "haddpd", v2f64, VR128, f128mem, X86fhadd>;
    defm HSUBPD : S3_Int<0x7D, "hsubpd", v2f64, VR128, f128mem, X86fhsub>;
  }
}

//===---------------------------------------------------------------------===//
// SSSE3 - Packed Absolute Instructions
//===---------------------------------------------------------------------===//


/// SS3I_unop_rm_int - Simple SSSE3 unary op whose type can be v*{i8,i16,i32}.
multiclass SS3I_unop_rm_int<bits<8> opc, string OpcodeStr,
                            Intrinsic IntId128> {
  def rr128 : SS38I<opc, MRMSrcReg, (outs VR128:$dst),
                    (ins VR128:$src),
                    !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                    [(set VR128:$dst, (IntId128 VR128:$src))], IIC_SSE_PABS_RR>,
                    OpSize, Sched<[WriteVecALU]>;

  def rm128 : SS38I<opc, MRMSrcMem, (outs VR128:$dst),
                    (ins i128mem:$src),
                    !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                    [(set VR128:$dst,
                      (IntId128
                       (bitconvert (memopv2i64 addr:$src))))], IIC_SSE_PABS_RM>,
                    OpSize, Sched<[WriteVecALULd]>;
}

/// SS3I_unop_rm_int_y - Simple SSSE3 unary op whose type can be v*{i8,i16,i32}.
multiclass SS3I_unop_rm_int_y<bits<8> opc, string OpcodeStr,
                              Intrinsic IntId256> {
  def rr256 : SS38I<opc, MRMSrcReg, (outs VR256:$dst),
                    (ins VR256:$src),
                    !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                    [(set VR256:$dst, (IntId256 VR256:$src))]>,
                    OpSize, Sched<[WriteVecALU]>;

  def rm256 : SS38I<opc, MRMSrcMem, (outs VR256:$dst),
                    (ins i256mem:$src),
                    !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                    [(set VR256:$dst,
                      (IntId256
                       (bitconvert (memopv4i64 addr:$src))))]>, OpSize,
                    Sched<[WriteVecALULd]>;
}

// Helper fragments to match sext vXi1 to vXiY.
def v16i1sextv16i8 : PatLeaf<(v16i8 (X86pcmpgt (bc_v16i8 (v4i32 immAllZerosV)),
                                               VR128:$src))>;
def v8i1sextv8i16  : PatLeaf<(v8i16 (X86vsrai VR128:$src, (i8 15)))>;
def v4i1sextv4i32  : PatLeaf<(v4i32 (X86vsrai VR128:$src, (i8 31)))>;
def v32i1sextv32i8 : PatLeaf<(v32i8 (X86pcmpgt (bc_v32i8 (v8i32 immAllZerosV)),
                                               VR256:$src))>;
def v16i1sextv16i16: PatLeaf<(v16i16 (X86vsrai VR256:$src, (i8 15)))>;
def v8i1sextv8i32  : PatLeaf<(v8i32 (X86vsrai VR256:$src, (i8 31)))>;

let Predicates = [HasAVX] in {
  defm VPABSB  : SS3I_unop_rm_int<0x1C, "vpabsb",
                                  int_x86_ssse3_pabs_b_128>, VEX;
  defm VPABSW  : SS3I_unop_rm_int<0x1D, "vpabsw",
                                  int_x86_ssse3_pabs_w_128>, VEX;
  defm VPABSD  : SS3I_unop_rm_int<0x1E, "vpabsd",
                                  int_x86_ssse3_pabs_d_128>, VEX;

  def : Pat<(xor
            (bc_v2i64 (v16i1sextv16i8)),
            (bc_v2i64 (add (v16i8 VR128:$src), (v16i1sextv16i8)))),
            (VPABSBrr128 VR128:$src)>;
  def : Pat<(xor
            (bc_v2i64 (v8i1sextv8i16)),
            (bc_v2i64 (add (v8i16 VR128:$src), (v8i1sextv8i16)))),
            (VPABSWrr128 VR128:$src)>;
  def : Pat<(xor
            (bc_v2i64 (v4i1sextv4i32)),
            (bc_v2i64 (add (v4i32 VR128:$src), (v4i1sextv4i32)))),
            (VPABSDrr128 VR128:$src)>;
}

let Predicates = [HasAVX2] in {
  defm VPABSB  : SS3I_unop_rm_int_y<0x1C, "vpabsb",
                                    int_x86_avx2_pabs_b>, VEX, VEX_L;
  defm VPABSW  : SS3I_unop_rm_int_y<0x1D, "vpabsw",
                                    int_x86_avx2_pabs_w>, VEX, VEX_L;
  defm VPABSD  : SS3I_unop_rm_int_y<0x1E, "vpabsd",
                                    int_x86_avx2_pabs_d>, VEX, VEX_L;

  def : Pat<(xor
            (bc_v4i64 (v32i1sextv32i8)),
            (bc_v4i64 (add (v32i8 VR256:$src), (v32i1sextv32i8)))),
            (VPABSBrr256 VR256:$src)>;
  def : Pat<(xor
            (bc_v4i64 (v16i1sextv16i16)),
            (bc_v4i64 (add (v16i16 VR256:$src), (v16i1sextv16i16)))),
            (VPABSWrr256 VR256:$src)>;
  def : Pat<(xor
            (bc_v4i64 (v8i1sextv8i32)),
            (bc_v4i64 (add (v8i32 VR256:$src), (v8i1sextv8i32)))),
            (VPABSDrr256 VR256:$src)>;
}

defm PABSB : SS3I_unop_rm_int<0x1C, "pabsb",
                              int_x86_ssse3_pabs_b_128>;
defm PABSW : SS3I_unop_rm_int<0x1D, "pabsw",
                              int_x86_ssse3_pabs_w_128>;
defm PABSD : SS3I_unop_rm_int<0x1E, "pabsd",
                              int_x86_ssse3_pabs_d_128>;

let Predicates = [HasSSSE3] in {
  def : Pat<(xor
            (bc_v2i64 (v16i1sextv16i8)),
            (bc_v2i64 (add (v16i8 VR128:$src), (v16i1sextv16i8)))),
            (PABSBrr128 VR128:$src)>;
  def : Pat<(xor
            (bc_v2i64 (v8i1sextv8i16)),
            (bc_v2i64 (add (v8i16 VR128:$src), (v8i1sextv8i16)))),
            (PABSWrr128 VR128:$src)>;
  def : Pat<(xor
            (bc_v2i64 (v4i1sextv4i32)),
            (bc_v2i64 (add (v4i32 VR128:$src), (v4i1sextv4i32)))),
            (PABSDrr128 VR128:$src)>;
}

//===---------------------------------------------------------------------===//
// SSSE3 - Packed Binary Operator Instructions
//===---------------------------------------------------------------------===//

let Sched = WriteVecALU in {
def SSE_PHADDSUBD : OpndItins<
  IIC_SSE_PHADDSUBD_RR, IIC_SSE_PHADDSUBD_RM
>;
def SSE_PHADDSUBSW : OpndItins<
  IIC_SSE_PHADDSUBSW_RR, IIC_SSE_PHADDSUBSW_RM
>;
def SSE_PHADDSUBW : OpndItins<
  IIC_SSE_PHADDSUBW_RR, IIC_SSE_PHADDSUBW_RM
>;
}
let Sched = WriteShuffle in
def SSE_PSHUFB : OpndItins<
  IIC_SSE_PSHUFB_RR, IIC_SSE_PSHUFB_RM
>;
let Sched = WriteVecALU in
def SSE_PSIGN : OpndItins<
  IIC_SSE_PSIGN_RR, IIC_SSE_PSIGN_RM
>;
let Sched = WriteVecIMul in
def SSE_PMULHRSW : OpndItins<
  IIC_SSE_PMULHRSW, IIC_SSE_PMULHRSW
>;

/// SS3I_binop_rm - Simple SSSE3 bin op
multiclass SS3I_binop_rm<bits<8> opc, string OpcodeStr, SDNode OpNode,
                         ValueType OpVT, RegisterClass RC, PatFrag memop_frag,
                         X86MemOperand x86memop, OpndItins itins,
                         bit Is2Addr = 1> {
  let isCommutable = 1 in
  def rr : SS38I<opc, MRMSrcReg, (outs RC:$dst),
       (ins RC:$src1, RC:$src2),
       !if(Is2Addr,
         !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
         !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set RC:$dst, (OpVT (OpNode RC:$src1, RC:$src2)))], itins.rr>,
       OpSize, Sched<[itins.Sched]>;
  def rm : SS38I<opc, MRMSrcMem, (outs RC:$dst),
       (ins RC:$src1, x86memop:$src2),
       !if(Is2Addr,
         !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
         !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set RC:$dst,
         (OpVT (OpNode RC:$src1,
          (bitconvert (memop_frag addr:$src2)))))], itins.rm>, OpSize,
       Sched<[itins.Sched.Folded, ReadAfterLd]>;
}

/// SS3I_binop_rm_int - Simple SSSE3 bin op whose type can be v*{i8,i16,i32}.
multiclass SS3I_binop_rm_int<bits<8> opc, string OpcodeStr,
                             Intrinsic IntId128, OpndItins itins,
                             bit Is2Addr = 1> {
  let isCommutable = 1 in
  def rr128 : SS38I<opc, MRMSrcReg, (outs VR128:$dst),
       (ins VR128:$src1, VR128:$src2),
       !if(Is2Addr,
         !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
         !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set VR128:$dst, (IntId128 VR128:$src1, VR128:$src2))]>,
       OpSize, Sched<[itins.Sched]>;
  def rm128 : SS38I<opc, MRMSrcMem, (outs VR128:$dst),
       (ins VR128:$src1, i128mem:$src2),
       !if(Is2Addr,
         !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
         !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set VR128:$dst,
         (IntId128 VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))))]>, OpSize,
       Sched<[itins.Sched.Folded, ReadAfterLd]>;
}

multiclass SS3I_binop_rm_int_y<bits<8> opc, string OpcodeStr,
                               Intrinsic IntId256> {
  let isCommutable = 1 in
  def rr256 : SS38I<opc, MRMSrcReg, (outs VR256:$dst),
       (ins VR256:$src1, VR256:$src2),
       !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
       [(set VR256:$dst, (IntId256 VR256:$src1, VR256:$src2))]>,
       OpSize;
  def rm256 : SS38I<opc, MRMSrcMem, (outs VR256:$dst),
       (ins VR256:$src1, i256mem:$src2),
       !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
       [(set VR256:$dst,
         (IntId256 VR256:$src1,
          (bitconvert (loadv4i64 addr:$src2))))]>, OpSize;
}

let ImmT = NoImm, Predicates = [HasAVX] in {
let isCommutable = 0 in {
  defm VPHADDW    : SS3I_binop_rm<0x01, "vphaddw", X86hadd, v8i16, VR128,
                                  loadv2i64, i128mem,
                                  SSE_PHADDSUBW, 0>, VEX_4V;
  defm VPHADDD    : SS3I_binop_rm<0x02, "vphaddd", X86hadd, v4i32, VR128,
                                  loadv2i64, i128mem,
                                  SSE_PHADDSUBD, 0>, VEX_4V;
  defm VPHSUBW    : SS3I_binop_rm<0x05, "vphsubw", X86hsub, v8i16, VR128,
                                  loadv2i64, i128mem,
                                  SSE_PHADDSUBW, 0>, VEX_4V;
  defm VPHSUBD    : SS3I_binop_rm<0x06, "vphsubd", X86hsub, v4i32, VR128,
                                  loadv2i64, i128mem,
                                  SSE_PHADDSUBD, 0>, VEX_4V;
  defm VPSIGNB    : SS3I_binop_rm<0x08, "vpsignb", X86psign, v16i8, VR128,
                                  loadv2i64, i128mem,
                                  SSE_PSIGN, 0>, VEX_4V;
  defm VPSIGNW    : SS3I_binop_rm<0x09, "vpsignw", X86psign, v8i16, VR128,
                                  loadv2i64, i128mem,
                                  SSE_PSIGN, 0>, VEX_4V;
  defm VPSIGND    : SS3I_binop_rm<0x0A, "vpsignd", X86psign, v4i32, VR128,
                                  loadv2i64, i128mem,
                                  SSE_PSIGN, 0>, VEX_4V;
  defm VPSHUFB    : SS3I_binop_rm<0x00, "vpshufb", X86pshufb, v16i8, VR128,
                                  loadv2i64, i128mem,
                                  SSE_PSHUFB, 0>, VEX_4V;
  defm VPHADDSW   : SS3I_binop_rm_int<0x03, "vphaddsw",
                                      int_x86_ssse3_phadd_sw_128,
                                      SSE_PHADDSUBSW, 0>, VEX_4V;
  defm VPHSUBSW   : SS3I_binop_rm_int<0x07, "vphsubsw",
                                      int_x86_ssse3_phsub_sw_128,
                                      SSE_PHADDSUBSW, 0>, VEX_4V;
  defm VPMADDUBSW : SS3I_binop_rm_int<0x04, "vpmaddubsw",
                                      int_x86_ssse3_pmadd_ub_sw_128,
                                      SSE_PMADD, 0>, VEX_4V;
}
defm VPMULHRSW    : SS3I_binop_rm_int<0x0B, "vpmulhrsw",
                                      int_x86_ssse3_pmul_hr_sw_128,
                                      SSE_PMULHRSW, 0>, VEX_4V;
}

let ImmT = NoImm, Predicates = [HasAVX2] in {
let isCommutable = 0 in {
  defm VPHADDWY   : SS3I_binop_rm<0x01, "vphaddw", X86hadd, v16i16, VR256,
                                  loadv4i64, i256mem,
                                  SSE_PHADDSUBW, 0>, VEX_4V, VEX_L;
  defm VPHADDDY   : SS3I_binop_rm<0x02, "vphaddd", X86hadd, v8i32, VR256,
                                  loadv4i64, i256mem,
                                  SSE_PHADDSUBW, 0>, VEX_4V, VEX_L;
  defm VPHSUBWY   : SS3I_binop_rm<0x05, "vphsubw", X86hsub, v16i16, VR256,
                                  loadv4i64, i256mem,
                                  SSE_PHADDSUBW, 0>, VEX_4V, VEX_L;
  defm VPHSUBDY   : SS3I_binop_rm<0x06, "vphsubd", X86hsub, v8i32, VR256,
                                  loadv4i64, i256mem,
                                  SSE_PHADDSUBW, 0>, VEX_4V, VEX_L;
  defm VPSIGNBY   : SS3I_binop_rm<0x08, "vpsignb", X86psign, v32i8, VR256,
                                  loadv4i64, i256mem,
                                  SSE_PHADDSUBW, 0>, VEX_4V, VEX_L;
  defm VPSIGNWY   : SS3I_binop_rm<0x09, "vpsignw", X86psign, v16i16, VR256,
                                  loadv4i64, i256mem,
                                  SSE_PHADDSUBW, 0>, VEX_4V, VEX_L;
  defm VPSIGNDY   : SS3I_binop_rm<0x0A, "vpsignd", X86psign, v8i32, VR256,
                                  loadv4i64, i256mem,
                                  SSE_PHADDSUBW, 0>, VEX_4V, VEX_L;
  defm VPSHUFBY   : SS3I_binop_rm<0x00, "vpshufb", X86pshufb, v32i8, VR256,
                                  loadv4i64, i256mem,
                                  SSE_PHADDSUBW, 0>, VEX_4V, VEX_L;
  defm VPHADDSW   : SS3I_binop_rm_int_y<0x03, "vphaddsw",
                                        int_x86_avx2_phadd_sw>, VEX_4V, VEX_L;
  defm VPHSUBSW   : SS3I_binop_rm_int_y<0x07, "vphsubsw",
                                        int_x86_avx2_phsub_sw>, VEX_4V, VEX_L;
  defm VPMADDUBSW : SS3I_binop_rm_int_y<0x04, "vpmaddubsw",
                                       int_x86_avx2_pmadd_ub_sw>, VEX_4V, VEX_L;
}
defm VPMULHRSW    : SS3I_binop_rm_int_y<0x0B, "vpmulhrsw",
                                        int_x86_avx2_pmul_hr_sw>, VEX_4V, VEX_L;
}

// None of these have i8 immediate fields.
let ImmT = NoImm, Constraints = "$src1 = $dst" in {
let isCommutable = 0 in {
  defm PHADDW    : SS3I_binop_rm<0x01, "phaddw", X86hadd, v8i16, VR128,
                                 memopv2i64, i128mem, SSE_PHADDSUBW>;
  defm PHADDD    : SS3I_binop_rm<0x02, "phaddd", X86hadd, v4i32, VR128,
                                 memopv2i64, i128mem, SSE_PHADDSUBD>;
  defm PHSUBW    : SS3I_binop_rm<0x05, "phsubw", X86hsub, v8i16, VR128,
                                 memopv2i64, i128mem, SSE_PHADDSUBW>;
  defm PHSUBD    : SS3I_binop_rm<0x06, "phsubd", X86hsub, v4i32, VR128,
                                 memopv2i64, i128mem, SSE_PHADDSUBD>;
  defm PSIGNB    : SS3I_binop_rm<0x08, "psignb", X86psign, v16i8, VR128,
                                 memopv2i64, i128mem, SSE_PSIGN>;
  defm PSIGNW    : SS3I_binop_rm<0x09, "psignw", X86psign, v8i16, VR128,
                                 memopv2i64, i128mem, SSE_PSIGN>;
  defm PSIGND    : SS3I_binop_rm<0x0A, "psignd", X86psign, v4i32, VR128,
                                 memopv2i64, i128mem, SSE_PSIGN>;
  defm PSHUFB    : SS3I_binop_rm<0x00, "pshufb", X86pshufb, v16i8, VR128,
                                 memopv2i64, i128mem, SSE_PSHUFB>;
  defm PHADDSW   : SS3I_binop_rm_int<0x03, "phaddsw",
                                     int_x86_ssse3_phadd_sw_128,
                                     SSE_PHADDSUBSW>;
  defm PHSUBSW   : SS3I_binop_rm_int<0x07, "phsubsw",
                                     int_x86_ssse3_phsub_sw_128,
                                     SSE_PHADDSUBSW>;
  defm PMADDUBSW : SS3I_binop_rm_int<0x04, "pmaddubsw",
                                     int_x86_ssse3_pmadd_ub_sw_128, SSE_PMADD>;
}
defm PMULHRSW    : SS3I_binop_rm_int<0x0B, "pmulhrsw",
                                     int_x86_ssse3_pmul_hr_sw_128,
                                     SSE_PMULHRSW>;
}

//===---------------------------------------------------------------------===//
// SSSE3 - Packed Align Instruction Patterns
//===---------------------------------------------------------------------===//

multiclass ssse3_palignr<string asm, bit Is2Addr = 1> {
  let neverHasSideEffects = 1 in {
  def R128rr : SS3AI<0x0F, MRMSrcReg, (outs VR128:$dst),
      (ins VR128:$src1, VR128:$src2, i8imm:$src3),
      !if(Is2Addr,
        !strconcat(asm, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
        !strconcat(asm,
                  "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}")),
      [], IIC_SSE_PALIGNRR>, OpSize, Sched<[WriteShuffle]>;
  let mayLoad = 1 in
  def R128rm : SS3AI<0x0F, MRMSrcMem, (outs VR128:$dst),
      (ins VR128:$src1, i128mem:$src2, i8imm:$src3),
      !if(Is2Addr,
        !strconcat(asm, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
        !strconcat(asm,
                  "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}")),
      [], IIC_SSE_PALIGNRM>, OpSize, Sched<[WriteShuffleLd, ReadAfterLd]>;
  }
}

multiclass ssse3_palignr_y<string asm, bit Is2Addr = 1> {
  let neverHasSideEffects = 1 in {
  def R256rr : SS3AI<0x0F, MRMSrcReg, (outs VR256:$dst),
      (ins VR256:$src1, VR256:$src2, i8imm:$src3),
      !strconcat(asm,
                 "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
      []>, OpSize, Sched<[WriteShuffle]>;
  let mayLoad = 1 in
  def R256rm : SS3AI<0x0F, MRMSrcMem, (outs VR256:$dst),
      (ins VR256:$src1, i256mem:$src2, i8imm:$src3),
      !strconcat(asm,
                 "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
      []>, OpSize, Sched<[WriteShuffleLd, ReadAfterLd]>;
  }
}

let Predicates = [HasAVX] in
  defm VPALIGN : ssse3_palignr<"vpalignr", 0>, VEX_4V;
let Predicates = [HasAVX2] in
  defm VPALIGN : ssse3_palignr_y<"vpalignr", 0>, VEX_4V, VEX_L;
let Constraints = "$src1 = $dst", Predicates = [UseSSSE3] in
  defm PALIGN : ssse3_palignr<"palignr">;

let Predicates = [HasAVX2] in {
def : Pat<(v8i32 (X86PAlignr VR256:$src1, VR256:$src2, (i8 imm:$imm))),
          (VPALIGNR256rr VR256:$src2, VR256:$src1, imm:$imm)>;
def : Pat<(v8f32 (X86PAlignr VR256:$src1, VR256:$src2, (i8 imm:$imm))),
          (VPALIGNR256rr VR256:$src2, VR256:$src1, imm:$imm)>;
def : Pat<(v16i16 (X86PAlignr VR256:$src1, VR256:$src2, (i8 imm:$imm))),
          (VPALIGNR256rr VR256:$src2, VR256:$src1, imm:$imm)>;
def : Pat<(v32i8 (X86PAlignr VR256:$src1, VR256:$src2, (i8 imm:$imm))),
          (VPALIGNR256rr VR256:$src2, VR256:$src1, imm:$imm)>;
}

let Predicates = [HasAVX] in {
def : Pat<(v4i32 (X86PAlignr VR128:$src1, VR128:$src2, (i8 imm:$imm))),
          (VPALIGNR128rr VR128:$src2, VR128:$src1, imm:$imm)>;
def : Pat<(v4f32 (X86PAlignr VR128:$src1, VR128:$src2, (i8 imm:$imm))),
          (VPALIGNR128rr VR128:$src2, VR128:$src1, imm:$imm)>;
def : Pat<(v8i16 (X86PAlignr VR128:$src1, VR128:$src2, (i8 imm:$imm))),
          (VPALIGNR128rr VR128:$src2, VR128:$src1, imm:$imm)>;
def : Pat<(v16i8 (X86PAlignr VR128:$src1, VR128:$src2, (i8 imm:$imm))),
          (VPALIGNR128rr VR128:$src2, VR128:$src1, imm:$imm)>;
}

let Predicates = [UseSSSE3] in {
def : Pat<(v4i32 (X86PAlignr VR128:$src1, VR128:$src2, (i8 imm:$imm))),
          (PALIGNR128rr VR128:$src2, VR128:$src1, imm:$imm)>;
def : Pat<(v4f32 (X86PAlignr VR128:$src1, VR128:$src2, (i8 imm:$imm))),
          (PALIGNR128rr VR128:$src2, VR128:$src1, imm:$imm)>;
def : Pat<(v8i16 (X86PAlignr VR128:$src1, VR128:$src2, (i8 imm:$imm))),
          (PALIGNR128rr VR128:$src2, VR128:$src1, imm:$imm)>;
def : Pat<(v16i8 (X86PAlignr VR128:$src1, VR128:$src2, (i8 imm:$imm))),
          (PALIGNR128rr VR128:$src2, VR128:$src1, imm:$imm)>;
}

//===---------------------------------------------------------------------===//
// SSSE3 - Thread synchronization
//===---------------------------------------------------------------------===//

let SchedRW = [WriteSystem] in {
let usesCustomInserter = 1 in {
def MONITOR : PseudoI<(outs), (ins i32mem:$src1, GR32:$src2, GR32:$src3),
                [(int_x86_sse3_monitor addr:$src1, GR32:$src2, GR32:$src3)]>,
                Requires<[HasSSE3]>;
}

let Uses = [EAX, ECX, EDX] in
def MONITORrrr : I<0x01, MRM_C8, (outs), (ins), "monitor", [], IIC_SSE_MONITOR>,
                 TB, Requires<[HasSSE3]>;
let Uses = [ECX, EAX] in
def MWAITrr   : I<0x01, MRM_C9, (outs), (ins), "mwait",
                [(int_x86_sse3_mwait ECX, EAX)], IIC_SSE_MWAIT>,
                TB, Requires<[HasSSE3]>;
} // SchedRW

def : InstAlias<"mwait\t{%eax, %ecx|ecx, eax}", (MWAITrr)>, Requires<[In32BitMode]>;
def : InstAlias<"mwait\t{%rax, %rcx|rcx, rax}", (MWAITrr)>, Requires<[In64BitMode]>;

def : InstAlias<"monitor\t{%eax, %ecx, %edx|edx, ecx, eax}", (MONITORrrr)>,
      Requires<[In32BitMode]>;
def : InstAlias<"monitor\t{%rax, %rcx, %rdx|rdx, rcx, rax}", (MONITORrrr)>,
      Requires<[In64BitMode]>;

//===----------------------------------------------------------------------===//
// SSE4.1 - Packed Move with Sign/Zero Extend
//===----------------------------------------------------------------------===//

multiclass SS41I_binop_rm_int8<bits<8> opc, string OpcodeStr, Intrinsic IntId,
                               OpndItins itins = DEFAULT_ITINS> {
  def rr : SS48I<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                 !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                 [(set VR128:$dst, (IntId VR128:$src))], itins.rr>, OpSize;

  def rm : SS48I<opc, MRMSrcMem, (outs VR128:$dst), (ins i64mem:$src),
                 !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
       [(set VR128:$dst,
         (IntId (bitconvert (v2i64 (scalar_to_vector (loadi64 addr:$src))))))],
         itins.rm>, OpSize;
}

multiclass SS41I_binop_rm_int16_y<bits<8> opc, string OpcodeStr,
                                 Intrinsic IntId> {
  def Yrr : SS48I<opc, MRMSrcReg, (outs VR256:$dst), (ins VR128:$src),
                  !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                  [(set VR256:$dst, (IntId VR128:$src))]>, OpSize;

  def Yrm : SS48I<opc, MRMSrcMem, (outs VR256:$dst), (ins i128mem:$src),
                  !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                  [(set VR256:$dst, (IntId (load addr:$src)))]>,
                  OpSize;
}

let Predicates = [HasAVX] in {
defm VPMOVSXBW : SS41I_binop_rm_int8<0x20, "vpmovsxbw",
                                     int_x86_sse41_pmovsxbw>, VEX;
defm VPMOVSXWD : SS41I_binop_rm_int8<0x23, "vpmovsxwd",
                                     int_x86_sse41_pmovsxwd>, VEX;
defm VPMOVSXDQ : SS41I_binop_rm_int8<0x25, "vpmovsxdq",
                                     int_x86_sse41_pmovsxdq>, VEX;
defm VPMOVZXBW : SS41I_binop_rm_int8<0x30, "vpmovzxbw",
                                     int_x86_sse41_pmovzxbw>, VEX;
defm VPMOVZXWD : SS41I_binop_rm_int8<0x33, "vpmovzxwd",
                                     int_x86_sse41_pmovzxwd>, VEX;
defm VPMOVZXDQ : SS41I_binop_rm_int8<0x35, "vpmovzxdq",
                                     int_x86_sse41_pmovzxdq>, VEX;
}

let Predicates = [HasAVX2] in {
defm VPMOVSXBW : SS41I_binop_rm_int16_y<0x20, "vpmovsxbw",
                                        int_x86_avx2_pmovsxbw>, VEX, VEX_L;
defm VPMOVSXWD : SS41I_binop_rm_int16_y<0x23, "vpmovsxwd",
                                        int_x86_avx2_pmovsxwd>, VEX, VEX_L;
defm VPMOVSXDQ : SS41I_binop_rm_int16_y<0x25, "vpmovsxdq",
                                        int_x86_avx2_pmovsxdq>, VEX, VEX_L;
defm VPMOVZXBW : SS41I_binop_rm_int16_y<0x30, "vpmovzxbw",
                                        int_x86_avx2_pmovzxbw>, VEX, VEX_L;
defm VPMOVZXWD : SS41I_binop_rm_int16_y<0x33, "vpmovzxwd",
                                        int_x86_avx2_pmovzxwd>, VEX, VEX_L;
defm VPMOVZXDQ : SS41I_binop_rm_int16_y<0x35, "vpmovzxdq",
                                        int_x86_avx2_pmovzxdq>, VEX, VEX_L;
}

defm PMOVSXBW   : SS41I_binop_rm_int8<0x20, "pmovsxbw", int_x86_sse41_pmovsxbw,                                       SSE_INTALU_ITINS_P>;
defm PMOVSXWD   : SS41I_binop_rm_int8<0x23, "pmovsxwd", int_x86_sse41_pmovsxwd,                                       SSE_INTALU_ITINS_P>;
defm PMOVSXDQ   : SS41I_binop_rm_int8<0x25, "pmovsxdq", int_x86_sse41_pmovsxdq,                                       SSE_INTALU_ITINS_P>;
defm PMOVZXBW   : SS41I_binop_rm_int8<0x30, "pmovzxbw", int_x86_sse41_pmovzxbw,                                       SSE_INTALU_ITINS_P>;
defm PMOVZXWD   : SS41I_binop_rm_int8<0x33, "pmovzxwd", int_x86_sse41_pmovzxwd,                                       SSE_INTALU_ITINS_P>;
defm PMOVZXDQ   : SS41I_binop_rm_int8<0x35, "pmovzxdq", int_x86_sse41_pmovzxdq,                                       SSE_INTALU_ITINS_P>;

let Predicates = [HasAVX] in {
  // Common patterns involving scalar load.
  def : Pat<(int_x86_sse41_pmovsxbw (vzmovl_v2i64 addr:$src)),
            (VPMOVSXBWrm addr:$src)>;
  def : Pat<(int_x86_sse41_pmovsxbw (vzload_v2i64 addr:$src)),
            (VPMOVSXBWrm addr:$src)>;
  def : Pat<(int_x86_sse41_pmovsxbw (bc_v16i8 (loadv2i64 addr:$src))),
            (VPMOVSXBWrm addr:$src)>;

  def : Pat<(int_x86_sse41_pmovsxwd (vzmovl_v2i64 addr:$src)),
            (VPMOVSXWDrm addr:$src)>;
  def : Pat<(int_x86_sse41_pmovsxwd (vzload_v2i64 addr:$src)),
            (VPMOVSXWDrm addr:$src)>;
  def : Pat<(int_x86_sse41_pmovsxwd (bc_v8i16 (loadv2i64 addr:$src))),
            (VPMOVSXWDrm addr:$src)>;

  def : Pat<(int_x86_sse41_pmovsxdq (vzmovl_v2i64 addr:$src)),
            (VPMOVSXDQrm addr:$src)>;
  def : Pat<(int_x86_sse41_pmovsxdq (vzload_v2i64 addr:$src)),
            (VPMOVSXDQrm addr:$src)>;
  def : Pat<(int_x86_sse41_pmovsxdq (bc_v4i32 (loadv2i64 addr:$src))),
            (VPMOVSXDQrm addr:$src)>;

  def : Pat<(int_x86_sse41_pmovzxbw (vzmovl_v2i64 addr:$src)),
            (VPMOVZXBWrm addr:$src)>;
  def : Pat<(int_x86_sse41_pmovzxbw (vzload_v2i64 addr:$src)),
            (VPMOVZXBWrm addr:$src)>;
  def : Pat<(int_x86_sse41_pmovzxbw (bc_v16i8 (loadv2i64 addr:$src))),
            (VPMOVZXBWrm addr:$src)>;

  def : Pat<(int_x86_sse41_pmovzxwd (vzmovl_v2i64 addr:$src)),
            (VPMOVZXWDrm addr:$src)>;
  def : Pat<(int_x86_sse41_pmovzxwd (vzload_v2i64 addr:$src)),
            (VPMOVZXWDrm addr:$src)>;
  def : Pat<(int_x86_sse41_pmovzxwd (bc_v8i16 (loadv2i64 addr:$src))),
            (VPMOVZXWDrm addr:$src)>;

  def : Pat<(int_x86_sse41_pmovzxdq (vzmovl_v2i64 addr:$src)),
            (VPMOVZXDQrm addr:$src)>;
  def : Pat<(int_x86_sse41_pmovzxdq (vzload_v2i64 addr:$src)),
            (VPMOVZXDQrm addr:$src)>;
  def : Pat<(int_x86_sse41_pmovzxdq (bc_v4i32 (loadv2i64 addr:$src))),
            (VPMOVZXDQrm addr:$src)>;
}

let Predicates = [UseSSE41] in {
  // Common patterns involving scalar load.
  def : Pat<(int_x86_sse41_pmovsxbw (vzmovl_v2i64 addr:$src)),
            (PMOVSXBWrm addr:$src)>;
  def : Pat<(int_x86_sse41_pmovsxbw (vzload_v2i64 addr:$src)),
            (PMOVSXBWrm addr:$src)>;
  def : Pat<(int_x86_sse41_pmovsxbw (bc_v16i8 (loadv2i64 addr:$src))),
            (PMOVSXBWrm addr:$src)>;

  def : Pat<(int_x86_sse41_pmovsxwd (vzmovl_v2i64 addr:$src)),
            (PMOVSXWDrm addr:$src)>;
  def : Pat<(int_x86_sse41_pmovsxwd (vzload_v2i64 addr:$src)),
            (PMOVSXWDrm addr:$src)>;
  def : Pat<(int_x86_sse41_pmovsxwd (bc_v8i16 (loadv2i64 addr:$src))),
            (PMOVSXWDrm addr:$src)>;

  def : Pat<(int_x86_sse41_pmovsxdq (vzmovl_v2i64 addr:$src)),
            (PMOVSXDQrm addr:$src)>;
  def : Pat<(int_x86_sse41_pmovsxdq (vzload_v2i64 addr:$src)),
            (PMOVSXDQrm addr:$src)>;
  def : Pat<(int_x86_sse41_pmovsxdq (bc_v4i32 (loadv2i64 addr:$src))),
            (PMOVSXDQrm addr:$src)>;

  def : Pat<(int_x86_sse41_pmovzxbw (vzmovl_v2i64 addr:$src)),
            (PMOVZXBWrm addr:$src)>;
  def : Pat<(int_x86_sse41_pmovzxbw (vzload_v2i64 addr:$src)),
            (PMOVZXBWrm addr:$src)>;
  def : Pat<(int_x86_sse41_pmovzxbw (bc_v16i8 (loadv2i64 addr:$src))),
            (PMOVZXBWrm addr:$src)>;

  def : Pat<(int_x86_sse41_pmovzxwd (vzmovl_v2i64 addr:$src)),
            (PMOVZXWDrm addr:$src)>;
  def : Pat<(int_x86_sse41_pmovzxwd (vzload_v2i64 addr:$src)),
            (PMOVZXWDrm addr:$src)>;
  def : Pat<(int_x86_sse41_pmovzxwd (bc_v8i16 (loadv2i64 addr:$src))),
            (PMOVZXWDrm addr:$src)>;

  def : Pat<(int_x86_sse41_pmovzxdq (vzmovl_v2i64 addr:$src)),
            (PMOVZXDQrm addr:$src)>;
  def : Pat<(int_x86_sse41_pmovzxdq (vzload_v2i64 addr:$src)),
            (PMOVZXDQrm addr:$src)>;
  def : Pat<(int_x86_sse41_pmovzxdq (bc_v4i32 (loadv2i64 addr:$src))),
            (PMOVZXDQrm addr:$src)>;
}

let Predicates = [HasAVX2] in {
  let AddedComplexity = 15 in {
    def : Pat<(v4i64 (X86vzmovly (v4i32 VR128:$src))),
              (VPMOVZXDQYrr VR128:$src)>;
    def : Pat<(v8i32 (X86vzmovly (v8i16 VR128:$src))),
              (VPMOVZXWDYrr VR128:$src)>;
    def : Pat<(v16i16 (X86vzmovly (v16i8 VR128:$src))),
              (VPMOVZXBWYrr VR128:$src)>;
  }

  def : Pat<(v4i64 (X86vsmovl (v4i32 VR128:$src))), (VPMOVSXDQYrr VR128:$src)>;
  def : Pat<(v8i32 (X86vsmovl (v8i16 VR128:$src))), (VPMOVSXWDYrr VR128:$src)>;
  def : Pat<(v16i16 (X86vsmovl (v16i8 VR128:$src))), (VPMOVSXBWYrr VR128:$src)>;
}

let Predicates = [HasAVX] in {
  def : Pat<(v2i64 (X86vsmovl (v4i32 VR128:$src))), (VPMOVSXDQrr VR128:$src)>;
  def : Pat<(v4i32 (X86vsmovl (v8i16 VR128:$src))), (VPMOVSXWDrr VR128:$src)>;
  def : Pat<(v8i16 (X86vsmovl (v16i8 VR128:$src))), (VPMOVSXBWrr VR128:$src)>;
}

let Predicates = [UseSSE41] in {
  def : Pat<(v2i64 (X86vsmovl (v4i32 VR128:$src))), (PMOVSXDQrr VR128:$src)>;
  def : Pat<(v4i32 (X86vsmovl (v8i16 VR128:$src))), (PMOVSXWDrr VR128:$src)>;
  def : Pat<(v8i16 (X86vsmovl (v16i8 VR128:$src))), (PMOVSXBWrr VR128:$src)>;
}


multiclass SS41I_binop_rm_int4<bits<8> opc, string OpcodeStr, Intrinsic IntId,
                               OpndItins itins = DEFAULT_ITINS> {
  def rr : SS48I<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                 !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                 [(set VR128:$dst, (IntId VR128:$src))], itins.rr>, OpSize;

  def rm : SS48I<opc, MRMSrcMem, (outs VR128:$dst), (ins i32mem:$src),
                 !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
       [(set VR128:$dst,
         (IntId (bitconvert (v4i32 (scalar_to_vector (loadi32 addr:$src))))))],
         itins.rm>,
          OpSize;
}

multiclass SS41I_binop_rm_int8_y<bits<8> opc, string OpcodeStr,
                                 Intrinsic IntId> {
  def Yrr : SS48I<opc, MRMSrcReg, (outs VR256:$dst), (ins VR128:$src),
                  !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                  [(set VR256:$dst, (IntId VR128:$src))]>, OpSize;

  def Yrm : SS48I<opc, MRMSrcMem, (outs VR256:$dst), (ins i32mem:$src),
                  !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
       [(set VR256:$dst,
         (IntId (bitconvert (v2i64 (scalar_to_vector (loadi64 addr:$src))))))]>,
          OpSize;
}

let Predicates = [HasAVX] in {
defm VPMOVSXBD : SS41I_binop_rm_int4<0x21, "vpmovsxbd", int_x86_sse41_pmovsxbd>,
                                     VEX;
defm VPMOVSXWQ : SS41I_binop_rm_int4<0x24, "vpmovsxwq", int_x86_sse41_pmovsxwq>,
                                     VEX;
defm VPMOVZXBD : SS41I_binop_rm_int4<0x31, "vpmovzxbd", int_x86_sse41_pmovzxbd>,
                                     VEX;
defm VPMOVZXWQ : SS41I_binop_rm_int4<0x34, "vpmovzxwq", int_x86_sse41_pmovzxwq>,
                                     VEX;
}

let Predicates = [HasAVX2] in {
defm VPMOVSXBD : SS41I_binop_rm_int8_y<0x21, "vpmovsxbd",
                                       int_x86_avx2_pmovsxbd>, VEX, VEX_L;
defm VPMOVSXWQ : SS41I_binop_rm_int8_y<0x24, "vpmovsxwq",
                                       int_x86_avx2_pmovsxwq>, VEX, VEX_L;
defm VPMOVZXBD : SS41I_binop_rm_int8_y<0x31, "vpmovzxbd",
                                       int_x86_avx2_pmovzxbd>, VEX, VEX_L;
defm VPMOVZXWQ : SS41I_binop_rm_int8_y<0x34, "vpmovzxwq",
                                       int_x86_avx2_pmovzxwq>, VEX, VEX_L;
}

defm PMOVSXBD   : SS41I_binop_rm_int4<0x21, "pmovsxbd", int_x86_sse41_pmovsxbd,
                                      SSE_INTALU_ITINS_P>;
defm PMOVSXWQ   : SS41I_binop_rm_int4<0x24, "pmovsxwq", int_x86_sse41_pmovsxwq,
                                      SSE_INTALU_ITINS_P>;
defm PMOVZXBD   : SS41I_binop_rm_int4<0x31, "pmovzxbd", int_x86_sse41_pmovzxbd,
                                      SSE_INTALU_ITINS_P>;
defm PMOVZXWQ   : SS41I_binop_rm_int4<0x34, "pmovzxwq", int_x86_sse41_pmovzxwq,
                                      SSE_INTALU_ITINS_P>;

let Predicates = [HasAVX] in {
  // Common patterns involving scalar load
  def : Pat<(int_x86_sse41_pmovsxbd (vzmovl_v4i32 addr:$src)),
            (VPMOVSXBDrm addr:$src)>;
  def : Pat<(int_x86_sse41_pmovsxwq (vzmovl_v4i32 addr:$src)),
            (VPMOVSXWQrm addr:$src)>;

  def : Pat<(int_x86_sse41_pmovzxbd (vzmovl_v4i32 addr:$src)),
            (VPMOVZXBDrm addr:$src)>;
  def : Pat<(int_x86_sse41_pmovzxwq (vzmovl_v4i32 addr:$src)),
            (VPMOVZXWQrm addr:$src)>;
}

let Predicates = [UseSSE41] in {
  // Common patterns involving scalar load
  def : Pat<(int_x86_sse41_pmovsxbd (vzmovl_v4i32 addr:$src)),
            (PMOVSXBDrm addr:$src)>;
  def : Pat<(int_x86_sse41_pmovsxwq (vzmovl_v4i32 addr:$src)),
            (PMOVSXWQrm addr:$src)>;

  def : Pat<(int_x86_sse41_pmovzxbd (vzmovl_v4i32 addr:$src)),
            (PMOVZXBDrm addr:$src)>;
  def : Pat<(int_x86_sse41_pmovzxwq (vzmovl_v4i32 addr:$src)),
            (PMOVZXWQrm addr:$src)>;
}

multiclass SS41I_binop_rm_int2<bits<8> opc, string OpcodeStr, Intrinsic IntId,
                               OpndItins itins = DEFAULT_ITINS> {
  def rr : SS48I<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                 !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                 [(set VR128:$dst, (IntId VR128:$src))]>, OpSize;

  // Expecting a i16 load any extended to i32 value.
  def rm : SS48I<opc, MRMSrcMem, (outs VR128:$dst), (ins i16mem:$src),
                 !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                 [(set VR128:$dst, (IntId (bitconvert
                     (v4i32 (scalar_to_vector (loadi16_anyext addr:$src))))))]>,
                 OpSize;
}

multiclass SS41I_binop_rm_int4_y<bits<8> opc, string OpcodeStr,
                                 Intrinsic IntId> {
  def Yrr : SS48I<opc, MRMSrcReg, (outs VR256:$dst), (ins VR128:$src),
                 !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                 [(set VR256:$dst, (IntId VR128:$src))]>, OpSize;

  // Expecting a i16 load any extended to i32 value.
  def Yrm : SS48I<opc, MRMSrcMem, (outs VR256:$dst), (ins i16mem:$src),
                  !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                  [(set VR256:$dst, (IntId (bitconvert
                      (v4i32 (scalar_to_vector (loadi32 addr:$src))))))]>,
                  OpSize;
}

let Predicates = [HasAVX] in {
defm VPMOVSXBQ : SS41I_binop_rm_int2<0x22, "vpmovsxbq", int_x86_sse41_pmovsxbq>,
                                     VEX;
defm VPMOVZXBQ : SS41I_binop_rm_int2<0x32, "vpmovzxbq", int_x86_sse41_pmovzxbq>,
                                     VEX;
}
let Predicates = [HasAVX2] in {
defm VPMOVSXBQ : SS41I_binop_rm_int4_y<0x22, "vpmovsxbq",
                                       int_x86_avx2_pmovsxbq>, VEX, VEX_L;
defm VPMOVZXBQ : SS41I_binop_rm_int4_y<0x32, "vpmovzxbq",
                                       int_x86_avx2_pmovzxbq>, VEX, VEX_L;
}
defm PMOVSXBQ   : SS41I_binop_rm_int2<0x22, "pmovsxbq", int_x86_sse41_pmovsxbq,
                                      SSE_INTALU_ITINS_P>;
defm PMOVZXBQ   : SS41I_binop_rm_int2<0x32, "pmovzxbq", int_x86_sse41_pmovzxbq,
                                      SSE_INTALU_ITINS_P>;

let Predicates = [HasAVX2] in {
  def : Pat<(v16i16 (X86vsext (v16i8 VR128:$src))), (VPMOVSXBWYrr VR128:$src)>;
  def : Pat<(v8i32  (X86vsext (v16i8 VR128:$src))), (VPMOVSXBDYrr VR128:$src)>;
  def : Pat<(v4i64  (X86vsext (v16i8 VR128:$src))), (VPMOVSXBQYrr VR128:$src)>;

  def : Pat<(v8i32  (X86vsext (v8i16 VR128:$src))), (VPMOVSXWDYrr VR128:$src)>;
  def : Pat<(v4i64  (X86vsext (v8i16 VR128:$src))), (VPMOVSXWQYrr VR128:$src)>;

  def : Pat<(v4i64  (X86vsext (v4i32 VR128:$src))), (VPMOVSXDQYrr VR128:$src)>;

  def : Pat<(v16i16 (X86vsext (v32i8 VR256:$src))),
            (VPMOVSXBWYrr (EXTRACT_SUBREG VR256:$src, sub_xmm))>;
  def : Pat<(v8i32 (X86vsext (v32i8 VR256:$src))),
            (VPMOVSXBDYrr (EXTRACT_SUBREG VR256:$src, sub_xmm))>;
  def : Pat<(v4i64 (X86vsext (v32i8 VR256:$src))),
            (VPMOVSXBQYrr (EXTRACT_SUBREG VR256:$src, sub_xmm))>;

  def : Pat<(v8i32 (X86vsext (v16i16 VR256:$src))),
            (VPMOVSXWDYrr (EXTRACT_SUBREG VR256:$src, sub_xmm))>;
  def : Pat<(v4i64 (X86vsext (v16i16 VR256:$src))),
            (VPMOVSXWQYrr (EXTRACT_SUBREG VR256:$src, sub_xmm))>;

  def : Pat<(v4i64 (X86vsext (v8i32 VR256:$src))),
            (VPMOVSXDQYrr (EXTRACT_SUBREG VR256:$src, sub_xmm))>;

  def : Pat<(v8i32 (X86vsmovl (v8i16 (bitconvert (v2i64 (load addr:$src)))))),
            (VPMOVSXWDYrm addr:$src)>;
  def : Pat<(v4i64 (X86vsmovl (v4i32 (bitconvert (v2i64 (load addr:$src)))))),
            (VPMOVSXDQYrm addr:$src)>;

  def : Pat<(v8i32 (X86vsext (v16i8 (bitconvert (v2i64 
                    (scalar_to_vector (loadi64 addr:$src))))))),
            (VPMOVSXBDYrm addr:$src)>;
  def : Pat<(v8i32 (X86vsext (v16i8 (bitconvert (v2f64 
                    (scalar_to_vector (loadf64 addr:$src))))))),
            (VPMOVSXBDYrm addr:$src)>;

  def : Pat<(v4i64 (X86vsext (v8i16 (bitconvert (v2i64 
                    (scalar_to_vector (loadi64 addr:$src))))))),
            (VPMOVSXWQYrm addr:$src)>;
  def : Pat<(v4i64 (X86vsext (v8i16 (bitconvert (v2f64 
                    (scalar_to_vector (loadf64 addr:$src))))))),
            (VPMOVSXWQYrm addr:$src)>;

  def : Pat<(v4i64 (X86vsext (v16i8 (bitconvert (v4i32 
                    (scalar_to_vector (loadi32 addr:$src))))))),
            (VPMOVSXBQYrm addr:$src)>;
}

let Predicates = [HasAVX] in {
  // Common patterns involving scalar load
  def : Pat<(int_x86_sse41_pmovsxbq
              (bitconvert (v4i32 (X86vzmovl
                            (v4i32 (scalar_to_vector (loadi32 addr:$src))))))),
            (VPMOVSXBQrm addr:$src)>;

  def : Pat<(int_x86_sse41_pmovzxbq
              (bitconvert (v4i32 (X86vzmovl
                            (v4i32 (scalar_to_vector (loadi32 addr:$src))))))),
            (VPMOVZXBQrm addr:$src)>;
}

let Predicates = [UseSSE41] in {
  def : Pat<(v8i16 (X86vsext (v16i8 VR128:$src))), (PMOVSXBWrr VR128:$src)>;
  def : Pat<(v4i32 (X86vsext (v16i8 VR128:$src))), (PMOVSXBDrr VR128:$src)>;
  def : Pat<(v2i64 (X86vsext (v16i8 VR128:$src))), (PMOVSXBQrr VR128:$src)>;

  def : Pat<(v4i32 (X86vsext (v8i16 VR128:$src))), (PMOVSXWDrr VR128:$src)>;
  def : Pat<(v2i64 (X86vsext (v8i16 VR128:$src))), (PMOVSXWQrr VR128:$src)>;

  def : Pat<(v2i64 (X86vsext (v4i32 VR128:$src))), (PMOVSXDQrr VR128:$src)>;

  // Common patterns involving scalar load
  def : Pat<(int_x86_sse41_pmovsxbq
              (bitconvert (v4i32 (X86vzmovl
                            (v4i32 (scalar_to_vector (loadi32 addr:$src))))))),
            (PMOVSXBQrm addr:$src)>;

  def : Pat<(int_x86_sse41_pmovzxbq
              (bitconvert (v4i32 (X86vzmovl
                            (v4i32 (scalar_to_vector (loadi32 addr:$src))))))),
            (PMOVZXBQrm addr:$src)>;

  def : Pat<(v4i32 (X86vsext (v8i16 (bitconvert (v2i64
                    (scalar_to_vector (loadi64 addr:$src))))))),
            (PMOVSXWDrm addr:$src)>;
  def : Pat<(v4i32 (X86vsext (v8i16 (bitconvert (v2f64
                    (scalar_to_vector (loadf64 addr:$src))))))),
            (PMOVSXWDrm addr:$src)>;
  def : Pat<(v4i32 (X86vsext (v16i8 (bitconvert (v4i32
                    (scalar_to_vector (loadi32 addr:$src))))))),
            (PMOVSXBDrm addr:$src)>;
  def : Pat<(v2i64 (X86vsext (v8i16 (bitconvert (v4i32
                    (scalar_to_vector (loadi32 addr:$src))))))),
            (PMOVSXWQrm addr:$src)>;
  def : Pat<(v2i64 (X86vsext (v16i8 (bitconvert (v4i32
                    (scalar_to_vector (extloadi32i16 addr:$src))))))),
            (PMOVSXBQrm addr:$src)>;
  def : Pat<(v2i64 (X86vsext (v4i32 (bitconvert (v2i64
                    (scalar_to_vector (loadi64 addr:$src))))))),
            (PMOVSXDQrm addr:$src)>;
  def : Pat<(v2i64 (X86vsext (v4i32 (bitconvert (v2f64
                    (scalar_to_vector (loadf64 addr:$src))))))),
            (PMOVSXDQrm addr:$src)>;
  def : Pat<(v8i16 (X86vsext (v16i8 (bitconvert (v2i64
                    (scalar_to_vector (loadi64 addr:$src))))))),
            (PMOVSXBWrm addr:$src)>;
  def : Pat<(v8i16 (X86vsext (v16i8 (bitconvert (v2f64
                    (scalar_to_vector (loadf64 addr:$src))))))),
            (PMOVSXBWrm addr:$src)>;
}

let Predicates = [HasAVX2] in {
  def : Pat<(v16i16 (X86vzext (v16i8 VR128:$src))), (VPMOVZXBWYrr VR128:$src)>;
  def : Pat<(v8i32  (X86vzext (v16i8 VR128:$src))), (VPMOVZXBDYrr VR128:$src)>;
  def : Pat<(v4i64  (X86vzext (v16i8 VR128:$src))), (VPMOVZXBQYrr VR128:$src)>;

  def : Pat<(v8i32  (X86vzext (v8i16 VR128:$src))), (VPMOVZXWDYrr VR128:$src)>;
  def : Pat<(v4i64  (X86vzext (v8i16 VR128:$src))), (VPMOVZXWQYrr VR128:$src)>;

  def : Pat<(v4i64  (X86vzext (v4i32 VR128:$src))), (VPMOVZXDQYrr VR128:$src)>;

  def : Pat<(v16i16 (X86vzext (v32i8 VR256:$src))),
            (VPMOVZXBWYrr (EXTRACT_SUBREG VR256:$src, sub_xmm))>;
  def : Pat<(v8i32 (X86vzext (v32i8 VR256:$src))),
            (VPMOVZXBDYrr (EXTRACT_SUBREG VR256:$src, sub_xmm))>;
  def : Pat<(v4i64 (X86vzext (v32i8 VR256:$src))),
            (VPMOVZXBQYrr (EXTRACT_SUBREG VR256:$src, sub_xmm))>;

  def : Pat<(v8i32 (X86vzext (v16i16 VR256:$src))),
            (VPMOVZXWDYrr (EXTRACT_SUBREG VR256:$src, sub_xmm))>;
  def : Pat<(v4i64 (X86vzext (v16i16 VR256:$src))),
            (VPMOVZXWQYrr (EXTRACT_SUBREG VR256:$src, sub_xmm))>;

  def : Pat<(v4i64 (X86vzext (v8i32 VR256:$src))),
            (VPMOVZXDQYrr (EXTRACT_SUBREG VR256:$src, sub_xmm))>;
}

let Predicates = [HasAVX] in {
  def : Pat<(v8i16 (X86vzext (v16i8 VR128:$src))), (VPMOVZXBWrr VR128:$src)>;
  def : Pat<(v4i32 (X86vzext (v16i8 VR128:$src))), (VPMOVZXBDrr VR128:$src)>;
  def : Pat<(v2i64 (X86vzext (v16i8 VR128:$src))), (VPMOVZXBQrr VR128:$src)>;

  def : Pat<(v4i32 (X86vzext (v8i16 VR128:$src))), (VPMOVZXWDrr VR128:$src)>;
  def : Pat<(v2i64 (X86vzext (v8i16 VR128:$src))), (VPMOVZXWQrr VR128:$src)>;

  def : Pat<(v2i64 (X86vzext (v4i32 VR128:$src))), (VPMOVZXDQrr VR128:$src)>;

  def : Pat<(v8i16 (X86vzext (v16i8 (bitconvert (v2i64 (scalar_to_vector (loadi64 addr:$src))))))),
            (VPMOVZXBWrm addr:$src)>;
  def : Pat<(v8i16 (X86vzext (v16i8 (bitconvert (v2f64 (scalar_to_vector (loadf64 addr:$src))))))),
            (VPMOVZXBWrm addr:$src)>;
  def : Pat<(v4i32 (X86vzext (v16i8 (bitconvert (v4i32 (scalar_to_vector (loadi32 addr:$src))))))),
            (VPMOVZXBDrm addr:$src)>;
  def : Pat<(v2i64 (X86vzext (v16i8 (bitconvert (v4i32 (scalar_to_vector (loadi16_anyext addr:$src))))))),
            (VPMOVZXBQrm addr:$src)>;

  def : Pat<(v4i32 (X86vzext (v8i16 (bitconvert (v2i64 (scalar_to_vector (loadi64 addr:$src))))))),
            (VPMOVZXWDrm addr:$src)>;
  def : Pat<(v4i32 (X86vzext (v8i16 (bitconvert (v2f64 (scalar_to_vector (loadf64 addr:$src))))))),
            (VPMOVZXWDrm addr:$src)>;
  def : Pat<(v2i64 (X86vzext (v8i16 (bitconvert (v4i32 (scalar_to_vector (loadi32 addr:$src))))))),
            (VPMOVZXWQrm addr:$src)>;

  def : Pat<(v2i64 (X86vzext (v4i32 (bitconvert (v2i64 (scalar_to_vector (loadi64 addr:$src))))))),
            (VPMOVZXDQrm addr:$src)>;
  def : Pat<(v2i64 (X86vzext (v4i32 (bitconvert (v2f64 (scalar_to_vector (loadf64 addr:$src))))))),
            (VPMOVZXDQrm addr:$src)>;
  def : Pat<(v2i64 (X86vzext (v4i32 (bitconvert (v2i64 (X86vzload addr:$src)))))),
            (VPMOVZXDQrm addr:$src)>;

  def : Pat<(v8i16 (X86vsext (v16i8 VR128:$src))), (VPMOVSXBWrr VR128:$src)>;
  def : Pat<(v4i32 (X86vsext (v16i8 VR128:$src))), (VPMOVSXBDrr VR128:$src)>;
  def : Pat<(v2i64 (X86vsext (v16i8 VR128:$src))), (VPMOVSXBQrr VR128:$src)>;

  def : Pat<(v4i32 (X86vsext (v8i16 VR128:$src))), (VPMOVSXWDrr VR128:$src)>;
  def : Pat<(v2i64 (X86vsext (v8i16 VR128:$src))), (VPMOVSXWQrr VR128:$src)>;

  def : Pat<(v2i64 (X86vsext (v4i32 VR128:$src))), (VPMOVSXDQrr VR128:$src)>;

  def : Pat<(v4i32 (X86vsext (v8i16 (bitconvert (v2i64
                    (scalar_to_vector (loadi64 addr:$src))))))),
            (VPMOVSXWDrm addr:$src)>;
  def : Pat<(v2i64 (X86vsext (v4i32 (bitconvert (v2i64
                    (scalar_to_vector (loadi64 addr:$src))))))),
            (VPMOVSXDQrm addr:$src)>;
  def : Pat<(v4i32 (X86vsext (v8i16 (bitconvert (v2f64
                    (scalar_to_vector (loadf64 addr:$src))))))),
            (VPMOVSXWDrm addr:$src)>;
  def : Pat<(v2i64 (X86vsext (v4i32 (bitconvert (v2f64
                    (scalar_to_vector (loadf64 addr:$src))))))),
            (VPMOVSXDQrm addr:$src)>;
  def : Pat<(v8i16 (X86vsext (v16i8 (bitconvert (v2i64
                    (scalar_to_vector (loadi64 addr:$src))))))),
            (VPMOVSXBWrm addr:$src)>;
  def : Pat<(v8i16 (X86vsext (v16i8 (bitconvert (v2f64
                    (scalar_to_vector (loadf64 addr:$src))))))),
            (VPMOVSXBWrm addr:$src)>;

  def : Pat<(v4i32 (X86vsext (v16i8 (bitconvert (v4i32
                    (scalar_to_vector (loadi32 addr:$src))))))),
            (VPMOVSXBDrm addr:$src)>;
  def : Pat<(v2i64 (X86vsext (v8i16 (bitconvert (v4i32
                    (scalar_to_vector (loadi32 addr:$src))))))),
            (VPMOVSXWQrm addr:$src)>;
  def : Pat<(v2i64 (X86vsext (v16i8 (bitconvert (v4i32
                    (scalar_to_vector (extloadi32i16 addr:$src))))))),
            (VPMOVSXBQrm addr:$src)>;
}

let Predicates = [UseSSE41] in {
  def : Pat<(v8i16 (X86vzext (v16i8 VR128:$src))), (PMOVZXBWrr VR128:$src)>;
  def : Pat<(v4i32 (X86vzext (v16i8 VR128:$src))), (PMOVZXBDrr VR128:$src)>;
  def : Pat<(v2i64 (X86vzext (v16i8 VR128:$src))), (PMOVZXBQrr VR128:$src)>;

  def : Pat<(v4i32 (X86vzext (v8i16 VR128:$src))), (PMOVZXWDrr VR128:$src)>;
  def : Pat<(v2i64 (X86vzext (v8i16 VR128:$src))), (PMOVZXWQrr VR128:$src)>;

  def : Pat<(v2i64 (X86vzext (v4i32 VR128:$src))), (PMOVZXDQrr VR128:$src)>;

  def : Pat<(v8i16 (X86vzext (v16i8 (bitconvert (v2i64 (scalar_to_vector (loadi64 addr:$src))))))),
            (PMOVZXBWrm addr:$src)>;
  def : Pat<(v8i16 (X86vzext (v16i8 (bitconvert (v2f64 (scalar_to_vector (loadf64 addr:$src))))))),
            (PMOVZXBWrm addr:$src)>;
  def : Pat<(v4i32 (X86vzext (v16i8 (bitconvert (v4i32 (scalar_to_vector (loadi32 addr:$src))))))),
            (PMOVZXBDrm addr:$src)>;
  def : Pat<(v2i64 (X86vzext (v16i8 (bitconvert (v4i32 (scalar_to_vector (loadi16_anyext addr:$src))))))),
            (PMOVZXBQrm addr:$src)>;

  def : Pat<(v4i32 (X86vzext (v8i16 (bitconvert (v2i64 (scalar_to_vector (loadi64 addr:$src))))))),
            (PMOVZXWDrm addr:$src)>;
  def : Pat<(v4i32 (X86vzext (v8i16 (bitconvert (v2f64 (scalar_to_vector (loadf64 addr:$src))))))),
            (PMOVZXWDrm addr:$src)>;
  def : Pat<(v2i64 (X86vzext (v8i16 (bitconvert (v4i32 (scalar_to_vector (loadi32 addr:$src))))))),
            (PMOVZXWQrm addr:$src)>;

  def : Pat<(v2i64 (X86vzext (v4i32 (bitconvert (v2i64 (scalar_to_vector (loadi64 addr:$src))))))),
            (PMOVZXDQrm addr:$src)>;
  def : Pat<(v2i64 (X86vzext (v4i32 (bitconvert (v2f64 (scalar_to_vector (loadf64 addr:$src))))))),
            (PMOVZXDQrm addr:$src)>;
  def : Pat<(v2i64 (X86vzext (v4i32 (bitconvert (v2i64 (X86vzload addr:$src)))))),
            (PMOVZXDQrm addr:$src)>;
}

//===----------------------------------------------------------------------===//
// SSE4.1 - Extract Instructions
//===----------------------------------------------------------------------===//

/// SS41I_binop_ext8 - SSE 4.1 extract 8 bits to 32 bit reg or 8 bit mem
multiclass SS41I_extract8<bits<8> opc, string OpcodeStr> {
  def rr : SS4AIi8<opc, MRMDestReg, (outs GR32orGR64:$dst),
                 (ins VR128:$src1, i32i8imm:$src2),
                 !strconcat(OpcodeStr,
                            "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                 [(set GR32orGR64:$dst, (X86pextrb (v16i8 VR128:$src1),
                                         imm:$src2))]>,
                 OpSize;
  let neverHasSideEffects = 1, mayStore = 1 in
  def mr : SS4AIi8<opc, MRMDestMem, (outs),
                 (ins i8mem:$dst, VR128:$src1, i32i8imm:$src2),
                 !strconcat(OpcodeStr,
                            "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                 []>, OpSize;
// FIXME:
// There's an AssertZext in the way of writing the store pattern
// (store (i8 (trunc (X86pextrb (v16i8 VR128:$src1), imm:$src2))), addr:$dst)
}

let Predicates = [HasAVX] in
  defm VPEXTRB : SS41I_extract8<0x14, "vpextrb">, VEX;

defm PEXTRB      : SS41I_extract8<0x14, "pextrb">;


/// SS41I_extract16 - SSE 4.1 extract 16 bits to memory destination
multiclass SS41I_extract16<bits<8> opc, string OpcodeStr> {
  let isCodeGenOnly = 1, hasSideEffects = 0 in
  def rr_REV : SS4AIi8<opc, MRMDestReg, (outs GR32orGR64:$dst),
                   (ins VR128:$src1, i32i8imm:$src2),
                   !strconcat(OpcodeStr,
                   "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                   []>, OpSize;

  let neverHasSideEffects = 1, mayStore = 1 in
  def mr : SS4AIi8<opc, MRMDestMem, (outs),
                 (ins i16mem:$dst, VR128:$src1, i32i8imm:$src2),
                 !strconcat(OpcodeStr,
                  "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                 []>, OpSize;
// FIXME:
// There's an AssertZext in the way of writing the store pattern
// (store (i16 (trunc (X86pextrw (v16i8 VR128:$src1), imm:$src2))), addr:$dst)
}

let Predicates = [HasAVX] in
  defm VPEXTRW : SS41I_extract16<0x15, "vpextrw">, VEX;

defm PEXTRW      : SS41I_extract16<0x15, "pextrw">;


/// SS41I_extract32 - SSE 4.1 extract 32 bits to int reg or memory destination
multiclass SS41I_extract32<bits<8> opc, string OpcodeStr> {
  def rr : SS4AIi8<opc, MRMDestReg, (outs GR32:$dst),
                 (ins VR128:$src1, i32i8imm:$src2),
                 !strconcat(OpcodeStr,
                  "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                 [(set GR32:$dst,
                  (extractelt (v4i32 VR128:$src1), imm:$src2))]>, OpSize;
  def mr : SS4AIi8<opc, MRMDestMem, (outs),
                 (ins i32mem:$dst, VR128:$src1, i32i8imm:$src2),
                 !strconcat(OpcodeStr,
                  "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                 [(store (extractelt (v4i32 VR128:$src1), imm:$src2),
                          addr:$dst)]>, OpSize;
}

let Predicates = [HasAVX] in
  defm VPEXTRD : SS41I_extract32<0x16, "vpextrd">, VEX;

defm PEXTRD      : SS41I_extract32<0x16, "pextrd">;

/// SS41I_extract32 - SSE 4.1 extract 32 bits to int reg or memory destination
multiclass SS41I_extract64<bits<8> opc, string OpcodeStr> {
  def rr : SS4AIi8<opc, MRMDestReg, (outs GR64:$dst),
                 (ins VR128:$src1, i32i8imm:$src2),
                 !strconcat(OpcodeStr,
                  "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                 [(set GR64:$dst,
                  (extractelt (v2i64 VR128:$src1), imm:$src2))]>, OpSize, REX_W;
  def mr : SS4AIi8<opc, MRMDestMem, (outs),
                 (ins i64mem:$dst, VR128:$src1, i32i8imm:$src2),
                 !strconcat(OpcodeStr,
                  "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                 [(store (extractelt (v2i64 VR128:$src1), imm:$src2),
                          addr:$dst)]>, OpSize, REX_W;
}

let Predicates = [HasAVX] in
  defm VPEXTRQ : SS41I_extract64<0x16, "vpextrq">, VEX, VEX_W;

defm PEXTRQ      : SS41I_extract64<0x16, "pextrq">;

/// SS41I_extractf32 - SSE 4.1 extract 32 bits fp value to int reg or memory
/// destination
multiclass SS41I_extractf32<bits<8> opc, string OpcodeStr,
                            OpndItins itins = DEFAULT_ITINS> {
  def rr : SS4AIi8<opc, MRMDestReg, (outs GR32orGR64:$dst),
                 (ins VR128:$src1, i32i8imm:$src2),
                 !strconcat(OpcodeStr,
                  "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                 [(set GR32orGR64:$dst,
                    (extractelt (bc_v4i32 (v4f32 VR128:$src1)), imm:$src2))],
                    itins.rr>,
           OpSize;
  def mr : SS4AIi8<opc, MRMDestMem, (outs),
                 (ins f32mem:$dst, VR128:$src1, i32i8imm:$src2),
                 !strconcat(OpcodeStr,
                  "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                 [(store (extractelt (bc_v4i32 (v4f32 VR128:$src1)), imm:$src2),
                          addr:$dst)], itins.rm>, OpSize;
}

let ExeDomain = SSEPackedSingle in {
  let Predicates = [UseAVX] in
    defm VEXTRACTPS : SS41I_extractf32<0x17, "vextractps">, VEX;
  defm EXTRACTPS   : SS41I_extractf32<0x17, "extractps", SSE_EXTRACT_ITINS>;
}

// Also match an EXTRACTPS store when the store is done as f32 instead of i32.
def : Pat<(store (f32 (bitconvert (extractelt (bc_v4i32 (v4f32 VR128:$src1)),
                                              imm:$src2))),
                 addr:$dst),
          (VEXTRACTPSmr addr:$dst, VR128:$src1, imm:$src2)>,
          Requires<[HasAVX]>;
def : Pat<(store (f32 (bitconvert (extractelt (bc_v4i32 (v4f32 VR128:$src1)),
                                              imm:$src2))),
                 addr:$dst),
          (EXTRACTPSmr addr:$dst, VR128:$src1, imm:$src2)>,
          Requires<[UseSSE41]>;

//===----------------------------------------------------------------------===//
// SSE4.1 - Insert Instructions
//===----------------------------------------------------------------------===//

multiclass SS41I_insert8<bits<8> opc, string asm, bit Is2Addr = 1> {
  def rr : SS4AIi8<opc, MRMSrcReg, (outs VR128:$dst),
      (ins VR128:$src1, GR32orGR64:$src2, i32i8imm:$src3),
      !if(Is2Addr,
        !strconcat(asm, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
        !strconcat(asm,
                   "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}")),
      [(set VR128:$dst,
        (X86pinsrb VR128:$src1, GR32orGR64:$src2, imm:$src3))]>, OpSize;
  def rm : SS4AIi8<opc, MRMSrcMem, (outs VR128:$dst),
      (ins VR128:$src1, i8mem:$src2, i32i8imm:$src3),
      !if(Is2Addr,
        !strconcat(asm, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
        !strconcat(asm,
                   "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}")),
      [(set VR128:$dst,
        (X86pinsrb VR128:$src1, (extloadi8 addr:$src2),
                   imm:$src3))]>, OpSize;
}

let Predicates = [HasAVX] in
  defm VPINSRB : SS41I_insert8<0x20, "vpinsrb", 0>, VEX_4V;
let Constraints = "$src1 = $dst" in
  defm PINSRB  : SS41I_insert8<0x20, "pinsrb">;

multiclass SS41I_insert32<bits<8> opc, string asm, bit Is2Addr = 1> {
  def rr : SS4AIi8<opc, MRMSrcReg, (outs VR128:$dst),
      (ins VR128:$src1, GR32:$src2, i32i8imm:$src3),
      !if(Is2Addr,
        !strconcat(asm, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
        !strconcat(asm,
                   "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}")),
      [(set VR128:$dst,
        (v4i32 (insertelt VR128:$src1, GR32:$src2, imm:$src3)))]>,
      OpSize;
  def rm : SS4AIi8<opc, MRMSrcMem, (outs VR128:$dst),
      (ins VR128:$src1, i32mem:$src2, i32i8imm:$src3),
      !if(Is2Addr,
        !strconcat(asm, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
        !strconcat(asm,
                   "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}")),
      [(set VR128:$dst,
        (v4i32 (insertelt VR128:$src1, (loadi32 addr:$src2),
                          imm:$src3)))]>, OpSize;
}

let Predicates = [HasAVX] in
  defm VPINSRD : SS41I_insert32<0x22, "vpinsrd", 0>, VEX_4V;
let Constraints = "$src1 = $dst" in
  defm PINSRD : SS41I_insert32<0x22, "pinsrd">;

multiclass SS41I_insert64<bits<8> opc, string asm, bit Is2Addr = 1> {
  def rr : SS4AIi8<opc, MRMSrcReg, (outs VR128:$dst),
      (ins VR128:$src1, GR64:$src2, i32i8imm:$src3),
      !if(Is2Addr,
        !strconcat(asm, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
        !strconcat(asm,
                   "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}")),
      [(set VR128:$dst,
        (v2i64 (insertelt VR128:$src1, GR64:$src2, imm:$src3)))]>,
      OpSize;
  def rm : SS4AIi8<opc, MRMSrcMem, (outs VR128:$dst),
      (ins VR128:$src1, i64mem:$src2, i32i8imm:$src3),
      !if(Is2Addr,
        !strconcat(asm, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
        !strconcat(asm,
                   "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}")),
      [(set VR128:$dst,
        (v2i64 (insertelt VR128:$src1, (loadi64 addr:$src2),
                          imm:$src3)))]>, OpSize;
}

let Predicates = [HasAVX] in
  defm VPINSRQ : SS41I_insert64<0x22, "vpinsrq", 0>, VEX_4V, VEX_W;
let Constraints = "$src1 = $dst" in
  defm PINSRQ : SS41I_insert64<0x22, "pinsrq">, REX_W;

// insertps has a few different modes, there's the first two here below which
// are optimized inserts that won't zero arbitrary elements in the destination
// vector. The next one matches the intrinsic and could zero arbitrary elements
// in the target vector.
multiclass SS41I_insertf32<bits<8> opc, string asm, bit Is2Addr = 1,
                           OpndItins itins = DEFAULT_ITINS> {
  def rr : SS4AIi8<opc, MRMSrcReg, (outs VR128:$dst),
      (ins VR128:$src1, VR128:$src2, u32u8imm:$src3),
      !if(Is2Addr,
        !strconcat(asm, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
        !strconcat(asm,
                   "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}")),
      [(set VR128:$dst,
        (X86insrtps VR128:$src1, VR128:$src2, imm:$src3))], itins.rr>,
      OpSize;
  def rm : SS4AIi8<opc, MRMSrcMem, (outs VR128:$dst),
      (ins VR128:$src1, f32mem:$src2, u32u8imm:$src3),
      !if(Is2Addr,
        !strconcat(asm, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
        !strconcat(asm,
                   "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}")),
      [(set VR128:$dst,
        (X86insrtps VR128:$src1,
                   (v4f32 (scalar_to_vector (loadf32 addr:$src2))),
                    imm:$src3))], itins.rm>, OpSize;
}

let ExeDomain = SSEPackedSingle in {
  let Predicates = [UseAVX] in
    defm VINSERTPS : SS41I_insertf32<0x21, "vinsertps", 0>, VEX_4V;
  let Constraints = "$src1 = $dst" in
    defm INSERTPS : SS41I_insertf32<0x21, "insertps", 1, SSE_INSERT_ITINS>;
}

//===----------------------------------------------------------------------===//
// SSE4.1 - Round Instructions
//===----------------------------------------------------------------------===//

multiclass sse41_fp_unop_rm<bits<8> opcps, bits<8> opcpd, string OpcodeStr,
                            X86MemOperand x86memop, RegisterClass RC,
                            PatFrag mem_frag32, PatFrag mem_frag64,
                            Intrinsic V4F32Int, Intrinsic V2F64Int> {
let ExeDomain = SSEPackedSingle in {
  // Intrinsic operation, reg.
  // Vector intrinsic operation, reg
  def PSr : SS4AIi8<opcps, MRMSrcReg,
                    (outs RC:$dst), (ins RC:$src1, i32i8imm:$src2),
                    !strconcat(OpcodeStr,
                    "ps\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                    [(set RC:$dst, (V4F32Int RC:$src1, imm:$src2))],
                    IIC_SSE_ROUNDPS_REG>,
                    OpSize;

  // Vector intrinsic operation, mem
  def PSm : SS4AIi8<opcps, MRMSrcMem,
                    (outs RC:$dst), (ins x86memop:$src1, i32i8imm:$src2),
                    !strconcat(OpcodeStr,
                    "ps\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                    [(set RC:$dst,
                          (V4F32Int (mem_frag32 addr:$src1),imm:$src2))],
                          IIC_SSE_ROUNDPS_MEM>,
                    OpSize;
} // ExeDomain = SSEPackedSingle

let ExeDomain = SSEPackedDouble in {
  // Vector intrinsic operation, reg
  def PDr : SS4AIi8<opcpd, MRMSrcReg,
                    (outs RC:$dst), (ins RC:$src1, i32i8imm:$src2),
                    !strconcat(OpcodeStr,
                    "pd\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                    [(set RC:$dst, (V2F64Int RC:$src1, imm:$src2))],
                    IIC_SSE_ROUNDPS_REG>,
                    OpSize;

  // Vector intrinsic operation, mem
  def PDm : SS4AIi8<opcpd, MRMSrcMem,
                    (outs RC:$dst), (ins x86memop:$src1, i32i8imm:$src2),
                    !strconcat(OpcodeStr,
                    "pd\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                    [(set RC:$dst,
                          (V2F64Int (mem_frag64 addr:$src1),imm:$src2))],
                          IIC_SSE_ROUNDPS_REG>,
                    OpSize;
} // ExeDomain = SSEPackedDouble
}

multiclass sse41_fp_binop_rm<bits<8> opcss, bits<8> opcsd,
                            string OpcodeStr,
                            Intrinsic F32Int,
                            Intrinsic F64Int, bit Is2Addr = 1> {
let ExeDomain = GenericDomain in {
  // Operation, reg.
  let hasSideEffects = 0 in
  def SSr : SS4AIi8<opcss, MRMSrcReg,
      (outs FR32:$dst), (ins FR32:$src1, FR32:$src2, i32i8imm:$src3),
      !if(Is2Addr,
          !strconcat(OpcodeStr,
              "ss\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
          !strconcat(OpcodeStr,
              "ss\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}")),
      []>, OpSize;

  // Intrinsic operation, reg.
  def SSr_Int : SS4AIi8<opcss, MRMSrcReg,
        (outs VR128:$dst), (ins VR128:$src1, VR128:$src2, i32i8imm:$src3),
        !if(Is2Addr,
            !strconcat(OpcodeStr,
                "ss\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
            !strconcat(OpcodeStr,
                "ss\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}")),
        [(set VR128:$dst, (F32Int VR128:$src1, VR128:$src2, imm:$src3))]>,
        OpSize;

  // Intrinsic operation, mem.
  def SSm : SS4AIi8<opcss, MRMSrcMem,
        (outs VR128:$dst), (ins VR128:$src1, ssmem:$src2, i32i8imm:$src3),
        !if(Is2Addr,
            !strconcat(OpcodeStr,
                "ss\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
            !strconcat(OpcodeStr,
                "ss\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}")),
        [(set VR128:$dst,
             (F32Int VR128:$src1, sse_load_f32:$src2, imm:$src3))]>,
        OpSize;

  // Operation, reg.
  let hasSideEffects = 0 in
  def SDr : SS4AIi8<opcsd, MRMSrcReg,
        (outs FR64:$dst), (ins FR64:$src1, FR64:$src2, i32i8imm:$src3),
        !if(Is2Addr,
            !strconcat(OpcodeStr,
                "sd\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
            !strconcat(OpcodeStr,
                "sd\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}")),
        []>, OpSize;

  // Intrinsic operation, reg.
  def SDr_Int : SS4AIi8<opcsd, MRMSrcReg,
        (outs VR128:$dst), (ins VR128:$src1, VR128:$src2, i32i8imm:$src3),
        !if(Is2Addr,
            !strconcat(OpcodeStr,
                "sd\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
            !strconcat(OpcodeStr,
                "sd\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}")),
        [(set VR128:$dst, (F64Int VR128:$src1, VR128:$src2, imm:$src3))]>,
        OpSize;

  // Intrinsic operation, mem.
  def SDm : SS4AIi8<opcsd, MRMSrcMem,
        (outs VR128:$dst), (ins VR128:$src1, sdmem:$src2, i32i8imm:$src3),
        !if(Is2Addr,
            !strconcat(OpcodeStr,
                "sd\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
            !strconcat(OpcodeStr,
                "sd\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}")),
        [(set VR128:$dst,
              (F64Int VR128:$src1, sse_load_f64:$src2, imm:$src3))]>,
        OpSize;
} // ExeDomain = GenericDomain
}

// FP round - roundss, roundps, roundsd, roundpd
let Predicates = [HasAVX] in {
  // Intrinsic form
  defm VROUND  : sse41_fp_unop_rm<0x08, 0x09, "vround", f128mem, VR128,
                                  loadv4f32, loadv2f64,
                                  int_x86_sse41_round_ps,
                                  int_x86_sse41_round_pd>, VEX;
  defm VROUNDY : sse41_fp_unop_rm<0x08, 0x09, "vround", f256mem, VR256,
                                  loadv8f32, loadv4f64,
                                  int_x86_avx_round_ps_256,
                                  int_x86_avx_round_pd_256>, VEX, VEX_L;
  defm VROUND  : sse41_fp_binop_rm<0x0A, 0x0B, "vround",
                                  int_x86_sse41_round_ss,
                                  int_x86_sse41_round_sd, 0>, VEX_4V, VEX_LIG;

  def : Pat<(ffloor FR32:$src),
            (VROUNDSSr (f32 (IMPLICIT_DEF)), FR32:$src, (i32 0x1))>;
  def : Pat<(f64 (ffloor FR64:$src)),
            (VROUNDSDr (f64 (IMPLICIT_DEF)), FR64:$src, (i32 0x1))>;
  def : Pat<(f32 (fnearbyint FR32:$src)),
            (VROUNDSSr (f32 (IMPLICIT_DEF)), FR32:$src, (i32 0xC))>;
  def : Pat<(f64 (fnearbyint FR64:$src)),
            (VROUNDSDr (f64 (IMPLICIT_DEF)), FR64:$src, (i32 0xC))>;
  def : Pat<(f32 (fceil FR32:$src)),
            (VROUNDSSr (f32 (IMPLICIT_DEF)), FR32:$src, (i32 0x2))>;
  def : Pat<(f64 (fceil FR64:$src)),
            (VROUNDSDr (f64 (IMPLICIT_DEF)), FR64:$src, (i32 0x2))>;
  def : Pat<(f32 (frint FR32:$src)),
            (VROUNDSSr (f32 (IMPLICIT_DEF)), FR32:$src, (i32 0x4))>;
  def : Pat<(f64 (frint FR64:$src)),
            (VROUNDSDr (f64 (IMPLICIT_DEF)), FR64:$src, (i32 0x4))>;
  def : Pat<(f32 (ftrunc FR32:$src)),
            (VROUNDSSr (f32 (IMPLICIT_DEF)), FR32:$src, (i32 0x3))>;
  def : Pat<(f64 (ftrunc FR64:$src)),
            (VROUNDSDr (f64 (IMPLICIT_DEF)), FR64:$src, (i32 0x3))>;

  def : Pat<(v4f32 (ffloor VR128:$src)),
            (VROUNDPSr VR128:$src, (i32 0x1))>;
  def : Pat<(v4f32 (fnearbyint VR128:$src)),
            (VROUNDPSr VR128:$src, (i32 0xC))>;
  def : Pat<(v4f32 (fceil VR128:$src)),
            (VROUNDPSr VR128:$src, (i32 0x2))>;
  def : Pat<(v4f32 (frint VR128:$src)),
            (VROUNDPSr VR128:$src, (i32 0x4))>;
  def : Pat<(v4f32 (ftrunc VR128:$src)),
            (VROUNDPSr VR128:$src, (i32 0x3))>;

  def : Pat<(v2f64 (ffloor VR128:$src)),
            (VROUNDPDr VR128:$src, (i32 0x1))>;
  def : Pat<(v2f64 (fnearbyint VR128:$src)),
            (VROUNDPDr VR128:$src, (i32 0xC))>;
  def : Pat<(v2f64 (fceil VR128:$src)),
            (VROUNDPDr VR128:$src, (i32 0x2))>;
  def : Pat<(v2f64 (frint VR128:$src)),
            (VROUNDPDr VR128:$src, (i32 0x4))>;
  def : Pat<(v2f64 (ftrunc VR128:$src)),
            (VROUNDPDr VR128:$src, (i32 0x3))>;

  def : Pat<(v8f32 (ffloor VR256:$src)),
            (VROUNDYPSr VR256:$src, (i32 0x1))>;
  def : Pat<(v8f32 (fnearbyint VR256:$src)),
            (VROUNDYPSr VR256:$src, (i32 0xC))>;
  def : Pat<(v8f32 (fceil VR256:$src)),
            (VROUNDYPSr VR256:$src, (i32 0x2))>;
  def : Pat<(v8f32 (frint VR256:$src)),
            (VROUNDYPSr VR256:$src, (i32 0x4))>;
  def : Pat<(v8f32 (ftrunc VR256:$src)),
            (VROUNDYPSr VR256:$src, (i32 0x3))>;

  def : Pat<(v4f64 (ffloor VR256:$src)),
            (VROUNDYPDr VR256:$src, (i32 0x1))>;
  def : Pat<(v4f64 (fnearbyint VR256:$src)),
            (VROUNDYPDr VR256:$src, (i32 0xC))>;
  def : Pat<(v4f64 (fceil VR256:$src)),
            (VROUNDYPDr VR256:$src, (i32 0x2))>;
  def : Pat<(v4f64 (frint VR256:$src)),
            (VROUNDYPDr VR256:$src, (i32 0x4))>;
  def : Pat<(v4f64 (ftrunc VR256:$src)),
            (VROUNDYPDr VR256:$src, (i32 0x3))>;
}

defm ROUND  : sse41_fp_unop_rm<0x08, 0x09, "round", f128mem, VR128,
                               memopv4f32, memopv2f64,
                               int_x86_sse41_round_ps, int_x86_sse41_round_pd>;
let Constraints = "$src1 = $dst" in
defm ROUND  : sse41_fp_binop_rm<0x0A, 0x0B, "round",
                               int_x86_sse41_round_ss, int_x86_sse41_round_sd>;

let Predicates = [UseSSE41] in {
  def : Pat<(ffloor FR32:$src),
            (ROUNDSSr (f32 (IMPLICIT_DEF)), FR32:$src, (i32 0x1))>;
  def : Pat<(f64 (ffloor FR64:$src)),
            (ROUNDSDr (f64 (IMPLICIT_DEF)), FR64:$src, (i32 0x1))>;
  def : Pat<(f32 (fnearbyint FR32:$src)),
            (ROUNDSSr (f32 (IMPLICIT_DEF)), FR32:$src, (i32 0xC))>;
  def : Pat<(f64 (fnearbyint FR64:$src)),
            (ROUNDSDr (f64 (IMPLICIT_DEF)), FR64:$src, (i32 0xC))>;
  def : Pat<(f32 (fceil FR32:$src)),
            (ROUNDSSr (f32 (IMPLICIT_DEF)), FR32:$src, (i32 0x2))>;
  def : Pat<(f64 (fceil FR64:$src)),
            (ROUNDSDr (f64 (IMPLICIT_DEF)), FR64:$src, (i32 0x2))>;
  def : Pat<(f32 (frint FR32:$src)),
            (ROUNDSSr (f32 (IMPLICIT_DEF)), FR32:$src, (i32 0x4))>;
  def : Pat<(f64 (frint FR64:$src)),
            (ROUNDSDr (f64 (IMPLICIT_DEF)), FR64:$src, (i32 0x4))>;
  def : Pat<(f32 (ftrunc FR32:$src)),
            (ROUNDSSr (f32 (IMPLICIT_DEF)), FR32:$src, (i32 0x3))>;
  def : Pat<(f64 (ftrunc FR64:$src)),
            (ROUNDSDr (f64 (IMPLICIT_DEF)), FR64:$src, (i32 0x3))>;

  def : Pat<(v4f32 (ffloor VR128:$src)),
            (ROUNDPSr VR128:$src, (i32 0x1))>;
  def : Pat<(v4f32 (fnearbyint VR128:$src)),
            (ROUNDPSr VR128:$src, (i32 0xC))>;
  def : Pat<(v4f32 (fceil VR128:$src)),
            (ROUNDPSr VR128:$src, (i32 0x2))>;
  def : Pat<(v4f32 (frint VR128:$src)),
            (ROUNDPSr VR128:$src, (i32 0x4))>;
  def : Pat<(v4f32 (ftrunc VR128:$src)),
            (ROUNDPSr VR128:$src, (i32 0x3))>;

  def : Pat<(v2f64 (ffloor VR128:$src)),
            (ROUNDPDr VR128:$src, (i32 0x1))>;
  def : Pat<(v2f64 (fnearbyint VR128:$src)),
            (ROUNDPDr VR128:$src, (i32 0xC))>;
  def : Pat<(v2f64 (fceil VR128:$src)),
            (ROUNDPDr VR128:$src, (i32 0x2))>;
  def : Pat<(v2f64 (frint VR128:$src)),
            (ROUNDPDr VR128:$src, (i32 0x4))>;
  def : Pat<(v2f64 (ftrunc VR128:$src)),
            (ROUNDPDr VR128:$src, (i32 0x3))>;
}

//===----------------------------------------------------------------------===//
// SSE4.1 - Packed Bit Test
//===----------------------------------------------------------------------===//

// ptest instruction we'll lower to this in X86ISelLowering primarily from
// the intel intrinsic that corresponds to this.
let Defs = [EFLAGS], Predicates = [HasAVX] in {
def VPTESTrr  : SS48I<0x17, MRMSrcReg, (outs), (ins VR128:$src1, VR128:$src2),
                "vptest\t{$src2, $src1|$src1, $src2}",
                [(set EFLAGS, (X86ptest VR128:$src1, (v2i64 VR128:$src2)))]>,
                OpSize, VEX;
def VPTESTrm  : SS48I<0x17, MRMSrcMem, (outs), (ins VR128:$src1, f128mem:$src2),
                "vptest\t{$src2, $src1|$src1, $src2}",
                [(set EFLAGS,(X86ptest VR128:$src1, (loadv2i64 addr:$src2)))]>,
                OpSize, VEX;

def VPTESTYrr : SS48I<0x17, MRMSrcReg, (outs), (ins VR256:$src1, VR256:$src2),
                "vptest\t{$src2, $src1|$src1, $src2}",
                [(set EFLAGS, (X86ptest VR256:$src1, (v4i64 VR256:$src2)))]>,
                OpSize, VEX, VEX_L;
def VPTESTYrm : SS48I<0x17, MRMSrcMem, (outs), (ins VR256:$src1, i256mem:$src2),
                "vptest\t{$src2, $src1|$src1, $src2}",
                [(set EFLAGS,(X86ptest VR256:$src1, (loadv4i64 addr:$src2)))]>,
                OpSize, VEX, VEX_L;
}

let Defs = [EFLAGS] in {
def PTESTrr : SS48I<0x17, MRMSrcReg, (outs), (ins VR128:$src1, VR128:$src2),
              "ptest\t{$src2, $src1|$src1, $src2}",
              [(set EFLAGS, (X86ptest VR128:$src1, (v2i64 VR128:$src2)))]>,
              OpSize;
def PTESTrm : SS48I<0x17, MRMSrcMem, (outs), (ins VR128:$src1, f128mem:$src2),
              "ptest\t{$src2, $src1|$src1, $src2}",
              [(set EFLAGS, (X86ptest VR128:$src1, (memopv2i64 addr:$src2)))]>,
              OpSize;
}

// The bit test instructions below are AVX only
multiclass avx_bittest<bits<8> opc, string OpcodeStr, RegisterClass RC,
                       X86MemOperand x86memop, PatFrag mem_frag, ValueType vt> {
  def rr : SS48I<opc, MRMSrcReg, (outs), (ins RC:$src1, RC:$src2),
            !strconcat(OpcodeStr, "\t{$src2, $src1|$src1, $src2}"),
            [(set EFLAGS, (X86testp RC:$src1, (vt RC:$src2)))]>, OpSize, VEX;
  def rm : SS48I<opc, MRMSrcMem, (outs), (ins RC:$src1, x86memop:$src2),
            !strconcat(OpcodeStr, "\t{$src2, $src1|$src1, $src2}"),
            [(set EFLAGS, (X86testp RC:$src1, (mem_frag addr:$src2)))]>,
            OpSize, VEX;
}

let Defs = [EFLAGS], Predicates = [HasAVX] in {
let ExeDomain = SSEPackedSingle in {
defm VTESTPS  : avx_bittest<0x0E, "vtestps", VR128, f128mem, loadv4f32, v4f32>;
defm VTESTPSY : avx_bittest<0x0E, "vtestps", VR256, f256mem, loadv8f32, v8f32>,
                            VEX_L;
}
let ExeDomain = SSEPackedDouble in {
defm VTESTPD  : avx_bittest<0x0F, "vtestpd", VR128, f128mem, loadv2f64, v2f64>;
defm VTESTPDY : avx_bittest<0x0F, "vtestpd", VR256, f256mem, loadv4f64, v4f64>,
                            VEX_L;
}
}

//===----------------------------------------------------------------------===//
// SSE4.1 - Misc Instructions
//===----------------------------------------------------------------------===//

let Defs = [EFLAGS], Predicates = [HasPOPCNT] in {
  def POPCNT16rr : I<0xB8, MRMSrcReg, (outs GR16:$dst), (ins GR16:$src),
                     "popcnt{w}\t{$src, $dst|$dst, $src}",
                     [(set GR16:$dst, (ctpop GR16:$src)), (implicit EFLAGS)],
                     IIC_SSE_POPCNT_RR>,
                     OpSize, XS;
  def POPCNT16rm : I<0xB8, MRMSrcMem, (outs GR16:$dst), (ins i16mem:$src),
                     "popcnt{w}\t{$src, $dst|$dst, $src}",
                     [(set GR16:$dst, (ctpop (loadi16 addr:$src))),
                      (implicit EFLAGS)], IIC_SSE_POPCNT_RM>, OpSize, XS;

  def POPCNT32rr : I<0xB8, MRMSrcReg, (outs GR32:$dst), (ins GR32:$src),
                     "popcnt{l}\t{$src, $dst|$dst, $src}",
                     [(set GR32:$dst, (ctpop GR32:$src)), (implicit EFLAGS)],
                     IIC_SSE_POPCNT_RR>,
                     XS;
  def POPCNT32rm : I<0xB8, MRMSrcMem, (outs GR32:$dst), (ins i32mem:$src),
                     "popcnt{l}\t{$src, $dst|$dst, $src}",
                     [(set GR32:$dst, (ctpop (loadi32 addr:$src))),
                      (implicit EFLAGS)], IIC_SSE_POPCNT_RM>, XS;

  def POPCNT64rr : RI<0xB8, MRMSrcReg, (outs GR64:$dst), (ins GR64:$src),
                      "popcnt{q}\t{$src, $dst|$dst, $src}",
                      [(set GR64:$dst, (ctpop GR64:$src)), (implicit EFLAGS)],
                      IIC_SSE_POPCNT_RR>,
                      XS;
  def POPCNT64rm : RI<0xB8, MRMSrcMem, (outs GR64:$dst), (ins i64mem:$src),
                      "popcnt{q}\t{$src, $dst|$dst, $src}",
                      [(set GR64:$dst, (ctpop (loadi64 addr:$src))),
                       (implicit EFLAGS)], IIC_SSE_POPCNT_RM>, XS;
}



// SS41I_unop_rm_int_v16 - SSE 4.1 unary operator whose type is v8i16.
multiclass SS41I_unop_rm_int_v16<bits<8> opc, string OpcodeStr,
                                 Intrinsic IntId128> {
  def rr128 : SS48I<opc, MRMSrcReg, (outs VR128:$dst),
                    (ins VR128:$src),
                    !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                    [(set VR128:$dst, (IntId128 VR128:$src))]>, OpSize;
  def rm128 : SS48I<opc, MRMSrcMem, (outs VR128:$dst),
                     (ins i128mem:$src),
                     !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                     [(set VR128:$dst,
                       (IntId128
                        (bitconvert (memopv2i64 addr:$src))))]>, OpSize;
}

let Predicates = [HasAVX] in
defm VPHMINPOSUW : SS41I_unop_rm_int_v16 <0x41, "vphminposuw",
                                         int_x86_sse41_phminposuw>, VEX;
defm PHMINPOSUW : SS41I_unop_rm_int_v16 <0x41, "phminposuw",
                                         int_x86_sse41_phminposuw>;

/// SS41I_binop_rm_int - Simple SSE 4.1 binary operator
multiclass SS41I_binop_rm_int<bits<8> opc, string OpcodeStr,
                              Intrinsic IntId128, bit Is2Addr = 1,
                              OpndItins itins = DEFAULT_ITINS> {
  let isCommutable = 1 in
  def rr : SS48I<opc, MRMSrcReg, (outs VR128:$dst),
       (ins VR128:$src1, VR128:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set VR128:$dst, (IntId128 VR128:$src1, VR128:$src2))],
        itins.rr>, OpSize;
  def rm : SS48I<opc, MRMSrcMem, (outs VR128:$dst),
       (ins VR128:$src1, i128mem:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set VR128:$dst,
         (IntId128 VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))))],
          itins.rm>, OpSize;
}

/// SS41I_binop_rm_int_y - Simple SSE 4.1 binary operator
multiclass SS41I_binop_rm_int_y<bits<8> opc, string OpcodeStr,
                                Intrinsic IntId256> {
  let isCommutable = 1 in
  def Yrr : SS48I<opc, MRMSrcReg, (outs VR256:$dst),
       (ins VR256:$src1, VR256:$src2),
       !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
       [(set VR256:$dst, (IntId256 VR256:$src1, VR256:$src2))]>, OpSize;
  def Yrm : SS48I<opc, MRMSrcMem, (outs VR256:$dst),
       (ins VR256:$src1, i256mem:$src2),
       !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
       [(set VR256:$dst,
         (IntId256 VR256:$src1,
          (bitconvert (loadv4i64 addr:$src2))))]>, OpSize;
}


/// SS48I_binop_rm - Simple SSE41 binary operator.
multiclass SS48I_binop_rm<bits<8> opc, string OpcodeStr, SDNode OpNode,
                          ValueType OpVT, RegisterClass RC, PatFrag memop_frag,
                          X86MemOperand x86memop, bit Is2Addr = 1,
                          OpndItins itins = DEFAULT_ITINS> {
  let isCommutable = 1 in
  def rr : SS48I<opc, MRMSrcReg, (outs RC:$dst),
       (ins RC:$src1, RC:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set RC:$dst, (OpVT (OpNode RC:$src1, RC:$src2)))]>, OpSize;
  def rm : SS48I<opc, MRMSrcMem, (outs RC:$dst),
       (ins RC:$src1, x86memop:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set RC:$dst,
         (OpVT (OpNode RC:$src1,
          (bitconvert (memop_frag addr:$src2)))))]>, OpSize;
}

let Predicates = [HasAVX] in {
  let isCommutable = 0 in
  defm VPACKUSDW : SS41I_binop_rm_int<0x2B, "vpackusdw", int_x86_sse41_packusdw,
                                                         0>, VEX_4V;
  defm VPMINSB   : SS48I_binop_rm<0x38, "vpminsb", X86smin, v16i8, VR128,
                                  loadv2i64, i128mem, 0>, VEX_4V;
  defm VPMINSD   : SS48I_binop_rm<0x39, "vpminsd", X86smin, v4i32, VR128,
                                  loadv2i64, i128mem, 0>, VEX_4V;
  defm VPMINUD   : SS48I_binop_rm<0x3B, "vpminud", X86umin, v4i32, VR128,
                                  loadv2i64, i128mem, 0>, VEX_4V;
  defm VPMINUW   : SS48I_binop_rm<0x3A, "vpminuw", X86umin, v8i16, VR128,
                                  loadv2i64, i128mem, 0>, VEX_4V;
  defm VPMAXSB   : SS48I_binop_rm<0x3C, "vpmaxsb", X86smax, v16i8, VR128,
                                  loadv2i64, i128mem, 0>, VEX_4V;
  defm VPMAXSD   : SS48I_binop_rm<0x3D, "vpmaxsd", X86smax, v4i32, VR128,
                                  loadv2i64, i128mem, 0>, VEX_4V;
  defm VPMAXUD   : SS48I_binop_rm<0x3F, "vpmaxud", X86umax, v4i32, VR128,
                                  loadv2i64, i128mem, 0>, VEX_4V;
  defm VPMAXUW   : SS48I_binop_rm<0x3E, "vpmaxuw", X86umax, v8i16, VR128,
                                  loadv2i64, i128mem, 0>, VEX_4V;
  defm VPMULDQ   : SS41I_binop_rm_int<0x28, "vpmuldq",   int_x86_sse41_pmuldq,
                                                         0>, VEX_4V;
}

let Predicates = [HasAVX2] in {
  let isCommutable = 0 in
  defm VPACKUSDW : SS41I_binop_rm_int_y<0x2B, "vpackusdw",
                                        int_x86_avx2_packusdw>, VEX_4V, VEX_L;
  defm VPMINSBY  : SS48I_binop_rm<0x38, "vpminsb", X86smin, v32i8, VR256,
                                  loadv4i64, i256mem, 0>, VEX_4V, VEX_L;
  defm VPMINSDY  : SS48I_binop_rm<0x39, "vpminsd", X86smin, v8i32, VR256,
                                  loadv4i64, i256mem, 0>, VEX_4V, VEX_L;
  defm VPMINUDY  : SS48I_binop_rm<0x3B, "vpminud", X86umin, v8i32, VR256,
                                  loadv4i64, i256mem, 0>, VEX_4V, VEX_L;
  defm VPMINUWY  : SS48I_binop_rm<0x3A, "vpminuw", X86umin, v16i16, VR256,
                                  loadv4i64, i256mem, 0>, VEX_4V, VEX_L;
  defm VPMAXSBY  : SS48I_binop_rm<0x3C, "vpmaxsb", X86smax, v32i8, VR256,
                                  loadv4i64, i256mem, 0>, VEX_4V, VEX_L;
  defm VPMAXSDY  : SS48I_binop_rm<0x3D, "vpmaxsd", X86smax, v8i32, VR256,
                                  loadv4i64, i256mem, 0>, VEX_4V, VEX_L;
  defm VPMAXUDY  : SS48I_binop_rm<0x3F, "vpmaxud", X86umax, v8i32, VR256,
                                  loadv4i64, i256mem, 0>, VEX_4V, VEX_L;
  defm VPMAXUWY  : SS48I_binop_rm<0x3E, "vpmaxuw", X86umax, v16i16, VR256,
                                  loadv4i64, i256mem, 0>, VEX_4V, VEX_L;
  defm VPMULDQ   : SS41I_binop_rm_int_y<0x28, "vpmuldq",
                                        int_x86_avx2_pmul_dq>, VEX_4V, VEX_L;
}

let Constraints = "$src1 = $dst" in {
  let isCommutable = 0 in
  defm PACKUSDW : SS41I_binop_rm_int<0x2B, "packusdw", int_x86_sse41_packusdw>;
  defm PMINSB   : SS48I_binop_rm<0x38, "pminsb", X86smin, v16i8, VR128,
                                 memopv2i64, i128mem, 1, SSE_INTALU_ITINS_P>;
  defm PMINSD   : SS48I_binop_rm<0x39, "pminsd", X86smin, v4i32, VR128,
                                 memopv2i64, i128mem, 1, SSE_INTALU_ITINS_P>;
  defm PMINUD   : SS48I_binop_rm<0x3B, "pminud", X86umin, v4i32, VR128,
                                 memopv2i64, i128mem, 1, SSE_INTALU_ITINS_P>;
  defm PMINUW   : SS48I_binop_rm<0x3A, "pminuw", X86umin, v8i16, VR128,
                                 memopv2i64, i128mem, 1, SSE_INTALU_ITINS_P>;
  defm PMAXSB   : SS48I_binop_rm<0x3C, "pmaxsb", X86smax, v16i8, VR128,
                                 memopv2i64, i128mem, 1, SSE_INTALU_ITINS_P>;
  defm PMAXSD   : SS48I_binop_rm<0x3D, "pmaxsd", X86smax, v4i32, VR128,
                                 memopv2i64, i128mem, 1, SSE_INTALU_ITINS_P>;
  defm PMAXUD   : SS48I_binop_rm<0x3F, "pmaxud", X86umax, v4i32, VR128,
                                 memopv2i64, i128mem, 1, SSE_INTALU_ITINS_P>;
  defm PMAXUW   : SS48I_binop_rm<0x3E, "pmaxuw", X86umax, v8i16, VR128,
                                 memopv2i64, i128mem, 1, SSE_INTALU_ITINS_P>;
  defm PMULDQ   : SS41I_binop_rm_int<0x28, "pmuldq",   int_x86_sse41_pmuldq,
                                     1, SSE_INTMUL_ITINS_P>;
}

let Predicates = [HasAVX] in {
  defm VPMULLD  : SS48I_binop_rm<0x40, "vpmulld", mul, v4i32, VR128,
                                memopv2i64, i128mem, 0>, VEX_4V;
  defm VPCMPEQQ : SS48I_binop_rm<0x29, "vpcmpeqq", X86pcmpeq, v2i64, VR128,
                                 memopv2i64, i128mem, 0>, VEX_4V;
}
let Predicates = [HasAVX2] in {
  defm VPMULLDY  : SS48I_binop_rm<0x40, "vpmulld", mul, v8i32, VR256,
                                  memopv4i64, i256mem, 0>, VEX_4V, VEX_L;
  defm VPCMPEQQY : SS48I_binop_rm<0x29, "vpcmpeqq", X86pcmpeq, v4i64, VR256,
                                  memopv4i64, i256mem, 0>, VEX_4V, VEX_L;
}

let Constraints = "$src1 = $dst" in {
  defm PMULLD  : SS48I_binop_rm<0x40, "pmulld", mul, v4i32, VR128,
                                memopv2i64, i128mem, 1, SSE_PMULLD_ITINS>;
  defm PCMPEQQ : SS48I_binop_rm<0x29, "pcmpeqq", X86pcmpeq, v2i64, VR128,
                                memopv2i64, i128mem, 1, SSE_INTALUQ_ITINS_P>;
}

/// SS41I_binop_rmi_int - SSE 4.1 binary operator with 8-bit immediate
multiclass SS41I_binop_rmi_int<bits<8> opc, string OpcodeStr,
                 Intrinsic IntId, RegisterClass RC, PatFrag memop_frag,
                 X86MemOperand x86memop, bit Is2Addr = 1,
                 OpndItins itins = DEFAULT_ITINS> {
  let isCommutable = 1 in
  def rri : SS4AIi8<opc, MRMSrcReg, (outs RC:$dst),
        (ins RC:$src1, RC:$src2, u32u8imm:$src3),
        !if(Is2Addr,
            !strconcat(OpcodeStr,
                "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
            !strconcat(OpcodeStr,
                "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}")),
        [(set RC:$dst, (IntId RC:$src1, RC:$src2, imm:$src3))], itins.rr>,
        OpSize;
  def rmi : SS4AIi8<opc, MRMSrcMem, (outs RC:$dst),
        (ins RC:$src1, x86memop:$src2, u32u8imm:$src3),
        !if(Is2Addr,
            !strconcat(OpcodeStr,
                "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
            !strconcat(OpcodeStr,
                "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}")),
        [(set RC:$dst,
          (IntId RC:$src1,
           (bitconvert (memop_frag addr:$src2)), imm:$src3))], itins.rm>,
        OpSize;
}

let Predicates = [HasAVX] in {
  let isCommutable = 0 in {
    let ExeDomain = SSEPackedSingle in {
    defm VBLENDPS : SS41I_binop_rmi_int<0x0C, "vblendps", int_x86_sse41_blendps,
                                        VR128, loadv4f32, f128mem, 0>, VEX_4V;
    defm VBLENDPSY : SS41I_binop_rmi_int<0x0C, "vblendps",
                                    int_x86_avx_blend_ps_256, VR256, loadv8f32,
                                    f256mem, 0>, VEX_4V, VEX_L;
    }
    let ExeDomain = SSEPackedDouble in {
    defm VBLENDPD : SS41I_binop_rmi_int<0x0D, "vblendpd", int_x86_sse41_blendpd,
                                        VR128, loadv2f64, f128mem, 0>, VEX_4V;
    defm VBLENDPDY : SS41I_binop_rmi_int<0x0D, "vblendpd",
                                     int_x86_avx_blend_pd_256,VR256, loadv4f64,
                                     f256mem, 0>, VEX_4V, VEX_L;
    }
  defm VPBLENDW : SS41I_binop_rmi_int<0x0E, "vpblendw", int_x86_sse41_pblendw,
                                      VR128, loadv2i64, i128mem, 0>, VEX_4V;
  defm VMPSADBW : SS41I_binop_rmi_int<0x42, "vmpsadbw", int_x86_sse41_mpsadbw,
                                      VR128, loadv2i64, i128mem, 0>, VEX_4V;
  }
  let ExeDomain = SSEPackedSingle in
  defm VDPPS : SS41I_binop_rmi_int<0x40, "vdpps", int_x86_sse41_dpps,
                                   VR128, loadv4f32, f128mem, 0>, VEX_4V;
  let ExeDomain = SSEPackedDouble in
  defm VDPPD : SS41I_binop_rmi_int<0x41, "vdppd", int_x86_sse41_dppd,
                                   VR128, loadv2f64, f128mem, 0>, VEX_4V;
  let ExeDomain = SSEPackedSingle in
  defm VDPPSY : SS41I_binop_rmi_int<0x40, "vdpps", int_x86_avx_dp_ps_256,
                                  VR256, loadv8f32, i256mem, 0>, VEX_4V, VEX_L;
}

let Predicates = [HasAVX2] in {
  let isCommutable = 0 in {
  defm VPBLENDWY : SS41I_binop_rmi_int<0x0E, "vpblendw", int_x86_avx2_pblendw,
                                  VR256, loadv4i64, i256mem, 0>, VEX_4V, VEX_L;
  defm VMPSADBWY : SS41I_binop_rmi_int<0x42, "vmpsadbw", int_x86_avx2_mpsadbw,
                                  VR256, loadv4i64, i256mem, 0>, VEX_4V, VEX_L;
  }
}

let Constraints = "$src1 = $dst" in {
  let isCommutable = 0 in {
  let ExeDomain = SSEPackedSingle in
  defm BLENDPS : SS41I_binop_rmi_int<0x0C, "blendps", int_x86_sse41_blendps,
                                     VR128, memopv4f32, f128mem,
                                     1, SSE_INTALU_ITINS_P>;
  let ExeDomain = SSEPackedDouble in
  defm BLENDPD : SS41I_binop_rmi_int<0x0D, "blendpd", int_x86_sse41_blendpd,
                                     VR128, memopv2f64, f128mem,
                                     1, SSE_INTALU_ITINS_P>;
  defm PBLENDW : SS41I_binop_rmi_int<0x0E, "pblendw", int_x86_sse41_pblendw,
                                     VR128, memopv2i64, i128mem,
                                     1, SSE_INTALU_ITINS_P>;
  defm MPSADBW : SS41I_binop_rmi_int<0x42, "mpsadbw", int_x86_sse41_mpsadbw,
                                     VR128, memopv2i64, i128mem,
                                     1, SSE_INTMUL_ITINS_P>;
  }
  let ExeDomain = SSEPackedSingle in
  defm DPPS : SS41I_binop_rmi_int<0x40, "dpps", int_x86_sse41_dpps,
                                  VR128, memopv4f32, f128mem, 1,
                                  SSE_DPPS_ITINS>;
  let ExeDomain = SSEPackedDouble in
  defm DPPD : SS41I_binop_rmi_int<0x41, "dppd", int_x86_sse41_dppd,
                                  VR128, memopv2f64, f128mem, 1,
                                  SSE_DPPD_ITINS>;
}

/// SS41I_quaternary_int_avx - AVX SSE 4.1 with 4 operators
multiclass SS41I_quaternary_int_avx<bits<8> opc, string OpcodeStr,
                                    RegisterClass RC, X86MemOperand x86memop,
                                    PatFrag mem_frag, Intrinsic IntId> {
  def rr : Ii8<opc, MRMSrcReg, (outs RC:$dst),
                  (ins RC:$src1, RC:$src2, RC:$src3),
                  !strconcat(OpcodeStr,
                    "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
                  [(set RC:$dst, (IntId RC:$src1, RC:$src2, RC:$src3))],
                  NoItinerary, SSEPackedInt>, OpSize, TA, VEX_4V, VEX_I8IMM;

  def rm : Ii8<opc, MRMSrcMem, (outs RC:$dst),
                  (ins RC:$src1, x86memop:$src2, RC:$src3),
                  !strconcat(OpcodeStr,
                    "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
                  [(set RC:$dst,
                        (IntId RC:$src1, (bitconvert (mem_frag addr:$src2)),
                               RC:$src3))],
                  NoItinerary, SSEPackedInt>, OpSize, TA, VEX_4V, VEX_I8IMM;
}

let Predicates = [HasAVX] in {
let ExeDomain = SSEPackedDouble in {
defm VBLENDVPD  : SS41I_quaternary_int_avx<0x4B, "vblendvpd", VR128, f128mem,
                                           loadv2f64, int_x86_sse41_blendvpd>;
defm VBLENDVPDY : SS41I_quaternary_int_avx<0x4B, "vblendvpd", VR256, f256mem,
                                  loadv4f64, int_x86_avx_blendv_pd_256>, VEX_L;
} // ExeDomain = SSEPackedDouble
let ExeDomain = SSEPackedSingle in {
defm VBLENDVPS  : SS41I_quaternary_int_avx<0x4A, "vblendvps", VR128, f128mem,
                                           loadv4f32, int_x86_sse41_blendvps>;
defm VBLENDVPSY : SS41I_quaternary_int_avx<0x4A, "vblendvps", VR256, f256mem,
                                  loadv8f32, int_x86_avx_blendv_ps_256>, VEX_L;
} // ExeDomain = SSEPackedSingle
defm VPBLENDVB  : SS41I_quaternary_int_avx<0x4C, "vpblendvb", VR128, i128mem,
                                           loadv2i64, int_x86_sse41_pblendvb>;
}

let Predicates = [HasAVX2] in {
defm VPBLENDVBY : SS41I_quaternary_int_avx<0x4C, "vpblendvb", VR256, i256mem,
                                      loadv4i64, int_x86_avx2_pblendvb>, VEX_L;
}

let Predicates = [HasAVX] in {
  def : Pat<(v16i8 (vselect (v16i8 VR128:$mask), (v16i8 VR128:$src1),
                            (v16i8 VR128:$src2))),
            (VPBLENDVBrr VR128:$src2, VR128:$src1, VR128:$mask)>;
  def : Pat<(v4i32 (vselect (v4i32 VR128:$mask), (v4i32 VR128:$src1),
                            (v4i32 VR128:$src2))),
            (VBLENDVPSrr VR128:$src2, VR128:$src1, VR128:$mask)>;
  def : Pat<(v4f32 (vselect (v4i32 VR128:$mask), (v4f32 VR128:$src1),
                            (v4f32 VR128:$src2))),
            (VBLENDVPSrr VR128:$src2, VR128:$src1, VR128:$mask)>;
  def : Pat<(v2i64 (vselect (v2i64 VR128:$mask), (v2i64 VR128:$src1),
                            (v2i64 VR128:$src2))),
            (VBLENDVPDrr VR128:$src2, VR128:$src1, VR128:$mask)>;
  def : Pat<(v2f64 (vselect (v2i64 VR128:$mask), (v2f64 VR128:$src1),
                            (v2f64 VR128:$src2))),
            (VBLENDVPDrr VR128:$src2, VR128:$src1, VR128:$mask)>;
  def : Pat<(v8i32 (vselect (v8i32 VR256:$mask), (v8i32 VR256:$src1),
                            (v8i32 VR256:$src2))),
            (VBLENDVPSYrr VR256:$src2, VR256:$src1, VR256:$mask)>;
  def : Pat<(v8f32 (vselect (v8i32 VR256:$mask), (v8f32 VR256:$src1),
                            (v8f32 VR256:$src2))),
            (VBLENDVPSYrr VR256:$src2, VR256:$src1, VR256:$mask)>;
  def : Pat<(v4i64 (vselect (v4i64 VR256:$mask), (v4i64 VR256:$src1),
                            (v4i64 VR256:$src2))),
            (VBLENDVPDYrr VR256:$src2, VR256:$src1, VR256:$mask)>;
  def : Pat<(v4f64 (vselect (v4i64 VR256:$mask), (v4f64 VR256:$src1),
                            (v4f64 VR256:$src2))),
            (VBLENDVPDYrr VR256:$src2, VR256:$src1, VR256:$mask)>;

  def : Pat<(v8f32 (X86Blendi (v8f32 VR256:$src1), (v8f32 VR256:$src2),
                               (imm:$mask))),
            (VBLENDPSYrri VR256:$src1, VR256:$src2, imm:$mask)>;
  def : Pat<(v4f64 (X86Blendi (v4f64 VR256:$src1), (v4f64 VR256:$src2),
                               (imm:$mask))),
            (VBLENDPDYrri VR256:$src1, VR256:$src2, imm:$mask)>;

  def : Pat<(v8i16 (X86Blendi (v8i16 VR128:$src1), (v8i16 VR128:$src2),
                               (imm:$mask))),
            (VPBLENDWrri VR128:$src1, VR128:$src2, imm:$mask)>;
  def : Pat<(v4f32 (X86Blendi (v4f32 VR128:$src1), (v4f32 VR128:$src2),
                               (imm:$mask))),
            (VBLENDPSrri VR128:$src1, VR128:$src2, imm:$mask)>;
  def : Pat<(v2f64 (X86Blendi (v2f64 VR128:$src1), (v2f64 VR128:$src2),
                               (imm:$mask))),
            (VBLENDPDrri VR128:$src1, VR128:$src2, imm:$mask)>;
}

let Predicates = [HasAVX2] in {
  def : Pat<(v32i8 (vselect (v32i8 VR256:$mask), (v32i8 VR256:$src1),
                            (v32i8 VR256:$src2))),
            (VPBLENDVBYrr VR256:$src2, VR256:$src1, VR256:$mask)>;
  def : Pat<(v16i16 (X86Blendi (v16i16 VR256:$src1), (v16i16 VR256:$src2),
                               (imm:$mask))),
            (VPBLENDWYrri VR256:$src1, VR256:$src2, imm:$mask)>;
}

/// SS41I_ternary_int - SSE 4.1 ternary operator
let Uses = [XMM0], Constraints = "$src1 = $dst" in {
  multiclass SS41I_ternary_int<bits<8> opc, string OpcodeStr, PatFrag mem_frag,
                               X86MemOperand x86memop, Intrinsic IntId,
                               OpndItins itins = DEFAULT_ITINS> {
    def rr0 : SS48I<opc, MRMSrcReg, (outs VR128:$dst),
                    (ins VR128:$src1, VR128:$src2),
                    !strconcat(OpcodeStr,
                     "\t{$src2, $dst|$dst, $src2}"),
                    [(set VR128:$dst, (IntId VR128:$src1, VR128:$src2, XMM0))],
                    itins.rr>, OpSize;

    def rm0 : SS48I<opc, MRMSrcMem, (outs VR128:$dst),
                    (ins VR128:$src1, x86memop:$src2),
                    !strconcat(OpcodeStr,
                     "\t{$src2, $dst|$dst, $src2}"),
                    [(set VR128:$dst,
                      (IntId VR128:$src1,
                       (bitconvert (mem_frag addr:$src2)), XMM0))],
                       itins.rm>, OpSize;
  }
}

let ExeDomain = SSEPackedDouble in
defm BLENDVPD : SS41I_ternary_int<0x15, "blendvpd", memopv2f64, f128mem,
                                  int_x86_sse41_blendvpd>;
let ExeDomain = SSEPackedSingle in
defm BLENDVPS : SS41I_ternary_int<0x14, "blendvps", memopv4f32, f128mem,
                                  int_x86_sse41_blendvps>;
defm PBLENDVB : SS41I_ternary_int<0x10, "pblendvb", memopv2i64, i128mem,
                                  int_x86_sse41_pblendvb>;

// Aliases with the implicit xmm0 argument
def : InstAlias<"blendvpd\t{%xmm0, $src2, $dst|$dst, $src2, xmm0}",
                (BLENDVPDrr0 VR128:$dst, VR128:$src2)>;
def : InstAlias<"blendvpd\t{%xmm0, $src2, $dst|$dst, $src2, xmm0}",
                (BLENDVPDrm0 VR128:$dst, f128mem:$src2)>;
def : InstAlias<"blendvps\t{%xmm0, $src2, $dst|$dst, $src2, xmm0}",
                (BLENDVPSrr0 VR128:$dst, VR128:$src2)>;
def : InstAlias<"blendvps\t{%xmm0, $src2, $dst|$dst, $src2, xmm0}",
                (BLENDVPSrm0 VR128:$dst, f128mem:$src2)>;
def : InstAlias<"pblendvb\t{%xmm0, $src2, $dst|$dst, $src2, xmm0}",
                (PBLENDVBrr0 VR128:$dst, VR128:$src2)>;
def : InstAlias<"pblendvb\t{%xmm0, $src2, $dst|$dst, $src2, xmm0}",
                (PBLENDVBrm0 VR128:$dst, i128mem:$src2)>;

let Predicates = [UseSSE41] in {
  def : Pat<(v16i8 (vselect (v16i8 XMM0), (v16i8 VR128:$src1),
                            (v16i8 VR128:$src2))),
            (PBLENDVBrr0 VR128:$src2, VR128:$src1)>;
  def : Pat<(v4i32 (vselect (v4i32 XMM0), (v4i32 VR128:$src1),
                            (v4i32 VR128:$src2))),
            (BLENDVPSrr0 VR128:$src2, VR128:$src1)>;
  def : Pat<(v4f32 (vselect (v4i32 XMM0), (v4f32 VR128:$src1),
                            (v4f32 VR128:$src2))),
            (BLENDVPSrr0 VR128:$src2, VR128:$src1)>;
  def : Pat<(v2i64 (vselect (v2i64 XMM0), (v2i64 VR128:$src1),
                            (v2i64 VR128:$src2))),
            (BLENDVPDrr0 VR128:$src2, VR128:$src1)>;
  def : Pat<(v2f64 (vselect (v2i64 XMM0), (v2f64 VR128:$src1),
                            (v2f64 VR128:$src2))),
            (BLENDVPDrr0 VR128:$src2, VR128:$src1)>;

  def : Pat<(v8i16 (X86Blendi (v8i16 VR128:$src1), (v8i16 VR128:$src2),
                               (imm:$mask))),
            (PBLENDWrri VR128:$src1, VR128:$src2, imm:$mask)>;
  def : Pat<(v4f32 (X86Blendi (v4f32 VR128:$src1), (v4f32 VR128:$src2),
                               (imm:$mask))),
            (BLENDPSrri VR128:$src1, VR128:$src2, imm:$mask)>;
  def : Pat<(v2f64 (X86Blendi (v2f64 VR128:$src1), (v2f64 VR128:$src2),
                               (imm:$mask))),
            (BLENDPDrri VR128:$src1, VR128:$src2, imm:$mask)>;

}

let Predicates = [HasAVX] in
def VMOVNTDQArm : SS48I<0x2A, MRMSrcMem, (outs VR128:$dst), (ins i128mem:$src),
                       "vmovntdqa\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse41_movntdqa addr:$src))]>,
                       OpSize, VEX;
let Predicates = [HasAVX2] in
def VMOVNTDQAYrm : SS48I<0x2A, MRMSrcMem, (outs VR256:$dst), (ins i256mem:$src),
                         "vmovntdqa\t{$src, $dst|$dst, $src}",
                         [(set VR256:$dst, (int_x86_avx2_movntdqa addr:$src))]>,
                         OpSize, VEX, VEX_L;
def MOVNTDQArm : SS48I<0x2A, MRMSrcMem, (outs VR128:$dst), (ins i128mem:$src),
                       "movntdqa\t{$src, $dst|$dst, $src}",
                       [(set VR128:$dst, (int_x86_sse41_movntdqa addr:$src))]>,
                       OpSize;

//===----------------------------------------------------------------------===//
// SSE4.2 - Compare Instructions
//===----------------------------------------------------------------------===//

/// SS42I_binop_rm - Simple SSE 4.2 binary operator
multiclass SS42I_binop_rm<bits<8> opc, string OpcodeStr, SDNode OpNode,
                          ValueType OpVT, RegisterClass RC, PatFrag memop_frag,
                          X86MemOperand x86memop, bit Is2Addr = 1> {
  def rr : SS428I<opc, MRMSrcReg, (outs RC:$dst),
       (ins RC:$src1, RC:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set RC:$dst, (OpVT (OpNode RC:$src1, RC:$src2)))]>,
       OpSize;
  def rm : SS428I<opc, MRMSrcMem, (outs RC:$dst),
       (ins RC:$src1, x86memop:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set RC:$dst,
         (OpVT (OpNode RC:$src1, (memop_frag addr:$src2))))]>, OpSize;
}

let Predicates = [HasAVX] in
  defm VPCMPGTQ : SS42I_binop_rm<0x37, "vpcmpgtq", X86pcmpgt, v2i64, VR128,
                                 loadv2i64, i128mem, 0>, VEX_4V;

let Predicates = [HasAVX2] in
  defm VPCMPGTQY : SS42I_binop_rm<0x37, "vpcmpgtq", X86pcmpgt, v4i64, VR256,
                                  loadv4i64, i256mem, 0>, VEX_4V, VEX_L;

let Constraints = "$src1 = $dst" in
  defm PCMPGTQ : SS42I_binop_rm<0x37, "pcmpgtq", X86pcmpgt, v2i64, VR128,
                                memopv2i64, i128mem>;

//===----------------------------------------------------------------------===//
// SSE4.2 - String/text Processing Instructions
//===----------------------------------------------------------------------===//

// Packed Compare Implicit Length Strings, Return Mask
multiclass pseudo_pcmpistrm<string asm> {
  def REG : PseudoI<(outs VR128:$dst),
                    (ins VR128:$src1, VR128:$src2, i8imm:$src3),
    [(set VR128:$dst, (int_x86_sse42_pcmpistrm128 VR128:$src1, VR128:$src2,
                                                  imm:$src3))]>;
  def MEM : PseudoI<(outs VR128:$dst),
                    (ins VR128:$src1, i128mem:$src2, i8imm:$src3),
    [(set VR128:$dst, (int_x86_sse42_pcmpistrm128 VR128:$src1,
                       (bc_v16i8 (memopv2i64 addr:$src2)), imm:$src3))]>;
}

let Defs = [EFLAGS], usesCustomInserter = 1 in {
  defm VPCMPISTRM128 : pseudo_pcmpistrm<"#VPCMPISTRM128">, Requires<[HasAVX]>;
  defm PCMPISTRM128 : pseudo_pcmpistrm<"#PCMPISTRM128">, Requires<[UseSSE42]>;
}

multiclass pcmpistrm_SS42AI<string asm> {
  def rr : SS42AI<0x62, MRMSrcReg, (outs),
    (ins VR128:$src1, VR128:$src2, i8imm:$src3),
    !strconcat(asm, "\t{$src3, $src2, $src1|$src1, $src2, $src3}"),
    []>, OpSize;
  let mayLoad = 1 in
  def rm :SS42AI<0x62, MRMSrcMem, (outs),
    (ins VR128:$src1, i128mem:$src2, i8imm:$src3),
    !strconcat(asm, "\t{$src3, $src2, $src1|$src1, $src2, $src3}"),
    []>, OpSize;
}

let Defs = [XMM0, EFLAGS], neverHasSideEffects = 1 in {
  let Predicates = [HasAVX] in
  defm VPCMPISTRM128 : pcmpistrm_SS42AI<"vpcmpistrm">, VEX;
  defm PCMPISTRM128  : pcmpistrm_SS42AI<"pcmpistrm"> ;
}

// Packed Compare Explicit Length Strings, Return Mask
multiclass pseudo_pcmpestrm<string asm> {
  def REG : PseudoI<(outs VR128:$dst),
                    (ins VR128:$src1, VR128:$src3, i8imm:$src5),
    [(set VR128:$dst, (int_x86_sse42_pcmpestrm128
                       VR128:$src1, EAX, VR128:$src3, EDX, imm:$src5))]>;
  def MEM : PseudoI<(outs VR128:$dst),
                    (ins VR128:$src1, i128mem:$src3, i8imm:$src5),
    [(set VR128:$dst, (int_x86_sse42_pcmpestrm128 VR128:$src1, EAX,
                       (bc_v16i8 (memopv2i64 addr:$src3)), EDX, imm:$src5))]>;
}

let Defs = [EFLAGS], Uses = [EAX, EDX], usesCustomInserter = 1 in {
  defm VPCMPESTRM128 : pseudo_pcmpestrm<"#VPCMPESTRM128">, Requires<[HasAVX]>;
  defm PCMPESTRM128 : pseudo_pcmpestrm<"#PCMPESTRM128">, Requires<[UseSSE42]>;
}

multiclass SS42AI_pcmpestrm<string asm> {
  def rr : SS42AI<0x60, MRMSrcReg, (outs),
    (ins VR128:$src1, VR128:$src3, i8imm:$src5),
    !strconcat(asm, "\t{$src5, $src3, $src1|$src1, $src3, $src5}"),
    []>, OpSize;
  let mayLoad = 1 in
  def rm : SS42AI<0x60, MRMSrcMem, (outs),
    (ins VR128:$src1, i128mem:$src3, i8imm:$src5),
    !strconcat(asm, "\t{$src5, $src3, $src1|$src1, $src3, $src5}"),
    []>, OpSize;
}

let Defs = [XMM0, EFLAGS], Uses = [EAX, EDX], neverHasSideEffects = 1 in {
  let Predicates = [HasAVX] in
  defm VPCMPESTRM128 : SS42AI_pcmpestrm<"vpcmpestrm">, VEX;
  defm PCMPESTRM128 :  SS42AI_pcmpestrm<"pcmpestrm">;
}

// Packed Compare Implicit Length Strings, Return Index
multiclass pseudo_pcmpistri<string asm> {
  def REG : PseudoI<(outs GR32:$dst),
                    (ins VR128:$src1, VR128:$src2, i8imm:$src3),
    [(set GR32:$dst, EFLAGS,
      (X86pcmpistri VR128:$src1, VR128:$src2, imm:$src3))]>;
  def MEM : PseudoI<(outs GR32:$dst),
                    (ins VR128:$src1, i128mem:$src2, i8imm:$src3),
    [(set GR32:$dst, EFLAGS, (X86pcmpistri VR128:$src1,
                              (bc_v16i8 (memopv2i64 addr:$src2)), imm:$src3))]>;
}

let Defs = [EFLAGS], usesCustomInserter = 1 in {
  defm VPCMPISTRI : pseudo_pcmpistri<"#VPCMPISTRI">, Requires<[HasAVX]>;
  defm PCMPISTRI  : pseudo_pcmpistri<"#PCMPISTRI">, Requires<[UseSSE42]>;
}

multiclass SS42AI_pcmpistri<string asm> {
  def rr : SS42AI<0x63, MRMSrcReg, (outs),
    (ins VR128:$src1, VR128:$src2, i8imm:$src3),
    !strconcat(asm, "\t{$src3, $src2, $src1|$src1, $src2, $src3}"),
    []>, OpSize;
  let mayLoad = 1 in
  def rm : SS42AI<0x63, MRMSrcMem, (outs),
    (ins VR128:$src1, i128mem:$src2, i8imm:$src3),
    !strconcat(asm, "\t{$src3, $src2, $src1|$src1, $src2, $src3}"),
    []>, OpSize;
}

let Defs = [ECX, EFLAGS], neverHasSideEffects = 1 in {
  let Predicates = [HasAVX] in
  defm VPCMPISTRI : SS42AI_pcmpistri<"vpcmpistri">, VEX;
  defm PCMPISTRI  : SS42AI_pcmpistri<"pcmpistri">;
}

// Packed Compare Explicit Length Strings, Return Index
multiclass pseudo_pcmpestri<string asm> {
  def REG : PseudoI<(outs GR32:$dst),
                    (ins VR128:$src1, VR128:$src3, i8imm:$src5),
    [(set GR32:$dst, EFLAGS,
      (X86pcmpestri VR128:$src1, EAX, VR128:$src3, EDX, imm:$src5))]>;
  def MEM : PseudoI<(outs GR32:$dst),
                    (ins VR128:$src1, i128mem:$src3, i8imm:$src5),
    [(set GR32:$dst, EFLAGS,
      (X86pcmpestri VR128:$src1, EAX, (bc_v16i8 (memopv2i64 addr:$src3)), EDX,
       imm:$src5))]>;
}

let Defs = [EFLAGS], Uses = [EAX, EDX], usesCustomInserter = 1 in {
  defm VPCMPESTRI : pseudo_pcmpestri<"#VPCMPESTRI">, Requires<[HasAVX]>;
  defm PCMPESTRI  : pseudo_pcmpestri<"#PCMPESTRI">, Requires<[UseSSE42]>;
}

multiclass SS42AI_pcmpestri<string asm> {
  def rr : SS42AI<0x61, MRMSrcReg, (outs),
    (ins VR128:$src1, VR128:$src3, i8imm:$src5),
    !strconcat(asm, "\t{$src5, $src3, $src1|$src1, $src3, $src5}"),
    []>, OpSize;
  let mayLoad = 1 in
  def rm : SS42AI<0x61, MRMSrcMem, (outs),
    (ins VR128:$src1, i128mem:$src3, i8imm:$src5),
    !strconcat(asm, "\t{$src5, $src3, $src1|$src1, $src3, $src5}"),
    []>, OpSize;
}

let Defs = [ECX, EFLAGS], Uses = [EAX, EDX], neverHasSideEffects = 1 in {
  let Predicates = [HasAVX] in
  defm VPCMPESTRI : SS42AI_pcmpestri<"vpcmpestri">, VEX;
  defm PCMPESTRI  : SS42AI_pcmpestri<"pcmpestri">;
}

//===----------------------------------------------------------------------===//
// SSE4.2 - CRC Instructions
//===----------------------------------------------------------------------===//

// No CRC instructions have AVX equivalents

// crc intrinsic instruction
// This set of instructions are only rm, the only difference is the size
// of r and m.
class SS42I_crc32r<bits<8> opc, string asm, RegisterClass RCOut,
                   RegisterClass RCIn, SDPatternOperator Int> :
  SS42FI<opc, MRMSrcReg, (outs RCOut:$dst), (ins RCOut:$src1, RCIn:$src2),
         !strconcat(asm, "\t{$src2, $src1|$src1, $src2}"),
         [(set RCOut:$dst, (Int RCOut:$src1, RCIn:$src2))], IIC_CRC32_REG>;

class SS42I_crc32m<bits<8> opc, string asm, RegisterClass RCOut,
                   X86MemOperand x86memop, SDPatternOperator Int> :
  SS42FI<opc, MRMSrcMem, (outs RCOut:$dst), (ins RCOut:$src1, x86memop:$src2),
         !strconcat(asm, "\t{$src2, $src1|$src1, $src2}"),
         [(set RCOut:$dst, (Int RCOut:$src1, (load addr:$src2)))],
         IIC_CRC32_MEM>;

let Constraints = "$src1 = $dst" in {
  def CRC32r32m8  : SS42I_crc32m<0xF0, "crc32{b}", GR32, i8mem,
                                 int_x86_sse42_crc32_32_8>;
  def CRC32r32r8  : SS42I_crc32r<0xF0, "crc32{b}", GR32, GR8,
                                 int_x86_sse42_crc32_32_8>;
  def CRC32r32m16 : SS42I_crc32m<0xF1, "crc32{w}", GR32, i16mem,
                                 int_x86_sse42_crc32_32_16>, OpSize;
  def CRC32r32r16 : SS42I_crc32r<0xF1, "crc32{w}", GR32, GR16,
                                 int_x86_sse42_crc32_32_16>, OpSize;
  def CRC32r32m32 : SS42I_crc32m<0xF1, "crc32{l}", GR32, i32mem,
                                 int_x86_sse42_crc32_32_32>;
  def CRC32r32r32 : SS42I_crc32r<0xF1, "crc32{l}", GR32, GR32,
                                 int_x86_sse42_crc32_32_32>;
  def CRC32r64m64 : SS42I_crc32m<0xF1, "crc32{q}", GR64, i64mem,
                                 int_x86_sse42_crc32_64_64>, REX_W;
  def CRC32r64r64 : SS42I_crc32r<0xF1, "crc32{q}", GR64, GR64,
                                 int_x86_sse42_crc32_64_64>, REX_W;
  let hasSideEffects = 0 in {
    let mayLoad = 1 in
    def CRC32r64m8 : SS42I_crc32m<0xF0, "crc32{b}", GR64, i8mem,
                                   null_frag>, REX_W;
    def CRC32r64r8 : SS42I_crc32r<0xF0, "crc32{b}", GR64, GR8,
                                   null_frag>, REX_W;
  }
}

//===----------------------------------------------------------------------===//
// SHA-NI Instructions
//===----------------------------------------------------------------------===//

multiclass SHAI_binop<bits<8> Opc, string OpcodeStr, Intrinsic IntId,
                      bit UsesXMM0 = 0> {
  def rr : I<Opc, MRMSrcReg, (outs VR128:$dst),
             (ins VR128:$src1, VR128:$src2),
             !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
             [!if(UsesXMM0,
                  (set VR128:$dst, (IntId VR128:$src1, VR128:$src2, XMM0)),
                  (set VR128:$dst, (IntId VR128:$src1, VR128:$src2)))]>, T8;

  def rm : I<Opc, MRMSrcMem, (outs VR128:$dst),
             (ins VR128:$src1, i128mem:$src2),
             !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
             [!if(UsesXMM0,
                  (set VR128:$dst, (IntId VR128:$src1,
                    (bc_v4i32 (memopv2i64 addr:$src2)), XMM0)),
                  (set VR128:$dst, (IntId VR128:$src1,
                    (bc_v4i32 (memopv2i64 addr:$src2)))))]>, T8;
}

let Constraints = "$src1 = $dst", Predicates = [HasSHA] in {
  def SHA1RNDS4rri : Ii8<0xCC, MRMSrcReg, (outs VR128:$dst),
                         (ins VR128:$src1, VR128:$src2, i8imm:$src3),
                         "sha1rnds4\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                         [(set VR128:$dst,
                           (int_x86_sha1rnds4 VR128:$src1, VR128:$src2,
                            (i8 imm:$src3)))]>, TA;
  def SHA1RNDS4rmi : Ii8<0xCC, MRMSrcMem, (outs VR128:$dst),
                         (ins VR128:$src1, i128mem:$src2, i8imm:$src3),
                         "sha1rnds4\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                         [(set VR128:$dst,
                           (int_x86_sha1rnds4 VR128:$src1,
                            (bc_v4i32 (memopv2i64 addr:$src2)),
                            (i8 imm:$src3)))]>, TA;

  defm SHA1NEXTE : SHAI_binop<0xC8, "sha1nexte", int_x86_sha1nexte>;
  defm SHA1MSG1  : SHAI_binop<0xC9, "sha1msg1", int_x86_sha1msg1>;
  defm SHA1MSG2  : SHAI_binop<0xCA, "sha1msg2", int_x86_sha1msg2>;

  let Uses=[XMM0] in
  defm SHA256RNDS2 : SHAI_binop<0xCB, "sha256rnds2", int_x86_sha256rnds2, 1>;

  defm SHA256MSG1 : SHAI_binop<0xCC, "sha256msg1", int_x86_sha256msg1>;
  defm SHA256MSG2 : SHAI_binop<0xCD, "sha256msg2", int_x86_sha256msg2>;
}

// Aliases with explicit %xmm0
def : InstAlias<"sha256rnds2\t{%xmm0, $src2, $dst|$dst, $src2, xmm0}",
                (SHA256RNDS2rr VR128:$dst, VR128:$src2)>;
def : InstAlias<"sha256rnds2\t{%xmm0, $src2, $dst|$dst, $src2, xmm0}",
                (SHA256RNDS2rm VR128:$dst, i128mem:$src2)>;

//===----------------------------------------------------------------------===//
// AES-NI Instructions
//===----------------------------------------------------------------------===//

multiclass AESI_binop_rm_int<bits<8> opc, string OpcodeStr,
                              Intrinsic IntId128, bit Is2Addr = 1> {
  def rr : AES8I<opc, MRMSrcReg, (outs VR128:$dst),
       (ins VR128:$src1, VR128:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set VR128:$dst, (IntId128 VR128:$src1, VR128:$src2))]>,
       OpSize;
  def rm : AES8I<opc, MRMSrcMem, (outs VR128:$dst),
       (ins VR128:$src1, i128mem:$src2),
       !if(Is2Addr,
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}")),
       [(set VR128:$dst,
         (IntId128 VR128:$src1, (memopv2i64 addr:$src2)))]>, OpSize;
}

// Perform One Round of an AES Encryption/Decryption Flow
let Predicates = [HasAVX, HasAES] in {
  defm VAESENC          : AESI_binop_rm_int<0xDC, "vaesenc",
                         int_x86_aesni_aesenc, 0>, VEX_4V;
  defm VAESENCLAST      : AESI_binop_rm_int<0xDD, "vaesenclast",
                         int_x86_aesni_aesenclast, 0>, VEX_4V;
  defm VAESDEC          : AESI_binop_rm_int<0xDE, "vaesdec",
                         int_x86_aesni_aesdec, 0>, VEX_4V;
  defm VAESDECLAST      : AESI_binop_rm_int<0xDF, "vaesdeclast",
                         int_x86_aesni_aesdeclast, 0>, VEX_4V;
}

let Constraints = "$src1 = $dst" in {
  defm AESENC          : AESI_binop_rm_int<0xDC, "aesenc",
                         int_x86_aesni_aesenc>;
  defm AESENCLAST      : AESI_binop_rm_int<0xDD, "aesenclast",
                         int_x86_aesni_aesenclast>;
  defm AESDEC          : AESI_binop_rm_int<0xDE, "aesdec",
                         int_x86_aesni_aesdec>;
  defm AESDECLAST      : AESI_binop_rm_int<0xDF, "aesdeclast",
                         int_x86_aesni_aesdeclast>;
}

// Perform the AES InvMixColumn Transformation
let Predicates = [HasAVX, HasAES] in {
  def VAESIMCrr : AES8I<0xDB, MRMSrcReg, (outs VR128:$dst),
      (ins VR128:$src1),
      "vaesimc\t{$src1, $dst|$dst, $src1}",
      [(set VR128:$dst,
        (int_x86_aesni_aesimc VR128:$src1))]>,
      OpSize, VEX;
  def VAESIMCrm : AES8I<0xDB, MRMSrcMem, (outs VR128:$dst),
      (ins i128mem:$src1),
      "vaesimc\t{$src1, $dst|$dst, $src1}",
      [(set VR128:$dst, (int_x86_aesni_aesimc (loadv2i64 addr:$src1)))]>,
      OpSize, VEX;
}
def AESIMCrr : AES8I<0xDB, MRMSrcReg, (outs VR128:$dst),
  (ins VR128:$src1),
  "aesimc\t{$src1, $dst|$dst, $src1}",
  [(set VR128:$dst,
    (int_x86_aesni_aesimc VR128:$src1))]>,
  OpSize;
def AESIMCrm : AES8I<0xDB, MRMSrcMem, (outs VR128:$dst),
  (ins i128mem:$src1),
  "aesimc\t{$src1, $dst|$dst, $src1}",
  [(set VR128:$dst, (int_x86_aesni_aesimc (memopv2i64 addr:$src1)))]>,
  OpSize;

// AES Round Key Generation Assist
let Predicates = [HasAVX, HasAES] in {
  def VAESKEYGENASSIST128rr : AESAI<0xDF, MRMSrcReg, (outs VR128:$dst),
      (ins VR128:$src1, i8imm:$src2),
      "vaeskeygenassist\t{$src2, $src1, $dst|$dst, $src1, $src2}",
      [(set VR128:$dst,
        (int_x86_aesni_aeskeygenassist VR128:$src1, imm:$src2))]>,
      OpSize, VEX;
  def VAESKEYGENASSIST128rm : AESAI<0xDF, MRMSrcMem, (outs VR128:$dst),
      (ins i128mem:$src1, i8imm:$src2),
      "vaeskeygenassist\t{$src2, $src1, $dst|$dst, $src1, $src2}",
      [(set VR128:$dst,
        (int_x86_aesni_aeskeygenassist (loadv2i64 addr:$src1), imm:$src2))]>,
      OpSize, VEX;
}
def AESKEYGENASSIST128rr : AESAI<0xDF, MRMSrcReg, (outs VR128:$dst),
  (ins VR128:$src1, i8imm:$src2),
  "aeskeygenassist\t{$src2, $src1, $dst|$dst, $src1, $src2}",
  [(set VR128:$dst,
    (int_x86_aesni_aeskeygenassist VR128:$src1, imm:$src2))]>,
  OpSize;
def AESKEYGENASSIST128rm : AESAI<0xDF, MRMSrcMem, (outs VR128:$dst),
  (ins i128mem:$src1, i8imm:$src2),
  "aeskeygenassist\t{$src2, $src1, $dst|$dst, $src1, $src2}",
  [(set VR128:$dst,
    (int_x86_aesni_aeskeygenassist (memopv2i64 addr:$src1), imm:$src2))]>,
  OpSize;

//===----------------------------------------------------------------------===//
// PCLMUL Instructions
//===----------------------------------------------------------------------===//

// AVX carry-less Multiplication instructions
def VPCLMULQDQrr : AVXPCLMULIi8<0x44, MRMSrcReg, (outs VR128:$dst),
           (ins VR128:$src1, VR128:$src2, i8imm:$src3),
           "vpclmulqdq\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
           [(set VR128:$dst,
             (int_x86_pclmulqdq VR128:$src1, VR128:$src2, imm:$src3))]>;

def VPCLMULQDQrm : AVXPCLMULIi8<0x44, MRMSrcMem, (outs VR128:$dst),
           (ins VR128:$src1, i128mem:$src2, i8imm:$src3),
           "vpclmulqdq\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
           [(set VR128:$dst, (int_x86_pclmulqdq VR128:$src1,
                              (loadv2i64 addr:$src2), imm:$src3))]>;

// Carry-less Multiplication instructions
let Constraints = "$src1 = $dst" in {
def PCLMULQDQrr : PCLMULIi8<0x44, MRMSrcReg, (outs VR128:$dst),
           (ins VR128:$src1, VR128:$src2, i8imm:$src3),
           "pclmulqdq\t{$src3, $src2, $dst|$dst, $src2, $src3}",
           [(set VR128:$dst,
             (int_x86_pclmulqdq VR128:$src1, VR128:$src2, imm:$src3))],
             IIC_SSE_PCLMULQDQ_RR>;

def PCLMULQDQrm : PCLMULIi8<0x44, MRMSrcMem, (outs VR128:$dst),
           (ins VR128:$src1, i128mem:$src2, i8imm:$src3),
           "pclmulqdq\t{$src3, $src2, $dst|$dst, $src2, $src3}",
           [(set VR128:$dst, (int_x86_pclmulqdq VR128:$src1,
                              (memopv2i64 addr:$src2), imm:$src3))],
                              IIC_SSE_PCLMULQDQ_RM>;
} // Constraints = "$src1 = $dst"


multiclass pclmul_alias<string asm, int immop> {
  def : InstAlias<!strconcat("pclmul", asm, "dq {$src, $dst|$dst, $src}"),
                  (PCLMULQDQrr VR128:$dst, VR128:$src, immop)>;

  def : InstAlias<!strconcat("pclmul", asm, "dq {$src, $dst|$dst, $src}"),
                  (PCLMULQDQrm VR128:$dst, i128mem:$src, immop)>;

  def : InstAlias<!strconcat("vpclmul", asm,
                             "dq {$src2, $src1, $dst|$dst, $src1, $src2}"),
                  (VPCLMULQDQrr VR128:$dst, VR128:$src1, VR128:$src2, immop)>;

  def : InstAlias<!strconcat("vpclmul", asm,
                             "dq {$src2, $src1, $dst|$dst, $src1, $src2}"),
                  (VPCLMULQDQrm VR128:$dst, VR128:$src1, i128mem:$src2, immop)>;
}
defm : pclmul_alias<"hqhq", 0x11>;
defm : pclmul_alias<"hqlq", 0x01>;
defm : pclmul_alias<"lqhq", 0x10>;
defm : pclmul_alias<"lqlq", 0x00>;

//===----------------------------------------------------------------------===//
// SSE4A Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasSSE4A] in {

let Constraints = "$src = $dst" in {
def EXTRQI : Ii8<0x78, MRM0r, (outs VR128:$dst),
                 (ins VR128:$src, i8imm:$len, i8imm:$idx),
                 "extrq\t{$idx, $len, $src|$src, $len, $idx}",
                 [(set VR128:$dst, (int_x86_sse4a_extrqi VR128:$src, imm:$len,
                                    imm:$idx))]>, TB, OpSize;
def EXTRQ  : I<0x79, MRMSrcReg, (outs VR128:$dst),
              (ins VR128:$src, VR128:$mask),
              "extrq\t{$mask, $src|$src, $mask}",
              [(set VR128:$dst, (int_x86_sse4a_extrq VR128:$src,
                                 VR128:$mask))]>, TB, OpSize;

def INSERTQI : Ii8<0x78, MRMSrcReg, (outs VR128:$dst),
                   (ins VR128:$src, VR128:$src2, i8imm:$len, i8imm:$idx),
                   "insertq\t{$idx, $len, $src2, $src|$src, $src2, $len, $idx}",
                   [(set VR128:$dst, (int_x86_sse4a_insertqi VR128:$src,
                                      VR128:$src2, imm:$len, imm:$idx))]>, XD;
def INSERTQ  : I<0x79, MRMSrcReg, (outs VR128:$dst),
                 (ins VR128:$src, VR128:$mask),
                 "insertq\t{$mask, $src|$src, $mask}",
                 [(set VR128:$dst, (int_x86_sse4a_insertq VR128:$src,
                                    VR128:$mask))]>, XD;
}

def MOVNTSS : I<0x2B, MRMDestMem, (outs), (ins f32mem:$dst, VR128:$src),
                "movntss\t{$src, $dst|$dst, $src}",
                [(int_x86_sse4a_movnt_ss addr:$dst, VR128:$src)]>, XS;

def MOVNTSD : I<0x2B, MRMDestMem, (outs), (ins f64mem:$dst, VR128:$src),
                "movntsd\t{$src, $dst|$dst, $src}",
                [(int_x86_sse4a_movnt_sd addr:$dst, VR128:$src)]>, XD;
}

//===----------------------------------------------------------------------===//
// AVX Instructions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// VBROADCAST - Load from memory and broadcast to all elements of the
//              destination operand
//
class avx_broadcast<bits<8> opc, string OpcodeStr, RegisterClass RC,
                    X86MemOperand x86memop, Intrinsic Int> :
  AVX8I<opc, MRMSrcMem, (outs RC:$dst), (ins x86memop:$src),
        !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
        [(set RC:$dst, (Int addr:$src))]>, VEX;

// AVX2 adds register forms
class avx2_broadcast_reg<bits<8> opc, string OpcodeStr, RegisterClass RC,
                         Intrinsic Int> :
  AVX28I<opc, MRMSrcReg, (outs RC:$dst), (ins VR128:$src),
         !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
         [(set RC:$dst, (Int VR128:$src))]>, VEX;

let ExeDomain = SSEPackedSingle in {
  def VBROADCASTSSrm  : avx_broadcast<0x18, "vbroadcastss", VR128, f32mem,
                                      int_x86_avx_vbroadcast_ss>;
  def VBROADCASTSSYrm : avx_broadcast<0x18, "vbroadcastss", VR256, f32mem,
                                      int_x86_avx_vbroadcast_ss_256>, VEX_L;
}
let ExeDomain = SSEPackedDouble in
def VBROADCASTSDYrm  : avx_broadcast<0x19, "vbroadcastsd", VR256, f64mem,
                                    int_x86_avx_vbroadcast_sd_256>, VEX_L;
def VBROADCASTF128 : avx_broadcast<0x1A, "vbroadcastf128", VR256, f128mem,
                                   int_x86_avx_vbroadcastf128_pd_256>, VEX_L;

let ExeDomain = SSEPackedSingle in {
  def VBROADCASTSSrr  : avx2_broadcast_reg<0x18, "vbroadcastss", VR128,
                                           int_x86_avx2_vbroadcast_ss_ps>;
  def VBROADCASTSSYrr : avx2_broadcast_reg<0x18, "vbroadcastss", VR256,
                                      int_x86_avx2_vbroadcast_ss_ps_256>, VEX_L;
}
let ExeDomain = SSEPackedDouble in
def VBROADCASTSDYrr  : avx2_broadcast_reg<0x19, "vbroadcastsd", VR256,
                                      int_x86_avx2_vbroadcast_sd_pd_256>, VEX_L;

let Predicates = [HasAVX2] in
def VBROADCASTI128 : avx_broadcast<0x5A, "vbroadcasti128", VR256, i128mem,
                                   int_x86_avx2_vbroadcasti128>, VEX_L;

let Predicates = [HasAVX] in
def : Pat<(int_x86_avx_vbroadcastf128_ps_256 addr:$src),
          (VBROADCASTF128 addr:$src)>;


//===----------------------------------------------------------------------===//
// VINSERTF128 - Insert packed floating-point values
//
let neverHasSideEffects = 1, ExeDomain = SSEPackedSingle in {
def VINSERTF128rr : AVXAIi8<0x18, MRMSrcReg, (outs VR256:$dst),
          (ins VR256:$src1, VR128:$src2, i8imm:$src3),
          "vinsertf128\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
          []>, VEX_4V, VEX_L;
let mayLoad = 1 in
def VINSERTF128rm : AVXAIi8<0x18, MRMSrcMem, (outs VR256:$dst),
          (ins VR256:$src1, f128mem:$src2, i8imm:$src3),
          "vinsertf128\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
          []>, VEX_4V, VEX_L;
}

let Predicates = [HasAVX] in {
def : Pat<(vinsert128_insert:$ins (v8f32 VR256:$src1), (v4f32 VR128:$src2),
                                   (iPTR imm)),
          (VINSERTF128rr VR256:$src1, VR128:$src2,
                         (INSERT_get_vinsert128_imm VR256:$ins))>;
def : Pat<(vinsert128_insert:$ins (v4f64 VR256:$src1), (v2f64 VR128:$src2),
                                   (iPTR imm)),
          (VINSERTF128rr VR256:$src1, VR128:$src2,
                         (INSERT_get_vinsert128_imm VR256:$ins))>;

def : Pat<(vinsert128_insert:$ins (v8f32 VR256:$src1), (loadv4f32 addr:$src2),
                                   (iPTR imm)),
          (VINSERTF128rm VR256:$src1, addr:$src2,
                         (INSERT_get_vinsert128_imm VR256:$ins))>;
def : Pat<(vinsert128_insert:$ins (v4f64 VR256:$src1), (loadv2f64 addr:$src2),
                                   (iPTR imm)),
          (VINSERTF128rm VR256:$src1, addr:$src2,
                         (INSERT_get_vinsert128_imm VR256:$ins))>;
}

let Predicates = [HasAVX1Only] in {
def : Pat<(vinsert128_insert:$ins (v4i64 VR256:$src1), (v2i64 VR128:$src2),
                                   (iPTR imm)),
          (VINSERTF128rr VR256:$src1, VR128:$src2,
                         (INSERT_get_vinsert128_imm VR256:$ins))>;
def : Pat<(vinsert128_insert:$ins (v8i32 VR256:$src1), (v4i32 VR128:$src2),
                                   (iPTR imm)),
          (VINSERTF128rr VR256:$src1, VR128:$src2,
                         (INSERT_get_vinsert128_imm VR256:$ins))>;
def : Pat<(vinsert128_insert:$ins (v32i8 VR256:$src1), (v16i8 VR128:$src2),
                                   (iPTR imm)),
          (VINSERTF128rr VR256:$src1, VR128:$src2,
                         (INSERT_get_vinsert128_imm VR256:$ins))>;
def : Pat<(vinsert128_insert:$ins (v16i16 VR256:$src1), (v8i16 VR128:$src2),
                                   (iPTR imm)),
          (VINSERTF128rr VR256:$src1, VR128:$src2,
                         (INSERT_get_vinsert128_imm VR256:$ins))>;

def : Pat<(vinsert128_insert:$ins (v4i64 VR256:$src1), (loadv2i64 addr:$src2),
                                   (iPTR imm)),
          (VINSERTF128rm VR256:$src1, addr:$src2,
                         (INSERT_get_vinsert128_imm VR256:$ins))>;
def : Pat<(vinsert128_insert:$ins (v8i32 VR256:$src1),
                                   (bc_v4i32 (loadv2i64 addr:$src2)),
                                   (iPTR imm)),
          (VINSERTF128rm VR256:$src1, addr:$src2,
                         (INSERT_get_vinsert128_imm VR256:$ins))>;
def : Pat<(vinsert128_insert:$ins (v32i8 VR256:$src1),
                                   (bc_v16i8 (loadv2i64 addr:$src2)),
                                   (iPTR imm)),
          (VINSERTF128rm VR256:$src1, addr:$src2,
                         (INSERT_get_vinsert128_imm VR256:$ins))>;
def : Pat<(vinsert128_insert:$ins (v16i16 VR256:$src1),
                                   (bc_v8i16 (loadv2i64 addr:$src2)),
                                   (iPTR imm)),
          (VINSERTF128rm VR256:$src1, addr:$src2,
                         (INSERT_get_vinsert128_imm VR256:$ins))>;
}

//===----------------------------------------------------------------------===//
// VEXTRACTF128 - Extract packed floating-point values
//
let neverHasSideEffects = 1, ExeDomain = SSEPackedSingle in {
def VEXTRACTF128rr : AVXAIi8<0x19, MRMDestReg, (outs VR128:$dst),
          (ins VR256:$src1, i8imm:$src2),
          "vextractf128\t{$src2, $src1, $dst|$dst, $src1, $src2}",
          []>, VEX, VEX_L;
let mayStore = 1 in
def VEXTRACTF128mr : AVXAIi8<0x19, MRMDestMem, (outs),
          (ins f128mem:$dst, VR256:$src1, i8imm:$src2),
          "vextractf128\t{$src2, $src1, $dst|$dst, $src1, $src2}",
          []>, VEX, VEX_L;
}

// AVX1 patterns
let Predicates = [HasAVX] in {
def : Pat<(vextract128_extract:$ext VR256:$src1, (iPTR imm)),
          (v4f32 (VEXTRACTF128rr
                    (v8f32 VR256:$src1),
                    (EXTRACT_get_vextract128_imm VR128:$ext)))>;
def : Pat<(vextract128_extract:$ext VR256:$src1, (iPTR imm)),
          (v2f64 (VEXTRACTF128rr
                    (v4f64 VR256:$src1),
                    (EXTRACT_get_vextract128_imm VR128:$ext)))>;

def : Pat<(store (v4f32 (vextract128_extract:$ext (v8f32 VR256:$src1),
                         (iPTR imm))), addr:$dst),
          (VEXTRACTF128mr addr:$dst, VR256:$src1,
           (EXTRACT_get_vextract128_imm VR128:$ext))>;
def : Pat<(store (v2f64 (vextract128_extract:$ext (v4f64 VR256:$src1),
                         (iPTR imm))), addr:$dst),
          (VEXTRACTF128mr addr:$dst, VR256:$src1,
           (EXTRACT_get_vextract128_imm VR128:$ext))>;
}

let Predicates = [HasAVX1Only] in {
def : Pat<(vextract128_extract:$ext VR256:$src1, (iPTR imm)),
          (v2i64 (VEXTRACTF128rr
                  (v4i64 VR256:$src1),
                  (EXTRACT_get_vextract128_imm VR128:$ext)))>;
def : Pat<(vextract128_extract:$ext VR256:$src1, (iPTR imm)),
          (v4i32 (VEXTRACTF128rr
                  (v8i32 VR256:$src1),
                  (EXTRACT_get_vextract128_imm VR128:$ext)))>;
def : Pat<(vextract128_extract:$ext VR256:$src1, (iPTR imm)),
          (v8i16 (VEXTRACTF128rr
                  (v16i16 VR256:$src1),
                  (EXTRACT_get_vextract128_imm VR128:$ext)))>;
def : Pat<(vextract128_extract:$ext VR256:$src1, (iPTR imm)),
          (v16i8 (VEXTRACTF128rr
                  (v32i8 VR256:$src1),
                  (EXTRACT_get_vextract128_imm VR128:$ext)))>;

def : Pat<(alignedstore (v2i64 (vextract128_extract:$ext (v4i64 VR256:$src1),
                                (iPTR imm))), addr:$dst),
          (VEXTRACTF128mr addr:$dst, VR256:$src1,
           (EXTRACT_get_vextract128_imm VR128:$ext))>;
def : Pat<(alignedstore (v4i32 (vextract128_extract:$ext (v8i32 VR256:$src1),
                                (iPTR imm))), addr:$dst),
          (VEXTRACTF128mr addr:$dst, VR256:$src1,
           (EXTRACT_get_vextract128_imm VR128:$ext))>;
def : Pat<(alignedstore (v8i16 (vextract128_extract:$ext (v16i16 VR256:$src1),
                                (iPTR imm))), addr:$dst),
          (VEXTRACTF128mr addr:$dst, VR256:$src1,
           (EXTRACT_get_vextract128_imm VR128:$ext))>;
def : Pat<(alignedstore (v16i8 (vextract128_extract:$ext (v32i8 VR256:$src1),
                                (iPTR imm))), addr:$dst),
          (VEXTRACTF128mr addr:$dst, VR256:$src1,
           (EXTRACT_get_vextract128_imm VR128:$ext))>;
}

//===----------------------------------------------------------------------===//
// VMASKMOV - Conditional SIMD Packed Loads and Stores
//
multiclass avx_movmask_rm<bits<8> opc_rm, bits<8> opc_mr, string OpcodeStr,
                          Intrinsic IntLd, Intrinsic IntLd256,
                          Intrinsic IntSt, Intrinsic IntSt256> {
  def rm  : AVX8I<opc_rm, MRMSrcMem, (outs VR128:$dst),
             (ins VR128:$src1, f128mem:$src2),
             !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set VR128:$dst, (IntLd addr:$src2, VR128:$src1))]>,
             VEX_4V;
  def Yrm : AVX8I<opc_rm, MRMSrcMem, (outs VR256:$dst),
             (ins VR256:$src1, f256mem:$src2),
             !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set VR256:$dst, (IntLd256 addr:$src2, VR256:$src1))]>,
             VEX_4V, VEX_L;
  def mr  : AVX8I<opc_mr, MRMDestMem, (outs),
             (ins f128mem:$dst, VR128:$src1, VR128:$src2),
             !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(IntSt addr:$dst, VR128:$src1, VR128:$src2)]>, VEX_4V;
  def Ymr : AVX8I<opc_mr, MRMDestMem, (outs),
             (ins f256mem:$dst, VR256:$src1, VR256:$src2),
             !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(IntSt256 addr:$dst, VR256:$src1, VR256:$src2)]>, VEX_4V, VEX_L;
}

let ExeDomain = SSEPackedSingle in
defm VMASKMOVPS : avx_movmask_rm<0x2C, 0x2E, "vmaskmovps",
                                 int_x86_avx_maskload_ps,
                                 int_x86_avx_maskload_ps_256,
                                 int_x86_avx_maskstore_ps,
                                 int_x86_avx_maskstore_ps_256>;
let ExeDomain = SSEPackedDouble in
defm VMASKMOVPD : avx_movmask_rm<0x2D, 0x2F, "vmaskmovpd",
                                 int_x86_avx_maskload_pd,
                                 int_x86_avx_maskload_pd_256,
                                 int_x86_avx_maskstore_pd,
                                 int_x86_avx_maskstore_pd_256>;

//===----------------------------------------------------------------------===//
// VPERMIL - Permute Single and Double Floating-Point Values
//
multiclass avx_permil<bits<8> opc_rm, bits<8> opc_rmi, string OpcodeStr,
                      RegisterClass RC, X86MemOperand x86memop_f,
                      X86MemOperand x86memop_i, PatFrag i_frag,
                      Intrinsic IntVar, ValueType vt> {
  def rr  : AVX8I<opc_rm, MRMSrcReg, (outs RC:$dst),
             (ins RC:$src1, RC:$src2),
             !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set RC:$dst, (IntVar RC:$src1, RC:$src2))]>, VEX_4V;
  def rm  : AVX8I<opc_rm, MRMSrcMem, (outs RC:$dst),
             (ins RC:$src1, x86memop_i:$src2),
             !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set RC:$dst, (IntVar RC:$src1,
                             (bitconvert (i_frag addr:$src2))))]>, VEX_4V;

  def ri  : AVXAIi8<opc_rmi, MRMSrcReg, (outs RC:$dst),
             (ins RC:$src1, i8imm:$src2),
             !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set RC:$dst, (vt (X86VPermilp RC:$src1, (i8 imm:$src2))))]>, VEX;
  def mi  : AVXAIi8<opc_rmi, MRMSrcMem, (outs RC:$dst),
             (ins x86memop_f:$src1, i8imm:$src2),
             !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set RC:$dst,
               (vt (X86VPermilp (memop addr:$src1), (i8 imm:$src2))))]>, VEX;
}

let ExeDomain = SSEPackedSingle in {
  defm VPERMILPS  : avx_permil<0x0C, 0x04, "vpermilps", VR128, f128mem, i128mem,
                               loadv2i64, int_x86_avx_vpermilvar_ps, v4f32>;
  defm VPERMILPSY : avx_permil<0x0C, 0x04, "vpermilps", VR256, f256mem, i256mem,
                       loadv4i64, int_x86_avx_vpermilvar_ps_256, v8f32>, VEX_L;
}
let ExeDomain = SSEPackedDouble in {
  defm VPERMILPD  : avx_permil<0x0D, 0x05, "vpermilpd", VR128, f128mem, i128mem,
                               loadv2i64, int_x86_avx_vpermilvar_pd, v2f64>;
  defm VPERMILPDY : avx_permil<0x0D, 0x05, "vpermilpd", VR256, f256mem, i256mem,
                       loadv4i64, int_x86_avx_vpermilvar_pd_256, v4f64>, VEX_L;
}

let Predicates = [HasAVX] in {
def : Pat<(v8i32 (X86VPermilp VR256:$src1, (i8 imm:$imm))),
          (VPERMILPSYri VR256:$src1, imm:$imm)>;
def : Pat<(v4i64 (X86VPermilp VR256:$src1, (i8 imm:$imm))),
          (VPERMILPDYri VR256:$src1, imm:$imm)>;
def : Pat<(v8i32 (X86VPermilp (bc_v8i32 (loadv4i64 addr:$src1)),
                               (i8 imm:$imm))),
          (VPERMILPSYmi addr:$src1, imm:$imm)>;
def : Pat<(v4i64 (X86VPermilp (loadv4i64 addr:$src1), (i8 imm:$imm))),
          (VPERMILPDYmi addr:$src1, imm:$imm)>;

def : Pat<(v2i64 (X86VPermilp VR128:$src1, (i8 imm:$imm))),
          (VPERMILPDri VR128:$src1, imm:$imm)>;
def : Pat<(v2i64 (X86VPermilp (loadv2i64 addr:$src1), (i8 imm:$imm))),
          (VPERMILPDmi addr:$src1, imm:$imm)>;
}

//===----------------------------------------------------------------------===//
// VPERM2F128 - Permute Floating-Point Values in 128-bit chunks
//
let ExeDomain = SSEPackedSingle in {
def VPERM2F128rr : AVXAIi8<0x06, MRMSrcReg, (outs VR256:$dst),
          (ins VR256:$src1, VR256:$src2, i8imm:$src3),
          "vperm2f128\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
          [(set VR256:$dst, (v8f32 (X86VPerm2x128 VR256:$src1, VR256:$src2,
                              (i8 imm:$src3))))]>, VEX_4V, VEX_L;
def VPERM2F128rm : AVXAIi8<0x06, MRMSrcMem, (outs VR256:$dst),
          (ins VR256:$src1, f256mem:$src2, i8imm:$src3),
          "vperm2f128\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
          [(set VR256:$dst, (X86VPerm2x128 VR256:$src1, (loadv8f32 addr:$src2),
                             (i8 imm:$src3)))]>, VEX_4V, VEX_L;
}

let Predicates = [HasAVX] in {
def : Pat<(v4f64 (X86VPerm2x128 VR256:$src1, VR256:$src2, (i8 imm:$imm))),
          (VPERM2F128rr VR256:$src1, VR256:$src2, imm:$imm)>;
def : Pat<(v4f64 (X86VPerm2x128 VR256:$src1,
                  (loadv4f64 addr:$src2), (i8 imm:$imm))),
          (VPERM2F128rm VR256:$src1, addr:$src2, imm:$imm)>;
}

let Predicates = [HasAVX1Only] in {
def : Pat<(v8i32 (X86VPerm2x128 VR256:$src1, VR256:$src2, (i8 imm:$imm))),
          (VPERM2F128rr VR256:$src1, VR256:$src2, imm:$imm)>;
def : Pat<(v4i64 (X86VPerm2x128 VR256:$src1, VR256:$src2, (i8 imm:$imm))),
          (VPERM2F128rr VR256:$src1, VR256:$src2, imm:$imm)>;
def : Pat<(v32i8 (X86VPerm2x128 VR256:$src1, VR256:$src2, (i8 imm:$imm))),
          (VPERM2F128rr VR256:$src1, VR256:$src2, imm:$imm)>;
def : Pat<(v16i16 (X86VPerm2x128 VR256:$src1, VR256:$src2, (i8 imm:$imm))),
          (VPERM2F128rr VR256:$src1, VR256:$src2, imm:$imm)>;

def : Pat<(v8i32 (X86VPerm2x128 VR256:$src1,
                  (bc_v8i32 (loadv4i64 addr:$src2)), (i8 imm:$imm))),
          (VPERM2F128rm VR256:$src1, addr:$src2, imm:$imm)>;
def : Pat<(v4i64 (X86VPerm2x128 VR256:$src1,
                  (loadv4i64 addr:$src2), (i8 imm:$imm))),
          (VPERM2F128rm VR256:$src1, addr:$src2, imm:$imm)>;
def : Pat<(v32i8 (X86VPerm2x128 VR256:$src1,
                  (bc_v32i8 (loadv4i64 addr:$src2)), (i8 imm:$imm))),
          (VPERM2F128rm VR256:$src1, addr:$src2, imm:$imm)>;
def : Pat<(v16i16 (X86VPerm2x128 VR256:$src1,
                  (bc_v16i16 (loadv4i64 addr:$src2)), (i8 imm:$imm))),
          (VPERM2F128rm VR256:$src1, addr:$src2, imm:$imm)>;
}

//===----------------------------------------------------------------------===//
// VZERO - Zero YMM registers
//
let Defs = [YMM0, YMM1, YMM2, YMM3, YMM4, YMM5, YMM6, YMM7,
            YMM8, YMM9, YMM10, YMM11, YMM12, YMM13, YMM14, YMM15] in {
  // Zero All YMM registers
  def VZEROALL : I<0x77, RawFrm, (outs), (ins), "vzeroall",
                  [(int_x86_avx_vzeroall)]>, TB, VEX, VEX_L, Requires<[HasAVX]>;

  // Zero Upper bits of YMM registers
  def VZEROUPPER : I<0x77, RawFrm, (outs), (ins), "vzeroupper",
                     [(int_x86_avx_vzeroupper)]>, TB, VEX, Requires<[HasAVX]>;
}

//===----------------------------------------------------------------------===//
// Half precision conversion instructions
//===----------------------------------------------------------------------===//
multiclass f16c_ph2ps<RegisterClass RC, X86MemOperand x86memop, Intrinsic Int> {
  def rr : I<0x13, MRMSrcReg, (outs RC:$dst), (ins VR128:$src),
             "vcvtph2ps\t{$src, $dst|$dst, $src}",
             [(set RC:$dst, (Int VR128:$src))]>,
             T8, OpSize, VEX;
  let neverHasSideEffects = 1, mayLoad = 1 in
  def rm : I<0x13, MRMSrcMem, (outs RC:$dst), (ins x86memop:$src),
             "vcvtph2ps\t{$src, $dst|$dst, $src}", []>, T8, OpSize, VEX;
}

multiclass f16c_ps2ph<RegisterClass RC, X86MemOperand x86memop, Intrinsic Int> {
  def rr : Ii8<0x1D, MRMDestReg, (outs VR128:$dst),
               (ins RC:$src1, i32i8imm:$src2),
               "vcvtps2ph\t{$src2, $src1, $dst|$dst, $src1, $src2}",
               [(set VR128:$dst, (Int RC:$src1, imm:$src2))]>,
               TA, OpSize, VEX;
  let neverHasSideEffects = 1, mayStore = 1 in
  def mr : Ii8<0x1D, MRMDestMem, (outs),
               (ins x86memop:$dst, RC:$src1, i32i8imm:$src2),
               "vcvtps2ph\t{$src2, $src1, $dst|$dst, $src1, $src2}", []>,
               TA, OpSize, VEX;
}

let Predicates = [HasF16C] in {
  defm VCVTPH2PS  : f16c_ph2ps<VR128, f64mem, int_x86_vcvtph2ps_128>;
  defm VCVTPH2PSY : f16c_ph2ps<VR256, f128mem, int_x86_vcvtph2ps_256>, VEX_L;
  defm VCVTPS2PH  : f16c_ps2ph<VR128, f64mem, int_x86_vcvtps2ph_128>;
  defm VCVTPS2PHY : f16c_ps2ph<VR256, f128mem, int_x86_vcvtps2ph_256>, VEX_L;
}

//===----------------------------------------------------------------------===//
// AVX2 Instructions
//===----------------------------------------------------------------------===//

/// AVX2_binop_rmi_int - AVX2 binary operator with 8-bit immediate
multiclass AVX2_binop_rmi_int<bits<8> opc, string OpcodeStr,
                 Intrinsic IntId, RegisterClass RC, PatFrag memop_frag,
                 X86MemOperand x86memop> {
  let isCommutable = 1 in
  def rri : AVX2AIi8<opc, MRMSrcReg, (outs RC:$dst),
        (ins RC:$src1, RC:$src2, u32u8imm:$src3),
        !strconcat(OpcodeStr,
            "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
        [(set RC:$dst, (IntId RC:$src1, RC:$src2, imm:$src3))]>,
        VEX_4V;
  def rmi : AVX2AIi8<opc, MRMSrcMem, (outs RC:$dst),
        (ins RC:$src1, x86memop:$src2, u32u8imm:$src3),
        !strconcat(OpcodeStr,
            "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
        [(set RC:$dst,
          (IntId RC:$src1,
           (bitconvert (memop_frag addr:$src2)), imm:$src3))]>,
        VEX_4V;
}

let isCommutable = 0 in {
defm VPBLENDD : AVX2_binop_rmi_int<0x02, "vpblendd", int_x86_avx2_pblendd_128,
                                   VR128, loadv2i64, i128mem>;
defm VPBLENDDY : AVX2_binop_rmi_int<0x02, "vpblendd", int_x86_avx2_pblendd_256,
                                    VR256, loadv4i64, i256mem>, VEX_L;
}

def : Pat<(v4i32 (X86Blendi (v4i32 VR128:$src1), (v4i32 VR128:$src2),
                  imm:$mask)),
          (VPBLENDDrri VR128:$src1, VR128:$src2, imm:$mask)>;
def : Pat<(v8i32 (X86Blendi (v8i32 VR256:$src1), (v8i32 VR256:$src2),
                  imm:$mask)),
          (VPBLENDDYrri VR256:$src1, VR256:$src2, imm:$mask)>;

//===----------------------------------------------------------------------===//
// VPBROADCAST - Load from memory and broadcast to all elements of the
//               destination operand
//
multiclass avx2_broadcast<bits<8> opc, string OpcodeStr,
                          X86MemOperand x86memop, PatFrag ld_frag,
                          Intrinsic Int128, Intrinsic Int256> {
  def rr : AVX28I<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
                  !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                  [(set VR128:$dst, (Int128 VR128:$src))]>, VEX;
  def rm : AVX28I<opc, MRMSrcMem, (outs VR128:$dst), (ins x86memop:$src),
                  !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                  [(set VR128:$dst,
                    (Int128 (scalar_to_vector (ld_frag addr:$src))))]>, VEX;
  def Yrr : AVX28I<opc, MRMSrcReg, (outs VR256:$dst), (ins VR128:$src),
                   !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                   [(set VR256:$dst, (Int256 VR128:$src))]>, VEX, VEX_L;
  def Yrm : AVX28I<opc, MRMSrcMem, (outs VR256:$dst), (ins x86memop:$src),
                   !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                   [(set VR256:$dst,
                    (Int256 (scalar_to_vector (ld_frag addr:$src))))]>,
                   VEX, VEX_L;
}

defm VPBROADCASTB  : avx2_broadcast<0x78, "vpbroadcastb", i8mem, loadi8,
                                    int_x86_avx2_pbroadcastb_128,
                                    int_x86_avx2_pbroadcastb_256>;
defm VPBROADCASTW  : avx2_broadcast<0x79, "vpbroadcastw", i16mem, loadi16,
                                    int_x86_avx2_pbroadcastw_128,
                                    int_x86_avx2_pbroadcastw_256>;
defm VPBROADCASTD  : avx2_broadcast<0x58, "vpbroadcastd", i32mem, loadi32,
                                    int_x86_avx2_pbroadcastd_128,
                                    int_x86_avx2_pbroadcastd_256>;
defm VPBROADCASTQ  : avx2_broadcast<0x59, "vpbroadcastq", i64mem, loadi64,
                                    int_x86_avx2_pbroadcastq_128,
                                    int_x86_avx2_pbroadcastq_256>;

let Predicates = [HasAVX2] in {
  def : Pat<(v16i8 (X86VBroadcast (loadi8 addr:$src))),
          (VPBROADCASTBrm addr:$src)>;
  def : Pat<(v32i8 (X86VBroadcast (loadi8 addr:$src))),
          (VPBROADCASTBYrm addr:$src)>;
  def : Pat<(v8i16 (X86VBroadcast (loadi16 addr:$src))),
          (VPBROADCASTWrm addr:$src)>;
  def : Pat<(v16i16 (X86VBroadcast (loadi16 addr:$src))),
          (VPBROADCASTWYrm addr:$src)>;
  def : Pat<(v4i32 (X86VBroadcast (loadi32 addr:$src))),
          (VPBROADCASTDrm addr:$src)>;
  def : Pat<(v8i32 (X86VBroadcast (loadi32 addr:$src))),
          (VPBROADCASTDYrm addr:$src)>;
  def : Pat<(v2i64 (X86VBroadcast (loadi64 addr:$src))),
          (VPBROADCASTQrm addr:$src)>;
  def : Pat<(v4i64 (X86VBroadcast (loadi64 addr:$src))),
          (VPBROADCASTQYrm addr:$src)>;

  def : Pat<(v16i8 (X86VBroadcast (v16i8 VR128:$src))),
          (VPBROADCASTBrr VR128:$src)>;
  def : Pat<(v32i8 (X86VBroadcast (v16i8 VR128:$src))),
          (VPBROADCASTBYrr VR128:$src)>;
  def : Pat<(v8i16 (X86VBroadcast (v8i16 VR128:$src))),
          (VPBROADCASTWrr VR128:$src)>;
  def : Pat<(v16i16 (X86VBroadcast (v8i16 VR128:$src))),
          (VPBROADCASTWYrr VR128:$src)>;
  def : Pat<(v4i32 (X86VBroadcast (v4i32 VR128:$src))),
          (VPBROADCASTDrr VR128:$src)>;
  def : Pat<(v8i32 (X86VBroadcast (v4i32 VR128:$src))),
          (VPBROADCASTDYrr VR128:$src)>;
  def : Pat<(v2i64 (X86VBroadcast (v2i64 VR128:$src))),
          (VPBROADCASTQrr VR128:$src)>;
  def : Pat<(v4i64 (X86VBroadcast (v2i64 VR128:$src))),
          (VPBROADCASTQYrr VR128:$src)>;
  def : Pat<(v4f32 (X86VBroadcast (v4f32 VR128:$src))),
          (VBROADCASTSSrr VR128:$src)>;
  def : Pat<(v8f32 (X86VBroadcast (v4f32 VR128:$src))),
          (VBROADCASTSSYrr VR128:$src)>;
  def : Pat<(v2f64 (X86VBroadcast (v2f64 VR128:$src))),
          (VPBROADCASTQrr VR128:$src)>;
  def : Pat<(v4f64 (X86VBroadcast (v2f64 VR128:$src))),
          (VBROADCASTSDYrr VR128:$src)>;

  // Provide fallback in case the load node that is used in the patterns above
  // is used by additional users, which prevents the pattern selection.
  let AddedComplexity = 20 in {
    def : Pat<(v4f32 (X86VBroadcast FR32:$src)),
              (VBROADCASTSSrr (COPY_TO_REGCLASS FR32:$src, VR128))>;
    def : Pat<(v8f32 (X86VBroadcast FR32:$src)),
              (VBROADCASTSSYrr (COPY_TO_REGCLASS FR32:$src, VR128))>;
    def : Pat<(v4f64 (X86VBroadcast FR64:$src)),
              (VBROADCASTSDYrr (COPY_TO_REGCLASS FR64:$src, VR128))>;

    def : Pat<(v4i32 (X86VBroadcast GR32:$src)),
              (VBROADCASTSSrr (COPY_TO_REGCLASS GR32:$src, VR128))>;
    def : Pat<(v8i32 (X86VBroadcast GR32:$src)),
              (VBROADCASTSSYrr (COPY_TO_REGCLASS GR32:$src, VR128))>;
    def : Pat<(v4i64 (X86VBroadcast GR64:$src)),
              (VBROADCASTSDYrr (COPY_TO_REGCLASS GR64:$src, VR128))>;
  }
}

// AVX1 broadcast patterns
let Predicates = [HasAVX1Only] in {
def : Pat<(v8i32 (X86VBroadcast (loadi32 addr:$src))),
          (VBROADCASTSSYrm addr:$src)>;
def : Pat<(v4i64 (X86VBroadcast (loadi64 addr:$src))),
          (VBROADCASTSDYrm addr:$src)>;
def : Pat<(v4i32 (X86VBroadcast (loadi32 addr:$src))),
          (VBROADCASTSSrm addr:$src)>;
}

let Predicates = [HasAVX] in {
def : Pat<(v8f32 (X86VBroadcast (loadf32 addr:$src))),
          (VBROADCASTSSYrm addr:$src)>;
def : Pat<(v4f64 (X86VBroadcast (loadf64 addr:$src))),
          (VBROADCASTSDYrm addr:$src)>;
def : Pat<(v4f32 (X86VBroadcast (loadf32 addr:$src))),
          (VBROADCASTSSrm addr:$src)>;

  // Provide fallback in case the load node that is used in the patterns above
  // is used by additional users, which prevents the pattern selection.
  let AddedComplexity = 20 in {
  // 128bit broadcasts:
  def : Pat<(v4f32 (X86VBroadcast FR32:$src)),
            (VPSHUFDri (COPY_TO_REGCLASS FR32:$src, VR128), 0)>;
  def : Pat<(v8f32 (X86VBroadcast FR32:$src)),
            (VINSERTF128rr (INSERT_SUBREG (v8f32 (IMPLICIT_DEF)),
              (VPSHUFDri (COPY_TO_REGCLASS FR32:$src, VR128), 0), sub_xmm),
              (VPSHUFDri (COPY_TO_REGCLASS FR32:$src, VR128), 0), 1)>;
  def : Pat<(v4f64 (X86VBroadcast FR64:$src)),
            (VINSERTF128rr (INSERT_SUBREG (v4f64 (IMPLICIT_DEF)),
              (VPSHUFDri (COPY_TO_REGCLASS FR64:$src, VR128), 0x44), sub_xmm),
              (VPSHUFDri (COPY_TO_REGCLASS FR64:$src, VR128), 0x44), 1)>;

  def : Pat<(v4i32 (X86VBroadcast GR32:$src)),
            (VPSHUFDri (COPY_TO_REGCLASS GR32:$src, VR128), 0)>;
  def : Pat<(v8i32 (X86VBroadcast GR32:$src)),
            (VINSERTF128rr (INSERT_SUBREG (v8i32 (IMPLICIT_DEF)),
              (VPSHUFDri (COPY_TO_REGCLASS GR32:$src, VR128), 0), sub_xmm),
              (VPSHUFDri (COPY_TO_REGCLASS GR32:$src, VR128), 0), 1)>;
  def : Pat<(v4i64 (X86VBroadcast GR64:$src)),
            (VINSERTF128rr (INSERT_SUBREG (v4i64 (IMPLICIT_DEF)),
              (VPSHUFDri (COPY_TO_REGCLASS GR64:$src, VR128), 0x44), sub_xmm),
              (VPSHUFDri (COPY_TO_REGCLASS GR64:$src, VR128), 0x44), 1)>;
  }
}

//===----------------------------------------------------------------------===//
// VPERM - Permute instructions
//

multiclass avx2_perm<bits<8> opc, string OpcodeStr, PatFrag mem_frag,
                     ValueType OpVT> {
  def Yrr : AVX28I<opc, MRMSrcReg, (outs VR256:$dst),
                   (ins VR256:$src1, VR256:$src2),
                   !strconcat(OpcodeStr,
                       "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                   [(set VR256:$dst,
                     (OpVT (X86VPermv VR256:$src1, VR256:$src2)))]>,
                   VEX_4V, VEX_L;
  def Yrm : AVX28I<opc, MRMSrcMem, (outs VR256:$dst),
                   (ins VR256:$src1, i256mem:$src2),
                   !strconcat(OpcodeStr,
                       "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                   [(set VR256:$dst,
                     (OpVT (X86VPermv VR256:$src1,
                            (bitconvert (mem_frag addr:$src2)))))]>,
                   VEX_4V, VEX_L;
}

defm VPERMD : avx2_perm<0x36, "vpermd", loadv4i64, v8i32>;
let ExeDomain = SSEPackedSingle in
defm VPERMPS : avx2_perm<0x16, "vpermps", loadv8f32, v8f32>;

multiclass avx2_perm_imm<bits<8> opc, string OpcodeStr, PatFrag mem_frag,
                         ValueType OpVT> {
  def Yri : AVX2AIi8<opc, MRMSrcReg, (outs VR256:$dst),
                     (ins VR256:$src1, i8imm:$src2),
                     !strconcat(OpcodeStr,
                         "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                     [(set VR256:$dst,
                       (OpVT (X86VPermi VR256:$src1, (i8 imm:$src2))))]>,
                     VEX, VEX_L;
  def Ymi : AVX2AIi8<opc, MRMSrcMem, (outs VR256:$dst),
                     (ins i256mem:$src1, i8imm:$src2),
                     !strconcat(OpcodeStr,
                         "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                     [(set VR256:$dst,
                       (OpVT (X86VPermi (mem_frag addr:$src1),
                              (i8 imm:$src2))))]>, VEX, VEX_L;
}

defm VPERMQ : avx2_perm_imm<0x00, "vpermq", loadv4i64, v4i64>, VEX_W;
let ExeDomain = SSEPackedDouble in
defm VPERMPD : avx2_perm_imm<0x01, "vpermpd", loadv4f64, v4f64>, VEX_W;

//===----------------------------------------------------------------------===//
// VPERM2I128 - Permute Floating-Point Values in 128-bit chunks
//
def VPERM2I128rr : AVX2AIi8<0x46, MRMSrcReg, (outs VR256:$dst),
          (ins VR256:$src1, VR256:$src2, i8imm:$src3),
          "vperm2i128\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
          [(set VR256:$dst, (v4i64 (X86VPerm2x128 VR256:$src1, VR256:$src2,
                            (i8 imm:$src3))))]>, VEX_4V, VEX_L;
def VPERM2I128rm : AVX2AIi8<0x46, MRMSrcMem, (outs VR256:$dst),
          (ins VR256:$src1, f256mem:$src2, i8imm:$src3),
          "vperm2i128\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
          [(set VR256:$dst, (X86VPerm2x128 VR256:$src1, (loadv4i64 addr:$src2),
                             (i8 imm:$src3)))]>, VEX_4V, VEX_L;

let Predicates = [HasAVX2] in {
def : Pat<(v8i32 (X86VPerm2x128 VR256:$src1, VR256:$src2, (i8 imm:$imm))),
          (VPERM2I128rr VR256:$src1, VR256:$src2, imm:$imm)>;
def : Pat<(v32i8 (X86VPerm2x128 VR256:$src1, VR256:$src2, (i8 imm:$imm))),
          (VPERM2I128rr VR256:$src1, VR256:$src2, imm:$imm)>;
def : Pat<(v16i16 (X86VPerm2x128 VR256:$src1, VR256:$src2, (i8 imm:$imm))),
          (VPERM2I128rr VR256:$src1, VR256:$src2, imm:$imm)>;

def : Pat<(v32i8 (X86VPerm2x128 VR256:$src1, (bc_v32i8 (loadv4i64 addr:$src2)),
                  (i8 imm:$imm))),
          (VPERM2I128rm VR256:$src1, addr:$src2, imm:$imm)>;
def : Pat<(v16i16 (X86VPerm2x128 VR256:$src1,
                   (bc_v16i16 (loadv4i64 addr:$src2)), (i8 imm:$imm))),
          (VPERM2I128rm VR256:$src1, addr:$src2, imm:$imm)>;
def : Pat<(v8i32 (X86VPerm2x128 VR256:$src1, (bc_v8i32 (loadv4i64 addr:$src2)),
                  (i8 imm:$imm))),
          (VPERM2I128rm VR256:$src1, addr:$src2, imm:$imm)>;
}


//===----------------------------------------------------------------------===//
// VINSERTI128 - Insert packed integer values
//
let neverHasSideEffects = 1 in {
def VINSERTI128rr : AVX2AIi8<0x38, MRMSrcReg, (outs VR256:$dst),
          (ins VR256:$src1, VR128:$src2, i8imm:$src3),
          "vinserti128\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
          []>, VEX_4V, VEX_L;
let mayLoad = 1 in
def VINSERTI128rm : AVX2AIi8<0x38, MRMSrcMem, (outs VR256:$dst),
          (ins VR256:$src1, i128mem:$src2, i8imm:$src3),
          "vinserti128\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
          []>, VEX_4V, VEX_L;
}

let Predicates = [HasAVX2] in {
def : Pat<(vinsert128_insert:$ins (v4i64 VR256:$src1), (v2i64 VR128:$src2),
                                   (iPTR imm)),
          (VINSERTI128rr VR256:$src1, VR128:$src2,
                         (INSERT_get_vinsert128_imm VR256:$ins))>;
def : Pat<(vinsert128_insert:$ins (v8i32 VR256:$src1), (v4i32 VR128:$src2),
                                   (iPTR imm)),
          (VINSERTI128rr VR256:$src1, VR128:$src2,
                         (INSERT_get_vinsert128_imm VR256:$ins))>;
def : Pat<(vinsert128_insert:$ins (v32i8 VR256:$src1), (v16i8 VR128:$src2),
                                   (iPTR imm)),
          (VINSERTI128rr VR256:$src1, VR128:$src2,
                         (INSERT_get_vinsert128_imm VR256:$ins))>;
def : Pat<(vinsert128_insert:$ins (v16i16 VR256:$src1), (v8i16 VR128:$src2),
                                   (iPTR imm)),
          (VINSERTI128rr VR256:$src1, VR128:$src2,
                         (INSERT_get_vinsert128_imm VR256:$ins))>;

def : Pat<(vinsert128_insert:$ins (v4i64 VR256:$src1), (loadv2i64 addr:$src2),
                                   (iPTR imm)),
          (VINSERTI128rm VR256:$src1, addr:$src2,
                         (INSERT_get_vinsert128_imm VR256:$ins))>;
def : Pat<(vinsert128_insert:$ins (v8i32 VR256:$src1),
                                   (bc_v4i32 (loadv2i64 addr:$src2)),
                                   (iPTR imm)),
          (VINSERTI128rm VR256:$src1, addr:$src2,
                         (INSERT_get_vinsert128_imm VR256:$ins))>;
def : Pat<(vinsert128_insert:$ins (v32i8 VR256:$src1),
                                   (bc_v16i8 (loadv2i64 addr:$src2)),
                                   (iPTR imm)),
          (VINSERTI128rm VR256:$src1, addr:$src2,
                         (INSERT_get_vinsert128_imm VR256:$ins))>;
def : Pat<(vinsert128_insert:$ins (v16i16 VR256:$src1),
                                   (bc_v8i16 (loadv2i64 addr:$src2)),
                                   (iPTR imm)),
          (VINSERTI128rm VR256:$src1, addr:$src2,
                         (INSERT_get_vinsert128_imm VR256:$ins))>;
}

//===----------------------------------------------------------------------===//
// VEXTRACTI128 - Extract packed integer values
//
def VEXTRACTI128rr : AVX2AIi8<0x39, MRMDestReg, (outs VR128:$dst),
          (ins VR256:$src1, i8imm:$src2),
          "vextracti128\t{$src2, $src1, $dst|$dst, $src1, $src2}",
          [(set VR128:$dst,
            (int_x86_avx2_vextracti128 VR256:$src1, imm:$src2))]>,
          VEX, VEX_L;
let neverHasSideEffects = 1, mayStore = 1 in
def VEXTRACTI128mr : AVX2AIi8<0x39, MRMDestMem, (outs),
          (ins i128mem:$dst, VR256:$src1, i8imm:$src2),
          "vextracti128\t{$src2, $src1, $dst|$dst, $src1, $src2}", []>,
          VEX, VEX_L;

let Predicates = [HasAVX2] in {
def : Pat<(vextract128_extract:$ext VR256:$src1, (iPTR imm)),
          (v2i64 (VEXTRACTI128rr
                    (v4i64 VR256:$src1),
                    (EXTRACT_get_vextract128_imm VR128:$ext)))>;
def : Pat<(vextract128_extract:$ext VR256:$src1, (iPTR imm)),
          (v4i32 (VEXTRACTI128rr
                    (v8i32 VR256:$src1),
                    (EXTRACT_get_vextract128_imm VR128:$ext)))>;
def : Pat<(vextract128_extract:$ext VR256:$src1, (iPTR imm)),
          (v8i16 (VEXTRACTI128rr
                    (v16i16 VR256:$src1),
                    (EXTRACT_get_vextract128_imm VR128:$ext)))>;
def : Pat<(vextract128_extract:$ext VR256:$src1, (iPTR imm)),
          (v16i8 (VEXTRACTI128rr
                    (v32i8 VR256:$src1),
                    (EXTRACT_get_vextract128_imm VR128:$ext)))>;

def : Pat<(store (v2i64 (vextract128_extract:$ext (v4i64 VR256:$src1),
                         (iPTR imm))), addr:$dst),
          (VEXTRACTI128mr addr:$dst, VR256:$src1,
           (EXTRACT_get_vextract128_imm VR128:$ext))>;
def : Pat<(store (v4i32 (vextract128_extract:$ext (v8i32 VR256:$src1),
                         (iPTR imm))), addr:$dst),
          (VEXTRACTI128mr addr:$dst, VR256:$src1,
           (EXTRACT_get_vextract128_imm VR128:$ext))>;
def : Pat<(store (v8i16 (vextract128_extract:$ext (v16i16 VR256:$src1),
                         (iPTR imm))), addr:$dst),
          (VEXTRACTI128mr addr:$dst, VR256:$src1,
           (EXTRACT_get_vextract128_imm VR128:$ext))>;
def : Pat<(store (v16i8 (vextract128_extract:$ext (v32i8 VR256:$src1),
                         (iPTR imm))), addr:$dst),
          (VEXTRACTI128mr addr:$dst, VR256:$src1,
           (EXTRACT_get_vextract128_imm VR128:$ext))>;
}

//===----------------------------------------------------------------------===//
// VPMASKMOV - Conditional SIMD Integer Packed Loads and Stores
//
multiclass avx2_pmovmask<string OpcodeStr,
                         Intrinsic IntLd128, Intrinsic IntLd256,
                         Intrinsic IntSt128, Intrinsic IntSt256> {
  def rm  : AVX28I<0x8c, MRMSrcMem, (outs VR128:$dst),
             (ins VR128:$src1, i128mem:$src2),
             !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set VR128:$dst, (IntLd128 addr:$src2, VR128:$src1))]>, VEX_4V;
  def Yrm : AVX28I<0x8c, MRMSrcMem, (outs VR256:$dst),
             (ins VR256:$src1, i256mem:$src2),
             !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set VR256:$dst, (IntLd256 addr:$src2, VR256:$src1))]>,
             VEX_4V, VEX_L;
  def mr  : AVX28I<0x8e, MRMDestMem, (outs),
             (ins i128mem:$dst, VR128:$src1, VR128:$src2),
             !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(IntSt128 addr:$dst, VR128:$src1, VR128:$src2)]>, VEX_4V;
  def Ymr : AVX28I<0x8e, MRMDestMem, (outs),
             (ins i256mem:$dst, VR256:$src1, VR256:$src2),
             !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(IntSt256 addr:$dst, VR256:$src1, VR256:$src2)]>, VEX_4V, VEX_L;
}

defm VPMASKMOVD : avx2_pmovmask<"vpmaskmovd",
                                int_x86_avx2_maskload_d,
                                int_x86_avx2_maskload_d_256,
                                int_x86_avx2_maskstore_d,
                                int_x86_avx2_maskstore_d_256>;
defm VPMASKMOVQ : avx2_pmovmask<"vpmaskmovq",
                                int_x86_avx2_maskload_q,
                                int_x86_avx2_maskload_q_256,
                                int_x86_avx2_maskstore_q,
                                int_x86_avx2_maskstore_q_256>, VEX_W;


//===----------------------------------------------------------------------===//
// Variable Bit Shifts
//
multiclass avx2_var_shift<bits<8> opc, string OpcodeStr, SDNode OpNode,
                          ValueType vt128, ValueType vt256> {
  def rr  : AVX28I<opc, MRMSrcReg, (outs VR128:$dst),
             (ins VR128:$src1, VR128:$src2),
             !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set VR128:$dst,
               (vt128 (OpNode VR128:$src1, (vt128 VR128:$src2))))]>,
             VEX_4V;
  def rm  : AVX28I<opc, MRMSrcMem, (outs VR128:$dst),
             (ins VR128:$src1, i128mem:$src2),
             !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set VR128:$dst,
               (vt128 (OpNode VR128:$src1,
                       (vt128 (bitconvert (loadv2i64 addr:$src2))))))]>,
             VEX_4V;
  def Yrr : AVX28I<opc, MRMSrcReg, (outs VR256:$dst),
             (ins VR256:$src1, VR256:$src2),
             !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set VR256:$dst,
               (vt256 (OpNode VR256:$src1, (vt256 VR256:$src2))))]>,
             VEX_4V, VEX_L;
  def Yrm : AVX28I<opc, MRMSrcMem, (outs VR256:$dst),
             (ins VR256:$src1, i256mem:$src2),
             !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set VR256:$dst,
               (vt256 (OpNode VR256:$src1,
                       (vt256 (bitconvert (loadv4i64 addr:$src2))))))]>,
             VEX_4V, VEX_L;
}

defm VPSLLVD : avx2_var_shift<0x47, "vpsllvd", shl, v4i32, v8i32>;
defm VPSLLVQ : avx2_var_shift<0x47, "vpsllvq", shl, v2i64, v4i64>, VEX_W;
defm VPSRLVD : avx2_var_shift<0x45, "vpsrlvd", srl, v4i32, v8i32>;
defm VPSRLVQ : avx2_var_shift<0x45, "vpsrlvq", srl, v2i64, v4i64>, VEX_W;
defm VPSRAVD : avx2_var_shift<0x46, "vpsravd", sra, v4i32, v8i32>;

//===----------------------------------------------------------------------===//
// VGATHER - GATHER Operations
multiclass avx2_gather<bits<8> opc, string OpcodeStr, RegisterClass RC256,
                       X86MemOperand memop128, X86MemOperand memop256> {
  def rm  : AVX28I<opc, MRMSrcMem, (outs VR128:$dst, VR128:$mask_wb),
            (ins VR128:$src1, memop128:$src2, VR128:$mask),
            !strconcat(OpcodeStr,
              "\t{$mask, $src2, $dst|$dst, $src2, $mask}"),
            []>, VEX_4VOp3;
  def Yrm : AVX28I<opc, MRMSrcMem, (outs RC256:$dst, RC256:$mask_wb),
            (ins RC256:$src1, memop256:$src2, RC256:$mask),
            !strconcat(OpcodeStr,
              "\t{$mask, $src2, $dst|$dst, $src2, $mask}"),
            []>, VEX_4VOp3, VEX_L;
}

let mayLoad = 1, Constraints
  = "@earlyclobber $dst,@earlyclobber $mask_wb, $src1 = $dst, $mask = $mask_wb"
  in {
  defm VGATHERDPD : avx2_gather<0x92, "vgatherdpd", VR256, vx64mem, vx64mem>, VEX_W;
  defm VGATHERQPD : avx2_gather<0x93, "vgatherqpd", VR256, vx64mem, vy64mem>, VEX_W;
  defm VGATHERDPS : avx2_gather<0x92, "vgatherdps", VR256, vx32mem, vy32mem>;
  defm VGATHERQPS : avx2_gather<0x93, "vgatherqps", VR128, vx32mem, vy32mem>;
  defm VPGATHERDQ : avx2_gather<0x90, "vpgatherdq", VR256, vx64mem, vx64mem>, VEX_W;
  defm VPGATHERQQ : avx2_gather<0x91, "vpgatherqq", VR256, vx64mem, vy64mem>, VEX_W;
  defm VPGATHERDD : avx2_gather<0x90, "vpgatherdd", VR256, vx32mem, vy32mem>;
  defm VPGATHERQD : avx2_gather<0x91, "vpgatherqd", VR128, vx32mem, vy32mem>;
}
