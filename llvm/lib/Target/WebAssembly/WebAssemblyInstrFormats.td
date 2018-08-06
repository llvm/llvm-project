//=- WebAssemblyInstrFormats.td - WebAssembly Instr. Formats -*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// WebAssembly instruction format definitions.
///
//===----------------------------------------------------------------------===//

// WebAssembly Instruction Format.
// We instantiate 2 of these for every actual instruction (register based
// and stack based), see below.
class WebAssemblyInst<bits<32> inst, string asmstr, bit stack> : Instruction {
  field bits<32> Inst = inst; // Instruction encoding.
  field bit StackBased = stack;
  let Namespace   = "WebAssembly";
  let Pattern     = [];
  let AsmString   = asmstr;
}

// Normal instructions. Default instantiation of a WebAssemblyInst.
class NI<dag oops, dag iops, list<dag> pattern, bit stack, string asmstr = "",
         bits<32> inst = -1>
    : WebAssemblyInst<inst, asmstr, stack> {
  dag OutOperandList = oops;
  dag InOperandList  = iops;
  let Pattern        = pattern;
}

// Generates both register and stack based versions of one actual instruction.
// We have 2 sets of operands (oops & iops) for the register and stack
// based version of this instruction, as well as the corresponding asmstr.
// The register versions have virtual-register operands which correspond to wasm
// locals or stack locations. Each use and def of the register corresponds to an
// implicit get_local / set_local or access of stack operands in wasm. These
// instructions are used for ISel and all MI passes. The stack versions of the
// instructions do not have register operands (they implicitly operate on the
// stack), and get_locals and set_locals are explicit. The register instructions
// are converted to their corresponding stack instructions before lowering to
// MC.
// Every instruction should want to be based on this multi-class to guarantee
// there is always an equivalent pair of instructions.
multiclass I<dag oops_r, dag iops_r, dag oops_s, dag iops_s,
             list<dag> pattern_r, string asmstr_r = "", string asmstr_s = "",
             bits<32> inst = -1> {
  def "" : NI<oops_r, iops_r, pattern_r, 0, asmstr_r, inst>;
  def _S : NI<oops_s, iops_s, [], 1, asmstr_s, inst>;
}

// For instructions that have no register ops, so both sets are the same.
multiclass NRI<dag oops, dag iops, list<dag> pattern, string asmstr = "",
               bits<32> inst = -1> {
  defm "": I<oops, iops, oops, iops, pattern, asmstr, asmstr, inst>;
}

multiclass SIMD_I<dag oops_r, dag iops_r, dag oops_s, dag iops_s,
                  list<dag> pattern_r, string asmstr_r = "",
                  string asmstr_s = "", bits<32> inst = -1> {
  defm "" : I<oops_r, iops_r, oops_s, iops_s, pattern_r, asmstr_r, asmstr_s,
              inst>,
            Requires<[HasSIMD128]>;
}

multiclass ATOMIC_I<dag oops_r, dag iops_r, dag oops_s, dag iops_s,
                    list<dag> pattern_r, string asmstr_r = "",
                    string asmstr_s = "", bits<32> inst = -1> {
  defm "" : I<oops_r, iops_r, oops_s, iops_s, pattern_r, asmstr_r, asmstr_s,
              inst>,
            Requires<[HasAtomics]>;
}

// Unary and binary instructions, for the local types that WebAssembly supports.
multiclass UnaryInt<SDNode node, string name, bits<32> i32Inst,
                    bits<32> i64Inst> {
  defm _I32 : I<(outs I32:$dst), (ins I32:$src), (outs), (ins),
                [(set I32:$dst, (node I32:$src))],
                !strconcat("i32.", !strconcat(name, "\t$dst, $src")),
                !strconcat("i32.", name), i32Inst>;
  defm _I64 : I<(outs I64:$dst), (ins I64:$src), (outs), (ins),
                [(set I64:$dst, (node I64:$src))],
                !strconcat("i64.", !strconcat(name, "\t$dst, $src")),
                !strconcat("i64.", name), i64Inst>;
}
multiclass BinaryInt<SDNode node, string name, bits<32> i32Inst,
                     bits<32> i64Inst> {
  defm _I32 : I<(outs I32:$dst), (ins I32:$lhs, I32:$rhs), (outs), (ins),
                [(set I32:$dst, (node I32:$lhs, I32:$rhs))],
                !strconcat("i32.", !strconcat(name, "\t$dst, $lhs, $rhs")),
                !strconcat("i32.", name), i32Inst>;
  defm _I64 : I<(outs I64:$dst), (ins I64:$lhs, I64:$rhs), (outs), (ins),
                [(set I64:$dst, (node I64:$lhs, I64:$rhs))],
                !strconcat("i64.", !strconcat(name, "\t$dst, $lhs, $rhs")),
                !strconcat("i64.", name), i64Inst>;
}
multiclass UnaryFP<SDNode node, string name, bits<32> f32Inst,
                   bits<32> f64Inst> {
  defm _F32 : I<(outs F32:$dst), (ins F32:$src), (outs), (ins),
                [(set F32:$dst, (node F32:$src))],
                !strconcat("f32.", !strconcat(name, "\t$dst, $src")),
                !strconcat("f32.", name), f32Inst>;
  defm _F64 : I<(outs F64:$dst), (ins F64:$src), (outs), (ins),
                [(set F64:$dst, (node F64:$src))],
                !strconcat("f64.", !strconcat(name, "\t$dst, $src")),
                !strconcat("f64.", name), f64Inst>;
}
multiclass BinaryFP<SDNode node, string name, bits<32> f32Inst,
                    bits<32> f64Inst> {
  defm _F32 : I<(outs F32:$dst), (ins F32:$lhs, F32:$rhs), (outs), (ins),
                [(set F32:$dst, (node F32:$lhs, F32:$rhs))],
                !strconcat("f32.", !strconcat(name, "\t$dst, $lhs, $rhs")),
                !strconcat("f32.", name), f32Inst>;
  defm _F64 : I<(outs F64:$dst), (ins F64:$lhs, F64:$rhs), (outs), (ins),
                [(set F64:$dst, (node F64:$lhs, F64:$rhs))],
                !strconcat("f64.", !strconcat(name, "\t$dst, $lhs, $rhs")),
                !strconcat("f64.", name), f64Inst>;
}
multiclass SIMDBinary<SDNode node, SDNode fnode, string name> {
  defm _I8x16 : SIMD_I<(outs V128:$dst), (ins V128:$lhs, V128:$rhs),
                       (outs), (ins),
                       [(set (v16i8 V128:$dst), (node V128:$lhs, V128:$rhs))],
                       !strconcat("i8x16.",
                         !strconcat(name, "\t$dst, $lhs, $rhs")),
                       !strconcat("i8x16.", name)>;
  defm _I16x8 : SIMD_I<(outs V128:$dst), (ins V128:$lhs, V128:$rhs),
                       (outs), (ins),
                       [(set (v8i16 V128:$dst), (node V128:$lhs, V128:$rhs))],
                       !strconcat("i16x8.",
                         !strconcat(name, "\t$dst, $lhs, $rhs")),
                       !strconcat("i16x8.", name)>;
  defm _I32x4 : SIMD_I<(outs V128:$dst), (ins V128:$lhs, V128:$rhs),
                       (outs), (ins),
                       [(set (v4i32 V128:$dst), (node V128:$lhs, V128:$rhs))],
                       !strconcat("i32x4.",
                         !strconcat(name, "\t$dst, $lhs, $rhs")),
                       !strconcat("i32x4.", name)>;
  defm _F32x4 : SIMD_I<(outs V128:$dst), (ins V128:$lhs, V128:$rhs),
                       (outs), (ins),
                       [(set (v4f32 V128:$dst), (fnode V128:$lhs, V128:$rhs))],
                       !strconcat("f32x4.",
                         !strconcat(name, "\t$dst, $lhs, $rhs")),
                       !strconcat("f32x4.", name)>;
}
multiclass ComparisonInt<CondCode cond, string name, bits<32> i32Inst, bits<32> i64Inst> {
  defm _I32 : I<(outs I32:$dst), (ins I32:$lhs, I32:$rhs), (outs), (ins),
                [(set I32:$dst, (setcc I32:$lhs, I32:$rhs, cond))],
                !strconcat("i32.", !strconcat(name, "\t$dst, $lhs, $rhs")),
                !strconcat("i32.", name), i32Inst>;
  defm _I64 : I<(outs I32:$dst), (ins I64:$lhs, I64:$rhs), (outs), (ins),
                [(set I32:$dst, (setcc I64:$lhs, I64:$rhs, cond))],
                !strconcat("i64.", !strconcat(name, "\t$dst, $lhs, $rhs")),
                !strconcat("i64.", name), i64Inst>;
}
multiclass ComparisonFP<CondCode cond, string name, bits<32> f32Inst, bits<32> f64Inst> {
  defm _F32 : I<(outs I32:$dst), (ins F32:$lhs, F32:$rhs), (outs), (ins),
                [(set I32:$dst, (setcc F32:$lhs, F32:$rhs, cond))],
                !strconcat("f32.", !strconcat(name, "\t$dst, $lhs, $rhs")),
                !strconcat("f32.", name), f32Inst>;
  defm  _F64 : I<(outs I32:$dst), (ins F64:$lhs, F64:$rhs), (outs), (ins),
                [(set I32:$dst, (setcc F64:$lhs, F64:$rhs, cond))],
                !strconcat("f64.", !strconcat(name, "\t$dst, $lhs, $rhs")),
                !strconcat("f64.", name), f64Inst>;
}
