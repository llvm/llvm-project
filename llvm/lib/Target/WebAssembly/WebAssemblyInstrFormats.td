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
/// \brief WebAssembly instruction format definitions.
///
//===----------------------------------------------------------------------===//

// WebAssembly Instruction Format.
class WebAssemblyInst<string asmstr> : Instruction {
  field bits<0> Inst; // Instruction encoding.
  let Namespace   = "WebAssembly";
  let Pattern     = [];
  let AsmString   = asmstr;
}

// Normal instructions.
class I<dag oops, dag iops, list<dag> pattern, string asmstr = "">
    : WebAssemblyInst<asmstr> {
  dag OutOperandList = oops;
  dag InOperandList  = iops;
  let Pattern        = pattern;
}

class SIMD_I<dag oops, dag iops, list<dag> pattern, string asmstr = "">
    : I<oops, iops, pattern, asmstr>, Requires<[HasSIMD128]>;

// Unary and binary instructions, for the local types that WebAssembly supports.
multiclass UnaryInt<SDNode node, string name> {
  def _I32 : I<(outs I32:$dst), (ins I32:$src),
               [(set I32:$dst, (node I32:$src))],
               !strconcat("i32.", !strconcat(name, "\t$dst, $src"))>;
  def _I64 : I<(outs I64:$dst), (ins I64:$src),
               [(set I64:$dst, (node I64:$src))],
               !strconcat("i64.", !strconcat(name, "\t$dst, $src"))>;
}
multiclass BinaryInt<SDNode node, string name> {
  def _I32 : I<(outs I32:$dst), (ins I32:$lhs, I32:$rhs),
               [(set I32:$dst, (node I32:$lhs, I32:$rhs))],
               !strconcat("i32.", !strconcat(name, "\t$dst, $lhs, $rhs"))>;
  def _I64 : I<(outs I64:$dst), (ins I64:$lhs, I64:$rhs),
               [(set I64:$dst, (node I64:$lhs, I64:$rhs))],
               !strconcat("i64.", !strconcat(name, "\t$dst, $lhs, $rhs"))>;
}
multiclass UnaryFP<SDNode node, string name> {
  def _F32 : I<(outs F32:$dst), (ins F32:$src),
               [(set F32:$dst, (node F32:$src))],
               !strconcat("f32.", !strconcat(name, "\t$dst, $src"))>;
  def _F64 : I<(outs F64:$dst), (ins F64:$src),
               [(set F64:$dst, (node F64:$src))],
               !strconcat("f64.", !strconcat(name, "\t$dst, $src"))>;
}
multiclass BinaryFP<SDNode node, string name> {
  def _F32 : I<(outs F32:$dst), (ins F32:$lhs, F32:$rhs),
               [(set F32:$dst, (node F32:$lhs, F32:$rhs))],
               !strconcat("f32.", !strconcat(name, "\t$dst, $lhs, $rhs"))>;
  def _F64 : I<(outs F64:$dst), (ins F64:$lhs, F64:$rhs),
               [(set F64:$dst, (node F64:$lhs, F64:$rhs))],
               !strconcat("f64.", !strconcat(name, "\t$dst, $lhs, $rhs"))>;
}
multiclass SIMDBinary<SDNode node, SDNode fnode, string name> {
  def _I8x16 : SIMD_I<(outs V128:$dst), (ins V128:$lhs, V128:$rhs),
                      [(set (v16i8 V128:$dst), (node V128:$lhs, V128:$rhs))],
                      !strconcat("i8x16.", !strconcat(name, "\t$dst, $lhs, $rhs"))>;
  def _I16x8 : SIMD_I<(outs V128:$dst), (ins V128:$lhs, V128:$rhs),
                      [(set (v8i16 V128:$dst), (node V128:$lhs, V128:$rhs))],
                      !strconcat("i16x8.", !strconcat(name, "\t$dst, $lhs, $rhs"))>;
  def _I32x4 : SIMD_I<(outs V128:$dst), (ins V128:$lhs, V128:$rhs),
                      [(set (v4i32 V128:$dst), (node V128:$lhs, V128:$rhs))],
                      !strconcat("i32x4.", !strconcat(name, "\t$dst, $lhs, $rhs"))>;
  def _F32x4 : SIMD_I<(outs V128:$dst), (ins V128:$lhs, V128:$rhs),
                      [(set (v4f32 V128:$dst), (fnode V128:$lhs, V128:$rhs))],
                      !strconcat("f32x4.", !strconcat(name, "\t$dst, $lhs, $rhs"))>;

}
multiclass ComparisonInt<CondCode cond, string name> {
  def _I32 : I<(outs I32:$dst), (ins I32:$lhs, I32:$rhs),
               [(set I32:$dst, (setcc I32:$lhs, I32:$rhs, cond))],
               !strconcat("i32.", !strconcat(name, "\t$dst, $lhs, $rhs"))>;
  def _I64 : I<(outs I32:$dst), (ins I64:$lhs, I64:$rhs),
               [(set I32:$dst, (setcc I64:$lhs, I64:$rhs, cond))],
               !strconcat("i64.", !strconcat(name, "\t$dst, $lhs, $rhs"))>;
}
multiclass ComparisonFP<CondCode cond, string name> {
  def _F32 : I<(outs I32:$dst), (ins F32:$lhs, F32:$rhs),
               [(set I32:$dst, (setcc F32:$lhs, F32:$rhs, cond))],
               !strconcat("f32.", !strconcat(name, "\t$dst, $lhs, $rhs"))>;
  def _F64 : I<(outs I32:$dst), (ins F64:$lhs, F64:$rhs),
               [(set I32:$dst, (setcc F64:$lhs, F64:$rhs, cond))],
               !strconcat("f64.", !strconcat(name, "\t$dst, $lhs, $rhs"))>;
}
