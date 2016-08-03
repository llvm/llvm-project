//===- WebAssemblyInstrControl.td-WebAssembly control-flow ------*- tablegen -*-
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief WebAssembly control-flow code-gen constructs.
///
//===----------------------------------------------------------------------===//

let Defs = [ARGUMENTS] in {

let isBranch = 1, isTerminator = 1, hasCtrlDep = 1 in {
// The condition operand is a boolean value which WebAssembly represents as i32.
def BR_IF : I<(outs), (ins bb_op:$dst, I32:$cond),
              [(brcond I32:$cond, bb:$dst)],
               "br_if   \t$dst, $cond">;
let isCodeGenOnly = 1 in
def BR_UNLESS : I<(outs), (ins bb_op:$dst, I32:$cond), [],
                   "br_unless\t$dst, $cond">;
let isBarrier = 1 in {
def BR   : I<(outs), (ins bb_op:$dst),
             [(br bb:$dst)],
             "br      \t$dst">;
} // isBarrier = 1
} // isBranch = 1, isTerminator = 1, hasCtrlDep = 1

} // Defs = [ARGUMENTS]

def : Pat<(brcond (i32 (setne I32:$cond, 0)), bb:$dst),
          (BR_IF bb_op:$dst, I32:$cond)>;
def : Pat<(brcond (i32 (seteq I32:$cond, 0)), bb:$dst),
          (BR_UNLESS bb_op:$dst, I32:$cond)>;

let Defs = [ARGUMENTS] in {

// TODO: SelectionDAG's lowering insists on using a pointer as the index for
// jump tables, so in practice we don't ever use BR_TABLE_I64 in wasm32 mode
// currently.
// Set TSFlags{0} to 1 to indicate that the variable_ops are immediates.
// Set TSFlags{1} to 1 to indicate that the immediates represent labels.
let isTerminator = 1, hasCtrlDep = 1, isBarrier = 1 in {
def BR_TABLE_I32 : I<(outs), (ins I32:$index, variable_ops),
                     [(WebAssemblybr_table I32:$index)],
                     "br_table \t$index"> {
  let TSFlags{0} = 1;
  let TSFlags{1} = 1;
}
def BR_TABLE_I64 : I<(outs), (ins I64:$index, variable_ops),
                     [(WebAssemblybr_table I64:$index)],
                     "br_table \t$index"> {
  let TSFlags{0} = 1;
  let TSFlags{1} = 1;
}
} // isTerminator = 1, hasCtrlDep = 1, isBarrier = 1

// Placemarkers to indicate the start or end of a block or loop scope. These
// use/clobber EXPR_STACK to prevent them from being moved into the middle of
// an expression tree.
let Uses = [EXPR_STACK], Defs = [EXPR_STACK] in {
def BLOCK     : I<(outs), (ins), [], "block">;
def LOOP      : I<(outs), (ins), [], "loop">;
def END_BLOCK : I<(outs), (ins), [], "end_block">;
def END_LOOP  : I<(outs), (ins), [], "end_loop">;
} // Uses = [EXPR_STACK], Defs = [EXPR_STACK]

multiclass RETURN<WebAssemblyRegClass vt> {
  def RETURN_#vt : I<(outs), (ins vt:$val), [(WebAssemblyreturn vt:$val)],
                     "return  \t$val">;
  // Equivalent to RETURN_#vt, for use at the end of a function when wasm
  // semantics return by falling off the end of the block.
  let isCodeGenOnly = 1 in
  def FALLTHROUGH_RETURN_#vt : I<(outs), (ins vt:$val), []>;
}

multiclass SIMD_RETURN<ValueType vt> {
  def RETURN_#vt : SIMD_I<(outs), (ins V128:$val),
                          [(WebAssemblyreturn (vt V128:$val))],
                          "return  \t$val">;
  // Equivalent to RETURN_#vt, for use at the end of a function when wasm
  // semantics return by falling off the end of the block.
  let isCodeGenOnly = 1 in
  def FALLTHROUGH_RETURN_#vt : SIMD_I<(outs), (ins V128:$val), []>;
}

let isTerminator = 1, hasCtrlDep = 1, isBarrier = 1 in {
let isReturn = 1 in {
  defm : RETURN<I32>;
  defm : RETURN<I64>;
  defm : RETURN<F32>;
  defm : RETURN<F64>;
  defm : SIMD_RETURN<v16i8>;
  defm : SIMD_RETURN<v8i16>;
  defm : SIMD_RETURN<v4i32>;
  defm : SIMD_RETURN<v4f32>;

  def RETURN_VOID : I<(outs), (ins), [(WebAssemblyreturn)], "return">;

  // This is to RETURN_VOID what FALLTHROUGH_RETURN_#vt is to RETURN_#vt.
  let isCodeGenOnly = 1 in
  def FALLTHROUGH_RETURN_VOID : I<(outs), (ins), []>;
} // isReturn = 1
  def UNREACHABLE : I<(outs), (ins), [(trap)], "unreachable">;
} // isTerminator = 1, hasCtrlDep = 1, isBarrier = 1

} // Defs = [ARGUMENTS]
