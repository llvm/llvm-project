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

/*
 * TODO(jfb): Add the following.
 *
 * block: a fixed-length sequence of statements
 * if: if statement
 * do_while: do while statement, basically a loop with a conditional branch
 * forever: infinite loop statement (like while (1)), basically an unconditional
 *          branch (back to the top of the loop)
 * continue: continue to start of nested loop
 * break: break to end from nested loop or block
 * switch: switch statement with fallthrough
 */

let isBranch = 1, isTerminator = 1, hasCtrlDep = 1 in {
def BR_IF_ : I<(outs), (ins bb_op:$dst, I32:$a),
               [(brcond I32:$a, bb:$dst)],
               "br_if $dst, $a">;
let isBarrier = 1 in {
def BR   : I<(outs), (ins bb_op:$dst),
             [(br bb:$dst)],
             "br $dst">;
} // isBarrier = 1
} // isBranch = 1, isTerminator = 1, hasCtrlDep = 1

// TODO: SelectionDAG's lowering insists on using a pointer as the index for
// jump tables, so in practice we don't ever use SWITCH_I64 in wasm32 mode
// currently.
let isTerminator = 1, hasCtrlDep = 1, isBarrier = 1 in {
def SWITCH_I32 : I<(outs), (ins I32:$index, variable_ops),
                   [(WebAssemblyswitch I32:$index)]>;
def SWITCH_I64 : I<(outs), (ins I64:$index, variable_ops),
                   [(WebAssemblyswitch I64:$index)]>;
} // isTerminator = 1, hasCtrlDep = 1, isBarrier = 1

// Placemarkers to indicate the start of a block or loop scope.
def BLOCK     : I<(outs), (ins bb_op:$dst), [], "block $dst">;
def LOOP      : I<(outs), (ins bb_op:$dst), [], "loop $dst">;

multiclass RETURN<WebAssemblyRegClass vt> {
  def RETURN_#vt : I<(outs), (ins vt:$val), [(WebAssemblyreturn vt:$val)],
                     "return $val">;
}
let isReturn = 1, isTerminator = 1, hasCtrlDep = 1, isBarrier = 1 in {
  defm : RETURN<I32>;
  defm : RETURN<I64>;
  defm : RETURN<F32>;
  defm : RETURN<F64>;
  def RETURN_VOID : I<(outs), (ins), [(WebAssemblyreturn)], "return">;
} // isReturn = 1, isTerminator = 1, hasCtrlDep = 1, isBarrier = 1
