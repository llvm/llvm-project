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

let isBranch = 1, isTerminator = 1, hasCtrlDep = 1 in {
def BR_IF : I<(outs), (ins I32:$a, bb_op:$dst),
              [(brcond I32:$a, bb:$dst)],
               "br_if   \t$a, $dst">;
let isBarrier = 1 in {
def BR   : I<(outs), (ins bb_op:$dst),
             [(br bb:$dst)],
             "br      \t$dst">;
} // isBarrier = 1
} // isBranch = 1, isTerminator = 1, hasCtrlDep = 1

// TODO: SelectionDAG's lowering insists on using a pointer as the index for
// jump tables, so in practice we don't ever use SWITCH_I64 in wasm32 mode
// currently.
let isTerminator = 1, hasCtrlDep = 1, isBarrier = 1 in {
def SWITCH_I32 : I<(outs), (ins I32:$index, variable_ops),
                   [(WebAssemblyswitch I32:$index)],
                   "switch  \t$index">;
def SWITCH_I64 : I<(outs), (ins I64:$index, variable_ops),
                   [(WebAssemblyswitch I64:$index)],
                   "switch  \t$index">;
} // isTerminator = 1, hasCtrlDep = 1, isBarrier = 1

// Placemarkers to indicate the start of a block or loop scope.
def BLOCK     : I<(outs), (ins bb_op:$dst), [], "block   \t$dst">;
def LOOP      : I<(outs), (ins bb_op:$dst), [], "loop    \t$dst">;

multiclass RETURN<WebAssemblyRegClass vt> {
  def RETURN_#vt : I<(outs), (ins vt:$val), [(WebAssemblyreturn vt:$val)],
                     "return  \t$val">;
}

let isTerminator = 1, hasCtrlDep = 1, isBarrier = 1 in {
let isReturn = 1 in {
  defm : RETURN<I32>;
  defm : RETURN<I64>;
  defm : RETURN<F32>;
  defm : RETURN<F64>;
  def RETURN_VOID : I<(outs), (ins), [(WebAssemblyreturn)], "return">;
} // isReturn = 1
  def UNREACHABLE : I<(outs), (ins), [(trap)], "unreachable">;
} // isTerminator = 1, hasCtrlDep = 1, isBarrier = 1
