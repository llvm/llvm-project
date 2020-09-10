//===- WebAssemblyInstrControl.td-WebAssembly control-flow ------*- tablegen -*-
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// WebAssembly control-flow code-gen constructs.
///
//===----------------------------------------------------------------------===//

let isBranch = 1, isTerminator = 1, hasCtrlDep = 1 in {
// The condition operand is a boolean value which WebAssembly represents as i32.
defm BR_IF : I<(outs), (ins bb_op:$dst, I32:$cond),
               (outs), (ins bb_op:$dst),
               [(brcond I32:$cond, bb:$dst)],
                "br_if   \t$dst, $cond", "br_if   \t$dst", 0x0d>;
let isCodeGenOnly = 1 in
defm BR_UNLESS : I<(outs), (ins bb_op:$dst, I32:$cond),
                   (outs), (ins bb_op:$dst), []>;
let isBarrier = 1 in
defm BR   : NRI<(outs), (ins bb_op:$dst),
                [(br bb:$dst)],
                "br      \t$dst", 0x0c>;
} // isBranch = 1, isTerminator = 1, hasCtrlDep = 1

def : Pat<(brcond (i32 (setne I32:$cond, 0)), bb:$dst),
          (BR_IF bb_op:$dst, I32:$cond)>;
def : Pat<(brcond (i32 (seteq I32:$cond, 0)), bb:$dst),
          (BR_UNLESS bb_op:$dst, I32:$cond)>;

// A list of branch targets enclosed in {} and separated by comma.
// Used by br_table only.
def BrListAsmOperand : AsmOperandClass { let Name = "BrList"; }
let OperandNamespace = "WebAssembly", OperandType = "OPERAND_BRLIST" in
def brlist : Operand<i32> {
  let ParserMatchClass = BrListAsmOperand;
  let PrintMethod = "printBrList";
}

// Duplicating a BR_TABLE is almost never a good idea. In particular, it can
// lead to some nasty irreducibility due to tail merging when the br_table is in
// a loop.
let isTerminator = 1, hasCtrlDep = 1, isBarrier = 1, isNotDuplicable = 1 in {

defm BR_TABLE_I32 : I<(outs), (ins I32:$index, variable_ops),
                      (outs), (ins brlist:$brl),
                      [(WebAssemblybr_table I32:$index)],
                      "br_table \t$index", "br_table \t$brl",
                      0x0e>;
// TODO: SelectionDAG's lowering insists on using a pointer as the index for
// jump tables, so in practice we don't ever use BR_TABLE_I64 in wasm32 mode
// currently.
defm BR_TABLE_I64 : I<(outs), (ins I64:$index, variable_ops),
                      (outs), (ins brlist:$brl),
                      [(WebAssemblybr_table I64:$index)],
                      "br_table \t$index", "br_table \t$brl",
                      0x0e>;
} // isTerminator = 1, hasCtrlDep = 1, isBarrier = 1, isNotDuplicable = 1

// This is technically a control-flow instruction, since all it affects is the
// IP.
defm NOP : NRI<(outs), (ins), [], "nop", 0x01>;

// Placemarkers to indicate the start or end of a block or loop scope.
// These use/clobber VALUE_STACK to prevent them from being moved into the
// middle of an expression tree.
let Uses = [VALUE_STACK], Defs = [VALUE_STACK] in {
defm BLOCK : NRI<(outs), (ins Signature:$sig), [], "block   \t$sig", 0x02>;
defm LOOP  : NRI<(outs), (ins Signature:$sig), [], "loop    \t$sig", 0x03>;

defm IF : I<(outs), (ins Signature:$sig, I32:$cond),
            (outs), (ins Signature:$sig),
            [], "if    \t$sig, $cond", "if    \t$sig", 0x04>;
defm ELSE : NRI<(outs), (ins), [], "else", 0x05>;

// END_BLOCK, END_LOOP, END_IF and END_FUNCTION are represented with the same
// opcode in wasm.
defm END_BLOCK : NRI<(outs), (ins), [], "end_block", 0x0b>;
defm END_LOOP  : NRI<(outs), (ins), [], "end_loop", 0x0b>;
defm END_IF    : NRI<(outs), (ins), [], "end_if", 0x0b>;
// Generic instruction, for disassembler.
let IsCanonical = 1 in
defm END       : NRI<(outs), (ins), [], "end", 0x0b>;
let isTerminator = 1, isBarrier = 1 in
defm END_FUNCTION : NRI<(outs), (ins), [], "end_function", 0x0b>;
} // Uses = [VALUE_STACK], Defs = [VALUE_STACK]


let hasCtrlDep = 1, isBarrier = 1 in {
let isTerminator = 1 in {
let isReturn = 1 in {

defm RETURN : I<(outs), (ins variable_ops), (outs), (ins),
                [(WebAssemblyreturn)],
                "return", "return", 0x0f>;
// Equivalent to RETURN, for use at the end of a function when wasm
// semantics return by falling off the end of the block.
let isCodeGenOnly = 1 in
defm FALLTHROUGH_RETURN : I<(outs), (ins variable_ops), (outs), (ins), []>;

} // isReturn = 1

let IsCanonical = 1, isTrap = 1 in
defm UNREACHABLE : NRI<(outs), (ins), [(trap)], "unreachable", 0x00>;

} // isTerminator = 1

// debugtrap explicitly returns despite trapping because it is supposed to just
// get the attention of the debugger. Unfortunately, because UNREACHABLE is a
// terminator, lowering debugtrap to UNREACHABLE can create an invalid
// MachineBasicBlock when there is additional code after it. Lower it to this
// non-terminator version instead.
// TODO: Actually execute the debugger statement when running on the Web
let isTrap = 1 in
defm DEBUG_UNREACHABLE : NRI<(outs), (ins), [(debugtrap)], "unreachable", 0x00>;

} // hasCtrlDep = 1, isBarrier = 1

//===----------------------------------------------------------------------===//
// Exception handling instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasExceptionHandling] in {

// Throwing an exception: throw / rethrow
let isTerminator = 1, hasCtrlDep = 1, isBarrier = 1 in {
defm THROW : I<(outs), (ins event_op:$tag, variable_ops),
               (outs), (ins event_op:$tag),
               [(WebAssemblythrow (WebAssemblywrapper texternalsym:$tag))],
               "throw   \t$tag", "throw   \t$tag", 0x08>;
defm RETHROW : I<(outs), (ins EXNREF:$exn), (outs), (ins), [],
                 "rethrow \t$exn", "rethrow", 0x09>;
// Pseudo instruction to be the lowering target of int_wasm_rethrow_in_catch
// intrinsic. Will be converted to the real rethrow instruction later.
let isPseudo = 1 in
defm RETHROW_IN_CATCH : NRI<(outs), (ins), [(int_wasm_rethrow_in_catch)],
                            "rethrow_in_catch", 0>;
} // isTerminator = 1, hasCtrlDep = 1, isBarrier = 1

// Region within which an exception is caught: try / end_try
let Uses = [VALUE_STACK], Defs = [VALUE_STACK] in {
defm TRY     : NRI<(outs), (ins Signature:$sig), [], "try     \t$sig", 0x06>;
defm END_TRY : NRI<(outs), (ins), [], "end_try", 0x0b>;
} // Uses = [VALUE_STACK], Defs = [VALUE_STACK]

// Catching an exception: catch / extract_exception
let hasCtrlDep = 1, hasSideEffects = 1 in
defm CATCH : I<(outs EXNREF:$dst), (ins), (outs), (ins), [],
               "catch   \t$dst", "catch", 0x07>;

// Querying / extracing exception: br_on_exn
// br_on_exn queries an exnref to see if it matches the corresponding exception
// tag index. If true it branches to the given label and pushes the
// corresponding argument values of the exception onto the stack.
let isBranch = 1, isTerminator = 1, hasCtrlDep = 1 in
defm BR_ON_EXN : I<(outs), (ins bb_op:$dst, event_op:$tag, EXNREF:$exn),
                   (outs), (ins bb_op:$dst, event_op:$tag), [],
                   "br_on_exn \t$dst, $tag, $exn", "br_on_exn \t$dst, $tag",
                   0x0a>;
// This is a pseudo instruction that simulates popping a value from stack, which
// has been pushed by br_on_exn
let isCodeGenOnly = 1, hasSideEffects = 1 in
defm EXTRACT_EXCEPTION_I32 : NRI<(outs I32:$dst), (ins),
                                 [(set I32:$dst, (int_wasm_extract_exception))],
                                 "extract_exception\t$dst">;

// Pseudo instructions: cleanupret / catchret
let isTerminator = 1, hasSideEffects = 1, isBarrier = 1, hasCtrlDep = 1,
    isPseudo = 1, isEHScopeReturn = 1 in {
  defm CLEANUPRET : NRI<(outs), (ins), [(cleanupret)], "cleanupret", 0>;
  defm CATCHRET : NRI<(outs), (ins bb_op:$dst, bb_op:$from),
                      [(catchret bb:$dst, bb:$from)], "catchret", 0>;
} // isTerminator = 1, hasSideEffects = 1, isBarrier = 1, hasCtrlDep = 1,
  // isPseudo = 1, isEHScopeReturn = 1
} // Predicates = [HasExceptionHandling]
