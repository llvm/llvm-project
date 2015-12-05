//===- WebAssemblyInstrCall.td-WebAssembly Call codegen support -*- tablegen -*-
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief WebAssembly Call operand code-gen constructs.
///
//===----------------------------------------------------------------------===//

let Defs = [ARGUMENTS] in {

// The call sequence start/end LLVM-isms isn't useful to WebAssembly since it's
// a virtual ISA.
let isCodeGenOnly = 1 in {
def ADJCALLSTACKDOWN : I<(outs), (ins i64imm:$amt),
                         [(WebAssemblycallseq_start timm:$amt)]>;
def ADJCALLSTACKUP : I<(outs), (ins i64imm:$amt),
                       [(WebAssemblycallseq_end timm:$amt, undef)]>;
} // isCodeGenOnly = 1

multiclass CALL<WebAssemblyRegClass vt> {
  def CALL_#vt : I<(outs vt:$dst), (ins global:$callee, variable_ops),
                   [(set vt:$dst, (WebAssemblycall1 (WebAssemblywrapper tglobaladdr:$callee)))],
                   "call    \t$dst, $callee">;
  def CALL_INDIRECT_#vt : I<(outs vt:$dst), (ins I32:$callee, variable_ops),
                            [(set vt:$dst, (WebAssemblycall1 I32:$callee))],
                            "call_indirect\t$dst, $callee">;
}
let Uses = [SP32, SP64], isCall = 1 in {
  defm : CALL<I32>;
  defm : CALL<I64>;
  defm : CALL<F32>;
  defm : CALL<F64>;

  def CALL_VOID : I<(outs), (ins global:$callee, variable_ops),
                    [(WebAssemblycall0 (WebAssemblywrapper tglobaladdr:$callee))],
                    "call    \t$callee">;
  def CALL_INDIRECT_VOID : I<(outs), (ins I32:$callee, variable_ops),
                             [(WebAssemblycall0 I32:$callee)],
                             "call_indirect\t$callee">;
} // Uses = [SP32,SP64], isCall = 1

} // Defs = [ARGUMENTS]
