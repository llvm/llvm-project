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

// The call sequence start/end LLVM-isms isn't useful to WebAssembly since it's
// a virtual ISA.
let isCodeGenOnly = 1 in {
def : I<(outs), (ins i64imm:$amt),
        [(WebAssemblycallseq_start timm:$amt)]>;
def : I<(outs), (ins i64imm:$amt1, i64imm:$amt2),
        [(WebAssemblycallseq_end timm:$amt1, timm:$amt2)]>;
} // isCodeGenOnly = 1

multiclass CALL<WebAssemblyRegClass vt> {
  def CALL_#vt : I<(outs vt:$dst), (ins global:$callee, variable_ops),
                   [(set vt:$dst, (WebAssemblycall1 (WebAssemblywrapper tglobaladdr:$callee)))]>;
  def CALL_INDIRECT_#vt : I<(outs vt:$dst), (ins I32:$callee, variable_ops),
                            [(set vt:$dst, (WebAssemblycall1 I32:$callee))]>;
}
let Uses = [SP32, SP64], isCall = 1 in {
  defm : CALL<I32>;
  defm : CALL<I64>;
  defm : CALL<F32>;
  defm : CALL<F64>;

  def CALL_VOID : I<(outs), (ins global:$callee, variable_ops),
                    [(WebAssemblycall0 (WebAssemblywrapper tglobaladdr:$callee))]>;
  def CALL_INDIRECT_VOID : I<(outs), (ins I32:$callee, variable_ops),
                             [(WebAssemblycall0 I32:$callee)]>;
} // Uses = [SP32,SP64], isCall = 1

/*
 * TODO(jfb): Add the following.
 *
 * addressof: obtain a function pointer value for a given function
 */
