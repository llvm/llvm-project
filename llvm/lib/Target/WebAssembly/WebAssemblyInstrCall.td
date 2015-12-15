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

// TODO: addr64: These currently assume the callee address is 32-bit.

let Defs = [ARGUMENTS] in {

// Call sequence markers. These have an immediate which represents the amount of
// stack space to allocate or free, which is used for varargs lowering.
let Uses = [SP32, SP64], Defs = [SP32, SP64], isCodeGenOnly = 1 in {
def ADJCALLSTACKDOWN : I<(outs), (ins i32imm:$amt),
                         [(WebAssemblycallseq_start timm:$amt)]>;
def ADJCALLSTACKUP : I<(outs), (ins i32imm:$amt),
                       [(WebAssemblycallseq_end timm:$amt, undef)]>;
} // isCodeGenOnly = 1

multiclass CALL<WebAssemblyRegClass vt, string prefix> {
  def CALL_#vt : I<(outs vt:$dst), (ins i32imm:$callee, variable_ops),
                   [(set vt:$dst, (WebAssemblycall1 (i32 imm:$callee)))],
                   !strconcat(prefix, "call\t$dst, $callee")>;
  def CALL_INDIRECT_#vt : I<(outs vt:$dst), (ins I32:$callee, variable_ops),
                            [(set vt:$dst, (WebAssemblycall1 I32:$callee))],
                            !strconcat(prefix, "call_indirect\t$dst, $callee")>;
}
let Uses = [SP32, SP64], isCall = 1 in {
  defm : CALL<I32, "i32.">;
  defm : CALL<I64, "i64.">;
  defm : CALL<F32, "f32.">;
  defm : CALL<F64, "f64.">;

  def CALL_VOID : I<(outs), (ins i32imm:$callee, variable_ops),
                    [(WebAssemblycall0 (i32 imm:$callee))],
                    "call    \t$callee">;
  def CALL_INDIRECT_VOID : I<(outs), (ins I32:$callee, variable_ops),
                             [(WebAssemblycall0 I32:$callee)],
                             "call_indirect\t$callee">;
} // Uses = [SP32,SP64], isCall = 1

} // Defs = [ARGUMENTS]

// Patterns for matching a direct call to a global address.
def : Pat<(i32 (WebAssemblycall1 (WebAssemblywrapper tglobaladdr:$callee))),
          (CALL_I32 tglobaladdr:$callee)>;
def : Pat<(i64 (WebAssemblycall1 (WebAssemblywrapper tglobaladdr:$callee))),
          (CALL_I64 tglobaladdr:$callee)>;
def : Pat<(f32 (WebAssemblycall1 (WebAssemblywrapper tglobaladdr:$callee))),
          (CALL_F32 tglobaladdr:$callee)>;
def : Pat<(f64 (WebAssemblycall1 (WebAssemblywrapper tglobaladdr:$callee))),
          (CALL_F64 tglobaladdr:$callee)>;
def : Pat<(WebAssemblycall0 (WebAssemblywrapper tglobaladdr:$callee)),
          (CALL_VOID tglobaladdr:$callee)>;

// Patterns for matching a direct call to an external symbol.
def : Pat<(i32 (WebAssemblycall1 (WebAssemblywrapper texternalsym:$callee))),
          (CALL_I32 texternalsym:$callee)>;
def : Pat<(i64 (WebAssemblycall1 (WebAssemblywrapper texternalsym:$callee))),
          (CALL_I64 texternalsym:$callee)>;
def : Pat<(f32 (WebAssemblycall1 (WebAssemblywrapper texternalsym:$callee))),
          (CALL_F32 texternalsym:$callee)>;
def : Pat<(f64 (WebAssemblycall1 (WebAssemblywrapper texternalsym:$callee))),
          (CALL_F64 texternalsym:$callee)>;
def : Pat<(WebAssemblycall0 (WebAssemblywrapper texternalsym:$callee)),
          (CALL_VOID texternalsym:$callee)>;
