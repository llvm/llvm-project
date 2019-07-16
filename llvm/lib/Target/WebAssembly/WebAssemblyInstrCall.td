//===- WebAssemblyInstrCall.td-WebAssembly Call codegen support -*- tablegen -*-
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// WebAssembly Call operand code-gen constructs.
///
//===----------------------------------------------------------------------===//

// TODO: addr64: These currently assume the callee address is 32-bit.
// FIXME: add $type to first call_indirect asmstr (and maybe $flags)

// Call sequence markers. These have an immediate which represents the amount of
// stack space to allocate or free, which is used for varargs lowering.
let Uses = [SP32, SP64], Defs = [SP32, SP64], isCodeGenOnly = 1 in {
defm ADJCALLSTACKDOWN : NRI<(outs), (ins i32imm:$amt, i32imm:$amt2),
                            [(WebAssemblycallseq_start timm:$amt, timm:$amt2)]>;
defm ADJCALLSTACKUP : NRI<(outs), (ins i32imm:$amt, i32imm:$amt2),
                          [(WebAssemblycallseq_end timm:$amt, timm:$amt2)]>;
} // Uses = [SP32, SP64], Defs = [SP32, SP64], isCodeGenOnly = 1

multiclass CALL<ValueType vt, WebAssemblyRegClass rt, string prefix,
                list<Predicate> preds = []> {
  defm CALL_#vt :
    I<(outs rt:$dst), (ins function32_op:$callee, variable_ops),
      (outs), (ins function32_op:$callee),
      [(set (vt rt:$dst), (WebAssemblycall1 (i32 imm:$callee)))],
      !strconcat(prefix, "call\t$dst, $callee"),
      !strconcat(prefix, "call\t$callee"),
      0x10>,
    Requires<preds>;

  let isCodeGenOnly = 1 in
  defm PCALL_INDIRECT_#vt :
    I<(outs rt:$dst), (ins I32:$callee, variable_ops),
      (outs), (ins I32:$callee),
      [(set (vt rt:$dst), (WebAssemblycall1 I32:$callee))],
      "PSEUDO CALL INDIRECT\t$callee",
      "PSEUDO CALL INDIRECT\t$callee">,
    Requires<preds>;

  defm CALL_INDIRECT_#vt :
    I<(outs rt:$dst),
      (ins TypeIndex:$type, i32imm:$flags, variable_ops),
      (outs), (ins TypeIndex:$type, i32imm:$flags),
      [],
      !strconcat(prefix, "call_indirect\t$dst"),
      !strconcat(prefix, "call_indirect\t$type"),
      0x11>,
    Requires<preds>;
}

let Uses = [SP32, SP64], isCall = 1 in {
defm "" : CALL<i32, I32, "i32.">;
defm "" : CALL<i64, I64, "i64.">;
defm "" : CALL<f32, F32, "f32.">;
defm "" : CALL<f64, F64, "f64.">;
defm "" : CALL<exnref, EXNREF, "exnref.", [HasExceptionHandling]>;
defm "" : CALL<v16i8, V128, "v128.", [HasSIMD128]>;
defm "" : CALL<v8i16, V128, "v128.", [HasSIMD128]>;
defm "" : CALL<v4i32, V128, "v128.", [HasSIMD128]>;
defm "" : CALL<v2i64, V128, "v128.", [HasSIMD128]>;
defm "" : CALL<v4f32, V128, "v128.", [HasSIMD128]>;
defm "" : CALL<v2f64, V128, "v128.", [HasSIMD128]>;

let IsCanonical = 1 in {
defm CALL_VOID :
  I<(outs), (ins function32_op:$callee, variable_ops),
    (outs), (ins function32_op:$callee),
    [(WebAssemblycall0 (i32 imm:$callee))],
    "call    \t$callee", "call\t$callee", 0x10>;

let isReturn = 1 in
defm RET_CALL :
  I<(outs), (ins function32_op:$callee, variable_ops),
    (outs), (ins function32_op:$callee),
    [(WebAssemblyretcall (i32 imm:$callee))],
    "return_call    \t$callee", "return_call\t$callee", 0x12>,
  Requires<[HasTailCall]>;

let isCodeGenOnly = 1 in
defm PCALL_INDIRECT_VOID :
  I<(outs), (ins I32:$callee, variable_ops),
    (outs), (ins I32:$callee),
    [(WebAssemblycall0 I32:$callee)],
    "PSEUDO CALL INDIRECT\t$callee",
    "PSEUDO CALL INDIRECT\t$callee">;

defm CALL_INDIRECT_VOID :
  I<(outs), (ins TypeIndex:$type, i32imm:$flags, variable_ops),
    (outs), (ins TypeIndex:$type, i32imm:$flags),
    [],
    "call_indirect\t", "call_indirect\t$type",
    0x11>;

let isReturn = 1 in
defm RET_CALL_INDIRECT :
  I<(outs), (ins TypeIndex:$type, i32imm:$flags, variable_ops),
    (outs), (ins TypeIndex:$type, i32imm:$flags),
    [],
    "return_call_indirect\t", "return_call_indirect\t$type",
    0x13>,
  Requires<[HasTailCall]>;

let isCodeGenOnly = 1, isReturn = 1 in
defm PRET_CALL_INDIRECT:
    I<(outs), (ins I32:$callee, variable_ops),
      (outs), (ins I32:$callee),
      [(WebAssemblyretcall I32:$callee)],
      "PSEUDO RET_CALL INDIRECT\t$callee",
      "PSEUDO RET_CALL INDIRECT\t$callee">,
    Requires<[HasTailCall]>;

} // IsCanonical = 1
} // Uses = [SP32,SP64], isCall = 1

// Patterns for matching a direct call to a global address.
def : Pat<(i32 (WebAssemblycall1 (WebAssemblywrapper tglobaladdr:$callee))),
          (CALL_i32 tglobaladdr:$callee)>;
def : Pat<(i64 (WebAssemblycall1 (WebAssemblywrapper tglobaladdr:$callee))),
          (CALL_i64 tglobaladdr:$callee)>;
def : Pat<(f32 (WebAssemblycall1 (WebAssemblywrapper tglobaladdr:$callee))),
          (CALL_f32 tglobaladdr:$callee)>;
def : Pat<(f64 (WebAssemblycall1 (WebAssemblywrapper tglobaladdr:$callee))),
          (CALL_f64 tglobaladdr:$callee)>;
def : Pat<(v16i8 (WebAssemblycall1 (WebAssemblywrapper tglobaladdr:$callee))),
          (CALL_v16i8 tglobaladdr:$callee)>, Requires<[HasSIMD128]>;
def : Pat<(v8i16 (WebAssemblycall1 (WebAssemblywrapper tglobaladdr:$callee))),
          (CALL_v8i16 tglobaladdr:$callee)>, Requires<[HasSIMD128]>;
def : Pat<(v4i32 (WebAssemblycall1 (WebAssemblywrapper tglobaladdr:$callee))),
          (CALL_v4i32 tglobaladdr:$callee)>, Requires<[HasSIMD128]>;
def : Pat<(v2i64 (WebAssemblycall1 (WebAssemblywrapper tglobaladdr:$callee))),
          (CALL_v2i64 tglobaladdr:$callee)>, Requires<[HasSIMD128]>;
def : Pat<(v4f32 (WebAssemblycall1 (WebAssemblywrapper tglobaladdr:$callee))),
          (CALL_v4f32 tglobaladdr:$callee)>, Requires<[HasSIMD128]>;
def : Pat<(v2f64 (WebAssemblycall1 (WebAssemblywrapper tglobaladdr:$callee))),
          (CALL_v2f64 tglobaladdr:$callee)>, Requires<[HasSIMD128]>;
def : Pat<(exnref (WebAssemblycall1 (WebAssemblywrapper tglobaladdr:$callee))),
          (CALL_exnref tglobaladdr:$callee)>,
      Requires<[HasExceptionHandling]>;
def : Pat<(WebAssemblycall0 (WebAssemblywrapper tglobaladdr:$callee)),
          (CALL_VOID tglobaladdr:$callee)>;
def : Pat<(WebAssemblyretcall (WebAssemblywrapper tglobaladdr:$callee)),
          (RET_CALL tglobaladdr:$callee)>, Requires<[HasTailCall]>;

// Patterns for matching a direct call to an external symbol.
def : Pat<(i32 (WebAssemblycall1 (WebAssemblywrapper texternalsym:$callee))),
          (CALL_i32 texternalsym:$callee)>;
def : Pat<(i64 (WebAssemblycall1 (WebAssemblywrapper texternalsym:$callee))),
          (CALL_i64 texternalsym:$callee)>;
def : Pat<(f32 (WebAssemblycall1 (WebAssemblywrapper texternalsym:$callee))),
          (CALL_f32 texternalsym:$callee)>;
def : Pat<(f64 (WebAssemblycall1 (WebAssemblywrapper texternalsym:$callee))),
          (CALL_f64 texternalsym:$callee)>;
def : Pat<(v16i8 (WebAssemblycall1 (WebAssemblywrapper texternalsym:$callee))),
          (CALL_v16i8 texternalsym:$callee)>, Requires<[HasSIMD128]>;
def : Pat<(v8i16 (WebAssemblycall1 (WebAssemblywrapper texternalsym:$callee))),
          (CALL_v8i16 texternalsym:$callee)>, Requires<[HasSIMD128]>;
def : Pat<(v4i32 (WebAssemblycall1 (WebAssemblywrapper texternalsym:$callee))),
          (CALL_v4i32 texternalsym:$callee)>, Requires<[HasSIMD128]>;
def : Pat<(v2i64 (WebAssemblycall1 (WebAssemblywrapper texternalsym:$callee))),
          (CALL_v2i64 texternalsym:$callee)>, Requires<[HasSIMD128]>;
def : Pat<(v4f32 (WebAssemblycall1 (WebAssemblywrapper texternalsym:$callee))),
          (CALL_v4f32 texternalsym:$callee)>, Requires<[HasSIMD128]>;
def : Pat<(v2f64 (WebAssemblycall1 (WebAssemblywrapper texternalsym:$callee))),
          (CALL_v2f64 texternalsym:$callee)>, Requires<[HasSIMD128]>;
def : Pat<(exnref (WebAssemblycall1 (WebAssemblywrapper texternalsym:$callee))),
          (CALL_exnref texternalsym:$callee)>,
      Requires<[HasExceptionHandling]>;
def : Pat<(WebAssemblycall0 (WebAssemblywrapper texternalsym:$callee)),
          (CALL_VOID texternalsym:$callee)>;
def : Pat<(WebAssemblyretcall (WebAssemblywrapper texternalsym:$callee)),
          (RET_CALL texternalsym:$callee)>, Requires<[HasTailCall]>;
