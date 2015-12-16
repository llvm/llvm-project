// WebAssemblyInstrMemory.td-WebAssembly Memory codegen support -*- tablegen -*-
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief WebAssembly Memory operand code-gen constructs.
///
//===----------------------------------------------------------------------===//

// TODO:
//  - HasAddr64
//  - WebAssemblyTargetLowering having to do with atomics
//  - Each has optional alignment.

// WebAssembly has i8/i16/i32/i64/f32/f64 memory types, but doesn't have i8/i16
// local types. These memory-only types instead zero- or sign-extend into local
// types when loading, and truncate when storing.

// WebAssembly constant offsets are performed as unsigned with infinite
// precision, so we need to check for NoUnsignedWrap so that we don't fold an
// offset for an add that needs wrapping.
def regPlusImm : PatFrag<(ops node:$off, node:$addr),
                         (add node:$addr, node:$off),
                         [{ return N->getFlags()->hasNoUnsignedWrap(); }]>;

let Defs = [ARGUMENTS] in {

// Basic load.
def LOAD_I32 : I<(outs I32:$dst), (ins i32imm:$off, I32:$addr), [],
                 "i32.load\t$dst, $off($addr)">;
def LOAD_I64 : I<(outs I64:$dst), (ins i32imm:$off, I32:$addr), [],
                 "i64.load\t$dst, $off($addr)">;
def LOAD_F32 : I<(outs F32:$dst), (ins i32imm:$off, I32:$addr), [],
                 "f32.load\t$dst, $off($addr)">;
def LOAD_F64 : I<(outs F64:$dst), (ins i32imm:$off, I32:$addr), [],
                 "f64.load\t$dst, $off($addr)">;

} // Defs = [ARGUMENTS]

// Select loads with no constant offset.
def : Pat<(i32 (load I32:$addr)), (LOAD_I32 0, $addr)>;
def : Pat<(i64 (load I32:$addr)), (LOAD_I64 0, $addr)>;
def : Pat<(f32 (load I32:$addr)), (LOAD_F32 0, $addr)>;
def : Pat<(f64 (load I32:$addr)), (LOAD_F64 0, $addr)>;

// Select loads with a constant offset.
def : Pat<(i32 (load (regPlusImm imm:$off, I32:$addr))),
          (LOAD_I32 imm:$off, $addr)>;
def : Pat<(i64 (load (regPlusImm imm:$off, I32:$addr))),
          (LOAD_I64 imm:$off, $addr)>;
def : Pat<(f32 (load (regPlusImm imm:$off, I32:$addr))),
          (LOAD_F32 imm:$off, $addr)>;
def : Pat<(f64 (load (regPlusImm imm:$off, I32:$addr))),
          (LOAD_F64 imm:$off, $addr)>;
def : Pat<(i32 (load (regPlusImm tglobaladdr:$off, I32:$addr))),
          (LOAD_I32 tglobaladdr:$off, $addr)>;
def : Pat<(i64 (load (regPlusImm tglobaladdr:$off, I32:$addr))),
          (LOAD_I64 tglobaladdr:$off, $addr)>;
def : Pat<(f32 (load (regPlusImm tglobaladdr:$off, I32:$addr))),
          (LOAD_F32 tglobaladdr:$off, $addr)>;
def : Pat<(f64 (load (regPlusImm tglobaladdr:$off, I32:$addr))),
          (LOAD_F64 tglobaladdr:$off, $addr)>;
def : Pat<(i32 (load (regPlusImm texternalsym:$off, I32:$addr))),
          (LOAD_I32 texternalsym:$off, $addr)>;
def : Pat<(i64 (load (regPlusImm texternalsym:$off, I32:$addr))),
          (LOAD_I64 texternalsym:$off, $addr)>;
def : Pat<(f32 (load (regPlusImm texternalsym:$off, I32:$addr))),
          (LOAD_F32 texternalsym:$off, $addr)>;
def : Pat<(f64 (load (regPlusImm texternalsym:$off, I32:$addr))),
          (LOAD_F64 texternalsym:$off, $addr)>;

// Select loads with just a constant offset.
def : Pat<(i32 (load imm:$off)), (LOAD_I32 imm:$off, (CONST_I32 0))>;
def : Pat<(i64 (load imm:$off)), (LOAD_I64 imm:$off, (CONST_I32 0))>;
def : Pat<(f32 (load imm:$off)), (LOAD_F32 imm:$off, (CONST_I32 0))>;
def : Pat<(f64 (load imm:$off)), (LOAD_F64 imm:$off, (CONST_I32 0))>;
def : Pat<(i32 (load (WebAssemblywrapper tglobaladdr:$off))),
          (LOAD_I32 tglobaladdr:$off, (CONST_I32 0))>;
def : Pat<(i64 (load (WebAssemblywrapper tglobaladdr:$off))),
          (LOAD_I64 tglobaladdr:$off, (CONST_I32 0))>;
def : Pat<(f32 (load (WebAssemblywrapper tglobaladdr:$off))),
          (LOAD_F32 tglobaladdr:$off, (CONST_I32 0))>;
def : Pat<(f64 (load (WebAssemblywrapper tglobaladdr:$off))),
          (LOAD_F64 tglobaladdr:$off, (CONST_I32 0))>;
def : Pat<(i32 (load (WebAssemblywrapper texternalsym:$off))),
          (LOAD_I32 texternalsym:$off, (CONST_I32 0))>;
def : Pat<(i64 (load (WebAssemblywrapper texternalsym:$off))),
          (LOAD_I64 texternalsym:$off, (CONST_I32 0))>;
def : Pat<(f32 (load (WebAssemblywrapper texternalsym:$off))),
          (LOAD_F32 texternalsym:$off, (CONST_I32 0))>;
def : Pat<(f64 (load (WebAssemblywrapper texternalsym:$off))),
          (LOAD_F64 texternalsym:$off, (CONST_I32 0))>;

let Defs = [ARGUMENTS] in {

// Extending load.
def LOAD8_S_I32  : I<(outs I32:$dst), (ins i32imm:$off, I32:$addr), [],
                     "i32.load8_s\t$dst, $off($addr)">;
def LOAD8_U_I32  : I<(outs I32:$dst), (ins i32imm:$off, I32:$addr), [],
                     "i32.load8_u\t$dst, $off($addr)">;
def LOAD16_S_I32 : I<(outs I32:$dst), (ins i32imm:$off, I32:$addr), [],
                     "i32.load16_s\t$dst, $off($addr)">;
def LOAD16_U_I32 : I<(outs I32:$dst), (ins i32imm:$off, I32:$addr), [],
                     "i32.load16_u\t$dst, $off($addr)">;
def LOAD8_S_I64  : I<(outs I64:$dst), (ins i32imm:$off, I32:$addr), [],
                     "i64.load8_s\t$dst, $off($addr)">;
def LOAD8_U_I64  : I<(outs I64:$dst), (ins i32imm:$off, I32:$addr), [],
                     "i64.load8_u\t$dst, $off($addr)">;
def LOAD16_S_I64 : I<(outs I64:$dst), (ins i32imm:$off, I32:$addr), [],
                     "i64.load16_s\t$dst, $off($addr)">;
def LOAD16_U_I64 : I<(outs I64:$dst), (ins i32imm:$off, I32:$addr), [],
                     "i64.load16_u\t$dst, $off($addr)">;
def LOAD32_S_I64 : I<(outs I64:$dst), (ins i32imm:$off, I32:$addr), [],
                     "i64.load32_s\t$dst, $off($addr)">;
def LOAD32_U_I64 : I<(outs I64:$dst), (ins i32imm:$off, I32:$addr), [],
                     "i64.load32_u\t$dst, $off($addr)">;

} // Defs = [ARGUMENTS]

// Select extending loads with no constant offset.
def : Pat<(i32 (sextloadi8 I32:$addr)), (LOAD8_S_I32 0, $addr)>;
def : Pat<(i32 (zextloadi8 I32:$addr)), (LOAD8_U_I32 0, $addr)>;
def : Pat<(i32 (sextloadi16 I32:$addr)), (LOAD16_S_I32 0, $addr)>;
def : Pat<(i32 (zextloadi16 I32:$addr)), (LOAD16_U_I32 0, $addr)>;
def : Pat<(i64 (sextloadi8 I32:$addr)), (LOAD8_S_I64 0, $addr)>;
def : Pat<(i64 (zextloadi8 I32:$addr)), (LOAD8_U_I64 0, $addr)>;
def : Pat<(i64 (sextloadi16 I32:$addr)), (LOAD16_S_I64 0, $addr)>;
def : Pat<(i64 (zextloadi16 I32:$addr)), (LOAD16_U_I64 0, $addr)>;
def : Pat<(i64 (sextloadi32 I32:$addr)), (LOAD32_S_I64 0, $addr)>;
def : Pat<(i64 (zextloadi32 I32:$addr)), (LOAD32_U_I64 0, $addr)>;

// Select extending loads with a constant offset.
def : Pat<(i32 (sextloadi8 (regPlusImm imm:$off, I32:$addr))),
          (LOAD8_S_I32 imm:$off, $addr)>;
def : Pat<(i32 (zextloadi8 (regPlusImm imm:$off, I32:$addr))),
          (LOAD8_U_I32 imm:$off, $addr)>;
def : Pat<(i32 (sextloadi16 (regPlusImm imm:$off, I32:$addr))),
          (LOAD16_S_I32 imm:$off, $addr)>;
def : Pat<(i32 (zextloadi16 (regPlusImm imm:$off, I32:$addr))),
          (LOAD16_U_I32 imm:$off, $addr)>;
def : Pat<(i64 (sextloadi8 (regPlusImm imm:$off, I32:$addr))),
          (LOAD8_S_I64 imm:$off, $addr)>;
def : Pat<(i64 (zextloadi8 (regPlusImm imm:$off, I32:$addr))),
          (LOAD8_U_I64 imm:$off, $addr)>;
def : Pat<(i64 (sextloadi16 (regPlusImm imm:$off, I32:$addr))),
          (LOAD16_S_I64 imm:$off, $addr)>;
def : Pat<(i64 (zextloadi16 (regPlusImm imm:$off, I32:$addr))),
          (LOAD16_U_I64 imm:$off, $addr)>;
def : Pat<(i64 (sextloadi32 (regPlusImm imm:$off, I32:$addr))),
          (LOAD32_S_I64 imm:$off, $addr)>;
def : Pat<(i64 (zextloadi32 (regPlusImm imm:$off, I32:$addr))),
          (LOAD32_U_I64 imm:$off, $addr)>;
def : Pat<(i32 (sextloadi8 (regPlusImm tglobaladdr:$off, I32:$addr))),
          (LOAD8_S_I32 tglobaladdr:$off, $addr)>;
def : Pat<(i32 (zextloadi8 (regPlusImm tglobaladdr:$off, I32:$addr))),
          (LOAD8_U_I32 tglobaladdr:$off, $addr)>;
def : Pat<(i32 (sextloadi16 (regPlusImm tglobaladdr:$off, I32:$addr))),
          (LOAD16_S_I32 tglobaladdr:$off, $addr)>;
def : Pat<(i32 (zextloadi16 (regPlusImm tglobaladdr:$off, I32:$addr))),
          (LOAD16_U_I32 tglobaladdr:$off, $addr)>;
def : Pat<(i64 (sextloadi8 (regPlusImm tglobaladdr:$off, I32:$addr))),
          (LOAD8_S_I64 tglobaladdr:$off, $addr)>;
def : Pat<(i64 (zextloadi8 (regPlusImm tglobaladdr:$off, I32:$addr))),
          (LOAD8_U_I64 tglobaladdr:$off, $addr)>;
def : Pat<(i64 (sextloadi16 (regPlusImm tglobaladdr:$off, I32:$addr))),
          (LOAD16_S_I64 tglobaladdr:$off, $addr)>;
def : Pat<(i64 (zextloadi16 (regPlusImm tglobaladdr:$off, I32:$addr))),
          (LOAD16_U_I64 tglobaladdr:$off, $addr)>;
def : Pat<(i64 (sextloadi32 (regPlusImm tglobaladdr:$off, I32:$addr))),
          (LOAD32_S_I64 tglobaladdr:$off, $addr)>;
def : Pat<(i64 (zextloadi32 (regPlusImm tglobaladdr:$off, I32:$addr))),
          (LOAD32_U_I64 tglobaladdr:$off, $addr)>;
def : Pat<(i32 (sextloadi8 (regPlusImm texternalsym:$off, I32:$addr))),
          (LOAD8_S_I32 texternalsym:$off, $addr)>;
def : Pat<(i32 (zextloadi8 (regPlusImm texternalsym:$off, I32:$addr))),
          (LOAD8_U_I32 texternalsym:$off, $addr)>;
def : Pat<(i32 (sextloadi16 (regPlusImm texternalsym:$off, I32:$addr))),
          (LOAD16_S_I32 texternalsym:$off, $addr)>;
def : Pat<(i32 (zextloadi16 (regPlusImm texternalsym:$off, I32:$addr))),
          (LOAD16_U_I32 texternalsym:$off, $addr)>;
def : Pat<(i64 (sextloadi8 (regPlusImm texternalsym:$off, I32:$addr))),
          (LOAD8_S_I64 texternalsym:$off, $addr)>;
def : Pat<(i64 (zextloadi8 (regPlusImm texternalsym:$off, I32:$addr))),
          (LOAD8_U_I64 texternalsym:$off, $addr)>;
def : Pat<(i64 (sextloadi16 (regPlusImm texternalsym:$off, I32:$addr))),
          (LOAD16_S_I64 texternalsym:$off, $addr)>;
def : Pat<(i64 (zextloadi16 (regPlusImm texternalsym:$off, I32:$addr))),
          (LOAD16_U_I64 texternalsym:$off, $addr)>;
def : Pat<(i64 (sextloadi32 (regPlusImm texternalsym:$off, I32:$addr))),
          (LOAD32_S_I64 texternalsym:$off, $addr)>;
def : Pat<(i64 (zextloadi32 (regPlusImm texternalsym:$off, I32:$addr))),
          (LOAD32_U_I64 texternalsym:$off, $addr)>;

// Select extending loads with just a constant offset.
def : Pat<(i32 (sextloadi8 imm:$off)), (LOAD8_S_I32 imm:$off, (CONST_I32 0))>;
def : Pat<(i32 (zextloadi8 imm:$off)), (LOAD8_U_I32 imm:$off, (CONST_I32 0))>;
def : Pat<(i32 (sextloadi16 imm:$off)), (LOAD16_S_I32 imm:$off, (CONST_I32 0))>;
def : Pat<(i32 (zextloadi16 imm:$off)), (LOAD16_U_I32 imm:$off, (CONST_I32 0))>;
def : Pat<(i64 (sextloadi8 imm:$off)), (LOAD8_S_I64 imm:$off, (CONST_I32 0))>;
def : Pat<(i64 (zextloadi8 imm:$off)), (LOAD8_U_I64 imm:$off, (CONST_I32 0))>;
def : Pat<(i64 (sextloadi16 imm:$off)), (LOAD16_S_I64 imm:$off, (CONST_I32 0))>;
def : Pat<(i64 (zextloadi16 imm:$off)), (LOAD16_U_I64 imm:$off, (CONST_I32 0))>;
def : Pat<(i64 (sextloadi32 imm:$off)), (LOAD32_S_I64 imm:$off, (CONST_I32 0))>;
def : Pat<(i64 (zextloadi32 imm:$off)), (LOAD32_U_I64 imm:$off, (CONST_I32 0))>;
def : Pat<(i32 (sextloadi8 (WebAssemblywrapper tglobaladdr:$off))),
          (LOAD8_S_I32 tglobaladdr:$off, (CONST_I32 0))>;
def : Pat<(i32 (zextloadi8 (WebAssemblywrapper tglobaladdr:$off))),
          (LOAD8_U_I32 tglobaladdr:$off, (CONST_I32 0))>;
def : Pat<(i32 (sextloadi16 (WebAssemblywrapper tglobaladdr:$off))),
          (LOAD16_S_I32 tglobaladdr:$off, (CONST_I32 0))>;
def : Pat<(i32 (zextloadi16 (WebAssemblywrapper tglobaladdr:$off))),
          (LOAD16_U_I32 tglobaladdr:$off, (CONST_I32 0))>;
def : Pat<(i64 (sextloadi8 (WebAssemblywrapper tglobaladdr:$off))),
          (LOAD8_S_I64 tglobaladdr:$off, (CONST_I32 0))>;
def : Pat<(i64 (zextloadi8 (WebAssemblywrapper tglobaladdr:$off))),
          (LOAD8_U_I64 tglobaladdr:$off, (CONST_I32 0))>;
def : Pat<(i64 (sextloadi16 (WebAssemblywrapper tglobaladdr:$off))),
          (LOAD16_S_I64 tglobaladdr:$off, (CONST_I32 0))>;
def : Pat<(i64 (zextloadi16 (WebAssemblywrapper tglobaladdr:$off))),
          (LOAD16_U_I64 tglobaladdr:$off, (CONST_I32 0))>;
def : Pat<(i64 (sextloadi32 (WebAssemblywrapper tglobaladdr:$off))),
          (LOAD32_S_I64 tglobaladdr:$off, (CONST_I32 0))>;
def : Pat<(i64 (zextloadi32 (WebAssemblywrapper tglobaladdr:$off))),
          (LOAD32_U_I64 tglobaladdr:$off, (CONST_I32 0))>;
def : Pat<(i32 (sextloadi8 (WebAssemblywrapper texternalsym:$off))),
          (LOAD8_S_I32 texternalsym:$off, (CONST_I32 0))>;
def : Pat<(i32 (zextloadi8 (WebAssemblywrapper texternalsym:$off))),
          (LOAD8_U_I32 texternalsym:$off, (CONST_I32 0))>;
def : Pat<(i32 (sextloadi16 (WebAssemblywrapper texternalsym:$off))),
          (LOAD16_S_I32 texternalsym:$off, (CONST_I32 0))>;
def : Pat<(i32 (zextloadi16 (WebAssemblywrapper texternalsym:$off))),
          (LOAD16_U_I32 texternalsym:$off, (CONST_I32 0))>;
def : Pat<(i64 (sextloadi8 (WebAssemblywrapper texternalsym:$off))),
          (LOAD8_S_I64 texternalsym:$off, (CONST_I32 0))>;
def : Pat<(i64 (zextloadi8 (WebAssemblywrapper texternalsym:$off))),
          (LOAD8_U_I64 texternalsym:$off, (CONST_I32 0))>;
def : Pat<(i64 (sextloadi16 (WebAssemblywrapper texternalsym:$off))),
          (LOAD16_S_I64 texternalsym:$off, (CONST_I32 0))>;
def : Pat<(i64 (zextloadi16 (WebAssemblywrapper texternalsym:$off))),
          (LOAD16_U_I64 texternalsym:$off, (CONST_I32 0))>;
def : Pat<(i64 (sextloadi32 (WebAssemblywrapper texternalsym:$off))),
          (LOAD32_S_I64 texternalsym:$off, (CONST_I32 0))>;
def : Pat<(i64 (zextloadi32 (WebAssemblywrapper texternalsym:$off))),
          (LOAD32_U_I64 texternalsym:$off, (CONST_I32 0))>;

// Resolve "don't care" extending loads to zero-extending loads. This is
// somewhat arbitrary, but zero-extending is conceptually simpler.

// Select "don't care" extending loads with no constant offset.
def : Pat<(i32 (extloadi8 I32:$addr)),  (LOAD8_U_I32 0, $addr)>;
def : Pat<(i32 (extloadi16 I32:$addr)), (LOAD16_U_I32 0, $addr)>;
def : Pat<(i64 (extloadi8 I32:$addr)),  (LOAD8_U_I64 0, $addr)>;
def : Pat<(i64 (extloadi16 I32:$addr)), (LOAD16_U_I64 0, $addr)>;
def : Pat<(i64 (extloadi32 I32:$addr)), (LOAD32_U_I64 0, $addr)>;

// Select "don't care" extending loads with a constant offset.
def : Pat<(i32 (extloadi8 (regPlusImm imm:$off, I32:$addr))),
          (LOAD8_U_I32 imm:$off, $addr)>;
def : Pat<(i32 (extloadi16 (regPlusImm imm:$off, I32:$addr))),
          (LOAD16_U_I32 imm:$off, $addr)>;
def : Pat<(i64 (extloadi8 (regPlusImm imm:$off, I32:$addr))),
          (LOAD8_U_I64 imm:$off, $addr)>;
def : Pat<(i64 (extloadi16 (regPlusImm imm:$off, I32:$addr))),
          (LOAD16_U_I64 imm:$off, $addr)>;
def : Pat<(i64 (extloadi32 (regPlusImm imm:$off, I32:$addr))),
          (LOAD32_U_I64 imm:$off, $addr)>;
def : Pat<(i32 (extloadi8 (regPlusImm tglobaladdr:$off, I32:$addr))),
          (LOAD8_U_I32 tglobaladdr:$off, $addr)>;
def : Pat<(i32 (extloadi16 (regPlusImm tglobaladdr:$off, I32:$addr))),
          (LOAD16_U_I32 tglobaladdr:$off, $addr)>;
def : Pat<(i64 (extloadi8 (regPlusImm tglobaladdr:$off, I32:$addr))),
          (LOAD8_U_I64 tglobaladdr:$off, $addr)>;
def : Pat<(i64 (extloadi16 (regPlusImm tglobaladdr:$off, I32:$addr))),
          (LOAD16_U_I64 tglobaladdr:$off, $addr)>;
def : Pat<(i64 (extloadi32 (regPlusImm tglobaladdr:$off, I32:$addr))),
          (LOAD32_U_I64 tglobaladdr:$off, $addr)>;
def : Pat<(i32 (extloadi8 (regPlusImm texternalsym:$off, I32:$addr))),
          (LOAD8_U_I32 texternalsym:$off, $addr)>;
def : Pat<(i32 (extloadi16 (regPlusImm texternalsym:$off, I32:$addr))),
          (LOAD16_U_I32 texternalsym:$off, $addr)>;
def : Pat<(i64 (extloadi8 (regPlusImm texternalsym:$off, I32:$addr))),
          (LOAD8_U_I64 texternalsym:$off, $addr)>;
def : Pat<(i64 (extloadi16 (regPlusImm texternalsym:$off, I32:$addr))),
          (LOAD16_U_I64 texternalsym:$off, $addr)>;
def : Pat<(i64 (extloadi32 (regPlusImm texternalsym:$off, I32:$addr))),
          (LOAD32_U_I64 texternalsym:$off, $addr)>;

// Select "don't care" extending loads with just a constant offset.
def : Pat<(i32 (extloadi8 imm:$off)), (LOAD8_U_I32 imm:$off, (CONST_I32 0))>;
def : Pat<(i32 (extloadi16 imm:$off)), (LOAD16_U_I32 imm:$off, (CONST_I32 0))>;
def : Pat<(i64 (extloadi8 imm:$off)), (LOAD8_U_I64 imm:$off, (CONST_I32 0))>;
def : Pat<(i64 (extloadi16 imm:$off)), (LOAD16_U_I64 imm:$off, (CONST_I32 0))>;
def : Pat<(i64 (extloadi32 imm:$off)), (LOAD32_U_I64 imm:$off, (CONST_I32 0))>;
def : Pat<(i32 (extloadi8 (WebAssemblywrapper tglobaladdr:$off))),
          (LOAD8_U_I32 tglobaladdr:$off, (CONST_I32 0))>;
def : Pat<(i32 (extloadi16 (WebAssemblywrapper tglobaladdr:$off))),
          (LOAD16_U_I32 tglobaladdr:$off, (CONST_I32 0))>;
def : Pat<(i64 (extloadi8 (WebAssemblywrapper tglobaladdr:$off))),
          (LOAD8_U_I64 tglobaladdr:$off, (CONST_I32 0))>;
def : Pat<(i64 (extloadi16 (WebAssemblywrapper tglobaladdr:$off))),
          (LOAD16_U_I64 tglobaladdr:$off, (CONST_I32 0))>;
def : Pat<(i64 (extloadi32 (WebAssemblywrapper tglobaladdr:$off))),
          (LOAD32_U_I64 tglobaladdr:$off, (CONST_I32 0))>;
def : Pat<(i32 (extloadi8 (WebAssemblywrapper texternalsym:$off))),
          (LOAD8_U_I32 texternalsym:$off, (CONST_I32 0))>;
def : Pat<(i32 (extloadi16 (WebAssemblywrapper texternalsym:$off))),
          (LOAD16_U_I32 texternalsym:$off, (CONST_I32 0))>;
def : Pat<(i64 (extloadi8 (WebAssemblywrapper texternalsym:$off))),
          (LOAD8_U_I64 texternalsym:$off, (CONST_I32 0))>;
def : Pat<(i64 (extloadi16 (WebAssemblywrapper texternalsym:$off))),
          (LOAD16_U_I64 texternalsym:$off, (CONST_I32 0))>;
def : Pat<(i64 (extloadi32 (WebAssemblywrapper texternalsym:$off))),
          (LOAD32_U_I64 tglobaladdr:$off, (CONST_I32 0))>;

let Defs = [ARGUMENTS] in {

// Basic store.
// Note that we split the patterns out of the instruction definitions because
// WebAssembly's stores return their operand value, and tablegen doesn't like
// instruction definition patterns that don't reference all of the output
// operands.
// Note: WebAssembly inverts SelectionDAG's usual operand order.
def STORE_I32  : I<(outs I32:$dst), (ins i32imm:$off, I32:$addr, I32:$val), [],
                   "i32.store\t$dst, $off($addr), $val">;
def STORE_I64  : I<(outs I64:$dst), (ins i32imm:$off, I32:$addr, I64:$val), [],
                   "i64.store\t$dst, $off($addr), $val">;
def STORE_F32  : I<(outs F32:$dst), (ins i32imm:$off, I32:$addr, F32:$val), [],
                   "f32.store\t$dst, $off($addr), $val">;
def STORE_F64  : I<(outs F64:$dst), (ins i32imm:$off, I32:$addr, F64:$val), [],
                   "f64.store\t$dst, $off($addr), $val">;

} // Defs = [ARGUMENTS]

// Select stores with no constant offset.
def : Pat<(store I32:$val, I32:$addr), (STORE_I32 0, I32:$addr, I32:$val)>;
def : Pat<(store I64:$val, I32:$addr), (STORE_I64 0, I32:$addr, I64:$val)>;
def : Pat<(store F32:$val, I32:$addr), (STORE_F32 0, I32:$addr, F32:$val)>;
def : Pat<(store F64:$val, I32:$addr), (STORE_F64 0, I32:$addr, F64:$val)>;

// Select stores with a constant offset.
def : Pat<(store I32:$val, (regPlusImm imm:$off, I32:$addr)),
          (STORE_I32 imm:$off, I32:$addr, I32:$val)>;
def : Pat<(store I64:$val, (regPlusImm imm:$off, I32:$addr)),
          (STORE_I64 imm:$off, I32:$addr, I64:$val)>;
def : Pat<(store F32:$val, (regPlusImm imm:$off, I32:$addr)),
          (STORE_F32 imm:$off, I32:$addr, F32:$val)>;
def : Pat<(store F64:$val, (regPlusImm imm:$off, I32:$addr)),
          (STORE_F64 imm:$off, I32:$addr, F64:$val)>;
def : Pat<(store I32:$val, (regPlusImm tglobaladdr:$off, I32:$addr)),
          (STORE_I32 tglobaladdr:$off, I32:$addr, I32:$val)>;
def : Pat<(store I64:$val, (regPlusImm tglobaladdr:$off, I32:$addr)),
          (STORE_I64 tglobaladdr:$off, I32:$addr, I64:$val)>;
def : Pat<(store F32:$val, (regPlusImm tglobaladdr:$off, I32:$addr)),
          (STORE_F32 tglobaladdr:$off, I32:$addr, F32:$val)>;
def : Pat<(store F64:$val, (regPlusImm tglobaladdr:$off, I32:$addr)),
          (STORE_F64 tglobaladdr:$off, I32:$addr, F64:$val)>;
def : Pat<(store I32:$val, (regPlusImm texternalsym:$off, I32:$addr)),
          (STORE_I32 texternalsym:$off, I32:$addr, I32:$val)>;
def : Pat<(store I64:$val, (regPlusImm texternalsym:$off, I32:$addr)),
          (STORE_I64 texternalsym:$off, I32:$addr, I64:$val)>;
def : Pat<(store F32:$val, (regPlusImm texternalsym:$off, I32:$addr)),
          (STORE_F32 texternalsym:$off, I32:$addr, F32:$val)>;
def : Pat<(store F64:$val, (regPlusImm texternalsym:$off, I32:$addr)),
          (STORE_F64 texternalsym:$off, I32:$addr, F64:$val)>;

// Select stores with just a constant offset.
def : Pat<(store I32:$val, imm:$off),
          (STORE_I32 imm:$off, (CONST_I32 0), I32:$val)>;
def : Pat<(store I64:$val, imm:$off),
          (STORE_I64 imm:$off, (CONST_I32 0), I64:$val)>;
def : Pat<(store F32:$val, imm:$off),
          (STORE_F32 imm:$off, (CONST_I32 0), F32:$val)>;
def : Pat<(store F64:$val, imm:$off),
          (STORE_F64 imm:$off, (CONST_I32 0), F64:$val)>;
def : Pat<(store I32:$val, (WebAssemblywrapper tglobaladdr:$off)),
          (STORE_I32 tglobaladdr:$off, (CONST_I32 0), I32:$val)>;
def : Pat<(store I64:$val, (WebAssemblywrapper tglobaladdr:$off)),
          (STORE_I64 tglobaladdr:$off, (CONST_I32 0), I64:$val)>;
def : Pat<(store F32:$val, (WebAssemblywrapper tglobaladdr:$off)),
          (STORE_F32 tglobaladdr:$off, (CONST_I32 0), F32:$val)>;
def : Pat<(store F64:$val, (WebAssemblywrapper tglobaladdr:$off)),
          (STORE_F64 tglobaladdr:$off, (CONST_I32 0), F64:$val)>;
def : Pat<(store I32:$val, (WebAssemblywrapper texternalsym:$off)),
          (STORE_I32 texternalsym:$off, (CONST_I32 0), I32:$val)>;
def : Pat<(store I64:$val, (WebAssemblywrapper texternalsym:$off)),
          (STORE_I64 texternalsym:$off, (CONST_I32 0), I64:$val)>;
def : Pat<(store F32:$val, (WebAssemblywrapper texternalsym:$off)),
          (STORE_F32 texternalsym:$off, (CONST_I32 0), F32:$val)>;
def : Pat<(store F64:$val, (WebAssemblywrapper texternalsym:$off)),
          (STORE_F64 texternalsym:$off, (CONST_I32 0), F64:$val)>;

let Defs = [ARGUMENTS] in {

// Truncating store.
def STORE8_I32  : I<(outs I32:$dst), (ins i32imm:$off, I32:$addr, I32:$val), [],
                    "i32.store8\t$dst, $off($addr), $val">;
def STORE16_I32 : I<(outs I32:$dst), (ins i32imm:$off, I32:$addr, I32:$val), [],
                    "i32.store16\t$dst, $off($addr), $val">;
def STORE8_I64  : I<(outs I64:$dst), (ins i32imm:$off, I32:$addr, I64:$val), [],
                    "i64.store8\t$dst, $off($addr), $val">;
def STORE16_I64 : I<(outs I64:$dst), (ins i32imm:$off, I32:$addr, I64:$val), [],
                    "i64.store16\t$dst, $off($addr), $val">;
def STORE32_I64 : I<(outs I64:$dst), (ins i32imm:$off, I32:$addr, I64:$val), [],
                    "i64.store32\t$dst, $off($addr), $val">;

} // Defs = [ARGUMENTS]

// Select truncating stores with no constant offset.
def : Pat<(truncstorei8 I32:$val, I32:$addr),
          (STORE8_I32 0, I32:$addr, I32:$val)>;
def : Pat<(truncstorei16 I32:$val, I32:$addr),
          (STORE16_I32 0, I32:$addr, I32:$val)>;
def : Pat<(truncstorei8 I64:$val, I32:$addr),
          (STORE8_I64 0, I32:$addr, I64:$val)>;
def : Pat<(truncstorei16 I64:$val, I32:$addr),
          (STORE16_I64 0, I32:$addr, I64:$val)>;
def : Pat<(truncstorei32 I64:$val, I32:$addr),
          (STORE32_I64 0, I32:$addr, I64:$val)>;

// Select truncating stores with a constant offset.
def : Pat<(truncstorei8 I32:$val, (regPlusImm imm:$off, I32:$addr)),
          (STORE8_I32 imm:$off, I32:$addr, I32:$val)>;
def : Pat<(truncstorei16 I32:$val, (regPlusImm imm:$off, I32:$addr)),
          (STORE16_I32 imm:$off, I32:$addr, I32:$val)>;
def : Pat<(truncstorei8 I64:$val, (regPlusImm imm:$off, I32:$addr)),
          (STORE8_I64 imm:$off, I32:$addr, I64:$val)>;
def : Pat<(truncstorei16 I64:$val, (regPlusImm imm:$off, I32:$addr)),
          (STORE16_I64 imm:$off, I32:$addr, I64:$val)>;
def : Pat<(truncstorei32 I64:$val, (regPlusImm imm:$off, I32:$addr)),
          (STORE32_I64 imm:$off, I32:$addr, I64:$val)>;
def : Pat<(truncstorei8 I32:$val, (regPlusImm tglobaladdr:$off, I32:$addr)),
          (STORE8_I32 tglobaladdr:$off, I32:$addr, I32:$val)>;
def : Pat<(truncstorei16 I32:$val, (regPlusImm tglobaladdr:$off, I32:$addr)),
          (STORE16_I32 tglobaladdr:$off, I32:$addr, I32:$val)>;
def : Pat<(truncstorei8 I64:$val, (regPlusImm tglobaladdr:$off, I32:$addr)),
          (STORE8_I64 tglobaladdr:$off, I32:$addr, I64:$val)>;
def : Pat<(truncstorei16 I64:$val, (regPlusImm tglobaladdr:$off, I32:$addr)),
          (STORE16_I64 tglobaladdr:$off, I32:$addr, I64:$val)>;
def : Pat<(truncstorei32 I64:$val, (regPlusImm tglobaladdr:$off, I32:$addr)),
          (STORE32_I64 tglobaladdr:$off, I32:$addr, I64:$val)>;
def : Pat<(truncstorei8 I32:$val, (regPlusImm texternalsym:$off, I32:$addr)),
          (STORE8_I32 texternalsym:$off, I32:$addr, I32:$val)>;
def : Pat<(truncstorei16 I32:$val, (regPlusImm texternalsym:$off, I32:$addr)),
          (STORE16_I32 texternalsym:$off, I32:$addr, I32:$val)>;
def : Pat<(truncstorei8 I64:$val, (regPlusImm texternalsym:$off, I32:$addr)),
          (STORE8_I64 texternalsym:$off, I32:$addr, I64:$val)>;
def : Pat<(truncstorei16 I64:$val, (regPlusImm texternalsym:$off, I32:$addr)),
          (STORE16_I64 texternalsym:$off, I32:$addr, I64:$val)>;
def : Pat<(truncstorei32 I64:$val, (regPlusImm texternalsym:$off, I32:$addr)),
          (STORE32_I64 texternalsym:$off, I32:$addr, I64:$val)>;

// Select truncating stores with just a constant offset.
def : Pat<(truncstorei8 I32:$val, imm:$off),
          (STORE8_I32 imm:$off, (CONST_I32 0), I32:$val)>;
def : Pat<(truncstorei16 I32:$val, imm:$off),
          (STORE16_I32 imm:$off, (CONST_I32 0), I32:$val)>;
def : Pat<(truncstorei8 I64:$val, imm:$off),
          (STORE8_I64 imm:$off, (CONST_I32 0), I64:$val)>;
def : Pat<(truncstorei16 I64:$val, imm:$off),
          (STORE16_I64 imm:$off, (CONST_I32 0), I64:$val)>;
def : Pat<(truncstorei32 I64:$val, imm:$off),
          (STORE32_I64 imm:$off, (CONST_I32 0), I64:$val)>;
def : Pat<(truncstorei8 I32:$val, (WebAssemblywrapper tglobaladdr:$off)),
          (STORE8_I32 tglobaladdr:$off, (CONST_I32 0), I32:$val)>;
def : Pat<(truncstorei16 I32:$val, (WebAssemblywrapper tglobaladdr:$off)),
          (STORE16_I32 tglobaladdr:$off, (CONST_I32 0), I32:$val)>;
def : Pat<(truncstorei8 I64:$val, (WebAssemblywrapper tglobaladdr:$off)),
          (STORE8_I64 tglobaladdr:$off, (CONST_I32 0), I64:$val)>;
def : Pat<(truncstorei16 I64:$val, (WebAssemblywrapper tglobaladdr:$off)),
          (STORE16_I64 tglobaladdr:$off, (CONST_I32 0), I64:$val)>;
def : Pat<(truncstorei32 I64:$val, (WebAssemblywrapper tglobaladdr:$off)),
          (STORE32_I64 tglobaladdr:$off, (CONST_I32 0), I64:$val)>;
def : Pat<(truncstorei8 I32:$val, (WebAssemblywrapper texternalsym:$off)),
          (STORE8_I32 texternalsym:$off, (CONST_I32 0), I32:$val)>;
def : Pat<(truncstorei16 I32:$val, (WebAssemblywrapper texternalsym:$off)),
          (STORE16_I32 texternalsym:$off, (CONST_I32 0), I32:$val)>;
def : Pat<(truncstorei8 I64:$val, (WebAssemblywrapper texternalsym:$off)),
          (STORE8_I64 texternalsym:$off, (CONST_I32 0), I64:$val)>;
def : Pat<(truncstorei16 I64:$val, (WebAssemblywrapper texternalsym:$off)),
          (STORE16_I64 texternalsym:$off, (CONST_I32 0), I64:$val)>;
def : Pat<(truncstorei32 I64:$val, (WebAssemblywrapper texternalsym:$off)),
          (STORE32_I64 texternalsym:$off, (CONST_I32 0), I64:$val)>;

let Defs = [ARGUMENTS] in {

// Memory size.
def MEMORY_SIZE_I32 : I<(outs I32:$dst), (ins),
                        [(set I32:$dst, (int_wasm_memory_size))],
                        "memory_size\t$dst">,
                      Requires<[HasAddr32]>;
def MEMORY_SIZE_I64 : I<(outs I64:$dst), (ins),
                        [(set I64:$dst, (int_wasm_memory_size))],
                        "memory_size\t$dst">,
                      Requires<[HasAddr64]>;

// Grow memory.
def GROW_MEMORY_I32 : I<(outs), (ins I32:$delta),
                        [(int_wasm_grow_memory I32:$delta)],
                        "grow_memory\t$delta">,
                      Requires<[HasAddr32]>;
def GROW_MEMORY_I64 : I<(outs), (ins I64:$delta),
                        [(int_wasm_grow_memory I64:$delta)],
                        "grow_memory\t$delta">,
                      Requires<[HasAddr64]>;

} // Defs = [ARGUMENTS]
