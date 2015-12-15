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
//  - WebAssemblyTargetLowering::isLegalAddressingMode
//  - WebAssemblyTargetLowering having to do with atomics
//  - Each has optional alignment and immediate byte offset.

// WebAssembly has i8/i16/i32/i64/f32/f64 memory types, but doesn't have i8/i16
// local types. These memory-only types instead zero- or sign-extend into local
// types when loading, and truncate when storing.

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

// Select loads with no constant offset.
def : Pat<(i32 (load I32:$addr)), (LOAD_I32 0, $addr)>;
def : Pat<(i64 (load I32:$addr)), (LOAD_I64 0, $addr)>;
def : Pat<(f32 (load I32:$addr)), (LOAD_F32 0, $addr)>;
def : Pat<(f64 (load I32:$addr)), (LOAD_F64 0, $addr)>;

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

// "Don't care" extending load become zero-extending load.
def : Pat<(i32 (extloadi8 I32:$addr)),  (LOAD8_U_I32 0, $addr)>;
def : Pat<(i32 (extloadi16 I32:$addr)), (LOAD16_U_I32 0, $addr)>;
def : Pat<(i64 (extloadi8 I32:$addr)),  (LOAD8_U_I64 0, $addr)>;
def : Pat<(i64 (extloadi16 I32:$addr)), (LOAD16_U_I64 0, $addr)>;
def : Pat<(i64 (extloadi32 I32:$addr)), (LOAD32_U_I64 0, $addr)>;

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

def : Pat<(store I32:$val, I32:$addr), (STORE_I32 0, I32:$addr, I32:$val)>;
def : Pat<(store I64:$val, I32:$addr), (STORE_I64 0, I32:$addr, I64:$val)>;
def : Pat<(store F32:$val, I32:$addr), (STORE_F32 0, I32:$addr, F32:$val)>;
def : Pat<(store F64:$val, I32:$addr), (STORE_F64 0, I32:$addr, F64:$val)>;

// FIXME: This pattern matches an immediate to actually use the offset field
// in the store instruction; however only unsigned offsets are supported in
// wasm, so we need to constrain the immediate we match. This may require
// custom code rather than a simple pattern.
// def : Pat<(store I32:$val, (add I32:$addr, (i32 imm:$off))),
//           (STORE_I32 imm:$off, I32:$addr, I32:$val)>;

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
