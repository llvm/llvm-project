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

// FIXME:
//  - HasAddr64
//  - WebAssemblyTargetLowering::isLegalAddressingMode
//  - WebAssemblyTargetLowering having to do with atomics
//  - Each has optional alignment and immediate byte offset.

// WebAssembly has i8/i16/i32/i64/f32/f64 memory types, but doesn't have i8/i16
// local types. These memory-only types instead zero- or sign-extend into local
// types when loading, and truncate when storing.

// Basic load.
def LOAD_I32 : I<(outs I32:$dst), (ins I32:$addr),
                 [(set I32:$dst, (load I32:$addr))],
                 "i32.load $dst, $addr">;
def LOAD_I64 : I<(outs I64:$dst), (ins I32:$addr),
                 [(set I64:$dst, (load I32:$addr))],
                 "i64.load $dst, $addr">;
def LOAD_F32 : I<(outs F32:$dst), (ins I32:$addr),
                 [(set F32:$dst, (load I32:$addr))],
                 "f32.load $dst, $addr">;
def LOAD_F64 : I<(outs F64:$dst), (ins I32:$addr),
                 [(set F64:$dst, (load I32:$addr))],
                 "f64.load $dst, $addr">;

// Extending load.
def LOAD8_S_I32  : I<(outs I32:$dst), (ins I32:$addr),
                     [(set I32:$dst, (sextloadi8 I32:$addr))],
                     "i32.load8_s $dst, $addr">;
def LOAD8_U_I32  : I<(outs I32:$dst), (ins I32:$addr),
                     [(set I32:$dst, (zextloadi8 I32:$addr))],
                     "i32.load8_u $dst, $addr">;
def LOAD16_S_I32 : I<(outs I32:$dst), (ins I32:$addr),
                     [(set I32:$dst, (sextloadi16 I32:$addr))],
                     "i32.load16_s $dst, $addr">;
def LOAD16_U_I32 : I<(outs I32:$dst), (ins I32:$addr),
                     [(set I32:$dst, (zextloadi16 I32:$addr))],
                     "i32.load16_u $dst, $addr">;
def LOAD8_S_I64  : I<(outs I64:$dst), (ins I32:$addr),
                     [(set I64:$dst, (sextloadi8 I32:$addr))],
                     "i64.load8_s $dst, $addr">;
def LOAD8_U_I64  : I<(outs I64:$dst), (ins I32:$addr),
                     [(set I64:$dst, (zextloadi8 I32:$addr))],
                     "i64.load8_u $dst, $addr">;
def LOAD16_S_I64 : I<(outs I64:$dst), (ins I32:$addr),
                     [(set I64:$dst, (sextloadi16 I32:$addr))],
                     "i64.load16_s $dst, $addr">;
def LOAD16_U_I64 : I<(outs I64:$dst), (ins I32:$addr),
                     [(set I64:$dst, (zextloadi16 I32:$addr))],
                     "i64.load16_u $dst, $addr">;
def LOAD32_S_I64 : I<(outs I64:$dst), (ins I32:$addr),
                     [(set I64:$dst, (sextloadi32 I32:$addr))],
                     "i64.load32_s $dst, $addr">;
def LOAD32_U_I64 : I<(outs I64:$dst), (ins I32:$addr),
                     [(set I64:$dst, (zextloadi32 I32:$addr))],
                     "i64.load32_u $dst, $addr">;

// "Don't care" extending load become zero-extending load.
def : Pat<(i32 (extloadi8 I32:$addr)),  (LOAD8_U_I32 $addr)>;
def : Pat<(i32 (extloadi16 I32:$addr)), (LOAD16_U_I32 $addr)>;
def : Pat<(i64 (extloadi8 I32:$addr)),  (LOAD8_U_I64 $addr)>;
def : Pat<(i64 (extloadi16 I32:$addr)), (LOAD16_U_I64 $addr)>;
def : Pat<(i64 (extloadi32 I32:$addr)), (LOAD32_U_I64 $addr)>;

// Basic store.
// Note: WebAssembly inverts SelectionDAG's usual operand order.
def STORE_I32  : I<(outs), (ins I32:$addr, I32:$val),
                   [(store i32:$val, I32:$addr)],
                   "i32.store $addr, $val">;
def STORE_I64  : I<(outs), (ins I32:$addr, I64:$val),
                   [(store i64:$val, I32:$addr)],
                   "i64.store $addr, $val">;
def STORE_F32  : I<(outs), (ins I32:$addr, F32:$val),
                   [(store f32:$val, I32:$addr)],
                   "f32.store $addr, $val">;
def STORE_F64  : I<(outs), (ins I32:$addr, F64:$val),
                   [(store f64:$val, I32:$addr)],
                   "f64.store $addr, $val">;

// Truncating store.
def STORE8_I32  : I<(outs), (ins I32:$addr, I32:$val),
                    [(truncstorei8 I32:$val, I32:$addr)],
                    "i32.store8 $addr, $val">;
def STORE16_I32 : I<(outs), (ins I32:$addr, I32:$val),
                    [(truncstorei16 I32:$val, I32:$addr)],
                    "i32.store16 $addr, $val">;
def STORE8_I64  : I<(outs), (ins I32:$addr, I64:$val),
                    [(truncstorei8 I64:$val, I32:$addr)],
                    "i64.store8 $addr, $val">;
def STORE16_I64 : I<(outs), (ins I32:$addr, I64:$val),
                    [(truncstorei16 I64:$val, I32:$addr)],
                    "i64.store16 $addr, $val">;
def STORE32_I64 : I<(outs), (ins I32:$addr, I64:$val),
                    [(truncstorei32 I64:$val, I32:$addr)],
                    "i64.store32 $addr, $val">;

// Memory size.
def memory_size_I32 : I<(outs I32:$dst), (ins),
                        [(set I32:$dst, (int_wasm_memory_size))],
                        "i32.memory_size $dst">,
                      Requires<[HasAddr32]>;
def memory_size_I64 : I<(outs I64:$dst), (ins),
                        [(set I64:$dst, (int_wasm_memory_size))],
                        "i64.memory_size $dst">,
                      Requires<[HasAddr64]>;

// Grow memory.
def grow_memory_I32 : I<(outs), (ins I32:$delta),
                        [(int_wasm_grow_memory I32:$delta)],
                        "i32.grow_memory $delta">,
                      Requires<[HasAddr32]>;
def grow_memory_I64 : I<(outs), (ins I64:$delta),
                        [(int_wasm_grow_memory I64:$delta)],
                        "i64.grow_memory $delta">,
                      Requires<[HasAddr64]>;
