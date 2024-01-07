// WebAssemblyInstrMemtag.td - WebAssembly Memtag codegen support -*- memtaggen -*-
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// WebAssembly Mem Table operand code-gen constructs.
/// Instructions that handle memtags
//===----------------------------------------------------------------------===//

multiclass MemoryTaggingOps<WebAssemblyRegClass rc, string B> {

defm MEMORY_RANDOMTAG_A#B: I<(outs rc:$dst), (ins table32_op:$table, rc:$src),
                    (outs), (ins table32_op:$table),
                    [(set rc:$dst,
                        (int_wasm_memory_randomtag (WebAssemblyWrapper tglobaladdr:$table), rc:$src))],
                    "memory.randomtag\t$dst, $table, $src",
                    "memory.randomtag\t$table",
                    0xfc20>;

defm MEMORY_COPYTAG_A#B: I<(outs rc:$dstwsrctag), (ins table32_op:$table, rc:$dst, rc:$src),
                    (outs), (ins table32_op:$table),
                    [(set rc:$dstwsrctag,
                        (int_wasm_memory_copytag (WebAssemblyWrapper tglobaladdr:$table), rc:$dst, rc:$src))],
                    "memory.copytag\t$dstwsrctag, $table, $dst, $src",
                    "memory.copytag\t$table",
                    0xfc21>;

defm MEMORY_SUBTAG_A#B: I<(outs rc:$diff), (ins table32_op:$table, rc:$minuendptr, rc:$subtrahendptr),
                    (outs), (ins table32_op:$table),
                    [(set rc:$diff,
                        (int_wasm_memory_subtag (WebAssemblyWrapper tglobaladdr:$table), rc:$minuendptr, rc:$subtrahendptr))],
                    "memory.subtag\t$diff, $table, $minuendptr, $subtrahendptr",
                    "memory.subtag\t$table",
                    0xfc22>;

defm MEMORY_LOADTAG_A#B: I<(outs rc:$taggedptr), (ins table32_op:$table, rc:$src),
                    (outs), (ins table32_op:$table),
                    [(set rc:$taggedptr,
                        (int_wasm_memory_loadtag (WebAssemblyWrapper tglobaladdr:$table), rc:$src))],
                    "memory.loadtag\t$taggedptr, $table, $src",
                    "memory.loadtag\t$table",
                    0xfc23>;

defm MEMORY_STORETAG_A#B: I<(outs), (ins table32_op:$table, rc:$src, rc:$b16),
                    (outs), (ins table32_op:$table),
                    [(int_wasm_memory_storetag (WebAssemblyWrapper tglobaladdr:$table), rc:$src, rc:$b16)],
                    "memory.storetag\t$table, $src, $b16",
                    "memory.storetag\t$table",
                    0xfc24>;

defm MEMORY_STOREZTAG_A#B: I<(outs), (ins table32_op:$table, rc:$src, rc:$b16),
                    (outs), (ins table32_op:$table),
                    [(int_wasm_memory_storeztag (WebAssemblyWrapper tglobaladdr:$table), rc:$src, rc:$b16)],
                    "memory.storeztag\t$table, $src, $b16",
                    "memory.storeztag\t$table",
                    0xfc25>;

defm MEMORY_STORE1TAG_A#B: I<(outs), (ins table32_op:$table, rc:$src, rc:$b16),
                    (outs), (ins table32_op:$table),
                    [(int_wasm_memory_store1tag (WebAssemblyWrapper tglobaladdr:$table), rc:$src, rc:$b16)],
                    "memory.store1tag\t$table, $src, $b16",
                    "memory.store1tag\t$table",
                    0xfc26>;

defm MEMORY_STOREZ1TAG_A#B: I<(outs), (ins table32_op:$table, rc:$src, rc:$b16),
                    (outs), (ins table32_op:$table),
                    [(int_wasm_memory_storez1tag (WebAssemblyWrapper tglobaladdr:$table), rc:$src, rc:$b16)],
                    "memory.storez1tag\t$table, $src, $b16",
                    "memory.storez1tag\t$table",
                    0xfc27>;

defm MEMORY_RANDOMSTORETAG_A#B: I<(outs rc:$dst), (ins table32_op:$table, rc:$src, rc:$b16),
                    (outs), (ins table32_op:$table),
                    [(set rc:$dst,
                        (int_wasm_memory_randomstoretag (WebAssemblyWrapper tglobaladdr:$table), rc:$src, rc:$b16))],
                    "memory.randomstoretag\t$dst, $table, $src, $b16",
                    "memory.randomstoretag\t$table",
                    0xfc28>;

defm MEMORY_RANDOMSTOREZTAG_A#B: I<(outs rc:$dst), (ins table32_op:$table, rc:$src, rc:$b16),
                    (outs), (ins table32_op:$table),
                    [(set rc:$dst,
                        (int_wasm_memory_randomstoreztag (WebAssemblyWrapper tglobaladdr:$table), rc:$src, rc:$b16))],
                    "memory.randomstoreztag\t$dst, $src, $table, $b16",
                    "memory.randomstoreztag\t$table",
                    0xfc29>;

defm MEMORY_RANDOMSTORE1TAG_A#B: I<(outs rc:$dst), (ins table32_op:$table, rc:$src, rc:$b16),
                    (outs), (ins table32_op:$table),
                    [(set rc:$dst,
                        (int_wasm_memory_randomstore1tag (WebAssemblyWrapper tglobaladdr:$table), rc:$src, rc:$b16))],
                    "memory.randomstore1tag\t$dst, $table, $src, $b16",
                    "memory.randomstore1tag\t$table",
                    0xfc2a>;

defm MEMORY_RANDOMSTOREZ1TAG_A#B: I<(outs rc:$dst), (ins table32_op:$table, rc:$src, rc:$b16),
                    (outs), (ins table32_op:$table),
                    [(set rc:$dst,
                        (int_wasm_memory_randomstorez1tag (WebAssemblyWrapper tglobaladdr:$table), rc:$src, rc:$b16))],
                    "memory.randomstorez1tag\t$dst, $table, $src, $b16",
                    "memory.randomstorez1tag\t$table",
                    0xfc2b>;

defm MEMORY_ADDNEXTTAG_A#B: I<(outs rc:$nexttaggedptr), (ins table32_op:$table, rc:$taggedptr, rc:$offset),
                    (outs), (ins table32_op:$table),
                    [(set rc:$nexttaggedptr,
                        (int_wasm_memory_addnexttag (WebAssemblyWrapper tglobaladdr:$table), rc:$taggedptr, rc:$offset))],
                    "memory.addnexttag\t$nexttaggedptr, $table, $taggedptr, $offset",
                    "memory.addnexttag\t$table",
                    0xfc2c>;

}

defm : MemoryTaggingOps<I32, "32">;
defm : MemoryTaggingOps<I64, "64">;
