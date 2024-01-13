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

let hasSideEffects = 1 in
defm MEMORY_RANDOMTAG_A#B: I<(outs rc:$dst), (ins i32imm:$tableidx, rc:$src),
                    (outs), (ins i32imm:$tableidx),
                    [(set rc:$dst,
                        (int_wasm_memory_randomtag (i32 imm:$tableidx), rc:$src))],
                    "memory.randomtag\t$dst, $tableidx, $src",
                    "memory.randomtag\t$tableidx",
                    0xfc20>;

defm MEMORY_COPYTAG_A#B: I<(outs rc:$dstwsrctag), (ins i32imm:$tableidx, rc:$dst, rc:$src),
                    (outs), (ins i32imm:$tableidx),
                    [(set rc:$dstwsrctag,
                        (int_wasm_memory_copytag (i32 imm:$tableidx), rc:$dst, rc:$src))],
                    "memory.copytag\t$dstwsrctag, $tableidx, $dst, $src",
                    "memory.copytag\t$tableidx",
                    0xfc21>;

defm MEMORY_SUBTAG_A#B: I<(outs rc:$diff), (ins i32imm:$tableidx, rc:$minuendptr, rc:$subtrahendptr),
                    (outs), (ins i32imm:$tableidx),
                    [(set rc:$diff,
                        (int_wasm_memory_subtag (i32 imm:$tableidx), rc:$minuendptr, rc:$subtrahendptr))],
                    "memory.subtag\t$diff, $tableidx, $minuendptr, $subtrahendptr",
                    "memory.subtag\t$tableidx",
                    0xfc22>;

let mayLoad = 1 in
defm MEMORY_LOADTAG_A#B: I<(outs rc:$taggedptr), (ins i32imm:$tableidx, rc:$src),
                    (outs), (ins i32imm:$tableidx),
                    [(set rc:$taggedptr,
                        (int_wasm_memory_loadtag (i32 imm:$tableidx), rc:$src))],
                    "memory.loadtag\t$taggedptr, $tableidx, $src",
                    "memory.loadtag\t$tableidx",
                    0xfc23>;

let mayStore = 1 in
defm MEMORY_STORETAG_A#B: I<(outs), (ins i32imm:$tableidx, rc:$src, rc:$b16),
                    (outs), (ins i32imm:$tableidx),
                    [(int_wasm_memory_storetag (i32 imm:$tableidx), rc:$src, rc:$b16)],
                    "memory.storetag\t$tableidx, $src, $b16",
                    "memory.storetag\t$tableidx",
                    0xfc24>;

let mayStore = 1 in
defm MEMORY_STOREZTAG_A#B: I<(outs), (ins i32imm:$tableidx, rc:$src, rc:$b16),
                    (outs), (ins i32imm:$tableidx),
                    [(int_wasm_memory_storeztag (i32 imm:$tableidx), rc:$src, rc:$b16)],
                    "memory.storeztag\t$tableidx, $src, $b16",
                    "memory.storeztag\t$tableidx",
                    0xfc25>;

let mayStore = 1 in
defm MEMORY_STORE1TAG_A#B: I<(outs), (ins i32imm:$tableidx, rc:$src, rc:$b16),
                    (outs), (ins i32imm:$tableidx),
                    [(int_wasm_memory_store1tag (i32 imm:$tableidx), rc:$src, rc:$b16)],
                    "memory.store1tag\t$tableidx, $src, $b16",
                    "memory.store1tag\t$tableidx",
                    0xfc26>;

let mayStore = 1 in
defm MEMORY_STOREZ1TAG_A#B: I<(outs), (ins i32imm:$tableidx, rc:$src, rc:$b16),
                    (outs), (ins i32imm:$tableidx),
                    [(int_wasm_memory_storez1tag (i32 imm:$tableidx), rc:$src, rc:$b16)],
                    "memory.storez1tag\t$tableidx, $src, $b16",
                    "memory.storez1tag\t$tableidx",
                    0xfc27>;

let hasSideEffects = 1, mayStore = 1 in
defm MEMORY_RANDOMSTORETAG_A#B: I<(outs rc:$dst), (ins i32imm:$tableidx, rc:$src, rc:$b16),
                    (outs), (ins i32imm:$tableidx),
                    [(set rc:$dst,
                        (int_wasm_memory_randomstoretag (i32 imm:$tableidx), rc:$src, rc:$b16))],
                    "memory.randomstoretag\t$dst, $tableidx, $src, $b16",
                    "memory.randomstoretag\t$tableidx",
                    0xfc28>;

let hasSideEffects = 1, mayStore = 1 in
defm MEMORY_RANDOMSTOREZTAG_A#B: I<(outs rc:$dst), (ins i32imm:$tableidx, rc:$src, rc:$b16),
                    (outs), (ins i32imm:$tableidx),
                    [(set rc:$dst,
                        (int_wasm_memory_randomstoreztag (i32 imm:$tableidx), rc:$src, rc:$b16))],
                    "memory.randomstoreztag\t$dst, $src, $tableidx, $b16",
                    "memory.randomstoreztag\t$tableidx",
                    0xfc29>;

let hasSideEffects = 1, mayStore = 1 in
defm MEMORY_RANDOMSTORE1TAG_A#B: I<(outs rc:$dst), (ins i32imm:$tableidx, rc:$src, rc:$b16),
                    (outs), (ins i32imm:$tableidx),
                    [(set rc:$dst,
                        (int_wasm_memory_randomstore1tag (i32 imm:$tableidx), rc:$src, rc:$b16))],
                    "memory.randomstore1tag\t$dst, $tableidx, $src, $b16",
                    "memory.randomstore1tag\t$tableidx",
                    0xfc2a>;

let hasSideEffects = 1, mayStore = 1 in
defm MEMORY_RANDOMSTOREZ1TAG_A#B: I<(outs rc:$dst), (ins i32imm:$tableidx, rc:$src, rc:$b16),
                    (outs), (ins i32imm:$tableidx),
                    [(set rc:$dst,
                        (int_wasm_memory_randomstorez1tag (i32 imm:$tableidx), rc:$src, rc:$b16))],
                    "memory.randomstorez1tag\t$dst, $tableidx, $src, $b16",
                    "memory.randomstorez1tag\t$tableidx",
                    0xfc2b>;

let hasSideEffects = 1 in
defm MEMORY_HINTTAG_A#B: I<(outs rc:$dst), (ins i32imm:$tableidx, rc:$src, rc:$hintaddr, rc:$index),
                    (outs), (ins i32imm:$tableidx),
                    [(set rc:$dst,
                        (int_wasm_memory_hinttag (i32 imm:$tableidx), rc:$src, rc:$hintaddr, rc:$index))],
                    "memory.hinttag\t$dst, $tableidx, $src, $hintaddr, $index",
                    "memory.hinttag\t$tableidx",
                    0xfc2c>;

let hasSideEffects = 1, mayStore = 1 in
defm MEMORY_HINTSTORETAG_A#B: I<(outs rc:$dst), (ins i32imm:$tableidx, rc:$src, rc:$b16, rc:$hintaddr, rc:$index),
                    (outs), (ins i32imm:$tableidx),
                    [(set rc:$dst,
                        (int_wasm_memory_hintstoretag (i32 imm:$tableidx), rc:$src, rc:$b16, rc:$hintaddr, rc:$index))],
                    "memory.hintstoretag\t$dst, $tableidx, $src, $b16, $hintaddr, $index",
                    "memory.hintstoretag\t$tableidx",
                    0xfc2d>;

let hasSideEffects = 1, mayStore = 1 in
defm MEMORY_HINTSTOREZTAG_A#B: I<(outs rc:$dst), (ins i32imm:$tableidx, rc:$src, rc:$b16, rc:$hintaddr, rc:$index),
                    (outs), (ins i32imm:$tableidx),
                    [(set rc:$dst,
                        (int_wasm_memory_hintstoreztag (i32 imm:$tableidx), rc:$src, rc:$b16, rc:$hintaddr, rc:$index))],
                    "memory.hintstoreztag\t$dst, $tableidx, $src, $b16, $hintaddr, $index",
                    "memory.hintstoreztag\t$tableidx",
                    0xfc2e>;

let hasSideEffects = 1, mayStore = 1 in
defm MEMORY_HINTSTORE1TAG_A#B: I<(outs rc:$dst), (ins i32imm:$tableidx, rc:$src, rc:$b16, rc:$hintaddr, rc:$index),
                    (outs), (ins i32imm:$tableidx),
                    [(set rc:$dst,
                        (int_wasm_memory_hintstore1tag (i32 imm:$tableidx), rc:$src, rc:$b16, rc:$hintaddr, rc:$index))],
                    "memory.hintstore1tag\t$dst, $tableidx, $src, $b16, $hintaddr, $index",
                    "memory.hintstore1tag\t$tableidx",
                    0xfc2f>;

let hasSideEffects = 1, mayStore = 1 in
defm MEMORY_HINTSTOREZ1TAG_A#B: I<(outs rc:$dst), (ins i32imm:$tableidx, rc:$src, rc:$b16, rc:$hintaddr, rc:$index),
                    (outs), (ins i32imm:$tableidx),
                    [(set rc:$dst,
                        (int_wasm_memory_hintstorez1tag (i32 imm:$tableidx), rc:$src, rc:$b16, rc:$hintaddr, rc:$index))],
                    "memory.hintstorez1tag\t$dst, $tableidx, $src, $b16, $hintaddr, $index",
                    "memory.hintstorez1tag\t$tableidx",
                    0xfc30>;

}

defm : MemoryTaggingOps<I32, "32">;
defm : MemoryTaggingOps<I64, "64">;
