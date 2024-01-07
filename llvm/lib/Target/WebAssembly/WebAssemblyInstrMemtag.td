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

defm MEMORY_RANDOMTAG_A#B: I<(outs rc:$dst), (ins i32imm:$flags, rc:$src),
                    (outs), (ins i32imm:$flags),
                    [(set rc:$dst,
		    	        (int_wasm_memory_randomtag i32imm:$flags, rc:$src))],
                    "memory.randomtag\t$dst, $flags, $src",
                    "memory.randomtag\t$flags",
                    0xfc20>;

defm MEMORY_COPYTAG_A#B: I<(outs rc:$dstwsrctag), (ins i32imm:$flags, rc:$dst, rc:$src),
                    (outs), (ins i32imm:$flags),
                    [(set rc:$dstwsrctag,
		    	        (int_wasm_memory_copytag i32imm:$flags, rc:$dst, rc:$src))],
                    "memory.copytag\t$dstwsrctag, $flags, $dst, $src",
                    "memory.copytag\t$flags",
                    0xfc21>;

defm MEMORY_SUBTAG_A#B: I<(outs rc:$diff), (ins i32imm:$flags, rc:$minuendptr, rc:$subtrahendptr),
                    (outs), (ins i32imm:$flags),
                    [(set rc:$diff,
                        (int_wasm_memory_subtag i32imm:$flags, rc:$minuendptr, rc:$subtrahendptr))],
                    "memory.subtag\t$diff, $flags, $minuendptr, $subtrahendptr",
                    "memory.subtag\t$flags",
                    0xfc22>;

defm MEMORY_LOADTAG_A#B: I<(outs rc:$taggedptr), (ins i32imm:$flags, rc:$src),
                    (outs), (ins i32imm:$flags),
                    [(set rc:$taggedptr,
		    	        (int_wasm_memory_loadtag i32imm:$flags, rc:$src))],
                    "memory.loadtag\t$taggedptr, $flags, $src",
                    "memory.loadtag\t$flags",
                    0xfc23>;

defm MEMORY_STORETAG_A#B: I<(outs), (ins i32imm:$flags, rc:$src, rc:$b16),
                    (outs), (ins i32imm:$flags),
                    [(int_wasm_memory_storetag i32imm:$flags, rc:$src, rc:$b16)],
                    "memory.storetag\t$flags, $src, $b16",
                    "memory.storetag\t$flags",
                    0xfc24>;

defm MEMORY_STOREZTAG_A#B: I<(outs), (ins i32imm:$flags, rc:$src, rc:$b16),
                    (outs), (ins i32imm:$flags),
                    [(int_wasm_memory_storeztag i32imm:$flags, rc:$src, rc:$b16)],
                    "memory.storeztag\t$flags, $src, $b16",
                    "memory.storeztag\t$flags",
                    0xfc25>;

defm MEMORY_STORE1TAG_A#B: I<(outs), (ins i32imm:$flags, rc:$src, rc:$b16),
                    (outs), (ins i32imm:$flags),
                    [(int_wasm_memory_store1tag i32imm:$flags, rc:$src, rc:$b16)],
                    "memory.store1tag\t$flags, $src, $b16",
                    "memory.store1tag\t$flags",
                    0xfc26>;

defm MEMORY_STOREZ1TAG_A#B: I<(outs), (ins i32imm:$flags, rc:$src, rc:$b16),
                    (outs), (ins i32imm:$flags),
                    [(int_wasm_memory_storez1tag i32imm:$flags, rc:$src, rc:$b16)],
                    "memory.storez1tag\t$flags, $src, $b16",
                    "memory.storez1tag\t$flags",
                    0xfc27>;

defm MEMORY_RANDOMSTORETAG_A#B: I<(outs rc:$dst), (ins i32imm:$flags, rc:$src, rc:$b16),
                    (outs), (ins i32imm:$flags),
                    [(set rc:$dst,
                        (int_wasm_memory_randomstoretag rc:$src, i32imm:$flags, rc:$b16))],
                    "memory.randomstoretag\t$dst, $flags, $src, $b16",
                    "memory.randomstoretag\t$flags",
                    0xfc28>;

defm MEMORY_RANDOMSTOREZTAG_A#B: I<(outs rc:$dst), (ins i32imm:$flags, rc:$src, rc:$b16),
                    (outs), (ins i32imm:$flags),
                    [(set rc:$dst,
                        (int_wasm_memory_randomstoreztag rc:$src, i32imm:$flags, rc:$b16))],
                    "memory.randomstoreztag\t$dst, $flags, $src, $b16",
                    "memory.randomstoreztag\t$flags",
                    0xfc29>;

defm MEMORY_RANDOMSTORE1TAG_A#B: I<(outs rc:$dst), (ins i32imm:$flags, rc:$src, rc:$b16),
                    (outs), (ins i32imm:$flags),
                    [(set rc:$dst,
                        (int_wasm_memory_randomstore1tag rc:$src, i32imm:$flags, rc:$b16))],
                    "memory.randomstore1tag\t$dst, $flags, $src, $b16",
                    "memory.randomstore1tag\t$flags",
                    0xfc2a>;

defm MEMORY_RANDOMSTOREZ1TAG_A#B: I<(outs rc:$dst), (ins i32imm:$flags, rc:$src, rc:$b16),
                    (outs), (ins i32imm:$flags),
                    [(set rc:$dst,
                        (int_wasm_memory_randomstorez1tag rc:$src, i32imm:$flags, rc:$b16))],
                    "memory.randomstorez1tag\t$dst, $flags, $src, $b16",
                    "memory.randomstorez1tag\t$flags",
                    0xfc2b>;

defm MEMORY_ADDNEXTTAG_A#B: I<(outs rc:$nexttaggedptr), (ins i32imm:$flags, rc:$taggedptr, rc:$offset),
                    (outs), (ins i32imm:$flags),
                    [(set rc:$nexttaggedptr,
                        (int_wasm_memory_addnexttag i32imm:$flags, rc:$taggedptr, rc:$offset))],
                    "memory.addnexttag\t$nexttaggedptr, $flags, $taggedptr, $offset",
                    "memory.addnexttag\t$flags",
                    0xfc2c>;

}

defm : MemoryTaggingOps<I32, "32">;
defm : MemoryTaggingOps<I64, "64">;
