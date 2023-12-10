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

defm MEMORY_RANDOMTAG_A#B: I<(outs rc:$dst), (ins rc:$src),
                    (outs), (ins),
                    [(set rc:$dst,
		    	(int_wasm_memory_randomtag rc:$src))],
                    "memory.randomtag\t$dst, $src",
                    "memory.randomtag\t",
                    0xfc20>;

defm MEMORY_COPYTAG_A#B: I<(outs rc:$dstwsrctag), (ins rc:$dst, rc:$src),
                    (outs), (ins),
                    [(set rc:$dstwsrctag,
		    	(int_wasm_memory_copytag rc:$dst, rc:$src))],
                    "memory.copytag\t$dstwsrctag, $dst, $src",
                    "memory.copytag\t",
                    0xfc21>;

defm MEMORY_SUBTAG_A#B: I<(outs rc:$diff), (ins rc:$minuendptr, rc:$subtrahendptr),
                    (outs), (ins),
                    [(set rc:$diff,
                        (int_wasm_memory_subtag rc:$minuendptr, rc:$subtrahendptr))],
                    "memory.subtag\t$diff, $minuendptr, $subtrahendptr",
                    "memory.subtag\t",
                    0xfc22>;

defm MEMORY_LOADTAG_A#B: I<(outs rc:$taggedptr), (ins rc:$src),
                    (outs), (ins),
                    [(set rc:$taggedptr,
		    	(int_wasm_memory_loadtag rc:$src))],
                    "memory.loadtag\t$taggedptr, $src",
                    "memory.loadtag\t",
                    0xfc23>;

defm MEMORY_STORETAG_A#B: I<(outs), (ins rc:$src, rc:$b16),
                    (outs), (ins),
                    [(int_wasm_memory_storetag rc:$src, rc:$b16)],
                    "memory.storetag\t$src, $b16",
                    "memory.storetag\t",
                    0xfc24>;

defm MEMORY_STOREZTAG_A#B: I<(outs), (ins rc:$src, rc:$b16),
                    (outs), (ins),
                    [(int_wasm_memory_storeztag rc:$src, rc:$b16)],
                    "memory.storeztag\t$src, $b16",
                    "memory.storeztag\t",
                    0xfc25>;

defm MEMORY_STORE1TAG_A#B: I<(outs), (ins rc:$src, rc:$b16),
                    (outs), (ins),
                    [(int_wasm_memory_store1tag rc:$src, rc:$b16)],
                    "memory.store1tag\t$src, $b16",
                    "memory.store1tag\t",
                    0xfc26>;

defm MEMORY_STOREZ1TAG_A#B: I<(outs), (ins rc:$src, rc:$b16),
                    (outs), (ins),
                    [(int_wasm_memory_storez1tag rc:$src, rc:$b16)],
                    "memory.storez1tag\t$src, $b16",
                    "memory.storez1tag\t",
                    0xfc27>;

defm MEMORY_RANDOMSTORETAG_A#B: I<(outs rc:$dst), (ins rc:$src, rc:$b16),
                    (outs), (ins),
                    [(set rc:$dst,
                        (int_wasm_memory_randomstoretag rc:$src, rc:$b16))],
                    "memory.randomstoretag\t$dst, $src, $b16",
                    "memory.randomstoretag\t",
                    0xfc28>;

defm MEMORY_RANDOMSTOREZTAG_A#B: I<(outs rc:$dst), (ins rc:$src, rc:$b16),
                    (outs), (ins),
                    [(set rc:$dst,
                        (int_wasm_memory_randomstoreztag rc:$src, rc:$b16))],
                    "memory.randomstoreztag\t$dst, $src, $b16",
                    "memory.randomstoreztag\t",
                    0xfc29>;

defm MEMORY_RANDOMSTORE1TAG_A#B: I<(outs rc:$dst), (ins rc:$src, rc:$b16),
                    (outs), (ins),
                    [(set rc:$dst,
                        (int_wasm_memory_randomstore1tag rc:$src, rc:$b16))],
                    "memory.randomstore1tag\t$dst, $src, $b16",
                    "memory.randomstore1tag\t",
                    0xfc30>;

defm MEMORY_RANDOMSTOREZ1TAG_A#B: I<(outs rc:$dst), (ins rc:$src, rc:$b16),
                    (outs), (ins),
                    [(set rc:$dst,
                        (int_wasm_memory_randomstorez1tag rc:$src, rc:$b16))],
                    "memory.randomstorez1tag\t$dst, $src, $b16",
                    "memory.randomstorez1tag\t",
                    0xfc31>;

}

defm : MemoryTaggingOps<I32, "32">;
defm : MemoryTaggingOps<I64, "64">;
