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
                    0xfc12>;

defm MEMORY_LOADTAG_A#B: I<(outs rc:$taggedptr), (ins rc:$src),
                    (outs), (ins),
                    [(set rc:$taggedptr,
		    	(int_wasm_memory_loadtag rc:$src))],
                    "memory.loadtag\t$taggedptr, $src",
                    "memory.loadtag\t",
                    0xfc13>;

defm MEMORY_STORETAG_A#B: I<(outs), (ins rc:$src, rc:$b16),
                    (outs), (ins),
                    [(int_wasm_memory_storetag rc:$src, rc:$b16)],
                    "memory.storetag\t$src, $b16",
                    "memory.storetag\t",
                    0xfc14>;

defm MEMORY_STOREZTAG_A#B: I<(outs), (ins rc:$src, rc:$b16),
                    (outs), (ins),
                    [(int_wasm_memory_storeztag rc:$src, rc:$b16)],
                    "memory.storeztag\t$src, $b16",
                    "memory.storeztag\t",
                    0xfc15>;

}

defm : MemoryTaggingOps<I32, "32">;
defm : MemoryTaggingOps<I64, "64">;
