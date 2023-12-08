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
                    0xfc12, true>;

}

defm : MemoryTaggingOps<I32, "32">;
defm : MemoryTaggingOps<I64, "64">;
