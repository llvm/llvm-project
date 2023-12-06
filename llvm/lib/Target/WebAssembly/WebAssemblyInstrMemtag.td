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

defm MEMORY_RANDOMTAG: I<(outs I64:$dst), (ins I64:$src),
                    (outs), (ins),
                    [(set I64:$dst,
		    	(int_wasm_memory_randomtag (WebAssemblyWrapper I64:$src)))],
                    "memory.randomtag\t$dst, $src",
                    "memory.randomtag\t",
                    0xfc12, true>;
