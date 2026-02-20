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

defm MEMTAG_STATUS_A#B: I<(outs rc:$statuscode), (ins i32imm:$tableidx),
                    (outs), (ins i32imm:$tableidx),
                    [(set rc:$statuscode,
                        (int_wasm_memtag_status (i32 imm:$tableidx)))],
                    "memtag.status\t$statuscode, $tableidx",
                    "memtag.status\t$tableidx",
                    0xfc20>;

defm MEMTAG_EXTRACT_A#B: I<(outs rc:$taggedval), (ins i32imm:$tableidx, rc:$ptr),
                    (outs), (ins i32imm:$tableidx),
                    [(set rc:$taggedval,
                        (int_wasm_memtag_extract (i32 imm:$tableidx), rc:$ptr))],
                    "memtag.extract\t$taggedval, $tableidx, $ptr",
                    "memtag.extract\t$tableidx",
                    0xfc21>;

defm MEMTAG_INSERT_A#B: I<(outs rc:$taggedptr), (ins i32imm:$tableidx, rc:$ptr, rc:$insertval),
                    (outs), (ins i32imm:$tableidx),
                    [(set rc:$taggedptr,
                        (int_wasm_memtag_insert (i32 imm:$tableidx), rc:$ptr, rc:$insertval))],
                    "memtag.insert\t$taggedptr, $tableidx, $ptr, $insertval",
                    "memtag.insert\t$tableidx",
                    0xfc22>;

defm MEMTAG_TAGBITS_A#B: I<(outs rc:$taggedbits), (ins i32imm:$tableidx),
                    (outs), (ins i32imm:$tableidx),
                    [(set rc:$taggedbits,
                        (int_wasm_memtag_tagbits (i32 imm:$tableidx)))],
                    "memtag.tagbits\t$taggedbits, $tableidx",
                    "memtag.tagbits\t$tableidx",
                    0xfc23>;

defm MEMTAG_STARTBIT_A#B: I<(outs rc:$startbit), (ins i32imm:$tableidx),
                    (outs), (ins i32imm:$tableidx),
                    [(set rc:$startbit,
                        (int_wasm_memtag_startbit (i32 imm:$tableidx)))],
                    "memtag.startbit\t$startbit, $tableidx",
                    "memtag.startbit\t$tableidx",
                    0xfc24>;

defm MEMTAG_COPY_A#B: I<(outs rc:$dstwsrctag), (ins i32imm:$tableidx, rc:$dst, rc:$src),
                    (outs), (ins i32imm:$tableidx),
                    [(set rc:$dstwsrctag,
                        (int_wasm_memtag_copy (i32 imm:$tableidx), rc:$dst, rc:$src))],
                    "memtag.copy\t$dstwsrctag, $tableidx, $dst, $src",
                    "memtag.copy\t$tableidx",
                    0xfc25>;

defm MEMTAG_SUB_A#B: I<(outs rc:$diff), (ins i32imm:$tableidx, rc:$minuendptr, rc:$subtrahendptr),
                    (outs), (ins i32imm:$tableidx),
                    [(set rc:$diff,
                        (int_wasm_memtag_sub (i32 imm:$tableidx), rc:$minuendptr, rc:$subtrahendptr))],
                    "memtag.sub\t$diff, $tableidx, $minuendptr, $subtrahendptr",
                    "memtag.sub\t$tableidx",
                    0xfc26>;

let mayLoad = 1 in
defm MEMTAG_LOAD_A#B: I<(outs rc:$taggedptr), (ins i32imm:$tableidx, rc:$src),
                    (outs), (ins i32imm:$tableidx),
                    [(set rc:$taggedptr,
                        (int_wasm_memtag_load (i32 imm:$tableidx), rc:$src))],
                    "memtag.load\t$taggedptr, $tableidx, $src",
                    "memtag.load\t$tableidx",
                    0xfc27>;

defm MEMTAG_UNTAG_A#B: I<(outs rc:$untaggedptr), (ins i32imm:$tableidx, rc:$ptr),
                    (outs), (ins i32imm:$tableidx),
                    [(set rc:$untaggedptr,
                        (int_wasm_memtag_untag (i32 imm:$tableidx), rc:$ptr))],
                    "memtag.untag\t$untaggedptr, $tableidx, $ptr",
                    "memtag.untag\t$tableidx",
                    0xfc28>;

let mayStore = 1 in
defm MEMTAG_UNTAGSTORE_A#B: I<(outs rc:$untaggedptr), (ins i32imm:$tableidx, rc:$src, rc:$b16),
                    (outs), (ins i32imm:$tableidx),
                    [(set rc:$untaggedptr,
                        (int_wasm_memtag_untagstore (i32 imm:$tableidx), rc:$src, rc:$b16))],
                    "memtag.untagstore\t$tableidx, $src, $b16",
                    "memtag.untagstore\t$tableidx",
                    0xfc29>;

let mayStore = 1 in
defm MEMTAG_UNTAGSTOREZ_A#B: I<(outs rc:$untaggedptr), (ins i32imm:$tableidx, rc:$src, rc:$b16),
                    (outs), (ins i32imm:$tableidx),
                    [(set rc:$untaggedptr,
                        (int_wasm_memtag_untagstorez (i32 imm:$tableidx), rc:$src, rc:$b16))],
                    "memtag.untagstorez\t$tableidx, $src, $b16",
                    "memtag.untagstorez\t$tableidx",
                    0xfc2a>;

let mayStore = 1 in
defm MEMTAG_STORE_A#B: I<(outs), (ins i32imm:$tableidx, rc:$src, rc:$b16),
                    (outs), (ins i32imm:$tableidx),
                    [(int_wasm_memtag_store (i32 imm:$tableidx), rc:$src, rc:$b16)],
                    "memtag.store\t$tableidx, $src, $b16",
                    "memtag.store\t$tableidx",
                    0xfc2b>;

let mayStore = 1 in
defm MEMTAG_STOREZ_A#B: I<(outs), (ins i32imm:$tableidx, rc:$src, rc:$b16),
                    (outs), (ins i32imm:$tableidx),
                    [(int_wasm_memtag_storez (i32 imm:$tableidx), rc:$src, rc:$b16)],
                    "memtag.storez\t$tableidx, $src, $b16",
                    "memtag.storez\t$tableidx",
                    0xfc2c>;

let hasSideEffects = 1 in
defm MEMTAG_RANDOM_A#B: I<(outs rc:$dst), (ins i32imm:$tableidx, rc:$src),
                    (outs), (ins i32imm:$tableidx),
                    [(set rc:$dst,
                        (int_wasm_memtag_random (i32 imm:$tableidx), rc:$src))],
                    "memtag.random\t$dst, $tableidx, $src",
                    "memtag.random\t$tableidx",
                    0xfc2d>;

let hasSideEffects = 1, mayStore = 1 in
defm MEMTAG_RANDOMSTORE_A#B: I<(outs rc:$dst), (ins i32imm:$tableidx, rc:$src, rc:$b16),
                    (outs), (ins i32imm:$tableidx),
                    [(set rc:$dst,
                        (int_wasm_memtag_randomstore (i32 imm:$tableidx), rc:$src, rc:$b16))],
                    "memtag.randomstore\t$dst, $tableidx, $src, $b16",
                    "memtag.randomstore\t$tableidx",
                    0xfc2e>;

let hasSideEffects = 1, mayStore = 1 in
defm MEMTAG_RANDOMSTOREZ_A#B: I<(outs rc:$dst), (ins i32imm:$tableidx, rc:$src, rc:$b16),
                    (outs), (ins i32imm:$tableidx),
                    [(set rc:$dst,
                        (int_wasm_memtag_randomstorez (i32 imm:$tableidx), rc:$src, rc:$b16))],
                    "memtag.randomstorez\t$dst, $src, $tableidx, $b16",
                    "memtag.randomstorez\t$tableidx",
                    0xfc2f>;

let hasSideEffects = 1 in
defm MEMTAG_RANDOMMASK_A#B: I<(outs rc:$dst), (ins i32imm:$tableidx, rc:$src, rc:$mask),
                    (outs), (ins i32imm:$tableidx),
                    [(set rc:$dst,
                        (int_wasm_memtag_randommask (i32 imm:$tableidx), rc:$src, rc:$mask))],
                    "memtag.randommask\t$dst, $tableidx, $src, $mask",
                    "memtag.randommask\t$tableidx",
                    0xfc30>;


let hasSideEffects = 1, mayStore = 1 in
defm MEMTAG_RANDOMMASKSTORE_A#B: I<(outs rc:$dst), (ins i32imm:$tableidx, rc:$src, rc:$b16, rc:$mask),
                    (outs), (ins i32imm:$tableidx),
                    [(set rc:$dst,
                        (int_wasm_memtag_randommaskstore (i32 imm:$tableidx), rc:$src, rc:$b16, rc:$mask))],
                    "memtag.randommaskstore\t$dst, $tableidx, $src, $b16, $mask",
                    "memtag.randommaskstore\t$tableidx",
                    0xfc31>;

let hasSideEffects = 1, mayStore = 1 in
defm MEMTAG_RANDOMMASKSTOREZ_A#B: I<(outs rc:$dst), (ins i32imm:$tableidx, rc:$src, rc:$b16, rc:$mask),
                    (outs), (ins i32imm:$tableidx),
                    [(set rc:$dst,
                        (int_wasm_memtag_randommaskstorez (i32 imm:$tableidx), rc:$src, rc:$b16, rc:$mask))],
                    "memtag.randommaskstorez\t$dst, $src, $tableidx, $b16, $mask",
                    "memtag.randommaskstorez\t$tableidx",
                    0xfc32>;

defm MEMTAG_ADD_A#B: I<(outs rc:$dst), (ins i32imm:$tableidx, rc:$src, rc:$addroffset, rc:$tagoffset),
                    (outs), (ins i32imm:$tableidx),
                    [(set rc:$dst,
                        (int_wasm_memtag_add (i32 imm:$tableidx), rc:$src, rc:$addroffset, rc:$tagoffset))],
                    "memtag.add\t$dst, $tableidx, $src, $addroffset, $tagoffset",
                    "memtag.add\t$tableidx",
                    0xfc33>;

let mayStore = 1 in
defm MEMTAG_ADDSTORE_A#B: I<(outs rc:$dst), (ins i32imm:$tableidx, rc:$src, rc:$b16, rc:$addroffset, rc:$tagoffset),
                    (outs), (ins i32imm:$tableidx),
                    [(set rc:$dst,
                        (int_wasm_memtag_addstore (i32 imm:$tableidx), rc:$src, rc:$b16, rc:$addroffset, rc:$tagoffset))],
                    "memtag.addstore\t$dst, $tableidx, $src, $b16, $addroffset, $tagoffset",
                    "memtag.addstore\t$tableidx",
                    0xfc34>;

let mayStore = 1 in
defm MEMTAG_ADDSTOREZ_A#B: I<(outs rc:$dst), (ins i32imm:$tableidx, rc:$src, rc:$b16, rc:$addroffset, rc:$tagoffset),
                    (outs), (ins i32imm:$tableidx),
                    [(set rc:$dst,
                        (int_wasm_memtag_addstorez (i32 imm:$tableidx), rc:$src, rc:$b16, rc:$addroffset, rc:$tagoffset))],
                    "memtag.addstorez\t$dst, $tableidx, $src, $b16, $addroffset, $tagoffset",
                    "memtag.addstorez\t$tableidx",
                    0xfc35>;

defm MEMTAG_HINT_A#B: I<(outs rc:$dst), (ins i32imm:$tableidx, rc:$src, rc:$hintaddr, rc:$index),
                    (outs), (ins i32imm:$tableidx),
                    [(set rc:$dst,
                        (int_wasm_memtag_hint (i32 imm:$tableidx), rc:$src, rc:$hintaddr, rc:$index))],
                    "memtag.hint\t$dst, $tableidx, $src, $hintaddr, $index",
                    "memtag.hint\t$tableidx",
                    0xfc36>;

let mayStore = 1 in
defm MEMTAG_HINTSTORE_A#B: I<(outs rc:$dst), (ins i32imm:$tableidx, rc:$src, rc:$b16, rc:$hintaddr, rc:$index),
                    (outs), (ins i32imm:$tableidx),
                    [(set rc:$dst,
                        (int_wasm_memtag_hintstore (i32 imm:$tableidx), rc:$src, rc:$b16, rc:$hintaddr, rc:$index))],
                    "memtag.hintstore\t$dst, $tableidx, $src, $b16, $hintaddr, $index",
                    "memtag.hintstore\t$tableidx",
                    0xfc37>;

let mayStore = 1 in
defm MEMTAG_HINTSTOREZ_A#B: I<(outs rc:$dst), (ins i32imm:$tableidx, rc:$src, rc:$b16, rc:$hintaddr, rc:$index),
                    (outs), (ins i32imm:$tableidx),
                    [(set rc:$dst,
                        (int_wasm_memtag_hintstorez (i32 imm:$tableidx), rc:$src, rc:$b16, rc:$hintaddr, rc:$index))],
                    "memtag.hintstorez\t$dst, $tableidx, $src, $b16, $hintaddr, $index",
                    "memtag.hintstorez\t$tableidx",
                    0xfc38>;

}

defm : MemoryTaggingOps<I32, "32">;
defm : MemoryTaggingOps<I64, "64">;
