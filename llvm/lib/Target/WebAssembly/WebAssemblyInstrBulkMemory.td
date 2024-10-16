// WebAssemblyInstrBulkMemory.td - bulk memory codegen support --*- tablegen -*-
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// WebAssembly bulk memory codegen constructs.
///
//===----------------------------------------------------------------------===//

// Instruction requiring HasBulkMemory and the bulk memory prefix byte
multiclass BULK_I<dag oops_r, dag iops_r, dag oops_s, dag iops_s,
                  list<dag> pattern_r, string asmstr_r = "",
                  string asmstr_s = "", bits<32> simdop = -1> {
  defm "" : I<oops_r, iops_r, oops_s, iops_s, pattern_r, asmstr_r, asmstr_s,
              !or(0xfc00, !and(0xff, simdop))>,
            Requires<[HasBulkMemory]>;
}

// Bespoke types and nodes for bulk memory ops

// memory.copy (may trap on empty ranges)
def wasm_memory_copy_t : SDTypeProfile<0, 5,
  [SDTCisInt<0>, SDTCisInt<1>, SDTCisPtrTy<2>, SDTCisPtrTy<3>, SDTCisInt<4>]
>;
def wasm_memory_copy : SDNode<"WebAssemblyISD::MEMORY_COPY", wasm_memory_copy_t,
                              [SDNPHasChain, SDNPMayLoad, SDNPMayStore]>;

// memory.copy with a branch to avoid trapping
def wasm_memcpy_t : SDTypeProfile<0, 5,
  [SDTCisInt<0>, SDTCisInt<1>, SDTCisPtrTy<2>, SDTCisPtrTy<3>, SDTCisInt<4>]
>;
def wasm_memcpy : SDNode<"WebAssemblyISD::MEMCPY", wasm_memcpy_t,
                         [SDNPHasChain, SDNPMayLoad, SDNPMayStore]>;

// memory.fill (may trap on empty ranges)
def wasm_memory_fill_t : SDTypeProfile<0, 4,
  [SDTCisInt<0>, SDTCisPtrTy<1>, SDTCisInt<2>, SDTCisInt<3>]
>;
def wasm_memory_fill : SDNode<"WebAssemblyISD::MEMORY_FILL", wasm_memory_fill_t,
                              [SDNPHasChain, SDNPMayStore]>;

// memory.fill with a branch to avoid trapping
def wasm_memset_t : SDTypeProfile<0, 4,
  [SDTCisInt<0>, SDTCisPtrTy<1>, SDTCisInt<2>, SDTCisInt<3>]
>;
def wasm_memset : SDNode<"WebAssemblyISD::MEMSET", wasm_memset_t,
                         [SDNPHasChain, SDNPMayStore]>;

multiclass BulkMemoryOps<WebAssemblyRegClass rc, string B> {

let mayStore = 1, hasSideEffects = 1 in
defm MEMORY_INIT_A#B :
  BULK_I<(outs),
         (ins i32imm_op:$seg, i32imm_op:$idx, rc:$dest,
              I32:$offset, I32:$size),
         (outs), (ins i32imm_op:$seg, i32imm_op:$idx),
         [],
         "memory.init\t$seg, $idx, $dest, $offset, $size",
         "memory.init\t$seg, $idx", 0x08>;

let hasSideEffects = 1 in
defm DATA_DROP :
  BULK_I<(outs), (ins i32imm_op:$seg), (outs), (ins i32imm_op:$seg),
         [],
         "data.drop\t$seg", "data.drop\t$seg", 0x09>;

}

defm : BulkMemoryOps<I32, "32">;
defm : BulkMemoryOps<I64, "64">;

// Define copy/fill manually instead of using the `BulkMemoryOps` multiclass
// because when a multiclass defines opcodes, it gives them anonymous names
// and we need opcodes with names so that we can handle them with custom code.

let mayLoad = 1, mayStore = 1 in
defm MEMORY_COPY_A32 :
  BULK_I<(outs), (ins i32imm_op:$src_idx, i32imm_op:$dst_idx,
                      I32:$dst, I32:$src, I32:$len),
         (outs), (ins i32imm_op:$src_idx, i32imm_op:$dst_idx),
         [(wasm_memory_copy (i32 imm:$src_idx), (i32 imm:$dst_idx),
           I32:$dst, I32:$src, I32:$len
         )],
         "memory.copy\t$src_idx, $dst_idx, $dst, $src, $len",
         "memory.copy\t$src_idx, $dst_idx", 0x0a>;

let mayStore = 1 in
defm MEMORY_FILL_A32 :
  BULK_I<(outs), (ins i32imm_op:$idx, I32:$dst, I32:$value, I32:$size),
         (outs), (ins i32imm_op:$idx),
         [(wasm_memory_fill (i32 imm:$idx), I32:$dst, I32:$value, I32:$size)],
         "memory.fill\t$idx, $dst, $value, $size",
         "memory.fill\t$idx", 0x0b>;

let mayLoad = 1, mayStore = 1 in
defm MEMORY_COPY_A64 :
  BULK_I<(outs), (ins i32imm_op:$src_idx, i32imm_op:$dst_idx,
                      I64:$dst, I64:$src, I64:$len),
         (outs), (ins i32imm_op:$src_idx, i32imm_op:$dst_idx),
         [(wasm_memory_copy (i32 imm:$src_idx), (i32 imm:$dst_idx),
           I64:$dst, I64:$src, I64:$len
         )],
         "memory.copy\t$src_idx, $dst_idx, $dst, $src, $len",
         "memory.copy\t$src_idx, $dst_idx", 0x0a>;

let mayStore = 1 in
defm MEMORY_FILL_A64 :
  BULK_I<(outs), (ins i32imm_op:$idx, I64:$dst, I32:$value, I64:$size),
         (outs), (ins i32imm_op:$idx),
         [(wasm_memory_fill (i32 imm:$idx), I64:$dst, I32:$value, I64:$size)],
         "memory.fill\t$idx, $dst, $value, $size",
         "memory.fill\t$idx", 0x0b>;

let usesCustomInserter = 1, isCodeGenOnly = 1, mayLoad = 1, mayStore = 1 in
defm MEMCPY_A32 : I<(outs), (ins i32imm_op:$src_idx, i32imm_op:$dst_idx,
                                 I32:$dst, I32:$src, I32:$len),
                    (outs), (ins i32imm_op:$src_idx, i32imm_op:$dst_idx),
                    [(wasm_memcpy (i32 imm:$src_idx), (i32 imm:$dst_idx),
                      I32:$dst, I32:$src, I32:$len
                    )],
                    "", "", 0>,
                  Requires<[HasBulkMemory]>;

let usesCustomInserter = 1, isCodeGenOnly = 1, mayStore = 1 in
defm MEMSET_A32 : I<(outs), (ins i32imm_op:$idx, I32:$dst, I32:$value, I32:$size),
                    (outs), (ins i32imm_op:$idx),
                    [(wasm_memset (i32 imm:$idx), I32:$dst, I32:$value, I32:$size)],
                    "", "", 0>,
                  Requires<[HasBulkMemory]>;

let usesCustomInserter = 1, isCodeGenOnly = 1, mayLoad = 1, mayStore = 1 in
defm MEMCPY_A64 : I<(outs), (ins i32imm_op:$src_idx, i32imm_op:$dst_idx,
                                 I64:$dst, I64:$src, I64:$len),
                    (outs), (ins i32imm_op:$src_idx, i32imm_op:$dst_idx),
                    [(wasm_memcpy (i32 imm:$src_idx), (i32 imm:$dst_idx),
                      I64:$dst, I64:$src, I64:$len
                    )],
                    "", "", 0>,
                  Requires<[HasBulkMemory]>;

let usesCustomInserter = 1, isCodeGenOnly = 1, mayStore = 1 in
defm MEMSET_A64 : I<(outs), (ins i32imm_op:$idx, I64:$dst, I32:$value, I64:$size),
                    (outs), (ins i32imm_op:$idx),
                    [(wasm_memset (i32 imm:$idx), I64:$dst, I32:$value, I64:$size)],
                    "", "", 0>,
                  Requires<[HasBulkMemory]>;
