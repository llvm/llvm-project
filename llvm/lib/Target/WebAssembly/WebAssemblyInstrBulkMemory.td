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
def wasm_memcpy_t : SDTypeProfile<0, 3,
  [SDTCisPtrTy<0>, SDTCisPtrTy<1>, SDTCisInt<2>]
>;
def wasm_memcpy : SDNode<"WebAssemblyISD::MEMORY_COPY", wasm_memcpy_t,
                         [SDNPHasChain, SDNPMayLoad, SDNPMayStore]>;

//===----------------------------------------------------------------------===//
// memory.copy
//===----------------------------------------------------------------------===//

let mayLoad = 1, mayStore = 1 in
defm MEMORY_COPY : BULK_I<(outs), (ins I32:$dst, I32:$src, I32:$len),
                          (outs), (ins),
                          [(wasm_memcpy I32:$dst, I32:$src, I32:$len)],
                          "memory.copy\t$dst, $src, $len",
                          "memory.copy", 0x0a>;
