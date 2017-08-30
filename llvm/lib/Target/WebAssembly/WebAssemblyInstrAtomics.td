// WebAssemblyInstrAtomics.td-WebAssembly Atomic codegen support-*- tablegen -*-
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief WebAssembly Atomic operand code-gen constructs.
///
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Atomic loads
//===----------------------------------------------------------------------===//

let Defs = [ARGUMENTS] in {
// TODO: add the rest of the atomic loads
// TODO: factor out 0xfe atomic prefix?
def ATOMIC_LOAD_I32 : ATOMIC_I<(outs I32:$dst),
                        (ins P2Align:$p2align, offset32_op:$off, I32:$addr),
                        [], "i32.atomic.load\t$dst, ${off}(${addr})${p2align}",
                        0xfe10>;
} // Defs = [ARGUMENTS]

// Select loads with no constant offset.
let Predicates = [HasAtomics] in {
def : Pat<(i32 (atomic_load I32:$addr)), (ATOMIC_LOAD_I32 0, 0, $addr)>;
}

//===----------------------------------------------------------------------===//
// Atomic stores
//===----------------------------------------------------------------------===//

// TODO: add atomic stores here...

//===----------------------------------------------------------------------===//
// Low-level exclusive operations
//===----------------------------------------------------------------------===//

// TODO: add exclusive operations here...

// Load-exclusives.

// Store-exclusives.

// Store-release-exclusives.

// And clear exclusive.

