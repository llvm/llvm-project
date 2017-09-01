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
def ATOMIC_LOAD_I32 : CLoadI32<"i32.atomic.load", 0xfe10>;
def ATOMIC_LOAD_I64 : CLoadI64<"i64.atomic.load", 0xfe11>;
} // Defs = [ARGUMENTS]

// Select loads with no constant offset.
let Predicates = [HasAtomics] in {
class ALoadPatNoOffset<ValueType ty, SDNode node, I inst> :
  Pat<(ty (node I32:$addr)), (inst 0, 0, $addr)>;
def : ALoadPatNoOffset<i32, atomic_load, ATOMIC_LOAD_I32>;
def : ALoadPatNoOffset<i64, atomic_load, ATOMIC_LOAD_I64>;

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

