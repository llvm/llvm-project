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
def ATOMIC_LOAD_I32 : WebAssemblyLoad<I32, "i32.atomic.load", 0xfe10>;
def ATOMIC_LOAD_I64 : WebAssemblyLoad<I64, "i64.atomic.load", 0xfe11>;
} // Defs = [ARGUMENTS]

// Select loads with no constant offset.
let Predicates = [HasAtomics] in {
def : LoadPatNoOffset<i32, atomic_load_32, ATOMIC_LOAD_I32>;
def : LoadPatNoOffset<i64, atomic_load_64, ATOMIC_LOAD_I64>;

// Select loads with a constant offset.

// Pattern with address + immediate offset
def : LoadPatImmOff<i32, atomic_load_32, regPlusImm, ATOMIC_LOAD_I32>;
def : LoadPatImmOff<i64, atomic_load_64, regPlusImm, ATOMIC_LOAD_I64>;
def : LoadPatImmOff<i32, atomic_load_32, or_is_add, ATOMIC_LOAD_I32>;
def : LoadPatImmOff<i64, atomic_load_64, or_is_add, ATOMIC_LOAD_I64>;

def : LoadPatGlobalAddr<i32, atomic_load_32, ATOMIC_LOAD_I32>;
def : LoadPatGlobalAddr<i64, atomic_load_64, ATOMIC_LOAD_I64>;

def : LoadPatExternalSym<i32, atomic_load_32, ATOMIC_LOAD_I32>;
def : LoadPatExternalSym<i64, atomic_load_64, ATOMIC_LOAD_I64>;


// Select loads with just a constant offset.
def : LoadPatOffsetOnly<i32, atomic_load_32, ATOMIC_LOAD_I32>;
def : LoadPatOffsetOnly<i64, atomic_load_64, ATOMIC_LOAD_I64>;

def : LoadPatGlobalAddrOffOnly<i32, atomic_load_32, ATOMIC_LOAD_I32>;
def : LoadPatGlobalAddrOffOnly<i64, atomic_load_64, ATOMIC_LOAD_I64>;

def : LoadPatExternSymOffOnly<i32, atomic_load_32, ATOMIC_LOAD_I32>;
def : LoadPatExternSymOffOnly<i64, atomic_load_64, ATOMIC_LOAD_I64>;

} // Predicates = [HasAtomics]

// Extending loads. Note that there are only zero-extending atomic loads, no
// sign-extending loads.
let Defs = [ARGUMENTS] in {
def ATOMIC_LOAD8_U_I32 : WebAssemblyLoad<I32, "i32.atomic.load8_u", 0xfe12>;
def ATOMIC_LOAD16_U_I32 : WebAssemblyLoad<I32, "i32.atomic.load16_u", 0xfe13>;
def ATOMIC_LOAD8_U_I64 : WebAssemblyLoad<I64, "i64.atomic.load8_u", 0xfe14>;
def ATOMIC_LOAD16_U_I64 : WebAssemblyLoad<I64, "i64.atomic.load16_u", 0xfe15>;
def ATOMIC_LOAD32_U_I64 : WebAssemblyLoad<I64, "i64.atomic.load32_u", 0xfe16>;
} // Defs = [ARGUMENTS]

// Fragments for exending loads. These are different from regular loads because
// the SDNodes are derived from AtomicSDNode rather than LoadSDNode and
// therefore don't have the extension type field. So instead of matching that,
// we match the patterns that the type legalizer expands them to.

// We directly match zext patterns and select the zext atomic loads.
// i32 (zext (i8 (atomic_load_8))) gets legalized to
// i32 (and (i32 (atomic_load_8)), 255)
// These can be selected to a single zero-extending atomic load instruction.
def zext_aload_8 : PatFrag<(ops node:$addr),
                           (and (i32 (atomic_load_8 node:$addr)), 255)>;
def zext_aload_16 : PatFrag<(ops node:$addr),
                            (and (i32 (atomic_load_16 node:$addr)), 65535)>;
// Unlike regular loads, extension to i64 is handled differently than i32.
// i64 (zext (i8 (atomic_load_8))) gets legalized to
// i64 (and (i64 (anyext (i32 (atomic_load_8)))), 255)
def zext_aload_8_64 :
  PatFrag<(ops node:$addr),
          (and (i64 (anyext (i32 (atomic_load_8 node:$addr)))), 255)>;
def zext_aload_16_64 :
  PatFrag<(ops node:$addr),
          (and (i64 (anyext (i32 (atomic_load_16 node:$addr)))), 65535)>;
def zext_aload_32_64 :
  PatFrag<(ops node:$addr),
          (zext (i32 (atomic_load node:$addr)))>;

// We don't have single sext atomic load instructions. So for sext loads, we
// match bare subword loads (for 32-bit results) and anyext loads (for 64-bit
// results) and select a zext load; the next instruction will be sext_inreg
// which is selected by itself.
def anyext_aload_8_64 :
  PatFrag<(ops node:$addr), (anyext (i32 (atomic_load_8 node:$addr)))>;
def anyext_aload_16_64 :
  PatFrag<(ops node:$addr), (anyext (i32 (atomic_load_16 node:$addr)))>;

let Predicates = [HasAtomics] in {
// Select zero-extending loads with no constant offset.
def : LoadPatNoOffset<i32, zext_aload_8, ATOMIC_LOAD8_U_I32>;
def : LoadPatNoOffset<i32, zext_aload_16, ATOMIC_LOAD16_U_I32>;
def : LoadPatNoOffset<i64, zext_aload_8_64, ATOMIC_LOAD8_U_I64>;
def : LoadPatNoOffset<i64, zext_aload_16_64, ATOMIC_LOAD16_U_I64>;
def : LoadPatNoOffset<i64, zext_aload_32_64, ATOMIC_LOAD32_U_I64>;

// Select sign-extending loads with no constant offset
def : LoadPatNoOffset<i32, atomic_load_8, ATOMIC_LOAD8_U_I32>;
def : LoadPatNoOffset<i32, atomic_load_16, ATOMIC_LOAD16_U_I32>;
def : LoadPatNoOffset<i64, anyext_aload_8_64, ATOMIC_LOAD8_U_I64>;
def : LoadPatNoOffset<i64, anyext_aload_16_64, ATOMIC_LOAD16_U_I64>;
// 32->64 sext load gets selected as i32.atomic.load, i64.extend_s/i64


// Zero-extending loads with constant offset
def : LoadPatImmOff<i32, zext_aload_8, regPlusImm, ATOMIC_LOAD8_U_I32>;
def : LoadPatImmOff<i32, zext_aload_16, regPlusImm, ATOMIC_LOAD16_U_I32>;
def : LoadPatImmOff<i32, zext_aload_8, or_is_add, ATOMIC_LOAD8_U_I32>;
def : LoadPatImmOff<i32, zext_aload_16, or_is_add, ATOMIC_LOAD16_U_I32>;
def : LoadPatImmOff<i64, zext_aload_8_64, regPlusImm, ATOMIC_LOAD8_U_I64>;
def : LoadPatImmOff<i64, zext_aload_16_64, regPlusImm, ATOMIC_LOAD16_U_I64>;
def : LoadPatImmOff<i64, zext_aload_32_64, regPlusImm, ATOMIC_LOAD32_U_I64>;
def : LoadPatImmOff<i64, zext_aload_8_64, or_is_add, ATOMIC_LOAD8_U_I64>;
def : LoadPatImmOff<i64, zext_aload_16_64, or_is_add, ATOMIC_LOAD16_U_I64>;
def : LoadPatImmOff<i64, zext_aload_32_64, or_is_add, ATOMIC_LOAD32_U_I64>;

// Sign-extending loads with constant offset
def : LoadPatImmOff<i32, atomic_load_8, regPlusImm, ATOMIC_LOAD8_U_I32>;
def : LoadPatImmOff<i32, atomic_load_16, regPlusImm, ATOMIC_LOAD16_U_I32>;
def : LoadPatImmOff<i32, atomic_load_8, or_is_add, ATOMIC_LOAD8_U_I32>;
def : LoadPatImmOff<i32, atomic_load_16, or_is_add, ATOMIC_LOAD16_U_I32>;
def : LoadPatImmOff<i64, anyext_aload_8_64, regPlusImm, ATOMIC_LOAD8_U_I64>;
def : LoadPatImmOff<i64, anyext_aload_16_64, regPlusImm, ATOMIC_LOAD16_U_I64>;
def : LoadPatImmOff<i64, anyext_aload_8_64, or_is_add, ATOMIC_LOAD8_U_I64>;
def : LoadPatImmOff<i64, anyext_aload_16_64, or_is_add, ATOMIC_LOAD16_U_I64>;
// No 32->64 patterns, just use i32.atomic.load and i64.extend_s/i64

def : LoadPatGlobalAddr<i32, zext_aload_8, ATOMIC_LOAD8_U_I32>;
def : LoadPatGlobalAddr<i32, zext_aload_16, ATOMIC_LOAD16_U_I32>;
def : LoadPatGlobalAddr<i64, zext_aload_8_64, ATOMIC_LOAD8_U_I64>;
def : LoadPatGlobalAddr<i64, zext_aload_16_64, ATOMIC_LOAD16_U_I64>;
def : LoadPatGlobalAddr<i64, zext_aload_32_64, ATOMIC_LOAD32_U_I64>;
def : LoadPatGlobalAddr<i32, atomic_load_8, ATOMIC_LOAD8_U_I32>;
def : LoadPatGlobalAddr<i32, atomic_load_16, ATOMIC_LOAD16_U_I32>;
def : LoadPatGlobalAddr<i64, anyext_aload_8_64, ATOMIC_LOAD8_U_I64>;
def : LoadPatGlobalAddr<i64, anyext_aload_16_64, ATOMIC_LOAD16_U_I64>;

def : LoadPatExternalSym<i32, zext_aload_8, ATOMIC_LOAD8_U_I32>;
def : LoadPatExternalSym<i32, zext_aload_16, ATOMIC_LOAD16_U_I32>;
def : LoadPatExternalSym<i64, zext_aload_8_64, ATOMIC_LOAD8_U_I64>;
def : LoadPatExternalSym<i64, zext_aload_16_64, ATOMIC_LOAD16_U_I64>;
def : LoadPatExternalSym<i64, zext_aload_32_64, ATOMIC_LOAD32_U_I64>;
def : LoadPatExternalSym<i32, atomic_load_8, ATOMIC_LOAD8_U_I32>;
def : LoadPatExternalSym<i32, atomic_load_16, ATOMIC_LOAD16_U_I32>;
def : LoadPatExternalSym<i64, anyext_aload_8_64, ATOMIC_LOAD8_U_I64>;
def : LoadPatExternalSym<i64, anyext_aload_16_64, ATOMIC_LOAD16_U_I64>;


// Extending loads with just a constant offset
def : LoadPatOffsetOnly<i32, zext_aload_8, ATOMIC_LOAD8_U_I32>;
def : LoadPatOffsetOnly<i32, zext_aload_16, ATOMIC_LOAD16_U_I32>;
def : LoadPatOffsetOnly<i64, zext_aload_8_64, ATOMIC_LOAD8_U_I64>;
def : LoadPatOffsetOnly<i64, zext_aload_16_64, ATOMIC_LOAD16_U_I64>;
def : LoadPatOffsetOnly<i64, zext_aload_32_64, ATOMIC_LOAD32_U_I64>;
def : LoadPatOffsetOnly<i32, atomic_load_8, ATOMIC_LOAD8_U_I32>;
def : LoadPatOffsetOnly<i32, atomic_load_16, ATOMIC_LOAD16_U_I32>;
def : LoadPatOffsetOnly<i64, anyext_aload_8_64, ATOMIC_LOAD8_U_I64>;
def : LoadPatOffsetOnly<i64, anyext_aload_16_64, ATOMIC_LOAD16_U_I64>;

def : LoadPatGlobalAddrOffOnly<i32, zext_aload_8, ATOMIC_LOAD8_U_I32>;
def : LoadPatGlobalAddrOffOnly<i32, zext_aload_16, ATOMIC_LOAD16_U_I32>;
def : LoadPatGlobalAddrOffOnly<i64, zext_aload_8_64, ATOMIC_LOAD8_U_I64>;
def : LoadPatGlobalAddrOffOnly<i64, zext_aload_16_64, ATOMIC_LOAD16_U_I64>;
def : LoadPatGlobalAddrOffOnly<i64, zext_aload_32_64, ATOMIC_LOAD32_U_I64>;
def : LoadPatGlobalAddrOffOnly<i32, atomic_load_8, ATOMIC_LOAD8_U_I32>;
def : LoadPatGlobalAddrOffOnly<i32, atomic_load_16, ATOMIC_LOAD16_U_I32>;
def : LoadPatGlobalAddrOffOnly<i64, anyext_aload_8_64, ATOMIC_LOAD8_U_I64>;
def : LoadPatGlobalAddrOffOnly<i64, anyext_aload_16_64, ATOMIC_LOAD16_U_I64>;

def : LoadPatExternSymOffOnly<i32, zext_aload_8, ATOMIC_LOAD8_U_I32>;
def : LoadPatExternSymOffOnly<i32, zext_aload_16, ATOMIC_LOAD16_U_I32>;
def : LoadPatExternSymOffOnly<i64, zext_aload_8_64, ATOMIC_LOAD8_U_I64>;
def : LoadPatExternSymOffOnly<i64, zext_aload_16_64, ATOMIC_LOAD16_U_I64>;
def : LoadPatExternSymOffOnly<i64, zext_aload_32_64, ATOMIC_LOAD32_U_I64>;
def : LoadPatExternSymOffOnly<i32, atomic_load_8, ATOMIC_LOAD8_U_I32>;
def : LoadPatExternSymOffOnly<i32, atomic_load_16, ATOMIC_LOAD16_U_I32>;
def : LoadPatExternSymOffOnly<i64, anyext_aload_8_64, ATOMIC_LOAD8_U_I64>;
def : LoadPatExternSymOffOnly<i64, anyext_aload_16_64, ATOMIC_LOAD16_U_I64>;


} // Predicates = [HasAtomics]

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

