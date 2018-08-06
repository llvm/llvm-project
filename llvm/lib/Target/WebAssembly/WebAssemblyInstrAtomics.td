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
/// WebAssembly Atomic operand code-gen constructs.
///
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Atomic loads
//===----------------------------------------------------------------------===//

let Defs = [ARGUMENTS] in {
defm ATOMIC_LOAD_I32 : WebAssemblyLoad<I32, "i32.atomic.load", 0xfe10>;
defm ATOMIC_LOAD_I64 : WebAssemblyLoad<I64, "i64.atomic.load", 0xfe11>;
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
defm ATOMIC_LOAD8_U_I32 : WebAssemblyLoad<I32, "i32.atomic.load8_u", 0xfe12>;
defm ATOMIC_LOAD16_U_I32 : WebAssemblyLoad<I32, "i32.atomic.load16_u", 0xfe13>;
defm ATOMIC_LOAD8_U_I64 : WebAssemblyLoad<I64, "i64.atomic.load8_u", 0xfe14>;
defm ATOMIC_LOAD16_U_I64 : WebAssemblyLoad<I64, "i64.atomic.load16_u", 0xfe15>;
defm ATOMIC_LOAD32_U_I64 : WebAssemblyLoad<I64, "i64.atomic.load32_u", 0xfe16>;
} // Defs = [ARGUMENTS]

// Fragments for extending loads. These are different from regular loads because
// the SDNodes are derived from AtomicSDNode rather than LoadSDNode and
// therefore don't have the extension type field. So instead of matching that,
// we match the patterns that the type legalizer expands them to.

// We directly match zext patterns and select the zext atomic loads.
// i32 (zext (i8 (atomic_load_8))) gets legalized to
// i32 (and (i32 (atomic_load_8)), 255)
// These can be selected to a single zero-extending atomic load instruction.
def zext_aload_8_32 :
  PatFrag<(ops node:$addr), (and (i32 (atomic_load_8 node:$addr)), 255)>;
def zext_aload_16_32 :
  PatFrag<(ops node:$addr), (and (i32 (atomic_load_16 node:$addr)), 65535)>;
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
def sext_aload_8_64 :
  PatFrag<(ops node:$addr), (anyext (i32 (atomic_load_8 node:$addr)))>;
def sext_aload_16_64 :
  PatFrag<(ops node:$addr), (anyext (i32 (atomic_load_16 node:$addr)))>;

let Predicates = [HasAtomics] in {
// Select zero-extending loads with no constant offset.
def : LoadPatNoOffset<i32, zext_aload_8_32, ATOMIC_LOAD8_U_I32>;
def : LoadPatNoOffset<i32, zext_aload_16_32, ATOMIC_LOAD16_U_I32>;
def : LoadPatNoOffset<i64, zext_aload_8_64, ATOMIC_LOAD8_U_I64>;
def : LoadPatNoOffset<i64, zext_aload_16_64, ATOMIC_LOAD16_U_I64>;
def : LoadPatNoOffset<i64, zext_aload_32_64, ATOMIC_LOAD32_U_I64>;

// Select sign-extending loads with no constant offset
def : LoadPatNoOffset<i32, atomic_load_8, ATOMIC_LOAD8_U_I32>;
def : LoadPatNoOffset<i32, atomic_load_16, ATOMIC_LOAD16_U_I32>;
def : LoadPatNoOffset<i64, sext_aload_8_64, ATOMIC_LOAD8_U_I64>;
def : LoadPatNoOffset<i64, sext_aload_16_64, ATOMIC_LOAD16_U_I64>;
// 32->64 sext load gets selected as i32.atomic.load, i64.extend_s/i32

// Zero-extending loads with constant offset
def : LoadPatImmOff<i32, zext_aload_8_32, regPlusImm, ATOMIC_LOAD8_U_I32>;
def : LoadPatImmOff<i32, zext_aload_16_32, regPlusImm, ATOMIC_LOAD16_U_I32>;
def : LoadPatImmOff<i32, zext_aload_8_32, or_is_add, ATOMIC_LOAD8_U_I32>;
def : LoadPatImmOff<i32, zext_aload_16_32, or_is_add, ATOMIC_LOAD16_U_I32>;
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
def : LoadPatImmOff<i64, sext_aload_8_64, regPlusImm, ATOMIC_LOAD8_U_I64>;
def : LoadPatImmOff<i64, sext_aload_16_64, regPlusImm, ATOMIC_LOAD16_U_I64>;
def : LoadPatImmOff<i64, sext_aload_8_64, or_is_add, ATOMIC_LOAD8_U_I64>;
def : LoadPatImmOff<i64, sext_aload_16_64, or_is_add, ATOMIC_LOAD16_U_I64>;
// No 32->64 patterns, just use i32.atomic.load and i64.extend_s/i64

def : LoadPatGlobalAddr<i32, zext_aload_8_32, ATOMIC_LOAD8_U_I32>;
def : LoadPatGlobalAddr<i32, zext_aload_16_32, ATOMIC_LOAD16_U_I32>;
def : LoadPatGlobalAddr<i64, zext_aload_8_64, ATOMIC_LOAD8_U_I64>;
def : LoadPatGlobalAddr<i64, zext_aload_16_64, ATOMIC_LOAD16_U_I64>;
def : LoadPatGlobalAddr<i64, zext_aload_32_64, ATOMIC_LOAD32_U_I64>;
def : LoadPatGlobalAddr<i32, atomic_load_8, ATOMIC_LOAD8_U_I32>;
def : LoadPatGlobalAddr<i32, atomic_load_16, ATOMIC_LOAD16_U_I32>;
def : LoadPatGlobalAddr<i64, sext_aload_8_64, ATOMIC_LOAD8_U_I64>;
def : LoadPatGlobalAddr<i64, sext_aload_16_64, ATOMIC_LOAD16_U_I64>;

def : LoadPatExternalSym<i32, zext_aload_8_32, ATOMIC_LOAD8_U_I32>;
def : LoadPatExternalSym<i32, zext_aload_16_32, ATOMIC_LOAD16_U_I32>;
def : LoadPatExternalSym<i64, zext_aload_8_64, ATOMIC_LOAD8_U_I64>;
def : LoadPatExternalSym<i64, zext_aload_16_64, ATOMIC_LOAD16_U_I64>;
def : LoadPatExternalSym<i64, zext_aload_32_64, ATOMIC_LOAD32_U_I64>;
def : LoadPatExternalSym<i32, atomic_load_8, ATOMIC_LOAD8_U_I32>;
def : LoadPatExternalSym<i32, atomic_load_16, ATOMIC_LOAD16_U_I32>;
def : LoadPatExternalSym<i64, sext_aload_8_64, ATOMIC_LOAD8_U_I64>;
def : LoadPatExternalSym<i64, sext_aload_16_64, ATOMIC_LOAD16_U_I64>;

// Extending loads with just a constant offset
def : LoadPatOffsetOnly<i32, zext_aload_8_32, ATOMIC_LOAD8_U_I32>;
def : LoadPatOffsetOnly<i32, zext_aload_16_32, ATOMIC_LOAD16_U_I32>;
def : LoadPatOffsetOnly<i64, zext_aload_8_64, ATOMIC_LOAD8_U_I64>;
def : LoadPatOffsetOnly<i64, zext_aload_16_64, ATOMIC_LOAD16_U_I64>;
def : LoadPatOffsetOnly<i64, zext_aload_32_64, ATOMIC_LOAD32_U_I64>;
def : LoadPatOffsetOnly<i32, atomic_load_8, ATOMIC_LOAD8_U_I32>;
def : LoadPatOffsetOnly<i32, atomic_load_16, ATOMIC_LOAD16_U_I32>;
def : LoadPatOffsetOnly<i64, sext_aload_8_64, ATOMIC_LOAD8_U_I64>;
def : LoadPatOffsetOnly<i64, sext_aload_16_64, ATOMIC_LOAD16_U_I64>;

def : LoadPatGlobalAddrOffOnly<i32, zext_aload_8_32, ATOMIC_LOAD8_U_I32>;
def : LoadPatGlobalAddrOffOnly<i32, zext_aload_16_32, ATOMIC_LOAD16_U_I32>;
def : LoadPatGlobalAddrOffOnly<i64, zext_aload_8_64, ATOMIC_LOAD8_U_I64>;
def : LoadPatGlobalAddrOffOnly<i64, zext_aload_16_64, ATOMIC_LOAD16_U_I64>;
def : LoadPatGlobalAddrOffOnly<i64, zext_aload_32_64, ATOMIC_LOAD32_U_I64>;
def : LoadPatGlobalAddrOffOnly<i32, atomic_load_8, ATOMIC_LOAD8_U_I32>;
def : LoadPatGlobalAddrOffOnly<i32, atomic_load_16, ATOMIC_LOAD16_U_I32>;
def : LoadPatGlobalAddrOffOnly<i64, sext_aload_8_64, ATOMIC_LOAD8_U_I64>;
def : LoadPatGlobalAddrOffOnly<i64, sext_aload_16_64, ATOMIC_LOAD16_U_I64>;

def : LoadPatExternSymOffOnly<i32, zext_aload_8_32, ATOMIC_LOAD8_U_I32>;
def : LoadPatExternSymOffOnly<i32, zext_aload_16_32, ATOMIC_LOAD16_U_I32>;
def : LoadPatExternSymOffOnly<i64, zext_aload_8_64, ATOMIC_LOAD8_U_I64>;
def : LoadPatExternSymOffOnly<i64, zext_aload_16_64, ATOMIC_LOAD16_U_I64>;
def : LoadPatExternSymOffOnly<i64, zext_aload_32_64, ATOMIC_LOAD32_U_I64>;
def : LoadPatExternSymOffOnly<i32, atomic_load_8, ATOMIC_LOAD8_U_I32>;
def : LoadPatExternSymOffOnly<i32, atomic_load_16, ATOMIC_LOAD16_U_I32>;
def : LoadPatExternSymOffOnly<i64, sext_aload_8_64, ATOMIC_LOAD8_U_I64>;
def : LoadPatExternSymOffOnly<i64, sext_aload_16_64, ATOMIC_LOAD16_U_I64>;

} // Predicates = [HasAtomics]

//===----------------------------------------------------------------------===//
// Atomic stores
//===----------------------------------------------------------------------===//

let Defs = [ARGUMENTS] in {
defm ATOMIC_STORE_I32 : WebAssemblyStore<I32, "i32.atomic.store", 0xfe17>;
defm ATOMIC_STORE_I64 : WebAssemblyStore<I64, "i64.atomic.store", 0xfe18>;
} // Defs = [ARGUMENTS]

// We need an 'atomic' version of store patterns because store and atomic_store
// nodes have different operand orders:
// store: (store $val, $ptr)
// atomic_store: (store $ptr, $val)

let Predicates = [HasAtomics] in {

// Select stores with no constant offset.
class AStorePatNoOffset<ValueType ty, PatFrag kind, NI inst> :
  Pat<(kind I32:$addr, ty:$val), (inst 0, 0, I32:$addr, ty:$val)>;
def : AStorePatNoOffset<i32, atomic_store_32, ATOMIC_STORE_I32>;
def : AStorePatNoOffset<i64, atomic_store_64, ATOMIC_STORE_I64>;

// Select stores with a constant offset.

// Pattern with address + immediate offset
class AStorePatImmOff<ValueType ty, PatFrag kind, PatFrag operand, NI inst> :
  Pat<(kind (operand I32:$addr, imm:$off), ty:$val),
      (inst 0, imm:$off, I32:$addr, ty:$val)>;
def : AStorePatImmOff<i32, atomic_store_32, regPlusImm, ATOMIC_STORE_I32>;
def : AStorePatImmOff<i64, atomic_store_64, regPlusImm, ATOMIC_STORE_I64>;
def : AStorePatImmOff<i32, atomic_store_32, or_is_add, ATOMIC_STORE_I32>;
def : AStorePatImmOff<i64, atomic_store_64, or_is_add, ATOMIC_STORE_I64>;

class AStorePatGlobalAddr<ValueType ty, PatFrag kind, NI inst> :
  Pat<(kind (regPlusGA I32:$addr, (WebAssemblywrapper tglobaladdr:$off)),
            ty:$val),
      (inst 0, tglobaladdr:$off, I32:$addr, ty:$val)>;
def : AStorePatGlobalAddr<i32, atomic_store_32, ATOMIC_STORE_I32>;
def : AStorePatGlobalAddr<i64, atomic_store_64, ATOMIC_STORE_I64>;

class AStorePatExternalSym<ValueType ty, PatFrag kind, NI inst> :
  Pat<(kind (add I32:$addr, (WebAssemblywrapper texternalsym:$off)), ty:$val),
      (inst 0, texternalsym:$off, I32:$addr, ty:$val)>;
def : AStorePatExternalSym<i32, atomic_store_32, ATOMIC_STORE_I32>;
def : AStorePatExternalSym<i64, atomic_store_64, ATOMIC_STORE_I64>;

// Select stores with just a constant offset.
class AStorePatOffsetOnly<ValueType ty, PatFrag kind, NI inst> :
  Pat<(kind imm:$off, ty:$val), (inst 0, imm:$off, (CONST_I32 0), ty:$val)>;
def : AStorePatOffsetOnly<i32, atomic_store_32, ATOMIC_STORE_I32>;
def : AStorePatOffsetOnly<i64, atomic_store_64, ATOMIC_STORE_I64>;

class AStorePatGlobalAddrOffOnly<ValueType ty, PatFrag kind, NI inst> :
  Pat<(kind (WebAssemblywrapper tglobaladdr:$off), ty:$val),
      (inst 0, tglobaladdr:$off, (CONST_I32 0), ty:$val)>;
def : AStorePatGlobalAddrOffOnly<i32, atomic_store_32, ATOMIC_STORE_I32>;
def : AStorePatGlobalAddrOffOnly<i64, atomic_store_64, ATOMIC_STORE_I64>;

class AStorePatExternSymOffOnly<ValueType ty, PatFrag kind, NI inst> :
  Pat<(kind (WebAssemblywrapper texternalsym:$off), ty:$val),
      (inst 0, texternalsym:$off, (CONST_I32 0), ty:$val)>;
def : AStorePatExternSymOffOnly<i32, atomic_store_32, ATOMIC_STORE_I32>;
def : AStorePatExternSymOffOnly<i64, atomic_store_64, ATOMIC_STORE_I64>;

} // Predicates = [HasAtomics]

// Truncating stores.
let Defs = [ARGUMENTS] in {
defm ATOMIC_STORE8_I32 : WebAssemblyStore<I32, "i32.atomic.store8", 0xfe19>;
defm ATOMIC_STORE16_I32 : WebAssemblyStore<I32, "i32.atomic.store16", 0xfe1a>;
defm ATOMIC_STORE8_I64 : WebAssemblyStore<I64, "i64.atomic.store8", 0xfe1b>;
defm ATOMIC_STORE16_I64 : WebAssemblyStore<I64, "i64.atomic.store16", 0xfe1c>;
defm ATOMIC_STORE32_I64 : WebAssemblyStore<I64, "i64.atomic.store32", 0xfe1d>;
} // Defs = [ARGUMENTS]

// Fragments for truncating stores.

// We don't have single truncating atomic store instructions. For 32-bit
// instructions, we just need to match bare atomic stores. On the other hand,
// truncating stores from i64 values are once truncated to i32 first.
class trunc_astore_64<PatFrag kind> :
  PatFrag<(ops node:$addr, node:$val),
          (kind node:$addr, (i32 (trunc (i64 node:$val))))>;
def trunc_astore_8_64 : trunc_astore_64<atomic_store_8>;
def trunc_astore_16_64 : trunc_astore_64<atomic_store_16>;
def trunc_astore_32_64 : trunc_astore_64<atomic_store_32>;

let Predicates = [HasAtomics] in {

// Truncating stores with no constant offset
def : AStorePatNoOffset<i32, atomic_store_8, ATOMIC_STORE8_I32>;
def : AStorePatNoOffset<i32, atomic_store_16, ATOMIC_STORE16_I32>;
def : AStorePatNoOffset<i64, trunc_astore_8_64, ATOMIC_STORE8_I64>;
def : AStorePatNoOffset<i64, trunc_astore_16_64, ATOMIC_STORE16_I64>;
def : AStorePatNoOffset<i64, trunc_astore_32_64, ATOMIC_STORE32_I64>;

// Truncating stores with a constant offset
def : AStorePatImmOff<i32, atomic_store_8, regPlusImm, ATOMIC_STORE8_I32>;
def : AStorePatImmOff<i32, atomic_store_16, regPlusImm, ATOMIC_STORE16_I32>;
def : AStorePatImmOff<i64, trunc_astore_8_64, regPlusImm, ATOMIC_STORE8_I64>;
def : AStorePatImmOff<i64, trunc_astore_16_64, regPlusImm, ATOMIC_STORE16_I64>;
def : AStorePatImmOff<i64, trunc_astore_32_64, regPlusImm, ATOMIC_STORE32_I64>;
def : AStorePatImmOff<i32, atomic_store_8, or_is_add, ATOMIC_STORE8_I32>;
def : AStorePatImmOff<i32, atomic_store_16, or_is_add, ATOMIC_STORE16_I32>;
def : AStorePatImmOff<i64, trunc_astore_8_64, or_is_add, ATOMIC_STORE8_I64>;
def : AStorePatImmOff<i64, trunc_astore_16_64, or_is_add, ATOMIC_STORE16_I64>;
def : AStorePatImmOff<i64, trunc_astore_32_64, or_is_add, ATOMIC_STORE32_I64>;

def : AStorePatGlobalAddr<i32, atomic_store_8, ATOMIC_STORE8_I32>;
def : AStorePatGlobalAddr<i32, atomic_store_16, ATOMIC_STORE16_I32>;
def : AStorePatGlobalAddr<i64, trunc_astore_8_64, ATOMIC_STORE8_I64>;
def : AStorePatGlobalAddr<i64, trunc_astore_16_64, ATOMIC_STORE16_I64>;
def : AStorePatGlobalAddr<i64, trunc_astore_32_64, ATOMIC_STORE32_I64>;

def : AStorePatExternalSym<i32, atomic_store_8, ATOMIC_STORE8_I32>;
def : AStorePatExternalSym<i32, atomic_store_16, ATOMIC_STORE16_I32>;
def : AStorePatExternalSym<i64, trunc_astore_8_64, ATOMIC_STORE8_I64>;
def : AStorePatExternalSym<i64, trunc_astore_16_64, ATOMIC_STORE16_I64>;
def : AStorePatExternalSym<i64, trunc_astore_32_64, ATOMIC_STORE32_I64>;

// Truncating stores with just a constant offset
def : AStorePatOffsetOnly<i32, atomic_store_8, ATOMIC_STORE8_I32>;
def : AStorePatOffsetOnly<i32, atomic_store_16, ATOMIC_STORE16_I32>;
def : AStorePatOffsetOnly<i64, trunc_astore_8_64, ATOMIC_STORE8_I64>;
def : AStorePatOffsetOnly<i64, trunc_astore_16_64, ATOMIC_STORE16_I64>;
def : AStorePatOffsetOnly<i64, trunc_astore_32_64, ATOMIC_STORE32_I64>;

def : AStorePatGlobalAddrOffOnly<i32, atomic_store_8, ATOMIC_STORE8_I32>;
def : AStorePatGlobalAddrOffOnly<i32, atomic_store_16, ATOMIC_STORE16_I32>;
def : AStorePatGlobalAddrOffOnly<i64, trunc_astore_8_64, ATOMIC_STORE8_I64>;
def : AStorePatGlobalAddrOffOnly<i64, trunc_astore_16_64, ATOMIC_STORE16_I64>;
def : AStorePatGlobalAddrOffOnly<i64, trunc_astore_32_64, ATOMIC_STORE32_I64>;

def : AStorePatExternSymOffOnly<i32, atomic_store_8, ATOMIC_STORE8_I32>;
def : AStorePatExternSymOffOnly<i32, atomic_store_16, ATOMIC_STORE16_I32>;
def : AStorePatExternSymOffOnly<i64, trunc_astore_8_64, ATOMIC_STORE8_I64>;
def : AStorePatExternSymOffOnly<i64, trunc_astore_16_64, ATOMIC_STORE16_I64>;
def : AStorePatExternSymOffOnly<i64, trunc_astore_32_64, ATOMIC_STORE32_I64>;

} // Predicates = [HasAtomics]

//===----------------------------------------------------------------------===//
// Atomic binary read-modify-writes
//===----------------------------------------------------------------------===//

let Defs = [ARGUMENTS] in {

multiclass WebAssemblyBinRMW<WebAssemblyRegClass rc, string Name, int Opcode> {
  defm "" : I<(outs rc:$dst),
              (ins P2Align:$p2align, offset32_op:$off, I32:$addr, rc:$val),
              (outs), (ins P2Align:$p2align, offset32_op:$off), [],
              !strconcat(Name, "\t$dst, ${off}(${addr})${p2align}, $val"),
              !strconcat(Name, "\t${off}, ${p2align}"), Opcode>;
}

defm ATOMIC_RMW_ADD_I32 : WebAssemblyBinRMW<I32, "i32.atomic.rmw.add", 0xfe1e>;
defm ATOMIC_RMW_ADD_I64 : WebAssemblyBinRMW<I64, "i64.atomic.rmw.add", 0xfe1f>;
defm ATOMIC_RMW8_U_ADD_I32 :
  WebAssemblyBinRMW<I32, "i32.atomic.rmw8_u.add", 0xfe20>;
defm ATOMIC_RMW16_U_ADD_I32 :
  WebAssemblyBinRMW<I32, "i32.atomic.rmw16_u.add", 0xfe21>;
defm ATOMIC_RMW8_U_ADD_I64 :
  WebAssemblyBinRMW<I64, "i64.atomic.rmw8_u.add", 0xfe22>;
defm ATOMIC_RMW16_U_ADD_I64 :
  WebAssemblyBinRMW<I64, "i64.atomic.rmw16_u.add", 0xfe23>;
defm ATOMIC_RMW32_U_ADD_I64 :
  WebAssemblyBinRMW<I64, "i64.atomic.rmw32_u.add", 0xfe24>;

defm ATOMIC_RMW_SUB_I32 : WebAssemblyBinRMW<I32, "i32.atomic.rmw.sub", 0xfe25>;
defm ATOMIC_RMW_SUB_I64 : WebAssemblyBinRMW<I64, "i64.atomic.rmw.sub", 0xfe26>;
defm ATOMIC_RMW8_U_SUB_I32 :
  WebAssemblyBinRMW<I32, "i32.atomic.rmw8_u.sub", 0xfe27>;
defm ATOMIC_RMW16_U_SUB_I32 :
  WebAssemblyBinRMW<I32, "i32.atomic.rmw16_u.sub", 0xfe28>;
defm ATOMIC_RMW8_U_SUB_I64 :
  WebAssemblyBinRMW<I64, "i64.atomic.rmw8_u.sub", 0xfe29>;
defm ATOMIC_RMW16_U_SUB_I64 :
  WebAssemblyBinRMW<I64, "i64.atomic.rmw16_u.sub", 0xfe2a>;
defm ATOMIC_RMW32_U_SUB_I64 :
  WebAssemblyBinRMW<I64, "i64.atomic.rmw32_u.sub", 0xfe2b>;

defm ATOMIC_RMW_AND_I32 : WebAssemblyBinRMW<I32, "i32.atomic.rmw.and", 0xfe2c>;
defm ATOMIC_RMW_AND_I64 : WebAssemblyBinRMW<I64, "i64.atomic.rmw.and", 0xfe2d>;
defm ATOMIC_RMW8_U_AND_I32 :
  WebAssemblyBinRMW<I32, "i32.atomic.rmw8_u.and", 0xfe2e>;
defm ATOMIC_RMW16_U_AND_I32 :
  WebAssemblyBinRMW<I32, "i32.atomic.rmw16_u.and", 0xfe2f>;
defm ATOMIC_RMW8_U_AND_I64 :
  WebAssemblyBinRMW<I64, "i64.atomic.rmw8_u.and", 0xfe30>;
defm ATOMIC_RMW16_U_AND_I64 :
  WebAssemblyBinRMW<I64, "i64.atomic.rmw16_u.and", 0xfe31>;
defm ATOMIC_RMW32_U_AND_I64 :
  WebAssemblyBinRMW<I64, "i64.atomic.rmw32_u.and", 0xfe32>;

defm ATOMIC_RMW_OR_I32 : WebAssemblyBinRMW<I32, "i32.atomic.rmw.or", 0xfe33>;
defm ATOMIC_RMW_OR_I64 : WebAssemblyBinRMW<I64, "i64.atomic.rmw.or", 0xfe34>;
defm ATOMIC_RMW8_U_OR_I32 :
  WebAssemblyBinRMW<I32, "i32.atomic.rmw8_u.or", 0xfe35>;
defm ATOMIC_RMW16_U_OR_I32 :
  WebAssemblyBinRMW<I32, "i32.atomic.rmw16_u.or", 0xfe36>;
defm ATOMIC_RMW8_U_OR_I64 :
  WebAssemblyBinRMW<I64, "i64.atomic.rmw8_u.or", 0xfe37>;
defm ATOMIC_RMW16_U_OR_I64 :
  WebAssemblyBinRMW<I64, "i64.atomic.rmw16_u.or", 0xfe38>;
defm ATOMIC_RMW32_U_OR_I64 :
  WebAssemblyBinRMW<I64, "i64.atomic.rmw32_u.or", 0xfe39>;

defm ATOMIC_RMW_XOR_I32 : WebAssemblyBinRMW<I32, "i32.atomic.rmw.xor", 0xfe3a>;
defm ATOMIC_RMW_XOR_I64 : WebAssemblyBinRMW<I64, "i64.atomic.rmw.xor", 0xfe3b>;
defm ATOMIC_RMW8_U_XOR_I32 :
  WebAssemblyBinRMW<I32, "i32.atomic.rmw8_u.xor", 0xfe3c>;
defm ATOMIC_RMW16_U_XOR_I32 :
  WebAssemblyBinRMW<I32, "i32.atomic.rmw16_u.xor", 0xfe3d>;
defm ATOMIC_RMW8_U_XOR_I64 :
  WebAssemblyBinRMW<I64, "i64.atomic.rmw8_u.xor", 0xfe3e>;
defm ATOMIC_RMW16_U_XOR_I64 :
  WebAssemblyBinRMW<I64, "i64.atomic.rmw16_u.xor", 0xfe3f>;
defm ATOMIC_RMW32_U_XOR_I64 :
  WebAssemblyBinRMW<I64, "i64.atomic.rmw32_u.xor", 0xfe40>;

defm ATOMIC_RMW_XCHG_I32 :
  WebAssemblyBinRMW<I32, "i32.atomic.rmw.xchg", 0xfe41>;
defm ATOMIC_RMW_XCHG_I64 :
  WebAssemblyBinRMW<I64, "i64.atomic.rmw.xchg", 0xfe42>;
defm ATOMIC_RMW8_U_XCHG_I32 :
  WebAssemblyBinRMW<I32, "i32.atomic.rmw8_u.xchg", 0xfe43>;
defm ATOMIC_RMW16_U_XCHG_I32 :
  WebAssemblyBinRMW<I32, "i32.atomic.rmw16_u.xchg", 0xfe44>;
defm ATOMIC_RMW8_U_XCHG_I64 :
  WebAssemblyBinRMW<I64, "i64.atomic.rmw8_u.xchg", 0xfe45>;
defm ATOMIC_RMW16_U_XCHG_I64 :
  WebAssemblyBinRMW<I64, "i64.atomic.rmw16_u.xchg", 0xfe46>;
defm ATOMIC_RMW32_U_XCHG_I64 :
  WebAssemblyBinRMW<I64, "i64.atomic.rmw32_u.xchg", 0xfe47>;
}

// Select binary RMWs with no constant offset.
class BinRMWPatNoOffset<ValueType ty, PatFrag kind, NI inst> :
  Pat<(ty (kind I32:$addr, ty:$val)), (inst 0, 0, I32:$addr, ty:$val)>;

// Select binary RMWs with a constant offset.

// Pattern with address + immediate offset
class BinRMWPatImmOff<ValueType ty, PatFrag kind, PatFrag operand, NI inst> :
  Pat<(ty (kind (operand I32:$addr, imm:$off), ty:$val)),
      (inst 0, imm:$off, I32:$addr, ty:$val)>;

class BinRMWPatGlobalAddr<ValueType ty, PatFrag kind, NI inst> :
  Pat<(ty (kind (regPlusGA I32:$addr, (WebAssemblywrapper tglobaladdr:$off)),
                ty:$val)),
      (inst 0, tglobaladdr:$off, I32:$addr, ty:$val)>;

class BinRMWPatExternalSym<ValueType ty, PatFrag kind, NI inst> :
  Pat<(ty (kind (add I32:$addr, (WebAssemblywrapper texternalsym:$off)),
                ty:$val)),
      (inst 0, texternalsym:$off, I32:$addr, ty:$val)>;

// Select binary RMWs with just a constant offset.
class BinRMWPatOffsetOnly<ValueType ty, PatFrag kind, NI inst> :
  Pat<(ty (kind imm:$off, ty:$val)),
      (inst 0, imm:$off, (CONST_I32 0), ty:$val)>;

class BinRMWPatGlobalAddrOffOnly<ValueType ty, PatFrag kind, NI inst> :
  Pat<(ty (kind (WebAssemblywrapper tglobaladdr:$off), ty:$val)),
      (inst 0, tglobaladdr:$off, (CONST_I32 0), ty:$val)>;

class BinRMWPatExternSymOffOnly<ValueType ty, PatFrag kind, NI inst> :
  Pat<(ty (kind (WebAssemblywrapper texternalsym:$off), ty:$val)),
      (inst 0, texternalsym:$off, (CONST_I32 0), ty:$val)>;

// Patterns for various addressing modes.
multiclass BinRMWPattern<PatFrag rmw_32, PatFrag rmw_64, NI inst_32,
                         NI inst_64> {
  def : BinRMWPatNoOffset<i32, rmw_32, inst_32>;
  def : BinRMWPatNoOffset<i64, rmw_64, inst_64>;

  def : BinRMWPatImmOff<i32, rmw_32, regPlusImm, inst_32>;
  def : BinRMWPatImmOff<i64, rmw_64, regPlusImm, inst_64>;
  def : BinRMWPatImmOff<i32, rmw_32, or_is_add, inst_32>;
  def : BinRMWPatImmOff<i64, rmw_64, or_is_add, inst_64>;

  def : BinRMWPatGlobalAddr<i32, rmw_32, inst_32>;
  def : BinRMWPatGlobalAddr<i64, rmw_64, inst_64>;

  def : BinRMWPatExternalSym<i32, rmw_32, inst_32>;
  def : BinRMWPatExternalSym<i64, rmw_64, inst_64>;

  def : BinRMWPatOffsetOnly<i32, rmw_32, inst_32>;
  def : BinRMWPatOffsetOnly<i64, rmw_64, inst_64>;

  def : BinRMWPatGlobalAddrOffOnly<i32, rmw_32, inst_32>;
  def : BinRMWPatGlobalAddrOffOnly<i64, rmw_64, inst_64>;

  def : BinRMWPatExternSymOffOnly<i32, rmw_32, inst_32>;
  def : BinRMWPatExternSymOffOnly<i64, rmw_64, inst_64>;
}

let Predicates = [HasAtomics] in {
defm : BinRMWPattern<atomic_load_add_32, atomic_load_add_64, ATOMIC_RMW_ADD_I32,
                     ATOMIC_RMW_ADD_I64>;
defm : BinRMWPattern<atomic_load_sub_32, atomic_load_sub_64, ATOMIC_RMW_SUB_I32,
                     ATOMIC_RMW_SUB_I64>;
defm : BinRMWPattern<atomic_load_and_32, atomic_load_and_64, ATOMIC_RMW_AND_I32,
                     ATOMIC_RMW_AND_I64>;
defm : BinRMWPattern<atomic_load_or_32, atomic_load_or_64, ATOMIC_RMW_OR_I32,
                     ATOMIC_RMW_OR_I64>;
defm : BinRMWPattern<atomic_load_xor_32, atomic_load_xor_64, ATOMIC_RMW_XOR_I32,
                     ATOMIC_RMW_XOR_I64>;
defm : BinRMWPattern<atomic_swap_32, atomic_swap_64, ATOMIC_RMW_XCHG_I32,
                     ATOMIC_RMW_XCHG_I64>;
} // Predicates = [HasAtomics]

// Truncating & zero-extending binary RMW patterns.
// These are combined patterns of truncating store patterns and zero-extending
// load patterns above.
class zext_bin_rmw_8_32<PatFrag kind> :
  PatFrag<(ops node:$addr, node:$val),
          (and (i32 (kind node:$addr, node:$val)), 255)>;
class zext_bin_rmw_16_32<PatFrag kind> :
  PatFrag<(ops node:$addr, node:$val),
          (and (i32 (kind node:$addr, node:$val)), 65535)>;
class zext_bin_rmw_8_64<PatFrag kind> :
  PatFrag<(ops node:$addr, node:$val),
    (and (i64 (anyext (i32 (kind node:$addr,
                                 (i32 (trunc (i64 node:$val))))))), 255)>;
class zext_bin_rmw_16_64<PatFrag kind> :
  PatFrag<(ops node:$addr, node:$val),
    (and (i64 (anyext (i32 (kind node:$addr,
                                 (i32 (trunc (i64 node:$val))))))), 65535)>;
class zext_bin_rmw_32_64<PatFrag kind> :
  PatFrag<(ops node:$addr, node:$val),
          (zext (i32 (kind node:$addr, (i32 (trunc (i64 node:$val))))))>;

// Truncating & sign-extending binary RMW patterns.
// These are combined patterns of truncating store patterns and sign-extending
// load patterns above. We match subword RMWs (for 32-bit) and anyext RMWs (for
// 64-bit) and select a zext RMW; the next instruction will be sext_inreg which
// is selected by itself.
class sext_bin_rmw_8_32<PatFrag kind> :
  PatFrag<(ops node:$addr, node:$val), (kind node:$addr, node:$val)>;
class sext_bin_rmw_16_32<PatFrag kind> : sext_bin_rmw_8_32<kind>;
class sext_bin_rmw_8_64<PatFrag kind> :
  PatFrag<(ops node:$addr, node:$val),
          (anyext (i32 (kind node:$addr, (i32 (trunc (i64 node:$val))))))>;
class sext_bin_rmw_16_64<PatFrag kind> : sext_bin_rmw_8_64<kind>;
// 32->64 sext RMW gets selected as i32.atomic.rmw.***, i64.extend_s/i32

// Patterns for various addressing modes for truncating-extending binary RMWs.
multiclass BinRMWTruncExtPattern<
  PatFrag rmw_8, PatFrag rmw_16, PatFrag rmw_32, PatFrag rmw_64,
  NI inst8_32, NI inst16_32, NI inst8_64, NI inst16_64, NI inst32_64> {
  // Truncating-extending binary RMWs with no constant offset
  def : BinRMWPatNoOffset<i32, zext_bin_rmw_8_32<rmw_8>, inst8_32>;
  def : BinRMWPatNoOffset<i32, zext_bin_rmw_16_32<rmw_16>, inst16_32>;
  def : BinRMWPatNoOffset<i64, zext_bin_rmw_8_64<rmw_8>, inst8_64>;
  def : BinRMWPatNoOffset<i64, zext_bin_rmw_16_64<rmw_16>, inst16_64>;
  def : BinRMWPatNoOffset<i64, zext_bin_rmw_32_64<rmw_32>, inst32_64>;

  def : BinRMWPatNoOffset<i32, sext_bin_rmw_8_32<rmw_8>, inst8_32>;
  def : BinRMWPatNoOffset<i32, sext_bin_rmw_16_32<rmw_16>, inst16_32>;
  def : BinRMWPatNoOffset<i64, sext_bin_rmw_8_64<rmw_8>, inst8_64>;
  def : BinRMWPatNoOffset<i64, sext_bin_rmw_16_64<rmw_16>, inst16_64>;

  // Truncating-extending binary RMWs with a constant offset
  def : BinRMWPatImmOff<i32, zext_bin_rmw_8_32<rmw_8>, regPlusImm, inst8_32>;
  def : BinRMWPatImmOff<i32, zext_bin_rmw_16_32<rmw_16>, regPlusImm, inst16_32>;
  def : BinRMWPatImmOff<i64, zext_bin_rmw_8_64<rmw_8>, regPlusImm, inst8_64>;
  def : BinRMWPatImmOff<i64, zext_bin_rmw_16_64<rmw_16>, regPlusImm, inst16_64>;
  def : BinRMWPatImmOff<i64, zext_bin_rmw_32_64<rmw_32>, regPlusImm, inst32_64>;
  def : BinRMWPatImmOff<i32, zext_bin_rmw_8_32<rmw_8>, or_is_add, inst8_32>;
  def : BinRMWPatImmOff<i32, zext_bin_rmw_16_32<rmw_16>, or_is_add, inst16_32>;
  def : BinRMWPatImmOff<i64, zext_bin_rmw_8_64<rmw_8>, or_is_add, inst8_64>;
  def : BinRMWPatImmOff<i64, zext_bin_rmw_16_64<rmw_16>, or_is_add, inst16_64>;
  def : BinRMWPatImmOff<i64, zext_bin_rmw_32_64<rmw_32>, or_is_add, inst32_64>;

  def : BinRMWPatImmOff<i32, sext_bin_rmw_8_32<rmw_8>, regPlusImm, inst8_32>;
  def : BinRMWPatImmOff<i32, sext_bin_rmw_16_32<rmw_16>, regPlusImm, inst16_32>;
  def : BinRMWPatImmOff<i64, sext_bin_rmw_8_64<rmw_8>, regPlusImm, inst8_64>;
  def : BinRMWPatImmOff<i64, sext_bin_rmw_16_64<rmw_16>, regPlusImm, inst16_64>;
  def : BinRMWPatImmOff<i32, sext_bin_rmw_8_32<rmw_8>, or_is_add, inst8_32>;
  def : BinRMWPatImmOff<i32, sext_bin_rmw_16_32<rmw_16>, or_is_add, inst16_32>;
  def : BinRMWPatImmOff<i64, sext_bin_rmw_8_64<rmw_8>, or_is_add, inst8_64>;
  def : BinRMWPatImmOff<i64, sext_bin_rmw_16_64<rmw_16>, or_is_add, inst16_64>;

  def : BinRMWPatGlobalAddr<i32, zext_bin_rmw_8_32<rmw_8>, inst8_32>;
  def : BinRMWPatGlobalAddr<i32, zext_bin_rmw_16_32<rmw_16>, inst16_32>;
  def : BinRMWPatGlobalAddr<i64, zext_bin_rmw_8_64<rmw_8>, inst8_64>;
  def : BinRMWPatGlobalAddr<i64, zext_bin_rmw_16_64<rmw_16>, inst16_64>;
  def : BinRMWPatGlobalAddr<i64, zext_bin_rmw_32_64<rmw_32>, inst32_64>;

  def : BinRMWPatGlobalAddr<i32, sext_bin_rmw_8_32<rmw_8>, inst8_32>;
  def : BinRMWPatGlobalAddr<i32, sext_bin_rmw_16_32<rmw_16>, inst16_32>;
  def : BinRMWPatGlobalAddr<i64, sext_bin_rmw_8_64<rmw_8>, inst8_64>;
  def : BinRMWPatGlobalAddr<i64, sext_bin_rmw_16_64<rmw_16>, inst16_64>;

  def : BinRMWPatExternalSym<i32, zext_bin_rmw_8_32<rmw_8>, inst8_32>;
  def : BinRMWPatExternalSym<i32, zext_bin_rmw_16_32<rmw_16>, inst16_32>;
  def : BinRMWPatExternalSym<i64, zext_bin_rmw_8_64<rmw_8>, inst8_64>;
  def : BinRMWPatExternalSym<i64, zext_bin_rmw_16_64<rmw_16>, inst16_64>;
  def : BinRMWPatExternalSym<i64, zext_bin_rmw_32_64<rmw_32>, inst32_64>;

  def : BinRMWPatExternalSym<i32, sext_bin_rmw_8_32<rmw_8>, inst8_32>;
  def : BinRMWPatExternalSym<i32, sext_bin_rmw_16_32<rmw_16>, inst16_32>;
  def : BinRMWPatExternalSym<i64, sext_bin_rmw_8_64<rmw_8>, inst8_64>;
  def : BinRMWPatExternalSym<i64, sext_bin_rmw_16_64<rmw_16>, inst16_64>;

  // Truncating-extending binary RMWs with just a constant offset
  def : BinRMWPatOffsetOnly<i32, zext_bin_rmw_8_32<rmw_8>, inst8_32>;
  def : BinRMWPatOffsetOnly<i32, zext_bin_rmw_16_32<rmw_16>, inst16_32>;
  def : BinRMWPatOffsetOnly<i64, zext_bin_rmw_8_64<rmw_8>, inst8_64>;
  def : BinRMWPatOffsetOnly<i64, zext_bin_rmw_16_64<rmw_16>, inst16_64>;
  def : BinRMWPatOffsetOnly<i64, zext_bin_rmw_32_64<rmw_32>, inst32_64>;

  def : BinRMWPatOffsetOnly<i32, sext_bin_rmw_8_32<rmw_8>, inst8_32>;
  def : BinRMWPatOffsetOnly<i32, sext_bin_rmw_16_32<rmw_16>, inst16_32>;
  def : BinRMWPatOffsetOnly<i64, sext_bin_rmw_8_64<rmw_8>, inst8_64>;
  def : BinRMWPatOffsetOnly<i64, sext_bin_rmw_16_64<rmw_16>, inst16_64>;

  def : BinRMWPatGlobalAddrOffOnly<i32, zext_bin_rmw_8_32<rmw_8>, inst8_32>;
  def : BinRMWPatGlobalAddrOffOnly<i32, zext_bin_rmw_16_32<rmw_16>, inst16_32>;
  def : BinRMWPatGlobalAddrOffOnly<i64, zext_bin_rmw_8_64<rmw_8>, inst8_64>;
  def : BinRMWPatGlobalAddrOffOnly<i64, zext_bin_rmw_16_64<rmw_16>, inst16_64>;
  def : BinRMWPatGlobalAddrOffOnly<i64, zext_bin_rmw_32_64<rmw_32>, inst32_64>;

  def : BinRMWPatGlobalAddrOffOnly<i32, sext_bin_rmw_8_32<rmw_8>, inst8_32>;
  def : BinRMWPatGlobalAddrOffOnly<i32, sext_bin_rmw_16_32<rmw_16>, inst16_32>;
  def : BinRMWPatGlobalAddrOffOnly<i64, sext_bin_rmw_8_64<rmw_8>, inst8_64>;
  def : BinRMWPatGlobalAddrOffOnly<i64, sext_bin_rmw_16_64<rmw_16>, inst16_64>;

  def : BinRMWPatExternSymOffOnly<i32, zext_bin_rmw_8_32<rmw_8>, inst8_32>;
  def : BinRMWPatExternSymOffOnly<i32, zext_bin_rmw_16_32<rmw_16>, inst16_32>;
  def : BinRMWPatExternSymOffOnly<i64, zext_bin_rmw_8_64<rmw_8>, inst8_64>;
  def : BinRMWPatExternSymOffOnly<i64, zext_bin_rmw_16_64<rmw_16>, inst16_64>;
  def : BinRMWPatExternSymOffOnly<i64, zext_bin_rmw_32_64<rmw_32>, inst32_64>;

  def : BinRMWPatExternSymOffOnly<i32, sext_bin_rmw_8_32<rmw_8>, inst8_32>;
  def : BinRMWPatExternSymOffOnly<i32, sext_bin_rmw_16_32<rmw_16>, inst16_32>;
  def : BinRMWPatExternSymOffOnly<i64, sext_bin_rmw_8_64<rmw_8>, inst8_64>;
  def : BinRMWPatExternSymOffOnly<i64, sext_bin_rmw_16_64<rmw_16>, inst16_64>;
}

let Predicates = [HasAtomics] in {
defm : BinRMWTruncExtPattern<
  atomic_load_add_8, atomic_load_add_16, atomic_load_add_32, atomic_load_add_64,
  ATOMIC_RMW8_U_ADD_I32, ATOMIC_RMW16_U_ADD_I32,
  ATOMIC_RMW8_U_ADD_I64, ATOMIC_RMW16_U_ADD_I64, ATOMIC_RMW32_U_ADD_I64>;
defm : BinRMWTruncExtPattern<
  atomic_load_sub_8, atomic_load_sub_16, atomic_load_sub_32, atomic_load_sub_64,
  ATOMIC_RMW8_U_SUB_I32, ATOMIC_RMW16_U_SUB_I32,
  ATOMIC_RMW8_U_SUB_I64, ATOMIC_RMW16_U_SUB_I64, ATOMIC_RMW32_U_SUB_I64>;
defm : BinRMWTruncExtPattern<
  atomic_load_and_8, atomic_load_and_16, atomic_load_and_32, atomic_load_and_64,
  ATOMIC_RMW8_U_AND_I32, ATOMIC_RMW16_U_AND_I32,
  ATOMIC_RMW8_U_AND_I64, ATOMIC_RMW16_U_AND_I64, ATOMIC_RMW32_U_AND_I64>;
defm : BinRMWTruncExtPattern<
  atomic_load_or_8, atomic_load_or_16, atomic_load_or_32, atomic_load_or_64,
  ATOMIC_RMW8_U_OR_I32, ATOMIC_RMW16_U_OR_I32,
  ATOMIC_RMW8_U_OR_I64, ATOMIC_RMW16_U_OR_I64, ATOMIC_RMW32_U_OR_I64>;
defm : BinRMWTruncExtPattern<
  atomic_load_xor_8, atomic_load_xor_16, atomic_load_xor_32, atomic_load_xor_64,
  ATOMIC_RMW8_U_XOR_I32, ATOMIC_RMW16_U_XOR_I32,
  ATOMIC_RMW8_U_XOR_I64, ATOMIC_RMW16_U_XOR_I64, ATOMIC_RMW32_U_XOR_I64>;
defm : BinRMWTruncExtPattern<
  atomic_swap_8, atomic_swap_16, atomic_swap_32, atomic_swap_64,
  ATOMIC_RMW8_U_XCHG_I32, ATOMIC_RMW16_U_XCHG_I32,
  ATOMIC_RMW8_U_XCHG_I64, ATOMIC_RMW16_U_XCHG_I64, ATOMIC_RMW32_U_XCHG_I64>;
} // Predicates = [HasAtomics]
