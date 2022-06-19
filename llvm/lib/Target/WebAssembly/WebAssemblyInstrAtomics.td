// WebAssemblyInstrAtomics.td-WebAssembly Atomic codegen support-*- tablegen -*-
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// WebAssembly Atomic operand code-gen constructs.
///
//===----------------------------------------------------------------------===//

let UseNamedOperandTable = 1 in
multiclass ATOMIC_I<dag oops_r, dag iops_r, dag oops_s, dag iops_s,
                    list<dag> pattern_r, string asmstr_r,
                    string asmstr_s, bits<32> atomic_op,
                    bit is64 = false> {
  defm "" : I<oops_r, iops_r, oops_s, iops_s, pattern_r, asmstr_r, asmstr_s,
              !or(0xfe00, !and(0xff, atomic_op)), is64>,
            Requires<[HasAtomics]>;
}

multiclass ATOMIC_NRI<dag oops, dag iops, list<dag> pattern, string asmstr = "",
                      bits<32> atomic_op = -1> {
  defm "" : NRI<oops, iops, pattern, asmstr,
                !or(0xfe00, !and(0xff, atomic_op))>,
            Requires<[HasAtomics]>;
}

//===----------------------------------------------------------------------===//
// Atomic wait / notify
//===----------------------------------------------------------------------===//

let hasSideEffects = 1 in {
defm MEMORY_ATOMIC_NOTIFY_A32 :
  ATOMIC_I<(outs I32:$dst),
           (ins P2Align:$p2align, offset32_op:$off, I32:$addr, I32:$count),
           (outs), (ins P2Align:$p2align, offset32_op:$off), [],
           "memory.atomic.notify \t$dst, ${off}(${addr})${p2align}, $count",
           "memory.atomic.notify \t${off}${p2align}", 0x00, false>;
defm MEMORY_ATOMIC_NOTIFY_A64 :
  ATOMIC_I<(outs I32:$dst),
           (ins P2Align:$p2align, offset64_op:$off, I64:$addr, I32:$count),
           (outs), (ins P2Align:$p2align, offset64_op:$off), [],
           "memory.atomic.notify \t$dst, ${off}(${addr})${p2align}, $count",
           "memory.atomic.notify \t${off}${p2align}", 0x00, true>;
let mayLoad = 1 in {
defm MEMORY_ATOMIC_WAIT32_A32 :
  ATOMIC_I<(outs I32:$dst),
           (ins P2Align:$p2align, offset32_op:$off, I32:$addr, I32:$exp,
                I64:$timeout),
           (outs), (ins P2Align:$p2align, offset32_op:$off), [],
           "memory.atomic.wait32 \t$dst, ${off}(${addr})${p2align}, $exp, $timeout",
           "memory.atomic.wait32 \t${off}${p2align}", 0x01, false>;
defm MEMORY_ATOMIC_WAIT32_A64 :
  ATOMIC_I<(outs I32:$dst),
           (ins P2Align:$p2align, offset64_op:$off, I64:$addr, I32:$exp,
                I64:$timeout),
           (outs), (ins P2Align:$p2align, offset64_op:$off), [],
           "memory.atomic.wait32 \t$dst, ${off}(${addr})${p2align}, $exp, $timeout",
           "memory.atomic.wait32 \t${off}${p2align}", 0x01, true>;
defm MEMORY_ATOMIC_WAIT64_A32 :
  ATOMIC_I<(outs I32:$dst),
           (ins P2Align:$p2align, offset32_op:$off, I32:$addr, I64:$exp,
                I64:$timeout),
           (outs), (ins P2Align:$p2align, offset32_op:$off), [],
           "memory.atomic.wait64 \t$dst, ${off}(${addr})${p2align}, $exp, $timeout",
           "memory.atomic.wait64 \t${off}${p2align}", 0x02, false>;
defm MEMORY_ATOMIC_WAIT64_A64 :
  ATOMIC_I<(outs I32:$dst),
           (ins P2Align:$p2align, offset64_op:$off, I64:$addr, I64:$exp,
                I64:$timeout),
           (outs), (ins P2Align:$p2align, offset64_op:$off), [],
           "memory.atomic.wait64 \t$dst, ${off}(${addr})${p2align}, $exp, $timeout",
           "memory.atomic.wait64 \t${off}${p2align}", 0x02, true>;
} // mayLoad = 1
} // hasSideEffects = 1

// Select notifys with no constant offset.
def NotifyPatNoOffset_A32 :
  Pat<(i32 (int_wasm_memory_atomic_notify I32:$addr, I32:$count)),
      (MEMORY_ATOMIC_NOTIFY_A32 0, 0, I32:$addr, I32:$count)>,
  Requires<[HasAddr32, HasAtomics]>;
def NotifyPatNoOffset_A64 :
  Pat<(i32 (int_wasm_memory_atomic_notify I64:$addr, I32:$count)),
      (MEMORY_ATOMIC_NOTIFY_A64 0, 0, I64:$addr, I32:$count)>,
  Requires<[HasAddr64, HasAtomics]>;

// Select notifys with a constant offset.

// Pattern with address + immediate offset
multiclass NotifyPatImmOff<PatFrag operand, string inst> {
  def : Pat<(i32 (int_wasm_memory_atomic_notify (operand I32:$addr, imm:$off),
                  I32:$count)),
            (!cast<NI>(inst#_A32) 0, imm:$off, I32:$addr, I32:$count)>,
        Requires<[HasAddr32, HasAtomics]>;
  def : Pat<(i32 (int_wasm_memory_atomic_notify (operand I64:$addr, imm:$off),
                  I32:$count)),
            (!cast<NI>(inst#_A64) 0, imm:$off, I64:$addr, I32:$count)>,
        Requires<[HasAddr64, HasAtomics]>;
}
defm : NotifyPatImmOff<regPlusImm, "MEMORY_ATOMIC_NOTIFY">;
defm : NotifyPatImmOff<or_is_add, "MEMORY_ATOMIC_NOTIFY">;

// Select notifys with just a constant offset.
def NotifyPatOffsetOnly_A32 :
  Pat<(i32 (int_wasm_memory_atomic_notify imm:$off, I32:$count)),
      (MEMORY_ATOMIC_NOTIFY_A32 0, imm:$off, (CONST_I32 0), I32:$count)>,
  Requires<[HasAddr32, HasAtomics]>;
def NotifyPatOffsetOnly_A64 :
  Pat<(i32 (int_wasm_memory_atomic_notify imm:$off, I32:$count)),
      (MEMORY_ATOMIC_NOTIFY_A64 0, imm:$off, (CONST_I64 0), I32:$count)>,
  Requires<[HasAddr64, HasAtomics]>;

def NotifyPatGlobalAddrOffOnly_A32 :
  Pat<(i32 (int_wasm_memory_atomic_notify (WebAssemblyWrapper tglobaladdr:$off),
                                          I32:$count)),
      (MEMORY_ATOMIC_NOTIFY_A32 0, tglobaladdr:$off, (CONST_I32 0), I32:$count)
     >,
  Requires<[HasAddr32, HasAtomics, IsNotPIC]>;
def NotifyPatGlobalAddrOffOnly_A64 :
  Pat<(i32 (int_wasm_memory_atomic_notify (WebAssemblyWrapper tglobaladdr:$off),
                                          I32:$count)),
      (MEMORY_ATOMIC_NOTIFY_A64 0, tglobaladdr:$off, (CONST_I64 0), I32:$count)
     >,
  Requires<[HasAddr64, HasAtomics, IsNotPIC]>;

// Select waits with no constant offset.
multiclass WaitPatNoOffset<ValueType ty, Intrinsic kind,
                      string inst> {
  def : Pat<(i32 (kind I32:$addr, ty:$exp, I64:$timeout)),
            (!cast<NI>(inst#_A32) 0, 0, I32:$addr, ty:$exp, I64:$timeout)>,
        Requires<[HasAddr32, HasAtomics]>;
  def : Pat<(i32 (kind I64:$addr, ty:$exp, I64:$timeout)),
            (!cast<NI>(inst#_A64) 0, 0, I64:$addr, ty:$exp, I64:$timeout)>,
        Requires<[HasAddr64, HasAtomics]>;
}
defm : WaitPatNoOffset<i32, int_wasm_memory_atomic_wait32,
                       "MEMORY_ATOMIC_WAIT32">;
defm : WaitPatNoOffset<i64, int_wasm_memory_atomic_wait64,
                       "MEMORY_ATOMIC_WAIT64">;
defm : WaitPatNoOffset<i32, int_wasm_memory_atomic_wait32,
                       "MEMORY_ATOMIC_WAIT32">;
defm : WaitPatNoOffset<i64, int_wasm_memory_atomic_wait64,
                       "MEMORY_ATOMIC_WAIT64">;

// Select waits with a constant offset.

// Pattern with address + immediate offset
multiclass WaitPatImmOff<ValueType ty, Intrinsic kind, PatFrag operand,
                         string inst> {
  def : Pat<(i32 (kind (operand I32:$addr, imm:$off), ty:$exp, I64:$timeout)),
            (!cast<NI>(inst#_A32) 0, imm:$off, I32:$addr, ty:$exp,
              I64:$timeout)>,
        Requires<[HasAddr32, HasAtomics]>;
  def : Pat<(i32 (kind (operand I64:$addr, imm:$off), ty:$exp, I64:$timeout)),
            (!cast<NI>(inst#_A64) 0, imm:$off, I64:$addr, ty:$exp,
              I64:$timeout)>,
        Requires<[HasAddr64, HasAtomics]>;
}
defm : WaitPatImmOff<i32, int_wasm_memory_atomic_wait32, regPlusImm,
                     "MEMORY_ATOMIC_WAIT32">;
defm : WaitPatImmOff<i32, int_wasm_memory_atomic_wait32, or_is_add,
                     "MEMORY_ATOMIC_WAIT32">;
defm : WaitPatImmOff<i64, int_wasm_memory_atomic_wait64, regPlusImm,
                     "MEMORY_ATOMIC_WAIT64">;
defm : WaitPatImmOff<i64, int_wasm_memory_atomic_wait64, or_is_add,
                     "MEMORY_ATOMIC_WAIT64">;

// Select waits with just a constant offset.
multiclass WaitPatOffsetOnly<ValueType ty, Intrinsic kind, string inst> {
  def : Pat<(i32 (kind imm:$off, ty:$exp, I64:$timeout)),
            (!cast<NI>(inst#_A32) 0, imm:$off, (CONST_I32 0), ty:$exp,
               I64:$timeout)>,
        Requires<[HasAddr32, HasAtomics]>;
  def : Pat<(i32 (kind imm:$off, ty:$exp, I64:$timeout)),
            (!cast<NI>(inst#_A64) 0, imm:$off, (CONST_I64 0), ty:$exp,
               I64:$timeout)>,
        Requires<[HasAddr64, HasAtomics]>;
}
defm : WaitPatOffsetOnly<i32, int_wasm_memory_atomic_wait32,
                         "MEMORY_ATOMIC_WAIT32">;
defm : WaitPatOffsetOnly<i64, int_wasm_memory_atomic_wait64,
                         "MEMORY_ATOMIC_WAIT64">;

multiclass WaitPatGlobalAddrOffOnly<ValueType ty, Intrinsic kind, string inst> {
  def : Pat<(i32 (kind (WebAssemblyWrapper tglobaladdr:$off), ty:$exp,
                  I64:$timeout)),
            (!cast<NI>(inst#_A32) 0, tglobaladdr:$off, (CONST_I32 0), ty:$exp,
               I64:$timeout)>,
        Requires<[HasAddr32, HasAtomics, IsNotPIC]>;
  def : Pat<(i32 (kind (WebAssemblyWrapper tglobaladdr:$off), ty:$exp,
                  I64:$timeout)),
            (!cast<NI>(inst#_A64) 0, tglobaladdr:$off, (CONST_I64 0), ty:$exp,
               I64:$timeout)>,
        Requires<[HasAddr64, HasAtomics, IsNotPIC]>;
}
defm : WaitPatGlobalAddrOffOnly<i32, int_wasm_memory_atomic_wait32,
                                "MEMORY_ATOMIC_WAIT32">;
defm : WaitPatGlobalAddrOffOnly<i64, int_wasm_memory_atomic_wait64,
                                "MEMORY_ATOMIC_WAIT64">;

//===----------------------------------------------------------------------===//
// Atomic fences
//===----------------------------------------------------------------------===//

// A compiler fence instruction that prevents reordering of instructions.
let Defs = [ARGUMENTS] in {
let isPseudo = 1, hasSideEffects = 1 in
defm COMPILER_FENCE : ATOMIC_NRI<(outs), (ins), [], "compiler_fence">;
let hasSideEffects = 1 in
defm ATOMIC_FENCE : ATOMIC_NRI<(outs), (ins i8imm:$flags), [], "atomic.fence",
                               0x03>;
} // Defs = [ARGUMENTS]

//===----------------------------------------------------------------------===//
// Atomic loads
//===----------------------------------------------------------------------===//

multiclass AtomicLoad<WebAssemblyRegClass rc, string name, int atomic_op> {
  defm "" : WebAssemblyLoad<rc, name, !or(0xfe00, !and(0xff, atomic_op)),
                            [HasAtomics]>;
}

defm ATOMIC_LOAD_I32 : AtomicLoad<I32, "i32.atomic.load", 0x10>;
defm ATOMIC_LOAD_I64 : AtomicLoad<I64, "i64.atomic.load", 0x11>;

// Select loads with no constant offset.
defm : LoadPatNoOffset<i32, atomic_load_32, "ATOMIC_LOAD_I32">;
defm : LoadPatNoOffset<i64, atomic_load_64, "ATOMIC_LOAD_I64">;

// Select loads with a constant offset.

// Pattern with address + immediate offset
defm : LoadPatImmOff<i32, atomic_load_32, regPlusImm, "ATOMIC_LOAD_I32">;
defm : LoadPatImmOff<i64, atomic_load_64, regPlusImm, "ATOMIC_LOAD_I64">;
defm : LoadPatImmOff<i32, atomic_load_32, or_is_add, "ATOMIC_LOAD_I32">;
defm : LoadPatImmOff<i64, atomic_load_64, or_is_add, "ATOMIC_LOAD_I64">;

// Select loads with just a constant offset.
defm : LoadPatOffsetOnly<i32, atomic_load_32, "ATOMIC_LOAD_I32">;
defm : LoadPatOffsetOnly<i64, atomic_load_64, "ATOMIC_LOAD_I64">;

defm : LoadPatGlobalAddrOffOnly<i32, atomic_load_32, "ATOMIC_LOAD_I32">;
defm : LoadPatGlobalAddrOffOnly<i64, atomic_load_64, "ATOMIC_LOAD_I64">;


// Extending loads. Note that there are only zero-extending atomic loads, no
// sign-extending loads.
defm ATOMIC_LOAD8_U_I32 : AtomicLoad<I32, "i32.atomic.load8_u", 0x12>;
defm ATOMIC_LOAD16_U_I32 : AtomicLoad<I32, "i32.atomic.load16_u", 0x13>;
defm ATOMIC_LOAD8_U_I64 : AtomicLoad<I64, "i64.atomic.load8_u", 0x14>;
defm ATOMIC_LOAD16_U_I64 : AtomicLoad<I64, "i64.atomic.load16_u", 0x15>;
defm ATOMIC_LOAD32_U_I64 : AtomicLoad<I64, "i64.atomic.load32_u", 0x16>;

// Fragments for extending loads. These are different from regular loads because
// the SDNodes are derived from AtomicSDNode rather than LoadSDNode and
// therefore don't have the extension type field. So instead of matching that,
// we match the patterns that the type legalizer expands them to.

// Unlike regular loads, extension to i64 is handled differently than i32.
// i64 (zext (i8 (atomic_load_8))) gets legalized to
// i64 (and (i64 (anyext (i32 (atomic_load_8)))), 255)
// Extension to i32 is elided by SelectionDAG as our atomic loads are
// zero-extending.
def zext_aload_8_64 :
  PatFrag<(ops node:$addr),
          (i64 (zext (i32 (atomic_load_8 node:$addr))))>;
def zext_aload_16_64 :
  PatFrag<(ops node:$addr),
          (i64 (zext (i32 (atomic_load_16 node:$addr))))>;
def zext_aload_32_64 :
  PatFrag<(ops node:$addr),
          (i64 (zext (i32 (atomic_load_32 node:$addr))))>;

// We don't have single sext atomic load instructions. So for sext loads, we
// match bare subword loads (for 32-bit results) and anyext loads (for 64-bit
// results) and select a zext load; the next instruction will be sext_inreg
// which is selected by itself.
def sext_aload_8_64 :
  PatFrag<(ops node:$addr), (anyext (i32 (atomic_load_8 node:$addr)))>;
def sext_aload_16_64 :
  PatFrag<(ops node:$addr), (anyext (i32 (atomic_load_16 node:$addr)))>;

// Select zero-extending loads with no constant offset.
defm : LoadPatNoOffset<i64, zext_aload_8_64, "ATOMIC_LOAD8_U_I64">;
defm : LoadPatNoOffset<i64, zext_aload_16_64, "ATOMIC_LOAD16_U_I64">;
defm : LoadPatNoOffset<i64, zext_aload_32_64, "ATOMIC_LOAD32_U_I64">;

// Select sign-extending loads with no constant offset
defm : LoadPatNoOffset<i32, atomic_load_8, "ATOMIC_LOAD8_U_I32">;
defm : LoadPatNoOffset<i32, atomic_load_16, "ATOMIC_LOAD16_U_I32">;
defm : LoadPatNoOffset<i64, sext_aload_8_64, "ATOMIC_LOAD8_U_I64">;
defm : LoadPatNoOffset<i64, sext_aload_16_64, "ATOMIC_LOAD16_U_I64">;
// 32->64 sext load gets selected as i32.atomic.load, i64.extend_i32_s

// Zero-extending loads with constant offset
defm : LoadPatImmOff<i64, zext_aload_8_64, regPlusImm, "ATOMIC_LOAD8_U_I64">;
defm : LoadPatImmOff<i64, zext_aload_16_64, regPlusImm, "ATOMIC_LOAD16_U_I64">;
defm : LoadPatImmOff<i64, zext_aload_32_64, regPlusImm, "ATOMIC_LOAD32_U_I64">;
defm : LoadPatImmOff<i64, zext_aload_8_64, or_is_add, "ATOMIC_LOAD8_U_I64">;
defm : LoadPatImmOff<i64, zext_aload_16_64, or_is_add, "ATOMIC_LOAD16_U_I64">;
defm : LoadPatImmOff<i64, zext_aload_32_64, or_is_add, "ATOMIC_LOAD32_U_I64">;

// Sign-extending loads with constant offset
defm : LoadPatImmOff<i32, atomic_load_8, regPlusImm, "ATOMIC_LOAD8_U_I32">;
defm : LoadPatImmOff<i32, atomic_load_16, regPlusImm, "ATOMIC_LOAD16_U_I32">;
defm : LoadPatImmOff<i32, atomic_load_8, or_is_add, "ATOMIC_LOAD8_U_I32">;
defm : LoadPatImmOff<i32, atomic_load_16, or_is_add, "ATOMIC_LOAD16_U_I32">;
defm : LoadPatImmOff<i64, sext_aload_8_64, regPlusImm, "ATOMIC_LOAD8_U_I64">;
defm : LoadPatImmOff<i64, sext_aload_16_64, regPlusImm, "ATOMIC_LOAD16_U_I64">;
defm : LoadPatImmOff<i64, sext_aload_8_64, or_is_add, "ATOMIC_LOAD8_U_I64">;
defm : LoadPatImmOff<i64, sext_aload_16_64, or_is_add, "ATOMIC_LOAD16_U_I64">;
// No 32->64 patterns, just use i32.atomic.load and i64.extend_s/i64

// Extending loads with just a constant offset
defm : LoadPatOffsetOnly<i64, zext_aload_8_64, "ATOMIC_LOAD8_U_I64">;
defm : LoadPatOffsetOnly<i64, zext_aload_16_64, "ATOMIC_LOAD16_U_I64">;
defm : LoadPatOffsetOnly<i64, zext_aload_32_64, "ATOMIC_LOAD32_U_I64">;
defm : LoadPatOffsetOnly<i32, atomic_load_8, "ATOMIC_LOAD8_U_I32">;
defm : LoadPatOffsetOnly<i32, atomic_load_16, "ATOMIC_LOAD16_U_I32">;
defm : LoadPatOffsetOnly<i64, sext_aload_8_64, "ATOMIC_LOAD8_U_I64">;
defm : LoadPatOffsetOnly<i64, sext_aload_16_64, "ATOMIC_LOAD16_U_I64">;

defm : LoadPatGlobalAddrOffOnly<i64, zext_aload_8_64, "ATOMIC_LOAD8_U_I64">;
defm : LoadPatGlobalAddrOffOnly<i64, zext_aload_16_64, "ATOMIC_LOAD16_U_I64">;
defm : LoadPatGlobalAddrOffOnly<i64, zext_aload_32_64, "ATOMIC_LOAD32_U_I64">;
defm : LoadPatGlobalAddrOffOnly<i32, atomic_load_8, "ATOMIC_LOAD8_U_I32">;
defm : LoadPatGlobalAddrOffOnly<i32, atomic_load_16, "ATOMIC_LOAD16_U_I32">;
defm : LoadPatGlobalAddrOffOnly<i64, sext_aload_8_64, "ATOMIC_LOAD8_U_I64">;
defm : LoadPatGlobalAddrOffOnly<i64, sext_aload_16_64, "ATOMIC_LOAD16_U_I64">;


//===----------------------------------------------------------------------===//
// Atomic stores
//===----------------------------------------------------------------------===//

multiclass AtomicStore<WebAssemblyRegClass rc, string name, int atomic_op> {
  defm "" : WebAssemblyStore<rc, name, !or(0xfe00, !and(0xff, atomic_op)),
                             [HasAtomics]>;
}

defm ATOMIC_STORE_I32 : AtomicStore<I32, "i32.atomic.store", 0x17>;
defm ATOMIC_STORE_I64 : AtomicStore<I64, "i64.atomic.store", 0x18>;

// We need an 'atomic' version of store patterns because store and atomic_store
// nodes have different operand orders:
// store: (store $val, $ptr)
// atomic_store: (store $ptr, $val)


// Select stores with no constant offset.
multiclass AStorePatNoOffset<ValueType ty, PatFrag kind, string inst> {
  def : Pat<(kind I32:$addr, ty:$val),
            (!cast<NI>(inst#_A32) 0, 0, I32:$addr, ty:$val)>,
        Requires<[HasAddr32, HasAtomics]>;
  def : Pat<(kind I64:$addr, ty:$val),
            (!cast<NI>(inst#_A64) 0, 0, I64:$addr, ty:$val)>,
        Requires<[HasAddr64, HasAtomics]>;
}
defm : AStorePatNoOffset<i32, atomic_store_32, "ATOMIC_STORE_I32">;
defm : AStorePatNoOffset<i64, atomic_store_64, "ATOMIC_STORE_I64">;

// Select stores with a constant offset.

// Pattern with address + immediate offset
multiclass AStorePatImmOff<ValueType ty, PatFrag kind, PatFrag operand,
                           string inst> {
  def : Pat<(kind (operand I32:$addr, imm:$off), ty:$val),
            (!cast<NI>(inst#_A32) 0, imm:$off, I32:$addr, ty:$val)>,
        Requires<[HasAddr32, HasAtomics]>;
  def : Pat<(kind (operand I64:$addr, imm:$off), ty:$val),
            (!cast<NI>(inst#_A64) 0, imm:$off, I64:$addr, ty:$val)>,
        Requires<[HasAddr64, HasAtomics]>;
}
defm : AStorePatImmOff<i32, atomic_store_32, regPlusImm, "ATOMIC_STORE_I32">;
defm : AStorePatImmOff<i64, atomic_store_64, regPlusImm, "ATOMIC_STORE_I64">;

// Select stores with just a constant offset.
multiclass AStorePatOffsetOnly<ValueType ty, PatFrag kind, string inst> {
  def : Pat<(kind imm:$off, ty:$val),
            (!cast<NI>(inst#_A32) 0, imm:$off, (CONST_I32 0), ty:$val)>,
        Requires<[HasAddr32, HasAtomics]>;
  def : Pat<(kind imm:$off, ty:$val),
            (!cast<NI>(inst#_A64) 0, imm:$off, (CONST_I64 0), ty:$val)>,
        Requires<[HasAddr64, HasAtomics]>;
}
defm : AStorePatOffsetOnly<i32, atomic_store_32, "ATOMIC_STORE_I32">;
defm : AStorePatOffsetOnly<i64, atomic_store_64, "ATOMIC_STORE_I64">;

multiclass AStorePatGlobalAddrOffOnly<ValueType ty, PatFrag kind, string inst> {
  def : Pat<(kind (WebAssemblyWrapper tglobaladdr:$off), ty:$val),
            (!cast<NI>(inst#_A32) 0, tglobaladdr:$off, (CONST_I32 0), ty:$val)>,
        Requires<[HasAddr32, HasAtomics, IsNotPIC]>;
  def : Pat<(kind (WebAssemblyWrapper tglobaladdr:$off), ty:$val),
            (!cast<NI>(inst#_A64) 0, tglobaladdr:$off, (CONST_I64 0), ty:$val)>,
        Requires<[HasAddr64, HasAtomics, IsNotPIC]>;
}
defm : AStorePatGlobalAddrOffOnly<i32, atomic_store_32, "ATOMIC_STORE_I32">;
defm : AStorePatGlobalAddrOffOnly<i64, atomic_store_64, "ATOMIC_STORE_I64">;


// Truncating stores.
defm ATOMIC_STORE8_I32 : AtomicStore<I32, "i32.atomic.store8", 0x19>;
defm ATOMIC_STORE16_I32 : AtomicStore<I32, "i32.atomic.store16", 0x1a>;
defm ATOMIC_STORE8_I64 : AtomicStore<I64, "i64.atomic.store8", 0x1b>;
defm ATOMIC_STORE16_I64 : AtomicStore<I64, "i64.atomic.store16", 0x1c>;
defm ATOMIC_STORE32_I64 : AtomicStore<I64, "i64.atomic.store32", 0x1d>;

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


// Truncating stores with no constant offset
defm : AStorePatNoOffset<i32, atomic_store_8, "ATOMIC_STORE8_I32">;
defm : AStorePatNoOffset<i32, atomic_store_16, "ATOMIC_STORE16_I32">;
defm : AStorePatNoOffset<i64, trunc_astore_8_64, "ATOMIC_STORE8_I64">;
defm : AStorePatNoOffset<i64, trunc_astore_16_64, "ATOMIC_STORE16_I64">;
defm : AStorePatNoOffset<i64, trunc_astore_32_64, "ATOMIC_STORE32_I64">;

// Truncating stores with a constant offset
defm : AStorePatImmOff<i32, atomic_store_8, regPlusImm, "ATOMIC_STORE8_I32">;
defm : AStorePatImmOff<i32, atomic_store_16, regPlusImm, "ATOMIC_STORE16_I32">;
defm : AStorePatImmOff<i64, trunc_astore_8_64, regPlusImm, "ATOMIC_STORE8_I64">;
defm : AStorePatImmOff<i64, trunc_astore_16_64, regPlusImm,
                       "ATOMIC_STORE16_I64">;
defm : AStorePatImmOff<i64, trunc_astore_32_64, regPlusImm,
                       "ATOMIC_STORE32_I64">;
defm : AStorePatImmOff<i32, atomic_store_8, or_is_add, "ATOMIC_STORE8_I32">;
defm : AStorePatImmOff<i32, atomic_store_16, or_is_add, "ATOMIC_STORE16_I32">;
defm : AStorePatImmOff<i64, trunc_astore_8_64, or_is_add, "ATOMIC_STORE8_I64">;
defm : AStorePatImmOff<i64, trunc_astore_16_64, or_is_add,
                       "ATOMIC_STORE16_I64">;
defm : AStorePatImmOff<i64, trunc_astore_32_64, or_is_add,
                       "ATOMIC_STORE32_I64">;

// Truncating stores with just a constant offset
defm : AStorePatOffsetOnly<i32, atomic_store_8, "ATOMIC_STORE8_I32">;
defm : AStorePatOffsetOnly<i32, atomic_store_16, "ATOMIC_STORE16_I32">;
defm : AStorePatOffsetOnly<i64, trunc_astore_8_64, "ATOMIC_STORE8_I64">;
defm : AStorePatOffsetOnly<i64, trunc_astore_16_64, "ATOMIC_STORE16_I64">;
defm : AStorePatOffsetOnly<i64, trunc_astore_32_64, "ATOMIC_STORE32_I64">;

defm : AStorePatGlobalAddrOffOnly<i32, atomic_store_8, "ATOMIC_STORE8_I32">;
defm : AStorePatGlobalAddrOffOnly<i32, atomic_store_16, "ATOMIC_STORE16_I32">;
defm : AStorePatGlobalAddrOffOnly<i64, trunc_astore_8_64, "ATOMIC_STORE8_I64">;
defm : AStorePatGlobalAddrOffOnly<i64, trunc_astore_16_64, "ATOMIC_STORE16_I64">;
defm : AStorePatGlobalAddrOffOnly<i64, trunc_astore_32_64, "ATOMIC_STORE32_I64">;


//===----------------------------------------------------------------------===//
// Atomic binary read-modify-writes
//===----------------------------------------------------------------------===//

multiclass WebAssemblyBinRMW<WebAssemblyRegClass rc, string name,
                             int atomic_op> {
  defm "_A32" :
    ATOMIC_I<(outs rc:$dst),
             (ins P2Align:$p2align, offset32_op:$off, I32:$addr, rc:$val),
             (outs), (ins P2Align:$p2align, offset32_op:$off), [],
             !strconcat(name, "\t$dst, ${off}(${addr})${p2align}, $val"),
             !strconcat(name, "\t${off}${p2align}"), atomic_op, false>;
  defm "_A64" :
    ATOMIC_I<(outs rc:$dst),
             (ins P2Align:$p2align, offset64_op:$off, I64:$addr, rc:$val),
             (outs), (ins P2Align:$p2align, offset64_op:$off), [],
             !strconcat(name, "\t$dst, ${off}(${addr})${p2align}, $val"),
             !strconcat(name, "\t${off}${p2align}"), atomic_op, true>;
}

defm ATOMIC_RMW_ADD_I32 : WebAssemblyBinRMW<I32, "i32.atomic.rmw.add", 0x1e>;
defm ATOMIC_RMW_ADD_I64 : WebAssemblyBinRMW<I64, "i64.atomic.rmw.add", 0x1f>;
defm ATOMIC_RMW8_U_ADD_I32 :
  WebAssemblyBinRMW<I32, "i32.atomic.rmw8.add_u", 0x20>;
defm ATOMIC_RMW16_U_ADD_I32 :
  WebAssemblyBinRMW<I32, "i32.atomic.rmw16.add_u", 0x21>;
defm ATOMIC_RMW8_U_ADD_I64 :
  WebAssemblyBinRMW<I64, "i64.atomic.rmw8.add_u", 0x22>;
defm ATOMIC_RMW16_U_ADD_I64 :
  WebAssemblyBinRMW<I64, "i64.atomic.rmw16.add_u", 0x23>;
defm ATOMIC_RMW32_U_ADD_I64 :
  WebAssemblyBinRMW<I64, "i64.atomic.rmw32.add_u", 0x24>;

defm ATOMIC_RMW_SUB_I32 : WebAssemblyBinRMW<I32, "i32.atomic.rmw.sub", 0x25>;
defm ATOMIC_RMW_SUB_I64 : WebAssemblyBinRMW<I64, "i64.atomic.rmw.sub", 0x26>;
defm ATOMIC_RMW8_U_SUB_I32 :
  WebAssemblyBinRMW<I32, "i32.atomic.rmw8.sub_u", 0x27>;
defm ATOMIC_RMW16_U_SUB_I32 :
  WebAssemblyBinRMW<I32, "i32.atomic.rmw16.sub_u", 0x28>;
defm ATOMIC_RMW8_U_SUB_I64 :
  WebAssemblyBinRMW<I64, "i64.atomic.rmw8.sub_u", 0x29>;
defm ATOMIC_RMW16_U_SUB_I64 :
  WebAssemblyBinRMW<I64, "i64.atomic.rmw16.sub_u", 0x2a>;
defm ATOMIC_RMW32_U_SUB_I64 :
  WebAssemblyBinRMW<I64, "i64.atomic.rmw32.sub_u", 0x2b>;

defm ATOMIC_RMW_AND_I32 : WebAssemblyBinRMW<I32, "i32.atomic.rmw.and", 0x2c>;
defm ATOMIC_RMW_AND_I64 : WebAssemblyBinRMW<I64, "i64.atomic.rmw.and", 0x2d>;
defm ATOMIC_RMW8_U_AND_I32 :
  WebAssemblyBinRMW<I32, "i32.atomic.rmw8.and_u", 0x2e>;
defm ATOMIC_RMW16_U_AND_I32 :
  WebAssemblyBinRMW<I32, "i32.atomic.rmw16.and_u", 0x2f>;
defm ATOMIC_RMW8_U_AND_I64 :
  WebAssemblyBinRMW<I64, "i64.atomic.rmw8.and_u", 0x30>;
defm ATOMIC_RMW16_U_AND_I64 :
  WebAssemblyBinRMW<I64, "i64.atomic.rmw16.and_u", 0x31>;
defm ATOMIC_RMW32_U_AND_I64 :
  WebAssemblyBinRMW<I64, "i64.atomic.rmw32.and_u", 0x32>;

defm ATOMIC_RMW_OR_I32 : WebAssemblyBinRMW<I32, "i32.atomic.rmw.or", 0x33>;
defm ATOMIC_RMW_OR_I64 : WebAssemblyBinRMW<I64, "i64.atomic.rmw.or", 0x34>;
defm ATOMIC_RMW8_U_OR_I32 :
  WebAssemblyBinRMW<I32, "i32.atomic.rmw8.or_u", 0x35>;
defm ATOMIC_RMW16_U_OR_I32 :
  WebAssemblyBinRMW<I32, "i32.atomic.rmw16.or_u", 0x36>;
defm ATOMIC_RMW8_U_OR_I64 :
  WebAssemblyBinRMW<I64, "i64.atomic.rmw8.or_u", 0x37>;
defm ATOMIC_RMW16_U_OR_I64 :
  WebAssemblyBinRMW<I64, "i64.atomic.rmw16.or_u", 0x38>;
defm ATOMIC_RMW32_U_OR_I64 :
  WebAssemblyBinRMW<I64, "i64.atomic.rmw32.or_u", 0x39>;

defm ATOMIC_RMW_XOR_I32 : WebAssemblyBinRMW<I32, "i32.atomic.rmw.xor", 0x3a>;
defm ATOMIC_RMW_XOR_I64 : WebAssemblyBinRMW<I64, "i64.atomic.rmw.xor", 0x3b>;
defm ATOMIC_RMW8_U_XOR_I32 :
  WebAssemblyBinRMW<I32, "i32.atomic.rmw8.xor_u", 0x3c>;
defm ATOMIC_RMW16_U_XOR_I32 :
  WebAssemblyBinRMW<I32, "i32.atomic.rmw16.xor_u", 0x3d>;
defm ATOMIC_RMW8_U_XOR_I64 :
  WebAssemblyBinRMW<I64, "i64.atomic.rmw8.xor_u", 0x3e>;
defm ATOMIC_RMW16_U_XOR_I64 :
  WebAssemblyBinRMW<I64, "i64.atomic.rmw16.xor_u", 0x3f>;
defm ATOMIC_RMW32_U_XOR_I64 :
  WebAssemblyBinRMW<I64, "i64.atomic.rmw32.xor_u", 0x40>;

defm ATOMIC_RMW_XCHG_I32 :
  WebAssemblyBinRMW<I32, "i32.atomic.rmw.xchg", 0x41>;
defm ATOMIC_RMW_XCHG_I64 :
  WebAssemblyBinRMW<I64, "i64.atomic.rmw.xchg", 0x42>;
defm ATOMIC_RMW8_U_XCHG_I32 :
  WebAssemblyBinRMW<I32, "i32.atomic.rmw8.xchg_u", 0x43>;
defm ATOMIC_RMW16_U_XCHG_I32 :
  WebAssemblyBinRMW<I32, "i32.atomic.rmw16.xchg_u", 0x44>;
defm ATOMIC_RMW8_U_XCHG_I64 :
  WebAssemblyBinRMW<I64, "i64.atomic.rmw8.xchg_u", 0x45>;
defm ATOMIC_RMW16_U_XCHG_I64 :
  WebAssemblyBinRMW<I64, "i64.atomic.rmw16.xchg_u", 0x46>;
defm ATOMIC_RMW32_U_XCHG_I64 :
  WebAssemblyBinRMW<I64, "i64.atomic.rmw32.xchg_u", 0x47>;

// Select binary RMWs with no constant offset.
multiclass BinRMWPatNoOffset<ValueType ty, PatFrag kind, string inst> {
  def : Pat<(ty (kind I32:$addr, ty:$val)),
            (!cast<NI>(inst#_A32) 0, 0, I32:$addr, ty:$val)>,
        Requires<[HasAddr32, HasAtomics]>;
  def : Pat<(ty (kind I64:$addr, ty:$val)),
            (!cast<NI>(inst#_A64) 0, 0, I64:$addr, ty:$val)>,
        Requires<[HasAddr64, HasAtomics]>;
}

// Select binary RMWs with a constant offset.

// Pattern with address + immediate offset
multiclass BinRMWPatImmOff<ValueType ty, PatFrag kind, PatFrag operand,
                           string inst> {
  def : Pat<(ty (kind (operand I32:$addr, imm:$off), ty:$val)),
            (!cast<NI>(inst#_A32) 0, imm:$off, I32:$addr, ty:$val)>,
        Requires<[HasAddr32, HasAtomics]>;
  def : Pat<(ty (kind (operand I64:$addr, imm:$off), ty:$val)),
            (!cast<NI>(inst#_A64) 0, imm:$off, I64:$addr, ty:$val)>,
        Requires<[HasAddr64, HasAtomics]>;
}

// Select binary RMWs with just a constant offset.
multiclass BinRMWPatOffsetOnly<ValueType ty, PatFrag kind, string inst> {
  def : Pat<(ty (kind imm:$off, ty:$val)),
            (!cast<NI>(inst#_A32) 0, imm:$off, (CONST_I32 0), ty:$val)>,
        Requires<[HasAddr32, HasAtomics]>;
  def : Pat<(ty (kind imm:$off, ty:$val)),
            (!cast<NI>(inst#_A64) 0, imm:$off, (CONST_I64 0), ty:$val)>,
        Requires<[HasAddr64, HasAtomics]>;
}

multiclass BinRMWPatGlobalAddrOffOnly<ValueType ty, PatFrag kind, string inst> {
  def : Pat<(ty (kind (WebAssemblyWrapper tglobaladdr:$off), ty:$val)),
            (!cast<NI>(inst#_A32) 0, tglobaladdr:$off, (CONST_I32 0), ty:$val)>,
        Requires<[HasAddr32, HasAtomics, IsNotPIC]>;
  def : Pat<(ty (kind (WebAssemblyWrapper tglobaladdr:$off), ty:$val)),
            (!cast<NI>(inst#_A64) 0, tglobaladdr:$off, (CONST_I64 0), ty:$val)>,
        Requires<[HasAddr64, HasAtomics, IsNotPIC]>;
}

// Patterns for various addressing modes.
multiclass BinRMWPattern<PatFrag rmw_32, PatFrag rmw_64, string inst_32,
                         string inst_64> {
  defm : BinRMWPatNoOffset<i32, rmw_32, inst_32>;
  defm : BinRMWPatNoOffset<i64, rmw_64, inst_64>;

  defm : BinRMWPatImmOff<i32, rmw_32, regPlusImm, inst_32>;
  defm : BinRMWPatImmOff<i64, rmw_64, regPlusImm, inst_64>;
  defm : BinRMWPatImmOff<i32, rmw_32, or_is_add, inst_32>;
  defm : BinRMWPatImmOff<i64, rmw_64, or_is_add, inst_64>;

  defm : BinRMWPatOffsetOnly<i32, rmw_32, inst_32>;
  defm : BinRMWPatOffsetOnly<i64, rmw_64, inst_64>;

  defm : BinRMWPatGlobalAddrOffOnly<i32, rmw_32, inst_32>;
  defm : BinRMWPatGlobalAddrOffOnly<i64, rmw_64, inst_64>;
}

defm : BinRMWPattern<atomic_load_add_32, atomic_load_add_64,
                     "ATOMIC_RMW_ADD_I32", "ATOMIC_RMW_ADD_I64">;
defm : BinRMWPattern<atomic_load_sub_32, atomic_load_sub_64,
                     "ATOMIC_RMW_SUB_I32", "ATOMIC_RMW_SUB_I64">;
defm : BinRMWPattern<atomic_load_and_32, atomic_load_and_64,
                     "ATOMIC_RMW_AND_I32", "ATOMIC_RMW_AND_I64">;
defm : BinRMWPattern<atomic_load_or_32, atomic_load_or_64,
                     "ATOMIC_RMW_OR_I32", "ATOMIC_RMW_OR_I64">;
defm : BinRMWPattern<atomic_load_xor_32, atomic_load_xor_64,
                     "ATOMIC_RMW_XOR_I32", "ATOMIC_RMW_XOR_I64">;
defm : BinRMWPattern<atomic_swap_32, atomic_swap_64,
                     "ATOMIC_RMW_XCHG_I32", "ATOMIC_RMW_XCHG_I64">;

// Truncating & zero-extending binary RMW patterns.
// These are combined patterns of truncating store patterns and zero-extending
// load patterns above.
class zext_bin_rmw_8_32<PatFrag kind> :
  PatFrag<(ops node:$addr, node:$val), (i32 (kind node:$addr, node:$val))>;
class zext_bin_rmw_16_32<PatFrag kind> : zext_bin_rmw_8_32<kind>;
class zext_bin_rmw_8_64<PatFrag kind> :
  PatFrag<(ops node:$addr, node:$val),
          (zext (i32 (kind node:$addr, (i32 (trunc (i64 node:$val))))))>;
class zext_bin_rmw_16_64<PatFrag kind> : zext_bin_rmw_8_64<kind>;
class zext_bin_rmw_32_64<PatFrag kind> : zext_bin_rmw_8_64<kind>;

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
// 32->64 sext RMW gets selected as i32.atomic.rmw.***, i64.extend_i32_s

// Patterns for various addressing modes for truncating-extending binary RMWs.
multiclass BinRMWTruncExtPattern<
  PatFrag rmw_8, PatFrag rmw_16, PatFrag rmw_32,
  string inst8_32, string inst16_32, string inst8_64, string inst16_64, string inst32_64> {
  // Truncating-extending binary RMWs with no constant offset
  defm : BinRMWPatNoOffset<i32, zext_bin_rmw_8_32<rmw_8>, inst8_32>;
  defm : BinRMWPatNoOffset<i32, zext_bin_rmw_16_32<rmw_16>, inst16_32>;
  defm : BinRMWPatNoOffset<i64, zext_bin_rmw_8_64<rmw_8>, inst8_64>;
  defm : BinRMWPatNoOffset<i64, zext_bin_rmw_16_64<rmw_16>, inst16_64>;
  defm : BinRMWPatNoOffset<i64, zext_bin_rmw_32_64<rmw_32>, inst32_64>;

  defm : BinRMWPatNoOffset<i32, sext_bin_rmw_8_32<rmw_8>, inst8_32>;
  defm : BinRMWPatNoOffset<i32, sext_bin_rmw_16_32<rmw_16>, inst16_32>;
  defm : BinRMWPatNoOffset<i64, sext_bin_rmw_8_64<rmw_8>, inst8_64>;
  defm : BinRMWPatNoOffset<i64, sext_bin_rmw_16_64<rmw_16>, inst16_64>;

  // Truncating-extending binary RMWs with a constant offset
  defm : BinRMWPatImmOff<i32, zext_bin_rmw_8_32<rmw_8>, regPlusImm, inst8_32>;
  defm : BinRMWPatImmOff<i32, zext_bin_rmw_16_32<rmw_16>, regPlusImm,
                         inst16_32>;
  defm : BinRMWPatImmOff<i64, zext_bin_rmw_8_64<rmw_8>, regPlusImm, inst8_64>;
  defm : BinRMWPatImmOff<i64, zext_bin_rmw_16_64<rmw_16>, regPlusImm,
                         inst16_64>;
  defm : BinRMWPatImmOff<i64, zext_bin_rmw_32_64<rmw_32>, regPlusImm,
                         inst32_64>;
  defm : BinRMWPatImmOff<i32, zext_bin_rmw_8_32<rmw_8>, or_is_add, inst8_32>;
  defm : BinRMWPatImmOff<i32, zext_bin_rmw_16_32<rmw_16>, or_is_add, inst16_32>;
  defm : BinRMWPatImmOff<i64, zext_bin_rmw_8_64<rmw_8>, or_is_add, inst8_64>;
  defm : BinRMWPatImmOff<i64, zext_bin_rmw_16_64<rmw_16>, or_is_add, inst16_64>;
  defm : BinRMWPatImmOff<i64, zext_bin_rmw_32_64<rmw_32>, or_is_add, inst32_64>;

  defm : BinRMWPatImmOff<i32, sext_bin_rmw_8_32<rmw_8>, regPlusImm, inst8_32>;
  defm : BinRMWPatImmOff<i32, sext_bin_rmw_16_32<rmw_16>, regPlusImm,
                         inst16_32>;
  defm : BinRMWPatImmOff<i64, sext_bin_rmw_8_64<rmw_8>, regPlusImm, inst8_64>;
  defm : BinRMWPatImmOff<i64, sext_bin_rmw_16_64<rmw_16>, regPlusImm,
                         inst16_64>;
  defm : BinRMWPatImmOff<i32, sext_bin_rmw_8_32<rmw_8>, or_is_add, inst8_32>;
  defm : BinRMWPatImmOff<i32, sext_bin_rmw_16_32<rmw_16>, or_is_add, inst16_32>;
  defm : BinRMWPatImmOff<i64, sext_bin_rmw_8_64<rmw_8>, or_is_add, inst8_64>;
  defm : BinRMWPatImmOff<i64, sext_bin_rmw_16_64<rmw_16>, or_is_add, inst16_64>;

  // Truncating-extending binary RMWs with just a constant offset
  defm : BinRMWPatOffsetOnly<i32, zext_bin_rmw_8_32<rmw_8>, inst8_32>;
  defm : BinRMWPatOffsetOnly<i32, zext_bin_rmw_16_32<rmw_16>, inst16_32>;
  defm : BinRMWPatOffsetOnly<i64, zext_bin_rmw_8_64<rmw_8>, inst8_64>;
  defm : BinRMWPatOffsetOnly<i64, zext_bin_rmw_16_64<rmw_16>, inst16_64>;
  defm : BinRMWPatOffsetOnly<i64, zext_bin_rmw_32_64<rmw_32>, inst32_64>;

  defm : BinRMWPatOffsetOnly<i32, sext_bin_rmw_8_32<rmw_8>, inst8_32>;
  defm : BinRMWPatOffsetOnly<i32, sext_bin_rmw_16_32<rmw_16>, inst16_32>;
  defm : BinRMWPatOffsetOnly<i64, sext_bin_rmw_8_64<rmw_8>, inst8_64>;
  defm : BinRMWPatOffsetOnly<i64, sext_bin_rmw_16_64<rmw_16>, inst16_64>;

  defm : BinRMWPatGlobalAddrOffOnly<i32, zext_bin_rmw_8_32<rmw_8>, inst8_32>;
  defm : BinRMWPatGlobalAddrOffOnly<i32, zext_bin_rmw_16_32<rmw_16>, inst16_32>;
  defm : BinRMWPatGlobalAddrOffOnly<i64, zext_bin_rmw_8_64<rmw_8>, inst8_64>;
  defm : BinRMWPatGlobalAddrOffOnly<i64, zext_bin_rmw_16_64<rmw_16>, inst16_64>;
  defm : BinRMWPatGlobalAddrOffOnly<i64, zext_bin_rmw_32_64<rmw_32>, inst32_64>;

  defm : BinRMWPatGlobalAddrOffOnly<i32, sext_bin_rmw_8_32<rmw_8>, inst8_32>;
  defm : BinRMWPatGlobalAddrOffOnly<i32, sext_bin_rmw_16_32<rmw_16>, inst16_32>;
  defm : BinRMWPatGlobalAddrOffOnly<i64, sext_bin_rmw_8_64<rmw_8>, inst8_64>;
  defm : BinRMWPatGlobalAddrOffOnly<i64, sext_bin_rmw_16_64<rmw_16>, inst16_64>;
}

defm : BinRMWTruncExtPattern<
  atomic_load_add_8, atomic_load_add_16, atomic_load_add_32,
  "ATOMIC_RMW8_U_ADD_I32", "ATOMIC_RMW16_U_ADD_I32",
  "ATOMIC_RMW8_U_ADD_I64", "ATOMIC_RMW16_U_ADD_I64", "ATOMIC_RMW32_U_ADD_I64">;
defm : BinRMWTruncExtPattern<
  atomic_load_sub_8, atomic_load_sub_16, atomic_load_sub_32,
  "ATOMIC_RMW8_U_SUB_I32", "ATOMIC_RMW16_U_SUB_I32",
  "ATOMIC_RMW8_U_SUB_I64", "ATOMIC_RMW16_U_SUB_I64", "ATOMIC_RMW32_U_SUB_I64">;
defm : BinRMWTruncExtPattern<
  atomic_load_and_8, atomic_load_and_16, atomic_load_and_32,
  "ATOMIC_RMW8_U_AND_I32", "ATOMIC_RMW16_U_AND_I32",
  "ATOMIC_RMW8_U_AND_I64", "ATOMIC_RMW16_U_AND_I64", "ATOMIC_RMW32_U_AND_I64">;
defm : BinRMWTruncExtPattern<
  atomic_load_or_8, atomic_load_or_16, atomic_load_or_32,
  "ATOMIC_RMW8_U_OR_I32", "ATOMIC_RMW16_U_OR_I32",
  "ATOMIC_RMW8_U_OR_I64", "ATOMIC_RMW16_U_OR_I64", "ATOMIC_RMW32_U_OR_I64">;
defm : BinRMWTruncExtPattern<
  atomic_load_xor_8, atomic_load_xor_16, atomic_load_xor_32,
  "ATOMIC_RMW8_U_XOR_I32", "ATOMIC_RMW16_U_XOR_I32",
  "ATOMIC_RMW8_U_XOR_I64", "ATOMIC_RMW16_U_XOR_I64", "ATOMIC_RMW32_U_XOR_I64">;
defm : BinRMWTruncExtPattern<
  atomic_swap_8, atomic_swap_16, atomic_swap_32,
  "ATOMIC_RMW8_U_XCHG_I32", "ATOMIC_RMW16_U_XCHG_I32",
  "ATOMIC_RMW8_U_XCHG_I64", "ATOMIC_RMW16_U_XCHG_I64",
  "ATOMIC_RMW32_U_XCHG_I64">;

//===----------------------------------------------------------------------===//
// Atomic ternary read-modify-writes
//===----------------------------------------------------------------------===//

// TODO LLVM IR's cmpxchg instruction returns a pair of {loaded value, success
// flag}. When we use the success flag or both values, we can't make use of i64
// truncate/extend versions of instructions for now, which is suboptimal.
// Consider adding a pass after instruction selection that optimizes this case
// if it is frequent.

multiclass WebAssemblyTerRMW<WebAssemblyRegClass rc, string name,
                             int atomic_op> {
  defm "_A32" :
    ATOMIC_I<(outs rc:$dst),
             (ins P2Align:$p2align, offset32_op:$off, I32:$addr, rc:$exp,
                  rc:$new_),
             (outs), (ins P2Align:$p2align, offset32_op:$off), [],
             !strconcat(name, "\t$dst, ${off}(${addr})${p2align}, $exp, $new_"),
             !strconcat(name, "\t${off}${p2align}"), atomic_op, false>;
  defm "_A64" :
    ATOMIC_I<(outs rc:$dst),
             (ins P2Align:$p2align, offset64_op:$off, I64:$addr, rc:$exp,
                  rc:$new_),
             (outs), (ins P2Align:$p2align, offset64_op:$off), [],
             !strconcat(name, "\t$dst, ${off}(${addr})${p2align}, $exp, $new_"),
             !strconcat(name, "\t${off}${p2align}"), atomic_op, true>;
}

defm ATOMIC_RMW_CMPXCHG_I32 :
  WebAssemblyTerRMW<I32, "i32.atomic.rmw.cmpxchg", 0x48>;
defm ATOMIC_RMW_CMPXCHG_I64 :
  WebAssemblyTerRMW<I64, "i64.atomic.rmw.cmpxchg", 0x49>;
defm ATOMIC_RMW8_U_CMPXCHG_I32 :
  WebAssemblyTerRMW<I32, "i32.atomic.rmw8.cmpxchg_u", 0x4a>;
defm ATOMIC_RMW16_U_CMPXCHG_I32 :
  WebAssemblyTerRMW<I32, "i32.atomic.rmw16.cmpxchg_u", 0x4b>;
defm ATOMIC_RMW8_U_CMPXCHG_I64 :
  WebAssemblyTerRMW<I64, "i64.atomic.rmw8.cmpxchg_u", 0x4c>;
defm ATOMIC_RMW16_U_CMPXCHG_I64 :
  WebAssemblyTerRMW<I64, "i64.atomic.rmw16.cmpxchg_u", 0x4d>;
defm ATOMIC_RMW32_U_CMPXCHG_I64 :
  WebAssemblyTerRMW<I64, "i64.atomic.rmw32.cmpxchg_u", 0x4e>;

// Select ternary RMWs with no constant offset.
multiclass TerRMWPatNoOffset<ValueType ty, PatFrag kind, string inst> {
  def : Pat<(ty (kind I32:$addr, ty:$exp, ty:$new)),
            (!cast<NI>(inst#_A32) 0, 0, I32:$addr, ty:$exp, ty:$new)>,
        Requires<[HasAddr32, HasAtomics]>;
  def : Pat<(ty (kind I64:$addr, ty:$exp, ty:$new)),
            (!cast<NI>(inst#_A64) 0, 0, I64:$addr, ty:$exp, ty:$new)>,
        Requires<[HasAddr64, HasAtomics]>;
}

// Select ternary RMWs with a constant offset.

// Pattern with address + immediate offset
multiclass TerRMWPatImmOff<ValueType ty, PatFrag kind, PatFrag operand,
                           string inst> {
  def : Pat<(ty (kind (operand I32:$addr, imm:$off), ty:$exp, ty:$new)),
            (!cast<NI>(inst#_A32) 0, imm:$off, I32:$addr, ty:$exp, ty:$new)>,
        Requires<[HasAddr32, HasAtomics]>;
  def : Pat<(ty (kind (operand I64:$addr, imm:$off), ty:$exp, ty:$new)),
            (!cast<NI>(inst#_A64) 0, imm:$off, I64:$addr, ty:$exp, ty:$new)>,
        Requires<[HasAddr64, HasAtomics]>;
}

// Select ternary RMWs with just a constant offset.
multiclass TerRMWPatOffsetOnly<ValueType ty, PatFrag kind, string inst> {
  def : Pat<(ty (kind imm:$off, ty:$exp, ty:$new)),
            (!cast<NI>(inst#_A32) 0, imm:$off, (CONST_I32 0), ty:$exp,
              ty:$new)>;
  def : Pat<(ty (kind imm:$off, ty:$exp, ty:$new)),
            (!cast<NI>(inst#_A64) 0, imm:$off, (CONST_I64 0), ty:$exp,
              ty:$new)>;
}

multiclass TerRMWPatGlobalAddrOffOnly<ValueType ty, PatFrag kind, string inst> {
  def : Pat<(ty (kind (WebAssemblyWrapper tglobaladdr:$off), ty:$exp, ty:$new)),
            (!cast<NI>(inst#_A32) 0, tglobaladdr:$off, (CONST_I32 0), ty:$exp,
              ty:$new)>,
        Requires<[HasAddr32, HasAtomics, IsNotPIC]>;
  def : Pat<(ty (kind (WebAssemblyWrapper tglobaladdr:$off), ty:$exp, ty:$new)),
            (!cast<NI>(inst#_A64) 0, tglobaladdr:$off, (CONST_I64 0), ty:$exp,
              ty:$new)>,
        Requires<[HasAddr64, HasAtomics, IsNotPIC]>;
}

// Patterns for various addressing modes.
multiclass TerRMWPattern<PatFrag rmw_32, PatFrag rmw_64, string inst_32,
                         string inst_64> {
  defm : TerRMWPatNoOffset<i32, rmw_32, inst_32>;
  defm : TerRMWPatNoOffset<i64, rmw_64, inst_64>;

  defm : TerRMWPatImmOff<i32, rmw_32, regPlusImm, inst_32>;
  defm : TerRMWPatImmOff<i64, rmw_64, regPlusImm, inst_64>;
  defm : TerRMWPatImmOff<i32, rmw_32, or_is_add, inst_32>;
  defm : TerRMWPatImmOff<i64, rmw_64, or_is_add, inst_64>;

  defm : TerRMWPatOffsetOnly<i32, rmw_32, inst_32>;
  defm : TerRMWPatOffsetOnly<i64, rmw_64, inst_64>;

  defm : TerRMWPatGlobalAddrOffOnly<i32, rmw_32, inst_32>;
  defm : TerRMWPatGlobalAddrOffOnly<i64, rmw_64, inst_64>;
}

defm : TerRMWPattern<atomic_cmp_swap_32, atomic_cmp_swap_64,
                     "ATOMIC_RMW_CMPXCHG_I32", "ATOMIC_RMW_CMPXCHG_I64">;

// Truncating & zero-extending ternary RMW patterns.
// DAG legalization & optimization before instruction selection may introduce
// additional nodes such as anyext or assertzext depending on operand types.
class zext_ter_rmw_8_32<PatFrag kind> :
  PatFrag<(ops node:$addr, node:$exp, node:$new),
          (i32 (kind node:$addr, node:$exp, node:$new))>;
class zext_ter_rmw_16_32<PatFrag kind> : zext_ter_rmw_8_32<kind>;
class zext_ter_rmw_8_64<PatFrag kind> :
  PatFrag<(ops node:$addr, node:$exp, node:$new),
          (zext (i32 (assertzext (i32 (kind node:$addr,
                                            (i32 (trunc (i64 node:$exp))),
                                            (i32 (trunc (i64 node:$new))))))))>;
class zext_ter_rmw_16_64<PatFrag kind> : zext_ter_rmw_8_64<kind>;
class zext_ter_rmw_32_64<PatFrag kind> :
  PatFrag<(ops node:$addr, node:$exp, node:$new),
          (zext (i32 (kind node:$addr,
                           (i32 (trunc (i64 node:$exp))),
                           (i32 (trunc (i64 node:$new))))))>;

// Truncating & sign-extending ternary RMW patterns.
// We match subword RMWs (for 32-bit) and anyext RMWs (for 64-bit) and select a
// zext RMW; the next instruction will be sext_inreg which is selected by
// itself.
class sext_ter_rmw_8_32<PatFrag kind> :
  PatFrag<(ops node:$addr, node:$exp, node:$new),
          (kind node:$addr, node:$exp, node:$new)>;
class sext_ter_rmw_16_32<PatFrag kind> : sext_ter_rmw_8_32<kind>;
class sext_ter_rmw_8_64<PatFrag kind> :
  PatFrag<(ops node:$addr, node:$exp, node:$new),
          (anyext (i32 (assertzext (i32
            (kind node:$addr,
                  (i32 (trunc (i64 node:$exp))),
                  (i32 (trunc (i64 node:$new))))))))>;
class sext_ter_rmw_16_64<PatFrag kind> : sext_ter_rmw_8_64<kind>;
// 32->64 sext RMW gets selected as i32.atomic.rmw.***, i64.extend_i32_s

// Patterns for various addressing modes for truncating-extending ternary RMWs.
multiclass TerRMWTruncExtPattern<
  PatFrag rmw_8, PatFrag rmw_16, PatFrag rmw_32,
  string inst8_32, string inst16_32, string inst8_64, string inst16_64,
  string inst32_64> {
  // Truncating-extending ternary RMWs with no constant offset
  defm : TerRMWPatNoOffset<i32, zext_ter_rmw_8_32<rmw_8>, inst8_32>;
  defm : TerRMWPatNoOffset<i32, zext_ter_rmw_16_32<rmw_16>, inst16_32>;
  defm : TerRMWPatNoOffset<i64, zext_ter_rmw_8_64<rmw_8>, inst8_64>;
  defm : TerRMWPatNoOffset<i64, zext_ter_rmw_16_64<rmw_16>, inst16_64>;
  defm : TerRMWPatNoOffset<i64, zext_ter_rmw_32_64<rmw_32>, inst32_64>;

  defm : TerRMWPatNoOffset<i32, sext_ter_rmw_8_32<rmw_8>, inst8_32>;
  defm : TerRMWPatNoOffset<i32, sext_ter_rmw_16_32<rmw_16>, inst16_32>;
  defm : TerRMWPatNoOffset<i64, sext_ter_rmw_8_64<rmw_8>, inst8_64>;
  defm : TerRMWPatNoOffset<i64, sext_ter_rmw_16_64<rmw_16>, inst16_64>;

  // Truncating-extending ternary RMWs with a constant offset
  defm : TerRMWPatImmOff<i32, zext_ter_rmw_8_32<rmw_8>, regPlusImm, inst8_32>;
  defm : TerRMWPatImmOff<i32, zext_ter_rmw_16_32<rmw_16>, regPlusImm,
                         inst16_32>;
  defm : TerRMWPatImmOff<i64, zext_ter_rmw_8_64<rmw_8>, regPlusImm, inst8_64>;
  defm : TerRMWPatImmOff<i64, zext_ter_rmw_16_64<rmw_16>, regPlusImm,
                         inst16_64>;
  defm : TerRMWPatImmOff<i64, zext_ter_rmw_32_64<rmw_32>, regPlusImm,
                         inst32_64>;
  defm : TerRMWPatImmOff<i32, zext_ter_rmw_8_32<rmw_8>, or_is_add, inst8_32>;
  defm : TerRMWPatImmOff<i32, zext_ter_rmw_16_32<rmw_16>, or_is_add, inst16_32>;
  defm : TerRMWPatImmOff<i64, zext_ter_rmw_8_64<rmw_8>, or_is_add, inst8_64>;
  defm : TerRMWPatImmOff<i64, zext_ter_rmw_16_64<rmw_16>, or_is_add, inst16_64>;
  defm : TerRMWPatImmOff<i64, zext_ter_rmw_32_64<rmw_32>, or_is_add, inst32_64>;

  defm : TerRMWPatImmOff<i32, sext_ter_rmw_8_32<rmw_8>, regPlusImm, inst8_32>;
  defm : TerRMWPatImmOff<i32, sext_ter_rmw_16_32<rmw_16>, regPlusImm,
                         inst16_32>;
  defm : TerRMWPatImmOff<i64, sext_ter_rmw_8_64<rmw_8>, regPlusImm, inst8_64>;
  defm : TerRMWPatImmOff<i64, sext_ter_rmw_16_64<rmw_16>, regPlusImm,
                         inst16_64>;
  defm : TerRMWPatImmOff<i32, sext_ter_rmw_8_32<rmw_8>, or_is_add, inst8_32>;
  defm : TerRMWPatImmOff<i32, sext_ter_rmw_16_32<rmw_16>, or_is_add, inst16_32>;
  defm : TerRMWPatImmOff<i64, sext_ter_rmw_8_64<rmw_8>, or_is_add, inst8_64>;
  defm : TerRMWPatImmOff<i64, sext_ter_rmw_16_64<rmw_16>, or_is_add, inst16_64>;

  // Truncating-extending ternary RMWs with just a constant offset
  defm : TerRMWPatOffsetOnly<i32, zext_ter_rmw_8_32<rmw_8>, inst8_32>;
  defm : TerRMWPatOffsetOnly<i32, zext_ter_rmw_16_32<rmw_16>, inst16_32>;
  defm : TerRMWPatOffsetOnly<i64, zext_ter_rmw_8_64<rmw_8>, inst8_64>;
  defm : TerRMWPatOffsetOnly<i64, zext_ter_rmw_16_64<rmw_16>, inst16_64>;
  defm : TerRMWPatOffsetOnly<i64, zext_ter_rmw_32_64<rmw_32>, inst32_64>;

  defm : TerRMWPatOffsetOnly<i32, sext_ter_rmw_8_32<rmw_8>, inst8_32>;
  defm : TerRMWPatOffsetOnly<i32, sext_ter_rmw_16_32<rmw_16>, inst16_32>;
  defm : TerRMWPatOffsetOnly<i64, sext_ter_rmw_8_64<rmw_8>, inst8_64>;
  defm : TerRMWPatOffsetOnly<i64, sext_ter_rmw_16_64<rmw_16>, inst16_64>;

  defm : TerRMWPatGlobalAddrOffOnly<i32, zext_ter_rmw_8_32<rmw_8>, inst8_32>;
  defm : TerRMWPatGlobalAddrOffOnly<i32, zext_ter_rmw_16_32<rmw_16>, inst16_32>;
  defm : TerRMWPatGlobalAddrOffOnly<i64, zext_ter_rmw_8_64<rmw_8>, inst8_64>;
  defm : TerRMWPatGlobalAddrOffOnly<i64, zext_ter_rmw_16_64<rmw_16>, inst16_64>;
  defm : TerRMWPatGlobalAddrOffOnly<i64, zext_ter_rmw_32_64<rmw_32>, inst32_64>;

  defm : TerRMWPatGlobalAddrOffOnly<i32, sext_ter_rmw_8_32<rmw_8>, inst8_32>;
  defm : TerRMWPatGlobalAddrOffOnly<i32, sext_ter_rmw_16_32<rmw_16>, inst16_32>;
  defm : TerRMWPatGlobalAddrOffOnly<i64, sext_ter_rmw_8_64<rmw_8>, inst8_64>;
  defm : TerRMWPatGlobalAddrOffOnly<i64, sext_ter_rmw_16_64<rmw_16>, inst16_64>;
}

defm : TerRMWTruncExtPattern<
  atomic_cmp_swap_8, atomic_cmp_swap_16, atomic_cmp_swap_32,
  "ATOMIC_RMW8_U_CMPXCHG_I32", "ATOMIC_RMW16_U_CMPXCHG_I32",
  "ATOMIC_RMW8_U_CMPXCHG_I64", "ATOMIC_RMW16_U_CMPXCHG_I64",
  "ATOMIC_RMW32_U_CMPXCHG_I64">;
