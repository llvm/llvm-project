// WebAssemblyInstrMemory.td-WebAssembly Memory codegen support -*- tablegen -*-
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// WebAssembly Memory operand code-gen constructs.
///
//===----------------------------------------------------------------------===//

// TODO:
//  - WebAssemblyTargetLowering having to do with atomics
//  - Each has optional alignment.

// WebAssembly has i8/i16/i32/i64/f32/f64 memory types, but doesn't have i8/i16
// local types. These memory-only types instead zero- or sign-extend into local
// types when loading, and truncate when storing.

// WebAssembly constant offsets are performed as unsigned with infinite
// precision, so we need to check for NoUnsignedWrap so that we don't fold an
// offset for an add that needs wrapping.
def regPlusImm : PatFrag<(ops node:$addr, node:$off),
                         (add node:$addr, node:$off),
                         [{ return N->getFlags().hasNoUnsignedWrap(); }]>;

// Treat an 'or' node as an 'add' if the or'ed bits are known to be zero.
def or_is_add : PatFrag<(ops node:$lhs, node:$rhs), (or node:$lhs, node:$rhs),[{
  if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N->getOperand(1)))
    return CurDAG->MaskedValueIsZero(N->getOperand(0), CN->getAPIntValue());

  KnownBits Known0 = CurDAG->computeKnownBits(N->getOperand(0), 0);
  KnownBits Known1 = CurDAG->computeKnownBits(N->getOperand(1), 0);
  return (~Known0.Zero & ~Known1.Zero) == 0;
}]>;

// We don't need a regPlusES because external symbols never have constant
// offsets folded into them, so we can just use add.

// Defines atomic and non-atomic loads, regular and extending.
multiclass WebAssemblyLoad<WebAssemblyRegClass rc, string Name, int Opcode,
                           list<Predicate> reqs = []> {
  let mayLoad = 1, UseNamedOperandTable = 1 in {
  defm "_A32": I<(outs rc:$dst),
                 (ins P2Align:$p2align, offset32_op:$off, I32:$addr),
                 (outs), (ins P2Align:$p2align, offset32_op:$off),
                 [], !strconcat(Name, "\t$dst, ${off}(${addr})${p2align}"),
                 !strconcat(Name, "\t${off}${p2align}"), Opcode, false>,
               Requires<reqs>;
  defm "_A64": I<(outs rc:$dst),
                 (ins P2Align:$p2align, offset64_op:$off, I64:$addr),
                 (outs), (ins P2Align:$p2align, offset64_op:$off),
                 [], !strconcat(Name, "\t$dst, ${off}(${addr})${p2align}"),
                 !strconcat(Name, "\t${off}${p2align}"), Opcode, true>,
               Requires<reqs>;
  }
}

// Basic load.
// FIXME: When we can break syntax compatibility, reorder the fields in the
// asmstrings to match the binary encoding.
defm LOAD_I32 : WebAssemblyLoad<I32, "i32.load", 0x28, []>;
defm LOAD_I64 : WebAssemblyLoad<I64, "i64.load", 0x29, []>;
defm LOAD_F32 : WebAssemblyLoad<F32, "f32.load", 0x2a, []>;
defm LOAD_F64 : WebAssemblyLoad<F64, "f64.load", 0x2b, []>;

// Select loads with no constant offset.
multiclass LoadPatNoOffset<ValueType ty, SDPatternOperator kind, string inst> {
  def : Pat<(ty (kind I32:$addr)), (!cast<NI>(inst # "_A32") 0, 0, I32:$addr)>,
        Requires<[HasAddr32]>;
  def : Pat<(ty (kind (i64 I64:$addr))), (!cast<NI>(inst # "_A64") 0, 0, I64:$addr)>,
        Requires<[HasAddr64]>;
}

defm : LoadPatNoOffset<i32, load, "LOAD_I32">;
defm : LoadPatNoOffset<i64, load, "LOAD_I64">;
defm : LoadPatNoOffset<f32, load, "LOAD_F32">;
defm : LoadPatNoOffset<f64, load, "LOAD_F64">;

// Select loads with a constant offset.

// Pattern with address + immediate offset
multiclass LoadPatImmOff<ValueType ty, SDPatternOperator kind, PatFrag operand,
                         string inst> {
  def : Pat<(ty (kind (operand I32:$addr, imm:$off))),
            (!cast<NI>(inst # "_A32") 0, imm:$off, I32:$addr)>,
        Requires<[HasAddr32]>;
  def : Pat<(ty (kind (operand I64:$addr, imm:$off))),
            (!cast<NI>(inst # "_A64") 0, imm:$off, I64:$addr)>,
        Requires<[HasAddr64]>;
}

defm : LoadPatImmOff<i32, load, regPlusImm, "LOAD_I32">;
defm : LoadPatImmOff<i64, load, regPlusImm, "LOAD_I64">;
defm : LoadPatImmOff<f32, load, regPlusImm, "LOAD_F32">;
defm : LoadPatImmOff<f64, load, regPlusImm, "LOAD_F64">;
defm : LoadPatImmOff<i32, load, or_is_add, "LOAD_I32">;
defm : LoadPatImmOff<i64, load, or_is_add, "LOAD_I64">;
defm : LoadPatImmOff<f32, load, or_is_add, "LOAD_F32">;
defm : LoadPatImmOff<f64, load, or_is_add, "LOAD_F64">;

// Select loads with just a constant offset.
multiclass LoadPatOffsetOnly<ValueType ty, SDPatternOperator kind, string inst> {
  def : Pat<(ty (kind imm:$off)),
            (!cast<NI>(inst # "_A32") 0, imm:$off, (CONST_I32 0))>,
        Requires<[HasAddr32]>;
  def : Pat<(ty (kind imm:$off)),
            (!cast<NI>(inst # "_A64") 0, imm:$off, (CONST_I64 0))>,
        Requires<[HasAddr64]>;
}

defm : LoadPatOffsetOnly<i32, load, "LOAD_I32">;
defm : LoadPatOffsetOnly<i64, load, "LOAD_I64">;
defm : LoadPatOffsetOnly<f32, load, "LOAD_F32">;
defm : LoadPatOffsetOnly<f64, load, "LOAD_F64">;

multiclass LoadPatGlobalAddrOffOnly<ValueType ty, SDPatternOperator kind, string inst> {
  def : Pat<(ty (kind (WebAssemblyWrapper tglobaladdr:$off))),
            (!cast<NI>(inst # "_A32") 0, tglobaladdr:$off, (CONST_I32 0))>,
        Requires<[IsNotPIC, HasAddr32]>;
  def : Pat<(ty (kind (WebAssemblyWrapper tglobaladdr:$off))),
            (!cast<NI>(inst # "_A64") 0, tglobaladdr:$off, (CONST_I64 0))>,
        Requires<[IsNotPIC, HasAddr64]>;
}

defm : LoadPatGlobalAddrOffOnly<i32, load, "LOAD_I32">;
defm : LoadPatGlobalAddrOffOnly<i64, load, "LOAD_I64">;
defm : LoadPatGlobalAddrOffOnly<f32, load, "LOAD_F32">;
defm : LoadPatGlobalAddrOffOnly<f64, load, "LOAD_F64">;

// Extending load.
defm LOAD8_S_I32 : WebAssemblyLoad<I32, "i32.load8_s", 0x2c, []>;
defm LOAD8_U_I32 : WebAssemblyLoad<I32, "i32.load8_u", 0x2d, []>;
defm LOAD16_S_I32 : WebAssemblyLoad<I32, "i32.load16_s", 0x2e, []>;
defm LOAD16_U_I32 : WebAssemblyLoad<I32, "i32.load16_u", 0x2f, []>;
defm LOAD8_S_I64 : WebAssemblyLoad<I64, "i64.load8_s", 0x30, []>;
defm LOAD8_U_I64 : WebAssemblyLoad<I64, "i64.load8_u", 0x31, []>;
defm LOAD16_S_I64 : WebAssemblyLoad<I64, "i64.load16_s", 0x32, []>;
defm LOAD16_U_I64 : WebAssemblyLoad<I64, "i64.load16_u", 0x33, []>;
defm LOAD32_S_I64 : WebAssemblyLoad<I64, "i64.load32_s", 0x34, []>;
defm LOAD32_U_I64 : WebAssemblyLoad<I64, "i64.load32_u", 0x35, []>;

// Select extending loads with no constant offset.
defm : LoadPatNoOffset<i32, sextloadi8, "LOAD8_S_I32">;
defm : LoadPatNoOffset<i32, zextloadi8, "LOAD8_U_I32">;
defm : LoadPatNoOffset<i32, sextloadi16, "LOAD16_S_I32">;
defm : LoadPatNoOffset<i32, zextloadi16, "LOAD16_U_I32">;
defm : LoadPatNoOffset<i64, sextloadi8, "LOAD8_S_I64">;
defm : LoadPatNoOffset<i64, zextloadi8, "LOAD8_U_I64">;
defm : LoadPatNoOffset<i64, sextloadi16, "LOAD16_S_I64">;
defm : LoadPatNoOffset<i64, zextloadi16, "LOAD16_U_I64">;
defm : LoadPatNoOffset<i64, sextloadi32, "LOAD32_S_I64">;
defm : LoadPatNoOffset<i64, zextloadi32, "LOAD32_U_I64">;

// Select extending loads with a constant offset.
defm : LoadPatImmOff<i32, sextloadi8, regPlusImm, "LOAD8_S_I32">;
defm : LoadPatImmOff<i32, zextloadi8, regPlusImm, "LOAD8_U_I32">;
defm : LoadPatImmOff<i32, sextloadi16, regPlusImm, "LOAD16_S_I32">;
defm : LoadPatImmOff<i32, zextloadi16, regPlusImm, "LOAD16_U_I32">;
defm : LoadPatImmOff<i64, sextloadi8, regPlusImm, "LOAD8_S_I64">;
defm : LoadPatImmOff<i64, zextloadi8, regPlusImm, "LOAD8_U_I64">;
defm : LoadPatImmOff<i64, sextloadi16, regPlusImm, "LOAD16_S_I64">;
defm : LoadPatImmOff<i64, zextloadi16, regPlusImm, "LOAD16_U_I64">;
defm : LoadPatImmOff<i64, sextloadi32, regPlusImm, "LOAD32_S_I64">;
defm : LoadPatImmOff<i64, zextloadi32, regPlusImm, "LOAD32_U_I64">;

defm : LoadPatImmOff<i32, sextloadi8, or_is_add, "LOAD8_S_I32">;
defm : LoadPatImmOff<i32, zextloadi8, or_is_add, "LOAD8_U_I32">;
defm : LoadPatImmOff<i32, sextloadi16, or_is_add, "LOAD16_S_I32">;
defm : LoadPatImmOff<i32, zextloadi16, or_is_add, "LOAD16_U_I32">;
defm : LoadPatImmOff<i64, sextloadi8, or_is_add, "LOAD8_S_I64">;
defm : LoadPatImmOff<i64, zextloadi8, or_is_add, "LOAD8_U_I64">;
defm : LoadPatImmOff<i64, sextloadi16, or_is_add, "LOAD16_S_I64">;
defm : LoadPatImmOff<i64, zextloadi16, or_is_add, "LOAD16_U_I64">;
defm : LoadPatImmOff<i64, sextloadi32, or_is_add, "LOAD32_S_I64">;
defm : LoadPatImmOff<i64, zextloadi32, or_is_add, "LOAD32_U_I64">;

// Select extending loads with just a constant offset.
defm : LoadPatOffsetOnly<i32, sextloadi8, "LOAD8_S_I32">;
defm : LoadPatOffsetOnly<i32, zextloadi8, "LOAD8_U_I32">;
defm : LoadPatOffsetOnly<i32, sextloadi16, "LOAD16_S_I32">;
defm : LoadPatOffsetOnly<i32, zextloadi16, "LOAD16_U_I32">;

defm : LoadPatOffsetOnly<i64, sextloadi8, "LOAD8_S_I64">;
defm : LoadPatOffsetOnly<i64, zextloadi8, "LOAD8_U_I64">;
defm : LoadPatOffsetOnly<i64, sextloadi16, "LOAD16_S_I64">;
defm : LoadPatOffsetOnly<i64, zextloadi16, "LOAD16_U_I64">;
defm : LoadPatOffsetOnly<i64, sextloadi32, "LOAD32_S_I64">;
defm : LoadPatOffsetOnly<i64, zextloadi32, "LOAD32_U_I64">;

defm : LoadPatGlobalAddrOffOnly<i32, sextloadi8, "LOAD8_S_I32">;
defm : LoadPatGlobalAddrOffOnly<i32, zextloadi8, "LOAD8_U_I32">;
defm : LoadPatGlobalAddrOffOnly<i32, sextloadi16, "LOAD16_S_I32">;
defm : LoadPatGlobalAddrOffOnly<i32, zextloadi16, "LOAD16_U_I32">;
defm : LoadPatGlobalAddrOffOnly<i64, sextloadi8, "LOAD8_S_I64">;
defm : LoadPatGlobalAddrOffOnly<i64, zextloadi8, "LOAD8_U_I64">;
defm : LoadPatGlobalAddrOffOnly<i64, sextloadi16, "LOAD16_S_I64">;
defm : LoadPatGlobalAddrOffOnly<i64, zextloadi16, "LOAD16_U_I64">;
defm : LoadPatGlobalAddrOffOnly<i64, sextloadi32, "LOAD32_S_I64">;
defm : LoadPatGlobalAddrOffOnly<i64, zextloadi32, "LOAD32_U_I64">;

// Resolve "don't care" extending loads to zero-extending loads. This is
// somewhat arbitrary, but zero-extending is conceptually simpler.

// Select "don't care" extending loads with no constant offset.
defm : LoadPatNoOffset<i32, extloadi8, "LOAD8_U_I32">;
defm : LoadPatNoOffset<i32, extloadi16, "LOAD16_U_I32">;
defm : LoadPatNoOffset<i64, extloadi8, "LOAD8_U_I64">;
defm : LoadPatNoOffset<i64, extloadi16, "LOAD16_U_I64">;
defm : LoadPatNoOffset<i64, extloadi32, "LOAD32_U_I64">;

// Select "don't care" extending loads with a constant offset.
defm : LoadPatImmOff<i32, extloadi8, regPlusImm, "LOAD8_U_I32">;
defm : LoadPatImmOff<i32, extloadi16, regPlusImm, "LOAD16_U_I32">;
defm : LoadPatImmOff<i64, extloadi8, regPlusImm, "LOAD8_U_I64">;
defm : LoadPatImmOff<i64, extloadi16, regPlusImm, "LOAD16_U_I64">;
defm : LoadPatImmOff<i64, extloadi32, regPlusImm, "LOAD32_U_I64">;
defm : LoadPatImmOff<i32, extloadi8, or_is_add, "LOAD8_U_I32">;
defm : LoadPatImmOff<i32, extloadi16, or_is_add, "LOAD16_U_I32">;
defm : LoadPatImmOff<i64, extloadi8, or_is_add, "LOAD8_U_I64">;
defm : LoadPatImmOff<i64, extloadi16, or_is_add, "LOAD16_U_I64">;
defm : LoadPatImmOff<i64, extloadi32, or_is_add, "LOAD32_U_I64">;

// Select "don't care" extending loads with just a constant offset.
defm : LoadPatOffsetOnly<i32, extloadi8, "LOAD8_U_I32">;
defm : LoadPatOffsetOnly<i32, extloadi16, "LOAD16_U_I32">;
defm : LoadPatOffsetOnly<i64, extloadi8, "LOAD8_U_I64">;
defm : LoadPatOffsetOnly<i64, extloadi16, "LOAD16_U_I64">;
defm : LoadPatOffsetOnly<i64, extloadi32, "LOAD32_U_I64">;
defm : LoadPatGlobalAddrOffOnly<i32, extloadi8, "LOAD8_U_I32">;
defm : LoadPatGlobalAddrOffOnly<i32, extloadi16, "LOAD16_U_I32">;
defm : LoadPatGlobalAddrOffOnly<i64, extloadi8, "LOAD8_U_I64">;
defm : LoadPatGlobalAddrOffOnly<i64, extloadi16, "LOAD16_U_I64">;
defm : LoadPatGlobalAddrOffOnly<i64, extloadi32, "LOAD32_U_I64">;

// Defines atomic and non-atomic stores, regular and truncating
multiclass WebAssemblyStore<WebAssemblyRegClass rc, string Name, int Opcode,
                            list<Predicate> reqs = []> {
  let mayStore = 1, UseNamedOperandTable = 1 in
  defm "_A32" : I<(outs),
                  (ins P2Align:$p2align, offset32_op:$off, I32:$addr, rc:$val),
                  (outs),
                  (ins P2Align:$p2align, offset32_op:$off), [],
                  !strconcat(Name, "\t${off}(${addr})${p2align}, $val"),
                  !strconcat(Name, "\t${off}${p2align}"), Opcode, false>,
                Requires<reqs>;
  let mayStore = 1, UseNamedOperandTable = 1 in
  defm "_A64" : I<(outs),
                  (ins P2Align:$p2align, offset64_op:$off, I64:$addr, rc:$val),
                  (outs),
                  (ins P2Align:$p2align, offset64_op:$off), [],
                  !strconcat(Name, "\t${off}(${addr})${p2align}, $val"),
                  !strconcat(Name, "\t${off}${p2align}"), Opcode, true>,
                Requires<reqs>;
}

// Basic store.
// Note: WebAssembly inverts SelectionDAG's usual operand order.
defm STORE_I32  : WebAssemblyStore<I32, "i32.store", 0x36>;
defm STORE_I64  : WebAssemblyStore<I64, "i64.store", 0x37>;
defm STORE_F32  : WebAssemblyStore<F32, "f32.store", 0x38>;
defm STORE_F64  : WebAssemblyStore<F64, "f64.store", 0x39>;

// Select stores with no constant offset.
multiclass StorePatNoOffset<ValueType ty, PatFrag node, string inst> {
  def : Pat<(node ty:$val, I32:$addr),
            (!cast<NI>(inst # "_A32") 0, 0, I32:$addr, ty:$val)>,
        Requires<[HasAddr32]>;
  def : Pat<(node ty:$val, I64:$addr),
            (!cast<NI>(inst # "_A64") 0, 0, I64:$addr, ty:$val)>,
        Requires<[HasAddr64]>;
}

defm : StorePatNoOffset<i32, store, "STORE_I32">;
defm : StorePatNoOffset<i64, store, "STORE_I64">;
defm : StorePatNoOffset<f32, store, "STORE_F32">;
defm : StorePatNoOffset<f64, store, "STORE_F64">;

// Select stores with a constant offset.
multiclass StorePatImmOff<ValueType ty, PatFrag kind, PatFrag operand,
                          string inst> {
  def : Pat<(kind ty:$val, (operand I32:$addr, imm:$off)),
            (!cast<NI>(inst # "_A32") 0, imm:$off, I32:$addr, ty:$val)>,
        Requires<[HasAddr32]>;
  def : Pat<(kind ty:$val, (operand I64:$addr, imm:$off)),
            (!cast<NI>(inst # "_A64") 0, imm:$off, I64:$addr, ty:$val)>,
        Requires<[HasAddr64]>;
}

defm : StorePatImmOff<i32, store, regPlusImm, "STORE_I32">;
defm : StorePatImmOff<i64, store, regPlusImm, "STORE_I64">;
defm : StorePatImmOff<f32, store, regPlusImm, "STORE_F32">;
defm : StorePatImmOff<f64, store, regPlusImm, "STORE_F64">;
defm : StorePatImmOff<i32, store, or_is_add, "STORE_I32">;
defm : StorePatImmOff<i64, store, or_is_add, "STORE_I64">;
defm : StorePatImmOff<f32, store, or_is_add, "STORE_F32">;
defm : StorePatImmOff<f64, store, or_is_add, "STORE_F64">;

// Select stores with just a constant offset.
multiclass StorePatOffsetOnly<ValueType ty, PatFrag kind, string inst> {
  def : Pat<(kind ty:$val, imm:$off),
            (!cast<NI>(inst # "_A32") 0, imm:$off, (CONST_I32 0), ty:$val)>,
        Requires<[HasAddr32]>;
  def : Pat<(kind ty:$val, imm:$off),
            (!cast<NI>(inst # "_A64") 0, imm:$off, (CONST_I64 0), ty:$val)>,
        Requires<[HasAddr64]>;
}
defm : StorePatOffsetOnly<i32, store, "STORE_I32">;
defm : StorePatOffsetOnly<i64, store, "STORE_I64">;
defm : StorePatOffsetOnly<f32, store, "STORE_F32">;
defm : StorePatOffsetOnly<f64, store, "STORE_F64">;

multiclass StorePatGlobalAddrOffOnly<ValueType ty, PatFrag kind, string inst> {
  def : Pat<(kind ty:$val, (WebAssemblyWrapper tglobaladdr:$off)),
            (!cast<NI>(inst # "_A32") 0, tglobaladdr:$off, (CONST_I32 0),
             ty:$val)>,
        Requires<[IsNotPIC, HasAddr32]>;
  def : Pat<(kind ty:$val, (WebAssemblyWrapper tglobaladdr:$off)),
            (!cast<NI>(inst # "_A64") 0, tglobaladdr:$off, (CONST_I64 0),
             ty:$val)>,
        Requires<[IsNotPIC, HasAddr64]>;
}
defm : StorePatGlobalAddrOffOnly<i32, store, "STORE_I32">;
defm : StorePatGlobalAddrOffOnly<i64, store, "STORE_I64">;
defm : StorePatGlobalAddrOffOnly<f32, store, "STORE_F32">;
defm : StorePatGlobalAddrOffOnly<f64, store, "STORE_F64">;

// Truncating store.
defm STORE8_I32 : WebAssemblyStore<I32, "i32.store8", 0x3a>;
defm STORE16_I32 : WebAssemblyStore<I32, "i32.store16", 0x3b>;
defm STORE8_I64 : WebAssemblyStore<I64, "i64.store8", 0x3c>;
defm STORE16_I64 : WebAssemblyStore<I64, "i64.store16", 0x3d>;
defm STORE32_I64 : WebAssemblyStore<I64, "i64.store32", 0x3e>;

// Select truncating stores with no constant offset.
defm : StorePatNoOffset<i32, truncstorei8, "STORE8_I32">;
defm : StorePatNoOffset<i32, truncstorei16, "STORE16_I32">;
defm : StorePatNoOffset<i64, truncstorei8, "STORE8_I64">;
defm : StorePatNoOffset<i64, truncstorei16, "STORE16_I64">;
defm : StorePatNoOffset<i64, truncstorei32, "STORE32_I64">;

// Select truncating stores with a constant offset.
defm : StorePatImmOff<i32, truncstorei8, regPlusImm, "STORE8_I32">;
defm : StorePatImmOff<i32, truncstorei16, regPlusImm, "STORE16_I32">;
defm : StorePatImmOff<i64, truncstorei8, regPlusImm, "STORE8_I64">;
defm : StorePatImmOff<i64, truncstorei16, regPlusImm, "STORE16_I64">;
defm : StorePatImmOff<i64, truncstorei32, regPlusImm, "STORE32_I64">;
defm : StorePatImmOff<i32, truncstorei8, or_is_add, "STORE8_I32">;
defm : StorePatImmOff<i32, truncstorei16, or_is_add, "STORE16_I32">;
defm : StorePatImmOff<i64, truncstorei8, or_is_add, "STORE8_I64">;
defm : StorePatImmOff<i64, truncstorei16, or_is_add, "STORE16_I64">;
defm : StorePatImmOff<i64, truncstorei32, or_is_add, "STORE32_I64">;

// Select truncating stores with just a constant offset.
defm : StorePatOffsetOnly<i32, truncstorei8, "STORE8_I32">;
defm : StorePatOffsetOnly<i32, truncstorei16, "STORE16_I32">;
defm : StorePatOffsetOnly<i64, truncstorei8, "STORE8_I64">;
defm : StorePatOffsetOnly<i64, truncstorei16, "STORE16_I64">;
defm : StorePatOffsetOnly<i64, truncstorei32, "STORE32_I64">;
defm : StorePatGlobalAddrOffOnly<i32, truncstorei8, "STORE8_I32">;
defm : StorePatGlobalAddrOffOnly<i32, truncstorei16, "STORE16_I32">;
defm : StorePatGlobalAddrOffOnly<i64, truncstorei8, "STORE8_I64">;
defm : StorePatGlobalAddrOffOnly<i64, truncstorei16, "STORE16_I64">;
defm : StorePatGlobalAddrOffOnly<i64, truncstorei32, "STORE32_I64">;

multiclass MemoryOps<WebAssemblyRegClass rc, string B> {
// Current memory size.
defm MEMORY_SIZE_A#B : I<(outs rc:$dst), (ins i32imm:$flags),
                         (outs), (ins i32imm:$flags),
                         [(set rc:$dst,
                           (int_wasm_memory_size (i32 imm:$flags)))],
                         "memory.size\t$dst, $flags", "memory.size\t$flags",
                         0x3f>;

// Grow memory.
defm MEMORY_GROW_A#B : I<(outs rc:$dst), (ins i32imm:$flags, rc:$delta),
                         (outs), (ins i32imm:$flags),
                         [(set rc:$dst,
                           (int_wasm_memory_grow (i32 imm:$flags),
                             rc:$delta))],
                         "memory.grow\t$dst, $flags, $delta",
                         "memory.grow\t$flags", 0x40>;
}

defm : MemoryOps<I32, "32">;
defm : MemoryOps<I64, "64">;
