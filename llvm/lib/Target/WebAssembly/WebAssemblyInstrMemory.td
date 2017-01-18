// WebAssemblyInstrMemory.td-WebAssembly Memory codegen support -*- tablegen -*-
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief WebAssembly Memory operand code-gen constructs.
///
//===----------------------------------------------------------------------===//

// TODO:
//  - HasAddr64
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
                         [{ return N->getFlags()->hasNoUnsignedWrap(); }]>;

// Treat an 'or' node as an 'add' if the or'ed bits are known to be zero.
def or_is_add : PatFrag<(ops node:$lhs, node:$rhs), (or node:$lhs, node:$rhs),[{
  if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N->getOperand(1)))
    return CurDAG->MaskedValueIsZero(N->getOperand(0), CN->getAPIntValue());

  APInt KnownZero0, KnownOne0;
  CurDAG->computeKnownBits(N->getOperand(0), KnownZero0, KnownOne0, 0);
  APInt KnownZero1, KnownOne1;
  CurDAG->computeKnownBits(N->getOperand(1), KnownZero1, KnownOne1, 0);
  return (~KnownZero0 & ~KnownZero1) == 0;
}]>;

// GlobalAddresses are conceptually unsigned values, so we can also fold them
// into immediate values as long as the add is 'nuw'.
// TODO: We'd like to also match GA offsets but there are cases where the
// register can have a negative value. Find out what more we can do.
def regPlusGA : PatFrag<(ops node:$addr, node:$off),
                        (add node:$addr, node:$off),
                        [{
  return N->getFlags()->hasNoUnsignedWrap();
}]>;

// We don't need a regPlusES because external symbols never have constant
// offsets folded into them, so we can just use add.

let Defs = [ARGUMENTS] in {

// Basic load.
// FIXME: When we can break syntax compatibility, reorder the fields in the
// asmstrings to match the binary encoding.
def LOAD_I32 : I<(outs I32:$dst),
                 (ins P2Align:$p2align, offset32_op:$off, I32:$addr),
                 [], "i32.load\t$dst, ${off}(${addr})${p2align}", 0x28>;
def LOAD_I64 : I<(outs I64:$dst),
                 (ins P2Align:$p2align, offset32_op:$off, I32:$addr),
                 [], "i64.load\t$dst, ${off}(${addr})${p2align}", 0x29>;
def LOAD_F32 : I<(outs F32:$dst),
                 (ins P2Align:$p2align, offset32_op:$off, I32:$addr),
                 [], "f32.load\t$dst, ${off}(${addr})${p2align}", 0x2a>;
def LOAD_F64 : I<(outs F64:$dst),
                 (ins P2Align:$p2align, offset32_op:$off, I32:$addr),
                 [], "f64.load\t$dst, ${off}(${addr})${p2align}", 0x2b>;

} // Defs = [ARGUMENTS]

// Select loads with no constant offset.
def : Pat<(i32 (load I32:$addr)), (LOAD_I32 0, 0, $addr)>;
def : Pat<(i64 (load I32:$addr)), (LOAD_I64 0, 0, $addr)>;
def : Pat<(f32 (load I32:$addr)), (LOAD_F32 0, 0, $addr)>;
def : Pat<(f64 (load I32:$addr)), (LOAD_F64 0, 0, $addr)>;

// Select loads with a constant offset.
def : Pat<(i32 (load (regPlusImm I32:$addr, imm:$off))),
          (LOAD_I32 0, imm:$off, $addr)>;
def : Pat<(i64 (load (regPlusImm I32:$addr, imm:$off))),
          (LOAD_I64 0, imm:$off, $addr)>;
def : Pat<(f32 (load (regPlusImm I32:$addr, imm:$off))),
          (LOAD_F32 0, imm:$off, $addr)>;
def : Pat<(f64 (load (regPlusImm I32:$addr, imm:$off))),
          (LOAD_F64 0, imm:$off, $addr)>;
def : Pat<(i32 (load (or_is_add I32:$addr, imm:$off))),
          (LOAD_I32 0, imm:$off, $addr)>;
def : Pat<(i64 (load (or_is_add I32:$addr, imm:$off))),
          (LOAD_I64 0, imm:$off, $addr)>;
def : Pat<(f32 (load (or_is_add I32:$addr, imm:$off))),
          (LOAD_F32 0, imm:$off, $addr)>;
def : Pat<(f64 (load (or_is_add I32:$addr, imm:$off))),
          (LOAD_F64 0, imm:$off, $addr)>;
def : Pat<(i32 (load (regPlusGA I32:$addr,
                                (WebAssemblywrapper tglobaladdr:$off)))),
          (LOAD_I32 0, tglobaladdr:$off, $addr)>;
def : Pat<(i64 (load (regPlusGA I32:$addr,
                                (WebAssemblywrapper tglobaladdr:$off)))),
          (LOAD_I64 0, tglobaladdr:$off, $addr)>;
def : Pat<(f32 (load (regPlusGA I32:$addr,
                                (WebAssemblywrapper tglobaladdr:$off)))),
          (LOAD_F32 0, tglobaladdr:$off, $addr)>;
def : Pat<(f64 (load (regPlusGA I32:$addr,
                                (WebAssemblywrapper tglobaladdr:$off)))),
          (LOAD_F64 0, tglobaladdr:$off, $addr)>;
def : Pat<(i32 (load (add I32:$addr, (WebAssemblywrapper texternalsym:$off)))),
          (LOAD_I32 0, texternalsym:$off, $addr)>;
def : Pat<(i64 (load (add I32:$addr, (WebAssemblywrapper texternalsym:$off)))),
          (LOAD_I64 0, texternalsym:$off, $addr)>;
def : Pat<(f32 (load (add I32:$addr, (WebAssemblywrapper texternalsym:$off)))),
          (LOAD_F32 0, texternalsym:$off, $addr)>;
def : Pat<(f64 (load (add I32:$addr, (WebAssemblywrapper texternalsym:$off)))),
          (LOAD_F64 0, texternalsym:$off, $addr)>;

// Select loads with just a constant offset.
def : Pat<(i32 (load imm:$off)), (LOAD_I32 0, imm:$off, (CONST_I32 0))>;
def : Pat<(i64 (load imm:$off)), (LOAD_I64 0, imm:$off, (CONST_I32 0))>;
def : Pat<(f32 (load imm:$off)), (LOAD_F32 0, imm:$off, (CONST_I32 0))>;
def : Pat<(f64 (load imm:$off)), (LOAD_F64 0, imm:$off, (CONST_I32 0))>;
def : Pat<(i32 (load (WebAssemblywrapper tglobaladdr:$off))),
          (LOAD_I32 0, tglobaladdr:$off, (CONST_I32 0))>;
def : Pat<(i64 (load (WebAssemblywrapper tglobaladdr:$off))),
          (LOAD_I64 0, tglobaladdr:$off, (CONST_I32 0))>;
def : Pat<(f32 (load (WebAssemblywrapper tglobaladdr:$off))),
          (LOAD_F32 0, tglobaladdr:$off, (CONST_I32 0))>;
def : Pat<(f64 (load (WebAssemblywrapper tglobaladdr:$off))),
          (LOAD_F64 0, tglobaladdr:$off, (CONST_I32 0))>;
def : Pat<(i32 (load (WebAssemblywrapper texternalsym:$off))),
          (LOAD_I32 0, texternalsym:$off, (CONST_I32 0))>;
def : Pat<(i64 (load (WebAssemblywrapper texternalsym:$off))),
          (LOAD_I64 0, texternalsym:$off, (CONST_I32 0))>;
def : Pat<(f32 (load (WebAssemblywrapper texternalsym:$off))),
          (LOAD_F32 0, texternalsym:$off, (CONST_I32 0))>;
def : Pat<(f64 (load (WebAssemblywrapper texternalsym:$off))),
          (LOAD_F64 0, texternalsym:$off, (CONST_I32 0))>;

let Defs = [ARGUMENTS] in {

// Extending load.
def LOAD8_S_I32  : I<(outs I32:$dst),
                     (ins P2Align:$p2align, offset32_op:$off, I32:$addr),
                     [], "i32.load8_s\t$dst, ${off}(${addr})${p2align}", 0x2c>;
def LOAD8_U_I32  : I<(outs I32:$dst),
                     (ins P2Align:$p2align, offset32_op:$off, I32:$addr),
                     [], "i32.load8_u\t$dst, ${off}(${addr})${p2align}", 0x2d>;
def LOAD16_S_I32 : I<(outs I32:$dst),
                     (ins P2Align:$p2align, offset32_op:$off, I32:$addr),
                     [], "i32.load16_s\t$dst, ${off}(${addr})${p2align}", 0x2e>;
def LOAD16_U_I32 : I<(outs I32:$dst),
                     (ins P2Align:$p2align, offset32_op:$off, I32:$addr),
                     [], "i32.load16_u\t$dst, ${off}(${addr})${p2align}", 0x2f>;
def LOAD8_S_I64  : I<(outs I64:$dst),
                     (ins P2Align:$p2align, offset32_op:$off, I32:$addr),
                     [], "i64.load8_s\t$dst, ${off}(${addr})${p2align}", 0x30>;
def LOAD8_U_I64  : I<(outs I64:$dst),
                     (ins P2Align:$p2align, offset32_op:$off, I32:$addr),
                     [], "i64.load8_u\t$dst, ${off}(${addr})${p2align}", 0x31>;
def LOAD16_S_I64 : I<(outs I64:$dst),
                     (ins P2Align:$p2align, offset32_op:$off, I32:$addr),
                     [], "i64.load16_s\t$dst, ${off}(${addr})${p2align}", 0x32>;
def LOAD16_U_I64 : I<(outs I64:$dst),
                     (ins P2Align:$p2align, offset32_op:$off, I32:$addr),
                     [], "i64.load16_u\t$dst, ${off}(${addr})${p2align}", 0x33>;
def LOAD32_S_I64 : I<(outs I64:$dst),
                     (ins P2Align:$p2align, offset32_op:$off, I32:$addr),
                     [], "i64.load32_s\t$dst, ${off}(${addr})${p2align}", 0x34>;
def LOAD32_U_I64 : I<(outs I64:$dst),
                     (ins P2Align:$p2align, offset32_op:$off, I32:$addr),
                     [], "i64.load32_u\t$dst, ${off}(${addr})${p2align}", 0x35>;

} // Defs = [ARGUMENTS]

// Select extending loads with no constant offset.
def : Pat<(i32 (sextloadi8 I32:$addr)), (LOAD8_S_I32 0, 0, $addr)>;
def : Pat<(i32 (zextloadi8 I32:$addr)), (LOAD8_U_I32 0, 0, $addr)>;
def : Pat<(i32 (sextloadi16 I32:$addr)), (LOAD16_S_I32 0, 0, $addr)>;
def : Pat<(i32 (zextloadi16 I32:$addr)), (LOAD16_U_I32 0, 0, $addr)>;
def : Pat<(i64 (sextloadi8 I32:$addr)), (LOAD8_S_I64 0, 0, $addr)>;
def : Pat<(i64 (zextloadi8 I32:$addr)), (LOAD8_U_I64 0, 0, $addr)>;
def : Pat<(i64 (sextloadi16 I32:$addr)), (LOAD16_S_I64 0, 0, $addr)>;
def : Pat<(i64 (zextloadi16 I32:$addr)), (LOAD16_U_I64 0, 0, $addr)>;
def : Pat<(i64 (sextloadi32 I32:$addr)), (LOAD32_S_I64 0, 0, $addr)>;
def : Pat<(i64 (zextloadi32 I32:$addr)), (LOAD32_U_I64 0, 0, $addr)>;

// Select extending loads with a constant offset.
def : Pat<(i32 (sextloadi8 (regPlusImm I32:$addr, imm:$off))),
          (LOAD8_S_I32 0, imm:$off, $addr)>;
def : Pat<(i32 (zextloadi8 (regPlusImm I32:$addr, imm:$off))),
          (LOAD8_U_I32 0, imm:$off, $addr)>;
def : Pat<(i32 (sextloadi16 (regPlusImm I32:$addr, imm:$off))),
          (LOAD16_S_I32 0, imm:$off, $addr)>;
def : Pat<(i32 (zextloadi16 (regPlusImm I32:$addr, imm:$off))),
          (LOAD16_U_I32 0, imm:$off, $addr)>;
def : Pat<(i64 (sextloadi8 (regPlusImm I32:$addr, imm:$off))),
          (LOAD8_S_I64 0, imm:$off, $addr)>;
def : Pat<(i64 (zextloadi8 (regPlusImm I32:$addr, imm:$off))),
          (LOAD8_U_I64 0, imm:$off, $addr)>;
def : Pat<(i64 (sextloadi16 (regPlusImm I32:$addr, imm:$off))),
          (LOAD16_S_I64 0, imm:$off, $addr)>;
def : Pat<(i64 (zextloadi16 (regPlusImm I32:$addr, imm:$off))),
          (LOAD16_U_I64 0, imm:$off, $addr)>;
def : Pat<(i64 (sextloadi32 (regPlusImm I32:$addr, imm:$off))),
          (LOAD32_S_I64 0, imm:$off, $addr)>;
def : Pat<(i64 (zextloadi32 (regPlusImm I32:$addr, imm:$off))),
          (LOAD32_U_I64 0, imm:$off, $addr)>;
def : Pat<(i32 (sextloadi8 (or_is_add I32:$addr, imm:$off))),
          (LOAD8_S_I32 0, imm:$off, $addr)>;
def : Pat<(i32 (zextloadi8 (or_is_add I32:$addr, imm:$off))),
          (LOAD8_U_I32 0, imm:$off, $addr)>;
def : Pat<(i32 (sextloadi16 (or_is_add I32:$addr, imm:$off))),
          (LOAD16_S_I32 0, imm:$off, $addr)>;
def : Pat<(i32 (zextloadi16 (or_is_add I32:$addr, imm:$off))),
          (LOAD16_U_I32 0, imm:$off, $addr)>;
def : Pat<(i64 (sextloadi8 (or_is_add I32:$addr, imm:$off))),
          (LOAD8_S_I64 0, imm:$off, $addr)>;
def : Pat<(i64 (zextloadi8 (or_is_add I32:$addr, imm:$off))),
          (LOAD8_U_I64 0, imm:$off, $addr)>;
def : Pat<(i64 (sextloadi16 (or_is_add I32:$addr, imm:$off))),
          (LOAD16_S_I64 0, imm:$off, $addr)>;
def : Pat<(i64 (zextloadi16 (or_is_add I32:$addr, imm:$off))),
          (LOAD16_U_I64 0, imm:$off, $addr)>;
def : Pat<(i64 (sextloadi32 (or_is_add I32:$addr, imm:$off))),
          (LOAD32_S_I64 0, imm:$off, $addr)>;
def : Pat<(i64 (zextloadi32 (or_is_add I32:$addr, imm:$off))),
          (LOAD32_U_I64 0, imm:$off, $addr)>;
def : Pat<(i32 (sextloadi8 (regPlusGA I32:$addr,
                                      (WebAssemblywrapper tglobaladdr:$off)))),
          (LOAD8_S_I32 0, tglobaladdr:$off, $addr)>;
def : Pat<(i32 (zextloadi8 (regPlusGA I32:$addr,
                                      (WebAssemblywrapper tglobaladdr:$off)))),
          (LOAD8_U_I32 0, tglobaladdr:$off, $addr)>;
def : Pat<(i32 (sextloadi16 (regPlusGA I32:$addr,
                                       (WebAssemblywrapper tglobaladdr:$off)))),
          (LOAD16_S_I32 0, tglobaladdr:$off, $addr)>;
def : Pat<(i32 (zextloadi16 (regPlusGA I32:$addr,
                                       (WebAssemblywrapper tglobaladdr:$off)))),
          (LOAD16_U_I32 0, tglobaladdr:$off, $addr)>;
def : Pat<(i64 (sextloadi8 (regPlusGA I32:$addr,
                                      (WebAssemblywrapper tglobaladdr:$off)))),
          (LOAD8_S_I64 0, tglobaladdr:$off, $addr)>;
def : Pat<(i64 (zextloadi8 (regPlusGA I32:$addr,
                                      (WebAssemblywrapper tglobaladdr:$off)))),
          (LOAD8_U_I64 0, tglobaladdr:$off, $addr)>;
def : Pat<(i64 (sextloadi16 (regPlusGA I32:$addr,
                                       (WebAssemblywrapper tglobaladdr:$off)))),
          (LOAD16_S_I64 0, tglobaladdr:$off, $addr)>;
def : Pat<(i64 (zextloadi16 (regPlusGA I32:$addr,
                                       (WebAssemblywrapper tglobaladdr:$off)))),
          (LOAD16_U_I64 0, tglobaladdr:$off, $addr)>;
def : Pat<(i64 (sextloadi32 (regPlusGA I32:$addr,
                                       (WebAssemblywrapper tglobaladdr:$off)))),
          (LOAD32_S_I64 0, tglobaladdr:$off, $addr)>;
def : Pat<(i64 (zextloadi32 (regPlusGA I32:$addr,
                                       (WebAssemblywrapper tglobaladdr:$off)))),
          (LOAD32_U_I64 0, tglobaladdr:$off, $addr)>;
def : Pat<(i32 (sextloadi8 (add I32:$addr,
                                (WebAssemblywrapper texternalsym:$off)))),
          (LOAD8_S_I32 0, texternalsym:$off, $addr)>;
def : Pat<(i32 (zextloadi8 (add I32:$addr,
                                (WebAssemblywrapper texternalsym:$off)))),
          (LOAD8_U_I32 0, texternalsym:$off, $addr)>;
def : Pat<(i32 (sextloadi16 (add I32:$addr,
                                 (WebAssemblywrapper texternalsym:$off)))),
          (LOAD16_S_I32 0, texternalsym:$off, $addr)>;
def : Pat<(i32 (zextloadi16 (add I32:$addr,
                                 (WebAssemblywrapper texternalsym:$off)))),
          (LOAD16_U_I32 0, texternalsym:$off, $addr)>;
def : Pat<(i64 (sextloadi8 (add I32:$addr,
                                (WebAssemblywrapper texternalsym:$off)))),
          (LOAD8_S_I64 0, texternalsym:$off, $addr)>;
def : Pat<(i64 (zextloadi8 (add I32:$addr,
                                (WebAssemblywrapper texternalsym:$off)))),
          (LOAD8_U_I64 0, texternalsym:$off, $addr)>;
def : Pat<(i64 (sextloadi16 (add I32:$addr,
                                 (WebAssemblywrapper texternalsym:$off)))),
          (LOAD16_S_I64 0, texternalsym:$off, $addr)>;
def : Pat<(i64 (zextloadi16 (add I32:$addr,
                                 (WebAssemblywrapper texternalsym:$off)))),
          (LOAD16_U_I64 0, texternalsym:$off, $addr)>;
def : Pat<(i64 (sextloadi32 (add I32:$addr,
                                 (WebAssemblywrapper texternalsym:$off)))),
          (LOAD32_S_I64 0, texternalsym:$off, $addr)>;
def : Pat<(i64 (zextloadi32 (add I32:$addr,
                                 (WebAssemblywrapper texternalsym:$off)))),
          (LOAD32_U_I64 0, texternalsym:$off, $addr)>;

// Select extending loads with just a constant offset.
def : Pat<(i32 (sextloadi8 imm:$off)),
          (LOAD8_S_I32 0, imm:$off, (CONST_I32 0))>;
def : Pat<(i32 (zextloadi8 imm:$off)),
          (LOAD8_U_I32 0, imm:$off, (CONST_I32 0))>;
def : Pat<(i32 (sextloadi16 imm:$off)),
          (LOAD16_S_I32 0, imm:$off, (CONST_I32 0))>;
def : Pat<(i32 (zextloadi16 imm:$off)),
          (LOAD16_U_I32 0, imm:$off, (CONST_I32 0))>;
def : Pat<(i64 (sextloadi8 imm:$off)),
          (LOAD8_S_I64 0, imm:$off, (CONST_I32 0))>;
def : Pat<(i64 (zextloadi8 imm:$off)),
          (LOAD8_U_I64 0, imm:$off, (CONST_I32 0))>;
def : Pat<(i64 (sextloadi16 imm:$off)),
          (LOAD16_S_I64 0, imm:$off, (CONST_I32 0))>;
def : Pat<(i64 (zextloadi16 imm:$off)),
          (LOAD16_U_I64 0, imm:$off, (CONST_I32 0))>;
def : Pat<(i64 (sextloadi32 imm:$off)),
          (LOAD32_S_I64 0, imm:$off, (CONST_I32 0))>;
def : Pat<(i64 (zextloadi32 imm:$off)),
          (LOAD32_U_I64 0, imm:$off, (CONST_I32 0))>;
def : Pat<(i32 (sextloadi8 (WebAssemblywrapper tglobaladdr:$off))),
          (LOAD8_S_I32 0, tglobaladdr:$off, (CONST_I32 0))>;
def : Pat<(i32 (zextloadi8 (WebAssemblywrapper tglobaladdr:$off))),
          (LOAD8_U_I32 0, tglobaladdr:$off, (CONST_I32 0))>;
def : Pat<(i32 (sextloadi16 (WebAssemblywrapper tglobaladdr:$off))),
          (LOAD16_S_I32 0, tglobaladdr:$off, (CONST_I32 0))>;
def : Pat<(i32 (zextloadi16 (WebAssemblywrapper tglobaladdr:$off))),
          (LOAD16_U_I32 0, tglobaladdr:$off, (CONST_I32 0))>;
def : Pat<(i64 (sextloadi8 (WebAssemblywrapper tglobaladdr:$off))),
          (LOAD8_S_I64 0, tglobaladdr:$off, (CONST_I32 0))>;
def : Pat<(i64 (zextloadi8 (WebAssemblywrapper tglobaladdr:$off))),
          (LOAD8_U_I64 0, tglobaladdr:$off, (CONST_I32 0))>;
def : Pat<(i64 (sextloadi16 (WebAssemblywrapper tglobaladdr:$off))),
          (LOAD16_S_I64 0, tglobaladdr:$off, (CONST_I32 0))>;
def : Pat<(i64 (zextloadi16 (WebAssemblywrapper tglobaladdr:$off))),
          (LOAD16_U_I64 0, tglobaladdr:$off, (CONST_I32 0))>;
def : Pat<(i64 (sextloadi32 (WebAssemblywrapper tglobaladdr:$off))),
          (LOAD32_S_I64 0, tglobaladdr:$off, (CONST_I32 0))>;
def : Pat<(i64 (zextloadi32 (WebAssemblywrapper tglobaladdr:$off))),
          (LOAD32_U_I64 0, tglobaladdr:$off, (CONST_I32 0))>;
def : Pat<(i32 (sextloadi8 (WebAssemblywrapper texternalsym:$off))),
          (LOAD8_S_I32 0, texternalsym:$off, (CONST_I32 0))>;
def : Pat<(i32 (zextloadi8 (WebAssemblywrapper texternalsym:$off))),
          (LOAD8_U_I32 0, texternalsym:$off, (CONST_I32 0))>;
def : Pat<(i32 (sextloadi16 (WebAssemblywrapper texternalsym:$off))),
          (LOAD16_S_I32 0, texternalsym:$off, (CONST_I32 0))>;
def : Pat<(i32 (zextloadi16 (WebAssemblywrapper texternalsym:$off))),
          (LOAD16_U_I32 0, texternalsym:$off, (CONST_I32 0))>;
def : Pat<(i64 (sextloadi8 (WebAssemblywrapper texternalsym:$off))),
          (LOAD8_S_I64 0, texternalsym:$off, (CONST_I32 0))>;
def : Pat<(i64 (zextloadi8 (WebAssemblywrapper texternalsym:$off))),
          (LOAD8_U_I64 0, texternalsym:$off, (CONST_I32 0))>;
def : Pat<(i64 (sextloadi16 (WebAssemblywrapper texternalsym:$off))),
          (LOAD16_S_I64 0, texternalsym:$off, (CONST_I32 0))>;
def : Pat<(i64 (zextloadi16 (WebAssemblywrapper texternalsym:$off))),
          (LOAD16_U_I64 0, texternalsym:$off, (CONST_I32 0))>;
def : Pat<(i64 (sextloadi32 (WebAssemblywrapper texternalsym:$off))),
          (LOAD32_S_I64 0, texternalsym:$off, (CONST_I32 0))>;
def : Pat<(i64 (zextloadi32 (WebAssemblywrapper texternalsym:$off))),
          (LOAD32_U_I64 0, texternalsym:$off, (CONST_I32 0))>;

// Resolve "don't care" extending loads to zero-extending loads. This is
// somewhat arbitrary, but zero-extending is conceptually simpler.

// Select "don't care" extending loads with no constant offset.
def : Pat<(i32 (extloadi8 I32:$addr)),  (LOAD8_U_I32 0, 0, $addr)>;
def : Pat<(i32 (extloadi16 I32:$addr)), (LOAD16_U_I32 0, 0, $addr)>;
def : Pat<(i64 (extloadi8 I32:$addr)),  (LOAD8_U_I64 0, 0, $addr)>;
def : Pat<(i64 (extloadi16 I32:$addr)), (LOAD16_U_I64 0, 0, $addr)>;
def : Pat<(i64 (extloadi32 I32:$addr)), (LOAD32_U_I64 0, 0, $addr)>;

// Select "don't care" extending loads with a constant offset.
def : Pat<(i32 (extloadi8 (regPlusImm I32:$addr, imm:$off))),
          (LOAD8_U_I32 0, imm:$off, $addr)>;
def : Pat<(i32 (extloadi16 (regPlusImm I32:$addr, imm:$off))),
          (LOAD16_U_I32 0, imm:$off, $addr)>;
def : Pat<(i64 (extloadi8 (regPlusImm I32:$addr, imm:$off))),
          (LOAD8_U_I64 0, imm:$off, $addr)>;
def : Pat<(i64 (extloadi16 (regPlusImm I32:$addr, imm:$off))),
          (LOAD16_U_I64 0, imm:$off, $addr)>;
def : Pat<(i64 (extloadi32 (regPlusImm I32:$addr, imm:$off))),
          (LOAD32_U_I64 0, imm:$off, $addr)>;
def : Pat<(i32 (extloadi8 (or_is_add I32:$addr, imm:$off))),
          (LOAD8_U_I32 0, imm:$off, $addr)>;
def : Pat<(i32 (extloadi16 (or_is_add I32:$addr, imm:$off))),
          (LOAD16_U_I32 0, imm:$off, $addr)>;
def : Pat<(i64 (extloadi8 (or_is_add I32:$addr, imm:$off))),
          (LOAD8_U_I64 0, imm:$off, $addr)>;
def : Pat<(i64 (extloadi16 (or_is_add I32:$addr, imm:$off))),
          (LOAD16_U_I64 0, imm:$off, $addr)>;
def : Pat<(i64 (extloadi32 (or_is_add I32:$addr, imm:$off))),
          (LOAD32_U_I64 0, imm:$off, $addr)>;
def : Pat<(i32 (extloadi8 (regPlusGA I32:$addr,
                                     (WebAssemblywrapper tglobaladdr:$off)))),
          (LOAD8_U_I32 0, tglobaladdr:$off, $addr)>;
def : Pat<(i32 (extloadi16 (regPlusGA I32:$addr,
                                      (WebAssemblywrapper tglobaladdr:$off)))),
          (LOAD16_U_I32 0, tglobaladdr:$off, $addr)>;
def : Pat<(i64 (extloadi8 (regPlusGA I32:$addr,
                                     (WebAssemblywrapper tglobaladdr:$off)))),
          (LOAD8_U_I64 0, tglobaladdr:$off, $addr)>;
def : Pat<(i64 (extloadi16 (regPlusGA I32:$addr,
                                      (WebAssemblywrapper tglobaladdr:$off)))),
          (LOAD16_U_I64 0, tglobaladdr:$off, $addr)>;
def : Pat<(i64 (extloadi32 (regPlusGA I32:$addr,
                                      (WebAssemblywrapper tglobaladdr:$off)))),
          (LOAD32_U_I64 0, tglobaladdr:$off, $addr)>;
def : Pat<(i32 (extloadi8 (add I32:$addr,
                               (WebAssemblywrapper texternalsym:$off)))),
          (LOAD8_U_I32 0, texternalsym:$off, $addr)>;
def : Pat<(i32 (extloadi16 (add I32:$addr,
                                (WebAssemblywrapper texternalsym:$off)))),
          (LOAD16_U_I32 0, texternalsym:$off, $addr)>;
def : Pat<(i64 (extloadi8 (add I32:$addr,
                               (WebAssemblywrapper texternalsym:$off)))),
          (LOAD8_U_I64 0, texternalsym:$off, $addr)>;
def : Pat<(i64 (extloadi16 (add I32:$addr,
                                (WebAssemblywrapper texternalsym:$off)))),
          (LOAD16_U_I64 0, texternalsym:$off, $addr)>;
def : Pat<(i64 (extloadi32 (add I32:$addr,
                                (WebAssemblywrapper texternalsym:$off)))),
          (LOAD32_U_I64 0, texternalsym:$off, $addr)>;

// Select "don't care" extending loads with just a constant offset.
def : Pat<(i32 (extloadi8 imm:$off)),
          (LOAD8_U_I32 0, imm:$off, (CONST_I32 0))>;
def : Pat<(i32 (extloadi16 imm:$off)),
          (LOAD16_U_I32 0, imm:$off, (CONST_I32 0))>;
def : Pat<(i64 (extloadi8 imm:$off)),
          (LOAD8_U_I64 0, imm:$off, (CONST_I32 0))>;
def : Pat<(i64 (extloadi16 imm:$off)),
          (LOAD16_U_I64 0, imm:$off, (CONST_I32 0))>;
def : Pat<(i64 (extloadi32 imm:$off)),
          (LOAD32_U_I64 0, imm:$off, (CONST_I32 0))>;
def : Pat<(i32 (extloadi8 (WebAssemblywrapper tglobaladdr:$off))),
          (LOAD8_U_I32 0, tglobaladdr:$off, (CONST_I32 0))>;
def : Pat<(i32 (extloadi16 (WebAssemblywrapper tglobaladdr:$off))),
          (LOAD16_U_I32 0, tglobaladdr:$off, (CONST_I32 0))>;
def : Pat<(i64 (extloadi8 (WebAssemblywrapper tglobaladdr:$off))),
          (LOAD8_U_I64 0, tglobaladdr:$off, (CONST_I32 0))>;
def : Pat<(i64 (extloadi16 (WebAssemblywrapper tglobaladdr:$off))),
          (LOAD16_U_I64 0, tglobaladdr:$off, (CONST_I32 0))>;
def : Pat<(i64 (extloadi32 (WebAssemblywrapper tglobaladdr:$off))),
          (LOAD32_U_I64 0, tglobaladdr:$off, (CONST_I32 0))>;
def : Pat<(i32 (extloadi8 (WebAssemblywrapper texternalsym:$off))),
          (LOAD8_U_I32 0, texternalsym:$off, (CONST_I32 0))>;
def : Pat<(i32 (extloadi16 (WebAssemblywrapper texternalsym:$off))),
          (LOAD16_U_I32 0, texternalsym:$off, (CONST_I32 0))>;
def : Pat<(i64 (extloadi8 (WebAssemblywrapper texternalsym:$off))),
          (LOAD8_U_I64 0, texternalsym:$off, (CONST_I32 0))>;
def : Pat<(i64 (extloadi16 (WebAssemblywrapper texternalsym:$off))),
          (LOAD16_U_I64 0, texternalsym:$off, (CONST_I32 0))>;
def : Pat<(i64 (extloadi32 (WebAssemblywrapper texternalsym:$off))),
          (LOAD32_U_I64 0, tglobaladdr:$off, (CONST_I32 0))>;

let Defs = [ARGUMENTS] in {

// Basic store.
// Note: WebAssembly inverts SelectionDAG's usual operand order.
def STORE_I32  : I<(outs), (ins P2Align:$p2align, offset32_op:$off, I32:$addr,
                            I32:$val), [],
                   "i32.store\t${off}(${addr})${p2align}, $val", 0x36>;
def STORE_I64  : I<(outs), (ins P2Align:$p2align, offset32_op:$off, I32:$addr,
                            I64:$val), [],
                   "i64.store\t${off}(${addr})${p2align}, $val", 0x37>;
def STORE_F32  : I<(outs), (ins P2Align:$p2align, offset32_op:$off, I32:$addr,
                            F32:$val), [],
                   "f32.store\t${off}(${addr})${p2align}, $val", 0x38>;
def STORE_F64  : I<(outs), (ins P2Align:$p2align, offset32_op:$off, I32:$addr,
                            F64:$val), [],
                   "f64.store\t${off}(${addr})${p2align}, $val", 0x39>;

} // Defs = [ARGUMENTS]

// Select stores with no constant offset.
def : Pat<(store I32:$val, I32:$addr), (STORE_I32 0, 0, I32:$addr, I32:$val)>;
def : Pat<(store I64:$val, I32:$addr), (STORE_I64 0, 0, I32:$addr, I64:$val)>;
def : Pat<(store F32:$val, I32:$addr), (STORE_F32 0, 0, I32:$addr, F32:$val)>;
def : Pat<(store F64:$val, I32:$addr), (STORE_F64 0, 0, I32:$addr, F64:$val)>;

// Select stores with a constant offset.
def : Pat<(store I32:$val, (regPlusImm I32:$addr, imm:$off)),
          (STORE_I32 0, imm:$off, I32:$addr, I32:$val)>;
def : Pat<(store I64:$val, (regPlusImm I32:$addr, imm:$off)),
          (STORE_I64 0, imm:$off, I32:$addr, I64:$val)>;
def : Pat<(store F32:$val, (regPlusImm I32:$addr, imm:$off)),
          (STORE_F32 0, imm:$off, I32:$addr, F32:$val)>;
def : Pat<(store F64:$val, (regPlusImm I32:$addr, imm:$off)),
          (STORE_F64 0, imm:$off, I32:$addr, F64:$val)>;
def : Pat<(store I32:$val, (or_is_add I32:$addr, imm:$off)),
          (STORE_I32 0, imm:$off, I32:$addr, I32:$val)>;
def : Pat<(store I64:$val, (or_is_add I32:$addr, imm:$off)),
          (STORE_I64 0, imm:$off, I32:$addr, I64:$val)>;
def : Pat<(store F32:$val, (or_is_add I32:$addr, imm:$off)),
          (STORE_F32 0, imm:$off, I32:$addr, F32:$val)>;
def : Pat<(store F64:$val, (or_is_add I32:$addr, imm:$off)),
          (STORE_F64 0, imm:$off, I32:$addr, F64:$val)>;
def : Pat<(store I32:$val, (regPlusGA I32:$addr,
                                      (WebAssemblywrapper tglobaladdr:$off))),
          (STORE_I32 0, tglobaladdr:$off, I32:$addr, I32:$val)>;
def : Pat<(store I64:$val, (regPlusGA I32:$addr,
                                      (WebAssemblywrapper tglobaladdr:$off))),
          (STORE_I64 0, tglobaladdr:$off, I32:$addr, I64:$val)>;
def : Pat<(store F32:$val, (regPlusGA I32:$addr,
                                      (WebAssemblywrapper tglobaladdr:$off))),
          (STORE_F32 0, tglobaladdr:$off, I32:$addr, F32:$val)>;
def : Pat<(store F64:$val, (regPlusGA I32:$addr,
                                      (WebAssemblywrapper tglobaladdr:$off))),
          (STORE_F64 0, tglobaladdr:$off, I32:$addr, F64:$val)>;
def : Pat<(store I32:$val, (add I32:$addr,
                                (WebAssemblywrapper texternalsym:$off))),
          (STORE_I32 0, texternalsym:$off, I32:$addr, I32:$val)>;
def : Pat<(store I64:$val, (add I32:$addr,
                                (WebAssemblywrapper texternalsym:$off))),
          (STORE_I64 0, texternalsym:$off, I32:$addr, I64:$val)>;
def : Pat<(store F32:$val, (add I32:$addr,
                                (WebAssemblywrapper texternalsym:$off))),
          (STORE_F32 0, texternalsym:$off, I32:$addr, F32:$val)>;
def : Pat<(store F64:$val, (add I32:$addr,
                                (WebAssemblywrapper texternalsym:$off))),
          (STORE_F64 0, texternalsym:$off, I32:$addr, F64:$val)>;

// Select stores with just a constant offset.
def : Pat<(store I32:$val, imm:$off),
          (STORE_I32 0, imm:$off, (CONST_I32 0), I32:$val)>;
def : Pat<(store I64:$val, imm:$off),
          (STORE_I64 0, imm:$off, (CONST_I32 0), I64:$val)>;
def : Pat<(store F32:$val, imm:$off),
          (STORE_F32 0, imm:$off, (CONST_I32 0), F32:$val)>;
def : Pat<(store F64:$val, imm:$off),
          (STORE_F64 0, imm:$off, (CONST_I32 0), F64:$val)>;
def : Pat<(store I32:$val, (WebAssemblywrapper tglobaladdr:$off)),
          (STORE_I32 0, tglobaladdr:$off, (CONST_I32 0), I32:$val)>;
def : Pat<(store I64:$val, (WebAssemblywrapper tglobaladdr:$off)),
          (STORE_I64 0, tglobaladdr:$off, (CONST_I32 0), I64:$val)>;
def : Pat<(store F32:$val, (WebAssemblywrapper tglobaladdr:$off)),
          (STORE_F32 0, tglobaladdr:$off, (CONST_I32 0), F32:$val)>;
def : Pat<(store F64:$val, (WebAssemblywrapper tglobaladdr:$off)),
          (STORE_F64 0, tglobaladdr:$off, (CONST_I32 0), F64:$val)>;
def : Pat<(store I32:$val, (WebAssemblywrapper texternalsym:$off)),
          (STORE_I32 0, texternalsym:$off, (CONST_I32 0), I32:$val)>;
def : Pat<(store I64:$val, (WebAssemblywrapper texternalsym:$off)),
          (STORE_I64 0, texternalsym:$off, (CONST_I32 0), I64:$val)>;
def : Pat<(store F32:$val, (WebAssemblywrapper texternalsym:$off)),
          (STORE_F32 0, texternalsym:$off, (CONST_I32 0), F32:$val)>;
def : Pat<(store F64:$val, (WebAssemblywrapper texternalsym:$off)),
          (STORE_F64 0, texternalsym:$off, (CONST_I32 0), F64:$val)>;

let Defs = [ARGUMENTS] in {

// Truncating store.
def STORE8_I32  : I<(outs), (ins P2Align:$p2align, offset32_op:$off, I32:$addr,
                             I32:$val), [],
                    "i32.store8\t${off}(${addr})${p2align}, $val", 0x3a>;
def STORE16_I32 : I<(outs), (ins P2Align:$p2align, offset32_op:$off, I32:$addr,
                             I32:$val), [],
                    "i32.store16\t${off}(${addr})${p2align}, $val", 0x3b>;
def STORE8_I64  : I<(outs), (ins P2Align:$p2align, offset32_op:$off, I32:$addr,
                             I64:$val), [],
                    "i64.store8\t${off}(${addr})${p2align}, $val", 0x3c>;
def STORE16_I64 : I<(outs), (ins P2Align:$p2align, offset32_op:$off, I32:$addr,
                             I64:$val), [],
                    "i64.store16\t${off}(${addr})${p2align}, $val", 0x3d>;
def STORE32_I64 : I<(outs), (ins P2Align:$p2align, offset32_op:$off, I32:$addr,
                             I64:$val), [],
                    "i64.store32\t${off}(${addr})${p2align}, $val", 0x3e>;

} // Defs = [ARGUMENTS]

// Select truncating stores with no constant offset.
def : Pat<(truncstorei8 I32:$val, I32:$addr),
          (STORE8_I32 0, 0, I32:$addr, I32:$val)>;
def : Pat<(truncstorei16 I32:$val, I32:$addr),
          (STORE16_I32 0, 0, I32:$addr, I32:$val)>;
def : Pat<(truncstorei8 I64:$val, I32:$addr),
          (STORE8_I64 0, 0, I32:$addr, I64:$val)>;
def : Pat<(truncstorei16 I64:$val, I32:$addr),
          (STORE16_I64 0, 0, I32:$addr, I64:$val)>;
def : Pat<(truncstorei32 I64:$val, I32:$addr),
          (STORE32_I64 0, 0, I32:$addr, I64:$val)>;

// Select truncating stores with a constant offset.
def : Pat<(truncstorei8 I32:$val, (regPlusImm I32:$addr, imm:$off)),
          (STORE8_I32 0, imm:$off, I32:$addr, I32:$val)>;
def : Pat<(truncstorei16 I32:$val, (regPlusImm I32:$addr, imm:$off)),
          (STORE16_I32 0, imm:$off, I32:$addr, I32:$val)>;
def : Pat<(truncstorei8 I64:$val, (regPlusImm I32:$addr, imm:$off)),
          (STORE8_I64 0, imm:$off, I32:$addr, I64:$val)>;
def : Pat<(truncstorei16 I64:$val, (regPlusImm I32:$addr, imm:$off)),
          (STORE16_I64 0, imm:$off, I32:$addr, I64:$val)>;
def : Pat<(truncstorei32 I64:$val, (regPlusImm I32:$addr, imm:$off)),
          (STORE32_I64 0, imm:$off, I32:$addr, I64:$val)>;
def : Pat<(truncstorei8 I32:$val, (or_is_add I32:$addr, imm:$off)),
          (STORE8_I32 0, imm:$off, I32:$addr, I32:$val)>;
def : Pat<(truncstorei16 I32:$val, (or_is_add I32:$addr, imm:$off)),
          (STORE16_I32 0, imm:$off, I32:$addr, I32:$val)>;
def : Pat<(truncstorei8 I64:$val, (or_is_add I32:$addr, imm:$off)),
          (STORE8_I64 0, imm:$off, I32:$addr, I64:$val)>;
def : Pat<(truncstorei16 I64:$val, (or_is_add I32:$addr, imm:$off)),
          (STORE16_I64 0, imm:$off, I32:$addr, I64:$val)>;
def : Pat<(truncstorei32 I64:$val, (or_is_add I32:$addr, imm:$off)),
          (STORE32_I64 0, imm:$off, I32:$addr, I64:$val)>;
def : Pat<(truncstorei8 I32:$val,
                        (regPlusGA I32:$addr,
                                   (WebAssemblywrapper tglobaladdr:$off))),
          (STORE8_I32 0, tglobaladdr:$off, I32:$addr, I32:$val)>;
def : Pat<(truncstorei16 I32:$val,
                         (regPlusGA I32:$addr,
                                    (WebAssemblywrapper tglobaladdr:$off))),
          (STORE16_I32 0, tglobaladdr:$off, I32:$addr, I32:$val)>;
def : Pat<(truncstorei8 I64:$val,
                        (regPlusGA I32:$addr,
                                   (WebAssemblywrapper tglobaladdr:$off))),
          (STORE8_I64 0, tglobaladdr:$off, I32:$addr, I64:$val)>;
def : Pat<(truncstorei16 I64:$val,
                         (regPlusGA I32:$addr,
                                    (WebAssemblywrapper tglobaladdr:$off))),
          (STORE16_I64 0, tglobaladdr:$off, I32:$addr, I64:$val)>;
def : Pat<(truncstorei32 I64:$val,
                         (regPlusGA I32:$addr,
                                    (WebAssemblywrapper tglobaladdr:$off))),
          (STORE32_I64 0, tglobaladdr:$off, I32:$addr, I64:$val)>;
def : Pat<(truncstorei8 I32:$val, (add I32:$addr,
                                       (WebAssemblywrapper texternalsym:$off))),
          (STORE8_I32 0, texternalsym:$off, I32:$addr, I32:$val)>;
def : Pat<(truncstorei16 I32:$val,
                         (add I32:$addr,
                              (WebAssemblywrapper texternalsym:$off))),
          (STORE16_I32 0, texternalsym:$off, I32:$addr, I32:$val)>;
def : Pat<(truncstorei8 I64:$val,
                        (add I32:$addr,
                             (WebAssemblywrapper texternalsym:$off))),
          (STORE8_I64 0, texternalsym:$off, I32:$addr, I64:$val)>;
def : Pat<(truncstorei16 I64:$val,
                         (add I32:$addr,
                              (WebAssemblywrapper texternalsym:$off))),
          (STORE16_I64 0, texternalsym:$off, I32:$addr, I64:$val)>;
def : Pat<(truncstorei32 I64:$val,
                         (add I32:$addr,
                              (WebAssemblywrapper texternalsym:$off))),
          (STORE32_I64 0, texternalsym:$off, I32:$addr, I64:$val)>;

// Select truncating stores with just a constant offset.
def : Pat<(truncstorei8 I32:$val, imm:$off),
          (STORE8_I32 0, imm:$off, (CONST_I32 0), I32:$val)>;
def : Pat<(truncstorei16 I32:$val, imm:$off),
          (STORE16_I32 0, imm:$off, (CONST_I32 0), I32:$val)>;
def : Pat<(truncstorei8 I64:$val, imm:$off),
          (STORE8_I64 0, imm:$off, (CONST_I32 0), I64:$val)>;
def : Pat<(truncstorei16 I64:$val, imm:$off),
          (STORE16_I64 0, imm:$off, (CONST_I32 0), I64:$val)>;
def : Pat<(truncstorei32 I64:$val, imm:$off),
          (STORE32_I64 0, imm:$off, (CONST_I32 0), I64:$val)>;
def : Pat<(truncstorei8 I32:$val, (WebAssemblywrapper tglobaladdr:$off)),
          (STORE8_I32 0, tglobaladdr:$off, (CONST_I32 0), I32:$val)>;
def : Pat<(truncstorei16 I32:$val, (WebAssemblywrapper tglobaladdr:$off)),
          (STORE16_I32 0, tglobaladdr:$off, (CONST_I32 0), I32:$val)>;
def : Pat<(truncstorei8 I64:$val, (WebAssemblywrapper tglobaladdr:$off)),
          (STORE8_I64 0, tglobaladdr:$off, (CONST_I32 0), I64:$val)>;
def : Pat<(truncstorei16 I64:$val, (WebAssemblywrapper tglobaladdr:$off)),
          (STORE16_I64 0, tglobaladdr:$off, (CONST_I32 0), I64:$val)>;
def : Pat<(truncstorei32 I64:$val, (WebAssemblywrapper tglobaladdr:$off)),
          (STORE32_I64 0, tglobaladdr:$off, (CONST_I32 0), I64:$val)>;
def : Pat<(truncstorei8 I32:$val, (WebAssemblywrapper texternalsym:$off)),
          (STORE8_I32 0, texternalsym:$off, (CONST_I32 0), I32:$val)>;
def : Pat<(truncstorei16 I32:$val, (WebAssemblywrapper texternalsym:$off)),
          (STORE16_I32 0, texternalsym:$off, (CONST_I32 0), I32:$val)>;
def : Pat<(truncstorei8 I64:$val, (WebAssemblywrapper texternalsym:$off)),
          (STORE8_I64 0, texternalsym:$off, (CONST_I32 0), I64:$val)>;
def : Pat<(truncstorei16 I64:$val, (WebAssemblywrapper texternalsym:$off)),
          (STORE16_I64 0, texternalsym:$off, (CONST_I32 0), I64:$val)>;
def : Pat<(truncstorei32 I64:$val, (WebAssemblywrapper texternalsym:$off)),
          (STORE32_I64 0, texternalsym:$off, (CONST_I32 0), I64:$val)>;

let Defs = [ARGUMENTS] in {

// Current memory size.
def CURRENT_MEMORY_I32 : I<(outs I32:$dst), (ins i32imm:$flags),
                           [],
                           "current_memory\t$dst", 0x3f>,
                         Requires<[HasAddr32]>;

// Grow memory.
def GROW_MEMORY_I32 : I<(outs I32:$dst), (ins i32imm:$flags, I32:$delta),
                        [],
                        "grow_memory\t$dst, $delta", 0x40>,
                      Requires<[HasAddr32]>;

} // Defs = [ARGUMENTS]

def : Pat<(int_wasm_current_memory),
          (CURRENT_MEMORY_I32 0)>;
def : Pat<(int_wasm_grow_memory I32:$delta),
          (GROW_MEMORY_I32 0, $delta)>;
