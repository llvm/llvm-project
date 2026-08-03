//===-- PISAInstrFormats.td - PISA Instruction Formats -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "PISAGenericOpcodes.td"
include "llvm/Target/PISA/MnemonicOperand.td"

class PISAAsmOperand<string am, string sig=am> : AsmOperandClass {
  let Name = am;
  string Signature = sig;
  let ParserMethod = "parse"#Signature;
  let PredicateMethod = "is"#Signature;
}

// This operand class provides a simple interface for quickly describing
// the syntax of memory operands that are formed by a sequence
// For example, specifying "R,i" for the args means that
// the operand would look something like: [%r0, 5].
class PISAMemSeqAsmOperand<string am, string args, string uniq> : AsmOperandClass {
  let Name = am#uniq;
  let ParserMethod = "parse"#am#"(\""#args#"\")";
  let PredicateMethod = "is"#am;
  let RenderMethod = "add"#am#"Operands";
}

class PISAOperandName<string name> {
  string pm = "print"#name;
  AsmOperandClass pmc = PISAAsmOperand<name>;
  string em = "encode"#name;
  string dm = "decode"#name;
}

class PISAAsmRegisterOperandPMC<RegisterClass rc, string sig>
    : PISAAsmOperand<rc#"Opnd", sig> {
  let RenderMethod = "add"#Signature;
}

class PISARegisterOperand<RegisterClass rc>
  : RegisterOperand<rc> {
  defvar name = "RegOpnd";
  defvar RCIDStr = "PISA::"#!cast<string>(rc)#"RegClassID";
  string Signature = name#"("#RCIDStr#")";

  let PrintMethod = "print"#Signature;
  let EncoderMethod = "encode"#Signature;
  let DecoderMethod = "decode"#Signature;
  let ParserMatchClass = PISAAsmRegisterOperandPMC<rc, Signature>;
}

class PISALocalVariableOperand
  : Operand<i32>, PISAOperandName<"LocalVariableOpnd"> {
  let PrintMethod = pm;
  let ParserMatchClass = pmc;
  let EncoderMethod = em;
  let DecoderMethod = dm;
}

def FrameIndex : PISALocalVariableOperand;

class PISAImmediateOperand<ValueType ty, string name>
  : Operand<ty>, PISAOperandName<name> {
  let OperandType = "OPERAND_IMMEDIATE";
  let PrintMethod = pm;
  let ParserMatchClass = pmc;
  let EncoderMethod = em;
  let DecoderMethod = dm;
}

class PISAImmediateRangeOperand<ValueType ty, string name, int min, int max>
  : PISAImmediateOperand<ty, name>,
    TImmLeaf<ty, "return Imm >= " # min # " && Imm <= " # max # ";">
{}

class PISABfnOpcodeOperand<ValueType ty, string name>
  : PISAImmediateOperand<ty, name> {
  let PrintMethod = "printBfnOpcode";
}


class PISASymbolOperand<ValueType ty, string name>
  : Operand<ty>, PISAOperandName<name> {
  let OperandType = "OPERAND_PCREL";
  let PrintMethod = pm;
  let ParserMatchClass = pmc;
  let EncoderMethod = em;
  let DecoderMethod = dm;
}

// Immediate address offset operand for the load/store
class AddrOffImm<ValueType VT> : Operand<VT>{
  let OperandType = "OPERAND_IMMEDIATE";
  let PrintMethod = "printAddrOffsetImm";
}

def PISAGlobalVariableOpnd : PISASymbolOperand<OtherVT, "GlobalVariableOpnd">;

// function name as operand (addrof)
def FunctionOpnd : PISASymbolOperand<OtherVT, "FunctionOpnd"> {
  // print the same as a global variable operand, i.e. "@func"
  let PrintMethod = "printGlobalVariableOpnd";
}

class IsQualifier<string Name, string QName> {
  defvar QIndex = !find(Name, QName#".");
  defvar QIndexAtEnd = !if(!gt(!size(Name), !size(QName)), !sub(!size(Name), !size(QName)), 0);
  defvar IsAtEnd = !if(!eq(QIndex, -1), !eq(!substr(Name, QIndexAtEnd, !size(QName)), QName), 0);
  bit ret = !or(!ne(QIndex, -1), IsAtEnd);
}

class ExtractVectorIndex<string name> {
  int ret = !foldl(-1, !range(0, 64), acc, idx,
                   !if(IsQualifier<name, "."#idx>.ret, idx, acc));
}

class _PISAInst<string Name = "UNKNOWN"> : Instruction {
  field bits<16> Inst;

  let Namespace = "PISA";
  let DecoderNamespace = "PISA";

  // !cast<void>(directive)
  let UseNamedOperandTable = 1;
}

class PISAInst<dag outs, dag ins, string asmstr, list<dag> pattern = []>
  : _PISAInst<asmstr> {
  dag OutOperandList = outs;
  dag InOperandList = ins;
  let AsmString = asmstr;
  let Pattern = pattern;
}

// pisaVT should be a container class only instead of a ValueType.
class pisaVT<ValueType vt, string d, RegisterClass rc, bit isHasRC = 1> {
  ValueType VT = vt;
  string directive = d;
  string opcode = !subst(".", "_", !if(vt.isFP, directive, !subst("b", "", directive)));
  string name = !subst("_", "", !cast<string>(NAME));

  string bitwidth = vt.Size#b;
  // mdirective specifies directive for memory operations (e.g. .32b for .f type)
  string mdirective = !if(!and(vt.isFP, !not(vt.isVector)), "."#bitwidth, directive);
  RegisterClass RC = rc;
  DAGOperand ImmOpnd;
  DAGOperand RegOpnd = !if(isHasRC, PISARegisterOperand<rc>, ?);
}

// Subclasses of pisaIntVT, pisaFloatVT, pisaPtrVT, pisaVectorVT, and pisaVoidVT
// are inherited from pisaVT.
// The main goal is to add the functionailty of !isa<subclass>,
// e.g., !isa<pisaIntVT>(_i16) return 1 to tell that it is an int type.
class pisaIntVT<ValueType vt, string d, RegisterClass rc> :
  pisaVT<vt, d, rc> {
  let ImmOpnd = PISAImmediateOperand <vt, "Imm"#vt.Size#"Opnd">;
}

class pisaFloatVT<ValueType vt, string d, RegisterClass rc> :
  pisaVT<vt, d, rc> {
  let ImmOpnd = PISAImmediateOperand <vt, "FpImm"#vt.Size#"Opnd">;
}

class pisaPtrVT<ValueType vt, string d, RegisterClass rc> :
  pisaVT<vt, d, rc>;

class pisaVectorVT<ValueType vt, string d, RegisterClass rc> :
  pisaVT<vt, d, rc>;

class pisaVoidVT<ValueType vt, string d, RegisterClass rc> :
  pisaVT<vt, d, rc, 0>;

def p0 : PtrValueType<i64, 0>;
def p1 : PtrValueType<i64, 1>;
def p2 : PtrValueType<i64, 2>;
def p3 : PtrValueType<i32, 3>;
def p4 : PtrValueType<i32, 4>;

//                        <ValueType, directive,  RegisterClass>;
def _void   : pisaVoidVT<isVoid,      "",         ?            >;
def _i1     : pisaIntVT<i1,           ".1b",      Pred         >;
def _i8     : pisaIntVT<i8,           ".8b",      Reg8b        >;
def _i16    : pisaIntVT<i16,          ".16b",     Reg16b       >;
def _i32    : pisaIntVT<i32,          ".32b",     Reg32b       >;
def _i64    : pisaIntVT<i64,          ".64b",     Reg64b       >;
def _i128   : pisaIntVT<i128,         ".128b",    Reg128b      >;
def _f16    : pisaFloatVT<f16,        ".hf",      Reg16b       >;
def _bf16   : pisaFloatVT<bf16,       ".bf",      Reg16b       >;
def _f32    : pisaFloatVT<f32,        ".f",       Reg32b       >;
def _f64    : pisaFloatVT<f64,        ".df",      Reg64b       >;
def _p0     : pisaPtrVT<p0,           ".64b",     Reg64b       >;
def _p1     : pisaPtrVT<p1,           ".64b",     Reg64b       >;
def _p2     : pisaPtrVT<p2,           ".64b",     Reg64b       >;
def _p3     : pisaPtrVT<p3,           ".32b",     Reg32b       >;
def _p4     : pisaPtrVT<p4,           ".32b",     Reg32b       >;
def _v2i8   : pisaVectorVT<v2i8,      ".v2.8b",   RegV2_8b     >;
def _v3i8   : pisaVectorVT<v3i8,      ".v3.8b",   RegV3_8b     >;
def _v4i8   : pisaVectorVT<v4i8,      ".v4.8b",   RegV4_8b     >;
def _v2i16  : pisaVectorVT<v2i16,     ".v2.16b",  RegV2_16b    >;
def _v3i16  : pisaVectorVT<v3i16,     ".v3.16b",  RegV3_16b    >;
def _v4i16  : pisaVectorVT<v4i16,     ".v4.16b",  RegV4_16b    >;
def _v2i32  : pisaVectorVT<v2i32,     ".v2.32b",  RegV2_32b    >;
def _v3i32  : pisaVectorVT<v3i32,     ".v3.32b",  RegV3_32b    >;
def _v4i32  : pisaVectorVT<v4i32,     ".v4.32b",  RegV4_32b    >;
def _v5i32  : pisaVectorVT<v5i32,     ".v5.32b",  RegV5_32b    >;
def _v6i32  : pisaVectorVT<v6i32,     ".v6.32b",  RegV6_32b    >;
def _v7i32  : pisaVectorVT<v7i32,     ".v7.32b",  RegV7_32b    >;
def _v8i32  : pisaVectorVT<v8i32,     ".v8.32b",  RegV8_32b    >;
def _v16i32 : pisaVectorVT<v16i32,    ".v16.32b", RegV16_32b   >;
def _v32i32 : pisaVectorVT<v32i32,    ".v32.32b", RegV32_32b   >;
def _v64i32 : pisaVectorVT<v64i32,    ".v64.32b", RegV64_32b   >;
def _v2i64  : pisaVectorVT<v2i64,     ".v2.64b",  RegV2_64b    >;
def _v3i64  : pisaVectorVT<v3i64,     ".v3.64b",  RegV3_64b    >;
def _v4i64  : pisaVectorVT<v4i64,     ".v4.64b",  RegV4_64b    >;
def _v2f16  : pisaVectorVT<v2f16,     ".v2.16b",  RegV2_16b    >;
def _v3f16  : pisaVectorVT<v3f16,     ".v3.16b",  RegV3_16b    >;
def _v4f16  : pisaVectorVT<v4f16,     ".v4.16b",  RegV4_16b    >;
def _v2bf16 : pisaVectorVT<v2bf16,    ".v2.16b",  RegV2_16b    >;
def _v3bf16 : pisaVectorVT<v3bf16,    ".v3.16b",  RegV3_16b    >;
def _v4bf16 : pisaVectorVT<v4bf16,    ".v4.16b",  RegV4_16b    >;
def _v2f32  : pisaVectorVT<v2f32,     ".v2.32b",  RegV2_32b    >;
def _v3f32  : pisaVectorVT<v3f32,     ".v3.32b",  RegV3_32b    >;
def _v4f32  : pisaVectorVT<v4f32,     ".v4.32b",  RegV4_32b    >;
def _v5f32  : pisaVectorVT<v5f32,     ".v5.32b",  RegV5_32b    >;
def _v6f32  : pisaVectorVT<v6f32,     ".v6.32b",  RegV6_32b    >;
def _v7f32  : pisaVectorVT<v7f32,     ".v7.32b",  RegV7_32b    >;
def _v8f32  : pisaVectorVT<v8f32,     ".v8.32b",  RegV8_32b    >;
def _v2f64  : pisaVectorVT<v2f64,     ".v2.64b",  RegV2_64b    >;
def _v3f64  : pisaVectorVT<v3f64,     ".v3.64b",  RegV3_64b    >;
def _v4f64  : pisaVectorVT<v4f64,     ".v4.64b",  RegV4_64b    >;

class PISAValueTypes {
  list<pisaVT> Integer = [_i8, _i16, _i32, _i64];
  list<pisaVT> Float = [_bf16, _f16, _f32, _f64];
  list<pisaVT> Ptr = [_p0, _p1, _p2, _p3, _p4];
  list<pisaVT> Vector = [_v2i8, _v2i16, _v2i32, _v2i64,
                         _v3i8, _v3i16, _v3i32, _v3i64,
                         _v4i8, _v4i16, _v4i32, _v4i64,
                         _v8i32];
  list<pisaVT> LargeVector = [_v5i32, _v6i32, _v7i32, _v16i32, _v32i32, _v64i32];
  // types that can be used in mov instruction
  // - register moves are limited to 128 bits
  list<pisaVT> MovImm = !listconcat(Integer, Float, [_i128]);
  list<pisaVT> MovReg = !filter(VT, !listconcat(MovImm, Vector), !le(VT.VT.Size, 128));
  // types that can be used as function parameters or return values
  list<pisaVT> CallReturn = !listconcat(Integer, Vector, LargeVector);
  // types that can be used in ld/st instructions
  list<pisaVT> LoadStore = [_i8, _i16, _i32, _i64,
                            _v2i8, _v2i16, _v2i32, _v2i64,
                            _v3i32, _v3i64,
                            _v4i8, _v4i16, _v4i32, _v4i64,
                            _v8i32];
  // types that can be used in ld.param instructions
  list<pisaVT> LoadParam = !listconcat(LoadStore, LargeVector);
  // types that can be used in atomic load/store
  list<pisaVT> AtomicLoadStore = [_i16, _i32, _i64, _i128];
  // types that can be used in lifetime.start
  list<pisaVT> LifetimeTypes = !listconcat([_i1, _i8, _i16, _i32, _i64, _i128],
                                           Vector, LargeVector);

}
def VTs: PISAValueTypes;

class getVectorType<int numElements, int bitSize> {
  pisaVT VT = !if(!eq(numElements, 1),
                  !cast<pisaVT>("_i"#bitSize),
                  !cast<pisaVT>("_v"#numElements#"i"#bitSize));
}

class FenceFlushOp {}
def evict     : FenceFlushOp;
def inval     : FenceFlushOp;
def clean     : FenceFlushOp;

class FenceScope {}
def subgroup  : FenceScope;
def workgroup : FenceScope;
def gpu       : FenceScope;
def system    : FenceScope;

class AddressSpace<int n, pisaVT vt, string _name = NAME> {
  int num = n;
  pisaVT VT = vt;
  PtrValueType PVT = PtrValueType<vt.VT, n>; // !cast<PtrValueType>("p"#num);
  DAGOperand RC = vt.RegOpnd;
  string name = _name;
  string directive = !if(!empty(name), "", "."#name);
}
def private : AddressSpace<4, _i32>;
def global  : AddressSpace<1, _i64>;
def const   : AddressSpace<2, _i64>;
def shared  : AddressSpace<3, _i32>;
def generic : AddressSpace<0, _i64>;
def default : AddressSpace<0, _i64, "">;
// FIXME: Rename address space as storage space.
// Special storage spaces.

class StoreAddrspacePat<AddressSpace AS, PatFrag pat> :
      PatFrag<(ops node:$data, node:$ptr), (pat node:$data, node:$ptr)> {
  let IsStore = true;
  let IsUnindexed = true;
  let IsNonExtLoad = true;
  let AddressSpaces = [ AS.num ];
}

class LoadAddrspacePat<AddressSpace AS, PatFrag pat> :
      PatFrag<(ops node:$ptr), (pat node:$ptr)> {
  let IsLoad = true;
  let IsUnindexed = true;
  let IsNonExtLoad = true;
  let AddressSpaces = [ AS.num ];
}


class UnaryAtomAddrspacePat<AddressSpace Addrspace, PatFrag pat>
    : PatFrag<(ops node:$ptr), (pat node:$ptr)> {
  let AddressSpaces = [Addrspace.num];
  let IsAtomic = 1;
}

class BinaryAtomAddrspacePat<AddressSpace Addrspace, PatFrag pat>
    : PatFrag<(ops node:$ptr, node:$other), (pat node:$ptr, node:$other)> {
  let AddressSpaces = [ Addrspace.num ];
  let IsAtomic = 1;
}

// Variant of BinaryAtomAddrspacePat that also checks the result register type
// in GlobalISel, needed to disambiguate bf16 from f16 (both are 16-bit).
class BinaryAtomAddrspaceMemVTPat<AddressSpace Addrspace, PatFrag pat,
                                  string MemVTCheck>
    : PatFrag<(ops node:$ptr, node:$other), (pat node:$ptr, node:$other)> {
  let AddressSpaces = [ Addrspace.num ];
  let IsAtomic = 1;
  let GISelPredicateCode = [{ return }] # MemVTCheck # [{; }];
}

// match cmpxchg instruction
class TernaryAtomAddrspacePat<AddressSpace Addrspace, SDPatternOperator pat> :
      PatFrag<(ops node:$ptr, node:$opnd1, node:$opnd2), (pat node:$ptr, node:$opnd1, node:$opnd2)> {
  let AddressSpaces = [ Addrspace.num ];
  let IsAtomic = 1;
  let IsLoad = true;
  let IsStore = true;
}

/********************************* PISAInst class hierarchy *********************************/
//                          +--------------------------------+
//                          | _PISAInst:                     |
//                          |   defines Inst, Namespace,     |
//             (Layer 1)    |           DecoderNamespace,    |
//                          |           UseNamedOperandTable |
//                          +--------------------------------+
//                             |             |            |
//             +---------------+             |            +------------------+
//             v                             v                               v
// +--------------------------+  +--------------------------+  +----------------------------+
// | PISAInst1:               |  | PISAInst2:               |  | PISAInst3:                 |
// |   defines srcRC,         |  |   defines src0RC, src1RC,|  |   defines src0RC, src1RC,  |
// |           dstRC,         |  |           dstRC,         |  |           src2RC, dstRC,   |
// |           AsmString,     |  |           AsmString,     |  |           AsmString,       |
// |           OutOperandList |  |           OutOperandList |  |           OutOperandList   |
// +--------------------------+  +--------------------------+  +----------------------------+
//             |                             |                               |
//             v                             v                               v
// +--------------------------+  +--------------------------+  +----------------------------+
// | PISAInst_r,              |  | PISAInst_rr, PISAInst_ri,|  | PISAInst_rrr, PISAInst_irr,|
// | PISAInst_i:              |  | PISAInst_ir, PISAInst_ii:|  | PISAInst_rri, PISAInst_iri,|
// |   defines InOperandList, |  |   defines InOperandList, |  | PISAInst_rir, PISAInst_iir,|
// |           Imm opd        |  |           Imm opd        |  | PISAInst_rii, PISAInst_irr,|
// +--------------------------+  +--------------------------+  | PISAInst_iii:              |
//             |                             |                 |    defines InOperandList,  |
//             |                             |                 |            Imm opd         |
//             |                             |                 +----------------------------+
//             |                             |                               |
//             | (Classes below are Layer 4) |                               |
//             | (x is either r or i)        |                               |
//             v                             v                               v
// +--------------------------+  +--------------------------+  +----------------------------+
// | UnaryInst_x:             |  | BinInst_xx:              |  | TernaryInst_xxx            |
// |   defines Pattern        |  |   defines Pattern        |  |   defines Pattern          |
// |                          |  |                          |  |                            |
// | See also -               |  | See also -               |  | See also -                 |
// |   FP2IntRndmodeInst_r    |  |   CmpBinaryInstReg_rx    |  |   MadInst_rxx              |
// |   Int2FPRndmodeInst_r    |  |   CmpBinaryInstPred_rx   |  |                            |
// |                          |  |                          |  |                            |
// +--------------------------+  +--------------------------+  +----------------------------+
//
// Description: This is the class hierarchy for PISAInst. Each layer defines different fields
//              to share common code. The normal usage is to inherit a class from Layer 3
//              with desired patterns like rr, ri, etc, and to define its own 'Pattern' for
//              DAG selection. All these classes are for non-memory operations. We need to
//              define another formats for memop.
//

class PISAInst1<string name, pisaVT dstVT, pisaVT srcVT> : _PISAInst<name> {
  DAGOperand srcRC = srcVT.RegOpnd;
  DAGOperand dstRC = dstVT.RegOpnd;
  let AsmString = !if(!isa<pisaVoidVT>(dstVT),
    name#" \t$src;",
    name#" \t$dst, $src;");
  let OutOperandList = !if(!isa<pisaVoidVT>(dstVT),
    (outs),
    (outs dstRC:$dst));
}

class PISAInst_r<string name, pisaVT dstVT, pisaVT srcVT> : PISAInst1<name, dstVT, srcVT> {
  let InOperandList = (ins srcRC:$src);
}

class PISAInst_i<string name, pisaVT dstVT, pisaVT srcVT> : PISAInst1<name, dstVT, srcVT> {
  SDNode immVal = !if(!isa<pisaFloatVT>(srcVT), fpimm, imm);
  DAGOperand immOpd = srcVT.ImmOpnd;
  let InOperandList = (ins immOpd:$src);
}

class PISAInst2<string name, pisaVT dstVT, pisaVT src0VT, pisaVT src1VT> : _PISAInst<name> {
  DAGOperand src0RC = src0VT.RegOpnd;
  DAGOperand src1RC = src1VT.RegOpnd;
  DAGOperand dstRC = dstVT.RegOpnd;
  let AsmString = !if(!isa<pisaVoidVT>(dstVT),
    name#" \t$src0, $src1;",
    name#" \t$dst, $src0, $src1;");
  let OutOperandList = !if(!isa<pisaVoidVT>(dstVT),
    (outs),
    (outs dstRC:$dst));
}

class PISAInst_rr<string name, pisaVT dstVT, pisaVT src0VT, pisaVT src1VT> : PISAInst2<name, dstVT, src0VT, src1VT> {
  let InOperandList = (ins src0RC:$src0, src1RC:$src1);
}

class PISAInst_ri<string name, pisaVT dstVT, pisaVT src0VT, pisaVT src1VT> : PISAInst2<name, dstVT, src0VT, src1VT> {
  SDNode immVal1 = !if(!isa<pisaFloatVT>(src1VT), fpimm, imm);
  DAGOperand immOpd1 = src1VT.ImmOpnd;
  let InOperandList = (ins src0RC:$src0, immOpd1:$src1);
}

class PISAInst_ir<string name, pisaVT dstVT, pisaVT src0VT, pisaVT src1VT> : PISAInst2<name, dstVT, src0VT, src1VT> {
  SDNode immVal0 = !if(!isa<pisaFloatVT>(src0VT), fpimm, imm);
  DAGOperand immOpd0 = src0VT.ImmOpnd;
  let InOperandList = (ins immOpd0:$src0, src1RC:$src1);
}

class PISAInst_ii<string name, pisaVT dstVT, pisaVT src0VT, pisaVT src1VT> : PISAInst2<name, dstVT, src0VT, src1VT> {
  SDNode immVal0 = !if(!isa<pisaFloatVT>(src0VT), fpimm, imm);
  SDNode immVal1 = !if(!isa<pisaFloatVT>(src1VT), fpimm, imm);
  DAGOperand immOpd0 = src0VT.ImmOpnd;
  DAGOperand immOpd1 = src1VT.ImmOpnd;
  let InOperandList = (ins immOpd0:$src0, immOpd1:$src1);
}

class PISAInst3<string name, pisaVT dstVT, pisaVT src0VT, pisaVT src1VT, pisaVT src2VT> : _PISAInst<name> {
  DAGOperand src0RC = src0VT.RegOpnd;
  DAGOperand src1RC = src1VT.RegOpnd;
  DAGOperand src2RC = src2VT.RegOpnd;
  DAGOperand dstRC = dstVT.RegOpnd;
  let AsmString = !if(!isa<pisaVoidVT>(dstVT),
    name#" \t$src0, $src1, $src2;",
    name#" \t$dst, $src0, $src1, $src2;");
  let OutOperandList = !if(!isa<pisaVoidVT>(dstVT),
    (outs),
    (outs dstRC:$dst));
}

class PISAInst_rrr<string name, pisaVT dstVT, pisaVT src0VT, pisaVT src1VT, pisaVT src2VT> : PISAInst3<name, dstVT, src0VT, src1VT, src2VT> {
  let InOperandList = (ins src0RC:$src0, src1RC:$src1, src2RC:$src2);
}

class PISAInst_rri<string name, pisaVT dstVT, pisaVT src0VT, pisaVT src1VT, pisaVT src2VT> : PISAInst3<name, dstVT, src0VT, src1VT, src2VT> {
  SDNode immVal2 = !if(!isa<pisaFloatVT>(src2VT), fpimm, imm);
  DAGOperand immOpd2 = src2VT.ImmOpnd;
  let InOperandList = (ins src0RC:$src0, src1RC:$src1, immOpd2:$src2);
}

class PISAInst_rir<string name, pisaVT dstVT, pisaVT src0VT, pisaVT src1VT, pisaVT src2VT> : PISAInst3<name, dstVT, src0VT, src1VT, src2VT> {
  SDNode immVal1 = !if(!isa<pisaFloatVT>(src1VT), fpimm, imm);
  DAGOperand immOpd1 = src1VT.ImmOpnd;
  let InOperandList = (ins src0RC:$src0, immOpd1:$src1, src2RC:$src2);
}

class PISAInst_rii<string name, pisaVT dstVT, pisaVT src0VT, pisaVT src1VT, pisaVT src2VT> : PISAInst3<name, dstVT, src0VT, src1VT, src2VT> {
  SDNode immVal1 = !if(!isa<pisaFloatVT>(src1VT), fpimm, imm);
  SDNode immVal2 = !if(!isa<pisaFloatVT>(src2VT), fpimm, imm);
  DAGOperand immOpd1 = src1VT.ImmOpnd;
  DAGOperand immOpd2 = src2VT.ImmOpnd;
  let InOperandList = (ins src0RC:$src0, immOpd1:$src1, immOpd2:$src2);
}

class PISAInst_irr<string name, pisaVT dstVT, pisaVT src0VT, pisaVT src1VT, pisaVT src2VT> : PISAInst3<name, dstVT, src0VT, src1VT, src2VT> {
  SDNode immVal0 = !if(!isa<pisaFloatVT>(src0VT), fpimm, imm);
  DAGOperand immOpd0 = src0VT.ImmOpnd;
  let InOperandList = (ins immOpd0:$src0, src1RC:$src1, src2RC:$src2);
}

class PISAInst_iri<string name, pisaVT dstVT, pisaVT src0VT, pisaVT src1VT, pisaVT src2VT> : PISAInst3<name, dstVT, src0VT, src1VT, src2VT> {
  SDNode immVal0 = !if(!isa<pisaFloatVT>(src0VT), fpimm, imm);
  SDNode immVal2 = !if(!isa<pisaFloatVT>(src2VT), fpimm, imm);
  DAGOperand immOpd0 = src0VT.ImmOpnd;
  DAGOperand immOpd2 = src2VT.ImmOpnd;
  let InOperandList = (ins immOpd0:$src0, src1RC:$src1, immOpd2:$src2);
}

class PISAInst_iir<string name, pisaVT dstVT, pisaVT src0VT, pisaVT src1VT, pisaVT src2VT> : PISAInst3<name, dstVT, src0VT, src1VT, src2VT> {
  SDNode immVal0 = !if(!isa<pisaFloatVT>(src0VT), fpimm, imm);
  SDNode immVal1 = !if(!isa<pisaFloatVT>(src1VT), fpimm, imm);
  DAGOperand immOpd0 = src0VT.ImmOpnd;
  DAGOperand immOpd1 = src1VT.ImmOpnd;
  let InOperandList = (ins immOpd0:$src0, immOpd1:$src1, src2RC:$src2);
}

class PISAInst_iii<string name, pisaVT dstVT, pisaVT src0VT, pisaVT src1VT, pisaVT src2VT> : PISAInst3<name, dstVT, src0VT, src1VT, src2VT> {
  SDNode immVal0 = !if(!isa<pisaFloatVT>(src0VT), fpimm, imm);
  SDNode immVal1 = !if(!isa<pisaFloatVT>(src1VT), fpimm, imm);
  SDNode immVal2 = !if(!isa<pisaFloatVT>(src2VT), fpimm, imm);
  DAGOperand immOpd0 = src0VT.ImmOpnd;
  DAGOperand immOpd1 = src1VT.ImmOpnd;
  DAGOperand immOpd2 = src2VT.ImmOpnd;
  let InOperandList = (ins immOpd0:$src0, immOpd1:$src1, immOpd2:$src2);
}

class PISAInst4<string name, pisaVT dstVT, pisaVT src0VT, pisaVT src1VT, pisaVT src2VT, pisaVT src3VT> : _PISAInst<name> {
  DAGOperand src0RC = src0VT.RegOpnd;
  DAGOperand src1RC = src1VT.RegOpnd;
  DAGOperand src2RC = src2VT.RegOpnd;
  DAGOperand src3RC = src3VT.RegOpnd;
  DAGOperand dstRC = dstVT.RegOpnd;
  let AsmString = !if(!isa<pisaVoidVT>(dstVT),
    name#" \t$src0, $src1, $src2, $src3;",
    name#" \t$dst, $src0, $src1, $src2, $src3;");
  let OutOperandList = !if(!isa<pisaVoidVT>(dstVT),
    (outs),
    (outs dstRC:$dst));
}

class PISAInst_rrrr<string name, pisaVT dstVT, pisaVT src0VT, pisaVT src1VT, pisaVT src2VT, pisaVT src3VT> : PISAInst4<name, dstVT, src0VT, src1VT, src2VT, src3VT> {
  let InOperandList = (ins src0RC:$src0, src1RC:$src1, src2RC:$src2, src3RC:$src3);
}

class PISAInst5<string name, pisaVT dstVT, pisaVT src0VT, pisaVT src1VT, pisaVT src2VT, pisaVT src3VT, pisaVT src4VT> : _PISAInst<name> {
  DAGOperand src0RC = src0VT.RegOpnd;
  DAGOperand src1RC = src1VT.RegOpnd;
  DAGOperand src2RC = src2VT.RegOpnd;
  DAGOperand src3RC = src3VT.RegOpnd;
  DAGOperand src4RC = src4VT.RegOpnd;
  DAGOperand dstRC = dstVT.RegOpnd;
  let AsmString = !if(!isa<pisaVoidVT>(dstVT),
    name#" \t$src0, $src1, $src2, $src3, $src4;",
    name#" \t$dst, $src0, $src1, $src2, $src3, $src4;");
  let OutOperandList = !if(!isa<pisaVoidVT>(dstVT),
    (outs),
    (outs dstRC:$dst));
}

/****************************** End of PISAInst class hierarchy *****************************/

/******************************* PISAMemInst class hierarchy ********************************/
//                          +--------------------------------+
//                          | _PISAInst:                     |
//                          |   defines Inst, Namespace,     |
//             (Layer 1)    |           DecoderNamespace,    |
//                          |           UseNamedOperandTable |
//                          +--------------------------------+
//                             |             |            |
//             +---------------+             |            +------------------+
//             v                             v                               v
// +--------------------------+  +--------------------------+  +--------------------------------+
// | PISAStore:               |  | PISALoad:                |  | PISABinaryAtom:                |
// |   defines InOperandList, |  |   defines InOperandList, |  | PISATernaryAtom:               |
// |           OutOperandList,|  |           OutOperandList,|  |    defines InOperandList,      |
// |           AsmString,     |  |           AsmString,     |  |            OutOperandList,     |
// |           Pattern        |  |           Pattern        |  |            AsmString,          |
// |                          |  |                          |  |            Pattern             |
// +--------------------------+  +--------------------------+  +--------------------------------+
//
//
// Description: This is the class hierarchy for PISAMemInst. The classes shown above define basic
//              structures of PISAMemInst for load, store and atomic operations, like inputs,
//              outputs, printing formats, patterns, etc. Each memory operation can have different
//              operand types for the addressing mode. We use MemProperty class to define
//              variants of formats for the operand types in a form of PISAMem_xx, where x
//              can be either v(Var), r(Reg), or i(Imm) for the corresponding operand.
//

def ADDR_rr : ComplexPattern<iPTR, 2, "SelectAddr_rr", [], [], 2>;
def ADDR_ri : ComplexPattern<iPTR, 2, "SelectAddr_ri", [], [], 1>;

class MEM_rr<AddressSpace AS> : Operand<i64> {
  let MIOperandInfo = (ops AS.RC, AS.RC);
  let PrintMethod   = "printMemOperand";
  let ParserMatchClass = PISAAsmOperand<"MemRR">;
}

class MEM_ri<AddressSpace AS> : Operand<i64> {
  let MIOperandInfo = (ops AS.RC, i64imm);
  let PrintMethod   = "printMemOperand";
  let ParserMatchClass = PISAAsmOperand<"MemRI">;
}

// A special memory operand for the kernel parameter *only*.
def MEM_ii : Operand<i64>, ComplexPattern<i64, 2, "selectParamSlot_ii", [], []> {
  let MIOperandInfo = (ops i32imm, i64imm);
  let PrintMethod   = "printParamMemOperand";
  let ParserMatchClass = PISAAsmOperand<"MemII">;
}

def MEM_ir : Operand<i64>, ComplexPattern<i64, 2, "selectParamSlot_ir", [], []> {
  let MIOperandInfo = (ops i32imm, Reg32b);
  let PrintMethod   = "printParamMemOperand";
  let ParserMatchClass = PISAAsmOperand<"MemIR">;
}

def gi_ADDR_rr : GIComplexOperandMatcher<s64, "SelectAddr_rr">, GIComplexPatternEquiv<ADDR_rr>;
def gi_ADDR_ri : GIComplexOperandMatcher<s64, "SelectAddr_ri">, GIComplexPatternEquiv<ADDR_ri>;
def gi_PARAM_SLOT_ii : GIComplexOperandMatcher<s64, "selectParamSlot_ii">, GIComplexPatternEquiv<MEM_ii>;
def gi_PARAM_SLOT_ir : GIComplexOperandMatcher<s64, "selectParamSlot_ir">, GIComplexPatternEquiv<MEM_ir>;

class MemProperty<AddressSpace as, dag i, ComplexPattern pattern> {
  dag Ins = i;
  ComplexPattern Pattern = pattern;
  AddressSpace AS = as;
}

class PISAMem_rr<AddressSpace AS> : MemProperty<AS, (ins MEM_rr<AS>:$addr), ADDR_rr>;
class PISAMem_ri<AddressSpace AS> : MemProperty<AS, (ins MEM_ri<AS>:$addr), ADDR_ri>;

// atomics syncscope("<target-scope>") definitions
defvar AtomicScopeControlDefault = 255;
defm AtomicScopeControl : EnumOptionOpndWithDefaultOpsDef<"AtomicScopeControl", false, AtomicScopeControlDefault>;
class AtomicScopeControlEntry<string name, string mnemonic, bits<8> val>
    : EnumOptionEntry<AtomicScopeControlClassID, name, val, "AtomicScopeControl", mnemonic>,
      GenericEnum {
  let FilterClass = "AtomicScopeControlEntry";
}
multiclass AtomicScopeControlDef<string mnemonic, bits<8> value> {
  defvar entry = !if(!empty(mnemonic), "NONE", !toupper(!subst(".", "_", mnemonic)));
  defvar EnumName = AtomicScopeControl.EnumPrefix # "_" # entry;
  def EnumName : AtomicScopeControlEntry<entry, mnemonic, value>;
}

// Must be in sync with llvm::pisa::MemoryScope definitions in
// llvm/include/llvm/IR/PISAIntrinsicUtils.h
defm NONE      : AtomicScopeControlDef<"", AtomicScopeControlDefault>;
defm SYSTEM    : AtomicScopeControlDef<"system",    0>;
defm GPU       : AtomicScopeControlDef<"gpu",       1>;
defm WORKGROUP : AtomicScopeControlDef<"workgroup", 2>;

// atomics cache control definitions
defm AtomicCacheControl : EnumOptionOpndWithDefaultOpsDef<"AtomicCacheControl", false, 0>;
class AtomicCacheControlEntry<string name, string mnemonic, bits<8> val>
    : EnumOptionEntry<AtomicCacheControlClassID, name, val, "AtomicCacheControl", mnemonic>,
      GenericEnum {
  let FilterClass = "AtomicCacheControlEntry";
}
multiclass AtomicCacheControlDef<string mnemonic, bits<8> value> {
  defvar entry = !if(!empty(mnemonic), "NONE", !toupper(!subst(".", "_", mnemonic)));
  defvar EnumName = AtomicCacheControl.EnumPrefix # "_" # entry;
  def EnumName : AtomicCacheControlEntry<entry, mnemonic, value>;
}
defm NONE           : AtomicCacheControlDef<"",     0>;
defm L1UC_L2UC_L3UC : AtomicCacheControlDef<"uc",   2>;
defm L1UC_L2UC_L3WB : AtomicCacheControlDef<"L3wb", 3>;
defm L1UC_L2WB_L3UC : AtomicCacheControlDef<"L2wb", 4>;

// Fence memory ordering definitions (matches llvm::AtomicOrdering)
defm FenceMemOrder : EnumOptionOpndWithDefaultOpsDef<"FenceMemOrder", false, 6>;
class FenceMemOrderEntry<string name, string mnemonic, bits<8> val>
    : EnumOptionEntry<FenceMemOrderClassID, name, val, "FenceMemOrder", mnemonic>,
      GenericEnum {
  let FilterClass = "FenceMemOrderEntry";
}
multiclass FenceMemOrderDef<string mnemonic, bits<8> value> {
  defvar entry = !toupper(mnemonic);
  defvar EnumName = FenceMemOrder.EnumPrefix # "_" # entry;
  def EnumName : FenceMemOrderEntry<entry, mnemonic, value>;
}
defm ACQUIRE : FenceMemOrderDef<"acquire", 4>;
defm RELEASE : FenceMemOrderDef<"release", 5>;
defm ACQ_REL : FenceMemOrderDef<"acq_rel", 6>;
defm SEQ_CST : FenceMemOrderDef<"seq_cst", 7>;


// load cache control definitions
defm LoadCacheControl : EnumOptionOpndWithDefaultOpsDef<"LoadCacheControl", false, 0>;
class LoadCacheControlEntry<string name, string mnemonic, bits<8> val>
    : EnumOptionEntry<LoadCacheControlClassID, name, val, "LoadCacheControl", mnemonic>,
      GenericEnum {
  let FilterClass = "LoadCacheControlEntry";
}
multiclass LoadCacheControlDef<string mnemonic, bits<8> value> {
  defvar entry = !if(!empty(mnemonic), "NONE", !toupper(!subst(".", "_", mnemonic)));
  defvar EnumName = LoadCacheControl.EnumPrefix # "_" # entry;
  def EnumName : LoadCacheControlEntry<entry, mnemonic, value>;
}
defm NONE           : LoadCacheControlDef<"",               0>;
defm L1UC_L2UC_L3UC : LoadCacheControlDef<"L1uc.L2uc.L3uc", 2>;
defm L1UC_L2UC_L3C  : LoadCacheControlDef<"L1uc.L2uc.L3c",  3>;
defm L1UC_L2C_L3UC  : LoadCacheControlDef<"L1uc.L2c.L3uc",  4>;
defm L1UC_L2C_L3C   : LoadCacheControlDef<"L1uc.L2c.L3c",   5>;
defm L1C_L2UC_L3UC  : LoadCacheControlDef<"L1c.L2uc.L3uc",  6>;
defm L1C_L2UC_L3C   : LoadCacheControlDef<"L1c.L2uc.L3c",   7>;
defm L1C_L2C_L3UC   : LoadCacheControlDef<"L1c.L2c.L3uc",   8>;
defm L1C_L2C_L3C    : LoadCacheControlDef<"L1c.L2c.L3c",    9>;
defm L1S_L2UC_L3UC  : LoadCacheControlDef<"L1s.L2uc.L3uc", 10>;
defm L1S_L2UC_L3C   : LoadCacheControlDef<"L1s.L2uc.L3c",  11>;
defm L1S_L2C_L3UC   : LoadCacheControlDef<"L1s.L2c.L3uc",  12>;
defm L1S_L2C_L3C    : LoadCacheControlDef<"L1s.L2c.L3c",   13>;
defm RI             : LoadCacheControlDef<"ri",            14>;

// store cache control definitions
defm StoreCacheControl : EnumOptionOpndWithDefaultOpsDef<"StoreCacheControl", false, 0>;
class StoreCacheControlEntry<string name, string mnemonic, bits<8> val>
    : EnumOptionEntry<StoreCacheControlClassID, name, val, "StoreCacheControl", mnemonic>,
      GenericEnum {
  let FilterClass = "StoreCacheControlEntry";
}
multiclass StoreCacheControlDef<string mnemonic, bits<8> value> {
  defvar entry = !if(!empty(mnemonic), "NONE", !toupper(!subst(".", "_", mnemonic)));
  defvar EnumName = StoreCacheControl.EnumPrefix # "_" # entry;
  def EnumName : StoreCacheControlEntry<entry, mnemonic, value>;
}
defm NONE           : StoreCacheControlDef<"",                0>;
defm L1UC_L2UC_L3UC : StoreCacheControlDef<"L1uc.L2uc.L3uc",  2>;
defm L1UC_L2UC_L3WB : StoreCacheControlDef<"L1uc.L2uc.L3wb",  3>;
defm L1UC_L2WB_L3UC : StoreCacheControlDef<"L1uc.L2wb.L3uc",  4>;
defm L1UC_L2WB_L3WB : StoreCacheControlDef<"L1uc.L2wb.L3wb",  5>;
defm L1WT_L2UC_L3UC : StoreCacheControlDef<"L1wt.L2uc.L3uc",  6>;
defm L1WT_L2UC_L3WB : StoreCacheControlDef<"L1wt.L2uc.L3wb",  7>;
defm L1WT_L2WB_L3UC : StoreCacheControlDef<"L1wt.L2wb.L3uc",  8>;
defm L1WT_L2WB_L3WB : StoreCacheControlDef<"L1wt.L2wb.L3wb",  9>;
defm L1S_L2UC_L3UC  : StoreCacheControlDef<"L1s.L2uc.L3uc",  10>;
defm L1S_L2UC_L3WB  : StoreCacheControlDef<"L1s.L2uc.L3wb",  11>;
defm L1S_L2WB_L3UC  : StoreCacheControlDef<"L1s.L2wb.L3uc",  12>;
defm L1WB_L2UC_L3UC : StoreCacheControlDef<"L1wb.L2uc.L3uc", 13>;
defm L1WB_L2WB_L3UC : StoreCacheControlDef<"L1wb.L2wb.L3uc", 14>;
defm L1WB_L2UC_L3WB : StoreCacheControlDef<"L1wb.L2uc.L3wb", 15>;


// load instruction
class PISALoad<string directive, AddressSpace AS, pisaVT VT, MemProperty M, PatFrag node> : _PISAInst<directive> {
  DAGOperand dstRC = VT.RegOpnd;
  dag OutOperandList = (outs dstRC:$dst);
  dag InOperandList = !con(M.Ins, (ins LoadCacheControlOpnd:$cachehint));
  string AsmString = directive#"${cachehint}"#VT.mdirective#"\t $dst, $addr;";
  list<dag> Pattern = [(set VT.VT:$dst, (LoadAddrspacePat<AS, node> M.Pattern:$addr))];
}
class PISAAtomicLoad<string directive, AddressSpace AS, pisaVT VT, MemProperty M, PatFrag node> : _PISAInst<directive> {
  DAGOperand dstRC = VT.RegOpnd;
  dag OutOperandList = (outs dstRC:$dst);
  dag InOperandList = !con(M.Ins, (ins AtomicScopeControlOpnd:$scope, AtomicCacheControlOpnd:$cachehint));
  string AsmString = directive#"${scope}${cachehint}"#VT.mdirective#"\t $dst, $addr;";
  list<dag> Pattern = [(set VT.VT:$dst, (LoadAddrspacePat<AS, node> M.Pattern:$addr))];
}

// store instruction
class PISAStore<string directive, AddressSpace AS, pisaVT VT, MemProperty M, PatFrag node> : _PISAInst<directive> {
  DAGOperand srcRC = VT.RegOpnd;
  dag OutOperandList = (outs);
  dag InOperandList = !con(M.Ins, (ins srcRC:$data, StoreCacheControlOpnd:$cachehint));
  string AsmString = directive#"${cachehint}"#VT.mdirective#"\t $addr, $data;";
  list<dag> Pattern = [(set (StoreAddrspacePat<AS, node> VT.VT:$data, M.Pattern:$addr))];
}
class PISAAtomicStore<string directive, AddressSpace AS, pisaVT VT, MemProperty M, PatFrag node> : _PISAInst<directive> {
  DAGOperand srcRC = VT.RegOpnd;
  dag OutOperandList = (outs);
  dag InOperandList = !con(M.Ins, (ins srcRC:$data, AtomicScopeControlOpnd:$scope, AtomicCacheControlOpnd:$cachehint));
  string AsmString = directive#"${scope}${cachehint}"#VT.mdirective#"\t $addr, $data;";
  list<dag> Pattern = [(set (StoreAddrspacePat<AS, node> VT.VT:$data, M.Pattern:$addr))];
}


// The definitions of $dst, $src0 and $src1 are defined in the PISA SPEC
// https://intel.github.io/pisa/instructions_memory.html#iatom
class PISAUnaryAtom<string directive, string op, AddressSpace AS, pisaVT VT,
                    MemProperty M, PatFrag node> : _PISAInst<directive#"."#op#VT.directive> {
  DAGOperand dstRC = VT.RegOpnd;
  dag OutOperandList = (outs dstRC:$dst);
  dag InOperandList = !con(M.Ins, (ins AtomicScopeControlOpnd:$scope, AtomicCacheControlOpnd:$cachehint));
  string AsmString =
      directive#"${scope}${cachehint}."#op#VT.directive#"\t $dst, $addr;";
  list<dag> Pattern = [(set VT.VT:$dst,
      (UnaryAtomAddrspacePat<AS, node> M.Pattern:$addr))];
}

class PISABinaryAtom<string directive, string op, AddressSpace AS, pisaVT VT,
                     MemProperty M, PatFrag node> : _PISAInst<directive#"."#op#VT.directive> {
  DAGOperand dstRC = VT.RegOpnd;
  DAGOperand srcRC = VT.RegOpnd;
  dag OutOperandList = (outs dstRC:$dst);
  dag InOperandList = !con(M.Ins, (ins srcRC:$src0, AtomicScopeControlOpnd:$scope, AtomicCacheControlOpnd:$cachehint));
  string AsmString = directive#"${scope}${cachehint}."#op#VT.directive#"\t $dst, $addr, $src0;";
  list<dag> Pattern = [(set VT.VT:$dst,
      (BinaryAtomAddrspacePat<AS, node> M.Pattern:$addr, VT.VT:$src0))];
}

// cmpxchg instruction
class PISATernaryAtom<string directive, string op, AddressSpace AS, pisaVT VT,
                      MemProperty M, SDPatternOperator node> : _PISAInst<directive#"."#op#VT.directive> {
  DAGOperand dstRC = VT.RegOpnd;
  DAGOperand srcRC = VT.RegOpnd;
  dag OutOperandList = (outs dstRC:$dst);
  dag InOperandList = !con(M.Ins, (ins srcRC:$src0, srcRC:$src1, AtomicScopeControlOpnd:$scope, AtomicCacheControlOpnd:$cachehint));
  string AsmString = directive#"${scope}${cachehint}."#op#VT.directive#"\t $dst, $addr, $src0, $src1;";
  list<dag> Pattern = [(set VT.VT:$dst, (TernaryAtomAddrspacePat<AS, node> M.Pattern:$addr, VT.VT:$src0, VT.VT:$src1))];
}

// Remap to PISA[MemOp]_xx for readability, where x represents either r(Reg), or i(Imm) for the corresponding operand
class PISALoad_rr        <string d, AddressSpace AS, pisaVT VT, PatFrag node> : PISALoad        <d, AS, VT, PISAMem_rr<AS>, node>;
class PISAAtomicLoad_rr  <string d, AddressSpace AS, pisaVT VT, PatFrag node> : PISAAtomicLoad  <d, AS, VT, PISAMem_rr<AS>, node>;
class PISAStore_rr       <string d, AddressSpace AS, pisaVT VT, PatFrag node> : PISAStore       <d, AS, VT, PISAMem_rr<AS>, node>;
class PISAAtomicStore_rr <string d, AddressSpace AS, pisaVT VT, PatFrag node> : PISAAtomicStore <d, AS, VT, PISAMem_rr<AS>, node>;
class PISAUnaryAtom_rr<string d, string op, AddressSpace AS, pisaVT VT,
                       PatFrag node>
    : PISAUnaryAtom<d, op, AS, VT, PISAMem_rr<AS>, node>;
class PISABinaryAtom_rr <string d, string op, AddressSpace AS, pisaVT VT, PatFrag node>
                                                                             : PISABinaryAtom <d, op, AS, VT, PISAMem_rr<AS>, node>;
class PISATernaryAtom_rr<string d, string op, AddressSpace AS, pisaVT VT, SDPatternOperator node>
                                                                             : PISATernaryAtom<d, op, AS, VT, PISAMem_rr<AS>, node>;

class PISALoad_ri        <string d, AddressSpace AS, pisaVT VT, PatFrag node> : PISALoad        <d, AS, VT, PISAMem_ri<AS>, node>;
class PISAAtomicLoad_ri  <string d, AddressSpace AS, pisaVT VT, PatFrag node> : PISAAtomicLoad  <d, AS, VT, PISAMem_ri<AS>, node>;
class PISAStore_ri       <string d, AddressSpace AS, pisaVT VT, PatFrag node> : PISAStore       <d, AS, VT, PISAMem_ri<AS>, node>;
class PISAAtomicStore_ri <string d, AddressSpace AS, pisaVT VT, PatFrag node> : PISAAtomicStore <d, AS, VT, PISAMem_ri<AS>, node>;
class PISAUnaryAtom_ri<string d, string op, AddressSpace AS, pisaVT VT,
                       PatFrag node>
    : PISAUnaryAtom<d, op, AS, VT, PISAMem_ri<AS>, node>;
class PISABinaryAtom_ri <string d, string op, AddressSpace AS, pisaVT VT, PatFrag node>
                                                                             : PISABinaryAtom <d, op, AS, VT, PISAMem_ri<AS>, node>;
class PISATernaryAtom_ri<string d, string op, AddressSpace AS, pisaVT VT, SDPatternOperator node>
                                                                             : PISATernaryAtom<d, op, AS, VT, PISAMem_ri<AS>, node>;


/**************************** End of PISAMemInst class hierarchy ****************************/
