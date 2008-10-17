//===- TargetSelectionDAG.td - Common code for DAG isels ---*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the target-independent interfaces used by SelectionDAG
// instruction selection generators.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Selection DAG Type Constraint definitions.
//
// Note that the semantics of these constraints are hard coded into tblgen.  To
// modify or add constraints, you have to hack tblgen.
//

class SDTypeConstraint<int opnum> {
  int OperandNum = opnum;
}

// SDTCisVT - The specified operand has exactly this VT.
class SDTCisVT<int OpNum, ValueType vt> : SDTypeConstraint<OpNum> {
  ValueType VT = vt;
}

class SDTCisPtrTy<int OpNum> : SDTypeConstraint<OpNum>;

// SDTCisInt - The specified operand is has integer type.
class SDTCisInt<int OpNum> : SDTypeConstraint<OpNum>;

// SDTCisFP - The specified operand is has floating point type.
class SDTCisFP<int OpNum> : SDTypeConstraint<OpNum>;

// SDTCisSameAs - The two specified operands have identical types.
class SDTCisSameAs<int OpNum, int OtherOp> : SDTypeConstraint<OpNum> {
  int OtherOperandNum = OtherOp;
}

// SDTCisVTSmallerThanOp - The specified operand is a VT SDNode, and its type is
// smaller than the 'Other' operand.
class SDTCisVTSmallerThanOp<int OpNum, int OtherOp> : SDTypeConstraint<OpNum> {
  int OtherOperandNum = OtherOp;
}

class SDTCisOpSmallerThanOp<int SmallOp, int BigOp> : SDTypeConstraint<SmallOp>{
  int BigOperandNum = BigOp;
}

/// SDTCisIntVectorOfSameSize - This indicates that ThisOp and OtherOp are
/// vector types, and that ThisOp is the result of 
/// MVT::getIntVectorWithNumElements with the number of elements
/// that ThisOp has.
class SDTCisIntVectorOfSameSize<int ThisOp, int OtherOp>
  : SDTypeConstraint<ThisOp> {
  int OtherOpNum = OtherOp;
}

/// SDTCisEltOfVec - This indicates that ThisOp is a scalar type of the same
/// type as the element type of OtherOp, which is a vector type.
class SDTCisEltOfVec<int ThisOp, int OtherOp>
  : SDTypeConstraint<ThisOp> {
  int OtherOpNum = OtherOp;
}

//===----------------------------------------------------------------------===//
// Selection DAG Type Profile definitions.
//
// These use the constraints defined above to describe the type requirements of
// the various nodes.  These are not hard coded into tblgen, allowing targets to
// add their own if needed.
//

// SDTypeProfile - This profile describes the type requirements of a Selection
// DAG node.
class SDTypeProfile<int numresults, int numoperands,
                    list<SDTypeConstraint> constraints> {
  int NumResults = numresults;
  int NumOperands = numoperands;
  list<SDTypeConstraint> Constraints = constraints;
}

// Builtin profiles.
def SDTIntLeaf: SDTypeProfile<1, 0, [SDTCisInt<0>]>;      // for 'imm'.
def SDTFPLeaf : SDTypeProfile<1, 0, [SDTCisFP<0>]>;       // for 'fpimm'.
def SDTPtrLeaf: SDTypeProfile<1, 0, [SDTCisPtrTy<0>]>;      // for '&g'.
def SDTOther  : SDTypeProfile<1, 0, [SDTCisVT<0, OtherVT>]>; // for 'vt'.
def SDTUNDEF  : SDTypeProfile<1, 0, []>; // for 'undef'.
def SDTUnaryOp  : SDTypeProfile<1, 1, []>; // bitconvert

def SDTIntBinOp : SDTypeProfile<1, 2, [   // add, and, or, xor, udiv, etc.
  SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>, SDTCisInt<0>
]>;
def SDTIntShiftOp : SDTypeProfile<1, 2, [   // shl, sra, srl
  SDTCisSameAs<0, 1>, SDTCisInt<0>, SDTCisInt<2>
]>;
def SDTFPBinOp : SDTypeProfile<1, 2, [      // fadd, fmul, etc.
  SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>, SDTCisFP<0>
]>;
def SDTFPSignOp : SDTypeProfile<1, 2, [      // fcopysign.
  SDTCisSameAs<0, 1>, SDTCisFP<0>, SDTCisFP<2>
]>;
def SDTFPTernaryOp : SDTypeProfile<1, 3, [      // fmadd, fnmsub, etc.
  SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>, SDTCisSameAs<0, 3>, SDTCisFP<0>
]>;
def SDTIntUnaryOp : SDTypeProfile<1, 1, [   // ctlz
  SDTCisSameAs<0, 1>, SDTCisInt<0>
]>;
def SDTIntExtendOp : SDTypeProfile<1, 1, [  // sext, zext, anyext
  SDTCisInt<0>, SDTCisInt<1>, SDTCisOpSmallerThanOp<1, 0>
]>;
def SDTIntTruncOp  : SDTypeProfile<1, 1, [  // trunc
  SDTCisInt<0>, SDTCisInt<1>, SDTCisOpSmallerThanOp<0, 1>
]>;
def SDTFPUnaryOp  : SDTypeProfile<1, 1, [   // fneg, fsqrt, etc
  SDTCisSameAs<0, 1>, SDTCisFP<0>
]>;
def SDTFPRoundOp  : SDTypeProfile<1, 1, [   // fround
  SDTCisFP<0>, SDTCisFP<1>, SDTCisOpSmallerThanOp<0, 1>
]>;
def SDTFPExtendOp  : SDTypeProfile<1, 1, [   // fextend
  SDTCisFP<0>, SDTCisFP<1>, SDTCisOpSmallerThanOp<1, 0>
]>;
def SDTIntToFPOp : SDTypeProfile<1, 1, [   // [su]int_to_fp 
  SDTCisFP<0>, SDTCisInt<1>
]>;
def SDTFPToIntOp : SDTypeProfile<1, 1, [   // fp_to_[su]int 
  SDTCisInt<0>, SDTCisFP<1>
]>;
def SDTExtInreg : SDTypeProfile<1, 2, [   // sext_inreg
  SDTCisSameAs<0, 1>, SDTCisInt<0>, SDTCisVT<2, OtherVT>,
  SDTCisVTSmallerThanOp<2, 1>
]>;

def SDTSetCC : SDTypeProfile<1, 3, [ // setcc
  SDTCisInt<0>, SDTCisSameAs<1, 2>, SDTCisVT<3, OtherVT>
]>;

def SDTSelect : SDTypeProfile<1, 3, [ // select 
  SDTCisInt<1>, SDTCisSameAs<0, 2>, SDTCisSameAs<2, 3>
]>;

def SDTSelectCC : SDTypeProfile<1, 5, [ // select_cc
  SDTCisSameAs<1, 2>, SDTCisSameAs<3, 4>, SDTCisSameAs<0, 3>,
  SDTCisVT<5, OtherVT>
]>;

def SDTBr : SDTypeProfile<0, 1, [ // br
  SDTCisVT<0, OtherVT>
]>;

def SDTBrcond : SDTypeProfile<0, 2, [ // brcond
  SDTCisInt<0>, SDTCisVT<1, OtherVT>
]>;

def SDTBrind : SDTypeProfile<0, 1, [ // brind
  SDTCisPtrTy<0>
]>;

def SDTNone : SDTypeProfile<0, 0, []>; // ret, trap

def SDTLoad : SDTypeProfile<1, 1, [ // load
  SDTCisPtrTy<1>  
]>;

def SDTStore : SDTypeProfile<0, 2, [ // store
  SDTCisPtrTy<1>  
]>;

def SDTIStore : SDTypeProfile<1, 3, [ // indexed store
  SDTCisSameAs<0, 2>, SDTCisPtrTy<0>, SDTCisPtrTy<3>
]>;

def SDTVecShuffle : SDTypeProfile<1, 3, [
  SDTCisSameAs<0, 1>, SDTCisSameAs<1, 2>, SDTCisIntVectorOfSameSize<3, 0>
]>;
def SDTVecExtract : SDTypeProfile<1, 2, [ // vector extract
  SDTCisEltOfVec<0, 1>, SDTCisPtrTy<2>
]>;
def SDTVecInsert : SDTypeProfile<1, 3, [  // vector insert
  SDTCisEltOfVec<2, 1>, SDTCisSameAs<0, 1>, SDTCisPtrTy<3>
]>;

def STDPrefetch : SDTypeProfile<0, 3, [  // prefetch
  SDTCisPtrTy<0>, SDTCisSameAs<1, 2>, SDTCisInt<1>
]>;

def STDMemBarrier : SDTypeProfile<0, 5, [ // memory barier
  SDTCisSameAs<0,1>,  SDTCisSameAs<0,2>,  SDTCisSameAs<0,3>, SDTCisSameAs<0,4>,
  SDTCisInt<0>
]>;
def STDAtomic3 : SDTypeProfile<1, 3, [
  SDTCisSameAs<0,2>,  SDTCisSameAs<0,3>, SDTCisInt<0>, SDTCisPtrTy<1>
]>;
def STDAtomic2 : SDTypeProfile<1, 2, [
  SDTCisSameAs<0,2>, SDTCisInt<0>, SDTCisPtrTy<1>
]>;

class SDCallSeqStart<list<SDTypeConstraint> constraints> :
        SDTypeProfile<0, 1, constraints>;
class SDCallSeqEnd<list<SDTypeConstraint> constraints> :
        SDTypeProfile<0, 2, constraints>;

//===----------------------------------------------------------------------===//
// Selection DAG Node Properties.
//
// Note: These are hard coded into tblgen.
//
class SDNodeProperty;
def SDNPCommutative : SDNodeProperty;   // X op Y == Y op X
def SDNPAssociative : SDNodeProperty;   // (X op Y) op Z == X op (Y op Z)
def SDNPHasChain    : SDNodeProperty;   // R/W chain operand and result
def SDNPOutFlag     : SDNodeProperty;   // Write a flag result
def SDNPInFlag      : SDNodeProperty;   // Read a flag operand
def SDNPOptInFlag   : SDNodeProperty;   // Optionally read a flag operand
def SDNPMayStore    : SDNodeProperty;   // May write to memory, sets 'mayStore'.
def SDNPMayLoad     : SDNodeProperty;   // May read memory, sets 'mayLoad'.
def SDNPSideEffect  : SDNodeProperty;   // Sets 'HasUnmodelledSideEffects'.
def SDNPMemOperand  : SDNodeProperty;   // Touches memory, has assoc MemOperand

//===----------------------------------------------------------------------===//
// Selection DAG Node definitions.
//
class SDNode<string opcode, SDTypeProfile typeprof,
             list<SDNodeProperty> props = [], string sdclass = "SDNode"> {
  string Opcode  = opcode;
  string SDClass = sdclass;
  list<SDNodeProperty> Properties = props;
  SDTypeProfile TypeProfile = typeprof;
}

def set;
def implicit;
def parallel;
def node;
def srcvalue;

def imm        : SDNode<"ISD::Constant"  , SDTIntLeaf , [], "ConstantSDNode">;
def timm   : SDNode<"ISD::TargetConstant", SDTIntLeaf , [], "ConstantSDNode">;
def fpimm      : SDNode<"ISD::ConstantFP", SDTFPLeaf  , [], "ConstantFPSDNode">;
def vt         : SDNode<"ISD::VALUETYPE" , SDTOther   , [], "VTSDNode">;
def bb         : SDNode<"ISD::BasicBlock", SDTOther   , [], "BasicBlockSDNode">;
def cond       : SDNode<"ISD::CONDCODE"  , SDTOther   , [], "CondCodeSDNode">;
def undef      : SDNode<"ISD::UNDEF"     , SDTUNDEF   , []>;
def globaladdr : SDNode<"ISD::GlobalAddress",         SDTPtrLeaf, [],
                        "GlobalAddressSDNode">;
def tglobaladdr : SDNode<"ISD::TargetGlobalAddress",  SDTPtrLeaf, [],
                         "GlobalAddressSDNode">;
def globaltlsaddr : SDNode<"ISD::GlobalTLSAddress",         SDTPtrLeaf, [],
                          "GlobalAddressSDNode">;
def tglobaltlsaddr : SDNode<"ISD::TargetGlobalTLSAddress",  SDTPtrLeaf, [],
                           "GlobalAddressSDNode">;
def constpool   : SDNode<"ISD::ConstantPool",         SDTPtrLeaf, [],
                         "ConstantPoolSDNode">;
def tconstpool  : SDNode<"ISD::TargetConstantPool",   SDTPtrLeaf, [],
                         "ConstantPoolSDNode">;
def jumptable   : SDNode<"ISD::JumpTable",            SDTPtrLeaf, [],
                         "JumpTableSDNode">;
def tjumptable  : SDNode<"ISD::TargetJumpTable",      SDTPtrLeaf, [],
                         "JumpTableSDNode">;
def frameindex  : SDNode<"ISD::FrameIndex",           SDTPtrLeaf, [],
                         "FrameIndexSDNode">;
def tframeindex : SDNode<"ISD::TargetFrameIndex",     SDTPtrLeaf, [],
                         "FrameIndexSDNode">;
def externalsym : SDNode<"ISD::ExternalSymbol",       SDTPtrLeaf, [],
                         "ExternalSymbolSDNode">;
def texternalsym: SDNode<"ISD::TargetExternalSymbol", SDTPtrLeaf, [],
                         "ExternalSymbolSDNode">;

def add        : SDNode<"ISD::ADD"       , SDTIntBinOp   ,
                        [SDNPCommutative, SDNPAssociative]>;
def sub        : SDNode<"ISD::SUB"       , SDTIntBinOp>;
def mul        : SDNode<"ISD::MUL"       , SDTIntBinOp,
                        [SDNPCommutative, SDNPAssociative]>;
def mulhs      : SDNode<"ISD::MULHS"     , SDTIntBinOp, [SDNPCommutative]>;
def mulhu      : SDNode<"ISD::MULHU"     , SDTIntBinOp, [SDNPCommutative]>;
def sdiv       : SDNode<"ISD::SDIV"      , SDTIntBinOp>;
def udiv       : SDNode<"ISD::UDIV"      , SDTIntBinOp>;
def srem       : SDNode<"ISD::SREM"      , SDTIntBinOp>;
def urem       : SDNode<"ISD::UREM"      , SDTIntBinOp>;
def srl        : SDNode<"ISD::SRL"       , SDTIntShiftOp>;
def sra        : SDNode<"ISD::SRA"       , SDTIntShiftOp>;
def shl        : SDNode<"ISD::SHL"       , SDTIntShiftOp>;
def rotl       : SDNode<"ISD::ROTL"      , SDTIntShiftOp>;
def rotr       : SDNode<"ISD::ROTR"      , SDTIntShiftOp>;
def and        : SDNode<"ISD::AND"       , SDTIntBinOp,
                        [SDNPCommutative, SDNPAssociative]>;
def or         : SDNode<"ISD::OR"        , SDTIntBinOp,
                        [SDNPCommutative, SDNPAssociative]>;
def xor        : SDNode<"ISD::XOR"       , SDTIntBinOp,
                        [SDNPCommutative, SDNPAssociative]>;
def addc       : SDNode<"ISD::ADDC"      , SDTIntBinOp,
                        [SDNPCommutative, SDNPOutFlag]>;
def adde       : SDNode<"ISD::ADDE"      , SDTIntBinOp,
                        [SDNPCommutative, SDNPOutFlag, SDNPInFlag]>;
def subc       : SDNode<"ISD::SUBC"      , SDTIntBinOp,
                        [SDNPOutFlag]>;
def sube       : SDNode<"ISD::SUBE"      , SDTIntBinOp,
                        [SDNPOutFlag, SDNPInFlag]>;
                        
def sext_inreg : SDNode<"ISD::SIGN_EXTEND_INREG", SDTExtInreg>;
def bswap      : SDNode<"ISD::BSWAP"      , SDTIntUnaryOp>;
def ctlz       : SDNode<"ISD::CTLZ"       , SDTIntUnaryOp>;
def cttz       : SDNode<"ISD::CTTZ"       , SDTIntUnaryOp>;
def ctpop      : SDNode<"ISD::CTPOP"      , SDTIntUnaryOp>;
def sext       : SDNode<"ISD::SIGN_EXTEND", SDTIntExtendOp>;
def zext       : SDNode<"ISD::ZERO_EXTEND", SDTIntExtendOp>;
def anyext     : SDNode<"ISD::ANY_EXTEND" , SDTIntExtendOp>;
def trunc      : SDNode<"ISD::TRUNCATE"   , SDTIntTruncOp>;
def bitconvert : SDNode<"ISD::BIT_CONVERT", SDTUnaryOp>;
def extractelt : SDNode<"ISD::EXTRACT_VECTOR_ELT", SDTVecExtract>;
def insertelt  : SDNode<"ISD::INSERT_VECTOR_ELT", SDTVecInsert>;

                        
def fadd       : SDNode<"ISD::FADD"       , SDTFPBinOp, [SDNPCommutative]>;
def fsub       : SDNode<"ISD::FSUB"       , SDTFPBinOp>;
def fmul       : SDNode<"ISD::FMUL"       , SDTFPBinOp, [SDNPCommutative]>;
def fdiv       : SDNode<"ISD::FDIV"       , SDTFPBinOp>;
def frem       : SDNode<"ISD::FREM"       , SDTFPBinOp>;
def fabs       : SDNode<"ISD::FABS"       , SDTFPUnaryOp>;
def fneg       : SDNode<"ISD::FNEG"       , SDTFPUnaryOp>;
def fsqrt      : SDNode<"ISD::FSQRT"      , SDTFPUnaryOp>;
def fsin       : SDNode<"ISD::FSIN"       , SDTFPUnaryOp>;
def fcos       : SDNode<"ISD::FCOS"       , SDTFPUnaryOp>;
def frint      : SDNode<"ISD::FRINT"      , SDTFPUnaryOp>;
def ftrunc     : SDNode<"ISD::FTRUNC"     , SDTFPUnaryOp>;
def fceil      : SDNode<"ISD::FCEIL"      , SDTFPUnaryOp>;
def ffloor     : SDNode<"ISD::FFLOOR"     , SDTFPUnaryOp>;
def fnearbyint : SDNode<"ISD::FNEARBYINT" , SDTFPUnaryOp>;

def fround     : SDNode<"ISD::FP_ROUND"   , SDTFPRoundOp>;
def fextend    : SDNode<"ISD::FP_EXTEND"  , SDTFPExtendOp>;
def fcopysign  : SDNode<"ISD::FCOPYSIGN"  , SDTFPSignOp>;

def sint_to_fp : SDNode<"ISD::SINT_TO_FP" , SDTIntToFPOp>;
def uint_to_fp : SDNode<"ISD::UINT_TO_FP" , SDTIntToFPOp>;
def fp_to_sint : SDNode<"ISD::FP_TO_SINT" , SDTFPToIntOp>;
def fp_to_uint : SDNode<"ISD::FP_TO_UINT" , SDTFPToIntOp>;

def setcc      : SDNode<"ISD::SETCC"      , SDTSetCC>;
def select     : SDNode<"ISD::SELECT"     , SDTSelect>;
def selectcc   : SDNode<"ISD::SELECT_CC"  , SDTSelectCC>;
def vsetcc     : SDNode<"ISD::VSETCC"     , SDTSetCC>;

def brcond     : SDNode<"ISD::BRCOND"     , SDTBrcond, [SDNPHasChain]>;
def brind      : SDNode<"ISD::BRIND"      , SDTBrind,  [SDNPHasChain]>;
def br         : SDNode<"ISD::BR"         , SDTBr,     [SDNPHasChain]>;
def ret        : SDNode<"ISD::RET"        , SDTNone,   [SDNPHasChain]>;
def trap       : SDNode<"ISD::TRAP"       , SDTNone,
                        [SDNPHasChain, SDNPSideEffect]>;

def prefetch   : SDNode<"ISD::PREFETCH"   , STDPrefetch,
                        [SDNPHasChain, SDNPMayLoad, SDNPMayStore]>;

def membarrier : SDNode<"ISD::MEMBARRIER" , STDMemBarrier,
                        [SDNPHasChain, SDNPSideEffect]>;

def atomic_cmp_swap_8 : SDNode<"ISD::ATOMIC_CMP_SWAP_8" , STDAtomic3,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_add_8 : SDNode<"ISD::ATOMIC_LOAD_ADD_8" , STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_swap_8     : SDNode<"ISD::ATOMIC_SWAP_8", STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_sub_8 : SDNode<"ISD::ATOMIC_LOAD_SUB_8" , STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_and_8 : SDNode<"ISD::ATOMIC_LOAD_AND_8" , STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_or_8  : SDNode<"ISD::ATOMIC_LOAD_OR_8" , STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_xor_8 : SDNode<"ISD::ATOMIC_LOAD_XOR_8" , STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_nand_8: SDNode<"ISD::ATOMIC_LOAD_NAND_8", STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_min_8 : SDNode<"ISD::ATOMIC_LOAD_MIN_8", STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_max_8 : SDNode<"ISD::ATOMIC_LOAD_MAX_8", STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_umin_8 : SDNode<"ISD::ATOMIC_LOAD_UMIN_8", STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_umax_8 : SDNode<"ISD::ATOMIC_LOAD_UMAX_8", STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_cmp_swap_16 : SDNode<"ISD::ATOMIC_CMP_SWAP_16" , STDAtomic3,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_add_16 : SDNode<"ISD::ATOMIC_LOAD_ADD_16" , STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_swap_16     : SDNode<"ISD::ATOMIC_SWAP_16", STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_sub_16 : SDNode<"ISD::ATOMIC_LOAD_SUB_16" , STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_and_16 : SDNode<"ISD::ATOMIC_LOAD_AND_16" , STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_or_16  : SDNode<"ISD::ATOMIC_LOAD_OR_16" , STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_xor_16 : SDNode<"ISD::ATOMIC_LOAD_XOR_16" , STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_nand_16: SDNode<"ISD::ATOMIC_LOAD_NAND_16", STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_min_16 : SDNode<"ISD::ATOMIC_LOAD_MIN_16", STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_max_16 : SDNode<"ISD::ATOMIC_LOAD_MAX_16", STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_umin_16 : SDNode<"ISD::ATOMIC_LOAD_UMIN_16", STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_umax_16 : SDNode<"ISD::ATOMIC_LOAD_UMAX_16", STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_cmp_swap_32 : SDNode<"ISD::ATOMIC_CMP_SWAP_32" , STDAtomic3,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_add_32 : SDNode<"ISD::ATOMIC_LOAD_ADD_32" , STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_swap_32     : SDNode<"ISD::ATOMIC_SWAP_32", STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_sub_32 : SDNode<"ISD::ATOMIC_LOAD_SUB_32" , STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_and_32 : SDNode<"ISD::ATOMIC_LOAD_AND_32" , STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_or_32  : SDNode<"ISD::ATOMIC_LOAD_OR_32" , STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_xor_32 : SDNode<"ISD::ATOMIC_LOAD_XOR_32" , STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_nand_32: SDNode<"ISD::ATOMIC_LOAD_NAND_32", STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_min_32 : SDNode<"ISD::ATOMIC_LOAD_MIN_32", STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_max_32 : SDNode<"ISD::ATOMIC_LOAD_MAX_32", STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_umin_32 : SDNode<"ISD::ATOMIC_LOAD_UMIN_32", STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_umax_32 : SDNode<"ISD::ATOMIC_LOAD_UMAX_32", STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_cmp_swap_64 : SDNode<"ISD::ATOMIC_CMP_SWAP_64" , STDAtomic3,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_add_64 : SDNode<"ISD::ATOMIC_LOAD_ADD_64" , STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_swap_64     : SDNode<"ISD::ATOMIC_SWAP_64", STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_sub_64 : SDNode<"ISD::ATOMIC_LOAD_SUB_64" , STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_and_64 : SDNode<"ISD::ATOMIC_LOAD_AND_64" , STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_or_64  : SDNode<"ISD::ATOMIC_LOAD_OR_64" , STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_xor_64 : SDNode<"ISD::ATOMIC_LOAD_XOR_64" , STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_nand_64: SDNode<"ISD::ATOMIC_LOAD_NAND_64", STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_min_64 : SDNode<"ISD::ATOMIC_LOAD_MIN_64", STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_max_64 : SDNode<"ISD::ATOMIC_LOAD_MAX_64", STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_umin_64 : SDNode<"ISD::ATOMIC_LOAD_UMIN_64", STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def atomic_load_umax_64 : SDNode<"ISD::ATOMIC_LOAD_UMAX_64", STDAtomic2,
                    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;

// Do not use ld, st directly. Use load, extload, sextload, zextload, store,
// and truncst (see below).
def ld         : SDNode<"ISD::LOAD"       , SDTLoad,
                        [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def st         : SDNode<"ISD::STORE"      , SDTStore,
                        [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
def ist        : SDNode<"ISD::STORE"      , SDTIStore,
                        [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;

def vector_shuffle : SDNode<"ISD::VECTOR_SHUFFLE", SDTVecShuffle, []>;
def build_vector : SDNode<"ISD::BUILD_VECTOR", SDTypeProfile<1, -1, []>, []>;
def scalar_to_vector : SDNode<"ISD::SCALAR_TO_VECTOR", SDTypeProfile<1, 1, []>,
                              []>;
def vector_extract : SDNode<"ISD::EXTRACT_VECTOR_ELT",
    SDTypeProfile<1, 2, [SDTCisPtrTy<2>]>, []>;
def vector_insert : SDNode<"ISD::INSERT_VECTOR_ELT",
    SDTypeProfile<1, 3, [SDTCisSameAs<0, 1>, SDTCisPtrTy<3>]>, []>;
    
def extract_subreg : SDNode<"ISD::EXTRACT_SUBREG", 
    SDTypeProfile<1, 2, []>>;
def insert_subreg : SDNode<"ISD::INSERT_SUBREG", 
    SDTypeProfile<1, 3, []>>;

// Nodes for intrinsics, you should use the intrinsic itself and let tblgen use
// these internally.  Don't reference these directly.
def intrinsic_void : SDNode<"ISD::INTRINSIC_VOID", 
                            SDTypeProfile<0, -1, [SDTCisPtrTy<0>]>,
                            [SDNPHasChain]>;
def intrinsic_w_chain : SDNode<"ISD::INTRINSIC_W_CHAIN", 
                               SDTypeProfile<1, -1, [SDTCisPtrTy<1>]>,
                               [SDNPHasChain]>;
def intrinsic_wo_chain : SDNode<"ISD::INTRINSIC_WO_CHAIN", 
                                SDTypeProfile<1, -1, [SDTCisPtrTy<1>]>, []>;


//===----------------------------------------------------------------------===//
// Selection DAG Condition Codes

class CondCode; // ISD::CondCode enums
def SETOEQ : CondCode; def SETOGT : CondCode;
def SETOGE : CondCode; def SETOLT : CondCode; def SETOLE : CondCode;
def SETONE : CondCode; def SETO   : CondCode; def SETUO  : CondCode;
def SETUEQ : CondCode; def SETUGT : CondCode; def SETUGE : CondCode;
def SETULT : CondCode; def SETULE : CondCode; def SETUNE : CondCode;

def SETEQ : CondCode; def SETGT : CondCode; def SETGE : CondCode;
def SETLT : CondCode; def SETLE : CondCode; def SETNE : CondCode;


//===----------------------------------------------------------------------===//
// Selection DAG Node Transformation Functions.
//
// This mechanism allows targets to manipulate nodes in the output DAG once a
// match has been formed.  This is typically used to manipulate immediate
// values.
//
class SDNodeXForm<SDNode opc, code xformFunction> {
  SDNode Opcode = opc;
  code XFormFunction = xformFunction;
}

def NOOP_SDNodeXForm : SDNodeXForm<imm, [{}]>;


//===----------------------------------------------------------------------===//
// Selection DAG Pattern Fragments.
//
// Pattern fragments are reusable chunks of dags that match specific things.
// They can take arguments and have C++ predicates that control whether they
// match.  They are intended to make the patterns for common instructions more
// compact and readable.
//

/// PatFrag - Represents a pattern fragment.  This can match something on the
/// DAG, frame a single node to multiply nested other fragments.
///
class PatFrag<dag ops, dag frag, code pred = [{}],
              SDNodeXForm xform = NOOP_SDNodeXForm> {
  dag Operands = ops;
  dag Fragment = frag;
  code Predicate = pred;
  SDNodeXForm OperandTransform = xform;
}

// PatLeaf's are pattern fragments that have no operands.  This is just a helper
// to define immediates and other common things concisely.
class PatLeaf<dag frag, code pred = [{}], SDNodeXForm xform = NOOP_SDNodeXForm>
 : PatFrag<(ops), frag, pred, xform>;

// Leaf fragments.

def vtInt      : PatLeaf<(vt),  [{ return N->getVT().isInteger(); }]>;
def vtFP       : PatLeaf<(vt),  [{ return N->getVT().isFloatingPoint(); }]>;

def immAllOnes : PatLeaf<(imm), [{ return N->isAllOnesValue(); }]>;
def immAllOnesV: PatLeaf<(build_vector), [{
  return ISD::isBuildVectorAllOnes(N);
}]>;
def immAllOnesV_bc: PatLeaf<(bitconvert), [{
  return ISD::isBuildVectorAllOnes(N);
}]>;
def immAllZerosV: PatLeaf<(build_vector), [{
  return ISD::isBuildVectorAllZeros(N);
}]>;
def immAllZerosV_bc: PatLeaf<(bitconvert), [{
  return ISD::isBuildVectorAllZeros(N);
}]>;



// Other helper fragments.
def not  : PatFrag<(ops node:$in), (xor node:$in, immAllOnes)>;
def vnot : PatFrag<(ops node:$in), (xor node:$in, immAllOnesV)>;
def vnot_conv : PatFrag<(ops node:$in), (xor node:$in, immAllOnesV_bc)>;
def ineg : PatFrag<(ops node:$in), (sub 0, node:$in)>;

// load fragments.
def load : PatFrag<(ops node:$ptr), (ld node:$ptr), [{
  LoadSDNode *LD = cast<LoadSDNode>(N);
  return LD->getExtensionType() == ISD::NON_EXTLOAD &&
         LD->getAddressingMode() == ISD::UNINDEXED;
}]>;

// extending load fragments.
def extloadi1  : PatFrag<(ops node:$ptr), (ld node:$ptr), [{
  LoadSDNode *LD = cast<LoadSDNode>(N);
  return LD->getExtensionType() == ISD::EXTLOAD &&
         LD->getAddressingMode() == ISD::UNINDEXED &&
         LD->getMemoryVT() == MVT::i1;
}]>;
def extloadi8  : PatFrag<(ops node:$ptr), (ld node:$ptr), [{
  LoadSDNode *LD = cast<LoadSDNode>(N);
  return LD->getExtensionType() == ISD::EXTLOAD &&
         LD->getAddressingMode() == ISD::UNINDEXED &&
         LD->getMemoryVT() == MVT::i8;
}]>;
def extloadi16 : PatFrag<(ops node:$ptr), (ld node:$ptr), [{
  LoadSDNode *LD = cast<LoadSDNode>(N);
  return LD->getExtensionType() == ISD::EXTLOAD &&
         LD->getAddressingMode() == ISD::UNINDEXED &&
         LD->getMemoryVT() == MVT::i16;
}]>;
def extloadi32 : PatFrag<(ops node:$ptr), (ld node:$ptr), [{
  LoadSDNode *LD = cast<LoadSDNode>(N);
  return LD->getExtensionType() == ISD::EXTLOAD &&
         LD->getAddressingMode() == ISD::UNINDEXED &&
         LD->getMemoryVT() == MVT::i32;
}]>;
def extloadf32 : PatFrag<(ops node:$ptr), (ld node:$ptr), [{
  LoadSDNode *LD = cast<LoadSDNode>(N);
  return LD->getExtensionType() == ISD::EXTLOAD &&
         LD->getAddressingMode() == ISD::UNINDEXED &&
         LD->getMemoryVT() == MVT::f32;
}]>;
def extloadf64 : PatFrag<(ops node:$ptr), (ld node:$ptr), [{
  LoadSDNode *LD = cast<LoadSDNode>(N);
  return LD->getExtensionType() == ISD::EXTLOAD &&
         LD->getAddressingMode() == ISD::UNINDEXED &&
         LD->getMemoryVT() == MVT::f64;
}]>;

def sextloadi1  : PatFrag<(ops node:$ptr), (ld node:$ptr), [{
  LoadSDNode *LD = cast<LoadSDNode>(N);
  return LD->getExtensionType() == ISD::SEXTLOAD &&
         LD->getAddressingMode() == ISD::UNINDEXED &&
         LD->getMemoryVT() == MVT::i1;
}]>;
def sextloadi8  : PatFrag<(ops node:$ptr), (ld node:$ptr), [{
  LoadSDNode *LD = cast<LoadSDNode>(N);
  return LD->getExtensionType() == ISD::SEXTLOAD &&
         LD->getAddressingMode() == ISD::UNINDEXED &&
         LD->getMemoryVT() == MVT::i8;
}]>;
def sextloadi16 : PatFrag<(ops node:$ptr), (ld node:$ptr), [{
  LoadSDNode *LD = cast<LoadSDNode>(N);
  return LD->getExtensionType() == ISD::SEXTLOAD &&
         LD->getAddressingMode() == ISD::UNINDEXED &&
         LD->getMemoryVT() == MVT::i16;
}]>;
def sextloadi32 : PatFrag<(ops node:$ptr), (ld node:$ptr), [{
  LoadSDNode *LD = cast<LoadSDNode>(N);
  return LD->getExtensionType() == ISD::SEXTLOAD &&
         LD->getAddressingMode() == ISD::UNINDEXED &&
         LD->getMemoryVT() == MVT::i32;
}]>;

def zextloadi1  : PatFrag<(ops node:$ptr), (ld node:$ptr), [{
  LoadSDNode *LD = cast<LoadSDNode>(N);
  return LD->getExtensionType() == ISD::ZEXTLOAD &&
         LD->getAddressingMode() == ISD::UNINDEXED &&
         LD->getMemoryVT() == MVT::i1;
}]>;
def zextloadi8  : PatFrag<(ops node:$ptr), (ld node:$ptr), [{
  LoadSDNode *LD = cast<LoadSDNode>(N);
  return LD->getExtensionType() == ISD::ZEXTLOAD &&
         LD->getAddressingMode() == ISD::UNINDEXED &&
         LD->getMemoryVT() == MVT::i8;
}]>;
def zextloadi16 : PatFrag<(ops node:$ptr), (ld node:$ptr), [{
  LoadSDNode *LD = cast<LoadSDNode>(N);
  return LD->getExtensionType() == ISD::ZEXTLOAD &&
         LD->getAddressingMode() == ISD::UNINDEXED &&
         LD->getMemoryVT() == MVT::i16;
}]>;
def zextloadi32 : PatFrag<(ops node:$ptr), (ld node:$ptr), [{
  LoadSDNode *LD = cast<LoadSDNode>(N);
  return LD->getExtensionType() == ISD::ZEXTLOAD &&
         LD->getAddressingMode() == ISD::UNINDEXED &&
         LD->getMemoryVT() == MVT::i32;
}]>;

// store fragments.
def store : PatFrag<(ops node:$val, node:$ptr),
                    (st node:$val, node:$ptr), [{
  StoreSDNode *ST = cast<StoreSDNode>(N);
  return !ST->isTruncatingStore() &&
         ST->getAddressingMode() == ISD::UNINDEXED;
}]>;

// truncstore fragments.
def truncstorei8 : PatFrag<(ops node:$val, node:$ptr),
                           (st node:$val, node:$ptr), [{
  StoreSDNode *ST = cast<StoreSDNode>(N);
  return ST->isTruncatingStore() && ST->getMemoryVT() == MVT::i8 &&
         ST->getAddressingMode() == ISD::UNINDEXED;
}]>;
def truncstorei16 : PatFrag<(ops node:$val, node:$ptr),
                            (st node:$val, node:$ptr), [{
  StoreSDNode *ST = cast<StoreSDNode>(N);
  return ST->isTruncatingStore() && ST->getMemoryVT() == MVT::i16 &&
         ST->getAddressingMode() == ISD::UNINDEXED;
}]>;
def truncstorei32 : PatFrag<(ops node:$val, node:$ptr),
                            (st node:$val, node:$ptr), [{
  StoreSDNode *ST = cast<StoreSDNode>(N);
  return ST->isTruncatingStore() && ST->getMemoryVT() == MVT::i32 &&
         ST->getAddressingMode() == ISD::UNINDEXED;
}]>;
def truncstoref32 : PatFrag<(ops node:$val, node:$ptr),
                            (st node:$val, node:$ptr), [{
  StoreSDNode *ST = cast<StoreSDNode>(N);
  return ST->isTruncatingStore() && ST->getMemoryVT() == MVT::f32 &&
         ST->getAddressingMode() == ISD::UNINDEXED;
}]>;
def truncstoref64 : PatFrag<(ops node:$val, node:$ptr),
                            (st node:$val, node:$ptr), [{
  StoreSDNode *ST = cast<StoreSDNode>(N);
  return ST->isTruncatingStore() && ST->getMemoryVT() == MVT::f64 &&
         ST->getAddressingMode() == ISD::UNINDEXED;
}]>;

// indexed store fragments.
def pre_store : PatFrag<(ops node:$val, node:$base, node:$offset),
                        (ist node:$val, node:$base, node:$offset), [{
  StoreSDNode *ST = cast<StoreSDNode>(N);
  ISD::MemIndexedMode AM = ST->getAddressingMode();
  return (AM == ISD::PRE_INC || AM == ISD::PRE_DEC) &&
         !ST->isTruncatingStore();
}]>;

def pre_truncsti1 : PatFrag<(ops node:$val, node:$base, node:$offset),
                            (ist node:$val, node:$base, node:$offset), [{
  StoreSDNode *ST = cast<StoreSDNode>(N);
  ISD::MemIndexedMode AM = ST->getAddressingMode();
  return (AM == ISD::PRE_INC || AM == ISD::PRE_DEC) &&
         ST->isTruncatingStore() && ST->getMemoryVT() == MVT::i1;
}]>;
def pre_truncsti8 : PatFrag<(ops node:$val, node:$base, node:$offset),
                            (ist node:$val, node:$base, node:$offset), [{
  StoreSDNode *ST = cast<StoreSDNode>(N);
  ISD::MemIndexedMode AM = ST->getAddressingMode();
  return (AM == ISD::PRE_INC || AM == ISD::PRE_DEC) &&
         ST->isTruncatingStore() && ST->getMemoryVT() == MVT::i8;
}]>;
def pre_truncsti16 : PatFrag<(ops node:$val, node:$base, node:$offset),
                             (ist node:$val, node:$base, node:$offset), [{
  StoreSDNode *ST = cast<StoreSDNode>(N);
  ISD::MemIndexedMode AM = ST->getAddressingMode();
  return (AM == ISD::PRE_INC || AM == ISD::PRE_DEC) &&
         ST->isTruncatingStore() && ST->getMemoryVT() == MVT::i16;
}]>;
def pre_truncsti32 : PatFrag<(ops node:$val, node:$base, node:$offset),
                             (ist node:$val, node:$base, node:$offset), [{
  StoreSDNode *ST = cast<StoreSDNode>(N);
  ISD::MemIndexedMode AM = ST->getAddressingMode();
  return (AM == ISD::PRE_INC || AM == ISD::PRE_DEC) &&
         ST->isTruncatingStore() && ST->getMemoryVT() == MVT::i32;
}]>;
def pre_truncstf32 : PatFrag<(ops node:$val, node:$base, node:$offset),
                             (ist node:$val, node:$base, node:$offset), [{
  StoreSDNode *ST = cast<StoreSDNode>(N);
  ISD::MemIndexedMode AM = ST->getAddressingMode();
  return (AM == ISD::PRE_INC || AM == ISD::PRE_DEC) &&
         ST->isTruncatingStore() && ST->getMemoryVT() == MVT::f32;
}]>;

def post_store : PatFrag<(ops node:$val, node:$ptr, node:$offset),
                         (ist node:$val, node:$ptr, node:$offset), [{
  StoreSDNode *ST = cast<StoreSDNode>(N);
  ISD::MemIndexedMode AM = ST->getAddressingMode();
  return !ST->isTruncatingStore() &&
          (AM == ISD::POST_INC || AM == ISD::POST_DEC);
}]>;

def post_truncsti1 : PatFrag<(ops node:$val, node:$base, node:$offset),
                             (ist node:$val, node:$base, node:$offset), [{
  StoreSDNode *ST = cast<StoreSDNode>(N);
  ISD::MemIndexedMode AM = ST->getAddressingMode();
  return (AM == ISD::POST_INC || AM == ISD::POST_DEC) &&
         ST->isTruncatingStore() && ST->getMemoryVT() == MVT::i1;
}]>;
def post_truncsti8 : PatFrag<(ops node:$val, node:$base, node:$offset),
                             (ist node:$val, node:$base, node:$offset), [{
  StoreSDNode *ST = cast<StoreSDNode>(N);
  ISD::MemIndexedMode AM = ST->getAddressingMode();
  return (AM == ISD::POST_INC || AM == ISD::POST_DEC) &&
         ST->isTruncatingStore() && ST->getMemoryVT() == MVT::i8;
}]>;
def post_truncsti16 : PatFrag<(ops node:$val, node:$base, node:$offset),
                              (ist node:$val, node:$base, node:$offset), [{
  StoreSDNode *ST = cast<StoreSDNode>(N);
  ISD::MemIndexedMode AM = ST->getAddressingMode();
  return (AM == ISD::POST_INC || AM == ISD::POST_DEC) &&
         ST->isTruncatingStore() && ST->getMemoryVT() == MVT::i16;
}]>;
def post_truncsti32 : PatFrag<(ops node:$val, node:$base, node:$offset),
                              (ist node:$val, node:$base, node:$offset), [{
  StoreSDNode *ST = cast<StoreSDNode>(N);
  ISD::MemIndexedMode AM = ST->getAddressingMode();
  return (AM == ISD::POST_INC || AM == ISD::POST_DEC) &&
         ST->isTruncatingStore() && ST->getMemoryVT() == MVT::i32;
}]>;
def post_truncstf32 : PatFrag<(ops node:$val, node:$base, node:$offset),
                              (ist node:$val, node:$base, node:$offset), [{
  StoreSDNode *ST = cast<StoreSDNode>(N);
  ISD::MemIndexedMode AM = ST->getAddressingMode();
  return (AM == ISD::POST_INC || AM == ISD::POST_DEC) &&
         ST->isTruncatingStore() && ST->getMemoryVT() == MVT::f32;
}]>;

// setcc convenience fragments.
def setoeq : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETOEQ)>;
def setogt : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETOGT)>;
def setoge : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETOGE)>;
def setolt : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETOLT)>;
def setole : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETOLE)>;
def setone : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETONE)>;
def seto   : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETO)>;
def setuo  : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETUO)>;
def setueq : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETUEQ)>;
def setugt : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETUGT)>;
def setuge : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETUGE)>;
def setult : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETULT)>;
def setule : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETULE)>;
def setune : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETUNE)>;
def seteq  : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETEQ)>;
def setgt  : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETGT)>;
def setge  : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETGE)>;
def setlt  : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETLT)>;
def setle  : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETLE)>;
def setne  : PatFrag<(ops node:$lhs, node:$rhs),
                     (setcc node:$lhs, node:$rhs, SETNE)>;

//===----------------------------------------------------------------------===//
// Selection DAG Pattern Support.
//
// Patterns are what are actually matched against the target-flavored
// instruction selection DAG.  Instructions defined by the target implicitly
// define patterns in most cases, but patterns can also be explicitly added when
// an operation is defined by a sequence of instructions (e.g. loading a large
// immediate value on RISC targets that do not support immediates as large as
// their GPRs).
//

class Pattern<dag patternToMatch, list<dag> resultInstrs> {
  dag             PatternToMatch  = patternToMatch;
  list<dag>       ResultInstrs    = resultInstrs;
  list<Predicate> Predicates      = [];  // See class Instruction in Target.td.
  int             AddedComplexity = 0;  // See class Instruction in Target.td.
}

// Pat - A simple (but common) form of a pattern, which produces a simple result
// not needing a full list.
class Pat<dag pattern, dag result> : Pattern<pattern, [result]>;

//===----------------------------------------------------------------------===//
// Complex pattern definitions.
//

class CPAttribute;
// Pass the parent Operand as root to CP function rather 
// than the root of the sub-DAG
def CPAttrParentAsRoot : CPAttribute;

// Complex patterns, e.g. X86 addressing mode, requires pattern matching code
// in C++. NumOperands is the number of operands returned by the select function;
// SelectFunc is the name of the function used to pattern match the max. pattern;
// RootNodes are the list of possible root nodes of the sub-dags to match.
// e.g. X86 addressing mode - def addr : ComplexPattern<4, "SelectAddr", [add]>;
//
class ComplexPattern<ValueType ty, int numops, string fn,
                     list<SDNode> roots = [], list<SDNodeProperty> props = [],
                     list<CPAttribute> attrs = []> {
  ValueType Ty = ty;
  int NumOperands = numops;
  string SelectFunc = fn;
  list<SDNode> RootNodes = roots;
  list<SDNodeProperty> Properties = props;
  list<CPAttribute> Attributes = attrs;
}

//===----------------------------------------------------------------------===//
// Dwarf support.
//
def SDT_dwarf_loc : SDTypeProfile<0, 3,
                      [SDTCisInt<0>, SDTCisInt<1>, SDTCisInt<2>]>;
def dwarf_loc : SDNode<"ISD::DEBUG_LOC", SDT_dwarf_loc,[SDNPHasChain]>;
