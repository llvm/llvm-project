//===----- AArch64InstrInfo.td - AArch64 Instruction Info ----*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the AArch64 scalar instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ARM Instruction Predicate Definitions.
//
def HasFPARMv8       : Predicate<"Subtarget->hasFPARMv8()">,
                               AssemblerPredicate<"FeatureFPARMv8", "fp-armv8">;
def HasNEON          : Predicate<"Subtarget->hasNEON()">,
                                 AssemblerPredicate<"FeatureNEON", "neon">;
def HasCrypto        : Predicate<"Subtarget->hasCrypto()">,
                                 AssemblerPredicate<"FeatureCrypto","crypto">;

// Use fused MAC if more precision in FP computation is allowed.
def UseFusedMAC      : Predicate<"(TM.Options.AllowFPOpFusion =="
                                 " FPOpFusion::Fast)">;
include "AArch64InstrFormats.td"

//===----------------------------------------------------------------------===//
// Target-specific ISD nodes and profiles
//===----------------------------------------------------------------------===//

def SDT_A64ret : SDTypeProfile<0, 0, []>;
def A64ret : SDNode<"AArch64ISD::Ret", SDT_A64ret, [SDNPHasChain,
                                                    SDNPOptInGlue,
                                                    SDNPVariadic]>;

// (ins NZCV, Condition, Dest)
def SDT_A64br_cc : SDTypeProfile<0, 3, [SDTCisVT<0, i32>]>;
def A64br_cc : SDNode<"AArch64ISD::BR_CC", SDT_A64br_cc, [SDNPHasChain]>;

// (outs Result), (ins NZCV, IfTrue, IfFalse, Condition)
def SDT_A64select_cc : SDTypeProfile<1, 4, [SDTCisVT<1, i32>,
                                            SDTCisSameAs<0, 2>,
                                            SDTCisSameAs<2, 3>]>;
def A64select_cc : SDNode<"AArch64ISD::SELECT_CC", SDT_A64select_cc>;

// (outs NZCV), (ins LHS, RHS, Condition)
def SDT_A64setcc : SDTypeProfile<1, 3, [SDTCisVT<0, i32>,
                                        SDTCisSameAs<1, 2>]>;
def A64setcc : SDNode<"AArch64ISD::SETCC", SDT_A64setcc>;


// (outs GPR64), (ins)
def A64threadpointer : SDNode<"AArch64ISD::THREAD_POINTER", SDTPtrLeaf>;

// A64 compares don't care about the cond really (they set all flags) so a
// simple binary operator is useful.
def A64cmp : PatFrag<(ops node:$lhs, node:$rhs),
                     (A64setcc node:$lhs, node:$rhs, cond)>;


// When matching a notional (CMP op1, (sub 0, op2)), we'd like to use a CMN
// instruction on the grounds that "op1 - (-op2) == op1 + op2". However, the C
// and V flags can be set differently by this operation. It comes down to
// whether "SInt(~op2)+1 == SInt(~op2+1)" (and the same for UInt). If they are
// then everything is fine. If not then the optimization is wrong. Thus general
// comparisons are only valid if op2 != 0.

// So, finally, the only LLVM-native comparisons that don't mention C and V are
// SETEQ and SETNE. They're the only ones we can safely use CMN for in the
// absence of information about op2.
def equality_cond : PatLeaf<(cond), [{
  return N->get() == ISD::SETEQ || N->get() == ISD::SETNE;
}]>;

def A64cmn : PatFrag<(ops node:$lhs, node:$rhs),
                     (A64setcc node:$lhs, (sub 0, node:$rhs), equality_cond)>;

// There are two layers of indirection here, driven by the following
// considerations.
//     + TableGen does not know CodeModel or Reloc so that decision should be
//       made for a variable/address at ISelLowering.
//     + The output of ISelLowering should be selectable (hence the Wrapper,
//       rather than a bare target opcode)
def SDTAArch64WrapperLarge : SDTypeProfile<1, 4, [SDTCisSameAs<0, 1>,
                                                  SDTCisSameAs<0, 2>,
                                                  SDTCisSameAs<0, 3>,
                                                  SDTCisSameAs<0, 4>,
                                                  SDTCisPtrTy<0>]>;

def A64WrapperLarge :SDNode<"AArch64ISD::WrapperLarge", SDTAArch64WrapperLarge>;

def SDTAArch64WrapperSmall : SDTypeProfile<1, 3, [SDTCisSameAs<0, 1>,
                                                  SDTCisSameAs<1, 2>,
                                                  SDTCisVT<3, i32>,
                                                  SDTCisPtrTy<0>]>;

def A64WrapperSmall :SDNode<"AArch64ISD::WrapperSmall", SDTAArch64WrapperSmall>;


def SDTAArch64GOTLoad : SDTypeProfile<1, 1, [SDTCisPtrTy<0>, SDTCisPtrTy<1>]>;
def A64GOTLoad : SDNode<"AArch64ISD::GOTLoad", SDTAArch64GOTLoad,
                        [SDNPHasChain]>;


// (A64BFI LHS, RHS, LSB, Width)
def SDTA64BFI : SDTypeProfile<1, 4, [SDTCisSameAs<0, 1>,
                                     SDTCisSameAs<1, 2>,
                                     SDTCisVT<3, i64>,
                                     SDTCisVT<4, i64>]>;

def A64Bfi : SDNode<"AArch64ISD::BFI", SDTA64BFI>;

// (A64EXTR HiReg, LoReg, LSB)
def SDTA64EXTR : SDTypeProfile<1, 3, [SDTCisSameAs<0, 1>, SDTCisSameAs<1, 2>,
                                      SDTCisVT<3, i64>]>;
def A64Extr : SDNode<"AArch64ISD::EXTR", SDTA64EXTR>;

// (A64[SU]BFX Field, ImmR, ImmS).
//
// Note that ImmR and ImmS are already encoded for the actual instructions. The
// more natural LSB and Width mix together to form ImmR and ImmS, something
// which TableGen can't handle.
def SDTA64BFX : SDTypeProfile<1, 3, [SDTCisVT<2, i64>, SDTCisVT<3, i64>]>;
def A64Sbfx : SDNode<"AArch64ISD::SBFX", SDTA64BFX>;

def A64Ubfx : SDNode<"AArch64ISD::UBFX", SDTA64BFX>;

class BinOpFrag<dag res> : PatFrag<(ops node:$LHS, node:$RHS), res>;

//===----------------------------------------------------------------------===//
// Call sequence pseudo-instructions
//===----------------------------------------------------------------------===//


def SDT_AArch64Call : SDTypeProfile<0, -1, [SDTCisPtrTy<0>]>;
def AArch64Call : SDNode<"AArch64ISD::Call", SDT_AArch64Call,
                     [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue, SDNPVariadic]>;

def AArch64tcret : SDNode<"AArch64ISD::TC_RETURN", SDT_AArch64Call,
                          [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;

// The TLSDESCCALL node is a variant call which goes to an indirectly calculated
// destination but needs a relocation against a fixed symbol. As such it has two
// certain operands: the callee and the relocated variable.
//
// The TLS ABI only allows it to be selected to a BLR instructin (with
// appropriate relocation).
def SDTTLSDescCall : SDTypeProfile<0, -2, [SDTCisPtrTy<0>, SDTCisPtrTy<1>]>;

def A64tlsdesc_blr : SDNode<"AArch64ISD::TLSDESCCALL", SDTTLSDescCall,
                            [SDNPInGlue, SDNPOutGlue, SDNPHasChain,
                             SDNPVariadic]>;


def SDT_AArch64CallSeqStart : SDCallSeqStart<[ SDTCisPtrTy<0> ]>;
def AArch64callseq_start : SDNode<"ISD::CALLSEQ_START", SDT_AArch64CallSeqStart,
                                  [SDNPHasChain, SDNPOutGlue]>;

def SDT_AArch64CallSeqEnd   : SDCallSeqEnd<[ SDTCisPtrTy<0>, SDTCisPtrTy<1> ]>;
def AArch64callseq_end : SDNode<"ISD::CALLSEQ_END",   SDT_AArch64CallSeqEnd,
                                [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue]>;



// These pseudo-instructions have special semantics by virtue of being passed to
// the InstrInfo constructor. CALLSEQ_START/CALLSEQ_END are produced by
// LowerCall to (in our case) tell the back-end about stack adjustments for
// arguments passed on the stack. Here we select those markers to
// pseudo-instructions which explicitly set the stack, and finally in the
// RegisterInfo we convert them to a true stack adjustment.
let Defs = [XSP], Uses = [XSP] in {
  def ADJCALLSTACKDOWN : PseudoInst<(outs), (ins i64imm:$amt),
                                    [(AArch64callseq_start timm:$amt)]>;

  def ADJCALLSTACKUP : PseudoInst<(outs), (ins i64imm:$amt1, i64imm:$amt2),
                                 [(AArch64callseq_end timm:$amt1, timm:$amt2)]>;
}

//===----------------------------------------------------------------------===//
// Atomic operation pseudo-instructions
//===----------------------------------------------------------------------===//

// These get selected from C++ code as a pretty much direct translation from the
// generic DAG nodes. The one exception is the AtomicOrdering is added as an
// operand so that the eventual lowering can make use of it and choose
// acquire/release operations when required.

let usesCustomInserter = 1, hasCtrlDep = 1, mayLoad = 1, mayStore = 1 in {
multiclass AtomicSizes {
  def _I8 : PseudoInst<(outs GPR32:$dst),
                       (ins GPR64xsp:$ptr, GPR32:$incr, i32imm:$ordering), []>;
  def _I16 : PseudoInst<(outs GPR32:$dst),
                        (ins GPR64xsp:$ptr, GPR32:$incr, i32imm:$ordering), []>;
  def _I32 : PseudoInst<(outs GPR32:$dst),
                        (ins GPR64xsp:$ptr, GPR32:$incr, i32imm:$ordering), []>;
  def _I64 : PseudoInst<(outs GPR64:$dst),
                        (ins GPR64xsp:$ptr, GPR64:$incr, i32imm:$ordering), []>;
}
}

defm ATOMIC_LOAD_ADD  : AtomicSizes;
defm ATOMIC_LOAD_SUB  : AtomicSizes;
defm ATOMIC_LOAD_AND  : AtomicSizes;
defm ATOMIC_LOAD_OR   : AtomicSizes;
defm ATOMIC_LOAD_XOR  : AtomicSizes;
defm ATOMIC_LOAD_NAND : AtomicSizes;
defm ATOMIC_SWAP      : AtomicSizes;
let Defs = [NZCV] in {
  // These operations need a CMP to calculate the correct value
  defm ATOMIC_LOAD_MIN  : AtomicSizes;
  defm ATOMIC_LOAD_MAX  : AtomicSizes;
  defm ATOMIC_LOAD_UMIN : AtomicSizes;
  defm ATOMIC_LOAD_UMAX : AtomicSizes;
}

class AtomicCmpSwap<RegisterClass GPRData>
  : PseudoInst<(outs GPRData:$dst),
               (ins GPR64xsp:$ptr, GPRData:$old, GPRData:$new,
                    i32imm:$ordering), []> {
  let usesCustomInserter = 1;
  let hasCtrlDep = 1;
  let mayLoad = 1;
  let mayStore = 1;
  let Defs = [NZCV];
}

def ATOMIC_CMP_SWAP_I8  : AtomicCmpSwap<GPR32>;
def ATOMIC_CMP_SWAP_I16 : AtomicCmpSwap<GPR32>;
def ATOMIC_CMP_SWAP_I32 : AtomicCmpSwap<GPR32>;
def ATOMIC_CMP_SWAP_I64 : AtomicCmpSwap<GPR64>;

//===----------------------------------------------------------------------===//
// Add-subtract (extended register) instructions
//===----------------------------------------------------------------------===//
// Contains: ADD, ADDS, SUB, SUBS + aliases CMN, CMP

// The RHS of these operations is conceptually a sign/zero-extended
// register, optionally shifted left by 1-4. The extension can be a
// NOP (e.g. "sxtx" sign-extending a 64-bit register to 64-bits) but
// must be specified with one exception:

// If one of the registers is sp/wsp then LSL is an alias for UXTW in
// 32-bit instructions and UXTX in 64-bit versions, the shift amount
// is not optional in that case (but can explicitly be 0), and the
// entire suffix can be skipped (e.g. "add sp, x3, x2").

multiclass extend_operands<string PREFIX, string Diag> {
     def _asmoperand : AsmOperandClass {
         let Name = PREFIX;
         let RenderMethod = "addRegExtendOperands";
         let PredicateMethod = "isRegExtend<A64SE::" # PREFIX # ">";
         let DiagnosticType = "AddSubRegExtend" # Diag;
     }

     def _operand : Operand<i64>,
                    ImmLeaf<i64, [{ return Imm >= 0 && Imm <= 4; }]> {
         let PrintMethod = "printRegExtendOperand<A64SE::" # PREFIX # ">";
         let DecoderMethod = "DecodeRegExtendOperand";
         let ParserMatchClass = !cast<AsmOperandClass>(PREFIX # "_asmoperand");
     }
}

defm UXTB : extend_operands<"UXTB", "Small">;
defm UXTH : extend_operands<"UXTH", "Small">;
defm UXTW : extend_operands<"UXTW", "Small">;
defm UXTX : extend_operands<"UXTX", "Large">;
defm SXTB : extend_operands<"SXTB", "Small">;
defm SXTH : extend_operands<"SXTH", "Small">;
defm SXTW : extend_operands<"SXTW", "Small">;
defm SXTX : extend_operands<"SXTX", "Large">;

def LSL_extasmoperand : AsmOperandClass {
    let Name = "RegExtendLSL";
    let RenderMethod = "addRegExtendOperands";
    let DiagnosticType = "AddSubRegExtendLarge";
}

def LSL_extoperand : Operand<i64> {
    let ParserMatchClass = LSL_extasmoperand;
}


// The patterns for various sign-extensions are a little ugly and
// non-uniform because everything has already been promoted to the
// legal i64 and i32 types. We'll wrap the various variants up in a
// class for use later.
class extend_types {
    dag uxtb; dag uxth; dag uxtw; dag uxtx;
    dag sxtb; dag sxth; dag sxtw; dag sxtx;
    ValueType ty;
    RegisterClass GPR;
}

def extends_to_i64 : extend_types {
    let uxtb = (and (anyext i32:$Rm), 255);
    let uxth = (and (anyext i32:$Rm), 65535);
    let uxtw = (zext i32:$Rm);
    let uxtx = (i64 $Rm);

    let sxtb = (sext_inreg (anyext i32:$Rm), i8);
    let sxth = (sext_inreg (anyext i32:$Rm), i16);
    let sxtw = (sext i32:$Rm);
    let sxtx = (i64 $Rm);

    let ty = i64;
    let GPR = GPR64xsp;
}


def extends_to_i32 : extend_types {
    let uxtb = (and i32:$Rm, 255);
    let uxth = (and i32:$Rm, 65535);
    let uxtw = (i32 i32:$Rm);
    let uxtx = (i32 i32:$Rm);

    let sxtb = (sext_inreg i32:$Rm, i8);
    let sxth = (sext_inreg i32:$Rm, i16);
    let sxtw = (i32 i32:$Rm);
    let sxtx = (i32 i32:$Rm);

    let ty = i32;
    let GPR = GPR32wsp;
}

// Now, six of the extensions supported are easy and uniform: if the source size
// is 32-bits or less, then Rm is always a 32-bit register. We'll instantiate
// those instructions in one block.

// The uxtx/sxtx could potentially be merged in, but three facts dissuaded me:
//     + It would break the naming scheme: either ADDxx_uxtx or ADDww_uxtx would
//       be impossible.
//     + Patterns are very different as well.
//     + Passing different registers would be ugly (more fields in extend_types
//       would probably be the best option).
multiclass addsub_exts<bit sf, bit op, bit S, string asmop,
                       SDPatternOperator opfrag,
                       dag outs, extend_types exts> {
    def w_uxtb : A64I_addsubext<sf, op, S, 0b00, 0b000,
                    outs, (ins exts.GPR:$Rn, GPR32:$Rm, UXTB_operand:$Imm3),
                    !strconcat(asmop, "$Rn, $Rm, $Imm3"),
                    [(opfrag exts.ty:$Rn, (shl exts.uxtb, UXTB_operand:$Imm3))],
                    NoItinerary>;
    def w_uxth : A64I_addsubext<sf, op, S, 0b00, 0b001,
                    outs, (ins exts.GPR:$Rn, GPR32:$Rm, UXTH_operand:$Imm3),
                    !strconcat(asmop, "$Rn, $Rm, $Imm3"),
                    [(opfrag exts.ty:$Rn, (shl exts.uxth, UXTH_operand:$Imm3))],
                    NoItinerary>;
    def w_uxtw : A64I_addsubext<sf, op, S, 0b00, 0b010,
                    outs, (ins exts.GPR:$Rn, GPR32:$Rm, UXTW_operand:$Imm3),
                    !strconcat(asmop, "$Rn, $Rm, $Imm3"),
                    [(opfrag exts.ty:$Rn, (shl exts.uxtw, UXTW_operand:$Imm3))],
                    NoItinerary>;

    def w_sxtb : A64I_addsubext<sf, op, S, 0b00, 0b100,
                    outs, (ins exts.GPR:$Rn, GPR32:$Rm, SXTB_operand:$Imm3),
                    !strconcat(asmop, "$Rn, $Rm, $Imm3"),
                    [(opfrag exts.ty:$Rn, (shl exts.sxtb, SXTB_operand:$Imm3))],
                    NoItinerary>;
    def w_sxth : A64I_addsubext<sf, op, S, 0b00, 0b101,
                    outs, (ins exts.GPR:$Rn, GPR32:$Rm, SXTH_operand:$Imm3),
                    !strconcat(asmop, "$Rn, $Rm, $Imm3"),
                    [(opfrag exts.ty:$Rn, (shl exts.sxth, SXTH_operand:$Imm3))],
                    NoItinerary>;
    def w_sxtw : A64I_addsubext<sf, op, S, 0b00, 0b110,
                    outs, (ins exts.GPR:$Rn, GPR32:$Rm, SXTW_operand:$Imm3),
                    !strconcat(asmop, "$Rn, $Rm, $Imm3"),
                    [(opfrag exts.ty:$Rn, (shl exts.sxtw, SXTW_operand:$Imm3))],
                    NoItinerary>;
}

// These two could be merge in with the above, but their patterns aren't really
// necessary and the naming-scheme would necessarily break:
multiclass addsub_xxtx<bit op, bit S, string asmop, SDPatternOperator opfrag,
                       dag outs> {
    def x_uxtx : A64I_addsubext<0b1, op, S, 0b00, 0b011,
                   outs,
                   (ins GPR64xsp:$Rn, GPR64:$Rm, UXTX_operand:$Imm3),
                   !strconcat(asmop, "$Rn, $Rm, $Imm3"),
                   [(opfrag i64:$Rn, (shl i64:$Rm, UXTX_operand:$Imm3))],
                   NoItinerary>;

    def x_sxtx : A64I_addsubext<0b1, op, S, 0b00, 0b111,
                   outs,
                   (ins GPR64xsp:$Rn, GPR64:$Rm, SXTX_operand:$Imm3),
                   !strconcat(asmop, "$Rn, $Rm, $Imm3"),
                   [/* No Pattern: same as uxtx */],
                   NoItinerary>;
}

multiclass addsub_wxtx<bit op, bit S, string asmop, dag outs> {
    def w_uxtx : A64I_addsubext<0b0, op, S, 0b00, 0b011,
                              outs,
                              (ins GPR32wsp:$Rn, GPR32:$Rm, UXTX_operand:$Imm3),
                              !strconcat(asmop, "$Rn, $Rm, $Imm3"),
                              [/* No pattern: probably same as uxtw */],
                              NoItinerary>;

    def w_sxtx : A64I_addsubext<0b0, op, S, 0b00, 0b111,
                              outs,
                              (ins GPR32wsp:$Rn, GPR32:$Rm, SXTX_operand:$Imm3),
                              !strconcat(asmop, "$Rn, $Rm, $Imm3"),
                              [/* No Pattern: probably same as uxtw */],
                              NoItinerary>;
}

class SetRD<RegisterClass RC, SDPatternOperator op>
 : PatFrag<(ops node:$lhs, node:$rhs), (set RC:$Rd, (op node:$lhs, node:$rhs))>;
class SetNZCV<SDPatternOperator op>
  : PatFrag<(ops node:$lhs, node:$rhs), (set NZCV, (op node:$lhs, node:$rhs))>;

defm ADDxx :addsub_exts<0b1, 0b0, 0b0, "add\t$Rd, ", SetRD<GPR64xsp, add>,
                        (outs GPR64xsp:$Rd), extends_to_i64>,
            addsub_xxtx<     0b0, 0b0, "add\t$Rd, ", SetRD<GPR64xsp, add>,
                        (outs GPR64xsp:$Rd)>;
defm ADDww :addsub_exts<0b0, 0b0, 0b0, "add\t$Rd, ", SetRD<GPR32wsp, add>,
                        (outs GPR32wsp:$Rd), extends_to_i32>,
            addsub_wxtx<     0b0, 0b0, "add\t$Rd, ",
                        (outs GPR32wsp:$Rd)>;
defm SUBxx :addsub_exts<0b1, 0b1, 0b0, "sub\t$Rd, ", SetRD<GPR64xsp, sub>,
                        (outs GPR64xsp:$Rd), extends_to_i64>,
            addsub_xxtx<     0b1, 0b0, "sub\t$Rd, ", SetRD<GPR64xsp, sub>,
                        (outs GPR64xsp:$Rd)>;
defm SUBww :addsub_exts<0b0, 0b1, 0b0, "sub\t$Rd, ", SetRD<GPR32wsp, sub>,
                        (outs GPR32wsp:$Rd), extends_to_i32>,
            addsub_wxtx<     0b1, 0b0, "sub\t$Rd, ",
                        (outs GPR32wsp:$Rd)>;

let Defs = [NZCV] in {
defm ADDSxx :addsub_exts<0b1, 0b0, 0b1, "adds\t$Rd, ", SetRD<GPR64, addc>,
                         (outs GPR64:$Rd), extends_to_i64>,
             addsub_xxtx<     0b0, 0b1, "adds\t$Rd, ", SetRD<GPR64, addc>,
                         (outs GPR64:$Rd)>;
defm ADDSww :addsub_exts<0b0, 0b0, 0b1, "adds\t$Rd, ", SetRD<GPR32, addc>,
                         (outs GPR32:$Rd), extends_to_i32>,
             addsub_wxtx<     0b0, 0b1, "adds\t$Rd, ",
                         (outs GPR32:$Rd)>;
defm SUBSxx :addsub_exts<0b1, 0b1, 0b1, "subs\t$Rd, ", SetRD<GPR64, subc>,
                         (outs GPR64:$Rd), extends_to_i64>,
             addsub_xxtx<     0b1, 0b1, "subs\t$Rd, ", SetRD<GPR64, subc>,
                         (outs GPR64:$Rd)>;
defm SUBSww :addsub_exts<0b0, 0b1, 0b1, "subs\t$Rd, ", SetRD<GPR32, subc>,
                         (outs GPR32:$Rd), extends_to_i32>,
             addsub_wxtx<     0b1, 0b1, "subs\t$Rd, ",
                         (outs GPR32:$Rd)>;


let Rd = 0b11111, isCompare = 1 in {
defm CMNx : addsub_exts<0b1, 0b0, 0b1, "cmn\t", SetNZCV<A64cmn>,
                        (outs), extends_to_i64>,
            addsub_xxtx<     0b0, 0b1, "cmn\t", SetNZCV<A64cmn>, (outs)>;
defm CMNw : addsub_exts<0b0, 0b0, 0b1, "cmn\t", SetNZCV<A64cmn>,
                        (outs), extends_to_i32>,
            addsub_wxtx<     0b0, 0b1, "cmn\t", (outs)>;
defm CMPx : addsub_exts<0b1, 0b1, 0b1, "cmp\t", SetNZCV<A64cmp>,
                        (outs), extends_to_i64>,
            addsub_xxtx<     0b1, 0b1, "cmp\t", SetNZCV<A64cmp>, (outs)>;
defm CMPw : addsub_exts<0b0, 0b1, 0b1, "cmp\t", SetNZCV<A64cmp>,
                        (outs), extends_to_i32>,
            addsub_wxtx<     0b1, 0b1, "cmp\t", (outs)>;
}
}

// Now patterns for the operation without a shift being needed. No patterns are
// created for uxtx/sxtx since they're non-uniform and it's expected that
// add/sub (shifted register) will handle those cases anyway.
multiclass addsubext_noshift_patterns<string prefix, SDPatternOperator nodeop,
                                      extend_types exts> {
    def : Pat<(nodeop exts.ty:$Rn, exts.uxtb),
              (!cast<Instruction>(prefix # "w_uxtb") $Rn, $Rm, 0)>;
    def : Pat<(nodeop exts.ty:$Rn, exts.uxth),
              (!cast<Instruction>(prefix # "w_uxth") $Rn, $Rm, 0)>;
    def : Pat<(nodeop exts.ty:$Rn, exts.uxtw),
              (!cast<Instruction>(prefix # "w_uxtw") $Rn, $Rm, 0)>;

    def : Pat<(nodeop exts.ty:$Rn, exts.sxtb),
              (!cast<Instruction>(prefix # "w_sxtb") $Rn, $Rm, 0)>;
    def : Pat<(nodeop exts.ty:$Rn, exts.sxth),
              (!cast<Instruction>(prefix # "w_sxth") $Rn, $Rm, 0)>;
    def : Pat<(nodeop exts.ty:$Rn, exts.sxtw),
              (!cast<Instruction>(prefix # "w_sxtw") $Rn, $Rm, 0)>;
}

defm : addsubext_noshift_patterns<"ADDxx", add, extends_to_i64>;
defm : addsubext_noshift_patterns<"ADDww", add, extends_to_i32>;
defm : addsubext_noshift_patterns<"SUBxx", sub, extends_to_i64>;
defm : addsubext_noshift_patterns<"SUBww", sub, extends_to_i32>;

defm : addsubext_noshift_patterns<"CMNx", A64cmn, extends_to_i64>;
defm : addsubext_noshift_patterns<"CMNw", A64cmn, extends_to_i32>;
defm : addsubext_noshift_patterns<"CMPx", A64cmp, extends_to_i64>;
defm : addsubext_noshift_patterns<"CMPw", A64cmp, extends_to_i32>;

// An extend of "lsl #imm" is valid if and only if one of Rn and Rd is
// sp/wsp. It is synonymous with uxtx/uxtw depending on the size of the
// operation. Also permitted in this case is complete omission of the argument,
// which implies "lsl #0".
multiclass lsl_aliases<string asmop, Instruction inst, RegisterClass GPR_Rd,
                       RegisterClass GPR_Rn, RegisterClass GPR_Rm> {
    def : InstAlias<!strconcat(asmop, " $Rd, $Rn, $Rm"),
                    (inst GPR_Rd:$Rd, GPR_Rn:$Rn, GPR_Rm:$Rm, 0)>;

    def : InstAlias<!strconcat(asmop, " $Rd, $Rn, $Rm, $LSL"),
                (inst GPR_Rd:$Rd, GPR_Rn:$Rn, GPR_Rm:$Rm, LSL_extoperand:$LSL)>;

}

defm : lsl_aliases<"add",  ADDxxx_uxtx,  Rxsp, GPR64xsp, GPR64>;
defm : lsl_aliases<"add",  ADDxxx_uxtx,  GPR64xsp, Rxsp, GPR64>;
defm : lsl_aliases<"add",  ADDwww_uxtw,  Rwsp, GPR32wsp, GPR32>;
defm : lsl_aliases<"add",  ADDwww_uxtw,  GPR32wsp, Rwsp, GPR32>;
defm : lsl_aliases<"sub",  SUBxxx_uxtx,  Rxsp, GPR64xsp, GPR64>;
defm : lsl_aliases<"sub",  SUBxxx_uxtx,  GPR64xsp, Rxsp, GPR64>;
defm : lsl_aliases<"sub",  SUBwww_uxtw,  Rwsp, GPR32wsp, GPR32>;
defm : lsl_aliases<"sub",  SUBwww_uxtw,  GPR32wsp, Rwsp, GPR32>;

// Rd cannot be sp for flag-setting variants so only half of the aliases are
// needed.
defm : lsl_aliases<"adds", ADDSxxx_uxtx, GPR64, Rxsp, GPR64>;
defm : lsl_aliases<"adds", ADDSwww_uxtw, GPR32, Rwsp, GPR32>;
defm : lsl_aliases<"subs", SUBSxxx_uxtx, GPR64, Rxsp, GPR64>;
defm : lsl_aliases<"subs", SUBSwww_uxtw, GPR32, Rwsp, GPR32>;

// CMP unfortunately has to be different because the instruction doesn't have a
// dest register.
multiclass cmp_lsl_aliases<string asmop, Instruction inst,
                       RegisterClass GPR_Rn, RegisterClass GPR_Rm> {
    def : InstAlias<!strconcat(asmop, " $Rn, $Rm"),
                    (inst GPR_Rn:$Rn, GPR_Rm:$Rm, 0)>;

    def : InstAlias<!strconcat(asmop, " $Rn, $Rm, $LSL"),
                    (inst GPR_Rn:$Rn, GPR_Rm:$Rm, LSL_extoperand:$LSL)>;
}

defm : cmp_lsl_aliases<"cmp", CMPxx_uxtx, Rxsp, GPR64>;
defm : cmp_lsl_aliases<"cmp", CMPww_uxtw, Rwsp, GPR32>;
defm : cmp_lsl_aliases<"cmn", CMNxx_uxtx, Rxsp, GPR64>;
defm : cmp_lsl_aliases<"cmn", CMNww_uxtw, Rwsp, GPR32>;

//===----------------------------------------------------------------------===//
// Add-subtract (immediate) instructions
//===----------------------------------------------------------------------===//
// Contains: ADD, ADDS, SUB, SUBS + aliases CMN, CMP, MOV

// These instructions accept a 12-bit unsigned immediate, optionally shifted
// left by 12 bits. Official assembly format specifies a 12 bit immediate with
// one of "", "LSL #0", "LSL #12" supplementary operands.

// There are surprisingly few ways to make this work with TableGen, so this
// implementation has separate instructions for the "LSL #0" and "LSL #12"
// variants.

// If the MCInst retained a single combined immediate (which could be 0x123000,
// for example) then both components (imm & shift) would have to be delegated to
// a single assembly operand. This would entail a separate operand parser
// (because the LSL would have to live in the same AArch64Operand as the
// immediate to be accessible); assembly parsing is rather complex and
// error-prone C++ code.
//
// By splitting the immediate, we can delegate handling this optional operand to
// an InstAlias. Supporting functions to generate the correct MCInst are still
// required, but these are essentially trivial and parsing can remain generic.
//
// Rejected plans with rationale:
// ------------------------------
//
// In an ideal world you'de have two first class immediate operands (in
// InOperandList, specifying imm12 and shift). Unfortunately this is not
// selectable by any means I could discover.
//
// An Instruction with two MCOperands hidden behind a single entry in
// InOperandList (expanded by ComplexPatterns and MIOperandInfo) was functional,
// but required more C++ code to handle encoding/decoding. Parsing (the intended
// main beneficiary) ended up equally complex because of the optional nature of
// "LSL #0".
//
// Attempting to circumvent the need for a custom OperandParser above by giving
// InstAliases without the "lsl #0" failed. add/sub could be accommodated but
// the cmp/cmn aliases didn't use the MIOperandInfo to determine how operands
// should be parsed: there was no way to accommodate an "lsl #12".

let ParserMethod = "ParseImmWithLSLOperand",
    RenderMethod = "addImmWithLSLOperands" in {
  // Derived PredicateMethod fields are different for each
  def addsubimm_lsl0_asmoperand : AsmOperandClass {
    let Name = "AddSubImmLSL0";
    // If an error is reported against this operand, instruction could also be a
    // register variant.
    let DiagnosticType = "AddSubSecondSource";
  }

  def addsubimm_lsl12_asmoperand : AsmOperandClass {
    let Name = "AddSubImmLSL12";
    let DiagnosticType = "AddSubSecondSource";
  }
}

def shr_12_XFORM : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getSExtValue() >> 12, MVT::i32);
}]>;

def shr_12_neg_XFORM : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant((-N->getSExtValue()) >> 12, MVT::i32);
}]>;

def neg_XFORM : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(-N->getSExtValue(), MVT::i32);
}]>;


multiclass addsub_imm_operands<ValueType ty> {
 let PrintMethod = "printAddSubImmLSL0Operand",
      EncoderMethod = "getAddSubImmOpValue",
      ParserMatchClass = addsubimm_lsl0_asmoperand in {
    def _posimm_lsl0 : Operand<ty>,
        ImmLeaf<ty, [{ return Imm >= 0 && (Imm & ~0xfff) == 0; }]>;
    def _negimm_lsl0 : Operand<ty>,
        ImmLeaf<ty, [{ return Imm < 0 && (-Imm & ~0xfff) == 0; }],
                neg_XFORM>;
  }

  let PrintMethod = "printAddSubImmLSL12Operand",
      EncoderMethod = "getAddSubImmOpValue",
      ParserMatchClass = addsubimm_lsl12_asmoperand in {
    def _posimm_lsl12 : Operand<ty>,
        ImmLeaf<ty, [{ return Imm >= 0 && (Imm & ~0xfff000) == 0; }],
                shr_12_XFORM>;

    def _negimm_lsl12 : Operand<ty>,
        ImmLeaf<ty, [{ return Imm < 0 && (-Imm & ~0xfff000) == 0; }],
                shr_12_neg_XFORM>;
  }
}

// The add operands don't need any transformation
defm addsubimm_operand_i32 : addsub_imm_operands<i32>;
defm addsubimm_operand_i64 : addsub_imm_operands<i64>;

multiclass addsubimm_varieties<string prefix, bit sf, bit op, bits<2> shift,
                               string asmop, string cmpasmop,
                               Operand imm_operand, Operand cmp_imm_operand,
                               RegisterClass GPR, RegisterClass GPRsp,
                               AArch64Reg ZR, ValueType Ty> {
    // All registers for non-S variants allow SP
  def _s : A64I_addsubimm<sf, op, 0b0, shift,
                         (outs GPRsp:$Rd),
                         (ins GPRsp:$Rn, imm_operand:$Imm12),
                         !strconcat(asmop, "\t$Rd, $Rn, $Imm12"),
                         [(set Ty:$Rd, (add Ty:$Rn, imm_operand:$Imm12))],
                         NoItinerary>;


  // S variants can read SP but would write to ZR
  def _S : A64I_addsubimm<sf, op, 0b1, shift,
                         (outs GPR:$Rd),
                         (ins GPRsp:$Rn, imm_operand:$Imm12),
                         !strconcat(asmop, "s\t$Rd, $Rn, $Imm12"),
                         [(set Ty:$Rd, (addc Ty:$Rn, imm_operand:$Imm12))],
                         NoItinerary> {
    let Defs = [NZCV];
  }

  // Note that the pattern here for ADDS is subtle. Canonically CMP
  // a, b becomes SUBS a, b. If b < 0 then this is equivalent to
  // ADDS a, (-b). This is not true in general.
  def _cmp : A64I_addsubimm<sf, op, 0b1, shift,
                            (outs), (ins GPRsp:$Rn, imm_operand:$Imm12),
                            !strconcat(cmpasmop, " $Rn, $Imm12"),
                            [(set NZCV,
                                  (A64cmp Ty:$Rn, cmp_imm_operand:$Imm12))],
                            NoItinerary> {
    let Rd = 0b11111;
    let Defs = [NZCV];
    let isCompare = 1;
  }
}


multiclass addsubimm_shifts<string prefix, bit sf, bit op,
           string asmop, string cmpasmop, string operand, string cmpoperand,
           RegisterClass GPR, RegisterClass GPRsp, AArch64Reg ZR,
           ValueType Ty> {
  defm _lsl0 : addsubimm_varieties<prefix # "_lsl0", sf, op, 0b00,
                                   asmop, cmpasmop,
                                   !cast<Operand>(operand # "_lsl0"),
                                   !cast<Operand>(cmpoperand # "_lsl0"),
                                   GPR, GPRsp, ZR, Ty>;

  defm _lsl12 : addsubimm_varieties<prefix # "_lsl12", sf, op, 0b01,
                                    asmop, cmpasmop,
                                    !cast<Operand>(operand # "_lsl12"),
                                    !cast<Operand>(cmpoperand # "_lsl12"),
                                    GPR, GPRsp, ZR, Ty>;
}

defm ADDwwi : addsubimm_shifts<"ADDwi", 0b0, 0b0, "add", "cmn",
                              "addsubimm_operand_i32_posimm",
                              "addsubimm_operand_i32_negimm",
                              GPR32, GPR32wsp, WZR, i32>;
defm ADDxxi : addsubimm_shifts<"ADDxi", 0b1, 0b0, "add", "cmn",
                              "addsubimm_operand_i64_posimm",
                              "addsubimm_operand_i64_negimm",
                              GPR64, GPR64xsp, XZR, i64>;
defm SUBwwi : addsubimm_shifts<"SUBwi", 0b0, 0b1, "sub", "cmp",
                              "addsubimm_operand_i32_negimm",
                              "addsubimm_operand_i32_posimm",
                              GPR32, GPR32wsp, WZR, i32>;
defm SUBxxi : addsubimm_shifts<"SUBxi", 0b1, 0b1, "sub", "cmp",
                              "addsubimm_operand_i64_negimm",
                              "addsubimm_operand_i64_posimm",
                              GPR64, GPR64xsp, XZR, i64>;

multiclass MOVsp<RegisterClass GPRsp, RegisterClass SP, Instruction addop> {
  def _fromsp : InstAlias<"mov $Rd, $Rn",
                          (addop GPRsp:$Rd, SP:$Rn, 0),
                          0b1>;

  def _tosp : InstAlias<"mov $Rd, $Rn",
                        (addop SP:$Rd, GPRsp:$Rn, 0),
                        0b1>;
}

// Recall Rxsp is a RegisterClass containing *just* xsp.
defm MOVxx : MOVsp<GPR64xsp, Rxsp, ADDxxi_lsl0_s>;
defm MOVww : MOVsp<GPR32wsp, Rwsp, ADDwwi_lsl0_s>;

//===----------------------------------------------------------------------===//
// Add-subtract (shifted register) instructions
//===----------------------------------------------------------------------===//
// Contains: ADD, ADDS, SUB, SUBS + aliases CMN, CMP, NEG, NEGS

//===-------------------------------
// 1. The "shifed register" operands. Shared with logical insts.
//===-------------------------------

multiclass shift_operands<string prefix, string form> {
  def _asmoperand_i32 : AsmOperandClass {
    let Name = "Shift" # form # "i32";
    let RenderMethod = "addShiftOperands";
    let PredicateMethod = "isShift<A64SE::" # form # ", false>";
    let DiagnosticType = "AddSubRegShift32";
  }

  // Note that the operand type is intentionally i64 because the DAGCombiner
  // puts these into a canonical form.
  def _i32 : Operand<i64>, ImmLeaf<i64, [{ return Imm >= 0 && Imm <= 31; }]> {
    let ParserMatchClass
          = !cast<AsmOperandClass>(prefix # "_asmoperand_i32");
    let PrintMethod = "printShiftOperand<A64SE::" # form # ">";
    let DecoderMethod = "Decode32BitShiftOperand";
  }

  def _asmoperand_i64 : AsmOperandClass {
      let Name = "Shift" # form # "i64";
      let RenderMethod = "addShiftOperands";
      let PredicateMethod = "isShift<A64SE::" # form # ", true>";
      let DiagnosticType = "AddSubRegShift64";
  }

  def _i64 : Operand<i64>, ImmLeaf<i64, [{ return Imm >= 0 && Imm <= 63; }]> {
    let ParserMatchClass
          = !cast<AsmOperandClass>(prefix # "_asmoperand_i64");
    let PrintMethod = "printShiftOperand<A64SE::" # form # ">";
  }
}

defm lsl_operand : shift_operands<"lsl_operand", "LSL">;
defm lsr_operand : shift_operands<"lsr_operand", "LSR">;
defm asr_operand : shift_operands<"asr_operand", "ASR">;

// Not used for add/sub, but defined here for completeness. The "logical
// (shifted register)" instructions *do* have an ROR variant.
defm ror_operand : shift_operands<"ror_operand", "ROR">;

//===-------------------------------
// 2. The basic 3.5-operand ADD/SUB/ADDS/SUBS instructions.
//===-------------------------------

// N.b. the commutable parameter is just !N. It will be first against the wall
// when the revolution comes.
multiclass addsub_shifts<string prefix, bit sf, bit op, bit s, bit commutable,
                         string asmop, SDPatternOperator opfrag, ValueType ty,
                         RegisterClass GPR, list<Register> defs> {
  let isCommutable = commutable, Defs = defs in {
  def _lsl : A64I_addsubshift<sf, op, s, 0b00,
                       (outs GPR:$Rd),
                       (ins GPR:$Rn, GPR:$Rm,
                            !cast<Operand>("lsl_operand_" # ty):$Imm6),
                       !strconcat(asmop, "\t$Rd, $Rn, $Rm, $Imm6"),
                       [(set GPR:$Rd, (opfrag ty:$Rn, (shl ty:$Rm,
                            !cast<Operand>("lsl_operand_" # ty):$Imm6))
                       )],
                       NoItinerary>;

  def _lsr : A64I_addsubshift<sf, op, s, 0b01,
                       (outs GPR:$Rd),
                       (ins GPR:$Rn, GPR:$Rm,
                            !cast<Operand>("lsr_operand_" # ty):$Imm6),
                       !strconcat(asmop, "\t$Rd, $Rn, $Rm, $Imm6"),
                       [(set ty:$Rd, (opfrag ty:$Rn, (srl ty:$Rm,
                            !cast<Operand>("lsr_operand_" # ty):$Imm6))
                       )],
                       NoItinerary>;

  def _asr : A64I_addsubshift<sf, op, s, 0b10,
                       (outs GPR:$Rd),
                       (ins GPR:$Rn, GPR:$Rm,
                            !cast<Operand>("asr_operand_" # ty):$Imm6),
                       !strconcat(asmop, "\t$Rd, $Rn, $Rm, $Imm6"),
                       [(set ty:$Rd, (opfrag ty:$Rn, (sra ty:$Rm,
                            !cast<Operand>("asr_operand_" # ty):$Imm6))
                       )],
                       NoItinerary>;
  }

  def _noshift
      : InstAlias<!strconcat(asmop, " $Rd, $Rn, $Rm"),
                 (!cast<Instruction>(prefix # "_lsl") GPR:$Rd, GPR:$Rn,
                                                      GPR:$Rm, 0)>;

  def : Pat<(opfrag ty:$Rn, ty:$Rm),
            (!cast<Instruction>(prefix # "_lsl") $Rn, $Rm, 0)>;
}

multiclass addsub_sizes<string prefix, bit op, bit s, bit commutable,
                         string asmop, SDPatternOperator opfrag,
                         list<Register> defs> {
  defm xxx : addsub_shifts<prefix # "xxx", 0b1, op, s,
                           commutable, asmop, opfrag, i64, GPR64, defs>;
  defm www : addsub_shifts<prefix # "www", 0b0, op, s,
                           commutable, asmop, opfrag, i32, GPR32, defs>;
}


defm ADD : addsub_sizes<"ADD", 0b0, 0b0, 0b1, "add", add, []>;
defm SUB : addsub_sizes<"SUB", 0b1, 0b0, 0b0, "sub", sub, []>;

defm ADDS : addsub_sizes<"ADDS", 0b0, 0b1, 0b1, "adds", addc, [NZCV]>;
defm SUBS : addsub_sizes<"SUBS", 0b1, 0b1, 0b0, "subs", subc, [NZCV]>;

//===-------------------------------
// 1. The NEG/NEGS aliases
//===-------------------------------

multiclass neg_alias<Instruction INST, RegisterClass GPR, Register ZR,
                     ValueType ty, Operand shift_operand, SDNode shiftop> {
   def : InstAlias<"neg $Rd, $Rm, $Imm6",
                   (INST GPR:$Rd, ZR, GPR:$Rm, shift_operand:$Imm6)>;

   def : Pat<(sub 0, (shiftop ty:$Rm, shift_operand:$Imm6)),
             (INST ZR, $Rm, shift_operand:$Imm6)>;
}

defm : neg_alias<SUBwww_lsl, GPR32, WZR, i32, lsl_operand_i32, shl>;
defm : neg_alias<SUBwww_lsr, GPR32, WZR, i32, lsr_operand_i32, srl>;
defm : neg_alias<SUBwww_asr, GPR32, WZR, i32, asr_operand_i32, sra>;
def : InstAlias<"neg $Rd, $Rm", (SUBwww_lsl GPR32:$Rd, WZR, GPR32:$Rm, 0)>;
def : Pat<(sub 0, i32:$Rm), (SUBwww_lsl WZR, $Rm, 0)>;

defm : neg_alias<SUBxxx_lsl, GPR64, XZR, i64, lsl_operand_i64, shl>;
defm : neg_alias<SUBxxx_lsr, GPR64, XZR, i64, lsr_operand_i64, srl>;
defm : neg_alias<SUBxxx_asr, GPR64, XZR, i64, asr_operand_i64, sra>;
def : InstAlias<"neg $Rd, $Rm", (SUBxxx_lsl GPR64:$Rd, XZR, GPR64:$Rm, 0)>;
def : Pat<(sub 0, i64:$Rm), (SUBxxx_lsl XZR, $Rm, 0)>;

// NEGS doesn't get any patterns yet: defining multiple outputs means C++ has to
// be involved.
class negs_alias<Instruction INST, RegisterClass GPR,
                 Register ZR, Operand shift_operand, SDNode shiftop>
  : InstAlias<"negs $Rd, $Rm, $Imm6",
              (INST GPR:$Rd, ZR, GPR:$Rm, shift_operand:$Imm6)>;

def : negs_alias<SUBSwww_lsl, GPR32, WZR, lsl_operand_i32, shl>;
def : negs_alias<SUBSwww_lsr, GPR32, WZR, lsr_operand_i32, srl>;
def : negs_alias<SUBSwww_asr, GPR32, WZR, asr_operand_i32, sra>;
def : InstAlias<"negs $Rd, $Rm", (SUBSwww_lsl GPR32:$Rd, WZR, GPR32:$Rm, 0)>;

def : negs_alias<SUBSxxx_lsl, GPR64, XZR, lsl_operand_i64, shl>;
def : negs_alias<SUBSxxx_lsr, GPR64, XZR, lsr_operand_i64, srl>;
def : negs_alias<SUBSxxx_asr, GPR64, XZR, asr_operand_i64, sra>;
def : InstAlias<"negs $Rd, $Rm", (SUBSxxx_lsl GPR64:$Rd, XZR, GPR64:$Rm, 0)>;

//===-------------------------------
// 1. The CMP/CMN aliases
//===-------------------------------

multiclass cmp_shifts<string prefix, bit sf, bit op, bit commutable,
                      string asmop, SDPatternOperator opfrag, ValueType ty,
                      RegisterClass GPR> {
  let isCommutable = commutable, Rd = 0b11111, Defs = [NZCV] in {
  def _lsl : A64I_addsubshift<sf, op, 0b1, 0b00,
                       (outs),
                       (ins GPR:$Rn, GPR:$Rm,
                            !cast<Operand>("lsl_operand_" # ty):$Imm6),
                       !strconcat(asmop, "\t$Rn, $Rm, $Imm6"),
                       [(set NZCV, (opfrag ty:$Rn, (shl ty:$Rm,
                            !cast<Operand>("lsl_operand_" # ty):$Imm6))
                       )],
                       NoItinerary>;

  def _lsr : A64I_addsubshift<sf, op, 0b1, 0b01,
                       (outs),
                       (ins GPR:$Rn, GPR:$Rm,
                            !cast<Operand>("lsr_operand_" # ty):$Imm6),
                       !strconcat(asmop, "\t$Rn, $Rm, $Imm6"),
                       [(set NZCV, (opfrag ty:$Rn, (srl ty:$Rm,
                            !cast<Operand>("lsr_operand_" # ty):$Imm6))
                       )],
                       NoItinerary>;

  def _asr : A64I_addsubshift<sf, op, 0b1, 0b10,
                       (outs),
                       (ins GPR:$Rn, GPR:$Rm,
                            !cast<Operand>("asr_operand_" # ty):$Imm6),
                       !strconcat(asmop, "\t$Rn, $Rm, $Imm6"),
                       [(set NZCV, (opfrag ty:$Rn, (sra ty:$Rm,
                            !cast<Operand>("asr_operand_" # ty):$Imm6))
                       )],
                       NoItinerary>;
  }

  def _noshift
      : InstAlias<!strconcat(asmop, " $Rn, $Rm"),
                 (!cast<Instruction>(prefix # "_lsl") GPR:$Rn, GPR:$Rm, 0)>;

  def : Pat<(opfrag ty:$Rn, ty:$Rm),
            (!cast<Instruction>(prefix # "_lsl") $Rn, $Rm, 0)>;
}

defm CMPww : cmp_shifts<"CMPww", 0b0, 0b1, 0b0, "cmp", A64cmp, i32, GPR32>;
defm CMPxx : cmp_shifts<"CMPxx", 0b1, 0b1, 0b0, "cmp", A64cmp, i64, GPR64>;

defm CMNww : cmp_shifts<"CMNww", 0b0, 0b0, 0b1, "cmn", A64cmn, i32, GPR32>;
defm CMNxx : cmp_shifts<"CMNxx", 0b1, 0b0, 0b1, "cmn", A64cmn, i64, GPR64>;

//===----------------------------------------------------------------------===//
// Add-subtract (with carry) instructions
//===----------------------------------------------------------------------===//
// Contains: ADC, ADCS, SBC, SBCS + aliases NGC, NGCS

multiclass A64I_addsubcarrySizes<bit op, bit s, string asmop> {
  let Uses = [NZCV] in {
    def www : A64I_addsubcarry<0b0, op, s, 0b000000,
                               (outs GPR32:$Rd), (ins GPR32:$Rn, GPR32:$Rm),
                               !strconcat(asmop, "\t$Rd, $Rn, $Rm"),
                               [], NoItinerary>;

    def xxx : A64I_addsubcarry<0b1, op, s, 0b000000,
                               (outs GPR64:$Rd), (ins GPR64:$Rn, GPR64:$Rm),
                               !strconcat(asmop, "\t$Rd, $Rn, $Rm"),
                               [], NoItinerary>;
  }
}

let isCommutable = 1 in {
  defm ADC : A64I_addsubcarrySizes<0b0, 0b0, "adc">;
}

defm SBC : A64I_addsubcarrySizes<0b1, 0b0, "sbc">;

let Defs = [NZCV] in {
  let isCommutable = 1 in {
    defm ADCS : A64I_addsubcarrySizes<0b0, 0b1, "adcs">;
  }

  defm SBCS : A64I_addsubcarrySizes<0b1, 0b1, "sbcs">;
}

def : InstAlias<"ngc $Rd, $Rm", (SBCwww GPR32:$Rd, WZR, GPR32:$Rm)>;
def : InstAlias<"ngc $Rd, $Rm", (SBCxxx GPR64:$Rd, XZR, GPR64:$Rm)>;
def : InstAlias<"ngcs $Rd, $Rm", (SBCSwww GPR32:$Rd, WZR, GPR32:$Rm)>;
def : InstAlias<"ngcs $Rd, $Rm", (SBCSxxx GPR64:$Rd, XZR, GPR64:$Rm)>;

// Note that adde and sube can form a chain longer than two (e.g. for 256-bit
// addition). So the flag-setting instructions are appropriate.
def : Pat<(adde i32:$Rn, i32:$Rm), (ADCSwww $Rn, $Rm)>;
def : Pat<(adde i64:$Rn, i64:$Rm), (ADCSxxx $Rn, $Rm)>;
def : Pat<(sube i32:$Rn, i32:$Rm), (SBCSwww $Rn, $Rm)>;
def : Pat<(sube i64:$Rn, i64:$Rm), (SBCSxxx $Rn, $Rm)>;

//===----------------------------------------------------------------------===//
// Bitfield
//===----------------------------------------------------------------------===//
// Contains: SBFM, BFM, UBFM, [SU]XT[BHW], ASR, LSR, LSL, SBFI[ZX], BFI, BFXIL,
//     UBFIZ, UBFX

// Because of the rather complicated nearly-overlapping aliases, the decoding of
// this range of instructions is handled manually. The architectural
// instructions are BFM, SBFM and UBFM but a disassembler should never produce
// these.
//
// In the end, the best option was to use BFM instructions for decoding under
// almost all circumstances, but to create aliasing *Instructions* for each of
// the canonical forms and specify a completely custom decoder which would
// substitute the correct MCInst as needed.
//
// This also simplifies instruction selection, parsing etc because the MCInsts
// have a shape that's closer to their use in code.

//===-------------------------------
// 1. The architectural BFM instructions
//===-------------------------------

def uimm5_asmoperand : AsmOperandClass {
  let Name = "UImm5";
  let PredicateMethod = "isUImm<5>";
  let RenderMethod = "addImmOperands";
  let DiagnosticType = "UImm5";
}

def uimm6_asmoperand : AsmOperandClass {
  let Name = "UImm6";
  let PredicateMethod = "isUImm<6>";
  let RenderMethod = "addImmOperands";
  let DiagnosticType = "UImm6";
}

def bitfield32_imm : Operand<i64>,
                     ImmLeaf<i64, [{ return Imm >= 0 && Imm < 32; }]> {
  let ParserMatchClass = uimm5_asmoperand;

  let DecoderMethod = "DecodeBitfield32ImmOperand";
}


def bitfield64_imm : Operand<i64>,
                     ImmLeaf<i64, [{ return Imm >= 0 && Imm < 64; }]> {
  let ParserMatchClass = uimm6_asmoperand;

  // Default decoder works in 64-bit case: the 6-bit field can take any value.
}

multiclass A64I_bitfieldSizes<bits<2> opc, string asmop> {
  def wwii : A64I_bitfield<0b0, opc, 0b0, (outs GPR32:$Rd),
                    (ins GPR32:$Rn, bitfield32_imm:$ImmR, bitfield32_imm:$ImmS),
                    !strconcat(asmop, "\t$Rd, $Rn, $ImmR, $ImmS"),
                    [], NoItinerary> {
    let DecoderMethod = "DecodeBitfieldInstruction";
  }

  def xxii : A64I_bitfield<0b1, opc, 0b1, (outs GPR64:$Rd),
                    (ins GPR64:$Rn, bitfield64_imm:$ImmR, bitfield64_imm:$ImmS),
                    !strconcat(asmop, "\t$Rd, $Rn, $ImmR, $ImmS"),
                    [], NoItinerary> {
    let DecoderMethod = "DecodeBitfieldInstruction";
  }
}

defm SBFM : A64I_bitfieldSizes<0b00, "sbfm">;
defm UBFM : A64I_bitfieldSizes<0b10, "ubfm">;

// BFM instructions modify the destination register rather than defining it
// completely.
def BFMwwii :
  A64I_bitfield<0b0, 0b01, 0b0, (outs GPR32:$Rd),
        (ins GPR32:$src, GPR32:$Rn, bitfield32_imm:$ImmR, bitfield32_imm:$ImmS),
        "bfm\t$Rd, $Rn, $ImmR, $ImmS", [], NoItinerary> {
  let DecoderMethod = "DecodeBitfieldInstruction";
  let Constraints = "$src = $Rd";
}

def BFMxxii :
  A64I_bitfield<0b1, 0b01, 0b1, (outs GPR64:$Rd),
        (ins GPR64:$src, GPR64:$Rn, bitfield64_imm:$ImmR, bitfield64_imm:$ImmS),
        "bfm\t$Rd, $Rn, $ImmR, $ImmS", [], NoItinerary> {
  let DecoderMethod = "DecodeBitfieldInstruction";
  let Constraints = "$src = $Rd";
}


//===-------------------------------
// 2. Extend aliases to 64-bit dest
//===-------------------------------

// Unfortunately the extensions that end up as 64-bits cannot be handled by an
// instruction alias: their syntax is (for example) "SXTB x0, w0", which needs
// to be mapped to "SBFM x0, x0, #0, 7" (changing the class of Rn). InstAlias is
// not capable of such a map as far as I'm aware

// Note that these instructions are strictly more specific than the
// BFM ones (in ImmR) so they can handle their own decoding.
class A64I_bf_ext<bit sf, bits<2> opc, RegisterClass GPRDest, ValueType dty,
                    string asmop, bits<6> imms, dag pattern>
  : A64I_bitfield<sf, opc, sf,
                  (outs GPRDest:$Rd), (ins GPR32:$Rn),
                  !strconcat(asmop, "\t$Rd, $Rn"),
                  [(set dty:$Rd, pattern)], NoItinerary> {
  let ImmR = 0b000000;
  let ImmS = imms;
}

// Signed extensions
def SXTBxw : A64I_bf_ext<0b1, 0b00, GPR64, i64, "sxtb", 7,
                         (sext_inreg (anyext i32:$Rn), i8)>;
def SXTBww : A64I_bf_ext<0b0, 0b00, GPR32, i32, "sxtb", 7,
                         (sext_inreg i32:$Rn, i8)>;
def SXTHxw : A64I_bf_ext<0b1, 0b00, GPR64, i64, "sxth", 15,
                         (sext_inreg (anyext i32:$Rn), i16)>;
def SXTHww : A64I_bf_ext<0b0, 0b00, GPR32, i32, "sxth", 15,
                         (sext_inreg i32:$Rn, i16)>;
def SXTWxw : A64I_bf_ext<0b1, 0b00, GPR64, i64, "sxtw", 31, (sext i32:$Rn)>;

// Unsigned extensions
def UXTBww : A64I_bf_ext<0b0, 0b10, GPR32, i32, "uxtb", 7,
                         (and i32:$Rn, 255)>;
def UXTHww : A64I_bf_ext<0b0, 0b10, GPR32, i32, "uxth", 15,
                         (and i32:$Rn, 65535)>;

// The 64-bit unsigned variants are not strictly architectural but recommended
// for consistency.
let isAsmParserOnly = 1 in {
  def UXTBxw : A64I_bf_ext<0b0, 0b10, GPR64, i64, "uxtb", 7,
                           (and (anyext i32:$Rn), 255)>;
  def UXTHxw : A64I_bf_ext<0b0, 0b10, GPR64, i64, "uxth", 15,
                           (and (anyext i32:$Rn), 65535)>;
}

// Extra patterns for when the source register is actually 64-bits
// too. There's no architectural difference here, it's just LLVM
// shinanigans. There's no need for equivalent zero-extension patterns
// because they'll already be caught by logical (immediate) matching.
def : Pat<(sext_inreg i64:$Rn, i8),
          (SXTBxw (EXTRACT_SUBREG $Rn, sub_32))>;
def : Pat<(sext_inreg i64:$Rn, i16),
          (SXTHxw (EXTRACT_SUBREG $Rn, sub_32))>;
def : Pat<(sext_inreg i64:$Rn, i32),
          (SXTWxw (EXTRACT_SUBREG $Rn, sub_32))>;


//===-------------------------------
// 3. Aliases for ASR and LSR (the simple shifts)
//===-------------------------------

// These also handle their own decoding because ImmS being set makes
// them take precedence over BFM.
multiclass A64I_shift<bits<2> opc, string asmop, SDNode opnode> {
  def wwi : A64I_bitfield<0b0, opc, 0b0,
                    (outs GPR32:$Rd), (ins GPR32:$Rn, bitfield32_imm:$ImmR),
                    !strconcat(asmop, "\t$Rd, $Rn, $ImmR"),
                    [(set i32:$Rd, (opnode i32:$Rn, bitfield32_imm:$ImmR))],
                    NoItinerary> {
    let ImmS = 31;
  }

  def xxi : A64I_bitfield<0b1, opc, 0b1,
                    (outs GPR64:$Rd), (ins GPR64:$Rn, bitfield64_imm:$ImmR),
                    !strconcat(asmop, "\t$Rd, $Rn, $ImmR"),
                    [(set i64:$Rd, (opnode i64:$Rn, bitfield64_imm:$ImmR))],
                    NoItinerary> {
    let ImmS = 63;
  }

}

defm ASR : A64I_shift<0b00, "asr", sra>;
defm LSR : A64I_shift<0b10, "lsr", srl>;

//===-------------------------------
// 4. Aliases for LSL
//===-------------------------------

// Unfortunately LSL and subsequent aliases are much more complicated. We need
// to be able to say certain output instruction fields depend in a complex
// manner on combinations of input assembly fields).
//
// MIOperandInfo *might* have been able to do it, but at the cost of
// significantly more C++ code.

// N.b. contrary to usual practice these operands store the shift rather than
// the machine bits in an MCInst. The complexity overhead of consistency
// outweighed the benefits in this case (custom asmparser, printer and selection
// vs custom encoder).
def bitfield32_lsl_imm : Operand<i64>,
                         ImmLeaf<i64, [{ return Imm >= 0 && Imm <= 31; }]> {
  let ParserMatchClass = uimm5_asmoperand;
  let EncoderMethod = "getBitfield32LSLOpValue";
}

def bitfield64_lsl_imm : Operand<i64>,
                         ImmLeaf<i64, [{ return Imm >= 0 && Imm <= 63; }]> {
  let ParserMatchClass = uimm6_asmoperand;
  let EncoderMethod = "getBitfield64LSLOpValue";
}

class A64I_bitfield_lsl<bit sf, RegisterClass GPR, ValueType ty,
                        Operand operand>
  : A64I_bitfield<sf, 0b10, sf, (outs GPR:$Rd), (ins GPR:$Rn, operand:$FullImm),
                  "lsl\t$Rd, $Rn, $FullImm",
                  [(set ty:$Rd, (shl ty:$Rn, operand:$FullImm))],
                  NoItinerary> {
  bits<12> FullImm;
  let ImmR = FullImm{5-0};
  let ImmS = FullImm{11-6};

  // No disassembler allowed because it would overlap with BFM which does the
  // actual work.
  let isAsmParserOnly = 1;
}

def LSLwwi : A64I_bitfield_lsl<0b0, GPR32, i32, bitfield32_lsl_imm>;
def LSLxxi : A64I_bitfield_lsl<0b1, GPR64, i64, bitfield64_lsl_imm>;

//===-------------------------------
// 5. Aliases for bitfield extract instructions
//===-------------------------------

def bfx32_width_asmoperand : AsmOperandClass {
  let Name = "BFX32Width";
  let PredicateMethod = "isBitfieldWidth<32>";
  let RenderMethod = "addBFXWidthOperands";
  let DiagnosticType = "Width32";
}

def bfx32_width : Operand<i64>, ImmLeaf<i64, [{ return true; }]> {
  let PrintMethod = "printBFXWidthOperand";
  let ParserMatchClass = bfx32_width_asmoperand;
}

def bfx64_width_asmoperand : AsmOperandClass {
  let Name = "BFX64Width";
  let PredicateMethod = "isBitfieldWidth<64>";
  let RenderMethod = "addBFXWidthOperands";
  let DiagnosticType = "Width64";
}

def bfx64_width : Operand<i64> {
  let PrintMethod = "printBFXWidthOperand";
  let ParserMatchClass = bfx64_width_asmoperand;
}


multiclass A64I_bitfield_extract<bits<2> opc, string asmop, SDNode op> {
  def wwii : A64I_bitfield<0b0, opc, 0b0, (outs GPR32:$Rd),
                       (ins GPR32:$Rn, bitfield32_imm:$ImmR, bfx32_width:$ImmS),
                       !strconcat(asmop, "\t$Rd, $Rn, $ImmR, $ImmS"),
                       [(set i32:$Rd, (op i32:$Rn, imm:$ImmR, imm:$ImmS))],
                       NoItinerary> {
    // As above, no disassembler allowed.
    let isAsmParserOnly = 1;
  }

  def xxii : A64I_bitfield<0b1, opc, 0b1, (outs GPR64:$Rd),
                       (ins GPR64:$Rn, bitfield64_imm:$ImmR, bfx64_width:$ImmS),
                       !strconcat(asmop, "\t$Rd, $Rn, $ImmR, $ImmS"),
                       [(set i64:$Rd, (op i64:$Rn, imm:$ImmR, imm:$ImmS))],
                       NoItinerary> {
    // As above, no disassembler allowed.
    let isAsmParserOnly = 1;
  }
}

defm SBFX :  A64I_bitfield_extract<0b00, "sbfx", A64Sbfx>;
defm UBFX :  A64I_bitfield_extract<0b10, "ubfx", A64Ubfx>;

// Again, variants based on BFM modify Rd so need it as an input too.
def BFXILwwii : A64I_bitfield<0b0, 0b01, 0b0, (outs GPR32:$Rd),
           (ins GPR32:$src, GPR32:$Rn, bitfield32_imm:$ImmR, bfx32_width:$ImmS),
           "bfxil\t$Rd, $Rn, $ImmR, $ImmS", [], NoItinerary> {
  // As above, no disassembler allowed.
  let isAsmParserOnly = 1;
  let Constraints = "$src = $Rd";
}

def BFXILxxii : A64I_bitfield<0b1, 0b01, 0b1, (outs GPR64:$Rd),
           (ins GPR64:$src, GPR64:$Rn, bitfield64_imm:$ImmR, bfx64_width:$ImmS),
           "bfxil\t$Rd, $Rn, $ImmR, $ImmS", [], NoItinerary> {
  // As above, no disassembler allowed.
  let isAsmParserOnly = 1;
  let Constraints = "$src = $Rd";
}

// SBFX instructions can do a 1-instruction sign-extension of boolean values.
def : Pat<(sext_inreg i64:$Rn, i1), (SBFXxxii $Rn, 0, 0)>;
def : Pat<(sext_inreg i32:$Rn, i1), (SBFXwwii $Rn, 0, 0)>;
def : Pat<(i64 (sext_inreg (anyext i32:$Rn), i1)),
          (SBFXxxii (SUBREG_TO_REG (i64 0), $Rn, sub_32), 0, 0)>;

// UBFX makes sense as an implementation of a 64-bit zero-extension too. Could
// use either 64-bit or 32-bit variant, but 32-bit might be more efficient.
def : Pat<(i64 (zext i32:$Rn)), (SUBREG_TO_REG (i64 0), (UBFXwwii $Rn, 0, 31),
                                         sub_32)>;

//===-------------------------------
// 6. Aliases for bitfield insert instructions
//===-------------------------------

def bfi32_lsb_asmoperand : AsmOperandClass {
  let Name = "BFI32LSB";
  let PredicateMethod = "isUImm<5>";
  let RenderMethod = "addBFILSBOperands<32>";
  let DiagnosticType = "UImm5";
}

def bfi32_lsb : Operand<i64>,
                ImmLeaf<i64, [{ return Imm >= 0 && Imm <= 31; }]> {
  let PrintMethod = "printBFILSBOperand<32>";
  let ParserMatchClass = bfi32_lsb_asmoperand;
}

def bfi64_lsb_asmoperand : AsmOperandClass {
  let Name = "BFI64LSB";
  let PredicateMethod = "isUImm<6>";
  let RenderMethod = "addBFILSBOperands<64>";
  let DiagnosticType = "UImm6";
}

def bfi64_lsb : Operand<i64>,
                ImmLeaf<i64, [{ return Imm >= 0 && Imm <= 63; }]> {
  let PrintMethod = "printBFILSBOperand<64>";
  let ParserMatchClass = bfi64_lsb_asmoperand;
}

// Width verification is performed during conversion so width operand can be
// shared between 32/64-bit cases. Still needed for the print method though
// because ImmR encodes "width - 1".
def bfi32_width_asmoperand : AsmOperandClass {
  let Name = "BFI32Width";
  let PredicateMethod = "isBitfieldWidth<32>";
  let RenderMethod = "addBFIWidthOperands";
  let DiagnosticType = "Width32";
}

def bfi32_width : Operand<i64>,
                  ImmLeaf<i64, [{ return Imm >= 1 && Imm <= 32; }]> {
  let PrintMethod = "printBFIWidthOperand";
  let ParserMatchClass = bfi32_width_asmoperand;
}

def bfi64_width_asmoperand : AsmOperandClass {
  let Name = "BFI64Width";
  let PredicateMethod = "isBitfieldWidth<64>";
  let RenderMethod = "addBFIWidthOperands";
  let DiagnosticType = "Width64";
}

def bfi64_width : Operand<i64>,
                  ImmLeaf<i64, [{ return Imm >= 1 && Imm <= 64; }]> {
  let PrintMethod = "printBFIWidthOperand";
  let ParserMatchClass = bfi64_width_asmoperand;
}

multiclass A64I_bitfield_insert<bits<2> opc, string asmop> {
  def wwii : A64I_bitfield<0b0, opc, 0b0, (outs GPR32:$Rd),
                           (ins GPR32:$Rn, bfi32_lsb:$ImmR, bfi32_width:$ImmS),
                           !strconcat(asmop, "\t$Rd, $Rn, $ImmR, $ImmS"),
                           [], NoItinerary> {
    // As above, no disassembler allowed.
    let isAsmParserOnly = 1;
  }

  def xxii : A64I_bitfield<0b1, opc, 0b1, (outs GPR64:$Rd),
                           (ins GPR64:$Rn, bfi64_lsb:$ImmR, bfi64_width:$ImmS),
                           !strconcat(asmop, "\t$Rd, $Rn, $ImmR, $ImmS"),
                           [], NoItinerary> {
    // As above, no disassembler allowed.
    let isAsmParserOnly = 1;
  }
}

defm SBFIZ :  A64I_bitfield_insert<0b00, "sbfiz">;
defm UBFIZ :  A64I_bitfield_insert<0b10, "ubfiz">;


def BFIwwii : A64I_bitfield<0b0, 0b01, 0b0, (outs GPR32:$Rd),
                (ins GPR32:$src, GPR32:$Rn, bfi32_lsb:$ImmR, bfi32_width:$ImmS),
                "bfi\t$Rd, $Rn, $ImmR, $ImmS", [], NoItinerary> {
  // As above, no disassembler allowed.
  let isAsmParserOnly = 1;
  let Constraints = "$src = $Rd";
}

def BFIxxii : A64I_bitfield<0b1, 0b01, 0b1, (outs GPR64:$Rd),
                (ins GPR64:$src, GPR64:$Rn, bfi64_lsb:$ImmR, bfi64_width:$ImmS),
                "bfi\t$Rd, $Rn, $ImmR, $ImmS", [], NoItinerary> {
  // As above, no disassembler allowed.
  let isAsmParserOnly = 1;
  let Constraints = "$src = $Rd";
}

//===----------------------------------------------------------------------===//
// Compare and branch (immediate)
//===----------------------------------------------------------------------===//
// Contains: CBZ, CBNZ

class label_asmoperand<int width, int scale> : AsmOperandClass {
  let Name = "Label" # width # "_" # scale;
  let PredicateMethod = "isLabel<" # width # "," # scale # ">";
  let RenderMethod = "addLabelOperands<" # width # ", " # scale # ">";
  let DiagnosticType = "Label";
}

def label_wid19_scal4_asmoperand : label_asmoperand<19, 4>;

// All conditional immediate branches are the same really: 19 signed bits scaled
// by the instruction-size (4).
def bcc_target : Operand<OtherVT> {
  // This label is a 19-bit offset from PC, scaled by the instruction-width: 4.
  let ParserMatchClass = label_wid19_scal4_asmoperand;
  let PrintMethod = "printLabelOperand<19, 4>";
  let EncoderMethod = "getLabelOpValue<AArch64::fixup_a64_condbr>";
  let OperandType = "OPERAND_PCREL";
}

multiclass cmpbr_sizes<bit op, string asmop, ImmLeaf SETOP> {
  let isBranch = 1, isTerminator = 1 in {
  def x : A64I_cmpbr<0b1, op,
                     (outs),
                     (ins GPR64:$Rt, bcc_target:$Label),
                     !strconcat(asmop,"\t$Rt, $Label"),
                     [(A64br_cc (A64cmp i64:$Rt, 0), SETOP, bb:$Label)],
                     NoItinerary>;

  def w : A64I_cmpbr<0b0, op,
                     (outs),
                     (ins GPR32:$Rt, bcc_target:$Label),
                     !strconcat(asmop,"\t$Rt, $Label"),
                     [(A64br_cc (A64cmp i32:$Rt, 0), SETOP, bb:$Label)],
                     NoItinerary>;
  }
}

defm CBZ  : cmpbr_sizes<0b0, "cbz",  ImmLeaf<i32, [{
  return Imm == A64CC::EQ;
}]> >;
defm CBNZ : cmpbr_sizes<0b1, "cbnz", ImmLeaf<i32, [{
  return Imm == A64CC::NE;
}]> >;

//===----------------------------------------------------------------------===//
// Conditional branch (immediate) instructions
//===----------------------------------------------------------------------===//
// Contains: B.cc

def cond_code_asmoperand : AsmOperandClass {
  let Name = "CondCode";
  let DiagnosticType = "CondCode";
}

def cond_code : Operand<i32>, ImmLeaf<i32, [{
  return Imm >= 0 && Imm <= 15;
}]> {
  let PrintMethod = "printCondCodeOperand";
  let ParserMatchClass = cond_code_asmoperand;
}

def Bcc : A64I_condbr<0b0, 0b0, (outs),
                (ins cond_code:$Cond, bcc_target:$Label),
                "b.$Cond $Label", [(A64br_cc NZCV, (i32 imm:$Cond), bb:$Label)],
                NoItinerary> {
  let Uses = [NZCV];
  let isBranch = 1;
  let isTerminator = 1;
}

//===----------------------------------------------------------------------===//
// Conditional compare (immediate) instructions
//===----------------------------------------------------------------------===//
// Contains: CCMN, CCMP

def uimm4_asmoperand : AsmOperandClass {
  let Name = "UImm4";
  let PredicateMethod = "isUImm<4>";
  let RenderMethod = "addImmOperands";
  let DiagnosticType = "UImm4";
}

def uimm4 : Operand<i32> {
  let ParserMatchClass = uimm4_asmoperand;
}

def uimm5 : Operand<i32> {
  let ParserMatchClass = uimm5_asmoperand;
}

// The only difference between this operand and the one for instructions like
// B.cc is that it's parsed manually. The other get parsed implicitly as part of
// the mnemonic handling.
def cond_code_op_asmoperand : AsmOperandClass {
  let Name = "CondCodeOp";
  let RenderMethod = "addCondCodeOperands";
  let PredicateMethod = "isCondCode";
  let ParserMethod = "ParseCondCodeOperand";
  let DiagnosticType = "CondCode";
}

def cond_code_op : Operand<i32> {
  let PrintMethod = "printCondCodeOperand";
  let ParserMatchClass = cond_code_op_asmoperand;
}

class A64I_condcmpimmImpl<bit sf, bit op, RegisterClass GPR, string asmop>
  : A64I_condcmpimm<sf, op, 0b0, 0b0, 0b1, (outs),
                (ins GPR:$Rn, uimm5:$UImm5, uimm4:$NZCVImm, cond_code_op:$Cond),
                !strconcat(asmop, "\t$Rn, $UImm5, $NZCVImm, $Cond"),
                [], NoItinerary> {
  let Defs = [NZCV];
}

def CCMNwi : A64I_condcmpimmImpl<0b0, 0b0, GPR32, "ccmn">;
def CCMNxi : A64I_condcmpimmImpl<0b1, 0b0, GPR64, "ccmn">;
def CCMPwi : A64I_condcmpimmImpl<0b0, 0b1, GPR32, "ccmp">;
def CCMPxi : A64I_condcmpimmImpl<0b1, 0b1, GPR64, "ccmp">;

//===----------------------------------------------------------------------===//
// Conditional compare (register) instructions
//===----------------------------------------------------------------------===//
// Contains: CCMN, CCMP

class A64I_condcmpregImpl<bit sf, bit op, RegisterClass GPR, string asmop>
  : A64I_condcmpreg<sf, op, 0b0, 0b0, 0b1,
                    (outs),
                    (ins GPR:$Rn, GPR:$Rm, uimm4:$NZCVImm, cond_code_op:$Cond),
                    !strconcat(asmop, "\t$Rn, $Rm, $NZCVImm, $Cond"),
                    [], NoItinerary> {
  let Defs = [NZCV];
}

def CCMNww : A64I_condcmpregImpl<0b0, 0b0, GPR32, "ccmn">;
def CCMNxx : A64I_condcmpregImpl<0b1, 0b0, GPR64, "ccmn">;
def CCMPww : A64I_condcmpregImpl<0b0, 0b1, GPR32, "ccmp">;
def CCMPxx : A64I_condcmpregImpl<0b1, 0b1, GPR64, "ccmp">;

//===----------------------------------------------------------------------===//
// Conditional select instructions
//===----------------------------------------------------------------------===//
// Contains: CSEL, CSINC, CSINV, CSNEG + aliases CSET, CSETM, CINC, CINV, CNEG

// Condition code which is encoded as the inversion (semantically rather than
// bitwise) in the instruction.
def inv_cond_code_op_asmoperand : AsmOperandClass {
  let Name = "InvCondCodeOp";
  let RenderMethod = "addInvCondCodeOperands";
  let PredicateMethod = "isCondCode";
  let ParserMethod = "ParseCondCodeOperand";
  let DiagnosticType = "CondCode";
}

def inv_cond_code_op : Operand<i32> {
  let ParserMatchClass = inv_cond_code_op_asmoperand;
}

// Having a separate operand for the selectable use-case is debatable, but gives
// consistency with cond_code.
def inv_cond_XFORM : SDNodeXForm<imm, [{
  A64CC::CondCodes CC = static_cast<A64CC::CondCodes>(N->getZExtValue());
  return CurDAG->getTargetConstant(A64InvertCondCode(CC), MVT::i32);
}]>;

def inv_cond_code
  : ImmLeaf<i32, [{ return Imm >= 0 && Imm <= 15; }], inv_cond_XFORM>;


multiclass A64I_condselSizes<bit op, bits<2> op2, string asmop,
                             SDPatternOperator select> {
  let Uses = [NZCV] in {
    def wwwc : A64I_condsel<0b0, op, 0b0, op2,
                            (outs GPR32:$Rd),
                            (ins GPR32:$Rn, GPR32:$Rm, cond_code_op:$Cond),
                            !strconcat(asmop, "\t$Rd, $Rn, $Rm, $Cond"),
                            [(set i32:$Rd, (select i32:$Rn, i32:$Rm))],
                            NoItinerary>;


    def xxxc : A64I_condsel<0b1, op, 0b0, op2,
                            (outs GPR64:$Rd),
                            (ins GPR64:$Rn, GPR64:$Rm, cond_code_op:$Cond),
                            !strconcat(asmop, "\t$Rd, $Rn, $Rm, $Cond"),
                            [(set i64:$Rd, (select i64:$Rn, i64:$Rm))],
                            NoItinerary>;
  }
}

def simple_select
  : PatFrag<(ops node:$lhs, node:$rhs),
            (A64select_cc NZCV, node:$lhs, node:$rhs, (i32 imm:$Cond))>;

class complex_select<SDPatternOperator opnode>
  : PatFrag<(ops node:$lhs, node:$rhs),
        (A64select_cc NZCV, node:$lhs, (opnode node:$rhs), (i32 imm:$Cond))>;


defm CSEL : A64I_condselSizes<0b0, 0b00, "csel", simple_select>;
defm CSINC : A64I_condselSizes<0b0, 0b01, "csinc",
                               complex_select<PatFrag<(ops node:$val),
                                                      (add node:$val, 1)>>>;
defm CSINV : A64I_condselSizes<0b1, 0b00, "csinv", complex_select<not>>;
defm CSNEG : A64I_condselSizes<0b1, 0b01, "csneg", complex_select<ineg>>;

// Now the instruction aliases, which fit nicely into LLVM's model:

def : InstAlias<"cset $Rd, $Cond",
                (CSINCwwwc GPR32:$Rd, WZR, WZR, inv_cond_code_op:$Cond)>;
def : InstAlias<"cset $Rd, $Cond",
                (CSINCxxxc GPR64:$Rd, XZR, XZR, inv_cond_code_op:$Cond)>;
def : InstAlias<"csetm $Rd, $Cond",
                (CSINVwwwc GPR32:$Rd, WZR, WZR, inv_cond_code_op:$Cond)>;
def : InstAlias<"csetm $Rd, $Cond",
                (CSINVxxxc GPR64:$Rd, XZR, XZR, inv_cond_code_op:$Cond)>;
def : InstAlias<"cinc $Rd, $Rn, $Cond",
           (CSINCwwwc GPR32:$Rd, GPR32:$Rn, GPR32:$Rn, inv_cond_code_op:$Cond)>;
def : InstAlias<"cinc $Rd, $Rn, $Cond",
           (CSINCxxxc GPR64:$Rd, GPR64:$Rn, GPR64:$Rn, inv_cond_code_op:$Cond)>;
def : InstAlias<"cinv $Rd, $Rn, $Cond",
           (CSINVwwwc GPR32:$Rd, GPR32:$Rn, GPR32:$Rn, inv_cond_code_op:$Cond)>;
def : InstAlias<"cinv $Rd, $Rn, $Cond",
           (CSINVxxxc GPR64:$Rd, GPR64:$Rn, GPR64:$Rn, inv_cond_code_op:$Cond)>;
def : InstAlias<"cneg $Rd, $Rn, $Cond",
           (CSNEGwwwc GPR32:$Rd, GPR32:$Rn, GPR32:$Rn, inv_cond_code_op:$Cond)>;
def : InstAlias<"cneg $Rd, $Rn, $Cond",
           (CSNEGxxxc GPR64:$Rd, GPR64:$Rn, GPR64:$Rn, inv_cond_code_op:$Cond)>;

// Finally some helper patterns.

// For CSET (a.k.a. zero-extension of icmp)
def : Pat<(A64select_cc NZCV, 0, 1, cond_code:$Cond),
          (CSINCwwwc WZR, WZR, cond_code:$Cond)>;
def : Pat<(A64select_cc NZCV, 1, 0, inv_cond_code:$Cond),
          (CSINCwwwc WZR, WZR, inv_cond_code:$Cond)>;

def : Pat<(A64select_cc NZCV, 0, 1, cond_code:$Cond),
          (CSINCxxxc XZR, XZR, cond_code:$Cond)>;
def : Pat<(A64select_cc NZCV, 1, 0, inv_cond_code:$Cond),
          (CSINCxxxc XZR, XZR, inv_cond_code:$Cond)>;

// For CSETM (a.k.a. sign-extension of icmp)
def : Pat<(A64select_cc NZCV, 0, -1, cond_code:$Cond),
          (CSINVwwwc WZR, WZR, cond_code:$Cond)>;
def : Pat<(A64select_cc NZCV, -1, 0, inv_cond_code:$Cond),
          (CSINVwwwc WZR, WZR, inv_cond_code:$Cond)>;

def : Pat<(A64select_cc NZCV, 0, -1, cond_code:$Cond),
          (CSINVxxxc XZR, XZR, cond_code:$Cond)>;
def : Pat<(A64select_cc NZCV, -1, 0, inv_cond_code:$Cond),
          (CSINVxxxc XZR, XZR, inv_cond_code:$Cond)>;

// CINC, CINV and CNEG get dealt with automatically, which leaves the issue of
// commutativity. The instructions are to complex for isCommutable to be used,
// so we have to create the patterns manually:

// No commutable pattern for CSEL since the commuted version is isomorphic.

// CSINC
def :Pat<(A64select_cc NZCV, (add i32:$Rm, 1), i32:$Rn, inv_cond_code:$Cond),
         (CSINCwwwc $Rn, $Rm, inv_cond_code:$Cond)>;
def :Pat<(A64select_cc NZCV, (add i64:$Rm, 1), i64:$Rn, inv_cond_code:$Cond),
         (CSINCxxxc $Rn, $Rm, inv_cond_code:$Cond)>;

// CSINV
def :Pat<(A64select_cc NZCV, (not i32:$Rm), i32:$Rn, inv_cond_code:$Cond),
         (CSINVwwwc $Rn, $Rm, inv_cond_code:$Cond)>;
def :Pat<(A64select_cc NZCV, (not i64:$Rm), i64:$Rn, inv_cond_code:$Cond),
         (CSINVxxxc $Rn, $Rm, inv_cond_code:$Cond)>;

// CSNEG
def :Pat<(A64select_cc NZCV, (ineg i32:$Rm), i32:$Rn, inv_cond_code:$Cond),
         (CSNEGwwwc $Rn, $Rm, inv_cond_code:$Cond)>;
def :Pat<(A64select_cc NZCV, (ineg i64:$Rm), i64:$Rn, inv_cond_code:$Cond),
         (CSNEGxxxc $Rn, $Rm, inv_cond_code:$Cond)>;

//===----------------------------------------------------------------------===//
// Data Processing (1 source) instructions
//===----------------------------------------------------------------------===//
// Contains: RBIT, REV16, REV, REV32, CLZ, CLS.

// We define an unary operator which always fails. We will use this to
// define unary operators that cannot be matched.

class A64I_dp_1src_impl<bit sf, bits<6> opcode, string asmop,
                   list<dag> patterns, RegisterClass GPRrc,
                   InstrItinClass itin>:
      A64I_dp_1src<sf,
                   0,
                   0b00000,
                   opcode,
                   !strconcat(asmop, "\t$Rd, $Rn"),
                   (outs GPRrc:$Rd),
                   (ins GPRrc:$Rn),
                   patterns,
                   itin>;

multiclass A64I_dp_1src <bits<6> opcode, string asmop> {
  let hasSideEffects = 0 in {
    def ww : A64I_dp_1src_impl<0b0, opcode, asmop, [], GPR32, NoItinerary>;
    def xx : A64I_dp_1src_impl<0b1, opcode, asmop, [], GPR64, NoItinerary>;
  }
}

defm RBIT  : A64I_dp_1src<0b000000, "rbit">;
defm CLS   : A64I_dp_1src<0b000101, "cls">;
defm CLZ   : A64I_dp_1src<0b000100, "clz">;

def : Pat<(ctlz i32:$Rn), (CLZww $Rn)>;
def : Pat<(ctlz i64:$Rn), (CLZxx $Rn)>;
def : Pat<(ctlz_zero_undef i32:$Rn), (CLZww $Rn)>;
def : Pat<(ctlz_zero_undef i64:$Rn), (CLZxx $Rn)>;

def : Pat<(cttz i32:$Rn), (CLZww (RBITww $Rn))>;
def : Pat<(cttz i64:$Rn), (CLZxx (RBITxx $Rn))>;
def : Pat<(cttz_zero_undef i32:$Rn), (CLZww (RBITww $Rn))>;
def : Pat<(cttz_zero_undef i64:$Rn), (CLZxx (RBITxx $Rn))>;


def REVww : A64I_dp_1src_impl<0b0, 0b000010, "rev",
                              [(set i32:$Rd, (bswap i32:$Rn))],
                              GPR32, NoItinerary>;
def REVxx : A64I_dp_1src_impl<0b1, 0b000011, "rev",
                              [(set i64:$Rd, (bswap i64:$Rn))],
                              GPR64, NoItinerary>;
def REV32xx : A64I_dp_1src_impl<0b1, 0b000010, "rev32",
                          [(set i64:$Rd, (bswap (rotr i64:$Rn, (i64 32))))],
                          GPR64, NoItinerary>;
def REV16ww : A64I_dp_1src_impl<0b0, 0b000001, "rev16",
                          [(set i32:$Rd, (bswap (rotr i32:$Rn, (i64 16))))],
                          GPR32,
                          NoItinerary>;
def REV16xx : A64I_dp_1src_impl<0b1, 0b000001, "rev16", [], GPR64, NoItinerary>;

//===----------------------------------------------------------------------===//
// Data Processing (2 sources) instructions
//===----------------------------------------------------------------------===//
// Contains: CRC32C?[BHWX], UDIV, SDIV, LSLV, LSRV, ASRV, RORV + aliases LSL,
//           LSR, ASR, ROR


class dp_2src_impl<bit sf, bits<6> opcode, string asmop, list<dag> patterns,
                   RegisterClass GPRsp,
                   InstrItinClass itin>:
      A64I_dp_2src<sf,
                   opcode,
                   0,
                   !strconcat(asmop, "\t$Rd, $Rn, $Rm"),
                   (outs GPRsp:$Rd),
                   (ins GPRsp:$Rn, GPRsp:$Rm),
                   patterns,
                   itin>;

multiclass dp_2src_crc<bit c, string asmop> {
  def B_www : dp_2src_impl<0b0, {0, 1, 0, c, 0, 0},
                           !strconcat(asmop, "b"), [], GPR32, NoItinerary>;
  def H_www : dp_2src_impl<0b0, {0, 1, 0, c, 0, 1},
                           !strconcat(asmop, "h"), [], GPR32, NoItinerary>;
  def W_www : dp_2src_impl<0b0, {0, 1, 0, c, 1, 0},
                           !strconcat(asmop, "w"), [], GPR32, NoItinerary>;
  def X_wwx : A64I_dp_2src<0b1, {0, 1, 0, c, 1, 1}, 0b0,
                           !strconcat(asmop, "x\t$Rd, $Rn, $Rm"),
                           (outs GPR32:$Rd), (ins GPR32:$Rn, GPR64:$Rm), [],
                           NoItinerary>;
}

multiclass dp_2src_zext <bits<6> opcode, string asmop, SDPatternOperator op> {
   def www : dp_2src_impl<0b0,
                         opcode,
                         asmop,
                         [(set i32:$Rd,
                               (op i32:$Rn, (i64 (zext i32:$Rm))))],
                         GPR32,
                         NoItinerary>;
   def xxx : dp_2src_impl<0b1,
                         opcode,
                         asmop,
                         [(set i64:$Rd, (op i64:$Rn, i64:$Rm))],
                         GPR64,
                         NoItinerary>;
}


multiclass dp_2src <bits<6> opcode, string asmop, SDPatternOperator op> {
    def www : dp_2src_impl<0b0,
                         opcode,
                         asmop,
                         [(set i32:$Rd, (op i32:$Rn, i32:$Rm))],
                         GPR32,
                         NoItinerary>;
   def xxx : dp_2src_impl<0b1,
                         opcode,
                         asmop,
                         [(set i64:$Rd, (op i64:$Rn, i64:$Rm))],
                         GPR64,
                         NoItinerary>;
}

// Here we define the data processing 2 source instructions.
defm CRC32  : dp_2src_crc<0b0, "crc32">;
defm CRC32C : dp_2src_crc<0b1, "crc32c">;

defm UDIV : dp_2src<0b000010, "udiv", udiv>;
defm SDIV : dp_2src<0b000011, "sdiv", sdiv>;

defm LSLV : dp_2src_zext<0b001000, "lsl", shl>;
defm LSRV : dp_2src_zext<0b001001, "lsr", srl>;
defm ASRV : dp_2src_zext<0b001010, "asr", sra>;
defm RORV : dp_2src_zext<0b001011, "ror", rotr>;

// Extra patterns for an incoming 64-bit value for a 32-bit
// operation. Since the LLVM operations are undefined (as in C) if the
// RHS is out of range, it's perfectly permissible to discard the high
// bits of the GPR64.
def : Pat<(shl i32:$Rn, i64:$Rm),
          (LSLVwww $Rn, (EXTRACT_SUBREG $Rm, sub_32))>;
def : Pat<(srl i32:$Rn, i64:$Rm),
          (LSRVwww $Rn, (EXTRACT_SUBREG $Rm, sub_32))>;
def : Pat<(sra i32:$Rn, i64:$Rm),
          (ASRVwww $Rn, (EXTRACT_SUBREG $Rm, sub_32))>;
def : Pat<(rotr i32:$Rn, i64:$Rm),
          (RORVwww $Rn, (EXTRACT_SUBREG $Rm, sub_32))>;

// Here we define the aliases for the data processing 2 source instructions.
def LSL_mnemonic : MnemonicAlias<"lslv", "lsl">;
def LSR_mnemonic : MnemonicAlias<"lsrv", "lsr">;
def ASR_menmonic : MnemonicAlias<"asrv", "asr">;
def ROR_menmonic : MnemonicAlias<"rorv", "ror">;

//===----------------------------------------------------------------------===//
// Data Processing (3 sources) instructions
//===----------------------------------------------------------------------===//
// Contains: MADD, MSUB, SMADDL, SMSUBL, SMULH, UMADDL, UMSUBL, UMULH
//    + aliases MUL, MNEG, SMULL, SMNEGL, UMULL, UMNEGL

class A64I_dp3_4operand<bit sf, bits<6> opcode, RegisterClass AccReg,
                        ValueType AccTy, RegisterClass SrcReg,
                        string asmop, dag pattern>
  : A64I_dp3<sf, opcode,
             (outs AccReg:$Rd), (ins SrcReg:$Rn, SrcReg:$Rm, AccReg:$Ra),
             !strconcat(asmop, "\t$Rd, $Rn, $Rm, $Ra"),
             [(set AccTy:$Rd, pattern)], NoItinerary> {
  RegisterClass AccGPR = AccReg;
  RegisterClass SrcGPR = SrcReg;
}

def MADDwwww : A64I_dp3_4operand<0b0, 0b000000, GPR32, i32, GPR32, "madd",
                                 (add i32:$Ra, (mul i32:$Rn, i32:$Rm))>;
def MADDxxxx : A64I_dp3_4operand<0b1, 0b000000, GPR64, i64, GPR64, "madd",
                                 (add i64:$Ra, (mul i64:$Rn, i64:$Rm))>;

def MSUBwwww : A64I_dp3_4operand<0b0, 0b000001, GPR32, i32, GPR32, "msub",
                                 (sub i32:$Ra, (mul i32:$Rn, i32:$Rm))>;
def MSUBxxxx : A64I_dp3_4operand<0b1, 0b000001, GPR64, i64, GPR64, "msub",
                                 (sub i64:$Ra, (mul i64:$Rn, i64:$Rm))>;

def SMADDLxwwx : A64I_dp3_4operand<0b1, 0b000010, GPR64, i64, GPR32, "smaddl",
                     (add i64:$Ra, (mul (i64 (sext i32:$Rn)), (sext i32:$Rm)))>;
def SMSUBLxwwx : A64I_dp3_4operand<0b1, 0b000011, GPR64, i64, GPR32, "smsubl",
                     (sub i64:$Ra, (mul (i64 (sext i32:$Rn)), (sext i32:$Rm)))>;

def UMADDLxwwx : A64I_dp3_4operand<0b1, 0b001010, GPR64, i64, GPR32, "umaddl",
                     (add i64:$Ra, (mul (i64 (zext i32:$Rn)), (zext i32:$Rm)))>;
def UMSUBLxwwx : A64I_dp3_4operand<0b1, 0b001011, GPR64, i64, GPR32, "umsubl",
                     (sub i64:$Ra, (mul (i64 (zext i32:$Rn)), (zext i32:$Rm)))>;

let isCommutable = 1, PostEncoderMethod = "fixMulHigh" in {
  def UMULHxxx : A64I_dp3<0b1, 0b001100, (outs GPR64:$Rd),
                          (ins GPR64:$Rn, GPR64:$Rm),
                          "umulh\t$Rd, $Rn, $Rm",
                          [(set i64:$Rd, (mulhu i64:$Rn, i64:$Rm))],
                          NoItinerary>;

  def SMULHxxx : A64I_dp3<0b1, 0b000100, (outs GPR64:$Rd),
                          (ins GPR64:$Rn, GPR64:$Rm),
                          "smulh\t$Rd, $Rn, $Rm",
                          [(set i64:$Rd, (mulhs i64:$Rn, i64:$Rm))],
                          NoItinerary>;
}

multiclass A64I_dp3_3operand<string asmop, A64I_dp3_4operand INST,
                             Register ZR, dag pattern> {
  def : InstAlias<asmop # " $Rd, $Rn, $Rm",
                  (INST INST.AccGPR:$Rd, INST.SrcGPR:$Rn, INST.SrcGPR:$Rm, ZR)>;

  def : Pat<pattern, (INST $Rn, $Rm, ZR)>;
}

defm : A64I_dp3_3operand<"mul", MADDwwww, WZR, (mul i32:$Rn, i32:$Rm)>;
defm : A64I_dp3_3operand<"mul", MADDxxxx, XZR, (mul i64:$Rn, i64:$Rm)>;

defm : A64I_dp3_3operand<"mneg", MSUBwwww, WZR,
                         (sub 0, (mul i32:$Rn, i32:$Rm))>;
defm : A64I_dp3_3operand<"mneg", MSUBxxxx, XZR,
                         (sub 0, (mul i64:$Rn, i64:$Rm))>;

defm : A64I_dp3_3operand<"smull", SMADDLxwwx, XZR,
                         (mul (i64 (sext i32:$Rn)), (sext i32:$Rm))>;
defm : A64I_dp3_3operand<"smnegl", SMSUBLxwwx, XZR,
                       (sub 0, (mul (i64 (sext i32:$Rn)), (sext i32:$Rm)))>;

defm : A64I_dp3_3operand<"umull", UMADDLxwwx, XZR,
                         (mul (i64 (zext i32:$Rn)), (zext i32:$Rm))>;
defm : A64I_dp3_3operand<"umnegl", UMSUBLxwwx, XZR,
                       (sub 0, (mul (i64 (zext i32:$Rn)), (zext i32:$Rm)))>;


//===----------------------------------------------------------------------===//
// Exception generation
//===----------------------------------------------------------------------===//
// Contains: SVC, HVC, SMC, BRK, HLT, DCPS1, DCPS2, DCPS3

def uimm16_asmoperand : AsmOperandClass {
  let Name = "UImm16";
  let PredicateMethod = "isUImm<16>";
  let RenderMethod = "addImmOperands";
  let DiagnosticType = "UImm16";
}

def uimm16 : Operand<i32> {
  let ParserMatchClass = uimm16_asmoperand;
}

class A64I_exceptImpl<bits<3> opc, bits<2> ll, string asmop>
  : A64I_exception<opc, 0b000, ll, (outs), (ins uimm16:$UImm16),
                   !strconcat(asmop, "\t$UImm16"), [], NoItinerary> {
  let isBranch = 1;
  let isTerminator = 1;
}

def SVCi : A64I_exceptImpl<0b000, 0b01, "svc">;
def HVCi : A64I_exceptImpl<0b000, 0b10, "hvc">;
def SMCi : A64I_exceptImpl<0b000, 0b11, "smc">;
def BRKi : A64I_exceptImpl<0b001, 0b00, "brk">;
def HLTi : A64I_exceptImpl<0b010, 0b00, "hlt">;

def DCPS1i : A64I_exceptImpl<0b101, 0b01, "dcps1">;
def DCPS2i : A64I_exceptImpl<0b101, 0b10, "dcps2">;
def DCPS3i : A64I_exceptImpl<0b101, 0b11, "dcps3">;

// The immediate is optional for the DCPS instructions, defaulting to 0.
def : InstAlias<"dcps1", (DCPS1i 0)>;
def : InstAlias<"dcps2", (DCPS2i 0)>;
def : InstAlias<"dcps3", (DCPS3i 0)>;

//===----------------------------------------------------------------------===//
// Extract (immediate)
//===----------------------------------------------------------------------===//
// Contains: EXTR + alias ROR

def EXTRwwwi : A64I_extract<0b0, 0b000, 0b0,
                            (outs GPR32:$Rd),
                            (ins GPR32:$Rn, GPR32:$Rm, bitfield32_imm:$LSB),
                            "extr\t$Rd, $Rn, $Rm, $LSB",
                            [(set i32:$Rd,
                                  (A64Extr i32:$Rn, i32:$Rm, imm:$LSB))],
                            NoItinerary>;
def EXTRxxxi : A64I_extract<0b1, 0b000, 0b1,
                            (outs GPR64:$Rd),
                            (ins GPR64:$Rn, GPR64:$Rm, bitfield64_imm:$LSB),
                            "extr\t$Rd, $Rn, $Rm, $LSB",
                            [(set i64:$Rd,
                                  (A64Extr i64:$Rn, i64:$Rm, imm:$LSB))],
                            NoItinerary>;

def : InstAlias<"ror $Rd, $Rs, $LSB",
               (EXTRwwwi GPR32:$Rd, GPR32:$Rs, GPR32:$Rs, bitfield32_imm:$LSB)>;
def : InstAlias<"ror $Rd, $Rs, $LSB",
               (EXTRxxxi GPR64:$Rd, GPR64:$Rs, GPR64:$Rs, bitfield64_imm:$LSB)>;

def : Pat<(rotr i32:$Rn, bitfield32_imm:$LSB),
          (EXTRwwwi $Rn, $Rn, bitfield32_imm:$LSB)>;
def : Pat<(rotr i64:$Rn, bitfield64_imm:$LSB),
          (EXTRxxxi $Rn, $Rn, bitfield64_imm:$LSB)>;

//===----------------------------------------------------------------------===//
// Floating-point compare instructions
//===----------------------------------------------------------------------===//
// Contains: FCMP, FCMPE

def fpzero_asmoperand : AsmOperandClass {
  let Name = "FPZero";
  let ParserMethod = "ParseFPImmOperand";
  let DiagnosticType = "FPZero";
}

def fpz32 : Operand<f32>,
            ComplexPattern<f32, 1, "SelectFPZeroOperand", [fpimm]> {
  let ParserMatchClass = fpzero_asmoperand;
  let PrintMethod = "printFPZeroOperand";
  let DecoderMethod = "DecodeFPZeroOperand";
}

def fpz64 : Operand<f64>,
            ComplexPattern<f64, 1, "SelectFPZeroOperand", [fpimm]> {
  let ParserMatchClass = fpzero_asmoperand;
  let PrintMethod = "printFPZeroOperand";
  let DecoderMethod = "DecodeFPZeroOperand";
}

def fpz64movi : Operand<i64>,
            ComplexPattern<f64, 1, "SelectFPZeroOperand", [fpimm]> {
  let ParserMatchClass = fpzero_asmoperand;
  let PrintMethod = "printFPZeroOperand";
  let DecoderMethod = "DecodeFPZeroOperand";
}

multiclass A64I_fpcmpSignal<bits<2> type, bit imm, dag ins, dag pattern> {
  def _quiet : A64I_fpcmp<0b0, 0b0, type, 0b00, {0b0, imm, 0b0, 0b0, 0b0},
                          (outs), ins, "fcmp\t$Rn, $Rm", [pattern],
                          NoItinerary> {
    let Defs = [NZCV];
  }

  def _sig : A64I_fpcmp<0b0, 0b0, type, 0b00, {0b1, imm, 0b0, 0b0, 0b0},
                        (outs), ins, "fcmpe\t$Rn, $Rm", [], NoItinerary> {
    let Defs = [NZCV];
  }
}

defm FCMPss : A64I_fpcmpSignal<0b00, 0b0, (ins FPR32:$Rn, FPR32:$Rm),
                               (set NZCV, (A64cmp f32:$Rn, f32:$Rm))>;
defm FCMPdd : A64I_fpcmpSignal<0b01, 0b0, (ins FPR64:$Rn, FPR64:$Rm),
                               (set NZCV, (A64cmp f64:$Rn, f64:$Rm))>;

// What would be Rm should be written as 0; note that even though it's called
// "$Rm" here to fit in with the InstrFormats, it's actually an immediate.
defm FCMPsi : A64I_fpcmpSignal<0b00, 0b1, (ins FPR32:$Rn, fpz32:$Rm),
                               (set NZCV, (A64cmp f32:$Rn, fpz32:$Rm))>;

defm FCMPdi : A64I_fpcmpSignal<0b01, 0b1, (ins FPR64:$Rn, fpz64:$Rm),
                               (set NZCV, (A64cmp f64:$Rn, fpz64:$Rm))>;


//===----------------------------------------------------------------------===//
// Floating-point conditional compare instructions
//===----------------------------------------------------------------------===//
// Contains: FCCMP, FCCMPE

class A64I_fpccmpImpl<bits<2> type, bit op, RegisterClass FPR, string asmop>
  : A64I_fpccmp<0b0, 0b0, type, op,
                (outs),
                (ins FPR:$Rn, FPR:$Rm, uimm4:$NZCVImm, cond_code_op:$Cond),
                !strconcat(asmop, "\t$Rn, $Rm, $NZCVImm, $Cond"),
                [], NoItinerary> {
  let Defs = [NZCV];
}

def FCCMPss : A64I_fpccmpImpl<0b00, 0b0, FPR32, "fccmp">;
def FCCMPEss : A64I_fpccmpImpl<0b00, 0b1, FPR32, "fccmpe">;
def FCCMPdd : A64I_fpccmpImpl<0b01, 0b0, FPR64, "fccmp">;
def FCCMPEdd : A64I_fpccmpImpl<0b01, 0b1, FPR64, "fccmpe">;

//===----------------------------------------------------------------------===//
// Floating-point conditional select instructions
//===----------------------------------------------------------------------===//
// Contains: FCSEL

let Uses = [NZCV] in {
  def FCSELsssc : A64I_fpcondsel<0b0, 0b0, 0b00, (outs FPR32:$Rd),
                                 (ins FPR32:$Rn, FPR32:$Rm, cond_code_op:$Cond),
                                 "fcsel\t$Rd, $Rn, $Rm, $Cond",
                                 [(set f32:$Rd, 
                                       (simple_select f32:$Rn, f32:$Rm))],
                                 NoItinerary>;


  def FCSELdddc : A64I_fpcondsel<0b0, 0b0, 0b01, (outs FPR64:$Rd),
                                 (ins FPR64:$Rn, FPR64:$Rm, cond_code_op:$Cond),
                                 "fcsel\t$Rd, $Rn, $Rm, $Cond",
                                 [(set f64:$Rd,
                                       (simple_select f64:$Rn, f64:$Rm))],
                                 NoItinerary>;
}

//===----------------------------------------------------------------------===//
// Floating-point data-processing (1 source)
//===----------------------------------------------------------------------===//
// Contains: FMOV, FABS, FNEG, FSQRT, FCVT, FRINT[NPMZAXI].

def FPNoUnop : PatFrag<(ops node:$val), (fneg node:$val),
                       [{ (void)N; return false; }]>;

// First we do the fairly trivial bunch with uniform "OP s, s" and "OP d, d"
// syntax. Default to no pattern because most are odd enough not to have one.
multiclass A64I_fpdp1sizes<bits<6> opcode, string asmstr,
                           SDPatternOperator opnode = FPNoUnop> {
  def ss : A64I_fpdp1<0b0, 0b0, 0b00, opcode, (outs FPR32:$Rd), (ins FPR32:$Rn),
                     !strconcat(asmstr, "\t$Rd, $Rn"),
                     [(set f32:$Rd, (opnode f32:$Rn))],
                     NoItinerary>;

  def dd : A64I_fpdp1<0b0, 0b0, 0b01, opcode, (outs FPR64:$Rd), (ins FPR64:$Rn),
                     !strconcat(asmstr, "\t$Rd, $Rn"),
                     [(set f64:$Rd, (opnode f64:$Rn))],
                     NoItinerary>;
}

defm FMOV   : A64I_fpdp1sizes<0b000000, "fmov">;
defm FABS   : A64I_fpdp1sizes<0b000001, "fabs", fabs>;
defm FNEG   : A64I_fpdp1sizes<0b000010, "fneg", fneg>;
defm FSQRT  : A64I_fpdp1sizes<0b000011, "fsqrt", fsqrt>;

defm FRINTN : A64I_fpdp1sizes<0b001000, "frintn">;
defm FRINTP : A64I_fpdp1sizes<0b001001, "frintp", fceil>;
defm FRINTM : A64I_fpdp1sizes<0b001010, "frintm", ffloor>;
defm FRINTZ : A64I_fpdp1sizes<0b001011, "frintz", ftrunc>;
defm FRINTA : A64I_fpdp1sizes<0b001100, "frinta">;
defm FRINTX : A64I_fpdp1sizes<0b001110, "frintx", frint>;
defm FRINTI : A64I_fpdp1sizes<0b001111, "frinti", fnearbyint>;

// The FCVT instrucitons have different source and destination register-types,
// but the fields are uniform everywhere a D-register (say) crops up. Package
// this information in a Record.
class FCVTRegType<RegisterClass rc, bits<2> fld, ValueType vt> {
    RegisterClass Class = rc;
    ValueType VT = vt;
    bit t1 = fld{1};
    bit t0 = fld{0};
}

def FCVT16 : FCVTRegType<FPR16, 0b11, f16>;
def FCVT32 : FCVTRegType<FPR32, 0b00, f32>;
def FCVT64 : FCVTRegType<FPR64, 0b01, f64>;

class A64I_fpdp1_fcvt<FCVTRegType DestReg, FCVTRegType SrcReg, SDNode opnode>
  : A64I_fpdp1<0b0, 0b0, {SrcReg.t1, SrcReg.t0},
               {0,0,0,1, DestReg.t1, DestReg.t0},
               (outs DestReg.Class:$Rd), (ins SrcReg.Class:$Rn),
               "fcvt\t$Rd, $Rn",
               [(set DestReg.VT:$Rd, (opnode SrcReg.VT:$Rn))], NoItinerary>;

def FCVTds : A64I_fpdp1_fcvt<FCVT64, FCVT32, fextend>;
def FCVThs : A64I_fpdp1_fcvt<FCVT16, FCVT32, fround>;
def FCVTsd : A64I_fpdp1_fcvt<FCVT32, FCVT64, fround>;
def FCVThd : A64I_fpdp1_fcvt<FCVT16, FCVT64, fround>;
def FCVTsh : A64I_fpdp1_fcvt<FCVT32, FCVT16, fextend>;
def FCVTdh : A64I_fpdp1_fcvt<FCVT64, FCVT16, fextend>;


//===----------------------------------------------------------------------===//
// Floating-point data-processing (2 sources) instructions
//===----------------------------------------------------------------------===//
// Contains: FMUL, FDIV, FADD, FSUB, FMAX, FMIN, FMAXNM, FMINNM, FNMUL

def FPNoBinop : PatFrag<(ops node:$lhs, node:$rhs), (fadd node:$lhs, node:$rhs),
                      [{ (void)N; return false; }]>;

multiclass A64I_fpdp2sizes<bits<4> opcode, string asmstr,
                           SDPatternOperator opnode> {
  def sss : A64I_fpdp2<0b0, 0b0, 0b00, opcode,
                      (outs FPR32:$Rd),
                      (ins FPR32:$Rn, FPR32:$Rm),
                      !strconcat(asmstr, "\t$Rd, $Rn, $Rm"),
                      [(set f32:$Rd, (opnode f32:$Rn, f32:$Rm))],
                      NoItinerary>;

  def ddd : A64I_fpdp2<0b0, 0b0, 0b01, opcode,
                      (outs FPR64:$Rd),
                      (ins FPR64:$Rn, FPR64:$Rm),
                      !strconcat(asmstr, "\t$Rd, $Rn, $Rm"),
                      [(set f64:$Rd, (opnode f64:$Rn, f64:$Rm))],
                      NoItinerary>;
}

let isCommutable = 1 in {
  defm FMUL   : A64I_fpdp2sizes<0b0000, "fmul", fmul>;
  defm FADD   : A64I_fpdp2sizes<0b0010, "fadd", fadd>;

  // No patterns for these.
  defm FMAX   : A64I_fpdp2sizes<0b0100, "fmax", FPNoBinop>;
  defm FMIN   : A64I_fpdp2sizes<0b0101, "fmin", FPNoBinop>;
  defm FMAXNM : A64I_fpdp2sizes<0b0110, "fmaxnm", FPNoBinop>;
  defm FMINNM : A64I_fpdp2sizes<0b0111, "fminnm", FPNoBinop>;

  defm FNMUL  : A64I_fpdp2sizes<0b1000, "fnmul",
                                PatFrag<(ops node:$lhs, node:$rhs),
                                        (fneg (fmul node:$lhs, node:$rhs))> >;
}

defm FDIV : A64I_fpdp2sizes<0b0001, "fdiv", fdiv>;
defm FSUB : A64I_fpdp2sizes<0b0011, "fsub", fsub>;

//===----------------------------------------------------------------------===//
// Floating-point data-processing (3 sources) instructions
//===----------------------------------------------------------------------===//
// Contains: FMADD, FMSUB, FNMADD, FNMSUB

def fmsub : PatFrag<(ops node:$Rn, node:$Rm, node:$Ra),
                    (fma (fneg node:$Rn),  node:$Rm, node:$Ra)>;
def fnmadd : PatFrag<(ops node:$Rn, node:$Rm, node:$Ra),
                     (fma node:$Rn,  node:$Rm, (fneg node:$Ra))>;
def fnmsub : PatFrag<(ops node:$Rn, node:$Rm, node:$Ra),
                     (fma (fneg node:$Rn),  node:$Rm, (fneg node:$Ra))>;

class A64I_fpdp3Impl<string asmop, RegisterClass FPR, ValueType VT,
                     bits<2> type, bit o1, bit o0, SDPatternOperator fmakind>
  : A64I_fpdp3<0b0, 0b0, type, o1, o0, (outs FPR:$Rd),
               (ins FPR:$Rn, FPR:$Rm, FPR:$Ra),
               !strconcat(asmop,"\t$Rd, $Rn, $Rm, $Ra"),
               [(set VT:$Rd, (fmakind VT:$Rn, VT:$Rm, VT:$Ra))],
               NoItinerary>;

def FMADDssss  : A64I_fpdp3Impl<"fmadd",  FPR32, f32, 0b00, 0b0, 0b0, fma>;
def FMSUBssss  : A64I_fpdp3Impl<"fmsub",  FPR32, f32, 0b00, 0b0, 0b1, fmsub>;
def FNMADDssss : A64I_fpdp3Impl<"fnmadd", FPR32, f32, 0b00, 0b1, 0b0, fnmadd>;
def FNMSUBssss : A64I_fpdp3Impl<"fnmsub", FPR32, f32, 0b00, 0b1, 0b1, fnmsub>;

def FMADDdddd  : A64I_fpdp3Impl<"fmadd",  FPR64, f64, 0b01, 0b0, 0b0, fma>;
def FMSUBdddd  : A64I_fpdp3Impl<"fmsub",  FPR64, f64, 0b01, 0b0, 0b1, fmsub>;
def FNMADDdddd : A64I_fpdp3Impl<"fnmadd", FPR64, f64, 0b01, 0b1, 0b0, fnmadd>;
def FNMSUBdddd : A64I_fpdp3Impl<"fnmsub", FPR64, f64, 0b01, 0b1, 0b1, fnmsub>;

// Extra patterns for when we're allowed to optimise separate multiplication and
// addition.
let Predicates = [HasFPARMv8, UseFusedMAC] in {
def : Pat<(f32 (fadd FPR32:$Ra, (f32 (fmul FPR32:$Rn, FPR32:$Rm)))),
          (FMADDssss FPR32:$Rn, FPR32:$Rm, FPR32:$Ra)>;
def : Pat<(f32 (fsub FPR32:$Ra, (f32 (fmul FPR32:$Rn, FPR32:$Rm)))),
          (FMSUBssss FPR32:$Rn, FPR32:$Rm, FPR32:$Ra)>;
def : Pat<(f32 (fsub (f32 (fmul FPR32:$Rn, FPR32:$Rm)), FPR32:$Ra)),
          (FNMADDssss FPR32:$Rn, FPR32:$Rm, FPR32:$Ra)>;
def : Pat<(f32 (fsub (f32 (fneg FPR32:$Ra)), (f32 (fmul FPR32:$Rn, FPR32:$Rm)))),
          (FNMSUBssss FPR32:$Rn, FPR32:$Rm, FPR32:$Ra)>;

def : Pat<(f64 (fadd FPR64:$Ra, (f64 (fmul FPR64:$Rn, FPR64:$Rm)))),
          (FMADDdddd FPR64:$Rn, FPR64:$Rm, FPR64:$Ra)>;
def : Pat<(f64 (fsub FPR64:$Ra, (f64 (fmul FPR64:$Rn, FPR64:$Rm)))),
          (FMSUBdddd FPR64:$Rn, FPR64:$Rm, FPR64:$Ra)>;
def : Pat<(f64 (fsub (f64 (fmul FPR64:$Rn, FPR64:$Rm)), FPR64:$Ra)),
          (FNMADDdddd FPR64:$Rn, FPR64:$Rm, FPR64:$Ra)>;
def : Pat<(f64 (fsub (f64 (fneg FPR64:$Ra)), (f64 (fmul FPR64:$Rn, FPR64:$Rm)))),
          (FNMSUBdddd FPR64:$Rn, FPR64:$Rm, FPR64:$Ra)>;
}


//===----------------------------------------------------------------------===//
// Floating-point <-> fixed-point conversion instructions
//===----------------------------------------------------------------------===//
// Contains: FCVTZS, FCVTZU, SCVTF, UCVTF

// #1-#32 allowed, encoded as "64 - <specified imm>
def fixedpos_asmoperand_i32 : AsmOperandClass {
  let Name = "CVTFixedPos32";
  let RenderMethod = "addCVTFixedPosOperands";
  let PredicateMethod = "isCVTFixedPos<32>";
  let DiagnosticType = "CVTFixedPos32";
}

// Also encoded as "64 - <specified imm>" but #1-#64 allowed.
def fixedpos_asmoperand_i64 : AsmOperandClass {
  let Name = "CVTFixedPos64";
  let RenderMethod = "addCVTFixedPosOperands";
  let PredicateMethod = "isCVTFixedPos<64>";
  let DiagnosticType = "CVTFixedPos64";
}

// We need the cartesian product of f32/f64 i32/i64 operands for
// conversions:
//   + Selection needs to use operands of correct floating type
//   + Assembly parsing and decoding depend on integer width
class cvtfix_i32_op<ValueType FloatVT>
  : Operand<FloatVT>,
    ComplexPattern<FloatVT, 1, "SelectCVTFixedPosOperand<32>", [fpimm]> {
  let ParserMatchClass = fixedpos_asmoperand_i32;
  let DecoderMethod = "DecodeCVT32FixedPosOperand";
  let PrintMethod = "printCVTFixedPosOperand";
}

class cvtfix_i64_op<ValueType FloatVT>
  : Operand<FloatVT>,
    ComplexPattern<FloatVT, 1, "SelectCVTFixedPosOperand<64>", [fpimm]> {
  let ParserMatchClass = fixedpos_asmoperand_i64;
  let PrintMethod = "printCVTFixedPosOperand";
}

// Because of the proliferation of weird operands, it's not really
// worth going for a multiclass here. Oh well.

class A64I_fptofix<bit sf, bits<2> type, bits<3> opcode,
                   RegisterClass GPR, RegisterClass FPR, 
                   ValueType DstTy, ValueType SrcTy, 
                   Operand scale_op, string asmop, SDNode cvtop>
  : A64I_fpfixed<sf, 0b0, type, 0b11, opcode,
                 (outs GPR:$Rd), (ins FPR:$Rn, scale_op:$Scale),
                 !strconcat(asmop, "\t$Rd, $Rn, $Scale"),
                 [(set DstTy:$Rd, (cvtop (fmul SrcTy:$Rn, scale_op:$Scale)))],
                 NoItinerary>;

def FCVTZSwsi : A64I_fptofix<0b0, 0b00, 0b000, GPR32, FPR32, i32, f32,
                             cvtfix_i32_op<f32>, "fcvtzs", fp_to_sint>;
def FCVTZSxsi : A64I_fptofix<0b1, 0b00, 0b000, GPR64, FPR32, i64, f32,
                             cvtfix_i64_op<f32>, "fcvtzs", fp_to_sint>;
def FCVTZUwsi : A64I_fptofix<0b0, 0b00, 0b001, GPR32, FPR32, i32, f32,
                             cvtfix_i32_op<f32>, "fcvtzu", fp_to_uint>;
def FCVTZUxsi : A64I_fptofix<0b1, 0b00, 0b001, GPR64, FPR32, i64, f32,
                             cvtfix_i64_op<f32>, "fcvtzu", fp_to_uint>;

def FCVTZSwdi : A64I_fptofix<0b0, 0b01, 0b000, GPR32, FPR64, i32, f64,
                             cvtfix_i32_op<f64>, "fcvtzs", fp_to_sint>;
def FCVTZSxdi : A64I_fptofix<0b1, 0b01, 0b000, GPR64, FPR64, i64, f64,
                             cvtfix_i64_op<f64>, "fcvtzs", fp_to_sint>;
def FCVTZUwdi : A64I_fptofix<0b0, 0b01, 0b001, GPR32, FPR64, i32, f64,
                             cvtfix_i32_op<f64>, "fcvtzu", fp_to_uint>;
def FCVTZUxdi : A64I_fptofix<0b1, 0b01, 0b001, GPR64, FPR64, i64, f64,
                             cvtfix_i64_op<f64>, "fcvtzu", fp_to_uint>;


class A64I_fixtofp<bit sf, bits<2> type, bits<3> opcode,
                   RegisterClass FPR, RegisterClass GPR,
                   ValueType DstTy, ValueType SrcTy,
                   Operand scale_op, string asmop, SDNode cvtop>
  : A64I_fpfixed<sf, 0b0, type, 0b00, opcode,
                 (outs FPR:$Rd), (ins GPR:$Rn, scale_op:$Scale),
                 !strconcat(asmop, "\t$Rd, $Rn, $Scale"),
                 [(set DstTy:$Rd, (fdiv (cvtop SrcTy:$Rn), scale_op:$Scale))],
                 NoItinerary>;

def SCVTFswi : A64I_fixtofp<0b0, 0b00, 0b010, FPR32, GPR32, f32, i32,
                            cvtfix_i32_op<f32>, "scvtf", sint_to_fp>;
def SCVTFsxi : A64I_fixtofp<0b1, 0b00, 0b010, FPR32, GPR64, f32, i64,
                            cvtfix_i64_op<f32>, "scvtf", sint_to_fp>;
def UCVTFswi : A64I_fixtofp<0b0, 0b00, 0b011, FPR32, GPR32, f32, i32,
                            cvtfix_i32_op<f32>, "ucvtf", uint_to_fp>;
def UCVTFsxi : A64I_fixtofp<0b1, 0b00, 0b011, FPR32, GPR64, f32, i64,
                            cvtfix_i64_op<f32>, "ucvtf", uint_to_fp>;
def SCVTFdwi : A64I_fixtofp<0b0, 0b01, 0b010, FPR64, GPR32, f64, i32,
                            cvtfix_i32_op<f64>, "scvtf", sint_to_fp>;
def SCVTFdxi : A64I_fixtofp<0b1, 0b01, 0b010, FPR64, GPR64, f64, i64,
                            cvtfix_i64_op<f64>, "scvtf", sint_to_fp>;
def UCVTFdwi : A64I_fixtofp<0b0, 0b01, 0b011, FPR64, GPR32, f64, i32,
                            cvtfix_i32_op<f64>, "ucvtf", uint_to_fp>;
def UCVTFdxi : A64I_fixtofp<0b1, 0b01, 0b011, FPR64, GPR64, f64, i64,
                            cvtfix_i64_op<f64>, "ucvtf", uint_to_fp>;

//===----------------------------------------------------------------------===//
// Floating-point <-> integer conversion instructions
//===----------------------------------------------------------------------===//
// Contains: FCVTZS, FCVTZU, SCVTF, UCVTF

class A64I_fpintI<bit sf, bits<2> type, bits<2> rmode, bits<3> opcode,
                   RegisterClass DestPR, RegisterClass SrcPR, string asmop>
  : A64I_fpint<sf, 0b0, type, rmode, opcode, (outs DestPR:$Rd), (ins SrcPR:$Rn),
               !strconcat(asmop, "\t$Rd, $Rn"), [], NoItinerary>;

multiclass A64I_fptointRM<bits<2> rmode, bit o2, string asmop> {
  def Sws : A64I_fpintI<0b0, 0b00, rmode, {o2, 0, 0},
                        GPR32, FPR32, asmop # "s">;
  def Sxs : A64I_fpintI<0b1, 0b00, rmode, {o2, 0, 0},
                        GPR64, FPR32, asmop # "s">;
  def Uws : A64I_fpintI<0b0, 0b00, rmode, {o2, 0, 1},
                        GPR32, FPR32, asmop # "u">;
  def Uxs : A64I_fpintI<0b1, 0b00, rmode, {o2, 0, 1},
                        GPR64, FPR32, asmop # "u">;

  def Swd : A64I_fpintI<0b0, 0b01, rmode, {o2, 0, 0},
                        GPR32, FPR64, asmop # "s">;
  def Sxd : A64I_fpintI<0b1, 0b01, rmode, {o2, 0, 0},
                        GPR64, FPR64, asmop # "s">;
  def Uwd : A64I_fpintI<0b0, 0b01, rmode, {o2, 0, 1},
                        GPR32, FPR64, asmop # "u">;
  def Uxd : A64I_fpintI<0b1, 0b01, rmode, {o2, 0, 1},
                        GPR64, FPR64, asmop # "u">;
}

defm FCVTN : A64I_fptointRM<0b00, 0b0, "fcvtn">;
defm FCVTP : A64I_fptointRM<0b01, 0b0, "fcvtp">;
defm FCVTM : A64I_fptointRM<0b10, 0b0, "fcvtm">;
defm FCVTZ : A64I_fptointRM<0b11, 0b0, "fcvtz">;
defm FCVTA : A64I_fptointRM<0b00, 0b1, "fcvta">;

let Predicates = [HasFPARMv8] in {
def : Pat<(i32 (fp_to_sint f32:$Rn)), (FCVTZSws $Rn)>;
def : Pat<(i64 (fp_to_sint f32:$Rn)), (FCVTZSxs $Rn)>;
def : Pat<(i32 (fp_to_uint f32:$Rn)), (FCVTZUws $Rn)>;
def : Pat<(i64 (fp_to_uint f32:$Rn)), (FCVTZUxs $Rn)>;
def : Pat<(i32 (fp_to_sint f64:$Rn)), (FCVTZSwd $Rn)>;
def : Pat<(i64 (fp_to_sint f64:$Rn)), (FCVTZSxd $Rn)>;
def : Pat<(i32 (fp_to_uint f64:$Rn)), (FCVTZUwd $Rn)>;
def : Pat<(i64 (fp_to_uint f64:$Rn)), (FCVTZUxd $Rn)>;
}

multiclass A64I_inttofp<bit o0, string asmop> {
  def CVTFsw : A64I_fpintI<0b0, 0b00, 0b00, {0, 1, o0}, FPR32, GPR32, asmop>;
  def CVTFsx : A64I_fpintI<0b1, 0b00, 0b00, {0, 1, o0}, FPR32, GPR64, asmop>;
  def CVTFdw : A64I_fpintI<0b0, 0b01, 0b00, {0, 1, o0}, FPR64, GPR32, asmop>;
  def CVTFdx : A64I_fpintI<0b1, 0b01, 0b00, {0, 1, o0}, FPR64, GPR64, asmop>;
}

defm S : A64I_inttofp<0b0, "scvtf">;
defm U : A64I_inttofp<0b1, "ucvtf">;

let Predicates = [HasFPARMv8] in {
def : Pat<(f32 (sint_to_fp i32:$Rn)), (SCVTFsw $Rn)>;
def : Pat<(f32 (sint_to_fp i64:$Rn)), (SCVTFsx $Rn)>;
def : Pat<(f64 (sint_to_fp i32:$Rn)), (SCVTFdw $Rn)>;
def : Pat<(f64 (sint_to_fp i64:$Rn)), (SCVTFdx $Rn)>;
def : Pat<(f32 (uint_to_fp i32:$Rn)), (UCVTFsw $Rn)>;
def : Pat<(f32 (uint_to_fp i64:$Rn)), (UCVTFsx $Rn)>;
def : Pat<(f64 (uint_to_fp i32:$Rn)), (UCVTFdw $Rn)>;
def : Pat<(f64 (uint_to_fp i64:$Rn)), (UCVTFdx $Rn)>;
}

def FMOVws : A64I_fpintI<0b0, 0b00, 0b00, 0b110, GPR32, FPR32, "fmov">;
def FMOVsw : A64I_fpintI<0b0, 0b00, 0b00, 0b111, FPR32, GPR32, "fmov">;
def FMOVxd : A64I_fpintI<0b1, 0b01, 0b00, 0b110, GPR64, FPR64, "fmov">;
def FMOVdx : A64I_fpintI<0b1, 0b01, 0b00, 0b111, FPR64, GPR64, "fmov">;

let Predicates = [HasFPARMv8] in {
def : Pat<(i32 (bitconvert f32:$Rn)), (FMOVws $Rn)>;
def : Pat<(f32 (bitconvert i32:$Rn)), (FMOVsw $Rn)>;
def : Pat<(i64 (bitconvert f64:$Rn)), (FMOVxd $Rn)>;
def : Pat<(f64 (bitconvert i64:$Rn)), (FMOVdx $Rn)>;
}

def lane1_asmoperand : AsmOperandClass {
  let Name = "Lane1";
  let RenderMethod = "addImmOperands";
  let DiagnosticType = "Lane1";
}

def lane1 : Operand<i32> {
  let ParserMatchClass = lane1_asmoperand;
  let PrintMethod = "printBareImmOperand";
}

let DecoderMethod =  "DecodeFMOVLaneInstruction" in {
  def FMOVxv : A64I_fpint<0b1, 0b0, 0b10, 0b01, 0b110,
                          (outs GPR64:$Rd), (ins VPR128:$Rn, lane1:$Lane),
                          "fmov\t$Rd, $Rn.d[$Lane]", [], NoItinerary>;

  def FMOVvx : A64I_fpint<0b1, 0b0, 0b10, 0b01, 0b111,
                          (outs VPR128:$Rd), (ins GPR64:$Rn, lane1:$Lane),
                          "fmov\t$Rd.d[$Lane], $Rn", [], NoItinerary>;
}

let Predicates = [HasFPARMv8] in {
def : InstAlias<"fmov $Rd, $Rn.2d[$Lane]",
                (FMOVxv GPR64:$Rd, VPR128:$Rn, lane1:$Lane), 0b0>;

def : InstAlias<"fmov $Rd.2d[$Lane], $Rn",
                (FMOVvx VPR128:$Rd, GPR64:$Rn, lane1:$Lane), 0b0>;
}

//===----------------------------------------------------------------------===//
// Floating-point immediate instructions
//===----------------------------------------------------------------------===//
// Contains: FMOV

def fpimm_asmoperand : AsmOperandClass {
  let Name = "FMOVImm";
  let ParserMethod = "ParseFPImmOperand";
  let DiagnosticType = "FPImm";
}

// The MCOperand for these instructions are the encoded 8-bit values.
def SDXF_fpimm : SDNodeXForm<fpimm, [{
  uint32_t Imm8;
  A64Imms::isFPImm(N->getValueAPF(), Imm8);
  return CurDAG->getTargetConstant(Imm8, MVT::i32);
}]>;

class fmov_operand<ValueType FT>
  : Operand<i32>,
    PatLeaf<(FT fpimm), [{ return A64Imms::isFPImm(N->getValueAPF()); }],
            SDXF_fpimm> {
  let PrintMethod = "printFPImmOperand";
  let ParserMatchClass = fpimm_asmoperand;
}

def fmov32_operand : fmov_operand<f32>;
def fmov64_operand : fmov_operand<f64>;

class A64I_fpimm_impl<bits<2> type, RegisterClass Reg, ValueType VT,
                      Operand fmov_operand>
  : A64I_fpimm<0b0, 0b0, type, 0b00000,
               (outs Reg:$Rd),
               (ins fmov_operand:$Imm8),
               "fmov\t$Rd, $Imm8",
               [(set VT:$Rd, fmov_operand:$Imm8)],
               NoItinerary>;

def FMOVsi : A64I_fpimm_impl<0b00, FPR32, f32, fmov32_operand>;
def FMOVdi : A64I_fpimm_impl<0b01, FPR64, f64, fmov64_operand>;

//===----------------------------------------------------------------------===//
// Load-register (literal) instructions
//===----------------------------------------------------------------------===//
// Contains: LDR, LDRSW, PRFM

def ldrlit_label_asmoperand : AsmOperandClass {
  let Name = "LoadLitLabel";
  let RenderMethod = "addLabelOperands<19, 4>";
  let DiagnosticType = "Label";
}

def ldrlit_label : Operand<i64> {
  let EncoderMethod = "getLoadLitLabelOpValue";

  // This label is a 19-bit offset from PC, scaled by the instruction-width: 4.
  let PrintMethod = "printLabelOperand<19, 4>";
  let ParserMatchClass = ldrlit_label_asmoperand;
  let OperandType = "OPERAND_PCREL";
}

// Various instructions take an immediate value (which can always be used),
// where some numbers have a symbolic name to make things easier. These operands
// and the associated functions abstract away the differences.
multiclass namedimm<string prefix, string mapper> {
  def _asmoperand : AsmOperandClass {
    let Name = "NamedImm" # prefix;
    let PredicateMethod = "isUImm";
    let RenderMethod = "addImmOperands";
    let ParserMethod = "ParseNamedImmOperand<" # mapper # ">";
    let DiagnosticType = "NamedImm_" # prefix;
  }

  def _op : Operand<i32> {
    let ParserMatchClass = !cast<AsmOperandClass>(prefix # "_asmoperand");
    let PrintMethod = "printNamedImmOperand<" # mapper # ">";
    let DecoderMethod = "DecodeNamedImmOperand<" # mapper # ">";
  }
}

defm prefetch : namedimm<"prefetch", "A64PRFM::PRFMMapper">;

class A64I_LDRlitSimple<bits<2> opc, bit v, RegisterClass OutReg,
                      list<dag> patterns = []>
   : A64I_LDRlit<opc, v, (outs OutReg:$Rt), (ins ldrlit_label:$Imm19),
                 "ldr\t$Rt, $Imm19", patterns, NoItinerary>;

let mayLoad = 1 in {
  def LDRw_lit : A64I_LDRlitSimple<0b00, 0b0, GPR32>;
  def LDRx_lit : A64I_LDRlitSimple<0b01, 0b0, GPR64>;
}

let Predicates = [HasFPARMv8] in {
def LDRs_lit  : A64I_LDRlitSimple<0b00, 0b1, FPR32>;
def LDRd_lit  : A64I_LDRlitSimple<0b01, 0b1, FPR64>;
}

let mayLoad = 1 in {
  let Predicates = [HasFPARMv8] in {
  def LDRq_lit : A64I_LDRlitSimple<0b10, 0b1, FPR128>;
  }


  def LDRSWx_lit : A64I_LDRlit<0b10, 0b0,
                               (outs GPR64:$Rt),
                               (ins ldrlit_label:$Imm19),
                               "ldrsw\t$Rt, $Imm19",
                               [], NoItinerary>;

  def PRFM_lit : A64I_LDRlit<0b11, 0b0,
                             (outs), (ins prefetch_op:$Rt, ldrlit_label:$Imm19),
                             "prfm\t$Rt, $Imm19",
                             [], NoItinerary>;
}

//===----------------------------------------------------------------------===//
// Load-store exclusive instructions
//===----------------------------------------------------------------------===//
// Contains: STXRB, STXRH, STXR, LDXRB, LDXRH, LDXR. STXP, LDXP, STLXRB,
//           STLXRH, STLXR, LDAXRB, LDAXRH, LDAXR, STLXP, LDAXP, STLRB,
//           STLRH, STLR, LDARB, LDARH, LDAR

// Since these instructions have the undefined register bits set to 1 in
// their canonical form, we need a post encoder method to set those bits
// to 1 when encoding these instructions. We do this using the
// fixLoadStoreExclusive function. This function has template parameters:
//
// fixLoadStoreExclusive<int hasRs, int hasRt2>
//
// hasRs indicates that the instruction uses the Rs field, so we won't set
// it to 1 (and the same for Rt2). We don't need template parameters for
// the other register fiels since Rt and Rn are always used.

// This operand parses a GPR64xsp register, followed by an optional immediate
// #0.
def GPR64xsp0_asmoperand : AsmOperandClass {
  let Name = "GPR64xsp0";
  let PredicateMethod = "isWrappedReg";
  let RenderMethod = "addRegOperands";
  let ParserMethod = "ParseLSXAddressOperand";
  // Diagnostics are provided by ParserMethod
}

def GPR64xsp0 : RegisterOperand<GPR64xsp> {
  let ParserMatchClass = GPR64xsp0_asmoperand;
}

//===----------------------------------
// Store-exclusive (releasing & normal)
//===----------------------------------

class A64I_SRexs_impl<bits<2> size, bits<3> opcode, string asm, dag outs,
                        dag ins, list<dag> pat,
                        InstrItinClass itin> :
       A64I_LDSTex_stn <size,
                        opcode{2}, 0, opcode{1}, opcode{0},
                        outs, ins,
                        !strconcat(asm, "\t$Rs, $Rt, [$Rn]"),
                        pat, itin> {
  let mayStore = 1;
  let PostEncoderMethod = "fixLoadStoreExclusive<1,0>";
  let Constraints = "@earlyclobber $Rs";
}

multiclass A64I_SRex<string asmstr, bits<3> opcode, string prefix> {
  def _byte:  A64I_SRexs_impl<0b00, opcode, !strconcat(asmstr, "b"),
                              (outs GPR32:$Rs), (ins GPR32:$Rt, GPR64xsp0:$Rn),
                              [], NoItinerary>;

  def _hword:  A64I_SRexs_impl<0b01, opcode, !strconcat(asmstr, "h"),
                               (outs GPR32:$Rs), (ins GPR32:$Rt, GPR64xsp0:$Rn),
                               [],NoItinerary>;

  def _word:  A64I_SRexs_impl<0b10, opcode, asmstr,
                              (outs GPR32:$Rs), (ins GPR32:$Rt, GPR64xsp0:$Rn),
                              [], NoItinerary>;

  def _dword: A64I_SRexs_impl<0b11, opcode, asmstr,
                              (outs GPR32:$Rs), (ins GPR64:$Rt, GPR64xsp0:$Rn),
                              [], NoItinerary>;
}

defm STXR  : A64I_SRex<"stxr",  0b000, "STXR">;
defm STLXR : A64I_SRex<"stlxr", 0b001, "STLXR">;

//===----------------------------------
// Loads
//===----------------------------------

class A64I_LRexs_impl<bits<2> size, bits<3> opcode, string asm, dag outs,
                        dag ins, list<dag> pat,
                        InstrItinClass itin> :
        A64I_LDSTex_tn <size,
                        opcode{2}, 1, opcode{1}, opcode{0},
                        outs, ins,
                        !strconcat(asm, "\t$Rt, [$Rn]"),
                        pat, itin> {
  let mayLoad = 1;
  let PostEncoderMethod = "fixLoadStoreExclusive<0,0>";
}

multiclass A64I_LRex<string asmstr, bits<3> opcode> {
  def _byte:  A64I_LRexs_impl<0b00, opcode, !strconcat(asmstr, "b"),
                            (outs GPR32:$Rt), (ins GPR64xsp0:$Rn),
                            [], NoItinerary>;

  def _hword:  A64I_LRexs_impl<0b01, opcode, !strconcat(asmstr, "h"),
                            (outs GPR32:$Rt), (ins GPR64xsp0:$Rn),
                            [], NoItinerary>;

  def _word:  A64I_LRexs_impl<0b10, opcode, asmstr,
                            (outs GPR32:$Rt), (ins GPR64xsp0:$Rn),
                            [], NoItinerary>;

  def _dword: A64I_LRexs_impl<0b11, opcode, asmstr,
                            (outs GPR64:$Rt), (ins GPR64xsp0:$Rn),
                            [], NoItinerary>;
}

defm LDXR  : A64I_LRex<"ldxr",  0b000>;
defm LDAXR : A64I_LRex<"ldaxr", 0b001>;
defm LDAR  : A64I_LRex<"ldar",  0b101>;

class acquiring_load<PatFrag base>
  : PatFrag<(ops node:$ptr), (base node:$ptr), [{
  AtomicOrdering Ordering = cast<AtomicSDNode>(N)->getOrdering();
  return Ordering == Acquire || Ordering == SequentiallyConsistent;
}]>;

def atomic_load_acquire_8  : acquiring_load<atomic_load_8>;
def atomic_load_acquire_16 : acquiring_load<atomic_load_16>;
def atomic_load_acquire_32 : acquiring_load<atomic_load_32>;
def atomic_load_acquire_64 : acquiring_load<atomic_load_64>;

def : Pat<(atomic_load_acquire_8  i64:$Rn), (LDAR_byte  $Rn)>;
def : Pat<(atomic_load_acquire_16 i64:$Rn), (LDAR_hword $Rn)>;
def : Pat<(atomic_load_acquire_32 i64:$Rn), (LDAR_word  $Rn)>;
def : Pat<(atomic_load_acquire_64 i64:$Rn), (LDAR_dword $Rn)>;

//===----------------------------------
// Store-release (no exclusivity)
//===----------------------------------

class A64I_SLexs_impl<bits<2> size, bits<3> opcode, string asm, dag outs,
                        dag ins, list<dag> pat,
                        InstrItinClass itin> :
        A64I_LDSTex_tn <size,
                        opcode{2}, 0, opcode{1}, opcode{0},
                        outs, ins,
                        !strconcat(asm, "\t$Rt, [$Rn]"),
                        pat, itin> {
  let mayStore = 1;
  let PostEncoderMethod = "fixLoadStoreExclusive<0,0>";
}

class releasing_store<PatFrag base>
  : PatFrag<(ops node:$ptr, node:$val), (base node:$ptr, node:$val), [{
  AtomicOrdering Ordering = cast<AtomicSDNode>(N)->getOrdering();
  return Ordering == Release || Ordering == SequentiallyConsistent;
}]>;

def atomic_store_release_8  : releasing_store<atomic_store_8>;
def atomic_store_release_16 : releasing_store<atomic_store_16>;
def atomic_store_release_32 : releasing_store<atomic_store_32>;
def atomic_store_release_64 : releasing_store<atomic_store_64>;

multiclass A64I_SLex<string asmstr, bits<3> opcode, string prefix> {
  def _byte:  A64I_SLexs_impl<0b00, opcode, !strconcat(asmstr, "b"),
                            (outs), (ins GPR32:$Rt, GPR64xsp0:$Rn),
                            [(atomic_store_release_8 i64:$Rn, i32:$Rt)],
                            NoItinerary>;

  def _hword:  A64I_SLexs_impl<0b01, opcode, !strconcat(asmstr, "h"),
                           (outs), (ins GPR32:$Rt, GPR64xsp0:$Rn),
                           [(atomic_store_release_16 i64:$Rn, i32:$Rt)],
                           NoItinerary>;

  def _word:  A64I_SLexs_impl<0b10, opcode, asmstr,
                           (outs), (ins GPR32:$Rt, GPR64xsp0:$Rn),
                           [(atomic_store_release_32 i64:$Rn, i32:$Rt)],
                           NoItinerary>;

  def _dword: A64I_SLexs_impl<0b11, opcode, asmstr,
                           (outs), (ins GPR64:$Rt, GPR64xsp0:$Rn),
                           [(atomic_store_release_64 i64:$Rn, i64:$Rt)],
                           NoItinerary>;
}

defm STLR  : A64I_SLex<"stlr", 0b101, "STLR">;

//===----------------------------------
// Store-exclusive pair (releasing & normal)
//===----------------------------------

class A64I_SPexs_impl<bits<2> size, bits<3> opcode, string asm, dag outs,
                        dag ins, list<dag> pat,
                        InstrItinClass itin> :
     A64I_LDSTex_stt2n <size,
                        opcode{2}, 0, opcode{1}, opcode{0},
                        outs, ins,
                        !strconcat(asm, "\t$Rs, $Rt, $Rt2, [$Rn]"),
                        pat, itin> {
  let mayStore = 1;
}


multiclass A64I_SPex<string asmstr, bits<3> opcode> {
  def _word:  A64I_SPexs_impl<0b10, opcode, asmstr, (outs),
                            (ins GPR32:$Rs, GPR32:$Rt, GPR32:$Rt2,
                                 GPR64xsp0:$Rn),
                            [], NoItinerary>;

  def _dword: A64I_SPexs_impl<0b11, opcode, asmstr, (outs),
                            (ins GPR32:$Rs, GPR64:$Rt, GPR64:$Rt2,
                                            GPR64xsp0:$Rn),
                            [], NoItinerary>;
}

defm STXP  : A64I_SPex<"stxp", 0b010>;
defm STLXP : A64I_SPex<"stlxp", 0b011>;

//===----------------------------------
// Load-exclusive pair (acquiring & normal)
//===----------------------------------

class A64I_LPexs_impl<bits<2> size, bits<3> opcode, string asm, dag outs,
                        dag ins, list<dag> pat,
                        InstrItinClass itin> :
      A64I_LDSTex_tt2n <size,
                        opcode{2}, 1, opcode{1}, opcode{0},
                        outs, ins,
                        !strconcat(asm, "\t$Rt, $Rt2, [$Rn]"),
                        pat, itin>{
  let mayLoad = 1;
  let DecoderMethod = "DecodeLoadPairExclusiveInstruction";
  let PostEncoderMethod = "fixLoadStoreExclusive<0,1>";
}

multiclass A64I_LPex<string asmstr, bits<3> opcode> {
  def _word:  A64I_LPexs_impl<0b10, opcode, asmstr,
                            (outs GPR32:$Rt, GPR32:$Rt2),
                            (ins GPR64xsp0:$Rn),
                            [], NoItinerary>;

  def _dword: A64I_LPexs_impl<0b11, opcode, asmstr,
                            (outs GPR64:$Rt, GPR64:$Rt2),
                            (ins GPR64xsp0:$Rn),
                            [], NoItinerary>;
}

defm LDXP  : A64I_LPex<"ldxp", 0b010>;
defm LDAXP : A64I_LPex<"ldaxp", 0b011>;

//===----------------------------------------------------------------------===//
// Load-store register (unscaled immediate) instructions
//===----------------------------------------------------------------------===//
// Contains: LDURB, LDURH, LDRUSB, LDRUSH, LDRUSW, STUR, STURB, STURH and PRFUM
//
// and
//
//===----------------------------------------------------------------------===//
// Load-store register (register offset) instructions
//===----------------------------------------------------------------------===//
// Contains: LDRB, LDRH, LDRSB, LDRSH, LDRSW, STR, STRB, STRH and PRFM
//
// and
//
//===----------------------------------------------------------------------===//
// Load-store register (unsigned immediate) instructions
//===----------------------------------------------------------------------===//
// Contains: LDRB, LDRH, LDRSB, LDRSH, LDRSW, STR, STRB, STRH and PRFM
//
// and
//
//===----------------------------------------------------------------------===//
// Load-store register (immediate post-indexed) instructions
//===----------------------------------------------------------------------===//
// Contains: STRB, STRH, STR, LDRB, LDRH, LDR, LDRSB, LDRSH, LDRSW
//
// and
//
//===----------------------------------------------------------------------===//
// Load-store register (immediate pre-indexed) instructions
//===----------------------------------------------------------------------===//
// Contains: STRB, STRH, STR, LDRB, LDRH, LDR, LDRSB, LDRSH, LDRSW

// Note that patterns are much later on in a completely separate section (they
// need ADRPxi to be defined).

//===-------------------------------
// 1. Various operands needed
//===-------------------------------

//===-------------------------------
// 1.1 Unsigned 12-bit immediate operands
//===-------------------------------
// The addressing mode for these instructions consists of an unsigned 12-bit
// immediate which is scaled by the size of the memory access.
//
// We represent this in the MC layer by two operands:
//     1. A base register.
//     2. A 12-bit immediate: not multiplied by access size, so "LDR x0,[x0,#8]"
//        would have '1' in this field.
// This means that separate functions are needed for converting representations
// which *are* aware of the intended access size.

// Anything that creates an MCInst (Decoding, selection and AsmParsing) has to
// know the access size via some means. An isolated operand does not have this
// information unless told from here, which means we need separate tablegen
// Operands for each access size. This multiclass takes care of instantiating
// the correct template functions in the rest of the backend.

//===-------------------------------
// 1.1 Unsigned 12-bit immediate operands
//===-------------------------------

multiclass offsets_uimm12<int MemSize, string prefix> {
  def uimm12_asmoperand : AsmOperandClass {
    let Name = "OffsetUImm12_" # MemSize;
    let PredicateMethod = "isOffsetUImm12<" # MemSize # ">";
    let RenderMethod = "addOffsetUImm12Operands<" # MemSize # ">";
    let DiagnosticType = "LoadStoreUImm12_" # MemSize;
  }

  // Pattern is really no more than an ImmLeaf, but predicated on MemSize which
  // complicates things beyond TableGen's ken.
  def uimm12 : Operand<i64>,
               ComplexPattern<i64, 1, "SelectOffsetUImm12<" # MemSize # ">"> {
    let ParserMatchClass
      = !cast<AsmOperandClass>(prefix # uimm12_asmoperand);

    let PrintMethod = "printOffsetUImm12Operand<" # MemSize # ">";
    let EncoderMethod = "getOffsetUImm12OpValue<" # MemSize # ">";
  }
}

defm byte_  : offsets_uimm12<1, "byte_">;
defm hword_ : offsets_uimm12<2, "hword_">;
defm word_  : offsets_uimm12<4, "word_">;
defm dword_ : offsets_uimm12<8, "dword_">;
defm qword_ : offsets_uimm12<16, "qword_">;

//===-------------------------------
// 1.1 Signed 9-bit immediate operands
//===-------------------------------

// The MCInst is expected to store the bit-wise encoding of the value,
// which amounts to lopping off the extended sign bits.
def SDXF_simm9 : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getZExtValue() & 0x1ff, MVT::i32);
}]>;

def simm9_asmoperand : AsmOperandClass {
  let Name = "SImm9";
  let PredicateMethod = "isSImm<9>";
  let RenderMethod = "addSImmOperands<9>";
  let DiagnosticType = "LoadStoreSImm9";
}

def simm9 : Operand<i64>,
            ImmLeaf<i64, [{ return Imm >= -0x100 && Imm <= 0xff; }],
            SDXF_simm9> {
  let PrintMethod = "printOffsetSImm9Operand";
  let ParserMatchClass = simm9_asmoperand;
}


//===-------------------------------
// 1.3 Register offset extensions
//===-------------------------------

// The assembly-syntax for these addressing-modes is:
//    [<Xn|SP>, <R><m> {, <extend> {<amount>}}]
//
// The essential semantics are:
//     + <amount> is a shift: #<log(transfer size)> or #0
//     + <R> can be W or X.
//     + If <R> is W, <extend> can be UXTW or SXTW
//     + If <R> is X, <extend> can be LSL or SXTX
//
// The trickiest of those constraints is that Rm can be either GPR32 or GPR64,
// which will need separate instructions for LLVM type-consistency. We'll also
// need separate operands, of course.
multiclass regexts<int MemSize, int RmSize, RegisterClass GPR,
                   string Rm, string prefix> {
  def regext_asmoperand : AsmOperandClass {
    let Name = "AddrRegExtend_" # MemSize # "_" #  Rm;
    let PredicateMethod = "isAddrRegExtend<" # MemSize # "," # RmSize # ">";
    let RenderMethod = "addAddrRegExtendOperands<" # MemSize # ">";
    let DiagnosticType = "LoadStoreExtend" # RmSize # "_" # MemSize;
  }

  def regext : Operand<i64> {
    let PrintMethod
      = "printAddrRegExtendOperand<" # MemSize # ", " # RmSize # ">";

    let DecoderMethod = "DecodeAddrRegExtendOperand";
    let ParserMatchClass
      = !cast<AsmOperandClass>(prefix # regext_asmoperand);
  }
}

multiclass regexts_wx<int MemSize, string prefix> {
  // Rm is an X-register if LSL or SXTX are specified as the shift.
  defm Xm_ : regexts<MemSize, 64, GPR64, "Xm", prefix # "Xm_">;

  // Rm is a W-register if UXTW or SXTW are specified as the shift.
  defm Wm_ : regexts<MemSize, 32, GPR32, "Wm", prefix # "Wm_">;
}

defm byte_  : regexts_wx<1, "byte_">;
defm hword_ : regexts_wx<2, "hword_">;
defm word_  : regexts_wx<4, "word_">;
defm dword_ : regexts_wx<8, "dword_">;
defm qword_ : regexts_wx<16, "qword_">;


//===------------------------------
// 2. The instructions themselves.
//===------------------------------

// We have the following instructions to implement:
// |                 | B     | H     | W     | X      |
// |-----------------+-------+-------+-------+--------|
// | unsigned str    | STRB  | STRH  | STR   | STR    |
// | unsigned ldr    | LDRB  | LDRH  | LDR   | LDR    |
// | signed ldr to W | LDRSB | LDRSH | -     | -      |
// | signed ldr to X | LDRSB | LDRSH | LDRSW | (PRFM) |

// This will instantiate the LDR/STR instructions you'd expect to use for an
// unsigned datatype (first two rows above) or floating-point register, which is
// reasonably uniform across all access sizes.


//===------------------------------
// 2.1 Regular instructions
//===------------------------------

// This class covers the basic unsigned or irrelevantly-signed loads and stores,
// to general-purpose and floating-point registers.

class AddrParams<string prefix> {
  Operand uimm12 = !cast<Operand>(prefix # "_uimm12");

  Operand regextWm = !cast<Operand>(prefix # "_Wm_regext");
  Operand regextXm = !cast<Operand>(prefix # "_Xm_regext");
}

def byte_addrparams : AddrParams<"byte">;
def hword_addrparams : AddrParams<"hword">;
def word_addrparams : AddrParams<"word">;
def dword_addrparams : AddrParams<"dword">;
def qword_addrparams : AddrParams<"qword">;

multiclass A64I_LDRSTR_unsigned<string prefix, bits<2> size, bit v,
                                bit high_opc, string asmsuffix,
                                RegisterClass GPR, AddrParams params> {
  // Unsigned immediate
  def _STR : A64I_LSunsigimm<size, v, {high_opc, 0b0},
                     (outs), (ins GPR:$Rt, GPR64xsp:$Rn, params.uimm12:$UImm12),
                     "str" # asmsuffix # "\t$Rt, [$Rn, $UImm12]",
                     [], NoItinerary> {
    let mayStore = 1;
  }
  def : InstAlias<"str" # asmsuffix # " $Rt, [$Rn]",
                (!cast<Instruction>(prefix # "_STR") GPR:$Rt, GPR64xsp:$Rn, 0)>;

  def _LDR : A64I_LSunsigimm<size, v, {high_opc, 0b1},
                      (outs GPR:$Rt), (ins GPR64xsp:$Rn, params.uimm12:$UImm12),
                      "ldr" #  asmsuffix # "\t$Rt, [$Rn, $UImm12]",
                      [], NoItinerary> {
    let mayLoad = 1;
  }
  def : InstAlias<"ldr" # asmsuffix # " $Rt, [$Rn]",
                (!cast<Instruction>(prefix # "_LDR") GPR:$Rt, GPR64xsp:$Rn, 0)>;

  // Register offset (four of these: load/store and Wm/Xm).
  let mayLoad = 1 in {
    def _Wm_RegOffset_LDR : A64I_LSregoff<size, v, {high_opc, 0b1}, 0b0,
                            (outs GPR:$Rt),
                            (ins GPR64xsp:$Rn, GPR32:$Rm, params.regextWm:$Ext),
                            "ldr" # asmsuffix # "\t$Rt, [$Rn, $Rm, $Ext]",
                            [], NoItinerary>;

    def _Xm_RegOffset_LDR : A64I_LSregoff<size, v, {high_opc, 0b1}, 0b1,
                            (outs GPR:$Rt),
                            (ins GPR64xsp:$Rn, GPR64:$Rm, params.regextXm:$Ext),
                            "ldr" # asmsuffix # "\t$Rt, [$Rn, $Rm, $Ext]",
                            [], NoItinerary>;
  }
  def : InstAlias<"ldr" # asmsuffix # " $Rt, [$Rn, $Rm]",
        (!cast<Instruction>(prefix # "_Xm_RegOffset_LDR") GPR:$Rt, GPR64xsp:$Rn,
                                                          GPR64:$Rm, 2)>;

  let mayStore = 1 in {
    def _Wm_RegOffset_STR : A64I_LSregoff<size, v, {high_opc, 0b0}, 0b0,
                                  (outs), (ins GPR:$Rt, GPR64xsp:$Rn, GPR32:$Rm,
                                               params.regextWm:$Ext),
                                  "str" # asmsuffix # "\t$Rt, [$Rn, $Rm, $Ext]",
                                  [], NoItinerary>;

    def _Xm_RegOffset_STR : A64I_LSregoff<size, v, {high_opc, 0b0}, 0b1,
                                  (outs), (ins GPR:$Rt, GPR64xsp:$Rn, GPR64:$Rm,
                                               params.regextXm:$Ext),
                                  "str" # asmsuffix # "\t$Rt, [$Rn, $Rm, $Ext]",
                                  [], NoItinerary>;
  }
  def : InstAlias<"str" # asmsuffix # " $Rt, [$Rn, $Rm]",
      (!cast<Instruction>(prefix # "_Xm_RegOffset_STR") GPR:$Rt, GPR64xsp:$Rn,
                                                        GPR64:$Rm, 2)>;

  // Unaligned immediate
  def _STUR : A64I_LSunalimm<size, v, {high_opc, 0b0},
                             (outs), (ins GPR:$Rt, GPR64xsp:$Rn, simm9:$SImm9),
                             "stur" # asmsuffix # "\t$Rt, [$Rn, $SImm9]",
                             [], NoItinerary> {
    let mayStore = 1;
  }
  def : InstAlias<"stur" # asmsuffix # " $Rt, [$Rn]",
               (!cast<Instruction>(prefix # "_STUR") GPR:$Rt, GPR64xsp:$Rn, 0)>;

  def _LDUR : A64I_LSunalimm<size, v, {high_opc, 0b1},
                             (outs GPR:$Rt), (ins GPR64xsp:$Rn, simm9:$SImm9),
                             "ldur" # asmsuffix # "\t$Rt, [$Rn, $SImm9]",
                             [], NoItinerary> {
    let mayLoad = 1;
  }
  def : InstAlias<"ldur" # asmsuffix # " $Rt, [$Rn]",
               (!cast<Instruction>(prefix # "_LDUR") GPR:$Rt, GPR64xsp:$Rn, 0)>;

  // Post-indexed
  def _PostInd_STR : A64I_LSpostind<size, v, {high_opc, 0b0},
                               (outs GPR64xsp:$Rn_wb),
                               (ins GPR:$Rt, GPR64xsp:$Rn, simm9:$SImm9),
                               "str" # asmsuffix # "\t$Rt, [$Rn], $SImm9",
                               [], NoItinerary> {
    let Constraints = "$Rn = $Rn_wb";
    let mayStore = 1;

    // Decoder only needed for unpredictability checking (FIXME).
    let DecoderMethod = "DecodeSingleIndexedInstruction";
  }

  def _PostInd_LDR : A64I_LSpostind<size, v, {high_opc, 0b1},
                                    (outs GPR:$Rt, GPR64xsp:$Rn_wb),
                                    (ins GPR64xsp:$Rn, simm9:$SImm9),
                                    "ldr" # asmsuffix # "\t$Rt, [$Rn], $SImm9",
                                    [], NoItinerary> {
    let mayLoad = 1;
    let Constraints = "$Rn = $Rn_wb";
    let DecoderMethod = "DecodeSingleIndexedInstruction";
  }

  // Pre-indexed
  def _PreInd_STR : A64I_LSpreind<size, v, {high_opc, 0b0},
                               (outs GPR64xsp:$Rn_wb),
                               (ins GPR:$Rt, GPR64xsp:$Rn, simm9:$SImm9),
                               "str" # asmsuffix # "\t$Rt, [$Rn, $SImm9]!",
                               [], NoItinerary> {
    let Constraints = "$Rn = $Rn_wb";
    let mayStore = 1;

    // Decoder only needed for unpredictability checking (FIXME).
    let DecoderMethod = "DecodeSingleIndexedInstruction";
  }

  def _PreInd_LDR : A64I_LSpreind<size, v, {high_opc, 0b1},
                                    (outs GPR:$Rt, GPR64xsp:$Rn_wb),
                                    (ins GPR64xsp:$Rn, simm9:$SImm9),
                                    "ldr" # asmsuffix # "\t$Rt, [$Rn, $SImm9]!",
                                    [], NoItinerary> {
    let mayLoad = 1;
    let Constraints = "$Rn = $Rn_wb";
    let DecoderMethod = "DecodeSingleIndexedInstruction";
  }

}

// STRB/LDRB: First define the instructions
defm LS8
  : A64I_LDRSTR_unsigned<"LS8", 0b00, 0b0, 0b0, "b", GPR32, byte_addrparams>;

// STRH/LDRH
defm LS16
  : A64I_LDRSTR_unsigned<"LS16", 0b01, 0b0, 0b0, "h", GPR32, hword_addrparams>;


// STR/LDR to/from a W register
defm LS32
  : A64I_LDRSTR_unsigned<"LS32", 0b10, 0b0, 0b0, "", GPR32, word_addrparams>;

// STR/LDR to/from an X register
defm LS64
  : A64I_LDRSTR_unsigned<"LS64", 0b11, 0b0, 0b0, "", GPR64, dword_addrparams>;

let Predicates = [HasFPARMv8] in {
// STR/LDR to/from a B register
defm LSFP8
  : A64I_LDRSTR_unsigned<"LSFP8", 0b00, 0b1, 0b0, "", FPR8, byte_addrparams>;

// STR/LDR to/from an H register
defm LSFP16
  : A64I_LDRSTR_unsigned<"LSFP16", 0b01, 0b1, 0b0, "", FPR16, hword_addrparams>;

// STR/LDR to/from an S register
defm LSFP32
  : A64I_LDRSTR_unsigned<"LSFP32", 0b10, 0b1, 0b0, "", FPR32, word_addrparams>;
// STR/LDR to/from a D register
defm LSFP64
  : A64I_LDRSTR_unsigned<"LSFP64", 0b11, 0b1, 0b0, "", FPR64, dword_addrparams>;
// STR/LDR to/from a Q register
defm LSFP128
  : A64I_LDRSTR_unsigned<"LSFP128", 0b00, 0b1, 0b1, "", FPR128,
                         qword_addrparams>;
}

//===------------------------------
// 2.3 Signed loads
//===------------------------------

// Byte and half-word signed loads can both go into either an X or a W register,
// so it's worth factoring out. Signed word loads don't fit because there is no
// W version.
multiclass A64I_LDR_signed<bits<2> size, string asmopcode, AddrParams params,
                           string prefix> {
  // Unsigned offset
  def w : A64I_LSunsigimm<size, 0b0, 0b11,
                          (outs GPR32:$Rt),
                          (ins GPR64xsp:$Rn, params.uimm12:$UImm12),
                          "ldrs" # asmopcode # "\t$Rt, [$Rn, $UImm12]",
                          [], NoItinerary> {
    let mayLoad = 1;
  }
  def : InstAlias<"ldrs" # asmopcode # " $Rt, [$Rn]",
                  (!cast<Instruction>(prefix # w) GPR32:$Rt, GPR64xsp:$Rn, 0)>;

  def x : A64I_LSunsigimm<size, 0b0, 0b10,
                          (outs GPR64:$Rt),
                          (ins GPR64xsp:$Rn, params.uimm12:$UImm12),
                          "ldrs" # asmopcode # "\t$Rt, [$Rn, $UImm12]",
                          [], NoItinerary> {
    let mayLoad = 1;
  }
  def : InstAlias<"ldrs" # asmopcode # " $Rt, [$Rn]",
                  (!cast<Instruction>(prefix # x) GPR64:$Rt, GPR64xsp:$Rn, 0)>;

  // Register offset
  let mayLoad = 1 in {
    def w_Wm_RegOffset : A64I_LSregoff<size, 0b0, 0b11, 0b0,
                            (outs GPR32:$Rt),
                            (ins GPR64xsp:$Rn, GPR32:$Rm, params.regextWm:$Ext),
                            "ldrs" # asmopcode # "\t$Rt, [$Rn, $Rm, $Ext]",
                            [], NoItinerary>;

    def w_Xm_RegOffset : A64I_LSregoff<size, 0b0, 0b11, 0b1,
                            (outs GPR32:$Rt),
                            (ins GPR64xsp:$Rn, GPR64:$Rm, params.regextXm:$Ext),
                            "ldrs" # asmopcode # "\t$Rt, [$Rn, $Rm, $Ext]",
                            [], NoItinerary>;

    def x_Wm_RegOffset : A64I_LSregoff<size, 0b0, 0b10, 0b0,
                            (outs GPR64:$Rt),
                            (ins GPR64xsp:$Rn, GPR32:$Rm, params.regextWm:$Ext),
                            "ldrs" # asmopcode # "\t$Rt, [$Rn, $Rm, $Ext]",
                            [], NoItinerary>;

    def x_Xm_RegOffset : A64I_LSregoff<size, 0b0, 0b10, 0b1,
                            (outs GPR64:$Rt),
                            (ins GPR64xsp:$Rn, GPR64:$Rm, params.regextXm:$Ext),
                            "ldrs" # asmopcode # "\t$Rt, [$Rn, $Rm, $Ext]",
                            [], NoItinerary>;
  }
  def : InstAlias<"ldrs" # asmopcode # " $Rt, [$Rn, $Rm]",
        (!cast<Instruction>(prefix # "w_Xm_RegOffset") GPR32:$Rt, GPR64xsp:$Rn,
                                                       GPR64:$Rm, 2)>;

  def : InstAlias<"ldrs" # asmopcode # " $Rt, [$Rn, $Rm]",
        (!cast<Instruction>(prefix # "x_Xm_RegOffset") GPR64:$Rt, GPR64xsp:$Rn,
                                                       GPR64:$Rm, 2)>;


  let mayLoad = 1 in {
    // Unaligned offset
    def w_U : A64I_LSunalimm<size, 0b0, 0b11,
                             (outs GPR32:$Rt),
                             (ins GPR64xsp:$Rn, simm9:$SImm9),
                             "ldurs" # asmopcode # "\t$Rt, [$Rn, $SImm9]",
                             [], NoItinerary>;

    def x_U : A64I_LSunalimm<size, 0b0, 0b10,
                             (outs GPR64:$Rt),
                             (ins GPR64xsp:$Rn, simm9:$SImm9),
                             "ldurs" # asmopcode # "\t$Rt, [$Rn, $SImm9]",
                             [], NoItinerary>;


    // Post-indexed
    def w_PostInd : A64I_LSpostind<size, 0b0, 0b11,
                                 (outs GPR32:$Rt, GPR64xsp:$Rn_wb),
                                 (ins GPR64xsp:$Rn, simm9:$SImm9),
                                 "ldrs" # asmopcode # "\t$Rt, [$Rn], $SImm9",
                                 [], NoItinerary> {
      let Constraints = "$Rn = $Rn_wb";
      let DecoderMethod = "DecodeSingleIndexedInstruction";
    }

    def x_PostInd : A64I_LSpostind<size, 0b0, 0b10,
                                   (outs GPR64:$Rt, GPR64xsp:$Rn_wb),
                                   (ins GPR64xsp:$Rn, simm9:$SImm9),
                                   "ldrs" # asmopcode # "\t$Rt, [$Rn], $SImm9",
                                   [], NoItinerary> {
      let Constraints = "$Rn = $Rn_wb";
      let DecoderMethod = "DecodeSingleIndexedInstruction";
    }

    // Pre-indexed
    def w_PreInd : A64I_LSpreind<size, 0b0, 0b11,
                                 (outs GPR32:$Rt, GPR64xsp:$Rn_wb),
                                 (ins GPR64xsp:$Rn, simm9:$SImm9),
                                 "ldrs" # asmopcode # "\t$Rt, [$Rn, $SImm9]!",
                                 [], NoItinerary> {
      let Constraints = "$Rn = $Rn_wb";
      let DecoderMethod = "DecodeSingleIndexedInstruction";
    }

    def x_PreInd : A64I_LSpreind<size, 0b0, 0b10,
                                 (outs GPR64:$Rt, GPR64xsp:$Rn_wb),
                                 (ins GPR64xsp:$Rn, simm9:$SImm9),
                                 "ldrs" # asmopcode # "\t$Rt, [$Rn, $SImm9]!",
                                 [], NoItinerary> {
      let Constraints = "$Rn = $Rn_wb";
      let DecoderMethod = "DecodeSingleIndexedInstruction";
    }
  } // let mayLoad = 1
}

// LDRSB
defm LDRSB : A64I_LDR_signed<0b00, "b", byte_addrparams, "LDRSB">;
// LDRSH
defm LDRSH : A64I_LDR_signed<0b01, "h", hword_addrparams, "LDRSH">;

// LDRSW: load a 32-bit register, sign-extending to 64-bits.
def LDRSWx
    : A64I_LSunsigimm<0b10, 0b0, 0b10,
                    (outs GPR64:$Rt),
                    (ins GPR64xsp:$Rn, word_uimm12:$UImm12),
                    "ldrsw\t$Rt, [$Rn, $UImm12]",
                    [], NoItinerary> {
  let mayLoad = 1;
}
def : InstAlias<"ldrsw $Rt, [$Rn]", (LDRSWx GPR64:$Rt, GPR64xsp:$Rn, 0)>;

let mayLoad = 1 in {
  def LDRSWx_Wm_RegOffset : A64I_LSregoff<0b10, 0b0, 0b10, 0b0,
                             (outs GPR64:$Rt),
                             (ins GPR64xsp:$Rn, GPR32:$Rm, word_Wm_regext:$Ext),
                             "ldrsw\t$Rt, [$Rn, $Rm, $Ext]",
                             [], NoItinerary>;

  def LDRSWx_Xm_RegOffset : A64I_LSregoff<0b10, 0b0, 0b10, 0b1,
                             (outs GPR64:$Rt),
                             (ins GPR64xsp:$Rn, GPR64:$Rm, word_Xm_regext:$Ext),
                             "ldrsw\t$Rt, [$Rn, $Rm, $Ext]",
                             [], NoItinerary>;
}
def : InstAlias<"ldrsw $Rt, [$Rn, $Rm]",
                (LDRSWx_Xm_RegOffset GPR64:$Rt, GPR64xsp:$Rn, GPR64:$Rm, 2)>;


def LDURSWx
    : A64I_LSunalimm<0b10, 0b0, 0b10,
                    (outs GPR64:$Rt),
                    (ins GPR64xsp:$Rn, simm9:$SImm9),
                    "ldursw\t$Rt, [$Rn, $SImm9]",
                    [], NoItinerary> {
  let mayLoad = 1;
}
def : InstAlias<"ldursw $Rt, [$Rn]", (LDURSWx GPR64:$Rt, GPR64xsp:$Rn, 0)>;

def LDRSWx_PostInd
    : A64I_LSpostind<0b10, 0b0, 0b10,
                    (outs GPR64:$Rt, GPR64xsp:$Rn_wb),
                    (ins GPR64xsp:$Rn, simm9:$SImm9),
                    "ldrsw\t$Rt, [$Rn], $SImm9",
                    [], NoItinerary> {
  let mayLoad = 1;
  let Constraints = "$Rn = $Rn_wb";
  let DecoderMethod = "DecodeSingleIndexedInstruction";
}

def LDRSWx_PreInd : A64I_LSpreind<0b10, 0b0, 0b10,
                                 (outs GPR64:$Rt, GPR64xsp:$Rn_wb),
                                 (ins GPR64xsp:$Rn, simm9:$SImm9),
                                 "ldrsw\t$Rt, [$Rn, $SImm9]!",
                                 [], NoItinerary> {
  let mayLoad = 1;
  let Constraints = "$Rn = $Rn_wb";
  let DecoderMethod = "DecodeSingleIndexedInstruction";
}

//===------------------------------
// 2.4 Prefetch operations
//===------------------------------

def PRFM : A64I_LSunsigimm<0b11, 0b0, 0b10, (outs),
                 (ins prefetch_op:$Rt, GPR64xsp:$Rn, dword_uimm12:$UImm12),
                 "prfm\t$Rt, [$Rn, $UImm12]",
                 [], NoItinerary> {
  let mayLoad = 1;
}
def : InstAlias<"prfm $Rt, [$Rn]",
                (PRFM prefetch_op:$Rt, GPR64xsp:$Rn, 0)>;

let mayLoad = 1 in {
  def PRFM_Wm_RegOffset : A64I_LSregoff<0b11, 0b0, 0b10, 0b0, (outs),
                                        (ins prefetch_op:$Rt, GPR64xsp:$Rn,
                                             GPR32:$Rm, dword_Wm_regext:$Ext),
                                        "prfm\t$Rt, [$Rn, $Rm, $Ext]",
                                        [], NoItinerary>;
  def PRFM_Xm_RegOffset : A64I_LSregoff<0b11, 0b0, 0b10, 0b1, (outs),
                                        (ins prefetch_op:$Rt, GPR64xsp:$Rn,
                                             GPR64:$Rm, dword_Xm_regext:$Ext),
                                        "prfm\t$Rt, [$Rn, $Rm, $Ext]",
                                        [], NoItinerary>;
}

def : InstAlias<"prfm $Rt, [$Rn, $Rm]",
                (PRFM_Xm_RegOffset prefetch_op:$Rt, GPR64xsp:$Rn,
                                   GPR64:$Rm, 2)>;


def PRFUM : A64I_LSunalimm<0b11, 0b0, 0b10, (outs),
                         (ins prefetch_op:$Rt, GPR64xsp:$Rn, simm9:$SImm9),
                         "prfum\t$Rt, [$Rn, $SImm9]",
                         [], NoItinerary> {
  let mayLoad = 1;
}
def : InstAlias<"prfum $Rt, [$Rn]",
                (PRFUM prefetch_op:$Rt, GPR64xsp:$Rn, 0)>;

//===----------------------------------------------------------------------===//
// Load-store register (unprivileged) instructions
//===----------------------------------------------------------------------===//
// Contains: LDTRB, LDTRH, LDTRSB, LDTRSH, LDTRSW, STTR, STTRB and STTRH

// These instructions very much mirror the "unscaled immediate" loads, but since
// there are no floating-point variants we need to split them out into their own
// section to avoid instantiation of "ldtr d0, [sp]" etc.

multiclass A64I_LDTRSTTR<bits<2> size, string asmsuffix, RegisterClass GPR,
                         string prefix> {
  def _UnPriv_STR : A64I_LSunpriv<size, 0b0, 0b00,
                              (outs), (ins GPR:$Rt, GPR64xsp:$Rn, simm9:$SImm9),
                              "sttr" # asmsuffix # "\t$Rt, [$Rn, $SImm9]",
                              [], NoItinerary> {
    let mayStore = 1;
  }

  def : InstAlias<"sttr" # asmsuffix # " $Rt, [$Rn]",
         (!cast<Instruction>(prefix # "_UnPriv_STR") GPR:$Rt, GPR64xsp:$Rn, 0)>;

  def _UnPriv_LDR : A64I_LSunpriv<size, 0b0, 0b01,
                               (outs GPR:$Rt), (ins GPR64xsp:$Rn, simm9:$SImm9),
                               "ldtr" # asmsuffix # "\t$Rt, [$Rn, $SImm9]",
                               [], NoItinerary> {
    let mayLoad = 1;
  }

  def : InstAlias<"ldtr" # asmsuffix # " $Rt, [$Rn]",
         (!cast<Instruction>(prefix # "_UnPriv_LDR") GPR:$Rt, GPR64xsp:$Rn, 0)>;

}

// STTRB/LDTRB: First define the instructions
defm LS8 : A64I_LDTRSTTR<0b00, "b", GPR32, "LS8">;

// STTRH/LDTRH
defm LS16 : A64I_LDTRSTTR<0b01, "h", GPR32, "LS16">;

// STTR/LDTR to/from a W register
defm LS32 : A64I_LDTRSTTR<0b10, "", GPR32, "LS32">;

// STTR/LDTR to/from an X register
defm LS64 : A64I_LDTRSTTR<0b11, "", GPR64, "LS64">;

// Now a class for the signed instructions that can go to either 32 or 64
// bits...
multiclass A64I_LDTR_signed<bits<2> size, string asmopcode, string prefix> {
  let mayLoad = 1 in {
    def w : A64I_LSunpriv<size, 0b0, 0b11,
                          (outs GPR32:$Rt),
                          (ins GPR64xsp:$Rn, simm9:$SImm9),
                          "ldtrs" # asmopcode # "\t$Rt, [$Rn, $SImm9]",
                          [], NoItinerary>;

    def x : A64I_LSunpriv<size, 0b0, 0b10,
                          (outs GPR64:$Rt),
                          (ins GPR64xsp:$Rn, simm9:$SImm9),
                          "ldtrs" # asmopcode # "\t$Rt, [$Rn, $SImm9]",
                          [], NoItinerary>;
  }

  def : InstAlias<"ldtrs" # asmopcode # " $Rt, [$Rn]",
                 (!cast<Instruction>(prefix # "w") GPR32:$Rt, GPR64xsp:$Rn, 0)>;

  def : InstAlias<"ldtrs" # asmopcode # " $Rt, [$Rn]",
                 (!cast<Instruction>(prefix # "x") GPR64:$Rt, GPR64xsp:$Rn, 0)>;

}

// LDTRSB
defm LDTRSB : A64I_LDTR_signed<0b00, "b", "LDTRSB">;
// LDTRSH
defm LDTRSH : A64I_LDTR_signed<0b01, "h", "LDTRSH">;

// And finally LDTRSW which only goes to 64 bits.
def LDTRSWx : A64I_LSunpriv<0b10, 0b0, 0b10,
                            (outs GPR64:$Rt),
                            (ins GPR64xsp:$Rn, simm9:$SImm9),
                            "ldtrsw\t$Rt, [$Rn, $SImm9]",
                            [], NoItinerary> {
  let mayLoad = 1;
}
def : InstAlias<"ldtrsw $Rt, [$Rn]", (LDTRSWx GPR64:$Rt, GPR64xsp:$Rn, 0)>;

//===----------------------------------------------------------------------===//
// Load-store register pair (offset) instructions
//===----------------------------------------------------------------------===//
//
// and
//
//===----------------------------------------------------------------------===//
// Load-store register pair (post-indexed) instructions
//===----------------------------------------------------------------------===//
// Contains: STP, LDP, LDPSW
//
// and
//
//===----------------------------------------------------------------------===//
// Load-store register pair (pre-indexed) instructions
//===----------------------------------------------------------------------===//
// Contains: STP, LDP, LDPSW
//
// and
//
//===----------------------------------------------------------------------===//
// Load-store non-temporal register pair (offset) instructions
//===----------------------------------------------------------------------===//
// Contains: STNP, LDNP


// Anything that creates an MCInst (Decoding, selection and AsmParsing) has to
// know the access size via some means. An isolated operand does not have this
// information unless told from here, which means we need separate tablegen
// Operands for each access size. This multiclass takes care of instantiating
// the correct template functions in the rest of the backend.

multiclass offsets_simm7<string MemSize, string prefix> {
  // The bare signed 7-bit immediate is used in post-indexed instructions, but
  // because of the scaling performed a generic "simm7" operand isn't
  // appropriate here either.
  def simm7_asmoperand : AsmOperandClass {
    let Name = "SImm7_Scaled" # MemSize;
    let PredicateMethod = "isSImm7Scaled<" # MemSize # ">";
    let RenderMethod = "addSImm7ScaledOperands<" # MemSize # ">";
    let DiagnosticType = "LoadStoreSImm7_" # MemSize;
  }

  def simm7 : Operand<i64> {
    let PrintMethod = "printSImm7ScaledOperand<" # MemSize # ">";
    let ParserMatchClass = !cast<AsmOperandClass>(prefix # "simm7_asmoperand");
  }
}

defm word_  : offsets_simm7<"4", "word_">;
defm dword_ : offsets_simm7<"8", "dword_">;
defm qword_ : offsets_simm7<"16", "qword_">;

multiclass A64I_LSPsimple<bits<2> opc, bit v, RegisterClass SomeReg,
                          Operand simm7, string prefix> {
  def _STR : A64I_LSPoffset<opc, v, 0b0, (outs),
                    (ins SomeReg:$Rt, SomeReg:$Rt2, GPR64xsp:$Rn, simm7:$SImm7),
                    "stp\t$Rt, $Rt2, [$Rn, $SImm7]", [], NoItinerary> {
    let mayStore = 1;
    let DecoderMethod = "DecodeLDSTPairInstruction";
  }
  def : InstAlias<"stp $Rt, $Rt2, [$Rn]",
                  (!cast<Instruction>(prefix # "_STR") SomeReg:$Rt,
                                                SomeReg:$Rt2, GPR64xsp:$Rn, 0)>;

  def _LDR : A64I_LSPoffset<opc, v, 0b1,
                            (outs SomeReg:$Rt, SomeReg:$Rt2),
                            (ins GPR64xsp:$Rn, simm7:$SImm7),
                            "ldp\t$Rt, $Rt2, [$Rn, $SImm7]", [], NoItinerary> {
    let mayLoad = 1;
    let DecoderMethod = "DecodeLDSTPairInstruction";
  }
  def : InstAlias<"ldp $Rt, $Rt2, [$Rn]",
                  (!cast<Instruction>(prefix # "_LDR") SomeReg:$Rt,
                                                SomeReg:$Rt2, GPR64xsp:$Rn, 0)>;

  def _PostInd_STR : A64I_LSPpostind<opc, v, 0b0,
                               (outs GPR64xsp:$Rn_wb),
                               (ins SomeReg:$Rt, SomeReg:$Rt2,
                                    GPR64xsp:$Rn,
                                    simm7:$SImm7),
                               "stp\t$Rt, $Rt2, [$Rn], $SImm7",
                               [], NoItinerary> {
    let mayStore = 1;
    let Constraints = "$Rn = $Rn_wb";

    // Decoder only needed for unpredictability checking (FIXME).
    let DecoderMethod = "DecodeLDSTPairInstruction";
  }

  def _PostInd_LDR : A64I_LSPpostind<opc, v, 0b1,
                        (outs SomeReg:$Rt, SomeReg:$Rt2, GPR64xsp:$Rn_wb),
                        (ins GPR64xsp:$Rn, simm7:$SImm7),
                        "ldp\t$Rt, $Rt2, [$Rn], $SImm7",
                        [], NoItinerary> {
    let mayLoad = 1;
    let Constraints = "$Rn = $Rn_wb";
    let DecoderMethod = "DecodeLDSTPairInstruction";
  }

  def _PreInd_STR : A64I_LSPpreind<opc, v, 0b0, (outs GPR64xsp:$Rn_wb),
                    (ins SomeReg:$Rt, SomeReg:$Rt2, GPR64xsp:$Rn, simm7:$SImm7),
                    "stp\t$Rt, $Rt2, [$Rn, $SImm7]!",
                    [], NoItinerary> {
    let mayStore = 1;
    let Constraints = "$Rn = $Rn_wb";
    let DecoderMethod = "DecodeLDSTPairInstruction";
  }

  def _PreInd_LDR : A64I_LSPpreind<opc, v, 0b1,
                              (outs SomeReg:$Rt, SomeReg:$Rt2, GPR64xsp:$Rn_wb),
                              (ins GPR64xsp:$Rn, simm7:$SImm7),
                              "ldp\t$Rt, $Rt2, [$Rn, $SImm7]!",
                              [], NoItinerary> {
    let mayLoad = 1;
    let Constraints = "$Rn = $Rn_wb";
    let DecoderMethod = "DecodeLDSTPairInstruction";
  }

  def _NonTemp_STR : A64I_LSPnontemp<opc, v, 0b0, (outs),
                    (ins SomeReg:$Rt, SomeReg:$Rt2, GPR64xsp:$Rn, simm7:$SImm7),
                    "stnp\t$Rt, $Rt2, [$Rn, $SImm7]", [], NoItinerary> {
    let mayStore = 1;
    let DecoderMethod = "DecodeLDSTPairInstruction";
  }
  def : InstAlias<"stnp $Rt, $Rt2, [$Rn]",
                  (!cast<Instruction>(prefix # "_NonTemp_STR") SomeReg:$Rt,
                                                SomeReg:$Rt2, GPR64xsp:$Rn, 0)>;

  def _NonTemp_LDR : A64I_LSPnontemp<opc, v, 0b1,
                            (outs SomeReg:$Rt, SomeReg:$Rt2),
                            (ins GPR64xsp:$Rn, simm7:$SImm7),
                            "ldnp\t$Rt, $Rt2, [$Rn, $SImm7]", [], NoItinerary> {
    let mayLoad = 1;
    let DecoderMethod = "DecodeLDSTPairInstruction";
  }
  def : InstAlias<"ldnp $Rt, $Rt2, [$Rn]",
                  (!cast<Instruction>(prefix # "_NonTemp_LDR") SomeReg:$Rt,
                                                SomeReg:$Rt2, GPR64xsp:$Rn, 0)>;

}


defm LSPair32 : A64I_LSPsimple<0b00, 0b0, GPR32, word_simm7, "LSPair32">;
defm LSPair64 : A64I_LSPsimple<0b10, 0b0, GPR64, dword_simm7, "LSPair64">;

let Predicates = [HasFPARMv8] in {
defm LSFPPair32 : A64I_LSPsimple<0b00, 0b1, FPR32, word_simm7, "LSFPPair32">;
defm LSFPPair64 : A64I_LSPsimple<0b01, 0b1, FPR64,  dword_simm7, "LSFPPair64">;
defm LSFPPair128 : A64I_LSPsimple<0b10, 0b1, FPR128, qword_simm7,
                                  "LSFPPair128">;
}


def LDPSWx : A64I_LSPoffset<0b01, 0b0, 0b1,
                           (outs GPR64:$Rt, GPR64:$Rt2),
                           (ins GPR64xsp:$Rn, word_simm7:$SImm7),
                           "ldpsw\t$Rt, $Rt2, [$Rn, $SImm7]", [], NoItinerary> {
  let mayLoad = 1;
  let DecoderMethod = "DecodeLDSTPairInstruction";
}
def : InstAlias<"ldpsw $Rt, $Rt2, [$Rn]",
                (LDPSWx GPR64:$Rt, GPR64:$Rt2, GPR64xsp:$Rn, 0)>;

def LDPSWx_PostInd : A64I_LSPpostind<0b01, 0b0, 0b1,
                                  (outs GPR64:$Rt, GPR64:$Rt2, GPR64:$Rn_wb),
                                  (ins GPR64xsp:$Rn, word_simm7:$SImm7),
                                  "ldpsw\t$Rt, $Rt2, [$Rn], $SImm7",
                                  [], NoItinerary> {
  let mayLoad = 1;
  let Constraints = "$Rn = $Rn_wb";
  let DecoderMethod = "DecodeLDSTPairInstruction";
}

def LDPSWx_PreInd : A64I_LSPpreind<0b01, 0b0, 0b1,
                                   (outs GPR64:$Rt, GPR64:$Rt2, GPR64:$Rn_wb),
                                   (ins GPR64xsp:$Rn, word_simm7:$SImm7),
                                   "ldpsw\t$Rt, $Rt2, [$Rn, $SImm7]!",
                                   [], NoItinerary> {
  let mayLoad = 1;
  let Constraints = "$Rn = $Rn_wb";
  let DecoderMethod = "DecodeLDSTPairInstruction";
}

//===----------------------------------------------------------------------===//
// Logical (immediate) instructions
//===----------------------------------------------------------------------===//
// Contains: AND, ORR, EOR, ANDS, + aliases TST, MOV

multiclass logical_imm_operands<string prefix, string note,
                                int size, ValueType VT> {
  def _asmoperand : AsmOperandClass {
    let Name = "LogicalImm" # note # size;
    let PredicateMethod = "isLogicalImm" # note # "<" # size # ">";
    let RenderMethod = "addLogicalImmOperands<" # size # ">";
    let DiagnosticType = "LogicalSecondSource";
  }

  def _operand
        : Operand<VT>, ComplexPattern<VT, 1, "SelectLogicalImm", [imm]> {
    let ParserMatchClass = !cast<AsmOperandClass>(prefix # "_asmoperand");
    let PrintMethod = "printLogicalImmOperand<" # size # ">";
    let DecoderMethod = "DecodeLogicalImmOperand<" # size # ">";
  }
}

defm logical_imm32 : logical_imm_operands<"logical_imm32", "", 32, i32>;
defm logical_imm64 : logical_imm_operands<"logical_imm64", "", 64, i64>;

// The mov versions only differ in assembly parsing, where they
// exclude values representable with either MOVZ or MOVN.
defm logical_imm32_mov
  : logical_imm_operands<"logical_imm32_mov", "MOV", 32, i32>;
defm logical_imm64_mov
  : logical_imm_operands<"logical_imm64_mov", "MOV", 64, i64>;


multiclass A64I_logimmSizes<bits<2> opc, string asmop, SDNode opnode> {
  def wwi : A64I_logicalimm<0b0, opc, (outs GPR32wsp:$Rd),
                         (ins GPR32:$Rn, logical_imm32_operand:$Imm),
                         !strconcat(asmop, "\t$Rd, $Rn, $Imm"),
                         [(set i32:$Rd,
                               (opnode i32:$Rn, logical_imm32_operand:$Imm))],
                         NoItinerary>;

  def xxi : A64I_logicalimm<0b1, opc, (outs GPR64xsp:$Rd),
                         (ins GPR64:$Rn, logical_imm64_operand:$Imm),
                         !strconcat(asmop, "\t$Rd, $Rn, $Imm"),
                         [(set i64:$Rd,
                               (opnode i64:$Rn, logical_imm64_operand:$Imm))],
                         NoItinerary>;
}

defm AND : A64I_logimmSizes<0b00, "and", and>;
defm ORR : A64I_logimmSizes<0b01, "orr", or>;
defm EOR : A64I_logimmSizes<0b10, "eor", xor>;

let Defs = [NZCV] in {
  def ANDSwwi : A64I_logicalimm<0b0, 0b11, (outs GPR32:$Rd),
                                (ins GPR32:$Rn, logical_imm32_operand:$Imm),
                                "ands\t$Rd, $Rn, $Imm",
                                [], NoItinerary>;

  def ANDSxxi : A64I_logicalimm<0b1, 0b11, (outs GPR64:$Rd),
                                (ins GPR64:$Rn, logical_imm64_operand:$Imm),
                                "ands\t$Rd, $Rn, $Imm",
                                [], NoItinerary>;
}


def : InstAlias<"tst $Rn, $Imm",
                (ANDSwwi WZR, GPR32:$Rn, logical_imm32_operand:$Imm)>;
def : InstAlias<"tst $Rn, $Imm",
                (ANDSxxi XZR, GPR64:$Rn, logical_imm64_operand:$Imm)>;
def : InstAlias<"mov $Rd, $Imm",
                (ORRwwi GPR32wsp:$Rd, WZR, logical_imm32_mov_operand:$Imm)>;
def : InstAlias<"mov $Rd, $Imm",
                (ORRxxi GPR64xsp:$Rd, XZR, logical_imm64_mov_operand:$Imm)>;

//===----------------------------------------------------------------------===//
// Logical (shifted register) instructions
//===----------------------------------------------------------------------===//
// Contains: AND, BIC, ORR, ORN, EOR, EON, ANDS, BICS + aliases TST, MVN, MOV

// Operand for optimizing (icmp (and LHS, RHS), 0, SomeCode). In theory "ANDS"
// behaves differently for unsigned comparisons, so we defensively only allow
// signed or n/a as the operand. In practice "unsigned greater than 0" is "not
// equal to 0" and LLVM gives us this.
def signed_cond : PatLeaf<(cond), [{
  return !isUnsignedIntSetCC(N->get());
}]>;


// These instructions share their "shift" operands with add/sub (shifted
// register instructions). They are defined there.

// N.b. the commutable parameter is just !N. It will be first against the wall
// when the revolution comes.
multiclass logical_shifts<string prefix, bit sf, bits<2> opc,
                          bit N, bit commutable,
                          string asmop, SDPatternOperator opfrag, ValueType ty,
                          RegisterClass GPR, list<Register> defs> {
  let isCommutable = commutable, Defs = defs in {
  def _lsl : A64I_logicalshift<sf, opc, 0b00, N,
                       (outs GPR:$Rd),
                       (ins GPR:$Rn, GPR:$Rm,
                            !cast<Operand>("lsl_operand_" # ty):$Imm6),
                       !strconcat(asmop, "\t$Rd, $Rn, $Rm, $Imm6"),
                       [(set ty:$Rd, (opfrag ty:$Rn, (shl ty:$Rm,
                            !cast<Operand>("lsl_operand_" # ty):$Imm6))
                       )],
                       NoItinerary>;

  def _lsr : A64I_logicalshift<sf, opc, 0b01, N,
                       (outs GPR:$Rd),
                       (ins GPR:$Rn, GPR:$Rm,
                            !cast<Operand>("lsr_operand_" # ty):$Imm6),
                       !strconcat(asmop, "\t$Rd, $Rn, $Rm, $Imm6"),
                       [(set ty:$Rd, (opfrag ty:$Rn, (srl ty:$Rm,
                            !cast<Operand>("lsr_operand_" # ty):$Imm6))
                       )],
                       NoItinerary>;

  def _asr : A64I_logicalshift<sf, opc, 0b10, N,
                       (outs GPR:$Rd),
                       (ins GPR:$Rn, GPR:$Rm,
                            !cast<Operand>("asr_operand_" # ty):$Imm6),
                       !strconcat(asmop, "\t$Rd, $Rn, $Rm, $Imm6"),
                       [(set ty:$Rd, (opfrag ty:$Rn, (sra ty:$Rm,
                            !cast<Operand>("asr_operand_" # ty):$Imm6))
                       )],
                       NoItinerary>;

  def _ror : A64I_logicalshift<sf, opc, 0b11, N,
                       (outs GPR:$Rd),
                       (ins GPR:$Rn, GPR:$Rm,
                            !cast<Operand>("ror_operand_" # ty):$Imm6),
                       !strconcat(asmop, "\t$Rd, $Rn, $Rm, $Imm6"),
                       [(set ty:$Rd, (opfrag ty:$Rn, (rotr ty:$Rm,
                            !cast<Operand>("ror_operand_" # ty):$Imm6))
                       )],
                       NoItinerary>;
  }

  def _noshift
      : InstAlias<!strconcat(asmop, " $Rd, $Rn, $Rm"),
                 (!cast<Instruction>(prefix # "_lsl") GPR:$Rd, GPR:$Rn,
                                                      GPR:$Rm, 0)>;

  def : Pat<(opfrag ty:$Rn, ty:$Rm),
            (!cast<Instruction>(prefix # "_lsl") $Rn, $Rm, 0)>;
}

multiclass logical_sizes<string prefix, bits<2> opc, bit N, bit commutable,
                         string asmop, SDPatternOperator opfrag,
                         list<Register> defs> {
  defm xxx : logical_shifts<prefix # "xxx", 0b1, opc, N,
                            commutable, asmop, opfrag, i64, GPR64, defs>;
  defm www : logical_shifts<prefix # "www", 0b0, opc, N,
                            commutable, asmop, opfrag, i32, GPR32, defs>;
}


defm AND : logical_sizes<"AND", 0b00, 0b0, 0b1, "and", and, []>;
defm ORR : logical_sizes<"ORR", 0b01, 0b0, 0b1, "orr", or, []>;
defm EOR : logical_sizes<"EOR", 0b10, 0b0, 0b1, "eor", xor, []>;
defm ANDS : logical_sizes<"ANDS", 0b11, 0b0, 0b1, "ands",
             PatFrag<(ops node:$lhs, node:$rhs), (and node:$lhs, node:$rhs),
                     [{ (void)N; return false; }]>,
             [NZCV]>;

defm BIC : logical_sizes<"BIC", 0b00, 0b1, 0b0, "bic",
                         PatFrag<(ops node:$lhs, node:$rhs),
                                 (and node:$lhs, (not node:$rhs))>, []>;
defm ORN : logical_sizes<"ORN", 0b01, 0b1, 0b0, "orn",
                         PatFrag<(ops node:$lhs, node:$rhs),
                                 (or node:$lhs, (not node:$rhs))>, []>;
defm EON : logical_sizes<"EON", 0b10, 0b1, 0b0, "eon",
                         PatFrag<(ops node:$lhs, node:$rhs),
                                 (xor node:$lhs, (not node:$rhs))>, []>;
defm BICS : logical_sizes<"BICS", 0b11, 0b1, 0b0, "bics",
                          PatFrag<(ops node:$lhs, node:$rhs),
                                  (and node:$lhs, (not node:$rhs)),
                                  [{ (void)N; return false; }]>,
                          [NZCV]>;

multiclass tst_shifts<string prefix, bit sf, ValueType ty, RegisterClass GPR> {
  let isCommutable = 1, Rd = 0b11111, Defs = [NZCV] in {
  def _lsl : A64I_logicalshift<sf, 0b11, 0b00, 0b0,
                       (outs),
                       (ins GPR:$Rn, GPR:$Rm,
                            !cast<Operand>("lsl_operand_" # ty):$Imm6),
                       "tst\t$Rn, $Rm, $Imm6",
                       [(set NZCV, (A64setcc (and ty:$Rn, (shl ty:$Rm,
                           !cast<Operand>("lsl_operand_" # ty):$Imm6)),
                                          0, signed_cond))],
                       NoItinerary>;


  def _lsr : A64I_logicalshift<sf, 0b11, 0b01, 0b0,
                       (outs),
                       (ins GPR:$Rn, GPR:$Rm,
                            !cast<Operand>("lsr_operand_" # ty):$Imm6),
                       "tst\t$Rn, $Rm, $Imm6",
                       [(set NZCV, (A64setcc (and ty:$Rn, (srl ty:$Rm,
                           !cast<Operand>("lsr_operand_" # ty):$Imm6)),
                                          0, signed_cond))],
                       NoItinerary>;

  def _asr : A64I_logicalshift<sf, 0b11, 0b10, 0b0,
                       (outs),
                       (ins GPR:$Rn, GPR:$Rm,
                            !cast<Operand>("asr_operand_" # ty):$Imm6),
                       "tst\t$Rn, $Rm, $Imm6",
                       [(set NZCV, (A64setcc (and ty:$Rn, (sra ty:$Rm,
                           !cast<Operand>("asr_operand_" # ty):$Imm6)),
                                          0, signed_cond))],
                       NoItinerary>;

  def _ror : A64I_logicalshift<sf, 0b11, 0b11, 0b0,
                       (outs),
                       (ins GPR:$Rn, GPR:$Rm,
                            !cast<Operand>("ror_operand_" # ty):$Imm6),
                       "tst\t$Rn, $Rm, $Imm6",
                       [(set NZCV, (A64setcc (and ty:$Rn, (rotr ty:$Rm,
                           !cast<Operand>("ror_operand_" # ty):$Imm6)),
                                          0, signed_cond))],
                       NoItinerary>;
  }

  def _noshift : InstAlias<"tst $Rn, $Rm",
                     (!cast<Instruction>(prefix # "_lsl") GPR:$Rn, GPR:$Rm, 0)>;

  def : Pat<(A64setcc (and ty:$Rn, ty:$Rm), 0, signed_cond),
            (!cast<Instruction>(prefix # "_lsl") $Rn, $Rm, 0)>;
}

defm TSTxx : tst_shifts<"TSTxx", 0b1, i64, GPR64>;
defm TSTww : tst_shifts<"TSTww", 0b0, i32, GPR32>;


multiclass mvn_shifts<string prefix, bit sf, ValueType ty, RegisterClass GPR> {
  let isCommutable = 0, Rn = 0b11111 in {
  def _lsl : A64I_logicalshift<sf, 0b01, 0b00, 0b1,
                       (outs GPR:$Rd),
                       (ins GPR:$Rm,
                            !cast<Operand>("lsl_operand_" # ty):$Imm6),
                       "mvn\t$Rd, $Rm, $Imm6",
                       [(set ty:$Rd, (not (shl ty:$Rm,
                         !cast<Operand>("lsl_operand_" # ty):$Imm6)))],
                       NoItinerary>;


  def _lsr : A64I_logicalshift<sf, 0b01, 0b01, 0b1,
                       (outs GPR:$Rd),
                       (ins GPR:$Rm,
                            !cast<Operand>("lsr_operand_" # ty):$Imm6),
                       "mvn\t$Rd, $Rm, $Imm6",
                       [(set ty:$Rd, (not (srl ty:$Rm,
                         !cast<Operand>("lsr_operand_" # ty):$Imm6)))],
                       NoItinerary>;

  def _asr : A64I_logicalshift<sf, 0b01, 0b10, 0b1,
                       (outs GPR:$Rd),
                       (ins GPR:$Rm,
                            !cast<Operand>("asr_operand_" # ty):$Imm6),
                       "mvn\t$Rd, $Rm, $Imm6",
                       [(set ty:$Rd, (not (sra ty:$Rm,
                         !cast<Operand>("asr_operand_" # ty):$Imm6)))],
                       NoItinerary>;

  def _ror : A64I_logicalshift<sf, 0b01, 0b11, 0b1,
                       (outs GPR:$Rd),
                       (ins GPR:$Rm,
                            !cast<Operand>("ror_operand_" # ty):$Imm6),
                       "mvn\t$Rd, $Rm, $Imm6",
                       [(set ty:$Rd, (not (rotr ty:$Rm,
                         !cast<Operand>("lsl_operand_" # ty):$Imm6)))],
                       NoItinerary>;
  }

  def _noshift : InstAlias<"mvn $Rn, $Rm",
                     (!cast<Instruction>(prefix # "_lsl") GPR:$Rn, GPR:$Rm, 0)>;

  def : Pat<(not ty:$Rm),
            (!cast<Instruction>(prefix # "_lsl") $Rm, 0)>;
}

defm MVNxx : mvn_shifts<"MVNxx", 0b1, i64, GPR64>;
defm MVNww : mvn_shifts<"MVNww", 0b0, i32, GPR32>;

def MOVxx :InstAlias<"mov $Rd, $Rm", (ORRxxx_lsl GPR64:$Rd, XZR, GPR64:$Rm, 0)>;
def MOVww :InstAlias<"mov $Rd, $Rm", (ORRwww_lsl GPR32:$Rd, WZR, GPR32:$Rm, 0)>;

//===----------------------------------------------------------------------===//
// Move wide (immediate) instructions
//===----------------------------------------------------------------------===//
// Contains: MOVN, MOVZ, MOVK + MOV aliases

// A wide variety of different relocations are needed for variants of these
// instructions, so it turns out that we need a different operand for all of
// them.
multiclass movw_operands<string prefix, string instname, int width> {
  def _imm_asmoperand : AsmOperandClass {
    let Name = instname # width # "Shifted" # shift;
    let PredicateMethod = "is" # instname # width # "Imm";
    let RenderMethod = "addMoveWideImmOperands";
    let ParserMethod = "ParseImmWithLSLOperand";
    let DiagnosticType = "MOVWUImm16";
  }

  def _imm : Operand<i64> {
    let ParserMatchClass = !cast<AsmOperandClass>(prefix # "_imm_asmoperand");
    let PrintMethod = "printMoveWideImmOperand";
    let EncoderMethod = "getMoveWideImmOpValue";
    let DecoderMethod = "DecodeMoveWideImmOperand<" # width # ">";

    let MIOperandInfo = (ops uimm16:$UImm16, imm:$Shift);
  }
}

defm movn32 : movw_operands<"movn32", "MOVN", 32>;
defm movn64 : movw_operands<"movn64", "MOVN", 64>;
defm movz32 : movw_operands<"movz32", "MOVZ", 32>;
defm movz64 : movw_operands<"movz64", "MOVZ", 64>;
defm movk32 : movw_operands<"movk32", "MOVK", 32>;
defm movk64 : movw_operands<"movk64", "MOVK", 64>;

multiclass A64I_movwSizes<bits<2> opc, string asmop, dag ins32bit,
                          dag ins64bit> {

  def wii : A64I_movw<0b0, opc, (outs GPR32:$Rd), ins32bit,
                      !strconcat(asmop, "\t$Rd, $FullImm"),
                      [], NoItinerary> {
    bits<18> FullImm;
    let UImm16 = FullImm{15-0};
    let Shift = FullImm{17-16};
  }

  def xii : A64I_movw<0b1, opc, (outs GPR64:$Rd), ins64bit,
                      !strconcat(asmop, "\t$Rd, $FullImm"),
                      [], NoItinerary> {
    bits<18> FullImm;
    let UImm16 = FullImm{15-0};
    let Shift = FullImm{17-16};
  }
}

let isMoveImm = 1, isReMaterializable = 1,
    isAsCheapAsAMove = 1, hasSideEffects = 0 in {
  defm MOVN : A64I_movwSizes<0b00, "movn",
                             (ins movn32_imm:$FullImm),
                             (ins movn64_imm:$FullImm)>;

  // Some relocations are able to convert between a MOVZ and a MOVN. If these
  // are applied the instruction must be emitted with the corresponding bits as
  // 0, which means a MOVZ needs to override that bit from the default.
  let PostEncoderMethod = "fixMOVZ" in
  defm MOVZ : A64I_movwSizes<0b10, "movz",
                             (ins movz32_imm:$FullImm),
                             (ins movz64_imm:$FullImm)>;
}

let Constraints = "$src = $Rd" in
defm MOVK : A64I_movwSizes<0b11, "movk",
                           (ins GPR32:$src, movk32_imm:$FullImm),
                           (ins GPR64:$src, movk64_imm:$FullImm)>;


// And now the "MOV" aliases. These also need their own operands because what
// they accept is completely different to what the base instructions accept.
multiclass movalias_operand<string prefix, string basename,
                            string immpredicate, int width> {
  def _asmoperand : AsmOperandClass {
    let Name = basename # width # "MovAlias";
    let PredicateMethod
          = "isMoveWideMovAlias<" # width # ", A64Imms::" # immpredicate # ">";
    let RenderMethod
      = "addMoveWideMovAliasOperands<" # width # ", "
                                       # "A64Imms::" # immpredicate # ">";
  }

  def _movimm : Operand<i64> {
    let ParserMatchClass = !cast<AsmOperandClass>(prefix # "_asmoperand");

    let MIOperandInfo = (ops uimm16:$UImm16, imm:$Shift);
  }
}

defm movz32 : movalias_operand<"movz32", "MOVZ", "isMOVZImm", 32>;
defm movz64 : movalias_operand<"movz64", "MOVZ", "isMOVZImm", 64>;
defm movn32 : movalias_operand<"movn32", "MOVN", "isOnlyMOVNImm", 32>;
defm movn64 : movalias_operand<"movn64", "MOVN", "isOnlyMOVNImm", 64>;

// FIXME: these are officially canonical aliases, but TableGen is too limited to
// print them at the moment. I believe in this case an "AliasPredicate" method
// will need to be implemented. to allow it, as well as the more generally
// useful handling of non-register, non-constant operands.
class movalias<Instruction INST, RegisterClass GPR, Operand operand>
  : InstAlias<"mov $Rd, $FullImm", (INST GPR:$Rd, operand:$FullImm)>;

def : movalias<MOVZwii, GPR32, movz32_movimm>;
def : movalias<MOVZxii, GPR64, movz64_movimm>;
def : movalias<MOVNwii, GPR32, movn32_movimm>;
def : movalias<MOVNxii, GPR64, movn64_movimm>;

def movw_addressref_g0 : ComplexPattern<i64, 2, "SelectMOVWAddressRef<0>">;
def movw_addressref_g1 : ComplexPattern<i64, 2, "SelectMOVWAddressRef<1>">;
def movw_addressref_g2 : ComplexPattern<i64, 2, "SelectMOVWAddressRef<2>">;
def movw_addressref_g3 : ComplexPattern<i64, 2, "SelectMOVWAddressRef<3>">;

def : Pat<(A64WrapperLarge movw_addressref_g3:$G3, movw_addressref_g2:$G2,
                           movw_addressref_g1:$G1, movw_addressref_g0:$G0),
          (MOVKxii (MOVKxii (MOVKxii (MOVZxii movw_addressref_g3:$G3),
                                     movw_addressref_g2:$G2),
                            movw_addressref_g1:$G1),
                   movw_addressref_g0:$G0)>;

//===----------------------------------------------------------------------===//
// PC-relative addressing instructions
//===----------------------------------------------------------------------===//
// Contains: ADR, ADRP

def adr_label : Operand<i64> {
  let EncoderMethod = "getLabelOpValue<AArch64::fixup_a64_adr_prel>";

  // This label is a 21-bit offset from PC, unscaled
  let PrintMethod = "printLabelOperand<21, 1>";
  let ParserMatchClass = label_asmoperand<21, 1>;
  let OperandType = "OPERAND_PCREL";
}

def adrp_label_asmoperand : AsmOperandClass {
  let Name = "AdrpLabel";
  let RenderMethod = "addLabelOperands<21, 4096>";
  let DiagnosticType = "Label";
}

def adrp_label : Operand<i64> {
  let EncoderMethod = "getAdrpLabelOpValue";

  // This label is a 21-bit offset from PC, scaled by the page-size: 4096.
  let PrintMethod = "printLabelOperand<21, 4096>";
  let ParserMatchClass = adrp_label_asmoperand;
  let OperandType = "OPERAND_PCREL";
}

let hasSideEffects = 0 in {
  def ADRxi : A64I_PCADR<0b0, (outs GPR64:$Rd), (ins adr_label:$Label),
                         "adr\t$Rd, $Label", [], NoItinerary>;

  def ADRPxi : A64I_PCADR<0b1, (outs GPR64:$Rd), (ins adrp_label:$Label),
                          "adrp\t$Rd, $Label", [], NoItinerary>;
}

//===----------------------------------------------------------------------===//
// System instructions
//===----------------------------------------------------------------------===//
// Contains: HINT, CLREX, DSB, DMB, ISB, MSR, SYS, SYSL, MRS
//    + aliases IC, DC, AT, TLBI, NOP, YIELD, WFE, WFI, SEV, SEVL

// Op1 and Op2 fields are sometimes simple 3-bit unsigned immediate values.
def uimm3_asmoperand : AsmOperandClass {
  let Name = "UImm3";
  let PredicateMethod = "isUImm<3>";
  let RenderMethod = "addImmOperands";
  let DiagnosticType = "UImm3";
}

def uimm3 : Operand<i32> {
  let ParserMatchClass = uimm3_asmoperand;
}

// The HINT alias can accept a simple unsigned 7-bit immediate.
def uimm7_asmoperand : AsmOperandClass {
  let Name = "UImm7";
  let PredicateMethod = "isUImm<7>";
  let RenderMethod = "addImmOperands";
  let DiagnosticType = "UImm7";
}

def uimm7 : Operand<i32> {
  let ParserMatchClass = uimm7_asmoperand;
}

// Multiclass namedimm is defined with the prefetch operands. Most of these fit
// into the NamedImmMapper scheme well: they either accept a named operand or
// any immediate under a particular value (which may be 0, implying no immediate
// is allowed).
defm dbarrier : namedimm<"dbarrier", "A64DB::DBarrierMapper">;
defm isb : namedimm<"isb", "A64ISB::ISBMapper">;
defm ic : namedimm<"ic", "A64IC::ICMapper">;
defm dc : namedimm<"dc", "A64DC::DCMapper">;
defm at : namedimm<"at", "A64AT::ATMapper">;
defm tlbi : namedimm<"tlbi", "A64TLBI::TLBIMapper">;

// However, MRS and MSR are more complicated for a few reasons:
//   * There are ~1000 generic names S3_<op1>_<CRn>_<CRm>_<Op2> which have an
//     implementation-defined effect
//   * Most registers are shared, but some are read-only or write-only.
//   * There is a variant of MSR which accepts the same register name (SPSel),
//     but which would have a different encoding.

// In principle these could be resolved in with more complicated subclasses of
// NamedImmMapper, however that imposes an overhead on other "named
// immediates". Both in concrete terms with virtual tables and in unnecessary
// abstraction.

// The solution adopted here is to take the MRS/MSR Mappers out of the usual
// hierarchy (they're not derived from NamedImmMapper) and to add logic for
// their special situation.
def mrs_asmoperand : AsmOperandClass {
  let Name = "MRS";
  let ParserMethod = "ParseSysRegOperand";
  let DiagnosticType = "MRS";
}

def mrs_op : Operand<i32> {
  let ParserMatchClass = mrs_asmoperand;
  let PrintMethod = "printMRSOperand";
  let DecoderMethod = "DecodeMRSOperand";
}

def msr_asmoperand : AsmOperandClass {
  let Name = "MSRWithReg";

  // Note that SPSel is valid for both this and the pstate operands, but with
  // different immediate encodings. This is why these operands provide a string
  // AArch64Operand rather than an immediate. The overlap is small enough that
  // it could be resolved with hackery now, but who can say in future?
  let ParserMethod = "ParseSysRegOperand";
  let DiagnosticType = "MSR";
}

def msr_op : Operand<i32> {
  let ParserMatchClass = msr_asmoperand;
  let PrintMethod = "printMSROperand";
  let DecoderMethod = "DecodeMSROperand";
}

def pstate_asmoperand : AsmOperandClass {
  let Name = "MSRPState";
  // See comment above about parser.
  let ParserMethod = "ParseSysRegOperand";
  let DiagnosticType = "MSR";
}

def pstate_op : Operand<i32> {
  let ParserMatchClass = pstate_asmoperand;
  let PrintMethod = "printNamedImmOperand<A64PState::PStateMapper>";
  let DecoderMethod = "DecodeNamedImmOperand<A64PState::PStateMapper>";
}

// When <CRn> is specified, an assembler should accept something like "C4", not
// the usual "#4" immediate.
def CRx_asmoperand : AsmOperandClass {
  let Name = "CRx";
  let PredicateMethod = "isUImm<4>";
  let RenderMethod = "addImmOperands";
  let ParserMethod = "ParseCRxOperand";
  // Diagnostics are handled in all cases by ParseCRxOperand.
}

def CRx : Operand<i32> {
  let ParserMatchClass = CRx_asmoperand;
  let PrintMethod = "printCRxOperand";
}


// Finally, we can start defining the instructions.

// HINT is straightforward, with a few aliases.
def HINTi : A64I_system<0b0, (outs), (ins uimm7:$UImm7), "hint\t$UImm7",
                        [], NoItinerary> {
  bits<7> UImm7;
  let CRm = UImm7{6-3};
  let Op2 = UImm7{2-0};

  let Op0 = 0b00;
  let Op1 = 0b011;
  let CRn = 0b0010;
  let Rt = 0b11111;
}

def : InstAlias<"nop", (HINTi 0)>;
def : InstAlias<"yield", (HINTi 1)>;
def : InstAlias<"wfe", (HINTi 2)>;
def : InstAlias<"wfi", (HINTi 3)>;
def : InstAlias<"sev", (HINTi 4)>;
def : InstAlias<"sevl", (HINTi 5)>;

// Quite a few instructions then follow a similar pattern of fixing common
// fields in the bitpattern, we'll define a helper-class for them.
class simple_sys<bits<2> op0, bits<3> op1, bits<4> crn, bits<3> op2,
                 Operand operand, string asmop>
  : A64I_system<0b0, (outs), (ins operand:$CRm), !strconcat(asmop, "\t$CRm"),
                [], NoItinerary> {
  let Op0 = op0;
  let Op1 = op1;
  let CRn = crn;
  let Op2 = op2;
  let Rt = 0b11111;
}


def CLREXi : simple_sys<0b00, 0b011, 0b0011, 0b010, uimm4, "clrex">;
def DSBi : simple_sys<0b00, 0b011, 0b0011, 0b100, dbarrier_op, "dsb">;
def DMBi : simple_sys<0b00, 0b011, 0b0011, 0b101, dbarrier_op, "dmb">;
def ISBi : simple_sys<0b00, 0b011, 0b0011, 0b110, isb_op, "isb">;

def : InstAlias<"clrex", (CLREXi 0b1111)>;
def : InstAlias<"isb", (ISBi 0b1111)>;

// (DMBi 0xb) is a "DMB ISH" instruciton, appropriate for Linux SMP
// configurations at least.
def : Pat<(atomic_fence imm, imm), (DMBi 0xb)>;

// Any SYS bitpattern can be represented with a complex and opaque "SYS"
// instruction.
def SYSiccix : A64I_system<0b0, (outs),
                           (ins uimm3:$Op1, CRx:$CRn, CRx:$CRm,
                                uimm3:$Op2, GPR64:$Rt),
                           "sys\t$Op1, $CRn, $CRm, $Op2, $Rt",
                           [], NoItinerary> {
  let Op0 = 0b01;
}

// You can skip the Xt argument whether it makes sense or not for the generic
// SYS instruction.
def : InstAlias<"sys $Op1, $CRn, $CRm, $Op2",
                (SYSiccix uimm3:$Op1, CRx:$CRn, CRx:$CRm, uimm3:$Op2, XZR)>;


// But many have aliases, which obviously don't fit into
class SYSalias<dag ins, string asmstring>
  : A64I_system<0b0, (outs), ins, asmstring, [], NoItinerary> {
  let isAsmParserOnly = 1;

  bits<14> SysOp;
  let Op0 = 0b01;
  let Op1 = SysOp{13-11};
  let CRn = SysOp{10-7};
  let CRm = SysOp{6-3};
  let Op2 = SysOp{2-0};
}

def ICix : SYSalias<(ins ic_op:$SysOp, GPR64:$Rt), "ic\t$SysOp, $Rt">;

def ICi : SYSalias<(ins ic_op:$SysOp), "ic\t$SysOp"> {
  let Rt = 0b11111;
}

def DCix : SYSalias<(ins dc_op:$SysOp, GPR64:$Rt), "dc\t$SysOp, $Rt">;
def ATix : SYSalias<(ins at_op:$SysOp, GPR64:$Rt), "at\t$SysOp, $Rt">;

def TLBIix : SYSalias<(ins tlbi_op:$SysOp, GPR64:$Rt), "tlbi\t$SysOp, $Rt">;

def TLBIi : SYSalias<(ins tlbi_op:$SysOp), "tlbi\t$SysOp"> {
  let Rt = 0b11111;
}


def SYSLxicci : A64I_system<0b1, (outs GPR64:$Rt),
                            (ins uimm3:$Op1, CRx:$CRn, CRx:$CRm, uimm3:$Op2),
                            "sysl\t$Rt, $Op1, $CRn, $CRm, $Op2",
                            [], NoItinerary> {
  let Op0 = 0b01;
}

// The instructions themselves are rather simple for MSR and MRS.
def MSRix : A64I_system<0b0, (outs), (ins msr_op:$SysReg, GPR64:$Rt),
                        "msr\t$SysReg, $Rt", [], NoItinerary> {
  bits<16> SysReg;
  let Op0 = SysReg{15-14};
  let Op1 = SysReg{13-11};
  let CRn = SysReg{10-7};
  let CRm = SysReg{6-3};
  let Op2 = SysReg{2-0};
}

def MRSxi : A64I_system<0b1, (outs GPR64:$Rt), (ins mrs_op:$SysReg),
                        "mrs\t$Rt, $SysReg", [], NoItinerary> {
  bits<16> SysReg;
  let Op0 = SysReg{15-14};
  let Op1 = SysReg{13-11};
  let CRn = SysReg{10-7};
  let CRm = SysReg{6-3};
  let Op2 = SysReg{2-0};
}

def MSRii : A64I_system<0b0, (outs), (ins pstate_op:$PState, uimm4:$CRm),
                        "msr\t$PState, $CRm", [], NoItinerary> {
  bits<6> PState;

  let Op0 = 0b00;
  let Op1 = PState{5-3};
  let CRn = 0b0100;
  let Op2 = PState{2-0};
  let Rt = 0b11111;
}

//===----------------------------------------------------------------------===//
// Test & branch (immediate) instructions
//===----------------------------------------------------------------------===//
// Contains: TBZ, TBNZ

// The bit to test is a simple unsigned 6-bit immediate in the X-register
// versions.
def uimm6 : Operand<i64> {
  let ParserMatchClass = uimm6_asmoperand;
}

def label_wid14_scal4_asmoperand : label_asmoperand<14, 4>;

def tbimm_target : Operand<OtherVT> {
  let EncoderMethod = "getLabelOpValue<AArch64::fixup_a64_tstbr>";

  // This label is a 14-bit offset from PC, scaled by the instruction-width: 4.
  let PrintMethod = "printLabelOperand<14, 4>";
  let ParserMatchClass = label_wid14_scal4_asmoperand;

  let OperandType = "OPERAND_PCREL";
}

def A64eq : ImmLeaf<i32, [{ return Imm == A64CC::EQ; }]>;
def A64ne : ImmLeaf<i32, [{ return Imm == A64CC::NE; }]>;

// These instructions correspond to patterns involving "and" with a power of
// two, which we need to be able to select.
def tstb64_pat : ComplexPattern<i64, 1, "SelectTSTBOperand<64>">;
def tstb32_pat : ComplexPattern<i32, 1, "SelectTSTBOperand<32>">;

let isBranch = 1, isTerminator = 1 in {
  def TBZxii : A64I_TBimm<0b0, (outs),
                        (ins GPR64:$Rt, uimm6:$Imm, tbimm_target:$Label),
                        "tbz\t$Rt, $Imm, $Label",
                        [(A64br_cc (A64cmp (and i64:$Rt, tstb64_pat:$Imm), 0),
                                   A64eq, bb:$Label)],
                        NoItinerary>;

  def TBNZxii : A64I_TBimm<0b1, (outs),
                        (ins GPR64:$Rt, uimm6:$Imm, tbimm_target:$Label),
                        "tbnz\t$Rt, $Imm, $Label",
                        [(A64br_cc (A64cmp (and i64:$Rt, tstb64_pat:$Imm), 0),
                                   A64ne, bb:$Label)],
                        NoItinerary>;


  // Note, these instructions overlap with the above 64-bit patterns. This is
  // intentional, "tbz x3, #1, somewhere" and "tbz w3, #1, somewhere" would both
  // do the same thing and are both permitted assembly. They also both have
  // sensible DAG patterns.
  def TBZwii : A64I_TBimm<0b0, (outs),
                        (ins GPR32:$Rt, uimm5:$Imm, tbimm_target:$Label),
                        "tbz\t$Rt, $Imm, $Label",
                        [(A64br_cc (A64cmp (and i32:$Rt, tstb32_pat:$Imm), 0),
                                   A64eq, bb:$Label)],
                        NoItinerary> {
    let Imm{5} = 0b0;
  }

  def TBNZwii : A64I_TBimm<0b1, (outs),
                        (ins GPR32:$Rt, uimm5:$Imm, tbimm_target:$Label),
                        "tbnz\t$Rt, $Imm, $Label",
                        [(A64br_cc (A64cmp (and i32:$Rt, tstb32_pat:$Imm), 0),
                                   A64ne, bb:$Label)],
                        NoItinerary> {
    let Imm{5} = 0b0;
  }
}

//===----------------------------------------------------------------------===//
// Unconditional branch (immediate) instructions
//===----------------------------------------------------------------------===//
// Contains: B, BL

def label_wid26_scal4_asmoperand : label_asmoperand<26, 4>;

def bimm_target : Operand<OtherVT> {
  let EncoderMethod = "getLabelOpValue<AArch64::fixup_a64_uncondbr>";

  // This label is a 26-bit offset from PC, scaled by the instruction-width: 4.
  let PrintMethod = "printLabelOperand<26, 4>";
  let ParserMatchClass = label_wid26_scal4_asmoperand;

  let OperandType = "OPERAND_PCREL";
}

def blimm_target : Operand<i64> {
  let EncoderMethod = "getLabelOpValue<AArch64::fixup_a64_call>";

  // This label is a 26-bit offset from PC, scaled by the instruction-width: 4.
  let PrintMethod = "printLabelOperand<26, 4>";
  let ParserMatchClass = label_wid26_scal4_asmoperand;

  let OperandType = "OPERAND_PCREL";
}

class A64I_BimmImpl<bit op, string asmop, list<dag> patterns, Operand lbl_type>
  : A64I_Bimm<op, (outs), (ins lbl_type:$Label),
              !strconcat(asmop, "\t$Label"), patterns,
              NoItinerary>;

let isBranch = 1 in {
  def Bimm : A64I_BimmImpl<0b0, "b", [(br bb:$Label)], bimm_target> {
    let isTerminator = 1;
    let isBarrier = 1;
  }

  def BLimm : A64I_BimmImpl<0b1, "bl",
                            [(AArch64Call tglobaladdr:$Label)], blimm_target> {
    let isCall = 1;
    let Defs = [X30];
  }
}

def : Pat<(AArch64Call texternalsym:$Label), (BLimm texternalsym:$Label)>;

//===----------------------------------------------------------------------===//
// Unconditional branch (register) instructions
//===----------------------------------------------------------------------===//
// Contains: BR, BLR, RET, ERET, DRP.

// Most of the notional opcode fields in the A64I_Breg format are fixed in A64
// at the moment.
class A64I_BregImpl<bits<4> opc,
                    dag outs, dag ins, string asmstr, list<dag> patterns,
                    InstrItinClass itin = NoItinerary>
  : A64I_Breg<opc, 0b11111, 0b000000, 0b00000,
              outs, ins, asmstr, patterns, itin> {
  let isBranch         = 1;
  let isIndirectBranch = 1;
}

// Note that these are not marked isCall or isReturn because as far as LLVM is
// concerned they're not. "ret" is just another jump unless it has been selected
// by LLVM as the function's return.

let isBranch = 1 in {
  def BRx : A64I_BregImpl<0b0000,(outs), (ins GPR64:$Rn),
                          "br\t$Rn", [(brind i64:$Rn)]> {
    let isBarrier = 1;
    let isTerminator = 1;
  }

  def BLRx : A64I_BregImpl<0b0001, (outs), (ins GPR64:$Rn),
                           "blr\t$Rn", [(AArch64Call i64:$Rn)]> {
    let isBarrier = 0;
    let isCall = 1;
    let Defs = [X30];
  }

  def RETx : A64I_BregImpl<0b0010, (outs), (ins GPR64:$Rn),
                           "ret\t$Rn", []> {
    let isBarrier = 1;
    let isTerminator = 1;
    let isReturn = 1;
  }

  // Create a separate pseudo-instruction for codegen to use so that we don't
  // flag x30 as used in every function. It'll be restored before the RET by the
  // epilogue if it's legitimately used.
  def RET : A64PseudoExpand<(outs), (ins), [(A64ret)], (RETx (ops X30))> {
    let isTerminator = 1;
    let isBarrier = 1;
    let isReturn = 1;
  }

  def ERET : A64I_BregImpl<0b0100, (outs), (ins), "eret", []> {
    let Rn = 0b11111;
    let isBarrier = 1;
    let isTerminator = 1;
    let isReturn = 1;
  }

  def DRPS : A64I_BregImpl<0b0101, (outs), (ins), "drps", []> {
    let Rn = 0b11111;
    let isBarrier = 1;
  }
}

def RETAlias : InstAlias<"ret", (RETx X30)>;


//===----------------------------------------------------------------------===//
// Address generation patterns
//===----------------------------------------------------------------------===//

// Primary method of address generation for the small/absolute memory model is
// an ADRP/ADR pair:
//     ADRP x0, some_variable
//     ADD x0, x0, #:lo12:some_variable
//
// The load/store elision of the ADD is accomplished when selecting
// addressing-modes. This just mops up the cases where that doesn't work and we
// really need an address in some register.

// This wrapper applies a LO12 modifier to the address. Otherwise we could just
// use the same address.

class ADRP_ADD<SDNode Wrapper, SDNode addrop>
 : Pat<(Wrapper addrop:$Hi, addrop:$Lo12, (i32 imm)),
       (ADDxxi_lsl0_s (ADRPxi addrop:$Hi), addrop:$Lo12)>;

def : ADRP_ADD<A64WrapperSmall, tblockaddress>;
def : ADRP_ADD<A64WrapperSmall, texternalsym>;
def : ADRP_ADD<A64WrapperSmall, tglobaladdr>;
def : ADRP_ADD<A64WrapperSmall, tglobaltlsaddr>;
def : ADRP_ADD<A64WrapperSmall, tjumptable>;

//===----------------------------------------------------------------------===//
// GOT access patterns
//===----------------------------------------------------------------------===//

class GOTLoadSmall<SDNode addrfrag>
  : Pat<(A64GOTLoad (A64WrapperSmall addrfrag:$Hi, addrfrag:$Lo12, 8)),
        (LS64_LDR (ADRPxi addrfrag:$Hi), addrfrag:$Lo12)>;

def : GOTLoadSmall<texternalsym>;
def : GOTLoadSmall<tglobaladdr>;
def : GOTLoadSmall<tglobaltlsaddr>;

//===----------------------------------------------------------------------===//
// Tail call handling
//===----------------------------------------------------------------------===//

let isCall = 1, isTerminator = 1, isReturn = 1, isBarrier = 1, Uses = [XSP] in {
  def TC_RETURNdi
    : PseudoInst<(outs), (ins i64imm:$dst, i32imm:$FPDiff),
                 [(AArch64tcret tglobaladdr:$dst, (i32 timm:$FPDiff))]>;

  def TC_RETURNxi
    : PseudoInst<(outs), (ins tcGPR64:$dst, i32imm:$FPDiff),
                 [(AArch64tcret i64:$dst, (i32 timm:$FPDiff))]>;
}

let isCall = 1, isTerminator = 1, isReturn = 1, isBarrier = 1,
    Uses = [XSP] in {
  def TAIL_Bimm : A64PseudoExpand<(outs), (ins bimm_target:$Label), [],
                                  (Bimm bimm_target:$Label)>;

  def TAIL_BRx : A64PseudoExpand<(outs), (ins tcGPR64:$Rd), [],
                                 (BRx GPR64:$Rd)>;
}


def : Pat<(AArch64tcret texternalsym:$dst, (i32 timm:$FPDiff)),
          (TC_RETURNdi texternalsym:$dst, imm:$FPDiff)>;

//===----------------------------------------------------------------------===//
// Thread local storage
//===----------------------------------------------------------------------===//

// This is a pseudo-instruction representing the ".tlsdesccall" directive in
// assembly. Its effect is to insert an R_AARCH64_TLSDESC_CALL relocation at the
// current location. It should always be immediately followed by a BLR
// instruction, and is intended solely for relaxation by the linker.

def : Pat<(A64threadpointer), (MRSxi 0xde82)>;

def TLSDESCCALL : PseudoInst<(outs), (ins i64imm:$Lbl), []> {
  let hasSideEffects = 1;
}

def TLSDESC_BLRx : PseudoInst<(outs), (ins GPR64:$Rn, i64imm:$Var),
                            [(A64tlsdesc_blr i64:$Rn, tglobaltlsaddr:$Var)]> {
  let isCall = 1;
  let Defs = [X30];
}

def : Pat<(A64tlsdesc_blr i64:$Rn, texternalsym:$Var),
          (TLSDESC_BLRx $Rn, texternalsym:$Var)>;

//===----------------------------------------------------------------------===//
// Bitfield patterns
//===----------------------------------------------------------------------===//

def bfi32_lsb_to_immr : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant((32 - N->getZExtValue()) % 32, MVT::i64);
}]>;

def bfi64_lsb_to_immr : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant((64 - N->getZExtValue()) % 64, MVT::i64);
}]>;

def bfi_width_to_imms : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getZExtValue() - 1, MVT::i64);
}]>;


// The simpler patterns deal with cases where no AND mask is actually needed
// (either all bits are used or the low 32 bits are used).
let AddedComplexity = 10 in {

def : Pat<(A64Bfi i64:$src, i64:$Rn, imm:$ImmR, imm:$ImmS),
           (BFIxxii $src, $Rn,
                    (bfi64_lsb_to_immr (i64 imm:$ImmR)),
                    (bfi_width_to_imms (i64 imm:$ImmS)))>;

def : Pat<(A64Bfi i32:$src, i32:$Rn, imm:$ImmR, imm:$ImmS),
          (BFIwwii $src, $Rn,
                   (bfi32_lsb_to_immr (i64 imm:$ImmR)),
                   (bfi_width_to_imms (i64 imm:$ImmS)))>;


def : Pat<(and (A64Bfi i64:$src, i64:$Rn, imm:$ImmR, imm:$ImmS),
               (i64 4294967295)),
          (SUBREG_TO_REG (i64 0),
                         (BFIwwii (EXTRACT_SUBREG $src, sub_32),
                                  (EXTRACT_SUBREG $Rn, sub_32),
                                  (bfi32_lsb_to_immr (i64 imm:$ImmR)),
                                  (bfi_width_to_imms (i64 imm:$ImmS))),
                         sub_32)>;

}

//===----------------------------------------------------------------------===//
// Miscellaneous patterns
//===----------------------------------------------------------------------===//

// Truncation from 64 to 32-bits just involves renaming your register.
def : Pat<(i32 (trunc i64:$val)), (EXTRACT_SUBREG $val, sub_32)>;

// Similarly, extension where we don't care about the high bits is
// just a rename.
def : Pat<(i64 (anyext i32:$val)),
          (INSERT_SUBREG (IMPLICIT_DEF), $val, sub_32)>;

// SELECT instructions providing f128 types need to be handled by a
// pseudo-instruction since the eventual code will need to introduce basic
// blocks and control flow.
def F128CSEL : PseudoInst<(outs FPR128:$Rd),
                         (ins FPR128:$Rn, FPR128:$Rm, cond_code_op:$Cond),
                         [(set f128:$Rd, (simple_select f128:$Rn, f128:$Rm))]> {
  let Uses = [NZCV];
  let usesCustomInserter = 1;
}

//===----------------------------------------------------------------------===//
// Load/store patterns
//===----------------------------------------------------------------------===//

// There are lots of patterns here, because we need to allow at least three
// parameters to vary independently.
//   1. Instruction: "ldrb w9, [sp]", "ldrh w9, [sp]", ...
//   2. LLVM source: zextloadi8, anyextloadi8, ...
//   3. Address-generation: A64Wrapper, (add BASE, OFFSET), ...
//
// The biggest problem turns out to be the address-generation variable. At the
// point of instantiation we need to produce two DAGs, one for the pattern and
// one for the instruction. Doing this at the lowest level of classes doesn't
// work.
//
// Consider the simple uimm12 addressing mode, and the desire to match both (add
// GPR64xsp:$Rn, uimm12:$Offset) and GPR64xsp:$Rn, particularly on the
// instruction side. We'd need to insert either "GPR64xsp" and "uimm12" or
// "GPR64xsp" and "0" into an unknown dag. !subst is not capable of this
// operation, and PatFrags are for selection not output.
//
// As a result, the address-generation patterns are the final
// instantiations. However, we do still need to vary the operand for the address
// further down (At the point we're deciding A64WrapperSmall, we don't know
// the memory width of the operation).

//===------------------------------
// 1. Basic infrastructural defs
//===------------------------------

// First, some simple classes for !foreach and !subst to use:
class Decls {
  dag pattern;
}

def decls : Decls;
def ALIGN;
def INST;
def OFFSET;
def SHIFT;

// You can't use !subst on an actual immediate, but you *can* use it on an
// operand record that happens to match a single immediate. So we do.
def imm_eq0 : ImmLeaf<i64, [{ return Imm == 0; }]>;
def imm_eq1 : ImmLeaf<i64, [{ return Imm == 1; }]>;
def imm_eq2 : ImmLeaf<i64, [{ return Imm == 2; }]>;
def imm_eq3 : ImmLeaf<i64, [{ return Imm == 3; }]>;
def imm_eq4 : ImmLeaf<i64, [{ return Imm == 4; }]>;

// If the low bits of a pointer are known to be 0 then an "or" is just as good
// as addition for computing an offset. This fragment forwards that check for
// TableGen's use.
def add_like_or : PatFrag<(ops node:$lhs, node:$rhs), (or node:$lhs, node:$rhs),
[{
  return CurDAG->isBaseWithConstantOffset(SDValue(N, 0));
}]>;

// Load/store (unsigned immediate) operations with relocations against global
// symbols (for lo12) are only valid if those symbols have correct alignment
// (since the immediate offset is divided by the access scale, it can't have a
// remainder).
//
// The guaranteed alignment is provided as part of the WrapperSmall
// operation, and checked against one of these.
def any_align   : ImmLeaf<i32, [{ (void)Imm; return true; }]>;
def min_align2  : ImmLeaf<i32, [{ return Imm >= 2; }]>;
def min_align4  : ImmLeaf<i32, [{ return Imm >= 4; }]>;
def min_align8  : ImmLeaf<i32, [{ return Imm >= 8; }]>;
def min_align16 : ImmLeaf<i32, [{ return Imm >= 16; }]>;

// "Normal" load/store instructions can be used on atomic operations, provided
// the ordering parameter is at most "monotonic". Anything above that needs
// special handling with acquire/release instructions.
class simple_load<PatFrag base>
  : PatFrag<(ops node:$ptr), (base node:$ptr), [{
  return cast<AtomicSDNode>(N)->getOrdering() <= Monotonic;
}]>;

def atomic_load_simple_i8  : simple_load<atomic_load_8>;
def atomic_load_simple_i16 : simple_load<atomic_load_16>;
def atomic_load_simple_i32 : simple_load<atomic_load_32>;
def atomic_load_simple_i64 : simple_load<atomic_load_64>;

class simple_store<PatFrag base>
  : PatFrag<(ops node:$ptr, node:$val), (base node:$ptr, node:$val), [{
  return cast<AtomicSDNode>(N)->getOrdering() <= Monotonic;
}]>;

def atomic_store_simple_i8  : simple_store<atomic_store_8>;
def atomic_store_simple_i16 : simple_store<atomic_store_16>;
def atomic_store_simple_i32 : simple_store<atomic_store_32>;
def atomic_store_simple_i64 : simple_store<atomic_store_64>;

//===------------------------------
// 2. UImm12 and SImm9
//===------------------------------

// These instructions have two operands providing the address so they can be
// treated similarly for most purposes.

//===------------------------------
// 2.1 Base patterns covering extend/truncate semantics
//===------------------------------

// Atomic patterns can be shared between integer operations of all sizes, a
// quick multiclass here allows reuse.
multiclass ls_atomic_pats<Instruction LOAD, Instruction STORE, dag Base,
                          dag Offset, dag address, ValueType transty,
                          ValueType sty> {
  def : Pat<(!cast<PatFrag>("atomic_load_simple_" # sty) address),
            (LOAD Base, Offset)>;

  def : Pat<(!cast<PatFrag>("atomic_store_simple_" # sty) address, transty:$Rt),
            (STORE $Rt, Base, Offset)>;
}

// Instructions accessing a memory chunk smaller than a register (or, in a
// pinch, the same size) have a characteristic set of patterns they want to
// match: extending loads and truncating stores. This class deals with the
// sign-neutral version of those patterns.
//
// It will be instantiated across multiple addressing-modes.
multiclass ls_small_pats<Instruction LOAD, Instruction STORE,
                         dag Base, dag Offset,
                         dag address, ValueType sty>
  : ls_atomic_pats<LOAD, STORE, Base, Offset, address, i32, sty> {
  def : Pat<(!cast<SDNode>(zextload # sty) address), (LOAD Base, Offset)>;

  def : Pat<(!cast<SDNode>(extload # sty) address), (LOAD Base, Offset)>;

  // For zero-extension to 64-bits we have to tell LLVM that the whole 64-bit
  // register was actually set.
  def : Pat<(i64 (!cast<SDNode>(zextload # sty) address)),
            (SUBREG_TO_REG (i64 0), (LOAD Base, Offset), sub_32)>;

  def : Pat<(i64 (!cast<SDNode>(extload # sty) address)),
            (SUBREG_TO_REG (i64 0), (LOAD Base, Offset), sub_32)>;

  def : Pat<(!cast<SDNode>(truncstore # sty) i32:$Rt, address),
            (STORE $Rt, Base, Offset)>;

  // For truncating store from 64-bits, we have to manually tell LLVM to
  // ignore the high bits of the x register.
  def : Pat<(!cast<SDNode>(truncstore # sty) i64:$Rt, address),
            (STORE (EXTRACT_SUBREG $Rt, sub_32), Base, Offset)>;
}

// Next come patterns for sign-extending loads.
multiclass load_signed_pats<string T, string U, dag Base, dag Offset,
                            dag address, ValueType sty> {
  def : Pat<(i32 (!cast<SDNode>("sextload" # sty) address)),
            (!cast<Instruction>("LDRS" # T # "w" # U) Base, Offset)>;

  def : Pat<(i64 (!cast<SDNode>("sextload" # sty) address)),
            (!cast<Instruction>("LDRS" # T # "x" # U) Base, Offset)>;

}

// and finally "natural-width" loads and stores come next.
multiclass ls_neutral_pats<Instruction LOAD, Instruction STORE, dag Base,
                           dag Offset, dag address, ValueType sty> {
  def : Pat<(sty (load address)), (LOAD Base, Offset)>;
  def : Pat<(store sty:$Rt, address), (STORE $Rt, Base, Offset)>;
}

// Integer operations also get atomic instructions to select for.
multiclass ls_int_neutral_pats<Instruction LOAD, Instruction STORE, dag Base,
                           dag Offset, dag address, ValueType sty>
  : ls_neutral_pats<LOAD, STORE, Base, Offset, address, sty>,
    ls_atomic_pats<LOAD, STORE, Base, Offset, address, sty, sty>;

//===------------------------------
// 2.2. Addressing-mode instantiations
//===------------------------------

multiclass uimm12_pats<dag address, dag Base, dag Offset> {
  defm : ls_small_pats<LS8_LDR, LS8_STR, Base,
                       !foreach(decls.pattern, Offset,
                                !subst(OFFSET, byte_uimm12, decls.pattern)),
                       !foreach(decls.pattern, address,
                                !subst(OFFSET, byte_uimm12,
                                !subst(ALIGN, any_align, decls.pattern))),
                       i8>;
  defm : ls_small_pats<LS16_LDR, LS16_STR, Base,
                       !foreach(decls.pattern, Offset,
                                !subst(OFFSET, hword_uimm12, decls.pattern)),
                       !foreach(decls.pattern, address,
                                !subst(OFFSET, hword_uimm12,
                                !subst(ALIGN, min_align2, decls.pattern))),
                       i16>;
  defm : ls_small_pats<LS32_LDR, LS32_STR, Base,
                       !foreach(decls.pattern, Offset,
                                !subst(OFFSET, word_uimm12, decls.pattern)),
                       !foreach(decls.pattern, address,
                                !subst(OFFSET, word_uimm12,
                                !subst(ALIGN, min_align4, decls.pattern))),
                       i32>;

  defm : ls_int_neutral_pats<LS32_LDR, LS32_STR, Base,
                          !foreach(decls.pattern, Offset,
                                   !subst(OFFSET, word_uimm12, decls.pattern)),
                          !foreach(decls.pattern, address,
                                   !subst(OFFSET, word_uimm12,
                                   !subst(ALIGN, min_align4, decls.pattern))),
                          i32>;

  defm : ls_int_neutral_pats<LS64_LDR, LS64_STR, Base,
                          !foreach(decls.pattern, Offset,
                                   !subst(OFFSET, dword_uimm12, decls.pattern)),
                          !foreach(decls.pattern, address,
                                   !subst(OFFSET, dword_uimm12,
                                   !subst(ALIGN, min_align8, decls.pattern))),
                          i64>;

  defm : ls_neutral_pats<LSFP16_LDR, LSFP16_STR, Base,
                          !foreach(decls.pattern, Offset,
                                   !subst(OFFSET, hword_uimm12, decls.pattern)),
                          !foreach(decls.pattern, address,
                                   !subst(OFFSET, hword_uimm12,
                                   !subst(ALIGN, min_align2, decls.pattern))),
                          f16>;

  defm : ls_neutral_pats<LSFP32_LDR, LSFP32_STR, Base,
                          !foreach(decls.pattern, Offset,
                                   !subst(OFFSET, word_uimm12, decls.pattern)),
                          !foreach(decls.pattern, address,
                                   !subst(OFFSET, word_uimm12,
                                   !subst(ALIGN, min_align4, decls.pattern))),
                          f32>;

  defm : ls_neutral_pats<LSFP64_LDR, LSFP64_STR, Base,
                          !foreach(decls.pattern, Offset,
                                   !subst(OFFSET, dword_uimm12, decls.pattern)),
                          !foreach(decls.pattern, address,
                                   !subst(OFFSET, dword_uimm12,
                                   !subst(ALIGN, min_align8, decls.pattern))),
                          f64>;

  defm : ls_neutral_pats<LSFP128_LDR, LSFP128_STR, Base,
                          !foreach(decls.pattern, Offset,
                                   !subst(OFFSET, qword_uimm12, decls.pattern)),
                          !foreach(decls.pattern, address,
                                   !subst(OFFSET, qword_uimm12,
                                   !subst(ALIGN, min_align16, decls.pattern))),
                          f128>;

  defm : load_signed_pats<"B", "", Base,
                          !foreach(decls.pattern, Offset,
                                   !subst(OFFSET, byte_uimm12, decls.pattern)),
                          !foreach(decls.pattern, address,
                                   !subst(OFFSET, byte_uimm12,
                                   !subst(ALIGN, any_align, decls.pattern))),
                          i8>;

  defm : load_signed_pats<"H", "", Base,
                          !foreach(decls.pattern, Offset,
                                   !subst(OFFSET, hword_uimm12, decls.pattern)),
                          !foreach(decls.pattern, address,
                                   !subst(OFFSET, hword_uimm12,
                                   !subst(ALIGN, min_align2, decls.pattern))),
                          i16>;

  def : Pat<(sextloadi32 !foreach(decls.pattern, address,
                                  !subst(OFFSET, word_uimm12,
                                  !subst(ALIGN, min_align4, decls.pattern)))),
            (LDRSWx Base, !foreach(decls.pattern, Offset,
                                  !subst(OFFSET, word_uimm12, decls.pattern)))>;
}

// Straightforward patterns of last resort: a pointer with or without an
// appropriate offset.
defm : uimm12_pats<(i64 i64:$Rn), (i64 i64:$Rn), (i64 0)>;
defm : uimm12_pats<(add i64:$Rn, OFFSET:$UImm12),
                   (i64 i64:$Rn), (i64 OFFSET:$UImm12)>;

// The offset could be hidden behind an "or", of course:
defm : uimm12_pats<(add_like_or i64:$Rn, OFFSET:$UImm12),
                   (i64 i64:$Rn), (i64 OFFSET:$UImm12)>;

// Global addresses under the small-absolute model should use these
// instructions. There are ELF relocations specifically for it.
defm : uimm12_pats<(A64WrapperSmall tglobaladdr:$Hi, tglobaladdr:$Lo12, ALIGN),
                   (ADRPxi tglobaladdr:$Hi), (i64 tglobaladdr:$Lo12)>;

defm : uimm12_pats<(A64WrapperSmall tglobaltlsaddr:$Hi, tglobaltlsaddr:$Lo12,
                                    ALIGN),
                   (ADRPxi tglobaltlsaddr:$Hi), (i64 tglobaltlsaddr:$Lo12)>;

// External symbols that make it this far should also get standard relocations.
defm : uimm12_pats<(A64WrapperSmall texternalsym:$Hi, texternalsym:$Lo12,
                                    ALIGN),
                   (ADRPxi texternalsym:$Hi), (i64 texternalsym:$Lo12)>;

defm : uimm12_pats<(A64WrapperSmall tconstpool:$Hi, tconstpool:$Lo12, ALIGN),
                   (ADRPxi tconstpool:$Hi), (i64 tconstpool:$Lo12)>;

// We also want to use uimm12 instructions for local variables at the moment.
def tframeindex_XFORM : SDNodeXForm<frameindex, [{
  int FI = cast<FrameIndexSDNode>(N)->getIndex();
  return CurDAG->getTargetFrameIndex(FI, MVT::i64);
}]>;

defm : uimm12_pats<(i64 frameindex:$Rn),
                   (tframeindex_XFORM tframeindex:$Rn), (i64 0)>;

// These can be much simpler than uimm12 because we don't to change the operand
// type (e.g. LDURB and LDURH take the same operands).
multiclass simm9_pats<dag address, dag Base, dag Offset> {
  defm : ls_small_pats<LS8_LDUR, LS8_STUR, Base, Offset, address, i8>;
  defm : ls_small_pats<LS16_LDUR, LS16_STUR, Base, Offset, address, i16>;

  defm : ls_int_neutral_pats<LS32_LDUR, LS32_STUR, Base, Offset, address, i32>;
  defm : ls_int_neutral_pats<LS64_LDUR, LS64_STUR, Base, Offset, address, i64>;

  defm : ls_neutral_pats<LSFP16_LDUR, LSFP16_STUR, Base, Offset, address, f16>;
  defm : ls_neutral_pats<LSFP32_LDUR, LSFP32_STUR, Base, Offset, address, f32>;
  defm : ls_neutral_pats<LSFP64_LDUR, LSFP64_STUR, Base, Offset, address, f64>;
  defm : ls_neutral_pats<LSFP128_LDUR, LSFP128_STUR, Base, Offset, address,
                         f128>;

  def : Pat<(i64 (zextloadi32 address)),
            (SUBREG_TO_REG (i64 0), (LS32_LDUR Base, Offset), sub_32)>;

  def : Pat<(truncstorei32 i64:$Rt, address),
            (LS32_STUR (EXTRACT_SUBREG $Rt, sub_32), Base, Offset)>;

  defm : load_signed_pats<"B", "_U", Base, Offset, address, i8>;
  defm : load_signed_pats<"H", "_U", Base, Offset, address, i16>;
  def : Pat<(sextloadi32 address), (LDURSWx Base, Offset)>;
}

defm : simm9_pats<(add i64:$Rn, simm9:$SImm9),
                  (i64 $Rn), (SDXF_simm9 simm9:$SImm9)>;

defm : simm9_pats<(add_like_or i64:$Rn, simm9:$SImm9),
                  (i64 $Rn), (SDXF_simm9 simm9:$SImm9)>;


//===------------------------------
// 3. Register offset patterns
//===------------------------------

// Atomic patterns can be shared between integer operations of all sizes, a
// quick multiclass here allows reuse.
multiclass ro_atomic_pats<Instruction LOAD, Instruction STORE, dag Base,
                          dag Offset, dag Extend, dag address,
                          ValueType transty, ValueType sty> {
  def : Pat<(!cast<PatFrag>("atomic_load_simple_" # sty) address),
            (LOAD Base, Offset, Extend)>;

  def : Pat<(!cast<PatFrag>("atomic_store_simple_" # sty) address, transty:$Rt),
            (STORE $Rt, Base, Offset, Extend)>;
}

// The register offset instructions take three operands giving the instruction,
// and have an annoying split between instructions where Rm is 32-bit and
// 64-bit. So we need a special hierarchy to describe them. Other than that the
// same operations should be supported as for simm9 and uimm12 addressing.

multiclass ro_small_pats<Instruction LOAD, Instruction STORE,
                         dag Base, dag Offset, dag Extend,
                         dag address, ValueType sty>
  : ro_atomic_pats<LOAD, STORE, Base, Offset, Extend, address, i32, sty> {
  def : Pat<(!cast<SDNode>(zextload # sty) address),
            (LOAD Base, Offset, Extend)>;

  def : Pat<(!cast<SDNode>(extload # sty) address),
            (LOAD Base, Offset, Extend)>;

  // For zero-extension to 64-bits we have to tell LLVM that the whole 64-bit
  // register was actually set.
  def : Pat<(i64 (!cast<SDNode>(zextload # sty) address)),
            (SUBREG_TO_REG (i64 0), (LOAD Base, Offset, Extend), sub_32)>;

  def : Pat<(i64 (!cast<SDNode>(extload # sty) address)),
            (SUBREG_TO_REG (i64 0), (LOAD Base, Offset, Extend), sub_32)>;

  def : Pat<(!cast<SDNode>(truncstore # sty) i32:$Rt, address),
            (STORE $Rt, Base, Offset, Extend)>;

  // For truncating store from 64-bits, we have to manually tell LLVM to
  // ignore the high bits of the x register.
  def : Pat<(!cast<SDNode>(truncstore # sty) i64:$Rt, address),
            (STORE (EXTRACT_SUBREG $Rt, sub_32), Base, Offset, Extend)>;

}

// Next come patterns for sign-extending loads.
multiclass ro_signed_pats<string T, string Rm, dag Base, dag Offset, dag Extend,
                          dag address, ValueType sty> {
  def : Pat<(i32 (!cast<SDNode>("sextload" # sty) address)),
            (!cast<Instruction>("LDRS" # T # "w_" # Rm # "_RegOffset")
              Base, Offset, Extend)>;

  def : Pat<(i64 (!cast<SDNode>("sextload" # sty) address)),
            (!cast<Instruction>("LDRS" # T # "x_" # Rm # "_RegOffset")
              Base, Offset, Extend)>;
}

// and finally "natural-width" loads and stores come next.
multiclass ro_neutral_pats<Instruction LOAD, Instruction STORE,
                           dag Base, dag Offset, dag Extend, dag address,
                           ValueType sty> {
  def : Pat<(sty (load address)), (LOAD Base, Offset, Extend)>;
  def : Pat<(store sty:$Rt, address),
            (STORE $Rt, Base, Offset, Extend)>;
}

multiclass ro_int_neutral_pats<Instruction LOAD, Instruction STORE,
                               dag Base, dag Offset, dag Extend, dag address,
                               ValueType sty>
  : ro_neutral_pats<LOAD, STORE, Base, Offset, Extend, address, sty>,
    ro_atomic_pats<LOAD, STORE, Base, Offset, Extend, address, sty, sty>;

multiclass regoff_pats<string Rm, dag address, dag Base, dag Offset,
                       dag Extend> {
  defm : ro_small_pats<!cast<Instruction>("LS8_" # Rm # "_RegOffset_LDR"),
                       !cast<Instruction>("LS8_" # Rm # "_RegOffset_STR"),
                       Base, Offset, Extend,
                       !foreach(decls.pattern, address,
                                !subst(SHIFT, imm_eq0, decls.pattern)),
                       i8>;
  defm : ro_small_pats<!cast<Instruction>("LS16_" # Rm # "_RegOffset_LDR"),
                       !cast<Instruction>("LS16_" # Rm # "_RegOffset_STR"),
                       Base, Offset, Extend,
                       !foreach(decls.pattern, address,
                                !subst(SHIFT, imm_eq1, decls.pattern)),
                       i16>;
  defm : ro_small_pats<!cast<Instruction>("LS32_" # Rm # "_RegOffset_LDR"),
                       !cast<Instruction>("LS32_" # Rm # "_RegOffset_STR"),
                       Base, Offset, Extend,
                       !foreach(decls.pattern, address,
                                !subst(SHIFT, imm_eq2, decls.pattern)),
                       i32>;

  defm : ro_int_neutral_pats<
                            !cast<Instruction>("LS32_" # Rm # "_RegOffset_LDR"),
                            !cast<Instruction>("LS32_" # Rm # "_RegOffset_STR"),
                            Base, Offset, Extend,
                            !foreach(decls.pattern, address,
                                     !subst(SHIFT, imm_eq2, decls.pattern)),
                            i32>;

  defm : ro_int_neutral_pats<
                            !cast<Instruction>("LS64_" # Rm # "_RegOffset_LDR"),
                            !cast<Instruction>("LS64_" # Rm # "_RegOffset_STR"),
                            Base, Offset, Extend,
                            !foreach(decls.pattern, address,
                                     !subst(SHIFT, imm_eq3, decls.pattern)),
                            i64>;

  defm : ro_neutral_pats<!cast<Instruction>("LSFP16_" # Rm # "_RegOffset_LDR"),
                         !cast<Instruction>("LSFP16_" # Rm # "_RegOffset_STR"),
                         Base, Offset, Extend,
                         !foreach(decls.pattern, address,
                                  !subst(SHIFT, imm_eq1, decls.pattern)),
                         f16>;

  defm : ro_neutral_pats<!cast<Instruction>("LSFP32_" # Rm # "_RegOffset_LDR"),
                         !cast<Instruction>("LSFP32_" # Rm # "_RegOffset_STR"),
                         Base, Offset, Extend,
                         !foreach(decls.pattern, address,
                                  !subst(SHIFT, imm_eq2, decls.pattern)),
                         f32>;

  defm : ro_neutral_pats<!cast<Instruction>("LSFP64_" # Rm # "_RegOffset_LDR"),
                         !cast<Instruction>("LSFP64_" # Rm # "_RegOffset_STR"),
                         Base, Offset, Extend,
                         !foreach(decls.pattern, address,
                                  !subst(SHIFT, imm_eq3, decls.pattern)),
                         f64>;

  defm : ro_neutral_pats<!cast<Instruction>("LSFP128_" # Rm # "_RegOffset_LDR"),
                         !cast<Instruction>("LSFP128_" # Rm # "_RegOffset_STR"),
                         Base, Offset, Extend,
                         !foreach(decls.pattern, address,
                                  !subst(SHIFT, imm_eq4, decls.pattern)),
                         f128>;

  defm : ro_signed_pats<"B", Rm, Base, Offset, Extend,
                        !foreach(decls.pattern, address,
                                 !subst(SHIFT, imm_eq0, decls.pattern)),
                        i8>;

  defm : ro_signed_pats<"H", Rm, Base, Offset, Extend,
                        !foreach(decls.pattern, address,
                                 !subst(SHIFT, imm_eq1, decls.pattern)),
                        i16>;

  def : Pat<(sextloadi32 !foreach(decls.pattern, address,
                                  !subst(SHIFT, imm_eq2, decls.pattern))),
            (!cast<Instruction>("LDRSWx_" # Rm # "_RegOffset")
              Base, Offset, Extend)>;
}


// Finally we're in a position to tell LLVM exactly what addresses are reachable
// using register-offset instructions. Essentially a base plus a possibly
// extended, possibly shifted (by access size) offset.

defm : regoff_pats<"Wm", (add i64:$Rn, (sext i32:$Rm)),
                   (i64 i64:$Rn), (i32 i32:$Rm), (i64 6)>;

defm : regoff_pats<"Wm", (add i64:$Rn, (shl (sext i32:$Rm), SHIFT)),
                   (i64 i64:$Rn), (i32 i32:$Rm), (i64 7)>;

defm : regoff_pats<"Wm", (add i64:$Rn, (zext i32:$Rm)),
                   (i64 i64:$Rn), (i32 i32:$Rm), (i64 2)>;

defm : regoff_pats<"Wm", (add i64:$Rn, (shl (zext i32:$Rm), SHIFT)),
                   (i64 i64:$Rn), (i32 i32:$Rm), (i64 3)>;

defm : regoff_pats<"Xm", (add i64:$Rn, i64:$Rm),
                   (i64 i64:$Rn), (i64 i64:$Rm), (i64 2)>;

defm : regoff_pats<"Xm", (add i64:$Rn, (shl i64:$Rm, SHIFT)),
                   (i64 i64:$Rn), (i64 i64:$Rm), (i64 3)>;

//===----------------------------------------------------------------------===//
// Advanced SIMD (NEON) Support
//

include "AArch64InstrNEON.td"
