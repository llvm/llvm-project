//===- PPCInstrVSX.td - The PowerPC VSX Extension --*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file describes the VSX extension to the PowerPC instruction set.
//
//===----------------------------------------------------------------------===//

// *********************************** NOTE ***********************************
// ** For POWER8 Little Endian, the VSX swap optimization relies on knowing  **
// ** which VMX and VSX instructions are lane-sensitive and which are not.   **
// ** A lane-sensitive instruction relies, implicitly or explicitly, on      **
// ** whether lanes are numbered from left to right.  An instruction like    **
// ** VADDFP is not lane-sensitive, because each lane of the result vector   **
// ** relies only on the corresponding lane of the source vectors.  However, **
// ** an instruction like VMULESB is lane-sensitive, because "even" and      **
// ** "odd" lanes are different for big-endian and little-endian numbering.  **
// **                                                                        **
// ** When adding new VMX and VSX instructions, please consider whether they **
// ** are lane-sensitive.  If so, they must be added to a switch statement   **
// ** in PPCVSXSwapRemoval::gatherVectorInstructions().                      **
// ****************************************************************************

def PPCRegVSRCAsmOperand : AsmOperandClass {
  let Name = "RegVSRC"; let PredicateMethod = "isVSRegNumber";
}
def vsrc : RegisterOperand<VSRC> {
  let ParserMatchClass = PPCRegVSRCAsmOperand;
}

def PPCRegVSFRCAsmOperand : AsmOperandClass {
  let Name = "RegVSFRC"; let PredicateMethod = "isVSRegNumber";
}
def vsfrc : RegisterOperand<VSFRC> {
  let ParserMatchClass = PPCRegVSFRCAsmOperand;
}

def PPCRegVSSRCAsmOperand : AsmOperandClass {
  let Name = "RegVSSRC"; let PredicateMethod = "isVSRegNumber";
}
def vssrc : RegisterOperand<VSSRC> {
  let ParserMatchClass = PPCRegVSSRCAsmOperand;
}

// Little-endian-specific nodes.
def SDT_PPClxvd2x : SDTypeProfile<1, 1, [
  SDTCisVT<0, v2f64>, SDTCisPtrTy<1>
]>;
def SDT_PPCstxvd2x : SDTypeProfile<0, 2, [
  SDTCisVT<0, v2f64>, SDTCisPtrTy<1>
]>;
def SDT_PPCxxswapd : SDTypeProfile<1, 1, [
  SDTCisSameAs<0, 1>
]>;

def PPClxvd2x  : SDNode<"PPCISD::LXVD2X", SDT_PPClxvd2x,
                        [SDNPHasChain, SDNPMayLoad]>;
def PPCstxvd2x : SDNode<"PPCISD::STXVD2X", SDT_PPCstxvd2x,
                        [SDNPHasChain, SDNPMayStore]>;
def PPCxxswapd : SDNode<"PPCISD::XXSWAPD", SDT_PPCxxswapd, [SDNPHasChain]>;
def PPCmfvsr : SDNode<"PPCISD::MFVSR", SDTUnaryOp, []>;
def PPCmtvsra : SDNode<"PPCISD::MTVSRA", SDTUnaryOp, []>;
def PPCmtvsrz : SDNode<"PPCISD::MTVSRZ", SDTUnaryOp, []>;

multiclass XX3Form_Rcr<bits<6> opcode, bits<7> xo, string asmbase,
                    string asmstr, InstrItinClass itin, Intrinsic Int,
                    ValueType OutTy, ValueType InTy> {
  let BaseName = asmbase in {
    def NAME : XX3Form_Rc<opcode, xo, (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                       !strconcat(asmbase, !strconcat(" ", asmstr)), itin,
                       [(set OutTy:$XT, (Int InTy:$XA, InTy:$XB))]>;
    let Defs = [CR6] in
    def o    : XX3Form_Rc<opcode, xo, (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                       !strconcat(asmbase, !strconcat(". ", asmstr)), itin,
                       [(set InTy:$XT,
                                (InTy (PPCvcmp_o InTy:$XA, InTy:$XB, xo)))]>,
                       isDOT;
  }
}

def HasVSX : Predicate<"PPCSubTarget->hasVSX()">;
def IsLittleEndian : Predicate<"PPCSubTarget->isLittleEndian()">;
def IsBigEndian : Predicate<"!PPCSubTarget->isLittleEndian()">;

let Predicates = [HasVSX] in {
let AddedComplexity = 400 in { // Prefer VSX patterns over non-VSX patterns.
let hasSideEffects = 0 in { // VSX instructions don't have side effects.
let Uses = [RM] in {

  // Load indexed instructions
  let mayLoad = 1 in {
    def LXSDX : XX1Form<31, 588,
                        (outs vsfrc:$XT), (ins memrr:$src),
                        "lxsdx $XT, $src", IIC_LdStLFD,
                        [(set f64:$XT, (load xoaddr:$src))]>;

    def LXVD2X : XX1Form<31, 844,
                         (outs vsrc:$XT), (ins memrr:$src),
                         "lxvd2x $XT, $src", IIC_LdStLFD,
                         [(set v2f64:$XT, (int_ppc_vsx_lxvd2x xoaddr:$src))]>;

    def LXVDSX : XX1Form<31, 332,
                         (outs vsrc:$XT), (ins memrr:$src),
                         "lxvdsx $XT, $src", IIC_LdStLFD, []>;

    def LXVW4X : XX1Form<31, 780,
                         (outs vsrc:$XT), (ins memrr:$src),
                         "lxvw4x $XT, $src", IIC_LdStLFD,
                         [(set v4i32:$XT, (int_ppc_vsx_lxvw4x xoaddr:$src))]>;
  } // mayLoad

  // Store indexed instructions
  let mayStore = 1 in {
    def STXSDX : XX1Form<31, 716,
                        (outs), (ins vsfrc:$XT, memrr:$dst),
                        "stxsdx $XT, $dst", IIC_LdStSTFD,
                        [(store f64:$XT, xoaddr:$dst)]>;

    def STXVD2X : XX1Form<31, 972,
                         (outs), (ins vsrc:$XT, memrr:$dst),
                         "stxvd2x $XT, $dst", IIC_LdStSTFD,
                         [(store v2f64:$XT, xoaddr:$dst)]>;

    def STXVW4X : XX1Form<31, 908,
                         (outs), (ins vsrc:$XT, memrr:$dst),
                         "stxvw4x $XT, $dst", IIC_LdStSTFD,
                         [(store v4i32:$XT, xoaddr:$dst)]>;

  } // mayStore

  // Add/Mul Instructions
  let isCommutable = 1 in {
    def XSADDDP : XX3Form<60, 32,
                          (outs vsfrc:$XT), (ins vsfrc:$XA, vsfrc:$XB),
                          "xsadddp $XT, $XA, $XB", IIC_VecFP,
                          [(set f64:$XT, (fadd f64:$XA, f64:$XB))]>;
    def XSMULDP : XX3Form<60, 48,
                          (outs vsfrc:$XT), (ins vsfrc:$XA, vsfrc:$XB),
                          "xsmuldp $XT, $XA, $XB", IIC_VecFP,
                          [(set f64:$XT, (fmul f64:$XA, f64:$XB))]>;

    def XVADDDP : XX3Form<60, 96,
                          (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                          "xvadddp $XT, $XA, $XB", IIC_VecFP,
                          [(set v2f64:$XT, (fadd v2f64:$XA, v2f64:$XB))]>;

    def XVADDSP : XX3Form<60, 64,
                          (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                          "xvaddsp $XT, $XA, $XB", IIC_VecFP,
                          [(set v4f32:$XT, (fadd v4f32:$XA, v4f32:$XB))]>;

    def XVMULDP : XX3Form<60, 112,
                          (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                          "xvmuldp $XT, $XA, $XB", IIC_VecFP,
                          [(set v2f64:$XT, (fmul v2f64:$XA, v2f64:$XB))]>;

    def XVMULSP : XX3Form<60, 80,
                          (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                          "xvmulsp $XT, $XA, $XB", IIC_VecFP,
                          [(set v4f32:$XT, (fmul v4f32:$XA, v4f32:$XB))]>;
  }

  // Subtract Instructions
  def XSSUBDP : XX3Form<60, 40,
                        (outs vsfrc:$XT), (ins vsfrc:$XA, vsfrc:$XB),
                        "xssubdp $XT, $XA, $XB", IIC_VecFP,
                        [(set f64:$XT, (fsub f64:$XA, f64:$XB))]>;

  def XVSUBDP : XX3Form<60, 104,
                        (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                        "xvsubdp $XT, $XA, $XB", IIC_VecFP,
                        [(set v2f64:$XT, (fsub v2f64:$XA, v2f64:$XB))]>;
  def XVSUBSP : XX3Form<60, 72,
                        (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                        "xvsubsp $XT, $XA, $XB", IIC_VecFP,
                        [(set v4f32:$XT, (fsub v4f32:$XA, v4f32:$XB))]>;

  // FMA Instructions
  let BaseName = "XSMADDADP" in {
  let isCommutable = 1 in
  def XSMADDADP : XX3Form<60, 33,
                          (outs vsfrc:$XT), (ins vsfrc:$XTi, vsfrc:$XA, vsfrc:$XB),
                          "xsmaddadp $XT, $XA, $XB", IIC_VecFP,
                          [(set f64:$XT, (fma f64:$XA, f64:$XB, f64:$XTi))]>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  let IsVSXFMAAlt = 1 in
  def XSMADDMDP : XX3Form<60, 41,
                          (outs vsfrc:$XT), (ins vsfrc:$XTi, vsfrc:$XA, vsfrc:$XB),
                          "xsmaddmdp $XT, $XA, $XB", IIC_VecFP, []>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  }

  let BaseName = "XSMSUBADP" in {
  let isCommutable = 1 in
  def XSMSUBADP : XX3Form<60, 49,
                          (outs vsfrc:$XT), (ins vsfrc:$XTi, vsfrc:$XA, vsfrc:$XB),
                          "xsmsubadp $XT, $XA, $XB", IIC_VecFP,
                          [(set f64:$XT, (fma f64:$XA, f64:$XB, (fneg f64:$XTi)))]>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  let IsVSXFMAAlt = 1 in
  def XSMSUBMDP : XX3Form<60, 57,
                          (outs vsfrc:$XT), (ins vsfrc:$XTi, vsfrc:$XA, vsfrc:$XB),
                          "xsmsubmdp $XT, $XA, $XB", IIC_VecFP, []>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  }

  let BaseName = "XSNMADDADP" in {
  let isCommutable = 1 in
  def XSNMADDADP : XX3Form<60, 161,
                          (outs vsfrc:$XT), (ins vsfrc:$XTi, vsfrc:$XA, vsfrc:$XB),
                          "xsnmaddadp $XT, $XA, $XB", IIC_VecFP,
                          [(set f64:$XT, (fneg (fma f64:$XA, f64:$XB, f64:$XTi)))]>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  let IsVSXFMAAlt = 1 in
  def XSNMADDMDP : XX3Form<60, 169,
                          (outs vsfrc:$XT), (ins vsfrc:$XTi, vsfrc:$XA, vsfrc:$XB),
                          "xsnmaddmdp $XT, $XA, $XB", IIC_VecFP, []>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  }

  let BaseName = "XSNMSUBADP" in {
  let isCommutable = 1 in
  def XSNMSUBADP : XX3Form<60, 177,
                          (outs vsfrc:$XT), (ins vsfrc:$XTi, vsfrc:$XA, vsfrc:$XB),
                          "xsnmsubadp $XT, $XA, $XB", IIC_VecFP,
                          [(set f64:$XT, (fneg (fma f64:$XA, f64:$XB, (fneg f64:$XTi))))]>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  let IsVSXFMAAlt = 1 in
  def XSNMSUBMDP : XX3Form<60, 185,
                          (outs vsfrc:$XT), (ins vsfrc:$XTi, vsfrc:$XA, vsfrc:$XB),
                          "xsnmsubmdp $XT, $XA, $XB", IIC_VecFP, []>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  }

  let BaseName = "XVMADDADP" in {
  let isCommutable = 1 in
  def XVMADDADP : XX3Form<60, 97,
                          (outs vsrc:$XT), (ins vsrc:$XTi, vsrc:$XA, vsrc:$XB),
                          "xvmaddadp $XT, $XA, $XB", IIC_VecFP,
                          [(set v2f64:$XT, (fma v2f64:$XA, v2f64:$XB, v2f64:$XTi))]>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  let IsVSXFMAAlt = 1 in
  def XVMADDMDP : XX3Form<60, 105,
                          (outs vsrc:$XT), (ins vsrc:$XTi, vsrc:$XA, vsrc:$XB),
                          "xvmaddmdp $XT, $XA, $XB", IIC_VecFP, []>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  }

  let BaseName = "XVMADDASP" in {
  let isCommutable = 1 in
  def XVMADDASP : XX3Form<60, 65,
                          (outs vsrc:$XT), (ins vsrc:$XTi, vsrc:$XA, vsrc:$XB),
                          "xvmaddasp $XT, $XA, $XB", IIC_VecFP,
                          [(set v4f32:$XT, (fma v4f32:$XA, v4f32:$XB, v4f32:$XTi))]>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  let IsVSXFMAAlt = 1 in
  def XVMADDMSP : XX3Form<60, 73,
                          (outs vsrc:$XT), (ins vsrc:$XTi, vsrc:$XA, vsrc:$XB),
                          "xvmaddmsp $XT, $XA, $XB", IIC_VecFP, []>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  }

  let BaseName = "XVMSUBADP" in {
  let isCommutable = 1 in
  def XVMSUBADP : XX3Form<60, 113,
                          (outs vsrc:$XT), (ins vsrc:$XTi, vsrc:$XA, vsrc:$XB),
                          "xvmsubadp $XT, $XA, $XB", IIC_VecFP,
                          [(set v2f64:$XT, (fma v2f64:$XA, v2f64:$XB, (fneg v2f64:$XTi)))]>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  let IsVSXFMAAlt = 1 in
  def XVMSUBMDP : XX3Form<60, 121,
                          (outs vsrc:$XT), (ins vsrc:$XTi, vsrc:$XA, vsrc:$XB),
                          "xvmsubmdp $XT, $XA, $XB", IIC_VecFP, []>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  }

  let BaseName = "XVMSUBASP" in {
  let isCommutable = 1 in
  def XVMSUBASP : XX3Form<60, 81,
                          (outs vsrc:$XT), (ins vsrc:$XTi, vsrc:$XA, vsrc:$XB),
                          "xvmsubasp $XT, $XA, $XB", IIC_VecFP,
                          [(set v4f32:$XT, (fma v4f32:$XA, v4f32:$XB, (fneg v4f32:$XTi)))]>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  let IsVSXFMAAlt = 1 in
  def XVMSUBMSP : XX3Form<60, 89,
                          (outs vsrc:$XT), (ins vsrc:$XTi, vsrc:$XA, vsrc:$XB),
                          "xvmsubmsp $XT, $XA, $XB", IIC_VecFP, []>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  }

  let BaseName = "XVNMADDADP" in {
  let isCommutable = 1 in
  def XVNMADDADP : XX3Form<60, 225,
                          (outs vsrc:$XT), (ins vsrc:$XTi, vsrc:$XA, vsrc:$XB),
                          "xvnmaddadp $XT, $XA, $XB", IIC_VecFP,
                          [(set v2f64:$XT, (fneg (fma v2f64:$XA, v2f64:$XB, v2f64:$XTi)))]>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  let IsVSXFMAAlt = 1 in
  def XVNMADDMDP : XX3Form<60, 233,
                          (outs vsrc:$XT), (ins vsrc:$XTi, vsrc:$XA, vsrc:$XB),
                          "xvnmaddmdp $XT, $XA, $XB", IIC_VecFP, []>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  }

  let BaseName = "XVNMADDASP" in {
  let isCommutable = 1 in
  def XVNMADDASP : XX3Form<60, 193,
                          (outs vsrc:$XT), (ins vsrc:$XTi, vsrc:$XA, vsrc:$XB),
                          "xvnmaddasp $XT, $XA, $XB", IIC_VecFP,
                          [(set v4f32:$XT, (fneg (fma v4f32:$XA, v4f32:$XB, v4f32:$XTi)))]>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  let IsVSXFMAAlt = 1 in
  def XVNMADDMSP : XX3Form<60, 201,
                          (outs vsrc:$XT), (ins vsrc:$XTi, vsrc:$XA, vsrc:$XB),
                          "xvnmaddmsp $XT, $XA, $XB", IIC_VecFP, []>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  }

  let BaseName = "XVNMSUBADP" in {
  let isCommutable = 1 in
  def XVNMSUBADP : XX3Form<60, 241,
                          (outs vsrc:$XT), (ins vsrc:$XTi, vsrc:$XA, vsrc:$XB),
                          "xvnmsubadp $XT, $XA, $XB", IIC_VecFP,
                          [(set v2f64:$XT, (fneg (fma v2f64:$XA, v2f64:$XB, (fneg v2f64:$XTi))))]>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  let IsVSXFMAAlt = 1 in
  def XVNMSUBMDP : XX3Form<60, 249,
                          (outs vsrc:$XT), (ins vsrc:$XTi, vsrc:$XA, vsrc:$XB),
                          "xvnmsubmdp $XT, $XA, $XB", IIC_VecFP, []>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  }

  let BaseName = "XVNMSUBASP" in {
  let isCommutable = 1 in
  def XVNMSUBASP : XX3Form<60, 209,
                          (outs vsrc:$XT), (ins vsrc:$XTi, vsrc:$XA, vsrc:$XB),
                          "xvnmsubasp $XT, $XA, $XB", IIC_VecFP,
                          [(set v4f32:$XT, (fneg (fma v4f32:$XA, v4f32:$XB, (fneg v4f32:$XTi))))]>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  let IsVSXFMAAlt = 1 in
  def XVNMSUBMSP : XX3Form<60, 217,
                          (outs vsrc:$XT), (ins vsrc:$XTi, vsrc:$XA, vsrc:$XB),
                          "xvnmsubmsp $XT, $XA, $XB", IIC_VecFP, []>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  }

  // Division Instructions
  def XSDIVDP : XX3Form<60, 56,
                        (outs vsfrc:$XT), (ins vsfrc:$XA, vsfrc:$XB),
                        "xsdivdp $XT, $XA, $XB", IIC_FPDivD,
                        [(set f64:$XT, (fdiv f64:$XA, f64:$XB))]>;
  def XSSQRTDP : XX2Form<60, 75,
                        (outs vsfrc:$XT), (ins vsfrc:$XB),
                        "xssqrtdp $XT, $XB", IIC_FPSqrtD,
                        [(set f64:$XT, (fsqrt f64:$XB))]>;

  def XSREDP : XX2Form<60, 90,
                        (outs vsfrc:$XT), (ins vsfrc:$XB),
                        "xsredp $XT, $XB", IIC_VecFP,
                        [(set f64:$XT, (PPCfre f64:$XB))]>;
  def XSRSQRTEDP : XX2Form<60, 74,
                           (outs vsfrc:$XT), (ins vsfrc:$XB),
                           "xsrsqrtedp $XT, $XB", IIC_VecFP,
                           [(set f64:$XT, (PPCfrsqrte f64:$XB))]>;

  def XSTDIVDP : XX3Form_1<60, 61,
                         (outs crrc:$crD), (ins vsfrc:$XA, vsfrc:$XB),
                         "xstdivdp $crD, $XA, $XB", IIC_FPCompare, []>;
  def XSTSQRTDP : XX2Form_1<60, 106,
                          (outs crrc:$crD), (ins vsfrc:$XB),
                          "xstsqrtdp $crD, $XB", IIC_FPCompare, []>;

  def XVDIVDP : XX3Form<60, 120,
                        (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                        "xvdivdp $XT, $XA, $XB", IIC_FPDivD,
                        [(set v2f64:$XT, (fdiv v2f64:$XA, v2f64:$XB))]>;
  def XVDIVSP : XX3Form<60, 88,
                        (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                        "xvdivsp $XT, $XA, $XB", IIC_FPDivS,
                        [(set v4f32:$XT, (fdiv v4f32:$XA, v4f32:$XB))]>;

  def XVSQRTDP : XX2Form<60, 203,
                        (outs vsrc:$XT), (ins vsrc:$XB),
                        "xvsqrtdp $XT, $XB", IIC_FPSqrtD,
                        [(set v2f64:$XT, (fsqrt v2f64:$XB))]>;
  def XVSQRTSP : XX2Form<60, 139,
                        (outs vsrc:$XT), (ins vsrc:$XB),
                        "xvsqrtsp $XT, $XB", IIC_FPSqrtS,
                        [(set v4f32:$XT, (fsqrt v4f32:$XB))]>;

  def XVTDIVDP : XX3Form_1<60, 125,
                         (outs crrc:$crD), (ins vsrc:$XA, vsrc:$XB),
                         "xvtdivdp $crD, $XA, $XB", IIC_FPCompare, []>;
  def XVTDIVSP : XX3Form_1<60, 93,
                         (outs crrc:$crD), (ins vsrc:$XA, vsrc:$XB),
                         "xvtdivsp $crD, $XA, $XB", IIC_FPCompare, []>;

  def XVTSQRTDP : XX2Form_1<60, 234,
                          (outs crrc:$crD), (ins vsrc:$XB),
                          "xvtsqrtdp $crD, $XB", IIC_FPCompare, []>;
  def XVTSQRTSP : XX2Form_1<60, 170,
                          (outs crrc:$crD), (ins vsrc:$XB),
                          "xvtsqrtsp $crD, $XB", IIC_FPCompare, []>;

  def XVREDP : XX2Form<60, 218,
                        (outs vsrc:$XT), (ins vsrc:$XB),
                        "xvredp $XT, $XB", IIC_VecFP,
                        [(set v2f64:$XT, (PPCfre v2f64:$XB))]>;
  def XVRESP : XX2Form<60, 154,
                        (outs vsrc:$XT), (ins vsrc:$XB),
                        "xvresp $XT, $XB", IIC_VecFP,
                        [(set v4f32:$XT, (PPCfre v4f32:$XB))]>;

  def XVRSQRTEDP : XX2Form<60, 202,
                           (outs vsrc:$XT), (ins vsrc:$XB),
                           "xvrsqrtedp $XT, $XB", IIC_VecFP,
                           [(set v2f64:$XT, (PPCfrsqrte v2f64:$XB))]>;
  def XVRSQRTESP : XX2Form<60, 138,
                           (outs vsrc:$XT), (ins vsrc:$XB),
                           "xvrsqrtesp $XT, $XB", IIC_VecFP,
                           [(set v4f32:$XT, (PPCfrsqrte v4f32:$XB))]>;

  // Compare Instructions
  def XSCMPODP : XX3Form_1<60, 43,
                           (outs crrc:$crD), (ins vsfrc:$XA, vsfrc:$XB),
                           "xscmpodp $crD, $XA, $XB", IIC_FPCompare, []>;
  def XSCMPUDP : XX3Form_1<60, 35,
                           (outs crrc:$crD), (ins vsfrc:$XA, vsfrc:$XB),
                           "xscmpudp $crD, $XA, $XB", IIC_FPCompare, []>;

  defm XVCMPEQDP : XX3Form_Rcr<60, 99,
                             "xvcmpeqdp", "$XT, $XA, $XB", IIC_VecFPCompare,
                             int_ppc_vsx_xvcmpeqdp, v2i64, v2f64>;
  defm XVCMPEQSP : XX3Form_Rcr<60, 67,
                             "xvcmpeqsp", "$XT, $XA, $XB", IIC_VecFPCompare,
                             int_ppc_vsx_xvcmpeqsp, v4i32, v4f32>;
  defm XVCMPGEDP : XX3Form_Rcr<60, 115,
                             "xvcmpgedp", "$XT, $XA, $XB", IIC_VecFPCompare,
                             int_ppc_vsx_xvcmpgedp, v2i64, v2f64>;
  defm XVCMPGESP : XX3Form_Rcr<60, 83,
                             "xvcmpgesp", "$XT, $XA, $XB", IIC_VecFPCompare,
                             int_ppc_vsx_xvcmpgesp, v4i32, v4f32>;
  defm XVCMPGTDP : XX3Form_Rcr<60, 107,
                             "xvcmpgtdp", "$XT, $XA, $XB", IIC_VecFPCompare,
                             int_ppc_vsx_xvcmpgtdp, v2i64, v2f64>;
  defm XVCMPGTSP : XX3Form_Rcr<60, 75,
                             "xvcmpgtsp", "$XT, $XA, $XB", IIC_VecFPCompare,
                             int_ppc_vsx_xvcmpgtsp, v4i32, v4f32>;

  // Move Instructions
  def XSABSDP : XX2Form<60, 345,
                      (outs vsfrc:$XT), (ins vsfrc:$XB),
                      "xsabsdp $XT, $XB", IIC_VecFP,
                      [(set f64:$XT, (fabs f64:$XB))]>;
  def XSNABSDP : XX2Form<60, 361,
                      (outs vsfrc:$XT), (ins vsfrc:$XB),
                      "xsnabsdp $XT, $XB", IIC_VecFP,
                      [(set f64:$XT, (fneg (fabs f64:$XB)))]>;
  def XSNEGDP : XX2Form<60, 377,
                      (outs vsfrc:$XT), (ins vsfrc:$XB),
                      "xsnegdp $XT, $XB", IIC_VecFP,
                      [(set f64:$XT, (fneg f64:$XB))]>;
  def XSCPSGNDP : XX3Form<60, 176,
                      (outs vsfrc:$XT), (ins vsfrc:$XA, vsfrc:$XB),
                      "xscpsgndp $XT, $XA, $XB", IIC_VecFP,
                      [(set f64:$XT, (fcopysign f64:$XB, f64:$XA))]>;

  def XVABSDP : XX2Form<60, 473,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvabsdp $XT, $XB", IIC_VecFP,
                      [(set v2f64:$XT, (fabs v2f64:$XB))]>;

  def XVABSSP : XX2Form<60, 409,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvabssp $XT, $XB", IIC_VecFP,
                      [(set v4f32:$XT, (fabs v4f32:$XB))]>;

  def XVCPSGNDP : XX3Form<60, 240,
                      (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                      "xvcpsgndp $XT, $XA, $XB", IIC_VecFP,
                      [(set v2f64:$XT, (fcopysign v2f64:$XB, v2f64:$XA))]>;
  def XVCPSGNSP : XX3Form<60, 208,
                      (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                      "xvcpsgnsp $XT, $XA, $XB", IIC_VecFP,
                      [(set v4f32:$XT, (fcopysign v4f32:$XB, v4f32:$XA))]>;

  def XVNABSDP : XX2Form<60, 489,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvnabsdp $XT, $XB", IIC_VecFP,
                      [(set v2f64:$XT, (fneg (fabs v2f64:$XB)))]>;
  def XVNABSSP : XX2Form<60, 425,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvnabssp $XT, $XB", IIC_VecFP,
                      [(set v4f32:$XT, (fneg (fabs v4f32:$XB)))]>;

  def XVNEGDP : XX2Form<60, 505,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvnegdp $XT, $XB", IIC_VecFP,
                      [(set v2f64:$XT, (fneg v2f64:$XB))]>;
  def XVNEGSP : XX2Form<60, 441,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvnegsp $XT, $XB", IIC_VecFP,
                      [(set v4f32:$XT, (fneg v4f32:$XB))]>;

  // Conversion Instructions
  def XSCVDPSP : XX2Form<60, 265,
                      (outs vsfrc:$XT), (ins vsfrc:$XB),
                      "xscvdpsp $XT, $XB", IIC_VecFP, []>;
  def XSCVDPSXDS : XX2Form<60, 344,
                      (outs vsfrc:$XT), (ins vsfrc:$XB),
                      "xscvdpsxds $XT, $XB", IIC_VecFP,
                      [(set f64:$XT, (PPCfctidz f64:$XB))]>;
  def XSCVDPSXWS : XX2Form<60, 88,
                      (outs vsfrc:$XT), (ins vsfrc:$XB),
                      "xscvdpsxws $XT, $XB", IIC_VecFP,
                      [(set f64:$XT, (PPCfctiwz f64:$XB))]>;
  def XSCVDPUXDS : XX2Form<60, 328,
                      (outs vsfrc:$XT), (ins vsfrc:$XB),
                      "xscvdpuxds $XT, $XB", IIC_VecFP,
                      [(set f64:$XT, (PPCfctiduz f64:$XB))]>;
  def XSCVDPUXWS : XX2Form<60, 72,
                      (outs vsfrc:$XT), (ins vsfrc:$XB),
                      "xscvdpuxws $XT, $XB", IIC_VecFP,
                      [(set f64:$XT, (PPCfctiwuz f64:$XB))]>;
  def XSCVSPDP : XX2Form<60, 329,
                      (outs vsfrc:$XT), (ins vsfrc:$XB),
                      "xscvspdp $XT, $XB", IIC_VecFP, []>;
  def XSCVSXDDP : XX2Form<60, 376,
                      (outs vsfrc:$XT), (ins vsfrc:$XB),
                      "xscvsxddp $XT, $XB", IIC_VecFP,
                      [(set f64:$XT, (PPCfcfid f64:$XB))]>;
  def XSCVUXDDP : XX2Form<60, 360,
                      (outs vsfrc:$XT), (ins vsfrc:$XB),
                      "xscvuxddp $XT, $XB", IIC_VecFP,
                      [(set f64:$XT, (PPCfcfidu f64:$XB))]>;

  def XVCVDPSP : XX2Form<60, 393,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvcvdpsp $XT, $XB", IIC_VecFP, []>;
  def XVCVDPSXDS : XX2Form<60, 472,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvcvdpsxds $XT, $XB", IIC_VecFP,
                      [(set v2i64:$XT, (fp_to_sint v2f64:$XB))]>;
  def XVCVDPSXWS : XX2Form<60, 216,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvcvdpsxws $XT, $XB", IIC_VecFP, []>;
  def XVCVDPUXDS : XX2Form<60, 456,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvcvdpuxds $XT, $XB", IIC_VecFP,
                      [(set v2i64:$XT, (fp_to_uint v2f64:$XB))]>;
  def XVCVDPUXWS : XX2Form<60, 200,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvcvdpuxws $XT, $XB", IIC_VecFP, []>;

  def XVCVSPDP : XX2Form<60, 457,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvcvspdp $XT, $XB", IIC_VecFP, []>;
  def XVCVSPSXDS : XX2Form<60, 408,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvcvspsxds $XT, $XB", IIC_VecFP, []>;
  def XVCVSPSXWS : XX2Form<60, 152,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvcvspsxws $XT, $XB", IIC_VecFP, []>;
  def XVCVSPUXDS : XX2Form<60, 392,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvcvspuxds $XT, $XB", IIC_VecFP, []>;
  def XVCVSPUXWS : XX2Form<60, 136,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvcvspuxws $XT, $XB", IIC_VecFP, []>;
  def XVCVSXDDP : XX2Form<60, 504,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvcvsxddp $XT, $XB", IIC_VecFP,
                      [(set v2f64:$XT, (sint_to_fp v2i64:$XB))]>;
  def XVCVSXDSP : XX2Form<60, 440,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvcvsxdsp $XT, $XB", IIC_VecFP, []>;
  def XVCVSXWDP : XX2Form<60, 248,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvcvsxwdp $XT, $XB", IIC_VecFP, []>;
  def XVCVSXWSP : XX2Form<60, 184,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvcvsxwsp $XT, $XB", IIC_VecFP, []>;
  def XVCVUXDDP : XX2Form<60, 488,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvcvuxddp $XT, $XB", IIC_VecFP,
                      [(set v2f64:$XT, (uint_to_fp v2i64:$XB))]>;
  def XVCVUXDSP : XX2Form<60, 424,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvcvuxdsp $XT, $XB", IIC_VecFP, []>;
  def XVCVUXWDP : XX2Form<60, 232,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvcvuxwdp $XT, $XB", IIC_VecFP, []>;
  def XVCVUXWSP : XX2Form<60, 168,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvcvuxwsp $XT, $XB", IIC_VecFP, []>;

  // Rounding Instructions
  def XSRDPI : XX2Form<60, 73,
                      (outs vsfrc:$XT), (ins vsfrc:$XB),
                      "xsrdpi $XT, $XB", IIC_VecFP,
                      [(set f64:$XT, (frnd f64:$XB))]>;
  def XSRDPIC : XX2Form<60, 107,
                      (outs vsfrc:$XT), (ins vsfrc:$XB),
                      "xsrdpic $XT, $XB", IIC_VecFP,
                      [(set f64:$XT, (fnearbyint f64:$XB))]>;
  def XSRDPIM : XX2Form<60, 121,
                      (outs vsfrc:$XT), (ins vsfrc:$XB),
                      "xsrdpim $XT, $XB", IIC_VecFP,
                      [(set f64:$XT, (ffloor f64:$XB))]>;
  def XSRDPIP : XX2Form<60, 105,
                      (outs vsfrc:$XT), (ins vsfrc:$XB),
                      "xsrdpip $XT, $XB", IIC_VecFP,
                      [(set f64:$XT, (fceil f64:$XB))]>;
  def XSRDPIZ : XX2Form<60, 89,
                      (outs vsfrc:$XT), (ins vsfrc:$XB),
                      "xsrdpiz $XT, $XB", IIC_VecFP,
                      [(set f64:$XT, (ftrunc f64:$XB))]>;

  def XVRDPI : XX2Form<60, 201,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvrdpi $XT, $XB", IIC_VecFP,
                      [(set v2f64:$XT, (frnd v2f64:$XB))]>;
  def XVRDPIC : XX2Form<60, 235,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvrdpic $XT, $XB", IIC_VecFP,
                      [(set v2f64:$XT, (fnearbyint v2f64:$XB))]>;
  def XVRDPIM : XX2Form<60, 249,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvrdpim $XT, $XB", IIC_VecFP,
                      [(set v2f64:$XT, (ffloor v2f64:$XB))]>;
  def XVRDPIP : XX2Form<60, 233,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvrdpip $XT, $XB", IIC_VecFP,
                      [(set v2f64:$XT, (fceil v2f64:$XB))]>;
  def XVRDPIZ : XX2Form<60, 217,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvrdpiz $XT, $XB", IIC_VecFP,
                      [(set v2f64:$XT, (ftrunc v2f64:$XB))]>;

  def XVRSPI : XX2Form<60, 137,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvrspi $XT, $XB", IIC_VecFP,
                      [(set v4f32:$XT, (frnd v4f32:$XB))]>;
  def XVRSPIC : XX2Form<60, 171,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvrspic $XT, $XB", IIC_VecFP,
                      [(set v4f32:$XT, (fnearbyint v4f32:$XB))]>;
  def XVRSPIM : XX2Form<60, 185,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvrspim $XT, $XB", IIC_VecFP,
                      [(set v4f32:$XT, (ffloor v4f32:$XB))]>;
  def XVRSPIP : XX2Form<60, 169,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvrspip $XT, $XB", IIC_VecFP,
                      [(set v4f32:$XT, (fceil v4f32:$XB))]>;
  def XVRSPIZ : XX2Form<60, 153,
                      (outs vsrc:$XT), (ins vsrc:$XB),
                      "xvrspiz $XT, $XB", IIC_VecFP,
                      [(set v4f32:$XT, (ftrunc v4f32:$XB))]>;

  // Max/Min Instructions
  let isCommutable = 1 in {
  def XSMAXDP : XX3Form<60, 160,
                        (outs vsfrc:$XT), (ins vsfrc:$XA, vsfrc:$XB),
                        "xsmaxdp $XT, $XA, $XB", IIC_VecFP,
                        [(set vsfrc:$XT,
                              (int_ppc_vsx_xsmaxdp vsfrc:$XA, vsfrc:$XB))]>;
  def XSMINDP : XX3Form<60, 168,
                        (outs vsfrc:$XT), (ins vsfrc:$XA, vsfrc:$XB),
                        "xsmindp $XT, $XA, $XB", IIC_VecFP,
                        [(set vsfrc:$XT,
                              (int_ppc_vsx_xsmindp vsfrc:$XA, vsfrc:$XB))]>;

  def XVMAXDP : XX3Form<60, 224,
                        (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                        "xvmaxdp $XT, $XA, $XB", IIC_VecFP,
                        [(set vsrc:$XT,
                              (int_ppc_vsx_xvmaxdp vsrc:$XA, vsrc:$XB))]>;
  def XVMINDP : XX3Form<60, 232,
                        (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                        "xvmindp $XT, $XA, $XB", IIC_VecFP,
                        [(set vsrc:$XT,
                              (int_ppc_vsx_xvmindp vsrc:$XA, vsrc:$XB))]>;

  def XVMAXSP : XX3Form<60, 192,
                        (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                        "xvmaxsp $XT, $XA, $XB", IIC_VecFP,
                        [(set vsrc:$XT,
                              (int_ppc_vsx_xvmaxsp vsrc:$XA, vsrc:$XB))]>;
  def XVMINSP : XX3Form<60, 200,
                        (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                        "xvminsp $XT, $XA, $XB", IIC_VecFP,
                        [(set vsrc:$XT,
                              (int_ppc_vsx_xvminsp vsrc:$XA, vsrc:$XB))]>;
  } // isCommutable
} // Uses = [RM]

  // Logical Instructions
  let isCommutable = 1 in
  def XXLAND : XX3Form<60, 130,
                       (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                       "xxland $XT, $XA, $XB", IIC_VecGeneral,
                       [(set v4i32:$XT, (and v4i32:$XA, v4i32:$XB))]>;
  def XXLANDC : XX3Form<60, 138,
                        (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                        "xxlandc $XT, $XA, $XB", IIC_VecGeneral,
                        [(set v4i32:$XT, (and v4i32:$XA,
                                              (vnot_ppc v4i32:$XB)))]>;
  let isCommutable = 1 in {
  def XXLNOR : XX3Form<60, 162,
                       (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                       "xxlnor $XT, $XA, $XB", IIC_VecGeneral,
                       [(set v4i32:$XT, (vnot_ppc (or v4i32:$XA,
                                                   v4i32:$XB)))]>;
  def XXLOR : XX3Form<60, 146,
                      (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                      "xxlor $XT, $XA, $XB", IIC_VecGeneral,
                      [(set v4i32:$XT, (or v4i32:$XA, v4i32:$XB))]>;
  let isCodeGenOnly = 1 in
  def XXLORf: XX3Form<60, 146,
                      (outs vsfrc:$XT), (ins vsfrc:$XA, vsfrc:$XB),
                      "xxlor $XT, $XA, $XB", IIC_VecGeneral, []>;
  def XXLXOR : XX3Form<60, 154,
                       (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                       "xxlxor $XT, $XA, $XB", IIC_VecGeneral,
                       [(set v4i32:$XT, (xor v4i32:$XA, v4i32:$XB))]>;
  } // isCommutable

  // Permutation Instructions
  def XXMRGHW : XX3Form<60, 18,
                       (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                       "xxmrghw $XT, $XA, $XB", IIC_VecPerm, []>;
  def XXMRGLW : XX3Form<60, 50,
                       (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                       "xxmrglw $XT, $XA, $XB", IIC_VecPerm, []>;

  def XXPERMDI : XX3Form_2<60, 10,
                       (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB, u2imm:$DM),
                       "xxpermdi $XT, $XA, $XB, $DM", IIC_VecPerm, []>;
  def XXSEL : XX4Form<60, 3,
                      (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB, vsrc:$XC),
                      "xxsel $XT, $XA, $XB, $XC", IIC_VecPerm, []>;

  def XXSLDWI : XX3Form_2<60, 2,
                       (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB, u2imm:$SHW),
                       "xxsldwi $XT, $XA, $XB, $SHW", IIC_VecPerm, []>;
  def XXSPLTW : XX2Form_2<60, 164,
                       (outs vsrc:$XT), (ins vsrc:$XB, u2imm:$UIM),
                       "xxspltw $XT, $XB, $UIM", IIC_VecPerm, []>;
} // hasSideEffects

// SELECT_CC_* - Used to implement the SELECT_CC DAG operation.  Expanded after
// instruction selection into a branch sequence.
let usesCustomInserter = 1,    // Expanded after instruction selection.
    PPC970_Single = 1 in {

  def SELECT_CC_VSRC: Pseudo<(outs vsrc:$dst),
                             (ins crrc:$cond, vsrc:$T, vsrc:$F, i32imm:$BROPC),
                             "#SELECT_CC_VSRC",
                             []>;
  def SELECT_VSRC: Pseudo<(outs vsrc:$dst),
                          (ins crbitrc:$cond, vsrc:$T, vsrc:$F),
                          "#SELECT_VSRC",
                          [(set v2f64:$dst,
                                (select i1:$cond, v2f64:$T, v2f64:$F))]>;
  def SELECT_CC_VSFRC: Pseudo<(outs f8rc:$dst),
                              (ins crrc:$cond, f8rc:$T, f8rc:$F,
                               i32imm:$BROPC), "#SELECT_CC_VSFRC",
                              []>;
  def SELECT_VSFRC: Pseudo<(outs f8rc:$dst),
                           (ins crbitrc:$cond, f8rc:$T, f8rc:$F),
                           "#SELECT_VSFRC",
                           [(set f64:$dst,
                                 (select i1:$cond, f64:$T, f64:$F))]>;
  def SELECT_CC_VSSRC: Pseudo<(outs f4rc:$dst),
                              (ins crrc:$cond, f4rc:$T, f4rc:$F,
                               i32imm:$BROPC), "#SELECT_CC_VSSRC",
                              []>;
  def SELECT_VSSRC: Pseudo<(outs f4rc:$dst),
                           (ins crbitrc:$cond, f4rc:$T, f4rc:$F),
                           "#SELECT_VSSRC",
                           [(set f32:$dst,
                                 (select i1:$cond, f32:$T, f32:$F))]>;
} // usesCustomInserter
} // AddedComplexity

def : InstAlias<"xvmovdp $XT, $XB",
                (XVCPSGNDP vsrc:$XT, vsrc:$XB, vsrc:$XB)>;
def : InstAlias<"xvmovsp $XT, $XB",
                (XVCPSGNSP vsrc:$XT, vsrc:$XB, vsrc:$XB)>;

def : InstAlias<"xxspltd $XT, $XB, 0",
                (XXPERMDI vsrc:$XT, vsrc:$XB, vsrc:$XB, 0)>;
def : InstAlias<"xxspltd $XT, $XB, 1",
                (XXPERMDI vsrc:$XT, vsrc:$XB, vsrc:$XB, 3)>;
def : InstAlias<"xxmrghd $XT, $XA, $XB",
                (XXPERMDI vsrc:$XT, vsrc:$XA, vsrc:$XB, 0)>;
def : InstAlias<"xxmrgld $XT, $XA, $XB",
                (XXPERMDI vsrc:$XT, vsrc:$XA, vsrc:$XB, 3)>;
def : InstAlias<"xxswapd $XT, $XB",
                (XXPERMDI vsrc:$XT, vsrc:$XB, vsrc:$XB, 2)>;

let AddedComplexity = 400 in { // Prefer VSX patterns over non-VSX patterns.

let Predicates = [IsBigEndian] in {
def : Pat<(v2f64 (scalar_to_vector f64:$A)),
          (v2f64 (SUBREG_TO_REG (i64 1), $A, sub_64))>;

def : Pat<(f64 (extractelt v2f64:$S, 0)),
          (f64 (EXTRACT_SUBREG $S, sub_64))>;
def : Pat<(f64 (extractelt v2f64:$S, 1)),
          (f64 (EXTRACT_SUBREG (XXPERMDI $S, $S, 2), sub_64))>;
}

let Predicates = [IsLittleEndian] in {
def : Pat<(v2f64 (scalar_to_vector f64:$A)),
          (v2f64 (XXPERMDI (SUBREG_TO_REG (i64 1), $A, sub_64),
                           (SUBREG_TO_REG (i64 1), $A, sub_64), 0))>;

def : Pat<(f64 (extractelt v2f64:$S, 0)),
          (f64 (EXTRACT_SUBREG (XXPERMDI $S, $S, 2), sub_64))>;
def : Pat<(f64 (extractelt v2f64:$S, 1)),
          (f64 (EXTRACT_SUBREG $S, sub_64))>;
}

// Additional fnmsub patterns: -a*c + b == -(a*c - b)
def : Pat<(fma (fneg f64:$A), f64:$C, f64:$B),
          (XSNMSUBADP $B, $C, $A)>;
def : Pat<(fma f64:$A, (fneg f64:$C), f64:$B),
          (XSNMSUBADP $B, $C, $A)>;

def : Pat<(fma (fneg v2f64:$A), v2f64:$C, v2f64:$B),
          (XVNMSUBADP $B, $C, $A)>;
def : Pat<(fma v2f64:$A, (fneg v2f64:$C), v2f64:$B),
          (XVNMSUBADP $B, $C, $A)>;

def : Pat<(fma (fneg v4f32:$A), v4f32:$C, v4f32:$B),
          (XVNMSUBASP $B, $C, $A)>;
def : Pat<(fma v4f32:$A, (fneg v4f32:$C), v4f32:$B),
          (XVNMSUBASP $B, $C, $A)>;

def : Pat<(v2f64 (bitconvert v4f32:$A)),
          (COPY_TO_REGCLASS $A, VSRC)>;
def : Pat<(v2f64 (bitconvert v4i32:$A)),
          (COPY_TO_REGCLASS $A, VSRC)>;
def : Pat<(v2f64 (bitconvert v8i16:$A)),
          (COPY_TO_REGCLASS $A, VSRC)>;
def : Pat<(v2f64 (bitconvert v16i8:$A)),
          (COPY_TO_REGCLASS $A, VSRC)>;

def : Pat<(v4f32 (bitconvert v2f64:$A)),
          (COPY_TO_REGCLASS $A, VRRC)>;
def : Pat<(v4i32 (bitconvert v2f64:$A)),
          (COPY_TO_REGCLASS $A, VRRC)>;
def : Pat<(v8i16 (bitconvert v2f64:$A)),
          (COPY_TO_REGCLASS $A, VRRC)>;
def : Pat<(v16i8 (bitconvert v2f64:$A)),
          (COPY_TO_REGCLASS $A, VRRC)>;

def : Pat<(v2i64 (bitconvert v4f32:$A)),
          (COPY_TO_REGCLASS $A, VSRC)>;
def : Pat<(v2i64 (bitconvert v4i32:$A)),
          (COPY_TO_REGCLASS $A, VSRC)>;
def : Pat<(v2i64 (bitconvert v8i16:$A)),
          (COPY_TO_REGCLASS $A, VSRC)>;
def : Pat<(v2i64 (bitconvert v16i8:$A)),
          (COPY_TO_REGCLASS $A, VSRC)>;

def : Pat<(v4f32 (bitconvert v2i64:$A)),
          (COPY_TO_REGCLASS $A, VRRC)>;
def : Pat<(v4i32 (bitconvert v2i64:$A)),
          (COPY_TO_REGCLASS $A, VRRC)>;
def : Pat<(v8i16 (bitconvert v2i64:$A)),
          (COPY_TO_REGCLASS $A, VRRC)>;
def : Pat<(v16i8 (bitconvert v2i64:$A)),
          (COPY_TO_REGCLASS $A, VRRC)>;

def : Pat<(v2f64 (bitconvert v2i64:$A)),
          (COPY_TO_REGCLASS $A, VRRC)>;
def : Pat<(v2i64 (bitconvert v2f64:$A)),
          (COPY_TO_REGCLASS $A, VRRC)>;

def : Pat<(v2f64 (bitconvert v1i128:$A)),
          (COPY_TO_REGCLASS $A, VRRC)>;
def : Pat<(v1i128 (bitconvert v2f64:$A)),
          (COPY_TO_REGCLASS $A, VRRC)>;

// sign extension patterns
// To extend "in place" from v2i32 to v2i64, we have input data like:
// | undef | i32 | undef | i32 |
// but xvcvsxwdp expects the input in big-Endian format:
// | i32 | undef | i32 | undef |
// so we need to shift everything to the left by one i32 (word) before
// the conversion.
def : Pat<(sext_inreg v2i64:$C, v2i32),
          (XVCVDPSXDS (XVCVSXWDP (XXSLDWI $C, $C, 1)))>;
def : Pat<(v2f64 (sint_to_fp (sext_inreg v2i64:$C, v2i32))),
          (XVCVSXWDP (XXSLDWI $C, $C, 1))>;

// Loads.
def : Pat<(v2f64 (load xoaddr:$src)), (LXVD2X xoaddr:$src)>;
def : Pat<(v2i64 (load xoaddr:$src)), (LXVD2X xoaddr:$src)>;
def : Pat<(v4i32 (load xoaddr:$src)), (LXVW4X xoaddr:$src)>;
def : Pat<(v2f64 (PPClxvd2x xoaddr:$src)), (LXVD2X xoaddr:$src)>;

// Stores.
def : Pat<(int_ppc_vsx_stxvd2x v2f64:$rS, xoaddr:$dst),
          (STXVD2X $rS, xoaddr:$dst)>;
def : Pat<(store v2i64:$rS, xoaddr:$dst), (STXVD2X $rS, xoaddr:$dst)>;
def : Pat<(int_ppc_vsx_stxvw4x v4i32:$rS, xoaddr:$dst),
          (STXVW4X $rS, xoaddr:$dst)>;
def : Pat<(PPCstxvd2x v2f64:$rS, xoaddr:$dst), (STXVD2X $rS, xoaddr:$dst)>;

// Permutes.
def : Pat<(v2f64 (PPCxxswapd v2f64:$src)), (XXPERMDI $src, $src, 2)>;
def : Pat<(v2i64 (PPCxxswapd v2i64:$src)), (XXPERMDI $src, $src, 2)>;
def : Pat<(v4f32 (PPCxxswapd v4f32:$src)), (XXPERMDI $src, $src, 2)>;
def : Pat<(v4i32 (PPCxxswapd v4i32:$src)), (XXPERMDI $src, $src, 2)>;

// Selects.
def : Pat<(v2f64 (selectcc i1:$lhs, i1:$rhs, v2f64:$tval, v2f64:$fval, SETLT)),
          (SELECT_VSRC (CRANDC $lhs, $rhs), $tval, $fval)>;
def : Pat<(v2f64 (selectcc i1:$lhs, i1:$rhs, v2f64:$tval, v2f64:$fval, SETULT)),
          (SELECT_VSRC (CRANDC $rhs, $lhs), $tval, $fval)>;
def : Pat<(v2f64 (selectcc i1:$lhs, i1:$rhs, v2f64:$tval, v2f64:$fval, SETLE)),
          (SELECT_VSRC (CRORC  $lhs, $rhs), $tval, $fval)>;
def : Pat<(v2f64 (selectcc i1:$lhs, i1:$rhs, v2f64:$tval, v2f64:$fval, SETULE)),
          (SELECT_VSRC (CRORC  $rhs, $lhs), $tval, $fval)>;
def : Pat<(v2f64 (selectcc i1:$lhs, i1:$rhs, v2f64:$tval, v2f64:$fval, SETEQ)),
          (SELECT_VSRC (CREQV $lhs, $rhs), $tval, $fval)>;
def : Pat<(v2f64 (selectcc i1:$lhs, i1:$rhs, v2f64:$tval, v2f64:$fval, SETGE)),
          (SELECT_VSRC (CRORC  $rhs, $lhs), $tval, $fval)>;
def : Pat<(v2f64 (selectcc i1:$lhs, i1:$rhs, v2f64:$tval, v2f64:$fval, SETUGE)),
          (SELECT_VSRC (CRORC  $lhs, $rhs), $tval, $fval)>;
def : Pat<(v2f64 (selectcc i1:$lhs, i1:$rhs, v2f64:$tval, v2f64:$fval, SETGT)),
          (SELECT_VSRC (CRANDC $rhs, $lhs), $tval, $fval)>;
def : Pat<(v2f64 (selectcc i1:$lhs, i1:$rhs, v2f64:$tval, v2f64:$fval, SETUGT)),
          (SELECT_VSRC (CRANDC $lhs, $rhs), $tval, $fval)>;
def : Pat<(v2f64 (selectcc i1:$lhs, i1:$rhs, v2f64:$tval, v2f64:$fval, SETNE)),
          (SELECT_VSRC (CRXOR $lhs, $rhs), $tval, $fval)>;

def : Pat<(f64 (selectcc i1:$lhs, i1:$rhs, f64:$tval, f64:$fval, SETLT)),
          (SELECT_VSFRC (CRANDC $lhs, $rhs), $tval, $fval)>;
def : Pat<(f64 (selectcc i1:$lhs, i1:$rhs, f64:$tval, f64:$fval, SETULT)),
          (SELECT_VSFRC (CRANDC $rhs, $lhs), $tval, $fval)>;
def : Pat<(f64 (selectcc i1:$lhs, i1:$rhs, f64:$tval, f64:$fval, SETLE)),
          (SELECT_VSFRC (CRORC  $lhs, $rhs), $tval, $fval)>;
def : Pat<(f64 (selectcc i1:$lhs, i1:$rhs, f64:$tval, f64:$fval, SETULE)),
          (SELECT_VSFRC (CRORC  $rhs, $lhs), $tval, $fval)>;
def : Pat<(f64 (selectcc i1:$lhs, i1:$rhs, f64:$tval, f64:$fval, SETEQ)),
          (SELECT_VSFRC (CREQV $lhs, $rhs), $tval, $fval)>;
def : Pat<(f64 (selectcc i1:$lhs, i1:$rhs, f64:$tval, f64:$fval, SETGE)),
          (SELECT_VSFRC (CRORC  $rhs, $lhs), $tval, $fval)>;
def : Pat<(f64 (selectcc i1:$lhs, i1:$rhs, f64:$tval, f64:$fval, SETUGE)),
          (SELECT_VSFRC (CRORC  $lhs, $rhs), $tval, $fval)>;
def : Pat<(f64 (selectcc i1:$lhs, i1:$rhs, f64:$tval, f64:$fval, SETGT)),
          (SELECT_VSFRC (CRANDC $rhs, $lhs), $tval, $fval)>;
def : Pat<(f64 (selectcc i1:$lhs, i1:$rhs, f64:$tval, f64:$fval, SETUGT)),
          (SELECT_VSFRC (CRANDC $lhs, $rhs), $tval, $fval)>;
def : Pat<(f64 (selectcc i1:$lhs, i1:$rhs, f64:$tval, f64:$fval, SETNE)),
          (SELECT_VSFRC (CRXOR $lhs, $rhs), $tval, $fval)>;

// Divides.
def : Pat<(int_ppc_vsx_xvdivsp v4f32:$A, v4f32:$B),
          (XVDIVSP $A, $B)>;
def : Pat<(int_ppc_vsx_xvdivdp v2f64:$A, v2f64:$B),
          (XVDIVDP $A, $B)>;

// Reciprocal estimate
def : Pat<(int_ppc_vsx_xvresp v4f32:$A),
          (XVRESP $A)>;
def : Pat<(int_ppc_vsx_xvredp v2f64:$A),
          (XVREDP $A)>;

// Recip. square root estimate
def : Pat<(int_ppc_vsx_xvrsqrtesp v4f32:$A),
          (XVRSQRTESP $A)>;
def : Pat<(int_ppc_vsx_xvrsqrtedp v2f64:$A),
          (XVRSQRTEDP $A)>;

} // AddedComplexity
} // HasVSX

// The following VSX instructions were introduced in Power ISA 2.07
/* FIXME: if the operands are v2i64, these patterns will not match.
   we should define new patterns or otherwise match the same patterns
   when the elements are larger than i32.
*/
def HasP8Vector : Predicate<"PPCSubTarget->hasP8Vector()">;
def HasDirectMove : Predicate<"PPCSubTarget->hasDirectMove()">;
let Predicates = [HasP8Vector] in {
let AddedComplexity = 400 in { // Prefer VSX patterns over non-VSX patterns.
  let isCommutable = 1 in {
    def XXLEQV : XX3Form<60, 186,
                         (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                         "xxleqv $XT, $XA, $XB", IIC_VecGeneral,
                         [(set v4i32:$XT, (vnot_ppc (xor v4i32:$XA, v4i32:$XB)))]>;
    def XXLNAND : XX3Form<60, 178,
                          (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                          "xxlnand $XT, $XA, $XB", IIC_VecGeneral,
                          [(set v4i32:$XT, (vnot_ppc (and v4i32:$XA,
                                                    v4i32:$XB)))]>;
  } // isCommutable

  def : Pat<(int_ppc_vsx_xxleqv v4i32:$A, v4i32:$B),
            (XXLEQV $A, $B)>;

  def XXLORC : XX3Form<60, 170,
                       (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                       "xxlorc $XT, $XA, $XB", IIC_VecGeneral,
                       [(set v4i32:$XT, (or v4i32:$XA, (vnot_ppc v4i32:$XB)))]>;

  // VSX scalar loads introduced in ISA 2.07
  let mayLoad = 1 in {
    def LXSSPX : XX1Form<31, 524, (outs vssrc:$XT), (ins memrr:$src),
                         "lxsspx $XT, $src", IIC_LdStLFD,
                         [(set f32:$XT, (load xoaddr:$src))]>;
    def LXSIWAX : XX1Form<31, 76, (outs vsfrc:$XT), (ins memrr:$src),
                          "lxsiwax $XT, $src", IIC_LdStLFD,
                          [(set f64:$XT, (PPClfiwax xoaddr:$src))]>;
    def LXSIWZX : XX1Form<31, 12, (outs vsfrc:$XT), (ins memrr:$src),
                          "lxsiwzx $XT, $src", IIC_LdStLFD,
                          [(set f64:$XT, (PPClfiwzx xoaddr:$src))]>;
  } // mayLoad

  // VSX scalar stores introduced in ISA 2.07
  let mayStore = 1 in {
    def STXSSPX : XX1Form<31, 652, (outs), (ins vssrc:$XT, memrr:$dst),
                          "stxsspx $XT, $dst", IIC_LdStSTFD,
                          [(store f32:$XT, xoaddr:$dst)]>;
    def STXSIWX : XX1Form<31, 140, (outs), (ins vsfrc:$XT, memrr:$dst),
                          "stxsiwx $XT, $dst", IIC_LdStSTFD,
                          [(PPCstfiwx f64:$XT, xoaddr:$dst)]>;
  } // mayStore

  def : Pat<(f64 (extloadf32 xoaddr:$src)),
            (COPY_TO_REGCLASS (LXSSPX xoaddr:$src), VSFRC)>;
  def : Pat<(f64 (fextend f32:$src)),
            (COPY_TO_REGCLASS $src, VSFRC)>;

  def : Pat<(f32 (selectcc i1:$lhs, i1:$rhs, f32:$tval, f32:$fval, SETLT)),
            (SELECT_VSSRC (CRANDC $lhs, $rhs), $tval, $fval)>;
  def : Pat<(f32 (selectcc i1:$lhs, i1:$rhs, f32:$tval, f32:$fval, SETULT)),
            (SELECT_VSSRC (CRANDC $rhs, $lhs), $tval, $fval)>;
  def : Pat<(f32 (selectcc i1:$lhs, i1:$rhs, f32:$tval, f32:$fval, SETLE)),
            (SELECT_VSSRC (CRORC  $lhs, $rhs), $tval, $fval)>;
  def : Pat<(f32 (selectcc i1:$lhs, i1:$rhs, f32:$tval, f32:$fval, SETULE)),
            (SELECT_VSSRC (CRORC  $rhs, $lhs), $tval, $fval)>;
  def : Pat<(f32 (selectcc i1:$lhs, i1:$rhs, f32:$tval, f32:$fval, SETEQ)),
            (SELECT_VSSRC (CREQV $lhs, $rhs), $tval, $fval)>;
  def : Pat<(f32 (selectcc i1:$lhs, i1:$rhs, f32:$tval, f32:$fval, SETGE)),
            (SELECT_VSSRC (CRORC  $rhs, $lhs), $tval, $fval)>;
  def : Pat<(f32 (selectcc i1:$lhs, i1:$rhs, f32:$tval, f32:$fval, SETUGE)),
            (SELECT_VSSRC (CRORC  $lhs, $rhs), $tval, $fval)>;
  def : Pat<(f32 (selectcc i1:$lhs, i1:$rhs, f32:$tval, f32:$fval, SETGT)),
            (SELECT_VSSRC (CRANDC $rhs, $lhs), $tval, $fval)>;
  def : Pat<(f32 (selectcc i1:$lhs, i1:$rhs, f32:$tval, f32:$fval, SETUGT)),
            (SELECT_VSSRC (CRANDC $lhs, $rhs), $tval, $fval)>;
  def : Pat<(f32 (selectcc i1:$lhs, i1:$rhs, f32:$tval, f32:$fval, SETNE)),
            (SELECT_VSSRC (CRXOR $lhs, $rhs), $tval, $fval)>;

  // VSX Elementary Scalar FP arithmetic (SP)
  let isCommutable = 1 in {
    def XSADDSP : XX3Form<60, 0,
                          (outs vssrc:$XT), (ins vssrc:$XA, vssrc:$XB),
                          "xsaddsp $XT, $XA, $XB", IIC_VecFP,
                          [(set f32:$XT, (fadd f32:$XA, f32:$XB))]>;
    def XSMULSP : XX3Form<60, 16,
                          (outs vssrc:$XT), (ins vssrc:$XA, vssrc:$XB),
                          "xsmulsp $XT, $XA, $XB", IIC_VecFP,
                          [(set f32:$XT, (fmul f32:$XA, f32:$XB))]>;
  } // isCommutable

  def XSDIVSP : XX3Form<60, 24,
                        (outs vssrc:$XT), (ins vssrc:$XA, vssrc:$XB),
                        "xsdivsp $XT, $XA, $XB", IIC_FPDivS,
                        [(set f32:$XT, (fdiv f32:$XA, f32:$XB))]>;
  def XSRESP : XX2Form<60, 26,
                        (outs vssrc:$XT), (ins vssrc:$XB),
                        "xsresp $XT, $XB", IIC_VecFP,
                        [(set f32:$XT, (PPCfre f32:$XB))]>;
  def XSSQRTSP : XX2Form<60, 11,
                        (outs vssrc:$XT), (ins vssrc:$XB),
                        "xssqrtsp $XT, $XB", IIC_FPSqrtS,
                        [(set f32:$XT, (fsqrt f32:$XB))]>;
  def XSRSQRTESP : XX2Form<60, 10,
                           (outs vssrc:$XT), (ins vssrc:$XB),
                           "xsrsqrtesp $XT, $XB", IIC_VecFP,
                           [(set f32:$XT, (PPCfrsqrte f32:$XB))]>;
  def XSSUBSP : XX3Form<60, 8,
                        (outs vssrc:$XT), (ins vssrc:$XA, vssrc:$XB),
                        "xssubsp $XT, $XA, $XB", IIC_VecFP,
                        [(set f32:$XT, (fsub f32:$XA, f32:$XB))]>;

  // FMA Instructions
  let BaseName = "XSMADDASP" in {
  let isCommutable = 1 in
  def XSMADDASP : XX3Form<60, 1,
                          (outs vssrc:$XT),
                          (ins vssrc:$XTi, vssrc:$XA, vssrc:$XB),
                          "xsmaddasp $XT, $XA, $XB", IIC_VecFP,
                          [(set f32:$XT, (fma f32:$XA, f32:$XB, f32:$XTi))]>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  let IsVSXFMAAlt = 1 in
  def XSMADDMSP : XX3Form<60, 9,
                          (outs vssrc:$XT),
                          (ins vssrc:$XTi, vssrc:$XA, vssrc:$XB),
                          "xsmaddmsp $XT, $XA, $XB", IIC_VecFP, []>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  }

  let BaseName = "XSMSUBASP" in {
  let isCommutable = 1 in
  def XSMSUBASP : XX3Form<60, 17,
                          (outs vssrc:$XT),
                          (ins vssrc:$XTi, vssrc:$XA, vssrc:$XB),
                          "xsmsubasp $XT, $XA, $XB", IIC_VecFP,
                          [(set f32:$XT, (fma f32:$XA, f32:$XB,
                                              (fneg f32:$XTi)))]>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  let IsVSXFMAAlt = 1 in
  def XSMSUBMSP : XX3Form<60, 25,
                          (outs vssrc:$XT),
                          (ins vssrc:$XTi, vssrc:$XA, vssrc:$XB),
                          "xsmsubmsp $XT, $XA, $XB", IIC_VecFP, []>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  }

  let BaseName = "XSNMADDASP" in {
  let isCommutable = 1 in
  def XSNMADDASP : XX3Form<60, 129,
                          (outs vssrc:$XT),
                          (ins vssrc:$XTi, vssrc:$XA, vssrc:$XB),
                          "xsnmaddasp $XT, $XA, $XB", IIC_VecFP,
                          [(set f32:$XT, (fneg (fma f32:$XA, f32:$XB,
                                                    f32:$XTi)))]>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  let IsVSXFMAAlt = 1 in
  def XSNMADDMSP : XX3Form<60, 137,
                          (outs vssrc:$XT),
                          (ins vssrc:$XTi, vssrc:$XA, vssrc:$XB),
                          "xsnmaddmsp $XT, $XA, $XB", IIC_VecFP, []>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  }

  let BaseName = "XSNMSUBASP" in {
  let isCommutable = 1 in
  def XSNMSUBASP : XX3Form<60, 145,
                          (outs vssrc:$XT),
                          (ins vssrc:$XTi, vssrc:$XA, vssrc:$XB),
                          "xsnmsubasp $XT, $XA, $XB", IIC_VecFP,
                          [(set f32:$XT, (fneg (fma f32:$XA, f32:$XB,
                                                    (fneg f32:$XTi))))]>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  let IsVSXFMAAlt = 1 in
  def XSNMSUBMSP : XX3Form<60, 153,
                          (outs vssrc:$XT),
                          (ins vssrc:$XTi, vssrc:$XA, vssrc:$XB),
                          "xsnmsubmsp $XT, $XA, $XB", IIC_VecFP, []>,
                          RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">,
                          AltVSXFMARel;
  }

  // Single Precision Conversions (FP <-> INT)
  def XSCVSXDSP : XX2Form<60, 312,
                      (outs vssrc:$XT), (ins vsfrc:$XB),
                      "xscvsxdsp $XT, $XB", IIC_VecFP,
                      [(set f32:$XT, (PPCfcfids f64:$XB))]>;
  def XSCVUXDSP : XX2Form<60, 296,
                      (outs vssrc:$XT), (ins vsfrc:$XB),
                      "xscvuxdsp $XT, $XB", IIC_VecFP,
                      [(set f32:$XT, (PPCfcfidus f64:$XB))]>;

  // Conversions between vector and scalar single precision
  def XSCVDPSPN : XX2Form<60, 267, (outs vsrc:$XT), (ins vssrc:$XB),
                          "xscvdpspn $XT, $XB", IIC_VecFP, []>;
  def XSCVSPDPN : XX2Form<60, 331, (outs vssrc:$XT), (ins vsrc:$XB),
                          "xscvspdpn $XT, $XB", IIC_VecFP, []>;

} // AddedComplexity = 400
} // HasP8Vector

let Predicates = [HasDirectMove] in {
  // VSX direct move instructions
  def MFVSRD : XX1_RS6_RD5_XO<31, 51, (outs g8rc:$rA), (ins vsfrc:$XT),
                              "mfvsrd $rA, $XT", IIC_VecGeneral,
                              [(set i64:$rA, (PPCmfvsr f64:$XT))]>,
      Requires<[In64BitMode]>;
  def MFVSRWZ : XX1_RS6_RD5_XO<31, 115, (outs gprc:$rA), (ins vsfrc:$XT),
                               "mfvsrwz $rA, $XT", IIC_VecGeneral,
                               [(set i32:$rA, (PPCmfvsr f64:$XT))]>;
  def MTVSRD : XX1_RS6_RD5_XO<31, 179, (outs vsfrc:$XT), (ins g8rc:$rA),
                              "mtvsrd $XT, $rA", IIC_VecGeneral,
                              [(set f64:$XT, (PPCmtvsra i64:$rA))]>,
      Requires<[In64BitMode]>;
  def MTVSRWA : XX1_RS6_RD5_XO<31, 211, (outs vsfrc:$XT), (ins gprc:$rA),
                               "mtvsrwa $XT, $rA", IIC_VecGeneral,
                               [(set f64:$XT, (PPCmtvsra i32:$rA))]>;
  def MTVSRWZ : XX1_RS6_RD5_XO<31, 243, (outs vsfrc:$XT), (ins gprc:$rA),
                               "mtvsrwz $XT, $rA", IIC_VecGeneral,
                               [(set f64:$XT, (PPCmtvsrz i32:$rA))]>;
} // HasDirectMove

let Predicates = [IsISA3_0, HasDirectMove] in {
  def MTVSRWS: XX1_RS6_RD5_XO<31, 403, (outs vsrc:$XT), (ins gprc:$rA),
                              "mtvsrws $XT, $rA", IIC_VecGeneral,
                              []>;

  def MTVSRDD: XX1Form<31, 435, (outs vsrc:$XT), (ins g8rc:$rA, g8rc:$rB),
                       "mtvsrdd $XT, $rA, $rB", IIC_VecGeneral,
                       []>, Requires<[In64BitMode]>;

  def MFVSRLD: XX1_RS6_RD5_XO<31, 307, (outs g8rc:$rA), (ins vsrc:$XT),
                              "mfvsrld $rA, $XT", IIC_VecGeneral,
                              []>, Requires<[In64BitMode]>;

} // IsISA3_0, HasDirectMove

/*  Direct moves of various widths from GPR's into VSR's. Each move lines
    the value up into element 0 (both BE and LE). Namely, entities smaller than
    a doubleword are shifted left and moved for BE. For LE, they're moved, then
    swapped to go into the least significant element of the VSR.
*/
def MovesToVSR {
  dag BE_BYTE_0 =
    (MTVSRD
      (RLDICR
        (INSERT_SUBREG (i64 (IMPLICIT_DEF)), $A, sub_32), 56, 7));
  dag BE_HALF_0 =
    (MTVSRD
      (RLDICR
        (INSERT_SUBREG (i64 (IMPLICIT_DEF)), $A, sub_32), 48, 15));
  dag BE_WORD_0 =
    (MTVSRD
      (RLDICR
        (INSERT_SUBREG (i64 (IMPLICIT_DEF)), $A, sub_32), 32, 31));
  dag BE_DWORD_0 = (MTVSRD $A);

  dag LE_MTVSRW = (MTVSRD (INSERT_SUBREG (i64 (IMPLICIT_DEF)), $A, sub_32));
  dag LE_WORD_1 = (v2i64 (INSERT_SUBREG (v2i64 (IMPLICIT_DEF)),
                                        LE_MTVSRW, sub_64));
  dag LE_WORD_0 = (XXPERMDI LE_WORD_1, LE_WORD_1, 2);
  dag LE_DWORD_1 = (v2i64 (INSERT_SUBREG (v2i64 (IMPLICIT_DEF)),
                                         BE_DWORD_0, sub_64));
  dag LE_DWORD_0 = (XXPERMDI LE_DWORD_1, LE_DWORD_1, 2);
}

/*  Patterns for extracting elements out of vectors. Integer elements are
    extracted using direct move operations. Patterns for extracting elements
    whose indices are not available at compile time are also provided with
    various _VARIABLE_ patterns.
    The numbering for the DAG's is for LE, but when used on BE, the correct
    LE element can just be used (i.e. LE_BYTE_2 == BE_BYTE_13).
*/
def VectorExtractions {
  // Doubleword extraction
  dag LE_DWORD_0 =
    (MFVSRD
      (EXTRACT_SUBREG
        (XXPERMDI (COPY_TO_REGCLASS $S, VSRC),
                  (COPY_TO_REGCLASS $S, VSRC), 2), sub_64));
  dag LE_DWORD_1 = (MFVSRD
                     (EXTRACT_SUBREG
                       (v2i64 (COPY_TO_REGCLASS $S, VSRC)), sub_64));

  // Word extraction
  dag LE_WORD_0 = (MFVSRWZ (EXTRACT_SUBREG (XXSLDWI $S, $S, 2), sub_64));
  dag LE_WORD_1 = (MFVSRWZ (EXTRACT_SUBREG (XXSLDWI $S, $S, 1), sub_64));
  dag LE_WORD_2 = (MFVSRWZ (EXTRACT_SUBREG
                             (v2i64 (COPY_TO_REGCLASS $S, VSRC)), sub_64));
  dag LE_WORD_3 = (MFVSRWZ (EXTRACT_SUBREG (XXSLDWI $S, $S, 3), sub_64));

  // Halfword extraction
  dag LE_HALF_0 = (i32 (EXTRACT_SUBREG (RLDICL LE_DWORD_0, 0, 48), sub_32));
  dag LE_HALF_1 = (i32 (EXTRACT_SUBREG (RLDICL LE_DWORD_0, 48, 48), sub_32));
  dag LE_HALF_2 = (i32 (EXTRACT_SUBREG (RLDICL LE_DWORD_0, 32, 48), sub_32));
  dag LE_HALF_3 = (i32 (EXTRACT_SUBREG (RLDICL LE_DWORD_0, 16, 48), sub_32));
  dag LE_HALF_4 = (i32 (EXTRACT_SUBREG (RLDICL LE_DWORD_1, 0, 48), sub_32));
  dag LE_HALF_5 = (i32 (EXTRACT_SUBREG (RLDICL LE_DWORD_1, 48, 48), sub_32));
  dag LE_HALF_6 = (i32 (EXTRACT_SUBREG (RLDICL LE_DWORD_1, 32, 48), sub_32));
  dag LE_HALF_7 = (i32 (EXTRACT_SUBREG (RLDICL LE_DWORD_1, 16, 48), sub_32));

  // Byte extraction
  dag LE_BYTE_0 = (i32 (EXTRACT_SUBREG (RLDICL LE_DWORD_0, 0, 56), sub_32));
  dag LE_BYTE_1 = (i32 (EXTRACT_SUBREG (RLDICL LE_DWORD_0, 56, 56), sub_32));
  dag LE_BYTE_2 = (i32 (EXTRACT_SUBREG (RLDICL LE_DWORD_0, 48, 56), sub_32));
  dag LE_BYTE_3 = (i32 (EXTRACT_SUBREG (RLDICL LE_DWORD_0, 40, 56), sub_32));
  dag LE_BYTE_4 = (i32 (EXTRACT_SUBREG (RLDICL LE_DWORD_0, 32, 56), sub_32));
  dag LE_BYTE_5 = (i32 (EXTRACT_SUBREG (RLDICL LE_DWORD_0, 24, 56), sub_32));
  dag LE_BYTE_6 = (i32 (EXTRACT_SUBREG (RLDICL LE_DWORD_0, 16, 56), sub_32));
  dag LE_BYTE_7 = (i32 (EXTRACT_SUBREG (RLDICL LE_DWORD_0, 8, 56), sub_32));
  dag LE_BYTE_8 = (i32 (EXTRACT_SUBREG (RLDICL LE_DWORD_1, 0, 56), sub_32));
  dag LE_BYTE_9 = (i32 (EXTRACT_SUBREG (RLDICL LE_DWORD_1, 56, 56), sub_32));
  dag LE_BYTE_10 = (i32 (EXTRACT_SUBREG (RLDICL LE_DWORD_1, 48, 56), sub_32));
  dag LE_BYTE_11 = (i32 (EXTRACT_SUBREG (RLDICL LE_DWORD_1, 40, 56), sub_32));
  dag LE_BYTE_12 = (i32 (EXTRACT_SUBREG (RLDICL LE_DWORD_1, 32, 56), sub_32));
  dag LE_BYTE_13 = (i32 (EXTRACT_SUBREG (RLDICL LE_DWORD_1, 24, 56), sub_32));
  dag LE_BYTE_14 = (i32 (EXTRACT_SUBREG (RLDICL LE_DWORD_1, 16, 56), sub_32));
  dag LE_BYTE_15 = (i32 (EXTRACT_SUBREG (RLDICL LE_DWORD_1, 8, 56), sub_32));

  /* Variable element number (BE and LE patterns must be specified separately)
     This is a rather involved process.

     Conceptually, this is how the move is accomplished:
     1. Identify which doubleword contains the element
     2. Shift in the VMX register so that the correct doubleword is correctly
        lined up for the MFVSRD
     3. Perform the move so that the element (along with some extra stuff)
        is in the GPR
     4. Right shift within the GPR so that the element is right-justified

     Of course, the index is an element number which has a different meaning
     on LE/BE so the patterns have to be specified separately.

     Note: The final result will be the element right-justified with high
           order bits being arbitrarily defined (namely, whatever was in the
           vector register to the left of the value originally).
  */

  /*  LE variable byte
      Number 1. above:
      - For elements 0-7, we shift left by 8 bytes since they're on the right
      - For elements 8-15, we need not shift (shift left by zero bytes)
      This is accomplished by inverting the bits of the index and AND-ing
      with 0x8 (i.e. clearing all bits of the index and inverting bit 60).
  */
  dag LE_VBYTE_PERM_VEC = (LVSL ZERO8, (ANDC8 (LI8 8), $Idx));

  //  Number 2. above:
  //  - Now that we set up the shift amount, we shift in the VMX register
  dag LE_VBYTE_PERMUTE = (VPERM $S, $S, LE_VBYTE_PERM_VEC);

  //  Number 3. above:
  //  - The doubleword containing our element is moved to a GPR
  dag LE_MV_VBYTE = (MFVSRD
                      (EXTRACT_SUBREG
                        (v2i64 (COPY_TO_REGCLASS LE_VBYTE_PERMUTE, VSRC)),
                        sub_64));

  /*  Number 4. above:
      - Truncate the element number to the range 0-7 (8-15 are symmetrical
        and out of range values are truncated accordingly)
      - Multiply by 8 as we need to shift right by the number of bits, not bytes
      - Shift right in the GPR by the calculated value
  */
  dag LE_VBYTE_SHIFT = (EXTRACT_SUBREG (RLDICR (AND8 (LI8 7), $Idx), 3, 60),
                                       sub_32);
  dag LE_VARIABLE_BYTE = (EXTRACT_SUBREG (SRD LE_MV_VBYTE, LE_VBYTE_SHIFT),
                                         sub_32);

  /*  LE variable halfword
      Number 1. above:
      - For elements 0-3, we shift left by 8 since they're on the right
      - For elements 4-7, we need not shift (shift left by zero bytes)
      Similarly to the byte pattern, we invert the bits of the index, but we
      AND with 0x4 (i.e. clear all bits of the index and invert bit 61).
      Of course, the shift is still by 8 bytes, so we must multiply by 2.
  */
  dag LE_VHALF_PERM_VEC = (LVSL ZERO8, (RLDICR (ANDC8 (LI8 4), $Idx), 1, 62));

  //  Number 2. above:
  //  - Now that we set up the shift amount, we shift in the VMX register
  dag LE_VHALF_PERMUTE = (VPERM $S, $S, LE_VHALF_PERM_VEC);

  //  Number 3. above:
  //  - The doubleword containing our element is moved to a GPR
  dag LE_MV_VHALF = (MFVSRD
                      (EXTRACT_SUBREG
                        (v2i64 (COPY_TO_REGCLASS LE_VHALF_PERMUTE, VSRC)),
                        sub_64));

  /*  Number 4. above:
      - Truncate the element number to the range 0-3 (4-7 are symmetrical
        and out of range values are truncated accordingly)
      - Multiply by 16 as we need to shift right by the number of bits
      - Shift right in the GPR by the calculated value
  */
  dag LE_VHALF_SHIFT = (EXTRACT_SUBREG (RLDICR (AND8 (LI8 3), $Idx), 4, 59),
                                       sub_32);
  dag LE_VARIABLE_HALF = (EXTRACT_SUBREG (SRD LE_MV_VHALF, LE_VHALF_SHIFT),
                                         sub_32);

  /*  LE variable word
      Number 1. above:
      - For elements 0-1, we shift left by 8 since they're on the right
      - For elements 2-3, we need not shift
  */
  dag LE_VWORD_PERM_VEC = (LVSL ZERO8, (RLDICR (ANDC8 (LI8 2), $Idx), 2, 61));

  //  Number 2. above:
  //  - Now that we set up the shift amount, we shift in the VMX register
  dag LE_VWORD_PERMUTE = (VPERM $S, $S, LE_VWORD_PERM_VEC);

  //  Number 3. above:
  //  - The doubleword containing our element is moved to a GPR
  dag LE_MV_VWORD = (MFVSRD
                      (EXTRACT_SUBREG
                        (v2i64 (COPY_TO_REGCLASS LE_VWORD_PERMUTE, VSRC)),
                        sub_64));

  /*  Number 4. above:
      - Truncate the element number to the range 0-1 (2-3 are symmetrical
        and out of range values are truncated accordingly)
      - Multiply by 32 as we need to shift right by the number of bits
      - Shift right in the GPR by the calculated value
  */
  dag LE_VWORD_SHIFT = (EXTRACT_SUBREG (RLDICR (AND8 (LI8 1), $Idx), 5, 58),
                                       sub_32);
  dag LE_VARIABLE_WORD = (EXTRACT_SUBREG (SRD LE_MV_VWORD, LE_VWORD_SHIFT),
                                         sub_32);

  /*  LE variable doubleword
      Number 1. above:
      - For element 0, we shift left by 8 since it's on the right
      - For element 1, we need not shift
  */
  dag LE_VDWORD_PERM_VEC = (LVSL ZERO8, (RLDICR (ANDC8 (LI8 1), $Idx), 3, 60));

  //  Number 2. above:
  //  - Now that we set up the shift amount, we shift in the VMX register
  dag LE_VDWORD_PERMUTE = (VPERM $S, $S, LE_VDWORD_PERM_VEC);

  // Number 3. above:
  //  - The doubleword containing our element is moved to a GPR
  //  - Number 4. is not needed for the doubleword as the value is 64-bits
  dag LE_VARIABLE_DWORD =
        (MFVSRD (EXTRACT_SUBREG
                  (v2i64 (COPY_TO_REGCLASS LE_VDWORD_PERMUTE, VSRC)),
                  sub_64));

  /*  LE variable float
      - Shift the vector to line up the desired element to BE Word 0
      - Convert 32-bit float to a 64-bit single precision float
  */
  dag LE_VFLOAT_PERM_VEC = (LVSL ZERO8, (RLDICR (XOR8 (LI8 3), $Idx), 2, 61));
  dag LE_VFLOAT_PERMUTE = (VPERM $S, $S, LE_VFLOAT_PERM_VEC);
  dag LE_VARIABLE_FLOAT = (XSCVSPDPN LE_VFLOAT_PERMUTE);

  /*  LE variable double
      Same as the LE doubleword except there is no move.
  */
  dag LE_VDOUBLE_PERMUTE = (VPERM (COPY_TO_REGCLASS $S, VRRC),
                                  (COPY_TO_REGCLASS $S, VRRC),
                                  LE_VDWORD_PERM_VEC);
  dag LE_VARIABLE_DOUBLE = (COPY_TO_REGCLASS LE_VDOUBLE_PERMUTE, VSRC);

  /*  BE variable byte
      The algorithm here is the same as the LE variable byte except:
      - The shift in the VMX register is by 0/8 for opposite element numbers so
        we simply AND the element number with 0x8
      - The order of elements after the move to GPR is reversed, so we invert
        the bits of the index prior to truncating to the range 0-7
  */
  dag BE_VBYTE_PERM_VEC = (LVSL ZERO8, (ANDIo8 $Idx, 8));
  dag BE_VBYTE_PERMUTE = (VPERM $S, $S, BE_VBYTE_PERM_VEC);
  dag BE_MV_VBYTE = (MFVSRD
                      (EXTRACT_SUBREG
                        (v2i64 (COPY_TO_REGCLASS BE_VBYTE_PERMUTE, VSRC)),
                        sub_64));
  dag BE_VBYTE_SHIFT = (EXTRACT_SUBREG (RLDICR (ANDC8 (LI8 7), $Idx), 3, 60),
                                       sub_32);
  dag BE_VARIABLE_BYTE = (EXTRACT_SUBREG (SRD BE_MV_VBYTE, BE_VBYTE_SHIFT),
                                         sub_32);

  /*  BE variable halfword
      The algorithm here is the same as the LE variable halfword except:
      - The shift in the VMX register is by 0/8 for opposite element numbers so
        we simply AND the element number with 0x4 and multiply by 2
      - The order of elements after the move to GPR is reversed, so we invert
        the bits of the index prior to truncating to the range 0-3
  */
  dag BE_VHALF_PERM_VEC = (LVSL ZERO8, (RLDICR (ANDIo8 $Idx, 4), 1, 62));
  dag BE_VHALF_PERMUTE = (VPERM $S, $S, BE_VHALF_PERM_VEC);
  dag BE_MV_VHALF = (MFVSRD
                      (EXTRACT_SUBREG
                        (v2i64 (COPY_TO_REGCLASS BE_VHALF_PERMUTE, VSRC)),
                        sub_64));
  dag BE_VHALF_SHIFT = (EXTRACT_SUBREG (RLDICR (ANDC8 (LI8 3), $Idx), 4, 59),
                                       sub_32);
  dag BE_VARIABLE_HALF = (EXTRACT_SUBREG (SRD BE_MV_VHALF, BE_VHALF_SHIFT),
                                         sub_32);

  /*  BE variable word
      The algorithm is the same as the LE variable word except:
      - The shift in the VMX register happens for opposite element numbers
      - The order of elements after the move to GPR is reversed, so we invert
        the bits of the index prior to truncating to the range 0-1
  */
  dag BE_VWORD_PERM_VEC = (LVSL ZERO8, (RLDICR (ANDIo8 $Idx, 2), 2, 61));
  dag BE_VWORD_PERMUTE = (VPERM $S, $S, BE_VWORD_PERM_VEC);
  dag BE_MV_VWORD = (MFVSRD
                      (EXTRACT_SUBREG
                        (v2i64 (COPY_TO_REGCLASS BE_VWORD_PERMUTE, VSRC)),
                        sub_64));
  dag BE_VWORD_SHIFT = (EXTRACT_SUBREG (RLDICR (ANDC8 (LI8 1), $Idx), 5, 58),
                                       sub_32);
  dag BE_VARIABLE_WORD = (EXTRACT_SUBREG (SRD BE_MV_VWORD, BE_VWORD_SHIFT),
                                         sub_32);

  /*  BE variable doubleword
      Same as the LE doubleword except we shift in the VMX register for opposite
      element indices.
  */
  dag BE_VDWORD_PERM_VEC = (LVSL ZERO8, (RLDICR (ANDIo8 $Idx, 1), 3, 60));
  dag BE_VDWORD_PERMUTE = (VPERM $S, $S, BE_VDWORD_PERM_VEC);
  dag BE_VARIABLE_DWORD =
        (MFVSRD (EXTRACT_SUBREG
                  (v2i64 (COPY_TO_REGCLASS BE_VDWORD_PERMUTE, VSRC)),
                  sub_64));

  /*  BE variable float
      - Shift the vector to line up the desired element to BE Word 0
      - Convert 32-bit float to a 64-bit single precision float
  */
  dag BE_VFLOAT_PERM_VEC = (LVSL ZERO8, (RLDICR $Idx, 2, 61));
  dag BE_VFLOAT_PERMUTE = (VPERM $S, $S, BE_VFLOAT_PERM_VEC);
  dag BE_VARIABLE_FLOAT = (XSCVSPDPN BE_VFLOAT_PERMUTE);

  /* BE variable double
      Same as the BE doubleword except there is no move.
  */
  dag BE_VDOUBLE_PERMUTE = (VPERM (COPY_TO_REGCLASS $S, VRRC),
                                  (COPY_TO_REGCLASS $S, VRRC),
                                  BE_VDWORD_PERM_VEC);
  dag BE_VARIABLE_DOUBLE = (COPY_TO_REGCLASS BE_VDOUBLE_PERMUTE, VSRC);
}

// v4f32 scalar <-> vector conversions (BE)
let Predicates = [IsBigEndian, HasP8Vector] in {
  def : Pat<(v4f32 (scalar_to_vector f32:$A)),
            (v4f32 (XSCVDPSPN $A))>;
  def : Pat<(f32 (vector_extract v4f32:$S, 0)),
            (f32 (XSCVSPDPN $S))>;
  def : Pat<(f32 (vector_extract v4f32:$S, 1)),
            (f32 (XSCVSPDPN (XXSLDWI $S, $S, 1)))>;
  def : Pat<(f32 (vector_extract v4f32:$S, 2)),
            (f32 (XSCVSPDPN (XXSLDWI $S, $S, 2)))>;
  def : Pat<(f32 (vector_extract v4f32:$S, 3)),
            (f32 (XSCVSPDPN (XXSLDWI $S, $S, 3)))>;
  def : Pat<(f32 (vector_extract v4f32:$S, i64:$Idx)),
            (f32 VectorExtractions.BE_VARIABLE_FLOAT)>;
} // IsBigEndian, HasP8Vector

// Variable index vector_extract for v2f64 does not require P8Vector
let Predicates = [IsBigEndian, HasVSX] in
  def : Pat<(f64 (vector_extract v2f64:$S, i64:$Idx)),
            (f64 VectorExtractions.BE_VARIABLE_DOUBLE)>;

let Predicates = [IsBigEndian, HasDirectMove] in {
  // v16i8 scalar <-> vector conversions (BE)
  def : Pat<(v16i8 (scalar_to_vector i32:$A)),
            (v16i8 (SUBREG_TO_REG (i64 1), MovesToVSR.BE_BYTE_0, sub_64))>;
  def : Pat<(v8i16 (scalar_to_vector i32:$A)),
            (v8i16 (SUBREG_TO_REG (i64 1), MovesToVSR.BE_HALF_0, sub_64))>;
  def : Pat<(v4i32 (scalar_to_vector i32:$A)),
            (v4i32 (SUBREG_TO_REG (i64 1), MovesToVSR.BE_WORD_0, sub_64))>;
  def : Pat<(v2i64 (scalar_to_vector i64:$A)),
            (v2i64 (SUBREG_TO_REG (i64 1), MovesToVSR.BE_DWORD_0, sub_64))>;
  def : Pat<(i32 (vector_extract v16i8:$S, 0)),
            (i32 VectorExtractions.LE_BYTE_15)>;
  def : Pat<(i32 (vector_extract v16i8:$S, 1)),
            (i32 VectorExtractions.LE_BYTE_14)>;
  def : Pat<(i32 (vector_extract v16i8:$S, 2)),
            (i32 VectorExtractions.LE_BYTE_13)>;
  def : Pat<(i32 (vector_extract v16i8:$S, 3)),
            (i32 VectorExtractions.LE_BYTE_12)>;
  def : Pat<(i32 (vector_extract v16i8:$S, 4)),
            (i32 VectorExtractions.LE_BYTE_11)>;
  def : Pat<(i32 (vector_extract v16i8:$S, 5)),
            (i32 VectorExtractions.LE_BYTE_10)>;
  def : Pat<(i32 (vector_extract v16i8:$S, 6)),
            (i32 VectorExtractions.LE_BYTE_9)>;
  def : Pat<(i32 (vector_extract v16i8:$S, 7)),
            (i32 VectorExtractions.LE_BYTE_8)>;
  def : Pat<(i32 (vector_extract v16i8:$S, 8)),
            (i32 VectorExtractions.LE_BYTE_7)>;
  def : Pat<(i32 (vector_extract v16i8:$S, 9)),
            (i32 VectorExtractions.LE_BYTE_6)>;
  def : Pat<(i32 (vector_extract v16i8:$S, 10)),
            (i32 VectorExtractions.LE_BYTE_5)>;
  def : Pat<(i32 (vector_extract v16i8:$S, 11)),
            (i32 VectorExtractions.LE_BYTE_4)>;
  def : Pat<(i32 (vector_extract v16i8:$S, 12)),
            (i32 VectorExtractions.LE_BYTE_3)>;
  def : Pat<(i32 (vector_extract v16i8:$S, 13)),
            (i32 VectorExtractions.LE_BYTE_2)>;
  def : Pat<(i32 (vector_extract v16i8:$S, 14)),
            (i32 VectorExtractions.LE_BYTE_1)>;
  def : Pat<(i32 (vector_extract v16i8:$S, 15)),
            (i32 VectorExtractions.LE_BYTE_0)>;
  def : Pat<(i32 (vector_extract v16i8:$S, i64:$Idx)),
            (i32 VectorExtractions.BE_VARIABLE_BYTE)>;

  // v8i16 scalar <-> vector conversions (BE)
  def : Pat<(i32 (vector_extract v8i16:$S, 0)),
            (i32 VectorExtractions.LE_HALF_7)>;
  def : Pat<(i32 (vector_extract v8i16:$S, 1)),
            (i32 VectorExtractions.LE_HALF_6)>;
  def : Pat<(i32 (vector_extract v8i16:$S, 2)),
            (i32 VectorExtractions.LE_HALF_5)>;
  def : Pat<(i32 (vector_extract v8i16:$S, 3)),
            (i32 VectorExtractions.LE_HALF_4)>;
  def : Pat<(i32 (vector_extract v8i16:$S, 4)),
            (i32 VectorExtractions.LE_HALF_3)>;
  def : Pat<(i32 (vector_extract v8i16:$S, 5)),
            (i32 VectorExtractions.LE_HALF_2)>;
  def : Pat<(i32 (vector_extract v8i16:$S, 6)),
            (i32 VectorExtractions.LE_HALF_1)>;
  def : Pat<(i32 (vector_extract v8i16:$S, 7)),
            (i32 VectorExtractions.LE_HALF_0)>;
  def : Pat<(i32 (vector_extract v8i16:$S, i64:$Idx)),
            (i32 VectorExtractions.BE_VARIABLE_HALF)>;

  // v4i32 scalar <-> vector conversions (BE)
  def : Pat<(i32 (vector_extract v4i32:$S, 0)),
            (i32 VectorExtractions.LE_WORD_3)>;
  def : Pat<(i32 (vector_extract v4i32:$S, 1)),
            (i32 VectorExtractions.LE_WORD_2)>;
  def : Pat<(i32 (vector_extract v4i32:$S, 2)),
            (i32 VectorExtractions.LE_WORD_1)>;
  def : Pat<(i32 (vector_extract v4i32:$S, 3)),
            (i32 VectorExtractions.LE_WORD_0)>;
  def : Pat<(i32 (vector_extract v4i32:$S, i64:$Idx)),
            (i32 VectorExtractions.BE_VARIABLE_WORD)>;

  // v2i64 scalar <-> vector conversions (BE)
  def : Pat<(i64 (vector_extract v2i64:$S, 0)),
            (i64 VectorExtractions.LE_DWORD_1)>;
  def : Pat<(i64 (vector_extract v2i64:$S, 1)),
            (i64 VectorExtractions.LE_DWORD_0)>;
  def : Pat<(i64 (vector_extract v2i64:$S, i64:$Idx)),
            (i64 VectorExtractions.BE_VARIABLE_DWORD)>;
} // IsBigEndian, HasDirectMove

// v4f32 scalar <-> vector conversions (LE)
let Predicates = [IsLittleEndian, HasP8Vector] in {
  def : Pat<(v4f32 (scalar_to_vector f32:$A)),
            (v4f32 (XXSLDWI (XSCVDPSPN $A), (XSCVDPSPN $A), 1))>;
  def : Pat<(f32 (vector_extract v4f32:$S, 0)),
            (f32 (XSCVSPDPN (XXSLDWI $S, $S, 3)))>;
  def : Pat<(f32 (vector_extract v4f32:$S, 1)),
            (f32 (XSCVSPDPN (XXSLDWI $S, $S, 2)))>;
  def : Pat<(f32 (vector_extract v4f32:$S, 2)),
            (f32 (XSCVSPDPN (XXSLDWI $S, $S, 1)))>;
  def : Pat<(f32 (vector_extract v4f32:$S, 3)),
            (f32 (XSCVSPDPN $S))>;
  def : Pat<(f32 (vector_extract v4f32:$S, i64:$Idx)),
            (f32 VectorExtractions.LE_VARIABLE_FLOAT)>;
} // IsLittleEndian, HasP8Vector

// Variable index vector_extract for v2f64 does not require P8Vector
let Predicates = [IsLittleEndian, HasVSX] in
  def : Pat<(f64 (vector_extract v2f64:$S, i64:$Idx)),
            (f64 VectorExtractions.LE_VARIABLE_DOUBLE)>;

let Predicates = [IsLittleEndian, HasDirectMove] in {
  // v16i8 scalar <-> vector conversions (LE)
  def : Pat<(v16i8 (scalar_to_vector i32:$A)),
            (v16i8 (COPY_TO_REGCLASS MovesToVSR.LE_WORD_0, VSRC))>;
  def : Pat<(v8i16 (scalar_to_vector i32:$A)),
            (v8i16 (COPY_TO_REGCLASS MovesToVSR.LE_WORD_0, VSRC))>;
  def : Pat<(v4i32 (scalar_to_vector i32:$A)),
            (v4i32 MovesToVSR.LE_WORD_0)>;
  def : Pat<(v2i64 (scalar_to_vector i64:$A)),
            (v2i64 MovesToVSR.LE_DWORD_0)>;
  def : Pat<(i32 (vector_extract v16i8:$S, 0)),
            (i32 VectorExtractions.LE_BYTE_0)>;
  def : Pat<(i32 (vector_extract v16i8:$S, 1)),
            (i32 VectorExtractions.LE_BYTE_1)>;
  def : Pat<(i32 (vector_extract v16i8:$S, 2)),
            (i32 VectorExtractions.LE_BYTE_2)>;
  def : Pat<(i32 (vector_extract v16i8:$S, 3)),
            (i32 VectorExtractions.LE_BYTE_3)>;
  def : Pat<(i32 (vector_extract v16i8:$S, 4)),
            (i32 VectorExtractions.LE_BYTE_4)>;
  def : Pat<(i32 (vector_extract v16i8:$S, 5)),
            (i32 VectorExtractions.LE_BYTE_5)>;
  def : Pat<(i32 (vector_extract v16i8:$S, 6)),
            (i32 VectorExtractions.LE_BYTE_6)>;
  def : Pat<(i32 (vector_extract v16i8:$S, 7)),
            (i32 VectorExtractions.LE_BYTE_7)>;
  def : Pat<(i32 (vector_extract v16i8:$S, 8)),
            (i32 VectorExtractions.LE_BYTE_8)>;
  def : Pat<(i32 (vector_extract v16i8:$S, 9)),
            (i32 VectorExtractions.LE_BYTE_9)>;
  def : Pat<(i32 (vector_extract v16i8:$S, 10)),
            (i32 VectorExtractions.LE_BYTE_10)>;
  def : Pat<(i32 (vector_extract v16i8:$S, 11)),
            (i32 VectorExtractions.LE_BYTE_11)>;
  def : Pat<(i32 (vector_extract v16i8:$S, 12)),
            (i32 VectorExtractions.LE_BYTE_12)>;
  def : Pat<(i32 (vector_extract v16i8:$S, 13)),
            (i32 VectorExtractions.LE_BYTE_13)>;
  def : Pat<(i32 (vector_extract v16i8:$S, 14)),
            (i32 VectorExtractions.LE_BYTE_14)>;
  def : Pat<(i32 (vector_extract v16i8:$S, 15)),
            (i32 VectorExtractions.LE_BYTE_15)>;
  def : Pat<(i32 (vector_extract v16i8:$S, i64:$Idx)),
            (i32 VectorExtractions.LE_VARIABLE_BYTE)>;

  // v8i16 scalar <-> vector conversions (LE)
  def : Pat<(i32 (vector_extract v8i16:$S, 0)),
            (i32 VectorExtractions.LE_HALF_0)>;
  def : Pat<(i32 (vector_extract v8i16:$S, 1)),
            (i32 VectorExtractions.LE_HALF_1)>;
  def : Pat<(i32 (vector_extract v8i16:$S, 2)),
            (i32 VectorExtractions.LE_HALF_2)>;
  def : Pat<(i32 (vector_extract v8i16:$S, 3)),
            (i32 VectorExtractions.LE_HALF_3)>;
  def : Pat<(i32 (vector_extract v8i16:$S, 4)),
            (i32 VectorExtractions.LE_HALF_4)>;
  def : Pat<(i32 (vector_extract v8i16:$S, 5)),
            (i32 VectorExtractions.LE_HALF_5)>;
  def : Pat<(i32 (vector_extract v8i16:$S, 6)),
            (i32 VectorExtractions.LE_HALF_6)>;
  def : Pat<(i32 (vector_extract v8i16:$S, 7)),
            (i32 VectorExtractions.LE_HALF_7)>;
  def : Pat<(i32 (vector_extract v8i16:$S, i64:$Idx)),
            (i32 VectorExtractions.LE_VARIABLE_HALF)>;

  // v4i32 scalar <-> vector conversions (LE)
  def : Pat<(i32 (vector_extract v4i32:$S, 0)),
            (i32 VectorExtractions.LE_WORD_0)>;
  def : Pat<(i32 (vector_extract v4i32:$S, 1)),
            (i32 VectorExtractions.LE_WORD_1)>;
  def : Pat<(i32 (vector_extract v4i32:$S, 2)),
            (i32 VectorExtractions.LE_WORD_2)>;
  def : Pat<(i32 (vector_extract v4i32:$S, 3)),
            (i32 VectorExtractions.LE_WORD_3)>;
  def : Pat<(i32 (vector_extract v4i32:$S, i64:$Idx)),
            (i32 VectorExtractions.LE_VARIABLE_WORD)>;

  // v2i64 scalar <-> vector conversions (LE)
  def : Pat<(i64 (vector_extract v2i64:$S, 0)),
            (i64 VectorExtractions.LE_DWORD_0)>;
  def : Pat<(i64 (vector_extract v2i64:$S, 1)),
            (i64 VectorExtractions.LE_DWORD_1)>;
  def : Pat<(i64 (vector_extract v2i64:$S, i64:$Idx)),
            (i64 VectorExtractions.LE_VARIABLE_DWORD)>;
} // IsLittleEndian, HasDirectMove

let Predicates = [HasDirectMove, HasVSX] in {
// bitconvert f32 -> i32
// (convert to 32-bit fp single, shift right 1 word, move to GPR)
def : Pat<(i32 (bitconvert f32:$S)),
          (i32 (MFVSRWZ (EXTRACT_SUBREG
                          (XXSLDWI (XSCVDPSPN $S),(XSCVDPSPN $S), 3),
                          sub_64)))>;
// bitconvert i32 -> f32
// (move to FPR, shift left 1 word, convert to 64-bit fp single)
def : Pat<(f32 (bitconvert i32:$A)),
          (f32 (XSCVSPDPN
                 (XXSLDWI MovesToVSR.LE_WORD_1, MovesToVSR.LE_WORD_1, 1)))>;

// bitconvert f64 -> i64
// (move to GPR, nothing else needed)
def : Pat<(i64 (bitconvert f64:$S)),
          (i64 (MFVSRD $S))>;

// bitconvert i64 -> f64
// (move to FPR, nothing else needed)
def : Pat<(f64 (bitconvert i64:$S)),
          (f64 (MTVSRD $S))>;
}

// The following VSX instructions were introduced in Power ISA 3.0
def HasP9Vector : Predicate<"PPCSubTarget->hasP9Vector()">;
let Predicates = [HasP9Vector] in {

  // [PO VRT XO VRB XO /]
  class X_VT5_XO5_VB5<bits<6> opcode, bits<5> xo2, bits<10> xo, string opc,
                      list<dag> pattern>
    : X_RD5_XO5_RS5<opcode, xo2, xo, (outs vrrc:$vT), (ins vrrc:$vB),
                    !strconcat(opc, " $vT, $vB"), IIC_VecFP, pattern>;

  // [PO VRT XO VRB XO RO], Round to Odd version of [PO VRT XO VRB XO /]
  class X_VT5_XO5_VB5_Ro<bits<6> opcode, bits<5> xo2, bits<10> xo, string opc,
                         list<dag> pattern>
    : X_VT5_XO5_VB5<opcode, xo2, xo, opc, pattern>, isDOT;

  // [PO VRT XO VRB XO /], but the VRB is only used the left 64 bits (or less),
  // So we use different operand class for VRB
  class X_VT5_XO5_VB5_TyVB<bits<6> opcode, bits<5> xo2, bits<10> xo, string opc,
                           RegisterOperand vbtype, list<dag> pattern>
    : X_RD5_XO5_RS5<opcode, xo2, xo, (outs vrrc:$vT), (ins vbtype:$vB),
                    !strconcat(opc, " $vT, $vB"), IIC_VecFP, pattern>;

  // [PO T XO B XO BX /]
  class XX2_RT5_XO5_XB6<bits<6> opcode, bits<5> xo2, bits<9> xo, string opc,
                        list<dag> pattern>
    : XX2_RD5_XO5_RS6<opcode, xo2, xo, (outs g8rc:$rT), (ins vsfrc:$XB),
                      !strconcat(opc, " $rT, $XB"), IIC_VecFP, pattern>;

  // [PO T XO B XO BX TX]
  class XX2_XT6_XO5_XB6<bits<6> opcode, bits<5> xo2, bits<9> xo, string opc,
                        RegisterOperand vtype, list<dag> pattern>
    : XX2_RD6_XO5_RS6<opcode, xo2, xo, (outs vtype:$XT), (ins vtype:$XB),
                      !strconcat(opc, " $XT, $XB"), IIC_VecFP, pattern>;

  // [PO T A B XO AX BX TX], src and dest register use different operand class
  class XX3_XT5_XA5_XB5<bits<6> opcode, bits<8> xo, string opc,
                  RegisterOperand xty, RegisterOperand aty, RegisterOperand bty,
                  InstrItinClass itin, list<dag> pattern>
    : XX3Form<opcode, xo, (outs xty:$XT), (ins aty:$XA, bty:$XB),
              !strconcat(opc, " $XT, $XA, $XB"), itin, pattern>;

  // [PO VRT VRA VRB XO /]
  class X_VT5_VA5_VB5<bits<6> opcode, bits<10> xo, string opc,
                      list<dag> pattern>
    : XForm_1<opcode, xo, (outs vrrc:$vT), (ins vrrc:$vA, vrrc:$vB),
              !strconcat(opc, " $vT, $vA, $vB"), IIC_VecFP, pattern>;

  // [PO VRT VRA VRB XO RO], Round to Odd version of [PO VRT VRA VRB XO /]
  class X_VT5_VA5_VB5_Ro<bits<6> opcode, bits<10> xo, string opc,
                         list<dag> pattern>
    : X_VT5_VA5_VB5<opcode, xo, opc, pattern>, isDOT;

  //===--------------------------------------------------------------------===//
  // Quad-Precision Scalar Move Instructions:

  // Copy Sign
  def XSCPSGNQP : X_VT5_VA5_VB5<63, 100, "xscpsgnqp", []>;

  // Absolute/Negative-Absolute/Negate
  def XSABSQP   : X_VT5_XO5_VB5<63,  0, 804, "xsabsqp" , []>;
  def XSNABSQP  : X_VT5_XO5_VB5<63,  8, 804, "xsnabsqp", []>;
  def XSNEGQP   : X_VT5_XO5_VB5<63, 16, 804, "xsnegqp" , []>;

  //===--------------------------------------------------------------------===//
  // Quad-Precision Scalar Floating-Point Arithmetic Instructions:

  // Add/Divide/Multiply/Subtract
  def XSADDQP   : X_VT5_VA5_VB5   <63,   4, "xsaddqp" , []>;
  def XSADDQPO  : X_VT5_VA5_VB5_Ro<63,   4, "xsaddqpo", []>;
  def XSDIVQP   : X_VT5_VA5_VB5   <63, 548, "xsdivqp" , []>;
  def XSDIVQPO  : X_VT5_VA5_VB5_Ro<63, 548, "xsdivqpo", []>;
  def XSMULQP   : X_VT5_VA5_VB5   <63,  36, "xsmulqp" , []>;
  def XSMULQPO  : X_VT5_VA5_VB5_Ro<63,  36, "xsmulqpo", []>;
  def XSSUBQP   : X_VT5_VA5_VB5   <63, 516, "xssubqp" , []>;
  def XSSUBQPO  : X_VT5_VA5_VB5_Ro<63, 516, "xssubqpo", []>;

  // Square-Root
  def XSSQRTQP  : X_VT5_XO5_VB5   <63, 27, 804, "xssqrtqp" , []>;
  def XSSQRTQPO : X_VT5_XO5_VB5_Ro<63, 27, 804, "xssqrtqpo", []>;

  // (Negative) Multiply-{Add/Subtract}
  def XSMADDQP  : X_VT5_VA5_VB5   <63, 388, "xsmaddqp"  , []>;
  def XSMADDQPO : X_VT5_VA5_VB5_Ro<63, 388, "xsmaddqpo" , []>;
  def XSMSUBQP  : X_VT5_VA5_VB5   <63, 420, "xsmsubqp"  , []>;
  def XSMSUBQPO : X_VT5_VA5_VB5_Ro<63, 420, "xsmsubqpo" , []>;
  def XSNMADDQP : X_VT5_VA5_VB5   <63, 452, "xsnmaddqp" , []>;
  def XSNMADDQPO: X_VT5_VA5_VB5_Ro<63, 452, "xsnmaddqpo", []>;
  def XSNMSUBQP : X_VT5_VA5_VB5   <63, 484, "xsnmsubqp" , []>;
  def XSNMSUBQPO: X_VT5_VA5_VB5_Ro<63, 484, "xsnmsubqpo", []>;

  //===--------------------------------------------------------------------===//
  // Quad/Double-Precision Compare Instructions:

  // [PO BF // VRA VRB XO /]
  class X_BF3_VA5_VB5<bits<6> opcode, bits<10> xo, string opc,
                      list<dag> pattern>
    : XForm_17<opcode, xo, (outs crrc:$crD), (ins vrrc:$VA, vrrc:$VB),
               !strconcat(opc, " $crD, $VA, $VB"), IIC_FPCompare> {
    let Pattern = pattern;
  }

  // QP Compare Ordered/Unordered
  def XSCMPOQP : X_BF3_VA5_VB5<63, 132, "xscmpoqp", []>;
  def XSCMPUQP : X_BF3_VA5_VB5<63, 644, "xscmpuqp", []>;

  // DP/QP Compare Exponents
  def XSCMPEXPDP : XX3Form_1<60, 59,
                             (outs crrc:$crD), (ins vsfrc:$XA, vsfrc:$XB),
                             "xscmpexpdp $crD, $XA, $XB", IIC_FPCompare, []>;
  def XSCMPEXPQP : X_BF3_VA5_VB5<63, 164, "xscmpexpqp", []>;

  // DP Compare ==, >=, >, !=
  // Use vsrc for XT, because the entire register of XT is set.
  // XT.dword[1] = 0x0000_0000_0000_0000
  def XSCMPEQDP : XX3_XT5_XA5_XB5<60,  3, "xscmpeqdp", vsrc, vsfrc, vsfrc,
                                  IIC_FPCompare, []>;
  def XSCMPGEDP : XX3_XT5_XA5_XB5<60, 19, "xscmpgedp", vsrc, vsfrc, vsfrc,
                                  IIC_FPCompare, []>;
  def XSCMPGTDP : XX3_XT5_XA5_XB5<60, 11, "xscmpgtdp", vsrc, vsfrc, vsfrc,
                                  IIC_FPCompare, []>;
  def XSCMPNEDP : XX3_XT5_XA5_XB5<60, 27, "xscmpnedp", vsrc, vsfrc, vsfrc,
                                  IIC_FPCompare, []>;
  // Vector Compare Not Equal
  def XVCMPNEDP  : XX3Form_Rc<60, 123,
                              (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                              "xvcmpnedp  $XT, $XA, $XB", IIC_VecFPCompare, []>;
  let Defs = [CR6] in
  def XVCMPNEDPo : XX3Form_Rc<60, 123,
                              (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                              "xvcmpnedp. $XT, $XA, $XB", IIC_VecFPCompare, []>,
                              isDOT;
  def XVCMPNESP  : XX3Form_Rc<60,  91,
                              (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                              "xvcmpnesp  $XT, $XA, $XB", IIC_VecFPCompare, []>;
  let Defs = [CR6] in
  def XVCMPNESPo : XX3Form_Rc<60,  91,
                              (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                              "xvcmpnesp. $XT, $XA, $XB", IIC_VecFPCompare, []>,
                              isDOT;

  //===--------------------------------------------------------------------===//
  // Quad-Precision Floating-Point Conversion Instructions:

  // Convert DP -> QP
  def XSCVDPQP  : X_VT5_XO5_VB5_TyVB<63, 22, 836, "xscvdpqp", vsfrc, []>;

  // Round & Convert QP -> DP (dword[1] is set to zero)
  def XSCVQPDP  : X_VT5_XO5_VB5   <63, 20, 836, "xscvqpdp" , []>;
  def XSCVQPDPO : X_VT5_XO5_VB5_Ro<63, 20, 836, "xscvqpdpo", []>;

  // Truncate & Convert QP -> (Un)Signed (D)Word (dword[1] is set to zero)
  def XSCVQPSDZ : X_VT5_XO5_VB5<63, 25, 836, "xscvqpsdz", []>;
  def XSCVQPSWZ : X_VT5_XO5_VB5<63,  9, 836, "xscvqpswz", []>;
  def XSCVQPUDZ : X_VT5_XO5_VB5<63, 17, 836, "xscvqpudz", []>;
  def XSCVQPUWZ : X_VT5_XO5_VB5<63,  1, 836, "xscvqpuwz", []>;

  // Convert (Un)Signed DWord -> QP
  def XSCVSDQP  : X_VT5_XO5_VB5_TyVB<63, 10, 836, "xscvsdqp", vsfrc, []>;
  def XSCVUDQP  : X_VT5_XO5_VB5_TyVB<63,  2, 836, "xscvudqp", vsfrc, []>;

  //===--------------------------------------------------------------------===//
  // Round to Floating-Point Integer Instructions

  // (Round &) Convert DP <-> HP
  // Note! xscvdphp's src and dest register both use the left 64 bits, so we use
  // vsfrc for src and dest register. xscvhpdp's src only use the left 16 bits,
  // but we still use vsfrc for it.
  def XSCVDPHP : XX2_XT6_XO5_XB6<60, 17, 347, "xscvdphp", vsfrc, []>;
  def XSCVHPDP : XX2_XT6_XO5_XB6<60, 16, 347, "xscvhpdp", vsfrc, []>;

  // Vector HP -> SP
  def XVCVHPSP : XX2_XT6_XO5_XB6<60, 24, 475, "xvcvhpsp", vsrc, []>;
  def XVCVSPHP : XX2_XT6_XO5_XB6<60, 25, 475, "xvcvsphp", vsrc, []>;

  class Z23_VT5_R1_VB5_RMC2_EX1<bits<6> opcode, bits<8> xo, bit ex, string opc,
                                list<dag> pattern>
    : Z23Form_1<opcode, xo,
                (outs vrrc:$vT), (ins u1imm:$r, vrrc:$vB, u2imm:$rmc),
                !strconcat(opc, " $r, $vT, $vB, $rmc"), IIC_VecFP, pattern> {
    let RC = ex;
  }

  // Round to Quad-Precision Integer [with Inexact]
  def XSRQPI   : Z23_VT5_R1_VB5_RMC2_EX1<63,  5, 0, "xsrqpi" , []>;
  def XSRQPIX  : Z23_VT5_R1_VB5_RMC2_EX1<63,  5, 1, "xsrqpix", []>;

  // Round Quad-Precision to Double-Extended Precision (fp80)
  def XSRQPXP  : Z23_VT5_R1_VB5_RMC2_EX1<63, 37, 0, "xsrqpxp", []>;

  //===--------------------------------------------------------------------===//
  // Insert/Extract Instructions

  // Insert Exponent DP/QP
  // XT NOTE: XT.dword[1] = 0xUUUU_UUUU_UUUU_UUUU
  def XSIEXPDP : XX1Form <60, 918, (outs vsrc:$XT), (ins g8rc:$rA, g8rc:$rB),
                          "xsiexpdp $XT, $rA, $rB", IIC_VecFP, []>;
  // vB NOTE: only vB.dword[0] is used, that's why we don't use
  //          X_VT5_VA5_VB5 form
  def XSIEXPQP : XForm_18<63, 868, (outs vrrc:$vT), (ins vrrc:$vA, vsfrc:$vB),
                          "xsiexpqp $vT, $vA, $vB", IIC_VecFP, []>;

  // Extract Exponent/Significand DP/QP
  def XSXEXPDP : XX2_RT5_XO5_XB6<60,  0, 347, "xsxexpdp", []>;
  def XSXSIGDP : XX2_RT5_XO5_XB6<60,  1, 347, "xsxsigdp", []>;
  def XSXEXPQP : X_VT5_XO5_VB5  <63,  2, 804, "xsxexpqp", []>;
  def XSXSIGQP : X_VT5_XO5_VB5  <63, 18, 804, "xsxsigqp", []>;

  // Vector Insert Word
  // XB NOTE: Only XB.dword[1] is used, but we use vsrc on XB.
  def XXINSERTW   : XX2_RD6_UIM5_RS6<60, 181,
                                  (outs vsrc:$XT), (ins u4imm:$UIMM, vsrc:$XB),
                                  "xxinsertw $XT, $XB, $UIMM", IIC_VecFP, []>;

  // Vector Extract Unsigned Word
  def XXEXTRACTUW : XX2_RD6_UIM5_RS6<60, 165,
                                  (outs vsrc:$XT), (ins u4imm:$UIMM, vsrc:$XB),
                                  "xxextractuw $XT, $XB, $UIMM", IIC_VecFP, []>;

  // Vector Insert Exponent DP/SP
  def XVIEXPDP : XX3_XT5_XA5_XB5<60, 248, "xviexpdp", vsrc, vsrc, vsrc,
                                 IIC_VecFP, []>;
  def XVIEXPSP : XX3_XT5_XA5_XB5<60, 216, "xviexpsp", vsrc, vsrc, vsrc,
                                 IIC_VecFP, []>;

  // Vector Extract Exponent/Significand DP/SP
  def XVXEXPDP : XX2_XT6_XO5_XB6<60,  0, 475, "xvxexpdp", vsrc, []>;
  def XVXEXPSP : XX2_XT6_XO5_XB6<60,  8, 475, "xvxexpsp", vsrc, []>;
  def XVXSIGDP : XX2_XT6_XO5_XB6<60,  1, 475, "xvxsigdp", vsrc, []>;
  def XVXSIGSP : XX2_XT6_XO5_XB6<60,  9, 475, "xvxsigsp", vsrc, []>;

  //===--------------------------------------------------------------------===//

  // Test Data Class SP/DP/QP
  def XSTSTDCSP : XX2_BF3_DCMX7_RS6<60, 298,
                              (outs crrc:$BF), (ins u7imm:$DCMX, vsfrc:$XB),
                              "xststdcsp $BF, $XB, $DCMX", IIC_VecFP, []>;
  def XSTSTDCDP : XX2_BF3_DCMX7_RS6<60, 362,
                              (outs crrc:$BF), (ins u7imm:$DCMX, vsfrc:$XB),
                              "xststdcdp $BF, $XB, $DCMX", IIC_VecFP, []>;
  def XSTSTDCQP : X_BF3_DCMX7_RS5  <63, 708,
                              (outs crrc:$BF), (ins u7imm:$DCMX, vrrc:$vB),
                              "xststdcqp $BF, $vB, $DCMX", IIC_VecFP, []>;

  // Vector Test Data Class SP/DP
  def XVTSTDCSP : XX2_RD6_DCMX7_RS6<60, 13, 5,
                              (outs vsrc:$XT), (ins u7imm:$DCMX, vsrc:$XB),
                              "xvtstdcsp $XT, $XB, $DCMX", IIC_VecFP, []>;
  def XVTSTDCDP : XX2_RD6_DCMX7_RS6<60, 15, 5,
                              (outs vsrc:$XT), (ins u7imm:$DCMX, vsrc:$XB),
                              "xvtstdcdp $XT, $XB, $DCMX", IIC_VecFP, []>;

  //===--------------------------------------------------------------------===//

  // Maximum/Minimum Type-C/Type-J DP
  // XT.dword[1] = 0xUUUU_UUUU_UUUU_UUUU, so we use vsrc for XT
  def XSMAXCDP : XX3_XT5_XA5_XB5<60, 128, "xsmaxcdp", vsrc, vsfrc, vsfrc,
                                 IIC_VecFP, []>;
  def XSMAXJDP : XX3_XT5_XA5_XB5<60, 144, "xsmaxjdp", vsrc, vsfrc, vsfrc,
                                 IIC_VecFP, []>;
  def XSMINCDP : XX3_XT5_XA5_XB5<60, 136, "xsmincdp", vsrc, vsfrc, vsfrc,
                                 IIC_VecFP, []>;
  def XSMINJDP : XX3_XT5_XA5_XB5<60, 152, "xsminjdp", vsrc, vsfrc, vsfrc,
                                 IIC_VecFP, []>;

  //===--------------------------------------------------------------------===//

  // Vector Byte-Reverse H/W/D/Q Word
  def XXBRH : XX2_XT6_XO5_XB6<60,  7, 475, "xxbrh", vsrc, []>;
  def XXBRW : XX2_XT6_XO5_XB6<60, 15, 475, "xxbrw", vsrc, []>;
  def XXBRD : XX2_XT6_XO5_XB6<60, 23, 475, "xxbrd", vsrc, []>;
  def XXBRQ : XX2_XT6_XO5_XB6<60, 31, 475, "xxbrq", vsrc, []>;

  // Vector Permute
  def XXPERM  : XX3_XT5_XA5_XB5<60, 26, "xxperm" , vsrc, vsrc, vsrc,
                                IIC_VecPerm, []>;
  def XXPERMR : XX3_XT5_XA5_XB5<60, 58, "xxpermr", vsrc, vsrc, vsrc,
                                IIC_VecPerm, []>;

  // Vector Splat Immediate Byte
  def XXSPLTIB : X_RD6_IMM8<60, 360, (outs vsrc:$XT), (ins u8imm:$IMM8),
                            "xxspltib $XT, $IMM8", IIC_VecPerm, []>;

  //===--------------------------------------------------------------------===//
  // Vector/Scalar Load/Store Instructions

  let mayLoad = 1 in {
  // Load Vector
  def LXV : DQ_RD6_RS5_DQ12<61, 1, (outs vsrc:$XT), (ins memrix16:$src),
                            "lxv $XT, $src", IIC_LdStLFD, []>;
  // Load DWord
  def LXSD  : DSForm_1<57, 2, (outs vrrc:$vD), (ins memrix:$src),
                       "lxsd $vD, $src", IIC_LdStLFD, []>;
  // Load SP from src, convert it to DP, and place in dword[0]
  def LXSSP : DSForm_1<57, 3, (outs vrrc:$vD), (ins memrix:$src),
                       "lxssp $vD, $src", IIC_LdStLFD, []>;

  // [PO T RA RB XO TX] almost equal to [PO S RA RB XO SX], but has different
  // "out" and "in" dag
  class X_XT6_RA5_RB5<bits<6> opcode, bits<10> xo, string opc,
                      RegisterOperand vtype, list<dag> pattern>
    : XX1Form<opcode, xo, (outs vtype:$XT), (ins memrr:$src),
              !strconcat(opc, " $XT, $src"), IIC_LdStLFD, pattern>;

  // Load as Integer Byte/Halfword & Zero Indexed
  def LXSIBZX : X_XT6_RA5_RB5<31, 781, "lxsibzx", vsfrc, []>;
  def LXSIHZX : X_XT6_RA5_RB5<31, 813, "lxsihzx", vsfrc, []>;

  // Load Vector Halfword*8/Byte*16 Indexed
  def LXVH8X  : X_XT6_RA5_RB5<31, 812, "lxvh8x" , vsrc, []>;
  def LXVB16X : X_XT6_RA5_RB5<31, 876, "lxvb16x", vsrc, []>;

  // Load Vector Indexed
  def LXVX    : X_XT6_RA5_RB5<31, 268, "lxvx"   , vsrc, []>;

  // Load Vector (Left-justified) with Length
  def LXVL    : X_XT6_RA5_RB5<31, 269, "lxvl"   , vsrc, []>;
  def LXVLL   : X_XT6_RA5_RB5<31, 301, "lxvll"  , vsrc, []>;

  // Load Vector Word & Splat Indexed
  def LXVWSX  : X_XT6_RA5_RB5<31, 364, "lxvwsx" , vsrc, []>;
  } // end mayLoad

  let mayStore = 1 in {
  // Store Vector
  def STXV : DQ_RD6_RS5_DQ12<61, 5, (outs), (ins vsrc:$XT, memrix16:$dst),
                             "stxv $XT, $dst", IIC_LdStSTFD, []>;
  // Store DWord
  def STXSD  : DSForm_1<61, 2, (outs), (ins vrrc:$vS, memrix:$dst),
                        "stxsd $vS, $dst", IIC_LdStSTFD, []>;
  // Convert DP of dword[0] to SP, and Store to dst
  def STXSSP : DSForm_1<61, 3, (outs), (ins vrrc:$vS, memrix:$dst),
                        "stxssp $vS, $dst", IIC_LdStSTFD, []>;

  // [PO S RA RB XO SX]
  class X_XS6_RA5_RB5<bits<6> opcode, bits<10> xo, string opc,
                      RegisterOperand vtype, list<dag> pattern>
    : XX1Form<opcode, xo, (outs), (ins vtype:$XT, memrr:$dst),
              !strconcat(opc, " $XT, $dst"), IIC_LdStSTFD, pattern>;

  // Store as Integer Byte/Halfword Indexed
  def STXSIBX  : X_XS6_RA5_RB5<31,  909, "stxsibx" , vsfrc, []>;
  def STXSIHX  : X_XS6_RA5_RB5<31,  941, "stxsihx" , vsfrc, []>;

  // Store Vector Halfword*8/Byte*16 Indexed
  def STXVH8X  : X_XS6_RA5_RB5<31,  940, "stxvh8x" , vsrc, []>;
  def STXVB16X : X_XS6_RA5_RB5<31, 1004, "stxvb16x", vsrc, []>;

  // Store Vector Indexed
  def STXVX    : X_XS6_RA5_RB5<31,  396, "stxvx"   , vsrc, []>;

  // Store Vector (Left-justified) with Length
  def STXVL    : X_XS6_RA5_RB5<31,  397, "stxvl"   , vsrc, []>;
  def STXVLL   : X_XS6_RA5_RB5<31,  429, "stxvll"  , vsrc, []>;
  } // end mayStore
} // end HasP9Vector
