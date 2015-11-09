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

multiclass XX3Form_Rcr<bits<6> opcode, bits<7> xo, dag OOL, dag IOL,
                    string asmbase, string asmstr, InstrItinClass itin,
                    list<dag> pattern> {
  let BaseName = asmbase in {
    def NAME : XX3Form_Rc<opcode, xo, OOL, IOL,
                       !strconcat(asmbase, !strconcat(" ", asmstr)), itin,
                       pattern>;
    let Defs = [CR6] in
    def o    : XX3Form_Rc<opcode, xo, OOL, IOL,
                       !strconcat(asmbase, !strconcat(". ", asmstr)), itin,
                       []>, isDOT;
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
                             (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                             "xvcmpeqdp", "$XT, $XA, $XB", IIC_VecFPCompare,
                             [(set v2i64:$XT,
                                (int_ppc_vsx_xvcmpeqdp v2f64:$XA, v2f64:$XB))]>;
  defm XVCMPEQSP : XX3Form_Rcr<60, 67,
                             (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                             "xvcmpeqsp", "$XT, $XA, $XB", IIC_VecFPCompare,
                             [(set v4i32:$XT,
                                (int_ppc_vsx_xvcmpeqsp v4f32:$XA, v4f32:$XB))]>;
  defm XVCMPGEDP : XX3Form_Rcr<60, 115,
                             (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                             "xvcmpgedp", "$XT, $XA, $XB", IIC_VecFPCompare,
                             [(set v2i64:$XT,
                                (int_ppc_vsx_xvcmpgedp v2f64:$XA, v2f64:$XB))]>;
  defm XVCMPGESP : XX3Form_Rcr<60, 83,
                             (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                             "xvcmpgesp", "$XT, $XA, $XB", IIC_VecFPCompare,
                             [(set v4i32:$XT,
                                (int_ppc_vsx_xvcmpgesp v4f32:$XA, v4f32:$XB))]>;
  defm XVCMPGTDP : XX3Form_Rcr<60, 107,
                             (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                             "xvcmpgtdp", "$XT, $XA, $XB", IIC_VecFPCompare,
                             [(set v2i64:$XT,
                                (int_ppc_vsx_xvcmpgtdp v2f64:$XA, v2f64:$XB))]>;
  defm XVCMPGTSP : XX3Form_Rcr<60, 75,
                             (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                             "xvcmpgtsp", "$XT, $XA, $XB", IIC_VecFPCompare,
                             [(set v4i32:$XT,
                                (int_ppc_vsx_xvcmpgtsp v4f32:$XA, v4f32:$XB))]>;

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

def : Pat<(f64 (vector_extract v2f64:$S, 0)),
          (f64 (EXTRACT_SUBREG $S, sub_64))>;
def : Pat<(f64 (vector_extract v2f64:$S, 1)),
          (f64 (EXTRACT_SUBREG (XXPERMDI $S, $S, 2), sub_64))>;
}

let Predicates = [IsLittleEndian] in {
def : Pat<(v2f64 (scalar_to_vector f64:$A)),
          (v2f64 (XXPERMDI (SUBREG_TO_REG (i64 1), $A, sub_64),
                           (SUBREG_TO_REG (i64 1), $A, sub_64), 0))>;

def : Pat<(f64 (vector_extract v2f64:$S, 0)),
          (f64 (EXTRACT_SUBREG (XXPERMDI $S, $S, 2), sub_64))>;
def : Pat<(f64 (vector_extract v2f64:$S, 1)),
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
} // AddedComplexity = 400
} // HasP8Vector

let Predicates = [HasDirectMove, HasVSX] in {
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
} // HasDirectMove, HasVSX
