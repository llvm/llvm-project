//===-- PPCInstrAltivec.td - The PowerPC Altivec Extension -*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the Altivec extension to the PowerPC instruction set.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Altivec transformation functions and pattern fragments.
//

// Since we canonicalize buildvectors to v16i8, all vnots "-1" operands will be
// of that type.
def vnot_ppc : PatFrag<(ops node:$in),
                       (xor node:$in, (bitconvert (v16i8 immAllOnesV)))>;

def vpkuhum_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                              (vector_shuffle node:$lhs, node:$rhs), [{
  return PPC::isVPKUHUMShuffleMask(cast<ShuffleVectorSDNode>(N), false);
}]>;
def vpkuwum_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                              (vector_shuffle node:$lhs, node:$rhs), [{
  return PPC::isVPKUWUMShuffleMask(cast<ShuffleVectorSDNode>(N), false);
}]>;
def vpkuhum_unary_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                                    (vector_shuffle node:$lhs, node:$rhs), [{
  return PPC::isVPKUHUMShuffleMask(cast<ShuffleVectorSDNode>(N), true);
}]>;
def vpkuwum_unary_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                                    (vector_shuffle node:$lhs, node:$rhs), [{
  return PPC::isVPKUWUMShuffleMask(cast<ShuffleVectorSDNode>(N), true);
}]>;


def vmrglb_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                             (vector_shuffle (v16i8 node:$lhs), node:$rhs), [{
  return PPC::isVMRGLShuffleMask(cast<ShuffleVectorSDNode>(N), 1, false);
}]>;
def vmrglh_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                             (vector_shuffle (v16i8 node:$lhs), node:$rhs), [{
  return PPC::isVMRGLShuffleMask(cast<ShuffleVectorSDNode>(N), 2, false);
}]>;
def vmrglw_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                             (vector_shuffle (v16i8 node:$lhs), node:$rhs), [{
  return PPC::isVMRGLShuffleMask(cast<ShuffleVectorSDNode>(N), 4, false);
}]>;
def vmrghb_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                             (vector_shuffle (v16i8 node:$lhs), node:$rhs), [{
  return PPC::isVMRGHShuffleMask(cast<ShuffleVectorSDNode>(N), 1, false);
}]>;
def vmrghh_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                             (vector_shuffle (v16i8 node:$lhs), node:$rhs), [{
  return PPC::isVMRGHShuffleMask(cast<ShuffleVectorSDNode>(N), 2, false);
}]>;
def vmrghw_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                             (vector_shuffle (v16i8 node:$lhs), node:$rhs), [{
  return PPC::isVMRGHShuffleMask(cast<ShuffleVectorSDNode>(N), 4, false);
}]>;


def vmrglb_unary_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                               (vector_shuffle (v16i8 node:$lhs), node:$rhs), [{
  return PPC::isVMRGLShuffleMask(cast<ShuffleVectorSDNode>(N), 1, true);
}]>;
def vmrglh_unary_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                                   (vector_shuffle node:$lhs, node:$rhs), [{
  return PPC::isVMRGLShuffleMask(cast<ShuffleVectorSDNode>(N), 2, true);
}]>;
def vmrglw_unary_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                                   (vector_shuffle node:$lhs, node:$rhs), [{
  return PPC::isVMRGLShuffleMask(cast<ShuffleVectorSDNode>(N), 4, true);
}]>;
def vmrghb_unary_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                                   (vector_shuffle node:$lhs, node:$rhs), [{
  return PPC::isVMRGHShuffleMask(cast<ShuffleVectorSDNode>(N), 1, true);
}]>;
def vmrghh_unary_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                                   (vector_shuffle node:$lhs, node:$rhs), [{
  return PPC::isVMRGHShuffleMask(cast<ShuffleVectorSDNode>(N), 2, true);
}]>;
def vmrghw_unary_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                                   (vector_shuffle node:$lhs, node:$rhs), [{
  return PPC::isVMRGHShuffleMask(cast<ShuffleVectorSDNode>(N), 4, true);
}]>;


def VSLDOI_get_imm : SDNodeXForm<vector_shuffle, [{
  return getI32Imm(PPC::isVSLDOIShuffleMask(N, false));
}]>;
def vsldoi_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                             (vector_shuffle node:$lhs, node:$rhs), [{
  return PPC::isVSLDOIShuffleMask(N, false) != -1;
}], VSLDOI_get_imm>;


/// VSLDOI_unary* - These are used to match vsldoi(X,X), which is turned into
/// vector_shuffle(X,undef,mask) by the dag combiner.
def VSLDOI_unary_get_imm : SDNodeXForm<vector_shuffle, [{
  return getI32Imm(PPC::isVSLDOIShuffleMask(N, true));
}]>;
def vsldoi_unary_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                                   (vector_shuffle node:$lhs, node:$rhs), [{
  return PPC::isVSLDOIShuffleMask(N, true) != -1;
}], VSLDOI_unary_get_imm>;


// VSPLT*_get_imm xform function: convert vector_shuffle mask to VSPLT* imm.
def VSPLTB_get_imm : SDNodeXForm<vector_shuffle, [{
  return getI32Imm(PPC::getVSPLTImmediate(N, 1));
}]>;
def vspltb_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                             (vector_shuffle node:$lhs, node:$rhs), [{
  return PPC::isSplatShuffleMask(cast<ShuffleVectorSDNode>(N), 1);
}], VSPLTB_get_imm>;
def VSPLTH_get_imm : SDNodeXForm<vector_shuffle, [{
  return getI32Imm(PPC::getVSPLTImmediate(N, 2));
}]>;
def vsplth_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                             (vector_shuffle node:$lhs, node:$rhs), [{
  return PPC::isSplatShuffleMask(cast<ShuffleVectorSDNode>(N), 2);
}], VSPLTH_get_imm>;
def VSPLTW_get_imm : SDNodeXForm<vector_shuffle, [{
  return getI32Imm(PPC::getVSPLTImmediate(N, 4));
}]>;
def vspltw_shuffle : PatFrag<(ops node:$lhs, node:$rhs),
                             (vector_shuffle node:$lhs, node:$rhs), [{
  return PPC::isSplatShuffleMask(cast<ShuffleVectorSDNode>(N), 4);
}], VSPLTW_get_imm>;


// VSPLTISB_get_imm xform function: convert build_vector to VSPLTISB imm.
def VSPLTISB_get_imm : SDNodeXForm<build_vector, [{
  return PPC::get_VSPLTI_elt(N, 1, *CurDAG);
}]>;
def vecspltisb : PatLeaf<(build_vector), [{
  return PPC::get_VSPLTI_elt(N, 1, *CurDAG).getNode() != 0;
}], VSPLTISB_get_imm>;

// VSPLTISH_get_imm xform function: convert build_vector to VSPLTISH imm.
def VSPLTISH_get_imm : SDNodeXForm<build_vector, [{
  return PPC::get_VSPLTI_elt(N, 2, *CurDAG);
}]>;
def vecspltish : PatLeaf<(build_vector), [{
  return PPC::get_VSPLTI_elt(N, 2, *CurDAG).getNode() != 0;
}], VSPLTISH_get_imm>;

// VSPLTISW_get_imm xform function: convert build_vector to VSPLTISW imm.
def VSPLTISW_get_imm : SDNodeXForm<build_vector, [{
  return PPC::get_VSPLTI_elt(N, 4, *CurDAG);
}]>;
def vecspltisw : PatLeaf<(build_vector), [{
  return PPC::get_VSPLTI_elt(N, 4, *CurDAG).getNode() != 0;
}], VSPLTISW_get_imm>;

def V_immneg0 : PatLeaf<(build_vector), [{
  return PPC::isAllNegativeZeroVector(N);
}]>;

//===----------------------------------------------------------------------===//
// Helpers for defining instructions that directly correspond to intrinsics.

// VA1a_Int - A VAForm_1a intrinsic definition.
class VA1a_Int<bits<6> xo, string opc, Intrinsic IntID>
  : VAForm_1a<xo, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB, VRRC:$vC),
              !strconcat(opc, " $vD, $vA, $vB, $vC"), VecFP,
                       [(set VRRC:$vD, (IntID VRRC:$vA, VRRC:$vB, VRRC:$vC))]>;

// VX1_Int - A VXForm_1 intrinsic definition.
class VX1_Int<bits<11> xo, string opc, Intrinsic IntID>
  : VXForm_1<xo, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
             !strconcat(opc, " $vD, $vA, $vB"), VecFP,
             [(set VRRC:$vD, (IntID VRRC:$vA, VRRC:$vB))]>;

// VX2_Int - A VXForm_2 intrinsic definition.
class VX2_Int<bits<11> xo, string opc, Intrinsic IntID>
  : VXForm_2<xo, (outs VRRC:$vD), (ins VRRC:$vB),
             !strconcat(opc, " $vD, $vB"), VecFP,
             [(set VRRC:$vD, (IntID VRRC:$vB))]>;

//===----------------------------------------------------------------------===//
// Instruction Definitions.

def DSS      : DSS_Form<822, (outs),
                        (ins u5imm:$ZERO0, u5imm:$STRM,u5imm:$ZERO1,u5imm:$ZERO2),
                        "dss $STRM", LdStLoad /*FIXME*/, []>;
def DSSALL   : DSS_Form<822, (outs),
                        (ins u5imm:$ONE, u5imm:$ZERO0,u5imm:$ZERO1,u5imm:$ZERO2),
                        "dssall", LdStLoad /*FIXME*/, []>;
def DST      : DSS_Form<342, (outs),
                        (ins u5imm:$ZERO, u5imm:$STRM, GPRC:$rA, GPRC:$rB),
                        "dst $rA, $rB, $STRM", LdStLoad /*FIXME*/, []>;
def DSTT     : DSS_Form<342, (outs),
                        (ins u5imm:$ONE, u5imm:$STRM, GPRC:$rA, GPRC:$rB),
                        "dstt $rA, $rB, $STRM", LdStLoad /*FIXME*/, []>;
def DSTST    : DSS_Form<374, (outs),
                        (ins u5imm:$ZERO, u5imm:$STRM, GPRC:$rA, GPRC:$rB),
                        "dstst $rA, $rB, $STRM", LdStLoad /*FIXME*/, []>;
def DSTSTT   : DSS_Form<374, (outs),
                        (ins u5imm:$ONE, u5imm:$STRM, GPRC:$rA, GPRC:$rB),
                        "dststt $rA, $rB, $STRM", LdStLoad /*FIXME*/, []>;

def DST64    : DSS_Form<342, (outs),
                        (ins u5imm:$ZERO, u5imm:$STRM, G8RC:$rA, GPRC:$rB),
                        "dst $rA, $rB, $STRM", LdStLoad /*FIXME*/, []>;
def DSTT64   : DSS_Form<342, (outs),
                        (ins u5imm:$ONE, u5imm:$STRM, G8RC:$rA, GPRC:$rB),
                        "dstt $rA, $rB, $STRM", LdStLoad /*FIXME*/, []>;
def DSTST64  : DSS_Form<374, (outs),
                        (ins u5imm:$ZERO, u5imm:$STRM, G8RC:$rA, GPRC:$rB),
                        "dstst $rA, $rB, $STRM", LdStLoad /*FIXME*/, []>;
def DSTSTT64 : DSS_Form<374, (outs),
                        (ins u5imm:$ONE, u5imm:$STRM, G8RC:$rA, GPRC:$rB),
                        "dststt $rA, $rB, $STRM", LdStLoad /*FIXME*/, []>;

def MFVSCR : VXForm_4<1540, (outs VRRC:$vD), (ins),
                      "mfvscr $vD", LdStStore,
                      [(set VRRC:$vD, (int_ppc_altivec_mfvscr))]>; 
def MTVSCR : VXForm_5<1604, (outs), (ins VRRC:$vB),
                      "mtvscr $vB", LdStLoad,
                      [(int_ppc_altivec_mtvscr VRRC:$vB)]>; 

let canFoldAsLoad = 1, PPC970_Unit = 2 in {  // Loads.
def LVEBX: XForm_1<31,   7, (outs VRRC:$vD), (ins memrr:$src),
                   "lvebx $vD, $src", LdStLoad,
                   [(set VRRC:$vD, (int_ppc_altivec_lvebx xoaddr:$src))]>;
def LVEHX: XForm_1<31,  39, (outs VRRC:$vD), (ins memrr:$src),
                   "lvehx $vD, $src", LdStLoad,
                   [(set VRRC:$vD, (int_ppc_altivec_lvehx xoaddr:$src))]>;
def LVEWX: XForm_1<31,  71, (outs VRRC:$vD), (ins memrr:$src),
                   "lvewx $vD, $src", LdStLoad,
                   [(set VRRC:$vD, (int_ppc_altivec_lvewx xoaddr:$src))]>;
def LVX  : XForm_1<31, 103, (outs VRRC:$vD), (ins memrr:$src),
                   "lvx $vD, $src", LdStLoad,
                   [(set VRRC:$vD, (int_ppc_altivec_lvx xoaddr:$src))]>;
def LVXL : XForm_1<31, 359, (outs VRRC:$vD), (ins memrr:$src),
                   "lvxl $vD, $src", LdStLoad,
                   [(set VRRC:$vD, (int_ppc_altivec_lvxl xoaddr:$src))]>;
}

def LVSL : XForm_1<31,   6, (outs VRRC:$vD), (ins memrr:$src),
                   "lvsl $vD, $src", LdStLoad,
                   [(set VRRC:$vD, (int_ppc_altivec_lvsl xoaddr:$src))]>,
                   PPC970_Unit_LSU;
def LVSR : XForm_1<31,  38, (outs VRRC:$vD), (ins memrr:$src),
                   "lvsr $vD, $src", LdStLoad,
                   [(set VRRC:$vD, (int_ppc_altivec_lvsr xoaddr:$src))]>,
                   PPC970_Unit_LSU;

let PPC970_Unit = 2 in {   // Stores.
def STVEBX: XForm_8<31, 135, (outs), (ins VRRC:$rS, memrr:$dst),
                   "stvebx $rS, $dst", LdStStore,
                   [(int_ppc_altivec_stvebx VRRC:$rS, xoaddr:$dst)]>;
def STVEHX: XForm_8<31, 167, (outs), (ins VRRC:$rS, memrr:$dst),
                   "stvehx $rS, $dst", LdStStore,
                   [(int_ppc_altivec_stvehx VRRC:$rS, xoaddr:$dst)]>;
def STVEWX: XForm_8<31, 199, (outs), (ins VRRC:$rS, memrr:$dst),
                   "stvewx $rS, $dst", LdStStore,
                   [(int_ppc_altivec_stvewx VRRC:$rS, xoaddr:$dst)]>;
def STVX  : XForm_8<31, 231, (outs), (ins VRRC:$rS, memrr:$dst),
                   "stvx $rS, $dst", LdStStore,
                   [(int_ppc_altivec_stvx VRRC:$rS, xoaddr:$dst)]>;
def STVXL : XForm_8<31, 487, (outs), (ins VRRC:$rS, memrr:$dst),
                   "stvxl $rS, $dst", LdStStore,
                   [(int_ppc_altivec_stvxl VRRC:$rS, xoaddr:$dst)]>;
}

let PPC970_Unit = 5 in {  // VALU Operations.
// VA-Form instructions.  3-input AltiVec ops.
def VMADDFP : VAForm_1<46, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vC, VRRC:$vB),
                       "vmaddfp $vD, $vA, $vC, $vB", VecFP,
                       [(set VRRC:$vD, (fma VRRC:$vA, VRRC:$vC, VRRC:$vB))]>;
def VNMSUBFP: VAForm_1<47, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vC, VRRC:$vB),
                       "vnmsubfp $vD, $vA, $vC, $vB", VecFP,
                       [(set VRRC:$vD, (fneg (fma VRRC:$vA, VRRC:$vC,
                                                  (fneg VRRC:$vB))))]>; 

def VMHADDSHS  : VA1a_Int<32, "vmhaddshs",  int_ppc_altivec_vmhaddshs>;
def VMHRADDSHS : VA1a_Int<33, "vmhraddshs", int_ppc_altivec_vmhraddshs>;
def VMLADDUHM  : VA1a_Int<34, "vmladduhm",  int_ppc_altivec_vmladduhm>;
def VPERM      : VA1a_Int<43, "vperm",      int_ppc_altivec_vperm>;
def VSEL       : VA1a_Int<42, "vsel",       int_ppc_altivec_vsel>;

// Shuffles.
def VSLDOI  : VAForm_2<44, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB, u5imm:$SH),
                       "vsldoi $vD, $vA, $vB, $SH", VecFP,
                       [(set VRRC:$vD, 
                         (vsldoi_shuffle:$SH (v16i8 VRRC:$vA), VRRC:$vB))]>;

// VX-Form instructions.  AltiVec arithmetic ops.
def VADDFP : VXForm_1<10, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                      "vaddfp $vD, $vA, $vB", VecFP,
                      [(set VRRC:$vD, (fadd VRRC:$vA, VRRC:$vB))]>;
                      
def VADDUBM : VXForm_1<0, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                      "vaddubm $vD, $vA, $vB", VecGeneral,
                      [(set VRRC:$vD, (add (v16i8 VRRC:$vA), VRRC:$vB))]>;
def VADDUHM : VXForm_1<64, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                      "vadduhm $vD, $vA, $vB", VecGeneral,
                      [(set VRRC:$vD, (add (v8i16 VRRC:$vA), VRRC:$vB))]>;
def VADDUWM : VXForm_1<128, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                      "vadduwm $vD, $vA, $vB", VecGeneral,
                      [(set VRRC:$vD, (add (v4i32 VRRC:$vA), VRRC:$vB))]>;
                      
def VADDCUW : VX1_Int<384, "vaddcuw", int_ppc_altivec_vaddcuw>;
def VADDSBS : VX1_Int<768, "vaddsbs", int_ppc_altivec_vaddsbs>;
def VADDSHS : VX1_Int<832, "vaddshs", int_ppc_altivec_vaddshs>;
def VADDSWS : VX1_Int<896, "vaddsws", int_ppc_altivec_vaddsws>;
def VADDUBS : VX1_Int<512, "vaddubs", int_ppc_altivec_vaddubs>;
def VADDUHS : VX1_Int<576, "vadduhs", int_ppc_altivec_vadduhs>;
def VADDUWS : VX1_Int<640, "vadduws", int_ppc_altivec_vadduws>;
                             
                             
def VAND : VXForm_1<1028, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                    "vand $vD, $vA, $vB", VecFP,
                    [(set VRRC:$vD, (and (v4i32 VRRC:$vA), VRRC:$vB))]>;
def VANDC : VXForm_1<1092, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                     "vandc $vD, $vA, $vB", VecFP,
                     [(set VRRC:$vD, (and (v4i32 VRRC:$vA),
                                          (vnot_ppc VRRC:$vB)))]>;

def VCFSX  : VXForm_1<842, (outs VRRC:$vD), (ins u5imm:$UIMM, VRRC:$vB),
                      "vcfsx $vD, $vB, $UIMM", VecFP,
                      [(set VRRC:$vD,
                             (int_ppc_altivec_vcfsx VRRC:$vB, imm:$UIMM))]>;
def VCFUX  : VXForm_1<778, (outs VRRC:$vD), (ins u5imm:$UIMM, VRRC:$vB),
                      "vcfux $vD, $vB, $UIMM", VecFP,
                      [(set VRRC:$vD,
                             (int_ppc_altivec_vcfux VRRC:$vB, imm:$UIMM))]>;
def VCTSXS : VXForm_1<970, (outs VRRC:$vD), (ins u5imm:$UIMM, VRRC:$vB),
                      "vctsxs $vD, $vB, $UIMM", VecFP,
                      [(set VRRC:$vD,
                             (int_ppc_altivec_vctsxs VRRC:$vB, imm:$UIMM))]>;
def VCTUXS : VXForm_1<906, (outs VRRC:$vD), (ins u5imm:$UIMM, VRRC:$vB),
                      "vctuxs $vD, $vB, $UIMM", VecFP,
                      [(set VRRC:$vD,
                             (int_ppc_altivec_vctuxs VRRC:$vB, imm:$UIMM))]>;

// Defines with the UIM field set to 0 for floating-point
// to integer (fp_to_sint/fp_to_uint) conversions and integer
// to floating-point (sint_to_fp/uint_to_fp) conversions.
let VA = 0 in {
def VCFSX_0 : VXForm_1<842, (outs VRRC:$vD), (ins VRRC:$vB),
                       "vcfsx $vD, $vB, 0", VecFP,
                       [(set VRRC:$vD,
                             (int_ppc_altivec_vcfsx VRRC:$vB, 0))]>;
def VCTUXS_0 : VXForm_1<906, (outs VRRC:$vD), (ins VRRC:$vB),
                        "vctuxs $vD, $vB, 0", VecFP,
                        [(set VRRC:$vD,
                               (int_ppc_altivec_vctuxs VRRC:$vB, 0))]>;
def VCFUX_0 : VXForm_1<778, (outs VRRC:$vD), (ins VRRC:$vB),
                       "vcfux $vD, $vB, 0", VecFP,
                       [(set VRRC:$vD,
                               (int_ppc_altivec_vcfux VRRC:$vB, 0))]>;
def VCTSXS_0 : VXForm_1<970, (outs VRRC:$vD), (ins VRRC:$vB),
                      "vctsxs $vD, $vB, 0", VecFP,
                      [(set VRRC:$vD,
                             (int_ppc_altivec_vctsxs VRRC:$vB, 0))]>;
}
def VEXPTEFP : VX2_Int<394, "vexptefp", int_ppc_altivec_vexptefp>;
def VLOGEFP  : VX2_Int<458, "vlogefp",  int_ppc_altivec_vlogefp>;

def VAVGSB : VX1_Int<1282, "vavgsb", int_ppc_altivec_vavgsb>;
def VAVGSH : VX1_Int<1346, "vavgsh", int_ppc_altivec_vavgsh>;
def VAVGSW : VX1_Int<1410, "vavgsw", int_ppc_altivec_vavgsw>;
def VAVGUB : VX1_Int<1026, "vavgub", int_ppc_altivec_vavgub>;
def VAVGUH : VX1_Int<1090, "vavguh", int_ppc_altivec_vavguh>;
def VAVGUW : VX1_Int<1154, "vavguw", int_ppc_altivec_vavguw>;

def VMAXFP : VX1_Int<1034, "vmaxfp", int_ppc_altivec_vmaxfp>;
def VMAXSB : VX1_Int< 258, "vmaxsb", int_ppc_altivec_vmaxsb>;
def VMAXSH : VX1_Int< 322, "vmaxsh", int_ppc_altivec_vmaxsh>;
def VMAXSW : VX1_Int< 386, "vmaxsw", int_ppc_altivec_vmaxsw>;
def VMAXUB : VX1_Int<   2, "vmaxub", int_ppc_altivec_vmaxub>;
def VMAXUH : VX1_Int<  66, "vmaxuh", int_ppc_altivec_vmaxuh>;
def VMAXUW : VX1_Int< 130, "vmaxuw", int_ppc_altivec_vmaxuw>;
def VMINFP : VX1_Int<1098, "vminfp", int_ppc_altivec_vminfp>;
def VMINSB : VX1_Int< 770, "vminsb", int_ppc_altivec_vminsb>;
def VMINSH : VX1_Int< 834, "vminsh", int_ppc_altivec_vminsh>;
def VMINSW : VX1_Int< 898, "vminsw", int_ppc_altivec_vminsw>;
def VMINUB : VX1_Int< 514, "vminub", int_ppc_altivec_vminub>;
def VMINUH : VX1_Int< 578, "vminuh", int_ppc_altivec_vminuh>;
def VMINUW : VX1_Int< 642, "vminuw", int_ppc_altivec_vminuw>;

def VMRGHB : VXForm_1< 12, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                      "vmrghb $vD, $vA, $vB", VecFP,
                      [(set VRRC:$vD, (vmrghb_shuffle VRRC:$vA, VRRC:$vB))]>;
def VMRGHH : VXForm_1< 76, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                      "vmrghh $vD, $vA, $vB", VecFP,
                      [(set VRRC:$vD, (vmrghh_shuffle VRRC:$vA, VRRC:$vB))]>;
def VMRGHW : VXForm_1<140, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                      "vmrghw $vD, $vA, $vB", VecFP,
                      [(set VRRC:$vD, (vmrghw_shuffle VRRC:$vA, VRRC:$vB))]>;
def VMRGLB : VXForm_1<268, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                      "vmrglb $vD, $vA, $vB", VecFP,
                      [(set VRRC:$vD, (vmrglb_shuffle VRRC:$vA, VRRC:$vB))]>;
def VMRGLH : VXForm_1<332, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                      "vmrglh $vD, $vA, $vB", VecFP,
                      [(set VRRC:$vD, (vmrglh_shuffle VRRC:$vA, VRRC:$vB))]>;
def VMRGLW : VXForm_1<396, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                      "vmrglw $vD, $vA, $vB", VecFP,
                      [(set VRRC:$vD, (vmrglw_shuffle VRRC:$vA, VRRC:$vB))]>;

def VMSUMMBM : VA1a_Int<37, "vmsummbm", int_ppc_altivec_vmsummbm>;
def VMSUMSHM : VA1a_Int<40, "vmsumshm", int_ppc_altivec_vmsumshm>;
def VMSUMSHS : VA1a_Int<41, "vmsumshs", int_ppc_altivec_vmsumshs>;
def VMSUMUBM : VA1a_Int<36, "vmsumubm", int_ppc_altivec_vmsumubm>;
def VMSUMUHM : VA1a_Int<38, "vmsumuhm", int_ppc_altivec_vmsumuhm>;
def VMSUMUHS : VA1a_Int<39, "vmsumuhs", int_ppc_altivec_vmsumuhs>;

def VMULESB : VX1_Int<776, "vmulesb", int_ppc_altivec_vmulesb>;
def VMULESH : VX1_Int<840, "vmulesh", int_ppc_altivec_vmulesh>;
def VMULEUB : VX1_Int<520, "vmuleub", int_ppc_altivec_vmuleub>;
def VMULEUH : VX1_Int<584, "vmuleuh", int_ppc_altivec_vmuleuh>;
def VMULOSB : VX1_Int<264, "vmulosb", int_ppc_altivec_vmulosb>;
def VMULOSH : VX1_Int<328, "vmulosh", int_ppc_altivec_vmulosh>;
def VMULOUB : VX1_Int<  8, "vmuloub", int_ppc_altivec_vmuloub>;
def VMULOUH : VX1_Int< 72, "vmulouh", int_ppc_altivec_vmulouh>;
                       
def VREFP     : VX2_Int<266, "vrefp",     int_ppc_altivec_vrefp>;
def VRFIM     : VX2_Int<714, "vrfim",     int_ppc_altivec_vrfim>;
def VRFIN     : VX2_Int<522, "vrfin",     int_ppc_altivec_vrfin>;
def VRFIP     : VX2_Int<650, "vrfip",     int_ppc_altivec_vrfip>;
def VRFIZ     : VX2_Int<586, "vrfiz",     int_ppc_altivec_vrfiz>;
def VRSQRTEFP : VX2_Int<330, "vrsqrtefp", int_ppc_altivec_vrsqrtefp>;

def VSUBCUW : VX1_Int<74, "vsubcuw", int_ppc_altivec_vsubcuw>;

def VSUBFP  : VXForm_1<74, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                      "vsubfp $vD, $vA, $vB", VecGeneral,
                      [(set VRRC:$vD, (fsub VRRC:$vA, VRRC:$vB))]>;
def VSUBUBM : VXForm_1<1024, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                      "vsububm $vD, $vA, $vB", VecGeneral,
                      [(set VRRC:$vD, (sub (v16i8 VRRC:$vA), VRRC:$vB))]>;
def VSUBUHM : VXForm_1<1088, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                      "vsubuhm $vD, $vA, $vB", VecGeneral,
                      [(set VRRC:$vD, (sub (v8i16 VRRC:$vA), VRRC:$vB))]>;
def VSUBUWM : VXForm_1<1152, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                      "vsubuwm $vD, $vA, $vB", VecGeneral,
                      [(set VRRC:$vD, (sub (v4i32 VRRC:$vA), VRRC:$vB))]>;
                      
def VSUBSBS : VX1_Int<1792, "vsubsbs" , int_ppc_altivec_vsubsbs>;
def VSUBSHS : VX1_Int<1856, "vsubshs" , int_ppc_altivec_vsubshs>;
def VSUBSWS : VX1_Int<1920, "vsubsws" , int_ppc_altivec_vsubsws>;
def VSUBUBS : VX1_Int<1536, "vsububs" , int_ppc_altivec_vsububs>;
def VSUBUHS : VX1_Int<1600, "vsubuhs" , int_ppc_altivec_vsubuhs>;
def VSUBUWS : VX1_Int<1664, "vsubuws" , int_ppc_altivec_vsubuws>;
def VSUMSWS : VX1_Int<1928, "vsumsws" , int_ppc_altivec_vsumsws>;
def VSUM2SWS: VX1_Int<1672, "vsum2sws", int_ppc_altivec_vsum2sws>;
def VSUM4SBS: VX1_Int<1672, "vsum4sbs", int_ppc_altivec_vsum4sbs>;
def VSUM4SHS: VX1_Int<1608, "vsum4shs", int_ppc_altivec_vsum4shs>;
def VSUM4UBS: VX1_Int<1544, "vsum4ubs", int_ppc_altivec_vsum4ubs>;

def VNOR : VXForm_1<1284, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                    "vnor $vD, $vA, $vB", VecFP,
                    [(set VRRC:$vD, (vnot_ppc (or (v4i32 VRRC:$vA),
                                                  VRRC:$vB)))]>;
def VOR : VXForm_1<1156, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                      "vor $vD, $vA, $vB", VecFP,
                      [(set VRRC:$vD, (or (v4i32 VRRC:$vA), VRRC:$vB))]>;
def VXOR : VXForm_1<1220, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                      "vxor $vD, $vA, $vB", VecFP,
                      [(set VRRC:$vD, (xor (v4i32 VRRC:$vA), VRRC:$vB))]>;

def VRLB   : VX1_Int<   4, "vrlb", int_ppc_altivec_vrlb>;
def VRLH   : VX1_Int<  68, "vrlh", int_ppc_altivec_vrlh>;
def VRLW   : VX1_Int< 132, "vrlw", int_ppc_altivec_vrlw>;

def VSL    : VX1_Int< 452, "vsl" , int_ppc_altivec_vsl >;
def VSLO   : VX1_Int<1036, "vslo", int_ppc_altivec_vslo>;
def VSLB   : VX1_Int< 260, "vslb", int_ppc_altivec_vslb>;
def VSLH   : VX1_Int< 324, "vslh", int_ppc_altivec_vslh>;
def VSLW   : VX1_Int< 388, "vslw", int_ppc_altivec_vslw>;

def VSPLTB : VXForm_1<524, (outs VRRC:$vD), (ins u5imm:$UIMM, VRRC:$vB),
                      "vspltb $vD, $vB, $UIMM", VecPerm,
                      [(set VRRC:$vD,
                        (vspltb_shuffle:$UIMM (v16i8 VRRC:$vB), (undef)))]>;
def VSPLTH : VXForm_1<588, (outs VRRC:$vD), (ins u5imm:$UIMM, VRRC:$vB),
                      "vsplth $vD, $vB, $UIMM", VecPerm,
                      [(set VRRC:$vD,
                        (vsplth_shuffle:$UIMM (v16i8 VRRC:$vB), (undef)))]>;
def VSPLTW : VXForm_1<652, (outs VRRC:$vD), (ins u5imm:$UIMM, VRRC:$vB),
                      "vspltw $vD, $vB, $UIMM", VecPerm,
                      [(set VRRC:$vD, 
                        (vspltw_shuffle:$UIMM (v16i8 VRRC:$vB), (undef)))]>;

def VSR    : VX1_Int< 708, "vsr"  , int_ppc_altivec_vsr>;
def VSRO   : VX1_Int<1100, "vsro" , int_ppc_altivec_vsro>;
def VSRAB  : VX1_Int< 772, "vsrab", int_ppc_altivec_vsrab>;
def VSRAH  : VX1_Int< 836, "vsrah", int_ppc_altivec_vsrah>;
def VSRAW  : VX1_Int< 900, "vsraw", int_ppc_altivec_vsraw>;
def VSRB   : VX1_Int< 516, "vsrb" , int_ppc_altivec_vsrb>;
def VSRH   : VX1_Int< 580, "vsrh" , int_ppc_altivec_vsrh>;
def VSRW   : VX1_Int< 644, "vsrw" , int_ppc_altivec_vsrw>;


def VSPLTISB : VXForm_3<780, (outs VRRC:$vD), (ins s5imm:$SIMM),
                       "vspltisb $vD, $SIMM", VecPerm,
                       [(set VRRC:$vD, (v16i8 vecspltisb:$SIMM))]>;
def VSPLTISH : VXForm_3<844, (outs VRRC:$vD), (ins s5imm:$SIMM),
                       "vspltish $vD, $SIMM", VecPerm,
                       [(set VRRC:$vD, (v8i16 vecspltish:$SIMM))]>;
def VSPLTISW : VXForm_3<908, (outs VRRC:$vD), (ins s5imm:$SIMM),
                       "vspltisw $vD, $SIMM", VecPerm,
                       [(set VRRC:$vD, (v4i32 vecspltisw:$SIMM))]>;

// Vector Pack.
def VPKPX   : VX1_Int<782, "vpkpx", int_ppc_altivec_vpkpx>;
def VPKSHSS : VX1_Int<398, "vpkshss", int_ppc_altivec_vpkshss>;
def VPKSHUS : VX1_Int<270, "vpkshus", int_ppc_altivec_vpkshus>;
def VPKSWSS : VX1_Int<462, "vpkswss", int_ppc_altivec_vpkswss>;
def VPKSWUS : VX1_Int<334, "vpkswus", int_ppc_altivec_vpkswus>;
def VPKUHUM : VXForm_1<14, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                       "vpkuhum $vD, $vA, $vB", VecFP,
                       [(set VRRC:$vD,
                         (vpkuhum_shuffle (v16i8 VRRC:$vA), VRRC:$vB))]>;
def VPKUHUS : VX1_Int<142, "vpkuhus", int_ppc_altivec_vpkuhus>;
def VPKUWUM : VXForm_1<78, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),
                       "vpkuwum $vD, $vA, $vB", VecFP,
                       [(set VRRC:$vD,
                         (vpkuwum_shuffle (v16i8 VRRC:$vA), VRRC:$vB))]>;
def VPKUWUS : VX1_Int<206, "vpkuwus", int_ppc_altivec_vpkuwus>;

// Vector Unpack.
def VUPKHPX : VX2_Int<846, "vupkhpx", int_ppc_altivec_vupkhpx>;
def VUPKHSB : VX2_Int<526, "vupkhsb", int_ppc_altivec_vupkhsb>;
def VUPKHSH : VX2_Int<590, "vupkhsh", int_ppc_altivec_vupkhsh>;
def VUPKLPX : VX2_Int<974, "vupklpx", int_ppc_altivec_vupklpx>;
def VUPKLSB : VX2_Int<654, "vupklsb", int_ppc_altivec_vupklsb>;
def VUPKLSH : VX2_Int<718, "vupklsh", int_ppc_altivec_vupklsh>;


// Altivec Comparisons.

class VCMP<bits<10> xo, string asmstr, ValueType Ty>
  : VXRForm_1<xo, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),asmstr,VecFPCompare,
              [(set VRRC:$vD, (Ty (PPCvcmp VRRC:$vA, VRRC:$vB, xo)))]>;
class VCMPo<bits<10> xo, string asmstr, ValueType Ty>
  : VXRForm_1<xo, (outs VRRC:$vD), (ins VRRC:$vA, VRRC:$vB),asmstr,VecFPCompare,
              [(set VRRC:$vD, (Ty (PPCvcmp_o VRRC:$vA, VRRC:$vB, xo)))]> {
  let Defs = [CR6];
  let RC = 1;
}

// f32 element comparisons.0
def VCMPBFP   : VCMP <966, "vcmpbfp $vD, $vA, $vB"  , v4f32>;
def VCMPBFPo  : VCMPo<966, "vcmpbfp. $vD, $vA, $vB" , v4f32>;
def VCMPEQFP  : VCMP <198, "vcmpeqfp $vD, $vA, $vB" , v4f32>;
def VCMPEQFPo : VCMPo<198, "vcmpeqfp. $vD, $vA, $vB", v4f32>;
def VCMPGEFP  : VCMP <454, "vcmpgefp $vD, $vA, $vB" , v4f32>;
def VCMPGEFPo : VCMPo<454, "vcmpgefp. $vD, $vA, $vB", v4f32>;
def VCMPGTFP  : VCMP <710, "vcmpgtfp $vD, $vA, $vB" , v4f32>;
def VCMPGTFPo : VCMPo<710, "vcmpgtfp. $vD, $vA, $vB", v4f32>;

// i8 element comparisons.
def VCMPEQUB  : VCMP <  6, "vcmpequb $vD, $vA, $vB" , v16i8>;
def VCMPEQUBo : VCMPo<  6, "vcmpequb. $vD, $vA, $vB", v16i8>;
def VCMPGTSB  : VCMP <774, "vcmpgtsb $vD, $vA, $vB" , v16i8>;
def VCMPGTSBo : VCMPo<774, "vcmpgtsb. $vD, $vA, $vB", v16i8>;
def VCMPGTUB  : VCMP <518, "vcmpgtub $vD, $vA, $vB" , v16i8>;
def VCMPGTUBo : VCMPo<518, "vcmpgtub. $vD, $vA, $vB", v16i8>;

// i16 element comparisons.
def VCMPEQUH  : VCMP < 70, "vcmpequh $vD, $vA, $vB" , v8i16>;
def VCMPEQUHo : VCMPo< 70, "vcmpequh. $vD, $vA, $vB", v8i16>;
def VCMPGTSH  : VCMP <838, "vcmpgtsh $vD, $vA, $vB" , v8i16>;
def VCMPGTSHo : VCMPo<838, "vcmpgtsh. $vD, $vA, $vB", v8i16>;
def VCMPGTUH  : VCMP <582, "vcmpgtuh $vD, $vA, $vB" , v8i16>;
def VCMPGTUHo : VCMPo<582, "vcmpgtuh. $vD, $vA, $vB", v8i16>;

// i32 element comparisons.
def VCMPEQUW  : VCMP <134, "vcmpequw $vD, $vA, $vB" , v4i32>;
def VCMPEQUWo : VCMPo<134, "vcmpequw. $vD, $vA, $vB", v4i32>;
def VCMPGTSW  : VCMP <902, "vcmpgtsw $vD, $vA, $vB" , v4i32>;
def VCMPGTSWo : VCMPo<902, "vcmpgtsw. $vD, $vA, $vB", v4i32>;
def VCMPGTUW  : VCMP <646, "vcmpgtuw $vD, $vA, $vB" , v4i32>;
def VCMPGTUWo : VCMPo<646, "vcmpgtuw. $vD, $vA, $vB", v4i32>;
                      
def V_SET0 : VXForm_setzero<1220, (outs VRRC:$vD), (ins),
                      "vxor $vD, $vD, $vD", VecFP,
                      [(set VRRC:$vD, (v4i32 immAllZerosV))]>;
}

//===----------------------------------------------------------------------===//
// Additional Altivec Patterns
//

// DS* intrinsics
def : Pat<(int_ppc_altivec_dssall), (DSSALL 1, 0, 0, 0)>;
def : Pat<(int_ppc_altivec_dss imm:$STRM), (DSS 0, imm:$STRM, 0, 0)>;

//  * 32-bit
def : Pat<(int_ppc_altivec_dst GPRC:$rA, GPRC:$rB, imm:$STRM),
          (DST 0, imm:$STRM, GPRC:$rA, GPRC:$rB)>;
def : Pat<(int_ppc_altivec_dstt GPRC:$rA, GPRC:$rB, imm:$STRM),
          (DSTT 1, imm:$STRM, GPRC:$rA, GPRC:$rB)>;
def : Pat<(int_ppc_altivec_dstst GPRC:$rA, GPRC:$rB, imm:$STRM),
          (DSTST 0, imm:$STRM, GPRC:$rA, GPRC:$rB)>;
def : Pat<(int_ppc_altivec_dststt GPRC:$rA, GPRC:$rB, imm:$STRM),
          (DSTSTT 1, imm:$STRM, GPRC:$rA, GPRC:$rB)>;

//  * 64-bit
def : Pat<(int_ppc_altivec_dst G8RC:$rA, GPRC:$rB, imm:$STRM),
          (DST64 0, imm:$STRM, (i64 G8RC:$rA), GPRC:$rB)>;
def : Pat<(int_ppc_altivec_dstt G8RC:$rA, GPRC:$rB, imm:$STRM),
          (DSTT64 1, imm:$STRM, (i64 G8RC:$rA), GPRC:$rB)>;
def : Pat<(int_ppc_altivec_dstst G8RC:$rA, GPRC:$rB, imm:$STRM),
          (DSTST64 0, imm:$STRM, (i64 G8RC:$rA), GPRC:$rB)>;
def : Pat<(int_ppc_altivec_dststt G8RC:$rA, GPRC:$rB, imm:$STRM),
          (DSTSTT64 1, imm:$STRM, (i64 G8RC:$rA), GPRC:$rB)>;

// Loads.
def : Pat<(v4i32 (load xoaddr:$src)), (LVX xoaddr:$src)>;

// Stores.
def : Pat<(store (v4i32 VRRC:$rS), xoaddr:$dst),
          (STVX (v4i32 VRRC:$rS), xoaddr:$dst)>;

// Bit conversions.
def : Pat<(v16i8 (bitconvert (v8i16 VRRC:$src))), (v16i8 VRRC:$src)>;
def : Pat<(v16i8 (bitconvert (v4i32 VRRC:$src))), (v16i8 VRRC:$src)>;
def : Pat<(v16i8 (bitconvert (v4f32 VRRC:$src))), (v16i8 VRRC:$src)>;

def : Pat<(v8i16 (bitconvert (v16i8 VRRC:$src))), (v8i16 VRRC:$src)>;
def : Pat<(v8i16 (bitconvert (v4i32 VRRC:$src))), (v8i16 VRRC:$src)>;
def : Pat<(v8i16 (bitconvert (v4f32 VRRC:$src))), (v8i16 VRRC:$src)>;

def : Pat<(v4i32 (bitconvert (v16i8 VRRC:$src))), (v4i32 VRRC:$src)>;
def : Pat<(v4i32 (bitconvert (v8i16 VRRC:$src))), (v4i32 VRRC:$src)>;
def : Pat<(v4i32 (bitconvert (v4f32 VRRC:$src))), (v4i32 VRRC:$src)>;

def : Pat<(v4f32 (bitconvert (v16i8 VRRC:$src))), (v4f32 VRRC:$src)>;
def : Pat<(v4f32 (bitconvert (v8i16 VRRC:$src))), (v4f32 VRRC:$src)>;
def : Pat<(v4f32 (bitconvert (v4i32 VRRC:$src))), (v4f32 VRRC:$src)>;

// Shuffles.

// Match vsldoi(x,x), vpkuwum(x,x), vpkuhum(x,x)
def:Pat<(vsldoi_unary_shuffle:$in (v16i8 VRRC:$vA), undef),
        (VSLDOI VRRC:$vA, VRRC:$vA, (VSLDOI_unary_get_imm VRRC:$in))>;
def:Pat<(vpkuwum_unary_shuffle (v16i8 VRRC:$vA), undef),
        (VPKUWUM VRRC:$vA, VRRC:$vA)>;
def:Pat<(vpkuhum_unary_shuffle (v16i8 VRRC:$vA), undef),
        (VPKUHUM VRRC:$vA, VRRC:$vA)>;

// Match vmrg*(x,x)
def:Pat<(vmrglb_unary_shuffle (v16i8 VRRC:$vA), undef),
        (VMRGLB VRRC:$vA, VRRC:$vA)>;
def:Pat<(vmrglh_unary_shuffle (v16i8 VRRC:$vA), undef),
        (VMRGLH VRRC:$vA, VRRC:$vA)>;
def:Pat<(vmrglw_unary_shuffle (v16i8 VRRC:$vA), undef),
        (VMRGLW VRRC:$vA, VRRC:$vA)>;
def:Pat<(vmrghb_unary_shuffle (v16i8 VRRC:$vA), undef),
        (VMRGHB VRRC:$vA, VRRC:$vA)>;
def:Pat<(vmrghh_unary_shuffle (v16i8 VRRC:$vA), undef),
        (VMRGHH VRRC:$vA, VRRC:$vA)>;
def:Pat<(vmrghw_unary_shuffle (v16i8 VRRC:$vA), undef),
        (VMRGHW VRRC:$vA, VRRC:$vA)>;

// Logical Operations
def : Pat<(v4i32 (vnot_ppc VRRC:$vA)), (VNOR VRRC:$vA, VRRC:$vA)>;

def : Pat<(v4i32 (vnot_ppc (or VRRC:$A, VRRC:$B))),
          (VNOR VRRC:$A, VRRC:$B)>;
def : Pat<(v4i32 (and VRRC:$A, (vnot_ppc VRRC:$B))),
          (VANDC VRRC:$A, VRRC:$B)>;

def : Pat<(fmul VRRC:$vA, VRRC:$vB),
          (VMADDFP VRRC:$vA, VRRC:$vB, (v4i32 (V_SET0)))>; 

// Fused multiply add and multiply sub for packed float.  These are represented
// separately from the real instructions above, for operations that must have
// the additional precision, such as Newton-Rhapson (used by divide, sqrt)
def : Pat<(PPCvmaddfp VRRC:$A, VRRC:$B, VRRC:$C),
          (VMADDFP VRRC:$A, VRRC:$B, VRRC:$C)>;
def : Pat<(PPCvnmsubfp VRRC:$A, VRRC:$B, VRRC:$C),
          (VNMSUBFP VRRC:$A, VRRC:$B, VRRC:$C)>;

def : Pat<(int_ppc_altivec_vmaddfp VRRC:$A, VRRC:$B, VRRC:$C),
          (VMADDFP VRRC:$A, VRRC:$B, VRRC:$C)>;
def : Pat<(int_ppc_altivec_vnmsubfp VRRC:$A, VRRC:$B, VRRC:$C),
          (VNMSUBFP VRRC:$A, VRRC:$B, VRRC:$C)>;

def : Pat<(PPCvperm (v16i8 VRRC:$vA), VRRC:$vB, VRRC:$vC),
          (VPERM VRRC:$vA, VRRC:$vB, VRRC:$vC)>;

// Vector shifts
def : Pat<(v16i8 (shl (v16i8 VRRC:$vA), (v16i8 VRRC:$vB))),
          (v16i8 (VSLB VRRC:$vA, VRRC:$vB))>;
def : Pat<(v8i16 (shl (v8i16 VRRC:$vA), (v8i16 VRRC:$vB))),
          (v8i16 (VSLH VRRC:$vA, VRRC:$vB))>;
def : Pat<(v4i32 (shl (v4i32 VRRC:$vA), (v4i32 VRRC:$vB))),
          (v4i32 (VSLW VRRC:$vA, VRRC:$vB))>;

def : Pat<(v16i8 (srl (v16i8 VRRC:$vA), (v16i8 VRRC:$vB))),
          (v16i8 (VSRB VRRC:$vA, VRRC:$vB))>;
def : Pat<(v8i16 (srl (v8i16 VRRC:$vA), (v8i16 VRRC:$vB))),
          (v8i16 (VSRH VRRC:$vA, VRRC:$vB))>;
def : Pat<(v4i32 (srl (v4i32 VRRC:$vA), (v4i32 VRRC:$vB))),
          (v4i32 (VSRW VRRC:$vA, VRRC:$vB))>;

def : Pat<(v16i8 (sra (v16i8 VRRC:$vA), (v16i8 VRRC:$vB))),
          (v16i8 (VSRAB VRRC:$vA, VRRC:$vB))>;
def : Pat<(v8i16 (sra (v8i16 VRRC:$vA), (v8i16 VRRC:$vB))),
          (v8i16 (VSRAH VRRC:$vA, VRRC:$vB))>;
def : Pat<(v4i32 (sra (v4i32 VRRC:$vA), (v4i32 VRRC:$vB))),
          (v4i32 (VSRAW VRRC:$vA, VRRC:$vB))>;

// Float to integer and integer to float conversions
def : Pat<(v4i32 (fp_to_sint (v4f32 VRRC:$vA))),
           (VCTSXS_0 VRRC:$vA)>;
def : Pat<(v4i32 (fp_to_uint (v4f32 VRRC:$vA))),
           (VCTUXS_0 VRRC:$vA)>;
def : Pat<(v4f32 (sint_to_fp (v4i32 VRRC:$vA))),
           (VCFSX_0 VRRC:$vA)>;
def : Pat<(v4f32 (uint_to_fp (v4i32 VRRC:$vA))),
           (VCFUX_0 VRRC:$vA)>;
