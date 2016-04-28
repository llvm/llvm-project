//==- HexagonAlias.td - Hexagon Instruction Aliases ---------*- tablegen -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//                     Hexagon Instruction Mappings
//===----------------------------------------------------------------------===//

// V6_vassignp: Vector assign mapping.
let hasNewValue = 1, opNewValue = 0, isAsmParserOnly = 1 in
def HEXAGON_V6_vassignpair: CVI_VA_DV_Resource <
  (outs VecDblRegs:$Vdd),
  (ins VecDblRegs:$Vss),
  "$Vdd = $Vss">;

// maps Vd = #0 to Vd = vxor(Vd, Vd)
def : InstAlias<"$Vd = #0",
      (V6_vxor VectorRegs:$Vd, VectorRegs:$Vd, VectorRegs:$Vd)>,
      Requires<[HasV60T]>;

// maps Vdd  = #0 to Vdd = vsub(Vdd, Vdd)
def : InstAlias<"$Vdd = #0",
      (V6_vsubw_dv VecDblRegs:$Vdd, VecDblRegs:$Vdd, VecDblRegs:$Vdd)>,
      Requires<[HasV60T]>;

// maps   "$Qd = vcmp.eq($Vu.uh, $Vv.uh)" -> "$Qd = vcmp.eq($Vu.h, $Vv.h)"
def : InstAlias<"$Qd = vcmp.eq($Vu.uh, $Vv.uh)",
      (V6_veqh VecPredRegs:$Qd, VectorRegs:$Vu, VectorRegs:$Vv)>,
      Requires<[HasV60T]>;

// maps   "$Qd &= vcmp.eq($Vu.uh, $Vv.uh)" -> "$Qd &= vcmp.eq($Vu.h, $Vv.h)"
def : InstAlias<"$Qd &= vcmp.eq($Vu.uh, $Vv.uh)",
      (V6_veqh_and VecPredRegs:$Qd, VectorRegs:$Vu, VectorRegs:$Vv)>,
      Requires<[HasV60T]>;

// maps   "$Qd |= vcmp.eq($Vu.uh, $Vv.uh)" -> "$Qd |= vcmp.eq($Vu.h, $Vv.h)"
def : InstAlias<"$Qd |= vcmp.eq($Vu.uh, $Vv.uh)",
      (V6_veqh_or VecPredRegs:$Qd, VectorRegs:$Vu, VectorRegs:$Vv)>,
      Requires<[HasV60T]>;

// maps   "$Qd ^= vcmp.eq($Vu.uh, $Vv.uh)" -> "$Qd ^= vcmp.eq($Vu.h, $Vv.h)"
def : InstAlias<"$Qd ^= vcmp.eq($Vu.uh, $Vv.uh)",
      (V6_veqh_xor VecPredRegs:$Qd, VectorRegs:$Vu, VectorRegs:$Vv)>,
      Requires<[HasV60T]>;

// maps   "$Qd = vcmp.eq($Vu.uw, $Vv.uw)" -> "$Qd = vcmp.eq($Vu.w, $Vv.w)"
def : InstAlias<"$Qd = vcmp.eq($Vu.uw, $Vv.uw)",
      (V6_veqw VecPredRegs:$Qd, VectorRegs:$Vu, VectorRegs:$Vv)>,
      Requires<[HasV60T]>;

// maps   "$Qd &= vcmp.eq($Vu.uw, $Vv.uw)" -> "$Qd &= vcmp.eq($Vu.w, $Vv.w)"
def : InstAlias<"$Qd &= vcmp.eq($Vu.uw, $Vv.uw)",
      (V6_veqw_and VecPredRegs:$Qd, VectorRegs:$Vu, VectorRegs:$Vv)>,
      Requires<[HasV60T]>;

// maps   "$Qd |= vcmp.eq($Vu.uw, $Vv.uw)" -> "$Qd |= vcmp.eq($Vu.w, $Vv.w)"
def : InstAlias<"$Qd |= vcmp.eq($Vu.uw, $Vv.uw)",
      (V6_veqh_or VecPredRegs:$Qd, VectorRegs:$Vu, VectorRegs:$Vv)>,
      Requires<[HasV60T]>;

// maps   "$Qd ^= vcmp.eq($Vu.uw, $Vv.uw)" -> "$Qd ^= vcmp.eq($Vu.w, $Vv.w)"
def : InstAlias<"$Qd ^= vcmp.eq($Vu.uw, $Vv.uw)",
      (V6_veqw_xor VecPredRegs:$Qd, VectorRegs:$Vu, VectorRegs:$Vv)>,
      Requires<[HasV60T]>;

// maps   "$Qd = vcmp.eq($Vu.ub, $Vv.ub)" -> "$Qd = vcmp.eq($Vu.b, $Vv.b)"
def : InstAlias<"$Qd = vcmp.eq($Vu.ub, $Vv.ub)",
      (V6_veqb VecPredRegs:$Qd, VectorRegs:$Vu, VectorRegs:$Vv)>,
      Requires<[HasV60T]>;

// maps   "$Qd &= vcmp.eq($Vu.ub, $Vv.ub)" -> "$Qd &= vcmp.eq($Vu.b, $Vv.b)"
def : InstAlias<"$Qd &= vcmp.eq($Vu.ub, $Vv.ub)",
      (V6_veqb_and VecPredRegs:$Qd, VectorRegs:$Vu, VectorRegs:$Vv)>,
      Requires<[HasV60T]>;

// maps   "$Qd |= vcmp.eq($Vu.ub, $Vv.ub)" -> "$Qd |= vcmp.eq($Vu.b, $Vv.b)"
def : InstAlias<"$Qd |= vcmp.eq($Vu.ub, $Vv.ub)",
      (V6_veqb_or VecPredRegs:$Qd, VectorRegs:$Vu, VectorRegs:$Vv)>,
      Requires<[HasV60T]>;

// maps   "$Qd ^= vcmp.eq($Vu.ub, $Vv.ub)" -> "$Qd ^= vcmp.eq($Vu.b, $Vv.b)"
def : InstAlias<"$Qd ^= vcmp.eq($Vu.ub, $Vv.ub)",
      (V6_veqb_xor VecPredRegs:$Qd, VectorRegs:$Vu, VectorRegs:$Vv)>,
      Requires<[HasV60T]>;

// maps   "$Rd.w = vextract($Vu, $Rs)" -> "$Rd = vextract($Vu, $Rs)"
def : InstAlias<"$Rd.w = vextract($Vu, $Rs)",
      (V6_extractw IntRegs:$Rd, VectorRegs:$Vu, IntRegs:$Rs)>,
      Requires<[HasV60T]>;
