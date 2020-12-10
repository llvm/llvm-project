// Pattern Matchings for VEL Intrinsics

// Define intrinsics written by hand

// The lsv and lvs patterns
def : Pat<(int_ve_vl_lsv_vvss v256f64:$pt, i32:$sy, i64:$sz),
          (LSVrr_v (INSERT_SUBREG (i64 (IMPLICIT_DEF)), i32:$sy, sub_i32),
                   i64:$sz, v256f64:$pt)>;
def : Pat<(int_ve_vl_lvsl_svs v256f64:$vx, i32:$sy),
          (LVSvr v256f64:$vx,
                 (INSERT_SUBREG (i64 (IMPLICIT_DEF)), i32:$sy, sub_i32))>;
def : Pat<(int_ve_vl_lvsd_svs v256f64:$vx, i32:$sy),
          (LVSvr v256f64:$vx,
                 (INSERT_SUBREG (i64 (IMPLICIT_DEF)), i32:$sy, sub_i32))>;
def : Pat<(int_ve_vl_lvss_svs v256f64:$vx, i32:$sy),
          (EXTRACT_SUBREG (LVSvr v256f64:$vx,
                                 (INSERT_SUBREG (i64 (IMPLICIT_DEF)), i32:$sy,
                                                sub_i32)),
                          sub_f32)>;

// Define intrinsics automatically generated
include "VEInstrIntrinsicVL.gen.td"
