/// Pattern Matchings for VEL intrinsic instructions.

/// Intrinsic patterns written by hand.

// SVOB pattern.
def : Pat<(int_ve_vl_svob), (SVOB)>;

// LSV patterns.
def : Pat<(int_ve_vl_lsv_vvss v256f64:$pt, i32:$sy, i64:$sz),
          (LSVrr_v (i2l i32:$sy), i64:$sz, v256f64:$pt)>;

// LVS patterns.
def : Pat<(int_ve_vl_lvsl_svs v256f64:$vx, i32:$sy),
          (LVSvr v256f64:$vx, (i2l i32:$sy))>;
def : Pat<(int_ve_vl_lvsd_svs v256f64:$vx, i32:$sy),
          (LVSvr v256f64:$vx, (i2l i32:$sy))>;
def : Pat<(int_ve_vl_lvss_svs v256f64:$vx, i32:$sy),
          (l2f (LVSvr v256f64:$vx, (i2l i32:$sy)))>;

/// Intrinsic patterns automatically generated.
include "VEInstrIntrinsicVL.gen.td"
