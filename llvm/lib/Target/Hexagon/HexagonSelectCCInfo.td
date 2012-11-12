//===-- HexagoSelectCCInfo.td - Selectcc mappings ----------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


//
// selectcc mappings.
//
def : Pat <(i32 (selectcc IntRegs:$lhs, IntRegs:$rhs, IntRegs:$tval,
                          IntRegs:$fval, SETEQ)),
      (i32 (MUX_rr (i1 (CMPEQrr IntRegs:$lhs, IntRegs:$rhs)),
                   IntRegs:$tval, IntRegs:$fval))>;

def : Pat <(i32 (selectcc IntRegs:$lhs, IntRegs:$rhs, IntRegs:$tval,
                          IntRegs:$fval, SETNE)),
      (i32 (MUX_rr (i1 (NOT_p (CMPEQrr IntRegs:$lhs, IntRegs:$rhs))),
                   IntRegs:$tval, IntRegs:$fval))>;

def : Pat <(i32 (selectcc IntRegs:$lhs, IntRegs:$rhs, IntRegs:$tval,
                          IntRegs:$fval, SETGT)),
      (i32 (MUX_rr (i1 (CMPGTrr IntRegs:$lhs, IntRegs:$rhs)),
                   IntRegs:$tval, IntRegs:$fval))>;

def : Pat <(i32 (selectcc IntRegs:$lhs, IntRegs:$rhs, IntRegs:$tval,
                          IntRegs:$fval, SETUGT)),
      (i32 (MUX_rr (i1 (CMPGTUrr IntRegs:$lhs, IntRegs:$rhs)),
                   IntRegs:$tval, IntRegs:$fval))>;



def : Pat <(i32 (selectcc IntRegs:$lhs, IntRegs:$rhs, IntRegs:$tval,
                          IntRegs:$fval, SETULT)),
      (i32 (MUX_rr (i1 (NOT_p (CMPGTUrr IntRegs:$lhs,
                                         (ADD_ri IntRegs:$rhs, -1)))),
                   IntRegs:$tval, IntRegs:$fval))>;

def : Pat <(i32 (selectcc IntRegs:$lhs, IntRegs:$rhs, IntRegs:$tval,
                          IntRegs:$fval, SETLT)),
      (i32 (MUX_rr (i1 (NOT_p (CMPGTrr IntRegs:$lhs,
                                        (ADD_ri IntRegs:$rhs, -1)))),
                   IntRegs:$tval, IntRegs:$fval))>;

def : Pat <(i32 (selectcc IntRegs:$lhs, IntRegs:$rhs, IntRegs:$tval,
                          IntRegs:$fval, SETLE)),
      (i32 (MUX_rr (i1 (NOT_p (CMPGTrr IntRegs:$lhs, IntRegs:$rhs))),
                   IntRegs:$tval, IntRegs:$fval))>;

def : Pat <(i32 (selectcc IntRegs:$lhs, IntRegs:$rhs, IntRegs:$tval,
                          IntRegs:$fval, SETULE)),
      (i32 (MUX_rr (i1 (NOT_p (CMPGTUrr IntRegs:$lhs, IntRegs:$rhs))),
                   IntRegs:$tval, IntRegs:$fval))>;


//
// selectcc mappings for greater-equal-to Rs => greater-than Rs-1.
//
def : Pat <(i32 (selectcc IntRegs:$lhs, IntRegs:$rhs, IntRegs:$tval,
                          IntRegs:$fval, SETGE)),
      (i32 (MUX_rr (i1 (CMPGTrr IntRegs:$lhs, (ADD_ri IntRegs:$rhs, -1))),
                   IntRegs:$tval, IntRegs:$fval))>;

def : Pat <(i32 (selectcc IntRegs:$lhs, IntRegs:$rhs, IntRegs:$tval,
                          IntRegs:$fval, SETUGE)),
      (i32 (MUX_rr (i1 (CMPGTUrr IntRegs:$lhs, (ADD_ri IntRegs:$rhs, -1))),
                   IntRegs:$tval, IntRegs:$fval))>;



//
// selectcc mappings for predicate comparisons.
//
// Convert Rd = selectcc(p0, p1, true_val, false_val, SETEQ) into:
//  pt = not(p1 xor p2)
//  Rd = mux(pt, true_val, false_val)
// and similarly for SETNE
//
def : Pat <(i32 (selectcc PredRegs:$lhs, PredRegs:$rhs, IntRegs:$tval,
                          IntRegs:$fval, SETNE)),
      (i32 (MUX_rr (i1 (XOR_pp PredRegs:$lhs, PredRegs:$rhs)), IntRegs:$tval,
                   IntRegs:$fval))>;

def : Pat <(i32 (selectcc PredRegs:$lhs, PredRegs:$rhs, IntRegs:$tval,
                          IntRegs:$fval, SETEQ)),
      (i32 (MUX_rr (i1 (NOT_p (XOR_pp PredRegs:$lhs, PredRegs:$rhs))),
                   IntRegs:$tval, IntRegs:$fval))>;


//
// selectcc mappings for 64-bit operands are messy. Hexagon does not have a
// MUX64 o, use this:
// selectcc(Rss, Rdd, tval, fval, cond) ->
//   combine(mux(cmp_cond(Rss, Rdd), tval.hi, fval.hi),
//           mux(cmp_cond(Rss, Rdd), tval.lo, fval.lo))

// setgt-64.
def : Pat<(i64 (selectcc DoubleRegs:$lhs, DoubleRegs:$rhs, DoubleRegs:$tval,
                         DoubleRegs:$fval, SETGT)),
      (COMBINE_rr (MUX_rr (CMPGT64rr DoubleRegs:$lhs, DoubleRegs:$rhs),
                           (EXTRACT_SUBREG DoubleRegs:$tval, subreg_hireg),
                           (EXTRACT_SUBREG DoubleRegs:$fval, subreg_hireg)),
                   (MUX_rr (CMPGT64rr DoubleRegs:$lhs, DoubleRegs:$rhs),
                           (EXTRACT_SUBREG DoubleRegs:$tval, subreg_loreg),
                           (EXTRACT_SUBREG DoubleRegs:$fval, subreg_loreg)))>;


// setlt-64 -> setgt-64.
def : Pat<(i64 (selectcc DoubleRegs:$lhs, DoubleRegs:$rhs, DoubleRegs:$tval,
                         DoubleRegs:$fval, SETLT)),
      (COMBINE_rr (MUX_rr (CMPGT64rr DoubleRegs:$lhs,
                                     (ADD64_rr DoubleRegs:$rhs, (TFRI64 -1))),
                           (EXTRACT_SUBREG DoubleRegs:$tval, subreg_hireg),
                           (EXTRACT_SUBREG DoubleRegs:$fval, subreg_hireg)),
                   (MUX_rr (CMPGT64rr DoubleRegs:$lhs,
                                      (ADD64_rr DoubleRegs:$rhs, (TFRI64 -1))),
                           (EXTRACT_SUBREG DoubleRegs:$tval, subreg_loreg),
                           (EXTRACT_SUBREG DoubleRegs:$fval, subreg_loreg)))>;
