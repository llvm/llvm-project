//====--- SPU64InstrInfo.td - Cell SPU 64-bit operations -*- tablegen -*--====//
//
//                     Cell SPU 64-bit operations
//
//===----------------------------------------------------------------------===//

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// 64-bit comparisons:
//
// 1. The instruction sequences for vector vice scalar differ by a
//    constant. In the scalar case, we're only interested in the
//    top two 32-bit slots, whereas we're interested in an exact
//    all-four-slot match in the vector case.
//
// 2. There are no "immediate" forms, since loading 64-bit constants
//    could be a constant pool load.
//
// 3. i64 setcc results are i32, which are subsequently converted to a FSM
//    mask when used in a select pattern.
//
// 4. v2i64 setcc results are v4i32, which can be converted to a FSM mask (TODO)
//    [Note: this may be moot, since gb produces v4i32 or r32.]
//
// 5. The code sequences for r64 and v2i64 are probably overly conservative,
//    compared to the code that gcc produces.
//
// M00$E B!tes Kan be Pretty N@sTi!!!!! (appologies to Monty!)
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

// selb instruction definition for i64. Note that the selection mask is
// a vector, produced by various forms of FSM:
def SELBr64_cond:
  SELBInst<(outs R64C:$rT), (ins R64C:$rA, R64C:$rB, VECREG:$rC),
           [/* no pattern */]>;

// The generic i64 select pattern, which assumes that the comparison result
// is in a 32-bit register that contains a select mask pattern (i.e., gather
// bits result):

def : Pat<(select R32C:$rCond, R64C:$rFalse, R64C:$rTrue),
          (SELBr64_cond R64C:$rTrue, R64C:$rFalse, (FSMr32 R32C:$rCond))>;

// select the negative condition:
class I64SELECTNegCond<PatFrag cond, CodeFrag compare>:
  Pat<(select (i32 (cond R64C:$rA, R64C:$rB)), R64C:$rTrue, R64C:$rFalse),
      (SELBr64_cond R64C:$rTrue, R64C:$rFalse, (FSMr32 compare.Fragment))>;

// setcc the negative condition:
class I64SETCCNegCond<PatFrag cond, CodeFrag compare>:
  Pat<(cond R64C:$rA, R64C:$rB),
      (XORIr32 compare.Fragment, -1)>;

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// The i64 seteq fragment that does the scalar->vector conversion and
// comparison:
def CEQr64compare:
    CodeFrag<(CGTIv4i32 (GBv4i32 (CEQv4i32 (ORv2i64_i64 R64C:$rA),
                                           (ORv2i64_i64 R64C:$rB))), 0xb)>;

// The i64 seteq fragment that does the vector comparison
def CEQv2i64compare:
    CodeFrag<(CEQIv4i32 (GBv4i32 (CEQv4i32 VECREG:$rA, VECREG:$rB)), 0xf)>;

// i64 seteq (equality): the setcc result is i32, which is converted to a
// vector FSM mask when used in a select pattern.
//
// v2i64 seteq (equality): the setcc result is v4i32
multiclass CompareEqual64 {
  // Plain old comparison, converts back to i32 scalar
  def r64: CodeFrag<(ORi32_v4i32 CEQr64compare.Fragment)>;
  def v2i64: CodeFrag<(ORi32_v4i32 CEQv2i64compare.Fragment)>;

  // SELB mask from FSM:
  def r64mask: CodeFrag<(ORi32_v4i32 (FSMv4i32 CEQr64compare.Fragment))>;
  def v2i64mask: CodeFrag<(ORi32_v4i32 (FSMv4i32 CEQv2i64compare.Fragment))>;
}

defm I64EQ: CompareEqual64;

def : Pat<(seteq R64C:$rA, R64C:$rB), I64EQr64.Fragment>;
def : Pat<(seteq (v2i64 VECREG:$rA), (v2i64 VECREG:$rB)), I64EQv2i64.Fragment>;

// i64 setne:
def : I64SETCCNegCond<setne, I64EQr64>;
def : I64SELECTNegCond<setne, I64EQr64>;

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// i64 setugt/setule:
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

def CLGTr64ugt:
    CodeFrag<(CLGTv4i32 (ORv2i64_i64 R64C:$rA), (ORv2i64_i64 R64C:$rB))>;

def CLGTr64eq:
    CodeFrag<(CEQv4i32 (ORv2i64_i64 R64C:$rA), (ORv2i64_i64 R64C:$rB))>;
    
def CLGTr64compare:
    CodeFrag<(SELBv2i64 CLGTr64ugt.Fragment,
                        (XSWDv2i64 CLGTr64ugt.Fragment),
                        CLGTr64eq.Fragment)>;

def CLGTv2i64ugt:
    CodeFrag<(CLGTv4i32 VECREG:$rA, VECREG:$rB)>;

def CLGTv2i64eq:
    CodeFrag<(CEQv4i32 VECREG:$rA, VECREG:$rB)>;
    
def CLGTv2i64compare:
    CodeFrag<(SELBv2i64 CLGTv2i64ugt.Fragment,
                        (XSWDv2i64 CLGTr64ugt.Fragment),
                        CLGTv2i64eq.Fragment)>;

multiclass CompareLogicalGreaterThan64 {
  // Plain old comparison, converts back to i32 scalar
  def r64: CodeFrag<(ORi32_v4i32 CLGTr64compare.Fragment)>;
  def v2i64: CodeFrag<CLGTv2i64compare.Fragment>;

  // SELB mask from FSM:
  def r64mask: CodeFrag<(ORi32_v4i32 (FSMv4i32 CLGTr64compare.Fragment))>;
  def v2i64mask: CodeFrag<(ORi32_v4i32 (FSMv4i32 CLGTv2i64compare.Fragment))>;
}

defm I64LGT: CompareLogicalGreaterThan64;

def : Pat<(setugt R64C:$rA, R64C:$rB), I64LGTr64.Fragment>;
def : Pat<(setugt (v2i64 VECREG:$rA), (v2i64 VECREG:$rB)),
                  I64LGTv2i64.Fragment>;

// i64 setult:
def : I64SETCCNegCond<setule, I64LGTr64>;
def : I64SELECTNegCond<setule, I64LGTr64>;

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// i64 setuge/setult:
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

def CLGEr64compare:
    CodeFrag<(CGTIv4i32 (GBv4i32 (ORv4i32 CLGTr64ugt.Fragment,
                                          CLGTr64eq.Fragment)), 0xb)>;

def CLGEv2i64compare:
    CodeFrag<(CEQIv4i32 (GBv4i32 (ORv4i32 CLGTv2i64ugt.Fragment,
                                          CLGTv2i64eq.Fragment)), 0xf)>;

multiclass CompareLogicalGreaterEqual64 {
  // Plain old comparison, converts back to i32 scalar
  def r64: CodeFrag<(ORi32_v4i32 CLGEr64compare.Fragment)>;
  def v2i64: CodeFrag<CLGEv2i64compare.Fragment>;

  // SELB mask from FSM:
  def r64mask: CodeFrag<(ORi32_v4i32 (FSMv4i32 CLGEr64compare.Fragment))>;
  def v2i64mask: CodeFrag<(ORi32_v4i32 (FSMv4i32 CLGEv2i64compare.Fragment))>;
}

defm I64LGE: CompareLogicalGreaterEqual64;

def : Pat<(setuge R64C:$rA, R64C:$rB), I64LGEr64.Fragment>;
def : Pat<(setuge (v2i64 VECREG:$rA), (v2i64 VECREG:$rB)),
                  I64LGEv2i64.Fragment>;

// i64 setult:
def : I64SETCCNegCond<setult, I64LGEr64>;
def : I64SELECTNegCond<setult, I64LGEr64>;

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// i64 setgt/setle:
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

def CGTr64sgt:
    CodeFrag<(CGTv4i32 (ORv2i64_i64 R64C:$rA), (ORv2i64_i64 R64C:$rB))>;

def CGTr64eq:
    CodeFrag<(CEQv4i32 (ORv2i64_i64 R64C:$rA), (ORv2i64_i64 R64C:$rB))>;
    
def CGTr64compare:
    CodeFrag<(SELBv2i64 CGTr64sgt.Fragment,
                        (XSWDv2i64 CGTr64sgt.Fragment),
                        CGTr64eq.Fragment)>;

def CGTv2i64sgt:
    CodeFrag<(CGTv4i32 VECREG:$rA, VECREG:$rB)>;

def CGTv2i64eq:
    CodeFrag<(CEQv4i32 VECREG:$rA, VECREG:$rB)>;
    
def CGTv2i64compare:
    CodeFrag<(SELBv2i64 CGTv2i64sgt.Fragment,
                        (XSWDv2i64 CGTr64sgt.Fragment),
                        CGTv2i64eq.Fragment)>;

multiclass CompareGreaterThan64 {
  // Plain old comparison, converts back to i32 scalar
  def r64: CodeFrag<(ORi32_v4i32 CGTr64compare.Fragment)>;
  def v2i64: CodeFrag<CGTv2i64compare.Fragment>;

  // SELB mask from FSM:
  def r64mask: CodeFrag<(ORi32_v4i32 (FSMv4i32 CGTr64compare.Fragment))>;
  def v2i64mask: CodeFrag<(ORi32_v4i32 (FSMv4i32 CGTv2i64compare.Fragment))>;
}

defm I64GT: CompareLogicalGreaterThan64;

def : Pat<(setgt R64C:$rA, R64C:$rB), I64GTr64.Fragment>;
def : Pat<(setgt (v2i64 VECREG:$rA), (v2i64 VECREG:$rB)),
                  I64GTv2i64.Fragment>;

// i64 setult:
def : I64SETCCNegCond<setle, I64GTr64>;
def : I64SELECTNegCond<setle, I64GTr64>;

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// i64 setge/setlt:
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    
def CGEr64compare:
    CodeFrag<(CGTIv4i32 (GBv4i32 (ORv4i32 CGTr64sgt.Fragment,
                                          CGTr64eq.Fragment)), 0xb)>;

def CGEv2i64compare:
    CodeFrag<(CEQIv4i32 (GBv4i32 (ORv4i32 CGTv2i64sgt.Fragment,
                                          CGTv2i64eq.Fragment)), 0xf)>;

multiclass CompareGreaterEqual64 {
  // Plain old comparison, converts back to i32 scalar
  def r64: CodeFrag<(ORi32_v4i32 CGEr64compare.Fragment)>;
  def v2i64: CodeFrag<CGEv2i64compare.Fragment>;

  // SELB mask from FSM:
  def r64mask: CodeFrag<(ORi32_v4i32 (FSMv4i32 CGEr64compare.Fragment))>;
  def v2i64mask: CodeFrag<(ORi32_v4i32 (FSMv4i32 CGEv2i64compare.Fragment))>;
}

defm I64GE: CompareGreaterEqual64;

def : Pat<(setge R64C:$rA, R64C:$rB), I64GEr64.Fragment>;
def : Pat<(setge (v2i64 VECREG:$rA), (v2i64 VECREG:$rB)),
                  I64GEv2i64.Fragment>;

// i64 setult:
def : I64SETCCNegCond<setlt, I64GEr64>;
def : I64SELECTNegCond<setlt, I64GEr64>;

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// v2i64, i64 add
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

class v2i64_add_cg<dag lhs, dag rhs>:
    CodeFrag<(CGv4i32 lhs, rhs)>;

class v2i64_add_1<dag lhs, dag rhs, dag cg, dag cg_mask>:
    CodeFrag<(ADDXv4i32 lhs, rhs, (SHUFBv4i32 cg, cg, cg_mask))>;

class v2i64_add<dag lhs, dag rhs, dag cg_mask>:
    v2i64_add_1<lhs, rhs, v2i64_add_cg<lhs, rhs>.Fragment, cg_mask>;

def : Pat<(SPUadd64 R64C:$rA, R64C:$rB, (v4i32 VECREG:$rCGmask)),
           (ORi64_v2i64 v2i64_add<(ORv2i64_i64 R64C:$rA),
                                  (ORv2i64_i64 R64C:$rB),
                                  (v4i32 VECREG:$rCGmask)>.Fragment)>;

def : Pat<(SPUadd64 (v2i64 VECREG:$rA), (v2i64 VECREG:$rB),
                    (v4i32 VECREG:$rCGmask)),
           v2i64_add<(v2i64 VECREG:$rA),
                     (v2i64 VECREG:$rB),
                     (v4i32 VECREG:$rCGmask)>.Fragment>;

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// v2i64, i64 subtraction
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

class v2i64_sub_bg<dag lhs, dag rhs>: CodeFrag<(BGv4i32 lhs, rhs)>;

class v2i64_sub<dag lhs, dag rhs, dag bg, dag bg_mask>:
    CodeFrag<(SFXv4i32 lhs, rhs, (SHUFBv4i32 bg, bg, bg_mask))>;

def : Pat<(SPUsub64 R64C:$rA, R64C:$rB, (v4i32 VECREG:$rCGmask)),
           (ORi64_v2i64 v2i64_sub<(ORv2i64_i64 R64C:$rA),
                                  (ORv2i64_i64 R64C:$rB),
                                  v2i64_sub_bg<(ORv2i64_i64 R64C:$rA),
                                               (ORv2i64_i64 R64C:$rB)>.Fragment,
                                  (v4i32 VECREG:$rCGmask)>.Fragment)>;

def : Pat<(SPUsub64 (v2i64 VECREG:$rA), (v2i64 VECREG:$rB),
                    (v4i32 VECREG:$rCGmask)),
           v2i64_sub<(v2i64 VECREG:$rA),
                     (v2i64 VECREG:$rB),
                     v2i64_sub_bg<(v2i64 VECREG:$rA),
                                  (v2i64 VECREG:$rB)>.Fragment,
                     (v4i32 VECREG:$rCGmask)>.Fragment>;

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// v2i64, i64 multiply
//
// Note: i64 multiply is simply the vector->scalar conversion of the
// full-on v2i64 multiply, since the entire vector has to be manipulated
// anyway.
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

class v2i64_mul_ahi64<dag rA> :
    CodeFrag<(SELBv4i32 rA, (ILv4i32 0), (FSMBIv4i32 0x0f0f))>;

class v2i64_mul_bhi64<dag rB> :
    CodeFrag<(SELBv4i32 rB, (ILv4i32 0), (FSMBIv4i32 0x0f0f))>;

class v2i64_mul_alo64<dag rB> :
    CodeFrag<(SELBv4i32 rB, (ILv4i32 0), (FSMBIv4i32 0xf0f0))>;

class v2i64_mul_blo64<dag rB> :
    CodeFrag<(SELBv4i32 rB, (ILv4i32 0), (FSMBIv4i32 0xf0f0))>;

class v2i64_mul_ashlq2<dag rA>:
    CodeFrag<(SHLQBYIv4i32 rA, 0x2)>;

class v2i64_mul_ashlq4<dag rA>:
    CodeFrag<(SHLQBYIv4i32 rA, 0x4)>;

class v2i64_mul_bshlq2<dag rB> :
    CodeFrag<(SHLQBYIv4i32 rB, 0x2)>;

class v2i64_mul_bshlq4<dag rB> :
    CodeFrag<(SHLQBYIv4i32 rB, 0x4)>;

class v2i64_highprod<dag rA, dag rB>:
    CodeFrag<(Av4i32
                (Av4i32
                  (MPYUv4i32 v2i64_mul_bshlq4<rB>.Fragment,     // a1 x b3
                             v2i64_mul_ahi64<rA>.Fragment),
                  (MPYHv4i32 v2i64_mul_ahi64<rA>.Fragment,      // a0 x b3
                             v2i64_mul_bshlq4<rB>.Fragment)),
                (Av4i32
                  (MPYHv4i32 v2i64_mul_bhi64<rB>.Fragment,
                             v2i64_mul_ashlq4<rA>.Fragment),
                  (Av4i32
                      (MPYHv4i32 v2i64_mul_ashlq4<rA>.Fragment,
                                 v2i64_mul_bhi64<rB>.Fragment),
                    (Av4i32
                      (MPYUv4i32 v2i64_mul_ashlq4<rA>.Fragment,
                                 v2i64_mul_bhi64<rB>.Fragment),
                      (Av4i32
                        (MPYHv4i32 v2i64_mul_ashlq2<rA>.Fragment,
                                   v2i64_mul_bshlq2<rB>.Fragment),
                        (MPYUv4i32 v2i64_mul_ashlq2<rA>.Fragment,
                                   v2i64_mul_bshlq2<rB>.Fragment))))))>;

class v2i64_mul_a3_b3<dag rA, dag rB>:
    CodeFrag<(MPYUv4i32 v2i64_mul_alo64<rA>.Fragment,
                        v2i64_mul_blo64<rB>.Fragment)>;

class v2i64_mul_a2_b3<dag rA, dag rB>:
    CodeFrag<(SELBv4i32 (SHLQBYIv4i32
                          (MPYHHUv4i32 v2i64_mul_alo64<rA>.Fragment,
                                       v2i64_mul_bshlq2<rB>.Fragment), 0x2),
                        (ILv4i32 0),
                        (FSMBIv4i32 0xc3c3))>;

class v2i64_mul_a3_b2<dag rA, dag rB>:
    CodeFrag<(SELBv4i32 (SHLQBYIv4i32
                          (MPYHHUv4i32 v2i64_mul_blo64<rB>.Fragment,
                                       v2i64_mul_ashlq2<rA>.Fragment), 0x2),
                        (ILv4i32 0),
                        (FSMBIv4i32 0xc3c3))>;

class v2i64_lowsum<dag rA, dag rB, dag rCGmask>:
    v2i64_add<v2i64_add<v2i64_mul_a3_b3<rA, rB>.Fragment,
                        v2i64_mul_a2_b3<rA, rB>.Fragment, rCGmask>.Fragment,
              v2i64_mul_a3_b2<rA, rB>.Fragment, rCGmask>;

class v2i64_mul<dag rA, dag rB, dag rCGmask>:
    v2i64_add<v2i64_lowsum<rA, rB, rCGmask>.Fragment,
              (SELBv4i32 v2i64_highprod<rA, rB>.Fragment,
                         (ILv4i32 0),
                         (FSMBIv4i32 0x0f0f)),
              rCGmask>;

def : Pat<(SPUmul64 R64C:$rA, R64C:$rB, (v4i32 VECREG:$rCGmask)),
          (ORi64_v2i64 v2i64_mul<(ORv2i64_i64 R64C:$rA),
                                 (ORv2i64_i64 R64C:$rB),
                                 (v4i32 VECREG:$rCGmask)>.Fragment)>;

def : Pat<(SPUmul64 (v2i64 VECREG:$rA), (v2i64 VECREG:$rB),
                    (v4i32 VECREG:$rCGmask)),
          v2i64_mul<(v2i64 VECREG:$rA), (v2i64 VECREG:$rB),
                    (v4i32 VECREG:$rCGmask)>.Fragment>;

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// f64 comparisons
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

// selb instruction definition for i64. Note that the selection mask is
// a vector, produced by various forms of FSM:
def SELBf64_cond:
   SELBInst<(outs R64FP:$rT), (ins R64FP:$rA, R64FP:$rB, R32C:$rC),
            [(set R64FP:$rT,
                  (select R32C:$rC, R64FP:$rB, R64FP:$rA))]>;
