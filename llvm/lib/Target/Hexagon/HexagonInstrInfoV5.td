def SDTHexagonFCONST32 : SDTypeProfile<1, 1, [
                                            SDTCisVT<0, f32>,
                                            SDTCisPtrTy<1>]>;
def HexagonFCONST32 : SDNode<"HexagonISD::FCONST32",     SDTHexagonFCONST32>;

let isReMaterializable = 1, isMoveImm = 1 in
def FCONST32_nsdata : LDInst<(outs IntRegs:$dst), (ins globaladdress:$global),
              "$dst = CONST32(#$global)",
              [(set (f32 IntRegs:$dst),
              (HexagonFCONST32 tglobaladdr:$global))]>,
               Requires<[HasV5T]>;

let isReMaterializable = 1, isMoveImm = 1 in
def CONST64_Float_Real : LDInst<(outs DoubleRegs:$dst), (ins f64imm:$src1),
                       "$dst = CONST64(#$src1)",
                       [(set DoubleRegs:$dst, fpimm:$src1)]>,
          Requires<[HasV5T]>;

let isReMaterializable = 1, isMoveImm = 1 in
def CONST32_Float_Real : LDInst<(outs IntRegs:$dst), (ins f32imm:$src1),
                       "$dst = CONST32(#$src1)",
                       [(set IntRegs:$dst, fpimm:$src1)]>,
          Requires<[HasV5T]>;

// Transfer immediate float.
// Only works with single precision fp value.
// For double precision, use CONST64_float_real, as 64bit transfer
// can only hold 40-bit values - 32 from const ext + 8 bit immediate.
let isMoveImm = 1, isReMaterializable = 1, isPredicable = 1 in
def TFRI_f : ALU32_ri<(outs IntRegs:$dst), (ins f32imm:$src1),
           "$dst = ##$src1",
           [(set IntRegs:$dst, fpimm:$src1)]>,
          Requires<[HasV5T]>;

def TFRI_cPt_f : ALU32_ri<(outs IntRegs:$dst),
                          (ins PredRegs:$src1, f32imm:$src2),
           "if ($src1) $dst = ##$src2",
           []>,
          Requires<[HasV5T]>;

let isPredicated = 1 in
def TFRI_cNotPt_f : ALU32_ri<(outs IntRegs:$dst),
                             (ins PredRegs:$src1, f32imm:$src2),
           "if (!$src1) $dst = ##$src2",
           []>,
          Requires<[HasV5T]>;

// Convert single precision to double precision and vice-versa.
def CONVERT_sf2df : ALU64_rr<(outs DoubleRegs:$dst), (ins IntRegs:$src),
                "$dst = convert_sf2df($src)",
                [(set DoubleRegs:$dst, (fextend IntRegs:$src))]>,
          Requires<[HasV5T]>;

def CONVERT_df2sf : ALU64_rr<(outs IntRegs:$dst), (ins DoubleRegs:$src),
                "$dst = convert_df2sf($src)",
                [(set IntRegs:$dst, (fround DoubleRegs:$src))]>,
          Requires<[HasV5T]>;


// Load.
def LDrid_f : LDInst<(outs DoubleRegs:$dst),
            (ins MEMri:$addr),
            "$dst = memd($addr)",
            [(set DoubleRegs:$dst, (f64 (load ADDRriS11_3:$addr)))]>,
          Requires<[HasV5T]>;


let AddedComplexity = 20 in
def LDrid_indexed_f : LDInst<(outs DoubleRegs:$dst),
            (ins IntRegs:$src1, s11_3Imm:$offset),
            "$dst = memd($src1+#$offset)",
            [(set DoubleRegs:$dst, (f64 (load (add IntRegs:$src1,
                                              s11_3ImmPred:$offset))))]>,
          Requires<[HasV5T]>;

def LDriw_f : LDInst<(outs IntRegs:$dst),
            (ins MEMri:$addr), "$dst = memw($addr)",
            [(set IntRegs:$dst, (f32 (load ADDRriS11_2:$addr)))]>,
          Requires<[HasV5T]>;


let AddedComplexity = 20 in
def LDriw_indexed_f : LDInst<(outs IntRegs:$dst),
            (ins IntRegs:$src1, s11_2Imm:$offset),
            "$dst = memw($src1+#$offset)",
            [(set IntRegs:$dst, (f32 (load (add IntRegs:$src1,
                                           s11_2ImmPred:$offset))))]>,
          Requires<[HasV5T]>;

// Store.
def STriw_f : STInst<(outs),
            (ins MEMri:$addr, IntRegs:$src1),
            "memw($addr) = $src1",
            [(store (f32 IntRegs:$src1), ADDRriS11_2:$addr)]>,
          Requires<[HasV5T]>;

let AddedComplexity = 10 in
def STriw_indexed_f : STInst<(outs),
            (ins IntRegs:$src1, s11_2Imm:$src2, IntRegs:$src3),
            "memw($src1+#$src2) = $src3",
            [(store (f32 IntRegs:$src3),
                (add IntRegs:$src1, s11_2ImmPred:$src2))]>,
          Requires<[HasV5T]>;

def STrid_f : STInst<(outs),
            (ins MEMri:$addr, DoubleRegs:$src1),
            "memd($addr) = $src1",
            [(store (f64 DoubleRegs:$src1), ADDRriS11_2:$addr)]>,
          Requires<[HasV5T]>;

// Indexed store double word.
let AddedComplexity = 10 in
def STrid_indexed_f : STInst<(outs),
            (ins IntRegs:$src1, s11_3Imm:$src2,  DoubleRegs:$src3),
            "memd($src1+#$src2) = $src3",
            [(store (f64 DoubleRegs:$src3),
                                (add IntRegs:$src1, s11_3ImmPred:$src2))]>,
          Requires<[HasV5T]>;


// Add
let isCommutable = 1 in
def fADD_rr : ALU64_rr<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2),
            "$dst = sfadd($src1, $src2)",
            [(set IntRegs:$dst, (fadd IntRegs:$src1, IntRegs:$src2))]>,
          Requires<[HasV5T]>;

let isCommutable = 1 in
def fADD64_rr : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1,
                                                     DoubleRegs:$src2),
               "$dst = dfadd($src1, $src2)",
               [(set DoubleRegs:$dst, (fadd DoubleRegs:$src1,
                                           DoubleRegs:$src2))]>,
          Requires<[HasV5T]>;

def fSUB_rr : ALU64_rr<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2),
            "$dst = sfsub($src1, $src2)",
            [(set IntRegs:$dst, (fsub IntRegs:$src1, IntRegs:$src2))]>,
          Requires<[HasV5T]>;

def fSUB64_rr : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1,
                                                     DoubleRegs:$src2),
               "$dst = dfsub($src1, $src2)",
               [(set DoubleRegs:$dst, (fsub DoubleRegs:$src1,
                                           DoubleRegs:$src2))]>,
               Requires<[HasV5T]>;

let isCommutable = 1 in
def fMUL_rr : ALU64_rr<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2),
            "$dst = sfmpy($src1, $src2)",
            [(set IntRegs:$dst, (fmul IntRegs:$src1, IntRegs:$src2))]>,
            Requires<[HasV5T]>;

let isCommutable = 1 in
def fMUL64_rr : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1,
                                                     DoubleRegs:$src2),
               "$dst = dfmpy($src1, $src2)",
               [(set DoubleRegs:$dst, (fmul DoubleRegs:$src1,
                                           DoubleRegs:$src2))]>,
               Requires<[HasV5T]>;

// Compare.
let isCompare = 1 in {
multiclass FCMP64_rr<string OpcStr, PatFrag OpNode> {
  def _rr : ALU64_rr<(outs PredRegs:$dst), (ins DoubleRegs:$b, DoubleRegs:$c),
                 !strconcat("$dst = ", !strconcat(OpcStr, "($b, $c)")),
                 [(set PredRegs:$dst,
                        (OpNode (f64 DoubleRegs:$b), (f64 DoubleRegs:$c)))]>,
                 Requires<[HasV5T]>;
}

multiclass FCMP32_rr<string OpcStr, PatFrag OpNode> {
  def _rr : ALU64_rr<(outs PredRegs:$dst), (ins IntRegs:$b, IntRegs:$c),
                 !strconcat("$dst = ", !strconcat(OpcStr, "($b, $c)")),
                 [(set PredRegs:$dst,
                        (OpNode (f32 IntRegs:$b), (f32 IntRegs:$c)))]>,
                 Requires<[HasV5T]>;
}
}

defm FCMPOEQ64 : FCMP64_rr<"dfcmp.eq", setoeq>;
defm FCMPUEQ64 : FCMP64_rr<"dfcmp.eq", setueq>;
defm FCMPOGT64 : FCMP64_rr<"dfcmp.gt", setogt>;
defm FCMPUGT64 : FCMP64_rr<"dfcmp.gt", setugt>;
defm FCMPOGE64 : FCMP64_rr<"dfcmp.ge", setoge>;
defm FCMPUGE64 : FCMP64_rr<"dfcmp.ge", setuge>;

defm FCMPOEQ32 : FCMP32_rr<"sfcmp.eq", setoeq>;
defm FCMPUEQ32 : FCMP32_rr<"sfcmp.eq", setueq>;
defm FCMPOGT32 : FCMP32_rr<"sfcmp.gt", setogt>;
defm FCMPUGT32 : FCMP32_rr<"sfcmp.gt", setugt>;
defm FCMPOGE32 : FCMP32_rr<"sfcmp.ge", setoge>;
defm FCMPUGE32 : FCMP32_rr<"sfcmp.ge", setuge>;

// olt.
def : Pat <(i1 (setolt (f32 IntRegs:$src1), (f32 IntRegs:$src2))),
      (i1 (FCMPOGT32_rr IntRegs:$src2, IntRegs:$src1))>,
      Requires<[HasV5T]>;

def : Pat <(i1 (setolt (f32 IntRegs:$src1), (fpimm:$src2))),
      (i1 (FCMPOGT32_rr (f32 (TFRI_f fpimm:$src2)), (f32 IntRegs:$src1)))>,
      Requires<[HasV5T]>;

def : Pat <(i1 (setolt (f64 DoubleRegs:$src1), (f64 DoubleRegs:$src2))),
      (i1 (FCMPOGT64_rr DoubleRegs:$src2, DoubleRegs:$src1))>,
      Requires<[HasV5T]>;

def : Pat <(i1 (setolt (f64 DoubleRegs:$src1), (fpimm:$src2))),
      (i1 (FCMPOGT64_rr (f64 (CONST64_Float_Real fpimm:$src2)),
                        (f64 DoubleRegs:$src1)))>,
      Requires<[HasV5T]>;

// gt.
def : Pat <(i1 (setugt (f64 DoubleRegs:$src1), (fpimm:$src2))),
      (i1 (FCMPUGT64_rr (f64 DoubleRegs:$src1),
                        (f64 (CONST64_Float_Real fpimm:$src2))))>,
      Requires<[HasV5T]>;

def : Pat <(i1 (setugt (f32 IntRegs:$src1), (fpimm:$src2))),
      (i1 (FCMPUGT32_rr (f32 IntRegs:$src1), (f32 (TFRI_f fpimm:$src2))))>,
      Requires<[HasV5T]>;

// ult.
def : Pat <(i1 (setult (f32 IntRegs:$src1), (f32 IntRegs:$src2))),
      (i1 (FCMPUGT32_rr IntRegs:$src2, IntRegs:$src1))>,
      Requires<[HasV5T]>;

def : Pat <(i1 (setult (f32 IntRegs:$src1), (fpimm:$src2))),
      (i1 (FCMPUGT32_rr (f32 (TFRI_f fpimm:$src2)), (f32 IntRegs:$src1)))>,
      Requires<[HasV5T]>;

def : Pat <(i1 (setult (f64 DoubleRegs:$src1), (f64 DoubleRegs:$src2))),
      (i1 (FCMPUGT64_rr DoubleRegs:$src2, DoubleRegs:$src1))>,
      Requires<[HasV5T]>;

def : Pat <(i1 (setult (f64 DoubleRegs:$src1), (fpimm:$src2))),
      (i1 (FCMPUGT64_rr (f64 (CONST64_Float_Real fpimm:$src2)),
                        (f64 DoubleRegs:$src1)))>,
      Requires<[HasV5T]>;

// le.
// rs <= rt -> rt >= rs.
def : Pat<(i1 (setole (f32 IntRegs:$src1), (f32 IntRegs:$src2))),
      (i1 (FCMPOGE32_rr IntRegs:$src2, IntRegs:$src1))>,
      Requires<[HasV5T]>;

def : Pat<(i1 (setole (f32 IntRegs:$src1), (fpimm:$src2))),
      (i1 (FCMPOGE32_rr (f32 (TFRI_f fpimm:$src2)), IntRegs:$src1))>,
      Requires<[HasV5T]>;


// Rss <= Rtt -> Rtt >= Rss.
def : Pat<(i1 (setole (f64 DoubleRegs:$src1), (f64 DoubleRegs:$src2))),
      (i1 (FCMPOGE64_rr DoubleRegs:$src2, DoubleRegs:$src1))>,
      Requires<[HasV5T]>;

def : Pat<(i1 (setole (f64 DoubleRegs:$src1), (fpimm:$src2))),
      (i1 (FCMPOGE64_rr (f64 (CONST64_Float_Real fpimm:$src2)),
                                DoubleRegs:$src1))>,
      Requires<[HasV5T]>;

// rs <= rt -> rt >= rs.
def : Pat<(i1 (setule (f32 IntRegs:$src1), (f32 IntRegs:$src2))),
      (i1 (FCMPUGE32_rr IntRegs:$src2, IntRegs:$src1))>,
      Requires<[HasV5T]>;

def : Pat<(i1 (setule (f32 IntRegs:$src1), (fpimm:$src2))),
      (i1 (FCMPUGE32_rr (f32 (TFRI_f fpimm:$src2)), IntRegs:$src1))>,
      Requires<[HasV5T]>;

// Rss <= Rtt -> Rtt >= Rss.
def : Pat<(i1 (setule (f64 DoubleRegs:$src1), (f64 DoubleRegs:$src2))),
      (i1 (FCMPUGE64_rr DoubleRegs:$src2, DoubleRegs:$src1))>,
      Requires<[HasV5T]>;

def : Pat<(i1 (setule (f64 DoubleRegs:$src1), (fpimm:$src2))),
      (i1 (FCMPUGE64_rr (f64 (CONST64_Float_Real fpimm:$src2)),
                                DoubleRegs:$src1))>,
      Requires<[HasV5T]>;

// ne.
def : Pat<(i1 (setone (f32 IntRegs:$src1), (f32 IntRegs:$src2))),
      (i1 (NOT_p (FCMPOEQ32_rr IntRegs:$src1, IntRegs:$src2)))>,
      Requires<[HasV5T]>;

def : Pat<(i1 (setone (f64 DoubleRegs:$src1), (f64 DoubleRegs:$src2))),
      (i1 (NOT_p (FCMPOEQ64_rr DoubleRegs:$src1, DoubleRegs:$src2)))>,
      Requires<[HasV5T]>;

def : Pat<(i1 (setune (f32 IntRegs:$src1), (f32 IntRegs:$src2))),
      (i1 (NOT_p (FCMPUEQ32_rr IntRegs:$src1, IntRegs:$src2)))>,
      Requires<[HasV5T]>;

def : Pat<(i1 (setune (f64 DoubleRegs:$src1), (f64 DoubleRegs:$src2))),
      (i1 (NOT_p (FCMPUEQ64_rr DoubleRegs:$src1, DoubleRegs:$src2)))>,
      Requires<[HasV5T]>;

def : Pat<(i1 (setone (f32 IntRegs:$src1), (fpimm:$src2))),
      (i1 (NOT_p (FCMPOEQ32_rr IntRegs:$src1, (f32 (TFRI_f fpimm:$src2)))))>,
      Requires<[HasV5T]>;

def : Pat<(i1 (setone (f64 DoubleRegs:$src1), (fpimm:$src2))),
      (i1 (NOT_p (FCMPOEQ64_rr DoubleRegs:$src1,
                              (f64 (CONST64_Float_Real fpimm:$src2)))))>,
      Requires<[HasV5T]>;

def : Pat<(i1 (setune (f32 IntRegs:$src1), (fpimm:$src2))),
      (i1 (NOT_p (FCMPUEQ32_rr IntRegs:$src1,  (f32 (TFRI_f fpimm:$src2)))))>,
      Requires<[HasV5T]>;

def : Pat<(i1 (setune (f64 DoubleRegs:$src1), (fpimm:$src2))),
      (i1 (NOT_p (FCMPUEQ64_rr DoubleRegs:$src1,
                              (f64 (CONST64_Float_Real fpimm:$src2)))))>,
      Requires<[HasV5T]>;

// Convert Integer to Floating Point.
def CONVERT_d2sf : ALU64_rr<(outs IntRegs:$dst), (ins DoubleRegs:$src),
              "$dst = convert_d2sf($src)",
              [(set (f32 IntRegs:$dst), (sint_to_fp (i64 DoubleRegs:$src)))]>,
              Requires<[HasV5T]>;

def CONVERT_ud2sf : ALU64_rr<(outs IntRegs:$dst), (ins DoubleRegs:$src),
              "$dst = convert_ud2sf($src)",
              [(set (f32 IntRegs:$dst), (uint_to_fp (i64 DoubleRegs:$src)))]>,
              Requires<[HasV5T]>;

def CONVERT_uw2sf : ALU64_rr<(outs IntRegs:$dst), (ins IntRegs:$src),
              "$dst = convert_uw2sf($src)",
              [(set (f32 IntRegs:$dst), (uint_to_fp (i32 IntRegs:$src)))]>,
              Requires<[HasV5T]>;

def CONVERT_w2sf : ALU64_rr<(outs IntRegs:$dst), (ins IntRegs:$src),
              "$dst = convert_w2sf($src)",
              [(set (f32 IntRegs:$dst), (sint_to_fp (i32 IntRegs:$src)))]>,
              Requires<[HasV5T]>;

def CONVERT_d2df : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src),
              "$dst = convert_d2df($src)",
              [(set (f64 DoubleRegs:$dst), (sint_to_fp (i64 DoubleRegs:$src)))]>,
              Requires<[HasV5T]>;

def CONVERT_ud2df : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src),
              "$dst = convert_ud2df($src)",
              [(set (f64 DoubleRegs:$dst), (uint_to_fp (i64 DoubleRegs:$src)))]>,
              Requires<[HasV5T]>;

def CONVERT_uw2df : ALU64_rr<(outs DoubleRegs:$dst), (ins IntRegs:$src),
              "$dst = convert_uw2df($src)",
              [(set (f64 DoubleRegs:$dst), (uint_to_fp (i32 IntRegs:$src)))]>,
              Requires<[HasV5T]>;

def CONVERT_w2df : ALU64_rr<(outs DoubleRegs:$dst), (ins IntRegs:$src),
              "$dst = convert_w2df($src)",
              [(set (f64 DoubleRegs:$dst), (sint_to_fp (i32 IntRegs:$src)))]>,
              Requires<[HasV5T]>;

// Convert Floating Point to Integer - default.
def CONVERT_df2uw : ALU64_rr<(outs IntRegs:$dst), (ins DoubleRegs:$src),
              "$dst = convert_df2uw($src):chop",
              [(set (i32 IntRegs:$dst), (fp_to_uint (f64 DoubleRegs:$src)))]>,
              Requires<[HasV5T]>;

def CONVERT_df2w : ALU64_rr<(outs IntRegs:$dst), (ins DoubleRegs:$src),
              "$dst = convert_df2w($src):chop",
              [(set (i32 IntRegs:$dst), (fp_to_sint (f64 DoubleRegs:$src)))]>,
              Requires<[HasV5T]>;

def CONVERT_sf2uw : ALU64_rr<(outs IntRegs:$dst), (ins IntRegs:$src),
              "$dst = convert_sf2uw($src):chop",
              [(set (i32 IntRegs:$dst), (fp_to_uint (f32 IntRegs:$src)))]>,
              Requires<[HasV5T]>;

def CONVERT_sf2w : ALU64_rr<(outs IntRegs:$dst), (ins IntRegs:$src),
              "$dst = convert_sf2w($src):chop",
              [(set (i32 IntRegs:$dst), (fp_to_sint (f32 IntRegs:$src)))]>,
              Requires<[HasV5T]>;

def CONVERT_df2d : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src),
              "$dst = convert_df2d($src):chop",
              [(set (i64 DoubleRegs:$dst), (fp_to_sint (f64 DoubleRegs:$src)))]>,
              Requires<[HasV5T]>;

def CONVERT_df2ud : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src),
              "$dst = convert_df2ud($src):chop",
              [(set (i64 DoubleRegs:$dst), (fp_to_uint (f64 DoubleRegs:$src)))]>,
              Requires<[HasV5T]>;

def CONVERT_sf2d : ALU64_rr<(outs DoubleRegs:$dst), (ins IntRegs:$src),
              "$dst = convert_sf2d($src):chop",
              [(set (i64 DoubleRegs:$dst), (fp_to_sint (f32 IntRegs:$src)))]>,
              Requires<[HasV5T]>;

def CONVERT_sf2ud : ALU64_rr<(outs DoubleRegs:$dst), (ins IntRegs:$src),
              "$dst = convert_sf2ud($src):chop",
              [(set (i64 DoubleRegs:$dst), (fp_to_uint (f32 IntRegs:$src)))]>,
              Requires<[HasV5T]>;

// Convert Floating Point to Integer: non-chopped.
let AddedComplexity = 20 in
def CONVERT_df2uw_nchop : ALU64_rr<(outs IntRegs:$dst), (ins DoubleRegs:$src),
              "$dst = convert_df2uw($src)",
              [(set (i32 IntRegs:$dst), (fp_to_uint (f64 DoubleRegs:$src)))]>,
              Requires<[HasV5T, IEEERndNearV5T]>;

let AddedComplexity = 20 in
def CONVERT_df2w_nchop : ALU64_rr<(outs IntRegs:$dst), (ins DoubleRegs:$src),
              "$dst = convert_df2w($src)",
              [(set (i32 IntRegs:$dst), (fp_to_sint (f64 DoubleRegs:$src)))]>,
              Requires<[HasV5T, IEEERndNearV5T]>;

let AddedComplexity = 20 in
def CONVERT_sf2uw_nchop : ALU64_rr<(outs IntRegs:$dst), (ins IntRegs:$src),
              "$dst = convert_sf2uw($src)",
              [(set (i32 IntRegs:$dst), (fp_to_uint (f32 IntRegs:$src)))]>,
              Requires<[HasV5T, IEEERndNearV5T]>;

let AddedComplexity = 20 in
def CONVERT_sf2w_nchop : ALU64_rr<(outs IntRegs:$dst), (ins IntRegs:$src),
              "$dst = convert_sf2w($src)",
              [(set (i32 IntRegs:$dst), (fp_to_sint (f32 IntRegs:$src)))]>,
              Requires<[HasV5T, IEEERndNearV5T]>;

let AddedComplexity = 20 in
def CONVERT_df2d_nchop : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src),
              "$dst = convert_df2d($src)",
              [(set (i64 DoubleRegs:$dst), (fp_to_sint (f64 DoubleRegs:$src)))]>,
              Requires<[HasV5T, IEEERndNearV5T]>;

let AddedComplexity = 20 in
def CONVERT_df2ud_nchop : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src),
              "$dst = convert_df2ud($src)",
              [(set (i64 DoubleRegs:$dst), (fp_to_uint (f64 DoubleRegs:$src)))]>,
              Requires<[HasV5T, IEEERndNearV5T]>;

let AddedComplexity = 20 in
def CONVERT_sf2d_nchop : ALU64_rr<(outs DoubleRegs:$dst), (ins IntRegs:$src),
              "$dst = convert_sf2d($src)",
              [(set (i64 DoubleRegs:$dst), (fp_to_sint (f32 IntRegs:$src)))]>,
              Requires<[HasV5T, IEEERndNearV5T]>;

let AddedComplexity = 20 in
def CONVERT_sf2ud_nchop : ALU64_rr<(outs DoubleRegs:$dst), (ins IntRegs:$src),
              "$dst = convert_sf2ud($src)",
              [(set (i64 DoubleRegs:$dst), (fp_to_uint (f32 IntRegs:$src)))]>,
              Requires<[HasV5T, IEEERndNearV5T]>;



// Bitcast is different than [fp|sint|uint]_to_[sint|uint|fp].
def : Pat <(i32 (bitconvert (f32 IntRegs:$src))),
           (i32 (TFR IntRegs:$src))>,
          Requires<[HasV5T]>;

def : Pat <(f32 (bitconvert (i32 IntRegs:$src))),
           (f32 (TFR IntRegs:$src))>,
          Requires<[HasV5T]>;

def : Pat <(i64 (bitconvert (f64 DoubleRegs:$src))),
           (i64 (TFR64 DoubleRegs:$src))>,
          Requires<[HasV5T]>;

def : Pat <(f64 (bitconvert (i64 DoubleRegs:$src))),
           (f64 (TFR64 DoubleRegs:$src))>,
          Requires<[HasV5T]>;

// Floating point fused multiply-add.
def FMADD_dp : ALU64_acc<(outs DoubleRegs:$dst),
                  (ins DoubleRegs:$src1, DoubleRegs:$src2, DoubleRegs:$src3),
              "$dst += dfmpy($src2, $src3)",
              [(set (f64 DoubleRegs:$dst),
                  (fma DoubleRegs:$src2, DoubleRegs:$src3, DoubleRegs:$src1))],
                  "$src1 = $dst">,
              Requires<[HasV5T]>;

def FMADD_sp : ALU64_acc<(outs IntRegs:$dst),
                  (ins IntRegs:$src1, IntRegs:$src2, IntRegs:$src3),
              "$dst += sfmpy($src2, $src3)",
              [(set (f32 IntRegs:$dst),
                  (fma IntRegs:$src2, IntRegs:$src3, IntRegs:$src1))],
                  "$src1 = $dst">,
              Requires<[HasV5T]>;


// Floating point max/min.
let AddedComplexity = 100 in
def FMAX_dp : ALU64_rr<(outs DoubleRegs:$dst),
                  (ins DoubleRegs:$src1, DoubleRegs:$src2),
              "$dst = dfmax($src1, $src2)",
              [(set DoubleRegs:$dst, (f64 (select (i1 (setolt DoubleRegs:$src2,
                                                        DoubleRegs:$src1)),
                                             DoubleRegs:$src1,
                                             DoubleRegs:$src2)))]>,
               Requires<[HasV5T]>;

let AddedComplexity = 100 in
def FMAX_sp : ALU64_rr<(outs IntRegs:$dst),
                  (ins IntRegs:$src1, IntRegs:$src2),
              "$dst = sfmax($src1, $src2)",
              [(set IntRegs:$dst, (f32 (select (i1 (setolt IntRegs:$src2,
                                                        IntRegs:$src1)),
                                             IntRegs:$src1,
                                             IntRegs:$src2)))]>,
               Requires<[HasV5T]>;

let AddedComplexity = 100 in
def FMIN_dp : ALU64_rr<(outs DoubleRegs:$dst),
                  (ins DoubleRegs:$src1, DoubleRegs:$src2),
              "$dst = dfmin($src1, $src2)",
              [(set DoubleRegs:$dst, (f64 (select (i1 (setogt DoubleRegs:$src2,
                                                        DoubleRegs:$src1)),
                                             DoubleRegs:$src1,
                                             DoubleRegs:$src2)))]>,
               Requires<[HasV5T]>;

let AddedComplexity = 100 in
def FMIN_sp : ALU64_rr<(outs IntRegs:$dst),
                  (ins IntRegs:$src1, IntRegs:$src2),
              "$dst = sfmin($src1, $src2)",
              [(set IntRegs:$dst, (f32 (select (i1 (setogt IntRegs:$src2,
                                                        IntRegs:$src1)),
                                             IntRegs:$src1,
                                             IntRegs:$src2)))]>,
               Requires<[HasV5T]>;

// Pseudo instruction to encode a set of conditional transfers.
// This instruction is used instead of a mux and trades-off codesize
// for performance. We conduct this transformation optimistically in
// the hope that these instructions get promoted to dot-new transfers.
let AddedComplexity = 100, isPredicated = 1 in
def TFR_condset_rr_f : ALU32_rr<(outs IntRegs:$dst), (ins PredRegs:$src1,
                                                        IntRegs:$src2,
                                                        IntRegs:$src3),
                     "Error; should not emit",
                     [(set IntRegs:$dst, (f32 (select PredRegs:$src1,
                                                 IntRegs:$src2,
                                                 IntRegs:$src3)))]>,
               Requires<[HasV5T]>;

let AddedComplexity = 100, isPredicated = 1 in
def TFR_condset_rr64_f : ALU32_rr<(outs DoubleRegs:$dst), (ins PredRegs:$src1,
                                                        DoubleRegs:$src2,
                                                        DoubleRegs:$src3),
                     "Error; should not emit",
                     [(set DoubleRegs:$dst, (f64 (select PredRegs:$src1,
                                                 DoubleRegs:$src2,
                                                 DoubleRegs:$src3)))]>,
               Requires<[HasV5T]>;



let AddedComplexity = 100, isPredicated = 1 in
def TFR_condset_ri_f : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, f32imm:$src3),
            "Error; should not emit",
            [(set IntRegs:$dst,
             (f32 (select PredRegs:$src1, IntRegs:$src2, fpimm:$src3)))]>,
               Requires<[HasV5T]>;

let AddedComplexity = 100, isPredicated = 1 in
def TFR_condset_ir_f : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, f32imm:$src2, IntRegs:$src3),
            "Error; should not emit",
            [(set IntRegs:$dst,
             (f32 (select PredRegs:$src1, fpimm:$src2, IntRegs:$src3)))]>,
               Requires<[HasV5T]>;

let AddedComplexity = 100, isPredicated = 1 in
def TFR_condset_ii_f : ALU32_rr<(outs IntRegs:$dst),
                              (ins PredRegs:$src1, f32imm:$src2, f32imm:$src3),
                     "Error; should not emit",
                     [(set IntRegs:$dst, (f32 (select PredRegs:$src1,
                                                 fpimm:$src2,
                                                 fpimm:$src3)))]>,
               Requires<[HasV5T]>;


def : Pat <(select (i1 (setult (f32 IntRegs:$src1), (f32 IntRegs:$src2))),
                   (f32 IntRegs:$src3),
                   (f32 IntRegs:$src4)),
    (TFR_condset_rr_f (FCMPUGT32_rr IntRegs:$src2, IntRegs:$src1), IntRegs:$src4,
                      IntRegs:$src3)>, Requires<[HasV5T]>;

def : Pat <(select (i1 (setult (f64 DoubleRegs:$src1), (f64 DoubleRegs:$src2))),
                   (f64 DoubleRegs:$src3),
                   (f64 DoubleRegs:$src4)),
      (TFR_condset_rr64_f (FCMPUGT64_rr DoubleRegs:$src2, DoubleRegs:$src1),
                DoubleRegs:$src4, DoubleRegs:$src3)>, Requires<[HasV5T]>;

// Map from p0 = pnot(p0); r0 = mux(p0, #i, #j) => r0 = mux(p0, #j, #i).
def : Pat <(select (not PredRegs:$src1), fpimm:$src2, fpimm:$src3),
      (TFR_condset_ii_f PredRegs:$src1, fpimm:$src3, fpimm:$src2)>;

// Map from p0 = pnot(p0); r0 = select(p0, #i, r1)
// => r0 = TFR_condset_ri(p0, r1, #i)
def : Pat <(select (not PredRegs:$src1), fpimm:$src2, IntRegs:$src3),
      (TFR_condset_ri_f PredRegs:$src1, IntRegs:$src3, fpimm:$src2)>;

// Map from p0 = pnot(p0); r0 = mux(p0, r1, #i)
// => r0 = TFR_condset_ir(p0, #i, r1)
def : Pat <(select (not PredRegs:$src1), IntRegs:$src2, fpimm:$src3),
      (TFR_condset_ir_f PredRegs:$src1, fpimm:$src3, IntRegs:$src2)>;

def : Pat <(i32 (fp_to_sint (f64 DoubleRegs:$src1))),
          (i32 (EXTRACT_SUBREG (i64 (CONVERT_df2d (f64 DoubleRegs:$src1))), subreg_loreg))>,
          Requires<[HasV5T]>;

def : Pat <(fabs (f32 IntRegs:$src1)),
           (CLRBIT_31 (f32 IntRegs:$src1), 31)>,
          Requires<[HasV5T]>;

def : Pat <(fneg (f32 IntRegs:$src1)),
           (TOGBIT_31 (f32 IntRegs:$src1), 31)>,
          Requires<[HasV5T]>;

/*
def : Pat <(fabs (f64 DoubleRegs:$src1)),
          (CLRBIT_31 (f32 (EXTRACT_SUBREG DoubleRegs:$src1, subreg_hireg)), 31)>,
          Requires<[HasV5T]>;

def : Pat <(fabs (f64 DoubleRegs:$src1)),
          (CLRBIT_31 (f32 (EXTRACT_SUBREG DoubleRegs:$src1, subreg_hireg)), 31)>,
          Requires<[HasV5T]>;
          */
