//==- SystemZInstrFP.td - Floating-point SystemZ instructions --*- tblgen-*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Select instructions
//===----------------------------------------------------------------------===//

// C's ?: operator for floating-point operands.
def SelectF32  : SelectWrapper<FP32>;
def SelectF64  : SelectWrapper<FP64>;
def SelectF128 : SelectWrapper<FP128>;

defm CondStoreF32 : CondStores<FP32, nonvolatile_store,
                               nonvolatile_load, bdxaddr20only>;
defm CondStoreF64 : CondStores<FP64, nonvolatile_store,
                               nonvolatile_load, bdxaddr20only>;

//===----------------------------------------------------------------------===//
// Move instructions
//===----------------------------------------------------------------------===//

// Load zero.
let hasSideEffects = 0, isAsCheapAsAMove = 1, isMoveImm = 1 in {
  def LZER : InherentRRE<"lzer", 0xB374, FP32,  (fpimm0)>;
  def LZDR : InherentRRE<"lzdr", 0xB375, FP64,  (fpimm0)>;
  def LZXR : InherentRRE<"lzxr", 0xB376, FP128, (fpimm0)>;
}

// Moves between two floating-point registers.
let hasSideEffects = 0 in {
  def LER : UnaryRR <"le", 0x38,   null_frag, FP32,  FP32>;
  def LDR : UnaryRR <"ld", 0x28,   null_frag, FP64,  FP64>;
  def LXR : UnaryRRE<"lx", 0xB365, null_frag, FP128, FP128>;

  // For z13 we prefer LDR over LER to avoid partial register dependencies.
  let isCodeGenOnly = 1 in
    def LDR32 : UnaryRR<"ld", 0x28, null_frag, FP32, FP32>;
}

// Moves between two floating-point registers that also set the condition
// codes.
let Defs = [CC], CCValues = 0xF, CompareZeroCCMask = 0xF in {
  defm LTEBR : LoadAndTestRRE<"lteb", 0xB302, FP32>;
  defm LTDBR : LoadAndTestRRE<"ltdb", 0xB312, FP64>;
  defm LTXBR : LoadAndTestRRE<"ltxb", 0xB342, FP128>;
}
// Note that LTxBRCompare is not available if we have vector support,
// since load-and-test instructions will partially clobber the target
// (vector) register.
let Predicates = [FeatureNoVector] in {
  defm : CompareZeroFP<LTEBRCompare, FP32>;
  defm : CompareZeroFP<LTDBRCompare, FP64>;
  defm : CompareZeroFP<LTXBRCompare, FP128>;
}

// Use a normal load-and-test for compare against zero in case of
// vector support (via a pseudo to simplify instruction selection).
let Defs = [CC], usesCustomInserter = 1 in {
  def LTEBRCompare_VecPseudo : Pseudo<(outs), (ins FP32:$R1, FP32:$R2), []>;
  def LTDBRCompare_VecPseudo : Pseudo<(outs), (ins FP64:$R1, FP64:$R2), []>;
  def LTXBRCompare_VecPseudo : Pseudo<(outs), (ins FP128:$R1, FP128:$R2), []>;
}
let Predicates = [FeatureVector] in {
  defm : CompareZeroFP<LTEBRCompare_VecPseudo, FP32>;
  defm : CompareZeroFP<LTDBRCompare_VecPseudo, FP64>;
  defm : CompareZeroFP<LTXBRCompare_VecPseudo, FP128>;
}

// Moves between 64-bit integer and floating-point registers.
def LGDR : UnaryRRE<"lgd", 0xB3CD, bitconvert, GR64, FP64>;
def LDGR : UnaryRRE<"ldg", 0xB3C1, bitconvert, FP64, GR64>;

// fcopysign with an FP32 result.
let isCodeGenOnly = 1 in {
  def CPSDRss : BinaryRRF<"cpsd", 0xB372, fcopysign, FP32, FP32>;
  def CPSDRsd : BinaryRRF<"cpsd", 0xB372, fcopysign, FP32, FP64>;
}

// The sign of an FP128 is in the high register.
def : Pat<(fcopysign FP32:$src1, FP128:$src2),
          (CPSDRsd FP32:$src1, (EXTRACT_SUBREG FP128:$src2, subreg_h64))>;

// fcopysign with an FP64 result.
let isCodeGenOnly = 1 in
  def CPSDRds : BinaryRRF<"cpsd", 0xB372, fcopysign, FP64, FP32>;
def CPSDRdd : BinaryRRF<"cpsd", 0xB372, fcopysign, FP64, FP64>;

// The sign of an FP128 is in the high register.
def : Pat<(fcopysign FP64:$src1, FP128:$src2),
          (CPSDRdd FP64:$src1, (EXTRACT_SUBREG FP128:$src2, subreg_h64))>;

// fcopysign with an FP128 result.  Use "upper" as the high half and leave
// the low half as-is.
class CopySign128<RegisterOperand cls, dag upper>
  : Pat<(fcopysign FP128:$src1, cls:$src2),
        (INSERT_SUBREG FP128:$src1, upper, subreg_h64)>;

def : CopySign128<FP32,  (CPSDRds (EXTRACT_SUBREG FP128:$src1, subreg_h64),
                                  FP32:$src2)>;
def : CopySign128<FP64,  (CPSDRdd (EXTRACT_SUBREG FP128:$src1, subreg_h64),
                                  FP64:$src2)>;
def : CopySign128<FP128, (CPSDRdd (EXTRACT_SUBREG FP128:$src1, subreg_h64),
                                  (EXTRACT_SUBREG FP128:$src2, subreg_h64))>;

defm LoadStoreF32  : MVCLoadStore<load, f32,  MVCSequence, 4>;
defm LoadStoreF64  : MVCLoadStore<load, f64,  MVCSequence, 8>;
defm LoadStoreF128 : MVCLoadStore<load, f128, MVCSequence, 16>;

//===----------------------------------------------------------------------===//
// Load instructions
//===----------------------------------------------------------------------===//

let canFoldAsLoad = 1, SimpleBDXLoad = 1 in {
  defm LE : UnaryRXPair<"le", 0x78, 0xED64, load, FP32, 4>;
  defm LD : UnaryRXPair<"ld", 0x68, 0xED65, load, FP64, 8>;

  // For z13 we prefer LDE over LE to avoid partial register dependencies.
  def LDE32 : UnaryRXE<"lde", 0xED24, null_frag, FP32, 4>;

  // These instructions are split after register allocation, so we don't
  // want a custom inserter.
  let Has20BitOffset = 1, HasIndex = 1, Is128Bit = 1 in {
    def LX : Pseudo<(outs FP128:$dst), (ins bdxaddr20only128:$src),
                     [(set FP128:$dst, (load bdxaddr20only128:$src))]>;
  }
}

//===----------------------------------------------------------------------===//
// Store instructions
//===----------------------------------------------------------------------===//

let SimpleBDXStore = 1 in {
  defm STE : StoreRXPair<"ste", 0x70, 0xED66, store, FP32, 4>;
  defm STD : StoreRXPair<"std", 0x60, 0xED67, store, FP64, 8>;

  // These instructions are split after register allocation, so we don't
  // want a custom inserter.
  let Has20BitOffset = 1, HasIndex = 1, Is128Bit = 1 in {
    def STX : Pseudo<(outs), (ins FP128:$src, bdxaddr20only128:$dst),
                     [(store FP128:$src, bdxaddr20only128:$dst)]>;
  }
}

//===----------------------------------------------------------------------===//
// Conversion instructions
//===----------------------------------------------------------------------===//

// Convert floating-point values to narrower representations, rounding
// according to the current mode.  The destination of LEXBR and LDXBR
// is a 128-bit value, but only the first register of the pair is used.
def LEDBR : UnaryRRE<"ledb", 0xB344, fpround,    FP32,  FP64>;
def LEXBR : UnaryRRE<"lexb", 0xB346, null_frag, FP128, FP128>;
def LDXBR : UnaryRRE<"ldxb", 0xB345, null_frag, FP128, FP128>;

def LEDBRA : UnaryRRF4<"ledbra", 0xB344, FP32,  FP64>,
             Requires<[FeatureFPExtension]>;
def LEXBRA : UnaryRRF4<"lexbra", 0xB346, FP128, FP128>,
             Requires<[FeatureFPExtension]>;
def LDXBRA : UnaryRRF4<"ldxbra", 0xB345, FP128, FP128>,
             Requires<[FeatureFPExtension]>;

def : Pat<(f32 (fpround FP128:$src)),
          (EXTRACT_SUBREG (LEXBR FP128:$src), subreg_hr32)>;
def : Pat<(f64 (fpround FP128:$src)),
          (EXTRACT_SUBREG (LDXBR FP128:$src), subreg_h64)>;

// Extend register floating-point values to wider representations.
def LDEBR : UnaryRRE<"ldeb", 0xB304, fpextend, FP64,  FP32>;
def LXEBR : UnaryRRE<"lxeb", 0xB306, fpextend, FP128, FP32>;
def LXDBR : UnaryRRE<"lxdb", 0xB305, fpextend, FP128, FP64>;

// Extend memory floating-point values to wider representations.
def LDEB : UnaryRXE<"ldeb", 0xED04, extloadf32, FP64,  4>;
def LXEB : UnaryRXE<"lxeb", 0xED06, extloadf32, FP128, 4>;
def LXDB : UnaryRXE<"lxdb", 0xED05, extloadf64, FP128, 8>;

// Convert a signed integer register value to a floating-point one.
def CEFBR : UnaryRRE<"cefb", 0xB394, sint_to_fp, FP32,  GR32>;
def CDFBR : UnaryRRE<"cdfb", 0xB395, sint_to_fp, FP64,  GR32>;
def CXFBR : UnaryRRE<"cxfb", 0xB396, sint_to_fp, FP128, GR32>;

def CEGBR : UnaryRRE<"cegb", 0xB3A4, sint_to_fp, FP32,  GR64>;
def CDGBR : UnaryRRE<"cdgb", 0xB3A5, sint_to_fp, FP64,  GR64>;
def CXGBR : UnaryRRE<"cxgb", 0xB3A6, sint_to_fp, FP128, GR64>;

// Convert am unsigned integer register value to a floating-point one.
let Predicates = [FeatureFPExtension] in {
  def CELFBR : UnaryRRF4<"celfbr", 0xB390, FP32,  GR32>;
  def CDLFBR : UnaryRRF4<"cdlfbr", 0xB391, FP64,  GR32>;
  def CXLFBR : UnaryRRF4<"cxlfbr", 0xB392, FP128, GR32>;

  def CELGBR : UnaryRRF4<"celgbr", 0xB3A0, FP32,  GR64>;
  def CDLGBR : UnaryRRF4<"cdlgbr", 0xB3A1, FP64,  GR64>;
  def CXLGBR : UnaryRRF4<"cxlgbr", 0xB3A2, FP128, GR64>;

  def : Pat<(f32  (uint_to_fp GR32:$src)), (CELFBR 0, GR32:$src, 0)>;
  def : Pat<(f64  (uint_to_fp GR32:$src)), (CDLFBR 0, GR32:$src, 0)>;
  def : Pat<(f128 (uint_to_fp GR32:$src)), (CXLFBR 0, GR32:$src, 0)>;

  def : Pat<(f32  (uint_to_fp GR64:$src)), (CELGBR 0, GR64:$src, 0)>;
  def : Pat<(f64  (uint_to_fp GR64:$src)), (CDLGBR 0, GR64:$src, 0)>;
  def : Pat<(f128 (uint_to_fp GR64:$src)), (CXLGBR 0, GR64:$src, 0)>;
}

// Convert a floating-point register value to a signed integer value,
// with the second operand (modifier M3) specifying the rounding mode.
let Defs = [CC] in {
  def CFEBR : UnaryRRF<"cfeb", 0xB398, GR32, FP32>;
  def CFDBR : UnaryRRF<"cfdb", 0xB399, GR32, FP64>;
  def CFXBR : UnaryRRF<"cfxb", 0xB39A, GR32, FP128>;

  def CGEBR : UnaryRRF<"cgeb", 0xB3A8, GR64, FP32>;
  def CGDBR : UnaryRRF<"cgdb", 0xB3A9, GR64, FP64>;
  def CGXBR : UnaryRRF<"cgxb", 0xB3AA, GR64, FP128>;
}

// fp_to_sint always rounds towards zero, which is modifier value 5.
def : Pat<(i32 (fp_to_sint FP32:$src)),  (CFEBR 5, FP32:$src)>;
def : Pat<(i32 (fp_to_sint FP64:$src)),  (CFDBR 5, FP64:$src)>;
def : Pat<(i32 (fp_to_sint FP128:$src)), (CFXBR 5, FP128:$src)>;

def : Pat<(i64 (fp_to_sint FP32:$src)),  (CGEBR 5, FP32:$src)>;
def : Pat<(i64 (fp_to_sint FP64:$src)),  (CGDBR 5, FP64:$src)>;
def : Pat<(i64 (fp_to_sint FP128:$src)), (CGXBR 5, FP128:$src)>;

// Convert a floating-point register value to an unsigned integer value.
let Predicates = [FeatureFPExtension] in {
  let Defs = [CC] in {
    def CLFEBR : UnaryRRF4<"clfebr", 0xB39C, GR32, FP32>;
    def CLFDBR : UnaryRRF4<"clfdbr", 0xB39D, GR32, FP64>;
    def CLFXBR : UnaryRRF4<"clfxbr", 0xB39E, GR32, FP128>;

    def CLGEBR : UnaryRRF4<"clgebr", 0xB3AC, GR64, FP32>;
    def CLGDBR : UnaryRRF4<"clgdbr", 0xB3AD, GR64, FP64>;
    def CLGXBR : UnaryRRF4<"clgxbr", 0xB3AE, GR64, FP128>;
  }

  def : Pat<(i32 (fp_to_uint FP32:$src)),  (CLFEBR 5, FP32:$src,  0)>;
  def : Pat<(i32 (fp_to_uint FP64:$src)),  (CLFDBR 5, FP64:$src,  0)>;
  def : Pat<(i32 (fp_to_uint FP128:$src)), (CLFXBR 5, FP128:$src, 0)>;

  def : Pat<(i64 (fp_to_uint FP32:$src)),  (CLGEBR 5, FP32:$src,  0)>;
  def : Pat<(i64 (fp_to_uint FP64:$src)),  (CLGDBR 5, FP64:$src,  0)>;
  def : Pat<(i64 (fp_to_uint FP128:$src)), (CLGXBR 5, FP128:$src, 0)>;
}


//===----------------------------------------------------------------------===//
// Unary arithmetic
//===----------------------------------------------------------------------===//

// We prefer generic instructions during isel, because they do not
// clobber CC and therefore give the scheduler more freedom. In cases
// the CC is actually useful, the SystemZElimCompare pass will try to
// convert generic instructions into opcodes that also set CC. Note
// that lcdf / lpdf / lndf only affect the sign bit, and can therefore
// be used with fp32 as well. This could be done for fp128, in which
// case the operands would have to be tied.

// Negation (Load Complement).
let Defs = [CC], CCValues = 0xF, CompareZeroCCMask = 0xF in {
  def LCEBR : UnaryRRE<"lceb", 0xB303, null_frag, FP32,  FP32>;
  def LCDBR : UnaryRRE<"lcdb", 0xB313, null_frag, FP64,  FP64>;
  def LCXBR : UnaryRRE<"lcxb", 0xB343, fneg, FP128, FP128>;
}
// Generic form, which does not set CC.
def LCDFR : UnaryRRE<"lcdf", 0xB373, fneg, FP64,  FP64>;
let isCodeGenOnly = 1 in
  def LCDFR_32 : UnaryRRE<"lcdf", 0xB373, fneg, FP32,  FP32>;

// Absolute value (Load Positive).
let Defs = [CC], CCValues = 0xF, CompareZeroCCMask = 0xF in {
  def LPEBR : UnaryRRE<"lpeb", 0xB300, null_frag, FP32,  FP32>;
  def LPDBR : UnaryRRE<"lpdb", 0xB310, null_frag, FP64,  FP64>;
  def LPXBR : UnaryRRE<"lpxb", 0xB340, fabs, FP128, FP128>;
}
// Generic form, which does not set CC.
def LPDFR : UnaryRRE<"lpdf", 0xB370, fabs, FP64,  FP64>;
let isCodeGenOnly = 1 in
  def LPDFR_32 : UnaryRRE<"lpdf", 0xB370, fabs, FP32,  FP32>;

// Negative absolute value (Load Negative).
let Defs = [CC], CCValues = 0xF, CompareZeroCCMask = 0xF in {
  def LNEBR : UnaryRRE<"lneb", 0xB301, null_frag, FP32,  FP32>;
  def LNDBR : UnaryRRE<"lndb", 0xB311, null_frag, FP64,  FP64>;
  def LNXBR : UnaryRRE<"lnxb", 0xB341, fnabs, FP128, FP128>;
}
// Generic form, which does not set CC.
def LNDFR : UnaryRRE<"lndf", 0xB371, fnabs, FP64,  FP64>;
let isCodeGenOnly = 1 in
  def LNDFR_32 : UnaryRRE<"lndf", 0xB371, fnabs, FP32,  FP32>;

// Square root.
def SQEBR : UnaryRRE<"sqeb", 0xB314, fsqrt, FP32,  FP32>;
def SQDBR : UnaryRRE<"sqdb", 0xB315, fsqrt, FP64,  FP64>;
def SQXBR : UnaryRRE<"sqxb", 0xB316, fsqrt, FP128, FP128>;

def SQEB : UnaryRXE<"sqeb", 0xED14, loadu<fsqrt>, FP32, 4>;
def SQDB : UnaryRXE<"sqdb", 0xED15, loadu<fsqrt>, FP64, 8>;

// Round to an integer, with the second operand (modifier M3) specifying
// the rounding mode.  These forms always check for inexact conditions.
def FIEBR : UnaryRRF<"fieb", 0xB357, FP32,  FP32>;
def FIDBR : UnaryRRF<"fidb", 0xB35F, FP64,  FP64>;
def FIXBR : UnaryRRF<"fixb", 0xB347, FP128, FP128>;

// frint rounds according to the current mode (modifier 0) and detects
// inexact conditions.
def : Pat<(frint FP32:$src),  (FIEBR 0, FP32:$src)>;
def : Pat<(frint FP64:$src),  (FIDBR 0, FP64:$src)>;
def : Pat<(frint FP128:$src), (FIXBR 0, FP128:$src)>;

let Predicates = [FeatureFPExtension] in {
  // Extended forms of the FIxBR instructions.  M4 can be set to 4
  // to suppress detection of inexact conditions.
  def FIEBRA : UnaryRRF4<"fiebra", 0xB357, FP32,  FP32>;
  def FIDBRA : UnaryRRF4<"fidbra", 0xB35F, FP64,  FP64>;
  def FIXBRA : UnaryRRF4<"fixbra", 0xB347, FP128, FP128>;

  // fnearbyint is like frint but does not detect inexact conditions.
  def : Pat<(fnearbyint FP32:$src),  (FIEBRA 0, FP32:$src,  4)>;
  def : Pat<(fnearbyint FP64:$src),  (FIDBRA 0, FP64:$src,  4)>;
  def : Pat<(fnearbyint FP128:$src), (FIXBRA 0, FP128:$src, 4)>;

  // floor is no longer allowed to raise an inexact condition,
  // so restrict it to the cases where the condition can be suppressed.
  // Mode 7 is round towards -inf.
  def : Pat<(ffloor FP32:$src),  (FIEBRA 7, FP32:$src,  4)>;
  def : Pat<(ffloor FP64:$src),  (FIDBRA 7, FP64:$src,  4)>;
  def : Pat<(ffloor FP128:$src), (FIXBRA 7, FP128:$src, 4)>;

  // Same idea for ceil, where mode 6 is round towards +inf.
  def : Pat<(fceil FP32:$src),  (FIEBRA 6, FP32:$src,  4)>;
  def : Pat<(fceil FP64:$src),  (FIDBRA 6, FP64:$src,  4)>;
  def : Pat<(fceil FP128:$src), (FIXBRA 6, FP128:$src, 4)>;

  // Same idea for trunc, where mode 5 is round towards zero.
  def : Pat<(ftrunc FP32:$src),  (FIEBRA 5, FP32:$src,  4)>;
  def : Pat<(ftrunc FP64:$src),  (FIDBRA 5, FP64:$src,  4)>;
  def : Pat<(ftrunc FP128:$src), (FIXBRA 5, FP128:$src, 4)>;

  // Same idea for round, where mode 1 is round towards nearest with
  // ties away from zero.
  def : Pat<(fround FP32:$src),  (FIEBRA 1, FP32:$src,  4)>;
  def : Pat<(fround FP64:$src),  (FIDBRA 1, FP64:$src,  4)>;
  def : Pat<(fround FP128:$src), (FIXBRA 1, FP128:$src, 4)>;
}

//===----------------------------------------------------------------------===//
// Binary arithmetic
//===----------------------------------------------------------------------===//

// Addition.
let Defs = [CC], CCValues = 0xF, CompareZeroCCMask = 0xF in {
  let isCommutable = 1 in {
    def AEBR : BinaryRRE<"aeb", 0xB30A, fadd, FP32,  FP32>;
    def ADBR : BinaryRRE<"adb", 0xB31A, fadd, FP64,  FP64>;
    def AXBR : BinaryRRE<"axb", 0xB34A, fadd, FP128, FP128>;
  }
  def AEB : BinaryRXE<"aeb", 0xED0A, fadd, FP32, load, 4>;
  def ADB : BinaryRXE<"adb", 0xED1A, fadd, FP64, load, 8>;
}

// Subtraction.
let Defs = [CC], CCValues = 0xF, CompareZeroCCMask = 0xF in {
  def SEBR : BinaryRRE<"seb", 0xB30B, fsub, FP32,  FP32>;
  def SDBR : BinaryRRE<"sdb", 0xB31B, fsub, FP64,  FP64>;
  def SXBR : BinaryRRE<"sxb", 0xB34B, fsub, FP128, FP128>;

  def SEB : BinaryRXE<"seb",  0xED0B, fsub, FP32, load, 4>;
  def SDB : BinaryRXE<"sdb",  0xED1B, fsub, FP64, load, 8>;
}

// Multiplication.
let isCommutable = 1 in {
  def MEEBR : BinaryRRE<"meeb", 0xB317, fmul, FP32,  FP32>;
  def MDBR  : BinaryRRE<"mdb",  0xB31C, fmul, FP64,  FP64>;
  def MXBR  : BinaryRRE<"mxb",  0xB34C, fmul, FP128, FP128>;
}
def MEEB : BinaryRXE<"meeb", 0xED17, fmul, FP32, load, 4>;
def MDB  : BinaryRXE<"mdb",  0xED1C, fmul, FP64, load, 8>;

// f64 multiplication of two FP32 registers.
def MDEBR : BinaryRRE<"mdeb", 0xB30C, null_frag, FP64, FP32>;
def : Pat<(fmul (f64 (fpextend FP32:$src1)), (f64 (fpextend FP32:$src2))),
          (MDEBR (INSERT_SUBREG (f64 (IMPLICIT_DEF)),
                                FP32:$src1, subreg_r32), FP32:$src2)>;

// f64 multiplication of an FP32 register and an f32 memory.
def MDEB : BinaryRXE<"mdeb", 0xED0C, null_frag, FP64, load, 4>;
def : Pat<(fmul (f64 (fpextend FP32:$src1)),
                (f64 (extloadf32 bdxaddr12only:$addr))),
          (MDEB (INSERT_SUBREG (f64 (IMPLICIT_DEF)), FP32:$src1, subreg_r32),
                bdxaddr12only:$addr)>;

// f128 multiplication of two FP64 registers.
def MXDBR : BinaryRRE<"mxdb", 0xB307, null_frag, FP128, FP64>;
def : Pat<(fmul (f128 (fpextend FP64:$src1)), (f128 (fpextend FP64:$src2))),
          (MXDBR (INSERT_SUBREG (f128 (IMPLICIT_DEF)),
                                FP64:$src1, subreg_h64), FP64:$src2)>;

// f128 multiplication of an FP64 register and an f64 memory.
def MXDB : BinaryRXE<"mxdb", 0xED07, null_frag, FP128, load, 8>;
def : Pat<(fmul (f128 (fpextend FP64:$src1)),
                (f128 (extloadf64 bdxaddr12only:$addr))),
          (MXDB (INSERT_SUBREG (f128 (IMPLICIT_DEF)), FP64:$src1, subreg_h64),
                bdxaddr12only:$addr)>;

// Fused multiply-add.
def MAEBR : TernaryRRD<"maeb", 0xB30E, z_fma, FP32>;
def MADBR : TernaryRRD<"madb", 0xB31E, z_fma, FP64>;

def MAEB : TernaryRXF<"maeb", 0xED0E, z_fma, FP32, load, 4>;
def MADB : TernaryRXF<"madb", 0xED1E, z_fma, FP64, load, 8>;

// Fused multiply-subtract.
def MSEBR : TernaryRRD<"mseb", 0xB30F, z_fms, FP32>;
def MSDBR : TernaryRRD<"msdb", 0xB31F, z_fms, FP64>;

def MSEB : TernaryRXF<"mseb", 0xED0F, z_fms, FP32, load, 4>;
def MSDB : TernaryRXF<"msdb", 0xED1F, z_fms, FP64, load, 8>;

// Division.
def DEBR : BinaryRRE<"deb", 0xB30D, fdiv, FP32,  FP32>;
def DDBR : BinaryRRE<"ddb", 0xB31D, fdiv, FP64,  FP64>;
def DXBR : BinaryRRE<"dxb", 0xB34D, fdiv, FP128, FP128>;

def DEB : BinaryRXE<"deb", 0xED0D, fdiv, FP32, load, 4>;
def DDB : BinaryRXE<"ddb", 0xED1D, fdiv, FP64, load, 8>;

//===----------------------------------------------------------------------===//
// Comparisons
//===----------------------------------------------------------------------===//

let Defs = [CC], CCValues = 0xF in {
  def CEBR : CompareRRE<"ceb", 0xB309, z_fcmp, FP32,  FP32>;
  def CDBR : CompareRRE<"cdb", 0xB319, z_fcmp, FP64,  FP64>;
  def CXBR : CompareRRE<"cxb", 0xB349, z_fcmp, FP128, FP128>;

  def CEB : CompareRXE<"ceb", 0xED09, z_fcmp, FP32, load, 4>;
  def CDB : CompareRXE<"cdb", 0xED19, z_fcmp, FP64, load, 8>;
}

// Test Data Class.
let Defs = [CC], CCValues = 0xC in {
  def TCEB : TestRXE<"tceb", 0xED10, z_tdc, FP32>;
  def TCDB : TestRXE<"tcdb", 0xED11, z_tdc, FP64>;
  def TCXB : TestRXE<"tcxb", 0xED12, z_tdc, FP128>;
}

//===----------------------------------------------------------------------===//
// Peepholes
//===----------------------------------------------------------------------===//

def : Pat<(f32  fpimmneg0), (LCDFR_32 (LZER))>;
def : Pat<(f64  fpimmneg0), (LCDFR (LZDR))>;
def : Pat<(f128 fpimmneg0), (LCXBR (LZXR))>;
