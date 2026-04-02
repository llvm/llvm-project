// Inspired from llvm/lib/Target/Mips/MipsInstrInfo.td
//      - I copied only the records I needed (not all)

//===----------------------------------------------------------------------===//
// Mips Operand, Complex Patterns and Transformations Definitions.
//===----------------------------------------------------------------------===//

class ConstantSImmAsmOperandClass<int Bits, list<AsmOperandClass> Supers = [],
                                  int Offset = 0> : AsmOperandClass {
  let Name = "ConstantSImm" # Bits # "_" # Offset;
  let RenderMethod = "addConstantSImmOperands<" # Bits # ", " # Offset # ">";
  let PredicateMethod = "isConstantSImm<" # Bits # ", " # Offset # ">";
  let SuperClasses = Supers;
  let DiagnosticType = "SImm" # Bits # "_" # Offset;
}

class ConstantUImmAsmOperandClass<int Bits, list<AsmOperandClass> Supers = [],
                                  int Offset = 0> : AsmOperandClass {
  let Name = "ConstantUImm" # Bits # "_" # Offset;
  let RenderMethod = "addConstantUImmOperands<" # Bits # ", " # Offset # ">";
  let PredicateMethod = "isConstantUImm<" # Bits # ", " # Offset # ">";
  let SuperClasses = Supers;
  let DiagnosticType = "UImm" # Bits # "_" # Offset;
}

class ConstantUImmRangeAsmOperandClass<int Bottom, int Top,
                                       list<AsmOperandClass> Supers = []>
    : AsmOperandClass {
  let Name = "ConstantUImmRange" # Bottom # "_" # Top;
  let RenderMethod = "addImmOperands";
  let PredicateMethod = "isConstantUImmRange<" # Bottom # ", " # Top # ">";
  let SuperClasses = Supers;
  let DiagnosticType = "UImmRange" # Bottom # "_" # Top;
}

class SImmAsmOperandClass<int Bits, list<AsmOperandClass> Supers = []>
    : AsmOperandClass {
  let Name = "SImm" # Bits;
  let RenderMethod = "addSImmOperands<" # Bits # ">";
  let PredicateMethod = "isSImm<" # Bits # ">";
  let SuperClasses = Supers;
  let DiagnosticType = "SImm" # Bits;
}

class UImmAsmOperandClass<int Bits, list<AsmOperandClass> Supers = []>
    : AsmOperandClass {
  let Name = "UImm" # Bits;
  let RenderMethod = "addUImmOperands<" # Bits # ">";
  let PredicateMethod = "isUImm<" # Bits # ">";
  let SuperClasses = Supers;
  let DiagnosticType = "UImm" # Bits;
}

// AsmOperandClasses require a strict ordering which is difficult to manage
// as a hierarchy. Instead, we use a linear ordering and impose an order that
// is in some places arbitrary.
//
// Here the rules that are in use:
// * Wider immediates are a superset of narrower immediates:
//     uimm4 < uimm5 < uimm6
// * For the same bit-width, unsigned immediates are a superset of signed
//   immediates::
//     simm4 < uimm4 < simm5 < uimm5
// * For the same upper-bound, signed immediates are a superset of unsigned
//   immediates:
//     uimm3 < simm4 < uimm4 < simm4
// * Modified immediates are a superset of ordinary immediates:
//     uimm5 < uimm5_plus1 (1..32) < uimm5_plus32 (32..63) < uimm6
//   The term 'superset' starts to break down here since the uimm5_plus* classes
//   are not true supersets of uimm5 (but they are still subsets of uimm6).
// * 'Relaxed' immediates are supersets of the corresponding unsigned immediate.
//     uimm16 < uimm16_relaxed
// * The codeGen pattern type is arbitrarily ordered.
//     uimm5 < uimm5_64, and uimm5 < vsplat_uimm5
//   This is entirely arbitrary. We need an ordering and what we pick is
//   unimportant since only one is possible for a given mnemonic.
def SImm32RelaxedAsmOperandClass
    : SImmAsmOperandClass<32, []> {
  let Name = "SImm32_Relaxed";
  let PredicateMethod = "isAnyImm<32>";
  let DiagnosticType = "SImm32_Relaxed";
}
def SImm32AsmOperandClass
    : SImmAsmOperandClass<32, [SImm32RelaxedAsmOperandClass]>;
def ConstantUImm26AsmOperandClass
    : ConstantUImmAsmOperandClass<26, [SImm32AsmOperandClass]>;
def ConstantUImm20AsmOperandClass
    : ConstantUImmAsmOperandClass<20, [ConstantUImm26AsmOperandClass]>;
def UImm16RelaxedAsmOperandClass
    : UImmAsmOperandClass<16, [ConstantUImm20AsmOperandClass]> {
  let Name = "UImm16_Relaxed";
  let PredicateMethod = "isAnyImm<16>";
  let DiagnosticType = "UImm16_Relaxed";
}
def UImm16AsmOperandClass
    : UImmAsmOperandClass<16, [UImm16RelaxedAsmOperandClass]>;
def SImm16RelaxedAsmOperandClass
    : SImmAsmOperandClass<16, [UImm16RelaxedAsmOperandClass]> {
  let Name = "SImm16_Relaxed";
  let PredicateMethod = "isAnyImm<16>";
  let DiagnosticType = "SImm16_Relaxed";
}
def SImm16AsmOperandClass
    : SImmAsmOperandClass<16, [SImm16RelaxedAsmOperandClass]>;


def ConstantSImm10Lsl3AsmOperandClass : AsmOperandClass {
  let Name = "SImm10Lsl3";
  let RenderMethod = "addImmOperands";
  let PredicateMethod = "isScaledSImm<10, 3>";
  let SuperClasses = [SImm16AsmOperandClass];
  let DiagnosticType = "SImm10_Lsl3";
}
def ConstantSImm10Lsl2AsmOperandClass : AsmOperandClass {
  let Name = "SImm10Lsl2";
  let RenderMethod = "addImmOperands";
  let PredicateMethod = "isScaledSImm<10, 2>";
  let SuperClasses = [ConstantSImm10Lsl3AsmOperandClass];
  let DiagnosticType = "SImm10_Lsl2";
}
def ConstantSImm11AsmOperandClass
    : ConstantSImmAsmOperandClass<11, [ConstantSImm10Lsl2AsmOperandClass]>;
def ConstantSImm10Lsl1AsmOperandClass : AsmOperandClass {
  let Name = "SImm10Lsl1";
  let RenderMethod = "addImmOperands";
  let PredicateMethod = "isScaledSImm<10, 1>";
  let SuperClasses = [ConstantSImm11AsmOperandClass];
  let DiagnosticType = "SImm10_Lsl1";
}
def ConstantUImm10AsmOperandClass
    : ConstantUImmAsmOperandClass<10, [ConstantSImm10Lsl1AsmOperandClass]>;

def ConstantSImm10AsmOperandClass
    : ConstantSImmAsmOperandClass<10, [ConstantUImm10AsmOperandClass]>;
def ConstantSImm9AsmOperandClass
    : ConstantSImmAsmOperandClass<9, [ConstantSImm10AsmOperandClass]>;
def ConstantSImm7Lsl2AsmOperandClass : AsmOperandClass {
  let Name = "SImm7Lsl2";
  let RenderMethod = "addImmOperands";
  let PredicateMethod = "isScaledSImm<7, 2>";
  let SuperClasses = [ConstantSImm9AsmOperandClass];
  let DiagnosticType = "SImm7_Lsl2";
}
def ConstantUImm8AsmOperandClass
    : ConstantUImmAsmOperandClass<8, [ConstantSImm7Lsl2AsmOperandClass]>;
def ConstantUImm7Sub1AsmOperandClass
    : ConstantUImmAsmOperandClass<7, [ConstantUImm8AsmOperandClass], -1> {
  // Specify the names since the -1 offset causes invalid identifiers otherwise.
  let Name = "UImm7_N1";
  let DiagnosticType = "UImm7_N1";
}
def ConstantUImm7AsmOperandClass
    : ConstantUImmAsmOperandClass<7, [ConstantUImm7Sub1AsmOperandClass]>;
def ConstantUImm6Lsl2AsmOperandClass : AsmOperandClass {
  let Name = "UImm6Lsl2";
  let RenderMethod = "addImmOperands";
  let PredicateMethod = "isScaledUImm<6, 2>";
  let SuperClasses = [ConstantUImm7AsmOperandClass];
  let DiagnosticType = "UImm6_Lsl2";
}
def ConstantUImm6AsmOperandClass
    : ConstantUImmAsmOperandClass<6, [ConstantUImm6Lsl2AsmOperandClass]>;
def ConstantSImm6AsmOperandClass
    : ConstantSImmAsmOperandClass<6, [ConstantUImm6AsmOperandClass]>;
def ConstantUImm5Lsl2AsmOperandClass : AsmOperandClass {
  let Name = "UImm5Lsl2";
  let RenderMethod = "addImmOperands";
  let PredicateMethod = "isScaledUImm<5, 2>";
  let SuperClasses = [ConstantSImm6AsmOperandClass];
  let DiagnosticType = "UImm5_Lsl2";
}

def ConstantUImm5_Range2_64AsmOperandClass
    : ConstantUImmRangeAsmOperandClass<2, 64,
                                       [ConstantUImm5Lsl2AsmOperandClass]>;

def ConstantUImm5Plus33AsmOperandClass
    : ConstantUImmAsmOperandClass<5, [ConstantUImm5_Range2_64AsmOperandClass],
                                  33>;
def ConstantUImm5ReportUImm6AsmOperandClass
    : ConstantUImmAsmOperandClass<5, [ConstantUImm5Plus33AsmOperandClass]> {
  let Name = "ConstantUImm5_0_Report_UImm6";
  let DiagnosticType = "UImm5_0_Report_UImm6";
}
def ConstantUImm5Plus32AsmOperandClass
    : ConstantUImmAsmOperandClass<
          5, [ConstantUImm5ReportUImm6AsmOperandClass], 32>;
def ConstantUImm5Plus32NormalizeAsmOperandClass
    : ConstantUImmAsmOperandClass<5, [ConstantUImm5Plus32AsmOperandClass], 32> {
  let Name = "ConstantUImm5_32_Norm";
  // We must also subtract 32 when we render the operand.
  let RenderMethod = "addConstantUImmOperands<5, 32, -32>";
}
def ConstantUImm5Plus1AsmOperandClass
    : ConstantUImmAsmOperandClass<
          5, [ConstantUImm5Plus32NormalizeAsmOperandClass], 1>;
def ConstantUImm5AsmOperandClass
    : ConstantUImmAsmOperandClass<5, [ConstantUImm5Plus1AsmOperandClass]>;
def ConstantSImm5AsmOperandClass
    : ConstantSImmAsmOperandClass<5, [ConstantUImm5AsmOperandClass]>;
def ConstantUImm4AsmOperandClass
    : ConstantUImmAsmOperandClass<4, [ConstantSImm5AsmOperandClass]>;
def ConstantSImm4AsmOperandClass
    : ConstantSImmAsmOperandClass<4, [ConstantUImm4AsmOperandClass]>;
def ConstantUImm3AsmOperandClass
    : ConstantUImmAsmOperandClass<3, [ConstantSImm4AsmOperandClass]>;
def ConstantUImm2Plus1AsmOperandClass
    : ConstantUImmAsmOperandClass<2, [ConstantUImm3AsmOperandClass], 1>;
def ConstantUImm2AsmOperandClass
    : ConstantUImmAsmOperandClass<2, [ConstantUImm3AsmOperandClass]>;
def ConstantUImm1AsmOperandClass
    : ConstantUImmAsmOperandClass<1, [ConstantUImm2AsmOperandClass]>;
def ConstantImmzAsmOperandClass : AsmOperandClass {
  let Name = "ConstantImmz";
  let RenderMethod = "addConstantUImmOperands<1>";
  let PredicateMethod = "isConstantImmz";
  let SuperClasses = [ConstantUImm1AsmOperandClass];
  let DiagnosticType = "Immz";
}

def ConstantSImm16Lsl2AsmOperandClass : AsmOperandClass {
  let Name = "SImm16Lsl2";
  let RenderMethod = "addImmOperands";
  let PredicateMethod = "isScaledSImm<16, 2>";
  let SuperClasses = [ConstantSImm9AsmOperandClass];
  let DiagnosticType = "SImm16_Lsl2";
}
def ConstantUImm16AsmOperandClass
    : ConstantUImmAsmOperandClass<16, [ConstantSImm16Lsl2AsmOperandClass]>;
def ConstantSImm16AsmOperandClass
    : ConstantSImmAsmOperandClass<16, [ConstantUImm16AsmOperandClass]>;


foreach I = {1, 2, 3, 4} in
  def uimm # I : Operand<i32> {
    let PrintMethod = "printUnsignedImm";
    let ParserMatchClass =
        !cast<AsmOperandClass>("ConstantUImm" # I # "AsmOperandClass");
  }

foreach I = {1, 2, 3, 4, 5, 6, 8, 16} in
  def vsplat_uimm # I : Operand<vAny> {
    let PrintMethod = "printUImm<" # I # ">";
    let ParserMatchClass =
        !cast<AsmOperandClass>("ConstantUImm" # I # "AsmOperandClass");
  }






foreach I = {5, 10, 16} in
  def vsplat_simm # I : Operand<vAny> {
    let ParserMatchClass =
        !cast<AsmOperandClass>("ConstantSImm" # I # "AsmOperandClass");
  }













// ...
def SDT_VSHF : SDTypeProfile<1, 3, [SDTCisInt<0>, SDTCisVec<0>,
                                    SDTCisInt<1>, SDTCisVec<1>,
                                    SDTCisSameAs<0, 2>, SDTCisSameAs<2, 3>]>;
// ...
//def MipsVSHF : SDNode<"MipsISD::VSHF", SDT_VSHF>;
def MipsVSHF : SDNode<"ConnexISD::VSHF", SDT_VSHF>;
// ...




class SplatPatLeaf<Operand opclass, dag frag, code pred = [{}],
                   SDNodeXForm xform = NOOP_SDNodeXForm>
  : PatLeaf<frag, pred, xform> {
  Operand OpClass = opclass;
}

class SplatComplexPattern<Operand opclass, ValueType ty, int numops, string fn,
                          list<SDNode> roots = [],
                          list<SDNodeProperty> props = []> :
  ComplexPattern<ty, numops, fn, roots, props> {
  Operand OpClass = opclass;
}


class MSA_ELM_SPLAT_DESC_BASE<string instr_asm, SplatComplexPattern SplatImm,
                              RegisterOperand ROWD,
                              RegisterOperand ROWS = ROWD,
                              InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs ROWD:$wd);
  dag InOperandList = (ins ROWS:$ws, SplatImm.OpClass:$n);
  string AsmString = !strconcat(instr_asm, "\t$wd, $ws[$n]");
  list<dag> Pattern = [(set ROWD:$wd, (MipsVSHF SplatImm:$n, ROWS:$ws,
                                                ROWS:$ws))];
  InstrItinClass Itinerary = itin;
}


// TODO_CHANGE_BACKEND:
// TODO!!!! Alex: add ~original def vsplati16
//def vsplati16 : PatFrag<(ops node:$e0),
def vsplati64 : PatFrag<(ops node:$e0),
                        (v8i64 (build_vector node:$e0, node:$e0,
                                             node:$e0, node:$e0,
                                             node:$e0, node:$e0,
                                             node:$e0, node:$e0))>;

// NEW32
def vsplati32 : PatFrag<(ops node:$e0),
                        //(v64i32 (build_vector node:$e0, node:$e0,
                        (v4i32 (build_vector node:$e0, node:$e0,
                                              node:$e0, node:$e0,
                                              node:$e0, node:$e0,
                                              node:$e0, node:$e0,
                                              node:$e0, node:$e0,
                                              node:$e0, node:$e0,
                                              node:$e0, node:$e0,
                                              node:$e0, node:$e0,
                                              node:$e0, node:$e0,
                                              node:$e0, node:$e0,
                                              node:$e0, node:$e0,
                                              node:$e0, node:$e0,
                                              node:$e0, node:$e0,
                                              node:$e0, node:$e0,
                                              node:$e0, node:$e0,
                                              node:$e0, node:$e0,
                                              node:$e0, node:$e0,
                                              node:$e0, node:$e0,
                                              node:$e0, node:$e0,
                                              node:$e0, node:$e0,
                                              node:$e0, node:$e0,
                                              node:$e0, node:$e0,
                                              node:$e0, node:$e0,
                                              node:$e0, node:$e0,
                                              node:$e0, node:$e0,
                                              node:$e0, node:$e0,
                                              node:$e0, node:$e0,
                                              node:$e0, node:$e0,
                                              node:$e0, node:$e0,
                                              node:$e0, node:$e0,
                                              node:$e0, node:$e0,
                                              node:$e0, node:$e0))>;



def vsplati64_elt : PatFrag<(ops node:$v, node:$i),
                            (MipsVSHF (vsplati64 node:$i), node:$v, node:$v)>;
def vsplati32_elt : PatFrag<(ops node:$v, node:$i),
                            (MipsVSHF (vsplati32 node:$i), node:$v, node:$v)>;













def vsplati8_uimm3 : SplatComplexPattern<vsplat_uimm3, v16i8, 1,
                                         "selectVSplatUimm3",
                                         [build_vector, bitconvert]>;

def vsplati8_uimm4 : SplatComplexPattern<vsplat_uimm4, v16i8, 1,
                                         "selectVSplatUimm4",
                                         [build_vector, bitconvert]>;
def vsplati8_uimm5 : SplatComplexPattern<vsplat_uimm5, v16i8, 1,
                                         "selectVSplatUimm5",
                                         [build_vector, bitconvert]>;

def vsplati8_uimm8 : SplatComplexPattern<vsplat_uimm8, v16i8, 1,
                                         "selectVSplatUimm8",
                                         [build_vector, bitconvert]>;

def vsplati8_simm5 : SplatComplexPattern<vsplat_simm5, v16i8, 1,
                                         "selectVSplatSimm5",
                                         [build_vector, bitconvert]>;

// Alex: changed v8i16 into v128i16
// TODO_CHANGE_BACKEND:
def vsplati16_uimm3 : SplatComplexPattern<vsplat_uimm3, v8i16, 1,
                                          "selectVSplatUimm3",
                                          [build_vector, bitconvert]>;

def vsplati16_uimm4 : SplatComplexPattern<vsplat_uimm4, v8i16, 1,
                                          "selectVSplatUimm4",
                                          [build_vector, bitconvert]>;

def vsplati16_uimm5 : SplatComplexPattern<vsplat_uimm5, v8i16, 1,
                                          "selectVSplatUimm5",
                                          [build_vector, bitconvert]>;

def vsplati16_simm5 : SplatComplexPattern<vsplat_simm5, v8i16, 1,
                                          "selectVSplatSimm5",
                                          [build_vector, bitconvert]>;

def vsplati16_uimm16 : SplatComplexPattern<vsplat_uimm16, v8i16, 1,
                                          "selectVSplatUimm16",
                                          [build_vector, bitconvert]>;

def vsplati16_simm16 : SplatComplexPattern<vsplat_simm16, v8i16, 1,
                                          "selectVSplatSimm16",
                                          [build_vector, bitconvert]>;

// Alex: changed v4i32 into v16i32
// TODO_CHANGE_BACKEND:
// NEW32
def vsplati32_uimm2 : SplatComplexPattern<vsplat_uimm2, v4i32, 1,
                                          "selectVSplatUimm2",
                                          [build_vector, bitconvert]>;

def vsplati32_uimm5 : SplatComplexPattern<vsplat_uimm5, v4i32, 1,
                                          "selectVSplatUimm5",
                                          [build_vector, bitconvert]>;

def vsplati32_simm5 : SplatComplexPattern<vsplat_simm5, v4i32, 1,
                                          "selectVSplatSimm5",
                                          [build_vector, bitconvert]>;

// Alex: changed v2i64 into v8i64
// TODO_CHANGE_BACKEND:
def vsplati64_uimm1 : SplatComplexPattern<vsplat_uimm1, v2i64, 1,
                                          "selectVSplatUimm1",
                                          [build_vector, bitconvert]>;

def vsplati64_uimm5 : SplatComplexPattern<vsplat_uimm5, v2i64, 1,
                                          "selectVSplatUimm5",
                                          [build_vector, bitconvert]>;

def vsplati64_uimm6 : SplatComplexPattern<vsplat_uimm6, v2i64, 1,
                                          "selectVSplatUimm6",
                                          [build_vector, bitconvert]>;

def vsplati64_simm5 : SplatComplexPattern<vsplat_simm5, v2i64, 1,
                                          "selectVSplatSimm5",
                                          [build_vector, bitconvert]>;



// Any build_vector that is a constant splat with a value that is an exact
// power of 2
def vsplat_uimm_pow2 : ComplexPattern<vAny, 1, "selectVSplatUimmPow2",
                                      [build_vector, bitconvert]>;

// Any build_vector that is a constant splat with a value that is the bitwise
// inverse of an exact power of 2
def vsplat_uimm_inv_pow2 : ComplexPattern<vAny, 1, "selectVSplatUimmInvPow2",
                                          [build_vector, bitconvert]>;







// Any build_vector that is a constant splat with only a consecutive sequence
// of left-most bits set.
def vsplat_maskl_bits_uimm3
    : SplatComplexPattern<vsplat_uimm3, vAny, 1, "selectVSplatMaskL",
                          [build_vector, bitconvert]>;
def vsplat_maskl_bits_uimm4
    : SplatComplexPattern<vsplat_uimm4, vAny, 1, "selectVSplatMaskL",
                          [build_vector, bitconvert]>;
def vsplat_maskl_bits_uimm5
    : SplatComplexPattern<vsplat_uimm5, vAny, 1, "selectVSplatMaskL",
                          [build_vector, bitconvert]>;
def vsplat_maskl_bits_uimm6
    : SplatComplexPattern<vsplat_uimm6, vAny, 1, "selectVSplatMaskL",
                          [build_vector, bitconvert]>;

// Any build_vector that is a constant splat with only a consecutive sequence
// of right-most bits set.
def vsplat_maskr_bits_uimm3
    : SplatComplexPattern<vsplat_uimm3, vAny, 1, "selectVSplatMaskR",
                          [build_vector, bitconvert]>;
def vsplat_maskr_bits_uimm4
    : SplatComplexPattern<vsplat_uimm4, vAny, 1, "selectVSplatMaskR",
                          [build_vector, bitconvert]>;
def vsplat_maskr_bits_uimm5
    : SplatComplexPattern<vsplat_uimm5, vAny, 1, "selectVSplatMaskR",
                          [build_vector, bitconvert]>;
def vsplat_maskr_bits_uimm6
    : SplatComplexPattern<vsplat_uimm6, vAny, 1, "selectVSplatMaskR",
                          [build_vector, bitconvert]>;





class SPLATI_D_DESC : MSA_ELM_SPLAT_DESC_BASE<"splati.h", vsplati64_uimm1,
                                              VectorHOpnd>;

