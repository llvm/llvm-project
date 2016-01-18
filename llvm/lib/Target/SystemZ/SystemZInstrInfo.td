//===-- SystemZInstrInfo.td - General SystemZ instructions ----*- tblgen-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Stack allocation
//===----------------------------------------------------------------------===//

def ADJCALLSTACKDOWN : Pseudo<(outs), (ins i64imm:$amt),
                              [(callseq_start timm:$amt)]>;
def ADJCALLSTACKUP   : Pseudo<(outs), (ins i64imm:$amt1, i64imm:$amt2),
                              [(callseq_end timm:$amt1, timm:$amt2)]>;

let hasSideEffects = 0 in {
  // Takes as input the value of the stack pointer after a dynamic allocation
  // has been made.  Sets the output to the address of the dynamically-
  // allocated area itself, skipping the outgoing arguments.
  //
  // This expands to an LA or LAY instruction.  We restrict the offset
  // to the range of LA and keep the LAY range in reserve for when
  // the size of the outgoing arguments is added.
  def ADJDYNALLOC : Pseudo<(outs GR64:$dst), (ins dynalloc12only:$src),
                           [(set GR64:$dst, dynalloc12only:$src)]>;
}

//===----------------------------------------------------------------------===//
// Control flow instructions
//===----------------------------------------------------------------------===//

// A return instruction (br %r14).
let isReturn = 1, isTerminator = 1, isBarrier = 1, hasCtrlDep = 1 in
  def Return : Alias<2, (outs), (ins), [(z_retflag)]>;

// Unconditional branches.  R1 is the condition-code mask (all 1s).
let isBranch = 1, isTerminator = 1, isBarrier = 1, R1 = 15 in {
  let isIndirectBranch = 1 in
    def BR : InstRR<0x07, (outs), (ins ADDR64:$R2),
                    "br\t$R2", [(brind ADDR64:$R2)]>;

  // An assembler extended mnemonic for BRC.
  def J : InstRI<0xA74, (outs), (ins brtarget16:$I2), "j\t$I2",
                 [(br bb:$I2)]>;

  // An assembler extended mnemonic for BRCL.  (The extension is "G"
  // rather than "L" because "JL" is "Jump if Less".)
  def JG : InstRIL<0xC04, (outs), (ins brtarget32:$I2), "jg\t$I2", []>;
}

// Conditional branches.  It's easier for LLVM to handle these branches
// in their raw BRC/BRCL form, with the 4-bit condition-code mask being
// the first operand.  It seems friendlier to use mnemonic forms like
// JE and JLH when writing out the assembly though.
let isBranch = 1, isTerminator = 1, Uses = [CC] in {
  let isCodeGenOnly = 1, CCMaskFirst = 1 in {
    def BRC : InstRI<0xA74, (outs), (ins cond4:$valid, cond4:$R1,
                                         brtarget16:$I2), "j$R1\t$I2",
                     [(z_br_ccmask cond4:$valid, cond4:$R1, bb:$I2)]>;
    def BRCL : InstRIL<0xC04, (outs), (ins cond4:$valid, cond4:$R1,
                                           brtarget32:$I2), "jg$R1\t$I2", []>;
  }
  def AsmBRC : InstRI<0xA74, (outs), (ins imm32zx4:$R1, brtarget16:$I2),
                      "brc\t$R1, $I2", []>;
  def AsmBRCL : InstRIL<0xC04, (outs), (ins imm32zx4:$R1, brtarget32:$I2),
                        "brcl\t$R1, $I2", []>;
  def AsmBCR : InstRR<0x07, (outs), (ins imm32zx4:$R1, GR64:$R2),
                      "bcr\t$R1, $R2", []>;
}

// Fused compare-and-branch instructions.  As for normal branches,
// we handle these instructions internally in their raw CRJ-like form,
// but use assembly macros like CRJE when writing them out.
//
// These instructions do not use or clobber the condition codes.
// We nevertheless pretend that they clobber CC, so that we can lower
// them to separate comparisons and BRCLs if the branch ends up being
// out of range.
multiclass CompareBranches<Operand ccmask, string pos1, string pos2> {
  let isBranch = 1, isTerminator = 1, Defs = [CC] in {
    def RJ  : InstRIEb<0xEC76, (outs), (ins GR32:$R1, GR32:$R2, ccmask:$M3,
                                            brtarget16:$RI4),
                       "crj"##pos1##"\t$R1, $R2, "##pos2##"$RI4", []>;
    def GRJ : InstRIEb<0xEC64, (outs), (ins GR64:$R1, GR64:$R2, ccmask:$M3,
                                            brtarget16:$RI4),
                       "cgrj"##pos1##"\t$R1, $R2, "##pos2##"$RI4", []>;
    def IJ  : InstRIEc<0xEC7E, (outs), (ins GR32:$R1, imm32sx8:$I2, ccmask:$M3,
                                            brtarget16:$RI4),
                       "cij"##pos1##"\t$R1, $I2, "##pos2##"$RI4", []>;
    def GIJ : InstRIEc<0xEC7C, (outs), (ins GR64:$R1, imm64sx8:$I2, ccmask:$M3,
                                            brtarget16:$RI4),
                       "cgij"##pos1##"\t$R1, $I2, "##pos2##"$RI4", []>;
    def LRJ  : InstRIEb<0xEC77, (outs), (ins GR32:$R1, GR32:$R2, ccmask:$M3,
                                             brtarget16:$RI4),
                        "clrj"##pos1##"\t$R1, $R2, "##pos2##"$RI4", []>;
    def LGRJ : InstRIEb<0xEC65, (outs), (ins GR64:$R1, GR64:$R2, ccmask:$M3,
                                             brtarget16:$RI4),
                        "clgrj"##pos1##"\t$R1, $R2, "##pos2##"$RI4", []>;
    def LIJ  : InstRIEc<0xEC7F, (outs), (ins GR32:$R1, imm32zx8:$I2, ccmask:$M3,
                                             brtarget16:$RI4),
                        "clij"##pos1##"\t$R1, $I2, "##pos2##"$RI4", []>;
    def LGIJ : InstRIEc<0xEC7D, (outs), (ins GR64:$R1, imm64zx8:$I2, ccmask:$M3,
                                             brtarget16:$RI4),
                        "clgij"##pos1##"\t$R1, $I2, "##pos2##"$RI4", []>;
  }
}
let isCodeGenOnly = 1 in
  defm C : CompareBranches<cond4, "$M3", "">;
defm AsmC : CompareBranches<imm32zx4, "", "$M3, ">;

// Define AsmParser mnemonics for each general condition-code mask
// (integer or floating-point)
multiclass CondExtendedMnemonic<bits<4> ccmask, string name> {
  let R1 = ccmask in {
    def J : InstRI<0xA74, (outs), (ins brtarget16:$I2),
                   "j"##name##"\t$I2", []>;
    def JG : InstRIL<0xC04, (outs), (ins brtarget32:$I2),
                     "jg"##name##"\t$I2", []>;
    def BR : InstRR<0x07, (outs), (ins ADDR64:$R2), "b"##name##"r\t$R2", []>;
  }
  def LOCR  : FixedCondUnaryRRF<"locr"##name,  0xB9F2, GR32, GR32, ccmask>;
  def LOCGR : FixedCondUnaryRRF<"locgr"##name, 0xB9E2, GR64, GR64, ccmask>;
  def LOC   : FixedCondUnaryRSY<"loc"##name,   0xEBF2, GR32, ccmask, 4>;
  def LOCG  : FixedCondUnaryRSY<"locg"##name,  0xEBE2, GR64, ccmask, 8>;
  def STOC  : FixedCondStoreRSY<"stoc"##name,  0xEBF3, GR32, ccmask, 4>;
  def STOCG : FixedCondStoreRSY<"stocg"##name, 0xEBE3, GR64, ccmask, 8>;
}
defm AsmO   : CondExtendedMnemonic<1,  "o">;
defm AsmH   : CondExtendedMnemonic<2,  "h">;
defm AsmNLE : CondExtendedMnemonic<3,  "nle">;
defm AsmL   : CondExtendedMnemonic<4,  "l">;
defm AsmNHE : CondExtendedMnemonic<5,  "nhe">;
defm AsmLH  : CondExtendedMnemonic<6,  "lh">;
defm AsmNE  : CondExtendedMnemonic<7,  "ne">;
defm AsmE   : CondExtendedMnemonic<8,  "e">;
defm AsmNLH : CondExtendedMnemonic<9,  "nlh">;
defm AsmHE  : CondExtendedMnemonic<10, "he">;
defm AsmNL  : CondExtendedMnemonic<11, "nl">;
defm AsmLE  : CondExtendedMnemonic<12, "le">;
defm AsmNH  : CondExtendedMnemonic<13, "nh">;
defm AsmNO  : CondExtendedMnemonic<14, "no">;

// Define AsmParser mnemonics for each integer condition-code mask.
// This is like the list above, except that condition 3 is not possible
// and that the low bit of the mask is therefore always 0.  This means
// that each condition has two names.  Conditions "o" and "no" are not used.
//
// We don't make one of the two names an alias of the other because
// we need the custom parsing routines to select the correct register class.
multiclass IntCondExtendedMnemonicA<bits<4> ccmask, string name> {
  let M3 = ccmask in {
    def CR  : InstRIEb<0xEC76, (outs), (ins GR32:$R1, GR32:$R2,
                                            brtarget16:$RI4),
                       "crj"##name##"\t$R1, $R2, $RI4", []>;
    def CGR : InstRIEb<0xEC64, (outs), (ins GR64:$R1, GR64:$R2,
                                            brtarget16:$RI4),
                       "cgrj"##name##"\t$R1, $R2, $RI4", []>;
    def CI  : InstRIEc<0xEC7E, (outs), (ins GR32:$R1, imm32sx8:$I2,
                                            brtarget16:$RI4),
                       "cij"##name##"\t$R1, $I2, $RI4", []>;
    def CGI : InstRIEc<0xEC7C, (outs), (ins GR64:$R1, imm64sx8:$I2,
                                            brtarget16:$RI4),
                       "cgij"##name##"\t$R1, $I2, $RI4", []>;
    def CLR  : InstRIEb<0xEC77, (outs), (ins GR32:$R1, GR32:$R2,
                                            brtarget16:$RI4),
                        "clrj"##name##"\t$R1, $R2, $RI4", []>;
    def CLGR : InstRIEb<0xEC65, (outs), (ins GR64:$R1, GR64:$R2,
                                             brtarget16:$RI4),
                        "clgrj"##name##"\t$R1, $R2, $RI4", []>;
    def CLI  : InstRIEc<0xEC7F, (outs), (ins GR32:$R1, imm32zx8:$I2,
                                             brtarget16:$RI4),
                        "clij"##name##"\t$R1, $I2, $RI4", []>;
    def CLGI : InstRIEc<0xEC7D, (outs), (ins GR64:$R1, imm64zx8:$I2,
                                             brtarget16:$RI4),
                        "clgij"##name##"\t$R1, $I2, $RI4", []>;
  }
}
multiclass IntCondExtendedMnemonic<bits<4> ccmask, string name1, string name2>
  : IntCondExtendedMnemonicA<ccmask, name1> {
  let isAsmParserOnly = 1 in
    defm Alt : IntCondExtendedMnemonicA<ccmask, name2>;
}
defm AsmJH   : IntCondExtendedMnemonic<2,  "h",  "nle">;
defm AsmJL   : IntCondExtendedMnemonic<4,  "l",  "nhe">;
defm AsmJLH  : IntCondExtendedMnemonic<6,  "lh", "ne">;
defm AsmJE   : IntCondExtendedMnemonic<8,  "e",  "nlh">;
defm AsmJHE  : IntCondExtendedMnemonic<10, "he", "nl">;
defm AsmJLE  : IntCondExtendedMnemonic<12, "le", "nh">;

// Decrement a register and branch if it is nonzero.  These don't clobber CC,
// but we might need to split long branches into sequences that do.
let Defs = [CC] in {
  def BRCT  : BranchUnaryRI<"brct",  0xA76, GR32>;
  def BRCTG : BranchUnaryRI<"brctg", 0xA77, GR64>;
}

//===----------------------------------------------------------------------===//
// Select instructions
//===----------------------------------------------------------------------===//

def Select32Mux : SelectWrapper<GRX32>, Requires<[FeatureHighWord]>;
def Select32    : SelectWrapper<GR32>;
def Select64    : SelectWrapper<GR64>;

// We don't define 32-bit Mux stores because the low-only STOC should
// always be used if possible.
defm CondStore8Mux  : CondStores<GRX32, nonvolatile_truncstorei8,
                                 nonvolatile_anyextloadi8, bdxaddr20only>,
                      Requires<[FeatureHighWord]>;
defm CondStore16Mux : CondStores<GRX32, nonvolatile_truncstorei16,
                                 nonvolatile_anyextloadi16, bdxaddr20only>,
                      Requires<[FeatureHighWord]>;
defm CondStore8     : CondStores<GR32, nonvolatile_truncstorei8,
                                 nonvolatile_anyextloadi8, bdxaddr20only>;
defm CondStore16    : CondStores<GR32, nonvolatile_truncstorei16,
                                 nonvolatile_anyextloadi16, bdxaddr20only>;
defm CondStore32    : CondStores<GR32, nonvolatile_store,
                                 nonvolatile_load, bdxaddr20only>;

defm : CondStores64<CondStore8, CondStore8Inv, nonvolatile_truncstorei8,
                    nonvolatile_anyextloadi8, bdxaddr20only>;
defm : CondStores64<CondStore16, CondStore16Inv, nonvolatile_truncstorei16,
                    nonvolatile_anyextloadi16, bdxaddr20only>;
defm : CondStores64<CondStore32, CondStore32Inv, nonvolatile_truncstorei32,
                    nonvolatile_anyextloadi32, bdxaddr20only>;
defm CondStore64 : CondStores<GR64, nonvolatile_store,
                              nonvolatile_load, bdxaddr20only>;

//===----------------------------------------------------------------------===//
// Call instructions
//===----------------------------------------------------------------------===//

let isCall = 1, Defs = [R14D, CC] in {
  def CallBRASL : Alias<6, (outs), (ins pcrel32:$I2, variable_ops),
                        [(z_call pcrel32:$I2)]>;
  def CallBASR  : Alias<2, (outs), (ins ADDR64:$R2, variable_ops),
                        [(z_call ADDR64:$R2)]>;
}

// Sibling calls.  Indirect sibling calls must be via R1, since R2 upwards
// are argument registers and since branching to R0 is a no-op.
let isCall = 1, isTerminator = 1, isReturn = 1, isBarrier = 1 in {
  def CallJG : Alias<6, (outs), (ins pcrel32:$I2),
                     [(z_sibcall pcrel32:$I2)]>;
  let Uses = [R1D] in
    def CallBR : Alias<2, (outs), (ins), [(z_sibcall R1D)]>;
}

// TLS calls.  These will be lowered into a call to __tls_get_offset,
// with an extra relocation specifying the TLS symbol.
let isCall = 1, Defs = [R14D, CC] in {
  def TLS_GDCALL : Alias<6, (outs), (ins tlssym:$I2, variable_ops),
                         [(z_tls_gdcall tglobaltlsaddr:$I2)]>;
  def TLS_LDCALL : Alias<6, (outs), (ins tlssym:$I2, variable_ops),
                         [(z_tls_ldcall tglobaltlsaddr:$I2)]>;
}

// Define the general form of the call instructions for the asm parser.
// These instructions don't hard-code %r14 as the return address register.
// Allow an optional TLS marker symbol to generate TLS call relocations.
def BRAS  : InstRI<0xA75, (outs), (ins GR64:$R1, brtarget16tls:$I2),
                   "bras\t$R1, $I2", []>;
def BRASL : InstRIL<0xC05, (outs), (ins GR64:$R1, brtarget32tls:$I2),
                    "brasl\t$R1, $I2", []>;
def BASR  : InstRR<0x0D, (outs), (ins GR64:$R1, ADDR64:$R2),
                   "basr\t$R1, $R2", []>;

//===----------------------------------------------------------------------===//
// Move instructions
//===----------------------------------------------------------------------===//

// Register moves.
let hasSideEffects = 0 in {
  // Expands to LR, RISBHG or RISBLG, depending on the choice of registers.
  def LRMux : UnaryRRPseudo<"l", null_frag, GRX32, GRX32>,
              Requires<[FeatureHighWord]>;
  def LR  : UnaryRR <"l",  0x18,   null_frag, GR32, GR32>;
  def LGR : UnaryRRE<"lg", 0xB904, null_frag, GR64, GR64>;
}
let Defs = [CC], CCValues = 0xE, CompareZeroCCMask = 0xE in {
  def LTR  : UnaryRR <"lt",  0x12,   null_frag, GR32, GR32>;
  def LTGR : UnaryRRE<"ltg", 0xB902, null_frag, GR64, GR64>;
}

// Move on condition.
let isCodeGenOnly = 1, Uses = [CC] in {
  def LOCR  : CondUnaryRRF<"loc",  0xB9F2, GR32, GR32>;
  def LOCGR : CondUnaryRRF<"locg", 0xB9E2, GR64, GR64>;
}
let Uses = [CC] in {
  def AsmLOCR  : AsmCondUnaryRRF<"loc",  0xB9F2, GR32, GR32>;
  def AsmLOCGR : AsmCondUnaryRRF<"locg", 0xB9E2, GR64, GR64>;
}

// Immediate moves.
let hasSideEffects = 0, isAsCheapAsAMove = 1, isMoveImm = 1,
    isReMaterializable = 1 in {
  // 16-bit sign-extended immediates.  LHIMux expands to LHI or IIHF,
  // deopending on the choice of register.
  def LHIMux : UnaryRIPseudo<bitconvert, GRX32, imm32sx16>,
               Requires<[FeatureHighWord]>;
  def LHI  : UnaryRI<"lhi",  0xA78, bitconvert, GR32, imm32sx16>;
  def LGHI : UnaryRI<"lghi", 0xA79, bitconvert, GR64, imm64sx16>;

  // Other 16-bit immediates.
  def LLILL : UnaryRI<"llill", 0xA5F, bitconvert, GR64, imm64ll16>;
  def LLILH : UnaryRI<"llilh", 0xA5E, bitconvert, GR64, imm64lh16>;
  def LLIHL : UnaryRI<"llihl", 0xA5D, bitconvert, GR64, imm64hl16>;
  def LLIHH : UnaryRI<"llihh", 0xA5C, bitconvert, GR64, imm64hh16>;

  // 32-bit immediates.
  def LGFI  : UnaryRIL<"lgfi",  0xC01, bitconvert, GR64, imm64sx32>;
  def LLILF : UnaryRIL<"llilf", 0xC0F, bitconvert, GR64, imm64lf32>;
  def LLIHF : UnaryRIL<"llihf", 0xC0E, bitconvert, GR64, imm64hf32>;
}

// Register loads.
let canFoldAsLoad = 1, SimpleBDXLoad = 1 in {
  // Expands to L, LY or LFH, depending on the choice of register.
  def LMux : UnaryRXYPseudo<"l", load, GRX32, 4>,
             Requires<[FeatureHighWord]>;
  defm L : UnaryRXPair<"l", 0x58, 0xE358, load, GR32, 4>;
  def LFH : UnaryRXY<"lfh", 0xE3CA, load, GRH32, 4>,
            Requires<[FeatureHighWord]>;
  def LG : UnaryRXY<"lg", 0xE304, load, GR64, 8>;

  // These instructions are split after register allocation, so we don't
  // want a custom inserter.
  let Has20BitOffset = 1, HasIndex = 1, Is128Bit = 1 in {
    def L128 : Pseudo<(outs GR128:$dst), (ins bdxaddr20only128:$src),
                      [(set GR128:$dst, (load bdxaddr20only128:$src))]>;
  }
}
let Defs = [CC], CCValues = 0xE, CompareZeroCCMask = 0xE in {
  def LT  : UnaryRXY<"lt",  0xE312, load, GR32, 4>;
  def LTG : UnaryRXY<"ltg", 0xE302, load, GR64, 8>;
}

let canFoldAsLoad = 1 in {
  def LRL  : UnaryRILPC<"lrl",  0xC4D, aligned_load, GR32>;
  def LGRL : UnaryRILPC<"lgrl", 0xC48, aligned_load, GR64>;
}

// Load on condition.
let isCodeGenOnly = 1, Uses = [CC] in {
  def LOC  : CondUnaryRSY<"loc",  0xEBF2, nonvolatile_load, GR32, 4>;
  def LOCG : CondUnaryRSY<"locg", 0xEBE2, nonvolatile_load, GR64, 8>;
}
let Uses = [CC] in {
  def AsmLOC  : AsmCondUnaryRSY<"loc",  0xEBF2, GR32, 4>;
  def AsmLOCG : AsmCondUnaryRSY<"locg", 0xEBE2, GR64, 8>;
}

// Register stores.
let SimpleBDXStore = 1 in {
  // Expands to ST, STY or STFH, depending on the choice of register.
  def STMux : StoreRXYPseudo<store, GRX32, 4>,
              Requires<[FeatureHighWord]>;
  defm ST : StoreRXPair<"st", 0x50, 0xE350, store, GR32, 4>;
  def STFH : StoreRXY<"stfh", 0xE3CB, store, GRH32, 4>,
             Requires<[FeatureHighWord]>;
  def STG : StoreRXY<"stg", 0xE324, store, GR64, 8>;

  // These instructions are split after register allocation, so we don't
  // want a custom inserter.
  let Has20BitOffset = 1, HasIndex = 1, Is128Bit = 1 in {
    def ST128 : Pseudo<(outs), (ins GR128:$src, bdxaddr20only128:$dst),
                       [(store GR128:$src, bdxaddr20only128:$dst)]>;
  }
}
def STRL  : StoreRILPC<"strl", 0xC4F, aligned_store, GR32>;
def STGRL : StoreRILPC<"stgrl", 0xC4B, aligned_store, GR64>;

// Store on condition.
let isCodeGenOnly = 1, Uses = [CC] in {
  def STOC  : CondStoreRSY<"stoc",  0xEBF3, GR32, 4>;
  def STOCG : CondStoreRSY<"stocg", 0xEBE3, GR64, 8>;
}
let Uses = [CC] in {
  def AsmSTOC  : AsmCondStoreRSY<"stoc",  0xEBF3, GR32, 4>;
  def AsmSTOCG : AsmCondStoreRSY<"stocg", 0xEBE3, GR64, 8>;
}

// 8-bit immediate stores to 8-bit fields.
defm MVI : StoreSIPair<"mvi", 0x92, 0xEB52, truncstorei8, imm32zx8trunc>;

// 16-bit immediate stores to 16-, 32- or 64-bit fields.
def MVHHI : StoreSIL<"mvhhi", 0xE544, truncstorei16, imm32sx16trunc>;
def MVHI  : StoreSIL<"mvhi",  0xE54C, store,         imm32sx16>;
def MVGHI : StoreSIL<"mvghi", 0xE548, store,         imm64sx16>;

// Memory-to-memory moves.
let mayLoad = 1, mayStore = 1 in
  defm MVC : MemorySS<"mvc", 0xD2, z_mvc, z_mvc_loop>;

// String moves.
let mayLoad = 1, mayStore = 1, Defs = [CC] in
  defm MVST : StringRRE<"mvst", 0xB255, z_stpcpy>;

//===----------------------------------------------------------------------===//
// Sign extensions
//===----------------------------------------------------------------------===//
//
// Note that putting these before zero extensions mean that we will prefer
// them for anyextload*.  There's not really much to choose between the two
// either way, but signed-extending loads have a short LH and a long LHY,
// while zero-extending loads have only the long LLH.
//
//===----------------------------------------------------------------------===//

// 32-bit extensions from registers.
let hasSideEffects = 0 in {
  def LBR : UnaryRRE<"lb", 0xB926, sext8,  GR32, GR32>;
  def LHR : UnaryRRE<"lh", 0xB927, sext16, GR32, GR32>;
}

// 64-bit extensions from registers.
let hasSideEffects = 0 in {
  def LGBR : UnaryRRE<"lgb", 0xB906, sext8,  GR64, GR64>;
  def LGHR : UnaryRRE<"lgh", 0xB907, sext16, GR64, GR64>;
  def LGFR : UnaryRRE<"lgf", 0xB914, sext32, GR64, GR32>;
}
let Defs = [CC], CCValues = 0xE, CompareZeroCCMask = 0xE in
  def LTGFR : UnaryRRE<"ltgf", 0xB912, null_frag, GR64, GR32>;

// Match 32-to-64-bit sign extensions in which the source is already
// in a 64-bit register.
def : Pat<(sext_inreg GR64:$src, i32),
          (LGFR (EXTRACT_SUBREG GR64:$src, subreg_l32))>;

// 32-bit extensions from 8-bit memory.  LBMux expands to LB or LBH,
// depending on the choice of register.
def LBMux : UnaryRXYPseudo<"lb", asextloadi8, GRX32, 1>,
            Requires<[FeatureHighWord]>;
def LB  : UnaryRXY<"lb", 0xE376, asextloadi8, GR32, 1>;
def LBH : UnaryRXY<"lbh", 0xE3C0, asextloadi8, GRH32, 1>,
          Requires<[FeatureHighWord]>;

// 32-bit extensions from 16-bit memory.  LHMux expands to LH or LHH,
// depending on the choice of register.
def LHMux : UnaryRXYPseudo<"lh", asextloadi16, GRX32, 2>,
            Requires<[FeatureHighWord]>;
defm LH   : UnaryRXPair<"lh", 0x48, 0xE378, asextloadi16, GR32, 2>;
def  LHH  : UnaryRXY<"lhh", 0xE3C4, asextloadi16, GRH32, 2>,
            Requires<[FeatureHighWord]>;
def  LHRL : UnaryRILPC<"lhrl", 0xC45, aligned_asextloadi16, GR32>;

// 64-bit extensions from memory.
def LGB   : UnaryRXY<"lgb", 0xE377, asextloadi8,  GR64, 1>;
def LGH   : UnaryRXY<"lgh", 0xE315, asextloadi16, GR64, 2>;
def LGF   : UnaryRXY<"lgf", 0xE314, asextloadi32, GR64, 4>;
def LGHRL : UnaryRILPC<"lghrl", 0xC44, aligned_asextloadi16, GR64>;
def LGFRL : UnaryRILPC<"lgfrl", 0xC4C, aligned_asextloadi32, GR64>;
let Defs = [CC], CCValues = 0xE, CompareZeroCCMask = 0xE in
  def LTGF : UnaryRXY<"ltgf", 0xE332, asextloadi32, GR64, 4>;

//===----------------------------------------------------------------------===//
// Zero extensions
//===----------------------------------------------------------------------===//

// 32-bit extensions from registers.
let hasSideEffects = 0 in {
  // Expands to LLCR or RISB[LH]G, depending on the choice of registers.
  def LLCRMux : UnaryRRPseudo<"llc", zext8, GRX32, GRX32>,
                Requires<[FeatureHighWord]>;
  def LLCR    : UnaryRRE<"llc", 0xB994, zext8,  GR32, GR32>;
  // Expands to LLHR or RISB[LH]G, depending on the choice of registers.
  def LLHRMux : UnaryRRPseudo<"llh", zext16, GRX32, GRX32>,
                Requires<[FeatureHighWord]>;
  def LLHR    : UnaryRRE<"llh", 0xB995, zext16, GR32, GR32>;
}

// 64-bit extensions from registers.
let hasSideEffects = 0 in {
  def LLGCR : UnaryRRE<"llgc", 0xB984, zext8,  GR64, GR64>;
  def LLGHR : UnaryRRE<"llgh", 0xB985, zext16, GR64, GR64>;
  def LLGFR : UnaryRRE<"llgf", 0xB916, zext32, GR64, GR32>;
}

// Match 32-to-64-bit zero extensions in which the source is already
// in a 64-bit register.
def : Pat<(and GR64:$src, 0xffffffff),
          (LLGFR (EXTRACT_SUBREG GR64:$src, subreg_l32))>;

// 32-bit extensions from 8-bit memory.  LLCMux expands to LLC or LLCH,
// depending on the choice of register.
def LLCMux : UnaryRXYPseudo<"llc", azextloadi8, GRX32, 1>,
             Requires<[FeatureHighWord]>;
def LLC  : UnaryRXY<"llc", 0xE394, azextloadi8, GR32, 1>;
def LLCH : UnaryRXY<"llch", 0xE3C2, azextloadi8, GRH32, 1>,
           Requires<[FeatureHighWord]>;

// 32-bit extensions from 16-bit memory.  LLHMux expands to LLH or LLHH,
// depending on the choice of register.
def LLHMux : UnaryRXYPseudo<"llh", azextloadi16, GRX32, 2>,
             Requires<[FeatureHighWord]>;
def LLH   : UnaryRXY<"llh", 0xE395, azextloadi16, GR32, 2>;
def LLHH  : UnaryRXY<"llhh", 0xE3C6, azextloadi16, GRH32, 2>,
            Requires<[FeatureHighWord]>;
def LLHRL : UnaryRILPC<"llhrl", 0xC42, aligned_azextloadi16, GR32>;

// 64-bit extensions from memory.
def LLGC   : UnaryRXY<"llgc", 0xE390, azextloadi8,  GR64, 1>;
def LLGH   : UnaryRXY<"llgh", 0xE391, azextloadi16, GR64, 2>;
def LLGF   : UnaryRXY<"llgf", 0xE316, azextloadi32, GR64, 4>;
def LLGHRL : UnaryRILPC<"llghrl", 0xC46, aligned_azextloadi16, GR64>;
def LLGFRL : UnaryRILPC<"llgfrl", 0xC4E, aligned_azextloadi32, GR64>;

//===----------------------------------------------------------------------===//
// Truncations
//===----------------------------------------------------------------------===//

// Truncations of 64-bit registers to 32-bit registers.
def : Pat<(i32 (trunc GR64:$src)),
          (EXTRACT_SUBREG GR64:$src, subreg_l32)>;

// Truncations of 32-bit registers to 8-bit memory.  STCMux expands to
// STC, STCY or STCH, depending on the choice of register.
def STCMux : StoreRXYPseudo<truncstorei8, GRX32, 1>,
             Requires<[FeatureHighWord]>;
defm STC : StoreRXPair<"stc", 0x42, 0xE372, truncstorei8, GR32, 1>;
def STCH : StoreRXY<"stch", 0xE3C3, truncstorei8, GRH32, 1>,
           Requires<[FeatureHighWord]>;

// Truncations of 32-bit registers to 16-bit memory.  STHMux expands to
// STH, STHY or STHH, depending on the choice of register.
def STHMux : StoreRXYPseudo<truncstorei16, GRX32, 1>,
             Requires<[FeatureHighWord]>;
defm STH : StoreRXPair<"sth", 0x40, 0xE370, truncstorei16, GR32, 2>;
def STHH : StoreRXY<"sthh", 0xE3C7, truncstorei16, GRH32, 2>,
           Requires<[FeatureHighWord]>;
def STHRL : StoreRILPC<"sthrl", 0xC47, aligned_truncstorei16, GR32>;

// Truncations of 64-bit registers to memory.
defm : StoreGR64Pair<STC, STCY, truncstorei8>;
defm : StoreGR64Pair<STH, STHY, truncstorei16>;
def  : StoreGR64PC<STHRL, aligned_truncstorei16>;
defm : StoreGR64Pair<ST, STY, truncstorei32>;
def  : StoreGR64PC<STRL, aligned_truncstorei32>;

//===----------------------------------------------------------------------===//
// Multi-register moves
//===----------------------------------------------------------------------===//

// Multi-register loads.
def LMG : LoadMultipleRSY<"lmg", 0xEB04, GR64>;

// Multi-register stores.
def STMG : StoreMultipleRSY<"stmg", 0xEB24, GR64>;

//===----------------------------------------------------------------------===//
// Byte swaps
//===----------------------------------------------------------------------===//

// Byte-swapping register moves.
let hasSideEffects = 0 in {
  def LRVR  : UnaryRRE<"lrv",  0xB91F, bswap, GR32, GR32>;
  def LRVGR : UnaryRRE<"lrvg", 0xB90F, bswap, GR64, GR64>;
}

// Byte-swapping loads.  Unlike normal loads, these instructions are
// allowed to access storage more than once.
def LRV  : UnaryRXY<"lrv",  0xE31E, loadu<bswap, nonvolatile_load>, GR32, 4>;
def LRVG : UnaryRXY<"lrvg", 0xE30F, loadu<bswap, nonvolatile_load>, GR64, 8>;

// Likewise byte-swapping stores.
def STRV  : StoreRXY<"strv", 0xE33E, storeu<bswap, nonvolatile_store>, GR32, 4>;
def STRVG : StoreRXY<"strvg", 0xE32F, storeu<bswap, nonvolatile_store>,
                     GR64, 8>;

//===----------------------------------------------------------------------===//
// Load address instructions
//===----------------------------------------------------------------------===//

// Load BDX-style addresses.
let hasSideEffects = 0, isAsCheapAsAMove = 1, isReMaterializable = 1,
    DispKey = "la" in {
  let DispSize = "12" in
    def LA : InstRX<0x41, (outs GR64:$R1), (ins laaddr12pair:$XBD2),
                    "la\t$R1, $XBD2",
                    [(set GR64:$R1, laaddr12pair:$XBD2)]>;
  let DispSize = "20" in
    def LAY : InstRXY<0xE371, (outs GR64:$R1), (ins laaddr20pair:$XBD2),
                      "lay\t$R1, $XBD2",
                      [(set GR64:$R1, laaddr20pair:$XBD2)]>;
}

// Load a PC-relative address.  There's no version of this instruction
// with a 16-bit offset, so there's no relaxation.
let hasSideEffects = 0, isAsCheapAsAMove = 1, isMoveImm = 1,
    isReMaterializable = 1 in {
  def LARL : InstRIL<0xC00, (outs GR64:$R1), (ins pcrel32:$I2),
                     "larl\t$R1, $I2",
                     [(set GR64:$R1, pcrel32:$I2)]>;
}

// Load the Global Offset Table address.  This will be lowered into a
//     larl $R1, _GLOBAL_OFFSET_TABLE_
// instruction.
def GOT : Alias<6, (outs GR64:$R1), (ins),
                [(set GR64:$R1, (global_offset_table))]>;

//===----------------------------------------------------------------------===//
// Absolute and Negation
//===----------------------------------------------------------------------===//

let Defs = [CC] in {
  let CCValues = 0xF, CompareZeroCCMask = 0x8 in {
    def LPR  : UnaryRR <"lp",  0x10,   z_iabs, GR32, GR32>;
    def LPGR : UnaryRRE<"lpg", 0xB900, z_iabs, GR64, GR64>;
  }
  let CCValues = 0xE, CompareZeroCCMask = 0xE in
    def LPGFR : UnaryRRE<"lpgf", 0xB910, null_frag, GR64, GR32>;
}
def : Pat<(z_iabs32 GR32:$src), (LPR  GR32:$src)>;
def : Pat<(z_iabs64 GR64:$src), (LPGR GR64:$src)>;
defm : SXU<z_iabs,   LPGFR>;
defm : SXU<z_iabs64, LPGFR>;

let Defs = [CC] in {
  let CCValues = 0xF, CompareZeroCCMask = 0x8 in {
    def LNR  : UnaryRR <"ln",  0x11,   z_inegabs, GR32, GR32>;
    def LNGR : UnaryRRE<"lng", 0xB901, z_inegabs, GR64, GR64>;
  }
  let CCValues = 0xE, CompareZeroCCMask = 0xE in
    def LNGFR : UnaryRRE<"lngf", 0xB911, null_frag, GR64, GR32>;
}
def : Pat<(z_inegabs32 GR32:$src), (LNR  GR32:$src)>;
def : Pat<(z_inegabs64 GR64:$src), (LNGR GR64:$src)>;
defm : SXU<z_inegabs,   LNGFR>;
defm : SXU<z_inegabs64, LNGFR>;

let Defs = [CC] in {
  let CCValues = 0xF, CompareZeroCCMask = 0x8 in {
    def LCR  : UnaryRR <"lc",  0x13,   ineg, GR32, GR32>;
    def LCGR : UnaryRRE<"lcg", 0xB903, ineg, GR64, GR64>;
  }
  let CCValues = 0xE, CompareZeroCCMask = 0xE in
    def LCGFR : UnaryRRE<"lcgf", 0xB913, null_frag, GR64, GR32>;
}
defm : SXU<ineg, LCGFR>;

//===----------------------------------------------------------------------===//
// Insertion
//===----------------------------------------------------------------------===//

let isCodeGenOnly = 1 in
  defm IC32 : BinaryRXPair<"ic", 0x43, 0xE373, inserti8, GR32, azextloadi8, 1>;
defm IC : BinaryRXPair<"ic", 0x43, 0xE373, inserti8, GR64, azextloadi8, 1>;

defm : InsertMem<"inserti8", IC32,  GR32, azextloadi8, bdxaddr12pair>;
defm : InsertMem<"inserti8", IC32Y, GR32, azextloadi8, bdxaddr20pair>;

defm : InsertMem<"inserti8", IC,  GR64, azextloadi8, bdxaddr12pair>;
defm : InsertMem<"inserti8", ICY, GR64, azextloadi8, bdxaddr20pair>;

// Insertions of a 16-bit immediate, leaving other bits unaffected.
// We don't have or_as_insert equivalents of these operations because
// OI is available instead.
//
// IIxMux expands to II[LH]x, depending on the choice of register.
def IILMux : BinaryRIPseudo<insertll, GRX32, imm32ll16>,
             Requires<[FeatureHighWord]>;
def IIHMux : BinaryRIPseudo<insertlh, GRX32, imm32lh16>,
             Requires<[FeatureHighWord]>;
def IILL : BinaryRI<"iill", 0xA53, insertll, GR32, imm32ll16>;
def IILH : BinaryRI<"iilh", 0xA52, insertlh, GR32, imm32lh16>;
def IIHL : BinaryRI<"iihl", 0xA51, insertll, GRH32, imm32ll16>;
def IIHH : BinaryRI<"iihh", 0xA50, insertlh, GRH32, imm32lh16>;
def IILL64 : BinaryAliasRI<insertll, GR64, imm64ll16>;
def IILH64 : BinaryAliasRI<insertlh, GR64, imm64lh16>;
def IIHL64 : BinaryAliasRI<inserthl, GR64, imm64hl16>;
def IIHH64 : BinaryAliasRI<inserthh, GR64, imm64hh16>;

// ...likewise for 32-bit immediates.  For GR32s this is a general
// full-width move.  (We use IILF rather than something like LLILF
// for 32-bit moves because IILF leaves the upper 32 bits of the
// GR64 unchanged.)
let isAsCheapAsAMove = 1, isMoveImm = 1, isReMaterializable = 1 in {
  def IIFMux : UnaryRIPseudo<bitconvert, GRX32, uimm32>,
               Requires<[FeatureHighWord]>;
  def IILF : UnaryRIL<"iilf", 0xC09, bitconvert, GR32, uimm32>;
  def IIHF : UnaryRIL<"iihf", 0xC08, bitconvert, GRH32, uimm32>;
}
def IILF64 : BinaryAliasRIL<insertlf, GR64, imm64lf32>;
def IIHF64 : BinaryAliasRIL<inserthf, GR64, imm64hf32>;

// An alternative model of inserthf, with the first operand being
// a zero-extended value.
def : Pat<(or (zext32 GR32:$src), imm64hf32:$imm),
          (IIHF64 (INSERT_SUBREG (i64 (IMPLICIT_DEF)), GR32:$src, subreg_l32),
                  imm64hf32:$imm)>;

//===----------------------------------------------------------------------===//
// Addition
//===----------------------------------------------------------------------===//

// Plain addition.
let Defs = [CC], CCValues = 0xF, CompareZeroCCMask = 0x8 in {
  // Addition of a register.
  let isCommutable = 1 in {
    defm AR : BinaryRRAndK<"a", 0x1A, 0xB9F8, add, GR32, GR32>;
    defm AGR : BinaryRREAndK<"ag", 0xB908, 0xB9E8, add, GR64, GR64>;
  }
  def AGFR : BinaryRRE<"agf", 0xB918, null_frag, GR64, GR32>;

  // Addition of signed 16-bit immediates.
  defm AHIMux : BinaryRIAndKPseudo<"ahimux", add, GRX32, imm32sx16>;
  defm AHI  : BinaryRIAndK<"ahi",  0xA7A, 0xECD8, add, GR32, imm32sx16>;
  defm AGHI : BinaryRIAndK<"aghi", 0xA7B, 0xECD9, add, GR64, imm64sx16>;

  // Addition of signed 32-bit immediates.
  def AFIMux : BinaryRIPseudo<add, GRX32, simm32>,
               Requires<[FeatureHighWord]>;
  def AFI  : BinaryRIL<"afi",  0xC29, add, GR32, simm32>;
  def AIH  : BinaryRIL<"aih",  0xCC8, add, GRH32, simm32>,
             Requires<[FeatureHighWord]>;
  def AGFI : BinaryRIL<"agfi", 0xC28, add, GR64, imm64sx32>;

  // Addition of memory.
  defm AH  : BinaryRXPair<"ah", 0x4A, 0xE37A, add, GR32, asextloadi16, 2>;
  defm A   : BinaryRXPair<"a",  0x5A, 0xE35A, add, GR32, load, 4>;
  def  AGF : BinaryRXY<"agf", 0xE318, add, GR64, asextloadi32, 4>;
  def  AG  : BinaryRXY<"ag",  0xE308, add, GR64, load, 8>;

  // Addition to memory.
  def ASI  : BinarySIY<"asi",  0xEB6A, add, imm32sx8>;
  def AGSI : BinarySIY<"agsi", 0xEB7A, add, imm64sx8>;
}
defm : SXB<add, GR64, AGFR>;

// Addition producing a carry.
let Defs = [CC] in {
  // Addition of a register.
  let isCommutable = 1 in {
    defm ALR : BinaryRRAndK<"al", 0x1E, 0xB9FA, addc, GR32, GR32>;
    defm ALGR : BinaryRREAndK<"alg", 0xB90A, 0xB9EA, addc, GR64, GR64>;
  }
  def ALGFR : BinaryRRE<"algf", 0xB91A, null_frag, GR64, GR32>;

  // Addition of signed 16-bit immediates.
  def ALHSIK  : BinaryRIE<"alhsik",  0xECDA, addc, GR32, imm32sx16>,
                Requires<[FeatureDistinctOps]>;
  def ALGHSIK : BinaryRIE<"alghsik", 0xECDB, addc, GR64, imm64sx16>,
                Requires<[FeatureDistinctOps]>;

  // Addition of unsigned 32-bit immediates.
  def ALFI  : BinaryRIL<"alfi",  0xC2B, addc, GR32, uimm32>;
  def ALGFI : BinaryRIL<"algfi", 0xC2A, addc, GR64, imm64zx32>;

  // Addition of memory.
  defm AL   : BinaryRXPair<"al", 0x5E, 0xE35E, addc, GR32, load, 4>;
  def  ALGF : BinaryRXY<"algf", 0xE31A, addc, GR64, azextloadi32, 4>;
  def  ALG  : BinaryRXY<"alg",  0xE30A, addc, GR64, load, 8>;
}
defm : ZXB<addc, GR64, ALGFR>;

// Addition producing and using a carry.
let Defs = [CC], Uses = [CC] in {
  // Addition of a register.
  def ALCR  : BinaryRRE<"alc",  0xB998, adde, GR32, GR32>;
  def ALCGR : BinaryRRE<"alcg", 0xB988, adde, GR64, GR64>;

  // Addition of memory.
  def ALC  : BinaryRXY<"alc",  0xE398, adde, GR32, load, 4>;
  def ALCG : BinaryRXY<"alcg", 0xE388, adde, GR64, load, 8>;
}

//===----------------------------------------------------------------------===//
// Subtraction
//===----------------------------------------------------------------------===//

// Plain subtraction.  Although immediate forms exist, we use the
// add-immediate instruction instead.
let Defs = [CC], CCValues = 0xF, CompareZeroCCMask = 0x8 in {
  // Subtraction of a register.
  defm SR : BinaryRRAndK<"s", 0x1B, 0xB9F9, sub, GR32, GR32>;
  def SGFR : BinaryRRE<"sgf", 0xB919, null_frag, GR64, GR32>;
  defm SGR : BinaryRREAndK<"sg", 0xB909, 0xB9E9, sub, GR64, GR64>;

  // Subtraction of memory.
  defm SH  : BinaryRXPair<"sh", 0x4B, 0xE37B, sub, GR32, asextloadi16, 2>;
  defm S   : BinaryRXPair<"s", 0x5B, 0xE35B, sub, GR32, load, 4>;
  def  SGF : BinaryRXY<"sgf", 0xE319, sub, GR64, asextloadi32, 4>;
  def  SG  : BinaryRXY<"sg",  0xE309, sub, GR64, load, 8>;
}
defm : SXB<sub, GR64, SGFR>;

// Subtraction producing a carry.
let Defs = [CC] in {
  // Subtraction of a register.
  defm SLR : BinaryRRAndK<"sl", 0x1F, 0xB9FB, subc, GR32, GR32>;
  def SLGFR : BinaryRRE<"slgf", 0xB91B, null_frag, GR64, GR32>;
  defm SLGR : BinaryRREAndK<"slg", 0xB90B, 0xB9EB, subc, GR64, GR64>;

  // Subtraction of unsigned 32-bit immediates.  These don't match
  // subc because we prefer addc for constants.
  def SLFI  : BinaryRIL<"slfi",  0xC25, null_frag, GR32, uimm32>;
  def SLGFI : BinaryRIL<"slgfi", 0xC24, null_frag, GR64, imm64zx32>;

  // Subtraction of memory.
  defm SL   : BinaryRXPair<"sl", 0x5F, 0xE35F, subc, GR32, load, 4>;
  def  SLGF : BinaryRXY<"slgf", 0xE31B, subc, GR64, azextloadi32, 4>;
  def  SLG  : BinaryRXY<"slg",  0xE30B, subc, GR64, load, 8>;
}
defm : ZXB<subc, GR64, SLGFR>;

// Subtraction producing and using a carry.
let Defs = [CC], Uses = [CC] in {
  // Subtraction of a register.
  def SLBR  : BinaryRRE<"slb",  0xB999, sube, GR32, GR32>;
  def SLGBR : BinaryRRE<"slbg", 0xB989, sube, GR64, GR64>;

  // Subtraction of memory.
  def SLB  : BinaryRXY<"slb",  0xE399, sube, GR32, load, 4>;
  def SLBG : BinaryRXY<"slbg", 0xE389, sube, GR64, load, 8>;
}

//===----------------------------------------------------------------------===//
// AND
//===----------------------------------------------------------------------===//

let Defs = [CC] in {
  // ANDs of a register.
  let isCommutable = 1, CCValues = 0xC, CompareZeroCCMask = 0x8 in {
    defm NR : BinaryRRAndK<"n", 0x14, 0xB9F4, and, GR32, GR32>;
    defm NGR : BinaryRREAndK<"ng", 0xB980, 0xB9E4, and, GR64, GR64>;
  }

  let isConvertibleToThreeAddress = 1 in {
    // ANDs of a 16-bit immediate, leaving other bits unaffected.
    // The CC result only reflects the 16-bit field, not the full register.
    //
    // NIxMux expands to NI[LH]x, depending on the choice of register.
    def NILMux : BinaryRIPseudo<and, GRX32, imm32ll16c>,
                 Requires<[FeatureHighWord]>;
    def NIHMux : BinaryRIPseudo<and, GRX32, imm32lh16c>,
                 Requires<[FeatureHighWord]>;
    def NILL : BinaryRI<"nill", 0xA57, and, GR32, imm32ll16c>;
    def NILH : BinaryRI<"nilh", 0xA56, and, GR32, imm32lh16c>;
    def NIHL : BinaryRI<"nihl", 0xA55, and, GRH32, imm32ll16c>;
    def NIHH : BinaryRI<"nihh", 0xA54, and, GRH32, imm32lh16c>;
    def NILL64 : BinaryAliasRI<and, GR64, imm64ll16c>;
    def NILH64 : BinaryAliasRI<and, GR64, imm64lh16c>;
    def NIHL64 : BinaryAliasRI<and, GR64, imm64hl16c>;
    def NIHH64 : BinaryAliasRI<and, GR64, imm64hh16c>;

    // ANDs of a 32-bit immediate, leaving other bits unaffected.
    // The CC result only reflects the 32-bit field, which means we can
    // use it as a zero indicator for i32 operations but not otherwise.
    let CCValues = 0xC, CompareZeroCCMask = 0x8 in {
      // Expands to NILF or NIHF, depending on the choice of register.
      def NIFMux : BinaryRIPseudo<and, GRX32, uimm32>,
                   Requires<[FeatureHighWord]>;
      def NILF : BinaryRIL<"nilf", 0xC0B, and, GR32, uimm32>;
      def NIHF : BinaryRIL<"nihf", 0xC0A, and, GRH32, uimm32>;
    }
    def NILF64 : BinaryAliasRIL<and, GR64, imm64lf32c>;
    def NIHF64 : BinaryAliasRIL<and, GR64, imm64hf32c>;
  }

  // ANDs of memory.
  let CCValues = 0xC, CompareZeroCCMask = 0x8 in {
    defm N  : BinaryRXPair<"n", 0x54, 0xE354, and, GR32, load, 4>;
    def  NG : BinaryRXY<"ng", 0xE380, and, GR64, load, 8>; 
  }

  // AND to memory
  defm NI : BinarySIPair<"ni", 0x94, 0xEB54, null_frag, imm32zx8>;

  // Block AND.
  let mayLoad = 1, mayStore = 1 in
    defm NC : MemorySS<"nc", 0xD4, z_nc, z_nc_loop>;
}
defm : RMWIByte<and, bdaddr12pair, NI>;
defm : RMWIByte<and, bdaddr20pair, NIY>;

//===----------------------------------------------------------------------===//
// OR
//===----------------------------------------------------------------------===//

let Defs = [CC] in {
  // ORs of a register.
  let isCommutable = 1, CCValues = 0xC, CompareZeroCCMask = 0x8 in {
    defm OR : BinaryRRAndK<"o", 0x16, 0xB9F6, or, GR32, GR32>;
    defm OGR : BinaryRREAndK<"og", 0xB981, 0xB9E6, or, GR64, GR64>;
  }

  // ORs of a 16-bit immediate, leaving other bits unaffected.
  // The CC result only reflects the 16-bit field, not the full register.
  //
  // OIxMux expands to OI[LH]x, depending on the choice of register.
  def OILMux : BinaryRIPseudo<or, GRX32, imm32ll16>,
               Requires<[FeatureHighWord]>;
  def OIHMux : BinaryRIPseudo<or, GRX32, imm32lh16>,
               Requires<[FeatureHighWord]>;
  def OILL : BinaryRI<"oill", 0xA5B, or, GR32, imm32ll16>;
  def OILH : BinaryRI<"oilh", 0xA5A, or, GR32, imm32lh16>;
  def OIHL : BinaryRI<"oihl", 0xA59, or, GRH32, imm32ll16>;
  def OIHH : BinaryRI<"oihh", 0xA58, or, GRH32, imm32lh16>;
  def OILL64 : BinaryAliasRI<or, GR64, imm64ll16>;
  def OILH64 : BinaryAliasRI<or, GR64, imm64lh16>;
  def OIHL64 : BinaryAliasRI<or, GR64, imm64hl16>;
  def OIHH64 : BinaryAliasRI<or, GR64, imm64hh16>;

  // ORs of a 32-bit immediate, leaving other bits unaffected.
  // The CC result only reflects the 32-bit field, which means we can
  // use it as a zero indicator for i32 operations but not otherwise.
  let CCValues = 0xC, CompareZeroCCMask = 0x8 in {
    // Expands to OILF or OIHF, depending on the choice of register.
    def OIFMux : BinaryRIPseudo<or, GRX32, uimm32>,
                 Requires<[FeatureHighWord]>;
    def OILF : BinaryRIL<"oilf", 0xC0D, or, GR32, uimm32>;
    def OIHF : BinaryRIL<"oihf", 0xC0C, or, GRH32, uimm32>;
  }
  def OILF64 : BinaryAliasRIL<or, GR64, imm64lf32>;
  def OIHF64 : BinaryAliasRIL<or, GR64, imm64hf32>;

  // ORs of memory.
  let CCValues = 0xC, CompareZeroCCMask = 0x8 in {
    defm O  : BinaryRXPair<"o", 0x56, 0xE356, or, GR32, load, 4>;
    def  OG : BinaryRXY<"og", 0xE381, or, GR64, load, 8>;
  }

  // OR to memory
  defm OI : BinarySIPair<"oi", 0x96, 0xEB56, null_frag, imm32zx8>;

  // Block OR.
  let mayLoad = 1, mayStore = 1 in
    defm OC : MemorySS<"oc", 0xD6, z_oc, z_oc_loop>;
}
defm : RMWIByte<or, bdaddr12pair, OI>;
defm : RMWIByte<or, bdaddr20pair, OIY>;

//===----------------------------------------------------------------------===//
// XOR
//===----------------------------------------------------------------------===//

let Defs = [CC] in {
  // XORs of a register.
  let isCommutable = 1, CCValues = 0xC, CompareZeroCCMask = 0x8 in {
    defm XR : BinaryRRAndK<"x", 0x17, 0xB9F7, xor, GR32, GR32>;
    defm XGR : BinaryRREAndK<"xg", 0xB982, 0xB9E7, xor, GR64, GR64>;
  }

  // XORs of a 32-bit immediate, leaving other bits unaffected.
  // The CC result only reflects the 32-bit field, which means we can
  // use it as a zero indicator for i32 operations but not otherwise.
  let CCValues = 0xC, CompareZeroCCMask = 0x8 in {
    // Expands to XILF or XIHF, depending on the choice of register.
    def XIFMux : BinaryRIPseudo<xor, GRX32, uimm32>,
                 Requires<[FeatureHighWord]>;
    def XILF : BinaryRIL<"xilf", 0xC07, xor, GR32, uimm32>;
    def XIHF : BinaryRIL<"xihf", 0xC06, xor, GRH32, uimm32>;
  }
  def XILF64 : BinaryAliasRIL<xor, GR64, imm64lf32>;
  def XIHF64 : BinaryAliasRIL<xor, GR64, imm64hf32>;

  // XORs of memory.
  let CCValues = 0xC, CompareZeroCCMask = 0x8 in {
    defm X  : BinaryRXPair<"x",0x57, 0xE357, xor, GR32, load, 4>;
    def  XG : BinaryRXY<"xg", 0xE382, xor, GR64, load, 8>;
  }

  // XOR to memory
  defm XI : BinarySIPair<"xi", 0x97, 0xEB57, null_frag, imm32zx8>;

  // Block XOR.
  let mayLoad = 1, mayStore = 1 in
    defm XC : MemorySS<"xc", 0xD7, z_xc, z_xc_loop>;
}
defm : RMWIByte<xor, bdaddr12pair, XI>;
defm : RMWIByte<xor, bdaddr20pair, XIY>;

//===----------------------------------------------------------------------===//
// Multiplication
//===----------------------------------------------------------------------===//

// Multiplication of a register.
let isCommutable = 1 in {
  def MSR  : BinaryRRE<"ms",  0xB252, mul, GR32, GR32>;
  def MSGR : BinaryRRE<"msg", 0xB90C, mul, GR64, GR64>;
}
def MSGFR : BinaryRRE<"msgf", 0xB91C, null_frag, GR64, GR32>;
defm : SXB<mul, GR64, MSGFR>;

// Multiplication of a signed 16-bit immediate.
def MHI  : BinaryRI<"mhi",  0xA7C, mul, GR32, imm32sx16>;
def MGHI : BinaryRI<"mghi", 0xA7D, mul, GR64, imm64sx16>;

// Multiplication of a signed 32-bit immediate.
def MSFI  : BinaryRIL<"msfi",  0xC21, mul, GR32, simm32>;
def MSGFI : BinaryRIL<"msgfi", 0xC20, mul, GR64, imm64sx32>;

// Multiplication of memory.
defm MH   : BinaryRXPair<"mh", 0x4C, 0xE37C, mul, GR32, asextloadi16, 2>;
defm MS   : BinaryRXPair<"ms", 0x71, 0xE351, mul, GR32, load, 4>;
def  MSGF : BinaryRXY<"msgf", 0xE31C, mul, GR64, asextloadi32, 4>;
def  MSG  : BinaryRXY<"msg",  0xE30C, mul, GR64, load, 8>;

// Multiplication of a register, producing two results.
def MLGR : BinaryRRE<"mlg", 0xB986, z_umul_lohi64, GR128, GR64>;

// Multiplication of memory, producing two results.
def MLG : BinaryRXY<"mlg", 0xE386, z_umul_lohi64, GR128, load, 8>;

//===----------------------------------------------------------------------===//
// Division and remainder
//===----------------------------------------------------------------------===//

// Division and remainder, from registers.
def DSGFR : BinaryRRE<"dsgf", 0xB91D, z_sdivrem32, GR128, GR32>;
def DSGR  : BinaryRRE<"dsg",  0xB90D, z_sdivrem64, GR128, GR64>;
def DLR   : BinaryRRE<"dl",   0xB997, z_udivrem32, GR128, GR32>;
def DLGR  : BinaryRRE<"dlg",  0xB987, z_udivrem64, GR128, GR64>;

// Division and remainder, from memory.
def DSGF : BinaryRXY<"dsgf", 0xE31D, z_sdivrem32, GR128, load, 4>;
def DSG  : BinaryRXY<"dsg",  0xE30D, z_sdivrem64, GR128, load, 8>;
def DL   : BinaryRXY<"dl",   0xE397, z_udivrem32, GR128, load, 4>;
def DLG  : BinaryRXY<"dlg",  0xE387, z_udivrem64, GR128, load, 8>;

//===----------------------------------------------------------------------===//
// Shifts
//===----------------------------------------------------------------------===//

// Shift left.
let hasSideEffects = 0 in {
  defm SLL : BinaryRSAndK<"sll", 0x89, 0xEBDF, shl, GR32>;
  def SLLG : BinaryRSY<"sllg", 0xEB0D, shl, GR64>;
}

// Logical shift right.
let hasSideEffects = 0 in {
  defm SRL : BinaryRSAndK<"srl", 0x88, 0xEBDE, srl, GR32>;
  def SRLG : BinaryRSY<"srlg", 0xEB0C, srl, GR64>;
}

// Arithmetic shift right.
let Defs = [CC], CCValues = 0xE, CompareZeroCCMask = 0xE in {
  defm SRA : BinaryRSAndK<"sra", 0x8A, 0xEBDC, sra, GR32>;
  def SRAG : BinaryRSY<"srag", 0xEB0A, sra, GR64>;
}

// Rotate left.
let hasSideEffects = 0 in {
  def RLL  : BinaryRSY<"rll",  0xEB1D, rotl, GR32>;
  def RLLG : BinaryRSY<"rllg", 0xEB1C, rotl, GR64>;
}

// Rotate second operand left and inserted selected bits into first operand.
// These can act like 32-bit operands provided that the constant start and
// end bits (operands 2 and 3) are in the range [32, 64).
let Defs = [CC] in {
  let isCodeGenOnly = 1 in
    def RISBG32 : RotateSelectRIEf<"risbg", 0xEC55, GR32, GR32>;
  let CCValues = 0xE, CompareZeroCCMask = 0xE in
    def RISBG : RotateSelectRIEf<"risbg", 0xEC55, GR64, GR64>;
}

// On zEC12 we have a variant of RISBG that does not set CC.
let Predicates = [FeatureMiscellaneousExtensions] in
  def RISBGN : RotateSelectRIEf<"risbgn", 0xEC59, GR64, GR64>;

// Forms of RISBG that only affect one word of the destination register.
// They do not set CC.
let Predicates = [FeatureHighWord] in {
  def RISBMux : RotateSelectRIEfPseudo<GRX32, GRX32>;
  def RISBLL  : RotateSelectAliasRIEf<GR32,  GR32>;
  def RISBLH  : RotateSelectAliasRIEf<GR32,  GRH32>;
  def RISBHL  : RotateSelectAliasRIEf<GRH32, GR32>;
  def RISBHH  : RotateSelectAliasRIEf<GRH32, GRH32>;
  def RISBLG  : RotateSelectRIEf<"risblg", 0xEC51, GR32, GR64>;
  def RISBHG  : RotateSelectRIEf<"risbhg", 0xEC5D, GRH32, GR64>;
}

// Rotate second operand left and perform a logical operation with selected
// bits of the first operand.  The CC result only describes the selected bits,
// so isn't useful for a full comparison against zero.
let Defs = [CC] in {
  def RNSBG : RotateSelectRIEf<"rnsbg", 0xEC54, GR64, GR64>;
  def ROSBG : RotateSelectRIEf<"rosbg", 0xEC56, GR64, GR64>;
  def RXSBG : RotateSelectRIEf<"rxsbg", 0xEC57, GR64, GR64>;
}

//===----------------------------------------------------------------------===//
// Comparison
//===----------------------------------------------------------------------===//

// Signed comparisons.  We put these before the unsigned comparisons because
// some of the signed forms have COMPARE AND BRANCH equivalents whereas none
// of the unsigned forms do.
let Defs = [CC], CCValues = 0xE in {
  // Comparison with a register.
  def CR   : CompareRR <"c",   0x19,   z_scmp,    GR32, GR32>;
  def CGFR : CompareRRE<"cgf", 0xB930, null_frag, GR64, GR32>;
  def CGR  : CompareRRE<"cg",  0xB920, z_scmp,    GR64, GR64>;

  // Comparison with a signed 16-bit immediate.
  def CHI  : CompareRI<"chi",  0xA7E, z_scmp, GR32, imm32sx16>;
  def CGHI : CompareRI<"cghi", 0xA7F, z_scmp, GR64, imm64sx16>;

  // Comparison with a signed 32-bit immediate.  CFIMux expands to CFI or CIH,
  // depending on the choice of register.
  def CFIMux : CompareRIPseudo<z_scmp, GRX32, simm32>,
               Requires<[FeatureHighWord]>;
  def CFI  : CompareRIL<"cfi",  0xC2D, z_scmp, GR32, simm32>;
  def CIH  : CompareRIL<"cih",  0xCCD, z_scmp, GRH32, simm32>,
             Requires<[FeatureHighWord]>;
  def CGFI : CompareRIL<"cgfi", 0xC2C, z_scmp, GR64, imm64sx32>;

  // Comparison with memory.
  defm CH    : CompareRXPair<"ch", 0x49, 0xE379, z_scmp, GR32, asextloadi16, 2>;
  def  CMux  : CompareRXYPseudo<z_scmp, GRX32, load, 4>,
               Requires<[FeatureHighWord]>;
  defm C     : CompareRXPair<"c",  0x59, 0xE359, z_scmp, GR32, load, 4>;
  def  CHF   : CompareRXY<"chf", 0xE3CD, z_scmp, GRH32, load, 4>,
               Requires<[FeatureHighWord]>;
  def  CGH   : CompareRXY<"cgh", 0xE334, z_scmp, GR64, asextloadi16, 2>;
  def  CGF   : CompareRXY<"cgf", 0xE330, z_scmp, GR64, asextloadi32, 4>;
  def  CG    : CompareRXY<"cg",  0xE320, z_scmp, GR64, load, 8>;
  def  CHRL  : CompareRILPC<"chrl",  0xC65, z_scmp, GR32, aligned_asextloadi16>;
  def  CRL   : CompareRILPC<"crl",   0xC6D, z_scmp, GR32, aligned_load>;
  def  CGHRL : CompareRILPC<"cghrl", 0xC64, z_scmp, GR64, aligned_asextloadi16>;
  def  CGFRL : CompareRILPC<"cgfrl", 0xC6C, z_scmp, GR64, aligned_asextloadi32>;
  def  CGRL  : CompareRILPC<"cgrl",  0xC68, z_scmp, GR64, aligned_load>;

  // Comparison between memory and a signed 16-bit immediate.
  def CHHSI : CompareSIL<"chhsi", 0xE554, z_scmp, asextloadi16, imm32sx16>;
  def CHSI  : CompareSIL<"chsi",  0xE55C, z_scmp, load, imm32sx16>;
  def CGHSI : CompareSIL<"cghsi", 0xE558, z_scmp, load, imm64sx16>;
}
defm : SXB<z_scmp, GR64, CGFR>;

// Unsigned comparisons.
let Defs = [CC], CCValues = 0xE, IsLogical = 1 in {
  // Comparison with a register.
  def CLR   : CompareRR <"cl",   0x15,   z_ucmp,    GR32, GR32>;
  def CLGFR : CompareRRE<"clgf", 0xB931, null_frag, GR64, GR32>;
  def CLGR  : CompareRRE<"clg",  0xB921, z_ucmp,    GR64, GR64>;

  // Comparison with an unsigned 32-bit immediate.  CLFIMux expands to CLFI
  // or CLIH, depending on the choice of register.
  def CLFIMux : CompareRIPseudo<z_ucmp, GRX32, uimm32>,
                Requires<[FeatureHighWord]>;
  def CLFI  : CompareRIL<"clfi",  0xC2F, z_ucmp, GR32, uimm32>;
  def CLIH  : CompareRIL<"clih",  0xCCF, z_ucmp, GRH32, uimm32>,
              Requires<[FeatureHighWord]>;
  def CLGFI : CompareRIL<"clgfi", 0xC2E, z_ucmp, GR64, imm64zx32>;

  // Comparison with memory.
  def  CLMux  : CompareRXYPseudo<z_ucmp, GRX32, load, 4>,
                Requires<[FeatureHighWord]>;
  defm CL     : CompareRXPair<"cl", 0x55, 0xE355, z_ucmp, GR32, load, 4>;
  def  CLHF   : CompareRXY<"clhf", 0xE3CF, z_ucmp, GRH32, load, 4>,
                Requires<[FeatureHighWord]>;
  def  CLGF   : CompareRXY<"clgf", 0xE331, z_ucmp, GR64, azextloadi32, 4>;
  def  CLG    : CompareRXY<"clg",  0xE321, z_ucmp, GR64, load, 8>;
  def  CLHRL  : CompareRILPC<"clhrl",  0xC67, z_ucmp, GR32,
                             aligned_azextloadi16>;
  def  CLRL   : CompareRILPC<"clrl",   0xC6F, z_ucmp, GR32,
                             aligned_load>;
  def  CLGHRL : CompareRILPC<"clghrl", 0xC66, z_ucmp, GR64,
                             aligned_azextloadi16>;
  def  CLGFRL : CompareRILPC<"clgfrl", 0xC6E, z_ucmp, GR64,
                             aligned_azextloadi32>;
  def  CLGRL  : CompareRILPC<"clgrl",  0xC6A, z_ucmp, GR64,
                             aligned_load>;

  // Comparison between memory and an unsigned 8-bit immediate.
  defm CLI : CompareSIPair<"cli", 0x95, 0xEB55, z_ucmp, azextloadi8, imm32zx8>;

  // Comparison between memory and an unsigned 16-bit immediate.
  def CLHHSI : CompareSIL<"clhhsi", 0xE555, z_ucmp, azextloadi16, imm32zx16>;
  def CLFHSI : CompareSIL<"clfhsi", 0xE55D, z_ucmp, load, imm32zx16>;
  def CLGHSI : CompareSIL<"clghsi", 0xE559, z_ucmp, load, imm64zx16>;
}
defm : ZXB<z_ucmp, GR64, CLGFR>;

// Memory-to-memory comparison.
let mayLoad = 1, Defs = [CC] in
  defm CLC : MemorySS<"clc", 0xD5, z_clc, z_clc_loop>;

// String comparison.
let mayLoad = 1, Defs = [CC] in
  defm CLST : StringRRE<"clst", 0xB25D, z_strcmp>;

// Test under mask.
let Defs = [CC] in {
  // TMxMux expands to TM[LH]x, depending on the choice of register.
  def TMLMux : CompareRIPseudo<z_tm_reg, GRX32, imm32ll16>,
               Requires<[FeatureHighWord]>;
  def TMHMux : CompareRIPseudo<z_tm_reg, GRX32, imm32lh16>,
               Requires<[FeatureHighWord]>;
  def TMLL : CompareRI<"tmll", 0xA71, z_tm_reg, GR32, imm32ll16>;
  def TMLH : CompareRI<"tmlh", 0xA70, z_tm_reg, GR32, imm32lh16>;
  def TMHL : CompareRI<"tmhl", 0xA73, z_tm_reg, GRH32, imm32ll16>;
  def TMHH : CompareRI<"tmhh", 0xA72, z_tm_reg, GRH32, imm32lh16>;

  def TMLL64 : CompareAliasRI<z_tm_reg, GR64, imm64ll16>;
  def TMLH64 : CompareAliasRI<z_tm_reg, GR64, imm64lh16>;
  def TMHL64 : CompareAliasRI<z_tm_reg, GR64, imm64hl16>;
  def TMHH64 : CompareAliasRI<z_tm_reg, GR64, imm64hh16>;

  defm TM : CompareSIPair<"tm", 0x91, 0xEB51, z_tm_mem, anyextloadi8, imm32zx8>;
}

//===----------------------------------------------------------------------===//
// Prefetch
//===----------------------------------------------------------------------===//

def PFD : PrefetchRXY<"pfd", 0xE336, z_prefetch>;
def PFDRL : PrefetchRILPC<"pfdrl", 0xC62, z_prefetch>;

//===----------------------------------------------------------------------===//
// Atomic operations
//===----------------------------------------------------------------------===//

def Serialize : Alias<2, (outs), (ins), [(z_serialize)]>;

let Predicates = [FeatureInterlockedAccess1], Defs = [CC] in {
  def LAA   : LoadAndOpRSY<"laa",   0xEBF8, atomic_load_add_32, GR32>;
  def LAAG  : LoadAndOpRSY<"laag",  0xEBE8, atomic_load_add_64, GR64>;
  def LAAL  : LoadAndOpRSY<"laal",  0xEBFA, null_frag, GR32>;
  def LAALG : LoadAndOpRSY<"laalg", 0xEBEA, null_frag, GR64>;
  def LAN   : LoadAndOpRSY<"lan",   0xEBF4, atomic_load_and_32, GR32>;
  def LANG  : LoadAndOpRSY<"lang",  0xEBE4, atomic_load_and_64, GR64>;
  def LAO   : LoadAndOpRSY<"lao",   0xEBF6, atomic_load_or_32, GR32>;
  def LAOG  : LoadAndOpRSY<"laog",  0xEBE6, atomic_load_or_64, GR64>;
  def LAX   : LoadAndOpRSY<"lax",   0xEBF7, atomic_load_xor_32, GR32>;
  def LAXG  : LoadAndOpRSY<"laxg",  0xEBE7, atomic_load_xor_64, GR64>;
}

def ATOMIC_SWAPW   : AtomicLoadWBinaryReg<z_atomic_swapw>;
def ATOMIC_SWAP_32 : AtomicLoadBinaryReg32<atomic_swap_32>;
def ATOMIC_SWAP_64 : AtomicLoadBinaryReg64<atomic_swap_64>;

def ATOMIC_LOADW_AR  : AtomicLoadWBinaryReg<z_atomic_loadw_add>;
def ATOMIC_LOADW_AFI : AtomicLoadWBinaryImm<z_atomic_loadw_add, simm32>;
let Predicates = [FeatureNoInterlockedAccess1] in {
  def ATOMIC_LOAD_AR   : AtomicLoadBinaryReg32<atomic_load_add_32>;
  def ATOMIC_LOAD_AHI  : AtomicLoadBinaryImm32<atomic_load_add_32, imm32sx16>;
  def ATOMIC_LOAD_AFI  : AtomicLoadBinaryImm32<atomic_load_add_32, simm32>;
  def ATOMIC_LOAD_AGR  : AtomicLoadBinaryReg64<atomic_load_add_64>;
  def ATOMIC_LOAD_AGHI : AtomicLoadBinaryImm64<atomic_load_add_64, imm64sx16>;
  def ATOMIC_LOAD_AGFI : AtomicLoadBinaryImm64<atomic_load_add_64, imm64sx32>;
}

def ATOMIC_LOADW_SR : AtomicLoadWBinaryReg<z_atomic_loadw_sub>;
def ATOMIC_LOAD_SR  : AtomicLoadBinaryReg32<atomic_load_sub_32>;
def ATOMIC_LOAD_SGR : AtomicLoadBinaryReg64<atomic_load_sub_64>;

def ATOMIC_LOADW_NR   : AtomicLoadWBinaryReg<z_atomic_loadw_and>;
def ATOMIC_LOADW_NILH : AtomicLoadWBinaryImm<z_atomic_loadw_and, imm32lh16c>;
let Predicates = [FeatureNoInterlockedAccess1] in {
  def ATOMIC_LOAD_NR     : AtomicLoadBinaryReg32<atomic_load_and_32>;
  def ATOMIC_LOAD_NILL   : AtomicLoadBinaryImm32<atomic_load_and_32,
                                                 imm32ll16c>;
  def ATOMIC_LOAD_NILH   : AtomicLoadBinaryImm32<atomic_load_and_32,
                                                 imm32lh16c>;
  def ATOMIC_LOAD_NILF   : AtomicLoadBinaryImm32<atomic_load_and_32, uimm32>;
  def ATOMIC_LOAD_NGR    : AtomicLoadBinaryReg64<atomic_load_and_64>;
  def ATOMIC_LOAD_NILL64 : AtomicLoadBinaryImm64<atomic_load_and_64,
                                                 imm64ll16c>;
  def ATOMIC_LOAD_NILH64 : AtomicLoadBinaryImm64<atomic_load_and_64,
                                                 imm64lh16c>;
  def ATOMIC_LOAD_NIHL64 : AtomicLoadBinaryImm64<atomic_load_and_64,
                                                 imm64hl16c>;
  def ATOMIC_LOAD_NIHH64 : AtomicLoadBinaryImm64<atomic_load_and_64,
                                                 imm64hh16c>;
  def ATOMIC_LOAD_NILF64 : AtomicLoadBinaryImm64<atomic_load_and_64,
                                                 imm64lf32c>;
  def ATOMIC_LOAD_NIHF64 : AtomicLoadBinaryImm64<atomic_load_and_64,
                                                 imm64hf32c>;
}

def ATOMIC_LOADW_OR     : AtomicLoadWBinaryReg<z_atomic_loadw_or>;
def ATOMIC_LOADW_OILH   : AtomicLoadWBinaryImm<z_atomic_loadw_or, imm32lh16>;
let Predicates = [FeatureNoInterlockedAccess1] in {
  def ATOMIC_LOAD_OR     : AtomicLoadBinaryReg32<atomic_load_or_32>;
  def ATOMIC_LOAD_OILL   : AtomicLoadBinaryImm32<atomic_load_or_32, imm32ll16>;
  def ATOMIC_LOAD_OILH   : AtomicLoadBinaryImm32<atomic_load_or_32, imm32lh16>;
  def ATOMIC_LOAD_OILF   : AtomicLoadBinaryImm32<atomic_load_or_32, uimm32>;
  def ATOMIC_LOAD_OGR    : AtomicLoadBinaryReg64<atomic_load_or_64>;
  def ATOMIC_LOAD_OILL64 : AtomicLoadBinaryImm64<atomic_load_or_64, imm64ll16>;
  def ATOMIC_LOAD_OILH64 : AtomicLoadBinaryImm64<atomic_load_or_64, imm64lh16>;
  def ATOMIC_LOAD_OIHL64 : AtomicLoadBinaryImm64<atomic_load_or_64, imm64hl16>;
  def ATOMIC_LOAD_OIHH64 : AtomicLoadBinaryImm64<atomic_load_or_64, imm64hh16>;
  def ATOMIC_LOAD_OILF64 : AtomicLoadBinaryImm64<atomic_load_or_64, imm64lf32>;
  def ATOMIC_LOAD_OIHF64 : AtomicLoadBinaryImm64<atomic_load_or_64, imm64hf32>;
}

def ATOMIC_LOADW_XR     : AtomicLoadWBinaryReg<z_atomic_loadw_xor>;
def ATOMIC_LOADW_XILF   : AtomicLoadWBinaryImm<z_atomic_loadw_xor, uimm32>;
let Predicates = [FeatureNoInterlockedAccess1] in {
  def ATOMIC_LOAD_XR     : AtomicLoadBinaryReg32<atomic_load_xor_32>;
  def ATOMIC_LOAD_XILF   : AtomicLoadBinaryImm32<atomic_load_xor_32, uimm32>;
  def ATOMIC_LOAD_XGR    : AtomicLoadBinaryReg64<atomic_load_xor_64>;
  def ATOMIC_LOAD_XILF64 : AtomicLoadBinaryImm64<atomic_load_xor_64, imm64lf32>;
  def ATOMIC_LOAD_XIHF64 : AtomicLoadBinaryImm64<atomic_load_xor_64, imm64hf32>;
}

def ATOMIC_LOADW_NRi    : AtomicLoadWBinaryReg<z_atomic_loadw_nand>;
def ATOMIC_LOADW_NILHi  : AtomicLoadWBinaryImm<z_atomic_loadw_nand,
                                               imm32lh16c>;
def ATOMIC_LOAD_NRi     : AtomicLoadBinaryReg32<atomic_load_nand_32>;
def ATOMIC_LOAD_NILLi   : AtomicLoadBinaryImm32<atomic_load_nand_32,
                                                imm32ll16c>;
def ATOMIC_LOAD_NILHi   : AtomicLoadBinaryImm32<atomic_load_nand_32,
                                                imm32lh16c>;
def ATOMIC_LOAD_NILFi   : AtomicLoadBinaryImm32<atomic_load_nand_32, uimm32>;
def ATOMIC_LOAD_NGRi    : AtomicLoadBinaryReg64<atomic_load_nand_64>;
def ATOMIC_LOAD_NILL64i : AtomicLoadBinaryImm64<atomic_load_nand_64,
                                                imm64ll16c>;
def ATOMIC_LOAD_NILH64i : AtomicLoadBinaryImm64<atomic_load_nand_64,
                                                imm64lh16c>;
def ATOMIC_LOAD_NIHL64i : AtomicLoadBinaryImm64<atomic_load_nand_64,
                                                imm64hl16c>;
def ATOMIC_LOAD_NIHH64i : AtomicLoadBinaryImm64<atomic_load_nand_64,
                                                imm64hh16c>;
def ATOMIC_LOAD_NILF64i : AtomicLoadBinaryImm64<atomic_load_nand_64,
                                                imm64lf32c>;
def ATOMIC_LOAD_NIHF64i : AtomicLoadBinaryImm64<atomic_load_nand_64,
                                                imm64hf32c>;

def ATOMIC_LOADW_MIN    : AtomicLoadWBinaryReg<z_atomic_loadw_min>;
def ATOMIC_LOAD_MIN_32  : AtomicLoadBinaryReg32<atomic_load_min_32>;
def ATOMIC_LOAD_MIN_64  : AtomicLoadBinaryReg64<atomic_load_min_64>;

def ATOMIC_LOADW_MAX    : AtomicLoadWBinaryReg<z_atomic_loadw_max>;
def ATOMIC_LOAD_MAX_32  : AtomicLoadBinaryReg32<atomic_load_max_32>;
def ATOMIC_LOAD_MAX_64  : AtomicLoadBinaryReg64<atomic_load_max_64>;

def ATOMIC_LOADW_UMIN   : AtomicLoadWBinaryReg<z_atomic_loadw_umin>;
def ATOMIC_LOAD_UMIN_32 : AtomicLoadBinaryReg32<atomic_load_umin_32>;
def ATOMIC_LOAD_UMIN_64 : AtomicLoadBinaryReg64<atomic_load_umin_64>;

def ATOMIC_LOADW_UMAX   : AtomicLoadWBinaryReg<z_atomic_loadw_umax>;
def ATOMIC_LOAD_UMAX_32 : AtomicLoadBinaryReg32<atomic_load_umax_32>;
def ATOMIC_LOAD_UMAX_64 : AtomicLoadBinaryReg64<atomic_load_umax_64>;

def ATOMIC_CMP_SWAPW
  : Pseudo<(outs GR32:$dst), (ins bdaddr20only:$addr, GR32:$cmp, GR32:$swap,
                                  ADDR32:$bitshift, ADDR32:$negbitshift,
                                  uimm32:$bitsize),
           [(set GR32:$dst,
                 (z_atomic_cmp_swapw bdaddr20only:$addr, GR32:$cmp, GR32:$swap,
                                     ADDR32:$bitshift, ADDR32:$negbitshift,
                                     uimm32:$bitsize))]> {
  let Defs = [CC];
  let mayLoad = 1;
  let mayStore = 1;
  let usesCustomInserter = 1;
}

let Defs = [CC] in {
  defm CS  : CmpSwapRSPair<"cs", 0xBA, 0xEB14, atomic_cmp_swap_32, GR32>;
  def  CSG : CmpSwapRSY<"csg", 0xEB30, atomic_cmp_swap_64, GR64>;
}

//===----------------------------------------------------------------------===//
// Transactional execution
//===----------------------------------------------------------------------===//

let Predicates = [FeatureTransactionalExecution] in {
  // Transaction Begin
  let hasSideEffects = 1, mayStore = 1,
      usesCustomInserter = 1, Defs = [CC] in {
    def TBEGIN : InstSIL<0xE560,
                         (outs), (ins bdaddr12only:$BD1, imm32zx16:$I2),
                         "tbegin\t$BD1, $I2",
                         [(z_tbegin bdaddr12only:$BD1, imm32zx16:$I2)]>;
    def TBEGIN_nofloat : Pseudo<(outs), (ins bdaddr12only:$BD1, imm32zx16:$I2),
                                [(z_tbegin_nofloat bdaddr12only:$BD1,
                                                   imm32zx16:$I2)]>;
    def TBEGINC : InstSIL<0xE561,
                          (outs), (ins bdaddr12only:$BD1, imm32zx16:$I2),
                          "tbeginc\t$BD1, $I2",
                          [(int_s390_tbeginc bdaddr12only:$BD1,
                                             imm32zx16:$I2)]>;
  }

  // Transaction End
  let hasSideEffects = 1, Defs = [CC], BD2 = 0 in
    def TEND : InstS<0xB2F8, (outs), (ins), "tend", [(z_tend)]>;

  // Transaction Abort
  let hasSideEffects = 1, isTerminator = 1, isBarrier = 1 in
    def TABORT : InstS<0xB2FC, (outs), (ins bdaddr12only:$BD2),
                       "tabort\t$BD2",
                       [(int_s390_tabort bdaddr12only:$BD2)]>;

  // Nontransactional Store
  let hasSideEffects = 1 in
    def NTSTG : StoreRXY<"ntstg", 0xE325, int_s390_ntstg, GR64, 8>;

  // Extract Transaction Nesting Depth
  let hasSideEffects = 1 in
    def ETND : InherentRRE<"etnd", 0xB2EC, GR32, (int_s390_etnd)>;
}

//===----------------------------------------------------------------------===//
// Processor assist
//===----------------------------------------------------------------------===//

let Predicates = [FeatureProcessorAssist] in {
  let hasSideEffects = 1, R4 = 0 in
    def PPA : InstRRF<0xB2E8, (outs), (ins GR64:$R1, GR64:$R2, imm32zx4:$R3),
                      "ppa\t$R1, $R2, $R3", []>;
  def : Pat<(int_s390_ppa_txassist GR32:$src),
            (PPA (INSERT_SUBREG (i64 (IMPLICIT_DEF)), GR32:$src, subreg_l32),
                 0, 1)>;
}

//===----------------------------------------------------------------------===//
// Miscellaneous Instructions.
//===----------------------------------------------------------------------===//

// Extract CC into bits 29 and 28 of a register.
let Uses = [CC] in
  def IPM : InherentRRE<"ipm", 0xB222, GR32, (z_ipm)>;

// Read a 32-bit access register into a GR32.  As with all GR32 operations,
// the upper 32 bits of the enclosing GR64 remain unchanged, which is useful
// when a 64-bit address is stored in a pair of access registers.
def EAR : InstRRE<0xB24F, (outs GR32:$R1), (ins access_reg:$R2),
                  "ear\t$R1, $R2",
                  [(set GR32:$R1, (z_extract_access access_reg:$R2))]>;

// Find leftmost one, AKA count leading zeros.  The instruction actually
// returns a pair of GR64s, the first giving the number of leading zeros
// and the second giving a copy of the source with the leftmost one bit
// cleared.  We only use the first result here.
let Defs = [CC] in {
  def FLOGR : UnaryRRE<"flog", 0xB983, null_frag, GR128, GR64>;
}
def : Pat<(ctlz GR64:$src),
          (EXTRACT_SUBREG (FLOGR GR64:$src), subreg_h64)>;

// Population count.  Counts bits set per byte.
let Predicates = [FeaturePopulationCount], Defs = [CC] in {
  def POPCNT : InstRRE<0xB9E1, (outs GR64:$R1), (ins GR64:$R2),
                       "popcnt\t$R1, $R2",
                       [(set GR64:$R1, (z_popcnt GR64:$R2))]>;
}

// Use subregs to populate the "don't care" bits in a 32-bit to 64-bit anyext.
def : Pat<(i64 (anyext GR32:$src)),
          (INSERT_SUBREG (i64 (IMPLICIT_DEF)), GR32:$src, subreg_l32)>;

// Extend GR32s and GR64s to GR128s.
let usesCustomInserter = 1 in {
  def AEXT128_64 : Pseudo<(outs GR128:$dst), (ins GR64:$src), []>;
  def ZEXT128_32 : Pseudo<(outs GR128:$dst), (ins GR32:$src), []>;
  def ZEXT128_64 : Pseudo<(outs GR128:$dst), (ins GR64:$src), []>;
}

// Search a block of memory for a character.
let mayLoad = 1, Defs = [CC] in
  defm SRST : StringRRE<"srst", 0xb25e, z_search_string>;

// Other instructions for inline assembly
let hasSideEffects = 1, Defs = [CC], mayStore = 1 in
  def STCK : InstS<0xB205, (outs), (ins bdaddr12only:$BD2),
                       "stck\t$BD2",
                       []>;
let hasSideEffects = 1, Defs = [CC], mayStore = 1 in
  def STCKF : InstS<0xB27C, (outs), (ins bdaddr12only:$BD2),
                       "stckf\t$BD2",
                       []>;
let hasSideEffects = 1, Defs = [CC], mayStore = 1 in
  def STCKE : InstS<0xB278, (outs), (ins bdaddr12only:$BD2),
                       "stcke\t$BD2",
                       []>;
let hasSideEffects = 1, Defs = [CC], mayStore = 1 in
  def STFLE : InstS<0xB2B0, (outs), (ins bdaddr12only:$BD2),
                       "stfle\t$BD2",
                       []>;



//===----------------------------------------------------------------------===//
// Peepholes.
//===----------------------------------------------------------------------===//

// Use AL* for GR64 additions of unsigned 32-bit values.
defm : ZXB<add, GR64, ALGFR>;
def  : Pat<(add GR64:$src1, imm64zx32:$src2),
           (ALGFI GR64:$src1, imm64zx32:$src2)>;
def  : Pat<(add GR64:$src1, (azextloadi32 bdxaddr20only:$addr)),
           (ALGF GR64:$src1, bdxaddr20only:$addr)>;

// Use SL* for GR64 subtractions of unsigned 32-bit values.
defm : ZXB<sub, GR64, SLGFR>;
def  : Pat<(add GR64:$src1, imm64zx32n:$src2),
           (SLGFI GR64:$src1, imm64zx32n:$src2)>;
def  : Pat<(sub GR64:$src1, (azextloadi32 bdxaddr20only:$addr)),
           (SLGF GR64:$src1, bdxaddr20only:$addr)>;

// Optimize sign-extended 1/0 selects to -1/0 selects.  This is important
// for vector legalization.
def : Pat<(sra (shl (i32 (z_select_ccmask 1, 0, imm32zx4:$valid, imm32zx4:$cc)),
                         (i32 31)),
                    (i32 31)),
          (Select32 (LHI -1), (LHI 0), imm32zx4:$valid, imm32zx4:$cc)>;
def : Pat<(sra (shl (i64 (anyext (i32 (z_select_ccmask 1, 0, imm32zx4:$valid,
                                                       imm32zx4:$cc)))),
                    (i32 63)),
               (i32 63)),
          (Select64 (LGHI -1), (LGHI 0), imm32zx4:$valid, imm32zx4:$cc)>;

// Peepholes for turning scalar operations into block operations.
defm : BlockLoadStore<anyextloadi8, i32, MVCSequence, NCSequence, OCSequence,
                      XCSequence, 1>;
defm : BlockLoadStore<anyextloadi16, i32, MVCSequence, NCSequence, OCSequence,
                      XCSequence, 2>;
defm : BlockLoadStore<load, i32, MVCSequence, NCSequence, OCSequence,
                      XCSequence, 4>;
defm : BlockLoadStore<anyextloadi8, i64, MVCSequence, NCSequence,
                      OCSequence, XCSequence, 1>;
defm : BlockLoadStore<anyextloadi16, i64, MVCSequence, NCSequence, OCSequence,
                      XCSequence, 2>;
defm : BlockLoadStore<anyextloadi32, i64, MVCSequence, NCSequence, OCSequence,
                      XCSequence, 4>;
defm : BlockLoadStore<load, i64, MVCSequence, NCSequence, OCSequence,
                      XCSequence, 8>;
