//===-- SuperHInstrDSP.td - DSP SuperH Instruction Definition --*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------------===//
///
/// \file
/// This file describes the FPU extensions for the SuperH Instruction Set.
///
//===-------------------------------------------------------------------------===//

def SHSDT_FCmp       : SDTypeProfile<0, 3, [SDTCisSameAs<0, 1>, SDTCisVT<2, i8>]>;
def SHfcmp           : SDNode<"SHISD::FCMP", SHSDT_FCmp, [SDNPOutGlue]>;

let Predicates = [HasFPU] in {





//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//
//                          Data Transfer Instructions
//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//

// fmov FRm, FRn
let isMoveReg = 1, hasSideEffects = 0 in
def FMOV              : InstRmRn<0b1111000000001100,
                                (outs FR32:$Rn), (ins FR32:$Rm),
                                "fmov $Rm,$Rn",
                                []>;




//===--------------------------------------------------------------------------===//
// Load Instructions
//===--------------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 1 in {

// fmov.s @Rm, FRn                     
def FMOV_LOAD         : InstRmRn<0b1111000000001000,
                                (outs FR32:$Rn), (ins IGPR:$Rm),
                                "fmov.s $Rm,$Rn",
                                []>;

// fmov.s @Rm+, FRn                     
def FMOV_RESTORE      : InstRmRn<0b1111000000001001,
                                (outs FR32:$Rn), (ins IGPRInc:$Rm),
                                "fmov.s $Rm,$Rn",
                                []>;

// fmov.s @(R0,Rm), FRn                     
def FMOV_INDEX_LOAD   : InstRmRn<0b1111000000000110,
                                (outs FR32:$Rn), (ins IGPR:$Rm),
                                "fmov.s $Rm,$Rn",
                                []>;
}

//===--------------------------------------------------------------------------===//
// Store Instructions
//===--------------------------------------------------------------------------===//

let hasSideEffects = 0, mayStore = 1 in {

// fmov.s FRm, @Rn                     
def FMOV_STORE        : InstRmRn<0b1111000000001010,
                                (outs), (ins FR32:$Rm, IGPR:$Rn),
                                "fmov.s $Rm,$Rn",
                                []>;

// fmov.s FRm, @-Rn                     
def FMOV_SAVE         : InstRmRn<0b1111000000001011,
                                (outs), (ins FR32:$Rm, IGPRDec:$Rn),
                                "fmov.s $Rm,$Rn",
                                []>;

// fmov.s FRm, @(R0,Rn)                    
def FMOV_INDEX_STORE  : InstRmRn<0b1111000000000111,
                                (outs), (ins FR32:$Rm, IGPR:$Rn),
                                "fmov.s $Rm,$Rn",
                                []>;

}




//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//
//                  Floating-Point Single-Precision Instructions
//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//

// fldi0 FRn
def FLDI0               : InstRn<0b1111000010001101,
                                (outs FR32:$Rn), (ins),
                                "fldi0 $Rn",
                                []>;

// fldi1 FRn
def FLDI1               : InstRn<0b1111000010011101,
                                (outs FR32:$Rn), (ins),
                                "fldi1 $Rn",
                                []>;

// flds FRm,fpul
let Defs = [FPUL] in
def FLDS                : InstRm<0b1111000000011101,
                                (outs), (ins FR32:$Rm),
                                "flds $Rm,fpul",
                                []>;

// fsts fpul,FRn
let Uses = [FPUL] in
def FSTS                : InstRn<0b1111000000001101,
                                (outs FR32:$Rn), (ins),
                                "fsts fpul,$Rn",
                                []>;

// fabs FRn
let Constraints = "$src = $Rn" in
def FABS                : InstRn<0b1111000001011101,
                                (outs FR32:$Rn), (ins FR32:$src),
                                "fabs $Rn",
                                [(set f32:$Rn, (fabs f32:$src))]>;

// fneg FRn
let Constraints = "$src = $Rn" in
def FNEG                : InstRn<0b1111000001001101,
                                (outs FR32:$Rn), (ins FR32:$src),
                                "fneg $Rn",
                                [(set f32:$Rn, (fneg f32:$src))]>;

// fadd FRm,FRn
let Constraints = "$src = $Rn" in
def FADD              : InstRmRn<0b1111000000000000,
                                (outs FR32:$Rn), (ins FR32:$Rm, FR32:$src),
                                "fadd $Rm,$Rn",
                                [(set f32:$Rn, (fadd f32:$Rm, f32:$src))]>;

// fsub FRm,FRn
let Constraints = "$src = $Rn" in
def FSUB              : InstRmRn<0b1111000000000001,
                                (outs FR32:$Rn), (ins FR32:$Rm, FR32:$src),
                                "fsub $Rm,$Rn",
                                [(set f32:$Rn, (fsub f32:$Rm, f32:$src))]>;

// fmul FRm,FRn
let Constraints = "$src = $Rn" in
def FMUL              : InstRmRn<0b1111000000000010,
                                (outs FR32:$Rn), (ins FR32:$Rm, FR32:$src),
                                "fmul $Rm,$Rn",
                                [(set f32:$Rn, (fmul f32:$Rm, f32:$src))]>;

// fmac fr0,FRm,FRn
let Defs = [FR0] in
def FMAC              : InstRmRn<0b1111000000001110,
                                (outs FR32:$Rn), (ins FR32:$Rm),
                                "fmac fr0,$Rm,$Rn",
                                []>;

// fdiv FRm,FRn
let Constraints = "$src = $Rn" in
def FDIV              : InstRmRn<0b1111000000000011,
                                (outs FR32:$Rn), (ins FR32:$Rm, FR32:$src),
                                "fdiv $Rm,$Rn",
                                [(set f32:$Rn, (fdiv f32:$Rm, f32:$src))]>;

// fsqrt FRn
let Constraints = "$src = $Rn" in
def FSQRT               : InstRn<0b1111000001101101,
                                (outs FR32:$Rn), (ins FR32:$src),
                                "fsqrt $Rn",
                                [(set f32:$Rn, (fsqrt f32:$src))]>;

// fcmp/eq FRm,FRn
let isCompare = 1, Defs = [SR] in
def FCMP_EQ           : InstRmRn<0b1111000000000100,
                                (outs), (ins FR32:$Rm, FR32:$Rn),
                                "fcmp/eq $Rm,$Rn",
                                []>;

// fcmp/gt FRm,FRn
let isCompare = 1, Defs = [SR] in
def FCMP_GT           : InstRmRn<0b1111000000000101,
                                (outs), (ins FR32:$Rm, FR32:$Rn),
                                "fcmp/eq $Rm,$Rn",
                                []>;

// float fpul,FRn
let Uses = [FPUL] in
def FLOAT32             : InstRn<0b1111000000101101,
                                (outs FR32:$Rn), (ins),
                                "float fpul,$Rn",
                                []>;

// ftrc fpul,FRn
let Defs = [FPUL] in
def FTRC32             : InstRm<0b1111000000111101,
                                (outs), (ins FR32:$Rm),
                                "ftrc $Rm,fpul",
                                []>;

// fipr FVm,FVn
def FIPR            : InstFVmFVn<0b1111000011101101,
                                (outs VEC128:$Rn), (ins VEC128:$Rm),
                                "fipr $Rm,$Rn",
                                []>;

// fipr FVm,FVn
let Uses = [XMTRX] in
def FTRV               : InstFVn<0b1111000111111101,
                                (outs VEC128:$Rn), (ins),
                                "ftrv xmtrx,$Rn",
                                []>;

// fsrra Fn
let Predicates = [HasFSRRA] in 
def FSRRA               : InstRn<0b1111000001111101,
                                (outs), (ins FR32:$Rn),
                                "fsrra $Rn",
                                []>;

// fsca FPUL,DRn
let Defs = [FPUL], Predicates = [HasFSCA] in 
def FSCA               : InstDRn<0b1111000011111101,
                                (outs), (ins FR64:$Rn),
                                "fsca fpul,$Rn",
                                []>;




//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//
//                      Floating-Point Control Instructions
//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//

// lds Rm,fpscr
let Defs = [FPSCR] in
def LDSFPSCR            : InstRm<0b0100000001101010,
                                (outs), (ins GPR:$Rm),
                                "lds $Rm,fpscr",
                                []>;

// sts fpscr,Rn
let Uses = [FPSCR] in
def STSFPSCR            : InstRn<0b0000000001101010,
                                (outs GPR:$Rn), (ins),
                                "sts fpscr,$Rn",
                                []>;

// lds @Rm+,fpscr
let Defs = [FPSCR] in
def LDSMFPSCR           : InstRm<0b0100000001100110,
                                (outs), (ins IGPRInc:$Rm),
                                "lds $Rm,fpscr",
                                []>;

// sts fpscr,@-Rn
let Uses = [FPSCR] in
def STSMFPSCR           : InstRn<0b0100000001100010,
                                (outs IGPRDec:$Rn), (ins),
                                "sts fpscr,$Rn",
                                []>;

// lds Rm,fpul
let Defs = [FPUL] in
def LDSFPUL             : InstRm<0b0100000001011010,
                                (outs), (ins GPR:$Rm),
                                "lds $Rm,fpul",
                                []>;

// sts fpul,Rn
let Uses = [FPUL] in
def STSFPUL             : InstRn<0b0000000001011010,
                                (outs GPR:$Rn), (ins),
                                "sts fpul,$Rn",
                                []>;

// lds @Rm+,fpul
let Defs = [FPUL] in
def LDSMFPUL            : InstRm<0b0100000001010110,
                                (outs), (ins IGPRInc:$Rm),
                                "lds $Rm,fpul",
                                []>;

// sts fpul,@-Rn
let Uses = [FPUL] in
def STSMFPUL            : InstRn<0b0100000001010010,
                                (outs IGPRDec:$Rn), (ins),
                                "sts fpul,$Rn",
                                []>;

// frchg
let Defs = [FPSCR] in
def FRCHG                 : Inst<0b1111101111111101,
                                (outs), (ins),
                                "frchg",
                                []>;

// fschg
let Defs = [FPSCR] in
def FSCHG                 : Inst<0b1111001111111101,
                                (outs), (ins),
                                "fschg",
                                []>;


//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//
//                              Pseudo instructions
//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//

//===--------------------------------------------------------------------------===//
// Load/Store
//===--------------------------------------------------------------------------===//




def MOVF32LF  : SHLoadPseudo<(outs FR32:$Rn), (ins (memri $Rm, $disp):$src),
                            "! MOVF32LF $src, $Rn",
                            []>;

def MOVF32LP  : SHLoadPseudo<(outs FR32:$Rn), (ins (memri $Rm, $disp):$src),
                            "! MOVF32LP $src, $Rn",
                            []>;

def MOVF32SF  : SHStorePseudo<(outs), (ins FR32:$src, (memri $Rm, $disp):$dst),
                             "! MOVF32SF $src, $dst",
                             []>;

def MOVF32SP  : SHStorePseudo<(outs), (ins FR32:$src, (memri $Rm, $disp):$dst),
                             "! MOVF32SP $src, $dst",
                             []>;

def MOVIF32       : SHPseudo<(outs FR32:$dst), (ins f32imm:$src),
                             "! MOVIF32 $src, $dst",
                             [(set f32:$dst, anyimmf32:$src)]>;




//===--------------------------------------------------------------------------===//
// Conversion
//===--------------------------------------------------------------------------===//

def F32TOI32 : SHPseudo<(outs GPR:$dst), (ins FR32:$src),
                             "! F32TOI32 $src, $dst",
                             []>;

def I32TOF32 : SHPseudo<(outs FR32:$dst), (ins GPR:$src),
                             "! I32TOF32 $src, $dst",
                             []>;




//===--------------------------------------------------------------------------===//
// Selection
//===--------------------------------------------------------------------------===//

def SelectF32   : SHSelectPseudo<(outs FR32:$dst),
                                (ins FR32:$src, FR32:$src2, i8imm:$cc),
                                "! SelectF32 PSEUDO",
                                [(set f32:$dst,
                                 (SHselectcc f32:$src, f32:$src2, timm:$cc))]>;




//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//
//                           Non-instruction patterns
//===--------------------------------------------------------------------------===//
//===--------------------------------------------------------------------------===//

def : Pat<(i32 (fp_to_sint (f32  FR32:$src))),  (F32TOI32  FR32:$src)>;
def : Pat<(f32 (sint_to_fp (i32   GPR:$src))),  (I32TOF32   GPR:$src)>;

//===--------------------------------------------------------------------------===//
// Loads
//===--------------------------------------------------------------------------===//

def : Pat<(f32 (load          IGPR:$src)),    (FMOV_LOAD    IGPR:$src)>;
def : Pat<(f32 (load          addr:$src)),    (MOVF32LP     addr:$src)>;
def : Pat<(f32 (load        fraddr:$src)),    (MOVF32LF   fraddr:$src)>;
def : Pat<(f32 (extloadf32    IGPR:$src)),    (FMOV_LOAD    IGPR:$src)>;
def : Pat<(f32 (extloadf32    addr:$src)),    (MOVF32LP     addr:$src)>;
def : Pat<(f32 (extloadf32  fraddr:$src)),    (MOVF32LF   fraddr:$src)>;

//===--------------------------------------------------------------------------===//
// Stores
//===--------------------------------------------------------------------------===//
def : Pat<(store           f32:$src,    GPR:$dst),  (FMOV_STORE FR32:$src,     GPR:$dst)>;
def : Pat<(store           f32:$src,   addr:$dst),  (MOVF32SP   FR32:$src,    addr:$dst)>;
def : Pat<(store           f32:$src, fraddr:$dst),  (MOVF32SF   FR32:$src,  fraddr:$dst)>;
}