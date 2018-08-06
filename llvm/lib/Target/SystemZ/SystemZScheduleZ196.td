//=- SystemZScheduleZ196.td - SystemZ Scheduling Definitions ---*- tblgen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the machine model for Z196 to support instruction
// scheduling and other instruction cost heuristics.
//
//===----------------------------------------------------------------------===//

def Z196Model : SchedMachineModel {

    let UnsupportedFeatures = Arch9UnsupportedFeatures.List;
    
    let IssueWidth = 5;
    let MicroOpBufferSize = 40;     // Issue queues
    let LoadLatency = 1;            // Optimistic load latency.

    let PostRAScheduler = 1;

    // Extra cycles for a mispredicted branch.
    let MispredictPenalty = 16;
}

let SchedModel = Z196Model in  {

// These definitions could be put in a subtarget common include file,
// but it seems the include system in Tablegen currently rejects
// multiple includes of same file.
def : WriteRes<GroupAlone, []> {
  let NumMicroOps = 0;
  let BeginGroup  = 1;
  let EndGroup    = 1;
}
def : WriteRes<EndGroup, []> {
  let NumMicroOps = 0;
  let EndGroup    = 1;
}
def : WriteRes<Lat2, []> { let Latency = 2; let NumMicroOps = 0;}
def : WriteRes<Lat3, []> { let Latency = 3; let NumMicroOps = 0;}
def : WriteRes<Lat4, []> { let Latency = 4; let NumMicroOps = 0;}
def : WriteRes<Lat5, []> { let Latency = 5; let NumMicroOps = 0;}
def : WriteRes<Lat6, []> { let Latency = 6; let NumMicroOps = 0;}
def : WriteRes<Lat7, []> { let Latency = 7; let NumMicroOps = 0;}
def : WriteRes<Lat8, []> { let Latency = 8; let NumMicroOps = 0;}
def : WriteRes<Lat9, []> { let Latency = 9; let NumMicroOps = 0;}
def : WriteRes<Lat10, []> { let Latency = 10; let NumMicroOps = 0;}
def : WriteRes<Lat11, []> { let Latency = 11; let NumMicroOps = 0;}
def : WriteRes<Lat12, []> { let Latency = 12; let NumMicroOps = 0;}
def : WriteRes<Lat15, []> { let Latency = 15; let NumMicroOps = 0;}
def : WriteRes<Lat20, []> { let Latency = 20; let NumMicroOps = 0;}
def : WriteRes<Lat30, []> { let Latency = 30; let NumMicroOps = 0;}

// Execution units.
def Z196_FXUnit : ProcResource<2>;
def Z196_LSUnit : ProcResource<2>;
def Z196_FPUnit : ProcResource<1>;
def Z196_DFUnit : ProcResource<1>;

// Subtarget specific definitions of scheduling resources.
def : WriteRes<FXU,       [Z196_FXUnit]> { let Latency = 1; }
def : WriteRes<LSU,       [Z196_LSUnit]> { let Latency = 4; }
def : WriteRes<LSU_lat1,  [Z196_LSUnit]> { let Latency = 1; }
def : WriteRes<FPU,       [Z196_FPUnit]> { let Latency = 8; }
def : WriteRes<FPU2,      [Z196_FPUnit, Z196_FPUnit]> { let Latency = 9; }
def : WriteRes<DFU,       [Z196_DFUnit]> { let Latency = 2; }
def : WriteRes<DFU2,      [Z196_DFUnit, Z196_DFUnit]> { let Latency = 3; }

// -------------------------- INSTRUCTIONS ---------------------------------- //

// InstRW constructs have been used in order to preserve the
// readability of the InstrInfo files.

// For each instruction, as matched by a regexp, provide a list of
// resources that it needs. These will be combined into a SchedClass.

//===----------------------------------------------------------------------===//
// Stack allocation
//===----------------------------------------------------------------------===//

def : InstRW<[FXU], (instregex "ADJDYNALLOC$")>; // Pseudo -> LA / LAY

//===----------------------------------------------------------------------===//
// Branch instructions
//===----------------------------------------------------------------------===//

// Branch
def : InstRW<[LSU, EndGroup], (instregex "(Call)?BRC(L)?(Asm.*)?$")>;
def : InstRW<[LSU, EndGroup], (instregex "(Call)?J(G)?(Asm.*)?$")>;
def : InstRW<[LSU, EndGroup], (instregex "(Call)?BC(R)?(Asm.*)?$")>;
def : InstRW<[LSU, EndGroup], (instregex "(Call)?B(R)?(Asm.*)?$")>;
def : InstRW<[FXU, LSU, Lat5, GroupAlone], (instregex "BRCT(G|H)?$")>;
def : InstRW<[FXU, LSU, Lat5, GroupAlone], (instregex "BCT(G)?(R)?$")>;
def : InstRW<[FXU, FXU, FXU, LSU, Lat7, GroupAlone],
             (instregex "B(R)?X(H|L).*$")>;

// Compare and branch
def : InstRW<[FXU, LSU, Lat5, GroupAlone],
             (instregex "C(L)?(G)?(I|R)J(Asm.*)?$")>;
def : InstRW<[FXU, LSU, Lat5, GroupAlone],
             (instregex "C(L)?(G)?(I|R)B(Call|Return|Asm.*)?$")>;

//===----------------------------------------------------------------------===//
// Trap instructions
//===----------------------------------------------------------------------===//

// Trap
def : InstRW<[LSU, EndGroup], (instregex "(Cond)?Trap$")>;

// Compare and trap
def : InstRW<[FXU], (instregex "C(G)?(I|R)T(Asm.*)?$")>;
def : InstRW<[FXU], (instregex "CL(G)?RT(Asm.*)?$")>;
def : InstRW<[FXU], (instregex "CL(F|G)IT(Asm.*)?$")>;

//===----------------------------------------------------------------------===//
// Call and return instructions
//===----------------------------------------------------------------------===//

// Call
def : InstRW<[LSU, FXU, FXU, Lat6, GroupAlone], (instregex "(Call)?BRAS$")>;
def : InstRW<[LSU, FXU, FXU, Lat6, GroupAlone], (instregex "(Call)?BRASL$")>;
def : InstRW<[FXU, FXU, LSU, Lat6, GroupAlone], (instregex "(Call)?BAS(R)?$")>;
def : InstRW<[LSU, FXU, FXU, Lat6, GroupAlone], (instregex "TLS_(G|L)DCALL$")>;

// Return
def : InstRW<[LSU_lat1, EndGroup], (instregex "Return$")>;
def : InstRW<[LSU_lat1, EndGroup], (instregex "CondReturn$")>;

//===----------------------------------------------------------------------===//
// Move instructions
//===----------------------------------------------------------------------===//

// Moves
def : InstRW<[FXU, LSU, Lat5], (instregex "MV(G|H)?HI$")>;
def : InstRW<[FXU, LSU, Lat5], (instregex "MVI(Y)?$")>;

// Move character
def : InstRW<[LSU, LSU, LSU, FXU, Lat8, GroupAlone], (instregex "MVC$")>;
def : InstRW<[LSU, Lat30, GroupAlone], (instregex "MVCL(E|U)?$")>;

// Pseudo -> reg move
def : InstRW<[FXU], (instregex "COPY(_TO_REGCLASS)?$")>;
def : InstRW<[FXU], (instregex "EXTRACT_SUBREG$")>;
def : InstRW<[FXU], (instregex "INSERT_SUBREG$")>;
def : InstRW<[FXU], (instregex "REG_SEQUENCE$")>;
def : InstRW<[FXU], (instregex "SUBREG_TO_REG$")>;

// Loads
def : InstRW<[LSU], (instregex "L(Y|FH|RL|Mux)?$")>;
def : InstRW<[LSU], (instregex "LG(RL)?$")>;
def : InstRW<[LSU], (instregex "L128$")>;

def : InstRW<[FXU], (instregex "LLIH(F|H|L)$")>;
def : InstRW<[FXU], (instregex "LLIL(F|H|L)$")>;

def : InstRW<[FXU], (instregex "LG(F|H)I$")>;
def : InstRW<[FXU], (instregex "LHI(Mux)?$")>;
def : InstRW<[FXU], (instregex "LR(Mux)?$")>;

// Load and test
def : InstRW<[FXU, LSU, Lat5], (instregex "LT(G)?$")>;
def : InstRW<[FXU], (instregex "LT(G)?R$")>;

// Stores
def : InstRW<[FXU, LSU, Lat5], (instregex "STG(RL)?$")>;
def : InstRW<[FXU, LSU, Lat5], (instregex "ST128$")>;
def : InstRW<[FXU, LSU, Lat5], (instregex "ST(Y|FH|RL|Mux)?$")>;

// String moves.
def : InstRW<[LSU, Lat30, GroupAlone], (instregex "MVST$")>;

//===----------------------------------------------------------------------===//
// Conditional move instructions
//===----------------------------------------------------------------------===//

def : InstRW<[FXU, Lat2, EndGroup], (instregex "LOC(G)?R(Asm.*)?$")>;
def : InstRW<[FXU, LSU, Lat6, EndGroup], (instregex "LOC(G)?(Asm.*)?$")>;
def : InstRW<[FXU, LSU, Lat5, EndGroup], (instregex "STOC(G)?(Asm.*)?$")>;

//===----------------------------------------------------------------------===//
// Sign extensions
//===----------------------------------------------------------------------===//
def : InstRW<[FXU], (instregex "L(B|H|G)R$")>;
def : InstRW<[FXU], (instregex "LG(B|H|F)R$")>;

def : InstRW<[FXU, LSU, Lat5], (instregex "LTGF$")>;
def : InstRW<[FXU], (instregex "LTGFR$")>;

def : InstRW<[FXU, LSU, Lat5], (instregex "LB(H|Mux)?$")>;
def : InstRW<[FXU, LSU, Lat5], (instregex "LH(Y)?$")>;
def : InstRW<[FXU, LSU, Lat5], (instregex "LH(H|Mux|RL)$")>;
def : InstRW<[FXU, LSU, Lat5], (instregex "LG(B|H|F)$")>;
def : InstRW<[FXU, LSU, Lat5], (instregex "LG(H|F)RL$")>;

//===----------------------------------------------------------------------===//
// Zero extensions
//===----------------------------------------------------------------------===//

def : InstRW<[FXU], (instregex "LLCR(Mux)?$")>;
def : InstRW<[FXU], (instregex "LLHR(Mux)?$")>;
def : InstRW<[FXU], (instregex "LLG(C|F|H|T)R$")>;
def : InstRW<[LSU], (instregex "LLC(Mux)?$")>;
def : InstRW<[LSU], (instregex "LLH(Mux)?$")>;
def : InstRW<[FXU, LSU, Lat5], (instregex "LL(C|H)H$")>;
def : InstRW<[LSU], (instregex "LLHRL$")>;
def : InstRW<[LSU], (instregex "LLG(C|F|H|T|FRL|HRL)$")>;

//===----------------------------------------------------------------------===//
// Truncations
//===----------------------------------------------------------------------===//

def : InstRW<[FXU, LSU, Lat5], (instregex "STC(H|Y|Mux)?$")>;
def : InstRW<[FXU, LSU, Lat5], (instregex "STH(H|Y|RL|Mux)?$")>;
def : InstRW<[FXU, LSU, Lat5], (instregex "STCM(H|Y)?$")>;

//===----------------------------------------------------------------------===//
// Multi-register moves
//===----------------------------------------------------------------------===//

// Load multiple (estimated average of 5 ops)
def : InstRW<[LSU, LSU, LSU, LSU, LSU, Lat10, GroupAlone],
             (instregex "LM(H|Y|G)?$")>;

// Load multiple disjoint
def : InstRW<[LSU, Lat30, GroupAlone], (instregex "LMD$")>;

// Store multiple (estimated average of 3 ops)
def : InstRW<[LSU, LSU, FXU, FXU, FXU, Lat10, GroupAlone],
             (instregex "STM(H|Y|G)?$")>;

//===----------------------------------------------------------------------===//
// Byte swaps
//===----------------------------------------------------------------------===//

def : InstRW<[FXU], (instregex "LRV(G)?R$")>;
def : InstRW<[FXU, LSU, Lat5], (instregex "LRV(G|H)?$")>;
def : InstRW<[FXU, LSU, Lat5], (instregex "STRV(G|H)?$")>;
def : InstRW<[LSU, Lat30, GroupAlone], (instregex "MVCIN$")>;

//===----------------------------------------------------------------------===//
// Load address instructions
//===----------------------------------------------------------------------===//

def : InstRW<[FXU], (instregex "LA(Y|RL)?$")>;

// Load the Global Offset Table address
def : InstRW<[FXU], (instregex "GOT$")>;

//===----------------------------------------------------------------------===//
// Absolute and Negation
//===----------------------------------------------------------------------===//

def : InstRW<[FXU, Lat2], (instregex "LP(G)?R$")>;
def : InstRW<[FXU, FXU, Lat3, GroupAlone], (instregex "L(N|P)GFR$")>;
def : InstRW<[FXU, Lat2], (instregex "LN(R|GR)$")>;
def : InstRW<[FXU], (instregex "LC(R|GR)$")>;
def : InstRW<[FXU, FXU, Lat2, GroupAlone], (instregex "LCGFR$")>;

//===----------------------------------------------------------------------===//
// Insertion
//===----------------------------------------------------------------------===//

def : InstRW<[FXU, LSU, Lat5], (instregex "IC(Y)?$")>;
def : InstRW<[FXU, LSU, Lat5], (instregex "IC32(Y)?$")>;
def : InstRW<[FXU, LSU, Lat5], (instregex "ICM(H|Y)?$")>;
def : InstRW<[FXU], (instregex "II(F|H|L)Mux$")>;
def : InstRW<[FXU], (instregex "IIHF(64)?$")>;
def : InstRW<[FXU], (instregex "IIHH(64)?$")>;
def : InstRW<[FXU], (instregex "IIHL(64)?$")>;
def : InstRW<[FXU], (instregex "IILF(64)?$")>;
def : InstRW<[FXU], (instregex "IILH(64)?$")>;
def : InstRW<[FXU], (instregex "IILL(64)?$")>;

//===----------------------------------------------------------------------===//
// Addition
//===----------------------------------------------------------------------===//

def : InstRW<[FXU, LSU, Lat5], (instregex "A(L)?(Y|SI)?$")>;
def : InstRW<[FXU, FXU, LSU, Lat6, GroupAlone], (instregex "AH(Y)?$")>;
def : InstRW<[FXU], (instregex "AIH$")>;
def : InstRW<[FXU], (instregex "AFI(Mux)?$")>;
def : InstRW<[FXU], (instregex "AGFI$")>;
def : InstRW<[FXU], (instregex "AGHI(K)?$")>;
def : InstRW<[FXU], (instregex "AGR(K)?$")>;
def : InstRW<[FXU], (instregex "AHI(K)?$")>;
def : InstRW<[FXU], (instregex "AHIMux(K)?$")>;
def : InstRW<[FXU], (instregex "AL(FI|HSIK)$")>;
def : InstRW<[FXU, LSU, Lat5], (instregex "ALGF$")>;
def : InstRW<[FXU], (instregex "ALGHSIK$")>;
def : InstRW<[FXU], (instregex "ALGF(I|R)$")>;
def : InstRW<[FXU], (instregex "ALGR(K)?$")>;
def : InstRW<[FXU], (instregex "ALR(K)?$")>;
def : InstRW<[FXU], (instregex "AR(K)?$")>;
def : InstRW<[FXU], (instregex "A(L)?HHHR$")>;
def : InstRW<[FXU, FXU, Lat3, GroupAlone], (instregex "A(L)?HHLR$")>;
def : InstRW<[FXU], (instregex "ALSIH(N)?$")>;
def : InstRW<[FXU, LSU, Lat5], (instregex "A(L)?G(SI)?$")>;

// Logical addition with carry
def : InstRW<[FXU, LSU, Lat7, GroupAlone], (instregex "ALC(G)?$")>;
def : InstRW<[FXU, Lat3, GroupAlone], (instregex "ALC(G)?R$")>;

// Add with sign extension (32 -> 64)
def : InstRW<[FXU, FXU, LSU, Lat6, GroupAlone], (instregex "AGF$")>;
def : InstRW<[FXU, FXU, Lat2, GroupAlone], (instregex "AGFR$")>;

//===----------------------------------------------------------------------===//
// Subtraction
//===----------------------------------------------------------------------===//

def : InstRW<[FXU, LSU, Lat5], (instregex "S(G|Y)?$")>;
def : InstRW<[FXU, FXU, LSU, Lat6, GroupAlone], (instregex "SH(Y)?$")>;
def : InstRW<[FXU], (instregex "SGR(K)?$")>;
def : InstRW<[FXU], (instregex "SLFI$")>;
def : InstRW<[FXU, LSU, Lat5], (instregex "SL(G|GF|Y)?$")>;
def : InstRW<[FXU], (instregex "SLGF(I|R)$")>;
def : InstRW<[FXU], (instregex "SLGR(K)?$")>;
def : InstRW<[FXU], (instregex "SLR(K)?$")>;
def : InstRW<[FXU], (instregex "SR(K)?$")>;
def : InstRW<[FXU], (instregex "S(L)?HHHR$")>;
def : InstRW<[FXU, FXU, Lat3, GroupAlone], (instregex "S(L)?HHLR$")>;

// Subtraction with borrow
def : InstRW<[FXU, LSU, Lat7, GroupAlone], (instregex "SLB(G)?$")>;
def : InstRW<[FXU, Lat3, GroupAlone], (instregex "SLB(G)?R$")>;

// Subtraction with sign extension (32 -> 64)
def : InstRW<[FXU, FXU, LSU, Lat6, GroupAlone], (instregex "SGF$")>;
def : InstRW<[FXU, FXU, Lat2, GroupAlone], (instregex "SGFR$")>;

//===----------------------------------------------------------------------===//
// AND
//===----------------------------------------------------------------------===//

def : InstRW<[FXU, LSU, Lat5], (instregex "N(G|Y)?$")>;
def : InstRW<[FXU], (instregex "NGR(K)?$")>;
def : InstRW<[FXU], (instregex "NI(FMux|HMux|LMux)$")>;
def : InstRW<[FXU, LSU, Lat5], (instregex "NI(Y)?$")>;
def : InstRW<[FXU], (instregex "NIHF(64)?$")>;
def : InstRW<[FXU], (instregex "NIHH(64)?$")>;
def : InstRW<[FXU], (instregex "NIHL(64)?$")>;
def : InstRW<[FXU], (instregex "NILF(64)?$")>;
def : InstRW<[FXU], (instregex "NILH(64)?$")>;
def : InstRW<[FXU], (instregex "NILL(64)?$")>;
def : InstRW<[FXU], (instregex "NR(K)?$")>;
def : InstRW<[LSU, LSU, FXU, Lat9, GroupAlone], (instregex "NC$")>;

//===----------------------------------------------------------------------===//
// OR
//===----------------------------------------------------------------------===//

def : InstRW<[FXU, LSU, Lat5], (instregex "O(G|Y)?$")>;
def : InstRW<[FXU], (instregex "OGR(K)?$")>;
def : InstRW<[FXU, LSU, Lat5], (instregex "OI(Y)?$")>;
def : InstRW<[FXU], (instregex "OI(FMux|HMux|LMux)$")>;
def : InstRW<[FXU], (instregex "OIHF(64)?$")>;
def : InstRW<[FXU], (instregex "OIHH(64)?$")>;
def : InstRW<[FXU], (instregex "OIHL(64)?$")>;
def : InstRW<[FXU], (instregex "OILF(64)?$")>;
def : InstRW<[FXU], (instregex "OILH(64)?$")>;
def : InstRW<[FXU], (instregex "OILL(64)?$")>;
def : InstRW<[FXU], (instregex "OR(K)?$")>;
def : InstRW<[LSU, LSU, FXU, Lat9, GroupAlone], (instregex "OC$")>;

//===----------------------------------------------------------------------===//
// XOR
//===----------------------------------------------------------------------===//

def : InstRW<[FXU, LSU, Lat5], (instregex "X(G|Y)?$")>;
def : InstRW<[FXU, LSU, Lat5], (instregex "XI(Y)?$")>;
def : InstRW<[FXU], (instregex "XIFMux$")>;
def : InstRW<[FXU], (instregex "XGR(K)?$")>;
def : InstRW<[FXU], (instregex "XIHF(64)?$")>;
def : InstRW<[FXU], (instregex "XILF(64)?$")>;
def : InstRW<[FXU], (instregex "XR(K)?$")>;
def : InstRW<[LSU, LSU, FXU, Lat9, GroupAlone], (instregex "XC$")>;

//===----------------------------------------------------------------------===//
// Multiplication
//===----------------------------------------------------------------------===//

def : InstRW<[FXU, LSU, Lat10], (instregex "MS(GF|Y)?$")>;
def : InstRW<[FXU, Lat6], (instregex "MS(R|FI)$")>;
def : InstRW<[FXU, LSU, Lat12], (instregex "MSG$")>;
def : InstRW<[FXU, Lat8], (instregex "MSGR$")>;
def : InstRW<[FXU, Lat6], (instregex "MSGF(I|R)$")>;
def : InstRW<[FXU, FXU, LSU, Lat15, GroupAlone], (instregex "MLG$")>;
def : InstRW<[FXU, FXU, Lat9, GroupAlone], (instregex "MLGR$")>;
def : InstRW<[FXU, Lat5], (instregex "MGHI$")>;
def : InstRW<[FXU, Lat5], (instregex "MHI$")>;
def : InstRW<[FXU, LSU, Lat9], (instregex "MH(Y)?$")>;
def : InstRW<[FXU, FXU, Lat7, GroupAlone], (instregex "M(L)?R$")>;
def : InstRW<[FXU, FXU, LSU, Lat7, GroupAlone], (instregex "M(FY|L)?$")>;

//===----------------------------------------------------------------------===//
// Division and remainder
//===----------------------------------------------------------------------===//

def : InstRW<[FPU2, FPU2, FXU, FXU, FXU, FXU, FXU, Lat30, GroupAlone],
              (instregex "DR$")>;
def : InstRW<[FPU2, FPU2, LSU, FXU, FXU, FXU, FXU, Lat30, GroupAlone],
              (instregex "D$")>;
def : InstRW<[FPU2, FPU2, FXU, FXU, FXU, FXU, Lat30, GroupAlone],
              (instregex "DSG(F)?R$")>;
def : InstRW<[FPU2, FPU2, LSU, FXU, FXU, FXU, Lat30, GroupAlone],
              (instregex "DSG(F)?$")>;
def : InstRW<[FPU2, FPU2, FXU, FXU, FXU, FXU, FXU, Lat30, GroupAlone],
              (instregex "DL(G)?R$")>;
def : InstRW<[FPU2, FPU2, LSU, FXU, FXU, FXU, FXU, Lat30, GroupAlone],
              (instregex "DL(G)?$")>;

//===----------------------------------------------------------------------===//
// Shifts
//===----------------------------------------------------------------------===//

def : InstRW<[FXU], (instregex "SLL(G|K)?$")>;
def : InstRW<[FXU], (instregex "SRL(G|K)?$")>;
def : InstRW<[FXU], (instregex "SRA(G|K)?$")>;
def : InstRW<[FXU, Lat2], (instregex "SLA(G|K)?$")>;
def : InstRW<[FXU, FXU, FXU, FXU, LSU, Lat8, GroupAlone],
             (instregex "S(L|R)D(A|L)$")>;

// Rotate
def : InstRW<[FXU, LSU, Lat6], (instregex "RLL(G)?$")>;

// Rotate and insert
def : InstRW<[FXU], (instregex "RISBG(32)?$")>;
def : InstRW<[FXU], (instregex "RISBH(G|H|L)$")>;
def : InstRW<[FXU], (instregex "RISBL(G|H|L)$")>;
def : InstRW<[FXU], (instregex "RISBMux$")>;

// Rotate and Select
def : InstRW<[FXU, FXU, Lat3, GroupAlone], (instregex "R(N|O|X)SBG$")>;

//===----------------------------------------------------------------------===//
// Comparison
//===----------------------------------------------------------------------===//

def : InstRW<[FXU, LSU, Lat5], (instregex "C(G|Y|Mux|RL)?$")>;
def : InstRW<[FXU], (instregex "C(F|H)I(Mux)?$")>;
def : InstRW<[FXU], (instregex "CG(F|H)I$")>;
def : InstRW<[FXU, LSU, Lat5], (instregex "CG(HSI|RL)$")>;
def : InstRW<[FXU], (instregex "C(G)?R$")>;
def : InstRW<[FXU], (instregex "CIH$")>;
def : InstRW<[FXU, LSU, Lat5], (instregex "CH(F|SI)$")>;
def : InstRW<[FXU, LSU, Lat5], (instregex "CL(Y|Mux|FHSI)?$")>;
def : InstRW<[FXU], (instregex "CLFI(Mux)?$")>;
def : InstRW<[FXU, LSU, Lat5], (instregex "CLG(HRL|HSI)?$")>;
def : InstRW<[FXU, LSU, Lat5], (instregex "CLGF(RL)?$")>;
def : InstRW<[FXU], (instregex "CLGF(I|R)$")>;
def : InstRW<[FXU], (instregex "CLGR$")>;
def : InstRW<[FXU, LSU, Lat5], (instregex "CLGRL$")>;
def : InstRW<[FXU, LSU, Lat5], (instregex "CLH(F|RL|HSI)$")>;
def : InstRW<[FXU], (instregex "CLIH$")>;
def : InstRW<[FXU, LSU, Lat5], (instregex "CLI(Y)?$")>;
def : InstRW<[FXU], (instregex "CLR$")>;
def : InstRW<[FXU, LSU, Lat5], (instregex "CLRL$")>;
def : InstRW<[FXU], (instregex "C(L)?HHR$")>;
def : InstRW<[FXU, FXU, Lat3, GroupAlone], (instregex "C(L)?HLR$")>;

// Compare halfword
def : InstRW<[FXU, LSU, FXU, Lat6, GroupAlone], (instregex "CH(Y|RL)?$")>;
def : InstRW<[FXU, LSU, FXU, Lat6, GroupAlone], (instregex "CGH(RL)?$")>;
def : InstRW<[FXU, LSU, FXU, Lat6, GroupAlone], (instregex "CHHSI$")>;

// Compare with sign extension (32 -> 64)
def : InstRW<[FXU, FXU, LSU, Lat6, Lat2, GroupAlone], (instregex "CGF(RL)?$")>;
def : InstRW<[FXU, FXU, Lat2, GroupAlone], (instregex "CGFR$")>;

// Compare logical character
def : InstRW<[LSU, LSU, FXU, Lat9, GroupAlone], (instregex "CLC$")>;
def : InstRW<[LSU, Lat30, GroupAlone], (instregex "CLCL(E|U)?$")>;
def : InstRW<[LSU, Lat30, GroupAlone], (instregex "CLST$")>;

// Test under mask
def : InstRW<[FXU, LSU, Lat5], (instregex "TM(Y)?$")>;
def : InstRW<[FXU], (instregex "TM(H|L)Mux$")>;
def : InstRW<[FXU], (instregex "TMHH(64)?$")>;
def : InstRW<[FXU], (instregex "TMHL(64)?$")>;
def : InstRW<[FXU], (instregex "TMLH(64)?$")>;
def : InstRW<[FXU], (instregex "TMLL(64)?$")>;

// Compare logical characters under mask
def : InstRW<[FXU, FXU, LSU, Lat5, GroupAlone], (instregex "CLM(H|Y)?$")>;

//===----------------------------------------------------------------------===//
// Prefetch
//===----------------------------------------------------------------------===//

def : InstRW<[LSU, GroupAlone], (instregex "PFD(RL)?$")>;

//===----------------------------------------------------------------------===//
// Atomic operations
//===----------------------------------------------------------------------===//

def : InstRW<[LSU, EndGroup], (instregex "Serialize$")>;

def : InstRW<[FXU, LSU, Lat5], (instregex "LAA(G)?$")>;
def : InstRW<[FXU, LSU, Lat5], (instregex "LAAL(G)?$")>;
def : InstRW<[FXU, LSU, Lat5], (instregex "LAN(G)?$")>;
def : InstRW<[FXU, LSU, Lat5], (instregex "LAO(G)?$")>;
def : InstRW<[FXU, LSU, Lat5], (instregex "LAX(G)?$")>;

// Test and set
def : InstRW<[FXU, LSU, Lat5, EndGroup], (instregex "TS$")>;

// Compare and swap
def : InstRW<[FXU, LSU, FXU, Lat6, GroupAlone], (instregex "CS(G|Y)?$")>;

// Compare double and swap
def : InstRW<[FXU, FXU, FXU, FXU, FXU, LSU, Lat10, GroupAlone],
             (instregex "CDS(Y)?$")>;
def : InstRW<[FXU, FXU, FXU, FXU, FXU, FXU, LSU, LSU, Lat12, GroupAlone],
             (instregex "CDSG$")>;

// Compare and swap and store
def : InstRW<[FXU, LSU, Lat30], (instregex "CSST$")>;

// Perform locked operation
def : InstRW<[LSU, Lat30, GroupAlone], (instregex "PLO$")>;

// Load/store pair from/to quadword
def : InstRW<[LSU, LSU, Lat5, GroupAlone], (instregex "LPQ$")>;
def : InstRW<[FXU, FXU, LSU, LSU, Lat6, GroupAlone], (instregex "STPQ$")>;

// Load pair disjoint
def : InstRW<[LSU, LSU, Lat5, GroupAlone], (instregex "LPD(G)?$")>;

//===----------------------------------------------------------------------===//
// Translate and convert
//===----------------------------------------------------------------------===//

def : InstRW<[LSU, Lat30, GroupAlone], (instregex "TR$")>;
def : InstRW<[LSU, Lat30, GroupAlone], (instregex "TRT$")>;
def : InstRW<[FXU, LSU, Lat30], (instregex "TRTR$")>;
def : InstRW<[FXU, Lat30], (instregex "TRE$")>;
def : InstRW<[FXU, Lat30], (instregex "TRT(R)?E(Opt)?$")>;
def : InstRW<[LSU, Lat30], (instregex "TR(T|O)(T|O)(Opt)?$")>;
def : InstRW<[FXU, Lat30], (instregex "CU(12|14|21|24|41|42)(Opt)?$")>;
def : InstRW<[FXU, Lat30], (instregex "(CUUTF|CUTFU)(Opt)?$")>;

//===----------------------------------------------------------------------===//
// Message-security assist
//===----------------------------------------------------------------------===//

def : InstRW<[FXU, Lat30], (instregex "KM(C|F|O|CTR)?$")>;
def : InstRW<[FXU, Lat30], (instregex "(KIMD|KLMD|KMAC|PCC)$")>;

//===----------------------------------------------------------------------===//
// Decimal arithmetic
//===----------------------------------------------------------------------===//

def : InstRW<[FXU, DFU2, LSU, LSU, Lat30, GroupAlone], (instregex "CVBG$")>;
def : InstRW<[FXU, DFU, LSU, Lat30, GroupAlone], (instregex "CVB(Y)?$")>;
def : InstRW<[FXU, FXU, FXU, DFU2, DFU2, LSU, Lat30, GroupAlone],
             (instregex "CVDG$")>;
def : InstRW<[FXU, FXU, DFU, LSU, Lat30, GroupAlone], (instregex "CVD(Y)?$")>;
def : InstRW<[LSU, Lat10, GroupAlone], (instregex "MVO$")>;
def : InstRW<[LSU, Lat30, GroupAlone], (instregex "MV(N|Z)$")>;
def : InstRW<[LSU, Lat30, GroupAlone], (instregex "(PACK|PKA|PKU)$")>;
def : InstRW<[LSU, Lat30, GroupAlone], (instregex "UNPK$")>;
def : InstRW<[LSU, Lat12, GroupAlone], (instregex "UNPK(A|U)$")>;

def : InstRW<[FXU, DFU2, DFU2, LSU, LSU, Lat15, GroupAlone],
             (instregex "(A|S|ZA)P$")>;
def : InstRW<[FXU, DFU2, DFU2, LSU, LSU, Lat30, GroupAlone],
             (instregex "(M|D)P$")>;
def : InstRW<[FXU, FXU, DFU2, DFU2, LSU, LSU, LSU, Lat15, GroupAlone],
             (instregex "SRP$")>;
def : InstRW<[DFU2, DFU2, LSU, LSU, Lat11, GroupAlone], (instregex "CP$")>;
def : InstRW<[DFU2, LSU, LSU, GroupAlone], (instregex "TP$")>;
def : InstRW<[LSU, Lat30, GroupAlone], (instregex "ED(MK)?$")>;

//===----------------------------------------------------------------------===//
// Access registers
//===----------------------------------------------------------------------===//

// Extract/set/copy access register
def : InstRW<[LSU], (instregex "(EAR|SAR|CPYA)$")>;

// Load address extended
def : InstRW<[LSU, FXU, Lat5, GroupAlone], (instregex "LAE(Y)?$")>;

// Load/store access multiple (not modeled precisely)
def : InstRW<[LSU, Lat30, GroupAlone], (instregex "(L|ST)AM(Y)?$")>;

//===----------------------------------------------------------------------===//
// Program mask and addressing mode
//===----------------------------------------------------------------------===//

// Insert Program Mask
def : InstRW<[FXU, Lat3, EndGroup], (instregex "IPM$")>;

// Set Program Mask
def : InstRW<[LSU, EndGroup], (instregex "SPM$")>;

// Branch and link
def : InstRW<[FXU, FXU, LSU, Lat8, GroupAlone], (instregex "BAL(R)?$")>;

// Test addressing mode
def : InstRW<[FXU], (instregex "TAM$")>;

// Set addressing mode
def : InstRW<[LSU, EndGroup], (instregex "SAM(24|31|64)$")>;

// Branch (and save) and set mode.
def : InstRW<[FXU, LSU, Lat5, GroupAlone], (instregex "BSM$")>;
def : InstRW<[FXU, FXU, LSU, Lat6, GroupAlone], (instregex "BASSM$")>;

//===----------------------------------------------------------------------===//
// Miscellaneous Instructions.
//===----------------------------------------------------------------------===//

// Find leftmost one
def : InstRW<[FXU, FXU, Lat7, GroupAlone], (instregex "FLOGR$")>;

// Population count
def : InstRW<[FXU, Lat3], (instregex "POPCNT$")>;

// String instructions
def : InstRW<[FXU, LSU, Lat30], (instregex "SRST$")>;
def : InstRW<[FXU, Lat30], (instregex "SRSTU$")>;
def : InstRW<[LSU, Lat30, GroupAlone], (instregex "CUSE$")>;

// Various complex instructions
def : InstRW<[LSU, Lat30], (instregex "CFC$")>;
def : InstRW<[FXU, LSU, Lat30], (instregex "UPT$")>;
def : InstRW<[LSU, Lat30], (instregex "CKSM$")>;
def : InstRW<[FXU, Lat30], (instregex "CMPSC$")>;

// Execute
def : InstRW<[LSU, GroupAlone], (instregex "EX(RL)?$")>;

//===----------------------------------------------------------------------===//
// .insn directive instructions
//===----------------------------------------------------------------------===//

// An "empty" sched-class will be assigned instead of the "invalid sched-class".
// getNumDecoderSlots() will then return 1 instead of 0.
def : InstRW<[], (instregex "Insn.*")>;


// ----------------------------- Floating point ----------------------------- //

//===----------------------------------------------------------------------===//
// FP: Move instructions
//===----------------------------------------------------------------------===//

// Load zero
def : InstRW<[FXU], (instregex "LZ(DR|ER)$")>;
def : InstRW<[FXU, FXU, Lat2, GroupAlone], (instregex "LZXR$")>;

// Load
def : InstRW<[FXU], (instregex "LER$")>;
def : InstRW<[FXU], (instregex "LD(R|R32|GR)$")>;
def : InstRW<[FXU, Lat3], (instregex "LGDR$")>;
def : InstRW<[FXU, FXU, Lat2, GroupAlone], (instregex "LXR$")>;

// Load and Test
def : InstRW<[FPU], (instregex "LT(D|E)BR$")>;
def : InstRW<[FPU], (instregex "LTEBRCompare$")>;
def : InstRW<[FPU], (instregex "LTDBRCompare$")>;
def : InstRW<[FPU2, FPU2, Lat9, GroupAlone], (instregex "LTXBR$")>;
def : InstRW<[FPU2, FPU2, Lat9, GroupAlone], (instregex "LTXBRCompare$")>;

// Copy sign
def : InstRW<[FXU, FXU, Lat5, GroupAlone], (instregex "CPSDRd(d|s)$")>;
def : InstRW<[FXU, FXU, Lat5, GroupAlone], (instregex "CPSDRs(d|s)$")>;

//===----------------------------------------------------------------------===//
// FP: Load instructions
//===----------------------------------------------------------------------===//

def : InstRW<[LSU], (instregex "LE(Y)?$")>;
def : InstRW<[LSU], (instregex "LD(Y|E32)?$")>;
def : InstRW<[LSU], (instregex "LX$")>;

//===----------------------------------------------------------------------===//
// FP: Store instructions
//===----------------------------------------------------------------------===//

def : InstRW<[FXU, LSU, Lat7], (instregex "STD(Y)?$")>;
def : InstRW<[FXU, LSU, Lat7], (instregex "STE(Y)?$")>;
def : InstRW<[FXU, LSU, Lat5], (instregex "STX$")>;

//===----------------------------------------------------------------------===//
// FP: Conversion instructions
//===----------------------------------------------------------------------===//

// Load rounded
def : InstRW<[FPU], (instregex "LEDBR(A)?$")>;
def : InstRW<[FPU, FPU, Lat20], (instregex "LEXBR(A)?$")>;
def : InstRW<[FPU, FPU, Lat20], (instregex "LDXBR(A)?$")>;

// Load lengthened
def : InstRW<[FPU, LSU, Lat12], (instregex "LDEB$")>;
def : InstRW<[FPU], (instregex "LDEBR$")>;
def : InstRW<[FPU2, FPU2, LSU, Lat15, GroupAlone], (instregex "LX(D|E)B$")>;
def : InstRW<[FPU2, FPU2, Lat10, GroupAlone], (instregex "LX(D|E)BR$")>;

// Convert from fixed / logical
def : InstRW<[FXU, FPU, Lat9, GroupAlone], (instregex "CE(F|G)BR(A)?$")>;
def : InstRW<[FXU, FPU, Lat9, GroupAlone], (instregex "CD(F|G)BR(A)?$")>;
def : InstRW<[FXU, FPU2, FPU2, Lat11, GroupAlone], (instregex "CX(F|G)BR(A)?$")>;
def : InstRW<[FXU, FPU, Lat9, GroupAlone], (instregex "CEL(F|G)BR$")>;
def : InstRW<[FXU, FPU, Lat9, GroupAlone], (instregex "CDL(F|G)BR$")>;
def : InstRW<[FXU, FPU2, FPU2, Lat11, GroupAlone], (instregex "CXL(F|G)BR$")>;

// Convert to fixed / logical
def : InstRW<[FXU, FPU, Lat12, GroupAlone], (instregex "CF(E|D)BR(A)?$")>;
def : InstRW<[FXU, FPU, Lat12, GroupAlone], (instregex "CG(E|D)BR(A)?$")>;
def : InstRW<[FXU, FPU, FPU, Lat20, GroupAlone], (instregex "C(F|G)XBR(A)?$")>;
def : InstRW<[FXU, FPU, Lat11, GroupAlone], (instregex "CLF(E|D)BR$")>;
def : InstRW<[FXU, FPU, Lat11, GroupAlone], (instregex "CLG(E|D)BR$")>;
def : InstRW<[FXU, FPU, FPU, Lat20, GroupAlone], (instregex "CL(F|G)XBR$")>;

//===----------------------------------------------------------------------===//
// FP: Unary arithmetic
//===----------------------------------------------------------------------===//

// Load Complement / Negative / Positive
def : InstRW<[FPU], (instregex "L(C|N|P)DBR$")>;
def : InstRW<[FPU], (instregex "L(C|N|P)EBR$")>;
def : InstRW<[FXU], (instregex "LCDFR(_32)?$")>;
def : InstRW<[FXU], (instregex "LNDFR(_32)?$")>;
def : InstRW<[FXU], (instregex "LPDFR(_32)?$")>;
def : InstRW<[FPU2, FPU2, Lat9, GroupAlone], (instregex "L(C|N|P)XBR$")>;

// Square root
def : InstRW<[FPU, LSU, Lat30], (instregex "SQ(E|D)B$")>;
def : InstRW<[FPU, Lat30], (instregex "SQ(E|D)BR$")>;
def : InstRW<[FPU2, FPU2, Lat30, GroupAlone], (instregex "SQXBR$")>;

// Load FP integer
def : InstRW<[FPU], (instregex "FIEBR(A)?$")>;
def : InstRW<[FPU], (instregex "FIDBR(A)?$")>;
def : InstRW<[FPU2, FPU2, Lat15, GroupAlone], (instregex "FIXBR(A)?$")>;

//===----------------------------------------------------------------------===//
// FP: Binary arithmetic
//===----------------------------------------------------------------------===//

// Addition
def : InstRW<[FPU, LSU, Lat12], (instregex "A(E|D)B$")>;
def : InstRW<[FPU], (instregex "A(E|D)BR$")>;
def : InstRW<[FPU2, FPU2, Lat20, GroupAlone], (instregex "AXBR$")>;

// Subtraction
def : InstRW<[FPU, LSU, Lat12], (instregex "S(E|D)B$")>;
def : InstRW<[FPU], (instregex "S(E|D)BR$")>;
def : InstRW<[FPU2, FPU2, Lat20, GroupAlone], (instregex "SXBR$")>;

// Multiply
def : InstRW<[FPU, LSU, Lat12], (instregex "M(D|DE|EE)B$")>;
def : InstRW<[FPU], (instregex "M(D|DE|EE)BR$")>;
def : InstRW<[FPU2, FPU2, LSU, Lat15, GroupAlone], (instregex "MXDB$")>;
def : InstRW<[FPU2, FPU2, Lat10, GroupAlone], (instregex "MXDBR$")>;
def : InstRW<[FPU2, FPU2, Lat30, GroupAlone], (instregex "MXBR$")>;

// Multiply and add / subtract
def : InstRW<[FPU, FPU, LSU, Lat12, GroupAlone], (instregex "M(A|S)EB$")>;
def : InstRW<[FPU, GroupAlone], (instregex "M(A|S)EBR$")>;
def : InstRW<[FPU, FPU, LSU, Lat12, GroupAlone], (instregex "M(A|S)DB$")>;
def : InstRW<[FPU, GroupAlone], (instregex "M(A|S)DBR$")>;

// Division
def : InstRW<[FPU, LSU, Lat30], (instregex "D(E|D)B$")>;
def : InstRW<[FPU, Lat30], (instregex "D(E|D)BR$")>;
def : InstRW<[FPU2, FPU2, Lat30, GroupAlone], (instregex "DXBR$")>;

// Divide to integer
def : InstRW<[FPU, Lat30], (instregex "DI(E|D)BR$")>;

//===----------------------------------------------------------------------===//
// FP: Comparisons
//===----------------------------------------------------------------------===//

// Compare
def : InstRW<[FPU, LSU, Lat12], (instregex "(K|C)(E|D)B$")>;
def : InstRW<[FPU], (instregex "(K|C)(E|D)BR$")>;
def : InstRW<[FPU, FPU, Lat30], (instregex "(K|C)XBR$")>;

// Test Data Class
def : InstRW<[FPU, LSU, Lat15], (instregex "TC(E|D)B$")>;
def : InstRW<[FPU2, FPU2, LSU, Lat15, GroupAlone], (instregex "TCXB$")>;

//===----------------------------------------------------------------------===//
// FP: Floating-point control register instructions
//===----------------------------------------------------------------------===//

def : InstRW<[FXU, LSU, Lat4, GroupAlone], (instregex "EFPC$")>;
def : InstRW<[LSU, Lat3, GroupAlone], (instregex "SFPC$")>;
def : InstRW<[LSU, LSU, Lat6, GroupAlone], (instregex "LFPC$")>;
def : InstRW<[FXU, LSU, Lat3, GroupAlone], (instregex "STFPC$")>;
def : InstRW<[FXU, Lat30], (instregex "SFASR$")>;
def : InstRW<[FXU, LSU, Lat30], (instregex "LFAS$")>;
def : InstRW<[FXU, Lat2, GroupAlone], (instregex "SRNM(B|T)?$")>;


// --------------------- Hexadecimal floating point ------------------------- //

//===----------------------------------------------------------------------===//
// HFP: Move instructions
//===----------------------------------------------------------------------===//

// Load and Test
def : InstRW<[FPU], (instregex "LT(D|E)R$")>;
def : InstRW<[FPU2, FPU2, Lat9, GroupAlone], (instregex "LTXR$")>;

//===----------------------------------------------------------------------===//
// HFP: Conversion instructions
//===----------------------------------------------------------------------===//

// Load rounded
def : InstRW<[FPU], (instregex "(LEDR|LRER)$")>;
def : InstRW<[FPU], (instregex "LEXR$")>;
def : InstRW<[FPU], (instregex "(LDXR|LRDR)$")>;

// Load lengthened
def : InstRW<[LSU], (instregex "LDE$")>;
def : InstRW<[FXU], (instregex "LDER$")>;
def : InstRW<[FPU2, FPU2, LSU, Lat15, GroupAlone], (instregex "LX(D|E)$")>;
def : InstRW<[FPU2, FPU2, Lat10, GroupAlone], (instregex "LX(D|E)R$")>;

// Convert from fixed
def : InstRW<[FXU, FPU, Lat9, GroupAlone], (instregex "CE(F|G)R$")>;
def : InstRW<[FXU, FPU, Lat9, GroupAlone], (instregex "CD(F|G)R$")>;
def : InstRW<[FXU, FPU2, FPU2, Lat11, GroupAlone], (instregex "CX(F|G)R$")>;

// Convert to fixed
def : InstRW<[FXU, FPU, Lat12, GroupAlone], (instregex "CF(E|D)R$")>;
def : InstRW<[FXU, FPU, Lat12, GroupAlone], (instregex "CG(E|D)R$")>;
def : InstRW<[FXU, FPU, FPU, Lat20, GroupAlone], (instregex "C(F|G)XR$")>;

// Convert BFP to HFP / HFP to BFP.
def : InstRW<[FPU], (instregex "THD(E)?R$")>;
def : InstRW<[FPU], (instregex "TB(E)?DR$")>;

//===----------------------------------------------------------------------===//
// HFP: Unary arithmetic
//===----------------------------------------------------------------------===//

// Load Complement / Negative / Positive
def : InstRW<[FPU], (instregex "L(C|N|P)DR$")>;
def : InstRW<[FPU], (instregex "L(C|N|P)ER$")>;
def : InstRW<[FPU2, FPU2, Lat9, GroupAlone], (instregex "L(C|N|P)XR$")>;

// Halve
def : InstRW<[FPU], (instregex "H(E|D)R$")>;

// Square root
def : InstRW<[FPU, LSU, Lat30], (instregex "SQ(E|D)$")>;
def : InstRW<[FPU, Lat30], (instregex "SQ(E|D)R$")>;
def : InstRW<[FPU2, FPU2, Lat30, GroupAlone], (instregex "SQXR$")>;

// Load FP integer
def : InstRW<[FPU], (instregex "FIER$")>;
def : InstRW<[FPU], (instregex "FIDR$")>;
def : InstRW<[FPU2, FPU2, Lat15, GroupAlone], (instregex "FIXR$")>;

//===----------------------------------------------------------------------===//
// HFP: Binary arithmetic
//===----------------------------------------------------------------------===//

// Addition
def : InstRW<[FPU, LSU, Lat12], (instregex "A(E|D|U|W)$")>;
def : InstRW<[FPU], (instregex "A(E|D|U|W)R$")>;
def : InstRW<[FPU2, FPU2, Lat20, GroupAlone], (instregex "AXR$")>;

// Subtraction
def : InstRW<[FPU, LSU, Lat12], (instregex "S(E|D|U|W)$")>;
def : InstRW<[FPU], (instregex "S(E|D|U|W)R$")>;
def : InstRW<[FPU2, FPU2, Lat20, GroupAlone], (instregex "SXR$")>;

// Multiply
def : InstRW<[FPU, LSU, Lat12], (instregex "M(D|DE|E|EE)$")>;
def : InstRW<[FPU], (instregex "M(D|DE|E|EE)R$")>;
def : InstRW<[FPU2, FPU2, LSU, Lat15, GroupAlone], (instregex "MXD$")>;
def : InstRW<[FPU2, FPU2, Lat10, GroupAlone], (instregex "MXDR$")>;
def : InstRW<[FPU2, FPU2, Lat30, GroupAlone], (instregex "MXR$")>;
def : InstRW<[FPU2, FPU2, LSU, Lat15, GroupAlone], (instregex "MY$")>;
def : InstRW<[FPU, FPU, LSU, Lat15, GroupAlone], (instregex "MY(H|L)$")>;
def : InstRW<[FPU2, FPU2,  Lat10, GroupAlone], (instregex "MYR$")>;
def : InstRW<[FPU, Lat10, GroupAlone], (instregex "MY(H|L)R$")>;

// Multiply and add / subtract
def : InstRW<[FPU, FPU, LSU, Lat12, GroupAlone], (instregex "M(A|S)E$")>;
def : InstRW<[FPU, GroupAlone], (instregex "M(A|S)ER$")>;
def : InstRW<[FPU, FPU, LSU, Lat12, GroupAlone], (instregex "M(A|S)D$")>;
def : InstRW<[FPU, GroupAlone], (instregex "M(A|S)DR$")>;
def : InstRW<[FPU2, FPU2, LSU, GroupAlone], (instregex "MAY$")>;
def : InstRW<[FPU2, FPU2, GroupAlone], (instregex "MAYR$")>;
def : InstRW<[FPU, FPU, LSU, Lat12, GroupAlone], (instregex "MAY(H|L)$")>;
def : InstRW<[FPU, GroupAlone], (instregex "MAY(H|L)R$")>;

// Division
def : InstRW<[FPU, LSU, Lat30], (instregex "D(E|D)$")>;
def : InstRW<[FPU, Lat30], (instregex "D(E|D)R$")>;
def : InstRW<[FPU2, FPU2, Lat30, GroupAlone], (instregex "DXR$")>;

//===----------------------------------------------------------------------===//
// HFP: Comparisons
//===----------------------------------------------------------------------===//

// Compare
def : InstRW<[FPU, LSU, Lat12], (instregex "C(E|D)$")>;
def : InstRW<[FPU], (instregex "C(E|D)R$")>;
def : InstRW<[FPU, FPU, Lat15], (instregex "CXR$")>;


// ------------------------ Decimal floating point -------------------------- //

//===----------------------------------------------------------------------===//
// DFP: Move instructions
//===----------------------------------------------------------------------===//

// Load and Test
def : InstRW<[DFU, Lat20], (instregex "LTDTR$")>;
def : InstRW<[DFU2, DFU2, Lat20, GroupAlone], (instregex "LTXTR$")>;

//===----------------------------------------------------------------------===//
// DFP: Conversion instructions
//===----------------------------------------------------------------------===//

// Load rounded
def : InstRW<[DFU, Lat30], (instregex "LEDTR$")>;
def : InstRW<[DFU, DFU, Lat30], (instregex "LDXTR$")>;

// Load lengthened
def : InstRW<[DFU, Lat20], (instregex "LDETR$")>;
def : InstRW<[DFU2, DFU2, Lat20, GroupAlone], (instregex "LXDTR$")>;

// Convert from fixed / logical
def : InstRW<[FXU, DFU, Lat9, GroupAlone], (instregex "CDFTR$")>;
def : InstRW<[FXU, DFU, Lat30, GroupAlone], (instregex "CDGTR(A)?$")>;
def : InstRW<[FXU, DFU2, DFU2, GroupAlone], (instregex "CXFTR$")>;
def : InstRW<[FXU, DFU2, DFU2, Lat30, GroupAlone], (instregex "CXGTR(A)?$")>;
def : InstRW<[FXU, DFU, Lat11, GroupAlone], (instregex "CDL(F|G)TR$")>;
def : InstRW<[FXU, DFU2, DFU2, Lat11, GroupAlone], (instregex "CXLFTR$")>;
def : InstRW<[FXU, DFU2, DFU2, Lat6, GroupAlone], (instregex "CXLGTR$")>;

// Convert to fixed / logical
def : InstRW<[FXU, DFU, Lat11, GroupAlone], (instregex "CFDTR(A)?$")>;
def : InstRW<[FXU, DFU, Lat30, GroupAlone], (instregex "CGDTR(A)?$")>;
def : InstRW<[FXU, DFU, DFU, Lat11, GroupAlone], (instregex "CFXTR$")>;
def : InstRW<[FXU, DFU, DFU, Lat30, GroupAlone], (instregex "CGXTR(A)?$")>;
def : InstRW<[FXU, DFU, Lat11, GroupAlone], (instregex "CL(F|G)DTR$")>;
def : InstRW<[FXU, DFU, DFU, Lat11, GroupAlone], (instregex "CL(F|G)XTR$")>;

// Convert from / to signed / unsigned packed
def : InstRW<[FXU, DFU, Lat12, GroupAlone], (instregex "CD(S|U)TR$")>;
def : InstRW<[FXU, FXU, DFU2, DFU2, Lat20, GroupAlone], (instregex "CX(S|U)TR$")>;
def : InstRW<[FXU, DFU, Lat12, GroupAlone], (instregex "C(S|U)DTR$")>;
def : InstRW<[FXU, FXU, DFU2, DFU2, Lat20, GroupAlone], (instregex "C(S|U)XTR$")>;

// Perform floating-point operation
def : InstRW<[FXU, Lat30], (instregex "PFPO$")>;

//===----------------------------------------------------------------------===//
// DFP: Unary arithmetic
//===----------------------------------------------------------------------===//

// Load FP integer
def : InstRW<[DFU, Lat20], (instregex "FIDTR$")>;
def : InstRW<[DFU2, DFU2, Lat20, GroupAlone], (instregex "FIXTR$")>;

// Extract biased exponent
def : InstRW<[FXU, DFU, Lat15, GroupAlone], (instregex "EEDTR$")>;
def : InstRW<[FXU, DFU2, Lat15, GroupAlone], (instregex "EEXTR$")>;

// Extract significance
def : InstRW<[FXU, DFU, Lat15, GroupAlone], (instregex "ESDTR$")>;
def : InstRW<[FXU, DFU, DFU, Lat20, GroupAlone], (instregex "ESXTR$")>;

//===----------------------------------------------------------------------===//
// DFP: Binary arithmetic
//===----------------------------------------------------------------------===//

// Addition
def : InstRW<[DFU, Lat30], (instregex "ADTR(A)?$")>;
def : InstRW<[DFU2, DFU2, Lat30, GroupAlone], (instregex "AXTR(A)?$")>;

// Subtraction
def : InstRW<[DFU, Lat30], (instregex "SDTR(A)?$")>;
def : InstRW<[DFU2, DFU2, Lat30, GroupAlone], (instregex "SXTR(A)?$")>;

// Multiply
def : InstRW<[DFU, Lat30], (instregex "MDTR(A)?$")>;
def : InstRW<[DFU2, DFU2, Lat30, GroupAlone], (instregex "MXTR(A)?$")>;

// Division
def : InstRW<[DFU, Lat30], (instregex "DDTR(A)?$")>;
def : InstRW<[DFU2, DFU2, Lat30, GroupAlone], (instregex "DXTR(A)?$")>;

// Quantize
def : InstRW<[DFU, Lat30], (instregex "QADTR$")>;
def : InstRW<[DFU2, DFU2, Lat30, GroupAlone], (instregex "QAXTR$")>;

// Reround
def : InstRW<[FXU, DFU, Lat30, GroupAlone], (instregex "RRDTR$")>;
def : InstRW<[FXU, DFU2, DFU2, Lat30, GroupAlone], (instregex "RRXTR$")>;

// Shift significand left/right
def : InstRW<[LSU, DFU, Lat11, GroupAlone], (instregex "S(L|R)DT$")>;
def : InstRW<[LSU, DFU2, DFU2, Lat15, GroupAlone], (instregex "S(L|R)XT$")>;

// Insert biased exponent
def : InstRW<[FXU, DFU, Lat11, GroupAlone], (instregex "IEDTR$")>;
def : InstRW<[FXU, DFU2, DFU2, Lat15, GroupAlone], (instregex "IEXTR$")>;

//===----------------------------------------------------------------------===//
// DFP: Comparisons
//===----------------------------------------------------------------------===//

// Compare
def : InstRW<[DFU, Lat11], (instregex "(K|C)DTR$")>;
def : InstRW<[DFU, DFU, Lat15], (instregex "(K|C)XTR$")>;

// Compare biased exponent
def : InstRW<[DFU, Lat8], (instregex "CEDTR$")>;
def : InstRW<[DFU2, Lat9], (instregex "CEXTR$")>;

// Test Data Class/Group
def : InstRW<[LSU, DFU, Lat15], (instregex "TD(C|G)(E|D)T$")>;
def : InstRW<[LSU, DFU2, Lat15], (instregex "TD(C|G)XT$")>;


// -------------------------------- System ---------------------------------- //

//===----------------------------------------------------------------------===//
// System: Program-Status Word Instructions
//===----------------------------------------------------------------------===//

def : InstRW<[FXU, Lat30], (instregex "EPSW$")>;
def : InstRW<[FXU, LSU, Lat30], (instregex "LPSW(E)?$")>;
def : InstRW<[FXU, Lat3, GroupAlone], (instregex "IPK$")>;
def : InstRW<[LSU, EndGroup], (instregex "SPKA$")>;
def : InstRW<[LSU, EndGroup], (instregex "SSM$")>;
def : InstRW<[FXU, LSU, GroupAlone], (instregex "ST(N|O)SM$")>;
def : InstRW<[FXU, Lat3], (instregex "IAC$")>;
def : InstRW<[LSU, EndGroup], (instregex "SAC(F)?$")>;

//===----------------------------------------------------------------------===//
// System: Control Register Instructions
//===----------------------------------------------------------------------===//

def : InstRW<[FXU, LSU, Lat30], (instregex "LCTL(G)?$")>;
def : InstRW<[FXU, LSU, LSU, LSU, LSU, Lat10, GroupAlone],
             (instregex "STCT(L|G)$")>;
def : InstRW<[LSU], (instregex "E(P|S)A(I)?R$")>;
def : InstRW<[FXU, Lat30], (instregex "SSA(I)?R$")>;
def : InstRW<[FXU, Lat30], (instregex "ESEA$")>;

//===----------------------------------------------------------------------===//
// System: Prefix-Register Instructions
//===----------------------------------------------------------------------===//

def : InstRW<[FXU, LSU, Lat30], (instregex "SPX$")>;
def : InstRW<[FXU, LSU, Lat30], (instregex "STPX$")>;

//===----------------------------------------------------------------------===//
// System: Storage-Key and Real Memory Instructions
//===----------------------------------------------------------------------===//

def : InstRW<[FXU, Lat30], (instregex "ISKE$")>;
def : InstRW<[FXU, Lat30], (instregex "IVSK$")>;
def : InstRW<[FXU, Lat30], (instregex "SSKE(Opt)?$")>;
def : InstRW<[FXU, Lat30], (instregex "RRB(E|M)$")>;
def : InstRW<[FXU, Lat30], (instregex "PFMF$")>;
def : InstRW<[FXU, Lat30], (instregex "TB$")>;
def : InstRW<[FXU, LSU, Lat30], (instregex "PGIN$")>;
def : InstRW<[FXU, LSU, Lat30], (instregex "PGOUT$")>;

//===----------------------------------------------------------------------===//
// System: Dynamic-Address-Translation Instructions
//===----------------------------------------------------------------------===//

def : InstRW<[FXU, LSU, Lat30], (instregex "IPTE(Opt)?(Opt)?$")>;
def : InstRW<[FXU, Lat30], (instregex "IDTE(Opt)?$")>;
def : InstRW<[FXU, Lat30], (instregex "PTLB$")>;
def : InstRW<[FXU, LSU, Lat30], (instregex "CSP(G)?$")>;
def : InstRW<[FXU, LSU, Lat30], (instregex "LPTEA$")>;
def : InstRW<[FXU, LSU, Lat30], (instregex "LRA(Y|G)?$")>;
def : InstRW<[FXU, LSU, Lat30], (instregex "STRAG$")>;
def : InstRW<[FXU, LSU, Lat30], (instregex "LURA(G)?$")>;
def : InstRW<[FXU, LSU, Lat30], (instregex "STUR(A|G)$")>;
def : InstRW<[FXU, LSU, Lat30], (instregex "TPROT$")>;

//===----------------------------------------------------------------------===//
// System: Memory-move Instructions
//===----------------------------------------------------------------------===//

def : InstRW<[LSU, Lat8, GroupAlone], (instregex "MVC(K|P|S)$")>;
def : InstRW<[LSU, Lat30, GroupAlone], (instregex "MVCSK$")>;
def : InstRW<[LSU, Lat6, GroupAlone], (instregex "MVCDK$")>;
def : InstRW<[FXU, LSU, Lat30], (instregex "MVCOS$")>;
def : InstRW<[LSU, Lat30, GroupAlone], (instregex "MVPG$")>;

//===----------------------------------------------------------------------===//
// System: Address-Space Instructions
//===----------------------------------------------------------------------===//

def : InstRW<[FXU, LSU, Lat30], (instregex "LASP$")>;
def : InstRW<[LSU, GroupAlone], (instregex "PALB$")>;
def : InstRW<[FXU, LSU, Lat30], (instregex "PC$")>;
def : InstRW<[FXU, Lat30], (instregex "PR$")>;
def : InstRW<[FXU, Lat30], (instregex "PT(I)?$")>;
def : InstRW<[FXU, LSU, Lat30], (instregex "RP$")>;
def : InstRW<[FXU, Lat30], (instregex "BS(G|A)$")>;
def : InstRW<[FXU, Lat20], (instregex "TAR$")>;

//===----------------------------------------------------------------------===//
// System: Linkage-Stack Instructions
//===----------------------------------------------------------------------===//

def : InstRW<[FXU, LSU, Lat30, EndGroup], (instregex "BAKR$")>;
def : InstRW<[FXU, Lat30], (instregex "EREG(G)?$")>;
def : InstRW<[FXU, Lat30], (instregex "(E|M)STA$")>;

//===----------------------------------------------------------------------===//
// System: Time-Related Instructions
//===----------------------------------------------------------------------===//

def : InstRW<[FXU, Lat30], (instregex "PTFF$")>;
def : InstRW<[FXU, LSU, Lat20], (instregex "SCK$")>;
def : InstRW<[FXU, Lat30], (instregex "SCKPF$")>;
def : InstRW<[FXU, LSU, Lat20], (instregex "SCKC$")>;
def : InstRW<[FXU, LSU, Lat20], (instregex "SPT$")>;
def : InstRW<[FXU, LSU, Lat15], (instregex "STCK$")>;
def : InstRW<[FXU, LSU, Lat12], (instregex "STCKF$")>;
def : InstRW<[FXU, LSU, Lat30], (instregex "STCKE$")>;
def : InstRW<[FXU, LSU, Lat9], (instregex "STCKC$")>;
def : InstRW<[FXU, LSU, Lat8], (instregex "STPT$")>;

//===----------------------------------------------------------------------===//
// System: CPU-Related Instructions
//===----------------------------------------------------------------------===//

def : InstRW<[FXU, LSU, Lat30], (instregex "STAP$")>;
def : InstRW<[FXU, LSU, Lat30], (instregex "STIDP$")>;
def : InstRW<[FXU, LSU, Lat30], (instregex "STSI$")>;
def : InstRW<[FXU, LSU, Lat30], (instregex "STFL(E)?$")>;
def : InstRW<[FXU, LSU, Lat30], (instregex "ECAG$")>;
def : InstRW<[FXU, LSU, Lat30], (instregex "ECTG$")>;
def : InstRW<[FXU, Lat30], (instregex "PTF$")>;
def : InstRW<[FXU, Lat30], (instregex "PCKMO$")>;

//===----------------------------------------------------------------------===//
// System: Miscellaneous Instructions
//===----------------------------------------------------------------------===//

def : InstRW<[FXU, Lat30], (instregex "SVC$")>;
def : InstRW<[FXU, GroupAlone], (instregex "MC$")>;
def : InstRW<[FXU, Lat30], (instregex "DIAG$")>;
def : InstRW<[FXU, LSU, Lat30], (instregex "TRAC(E|G)$")>;
def : InstRW<[FXU, Lat30], (instregex "TRAP(2|4)$")>;
def : InstRW<[FXU, Lat30], (instregex "SIGP$")>;
def : InstRW<[FXU, LSU, Lat30], (instregex "SIGA$")>;
def : InstRW<[FXU, LSU, Lat30], (instregex "SIE$")>;

//===----------------------------------------------------------------------===//
// System: CPU-Measurement Facility Instructions
//===----------------------------------------------------------------------===//

def : InstRW<[FXU], (instregex "LPP$")>;
def : InstRW<[FXU, Lat30], (instregex "ECPGA$")>;
def : InstRW<[FXU, Lat30], (instregex "E(C|P)CTR$")>;
def : InstRW<[FXU, Lat30], (instregex "LCCTL$")>;
def : InstRW<[FXU, LSU, Lat30], (instregex "L(P|S)CTL$")>;
def : InstRW<[FXU, LSU, Lat30], (instregex "Q(S|CTR)I$")>;
def : InstRW<[FXU, Lat30], (instregex "S(C|P)CTR$")>;

//===----------------------------------------------------------------------===//
// System: I/O Instructions
//===----------------------------------------------------------------------===//

def : InstRW<[FXU, Lat30], (instregex "(C|H|R|X)SCH$")>;
def : InstRW<[FXU, LSU, Lat30], (instregex "(M|S|ST|T)SCH$")>;
def : InstRW<[FXU, Lat30], (instregex "RCHP$")>;
def : InstRW<[FXU, Lat30], (instregex "SCHM$")>;
def : InstRW<[FXU, LSU, Lat30], (instregex "STC(PS|RW)$")>;
def : InstRW<[FXU, LSU, Lat30], (instregex "TPI$")>;
def : InstRW<[FXU, Lat30], (instregex "SAL$")>;

}

