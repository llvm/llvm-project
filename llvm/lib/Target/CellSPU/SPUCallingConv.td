//===- SPUCallingConv.td - Calling Conventions for CellSPU -*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This describes the calling conventions for the STI Cell SPU architecture.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Return Value Calling Convention
//===----------------------------------------------------------------------===//

// Return-value convention for Cell SPU: return value to be passed in reg 3-74
def RetCC_SPU : CallingConv<[
  CCIfType<[i8,i16,i32,i64,i128,f32,f64,v16i8,v8i16,v4i32,v2i64,v4f32,v2f64],
  CCAssignToReg<[R3,   R4,  R5,  R6,  R7,  R8,  R9, R10, R11,
                 R12, R13, R14, R15, R16, R17, R18, R19, R20,
                 R21, R22, R23, R24, R25, R26, R27, R28, R29,
                 R30, R31, R32, R33, R34, R35, R36, R37, R38,
                 R39, R40, R41, R42, R43, R44, R45, R46, R47,
                 R48, R49, R50, R51, R52, R53, R54, R55, R56,
                 R57, R58, R59, R60, R61, R62, R63, R64, R65,
                 R66, R67, R68, R69, R70, R71, R72, R73, R74]>>
]>;


//===----------------------------------------------------------------------===//
// CellSPU Argument Calling Conventions
//===----------------------------------------------------------------------===//
def CCC_SPU : CallingConv<[
  CCIfType<[i8, i16, i32, i64, i128, f32, f64, 
            v16i8, v8i16, v4i32, v4f32, v2i64, v2f64],
            CCAssignToReg<[R3,   R4,  R5,  R6,  R7,  R8,  R9, R10, R11,
                           R12, R13, R14, R15, R16, R17, R18, R19, R20,
                           R21, R22, R23, R24, R25, R26, R27, R28, R29,
                           R30, R31, R32, R33, R34, R35, R36, R37, R38,
                           R39, R40, R41, R42, R43, R44, R45, R46, R47,
                           R48, R49, R50, R51, R52, R53, R54, R55, R56,
                           R57, R58, R59, R60, R61, R62, R63, R64, R65,
                           R66, R67, R68, R69, R70, R71, R72, R73, R74]>>,
  // Integer/FP values get stored in stack slots that are 8 bytes in size and
  // 8-byte aligned if there are no more registers to hold them.
  CCIfType<[i32, i64, f32, f64], CCAssignToStack<8, 8>>,
  
  // Vectors get 16-byte stack slots that are 16-byte aligned.
  CCIfType<[v16i8, v8i16, v4i32, v2i64, v4f32, v2f64],
              CCAssignToStack<16, 16>>
]>;
