//=- ARMScheduleSwift.td - Swift Scheduling Definitions -*- tablegen -*----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the itinerary class data for the Swift processor..
//
//===----------------------------------------------------------------------===//

// ===---------------------------------------------------------------------===//
// This section contains legacy support for itineraries. This is
// required until SD and PostRA schedulers are replaced by MachineScheduler.

def SW_DIS0 : FuncUnit;
def SW_DIS1 : FuncUnit;
def SW_DIS2 : FuncUnit;

def SW_ALU0 : FuncUnit;
def SW_ALU1 : FuncUnit;
def SW_LS   : FuncUnit;
def SW_IDIV : FuncUnit;
def SW_FDIV : FuncUnit;

// FIXME: Need bypasses.
// FIXME: Model the multiple stages of IIC_iMOVix2, IIC_iMOVix2addpc, and
//        IIC_iMOVix2ld better.
// FIXME: Model the special immediate shifts that are not microcoded.
// FIXME: Do we need to model the fact that uses of r15 in a micro-op force it
//        to issue on pipe 1?
// FIXME: Model the pipelined behavior of CMP / TST instructions.
// FIXME: Better model the microcode stages of multiply instructions, especially
//        conditional variants.
// FIXME: Add preload instruction when it is documented.
// FIXME: Model non-pipelined nature of FP div / sqrt unit.

def SwiftItineraries : ProcessorItineraries<
  [SW_DIS0, SW_DIS1, SW_DIS2, SW_ALU0, SW_ALU1, SW_LS, SW_IDIV, SW_FDIV], [], [
  //
  // Move instructions, unconditional
  InstrItinData<IIC_iMOVi   , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>],
                              [1]>,
  InstrItinData<IIC_iMOVr   , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>],
                              [1]>,
  InstrItinData<IIC_iMOVsi  , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>],
                              [1]>,
  InstrItinData<IIC_iMOVsr  , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>],
                              [1]>,
  InstrItinData<IIC_iMOVix2 , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>],
                              [2]>,
  InstrItinData<IIC_iMOVix2addpc,[InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                                  InstrStage<1, [SW_ALU0, SW_ALU1]>,
                                  InstrStage<1, [SW_ALU0, SW_ALU1]>,
                                  InstrStage<1, [SW_ALU0, SW_ALU1]>],
                                 [3]>,
  InstrItinData<IIC_iMOVix2ld,[InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>,
                               InstrStage<1, [SW_LS]>],
                              [5]>,
  //
  // MVN instructions
  InstrItinData<IIC_iMVNi   , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>],
                              [1]>,
  InstrItinData<IIC_iMVNr   , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>],
                              [1]>,
  InstrItinData<IIC_iMVNsi  , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>],
                              [1]>,
  InstrItinData<IIC_iMVNsr  , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>],
                              [1]>,
  //
  // No operand cycles
  InstrItinData<IIC_iALUx   , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>]>,
  //
  // Binary Instructions that produce a result
  InstrItinData<IIC_iALUi , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                             InstrStage<1, [SW_ALU0, SW_ALU1]>],
                            [1, 1]>,
  InstrItinData<IIC_iALUr , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                             InstrStage<1, [SW_ALU0, SW_ALU1]>],
                            [1, 1, 1]>,
  InstrItinData<IIC_iALUsi, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                             InstrStage<1, [SW_ALU0, SW_ALU1]>],
                            [2, 1, 1]>,
  InstrItinData<IIC_iALUsir,[InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                             InstrStage<1, [SW_ALU0, SW_ALU1]>],
                            [2, 1, 1]>,
  InstrItinData<IIC_iALUsr, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                             InstrStage<1, [SW_ALU0, SW_ALU1]>],
                            [2, 1, 1, 1]>,
  //
  // Bitwise Instructions that produce a result
  InstrItinData<IIC_iBITi , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                             InstrStage<1, [SW_ALU0, SW_ALU1]>],
                            [1, 1]>,
  InstrItinData<IIC_iBITr , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                             InstrStage<1, [SW_ALU0, SW_ALU1]>],
                            [1, 1, 1]>,
  InstrItinData<IIC_iBITsi, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                             InstrStage<1, [SW_ALU0, SW_ALU1]>],
                            [2, 1, 1]>,
  InstrItinData<IIC_iBITsr, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                             InstrStage<1, [SW_ALU0, SW_ALU1]>],
                            [2, 1, 1, 1]>,
  //
  // Unary Instructions that produce a result

  // CLZ, RBIT, etc.
  InstrItinData<IIC_iUNAr , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                             InstrStage<1, [SW_ALU0, SW_ALU1]>],
                            [1, 1]>,

  // BFC, BFI, UBFX, SBFX
  InstrItinData<IIC_iUNAsi, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                             InstrStage<1, [SW_ALU0, SW_ALU1]>],
                            [2, 1]>,

  //
  // Zero and sign extension instructions
  InstrItinData<IIC_iEXTr , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                             InstrStage<1, [SW_ALU0, SW_ALU1]>],
                            [1, 1]>,
  InstrItinData<IIC_iEXTAr, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                             InstrStage<1, [SW_ALU0, SW_ALU1]>],
                            [1, 1, 1]>,
  InstrItinData<IIC_iEXTAsr,[InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                             InstrStage<1, [SW_ALU0, SW_ALU1]>],
                            [1, 1, 1, 1]>,
  //
  // Compare instructions
  InstrItinData<IIC_iCMPi   , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>],
                              [1]>,
  InstrItinData<IIC_iCMPr   , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>],
                              [1, 1]>,
  InstrItinData<IIC_iCMPsi  , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<2, [SW_ALU0, SW_ALU1]>],
                              [1, 1]>,
  InstrItinData<IIC_iCMPsr  , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<2, [SW_ALU0, SW_ALU1]>],
                              [1, 1, 1]>,
  //
  // Test instructions
  InstrItinData<IIC_iTSTi   , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>],
                              [1]>,
  InstrItinData<IIC_iTSTr   , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>],
                              [1, 1]>,
  InstrItinData<IIC_iTSTsi  , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<2, [SW_ALU0, SW_ALU1]>],
                              [1, 1]>,
  InstrItinData<IIC_iTSTsr  , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<2, [SW_ALU0, SW_ALU1]>],
                              [1, 1, 1]>,
  //
  // Move instructions, conditional
  // FIXME: Correctly model the extra input dep on the destination.
  InstrItinData<IIC_iCMOVi  , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>],
                              [1]>,
  InstrItinData<IIC_iCMOVr  , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>],
                              [1, 1]>,
  InstrItinData<IIC_iCMOVsi , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>],
                              [1, 1]>,
  InstrItinData<IIC_iCMOVsr , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>],
                              [2, 1, 1]>,
  InstrItinData<IIC_iCMOVix2, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>],
                              [2]>,

  // Integer multiply pipeline
  //
  InstrItinData<IIC_iMUL16  , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [3, 1, 1]>,
  InstrItinData<IIC_iMAC16  , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [3, 1, 1, 1]>,
  InstrItinData<IIC_iMUL32  , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [4, 1, 1]>,
  InstrItinData<IIC_iMAC32  , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [4, 1, 1, 1]>,
  InstrItinData<IIC_iMUL64  , [InstrStage<1, [SW_DIS0], 0>,
                               InstrStage<1, [SW_DIS1], 0>,
                               InstrStage<1, [SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0], 1>,
                               InstrStage<1, [SW_ALU0], 3>,
                               InstrStage<1, [SW_ALU0]>],
                              [5, 5, 1, 1]>,
  InstrItinData<IIC_iMAC64  , [InstrStage<1, [SW_DIS0], 0>,
                               InstrStage<1, [SW_DIS1], 0>,
                               InstrStage<1, [SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0], 1>,
                               InstrStage<1, [SW_ALU0], 1>,
                               InstrStage<1, [SW_ALU0, SW_ALU1], 3>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>],
                              [5, 6, 1, 1]>,
  //
  // Integer divide
  InstrItinData<IIC_iDIV  , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                             InstrStage<1, [SW_ALU0], 0>,
                             InstrStage<14, [SW_IDIV]>],
                            [14, 1, 1]>,

  // Integer load pipeline
  // FIXME: The timings are some rough approximations
  //
  // Immediate offset
  InstrItinData<IIC_iLoad_i   , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                                 InstrStage<1, [SW_LS]>],
                                [3, 1]>,
  InstrItinData<IIC_iLoad_bh_i, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                                 InstrStage<1, [SW_LS]>],
                                [3, 1]>,
  InstrItinData<IIC_iLoad_d_i , [InstrStage<1, [SW_DIS0], 0>,
                                 InstrStage<1, [SW_DIS1], 0>,
                                 InstrStage<1, [SW_LS], 1>,
                                 InstrStage<1, [SW_LS]>],
                                [3, 4, 1]>,
  //
  // Register offset
  InstrItinData<IIC_iLoad_r   , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                                 InstrStage<1, [SW_LS]>],
                                [3, 1, 1]>,
  InstrItinData<IIC_iLoad_bh_r, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                                 InstrStage<1, [SW_LS]>],
                                [3, 1, 1]>,
  InstrItinData<IIC_iLoad_d_r , [InstrStage<1, [SW_DIS0], 0>,
                                 InstrStage<1, [SW_DIS1], 0>,
                                 InstrStage<1, [SW_DIS2], 0>,
                                 InstrStage<1, [SW_LS], 1>,
                                 InstrStage<1, [SW_LS], 3>,
                                 InstrStage<1, [SW_ALU0, SW_ALU1]>],
                                [3, 4, 1, 1]>,
  //
  // Scaled register offset
  InstrItinData<IIC_iLoad_si  , [InstrStage<1, [SW_DIS0], 0>,
                                 InstrStage<1, [SW_DIS1], 0>,
                                 InstrStage<1, [SW_ALU0, SW_ALU1], 2>,
                                 InstrStage<1, [SW_LS]>],
                                [5, 1, 1]>,
  InstrItinData<IIC_iLoad_bh_si,[InstrStage<1, [SW_DIS0], 0>,
                                 InstrStage<1, [SW_DIS1], 0>,
                                 InstrStage<1, [SW_ALU0, SW_ALU1], 2>,
                                 InstrStage<1, [SW_LS]>],
                                [5, 1, 1]>,
  //
  // Immediate offset with update
  InstrItinData<IIC_iLoad_iu  , [InstrStage<1, [SW_DIS0], 0>,
                                 InstrStage<1, [SW_DIS1], 0>,
                                 InstrStage<1, [SW_ALU0, SW_ALU1], 1>,
                                 InstrStage<1, [SW_LS]>],
                                [3, 1, 1]>,
  InstrItinData<IIC_iLoad_bh_iu,[InstrStage<1, [SW_DIS0], 0>,
                                 InstrStage<1, [SW_DIS1], 0>,
                                 InstrStage<1, [SW_ALU0, SW_ALU1], 1>,
                                 InstrStage<1, [SW_LS]>],
                                [3, 1, 1]>,
  //
  // Register offset with update
  InstrItinData<IIC_iLoad_ru  , [InstrStage<1, [SW_DIS0], 0>,
                                 InstrStage<1, [SW_DIS1], 0>,
                                 InstrStage<1, [SW_ALU0], 1>,
                                 InstrStage<1, [SW_LS]>],
                                [3, 1, 1, 1]>,
  InstrItinData<IIC_iLoad_bh_ru,[InstrStage<1, [SW_DIS0], 0>,
                                 InstrStage<1, [SW_DIS1], 0>,
                                 InstrStage<1, [SW_ALU0], 1>,
                                 InstrStage<1, [SW_LS]>],
                                [3, 1, 1, 1]>,
  InstrItinData<IIC_iLoad_d_ru, [InstrStage<1, [SW_DIS0], 0>,
                                 InstrStage<1, [SW_DIS1], 0>,
                                 InstrStage<1, [SW_DIS2], 0>,
                                 InstrStage<1, [SW_ALU0, SW_ALU1], 0>,
                                 InstrStage<1, [SW_LS], 3>,
                                 InstrStage<1, [SW_LS], 0>,
                                 InstrStage<1, [SW_ALU0, SW_ALU1]>],
                                [3, 4, 1, 1]>,
  //
  // Scaled register offset with update
  InstrItinData<IIC_iLoad_siu , [InstrStage<1, [SW_DIS0], 0>,
                                 InstrStage<1, [SW_DIS1], 0>,
                                 InstrStage<1, [SW_DIS2], 0>,
                                 InstrStage<1, [SW_ALU0, SW_ALU1], 2>,
                                 InstrStage<1, [SW_LS], 3>,
                                 InstrStage<1, [SW_ALU0, SW_ALU1]>],
                                [5, 3, 1, 1]>,
  InstrItinData<IIC_iLoad_bh_siu,[InstrStage<1, [SW_DIS0], 0>,
                                  InstrStage<1, [SW_DIS1], 0>,
                                  InstrStage<1, [SW_DIS2], 0>,
                                  InstrStage<1, [SW_ALU0, SW_ALU1], 2>,
                                  InstrStage<1, [SW_LS], 0>,
                                  InstrStage<1, [SW_ALU0, SW_ALU1]>],
                                [5, 3, 1, 1]>,
  //
  // Load multiple, def is the 5th operand.
  // FIXME: This assumes 3 to 4 registers.
  InstrItinData<IIC_iLoad_m  , [InstrStage<1, [SW_DIS0], 0>,
                                InstrStage<1, [SW_DIS1], 0>,
                                InstrStage<1, [SW_DIS2], 0>,
                                InstrStage<1, [SW_ALU0, SW_ALU1], 1>,
                                InstrStage<1, [SW_LS]>],
                               [1, 1, 1, 1, 3], [], -1>, // dynamic uops

  //
  // Load multiple + update, defs are the 1st and 5th operands.
  InstrItinData<IIC_iLoad_mu , [InstrStage<1, [SW_DIS0], 0>,
                                InstrStage<1, [SW_DIS1], 0>,
                                InstrStage<1, [SW_DIS2], 0>,
                                InstrStage<1, [SW_ALU0, SW_ALU1], 0>,
                                InstrStage<1, [SW_LS], 3>,
                                InstrStage<1, [SW_ALU0, SW_ALU1]>],
                               [2, 1, 1, 1, 3], [], -1>, // dynamic uops
  //
  // Load multiple plus branch
  InstrItinData<IIC_iLoad_mBr, [InstrStage<1, [SW_DIS0], 0>,
                                InstrStage<1, [SW_DIS1], 0>,
                                InstrStage<1, [SW_DIS2], 0>,
                                InstrStage<1, [SW_ALU0, SW_ALU1], 1>,
                                InstrStage<1, [SW_LS]>],
                               [1, 1, 1, 1, 3], [], -1>, // dynamic uops
  //
  // Pop, def is the 3rd operand.
  InstrItinData<IIC_iPop  ,    [InstrStage<1, [SW_DIS0], 0>,
                                InstrStage<1, [SW_DIS1], 0>,
                                InstrStage<1, [SW_ALU0, SW_ALU1], 1>,
                                InstrStage<1, [SW_LS]>],
                               [1, 1, 3], [], -1>, // dynamic uops
  //
  // Pop + branch, def is the 3rd operand.
  InstrItinData<IIC_iPop_Br,   [InstrStage<1, [SW_DIS0], 0>,
                                InstrStage<1, [SW_DIS1], 0>,
                                InstrStage<1, [SW_ALU0, SW_ALU1], 1>,
                                InstrStage<1, [SW_LS]>],
                               [1, 1, 3], [], -1>, // dynamic uops

  //
  // iLoadi + iALUr for t2LDRpci_pic.
  InstrItinData<IIC_iLoadiALU, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                                InstrStage<1, [SW_LS], 3>,
                                InstrStage<1, [SW_ALU0, SW_ALU1]>],
                               [4, 1]>,

  // Integer store pipeline
  ///
  // Immediate offset
  InstrItinData<IIC_iStore_i  , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                                 InstrStage<1, [SW_LS]>],
                                [1, 1]>,
  InstrItinData<IIC_iStore_bh_i,[InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                                 InstrStage<1, [SW_LS]>],
                                [1, 1]>,
  InstrItinData<IIC_iStore_d_i, [InstrStage<1, [SW_DIS0], 0>,
                                 InstrStage<1, [SW_DIS1], 0>,
                                 InstrStage<1, [SW_DIS2], 0>,
                                 InstrStage<1, [SW_LS], 0>,
                                 InstrStage<1, [SW_ALU0, SW_ALU1], 1>,
                                 InstrStage<1, [SW_LS]>],
                                [1, 1]>,
  //
  // Register offset
  InstrItinData<IIC_iStore_r  , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                                 InstrStage<1, [SW_LS]>],
                                [1, 1, 1]>,
  InstrItinData<IIC_iStore_bh_r,[InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                                 InstrStage<1, [SW_LS]>],
                                [1, 1, 1]>,
  InstrItinData<IIC_iStore_d_r, [InstrStage<1, [SW_DIS0], 0>,
                                 InstrStage<1, [SW_DIS1], 0>,
                                 InstrStage<1, [SW_DIS2], 0>,
                                 InstrStage<1, [SW_LS], 0>,
                                 InstrStage<1, [SW_ALU0, SW_ALU1], 1>,
                                 InstrStage<1, [SW_LS]>],
                                [1, 1, 1]>,
  //
  // Scaled register offset
  InstrItinData<IIC_iStore_si ,  [InstrStage<1, [SW_DIS0], 0>,
                                  InstrStage<1, [SW_DIS1], 0>,
                                  InstrStage<1, [SW_ALU0, SW_ALU1], 2>,
                                  InstrStage<1, [SW_LS]>],
                                 [1, 1, 1]>,
  InstrItinData<IIC_iStore_bh_si,[InstrStage<1, [SW_DIS0], 0>,
                                  InstrStage<1, [SW_DIS1], 0>,
                                  InstrStage<1, [SW_ALU0, SW_ALU1], 2>,
                                  InstrStage<1, [SW_LS]>],
                                 [1, 1, 1]>,
  //
  // Immediate offset with update
  InstrItinData<IIC_iStore_iu ,  [InstrStage<1, [SW_DIS0], 0>,
                                  InstrStage<1, [SW_DIS1], 0>,
                                  InstrStage<1, [SW_ALU0, SW_ALU1], 1>,
                                  InstrStage<1, [SW_LS]>],
                                 [1, 1, 1]>,
  InstrItinData<IIC_iStore_bh_iu,[InstrStage<1, [SW_DIS0], 0>,
                                  InstrStage<1, [SW_DIS1], 0>,
                                  InstrStage<1, [SW_ALU0, SW_ALU1], 1>,
                                  InstrStage<1, [SW_LS]>],
                                 [1, 1, 1]>,
  //
  // Register offset with update
  InstrItinData<IIC_iStore_ru ,  [InstrStage<1, [SW_DIS0], 0>,
                                  InstrStage<1, [SW_DIS1], 0>,
                                  InstrStage<1, [SW_ALU0, SW_ALU1], 1>,
                                  InstrStage<1, [SW_LS]>],
                                 [1, 1, 1, 1]>,
  InstrItinData<IIC_iStore_bh_ru,[InstrStage<1, [SW_DIS0], 0>,
                                  InstrStage<1, [SW_DIS1], 0>,
                                  InstrStage<1, [SW_ALU0, SW_ALU1], 1>,
                                  InstrStage<1, [SW_LS]>],
                                 [1, 1, 1, 1]>,
  InstrItinData<IIC_iStore_d_ru, [InstrStage<1, [SW_DIS0], 0>,
                                  InstrStage<1, [SW_DIS1], 0>,
                                  InstrStage<1, [SW_ALU0, SW_ALU1], 1>,
                                  InstrStage<1, [SW_LS]>],
                                 [1, 1, 1, 1]>,
  //
  // Scaled register offset with update
  InstrItinData<IIC_iStore_siu,    [InstrStage<1, [SW_DIS0], 0>,
                                    InstrStage<1, [SW_DIS1], 0>,
                                    InstrStage<1, [SW_ALU0, SW_ALU1], 2>,
                                    InstrStage<1, [SW_LS], 0>,
                                    InstrStage<1, [SW_ALU0, SW_ALU1], 1>],
                                   [3, 1, 1, 1]>,
  InstrItinData<IIC_iStore_bh_siu, [InstrStage<1, [SW_DIS0], 0>,
                                    InstrStage<1, [SW_DIS1], 0>,
                                    InstrStage<1, [SW_ALU0, SW_ALU1], 2>,
                                    InstrStage<1, [SW_LS], 0>,
                                    InstrStage<1, [SW_ALU0, SW_ALU1], 1>],
                                   [3, 1, 1, 1]>,
  //
  // Store multiple
  InstrItinData<IIC_iStore_m , [InstrStage<1, [SW_DIS0], 0>,
                                InstrStage<1, [SW_DIS1], 0>,
                                InstrStage<1, [SW_DIS2], 0>,
                                InstrStage<1, [SW_ALU0, SW_ALU1], 1>,
                                InstrStage<1, [SW_LS], 1>,
                                InstrStage<1, [SW_ALU0, SW_ALU1], 1>,
                                InstrStage<1, [SW_LS], 1>,
                                InstrStage<1, [SW_ALU0, SW_ALU1], 1>,
                                InstrStage<1, [SW_LS]>],
                                [], [], -1>, // dynamic uops
  //
  // Store multiple + update
  InstrItinData<IIC_iStore_mu, [InstrStage<1, [SW_DIS0], 0>,
                                InstrStage<1, [SW_DIS1], 0>,
                                InstrStage<1, [SW_DIS2], 0>,
                                InstrStage<1, [SW_ALU0, SW_ALU1], 1>,
                                InstrStage<1, [SW_LS], 1>,
                                InstrStage<1, [SW_ALU0, SW_ALU1], 1>,
                                InstrStage<1, [SW_LS], 1>,
                                InstrStage<1, [SW_ALU0, SW_ALU1], 1>,
                                InstrStage<1, [SW_LS]>],
                               [2], [], -1>, // dynamic uops

  //
  // Preload
  InstrItinData<IIC_Preload,   [InstrStage<1, [SW_DIS0], 0>], [1, 1]>,

  // Branch
  //
  // no delay slots, so the latency of a branch is unimportant
  InstrItinData<IIC_Br       , [InstrStage<1, [SW_DIS0], 0>]>,

  // FP Special Register to Integer Register File Move
  InstrItinData<IIC_fpSTAT , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                              InstrStage<1, [SW_ALU0, SW_ALU1]>],
                             [1]>,
  //
  // Single-precision FP Unary
  //
  // Most floating-point moves get issued on ALU0.
  InstrItinData<IIC_fpUNA32 , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [2, 1]>,
  //
  // Double-precision FP Unary
  InstrItinData<IIC_fpUNA64 , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [2, 1]>,

  //
  // Single-precision FP Compare
  InstrItinData<IIC_fpCMP32 , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [1, 1]>,
  //
  // Double-precision FP Compare
  InstrItinData<IIC_fpCMP64 , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [1, 1]>,
  //
  // Single to Double FP Convert
  InstrItinData<IIC_fpCVTSD , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [4, 1]>,
  //
  // Double to Single FP Convert
  InstrItinData<IIC_fpCVTDS , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [4, 1]>,

  //
  // Single to Half FP Convert
  InstrItinData<IIC_fpCVTSH , [InstrStage<1, [SW_DIS0], 0>,
                               InstrStage<1, [SW_DIS1], 0>,
                               InstrStage<1, [SW_ALU1], 4>,
                               InstrStage<1, [SW_ALU1]>],
                              [6, 1]>,
  //
  // Half to Single FP Convert
  InstrItinData<IIC_fpCVTHS , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [4, 1]>,

  //
  // Single-Precision FP to Integer Convert
  InstrItinData<IIC_fpCVTSI , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [4, 1]>,
  //
  // Double-Precision FP to Integer Convert
  InstrItinData<IIC_fpCVTDI , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [4, 1]>,
  //
  // Integer to Single-Precision FP Convert
  InstrItinData<IIC_fpCVTIS , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [4, 1]>,
  //
  // Integer to Double-Precision FP Convert
  InstrItinData<IIC_fpCVTID , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [4, 1]>,
  //
  // Single-precision FP ALU
  InstrItinData<IIC_fpALU32 , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [2, 1, 1]>,
  //
  // Double-precision FP ALU
  InstrItinData<IIC_fpALU64 , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [2, 1, 1]>,
  //
  // Single-precision FP Multiply
  InstrItinData<IIC_fpMUL32 , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [4, 1, 1]>,
  //
  // Double-precision FP Multiply
  InstrItinData<IIC_fpMUL64 , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [6, 1, 1]>,
  //
  // Single-precision FP MAC
  InstrItinData<IIC_fpMAC32 , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [8, 1, 1]>,
  //
  // Double-precision FP MAC
  InstrItinData<IIC_fpMAC64 , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [12, 1, 1]>,
  //
  // Single-precision Fused FP MAC
  InstrItinData<IIC_fpFMAC32, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [8, 1, 1]>,
  //
  // Double-precision Fused FP MAC
  InstrItinData<IIC_fpFMAC64, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [12, 1, 1]>,
  //
  // Single-precision FP DIV
  InstrItinData<IIC_fpDIV32 , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1], 0>,
                               InstrStage<15, [SW_FDIV]>],
                              [17, 1, 1]>,
  //
  // Double-precision FP DIV
  InstrItinData<IIC_fpDIV64 , [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1], 0>,
                               InstrStage<30, [SW_FDIV]>],
                              [32, 1, 1]>,
  //
  // Single-precision FP SQRT
  InstrItinData<IIC_fpSQRT32, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1], 0>,
                               InstrStage<15, [SW_FDIV]>],
                              [17, 1]>,
  //
  // Double-precision FP SQRT
  InstrItinData<IIC_fpSQRT64, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1], 0>,
                               InstrStage<30, [SW_FDIV]>],
                              [32, 1, 1]>,

  //
  // Integer to Single-precision Move
  InstrItinData<IIC_fpMOVIS,  [InstrStage<1, [SW_DIS0], 0>,
                               InstrStage<1, [SW_DIS1], 0>,
                               InstrStage<1, [SW_LS], 4>,
                               InstrStage<1, [SW_ALU0]>],
                              [6, 1]>,
  //
  // Integer to Double-precision Move
  InstrItinData<IIC_fpMOVID,  [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_LS]>],
                              [4, 1]>,
  //
  // Single-precision to Integer Move
  InstrItinData<IIC_fpMOVSI,  [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_LS]>],
                              [3, 1]>,
  //
  // Double-precision to Integer Move
  InstrItinData<IIC_fpMOVDI,  [InstrStage<1, [SW_DIS0], 0>,
                               InstrStage<1, [SW_DIS1], 0>,
                               InstrStage<1, [SW_LS], 3>,
                               InstrStage<1, [SW_LS]>],
                              [3, 4, 1]>,
  //
  // Single-precision FP Load
  InstrItinData<IIC_fpLoad32, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_LS]>],
                              [4, 1]>,
  //
  // Double-precision FP Load
  InstrItinData<IIC_fpLoad64, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_LS]>],
                              [4, 1]>,
  //
  // FP Load Multiple
  // FIXME: Assumes a single Q register.
  InstrItinData<IIC_fpLoad_m, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_LS]>],
                              [1, 1, 1, 4], [], -1>, // dynamic uops
  //
  // FP Load Multiple + update
  // FIXME: Assumes a single Q register.
  InstrItinData<IIC_fpLoad_mu,[InstrStage<1, [SW_DIS0], 0>,
                               InstrStage<1, [SW_DIS1], 0>,
                               InstrStage<1, [SW_LS], 4>,
                               InstrStage<1, [SW_ALU0, SW_ALU1]>],
                              [2, 1, 1, 1, 4], [], -1>, // dynamic uops
  //
  // Single-precision FP Store
  InstrItinData<IIC_fpStore32,[InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_LS]>],
                              [1, 1]>,
  //
  // Double-precision FP Store
  InstrItinData<IIC_fpStore64,[InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_LS]>],
                              [1, 1]>,
  //
  // FP Store Multiple
  // FIXME: Assumes a single Q register.
  InstrItinData<IIC_fpStore_m,[InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_LS]>],
                              [1, 1, 1], [], -1>, // dynamic uops
  //
  // FP Store Multiple + update
  // FIXME: Assumes a single Q register.
  InstrItinData<IIC_fpStore_mu,[InstrStage<1, [SW_DIS0], 0>,
                                InstrStage<1, [SW_DIS1], 0>,
                                InstrStage<1, [SW_LS], 4>,
                                InstrStage<1, [SW_ALU0, SW_ALU1]>],
                               [2, 1, 1, 1], [], -1>, // dynamic uops
  // NEON
  //
  // Double-register Integer Unary
  InstrItinData<IIC_VUNAiD,   [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [4, 1]>,
  //
  // Quad-register Integer Unary
  InstrItinData<IIC_VUNAiQ,   [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [4, 1]>,
  //
  // Double-register Integer Q-Unary
  InstrItinData<IIC_VQUNAiD,  [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [4, 1]>,
  //
  // Quad-register Integer CountQ-Unary
  InstrItinData<IIC_VQUNAiQ,  [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [4, 1]>,
  //
  // Double-register Integer Binary
  InstrItinData<IIC_VBINiD,   [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [2, 1, 1]>,
  //
  // Quad-register Integer Binary
  InstrItinData<IIC_VBINiQ,   [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [2, 1, 1]>,
  //
  // Double-register Integer Subtract
  InstrItinData<IIC_VSUBiD,   [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [2, 1, 1]>,
  //
  // Quad-register Integer Subtract
  InstrItinData<IIC_VSUBiQ,   [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [2, 1, 1]>,
  //
  // Double-register Integer Shift
  InstrItinData<IIC_VSHLiD,   [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [2, 1, 1]>,
  //
  // Quad-register Integer Shift
  InstrItinData<IIC_VSHLiQ,   [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [2, 1, 1]>,
  //
  // Double-register Integer Shift (4 cycle)
  InstrItinData<IIC_VSHLi4D,  [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [4, 1, 1]>,
  //
  // Quad-register Integer Shift (4 cycle)
  InstrItinData<IIC_VSHLi4Q,  [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [4, 1, 1]>,
  //
  // Double-register Integer Binary (4 cycle)
  InstrItinData<IIC_VBINi4D,  [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [4, 1, 1]>,
  //
  // Quad-register Integer Binary (4 cycle)
  InstrItinData<IIC_VBINi4Q,  [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [4, 1, 1]>,
  //
  // Double-register Integer Subtract (4 cycle)
  InstrItinData<IIC_VSUBi4D,  [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [4, 1, 1]>,
  //
  // Quad-register Integer Subtract (4 cycle)
  InstrItinData<IIC_VSUBi4Q,  [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [4, 1, 1]>,

  //
  // Double-register Integer Count
  InstrItinData<IIC_VCNTiD,   [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [2, 1, 1]>,
  //
  // Quad-register Integer Count
  InstrItinData<IIC_VCNTiQ,   [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [2, 1, 1]>,
  //
  // Double-register Absolute Difference and Accumulate
  InstrItinData<IIC_VABAD,    [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [4, 1, 1, 1]>,
  //
  // Quad-register Absolute Difference and Accumulate
  InstrItinData<IIC_VABAQ,    [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [4, 1, 1, 1]>,
  //
  // Double-register Integer Pair Add Long
  InstrItinData<IIC_VPALiD,   [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [4, 1, 1]>,
  //
  // Quad-register Integer Pair Add Long
  InstrItinData<IIC_VPALiQ,   [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [4, 1, 1]>,

  //
  // Double-register Integer Multiply (.8, .16)
  InstrItinData<IIC_VMULi16D, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [4, 1, 1]>,
  //
  // Quad-register Integer Multiply (.8, .16)
  InstrItinData<IIC_VMULi16Q, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [4, 1, 1]>,

  //
  // Double-register Integer Multiply (.32)
  InstrItinData<IIC_VMULi32D, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [4, 1, 1]>,
  //
  // Quad-register Integer Multiply (.32)
  InstrItinData<IIC_VMULi32Q, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [4, 1, 1]>,
  //
  // Double-register Integer Multiply-Accumulate (.8, .16)
  InstrItinData<IIC_VMACi16D, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [4, 1, 1, 1]>,
  //
  // Double-register Integer Multiply-Accumulate (.32)
  InstrItinData<IIC_VMACi32D, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [4, 1, 1, 1]>,
  //
  // Quad-register Integer Multiply-Accumulate (.8, .16)
  InstrItinData<IIC_VMACi16Q, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [4, 1, 1, 1]>,
  //
  // Quad-register Integer Multiply-Accumulate (.32)
  InstrItinData<IIC_VMACi32Q, [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [4, 1, 1, 1]>,

  //
  // Move
  InstrItinData<IIC_VMOV,     [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [2, 1]>,
  //
  // Move Immediate
  InstrItinData<IIC_VMOVImm,  [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [2]>,
  //
  // Double-register Permute Move
  InstrItinData<IIC_VMOVD,    [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [2, 1]>,
  //
  // Quad-register Permute Move
  InstrItinData<IIC_VMOVQ,    [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [2, 1]>,
  //
  // Integer to Single-precision Move
  InstrItinData<IIC_VMOVIS ,  [InstrStage<1, [SW_DIS0], 0>,
                               InstrStage<1, [SW_DIS1], 0>,
                               InstrStage<1, [SW_LS], 4>,
                               InstrStage<1, [SW_ALU0]>],
                              [6, 1]>,
  //
  // Integer to Double-precision Move
  InstrItinData<IIC_VMOVID ,  [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_LS]>],
                              [4, 1, 1]>,
  //
  // Single-precision to Integer Move
  InstrItinData<IIC_VMOVSI ,  [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_LS]>],
                              [3, 1]>,
  //
  // Double-precision to Integer Move
  InstrItinData<IIC_VMOVDI ,  [InstrStage<1, [SW_DIS0], 0>,
                               InstrStage<1, [SW_DIS1], 0>,
                               InstrStage<1, [SW_LS], 3>,
                               InstrStage<1, [SW_LS]>],
                              [3, 4, 1]>,
  //
  // Integer to Lane Move
  // FIXME: I think this is correct, but it is not clear from the tuning guide.
  InstrItinData<IIC_VMOVISL , [InstrStage<1, [SW_DIS0], 0>,
                               InstrStage<1, [SW_DIS1], 0>,
                               InstrStage<1, [SW_LS], 4>,
                               InstrStage<1, [SW_ALU0]>],
                              [6, 1]>,

  //
  // Vector narrow move
  InstrItinData<IIC_VMOVN,    [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [2, 1]>,
  //
  // Double-register FP Unary
  // FIXME: VRECPE / VRSQRTE has a longer latency than VABS, which is used here,
  //        and they issue on a different pipeline.
  InstrItinData<IIC_VUNAD,    [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [2, 1]>,
  //
  // Quad-register FP Unary
  // FIXME: VRECPE / VRSQRTE has a longer latency than VABS, which is used here,
  //        and they issue on a different pipeline.
  InstrItinData<IIC_VUNAQ,    [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [2, 1]>,
  //
  // Double-register FP Binary
  // FIXME: We're using this itin for many instructions.
  InstrItinData<IIC_VBIND,    [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [4, 1, 1]>,

  //
  // VPADD, etc.
  InstrItinData<IIC_VPBIND,   [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [4, 1, 1]>,
  //
  // Double-register FP VMUL
  InstrItinData<IIC_VFMULD,   [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [4, 1, 1]>,
  //
  // Quad-register FP Binary
  InstrItinData<IIC_VBINQ,    [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU0]>],
                              [4, 1, 1]>,
  //
  // Quad-register FP VMUL
  InstrItinData<IIC_VFMULQ,   [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [4, 1, 1]>,
  //
  // Double-register FP Multiple-Accumulate
  InstrItinData<IIC_VMACD,    [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [8, 1, 1]>,
  //
  // Quad-register FP Multiple-Accumulate
  InstrItinData<IIC_VMACQ,    [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [8, 1, 1]>,
  //
  // Double-register Fused FP Multiple-Accumulate
  InstrItinData<IIC_VFMACD,   [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [8, 1, 1]>,
  //
  // Quad-register FusedF P Multiple-Accumulate
  InstrItinData<IIC_VFMACQ,   [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [8, 1, 1]>,
  //
  // Double-register Reciprical Step
  InstrItinData<IIC_VRECSD,   [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [8, 1, 1]>,
  //
  // Quad-register Reciprical Step
  InstrItinData<IIC_VRECSQ,   [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [8, 1, 1]>,
  //
  // Double-register Permute
  // FIXME: The latencies are unclear from the documentation.
  InstrItinData<IIC_VPERMD,   [InstrStage<1, [SW_DIS0], 0>,
                               InstrStage<1, [SW_DIS1], 0>,
                               InstrStage<1, [SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1], 2>,
                               InstrStage<1, [SW_ALU1], 2>,
                               InstrStage<1, [SW_ALU1]>],
                              [3, 4, 3, 4]>,
  //
  // Quad-register Permute
  // FIXME: The latencies are unclear from the documentation.
  InstrItinData<IIC_VPERMQ,   [InstrStage<1, [SW_DIS0], 0>,
                               InstrStage<1, [SW_DIS1], 0>,
                               InstrStage<1, [SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1], 2>,
                               InstrStage<1, [SW_ALU1], 2>,
                               InstrStage<1, [SW_ALU1]>],
                              [3, 4, 3, 4]>,
  //
  // Quad-register Permute (3 cycle issue on A9)
  InstrItinData<IIC_VPERMQ3,  [InstrStage<1, [SW_DIS0], 0>,
                               InstrStage<1, [SW_DIS1], 0>,
                               InstrStage<1, [SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1], 2>,
                               InstrStage<1, [SW_ALU1], 2>,
                               InstrStage<1, [SW_ALU1]>],
                              [3, 4, 3, 4]>,

  //
  // Double-register VEXT
  InstrItinData<IIC_VEXTD,    [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [2, 1, 1]>,
  //
  // Quad-register VEXT
  InstrItinData<IIC_VEXTQ,    [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [2, 1, 1]>,
  //
  // VTB
  InstrItinData<IIC_VTB1,     [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [2, 1, 1]>,
  InstrItinData<IIC_VTB2,     [InstrStage<1, [SW_DIS0], 0>,
                               InstrStage<1, [SW_DIS1], 0>,
                               InstrStage<1, [SW_ALU1], 2>,
                               InstrStage<1, [SW_ALU1]>],
                              [4, 1, 3, 3]>,
  InstrItinData<IIC_VTB3,     [InstrStage<1, [SW_DIS0], 0>,
                               InstrStage<1, [SW_DIS1], 0>,
                               InstrStage<1, [SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1], 2>,
                               InstrStage<1, [SW_ALU1], 2>,
                               InstrStage<1, [SW_ALU1]>],
                              [6, 1, 3, 5, 5]>,
  InstrItinData<IIC_VTB4,     [InstrStage<1, [SW_DIS0], 0>,
                               InstrStage<1, [SW_DIS1], 0>,
                               InstrStage<1, [SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1], 2>,
                               InstrStage<1, [SW_ALU1], 2>,
                               InstrStage<1, [SW_ALU1], 2>,
                               InstrStage<1, [SW_ALU1]>],
                              [8, 1, 3, 5, 7, 7]>,
  //
  // VTBX
  InstrItinData<IIC_VTBX1,    [InstrStage<1, [SW_DIS0, SW_DIS1, SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1]>],
                              [2, 1, 1]>,
  InstrItinData<IIC_VTBX2,    [InstrStage<1, [SW_DIS0], 0>,
                               InstrStage<1, [SW_DIS1], 0>,
                               InstrStage<1, [SW_ALU1], 2>,
                               InstrStage<1, [SW_ALU1]>],
                              [4, 1, 3, 3]>,
  InstrItinData<IIC_VTBX3,    [InstrStage<1, [SW_DIS0], 0>,
                               InstrStage<1, [SW_DIS1], 0>,
                               InstrStage<1, [SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1], 2>,
                               InstrStage<1, [SW_ALU1], 2>,
                               InstrStage<1, [SW_ALU1]>],
                              [6, 1, 3, 5, 5]>,
  InstrItinData<IIC_VTBX4,    [InstrStage<1, [SW_DIS0], 0>,
                               InstrStage<1, [SW_DIS1], 0>,
                               InstrStage<1, [SW_DIS2], 0>,
                               InstrStage<1, [SW_ALU1], 2>,
                               InstrStage<1, [SW_ALU1], 2>,
                               InstrStage<1, [SW_ALU1], 2>,
                               InstrStage<1, [SW_ALU1]>],
                              [8, 1, 3, 5, 7, 7]>
]>;

// ===---------------------------------------------------------------------===//
// This following definitions describe the simple machine model which
// will replace itineraries.

// Swift machine model for scheduling and other instruction cost heuristics.
def SwiftModel : SchedMachineModel {
  let IssueWidth = 3; // 3 micro-ops are dispatched per cycle.
  let MinLatency = 0; // Data dependencies are allowed within dispatch groups.
  let LoadLatency = 3;

  let Itineraries = SwiftItineraries;
}

// TODO: Add Swift processor and scheduler resources.
