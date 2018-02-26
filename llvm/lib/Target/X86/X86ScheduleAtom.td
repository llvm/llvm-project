//===- X86ScheduleAtom.td - X86 Atom Scheduling Definitions -*- tablegen -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the itinerary class data for the Intel Atom
// in order (Saltwell-32nm/Bonnell-45nm) processors.
//
//===----------------------------------------------------------------------===//

//
// Scheduling information derived from the "Intel 64 and IA32 Architectures
// Optimization Reference Manual", Chapter 13, Section 4.
// Functional Units
//    Port 0
def Port0 : FuncUnit; // ALU: ALU0, shift/rotate, load/store
                      // SIMD/FP: SIMD ALU, Shuffle,SIMD/FP multiply, divide
def Port1 : FuncUnit; // ALU: ALU1, bit processing, jump, and LEA
                      // SIMD/FP: SIMD ALU, FP Adder

def AtomItineraries : ProcessorItineraries<
  [ Port0, Port1 ],
  [], [
  // P0 only
  // InstrItinData<class, [InstrStage<N, [P0]>] >,
  // P0 or P1
  // InstrItinData<class, [InstrStage<N, [P0, P1]>] >,
  // P0 and P1
  // InstrItinData<class, [InstrStage<N, [P0], 0>,  InstrStage<N, [P1]>] >,
  //
  // Default is 1 cycle, port0 or port1
  InstrItinData<IIC_ALU_MEM, [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_ALU_NONMEM, [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_LEA, [InstrStage<1, [Port1]>] >,
  InstrItinData<IIC_LEA_16, [InstrStage<2, [Port0, Port1]>] >,
  // mul
  InstrItinData<IIC_MUL8, [InstrStage<7, [Port0, Port1]>] >,
  InstrItinData<IIC_MUL16_MEM, [InstrStage<8, [Port0, Port1]>] >,
  InstrItinData<IIC_MUL16_REG, [InstrStage<7, [Port0, Port1]>] >,
  InstrItinData<IIC_MUL32_MEM, [InstrStage<7, [Port0, Port1]>] >,
  InstrItinData<IIC_MUL32_REG, [InstrStage<6, [Port0, Port1]>] >,
  InstrItinData<IIC_MUL64, [InstrStage<12, [Port0, Port1]>] >,
  // imul by al, ax, eax, rax
  InstrItinData<IIC_IMUL8, [InstrStage<7, [Port0, Port1]>] >,
  InstrItinData<IIC_IMUL16_MEM, [InstrStage<8, [Port0, Port1]>] >,
  InstrItinData<IIC_IMUL16_REG, [InstrStage<7, [Port0, Port1]>] >,
  InstrItinData<IIC_IMUL32_MEM, [InstrStage<7, [Port0, Port1]>] >,
  InstrItinData<IIC_IMUL32_REG, [InstrStage<6, [Port0, Port1]>] >,
  InstrItinData<IIC_IMUL64, [InstrStage<12, [Port0, Port1]>] >,
  // imul reg by reg|mem
  InstrItinData<IIC_IMUL16_RM, [InstrStage<7, [Port0, Port1]>] >,
  InstrItinData<IIC_IMUL16_RR, [InstrStage<6, [Port0, Port1]>] >,
  InstrItinData<IIC_IMUL32_RM, [InstrStage<5, [Port0]>] >,
  InstrItinData<IIC_IMUL32_RR, [InstrStage<5, [Port0]>] >,
  InstrItinData<IIC_IMUL64_RM, [InstrStage<12, [Port0, Port1]>] >,
  InstrItinData<IIC_IMUL64_RR, [InstrStage<12, [Port0, Port1]>] >,
  // imul reg = reg/mem * imm
  InstrItinData<IIC_IMUL16_RRI, [InstrStage<6, [Port0, Port1]>] >,
  InstrItinData<IIC_IMUL32_RRI, [InstrStage<5, [Port0]>] >,
  InstrItinData<IIC_IMUL64_RRI, [InstrStage<14, [Port0, Port1]>] >,
  InstrItinData<IIC_IMUL16_RMI, [InstrStage<7, [Port0, Port1]>] >,
  InstrItinData<IIC_IMUL32_RMI, [InstrStage<5, [Port0]>] >,
  InstrItinData<IIC_IMUL64_RMI, [InstrStage<14, [Port0, Port1]>] >,
  // idiv
  InstrItinData<IIC_IDIV8, [InstrStage<62, [Port0, Port1]>] >,
  InstrItinData<IIC_IDIV16, [InstrStage<62, [Port0, Port1]>] >,
  InstrItinData<IIC_IDIV32, [InstrStage<62, [Port0, Port1]>] >,
  InstrItinData<IIC_IDIV64, [InstrStage<130, [Port0, Port1]>] >,
  // div
  InstrItinData<IIC_DIV8_REG, [InstrStage<50, [Port0, Port1]>] >,
  InstrItinData<IIC_DIV8_MEM, [InstrStage<68, [Port0, Port1]>] >,
  InstrItinData<IIC_DIV16, [InstrStage<50, [Port0, Port1]>] >,
  InstrItinData<IIC_DIV32, [InstrStage<50, [Port0, Port1]>] >,
  InstrItinData<IIC_DIV64, [InstrStage<130, [Port0, Port1]>] >,
  // neg/not/inc/dec
  InstrItinData<IIC_UNARY_REG, [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_UNARY_MEM, [InstrStage<1, [Port0]>] >,
  // add/sub/and/or/xor/cmp/test
  InstrItinData<IIC_BIN_NONMEM, [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_BIN_MEM, [InstrStage<1, [Port0]>] >,
  // adc/sbc
  InstrItinData<IIC_BIN_CARRY_NONMEM, [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_BIN_CARRY_MEM, [InstrStage<1, [Port0]>] >,
  // shift/rotate
  InstrItinData<IIC_SR, [InstrStage<1, [Port0]>] >,
  // shift double
  InstrItinData<IIC_SHD16_REG_IM, [InstrStage<6, [Port0, Port1]>] >,
  InstrItinData<IIC_SHD16_REG_CL, [InstrStage<6, [Port0, Port1]>] >,
  InstrItinData<IIC_SHD16_MEM_IM, [InstrStage<6, [Port0, Port1]>] >,
  InstrItinData<IIC_SHD16_MEM_CL, [InstrStage<6, [Port0, Port1]>] >,
  InstrItinData<IIC_SHD32_REG_IM, [InstrStage<2, [Port0, Port1]>] >,
  InstrItinData<IIC_SHD32_REG_CL, [InstrStage<2, [Port0, Port1]>] >,
  InstrItinData<IIC_SHD32_MEM_IM, [InstrStage<4, [Port0, Port1]>] >,
  InstrItinData<IIC_SHD32_MEM_CL, [InstrStage<4, [Port0, Port1]>] >,
  InstrItinData<IIC_SHD64_REG_IM, [InstrStage<9, [Port0, Port1]>] >,
  InstrItinData<IIC_SHD64_REG_CL, [InstrStage<8, [Port0, Port1]>] >,
  InstrItinData<IIC_SHD64_MEM_IM, [InstrStage<9, [Port0, Port1]>] >,
  InstrItinData<IIC_SHD64_MEM_CL, [InstrStage<9, [Port0, Port1]>] >,
  // cmov
  InstrItinData<IIC_CMOV16_RM, [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_CMOV16_RR, [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_CMOV32_RM, [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_CMOV32_RR, [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_CMOV64_RM, [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_CMOV64_RR, [InstrStage<1, [Port0, Port1]>] >,
  // set
  InstrItinData<IIC_SET_M, [InstrStage<2, [Port0, Port1]>] >,
  InstrItinData<IIC_SET_R, [InstrStage<1, [Port0, Port1]>] >,
  // jcc
  InstrItinData<IIC_Jcc, [InstrStage<1, [Port1]>] >,
  // jcxz/jecxz/jrcxz
  InstrItinData<IIC_JCXZ, [InstrStage<4, [Port0, Port1]>] >,
  // jmp rel
  InstrItinData<IIC_JMP_REL, [InstrStage<1, [Port1]>] >,
  // jmp indirect
  InstrItinData<IIC_JMP_REG, [InstrStage<1, [Port1]>] >,
  InstrItinData<IIC_JMP_MEM, [InstrStage<2, [Port0, Port1]>] >,
  // jmp far
  InstrItinData<IIC_JMP_FAR_MEM, [InstrStage<32, [Port0, Port1]>] >,
  InstrItinData<IIC_JMP_FAR_PTR, [InstrStage<31, [Port0, Port1]>] >,
  // loop/loope/loopne
  InstrItinData<IIC_LOOP, [InstrStage<18, [Port0, Port1]>] >,
  InstrItinData<IIC_LOOPE, [InstrStage<8, [Port0, Port1]>] >,
  InstrItinData<IIC_LOOPNE, [InstrStage<17, [Port0, Port1]>] >,
  // call - all but reg/imm
  InstrItinData<IIC_CALL_RI, [InstrStage<1, [Port0], 0>,
                              InstrStage<1, [Port1]>] >,
  InstrItinData<IIC_CALL_MEM, [InstrStage<15, [Port0, Port1]>] >,
  InstrItinData<IIC_CALL_FAR_MEM, [InstrStage<40, [Port0, Port1]>] >,
  InstrItinData<IIC_CALL_FAR_PTR, [InstrStage<39, [Port0, Port1]>] >,
  //ret
  InstrItinData<IIC_RET, [InstrStage<79, [Port0, Port1]>] >,
  InstrItinData<IIC_RET_IMM, [InstrStage<1, [Port0], 0>,  InstrStage<1, [Port1]>] >,
  //sign extension movs
  InstrItinData<IIC_MOVSX,[InstrStage<1, [Port0] >] >,
  InstrItinData<IIC_MOVSX_R16_R8, [InstrStage<2, [Port0, Port1]>] >,
  InstrItinData<IIC_MOVSX_R16_M8, [InstrStage<3, [Port0, Port1]>] >,
  InstrItinData<IIC_MOVSX_R16_R16, [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_MOVSX_R32_R32, [InstrStage<1, [Port0, Port1]>] >,
  //zero extension movs
  InstrItinData<IIC_MOVZX,[InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_MOVZX_R16_R8, [InstrStage<2, [Port0, Port1]>] >,
  InstrItinData<IIC_MOVZX_R16_M8, [InstrStage<3, [Port0, Port1]>] >,

  InstrItinData<IIC_REP_MOVS, [InstrStage<75, [Port0, Port1]>] >,
  InstrItinData<IIC_REP_STOS, [InstrStage<74, [Port0, Port1]>] >,

  // SSE binary operations
  // arithmetic fp scalar
  InstrItinData<IIC_SSE_ALU_F32S_RR, [InstrStage<5, [Port1]>] >,
  InstrItinData<IIC_SSE_ALU_F32S_RM, [InstrStage<5, [Port0], 0>,
                                   InstrStage<5, [Port1]>] >,
  InstrItinData<IIC_SSE_ALU_F64S_RR, [InstrStage<5, [Port1]>] >,
  InstrItinData<IIC_SSE_ALU_F64S_RM, [InstrStage<5, [Port0], 0>,
                                   InstrStage<5, [Port1]>] >,
  InstrItinData<IIC_SSE_MUL_F32S_RR, [InstrStage<4, [Port0]>] >,
  InstrItinData<IIC_SSE_MUL_F32S_RM, [InstrStage<4, [Port0]>] >,
  InstrItinData<IIC_SSE_MUL_F64S_RR, [InstrStage<5, [Port0]>] >,
  InstrItinData<IIC_SSE_MUL_F64S_RM, [InstrStage<5, [Port0]>] >,
  InstrItinData<IIC_SSE_DIV_F32S_RR, [InstrStage<34, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_DIV_F32S_RM, [InstrStage<34, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_DIV_F64S_RR, [InstrStage<62, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_DIV_F64S_RM, [InstrStage<62, [Port0, Port1]>] >,

  InstrItinData<IIC_SSE_COMIS_RR, [InstrStage<9, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_COMIS_RM, [InstrStage<10, [Port0, Port1]>] >,

  InstrItinData<IIC_SSE_HADDSUB_RR, [InstrStage<8, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_HADDSUB_RM, [InstrStage<9, [Port0, Port1]>] >,

  // arithmetic fp parallel
  InstrItinData<IIC_SSE_ALU_F32P_RR, [InstrStage<5, [Port1]>] >,
  InstrItinData<IIC_SSE_ALU_F32P_RM, [InstrStage<5, [Port0], 0>,
                                   InstrStage<5, [Port1]>] >,
  InstrItinData<IIC_SSE_ALU_F64P_RR, [InstrStage<6, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_ALU_F64P_RM, [InstrStage<7, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_MUL_F32P_RR, [InstrStage<5, [Port0]>] >,
  InstrItinData<IIC_SSE_MUL_F32P_RM, [InstrStage<5, [Port0]>] >,
  InstrItinData<IIC_SSE_MUL_F64P_RR, [InstrStage<9, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_MUL_F64P_RM, [InstrStage<10, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_DIV_F32P_RR, [InstrStage<70, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_DIV_F32P_RM, [InstrStage<70, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_DIV_F64P_RR, [InstrStage<125, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_DIV_F64P_RM, [InstrStage<125, [Port0, Port1]>] >,

  // bitwise parallel
  InstrItinData<IIC_SSE_BIT_P_RR, [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_BIT_P_RM, [InstrStage<1, [Port0]>] >,

  // arithmetic int parallel
  InstrItinData<IIC_SSE_INTALU_P_RR, [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_INTALU_P_RM, [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_SSE_INTALUQ_P_RR, [InstrStage<2, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_INTALUQ_P_RM, [InstrStage<3, [Port0, Port1]>] >,

  // multiply int parallel
  InstrItinData<IIC_SSE_INTMUL_P_RR, [InstrStage<5, [Port0]>] >,
  InstrItinData<IIC_SSE_INTMUL_P_RM, [InstrStage<5, [Port0]>] >,

  // shift parallel
  InstrItinData<IIC_SSE_INTSH_P_RR, [InstrStage<2, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_INTSH_P_RM, [InstrStage<3, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_INTSH_P_RI, [InstrStage<1, [Port0, Port1]>] >,

  InstrItinData<IIC_SSE_INTSHDQ_P_RI, [InstrStage<1, [Port0, Port1]>] >,

  InstrItinData<IIC_SSE_SHUFP, [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_SSE_PSHUF_RI, [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_SSE_PSHUF_MI, [InstrStage<1, [Port0]>] >,

  InstrItinData<IIC_SSE_PACK, [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_SSE_UNPCK, [InstrStage<1, [Port0]>] >,

  InstrItinData<IIC_SSE_SQRTPS_RR, [InstrStage<70, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_SQRTPS_RM, [InstrStage<70, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_SQRTSS_RR, [InstrStage<34, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_SQRTSS_RM, [InstrStage<34, [Port0, Port1]>] >,

  InstrItinData<IIC_SSE_SQRTPD_RR, [InstrStage<125, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_SQRTPD_RM, [InstrStage<125, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_SQRTSD_RR, [InstrStage<62, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_SQRTSD_RM, [InstrStage<62, [Port0, Port1]>] >,

  InstrItinData<IIC_SSE_RSQRTPS_RR, [InstrStage<9, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_RSQRTPS_RM, [InstrStage<10, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_RSQRTSS_RR, [InstrStage<4, [Port0]>] >,
  InstrItinData<IIC_SSE_RSQRTSS_RM, [InstrStage<4, [Port0]>] >,

  InstrItinData<IIC_SSE_RCPP_RR, [InstrStage<9, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_RCPP_RM, [InstrStage<10, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_RCPS_RR, [InstrStage<4, [Port0]>] >,
  InstrItinData<IIC_SSE_RCPS_RM, [InstrStage<4, [Port0]>] >,

  InstrItinData<IIC_SSE_MOVMSK, [InstrStage<3, [Port0]>] >,
  InstrItinData<IIC_SSE_MASKMOV, [InstrStage<2, [Port0, Port1]>] >,

  InstrItinData<IIC_SSE_PEXTRW, [InstrStage<4, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_PINSRW, [InstrStage<1, [Port0]>] >,

  InstrItinData<IIC_SSE_PABS_RR, [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_PABS_RM, [InstrStage<1, [Port0]>] >,

  InstrItinData<IIC_SSE_MOV_S_RR, [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_MOV_S_RM, [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_SSE_MOV_S_MR, [InstrStage<1, [Port0]>] >,

  InstrItinData<IIC_SSE_MOVA_P_RR, [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_MOVA_P_RM, [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_SSE_MOVA_P_MR, [InstrStage<1, [Port0]>] >,

  InstrItinData<IIC_SSE_MOVU_P_RR, [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_MOVU_P_RM, [InstrStage<3, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_MOVU_P_MR, [InstrStage<2, [Port0, Port1]>] >,

  InstrItinData<IIC_SSE_MOV_LH, [InstrStage<1, [Port0]>] >,

  InstrItinData<IIC_SSE_LDDQU, [InstrStage<3, [Port0, Port1]>] >,

  InstrItinData<IIC_SSE_MOVDQ, [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_SSE_MOVD_ToGP, [InstrStage<3, [Port0]>] >,
  InstrItinData<IIC_SSE_MOVQ_RR, [InstrStage<1, [Port0, Port1]>] >,

  InstrItinData<IIC_SSE_MOVNT, [InstrStage<1, [Port0]>] >,

  InstrItinData<IIC_SSE_PREFETCH, [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_SSE_PAUSE, [InstrStage<17, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_LFENCE, [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_MFENCE, [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_SSE_SFENCE, [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_SSE_LDMXCSR, [InstrStage<5, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_STMXCSR, [InstrStage<15, [Port0, Port1]>] >,

  InstrItinData<IIC_SSE_PHADDSUBD_RR, [InstrStage<3, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_PHADDSUBD_RM, [InstrStage<4, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_PHADDSUBSW_RR, [InstrStage<7, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_PHADDSUBSW_RM, [InstrStage<8, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_PHADDSUBW_RR, [InstrStage<7, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_PHADDSUBW_RM, [InstrStage<8, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_PSHUFB_RR, [InstrStage<4, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_PSHUFB_RM, [InstrStage<5, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_PSIGN_RR, [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_PSIGN_RM, [InstrStage<1, [Port0]>] >,

  InstrItinData<IIC_SSE_PMADD, [InstrStage<5, [Port0]>] >,
  InstrItinData<IIC_SSE_PMULHRSW, [InstrStage<5, [Port0]>] >,
  InstrItinData<IIC_SSE_PALIGNRR, [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_SSE_PALIGNRM, [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_SSE_MWAIT, [InstrStage<46, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_MONITOR, [InstrStage<45, [Port0, Port1]>] >,

  // conversions
  // to/from PD ...
  InstrItinData<IIC_SSE_CVT_PD_RR, [InstrStage<7, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_CVT_PD_RM, [InstrStage<8, [Port0, Port1]>] >,
  // to/from PS except to/from PD and PS2PI
  InstrItinData<IIC_SSE_CVT_PS_RR, [InstrStage<6, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_CVT_PS_RM, [InstrStage<7, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_CVT_Scalar_RR, [InstrStage<6, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_CVT_Scalar_RM, [InstrStage<7, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_CVT_SS2SI32_RR, [InstrStage<8, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_CVT_SS2SI32_RM, [InstrStage<9, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_CVT_SS2SI64_RR, [InstrStage<9, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_CVT_SS2SI64_RM, [InstrStage<10, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_CVT_SD2SI_RR, [InstrStage<8, [Port0, Port1]>] >,
  InstrItinData<IIC_SSE_CVT_SD2SI_RM, [InstrStage<9, [Port0, Port1]>] >,

  // MMX MOVs
  InstrItinData<IIC_MMX_MOV_MM_RM,  [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_MMX_MOV_REG_MM, [InstrStage<3, [Port0]>] >,
  InstrItinData<IIC_MMX_MOVQ_RM, [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_MMX_MOVQ_RR, [InstrStage<1, [Port0, Port1]>] >,
  // other MMX
  InstrItinData<IIC_MMX_ALU_RM,  [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_MMX_ALU_RR,  [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_MMX_ALUQ_RM, [InstrStage<3, [Port0, Port1]>] >,
  InstrItinData<IIC_MMX_ALUQ_RR, [InstrStage<2, [Port0, Port1]>] >,
  InstrItinData<IIC_MMX_PHADDSUBW_RM, [InstrStage<6, [Port0, Port1]>] >,
  InstrItinData<IIC_MMX_PHADDSUBW_RR, [InstrStage<5, [Port0, Port1]>] >,
  InstrItinData<IIC_MMX_PHADDSUBD_RM, [InstrStage<4, [Port0, Port1]>] >,
  InstrItinData<IIC_MMX_PHADDSUBD_RR, [InstrStage<3, [Port0, Port1]>] >,
  InstrItinData<IIC_MMX_PMUL, [InstrStage<4, [Port0]>] >,
  InstrItinData<IIC_MMX_MISC_FUNC_MEM, [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_MMX_MISC_FUNC_REG, [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_MMX_PSADBW,   [InstrStage<4, [Port0, Port1]>] >,
  InstrItinData<IIC_MMX_SHIFT_RI, [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_MMX_SHIFT_RM, [InstrStage<3, [Port0, Port1]>] >,
  InstrItinData<IIC_MMX_SHIFT_RR, [InstrStage<2, [Port0, Port1]>] >,
  InstrItinData<IIC_MMX_UNPCK_H_RM, [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_MMX_UNPCK_H_RR, [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_MMX_UNPCK_L, [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_MMX_PCK_RM,  [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_MMX_PCK_RR,  [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_MMX_PSHUF,   [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_MMX_PEXTR,   [InstrStage<4, [Port0, Port1]>] >,
  InstrItinData<IIC_MMX_PINSRW,  [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_MMX_MASKMOV, [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_MMX_MOVMSK, [InstrStage<3, [Port0]>] >,
  // conversions
  // from/to PD
  InstrItinData<IIC_MMX_CVT_PD_RR, [InstrStage<7, [Port0, Port1]>] >,
  InstrItinData<IIC_MMX_CVT_PD_RM, [InstrStage<8, [Port0, Port1]>] >,
  // from/to PI
  InstrItinData<IIC_MMX_CVT_PS_RR, [InstrStage<5, [Port1]>] >,
  InstrItinData<IIC_MMX_CVT_PS_RM, [InstrStage<5, [Port0], 0>,
                                    InstrStage<5, [Port1]>]>,

  InstrItinData<IIC_CMPX_LOCK, [InstrStage<14, [Port0, Port1]>] >,
  InstrItinData<IIC_CMPX_LOCK_8, [InstrStage<6, [Port0, Port1]>] >,
  InstrItinData<IIC_CMPX_LOCK_8B, [InstrStage<18, [Port0, Port1]>] >,
  InstrItinData<IIC_CMPX_LOCK_16B, [InstrStage<22, [Port0, Port1]>] >,

  InstrItinData<IIC_XADD_LOCK_MEM, [InstrStage<2, [Port0, Port1]>] >,
  InstrItinData<IIC_XADD_LOCK_MEM, [InstrStage<3, [Port0, Port1]>] >,

  InstrItinData<IIC_FILD, [InstrStage<5, [Port0], 0>, InstrStage<5, [Port1]>] >,
  InstrItinData<IIC_FLD,  [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_FLD80, [InstrStage<4, [Port0, Port1]>] >,

  InstrItinData<IIC_FST,   [InstrStage<2, [Port0, Port1]>] >,
  InstrItinData<IIC_FST80, [InstrStage<5, [Port0, Port1]>] >,
  InstrItinData<IIC_FIST,  [InstrStage<6, [Port0, Port1]>] >,

  InstrItinData<IIC_FCMOV,  [InstrStage<9, [Port0, Port1]>] >,
  InstrItinData<IIC_FLDZ,   [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_FUCOM,  [InstrStage<1, [Port1]>] >,
  InstrItinData<IIC_FUCOMI, [InstrStage<9, [Port0, Port1]>] >,
  InstrItinData<IIC_FCOMI,  [InstrStage<9, [Port0, Port1]>] >,
  InstrItinData<IIC_FNSTSW, [InstrStage<10, [Port0, Port1]>] >,
  InstrItinData<IIC_FNSTCW, [InstrStage<8, [Port0, Port1]>] >,
  InstrItinData<IIC_FLDCW,  [InstrStage<5, [Port0, Port1]>] >,
  InstrItinData<IIC_FNINIT, [InstrStage<63, [Port0, Port1]>] >,
  InstrItinData<IIC_FFREE,  [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_FNCLEX, [InstrStage<25, [Port0, Port1]>] >,
  InstrItinData<IIC_WAIT,  [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_FXAM,  [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_FNOP,  [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_FLDL,  [InstrStage<10, [Port0, Port1]>] >,
  InstrItinData<IIC_F2XM1,  [InstrStage<99, [Port0, Port1]>] >,
  InstrItinData<IIC_FYL2X,  [InstrStage<146, [Port0, Port1]>] >,
  InstrItinData<IIC_FPTAN,  [InstrStage<168, [Port0, Port1]>] >,
  InstrItinData<IIC_FPATAN,  [InstrStage<183, [Port0, Port1]>] >,
  InstrItinData<IIC_FXTRACT,  [InstrStage<25, [Port0, Port1]>] >,
  InstrItinData<IIC_FPREM1,  [InstrStage<71, [Port0, Port1]>] >,
  InstrItinData<IIC_FPSTP,  [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_FPREM,  [InstrStage<55, [Port0, Port1]>] >,
  InstrItinData<IIC_FYL2XP1,  [InstrStage<147, [Port0, Port1]>] >,
  InstrItinData<IIC_FSINCOS,  [InstrStage<174, [Port0, Port1]>] >,
  InstrItinData<IIC_FRNDINT,  [InstrStage<46, [Port0, Port1]>] >,
  InstrItinData<IIC_FSCALE,  [InstrStage<77, [Port0, Port1]>] >,
  InstrItinData<IIC_FCOMPP,  [InstrStage<1, [Port1]>] >,
  InstrItinData<IIC_FXSAVE,  [InstrStage<140, [Port0, Port1]>] >,
  InstrItinData<IIC_FXRSTOR,  [InstrStage<141, [Port0, Port1]>] >,
  InstrItinData<IIC_FXCH, [InstrStage<1, [Port0], 0>, InstrStage<1, [Port1]>] >,
  InstrItinData<IIC_FSIGN,  [InstrStage<1, [Port1]>] >,
  InstrItinData<IIC_FSQRT,  [InstrStage<71, [Port0, Port1]>] >,

  // System instructions
  InstrItinData<IIC_CPUID, [InstrStage<121, [Port0, Port1]>] >,
  InstrItinData<IIC_INT,   [InstrStage<127, [Port0, Port1]>] >,
  InstrItinData<IIC_INT3,  [InstrStage<130, [Port0, Port1]>] >,
  InstrItinData<IIC_INVD,  [InstrStage<1003, [Port0, Port1]>] >,
  InstrItinData<IIC_INVLPG, [InstrStage<71, [Port0, Port1]>] >,
  InstrItinData<IIC_IRET,  [InstrStage<109, [Port0, Port1]>] >,
  InstrItinData<IIC_HLT,   [InstrStage<121, [Port0, Port1]>] >,
  InstrItinData<IIC_LXS,   [InstrStage<10, [Port0, Port1]>] >,
  InstrItinData<IIC_LTR,   [InstrStage<83, [Port0, Port1]>] >,
  InstrItinData<IIC_RDTSC, [InstrStage<30, [Port0, Port1]>] >,
  InstrItinData<IIC_RDTSCP, [InstrStage<30, [Port0, Port1]>] >,
  InstrItinData<IIC_RSM,   [InstrStage<741, [Port0, Port1]>] >,
  InstrItinData<IIC_SIDT,  [InstrStage<4, [Port0, Port1]>] >,
  InstrItinData<IIC_SGDT,  [InstrStage<4, [Port0, Port1]>] >,
  InstrItinData<IIC_SLDT,  [InstrStage<3, [Port0, Port1]>] >,
  InstrItinData<IIC_STR,    [InstrStage<3, [Port0, Port1]>] >,
  InstrItinData<IIC_SWAPGS, [InstrStage<22, [Port0, Port1]>] >,
  InstrItinData<IIC_SYSCALL, [InstrStage<96, [Port0, Port1]>] >,
  InstrItinData<IIC_SYS_ENTER_EXIT, [InstrStage<88, [Port0, Port1]>] >,

  InstrItinData<IIC_IN_RR,  [InstrStage<94, [Port0, Port1]>] >,
  InstrItinData<IIC_IN_RI,  [InstrStage<92, [Port0, Port1]>] >,
  InstrItinData<IIC_OUT_RR, [InstrStage<68, [Port0, Port1]>] >,
  InstrItinData<IIC_OUT_IR, [InstrStage<72, [Port0, Port1]>] >,
  InstrItinData<IIC_INS,    [InstrStage<59, [Port0, Port1]>] >,

  InstrItinData<IIC_MOV_REG_DR, [InstrStage<88, [Port0, Port1]>] >,
  InstrItinData<IIC_MOV_DR_REG, [InstrStage<123, [Port0, Port1]>] >,
  // worst case for mov REG_CRx
  InstrItinData<IIC_MOV_REG_CR, [InstrStage<12, [Port0, Port1]>] >,
  InstrItinData<IIC_MOV_CR_REG, [InstrStage<136, [Port0, Port1]>] >,

  InstrItinData<IIC_MOV_REG_SR, [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_MOV_MEM_SR, [InstrStage<2, [Port0, Port1]>] >,
  InstrItinData<IIC_MOV_SR_REG, [InstrStage<21, [Port0, Port1]>] >,
  InstrItinData<IIC_MOV_SR_MEM, [InstrStage<26, [Port0, Port1]>] >,
  // LAR
  InstrItinData<IIC_LAR_RM,  [InstrStage<50, [Port0, Port1]>] >,
  InstrItinData<IIC_LAR_RR,  [InstrStage<54, [Port0, Port1]>] >,
  // LSL
  InstrItinData<IIC_LSL_RM,  [InstrStage<46, [Port0, Port1]>] >,
  InstrItinData<IIC_LSL_RR,  [InstrStage<49, [Port0, Port1]>] >,

  InstrItinData<IIC_LGDT, [InstrStage<44, [Port0, Port1]>] >,
  InstrItinData<IIC_LIDT, [InstrStage<44, [Port0, Port1]>] >,
  InstrItinData<IIC_LLDT_REG, [InstrStage<60, [Port0, Port1]>] >,
  InstrItinData<IIC_LLDT_MEM, [InstrStage<64, [Port0, Port1]>] >,
  // push control register, segment registers
  InstrItinData<IIC_PUSH_CS, [InstrStage<2, [Port0, Port1]>] >,
  InstrItinData<IIC_PUSH_SR, [InstrStage<2, [Port0, Port1]>] >,
  // pop control register, segment registers
  InstrItinData<IIC_POP_SR,    [InstrStage<29, [Port0, Port1]>] >,
  InstrItinData<IIC_POP_SR_SS, [InstrStage<48, [Port0, Port1]>] >,
  // VERR, VERW
  InstrItinData<IIC_VERR,     [InstrStage<41, [Port0, Port1]>] >,
  InstrItinData<IIC_VERW_REG, [InstrStage<51, [Port0, Port1]>] >,
  InstrItinData<IIC_VERW_MEM, [InstrStage<50, [Port0, Port1]>] >,
  // WRMSR, RDMSR
  InstrItinData<IIC_WRMSR, [InstrStage<202, [Port0, Port1]>] >,
  InstrItinData<IIC_RDMSR, [InstrStage<78, [Port0, Port1]>] >,
  InstrItinData<IIC_RDPMC, [InstrStage<46, [Port0, Port1]>] >,
  // SMSW, LMSW
  InstrItinData<IIC_SMSW, [InstrStage<9, [Port0, Port1]>] >,
  InstrItinData<IIC_LMSW_REG, [InstrStage<69, [Port0, Port1]>] >,
  InstrItinData<IIC_LMSW_MEM, [InstrStage<67, [Port0, Port1]>] >,

  InstrItinData<IIC_ENTER, [InstrStage<32, [Port0, Port1]>] >,
  InstrItinData<IIC_LEAVE, [InstrStage<2, [Port0, Port1]>] >,

  InstrItinData<IIC_POP_MEM, [InstrStage<3, [Port0, Port1]>] >,
  InstrItinData<IIC_POP_REG16, [InstrStage<2, [Port0, Port1]>] >,
  InstrItinData<IIC_POP_REG, [InstrStage<1, [Port0], 0>,
                            InstrStage<1, [Port1]>] >,
  InstrItinData<IIC_POP_F, [InstrStage<32, [Port0, Port1]>] >,
  InstrItinData<IIC_POP_FD, [InstrStage<26, [Port0, Port1]>] >,
  InstrItinData<IIC_POP_A, [InstrStage<9, [Port0, Port1]>] >,

  InstrItinData<IIC_PUSH_IMM, [InstrStage<1, [Port0], 0>,
                               InstrStage<1, [Port1]>] >,
  InstrItinData<IIC_PUSH_MEM, [InstrStage<2, [Port0, Port1]>] >,
  InstrItinData<IIC_PUSH_REG, [InstrStage<1, [Port0], 0>,
                               InstrStage<1, [Port1]>] >,
  InstrItinData<IIC_PUSH_F, [InstrStage<9, [Port0, Port1]>] >,
  InstrItinData<IIC_PUSH_A, [InstrStage<8, [Port0, Port1]>] >,

  InstrItinData<IIC_BSWAP, [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_BIT_SCAN_MEM, [InstrStage<16, [Port0, Port1]>] >,
  InstrItinData<IIC_BIT_SCAN_REG, [InstrStage<16, [Port0, Port1]>] >,
  InstrItinData<IIC_MOVS, [InstrStage<3, [Port0, Port1]>] >,
  InstrItinData<IIC_STOS, [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_SCAS, [InstrStage<2, [Port0, Port1]>] >,
  InstrItinData<IIC_CMPS, [InstrStage<3, [Port0, Port1]>] >,
  InstrItinData<IIC_MOV, [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_MOV_MEM, [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_AHF, [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_BT_MI, [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_BT_MR, [InstrStage<9, [Port0, Port1]>] >,
  InstrItinData<IIC_BT_RI, [InstrStage<1, [Port1]>] >,
  InstrItinData<IIC_BT_RR, [InstrStage<1, [Port1]>] >,
  InstrItinData<IIC_BTX_MI, [InstrStage<2, [Port0, Port1]>] >,
  InstrItinData<IIC_BTX_MR, [InstrStage<11, [Port0, Port1]>] >,
  InstrItinData<IIC_BTX_RI, [InstrStage<1, [Port1]>] >,
  InstrItinData<IIC_BTX_RR, [InstrStage<1, [Port1]>] >,
  InstrItinData<IIC_XCHG_REG, [InstrStage<2, [Port0, Port1]>] >,
  InstrItinData<IIC_XCHG_MEM, [InstrStage<3, [Port0, Port1]>] >,
  InstrItinData<IIC_XADD_REG, [InstrStage<2, [Port0, Port1]>] >,
  InstrItinData<IIC_XADD_MEM, [InstrStage<3, [Port0, Port1]>] >,
  InstrItinData<IIC_CMPXCHG_MEM, [InstrStage<14, [Port0, Port1]>] >,
  InstrItinData<IIC_CMPXCHG_REG, [InstrStage<15, [Port0, Port1]>] >,
  InstrItinData<IIC_CMPXCHG_MEM8, [InstrStage<6, [Port0, Port1]>] >,
  InstrItinData<IIC_CMPXCHG_REG8, [InstrStage<9, [Port0, Port1]>] >,
  InstrItinData<IIC_CMPXCHG_8B, [InstrStage<18, [Port0, Port1]>] >,
  InstrItinData<IIC_CMPXCHG_16B, [InstrStage<22, [Port0, Port1]>] >,
  InstrItinData<IIC_LODS, [InstrStage<2, [Port0, Port1]>] >,
  InstrItinData<IIC_OUTS, [InstrStage<74, [Port0, Port1]>] >,
  InstrItinData<IIC_CLC, [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_CLD, [InstrStage<3, [Port0, Port1]>] >,
  InstrItinData<IIC_CLI, [InstrStage<14, [Port0, Port1]>] >,
  InstrItinData<IIC_CMC, [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_CLTS, [InstrStage<33, [Port0, Port1]>] >,
  InstrItinData<IIC_STC, [InstrStage<1, [Port0, Port1]>] >,
  InstrItinData<IIC_STI, [InstrStage<17, [Port0, Port1]>] >,
  InstrItinData<IIC_STD, [InstrStage<21, [Port0, Port1]>] >,
  InstrItinData<IIC_XLAT, [InstrStage<6, [Port0, Port1]>] >,
  InstrItinData<IIC_AAA, [InstrStage<13, [Port0, Port1]>] >,
  InstrItinData<IIC_AAD, [InstrStage<7, [Port0, Port1]>] >,
  InstrItinData<IIC_AAM, [InstrStage<21, [Port0, Port1]>] >,
  InstrItinData<IIC_AAS, [InstrStage<13, [Port0, Port1]>] >,
  InstrItinData<IIC_DAA, [InstrStage<18, [Port0, Port1]>] >,
  InstrItinData<IIC_DAS, [InstrStage<20, [Port0, Port1]>] >,
  InstrItinData<IIC_BOUND, [InstrStage<11, [Port0, Port1]>] >,
  InstrItinData<IIC_ARPL_REG, [InstrStage<24, [Port0, Port1]>] >,
  InstrItinData<IIC_ARPL_MEM, [InstrStage<23, [Port0, Port1]>] >,
  InstrItinData<IIC_MOVBE, [InstrStage<1, [Port0]>] >,
  InstrItinData<IIC_CBW, [InstrStage<4, [Port0, Port1]>] >,
  InstrItinData<IIC_MMX_EMMS, [InstrStage<5, [Port0, Port1]>] >,

  InstrItinData<IIC_NOP, [InstrStage<1, [Port0, Port1]>] >
  ]>;

// Atom machine model.
def AtomModel : SchedMachineModel {
  let IssueWidth = 2;  // Allows 2 instructions per scheduling group.
  let MicroOpBufferSize = 0; // In-order execution, always hide latency.
  let LoadLatency = 3; // Expected cycles, may be overriden by OperandCycles.
  let HighLatency = 30;// Expected, may be overriden by OperandCycles.

  // On the Atom, the throughput for taken branches is 2 cycles. For small
  // simple loops, expand by a small factor to hide the backedge cost.
  let LoopMicroOpBufferSize = 10;
  let PostRAScheduler = 1;
  let CompleteModel = 0;

  let Itineraries = AtomItineraries;
}
