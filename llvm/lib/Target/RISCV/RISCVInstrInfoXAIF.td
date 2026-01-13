//===-- RISCVInstrInfoXAIF.td ------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Instruction definitions for the AI Foundry ET extensions, formerly known as
// the ET-SoC-1 Minion extensions by Esperanto Technologies.
//
// Full documentation for these extensions is publicly available at the
// following URL:
//
//   https://github.com/aifoundry-org/et-man/blob/main/ET%20Programmer's%20Reference%20Manual.pdf
//
// The following conventions are used in this file:
//
// 1. Specific instruction formats are named
//
//        RVInstET<#>
//
//    where <#> is a number from 1 to 14. Each number corresponds to a template
//    fitting a given instruction and other similar ones, as defined in
//    RISCVInstrFormatsXAIF.td
//
// 2. Each masked vector instruction <INST> is handled with an ETOps<Operands>
//    multiclass defining one instruction and two pseudo instructions:
//
//     - <INST>: regular masked vector instructions as described in the manual,
//       implicitly using M0 register for masking out destination register
//       elements.
//
//     - <INST>_EX: masked vector pseudo-instruction explictly accepting M0 as
//       its last operand, as opposed to <INST> where the register dependency
//       is just implicit. In addition to the explicit operands accepted by
//       <INST>, we must add m0 as the last operand.
//
//       The purpose of this instruction variant is to explicitly mark the
//       dependency on the m0 register contents in MIR (Machine Instruction
//       Representation).
//
//     - <INST>_PASSTHRU_EX: Similar to <INST_EX>, but also making explicit the
//       dependency on the previous destination register contents. In addition
//       to the explicit operands accepted by <INST_EX>, we must add the
//       destination register as the first operand.
//
//       Destination and first source operands are constrained to be assigned
//       the same physical register during register allocation.
//
//       This variant should be the choice during instruction selection for
//       cases when m0 is not guaranteed to be all ones.
//
//===----------------------------------------------------------------------===//

let DecoderNamespace = "XAIF", Predicates = [HasXAIFET] in {

  let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in
  def AIF_FLQ2 : RVInstI<0b101, OPC_LOAD_FP,
                        (outs FPR256:$rd),
                        (ins simm12_lo:$imm12, GPR:$rs1),
                        "aif.flq2", "$rd, ${imm12}(${rs1})">;

  let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in
  def AIF_FSQ2 : RVInstS<0b101, OPC_STORE_FP,
                         (outs),
                         (ins FPR256:$rs2, simm12_lo:$imm12, GPR:$rs1),
                         "aif.fsq2", "$rs2, ${imm12}(${rs1})">;

  let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in {
    def AIF_FLW_PS : RVInstI<0b010, OPC_ET_MEM_PS,
                            (outs FPR256:$rd),
                            (ins simm12_lo:$imm12, GPR:$rs1),
                            "aif.flw.ps", "$rd, ${imm12}(${rs1})">;
    def AIF_FLW_PS_EX : Pseudo<(outs FPR256:$rd),
                               (ins simm12_lo:$imm12, GPR:$rs1, MR0:$m0), [],
                               "aif.flw.ps", "$rd, ${imm12}(${rs1}), $m0">;
    def AIF_FLW_PS_PASSTHRU_EX : Pseudo<(outs FPR256:$rd),
                                        (ins FPR256:$in, simm12_lo:$imm12, GPR:$rs1, MR0:$m0), [],
                                        "aif.flw.ps", "$rd, ${imm12}(${rs1}), $m0"> {
      let Constraints = "$rd = $in";
    }
  }

  let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in {
    def AIF_FSW_PS : RVInstS<0b110, OPC_ET_MEM_PS,
                             (outs),
                             (ins FPR256:$rs2, simm12_lo:$imm12, GPR:$rs1),
                             "aif.fsw.ps", "$rs2, ${imm12}(${rs1})">;
    def AIF_FSW_PS_EX : Pseudo<(outs),
                               (ins FPR256:$rs2, simm12_lo:$imm12, GPR:$rs1, MR0:$m0),
                               [], "aif.fsw.ps", "$rs2, ${imm12}(${rs1}), $m0">;
  }

  let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in {
    def AIF_FBC_PS : RVInstI<0b000, OPC_ET_MEM_PS,
                             (outs FPR256:$rd),
                             (ins simm12_lo:$imm12, GPR:$rs1),
                             "aif.fbc.ps", "$rd, ${imm12}(${rs1})">;
    def AIF_FBC_PS_EX : Pseudo<(outs FPR256:$rd),
                               (ins simm12_lo:$imm12, GPR:$rs1, MR0:$m0), [],
                               "aif.fbc.ps", "$rd, ${imm12}(${rs1}), $m0">;
  }

  let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
    def AIF_FBCX_PS : RVInstI<0b011, OPC_ET_MEM_PS,
                              (outs FPR256:$rd),
                              (ins GPR:$rs1),
                              "aif.fbcx.ps", "$rd, ${rs1}"> {
      let imm12 = 0;
    }
    def AIF_FBCX_PS_EX : Pseudo<(outs FPR256:$rd),
                                (ins GPR:$rs1, MR0:$m0), [],
                                "aif.fbcx.ps", "$rd, ${rs1}, $m0">;
    def AIF_FBCX_PS_PASSTHRU_EX : Pseudo<(outs FPR256:$rd),
                                         (ins FPR256:$in, GPR:$rs1, MR0:$m0), [],
                                         "aif.fbcx.ps", "$rd, ${rs1}, $m0"> {
      let Constraints = "$rd = $in";
    }

    def AIF_FBCI_PS : RVInstU<OPC_ET_IMM_PS,
                              (outs FPR256:$rd),
                              (ins uimm20_lui:$imm20),
                              "aif.fbci.ps", "$rd, $imm20"> ;

    def AIF_FBCI_PS_EX : Pseudo<(outs FPR256:$rd),
                                (ins uimm20_lui:$imm20, MR0:$m0), [],
                                "aif.fbci.ps", "$rd, $imm20, $m0">;
    def AIF_FBCI_PS_PASSTHRU_EX : Pseudo<(outs FPR256:$rd),
                                         (ins FPR256:$in, uimm20_lui:$imm20, MR0:$m0), [],
                                         "aif.fbci.ps", "$rd, $imm20, $m0"> {
      let Constraints = "$rd = $in";
    }

    multiclass ETOpsRdF1R2G<bits<7> funct7, string opcodestr> {
      def "" : RVInstR<funct7, 0b001, OPC_ET_MEM_PS,
                       (outs FPR256:$rd),
                       (ins FPR256:$rs1, GPRMemZeroOffset:$rs2),
                       opcodestr, "$rd, ${rs1}, ${rs2}">;
      def _EX : Pseudo<(outs FPR256:$rd),
                       (ins FPR256:$rs1, GPRMemZeroOffset:$rs2, MR0:$m0), [],
                       opcodestr, "$rd, ${rs1}, ${rs2}, $m0">;
      def _PASSTHRU_EX : Pseudo<(outs FPR256:$rd),
                                (ins FPR256:$in, FPR256:$rs1, GPRMemZeroOffset:$rs2, MR0:$m0), [],
                                  opcodestr, "$rd, ${rs1}, ${rs2}, $m0"> {
        let Constraints = "$rd = $in";
      }
    }

    let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in {
      defm AIF_FGW_PS : ETOpsRdF1R2G<0b0110000, "aif.fgw.ps">;
      defm AIF_FGH_PS : ETOpsRdF1R2G<0b0101000, "aif.fgh.ps">;
      defm AIF_FGB_PS : ETOpsRdF1R2G<0b0100100, "aif.fgb.ps">;
    }

    multiclass ETOpsR3R1R2<bits<7> funct7, bits<3> funct3, string opcodestr> {
      def "" : RVInstET12<funct7, funct3, OPC_ET_MEM_PS,
                          (outs),
                          (ins FPR256:$rs3, FPR256:$rs1, GPRMemZeroOffset:$rs2),
                          opcodestr, "$rs3, ${rs1}, ${rs2}">;
      def _EX : Pseudo<(outs),
                       (ins FPR256:$rs3, FPR256:$rs1, GPRMemZeroOffset:$rs2, MR0:$m0), [],
                       opcodestr, "$rs3, ${rs1}, ${rs2}, $m0">;
    }

    let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in {
      defm AIF_FSCW_PS : ETOpsR3R1R2<0b1110000, 0b001, "aif.fscw.ps">;
      defm AIF_FSCH_PS : ETOpsR3R1R2<0b1101000, 0b001, "aif.fsch.ps">;
      defm AIF_FSCB_PS : ETOpsR3R1R2<0b1100100, 0b001, "aif.fscb.ps">;
    }

    multiclass ETOpsRdR1R2Rm<bits<7> funct7, RISCVOpcode opcode,
                             string opcodestr> {
      def "" : RVInstET3<funct7, opcode,
                         (outs FPR256:$rd),
                         (ins FPR256:$rs1, FPR256:$rs2, frmarg:$rm),
                         opcodestr, "$rd, $rs1, $rs2$rm">;
      def _EX : Pseudo<(outs FPR256:$rd),
                       (ins FPR256:$rs1, FPR256:$rs2, frmarg:$rm, MR0:$m0),
                       [], opcodestr, "$rd, $rs1, $rs2$rm, $m0">;
      def _PASSTHRU_EX : Pseudo<(outs FPR256:$rd),
                                (ins FPR256:$in, FPR256:$rs1, FPR256:$rs2, frmarg:$rm, MR0:$m0),
                                [], opcodestr, "$rd, $rs1, $rs2$rm, $m0">  {
        let Constraints = "$rd = $in";
      }
    }

    defm AIF_FADD_PS : ETOpsRdR1R2Rm<0b0000000, OPC_ET_OP_PS, "aif.fadd.ps">;
    defm AIF_FSUB_PS : ETOpsRdR1R2Rm<0b0000100, OPC_ET_OP_PS, "aif.fsub.ps">;
    defm AIF_FMUL_PS : ETOpsRdR1R2Rm<0b0001000, OPC_ET_OP_PS, "aif.fmul.ps">;
    defm AIF_FDIV_PS : ETOpsRdR1R2Rm<0b0001100, OPC_ET_OP_PS, "aif.fdiv.ps">;

    multiclass ETOpsRdR1R2<bits<7> funct7, bits<3> funct3, RISCVOpcode opcode,
                           string opcodestr> {
      def "" : RVInstR<funct7, funct3, opcode,
                       (outs FPR256:$rd),
                       (ins FPR256:$rs1, FPR256:$rs2),
                       opcodestr, "$rd, $rs1, $rs2">;
      def _EX : Pseudo<(outs FPR256:$rd),
                       (ins FPR256:$rs1, FPR256:$rs2, MR0:$m0),
                       [], opcodestr, "$rd, $rs1, $rs2, $m0">;
      def _PASSTHRU_EX : Pseudo<(outs FPR256:$rd),
                                (ins FPR256:$in, FPR256:$rs1, FPR256:$rs2, MR0:$m0),
                                [], opcodestr, "$rd, $rs1, $rs2, $m0">  {
        let Constraints = "$rd = $in";
      }
    }

    defm AIF_FMIN_PS : ETOpsRdR1R2<0b0010100, 0b000, OPC_ET_OP_PS, "aif.fmin.ps">;
    defm AIF_FMAX_PS : ETOpsRdR1R2<0b0010100, 0b001, OPC_ET_OP_PS, "aif.fmax.ps">;

    multiclass ETOpsRdR1R2R3Rm<bits<2> funct2, RISCVOpcode opcode,
                               string opcodestr> {
      def "" : RVInstET4<funct2, opcode,
                         (outs FPR256:$rd),
                         (ins FPR256:$rs1, FPR256:$rs2, FPR256:$rs3, frmarg:$rm),
                         opcodestr, "$rd, $rs1, $rs2, $rs3$rm">;
      def _EX : Pseudo<(outs FPR256:$rd),
                       (ins FPR256:$rs1,
                         FPR256:$rs2, FPR256:$rs3, frmarg:$rm, MR0:$m0),
                       [], opcodestr, "$rd, $rs1, $rs2, $rs3$rm, $m0">;
      def _PASSTHRU_EX : Pseudo<(outs FPR256:$rd),
                                (ins FPR256:$in, FPR256:$rs1,
                                FPR256:$rs2, FPR256:$rs3, frmarg:$rm, MR0:$m0),
                                [], opcodestr, "$rd, $rs1, $rs2, $rs3$rm, $m0"> {
        let Constraints = "$rd = $in";
      }
    }

    defm AIF_FMADD_PS  : ETOpsRdR1R2R3Rm<0b00, OPC_ET_OP3_PS, "aif.fmadd.ps">;
    defm AIF_FMSUB_PS  : ETOpsRdR1R2R3Rm<0b01, OPC_ET_OP3_PS, "aif.fmsub.ps">;
    defm AIF_FNMSUB_PS : ETOpsRdR1R2R3Rm<0b10, OPC_ET_OP3_PS, "aif.fnmsub.ps">;
    defm AIF_FNMADD_PS : ETOpsRdR1R2R3Rm<0b11, OPC_ET_OP3_PS, "aif.fnmadd.ps">;

    def AIF_FCMOV_PS : RVInstET4<0b10, OPC_ET_IMM10_PI,
                                 (outs FPR256:$rd),
                                 (ins FPR256:$rs1, FPR256:$rs2, FPR256:$rs3),
                                 "aif.fcmov.ps", "$rd, $rs1, $rs2, $rs3"> {
      let rm = 0b010;
    }

    def AIF_FCMOV_PS_EX : Pseudo<(outs FPR256:$rd),
                                 (ins FPR256:$rs1, FPR256:$rs2,
                                   FPR256:$rs3, MR0:$m0), [],
                                 "aif.fcmov.ps", "$rd, $rs1, $rs2, $rs3, $m0">;
    def AIF_FCMOV_PS_PASSTHRU_EX : Pseudo<(outs FPR256:$rd),
                                          (ins FPR256:$in, FPR256:$rs1, FPR256:$rs2,
                                          FPR256:$rs3, MR0:$m0), [],
                                          "aif.fcmov.ps", "$rd, $rs1, $rs2, $rs3, $m0"> {
      let Constraints = "$rd = $in";
    }

    def AIF_FSWIZZ_PS : RVInstET1<0b1110011, OPC_ET_OP_PS,
                                  (outs FPR256:$rd),
                                  (ins FPR256:$rs1, uimm8:$imm),
                                  "aif.fswizz.ps", "$rd, $rs1, $imm">;
    def AIF_FSWIZZ_PS_EX : Pseudo<(outs FPR256:$rd),
                                  (ins FPR256:$rs1, uimm8:$imm, MR0:$m0),
                                  [], "aif.fswizz.ps", "$rd, $rs1, $imm, $m0">;
    def AIF_FSWIZZ_PS_PASSTHRU_EX : Pseudo<(outs FPR256:$rd),
                                           (ins FPR256:$in, FPR256:$rs1, uimm8:$imm, MR0:$m0),
                                           [], "aif.fswizz.ps", "$rd, $rs1, $imm, $m0"> {
      let Constraints = "$rd = $in";
    }

    multiclass ETOpsRdR1Rm<bits<7> funct7, bits<5> funct1, RISCVOpcode opcode,
                           string opcodestr> {
      def "" : RVInstET3<funct7, opcode,
                         (outs FPR256:$rd),
                         (ins FPR256:$rs1, frmarg:$rm),
                         opcodestr, "$rd, $rs1$rm"> {
        let rs2 = funct1;
      }
      def _EX : Pseudo<(outs FPR256:$rd),
                       (ins FPR256:$rs1, frmarg:$rm, MR0:$m0),
                       [], opcodestr, "$rd, $rs1$rm, $m0">;
      def _PASSTHRU_EX : Pseudo<(outs FPR256:$rd),
                                (ins FPR256:$in, FPR256:$rs1, frmarg:$rm, MR0:$m0),
                                [], opcodestr, "$rd, $rs1$rm, $m0">  {
        let Constraints = "$rd = $in";
      }
    }

    defm AIF_FCVT_PW_PS  : ETOpsRdR1Rm<0b1100000, 0, OPC_ET_OP_PS, "aif.fcvt.pw.ps">;
    defm AIF_FCVT_PWU_PS : ETOpsRdR1Rm<0b1100000, 1, OPC_ET_OP_PS, "aif.fcvt.pwu.ps">;
    defm AIF_FCVT_PS_PW  : ETOpsRdR1Rm<0b1101000, 0, OPC_ET_OP_PS, "aif.fcvt.ps.pw">;
    defm AIF_FCVT_PS_PWU : ETOpsRdR1Rm<0b1101000, 1, OPC_ET_OP_PS, "aif.fcvt.ps.pwu">;

    class ETOpsRdR1Idx<bits<7> funct7, bits<3> funct3, RISCVOpcode opcode,
                       string opcodestr> :
      RVInstET5<funct7, funct3, opcode,
                (outs GPR:$rd),
                (ins FPR256:$rs1, uimm3:$idx),
                opcodestr, "$rd, $rs1, $idx">;

    def AIF_FMVZ_X_PS : ETOpsRdR1Idx<0b1110000, 0b000, OPC_ET_OP_PS, "aif.fmvz.x.ps">;
    // no _EX variant
    def AIF_FMVS_X_PS : ETOpsRdR1Idx<0b1110000, 0b010, OPC_ET_OP_PS, "aif.fmvs.x.ps">;
    // no _EX variant

    defm AIF_FSGNJN_PS : ETOpsRdR1R2<0b0010000, 0b001, OPC_ET_OP_PS, "aif.fsgnjn.ps">;
    defm AIF_FSGNJ_PS  : ETOpsRdR1R2<0b0010000, 0b000, OPC_ET_OP_PS, "aif.fsgnj.ps">;
    defm AIF_FSGNJX_PS : ETOpsRdR1R2<0b0010000, 0b010, OPC_ET_OP_PS, "aif.fsgnjx.ps">;
    defm AIF_FLE_PS    : ETOpsRdR1R2<0b1010000, 0b000, OPC_ET_OP_PS, "aif.fle.ps">;
    defm AIF_FLT_PS    : ETOpsRdR1R2<0b1010000, 0b001, OPC_ET_OP_PS, "aif.flt.ps">;
    defm AIF_FEQ_PS    : ETOpsRdR1R2<0b1010000, 0b010, OPC_ET_OP_PS, "aif.feq.ps">;

    multiclass ETOpsMdR1R2<bits<7> funct7, bits<3> funct3, RISCVOpcode opcode,
                           string opcodestr> {
      def "" : RVInstR<funct7, funct3, opcode,
                       (outs MR:$rd),
                       (ins FPR256:$rs1, FPR256:$rs2),
                       opcodestr, "$rd, $rs1, $rs2">;
      def _EX : Pseudo<(outs MR:$rd),
                       (ins FPR256:$rs1, FPR256:$rs2, MR0:$m0), [],
                       opcodestr, "$rd, $rs1, $rs2, $m0">;
      def _PASSTHRU_EX : Pseudo<(outs MR:$rd),
                                (ins MR:$in, FPR256:$rs1, FPR256:$rs2, MR0:$m0), [],
                                opcodestr, "$rd, $rs1, $rs2, $m0"> {
        let Constraints = "$rd = $in";
      }
    }

    defm AIF_FLEM_PS : ETOpsMdR1R2<0b1010000, 0b100, OPC_ET_OP_PS, "aif.flem.ps">;
    defm AIF_FLTM_PS : ETOpsMdR1R2<0b1010000, 0b101, OPC_ET_OP_PS, "aif.fltm.ps">;
    defm AIF_FEQM_PS : ETOpsMdR1R2<0b1010000, 0b110, OPC_ET_OP_PS, "aif.feqm.ps">;

    multiclass ETOpsRdR1<bits<7> funct7, bits<5> funct5, bits<3> funct3,
                         RISCVOpcode opcode, string opcodestr> {
      def "" : RVInstR<funct7, funct3, opcode,
                       (outs FPR256:$rd),
                       (ins FPR256:$rs1),
                       opcodestr, "$rd, $rs1"> {
        let rs2 = funct5;
      }
      def _EX : Pseudo<(outs FPR256:$rd),
                       (ins FPR256:$in, FPR256:$rs1, MR0:$m0), [],
                       opcodestr, "$rd, $rs1, $m0">;
      def _PASSTHRU_EX : Pseudo<(outs FPR256:$rd),
                                (ins FPR256:$in, FPR256:$rs1, MR0:$m0), [],
                                opcodestr, "$rd, $rs1, $m0"> {
        let Constraints = "$rd = $in";
      }
    }

    defm AIF_FCLASS_PS : ETOpsRdR1<0b1110000, 0b00000, 0b001, OPC_ET_OP_PS, "aif.fclass.ps">;

    multiclass ETOpsCvt<bits<7> funct7, bits<5> funct5, string opcodestr> {
      defm "" : ETOpsRdR1<funct7, funct5, 0b000,
                          OPC_ET_OP_PS, opcodestr>;
    }

    defm AIF_FCVT_PS_F10  : ETOpsCvt<0b1101000, 0b01000, "aif.fcvt.ps.f10">;
    defm AIF_FCVT_PS_F11  : ETOpsCvt<0b1101000, 0b01001, "aif.fcvt.ps.f11">;
    defm AIF_FCVT_PS_F16  : ETOpsCvt<0b1101000, 0b01010, "aif.fcvt.ps.f16">;
    defm AIF_FCVT_PS_UN24 : ETOpsCvt<0b1101000, 0b10000, "aif.fcvt.ps.un24">;
    defm AIF_FCVT_PS_UN16 : ETOpsCvt<0b1101000, 0b10001, "aif.fcvt.ps.un16">;
    defm AIF_FCVT_PS_UN10 : ETOpsCvt<0b1101000, 0b10010, "aif.fcvt.ps.un10">;
    defm AIF_FCVT_PS_UN8  : ETOpsCvt<0b1101000, 0b10011, "aif.fcvt.ps.un8">;
    defm AIF_FCVT_PS_UN2  : ETOpsCvt<0b1101000, 0b10111, "aif.fcvt.ps.un2">;
    defm AIF_FCVT_PS_SN16 : ETOpsCvt<0b1101000, 0b11001, "aif.fcvt.ps.sn16">;
    defm AIF_FCVT_PS_SN8  : ETOpsCvt<0b1101000, 0b11011, "aif.fcvt.ps.sn8">;
    defm AIF_FCVT_F11_PS  : ETOpsCvt<0b1101100, 0b01000, "aif.fcvt.f11.ps">;
    defm AIF_FCVT_F16_PS  : ETOpsCvt<0b1101100, 0b01001, "aif.fcvt.f16.ps">;
    defm AIF_FCVT_F10_PS  : ETOpsCvt<0b1101100, 0b01011, "aif.fcvt.f10.ps">;
    defm AIF_FCVT_UN24_PS : ETOpsCvt<0b1101100, 0b10000, "aif.fcvt.un24.ps">;
    defm AIF_FCVT_UN16_PS : ETOpsCvt<0b1101100, 0b10001, "aif.fcvt.un16.ps">;
    defm AIF_FCVT_UN10_PS : ETOpsCvt<0b1101100, 0b10010, "aif.fcvt.un10.ps">;
    defm AIF_FCVT_UN8_PS  : ETOpsCvt<0b1101100, 0b10011, "aif.fcvt.un8.ps">;
    defm AIF_FCVT_UN2_PS  : ETOpsCvt<0b1101100, 0b10111, "aif.fcvt.un2.ps">;
    defm AIF_FCVT_SN16_PS : ETOpsCvt<0b1101100, 0b11001, "aif.fcvt.sn16.ps">;
    defm AIF_FCVT_SN8_PS  : ETOpsCvt<0b1101100, 0b11011, "aif.fcvt.sn8.ps">;

    def AIF_FBCI_PI : RVInstU<OPC_ET_IMM_PI,
                              (outs FPR256:$rd),
                              (ins uimm20_lui:$imm20),
                              "aif.fbci.pi", "$rd, ${imm20}">;
    def AIF_FBCI_PI_EX : Pseudo<(outs FPR256:$rd),
                                (ins uimm20_lui:$imm20, MR0:$m0), [],
                                "aif.fbci.pi", "$rd, ${imm20}, $m0">;
    def AIF_FBCI_PI_PASSTHRU_EX : Pseudo<(outs FPR256:$rd),
                                         (ins FPR256:$in, uimm20_lui:$imm20, MR0:$m0), [],
                                         "aif.fbci.pi", "$rd, ${imm20}, $m0"> {
      let Constraints = "$rd = $in";
    }

    defm AIF_FADD_PI   : ETOpsRdR1R2<0b0000011, 0b000, OPC_ET_OP_PS, "aif.fadd.pi">;
    defm AIF_FSUB_PI   : ETOpsRdR1R2<0b0000111, 0b000, OPC_ET_OP_PS, "aif.fsub.pi">;
    defm AIF_FMUL_PI   : ETOpsRdR1R2<0b0001011, 0b000, OPC_ET_OP_PS, "aif.fmul.pi">;
    defm AIF_FMULH_PI  : ETOpsRdR1R2<0b0001011, 0b001, OPC_ET_OP_PS, "aif.fmulh.pi">;
    defm AIF_FMULHU_PI : ETOpsRdR1R2<0b0001011, 0b010, OPC_ET_OP_PS, "aif.fmulhu.pi">;
    defm AIF_FDIV_PI   : ETOpsRdR1R2<0b0001111, 0b000, OPC_ET_OP_PS, "aif.fdiv.pi">;
    defm AIF_FDIVU_PI  : ETOpsRdR1R2<0b0001111, 0b001, OPC_ET_OP_PS, "aif.fdivu.pi">;
    defm AIF_FREM_PI   : ETOpsRdR1R2<0b0001111, 0b010, OPC_ET_OP_PS, "aif.frem.pi">;
    defm AIF_FREMU_PI  : ETOpsRdR1R2<0b0001111, 0b011, OPC_ET_OP_PS, "aif.fremu.pi">;
    defm AIF_FMIN_PI   : ETOpsRdR1R2<0b0010111, 0b000, OPC_ET_OP_PS, "aif.fmin.pi">;
    defm AIF_FMAX_PI   : ETOpsRdR1R2<0b0010111, 0b001, OPC_ET_OP_PS, "aif.fmax.pi">;
    defm AIF_FMINU_PI  : ETOpsRdR1R2<0b0010111, 0b010, OPC_ET_OP_PS, "aif.fminu.pi">;
    defm AIF_FMAXU_PI  : ETOpsRdR1R2<0b0010111, 0b011, OPC_ET_OP_PS, "aif.fmaxu.pi">;

    defm AIF_FLTM_PI : ETOpsMdR1R2<0b0011111, 0b000, OPC_ET_OP_PS, "aif.fltm.pi">;

    defm AIF_FAND_PI : ETOpsRdR1R2<0b0000011, 0b111, OPC_ET_OP_PS, "aif.fand.pi">;
    defm AIF_FOR_PI  : ETOpsRdR1R2<0b0000011, 0b110, OPC_ET_OP_PS, "aif.for.pi">;
    defm AIF_FNOT_PI : ETOpsRdR1<0b0000011, 0b00000, 0b010, OPC_ET_OP_PS, "aif.fnot.pi">;
    defm AIF_FXOR_PI : ETOpsRdR1R2<0b0000011, 0b100, OPC_ET_OP_PS, "aif.fxor.pi">;
    defm AIF_FSLL_PI : ETOpsRdR1R2<0b0000011, 0b001, OPC_ET_OP_PS, "aif.fsll.pi">;
    defm AIF_FSRL_PI : ETOpsRdR1R2<0b0000011, 0b101, OPC_ET_OP_PS, "aif.fsrl.pi">;
    defm AIF_FSRA_PI : ETOpsRdR1R2<0b0000111, 0b101, OPC_ET_OP_PS, "aif.fsra.pi">;
    defm AIF_FLE_PI  : ETOpsRdR1R2<0b1010011, 0b000, OPC_ET_OP_PS, "aif.fle.pi">;
    defm AIF_FLT_PI  : ETOpsRdR1R2<0b1010011, 0b001, OPC_ET_OP_PS, "aif.flt.pi">;
    defm AIF_FEQ_PI  : ETOpsRdR1R2<0b1010011, 0b010, OPC_ET_OP_PS, "aif.feq.pi">;
    defm AIF_FLTU_PI : ETOpsRdR1R2<0b1010011, 0b011, OPC_ET_OP_PS, "aif.fltu.pi">;

    multiclass ETOpsMdR1<bits<7> funct7, bits<3> funct3,
                         RISCVOpcode opcode, string opcodestr> {
      def "" : RVInstR<funct7, funct3, opcode,
                       (outs MR:$rd),
                       (ins FPR256:$rs1),
                       opcodestr, "$rd, $rs1"> {
        let rs2 = 0;
      }
      def _EX : Pseudo<(outs MR:$rd),
                       (ins FPR256:$rs1, MR0:$m0),
                       [], opcodestr, "$rd, $rs1, $m0">;
      def _PASSTHRU_EX : Pseudo<(outs MR:$rd),
                                (ins MR:$in, FPR256:$rs1, MR0:$m0),
                                [], opcodestr, "$rd, $rs1, $m0"> {
        let Constraints = "$rd = $in";
      }
    }

    defm AIF_FSETM_PI : ETOpsMdR1<0b1010011, 0b100, OPC_ET_OP_PS, "aif.fsetm.pi">;

    multiclass ETOpsRdR1I10<bits<3> funct3, RISCVOpcode opcode, string opcodestr> {
      def "" : RVInstET2<funct3, opcode,
                         (outs FPR256:$rd),
                         (ins FPR256:$rs1, simm10:$imm),
                         opcodestr, "$rd, $rs1, ${imm}">;
      def _EX : Pseudo<(outs FPR256:$rd),
                       (ins FPR256:$in, FPR256:$rs1, simm10:$imm, MR0:$m0), [],
                       opcodestr, "$rd, $rs1, ${imm}, $m0">;
      def _PASSTHRU_EX : Pseudo<(outs FPR256:$rd),
                                (ins FPR256:$in, FPR256:$rs1, simm10:$imm, MR0:$m0), [],
                                opcodestr, "$rd, $rs1, ${imm}, $m0"> {
        let Constraints = "$rd = $in";
      }
    }

    defm AIF_FADDI_PI : ETOpsRdR1I10<0b000, OPC_ET_IMM10_PI, "aif.faddi.pi">;
    defm AIF_FANDI_PI : ETOpsRdR1I10<0b001, OPC_ET_IMM10_PI, "aif.fandi.pi">;

    multiclass ETOpsRdR1I5<bits<3> funct3, RISCVOpcode opcode, string opcodestr> {
      def "" : RVInstET6<funct3, opcode,
                         (outs FPR256:$rd),
                         (ins FPR256:$rs1, uimm5:$imm),
                         opcodestr, "$rd, $rs1, ${imm}">;
      def _EX : Pseudo<(outs FPR256:$rd),
                       (ins FPR256:$in, FPR256:$rs1, uimm5:$imm, MR0:$m0), [],
                       opcodestr, "$rd, $rs1, ${imm}, $m0">;
      def _PASSTHRU_EX : Pseudo<(outs FPR256:$rd),
                                (ins FPR256:$in, FPR256:$rs1, uimm5:$imm, MR0:$m0), [],
                                opcodestr, "$rd, $rs1, ${imm}, $m0"> {
        let Constraints = "$rd = $in";
      }
    }

    defm AIF_FSLLI_PI : ETOpsRdR1I5<0b001, OPC_ET_OP_PS, "aif.fslli.pi">;
    defm AIF_FSRLI_PI : ETOpsRdR1I5<0b101, OPC_ET_OP_PS, "aif.fsrli.pi">;
    defm AIF_FSRAI_PI : ETOpsRdR1I5<0b111, OPC_ET_OP_PS, "aif.fsrai.pi">;

    defm AIF_FSIN_PS : ETOpsRdR1<0b0101100, 0b00110, 0b000, OPC_ET_OP_PS, "aif.fsin.ps">;
    defm AIF_FEXP_PS : ETOpsRdR1<0b0101100, 0b00100, 0b000, OPC_ET_OP_PS, "aif.fexp.ps">;
    defm AIF_FLOG_PS : ETOpsRdR1<0b0101100, 0b00011, 0b000, OPC_ET_OP_PS, "aif.flog.ps">;
    defm AIF_FFRC_PS : ETOpsRdR1<0b0101100, 0b00010, 0b000, OPC_ET_OP_PS, "aif.ffrc.ps">;

    def AIF_FROUND_PS : RVInstET7<OPC_ET_OP_PS,
                                  (outs FPR256:$rd),
                                  (ins FPR256:$rs1, frmarg:$rm),
                                  "aif.fround.ps", "$rd, $rs1$rm">;
    def AIF_FROUND_PS_EX : Pseudo<(outs FPR256:$rd),
                                  (ins FPR256:$rs1, frmarg:$rm, MR0:$m0),
                                  [], "aif.fround.ps", "$rd, $rs1$rm, $m0">;

    defm AIF_FSQRT_PS : ETOpsRdR1<0b0101100, 0b00000, 0b000, OPC_ET_OP_PS, "aif.fsqrt.ps">;
    defm AIF_FRCP_PS  : ETOpsRdR1<0b0101100, 0b00111, 0b000, OPC_ET_OP_PS, "aif.frcp.ps">;
    defm AIF_FRSQ_PS  : ETOpsRdR1<0b0101100, 0b01000, 0b000, OPC_ET_OP_PS, "aif.frsq.ps">;

    defm AIF_CUBEFACE_PS    : ETOpsRdR1R2<0b1000100, 0b000, OPC_ET_OP_PS, "aif.cubeface.ps">;
    defm AIF_CUBEFACEIDX_PS : ETOpsRdR1R2<0b1000100, 0b001, OPC_ET_OP_PS, "aif.cubefaceidx.ps">;
    defm AIF_CUBESGNSC_PS   : ETOpsRdR1R2<0b1000100, 0b010, OPC_ET_OP_PS, "aif.cubesgnsc.ps">;
    defm AIF_CUBESGNTC_PS   : ETOpsRdR1R2<0b1000100, 0b011, OPC_ET_OP_PS, "aif.cubesgntc.ps">;

    def AIF_PACKB   : ALUW_rr<0b1000000, 0b110, "aif.packb">;
    def AIF_BITMIXB : ALUW_rr<0b1000000, 0b111, "aif.bitmixb">;

    def AIF_MOV_M_X : RVInstET8<OPC_ET_OP_PS,
                                (outs MR:$rd),
                                (ins GPR:$rs1, uimm8:$imm),
                                "aif.mov.m.x", "$rd, $rs1, $imm">;
    def AIF_MOVA_X_M : RVInstR<0b1101011, 0b000, OPC_ET_OP_PS,
                               (outs GPR:$rd),
                               (ins),
                               "aif.mova.x.m", "$rd"> {
      let rs1 = 0;
      let rs2 = 0;
    }

    def AIF_MOVA_M_X : RVInstR<0b1101011, 0b001, OPC_ET_OP_PS,
                               (outs),
                               (ins GPR:$rs1),
                               "aif.mova.m.x", "$rs1"> {
      let rd = 0;
      let rs2 = 0;
    }

    class ETOpsMdM1M2<bits<3> funct3, string codestr> :
      RVInstET9<funct3, OPC_ET_OP_PS, (outs MR:$rd),
                (ins MR:$rs1, MR:$rs2), codestr, "$rd, $rs1, $rs2">;

    def AIF_MASKAND : ETOpsMdM1M2<0b111, "aif.maskand">;
    def AIF_MASKOR  : ETOpsMdM1M2<0b110, "aif.maskor">;
    def AIF_MASKNOT : RVInstET9<0b010, OPC_ET_OP_PS, (outs MR:$rd),
                                (ins MR:$rs1), "aif.masknot", "$rd, $rs1"> {
      let rs2 = 0;
    }
    def AIF_MASKXOR  : ETOpsMdM1M2<0b100, "aif.maskxor">;

    class ETOpsRdM1<bits<2> funct2, string opcodestr> :
      RVInstET10<funct2, OPC_ET_OP_PS, (outs GPR:$rd), (ins MR:$rs1),
                 opcodestr, "$rd, $rs1">;

    def AIF_MASKPOPC  : ETOpsRdM1<0b01, "aif.maskpopc">;
    def AIF_MASKPOPCZ : ETOpsRdM1<0b10, "aif.maskpopcz">;

    def AIF_MASKPOPC_ET_RAST :
      RVInstET11<OPC_ET_OP_PS, (outs MR:$rd),
                 (ins MR:$rs1, MR:$rs2, uimm4:$imm),
                 "aif.maskpopc.rast", "$rd, $rs1, $rs2, $imm">;

    defm AIF_FPACKREPB_PI :
      ETOpsRdR1<0b0010011, 0b00000, 0b000, OPC_ET_OP_PS, "aif.fpackrepb.pi">;
    defm AIF_FPACKREPH_PI :
      ETOpsRdR1<0b0010011, 0b00000, 0b001, OPC_ET_OP_PS, "aif.fpackreph.pi">;

    multiclass ETOpsRdR1R2G32<bits<7> funct7, string opcodestr> {
      def "" : RVInstR<funct7, 0b001, OPC_ET_MEM_PS,
                       (outs FPR256:$rd),
                       (ins GPR:$rs1, GPRMemZeroOffset:$rs2),
                       opcodestr, "$rd, ${rs1}, ${rs2}">;
      def _EX : Pseudo<(outs FPR256:$rd),
                       (ins GPR:$rs1, GPRMemZeroOffset:$rs2, MR0:$m0), [],
                       opcodestr, "$rd, ${rs1}, ${rs2}, m0">;
    }

    let mayLoad = 1 in {
      defm AIF_FG32W_PS : ETOpsRdR1R2G32<0b0010000, "aif.fg32w.ps">;
      defm AIF_FG32H_PS : ETOpsRdR1R2G32<0b0001000, "aif.fg32h.ps">;
      defm AIF_FG32B_PS : ETOpsRdR1R2G32<0b0000100, "aif.fg32b.ps">;
    }

    multiclass ETOpsR3R1R2SC32<bits<7> funct7, string opcodestr> {
      def "" : RVInstET12<funct7, 0b001, OPC_ET_MEM_PS,
                          (outs),
                          (ins FPR256:$rs3, GPR:$rs1, GPRMemZeroOffset:$rs2),
                          opcodestr, "$rs3, ${rs1}, ${rs2}">;
     def _EX : Pseudo<(outs),
                      (ins FPR256:$rs3, GPR:$rs1, GPRMemZeroOffset:$rs2, MR0:$m0), [],
                      opcodestr, "$rs3, ${rs1}, ${rs2}, $m0">;
    }

    let mayStore = 1 in {
      defm AIF_FSC32W_PS : ETOpsR3R1R2SC32<0b1010000, "aif.fsc32w.ps">;
      defm AIF_FSC32H_PS : ETOpsR3R1R2SC32<0b1001000, "aif.fsc32h.ps">;
      defm AIF_FSC32B_PS : ETOpsR3R1R2SC32<0b1000100, "aif.fsc32b.ps">;
    }

    // Specialse FCMOVM has no implicit input since all of output
    // is defined.
    def AIF_FCMOVM_PS : RVInstR<0b0000000, 0b000, OPC_ET_CVT_PS,
                                (outs FPR256:$rd),
                                (ins FPR256:$rs1, FPR256:$rs2),
                                "aif.fcmovm.ps", "$rd, $rs1, $rs2">;
    def AIF_FCMOVM_PS_EX : Pseudo<(outs FPR256:$rd),
                                  (ins FPR256:$rs1, FPR256:$rs2, MR0:$m0), [],
                                  "aif.fcmovm.ps", "$rd, $rs1, $rs2, $m0">;

    defm AIF_FRCP_FIX_RAST : ETOpsRdR1R2<0b0011000, 0b000, OPC_ET_OP_PS, "aif.frcp_fix.rast">;
    defm AIF_FCVT_RAST_PS : ETOpsRdR1<0b1100000, 0b00010, 0b000, OPC_ET_OP_PS, "aif.fcvt.rast.ps">;
    defm AIF_FCVT_PS_RAST : ETOpsRdR1<0b1101000, 0b00010, 0b000, OPC_ET_OP_PS, "aif.fcvt.ps.rast">;
    defm AIF_FSAT8_PI : ETOpsRdR1<0b0000011, 0b00000, 0b011, OPC_ET_OP_PS, "aif.fsat8.pi">;
    defm AIF_FSATU8_PI : ETOpsRdR1<0b0000011, 0b00001, 0b011, OPC_ET_OP_PS, "aif.fsatu8.pi">;

    multiclass ETOpsFdR1<bits<7> funct7, string opcodestr> {
      def "" : RVInstR<funct7, 0b111, OPC_ET_MEM_PS,
                       (outs FPR256:$rd), (ins GPR:$rs1),
                       opcodestr, "$rd, (${rs1})"> {
        let rs2 = 0;
      }
      def _EX : Pseudo<(outs FPR256:$rd),
                       (ins GPR:$rs1, MR0:$m0), [],
                       opcodestr, "$rd, (${rs1}), $m0">;
    }

    let mayLoad = 1 in {
      defm AIF_FLWL_PS : ETOpsFdR1<0b0001000, "aif.flwl.ps">;
      defm AIF_FLWG_PS : ETOpsFdR1<0b0001001, "aif.flwg.ps">;
    }

    multiclass ETOpsFs3Rs1<bits<7> funct7, string opcodestr> {
      def "" : RVInstET14<funct7, 0b111, OPC_ET_MEM_PS,
                          (outs), (ins FPR256:$fs3, GPR:$rs1),
                          opcodestr, "$fs3, (${rs1})">;
      def _EX : Pseudo<(outs), (ins FPR256:$fs3, GPR:$rs1, MR0:$m0), [],
                       opcodestr, "$fs3, (${rs1}), $m0">;
    }

    let mayStore = 1 in {
      defm AIF_FSWL_PS : ETOpsFs3Rs1<0b0101000, "aif.fswl.ps">;
      defm AIF_FSWG_PS : ETOpsFs3Rs1<0b0101001, "aif.fswg.ps">;
    }

    multiclass ETOpsRdR1R2FG<bits<7> funct7, bits<3> funct3, string opcodestr> {
      def "" : RVInstR<funct7, funct3, OPC_ET_MEM_PS, (outs FPR256:$rd),
                       (ins FPR256:$rs1, GPRMemZeroOffset:$rs2), opcodestr,
                       "$rd, ${rs1}, ${rs2}">;
      def _EX : Pseudo<(outs FPR256:$rd),
                       (ins FPR256:$rs1, GPRMemZeroOffset:$rs2, MR0:$m0), [], opcodestr,
                       "$rd, ${rs1}, ${rs2}, $m0">;
      def _PASSTHRU_EX : Pseudo<(outs FPR256:$rd),
                                (ins FPR256:$in, FPR256:$rs1, GPRMemZeroOffset:$rs2, MR0:$m0), [], opcodestr,
                                "$rd, ${rs1}, ${rs2}, $m0"> {
        let Constraints = "$rd = $in";
      }
    }

    let mayLoad = 1 in {
      defm AIF_FGBL_PS : ETOpsRdR1R2FG<0b1000000, 0b111, "aif.fgbl.ps">;
      defm AIF_FGHL_PS : ETOpsRdR1R2FG<0b1000100, 0b111, "aif.fghl.ps">;
      defm AIF_FGWL_PS : ETOpsRdR1R2FG<0b1001000, 0b111, "aif.fgwl.ps">;
    }

    multiclass ETOpsR3R1R2FS<bits<7> funct7, bits<3> funct3, string opcodestr> {
      def "" : RVInstET12<funct7, funct3, OPC_ET_MEM_PS, (outs),
                          (ins FPR256:$rs3, FPR256:$rs1, GPRMemZeroOffset:$rs2), opcodestr,
                          "${rs3}, ${rs1}, ${rs2}">;
      def _EX : Pseudo<(outs),
                       (ins FPR256:$rs3, FPR256:$rs1, GPRMemZeroOffset:$rs2, MR0:$m0),
                       [], opcodestr,
                       "${rs3}, ${rs1}, ${rs2}, $m0">;
    }

    let mayStore = 1 in {
      defm AIF_FSCBL_PS : ETOpsR3R1R2FS<0b1100000, 0b111, "aif.fscbl.ps">;
      defm AIF_FSCHL_PS : ETOpsR3R1R2FS<0b1100100, 0b111, "aif.fschl.ps">;
      defm AIF_FSCWL_PS : ETOpsR3R1R2FS<0b1101000, 0b111, "aif.fscwl.ps">;
    }

    let mayLoad = 1 in {
      defm AIF_FGBG_PS : ETOpsRdR1R2FG<0b1000001, 0b111, "aif.fgbg.ps">;
      defm AIF_FGHG_PS : ETOpsRdR1R2FG<0b1000101, 0b111, "aif.fghg.ps">;
      defm AIF_FGWG_PS : ETOpsRdR1R2FG<0b1001001, 0b111, "aif.fgwg.ps">;
    }

    let mayStore = 1 in {
      defm AIF_FSCBG_PS : ETOpsR3R1R2FS<0b1100001, 0b111, "aif.fscbg.ps">;
      defm AIF_FSCHG_PS : ETOpsR3R1R2FS<0b1100101, 0b111, "aif.fschg.ps">;
      defm AIF_FSCWG_PS : ETOpsR3R1R2FS<0b1101001, 0b111, "aif.fscwg.ps">;
    }

    let mayLoad = 1, mayStore = 1 in {
      defm AIF_FAMOADDL_PI  : ETOpsRdR1R2FG<0b0000011, 0b100, "aif.famoaddl.pi">;
      defm AIF_FAMOSWAPL_PI : ETOpsRdR1R2FG<0b0000111, 0b100, "aif.famoswapl.pi">;
      defm AIF_FAMOANDL_PI  : ETOpsRdR1R2FG<0b0001011, 0b100, "aif.famoandl.pi">;
      defm AIF_FAMOORL_PI   : ETOpsRdR1R2FG<0b0001111, 0b100, "aif.famoorl.pi">;
      defm AIF_FAMOXORL_PI  : ETOpsRdR1R2FG<0b0010011, 0b100, "aif.famoxorl.pi">;
      defm AIF_FAMOMINL_PI  : ETOpsRdR1R2FG<0b0010111, 0b100, "aif.famominl.pi">;
      defm AIF_FAMOMAXL_PI  : ETOpsRdR1R2FG<0b0011011, 0b100, "aif.famomaxl.pi">;
      defm AIF_FAMOMINUL_PI : ETOpsRdR1R2FG<0b0011111, 0b100, "aif.famominul.pi">;
      defm AIF_FAMOMAXUL_PI : ETOpsRdR1R2FG<0b0100011, 0b100, "aif.famomaxul.pi">;
      defm AIF_FAMOMINL_PS  : ETOpsRdR1R2FG<0b0011000, 0b100, "aif.famominl.ps">;
      defm AIF_FAMOMAXL_PS  : ETOpsRdR1R2FG<0b0010100, 0b100, "aif.famomaxl.ps">;

      defm AIF_FAMOADDG_PI  : ETOpsRdR1R2FG<0b1000011, 0b100, "aif.famoaddg.pi">;
      defm AIF_FAMOSWAPG_PI : ETOpsRdR1R2FG<0b1000111, 0b100, "aif.famoswapg.pi">;
      defm AIF_FAMOANDG_PI  : ETOpsRdR1R2FG<0b1001011, 0b100, "aif.famoandg.pi">;
      defm AIF_FAMOORG_PI   : ETOpsRdR1R2FG<0b1001111, 0b100, "aif.famoorg.pi">;
      defm AIF_FAMOXORG_PI  : ETOpsRdR1R2FG<0b1010011, 0b100, "aif.famoxorg.pi">;
      defm AIF_FAMOMING_PI  : ETOpsRdR1R2FG<0b1010111, 0b100, "aif.famoming.pi">;
      defm AIF_FAMOMAXG_PI  : ETOpsRdR1R2FG<0b1011011, 0b100, "aif.famomaxg.pi">;
      defm AIF_FAMOMINUG_PI : ETOpsRdR1R2FG<0b1011111, 0b100, "aif.famominug.pi">;
      defm AIF_FAMOMAXUG_PI : ETOpsRdR1R2FG<0b1100011, 0b100, "aif.famomaxug.pi">;
      defm AIF_FAMOMING_PS  : ETOpsRdR1R2FG<0b1011000, 0b100, "aif.famoming.ps">;
      defm AIF_FAMOMAXG_PS  : ETOpsRdR1R2FG<0b1010100, 0b100, "aif.famomaxg.ps">;
    }

    class ETOpsR2R1<bits<7> funct7, bits<3> funct3, string opcodestr> :
      RVInstR<funct7, funct3, OPC_OP_32, (outs GPR:$rd),
              (ins GPR:$rs2, GPRMemZeroOffset:$rs1), opcodestr,
              "${rd}, ${rs2}, ${rs1}">;

    let mayLoad = 1, mayStore = 1 in {
      def AIF_AMOSWAPL_W    : ETOpsR2R1<0b0000100, 0b010, "aif.amoswapl.w">;
      def AIF_AMOADDL_W     : ETOpsR2R1<0b0000000, 0b010, "aif.amoaddl.w">;
      def AIF_AMOXORL_W     : ETOpsR2R1<0b0010000, 0b010, "aif.amoxorl.w">;
      def AIF_AMOANDL_W     : ETOpsR2R1<0b0110000, 0b010, "aif.amoandl.w">;
      def AIF_AMOORL_W      : ETOpsR2R1<0b0100000, 0b010, "aif.amoorl.w">;
      def AIF_AMOMINL_W     : ETOpsR2R1<0b1000000, 0b010, "aif.amominl.w">;
      def AIF_AMOMAXL_W     : ETOpsR2R1<0b1010000, 0b010, "aif.amomaxl.w">;
      def AIF_AMOMINUL_W    : ETOpsR2R1<0b1100000, 0b010, "aif.amominul.w">;
      def AIF_AMOMAXUL_W    : ETOpsR2R1<0b1110000, 0b010, "aif.amomaxul.w">;
      def AIF_AMOCMPSWAPL_W : ETOpsR2R1<0b1111000, 0b010, "aif.amocmpswapl.w">;

      def AIF_AMOSWAPG_W    : ETOpsR2R1<0b0000101, 0b010, "aif.amoswapg.w">;
      def AIF_AMOADDG_W     : ETOpsR2R1<0b0000001, 0b010, "aif.amoaddg.w">;
      def AIF_AMOXORG_W     : ETOpsR2R1<0b0010001, 0b010, "aif.amoxorg.w">;
      def AIF_AMOANDG_W     : ETOpsR2R1<0b0110001, 0b010, "aif.amoandg.w">;
      def AIF_AMOORG_W      : ETOpsR2R1<0b0100001, 0b010, "aif.amoorg.w">;
      def AIF_AMOMING_W     : ETOpsR2R1<0b1000001, 0b010, "aif.amoming.w">;
      def AIF_AMOMAXG_W     : ETOpsR2R1<0b1010001, 0b010, "aif.amomaxg.w">;
      def AIF_AMOMINUG_W    : ETOpsR2R1<0b1100001, 0b010, "aif.amominug.w">;
      def AIF_AMOMAXUG_W    : ETOpsR2R1<0b1110001, 0b010, "aif.amomaxug.w">;
      def AIF_AMOCMPSWAPG_W : ETOpsR2R1<0b1111001, 0b010, "aif.amocmpswapg.w">;

      def AIF_AMOSWAPL_D    : ETOpsR2R1<0b0000100, 0b011, "aif.amoswapl.d">;
      def AIF_AMOADDL_D     : ETOpsR2R1<0b0000000, 0b011, "aif.amoaddl.d">;
      def AIF_AMOXORL_D     : ETOpsR2R1<0b0010000, 0b011, "aif.amoxorl.d">;
      def AIF_AMOANDL_D     : ETOpsR2R1<0b0110000, 0b011, "aif.amoandl.d">;
      def AIF_AMOORL_D      : ETOpsR2R1<0b0100000, 0b011, "aif.amoorl.d">;
      def AIF_AMOMINL_D     : ETOpsR2R1<0b1000000, 0b011, "aif.amominl.d">;
      def AIF_AMOMAXL_D     : ETOpsR2R1<0b1010000, 0b011, "aif.amomaxl.d">;
      def AIF_AMOMINUL_D    : ETOpsR2R1<0b1100000, 0b011, "aif.amominul.d">;
      def AIF_AMOMAXUL_D    : ETOpsR2R1<0b1110000, 0b011, "aif.amomaxul.d">;
      def AIF_AMOCMPSWAPL_D : ETOpsR2R1<0b1111000, 0b011, "aif.amocmpswapl.d">;

      def AIF_AMOSWAPG_D    : ETOpsR2R1<0b0000101, 0b011, "aif.amoswapg.d">;
      def AIF_AMOADDG_D     : ETOpsR2R1<0b0000001, 0b011, "aif.amoaddg.d">;
      def AIF_AMOXORG_D     : ETOpsR2R1<0b0010001, 0b011, "aif.amoxorg.d">;
      def AIF_AMOANDG_D     : ETOpsR2R1<0b0110001, 0b011, "aif.amoandg.d">;
      def AIF_AMOORG_D      : ETOpsR2R1<0b0100001, 0b011, "aif.amoorg.d">;
      def AIF_AMOMING_D     : ETOpsR2R1<0b1000001, 0b011, "aif.amoming.d">;
      def AIF_AMOMAXG_D     : ETOpsR2R1<0b1010001, 0b011, "aif.amomaxg.d">;
      def AIF_AMOMINUG_D    : ETOpsR2R1<0b1100001, 0b011, "aif.amominug.d">;
      def AIF_AMOMAXUG_D    : ETOpsR2R1<0b1110001, 0b011, "aif.amomaxug.d">;
      def AIF_AMOCMPSWAPG_D : ETOpsR2R1<0b1111001, 0b011, "aif.amocmpswapg.d">;
    } // let mayLoad = 1, mayStore =1 in

    class ETOpsR1R2S<bits<7> funct7, bits<3> funct3, string opcodestr> :
      RVInstR<funct7, funct3, OPC_OP_32, (outs),
              (ins GPR:$rs2, GPRMemZeroOffset:$rs1), opcodestr,
              "${rs2}, ${rs1}"> {
      let rd = 0;
    }

    let mayLoad = 1 in {
      def AIF_SBL : ETOpsR1R2S<0b0001000, 0b011, "aif.sbl">;
      def AIF_SHL : ETOpsR1R2S<0b0001100, 0b011, "aif.shl">;
      def AIF_SBG : ETOpsR1R2S<0b0001001, 0b011, "aif.sbg">;
      def AIF_SHG : ETOpsR1R2S<0b0001101, 0b011, "aif.shg">;
    }

    // These pseudo-instructions are used for stack references
    // where various places assume the offset immediate follows
    // the address operand.
    let mayStore = 1 in
    def AIF_StackFSQ2 : Pseudo<(outs),
                               (ins FPR256:$rs2, GPR:$rs1, simm12_lo:$imm), [],
                               "aif.fsq2", "$rs2, ${imm}(${rs1})">;
    let mayLoad = 1 in
    def AIF_StackFLQ2 : Pseudo<(outs FPR256:$rd),
                               (ins GPR:$rs1, simm12_lo:$imm), [],
                               "aif.flq2", "$rd, ${imm}(${rs1})">;

    // These pseudo-instructions are used for spill and reload
    // of mask registers
    let mayStore = 1 in
    def AIF_StackMS : Pseudo<(outs),
                             (ins MR:$rs2, GPR:$rs1, simm12_lo:$imm), [],
                             "aif.ms", "$rs2, ${imm}(${rs1})">;
    let mayLoad = 1 in
    def AIF_StackML : Pseudo<(outs MR:$rd),
                             (ins GPR:$rs1, simm12_lo:$imm), [],
                             "aif.ml", "$rd, ${imm}(${rs1})">;

  } // let hasSideEffects = 0, mayLoad = 0, mayStore = 0

} // let Predicates = [HasXAIFET], DecoderNamespace = "XAIF"
