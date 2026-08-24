/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AArch64 MIR target operations                                              *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: a64.td                                                               *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

include "mlir/Dialect/MIR/IR/MIRTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def AArch64MIR_abs : AArch64MIR_TargetOp<"abs", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ABSWr ABSXr ABS_ZPmZ_B ABS_ZPmZ_D ABS_ZPmZ_H ABS_ZPmZ_S ABS_ZPzZ_B ABS_ZPzZ_D ABS_ZPzZ_H ABS_ZPzZ_S ABSv1i64 ABSv2i32 ABSv2i64 ABSv4i16 ABSv4i32 ABSv8i8 ABSv8i16 ABSv16i8
}

def AArch64MIR_adclb : AArch64MIR_TargetOp<"adclb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ADCLB_ZZZ_D ADCLB_ZZZ_S
}

def AArch64MIR_adclt : AArch64MIR_TargetOp<"adclt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ADCLT_ZZZ_D ADCLT_ZZZ_S
}

def AArch64MIR_adcs : AArch64MIR_TargetOp<"adcs", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ADCSWr ADCSXr
}

def AArch64MIR_adc : AArch64MIR_TargetOp<"adc", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ADCWr ADCXr
}

def AArch64MIR_addg : AArch64MIR_TargetOp<"addg", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ADDG
}

def AArch64MIR_addha : AArch64MIR_TargetOp<"addha", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ADDHA_MPPZ_D ADDHA_MPPZ_S
}

def AArch64MIR_addhnb : AArch64MIR_TargetOp<"addhnb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ADDHNB_ZZZ_B ADDHNB_ZZZ_H ADDHNB_ZZZ_S
}

def AArch64MIR_addhnt : AArch64MIR_TargetOp<"addhnt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ADDHNT_ZZZ_B ADDHNT_ZZZ_H ADDHNT_ZZZ_S
}

def AArch64MIR_addhn : AArch64MIR_TargetOp<"addhn", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ADDHNv2i64_v2i32 ADDHNv4i32_v4i16 ADDHNv8i16_v8i8
}

def AArch64MIR_addhn2 : AArch64MIR_TargetOp<"addhn2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ADDHNv2i64_v4i32 ADDHNv4i32_v8i16 ADDHNv8i16_v16i8
}

def AArch64MIR_addpl : AArch64MIR_TargetOp<"addpl", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ADDPL_XXI
}

def AArch64MIR_addpt : AArch64MIR_TargetOp<"addpt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ADDPT_shift ADD_ZPmZ_CPA ADD_ZZZ_CPA
}

def AArch64MIR_addp : AArch64MIR_TargetOp<"addp", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ADDP_ZPmZ_B ADDP_ZPmZ_D ADDP_ZPmZ_H ADDP_ZPmZ_S ADDPv2i32 ADDPv2i64 ADDPv2i64p ADDPv4i16 ADDPv4i32 ADDPv8i8 ADDPv8i16 ADDPv16i8
}

def AArch64MIR_addqp : AArch64MIR_TargetOp<"addqp", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ADDQP_ZZZ_B ADDQP_ZZZ_D ADDQP_ZZZ_H ADDQP_ZZZ_S
}

def AArch64MIR_addqv : AArch64MIR_TargetOp<"addqv", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ADDQV_VPZ_B ADDQV_VPZ_D ADDQV_VPZ_H ADDQV_VPZ_S
}

def AArch64MIR_addspl : AArch64MIR_TargetOp<"addspl", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ADDSPL_XXI
}

def AArch64MIR_addsubp : AArch64MIR_TargetOp<"addsubp", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ADDSUBP_ZZZ_B ADDSUBP_ZZZ_D ADDSUBP_ZZZ_H ADDSUBP_ZZZ_S
}

def AArch64MIR_addsvl : AArch64MIR_TargetOp<"addsvl", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ADDSVL_XXI
}

def AArch64MIR_adds : AArch64MIR_TargetOp<"adds", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ADDSWri ADDSWrs ADDSWrx ADDSXri ADDSXrs ADDSXrx ADDSXrx64
}

def AArch64MIR_addva : AArch64MIR_TargetOp<"addva", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ADDVA_MPPZ_D ADDVA_MPPZ_S
}

def AArch64MIR_addvl : AArch64MIR_TargetOp<"addvl", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ADDVL_XXI
}

def AArch64MIR_addv : AArch64MIR_TargetOp<"addv", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ADDVv4i16v ADDVv4i32v ADDVv8i8v ADDVv8i16v ADDVv16i8v
}

def AArch64MIR_add : AArch64MIR_TargetOp<"add", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ADDWri ADDWrs ADDWrx ADDXri ADDXrs ADDXrx ADDXrx64 ADD_VG2_2ZZ_B ADD_VG2_2ZZ_D ADD_VG2_2ZZ_H ADD_VG2_2ZZ_S ADD_VG2_M2Z2Z_D ADD_VG2_M2Z2Z_S ADD_VG2_M2ZZ_D ADD_VG2_M2ZZ_S ADD_VG2_M2Z_D ADD_VG2_M2Z_S ADD_VG4_4ZZ_B ADD_VG4_4ZZ_D ADD_VG4_4ZZ_H ADD_VG4_4ZZ_S ADD_VG4_M4Z4Z_D ADD_VG4_M4Z4Z_S ADD_VG4_M4ZZ_D ADD_VG4_M4ZZ_S ADD_VG4_M4Z_D ADD_VG4_M4Z_S ADD_ZI_B ADD_ZI_D ADD_ZI_H ADD_ZI_S ADD_ZPmZ_B ADD_ZPmZ_D ADD_ZPmZ_H ADD_ZPmZ_S ADD_ZZZ_B ADD_ZZZ_D ADD_ZZZ_H ADD_ZZZ_S ADDv1i64 ADDv2i32 ADDv2i64 ADDv4i16 ADDv4i32 ADDv8i8 ADDv8i16 ADDv16i8
}

def AArch64MIR_adr : AArch64MIR_TargetOp<"adr", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ADR ADR_LSL_ZZZ_D_0 ADR_LSL_ZZZ_D_1 ADR_LSL_ZZZ_D_2 ADR_LSL_ZZZ_D_3 ADR_LSL_ZZZ_S_0 ADR_LSL_ZZZ_S_1 ADR_LSL_ZZZ_S_2 ADR_LSL_ZZZ_S_3 ADR_SXTW_ZZZ_D_0 ADR_SXTW_ZZZ_D_1 ADR_SXTW_ZZZ_D_2 ADR_SXTW_ZZZ_D_3 ADR_UXTW_ZZZ_D_0 ADR_UXTW_ZZZ_D_1 ADR_UXTW_ZZZ_D_2 ADR_UXTW_ZZZ_D_3
}

def AArch64MIR_adrp : AArch64MIR_TargetOp<"adrp", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ADRP
}

def AArch64MIR_aesdimc : AArch64MIR_TargetOp<"aesdimc", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: AESDIMC_2ZZI_B AESDIMC_4ZZI_B
}

def AArch64MIR_aesd : AArch64MIR_TargetOp<"aesd", [Pure, Commutative]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: AESD_2ZZI_B AESD_4ZZI_B AESD_ZZZ_B AESDrr
}

def AArch64MIR_aesemc : AArch64MIR_TargetOp<"aesemc", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: AESEMC_2ZZI_B AESEMC_4ZZI_B
}

def AArch64MIR_aese : AArch64MIR_TargetOp<"aese", [Pure, Commutative]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: AESE_2ZZI_B AESE_4ZZI_B AESE_ZZZ_B AESErr
}

def AArch64MIR_aesimc : AArch64MIR_TargetOp<"aesimc", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: AESIMC_ZZ_B AESIMCrr
}

def AArch64MIR_aesmc : AArch64MIR_TargetOp<"aesmc", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: AESMC_ZZ_B AESMCrr
}

def AArch64MIR_andqv : AArch64MIR_TargetOp<"andqv", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ANDQV_VPZ_B ANDQV_VPZ_D ANDQV_VPZ_H ANDQV_VPZ_S
}

def AArch64MIR_ands : AArch64MIR_TargetOp<"ands", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ANDSWri ANDSWrs ANDSXri ANDSXrs ANDS_PPzPP
}

def AArch64MIR_andv : AArch64MIR_TargetOp<"andv", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ANDV_VPZ_B ANDV_VPZ_D ANDV_VPZ_H ANDV_VPZ_S
}

def AArch64MIR_and_ : AArch64MIR_TargetOp<"and", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ANDWri ANDWrs ANDXri ANDXrs AND_PPzPP AND_ZI AND_ZPmZ_B AND_ZPmZ_D AND_ZPmZ_H AND_ZPmZ_S AND_ZZZ ANDv8i8 ANDv16i8
}

def AArch64MIR_asrd : AArch64MIR_TargetOp<"asrd", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ASRD_ZPmI_B ASRD_ZPmI_D ASRD_ZPmI_H ASRD_ZPmI_S
}

def AArch64MIR_asrr : AArch64MIR_TargetOp<"asrr", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ASRR_ZPmZ_B ASRR_ZPmZ_D ASRR_ZPmZ_H ASRR_ZPmZ_S
}

def AArch64MIR_asr : AArch64MIR_TargetOp<"asr", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ASRVWr ASRVXr ASR_WIDE_ZPmZ_B ASR_WIDE_ZPmZ_H ASR_WIDE_ZPmZ_S ASR_WIDE_ZZZ_B ASR_WIDE_ZZZ_H ASR_WIDE_ZZZ_S ASR_ZPmI_B ASR_ZPmI_D ASR_ZPmI_H ASR_ZPmI_S ASR_ZPmZ_B ASR_ZPmZ_D ASR_ZPmZ_H ASR_ZPmZ_S ASR_ZZI_B ASR_ZZI_D ASR_ZZI_H ASR_ZZI_S
}

def AArch64MIR_autda : AArch64MIR_TargetOp<"autda", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: AUTDA
}

def AArch64MIR_autdb : AArch64MIR_TargetOp<"autdb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: AUTDB
}

def AArch64MIR_autdza : AArch64MIR_TargetOp<"autdza", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: AUTDZA
}

def AArch64MIR_autdzb : AArch64MIR_TargetOp<"autdzb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: AUTDZB
}

def AArch64MIR_autia : AArch64MIR_TargetOp<"autia", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: AUTIA
}

def AArch64MIR_hint : AArch64MIR_TargetOp<"hint", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: AUTIA1716 AUTIASP AUTIAZ AUTIB1716 AUTIBSP AUTIBZ CHKFEAT HINT NOP PACIA1716 PACIASP PACIAZ PACIB1716 PACIBSP PACIBZ PACM XPACLRI
}

def AArch64MIR_autia171615 : AArch64MIR_TargetOp<"autia171615", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: AUTIA171615
}

def AArch64MIR_autiasppc : AArch64MIR_TargetOp<"autiasppc", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: AUTIASPPCi
}

def AArch64MIR_autiasppcr : AArch64MIR_TargetOp<"autiasppcr", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: AUTIASPPCr
}

def AArch64MIR_autib : AArch64MIR_TargetOp<"autib", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: AUTIB
}

def AArch64MIR_autib171615 : AArch64MIR_TargetOp<"autib171615", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: AUTIB171615
}

def AArch64MIR_autibsppc : AArch64MIR_TargetOp<"autibsppc", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: AUTIBSPPCi
}

def AArch64MIR_autibsppcr : AArch64MIR_TargetOp<"autibsppcr", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: AUTIBSPPCr
}

def AArch64MIR_autiza : AArch64MIR_TargetOp<"autiza", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: AUTIZA
}

def AArch64MIR_autizb : AArch64MIR_TargetOp<"autizb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: AUTIZB
}

def AArch64MIR_axflag : AArch64MIR_TargetOp<"axflag", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: AXFLAG
}

def AArch64MIR_b : AArch64MIR_TargetOp<"b", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: B
}

def AArch64MIR_bcax : AArch64MIR_TargetOp<"bcax", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BCAX BCAX_ZZZZ
}

def AArch64MIR_bc_cond : AArch64MIR_TargetOp<"bc_cond", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BCcc
}

def AArch64MIR_bdep : AArch64MIR_TargetOp<"bdep", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BDEP_ZZZ_B BDEP_ZZZ_D BDEP_ZZZ_H BDEP_ZZZ_S
}

def AArch64MIR_bext : AArch64MIR_TargetOp<"bext", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BEXT_ZZZ_B BEXT_ZZZ_D BEXT_ZZZ_H BEXT_ZZZ_S
}

def AArch64MIR_bf1cvtl : AArch64MIR_TargetOp<"bf1cvtl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BF1CVTL BF1CVTL_2ZZ_BtoH
}

def AArch64MIR_bf1cvtl2 : AArch64MIR_TargetOp<"bf1cvtl2", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BF1CVTL2
}

def AArch64MIR_bf1cvtlt : AArch64MIR_TargetOp<"bf1cvtlt", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BF1CVTLT_ZZ_BtoH
}

def AArch64MIR_bf1cvt : AArch64MIR_TargetOp<"bf1cvt", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BF1CVT_2ZZ_BtoH BF1CVT_ZZ_BtoH
}

def AArch64MIR_bf2cvtl : AArch64MIR_TargetOp<"bf2cvtl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BF2CVTL BF2CVTL_2ZZ_BtoH
}

def AArch64MIR_bf2cvtl2 : AArch64MIR_TargetOp<"bf2cvtl2", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BF2CVTL2
}

def AArch64MIR_bf2cvtlt : AArch64MIR_TargetOp<"bf2cvtlt", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BF2CVTLT_ZZ_BtoH
}

def AArch64MIR_bf2cvt : AArch64MIR_TargetOp<"bf2cvt", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BF2CVT_2ZZ_BtoH BF2CVT_ZZ_BtoH
}

def AArch64MIR_bfdot : AArch64MIR_TargetOp<"bfdot", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BF16DOTlanev4bf16 BF16DOTlanev8bf16 BFDOT_VG2_M2Z2Z_HtoS BFDOT_VG2_M2ZZI_HtoS BFDOT_VG2_M2ZZ_HtoS BFDOT_VG4_M4Z4Z_HtoS BFDOT_VG4_M4ZZI_HtoS BFDOT_VG4_M4ZZ_HtoS BFDOT_ZZI BFDOT_ZZZ BFDOTv4bf16 BFDOTv8bf16
}

def AArch64MIR_bfadd : AArch64MIR_TargetOp<"bfadd", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BFADD_VG2_M2Z_H BFADD_VG4_M4Z_H BFADD_ZPmZ BFADD_ZZZ
}

def AArch64MIR_bfclamp : AArch64MIR_TargetOp<"bfclamp", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BFCLAMP_VG2_2ZZZ_H BFCLAMP_VG4_4ZZZ_H BFCLAMP_ZZZ
}

def AArch64MIR_bfcvt : AArch64MIR_TargetOp<"bfcvt", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BFCVT BFCVT_Z2Z_HtoB BFCVT_Z2Z_StoH BFCVT_ZPmZ BFCVT_ZPzZ_StoH
}

def AArch64MIR_bfcvtn : AArch64MIR_TargetOp<"bfcvtn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BFCVTN BFCVTN_Z2Z_HtoB BFCVTN_Z2Z_StoH
}

def AArch64MIR_bfcvtn2 : AArch64MIR_TargetOp<"bfcvtn2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BFCVTN2
}

def AArch64MIR_bfcvtnt : AArch64MIR_TargetOp<"bfcvtnt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BFCVTNT_ZPmZ BFCVTNT_ZPzZ_StoH
}

def AArch64MIR_bfmaxnm : AArch64MIR_TargetOp<"bfmaxnm", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BFMAXNM_VG2_2Z2Z_H BFMAXNM_VG2_2ZZ_H BFMAXNM_VG4_4Z2Z_H BFMAXNM_VG4_4ZZ_H BFMAXNM_ZPmZ
}

def AArch64MIR_bfmax : AArch64MIR_TargetOp<"bfmax", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BFMAX_VG2_2Z2Z_H BFMAX_VG2_2ZZ_H BFMAX_VG4_4Z2Z_H BFMAX_VG4_4ZZ_H BFMAX_ZPmZ
}

def AArch64MIR_bfminnm : AArch64MIR_TargetOp<"bfminnm", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BFMINNM_VG2_2Z2Z_H BFMINNM_VG2_2ZZ_H BFMINNM_VG4_4Z2Z_H BFMINNM_VG4_4ZZ_H BFMINNM_ZPmZ
}

def AArch64MIR_bfmin : AArch64MIR_TargetOp<"bfmin", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BFMIN_VG2_2Z2Z_H BFMIN_VG2_2ZZ_H BFMIN_VG4_4Z2Z_H BFMIN_VG4_4ZZ_H BFMIN_ZPmZ
}

def AArch64MIR_bfmlalb : AArch64MIR_TargetOp<"bfmlalb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BFMLALB BFMLALBIdx BFMLALB_ZZZ BFMLALB_ZZZI
}

def AArch64MIR_bfmlalt : AArch64MIR_TargetOp<"bfmlalt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BFMLALT BFMLALTIdx BFMLALT_ZZZ BFMLALT_ZZZI
}

def AArch64MIR_bfmlal : AArch64MIR_TargetOp<"bfmlal", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BFMLAL_MZZI_HtoS BFMLAL_MZZ_HtoS BFMLAL_VG2_M2Z2Z_HtoS BFMLAL_VG2_M2ZZI_HtoS BFMLAL_VG2_M2ZZ_HtoS BFMLAL_VG4_M4Z4Z_HtoS BFMLAL_VG4_M4ZZI_HtoS BFMLAL_VG4_M4ZZ_HtoS
}

def AArch64MIR_bfmla : AArch64MIR_TargetOp<"bfmla", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BFMLA_VG2_M2Z2Z BFMLA_VG2_M2ZZ BFMLA_VG2_M2ZZI BFMLA_VG4_M4Z4Z BFMLA_VG4_M4ZZ BFMLA_VG4_M4ZZI BFMLA_ZPmZZ BFMLA_ZZZI
}

def AArch64MIR_bfmlslb : AArch64MIR_TargetOp<"bfmlslb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BFMLSLB_ZZZI_S BFMLSLB_ZZZ_S
}

def AArch64MIR_bfmlslt : AArch64MIR_TargetOp<"bfmlslt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BFMLSLT_ZZZI_S BFMLSLT_ZZZ_S
}

def AArch64MIR_bfmlsl : AArch64MIR_TargetOp<"bfmlsl", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BFMLSL_MZZI_HtoS BFMLSL_MZZ_HtoS BFMLSL_VG2_M2Z2Z_HtoS BFMLSL_VG2_M2ZZI_HtoS BFMLSL_VG2_M2ZZ_HtoS BFMLSL_VG4_M4Z4Z_HtoS BFMLSL_VG4_M4ZZI_HtoS BFMLSL_VG4_M4ZZ_HtoS
}

def AArch64MIR_bfmls : AArch64MIR_TargetOp<"bfmls", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BFMLS_VG2_M2Z2Z BFMLS_VG2_M2ZZ BFMLS_VG2_M2ZZI BFMLS_VG4_M4Z4Z BFMLS_VG4_M4ZZ BFMLS_VG4_M4ZZI BFMLS_ZPmZZ BFMLS_ZZZI
}

def AArch64MIR_bfmmla : AArch64MIR_TargetOp<"bfmmla", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BFMMLA BFMMLA_ZZZ_H BFMMLA_ZZZ_HtoS
}

def AArch64MIR_bfmop4a : AArch64MIR_TargetOp<"bfmop4a", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BFMOP4A_M2Z2Z_H BFMOP4A_M2Z2Z_S BFMOP4A_M2ZZ_H BFMOP4A_M2ZZ_S BFMOP4A_MZ2Z_H BFMOP4A_MZ2Z_S BFMOP4A_MZZ_H BFMOP4A_MZZ_S
}

def AArch64MIR_bfmop4s : AArch64MIR_TargetOp<"bfmop4s", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BFMOP4S_M2Z2Z_H BFMOP4S_M2Z2Z_S BFMOP4S_M2ZZ_H BFMOP4S_M2ZZ_S BFMOP4S_MZ2Z_H BFMOP4S_MZ2Z_S BFMOP4S_MZZ_H BFMOP4S_MZZ_S
}

def AArch64MIR_bfmopa : AArch64MIR_TargetOp<"bfmopa", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BFMOPA_MPPZZ BFMOPA_MPPZZ_H
}

def AArch64MIR_bfmops : AArch64MIR_TargetOp<"bfmops", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BFMOPS_MPPZZ BFMOPS_MPPZZ_H
}

def AArch64MIR_bfmul : AArch64MIR_TargetOp<"bfmul", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BFMUL_2Z2Z BFMUL_2ZZ BFMUL_4Z4Z BFMUL_4ZZ BFMUL_ZPmZ BFMUL_ZZZ BFMUL_ZZZI
}

def AArch64MIR_bfm : AArch64MIR_TargetOp<"bfm", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BFMWri BFMXri
}

def AArch64MIR_bfscale : AArch64MIR_TargetOp<"bfscale", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BFSCALE_2Z2Z BFSCALE_2ZZ BFSCALE_4Z4Z BFSCALE_4ZZ BFSCALE_ZPmZ
}

def AArch64MIR_bfsub : AArch64MIR_TargetOp<"bfsub", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BFSUB_VG2_M2Z_H BFSUB_VG4_M4Z_H BFSUB_ZPmZ BFSUB_ZZZ
}

def AArch64MIR_bftmopa : AArch64MIR_TargetOp<"bftmopa", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BFTMOPA_M2ZZZI_HtoH BFTMOPA_M2ZZZI_HtoS
}

def AArch64MIR_bfvdot : AArch64MIR_TargetOp<"bfvdot", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BFVDOT_VG2_M2ZZI_HtoS
}

def AArch64MIR_bgrp : AArch64MIR_TargetOp<"bgrp", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BGRP_ZZZ_B BGRP_ZZZ_D BGRP_ZZZ_H BGRP_ZZZ_S
}

def AArch64MIR_bics : AArch64MIR_TargetOp<"bics", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BICSWrs BICSXrs BICS_PPzPP
}

def AArch64MIR_bic : AArch64MIR_TargetOp<"bic", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BICWrs BICXrs BIC_PPzPP BIC_ZPmZ_B BIC_ZPmZ_D BIC_ZPmZ_H BIC_ZPmZ_S BIC_ZZZ BICv2i32 BICv4i16 BICv4i32 BICv8i8 BICv8i16 BICv16i8
}

def AArch64MIR_bif : AArch64MIR_TargetOp<"bif", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BIFv8i8 BIFv16i8
}

def AArch64MIR_bit : AArch64MIR_TargetOp<"bit", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BITv8i8 BITv16i8
}

def AArch64MIR_bl : AArch64MIR_TargetOp<"bl", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BL
}

def AArch64MIR_blr : AArch64MIR_TargetOp<"blr", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BLR
}

def AArch64MIR_blraa : AArch64MIR_TargetOp<"blraa", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BLRAA
}

def AArch64MIR_blraaz : AArch64MIR_TargetOp<"blraaz", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BLRAAZ
}

def AArch64MIR_blrab : AArch64MIR_TargetOp<"blrab", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BLRAB
}

def AArch64MIR_blrabz : AArch64MIR_TargetOp<"blrabz", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BLRABZ
}

def AArch64MIR_bmopa : AArch64MIR_TargetOp<"bmopa", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BMOPA_MPPZZ_S
}

def AArch64MIR_bmops : AArch64MIR_TargetOp<"bmops", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BMOPS_MPPZZ_S
}

def AArch64MIR_br : AArch64MIR_TargetOp<"br", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BR
}

def AArch64MIR_braa : AArch64MIR_TargetOp<"braa", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BRAA
}

def AArch64MIR_braaz : AArch64MIR_TargetOp<"braaz", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BRAAZ
}

def AArch64MIR_brab : AArch64MIR_TargetOp<"brab", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BRAB
}

def AArch64MIR_brabz : AArch64MIR_TargetOp<"brabz", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BRABZ
}

def AArch64MIR_brk : AArch64MIR_TargetOp<"brk", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BRK
}

def AArch64MIR_brkas : AArch64MIR_TargetOp<"brkas", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BRKAS_PPzP
}

def AArch64MIR_brka : AArch64MIR_TargetOp<"brka", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BRKA_PPmP BRKA_PPzP
}

def AArch64MIR_brkbs : AArch64MIR_TargetOp<"brkbs", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BRKBS_PPzP
}

def AArch64MIR_brkb : AArch64MIR_TargetOp<"brkb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BRKB_PPmP BRKB_PPzP
}

def AArch64MIR_brkns : AArch64MIR_TargetOp<"brkns", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BRKNS_PPzP
}

def AArch64MIR_brkn : AArch64MIR_TargetOp<"brkn", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BRKN_PPzP
}

def AArch64MIR_brkpas : AArch64MIR_TargetOp<"brkpas", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BRKPAS_PPzPP
}

def AArch64MIR_brkpa : AArch64MIR_TargetOp<"brkpa", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BRKPA_PPzPP
}

def AArch64MIR_brkpbs : AArch64MIR_TargetOp<"brkpbs", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BRKPBS_PPzPP
}

def AArch64MIR_brkpb : AArch64MIR_TargetOp<"brkpb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BRKPB_PPzPP
}

def AArch64MIR_bsl1n : AArch64MIR_TargetOp<"bsl1n", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BSL1N_ZZZZ
}

def AArch64MIR_bsl2n : AArch64MIR_TargetOp<"bsl2n", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BSL2N_ZZZZ
}

def AArch64MIR_bsl : AArch64MIR_TargetOp<"bsl", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: BSL_ZZZZ BSLv8i8 BSLv16i8
}

def AArch64MIR_b_cond : AArch64MIR_TargetOp<"b_cond", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: Bcc
}

def AArch64MIR_cadd : AArch64MIR_TargetOp<"cadd", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CADD_ZZI_B CADD_ZZI_D CADD_ZZI_H CADD_ZZI_S
}

def AArch64MIR_casab : AArch64MIR_TargetOp<"casab", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CASAB
}

def AArch64MIR_casah : AArch64MIR_TargetOp<"casah", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CASAH
}

def AArch64MIR_casalb : AArch64MIR_TargetOp<"casalb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CASALB
}

def AArch64MIR_casalh : AArch64MIR_TargetOp<"casalh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CASALH
}

def AArch64MIR_casalt : AArch64MIR_TargetOp<"casalt", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CASALTX
}

def AArch64MIR_casal : AArch64MIR_TargetOp<"casal", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CASALW CASALX
}

def AArch64MIR_casat : AArch64MIR_TargetOp<"casat", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CASATX
}

def AArch64MIR_casa : AArch64MIR_TargetOp<"casa", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CASAW CASAX
}

def AArch64MIR_casb : AArch64MIR_TargetOp<"casb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CASB
}

def AArch64MIR_cash : AArch64MIR_TargetOp<"cash", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CASH
}

def AArch64MIR_caslb : AArch64MIR_TargetOp<"caslb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CASLB
}

def AArch64MIR_caslh : AArch64MIR_TargetOp<"caslh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CASLH
}

def AArch64MIR_caslt : AArch64MIR_TargetOp<"caslt", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CASLTX
}

def AArch64MIR_casl : AArch64MIR_TargetOp<"casl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CASLW CASLX
}

def AArch64MIR_caspalt : AArch64MIR_TargetOp<"caspalt", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CASPALTX
}

def AArch64MIR_caspal : AArch64MIR_TargetOp<"caspal", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CASPALW CASPALX
}

def AArch64MIR_caspat : AArch64MIR_TargetOp<"caspat", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CASPATX
}

def AArch64MIR_caspa : AArch64MIR_TargetOp<"caspa", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CASPAW CASPAX
}

def AArch64MIR_casplt : AArch64MIR_TargetOp<"casplt", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CASPLTX
}

def AArch64MIR_caspl : AArch64MIR_TargetOp<"caspl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CASPLW CASPLX
}

def AArch64MIR_caspt : AArch64MIR_TargetOp<"caspt", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CASPTX
}

def AArch64MIR_casp : AArch64MIR_TargetOp<"casp", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CASPW CASPX
}

def AArch64MIR_cast : AArch64MIR_TargetOp<"cast", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CASTX
}

def AArch64MIR_cas : AArch64MIR_TargetOp<"cas", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CASW CASX
}

def AArch64MIR_cbbeq : AArch64MIR_TargetOp<"cbbeq", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CBBEQWrr
}

def AArch64MIR_cbbge : AArch64MIR_TargetOp<"cbbge", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CBBGEWrr
}

def AArch64MIR_cbbgt : AArch64MIR_TargetOp<"cbbgt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CBBGTWrr
}

def AArch64MIR_cbbhi : AArch64MIR_TargetOp<"cbbhi", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CBBHIWrr
}

def AArch64MIR_cbbhs : AArch64MIR_TargetOp<"cbbhs", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CBBHSWrr
}

def AArch64MIR_cbbne : AArch64MIR_TargetOp<"cbbne", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CBBNEWrr
}

def AArch64MIR_cbeq : AArch64MIR_TargetOp<"cbeq", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CBEQWri CBEQWrr CBEQXri CBEQXrr
}

def AArch64MIR_cbge : AArch64MIR_TargetOp<"cbge", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CBGEWrr CBGEXrr
}

def AArch64MIR_cbgt : AArch64MIR_TargetOp<"cbgt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CBGTWri CBGTWrr CBGTXri CBGTXrr
}

def AArch64MIR_cbheq : AArch64MIR_TargetOp<"cbheq", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CBHEQWrr
}

def AArch64MIR_cbhge : AArch64MIR_TargetOp<"cbhge", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CBHGEWrr
}

def AArch64MIR_cbhgt : AArch64MIR_TargetOp<"cbhgt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CBHGTWrr
}

def AArch64MIR_cbhhi : AArch64MIR_TargetOp<"cbhhi", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CBHHIWrr
}

def AArch64MIR_cbhhs : AArch64MIR_TargetOp<"cbhhs", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CBHHSWrr
}

def AArch64MIR_cbhi : AArch64MIR_TargetOp<"cbhi", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CBHIWri CBHIWrr CBHIXri CBHIXrr
}

def AArch64MIR_cbhne : AArch64MIR_TargetOp<"cbhne", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CBHNEWrr
}

def AArch64MIR_cbhs : AArch64MIR_TargetOp<"cbhs", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CBHSWrr CBHSXrr
}

def AArch64MIR_cblo : AArch64MIR_TargetOp<"cblo", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CBLOWri CBLOXri
}

def AArch64MIR_cblt : AArch64MIR_TargetOp<"cblt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CBLTWri CBLTXri
}

def AArch64MIR_cbne : AArch64MIR_TargetOp<"cbne", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CBNEWri CBNEWrr CBNEXri CBNEXrr
}

def AArch64MIR_cbnz : AArch64MIR_TargetOp<"cbnz", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CBNZW CBNZX
}

def AArch64MIR_cbz : AArch64MIR_TargetOp<"cbz", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CBZW CBZX
}

def AArch64MIR_ccmn : AArch64MIR_TargetOp<"ccmn", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CCMNWi CCMNWr CCMNXi CCMNXr
}

def AArch64MIR_ccmp : AArch64MIR_TargetOp<"ccmp", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CCMPWi CCMPWr CCMPXi CCMPXr
}

def AArch64MIR_cdot : AArch64MIR_TargetOp<"cdot", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CDOT_ZZZI_D CDOT_ZZZI_S CDOT_ZZZ_D CDOT_ZZZ_S
}

def AArch64MIR_cfinv : AArch64MIR_TargetOp<"cfinv", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CFINV
}

def AArch64MIR_clasta : AArch64MIR_TargetOp<"clasta", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CLASTA_RPZ_B CLASTA_RPZ_D CLASTA_RPZ_H CLASTA_RPZ_S CLASTA_VPZ_B CLASTA_VPZ_D CLASTA_VPZ_H CLASTA_VPZ_S CLASTA_ZPZ_B CLASTA_ZPZ_D CLASTA_ZPZ_H CLASTA_ZPZ_S
}

def AArch64MIR_clastb : AArch64MIR_TargetOp<"clastb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CLASTB_RPZ_B CLASTB_RPZ_D CLASTB_RPZ_H CLASTB_RPZ_S CLASTB_VPZ_B CLASTB_VPZ_D CLASTB_VPZ_H CLASTB_VPZ_S CLASTB_ZPZ_B CLASTB_ZPZ_D CLASTB_ZPZ_H CLASTB_ZPZ_S
}

def AArch64MIR_clrex : AArch64MIR_TargetOp<"clrex", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CLREX
}

def AArch64MIR_cls : AArch64MIR_TargetOp<"cls", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CLSWr CLSXr CLS_ZPmZ_B CLS_ZPmZ_D CLS_ZPmZ_H CLS_ZPmZ_S CLS_ZPzZ_B CLS_ZPzZ_D CLS_ZPzZ_H CLS_ZPzZ_S CLSv2i32 CLSv4i16 CLSv4i32 CLSv8i8 CLSv8i16 CLSv16i8
}

def AArch64MIR_clz : AArch64MIR_TargetOp<"clz", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CLZWr CLZXr CLZ_ZPmZ_B CLZ_ZPmZ_D CLZ_ZPmZ_H CLZ_ZPmZ_S CLZ_ZPzZ_B CLZ_ZPzZ_D CLZ_ZPzZ_H CLZ_ZPzZ_S CLZv2i32 CLZv4i16 CLZv4i32 CLZv8i8 CLZv8i16 CLZv16i8
}

def AArch64MIR_cmeq : AArch64MIR_TargetOp<"cmeq", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CMEQv1i64 CMEQv1i64rz CMEQv2i32 CMEQv2i32rz CMEQv2i64 CMEQv2i64rz CMEQv4i16 CMEQv4i16rz CMEQv4i32 CMEQv4i32rz CMEQv8i8 CMEQv8i8rz CMEQv8i16 CMEQv8i16rz CMEQv16i8 CMEQv16i8rz
}

def AArch64MIR_cmge : AArch64MIR_TargetOp<"cmge", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CMGEv1i64 CMGEv1i64rz CMGEv2i32 CMGEv2i32rz CMGEv2i64 CMGEv2i64rz CMGEv4i16 CMGEv4i16rz CMGEv4i32 CMGEv4i32rz CMGEv8i8 CMGEv8i8rz CMGEv8i16 CMGEv8i16rz CMGEv16i8 CMGEv16i8rz
}

def AArch64MIR_cmgt : AArch64MIR_TargetOp<"cmgt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CMGTv1i64 CMGTv1i64rz CMGTv2i32 CMGTv2i32rz CMGTv2i64 CMGTv2i64rz CMGTv4i16 CMGTv4i16rz CMGTv4i32 CMGTv4i32rz CMGTv8i8 CMGTv8i8rz CMGTv8i16 CMGTv8i16rz CMGTv16i8 CMGTv16i8rz
}

def AArch64MIR_cmhi : AArch64MIR_TargetOp<"cmhi", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CMHIv1i64 CMHIv2i32 CMHIv2i64 CMHIv4i16 CMHIv4i32 CMHIv8i8 CMHIv8i16 CMHIv16i8
}

def AArch64MIR_cmhs : AArch64MIR_TargetOp<"cmhs", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CMHSv1i64 CMHSv2i32 CMHSv2i64 CMHSv4i16 CMHSv4i32 CMHSv8i8 CMHSv8i16 CMHSv16i8
}

def AArch64MIR_cmla : AArch64MIR_TargetOp<"cmla", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CMLA_ZZZI_H CMLA_ZZZI_S CMLA_ZZZ_B CMLA_ZZZ_D CMLA_ZZZ_H CMLA_ZZZ_S
}

def AArch64MIR_cmle : AArch64MIR_TargetOp<"cmle", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CMLEv1i64rz CMLEv2i32rz CMLEv2i64rz CMLEv4i16rz CMLEv4i32rz CMLEv8i8rz CMLEv8i16rz CMLEv16i8rz
}

def AArch64MIR_cmlt : AArch64MIR_TargetOp<"cmlt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CMLTv1i64rz CMLTv2i32rz CMLTv2i64rz CMLTv4i16rz CMLTv4i32rz CMLTv8i8rz CMLTv8i16rz CMLTv16i8rz
}

def AArch64MIR_cmpeq : AArch64MIR_TargetOp<"cmpeq", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CMPEQ_PPzZI_B CMPEQ_PPzZI_D CMPEQ_PPzZI_H CMPEQ_PPzZI_S CMPEQ_PPzZZ_B CMPEQ_PPzZZ_D CMPEQ_PPzZZ_H CMPEQ_PPzZZ_S CMPEQ_WIDE_PPzZZ_B CMPEQ_WIDE_PPzZZ_H CMPEQ_WIDE_PPzZZ_S
}

def AArch64MIR_cmpge : AArch64MIR_TargetOp<"cmpge", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CMPGE_PPzZI_B CMPGE_PPzZI_D CMPGE_PPzZI_H CMPGE_PPzZI_S CMPGE_PPzZZ_B CMPGE_PPzZZ_D CMPGE_PPzZZ_H CMPGE_PPzZZ_S CMPGE_WIDE_PPzZZ_B CMPGE_WIDE_PPzZZ_H CMPGE_WIDE_PPzZZ_S
}

def AArch64MIR_cmpgt : AArch64MIR_TargetOp<"cmpgt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CMPGT_PPzZI_B CMPGT_PPzZI_D CMPGT_PPzZI_H CMPGT_PPzZI_S CMPGT_PPzZZ_B CMPGT_PPzZZ_D CMPGT_PPzZZ_H CMPGT_PPzZZ_S CMPGT_WIDE_PPzZZ_B CMPGT_WIDE_PPzZZ_H CMPGT_WIDE_PPzZZ_S
}

def AArch64MIR_cmphi : AArch64MIR_TargetOp<"cmphi", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CMPHI_PPzZI_B CMPHI_PPzZI_D CMPHI_PPzZI_H CMPHI_PPzZI_S CMPHI_PPzZZ_B CMPHI_PPzZZ_D CMPHI_PPzZZ_H CMPHI_PPzZZ_S CMPHI_WIDE_PPzZZ_B CMPHI_WIDE_PPzZZ_H CMPHI_WIDE_PPzZZ_S
}

def AArch64MIR_cmphs : AArch64MIR_TargetOp<"cmphs", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CMPHS_PPzZI_B CMPHS_PPzZI_D CMPHS_PPzZI_H CMPHS_PPzZI_S CMPHS_PPzZZ_B CMPHS_PPzZZ_D CMPHS_PPzZZ_H CMPHS_PPzZZ_S CMPHS_WIDE_PPzZZ_B CMPHS_WIDE_PPzZZ_H CMPHS_WIDE_PPzZZ_S
}

def AArch64MIR_cmple : AArch64MIR_TargetOp<"cmple", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CMPLE_PPzZI_B CMPLE_PPzZI_D CMPLE_PPzZI_H CMPLE_PPzZI_S CMPLE_WIDE_PPzZZ_B CMPLE_WIDE_PPzZZ_H CMPLE_WIDE_PPzZZ_S
}

def AArch64MIR_cmplo : AArch64MIR_TargetOp<"cmplo", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CMPLO_PPzZI_B CMPLO_PPzZI_D CMPLO_PPzZI_H CMPLO_PPzZI_S CMPLO_WIDE_PPzZZ_B CMPLO_WIDE_PPzZZ_H CMPLO_WIDE_PPzZZ_S
}

def AArch64MIR_cmpls : AArch64MIR_TargetOp<"cmpls", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CMPLS_PPzZI_B CMPLS_PPzZI_D CMPLS_PPzZI_H CMPLS_PPzZI_S CMPLS_WIDE_PPzZZ_B CMPLS_WIDE_PPzZZ_H CMPLS_WIDE_PPzZZ_S
}

def AArch64MIR_cmplt : AArch64MIR_TargetOp<"cmplt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CMPLT_PPzZI_B CMPLT_PPzZI_D CMPLT_PPzZI_H CMPLT_PPzZI_S CMPLT_WIDE_PPzZZ_B CMPLT_WIDE_PPzZZ_H CMPLT_WIDE_PPzZZ_S
}

def AArch64MIR_cmpne : AArch64MIR_TargetOp<"cmpne", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CMPNE_PPzZI_B CMPNE_PPzZI_D CMPNE_PPzZI_H CMPNE_PPzZI_S CMPNE_PPzZZ_B CMPNE_PPzZZ_D CMPNE_PPzZZ_H CMPNE_PPzZZ_S CMPNE_WIDE_PPzZZ_B CMPNE_WIDE_PPzZZ_H CMPNE_WIDE_PPzZZ_S
}

def AArch64MIR_cmtst : AArch64MIR_TargetOp<"cmtst", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CMTSTv1i64 CMTSTv2i32 CMTSTv2i64 CMTSTv4i16 CMTSTv4i32 CMTSTv8i8 CMTSTv8i16 CMTSTv16i8
}

def AArch64MIR_cnot : AArch64MIR_TargetOp<"cnot", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CNOT_ZPmZ_B CNOT_ZPmZ_D CNOT_ZPmZ_H CNOT_ZPmZ_S CNOT_ZPzZ_B CNOT_ZPzZ_D CNOT_ZPzZ_H CNOT_ZPzZ_S
}

def AArch64MIR_cntb : AArch64MIR_TargetOp<"cntb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CNTB_XPiI
}

def AArch64MIR_cntd : AArch64MIR_TargetOp<"cntd", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CNTD_XPiI
}

def AArch64MIR_cnth : AArch64MIR_TargetOp<"cnth", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CNTH_XPiI
}

def AArch64MIR_cntp : AArch64MIR_TargetOp<"cntp", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CNTP_XCI_B CNTP_XCI_D CNTP_XCI_H CNTP_XCI_S CNTP_XPP_B CNTP_XPP_D CNTP_XPP_H CNTP_XPP_S
}

def AArch64MIR_cntw : AArch64MIR_TargetOp<"cntw", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CNTW_XPiI
}

def AArch64MIR_cnt : AArch64MIR_TargetOp<"cnt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CNTWr CNTXr CNT_ZPmZ_B CNT_ZPmZ_D CNT_ZPmZ_H CNT_ZPmZ_S CNT_ZPzZ_B CNT_ZPzZ_D CNT_ZPzZ_H CNT_ZPzZ_S CNTv8i8 CNTv16i8
}

def AArch64MIR_compact : AArch64MIR_TargetOp<"compact", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: COMPACT_ZPZ_B COMPACT_ZPZ_D COMPACT_ZPZ_H COMPACT_ZPZ_S
}

def AArch64MIR_cpye : AArch64MIR_TargetOp<"cpye", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYE
}

def AArch64MIR_cpyen : AArch64MIR_TargetOp<"cpyen", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYEN
}

def AArch64MIR_cpyern : AArch64MIR_TargetOp<"cpyern", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYERN
}

def AArch64MIR_cpyert : AArch64MIR_TargetOp<"cpyert", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYERT
}

def AArch64MIR_cpyertn : AArch64MIR_TargetOp<"cpyertn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYERTN
}

def AArch64MIR_cpyertrn : AArch64MIR_TargetOp<"cpyertrn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYERTRN
}

def AArch64MIR_cpyertwn : AArch64MIR_TargetOp<"cpyertwn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYERTWN
}

def AArch64MIR_cpyet : AArch64MIR_TargetOp<"cpyet", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYET
}

def AArch64MIR_cpyetn : AArch64MIR_TargetOp<"cpyetn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYETN
}

def AArch64MIR_cpyetrn : AArch64MIR_TargetOp<"cpyetrn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYETRN
}

def AArch64MIR_cpyetwn : AArch64MIR_TargetOp<"cpyetwn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYETWN
}

def AArch64MIR_cpyewn : AArch64MIR_TargetOp<"cpyewn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYEWN
}

def AArch64MIR_cpyewt : AArch64MIR_TargetOp<"cpyewt", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYEWT
}

def AArch64MIR_cpyewtn : AArch64MIR_TargetOp<"cpyewtn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYEWTN
}

def AArch64MIR_cpyewtrn : AArch64MIR_TargetOp<"cpyewtrn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYEWTRN
}

def AArch64MIR_cpyewtwn : AArch64MIR_TargetOp<"cpyewtwn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYEWTWN
}

def AArch64MIR_cpyfe : AArch64MIR_TargetOp<"cpyfe", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFE
}

def AArch64MIR_cpyfen : AArch64MIR_TargetOp<"cpyfen", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFEN
}

def AArch64MIR_cpyfern : AArch64MIR_TargetOp<"cpyfern", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFERN
}

def AArch64MIR_cpyfert : AArch64MIR_TargetOp<"cpyfert", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFERT
}

def AArch64MIR_cpyfertn : AArch64MIR_TargetOp<"cpyfertn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFERTN
}

def AArch64MIR_cpyfertrn : AArch64MIR_TargetOp<"cpyfertrn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFERTRN
}

def AArch64MIR_cpyfertwn : AArch64MIR_TargetOp<"cpyfertwn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFERTWN
}

def AArch64MIR_cpyfet : AArch64MIR_TargetOp<"cpyfet", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFET
}

def AArch64MIR_cpyfetn : AArch64MIR_TargetOp<"cpyfetn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFETN
}

def AArch64MIR_cpyfetrn : AArch64MIR_TargetOp<"cpyfetrn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFETRN
}

def AArch64MIR_cpyfetwn : AArch64MIR_TargetOp<"cpyfetwn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFETWN
}

def AArch64MIR_cpyfewn : AArch64MIR_TargetOp<"cpyfewn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFEWN
}

def AArch64MIR_cpyfewt : AArch64MIR_TargetOp<"cpyfewt", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFEWT
}

def AArch64MIR_cpyfewtn : AArch64MIR_TargetOp<"cpyfewtn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFEWTN
}

def AArch64MIR_cpyfewtrn : AArch64MIR_TargetOp<"cpyfewtrn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFEWTRN
}

def AArch64MIR_cpyfewtwn : AArch64MIR_TargetOp<"cpyfewtwn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFEWTWN
}

def AArch64MIR_cpyfm : AArch64MIR_TargetOp<"cpyfm", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFM
}

def AArch64MIR_cpyfmn : AArch64MIR_TargetOp<"cpyfmn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFMN
}

def AArch64MIR_cpyfmrn : AArch64MIR_TargetOp<"cpyfmrn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFMRN
}

def AArch64MIR_cpyfmrt : AArch64MIR_TargetOp<"cpyfmrt", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFMRT
}

def AArch64MIR_cpyfmrtn : AArch64MIR_TargetOp<"cpyfmrtn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFMRTN
}

def AArch64MIR_cpyfmrtrn : AArch64MIR_TargetOp<"cpyfmrtrn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFMRTRN
}

def AArch64MIR_cpyfmrtwn : AArch64MIR_TargetOp<"cpyfmrtwn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFMRTWN
}

def AArch64MIR_cpyfmt : AArch64MIR_TargetOp<"cpyfmt", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFMT
}

def AArch64MIR_cpyfmtn : AArch64MIR_TargetOp<"cpyfmtn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFMTN
}

def AArch64MIR_cpyfmtrn : AArch64MIR_TargetOp<"cpyfmtrn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFMTRN
}

def AArch64MIR_cpyfmtwn : AArch64MIR_TargetOp<"cpyfmtwn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFMTWN
}

def AArch64MIR_cpyfmwn : AArch64MIR_TargetOp<"cpyfmwn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFMWN
}

def AArch64MIR_cpyfmwt : AArch64MIR_TargetOp<"cpyfmwt", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFMWT
}

def AArch64MIR_cpyfmwtn : AArch64MIR_TargetOp<"cpyfmwtn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFMWTN
}

def AArch64MIR_cpyfmwtrn : AArch64MIR_TargetOp<"cpyfmwtrn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFMWTRN
}

def AArch64MIR_cpyfmwtwn : AArch64MIR_TargetOp<"cpyfmwtwn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFMWTWN
}

def AArch64MIR_cpyfp : AArch64MIR_TargetOp<"cpyfp", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFP
}

def AArch64MIR_cpyfpn : AArch64MIR_TargetOp<"cpyfpn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFPN
}

def AArch64MIR_cpyfprn : AArch64MIR_TargetOp<"cpyfprn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFPRN
}

def AArch64MIR_cpyfprt : AArch64MIR_TargetOp<"cpyfprt", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFPRT
}

def AArch64MIR_cpyfprtn : AArch64MIR_TargetOp<"cpyfprtn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFPRTN
}

def AArch64MIR_cpyfprtrn : AArch64MIR_TargetOp<"cpyfprtrn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFPRTRN
}

def AArch64MIR_cpyfprtwn : AArch64MIR_TargetOp<"cpyfprtwn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFPRTWN
}

def AArch64MIR_cpyfpt : AArch64MIR_TargetOp<"cpyfpt", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFPT
}

def AArch64MIR_cpyfptn : AArch64MIR_TargetOp<"cpyfptn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFPTN
}

def AArch64MIR_cpyfptrn : AArch64MIR_TargetOp<"cpyfptrn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFPTRN
}

def AArch64MIR_cpyfptwn : AArch64MIR_TargetOp<"cpyfptwn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFPTWN
}

def AArch64MIR_cpyfpwn : AArch64MIR_TargetOp<"cpyfpwn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFPWN
}

def AArch64MIR_cpyfpwt : AArch64MIR_TargetOp<"cpyfpwt", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFPWT
}

def AArch64MIR_cpyfpwtn : AArch64MIR_TargetOp<"cpyfpwtn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFPWTN
}

def AArch64MIR_cpyfpwtrn : AArch64MIR_TargetOp<"cpyfpwtrn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFPWTRN
}

def AArch64MIR_cpyfpwtwn : AArch64MIR_TargetOp<"cpyfpwtwn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYFPWTWN
}

def AArch64MIR_cpym : AArch64MIR_TargetOp<"cpym", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYM
}

def AArch64MIR_cpymn : AArch64MIR_TargetOp<"cpymn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYMN
}

def AArch64MIR_cpymrn : AArch64MIR_TargetOp<"cpymrn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYMRN
}

def AArch64MIR_cpymrt : AArch64MIR_TargetOp<"cpymrt", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYMRT
}

def AArch64MIR_cpymrtn : AArch64MIR_TargetOp<"cpymrtn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYMRTN
}

def AArch64MIR_cpymrtrn : AArch64MIR_TargetOp<"cpymrtrn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYMRTRN
}

def AArch64MIR_cpymrtwn : AArch64MIR_TargetOp<"cpymrtwn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYMRTWN
}

def AArch64MIR_cpymt : AArch64MIR_TargetOp<"cpymt", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYMT
}

def AArch64MIR_cpymtn : AArch64MIR_TargetOp<"cpymtn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYMTN
}

def AArch64MIR_cpymtrn : AArch64MIR_TargetOp<"cpymtrn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYMTRN
}

def AArch64MIR_cpymtwn : AArch64MIR_TargetOp<"cpymtwn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYMTWN
}

def AArch64MIR_cpymwn : AArch64MIR_TargetOp<"cpymwn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYMWN
}

def AArch64MIR_cpymwt : AArch64MIR_TargetOp<"cpymwt", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYMWT
}

def AArch64MIR_cpymwtn : AArch64MIR_TargetOp<"cpymwtn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYMWTN
}

def AArch64MIR_cpymwtrn : AArch64MIR_TargetOp<"cpymwtrn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYMWTRN
}

def AArch64MIR_cpymwtwn : AArch64MIR_TargetOp<"cpymwtwn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYMWTWN
}

def AArch64MIR_cpyp : AArch64MIR_TargetOp<"cpyp", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYP
}

def AArch64MIR_cpypn : AArch64MIR_TargetOp<"cpypn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYPN
}

def AArch64MIR_cpyprn : AArch64MIR_TargetOp<"cpyprn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYPRN
}

def AArch64MIR_cpyprt : AArch64MIR_TargetOp<"cpyprt", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYPRT
}

def AArch64MIR_cpyprtn : AArch64MIR_TargetOp<"cpyprtn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYPRTN
}

def AArch64MIR_cpyprtrn : AArch64MIR_TargetOp<"cpyprtrn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYPRTRN
}

def AArch64MIR_cpyprtwn : AArch64MIR_TargetOp<"cpyprtwn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYPRTWN
}

def AArch64MIR_cpypt : AArch64MIR_TargetOp<"cpypt", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYPT
}

def AArch64MIR_cpyptn : AArch64MIR_TargetOp<"cpyptn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYPTN
}

def AArch64MIR_cpyptrn : AArch64MIR_TargetOp<"cpyptrn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYPTRN
}

def AArch64MIR_cpyptwn : AArch64MIR_TargetOp<"cpyptwn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYPTWN
}

def AArch64MIR_cpypwn : AArch64MIR_TargetOp<"cpypwn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYPWN
}

def AArch64MIR_cpypwt : AArch64MIR_TargetOp<"cpypwt", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYPWT
}

def AArch64MIR_cpypwtn : AArch64MIR_TargetOp<"cpypwtn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYPWTN
}

def AArch64MIR_cpypwtrn : AArch64MIR_TargetOp<"cpypwtrn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYPWTRN
}

def AArch64MIR_cpypwtwn : AArch64MIR_TargetOp<"cpypwtwn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPYPWTWN
}

def AArch64MIR_cpy : AArch64MIR_TargetOp<"cpy", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CPY_ZPmI_B CPY_ZPmI_D CPY_ZPmI_H CPY_ZPmI_S CPY_ZPmR_B CPY_ZPmR_D CPY_ZPmR_H CPY_ZPmR_S CPY_ZPmV_B CPY_ZPmV_D CPY_ZPmV_H CPY_ZPmV_S CPY_ZPzI_B CPY_ZPzI_D CPY_ZPzI_H CPY_ZPzI_S
}

def AArch64MIR_crc32b : AArch64MIR_TargetOp<"crc32b", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CRC32Brr
}

def AArch64MIR_crc32cb : AArch64MIR_TargetOp<"crc32cb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CRC32CBrr
}

def AArch64MIR_crc32ch : AArch64MIR_TargetOp<"crc32ch", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CRC32CHrr
}

def AArch64MIR_crc32cw : AArch64MIR_TargetOp<"crc32cw", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CRC32CWrr
}

def AArch64MIR_crc32cx : AArch64MIR_TargetOp<"crc32cx", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CRC32CXrr
}

def AArch64MIR_crc32h : AArch64MIR_TargetOp<"crc32h", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CRC32Hrr
}

def AArch64MIR_crc32w : AArch64MIR_TargetOp<"crc32w", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CRC32Wrr
}

def AArch64MIR_crc32x : AArch64MIR_TargetOp<"crc32x", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CRC32Xrr
}

def AArch64MIR_csel : AArch64MIR_TargetOp<"csel", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CSELWr CSELXr
}

def AArch64MIR_csinc : AArch64MIR_TargetOp<"csinc", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CSINCWr CSINCXr
}

def AArch64MIR_csinv : AArch64MIR_TargetOp<"csinv", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CSINVWr CSINVXr
}

def AArch64MIR_csneg : AArch64MIR_TargetOp<"csneg", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CSNEGWr CSNEGXr
}

def AArch64MIR_ctermeq : AArch64MIR_TargetOp<"ctermeq", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CTERMEQ_WW CTERMEQ_XX
}

def AArch64MIR_ctermne : AArch64MIR_TargetOp<"ctermne", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CTERMNE_WW CTERMNE_XX
}

def AArch64MIR_ctz : AArch64MIR_TargetOp<"ctz", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: CTZWr CTZXr
}

def AArch64MIR_dcps1 : AArch64MIR_TargetOp<"dcps1", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: DCPS1
}

def AArch64MIR_dcps2 : AArch64MIR_TargetOp<"dcps2", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: DCPS2
}

def AArch64MIR_dcps3 : AArch64MIR_TargetOp<"dcps3", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: DCPS3
}

def AArch64MIR_decb : AArch64MIR_TargetOp<"decb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: DECB_XPiI
}

def AArch64MIR_decd : AArch64MIR_TargetOp<"decd", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: DECD_XPiI DECD_ZPiI
}

def AArch64MIR_dech : AArch64MIR_TargetOp<"dech", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: DECH_XPiI DECH_ZPiI
}

def AArch64MIR_decp : AArch64MIR_TargetOp<"decp", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: DECP_XP_B DECP_XP_D DECP_XP_H DECP_XP_S DECP_ZP_D DECP_ZP_H DECP_ZP_S
}

def AArch64MIR_decw : AArch64MIR_TargetOp<"decw", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: DECW_XPiI DECW_ZPiI
}

def AArch64MIR_dmb : AArch64MIR_TargetOp<"dmb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: DMB
}

def AArch64MIR_drps : AArch64MIR_TargetOp<"drps", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: DRPS
}

def AArch64MIR_dsb : AArch64MIR_TargetOp<"dsb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: DSB DSBnXS
}

def AArch64MIR_dupm : AArch64MIR_TargetOp<"dupm", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: DUPM_ZI
}

def AArch64MIR_dupq : AArch64MIR_TargetOp<"dupq", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: DUPQ_ZZI_B DUPQ_ZZI_D DUPQ_ZZI_H DUPQ_ZZI_S
}

def AArch64MIR_dup : AArch64MIR_TargetOp<"dup", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: DUP_ZI_B DUP_ZI_D DUP_ZI_H DUP_ZI_S DUP_ZR_B DUP_ZR_D DUP_ZR_H DUP_ZR_S DUP_ZZI_B DUP_ZZI_D DUP_ZZI_H DUP_ZZI_Q DUP_ZZI_S DUPv2i32gpr DUPv2i32lane DUPv2i64gpr DUPv2i64lane DUPv4i16gpr DUPv4i16lane DUPv4i32gpr DUPv4i32lane DUPv8i8gpr DUPv8i8lane DUPv8i16gpr DUPv8i16lane DUPv16i8gpr DUPv16i8lane
}

def AArch64MIR_mov : AArch64MIR_TargetOp<"mov", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: DUPi8 DUPi16 DUPi32 DUPi64
}

def AArch64MIR_eon : AArch64MIR_TargetOp<"eon", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: EONWrs EONXrs
}

def AArch64MIR_eor3 : AArch64MIR_TargetOp<"eor3", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: EOR3 EOR3_ZZZZ
}

def AArch64MIR_eorbt : AArch64MIR_TargetOp<"eorbt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: EORBT_ZZZ_B EORBT_ZZZ_D EORBT_ZZZ_H EORBT_ZZZ_S
}

def AArch64MIR_eorqv : AArch64MIR_TargetOp<"eorqv", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: EORQV_VPZ_B EORQV_VPZ_D EORQV_VPZ_H EORQV_VPZ_S
}

def AArch64MIR_eors : AArch64MIR_TargetOp<"eors", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: EORS_PPzPP
}

def AArch64MIR_eortb : AArch64MIR_TargetOp<"eortb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: EORTB_ZZZ_B EORTB_ZZZ_D EORTB_ZZZ_H EORTB_ZZZ_S
}

def AArch64MIR_eorv : AArch64MIR_TargetOp<"eorv", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: EORV_VPZ_B EORV_VPZ_D EORV_VPZ_H EORV_VPZ_S
}

def AArch64MIR_eor : AArch64MIR_TargetOp<"eor", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: EORWri EORWrs EORXri EORXrs EOR_PPzPP EOR_ZI EOR_ZPmZ_B EOR_ZPmZ_D EOR_ZPmZ_H EOR_ZPmZ_S EOR_ZZZ EORv8i8 EORv16i8
}

def AArch64MIR_eret : AArch64MIR_TargetOp<"eret", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ERET
}

def AArch64MIR_eretaa : AArch64MIR_TargetOp<"eretaa", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ERETAA
}

def AArch64MIR_eretab : AArch64MIR_TargetOp<"eretab", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ERETAB
}

def AArch64MIR_expand : AArch64MIR_TargetOp<"expand", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: EXPAND_ZPZ_B EXPAND_ZPZ_D EXPAND_ZPZ_H EXPAND_ZPZ_S
}

def AArch64MIR_extq : AArch64MIR_TargetOp<"extq", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: EXTQ_ZZI
}

def AArch64MIR_mova : AArch64MIR_TargetOp<"mova", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: EXTRACT_ZPMXI_H_B EXTRACT_ZPMXI_H_D EXTRACT_ZPMXI_H_H EXTRACT_ZPMXI_H_Q EXTRACT_ZPMXI_H_S EXTRACT_ZPMXI_V_B EXTRACT_ZPMXI_V_D EXTRACT_ZPMXI_V_H EXTRACT_ZPMXI_V_Q EXTRACT_ZPMXI_V_S INSERT_MXIPZ_H_B INSERT_MXIPZ_H_D INSERT_MXIPZ_H_H INSERT_MXIPZ_H_Q INSERT_MXIPZ_H_S INSERT_MXIPZ_V_B INSERT_MXIPZ_V_D INSERT_MXIPZ_V_H INSERT_MXIPZ_V_Q INSERT_MXIPZ_V_S MOVA_2ZMXI_H_B MOVA_2ZMXI_H_D MOVA_2ZMXI_H_H MOVA_2ZMXI_H_S MOVA_2ZMXI_V_B MOVA_2ZMXI_V_D MOVA_2ZMXI_V_H MOVA_2ZMXI_V_S MOVA_4ZMXI_H_B MOVA_4ZMXI_H_D MOVA_4ZMXI_H_H MOVA_4ZMXI_H_S MOVA_4ZMXI_V_B MOVA_4ZMXI_V_D MOVA_4ZMXI_V_H MOVA_4ZMXI_V_S MOVA_MXI2Z_H_B MOVA_MXI2Z_H_D MOVA_MXI2Z_H_H MOVA_MXI2Z_H_S MOVA_MXI2Z_V_B MOVA_MXI2Z_V_D MOVA_MXI2Z_V_H MOVA_MXI2Z_V_S MOVA_MXI4Z_H_B MOVA_MXI4Z_H_D MOVA_MXI4Z_H_H MOVA_MXI4Z_H_S MOVA_MXI4Z_V_B MOVA_MXI4Z_V_D MOVA_MXI4Z_V_H MOVA_MXI4Z_V_S MOVA_VG2_2ZMXI MOVA_VG2_MXI2Z MOVA_VG4_4ZMXI MOVA_VG4_MXI4Z
}

def AArch64MIR_extr : AArch64MIR_TargetOp<"extr", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: EXTRWrri EXTRXrri
}

def AArch64MIR_ext : AArch64MIR_TargetOp<"ext", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: EXT_ZZI EXT_ZZI_B EXTv8i8 EXTv16i8
}

def AArch64MIR_f1cvtl : AArch64MIR_TargetOp<"f1cvtl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: F1CVTL F1CVTL_2ZZ_BtoH
}

def AArch64MIR_f1cvtl2 : AArch64MIR_TargetOp<"f1cvtl2", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: F1CVTL2
}

def AArch64MIR_f1cvtlt : AArch64MIR_TargetOp<"f1cvtlt", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: F1CVTLT_ZZ_BtoH
}

def AArch64MIR_f1cvt : AArch64MIR_TargetOp<"f1cvt", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: F1CVT_2ZZ_BtoH F1CVT_ZZ_BtoH
}

def AArch64MIR_f2cvtl : AArch64MIR_TargetOp<"f2cvtl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: F2CVTL F2CVTL_2ZZ_BtoH
}

def AArch64MIR_f2cvtl2 : AArch64MIR_TargetOp<"f2cvtl2", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: F2CVTL2
}

def AArch64MIR_f2cvtlt : AArch64MIR_TargetOp<"f2cvtlt", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: F2CVTLT_ZZ_BtoH
}

def AArch64MIR_f2cvt : AArch64MIR_TargetOp<"f2cvt", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: F2CVT_2ZZ_BtoH F2CVT_ZZ_BtoH
}

def AArch64MIR_fabd : AArch64MIR_TargetOp<"fabd", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FABD16 FABD32 FABD64 FABD_ZPmZ_D FABD_ZPmZ_H FABD_ZPmZ_S FABDv2f32 FABDv2f64 FABDv4f16 FABDv4f32 FABDv8f16
}

def AArch64MIR_fabs : AArch64MIR_TargetOp<"fabs", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FABSDr FABSHr FABSSr FABS_ZPmZ_D FABS_ZPmZ_H FABS_ZPmZ_S FABS_ZPzZ_D FABS_ZPzZ_H FABS_ZPzZ_S FABSv2f32 FABSv2f64 FABSv4f16 FABSv4f32 FABSv8f16
}

def AArch64MIR_facge : AArch64MIR_TargetOp<"facge", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FACGE16 FACGE32 FACGE64 FACGE_PPzZZ_D FACGE_PPzZZ_H FACGE_PPzZZ_S FACGEv2f32 FACGEv2f64 FACGEv4f16 FACGEv4f32 FACGEv8f16
}

def AArch64MIR_facgt : AArch64MIR_TargetOp<"facgt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FACGT16 FACGT32 FACGT64 FACGT_PPzZZ_D FACGT_PPzZZ_H FACGT_PPzZZ_S FACGTv2f32 FACGTv2f64 FACGTv4f16 FACGTv4f32 FACGTv8f16
}

def AArch64MIR_fadda : AArch64MIR_TargetOp<"fadda", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FADDA_VPZ_D FADDA_VPZ_H FADDA_VPZ_S
}

def AArch64MIR_fadd : AArch64MIR_TargetOp<"fadd", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FADDDrr FADDHrr FADDSrr FADD_VG2_M2Z_D FADD_VG2_M2Z_H FADD_VG2_M2Z_S FADD_VG4_M4Z_D FADD_VG4_M4Z_H FADD_VG4_M4Z_S FADD_ZPmI_D FADD_ZPmI_H FADD_ZPmI_S FADD_ZPmZ_D FADD_ZPmZ_H FADD_ZPmZ_S FADD_ZZZ_D FADD_ZZZ_H FADD_ZZZ_S FADDv2f32 FADDv2f64 FADDv4f16 FADDv4f32 FADDv8f16
}

def AArch64MIR_faddp : AArch64MIR_TargetOp<"faddp", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FADDP_ZPmZZ_D FADDP_ZPmZZ_H FADDP_ZPmZZ_S FADDPv2f32 FADDPv2f64 FADDPv2i16p FADDPv2i32p FADDPv2i64p FADDPv4f16 FADDPv4f32 FADDPv8f16
}

def AArch64MIR_faddqv : AArch64MIR_TargetOp<"faddqv", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FADDQV_D FADDQV_H FADDQV_S
}

def AArch64MIR_faddv : AArch64MIR_TargetOp<"faddv", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FADDV_VPZ_D FADDV_VPZ_H FADDV_VPZ_S
}

def AArch64MIR_famax : AArch64MIR_TargetOp<"famax", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FAMAX_2Z2Z_D FAMAX_2Z2Z_H FAMAX_2Z2Z_S FAMAX_4Z4Z_D FAMAX_4Z4Z_H FAMAX_4Z4Z_S FAMAX_ZPmZ_D FAMAX_ZPmZ_H FAMAX_ZPmZ_S FAMAXv2f32 FAMAXv2f64 FAMAXv4f16 FAMAXv4f32 FAMAXv8f16
}

def AArch64MIR_famin : AArch64MIR_TargetOp<"famin", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FAMIN_2Z2Z_D FAMIN_2Z2Z_H FAMIN_2Z2Z_S FAMIN_4Z4Z_D FAMIN_4Z4Z_H FAMIN_4Z4Z_S FAMIN_ZPmZ_D FAMIN_ZPmZ_H FAMIN_ZPmZ_S FAMINv2f32 FAMINv2f64 FAMINv4f16 FAMINv4f32 FAMINv8f16
}

def AArch64MIR_fcadd : AArch64MIR_TargetOp<"fcadd", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FCADD_ZPmZ_D FCADD_ZPmZ_H FCADD_ZPmZ_S FCADDv2f32 FCADDv2f64 FCADDv4f16 FCADDv4f32 FCADDv8f16
}

def AArch64MIR_fccmp : AArch64MIR_TargetOp<"fccmp", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FCCMPDrr FCCMPHrr FCCMPSrr
}

def AArch64MIR_fccmpe : AArch64MIR_TargetOp<"fccmpe", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FCCMPEDrr FCCMPEHrr FCCMPESrr
}

def AArch64MIR_fclamp : AArch64MIR_TargetOp<"fclamp", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FCLAMP_VG2_2Z2Z_D FCLAMP_VG2_2Z2Z_H FCLAMP_VG2_2Z2Z_S FCLAMP_VG4_4Z4Z_D FCLAMP_VG4_4Z4Z_H FCLAMP_VG4_4Z4Z_S FCLAMP_ZZZ_D FCLAMP_ZZZ_H FCLAMP_ZZZ_S
}

def AArch64MIR_fcmeq : AArch64MIR_TargetOp<"fcmeq", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FCMEQ16 FCMEQ32 FCMEQ64 FCMEQ_PPzZ0_D FCMEQ_PPzZ0_H FCMEQ_PPzZ0_S FCMEQ_PPzZZ_D FCMEQ_PPzZZ_H FCMEQ_PPzZZ_S FCMEQv1i16rz FCMEQv1i32rz FCMEQv1i64rz FCMEQv2f32 FCMEQv2f64 FCMEQv2i32rz FCMEQv2i64rz FCMEQv4f16 FCMEQv4f32 FCMEQv4i16rz FCMEQv4i32rz FCMEQv8f16 FCMEQv8i16rz
}

def AArch64MIR_fcmge : AArch64MIR_TargetOp<"fcmge", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FCMGE16 FCMGE32 FCMGE64 FCMGE_PPzZ0_D FCMGE_PPzZ0_H FCMGE_PPzZ0_S FCMGE_PPzZZ_D FCMGE_PPzZZ_H FCMGE_PPzZZ_S FCMGEv1i16rz FCMGEv1i32rz FCMGEv1i64rz FCMGEv2f32 FCMGEv2f64 FCMGEv2i32rz FCMGEv2i64rz FCMGEv4f16 FCMGEv4f32 FCMGEv4i16rz FCMGEv4i32rz FCMGEv8f16 FCMGEv8i16rz
}

def AArch64MIR_fcmgt : AArch64MIR_TargetOp<"fcmgt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FCMGT16 FCMGT32 FCMGT64 FCMGT_PPzZ0_D FCMGT_PPzZ0_H FCMGT_PPzZ0_S FCMGT_PPzZZ_D FCMGT_PPzZZ_H FCMGT_PPzZZ_S FCMGTv1i16rz FCMGTv1i32rz FCMGTv1i64rz FCMGTv2f32 FCMGTv2f64 FCMGTv2i32rz FCMGTv2i64rz FCMGTv4f16 FCMGTv4f32 FCMGTv4i16rz FCMGTv4i32rz FCMGTv8f16 FCMGTv8i16rz
}

def AArch64MIR_fcmla : AArch64MIR_TargetOp<"fcmla", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FCMLA_ZPmZZ_D FCMLA_ZPmZZ_H FCMLA_ZPmZZ_S FCMLA_ZZZI_H FCMLA_ZZZI_S FCMLAv2f32 FCMLAv2f64 FCMLAv4f16 FCMLAv4f16_indexed FCMLAv4f32 FCMLAv4f32_indexed FCMLAv8f16 FCMLAv8f16_indexed
}

def AArch64MIR_fcmle : AArch64MIR_TargetOp<"fcmle", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FCMLE_PPzZ0_D FCMLE_PPzZ0_H FCMLE_PPzZ0_S FCMLEv1i16rz FCMLEv1i32rz FCMLEv1i64rz FCMLEv2i32rz FCMLEv2i64rz FCMLEv4i16rz FCMLEv4i32rz FCMLEv8i16rz
}

def AArch64MIR_fcmlt : AArch64MIR_TargetOp<"fcmlt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FCMLT_PPzZ0_D FCMLT_PPzZ0_H FCMLT_PPzZ0_S FCMLTv1i16rz FCMLTv1i32rz FCMLTv1i64rz FCMLTv2i32rz FCMLTv2i64rz FCMLTv4i16rz FCMLTv4i32rz FCMLTv8i16rz
}

def AArch64MIR_fcmne : AArch64MIR_TargetOp<"fcmne", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FCMNE_PPzZ0_D FCMNE_PPzZ0_H FCMNE_PPzZ0_S FCMNE_PPzZZ_D FCMNE_PPzZZ_H FCMNE_PPzZZ_S
}

def AArch64MIR_fcmp : AArch64MIR_TargetOp<"fcmp", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FCMPDri FCMPDrr FCMPHri FCMPHrr FCMPSri FCMPSrr
}

def AArch64MIR_fcmpe : AArch64MIR_TargetOp<"fcmpe", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FCMPEDri FCMPEDrr FCMPEHri FCMPEHrr FCMPESri FCMPESrr
}

def AArch64MIR_fcmuo : AArch64MIR_TargetOp<"fcmuo", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FCMUO_PPzZZ_D FCMUO_PPzZZ_H FCMUO_PPzZZ_S
}

def AArch64MIR_fcpy : AArch64MIR_TargetOp<"fcpy", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FCPY_ZPmI_D FCPY_ZPmI_H FCPY_ZPmI_S
}

def AArch64MIR_fcsel : AArch64MIR_TargetOp<"fcsel", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FCSELDrrr FCSELHrrr FCSELSrrr
}

def AArch64MIR_fcvtas : AArch64MIR_TargetOp<"fcvtas", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FCVTASDHr FCVTASDSr FCVTASSDr FCVTASSHr FCVTASUWDr FCVTASUWHr FCVTASUWSr FCVTASUXDr FCVTASUXHr FCVTASUXSr FCVTASv1f16 FCVTASv1i32 FCVTASv1i64 FCVTASv2f32 FCVTASv2f64 FCVTASv4f16 FCVTASv4f32 FCVTASv8f16
}

def AArch64MIR_fcvtau : AArch64MIR_TargetOp<"fcvtau", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FCVTAUDHr FCVTAUDSr FCVTAUSDr FCVTAUSHr FCVTAUUWDr FCVTAUUWHr FCVTAUUWSr FCVTAUUXDr FCVTAUUXHr FCVTAUUXSr FCVTAUv1f16 FCVTAUv1i32 FCVTAUv1i64 FCVTAUv2f32 FCVTAUv2f64 FCVTAUv4f16 FCVTAUv4f32 FCVTAUv8f16
}

def AArch64MIR_fcvt : AArch64MIR_TargetOp<"fcvt", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FCVTDHr FCVTDSr FCVTHDr FCVTHSr FCVTSDr FCVTSHr FCVT_2ZZ_H_S FCVT_Z2Z_HtoB FCVT_Z2Z_StoH FCVT_Z4Z_StoB FCVT_ZPmZ_DtoH FCVT_ZPmZ_DtoS FCVT_ZPmZ_HtoD FCVT_ZPmZ_HtoS FCVT_ZPmZ_StoD FCVT_ZPmZ_StoH FCVT_ZPzZ_DtoH FCVT_ZPzZ_DtoS FCVT_ZPzZ_HtoD FCVT_ZPzZ_HtoS FCVT_ZPzZ_StoD FCVT_ZPzZ_StoH
}

def AArch64MIR_fcvtlt : AArch64MIR_TargetOp<"fcvtlt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FCVTLT_ZPmZ_HtoS FCVTLT_ZPmZ_StoD FCVTLT_ZPzZ_HtoS FCVTLT_ZPzZ_StoD
}

def AArch64MIR_fcvtl : AArch64MIR_TargetOp<"fcvtl", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FCVTL_2ZZ_H_S FCVTLv2i32 FCVTLv4i16
}

def AArch64MIR_fcvtl2 : AArch64MIR_TargetOp<"fcvtl2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FCVTLv4i32 FCVTLv8i16
}

def AArch64MIR_fcvtms : AArch64MIR_TargetOp<"fcvtms", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FCVTMSDHr FCVTMSDSr FCVTMSSDr FCVTMSSHr FCVTMSUWDr FCVTMSUWHr FCVTMSUWSr FCVTMSUXDr FCVTMSUXHr FCVTMSUXSr FCVTMSv1f16 FCVTMSv1i32 FCVTMSv1i64 FCVTMSv2f32 FCVTMSv2f64 FCVTMSv4f16 FCVTMSv4f32 FCVTMSv8f16
}

def AArch64MIR_fcvtmu : AArch64MIR_TargetOp<"fcvtmu", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FCVTMUDHr FCVTMUDSr FCVTMUSDr FCVTMUSHr FCVTMUUWDr FCVTMUUWHr FCVTMUUWSr FCVTMUUXDr FCVTMUUXHr FCVTMUUXSr FCVTMUv1f16 FCVTMUv1i32 FCVTMUv1i64 FCVTMUv2f32 FCVTMUv2f64 FCVTMUv4f16 FCVTMUv4f32 FCVTMUv8f16
}

def AArch64MIR_fcvtnb : AArch64MIR_TargetOp<"fcvtnb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FCVTNB_Z2Z_StoB
}

def AArch64MIR_fcvtns : AArch64MIR_TargetOp<"fcvtns", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FCVTNSDHr FCVTNSDSr FCVTNSSDr FCVTNSSHr FCVTNSUWDr FCVTNSUWHr FCVTNSUWSr FCVTNSUXDr FCVTNSUXHr FCVTNSUXSr FCVTNSv1f16 FCVTNSv1i32 FCVTNSv1i64 FCVTNSv2f32 FCVTNSv2f64 FCVTNSv4f16 FCVTNSv4f32 FCVTNSv8f16
}

def AArch64MIR_fcvtnt : AArch64MIR_TargetOp<"fcvtnt", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FCVTNT_Z2Z_StoB FCVTNT_ZPmZ_DtoS FCVTNT_ZPmZ_StoH FCVTNT_ZPzZ_DtoS FCVTNT_ZPzZ_StoH
}

def AArch64MIR_fcvtnu : AArch64MIR_TargetOp<"fcvtnu", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FCVTNUDHr FCVTNUDSr FCVTNUSDr FCVTNUSHr FCVTNUUWDr FCVTNUUWHr FCVTNUUWSr FCVTNUUXDr FCVTNUUXHr FCVTNUUXSr FCVTNUv1f16 FCVTNUv1i32 FCVTNUv1i64 FCVTNUv2f32 FCVTNUv2f64 FCVTNUv4f16 FCVTNUv4f32 FCVTNUv8f16
}

def AArch64MIR_fcvtn : AArch64MIR_TargetOp<"fcvtn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FCVTN_F16v8f8 FCVTN_F16v16f8 FCVTN_F32v8f8 FCVTN_Z2Z_HtoB FCVTN_Z2Z_StoH FCVTN_Z4Z_StoB FCVTNv2i32 FCVTNv4i16
}

def AArch64MIR_fcvtn2 : AArch64MIR_TargetOp<"fcvtn2", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FCVTN_F322v16f8 FCVTNv4i32 FCVTNv8i16
}

def AArch64MIR_fcvtps : AArch64MIR_TargetOp<"fcvtps", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FCVTPSDHr FCVTPSDSr FCVTPSSDr FCVTPSSHr FCVTPSUWDr FCVTPSUWHr FCVTPSUWSr FCVTPSUXDr FCVTPSUXHr FCVTPSUXSr FCVTPSv1f16 FCVTPSv1i32 FCVTPSv1i64 FCVTPSv2f32 FCVTPSv2f64 FCVTPSv4f16 FCVTPSv4f32 FCVTPSv8f16
}

def AArch64MIR_fcvtpu : AArch64MIR_TargetOp<"fcvtpu", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FCVTPUDHr FCVTPUDSr FCVTPUSDr FCVTPUSHr FCVTPUUWDr FCVTPUUWHr FCVTPUUWSr FCVTPUUXDr FCVTPUUXHr FCVTPUUXSr FCVTPUv1f16 FCVTPUv1i32 FCVTPUv1i64 FCVTPUv2f32 FCVTPUv2f64 FCVTPUv4f16 FCVTPUv4f32 FCVTPUv8f16
}

def AArch64MIR_fcvtxnt : AArch64MIR_TargetOp<"fcvtxnt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FCVTXNT_ZPmZ_DtoS FCVTXNT_ZPzZ_StoD
}

def AArch64MIR_fcvtxn : AArch64MIR_TargetOp<"fcvtxn", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FCVTXNv1i64 FCVTXNv2f32
}

def AArch64MIR_fcvtxn2 : AArch64MIR_TargetOp<"fcvtxn2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FCVTXNv4f32
}

def AArch64MIR_fcvtx : AArch64MIR_TargetOp<"fcvtx", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FCVTX_ZPmZ_DtoS FCVTX_ZPzZ_DtoS
}

def AArch64MIR_fcvtzs : AArch64MIR_TargetOp<"fcvtzs", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FCVTZSDHr FCVTZSDSr FCVTZSSDr FCVTZSSHr FCVTZSSWDri FCVTZSSWHri FCVTZSSWSri FCVTZSSXDri FCVTZSSXHri FCVTZSSXSri FCVTZSUWDr FCVTZSUWHr FCVTZSUWSr FCVTZSUXDr FCVTZSUXHr FCVTZSUXSr FCVTZS_2Z2Z_StoS FCVTZS_4Z4Z_StoS FCVTZS_ZPmZ_DtoD FCVTZS_ZPmZ_DtoS FCVTZS_ZPmZ_HtoD FCVTZS_ZPmZ_HtoH FCVTZS_ZPmZ_HtoS FCVTZS_ZPmZ_StoD FCVTZS_ZPmZ_StoS FCVTZS_ZPzZ_DtoD FCVTZS_ZPzZ_DtoS FCVTZS_ZPzZ_HtoD FCVTZS_ZPzZ_HtoH FCVTZS_ZPzZ_HtoS FCVTZS_ZPzZ_StoD FCVTZS_ZPzZ_StoS FCVTZSd FCVTZSh FCVTZSs FCVTZSv1f16 FCVTZSv1i32 FCVTZSv1i64 FCVTZSv2f32 FCVTZSv2f64 FCVTZSv2i32_shift FCVTZSv2i64_shift FCVTZSv4f16 FCVTZSv4f32 FCVTZSv4i16_shift FCVTZSv4i32_shift FCVTZSv8f16 FCVTZSv8i16_shift
}

def AArch64MIR_fcvtzsn : AArch64MIR_TargetOp<"fcvtzsn", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FCVTZSN_Z2Z_DtoS FCVTZSN_Z2Z_HtoB FCVTZSN_Z2Z_StoH
}

def AArch64MIR_fcvtzu : AArch64MIR_TargetOp<"fcvtzu", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FCVTZUDHr FCVTZUDSr FCVTZUSDr FCVTZUSHr FCVTZUSWDri FCVTZUSWHri FCVTZUSWSri FCVTZUSXDri FCVTZUSXHri FCVTZUSXSri FCVTZUUWDr FCVTZUUWHr FCVTZUUWSr FCVTZUUXDr FCVTZUUXHr FCVTZUUXSr FCVTZU_2Z2Z_StoS FCVTZU_4Z4Z_StoS FCVTZU_ZPmZ_DtoD FCVTZU_ZPmZ_DtoS FCVTZU_ZPmZ_HtoD FCVTZU_ZPmZ_HtoH FCVTZU_ZPmZ_HtoS FCVTZU_ZPmZ_StoD FCVTZU_ZPmZ_StoS FCVTZU_ZPzZ_DtoD FCVTZU_ZPzZ_DtoS FCVTZU_ZPzZ_HtoD FCVTZU_ZPzZ_HtoH FCVTZU_ZPzZ_HtoS FCVTZU_ZPzZ_StoD FCVTZU_ZPzZ_StoS FCVTZUd FCVTZUh FCVTZUs FCVTZUv1f16 FCVTZUv1i32 FCVTZUv1i64 FCVTZUv2f32 FCVTZUv2f64 FCVTZUv2i32_shift FCVTZUv2i64_shift FCVTZUv4f16 FCVTZUv4f32 FCVTZUv4i16_shift FCVTZUv4i32_shift FCVTZUv8f16 FCVTZUv8i16_shift
}

def AArch64MIR_fcvtzun : AArch64MIR_TargetOp<"fcvtzun", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FCVTZUN_Z2Z_DtoS FCVTZUN_Z2Z_HtoB FCVTZUN_Z2Z_StoH
}

def AArch64MIR_fdiv : AArch64MIR_TargetOp<"fdiv", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FDIVDrr FDIVHrr FDIVSrr FDIV_ZPmZ_D FDIV_ZPmZ_H FDIV_ZPmZ_S FDIVv2f32 FDIVv2f64 FDIVv4f16 FDIVv4f32 FDIVv8f16
}

def AArch64MIR_fdivr : AArch64MIR_TargetOp<"fdivr", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FDIVR_ZPmZ_D FDIVR_ZPmZ_H FDIVR_ZPmZ_S
}

def AArch64MIR_fdot : AArch64MIR_TargetOp<"fdot", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FDOT_VG2_M2Z2Z_BtoH FDOT_VG2_M2Z2Z_BtoS FDOT_VG2_M2Z2Z_HtoS FDOT_VG2_M2ZZI_BtoH FDOT_VG2_M2ZZI_BtoS FDOT_VG2_M2ZZI_HtoS FDOT_VG2_M2ZZ_BtoH FDOT_VG2_M2ZZ_BtoS FDOT_VG2_M2ZZ_HtoS FDOT_VG4_M4Z4Z_BtoH FDOT_VG4_M4Z4Z_BtoS FDOT_VG4_M4Z4Z_HtoS FDOT_VG4_M4ZZI_BtoH FDOT_VG4_M4ZZI_BtoS FDOT_VG4_M4ZZI_HtoS FDOT_VG4_M4ZZ_BtoH FDOT_VG4_M4ZZ_BtoS FDOT_VG4_M4ZZ_HtoS FDOT_ZZZI_BtoH FDOT_ZZZI_BtoS FDOT_ZZZI_S FDOT_ZZZ_BtoH FDOT_ZZZ_BtoS FDOT_ZZZ_S FDOTlanev2f32 FDOTlanev4f16 FDOTlanev4f16_v2f32 FDOTlanev4f32 FDOTlanev8f16 FDOTlanev8f16_v4f32 FDOTv2f32 FDOTv4f16 FDOTv4f16_v2f32 FDOTv4f32 FDOTv8f16 FDOTv8f16_v4f32
}

def AArch64MIR_fdup : AArch64MIR_TargetOp<"fdup", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FDUP_ZI_D FDUP_ZI_H FDUP_ZI_S
}

def AArch64MIR_fexpa : AArch64MIR_TargetOp<"fexpa", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FEXPA_ZZ_D FEXPA_ZZ_H FEXPA_ZZ_S
}

def AArch64MIR_firstp : AArch64MIR_TargetOp<"firstp", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FIRSTP_XPP_B FIRSTP_XPP_D FIRSTP_XPP_H FIRSTP_XPP_S
}

def AArch64MIR_fjcvtzs : AArch64MIR_TargetOp<"fjcvtzs", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FJCVTZS
}

def AArch64MIR_flogb : AArch64MIR_TargetOp<"flogb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FLOGB_ZPmZ_D FLOGB_ZPmZ_H FLOGB_ZPmZ_S FLOGB_ZPzZ_D FLOGB_ZPzZ_H FLOGB_ZPzZ_S
}

def AArch64MIR_fmadd : AArch64MIR_TargetOp<"fmadd", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMADDDrrr FMADDHrrr FMADDSrrr
}

def AArch64MIR_fmad : AArch64MIR_TargetOp<"fmad", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMAD_ZPmZZ_D FMAD_ZPmZZ_H FMAD_ZPmZZ_S
}

def AArch64MIR_fmax : AArch64MIR_TargetOp<"fmax", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMAXDrr FMAXHrr FMAXSrr FMAX_VG2_2Z2Z_D FMAX_VG2_2Z2Z_H FMAX_VG2_2Z2Z_S FMAX_VG2_2ZZ_D FMAX_VG2_2ZZ_H FMAX_VG2_2ZZ_S FMAX_VG4_4Z4Z_D FMAX_VG4_4Z4Z_H FMAX_VG4_4Z4Z_S FMAX_VG4_4ZZ_D FMAX_VG4_4ZZ_H FMAX_VG4_4ZZ_S FMAX_ZPmI_D FMAX_ZPmI_H FMAX_ZPmI_S FMAX_ZPmZ_D FMAX_ZPmZ_H FMAX_ZPmZ_S FMAXv2f32 FMAXv2f64 FMAXv4f16 FMAXv4f32 FMAXv8f16
}

def AArch64MIR_fmaxnm : AArch64MIR_TargetOp<"fmaxnm", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMAXNMDrr FMAXNMHrr FMAXNMSrr FMAXNM_VG2_2Z2Z_D FMAXNM_VG2_2Z2Z_H FMAXNM_VG2_2Z2Z_S FMAXNM_VG2_2ZZ_D FMAXNM_VG2_2ZZ_H FMAXNM_VG2_2ZZ_S FMAXNM_VG4_4Z4Z_D FMAXNM_VG4_4Z4Z_H FMAXNM_VG4_4Z4Z_S FMAXNM_VG4_4ZZ_D FMAXNM_VG4_4ZZ_H FMAXNM_VG4_4ZZ_S FMAXNM_ZPmI_D FMAXNM_ZPmI_H FMAXNM_ZPmI_S FMAXNM_ZPmZ_D FMAXNM_ZPmZ_H FMAXNM_ZPmZ_S FMAXNMv2f32 FMAXNMv2f64 FMAXNMv4f16 FMAXNMv4f32 FMAXNMv8f16
}

def AArch64MIR_fmaxnmp : AArch64MIR_TargetOp<"fmaxnmp", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMAXNMP_ZPmZZ_D FMAXNMP_ZPmZZ_H FMAXNMP_ZPmZZ_S FMAXNMPv2f32 FMAXNMPv2f64 FMAXNMPv2i16p FMAXNMPv2i32p FMAXNMPv2i64p FMAXNMPv4f16 FMAXNMPv4f32 FMAXNMPv8f16
}

def AArch64MIR_fmaxnmqv : AArch64MIR_TargetOp<"fmaxnmqv", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMAXNMQV_D FMAXNMQV_H FMAXNMQV_S
}

def AArch64MIR_fmaxnmv : AArch64MIR_TargetOp<"fmaxnmv", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMAXNMV_VPZ_D FMAXNMV_VPZ_H FMAXNMV_VPZ_S FMAXNMVv4i16v FMAXNMVv4i32v FMAXNMVv8i16v
}

def AArch64MIR_fmaxp : AArch64MIR_TargetOp<"fmaxp", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMAXP_ZPmZZ_D FMAXP_ZPmZZ_H FMAXP_ZPmZZ_S FMAXPv2f32 FMAXPv2f64 FMAXPv2i16p FMAXPv2i32p FMAXPv2i64p FMAXPv4f16 FMAXPv4f32 FMAXPv8f16
}

def AArch64MIR_fmaxqv : AArch64MIR_TargetOp<"fmaxqv", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMAXQV_D FMAXQV_H FMAXQV_S
}

def AArch64MIR_fmaxv : AArch64MIR_TargetOp<"fmaxv", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMAXV_VPZ_D FMAXV_VPZ_H FMAXV_VPZ_S FMAXVv4i16v FMAXVv4i32v FMAXVv8i16v
}

def AArch64MIR_fmin : AArch64MIR_TargetOp<"fmin", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMINDrr FMINHrr FMINSrr FMIN_VG2_2Z2Z_D FMIN_VG2_2Z2Z_H FMIN_VG2_2Z2Z_S FMIN_VG2_2ZZ_D FMIN_VG2_2ZZ_H FMIN_VG2_2ZZ_S FMIN_VG4_4Z4Z_D FMIN_VG4_4Z4Z_H FMIN_VG4_4Z4Z_S FMIN_VG4_4ZZ_D FMIN_VG4_4ZZ_H FMIN_VG4_4ZZ_S FMIN_ZPmI_D FMIN_ZPmI_H FMIN_ZPmI_S FMIN_ZPmZ_D FMIN_ZPmZ_H FMIN_ZPmZ_S FMINv2f32 FMINv2f64 FMINv4f16 FMINv4f32 FMINv8f16
}

def AArch64MIR_fminnm : AArch64MIR_TargetOp<"fminnm", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMINNMDrr FMINNMHrr FMINNMSrr FMINNM_VG2_2Z2Z_D FMINNM_VG2_2Z2Z_H FMINNM_VG2_2Z2Z_S FMINNM_VG2_2ZZ_D FMINNM_VG2_2ZZ_H FMINNM_VG2_2ZZ_S FMINNM_VG4_4Z4Z_D FMINNM_VG4_4Z4Z_H FMINNM_VG4_4Z4Z_S FMINNM_VG4_4ZZ_D FMINNM_VG4_4ZZ_H FMINNM_VG4_4ZZ_S FMINNM_ZPmI_D FMINNM_ZPmI_H FMINNM_ZPmI_S FMINNM_ZPmZ_D FMINNM_ZPmZ_H FMINNM_ZPmZ_S FMINNMv2f32 FMINNMv2f64 FMINNMv4f16 FMINNMv4f32 FMINNMv8f16
}

def AArch64MIR_fminnmp : AArch64MIR_TargetOp<"fminnmp", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMINNMP_ZPmZZ_D FMINNMP_ZPmZZ_H FMINNMP_ZPmZZ_S FMINNMPv2f32 FMINNMPv2f64 FMINNMPv2i16p FMINNMPv2i32p FMINNMPv2i64p FMINNMPv4f16 FMINNMPv4f32 FMINNMPv8f16
}

def AArch64MIR_fminnmqv : AArch64MIR_TargetOp<"fminnmqv", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMINNMQV_D FMINNMQV_H FMINNMQV_S
}

def AArch64MIR_fminnmv : AArch64MIR_TargetOp<"fminnmv", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMINNMV_VPZ_D FMINNMV_VPZ_H FMINNMV_VPZ_S FMINNMVv4i16v FMINNMVv4i32v FMINNMVv8i16v
}

def AArch64MIR_fminp : AArch64MIR_TargetOp<"fminp", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMINP_ZPmZZ_D FMINP_ZPmZZ_H FMINP_ZPmZZ_S FMINPv2f32 FMINPv2f64 FMINPv2i16p FMINPv2i32p FMINPv2i64p FMINPv4f16 FMINPv4f32 FMINPv8f16
}

def AArch64MIR_fminqv : AArch64MIR_TargetOp<"fminqv", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMINQV_D FMINQV_H FMINQV_S
}

def AArch64MIR_fminv : AArch64MIR_TargetOp<"fminv", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMINV_VPZ_D FMINV_VPZ_H FMINV_VPZ_S FMINVv4i16v FMINVv4i32v FMINVv8i16v
}

def AArch64MIR_fmlal2 : AArch64MIR_TargetOp<"fmlal2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMLAL2lanev4f16 FMLAL2lanev8f16 FMLAL2v4f16 FMLAL2v8f16
}

def AArch64MIR_fmlalb : AArch64MIR_TargetOp<"fmlalb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMLALB_ZZZ FMLALB_ZZZI FMLALB_ZZZI_SHH FMLALB_ZZZ_SHH FMLALBlanev8f16 FMLALBv16i8_v8f16
}

def AArch64MIR_fmlallbb : AArch64MIR_TargetOp<"fmlallbb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMLALLBB_ZZZ FMLALLBB_ZZZI FMLALLBBlanev4f32 FMLALLBBv4f32
}

def AArch64MIR_fmlallbt : AArch64MIR_TargetOp<"fmlallbt", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMLALLBT_ZZZ FMLALLBT_ZZZI FMLALLBTlanev4f32 FMLALLBTv4f32
}

def AArch64MIR_fmlalltb : AArch64MIR_TargetOp<"fmlalltb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMLALLTB_ZZZ FMLALLTB_ZZZI FMLALLTBlanev4f32 FMLALLTBv4f32
}

def AArch64MIR_fmlalltt : AArch64MIR_TargetOp<"fmlalltt", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMLALLTT_ZZZ FMLALLTT_ZZZI FMLALLTTlanev4f32 FMLALLTTv4f32
}

def AArch64MIR_fmlall : AArch64MIR_TargetOp<"fmlall", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMLALL_MZZI_BtoS FMLALL_MZZ_BtoS FMLALL_VG2_M2Z2Z_BtoS FMLALL_VG2_M2ZZI_BtoS FMLALL_VG2_M2ZZ_BtoS FMLALL_VG4_M4Z4Z_BtoS FMLALL_VG4_M4ZZI_BtoS FMLALL_VG4_M4ZZ_BtoS
}

def AArch64MIR_fmlalt : AArch64MIR_TargetOp<"fmlalt", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMLALT_ZZZ FMLALT_ZZZI FMLALT_ZZZI_SHH FMLALT_ZZZ_SHH FMLALTlanev8f16 FMLALTv16i8_v8f16
}

def AArch64MIR_fmlal : AArch64MIR_TargetOp<"fmlal", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMLAL_MZZI_BtoH FMLAL_MZZI_HtoS FMLAL_MZZ_HtoS FMLAL_VG2_M2Z2Z_BtoH FMLAL_VG2_M2Z2Z_HtoS FMLAL_VG2_M2ZZI_BtoH FMLAL_VG2_M2ZZI_HtoS FMLAL_VG2_M2ZZ_BtoH FMLAL_VG2_M2ZZ_HtoS FMLAL_VG2_MZZ_BtoH FMLAL_VG4_M4Z4Z_BtoH FMLAL_VG4_M4Z4Z_HtoS FMLAL_VG4_M4ZZI_BtoH FMLAL_VG4_M4ZZI_HtoS FMLAL_VG4_M4ZZ_BtoH FMLAL_VG4_M4ZZ_HtoS FMLALlanev4f16 FMLALlanev8f16 FMLALv4f16 FMLALv8f16
}

def AArch64MIR_fmla : AArch64MIR_TargetOp<"fmla", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMLA_VG2_M2Z2Z_D FMLA_VG2_M2Z2Z_H FMLA_VG2_M2Z2Z_S FMLA_VG2_M2ZZI_D FMLA_VG2_M2ZZI_H FMLA_VG2_M2ZZI_S FMLA_VG2_M2ZZ_D FMLA_VG2_M2ZZ_H FMLA_VG2_M2ZZ_S FMLA_VG4_M4Z4Z_D FMLA_VG4_M4Z4Z_H FMLA_VG4_M4Z4Z_S FMLA_VG4_M4ZZI_D FMLA_VG4_M4ZZI_H FMLA_VG4_M4ZZI_S FMLA_VG4_M4ZZ_D FMLA_VG4_M4ZZ_H FMLA_VG4_M4ZZ_S FMLA_ZPmZZ_D FMLA_ZPmZZ_H FMLA_ZPmZZ_S FMLA_ZZZI_D FMLA_ZZZI_H FMLA_ZZZI_S FMLAv1i16_indexed FMLAv1i32_indexed FMLAv1i64_indexed FMLAv2f32 FMLAv2f64 FMLAv2i32_indexed FMLAv2i64_indexed FMLAv4f16 FMLAv4f32 FMLAv4i16_indexed FMLAv4i32_indexed FMLAv8f16 FMLAv8i16_indexed
}

def AArch64MIR_fmmla : AArch64MIR_TargetOp<"fmmla", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMLLA_ZZZ_HtoS FMMLA_ZZZ_BtoH FMMLA_ZZZ_BtoS FMMLA_ZZZ_D FMMLA_ZZZ_H FMMLA_ZZZ_S FMMLAv4f32 FMMLAv8f16 FMMLAv8f16_v4f32 FMMLAv8f16_v8f16
}

def AArch64MIR_fmlsl2 : AArch64MIR_TargetOp<"fmlsl2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMLSL2lanev4f16 FMLSL2lanev8f16 FMLSL2v4f16 FMLSL2v8f16
}

def AArch64MIR_fmlslb : AArch64MIR_TargetOp<"fmlslb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMLSLB_ZZZI_SHH FMLSLB_ZZZ_SHH
}

def AArch64MIR_fmlslt : AArch64MIR_TargetOp<"fmlslt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMLSLT_ZZZI_SHH FMLSLT_ZZZ_SHH
}

def AArch64MIR_fmlsl : AArch64MIR_TargetOp<"fmlsl", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMLSL_MZZI_HtoS FMLSL_MZZ_HtoS FMLSL_VG2_M2Z2Z_HtoS FMLSL_VG2_M2ZZI_HtoS FMLSL_VG2_M2ZZ_HtoS FMLSL_VG4_M4Z4Z_HtoS FMLSL_VG4_M4ZZI_HtoS FMLSL_VG4_M4ZZ_HtoS FMLSLlanev4f16 FMLSLlanev8f16 FMLSLv4f16 FMLSLv8f16
}

def AArch64MIR_fmls : AArch64MIR_TargetOp<"fmls", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMLS_VG2_M2Z2Z_D FMLS_VG2_M2Z2Z_H FMLS_VG2_M2Z2Z_S FMLS_VG2_M2ZZI_D FMLS_VG2_M2ZZI_H FMLS_VG2_M2ZZI_S FMLS_VG2_M2ZZ_D FMLS_VG2_M2ZZ_H FMLS_VG2_M2ZZ_S FMLS_VG4_M4Z4Z_D FMLS_VG4_M4Z4Z_H FMLS_VG4_M4Z4Z_S FMLS_VG4_M4ZZI_D FMLS_VG4_M4ZZI_H FMLS_VG4_M4ZZI_S FMLS_VG4_M4ZZ_D FMLS_VG4_M4ZZ_H FMLS_VG4_M4ZZ_S FMLS_ZPmZZ_D FMLS_ZPmZZ_H FMLS_ZPmZZ_S FMLS_ZZZI_D FMLS_ZZZI_H FMLS_ZZZI_S FMLSv1i16_indexed FMLSv1i32_indexed FMLSv1i64_indexed FMLSv2f32 FMLSv2f64 FMLSv2i32_indexed FMLSv2i64_indexed FMLSv4f16 FMLSv4f32 FMLSv4i16_indexed FMLSv4i32_indexed FMLSv8f16 FMLSv8i16_indexed
}

def AArch64MIR_fmop4a : AArch64MIR_TargetOp<"fmop4a", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMOP4A_M2Z2Z_BtoH FMOP4A_M2Z2Z_BtoS FMOP4A_M2Z2Z_D FMOP4A_M2Z2Z_H FMOP4A_M2Z2Z_HtoS FMOP4A_M2Z2Z_S FMOP4A_M2ZZ_BtoH FMOP4A_M2ZZ_BtoS FMOP4A_M2ZZ_D FMOP4A_M2ZZ_H FMOP4A_M2ZZ_HtoS FMOP4A_M2ZZ_S FMOP4A_MZ2Z_BtoH FMOP4A_MZ2Z_BtoS FMOP4A_MZ2Z_D FMOP4A_MZ2Z_H FMOP4A_MZ2Z_HtoS FMOP4A_MZ2Z_S FMOP4A_MZZ_BtoH FMOP4A_MZZ_BtoS FMOP4A_MZZ_D FMOP4A_MZZ_H FMOP4A_MZZ_HtoS FMOP4A_MZZ_S
}

def AArch64MIR_fmop4s : AArch64MIR_TargetOp<"fmop4s", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMOP4S_M2Z2Z_D FMOP4S_M2Z2Z_H FMOP4S_M2Z2Z_HtoS FMOP4S_M2Z2Z_S FMOP4S_M2ZZ_D FMOP4S_M2ZZ_H FMOP4S_M2ZZ_HtoS FMOP4S_M2ZZ_S FMOP4S_MZ2Z_D FMOP4S_MZ2Z_H FMOP4S_MZ2Z_HtoS FMOP4S_MZ2Z_S FMOP4S_MZZ_D FMOP4S_MZZ_H FMOP4S_MZZ_HtoS FMOP4S_MZZ_S
}

def AArch64MIR_fmopa : AArch64MIR_TargetOp<"fmopa", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMOPAL_MPPZZ FMOPA_MPPZZ_BtoH FMOPA_MPPZZ_BtoS FMOPA_MPPZZ_D FMOPA_MPPZZ_H FMOPA_MPPZZ_S
}

def AArch64MIR_fmops : AArch64MIR_TargetOp<"fmops", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMOPSL_MPPZZ FMOPS_MPPZZ_D FMOPS_MPPZZ_H FMOPS_MPPZZ_S
}

def AArch64MIR_fmov : AArch64MIR_TargetOp<"fmov", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMOVDXHighr FMOVDXr FMOVDi FMOVDr FMOVHWr FMOVHXr FMOVHi FMOVHr FMOVSWr FMOVSi FMOVSr FMOVWHr FMOVWSr FMOVXDHighr FMOVXDr FMOVXHr FMOVv2f32_ns FMOVv2f64_ns FMOVv4f16_ns FMOVv4f32_ns FMOVv8f16_ns
}

def AArch64MIR_fmsb : AArch64MIR_TargetOp<"fmsb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMSB_ZPmZZ_D FMSB_ZPmZZ_H FMSB_ZPmZZ_S
}

def AArch64MIR_fmsub : AArch64MIR_TargetOp<"fmsub", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMSUBDrrr FMSUBHrrr FMSUBSrrr
}

def AArch64MIR_fmul : AArch64MIR_TargetOp<"fmul", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMULDrr FMULHrr FMULSrr FMUL_2Z2Z_D FMUL_2Z2Z_H FMUL_2Z2Z_S FMUL_2ZZ_D FMUL_2ZZ_H FMUL_2ZZ_S FMUL_4Z4Z_D FMUL_4Z4Z_H FMUL_4Z4Z_S FMUL_4ZZ_D FMUL_4ZZ_H FMUL_4ZZ_S FMUL_ZPmI_D FMUL_ZPmI_H FMUL_ZPmI_S FMUL_ZPmZ_D FMUL_ZPmZ_H FMUL_ZPmZ_S FMUL_ZZZI_D FMUL_ZZZI_H FMUL_ZZZI_S FMUL_ZZZ_D FMUL_ZZZ_H FMUL_ZZZ_S FMULv1i16_indexed FMULv1i32_indexed FMULv1i64_indexed FMULv2f32 FMULv2f64 FMULv2i32_indexed FMULv2i64_indexed FMULv4f16 FMULv4f32 FMULv4i16_indexed FMULv4i32_indexed FMULv8f16 FMULv8i16_indexed
}

def AArch64MIR_fmulx : AArch64MIR_TargetOp<"fmulx", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FMULX16 FMULX32 FMULX64 FMULX_ZPmZ_D FMULX_ZPmZ_H FMULX_ZPmZ_S FMULXv1i16_indexed FMULXv1i32_indexed FMULXv1i64_indexed FMULXv2f32 FMULXv2f64 FMULXv2i32_indexed FMULXv2i64_indexed FMULXv4f16 FMULXv4f32 FMULXv4i16_indexed FMULXv4i32_indexed FMULXv8f16 FMULXv8i16_indexed
}

def AArch64MIR_fneg : AArch64MIR_TargetOp<"fneg", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FNEGDr FNEGHr FNEGSr FNEG_ZPmZ_D FNEG_ZPmZ_H FNEG_ZPmZ_S FNEG_ZPzZ_D FNEG_ZPzZ_H FNEG_ZPzZ_S FNEGv2f32 FNEGv2f64 FNEGv4f16 FNEGv4f32 FNEGv8f16
}

def AArch64MIR_fnmadd : AArch64MIR_TargetOp<"fnmadd", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FNMADDDrrr FNMADDHrrr FNMADDSrrr
}

def AArch64MIR_fnmad : AArch64MIR_TargetOp<"fnmad", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FNMAD_ZPmZZ_D FNMAD_ZPmZZ_H FNMAD_ZPmZZ_S
}

def AArch64MIR_fnmla : AArch64MIR_TargetOp<"fnmla", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FNMLA_ZPmZZ_D FNMLA_ZPmZZ_H FNMLA_ZPmZZ_S
}

def AArch64MIR_fnmls : AArch64MIR_TargetOp<"fnmls", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FNMLS_ZPmZZ_D FNMLS_ZPmZZ_H FNMLS_ZPmZZ_S
}

def AArch64MIR_fnmsb : AArch64MIR_TargetOp<"fnmsb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FNMSB_ZPmZZ_D FNMSB_ZPmZZ_H FNMSB_ZPmZZ_S
}

def AArch64MIR_fnmsub : AArch64MIR_TargetOp<"fnmsub", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FNMSUBDrrr FNMSUBHrrr FNMSUBSrrr
}

def AArch64MIR_fnmul : AArch64MIR_TargetOp<"fnmul", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FNMULDrr FNMULHrr FNMULSrr
}

def AArch64MIR_frecpe : AArch64MIR_TargetOp<"frecpe", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FRECPE_ZZ_D FRECPE_ZZ_H FRECPE_ZZ_S FRECPEv1f16 FRECPEv1i32 FRECPEv1i64 FRECPEv2f32 FRECPEv2f64 FRECPEv4f16 FRECPEv4f32 FRECPEv8f16
}

def AArch64MIR_frecps : AArch64MIR_TargetOp<"frecps", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FRECPS16 FRECPS32 FRECPS64 FRECPS_ZZZ_D FRECPS_ZZZ_H FRECPS_ZZZ_S FRECPSv2f32 FRECPSv2f64 FRECPSv4f16 FRECPSv4f32 FRECPSv8f16
}

def AArch64MIR_frecpx : AArch64MIR_TargetOp<"frecpx", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FRECPX_ZPmZ_D FRECPX_ZPmZ_H FRECPX_ZPmZ_S FRECPX_ZPzZ_D FRECPX_ZPzZ_H FRECPX_ZPzZ_S FRECPXv1f16 FRECPXv1i32 FRECPXv1i64
}

def AArch64MIR_frint32x : AArch64MIR_TargetOp<"frint32x", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FRINT32XDr FRINT32XSr FRINT32X_ZPmZ_D FRINT32X_ZPmZ_S FRINT32X_ZPzZ_D FRINT32X_ZPzZ_S FRINT32Xv2f32 FRINT32Xv2f64 FRINT32Xv4f32
}

def AArch64MIR_frint32z : AArch64MIR_TargetOp<"frint32z", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FRINT32ZDr FRINT32ZSr FRINT32Z_ZPmZ_D FRINT32Z_ZPmZ_S FRINT32Z_ZPzZ_D FRINT32Z_ZPzZ_S FRINT32Zv2f32 FRINT32Zv2f64 FRINT32Zv4f32
}

def AArch64MIR_frint64x : AArch64MIR_TargetOp<"frint64x", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FRINT64XDr FRINT64XSr FRINT64X_ZPmZ_D FRINT64X_ZPmZ_S FRINT64X_ZPzZ_D FRINT64X_ZPzZ_S FRINT64Xv2f32 FRINT64Xv2f64 FRINT64Xv4f32
}

def AArch64MIR_frint64z : AArch64MIR_TargetOp<"frint64z", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FRINT64ZDr FRINT64ZSr FRINT64Z_ZPmZ_D FRINT64Z_ZPmZ_S FRINT64Z_ZPzZ_D FRINT64Z_ZPzZ_S FRINT64Zv2f32 FRINT64Zv2f64 FRINT64Zv4f32
}

def AArch64MIR_frinta : AArch64MIR_TargetOp<"frinta", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FRINTADr FRINTAHr FRINTASr FRINTA_2Z2Z_S FRINTA_4Z4Z_S FRINTA_ZPmZ_D FRINTA_ZPmZ_H FRINTA_ZPmZ_S FRINTA_ZPzZ_D FRINTA_ZPzZ_H FRINTA_ZPzZ_S FRINTAv2f32 FRINTAv2f64 FRINTAv4f16 FRINTAv4f32 FRINTAv8f16
}

def AArch64MIR_frinti : AArch64MIR_TargetOp<"frinti", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FRINTIDr FRINTIHr FRINTISr FRINTI_ZPmZ_D FRINTI_ZPmZ_H FRINTI_ZPmZ_S FRINTI_ZPzZ_D FRINTI_ZPzZ_H FRINTI_ZPzZ_S FRINTIv2f32 FRINTIv2f64 FRINTIv4f16 FRINTIv4f32 FRINTIv8f16
}

def AArch64MIR_frintm : AArch64MIR_TargetOp<"frintm", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FRINTMDr FRINTMHr FRINTMSr FRINTM_2Z2Z_S FRINTM_4Z4Z_S FRINTM_ZPmZ_D FRINTM_ZPmZ_H FRINTM_ZPmZ_S FRINTM_ZPzZ_D FRINTM_ZPzZ_H FRINTM_ZPzZ_S FRINTMv2f32 FRINTMv2f64 FRINTMv4f16 FRINTMv4f32 FRINTMv8f16
}

def AArch64MIR_frintn : AArch64MIR_TargetOp<"frintn", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FRINTNDr FRINTNHr FRINTNSr FRINTN_2Z2Z_S FRINTN_4Z4Z_S FRINTN_ZPmZ_D FRINTN_ZPmZ_H FRINTN_ZPmZ_S FRINTN_ZPzZ_D FRINTN_ZPzZ_H FRINTN_ZPzZ_S FRINTNv2f32 FRINTNv2f64 FRINTNv4f16 FRINTNv4f32 FRINTNv8f16
}

def AArch64MIR_frintp : AArch64MIR_TargetOp<"frintp", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FRINTPDr FRINTPHr FRINTPSr FRINTP_2Z2Z_S FRINTP_4Z4Z_S FRINTP_ZPmZ_D FRINTP_ZPmZ_H FRINTP_ZPmZ_S FRINTP_ZPzZ_D FRINTP_ZPzZ_H FRINTP_ZPzZ_S FRINTPv2f32 FRINTPv2f64 FRINTPv4f16 FRINTPv4f32 FRINTPv8f16
}

def AArch64MIR_frintx : AArch64MIR_TargetOp<"frintx", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FRINTXDr FRINTXHr FRINTXSr FRINTX_ZPmZ_D FRINTX_ZPmZ_H FRINTX_ZPmZ_S FRINTX_ZPzZ_D FRINTX_ZPzZ_H FRINTX_ZPzZ_S FRINTXv2f32 FRINTXv2f64 FRINTXv4f16 FRINTXv4f32 FRINTXv8f16
}

def AArch64MIR_frintz : AArch64MIR_TargetOp<"frintz", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FRINTZDr FRINTZHr FRINTZSr FRINTZ_ZPmZ_D FRINTZ_ZPmZ_H FRINTZ_ZPmZ_S FRINTZ_ZPzZ_D FRINTZ_ZPzZ_H FRINTZ_ZPzZ_S FRINTZv2f32 FRINTZv2f64 FRINTZv4f16 FRINTZv4f32 FRINTZv8f16
}

def AArch64MIR_frsqrte : AArch64MIR_TargetOp<"frsqrte", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FRSQRTE_ZZ_D FRSQRTE_ZZ_H FRSQRTE_ZZ_S FRSQRTEv1f16 FRSQRTEv1i32 FRSQRTEv1i64 FRSQRTEv2f32 FRSQRTEv2f64 FRSQRTEv4f16 FRSQRTEv4f32 FRSQRTEv8f16
}

def AArch64MIR_frsqrts : AArch64MIR_TargetOp<"frsqrts", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FRSQRTS16 FRSQRTS32 FRSQRTS64 FRSQRTS_ZZZ_D FRSQRTS_ZZZ_H FRSQRTS_ZZZ_S FRSQRTSv2f32 FRSQRTSv2f64 FRSQRTSv4f16 FRSQRTSv4f32 FRSQRTSv8f16
}

def AArch64MIR_fscale : AArch64MIR_TargetOp<"fscale", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FSCALE_2Z2Z_D FSCALE_2Z2Z_H FSCALE_2Z2Z_S FSCALE_2ZZ_D FSCALE_2ZZ_H FSCALE_2ZZ_S FSCALE_4Z4Z_D FSCALE_4Z4Z_H FSCALE_4Z4Z_S FSCALE_4ZZ_D FSCALE_4ZZ_H FSCALE_4ZZ_S FSCALE_ZPmZ_D FSCALE_ZPmZ_H FSCALE_ZPmZ_S FSCALEv2f32 FSCALEv2f64 FSCALEv4f16 FSCALEv4f32 FSCALEv8f16
}

def AArch64MIR_fsqrt : AArch64MIR_TargetOp<"fsqrt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FSQRTDr FSQRTHr FSQRTSr FSQRT_ZPZz_D FSQRT_ZPZz_H FSQRT_ZPZz_S FSQRT_ZPmZ_D FSQRT_ZPmZ_H FSQRT_ZPmZ_S FSQRTv2f32 FSQRTv2f64 FSQRTv4f16 FSQRTv4f32 FSQRTv8f16
}

def AArch64MIR_fsub : AArch64MIR_TargetOp<"fsub", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FSUBDrr FSUBHrr FSUBSrr FSUB_VG2_M2Z_D FSUB_VG2_M2Z_H FSUB_VG2_M2Z_S FSUB_VG4_M4Z_D FSUB_VG4_M4Z_H FSUB_VG4_M4Z_S FSUB_ZPmI_D FSUB_ZPmI_H FSUB_ZPmI_S FSUB_ZPmZ_D FSUB_ZPmZ_H FSUB_ZPmZ_S FSUB_ZZZ_D FSUB_ZZZ_H FSUB_ZZZ_S FSUBv2f32 FSUBv2f64 FSUBv4f16 FSUBv4f32 FSUBv8f16
}

def AArch64MIR_fsubr : AArch64MIR_TargetOp<"fsubr", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FSUBR_ZPmI_D FSUBR_ZPmI_H FSUBR_ZPmI_S FSUBR_ZPmZ_D FSUBR_ZPmZ_H FSUBR_ZPmZ_S
}

def AArch64MIR_ftmad : AArch64MIR_TargetOp<"ftmad", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FTMAD_ZZI_D FTMAD_ZZI_H FTMAD_ZZI_S
}

def AArch64MIR_ftmopa : AArch64MIR_TargetOp<"ftmopa", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FTMOPA_M2ZZZI_BtoH FTMOPA_M2ZZZI_BtoS FTMOPA_M2ZZZI_HtoH FTMOPA_M2ZZZI_HtoS FTMOPA_M2ZZZI_StoS
}

def AArch64MIR_ftsmul : AArch64MIR_TargetOp<"ftsmul", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FTSMUL_ZZZ_D FTSMUL_ZZZ_H FTSMUL_ZZZ_S
}

def AArch64MIR_ftssel : AArch64MIR_TargetOp<"ftssel", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FTSSEL_ZZZ_D FTSSEL_ZZZ_H FTSSEL_ZZZ_S
}

def AArch64MIR_fvdotb : AArch64MIR_TargetOp<"fvdotb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FVDOTB_VG4_M2ZZI_BtoS
}

def AArch64MIR_fvdott : AArch64MIR_TargetOp<"fvdott", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FVDOTT_VG4_M2ZZI_BtoS
}

def AArch64MIR_fvdot : AArch64MIR_TargetOp<"fvdot", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: FVDOT_VG2_M2ZZI_BtoH FVDOT_VG2_M2ZZI_HtoS
}

def AArch64MIR_gcspopcx : AArch64MIR_TargetOp<"gcspopcx", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: GCSPOPCX
}

def AArch64MIR_gcspopm : AArch64MIR_TargetOp<"gcspopm", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: GCSPOPM
}

def AArch64MIR_gcspopx : AArch64MIR_TargetOp<"gcspopx", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: GCSPOPX
}

def AArch64MIR_gcspushm : AArch64MIR_TargetOp<"gcspushm", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: GCSPUSHM
}

def AArch64MIR_gcspushx : AArch64MIR_TargetOp<"gcspushx", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: GCSPUSHX
}

def AArch64MIR_gcsss1 : AArch64MIR_TargetOp<"gcsss1", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: GCSSS1
}

def AArch64MIR_gcsss2 : AArch64MIR_TargetOp<"gcsss2", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: GCSSS2
}

def AArch64MIR_gcsstr : AArch64MIR_TargetOp<"gcsstr", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: GCSSTR
}

def AArch64MIR_gcssttr : AArch64MIR_TargetOp<"gcssttr", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: GCSSTTR
}

def AArch64MIR_ld1b : AArch64MIR_TargetOp<"ld1b", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: GLD1B_D GLD1B_D_IMM GLD1B_D_SXTW GLD1B_D_UXTW GLD1B_S_IMM GLD1B_S_SXTW GLD1B_S_UXTW LD1B LD1B_2Z LD1B_2Z_IMM LD1B_2Z_STRIDED LD1B_2Z_STRIDED_IMM LD1B_4Z LD1B_4Z_IMM LD1B_4Z_STRIDED LD1B_4Z_STRIDED_IMM LD1B_D LD1B_D_IMM LD1B_H LD1B_H_IMM LD1B_IMM LD1B_S LD1B_S_IMM LD1_MXIPXX_H_B LD1_MXIPXX_V_B
}

def AArch64MIR_ld1d : AArch64MIR_TargetOp<"ld1d", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: GLD1D GLD1D_IMM GLD1D_SCALED GLD1D_SXTW GLD1D_SXTW_SCALED GLD1D_UXTW GLD1D_UXTW_SCALED LD1D LD1D_2Z LD1D_2Z_IMM LD1D_2Z_STRIDED LD1D_2Z_STRIDED_IMM LD1D_4Z LD1D_4Z_IMM LD1D_4Z_STRIDED LD1D_4Z_STRIDED_IMM LD1D_IMM LD1D_Q LD1D_Q_IMM LD1_MXIPXX_H_D LD1_MXIPXX_V_D
}

def AArch64MIR_ld1h : AArch64MIR_TargetOp<"ld1h", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: GLD1H_D GLD1H_D_IMM GLD1H_D_SCALED GLD1H_D_SXTW GLD1H_D_SXTW_SCALED GLD1H_D_UXTW GLD1H_D_UXTW_SCALED GLD1H_S_IMM GLD1H_S_SXTW GLD1H_S_SXTW_SCALED GLD1H_S_UXTW GLD1H_S_UXTW_SCALED LD1H LD1H_2Z LD1H_2Z_IMM LD1H_2Z_STRIDED LD1H_2Z_STRIDED_IMM LD1H_4Z LD1H_4Z_IMM LD1H_4Z_STRIDED LD1H_4Z_STRIDED_IMM LD1H_D LD1H_D_IMM LD1H_IMM LD1H_S LD1H_S_IMM LD1_MXIPXX_H_H LD1_MXIPXX_V_H
}

def AArch64MIR_ld1q : AArch64MIR_TargetOp<"ld1q", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: GLD1Q LD1_MXIPXX_H_Q LD1_MXIPXX_V_Q
}

def AArch64MIR_ld1sb : AArch64MIR_TargetOp<"ld1sb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: GLD1SB_D GLD1SB_D_IMM GLD1SB_D_SXTW GLD1SB_D_UXTW GLD1SB_S_IMM GLD1SB_S_SXTW GLD1SB_S_UXTW LD1SB_D LD1SB_D_IMM LD1SB_H LD1SB_H_IMM LD1SB_S LD1SB_S_IMM
}

def AArch64MIR_ld1sh : AArch64MIR_TargetOp<"ld1sh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: GLD1SH_D GLD1SH_D_IMM GLD1SH_D_SCALED GLD1SH_D_SXTW GLD1SH_D_SXTW_SCALED GLD1SH_D_UXTW GLD1SH_D_UXTW_SCALED GLD1SH_S_IMM GLD1SH_S_SXTW GLD1SH_S_SXTW_SCALED GLD1SH_S_UXTW GLD1SH_S_UXTW_SCALED LD1SH_D LD1SH_D_IMM LD1SH_S LD1SH_S_IMM
}

def AArch64MIR_ld1sw : AArch64MIR_TargetOp<"ld1sw", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: GLD1SW_D GLD1SW_D_IMM GLD1SW_D_SCALED GLD1SW_D_SXTW GLD1SW_D_SXTW_SCALED GLD1SW_D_UXTW GLD1SW_D_UXTW_SCALED LD1SW_D LD1SW_D_IMM
}

def AArch64MIR_ld1w : AArch64MIR_TargetOp<"ld1w", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: GLD1W_D GLD1W_D_IMM GLD1W_D_SCALED GLD1W_D_SXTW GLD1W_D_SXTW_SCALED GLD1W_D_UXTW GLD1W_D_UXTW_SCALED GLD1W_IMM GLD1W_SXTW GLD1W_SXTW_SCALED GLD1W_UXTW GLD1W_UXTW_SCALED LD1W LD1W_2Z LD1W_2Z_IMM LD1W_2Z_STRIDED LD1W_2Z_STRIDED_IMM LD1W_4Z LD1W_4Z_IMM LD1W_4Z_STRIDED LD1W_4Z_STRIDED_IMM LD1W_D LD1W_D_IMM LD1W_IMM LD1W_Q LD1W_Q_IMM LD1_MXIPXX_H_S LD1_MXIPXX_V_S
}

def AArch64MIR_ldff1b : AArch64MIR_TargetOp<"ldff1b", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: GLDFF1B_D GLDFF1B_D_IMM GLDFF1B_D_SXTW GLDFF1B_D_UXTW GLDFF1B_S_IMM GLDFF1B_S_SXTW GLDFF1B_S_UXTW LDFF1B LDFF1B_D LDFF1B_H LDFF1B_S
}

def AArch64MIR_ldff1d : AArch64MIR_TargetOp<"ldff1d", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: GLDFF1D GLDFF1D_IMM GLDFF1D_SCALED GLDFF1D_SXTW GLDFF1D_SXTW_SCALED GLDFF1D_UXTW GLDFF1D_UXTW_SCALED LDFF1D
}

def AArch64MIR_ldff1h : AArch64MIR_TargetOp<"ldff1h", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: GLDFF1H_D GLDFF1H_D_IMM GLDFF1H_D_SCALED GLDFF1H_D_SXTW GLDFF1H_D_SXTW_SCALED GLDFF1H_D_UXTW GLDFF1H_D_UXTW_SCALED GLDFF1H_S_IMM GLDFF1H_S_SXTW GLDFF1H_S_SXTW_SCALED GLDFF1H_S_UXTW GLDFF1H_S_UXTW_SCALED LDFF1H LDFF1H_D LDFF1H_S
}

def AArch64MIR_ldff1sb : AArch64MIR_TargetOp<"ldff1sb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: GLDFF1SB_D GLDFF1SB_D_IMM GLDFF1SB_D_SXTW GLDFF1SB_D_UXTW GLDFF1SB_S_IMM GLDFF1SB_S_SXTW GLDFF1SB_S_UXTW LDFF1SB_D LDFF1SB_H LDFF1SB_S
}

def AArch64MIR_ldff1sh : AArch64MIR_TargetOp<"ldff1sh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: GLDFF1SH_D GLDFF1SH_D_IMM GLDFF1SH_D_SCALED GLDFF1SH_D_SXTW GLDFF1SH_D_SXTW_SCALED GLDFF1SH_D_UXTW GLDFF1SH_D_UXTW_SCALED GLDFF1SH_S_IMM GLDFF1SH_S_SXTW GLDFF1SH_S_SXTW_SCALED GLDFF1SH_S_UXTW GLDFF1SH_S_UXTW_SCALED LDFF1SH_D LDFF1SH_S
}

def AArch64MIR_ldff1sw : AArch64MIR_TargetOp<"ldff1sw", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: GLDFF1SW_D GLDFF1SW_D_IMM GLDFF1SW_D_SCALED GLDFF1SW_D_SXTW GLDFF1SW_D_SXTW_SCALED GLDFF1SW_D_UXTW GLDFF1SW_D_UXTW_SCALED LDFF1SW_D
}

def AArch64MIR_ldff1w : AArch64MIR_TargetOp<"ldff1w", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: GLDFF1W_D GLDFF1W_D_IMM GLDFF1W_D_SCALED GLDFF1W_D_SXTW GLDFF1W_D_SXTW_SCALED GLDFF1W_D_UXTW GLDFF1W_D_UXTW_SCALED GLDFF1W_IMM GLDFF1W_SXTW GLDFF1W_SXTW_SCALED GLDFF1W_UXTW GLDFF1W_UXTW_SCALED LDFF1W LDFF1W_D
}

def AArch64MIR_gmi : AArch64MIR_TargetOp<"gmi", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: GMI
}

def AArch64MIR_hinte : AArch64MIR_TargetOp<"hinte", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: HINTE
}

def AArch64MIR_histcnt : AArch64MIR_TargetOp<"histcnt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: HISTCNT_ZPzZZ_D HISTCNT_ZPzZZ_S
}

def AArch64MIR_histseg : AArch64MIR_TargetOp<"histseg", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: HISTSEG_ZZZ
}

def AArch64MIR_hlt : AArch64MIR_TargetOp<"hlt", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: HLT
}

def AArch64MIR_hvc : AArch64MIR_TargetOp<"hvc", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: HVC
}

def AArch64MIR_incb : AArch64MIR_TargetOp<"incb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: INCB_XPiI
}

def AArch64MIR_incd : AArch64MIR_TargetOp<"incd", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: INCD_XPiI INCD_ZPiI
}

def AArch64MIR_inch : AArch64MIR_TargetOp<"inch", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: INCH_XPiI INCH_ZPiI
}

def AArch64MIR_incp : AArch64MIR_TargetOp<"incp", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: INCP_XP_B INCP_XP_D INCP_XP_H INCP_XP_S INCP_ZP_D INCP_ZP_H INCP_ZP_S
}

def AArch64MIR_incw : AArch64MIR_TargetOp<"incw", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: INCW_XPiI INCW_ZPiI
}

def AArch64MIR_index : AArch64MIR_TargetOp<"index", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: INDEX_II_B INDEX_II_D INDEX_II_H INDEX_II_S INDEX_IR_B INDEX_IR_D INDEX_IR_H INDEX_IR_S INDEX_RI_B INDEX_RI_D INDEX_RI_H INDEX_RI_S INDEX_RR_B INDEX_RR_D INDEX_RR_H INDEX_RR_S
}

def AArch64MIR_insr : AArch64MIR_TargetOp<"insr", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: INSR_ZR_B INSR_ZR_D INSR_ZR_H INSR_ZR_S INSR_ZV_B INSR_ZV_D INSR_ZV_H INSR_ZV_S
}

def AArch64MIR_ins : AArch64MIR_TargetOp<"ins", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: INSvi8gpr INSvi8lane INSvi16gpr INSvi16lane INSvi32gpr INSvi32lane INSvi64gpr INSvi64lane
}

def AArch64MIR_irg : AArch64MIR_TargetOp<"irg", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: IRG
}

def AArch64MIR_isb : AArch64MIR_TargetOp<"isb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ISB
}

def AArch64MIR_lasta : AArch64MIR_TargetOp<"lasta", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LASTA_RPZ_B LASTA_RPZ_D LASTA_RPZ_H LASTA_RPZ_S LASTA_VPZ_B LASTA_VPZ_D LASTA_VPZ_H LASTA_VPZ_S
}

def AArch64MIR_lastb : AArch64MIR_TargetOp<"lastb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LASTB_RPZ_B LASTB_RPZ_D LASTB_RPZ_H LASTB_RPZ_S LASTB_VPZ_B LASTB_VPZ_D LASTB_VPZ_H LASTB_VPZ_S
}

def AArch64MIR_lastp : AArch64MIR_TargetOp<"lastp", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LASTP_XPP_B LASTP_XPP_D LASTP_XPP_H LASTP_XPP_S
}

def AArch64MIR_ld1 : AArch64MIR_TargetOp<"ld1", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LD1Fourv1d LD1Fourv1d_POST LD1Fourv2d LD1Fourv2d_POST LD1Fourv2s LD1Fourv2s_POST LD1Fourv4h LD1Fourv4h_POST LD1Fourv4s LD1Fourv4s_POST LD1Fourv8b LD1Fourv8b_POST LD1Fourv8h LD1Fourv8h_POST LD1Fourv16b LD1Fourv16b_POST LD1Onev1d LD1Onev1d_POST LD1Onev2d LD1Onev2d_POST LD1Onev2s LD1Onev2s_POST LD1Onev4h LD1Onev4h_POST LD1Onev4s LD1Onev4s_POST LD1Onev8b LD1Onev8b_POST LD1Onev8h LD1Onev8h_POST LD1Onev16b LD1Onev16b_POST LD1Threev1d LD1Threev1d_POST LD1Threev2d LD1Threev2d_POST LD1Threev2s LD1Threev2s_POST LD1Threev4h LD1Threev4h_POST LD1Threev4s LD1Threev4s_POST LD1Threev8b LD1Threev8b_POST LD1Threev8h LD1Threev8h_POST LD1Threev16b LD1Threev16b_POST LD1Twov1d LD1Twov1d_POST LD1Twov2d LD1Twov2d_POST LD1Twov2s LD1Twov2s_POST LD1Twov4h LD1Twov4h_POST LD1Twov4s LD1Twov4s_POST LD1Twov8b LD1Twov8b_POST LD1Twov8h LD1Twov8h_POST LD1Twov16b LD1Twov16b_POST LD1i8 LD1i8_POST LD1i16 LD1i16_POST LD1i32 LD1i32_POST LD1i64 LD1i64_POST
}

def AArch64MIR_ld1rb : AArch64MIR_TargetOp<"ld1rb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LD1RB_D_IMM LD1RB_H_IMM LD1RB_IMM LD1RB_S_IMM
}

def AArch64MIR_ld1rd : AArch64MIR_TargetOp<"ld1rd", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LD1RD_IMM
}

def AArch64MIR_ld1rh : AArch64MIR_TargetOp<"ld1rh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LD1RH_D_IMM LD1RH_IMM LD1RH_S_IMM
}

def AArch64MIR_ld1rob : AArch64MIR_TargetOp<"ld1rob", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LD1RO_B LD1RO_B_IMM
}

def AArch64MIR_ld1rod : AArch64MIR_TargetOp<"ld1rod", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LD1RO_D LD1RO_D_IMM
}

def AArch64MIR_ld1roh : AArch64MIR_TargetOp<"ld1roh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LD1RO_H LD1RO_H_IMM
}

def AArch64MIR_ld1row : AArch64MIR_TargetOp<"ld1row", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LD1RO_W LD1RO_W_IMM
}

def AArch64MIR_ld1rqb : AArch64MIR_TargetOp<"ld1rqb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LD1RQ_B LD1RQ_B_IMM
}

def AArch64MIR_ld1rqd : AArch64MIR_TargetOp<"ld1rqd", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LD1RQ_D LD1RQ_D_IMM
}

def AArch64MIR_ld1rqh : AArch64MIR_TargetOp<"ld1rqh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LD1RQ_H LD1RQ_H_IMM
}

def AArch64MIR_ld1rqw : AArch64MIR_TargetOp<"ld1rqw", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LD1RQ_W LD1RQ_W_IMM
}

def AArch64MIR_ld1rsb : AArch64MIR_TargetOp<"ld1rsb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LD1RSB_D_IMM LD1RSB_H_IMM LD1RSB_S_IMM
}

def AArch64MIR_ld1rsh : AArch64MIR_TargetOp<"ld1rsh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LD1RSH_D_IMM LD1RSH_S_IMM
}

def AArch64MIR_ld1rsw : AArch64MIR_TargetOp<"ld1rsw", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LD1RSW_IMM
}

def AArch64MIR_ld1rw : AArch64MIR_TargetOp<"ld1rw", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LD1RW_D_IMM LD1RW_IMM
}

def AArch64MIR_ld1r : AArch64MIR_TargetOp<"ld1r", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LD1Rv1d LD1Rv1d_POST LD1Rv2d LD1Rv2d_POST LD1Rv2s LD1Rv2s_POST LD1Rv4h LD1Rv4h_POST LD1Rv4s LD1Rv4s_POST LD1Rv8b LD1Rv8b_POST LD1Rv8h LD1Rv8h_POST LD1Rv16b LD1Rv16b_POST
}

def AArch64MIR_ld2b : AArch64MIR_TargetOp<"ld2b", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LD2B LD2B_IMM
}

def AArch64MIR_ld2d : AArch64MIR_TargetOp<"ld2d", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LD2D LD2D_IMM
}

def AArch64MIR_ld2h : AArch64MIR_TargetOp<"ld2h", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LD2H LD2H_IMM
}

def AArch64MIR_ld2q : AArch64MIR_TargetOp<"ld2q", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LD2Q LD2Q_IMM
}

def AArch64MIR_ld2r : AArch64MIR_TargetOp<"ld2r", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LD2Rv1d LD2Rv1d_POST LD2Rv2d LD2Rv2d_POST LD2Rv2s LD2Rv2s_POST LD2Rv4h LD2Rv4h_POST LD2Rv4s LD2Rv4s_POST LD2Rv8b LD2Rv8b_POST LD2Rv8h LD2Rv8h_POST LD2Rv16b LD2Rv16b_POST
}

def AArch64MIR_ld2 : AArch64MIR_TargetOp<"ld2", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LD2Twov2d LD2Twov2d_POST LD2Twov2s LD2Twov2s_POST LD2Twov4h LD2Twov4h_POST LD2Twov4s LD2Twov4s_POST LD2Twov8b LD2Twov8b_POST LD2Twov8h LD2Twov8h_POST LD2Twov16b LD2Twov16b_POST LD2i8 LD2i8_POST LD2i16 LD2i16_POST LD2i32 LD2i32_POST LD2i64 LD2i64_POST
}

def AArch64MIR_ld2w : AArch64MIR_TargetOp<"ld2w", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LD2W LD2W_IMM
}

def AArch64MIR_ld3b : AArch64MIR_TargetOp<"ld3b", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LD3B LD3B_IMM
}

def AArch64MIR_ld3d : AArch64MIR_TargetOp<"ld3d", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LD3D LD3D_IMM
}

def AArch64MIR_ld3h : AArch64MIR_TargetOp<"ld3h", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LD3H LD3H_IMM
}

def AArch64MIR_ld3q : AArch64MIR_TargetOp<"ld3q", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LD3Q LD3Q_IMM
}

def AArch64MIR_ld3r : AArch64MIR_TargetOp<"ld3r", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LD3Rv1d LD3Rv1d_POST LD3Rv2d LD3Rv2d_POST LD3Rv2s LD3Rv2s_POST LD3Rv4h LD3Rv4h_POST LD3Rv4s LD3Rv4s_POST LD3Rv8b LD3Rv8b_POST LD3Rv8h LD3Rv8h_POST LD3Rv16b LD3Rv16b_POST
}

def AArch64MIR_ld3 : AArch64MIR_TargetOp<"ld3", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LD3Threev2d LD3Threev2d_POST LD3Threev2s LD3Threev2s_POST LD3Threev4h LD3Threev4h_POST LD3Threev4s LD3Threev4s_POST LD3Threev8b LD3Threev8b_POST LD3Threev8h LD3Threev8h_POST LD3Threev16b LD3Threev16b_POST LD3i8 LD3i8_POST LD3i16 LD3i16_POST LD3i32 LD3i32_POST LD3i64 LD3i64_POST
}

def AArch64MIR_ld3w : AArch64MIR_TargetOp<"ld3w", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LD3W LD3W_IMM
}

def AArch64MIR_ld4b : AArch64MIR_TargetOp<"ld4b", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LD4B LD4B_IMM
}

def AArch64MIR_ld4d : AArch64MIR_TargetOp<"ld4d", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LD4D LD4D_IMM
}

def AArch64MIR_ld4 : AArch64MIR_TargetOp<"ld4", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LD4Fourv2d LD4Fourv2d_POST LD4Fourv2s LD4Fourv2s_POST LD4Fourv4h LD4Fourv4h_POST LD4Fourv4s LD4Fourv4s_POST LD4Fourv8b LD4Fourv8b_POST LD4Fourv8h LD4Fourv8h_POST LD4Fourv16b LD4Fourv16b_POST LD4i8 LD4i8_POST LD4i16 LD4i16_POST LD4i32 LD4i32_POST LD4i64 LD4i64_POST
}

def AArch64MIR_ld4h : AArch64MIR_TargetOp<"ld4h", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LD4H LD4H_IMM
}

def AArch64MIR_ld4q : AArch64MIR_TargetOp<"ld4q", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LD4Q LD4Q_IMM
}

def AArch64MIR_ld4r : AArch64MIR_TargetOp<"ld4r", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LD4Rv1d LD4Rv1d_POST LD4Rv2d LD4Rv2d_POST LD4Rv2s LD4Rv2s_POST LD4Rv4h LD4Rv4h_POST LD4Rv4s LD4Rv4s_POST LD4Rv8b LD4Rv8b_POST LD4Rv8h LD4Rv8h_POST LD4Rv16b LD4Rv16b_POST
}

def AArch64MIR_ld4w : AArch64MIR_TargetOp<"ld4w", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LD4W LD4W_IMM
}

def AArch64MIR_ld64b : AArch64MIR_TargetOp<"ld64b", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LD64B
}

def AArch64MIR_ldaddab : AArch64MIR_TargetOp<"ldaddab", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDADDAB
}

def AArch64MIR_ldaddah : AArch64MIR_TargetOp<"ldaddah", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDADDAH
}

def AArch64MIR_ldaddalb : AArch64MIR_TargetOp<"ldaddalb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDADDALB
}

def AArch64MIR_ldaddalh : AArch64MIR_TargetOp<"ldaddalh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDADDALH
}

def AArch64MIR_ldaddal : AArch64MIR_TargetOp<"ldaddal", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDADDALW LDADDALX
}

def AArch64MIR_ldadda : AArch64MIR_TargetOp<"ldadda", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDADDAW LDADDAX
}

def AArch64MIR_ldaddb : AArch64MIR_TargetOp<"ldaddb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDADDB
}

def AArch64MIR_ldaddh : AArch64MIR_TargetOp<"ldaddh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDADDH
}

def AArch64MIR_ldaddlb : AArch64MIR_TargetOp<"ldaddlb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDADDLB
}

def AArch64MIR_ldaddlh : AArch64MIR_TargetOp<"ldaddlh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDADDLH
}

def AArch64MIR_ldaddl : AArch64MIR_TargetOp<"ldaddl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDADDLW LDADDLX
}

def AArch64MIR_ldadd : AArch64MIR_TargetOp<"ldadd", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDADDW LDADDX
}

def AArch64MIR_ldap1 : AArch64MIR_TargetOp<"ldap1", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDAP1
}

def AArch64MIR_ldapp : AArch64MIR_TargetOp<"ldapp", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDAPPi
}

def AArch64MIR_ldaprb : AArch64MIR_TargetOp<"ldaprb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDAPRB
}

def AArch64MIR_ldaprh : AArch64MIR_TargetOp<"ldaprh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDAPRH
}

def AArch64MIR_ldapr : AArch64MIR_TargetOp<"ldapr", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDAPRW LDAPRWpost LDAPRX LDAPRXpost
}

def AArch64MIR_ldapurb : AArch64MIR_TargetOp<"ldapurb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDAPURBi
}

def AArch64MIR_ldapurh : AArch64MIR_TargetOp<"ldapurh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDAPURHi
}

def AArch64MIR_ldapursb : AArch64MIR_TargetOp<"ldapursb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDAPURSBWi LDAPURSBXi
}

def AArch64MIR_ldapursh : AArch64MIR_TargetOp<"ldapursh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDAPURSHWi LDAPURSHXi
}

def AArch64MIR_ldapursw : AArch64MIR_TargetOp<"ldapursw", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDAPURSWi
}

def AArch64MIR_ldapur : AArch64MIR_TargetOp<"ldapur", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDAPURXi LDAPURbi LDAPURdi LDAPURhi LDAPURi LDAPURqi LDAPURsi
}

def AArch64MIR_ldap : AArch64MIR_TargetOp<"ldap", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDAPi
}

def AArch64MIR_ldarb : AArch64MIR_TargetOp<"ldarb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDARB
}

def AArch64MIR_ldarh : AArch64MIR_TargetOp<"ldarh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDARH
}

def AArch64MIR_ldar : AArch64MIR_TargetOp<"ldar", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDARW LDARX
}

def AArch64MIR_ldatxr : AArch64MIR_TargetOp<"ldatxr", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDATXRW LDATXRX
}

def AArch64MIR_ldaxp : AArch64MIR_TargetOp<"ldaxp", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDAXPW LDAXPX
}

def AArch64MIR_ldaxrb : AArch64MIR_TargetOp<"ldaxrb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDAXRB
}

def AArch64MIR_ldaxrh : AArch64MIR_TargetOp<"ldaxrh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDAXRH
}

def AArch64MIR_ldaxr : AArch64MIR_TargetOp<"ldaxr", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDAXRW LDAXRX
}

def AArch64MIR_ldbfadd : AArch64MIR_TargetOp<"ldbfadd", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDBFADD
}

def AArch64MIR_ldbfadda : AArch64MIR_TargetOp<"ldbfadda", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDBFADDA
}

def AArch64MIR_ldbfaddal : AArch64MIR_TargetOp<"ldbfaddal", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDBFADDAL
}

def AArch64MIR_ldbfaddl : AArch64MIR_TargetOp<"ldbfaddl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDBFADDL
}

def AArch64MIR_ldbfmax : AArch64MIR_TargetOp<"ldbfmax", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDBFMAX
}

def AArch64MIR_ldbfmaxa : AArch64MIR_TargetOp<"ldbfmaxa", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDBFMAXA
}

def AArch64MIR_ldbfmaxal : AArch64MIR_TargetOp<"ldbfmaxal", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDBFMAXAL
}

def AArch64MIR_ldbfmaxl : AArch64MIR_TargetOp<"ldbfmaxl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDBFMAXL
}

def AArch64MIR_ldbfmaxnm : AArch64MIR_TargetOp<"ldbfmaxnm", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDBFMAXNM
}

def AArch64MIR_ldbfmaxnma : AArch64MIR_TargetOp<"ldbfmaxnma", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDBFMAXNMA
}

def AArch64MIR_ldbfmaxnmal : AArch64MIR_TargetOp<"ldbfmaxnmal", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDBFMAXNMAL
}

def AArch64MIR_ldbfmaxnml : AArch64MIR_TargetOp<"ldbfmaxnml", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDBFMAXNML
}

def AArch64MIR_ldbfmin : AArch64MIR_TargetOp<"ldbfmin", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDBFMIN
}

def AArch64MIR_ldbfmina : AArch64MIR_TargetOp<"ldbfmina", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDBFMINA
}

def AArch64MIR_ldbfminal : AArch64MIR_TargetOp<"ldbfminal", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDBFMINAL
}

def AArch64MIR_ldbfminl : AArch64MIR_TargetOp<"ldbfminl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDBFMINL
}

def AArch64MIR_ldbfminnm : AArch64MIR_TargetOp<"ldbfminnm", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDBFMINNM
}

def AArch64MIR_ldbfminnma : AArch64MIR_TargetOp<"ldbfminnma", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDBFMINNMA
}

def AArch64MIR_ldbfminnmal : AArch64MIR_TargetOp<"ldbfminnmal", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDBFMINNMAL
}

def AArch64MIR_ldbfminnml : AArch64MIR_TargetOp<"ldbfminnml", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDBFMINNML
}

def AArch64MIR_ldclrab : AArch64MIR_TargetOp<"ldclrab", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDCLRAB
}

def AArch64MIR_ldclrah : AArch64MIR_TargetOp<"ldclrah", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDCLRAH
}

def AArch64MIR_ldclralb : AArch64MIR_TargetOp<"ldclralb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDCLRALB
}

def AArch64MIR_ldclralh : AArch64MIR_TargetOp<"ldclralh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDCLRALH
}

def AArch64MIR_ldclral : AArch64MIR_TargetOp<"ldclral", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDCLRALW LDCLRALX
}

def AArch64MIR_ldclra : AArch64MIR_TargetOp<"ldclra", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDCLRAW LDCLRAX
}

def AArch64MIR_ldclrb : AArch64MIR_TargetOp<"ldclrb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDCLRB
}

def AArch64MIR_ldclrh : AArch64MIR_TargetOp<"ldclrh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDCLRH
}

def AArch64MIR_ldclrlb : AArch64MIR_TargetOp<"ldclrlb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDCLRLB
}

def AArch64MIR_ldclrlh : AArch64MIR_TargetOp<"ldclrlh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDCLRLH
}

def AArch64MIR_ldclrl : AArch64MIR_TargetOp<"ldclrl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDCLRLW LDCLRLX
}

def AArch64MIR_ldclrp : AArch64MIR_TargetOp<"ldclrp", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDCLRP
}

def AArch64MIR_ldclrpa : AArch64MIR_TargetOp<"ldclrpa", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDCLRPA
}

def AArch64MIR_ldclrpal : AArch64MIR_TargetOp<"ldclrpal", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDCLRPAL
}

def AArch64MIR_ldclrpl : AArch64MIR_TargetOp<"ldclrpl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDCLRPL
}

def AArch64MIR_ldclr : AArch64MIR_TargetOp<"ldclr", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDCLRW LDCLRX
}

def AArch64MIR_ldeorab : AArch64MIR_TargetOp<"ldeorab", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDEORAB
}

def AArch64MIR_ldeorah : AArch64MIR_TargetOp<"ldeorah", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDEORAH
}

def AArch64MIR_ldeoralb : AArch64MIR_TargetOp<"ldeoralb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDEORALB
}

def AArch64MIR_ldeoralh : AArch64MIR_TargetOp<"ldeoralh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDEORALH
}

def AArch64MIR_ldeoral : AArch64MIR_TargetOp<"ldeoral", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDEORALW LDEORALX
}

def AArch64MIR_ldeora : AArch64MIR_TargetOp<"ldeora", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDEORAW LDEORAX
}

def AArch64MIR_ldeorb : AArch64MIR_TargetOp<"ldeorb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDEORB
}

def AArch64MIR_ldeorh : AArch64MIR_TargetOp<"ldeorh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDEORH
}

def AArch64MIR_ldeorlb : AArch64MIR_TargetOp<"ldeorlb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDEORLB
}

def AArch64MIR_ldeorlh : AArch64MIR_TargetOp<"ldeorlh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDEORLH
}

def AArch64MIR_ldeorl : AArch64MIR_TargetOp<"ldeorl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDEORLW LDEORLX
}

def AArch64MIR_ldeor : AArch64MIR_TargetOp<"ldeor", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDEORW LDEORX
}

def AArch64MIR_ldfadda : AArch64MIR_TargetOp<"ldfadda", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDFADDAD LDFADDAH LDFADDAS
}

def AArch64MIR_ldfaddal : AArch64MIR_TargetOp<"ldfaddal", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDFADDALD LDFADDALH LDFADDALS
}

def AArch64MIR_ldfadd : AArch64MIR_TargetOp<"ldfadd", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDFADDD LDFADDH LDFADDS
}

def AArch64MIR_ldfaddl : AArch64MIR_TargetOp<"ldfaddl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDFADDLD LDFADDLH LDFADDLS
}

def AArch64MIR_ldfmaxa : AArch64MIR_TargetOp<"ldfmaxa", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDFMAXAD LDFMAXAH LDFMAXAS
}

def AArch64MIR_ldfmaxal : AArch64MIR_TargetOp<"ldfmaxal", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDFMAXALD LDFMAXALH LDFMAXALS
}

def AArch64MIR_ldfmax : AArch64MIR_TargetOp<"ldfmax", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDFMAXD LDFMAXH LDFMAXS
}

def AArch64MIR_ldfmaxl : AArch64MIR_TargetOp<"ldfmaxl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDFMAXLD LDFMAXLH LDFMAXLS
}

def AArch64MIR_ldfmaxnma : AArch64MIR_TargetOp<"ldfmaxnma", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDFMAXNMAD LDFMAXNMAH LDFMAXNMAS
}

def AArch64MIR_ldfmaxnmal : AArch64MIR_TargetOp<"ldfmaxnmal", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDFMAXNMALD LDFMAXNMALH LDFMAXNMALS
}

def AArch64MIR_ldfmaxnm : AArch64MIR_TargetOp<"ldfmaxnm", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDFMAXNMD LDFMAXNMH LDFMAXNMS
}

def AArch64MIR_ldfmaxnml : AArch64MIR_TargetOp<"ldfmaxnml", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDFMAXNMLD LDFMAXNMLH LDFMAXNMLS
}

def AArch64MIR_ldfmina : AArch64MIR_TargetOp<"ldfmina", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDFMINAD LDFMINAH LDFMINAS
}

def AArch64MIR_ldfminal : AArch64MIR_TargetOp<"ldfminal", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDFMINALD LDFMINALH LDFMINALS
}

def AArch64MIR_ldfmin : AArch64MIR_TargetOp<"ldfmin", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDFMIND LDFMINH LDFMINS
}

def AArch64MIR_ldfminl : AArch64MIR_TargetOp<"ldfminl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDFMINLD LDFMINLH LDFMINLS
}

def AArch64MIR_ldfminnma : AArch64MIR_TargetOp<"ldfminnma", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDFMINNMAD LDFMINNMAH LDFMINNMAS
}

def AArch64MIR_ldfminnmal : AArch64MIR_TargetOp<"ldfminnmal", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDFMINNMALD LDFMINNMALH LDFMINNMALS
}

def AArch64MIR_ldfminnm : AArch64MIR_TargetOp<"ldfminnm", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDFMINNMD LDFMINNMH LDFMINNMS
}

def AArch64MIR_ldfminnml : AArch64MIR_TargetOp<"ldfminnml", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDFMINNMLD LDFMINNMLH LDFMINNMLS
}

def AArch64MIR_ldg : AArch64MIR_TargetOp<"ldg", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDG
}

def AArch64MIR_ldgm : AArch64MIR_TargetOp<"ldgm", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDGM
}

def AArch64MIR_ldiapp : AArch64MIR_TargetOp<"ldiapp", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDIAPPW LDIAPPWpost LDIAPPX LDIAPPXpost
}

def AArch64MIR_ldlarb : AArch64MIR_TargetOp<"ldlarb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDLARB
}

def AArch64MIR_ldlarh : AArch64MIR_TargetOp<"ldlarh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDLARH
}

def AArch64MIR_ldlar : AArch64MIR_TargetOp<"ldlar", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDLARW LDLARX
}

def AArch64MIR_ldnf1b : AArch64MIR_TargetOp<"ldnf1b", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDNF1B_D_IMM LDNF1B_H_IMM LDNF1B_IMM LDNF1B_S_IMM
}

def AArch64MIR_ldnf1d : AArch64MIR_TargetOp<"ldnf1d", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDNF1D_IMM
}

def AArch64MIR_ldnf1h : AArch64MIR_TargetOp<"ldnf1h", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDNF1H_D_IMM LDNF1H_IMM LDNF1H_S_IMM
}

def AArch64MIR_ldnf1sb : AArch64MIR_TargetOp<"ldnf1sb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDNF1SB_D_IMM LDNF1SB_H_IMM LDNF1SB_S_IMM
}

def AArch64MIR_ldnf1sh : AArch64MIR_TargetOp<"ldnf1sh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDNF1SH_D_IMM LDNF1SH_S_IMM
}

def AArch64MIR_ldnf1sw : AArch64MIR_TargetOp<"ldnf1sw", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDNF1SW_D_IMM
}

def AArch64MIR_ldnf1w : AArch64MIR_TargetOp<"ldnf1w", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDNF1W_D_IMM LDNF1W_IMM
}

def AArch64MIR_ldnp : AArch64MIR_TargetOp<"ldnp", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDNPDi LDNPQi LDNPSi LDNPWi LDNPXi
}

def AArch64MIR_ldnt1b : AArch64MIR_TargetOp<"ldnt1b", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDNT1B_2Z LDNT1B_2Z_IMM LDNT1B_2Z_STRIDED LDNT1B_2Z_STRIDED_IMM LDNT1B_4Z LDNT1B_4Z_IMM LDNT1B_4Z_STRIDED LDNT1B_4Z_STRIDED_IMM LDNT1B_ZRI LDNT1B_ZRR LDNT1B_ZZR_D LDNT1B_ZZR_S
}

def AArch64MIR_ldnt1d : AArch64MIR_TargetOp<"ldnt1d", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDNT1D_2Z LDNT1D_2Z_IMM LDNT1D_2Z_STRIDED LDNT1D_2Z_STRIDED_IMM LDNT1D_4Z LDNT1D_4Z_IMM LDNT1D_4Z_STRIDED LDNT1D_4Z_STRIDED_IMM LDNT1D_ZRI LDNT1D_ZRR LDNT1D_ZZR_D
}

def AArch64MIR_ldnt1h : AArch64MIR_TargetOp<"ldnt1h", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDNT1H_2Z LDNT1H_2Z_IMM LDNT1H_2Z_STRIDED LDNT1H_2Z_STRIDED_IMM LDNT1H_4Z LDNT1H_4Z_IMM LDNT1H_4Z_STRIDED LDNT1H_4Z_STRIDED_IMM LDNT1H_ZRI LDNT1H_ZRR LDNT1H_ZZR_D LDNT1H_ZZR_S
}

def AArch64MIR_ldnt1sb : AArch64MIR_TargetOp<"ldnt1sb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDNT1SB_ZZR_D LDNT1SB_ZZR_S
}

def AArch64MIR_ldnt1sh : AArch64MIR_TargetOp<"ldnt1sh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDNT1SH_ZZR_D LDNT1SH_ZZR_S
}

def AArch64MIR_ldnt1sw : AArch64MIR_TargetOp<"ldnt1sw", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDNT1SW_ZZR_D
}

def AArch64MIR_ldnt1w : AArch64MIR_TargetOp<"ldnt1w", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDNT1W_2Z LDNT1W_2Z_IMM LDNT1W_2Z_STRIDED LDNT1W_2Z_STRIDED_IMM LDNT1W_4Z LDNT1W_4Z_IMM LDNT1W_4Z_STRIDED LDNT1W_4Z_STRIDED_IMM LDNT1W_ZRI LDNT1W_ZRR LDNT1W_ZZR_D LDNT1W_ZZR_S
}

def AArch64MIR_ldp : AArch64MIR_TargetOp<"ldp", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDPDi LDPDpost LDPDpre LDPQi LDPQpost LDPQpre LDPSi LDPSpost LDPSpre LDPWi LDPWpost LDPWpre LDPXi LDPXpost LDPXpre
}

def AArch64MIR_ldpsw : AArch64MIR_TargetOp<"ldpsw", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDPSWi LDPSWpost LDPSWpre
}

def AArch64MIR_ldraa : AArch64MIR_TargetOp<"ldraa", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDRAAindexed LDRAAwriteback
}

def AArch64MIR_ldrab : AArch64MIR_TargetOp<"ldrab", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDRABindexed LDRABwriteback
}

def AArch64MIR_ldrb : AArch64MIR_TargetOp<"ldrb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDRBBpost LDRBBpre LDRBBroW LDRBBroX LDRBBui
}

def AArch64MIR_ldr : AArch64MIR_TargetOp<"ldr", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDRBpost LDRBpre LDRBroW LDRBroX LDRBui LDRDl LDRDpost LDRDpre LDRDroW LDRDroX LDRDui LDRHpost LDRHpre LDRHroW LDRHroX LDRHui LDRQl LDRQpost LDRQpre LDRQroW LDRQroX LDRQui LDRSl LDRSpost LDRSpre LDRSroW LDRSroX LDRSui LDRWl LDRWpost LDRWpre LDRWroW LDRWroX LDRWui LDRXl LDRXpost LDRXpre LDRXroW LDRXroX LDRXui LDR_PXI LDR_TX LDR_ZA LDR_ZXI
}

def AArch64MIR_ldrh : AArch64MIR_TargetOp<"ldrh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDRHHpost LDRHHpre LDRHHroW LDRHHroX LDRHHui
}

def AArch64MIR_ldrsb : AArch64MIR_TargetOp<"ldrsb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDRSBWpost LDRSBWpre LDRSBWroW LDRSBWroX LDRSBWui LDRSBXpost LDRSBXpre LDRSBXroW LDRSBXroX LDRSBXui
}

def AArch64MIR_ldrsh : AArch64MIR_TargetOp<"ldrsh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDRSHWpost LDRSHWpre LDRSHWroW LDRSHWroX LDRSHWui LDRSHXpost LDRSHXpre LDRSHXroW LDRSHXroX LDRSHXui
}

def AArch64MIR_ldrsw : AArch64MIR_TargetOp<"ldrsw", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDRSWl LDRSWpost LDRSWpre LDRSWroW LDRSWroX LDRSWui
}

def AArch64MIR_ldsetab : AArch64MIR_TargetOp<"ldsetab", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDSETAB
}

def AArch64MIR_ldsetah : AArch64MIR_TargetOp<"ldsetah", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDSETAH
}

def AArch64MIR_ldsetalb : AArch64MIR_TargetOp<"ldsetalb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDSETALB
}

def AArch64MIR_ldsetalh : AArch64MIR_TargetOp<"ldsetalh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDSETALH
}

def AArch64MIR_ldsetal : AArch64MIR_TargetOp<"ldsetal", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDSETALW LDSETALX
}

def AArch64MIR_ldseta : AArch64MIR_TargetOp<"ldseta", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDSETAW LDSETAX
}

def AArch64MIR_ldsetb : AArch64MIR_TargetOp<"ldsetb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDSETB
}

def AArch64MIR_ldseth : AArch64MIR_TargetOp<"ldseth", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDSETH
}

def AArch64MIR_ldsetlb : AArch64MIR_TargetOp<"ldsetlb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDSETLB
}

def AArch64MIR_ldsetlh : AArch64MIR_TargetOp<"ldsetlh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDSETLH
}

def AArch64MIR_ldsetl : AArch64MIR_TargetOp<"ldsetl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDSETLW LDSETLX
}

def AArch64MIR_ldsetp : AArch64MIR_TargetOp<"ldsetp", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDSETP
}

def AArch64MIR_ldsetpa : AArch64MIR_TargetOp<"ldsetpa", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDSETPA
}

def AArch64MIR_ldsetpal : AArch64MIR_TargetOp<"ldsetpal", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDSETPAL
}

def AArch64MIR_ldsetpl : AArch64MIR_TargetOp<"ldsetpl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDSETPL
}

def AArch64MIR_ldset : AArch64MIR_TargetOp<"ldset", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDSETW LDSETX
}

def AArch64MIR_ldsmaxab : AArch64MIR_TargetOp<"ldsmaxab", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDSMAXAB
}

def AArch64MIR_ldsmaxah : AArch64MIR_TargetOp<"ldsmaxah", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDSMAXAH
}

def AArch64MIR_ldsmaxalb : AArch64MIR_TargetOp<"ldsmaxalb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDSMAXALB
}

def AArch64MIR_ldsmaxalh : AArch64MIR_TargetOp<"ldsmaxalh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDSMAXALH
}

def AArch64MIR_ldsmaxal : AArch64MIR_TargetOp<"ldsmaxal", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDSMAXALW LDSMAXALX
}

def AArch64MIR_ldsmaxa : AArch64MIR_TargetOp<"ldsmaxa", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDSMAXAW LDSMAXAX
}

def AArch64MIR_ldsmaxb : AArch64MIR_TargetOp<"ldsmaxb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDSMAXB
}

def AArch64MIR_ldsmaxh : AArch64MIR_TargetOp<"ldsmaxh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDSMAXH
}

def AArch64MIR_ldsmaxlb : AArch64MIR_TargetOp<"ldsmaxlb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDSMAXLB
}

def AArch64MIR_ldsmaxlh : AArch64MIR_TargetOp<"ldsmaxlh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDSMAXLH
}

def AArch64MIR_ldsmaxl : AArch64MIR_TargetOp<"ldsmaxl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDSMAXLW LDSMAXLX
}

def AArch64MIR_ldsmax : AArch64MIR_TargetOp<"ldsmax", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDSMAXW LDSMAXX
}

def AArch64MIR_ldsminab : AArch64MIR_TargetOp<"ldsminab", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDSMINAB
}

def AArch64MIR_ldsminah : AArch64MIR_TargetOp<"ldsminah", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDSMINAH
}

def AArch64MIR_ldsminalb : AArch64MIR_TargetOp<"ldsminalb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDSMINALB
}

def AArch64MIR_ldsminalh : AArch64MIR_TargetOp<"ldsminalh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDSMINALH
}

def AArch64MIR_ldsminal : AArch64MIR_TargetOp<"ldsminal", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDSMINALW LDSMINALX
}

def AArch64MIR_ldsmina : AArch64MIR_TargetOp<"ldsmina", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDSMINAW LDSMINAX
}

def AArch64MIR_ldsminb : AArch64MIR_TargetOp<"ldsminb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDSMINB
}

def AArch64MIR_ldsminh : AArch64MIR_TargetOp<"ldsminh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDSMINH
}

def AArch64MIR_ldsminlb : AArch64MIR_TargetOp<"ldsminlb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDSMINLB
}

def AArch64MIR_ldsminlh : AArch64MIR_TargetOp<"ldsminlh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDSMINLH
}

def AArch64MIR_ldsminl : AArch64MIR_TargetOp<"ldsminl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDSMINLW LDSMINLX
}

def AArch64MIR_ldsmin : AArch64MIR_TargetOp<"ldsmin", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDSMINW LDSMINX
}

def AArch64MIR_ldtaddal : AArch64MIR_TargetOp<"ldtaddal", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDTADDALW LDTADDALX
}

def AArch64MIR_ldtadda : AArch64MIR_TargetOp<"ldtadda", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDTADDAW LDTADDAX
}

def AArch64MIR_ldtaddl : AArch64MIR_TargetOp<"ldtaddl", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDTADDLW LDTADDLX
}

def AArch64MIR_ldtadd : AArch64MIR_TargetOp<"ldtadd", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDTADDW LDTADDX
}

def AArch64MIR_ldtclral : AArch64MIR_TargetOp<"ldtclral", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDTCLRALW LDTCLRALX
}

def AArch64MIR_ldtclra : AArch64MIR_TargetOp<"ldtclra", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDTCLRAW LDTCLRAX
}

def AArch64MIR_ldtclrl : AArch64MIR_TargetOp<"ldtclrl", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDTCLRLW LDTCLRLX
}

def AArch64MIR_ldtclr : AArch64MIR_TargetOp<"ldtclr", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDTCLRW LDTCLRX
}

def AArch64MIR_ldtnp : AArch64MIR_TargetOp<"ldtnp", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDTNPQi LDTNPXi
}

def AArch64MIR_ldtp : AArch64MIR_TargetOp<"ldtp", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDTPQi LDTPQpost LDTPQpre LDTPi LDTPpost LDTPpre
}

def AArch64MIR_ldtrb : AArch64MIR_TargetOp<"ldtrb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDTRBi
}

def AArch64MIR_ldtrh : AArch64MIR_TargetOp<"ldtrh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDTRHi
}

def AArch64MIR_ldtrsb : AArch64MIR_TargetOp<"ldtrsb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDTRSBWi LDTRSBXi
}

def AArch64MIR_ldtrsh : AArch64MIR_TargetOp<"ldtrsh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDTRSHWi LDTRSHXi
}

def AArch64MIR_ldtrsw : AArch64MIR_TargetOp<"ldtrsw", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDTRSWi
}

def AArch64MIR_ldtr : AArch64MIR_TargetOp<"ldtr", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDTRWi LDTRXi
}

def AArch64MIR_ldtsetal : AArch64MIR_TargetOp<"ldtsetal", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDTSETALW LDTSETALX
}

def AArch64MIR_ldtseta : AArch64MIR_TargetOp<"ldtseta", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDTSETAW LDTSETAX
}

def AArch64MIR_ldtsetl : AArch64MIR_TargetOp<"ldtsetl", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDTSETLW LDTSETLX
}

def AArch64MIR_ldtset : AArch64MIR_TargetOp<"ldtset", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDTSETW LDTSETX
}

def AArch64MIR_ldtxr : AArch64MIR_TargetOp<"ldtxr", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDTXRWr LDTXRXr
}

def AArch64MIR_ldumaxab : AArch64MIR_TargetOp<"ldumaxab", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDUMAXAB
}

def AArch64MIR_ldumaxah : AArch64MIR_TargetOp<"ldumaxah", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDUMAXAH
}

def AArch64MIR_ldumaxalb : AArch64MIR_TargetOp<"ldumaxalb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDUMAXALB
}

def AArch64MIR_ldumaxalh : AArch64MIR_TargetOp<"ldumaxalh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDUMAXALH
}

def AArch64MIR_ldumaxal : AArch64MIR_TargetOp<"ldumaxal", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDUMAXALW LDUMAXALX
}

def AArch64MIR_ldumaxa : AArch64MIR_TargetOp<"ldumaxa", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDUMAXAW LDUMAXAX
}

def AArch64MIR_ldumaxb : AArch64MIR_TargetOp<"ldumaxb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDUMAXB
}

def AArch64MIR_ldumaxh : AArch64MIR_TargetOp<"ldumaxh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDUMAXH
}

def AArch64MIR_ldumaxlb : AArch64MIR_TargetOp<"ldumaxlb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDUMAXLB
}

def AArch64MIR_ldumaxlh : AArch64MIR_TargetOp<"ldumaxlh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDUMAXLH
}

def AArch64MIR_ldumaxl : AArch64MIR_TargetOp<"ldumaxl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDUMAXLW LDUMAXLX
}

def AArch64MIR_ldumax : AArch64MIR_TargetOp<"ldumax", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDUMAXW LDUMAXX
}

def AArch64MIR_lduminab : AArch64MIR_TargetOp<"lduminab", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDUMINAB
}

def AArch64MIR_lduminah : AArch64MIR_TargetOp<"lduminah", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDUMINAH
}

def AArch64MIR_lduminalb : AArch64MIR_TargetOp<"lduminalb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDUMINALB
}

def AArch64MIR_lduminalh : AArch64MIR_TargetOp<"lduminalh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDUMINALH
}

def AArch64MIR_lduminal : AArch64MIR_TargetOp<"lduminal", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDUMINALW LDUMINALX
}

def AArch64MIR_ldumina : AArch64MIR_TargetOp<"ldumina", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDUMINAW LDUMINAX
}

def AArch64MIR_lduminb : AArch64MIR_TargetOp<"lduminb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDUMINB
}

def AArch64MIR_lduminh : AArch64MIR_TargetOp<"lduminh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDUMINH
}

def AArch64MIR_lduminlb : AArch64MIR_TargetOp<"lduminlb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDUMINLB
}

def AArch64MIR_lduminlh : AArch64MIR_TargetOp<"lduminlh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDUMINLH
}

def AArch64MIR_lduminl : AArch64MIR_TargetOp<"lduminl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDUMINLW LDUMINLX
}

def AArch64MIR_ldumin : AArch64MIR_TargetOp<"ldumin", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDUMINW LDUMINX
}

def AArch64MIR_ldurb : AArch64MIR_TargetOp<"ldurb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDURBBi
}

def AArch64MIR_ldur : AArch64MIR_TargetOp<"ldur", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDURBi LDURDi LDURHi LDURQi LDURSi LDURWi LDURXi
}

def AArch64MIR_ldurh : AArch64MIR_TargetOp<"ldurh", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDURHHi
}

def AArch64MIR_ldursb : AArch64MIR_TargetOp<"ldursb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDURSBWi LDURSBXi
}

def AArch64MIR_ldursh : AArch64MIR_TargetOp<"ldursh", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDURSHWi LDURSHXi
}

def AArch64MIR_ldursw : AArch64MIR_TargetOp<"ldursw", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDURSWi
}

def AArch64MIR_ldxp : AArch64MIR_TargetOp<"ldxp", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDXPW LDXPX
}

def AArch64MIR_ldxrb : AArch64MIR_TargetOp<"ldxrb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDXRB
}

def AArch64MIR_ldxrh : AArch64MIR_TargetOp<"ldxrh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDXRH
}

def AArch64MIR_ldxr : AArch64MIR_TargetOp<"ldxr", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LDXRW LDXRX
}

def AArch64MIR_lslr : AArch64MIR_TargetOp<"lslr", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LSLR_ZPmZ_B LSLR_ZPmZ_D LSLR_ZPmZ_H LSLR_ZPmZ_S
}

def AArch64MIR_lsl : AArch64MIR_TargetOp<"lsl", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LSLVWr LSLVXr LSL_WIDE_ZPmZ_B LSL_WIDE_ZPmZ_H LSL_WIDE_ZPmZ_S LSL_WIDE_ZZZ_B LSL_WIDE_ZZZ_H LSL_WIDE_ZZZ_S LSL_ZPmI_B LSL_ZPmI_D LSL_ZPmI_H LSL_ZPmI_S LSL_ZPmZ_B LSL_ZPmZ_D LSL_ZPmZ_H LSL_ZPmZ_S LSL_ZZI_B LSL_ZZI_D LSL_ZZI_H LSL_ZZI_S
}

def AArch64MIR_lsrr : AArch64MIR_TargetOp<"lsrr", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LSRR_ZPmZ_B LSRR_ZPmZ_D LSRR_ZPmZ_H LSRR_ZPmZ_S
}

def AArch64MIR_lsr : AArch64MIR_TargetOp<"lsr", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LSRVWr LSRVXr LSR_WIDE_ZPmZ_B LSR_WIDE_ZPmZ_H LSR_WIDE_ZPmZ_S LSR_WIDE_ZZZ_B LSR_WIDE_ZZZ_H LSR_WIDE_ZZZ_S LSR_ZPmI_B LSR_ZPmI_D LSR_ZPmI_H LSR_ZPmI_S LSR_ZPmZ_B LSR_ZPmZ_D LSR_ZPmZ_H LSR_ZPmZ_S LSR_ZZI_B LSR_ZZI_D LSR_ZZI_H LSR_ZZI_S
}

def AArch64MIR_luti2 : AArch64MIR_TargetOp<"luti2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LUT2_B LUT2_H LUTI2_2ZTZI_B LUTI2_2ZTZI_H LUTI2_2ZTZI_S LUTI2_4ZTZI_B LUTI2_4ZTZI_H LUTI2_4ZTZI_S LUTI2_S_2ZTZI_B LUTI2_S_2ZTZI_H LUTI2_S_4ZTZI_B LUTI2_S_4ZTZI_H LUTI2_ZTZI_B LUTI2_ZTZI_H LUTI2_ZTZI_S LUTI2_ZZZI_B LUTI2_ZZZI_H
}

def AArch64MIR_luti4 : AArch64MIR_TargetOp<"luti4", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LUT4_B LUT4_H LUTI4_2ZTZI_B LUTI4_2ZTZI_H LUTI4_2ZTZI_S LUTI4_4ZTZI_H LUTI4_4ZTZI_S LUTI4_4ZZT2Z LUTI4_S_2ZTZI_B LUTI4_S_2ZTZI_H LUTI4_S_4ZTZI_H LUTI4_S_4ZZT2Z LUTI4_Z2ZZI LUTI4_ZTZI_B LUTI4_ZTZI_H LUTI4_ZTZI_S LUTI4_ZZZI_B LUTI4_ZZZI_H
}

def AArch64MIR_luti6 : AArch64MIR_TargetOp<"luti6", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: LUTI6_4Z2Z2ZI LUTI6_4ZT3Z LUTI6_S_4Z2Z2ZI LUTI6_S_4ZT3Z LUTI6_Z2ZZ LUTI6_Z2ZZI_H LUTI6_ZTZ
}

def AArch64MIR_maddpt : AArch64MIR_TargetOp<"maddpt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: MADDPT
}

def AArch64MIR_madd : AArch64MIR_TargetOp<"madd", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: MADDWrrr MADDXrrr
}

def AArch64MIR_madpt : AArch64MIR_TargetOp<"madpt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: MAD_CPA
}

def AArch64MIR_mad : AArch64MIR_TargetOp<"mad", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: MAD_ZPmZZ_B MAD_ZPmZZ_D MAD_ZPmZZ_H MAD_ZPmZZ_S
}

def AArch64MIR_match : AArch64MIR_TargetOp<"match", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: MATCH_PPzZZ_B MATCH_PPzZZ_H
}

def AArch64MIR_mlapt : AArch64MIR_TargetOp<"mlapt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: MLA_CPA
}

def AArch64MIR_mla : AArch64MIR_TargetOp<"mla", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: MLA_ZPmZZ_B MLA_ZPmZZ_D MLA_ZPmZZ_H MLA_ZPmZZ_S MLA_ZZZI_D MLA_ZZZI_H MLA_ZZZI_S MLAv2i32 MLAv2i32_indexed MLAv4i16 MLAv4i16_indexed MLAv4i32 MLAv4i32_indexed MLAv8i8 MLAv8i16 MLAv8i16_indexed MLAv16i8
}

def AArch64MIR_mls : AArch64MIR_TargetOp<"mls", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: MLS_ZPmZZ_B MLS_ZPmZZ_D MLS_ZPmZZ_H MLS_ZPmZZ_S MLS_ZZZI_D MLS_ZZZI_H MLS_ZZZI_S MLSv2i32 MLSv2i32_indexed MLSv4i16 MLSv4i16_indexed MLSv4i32 MLSv4i32_indexed MLSv8i8 MLSv8i16 MLSv8i16_indexed MLSv16i8
}

def AArch64MIR_setge : AArch64MIR_TargetOp<"setge", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: MOPSSETGE
}

def AArch64MIR_setgen : AArch64MIR_TargetOp<"setgen", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: MOPSSETGEN
}

def AArch64MIR_setget : AArch64MIR_TargetOp<"setget", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: MOPSSETGET
}

def AArch64MIR_setgetn : AArch64MIR_TargetOp<"setgetn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: MOPSSETGETN
}

def AArch64MIR_movaz : AArch64MIR_TargetOp<"movaz", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: MOVAZ_2ZMI_H_B MOVAZ_2ZMI_H_D MOVAZ_2ZMI_H_H MOVAZ_2ZMI_H_S MOVAZ_2ZMI_V_B MOVAZ_2ZMI_V_D MOVAZ_2ZMI_V_H MOVAZ_2ZMI_V_S MOVAZ_4ZMI_H_B MOVAZ_4ZMI_H_D MOVAZ_4ZMI_H_H MOVAZ_4ZMI_H_S MOVAZ_4ZMI_V_B MOVAZ_4ZMI_V_D MOVAZ_4ZMI_V_H MOVAZ_4ZMI_V_S MOVAZ_VG2_2ZMXI MOVAZ_VG4_4ZMXI MOVAZ_ZMI_H_B MOVAZ_ZMI_H_D MOVAZ_ZMI_H_H MOVAZ_ZMI_H_Q MOVAZ_ZMI_H_S MOVAZ_ZMI_V_B MOVAZ_ZMI_V_D MOVAZ_ZMI_V_H MOVAZ_ZMI_V_Q MOVAZ_ZMI_V_S
}

def AArch64MIR_movi : AArch64MIR_TargetOp<"movi", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: MOVID MOVIv2d_ns MOVIv2i32 MOVIv2s_msl MOVIv4i16 MOVIv4i32 MOVIv4s_msl MOVIv8b_ns MOVIv8i16 MOVIv16b_ns
}

def AArch64MIR_movk : AArch64MIR_TargetOp<"movk", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: MOVKWi MOVKXi
}

def AArch64MIR_movn : AArch64MIR_TargetOp<"movn", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: MOVNWi MOVNXi
}

def AArch64MIR_movprfx : AArch64MIR_TargetOp<"movprfx", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: MOVPRFX_ZPmZ_B MOVPRFX_ZPmZ_D MOVPRFX_ZPmZ_H MOVPRFX_ZPmZ_S MOVPRFX_ZPzZ_B MOVPRFX_ZPzZ_D MOVPRFX_ZPzZ_H MOVPRFX_ZPzZ_S MOVPRFX_ZZ
}

def AArch64MIR_movt : AArch64MIR_TargetOp<"movt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: MOVT_TIX MOVT_TIZ MOVT_XTI
}

def AArch64MIR_movz : AArch64MIR_TargetOp<"movz", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: MOVZWi MOVZXi
}

def AArch64MIR_mrrs : AArch64MIR_TargetOp<"mrrs", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: MRRS
}

def AArch64MIR_mrs : AArch64MIR_TargetOp<"mrs", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: MRS
}

def AArch64MIR_msb : AArch64MIR_TargetOp<"msb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: MSB_ZPmZZ_B MSB_ZPmZZ_D MSB_ZPmZZ_H MSB_ZPmZZ_S
}

def AArch64MIR_msr : AArch64MIR_TargetOp<"msr", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: MSR MSRpstateImm1 MSRpstateImm4 MSRpstatesvcrImm1
}

def AArch64MIR_msrr : AArch64MIR_TargetOp<"msrr", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: MSRR
}

def AArch64MIR_msubpt : AArch64MIR_TargetOp<"msubpt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: MSUBPT
}

def AArch64MIR_msub : AArch64MIR_TargetOp<"msub", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: MSUBWrrr MSUBXrrr
}

def AArch64MIR_mul : AArch64MIR_TargetOp<"mul", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: MUL_ZI_B MUL_ZI_D MUL_ZI_H MUL_ZI_S MUL_ZPmZ_B MUL_ZPmZ_D MUL_ZPmZ_H MUL_ZPmZ_S MUL_ZZZI_D MUL_ZZZI_H MUL_ZZZI_S MUL_ZZZ_B MUL_ZZZ_D MUL_ZZZ_H MUL_ZZZ_S MULv2i32 MULv2i32_indexed MULv4i16 MULv4i16_indexed MULv4i32 MULv4i32_indexed MULv8i8 MULv8i16 MULv8i16_indexed MULv16i8
}

def AArch64MIR_mvni : AArch64MIR_TargetOp<"mvni", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: MVNIv2i32 MVNIv2s_msl MVNIv4i16 MVNIv4i32 MVNIv4s_msl MVNIv8i16
}

def AArch64MIR_nands : AArch64MIR_TargetOp<"nands", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: NANDS_PPzPP
}

def AArch64MIR_nand : AArch64MIR_TargetOp<"nand", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: NAND_PPzPP
}

def AArch64MIR_nbsl : AArch64MIR_TargetOp<"nbsl", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: NBSL_ZZZZ
}

def AArch64MIR_neg : AArch64MIR_TargetOp<"neg", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: NEG_ZPmZ_B NEG_ZPmZ_D NEG_ZPmZ_H NEG_ZPmZ_S NEG_ZPzZ_B NEG_ZPzZ_D NEG_ZPzZ_H NEG_ZPzZ_S NEGv1i64 NEGv2i32 NEGv2i64 NEGv4i16 NEGv4i32 NEGv8i8 NEGv8i16 NEGv16i8
}

def AArch64MIR_nmatch : AArch64MIR_TargetOp<"nmatch", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: NMATCH_PPzZZ_B NMATCH_PPzZZ_H
}

def AArch64MIR_nors : AArch64MIR_TargetOp<"nors", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: NORS_PPzPP
}

def AArch64MIR_nor : AArch64MIR_TargetOp<"nor", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: NOR_PPzPP
}

def AArch64MIR_not_ : AArch64MIR_TargetOp<"not", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: NOT_ZPmZ_B NOT_ZPmZ_D NOT_ZPmZ_H NOT_ZPmZ_S NOT_ZPzZ_B NOT_ZPzZ_D NOT_ZPzZ_H NOT_ZPzZ_S NOTv8i8 NOTv16i8
}

def AArch64MIR_orns : AArch64MIR_TargetOp<"orns", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ORNS_PPzPP
}

def AArch64MIR_orn : AArch64MIR_TargetOp<"orn", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ORNWrs ORNXrs ORN_PPzPP ORNv8i8 ORNv16i8
}

def AArch64MIR_orqv : AArch64MIR_TargetOp<"orqv", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ORQV_VPZ_B ORQV_VPZ_D ORQV_VPZ_H ORQV_VPZ_S
}

def AArch64MIR_orrs : AArch64MIR_TargetOp<"orrs", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ORRS_PPzPP
}

def AArch64MIR_orr : AArch64MIR_TargetOp<"orr", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ORRWri ORRWrs ORRXri ORRXrs ORR_PPzPP ORR_ZI ORR_ZPmZ_B ORR_ZPmZ_D ORR_ZPmZ_H ORR_ZPmZ_S ORR_ZZZ ORRv2i32 ORRv4i16 ORRv4i32 ORRv8i8 ORRv8i16 ORRv16i8
}

def AArch64MIR_orv : AArch64MIR_TargetOp<"orv", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ORV_VPZ_B ORV_VPZ_D ORV_VPZ_H ORV_VPZ_S
}

def AArch64MIR_pacda : AArch64MIR_TargetOp<"pacda", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: PACDA
}

def AArch64MIR_pacdb : AArch64MIR_TargetOp<"pacdb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: PACDB
}

def AArch64MIR_pacdza : AArch64MIR_TargetOp<"pacdza", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: PACDZA
}

def AArch64MIR_pacdzb : AArch64MIR_TargetOp<"pacdzb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: PACDZB
}

def AArch64MIR_pacga : AArch64MIR_TargetOp<"pacga", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: PACGA
}

def AArch64MIR_pacia : AArch64MIR_TargetOp<"pacia", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: PACIA
}

def AArch64MIR_pacia171615 : AArch64MIR_TargetOp<"pacia171615", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: PACIA171615
}

def AArch64MIR_paciasppc : AArch64MIR_TargetOp<"paciasppc", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: PACIASPPC
}

def AArch64MIR_pacib : AArch64MIR_TargetOp<"pacib", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: PACIB
}

def AArch64MIR_pacib171615 : AArch64MIR_TargetOp<"pacib171615", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: PACIB171615
}

def AArch64MIR_pacibsppc : AArch64MIR_TargetOp<"pacibsppc", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: PACIBSPPC
}

def AArch64MIR_paciza : AArch64MIR_TargetOp<"paciza", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: PACIZA
}

def AArch64MIR_pacizb : AArch64MIR_TargetOp<"pacizb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: PACIZB
}

def AArch64MIR_pacnbiasppc : AArch64MIR_TargetOp<"pacnbiasppc", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: PACNBIASPPC
}

def AArch64MIR_pacnbibsppc : AArch64MIR_TargetOp<"pacnbibsppc", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: PACNBIBSPPC
}

def AArch64MIR_pext : AArch64MIR_TargetOp<"pext", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: PEXT_2PCI_B PEXT_2PCI_D PEXT_2PCI_H PEXT_2PCI_S PEXT_PCI_B PEXT_PCI_D PEXT_PCI_H PEXT_PCI_S
}

def AArch64MIR_pfalse : AArch64MIR_TargetOp<"pfalse", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: PFALSE
}

def AArch64MIR_pfirst : AArch64MIR_TargetOp<"pfirst", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: PFIRST_B
}

def AArch64MIR_pmlal : AArch64MIR_TargetOp<"pmlal", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: PMLAL_2ZZZ_Q
}

def AArch64MIR_pmov : AArch64MIR_TargetOp<"pmov", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: PMOV_PZI_B PMOV_PZI_D PMOV_PZI_H PMOV_PZI_S PMOV_ZIP_B PMOV_ZIP_D PMOV_ZIP_H PMOV_ZIP_S
}

def AArch64MIR_pmullb : AArch64MIR_TargetOp<"pmullb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: PMULLB_ZZZ_D PMULLB_ZZZ_H PMULLB_ZZZ_Q
}

def AArch64MIR_pmullt : AArch64MIR_TargetOp<"pmullt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: PMULLT_ZZZ_D PMULLT_ZZZ_H PMULLT_ZZZ_Q
}

def AArch64MIR_pmull : AArch64MIR_TargetOp<"pmull", [Pure, Commutative]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: PMULL_2ZZZ_Q PMULLv1i64 PMULLv8i8
}

def AArch64MIR_pmull2 : AArch64MIR_TargetOp<"pmull2", [Pure, Commutative]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: PMULLv2i64 PMULLv16i8
}

def AArch64MIR_pmul : AArch64MIR_TargetOp<"pmul", [Pure, Commutative]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: PMUL_ZZZ_B PMULv8i8 PMULv16i8
}

def AArch64MIR_pnext : AArch64MIR_TargetOp<"pnext", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: PNEXT_B PNEXT_D PNEXT_H PNEXT_S
}

def AArch64MIR_prfb : AArch64MIR_TargetOp<"prfb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: PRFB_D_PZI PRFB_D_SCALED PRFB_D_SXTW_SCALED PRFB_D_UXTW_SCALED PRFB_PRI PRFB_PRR PRFB_S_PZI PRFB_S_SXTW_SCALED PRFB_S_UXTW_SCALED
}

def AArch64MIR_prfd : AArch64MIR_TargetOp<"prfd", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: PRFD_D_PZI PRFD_D_SCALED PRFD_D_SXTW_SCALED PRFD_D_UXTW_SCALED PRFD_PRI PRFD_PRR PRFD_S_PZI PRFD_S_SXTW_SCALED PRFD_S_UXTW_SCALED
}

def AArch64MIR_prfh : AArch64MIR_TargetOp<"prfh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: PRFH_D_PZI PRFH_D_SCALED PRFH_D_SXTW_SCALED PRFH_D_UXTW_SCALED PRFH_PRI PRFH_PRR PRFH_S_PZI PRFH_S_SXTW_SCALED PRFH_S_UXTW_SCALED
}

def AArch64MIR_prfm : AArch64MIR_TargetOp<"prfm", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: PRFMl PRFMroW PRFMroX PRFMui
}

def AArch64MIR_prfum : AArch64MIR_TargetOp<"prfum", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: PRFUMi
}

def AArch64MIR_prfw : AArch64MIR_TargetOp<"prfw", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: PRFW_D_PZI PRFW_D_SCALED PRFW_D_SXTW_SCALED PRFW_D_UXTW_SCALED PRFW_PRI PRFW_PRR PRFW_S_PZI PRFW_S_SXTW_SCALED PRFW_S_UXTW_SCALED
}

def AArch64MIR_psel : AArch64MIR_TargetOp<"psel", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: PSEL_PPPRI_B PSEL_PPPRI_D PSEL_PPPRI_H PSEL_PPPRI_S
}

def AArch64MIR_ptest : AArch64MIR_TargetOp<"ptest", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: PTEST_PP
}

def AArch64MIR_ptrues : AArch64MIR_TargetOp<"ptrues", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: PTRUES_B PTRUES_D PTRUES_H PTRUES_S
}

def AArch64MIR_ptrue : AArch64MIR_TargetOp<"ptrue", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: PTRUE_B PTRUE_C_B PTRUE_C_D PTRUE_C_H PTRUE_C_S PTRUE_D PTRUE_H PTRUE_S
}

def AArch64MIR_punpkhi : AArch64MIR_TargetOp<"punpkhi", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: PUNPKHI_PP
}

def AArch64MIR_punpklo : AArch64MIR_TargetOp<"punpklo", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: PUNPKLO_PP
}

def AArch64MIR_raddhnb : AArch64MIR_TargetOp<"raddhnb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RADDHNB_ZZZ_B RADDHNB_ZZZ_H RADDHNB_ZZZ_S
}

def AArch64MIR_raddhnt : AArch64MIR_TargetOp<"raddhnt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RADDHNT_ZZZ_B RADDHNT_ZZZ_H RADDHNT_ZZZ_S
}

def AArch64MIR_raddhn : AArch64MIR_TargetOp<"raddhn", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RADDHNv2i64_v2i32 RADDHNv4i32_v4i16 RADDHNv8i16_v8i8
}

def AArch64MIR_raddhn2 : AArch64MIR_TargetOp<"raddhn2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RADDHNv2i64_v4i32 RADDHNv4i32_v8i16 RADDHNv8i16_v16i8
}

def AArch64MIR_rax1 : AArch64MIR_TargetOp<"rax1", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RAX1 RAX1_ZZZ_D
}

def AArch64MIR_rbit : AArch64MIR_TargetOp<"rbit", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RBITWr RBITXr RBIT_ZPmZ_B RBIT_ZPmZ_D RBIT_ZPmZ_H RBIT_ZPmZ_S RBIT_ZPzZ_B RBIT_ZPzZ_D RBIT_ZPzZ_H RBIT_ZPzZ_S RBITv8i8 RBITv16i8
}

def AArch64MIR_rcwcas : AArch64MIR_TargetOp<"rcwcas", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWCAS
}

def AArch64MIR_rcwcasa : AArch64MIR_TargetOp<"rcwcasa", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWCASA
}

def AArch64MIR_rcwcasal : AArch64MIR_TargetOp<"rcwcasal", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWCASAL
}

def AArch64MIR_rcwcasl : AArch64MIR_TargetOp<"rcwcasl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWCASL
}

def AArch64MIR_rcwcasp : AArch64MIR_TargetOp<"rcwcasp", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWCASP
}

def AArch64MIR_rcwcaspa : AArch64MIR_TargetOp<"rcwcaspa", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWCASPA
}

def AArch64MIR_rcwcaspal : AArch64MIR_TargetOp<"rcwcaspal", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWCASPAL
}

def AArch64MIR_rcwcaspl : AArch64MIR_TargetOp<"rcwcaspl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWCASPL
}

def AArch64MIR_rcwclr : AArch64MIR_TargetOp<"rcwclr", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWCLR
}

def AArch64MIR_rcwclra : AArch64MIR_TargetOp<"rcwclra", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWCLRA
}

def AArch64MIR_rcwclral : AArch64MIR_TargetOp<"rcwclral", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWCLRAL
}

def AArch64MIR_rcwclrl : AArch64MIR_TargetOp<"rcwclrl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWCLRL
}

def AArch64MIR_rcwclrp : AArch64MIR_TargetOp<"rcwclrp", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWCLRP
}

def AArch64MIR_rcwclrpa : AArch64MIR_TargetOp<"rcwclrpa", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWCLRPA
}

def AArch64MIR_rcwclrpal : AArch64MIR_TargetOp<"rcwclrpal", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWCLRPAL
}

def AArch64MIR_rcwclrpl : AArch64MIR_TargetOp<"rcwclrpl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWCLRPL
}

def AArch64MIR_rcwsclr : AArch64MIR_TargetOp<"rcwsclr", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWCLRS
}

def AArch64MIR_rcwsclra : AArch64MIR_TargetOp<"rcwsclra", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWCLRSA
}

def AArch64MIR_rcwsclral : AArch64MIR_TargetOp<"rcwsclral", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWCLRSAL
}

def AArch64MIR_rcwsclrl : AArch64MIR_TargetOp<"rcwsclrl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWCLRSL
}

def AArch64MIR_rcwsclrp : AArch64MIR_TargetOp<"rcwsclrp", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWCLRSP
}

def AArch64MIR_rcwsclrpa : AArch64MIR_TargetOp<"rcwsclrpa", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWCLRSPA
}

def AArch64MIR_rcwsclrpal : AArch64MIR_TargetOp<"rcwsclrpal", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWCLRSPAL
}

def AArch64MIR_rcwsclrpl : AArch64MIR_TargetOp<"rcwsclrpl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWCLRSPL
}

def AArch64MIR_rcwscas : AArch64MIR_TargetOp<"rcwscas", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWSCAS
}

def AArch64MIR_rcwscasa : AArch64MIR_TargetOp<"rcwscasa", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWSCASA
}

def AArch64MIR_rcwscasal : AArch64MIR_TargetOp<"rcwscasal", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWSCASAL
}

def AArch64MIR_rcwscasl : AArch64MIR_TargetOp<"rcwscasl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWSCASL
}

def AArch64MIR_rcwscasp : AArch64MIR_TargetOp<"rcwscasp", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWSCASP
}

def AArch64MIR_rcwscaspa : AArch64MIR_TargetOp<"rcwscaspa", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWSCASPA
}

def AArch64MIR_rcwscaspal : AArch64MIR_TargetOp<"rcwscaspal", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWSCASPAL
}

def AArch64MIR_rcwscaspl : AArch64MIR_TargetOp<"rcwscaspl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWSCASPL
}

def AArch64MIR_rcwset : AArch64MIR_TargetOp<"rcwset", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWSET
}

def AArch64MIR_rcwseta : AArch64MIR_TargetOp<"rcwseta", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWSETA
}

def AArch64MIR_rcwsetal : AArch64MIR_TargetOp<"rcwsetal", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWSETAL
}

def AArch64MIR_rcwsetl : AArch64MIR_TargetOp<"rcwsetl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWSETL
}

def AArch64MIR_rcwsetp : AArch64MIR_TargetOp<"rcwsetp", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWSETP
}

def AArch64MIR_rcwsetpa : AArch64MIR_TargetOp<"rcwsetpa", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWSETPA
}

def AArch64MIR_rcwsetpal : AArch64MIR_TargetOp<"rcwsetpal", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWSETPAL
}

def AArch64MIR_rcwsetpl : AArch64MIR_TargetOp<"rcwsetpl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWSETPL
}

def AArch64MIR_rcwsset : AArch64MIR_TargetOp<"rcwsset", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWSETS
}

def AArch64MIR_rcwsseta : AArch64MIR_TargetOp<"rcwsseta", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWSETSA
}

def AArch64MIR_rcwssetal : AArch64MIR_TargetOp<"rcwssetal", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWSETSAL
}

def AArch64MIR_rcwssetl : AArch64MIR_TargetOp<"rcwssetl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWSETSL
}

def AArch64MIR_rcwssetp : AArch64MIR_TargetOp<"rcwssetp", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWSETSP
}

def AArch64MIR_rcwssetpa : AArch64MIR_TargetOp<"rcwssetpa", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWSETSPA
}

def AArch64MIR_rcwssetpal : AArch64MIR_TargetOp<"rcwssetpal", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWSETSPAL
}

def AArch64MIR_rcwssetpl : AArch64MIR_TargetOp<"rcwssetpl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWSETSPL
}

def AArch64MIR_rcwswp : AArch64MIR_TargetOp<"rcwswp", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWSWP
}

def AArch64MIR_rcwswpa : AArch64MIR_TargetOp<"rcwswpa", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWSWPA
}

def AArch64MIR_rcwswpal : AArch64MIR_TargetOp<"rcwswpal", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWSWPAL
}

def AArch64MIR_rcwswpl : AArch64MIR_TargetOp<"rcwswpl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWSWPL
}

def AArch64MIR_rcwswpp : AArch64MIR_TargetOp<"rcwswpp", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWSWPP
}

def AArch64MIR_rcwswppa : AArch64MIR_TargetOp<"rcwswppa", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWSWPPA
}

def AArch64MIR_rcwswppal : AArch64MIR_TargetOp<"rcwswppal", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWSWPPAL
}

def AArch64MIR_rcwswppl : AArch64MIR_TargetOp<"rcwswppl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWSWPPL
}

def AArch64MIR_rcwsswp : AArch64MIR_TargetOp<"rcwsswp", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWSWPS
}

def AArch64MIR_rcwsswpa : AArch64MIR_TargetOp<"rcwsswpa", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWSWPSA
}

def AArch64MIR_rcwsswpal : AArch64MIR_TargetOp<"rcwsswpal", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWSWPSAL
}

def AArch64MIR_rcwsswpl : AArch64MIR_TargetOp<"rcwsswpl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWSWPSL
}

def AArch64MIR_rcwsswpp : AArch64MIR_TargetOp<"rcwsswpp", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWSWPSP
}

def AArch64MIR_rcwsswppa : AArch64MIR_TargetOp<"rcwsswppa", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWSWPSPA
}

def AArch64MIR_rcwsswppal : AArch64MIR_TargetOp<"rcwsswppal", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWSWPSPAL
}

def AArch64MIR_rcwsswppl : AArch64MIR_TargetOp<"rcwsswppl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RCWSWPSPL
}

def AArch64MIR_rdffrs : AArch64MIR_TargetOp<"rdffrs", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RDFFRS_PPz
}

def AArch64MIR_rdffr : AArch64MIR_TargetOp<"rdffr", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RDFFR_P RDFFR_PPz
}

def AArch64MIR_rdsvl : AArch64MIR_TargetOp<"rdsvl", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RDSVLI_XI
}

def AArch64MIR_rdvl : AArch64MIR_TargetOp<"rdvl", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RDVLI_XI
}

def AArch64MIR_ret : AArch64MIR_TargetOp<"ret", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RET
}

def AArch64MIR_retaa : AArch64MIR_TargetOp<"retaa", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RETAA
}

def AArch64MIR_retaasppc : AArch64MIR_TargetOp<"retaasppc", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RETAASPPCi
}

def AArch64MIR_retaasppcr : AArch64MIR_TargetOp<"retaasppcr", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RETAASPPCr
}

def AArch64MIR_retab : AArch64MIR_TargetOp<"retab", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RETAB
}

def AArch64MIR_retabsppc : AArch64MIR_TargetOp<"retabsppc", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RETABSPPCi
}

def AArch64MIR_retabsppcr : AArch64MIR_TargetOp<"retabsppcr", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RETABSPPCr
}

def AArch64MIR_rev16 : AArch64MIR_TargetOp<"rev16", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: REV16Wr REV16Xr REV16v8i8 REV16v16i8
}

def AArch64MIR_rev32 : AArch64MIR_TargetOp<"rev32", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: REV32Xr REV32v4i16 REV32v8i8 REV32v8i16 REV32v16i8
}

def AArch64MIR_rev64 : AArch64MIR_TargetOp<"rev64", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: REV64v2i32 REV64v4i16 REV64v4i32 REV64v8i8 REV64v8i16 REV64v16i8
}

def AArch64MIR_revb : AArch64MIR_TargetOp<"revb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: REVB_ZPmZ_D REVB_ZPmZ_H REVB_ZPmZ_S REVB_ZPzZ_D REVB_ZPzZ_H REVB_ZPzZ_S
}

def AArch64MIR_revd : AArch64MIR_TargetOp<"revd", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: REVD_ZPmZ REVD_ZPzZ
}

def AArch64MIR_revh : AArch64MIR_TargetOp<"revh", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: REVH_ZPmZ_D REVH_ZPmZ_S REVH_ZPzZ_D REVH_ZPzZ_S
}

def AArch64MIR_revw : AArch64MIR_TargetOp<"revw", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: REVW_ZPmZ_D REVW_ZPzZ_D
}

def AArch64MIR_rev : AArch64MIR_TargetOp<"rev", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: REVWr REVXr REV_PP_B REV_PP_D REV_PP_H REV_PP_S REV_ZZ_B REV_ZZ_D REV_ZZ_H REV_ZZ_S
}

def AArch64MIR_rmif : AArch64MIR_TargetOp<"rmif", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RMIF
}

def AArch64MIR_ror : AArch64MIR_TargetOp<"ror", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RORVWr RORVXr
}

def AArch64MIR_rprfm : AArch64MIR_TargetOp<"rprfm", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RPRFM
}

def AArch64MIR_rshrnb : AArch64MIR_TargetOp<"rshrnb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RSHRNB_ZZI_B RSHRNB_ZZI_H RSHRNB_ZZI_S
}

def AArch64MIR_rshrnt : AArch64MIR_TargetOp<"rshrnt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RSHRNT_ZZI_B RSHRNT_ZZI_H RSHRNT_ZZI_S
}

def AArch64MIR_rshrn : AArch64MIR_TargetOp<"rshrn", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RSHRNv2i32_shift RSHRNv4i16_shift RSHRNv8i8_shift
}

def AArch64MIR_rshrn2 : AArch64MIR_TargetOp<"rshrn2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RSHRNv4i32_shift RSHRNv8i16_shift RSHRNv16i8_shift
}

def AArch64MIR_rsubhnb : AArch64MIR_TargetOp<"rsubhnb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RSUBHNB_ZZZ_B RSUBHNB_ZZZ_H RSUBHNB_ZZZ_S
}

def AArch64MIR_rsubhnt : AArch64MIR_TargetOp<"rsubhnt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RSUBHNT_ZZZ_B RSUBHNT_ZZZ_H RSUBHNT_ZZZ_S
}

def AArch64MIR_rsubhn : AArch64MIR_TargetOp<"rsubhn", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RSUBHNv2i64_v2i32 RSUBHNv4i32_v4i16 RSUBHNv8i16_v8i8
}

def AArch64MIR_rsubhn2 : AArch64MIR_TargetOp<"rsubhn2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: RSUBHNv2i64_v4i32 RSUBHNv4i32_v8i16 RSUBHNv8i16_v16i8
}

def AArch64MIR_sabalb : AArch64MIR_TargetOp<"sabalb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SABALB_ZZZ_D SABALB_ZZZ_H SABALB_ZZZ_S
}

def AArch64MIR_sabalt : AArch64MIR_TargetOp<"sabalt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SABALT_ZZZ_D SABALT_ZZZ_H SABALT_ZZZ_S
}

def AArch64MIR_sabal : AArch64MIR_TargetOp<"sabal", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SABAL_ZZZ_BtoH SABAL_ZZZ_HtoS SABAL_ZZZ_StoD SABALv2i32_v2i64 SABALv4i16_v4i32 SABALv8i8_v8i16
}

def AArch64MIR_sabal2 : AArch64MIR_TargetOp<"sabal2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SABALv4i32_v2i64 SABALv8i16_v4i32 SABALv16i8_v8i16
}

def AArch64MIR_saba : AArch64MIR_TargetOp<"saba", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SABA_ZZZ_B SABA_ZZZ_D SABA_ZZZ_H SABA_ZZZ_S SABAv2i32 SABAv4i16 SABAv4i32 SABAv8i8 SABAv8i16 SABAv16i8
}

def AArch64MIR_sabdlb : AArch64MIR_TargetOp<"sabdlb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SABDLB_ZZZ_D SABDLB_ZZZ_H SABDLB_ZZZ_S
}

def AArch64MIR_sabdlt : AArch64MIR_TargetOp<"sabdlt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SABDLT_ZZZ_D SABDLT_ZZZ_H SABDLT_ZZZ_S
}

def AArch64MIR_sabdl : AArch64MIR_TargetOp<"sabdl", [Pure, Commutative]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SABDLv2i32_v2i64 SABDLv4i16_v4i32 SABDLv8i8_v8i16
}

def AArch64MIR_sabdl2 : AArch64MIR_TargetOp<"sabdl2", [Pure, Commutative]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SABDLv4i32_v2i64 SABDLv8i16_v4i32 SABDLv16i8_v8i16
}

def AArch64MIR_sabd : AArch64MIR_TargetOp<"sabd", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SABD_ZPmZ_B SABD_ZPmZ_D SABD_ZPmZ_H SABD_ZPmZ_S SABDv2i32 SABDv4i16 SABDv4i32 SABDv8i8 SABDv8i16 SABDv16i8
}

def AArch64MIR_sadalp : AArch64MIR_TargetOp<"sadalp", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SADALP_ZPmZ_D SADALP_ZPmZ_H SADALP_ZPmZ_S SADALPv2i32_v1i64 SADALPv4i16_v2i32 SADALPv4i32_v2i64 SADALPv8i8_v4i16 SADALPv8i16_v4i32 SADALPv16i8_v8i16
}

def AArch64MIR_saddlbt : AArch64MIR_TargetOp<"saddlbt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SADDLBT_ZZZ_D SADDLBT_ZZZ_H SADDLBT_ZZZ_S
}

def AArch64MIR_saddlb : AArch64MIR_TargetOp<"saddlb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SADDLB_ZZZ_D SADDLB_ZZZ_H SADDLB_ZZZ_S
}

def AArch64MIR_saddlp : AArch64MIR_TargetOp<"saddlp", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SADDLPv2i32_v1i64 SADDLPv4i16_v2i32 SADDLPv4i32_v2i64 SADDLPv8i8_v4i16 SADDLPv8i16_v4i32 SADDLPv16i8_v8i16
}

def AArch64MIR_saddlt : AArch64MIR_TargetOp<"saddlt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SADDLT_ZZZ_D SADDLT_ZZZ_H SADDLT_ZZZ_S
}

def AArch64MIR_saddlv : AArch64MIR_TargetOp<"saddlv", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SADDLVv4i16v SADDLVv4i32v SADDLVv8i8v SADDLVv8i16v SADDLVv16i8v
}

def AArch64MIR_saddl : AArch64MIR_TargetOp<"saddl", [Pure, Commutative]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SADDLv2i32_v2i64 SADDLv4i16_v4i32 SADDLv8i8_v8i16
}

def AArch64MIR_saddl2 : AArch64MIR_TargetOp<"saddl2", [Pure, Commutative]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SADDLv4i32_v2i64 SADDLv8i16_v4i32 SADDLv16i8_v8i16
}

def AArch64MIR_saddv : AArch64MIR_TargetOp<"saddv", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SADDV_VPZ_B SADDV_VPZ_H SADDV_VPZ_S
}

def AArch64MIR_saddwb : AArch64MIR_TargetOp<"saddwb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SADDWB_ZZZ_D SADDWB_ZZZ_H SADDWB_ZZZ_S
}

def AArch64MIR_saddwt : AArch64MIR_TargetOp<"saddwt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SADDWT_ZZZ_D SADDWT_ZZZ_H SADDWT_ZZZ_S
}

def AArch64MIR_saddw : AArch64MIR_TargetOp<"saddw", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SADDWv2i32_v2i64 SADDWv4i16_v4i32 SADDWv8i8_v8i16
}

def AArch64MIR_saddw2 : AArch64MIR_TargetOp<"saddw2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SADDWv4i32_v2i64 SADDWv8i16_v4i32 SADDWv16i8_v8i16
}

def AArch64MIR_sb : AArch64MIR_TargetOp<"sb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SB
}

def AArch64MIR_sbclb : AArch64MIR_TargetOp<"sbclb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SBCLB_ZZZ_D SBCLB_ZZZ_S
}

def AArch64MIR_sbclt : AArch64MIR_TargetOp<"sbclt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SBCLT_ZZZ_D SBCLT_ZZZ_S
}

def AArch64MIR_sbcs : AArch64MIR_TargetOp<"sbcs", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SBCSWr SBCSXr
}

def AArch64MIR_sbc : AArch64MIR_TargetOp<"sbc", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SBCWr SBCXr
}

def AArch64MIR_sbfm : AArch64MIR_TargetOp<"sbfm", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SBFMWri SBFMXri
}

def AArch64MIR_sclamp : AArch64MIR_TargetOp<"sclamp", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SCLAMP_VG2_2Z2Z_B SCLAMP_VG2_2Z2Z_D SCLAMP_VG2_2Z2Z_H SCLAMP_VG2_2Z2Z_S SCLAMP_VG4_4Z4Z_B SCLAMP_VG4_4Z4Z_D SCLAMP_VG4_4Z4Z_H SCLAMP_VG4_4Z4Z_S SCLAMP_ZZZ_B SCLAMP_ZZZ_D SCLAMP_ZZZ_H SCLAMP_ZZZ_S
}

def AArch64MIR_scvtf : AArch64MIR_TargetOp<"scvtf", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SCVTFDSr SCVTFHDr SCVTFHSr SCVTFSDr SCVTFSWDri SCVTFSWHri SCVTFSWSri SCVTFSXDri SCVTFSXHri SCVTFSXSri SCVTFUWDri SCVTFUWHri SCVTFUWSri SCVTFUXDri SCVTFUXHri SCVTFUXSri SCVTF_2Z2Z_StoS SCVTF_4Z4Z_StoS SCVTF_ZPmZ_DtoD SCVTF_ZPmZ_DtoH SCVTF_ZPmZ_DtoS SCVTF_ZPmZ_HtoH SCVTF_ZPmZ_StoD SCVTF_ZPmZ_StoH SCVTF_ZPmZ_StoS SCVTF_ZPzZ_DtoD SCVTF_ZPzZ_DtoH SCVTF_ZPzZ_DtoS SCVTF_ZPzZ_HtoH SCVTF_ZPzZ_StoD SCVTF_ZPzZ_StoH SCVTF_ZPzZ_StoS SCVTF_ZZ_BtoH SCVTF_ZZ_HtoS SCVTF_ZZ_StoD SCVTFd SCVTFh SCVTFs SCVTFv1i16 SCVTFv1i32 SCVTFv1i64 SCVTFv2f32 SCVTFv2f64 SCVTFv2i32_shift SCVTFv2i64_shift SCVTFv4f16 SCVTFv4f32 SCVTFv4i16_shift SCVTFv4i32_shift SCVTFv8f16 SCVTFv8i16_shift
}

def AArch64MIR_scvtflt : AArch64MIR_TargetOp<"scvtflt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SCVTFLT_ZZ_BtoH SCVTFLT_ZZ_HtoS SCVTFLT_ZZ_StoD
}

def AArch64MIR_sdivr : AArch64MIR_TargetOp<"sdivr", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SDIVR_ZPmZ_D SDIVR_ZPmZ_S
}

def AArch64MIR_sdiv : AArch64MIR_TargetOp<"sdiv", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SDIVWr SDIVXr SDIV_ZPmZ_D SDIV_ZPmZ_S
}

def AArch64MIR_sdot : AArch64MIR_TargetOp<"sdot", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SDOT_VG2_M2Z2Z_BtoS SDOT_VG2_M2Z2Z_HtoD SDOT_VG2_M2Z2Z_HtoS SDOT_VG2_M2ZZI_BToS SDOT_VG2_M2ZZI_HToS SDOT_VG2_M2ZZI_HtoD SDOT_VG2_M2ZZ_BtoS SDOT_VG2_M2ZZ_HtoD SDOT_VG2_M2ZZ_HtoS SDOT_VG4_M4Z4Z_BtoS SDOT_VG4_M4Z4Z_HtoD SDOT_VG4_M4Z4Z_HtoS SDOT_VG4_M4ZZI_BToS SDOT_VG4_M4ZZI_HToS SDOT_VG4_M4ZZI_HtoD SDOT_VG4_M4ZZ_BtoS SDOT_VG4_M4ZZ_HtoD SDOT_VG4_M4ZZ_HtoS SDOT_ZZZI_BtoH SDOT_ZZZI_BtoS SDOT_ZZZI_HtoD SDOT_ZZZI_HtoS SDOT_ZZZ_BtoH SDOT_ZZZ_BtoS SDOT_ZZZ_HtoD SDOT_ZZZ_HtoS SDOTlanev8i8 SDOTlanev16i8 SDOTv8i8 SDOTv16i8
}

def AArch64MIR_sel : AArch64MIR_TargetOp<"sel", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SEL_PPPP SEL_VG2_2ZC2Z2Z_B SEL_VG2_2ZC2Z2Z_D SEL_VG2_2ZC2Z2Z_H SEL_VG2_2ZC2Z2Z_S SEL_VG4_4ZC4Z4Z_B SEL_VG4_4ZC4Z4Z_D SEL_VG4_4ZC4Z4Z_H SEL_VG4_4ZC4Z4Z_S SEL_ZPZZ_B SEL_ZPZZ_D SEL_ZPZZ_H SEL_ZPZZ_S
}

def AArch64MIR_sete : AArch64MIR_TargetOp<"sete", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SETE
}

def AArch64MIR_seten : AArch64MIR_TargetOp<"seten", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SETEN
}

def AArch64MIR_setet : AArch64MIR_TargetOp<"setet", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SETET
}

def AArch64MIR_setetn : AArch64MIR_TargetOp<"setetn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SETETN
}

def AArch64MIR_setf8 : AArch64MIR_TargetOp<"setf8", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SETF8
}

def AArch64MIR_setf16 : AArch64MIR_TargetOp<"setf16", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SETF16
}

def AArch64MIR_setffr : AArch64MIR_TargetOp<"setffr", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SETFFR
}

def AArch64MIR_setgm : AArch64MIR_TargetOp<"setgm", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SETGM
}

def AArch64MIR_setgmn : AArch64MIR_TargetOp<"setgmn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SETGMN
}

def AArch64MIR_setgmt : AArch64MIR_TargetOp<"setgmt", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SETGMT
}

def AArch64MIR_setgmtn : AArch64MIR_TargetOp<"setgmtn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SETGMTN
}

def AArch64MIR_setgoe : AArch64MIR_TargetOp<"setgoe", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SETGOE
}

def AArch64MIR_setgoen : AArch64MIR_TargetOp<"setgoen", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SETGOEN
}

def AArch64MIR_setgoet : AArch64MIR_TargetOp<"setgoet", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SETGOET
}

def AArch64MIR_setgoetn : AArch64MIR_TargetOp<"setgoetn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SETGOETN
}

def AArch64MIR_setgom : AArch64MIR_TargetOp<"setgom", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SETGOM
}

def AArch64MIR_setgomn : AArch64MIR_TargetOp<"setgomn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SETGOMN
}

def AArch64MIR_setgomt : AArch64MIR_TargetOp<"setgomt", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SETGOMT
}

def AArch64MIR_setgomtn : AArch64MIR_TargetOp<"setgomtn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SETGOMTN
}

def AArch64MIR_setgop : AArch64MIR_TargetOp<"setgop", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SETGOP
}

def AArch64MIR_setgopn : AArch64MIR_TargetOp<"setgopn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SETGOPN
}

def AArch64MIR_setgopt : AArch64MIR_TargetOp<"setgopt", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SETGOPT
}

def AArch64MIR_setgoptn : AArch64MIR_TargetOp<"setgoptn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SETGOPTN
}

def AArch64MIR_setgp : AArch64MIR_TargetOp<"setgp", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SETGP
}

def AArch64MIR_setgpn : AArch64MIR_TargetOp<"setgpn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SETGPN
}

def AArch64MIR_setgpt : AArch64MIR_TargetOp<"setgpt", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SETGPT
}

def AArch64MIR_setgptn : AArch64MIR_TargetOp<"setgptn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SETGPTN
}

def AArch64MIR_setm : AArch64MIR_TargetOp<"setm", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SETM
}

def AArch64MIR_setmn : AArch64MIR_TargetOp<"setmn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SETMN
}

def AArch64MIR_setmt : AArch64MIR_TargetOp<"setmt", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SETMT
}

def AArch64MIR_setmtn : AArch64MIR_TargetOp<"setmtn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SETMTN
}

def AArch64MIR_setp : AArch64MIR_TargetOp<"setp", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SETP
}

def AArch64MIR_setpn : AArch64MIR_TargetOp<"setpn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SETPN
}

def AArch64MIR_setpt : AArch64MIR_TargetOp<"setpt", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SETPT
}

def AArch64MIR_setptn : AArch64MIR_TargetOp<"setptn", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SETPTN
}

def AArch64MIR_sha1c : AArch64MIR_TargetOp<"sha1c", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SHA1Crrr
}

def AArch64MIR_sha1h : AArch64MIR_TargetOp<"sha1h", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SHA1Hrr
}

def AArch64MIR_sha1m : AArch64MIR_TargetOp<"sha1m", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SHA1Mrrr
}

def AArch64MIR_sha1p : AArch64MIR_TargetOp<"sha1p", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SHA1Prrr
}

def AArch64MIR_sha1su0 : AArch64MIR_TargetOp<"sha1su0", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SHA1SU0rrr
}

def AArch64MIR_sha1su1 : AArch64MIR_TargetOp<"sha1su1", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SHA1SU1rr
}

def AArch64MIR_sha256h2 : AArch64MIR_TargetOp<"sha256h2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SHA256H2rrr
}

def AArch64MIR_sha256h : AArch64MIR_TargetOp<"sha256h", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SHA256Hrrr
}

def AArch64MIR_sha256su0 : AArch64MIR_TargetOp<"sha256su0", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SHA256SU0rr
}

def AArch64MIR_sha256su1 : AArch64MIR_TargetOp<"sha256su1", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SHA256SU1rrr
}

def AArch64MIR_sha512h : AArch64MIR_TargetOp<"sha512h", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SHA512H
}

def AArch64MIR_sha512h2 : AArch64MIR_TargetOp<"sha512h2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SHA512H2
}

def AArch64MIR_sha512su0 : AArch64MIR_TargetOp<"sha512su0", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SHA512SU0
}

def AArch64MIR_sha512su1 : AArch64MIR_TargetOp<"sha512su1", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SHA512SU1
}

def AArch64MIR_shadd : AArch64MIR_TargetOp<"shadd", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SHADD_ZPmZ_B SHADD_ZPmZ_D SHADD_ZPmZ_H SHADD_ZPmZ_S SHADDv2i32 SHADDv4i16 SHADDv4i32 SHADDv8i8 SHADDv8i16 SHADDv16i8
}

def AArch64MIR_shll : AArch64MIR_TargetOp<"shll", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SHLLv2i32 SHLLv4i16 SHLLv8i8
}

def AArch64MIR_shll2 : AArch64MIR_TargetOp<"shll2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SHLLv4i32 SHLLv8i16 SHLLv16i8
}

def AArch64MIR_shl : AArch64MIR_TargetOp<"shl", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SHLd SHLv2i32_shift SHLv2i64_shift SHLv4i16_shift SHLv4i32_shift SHLv8i8_shift SHLv8i16_shift SHLv16i8_shift
}

def AArch64MIR_shrnb : AArch64MIR_TargetOp<"shrnb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SHRNB_ZZI_B SHRNB_ZZI_H SHRNB_ZZI_S
}

def AArch64MIR_shrnt : AArch64MIR_TargetOp<"shrnt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SHRNT_ZZI_B SHRNT_ZZI_H SHRNT_ZZI_S
}

def AArch64MIR_shrn : AArch64MIR_TargetOp<"shrn", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SHRNv2i32_shift SHRNv4i16_shift SHRNv8i8_shift
}

def AArch64MIR_shrn2 : AArch64MIR_TargetOp<"shrn2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SHRNv4i32_shift SHRNv8i16_shift SHRNv16i8_shift
}

def AArch64MIR_shsubr : AArch64MIR_TargetOp<"shsubr", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SHSUBR_ZPmZ_B SHSUBR_ZPmZ_D SHSUBR_ZPmZ_H SHSUBR_ZPmZ_S
}

def AArch64MIR_shsub : AArch64MIR_TargetOp<"shsub", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SHSUB_ZPmZ_B SHSUB_ZPmZ_D SHSUB_ZPmZ_H SHSUB_ZPmZ_S SHSUBv2i32 SHSUBv4i16 SHSUBv4i32 SHSUBv8i8 SHSUBv8i16 SHSUBv16i8
}

def AArch64MIR_sli : AArch64MIR_TargetOp<"sli", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SLI_ZZI_B SLI_ZZI_D SLI_ZZI_H SLI_ZZI_S SLId SLIv2i32_shift SLIv2i64_shift SLIv4i16_shift SLIv4i32_shift SLIv8i8_shift SLIv8i16_shift SLIv16i8_shift
}

def AArch64MIR_sm3partw1 : AArch64MIR_TargetOp<"sm3partw1", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SM3PARTW1
}

def AArch64MIR_sm3partw2 : AArch64MIR_TargetOp<"sm3partw2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SM3PARTW2
}

def AArch64MIR_sm3ss1 : AArch64MIR_TargetOp<"sm3ss1", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SM3SS1
}

def AArch64MIR_sm3tt1a : AArch64MIR_TargetOp<"sm3tt1a", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SM3TT1A
}

def AArch64MIR_sm3tt1b : AArch64MIR_TargetOp<"sm3tt1b", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SM3TT1B
}

def AArch64MIR_sm3tt2a : AArch64MIR_TargetOp<"sm3tt2a", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SM3TT2A
}

def AArch64MIR_sm3tt2b : AArch64MIR_TargetOp<"sm3tt2b", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SM3TT2B
}

def AArch64MIR_sm4e : AArch64MIR_TargetOp<"sm4e", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SM4E SM4E_ZZZ_S
}

def AArch64MIR_sm4ekey : AArch64MIR_TargetOp<"sm4ekey", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SM4EKEY_ZZZ_S SM4ENCKEY
}

def AArch64MIR_smaddl : AArch64MIR_TargetOp<"smaddl", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SMADDLrrr
}

def AArch64MIR_smaxp : AArch64MIR_TargetOp<"smaxp", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SMAXP_ZPmZ_B SMAXP_ZPmZ_D SMAXP_ZPmZ_H SMAXP_ZPmZ_S SMAXPv2i32 SMAXPv4i16 SMAXPv4i32 SMAXPv8i8 SMAXPv8i16 SMAXPv16i8
}

def AArch64MIR_smaxqv : AArch64MIR_TargetOp<"smaxqv", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SMAXQV_VPZ_B SMAXQV_VPZ_D SMAXQV_VPZ_H SMAXQV_VPZ_S
}

def AArch64MIR_smaxv : AArch64MIR_TargetOp<"smaxv", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SMAXV_VPZ_B SMAXV_VPZ_D SMAXV_VPZ_H SMAXV_VPZ_S SMAXVv4i16v SMAXVv4i32v SMAXVv8i8v SMAXVv8i16v SMAXVv16i8v
}

def AArch64MIR_smax : AArch64MIR_TargetOp<"smax", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SMAXWri SMAXWrr SMAXXri SMAXXrr SMAX_VG2_2Z2Z_B SMAX_VG2_2Z2Z_D SMAX_VG2_2Z2Z_H SMAX_VG2_2Z2Z_S SMAX_VG2_2ZZ_B SMAX_VG2_2ZZ_D SMAX_VG2_2ZZ_H SMAX_VG2_2ZZ_S SMAX_VG4_4Z4Z_B SMAX_VG4_4Z4Z_D SMAX_VG4_4Z4Z_H SMAX_VG4_4Z4Z_S SMAX_VG4_4ZZ_B SMAX_VG4_4ZZ_D SMAX_VG4_4ZZ_H SMAX_VG4_4ZZ_S SMAX_ZI_B SMAX_ZI_D SMAX_ZI_H SMAX_ZI_S SMAX_ZPmZ_B SMAX_ZPmZ_D SMAX_ZPmZ_H SMAX_ZPmZ_S SMAXv2i32 SMAXv4i16 SMAXv4i32 SMAXv8i8 SMAXv8i16 SMAXv16i8
}

def AArch64MIR_smc : AArch64MIR_TargetOp<"smc", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SMC
}

def AArch64MIR_sminp : AArch64MIR_TargetOp<"sminp", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SMINP_ZPmZ_B SMINP_ZPmZ_D SMINP_ZPmZ_H SMINP_ZPmZ_S SMINPv2i32 SMINPv4i16 SMINPv4i32 SMINPv8i8 SMINPv8i16 SMINPv16i8
}

def AArch64MIR_sminqv : AArch64MIR_TargetOp<"sminqv", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SMINQV_VPZ_B SMINQV_VPZ_D SMINQV_VPZ_H SMINQV_VPZ_S
}

def AArch64MIR_sminv : AArch64MIR_TargetOp<"sminv", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SMINV_VPZ_B SMINV_VPZ_D SMINV_VPZ_H SMINV_VPZ_S SMINVv4i16v SMINVv4i32v SMINVv8i8v SMINVv8i16v SMINVv16i8v
}

def AArch64MIR_smin : AArch64MIR_TargetOp<"smin", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SMINWri SMINWrr SMINXri SMINXrr SMIN_VG2_2Z2Z_B SMIN_VG2_2Z2Z_D SMIN_VG2_2Z2Z_H SMIN_VG2_2Z2Z_S SMIN_VG2_2ZZ_B SMIN_VG2_2ZZ_D SMIN_VG2_2ZZ_H SMIN_VG2_2ZZ_S SMIN_VG4_4Z4Z_B SMIN_VG4_4Z4Z_D SMIN_VG4_4Z4Z_H SMIN_VG4_4Z4Z_S SMIN_VG4_4ZZ_B SMIN_VG4_4ZZ_D SMIN_VG4_4ZZ_H SMIN_VG4_4ZZ_S SMIN_ZI_B SMIN_ZI_D SMIN_ZI_H SMIN_ZI_S SMIN_ZPmZ_B SMIN_ZPmZ_D SMIN_ZPmZ_H SMIN_ZPmZ_S SMINv2i32 SMINv4i16 SMINv4i32 SMINv8i8 SMINv8i16 SMINv16i8
}

def AArch64MIR_smlalb : AArch64MIR_TargetOp<"smlalb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SMLALB_ZZZI_D SMLALB_ZZZI_S SMLALB_ZZZ_D SMLALB_ZZZ_H SMLALB_ZZZ_S
}

def AArch64MIR_smlall : AArch64MIR_TargetOp<"smlall", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SMLALL_MZZI_BtoS SMLALL_MZZI_HtoD SMLALL_MZZ_BtoS SMLALL_MZZ_HtoD SMLALL_VG2_M2Z2Z_BtoS SMLALL_VG2_M2Z2Z_HtoD SMLALL_VG2_M2ZZI_BtoS SMLALL_VG2_M2ZZI_HtoD SMLALL_VG2_M2ZZ_BtoS SMLALL_VG2_M2ZZ_HtoD SMLALL_VG4_M4Z4Z_BtoS SMLALL_VG4_M4Z4Z_HtoD SMLALL_VG4_M4ZZI_BtoS SMLALL_VG4_M4ZZI_HtoD SMLALL_VG4_M4ZZ_BtoS SMLALL_VG4_M4ZZ_HtoD
}

def AArch64MIR_smlalt : AArch64MIR_TargetOp<"smlalt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SMLALT_ZZZI_D SMLALT_ZZZI_S SMLALT_ZZZ_D SMLALT_ZZZ_H SMLALT_ZZZ_S
}

def AArch64MIR_smlal : AArch64MIR_TargetOp<"smlal", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SMLAL_MZZI_HtoS SMLAL_MZZ_HtoS SMLAL_VG2_M2Z2Z_HtoS SMLAL_VG2_M2ZZI_S SMLAL_VG2_M2ZZ_HtoS SMLAL_VG4_M4Z4Z_HtoS SMLAL_VG4_M4ZZI_HtoS SMLAL_VG4_M4ZZ_HtoS SMLALv2i32_indexed SMLALv2i32_v2i64 SMLALv4i16_indexed SMLALv4i16_v4i32 SMLALv8i8_v8i16
}

def AArch64MIR_smlal2 : AArch64MIR_TargetOp<"smlal2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SMLALv4i32_indexed SMLALv4i32_v2i64 SMLALv8i16_indexed SMLALv8i16_v4i32 SMLALv16i8_v8i16
}

def AArch64MIR_smlslb : AArch64MIR_TargetOp<"smlslb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SMLSLB_ZZZI_D SMLSLB_ZZZI_S SMLSLB_ZZZ_D SMLSLB_ZZZ_H SMLSLB_ZZZ_S
}

def AArch64MIR_smlsll : AArch64MIR_TargetOp<"smlsll", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SMLSLL_MZZI_BtoS SMLSLL_MZZI_HtoD SMLSLL_MZZ_BtoS SMLSLL_MZZ_HtoD SMLSLL_VG2_M2Z2Z_BtoS SMLSLL_VG2_M2Z2Z_HtoD SMLSLL_VG2_M2ZZI_BtoS SMLSLL_VG2_M2ZZI_HtoD SMLSLL_VG2_M2ZZ_BtoS SMLSLL_VG2_M2ZZ_HtoD SMLSLL_VG4_M4Z4Z_BtoS SMLSLL_VG4_M4Z4Z_HtoD SMLSLL_VG4_M4ZZI_BtoS SMLSLL_VG4_M4ZZI_HtoD SMLSLL_VG4_M4ZZ_BtoS SMLSLL_VG4_M4ZZ_HtoD
}

def AArch64MIR_smlslt : AArch64MIR_TargetOp<"smlslt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SMLSLT_ZZZI_D SMLSLT_ZZZI_S SMLSLT_ZZZ_D SMLSLT_ZZZ_H SMLSLT_ZZZ_S
}

def AArch64MIR_smlsl : AArch64MIR_TargetOp<"smlsl", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SMLSL_MZZI_HtoS SMLSL_MZZ_HtoS SMLSL_VG2_M2Z2Z_HtoS SMLSL_VG2_M2ZZI_S SMLSL_VG2_M2ZZ_HtoS SMLSL_VG4_M4Z4Z_HtoS SMLSL_VG4_M4ZZI_HtoS SMLSL_VG4_M4ZZ_HtoS SMLSLv2i32_indexed SMLSLv2i32_v2i64 SMLSLv4i16_indexed SMLSLv4i16_v4i32 SMLSLv8i8_v8i16
}

def AArch64MIR_smlsl2 : AArch64MIR_TargetOp<"smlsl2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SMLSLv4i32_indexed SMLSLv4i32_v2i64 SMLSLv8i16_indexed SMLSLv8i16_v4i32 SMLSLv16i8_v8i16
}

def AArch64MIR_smmla : AArch64MIR_TargetOp<"smmla", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SMMLA SMMLA_ZZZ
}

def AArch64MIR_smop4a : AArch64MIR_TargetOp<"smop4a", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SMOP4A_M2Z2Z_BToS SMOP4A_M2Z2Z_HToS SMOP4A_M2Z2Z_HtoD SMOP4A_M2ZZ_BToS SMOP4A_M2ZZ_HToS SMOP4A_M2ZZ_HtoD SMOP4A_MZ2Z_BToS SMOP4A_MZ2Z_HToS SMOP4A_MZ2Z_HtoD SMOP4A_MZZ_BToS SMOP4A_MZZ_HToS SMOP4A_MZZ_HtoD
}

def AArch64MIR_smop4s : AArch64MIR_TargetOp<"smop4s", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SMOP4S_M2Z2Z_BToS SMOP4S_M2Z2Z_HToS SMOP4S_M2Z2Z_HtoD SMOP4S_M2ZZ_BToS SMOP4S_M2ZZ_HToS SMOP4S_M2ZZ_HtoD SMOP4S_MZ2Z_BToS SMOP4S_MZ2Z_HToS SMOP4S_MZ2Z_HtoD SMOP4S_MZZ_BToS SMOP4S_MZZ_HToS SMOP4S_MZZ_HtoD
}

def AArch64MIR_smopa : AArch64MIR_TargetOp<"smopa", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SMOPA_MPPZZ_D SMOPA_MPPZZ_HtoS SMOPA_MPPZZ_S
}

def AArch64MIR_smops : AArch64MIR_TargetOp<"smops", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SMOPS_MPPZZ_D SMOPS_MPPZZ_HtoS SMOPS_MPPZZ_S
}

def AArch64MIR_smov : AArch64MIR_TargetOp<"smov", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SMOVvi8to32 SMOVvi8to32_idx0 SMOVvi8to64 SMOVvi8to64_idx0 SMOVvi16to32 SMOVvi16to32_idx0 SMOVvi16to64 SMOVvi16to64_idx0 SMOVvi32to64 SMOVvi32to64_idx0
}

def AArch64MIR_smsubl : AArch64MIR_TargetOp<"smsubl", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SMSUBLrrr
}

def AArch64MIR_smulh : AArch64MIR_TargetOp<"smulh", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SMULH_ZPmZ_B SMULH_ZPmZ_D SMULH_ZPmZ_H SMULH_ZPmZ_S SMULH_ZZZ_B SMULH_ZZZ_D SMULH_ZZZ_H SMULH_ZZZ_S SMULHrr
}

def AArch64MIR_smullb : AArch64MIR_TargetOp<"smullb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SMULLB_ZZZI_D SMULLB_ZZZI_S SMULLB_ZZZ_D SMULLB_ZZZ_H SMULLB_ZZZ_S
}

def AArch64MIR_smullt : AArch64MIR_TargetOp<"smullt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SMULLT_ZZZI_D SMULLT_ZZZI_S SMULLT_ZZZ_D SMULLT_ZZZ_H SMULLT_ZZZ_S
}

def AArch64MIR_smull : AArch64MIR_TargetOp<"smull", [Pure, Commutative]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SMULLv2i32_indexed SMULLv2i32_v2i64 SMULLv4i16_indexed SMULLv4i16_v4i32 SMULLv8i8_v8i16
}

def AArch64MIR_smull2 : AArch64MIR_TargetOp<"smull2", [Pure, Commutative]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SMULLv4i32_indexed SMULLv4i32_v2i64 SMULLv8i16_indexed SMULLv8i16_v4i32 SMULLv16i8_v8i16
}

def AArch64MIR_splice : AArch64MIR_TargetOp<"splice", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SPLICE_ZPZZ_B SPLICE_ZPZZ_D SPLICE_ZPZZ_H SPLICE_ZPZZ_S SPLICE_ZPZ_B SPLICE_ZPZ_D SPLICE_ZPZ_H SPLICE_ZPZ_S
}

def AArch64MIR_sqabs : AArch64MIR_TargetOp<"sqabs", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQABS_ZPmZ_B SQABS_ZPmZ_D SQABS_ZPmZ_H SQABS_ZPmZ_S SQABS_ZPzZ_B SQABS_ZPzZ_D SQABS_ZPzZ_H SQABS_ZPzZ_S SQABSv1i8 SQABSv1i16 SQABSv1i32 SQABSv1i64 SQABSv2i32 SQABSv2i64 SQABSv4i16 SQABSv4i32 SQABSv8i8 SQABSv8i16 SQABSv16i8
}

def AArch64MIR_sqadd : AArch64MIR_TargetOp<"sqadd", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQADD_ZI_B SQADD_ZI_D SQADD_ZI_H SQADD_ZI_S SQADD_ZPmZ_B SQADD_ZPmZ_D SQADD_ZPmZ_H SQADD_ZPmZ_S SQADD_ZZZ_B SQADD_ZZZ_D SQADD_ZZZ_H SQADD_ZZZ_S SQADDv1i8 SQADDv1i16 SQADDv1i32 SQADDv1i64 SQADDv2i32 SQADDv2i64 SQADDv4i16 SQADDv4i32 SQADDv8i8 SQADDv8i16 SQADDv16i8
}

def AArch64MIR_sqcadd : AArch64MIR_TargetOp<"sqcadd", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQCADD_ZZI_B SQCADD_ZZI_D SQCADD_ZZI_H SQCADD_ZZI_S
}

def AArch64MIR_sqcvtn : AArch64MIR_TargetOp<"sqcvtn", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQCVTN_Z2Z_StoH SQCVTN_Z4Z_DtoH SQCVTN_Z4Z_StoB
}

def AArch64MIR_sqcvtun : AArch64MIR_TargetOp<"sqcvtun", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQCVTUN_Z2Z_StoH SQCVTUN_Z4Z_DtoH SQCVTUN_Z4Z_StoB
}

def AArch64MIR_sqcvtu : AArch64MIR_TargetOp<"sqcvtu", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQCVTU_Z2Z_StoH SQCVTU_Z4Z_DtoH SQCVTU_Z4Z_StoB
}

def AArch64MIR_sqcvt : AArch64MIR_TargetOp<"sqcvt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQCVT_Z2Z_StoH SQCVT_Z4Z_DtoH SQCVT_Z4Z_StoB
}

def AArch64MIR_sqdecb : AArch64MIR_TargetOp<"sqdecb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQDECB_XPiI SQDECB_XPiWdI
}

def AArch64MIR_sqdecd : AArch64MIR_TargetOp<"sqdecd", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQDECD_XPiI SQDECD_XPiWdI SQDECD_ZPiI
}

def AArch64MIR_sqdech : AArch64MIR_TargetOp<"sqdech", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQDECH_XPiI SQDECH_XPiWdI SQDECH_ZPiI
}

def AArch64MIR_sqdecp : AArch64MIR_TargetOp<"sqdecp", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQDECP_XPWd_B SQDECP_XPWd_D SQDECP_XPWd_H SQDECP_XPWd_S SQDECP_XP_B SQDECP_XP_D SQDECP_XP_H SQDECP_XP_S SQDECP_ZP_D SQDECP_ZP_H SQDECP_ZP_S
}

def AArch64MIR_sqdecw : AArch64MIR_TargetOp<"sqdecw", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQDECW_XPiI SQDECW_XPiWdI SQDECW_ZPiI
}

def AArch64MIR_sqdmlalbt : AArch64MIR_TargetOp<"sqdmlalbt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQDMLALBT_ZZZ_D SQDMLALBT_ZZZ_H SQDMLALBT_ZZZ_S
}

def AArch64MIR_sqdmlalb : AArch64MIR_TargetOp<"sqdmlalb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQDMLALB_ZZZI_D SQDMLALB_ZZZI_S SQDMLALB_ZZZ_D SQDMLALB_ZZZ_H SQDMLALB_ZZZ_S
}

def AArch64MIR_sqdmlalt : AArch64MIR_TargetOp<"sqdmlalt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQDMLALT_ZZZI_D SQDMLALT_ZZZI_S SQDMLALT_ZZZ_D SQDMLALT_ZZZ_H SQDMLALT_ZZZ_S
}

def AArch64MIR_sqdmlal : AArch64MIR_TargetOp<"sqdmlal", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQDMLALi16 SQDMLALi32 SQDMLALv1i32_indexed SQDMLALv1i64_indexed SQDMLALv2i32_indexed SQDMLALv2i32_v2i64 SQDMLALv4i16_indexed SQDMLALv4i16_v4i32
}

def AArch64MIR_sqdmlal2 : AArch64MIR_TargetOp<"sqdmlal2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQDMLALv4i32_indexed SQDMLALv4i32_v2i64 SQDMLALv8i16_indexed SQDMLALv8i16_v4i32
}

def AArch64MIR_sqdmlslbt : AArch64MIR_TargetOp<"sqdmlslbt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQDMLSLBT_ZZZ_D SQDMLSLBT_ZZZ_H SQDMLSLBT_ZZZ_S
}

def AArch64MIR_sqdmlslb : AArch64MIR_TargetOp<"sqdmlslb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQDMLSLB_ZZZI_D SQDMLSLB_ZZZI_S SQDMLSLB_ZZZ_D SQDMLSLB_ZZZ_H SQDMLSLB_ZZZ_S
}

def AArch64MIR_sqdmlslt : AArch64MIR_TargetOp<"sqdmlslt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQDMLSLT_ZZZI_D SQDMLSLT_ZZZI_S SQDMLSLT_ZZZ_D SQDMLSLT_ZZZ_H SQDMLSLT_ZZZ_S
}

def AArch64MIR_sqdmlsl : AArch64MIR_TargetOp<"sqdmlsl", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQDMLSLi16 SQDMLSLi32 SQDMLSLv1i32_indexed SQDMLSLv1i64_indexed SQDMLSLv2i32_indexed SQDMLSLv2i32_v2i64 SQDMLSLv4i16_indexed SQDMLSLv4i16_v4i32
}

def AArch64MIR_sqdmlsl2 : AArch64MIR_TargetOp<"sqdmlsl2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQDMLSLv4i32_indexed SQDMLSLv4i32_v2i64 SQDMLSLv8i16_indexed SQDMLSLv8i16_v4i32
}

def AArch64MIR_sqdmulh : AArch64MIR_TargetOp<"sqdmulh", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQDMULH_VG2_2Z2Z_B SQDMULH_VG2_2Z2Z_D SQDMULH_VG2_2Z2Z_H SQDMULH_VG2_2Z2Z_S SQDMULH_VG2_2ZZ_B SQDMULH_VG2_2ZZ_D SQDMULH_VG2_2ZZ_H SQDMULH_VG2_2ZZ_S SQDMULH_VG4_4Z4Z_B SQDMULH_VG4_4Z4Z_D SQDMULH_VG4_4Z4Z_H SQDMULH_VG4_4Z4Z_S SQDMULH_VG4_4ZZ_B SQDMULH_VG4_4ZZ_D SQDMULH_VG4_4ZZ_H SQDMULH_VG4_4ZZ_S SQDMULH_ZZZI_D SQDMULH_ZZZI_H SQDMULH_ZZZI_S SQDMULH_ZZZ_B SQDMULH_ZZZ_D SQDMULH_ZZZ_H SQDMULH_ZZZ_S SQDMULHv1i16 SQDMULHv1i16_indexed SQDMULHv1i32 SQDMULHv1i32_indexed SQDMULHv2i32 SQDMULHv2i32_indexed SQDMULHv4i16 SQDMULHv4i16_indexed SQDMULHv4i32 SQDMULHv4i32_indexed SQDMULHv8i16 SQDMULHv8i16_indexed
}

def AArch64MIR_sqdmullb : AArch64MIR_TargetOp<"sqdmullb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQDMULLB_ZZZI_D SQDMULLB_ZZZI_S SQDMULLB_ZZZ_D SQDMULLB_ZZZ_H SQDMULLB_ZZZ_S
}

def AArch64MIR_sqdmullt : AArch64MIR_TargetOp<"sqdmullt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQDMULLT_ZZZI_D SQDMULLT_ZZZI_S SQDMULLT_ZZZ_D SQDMULLT_ZZZ_H SQDMULLT_ZZZ_S
}

def AArch64MIR_sqdmull : AArch64MIR_TargetOp<"sqdmull", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQDMULLi16 SQDMULLi32 SQDMULLv1i32_indexed SQDMULLv1i64_indexed SQDMULLv2i32_indexed SQDMULLv2i32_v2i64 SQDMULLv4i16_indexed SQDMULLv4i16_v4i32
}

def AArch64MIR_sqdmull2 : AArch64MIR_TargetOp<"sqdmull2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQDMULLv4i32_indexed SQDMULLv4i32_v2i64 SQDMULLv8i16_indexed SQDMULLv8i16_v4i32
}

def AArch64MIR_sqincb : AArch64MIR_TargetOp<"sqincb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQINCB_XPiI SQINCB_XPiWdI
}

def AArch64MIR_sqincd : AArch64MIR_TargetOp<"sqincd", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQINCD_XPiI SQINCD_XPiWdI SQINCD_ZPiI
}

def AArch64MIR_sqinch : AArch64MIR_TargetOp<"sqinch", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQINCH_XPiI SQINCH_XPiWdI SQINCH_ZPiI
}

def AArch64MIR_sqincp : AArch64MIR_TargetOp<"sqincp", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQINCP_XPWd_B SQINCP_XPWd_D SQINCP_XPWd_H SQINCP_XPWd_S SQINCP_XP_B SQINCP_XP_D SQINCP_XP_H SQINCP_XP_S SQINCP_ZP_D SQINCP_ZP_H SQINCP_ZP_S
}

def AArch64MIR_sqincw : AArch64MIR_TargetOp<"sqincw", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQINCW_XPiI SQINCW_XPiWdI SQINCW_ZPiI
}

def AArch64MIR_sqneg : AArch64MIR_TargetOp<"sqneg", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQNEG_ZPmZ_B SQNEG_ZPmZ_D SQNEG_ZPmZ_H SQNEG_ZPmZ_S SQNEG_ZPzZ_B SQNEG_ZPzZ_D SQNEG_ZPzZ_H SQNEG_ZPzZ_S SQNEGv1i8 SQNEGv1i16 SQNEGv1i32 SQNEGv1i64 SQNEGv2i32 SQNEGv2i64 SQNEGv4i16 SQNEGv4i32 SQNEGv8i8 SQNEGv8i16 SQNEGv16i8
}

def AArch64MIR_sqrdcmlah : AArch64MIR_TargetOp<"sqrdcmlah", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQRDCMLAH_ZZZI_H SQRDCMLAH_ZZZI_S SQRDCMLAH_ZZZ_B SQRDCMLAH_ZZZ_D SQRDCMLAH_ZZZ_H SQRDCMLAH_ZZZ_S
}

def AArch64MIR_sqrdmlah : AArch64MIR_TargetOp<"sqrdmlah", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQRDMLAH_ZZZI_D SQRDMLAH_ZZZI_H SQRDMLAH_ZZZI_S SQRDMLAH_ZZZ_B SQRDMLAH_ZZZ_D SQRDMLAH_ZZZ_H SQRDMLAH_ZZZ_S SQRDMLAHv1i16 SQRDMLAHv1i16_indexed SQRDMLAHv1i32 SQRDMLAHv1i32_indexed SQRDMLAHv2i32 SQRDMLAHv2i32_indexed SQRDMLAHv4i16 SQRDMLAHv4i16_indexed SQRDMLAHv4i32 SQRDMLAHv4i32_indexed SQRDMLAHv8i16 SQRDMLAHv8i16_indexed
}

def AArch64MIR_sqrdmlsh : AArch64MIR_TargetOp<"sqrdmlsh", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQRDMLSH_ZZZI_D SQRDMLSH_ZZZI_H SQRDMLSH_ZZZI_S SQRDMLSH_ZZZ_B SQRDMLSH_ZZZ_D SQRDMLSH_ZZZ_H SQRDMLSH_ZZZ_S SQRDMLSHv1i16 SQRDMLSHv1i16_indexed SQRDMLSHv1i32 SQRDMLSHv1i32_indexed SQRDMLSHv2i32 SQRDMLSHv2i32_indexed SQRDMLSHv4i16 SQRDMLSHv4i16_indexed SQRDMLSHv4i32 SQRDMLSHv4i32_indexed SQRDMLSHv8i16 SQRDMLSHv8i16_indexed
}

def AArch64MIR_sqrdmulh : AArch64MIR_TargetOp<"sqrdmulh", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQRDMULH_ZZZI_D SQRDMULH_ZZZI_H SQRDMULH_ZZZI_S SQRDMULH_ZZZ_B SQRDMULH_ZZZ_D SQRDMULH_ZZZ_H SQRDMULH_ZZZ_S SQRDMULHv1i16 SQRDMULHv1i16_indexed SQRDMULHv1i32 SQRDMULHv1i32_indexed SQRDMULHv2i32 SQRDMULHv2i32_indexed SQRDMULHv4i16 SQRDMULHv4i16_indexed SQRDMULHv4i32 SQRDMULHv4i32_indexed SQRDMULHv8i16 SQRDMULHv8i16_indexed
}

def AArch64MIR_sqrshlr : AArch64MIR_TargetOp<"sqrshlr", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQRSHLR_ZPmZ_B SQRSHLR_ZPmZ_D SQRSHLR_ZPmZ_H SQRSHLR_ZPmZ_S
}

def AArch64MIR_sqrshl : AArch64MIR_TargetOp<"sqrshl", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQRSHL_ZPmZ_B SQRSHL_ZPmZ_D SQRSHL_ZPmZ_H SQRSHL_ZPmZ_S SQRSHLv1i8 SQRSHLv1i16 SQRSHLv1i32 SQRSHLv1i64 SQRSHLv2i32 SQRSHLv2i64 SQRSHLv4i16 SQRSHLv4i32 SQRSHLv8i8 SQRSHLv8i16 SQRSHLv16i8
}

def AArch64MIR_sqrshrnb : AArch64MIR_TargetOp<"sqrshrnb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQRSHRNB_ZZI_B SQRSHRNB_ZZI_H SQRSHRNB_ZZI_S
}

def AArch64MIR_sqrshrnt : AArch64MIR_TargetOp<"sqrshrnt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQRSHRNT_ZZI_B SQRSHRNT_ZZI_H SQRSHRNT_ZZI_S
}

def AArch64MIR_sqrshrn : AArch64MIR_TargetOp<"sqrshrn", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQRSHRN_VG4_Z4ZI_B SQRSHRN_VG4_Z4ZI_H SQRSHRN_Z2ZI_HtoB SQRSHRN_Z2ZI_StoH SQRSHRNb SQRSHRNh SQRSHRNs SQRSHRNv2i32_shift SQRSHRNv4i16_shift SQRSHRNv8i8_shift
}

def AArch64MIR_sqrshrn2 : AArch64MIR_TargetOp<"sqrshrn2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQRSHRNv4i32_shift SQRSHRNv8i16_shift SQRSHRNv16i8_shift
}

def AArch64MIR_sqrshrunb : AArch64MIR_TargetOp<"sqrshrunb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQRSHRUNB_ZZI_B SQRSHRUNB_ZZI_H SQRSHRUNB_ZZI_S
}

def AArch64MIR_sqrshrunt : AArch64MIR_TargetOp<"sqrshrunt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQRSHRUNT_ZZI_B SQRSHRUNT_ZZI_H SQRSHRUNT_ZZI_S
}

def AArch64MIR_sqrshrun : AArch64MIR_TargetOp<"sqrshrun", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQRSHRUN_VG4_Z4ZI_B SQRSHRUN_VG4_Z4ZI_H SQRSHRUN_Z2ZI_HtoB SQRSHRUN_Z2ZI_StoH SQRSHRUNb SQRSHRUNh SQRSHRUNs SQRSHRUNv2i32_shift SQRSHRUNv4i16_shift SQRSHRUNv8i8_shift
}

def AArch64MIR_sqrshrun2 : AArch64MIR_TargetOp<"sqrshrun2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQRSHRUNv4i32_shift SQRSHRUNv8i16_shift SQRSHRUNv16i8_shift
}

def AArch64MIR_sqrshru : AArch64MIR_TargetOp<"sqrshru", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQRSHRU_VG2_Z2ZI_H SQRSHRU_VG4_Z4ZI_B SQRSHRU_VG4_Z4ZI_H
}

def AArch64MIR_sqrshr : AArch64MIR_TargetOp<"sqrshr", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQRSHR_VG2_Z2ZI_H SQRSHR_VG4_Z4ZI_B SQRSHR_VG4_Z4ZI_H
}

def AArch64MIR_sqshlr : AArch64MIR_TargetOp<"sqshlr", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQSHLR_ZPmZ_B SQSHLR_ZPmZ_D SQSHLR_ZPmZ_H SQSHLR_ZPmZ_S
}

def AArch64MIR_sqshlu : AArch64MIR_TargetOp<"sqshlu", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQSHLU_ZPmI_B SQSHLU_ZPmI_D SQSHLU_ZPmI_H SQSHLU_ZPmI_S SQSHLUb SQSHLUd SQSHLUh SQSHLUs SQSHLUv2i32_shift SQSHLUv2i64_shift SQSHLUv4i16_shift SQSHLUv4i32_shift SQSHLUv8i8_shift SQSHLUv8i16_shift SQSHLUv16i8_shift
}

def AArch64MIR_sqshl : AArch64MIR_TargetOp<"sqshl", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQSHL_ZPmI_B SQSHL_ZPmI_D SQSHL_ZPmI_H SQSHL_ZPmI_S SQSHL_ZPmZ_B SQSHL_ZPmZ_D SQSHL_ZPmZ_H SQSHL_ZPmZ_S SQSHLb SQSHLd SQSHLh SQSHLs SQSHLv1i8 SQSHLv1i16 SQSHLv1i32 SQSHLv1i64 SQSHLv2i32 SQSHLv2i32_shift SQSHLv2i64 SQSHLv2i64_shift SQSHLv4i16 SQSHLv4i16_shift SQSHLv4i32 SQSHLv4i32_shift SQSHLv8i8 SQSHLv8i8_shift SQSHLv8i16 SQSHLv8i16_shift SQSHLv16i8 SQSHLv16i8_shift
}

def AArch64MIR_sqshrnb : AArch64MIR_TargetOp<"sqshrnb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQSHRNB_ZZI_B SQSHRNB_ZZI_H SQSHRNB_ZZI_S
}

def AArch64MIR_sqshrnt : AArch64MIR_TargetOp<"sqshrnt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQSHRNT_ZZI_B SQSHRNT_ZZI_H SQSHRNT_ZZI_S
}

def AArch64MIR_sqshrn : AArch64MIR_TargetOp<"sqshrn", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQSHRN_Z2ZI_HtoB SQSHRN_Z2ZI_StoH SQSHRNb SQSHRNh SQSHRNs SQSHRNv2i32_shift SQSHRNv4i16_shift SQSHRNv8i8_shift
}

def AArch64MIR_sqshrn2 : AArch64MIR_TargetOp<"sqshrn2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQSHRNv4i32_shift SQSHRNv8i16_shift SQSHRNv16i8_shift
}

def AArch64MIR_sqshrunb : AArch64MIR_TargetOp<"sqshrunb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQSHRUNB_ZZI_B SQSHRUNB_ZZI_H SQSHRUNB_ZZI_S
}

def AArch64MIR_sqshrunt : AArch64MIR_TargetOp<"sqshrunt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQSHRUNT_ZZI_B SQSHRUNT_ZZI_H SQSHRUNT_ZZI_S
}

def AArch64MIR_sqshrun : AArch64MIR_TargetOp<"sqshrun", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQSHRUN_Z2ZI_HtoB SQSHRUN_Z2ZI_StoH SQSHRUNb SQSHRUNh SQSHRUNs SQSHRUNv2i32_shift SQSHRUNv4i16_shift SQSHRUNv8i8_shift
}

def AArch64MIR_sqshrun2 : AArch64MIR_TargetOp<"sqshrun2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQSHRUNv4i32_shift SQSHRUNv8i16_shift SQSHRUNv16i8_shift
}

def AArch64MIR_sqsubr : AArch64MIR_TargetOp<"sqsubr", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQSUBR_ZPmZ_B SQSUBR_ZPmZ_D SQSUBR_ZPmZ_H SQSUBR_ZPmZ_S
}

def AArch64MIR_sqsub : AArch64MIR_TargetOp<"sqsub", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQSUB_ZI_B SQSUB_ZI_D SQSUB_ZI_H SQSUB_ZI_S SQSUB_ZPmZ_B SQSUB_ZPmZ_D SQSUB_ZPmZ_H SQSUB_ZPmZ_S SQSUB_ZZZ_B SQSUB_ZZZ_D SQSUB_ZZZ_H SQSUB_ZZZ_S SQSUBv1i8 SQSUBv1i16 SQSUBv1i32 SQSUBv1i64 SQSUBv2i32 SQSUBv2i64 SQSUBv4i16 SQSUBv4i32 SQSUBv8i8 SQSUBv8i16 SQSUBv16i8
}

def AArch64MIR_sqxtnb : AArch64MIR_TargetOp<"sqxtnb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQXTNB_ZZ_B SQXTNB_ZZ_H SQXTNB_ZZ_S
}

def AArch64MIR_sqxtnt : AArch64MIR_TargetOp<"sqxtnt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQXTNT_ZZ_B SQXTNT_ZZ_H SQXTNT_ZZ_S
}

def AArch64MIR_sqxtn : AArch64MIR_TargetOp<"sqxtn", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQXTNv1i8 SQXTNv1i16 SQXTNv1i32 SQXTNv2i32 SQXTNv4i16 SQXTNv8i8
}

def AArch64MIR_sqxtn2 : AArch64MIR_TargetOp<"sqxtn2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQXTNv4i32 SQXTNv8i16 SQXTNv16i8
}

def AArch64MIR_sqxtunb : AArch64MIR_TargetOp<"sqxtunb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQXTUNB_ZZ_B SQXTUNB_ZZ_H SQXTUNB_ZZ_S
}

def AArch64MIR_sqxtunt : AArch64MIR_TargetOp<"sqxtunt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQXTUNT_ZZ_B SQXTUNT_ZZ_H SQXTUNT_ZZ_S
}

def AArch64MIR_sqxtun : AArch64MIR_TargetOp<"sqxtun", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQXTUNv1i8 SQXTUNv1i16 SQXTUNv1i32 SQXTUNv2i32 SQXTUNv4i16 SQXTUNv8i8
}

def AArch64MIR_sqxtun2 : AArch64MIR_TargetOp<"sqxtun2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SQXTUNv4i32 SQXTUNv8i16 SQXTUNv16i8
}

def AArch64MIR_srhadd : AArch64MIR_TargetOp<"srhadd", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SRHADD_ZPmZ_B SRHADD_ZPmZ_D SRHADD_ZPmZ_H SRHADD_ZPmZ_S SRHADDv2i32 SRHADDv4i16 SRHADDv4i32 SRHADDv8i8 SRHADDv8i16 SRHADDv16i8
}

def AArch64MIR_sri : AArch64MIR_TargetOp<"sri", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SRI_ZZI_B SRI_ZZI_D SRI_ZZI_H SRI_ZZI_S SRId SRIv2i32_shift SRIv2i64_shift SRIv4i16_shift SRIv4i32_shift SRIv8i8_shift SRIv8i16_shift SRIv16i8_shift
}

def AArch64MIR_srshlr : AArch64MIR_TargetOp<"srshlr", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SRSHLR_ZPmZ_B SRSHLR_ZPmZ_D SRSHLR_ZPmZ_H SRSHLR_ZPmZ_S
}

def AArch64MIR_srshl : AArch64MIR_TargetOp<"srshl", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SRSHL_VG2_2Z2Z_B SRSHL_VG2_2Z2Z_D SRSHL_VG2_2Z2Z_H SRSHL_VG2_2Z2Z_S SRSHL_VG2_2ZZ_B SRSHL_VG2_2ZZ_D SRSHL_VG2_2ZZ_H SRSHL_VG2_2ZZ_S SRSHL_VG4_4Z4Z_B SRSHL_VG4_4Z4Z_D SRSHL_VG4_4Z4Z_H SRSHL_VG4_4Z4Z_S SRSHL_VG4_4ZZ_B SRSHL_VG4_4ZZ_D SRSHL_VG4_4ZZ_H SRSHL_VG4_4ZZ_S SRSHL_ZPmZ_B SRSHL_ZPmZ_D SRSHL_ZPmZ_H SRSHL_ZPmZ_S SRSHLv1i64 SRSHLv2i32 SRSHLv2i64 SRSHLv4i16 SRSHLv4i32 SRSHLv8i8 SRSHLv8i16 SRSHLv16i8
}

def AArch64MIR_srshr : AArch64MIR_TargetOp<"srshr", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SRSHR_ZPmI_B SRSHR_ZPmI_D SRSHR_ZPmI_H SRSHR_ZPmI_S SRSHRd SRSHRv2i32_shift SRSHRv2i64_shift SRSHRv4i16_shift SRSHRv4i32_shift SRSHRv8i8_shift SRSHRv8i16_shift SRSHRv16i8_shift
}

def AArch64MIR_srsra : AArch64MIR_TargetOp<"srsra", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SRSRA_ZZI_B SRSRA_ZZI_D SRSRA_ZZI_H SRSRA_ZZI_S SRSRAd SRSRAv2i32_shift SRSRAv2i64_shift SRSRAv4i16_shift SRSRAv4i32_shift SRSRAv8i8_shift SRSRAv8i16_shift SRSRAv16i8_shift
}

def AArch64MIR_sshllb : AArch64MIR_TargetOp<"sshllb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SSHLLB_ZZI_D SSHLLB_ZZI_H SSHLLB_ZZI_S
}

def AArch64MIR_sshllt : AArch64MIR_TargetOp<"sshllt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SSHLLT_ZZI_D SSHLLT_ZZI_H SSHLLT_ZZI_S
}

def AArch64MIR_sshll : AArch64MIR_TargetOp<"sshll", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SSHLLv2i32_shift SSHLLv4i16_shift SSHLLv8i8_shift
}

def AArch64MIR_sshll2 : AArch64MIR_TargetOp<"sshll2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SSHLLv4i32_shift SSHLLv8i16_shift SSHLLv16i8_shift
}

def AArch64MIR_sshl : AArch64MIR_TargetOp<"sshl", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SSHLv1i64 SSHLv2i32 SSHLv2i64 SSHLv4i16 SSHLv4i32 SSHLv8i8 SSHLv8i16 SSHLv16i8
}

def AArch64MIR_sshr : AArch64MIR_TargetOp<"sshr", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SSHRd SSHRv2i32_shift SSHRv2i64_shift SSHRv4i16_shift SSHRv4i32_shift SSHRv8i8_shift SSHRv8i16_shift SSHRv16i8_shift
}

def AArch64MIR_ssra : AArch64MIR_TargetOp<"ssra", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SSRA_ZZI_B SSRA_ZZI_D SSRA_ZZI_H SSRA_ZZI_S SSRAd SSRAv2i32_shift SSRAv2i64_shift SSRAv4i16_shift SSRAv4i32_shift SSRAv8i8_shift SSRAv8i16_shift SSRAv16i8_shift
}

def AArch64MIR_st1b : AArch64MIR_TargetOp<"st1b", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SST1B_D SST1B_D_IMM SST1B_D_SXTW SST1B_D_UXTW SST1B_S_IMM SST1B_S_SXTW SST1B_S_UXTW ST1B ST1B_2Z ST1B_2Z_IMM ST1B_2Z_STRIDED ST1B_2Z_STRIDED_IMM ST1B_4Z ST1B_4Z_IMM ST1B_4Z_STRIDED ST1B_4Z_STRIDED_IMM ST1B_D ST1B_D_IMM ST1B_H ST1B_H_IMM ST1B_IMM ST1B_S ST1B_S_IMM ST1_MXIPXX_H_B ST1_MXIPXX_V_B
}

def AArch64MIR_st1d : AArch64MIR_TargetOp<"st1d", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SST1D SST1D_IMM SST1D_SCALED SST1D_SXTW SST1D_SXTW_SCALED SST1D_UXTW SST1D_UXTW_SCALED ST1D ST1D_2Z ST1D_2Z_IMM ST1D_2Z_STRIDED ST1D_2Z_STRIDED_IMM ST1D_4Z ST1D_4Z_IMM ST1D_4Z_STRIDED ST1D_4Z_STRIDED_IMM ST1D_IMM ST1D_Q ST1D_Q_IMM ST1_MXIPXX_H_D ST1_MXIPXX_V_D
}

def AArch64MIR_st1h : AArch64MIR_TargetOp<"st1h", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SST1H_D SST1H_D_IMM SST1H_D_SCALED SST1H_D_SXTW SST1H_D_SXTW_SCALED SST1H_D_UXTW SST1H_D_UXTW_SCALED SST1H_S_IMM SST1H_S_SXTW SST1H_S_SXTW_SCALED SST1H_S_UXTW SST1H_S_UXTW_SCALED ST1H ST1H_2Z ST1H_2Z_IMM ST1H_2Z_STRIDED ST1H_2Z_STRIDED_IMM ST1H_4Z ST1H_4Z_IMM ST1H_4Z_STRIDED ST1H_4Z_STRIDED_IMM ST1H_D ST1H_D_IMM ST1H_IMM ST1H_S ST1H_S_IMM ST1_MXIPXX_H_H ST1_MXIPXX_V_H
}

def AArch64MIR_st1q : AArch64MIR_TargetOp<"st1q", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SST1Q ST1_MXIPXX_H_Q ST1_MXIPXX_V_Q
}

def AArch64MIR_st1w : AArch64MIR_TargetOp<"st1w", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SST1W_D SST1W_D_IMM SST1W_D_SCALED SST1W_D_SXTW SST1W_D_SXTW_SCALED SST1W_D_UXTW SST1W_D_UXTW_SCALED SST1W_IMM SST1W_SXTW SST1W_SXTW_SCALED SST1W_UXTW SST1W_UXTW_SCALED ST1W ST1W_2Z ST1W_2Z_IMM ST1W_2Z_STRIDED ST1W_2Z_STRIDED_IMM ST1W_4Z ST1W_4Z_IMM ST1W_4Z_STRIDED ST1W_4Z_STRIDED_IMM ST1W_D ST1W_D_IMM ST1W_IMM ST1W_Q ST1W_Q_IMM ST1_MXIPXX_H_S ST1_MXIPXX_V_S
}

def AArch64MIR_ssublbt : AArch64MIR_TargetOp<"ssublbt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SSUBLBT_ZZZ_D SSUBLBT_ZZZ_H SSUBLBT_ZZZ_S
}

def AArch64MIR_ssublb : AArch64MIR_TargetOp<"ssublb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SSUBLB_ZZZ_D SSUBLB_ZZZ_H SSUBLB_ZZZ_S
}

def AArch64MIR_ssubltb : AArch64MIR_TargetOp<"ssubltb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SSUBLTB_ZZZ_D SSUBLTB_ZZZ_H SSUBLTB_ZZZ_S
}

def AArch64MIR_ssublt : AArch64MIR_TargetOp<"ssublt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SSUBLT_ZZZ_D SSUBLT_ZZZ_H SSUBLT_ZZZ_S
}

def AArch64MIR_ssubl : AArch64MIR_TargetOp<"ssubl", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SSUBLv2i32_v2i64 SSUBLv4i16_v4i32 SSUBLv8i8_v8i16
}

def AArch64MIR_ssubl2 : AArch64MIR_TargetOp<"ssubl2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SSUBLv4i32_v2i64 SSUBLv8i16_v4i32 SSUBLv16i8_v8i16
}

def AArch64MIR_ssubwb : AArch64MIR_TargetOp<"ssubwb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SSUBWB_ZZZ_D SSUBWB_ZZZ_H SSUBWB_ZZZ_S
}

def AArch64MIR_ssubwt : AArch64MIR_TargetOp<"ssubwt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SSUBWT_ZZZ_D SSUBWT_ZZZ_H SSUBWT_ZZZ_S
}

def AArch64MIR_ssubw : AArch64MIR_TargetOp<"ssubw", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SSUBWv2i32_v2i64 SSUBWv4i16_v4i32 SSUBWv8i8_v8i16
}

def AArch64MIR_ssubw2 : AArch64MIR_TargetOp<"ssubw2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SSUBWv4i32_v2i64 SSUBWv8i16_v4i32 SSUBWv16i8_v8i16
}

def AArch64MIR_st1 : AArch64MIR_TargetOp<"st1", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ST1Fourv1d ST1Fourv1d_POST ST1Fourv2d ST1Fourv2d_POST ST1Fourv2s ST1Fourv2s_POST ST1Fourv4h ST1Fourv4h_POST ST1Fourv4s ST1Fourv4s_POST ST1Fourv8b ST1Fourv8b_POST ST1Fourv8h ST1Fourv8h_POST ST1Fourv16b ST1Fourv16b_POST ST1Onev1d ST1Onev1d_POST ST1Onev2d ST1Onev2d_POST ST1Onev2s ST1Onev2s_POST ST1Onev4h ST1Onev4h_POST ST1Onev4s ST1Onev4s_POST ST1Onev8b ST1Onev8b_POST ST1Onev8h ST1Onev8h_POST ST1Onev16b ST1Onev16b_POST ST1Threev1d ST1Threev1d_POST ST1Threev2d ST1Threev2d_POST ST1Threev2s ST1Threev2s_POST ST1Threev4h ST1Threev4h_POST ST1Threev4s ST1Threev4s_POST ST1Threev8b ST1Threev8b_POST ST1Threev8h ST1Threev8h_POST ST1Threev16b ST1Threev16b_POST ST1Twov1d ST1Twov1d_POST ST1Twov2d ST1Twov2d_POST ST1Twov2s ST1Twov2s_POST ST1Twov4h ST1Twov4h_POST ST1Twov4s ST1Twov4s_POST ST1Twov8b ST1Twov8b_POST ST1Twov8h ST1Twov8h_POST ST1Twov16b ST1Twov16b_POST ST1i8 ST1i8_POST ST1i16 ST1i16_POST ST1i32 ST1i32_POST ST1i64 ST1i64_POST
}

def AArch64MIR_st2b : AArch64MIR_TargetOp<"st2b", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ST2B ST2B_IMM
}

def AArch64MIR_st2d : AArch64MIR_TargetOp<"st2d", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ST2D ST2D_IMM
}

def AArch64MIR_st2g : AArch64MIR_TargetOp<"st2g", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ST2GPostIndex ST2GPreIndex ST2Gi
}

def AArch64MIR_st2h : AArch64MIR_TargetOp<"st2h", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ST2H ST2H_IMM
}

def AArch64MIR_st2q : AArch64MIR_TargetOp<"st2q", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ST2Q ST2Q_IMM
}

def AArch64MIR_st2 : AArch64MIR_TargetOp<"st2", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ST2Twov2d ST2Twov2d_POST ST2Twov2s ST2Twov2s_POST ST2Twov4h ST2Twov4h_POST ST2Twov4s ST2Twov4s_POST ST2Twov8b ST2Twov8b_POST ST2Twov8h ST2Twov8h_POST ST2Twov16b ST2Twov16b_POST ST2i8 ST2i8_POST ST2i16 ST2i16_POST ST2i32 ST2i32_POST ST2i64 ST2i64_POST
}

def AArch64MIR_st2w : AArch64MIR_TargetOp<"st2w", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ST2W ST2W_IMM
}

def AArch64MIR_st3b : AArch64MIR_TargetOp<"st3b", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ST3B ST3B_IMM
}

def AArch64MIR_st3d : AArch64MIR_TargetOp<"st3d", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ST3D ST3D_IMM
}

def AArch64MIR_st3h : AArch64MIR_TargetOp<"st3h", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ST3H ST3H_IMM
}

def AArch64MIR_st3q : AArch64MIR_TargetOp<"st3q", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ST3Q ST3Q_IMM
}

def AArch64MIR_st3 : AArch64MIR_TargetOp<"st3", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ST3Threev2d ST3Threev2d_POST ST3Threev2s ST3Threev2s_POST ST3Threev4h ST3Threev4h_POST ST3Threev4s ST3Threev4s_POST ST3Threev8b ST3Threev8b_POST ST3Threev8h ST3Threev8h_POST ST3Threev16b ST3Threev16b_POST ST3i8 ST3i8_POST ST3i16 ST3i16_POST ST3i32 ST3i32_POST ST3i64 ST3i64_POST
}

def AArch64MIR_st3w : AArch64MIR_TargetOp<"st3w", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ST3W ST3W_IMM
}

def AArch64MIR_st4b : AArch64MIR_TargetOp<"st4b", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ST4B ST4B_IMM
}

def AArch64MIR_st4d : AArch64MIR_TargetOp<"st4d", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ST4D ST4D_IMM
}

def AArch64MIR_st4 : AArch64MIR_TargetOp<"st4", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ST4Fourv2d ST4Fourv2d_POST ST4Fourv2s ST4Fourv2s_POST ST4Fourv4h ST4Fourv4h_POST ST4Fourv4s ST4Fourv4s_POST ST4Fourv8b ST4Fourv8b_POST ST4Fourv8h ST4Fourv8h_POST ST4Fourv16b ST4Fourv16b_POST ST4i8 ST4i8_POST ST4i16 ST4i16_POST ST4i32 ST4i32_POST ST4i64 ST4i64_POST
}

def AArch64MIR_st4h : AArch64MIR_TargetOp<"st4h", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ST4H ST4H_IMM
}

def AArch64MIR_st4q : AArch64MIR_TargetOp<"st4q", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ST4Q ST4Q_IMM
}

def AArch64MIR_st4w : AArch64MIR_TargetOp<"st4w", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ST4W ST4W_IMM
}

def AArch64MIR_st64b : AArch64MIR_TargetOp<"st64b", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ST64B
}

def AArch64MIR_st64bv : AArch64MIR_TargetOp<"st64bv", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ST64BV
}

def AArch64MIR_st64bv0 : AArch64MIR_TargetOp<"st64bv0", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ST64BV0
}

def AArch64MIR_stbfadd : AArch64MIR_TargetOp<"stbfadd", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STBFADD
}

def AArch64MIR_stbfaddl : AArch64MIR_TargetOp<"stbfaddl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STBFADDL
}

def AArch64MIR_stbfmax : AArch64MIR_TargetOp<"stbfmax", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STBFMAX
}

def AArch64MIR_stbfmaxl : AArch64MIR_TargetOp<"stbfmaxl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STBFMAXL
}

def AArch64MIR_stbfmaxnm : AArch64MIR_TargetOp<"stbfmaxnm", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STBFMAXNM
}

def AArch64MIR_stbfmaxnml : AArch64MIR_TargetOp<"stbfmaxnml", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STBFMAXNML
}

def AArch64MIR_stbfmin : AArch64MIR_TargetOp<"stbfmin", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STBFMIN
}

def AArch64MIR_stbfminl : AArch64MIR_TargetOp<"stbfminl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STBFMINL
}

def AArch64MIR_stbfminnm : AArch64MIR_TargetOp<"stbfminnm", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STBFMINNM
}

def AArch64MIR_stbfminnml : AArch64MIR_TargetOp<"stbfminnml", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STBFMINNML
}

def AArch64MIR_stfadd : AArch64MIR_TargetOp<"stfadd", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STFADDD STFADDH STFADDS
}

def AArch64MIR_stfaddl : AArch64MIR_TargetOp<"stfaddl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STFADDLD STFADDLH STFADDLS
}

def AArch64MIR_stfmax : AArch64MIR_TargetOp<"stfmax", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STFMAXD STFMAXH STFMAXS
}

def AArch64MIR_stfmaxl : AArch64MIR_TargetOp<"stfmaxl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STFMAXLD STFMAXLH STFMAXLS
}

def AArch64MIR_stfmaxnm : AArch64MIR_TargetOp<"stfmaxnm", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STFMAXNMD STFMAXNMH STFMAXNMS
}

def AArch64MIR_stfmaxnml : AArch64MIR_TargetOp<"stfmaxnml", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STFMAXNMLD STFMAXNMLH STFMAXNMLS
}

def AArch64MIR_stfmin : AArch64MIR_TargetOp<"stfmin", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STFMIND STFMINH STFMINS
}

def AArch64MIR_stfminl : AArch64MIR_TargetOp<"stfminl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STFMINLD STFMINLH STFMINLS
}

def AArch64MIR_stfminnm : AArch64MIR_TargetOp<"stfminnm", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STFMINNMD STFMINNMH STFMINNMS
}

def AArch64MIR_stfminnml : AArch64MIR_TargetOp<"stfminnml", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STFMINNMLD STFMINNMLH STFMINNMLS
}

def AArch64MIR_stgm : AArch64MIR_TargetOp<"stgm", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STGM
}

def AArch64MIR_stgp : AArch64MIR_TargetOp<"stgp", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STGPi STGPpost STGPpre
}

def AArch64MIR_stg : AArch64MIR_TargetOp<"stg", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STGPostIndex STGPreIndex STGi
}

def AArch64MIR_stilp : AArch64MIR_TargetOp<"stilp", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STILPW STILPWpre STILPX STILPXpre
}

def AArch64MIR_stl1 : AArch64MIR_TargetOp<"stl1", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STL1
}

def AArch64MIR_stllrb : AArch64MIR_TargetOp<"stllrb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STLLRB
}

def AArch64MIR_stllrh : AArch64MIR_TargetOp<"stllrh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STLLRH
}

def AArch64MIR_stllr : AArch64MIR_TargetOp<"stllr", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STLLRW STLLRX
}

def AArch64MIR_stlp : AArch64MIR_TargetOp<"stlp", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STLPi
}

def AArch64MIR_stlrb : AArch64MIR_TargetOp<"stlrb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STLRB
}

def AArch64MIR_stlrh : AArch64MIR_TargetOp<"stlrh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STLRH
}

def AArch64MIR_stlr : AArch64MIR_TargetOp<"stlr", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STLRW STLRWpre STLRX STLRXpre
}

def AArch64MIR_stltxr : AArch64MIR_TargetOp<"stltxr", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STLTXRW STLTXRX
}

def AArch64MIR_stlurb : AArch64MIR_TargetOp<"stlurb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STLURBi
}

def AArch64MIR_stlurh : AArch64MIR_TargetOp<"stlurh", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STLURHi
}

def AArch64MIR_stlur : AArch64MIR_TargetOp<"stlur", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STLURWi STLURXi STLURbi STLURdi STLURhi STLURqi STLURsi
}

def AArch64MIR_stlxp : AArch64MIR_TargetOp<"stlxp", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STLXPW STLXPX
}

def AArch64MIR_stlxrb : AArch64MIR_TargetOp<"stlxrb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STLXRB
}

def AArch64MIR_stlxrh : AArch64MIR_TargetOp<"stlxrh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STLXRH
}

def AArch64MIR_stlxr : AArch64MIR_TargetOp<"stlxr", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STLXRW STLXRX
}

def AArch64MIR_stmopa : AArch64MIR_TargetOp<"stmopa", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STMOPA_M2ZZZI_BtoS STMOPA_M2ZZZI_HtoS
}

def AArch64MIR_stnp : AArch64MIR_TargetOp<"stnp", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STNPDi STNPQi STNPSi STNPWi STNPXi
}

def AArch64MIR_stnt1b : AArch64MIR_TargetOp<"stnt1b", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STNT1B_2Z STNT1B_2Z_IMM STNT1B_2Z_STRIDED STNT1B_2Z_STRIDED_IMM STNT1B_4Z STNT1B_4Z_IMM STNT1B_4Z_STRIDED STNT1B_4Z_STRIDED_IMM STNT1B_ZRI STNT1B_ZRR STNT1B_ZZR_D STNT1B_ZZR_S
}

def AArch64MIR_stnt1d : AArch64MIR_TargetOp<"stnt1d", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STNT1D_2Z STNT1D_2Z_IMM STNT1D_2Z_STRIDED STNT1D_2Z_STRIDED_IMM STNT1D_4Z STNT1D_4Z_IMM STNT1D_4Z_STRIDED STNT1D_4Z_STRIDED_IMM STNT1D_ZRI STNT1D_ZRR STNT1D_ZZR_D
}

def AArch64MIR_stnt1h : AArch64MIR_TargetOp<"stnt1h", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STNT1H_2Z STNT1H_2Z_IMM STNT1H_2Z_STRIDED STNT1H_2Z_STRIDED_IMM STNT1H_4Z STNT1H_4Z_IMM STNT1H_4Z_STRIDED STNT1H_4Z_STRIDED_IMM STNT1H_ZRI STNT1H_ZRR STNT1H_ZZR_D STNT1H_ZZR_S
}

def AArch64MIR_stnt1w : AArch64MIR_TargetOp<"stnt1w", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STNT1W_2Z STNT1W_2Z_IMM STNT1W_2Z_STRIDED STNT1W_2Z_STRIDED_IMM STNT1W_4Z STNT1W_4Z_IMM STNT1W_4Z_STRIDED STNT1W_4Z_STRIDED_IMM STNT1W_ZRI STNT1W_ZRR STNT1W_ZZR_D STNT1W_ZZR_S
}

def AArch64MIR_stp : AArch64MIR_TargetOp<"stp", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STPDi STPDpost STPDpre STPQi STPQpost STPQpre STPSi STPSpost STPSpre STPWi STPWpost STPWpre STPXi STPXpost STPXpre
}

def AArch64MIR_strb : AArch64MIR_TargetOp<"strb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STRBBpost STRBBpre STRBBroW STRBBroX STRBBui
}

def AArch64MIR_str : AArch64MIR_TargetOp<"str", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STRBpost STRBpre STRBroW STRBroX STRBui STRDpost STRDpre STRDroW STRDroX STRDui STRHpost STRHpre STRHroW STRHroX STRHui STRQpost STRQpre STRQroW STRQroX STRQui STRSpost STRSpre STRSroW STRSroX STRSui STRWpost STRWpre STRWroW STRWroX STRWui STRXpost STRXpre STRXroW STRXroX STRXui STR_PXI STR_TX STR_ZA STR_ZXI
}

def AArch64MIR_strh : AArch64MIR_TargetOp<"strh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STRHHpost STRHHpre STRHHroW STRHHroX STRHHui
}

def AArch64MIR_sttnp : AArch64MIR_TargetOp<"sttnp", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STTNPQi STTNPXi
}

def AArch64MIR_sttp : AArch64MIR_TargetOp<"sttp", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STTPQi STTPQpost STTPQpre STTPi STTPpost STTPpre
}

def AArch64MIR_sttrb : AArch64MIR_TargetOp<"sttrb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STTRBi
}

def AArch64MIR_sttrh : AArch64MIR_TargetOp<"sttrh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STTRHi
}

def AArch64MIR_sttr : AArch64MIR_TargetOp<"sttr", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STTRWi STTRXi
}

def AArch64MIR_sttxr : AArch64MIR_TargetOp<"sttxr", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STTXRWr STTXRXr
}

def AArch64MIR_sturb : AArch64MIR_TargetOp<"sturb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STURBBi
}

def AArch64MIR_stur : AArch64MIR_TargetOp<"stur", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STURBi STURDi STURHi STURQi STURSi STURWi STURXi
}

def AArch64MIR_sturh : AArch64MIR_TargetOp<"sturh", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STURHHi
}

def AArch64MIR_stxp : AArch64MIR_TargetOp<"stxp", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STXPW STXPX
}

def AArch64MIR_stxrb : AArch64MIR_TargetOp<"stxrb", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STXRB
}

def AArch64MIR_stxrh : AArch64MIR_TargetOp<"stxrh", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STXRH
}

def AArch64MIR_stxr : AArch64MIR_TargetOp<"stxr", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STXRW STXRX
}

def AArch64MIR_stz2g : AArch64MIR_TargetOp<"stz2g", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STZ2GPostIndex STZ2GPreIndex STZ2Gi
}

def AArch64MIR_stzgm : AArch64MIR_TargetOp<"stzgm", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STZGM
}

def AArch64MIR_stzg : AArch64MIR_TargetOp<"stzg", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: STZGPostIndex STZGPreIndex STZGi
}

def AArch64MIR_subg : AArch64MIR_TargetOp<"subg", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SUBG
}

def AArch64MIR_subhnb : AArch64MIR_TargetOp<"subhnb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SUBHNB_ZZZ_B SUBHNB_ZZZ_H SUBHNB_ZZZ_S
}

def AArch64MIR_subhnt : AArch64MIR_TargetOp<"subhnt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SUBHNT_ZZZ_B SUBHNT_ZZZ_H SUBHNT_ZZZ_S
}

def AArch64MIR_subhn : AArch64MIR_TargetOp<"subhn", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SUBHNv2i64_v2i32 SUBHNv4i32_v4i16 SUBHNv8i16_v8i8
}

def AArch64MIR_subhn2 : AArch64MIR_TargetOp<"subhn2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SUBHNv2i64_v4i32 SUBHNv4i32_v8i16 SUBHNv8i16_v16i8
}

def AArch64MIR_subp : AArch64MIR_TargetOp<"subp", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SUBP SUBP_ZPmZZ_B SUBP_ZPmZZ_D SUBP_ZPmZZ_H SUBP_ZPmZZ_S
}

def AArch64MIR_subps : AArch64MIR_TargetOp<"subps", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SUBPS
}

def AArch64MIR_subpt : AArch64MIR_TargetOp<"subpt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SUBPT_shift SUB_ZPmZ_CPA SUB_ZZZ_CPA
}

def AArch64MIR_subr : AArch64MIR_TargetOp<"subr", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SUBR_ZI_B SUBR_ZI_D SUBR_ZI_H SUBR_ZI_S SUBR_ZPmZ_B SUBR_ZPmZ_D SUBR_ZPmZ_H SUBR_ZPmZ_S
}

def AArch64MIR_subs : AArch64MIR_TargetOp<"subs", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SUBSWri SUBSWrs SUBSWrx SUBSXri SUBSXrs SUBSXrx SUBSXrx64
}

def AArch64MIR_sub : AArch64MIR_TargetOp<"sub", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SUBWri SUBWrs SUBWrx SUBXri SUBXrs SUBXrx SUBXrx64 SUB_VG2_M2Z2Z_D SUB_VG2_M2Z2Z_S SUB_VG2_M2ZZ_D SUB_VG2_M2ZZ_S SUB_VG2_M2Z_D SUB_VG2_M2Z_S SUB_VG4_M4Z4Z_D SUB_VG4_M4Z4Z_S SUB_VG4_M4ZZ_D SUB_VG4_M4ZZ_S SUB_VG4_M4Z_D SUB_VG4_M4Z_S SUB_ZI_B SUB_ZI_D SUB_ZI_H SUB_ZI_S SUB_ZPmZ_B SUB_ZPmZ_D SUB_ZPmZ_H SUB_ZPmZ_S SUB_ZZZ_B SUB_ZZZ_D SUB_ZZZ_H SUB_ZZZ_S SUBv1i64 SUBv2i32 SUBv2i64 SUBv4i16 SUBv4i32 SUBv8i8 SUBv8i16 SUBv16i8
}

def AArch64MIR_sudot : AArch64MIR_TargetOp<"sudot", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SUDOT_VG2_M2ZZI_BToS SUDOT_VG2_M2ZZ_BToS SUDOT_VG4_M4ZZI_BToS SUDOT_VG4_M4ZZ_BToS SUDOT_ZZZI SUDOTlanev8i8 SUDOTlanev16i8
}

def AArch64MIR_sumlall : AArch64MIR_TargetOp<"sumlall", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SUMLALL_MZZI_BtoS SUMLALL_VG2_M2ZZI_BtoS SUMLALL_VG2_M2ZZ_BtoS SUMLALL_VG4_M4ZZI_BtoS SUMLALL_VG4_M4ZZ_BtoS
}

def AArch64MIR_sumop4a : AArch64MIR_TargetOp<"sumop4a", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SUMOP4A_M2Z2Z_BToS SUMOP4A_M2Z2Z_HtoD SUMOP4A_M2ZZ_BToS SUMOP4A_M2ZZ_HtoD SUMOP4A_MZ2Z_BToS SUMOP4A_MZ2Z_HtoD SUMOP4A_MZZ_BToS SUMOP4A_MZZ_HtoD
}

def AArch64MIR_sumop4s : AArch64MIR_TargetOp<"sumop4s", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SUMOP4S_M2Z2Z_BToS SUMOP4S_M2Z2Z_HtoD SUMOP4S_M2ZZ_BToS SUMOP4S_M2ZZ_HtoD SUMOP4S_MZ2Z_BToS SUMOP4S_MZ2Z_HtoD SUMOP4S_MZZ_BToS SUMOP4S_MZZ_HtoD
}

def AArch64MIR_sumopa : AArch64MIR_TargetOp<"sumopa", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SUMOPA_MPPZZ_D SUMOPA_MPPZZ_S
}

def AArch64MIR_sumops : AArch64MIR_TargetOp<"sumops", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SUMOPS_MPPZZ_D SUMOPS_MPPZZ_S
}

def AArch64MIR_sunpkhi : AArch64MIR_TargetOp<"sunpkhi", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SUNPKHI_ZZ_D SUNPKHI_ZZ_H SUNPKHI_ZZ_S
}

def AArch64MIR_sunpklo : AArch64MIR_TargetOp<"sunpklo", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SUNPKLO_ZZ_D SUNPKLO_ZZ_H SUNPKLO_ZZ_S
}

def AArch64MIR_sunpk : AArch64MIR_TargetOp<"sunpk", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SUNPK_VG2_2ZZ_D SUNPK_VG2_2ZZ_H SUNPK_VG2_2ZZ_S SUNPK_VG4_4Z2Z_D SUNPK_VG4_4Z2Z_H SUNPK_VG4_4Z2Z_S
}

def AArch64MIR_suqadd : AArch64MIR_TargetOp<"suqadd", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SUQADD_ZPmZ_B SUQADD_ZPmZ_D SUQADD_ZPmZ_H SUQADD_ZPmZ_S SUQADDv1i8 SUQADDv1i16 SUQADDv1i32 SUQADDv1i64 SUQADDv2i32 SUQADDv2i64 SUQADDv4i16 SUQADDv4i32 SUQADDv8i8 SUQADDv8i16 SUQADDv16i8
}

def AArch64MIR_sutmopa : AArch64MIR_TargetOp<"sutmopa", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SUTMOPA_M2ZZZI_BtoS
}

def AArch64MIR_suvdot : AArch64MIR_TargetOp<"suvdot", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SUVDOT_VG4_M4ZZI_BToS
}

def AArch64MIR_svc : AArch64MIR_TargetOp<"svc", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SVC
}

def AArch64MIR_svdot : AArch64MIR_TargetOp<"svdot", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SVDOT_VG2_M2ZZI_HtoS SVDOT_VG4_M4ZZI_BtoS SVDOT_VG4_M4ZZI_HtoD
}

def AArch64MIR_swpab : AArch64MIR_TargetOp<"swpab", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SWPAB
}

def AArch64MIR_swpah : AArch64MIR_TargetOp<"swpah", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SWPAH
}

def AArch64MIR_swpalb : AArch64MIR_TargetOp<"swpalb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SWPALB
}

def AArch64MIR_swpalh : AArch64MIR_TargetOp<"swpalh", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SWPALH
}

def AArch64MIR_swpal : AArch64MIR_TargetOp<"swpal", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SWPALW SWPALX
}

def AArch64MIR_swpa : AArch64MIR_TargetOp<"swpa", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SWPAW SWPAX
}

def AArch64MIR_swpb : AArch64MIR_TargetOp<"swpb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SWPB
}

def AArch64MIR_swph : AArch64MIR_TargetOp<"swph", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SWPH
}

def AArch64MIR_swplb : AArch64MIR_TargetOp<"swplb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SWPLB
}

def AArch64MIR_swplh : AArch64MIR_TargetOp<"swplh", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SWPLH
}

def AArch64MIR_swpl : AArch64MIR_TargetOp<"swpl", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SWPLW SWPLX
}

def AArch64MIR_swpp : AArch64MIR_TargetOp<"swpp", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SWPP
}

def AArch64MIR_swppa : AArch64MIR_TargetOp<"swppa", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SWPPA
}

def AArch64MIR_swppal : AArch64MIR_TargetOp<"swppal", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SWPPAL
}

def AArch64MIR_swppl : AArch64MIR_TargetOp<"swppl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SWPPL
}

def AArch64MIR_swptal : AArch64MIR_TargetOp<"swptal", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SWPTALW SWPTALX
}

def AArch64MIR_swpta : AArch64MIR_TargetOp<"swpta", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SWPTAW SWPTAX
}

def AArch64MIR_swptl : AArch64MIR_TargetOp<"swptl", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SWPTLW SWPTLX
}

def AArch64MIR_swpt : AArch64MIR_TargetOp<"swpt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SWPTW SWPTX
}

def AArch64MIR_swp : AArch64MIR_TargetOp<"swp", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SWPW SWPX
}

def AArch64MIR_sxtb : AArch64MIR_TargetOp<"sxtb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SXTB_ZPmZ_D SXTB_ZPmZ_H SXTB_ZPmZ_S SXTB_ZPzZ_D SXTB_ZPzZ_H SXTB_ZPzZ_S
}

def AArch64MIR_sxth : AArch64MIR_TargetOp<"sxth", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SXTH_ZPmZ_D SXTH_ZPmZ_S SXTH_ZPzZ_D SXTH_ZPzZ_S
}

def AArch64MIR_sxtw : AArch64MIR_TargetOp<"sxtw", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SXTW_ZPmZ_D SXTW_ZPzZ_D
}

def AArch64MIR_sysl : AArch64MIR_TargetOp<"sysl", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SYSLxt
}

def AArch64MIR_sysp : AArch64MIR_TargetOp<"sysp", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SYSPxt SYSPxt_XZR
}

def AArch64MIR_sys : AArch64MIR_TargetOp<"sys", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: SYSxt
}

def AArch64MIR_tblq : AArch64MIR_TargetOp<"tblq", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: TBLQ_ZZZ_B TBLQ_ZZZ_D TBLQ_ZZZ_H TBLQ_ZZZ_S
}

def AArch64MIR_tbl : AArch64MIR_TargetOp<"tbl", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: TBL_ZZZZ_B TBL_ZZZZ_D TBL_ZZZZ_H TBL_ZZZZ_S TBL_ZZZ_B TBL_ZZZ_D TBL_ZZZ_H TBL_ZZZ_S TBLv8i8Four TBLv8i8One TBLv8i8Three TBLv8i8Two TBLv16i8Four TBLv16i8One TBLv16i8Three TBLv16i8Two
}

def AArch64MIR_tbnz : AArch64MIR_TargetOp<"tbnz", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: TBNZW TBNZX
}

def AArch64MIR_tbxq : AArch64MIR_TargetOp<"tbxq", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: TBXQ_ZZZ_B TBXQ_ZZZ_D TBXQ_ZZZ_H TBXQ_ZZZ_S
}

def AArch64MIR_tbx : AArch64MIR_TargetOp<"tbx", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: TBX_ZZZ_B TBX_ZZZ_D TBX_ZZZ_H TBX_ZZZ_S TBXv8i8Four TBXv8i8One TBXv8i8Three TBXv8i8Two TBXv16i8Four TBXv16i8One TBXv16i8Three TBXv16i8Two
}

def AArch64MIR_tbz : AArch64MIR_TargetOp<"tbz", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: TBZW TBZX
}

def AArch64MIR_tchangeb : AArch64MIR_TargetOp<"tchangeb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: TCHANGEBri TCHANGEBrr
}

def AArch64MIR_tchangef : AArch64MIR_TargetOp<"tchangef", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: TCHANGEFri TCHANGEFrr
}

def AArch64MIR_tenter : AArch64MIR_TargetOp<"tenter", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: TENTER
}

def AArch64MIR_texit : AArch64MIR_TargetOp<"texit", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: TEXIT
}

def AArch64MIR_trn1 : AArch64MIR_TargetOp<"trn1", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: TRN1_PPP_B TRN1_PPP_D TRN1_PPP_H TRN1_PPP_S TRN1_ZZZ_B TRN1_ZZZ_D TRN1_ZZZ_H TRN1_ZZZ_Q TRN1_ZZZ_S TRN1v2i32 TRN1v2i64 TRN1v4i16 TRN1v4i32 TRN1v8i8 TRN1v8i16 TRN1v16i8
}

def AArch64MIR_trn2 : AArch64MIR_TargetOp<"trn2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: TRN2_PPP_B TRN2_PPP_D TRN2_PPP_H TRN2_PPP_S TRN2_ZZZ_B TRN2_ZZZ_D TRN2_ZZZ_H TRN2_ZZZ_Q TRN2_ZZZ_S TRN2v2i32 TRN2v2i64 TRN2v4i16 TRN2v4i32 TRN2v8i8 TRN2v8i16 TRN2v16i8
}

def AArch64MIR_uabalb : AArch64MIR_TargetOp<"uabalb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UABALB_ZZZ_D UABALB_ZZZ_H UABALB_ZZZ_S
}

def AArch64MIR_uabalt : AArch64MIR_TargetOp<"uabalt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UABALT_ZZZ_D UABALT_ZZZ_H UABALT_ZZZ_S
}

def AArch64MIR_uabal : AArch64MIR_TargetOp<"uabal", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UABAL_ZZZ_BtoH UABAL_ZZZ_HtoS UABAL_ZZZ_StoD UABALv2i32_v2i64 UABALv4i16_v4i32 UABALv8i8_v8i16
}

def AArch64MIR_uabal2 : AArch64MIR_TargetOp<"uabal2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UABALv4i32_v2i64 UABALv8i16_v4i32 UABALv16i8_v8i16
}

def AArch64MIR_uaba : AArch64MIR_TargetOp<"uaba", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UABA_ZZZ_B UABA_ZZZ_D UABA_ZZZ_H UABA_ZZZ_S UABAv2i32 UABAv4i16 UABAv4i32 UABAv8i8 UABAv8i16 UABAv16i8
}

def AArch64MIR_uabdlb : AArch64MIR_TargetOp<"uabdlb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UABDLB_ZZZ_D UABDLB_ZZZ_H UABDLB_ZZZ_S
}

def AArch64MIR_uabdlt : AArch64MIR_TargetOp<"uabdlt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UABDLT_ZZZ_D UABDLT_ZZZ_H UABDLT_ZZZ_S
}

def AArch64MIR_uabdl : AArch64MIR_TargetOp<"uabdl", [Pure, Commutative]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UABDLv2i32_v2i64 UABDLv4i16_v4i32 UABDLv8i8_v8i16
}

def AArch64MIR_uabdl2 : AArch64MIR_TargetOp<"uabdl2", [Pure, Commutative]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UABDLv4i32_v2i64 UABDLv8i16_v4i32 UABDLv16i8_v8i16
}

def AArch64MIR_uabd : AArch64MIR_TargetOp<"uabd", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UABD_ZPmZ_B UABD_ZPmZ_D UABD_ZPmZ_H UABD_ZPmZ_S UABDv2i32 UABDv4i16 UABDv4i32 UABDv8i8 UABDv8i16 UABDv16i8
}

def AArch64MIR_uadalp : AArch64MIR_TargetOp<"uadalp", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UADALP_ZPmZ_D UADALP_ZPmZ_H UADALP_ZPmZ_S UADALPv2i32_v1i64 UADALPv4i16_v2i32 UADALPv4i32_v2i64 UADALPv8i8_v4i16 UADALPv8i16_v4i32 UADALPv16i8_v8i16
}

def AArch64MIR_uaddlb : AArch64MIR_TargetOp<"uaddlb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UADDLB_ZZZ_D UADDLB_ZZZ_H UADDLB_ZZZ_S
}

def AArch64MIR_uaddlp : AArch64MIR_TargetOp<"uaddlp", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UADDLPv2i32_v1i64 UADDLPv4i16_v2i32 UADDLPv4i32_v2i64 UADDLPv8i8_v4i16 UADDLPv8i16_v4i32 UADDLPv16i8_v8i16
}

def AArch64MIR_uaddlt : AArch64MIR_TargetOp<"uaddlt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UADDLT_ZZZ_D UADDLT_ZZZ_H UADDLT_ZZZ_S
}

def AArch64MIR_uaddlv : AArch64MIR_TargetOp<"uaddlv", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UADDLVv4i16v UADDLVv4i32v UADDLVv8i8v UADDLVv8i16v UADDLVv16i8v
}

def AArch64MIR_uaddl : AArch64MIR_TargetOp<"uaddl", [Pure, Commutative]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UADDLv2i32_v2i64 UADDLv4i16_v4i32 UADDLv8i8_v8i16
}

def AArch64MIR_uaddl2 : AArch64MIR_TargetOp<"uaddl2", [Pure, Commutative]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UADDLv4i32_v2i64 UADDLv8i16_v4i32 UADDLv16i8_v8i16
}

def AArch64MIR_uaddv : AArch64MIR_TargetOp<"uaddv", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UADDV_VPZ_B UADDV_VPZ_D UADDV_VPZ_H UADDV_VPZ_S
}

def AArch64MIR_uaddwb : AArch64MIR_TargetOp<"uaddwb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UADDWB_ZZZ_D UADDWB_ZZZ_H UADDWB_ZZZ_S
}

def AArch64MIR_uaddwt : AArch64MIR_TargetOp<"uaddwt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UADDWT_ZZZ_D UADDWT_ZZZ_H UADDWT_ZZZ_S
}

def AArch64MIR_uaddw : AArch64MIR_TargetOp<"uaddw", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UADDWv2i32_v2i64 UADDWv4i16_v4i32 UADDWv8i8_v8i16
}

def AArch64MIR_uaddw2 : AArch64MIR_TargetOp<"uaddw2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UADDWv4i32_v2i64 UADDWv8i16_v4i32 UADDWv16i8_v8i16
}

def AArch64MIR_ubfm : AArch64MIR_TargetOp<"ubfm", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UBFMWri UBFMXri
}

def AArch64MIR_uclamp : AArch64MIR_TargetOp<"uclamp", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UCLAMP_VG2_2Z2Z_B UCLAMP_VG2_2Z2Z_D UCLAMP_VG2_2Z2Z_H UCLAMP_VG2_2Z2Z_S UCLAMP_VG4_4Z4Z_B UCLAMP_VG4_4Z4Z_D UCLAMP_VG4_4Z4Z_H UCLAMP_VG4_4Z4Z_S UCLAMP_ZZZ_B UCLAMP_ZZZ_D UCLAMP_ZZZ_H UCLAMP_ZZZ_S
}

def AArch64MIR_ucvtf : AArch64MIR_TargetOp<"ucvtf", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UCVTFDSr UCVTFHDr UCVTFHSr UCVTFSDr UCVTFSWDri UCVTFSWHri UCVTFSWSri UCVTFSXDri UCVTFSXHri UCVTFSXSri UCVTFUWDri UCVTFUWHri UCVTFUWSri UCVTFUXDri UCVTFUXHri UCVTFUXSri UCVTF_2Z2Z_StoS UCVTF_4Z4Z_StoS UCVTF_ZPmZ_DtoD UCVTF_ZPmZ_DtoH UCVTF_ZPmZ_DtoS UCVTF_ZPmZ_HtoH UCVTF_ZPmZ_StoD UCVTF_ZPmZ_StoH UCVTF_ZPmZ_StoS UCVTF_ZPzZ_DtoD UCVTF_ZPzZ_DtoH UCVTF_ZPzZ_DtoS UCVTF_ZPzZ_HtoH UCVTF_ZPzZ_StoD UCVTF_ZPzZ_StoH UCVTF_ZPzZ_StoS UCVTF_ZZ_BtoH UCVTF_ZZ_HtoS UCVTF_ZZ_StoD UCVTFd UCVTFh UCVTFs UCVTFv1i16 UCVTFv1i32 UCVTFv1i64 UCVTFv2f32 UCVTFv2f64 UCVTFv2i32_shift UCVTFv2i64_shift UCVTFv4f16 UCVTFv4f32 UCVTFv4i16_shift UCVTFv4i32_shift UCVTFv8f16 UCVTFv8i16_shift
}

def AArch64MIR_ucvtflt : AArch64MIR_TargetOp<"ucvtflt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UCVTFLT_ZZ_BtoH UCVTFLT_ZZ_HtoS UCVTFLT_ZZ_StoD
}

def AArch64MIR_udf : AArch64MIR_TargetOp<"udf", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UDF
}

def AArch64MIR_udivr : AArch64MIR_TargetOp<"udivr", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UDIVR_ZPmZ_D UDIVR_ZPmZ_S
}

def AArch64MIR_udiv : AArch64MIR_TargetOp<"udiv", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UDIVWr UDIVXr UDIV_ZPmZ_D UDIV_ZPmZ_S
}

def AArch64MIR_udot : AArch64MIR_TargetOp<"udot", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UDOT_VG2_M2Z2Z_BtoS UDOT_VG2_M2Z2Z_HtoD UDOT_VG2_M2Z2Z_HtoS UDOT_VG2_M2ZZI_BToS UDOT_VG2_M2ZZI_HToS UDOT_VG2_M2ZZI_HtoD UDOT_VG2_M2ZZ_BtoS UDOT_VG2_M2ZZ_HtoD UDOT_VG2_M2ZZ_HtoS UDOT_VG4_M4Z4Z_BtoS UDOT_VG4_M4Z4Z_HtoD UDOT_VG4_M4Z4Z_HtoS UDOT_VG4_M4ZZI_BtoS UDOT_VG4_M4ZZI_HToS UDOT_VG4_M4ZZI_HtoD UDOT_VG4_M4ZZ_BtoS UDOT_VG4_M4ZZ_HtoD UDOT_VG4_M4ZZ_HtoS UDOT_ZZZI_BtoH UDOT_ZZZI_BtoS UDOT_ZZZI_HtoD UDOT_ZZZI_HtoS UDOT_ZZZ_BtoH UDOT_ZZZ_BtoS UDOT_ZZZ_HtoD UDOT_ZZZ_HtoS UDOTlanev8i8 UDOTlanev16i8 UDOTv8i8 UDOTv16i8
}

def AArch64MIR_uhadd : AArch64MIR_TargetOp<"uhadd", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UHADD_ZPmZ_B UHADD_ZPmZ_D UHADD_ZPmZ_H UHADD_ZPmZ_S UHADDv2i32 UHADDv4i16 UHADDv4i32 UHADDv8i8 UHADDv8i16 UHADDv16i8
}

def AArch64MIR_uhsubr : AArch64MIR_TargetOp<"uhsubr", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UHSUBR_ZPmZ_B UHSUBR_ZPmZ_D UHSUBR_ZPmZ_H UHSUBR_ZPmZ_S
}

def AArch64MIR_uhsub : AArch64MIR_TargetOp<"uhsub", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UHSUB_ZPmZ_B UHSUB_ZPmZ_D UHSUB_ZPmZ_H UHSUB_ZPmZ_S UHSUBv2i32 UHSUBv4i16 UHSUBv4i32 UHSUBv8i8 UHSUBv8i16 UHSUBv16i8
}

def AArch64MIR_umaddl : AArch64MIR_TargetOp<"umaddl", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UMADDLrrr
}

def AArch64MIR_umaxp : AArch64MIR_TargetOp<"umaxp", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UMAXP_ZPmZ_B UMAXP_ZPmZ_D UMAXP_ZPmZ_H UMAXP_ZPmZ_S UMAXPv2i32 UMAXPv4i16 UMAXPv4i32 UMAXPv8i8 UMAXPv8i16 UMAXPv16i8
}

def AArch64MIR_umaxqv : AArch64MIR_TargetOp<"umaxqv", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UMAXQV_VPZ_B UMAXQV_VPZ_D UMAXQV_VPZ_H UMAXQV_VPZ_S
}

def AArch64MIR_umaxv : AArch64MIR_TargetOp<"umaxv", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UMAXV_VPZ_B UMAXV_VPZ_D UMAXV_VPZ_H UMAXV_VPZ_S UMAXVv4i16v UMAXVv4i32v UMAXVv8i8v UMAXVv8i16v UMAXVv16i8v
}

def AArch64MIR_umax : AArch64MIR_TargetOp<"umax", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UMAXWri UMAXWrr UMAXXri UMAXXrr UMAX_VG2_2Z2Z_B UMAX_VG2_2Z2Z_D UMAX_VG2_2Z2Z_H UMAX_VG2_2Z2Z_S UMAX_VG2_2ZZ_B UMAX_VG2_2ZZ_D UMAX_VG2_2ZZ_H UMAX_VG2_2ZZ_S UMAX_VG4_4Z4Z_B UMAX_VG4_4Z4Z_D UMAX_VG4_4Z4Z_H UMAX_VG4_4Z4Z_S UMAX_VG4_4ZZ_B UMAX_VG4_4ZZ_D UMAX_VG4_4ZZ_H UMAX_VG4_4ZZ_S UMAX_ZI_B UMAX_ZI_D UMAX_ZI_H UMAX_ZI_S UMAX_ZPmZ_B UMAX_ZPmZ_D UMAX_ZPmZ_H UMAX_ZPmZ_S UMAXv2i32 UMAXv4i16 UMAXv4i32 UMAXv8i8 UMAXv8i16 UMAXv16i8
}

def AArch64MIR_uminp : AArch64MIR_TargetOp<"uminp", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UMINP_ZPmZ_B UMINP_ZPmZ_D UMINP_ZPmZ_H UMINP_ZPmZ_S UMINPv2i32 UMINPv4i16 UMINPv4i32 UMINPv8i8 UMINPv8i16 UMINPv16i8
}

def AArch64MIR_uminqv : AArch64MIR_TargetOp<"uminqv", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UMINQV_VPZ_B UMINQV_VPZ_D UMINQV_VPZ_H UMINQV_VPZ_S
}

def AArch64MIR_uminv : AArch64MIR_TargetOp<"uminv", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UMINV_VPZ_B UMINV_VPZ_D UMINV_VPZ_H UMINV_VPZ_S UMINVv4i16v UMINVv4i32v UMINVv8i8v UMINVv8i16v UMINVv16i8v
}

def AArch64MIR_umin : AArch64MIR_TargetOp<"umin", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UMINWri UMINWrr UMINXri UMINXrr UMIN_VG2_2Z2Z_B UMIN_VG2_2Z2Z_D UMIN_VG2_2Z2Z_H UMIN_VG2_2Z2Z_S UMIN_VG2_2ZZ_B UMIN_VG2_2ZZ_D UMIN_VG2_2ZZ_H UMIN_VG2_2ZZ_S UMIN_VG4_4Z4Z_B UMIN_VG4_4Z4Z_D UMIN_VG4_4Z4Z_H UMIN_VG4_4Z4Z_S UMIN_VG4_4ZZ_B UMIN_VG4_4ZZ_D UMIN_VG4_4ZZ_H UMIN_VG4_4ZZ_S UMIN_ZI_B UMIN_ZI_D UMIN_ZI_H UMIN_ZI_S UMIN_ZPmZ_B UMIN_ZPmZ_D UMIN_ZPmZ_H UMIN_ZPmZ_S UMINv2i32 UMINv4i16 UMINv4i32 UMINv8i8 UMINv8i16 UMINv16i8
}

def AArch64MIR_umlalb : AArch64MIR_TargetOp<"umlalb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UMLALB_ZZZI_D UMLALB_ZZZI_S UMLALB_ZZZ_D UMLALB_ZZZ_H UMLALB_ZZZ_S
}

def AArch64MIR_umlall : AArch64MIR_TargetOp<"umlall", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UMLALL_MZZI_BtoS UMLALL_MZZI_HtoD UMLALL_MZZ_BtoS UMLALL_MZZ_HtoD UMLALL_VG2_M2Z2Z_BtoS UMLALL_VG2_M2Z2Z_HtoD UMLALL_VG2_M2ZZI_BtoS UMLALL_VG2_M2ZZI_HtoD UMLALL_VG2_M2ZZ_BtoS UMLALL_VG2_M2ZZ_HtoD UMLALL_VG4_M4Z4Z_BtoS UMLALL_VG4_M4Z4Z_HtoD UMLALL_VG4_M4ZZI_BtoS UMLALL_VG4_M4ZZI_HtoD UMLALL_VG4_M4ZZ_BtoS UMLALL_VG4_M4ZZ_HtoD
}

def AArch64MIR_umlalt : AArch64MIR_TargetOp<"umlalt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UMLALT_ZZZI_D UMLALT_ZZZI_S UMLALT_ZZZ_D UMLALT_ZZZ_H UMLALT_ZZZ_S
}

def AArch64MIR_umlal : AArch64MIR_TargetOp<"umlal", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UMLAL_MZZI_HtoS UMLAL_MZZ_HtoS UMLAL_VG2_M2Z2Z_HtoS UMLAL_VG2_M2ZZI_S UMLAL_VG2_M2ZZ_HtoS UMLAL_VG4_M4Z4Z_HtoS UMLAL_VG4_M4ZZI_HtoS UMLAL_VG4_M4ZZ_HtoS UMLALv2i32_indexed UMLALv2i32_v2i64 UMLALv4i16_indexed UMLALv4i16_v4i32 UMLALv8i8_v8i16
}

def AArch64MIR_umlal2 : AArch64MIR_TargetOp<"umlal2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UMLALv4i32_indexed UMLALv4i32_v2i64 UMLALv8i16_indexed UMLALv8i16_v4i32 UMLALv16i8_v8i16
}

def AArch64MIR_umlslb : AArch64MIR_TargetOp<"umlslb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UMLSLB_ZZZI_D UMLSLB_ZZZI_S UMLSLB_ZZZ_D UMLSLB_ZZZ_H UMLSLB_ZZZ_S
}

def AArch64MIR_umlsll : AArch64MIR_TargetOp<"umlsll", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UMLSLL_MZZI_BtoS UMLSLL_MZZI_HtoD UMLSLL_MZZ_BtoS UMLSLL_MZZ_HtoD UMLSLL_VG2_M2Z2Z_BtoS UMLSLL_VG2_M2Z2Z_HtoD UMLSLL_VG2_M2ZZI_BtoS UMLSLL_VG2_M2ZZI_HtoD UMLSLL_VG2_M2ZZ_BtoS UMLSLL_VG2_M2ZZ_HtoD UMLSLL_VG4_M4Z4Z_BtoS UMLSLL_VG4_M4Z4Z_HtoD UMLSLL_VG4_M4ZZI_BtoS UMLSLL_VG4_M4ZZI_HtoD UMLSLL_VG4_M4ZZ_BtoS UMLSLL_VG4_M4ZZ_HtoD
}

def AArch64MIR_umlslt : AArch64MIR_TargetOp<"umlslt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UMLSLT_ZZZI_D UMLSLT_ZZZI_S UMLSLT_ZZZ_D UMLSLT_ZZZ_H UMLSLT_ZZZ_S
}

def AArch64MIR_umlsl : AArch64MIR_TargetOp<"umlsl", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UMLSL_MZZI_HtoS UMLSL_MZZ_HtoS UMLSL_VG2_M2Z2Z_HtoS UMLSL_VG2_M2ZZI_S UMLSL_VG2_M2ZZ_HtoS UMLSL_VG4_M4Z4Z_HtoS UMLSL_VG4_M4ZZI_HtoS UMLSL_VG4_M4ZZ_HtoS UMLSLv2i32_indexed UMLSLv2i32_v2i64 UMLSLv4i16_indexed UMLSLv4i16_v4i32 UMLSLv8i8_v8i16
}

def AArch64MIR_umlsl2 : AArch64MIR_TargetOp<"umlsl2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UMLSLv4i32_indexed UMLSLv4i32_v2i64 UMLSLv8i16_indexed UMLSLv8i16_v4i32 UMLSLv16i8_v8i16
}

def AArch64MIR_ummla : AArch64MIR_TargetOp<"ummla", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UMMLA UMMLA_ZZZ
}

def AArch64MIR_umop4a : AArch64MIR_TargetOp<"umop4a", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UMOP4A_M2Z2Z_BToS UMOP4A_M2Z2Z_HToS UMOP4A_M2Z2Z_HtoD UMOP4A_M2ZZ_BToS UMOP4A_M2ZZ_HToS UMOP4A_M2ZZ_HtoD UMOP4A_MZ2Z_BToS UMOP4A_MZ2Z_HToS UMOP4A_MZ2Z_HtoD UMOP4A_MZZ_BToS UMOP4A_MZZ_HToS UMOP4A_MZZ_HtoD
}

def AArch64MIR_umop4s : AArch64MIR_TargetOp<"umop4s", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UMOP4S_M2Z2Z_BToS UMOP4S_M2Z2Z_HToS UMOP4S_M2Z2Z_HtoD UMOP4S_M2ZZ_BToS UMOP4S_M2ZZ_HToS UMOP4S_M2ZZ_HtoD UMOP4S_MZ2Z_BToS UMOP4S_MZ2Z_HToS UMOP4S_MZ2Z_HtoD UMOP4S_MZZ_BToS UMOP4S_MZZ_HToS UMOP4S_MZZ_HtoD
}

def AArch64MIR_umopa : AArch64MIR_TargetOp<"umopa", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UMOPA_MPPZZ_D UMOPA_MPPZZ_HtoS UMOPA_MPPZZ_S
}

def AArch64MIR_umops : AArch64MIR_TargetOp<"umops", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UMOPS_MPPZZ_D UMOPS_MPPZZ_HtoS UMOPS_MPPZZ_S
}

def AArch64MIR_umov : AArch64MIR_TargetOp<"umov", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UMOVvi8 UMOVvi8_idx0 UMOVvi16 UMOVvi16_idx0 UMOVvi32 UMOVvi32_idx0 UMOVvi64 UMOVvi64_idx0
}

def AArch64MIR_umsubl : AArch64MIR_TargetOp<"umsubl", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UMSUBLrrr
}

def AArch64MIR_umulh : AArch64MIR_TargetOp<"umulh", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UMULH_ZPmZ_B UMULH_ZPmZ_D UMULH_ZPmZ_H UMULH_ZPmZ_S UMULH_ZZZ_B UMULH_ZZZ_D UMULH_ZZZ_H UMULH_ZZZ_S UMULHrr
}

def AArch64MIR_umullb : AArch64MIR_TargetOp<"umullb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UMULLB_ZZZI_D UMULLB_ZZZI_S UMULLB_ZZZ_D UMULLB_ZZZ_H UMULLB_ZZZ_S
}

def AArch64MIR_umullt : AArch64MIR_TargetOp<"umullt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UMULLT_ZZZI_D UMULLT_ZZZI_S UMULLT_ZZZ_D UMULLT_ZZZ_H UMULLT_ZZZ_S
}

def AArch64MIR_umull : AArch64MIR_TargetOp<"umull", [Pure, Commutative]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UMULLv2i32_indexed UMULLv2i32_v2i64 UMULLv4i16_indexed UMULLv4i16_v4i32 UMULLv8i8_v8i16
}

def AArch64MIR_umull2 : AArch64MIR_TargetOp<"umull2", [Pure, Commutative]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UMULLv4i32_indexed UMULLv4i32_v2i64 UMULLv8i16_indexed UMULLv8i16_v4i32 UMULLv16i8_v8i16
}

def AArch64MIR_uqadd : AArch64MIR_TargetOp<"uqadd", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UQADD_ZI_B UQADD_ZI_D UQADD_ZI_H UQADD_ZI_S UQADD_ZPmZ_B UQADD_ZPmZ_D UQADD_ZPmZ_H UQADD_ZPmZ_S UQADD_ZZZ_B UQADD_ZZZ_D UQADD_ZZZ_H UQADD_ZZZ_S UQADDv1i8 UQADDv1i16 UQADDv1i32 UQADDv1i64 UQADDv2i32 UQADDv2i64 UQADDv4i16 UQADDv4i32 UQADDv8i8 UQADDv8i16 UQADDv16i8
}

def AArch64MIR_uqcvtn : AArch64MIR_TargetOp<"uqcvtn", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UQCVTN_Z2Z_StoH UQCVTN_Z4Z_DtoH UQCVTN_Z4Z_StoB
}

def AArch64MIR_uqcvt : AArch64MIR_TargetOp<"uqcvt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UQCVT_Z2Z_StoH UQCVT_Z4Z_DtoH UQCVT_Z4Z_StoB
}

def AArch64MIR_uqdecb : AArch64MIR_TargetOp<"uqdecb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UQDECB_WPiI UQDECB_XPiI
}

def AArch64MIR_uqdecd : AArch64MIR_TargetOp<"uqdecd", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UQDECD_WPiI UQDECD_XPiI UQDECD_ZPiI
}

def AArch64MIR_uqdech : AArch64MIR_TargetOp<"uqdech", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UQDECH_WPiI UQDECH_XPiI UQDECH_ZPiI
}

def AArch64MIR_uqdecp : AArch64MIR_TargetOp<"uqdecp", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UQDECP_WP_B UQDECP_WP_D UQDECP_WP_H UQDECP_WP_S UQDECP_XP_B UQDECP_XP_D UQDECP_XP_H UQDECP_XP_S UQDECP_ZP_D UQDECP_ZP_H UQDECP_ZP_S
}

def AArch64MIR_uqdecw : AArch64MIR_TargetOp<"uqdecw", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UQDECW_WPiI UQDECW_XPiI UQDECW_ZPiI
}

def AArch64MIR_uqincb : AArch64MIR_TargetOp<"uqincb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UQINCB_WPiI UQINCB_XPiI
}

def AArch64MIR_uqincd : AArch64MIR_TargetOp<"uqincd", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UQINCD_WPiI UQINCD_XPiI UQINCD_ZPiI
}

def AArch64MIR_uqinch : AArch64MIR_TargetOp<"uqinch", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UQINCH_WPiI UQINCH_XPiI UQINCH_ZPiI
}

def AArch64MIR_uqincp : AArch64MIR_TargetOp<"uqincp", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UQINCP_WP_B UQINCP_WP_D UQINCP_WP_H UQINCP_WP_S UQINCP_XP_B UQINCP_XP_D UQINCP_XP_H UQINCP_XP_S UQINCP_ZP_D UQINCP_ZP_H UQINCP_ZP_S
}

def AArch64MIR_uqincw : AArch64MIR_TargetOp<"uqincw", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UQINCW_WPiI UQINCW_XPiI UQINCW_ZPiI
}

def AArch64MIR_uqrshlr : AArch64MIR_TargetOp<"uqrshlr", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UQRSHLR_ZPmZ_B UQRSHLR_ZPmZ_D UQRSHLR_ZPmZ_H UQRSHLR_ZPmZ_S
}

def AArch64MIR_uqrshl : AArch64MIR_TargetOp<"uqrshl", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UQRSHL_ZPmZ_B UQRSHL_ZPmZ_D UQRSHL_ZPmZ_H UQRSHL_ZPmZ_S UQRSHLv1i8 UQRSHLv1i16 UQRSHLv1i32 UQRSHLv1i64 UQRSHLv2i32 UQRSHLv2i64 UQRSHLv4i16 UQRSHLv4i32 UQRSHLv8i8 UQRSHLv8i16 UQRSHLv16i8
}

def AArch64MIR_uqrshrnb : AArch64MIR_TargetOp<"uqrshrnb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UQRSHRNB_ZZI_B UQRSHRNB_ZZI_H UQRSHRNB_ZZI_S
}

def AArch64MIR_uqrshrnt : AArch64MIR_TargetOp<"uqrshrnt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UQRSHRNT_ZZI_B UQRSHRNT_ZZI_H UQRSHRNT_ZZI_S
}

def AArch64MIR_uqrshrn : AArch64MIR_TargetOp<"uqrshrn", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UQRSHRN_VG4_Z4ZI_B UQRSHRN_VG4_Z4ZI_H UQRSHRN_Z2ZI_HtoB UQRSHRN_Z2ZI_StoH UQRSHRNb UQRSHRNh UQRSHRNs UQRSHRNv2i32_shift UQRSHRNv4i16_shift UQRSHRNv8i8_shift
}

def AArch64MIR_uqrshrn2 : AArch64MIR_TargetOp<"uqrshrn2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UQRSHRNv4i32_shift UQRSHRNv8i16_shift UQRSHRNv16i8_shift
}

def AArch64MIR_uqrshr : AArch64MIR_TargetOp<"uqrshr", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UQRSHR_VG2_Z2ZI_H UQRSHR_VG4_Z4ZI_B UQRSHR_VG4_Z4ZI_H
}

def AArch64MIR_uqshlr : AArch64MIR_TargetOp<"uqshlr", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UQSHLR_ZPmZ_B UQSHLR_ZPmZ_D UQSHLR_ZPmZ_H UQSHLR_ZPmZ_S
}

def AArch64MIR_uqshl : AArch64MIR_TargetOp<"uqshl", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UQSHL_ZPmI_B UQSHL_ZPmI_D UQSHL_ZPmI_H UQSHL_ZPmI_S UQSHL_ZPmZ_B UQSHL_ZPmZ_D UQSHL_ZPmZ_H UQSHL_ZPmZ_S UQSHLb UQSHLd UQSHLh UQSHLs UQSHLv1i8 UQSHLv1i16 UQSHLv1i32 UQSHLv1i64 UQSHLv2i32 UQSHLv2i32_shift UQSHLv2i64 UQSHLv2i64_shift UQSHLv4i16 UQSHLv4i16_shift UQSHLv4i32 UQSHLv4i32_shift UQSHLv8i8 UQSHLv8i8_shift UQSHLv8i16 UQSHLv8i16_shift UQSHLv16i8 UQSHLv16i8_shift
}

def AArch64MIR_uqshrnb : AArch64MIR_TargetOp<"uqshrnb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UQSHRNB_ZZI_B UQSHRNB_ZZI_H UQSHRNB_ZZI_S
}

def AArch64MIR_uqshrnt : AArch64MIR_TargetOp<"uqshrnt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UQSHRNT_ZZI_B UQSHRNT_ZZI_H UQSHRNT_ZZI_S
}

def AArch64MIR_uqshrn : AArch64MIR_TargetOp<"uqshrn", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UQSHRN_Z2ZI_HtoB UQSHRN_Z2ZI_StoH UQSHRNb UQSHRNh UQSHRNs UQSHRNv2i32_shift UQSHRNv4i16_shift UQSHRNv8i8_shift
}

def AArch64MIR_uqshrn2 : AArch64MIR_TargetOp<"uqshrn2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UQSHRNv4i32_shift UQSHRNv8i16_shift UQSHRNv16i8_shift
}

def AArch64MIR_uqsubr : AArch64MIR_TargetOp<"uqsubr", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UQSUBR_ZPmZ_B UQSUBR_ZPmZ_D UQSUBR_ZPmZ_H UQSUBR_ZPmZ_S
}

def AArch64MIR_uqsub : AArch64MIR_TargetOp<"uqsub", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UQSUB_ZI_B UQSUB_ZI_D UQSUB_ZI_H UQSUB_ZI_S UQSUB_ZPmZ_B UQSUB_ZPmZ_D UQSUB_ZPmZ_H UQSUB_ZPmZ_S UQSUB_ZZZ_B UQSUB_ZZZ_D UQSUB_ZZZ_H UQSUB_ZZZ_S UQSUBv1i8 UQSUBv1i16 UQSUBv1i32 UQSUBv1i64 UQSUBv2i32 UQSUBv2i64 UQSUBv4i16 UQSUBv4i32 UQSUBv8i8 UQSUBv8i16 UQSUBv16i8
}

def AArch64MIR_uqxtnb : AArch64MIR_TargetOp<"uqxtnb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UQXTNB_ZZ_B UQXTNB_ZZ_H UQXTNB_ZZ_S
}

def AArch64MIR_uqxtnt : AArch64MIR_TargetOp<"uqxtnt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UQXTNT_ZZ_B UQXTNT_ZZ_H UQXTNT_ZZ_S
}

def AArch64MIR_uqxtn : AArch64MIR_TargetOp<"uqxtn", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UQXTNv1i8 UQXTNv1i16 UQXTNv1i32 UQXTNv2i32 UQXTNv4i16 UQXTNv8i8
}

def AArch64MIR_uqxtn2 : AArch64MIR_TargetOp<"uqxtn2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UQXTNv4i32 UQXTNv8i16 UQXTNv16i8
}

def AArch64MIR_urecpe : AArch64MIR_TargetOp<"urecpe", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: URECPE_ZPmZ_S URECPE_ZPzZ_S URECPEv2i32 URECPEv4i32
}

def AArch64MIR_urhadd : AArch64MIR_TargetOp<"urhadd", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: URHADD_ZPmZ_B URHADD_ZPmZ_D URHADD_ZPmZ_H URHADD_ZPmZ_S URHADDv2i32 URHADDv4i16 URHADDv4i32 URHADDv8i8 URHADDv8i16 URHADDv16i8
}

def AArch64MIR_urshlr : AArch64MIR_TargetOp<"urshlr", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: URSHLR_ZPmZ_B URSHLR_ZPmZ_D URSHLR_ZPmZ_H URSHLR_ZPmZ_S
}

def AArch64MIR_urshl : AArch64MIR_TargetOp<"urshl", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: URSHL_VG2_2Z2Z_B URSHL_VG2_2Z2Z_D URSHL_VG2_2Z2Z_H URSHL_VG2_2Z2Z_S URSHL_VG2_2ZZ_B URSHL_VG2_2ZZ_D URSHL_VG2_2ZZ_H URSHL_VG2_2ZZ_S URSHL_VG4_4Z4Z_B URSHL_VG4_4Z4Z_D URSHL_VG4_4Z4Z_H URSHL_VG4_4Z4Z_S URSHL_VG4_4ZZ_B URSHL_VG4_4ZZ_D URSHL_VG4_4ZZ_H URSHL_VG4_4ZZ_S URSHL_ZPmZ_B URSHL_ZPmZ_D URSHL_ZPmZ_H URSHL_ZPmZ_S URSHLv1i64 URSHLv2i32 URSHLv2i64 URSHLv4i16 URSHLv4i32 URSHLv8i8 URSHLv8i16 URSHLv16i8
}

def AArch64MIR_urshr : AArch64MIR_TargetOp<"urshr", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: URSHR_ZPmI_B URSHR_ZPmI_D URSHR_ZPmI_H URSHR_ZPmI_S URSHRd URSHRv2i32_shift URSHRv2i64_shift URSHRv4i16_shift URSHRv4i32_shift URSHRv8i8_shift URSHRv8i16_shift URSHRv16i8_shift
}

def AArch64MIR_ursqrte : AArch64MIR_TargetOp<"ursqrte", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: URSQRTE_ZPmZ_S URSQRTE_ZPzZ_S URSQRTEv2i32 URSQRTEv4i32
}

def AArch64MIR_ursra : AArch64MIR_TargetOp<"ursra", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: URSRA_ZZI_B URSRA_ZZI_D URSRA_ZZI_H URSRA_ZZI_S URSRAd URSRAv2i32_shift URSRAv2i64_shift URSRAv4i16_shift URSRAv4i32_shift URSRAv8i8_shift URSRAv8i16_shift URSRAv16i8_shift
}

def AArch64MIR_usdot : AArch64MIR_TargetOp<"usdot", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: USDOT_VG2_M2Z2Z_BToS USDOT_VG2_M2ZZI_BToS USDOT_VG2_M2ZZ_BToS USDOT_VG4_M4Z4Z_BToS USDOT_VG4_M4ZZI_BToS USDOT_VG4_M4ZZ_BToS USDOT_ZZZ USDOT_ZZZI USDOTlanev8i8 USDOTlanev16i8 USDOTv8i8 USDOTv16i8
}

def AArch64MIR_ushllb : AArch64MIR_TargetOp<"ushllb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: USHLLB_ZZI_D USHLLB_ZZI_H USHLLB_ZZI_S
}

def AArch64MIR_ushllt : AArch64MIR_TargetOp<"ushllt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: USHLLT_ZZI_D USHLLT_ZZI_H USHLLT_ZZI_S
}

def AArch64MIR_ushll : AArch64MIR_TargetOp<"ushll", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: USHLLv2i32_shift USHLLv4i16_shift USHLLv8i8_shift
}

def AArch64MIR_ushll2 : AArch64MIR_TargetOp<"ushll2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: USHLLv4i32_shift USHLLv8i16_shift USHLLv16i8_shift
}

def AArch64MIR_ushl : AArch64MIR_TargetOp<"ushl", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: USHLv1i64 USHLv2i32 USHLv2i64 USHLv4i16 USHLv4i32 USHLv8i8 USHLv8i16 USHLv16i8
}

def AArch64MIR_ushr : AArch64MIR_TargetOp<"ushr", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: USHRd USHRv2i32_shift USHRv2i64_shift USHRv4i16_shift USHRv4i32_shift USHRv8i8_shift USHRv8i16_shift USHRv16i8_shift
}

def AArch64MIR_usmlall : AArch64MIR_TargetOp<"usmlall", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: USMLALL_MZZI_BtoS USMLALL_MZZ_BtoS USMLALL_VG2_M2Z2Z_BtoS USMLALL_VG2_M2ZZI_BtoS USMLALL_VG2_M2ZZ_BtoS USMLALL_VG4_M4Z4Z_BtoS USMLALL_VG4_M4ZZI_BtoS USMLALL_VG4_M4ZZ_BtoS
}

def AArch64MIR_usmmla : AArch64MIR_TargetOp<"usmmla", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: USMMLA USMMLA_ZZZ
}

def AArch64MIR_usmop4a : AArch64MIR_TargetOp<"usmop4a", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: USMOP4A_M2Z2Z_BToS USMOP4A_M2Z2Z_HtoD USMOP4A_M2ZZ_BToS USMOP4A_M2ZZ_HtoD USMOP4A_MZ2Z_BToS USMOP4A_MZ2Z_HtoD USMOP4A_MZZ_BToS USMOP4A_MZZ_HtoD
}

def AArch64MIR_usmop4s : AArch64MIR_TargetOp<"usmop4s", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: USMOP4S_M2Z2Z_BToS USMOP4S_M2Z2Z_HtoD USMOP4S_M2ZZ_BToS USMOP4S_M2ZZ_HtoD USMOP4S_MZ2Z_BToS USMOP4S_MZ2Z_HtoD USMOP4S_MZZ_BToS USMOP4S_MZZ_HtoD
}

def AArch64MIR_usmopa : AArch64MIR_TargetOp<"usmopa", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: USMOPA_MPPZZ_D USMOPA_MPPZZ_S
}

def AArch64MIR_usmops : AArch64MIR_TargetOp<"usmops", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: USMOPS_MPPZZ_D USMOPS_MPPZZ_S
}

def AArch64MIR_usqadd : AArch64MIR_TargetOp<"usqadd", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: USQADD_ZPmZ_B USQADD_ZPmZ_D USQADD_ZPmZ_H USQADD_ZPmZ_S USQADDv1i8 USQADDv1i16 USQADDv1i32 USQADDv1i64 USQADDv2i32 USQADDv2i64 USQADDv4i16 USQADDv4i32 USQADDv8i8 USQADDv8i16 USQADDv16i8
}

def AArch64MIR_usra : AArch64MIR_TargetOp<"usra", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: USRA_ZZI_B USRA_ZZI_D USRA_ZZI_H USRA_ZZI_S USRAd USRAv2i32_shift USRAv2i64_shift USRAv4i16_shift USRAv4i32_shift USRAv8i8_shift USRAv8i16_shift USRAv16i8_shift
}

def AArch64MIR_ustmopa : AArch64MIR_TargetOp<"ustmopa", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: USTMOPA_M2ZZZI_BtoS
}

def AArch64MIR_usublb : AArch64MIR_TargetOp<"usublb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: USUBLB_ZZZ_D USUBLB_ZZZ_H USUBLB_ZZZ_S
}

def AArch64MIR_usublt : AArch64MIR_TargetOp<"usublt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: USUBLT_ZZZ_D USUBLT_ZZZ_H USUBLT_ZZZ_S
}

def AArch64MIR_usubl : AArch64MIR_TargetOp<"usubl", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: USUBLv2i32_v2i64 USUBLv4i16_v4i32 USUBLv8i8_v8i16
}

def AArch64MIR_usubl2 : AArch64MIR_TargetOp<"usubl2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: USUBLv4i32_v2i64 USUBLv8i16_v4i32 USUBLv16i8_v8i16
}

def AArch64MIR_usubwb : AArch64MIR_TargetOp<"usubwb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: USUBWB_ZZZ_D USUBWB_ZZZ_H USUBWB_ZZZ_S
}

def AArch64MIR_usubwt : AArch64MIR_TargetOp<"usubwt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: USUBWT_ZZZ_D USUBWT_ZZZ_H USUBWT_ZZZ_S
}

def AArch64MIR_usubw : AArch64MIR_TargetOp<"usubw", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: USUBWv2i32_v2i64 USUBWv4i16_v4i32 USUBWv8i8_v8i16
}

def AArch64MIR_usubw2 : AArch64MIR_TargetOp<"usubw2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: USUBWv4i32_v2i64 USUBWv8i16_v4i32 USUBWv16i8_v8i16
}

def AArch64MIR_usvdot : AArch64MIR_TargetOp<"usvdot", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: USVDOT_VG4_M4ZZI_BToS
}

def AArch64MIR_utmopa : AArch64MIR_TargetOp<"utmopa", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UTMOPA_M2ZZZI_BtoS UTMOPA_M2ZZZI_HtoS
}

def AArch64MIR_uunpkhi : AArch64MIR_TargetOp<"uunpkhi", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UUNPKHI_ZZ_D UUNPKHI_ZZ_H UUNPKHI_ZZ_S
}

def AArch64MIR_uunpklo : AArch64MIR_TargetOp<"uunpklo", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UUNPKLO_ZZ_D UUNPKLO_ZZ_H UUNPKLO_ZZ_S
}

def AArch64MIR_uunpk : AArch64MIR_TargetOp<"uunpk", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UUNPK_VG2_2ZZ_D UUNPK_VG2_2ZZ_H UUNPK_VG2_2ZZ_S UUNPK_VG4_4Z2Z_D UUNPK_VG4_4Z2Z_H UUNPK_VG4_4Z2Z_S
}

def AArch64MIR_uvdot : AArch64MIR_TargetOp<"uvdot", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UVDOT_VG2_M2ZZI_HtoS UVDOT_VG4_M4ZZI_BtoS UVDOT_VG4_M4ZZI_HtoD
}

def AArch64MIR_uxtb : AArch64MIR_TargetOp<"uxtb", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UXTB_ZPmZ_D UXTB_ZPmZ_H UXTB_ZPmZ_S UXTB_ZPzZ_D UXTB_ZPzZ_H UXTB_ZPzZ_S
}

def AArch64MIR_uxth : AArch64MIR_TargetOp<"uxth", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UXTH_ZPmZ_D UXTH_ZPmZ_S UXTH_ZPzZ_D UXTH_ZPzZ_S
}

def AArch64MIR_uxtw : AArch64MIR_TargetOp<"uxtw", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UXTW_ZPmZ_D UXTW_ZPzZ_D
}

def AArch64MIR_uzp1 : AArch64MIR_TargetOp<"uzp1", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UZP1_PPP_B UZP1_PPP_D UZP1_PPP_H UZP1_PPP_S UZP1_ZZZ_B UZP1_ZZZ_D UZP1_ZZZ_H UZP1_ZZZ_Q UZP1_ZZZ_S UZP1v2i32 UZP1v2i64 UZP1v4i16 UZP1v4i32 UZP1v8i8 UZP1v8i16 UZP1v16i8
}

def AArch64MIR_uzp2 : AArch64MIR_TargetOp<"uzp2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UZP2_PPP_B UZP2_PPP_D UZP2_PPP_H UZP2_PPP_S UZP2_ZZZ_B UZP2_ZZZ_D UZP2_ZZZ_H UZP2_ZZZ_Q UZP2_ZZZ_S UZP2v2i32 UZP2v2i64 UZP2v4i16 UZP2v4i32 UZP2v8i8 UZP2v8i16 UZP2v16i8
}

def AArch64MIR_uzpq1 : AArch64MIR_TargetOp<"uzpq1", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UZPQ1_ZZZ_B UZPQ1_ZZZ_D UZPQ1_ZZZ_H UZPQ1_ZZZ_S
}

def AArch64MIR_uzpq2 : AArch64MIR_TargetOp<"uzpq2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UZPQ2_ZZZ_B UZPQ2_ZZZ_D UZPQ2_ZZZ_H UZPQ2_ZZZ_S
}

def AArch64MIR_uzp : AArch64MIR_TargetOp<"uzp", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: UZP_VG2_2ZZZ_B UZP_VG2_2ZZZ_D UZP_VG2_2ZZZ_H UZP_VG2_2ZZZ_Q UZP_VG2_2ZZZ_S UZP_VG4_4Z4Z_B UZP_VG4_4Z4Z_D UZP_VG4_4Z4Z_H UZP_VG4_4Z4Z_Q UZP_VG4_4Z4Z_S
}

def AArch64MIR_wfet : AArch64MIR_TargetOp<"wfet", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: WFET
}

def AArch64MIR_wfit : AArch64MIR_TargetOp<"wfit", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: WFIT
}

def AArch64MIR_whilege : AArch64MIR_TargetOp<"whilege", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: WHILEGE_2PXX_B WHILEGE_2PXX_D WHILEGE_2PXX_H WHILEGE_2PXX_S WHILEGE_CXX_B WHILEGE_CXX_D WHILEGE_CXX_H WHILEGE_CXX_S WHILEGE_PWW_B WHILEGE_PWW_D WHILEGE_PWW_H WHILEGE_PWW_S WHILEGE_PXX_B WHILEGE_PXX_D WHILEGE_PXX_H WHILEGE_PXX_S
}

def AArch64MIR_whilegt : AArch64MIR_TargetOp<"whilegt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: WHILEGT_2PXX_B WHILEGT_2PXX_D WHILEGT_2PXX_H WHILEGT_2PXX_S WHILEGT_CXX_B WHILEGT_CXX_D WHILEGT_CXX_H WHILEGT_CXX_S WHILEGT_PWW_B WHILEGT_PWW_D WHILEGT_PWW_H WHILEGT_PWW_S WHILEGT_PXX_B WHILEGT_PXX_D WHILEGT_PXX_H WHILEGT_PXX_S
}

def AArch64MIR_whilehi : AArch64MIR_TargetOp<"whilehi", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: WHILEHI_2PXX_B WHILEHI_2PXX_D WHILEHI_2PXX_H WHILEHI_2PXX_S WHILEHI_CXX_B WHILEHI_CXX_D WHILEHI_CXX_H WHILEHI_CXX_S WHILEHI_PWW_B WHILEHI_PWW_D WHILEHI_PWW_H WHILEHI_PWW_S WHILEHI_PXX_B WHILEHI_PXX_D WHILEHI_PXX_H WHILEHI_PXX_S
}

def AArch64MIR_whilehs : AArch64MIR_TargetOp<"whilehs", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: WHILEHS_2PXX_B WHILEHS_2PXX_D WHILEHS_2PXX_H WHILEHS_2PXX_S WHILEHS_CXX_B WHILEHS_CXX_D WHILEHS_CXX_H WHILEHS_CXX_S WHILEHS_PWW_B WHILEHS_PWW_D WHILEHS_PWW_H WHILEHS_PWW_S WHILEHS_PXX_B WHILEHS_PXX_D WHILEHS_PXX_H WHILEHS_PXX_S
}

def AArch64MIR_whilele : AArch64MIR_TargetOp<"whilele", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: WHILELE_2PXX_B WHILELE_2PXX_D WHILELE_2PXX_H WHILELE_2PXX_S WHILELE_CXX_B WHILELE_CXX_D WHILELE_CXX_H WHILELE_CXX_S WHILELE_PWW_B WHILELE_PWW_D WHILELE_PWW_H WHILELE_PWW_S WHILELE_PXX_B WHILELE_PXX_D WHILELE_PXX_H WHILELE_PXX_S
}

def AArch64MIR_whilelo : AArch64MIR_TargetOp<"whilelo", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: WHILELO_2PXX_B WHILELO_2PXX_D WHILELO_2PXX_H WHILELO_2PXX_S WHILELO_CXX_B WHILELO_CXX_D WHILELO_CXX_H WHILELO_CXX_S WHILELO_PWW_B WHILELO_PWW_D WHILELO_PWW_H WHILELO_PWW_S WHILELO_PXX_B WHILELO_PXX_D WHILELO_PXX_H WHILELO_PXX_S
}

def AArch64MIR_whilels : AArch64MIR_TargetOp<"whilels", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: WHILELS_2PXX_B WHILELS_2PXX_D WHILELS_2PXX_H WHILELS_2PXX_S WHILELS_CXX_B WHILELS_CXX_D WHILELS_CXX_H WHILELS_CXX_S WHILELS_PWW_B WHILELS_PWW_D WHILELS_PWW_H WHILELS_PWW_S WHILELS_PXX_B WHILELS_PXX_D WHILELS_PXX_H WHILELS_PXX_S
}

def AArch64MIR_whilelt : AArch64MIR_TargetOp<"whilelt", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: WHILELT_2PXX_B WHILELT_2PXX_D WHILELT_2PXX_H WHILELT_2PXX_S WHILELT_CXX_B WHILELT_CXX_D WHILELT_CXX_H WHILELT_CXX_S WHILELT_PWW_B WHILELT_PWW_D WHILELT_PWW_H WHILELT_PWW_S WHILELT_PXX_B WHILELT_PXX_D WHILELT_PXX_H WHILELT_PXX_S
}

def AArch64MIR_whilerw : AArch64MIR_TargetOp<"whilerw", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: WHILERW_PXX_B WHILERW_PXX_D WHILERW_PXX_H WHILERW_PXX_S
}

def AArch64MIR_whilewr : AArch64MIR_TargetOp<"whilewr", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: WHILEWR_PXX_B WHILEWR_PXX_D WHILEWR_PXX_H WHILEWR_PXX_S
}

def AArch64MIR_wrffr : AArch64MIR_TargetOp<"wrffr", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: WRFFR
}

def AArch64MIR_xaflag : AArch64MIR_TargetOp<"xaflag", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: XAFLAG
}

def AArch64MIR_xar : AArch64MIR_TargetOp<"xar", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: XAR XAR_ZZZI_B XAR_ZZZI_D XAR_ZZZI_H XAR_ZZZI_S
}

def AArch64MIR_xpacd : AArch64MIR_TargetOp<"xpacd", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: XPACD
}

def AArch64MIR_xpaci : AArch64MIR_TargetOp<"xpaci", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: XPACI
}

def AArch64MIR_xtn : AArch64MIR_TargetOp<"xtn", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: XTNv2i32 XTNv4i16 XTNv8i8
}

def AArch64MIR_xtn2 : AArch64MIR_TargetOp<"xtn2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: XTNv4i32 XTNv8i16 XTNv16i8
}

def AArch64MIR_zero : AArch64MIR_TargetOp<"zero", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ZERO_M ZERO_MXI_2Z ZERO_MXI_4Z ZERO_MXI_VG2_2Z ZERO_MXI_VG2_4Z ZERO_MXI_VG2_Z ZERO_MXI_VG4_2Z ZERO_MXI_VG4_4Z ZERO_MXI_VG4_Z ZERO_T
}

def AArch64MIR_zip1 : AArch64MIR_TargetOp<"zip1", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ZIP1_PPP_B ZIP1_PPP_D ZIP1_PPP_H ZIP1_PPP_S ZIP1_ZZZ_B ZIP1_ZZZ_D ZIP1_ZZZ_H ZIP1_ZZZ_Q ZIP1_ZZZ_S ZIP1v2i32 ZIP1v2i64 ZIP1v4i16 ZIP1v4i32 ZIP1v8i8 ZIP1v8i16 ZIP1v16i8
}

def AArch64MIR_zip2 : AArch64MIR_TargetOp<"zip2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ZIP2_PPP_B ZIP2_PPP_D ZIP2_PPP_H ZIP2_PPP_S ZIP2_ZZZ_B ZIP2_ZZZ_D ZIP2_ZZZ_H ZIP2_ZZZ_Q ZIP2_ZZZ_S ZIP2v2i32 ZIP2v2i64 ZIP2v4i16 ZIP2v4i32 ZIP2v8i8 ZIP2v8i16 ZIP2v16i8
}

def AArch64MIR_zipq1 : AArch64MIR_TargetOp<"zipq1", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ZIPQ1_ZZZ_B ZIPQ1_ZZZ_D ZIPQ1_ZZZ_H ZIPQ1_ZZZ_S
}

def AArch64MIR_zipq2 : AArch64MIR_TargetOp<"zipq2", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ZIPQ2_ZZZ_B ZIPQ2_ZZZ_D ZIPQ2_ZZZ_H ZIPQ2_ZZZ_S
}

def AArch64MIR_zip : AArch64MIR_TargetOp<"zip", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,
                       OptionalAttr<StrAttr>:$variant);
  let results = (outs Variadic<MIR_AnyLLT>:$rets);
  // opcodes: ZIP_VG2_2ZZZ_B ZIP_VG2_2ZZZ_D ZIP_VG2_2ZZZ_H ZIP_VG2_2ZZZ_Q ZIP_VG2_2ZZZ_S ZIP_VG4_4Z4Z_B ZIP_VG4_4Z4Z_D ZIP_VG4_4Z4Z_H ZIP_VG4_4Z4Z_Q ZIP_VG4_4Z4Z_S
}

// Considered 7484 instructions, emitted 1551 ops across 1551 mnemonics.
