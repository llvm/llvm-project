//===---- SMInstructions.td - Scalar Memory Instruction Definitions -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

def smrd_offset_8 : ImmOperand<i32, "SMRDOffset8", 1>;

let OperandType = "OPERAND_IMMEDIATE",
    EncoderMethod = "getSMEMOffsetEncoding",
    DecoderMethod = "decodeSMEMOffset" in {
def smem_offset : ImmOperand<i32, "SMEMOffset", 1>;
def smem_offset_mod : NamedIntOperand<i32, "offset", "SMEMOffsetMod">;
}

//===----------------------------------------------------------------------===//
// Scalar Memory classes
//===----------------------------------------------------------------------===//

class SM_Pseudo <string opName, dag outs, dag ins, string asmOps, list<dag> pattern=[]> :
  InstSI <outs, ins, "", pattern>,
  SIMCInstr<opName, SIEncodingFamily.NONE> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;

  let LGKM_CNT = 1;
  let SMRD = 1;
  let mayStore = 0;
  let mayLoad = 1;
  let hasSideEffects = 0;
  let UseNamedOperandTable = 1;
  let SchedRW = [WriteSMEM];

  string Mnemonic = opName;
  string AsmOperands = asmOps;

  bits<1> has_sbase = 1;
  bits<1> has_sdst = 1;
  bit has_glc = 0;
  bit has_dlc = 0;
  bit has_offset = 0;
  bit has_soffset = 0;
  bit is_buffer = 0;
}

class SM_Real <SM_Pseudo ps, string opName = ps.Mnemonic>
  : InstSI<ps.OutOperandList, ps.InOperandList, opName # ps.AsmOperands> {

  let isPseudo = 0;
  let isCodeGenOnly = 0;

  Instruction Opcode = !cast<Instruction>(NAME);

  // copy relevant pseudo op flags
  let LGKM_CNT             = ps.LGKM_CNT;
  let SMRD                 = ps.SMRD;
  let mayStore             = ps.mayStore;
  let mayLoad              = ps.mayLoad;
  let hasSideEffects       = ps.hasSideEffects;
  let UseNamedOperandTable = ps.UseNamedOperandTable;
  let SchedRW              = ps.SchedRW;
  let SubtargetPredicate   = ps.SubtargetPredicate;
  let AsmMatchConverter    = ps.AsmMatchConverter;
  let IsAtomicRet          = ps.IsAtomicRet;
  let IsAtomicNoRet        = ps.IsAtomicNoRet;

  let TSFlags = ps.TSFlags;

  bit is_buffer = ps.is_buffer;

  // encoding
  bits<7>  sbase;
  bits<7>  sdst;
  bits<32> offset;
  bits<8>  soffset;
  bits<5> cpol;
}

class OffsetMode<bit hasOffset, bit hasSOffset, string variant,
                 dag ins, string asm> {
  bit HasOffset = hasOffset;
  bit HasSOffset = hasSOffset;
  string Variant = variant;
  dag Ins = ins;
  string Asm = asm;
}

def IMM_Offset : OffsetMode<1, 0, "_IMM", (ins smem_offset:$offset), "$offset">;
def SGPR_Offset : OffsetMode<0, 1, "_SGPR", (ins SReg_32:$soffset), "$soffset">;
def SGPR_IMM_Offset : OffsetMode<1, 1, "_SGPR_IMM",
                                 (ins SReg_32:$soffset, smem_offset_mod:$offset),
                                 "$soffset$offset">;

class SM_Probe_Pseudo <string opName, RegisterClass baseClass, OffsetMode offsets>
  : SM_Pseudo<opName, (outs),
              !con((ins i8imm:$sdata, baseClass:$sbase), offsets.Ins),
              " $sdata, $sbase, " # offsets.Asm> {
  let mayLoad = 0;
  let mayStore = 0;
  let has_glc = 0;
  let LGKM_CNT = 0;
  let ScalarStore = 0;
  let hasSideEffects = 1;
  let has_offset = offsets.HasOffset;
  let has_soffset = offsets.HasSOffset;
  let PseudoInstr = opName # offsets.Variant;
}

class SM_Load_Pseudo <string opName, RegisterClass baseClass,
                      RegisterClass dstClass, OffsetMode offsets>
  : SM_Pseudo<opName, (outs dstClass:$sdst),
              !con((ins baseClass:$sbase), offsets.Ins, (ins CPol:$cpol)),
              " $sdst, $sbase, " # offsets.Asm # "$cpol", []> {
  RegisterClass BaseClass = baseClass;
  let mayLoad = 1;
  let mayStore = 0;
  let has_glc = 1;
  let has_dlc = 1;
  let has_offset = offsets.HasOffset;
  let has_soffset = offsets.HasSOffset;
  let PseudoInstr = opName # offsets.Variant;
}

class SM_Store_Pseudo <string opName, RegisterClass baseClass,
                       RegisterClass srcClass, OffsetMode offsets>
  : SM_Pseudo<opName, (outs), !con((ins srcClass:$sdata, baseClass:$sbase),
                                   offsets.Ins, (ins CPol:$cpol)),
              " $sdata, $sbase, " # offsets.Asm # "$cpol"> {
  RegisterClass BaseClass = baseClass;
  let mayLoad = 0;
  let mayStore = 1;
  let has_glc = 1;
  let has_dlc = 1;
  let has_offset = offsets.HasOffset;
  let has_soffset = offsets.HasSOffset;
  let ScalarStore = 1;
  let PseudoInstr = opName # offsets.Variant;
}

class SM_Discard_Pseudo <string opName, OffsetMode offsets>
  : SM_Pseudo<opName, (outs), !con((ins SReg_64:$sbase), offsets.Ins),
              " $sbase, " # offsets.Asm> {
  let mayLoad = 0;
  let mayStore = 0;
  let has_glc = 0;
  let has_sdst = 0;
  let ScalarStore = 0;
  let hasSideEffects = 1;
  let has_offset = offsets.HasOffset;
  let has_soffset = offsets.HasSOffset;
  let PseudoInstr = opName # offsets.Variant;
}

multiclass SM_Pseudo_Loads<RegisterClass baseClass,
                           RegisterClass dstClass> {
  defvar opName = !tolower(NAME);
  def _IMM : SM_Load_Pseudo <opName, baseClass, dstClass, IMM_Offset>;
  def _SGPR : SM_Load_Pseudo <opName, baseClass, dstClass, SGPR_Offset>;
  def _SGPR_IMM : SM_Load_Pseudo <opName, baseClass, dstClass, SGPR_IMM_Offset>;
}

multiclass SM_Pseudo_Stores<RegisterClass baseClass,
                            RegisterClass srcClass> {
  defvar opName = !tolower(NAME);
  def _IMM : SM_Store_Pseudo <opName, baseClass, srcClass, IMM_Offset>;
  def _SGPR : SM_Store_Pseudo <opName, baseClass, srcClass, SGPR_Offset>;
  def _SGPR_IMM : SM_Store_Pseudo <opName, baseClass, srcClass, SGPR_IMM_Offset>;
}

multiclass SM_Pseudo_Discards {
  defvar opName = !tolower(NAME);
  def _IMM : SM_Discard_Pseudo <opName, IMM_Offset>;
  def _SGPR : SM_Discard_Pseudo <opName, SGPR_Offset>;
  def _SGPR_IMM : SM_Discard_Pseudo <opName, SGPR_IMM_Offset>;
}

class SM_Time_Pseudo<string opName, SDPatternOperator node = null_frag> : SM_Pseudo<
  opName, (outs SReg_64_XEXEC:$sdst), (ins),
  " $sdst", [(set i64:$sdst, (node))]> {
  let hasSideEffects = 1;

  let mayStore = 0;
  let mayLoad = 0;
  let has_sbase = 0;
}

class SM_Inval_Pseudo <string opName, SDPatternOperator node = null_frag> : SM_Pseudo<
  opName, (outs), (ins), "", [(node)]> {
  let hasSideEffects = 1;
  let mayLoad = 0;
  let mayStore = 0;
  let has_sdst = 0;
  let has_sbase = 0;
}

multiclass SM_Pseudo_Probe<RegisterClass baseClass> {
  defvar opName = !tolower(NAME);
  def _IMM  : SM_Probe_Pseudo <opName, baseClass, IMM_Offset>;
  def _SGPR : SM_Probe_Pseudo <opName, baseClass, SGPR_Offset>;
  def _SGPR_IMM : SM_Probe_Pseudo <opName, baseClass, SGPR_IMM_Offset>;
}

class SM_WaveId_Pseudo<string opName, SDPatternOperator node> : SM_Pseudo<
  opName, (outs SReg_32_XM0_XEXEC:$sdst), (ins),
  " $sdst", [(set i32:$sdst, (node))]> {
  let hasSideEffects = 1;
  let mayStore = 0;
  let mayLoad = 0;
  let has_sbase = 0;
}

//===----------------------------------------------------------------------===//
// Scalar Atomic Memory Classes
//===----------------------------------------------------------------------===//

class SM_Atomic_Pseudo <string opName,
                        dag outs, dag ins, string asmOps, bit isRet>
  : SM_Pseudo<opName, outs, ins, asmOps, []> {

  bit glc = isRet;

  let mayLoad = 1;
  let mayStore = 1;
  let has_glc = 1;
  let has_dlc = 1;
  let has_soffset = 1;

  // Should these be set?
  let ScalarStore = 1;
  let hasSideEffects = 1;
  let maybeAtomic = 1;

  let IsAtomicNoRet = !not(isRet);
  let IsAtomicRet = isRet;

  let AsmMatchConverter = "cvtSMEMAtomic";
}

class SM_Pseudo_Atomic<string opName,
                       RegisterClass baseClass,
                       RegisterClass dataClass,
                       OffsetMode offsets,
                       bit isRet,
                       string opNameWithSuffix =
                         opName # offsets.Variant # !if(isRet, "_RTN", ""),
                       Operand CPolTy = !if(isRet, CPol_GLC1, CPol)> :
  SM_Atomic_Pseudo<opName,
                   !if(isRet, (outs dataClass:$sdst), (outs)),
                   !con((ins dataClass:$sdata, baseClass:$sbase), offsets.Ins,
                        (ins CPolTy:$cpol)),
                   !if(isRet, " $sdst", " $sdata") #
                     ", $sbase, " # offsets.Asm # "$cpol",
                   isRet>,
  AtomicNoRet <opNameWithSuffix, isRet> {
  let has_offset = offsets.HasOffset;
  let has_soffset = offsets.HasSOffset;
  let PseudoInstr = opNameWithSuffix;

  let Constraints = !if(isRet, "$sdst = $sdata", "");
  let DisableEncoding = !if(isRet, "$sdata", "");
}

multiclass SM_Pseudo_Atomics<RegisterClass baseClass,
                             RegisterClass dataClass> {
  defvar opName = !tolower(NAME);
  def _IMM      : SM_Pseudo_Atomic <opName, baseClass, dataClass, IMM_Offset, 0>;
  def _SGPR     : SM_Pseudo_Atomic <opName, baseClass, dataClass, SGPR_Offset, 0>;
  def _SGPR_IMM : SM_Pseudo_Atomic <opName, baseClass, dataClass, SGPR_IMM_Offset, 0>;
  def _IMM_RTN  : SM_Pseudo_Atomic <opName, baseClass, dataClass, IMM_Offset, 1>;
  def _SGPR_RTN : SM_Pseudo_Atomic <opName, baseClass, dataClass, SGPR_Offset, 1>;
  def _SGPR_IMM_RTN : SM_Pseudo_Atomic <opName, baseClass, dataClass, SGPR_IMM_Offset, 1>;
}

//===----------------------------------------------------------------------===//
// Scalar Memory Instructions
//===----------------------------------------------------------------------===//

// We are using the SReg_32_XM0 and not the SReg_32 register class for 32-bit
// SMRD instructions, because the SReg_32_XM0 register class does not include M0
// and writing to M0 from an SMRD instruction will hang the GPU.

// XXX - SMEM instructions do not allow exec for data operand, but
// does sdst for SMRD on SI/CI?
defm S_LOAD_DWORD    : SM_Pseudo_Loads <SReg_64, SReg_32_XM0_XEXEC>;
defm S_LOAD_DWORDX2  : SM_Pseudo_Loads <SReg_64, SReg_64_XEXEC>;
defm S_LOAD_DWORDX4  : SM_Pseudo_Loads <SReg_64, SReg_128>;
defm S_LOAD_DWORDX8  : SM_Pseudo_Loads <SReg_64, SReg_256>;
defm S_LOAD_DWORDX16 : SM_Pseudo_Loads <SReg_64, SReg_512>;

let is_buffer = 1 in {
defm S_BUFFER_LOAD_DWORD : SM_Pseudo_Loads <SReg_128, SReg_32_XM0_XEXEC>;
// FIXME: exec_lo/exec_hi appear to be allowed for SMRD loads on
// SI/CI, bit disallowed for SMEM on VI.
defm S_BUFFER_LOAD_DWORDX2 : SM_Pseudo_Loads <SReg_128, SReg_64_XEXEC>;
defm S_BUFFER_LOAD_DWORDX4 : SM_Pseudo_Loads <SReg_128, SReg_128>;
defm S_BUFFER_LOAD_DWORDX8 : SM_Pseudo_Loads <SReg_128, SReg_256>;
defm S_BUFFER_LOAD_DWORDX16 : SM_Pseudo_Loads <SReg_128, SReg_512>;
}

let SubtargetPredicate = HasScalarStores in {
defm S_STORE_DWORD : SM_Pseudo_Stores <SReg_64, SReg_32_XM0_XEXEC>;
defm S_STORE_DWORDX2 : SM_Pseudo_Stores <SReg_64, SReg_64_XEXEC>;
defm S_STORE_DWORDX4 : SM_Pseudo_Stores <SReg_64, SReg_128>;

let is_buffer = 1 in {
defm S_BUFFER_STORE_DWORD : SM_Pseudo_Stores <SReg_128, SReg_32_XM0_XEXEC>;
defm S_BUFFER_STORE_DWORDX2 : SM_Pseudo_Stores <SReg_128, SReg_64_XEXEC>;
defm S_BUFFER_STORE_DWORDX4 : SM_Pseudo_Stores <SReg_128, SReg_128>;
}
} // End SubtargetPredicate = HasScalarStores

let SubtargetPredicate = HasSMemTimeInst in
def S_MEMTIME : SM_Time_Pseudo <"s_memtime", int_amdgcn_s_memtime>;
def S_DCACHE_INV : SM_Inval_Pseudo <"s_dcache_inv", int_amdgcn_s_dcache_inv>;

let SubtargetPredicate = isGFX7GFX8GFX9 in {
def S_DCACHE_INV_VOL : SM_Inval_Pseudo <"s_dcache_inv_vol", int_amdgcn_s_dcache_inv_vol>;
} // let SubtargetPredicate = isGFX7GFX8GFX9

let SubtargetPredicate = isGFX8Plus in {
let OtherPredicates = [HasScalarStores] in {
def S_DCACHE_WB     : SM_Inval_Pseudo <"s_dcache_wb", int_amdgcn_s_dcache_wb>;
def S_DCACHE_WB_VOL : SM_Inval_Pseudo <"s_dcache_wb_vol", int_amdgcn_s_dcache_wb_vol>;
} // End OtherPredicates = [HasScalarStores]

defm S_ATC_PROBE        : SM_Pseudo_Probe <SReg_64>;
let is_buffer = 1 in {
defm S_ATC_PROBE_BUFFER : SM_Pseudo_Probe <SReg_128>;
}
} // SubtargetPredicate = isGFX8Plus

let SubtargetPredicate = HasSMemRealTime in
def S_MEMREALTIME   : SM_Time_Pseudo <"s_memrealtime", int_amdgcn_s_memrealtime>;

let SubtargetPredicate = isGFX10Plus in
def S_GL1_INV : SM_Inval_Pseudo<"s_gl1_inv">;
let SubtargetPredicate = HasGetWaveIdInst in
def S_GET_WAVEID_IN_WORKGROUP : SM_WaveId_Pseudo <"s_get_waveid_in_workgroup", int_amdgcn_s_get_waveid_in_workgroup>;


let SubtargetPredicate = HasScalarFlatScratchInsts, Uses = [FLAT_SCR] in {
defm S_SCRATCH_LOAD_DWORD    : SM_Pseudo_Loads <SReg_64, SReg_32_XM0_XEXEC>;
defm S_SCRATCH_LOAD_DWORDX2  : SM_Pseudo_Loads <SReg_64, SReg_64_XEXEC>;
defm S_SCRATCH_LOAD_DWORDX4  : SM_Pseudo_Loads <SReg_64, SReg_128>;

defm S_SCRATCH_STORE_DWORD   : SM_Pseudo_Stores <SReg_64, SReg_32_XM0_XEXEC>;
defm S_SCRATCH_STORE_DWORDX2 : SM_Pseudo_Stores <SReg_64, SReg_64_XEXEC>;
defm S_SCRATCH_STORE_DWORDX4 : SM_Pseudo_Stores <SReg_64, SReg_128>;
} // SubtargetPredicate = HasScalarFlatScratchInsts

let SubtargetPredicate = HasScalarAtomics in {

let is_buffer = 1 in {
defm S_BUFFER_ATOMIC_SWAP         : SM_Pseudo_Atomics <SReg_128, SReg_32_XM0_XEXEC>;
defm S_BUFFER_ATOMIC_CMPSWAP      : SM_Pseudo_Atomics <SReg_128, SReg_64_XEXEC>;
defm S_BUFFER_ATOMIC_ADD          : SM_Pseudo_Atomics <SReg_128, SReg_32_XM0_XEXEC>;
defm S_BUFFER_ATOMIC_SUB          : SM_Pseudo_Atomics <SReg_128, SReg_32_XM0_XEXEC>;
defm S_BUFFER_ATOMIC_SMIN         : SM_Pseudo_Atomics <SReg_128, SReg_32_XM0_XEXEC>;
defm S_BUFFER_ATOMIC_UMIN         : SM_Pseudo_Atomics <SReg_128, SReg_32_XM0_XEXEC>;
defm S_BUFFER_ATOMIC_SMAX         : SM_Pseudo_Atomics <SReg_128, SReg_32_XM0_XEXEC>;
defm S_BUFFER_ATOMIC_UMAX         : SM_Pseudo_Atomics <SReg_128, SReg_32_XM0_XEXEC>;
defm S_BUFFER_ATOMIC_AND          : SM_Pseudo_Atomics <SReg_128, SReg_32_XM0_XEXEC>;
defm S_BUFFER_ATOMIC_OR           : SM_Pseudo_Atomics <SReg_128, SReg_32_XM0_XEXEC>;
defm S_BUFFER_ATOMIC_XOR          : SM_Pseudo_Atomics <SReg_128, SReg_32_XM0_XEXEC>;
defm S_BUFFER_ATOMIC_INC          : SM_Pseudo_Atomics <SReg_128, SReg_32_XM0_XEXEC>;
defm S_BUFFER_ATOMIC_DEC          : SM_Pseudo_Atomics <SReg_128, SReg_32_XM0_XEXEC>;

defm S_BUFFER_ATOMIC_SWAP_X2      : SM_Pseudo_Atomics <SReg_128, SReg_64_XEXEC>;
defm S_BUFFER_ATOMIC_CMPSWAP_X2   : SM_Pseudo_Atomics <SReg_128, SReg_128>;
defm S_BUFFER_ATOMIC_ADD_X2       : SM_Pseudo_Atomics <SReg_128, SReg_64_XEXEC>;
defm S_BUFFER_ATOMIC_SUB_X2       : SM_Pseudo_Atomics <SReg_128, SReg_64_XEXEC>;
defm S_BUFFER_ATOMIC_SMIN_X2      : SM_Pseudo_Atomics <SReg_128, SReg_64_XEXEC>;
defm S_BUFFER_ATOMIC_UMIN_X2      : SM_Pseudo_Atomics <SReg_128, SReg_64_XEXEC>;
defm S_BUFFER_ATOMIC_SMAX_X2      : SM_Pseudo_Atomics <SReg_128, SReg_64_XEXEC>;
defm S_BUFFER_ATOMIC_UMAX_X2      : SM_Pseudo_Atomics <SReg_128, SReg_64_XEXEC>;
defm S_BUFFER_ATOMIC_AND_X2       : SM_Pseudo_Atomics <SReg_128, SReg_64_XEXEC>;
defm S_BUFFER_ATOMIC_OR_X2        : SM_Pseudo_Atomics <SReg_128, SReg_64_XEXEC>;
defm S_BUFFER_ATOMIC_XOR_X2       : SM_Pseudo_Atomics <SReg_128, SReg_64_XEXEC>;
defm S_BUFFER_ATOMIC_INC_X2       : SM_Pseudo_Atomics <SReg_128, SReg_64_XEXEC>;
defm S_BUFFER_ATOMIC_DEC_X2       : SM_Pseudo_Atomics <SReg_128, SReg_64_XEXEC>;
}

defm S_ATOMIC_SWAP                : SM_Pseudo_Atomics <SReg_64, SReg_32_XM0_XEXEC>;
defm S_ATOMIC_CMPSWAP             : SM_Pseudo_Atomics <SReg_64, SReg_64_XEXEC>;
defm S_ATOMIC_ADD                 : SM_Pseudo_Atomics <SReg_64, SReg_32_XM0_XEXEC>;
defm S_ATOMIC_SUB                 : SM_Pseudo_Atomics <SReg_64, SReg_32_XM0_XEXEC>;
defm S_ATOMIC_SMIN                : SM_Pseudo_Atomics <SReg_64, SReg_32_XM0_XEXEC>;
defm S_ATOMIC_UMIN                : SM_Pseudo_Atomics <SReg_64, SReg_32_XM0_XEXEC>;
defm S_ATOMIC_SMAX                : SM_Pseudo_Atomics <SReg_64, SReg_32_XM0_XEXEC>;
defm S_ATOMIC_UMAX                : SM_Pseudo_Atomics <SReg_64, SReg_32_XM0_XEXEC>;
defm S_ATOMIC_AND                 : SM_Pseudo_Atomics <SReg_64, SReg_32_XM0_XEXEC>;
defm S_ATOMIC_OR                  : SM_Pseudo_Atomics <SReg_64, SReg_32_XM0_XEXEC>;
defm S_ATOMIC_XOR                 : SM_Pseudo_Atomics <SReg_64, SReg_32_XM0_XEXEC>;
defm S_ATOMIC_INC                 : SM_Pseudo_Atomics <SReg_64, SReg_32_XM0_XEXEC>;
defm S_ATOMIC_DEC                 : SM_Pseudo_Atomics <SReg_64, SReg_32_XM0_XEXEC>;

defm S_ATOMIC_SWAP_X2             : SM_Pseudo_Atomics <SReg_64, SReg_64_XEXEC>;
defm S_ATOMIC_CMPSWAP_X2          : SM_Pseudo_Atomics <SReg_64, SReg_128>;
defm S_ATOMIC_ADD_X2              : SM_Pseudo_Atomics <SReg_64, SReg_64_XEXEC>;
defm S_ATOMIC_SUB_X2              : SM_Pseudo_Atomics <SReg_64, SReg_64_XEXEC>;
defm S_ATOMIC_SMIN_X2             : SM_Pseudo_Atomics <SReg_64, SReg_64_XEXEC>;
defm S_ATOMIC_UMIN_X2             : SM_Pseudo_Atomics <SReg_64, SReg_64_XEXEC>;
defm S_ATOMIC_SMAX_X2             : SM_Pseudo_Atomics <SReg_64, SReg_64_XEXEC>;
defm S_ATOMIC_UMAX_X2             : SM_Pseudo_Atomics <SReg_64, SReg_64_XEXEC>;
defm S_ATOMIC_AND_X2              : SM_Pseudo_Atomics <SReg_64, SReg_64_XEXEC>;
defm S_ATOMIC_OR_X2               : SM_Pseudo_Atomics <SReg_64, SReg_64_XEXEC>;
defm S_ATOMIC_XOR_X2              : SM_Pseudo_Atomics <SReg_64, SReg_64_XEXEC>;
defm S_ATOMIC_INC_X2              : SM_Pseudo_Atomics <SReg_64, SReg_64_XEXEC>;
defm S_ATOMIC_DEC_X2              : SM_Pseudo_Atomics <SReg_64, SReg_64_XEXEC>;

} // let SubtargetPredicate = HasScalarAtomics

let SubtargetPredicate = HasScalarAtomics in {
defm S_DCACHE_DISCARD    : SM_Pseudo_Discards;
defm S_DCACHE_DISCARD_X2 : SM_Pseudo_Discards;
}

//===----------------------------------------------------------------------===//
// Targets
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// SI
//===----------------------------------------------------------------------===//

class SMRD_Real_si <bits<5> op, SM_Pseudo ps>
  : SM_Real<ps>
  , SIMCInstr<ps.PseudoInstr, SIEncodingFamily.SI>
  , Enc32 {

  let AssemblerPredicate = isGFX6GFX7;
  let DecoderNamespace = "GFX6GFX7";

  let Inst{7-0}   = !if(ps.has_offset, offset{7-0}, !if(ps.has_soffset, soffset, ?));
  let Inst{8}     = ps.has_offset;
  let Inst{14-9}  = !if(ps.has_sbase, sbase{6-1}, ?);
  let Inst{21-15} = !if(ps.has_sdst, sdst{6-0}, ?);
  let Inst{26-22} = op;
  let Inst{31-27} = 0x18; //encoding
}

multiclass SM_Real_Loads_si<bits<5> op> {
  defvar ps = NAME;
  defvar immPs = !cast<SM_Load_Pseudo>(ps#_IMM);
  def _IMM_si : SMRD_Real_si <op, immPs> {
    let InOperandList = (ins immPs.BaseClass:$sbase, smrd_offset_8:$offset, CPol:$cpol);
  }

  defvar sgprPs = !cast<SM_Load_Pseudo>(ps#_SGPR);
  def _SGPR_si : SMRD_Real_si <op, sgprPs>;
}

defm S_LOAD_DWORD           : SM_Real_Loads_si <0x00>;
defm S_LOAD_DWORDX2         : SM_Real_Loads_si <0x01>;
defm S_LOAD_DWORDX4         : SM_Real_Loads_si <0x02>;
defm S_LOAD_DWORDX8         : SM_Real_Loads_si <0x03>;
defm S_LOAD_DWORDX16        : SM_Real_Loads_si <0x04>;
defm S_BUFFER_LOAD_DWORD    : SM_Real_Loads_si <0x08>;
defm S_BUFFER_LOAD_DWORDX2  : SM_Real_Loads_si <0x09>;
defm S_BUFFER_LOAD_DWORDX4  : SM_Real_Loads_si <0x0a>;
defm S_BUFFER_LOAD_DWORDX8  : SM_Real_Loads_si <0x0b>;
defm S_BUFFER_LOAD_DWORDX16 : SM_Real_Loads_si <0x0c>;

def S_MEMTIME_si    : SMRD_Real_si <0x1e, S_MEMTIME>;
def S_DCACHE_INV_si : SMRD_Real_si <0x1f, S_DCACHE_INV>;


//===----------------------------------------------------------------------===//
// VI and GFX9.
//===----------------------------------------------------------------------===//

class SMEM_Real_vi <bits<8> op, SM_Pseudo ps>
  : SM_Real<ps>
  , SIMCInstr<ps.PseudoInstr, SIEncodingFamily.VI>
  , Enc64 {
  field bit IsGFX9SpecificEncoding = false;
  let AssemblerPredicate = !if(IsGFX9SpecificEncoding, isGFX9Only, isGFX8GFX9);
  let DecoderNamespace = "GFX8";

  let Inst{5-0}   = !if(ps.has_sbase, sbase{6-1}, ?);
  let Inst{12-6}  = !if(ps.has_sdst, sdst{6-0}, ?);

  // Note that for GFX9 instructions with immediate offsets, soffset_en
  // must be defined, whereas in GFX8 it's undefined in all cases,
  // meaning GFX9 is not perfectly backward-compatible with GFX8, despite
  // documentation suggesting otherwise.
  field bit SOffsetEn = !if(IsGFX9SpecificEncoding,
    !if(ps.has_offset, ps.has_soffset, !if(ps.has_soffset, 0, ?)),
    ?);
  let Inst{14} = SOffsetEn;

  let Inst{16} = !if(ps.has_glc, cpol{CPolBit.GLC}, ?);

  // imm
  // TODO: Shall not be defined if the instruction has no offset nor
  // soffset.
  let Inst{17} = ps.has_offset;

  let Inst{25-18} = op;
  let Inst{31-26} = 0x30; //encoding

  // VI supports 20-bit unsigned offsets while GFX9+ supports 21-bit signed.
  // Offset value is corrected accordingly when offset is encoded/decoded.
  // TODO: Forbid non-M0 register offsets for GFX8 stores and atomics.
  field bits<21> Offset;
  let Offset{6-0} = !if(ps.has_offset, offset{6-0},
                                       !if(ps.has_soffset, soffset{6-0}, ?));
  let Offset{20-7} = !if(ps.has_offset, offset{20-7}, ?);
  let Inst{52-32} = Offset;

  // soffset
  let Inst{63-57} = !if(!and(IsGFX9SpecificEncoding, ps.has_soffset),
                        soffset{6-0}, ?);
}

class SMEM_Real_Load_vi<bits<8> op, string ps>
    : SMEM_Real_vi<op, !cast<SM_Pseudo>(ps)>;

// The alternative GFX9 SGPR encoding using soffset to encode the
// offset register. Not available in assembler and goes to the GFX9
// encoding family to avoid conflicts with the primary SGPR variant.
class SMEM_Real_SGPR_alt_gfx9 {
  bit IsGFX9SpecificEncoding = true;
  bit SOffsetEn = 1;
  bit Offset = ?;
  int Subtarget = SIEncodingFamily.GFX9;
  string AsmVariantName = "NonParsable";
}

multiclass SM_Real_Loads_vi<bits<8> op> {
  defvar ps = NAME;
  def _IMM_vi : SMEM_Real_Load_vi <op, ps#"_IMM">;
  def _SGPR_vi : SMEM_Real_Load_vi <op, ps#"_SGPR">;
  def _SGPR_alt_gfx9 : SMEM_Real_Load_vi <op, ps#"_SGPR">,
                       SMEM_Real_SGPR_alt_gfx9;
  let IsGFX9SpecificEncoding = true in
  def _SGPR_IMM_gfx9 : SMEM_Real_Load_vi <op, ps#"_SGPR_IMM">;
}

class SMEM_Real_Store_Base_vi <bits<8> op, SM_Pseudo ps> : SMEM_Real_vi <op, ps> {
  // encoding
  bits<7> sdata;

  let sdst = ?;
  let Inst{12-6}  = !if(ps.has_sdst, sdata{6-0}, ?);
}

class SMEM_Real_Store_vi <bits<8> op, string ps>
    : SMEM_Real_Store_Base_vi <op, !cast<SM_Pseudo>(ps)>;

multiclass SM_Real_Stores_vi<bits<8> op> {
  defvar ps = NAME;
  def _IMM_vi : SMEM_Real_Store_vi <op, ps#"_IMM">;
  def _SGPR_vi : SMEM_Real_Store_vi <op, ps#"_SGPR">;
  def _SGPR_alt_gfx9 : SMEM_Real_Store_vi <op, ps#"_SGPR">,
                       SMEM_Real_SGPR_alt_gfx9;
  let IsGFX9SpecificEncoding = true in
  def _SGPR_IMM_gfx9 : SMEM_Real_Store_vi <op, ps#"_SGPR_IMM">;
}

multiclass SM_Real_Probe_vi<bits<8> op> {
  defvar ps = NAME;
  def _IMM_vi  : SMEM_Real_Store_Base_vi <op, !cast<SM_Probe_Pseudo>(ps#_IMM)>;
  def _SGPR_vi : SMEM_Real_Store_Base_vi <op, !cast<SM_Probe_Pseudo>(ps#_SGPR)>;
  def _SGPR_alt_gfx9
    : SMEM_Real_Store_Base_vi <op, !cast<SM_Probe_Pseudo>(ps#_SGPR)>,
      SMEM_Real_SGPR_alt_gfx9;
  let IsGFX9SpecificEncoding = true in
  def _SGPR_IMM_gfx9
    : SMEM_Real_Store_Base_vi <op, !cast<SM_Probe_Pseudo>(ps#_SGPR_IMM)>;
}

defm S_LOAD_DWORD           : SM_Real_Loads_vi <0x00>;
defm S_LOAD_DWORDX2         : SM_Real_Loads_vi <0x01>;
defm S_LOAD_DWORDX4         : SM_Real_Loads_vi <0x02>;
defm S_LOAD_DWORDX8         : SM_Real_Loads_vi <0x03>;
defm S_LOAD_DWORDX16        : SM_Real_Loads_vi <0x04>;
defm S_BUFFER_LOAD_DWORD    : SM_Real_Loads_vi <0x08>;
defm S_BUFFER_LOAD_DWORDX2  : SM_Real_Loads_vi <0x09>;
defm S_BUFFER_LOAD_DWORDX4  : SM_Real_Loads_vi <0x0a>;
defm S_BUFFER_LOAD_DWORDX8  : SM_Real_Loads_vi <0x0b>;
defm S_BUFFER_LOAD_DWORDX16 : SM_Real_Loads_vi <0x0c>;

defm S_STORE_DWORD : SM_Real_Stores_vi <0x10>;
defm S_STORE_DWORDX2 : SM_Real_Stores_vi <0x11>;
defm S_STORE_DWORDX4 : SM_Real_Stores_vi <0x12>;

defm S_BUFFER_STORE_DWORD    : SM_Real_Stores_vi <0x18>;
defm S_BUFFER_STORE_DWORDX2  : SM_Real_Stores_vi <0x19>;
defm S_BUFFER_STORE_DWORDX4  : SM_Real_Stores_vi <0x1a>;

// These instructions use same encoding
def S_DCACHE_INV_vi         : SMEM_Real_vi <0x20, S_DCACHE_INV>;
def S_DCACHE_WB_vi          : SMEM_Real_vi <0x21, S_DCACHE_WB>;
def S_DCACHE_INV_VOL_vi     : SMEM_Real_vi <0x22, S_DCACHE_INV_VOL>;
def S_DCACHE_WB_VOL_vi      : SMEM_Real_vi <0x23, S_DCACHE_WB_VOL>;
def S_MEMTIME_vi            : SMEM_Real_vi <0x24, S_MEMTIME>;
def S_MEMREALTIME_vi        : SMEM_Real_vi <0x25, S_MEMREALTIME>;

defm S_SCRATCH_LOAD_DWORD    : SM_Real_Loads_vi <0x05>;
defm S_SCRATCH_LOAD_DWORDX2  : SM_Real_Loads_vi <0x06>;
defm S_SCRATCH_LOAD_DWORDX4  : SM_Real_Loads_vi <0x07>;

defm S_SCRATCH_STORE_DWORD   : SM_Real_Stores_vi <0x15>;
defm S_SCRATCH_STORE_DWORDX2 : SM_Real_Stores_vi <0x16>;
defm S_SCRATCH_STORE_DWORDX4 : SM_Real_Stores_vi <0x17>;

defm S_ATC_PROBE        : SM_Real_Probe_vi <0x26>;
defm S_ATC_PROBE_BUFFER : SM_Real_Probe_vi <0x27>;

//===----------------------------------------------------------------------===//
// GFX9
//===----------------------------------------------------------------------===//

class SMEM_Atomic_Real_vi <bits<8> op, SM_Atomic_Pseudo ps>
  : SMEM_Real_vi <op, ps>,
    AtomicNoRet <!subst("_RTN","",NAME), ps.glc> {

  bits<7> sdata;

  let Constraints = ps.Constraints;
  let DisableEncoding = ps.DisableEncoding;

  let cpol{CPolBit.GLC} = ps.glc;
  let Inst{12-6} = !if(ps.glc, sdst{6-0}, sdata{6-0});
}

multiclass SM_Real_Atomics_vi<bits<8> op> {
  defvar ps = NAME;
  def _IMM_vi       : SMEM_Atomic_Real_vi <op, !cast<SM_Atomic_Pseudo>(ps#_IMM)>;
  def _SGPR_vi      : SMEM_Atomic_Real_vi <op, !cast<SM_Atomic_Pseudo>(ps#_SGPR)>;
  def _SGPR_alt_gfx9
    : SMEM_Atomic_Real_vi <op, !cast<SM_Atomic_Pseudo>(ps#_SGPR)>,
      SMEM_Real_SGPR_alt_gfx9;
  let IsGFX9SpecificEncoding = true in
  def _SGPR_IMM_gfx9
    : SMEM_Atomic_Real_vi <op, !cast<SM_Atomic_Pseudo>(ps#_SGPR_IMM)>;
  def _IMM_RTN_vi   : SMEM_Atomic_Real_vi <op, !cast<SM_Atomic_Pseudo>(ps#_IMM_RTN)>;
  def _SGPR_RTN_vi  : SMEM_Atomic_Real_vi <op, !cast<SM_Atomic_Pseudo>(ps#_SGPR_RTN)>;
  def _SGPR_RTN_alt_gfx9
    : SMEM_Atomic_Real_vi <op, !cast<SM_Atomic_Pseudo>(ps#_SGPR_RTN)>,
      SMEM_Real_SGPR_alt_gfx9;
  let IsGFX9SpecificEncoding = true in
  def _SGPR_IMM_RTN_gfx9
    : SMEM_Atomic_Real_vi <op, !cast<SM_Atomic_Pseudo>(ps#_SGPR_IMM_RTN)>;
}

defm S_BUFFER_ATOMIC_SWAP         : SM_Real_Atomics_vi <0x40>;
defm S_BUFFER_ATOMIC_CMPSWAP      : SM_Real_Atomics_vi <0x41>;
defm S_BUFFER_ATOMIC_ADD          : SM_Real_Atomics_vi <0x42>;
defm S_BUFFER_ATOMIC_SUB          : SM_Real_Atomics_vi <0x43>;
defm S_BUFFER_ATOMIC_SMIN         : SM_Real_Atomics_vi <0x44>;
defm S_BUFFER_ATOMIC_UMIN         : SM_Real_Atomics_vi <0x45>;
defm S_BUFFER_ATOMIC_SMAX         : SM_Real_Atomics_vi <0x46>;
defm S_BUFFER_ATOMIC_UMAX         : SM_Real_Atomics_vi <0x47>;
defm S_BUFFER_ATOMIC_AND          : SM_Real_Atomics_vi <0x48>;
defm S_BUFFER_ATOMIC_OR           : SM_Real_Atomics_vi <0x49>;
defm S_BUFFER_ATOMIC_XOR          : SM_Real_Atomics_vi <0x4a>;
defm S_BUFFER_ATOMIC_INC          : SM_Real_Atomics_vi <0x4b>;
defm S_BUFFER_ATOMIC_DEC          : SM_Real_Atomics_vi <0x4c>;

defm S_BUFFER_ATOMIC_SWAP_X2      : SM_Real_Atomics_vi <0x60>;
defm S_BUFFER_ATOMIC_CMPSWAP_X2   : SM_Real_Atomics_vi <0x61>;
defm S_BUFFER_ATOMIC_ADD_X2       : SM_Real_Atomics_vi <0x62>;
defm S_BUFFER_ATOMIC_SUB_X2       : SM_Real_Atomics_vi <0x63>;
defm S_BUFFER_ATOMIC_SMIN_X2      : SM_Real_Atomics_vi <0x64>;
defm S_BUFFER_ATOMIC_UMIN_X2      : SM_Real_Atomics_vi <0x65>;
defm S_BUFFER_ATOMIC_SMAX_X2      : SM_Real_Atomics_vi <0x66>;
defm S_BUFFER_ATOMIC_UMAX_X2      : SM_Real_Atomics_vi <0x67>;
defm S_BUFFER_ATOMIC_AND_X2       : SM_Real_Atomics_vi <0x68>;
defm S_BUFFER_ATOMIC_OR_X2        : SM_Real_Atomics_vi <0x69>;
defm S_BUFFER_ATOMIC_XOR_X2       : SM_Real_Atomics_vi <0x6a>;
defm S_BUFFER_ATOMIC_INC_X2       : SM_Real_Atomics_vi <0x6b>;
defm S_BUFFER_ATOMIC_DEC_X2       : SM_Real_Atomics_vi <0x6c>;

defm S_ATOMIC_SWAP                : SM_Real_Atomics_vi <0x80>;
defm S_ATOMIC_CMPSWAP             : SM_Real_Atomics_vi <0x81>;
defm S_ATOMIC_ADD                 : SM_Real_Atomics_vi <0x82>;
defm S_ATOMIC_SUB                 : SM_Real_Atomics_vi <0x83>;
defm S_ATOMIC_SMIN                : SM_Real_Atomics_vi <0x84>;
defm S_ATOMIC_UMIN                : SM_Real_Atomics_vi <0x85>;
defm S_ATOMIC_SMAX                : SM_Real_Atomics_vi <0x86>;
defm S_ATOMIC_UMAX                : SM_Real_Atomics_vi <0x87>;
defm S_ATOMIC_AND                 : SM_Real_Atomics_vi <0x88>;
defm S_ATOMIC_OR                  : SM_Real_Atomics_vi <0x89>;
defm S_ATOMIC_XOR                 : SM_Real_Atomics_vi <0x8a>;
defm S_ATOMIC_INC                 : SM_Real_Atomics_vi <0x8b>;
defm S_ATOMIC_DEC                 : SM_Real_Atomics_vi <0x8c>;

defm S_ATOMIC_SWAP_X2             : SM_Real_Atomics_vi <0xa0>;
defm S_ATOMIC_CMPSWAP_X2          : SM_Real_Atomics_vi <0xa1>;
defm S_ATOMIC_ADD_X2              : SM_Real_Atomics_vi <0xa2>;
defm S_ATOMIC_SUB_X2              : SM_Real_Atomics_vi <0xa3>;
defm S_ATOMIC_SMIN_X2             : SM_Real_Atomics_vi <0xa4>;
defm S_ATOMIC_UMIN_X2             : SM_Real_Atomics_vi <0xa5>;
defm S_ATOMIC_SMAX_X2             : SM_Real_Atomics_vi <0xa6>;
defm S_ATOMIC_UMAX_X2             : SM_Real_Atomics_vi <0xa7>;
defm S_ATOMIC_AND_X2              : SM_Real_Atomics_vi <0xa8>;
defm S_ATOMIC_OR_X2               : SM_Real_Atomics_vi <0xa9>;
defm S_ATOMIC_XOR_X2              : SM_Real_Atomics_vi <0xaa>;
defm S_ATOMIC_INC_X2              : SM_Real_Atomics_vi <0xab>;
defm S_ATOMIC_DEC_X2              : SM_Real_Atomics_vi <0xac>;

multiclass SM_Real_Discard_vi<bits<8> op> {
  defvar ps = NAME;
  def _IMM_vi  : SMEM_Real_vi <op, !cast<SM_Discard_Pseudo>(ps#_IMM)>;
  def _SGPR_vi : SMEM_Real_vi <op, !cast<SM_Discard_Pseudo>(ps#_SGPR)>;
  def _SGPR_alt_gfx9 : SMEM_Real_vi <op, !cast<SM_Discard_Pseudo>(ps#_SGPR)>,
                       SMEM_Real_SGPR_alt_gfx9;
  let IsGFX9SpecificEncoding = true in
  def _SGPR_IMM_gfx9 : SMEM_Real_vi <op, !cast<SM_Discard_Pseudo>(ps#_SGPR_IMM)>;
}

defm S_DCACHE_DISCARD    : SM_Real_Discard_vi <0x28>;
defm S_DCACHE_DISCARD_X2 : SM_Real_Discard_vi <0x29>;

//===----------------------------------------------------------------------===//
// CI
//===----------------------------------------------------------------------===//

def smrd_literal_offset : ImmOperand<i32, "SMRDLiteralOffset">;

class SMRD_Real_Load_IMM_ci <bits<5> op, SM_Load_Pseudo ps> :
  SM_Real<ps>,
  Enc64 {

  let AssemblerPredicate = isGFX7Only;
  let DecoderNamespace = "GFX7";
  let InOperandList = (ins ps.BaseClass:$sbase, smrd_literal_offset:$offset, CPol:$cpol);

  let Inst{7-0}   = 0xff;
  let Inst{8}     = 0;
  let Inst{14-9}  = sbase{6-1};
  let Inst{21-15} = sdst{6-0};
  let Inst{26-22} = op;
  let Inst{31-27} = 0x18; //encoding
  let Inst{63-32} = offset{31-0};
}

def S_LOAD_DWORD_IMM_ci           : SMRD_Real_Load_IMM_ci <0x00, S_LOAD_DWORD_IMM>;
def S_LOAD_DWORDX2_IMM_ci         : SMRD_Real_Load_IMM_ci <0x01, S_LOAD_DWORDX2_IMM>;
def S_LOAD_DWORDX4_IMM_ci         : SMRD_Real_Load_IMM_ci <0x02, S_LOAD_DWORDX4_IMM>;
def S_LOAD_DWORDX8_IMM_ci         : SMRD_Real_Load_IMM_ci <0x03, S_LOAD_DWORDX8_IMM>;
def S_LOAD_DWORDX16_IMM_ci        : SMRD_Real_Load_IMM_ci <0x04, S_LOAD_DWORDX16_IMM>;
def S_BUFFER_LOAD_DWORD_IMM_ci    : SMRD_Real_Load_IMM_ci <0x08, S_BUFFER_LOAD_DWORD_IMM>;
def S_BUFFER_LOAD_DWORDX2_IMM_ci  : SMRD_Real_Load_IMM_ci <0x09, S_BUFFER_LOAD_DWORDX2_IMM>;
def S_BUFFER_LOAD_DWORDX4_IMM_ci  : SMRD_Real_Load_IMM_ci <0x0a, S_BUFFER_LOAD_DWORDX4_IMM>;
def S_BUFFER_LOAD_DWORDX8_IMM_ci  : SMRD_Real_Load_IMM_ci <0x0b, S_BUFFER_LOAD_DWORDX8_IMM>;
def S_BUFFER_LOAD_DWORDX16_IMM_ci : SMRD_Real_Load_IMM_ci <0x0c, S_BUFFER_LOAD_DWORDX16_IMM>;

class SMRD_Real_ci <bits<5> op, SM_Pseudo ps>
  : SM_Real<ps>
  , SIMCInstr<ps.PseudoInstr, SIEncodingFamily.SI>
  , Enc32 {

  let AssemblerPredicate = isGFX7Only;
  let DecoderNamespace = "GFX7";

  let Inst{7-0}   = !if(ps.has_offset, offset{7-0}, !if(ps.has_soffset, soffset, ?));
  let Inst{8}     = ps.has_offset;
  let Inst{14-9}  = !if(ps.has_sbase, sbase{6-1}, ?);
  let Inst{21-15} = !if(ps.has_sdst, sdst{6-0}, ?);
  let Inst{26-22} = op;
  let Inst{31-27} = 0x18; //encoding
}

def S_DCACHE_INV_VOL_ci : SMRD_Real_ci <0x1d, S_DCACHE_INV_VOL>;

//===----------------------------------------------------------------------===//
// Scalar Memory Patterns
//===----------------------------------------------------------------------===//

def smrd_load : PatFrag <(ops node:$ptr), (load node:$ptr), [{ return isUniformLoad(N);}]> {
  let GISelPredicateCode = [{
    if (!MI.hasOneMemOperand())
      return false;
    if (!isInstrUniform(MI))
      return false;

    // FIXME: We should probably be caching this.
    SmallVector<GEPInfo, 4> AddrInfo;
    getAddrModeInfo(MI, MRI, AddrInfo);

    if (hasVgprParts(AddrInfo))
      return false;
    return true;
  }];
}

def SMRDImm         : ComplexPattern<iPTR, 2, "SelectSMRDImm">;
def SMRDImm32       : ComplexPattern<iPTR, 2, "SelectSMRDImm32">;
def SMRDSgpr        : ComplexPattern<iPTR, 2, "SelectSMRDSgpr">;
def SMRDSgprImm     : ComplexPattern<iPTR, 3, "SelectSMRDSgprImm">;
def SMRDBufferImm   : ComplexPattern<iPTR, 1, "SelectSMRDBufferImm">;
def SMRDBufferImm32 : ComplexPattern<iPTR, 1, "SelectSMRDBufferImm32">;
def SMRDBufferSgprImm : ComplexPattern<iPTR, 2, "SelectSMRDBufferSgprImm">;

multiclass SMRD_Pattern <string Instr, ValueType vt> {

  // 1. IMM offset
  def : GCNPat <
    (smrd_load (SMRDImm i64:$sbase, i32:$offset)),
    (vt (!cast<SM_Pseudo>(Instr#"_IMM") $sbase, $offset, 0))
  >;

  // 2. 32-bit IMM offset on CI
  def : GCNPat <
    (smrd_load (SMRDImm32 i64:$sbase, i32:$offset)),
    (vt (!cast<InstSI>(Instr#"_IMM_ci") $sbase, $offset, 0))> {
    let OtherPredicates = [isGFX7Only];
  }

  // 3. SGPR offset
  def : GCNPat <
    (smrd_load (SMRDSgpr i64:$sbase, i32:$soffset)),
    (vt (!cast<SM_Pseudo>(Instr#"_SGPR") $sbase, $soffset, 0))
  >;

  // 4. SGPR+IMM offset
  def : GCNPat <
    (smrd_load (SMRDSgprImm i64:$sbase, i32:$soffset, i32:$offset)),
    (vt (!cast<SM_Pseudo>(Instr#"_SGPR_IMM") $sbase, $soffset, $offset, 0))> {
    let OtherPredicates = [isGFX9Plus];
  }

  // 5. No offset
  def : GCNPat <
    (vt (smrd_load (i64 SReg_64:$sbase))),
    (vt (!cast<SM_Pseudo>(Instr#"_IMM") i64:$sbase, 0, 0))
  >;
}

multiclass SMLoad_Pattern <string Instr, ValueType vt> {
  // 1. Offset as an immediate
  def : GCNPat <
    (SIsbuffer_load v4i32:$sbase, (SMRDBufferImm i32:$offset), timm:$cachepolicy),
    (vt (!cast<SM_Pseudo>(Instr#"_IMM") SReg_128:$sbase, i32imm:$offset, (extract_cpol $cachepolicy)))> {
    let AddedComplexity = 2;
  }

  // 2. 32-bit IMM offset on CI
  def : GCNPat <
    (vt (SIsbuffer_load v4i32:$sbase, (SMRDBufferImm32 i32:$offset), timm:$cachepolicy)),
    (!cast<InstSI>(Instr#"_IMM_ci") SReg_128:$sbase, smrd_literal_offset:$offset,
                                    (extract_cpol $cachepolicy))> {
    let OtherPredicates = [isGFX7Only];
    let AddedComplexity = 1;
  }

  // 3. Offset loaded in an 32bit SGPR
  def : GCNPat <
    (SIsbuffer_load v4i32:$sbase, i32:$soffset, timm:$cachepolicy),
    (vt (!cast<SM_Pseudo>(Instr#"_SGPR") SReg_128:$sbase, SReg_32:$soffset, (extract_cpol $cachepolicy)))
  >;

  // 4. Offset as an 32-bit SGPR + immediate
  def : GCNPat <
    (SIsbuffer_load v4i32:$sbase, (SMRDBufferSgprImm i32:$soffset, i32:$offset),
                    timm:$cachepolicy),
    (vt (!cast<SM_Pseudo>(Instr#"_SGPR_IMM") SReg_128:$sbase, SReg_32:$soffset, i32imm:$offset,
                                             (extract_cpol $cachepolicy)))> {
    let OtherPredicates = [isGFX9Plus];
  }
}

// Global and constant loads can be selected to either MUBUF or SMRD
// instructions, but SMRD instructions are faster so we want the instruction
// selector to prefer those.
let AddedComplexity = 100 in {

foreach vt = Reg32Types.types in {
defm : SMRD_Pattern <"S_LOAD_DWORD", vt>;
}

foreach vt = SReg_64.RegTypes in {
defm : SMRD_Pattern <"S_LOAD_DWORDX2", vt>;
}

foreach vt = SReg_128.RegTypes in {
defm : SMRD_Pattern <"S_LOAD_DWORDX4", vt>;
}

foreach vt = SReg_256.RegTypes in {
defm : SMRD_Pattern <"S_LOAD_DWORDX8", vt>;
}

foreach vt = SReg_512.RegTypes in {
defm : SMRD_Pattern <"S_LOAD_DWORDX16", vt>;
}

defm : SMLoad_Pattern <"S_BUFFER_LOAD_DWORD",     i32>;
defm : SMLoad_Pattern <"S_BUFFER_LOAD_DWORDX2",   v2i32>;
defm : SMLoad_Pattern <"S_BUFFER_LOAD_DWORDX4",   v4i32>;
defm : SMLoad_Pattern <"S_BUFFER_LOAD_DWORDX8",   v8i32>;
defm : SMLoad_Pattern <"S_BUFFER_LOAD_DWORDX16",  v16i32>;

defm : SMLoad_Pattern <"S_BUFFER_LOAD_DWORD",     f32>;
defm : SMLoad_Pattern <"S_BUFFER_LOAD_DWORDX2",   v2f32>;
defm : SMLoad_Pattern <"S_BUFFER_LOAD_DWORDX4",   v4f32>;
defm : SMLoad_Pattern <"S_BUFFER_LOAD_DWORDX8",   v8f32>;
defm : SMLoad_Pattern <"S_BUFFER_LOAD_DWORDX16",  v16f32>;
} // End let AddedComplexity = 100

let OtherPredicates = [HasSMemTimeInst] in {
def : GCNPat <
  (i64 (readcyclecounter)),
  (S_MEMTIME)
>;
} // let OtherPredicates = [HasSMemTimeInst]

let OtherPredicates = [HasShaderCyclesRegister] in {
def : GCNPat <
  (i64 (readcyclecounter)),
  (REG_SEQUENCE SReg_64,
    (S_GETREG_B32 getHwRegImm<HWREG.SHADER_CYCLES, 0, -12>.ret), sub0,
    (S_MOV_B32 (i32 0)), sub1)> {
  // Prefer this to s_memtime because it has lower and more predictable latency.
  let AddedComplexity = 1;
}
} // let OtherPredicates = [HasShaderCyclesRegister]

//===----------------------------------------------------------------------===//
// GFX10.
//===----------------------------------------------------------------------===//

class SMEM_Real_10Plus_common<bits<8> op, SM_Pseudo ps, string opName,
                              int subtarget, RegisterWithSubRegs sgpr_null> :
    SM_Real<ps, opName>, SIMCInstr<ps.PseudoInstr, subtarget>, Enc64 {
  let Inst{5-0}   = !if(ps.has_sbase, sbase{6-1}, ?);
  let Inst{12-6}  = !if(ps.has_sdst, sdst{6-0}, ?);
  let Inst{25-18} = op;
  let Inst{31-26} = 0x3d;
  // There are SMEM instructions that do not employ any of the offset
  // fields, in which case we need them to remain undefined.
  let Inst{52-32} = !if(ps.has_offset, offset{20-0}, !if(ps.has_soffset, 0, ?));
  let Inst{63-57} = !if(ps.has_soffset, soffset{6-0},
                        !if(ps.has_offset, sgpr_null.HWEncoding{6-0}, ?));
}

class SMEM_Real_gfx10<bits<8> op, SM_Pseudo ps>
    : SMEM_Real_10Plus_common<op, ps, ps.Mnemonic, SIEncodingFamily.GFX10,
                              SGPR_NULL_gfxpre11> {
  let AssemblerPredicate = isGFX10Only;
  let DecoderNamespace = "GFX10";
  let Inst{14}    = !if(ps.has_dlc, cpol{CPolBit.DLC}, ?);
  let Inst{16}    = !if(ps.has_glc, cpol{CPolBit.GLC}, ?);
}

class SMEM_Real_Load_gfx10<bits<8> op, string ps>
    : SMEM_Real_gfx10<op, !cast<SM_Pseudo>(ps)>;

multiclass SM_Real_Loads_gfx10<bits<8> op> {
  defvar ps = NAME;
  def _IMM_gfx10 : SMEM_Real_Load_gfx10<op, ps#"_IMM">;
  def _SGPR_gfx10 : SMEM_Real_Load_gfx10<op, ps#"_SGPR">;
  def _SGPR_IMM_gfx10 : SMEM_Real_Load_gfx10<op, ps#"_SGPR_IMM">;
}

class SMEM_Real_Store_gfx10<bits<8> op, SM_Pseudo ps> : SMEM_Real_gfx10<op, ps> {
  bits<7> sdata;

  let sdst = ?;
  let Inst{12-6} = !if(ps.has_sdst, sdata{6-0}, ?);
}

multiclass SM_Real_Stores_gfx10<bits<8> op> {
  defvar ps = NAME;
  defvar immPs = !cast<SM_Store_Pseudo>(ps#_IMM);
  def _IMM_gfx10 : SMEM_Real_Store_gfx10 <op, immPs>;

  defvar sgprPs = !cast<SM_Store_Pseudo>(ps#_SGPR);
  def _SGPR_gfx10 : SMEM_Real_Store_gfx10 <op, sgprPs>;

  defvar sgprImmPs = !cast<SM_Store_Pseudo>(ps#_SGPR_IMM);
  def _SGPR_IMM_gfx10 : SMEM_Real_Store_gfx10 <op, sgprImmPs>;
}

defm S_LOAD_DWORD            : SM_Real_Loads_gfx10<0x000>;
defm S_LOAD_DWORDX2          : SM_Real_Loads_gfx10<0x001>;
defm S_LOAD_DWORDX4          : SM_Real_Loads_gfx10<0x002>;
defm S_LOAD_DWORDX8          : SM_Real_Loads_gfx10<0x003>;
defm S_LOAD_DWORDX16         : SM_Real_Loads_gfx10<0x004>;

let SubtargetPredicate = HasScalarFlatScratchInsts in {
defm S_SCRATCH_LOAD_DWORD    : SM_Real_Loads_gfx10<0x005>;
defm S_SCRATCH_LOAD_DWORDX2  : SM_Real_Loads_gfx10<0x006>;
defm S_SCRATCH_LOAD_DWORDX4  : SM_Real_Loads_gfx10<0x007>;
} // End SubtargetPredicate = HasScalarFlatScratchInsts

defm S_BUFFER_LOAD_DWORD     : SM_Real_Loads_gfx10<0x008>;
defm S_BUFFER_LOAD_DWORDX2   : SM_Real_Loads_gfx10<0x009>;
defm S_BUFFER_LOAD_DWORDX4   : SM_Real_Loads_gfx10<0x00a>;
defm S_BUFFER_LOAD_DWORDX8   : SM_Real_Loads_gfx10<0x00b>;
defm S_BUFFER_LOAD_DWORDX16  : SM_Real_Loads_gfx10<0x00c>;

let SubtargetPredicate = HasScalarStores in {
defm S_STORE_DWORD           : SM_Real_Stores_gfx10<0x010>;
defm S_STORE_DWORDX2         : SM_Real_Stores_gfx10<0x011>;
defm S_STORE_DWORDX4         : SM_Real_Stores_gfx10<0x012>;
let OtherPredicates = [HasScalarFlatScratchInsts] in {
defm S_SCRATCH_STORE_DWORD   : SM_Real_Stores_gfx10<0x015>;
defm S_SCRATCH_STORE_DWORDX2 : SM_Real_Stores_gfx10<0x016>;
defm S_SCRATCH_STORE_DWORDX4 : SM_Real_Stores_gfx10<0x017>;
} // End OtherPredicates = [HasScalarFlatScratchInsts]
defm S_BUFFER_STORE_DWORD    : SM_Real_Stores_gfx10<0x018>;
defm S_BUFFER_STORE_DWORDX2  : SM_Real_Stores_gfx10<0x019>;
defm S_BUFFER_STORE_DWORDX4  : SM_Real_Stores_gfx10<0x01a>;
} // End SubtargetPredicate = HasScalarStores

def S_MEMREALTIME_gfx10              : SMEM_Real_gfx10<0x025, S_MEMREALTIME>;
def S_MEMTIME_gfx10                  : SMEM_Real_gfx10<0x024, S_MEMTIME>;
def S_GL1_INV_gfx10                  : SMEM_Real_gfx10<0x01f, S_GL1_INV>;
def S_GET_WAVEID_IN_WORKGROUP_gfx10  : SMEM_Real_gfx10<0x02a, S_GET_WAVEID_IN_WORKGROUP>;
def S_DCACHE_INV_gfx10               : SMEM_Real_gfx10<0x020, S_DCACHE_INV>;

let SubtargetPredicate = HasScalarStores in {
def S_DCACHE_WB_gfx10                : SMEM_Real_gfx10<0x021, S_DCACHE_WB>;
} // End SubtargetPredicate = HasScalarStores

multiclass SM_Real_Probe_gfx10<bits<8> op> {
  defvar ps = NAME;
  def _IMM_gfx10  : SMEM_Real_Store_gfx10 <op, !cast<SM_Pseudo>(ps#_IMM)>;
  def _SGPR_gfx10 : SMEM_Real_Store_gfx10 <op, !cast<SM_Pseudo>(ps#_SGPR)>;
  def _SGPR_IMM_gfx10
    : SMEM_Real_Store_gfx10 <op, !cast<SM_Pseudo>(ps#_SGPR_IMM)>;
}

defm S_ATC_PROBE        : SM_Real_Probe_gfx10 <0x26>;
defm S_ATC_PROBE_BUFFER : SM_Real_Probe_gfx10 <0x27>;

class SMEM_Atomic_Real_gfx10 <bits<8> op, SM_Atomic_Pseudo ps>
  : SMEM_Real_gfx10 <op, ps>,
    AtomicNoRet <!subst("_RTN","",NAME), ps.glc> {

  bits<7> sdata;

  let Constraints = ps.Constraints;
  let DisableEncoding = ps.DisableEncoding;

  let cpol{CPolBit.GLC} = ps.glc;

  let Inst{14} = !if(ps.has_dlc, cpol{CPolBit.DLC}, 0);
  let Inst{12-6} = !if(ps.glc, sdst{6-0}, sdata{6-0});
}

multiclass SM_Real_Atomics_gfx10<bits<8> op> {
  defvar ps = NAME;
  def _IMM_gfx10       : SMEM_Atomic_Real_gfx10 <op, !cast<SM_Atomic_Pseudo>(ps#_IMM)>;
  def _SGPR_gfx10      : SMEM_Atomic_Real_gfx10 <op, !cast<SM_Atomic_Pseudo>(ps#_SGPR)>;
  def _SGPR_IMM_gfx10  : SMEM_Atomic_Real_gfx10 <op, !cast<SM_Atomic_Pseudo>(ps#_SGPR_IMM)>;
  def _IMM_RTN_gfx10   : SMEM_Atomic_Real_gfx10 <op, !cast<SM_Atomic_Pseudo>(ps#_IMM_RTN)>;
  def _SGPR_RTN_gfx10  : SMEM_Atomic_Real_gfx10 <op, !cast<SM_Atomic_Pseudo>(ps#_SGPR_RTN)>;
  def _SGPR_IMM_RTN_gfx10 : SMEM_Atomic_Real_gfx10 <op, !cast<SM_Atomic_Pseudo>(ps#_SGPR_IMM_RTN)>;
}

let SubtargetPredicate = HasScalarAtomics in {

defm S_BUFFER_ATOMIC_SWAP         : SM_Real_Atomics_gfx10 <0x40>;
defm S_BUFFER_ATOMIC_CMPSWAP      : SM_Real_Atomics_gfx10 <0x41>;
defm S_BUFFER_ATOMIC_ADD          : SM_Real_Atomics_gfx10 <0x42>;
defm S_BUFFER_ATOMIC_SUB          : SM_Real_Atomics_gfx10 <0x43>;
defm S_BUFFER_ATOMIC_SMIN         : SM_Real_Atomics_gfx10 <0x44>;
defm S_BUFFER_ATOMIC_UMIN         : SM_Real_Atomics_gfx10 <0x45>;
defm S_BUFFER_ATOMIC_SMAX         : SM_Real_Atomics_gfx10 <0x46>;
defm S_BUFFER_ATOMIC_UMAX         : SM_Real_Atomics_gfx10 <0x47>;
defm S_BUFFER_ATOMIC_AND          : SM_Real_Atomics_gfx10 <0x48>;
defm S_BUFFER_ATOMIC_OR           : SM_Real_Atomics_gfx10 <0x49>;
defm S_BUFFER_ATOMIC_XOR          : SM_Real_Atomics_gfx10 <0x4a>;
defm S_BUFFER_ATOMIC_INC          : SM_Real_Atomics_gfx10 <0x4b>;
defm S_BUFFER_ATOMIC_DEC          : SM_Real_Atomics_gfx10 <0x4c>;

defm S_BUFFER_ATOMIC_SWAP_X2      : SM_Real_Atomics_gfx10 <0x60>;
defm S_BUFFER_ATOMIC_CMPSWAP_X2   : SM_Real_Atomics_gfx10 <0x61>;
defm S_BUFFER_ATOMIC_ADD_X2       : SM_Real_Atomics_gfx10 <0x62>;
defm S_BUFFER_ATOMIC_SUB_X2       : SM_Real_Atomics_gfx10 <0x63>;
defm S_BUFFER_ATOMIC_SMIN_X2      : SM_Real_Atomics_gfx10 <0x64>;
defm S_BUFFER_ATOMIC_UMIN_X2      : SM_Real_Atomics_gfx10 <0x65>;
defm S_BUFFER_ATOMIC_SMAX_X2      : SM_Real_Atomics_gfx10 <0x66>;
defm S_BUFFER_ATOMIC_UMAX_X2      : SM_Real_Atomics_gfx10 <0x67>;
defm S_BUFFER_ATOMIC_AND_X2       : SM_Real_Atomics_gfx10 <0x68>;
defm S_BUFFER_ATOMIC_OR_X2        : SM_Real_Atomics_gfx10 <0x69>;
defm S_BUFFER_ATOMIC_XOR_X2       : SM_Real_Atomics_gfx10 <0x6a>;
defm S_BUFFER_ATOMIC_INC_X2       : SM_Real_Atomics_gfx10 <0x6b>;
defm S_BUFFER_ATOMIC_DEC_X2       : SM_Real_Atomics_gfx10 <0x6c>;

defm S_ATOMIC_SWAP                : SM_Real_Atomics_gfx10 <0x80>;
defm S_ATOMIC_CMPSWAP             : SM_Real_Atomics_gfx10 <0x81>;
defm S_ATOMIC_ADD                 : SM_Real_Atomics_gfx10 <0x82>;
defm S_ATOMIC_SUB                 : SM_Real_Atomics_gfx10 <0x83>;
defm S_ATOMIC_SMIN                : SM_Real_Atomics_gfx10 <0x84>;
defm S_ATOMIC_UMIN                : SM_Real_Atomics_gfx10 <0x85>;
defm S_ATOMIC_SMAX                : SM_Real_Atomics_gfx10 <0x86>;
defm S_ATOMIC_UMAX                : SM_Real_Atomics_gfx10 <0x87>;
defm S_ATOMIC_AND                 : SM_Real_Atomics_gfx10 <0x88>;
defm S_ATOMIC_OR                  : SM_Real_Atomics_gfx10 <0x89>;
defm S_ATOMIC_XOR                 : SM_Real_Atomics_gfx10 <0x8a>;
defm S_ATOMIC_INC                 : SM_Real_Atomics_gfx10 <0x8b>;
defm S_ATOMIC_DEC                 : SM_Real_Atomics_gfx10 <0x8c>;

defm S_ATOMIC_SWAP_X2             : SM_Real_Atomics_gfx10 <0xa0>;
defm S_ATOMIC_CMPSWAP_X2          : SM_Real_Atomics_gfx10 <0xa1>;
defm S_ATOMIC_ADD_X2              : SM_Real_Atomics_gfx10 <0xa2>;
defm S_ATOMIC_SUB_X2              : SM_Real_Atomics_gfx10 <0xa3>;
defm S_ATOMIC_SMIN_X2             : SM_Real_Atomics_gfx10 <0xa4>;
defm S_ATOMIC_UMIN_X2             : SM_Real_Atomics_gfx10 <0xa5>;
defm S_ATOMIC_SMAX_X2             : SM_Real_Atomics_gfx10 <0xa6>;
defm S_ATOMIC_UMAX_X2             : SM_Real_Atomics_gfx10 <0xa7>;
defm S_ATOMIC_AND_X2              : SM_Real_Atomics_gfx10 <0xa8>;
defm S_ATOMIC_OR_X2               : SM_Real_Atomics_gfx10 <0xa9>;
defm S_ATOMIC_XOR_X2              : SM_Real_Atomics_gfx10 <0xaa>;
defm S_ATOMIC_INC_X2              : SM_Real_Atomics_gfx10 <0xab>;
defm S_ATOMIC_DEC_X2              : SM_Real_Atomics_gfx10 <0xac>;

multiclass SM_Real_Discard_gfx10<bits<8> op> {
  defvar ps = NAME;
  def _IMM_gfx10  : SMEM_Real_gfx10 <op, !cast<SM_Pseudo>(ps#_IMM)>;
  def _SGPR_gfx10 : SMEM_Real_gfx10 <op, !cast<SM_Pseudo>(ps#_SGPR)>;
  def _SGPR_IMM_gfx10 : SMEM_Real_gfx10 <op, !cast<SM_Pseudo>(ps#_SGPR_IMM)>;
}

defm S_DCACHE_DISCARD    : SM_Real_Discard_gfx10 <0x28>;
defm S_DCACHE_DISCARD_X2 : SM_Real_Discard_gfx10 <0x29>;

} // End SubtargetPredicate = HasScalarAtomics

def SMInfoTable : GenericTable {
  let FilterClass = "SM_Real";
  let CppTypeName = "SMInfo";
  let Fields = ["Opcode", "is_buffer"];

  let PrimaryKey = ["Opcode"];
  let PrimaryKeyName = "getSMEMOpcodeHelper";
}

//===----------------------------------------------------------------------===//
// GFX11.
//===----------------------------------------------------------------------===//

class SMEM_Real_gfx11<bits<8> op, SM_Pseudo ps, string opName = ps.Mnemonic> :
    SMEM_Real_10Plus_common<op, ps, opName, SIEncodingFamily.GFX11,
                            SGPR_NULL_gfx11plus> {
  let AssemblerPredicate = isGFX11Plus;
  let DecoderNamespace = "GFX11";
  let Inst{13}    = !if(ps.has_dlc, cpol{CPolBit.DLC}, 0);
  let Inst{14}    = !if(ps.has_glc, cpol{CPolBit.GLC}, 0);
}

class SMEM_Real_Load_gfx11<bits<8> op, string ps, string opName> :
    SMEM_Real_gfx11<op, !cast<SM_Pseudo>(ps), opName>;

multiclass SM_Real_Loads_gfx11<bits<8> op, string ps> {
  defvar opName = !tolower(NAME);
  def _IMM_gfx11 : SMEM_Real_Load_gfx11<op, ps#"_IMM", opName>;
  def _SGPR_gfx11 : SMEM_Real_Load_gfx11<op, ps#"_SGPR", opName>;
  def _SGPR_IMM_gfx11 : SMEM_Real_Load_gfx11<op, ps#"_SGPR_IMM", opName>;
  def : MnemonicAlias<!cast<SM_Pseudo>(ps#"_IMM").Mnemonic, opName>,
                      Requires<[isGFX11Plus]>;
}

defm S_LOAD_B32  : SM_Real_Loads_gfx11<0x000, "S_LOAD_DWORD">;
defm S_LOAD_B64  : SM_Real_Loads_gfx11<0x001, "S_LOAD_DWORDX2">;
defm S_LOAD_B128 : SM_Real_Loads_gfx11<0x002, "S_LOAD_DWORDX4">;
defm S_LOAD_B256 : SM_Real_Loads_gfx11<0x003, "S_LOAD_DWORDX8">;
defm S_LOAD_B512 : SM_Real_Loads_gfx11<0x004, "S_LOAD_DWORDX16">;

defm S_BUFFER_LOAD_B32  : SM_Real_Loads_gfx11<0x008, "S_BUFFER_LOAD_DWORD">;
defm S_BUFFER_LOAD_B64  : SM_Real_Loads_gfx11<0x009, "S_BUFFER_LOAD_DWORDX2">;
defm S_BUFFER_LOAD_B128 : SM_Real_Loads_gfx11<0x00a, "S_BUFFER_LOAD_DWORDX4">;
defm S_BUFFER_LOAD_B256 : SM_Real_Loads_gfx11<0x00b, "S_BUFFER_LOAD_DWORDX8">;
defm S_BUFFER_LOAD_B512 : SM_Real_Loads_gfx11<0x00c, "S_BUFFER_LOAD_DWORDX16">;

def S_GL1_INV_gfx11    : SMEM_Real_gfx11<0x020, S_GL1_INV>;
def S_DCACHE_INV_gfx11 : SMEM_Real_gfx11<0x021, S_DCACHE_INV>;

class SMEM_Real_Store_gfx11 <bits<8> op, SM_Pseudo ps> : SMEM_Real_gfx11<op, ps> {
  // encoding
  bits<7> sdata;

  let sdst = ?;
  let Inst{12-6}  = !if(ps.has_sdst, sdata{6-0}, ?);
}

multiclass SM_Real_Probe_gfx11<bits<8> op> {
  defvar ps = NAME;
  def _IMM_gfx11  : SMEM_Real_Store_gfx11 <op, !cast<SM_Probe_Pseudo>(ps#_IMM)>;
  def _SGPR_gfx11 : SMEM_Real_Store_gfx11 <op, !cast<SM_Probe_Pseudo>(ps#_SGPR)>;
  def _SGPR_IMM_gfx11
    : SMEM_Real_Store_gfx11 <op, !cast<SM_Probe_Pseudo>(ps#_SGPR_IMM)>;
}

defm S_ATC_PROBE        : SM_Real_Probe_gfx11 <0x22>;
defm S_ATC_PROBE_BUFFER : SM_Real_Probe_gfx11 <0x23>;
