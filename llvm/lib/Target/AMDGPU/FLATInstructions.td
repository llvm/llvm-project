//===-- FLATInstructions.td - FLAT Instruction Definitions ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

def FlatOffset : ComplexPattern<iPTR, 2, "SelectFlatOffset", [], [SDNPWantRoot], -10>;
def GlobalOffset : ComplexPattern<iPTR, 2, "SelectGlobalOffset", [], [SDNPWantRoot], -10>;
def ScratchOffset : ComplexPattern<iPTR, 2, "SelectScratchOffset", [], [SDNPWantRoot], -10>;

def GlobalSAddr : ComplexPattern<iPTR, 3, "SelectGlobalSAddr", [], [SDNPWantRoot], -10>;
def ScratchSAddr : ComplexPattern<iPTR, 2, "SelectScratchSAddr", [], [SDNPWantRoot], -10>;
def ScratchSVAddr : ComplexPattern<iPTR, 3, "SelectScratchSVAddr", [], [SDNPWantRoot], -10>;

//===----------------------------------------------------------------------===//
// FLAT classes
//===----------------------------------------------------------------------===//

class FLAT_Pseudo<string opName, dag outs, dag ins,
                  string asmOps, list<dag> pattern=[]> :
  InstSI<outs, ins, "", pattern>,
  SIMCInstr<opName, SIEncodingFamily.NONE> {

  let isPseudo = 1;
  let isCodeGenOnly = 1;

  let FLAT = 1;

  let UseNamedOperandTable = 1;
  let hasSideEffects = 0;
  let SchedRW = [WriteVMEM];

  string Mnemonic = opName;
  string AsmOperands = asmOps;

  bits<1> is_flat_global = 0;
  bits<1> is_flat_scratch = 0;

  bits<1> has_vdst = 1;

  // We need to distinguish having saddr and enabling saddr because
  // saddr is only valid for scratch and global instructions. Pre-gfx9
  // these bits were reserved, so we also don't necessarily want to
  // set these bits to the disabled value for the original flat
  // segment instructions.
  bits<1> has_saddr = 0;
  bits<1> enabled_saddr = 0;
  bits<7> saddr_value = 0;
  bits<1> has_vaddr = 1;

  bits<1> has_data = 1;
  bits<1> has_glc  = 1;
  bits<1> glcValue = 0;
  bits<1> has_dlc  = 1;
  bits<1> dlcValue = 0;
  bits<1> has_sccb  = 1;
  bits<1> sccbValue = 0;
  bits<1> has_sve  = 0; // Scratch VGPR Enable
  bits<1> lds = 0;
  bits<1> sve = 0;

  let SubtargetPredicate = !if(is_flat_global, HasFlatGlobalInsts,
    !if(is_flat_scratch, HasFlatScratchInsts, HasFlatAddressSpace));

  // TODO: M0 if it could possibly access LDS (before gfx9? only)?
  let Uses = !if(is_flat_global, [EXEC], [EXEC, FLAT_SCR]);

  // Internally, FLAT instruction are executed as both an LDS and a
  // Buffer instruction; so, they increment both VM_CNT and LGKM_CNT
  // and are not considered done until both have been decremented.
  let VM_CNT = 1;
  let LGKM_CNT = !not(!or(is_flat_global, is_flat_scratch));

  let FlatGlobal = is_flat_global;

  let FlatScratch = is_flat_scratch;
}

class FLAT_Real <bits<7> op, FLAT_Pseudo ps, string opName = ps.Mnemonic> :
  InstSI <ps.OutOperandList, ps.InOperandList, opName # ps.AsmOperands, []>,
  Enc64 {

  let isPseudo = 0;
  let isCodeGenOnly = 0;

  let FLAT = 1;

  // copy relevant pseudo op flags
  let SubtargetPredicate   = ps.SubtargetPredicate;
  let AsmMatchConverter    = ps.AsmMatchConverter;
  let OtherPredicates      = ps.OtherPredicates;
  let TSFlags              = ps.TSFlags;
  let UseNamedOperandTable = ps.UseNamedOperandTable;
  let SchedRW              = ps.SchedRW;
  let mayLoad              = ps.mayLoad;
  let mayStore             = ps.mayStore;
  let IsAtomicRet          = ps.IsAtomicRet;
  let IsAtomicNoRet        = ps.IsAtomicNoRet;
  let VM_CNT               = ps.VM_CNT;
  let LGKM_CNT             = ps.LGKM_CNT;
  let VALU                 = ps.VALU;

  // encoding fields
  bits<8> vaddr;
  bits<10> vdata;
  bits<7> saddr;
  bits<10> vdst;

  bits<5> cpol;

  // Only valid on gfx9
  bits<1> lds = ps.lds; // LDS DMA for global and scratch

  // Segment, 00=flat, 01=scratch, 10=global, 11=reserved
  bits<2> seg = !if(ps.is_flat_global, 0b10,
                  !if(ps.is_flat_scratch, 0b01, 0));

  // Signed offset. Highest bit ignored for flat and treated as 12-bit
  // unsigned for flat accesses.
  bits<13> offset;
  // GFX90A+ only: instruction uses AccVGPR for data
  bits<1> acc = !if(ps.has_vdst, vdst{9}, !if(ps.has_data, vdata{9}, 0));

  // We don't use tfe right now, and it was removed in gfx9.
  bits<1> tfe = 0;

  // Only valid on GFX9+
  let Inst{12-0} = offset;
  let Inst{13} = !if(ps.has_sve, ps.sve, lds);
  let Inst{15-14} = seg;

  let Inst{16}    = !if(ps.has_glc, cpol{CPolBit.GLC}, ps.glcValue);
  let Inst{17}    = cpol{CPolBit.SLC};
  let Inst{24-18} = op;
  let Inst{31-26} = 0x37; // Encoding.
  let Inst{39-32} = !if(ps.has_vaddr, vaddr, ?);
  let Inst{47-40} = !if(ps.has_data, vdata{7-0}, ?);
  let Inst{54-48} = !if(ps.has_saddr, !if(ps.enabled_saddr, saddr, 0x7f), 0);

  // 54-48 is reserved.
  let Inst{55}    = acc; // nv on GFX9+, TFE before. AccVGPR for data on GFX90A.
  let Inst{63-56} = !if(ps.has_vdst, vdst{7-0}, ?);
}

class GlobalSaddrTable <bit is_saddr, string Name = ""> {
  bit IsSaddr = is_saddr;
  string SaddrOp = Name;
}

// TODO: Is exec allowed for saddr? The disabled value 0x7f is the
// same encoding value as exec_hi, so it isn't possible to use that if
// saddr is 32-bit (which isn't handled here yet).
class FLAT_Load_Pseudo <string opName, RegisterClass regClass,
  bit HasTiedOutput = 0,
  bit HasSaddr = 0, bit EnableSaddr = 0,
  RegisterOperand vdata_op = getLdStRegisterOperand<regClass>.ret> : FLAT_Pseudo<
  opName,
  (outs vdata_op:$vdst),
  !con(
    !con(
      !if(EnableSaddr,
        (ins SReg_64:$saddr, VGPR_32:$vaddr),
        (ins VReg_64:$vaddr)),
        (ins flat_offset:$offset)),
        // FIXME: Operands with default values do not work with following non-optional operands.
        !if(HasTiedOutput, (ins CPol:$cpol, vdata_op:$vdst_in),
                           (ins CPol_0:$cpol))),
  " $vdst, $vaddr"#!if(HasSaddr, !if(EnableSaddr, ", $saddr", ", off"), "")#"$offset$cpol"> {
  let has_data = 0;
  let mayLoad = 1;
  let has_saddr = HasSaddr;
  let enabled_saddr = EnableSaddr;
  let PseudoInstr = opName#!if(!and(HasSaddr, EnableSaddr), "_SADDR", "");
  let maybeAtomic = 1;

  let Constraints = !if(HasTiedOutput, "$vdst = $vdst_in", "");
  let DisableEncoding = !if(HasTiedOutput, "$vdst_in", "");
}

class FLAT_Store_Pseudo <string opName, RegisterClass vdataClass,
  bit HasSaddr = 0, bit EnableSaddr = 0> : FLAT_Pseudo<
  opName,
  (outs),
  !con(
    !if(EnableSaddr,
      (ins VGPR_32:$vaddr, getLdStRegisterOperand<vdataClass>.ret:$vdata, SReg_64:$saddr),
      (ins VReg_64:$vaddr, getLdStRegisterOperand<vdataClass>.ret:$vdata)),
      (ins flat_offset:$offset, CPol_0:$cpol)),
  " $vaddr, $vdata"#!if(HasSaddr, !if(EnableSaddr, ", $saddr", ", off"), "")#"$offset$cpol"> {
  let mayLoad  = 0;
  let mayStore = 1;
  let has_vdst = 0;
  let has_saddr = HasSaddr;
  let enabled_saddr = EnableSaddr;
  let PseudoInstr = opName#!if(!and(HasSaddr, EnableSaddr), "_SADDR", "");
  let maybeAtomic = 1;
}

multiclass FLAT_Global_Load_Pseudo<string opName, RegisterClass regClass, bit HasTiedInput = 0> {
  let is_flat_global = 1, SubtargetPredicate = HasFlatGlobalInsts in {
    def "" : FLAT_Load_Pseudo<opName, regClass, HasTiedInput, 1>,
      GlobalSaddrTable<0, opName>;
    def _SADDR : FLAT_Load_Pseudo<opName, regClass, HasTiedInput, 1, 1>,
      GlobalSaddrTable<1, opName>;
  }
}

class FLAT_Global_Load_AddTid_Pseudo <string opName, RegisterClass regClass,
  bit HasTiedOutput = 0, bit EnableSaddr = 0> : FLAT_Pseudo<
  opName,
  (outs regClass:$vdst),
  !con(!if(EnableSaddr, (ins SReg_64:$saddr), (ins)),
    (ins flat_offset:$offset, CPol_0:$cpol),
    !if(HasTiedOutput, (ins regClass:$vdst_in), (ins))),
  " $vdst, "#!if(EnableSaddr, "$saddr", "off")#"$offset$cpol"> {
  let is_flat_global = 1;
  let has_data = 0;
  let mayLoad = 1;
  let has_vaddr = 0;
  let has_saddr = 1;
  let enabled_saddr = EnableSaddr;
  let maybeAtomic = 1;
  let PseudoInstr = opName#!if(EnableSaddr, "_SADDR", "");

  let Constraints = !if(HasTiedOutput, "$vdst = $vdst_in", "");
  let DisableEncoding = !if(HasTiedOutput, "$vdst_in", "");
}

multiclass FLAT_Global_Load_AddTid_Pseudo<string opName, RegisterClass regClass,
  bit HasTiedOutput = 0> {
  def "" : FLAT_Global_Load_AddTid_Pseudo<opName, regClass, HasTiedOutput>,
    GlobalSaddrTable<0, opName>;
  def _SADDR : FLAT_Global_Load_AddTid_Pseudo<opName, regClass, HasTiedOutput, 1>,
    GlobalSaddrTable<1, opName>;
}

multiclass FLAT_Global_Store_Pseudo<string opName, RegisterClass regClass> {
  let is_flat_global = 1, SubtargetPredicate = HasFlatGlobalInsts in {
    def "" : FLAT_Store_Pseudo<opName, regClass, 1>,
      GlobalSaddrTable<0, opName>;
    def _SADDR : FLAT_Store_Pseudo<opName, regClass, 1, 1>,
      GlobalSaddrTable<1, opName>;
  }
}

class FLAT_Global_Load_LDS_Pseudo <string opName, bit EnableSaddr = 0> : FLAT_Pseudo<
  opName,
  (outs ),
  !con(
      !if(EnableSaddr, (ins SReg_64:$saddr, VGPR_32:$vaddr), (ins VReg_64:$vaddr)),
      (ins flat_offset:$offset, CPol_0:$cpol)),
  " $vaddr"#!if(EnableSaddr, ", $saddr", ", off")#"$offset$cpol"> {
  let LGKM_CNT = 1;
  let is_flat_global = 1;
  let lds = 1;
  let has_data = 0;
  let has_vdst = 0;
  let mayLoad = 1;
  let mayStore = 1;
  let has_saddr = 1;
  let enabled_saddr = EnableSaddr;
  let VALU = 1;
  let PseudoInstr = opName#!if(EnableSaddr, "_SADDR", "");
  let Uses = [M0, EXEC];
  let SchedRW = [WriteVMEM, WriteLDS];
}

multiclass FLAT_Global_Load_LDS_Pseudo<string opName> {
  def ""     : FLAT_Global_Load_LDS_Pseudo<opName>,
    GlobalSaddrTable<0, opName>;
  def _SADDR : FLAT_Global_Load_LDS_Pseudo<opName, 1>,
    GlobalSaddrTable<1, opName>;
}

class FLAT_Global_Store_AddTid_Pseudo <string opName, RegisterClass vdataClass,
  bit EnableSaddr = 0> : FLAT_Pseudo<
  opName,
  (outs),
  !con(!if(EnableSaddr, (ins vdataClass:$vdata, SReg_64:$saddr), (ins vdataClass:$vdata)),
    (ins flat_offset:$offset, CPol:$cpol)),
  " $vdata, "#!if(EnableSaddr, "$saddr", "off")#"$offset$cpol"> {
  let is_flat_global = 1;
  let mayLoad  = 0;
  let mayStore = 1;
  let has_vdst = 0;
  let has_vaddr = 0;
  let has_saddr = 1;
  let enabled_saddr = EnableSaddr;
  let maybeAtomic = 1;
  let PseudoInstr = opName#!if(EnableSaddr, "_SADDR", "");
}

multiclass FLAT_Global_Store_AddTid_Pseudo<string opName, RegisterClass regClass> {
  def "" : FLAT_Global_Store_AddTid_Pseudo<opName, regClass>,
    GlobalSaddrTable<0, opName>;
  def _SADDR : FLAT_Global_Store_AddTid_Pseudo<opName, regClass, 1>,
    GlobalSaddrTable<1, opName>;
}

class FlatScratchInst <string sv_op, string mode> {
  string SVOp = sv_op;
  string Mode = mode;
}

class FLAT_Scratch_Load_Pseudo <string opName, RegisterClass regClass,
  bit HasTiedOutput = 0,
  bit EnableSaddr = 0,
  bit EnableSVE = 0,
  bit EnableVaddr = !or(EnableSVE, !not(EnableSaddr))>
  : FLAT_Pseudo<
  opName,
  (outs getLdStRegisterOperand<regClass>.ret:$vdst),
  !con(
    !if(EnableSVE,
        (ins VGPR_32:$vaddr, SReg_32_XEXEC_HI:$saddr, flat_offset:$offset),
        !if(EnableSaddr,
          (ins SReg_32_XEXEC_HI:$saddr, flat_offset:$offset),
          !if(EnableVaddr,
            (ins VGPR_32:$vaddr, flat_offset:$offset),
            (ins flat_offset:$offset)))),
     !if(HasTiedOutput, (ins CPol:$cpol, getLdStRegisterOperand<regClass>.ret:$vdst_in),
                        (ins CPol_0:$cpol))),
  " $vdst, "#!if(EnableVaddr, "$vaddr, ", "off, ")#!if(EnableSaddr, "$saddr", "off")#"$offset$cpol"> {
  let has_data = 0;
  let mayLoad = 1;
  let has_saddr = 1;
  let enabled_saddr = EnableSaddr;
  let has_vaddr = EnableVaddr;
  let has_sve = EnableSVE;
  let sve = EnableVaddr;
  let PseudoInstr = opName#!if(EnableSVE, "_SVS", !if(EnableSaddr, "_SADDR", !if(EnableVaddr, "", "_ST")));
  let maybeAtomic = 1;

  let Constraints = !if(HasTiedOutput, "$vdst = $vdst_in", "");
  let DisableEncoding = !if(HasTiedOutput, "$vdst_in", "");
}

class FLAT_Scratch_Store_Pseudo <string opName, RegisterClass vdataClass, bit EnableSaddr = 0,
  bit EnableSVE = 0,
  bit EnableVaddr = !or(EnableSVE, !not(EnableSaddr)),
  RegisterOperand vdata_op = getLdStRegisterOperand<vdataClass>.ret> : FLAT_Pseudo<
  opName,
  (outs),
  !if(EnableSVE,
    (ins vdata_op:$vdata, VGPR_32:$vaddr, SReg_32_XEXEC_HI:$saddr, flat_offset:$offset, CPol_0:$cpol),
    !if(EnableSaddr,
      (ins vdata_op:$vdata, SReg_32_XEXEC_HI:$saddr, flat_offset:$offset, CPol_0:$cpol),
      !if(EnableVaddr,
        (ins vdata_op:$vdata, VGPR_32:$vaddr, flat_offset:$offset, CPol_0:$cpol),
        (ins vdata_op:$vdata, flat_offset:$offset, CPol_0:$cpol)))),
  " "#!if(EnableVaddr, "$vaddr", "off")#", $vdata, "#!if(EnableSaddr, "$saddr", "off")#"$offset$cpol"> {
  let mayLoad  = 0;
  let mayStore = 1;
  let has_vdst = 0;
  let has_saddr = 1;
  let enabled_saddr = EnableSaddr;
  let has_vaddr = EnableVaddr;
  let has_sve = EnableSVE;
  let sve = EnableVaddr;
  let PseudoInstr = opName#!if(EnableSVE, "_SVS", !if(EnableSaddr, "_SADDR", !if(EnableVaddr, "", "_ST")));
  let maybeAtomic = 1;
}

multiclass FLAT_Scratch_Load_Pseudo<string opName, RegisterClass regClass, bit HasTiedOutput = 0> {
  let is_flat_scratch = 1 in {
    def "" : FLAT_Scratch_Load_Pseudo<opName, regClass, HasTiedOutput>,
             FlatScratchInst<opName, "SV">;
    def _SADDR : FLAT_Scratch_Load_Pseudo<opName, regClass, HasTiedOutput, 1>,
                 FlatScratchInst<opName, "SS">;

    let SubtargetPredicate = HasFlatScratchSVSMode in
    def _SVS : FLAT_Scratch_Load_Pseudo<opName, regClass, HasTiedOutput, 1, 1>,
               FlatScratchInst<opName, "SVS">;

    let SubtargetPredicate = HasFlatScratchSTMode in
    def _ST  : FLAT_Scratch_Load_Pseudo<opName, regClass, HasTiedOutput, 0, 0, 0>,
               FlatScratchInst<opName, "ST">;
  }
}

multiclass FLAT_Scratch_Store_Pseudo<string opName, RegisterClass regClass> {
  let is_flat_scratch = 1 in {
    def "" : FLAT_Scratch_Store_Pseudo<opName, regClass>,
             FlatScratchInst<opName, "SV">;
    def _SADDR : FLAT_Scratch_Store_Pseudo<opName, regClass, 1>,
                 FlatScratchInst<opName, "SS">;

    let SubtargetPredicate = HasFlatScratchSVSMode in
    def _SVS : FLAT_Scratch_Store_Pseudo<opName, regClass, 1, 1>,
               FlatScratchInst<opName, "SVS">;

    let SubtargetPredicate = HasFlatScratchSTMode in
    def _ST  : FLAT_Scratch_Store_Pseudo<opName, regClass, 0, 0, 0>,
               FlatScratchInst<opName, "ST">;
  }
}

class FLAT_Scratch_Load_LDS_Pseudo <string opName, bit EnableSaddr = 0,
  bit EnableSVE = 0,
  bit EnableVaddr = !or(EnableSVE, !not(EnableSaddr))> : FLAT_Pseudo<
  opName,
  (outs ),
  !if(EnableSVE,
    (ins VGPR_32:$vaddr, SReg_32_XEXEC_HI:$saddr, flat_offset:$offset, CPol:$cpol),
    !if(EnableSaddr,
      (ins SReg_32_XEXEC_HI:$saddr, flat_offset:$offset, CPol:$cpol),
      !if(EnableVaddr,
        (ins VGPR_32:$vaddr, flat_offset:$offset, CPol:$cpol),
        (ins flat_offset:$offset, CPol:$cpol)))),
  " "#!if(EnableVaddr, "$vaddr, ", "off, ")#!if(EnableSaddr, "$saddr", "off")#"$offset$cpol"> {

  let LGKM_CNT = 1;
  let is_flat_scratch = 1;
  let lds = 1;
  let has_data = 0;
  let has_vdst = 0;
  let mayLoad = 1;
  let mayStore = 1;
  let has_saddr = 1;
  let enabled_saddr = EnableSaddr;
  let has_vaddr = EnableVaddr;
  let has_sve = EnableSVE;
  let sve = EnableVaddr;
  let VALU = 1;
  let PseudoInstr = opName#!if(EnableSVE, "_SVS", !if(EnableSaddr, "_SADDR", !if(EnableVaddr, "", "_ST")));
  let Uses = [M0, EXEC];
  let SchedRW = [WriteVMEM, WriteLDS];
}

multiclass FLAT_Scratch_Load_LDS_Pseudo<string opName> {
  def ""     : FLAT_Scratch_Load_LDS_Pseudo<opName>,
               FlatScratchInst<opName, "SV">;
  def _SADDR : FLAT_Scratch_Load_LDS_Pseudo<opName, 1>,
               FlatScratchInst<opName, "SS">;
  def _SVS   : FLAT_Scratch_Load_LDS_Pseudo<opName, 1, 1>,
               FlatScratchInst<opName, "SVS">;
  def _ST    : FLAT_Scratch_Load_LDS_Pseudo<opName, 0, 0, 0>,
               FlatScratchInst<opName, "ST">;
}

class FLAT_AtomicNoRet_Pseudo<string opName, dag outs, dag ins,
                               string asm, list<dag> pattern = []> :
  FLAT_Pseudo<opName, outs, ins, asm, pattern> {
    let mayLoad = 1;
    let mayStore = 1;
    let has_glc  = 0;
    let glcValue = 0;
    let has_vdst = 0;
    let has_sccb  = 1;
    let sccbValue = 0;
    let maybeAtomic = 1;
    let IsAtomicNoRet = 1;
}

class FLAT_AtomicRet_Pseudo<string opName, dag outs, dag ins,
                            string asm, list<dag> pattern = []>
  : FLAT_AtomicNoRet_Pseudo<opName, outs, ins, asm, pattern> {
  let hasPostISelHook = 1;
  let has_vdst = 1;
  let glcValue = 1;
  let sccbValue = 0;
  let IsAtomicNoRet = 0;
  let IsAtomicRet = 1;
  let PseudoInstr = NAME # "_RTN";
}

multiclass FLAT_Atomic_Pseudo<
  string opName,
  RegisterClass vdst_rc,
  ValueType vt,
  ValueType data_vt = vt,
  RegisterClass data_rc = vdst_rc,
  bit isFP = isFloatType<data_vt>.ret,
  RegisterOperand data_op = getLdStRegisterOperand<data_rc>.ret> {
  def "" : FLAT_AtomicNoRet_Pseudo <opName,
    (outs),
    (ins VReg_64:$vaddr, data_op:$vdata, flat_offset:$offset, CPol_0:$cpol),
    " $vaddr, $vdata$offset$cpol">,
    GlobalSaddrTable<0, opName>,
    AtomicNoRet <opName, 0> {
    let PseudoInstr = NAME;
    let FPAtomic = isFP;
    let AddedComplexity = -1; // Prefer global atomics if available
  }

  def _RTN : FLAT_AtomicRet_Pseudo <opName,
    (outs getLdStRegisterOperand<vdst_rc>.ret:$vdst),
    (ins VReg_64:$vaddr, data_op:$vdata, flat_offset:$offset, CPol_GLC1:$cpol),
    " $vdst, $vaddr, $vdata$offset$cpol">,
    GlobalSaddrTable<0, opName#"_rtn">,
    AtomicNoRet <opName, 1> {
    let FPAtomic = isFP;
    let AddedComplexity = -1; // Prefer global atomics if available
  }
}

multiclass FLAT_Global_Atomic_Pseudo_NO_RTN<
  string opName,
  RegisterClass vdst_rc,
  ValueType vt,
  ValueType data_vt = vt,
  RegisterClass data_rc = vdst_rc,
  bit isFP = isFloatType<data_vt>.ret,
  RegisterOperand data_op = getLdStRegisterOperand<data_rc>.ret> {

  def "" : FLAT_AtomicNoRet_Pseudo <opName,
    (outs),
    (ins VReg_64:$vaddr, data_op:$vdata, flat_offset:$offset, CPol_0:$cpol),
    " $vaddr, $vdata, off$offset$cpol">,
    GlobalSaddrTable<0, opName>,
    AtomicNoRet <opName, 0> {
    let has_saddr = 1;
    let PseudoInstr = NAME;
    let FPAtomic = isFP;
  }

  def _SADDR : FLAT_AtomicNoRet_Pseudo <opName,
    (outs),
    (ins VGPR_32:$vaddr, data_op:$vdata, SReg_64:$saddr, flat_offset:$offset, CPol_0:$cpol),
    " $vaddr, $vdata, $saddr$offset$cpol">,
    GlobalSaddrTable<1, opName>,
    AtomicNoRet <opName#"_saddr", 0> {
    let has_saddr = 1;
    let enabled_saddr = 1;
    let PseudoInstr = NAME#"_SADDR";
    let FPAtomic = isFP;
  }
}

multiclass FLAT_Global_Atomic_Pseudo_RTN<
  string opName,
  RegisterClass vdst_rc,
  ValueType vt,
  ValueType data_vt = vt,
  RegisterClass data_rc = vdst_rc,
  bit isFP = isFloatType<data_vt>.ret,
  RegisterOperand data_op = getLdStRegisterOperand<data_rc>.ret,
  RegisterOperand vdst_op = getLdStRegisterOperand<vdst_rc>.ret> {

  def _RTN : FLAT_AtomicRet_Pseudo <opName,
    (outs vdst_op:$vdst),
      (ins VReg_64:$vaddr, data_op:$vdata, flat_offset:$offset, CPol_GLC1:$cpol),
    " $vdst, $vaddr, $vdata, off$offset$cpol">,
    GlobalSaddrTable<0, opName#"_rtn">,
    AtomicNoRet <opName, 1> {
    let has_saddr = 1;
    let FPAtomic = isFP;
  }

  def _SADDR_RTN : FLAT_AtomicRet_Pseudo <opName,
    (outs vdst_op:$vdst),
      (ins VGPR_32:$vaddr, data_op:$vdata, SReg_64:$saddr, flat_offset:$offset, CPol_GLC1:$cpol),
    " $vdst, $vaddr, $vdata, $saddr$offset$cpol">,
    GlobalSaddrTable<1, opName#"_rtn">,
    AtomicNoRet <opName#"_saddr", 1> {
     let has_saddr = 1;
     let enabled_saddr = 1;
     let PseudoInstr = NAME#"_SADDR_RTN";
     let FPAtomic = isFP;
  }
}

multiclass FLAT_Global_Atomic_Pseudo<
  string opName,
  RegisterClass vdst_rc,
  ValueType vt,
  ValueType data_vt = vt,
  RegisterClass data_rc = vdst_rc> {
  let is_flat_global = 1, SubtargetPredicate = HasFlatGlobalInsts in {
    defm "" : FLAT_Global_Atomic_Pseudo_NO_RTN<opName, vdst_rc, vt, data_vt, data_rc>;
    defm "" : FLAT_Global_Atomic_Pseudo_RTN<opName, vdst_rc, vt, data_vt, data_rc>;
  }
}

//===----------------------------------------------------------------------===//
// Flat Instructions
//===----------------------------------------------------------------------===//

def FLAT_LOAD_UBYTE    : FLAT_Load_Pseudo <"flat_load_ubyte", VGPR_32>;
def FLAT_LOAD_SBYTE    : FLAT_Load_Pseudo <"flat_load_sbyte", VGPR_32>;
def FLAT_LOAD_USHORT   : FLAT_Load_Pseudo <"flat_load_ushort", VGPR_32>;
def FLAT_LOAD_SSHORT   : FLAT_Load_Pseudo <"flat_load_sshort", VGPR_32>;
def FLAT_LOAD_DWORD    : FLAT_Load_Pseudo <"flat_load_dword", VGPR_32>;
def FLAT_LOAD_DWORDX2  : FLAT_Load_Pseudo <"flat_load_dwordx2", VReg_64>;
def FLAT_LOAD_DWORDX4  : FLAT_Load_Pseudo <"flat_load_dwordx4", VReg_128>;
def FLAT_LOAD_DWORDX3  : FLAT_Load_Pseudo <"flat_load_dwordx3", VReg_96>;

def FLAT_STORE_BYTE    : FLAT_Store_Pseudo <"flat_store_byte", VGPR_32>;
def FLAT_STORE_SHORT   : FLAT_Store_Pseudo <"flat_store_short", VGPR_32>;
def FLAT_STORE_DWORD   : FLAT_Store_Pseudo <"flat_store_dword", VGPR_32>;
def FLAT_STORE_DWORDX2 : FLAT_Store_Pseudo <"flat_store_dwordx2", VReg_64>;
def FLAT_STORE_DWORDX4 : FLAT_Store_Pseudo <"flat_store_dwordx4", VReg_128>;
def FLAT_STORE_DWORDX3 : FLAT_Store_Pseudo <"flat_store_dwordx3", VReg_96>;

let SubtargetPredicate = HasD16LoadStore in {
let TiedSourceNotRead = 1 in {
def FLAT_LOAD_UBYTE_D16     : FLAT_Load_Pseudo <"flat_load_ubyte_d16", VGPR_32, 1>;
def FLAT_LOAD_UBYTE_D16_HI  : FLAT_Load_Pseudo <"flat_load_ubyte_d16_hi", VGPR_32, 1>;
def FLAT_LOAD_SBYTE_D16     : FLAT_Load_Pseudo <"flat_load_sbyte_d16", VGPR_32, 1>;
def FLAT_LOAD_SBYTE_D16_HI  : FLAT_Load_Pseudo <"flat_load_sbyte_d16_hi", VGPR_32, 1>;
def FLAT_LOAD_SHORT_D16     : FLAT_Load_Pseudo <"flat_load_short_d16", VGPR_32, 1>;
def FLAT_LOAD_SHORT_D16_HI  : FLAT_Load_Pseudo <"flat_load_short_d16_hi", VGPR_32, 1>;
}

def FLAT_STORE_BYTE_D16_HI  : FLAT_Store_Pseudo <"flat_store_byte_d16_hi", VGPR_32>;
def FLAT_STORE_SHORT_D16_HI : FLAT_Store_Pseudo <"flat_store_short_d16_hi", VGPR_32>;
}

defm FLAT_ATOMIC_CMPSWAP    : FLAT_Atomic_Pseudo <"flat_atomic_cmpswap",
                                VGPR_32, i32, v2i32, VReg_64>;

defm FLAT_ATOMIC_CMPSWAP_X2 : FLAT_Atomic_Pseudo <"flat_atomic_cmpswap_x2",
                                VReg_64, i64, v2i64, VReg_128>;

defm FLAT_ATOMIC_SWAP       : FLAT_Atomic_Pseudo <"flat_atomic_swap",
                                VGPR_32, i32>;

defm FLAT_ATOMIC_SWAP_X2    : FLAT_Atomic_Pseudo <"flat_atomic_swap_x2",
                                VReg_64, i64>;

defm FLAT_ATOMIC_ADD        : FLAT_Atomic_Pseudo <"flat_atomic_add",
                                VGPR_32, i32>;

defm FLAT_ATOMIC_SUB        : FLAT_Atomic_Pseudo <"flat_atomic_sub",
                                VGPR_32, i32>;

defm FLAT_ATOMIC_SMIN       : FLAT_Atomic_Pseudo <"flat_atomic_smin",
                                VGPR_32, i32>;

defm FLAT_ATOMIC_UMIN       : FLAT_Atomic_Pseudo <"flat_atomic_umin",
                                VGPR_32, i32>;

defm FLAT_ATOMIC_SMAX       : FLAT_Atomic_Pseudo <"flat_atomic_smax",
                                VGPR_32, i32>;

defm FLAT_ATOMIC_UMAX       : FLAT_Atomic_Pseudo <"flat_atomic_umax",
                                VGPR_32, i32>;

defm FLAT_ATOMIC_AND        : FLAT_Atomic_Pseudo <"flat_atomic_and",
                                VGPR_32, i32>;

defm FLAT_ATOMIC_OR         : FLAT_Atomic_Pseudo <"flat_atomic_or",
                                VGPR_32, i32>;

defm FLAT_ATOMIC_XOR        : FLAT_Atomic_Pseudo <"flat_atomic_xor",
                                VGPR_32, i32>;

defm FLAT_ATOMIC_INC        : FLAT_Atomic_Pseudo <"flat_atomic_inc",
                                VGPR_32, i32>;

defm FLAT_ATOMIC_DEC        : FLAT_Atomic_Pseudo <"flat_atomic_dec",
                                VGPR_32, i32>;

defm FLAT_ATOMIC_ADD_X2     : FLAT_Atomic_Pseudo <"flat_atomic_add_x2",
                                VReg_64, i64>;

defm FLAT_ATOMIC_SUB_X2     : FLAT_Atomic_Pseudo <"flat_atomic_sub_x2",
                                VReg_64, i64>;

defm FLAT_ATOMIC_SMIN_X2    : FLAT_Atomic_Pseudo <"flat_atomic_smin_x2",
                                VReg_64, i64>;

defm FLAT_ATOMIC_UMIN_X2    : FLAT_Atomic_Pseudo <"flat_atomic_umin_x2",
                                VReg_64, i64>;

defm FLAT_ATOMIC_SMAX_X2    : FLAT_Atomic_Pseudo <"flat_atomic_smax_x2",
                                VReg_64, i64>;

defm FLAT_ATOMIC_UMAX_X2    : FLAT_Atomic_Pseudo <"flat_atomic_umax_x2",
                                VReg_64, i64>;

defm FLAT_ATOMIC_AND_X2     : FLAT_Atomic_Pseudo <"flat_atomic_and_x2",
                                VReg_64, i64>;

defm FLAT_ATOMIC_OR_X2      : FLAT_Atomic_Pseudo <"flat_atomic_or_x2",
                                VReg_64, i64>;

defm FLAT_ATOMIC_XOR_X2     : FLAT_Atomic_Pseudo <"flat_atomic_xor_x2",
                                VReg_64, i64>;

defm FLAT_ATOMIC_INC_X2     : FLAT_Atomic_Pseudo <"flat_atomic_inc_x2",
                                VReg_64, i64>;

defm FLAT_ATOMIC_DEC_X2     : FLAT_Atomic_Pseudo <"flat_atomic_dec_x2",
                                VReg_64, i64>;

// GFX7-, GFX10-only flat instructions.
let SubtargetPredicate = isGFX7GFX10 in {

defm FLAT_ATOMIC_FCMPSWAP_X2 : FLAT_Atomic_Pseudo <"flat_atomic_fcmpswap_x2",
                                VReg_64, f64, v2f64, VReg_128>;

defm FLAT_ATOMIC_FMIN_X2     : FLAT_Atomic_Pseudo <"flat_atomic_fmin_x2",
                                VReg_64, f64>;

defm FLAT_ATOMIC_FMAX_X2     : FLAT_Atomic_Pseudo <"flat_atomic_fmax_x2",
                                VReg_64, f64>;

} // End SubtargetPredicate = isGFX7GFX10

let SubtargetPredicate = isGFX90APlus in {
  defm FLAT_ATOMIC_ADD_F64   : FLAT_Atomic_Pseudo<"flat_atomic_add_f64", VReg_64, f64>;
  defm FLAT_ATOMIC_MIN_F64   : FLAT_Atomic_Pseudo<"flat_atomic_min_f64", VReg_64, f64>;
  defm FLAT_ATOMIC_MAX_F64   : FLAT_Atomic_Pseudo<"flat_atomic_max_f64", VReg_64, f64>;
  defm GLOBAL_ATOMIC_ADD_F64 : FLAT_Global_Atomic_Pseudo<"global_atomic_add_f64", VReg_64, f64>;
  defm GLOBAL_ATOMIC_MIN_F64 : FLAT_Global_Atomic_Pseudo<"global_atomic_min_f64", VReg_64, f64>;
  defm GLOBAL_ATOMIC_MAX_F64 : FLAT_Global_Atomic_Pseudo<"global_atomic_max_f64", VReg_64, f64>;
} // End SubtargetPredicate = isGFX90APlus

let SubtargetPredicate = isGFX940Plus in {
  defm FLAT_ATOMIC_PK_ADD_F16    : FLAT_Atomic_Pseudo<"flat_atomic_pk_add_f16",  VGPR_32, v2f16>;
  defm FLAT_ATOMIC_PK_ADD_BF16   : FLAT_Atomic_Pseudo<"flat_atomic_pk_add_bf16", VGPR_32, v2f16>;
  defm GLOBAL_ATOMIC_PK_ADD_BF16 : FLAT_Global_Atomic_Pseudo<"global_atomic_pk_add_bf16", VGPR_32, v2f16>;
} // End SubtargetPredicate = isGFX940Plus

// GFX7-, GFX10-, GFX11-only flat instructions.
let SubtargetPredicate = isGFX7GFX10GFX11 in {

defm FLAT_ATOMIC_FCMPSWAP    : FLAT_Atomic_Pseudo <"flat_atomic_fcmpswap",
                                VGPR_32, f32, v2f32, VReg_64>;

defm FLAT_ATOMIC_FMIN        : FLAT_Atomic_Pseudo <"flat_atomic_fmin",
                                VGPR_32, f32>;

defm FLAT_ATOMIC_FMAX        : FLAT_Atomic_Pseudo <"flat_atomic_fmax",
                                VGPR_32, f32>;

} // End SubtargetPredicate = isGFX7GFX10GFX11

// GFX940-, GFX11-only flat instructions.
let SubtargetPredicate = HasFlatAtomicFaddF32Inst in {
  defm FLAT_ATOMIC_ADD_F32       : FLAT_Atomic_Pseudo<"flat_atomic_add_f32",     VGPR_32, f32>;
} // End SubtargetPredicate = HasFlatAtomicFaddF32Inst

defm GLOBAL_LOAD_UBYTE    : FLAT_Global_Load_Pseudo <"global_load_ubyte", VGPR_32>;
defm GLOBAL_LOAD_SBYTE    : FLAT_Global_Load_Pseudo <"global_load_sbyte", VGPR_32>;
defm GLOBAL_LOAD_USHORT   : FLAT_Global_Load_Pseudo <"global_load_ushort", VGPR_32>;
defm GLOBAL_LOAD_SSHORT   : FLAT_Global_Load_Pseudo <"global_load_sshort", VGPR_32>;
defm GLOBAL_LOAD_DWORD    : FLAT_Global_Load_Pseudo <"global_load_dword", VGPR_32>;
defm GLOBAL_LOAD_DWORDX2  : FLAT_Global_Load_Pseudo <"global_load_dwordx2", VReg_64>;
defm GLOBAL_LOAD_DWORDX3  : FLAT_Global_Load_Pseudo <"global_load_dwordx3", VReg_96>;
defm GLOBAL_LOAD_DWORDX4  : FLAT_Global_Load_Pseudo <"global_load_dwordx4", VReg_128>;

let TiedSourceNotRead = 1 in {
defm GLOBAL_LOAD_UBYTE_D16    : FLAT_Global_Load_Pseudo <"global_load_ubyte_d16", VGPR_32, 1>;
defm GLOBAL_LOAD_UBYTE_D16_HI : FLAT_Global_Load_Pseudo <"global_load_ubyte_d16_hi", VGPR_32, 1>;
defm GLOBAL_LOAD_SBYTE_D16    : FLAT_Global_Load_Pseudo <"global_load_sbyte_d16", VGPR_32, 1>;
defm GLOBAL_LOAD_SBYTE_D16_HI : FLAT_Global_Load_Pseudo <"global_load_sbyte_d16_hi", VGPR_32, 1>;
defm GLOBAL_LOAD_SHORT_D16    : FLAT_Global_Load_Pseudo <"global_load_short_d16", VGPR_32, 1>;
defm GLOBAL_LOAD_SHORT_D16_HI : FLAT_Global_Load_Pseudo <"global_load_short_d16_hi", VGPR_32, 1>;
}

let OtherPredicates = [HasGFX10_BEncoding] in
defm GLOBAL_LOAD_DWORD_ADDTID : FLAT_Global_Load_AddTid_Pseudo <"global_load_dword_addtid", VGPR_32>;

defm GLOBAL_STORE_BYTE    : FLAT_Global_Store_Pseudo <"global_store_byte", VGPR_32>;
defm GLOBAL_STORE_SHORT   : FLAT_Global_Store_Pseudo <"global_store_short", VGPR_32>;
defm GLOBAL_STORE_DWORD   : FLAT_Global_Store_Pseudo <"global_store_dword", VGPR_32>;
defm GLOBAL_STORE_DWORDX2 : FLAT_Global_Store_Pseudo <"global_store_dwordx2", VReg_64>;
defm GLOBAL_STORE_DWORDX3 : FLAT_Global_Store_Pseudo <"global_store_dwordx3", VReg_96>;
defm GLOBAL_STORE_DWORDX4 : FLAT_Global_Store_Pseudo <"global_store_dwordx4", VReg_128>;
let OtherPredicates = [HasGFX10_BEncoding] in
defm GLOBAL_STORE_DWORD_ADDTID : FLAT_Global_Store_AddTid_Pseudo <"global_store_dword_addtid", VGPR_32>;

defm GLOBAL_STORE_BYTE_D16_HI  : FLAT_Global_Store_Pseudo <"global_store_byte_d16_hi", VGPR_32>;
defm GLOBAL_STORE_SHORT_D16_HI : FLAT_Global_Store_Pseudo <"global_store_short_d16_hi", VGPR_32>;

let is_flat_global = 1 in {
defm GLOBAL_ATOMIC_CMPSWAP : FLAT_Global_Atomic_Pseudo <"global_atomic_cmpswap",
                               VGPR_32, i32, v2i32, VReg_64>;

defm GLOBAL_ATOMIC_CMPSWAP_X2 : FLAT_Global_Atomic_Pseudo <"global_atomic_cmpswap_x2",
                                  VReg_64, i64, v2i64, VReg_128>;

defm GLOBAL_ATOMIC_SWAP : FLAT_Global_Atomic_Pseudo <"global_atomic_swap",
                             VGPR_32, i32>;

defm GLOBAL_ATOMIC_SWAP_X2 : FLAT_Global_Atomic_Pseudo <"global_atomic_swap_x2",
                                VReg_64, i64>;

defm GLOBAL_ATOMIC_ADD : FLAT_Global_Atomic_Pseudo <"global_atomic_add",
                           VGPR_32, i32>;

defm GLOBAL_ATOMIC_SUB : FLAT_Global_Atomic_Pseudo <"global_atomic_sub",
                           VGPR_32, i32>;

defm GLOBAL_ATOMIC_SMIN : FLAT_Global_Atomic_Pseudo <"global_atomic_smin",
                            VGPR_32, i32>;

defm GLOBAL_ATOMIC_UMIN : FLAT_Global_Atomic_Pseudo <"global_atomic_umin",
                            VGPR_32, i32>;

defm GLOBAL_ATOMIC_SMAX : FLAT_Global_Atomic_Pseudo <"global_atomic_smax",
                            VGPR_32, i32>;

defm GLOBAL_ATOMIC_UMAX : FLAT_Global_Atomic_Pseudo <"global_atomic_umax",
                            VGPR_32, i32>;

defm GLOBAL_ATOMIC_AND : FLAT_Global_Atomic_Pseudo <"global_atomic_and",
                           VGPR_32, i32>;

defm GLOBAL_ATOMIC_OR : FLAT_Global_Atomic_Pseudo <"global_atomic_or",
                          VGPR_32, i32>;

defm GLOBAL_ATOMIC_XOR : FLAT_Global_Atomic_Pseudo <"global_atomic_xor",
                           VGPR_32, i32>;

defm GLOBAL_ATOMIC_INC : FLAT_Global_Atomic_Pseudo <"global_atomic_inc",
                           VGPR_32, i32>;

defm GLOBAL_ATOMIC_DEC : FLAT_Global_Atomic_Pseudo <"global_atomic_dec",
                           VGPR_32, i32>;

defm GLOBAL_ATOMIC_ADD_X2 : FLAT_Global_Atomic_Pseudo <"global_atomic_add_x2",
                              VReg_64, i64>;

defm GLOBAL_ATOMIC_SUB_X2 : FLAT_Global_Atomic_Pseudo <"global_atomic_sub_x2",
                              VReg_64, i64>;

defm GLOBAL_ATOMIC_SMIN_X2 : FLAT_Global_Atomic_Pseudo <"global_atomic_smin_x2",
                               VReg_64, i64>;

defm GLOBAL_ATOMIC_UMIN_X2 : FLAT_Global_Atomic_Pseudo <"global_atomic_umin_x2",
                               VReg_64, i64>;

defm GLOBAL_ATOMIC_SMAX_X2 : FLAT_Global_Atomic_Pseudo <"global_atomic_smax_x2",
                               VReg_64, i64>;

defm GLOBAL_ATOMIC_UMAX_X2 : FLAT_Global_Atomic_Pseudo <"global_atomic_umax_x2",
                               VReg_64, i64>;

defm GLOBAL_ATOMIC_AND_X2 : FLAT_Global_Atomic_Pseudo <"global_atomic_and_x2",
                              VReg_64, i64>;

defm GLOBAL_ATOMIC_OR_X2 : FLAT_Global_Atomic_Pseudo <"global_atomic_or_x2",
                             VReg_64, i64>;

defm GLOBAL_ATOMIC_XOR_X2 : FLAT_Global_Atomic_Pseudo <"global_atomic_xor_x2",
                              VReg_64, i64>;

defm GLOBAL_ATOMIC_INC_X2 : FLAT_Global_Atomic_Pseudo <"global_atomic_inc_x2",
                              VReg_64, i64>;

defm GLOBAL_ATOMIC_DEC_X2 : FLAT_Global_Atomic_Pseudo <"global_atomic_dec_x2",
                              VReg_64, i64>;

let SubtargetPredicate = HasGFX10_BEncoding in
defm GLOBAL_ATOMIC_CSUB : FLAT_Global_Atomic_Pseudo_RTN <"global_atomic_csub",
                              VGPR_32, i32>;

defm GLOBAL_LOAD_LDS_UBYTE  : FLAT_Global_Load_LDS_Pseudo <"global_load_lds_ubyte">;
defm GLOBAL_LOAD_LDS_SBYTE  : FLAT_Global_Load_LDS_Pseudo <"global_load_lds_sbyte">;
defm GLOBAL_LOAD_LDS_USHORT : FLAT_Global_Load_LDS_Pseudo <"global_load_lds_ushort">;
defm GLOBAL_LOAD_LDS_SSHORT : FLAT_Global_Load_LDS_Pseudo <"global_load_lds_sshort">;
defm GLOBAL_LOAD_LDS_DWORD  : FLAT_Global_Load_LDS_Pseudo <"global_load_lds_dword">;

} // End is_flat_global = 1



let SubtargetPredicate = HasFlatScratchInsts in {
defm SCRATCH_LOAD_UBYTE    : FLAT_Scratch_Load_Pseudo <"scratch_load_ubyte", VGPR_32>;
defm SCRATCH_LOAD_SBYTE    : FLAT_Scratch_Load_Pseudo <"scratch_load_sbyte", VGPR_32>;
defm SCRATCH_LOAD_USHORT   : FLAT_Scratch_Load_Pseudo <"scratch_load_ushort", VGPR_32>;
defm SCRATCH_LOAD_SSHORT   : FLAT_Scratch_Load_Pseudo <"scratch_load_sshort", VGPR_32>;
defm SCRATCH_LOAD_DWORD    : FLAT_Scratch_Load_Pseudo <"scratch_load_dword", VGPR_32>;
defm SCRATCH_LOAD_DWORDX2  : FLAT_Scratch_Load_Pseudo <"scratch_load_dwordx2", VReg_64>;
defm SCRATCH_LOAD_DWORDX3  : FLAT_Scratch_Load_Pseudo <"scratch_load_dwordx3", VReg_96>;
defm SCRATCH_LOAD_DWORDX4  : FLAT_Scratch_Load_Pseudo <"scratch_load_dwordx4", VReg_128>;

let TiedSourceNotRead = 1 in {
defm SCRATCH_LOAD_UBYTE_D16    : FLAT_Scratch_Load_Pseudo <"scratch_load_ubyte_d16", VGPR_32, 1>;
defm SCRATCH_LOAD_UBYTE_D16_HI : FLAT_Scratch_Load_Pseudo <"scratch_load_ubyte_d16_hi", VGPR_32, 1>;
defm SCRATCH_LOAD_SBYTE_D16    : FLAT_Scratch_Load_Pseudo <"scratch_load_sbyte_d16", VGPR_32, 1>;
defm SCRATCH_LOAD_SBYTE_D16_HI : FLAT_Scratch_Load_Pseudo <"scratch_load_sbyte_d16_hi", VGPR_32, 1>;
defm SCRATCH_LOAD_SHORT_D16    : FLAT_Scratch_Load_Pseudo <"scratch_load_short_d16", VGPR_32, 1>;
defm SCRATCH_LOAD_SHORT_D16_HI : FLAT_Scratch_Load_Pseudo <"scratch_load_short_d16_hi", VGPR_32, 1>;
}

defm SCRATCH_STORE_BYTE    : FLAT_Scratch_Store_Pseudo <"scratch_store_byte", VGPR_32>;
defm SCRATCH_STORE_SHORT   : FLAT_Scratch_Store_Pseudo <"scratch_store_short", VGPR_32>;
defm SCRATCH_STORE_DWORD   : FLAT_Scratch_Store_Pseudo <"scratch_store_dword", VGPR_32>;
defm SCRATCH_STORE_DWORDX2 : FLAT_Scratch_Store_Pseudo <"scratch_store_dwordx2", VReg_64>;
defm SCRATCH_STORE_DWORDX3 : FLAT_Scratch_Store_Pseudo <"scratch_store_dwordx3", VReg_96>;
defm SCRATCH_STORE_DWORDX4 : FLAT_Scratch_Store_Pseudo <"scratch_store_dwordx4", VReg_128>;

defm SCRATCH_STORE_BYTE_D16_HI : FLAT_Scratch_Store_Pseudo <"scratch_store_byte_d16_hi", VGPR_32>;
defm SCRATCH_STORE_SHORT_D16_HI : FLAT_Scratch_Store_Pseudo <"scratch_store_short_d16_hi", VGPR_32>;

defm SCRATCH_LOAD_LDS_UBYTE  : FLAT_Scratch_Load_LDS_Pseudo <"scratch_load_lds_ubyte">;
defm SCRATCH_LOAD_LDS_SBYTE  : FLAT_Scratch_Load_LDS_Pseudo <"scratch_load_lds_sbyte">;
defm SCRATCH_LOAD_LDS_USHORT : FLAT_Scratch_Load_LDS_Pseudo <"scratch_load_lds_ushort">;
defm SCRATCH_LOAD_LDS_SSHORT : FLAT_Scratch_Load_LDS_Pseudo <"scratch_load_lds_sshort">;
defm SCRATCH_LOAD_LDS_DWORD  : FLAT_Scratch_Load_LDS_Pseudo <"scratch_load_lds_dword">;

} // End SubtargetPredicate = HasFlatScratchInsts

let SubtargetPredicate = isGFX10Plus, is_flat_global = 1 in {
  defm GLOBAL_ATOMIC_FCMPSWAP :
    FLAT_Global_Atomic_Pseudo<"global_atomic_fcmpswap", VGPR_32, f32, v2f32, VReg_64>;
  defm GLOBAL_ATOMIC_FMIN :
    FLAT_Global_Atomic_Pseudo<"global_atomic_fmin", VGPR_32, f32>;
  defm GLOBAL_ATOMIC_FMAX :
    FLAT_Global_Atomic_Pseudo<"global_atomic_fmax", VGPR_32, f32>;
  defm GLOBAL_ATOMIC_FCMPSWAP_X2 :
    FLAT_Global_Atomic_Pseudo<"global_atomic_fcmpswap_x2", VReg_64, f64, v2f64, VReg_128>;
  defm GLOBAL_ATOMIC_FMIN_X2 :
    FLAT_Global_Atomic_Pseudo<"global_atomic_fmin_x2", VReg_64, f64>;
  defm GLOBAL_ATOMIC_FMAX_X2 :
    FLAT_Global_Atomic_Pseudo<"global_atomic_fmax_x2", VReg_64, f64>;
} // End SubtargetPredicate = isGFX10Plus, is_flat_global = 1

let is_flat_global = 1 in {
let OtherPredicates = [HasAtomicFaddNoRtnInsts] in
  defm GLOBAL_ATOMIC_ADD_F32 : FLAT_Global_Atomic_Pseudo_NO_RTN <
    "global_atomic_add_f32", VGPR_32, f32
  >;
let OtherPredicates = [HasAtomicPkFaddNoRtnInsts] in
  defm GLOBAL_ATOMIC_PK_ADD_F16 : FLAT_Global_Atomic_Pseudo_NO_RTN <
    "global_atomic_pk_add_f16", VGPR_32, v2f16
  >;
let OtherPredicates = [HasAtomicFaddRtnInsts] in
  defm GLOBAL_ATOMIC_ADD_F32 : FLAT_Global_Atomic_Pseudo_RTN <
    "global_atomic_add_f32", VGPR_32, f32
  >;
let OtherPredicates = [isGFX90APlus] in
  defm GLOBAL_ATOMIC_PK_ADD_F16 : FLAT_Global_Atomic_Pseudo_RTN <
    "global_atomic_pk_add_f16", VGPR_32, v2f16
  >;
} // End is_flat_global = 1

//===----------------------------------------------------------------------===//
// Flat Patterns
//===----------------------------------------------------------------------===//

// Patterns for global loads with no offset.
class FlatLoadPat <FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> : GCNPat <
  (vt (node (FlatOffset i64:$vaddr, i32:$offset))),
  (inst $vaddr, $offset)
>;

class FlatLoadPat_D16 <FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> : GCNPat <
  (node (FlatOffset (i64 VReg_64:$vaddr), i32:$offset), vt:$in),
  (inst $vaddr, $offset, 0, $in)
>;

class FlatSignedLoadPat_D16 <FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> : GCNPat <
  (node (GlobalOffset (i64 VReg_64:$vaddr), i32:$offset), vt:$in),
  (inst $vaddr, $offset, 0, $in)
>;

class GlobalLoadSaddrPat_D16 <FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> : GCNPat <
  (vt (node (GlobalSAddr (i64 SReg_64:$saddr), (i32 VGPR_32:$voffset), i32:$offset), vt:$in)),
  (inst $saddr, $voffset, $offset, 0, $in)
>;

class FlatLoadSignedPat <FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> : GCNPat <
  (vt (node (GlobalOffset (i64 VReg_64:$vaddr), i32:$offset))),
  (inst $vaddr, $offset)
>;

class GlobalLoadSaddrPat <FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> : GCNPat <
  (vt (node (GlobalSAddr (i64 SReg_64:$saddr), (i32 VGPR_32:$voffset), i32:$offset))),
  (inst $saddr, $voffset, $offset, 0)
>;

class GlobalStoreSaddrPat <FLAT_Pseudo inst, SDPatternOperator node,
                           ValueType vt> : GCNPat <
  (node vt:$data, (GlobalSAddr (i64 SReg_64:$saddr), (i32 VGPR_32:$voffset), i32:$offset)),
  (inst $voffset, getVregSrcForVT<vt>.ret:$data, $saddr, $offset)
>;

class GlobalAtomicStoreSaddrPat <FLAT_Pseudo inst, SDPatternOperator node,
                                 ValueType vt> : GCNPat <
  (node (GlobalSAddr (i64 SReg_64:$saddr), (i32 VGPR_32:$voffset), i32:$offset), vt:$data),
  (inst $voffset, getVregSrcForVT<vt>.ret:$data, $saddr, $offset)
>;

class GlobalAtomicSaddrPat <FLAT_Pseudo inst, SDPatternOperator node,
                            ValueType vt, ValueType data_vt = vt> : GCNPat <
  (vt (node (GlobalSAddr (i64 SReg_64:$saddr), (i32 VGPR_32:$voffset), i32:$offset), data_vt:$data)),
  (inst $voffset, getVregSrcForVT<data_vt>.ret:$data, $saddr, $offset)
>;

class GlobalAtomicNoRtnSaddrPat <FLAT_Pseudo inst, SDPatternOperator node,
                                 ValueType vt> : GCNPat <
  (node (GlobalSAddr (i64 SReg_64:$saddr), (i32 VGPR_32:$voffset), i32:$offset), vt:$data),
  (inst $voffset, getVregSrcForVT<vt>.ret:$data, $saddr, $offset)
>;

class FlatStorePat <FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> : GCNPat <
  (node vt:$data, (FlatOffset i64:$vaddr, i32:$offset)),
  (inst $vaddr, getVregSrcForVT<vt>.ret:$data, $offset)
>;

class FlatStoreSignedPat <FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> : GCNPat <
  (node vt:$data, (GlobalOffset i64:$vaddr, i32:$offset)),
  (inst $vaddr, getVregSrcForVT<vt>.ret:$data, $offset)
>;

class FlatStoreAtomicPat <FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> : GCNPat <
  // atomic store follows atomic binop convention so the address comes
  // first.
  (node (FlatOffset i64:$vaddr, i32:$offset), vt:$data),
  (inst $vaddr, getVregSrcForVT<vt>.ret:$data, $offset)
>;

class FlatStoreSignedAtomicPat <FLAT_Pseudo inst, SDPatternOperator node,
                                ValueType vt, ValueType data_vt = vt> : GCNPat <
  // atomic store follows atomic binop convention so the address comes
  // first.
  (node (GlobalOffset i64:$vaddr, i32:$offset), data_vt:$data),
  (inst $vaddr, getVregSrcForVT<data_vt>.ret:$data, $offset)
>;

multiclass FlatAtomicPat <string inst, string node, ValueType vt,
                          ValueType data_vt = vt> {
  defvar rtnNode = !cast<PatFrags>(node#"_"#vt.Size);
  defvar noRtnNode = !cast<PatFrags>(node#"_noret_"#vt.Size);

  def : GCNPat <(vt (rtnNode (FlatOffset i64:$vaddr, i32:$offset), data_vt:$data)),
    (!cast<FLAT_Pseudo>(inst#"_RTN") VReg_64:$vaddr, getVregSrcForVT<data_vt>.ret:$data, $offset)>;

  let AddedComplexity = 1 in
  def : GCNPat <(vt (noRtnNode (FlatOffset i64:$vaddr, i32:$offset), data_vt:$data)),
    (!cast<FLAT_Pseudo>(inst) VReg_64:$vaddr, getVregSrcForVT<data_vt>.ret:$data, $offset)>;
}

class FlatSignedAtomicPatBase <FLAT_Pseudo inst, SDPatternOperator node,
                               ValueType vt, ValueType data_vt = vt> : GCNPat <
  (vt (node (GlobalOffset i64:$vaddr, i32:$offset), data_vt:$data)),
  (inst VReg_64:$vaddr, getVregSrcForVT<data_vt>.ret:$data, $offset)
>;

multiclass FlatSignedAtomicPat <string inst, string node, ValueType vt,
                                ValueType data_vt = vt, int complexity = 0,
                                bit isIntr = 0> {
  defvar rtnNode = !cast<SDPatternOperator>(node # !if(isIntr, "", "_" # vt.Size));
  defvar noRtnNode = !cast<PatFrags>(node # "_noret" # !if(isIntr, "", "_" # vt.Size));

  let AddedComplexity = complexity in
  def : FlatSignedAtomicPatBase<!cast<FLAT_Pseudo>(inst#"_RTN"), rtnNode, vt, data_vt>;

  let AddedComplexity = !add(complexity, 1) in
  def : FlatSignedAtomicPatBase<!cast<FLAT_Pseudo>(inst), noRtnNode, vt, data_vt>;
}

multiclass FlatSignedAtomicIntrPat <string inst, string node, ValueType vt,
                                    ValueType data_vt = vt> {
  defm : FlatSignedAtomicPat<inst, node, vt, data_vt, /* complexity */ 0, /* isIntr */ 1>;
}

multiclass FlatSignedAtomicPatWithAddrSpace<string inst, string intr, string addrSpaceSuffix,
                                            ValueType vt, ValueType data_vt = vt> {
  defvar noRtnNode = !cast<PatFrags>(intr # "_noret_" # addrSpaceSuffix);
  defvar rtnNode = !cast<PatFrags>(intr # "_" # addrSpaceSuffix);

  let AddedComplexity = 1 in
  def : FlatSignedAtomicPatBase<!cast<FLAT_Pseudo>(inst), noRtnNode, vt, data_vt>;
  def : FlatSignedAtomicPatBase<!cast<FLAT_Pseudo>(inst#"_RTN"), rtnNode, vt, data_vt>;
}

class ScratchLoadSignedPat <FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> : GCNPat <
  (vt (node (ScratchOffset (i32 VGPR_32:$vaddr), i32:$offset))),
  (inst $vaddr, $offset)
>;

class ScratchLoadSignedPat_D16 <FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> : GCNPat <
  (node (ScratchOffset (i32 VGPR_32:$vaddr), i32:$offset), vt:$in),
  (inst $vaddr, $offset, 0, $in)
>;

class ScratchStoreSignedPat <FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> : GCNPat <
  (node vt:$data, (ScratchOffset (i32 VGPR_32:$vaddr), i32:$offset)),
  (inst getVregSrcForVT<vt>.ret:$data, $vaddr, $offset)
>;

class ScratchLoadSaddrPat <FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> : GCNPat <
  (vt (node (ScratchSAddr (i32 SGPR_32:$saddr), i32:$offset))),
  (inst $saddr, $offset)
>;

class ScratchLoadSaddrPat_D16 <FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> : GCNPat <
  (vt (node (ScratchSAddr (i32 SGPR_32:$saddr), i32:$offset), vt:$in)),
  (inst $saddr, $offset, 0, $in)
>;

class ScratchStoreSaddrPat <FLAT_Pseudo inst, SDPatternOperator node,
                            ValueType vt> : GCNPat <
  (node vt:$data, (ScratchSAddr (i32 SGPR_32:$saddr), i32:$offset)),
  (inst getVregSrcForVT<vt>.ret:$data, $saddr, $offset)
>;

class ScratchLoadSVaddrPat <FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> : GCNPat <
  (vt (node (ScratchSVAddr (i32 VGPR_32:$vaddr), (i32 SGPR_32:$saddr), i32:$offset))),
  (inst $vaddr, $saddr, $offset, 0)
>;

class ScratchStoreSVaddrPat <FLAT_Pseudo inst, SDPatternOperator node,
                             ValueType vt> : GCNPat <
  (node vt:$data, (ScratchSVAddr (i32 VGPR_32:$vaddr), (i32 SGPR_32:$saddr), i32:$offset)),
  (inst getVregSrcForVT<vt>.ret:$data, $vaddr, $saddr, $offset)
>;

class ScratchLoadSVaddrPat_D16 <FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> : GCNPat <
  (vt (node (ScratchSVAddr (i32 VGPR_32:$vaddr), (i32 SGPR_32:$saddr), i32:$offset), vt:$in)),
  (inst $vaddr, $saddr, $offset, 0, $in)
>;

let OtherPredicates = [HasFlatAddressSpace] in {

def : FlatLoadPat <FLAT_LOAD_UBYTE, atomic_load_8_flat, i32>;
def : FlatLoadPat <FLAT_LOAD_UBYTE, atomic_load_8_flat, i16>;
def : FlatLoadPat <FLAT_LOAD_USHORT, atomic_load_16_flat, i32>;
def : FlatLoadPat <FLAT_LOAD_USHORT, atomic_load_16_flat, i16>;
def : FlatLoadPat <FLAT_LOAD_UBYTE, extloadi8_flat, i32>;
def : FlatLoadPat <FLAT_LOAD_UBYTE, zextloadi8_flat, i32>;
def : FlatLoadPat <FLAT_LOAD_SBYTE, sextloadi8_flat, i32>;
def : FlatLoadPat <FLAT_LOAD_UBYTE, extloadi8_flat, i16>;
def : FlatLoadPat <FLAT_LOAD_UBYTE, zextloadi8_flat, i16>;
def : FlatLoadPat <FLAT_LOAD_SBYTE, sextloadi8_flat, i16>;
def : FlatLoadPat <FLAT_LOAD_USHORT, extloadi16_flat, i32>;
def : FlatLoadPat <FLAT_LOAD_USHORT, zextloadi16_flat, i32>;
def : FlatLoadPat <FLAT_LOAD_USHORT, load_flat, i16>;
def : FlatLoadPat <FLAT_LOAD_SSHORT, sextloadi16_flat, i32>;
def : FlatLoadPat <FLAT_LOAD_DWORDX3, load_flat, v3i32>;

def : FlatLoadPat <FLAT_LOAD_DWORD, atomic_load_32_flat, i32>;
def : FlatLoadPat <FLAT_LOAD_DWORDX2, atomic_load_64_flat, i64>;

def : FlatStorePat <FLAT_STORE_BYTE, truncstorei8_flat, i32>;
def : FlatStorePat <FLAT_STORE_SHORT, truncstorei16_flat, i32>;

foreach vt = Reg32Types.types in {
def : FlatLoadPat <FLAT_LOAD_DWORD, load_flat, vt>;
def : FlatStorePat <FLAT_STORE_DWORD, store_flat, vt>;
}

foreach vt = VReg_64.RegTypes in {
def : FlatStorePat <FLAT_STORE_DWORDX2, store_flat, vt>;
def : FlatLoadPat <FLAT_LOAD_DWORDX2, load_flat, vt>;
}

def : FlatStorePat <FLAT_STORE_DWORDX3, store_flat, v3i32>;

foreach vt = VReg_128.RegTypes in {
def : FlatLoadPat <FLAT_LOAD_DWORDX4, load_flat, vt>;
def : FlatStorePat <FLAT_STORE_DWORDX4, store_flat, vt>;
}

def : FlatStoreAtomicPat <FLAT_STORE_DWORD, atomic_store_32_flat, i32>;
def : FlatStoreAtomicPat <FLAT_STORE_DWORDX2, atomic_store_64_flat, i64>;
def : FlatStoreAtomicPat <FLAT_STORE_BYTE, atomic_store_8_flat, i32>;
def : FlatStoreAtomicPat <FLAT_STORE_BYTE, atomic_store_8_flat, i16>;
def : FlatStoreAtomicPat <FLAT_STORE_SHORT, atomic_store_16_flat, i32>;
def : FlatStoreAtomicPat <FLAT_STORE_SHORT, atomic_store_16_flat, i16>;

foreach as = [ "flat", "global" ] in {
defm : FlatAtomicPat <"FLAT_ATOMIC_ADD", "atomic_load_add_"#as, i32>;
defm : FlatAtomicPat <"FLAT_ATOMIC_SUB", "atomic_load_sub_"#as, i32>;
defm : FlatAtomicPat <"FLAT_ATOMIC_INC", "atomic_load_uinc_wrap_"#as, i32>;
defm : FlatAtomicPat <"FLAT_ATOMIC_DEC", "atomic_load_udec_wrap_"#as, i32>;
defm : FlatAtomicPat <"FLAT_ATOMIC_AND", "atomic_load_and_"#as, i32>;
defm : FlatAtomicPat <"FLAT_ATOMIC_SMAX", "atomic_load_max_"#as, i32>;
defm : FlatAtomicPat <"FLAT_ATOMIC_UMAX", "atomic_load_umax_"#as, i32>;
defm : FlatAtomicPat <"FLAT_ATOMIC_SMIN", "atomic_load_min_"#as, i32>;
defm : FlatAtomicPat <"FLAT_ATOMIC_UMIN", "atomic_load_umin_"#as, i32>;
defm : FlatAtomicPat <"FLAT_ATOMIC_OR", "atomic_load_or_"#as, i32>;
defm : FlatAtomicPat <"FLAT_ATOMIC_SWAP", "atomic_swap_"#as, i32>;
defm : FlatAtomicPat <"FLAT_ATOMIC_CMPSWAP", "AMDGPUatomic_cmp_swap_"#as, i32, v2i32>;
defm : FlatAtomicPat <"FLAT_ATOMIC_XOR", "atomic_load_xor_"#as, i32>;

defm : FlatAtomicPat <"FLAT_ATOMIC_ADD_X2", "atomic_load_add_"#as, i64>;
defm : FlatAtomicPat <"FLAT_ATOMIC_SUB_X2", "atomic_load_sub_"#as, i64>;
defm : FlatAtomicPat <"FLAT_ATOMIC_INC_X2", "atomic_load_uinc_wrap_"#as, i64>;
defm : FlatAtomicPat <"FLAT_ATOMIC_DEC_X2", "atomic_load_udec_wrap_"#as, i64>;
defm : FlatAtomicPat <"FLAT_ATOMIC_AND_X2", "atomic_load_and_"#as, i64>;
defm : FlatAtomicPat <"FLAT_ATOMIC_SMAX_X2", "atomic_load_max_"#as, i64>;
defm : FlatAtomicPat <"FLAT_ATOMIC_UMAX_X2", "atomic_load_umax_"#as, i64>;
defm : FlatAtomicPat <"FLAT_ATOMIC_SMIN_X2", "atomic_load_min_"#as, i64>;
defm : FlatAtomicPat <"FLAT_ATOMIC_UMIN_X2", "atomic_load_umin_"#as, i64>;
defm : FlatAtomicPat <"FLAT_ATOMIC_OR_X2", "atomic_load_or_"#as, i64>;
defm : FlatAtomicPat <"FLAT_ATOMIC_SWAP_X2", "atomic_swap_"#as, i64>;
defm : FlatAtomicPat <"FLAT_ATOMIC_CMPSWAP_X2", "AMDGPUatomic_cmp_swap_"#as, i64, v2i64>;
defm : FlatAtomicPat <"FLAT_ATOMIC_XOR_X2", "atomic_load_xor_"#as, i64>;
} // end foreach as

def : FlatStorePat <FLAT_STORE_BYTE, truncstorei8_flat, i16>;
def : FlatStorePat <FLAT_STORE_SHORT, store_flat, i16>;

let OtherPredicates = [HasD16LoadStore] in {
def : FlatStorePat <FLAT_STORE_SHORT_D16_HI, truncstorei16_hi16_flat, i32>;
def : FlatStorePat <FLAT_STORE_BYTE_D16_HI, truncstorei8_hi16_flat, i32>;
}

let OtherPredicates = [D16PreservesUnusedBits] in {
def : FlatLoadPat_D16 <FLAT_LOAD_UBYTE_D16_HI, az_extloadi8_d16_hi_flat, v2i16>;
def : FlatLoadPat_D16 <FLAT_LOAD_UBYTE_D16_HI, az_extloadi8_d16_hi_flat, v2f16>;
def : FlatLoadPat_D16 <FLAT_LOAD_SBYTE_D16_HI, sextloadi8_d16_hi_flat, v2i16>;
def : FlatLoadPat_D16 <FLAT_LOAD_SBYTE_D16_HI, sextloadi8_d16_hi_flat, v2f16>;
def : FlatLoadPat_D16 <FLAT_LOAD_SHORT_D16_HI, load_d16_hi_flat, v2i16>;
def : FlatLoadPat_D16 <FLAT_LOAD_SHORT_D16_HI, load_d16_hi_flat, v2f16>;

def : FlatLoadPat_D16 <FLAT_LOAD_UBYTE_D16, az_extloadi8_d16_lo_flat, v2i16>;
def : FlatLoadPat_D16 <FLAT_LOAD_UBYTE_D16, az_extloadi8_d16_lo_flat, v2f16>;
def : FlatLoadPat_D16 <FLAT_LOAD_SBYTE_D16, sextloadi8_d16_lo_flat, v2i16>;
def : FlatLoadPat_D16 <FLAT_LOAD_SBYTE_D16, sextloadi8_d16_lo_flat, v2f16>;
def : FlatLoadPat_D16 <FLAT_LOAD_SHORT_D16, load_d16_lo_flat, v2i16>;
def : FlatLoadPat_D16 <FLAT_LOAD_SHORT_D16, load_d16_lo_flat, v2f16>;
}

} // End OtherPredicates = [HasFlatAddressSpace]


multiclass GlobalFLATLoadPats<FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> {
  def : FlatLoadSignedPat <inst, node, vt> {
    let AddedComplexity = 10;
  }

  def : GlobalLoadSaddrPat<!cast<FLAT_Pseudo>(!cast<string>(inst)#"_SADDR"), node, vt> {
    let AddedComplexity = 11;
  }
}

multiclass GlobalFLATLoadPats_D16<FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> {
  def : FlatSignedLoadPat_D16 <inst, node, vt> {
    let AddedComplexity = 10;
  }

  def : GlobalLoadSaddrPat_D16<!cast<FLAT_Pseudo>(!cast<string>(inst)#"_SADDR"), node, vt> {
    let AddedComplexity = 11;
  }
}

multiclass GlobalFLATStorePats<FLAT_Pseudo inst, SDPatternOperator node,
                               ValueType vt> {
  def : FlatStoreSignedPat <inst, node, vt> {
    let AddedComplexity = 10;
  }

  def : GlobalStoreSaddrPat<!cast<FLAT_Pseudo>(!cast<string>(inst)#"_SADDR"), node, vt> {
    let AddedComplexity = 11;
  }
}

// Deal with swapped operands for atomic_store vs. regular store
multiclass GlobalFLATAtomicStorePats<FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> {
  def : FlatStoreSignedAtomicPat <inst, node, vt> {
    let AddedComplexity = 10;
  }

  def : GlobalAtomicStoreSaddrPat<!cast<FLAT_Pseudo>(!cast<string>(inst)#"_SADDR"), node, vt> {
    let AddedComplexity = 11;
  }
}

multiclass GlobalFLATAtomicPatsNoRtnBase<string inst, string node, ValueType vt,
                                         ValueType data_vt = vt> {
  let AddedComplexity = 11 in
  def : FlatSignedAtomicPatBase<!cast<FLAT_Pseudo>(inst), !cast<PatFrags>(node), vt, data_vt>;

  let AddedComplexity = 13 in
  def : GlobalAtomicSaddrPat<!cast<FLAT_Pseudo>(inst#"_SADDR"), !cast<PatFrags>(node), vt, data_vt>;
}

multiclass GlobalFLATAtomicPatsRtnBase<string inst, string node, ValueType vt,
                                       ValueType data_vt = vt, bit isPatFrags = 0> {
  defvar rtnNode = !if(isPatFrags, !cast<PatFrags>(node), !cast<SDPatternOperator>(node));

  let AddedComplexity = 10 in
  def : FlatSignedAtomicPatBase<!cast<FLAT_Pseudo>(inst#"_RTN"), rtnNode, vt, data_vt>;

  let AddedComplexity = 12 in
  def : GlobalAtomicSaddrPat<!cast<FLAT_Pseudo>(inst#"_SADDR_RTN"), rtnNode, vt, data_vt>;
}

multiclass GlobalFLATAtomicPatsNoRtn<string inst, string node, ValueType vt,
                                     ValueType data_vt = vt, bit isIntr = 0> :
  GlobalFLATAtomicPatsNoRtnBase<inst, node # "_noret" # !if(isIntr, "", "_" # vt.Size), vt, data_vt>;

multiclass GlobalFLATAtomicPatsRtn<string inst, string node, ValueType vt,
                                   ValueType data_vt = vt, bit isIntr = 0> :
  GlobalFLATAtomicPatsRtnBase<inst, node # !if(isIntr, "", "_" # vt.Size), vt, data_vt>;

multiclass GlobalFLATAtomicPats<string inst, string node, ValueType vt,
                                ValueType data_vt = vt, bit isIntr = 0> :
  GlobalFLATAtomicPatsNoRtn<inst, node, vt, data_vt, isIntr>,
  GlobalFLATAtomicPatsRtn<inst, node, vt, data_vt, isIntr>;

multiclass GlobalFLATAtomicPatsNoRtnWithAddrSpace<string inst, string intr, string addrSpaceSuffix,
                                                  ValueType vt, ValueType data_vt = vt> :
  GlobalFLATAtomicPatsNoRtnBase<inst, intr # "_noret_" # addrSpaceSuffix, vt, data_vt>;

multiclass GlobalFLATAtomicPatsRtnWithAddrSpace<string inst, string intr, string addrSpaceSuffix,
                                                ValueType vt, ValueType data_vt = vt> :
  GlobalFLATAtomicPatsRtnBase<inst, intr # "_" # addrSpaceSuffix, vt, data_vt, /*isPatFrags*/ 1>;

multiclass GlobalFLATAtomicPatsWithAddrSpace<string inst, string intr, string addrSpaceSuffix,
                                             ValueType vt, ValueType data_vt = vt> :
  GlobalFLATAtomicPatsNoRtnWithAddrSpace<inst, intr, addrSpaceSuffix, vt, data_vt>,
  GlobalFLATAtomicPatsRtnWithAddrSpace<inst, intr, addrSpaceSuffix, vt, data_vt>;

multiclass GlobalFLATAtomicIntrPats<string inst, string node, ValueType vt,
                                    ValueType data_vt = vt> {
  defm : GlobalFLATAtomicPats<inst, node, vt, data_vt, /* isIntr */ 1>;
}

multiclass ScratchFLATLoadPats<FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> {
  def : ScratchLoadSignedPat <inst, node, vt> {
    let AddedComplexity = 25;
  }

  def : ScratchLoadSaddrPat<!cast<FLAT_Pseudo>(!cast<string>(inst)#"_SADDR"), node, vt> {
    let AddedComplexity = 26;
  }

  def : ScratchLoadSVaddrPat<!cast<FLAT_Pseudo>(!cast<string>(inst)#"_SVS"), node, vt> {
    let SubtargetPredicate = HasFlatScratchSVSMode;
    let AddedComplexity = 27;
  }
}

multiclass ScratchFLATStorePats<FLAT_Pseudo inst, SDPatternOperator node,
                               ValueType vt> {
  def : ScratchStoreSignedPat <inst, node, vt> {
    let AddedComplexity = 25;
  }

  def : ScratchStoreSaddrPat<!cast<FLAT_Pseudo>(!cast<string>(inst)#"_SADDR"), node, vt> {
    let AddedComplexity = 26;
  }

  def : ScratchStoreSVaddrPat<!cast<FLAT_Pseudo>(!cast<string>(inst)#"_SVS"), node, vt> {
    let SubtargetPredicate = HasFlatScratchSVSMode;
    let AddedComplexity = 27;
  }
}

multiclass ScratchFLATLoadPats_D16<FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> {
  def : ScratchLoadSignedPat_D16 <inst, node, vt> {
    let AddedComplexity = 25;
  }

  def : ScratchLoadSaddrPat_D16<!cast<FLAT_Pseudo>(!cast<string>(inst)#"_SADDR"), node, vt> {
    let AddedComplexity = 26;
  }

  def : ScratchLoadSVaddrPat_D16 <!cast<FLAT_Pseudo>(!cast<string>(inst)#"_SVS"), node, vt> {
    let SubtargetPredicate = HasFlatScratchSVSMode;
    let AddedComplexity = 27;
  }
}

let OtherPredicates = [HasFlatGlobalInsts] in {

defm : GlobalFLATLoadPats <GLOBAL_LOAD_UBYTE, atomic_load_8_global, i32>;
defm : GlobalFLATLoadPats <GLOBAL_LOAD_UBYTE, atomic_load_8_global, i16>;
defm : GlobalFLATLoadPats <GLOBAL_LOAD_USHORT, atomic_load_16_global, i32>;
defm : GlobalFLATLoadPats <GLOBAL_LOAD_USHORT, atomic_load_16_global, i16>;
defm : GlobalFLATLoadPats <GLOBAL_LOAD_UBYTE, extloadi8_global, i32>;
defm : GlobalFLATLoadPats <GLOBAL_LOAD_UBYTE, zextloadi8_global, i32>;
defm : GlobalFLATLoadPats <GLOBAL_LOAD_SBYTE, sextloadi8_global, i32>;
defm : GlobalFLATLoadPats <GLOBAL_LOAD_UBYTE, extloadi8_global, i16>;
defm : GlobalFLATLoadPats <GLOBAL_LOAD_UBYTE, zextloadi8_global, i16>;
defm : GlobalFLATLoadPats <GLOBAL_LOAD_SBYTE, sextloadi8_global, i16>;
defm : GlobalFLATLoadPats <GLOBAL_LOAD_USHORT, extloadi16_global, i32>;
defm : GlobalFLATLoadPats <GLOBAL_LOAD_USHORT, zextloadi16_global, i32>;
defm : GlobalFLATLoadPats <GLOBAL_LOAD_SSHORT, sextloadi16_global, i32>;
defm : GlobalFLATLoadPats <GLOBAL_LOAD_USHORT, load_global, i16>;

foreach vt = Reg32Types.types in {
defm : GlobalFLATLoadPats <GLOBAL_LOAD_DWORD, load_global, vt>;
defm : GlobalFLATStorePats <GLOBAL_STORE_DWORD, store_global, vt>;
}

foreach vt = VReg_64.RegTypes in {
defm : GlobalFLATLoadPats <GLOBAL_LOAD_DWORDX2, load_global, vt>;
defm : GlobalFLATStorePats <GLOBAL_STORE_DWORDX2, store_global, vt>;
}

defm : GlobalFLATLoadPats <GLOBAL_LOAD_DWORDX3, load_global, v3i32>;

foreach vt = VReg_128.RegTypes in {
defm : GlobalFLATLoadPats <GLOBAL_LOAD_DWORDX4, load_global, vt>;
defm : GlobalFLATStorePats <GLOBAL_STORE_DWORDX4, store_global, vt>;
}

// There is no distinction for atomic load lowering during selection;
// the memory legalizer will set the cache bits and insert the
// appropriate waits.
defm : GlobalFLATLoadPats <GLOBAL_LOAD_DWORD, atomic_load_32_global, i32>;
defm : GlobalFLATLoadPats <GLOBAL_LOAD_DWORDX2, atomic_load_64_global, i64>;

defm : GlobalFLATStorePats <GLOBAL_STORE_BYTE, truncstorei8_global, i32>;
defm : GlobalFLATStorePats <GLOBAL_STORE_BYTE, truncstorei8_global, i16>;
defm : GlobalFLATStorePats <GLOBAL_STORE_SHORT, truncstorei16_global, i32>;
defm : GlobalFLATStorePats <GLOBAL_STORE_SHORT, store_global, i16>;
defm : GlobalFLATStorePats <GLOBAL_STORE_DWORDX3, store_global, v3i32>;

let OtherPredicates = [HasD16LoadStore] in {
defm : GlobalFLATStorePats <GLOBAL_STORE_SHORT_D16_HI, truncstorei16_hi16_global, i32>;
defm : GlobalFLATStorePats <GLOBAL_STORE_BYTE_D16_HI, truncstorei8_hi16_global, i32>;
}

let OtherPredicates = [D16PreservesUnusedBits] in {
defm : GlobalFLATLoadPats_D16 <GLOBAL_LOAD_UBYTE_D16_HI, az_extloadi8_d16_hi_global, v2i16>;
defm : GlobalFLATLoadPats_D16 <GLOBAL_LOAD_UBYTE_D16_HI, az_extloadi8_d16_hi_global, v2f16>;
defm : GlobalFLATLoadPats_D16 <GLOBAL_LOAD_SBYTE_D16_HI, sextloadi8_d16_hi_global, v2i16>;
defm : GlobalFLATLoadPats_D16 <GLOBAL_LOAD_SBYTE_D16_HI, sextloadi8_d16_hi_global, v2f16>;
defm : GlobalFLATLoadPats_D16 <GLOBAL_LOAD_SHORT_D16_HI, load_d16_hi_global, v2i16>;
defm : GlobalFLATLoadPats_D16 <GLOBAL_LOAD_SHORT_D16_HI, load_d16_hi_global, v2f16>;

defm : GlobalFLATLoadPats_D16 <GLOBAL_LOAD_UBYTE_D16, az_extloadi8_d16_lo_global, v2i16>;
defm : GlobalFLATLoadPats_D16 <GLOBAL_LOAD_UBYTE_D16, az_extloadi8_d16_lo_global, v2f16>;
defm : GlobalFLATLoadPats_D16 <GLOBAL_LOAD_SBYTE_D16, sextloadi8_d16_lo_global, v2i16>;
defm : GlobalFLATLoadPats_D16 <GLOBAL_LOAD_SBYTE_D16, sextloadi8_d16_lo_global, v2f16>;
defm : GlobalFLATLoadPats_D16 <GLOBAL_LOAD_SHORT_D16, load_d16_lo_global, v2i16>;
defm : GlobalFLATLoadPats_D16 <GLOBAL_LOAD_SHORT_D16, load_d16_lo_global, v2f16>;
}

defm : GlobalFLATAtomicStorePats <GLOBAL_STORE_BYTE, atomic_store_8_global, i32>;
defm : GlobalFLATAtomicStorePats <GLOBAL_STORE_BYTE, atomic_store_8_global, i16>;
defm : GlobalFLATAtomicStorePats <GLOBAL_STORE_SHORT, atomic_store_16_global, i32>;
defm : GlobalFLATAtomicStorePats <GLOBAL_STORE_SHORT, atomic_store_16_global, i16>;
defm : GlobalFLATAtomicStorePats <GLOBAL_STORE_DWORD, atomic_store_32_global, i32>;
defm : GlobalFLATAtomicStorePats <GLOBAL_STORE_DWORDX2, atomic_store_64_global, i64>;

defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_ADD", "atomic_load_add_global", i32>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_SUB", "atomic_load_sub_global", i32>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_INC", "atomic_load_uinc_wrap_global", i32>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_DEC", "atomic_load_udec_wrap_global", i32>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_AND", "atomic_load_and_global", i32>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_SMAX", "atomic_load_max_global", i32>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_UMAX", "atomic_load_umax_global", i32>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_SMIN", "atomic_load_min_global", i32>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_UMIN", "atomic_load_umin_global", i32>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_OR", "atomic_load_or_global", i32>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_SWAP", "atomic_swap_global", i32>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_CMPSWAP", "AMDGPUatomic_cmp_swap_global", i32, v2i32>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_XOR", "atomic_load_xor_global", i32>;
defm : GlobalFLATAtomicPatsRtn <"GLOBAL_ATOMIC_CSUB", "int_amdgcn_global_atomic_csub", i32, i32, /* isIntr */ 1>;

defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_ADD_X2", "atomic_load_add_global", i64>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_SUB_X2", "atomic_load_sub_global", i64>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_INC_X2", "atomic_load_uinc_wrap_global", i64>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_DEC_X2", "atomic_load_udec_wrap_global", i64>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_AND_X2", "atomic_load_and_global", i64>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_SMAX_X2", "atomic_load_max_global", i64>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_UMAX_X2", "atomic_load_umax_global", i64>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_SMIN_X2", "atomic_load_min_global", i64>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_UMIN_X2", "atomic_load_umin_global", i64>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_OR_X2", "atomic_load_or_global", i64>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_SWAP_X2", "atomic_swap_global", i64>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_CMPSWAP_X2", "AMDGPUatomic_cmp_swap_global", i64, v2i64>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_XOR_X2", "atomic_load_xor_global", i64>;

let OtherPredicates = [isGFX10Plus] in {
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_FMIN", "atomic_load_fmin_global", f32>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_FMAX", "atomic_load_fmax_global", f32>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_FMIN_X2", "atomic_load_fmin_global", f64>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_FMAX_X2", "atomic_load_fmax_global", f64>;
defm : GlobalFLATAtomicIntrPats <"GLOBAL_ATOMIC_FMIN", "int_amdgcn_global_atomic_fmin", f32>;
defm : GlobalFLATAtomicIntrPats <"GLOBAL_ATOMIC_FMAX", "int_amdgcn_global_atomic_fmax", f32>;
defm : GlobalFLATAtomicIntrPats <"GLOBAL_ATOMIC_FMIN_X2", "int_amdgcn_global_atomic_fmin", f64>;
defm : GlobalFLATAtomicIntrPats <"GLOBAL_ATOMIC_FMAX_X2", "int_amdgcn_global_atomic_fmax", f64>;
}

let OtherPredicates = [HasAtomicFaddNoRtnInsts] in {
defm : GlobalFLATAtomicPatsNoRtn <"GLOBAL_ATOMIC_ADD_F32", "atomic_load_fadd_global", f32>;
defm : GlobalFLATAtomicPatsNoRtnWithAddrSpace <"GLOBAL_ATOMIC_ADD_F32", "int_amdgcn_flat_atomic_fadd", "global_addrspace", f32>;
defm : GlobalFLATAtomicPatsNoRtnWithAddrSpace <"GLOBAL_ATOMIC_ADD_F32", "int_amdgcn_global_atomic_fadd", "global_addrspace", f32>;
}

let OtherPredicates = [HasAtomicPkFaddNoRtnInsts] in {
defm : GlobalFLATAtomicPatsNoRtnWithAddrSpace <"GLOBAL_ATOMIC_PK_ADD_F16", "int_amdgcn_flat_atomic_fadd", "global_addrspace", v2f16>;
defm : GlobalFLATAtomicPatsNoRtnWithAddrSpace <"GLOBAL_ATOMIC_PK_ADD_F16", "int_amdgcn_global_atomic_fadd", "global_addrspace", v2f16>;
}

let OtherPredicates = [HasAtomicFaddRtnInsts] in {
defm : GlobalFLATAtomicPatsRtn <"GLOBAL_ATOMIC_ADD_F32", "atomic_load_fadd_global", f32>;
defm : GlobalFLATAtomicPatsRtnWithAddrSpace <"GLOBAL_ATOMIC_ADD_F32", "int_amdgcn_flat_atomic_fadd", "global_addrspace", f32>;
defm : GlobalFLATAtomicPatsRtnWithAddrSpace <"GLOBAL_ATOMIC_ADD_F32", "int_amdgcn_global_atomic_fadd", "global_addrspace", f32>;
}

let OtherPredicates = [isGFX90APlus] in {
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_ADD_F64", "atomic_load_fadd_global", f64>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_MIN_F64", "atomic_load_fmin_global", f64>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_MAX_F64", "atomic_load_fmax_global", f64>;
defm : GlobalFLATAtomicPatsWithAddrSpace<"GLOBAL_ATOMIC_ADD_F64", "int_amdgcn_flat_atomic_fadd", "global_addrspace", f64>;
defm : GlobalFLATAtomicPatsWithAddrSpace<"GLOBAL_ATOMIC_ADD_F64", "int_amdgcn_global_atomic_fadd", "global_addrspace", f64>;
defm : GlobalFLATAtomicPatsRtnWithAddrSpace <"GLOBAL_ATOMIC_PK_ADD_F16", "int_amdgcn_flat_atomic_fadd", "global_addrspace", v2f16>;
defm : GlobalFLATAtomicPatsRtnWithAddrSpace <"GLOBAL_ATOMIC_PK_ADD_F16", "int_amdgcn_global_atomic_fadd", "global_addrspace", v2f16>;
defm : GlobalFLATAtomicIntrPats <"GLOBAL_ATOMIC_MIN_F64", "int_amdgcn_global_atomic_fmin", f64>;
defm : GlobalFLATAtomicIntrPats <"GLOBAL_ATOMIC_MAX_F64", "int_amdgcn_global_atomic_fmax", f64>;
defm : FlatSignedAtomicPat <"FLAT_ATOMIC_ADD_F64", "atomic_load_fadd_flat", f64>;
defm : FlatSignedAtomicPat <"FLAT_ATOMIC_MIN_F64", "atomic_load_fmin_flat", f64>;
defm : FlatSignedAtomicPat <"FLAT_ATOMIC_MAX_F64", "atomic_load_fmax_flat", f64>;
defm : FlatSignedAtomicPatWithAddrSpace <"FLAT_ATOMIC_ADD_F64", "int_amdgcn_flat_atomic_fadd", "flat_addrspace", f64>;
defm : FlatSignedAtomicIntrPat <"FLAT_ATOMIC_MIN_F64", "int_amdgcn_flat_atomic_fmin", f64>;
defm : FlatSignedAtomicIntrPat <"FLAT_ATOMIC_MAX_F64", "int_amdgcn_flat_atomic_fmax", f64>;
}

let OtherPredicates = [HasFlatAtomicFaddF32Inst] in {
defm : FlatSignedAtomicPat <"FLAT_ATOMIC_ADD_F32", "atomic_load_fadd_flat", f32>;
defm : FlatSignedAtomicPatWithAddrSpace <"FLAT_ATOMIC_ADD_F32", "int_amdgcn_flat_atomic_fadd", "flat_addrspace", f32>;
}

let OtherPredicates = [isGFX940Plus] in {
defm : FlatSignedAtomicPatWithAddrSpace <"FLAT_ATOMIC_PK_ADD_F16", "int_amdgcn_flat_atomic_fadd", "flat_addrspace", v2f16>;
defm : FlatSignedAtomicIntrPat <"FLAT_ATOMIC_PK_ADD_BF16", "int_amdgcn_flat_atomic_fadd_v2bf16", v2i16>;
defm : GlobalFLATAtomicIntrPats <"GLOBAL_ATOMIC_PK_ADD_BF16", "int_amdgcn_global_atomic_fadd_v2bf16", v2i16>;
}

} // End OtherPredicates = [HasFlatGlobalInsts], AddedComplexity = 10

let OtherPredicates = [HasFlatScratchInsts, EnableFlatScratch] in {

defm : ScratchFLATLoadPats <SCRATCH_LOAD_UBYTE, extloadi8_private, i32>;
defm : ScratchFLATLoadPats <SCRATCH_LOAD_UBYTE, zextloadi8_private, i32>;
defm : ScratchFLATLoadPats <SCRATCH_LOAD_SBYTE, sextloadi8_private, i32>;
defm : ScratchFLATLoadPats <SCRATCH_LOAD_UBYTE, extloadi8_private, i16>;
defm : ScratchFLATLoadPats <SCRATCH_LOAD_UBYTE, zextloadi8_private, i16>;
defm : ScratchFLATLoadPats <SCRATCH_LOAD_SBYTE, sextloadi8_private, i16>;
defm : ScratchFLATLoadPats <SCRATCH_LOAD_USHORT, extloadi16_private, i32>;
defm : ScratchFLATLoadPats <SCRATCH_LOAD_USHORT, zextloadi16_private, i32>;
defm : ScratchFLATLoadPats <SCRATCH_LOAD_SSHORT, sextloadi16_private, i32>;
defm : ScratchFLATLoadPats <SCRATCH_LOAD_USHORT, load_private, i16>;

foreach vt = Reg32Types.types in {
defm : ScratchFLATLoadPats <SCRATCH_LOAD_DWORD, load_private, vt>;
defm : ScratchFLATStorePats <SCRATCH_STORE_DWORD, store_private, vt>;
}

foreach vt = VReg_64.RegTypes in {
defm : ScratchFLATLoadPats <SCRATCH_LOAD_DWORDX2, load_private, vt>;
defm : ScratchFLATStorePats <SCRATCH_STORE_DWORDX2, store_private, vt>;
}

defm : ScratchFLATLoadPats <SCRATCH_LOAD_DWORDX3, load_private, v3i32>;

foreach vt = VReg_128.RegTypes in {
defm : ScratchFLATLoadPats <SCRATCH_LOAD_DWORDX4, load_private, vt>;
defm : ScratchFLATStorePats <SCRATCH_STORE_DWORDX4, store_private, vt>;
}

defm : ScratchFLATStorePats <SCRATCH_STORE_BYTE, truncstorei8_private, i32>;
defm : ScratchFLATStorePats <SCRATCH_STORE_BYTE, truncstorei8_private, i16>;
defm : ScratchFLATStorePats <SCRATCH_STORE_SHORT, truncstorei16_private, i32>;
defm : ScratchFLATStorePats <SCRATCH_STORE_SHORT, store_private, i16>;
defm : ScratchFLATStorePats <SCRATCH_STORE_DWORDX3, store_private, v3i32>;

let OtherPredicates = [HasD16LoadStore, HasFlatScratchInsts, EnableFlatScratch] in {
defm : ScratchFLATStorePats <SCRATCH_STORE_SHORT_D16_HI, truncstorei16_hi16_private, i32>;
defm : ScratchFLATStorePats <SCRATCH_STORE_BYTE_D16_HI, truncstorei8_hi16_private, i32>;
}

let OtherPredicates = [D16PreservesUnusedBits, HasFlatScratchInsts, EnableFlatScratch] in {
defm : ScratchFLATLoadPats_D16 <SCRATCH_LOAD_UBYTE_D16_HI, az_extloadi8_d16_hi_private, v2i16>;
defm : ScratchFLATLoadPats_D16 <SCRATCH_LOAD_UBYTE_D16_HI, az_extloadi8_d16_hi_private, v2f16>;
defm : ScratchFLATLoadPats_D16 <SCRATCH_LOAD_SBYTE_D16_HI, sextloadi8_d16_hi_private, v2i16>;
defm : ScratchFLATLoadPats_D16 <SCRATCH_LOAD_SBYTE_D16_HI, sextloadi8_d16_hi_private, v2f16>;
defm : ScratchFLATLoadPats_D16 <SCRATCH_LOAD_SHORT_D16_HI, load_d16_hi_private, v2i16>;
defm : ScratchFLATLoadPats_D16 <SCRATCH_LOAD_SHORT_D16_HI, load_d16_hi_private, v2f16>;

defm : ScratchFLATLoadPats_D16 <SCRATCH_LOAD_UBYTE_D16, az_extloadi8_d16_lo_private, v2i16>;
defm : ScratchFLATLoadPats_D16 <SCRATCH_LOAD_UBYTE_D16, az_extloadi8_d16_lo_private, v2f16>;
defm : ScratchFLATLoadPats_D16 <SCRATCH_LOAD_SBYTE_D16, sextloadi8_d16_lo_private, v2i16>;
defm : ScratchFLATLoadPats_D16 <SCRATCH_LOAD_SBYTE_D16, sextloadi8_d16_lo_private, v2f16>;
defm : ScratchFLATLoadPats_D16 <SCRATCH_LOAD_SHORT_D16, load_d16_lo_private, v2i16>;
defm : ScratchFLATLoadPats_D16 <SCRATCH_LOAD_SHORT_D16, load_d16_lo_private, v2f16>;
}

} // End OtherPredicates = [HasFlatScratchInsts,EnableFlatScratch]

//===----------------------------------------------------------------------===//
// Target
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CI
//===----------------------------------------------------------------------===//

class FLAT_Real_ci <bits<7> op, FLAT_Pseudo ps> :
  FLAT_Real <op, ps>,
  SIMCInstr <ps.PseudoInstr, SIEncodingFamily.SI> {
  let AssemblerPredicate = isGFX7Only;
  let DecoderNamespace="GFX7";
}

def FLAT_LOAD_UBYTE_ci         : FLAT_Real_ci <0x8,  FLAT_LOAD_UBYTE>;
def FLAT_LOAD_SBYTE_ci         : FLAT_Real_ci <0x9,  FLAT_LOAD_SBYTE>;
def FLAT_LOAD_USHORT_ci        : FLAT_Real_ci <0xa,  FLAT_LOAD_USHORT>;
def FLAT_LOAD_SSHORT_ci        : FLAT_Real_ci <0xb,  FLAT_LOAD_SSHORT>;
def FLAT_LOAD_DWORD_ci         : FLAT_Real_ci <0xc,  FLAT_LOAD_DWORD>;
def FLAT_LOAD_DWORDX2_ci       : FLAT_Real_ci <0xd,  FLAT_LOAD_DWORDX2>;
def FLAT_LOAD_DWORDX4_ci       : FLAT_Real_ci <0xe,  FLAT_LOAD_DWORDX4>;
def FLAT_LOAD_DWORDX3_ci       : FLAT_Real_ci <0xf,  FLAT_LOAD_DWORDX3>;

def FLAT_STORE_BYTE_ci         : FLAT_Real_ci <0x18, FLAT_STORE_BYTE>;
def FLAT_STORE_SHORT_ci        : FLAT_Real_ci <0x1a, FLAT_STORE_SHORT>;
def FLAT_STORE_DWORD_ci        : FLAT_Real_ci <0x1c, FLAT_STORE_DWORD>;
def FLAT_STORE_DWORDX2_ci      : FLAT_Real_ci <0x1d, FLAT_STORE_DWORDX2>;
def FLAT_STORE_DWORDX4_ci      : FLAT_Real_ci <0x1e, FLAT_STORE_DWORDX4>;
def FLAT_STORE_DWORDX3_ci      : FLAT_Real_ci <0x1f, FLAT_STORE_DWORDX3>;

multiclass FLAT_Real_Atomics_ci <bits<7> op, FLAT_Pseudo ps> {
  def _ci     : FLAT_Real_ci<op, !cast<FLAT_Pseudo>(ps.PseudoInstr)>;
  def _RTN_ci : FLAT_Real_ci<op, !cast<FLAT_Pseudo>(ps.PseudoInstr # "_RTN")>;
}

defm FLAT_ATOMIC_SWAP          : FLAT_Real_Atomics_ci <0x30, FLAT_ATOMIC_SWAP>;
defm FLAT_ATOMIC_CMPSWAP       : FLAT_Real_Atomics_ci <0x31, FLAT_ATOMIC_CMPSWAP>;
defm FLAT_ATOMIC_ADD           : FLAT_Real_Atomics_ci <0x32, FLAT_ATOMIC_ADD>;
defm FLAT_ATOMIC_SUB           : FLAT_Real_Atomics_ci <0x33, FLAT_ATOMIC_SUB>;
defm FLAT_ATOMIC_SMIN          : FLAT_Real_Atomics_ci <0x35, FLAT_ATOMIC_SMIN>;
defm FLAT_ATOMIC_UMIN          : FLAT_Real_Atomics_ci <0x36, FLAT_ATOMIC_UMIN>;
defm FLAT_ATOMIC_SMAX          : FLAT_Real_Atomics_ci <0x37, FLAT_ATOMIC_SMAX>;
defm FLAT_ATOMIC_UMAX          : FLAT_Real_Atomics_ci <0x38, FLAT_ATOMIC_UMAX>;
defm FLAT_ATOMIC_AND           : FLAT_Real_Atomics_ci <0x39, FLAT_ATOMIC_AND>;
defm FLAT_ATOMIC_OR            : FLAT_Real_Atomics_ci <0x3a, FLAT_ATOMIC_OR>;
defm FLAT_ATOMIC_XOR           : FLAT_Real_Atomics_ci <0x3b, FLAT_ATOMIC_XOR>;
defm FLAT_ATOMIC_INC           : FLAT_Real_Atomics_ci <0x3c, FLAT_ATOMIC_INC>;
defm FLAT_ATOMIC_DEC           : FLAT_Real_Atomics_ci <0x3d, FLAT_ATOMIC_DEC>;
defm FLAT_ATOMIC_SWAP_X2       : FLAT_Real_Atomics_ci <0x50, FLAT_ATOMIC_SWAP_X2>;
defm FLAT_ATOMIC_CMPSWAP_X2    : FLAT_Real_Atomics_ci <0x51, FLAT_ATOMIC_CMPSWAP_X2>;
defm FLAT_ATOMIC_ADD_X2        : FLAT_Real_Atomics_ci <0x52, FLAT_ATOMIC_ADD_X2>;
defm FLAT_ATOMIC_SUB_X2        : FLAT_Real_Atomics_ci <0x53, FLAT_ATOMIC_SUB_X2>;
defm FLAT_ATOMIC_SMIN_X2       : FLAT_Real_Atomics_ci <0x55, FLAT_ATOMIC_SMIN_X2>;
defm FLAT_ATOMIC_UMIN_X2       : FLAT_Real_Atomics_ci <0x56, FLAT_ATOMIC_UMIN_X2>;
defm FLAT_ATOMIC_SMAX_X2       : FLAT_Real_Atomics_ci <0x57, FLAT_ATOMIC_SMAX_X2>;
defm FLAT_ATOMIC_UMAX_X2       : FLAT_Real_Atomics_ci <0x58, FLAT_ATOMIC_UMAX_X2>;
defm FLAT_ATOMIC_AND_X2        : FLAT_Real_Atomics_ci <0x59, FLAT_ATOMIC_AND_X2>;
defm FLAT_ATOMIC_OR_X2         : FLAT_Real_Atomics_ci <0x5a, FLAT_ATOMIC_OR_X2>;
defm FLAT_ATOMIC_XOR_X2        : FLAT_Real_Atomics_ci <0x5b, FLAT_ATOMIC_XOR_X2>;
defm FLAT_ATOMIC_INC_X2        : FLAT_Real_Atomics_ci <0x5c, FLAT_ATOMIC_INC_X2>;
defm FLAT_ATOMIC_DEC_X2        : FLAT_Real_Atomics_ci <0x5d, FLAT_ATOMIC_DEC_X2>;

// CI Only flat instructions
defm FLAT_ATOMIC_FCMPSWAP      : FLAT_Real_Atomics_ci <0x3e, FLAT_ATOMIC_FCMPSWAP>;
defm FLAT_ATOMIC_FMIN          : FLAT_Real_Atomics_ci <0x3f, FLAT_ATOMIC_FMIN>;
defm FLAT_ATOMIC_FMAX          : FLAT_Real_Atomics_ci <0x40, FLAT_ATOMIC_FMAX>;
defm FLAT_ATOMIC_FCMPSWAP_X2   : FLAT_Real_Atomics_ci <0x5e, FLAT_ATOMIC_FCMPSWAP_X2>;
defm FLAT_ATOMIC_FMIN_X2       : FLAT_Real_Atomics_ci <0x5f, FLAT_ATOMIC_FMIN_X2>;
defm FLAT_ATOMIC_FMAX_X2       : FLAT_Real_Atomics_ci <0x60, FLAT_ATOMIC_FMAX_X2>;


//===----------------------------------------------------------------------===//
// VI
//===----------------------------------------------------------------------===//

class FLAT_Real_vi <bits<7> op, FLAT_Pseudo ps, bit has_sccb = ps.has_sccb> :
  FLAT_Real <op, ps>,
  SIMCInstr <ps.PseudoInstr, SIEncodingFamily.VI> {
  let AssemblerPredicate = isGFX8GFX9;
  let DecoderNamespace = "GFX8";

  let Inst{25} = !if(has_sccb, cpol{CPolBit.SCC}, ps.sccbValue);
  let AsmString = ps.Mnemonic #
                  !subst("$sccb", !if(has_sccb, "$sccb",""), ps.AsmOperands);
}

multiclass FLAT_Real_AllAddr_vi<bits<7> op,
  bit has_sccb = !cast<FLAT_Pseudo>(NAME).has_sccb> {
  def _vi : FLAT_Real_vi<op, !cast<FLAT_Pseudo>(NAME), has_sccb>;
  def _SADDR_vi : FLAT_Real_vi<op, !cast<FLAT_Pseudo>(NAME#"_SADDR"), has_sccb>;
}

class FLAT_Real_gfx940 <bits<7> op, FLAT_Pseudo ps> :
  FLAT_Real <op, ps>,
  SIMCInstr <ps.PseudoInstr, SIEncodingFamily.GFX940> {
  let AssemblerPredicate = isGFX940Plus;
  let DecoderNamespace = "GFX9";
  let Inst{13} = ps.sve;
  let Inst{25} = !if(ps.has_sccb, cpol{CPolBit.SCC}, ps.sccbValue);
}

multiclass FLAT_Real_AllAddr_SVE_vi<bits<7> op> {
  def _vi : FLAT_Real_vi<op, !cast<FLAT_Pseudo>(NAME)> {
    let AssemblerPredicate = isGFX8GFX9NotGFX940;
    let OtherPredicates = [isGFX8GFX9NotGFX940];
  }
  def _SADDR_vi : FLAT_Real_vi<op, !cast<FLAT_Pseudo>(NAME#"_SADDR")> {
    let DecoderNamespace = "GFX9";
  }
  let AssemblerPredicate = isGFX940Plus, SubtargetPredicate = isGFX940Plus in {
    def _VE_gfx940  : FLAT_Real_gfx940<op, !cast<FLAT_Pseudo>(NAME)>;
    def _SVS_gfx940 : FLAT_Real_gfx940<op, !cast<FLAT_Pseudo>(NAME#"_SVS")>;
    def _ST_gfx940  : FLAT_Real_gfx940<op, !cast<FLAT_Pseudo>(NAME#"_ST")>;
  }
}

multiclass FLAT_Real_AllAddr_LDS<bits<7> op, bits<7> pre_gfx940_op,
  string pre_gfx940_name = !subst("_lds", "", !cast<FLAT_Pseudo>(NAME).PseudoInstr),
  bit has_sccb = !cast<FLAT_Pseudo>(NAME).has_sccb> {

  let OtherPredicates = [isGFX8GFX9NotGFX940] in {
    def _vi : FLAT_Real_vi<pre_gfx940_op, !cast<FLAT_Pseudo>(NAME), has_sccb> {
      let AsmString = pre_gfx940_name # !cast<FLAT_Pseudo>(NAME).AsmOperands # " lds";
    }
    def _SADDR_vi : FLAT_Real_vi<pre_gfx940_op, !cast<FLAT_Pseudo>(NAME#"_SADDR"), has_sccb> {
      let AsmString = pre_gfx940_name # !cast<FLAT_Pseudo>(NAME#"_SADDR").AsmOperands # " lds";
    }
  }

  let SubtargetPredicate = isGFX940Plus in {
    def _gfx940 : FLAT_Real_gfx940<op, !cast<FLAT_Pseudo>(NAME)>;
    def _SADDR_gfx940 : FLAT_Real_gfx940<op, !cast<FLAT_Pseudo>(NAME#"_SADDR")>;
  }
}

multiclass FLAT_Real_AllAddr_SVE_LDS<bits<7> op, bits<7> pre_gfx940_op> {
  defm "" : FLAT_Real_AllAddr_LDS<op, pre_gfx940_op>;
  let SubtargetPredicate = isGFX940Plus in {
    def _SVS_gfx940 : FLAT_Real_gfx940<op, !cast<FLAT_Pseudo>(NAME#"_SVS")>;
    def _ST_gfx940  : FLAT_Real_gfx940<op, !cast<FLAT_Pseudo>(NAME#"_ST")>;
  }
}

def FLAT_LOAD_UBYTE_vi         : FLAT_Real_vi <0x10, FLAT_LOAD_UBYTE>;
def FLAT_LOAD_SBYTE_vi         : FLAT_Real_vi <0x11, FLAT_LOAD_SBYTE>;
def FLAT_LOAD_USHORT_vi        : FLAT_Real_vi <0x12, FLAT_LOAD_USHORT>;
def FLAT_LOAD_SSHORT_vi        : FLAT_Real_vi <0x13, FLAT_LOAD_SSHORT>;
def FLAT_LOAD_DWORD_vi         : FLAT_Real_vi <0x14, FLAT_LOAD_DWORD>;
def FLAT_LOAD_DWORDX2_vi       : FLAT_Real_vi <0x15, FLAT_LOAD_DWORDX2>;
def FLAT_LOAD_DWORDX4_vi       : FLAT_Real_vi <0x17, FLAT_LOAD_DWORDX4>;
def FLAT_LOAD_DWORDX3_vi       : FLAT_Real_vi <0x16, FLAT_LOAD_DWORDX3>;

def FLAT_STORE_BYTE_vi         : FLAT_Real_vi <0x18, FLAT_STORE_BYTE>;
def FLAT_STORE_BYTE_D16_HI_vi  : FLAT_Real_vi <0x19, FLAT_STORE_BYTE_D16_HI>;
def FLAT_STORE_SHORT_vi        : FLAT_Real_vi <0x1a, FLAT_STORE_SHORT>;
def FLAT_STORE_SHORT_D16_HI_vi : FLAT_Real_vi <0x1b, FLAT_STORE_SHORT_D16_HI>;
def FLAT_STORE_DWORD_vi        : FLAT_Real_vi <0x1c, FLAT_STORE_DWORD>;
def FLAT_STORE_DWORDX2_vi      : FLAT_Real_vi <0x1d, FLAT_STORE_DWORDX2>;
def FLAT_STORE_DWORDX4_vi      : FLAT_Real_vi <0x1f, FLAT_STORE_DWORDX4>;
def FLAT_STORE_DWORDX3_vi      : FLAT_Real_vi <0x1e, FLAT_STORE_DWORDX3>;

def FLAT_LOAD_UBYTE_D16_vi    : FLAT_Real_vi <0x20, FLAT_LOAD_UBYTE_D16>;
def FLAT_LOAD_UBYTE_D16_HI_vi : FLAT_Real_vi <0x21, FLAT_LOAD_UBYTE_D16_HI>;
def FLAT_LOAD_SBYTE_D16_vi    : FLAT_Real_vi <0x22, FLAT_LOAD_SBYTE_D16>;
def FLAT_LOAD_SBYTE_D16_HI_vi : FLAT_Real_vi <0x23, FLAT_LOAD_SBYTE_D16_HI>;
def FLAT_LOAD_SHORT_D16_vi    : FLAT_Real_vi <0x24, FLAT_LOAD_SHORT_D16>;
def FLAT_LOAD_SHORT_D16_HI_vi : FLAT_Real_vi <0x25, FLAT_LOAD_SHORT_D16_HI>;

multiclass FLAT_Real_Atomics_vi <bits<7> op, FLAT_Pseudo ps,
  bit has_sccb = !cast<FLAT_Pseudo>(NAME).has_sccb> {
  def _vi     : FLAT_Real_vi<op, !cast<FLAT_Pseudo>(ps.PseudoInstr), has_sccb>;
  def _RTN_vi : FLAT_Real_vi<op, !cast<FLAT_Pseudo>(ps.PseudoInstr # "_RTN"), has_sccb>;
}

multiclass FLAT_Global_Real_Atomics_vi<bits<7> op,
  bit has_sccb = !cast<FLAT_Pseudo>(NAME).has_sccb> :
  FLAT_Real_AllAddr_vi<op, has_sccb> {
  def _RTN_vi  : FLAT_Real_vi <op, !cast<FLAT_Pseudo>(NAME#"_RTN"), has_sccb>;
  def _SADDR_RTN_vi : FLAT_Real_vi <op, !cast<FLAT_Pseudo>(NAME#"_SADDR_RTN"), has_sccb>;
}


defm FLAT_ATOMIC_SWAP       : FLAT_Real_Atomics_vi <0x40, FLAT_ATOMIC_SWAP>;
defm FLAT_ATOMIC_CMPSWAP    : FLAT_Real_Atomics_vi <0x41, FLAT_ATOMIC_CMPSWAP>;
defm FLAT_ATOMIC_ADD        : FLAT_Real_Atomics_vi <0x42, FLAT_ATOMIC_ADD>;
defm FLAT_ATOMIC_SUB        : FLAT_Real_Atomics_vi <0x43, FLAT_ATOMIC_SUB>;
defm FLAT_ATOMIC_SMIN       : FLAT_Real_Atomics_vi <0x44, FLAT_ATOMIC_SMIN>;
defm FLAT_ATOMIC_UMIN       : FLAT_Real_Atomics_vi <0x45, FLAT_ATOMIC_UMIN>;
defm FLAT_ATOMIC_SMAX       : FLAT_Real_Atomics_vi <0x46, FLAT_ATOMIC_SMAX>;
defm FLAT_ATOMIC_UMAX       : FLAT_Real_Atomics_vi <0x47, FLAT_ATOMIC_UMAX>;
defm FLAT_ATOMIC_AND        : FLAT_Real_Atomics_vi <0x48, FLAT_ATOMIC_AND>;
defm FLAT_ATOMIC_OR         : FLAT_Real_Atomics_vi <0x49, FLAT_ATOMIC_OR>;
defm FLAT_ATOMIC_XOR        : FLAT_Real_Atomics_vi <0x4a, FLAT_ATOMIC_XOR>;
defm FLAT_ATOMIC_INC        : FLAT_Real_Atomics_vi <0x4b, FLAT_ATOMIC_INC>;
defm FLAT_ATOMIC_DEC        : FLAT_Real_Atomics_vi <0x4c, FLAT_ATOMIC_DEC>;
defm FLAT_ATOMIC_SWAP_X2    : FLAT_Real_Atomics_vi <0x60, FLAT_ATOMIC_SWAP_X2>;
defm FLAT_ATOMIC_CMPSWAP_X2 : FLAT_Real_Atomics_vi <0x61, FLAT_ATOMIC_CMPSWAP_X2>;
defm FLAT_ATOMIC_ADD_X2     : FLAT_Real_Atomics_vi <0x62, FLAT_ATOMIC_ADD_X2>;
defm FLAT_ATOMIC_SUB_X2     : FLAT_Real_Atomics_vi <0x63, FLAT_ATOMIC_SUB_X2>;
defm FLAT_ATOMIC_SMIN_X2    : FLAT_Real_Atomics_vi <0x64, FLAT_ATOMIC_SMIN_X2>;
defm FLAT_ATOMIC_UMIN_X2    : FLAT_Real_Atomics_vi <0x65, FLAT_ATOMIC_UMIN_X2>;
defm FLAT_ATOMIC_SMAX_X2    : FLAT_Real_Atomics_vi <0x66, FLAT_ATOMIC_SMAX_X2>;
defm FLAT_ATOMIC_UMAX_X2    : FLAT_Real_Atomics_vi <0x67, FLAT_ATOMIC_UMAX_X2>;
defm FLAT_ATOMIC_AND_X2     : FLAT_Real_Atomics_vi <0x68, FLAT_ATOMIC_AND_X2>;
defm FLAT_ATOMIC_OR_X2      : FLAT_Real_Atomics_vi <0x69, FLAT_ATOMIC_OR_X2>;
defm FLAT_ATOMIC_XOR_X2     : FLAT_Real_Atomics_vi <0x6a, FLAT_ATOMIC_XOR_X2>;
defm FLAT_ATOMIC_INC_X2     : FLAT_Real_Atomics_vi <0x6b, FLAT_ATOMIC_INC_X2>;
defm FLAT_ATOMIC_DEC_X2     : FLAT_Real_Atomics_vi <0x6c, FLAT_ATOMIC_DEC_X2>;

defm GLOBAL_LOAD_UBYTE : FLAT_Real_AllAddr_vi <0x10>;
defm GLOBAL_LOAD_SBYTE : FLAT_Real_AllAddr_vi <0x11>;
defm GLOBAL_LOAD_USHORT : FLAT_Real_AllAddr_vi <0x12>;
defm GLOBAL_LOAD_SSHORT : FLAT_Real_AllAddr_vi <0x13>;
defm GLOBAL_LOAD_DWORD : FLAT_Real_AllAddr_vi <0x14>;
defm GLOBAL_LOAD_DWORDX2 : FLAT_Real_AllAddr_vi <0x15>;
defm GLOBAL_LOAD_DWORDX3 : FLAT_Real_AllAddr_vi <0x16>;
defm GLOBAL_LOAD_DWORDX4 : FLAT_Real_AllAddr_vi <0x17>;

defm GLOBAL_LOAD_UBYTE_D16    : FLAT_Real_AllAddr_vi <0x20>;
defm GLOBAL_LOAD_UBYTE_D16_HI : FLAT_Real_AllAddr_vi <0x21>;
defm GLOBAL_LOAD_SBYTE_D16    : FLAT_Real_AllAddr_vi <0x22>;
defm GLOBAL_LOAD_SBYTE_D16_HI : FLAT_Real_AllAddr_vi <0x23>;
defm GLOBAL_LOAD_SHORT_D16    : FLAT_Real_AllAddr_vi <0x24>;
defm GLOBAL_LOAD_SHORT_D16_HI : FLAT_Real_AllAddr_vi <0x25>;

defm GLOBAL_STORE_BYTE : FLAT_Real_AllAddr_vi <0x18>;
defm GLOBAL_STORE_BYTE_D16_HI : FLAT_Real_AllAddr_vi <0x19>;
defm GLOBAL_STORE_SHORT : FLAT_Real_AllAddr_vi <0x1a>;
defm GLOBAL_STORE_SHORT_D16_HI : FLAT_Real_AllAddr_vi <0x1b>;
defm GLOBAL_STORE_DWORD : FLAT_Real_AllAddr_vi <0x1c>;
defm GLOBAL_STORE_DWORDX2 : FLAT_Real_AllAddr_vi <0x1d>;
defm GLOBAL_STORE_DWORDX3 : FLAT_Real_AllAddr_vi <0x1e>;
defm GLOBAL_STORE_DWORDX4 : FLAT_Real_AllAddr_vi <0x1f>;

defm GLOBAL_LOAD_LDS_UBYTE  : FLAT_Real_AllAddr_LDS <0x026, 0x10>;
defm GLOBAL_LOAD_LDS_SBYTE  : FLAT_Real_AllAddr_LDS <0x027, 0x11>;
defm GLOBAL_LOAD_LDS_USHORT : FLAT_Real_AllAddr_LDS <0x028, 0x12>;
defm GLOBAL_LOAD_LDS_SSHORT : FLAT_Real_AllAddr_LDS <0x029, 0x13>;
defm GLOBAL_LOAD_LDS_DWORD  : FLAT_Real_AllAddr_LDS <0x02a, 0x14>;

defm GLOBAL_ATOMIC_SWAP       : FLAT_Global_Real_Atomics_vi <0x40>;
defm GLOBAL_ATOMIC_CMPSWAP    : FLAT_Global_Real_Atomics_vi <0x41>;
defm GLOBAL_ATOMIC_ADD        : FLAT_Global_Real_Atomics_vi <0x42>;
defm GLOBAL_ATOMIC_SUB        : FLAT_Global_Real_Atomics_vi <0x43>;
defm GLOBAL_ATOMIC_SMIN       : FLAT_Global_Real_Atomics_vi <0x44>;
defm GLOBAL_ATOMIC_UMIN       : FLAT_Global_Real_Atomics_vi <0x45>;
defm GLOBAL_ATOMIC_SMAX       : FLAT_Global_Real_Atomics_vi <0x46>;
defm GLOBAL_ATOMIC_UMAX       : FLAT_Global_Real_Atomics_vi <0x47>;
defm GLOBAL_ATOMIC_AND        : FLAT_Global_Real_Atomics_vi <0x48>;
defm GLOBAL_ATOMIC_OR         : FLAT_Global_Real_Atomics_vi <0x49>;
defm GLOBAL_ATOMIC_XOR        : FLAT_Global_Real_Atomics_vi <0x4a>;
defm GLOBAL_ATOMIC_INC        : FLAT_Global_Real_Atomics_vi <0x4b>;
defm GLOBAL_ATOMIC_DEC        : FLAT_Global_Real_Atomics_vi <0x4c>;
defm GLOBAL_ATOMIC_SWAP_X2    : FLAT_Global_Real_Atomics_vi <0x60>;
defm GLOBAL_ATOMIC_CMPSWAP_X2 : FLAT_Global_Real_Atomics_vi <0x61>;
defm GLOBAL_ATOMIC_ADD_X2     : FLAT_Global_Real_Atomics_vi <0x62>;
defm GLOBAL_ATOMIC_SUB_X2     : FLAT_Global_Real_Atomics_vi <0x63>;
defm GLOBAL_ATOMIC_SMIN_X2    : FLAT_Global_Real_Atomics_vi <0x64>;
defm GLOBAL_ATOMIC_UMIN_X2    : FLAT_Global_Real_Atomics_vi <0x65>;
defm GLOBAL_ATOMIC_SMAX_X2    : FLAT_Global_Real_Atomics_vi <0x66>;
defm GLOBAL_ATOMIC_UMAX_X2    : FLAT_Global_Real_Atomics_vi <0x67>;
defm GLOBAL_ATOMIC_AND_X2     : FLAT_Global_Real_Atomics_vi <0x68>;
defm GLOBAL_ATOMIC_OR_X2      : FLAT_Global_Real_Atomics_vi <0x69>;
defm GLOBAL_ATOMIC_XOR_X2     : FLAT_Global_Real_Atomics_vi <0x6a>;
defm GLOBAL_ATOMIC_INC_X2     : FLAT_Global_Real_Atomics_vi <0x6b>;
defm GLOBAL_ATOMIC_DEC_X2     : FLAT_Global_Real_Atomics_vi <0x6c>;

defm SCRATCH_LOAD_LDS_UBYTE  : FLAT_Real_AllAddr_SVE_LDS <0x026, 0x10>;
defm SCRATCH_LOAD_LDS_SBYTE  : FLAT_Real_AllAddr_SVE_LDS <0x027, 0x11>;
defm SCRATCH_LOAD_LDS_USHORT : FLAT_Real_AllAddr_SVE_LDS <0x028, 0x12>;
defm SCRATCH_LOAD_LDS_SSHORT : FLAT_Real_AllAddr_SVE_LDS <0x029, 0x13>;
defm SCRATCH_LOAD_LDS_DWORD  : FLAT_Real_AllAddr_SVE_LDS <0x02a, 0x14>;

defm SCRATCH_LOAD_UBYTE         : FLAT_Real_AllAddr_SVE_vi <0x10>;
defm SCRATCH_LOAD_SBYTE         : FLAT_Real_AllAddr_SVE_vi <0x11>;
defm SCRATCH_LOAD_USHORT        : FLAT_Real_AllAddr_SVE_vi <0x12>;
defm SCRATCH_LOAD_SSHORT        : FLAT_Real_AllAddr_SVE_vi <0x13>;
defm SCRATCH_LOAD_DWORD         : FLAT_Real_AllAddr_SVE_vi <0x14>;
defm SCRATCH_LOAD_DWORDX2       : FLAT_Real_AllAddr_SVE_vi <0x15>;
defm SCRATCH_LOAD_DWORDX3       : FLAT_Real_AllAddr_SVE_vi <0x16>;
defm SCRATCH_LOAD_DWORDX4       : FLAT_Real_AllAddr_SVE_vi <0x17>;
defm SCRATCH_STORE_BYTE         : FLAT_Real_AllAddr_SVE_vi <0x18>;
defm SCRATCH_STORE_BYTE_D16_HI  : FLAT_Real_AllAddr_SVE_vi <0x19>;
defm SCRATCH_LOAD_UBYTE_D16     : FLAT_Real_AllAddr_SVE_vi <0x20>;
defm SCRATCH_LOAD_UBYTE_D16_HI  : FLAT_Real_AllAddr_SVE_vi <0x21>;
defm SCRATCH_LOAD_SBYTE_D16     : FLAT_Real_AllAddr_SVE_vi <0x22>;
defm SCRATCH_LOAD_SBYTE_D16_HI  : FLAT_Real_AllAddr_SVE_vi <0x23>;
defm SCRATCH_LOAD_SHORT_D16     : FLAT_Real_AllAddr_SVE_vi <0x24>;
defm SCRATCH_LOAD_SHORT_D16_HI  : FLAT_Real_AllAddr_SVE_vi <0x25>;
defm SCRATCH_STORE_SHORT        : FLAT_Real_AllAddr_SVE_vi <0x1a>;
defm SCRATCH_STORE_SHORT_D16_HI : FLAT_Real_AllAddr_SVE_vi <0x1b>;
defm SCRATCH_STORE_DWORD        : FLAT_Real_AllAddr_SVE_vi <0x1c>;
defm SCRATCH_STORE_DWORDX2      : FLAT_Real_AllAddr_SVE_vi <0x1d>;
defm SCRATCH_STORE_DWORDX3      : FLAT_Real_AllAddr_SVE_vi <0x1e>;
defm SCRATCH_STORE_DWORDX4      : FLAT_Real_AllAddr_SVE_vi <0x1f>;

let SubtargetPredicate = isGFX8GFX9NotGFX940 in {
  // These instructions are encoded differently on gfx90* and gfx940.
  defm GLOBAL_ATOMIC_ADD_F32    : FLAT_Global_Real_Atomics_vi <0x04d, 0>;
  defm GLOBAL_ATOMIC_PK_ADD_F16 : FLAT_Global_Real_Atomics_vi <0x04e, 0>;
}

let SubtargetPredicate = isGFX90AOnly in {
  defm FLAT_ATOMIC_ADD_F64   : FLAT_Real_Atomics_vi<0x4f, FLAT_ATOMIC_ADD_F64, 0>;
  defm FLAT_ATOMIC_MIN_F64   : FLAT_Real_Atomics_vi<0x50, FLAT_ATOMIC_MIN_F64, 0>;
  defm FLAT_ATOMIC_MAX_F64   : FLAT_Real_Atomics_vi<0x51, FLAT_ATOMIC_MAX_F64, 0>;
  defm GLOBAL_ATOMIC_ADD_F64 : FLAT_Global_Real_Atomics_vi<0x4f, 0>;
  defm GLOBAL_ATOMIC_MIN_F64 : FLAT_Global_Real_Atomics_vi<0x50, 0>;
  defm GLOBAL_ATOMIC_MAX_F64 : FLAT_Global_Real_Atomics_vi<0x51, 0>;
} // End SubtargetPredicate = isGFX90AOnly

multiclass FLAT_Real_AllAddr_gfx940<bits<7> op> {
  def _gfx940       : FLAT_Real_gfx940<op, !cast<FLAT_Pseudo>(NAME)>;
  def _SADDR_gfx940 : FLAT_Real_gfx940<op, !cast<FLAT_Pseudo>(NAME#"_SADDR")>;
}

multiclass FLAT_Real_Atomics_gfx940 <bits<7> op, FLAT_Pseudo ps> {
  def _gfx940     : FLAT_Real_gfx940<op, !cast<FLAT_Pseudo>(ps.PseudoInstr)>;
  def _RTN_gfx940 : FLAT_Real_gfx940<op, !cast<FLAT_Pseudo>(ps.PseudoInstr # "_RTN")>;
}

multiclass FLAT_Global_Real_Atomics_gfx940<bits<7> op> :
  FLAT_Real_AllAddr_gfx940<op> {
  def _RTN_gfx940       : FLAT_Real_gfx940 <op, !cast<FLAT_Pseudo>(NAME#"_RTN")>;
  def _SADDR_RTN_gfx940 : FLAT_Real_gfx940 <op, !cast<FLAT_Pseudo>(NAME#"_SADDR_RTN")>;
}

let SubtargetPredicate = isGFX940Plus in {
  // These instructions are encoded differently on gfx90* and gfx940.
  defm GLOBAL_ATOMIC_ADD_F32     : FLAT_Global_Real_Atomics_gfx940 <0x04d>;
  defm GLOBAL_ATOMIC_PK_ADD_F16  : FLAT_Global_Real_Atomics_gfx940 <0x04e>;

  defm FLAT_ATOMIC_ADD_F64       : FLAT_Real_Atomics_gfx940<0x4f, FLAT_ATOMIC_ADD_F64>;
  defm FLAT_ATOMIC_MIN_F64       : FLAT_Real_Atomics_gfx940<0x50, FLAT_ATOMIC_MIN_F64>;
  defm FLAT_ATOMIC_MAX_F64       : FLAT_Real_Atomics_gfx940<0x51, FLAT_ATOMIC_MAX_F64>;
  defm GLOBAL_ATOMIC_ADD_F64     : FLAT_Global_Real_Atomics_gfx940<0x4f>;
  defm GLOBAL_ATOMIC_MIN_F64     : FLAT_Global_Real_Atomics_gfx940<0x50>;
  defm GLOBAL_ATOMIC_MAX_F64     : FLAT_Global_Real_Atomics_gfx940<0x51>;
  defm FLAT_ATOMIC_ADD_F32       : FLAT_Real_Atomics_vi<0x4d, FLAT_ATOMIC_ADD_F32>;
  defm FLAT_ATOMIC_PK_ADD_F16    : FLAT_Real_Atomics_vi<0x4e, FLAT_ATOMIC_PK_ADD_F16>;
  defm FLAT_ATOMIC_PK_ADD_BF16   : FLAT_Real_Atomics_vi<0x52, FLAT_ATOMIC_PK_ADD_BF16>;
  defm GLOBAL_ATOMIC_PK_ADD_BF16 : FLAT_Global_Real_Atomics_vi<0x52>;
} // End SubtargetPredicate = isGFX940Plus

//===----------------------------------------------------------------------===//
// GFX10.
//===----------------------------------------------------------------------===//

class FLAT_Real_gfx10<bits<7> op, FLAT_Pseudo ps> :
    FLAT_Real<op, ps>, SIMCInstr<ps.PseudoInstr, SIEncodingFamily.GFX10> {
  let AssemblerPredicate = isGFX10Only;
  let DecoderNamespace = "GFX10";

  let Inst{11-0}  = offset{11-0};
  let Inst{12}    = !if(ps.has_dlc, cpol{CPolBit.DLC}, ps.dlcValue);
  let Inst{54-48} = !if(ps.has_saddr, !if(ps.enabled_saddr, saddr, 0x7d), 0x7d);
  let Inst{55}    = 0;
}


multiclass FLAT_Real_Base_gfx10<bits<7> op> {
  def _gfx10 :
    FLAT_Real_gfx10<op, !cast<FLAT_Pseudo>(NAME)>;
}

multiclass FLAT_Real_RTN_gfx10<bits<7> op> {
  def _RTN_gfx10 :
    FLAT_Real_gfx10<op, !cast<FLAT_Pseudo>(NAME#"_RTN")>;
}

multiclass FLAT_Real_SADDR_gfx10<bits<7> op> {
  def _SADDR_gfx10 :
    FLAT_Real_gfx10<op, !cast<FLAT_Pseudo>(NAME#"_SADDR")>;
}

multiclass FLAT_Real_SADDR_RTN_gfx10<bits<7> op> {
  def _SADDR_RTN_gfx10 :
    FLAT_Real_gfx10<op, !cast<FLAT_Pseudo>(NAME#"_SADDR_RTN")>;
}

multiclass FLAT_Real_ST_gfx10<bits<7> op> {
  def _ST_gfx10 :
    FLAT_Real_gfx10<op, !cast<FLAT_Pseudo>(NAME#"_ST")> {
      let Inst{54-48} = !cast<int>(EXEC_HI.HWEncoding);
      let OtherPredicates = [HasFlatScratchSTMode];
    }
}

multiclass FLAT_Real_AllAddr_gfx10<bits<7> op> :
  FLAT_Real_Base_gfx10<op>,
  FLAT_Real_SADDR_gfx10<op>;

multiclass FLAT_Real_Atomics_gfx10<bits<7> op> :
  FLAT_Real_Base_gfx10<op>,
  FLAT_Real_RTN_gfx10<op>;

multiclass FLAT_Real_GlblAtomics_gfx10<bits<7> op> :
  FLAT_Real_AllAddr_gfx10<op>,
  FLAT_Real_RTN_gfx10<op>,
  FLAT_Real_SADDR_RTN_gfx10<op>;

multiclass FLAT_Real_GlblAtomics_RTN_gfx10<bits<7> op> :
  FLAT_Real_RTN_gfx10<op>,
  FLAT_Real_SADDR_RTN_gfx10<op>;

multiclass FLAT_Real_ScratchAllAddr_gfx10<bits<7> op> :
  FLAT_Real_Base_gfx10<op>,
  FLAT_Real_SADDR_gfx10<op>,
  FLAT_Real_ST_gfx10<op>;

multiclass FLAT_Real_AllAddr_LDS_gfx10<bits<7> op,
  string opname = !subst("_lds", "", !cast<FLAT_Pseudo>(NAME).PseudoInstr)> {
  let AsmString = opname # !cast<FLAT_Pseudo>(NAME).AsmOperands # " lds" in
  defm "" : FLAT_Real_Base_gfx10<op>;

  let AsmString = opname # !cast<FLAT_Pseudo>(NAME#"_SADDR").AsmOperands # " lds" in
  defm "" : FLAT_Real_SADDR_gfx10<op>;
}

multiclass FLAT_Real_ScratchAllAddr_LDS_gfx10<bits<7> op,
  string opname = !subst("_lds", "", !cast<FLAT_Pseudo>(NAME).PseudoInstr)> {
  defm "" : FLAT_Real_AllAddr_LDS_gfx10<op>;

  let AsmString = opname # !cast<FLAT_Pseudo>(NAME#"_ST").AsmOperands # " lds" in
  defm "" : FLAT_Real_ST_gfx10<op>;
}

// ENC_FLAT.
defm FLAT_LOAD_UBYTE            : FLAT_Real_Base_gfx10<0x008>;
defm FLAT_LOAD_SBYTE            : FLAT_Real_Base_gfx10<0x009>;
defm FLAT_LOAD_USHORT           : FLAT_Real_Base_gfx10<0x00a>;
defm FLAT_LOAD_SSHORT           : FLAT_Real_Base_gfx10<0x00b>;
defm FLAT_LOAD_DWORD            : FLAT_Real_Base_gfx10<0x00c>;
defm FLAT_LOAD_DWORDX2          : FLAT_Real_Base_gfx10<0x00d>;
defm FLAT_LOAD_DWORDX4          : FLAT_Real_Base_gfx10<0x00e>;
defm FLAT_LOAD_DWORDX3          : FLAT_Real_Base_gfx10<0x00f>;
defm FLAT_STORE_BYTE            : FLAT_Real_Base_gfx10<0x018>;
defm FLAT_STORE_BYTE_D16_HI     : FLAT_Real_Base_gfx10<0x019>;
defm FLAT_STORE_SHORT           : FLAT_Real_Base_gfx10<0x01a>;
defm FLAT_STORE_SHORT_D16_HI    : FLAT_Real_Base_gfx10<0x01b>;
defm FLAT_STORE_DWORD           : FLAT_Real_Base_gfx10<0x01c>;
defm FLAT_STORE_DWORDX2         : FLAT_Real_Base_gfx10<0x01d>;
defm FLAT_STORE_DWORDX4         : FLAT_Real_Base_gfx10<0x01e>;
defm FLAT_STORE_DWORDX3         : FLAT_Real_Base_gfx10<0x01f>;
defm FLAT_LOAD_UBYTE_D16        : FLAT_Real_Base_gfx10<0x020>;
defm FLAT_LOAD_UBYTE_D16_HI     : FLAT_Real_Base_gfx10<0x021>;
defm FLAT_LOAD_SBYTE_D16        : FLAT_Real_Base_gfx10<0x022>;
defm FLAT_LOAD_SBYTE_D16_HI     : FLAT_Real_Base_gfx10<0x023>;
defm FLAT_LOAD_SHORT_D16        : FLAT_Real_Base_gfx10<0x024>;
defm FLAT_LOAD_SHORT_D16_HI     : FLAT_Real_Base_gfx10<0x025>;
defm FLAT_ATOMIC_SWAP           : FLAT_Real_Atomics_gfx10<0x030>;
defm FLAT_ATOMIC_CMPSWAP        : FLAT_Real_Atomics_gfx10<0x031>;
defm FLAT_ATOMIC_ADD            : FLAT_Real_Atomics_gfx10<0x032>;
defm FLAT_ATOMIC_SUB            : FLAT_Real_Atomics_gfx10<0x033>;
defm FLAT_ATOMIC_SMIN           : FLAT_Real_Atomics_gfx10<0x035>;
defm FLAT_ATOMIC_UMIN           : FLAT_Real_Atomics_gfx10<0x036>;
defm FLAT_ATOMIC_SMAX           : FLAT_Real_Atomics_gfx10<0x037>;
defm FLAT_ATOMIC_UMAX           : FLAT_Real_Atomics_gfx10<0x038>;
defm FLAT_ATOMIC_AND            : FLAT_Real_Atomics_gfx10<0x039>;
defm FLAT_ATOMIC_OR             : FLAT_Real_Atomics_gfx10<0x03a>;
defm FLAT_ATOMIC_XOR            : FLAT_Real_Atomics_gfx10<0x03b>;
defm FLAT_ATOMIC_INC            : FLAT_Real_Atomics_gfx10<0x03c>;
defm FLAT_ATOMIC_DEC            : FLAT_Real_Atomics_gfx10<0x03d>;
defm FLAT_ATOMIC_FCMPSWAP       : FLAT_Real_Atomics_gfx10<0x03e>;
defm FLAT_ATOMIC_FMIN           : FLAT_Real_Atomics_gfx10<0x03f>;
defm FLAT_ATOMIC_FMAX           : FLAT_Real_Atomics_gfx10<0x040>;
defm FLAT_ATOMIC_SWAP_X2        : FLAT_Real_Atomics_gfx10<0x050>;
defm FLAT_ATOMIC_CMPSWAP_X2     : FLAT_Real_Atomics_gfx10<0x051>;
defm FLAT_ATOMIC_ADD_X2         : FLAT_Real_Atomics_gfx10<0x052>;
defm FLAT_ATOMIC_SUB_X2         : FLAT_Real_Atomics_gfx10<0x053>;
defm FLAT_ATOMIC_SMIN_X2        : FLAT_Real_Atomics_gfx10<0x055>;
defm FLAT_ATOMIC_UMIN_X2        : FLAT_Real_Atomics_gfx10<0x056>;
defm FLAT_ATOMIC_SMAX_X2        : FLAT_Real_Atomics_gfx10<0x057>;
defm FLAT_ATOMIC_UMAX_X2        : FLAT_Real_Atomics_gfx10<0x058>;
defm FLAT_ATOMIC_AND_X2         : FLAT_Real_Atomics_gfx10<0x059>;
defm FLAT_ATOMIC_OR_X2          : FLAT_Real_Atomics_gfx10<0x05a>;
defm FLAT_ATOMIC_XOR_X2         : FLAT_Real_Atomics_gfx10<0x05b>;
defm FLAT_ATOMIC_INC_X2         : FLAT_Real_Atomics_gfx10<0x05c>;
defm FLAT_ATOMIC_DEC_X2         : FLAT_Real_Atomics_gfx10<0x05d>;
defm FLAT_ATOMIC_FCMPSWAP_X2    : FLAT_Real_Atomics_gfx10<0x05e>;
defm FLAT_ATOMIC_FMIN_X2        : FLAT_Real_Atomics_gfx10<0x05f>;
defm FLAT_ATOMIC_FMAX_X2        : FLAT_Real_Atomics_gfx10<0x060>;


// ENC_FLAT_GLBL.
defm GLOBAL_LOAD_UBYTE          : FLAT_Real_AllAddr_gfx10<0x008>;
defm GLOBAL_LOAD_SBYTE          : FLAT_Real_AllAddr_gfx10<0x009>;
defm GLOBAL_LOAD_USHORT         : FLAT_Real_AllAddr_gfx10<0x00a>;
defm GLOBAL_LOAD_SSHORT         : FLAT_Real_AllAddr_gfx10<0x00b>;
defm GLOBAL_LOAD_DWORD          : FLAT_Real_AllAddr_gfx10<0x00c>;
defm GLOBAL_LOAD_DWORDX2        : FLAT_Real_AllAddr_gfx10<0x00d>;
defm GLOBAL_LOAD_DWORDX4        : FLAT_Real_AllAddr_gfx10<0x00e>;
defm GLOBAL_LOAD_DWORDX3        : FLAT_Real_AllAddr_gfx10<0x00f>;
defm GLOBAL_STORE_BYTE          : FLAT_Real_AllAddr_gfx10<0x018>;
defm GLOBAL_STORE_BYTE_D16_HI   : FLAT_Real_AllAddr_gfx10<0x019>;
defm GLOBAL_STORE_SHORT         : FLAT_Real_AllAddr_gfx10<0x01a>;
defm GLOBAL_STORE_SHORT_D16_HI  : FLAT_Real_AllAddr_gfx10<0x01b>;
defm GLOBAL_STORE_DWORD         : FLAT_Real_AllAddr_gfx10<0x01c>;
defm GLOBAL_STORE_DWORDX2       : FLAT_Real_AllAddr_gfx10<0x01d>;
defm GLOBAL_STORE_DWORDX4       : FLAT_Real_AllAddr_gfx10<0x01e>;
defm GLOBAL_STORE_DWORDX3       : FLAT_Real_AllAddr_gfx10<0x01f>;
defm GLOBAL_LOAD_UBYTE_D16      : FLAT_Real_AllAddr_gfx10<0x020>;
defm GLOBAL_LOAD_UBYTE_D16_HI   : FLAT_Real_AllAddr_gfx10<0x021>;
defm GLOBAL_LOAD_SBYTE_D16      : FLAT_Real_AllAddr_gfx10<0x022>;
defm GLOBAL_LOAD_SBYTE_D16_HI   : FLAT_Real_AllAddr_gfx10<0x023>;
defm GLOBAL_LOAD_SHORT_D16      : FLAT_Real_AllAddr_gfx10<0x024>;
defm GLOBAL_LOAD_SHORT_D16_HI   : FLAT_Real_AllAddr_gfx10<0x025>;
defm GLOBAL_ATOMIC_SWAP         : FLAT_Real_GlblAtomics_gfx10<0x030>;
defm GLOBAL_ATOMIC_CMPSWAP      : FLAT_Real_GlblAtomics_gfx10<0x031>;
defm GLOBAL_ATOMIC_ADD          : FLAT_Real_GlblAtomics_gfx10<0x032>;
defm GLOBAL_ATOMIC_SUB          : FLAT_Real_GlblAtomics_gfx10<0x033>;
defm GLOBAL_ATOMIC_CSUB         : FLAT_Real_GlblAtomics_RTN_gfx10<0x034>;
defm GLOBAL_ATOMIC_SMIN         : FLAT_Real_GlblAtomics_gfx10<0x035>;
defm GLOBAL_ATOMIC_UMIN         : FLAT_Real_GlblAtomics_gfx10<0x036>;
defm GLOBAL_ATOMIC_SMAX         : FLAT_Real_GlblAtomics_gfx10<0x037>;
defm GLOBAL_ATOMIC_UMAX         : FLAT_Real_GlblAtomics_gfx10<0x038>;
defm GLOBAL_ATOMIC_AND          : FLAT_Real_GlblAtomics_gfx10<0x039>;
defm GLOBAL_ATOMIC_OR           : FLAT_Real_GlblAtomics_gfx10<0x03a>;
defm GLOBAL_ATOMIC_XOR          : FLAT_Real_GlblAtomics_gfx10<0x03b>;
defm GLOBAL_ATOMIC_INC          : FLAT_Real_GlblAtomics_gfx10<0x03c>;
defm GLOBAL_ATOMIC_DEC          : FLAT_Real_GlblAtomics_gfx10<0x03d>;
defm GLOBAL_ATOMIC_FCMPSWAP     : FLAT_Real_GlblAtomics_gfx10<0x03e>;
defm GLOBAL_ATOMIC_FMIN         : FLAT_Real_GlblAtomics_gfx10<0x03f>;
defm GLOBAL_ATOMIC_FMAX         : FLAT_Real_GlblAtomics_gfx10<0x040>;
defm GLOBAL_ATOMIC_SWAP_X2      : FLAT_Real_GlblAtomics_gfx10<0x050>;
defm GLOBAL_ATOMIC_CMPSWAP_X2   : FLAT_Real_GlblAtomics_gfx10<0x051>;
defm GLOBAL_ATOMIC_ADD_X2       : FLAT_Real_GlblAtomics_gfx10<0x052>;
defm GLOBAL_ATOMIC_SUB_X2       : FLAT_Real_GlblAtomics_gfx10<0x053>;
defm GLOBAL_ATOMIC_SMIN_X2      : FLAT_Real_GlblAtomics_gfx10<0x055>;
defm GLOBAL_ATOMIC_UMIN_X2      : FLAT_Real_GlblAtomics_gfx10<0x056>;
defm GLOBAL_ATOMIC_SMAX_X2      : FLAT_Real_GlblAtomics_gfx10<0x057>;
defm GLOBAL_ATOMIC_UMAX_X2      : FLAT_Real_GlblAtomics_gfx10<0x058>;
defm GLOBAL_ATOMIC_AND_X2       : FLAT_Real_GlblAtomics_gfx10<0x059>;
defm GLOBAL_ATOMIC_OR_X2        : FLAT_Real_GlblAtomics_gfx10<0x05a>;
defm GLOBAL_ATOMIC_XOR_X2       : FLAT_Real_GlblAtomics_gfx10<0x05b>;
defm GLOBAL_ATOMIC_INC_X2       : FLAT_Real_GlblAtomics_gfx10<0x05c>;
defm GLOBAL_ATOMIC_DEC_X2       : FLAT_Real_GlblAtomics_gfx10<0x05d>;
defm GLOBAL_ATOMIC_FCMPSWAP_X2  : FLAT_Real_GlblAtomics_gfx10<0x05e>;
defm GLOBAL_ATOMIC_FMIN_X2      : FLAT_Real_GlblAtomics_gfx10<0x05f>;
defm GLOBAL_ATOMIC_FMAX_X2      : FLAT_Real_GlblAtomics_gfx10<0x060>;
defm GLOBAL_LOAD_DWORD_ADDTID   : FLAT_Real_AllAddr_gfx10<0x016>;
defm GLOBAL_STORE_DWORD_ADDTID  : FLAT_Real_AllAddr_gfx10<0x017>;

defm GLOBAL_LOAD_LDS_UBYTE      : FLAT_Real_AllAddr_LDS_gfx10 <0x008>;
defm GLOBAL_LOAD_LDS_SBYTE      : FLAT_Real_AllAddr_LDS_gfx10 <0x009>;
defm GLOBAL_LOAD_LDS_USHORT     : FLAT_Real_AllAddr_LDS_gfx10 <0x00a>;
defm GLOBAL_LOAD_LDS_SSHORT     : FLAT_Real_AllAddr_LDS_gfx10 <0x00b>;
defm GLOBAL_LOAD_LDS_DWORD      : FLAT_Real_AllAddr_LDS_gfx10 <0x00c>;

// ENC_FLAT_SCRATCH.
defm SCRATCH_LOAD_UBYTE         : FLAT_Real_ScratchAllAddr_gfx10<0x008>;
defm SCRATCH_LOAD_SBYTE         : FLAT_Real_ScratchAllAddr_gfx10<0x009>;
defm SCRATCH_LOAD_USHORT        : FLAT_Real_ScratchAllAddr_gfx10<0x00a>;
defm SCRATCH_LOAD_SSHORT        : FLAT_Real_ScratchAllAddr_gfx10<0x00b>;
defm SCRATCH_LOAD_DWORD         : FLAT_Real_ScratchAllAddr_gfx10<0x00c>;
defm SCRATCH_LOAD_DWORDX2       : FLAT_Real_ScratchAllAddr_gfx10<0x00d>;
defm SCRATCH_LOAD_DWORDX4       : FLAT_Real_ScratchAllAddr_gfx10<0x00e>;
defm SCRATCH_LOAD_DWORDX3       : FLAT_Real_ScratchAllAddr_gfx10<0x00f>;
defm SCRATCH_STORE_BYTE         : FLAT_Real_ScratchAllAddr_gfx10<0x018>;
defm SCRATCH_STORE_BYTE_D16_HI  : FLAT_Real_ScratchAllAddr_gfx10<0x019>;
defm SCRATCH_STORE_SHORT        : FLAT_Real_ScratchAllAddr_gfx10<0x01a>;
defm SCRATCH_STORE_SHORT_D16_HI : FLAT_Real_ScratchAllAddr_gfx10<0x01b>;
defm SCRATCH_STORE_DWORD        : FLAT_Real_ScratchAllAddr_gfx10<0x01c>;
defm SCRATCH_STORE_DWORDX2      : FLAT_Real_ScratchAllAddr_gfx10<0x01d>;
defm SCRATCH_STORE_DWORDX4      : FLAT_Real_ScratchAllAddr_gfx10<0x01e>;
defm SCRATCH_STORE_DWORDX3      : FLAT_Real_ScratchAllAddr_gfx10<0x01f>;
defm SCRATCH_LOAD_UBYTE_D16     : FLAT_Real_ScratchAllAddr_gfx10<0x020>;
defm SCRATCH_LOAD_UBYTE_D16_HI  : FLAT_Real_ScratchAllAddr_gfx10<0x021>;
defm SCRATCH_LOAD_SBYTE_D16     : FLAT_Real_ScratchAllAddr_gfx10<0x022>;
defm SCRATCH_LOAD_SBYTE_D16_HI  : FLAT_Real_ScratchAllAddr_gfx10<0x023>;
defm SCRATCH_LOAD_SHORT_D16     : FLAT_Real_ScratchAllAddr_gfx10<0x024>;
defm SCRATCH_LOAD_SHORT_D16_HI  : FLAT_Real_ScratchAllAddr_gfx10<0x025>;

defm SCRATCH_LOAD_LDS_UBYTE     : FLAT_Real_ScratchAllAddr_LDS_gfx10 <0x008>;
defm SCRATCH_LOAD_LDS_SBYTE     : FLAT_Real_ScratchAllAddr_LDS_gfx10 <0x009>;
defm SCRATCH_LOAD_LDS_USHORT    : FLAT_Real_ScratchAllAddr_LDS_gfx10 <0x00a>;
defm SCRATCH_LOAD_LDS_SSHORT    : FLAT_Real_ScratchAllAddr_LDS_gfx10 <0x00b>;
defm SCRATCH_LOAD_LDS_DWORD     : FLAT_Real_ScratchAllAddr_LDS_gfx10 <0x00c>;

//===----------------------------------------------------------------------===//
// GFX11
//===----------------------------------------------------------------------===//

class FLAT_Real_gfx11 <bits<7> op, FLAT_Pseudo ps, string opName = ps.Mnemonic> :
  FLAT_Real <op, ps, opName>,
  SIMCInstr <ps.PseudoInstr, SIEncodingFamily.GFX11> {
  let AssemblerPredicate = isGFX11Plus;
  let DecoderNamespace = "GFX11";

  let Inst{13}    = !if(ps.has_dlc, cpol{CPolBit.DLC}, ps.dlcValue);
  let Inst{14}    = !if(ps.has_glc, cpol{CPolBit.GLC}, ps.glcValue);
  let Inst{15}    = cpol{CPolBit.SLC};
  let Inst{17-16} = seg;
  let Inst{55}    = ps.sve;
}

multiclass FLAT_Real_Base_gfx11<bits<7> op, string ps, string opName, int renamed = false> {
  def _gfx11 : FLAT_Real_gfx11<op, !cast<FLAT_Pseudo>(ps), opName> {
    let Inst{54-48} = !cast<int>(SGPR_NULL_gfx11plus.HWEncoding);
  }
  if renamed then
    def _renamed_gfx11 : MnemonicAlias<!cast<FLAT_Pseudo>(ps).Mnemonic, opName>, Requires<[isGFX11Plus]>;
}

multiclass FLAT_Real_RTN_gfx11<bits<7> op, string ps, string opName> {
  def _RTN_gfx11 : FLAT_Real_gfx11<op, !cast<FLAT_Pseudo>(ps#"_RTN"), opName> {
    let Inst{54-48} = !cast<int>(SGPR_NULL_gfx11plus.HWEncoding);
  }
}

multiclass FLAT_Real_SADDR_gfx11<bits<7> op, string ps, string opName> {
  def _SADDR_gfx11 : FLAT_Real_gfx11<op, !cast<FLAT_Pseudo>(ps#"_SADDR"), opName>;
}

multiclass FLAT_Real_SADDR_RTN_gfx11<bits<7> op, string ps, string opName> {
  def _SADDR_RTN_gfx11 : FLAT_Real_gfx11<op, !cast<FLAT_Pseudo>(ps#"_SADDR_RTN"), opName>;
}

multiclass FLAT_Real_ST_gfx11<bits<7> op, string ps, string opName> {
  def _ST_gfx11 : FLAT_Real_gfx11<op, !cast<FLAT_Pseudo>(ps#"_ST"), opName> {
    let Inst{54-48} = !cast<int>(SGPR_NULL_gfx11plus.HWEncoding);
    let OtherPredicates = [HasFlatScratchSTMode];
  }
}

multiclass FLAT_Real_SVS_gfx11<bits<7> op, string ps, string opName> {
  def _SVS_gfx11 : FLAT_Real_gfx11<op, !cast<FLAT_Pseudo>(ps#"_SVS"), opName> {
    let OtherPredicates = [HasFlatScratchSVSMode];
  }
}

multiclass FLAT_Real_AllAddr_gfx11<bits<7> op, string ps, string opName, int renamed = false> :
  FLAT_Real_Base_gfx11<op, ps, opName, renamed>,
  FLAT_Real_SADDR_gfx11<op, ps, opName>;

multiclass FLAT_Real_Atomics_gfx11<bits<7> op, string ps, string opName, int renamed = false> :
  FLAT_Real_Base_gfx11<op, ps, opName, renamed>,
  FLAT_Real_RTN_gfx11<op, ps, opName>;

multiclass FLAT_Real_GlblAtomics_gfx11<bits<7> op, string ps, string opName, int renamed = false> :
  FLAT_Real_AllAddr_gfx11<op, ps, opName, renamed>,
  FLAT_Real_RTN_gfx11<op, ps, opName>,
  FLAT_Real_SADDR_RTN_gfx11<op, ps, opName>;

multiclass FLAT_Real_GlblAtomics_RTN_gfx11<bits<7> op, string ps, string opName> :
  FLAT_Real_RTN_gfx11<op, ps, opName>,
  FLAT_Real_SADDR_RTN_gfx11<op, ps, opName>;

multiclass FLAT_Real_ScratchAllAddr_gfx11<bits<7> op, string ps, string opName, int renamed = false> :
  FLAT_Real_Base_gfx11<op, ps, opName, renamed>,
  FLAT_Real_SADDR_gfx11<op, ps, opName>,
  FLAT_Real_ST_gfx11<op, ps, opName>,
  FLAT_Real_SVS_gfx11<op, ps, opName>;

// ENC_FLAT.
defm FLAT_LOAD_U8               : FLAT_Real_Base_gfx11<0x010, "FLAT_LOAD_UBYTE", "flat_load_u8", true>;
defm FLAT_LOAD_I8               : FLAT_Real_Base_gfx11<0x011, "FLAT_LOAD_SBYTE", "flat_load_i8", true>;
defm FLAT_LOAD_U16              : FLAT_Real_Base_gfx11<0x012, "FLAT_LOAD_USHORT", "flat_load_u16", true>;
defm FLAT_LOAD_I16              : FLAT_Real_Base_gfx11<0x013, "FLAT_LOAD_SSHORT", "flat_load_i16", true>;
defm FLAT_LOAD_B32              : FLAT_Real_Base_gfx11<0x014, "FLAT_LOAD_DWORD", "flat_load_b32", true>;
defm FLAT_LOAD_B64              : FLAT_Real_Base_gfx11<0x015, "FLAT_LOAD_DWORDX2", "flat_load_b64", true>;
defm FLAT_LOAD_B96              : FLAT_Real_Base_gfx11<0x016, "FLAT_LOAD_DWORDX3", "flat_load_b96", true>;
defm FLAT_LOAD_B128             : FLAT_Real_Base_gfx11<0x017, "FLAT_LOAD_DWORDX4", "flat_load_b128", true>;
defm FLAT_STORE_B8              : FLAT_Real_Base_gfx11<0x018, "FLAT_STORE_BYTE", "flat_store_b8", true>;
defm FLAT_STORE_B16             : FLAT_Real_Base_gfx11<0x019, "FLAT_STORE_SHORT", "flat_store_b16", true>;
defm FLAT_STORE_B32             : FLAT_Real_Base_gfx11<0x01a, "FLAT_STORE_DWORD", "flat_store_b32", true>;
defm FLAT_STORE_B64             : FLAT_Real_Base_gfx11<0x01b, "FLAT_STORE_DWORDX2", "flat_store_b64", true>;
defm FLAT_STORE_B96             : FLAT_Real_Base_gfx11<0x01c, "FLAT_STORE_DWORDX3", "flat_store_b96", true>;
defm FLAT_STORE_B128            : FLAT_Real_Base_gfx11<0x01d, "FLAT_STORE_DWORDX4", "flat_store_b128", true>;
defm FLAT_LOAD_D16_U8           : FLAT_Real_Base_gfx11<0x01e, "FLAT_LOAD_UBYTE_D16", "flat_load_d16_u8">;
defm FLAT_LOAD_D16_I8           : FLAT_Real_Base_gfx11<0x01f, "FLAT_LOAD_SBYTE_D16", "flat_load_d16_i8">;
defm FLAT_LOAD_D16_B16          : FLAT_Real_Base_gfx11<0x020, "FLAT_LOAD_SHORT_D16", "flat_load_d16_b16">;
defm FLAT_LOAD_D16_HI_U8        : FLAT_Real_Base_gfx11<0x021, "FLAT_LOAD_UBYTE_D16_HI", "flat_load_d16_hi_u8">;
defm FLAT_LOAD_D16_HI_I8        : FLAT_Real_Base_gfx11<0x022, "FLAT_LOAD_SBYTE_D16_HI", "flat_load_d16_hi_i8">;
defm FLAT_LOAD_D16_HI_B16       : FLAT_Real_Base_gfx11<0x023, "FLAT_LOAD_SHORT_D16_HI", "flat_load_d16_hi_b16">;
defm FLAT_STORE_D16_HI_B8       : FLAT_Real_Base_gfx11<0x024, "FLAT_STORE_BYTE_D16_HI", "flat_store_d16_hi_b8">;
defm FLAT_STORE_D16_HI_B16      : FLAT_Real_Base_gfx11<0x025, "FLAT_STORE_SHORT_D16_HI", "flat_store_d16_hi_b16">;
defm FLAT_ATOMIC_SWAP_B32       : FLAT_Real_Atomics_gfx11<0x033, "FLAT_ATOMIC_SWAP", "flat_atomic_swap_b32", true>;
defm FLAT_ATOMIC_CMPSWAP_B32    : FLAT_Real_Atomics_gfx11<0x034, "FLAT_ATOMIC_CMPSWAP", "flat_atomic_cmpswap_b32", true>;
defm FLAT_ATOMIC_ADD_U32        : FLAT_Real_Atomics_gfx11<0x035, "FLAT_ATOMIC_ADD", "flat_atomic_add_u32", true>;
defm FLAT_ATOMIC_SUB_U32        : FLAT_Real_Atomics_gfx11<0x036, "FLAT_ATOMIC_SUB", "flat_atomic_sub_u32", true>;
defm FLAT_ATOMIC_MIN_I32        : FLAT_Real_Atomics_gfx11<0x038, "FLAT_ATOMIC_SMIN", "flat_atomic_min_i32", true>;
defm FLAT_ATOMIC_MIN_U32        : FLAT_Real_Atomics_gfx11<0x039, "FLAT_ATOMIC_UMIN", "flat_atomic_min_u32", true>;
defm FLAT_ATOMIC_MAX_I32        : FLAT_Real_Atomics_gfx11<0x03a, "FLAT_ATOMIC_SMAX", "flat_atomic_max_i32", true>;
defm FLAT_ATOMIC_MAX_U32        : FLAT_Real_Atomics_gfx11<0x03b, "FLAT_ATOMIC_UMAX", "flat_atomic_max_u32", true>;
defm FLAT_ATOMIC_AND_B32        : FLAT_Real_Atomics_gfx11<0x03c, "FLAT_ATOMIC_AND", "flat_atomic_and_b32", true>;
defm FLAT_ATOMIC_OR_B32         : FLAT_Real_Atomics_gfx11<0x03d, "FLAT_ATOMIC_OR", "flat_atomic_or_b32", true>;
defm FLAT_ATOMIC_XOR_B32        : FLAT_Real_Atomics_gfx11<0x03e, "FLAT_ATOMIC_XOR", "flat_atomic_xor_b32", true>;
defm FLAT_ATOMIC_INC_U32        : FLAT_Real_Atomics_gfx11<0x03f, "FLAT_ATOMIC_INC", "flat_atomic_inc_u32", true>;
defm FLAT_ATOMIC_DEC_U32        : FLAT_Real_Atomics_gfx11<0x040, "FLAT_ATOMIC_DEC", "flat_atomic_dec_u32", true>;
defm FLAT_ATOMIC_SWAP_B64       : FLAT_Real_Atomics_gfx11<0x041, "FLAT_ATOMIC_SWAP_X2", "flat_atomic_swap_b64", true>;
defm FLAT_ATOMIC_CMPSWAP_B64    : FLAT_Real_Atomics_gfx11<0x042, "FLAT_ATOMIC_CMPSWAP_X2", "flat_atomic_cmpswap_b64", true>;
defm FLAT_ATOMIC_ADD_U64        : FLAT_Real_Atomics_gfx11<0x043, "FLAT_ATOMIC_ADD_X2", "flat_atomic_add_u64", true>;
defm FLAT_ATOMIC_SUB_U64        : FLAT_Real_Atomics_gfx11<0x044, "FLAT_ATOMIC_SUB_X2", "flat_atomic_sub_u64", true>;
defm FLAT_ATOMIC_MIN_I64        : FLAT_Real_Atomics_gfx11<0x045, "FLAT_ATOMIC_SMIN_X2", "flat_atomic_min_i64", true>;
defm FLAT_ATOMIC_MIN_U64        : FLAT_Real_Atomics_gfx11<0x046, "FLAT_ATOMIC_UMIN_X2", "flat_atomic_min_u64", true>;
defm FLAT_ATOMIC_MAX_I64        : FLAT_Real_Atomics_gfx11<0x047, "FLAT_ATOMIC_SMAX_X2", "flat_atomic_max_i64", true>;
defm FLAT_ATOMIC_MAX_U64        : FLAT_Real_Atomics_gfx11<0x048, "FLAT_ATOMIC_UMAX_X2", "flat_atomic_max_u64", true>;
defm FLAT_ATOMIC_AND_B64        : FLAT_Real_Atomics_gfx11<0x049, "FLAT_ATOMIC_AND_X2", "flat_atomic_and_b64", true>;
defm FLAT_ATOMIC_OR_B64         : FLAT_Real_Atomics_gfx11<0x04a, "FLAT_ATOMIC_OR_X2", "flat_atomic_or_b64", true>;
defm FLAT_ATOMIC_XOR_B64        : FLAT_Real_Atomics_gfx11<0x04b, "FLAT_ATOMIC_XOR_X2", "flat_atomic_xor_b64", true>;
defm FLAT_ATOMIC_INC_U64        : FLAT_Real_Atomics_gfx11<0x04c, "FLAT_ATOMIC_INC_X2", "flat_atomic_inc_u64", true>;
defm FLAT_ATOMIC_DEC_U64        : FLAT_Real_Atomics_gfx11<0x04d, "FLAT_ATOMIC_DEC_X2", "flat_atomic_dec_u64", true>;
defm FLAT_ATOMIC_CMPSWAP_F32    : FLAT_Real_Atomics_gfx11<0x050, "FLAT_ATOMIC_FCMPSWAP", "flat_atomic_cmpswap_f32">;
defm FLAT_ATOMIC_MIN_F32        : FLAT_Real_Atomics_gfx11<0x051, "FLAT_ATOMIC_FMIN", "flat_atomic_min_f32">;
defm FLAT_ATOMIC_MAX_F32        : FLAT_Real_Atomics_gfx11<0x052, "FLAT_ATOMIC_FMAX", "flat_atomic_max_f32">;
defm FLAT_ATOMIC_ADD_F32        : FLAT_Real_Atomics_gfx11<0x056, "FLAT_ATOMIC_ADD_F32", "flat_atomic_add_f32">;

// ENC_FLAT_GLBL.
defm GLOBAL_LOAD_U8             : FLAT_Real_AllAddr_gfx11<0x010, "GLOBAL_LOAD_UBYTE", "global_load_u8", true>;
defm GLOBAL_LOAD_I8             : FLAT_Real_AllAddr_gfx11<0x011, "GLOBAL_LOAD_SBYTE", "global_load_i8", true>;
defm GLOBAL_LOAD_U16            : FLAT_Real_AllAddr_gfx11<0x012, "GLOBAL_LOAD_USHORT", "global_load_u16", true>;
defm GLOBAL_LOAD_I16            : FLAT_Real_AllAddr_gfx11<0x013, "GLOBAL_LOAD_SSHORT", "global_load_i16", true>;
defm GLOBAL_LOAD_B32            : FLAT_Real_AllAddr_gfx11<0x014, "GLOBAL_LOAD_DWORD", "global_load_b32", true>;
defm GLOBAL_LOAD_B64            : FLAT_Real_AllAddr_gfx11<0x015, "GLOBAL_LOAD_DWORDX2", "global_load_b64", true>;
defm GLOBAL_LOAD_B96            : FLAT_Real_AllAddr_gfx11<0x016, "GLOBAL_LOAD_DWORDX3", "global_load_b96", true>;
defm GLOBAL_LOAD_B128           : FLAT_Real_AllAddr_gfx11<0x017, "GLOBAL_LOAD_DWORDX4", "global_load_b128", true>;
defm GLOBAL_STORE_B8            : FLAT_Real_AllAddr_gfx11<0x018, "GLOBAL_STORE_BYTE", "global_store_b8", true>;
defm GLOBAL_STORE_B16           : FLAT_Real_AllAddr_gfx11<0x019, "GLOBAL_STORE_SHORT", "global_store_b16", true>;
defm GLOBAL_STORE_B32           : FLAT_Real_AllAddr_gfx11<0x01a, "GLOBAL_STORE_DWORD", "global_store_b32", true>;
defm GLOBAL_STORE_B64           : FLAT_Real_AllAddr_gfx11<0x01b, "GLOBAL_STORE_DWORDX2", "global_store_b64", true>;
defm GLOBAL_STORE_B96           : FLAT_Real_AllAddr_gfx11<0x01c, "GLOBAL_STORE_DWORDX3", "global_store_b96", true>;
defm GLOBAL_STORE_B128          : FLAT_Real_AllAddr_gfx11<0x01d, "GLOBAL_STORE_DWORDX4", "global_store_b128", true>;
defm GLOBAL_LOAD_D16_U8         : FLAT_Real_AllAddr_gfx11<0x01e, "GLOBAL_LOAD_UBYTE_D16", "global_load_d16_u8">;
defm GLOBAL_LOAD_D16_I8         : FLAT_Real_AllAddr_gfx11<0x01f, "GLOBAL_LOAD_SBYTE_D16", "global_load_d16_i8">;
defm GLOBAL_LOAD_D16_B16        : FLAT_Real_AllAddr_gfx11<0x020, "GLOBAL_LOAD_SHORT_D16", "global_load_d16_b16">;
defm GLOBAL_LOAD_D16_HI_U8      : FLAT_Real_AllAddr_gfx11<0x021, "GLOBAL_LOAD_UBYTE_D16_HI", "global_load_d16_hi_u8">;
defm GLOBAL_LOAD_D16_HI_I8      : FLAT_Real_AllAddr_gfx11<0x022, "GLOBAL_LOAD_SBYTE_D16_HI", "global_load_d16_hi_i8">;
defm GLOBAL_LOAD_D16_HI_B16     : FLAT_Real_AllAddr_gfx11<0x023, "GLOBAL_LOAD_SHORT_D16_HI", "global_load_d16_hi_b16">;
defm GLOBAL_STORE_D16_HI_B8     : FLAT_Real_AllAddr_gfx11<0x024, "GLOBAL_STORE_BYTE_D16_HI", "global_store_d16_hi_b8">;
defm GLOBAL_STORE_D16_HI_B16    : FLAT_Real_AllAddr_gfx11<0x025, "GLOBAL_STORE_SHORT_D16_HI", "global_store_d16_hi_b16">;
defm GLOBAL_LOAD_ADDTID_B32     : FLAT_Real_AllAddr_gfx11<0x028, "GLOBAL_LOAD_DWORD_ADDTID", "global_load_addtid_b32">;
defm GLOBAL_STORE_ADDTID_B32    : FLAT_Real_AllAddr_gfx11<0x029, "GLOBAL_STORE_DWORD_ADDTID", "global_store_addtid_b32">;
defm GLOBAL_ATOMIC_SWAP_B32     : FLAT_Real_GlblAtomics_gfx11<0x033, "GLOBAL_ATOMIC_SWAP", "global_atomic_swap_b32", true>;
defm GLOBAL_ATOMIC_CMPSWAP_B32  : FLAT_Real_GlblAtomics_gfx11<0x034, "GLOBAL_ATOMIC_CMPSWAP", "global_atomic_cmpswap_b32", true>;
defm GLOBAL_ATOMIC_ADD_U32      : FLAT_Real_GlblAtomics_gfx11<0x035, "GLOBAL_ATOMIC_ADD", "global_atomic_add_u32", true>;
defm GLOBAL_ATOMIC_SUB_U32      : FLAT_Real_GlblAtomics_gfx11<0x036, "GLOBAL_ATOMIC_SUB", "global_atomic_sub_u32", true>;
defm GLOBAL_ATOMIC_CSUB_U32     : FLAT_Real_GlblAtomics_RTN_gfx11<0x037, "GLOBAL_ATOMIC_CSUB", "global_atomic_csub_u32">;
defm GLOBAL_ATOMIC_MIN_I32      : FLAT_Real_GlblAtomics_gfx11<0x038, "GLOBAL_ATOMIC_SMIN", "global_atomic_min_i32", true>;
defm GLOBAL_ATOMIC_MIN_U32      : FLAT_Real_GlblAtomics_gfx11<0x039, "GLOBAL_ATOMIC_UMIN", "global_atomic_min_u32", true>;
defm GLOBAL_ATOMIC_MAX_I32      : FLAT_Real_GlblAtomics_gfx11<0x03a, "GLOBAL_ATOMIC_SMAX", "global_atomic_max_i32", true>;
defm GLOBAL_ATOMIC_MAX_U32      : FLAT_Real_GlblAtomics_gfx11<0x03b, "GLOBAL_ATOMIC_UMAX", "global_atomic_max_u32", true>;
defm GLOBAL_ATOMIC_AND_B32      : FLAT_Real_GlblAtomics_gfx11<0x03c, "GLOBAL_ATOMIC_AND", "global_atomic_and_b32", true>;
defm GLOBAL_ATOMIC_OR_B32       : FLAT_Real_GlblAtomics_gfx11<0x03d, "GLOBAL_ATOMIC_OR", "global_atomic_or_b32", true>;
defm GLOBAL_ATOMIC_XOR_B32      : FLAT_Real_GlblAtomics_gfx11<0x03e, "GLOBAL_ATOMIC_XOR", "global_atomic_xor_b32", true>;
defm GLOBAL_ATOMIC_INC_U32      : FLAT_Real_GlblAtomics_gfx11<0x03f, "GLOBAL_ATOMIC_INC", "global_atomic_inc_u32", true>;
defm GLOBAL_ATOMIC_DEC_U32      : FLAT_Real_GlblAtomics_gfx11<0x040, "GLOBAL_ATOMIC_DEC", "global_atomic_dec_u32", true>;
defm GLOBAL_ATOMIC_SWAP_B64     : FLAT_Real_GlblAtomics_gfx11<0x041, "GLOBAL_ATOMIC_SWAP_X2", "global_atomic_swap_b64", true>;
defm GLOBAL_ATOMIC_CMPSWAP_B64  : FLAT_Real_GlblAtomics_gfx11<0x042, "GLOBAL_ATOMIC_CMPSWAP_X2", "global_atomic_cmpswap_b64", true>;
defm GLOBAL_ATOMIC_ADD_U64      : FLAT_Real_GlblAtomics_gfx11<0x043, "GLOBAL_ATOMIC_ADD_X2", "global_atomic_add_u64", true>;
defm GLOBAL_ATOMIC_SUB_U64      : FLAT_Real_GlblAtomics_gfx11<0x044, "GLOBAL_ATOMIC_SUB_X2", "global_atomic_sub_u64", true>;
defm GLOBAL_ATOMIC_MIN_I64      : FLAT_Real_GlblAtomics_gfx11<0x045, "GLOBAL_ATOMIC_SMIN_X2", "global_atomic_min_i64", true>;
defm GLOBAL_ATOMIC_MIN_U64      : FLAT_Real_GlblAtomics_gfx11<0x046, "GLOBAL_ATOMIC_UMIN_X2", "global_atomic_min_u64", true>;
defm GLOBAL_ATOMIC_MAX_I64      : FLAT_Real_GlblAtomics_gfx11<0x047, "GLOBAL_ATOMIC_SMAX_X2", "global_atomic_max_i64", true>;
defm GLOBAL_ATOMIC_MAX_U64      : FLAT_Real_GlblAtomics_gfx11<0x048, "GLOBAL_ATOMIC_UMAX_X2", "global_atomic_max_u64", true>;
defm GLOBAL_ATOMIC_AND_B64      : FLAT_Real_GlblAtomics_gfx11<0x049, "GLOBAL_ATOMIC_AND_X2", "global_atomic_and_b64", true>;
defm GLOBAL_ATOMIC_OR_B64       : FLAT_Real_GlblAtomics_gfx11<0x04a, "GLOBAL_ATOMIC_OR_X2", "global_atomic_or_b64", true>;
defm GLOBAL_ATOMIC_XOR_B64      : FLAT_Real_GlblAtomics_gfx11<0x04b, "GLOBAL_ATOMIC_XOR_X2", "global_atomic_xor_b64", true>;
defm GLOBAL_ATOMIC_INC_U64      : FLAT_Real_GlblAtomics_gfx11<0x04c, "GLOBAL_ATOMIC_INC_X2", "global_atomic_inc_u64", true>;
defm GLOBAL_ATOMIC_DEC_U64      : FLAT_Real_GlblAtomics_gfx11<0x04d, "GLOBAL_ATOMIC_DEC_X2", "global_atomic_dec_u64", true>;
defm GLOBAL_ATOMIC_CMPSWAP_F32  : FLAT_Real_GlblAtomics_gfx11<0x050, "GLOBAL_ATOMIC_FCMPSWAP", "global_atomic_cmpswap_f32">;
defm GLOBAL_ATOMIC_MIN_F32      : FLAT_Real_GlblAtomics_gfx11<0x051, "GLOBAL_ATOMIC_FMIN", "global_atomic_min_f32">;
defm GLOBAL_ATOMIC_MAX_F32      : FLAT_Real_GlblAtomics_gfx11<0x052, "GLOBAL_ATOMIC_FMAX", "global_atomic_max_f32">;
defm GLOBAL_ATOMIC_ADD_F32      : FLAT_Real_GlblAtomics_gfx11<0x056, "GLOBAL_ATOMIC_ADD_F32", "global_atomic_add_f32">;

// ENC_FLAT_SCRATCH.
defm SCRATCH_LOAD_U8            : FLAT_Real_ScratchAllAddr_gfx11<0x10, "SCRATCH_LOAD_UBYTE", "scratch_load_u8", true>;
defm SCRATCH_LOAD_I8            : FLAT_Real_ScratchAllAddr_gfx11<0x11, "SCRATCH_LOAD_SBYTE", "scratch_load_i8", true>;
defm SCRATCH_LOAD_U16           : FLAT_Real_ScratchAllAddr_gfx11<0x12, "SCRATCH_LOAD_USHORT", "scratch_load_u16", true>;
defm SCRATCH_LOAD_I16           : FLAT_Real_ScratchAllAddr_gfx11<0x13, "SCRATCH_LOAD_SSHORT", "scratch_load_i16", true>;
defm SCRATCH_LOAD_B32           : FLAT_Real_ScratchAllAddr_gfx11<0x14, "SCRATCH_LOAD_DWORD", "scratch_load_b32", true>;
defm SCRATCH_LOAD_B64           : FLAT_Real_ScratchAllAddr_gfx11<0x15, "SCRATCH_LOAD_DWORDX2", "scratch_load_b64", true>;
defm SCRATCH_LOAD_B96           : FLAT_Real_ScratchAllAddr_gfx11<0x16, "SCRATCH_LOAD_DWORDX3", "scratch_load_b96", true>;
defm SCRATCH_LOAD_B128          : FLAT_Real_ScratchAllAddr_gfx11<0x17, "SCRATCH_LOAD_DWORDX4", "scratch_load_b128", true>;
defm SCRATCH_STORE_B8           : FLAT_Real_ScratchAllAddr_gfx11<0x18, "SCRATCH_STORE_BYTE", "scratch_store_b8", true>;
defm SCRATCH_STORE_B16          : FLAT_Real_ScratchAllAddr_gfx11<0x19, "SCRATCH_STORE_SHORT", "scratch_store_b16", true>;
defm SCRATCH_STORE_B32          : FLAT_Real_ScratchAllAddr_gfx11<0x1a, "SCRATCH_STORE_DWORD", "scratch_store_b32", true>;
defm SCRATCH_STORE_B64          : FLAT_Real_ScratchAllAddr_gfx11<0x1b, "SCRATCH_STORE_DWORDX2", "scratch_store_b64", true>;
defm SCRATCH_STORE_B96          : FLAT_Real_ScratchAllAddr_gfx11<0x1c, "SCRATCH_STORE_DWORDX3", "scratch_store_b96", true>;
defm SCRATCH_STORE_B128         : FLAT_Real_ScratchAllAddr_gfx11<0x1d, "SCRATCH_STORE_DWORDX4", "scratch_store_b128", true>;
defm SCRATCH_LOAD_D16_U8        : FLAT_Real_ScratchAllAddr_gfx11<0x1e, "SCRATCH_LOAD_UBYTE_D16", "scratch_load_d16_u8">;
defm SCRATCH_LOAD_D16_I8        : FLAT_Real_ScratchAllAddr_gfx11<0x1f, "SCRATCH_LOAD_SBYTE_D16", "scratch_load_d16_i8">;
defm SCRATCH_LOAD_D16_B16       : FLAT_Real_ScratchAllAddr_gfx11<0x20, "SCRATCH_LOAD_SHORT_D16", "scratch_load_d16_b16">;
defm SCRATCH_LOAD_D16_HI_U8     : FLAT_Real_ScratchAllAddr_gfx11<0x21, "SCRATCH_LOAD_UBYTE_D16_HI", "scratch_load_d16_hi_u8">;
defm SCRATCH_LOAD_D16_HI_I8     : FLAT_Real_ScratchAllAddr_gfx11<0x22, "SCRATCH_LOAD_SBYTE_D16_HI", "scratch_load_d16_hi_i8">;
defm SCRATCH_LOAD_D16_HI_B16    : FLAT_Real_ScratchAllAddr_gfx11<0x23, "SCRATCH_LOAD_SHORT_D16_HI", "scratch_load_d16_hi_b16">;
defm SCRATCH_STORE_D16_HI_B8    : FLAT_Real_ScratchAllAddr_gfx11<0x24, "SCRATCH_STORE_BYTE_D16_HI", "scratch_store_d16_hi_b8">;
defm SCRATCH_STORE_D16_HI_B16   : FLAT_Real_ScratchAllAddr_gfx11<0x25, "SCRATCH_STORE_SHORT_D16_HI", "scratch_store_d16_hi_b16">;
