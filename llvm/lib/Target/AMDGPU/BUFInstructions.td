//===-- BUFInstructions.td - Buffer Instruction Definitions ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

def MUBUFAddr64 : ComplexPattern<i64, 4, "SelectMUBUFAddr64">;
def MUBUFOffset : ComplexPattern<i64, 3, "SelectMUBUFOffset">;

def MUBUFScratchOffen : ComplexPattern<i64, 4, "SelectMUBUFScratchOffen", [], [SDNPWantParent]>;
def MUBUFScratchOffset : ComplexPattern<i64, 3, "SelectMUBUFScratchOffset", [], [SDNPWantParent], 20>;

def BUFAddrKind {
  int Offset = 0;
  int OffEn  = 1;
  int IdxEn  = 2;
  int BothEn = 3;
  int Addr64 = 4;
}

class getAddrName<int addrKind> {
  string ret =
    !if(!eq(addrKind, BUFAddrKind.Offset), "offset",
    !if(!eq(addrKind, BUFAddrKind.OffEn),  "offen",
    !if(!eq(addrKind, BUFAddrKind.IdxEn),  "idxen",
    !if(!eq(addrKind, BUFAddrKind.BothEn), "bothen",
    !if(!eq(addrKind, BUFAddrKind.Addr64), "addr64",
    "")))));
}

class MUBUFAddr64Table <bit is_addr64, string Name> {
  bit IsAddr64 = is_addr64;
  string OpName = Name;
}

class MUBUFLdsTable <bit is_lds, string Name> {
  bit IsLds = is_lds;
  string OpName = Name;
}

class MTBUFAddr64Table <bit is_addr64, string Name> {
  bit IsAddr64 = is_addr64;
  string OpName = Name;
}

//===----------------------------------------------------------------------===//
// MTBUF classes
//===----------------------------------------------------------------------===//

class MTBUFGetBaseOpcode<string Op> {
  string ret = !subst("FORMAT_XY", "FORMAT_X",
    !subst("FORMAT_XYZ", "FORMAT_X",
    !subst("FORMAT_XYZW", "FORMAT_X", Op)));
}

class getMTBUFElements<string Op> {
  int ret = 1;
}


class MTBUF_Pseudo <string opName, dag outs, dag ins,
                    string asmOps, list<dag> pattern=[]> :
  InstSI<outs, ins, "", pattern>,
  SIMCInstr<opName, SIEncodingFamily.NONE> {

  let isPseudo = 1;
  let isCodeGenOnly = 1;
  let Size = 8;
  let UseNamedOperandTable = 1;

  string Mnemonic = opName;
  string AsmOperands = asmOps;

  Instruction Opcode = !cast<Instruction>(NAME);
  Instruction BaseOpcode = !cast<Instruction>(MTBUFGetBaseOpcode<NAME>.ret);

  let VM_CNT = 1;
  let EXP_CNT = 1;
  let MTBUF = 1;
  let Uses = [EXEC];
  let hasSideEffects = 0;
  let SchedRW = [WriteVMEM];

  let AsmMatchConverter = "cvtMtbuf";

  bits<1> offen       = 0;
  bits<1> idxen       = 0;
  bits<1> addr64      = 0;
  bits<1> has_vdata   = 1;
  bits<1> has_vaddr   = 1;
  bits<1> has_glc     = 1;
  bits<1> has_dlc     = 1;
  bits<1> glc_value   = 0; // the value for glc if no such operand
  bits<1> dlc_value   = 0; // the value for dlc if no such operand
  bits<1> has_srsrc   = 1;
  bits<1> has_soffset = 1;
  bits<1> has_offset  = 1;
  bits<1> has_slc     = 1;
  bits<1> has_tfe     = 1;
  bits<4> elements    = 0;
  bits<1> has_sccb    = 1;
  bits<1> sccb_value  = 0;
}

class MTBUF_Real <MTBUF_Pseudo ps> :
  InstSI <ps.OutOperandList, ps.InOperandList, ps.Mnemonic # ps.AsmOperands, []> {

  let isPseudo = 0;
  let isCodeGenOnly = 0;

  // copy relevant pseudo op flags
  let UseNamedOperandTable = ps.UseNamedOperandTable;
  let SubtargetPredicate = ps.SubtargetPredicate;
  let AsmMatchConverter  = ps.AsmMatchConverter;
  let Constraints        = ps.Constraints;
  let DisableEncoding    = ps.DisableEncoding;
  let TSFlags            = ps.TSFlags;
  let SchedRW            = ps.SchedRW;

  bits<12> offset;
  bits<5>  cpol;
  bits<7>  format;
  bits<8>  vaddr;
  bits<10> vdata;
  bits<7>  srsrc;
  bits<1>  tfe;
  bits<8>  soffset;

  bits<4> dfmt = format{3-0};
  bits<3> nfmt = format{6-4};

  // GFX90A+ only: instruction uses AccVGPR for data
  // Bit superceedes tfe.
  bits<1> acc = !if(ps.has_vdata, vdata{9}, 0);
}

class getMTBUFInsDA<list<RegisterClass> vdataList,
                    list<RegisterClass> vaddrList=[]> {
  RegisterClass vdataClass = !if(!empty(vdataList), ?, !head(vdataList));
  RegisterClass vaddrClass = !if(!empty(vaddrList), ?, !head(vaddrList));
  RegisterOperand vdata_op = getLdStRegisterOperand<vdataClass>.ret;
  dag InsNoData = !if(!empty(vaddrList),
    (ins                    SReg_128:$srsrc, SCSrc_b32:$soffset,
         offset:$offset, FORMAT:$format, CPol:$cpol, TFE:$tfe, SWZ:$swz),
    (ins vaddrClass:$vaddr, SReg_128:$srsrc, SCSrc_b32:$soffset,
         offset:$offset, FORMAT:$format, CPol:$cpol, TFE:$tfe, SWZ:$swz)
  );
  dag InsData = !if(!empty(vaddrList),
    (ins vdata_op:$vdata,                    SReg_128:$srsrc,
         SCSrc_b32:$soffset, offset:$offset, FORMAT:$format, CPol:$cpol,
         TFE:$tfe, SWZ:$swz),
    (ins vdata_op:$vdata, vaddrClass:$vaddr, SReg_128:$srsrc,
         SCSrc_b32:$soffset, offset:$offset, FORMAT:$format, CPol:$cpol,
         TFE:$tfe, SWZ:$swz)
  );
  dag ret = !if(!empty(vdataList), InsNoData, InsData);
}

class getMTBUFIns<int addrKind, list<RegisterClass> vdataList=[]> {
  dag ret =
    !if(!eq(addrKind, BUFAddrKind.Offset), getMTBUFInsDA<vdataList>.ret,
    !if(!eq(addrKind, BUFAddrKind.OffEn),  getMTBUFInsDA<vdataList, [VGPR_32]>.ret,
    !if(!eq(addrKind, BUFAddrKind.IdxEn),  getMTBUFInsDA<vdataList, [VGPR_32]>.ret,
    !if(!eq(addrKind, BUFAddrKind.BothEn), getMTBUFInsDA<vdataList, [VReg_64]>.ret,
    !if(!eq(addrKind, BUFAddrKind.Addr64), getMTBUFInsDA<vdataList, [VReg_64]>.ret,
    (ins))))));
}

class getMTBUFAsmOps<int addrKind> {
  string Pfx =
    !if(!eq(addrKind, BUFAddrKind.Offset), "off, $srsrc,$format $soffset",
    !if(!eq(addrKind, BUFAddrKind.OffEn),
            "$vaddr, $srsrc,$format $soffset offen",
    !if(!eq(addrKind, BUFAddrKind.IdxEn),
            "$vaddr, $srsrc,$format $soffset idxen",
    !if(!eq(addrKind, BUFAddrKind.BothEn),
            "$vaddr, $srsrc,$format $soffset idxen offen",
    !if(!eq(addrKind, BUFAddrKind.Addr64),
            "$vaddr, $srsrc,$format $soffset addr64",
    "")))));
  string ret = Pfx # "$offset";
}

class MTBUF_SetupAddr<int addrKind> {
  bits<1> offen  = !or(!eq(addrKind, BUFAddrKind.OffEn),
                       !eq(addrKind, BUFAddrKind.BothEn));

  bits<1> idxen  = !or(!eq(addrKind, BUFAddrKind.IdxEn),
                       !eq(addrKind, BUFAddrKind.BothEn));

  bits<1> addr64 = !eq(addrKind, BUFAddrKind.Addr64);

  bits<1> has_vaddr = !ne(addrKind, BUFAddrKind.Offset);
}

class MTBUF_Load_Pseudo <string opName,
                         int addrKind,
                         RegisterClass vdataClass,
                         int elems,
                         list<dag> pattern=[],
                         // Workaround bug bz30254
                         int addrKindCopy = addrKind>
  : MTBUF_Pseudo<opName,
                 (outs getLdStRegisterOperand<vdataClass>.ret:$vdata),
                 getMTBUFIns<addrKindCopy>.ret,
                 " $vdata, " # getMTBUFAsmOps<addrKindCopy>.ret # "$cpol$tfe$swz",
                 pattern>,
    MTBUF_SetupAddr<addrKindCopy> {
  let PseudoInstr = opName # "_" # getAddrName<addrKindCopy>.ret;
  let mayLoad = 1;
  let mayStore = 0;
  let elements = elems;
}

multiclass MTBUF_Pseudo_Loads<string opName, RegisterClass vdataClass,
                              int elems, ValueType load_vt = i32,
                              SDPatternOperator ld = null_frag> {

  def _OFFSET : MTBUF_Load_Pseudo <opName, BUFAddrKind.Offset, vdataClass, elems>,
                MTBUFAddr64Table<0, NAME>;

  def _ADDR64 : MTBUF_Load_Pseudo <opName, BUFAddrKind.Addr64, vdataClass, elems>,
                MTBUFAddr64Table<1, NAME>;

  def _OFFEN  : MTBUF_Load_Pseudo <opName, BUFAddrKind.OffEn, vdataClass, elems>;
  def _IDXEN  : MTBUF_Load_Pseudo <opName, BUFAddrKind.IdxEn, vdataClass, elems>;
  def _BOTHEN : MTBUF_Load_Pseudo <opName, BUFAddrKind.BothEn, vdataClass, elems>;

  let DisableWQM = 1 in {
    def _OFFSET_exact : MTBUF_Load_Pseudo <opName, BUFAddrKind.Offset, vdataClass, elems>;
    def _OFFEN_exact  : MTBUF_Load_Pseudo <opName, BUFAddrKind.OffEn, vdataClass, elems>;
    def _IDXEN_exact  : MTBUF_Load_Pseudo <opName, BUFAddrKind.IdxEn, vdataClass, elems>;
    def _BOTHEN_exact : MTBUF_Load_Pseudo <opName, BUFAddrKind.BothEn, vdataClass, elems>;
  }
}

class MTBUF_Store_Pseudo <string opName,
                          int addrKind,
                          RegisterClass vdataClass,
                          int elems,
                          list<dag> pattern=[],
                          // Workaround bug bz30254
                          int addrKindCopy = addrKind,
                          RegisterClass vdataClassCopy = vdataClass>
  : MTBUF_Pseudo<opName,
                 (outs),
                 getMTBUFIns<addrKindCopy, [vdataClassCopy]>.ret,
                 " $vdata, " # getMTBUFAsmOps<addrKindCopy>.ret # "$cpol$tfe$swz",
                 pattern>,
    MTBUF_SetupAddr<addrKindCopy> {
  let PseudoInstr = opName # "_" # getAddrName<addrKindCopy>.ret;
  let mayLoad = 0;
  let mayStore = 1;
  let elements = elems;
}

multiclass MTBUF_Pseudo_Stores<string opName, RegisterClass vdataClass,
                               int elems, ValueType store_vt = i32,
                               SDPatternOperator st = null_frag> {

  def _OFFSET : MTBUF_Store_Pseudo <opName, BUFAddrKind.Offset, vdataClass, elems>,
    MTBUFAddr64Table<0, NAME>;

  def _ADDR64 : MTBUF_Store_Pseudo <opName, BUFAddrKind.Addr64, vdataClass, elems>,
    MTBUFAddr64Table<1, NAME>;

  def _OFFEN  : MTBUF_Store_Pseudo <opName, BUFAddrKind.OffEn, vdataClass, elems>;
  def _IDXEN  : MTBUF_Store_Pseudo <opName, BUFAddrKind.IdxEn, vdataClass, elems>;
  def _BOTHEN : MTBUF_Store_Pseudo <opName, BUFAddrKind.BothEn, vdataClass, elems>;

  let DisableWQM = 1 in {
    def _OFFSET_exact : MTBUF_Store_Pseudo <opName, BUFAddrKind.Offset, vdataClass, elems>;
    def _OFFEN_exact  : MTBUF_Store_Pseudo <opName, BUFAddrKind.OffEn, vdataClass, elems>;
    def _IDXEN_exact  : MTBUF_Store_Pseudo <opName, BUFAddrKind.IdxEn, vdataClass, elems>;
    def _BOTHEN_exact : MTBUF_Store_Pseudo <opName, BUFAddrKind.BothEn, vdataClass, elems>;
  }
}


//===----------------------------------------------------------------------===//
// MUBUF classes
//===----------------------------------------------------------------------===//

class MUBUFGetBaseOpcode<string Op> {
  string ret = !subst("DWORDX2", "DWORD",
    !subst("DWORDX3", "DWORD",
    !subst("DWORDX4", "DWORD", Op)));
}

class MUBUF_Pseudo <string opName, dag outs, dag ins,
                    string asmOps, list<dag> pattern=[]> :
  InstSI<outs, ins, "", pattern>,
  SIMCInstr<opName, SIEncodingFamily.NONE> {

  let isPseudo = 1;
  let isCodeGenOnly = 1;
  let Size = 8;
  let UseNamedOperandTable = 1;

  string Mnemonic = opName;
  string AsmOperands = asmOps;

  Instruction Opcode = !cast<Instruction>(NAME);
  Instruction BaseOpcode = !cast<Instruction>(MUBUFGetBaseOpcode<NAME>.ret);

  let VM_CNT = 1;
  let EXP_CNT = 1;
  let MUBUF = 1;
  let Uses = [EXEC];
  let hasSideEffects = 0;
  let SchedRW = [WriteVMEM];

  let AsmMatchConverter = "cvtMubuf";

  bits<1> offen       = 0;
  bits<1> idxen       = 0;
  bits<1> addr64      = 0;
  bits<1> lds         = 0;
  bits<1> has_vdata   = 1;
  bits<1> has_vaddr   = 1;
  bits<1> has_glc     = 1;
  bits<1> has_dlc     = 1;
  bits<1> glc_value   = 0; // the value for glc if no such operand
  bits<1> dlc_value   = 0; // the value for dlc if no such operand
  bits<1> has_srsrc   = 1;
  bits<1> has_soffset = 1;
  bits<1> has_offset  = 1;
  bits<1> has_slc     = 1;
  bits<1> has_tfe     = 1;
  bits<4> elements    = 0;
  bits<1> has_sccb    = 1;
  bits<1> sccb_value  = 0;
}

class MUBUF_Real <MUBUF_Pseudo ps> :
  InstSI <ps.OutOperandList, ps.InOperandList, ps.Mnemonic # ps.AsmOperands, []> {

  let isPseudo = 0;
  let isCodeGenOnly = 0;

  // copy relevant pseudo op flags
  let SubtargetPredicate   = ps.SubtargetPredicate;
  let AsmMatchConverter    = ps.AsmMatchConverter;
  let OtherPredicates      = ps.OtherPredicates;
  let Constraints          = ps.Constraints;
  let DisableEncoding      = ps.DisableEncoding;
  let TSFlags              = ps.TSFlags;
  let UseNamedOperandTable = ps.UseNamedOperandTable;
  let SchedRW              = ps.SchedRW;

  bits<12> offset;
  bits<5>  cpol;
  bits<8>  vaddr;
  bits<10> vdata;
  bits<7>  srsrc;
  bits<1>  tfe;
  bits<8>  soffset;

  // GFX90A+ only: instruction uses AccVGPR for data
  // Bit superceedes tfe.
  bits<1> acc = !if(ps.has_vdata, vdata{9}, 0);
}


// For cache invalidation instructions.
class MUBUF_Invalidate <string opName, SDPatternOperator node = null_frag> :
  MUBUF_Pseudo<opName, (outs), (ins), "", [(node)]> {

  let AsmMatchConverter = "";

  let hasSideEffects = 1;
  let mayLoad = 0;
  let mayStore = 0;

  // Set everything to 0.
  let offen       = 0;
  let idxen       = 0;
  let addr64      = 0;
  let has_vdata   = 0;
  let has_vaddr   = 0;
  let has_glc     = 0;
  let has_dlc     = 0;
  let glc_value   = 0;
  let dlc_value   = 0;
  let has_srsrc   = 0;
  let has_soffset = 0;
  let has_offset  = 0;
  let has_slc     = 0;
  let has_tfe     = 0;
  let has_sccb    = 0;
  let sccb_value  = 0;
}

class getMUBUFInsDA<list<RegisterClass> vdataList,
                    list<RegisterClass> vaddrList=[],
                    bit isLds = 0> {
  RegisterClass vdataClass = !if(!empty(vdataList), ?, !head(vdataList));
  RegisterClass vaddrClass = !if(!empty(vaddrList), ?, !head(vaddrList));
  RegisterOperand vdata_op = getLdStRegisterOperand<vdataClass>.ret;
  dag InsNoData = !if(!empty(vaddrList),
    (ins                    SReg_128:$srsrc, SCSrc_b32:$soffset,
         offset:$offset, CPol_0:$cpol),
    (ins vaddrClass:$vaddr, SReg_128:$srsrc, SCSrc_b32:$soffset,
         offset:$offset, CPol_0:$cpol)
  );
  dag InsData = !if(!empty(vaddrList),
    (ins vdata_op:$vdata,                    SReg_128:$srsrc,
         SCSrc_b32:$soffset, offset:$offset, CPol_0:$cpol),
    (ins vdata_op:$vdata, vaddrClass:$vaddr, SReg_128:$srsrc,
         SCSrc_b32:$soffset, offset:$offset, CPol_0:$cpol)
  );
  dag ret = !con(
              !if(!empty(vdataList), InsNoData, InsData),
              !if(isLds, (ins SWZ_0:$swz), (ins TFE_0:$tfe, SWZ_0:$swz))
             );
}

class getMUBUFElements<ValueType vt> {
  // eq does not support ValueType for some reason.
  string vtAsStr = !cast<string>(vt);

  int ret =
    !if(!eq(vtAsStr, "f16"), 1,
      !if(!eq(vtAsStr, "v2f16"), 2,
        !if(!eq(vtAsStr, "v3f16"), 3,
          !if(!eq(vtAsStr, "v4f16"), 4,
            !if(!eq(vt.Size, 32), 1,
              !if(!eq(vt.Size, 64), 2,
                !if(!eq(vt.Size, 96), 3,
                  !if(!eq(vt.Size, 128), 4, 0)
                )
              )
            )
          )
        )
      )
    );
}

class getMUBUFIns<int addrKind, list<RegisterClass> vdataList=[], bit isLds = 0> {
  dag ret =
    !if(!eq(addrKind, BUFAddrKind.Offset), getMUBUFInsDA<vdataList, [], isLds>.ret,
    !if(!eq(addrKind, BUFAddrKind.OffEn),  getMUBUFInsDA<vdataList, [VGPR_32], isLds>.ret,
    !if(!eq(addrKind, BUFAddrKind.IdxEn),  getMUBUFInsDA<vdataList, [VGPR_32], isLds>.ret,
    !if(!eq(addrKind, BUFAddrKind.BothEn), getMUBUFInsDA<vdataList, [VReg_64], isLds>.ret,
    !if(!eq(addrKind, BUFAddrKind.Addr64), getMUBUFInsDA<vdataList, [VReg_64], isLds>.ret,
    (ins))))));
}

class getMUBUFAsmOps<int addrKind> {
  string Pfx =
    !if(!eq(addrKind, BUFAddrKind.Offset), "off, $srsrc, $soffset",
    !if(!eq(addrKind, BUFAddrKind.OffEn),  "$vaddr, $srsrc, $soffset offen",
    !if(!eq(addrKind, BUFAddrKind.IdxEn),  "$vaddr, $srsrc, $soffset idxen",
    !if(!eq(addrKind, BUFAddrKind.BothEn), "$vaddr, $srsrc, $soffset idxen offen",
    !if(!eq(addrKind, BUFAddrKind.Addr64), "$vaddr, $srsrc, $soffset addr64",
    "")))));
  string ret = Pfx # "$offset";
}

class MUBUF_SetupAddr<int addrKind> {
  bits<1> offen  = !or(!eq(addrKind, BUFAddrKind.OffEn),
                       !eq(addrKind, BUFAddrKind.BothEn));

  bits<1> idxen  = !or(!eq(addrKind, BUFAddrKind.IdxEn),
                       !eq(addrKind, BUFAddrKind.BothEn));

  bits<1> addr64 = !eq(addrKind, BUFAddrKind.Addr64);

  bits<1> has_vaddr = !ne(addrKind, BUFAddrKind.Offset);
}

class MUBUF_Load_Pseudo <string opName,
                         int addrKind,
                         ValueType vdata_vt,
                         bit HasTiedDest = 0,
                         bit isLds = 0,
                         list<dag> pattern=[],
                         // Workaround bug bz30254
                         int addrKindCopy = addrKind,
                         RegisterClass vdata_rc = getVregSrcForVT<vdata_vt>.ret,
                         RegisterOperand vdata_op = getLdStRegisterOperand<vdata_rc>.ret>
  : MUBUF_Pseudo<opName,
                 (outs vdata_op:$vdata),
                 !con(getMUBUFIns<addrKindCopy, [], isLds>.ret,
                      !if(HasTiedDest, (ins vdata_op:$vdata_in), (ins))),
                 " $vdata, " # getMUBUFAsmOps<addrKindCopy>.ret # "$cpol" #
                   !if(isLds, " lds", "$tfe") # "$swz",
                 pattern>,
    MUBUF_SetupAddr<addrKindCopy> {
  let PseudoInstr = opName # !if(isLds, "_lds", "") #
                    "_" # getAddrName<addrKindCopy>.ret;
  let AsmMatchConverter = !if(isLds, "cvtMubufLds", "cvtMubuf");

  let Constraints = !if(HasTiedDest, "$vdata = $vdata_in", "");
  let mayLoad = 1;
  let mayStore = 0;
  let maybeAtomic = 1;
  let Uses = !if(isLds, [EXEC, M0], [EXEC]);
  let has_tfe = !not(isLds);
  let lds = isLds;
  let elements = getMUBUFElements<vdata_vt>.ret;
}

class MUBUF_Offset_Load_Pat <Instruction inst, ValueType load_vt = i32, SDPatternOperator ld = null_frag> : Pat <
  (load_vt (ld (MUBUFOffset v4i32:$srsrc, i32:$soffset, i16:$offset))),
  (load_vt (inst v4i32:$srsrc, i32:$soffset, i16:$offset))
>;

class MUBUF_Addr64_Load_Pat <Instruction inst,
                            ValueType load_vt = i32,
                            SDPatternOperator ld = null_frag> : Pat <
  (load_vt (ld (MUBUFAddr64 v4i32:$srsrc, i64:$vaddr, i32:$soffset, i16:$offset))),
  (load_vt (inst i64:$vaddr, v4i32:$srsrc, i32:$soffset, i16:$offset))
>;

multiclass MUBUF_Pseudo_Load_Pats<string BaseInst, ValueType load_vt = i32, SDPatternOperator ld = null_frag> {
  def : MUBUF_Offset_Load_Pat<!cast<Instruction>(BaseInst#"_OFFSET"), load_vt, ld>;
  def : MUBUF_Addr64_Load_Pat<!cast<Instruction>(BaseInst#"_ADDR64"), load_vt, ld>;
}


// FIXME: tfe can't be an operand because it requires a separate
// opcode because it needs an N+1 register class dest register.
multiclass MUBUF_Pseudo_Loads<string opName,
                              ValueType load_vt = i32,
                              SDPatternOperator ld = null_frag,
                              bit TiedDest = 0,
                              bit isLds = 0> {

  defvar legal_load_vt = !if(!eq(!cast<string>(load_vt), !cast<string>(v3f16)), v4f16, load_vt);

  def _OFFSET : MUBUF_Load_Pseudo <opName, BUFAddrKind.Offset, legal_load_vt, TiedDest, isLds>,
    MUBUFAddr64Table<0, NAME # !if(isLds, "_LDS", "")>;

  def _ADDR64 : MUBUF_Load_Pseudo <opName, BUFAddrKind.Addr64, legal_load_vt, TiedDest, isLds>,
    MUBUFAddr64Table<1, NAME # !if(isLds, "_LDS", "")>;

  def _OFFEN  : MUBUF_Load_Pseudo <opName, BUFAddrKind.OffEn, legal_load_vt, TiedDest, isLds>;
  def _IDXEN  : MUBUF_Load_Pseudo <opName, BUFAddrKind.IdxEn, legal_load_vt, TiedDest, isLds>;
  def _BOTHEN : MUBUF_Load_Pseudo <opName, BUFAddrKind.BothEn, legal_load_vt, TiedDest, isLds>;

  let DisableWQM = 1 in {
    def _OFFSET_exact : MUBUF_Load_Pseudo <opName, BUFAddrKind.Offset, legal_load_vt, TiedDest, isLds>;
    def _OFFEN_exact  : MUBUF_Load_Pseudo <opName, BUFAddrKind.OffEn, legal_load_vt, TiedDest, isLds>;
    def _IDXEN_exact  : MUBUF_Load_Pseudo <opName, BUFAddrKind.IdxEn, legal_load_vt, TiedDest, isLds>;
    def _BOTHEN_exact : MUBUF_Load_Pseudo <opName, BUFAddrKind.BothEn, legal_load_vt, TiedDest, isLds>;
  }
}

multiclass MUBUF_Pseudo_Loads_Lds<string opName, ValueType load_vt = i32,
                                  SDPatternOperator ld_nolds = null_frag,
                                  SDPatternOperator ld_lds = null_frag> {
  defm NAME : MUBUF_Pseudo_Loads<opName, load_vt, ld_nolds>;
  defm _LDS : MUBUF_Pseudo_Loads<opName, load_vt, ld_lds, 0, 1>;
}

class MUBUF_Store_Pseudo <string opName,
                          int addrKind,
                          ValueType store_vt,
                          list<dag> pattern=[],
                          // Workaround bug bz30254
                          int addrKindCopy = addrKind>
  : MUBUF_Pseudo<opName,
                 (outs),
                 getMUBUFIns<addrKindCopy, [getVregSrcForVT<store_vt>.ret]>.ret,
                 " $vdata, " # getMUBUFAsmOps<addrKindCopy>.ret # "$cpol$tfe$swz",
                 pattern>,
    MUBUF_SetupAddr<addrKindCopy> {
  let PseudoInstr = opName # "_" # getAddrName<addrKindCopy>.ret;
  let mayLoad = 0;
  let mayStore = 1;
  let maybeAtomic = 1;
  let elements = getMUBUFElements<store_vt>.ret;
}

multiclass MUBUF_Pseudo_Stores<string opName,
                               ValueType store_vt = i32,
                               SDPatternOperator st = null_frag> {

  defvar legal_store_vt = !if(!eq(!cast<string>(store_vt), !cast<string>(v3f16)), v4f16, store_vt);

  def _OFFSET : MUBUF_Store_Pseudo <opName, BUFAddrKind.Offset, legal_store_vt,
    [(st legal_store_vt:$vdata, (MUBUFOffset v4i32:$srsrc, i32:$soffset,
                                             i16:$offset))]>,
    MUBUFAddr64Table<0, NAME>;

  def _ADDR64 : MUBUF_Store_Pseudo <opName, BUFAddrKind.Addr64, legal_store_vt,
    [(st legal_store_vt:$vdata, (MUBUFAddr64 v4i32:$srsrc, i64:$vaddr, i32:$soffset,
                                             i16:$offset))]>,
    MUBUFAddr64Table<1, NAME>;

  def _OFFEN  : MUBUF_Store_Pseudo <opName, BUFAddrKind.OffEn, legal_store_vt>;
  def _IDXEN  : MUBUF_Store_Pseudo <opName, BUFAddrKind.IdxEn, legal_store_vt>;
  def _BOTHEN : MUBUF_Store_Pseudo <opName, BUFAddrKind.BothEn, legal_store_vt>;

  let DisableWQM = 1 in {
    def _OFFSET_exact : MUBUF_Store_Pseudo <opName, BUFAddrKind.Offset, legal_store_vt>;
    def _OFFEN_exact  : MUBUF_Store_Pseudo <opName, BUFAddrKind.OffEn, legal_store_vt>;
    def _IDXEN_exact  : MUBUF_Store_Pseudo <opName, BUFAddrKind.IdxEn, legal_store_vt>;
    def _BOTHEN_exact : MUBUF_Store_Pseudo <opName, BUFAddrKind.BothEn, legal_store_vt>;
  }
}

class MUBUF_Pseudo_Store_Lds<string opName>
  : MUBUF_Pseudo<opName,
                 (outs),
                 (ins SReg_128:$srsrc, SCSrc_b32:$soffset, offset:$offset, CPol:$cpol, SWZ:$swz),
                 " $srsrc, $soffset$offset lds$cpol$swz"> {
  let mayLoad = 0;
  let mayStore = 1;
  let maybeAtomic = 1;

  let has_vdata = 0;
  let has_vaddr = 0;
  let has_tfe = 0;
  let lds = 1;

  let Uses = [EXEC, M0];
  let AsmMatchConverter = "cvtMubufLds";
}

class getMUBUFAtomicInsDA<RegisterClass vdataClass, bit vdata_in,
                          list<RegisterClass> vaddrList=[]> {
  RegisterClass vaddrClass = !if(!empty(vaddrList), ?, !head(vaddrList));
  RegisterOperand vdata_op = getLdStRegisterOperand<vdataClass>.ret;
  dag ret = !if(vdata_in,
    !if(!empty(vaddrList),
      (ins vdata_op:$vdata_in,
           SReg_128:$srsrc, SCSrc_b32:$soffset, offset:$offset, CPol_GLC1:$cpol),
      (ins vdata_op:$vdata_in, vaddrClass:$vaddr,
           SReg_128:$srsrc, SCSrc_b32:$soffset, offset:$offset, CPol_GLC1:$cpol)
    ),
    !if(!empty(vaddrList),
      (ins vdata_op:$vdata,
           SReg_128:$srsrc, SCSrc_b32:$soffset, offset:$offset, CPol_0:$cpol),
      (ins vdata_op:$vdata, vaddrClass:$vaddr,
           SReg_128:$srsrc, SCSrc_b32:$soffset, offset:$offset, CPol_0:$cpol)
  ));
}

class getMUBUFAtomicIns<int addrKind,
                        RegisterClass vdataClass,
                        bit vdata_in,
                        // Workaround bug bz30254
                        RegisterClass vdataClassCopy=vdataClass> {
  dag ret =
    !if(!eq(addrKind, BUFAddrKind.Offset),
            getMUBUFAtomicInsDA<vdataClassCopy, vdata_in>.ret,
    !if(!eq(addrKind, BUFAddrKind.OffEn),
            getMUBUFAtomicInsDA<vdataClassCopy, vdata_in, [VGPR_32]>.ret,
    !if(!eq(addrKind, BUFAddrKind.IdxEn),
            getMUBUFAtomicInsDA<vdataClassCopy, vdata_in, [VGPR_32]>.ret,
    !if(!eq(addrKind, BUFAddrKind.BothEn),
            getMUBUFAtomicInsDA<vdataClassCopy, vdata_in, [VReg_64]>.ret,
    !if(!eq(addrKind, BUFAddrKind.Addr64),
            getMUBUFAtomicInsDA<vdataClassCopy, vdata_in, [VReg_64]>.ret,
    (ins))))));
}

class MUBUF_Atomic_Pseudo<string opName,
                          int addrKind,
                          dag outs,
                          dag ins,
                          string asmOps,
                          list<dag> pattern=[],
                          // Workaround bug bz30254
                          int addrKindCopy = addrKind>
  : MUBUF_Pseudo<opName, outs, ins, asmOps, pattern>,
    MUBUF_SetupAddr<addrKindCopy> {
  let mayStore = 1;
  let mayLoad = 1;
  let hasPostISelHook = 1;
  let hasSideEffects = 1;
  let DisableWQM = 1;
  let has_glc = 0;
  let has_dlc = 0;
  let has_tfe = 0;
  let has_sccb = 1;
  let maybeAtomic = 1;
  let AsmMatchConverter = "cvtMubufAtomic";
}

class MUBUF_AtomicNoRet_Pseudo<string opName, int addrKind,
                               RegisterClass vdataClass,
                               list<dag> pattern=[],
                               // Workaround bug bz30254
                               int addrKindCopy = addrKind,
                               RegisterClass vdataClassCopy = vdataClass>
  : MUBUF_Atomic_Pseudo<opName, addrKindCopy,
                        (outs),
                        getMUBUFAtomicIns<addrKindCopy, vdataClassCopy, 0>.ret,
                        " $vdata, " # getMUBUFAsmOps<addrKindCopy>.ret # "$cpol",
                        pattern>,
    AtomicNoRet<opName # "_" # getAddrName<addrKindCopy>.ret, 0> {
  let PseudoInstr = opName # "_" # getAddrName<addrKindCopy>.ret;
  let glc_value = 0;
  let dlc_value = 0;
  let sccb_value = 0;
  let IsAtomicNoRet = 1;
}

class MUBUF_AtomicRet_Pseudo<string opName, int addrKind,
                             RegisterClass vdataClass,
                             list<dag> pattern=[],
                             // Workaround bug bz30254
                             int addrKindCopy = addrKind,
                             RegisterClass vdataClassCopy = vdataClass,
                             RegisterOperand vdata_op = getLdStRegisterOperand<vdataClass>.ret>
  : MUBUF_Atomic_Pseudo<opName, addrKindCopy,
                        (outs vdata_op:$vdata),
                        getMUBUFAtomicIns<addrKindCopy, vdataClassCopy, 1>.ret,
                        " $vdata, " # getMUBUFAsmOps<addrKindCopy>.ret # "$cpol",
                        pattern>,
    AtomicNoRet<opName # "_" # getAddrName<addrKindCopy>.ret, 1> {
  let PseudoInstr = opName # "_rtn_" # getAddrName<addrKindCopy>.ret;
  let glc_value = 1;
  let dlc_value = 0;
  let sccb_value = 0;
  let IsAtomicRet = 1;
  let Constraints = "$vdata = $vdata_in";
  let DisableEncoding = "$vdata_in";
}

multiclass MUBUF_Pseudo_Atomics_NO_RTN <string opName,
                                        RegisterClass vdataClass,
                                        ValueType vdataType,
                                        SDPatternOperator atomic,
                                        bit isFP = isFloatType<vdataType>.ret> {
  let FPAtomic = isFP in
  def _OFFSET : MUBUF_AtomicNoRet_Pseudo <opName, BUFAddrKind.Offset, vdataClass>,
                MUBUFAddr64Table <0, NAME>;

  let FPAtomic = isFP in
  def _ADDR64 : MUBUF_AtomicNoRet_Pseudo <opName, BUFAddrKind.Addr64, vdataClass>,
                MUBUFAddr64Table <1, NAME>;

  let FPAtomic = isFP in
  def _OFFEN  : MUBUF_AtomicNoRet_Pseudo <opName, BUFAddrKind.OffEn,  vdataClass>;

  let FPAtomic = isFP in

  def _IDXEN  : MUBUF_AtomicNoRet_Pseudo <opName, BUFAddrKind.IdxEn,  vdataClass>;

  let FPAtomic = isFP in
  def _BOTHEN : MUBUF_AtomicNoRet_Pseudo <opName, BUFAddrKind.BothEn, vdataClass>;
}

multiclass MUBUF_Pseudo_Atomics_RTN <string opName,
                                     RegisterClass vdataClass,
                                     ValueType vdataType,
                                     SDPatternOperator atomic,
                                     bit isFP = isFloatType<vdataType>.ret> {
  let FPAtomic = isFP in
  def _OFFSET_RTN : MUBUF_AtomicRet_Pseudo <opName, BUFAddrKind.Offset, vdataClass,
    [(set vdataType:$vdata,
     (atomic (MUBUFOffset v4i32:$srsrc, i32:$soffset, i16:$offset),
             vdataType:$vdata_in))]>,
    MUBUFAddr64Table <0, NAME # "_RTN">;

  let FPAtomic = isFP in
  def _ADDR64_RTN : MUBUF_AtomicRet_Pseudo <opName, BUFAddrKind.Addr64, vdataClass,
    [(set vdataType:$vdata,
     (atomic (MUBUFAddr64 v4i32:$srsrc, i64:$vaddr, i32:$soffset, i16:$offset),
              vdataType:$vdata_in))]>,
    MUBUFAddr64Table <1, NAME # "_RTN">;

  let FPAtomic = isFP in
  def _OFFEN_RTN  : MUBUF_AtomicRet_Pseudo <opName, BUFAddrKind.OffEn,  vdataClass>;

  let FPAtomic = isFP in
  def _IDXEN_RTN  : MUBUF_AtomicRet_Pseudo <opName, BUFAddrKind.IdxEn,  vdataClass>;

  let FPAtomic = isFP in
  def _BOTHEN_RTN : MUBUF_AtomicRet_Pseudo <opName, BUFAddrKind.BothEn, vdataClass>;
}

multiclass MUBUF_Pseudo_Atomics <string opName,
                                 RegisterClass vdataClass,
                                 ValueType vdataType,
                                 SDPatternOperator atomic> :
  MUBUF_Pseudo_Atomics_NO_RTN<opName, vdataClass, vdataType, atomic>,
  MUBUF_Pseudo_Atomics_RTN<opName, vdataClass, vdataType, atomic>;


//===----------------------------------------------------------------------===//
// MUBUF Instructions
//===----------------------------------------------------------------------===//

defm BUFFER_LOAD_FORMAT_X : MUBUF_Pseudo_Loads_Lds <
  "buffer_load_format_x", f32
>;
defm BUFFER_LOAD_FORMAT_XY : MUBUF_Pseudo_Loads <
  "buffer_load_format_xy", v2f32
>;
defm BUFFER_LOAD_FORMAT_XYZ : MUBUF_Pseudo_Loads <
  "buffer_load_format_xyz", v3f32
>;
defm BUFFER_LOAD_FORMAT_XYZW : MUBUF_Pseudo_Loads <
  "buffer_load_format_xyzw", v4f32
>;
defm BUFFER_STORE_FORMAT_X : MUBUF_Pseudo_Stores <
  "buffer_store_format_x", f32
>;
defm BUFFER_STORE_FORMAT_XY : MUBUF_Pseudo_Stores <
  "buffer_store_format_xy", v2f32
>;
defm BUFFER_STORE_FORMAT_XYZ : MUBUF_Pseudo_Stores <
  "buffer_store_format_xyz", v3f32
>;
defm BUFFER_STORE_FORMAT_XYZW : MUBUF_Pseudo_Stores <
  "buffer_store_format_xyzw", v4f32
>;

let SubtargetPredicate = HasUnpackedD16VMem, D16Buf = 1 in {
  defm BUFFER_LOAD_FORMAT_D16_X_gfx80 : MUBUF_Pseudo_Loads <
    "buffer_load_format_d16_x", i32
  >;
  defm BUFFER_LOAD_FORMAT_D16_XY_gfx80 : MUBUF_Pseudo_Loads <
    "buffer_load_format_d16_xy", v2i32
  >;
  defm BUFFER_LOAD_FORMAT_D16_XYZ_gfx80 : MUBUF_Pseudo_Loads <
    "buffer_load_format_d16_xyz", v3i32
  >;
  defm BUFFER_LOAD_FORMAT_D16_XYZW_gfx80 : MUBUF_Pseudo_Loads <
   "buffer_load_format_d16_xyzw", v4i32
  >;
  defm BUFFER_STORE_FORMAT_D16_X_gfx80 : MUBUF_Pseudo_Stores <
    "buffer_store_format_d16_x", i32
  >;
  defm BUFFER_STORE_FORMAT_D16_XY_gfx80 : MUBUF_Pseudo_Stores <
    "buffer_store_format_d16_xy", v2i32
  >;
  defm BUFFER_STORE_FORMAT_D16_XYZ_gfx80 : MUBUF_Pseudo_Stores <
    "buffer_store_format_d16_xyz", v3i32
  >;
  defm BUFFER_STORE_FORMAT_D16_XYZW_gfx80 : MUBUF_Pseudo_Stores <
    "buffer_store_format_d16_xyzw", v4i32
  >;
} // End HasUnpackedD16VMem.

let SubtargetPredicate = HasPackedD16VMem, D16Buf = 1 in {
  defm BUFFER_LOAD_FORMAT_D16_X : MUBUF_Pseudo_Loads <
    "buffer_load_format_d16_x", f16
  >;
  defm BUFFER_LOAD_FORMAT_D16_XY : MUBUF_Pseudo_Loads <
    "buffer_load_format_d16_xy", v2f16
  >;
  defm BUFFER_LOAD_FORMAT_D16_XYZ : MUBUF_Pseudo_Loads <
    "buffer_load_format_d16_xyz", v3f16
  >;
  defm BUFFER_LOAD_FORMAT_D16_XYZW : MUBUF_Pseudo_Loads <
    "buffer_load_format_d16_xyzw", v4f16
  >;
  defm BUFFER_STORE_FORMAT_D16_X : MUBUF_Pseudo_Stores <
    "buffer_store_format_d16_x", f16
  >;
  defm BUFFER_STORE_FORMAT_D16_XY : MUBUF_Pseudo_Stores <
    "buffer_store_format_d16_xy", v2f16
  >;
  defm BUFFER_STORE_FORMAT_D16_XYZ : MUBUF_Pseudo_Stores <
    "buffer_store_format_d16_xyz", v3f16
  >;
  defm BUFFER_STORE_FORMAT_D16_XYZW : MUBUF_Pseudo_Stores <
    "buffer_store_format_d16_xyzw", v4f16
  >;
} // End HasPackedD16VMem.

defm BUFFER_LOAD_UBYTE : MUBUF_Pseudo_Loads_Lds <
  "buffer_load_ubyte", i32
>;
defm BUFFER_LOAD_SBYTE : MUBUF_Pseudo_Loads_Lds <
  "buffer_load_sbyte", i32
>;
defm BUFFER_LOAD_USHORT : MUBUF_Pseudo_Loads_Lds <
  "buffer_load_ushort", i32
>;
defm BUFFER_LOAD_SSHORT : MUBUF_Pseudo_Loads_Lds <
  "buffer_load_sshort", i32
>;
defm BUFFER_LOAD_DWORD : MUBUF_Pseudo_Loads_Lds <
  "buffer_load_dword", i32
>;
defm BUFFER_LOAD_DWORDX2 : MUBUF_Pseudo_Loads <
  "buffer_load_dwordx2", v2i32
>;
defm BUFFER_LOAD_DWORDX3 : MUBUF_Pseudo_Loads <
  "buffer_load_dwordx3", v3i32
>;
defm BUFFER_LOAD_DWORDX4 : MUBUF_Pseudo_Loads <
  "buffer_load_dwordx4", v4i32
>;

defm : MUBUF_Pseudo_Load_Pats<"BUFFER_LOAD_UBYTE", i32, extloadi8_global>;
defm : MUBUF_Pseudo_Load_Pats<"BUFFER_LOAD_UBYTE", i32, zextloadi8_global>;
defm : MUBUF_Pseudo_Load_Pats<"BUFFER_LOAD_SBYTE", i32, sextloadi8_global>;
defm : MUBUF_Pseudo_Load_Pats<"BUFFER_LOAD_USHORT", i32, extloadi16_global>;
defm : MUBUF_Pseudo_Load_Pats<"BUFFER_LOAD_USHORT", i32, zextloadi16_global>;
defm : MUBUF_Pseudo_Load_Pats<"BUFFER_LOAD_SSHORT", i32, sextloadi16_global>;
defm : MUBUF_Pseudo_Load_Pats<"BUFFER_LOAD_DWORD", i32, load_global>;
defm : MUBUF_Pseudo_Load_Pats<"BUFFER_LOAD_DWORDX2", v2i32, load_global>;
defm : MUBUF_Pseudo_Load_Pats<"BUFFER_LOAD_DWORDX3", v3i32, load_global>;
defm : MUBUF_Pseudo_Load_Pats<"BUFFER_LOAD_DWORDX4", v4i32, load_global>;

// This is not described in AMD documentation,
// but 'lds' versions of these opcodes are available
// in at least GFX8+ chips. See Bug 37653.
let SubtargetPredicate = isGFX8GFX9 in {
defm BUFFER_LOAD_DWORDX2_LDS : MUBUF_Pseudo_Loads <
  "buffer_load_dwordx2", v2i32, null_frag, 0, 1
>;
defm BUFFER_LOAD_DWORDX3_LDS : MUBUF_Pseudo_Loads <
  "buffer_load_dwordx3", v3i32, null_frag, 0, 1
>;
defm BUFFER_LOAD_DWORDX4_LDS : MUBUF_Pseudo_Loads <
  "buffer_load_dwordx4", v4i32, null_frag, 0, 1
>;
}

defm BUFFER_STORE_BYTE : MUBUF_Pseudo_Stores <
  "buffer_store_byte", i32, truncstorei8_global
>;
defm BUFFER_STORE_SHORT : MUBUF_Pseudo_Stores <
  "buffer_store_short", i32, truncstorei16_global
>;
defm BUFFER_STORE_DWORD : MUBUF_Pseudo_Stores <
  "buffer_store_dword", i32, store_global
>;
defm BUFFER_STORE_DWORDX2 : MUBUF_Pseudo_Stores <
  "buffer_store_dwordx2", v2i32, store_global
>;
defm BUFFER_STORE_DWORDX3 : MUBUF_Pseudo_Stores <
  "buffer_store_dwordx3", v3i32, store_global
>;
defm BUFFER_STORE_DWORDX4 : MUBUF_Pseudo_Stores <
  "buffer_store_dwordx4", v4i32, store_global
>;
defm BUFFER_ATOMIC_SWAP : MUBUF_Pseudo_Atomics <
  "buffer_atomic_swap", VGPR_32, i32, atomic_swap_global_32
>;
defm BUFFER_ATOMIC_CMPSWAP : MUBUF_Pseudo_Atomics <
  "buffer_atomic_cmpswap", VReg_64, v2i32, null_frag
>;
defm BUFFER_ATOMIC_ADD : MUBUF_Pseudo_Atomics <
  "buffer_atomic_add", VGPR_32, i32, atomic_load_add_global_32
>;
defm BUFFER_ATOMIC_SUB : MUBUF_Pseudo_Atomics <
  "buffer_atomic_sub", VGPR_32, i32, atomic_load_sub_global_32
>;
defm BUFFER_ATOMIC_SMIN : MUBUF_Pseudo_Atomics <
  "buffer_atomic_smin", VGPR_32, i32, atomic_load_min_global_32
>;
defm BUFFER_ATOMIC_UMIN : MUBUF_Pseudo_Atomics <
  "buffer_atomic_umin", VGPR_32, i32, atomic_load_umin_global_32
>;
defm BUFFER_ATOMIC_SMAX : MUBUF_Pseudo_Atomics <
  "buffer_atomic_smax", VGPR_32, i32, atomic_load_max_global_32
>;
defm BUFFER_ATOMIC_UMAX : MUBUF_Pseudo_Atomics <
  "buffer_atomic_umax", VGPR_32, i32, atomic_load_umax_global_32
>;
defm BUFFER_ATOMIC_AND : MUBUF_Pseudo_Atomics <
  "buffer_atomic_and", VGPR_32, i32, atomic_load_and_global_32
>;
defm BUFFER_ATOMIC_OR : MUBUF_Pseudo_Atomics <
  "buffer_atomic_or", VGPR_32, i32, atomic_load_or_global_32
>;
defm BUFFER_ATOMIC_XOR : MUBUF_Pseudo_Atomics <
  "buffer_atomic_xor", VGPR_32, i32, atomic_load_xor_global_32
>;
defm BUFFER_ATOMIC_INC : MUBUF_Pseudo_Atomics <
  "buffer_atomic_inc", VGPR_32, i32, atomic_inc_global_32
>;
defm BUFFER_ATOMIC_DEC : MUBUF_Pseudo_Atomics <
  "buffer_atomic_dec", VGPR_32, i32, atomic_dec_global_32
>;
defm BUFFER_ATOMIC_SWAP_X2 : MUBUF_Pseudo_Atomics <
  "buffer_atomic_swap_x2", VReg_64, i64, atomic_swap_global_64
>;
defm BUFFER_ATOMIC_CMPSWAP_X2 : MUBUF_Pseudo_Atomics <
  "buffer_atomic_cmpswap_x2", VReg_128, v2i64, null_frag
>;
defm BUFFER_ATOMIC_ADD_X2 : MUBUF_Pseudo_Atomics <
  "buffer_atomic_add_x2", VReg_64, i64, atomic_load_add_global_64
>;
defm BUFFER_ATOMIC_SUB_X2 : MUBUF_Pseudo_Atomics <
  "buffer_atomic_sub_x2", VReg_64, i64, atomic_load_sub_global_64
>;
defm BUFFER_ATOMIC_SMIN_X2 : MUBUF_Pseudo_Atomics <
  "buffer_atomic_smin_x2", VReg_64, i64, atomic_load_min_global_64
>;
defm BUFFER_ATOMIC_UMIN_X2 : MUBUF_Pseudo_Atomics <
  "buffer_atomic_umin_x2", VReg_64, i64, atomic_load_umin_global_64
>;
defm BUFFER_ATOMIC_SMAX_X2 : MUBUF_Pseudo_Atomics <
  "buffer_atomic_smax_x2", VReg_64, i64, atomic_load_max_global_64
>;
defm BUFFER_ATOMIC_UMAX_X2 : MUBUF_Pseudo_Atomics <
  "buffer_atomic_umax_x2", VReg_64, i64, atomic_load_umax_global_64
>;
defm BUFFER_ATOMIC_AND_X2 : MUBUF_Pseudo_Atomics <
  "buffer_atomic_and_x2", VReg_64, i64, atomic_load_and_global_64
>;
defm BUFFER_ATOMIC_OR_X2 : MUBUF_Pseudo_Atomics <
  "buffer_atomic_or_x2", VReg_64, i64, atomic_load_or_global_64
>;
defm BUFFER_ATOMIC_XOR_X2 : MUBUF_Pseudo_Atomics <
  "buffer_atomic_xor_x2", VReg_64, i64, atomic_load_xor_global_64
>;
defm BUFFER_ATOMIC_INC_X2 : MUBUF_Pseudo_Atomics <
  "buffer_atomic_inc_x2", VReg_64, i64, atomic_inc_global_64
>;
defm BUFFER_ATOMIC_DEC_X2 : MUBUF_Pseudo_Atomics <
  "buffer_atomic_dec_x2", VReg_64, i64, atomic_dec_global_64
>;

let SubtargetPredicate = HasGFX10_BEncoding in
defm BUFFER_ATOMIC_CSUB : MUBUF_Pseudo_Atomics_RTN <
  "buffer_atomic_csub", VGPR_32, i32, int_amdgcn_global_atomic_csub
>;

let SubtargetPredicate = isGFX8GFX9 in {
def BUFFER_STORE_LDS_DWORD : MUBUF_Pseudo_Store_Lds <"buffer_store_lds_dword">;
}

let SubtargetPredicate = isGFX6 in { // isn't on CI & VI
/*
defm BUFFER_ATOMIC_RSUB        : MUBUF_Pseudo_Atomics <"buffer_atomic_rsub">;
defm BUFFER_ATOMIC_RSUB_X2     : MUBUF_Pseudo_Atomics <"buffer_atomic_rsub_x2">;
*/

def BUFFER_WBINVL1_SC : MUBUF_Invalidate <"buffer_wbinvl1_sc",
                                          int_amdgcn_buffer_wbinvl1_sc>;
}

let SubtargetPredicate = isGFX6GFX7GFX10 in {

defm BUFFER_ATOMIC_FCMPSWAP : MUBUF_Pseudo_Atomics <
  "buffer_atomic_fcmpswap", VReg_64, v2f32, null_frag
>;
defm BUFFER_ATOMIC_FMIN : MUBUF_Pseudo_Atomics <
  "buffer_atomic_fmin", VGPR_32, f32, null_frag
>;
defm BUFFER_ATOMIC_FMAX : MUBUF_Pseudo_Atomics <
  "buffer_atomic_fmax", VGPR_32, f32, null_frag
>;
defm BUFFER_ATOMIC_FCMPSWAP_X2 : MUBUF_Pseudo_Atomics <
  "buffer_atomic_fcmpswap_x2", VReg_128, v2f64, null_frag
>;
defm BUFFER_ATOMIC_FMIN_X2 : MUBUF_Pseudo_Atomics <
  "buffer_atomic_fmin_x2", VReg_64, f64, null_frag
>;
defm BUFFER_ATOMIC_FMAX_X2 : MUBUF_Pseudo_Atomics <
  "buffer_atomic_fmax_x2", VReg_64, f64, null_frag
>;

}

let SubtargetPredicate = HasD16LoadStore in {

defm BUFFER_LOAD_UBYTE_D16 : MUBUF_Pseudo_Loads <
  "buffer_load_ubyte_d16", i32, null_frag, 1
>;

defm BUFFER_LOAD_UBYTE_D16_HI : MUBUF_Pseudo_Loads <
  "buffer_load_ubyte_d16_hi", i32, null_frag, 1
>;

defm BUFFER_LOAD_SBYTE_D16 : MUBUF_Pseudo_Loads <
  "buffer_load_sbyte_d16", i32, null_frag, 1
>;

defm BUFFER_LOAD_SBYTE_D16_HI : MUBUF_Pseudo_Loads <
  "buffer_load_sbyte_d16_hi", i32, null_frag, 1
>;

defm BUFFER_LOAD_SHORT_D16 : MUBUF_Pseudo_Loads <
  "buffer_load_short_d16", i32, null_frag, 1
>;

defm BUFFER_LOAD_SHORT_D16_HI : MUBUF_Pseudo_Loads <
  "buffer_load_short_d16_hi", i32, null_frag, 1
>;

defm BUFFER_STORE_BYTE_D16_HI : MUBUF_Pseudo_Stores <
  "buffer_store_byte_d16_hi", i32
>;

defm BUFFER_STORE_SHORT_D16_HI : MUBUF_Pseudo_Stores <
  "buffer_store_short_d16_hi", i32
>;

defm BUFFER_LOAD_FORMAT_D16_HI_X : MUBUF_Pseudo_Loads <
  "buffer_load_format_d16_hi_x", i32
>;
defm BUFFER_STORE_FORMAT_D16_HI_X : MUBUF_Pseudo_Stores <
  "buffer_store_format_d16_hi_x", i32
>;

} // End HasD16LoadStore

def BUFFER_WBINVL1 : MUBUF_Invalidate <"buffer_wbinvl1",
                                       int_amdgcn_buffer_wbinvl1>;

let SubtargetPredicate = HasAtomicFaddInsts in {
defm BUFFER_ATOMIC_ADD_F32 : MUBUF_Pseudo_Atomics_NO_RTN <
  "buffer_atomic_add_f32", VGPR_32, f32, atomic_load_fadd_global_noret_32
>;
defm BUFFER_ATOMIC_PK_ADD_F16 : MUBUF_Pseudo_Atomics_NO_RTN <
  "buffer_atomic_pk_add_f16", VGPR_32, v2f16, atomic_load_fadd_v2f16_global_noret_32
>;

let OtherPredicates = [isGFX90APlus] in {
defm BUFFER_ATOMIC_ADD_F32 : MUBUF_Pseudo_Atomics_RTN <
  "buffer_atomic_add_f32", VGPR_32, f32, atomic_load_fadd_global_32
>;
defm BUFFER_ATOMIC_PK_ADD_F16 : MUBUF_Pseudo_Atomics_RTN <
  "buffer_atomic_pk_add_f16", VGPR_32, v2f16, atomic_load_fadd_v2f16_global_32
>;
}
} // End SubtargetPredicate = HasAtomicFaddInsts

//===----------------------------------------------------------------------===//
// MTBUF Instructions
//===----------------------------------------------------------------------===//

defm TBUFFER_LOAD_FORMAT_X     : MTBUF_Pseudo_Loads  <"tbuffer_load_format_x",     VGPR_32,  1>;
defm TBUFFER_LOAD_FORMAT_XY    : MTBUF_Pseudo_Loads  <"tbuffer_load_format_xy",    VReg_64,  2>;
defm TBUFFER_LOAD_FORMAT_XYZ   : MTBUF_Pseudo_Loads  <"tbuffer_load_format_xyz",   VReg_96,  3>;
defm TBUFFER_LOAD_FORMAT_XYZW  : MTBUF_Pseudo_Loads  <"tbuffer_load_format_xyzw",  VReg_128, 4>;
defm TBUFFER_STORE_FORMAT_X    : MTBUF_Pseudo_Stores <"tbuffer_store_format_x",    VGPR_32,  1>;
defm TBUFFER_STORE_FORMAT_XY   : MTBUF_Pseudo_Stores <"tbuffer_store_format_xy",   VReg_64,  2>;
defm TBUFFER_STORE_FORMAT_XYZ  : MTBUF_Pseudo_Stores <"tbuffer_store_format_xyz",  VReg_96,  3>;
defm TBUFFER_STORE_FORMAT_XYZW : MTBUF_Pseudo_Stores <"tbuffer_store_format_xyzw", VReg_128, 4>;

let SubtargetPredicate = HasUnpackedD16VMem, D16Buf = 1 in {
  defm TBUFFER_LOAD_FORMAT_D16_X_gfx80     : MTBUF_Pseudo_Loads  <"tbuffer_load_format_d16_x",     VGPR_32,  1>;
  defm TBUFFER_LOAD_FORMAT_D16_XY_gfx80    : MTBUF_Pseudo_Loads  <"tbuffer_load_format_d16_xy",    VReg_64,  2>;
  defm TBUFFER_LOAD_FORMAT_D16_XYZ_gfx80   : MTBUF_Pseudo_Loads  <"tbuffer_load_format_d16_xyz",   VReg_96,  3>;
  defm TBUFFER_LOAD_FORMAT_D16_XYZW_gfx80  : MTBUF_Pseudo_Loads  <"tbuffer_load_format_d16_xyzw",  VReg_128, 4>;
  defm TBUFFER_STORE_FORMAT_D16_X_gfx80    : MTBUF_Pseudo_Stores <"tbuffer_store_format_d16_x",    VGPR_32,  1>;
  defm TBUFFER_STORE_FORMAT_D16_XY_gfx80   : MTBUF_Pseudo_Stores <"tbuffer_store_format_d16_xy",   VReg_64,  2>;
  defm TBUFFER_STORE_FORMAT_D16_XYZ_gfx80  : MTBUF_Pseudo_Stores <"tbuffer_store_format_d16_xyz",  VReg_96,  3>;
  defm TBUFFER_STORE_FORMAT_D16_XYZW_gfx80 : MTBUF_Pseudo_Stores <"tbuffer_store_format_d16_xyzw", VReg_128, 4>;
} // End HasUnpackedD16VMem.

let SubtargetPredicate = HasPackedD16VMem, D16Buf = 1 in {
  defm TBUFFER_LOAD_FORMAT_D16_X     : MTBUF_Pseudo_Loads  <"tbuffer_load_format_d16_x",     VGPR_32, 1>;
  defm TBUFFER_LOAD_FORMAT_D16_XY    : MTBUF_Pseudo_Loads  <"tbuffer_load_format_d16_xy",    VGPR_32, 2>;
  defm TBUFFER_LOAD_FORMAT_D16_XYZ   : MTBUF_Pseudo_Loads  <"tbuffer_load_format_d16_xyz",   VReg_64, 3>;
  defm TBUFFER_LOAD_FORMAT_D16_XYZW  : MTBUF_Pseudo_Loads  <"tbuffer_load_format_d16_xyzw",  VReg_64, 4>;
  defm TBUFFER_STORE_FORMAT_D16_X    : MTBUF_Pseudo_Stores <"tbuffer_store_format_d16_x",    VGPR_32, 1>;
  defm TBUFFER_STORE_FORMAT_D16_XY   : MTBUF_Pseudo_Stores <"tbuffer_store_format_d16_xy",   VGPR_32, 2>;
  defm TBUFFER_STORE_FORMAT_D16_XYZ  : MTBUF_Pseudo_Stores <"tbuffer_store_format_d16_xyz",  VReg_64, 3>;
  defm TBUFFER_STORE_FORMAT_D16_XYZW : MTBUF_Pseudo_Stores <"tbuffer_store_format_d16_xyzw", VReg_64, 4>;
} // End HasPackedD16VMem.

let SubtargetPredicate = isGFX7Plus in {

//===----------------------------------------------------------------------===//
// Instruction definitions for CI and newer.
//===----------------------------------------------------------------------===//

def BUFFER_WBINVL1_VOL : MUBUF_Invalidate <"buffer_wbinvl1_vol",
                                           int_amdgcn_buffer_wbinvl1_vol>;

} // End let SubtargetPredicate = isGFX7Plus

let SubtargetPredicate = isGFX90APlus in {
  def BUFFER_WBL2  : MUBUF_Invalidate<"buffer_wbl2"> {
  }
  def BUFFER_INVL2 : MUBUF_Invalidate<"buffer_invl2"> {
  }

  defm BUFFER_ATOMIC_ADD_F64 : MUBUF_Pseudo_Atomics<"buffer_atomic_add_f64", VReg_64, f64, int_amdgcn_global_atomic_fadd>;
  defm BUFFER_ATOMIC_MIN_F64 : MUBUF_Pseudo_Atomics<"buffer_atomic_min_f64", VReg_64, f64, int_amdgcn_global_atomic_fmin>;
  defm BUFFER_ATOMIC_MAX_F64 : MUBUF_Pseudo_Atomics<"buffer_atomic_max_f64", VReg_64, f64, int_amdgcn_global_atomic_fmax>;
} // End SubtargetPredicate = isGFX90APlus

let SubtargetPredicate = isGFX10Plus in {
  def BUFFER_GL0_INV : MUBUF_Invalidate<"buffer_gl0_inv">;
  def BUFFER_GL1_INV : MUBUF_Invalidate<"buffer_gl1_inv">;
} // End SubtargetPredicate = isGFX10Plus

//===----------------------------------------------------------------------===//
// MUBUF Patterns
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// buffer_load/store_format patterns
//===----------------------------------------------------------------------===//

multiclass MUBUF_LoadIntrinsicPat<SDPatternOperator name, ValueType vt,
                                  string opcode, ValueType memoryVt = vt> {
  defvar st = !if(!eq(!cast<string>(memoryVt), !cast<string>(vt)), name, mubuf_intrinsic_load<name, memoryVt>);

  def : GCNPat<
    (vt (st v4i32:$rsrc, 0, 0, i32:$soffset, timm:$offset,
              timm:$auxiliary, 0)),
    (!cast<MUBUF_Pseudo>(opcode # _OFFSET) SReg_128:$rsrc, SCSrc_b32:$soffset, (as_i16timm $offset),
      (extract_cpol $auxiliary), 0, (extract_swz $auxiliary))
  >;

  def : GCNPat<
    (vt (st v4i32:$rsrc, 0, i32:$voffset, i32:$soffset, timm:$offset,
              timm:$auxiliary, 0)),
    (!cast<MUBUF_Pseudo>(opcode # _OFFEN) VGPR_32:$voffset, SReg_128:$rsrc, SCSrc_b32:$soffset, (as_i16timm $offset),
      (extract_cpol $auxiliary), 0, (extract_swz $auxiliary))
  >;

  def : GCNPat<
    (vt (st v4i32:$rsrc, i32:$vindex, 0, i32:$soffset, timm:$offset,
              timm:$auxiliary, timm)),
    (!cast<MUBUF_Pseudo>(opcode # _IDXEN) VGPR_32:$vindex, SReg_128:$rsrc, SCSrc_b32:$soffset, (as_i16timm $offset),
      (extract_cpol $auxiliary), 0, (extract_swz $auxiliary))
  >;

  def : GCNPat<
    (vt (st v4i32:$rsrc, i32:$vindex, i32:$voffset, i32:$soffset, timm:$offset,
              timm:$auxiliary, timm)),
    (!cast<MUBUF_Pseudo>(opcode # _BOTHEN)
      (REG_SEQUENCE VReg_64, VGPR_32:$vindex, sub0, VGPR_32:$voffset, sub1),
      SReg_128:$rsrc, SCSrc_b32:$soffset, (as_i16timm $offset),
      (extract_cpol $auxiliary), 0, (extract_swz $auxiliary))
  >;
}

defm : MUBUF_LoadIntrinsicPat<SIbuffer_load_format, f32, "BUFFER_LOAD_FORMAT_X">;
defm : MUBUF_LoadIntrinsicPat<SIbuffer_load_format, i32, "BUFFER_LOAD_FORMAT_X">;
defm : MUBUF_LoadIntrinsicPat<SIbuffer_load_format, v2f32, "BUFFER_LOAD_FORMAT_XY">;
defm : MUBUF_LoadIntrinsicPat<SIbuffer_load_format, v2i32, "BUFFER_LOAD_FORMAT_XY">;
defm : MUBUF_LoadIntrinsicPat<SIbuffer_load_format, v3f32, "BUFFER_LOAD_FORMAT_XYZ">;
defm : MUBUF_LoadIntrinsicPat<SIbuffer_load_format, v3i32, "BUFFER_LOAD_FORMAT_XYZ">;
defm : MUBUF_LoadIntrinsicPat<SIbuffer_load_format, v4f32, "BUFFER_LOAD_FORMAT_XYZW">;
defm : MUBUF_LoadIntrinsicPat<SIbuffer_load_format, v4i32, "BUFFER_LOAD_FORMAT_XYZW">;

let SubtargetPredicate = HasUnpackedD16VMem in {
  defm : MUBUF_LoadIntrinsicPat<SIbuffer_load_format_d16, f16, "BUFFER_LOAD_FORMAT_D16_X_gfx80">;
  defm : MUBUF_LoadIntrinsicPat<SIbuffer_load_format_d16, i16, "BUFFER_LOAD_FORMAT_D16_X_gfx80">;
  defm : MUBUF_LoadIntrinsicPat<SIbuffer_load_format_d16, i32, "BUFFER_LOAD_FORMAT_D16_X_gfx80">;
  defm : MUBUF_LoadIntrinsicPat<SIbuffer_load_format_d16, v2i32, "BUFFER_LOAD_FORMAT_D16_XY_gfx80">;
  defm : MUBUF_LoadIntrinsicPat<SIbuffer_load_format_d16, v3i32, "BUFFER_LOAD_FORMAT_D16_XYZ_gfx80">;
  defm : MUBUF_LoadIntrinsicPat<SIbuffer_load_format_d16, v4i32, "BUFFER_LOAD_FORMAT_D16_XYZW_gfx80">;
} // End HasUnpackedD16VMem.

let SubtargetPredicate = HasPackedD16VMem in {
  defm : MUBUF_LoadIntrinsicPat<SIbuffer_load_format_d16, f16, "BUFFER_LOAD_FORMAT_D16_X">;
  defm : MUBUF_LoadIntrinsicPat<SIbuffer_load_format_d16, i16, "BUFFER_LOAD_FORMAT_D16_X">;
  defm : MUBUF_LoadIntrinsicPat<SIbuffer_load_format_d16, i32, "BUFFER_LOAD_FORMAT_D16_X">;
  defm : MUBUF_LoadIntrinsicPat<SIbuffer_load_format_d16, v2f16, "BUFFER_LOAD_FORMAT_D16_XY">;
  defm : MUBUF_LoadIntrinsicPat<SIbuffer_load_format_d16, v2i16, "BUFFER_LOAD_FORMAT_D16_XY">;
  defm : MUBUF_LoadIntrinsicPat<SIbuffer_load_format_d16, v4f16, "BUFFER_LOAD_FORMAT_D16_XYZ", v3f16>;
  defm : MUBUF_LoadIntrinsicPat<SIbuffer_load_format_d16, v4i16, "BUFFER_LOAD_FORMAT_D16_XYZ", v3i16>;
  defm : MUBUF_LoadIntrinsicPat<SIbuffer_load_format_d16, v4f16, "BUFFER_LOAD_FORMAT_D16_XYZW">;
  defm : MUBUF_LoadIntrinsicPat<SIbuffer_load_format_d16, v4i16, "BUFFER_LOAD_FORMAT_D16_XYZW">;
} // End HasPackedD16VMem.

defm : MUBUF_LoadIntrinsicPat<SIbuffer_load, f32, "BUFFER_LOAD_DWORD">;
defm : MUBUF_LoadIntrinsicPat<SIbuffer_load, i32, "BUFFER_LOAD_DWORD">;
defm : MUBUF_LoadIntrinsicPat<SIbuffer_load, v2i16, "BUFFER_LOAD_DWORD">;
defm : MUBUF_LoadIntrinsicPat<SIbuffer_load, v2f16, "BUFFER_LOAD_DWORD">;
defm : MUBUF_LoadIntrinsicPat<SIbuffer_load, v2f32, "BUFFER_LOAD_DWORDX2">;
defm : MUBUF_LoadIntrinsicPat<SIbuffer_load, v2i32, "BUFFER_LOAD_DWORDX2">;
defm : MUBUF_LoadIntrinsicPat<SIbuffer_load, v4i16, "BUFFER_LOAD_DWORDX2">;
defm : MUBUF_LoadIntrinsicPat<SIbuffer_load, v4f16, "BUFFER_LOAD_DWORDX2">;
defm : MUBUF_LoadIntrinsicPat<SIbuffer_load, v3f32, "BUFFER_LOAD_DWORDX3">;
defm : MUBUF_LoadIntrinsicPat<SIbuffer_load, v3i32, "BUFFER_LOAD_DWORDX3">;
defm : MUBUF_LoadIntrinsicPat<SIbuffer_load, v4f32, "BUFFER_LOAD_DWORDX4">;
defm : MUBUF_LoadIntrinsicPat<SIbuffer_load, v4i32, "BUFFER_LOAD_DWORDX4">;
defm : MUBUF_LoadIntrinsicPat<SIbuffer_load_byte, i32, "BUFFER_LOAD_SBYTE">;
defm : MUBUF_LoadIntrinsicPat<SIbuffer_load_short, i32, "BUFFER_LOAD_SSHORT">;
defm : MUBUF_LoadIntrinsicPat<SIbuffer_load_ubyte, i32, "BUFFER_LOAD_UBYTE">;
defm : MUBUF_LoadIntrinsicPat<SIbuffer_load_ushort,  i32, "BUFFER_LOAD_USHORT">;

multiclass MUBUF_StoreIntrinsicPat<SDPatternOperator name, ValueType vt,
                                   string opcode, ValueType memoryVt = vt> {
  defvar st = !if(!eq(!cast<string>(memoryVt), !cast<string>(vt)), name, mubuf_intrinsic_store<name, memoryVt>);

  def : GCNPat<
    (st vt:$vdata, v4i32:$rsrc, 0, 0, i32:$soffset, timm:$offset,
              timm:$auxiliary, 0),
    (!cast<MUBUF_Pseudo>(opcode # _OFFSET_exact) getVregSrcForVT<vt>.ret:$vdata, SReg_128:$rsrc, SCSrc_b32:$soffset, (as_i16timm $offset),
      (extract_cpol $auxiliary), 0,  (extract_swz $auxiliary))
  >;

  def : GCNPat<
    (st vt:$vdata, v4i32:$rsrc, 0, i32:$voffset, i32:$soffset, timm:$offset,
              timm:$auxiliary, 0),
    (!cast<MUBUF_Pseudo>(opcode # _OFFEN_exact) getVregSrcForVT<vt>.ret:$vdata, VGPR_32:$voffset, SReg_128:$rsrc, SCSrc_b32:$soffset,
      (as_i16timm $offset), (extract_cpol $auxiliary), 0, (extract_swz $auxiliary))
  >;

  def : GCNPat<
    (st vt:$vdata, v4i32:$rsrc, i32:$vindex, 0, i32:$soffset, timm:$offset,
              timm:$auxiliary, timm),
    (!cast<MUBUF_Pseudo>(opcode # _IDXEN_exact) getVregSrcForVT<vt>.ret:$vdata, VGPR_32:$vindex, SReg_128:$rsrc, SCSrc_b32:$soffset,
      (as_i16timm $offset), (extract_cpol $auxiliary), 0,  (extract_swz $auxiliary))
  >;

  def : GCNPat<
    (st vt:$vdata, v4i32:$rsrc, i32:$vindex, i32:$voffset, i32:$soffset, timm:$offset,
              timm:$auxiliary, timm),
    (!cast<MUBUF_Pseudo>(opcode # _BOTHEN_exact)
      getVregSrcForVT<vt>.ret:$vdata,
      (REG_SEQUENCE VReg_64, VGPR_32:$vindex, sub0, VGPR_32:$voffset, sub1),
      SReg_128:$rsrc, SCSrc_b32:$soffset, (as_i16timm $offset), (extract_cpol $auxiliary),
      0,  (extract_swz $auxiliary))
  >;
}

defm : MUBUF_StoreIntrinsicPat<SIbuffer_store_format, f32, "BUFFER_STORE_FORMAT_X">;
defm : MUBUF_StoreIntrinsicPat<SIbuffer_store_format, i32, "BUFFER_STORE_FORMAT_X">;
defm : MUBUF_StoreIntrinsicPat<SIbuffer_store_format, v2f32, "BUFFER_STORE_FORMAT_XY">;
defm : MUBUF_StoreIntrinsicPat<SIbuffer_store_format, v2i32, "BUFFER_STORE_FORMAT_XY">;
defm : MUBUF_StoreIntrinsicPat<SIbuffer_store_format, v3f32, "BUFFER_STORE_FORMAT_XYZ">;
defm : MUBUF_StoreIntrinsicPat<SIbuffer_store_format, v3i32, "BUFFER_STORE_FORMAT_XYZ">;
defm : MUBUF_StoreIntrinsicPat<SIbuffer_store_format, v4f32, "BUFFER_STORE_FORMAT_XYZW">;
defm : MUBUF_StoreIntrinsicPat<SIbuffer_store_format, v4i32, "BUFFER_STORE_FORMAT_XYZW">;

let SubtargetPredicate = HasUnpackedD16VMem in {
  defm : MUBUF_StoreIntrinsicPat<SIbuffer_store_format_d16, f16, "BUFFER_STORE_FORMAT_D16_X_gfx80">;
  defm : MUBUF_StoreIntrinsicPat<SIbuffer_store_format_d16, i16, "BUFFER_STORE_FORMAT_D16_X_gfx80">;
  defm : MUBUF_StoreIntrinsicPat<SIbuffer_store_format_d16, i32, "BUFFER_STORE_FORMAT_D16_X_gfx80">;
  defm : MUBUF_StoreIntrinsicPat<SIbuffer_store_format_d16, v2i32, "BUFFER_STORE_FORMAT_D16_XY_gfx80">;
  defm : MUBUF_StoreIntrinsicPat<SIbuffer_store_format_d16, v3i32, "BUFFER_STORE_FORMAT_D16_XYZ_gfx80">;
  defm : MUBUF_StoreIntrinsicPat<SIbuffer_store_format_d16, v4i32, "BUFFER_STORE_FORMAT_D16_XYZW_gfx80">;
} // End HasUnpackedD16VMem.

let SubtargetPredicate = HasPackedD16VMem in {
  defm : MUBUF_StoreIntrinsicPat<SIbuffer_store_format_d16, f16, "BUFFER_STORE_FORMAT_D16_X">;
  defm : MUBUF_StoreIntrinsicPat<SIbuffer_store_format_d16, i16, "BUFFER_STORE_FORMAT_D16_X">;
  defm : MUBUF_StoreIntrinsicPat<SIbuffer_store_format_d16, i32, "BUFFER_STORE_FORMAT_D16_X">;
  defm : MUBUF_StoreIntrinsicPat<SIbuffer_store_format_d16, v2f16, "BUFFER_STORE_FORMAT_D16_XY">;
  defm : MUBUF_StoreIntrinsicPat<SIbuffer_store_format_d16, v2i16, "BUFFER_STORE_FORMAT_D16_XY">;
  defm : MUBUF_StoreIntrinsicPat<SIbuffer_store_format_d16, v4f16, "BUFFER_STORE_FORMAT_D16_XYZ", v3f16>;
  defm : MUBUF_StoreIntrinsicPat<SIbuffer_store_format_d16, v4i16, "BUFFER_STORE_FORMAT_D16_XYZ", v3i16>;
  defm : MUBUF_StoreIntrinsicPat<SIbuffer_store_format_d16, v4f16, "BUFFER_STORE_FORMAT_D16_XYZW">;
  defm : MUBUF_StoreIntrinsicPat<SIbuffer_store_format_d16, v4i16, "BUFFER_STORE_FORMAT_D16_XYZW">;
} // End HasPackedD16VMem.

defm : MUBUF_StoreIntrinsicPat<SIbuffer_store, f32, "BUFFER_STORE_DWORD">;
defm : MUBUF_StoreIntrinsicPat<SIbuffer_store, i32, "BUFFER_STORE_DWORD">;
defm : MUBUF_StoreIntrinsicPat<SIbuffer_store, v2i16, "BUFFER_STORE_DWORD">;
defm : MUBUF_StoreIntrinsicPat<SIbuffer_store, v2f16, "BUFFER_STORE_DWORD">;
defm : MUBUF_StoreIntrinsicPat<SIbuffer_store, v2f32, "BUFFER_STORE_DWORDX2">;
defm : MUBUF_StoreIntrinsicPat<SIbuffer_store, v2i32, "BUFFER_STORE_DWORDX2">;
defm : MUBUF_StoreIntrinsicPat<SIbuffer_store, v4i16, "BUFFER_STORE_DWORDX2">;
defm : MUBUF_StoreIntrinsicPat<SIbuffer_store, v4f16, "BUFFER_STORE_DWORDX2">;
defm : MUBUF_StoreIntrinsicPat<SIbuffer_store, v3f32, "BUFFER_STORE_DWORDX3">;
defm : MUBUF_StoreIntrinsicPat<SIbuffer_store, v3i32, "BUFFER_STORE_DWORDX3">;
defm : MUBUF_StoreIntrinsicPat<SIbuffer_store, v4f32, "BUFFER_STORE_DWORDX4">;
defm : MUBUF_StoreIntrinsicPat<SIbuffer_store, v4i32, "BUFFER_STORE_DWORDX4">;
defm : MUBUF_StoreIntrinsicPat<SIbuffer_store_byte, i32, "BUFFER_STORE_BYTE">;
defm : MUBUF_StoreIntrinsicPat<SIbuffer_store_short, i32, "BUFFER_STORE_SHORT">;

//===----------------------------------------------------------------------===//
// buffer_atomic patterns
//===----------------------------------------------------------------------===//

multiclass BufferAtomicPatterns<SDPatternOperator name, ValueType vt,
                                string opcode> {
  def : GCNPat<
    (vt (name vt:$vdata_in, v4i32:$rsrc, 0, 0, i32:$soffset,
              timm:$offset, timm:$cachepolicy, 0)),
    (!cast<MUBUF_Pseudo>(opcode # _OFFSET_RTN)
      getVregSrcForVT<vt>.ret:$vdata_in, SReg_128:$rsrc, SCSrc_b32:$soffset,
      (as_i16timm $offset), (set_glc $cachepolicy))
  >;

  def : GCNPat<
    (vt (name vt:$vdata_in, v4i32:$rsrc, i32:$vindex, 0, i32:$soffset,
              timm:$offset, timm:$cachepolicy, timm)),
    (!cast<MUBUF_Pseudo>(opcode # _IDXEN_RTN) getVregSrcForVT<vt>.ret:$vdata_in,
      VGPR_32:$vindex, SReg_128:$rsrc, SCSrc_b32:$soffset,
      (as_i16timm $offset), (set_glc $cachepolicy))
  >;

  def : GCNPat<
    (vt (name vt:$vdata_in, v4i32:$rsrc, 0, i32:$voffset,
              i32:$soffset, timm:$offset, timm:$cachepolicy, 0)),
    (!cast<MUBUF_Pseudo>(opcode # _OFFEN_RTN) getVregSrcForVT<vt>.ret:$vdata_in,
      VGPR_32:$voffset, SReg_128:$rsrc, SCSrc_b32:$soffset,
      (as_i16timm $offset), (set_glc $cachepolicy))
  >;

  def : GCNPat<
    (vt (name vt:$vdata_in, v4i32:$rsrc, i32:$vindex, i32:$voffset,
              i32:$soffset, timm:$offset, timm:$cachepolicy, timm)),
    (!cast<MUBUF_Pseudo>(opcode # _BOTHEN_RTN)
      getVregSrcForVT<vt>.ret:$vdata_in,
      (REG_SEQUENCE VReg_64, VGPR_32:$vindex, sub0, VGPR_32:$voffset, sub1),
        SReg_128:$rsrc, SCSrc_b32:$soffset, (as_i16timm $offset),
        (set_glc $cachepolicy))
  >;
}

defm : BufferAtomicPatterns<SIbuffer_atomic_swap, i32, "BUFFER_ATOMIC_SWAP">;
defm : BufferAtomicPatterns<SIbuffer_atomic_swap, f32, "BUFFER_ATOMIC_SWAP">;
defm : BufferAtomicPatterns<SIbuffer_atomic_add, i32, "BUFFER_ATOMIC_ADD">;
defm : BufferAtomicPatterns<SIbuffer_atomic_sub, i32, "BUFFER_ATOMIC_SUB">;
defm : BufferAtomicPatterns<SIbuffer_atomic_smin, i32, "BUFFER_ATOMIC_SMIN">;
defm : BufferAtomicPatterns<SIbuffer_atomic_umin, i32, "BUFFER_ATOMIC_UMIN">;
defm : BufferAtomicPatterns<SIbuffer_atomic_smax, i32, "BUFFER_ATOMIC_SMAX">;
defm : BufferAtomicPatterns<SIbuffer_atomic_umax, i32, "BUFFER_ATOMIC_UMAX">;
defm : BufferAtomicPatterns<SIbuffer_atomic_and, i32, "BUFFER_ATOMIC_AND">;
defm : BufferAtomicPatterns<SIbuffer_atomic_or, i32, "BUFFER_ATOMIC_OR">;
defm : BufferAtomicPatterns<SIbuffer_atomic_xor, i32, "BUFFER_ATOMIC_XOR">;
defm : BufferAtomicPatterns<SIbuffer_atomic_inc, i32, "BUFFER_ATOMIC_INC">;
defm : BufferAtomicPatterns<SIbuffer_atomic_dec, i32, "BUFFER_ATOMIC_DEC">;
defm : BufferAtomicPatterns<SIbuffer_atomic_csub, i32, "BUFFER_ATOMIC_CSUB">;
defm : BufferAtomicPatterns<SIbuffer_atomic_swap, i64, "BUFFER_ATOMIC_SWAP_X2">;
defm : BufferAtomicPatterns<SIbuffer_atomic_add, i64,  "BUFFER_ATOMIC_ADD_X2">;
defm : BufferAtomicPatterns<SIbuffer_atomic_sub, i64, "BUFFER_ATOMIC_SUB_X2">;
defm : BufferAtomicPatterns<SIbuffer_atomic_smin, i64, "BUFFER_ATOMIC_SMIN_X2">;
defm : BufferAtomicPatterns<SIbuffer_atomic_umin, i64, "BUFFER_ATOMIC_UMIN_X2">;
defm : BufferAtomicPatterns<SIbuffer_atomic_smax, i64, "BUFFER_ATOMIC_SMAX_X2">;
defm : BufferAtomicPatterns<SIbuffer_atomic_umax, i64, "BUFFER_ATOMIC_UMAX_X2">;
defm : BufferAtomicPatterns<SIbuffer_atomic_and, i64, "BUFFER_ATOMIC_AND_X2">;
defm : BufferAtomicPatterns<SIbuffer_atomic_or, i64, "BUFFER_ATOMIC_OR_X2">;
defm : BufferAtomicPatterns<SIbuffer_atomic_xor, i64, "BUFFER_ATOMIC_XOR_X2">;
defm : BufferAtomicPatterns<SIbuffer_atomic_inc, i64, "BUFFER_ATOMIC_INC_X2">;
defm : BufferAtomicPatterns<SIbuffer_atomic_dec, i64, "BUFFER_ATOMIC_DEC_X2">;

class NoUseBufferAtomic<SDPatternOperator Op, ValueType vt> : PatFrag <
  (ops node:$src0, node:$src1, node:$src2, node:$src3, node:$src4, node:$src5, node:$src6, node:$src7),
  (vt (Op $src0, $src1, $src2, $src3, $src4, $src5, $src6, $src7)),
  [{ return SDValue(N, 0).use_empty(); }]> {

  let GISelPredicateCode = [{
    return MRI.use_nodbg_empty(MI.getOperand(0).getReg());
  }];
}

multiclass BufferAtomicPatterns_NO_RTN<SDPatternOperator name, ValueType vt,
                                       string opcode> {
  def : GCNPat<
    (NoUseBufferAtomic<name, vt> vt:$vdata_in, v4i32:$rsrc, 0,
                                 0, i32:$soffset, timm:$offset,
                                 timm:$cachepolicy, 0),
    (!cast<MUBUF_Pseudo>(opcode # _OFFSET) getVregSrcForVT<vt>.ret:$vdata_in, SReg_128:$rsrc, SCSrc_b32:$soffset,
                                          (as_i16timm $offset), $cachepolicy)
  >;

  def : GCNPat<
    (NoUseBufferAtomic<name, vt> vt:$vdata_in, v4i32:$rsrc, i32:$vindex,
                                 0, i32:$soffset, timm:$offset,
                                 timm:$cachepolicy, timm),
    (!cast<MUBUF_Pseudo>(opcode # _IDXEN) getVregSrcForVT<vt>.ret:$vdata_in, VGPR_32:$vindex, SReg_128:$rsrc, SCSrc_b32:$soffset,
                                          (as_i16timm $offset), $cachepolicy)
  >;

  def : GCNPat<
    (NoUseBufferAtomic<name, vt> vt:$vdata_in, v4i32:$rsrc, 0,
                                 i32:$voffset, i32:$soffset, timm:$offset,
                                 timm:$cachepolicy, 0),
    (!cast<MUBUF_Pseudo>(opcode # _OFFEN) getVregSrcForVT<vt>.ret:$vdata_in, VGPR_32:$voffset, SReg_128:$rsrc, SCSrc_b32:$soffset,
                                          (as_i16timm $offset), $cachepolicy)
  >;

  def : GCNPat<
    (NoUseBufferAtomic<name, vt> vt:$vdata_in, v4i32:$rsrc, i32:$vindex,
                                 i32:$voffset, i32:$soffset, timm:$offset,
                                 timm:$cachepolicy, timm),
    (!cast<MUBUF_Pseudo>(opcode # _BOTHEN)
      getVregSrcForVT<vt>.ret:$vdata_in,
      (REG_SEQUENCE VReg_64, VGPR_32:$vindex, sub0, VGPR_32:$voffset, sub1),
      SReg_128:$rsrc, SCSrc_b32:$soffset, (as_i16timm $offset), $cachepolicy)
  >;
}

let SubtargetPredicate = HasAtomicFaddInsts in {
defm : BufferAtomicPatterns_NO_RTN<SIbuffer_atomic_fadd, f32, "BUFFER_ATOMIC_ADD_F32">;
defm : BufferAtomicPatterns_NO_RTN<SIbuffer_atomic_fadd, v2f16, "BUFFER_ATOMIC_PK_ADD_F16">;
}

let SubtargetPredicate = isGFX90APlus in {
  defm : BufferAtomicPatterns<SIbuffer_atomic_fadd, f32,   "BUFFER_ATOMIC_ADD_F32">;
  defm : BufferAtomicPatterns<SIbuffer_atomic_fadd, v2f16, "BUFFER_ATOMIC_PK_ADD_F16">;

  defm : BufferAtomicPatterns<SIbuffer_atomic_fadd, f64, "BUFFER_ATOMIC_ADD_F64">;
  defm : BufferAtomicPatterns<SIbuffer_atomic_fmin, f64, "BUFFER_ATOMIC_MIN_F64">;
  defm : BufferAtomicPatterns<SIbuffer_atomic_fmax, f64, "BUFFER_ATOMIC_MAX_F64">;
} // End SubtargetPredicate = isGFX90APlus

def : GCNPat<
  (SIbuffer_atomic_cmpswap
      i32:$data, i32:$cmp, v4i32:$rsrc, 0, 0, i32:$soffset,
      timm:$offset, timm:$cachepolicy, 0),
  (EXTRACT_SUBREG (i64 (COPY_TO_REGCLASS
    (BUFFER_ATOMIC_CMPSWAP_OFFSET_RTN
      (REG_SEQUENCE VReg_64, VGPR_32:$data, sub0, VGPR_32:$cmp, sub1),
        SReg_128:$rsrc, SCSrc_b32:$soffset, (as_i16timm $offset),
        (set_glc $cachepolicy)), VReg_64)), sub0)
>;

def : GCNPat<
  (SIbuffer_atomic_cmpswap
      i32:$data, i32:$cmp, v4i32:$rsrc, i32:$vindex,
      0, i32:$soffset, timm:$offset,
      timm:$cachepolicy, timm),
  (EXTRACT_SUBREG (i64 (COPY_TO_REGCLASS
    (BUFFER_ATOMIC_CMPSWAP_IDXEN_RTN
      (REG_SEQUENCE VReg_64, VGPR_32:$data, sub0, VGPR_32:$cmp, sub1),
      VGPR_32:$vindex, SReg_128:$rsrc, SCSrc_b32:$soffset, (as_i16timm $offset),
      (set_glc $cachepolicy)), VReg_64)),
    sub0)
>;

def : GCNPat<
  (SIbuffer_atomic_cmpswap
      i32:$data, i32:$cmp, v4i32:$rsrc, 0,
      i32:$voffset, i32:$soffset, timm:$offset,
      timm:$cachepolicy, 0),
  (EXTRACT_SUBREG (i64 (COPY_TO_REGCLASS
    (BUFFER_ATOMIC_CMPSWAP_OFFEN_RTN
      (REG_SEQUENCE VReg_64, VGPR_32:$data, sub0, VGPR_32:$cmp, sub1),
      VGPR_32:$voffset, SReg_128:$rsrc, SCSrc_b32:$soffset, (as_i16timm $offset),
      (set_glc $cachepolicy)), VReg_64)),
    sub0)
>;

def : GCNPat<
  (SIbuffer_atomic_cmpswap
      i32:$data, i32:$cmp, v4i32:$rsrc, i32:$vindex,
      i32:$voffset, i32:$soffset, timm:$offset,
      timm:$cachepolicy, timm),
  (EXTRACT_SUBREG (i64 (COPY_TO_REGCLASS
    (BUFFER_ATOMIC_CMPSWAP_BOTHEN_RTN
      (REG_SEQUENCE VReg_64, VGPR_32:$data, sub0, VGPR_32:$cmp, sub1),
      (REG_SEQUENCE VReg_64, VGPR_32:$vindex, sub0, VGPR_32:$voffset, sub1),
      SReg_128:$rsrc, SCSrc_b32:$soffset, (as_i16timm $offset),
      (set_glc $cachepolicy)), VReg_64)),
    sub0)
>;

class MUBUFLoad_PatternADDR64 <MUBUF_Pseudo Instr_ADDR64, ValueType vt,
                              PatFrag constant_ld> : GCNPat <
     (vt (constant_ld (MUBUFAddr64 v4i32:$srsrc, i64:$vaddr, i32:$soffset,
                                   i16:$offset))),
     (Instr_ADDR64 $vaddr, $srsrc, $soffset, $offset)
  >;

multiclass MUBUFLoad_Atomic_Pattern <MUBUF_Pseudo Instr_ADDR64, MUBUF_Pseudo Instr_OFFSET,
                                     ValueType vt, PatFrag atomic_ld> {
  def : GCNPat <
     (vt (atomic_ld (MUBUFAddr64 v4i32:$srsrc, i64:$vaddr, i32:$soffset, i16:$offset))),
     (Instr_ADDR64 $vaddr, $srsrc, $soffset, $offset)
  >;

  def : GCNPat <
    (vt (atomic_ld (MUBUFOffset v4i32:$rsrc, i32:$soffset, i16:$offset))),
    (Instr_OFFSET $rsrc, $soffset, (as_i16imm $offset))
  >;
}

let SubtargetPredicate = isGFX6GFX7 in {
def : MUBUFLoad_PatternADDR64 <BUFFER_LOAD_SBYTE_ADDR64, i32, sextloadi8_constant>;
def : MUBUFLoad_PatternADDR64 <BUFFER_LOAD_UBYTE_ADDR64, i32, extloadi8_constant>;
def : MUBUFLoad_PatternADDR64 <BUFFER_LOAD_UBYTE_ADDR64, i32, zextloadi8_constant>;
def : MUBUFLoad_PatternADDR64 <BUFFER_LOAD_SSHORT_ADDR64, i32, sextloadi16_constant>;
def : MUBUFLoad_PatternADDR64 <BUFFER_LOAD_USHORT_ADDR64, i32, extloadi16_constant>;
def : MUBUFLoad_PatternADDR64 <BUFFER_LOAD_USHORT_ADDR64, i32, zextloadi16_constant>;

defm : MUBUFLoad_Atomic_Pattern <BUFFER_LOAD_DWORD_ADDR64, BUFFER_LOAD_DWORD_OFFSET, i32, atomic_load_32_global>;
defm : MUBUFLoad_Atomic_Pattern <BUFFER_LOAD_DWORDX2_ADDR64, BUFFER_LOAD_DWORDX2_OFFSET, i64, atomic_load_64_global>;
} // End SubtargetPredicate = isGFX6GFX7

multiclass MUBUFLoad_Pattern <MUBUF_Pseudo Instr_OFFSET, ValueType vt,
                               PatFrag ld> {

  def : GCNPat <
    (vt (ld (MUBUFOffset v4i32:$srsrc, i32:$soffset, i16:$offset))),
    (Instr_OFFSET $srsrc, $soffset, $offset)
  >;
}

let OtherPredicates = [Has16BitInsts] in {

defm : MUBUFLoad_Pattern <BUFFER_LOAD_SBYTE_OFFSET, i16, sextloadi8_constant>;
defm : MUBUFLoad_Pattern <BUFFER_LOAD_UBYTE_OFFSET, i16, extloadi8_constant>;
defm : MUBUFLoad_Pattern <BUFFER_LOAD_UBYTE_OFFSET, i16, zextloadi8_constant>;
defm : MUBUFLoad_Pattern <BUFFER_LOAD_SBYTE_OFFSET, i16, sextloadi8_global>;
defm : MUBUFLoad_Pattern <BUFFER_LOAD_UBYTE_OFFSET, i16, extloadi8_global>;
defm : MUBUFLoad_Pattern <BUFFER_LOAD_UBYTE_OFFSET, i16, zextloadi8_global>;

defm : MUBUFLoad_Pattern <BUFFER_LOAD_USHORT_OFFSET, i16, load_global>;

} // End OtherPredicates = [Has16BitInsts]

multiclass MUBUFScratchLoadPat <MUBUF_Pseudo InstrOffen,
                                MUBUF_Pseudo InstrOffset,
                                ValueType vt, PatFrag ld> {
  def : GCNPat <
    (vt (ld (MUBUFScratchOffen v4i32:$srsrc, i32:$vaddr,
                               i32:$soffset, u16imm:$offset))),
    (InstrOffen $vaddr, $srsrc, $soffset, $offset, 0, 0, 0)
  >;

  def : GCNPat <
    (vt (ld (MUBUFScratchOffset v4i32:$srsrc, i32:$soffset, u16imm:$offset))),
    (InstrOffset $srsrc, $soffset, $offset, 0, 0, 0)
  >;
}

// XXX - Is it possible to have a complex pattern in a PatFrag?
multiclass MUBUFScratchLoadPat_D16 <MUBUF_Pseudo InstrOffen,
                                MUBUF_Pseudo InstrOffset,
                                ValueType vt, PatFrag ld_frag> {
  def : GCNPat <
    (ld_frag (MUBUFScratchOffen v4i32:$srsrc, i32:$vaddr, i32:$soffset, u16imm:$offset), vt:$in),
    (InstrOffen $vaddr, $srsrc, $soffset, $offset, $in)
  >;

  def : GCNPat <
    (ld_frag (MUBUFScratchOffset v4i32:$srsrc, i32:$soffset, u16imm:$offset), vt:$in),
    (InstrOffset $srsrc, $soffset, $offset, $in)
  >;
}

let OtherPredicates = [DisableFlatScratch] in {
defm : MUBUFScratchLoadPat <BUFFER_LOAD_SBYTE_OFFEN, BUFFER_LOAD_SBYTE_OFFSET, i32, sextloadi8_private>;
defm : MUBUFScratchLoadPat <BUFFER_LOAD_UBYTE_OFFEN, BUFFER_LOAD_UBYTE_OFFSET, i32, extloadi8_private>;
defm : MUBUFScratchLoadPat <BUFFER_LOAD_UBYTE_OFFEN, BUFFER_LOAD_UBYTE_OFFSET, i32, zextloadi8_private>;
defm : MUBUFScratchLoadPat <BUFFER_LOAD_SBYTE_OFFEN, BUFFER_LOAD_SBYTE_OFFSET, i16, sextloadi8_private>;
defm : MUBUFScratchLoadPat <BUFFER_LOAD_UBYTE_OFFEN, BUFFER_LOAD_UBYTE_OFFSET, i16, extloadi8_private>;
defm : MUBUFScratchLoadPat <BUFFER_LOAD_UBYTE_OFFEN, BUFFER_LOAD_UBYTE_OFFSET, i16, zextloadi8_private>;
defm : MUBUFScratchLoadPat <BUFFER_LOAD_SSHORT_OFFEN, BUFFER_LOAD_SSHORT_OFFSET, i32, sextloadi16_private>;
defm : MUBUFScratchLoadPat <BUFFER_LOAD_USHORT_OFFEN, BUFFER_LOAD_USHORT_OFFSET, i32, extloadi16_private>;
defm : MUBUFScratchLoadPat <BUFFER_LOAD_USHORT_OFFEN, BUFFER_LOAD_USHORT_OFFSET, i32, zextloadi16_private>;
defm : MUBUFScratchLoadPat <BUFFER_LOAD_USHORT_OFFEN, BUFFER_LOAD_USHORT_OFFSET, i16, load_private>;

foreach vt = Reg32Types.types in {
defm : MUBUFScratchLoadPat <BUFFER_LOAD_DWORD_OFFEN, BUFFER_LOAD_DWORD_OFFSET, vt, load_private>;
}
defm : MUBUFScratchLoadPat <BUFFER_LOAD_DWORDX2_OFFEN, BUFFER_LOAD_DWORDX2_OFFSET, v2i32, load_private>;
defm : MUBUFScratchLoadPat <BUFFER_LOAD_DWORDX3_OFFEN, BUFFER_LOAD_DWORDX3_OFFSET, v3i32, load_private>;
defm : MUBUFScratchLoadPat <BUFFER_LOAD_DWORDX4_OFFEN, BUFFER_LOAD_DWORDX4_OFFSET, v4i32, load_private>;

let OtherPredicates = [D16PreservesUnusedBits, DisableFlatScratch] in {
defm : MUBUFScratchLoadPat_D16<BUFFER_LOAD_SHORT_D16_HI_OFFEN, BUFFER_LOAD_SHORT_D16_HI_OFFSET, v2i16, load_d16_hi_private>;
defm : MUBUFScratchLoadPat_D16<BUFFER_LOAD_UBYTE_D16_HI_OFFEN, BUFFER_LOAD_UBYTE_D16_HI_OFFSET, v2i16, az_extloadi8_d16_hi_private>;
defm : MUBUFScratchLoadPat_D16<BUFFER_LOAD_SBYTE_D16_HI_OFFEN, BUFFER_LOAD_SBYTE_D16_HI_OFFSET, v2i16, sextloadi8_d16_hi_private>;
defm : MUBUFScratchLoadPat_D16<BUFFER_LOAD_SHORT_D16_HI_OFFEN, BUFFER_LOAD_SHORT_D16_HI_OFFSET, v2f16, load_d16_hi_private>;
defm : MUBUFScratchLoadPat_D16<BUFFER_LOAD_UBYTE_D16_HI_OFFEN, BUFFER_LOAD_UBYTE_D16_HI_OFFSET, v2f16, az_extloadi8_d16_hi_private>;
defm : MUBUFScratchLoadPat_D16<BUFFER_LOAD_SBYTE_D16_HI_OFFEN, BUFFER_LOAD_SBYTE_D16_HI_OFFSET, v2f16, sextloadi8_d16_hi_private>;

defm : MUBUFScratchLoadPat_D16<BUFFER_LOAD_SHORT_D16_OFFEN, BUFFER_LOAD_SHORT_D16_OFFSET, v2i16, load_d16_lo_private>;
defm : MUBUFScratchLoadPat_D16<BUFFER_LOAD_UBYTE_D16_OFFEN, BUFFER_LOAD_UBYTE_D16_OFFSET, v2i16, az_extloadi8_d16_lo_private>;
defm : MUBUFScratchLoadPat_D16<BUFFER_LOAD_SBYTE_D16_OFFEN, BUFFER_LOAD_SBYTE_D16_OFFSET, v2i16, sextloadi8_d16_lo_private>;
defm : MUBUFScratchLoadPat_D16<BUFFER_LOAD_SHORT_D16_OFFEN, BUFFER_LOAD_SHORT_D16_OFFSET, v2f16, load_d16_lo_private>;
defm : MUBUFScratchLoadPat_D16<BUFFER_LOAD_UBYTE_D16_OFFEN, BUFFER_LOAD_UBYTE_D16_OFFSET, v2f16, az_extloadi8_d16_lo_private>;
defm : MUBUFScratchLoadPat_D16<BUFFER_LOAD_SBYTE_D16_OFFEN, BUFFER_LOAD_SBYTE_D16_OFFSET, v2f16, sextloadi8_d16_lo_private>;
}

} // End OtherPredicates = [DisableFlatScratch]

multiclass MUBUFStore_Atomic_Pattern <MUBUF_Pseudo Instr_ADDR64, MUBUF_Pseudo Instr_OFFSET,
                                      ValueType vt, PatFrag atomic_st> {
  // Store follows atomic op convention so address is first
  def : GCNPat <
     (atomic_st (MUBUFAddr64 v4i32:$srsrc, i64:$vaddr, i32:$soffset, i16:$offset), vt:$val),
     (Instr_ADDR64 $val, $vaddr, $srsrc, $soffset, $offset)
  >;

  def : GCNPat <
    (atomic_st (MUBUFOffset v4i32:$rsrc, i32:$soffset, i16:$offset), vt:$val),
    (Instr_OFFSET $val, $rsrc, $soffset, (as_i16imm $offset))
  >;
}
let SubtargetPredicate = isGFX6GFX7 in {
defm : MUBUFStore_Atomic_Pattern <BUFFER_STORE_DWORD_ADDR64, BUFFER_STORE_DWORD_OFFSET, i32, atomic_store_global_32>;
defm : MUBUFStore_Atomic_Pattern <BUFFER_STORE_DWORDX2_ADDR64, BUFFER_STORE_DWORDX2_OFFSET, i64, atomic_store_global_64>;
} // End Predicates = isGFX6GFX7


multiclass MUBUFStore_Pattern <MUBUF_Pseudo Instr_OFFSET, ValueType vt,
                               PatFrag st> {

  def : GCNPat <
    (st vt:$vdata, (MUBUFOffset v4i32:$srsrc, i32:$soffset, i16:$offset)),
    (Instr_OFFSET $vdata, $srsrc, $soffset, $offset)
  >;
}

defm : MUBUFStore_Pattern <BUFFER_STORE_BYTE_OFFSET, i16, truncstorei8_global>;
defm : MUBUFStore_Pattern <BUFFER_STORE_SHORT_OFFSET, i16, store_global>;

multiclass MUBUFScratchStorePat <MUBUF_Pseudo InstrOffen,
                                 MUBUF_Pseudo InstrOffset,
                                 ValueType vt, PatFrag st,
                                 RegisterClass rc = VGPR_32> {
  def : GCNPat <
    (st vt:$value, (MUBUFScratchOffen v4i32:$srsrc, i32:$vaddr,
                                      i32:$soffset, u16imm:$offset)),
    (InstrOffen rc:$value, $vaddr, $srsrc, $soffset, $offset, 0, 0, 0)
  >;

  def : GCNPat <
    (st vt:$value, (MUBUFScratchOffset v4i32:$srsrc, i32:$soffset,
                                       u16imm:$offset)),
    (InstrOffset rc:$value, $srsrc, $soffset, $offset, 0, 0, 0)
  >;
}

let OtherPredicates = [DisableFlatScratch] in {
defm : MUBUFScratchStorePat <BUFFER_STORE_BYTE_OFFEN, BUFFER_STORE_BYTE_OFFSET, i32, truncstorei8_private>;
defm : MUBUFScratchStorePat <BUFFER_STORE_SHORT_OFFEN, BUFFER_STORE_SHORT_OFFSET, i32, truncstorei16_private>;
defm : MUBUFScratchStorePat <BUFFER_STORE_BYTE_OFFEN, BUFFER_STORE_BYTE_OFFSET, i16, truncstorei8_private>;
defm : MUBUFScratchStorePat <BUFFER_STORE_SHORT_OFFEN, BUFFER_STORE_SHORT_OFFSET, i16, store_private>;

foreach vt = Reg32Types.types in {
defm : MUBUFScratchStorePat <BUFFER_STORE_DWORD_OFFEN, BUFFER_STORE_DWORD_OFFSET, vt, store_private>;
}

defm : MUBUFScratchStorePat <BUFFER_STORE_DWORDX2_OFFEN, BUFFER_STORE_DWORDX2_OFFSET, v2i32, store_private, VReg_64>;
defm : MUBUFScratchStorePat <BUFFER_STORE_DWORDX3_OFFEN, BUFFER_STORE_DWORDX3_OFFSET, v3i32, store_private, VReg_96>;
defm : MUBUFScratchStorePat <BUFFER_STORE_DWORDX4_OFFEN, BUFFER_STORE_DWORDX4_OFFSET, v4i32, store_private, VReg_128>;


let OtherPredicates = [D16PreservesUnusedBits, DisableFlatScratch] in {
 // Hiding the extract high pattern in the PatFrag seems to not
 // automatically increase the complexity.
let AddedComplexity = 1 in {
defm : MUBUFScratchStorePat <BUFFER_STORE_SHORT_D16_HI_OFFEN, BUFFER_STORE_SHORT_D16_HI_OFFSET, i32, store_hi16_private>;
defm : MUBUFScratchStorePat <BUFFER_STORE_BYTE_D16_HI_OFFEN, BUFFER_STORE_BYTE_D16_HI_OFFSET, i32, truncstorei8_hi16_private>;
}
}
} // End OtherPredicates = [DisableFlatScratch]

//===----------------------------------------------------------------------===//
// MTBUF Patterns
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// tbuffer_load/store_format patterns
//===----------------------------------------------------------------------===//

multiclass MTBUF_LoadIntrinsicPat<SDPatternOperator name, ValueType vt,
                                  string opcode, ValueType memoryVt = vt> {
  defvar st = !if(!eq(!cast<string>(memoryVt), !cast<string>(vt)), name, mtbuf_intrinsic_load<name, memoryVt>);

  def : GCNPat<
    (vt (st v4i32:$rsrc, 0, 0, i32:$soffset, timm:$offset,
              timm:$format, timm:$auxiliary, 0)),
    (!cast<MTBUF_Pseudo>(opcode # _OFFSET) SReg_128:$rsrc, SCSrc_b32:$soffset, (as_i16timm $offset),
      (as_i8timm $format),
      (extract_cpol $auxiliary), 0, (extract_swz $auxiliary))
  >;

  def : GCNPat<
    (vt (st v4i32:$rsrc, i32:$vindex, 0, i32:$soffset, timm:$offset,
              timm:$format, timm:$auxiliary, timm)),
    (!cast<MTBUF_Pseudo>(opcode # _IDXEN) VGPR_32:$vindex, SReg_128:$rsrc, SCSrc_b32:$soffset, (as_i16timm $offset),
      (as_i8timm $format),
      (extract_cpol $auxiliary), 0, (extract_swz $auxiliary))
  >;

  def : GCNPat<
    (vt (st v4i32:$rsrc, 0, i32:$voffset, i32:$soffset, timm:$offset,
              timm:$format, timm:$auxiliary, 0)),
    (!cast<MTBUF_Pseudo>(opcode # _OFFEN) VGPR_32:$voffset, SReg_128:$rsrc, SCSrc_b32:$soffset, (as_i16timm $offset),
      (as_i8timm $format),
      (extract_cpol $auxiliary), 0, (extract_swz $auxiliary))
  >;

  def : GCNPat<
    (vt (st v4i32:$rsrc, i32:$vindex, i32:$voffset, i32:$soffset, timm:$offset,
              timm:$format, timm:$auxiliary, timm)),
    (!cast<MTBUF_Pseudo>(opcode # _BOTHEN)
      (REG_SEQUENCE VReg_64, VGPR_32:$vindex, sub0, VGPR_32:$voffset, sub1),
      SReg_128:$rsrc, SCSrc_b32:$soffset, (as_i16timm $offset),
      (as_i8timm $format),
      (extract_cpol $auxiliary), 0, (extract_swz $auxiliary))
  >;
}

defm : MTBUF_LoadIntrinsicPat<SItbuffer_load, i32,   "TBUFFER_LOAD_FORMAT_X">;
defm : MTBUF_LoadIntrinsicPat<SItbuffer_load, v2i32, "TBUFFER_LOAD_FORMAT_XY">;
defm : MTBUF_LoadIntrinsicPat<SItbuffer_load, v3i32, "TBUFFER_LOAD_FORMAT_XYZ">;
defm : MTBUF_LoadIntrinsicPat<SItbuffer_load, v4i32, "TBUFFER_LOAD_FORMAT_XYZW">;
defm : MTBUF_LoadIntrinsicPat<SItbuffer_load, f32,   "TBUFFER_LOAD_FORMAT_X">;
defm : MTBUF_LoadIntrinsicPat<SItbuffer_load, v2f32, "TBUFFER_LOAD_FORMAT_XY">;
defm : MTBUF_LoadIntrinsicPat<SItbuffer_load, v3f32, "TBUFFER_LOAD_FORMAT_XYZ">;
defm : MTBUF_LoadIntrinsicPat<SItbuffer_load, v4f32, "TBUFFER_LOAD_FORMAT_XYZW">;

let SubtargetPredicate = HasUnpackedD16VMem in {
  defm : MTBUF_LoadIntrinsicPat<SItbuffer_load_d16, f16,   "TBUFFER_LOAD_FORMAT_D16_X_gfx80">;
  defm : MTBUF_LoadIntrinsicPat<SItbuffer_load_d16, i32,   "TBUFFER_LOAD_FORMAT_D16_X_gfx80">;
  defm : MTBUF_LoadIntrinsicPat<SItbuffer_load_d16, v2i32, "TBUFFER_LOAD_FORMAT_D16_XY_gfx80">;
  defm : MTBUF_LoadIntrinsicPat<SItbuffer_load_d16, v3i32, "TBUFFER_LOAD_FORMAT_D16_XYZ_gfx80">;
  defm : MTBUF_LoadIntrinsicPat<SItbuffer_load_d16, v4i32, "TBUFFER_LOAD_FORMAT_D16_XYZW_gfx80">;
} // End HasUnpackedD16VMem.

let SubtargetPredicate = HasPackedD16VMem in {
  defm : MTBUF_LoadIntrinsicPat<SItbuffer_load_d16, f16,   "TBUFFER_LOAD_FORMAT_D16_X">;
  defm : MTBUF_LoadIntrinsicPat<SItbuffer_load_d16, i32,   "TBUFFER_LOAD_FORMAT_D16_X">;
  defm : MTBUF_LoadIntrinsicPat<SItbuffer_load_d16, v2f16, "TBUFFER_LOAD_FORMAT_D16_XY">;
  defm : MTBUF_LoadIntrinsicPat<SItbuffer_load_d16, v4f16, "TBUFFER_LOAD_FORMAT_D16_XYZ", v3f16>;
  defm : MTBUF_LoadIntrinsicPat<SItbuffer_load_d16, v4f16, "TBUFFER_LOAD_FORMAT_D16_XYZW">;
} // End HasPackedD16VMem.

multiclass MTBUF_StoreIntrinsicPat<SDPatternOperator name, ValueType vt,
                                        string opcode, ValueType memoryVt = vt> {
  defvar st = !if(!eq(!cast<string>(memoryVt), !cast<string>(vt)), name, mtbuf_intrinsic_store<name, memoryVt>);

  def : GCNPat<
    (st vt:$vdata, v4i32:$rsrc, 0, 0, i32:$soffset, timm:$offset,
          timm:$format, timm:$auxiliary, 0),
    (!cast<MTBUF_Pseudo>(opcode # _OFFSET_exact) getVregSrcForVT<vt>.ret:$vdata, SReg_128:$rsrc, SCSrc_b32:$soffset,
      (as_i16timm $offset), (as_i8timm $format),
      (extract_cpol $auxiliary), 0, (extract_swz $auxiliary))
  >;

  def : GCNPat<
    (st vt:$vdata, v4i32:$rsrc, i32:$vindex, 0, i32:$soffset, timm:$offset,
          timm:$format, timm:$auxiliary, timm),
    (!cast<MTBUF_Pseudo>(opcode # _IDXEN_exact) getVregSrcForVT<vt>.ret:$vdata, VGPR_32:$vindex, SReg_128:$rsrc, SCSrc_b32:$soffset,
      (as_i16timm $offset), (as_i8timm $format),
      (extract_cpol $auxiliary), 0, (extract_swz $auxiliary))
  >;

  def : GCNPat<
    (st vt:$vdata, v4i32:$rsrc, 0, i32:$voffset, i32:$soffset, timm:$offset,
          timm:$format, timm:$auxiliary, 0),
    (!cast<MTBUF_Pseudo>(opcode # _OFFEN_exact) getVregSrcForVT<vt>.ret:$vdata, VGPR_32:$voffset, SReg_128:$rsrc, SCSrc_b32:$soffset,
      (as_i16timm $offset), (as_i8timm $format),
      (extract_cpol $auxiliary), 0, (extract_swz $auxiliary))
  >;

  def : GCNPat<
    (st vt:$vdata, v4i32:$rsrc, i32:$vindex, i32:$voffset, i32:$soffset,
          timm:$offset, timm:$format, timm:$auxiliary, timm),
    (!cast<MTBUF_Pseudo>(opcode # _BOTHEN_exact)
      getVregSrcForVT<vt>.ret:$vdata,
      (REG_SEQUENCE VReg_64, VGPR_32:$vindex, sub0, VGPR_32:$voffset, sub1),
      SReg_128:$rsrc, SCSrc_b32:$soffset, (as_i16timm $offset), (as_i8timm $format),
      (extract_cpol $auxiliary), 0, (extract_swz $auxiliary))
  >;
}

defm : MTBUF_StoreIntrinsicPat<SItbuffer_store, i32,   "TBUFFER_STORE_FORMAT_X">;
defm : MTBUF_StoreIntrinsicPat<SItbuffer_store, v2i32, "TBUFFER_STORE_FORMAT_XY">;
defm : MTBUF_StoreIntrinsicPat<SItbuffer_store, v3i32, "TBUFFER_STORE_FORMAT_XYZ">;
defm : MTBUF_StoreIntrinsicPat<SItbuffer_store, v4i32, "TBUFFER_STORE_FORMAT_XYZW">;
defm : MTBUF_StoreIntrinsicPat<SItbuffer_store, f32,   "TBUFFER_STORE_FORMAT_X">;
defm : MTBUF_StoreIntrinsicPat<SItbuffer_store, v2f32, "TBUFFER_STORE_FORMAT_XY">;
defm : MTBUF_StoreIntrinsicPat<SItbuffer_store, v3f32, "TBUFFER_STORE_FORMAT_XYZ">;
defm : MTBUF_StoreIntrinsicPat<SItbuffer_store, v4f32, "TBUFFER_STORE_FORMAT_XYZW">;

let SubtargetPredicate = HasUnpackedD16VMem in {
  defm : MTBUF_StoreIntrinsicPat<SItbuffer_store_d16, f16,   "TBUFFER_STORE_FORMAT_D16_X_gfx80">;
  defm : MTBUF_StoreIntrinsicPat<SItbuffer_store_d16, i32,   "TBUFFER_STORE_FORMAT_D16_X_gfx80">;
  defm : MTBUF_StoreIntrinsicPat<SItbuffer_store_d16, v2i32, "TBUFFER_STORE_FORMAT_D16_XY_gfx80">;
  defm : MTBUF_StoreIntrinsicPat<SItbuffer_store_d16, v3i32, "TBUFFER_STORE_FORMAT_D16_XYZ_gfx80">;
  defm : MTBUF_StoreIntrinsicPat<SItbuffer_store_d16, v4i32, "TBUFFER_STORE_FORMAT_D16_XYZW_gfx80">;
} // End HasUnpackedD16VMem.

let SubtargetPredicate = HasPackedD16VMem in {
  defm : MTBUF_StoreIntrinsicPat<SItbuffer_store_d16, f16,   "TBUFFER_STORE_FORMAT_D16_X">;
  defm : MTBUF_StoreIntrinsicPat<SItbuffer_store_d16, i32,   "TBUFFER_STORE_FORMAT_D16_X">;
  defm : MTBUF_StoreIntrinsicPat<SItbuffer_store_d16, v2f16, "TBUFFER_STORE_FORMAT_D16_XY">;
  defm : MTBUF_StoreIntrinsicPat<SItbuffer_store_d16, v4f16, "TBUFFER_STORE_FORMAT_D16_XYZ", v3f16>;
  defm : MTBUF_StoreIntrinsicPat<SItbuffer_store_d16, v4f16, "TBUFFER_STORE_FORMAT_D16_XYZW">;
} // End HasPackedD16VMem.

//===----------------------------------------------------------------------===//
// Target-specific instruction encodings.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Base ENC_MUBUF for GFX6, GFX7, GFX10.
//===----------------------------------------------------------------------===//

class Base_MUBUF_Real_gfx6_gfx7_gfx10<bits<7> op, MUBUF_Pseudo ps, int ef> :
    MUBUF_Real<ps>, Enc64, SIMCInstr<ps.PseudoInstr, ef> {
  let Inst{11-0}  = !if(ps.has_offset, offset, ?);
  let Inst{12}    = ps.offen;
  let Inst{13}    = ps.idxen;
  let Inst{14}    = !if(ps.has_glc, cpol{CPolBit.GLC}, ps.glc_value);
  let Inst{16}    = ps.lds;
  let Inst{24-18} = op;
  let Inst{31-26} = 0x38;
  let Inst{39-32} = !if(ps.has_vaddr, vaddr, ?);
  let Inst{47-40} = !if(ps.has_vdata, vdata{7-0}, ?);
  let Inst{52-48} = !if(ps.has_srsrc, srsrc{6-2}, ?);
  let Inst{54}    = !if(ps.has_slc, cpol{CPolBit.SLC}, ?);
  let Inst{55}    = !if(ps.has_tfe, tfe, ?);
  let Inst{63-56} = !if(ps.has_soffset, soffset, ?);
}

class MUBUF_Real_gfx10<bits<8> op, MUBUF_Pseudo ps> :
    Base_MUBUF_Real_gfx6_gfx7_gfx10<op{6-0}, ps, SIEncodingFamily.GFX10> {
  let Inst{15} = !if(ps.has_dlc, cpol{CPolBit.DLC}, ps.dlc_value);
  let Inst{25} = op{7};
}

class MUBUF_Real_gfx6_gfx7<bits<8> op, MUBUF_Pseudo ps> :
    Base_MUBUF_Real_gfx6_gfx7_gfx10<op{6-0}, ps, SIEncodingFamily.SI> {
  let Inst{15} = ps.addr64;
}

//===----------------------------------------------------------------------===//
// MUBUF - GFX10.
//===----------------------------------------------------------------------===//

let AssemblerPredicate = isGFX10Plus, DecoderNamespace = "GFX10" in {
  multiclass MUBUF_Real_gfx10_with_name<bits<8> op, string opName,
                                        string asmName> {
    def _gfx10 : MUBUF_Real_gfx10<op, !cast<MUBUF_Pseudo>(opName)> {
      MUBUF_Pseudo ps = !cast<MUBUF_Pseudo>(opName);
      let AsmString = asmName # ps.AsmOperands;
    }
  }
  multiclass MUBUF_Real_AllAddr_gfx10<bits<8> op> {
    def _BOTHEN_gfx10 :
      MUBUF_Real_gfx10<op, !cast<MUBUF_Pseudo>(NAME#"_BOTHEN")>;
    def _IDXEN_gfx10 :
      MUBUF_Real_gfx10<op, !cast<MUBUF_Pseudo>(NAME#"_IDXEN")>;
    def _OFFEN_gfx10 :
      MUBUF_Real_gfx10<op, !cast<MUBUF_Pseudo>(NAME#"_OFFEN")>;
    def _OFFSET_gfx10 :
      MUBUF_Real_gfx10<op, !cast<MUBUF_Pseudo>(NAME#"_OFFSET")>;
  }
  multiclass MUBUF_Real_AllAddr_Lds_gfx10<bits<8> op> {
    def _OFFSET_gfx10 : MUBUF_Real_gfx10<op, !cast<MUBUF_Pseudo>(NAME#"_OFFSET")>,
                        MUBUFLdsTable<0, NAME # "_OFFSET_gfx10">;
    def _OFFEN_gfx10  : MUBUF_Real_gfx10<op, !cast<MUBUF_Pseudo>(NAME#"_OFFEN")>,
                        MUBUFLdsTable<0, NAME # "_OFFEN_gfx10">;
    def _IDXEN_gfx10  : MUBUF_Real_gfx10<op, !cast<MUBUF_Pseudo>(NAME#"_IDXEN")>,
                        MUBUFLdsTable<0, NAME # "_IDXEN_gfx10">;
    def _BOTHEN_gfx10 : MUBUF_Real_gfx10<op, !cast<MUBUF_Pseudo>(NAME#"_BOTHEN")>,
                        MUBUFLdsTable<0, NAME # "_BOTHEN_gfx10">;

    def _LDS_OFFSET_gfx10 : MUBUF_Real_gfx10<op, !cast<MUBUF_Pseudo>(NAME#"_LDS_OFFSET")>,
                            MUBUFLdsTable<1, NAME # "_OFFSET_gfx10">;
    def _LDS_OFFEN_gfx10  : MUBUF_Real_gfx10<op, !cast<MUBUF_Pseudo>(NAME#"_LDS_OFFEN")>,
                            MUBUFLdsTable<1, NAME # "_OFFEN_gfx10">;
    def _LDS_IDXEN_gfx10  : MUBUF_Real_gfx10<op, !cast<MUBUF_Pseudo>(NAME#"_LDS_IDXEN")>,
                            MUBUFLdsTable<1, NAME # "_IDXEN_gfx10">;
    def _LDS_BOTHEN_gfx10 : MUBUF_Real_gfx10<op, !cast<MUBUF_Pseudo>(NAME#"_LDS_BOTHEN")>,
                            MUBUFLdsTable<1, NAME # "_BOTHEN_gfx10">;
  }
  multiclass MUBUF_Real_Atomics_RTN_gfx10<bits<8> op> {
    def _BOTHEN_RTN_gfx10 :
      MUBUF_Real_gfx10<op, !cast<MUBUF_Pseudo>(NAME#"_BOTHEN_RTN")>,
      AtomicNoRet<NAME # "_BOTHEN_gfx10", 1>;
    def _IDXEN_RTN_gfx10 :
      MUBUF_Real_gfx10<op, !cast<MUBUF_Pseudo>(NAME#"_IDXEN_RTN")>,
      AtomicNoRet<NAME # "_IDXEN_gfx10", 1>;
    def _OFFEN_RTN_gfx10 :
      MUBUF_Real_gfx10<op, !cast<MUBUF_Pseudo>(NAME#"_OFFEN_RTN")>,
      AtomicNoRet<NAME # "_OFFEN_gfx10", 1>;
    def _OFFSET_RTN_gfx10 :
      MUBUF_Real_gfx10<op, !cast<MUBUF_Pseudo>(NAME#"_OFFSET_RTN")>,
      AtomicNoRet<NAME # "_OFFSET_gfx10", 1>;
  }
  multiclass MUBUF_Real_Atomics_gfx10<bits<8> op> :
      MUBUF_Real_Atomics_RTN_gfx10<op> {
    def _BOTHEN_gfx10 :
      MUBUF_Real_gfx10<op, !cast<MUBUF_Pseudo>(NAME#"_BOTHEN")>,
      AtomicNoRet<NAME # "_BOTHEN_gfx10", 0>;
    def _IDXEN_gfx10 :
      MUBUF_Real_gfx10<op, !cast<MUBUF_Pseudo>(NAME#"_IDXEN")>,
      AtomicNoRet<NAME # "_IDXEN_gfx10", 0>;
    def _OFFEN_gfx10 :
      MUBUF_Real_gfx10<op, !cast<MUBUF_Pseudo>(NAME#"_OFFEN")>,
      AtomicNoRet<NAME # "_OFFEN_gfx10", 0>;
    def _OFFSET_gfx10 :
      MUBUF_Real_gfx10<op, !cast<MUBUF_Pseudo>(NAME#"_OFFSET")>,
      AtomicNoRet<NAME # "_OFFSET_gfx10", 0>;
  }
} // End AssemblerPredicate = isGFX10Plus, DecoderNamespace = "GFX10"

defm BUFFER_STORE_BYTE_D16_HI     : MUBUF_Real_AllAddr_gfx10<0x019>;
defm BUFFER_STORE_SHORT_D16_HI    : MUBUF_Real_AllAddr_gfx10<0x01b>;
defm BUFFER_LOAD_UBYTE_D16        : MUBUF_Real_AllAddr_gfx10<0x020>;
defm BUFFER_LOAD_UBYTE_D16_HI     : MUBUF_Real_AllAddr_gfx10<0x021>;
defm BUFFER_LOAD_SBYTE_D16        : MUBUF_Real_AllAddr_gfx10<0x022>;
defm BUFFER_LOAD_SBYTE_D16_HI     : MUBUF_Real_AllAddr_gfx10<0x023>;
defm BUFFER_LOAD_SHORT_D16        : MUBUF_Real_AllAddr_gfx10<0x024>;
defm BUFFER_LOAD_SHORT_D16_HI     : MUBUF_Real_AllAddr_gfx10<0x025>;
// FIXME-GFX10: Add following instructions:
//defm BUFFER_LOAD_FORMAT_D16_HI_X  : MUBUF_Real_AllAddr_gfx10<0x026>;
//defm BUFFER_STORE_FORMAT_D16_HI_X : MUBUF_Real_AllAddr_gfx10<0x027>;
defm BUFFER_LOAD_FORMAT_D16_X     : MUBUF_Real_AllAddr_gfx10<0x080>;
defm BUFFER_LOAD_FORMAT_D16_XY    : MUBUF_Real_AllAddr_gfx10<0x081>;
defm BUFFER_LOAD_FORMAT_D16_XYZ   : MUBUF_Real_AllAddr_gfx10<0x082>;
defm BUFFER_LOAD_FORMAT_D16_XYZW  : MUBUF_Real_AllAddr_gfx10<0x083>;
defm BUFFER_STORE_FORMAT_D16_X    : MUBUF_Real_AllAddr_gfx10<0x084>;
defm BUFFER_STORE_FORMAT_D16_XY   : MUBUF_Real_AllAddr_gfx10<0x085>;
defm BUFFER_STORE_FORMAT_D16_XYZ  : MUBUF_Real_AllAddr_gfx10<0x086>;
defm BUFFER_STORE_FORMAT_D16_XYZW : MUBUF_Real_AllAddr_gfx10<0x087>;

def BUFFER_GL0_INV_gfx10 :
  MUBUF_Real_gfx10<0x071, BUFFER_GL0_INV>;
def BUFFER_GL1_INV_gfx10 :
  MUBUF_Real_gfx10<0x072, BUFFER_GL1_INV>;

//===----------------------------------------------------------------------===//
// MUBUF - GFX6, GFX7, GFX10.
//===----------------------------------------------------------------------===//

let AssemblerPredicate = isGFX6, DecoderNamespace = "GFX6" in {
  multiclass MUBUF_Real_gfx6<bits<8> op> {
    def _gfx6 : MUBUF_Real_gfx6_gfx7<op, !cast<MUBUF_Pseudo>(NAME)>;
  }
} // End AssemblerPredicate = isGFX6, DecoderNamespace = "GFX6"

let AssemblerPredicate = isGFX7Only, DecoderNamespace = "GFX7" in {
  multiclass MUBUF_Real_gfx7<bits<8> op> {
    def _gfx7 : MUBUF_Real_gfx6_gfx7<op, !cast<MUBUF_Pseudo>(NAME)>;
  }
} // End AssemblerPredicate = isGFX7Only, DecoderNamespace = "GFX7"

let AssemblerPredicate = isGFX6GFX7, DecoderNamespace = "GFX6GFX7" in {
  multiclass MUBUF_Real_AllAddr_gfx6_gfx7<bits<8> op> {
    def _ADDR64_gfx6_gfx7 :
      MUBUF_Real_gfx6_gfx7<op, !cast<MUBUF_Pseudo>(NAME#"_ADDR64")>;
    def _BOTHEN_gfx6_gfx7 :
      MUBUF_Real_gfx6_gfx7<op, !cast<MUBUF_Pseudo>(NAME#"_BOTHEN")>;
    def _IDXEN_gfx6_gfx7 :
      MUBUF_Real_gfx6_gfx7<op, !cast<MUBUF_Pseudo>(NAME#"_IDXEN")>;
    def _OFFEN_gfx6_gfx7 :
      MUBUF_Real_gfx6_gfx7<op, !cast<MUBUF_Pseudo>(NAME#"_OFFEN")>;
    def _OFFSET_gfx6_gfx7 :
      MUBUF_Real_gfx6_gfx7<op, !cast<MUBUF_Pseudo>(NAME#"_OFFSET")>;
  }
  multiclass MUBUF_Real_AllAddr_Lds_gfx6_gfx7<bits<8> op> {
    def _OFFSET_gfx6_gfx7 : MUBUF_Real_gfx6_gfx7<op, !cast<MUBUF_Pseudo>(NAME#"_OFFSET")>,
                            MUBUFLdsTable<0, NAME # "_OFFSET_gfx6_gfx7">;
    def _ADDR64_gfx6_gfx7 : MUBUF_Real_gfx6_gfx7<op, !cast<MUBUF_Pseudo>(NAME#"_ADDR64")>,
                            MUBUFLdsTable<0, NAME # "_ADDR64_gfx6_gfx7">;
    def _OFFEN_gfx6_gfx7  : MUBUF_Real_gfx6_gfx7<op, !cast<MUBUF_Pseudo>(NAME#"_OFFEN")>,
                            MUBUFLdsTable<0, NAME # "_OFFEN_gfx6_gfx7">;
    def _IDXEN_gfx6_gfx7  : MUBUF_Real_gfx6_gfx7<op, !cast<MUBUF_Pseudo>(NAME#"_IDXEN")>,
                            MUBUFLdsTable<0, NAME # "_IDXEN_gfx6_gfx7">;
    def _BOTHEN_gfx6_gfx7 : MUBUF_Real_gfx6_gfx7<op, !cast<MUBUF_Pseudo>(NAME#"_BOTHEN")>,
                            MUBUFLdsTable<0, NAME # "_BOTHEN_gfx6_gfx7">;

    def _LDS_OFFSET_gfx6_gfx7 : MUBUF_Real_gfx6_gfx7<op, !cast<MUBUF_Pseudo>(NAME#"_LDS_OFFSET")>,
                                MUBUFLdsTable<1, NAME # "_OFFSET_gfx6_gfx7">;
    def _LDS_ADDR64_gfx6_gfx7 : MUBUF_Real_gfx6_gfx7<op, !cast<MUBUF_Pseudo>(NAME#"_LDS_ADDR64")>,
                                MUBUFLdsTable<1, NAME # "_ADDR64_gfx6_gfx7">;
    def _LDS_OFFEN_gfx6_gfx7  : MUBUF_Real_gfx6_gfx7<op, !cast<MUBUF_Pseudo>(NAME#"_LDS_OFFEN")>,
                                MUBUFLdsTable<1, NAME # "_OFFEN_gfx6_gfx7">;
    def _LDS_IDXEN_gfx6_gfx7  : MUBUF_Real_gfx6_gfx7<op, !cast<MUBUF_Pseudo>(NAME#"_LDS_IDXEN")>,
                                MUBUFLdsTable<1, NAME # "_IDXEN_gfx6_gfx7">;
    def _LDS_BOTHEN_gfx6_gfx7 : MUBUF_Real_gfx6_gfx7<op, !cast<MUBUF_Pseudo>(NAME#"_LDS_BOTHEN")>,
                                MUBUFLdsTable<1, NAME # "_BOTHEN_gfx6_gfx7">;
  }
  multiclass MUBUF_Real_Atomics_gfx6_gfx7<bits<8> op> {
    def _ADDR64_gfx6_gfx7 :
      MUBUF_Real_gfx6_gfx7<op, !cast<MUBUF_Pseudo>(NAME#"_ADDR64")>,
      AtomicNoRet<NAME # "_ADDR64_gfx6_gfx7", 0>;
    def _BOTHEN_gfx6_gfx7 :
      MUBUF_Real_gfx6_gfx7<op, !cast<MUBUF_Pseudo>(NAME#"_BOTHEN")>,
      AtomicNoRet<NAME # "_BOTHEN_gfx6_gfx7", 0>;
    def _IDXEN_gfx6_gfx7 :
      MUBUF_Real_gfx6_gfx7<op, !cast<MUBUF_Pseudo>(NAME#"_IDXEN")>,
      AtomicNoRet<NAME # "_IDXEN_gfx6_gfx7", 0>;
    def _OFFEN_gfx6_gfx7 :
      MUBUF_Real_gfx6_gfx7<op, !cast<MUBUF_Pseudo>(NAME#"_OFFEN")>,
      AtomicNoRet<NAME # "_OFFEN_gfx6_gfx7", 0>;
    def _OFFSET_gfx6_gfx7 :
      MUBUF_Real_gfx6_gfx7<op, !cast<MUBUF_Pseudo>(NAME#"_OFFSET")>,
      AtomicNoRet<NAME # "_OFFSET_gfx6_gfx7", 0>;

    def _ADDR64_RTN_gfx6_gfx7 :
      MUBUF_Real_gfx6_gfx7<op, !cast<MUBUF_Pseudo>(NAME#"_ADDR64_RTN")>,
      AtomicNoRet<NAME # "_ADDR64_gfx6_gfx7", 1>;
    def _BOTHEN_RTN_gfx6_gfx7 :
      MUBUF_Real_gfx6_gfx7<op, !cast<MUBUF_Pseudo>(NAME#"_BOTHEN_RTN")>,
      AtomicNoRet<NAME # "_BOTHEN_gfx6_gfx7", 1>;
    def _IDXEN_RTN_gfx6_gfx7 :
      MUBUF_Real_gfx6_gfx7<op, !cast<MUBUF_Pseudo>(NAME#"_IDXEN_RTN")>,
      AtomicNoRet<NAME # "_IDXEN_gfx6_gfx7", 1>;
    def _OFFEN_RTN_gfx6_gfx7 :
      MUBUF_Real_gfx6_gfx7<op, !cast<MUBUF_Pseudo>(NAME#"_OFFEN_RTN")>,
      AtomicNoRet<NAME # "_OFFEN_gfx6_gfx7", 1>;
    def _OFFSET_RTN_gfx6_gfx7 :
      MUBUF_Real_gfx6_gfx7<op, !cast<MUBUF_Pseudo>(NAME#"_OFFSET_RTN")>,
      AtomicNoRet<NAME # "_OFFSET_gfx6_gfx7", 1>;
  }
} // End AssemblerPredicate = isGFX6GFX7, DecoderNamespace = "GFX6GFX7"

multiclass MUBUF_Real_AllAddr_gfx6_gfx7_gfx10<bits<8> op> :
  MUBUF_Real_AllAddr_gfx6_gfx7<op>, MUBUF_Real_AllAddr_gfx10<op>;

multiclass MUBUF_Real_AllAddr_Lds_gfx6_gfx7_gfx10<bits<8> op> :
  MUBUF_Real_AllAddr_Lds_gfx6_gfx7<op>, MUBUF_Real_AllAddr_Lds_gfx10<op>;

multiclass MUBUF_Real_Atomics_gfx6_gfx7_gfx10<bits<8> op> :
  MUBUF_Real_Atomics_gfx6_gfx7<op>, MUBUF_Real_Atomics_gfx10<op>;

// FIXME-GFX6: Following instructions are available only on GFX6.
//defm BUFFER_ATOMIC_RSUB         : MUBUF_Real_Atomics_gfx6 <0x034>;
//defm BUFFER_ATOMIC_RSUB_X2      : MUBUF_Real_Atomics_gfx6 <0x054>;

defm BUFFER_LOAD_FORMAT_X     : MUBUF_Real_AllAddr_Lds_gfx6_gfx7_gfx10<0x000>;
defm BUFFER_LOAD_FORMAT_XY    : MUBUF_Real_AllAddr_gfx6_gfx7_gfx10<0x001>;
defm BUFFER_LOAD_FORMAT_XYZ   : MUBUF_Real_AllAddr_gfx6_gfx7_gfx10<0x002>;
defm BUFFER_LOAD_FORMAT_XYZW  : MUBUF_Real_AllAddr_gfx6_gfx7_gfx10<0x003>;
defm BUFFER_STORE_FORMAT_X    : MUBUF_Real_AllAddr_gfx6_gfx7_gfx10<0x004>;
defm BUFFER_STORE_FORMAT_XY   : MUBUF_Real_AllAddr_gfx6_gfx7_gfx10<0x005>;
defm BUFFER_STORE_FORMAT_XYZ  : MUBUF_Real_AllAddr_gfx6_gfx7_gfx10<0x006>;
defm BUFFER_STORE_FORMAT_XYZW : MUBUF_Real_AllAddr_gfx6_gfx7_gfx10<0x007>;
defm BUFFER_LOAD_UBYTE        : MUBUF_Real_AllAddr_Lds_gfx6_gfx7_gfx10<0x008>;
defm BUFFER_LOAD_SBYTE        : MUBUF_Real_AllAddr_Lds_gfx6_gfx7_gfx10<0x009>;
defm BUFFER_LOAD_USHORT       : MUBUF_Real_AllAddr_Lds_gfx6_gfx7_gfx10<0x00a>;
defm BUFFER_LOAD_SSHORT       : MUBUF_Real_AllAddr_Lds_gfx6_gfx7_gfx10<0x00b>;
defm BUFFER_LOAD_DWORD        : MUBUF_Real_AllAddr_Lds_gfx6_gfx7_gfx10<0x00c>;
defm BUFFER_LOAD_DWORDX2      : MUBUF_Real_AllAddr_gfx6_gfx7_gfx10<0x00d>;
defm BUFFER_LOAD_DWORDX4      : MUBUF_Real_AllAddr_gfx6_gfx7_gfx10<0x00e>;
defm BUFFER_LOAD_DWORDX3      : MUBUF_Real_AllAddr_gfx6_gfx7_gfx10<0x00f>;
defm BUFFER_STORE_BYTE        : MUBUF_Real_AllAddr_gfx6_gfx7_gfx10<0x018>;
defm BUFFER_STORE_SHORT       : MUBUF_Real_AllAddr_gfx6_gfx7_gfx10<0x01a>;
defm BUFFER_STORE_DWORD       : MUBUF_Real_AllAddr_gfx6_gfx7_gfx10<0x01c>;
defm BUFFER_STORE_DWORDX2     : MUBUF_Real_AllAddr_gfx6_gfx7_gfx10<0x01d>;
defm BUFFER_STORE_DWORDX4     : MUBUF_Real_AllAddr_gfx6_gfx7_gfx10<0x01e>;
defm BUFFER_STORE_DWORDX3     : MUBUF_Real_AllAddr_gfx6_gfx7_gfx10<0x01f>;

defm BUFFER_ATOMIC_SWAP        : MUBUF_Real_Atomics_gfx6_gfx7_gfx10<0x030>;
defm BUFFER_ATOMIC_CMPSWAP     : MUBUF_Real_Atomics_gfx6_gfx7_gfx10<0x031>;
defm BUFFER_ATOMIC_ADD         : MUBUF_Real_Atomics_gfx6_gfx7_gfx10<0x032>;
defm BUFFER_ATOMIC_SUB         : MUBUF_Real_Atomics_gfx6_gfx7_gfx10<0x033>;
defm BUFFER_ATOMIC_SMIN        : MUBUF_Real_Atomics_gfx6_gfx7_gfx10<0x035>;
defm BUFFER_ATOMIC_UMIN        : MUBUF_Real_Atomics_gfx6_gfx7_gfx10<0x036>;
defm BUFFER_ATOMIC_SMAX        : MUBUF_Real_Atomics_gfx6_gfx7_gfx10<0x037>;
defm BUFFER_ATOMIC_UMAX        : MUBUF_Real_Atomics_gfx6_gfx7_gfx10<0x038>;
defm BUFFER_ATOMIC_AND         : MUBUF_Real_Atomics_gfx6_gfx7_gfx10<0x039>;
defm BUFFER_ATOMIC_OR          : MUBUF_Real_Atomics_gfx6_gfx7_gfx10<0x03a>;
defm BUFFER_ATOMIC_XOR         : MUBUF_Real_Atomics_gfx6_gfx7_gfx10<0x03b>;
defm BUFFER_ATOMIC_INC         : MUBUF_Real_Atomics_gfx6_gfx7_gfx10<0x03c>;
defm BUFFER_ATOMIC_DEC         : MUBUF_Real_Atomics_gfx6_gfx7_gfx10<0x03d>;
defm BUFFER_ATOMIC_FCMPSWAP    : MUBUF_Real_Atomics_gfx6_gfx7_gfx10<0x03e>;
defm BUFFER_ATOMIC_FMIN        : MUBUF_Real_Atomics_gfx6_gfx7_gfx10<0x03f>;
defm BUFFER_ATOMIC_FMAX        : MUBUF_Real_Atomics_gfx6_gfx7_gfx10<0x040>;
defm BUFFER_ATOMIC_SWAP_X2     : MUBUF_Real_Atomics_gfx6_gfx7_gfx10<0x050>;
defm BUFFER_ATOMIC_CMPSWAP_X2  : MUBUF_Real_Atomics_gfx6_gfx7_gfx10<0x051>;
defm BUFFER_ATOMIC_ADD_X2      : MUBUF_Real_Atomics_gfx6_gfx7_gfx10<0x052>;
defm BUFFER_ATOMIC_SUB_X2      : MUBUF_Real_Atomics_gfx6_gfx7_gfx10<0x053>;
defm BUFFER_ATOMIC_SMIN_X2     : MUBUF_Real_Atomics_gfx6_gfx7_gfx10<0x055>;
defm BUFFER_ATOMIC_UMIN_X2     : MUBUF_Real_Atomics_gfx6_gfx7_gfx10<0x056>;
defm BUFFER_ATOMIC_SMAX_X2     : MUBUF_Real_Atomics_gfx6_gfx7_gfx10<0x057>;
defm BUFFER_ATOMIC_UMAX_X2     : MUBUF_Real_Atomics_gfx6_gfx7_gfx10<0x058>;
defm BUFFER_ATOMIC_AND_X2      : MUBUF_Real_Atomics_gfx6_gfx7_gfx10<0x059>;
defm BUFFER_ATOMIC_OR_X2       : MUBUF_Real_Atomics_gfx6_gfx7_gfx10<0x05a>;
defm BUFFER_ATOMIC_XOR_X2      : MUBUF_Real_Atomics_gfx6_gfx7_gfx10<0x05b>;
defm BUFFER_ATOMIC_INC_X2      : MUBUF_Real_Atomics_gfx6_gfx7_gfx10<0x05c>;
defm BUFFER_ATOMIC_DEC_X2      : MUBUF_Real_Atomics_gfx6_gfx7_gfx10<0x05d>;
// FIXME-GFX7: Need to handle hazard for BUFFER_ATOMIC_FCMPSWAP_X2 on GFX7.
defm BUFFER_ATOMIC_FCMPSWAP_X2 : MUBUF_Real_Atomics_gfx6_gfx7_gfx10<0x05e>;
defm BUFFER_ATOMIC_FMIN_X2     : MUBUF_Real_Atomics_gfx6_gfx7_gfx10<0x05f>;
defm BUFFER_ATOMIC_FMAX_X2     : MUBUF_Real_Atomics_gfx6_gfx7_gfx10<0x060>;

defm BUFFER_ATOMIC_CSUB       : MUBUF_Real_Atomics_RTN_gfx10<0x034>;

defm BUFFER_WBINVL1_SC        : MUBUF_Real_gfx6<0x070>;
defm BUFFER_WBINVL1_VOL       : MUBUF_Real_gfx7<0x070>;
def  BUFFER_WBINVL1_gfx6_gfx7 : MUBUF_Real_gfx6_gfx7<0x071, BUFFER_WBINVL1>;

//===----------------------------------------------------------------------===//
// Base ENC_MTBUF for GFX6, GFX7, GFX10.
//===----------------------------------------------------------------------===//

class Base_MTBUF_Real_gfx6_gfx7_gfx10<bits<3> op, MTBUF_Pseudo ps, int ef> :
    MTBUF_Real<ps>, Enc64, SIMCInstr<ps.PseudoInstr, ef> {
  let Inst{11-0}  = !if(ps.has_offset, offset, ?);
  let Inst{12}    = ps.offen;
  let Inst{13}    = ps.idxen;
  let Inst{14}    = !if(ps.has_glc, cpol{CPolBit.GLC}, ps.glc_value);
  let Inst{18-16} = op;
  let Inst{31-26} = 0x3a; //encoding
  let Inst{39-32} = !if(ps.has_vaddr, vaddr, ?);
  let Inst{47-40} = !if(ps.has_vdata, vdata{7-0}, ?);
  let Inst{52-48} = !if(ps.has_srsrc, srsrc{6-2}, ?);
  let Inst{54}    = !if(ps.has_slc, cpol{CPolBit.SLC}, ?);
  let Inst{55}    = !if(ps.has_tfe, tfe, ?);
  let Inst{63-56} = !if(ps.has_soffset, soffset, ?);
}

//===----------------------------------------------------------------------===//
// MTBUF - GFX10.
//===----------------------------------------------------------------------===//

class MTBUF_Real_gfx10<bits<4> op, MTBUF_Pseudo ps> :
    Base_MTBUF_Real_gfx6_gfx7_gfx10<op{2-0}, ps, SIEncodingFamily.GFX10> {
  let Inst{15} = !if(ps.has_dlc, cpol{CPolBit.DLC}, ps.dlc_value);
  let Inst{25-19} = format;
  let Inst{53} = op{3};
}

let AssemblerPredicate = isGFX10Plus, DecoderNamespace = "GFX10" in {
  multiclass MTBUF_Real_AllAddr_gfx10<bits<4> op> {
    def _BOTHEN_gfx10 :
      MTBUF_Real_gfx10<op, !cast<MTBUF_Pseudo>(NAME#"_BOTHEN")>;
    def _IDXEN_gfx10 :
      MTBUF_Real_gfx10<op, !cast<MTBUF_Pseudo>(NAME#"_IDXEN")>;
    def _OFFEN_gfx10 :
      MTBUF_Real_gfx10<op, !cast<MTBUF_Pseudo>(NAME#"_OFFEN")>;
    def _OFFSET_gfx10 :
      MTBUF_Real_gfx10<op, !cast<MTBUF_Pseudo>(NAME#"_OFFSET")>;
  }
} // End AssemblerPredicate = isGFX10Plus, DecoderNamespace = "GFX10"

defm TBUFFER_LOAD_FORMAT_D16_X     : MTBUF_Real_AllAddr_gfx10<0x008>;
defm TBUFFER_LOAD_FORMAT_D16_XY    : MTBUF_Real_AllAddr_gfx10<0x009>;
defm TBUFFER_LOAD_FORMAT_D16_XYZ   : MTBUF_Real_AllAddr_gfx10<0x00a>;
defm TBUFFER_LOAD_FORMAT_D16_XYZW  : MTBUF_Real_AllAddr_gfx10<0x00b>;
defm TBUFFER_STORE_FORMAT_D16_X    : MTBUF_Real_AllAddr_gfx10<0x00c>;
defm TBUFFER_STORE_FORMAT_D16_XY   : MTBUF_Real_AllAddr_gfx10<0x00d>;
defm TBUFFER_STORE_FORMAT_D16_XYZ  : MTBUF_Real_AllAddr_gfx10<0x00e>;
defm TBUFFER_STORE_FORMAT_D16_XYZW : MTBUF_Real_AllAddr_gfx10<0x00f>;

//===----------------------------------------------------------------------===//
// MTBUF - GFX6, GFX7, GFX10.
//===----------------------------------------------------------------------===//

class MTBUF_Real_gfx6_gfx7<bits<4> op, MTBUF_Pseudo ps> :
    Base_MTBUF_Real_gfx6_gfx7_gfx10<op{2-0}, ps, SIEncodingFamily.SI> {
  let Inst{15} = ps.addr64;
  let Inst{22-19} = dfmt;
  let Inst{25-23} = nfmt;
}

let AssemblerPredicate = isGFX6GFX7, DecoderNamespace = "GFX6GFX7" in {
  multiclass MTBUF_Real_AllAddr_gfx6_gfx7<bits<4> op> {
    def _ADDR64_gfx6_gfx7 :
      MTBUF_Real_gfx6_gfx7<op, !cast<MTBUF_Pseudo>(NAME#"_ADDR64")>;
    def _BOTHEN_gfx6_gfx7 :
      MTBUF_Real_gfx6_gfx7<op, !cast<MTBUF_Pseudo>(NAME#"_BOTHEN")>;
    def _IDXEN_gfx6_gfx7 :
      MTBUF_Real_gfx6_gfx7<op, !cast<MTBUF_Pseudo>(NAME#"_IDXEN")>;
    def _OFFEN_gfx6_gfx7 :
      MTBUF_Real_gfx6_gfx7<op, !cast<MTBUF_Pseudo>(NAME#"_OFFEN")>;
    def _OFFSET_gfx6_gfx7 :
      MTBUF_Real_gfx6_gfx7<op, !cast<MTBUF_Pseudo>(NAME#"_OFFSET")>;
  }
} // End AssemblerPredicate = isGFX6GFX7, DecoderNamespace = "GFX6GFX7"

multiclass MTBUF_Real_AllAddr_gfx6_gfx7_gfx10<bits<4> op> :
  MTBUF_Real_AllAddr_gfx6_gfx7<op>, MTBUF_Real_AllAddr_gfx10<op>;

defm TBUFFER_LOAD_FORMAT_X     : MTBUF_Real_AllAddr_gfx6_gfx7_gfx10<0x000>;
defm TBUFFER_LOAD_FORMAT_XY    : MTBUF_Real_AllAddr_gfx6_gfx7_gfx10<0x001>;
defm TBUFFER_LOAD_FORMAT_XYZ   : MTBUF_Real_AllAddr_gfx6_gfx7_gfx10<0x002>;
defm TBUFFER_LOAD_FORMAT_XYZW  : MTBUF_Real_AllAddr_gfx6_gfx7_gfx10<0x003>;
defm TBUFFER_STORE_FORMAT_X    : MTBUF_Real_AllAddr_gfx6_gfx7_gfx10<0x004>;
defm TBUFFER_STORE_FORMAT_XY   : MTBUF_Real_AllAddr_gfx6_gfx7_gfx10<0x005>;
defm TBUFFER_STORE_FORMAT_XYZ  : MTBUF_Real_AllAddr_gfx6_gfx7_gfx10<0x006>;
defm TBUFFER_STORE_FORMAT_XYZW : MTBUF_Real_AllAddr_gfx6_gfx7_gfx10<0x007>;

//===----------------------------------------------------------------------===//
// GFX8, GFX9 (VI).
//===----------------------------------------------------------------------===//

class MUBUF_Real_Base_vi <bits<7> op, MUBUF_Pseudo ps, int Enc,
                          bit has_sccb = ps.has_sccb> :
  MUBUF_Real<ps>,
  Enc64,
  SIMCInstr<ps.PseudoInstr, Enc>,
  AtomicNoRet<!subst("_RTN","",NAME), !if(ps.IsAtomicNoRet, 0,
                                        !if(ps.IsAtomicRet, 1, ?))> {

  let Inst{11-0}  = !if(ps.has_offset, offset, ?);
  let Inst{12}    = ps.offen;
  let Inst{13}    = ps.idxen;
  let Inst{14}    = !if(ps.has_glc, cpol{CPolBit.GLC}, ps.glc_value);
  let Inst{15}    = !if(has_sccb, cpol{CPolBit.SCC}, ps.sccb_value);
  let Inst{16}    = ps.lds;
  let Inst{17}    = !if(ps.has_slc, cpol{CPolBit.SLC}, ?);
  let Inst{24-18} = op;
  let Inst{31-26} = 0x38; //encoding
  let Inst{39-32} = !if(ps.has_vaddr, vaddr, ?);
  let Inst{47-40} = !if(ps.has_vdata, vdata{7-0}, ?);
  let Inst{52-48} = !if(ps.has_srsrc, srsrc{6-2}, ?);
  let Inst{63-56} = !if(ps.has_soffset, soffset, ?);
}

class MUBUF_Real_vi <bits<7> op, MUBUF_Pseudo ps, bit has_sccb = ps.has_sccb> :
  MUBUF_Real_Base_vi<op, ps, SIEncodingFamily.VI, has_sccb> {
  let AssemblerPredicate = isGFX8GFX9NotGFX90A;
  let DecoderNamespace = "GFX8";

  let Inst{55}    = !if(ps.has_tfe, tfe, ?);
}

class MUBUF_Real_gfx90a <bits<7> op, MUBUF_Pseudo ps,
                         bit has_sccb = ps.has_sccb> :
  MUBUF_Real_Base_vi<op, ps, SIEncodingFamily.GFX90A, has_sccb> {
  let AssemblerPredicate = isGFX90APlus;
  let DecoderNamespace = "GFX90A";
  let AsmString = ps.Mnemonic # !subst("$sccb", !if(has_sccb, "$sccb",""),
                                !subst("$tfe", "", ps.AsmOperands));

  let Inst{55}    = acc;
}

multiclass MUBUF_Real_vi_gfx90a<bits<7> op, MUBUF_Pseudo ps> {
  def _vi :     MUBUF_Real_vi<op, ps>;
  def _gfx90a : MUBUF_Real_gfx90a<op, ps, !and(ps.has_sccb,!not(ps.FPAtomic))>;
}

multiclass MUBUF_Real_AllAddr_vi<bits<7> op> {
  defm _OFFSET : MUBUF_Real_vi_gfx90a <op, !cast<MUBUF_Pseudo>(NAME#"_OFFSET")>;
  defm _OFFEN  : MUBUF_Real_vi_gfx90a <op, !cast<MUBUF_Pseudo>(NAME#"_OFFEN")>;
  defm _IDXEN  : MUBUF_Real_vi_gfx90a <op, !cast<MUBUF_Pseudo>(NAME#"_IDXEN")>;
  defm _BOTHEN : MUBUF_Real_vi_gfx90a <op, !cast<MUBUF_Pseudo>(NAME#"_BOTHEN")>;
}

multiclass MUBUF_Real_AllAddr_Lds_vi<bits<7> op> {

  def _OFFSET_vi : MUBUF_Real_vi <op, !cast<MUBUF_Pseudo>(NAME#"_OFFSET")>,
                   MUBUFLdsTable<0, NAME # "_OFFSET_vi">;
  def _OFFEN_vi  : MUBUF_Real_vi <op, !cast<MUBUF_Pseudo>(NAME#"_OFFEN")>,
                   MUBUFLdsTable<0, NAME # "_OFFEN_vi">;
  def _IDXEN_vi  : MUBUF_Real_vi <op, !cast<MUBUF_Pseudo>(NAME#"_IDXEN")>,
                   MUBUFLdsTable<0, NAME # "_IDXEN_vi">;
  def _BOTHEN_vi : MUBUF_Real_vi <op, !cast<MUBUF_Pseudo>(NAME#"_BOTHEN")>,
                   MUBUFLdsTable<0, NAME # "_BOTHEN_vi">;

  def _LDS_OFFSET_vi : MUBUF_Real_vi <op, !cast<MUBUF_Pseudo>(NAME#"_LDS_OFFSET")>,
                       MUBUFLdsTable<1, NAME # "_OFFSET_vi">;
  def _LDS_OFFEN_vi  : MUBUF_Real_vi <op, !cast<MUBUF_Pseudo>(NAME#"_LDS_OFFEN")>,
                       MUBUFLdsTable<1, NAME # "_OFFEN_vi">;
  def _LDS_IDXEN_vi  : MUBUF_Real_vi <op, !cast<MUBUF_Pseudo>(NAME#"_LDS_IDXEN")>,
                       MUBUFLdsTable<1, NAME # "_IDXEN_vi">;
  def _LDS_BOTHEN_vi : MUBUF_Real_vi <op, !cast<MUBUF_Pseudo>(NAME#"_LDS_BOTHEN")>,
                       MUBUFLdsTable<1, NAME # "_BOTHEN_vi">;

  def _OFFSET_gfx90a : MUBUF_Real_gfx90a <op, !cast<MUBUF_Pseudo>(NAME#"_OFFSET")>,
                   MUBUFLdsTable<0, NAME # "_OFFSET_gfx90a">;
  def _OFFEN_gfx90a  : MUBUF_Real_gfx90a <op, !cast<MUBUF_Pseudo>(NAME#"_OFFEN")>,
                   MUBUFLdsTable<0, NAME # "_OFFEN_gfx90a">;
  def _IDXEN_gfx90a  : MUBUF_Real_gfx90a <op, !cast<MUBUF_Pseudo>(NAME#"_IDXEN")>,
                   MUBUFLdsTable<0, NAME # "_IDXEN_gfx90a">;
  def _BOTHEN_gfx90a : MUBUF_Real_gfx90a <op, !cast<MUBUF_Pseudo>(NAME#"_BOTHEN")>,
                   MUBUFLdsTable<0, NAME # "_BOTHEN_gfx90a">;

  def _LDS_OFFSET_gfx90a : MUBUF_Real_gfx90a <op, !cast<MUBUF_Pseudo>(NAME#"_LDS_OFFSET")>,
                       MUBUFLdsTable<1, NAME # "_OFFSET_gfx90a">;
  def _LDS_OFFEN_gfx90a  : MUBUF_Real_gfx90a <op, !cast<MUBUF_Pseudo>(NAME#"_LDS_OFFEN")>,
                       MUBUFLdsTable<1, NAME # "_OFFEN_gfx90a">;
  def _LDS_IDXEN_gfx90a  : MUBUF_Real_gfx90a <op, !cast<MUBUF_Pseudo>(NAME#"_LDS_IDXEN")>,
                       MUBUFLdsTable<1, NAME # "_IDXEN_gfx90a">;
  def _LDS_BOTHEN_gfx90a : MUBUF_Real_gfx90a <op, !cast<MUBUF_Pseudo>(NAME#"_LDS_BOTHEN")>,
                       MUBUFLdsTable<1, NAME # "_BOTHEN_gfx90a">;
}

class MUBUF_Real_gfx80 <bits<7> op, MUBUF_Pseudo ps> :
  MUBUF_Real<ps>,
  Enc64,
  SIMCInstr<ps.PseudoInstr, SIEncodingFamily.GFX80> {
  let AssemblerPredicate=HasUnpackedD16VMem;
  let DecoderNamespace="GFX80_UNPACKED";

  let Inst{11-0}  = !if(ps.has_offset, offset, ?);
  let Inst{12}    = ps.offen;
  let Inst{13}    = ps.idxen;
  let Inst{14}    = !if(ps.has_glc, cpol{CPolBit.GLC}, ps.glc_value);
  let Inst{16}    = ps.lds;
  let Inst{17}    = !if(ps.has_slc, cpol{CPolBit.SLC}, ?);
  let Inst{24-18} = op;
  let Inst{31-26} = 0x38; //encoding
  let Inst{39-32} = !if(ps.has_vaddr, vaddr, ?);
  let Inst{47-40} = !if(ps.has_vdata, vdata{7-0}, ?);
  let Inst{52-48} = !if(ps.has_srsrc, srsrc{6-2}, ?);
  let Inst{55}    = !if(ps.has_tfe, tfe, ?);
  let Inst{63-56} = !if(ps.has_soffset, soffset, ?);
}

multiclass MUBUF_Real_AllAddr_gfx80<bits<7> op> {
  def _OFFSET_gfx80 : MUBUF_Real_gfx80 <op, !cast<MUBUF_Pseudo>(NAME#"_OFFSET")>;
  def _OFFEN_gfx80  : MUBUF_Real_gfx80 <op, !cast<MUBUF_Pseudo>(NAME#"_OFFEN")>;
  def _IDXEN_gfx80  : MUBUF_Real_gfx80 <op, !cast<MUBUF_Pseudo>(NAME#"_IDXEN")>;
  def _BOTHEN_gfx80 : MUBUF_Real_gfx80 <op, !cast<MUBUF_Pseudo>(NAME#"_BOTHEN")>;
}

multiclass MUBUF_Real_Atomic_vi<bits<7> op> :
  MUBUF_Real_AllAddr_vi<op> {
  defm _OFFSET_RTN : MUBUF_Real_vi_gfx90a <op, !cast<MUBUF_Pseudo>(NAME#"_OFFSET_RTN")>;
  defm _OFFEN_RTN  : MUBUF_Real_vi_gfx90a <op, !cast<MUBUF_Pseudo>(NAME#"_OFFEN_RTN")>;
  defm _IDXEN_RTN  : MUBUF_Real_vi_gfx90a <op, !cast<MUBUF_Pseudo>(NAME#"_IDXEN_RTN")>;
  defm _BOTHEN_RTN : MUBUF_Real_vi_gfx90a <op, !cast<MUBUF_Pseudo>(NAME#"_BOTHEN_RTN")>;
}

defm BUFFER_LOAD_FORMAT_X       : MUBUF_Real_AllAddr_Lds_vi <0x00>;
defm BUFFER_LOAD_FORMAT_XY      : MUBUF_Real_AllAddr_vi <0x01>;
defm BUFFER_LOAD_FORMAT_XYZ     : MUBUF_Real_AllAddr_vi <0x02>;
defm BUFFER_LOAD_FORMAT_XYZW    : MUBUF_Real_AllAddr_vi <0x03>;
defm BUFFER_STORE_FORMAT_X      : MUBUF_Real_AllAddr_vi <0x04>;
defm BUFFER_STORE_FORMAT_XY     : MUBUF_Real_AllAddr_vi <0x05>;
defm BUFFER_STORE_FORMAT_XYZ    : MUBUF_Real_AllAddr_vi <0x06>;
defm BUFFER_STORE_FORMAT_XYZW   : MUBUF_Real_AllAddr_vi <0x07>;
let SubtargetPredicate = HasUnpackedD16VMem in {
  defm BUFFER_LOAD_FORMAT_D16_X_gfx80       : MUBUF_Real_AllAddr_gfx80 <0x08>;
  defm BUFFER_LOAD_FORMAT_D16_XY_gfx80      : MUBUF_Real_AllAddr_gfx80 <0x09>;
  defm BUFFER_LOAD_FORMAT_D16_XYZ_gfx80     : MUBUF_Real_AllAddr_gfx80 <0x0a>;
  defm BUFFER_LOAD_FORMAT_D16_XYZW_gfx80    : MUBUF_Real_AllAddr_gfx80 <0x0b>;
  defm BUFFER_STORE_FORMAT_D16_X_gfx80      : MUBUF_Real_AllAddr_gfx80 <0x0c>;
  defm BUFFER_STORE_FORMAT_D16_XY_gfx80     : MUBUF_Real_AllAddr_gfx80 <0x0d>;
  defm BUFFER_STORE_FORMAT_D16_XYZ_gfx80    : MUBUF_Real_AllAddr_gfx80 <0x0e>;
  defm BUFFER_STORE_FORMAT_D16_XYZW_gfx80   : MUBUF_Real_AllAddr_gfx80 <0x0f>;
} // End HasUnpackedD16VMem.
let SubtargetPredicate = HasPackedD16VMem in {
  defm BUFFER_LOAD_FORMAT_D16_X       : MUBUF_Real_AllAddr_vi <0x08>;
  defm BUFFER_LOAD_FORMAT_D16_XY      : MUBUF_Real_AllAddr_vi <0x09>;
  defm BUFFER_LOAD_FORMAT_D16_XYZ     : MUBUF_Real_AllAddr_vi <0x0a>;
  defm BUFFER_LOAD_FORMAT_D16_XYZW    : MUBUF_Real_AllAddr_vi <0x0b>;
  defm BUFFER_STORE_FORMAT_D16_X      : MUBUF_Real_AllAddr_vi <0x0c>;
  defm BUFFER_STORE_FORMAT_D16_XY     : MUBUF_Real_AllAddr_vi <0x0d>;
  defm BUFFER_STORE_FORMAT_D16_XYZ    : MUBUF_Real_AllAddr_vi <0x0e>;
  defm BUFFER_STORE_FORMAT_D16_XYZW   : MUBUF_Real_AllAddr_vi <0x0f>;
} // End HasPackedD16VMem.
defm BUFFER_LOAD_UBYTE          : MUBUF_Real_AllAddr_Lds_vi <0x10>;
defm BUFFER_LOAD_SBYTE          : MUBUF_Real_AllAddr_Lds_vi <0x11>;
defm BUFFER_LOAD_USHORT         : MUBUF_Real_AllAddr_Lds_vi <0x12>;
defm BUFFER_LOAD_SSHORT         : MUBUF_Real_AllAddr_Lds_vi <0x13>;
defm BUFFER_LOAD_DWORD          : MUBUF_Real_AllAddr_Lds_vi <0x14>;
defm BUFFER_LOAD_DWORDX2        : MUBUF_Real_AllAddr_Lds_vi <0x15>;
defm BUFFER_LOAD_DWORDX3        : MUBUF_Real_AllAddr_Lds_vi <0x16>;
defm BUFFER_LOAD_DWORDX4        : MUBUF_Real_AllAddr_Lds_vi <0x17>;
defm BUFFER_STORE_BYTE          : MUBUF_Real_AllAddr_vi <0x18>;
defm BUFFER_STORE_BYTE_D16_HI   : MUBUF_Real_AllAddr_vi <0x19>;
defm BUFFER_STORE_SHORT         : MUBUF_Real_AllAddr_vi <0x1a>;
defm BUFFER_STORE_SHORT_D16_HI  : MUBUF_Real_AllAddr_vi <0x1b>;
defm BUFFER_STORE_DWORD         : MUBUF_Real_AllAddr_vi <0x1c>;
defm BUFFER_STORE_DWORDX2       : MUBUF_Real_AllAddr_vi <0x1d>;
defm BUFFER_STORE_DWORDX3       : MUBUF_Real_AllAddr_vi <0x1e>;
defm BUFFER_STORE_DWORDX4       : MUBUF_Real_AllAddr_vi <0x1f>;

defm BUFFER_LOAD_UBYTE_D16      : MUBUF_Real_AllAddr_vi <0x20>;
defm BUFFER_LOAD_UBYTE_D16_HI   : MUBUF_Real_AllAddr_vi <0x21>;
defm BUFFER_LOAD_SBYTE_D16      : MUBUF_Real_AllAddr_vi <0x22>;
defm BUFFER_LOAD_SBYTE_D16_HI   : MUBUF_Real_AllAddr_vi <0x23>;
defm BUFFER_LOAD_SHORT_D16      : MUBUF_Real_AllAddr_vi <0x24>;
defm BUFFER_LOAD_SHORT_D16_HI   : MUBUF_Real_AllAddr_vi <0x25>;

defm BUFFER_LOAD_FORMAT_D16_HI_X  : MUBUF_Real_AllAddr_vi <0x26>;
defm BUFFER_STORE_FORMAT_D16_HI_X : MUBUF_Real_AllAddr_vi <0x27>;

defm BUFFER_ATOMIC_SWAP         : MUBUF_Real_Atomic_vi <0x40>;
defm BUFFER_ATOMIC_CMPSWAP      : MUBUF_Real_Atomic_vi <0x41>;
defm BUFFER_ATOMIC_ADD          : MUBUF_Real_Atomic_vi <0x42>;
defm BUFFER_ATOMIC_SUB          : MUBUF_Real_Atomic_vi <0x43>;
defm BUFFER_ATOMIC_SMIN         : MUBUF_Real_Atomic_vi <0x44>;
defm BUFFER_ATOMIC_UMIN         : MUBUF_Real_Atomic_vi <0x45>;
defm BUFFER_ATOMIC_SMAX         : MUBUF_Real_Atomic_vi <0x46>;
defm BUFFER_ATOMIC_UMAX         : MUBUF_Real_Atomic_vi <0x47>;
defm BUFFER_ATOMIC_AND          : MUBUF_Real_Atomic_vi <0x48>;
defm BUFFER_ATOMIC_OR           : MUBUF_Real_Atomic_vi <0x49>;
defm BUFFER_ATOMIC_XOR          : MUBUF_Real_Atomic_vi <0x4a>;
defm BUFFER_ATOMIC_INC          : MUBUF_Real_Atomic_vi <0x4b>;
defm BUFFER_ATOMIC_DEC          : MUBUF_Real_Atomic_vi <0x4c>;

defm BUFFER_ATOMIC_SWAP_X2      : MUBUF_Real_Atomic_vi <0x60>;
defm BUFFER_ATOMIC_CMPSWAP_X2   : MUBUF_Real_Atomic_vi <0x61>;
defm BUFFER_ATOMIC_ADD_X2       : MUBUF_Real_Atomic_vi <0x62>;
defm BUFFER_ATOMIC_SUB_X2       : MUBUF_Real_Atomic_vi <0x63>;
defm BUFFER_ATOMIC_SMIN_X2      : MUBUF_Real_Atomic_vi <0x64>;
defm BUFFER_ATOMIC_UMIN_X2      : MUBUF_Real_Atomic_vi <0x65>;
defm BUFFER_ATOMIC_SMAX_X2      : MUBUF_Real_Atomic_vi <0x66>;
defm BUFFER_ATOMIC_UMAX_X2      : MUBUF_Real_Atomic_vi <0x67>;
defm BUFFER_ATOMIC_AND_X2       : MUBUF_Real_Atomic_vi <0x68>;
defm BUFFER_ATOMIC_OR_X2        : MUBUF_Real_Atomic_vi <0x69>;
defm BUFFER_ATOMIC_XOR_X2       : MUBUF_Real_Atomic_vi <0x6a>;
defm BUFFER_ATOMIC_INC_X2       : MUBUF_Real_Atomic_vi <0x6b>;
defm BUFFER_ATOMIC_DEC_X2       : MUBUF_Real_Atomic_vi <0x6c>;

defm BUFFER_STORE_LDS_DWORD     : MUBUF_Real_vi_gfx90a <0x3d, BUFFER_STORE_LDS_DWORD>;

let AssemblerPredicate = isGFX8GFX9 in {
def BUFFER_WBINVL1_vi           : MUBUF_Real_vi <0x3e, BUFFER_WBINVL1>;
def BUFFER_WBINVL1_VOL_vi       : MUBUF_Real_vi <0x3f, BUFFER_WBINVL1_VOL>;
} // End AssemblerPredicate = isGFX8GFX9

let SubtargetPredicate = HasAtomicFaddInsts in {

defm BUFFER_ATOMIC_ADD_F32    : MUBUF_Real_Atomic_vi <0x4d>;
defm BUFFER_ATOMIC_PK_ADD_F16 : MUBUF_Real_Atomic_vi <0x4e>;

} // End SubtargetPredicate = HasAtomicFaddInsts

let SubtargetPredicate = isGFX90APlus in {
  defm BUFFER_ATOMIC_ADD_F64 : MUBUF_Real_Atomic_vi<0x4f>;
  defm BUFFER_ATOMIC_MIN_F64 : MUBUF_Real_Atomic_vi<0x50>;
  defm BUFFER_ATOMIC_MAX_F64 : MUBUF_Real_Atomic_vi<0x51>;
} // End SubtargetPredicate = isGFX90APlus, AssemblerPredicate = isGFX90APlus

def BUFFER_WBL2_gfx90a  : MUBUF_Real_gfx90a<0x28, BUFFER_WBL2> {
}
def BUFFER_INVL2_gfx90a : MUBUF_Real_gfx90a<0x29, BUFFER_INVL2>;

class MTBUF_Real_Base_vi <bits<4> op, MTBUF_Pseudo ps, int Enc> :
  MTBUF_Real<ps>,
  Enc64,
  SIMCInstr<ps.PseudoInstr, Enc> {

  let Inst{11-0}  = !if(ps.has_offset, offset, ?);
  let Inst{12}    = ps.offen;
  let Inst{13}    = ps.idxen;
  let Inst{14}    = !if(ps.has_glc, cpol{CPolBit.GLC}, ps.glc_value);
  let Inst{18-15} = op;
  let Inst{22-19} = dfmt;
  let Inst{25-23} = nfmt;
  let Inst{31-26} = 0x3a; //encoding
  let Inst{39-32} = !if(ps.has_vaddr, vaddr, ?);
  let Inst{47-40} = !if(ps.has_vdata, vdata{7-0}, ?);
  let Inst{52-48} = !if(ps.has_srsrc, srsrc{6-2}, ?);
  let Inst{53}    = !if(ps.has_sccb, cpol{CPolBit.SCC}, ps.sccb_value);
  let Inst{54}    = !if(ps.has_slc, cpol{CPolBit.SLC}, ?);
  let Inst{55}    = !if(ps.has_tfe, tfe, ?);
  let Inst{63-56} = !if(ps.has_soffset, soffset, ?);
}

class MTBUF_Real_vi <bits<4> op, MTBUF_Pseudo ps> :
  MTBUF_Real_Base_vi <op, ps, SIEncodingFamily.VI> {
  let AssemblerPredicate = isGFX8GFX9NotGFX90A;
  let DecoderNamespace = "GFX8";

  let Inst{55}    = !if(ps.has_tfe, tfe, ?);
}

class MTBUF_Real_gfx90a <bits<4> op, MTBUF_Pseudo ps> :
  MTBUF_Real_Base_vi <op, ps, SIEncodingFamily.GFX90A> {
  let AssemblerPredicate = isGFX90APlus;
  let DecoderNamespace = "GFX90A";
  let AsmString = ps.Mnemonic # !subst("$tfe", "", ps.AsmOperands);

  let Inst{55}    = acc;
}

multiclass MTBUF_Real_vi_gfx90a<bits<4> op, MTBUF_Pseudo ps> {
  def _vi :     MTBUF_Real_vi<op, ps>;
  def _gfx90a : MTBUF_Real_gfx90a<op, ps>;
}

multiclass MTBUF_Real_AllAddr_vi<bits<4> op> {
  defm _OFFSET : MTBUF_Real_vi_gfx90a <op, !cast<MTBUF_Pseudo>(NAME#"_OFFSET")>;
  defm _OFFEN  : MTBUF_Real_vi_gfx90a <op, !cast<MTBUF_Pseudo>(NAME#"_OFFEN")>;
  defm _IDXEN  : MTBUF_Real_vi_gfx90a <op, !cast<MTBUF_Pseudo>(NAME#"_IDXEN")>;
  defm _BOTHEN : MTBUF_Real_vi_gfx90a <op, !cast<MTBUF_Pseudo>(NAME#"_BOTHEN")>;
}

class MTBUF_Real_gfx80 <bits<4> op, MTBUF_Pseudo ps> :
  MTBUF_Real<ps>,
  Enc64,
  SIMCInstr<ps.PseudoInstr, SIEncodingFamily.GFX80> {
  let AssemblerPredicate=HasUnpackedD16VMem;
  let DecoderNamespace="GFX80_UNPACKED";

  let Inst{11-0}  = !if(ps.has_offset, offset, ?);
  let Inst{12}    = ps.offen;
  let Inst{13}    = ps.idxen;
  let Inst{14}    = !if(ps.has_glc, cpol{CPolBit.GLC}, ps.glc_value);
  let Inst{18-15} = op;
  let Inst{22-19} = dfmt;
  let Inst{25-23} = nfmt;
  let Inst{31-26} = 0x3a; //encoding
  let Inst{39-32} = !if(ps.has_vaddr, vaddr, ?);
  let Inst{47-40} = !if(ps.has_vdata, vdata{7-0}, ?);
  let Inst{52-48} = !if(ps.has_srsrc, srsrc{6-2}, ?);
  let Inst{54}    = !if(ps.has_slc, cpol{CPolBit.SLC}, ?);
  let Inst{55}    = !if(ps.has_tfe, tfe, ?);
  let Inst{63-56} = !if(ps.has_soffset, soffset, ?);
}

multiclass MTBUF_Real_AllAddr_gfx80<bits<4> op> {
  def _OFFSET_gfx80 : MTBUF_Real_gfx80 <op, !cast<MTBUF_Pseudo>(NAME#"_OFFSET")>;
  def _OFFEN_gfx80  : MTBUF_Real_gfx80 <op, !cast<MTBUF_Pseudo>(NAME#"_OFFEN")>;
  def _IDXEN_gfx80  : MTBUF_Real_gfx80 <op, !cast<MTBUF_Pseudo>(NAME#"_IDXEN")>;
  def _BOTHEN_gfx80 : MTBUF_Real_gfx80 <op, !cast<MTBUF_Pseudo>(NAME#"_BOTHEN")>;
}

defm TBUFFER_LOAD_FORMAT_X     : MTBUF_Real_AllAddr_vi <0x00>;
defm TBUFFER_LOAD_FORMAT_XY    : MTBUF_Real_AllAddr_vi <0x01>;
defm TBUFFER_LOAD_FORMAT_XYZ   : MTBUF_Real_AllAddr_vi <0x02>;
defm TBUFFER_LOAD_FORMAT_XYZW  : MTBUF_Real_AllAddr_vi <0x03>;
defm TBUFFER_STORE_FORMAT_X    : MTBUF_Real_AllAddr_vi <0x04>;
defm TBUFFER_STORE_FORMAT_XY   : MTBUF_Real_AllAddr_vi <0x05>;
defm TBUFFER_STORE_FORMAT_XYZ  : MTBUF_Real_AllAddr_vi <0x06>;
defm TBUFFER_STORE_FORMAT_XYZW : MTBUF_Real_AllAddr_vi <0x07>;
let SubtargetPredicate = HasUnpackedD16VMem in {
  defm TBUFFER_LOAD_FORMAT_D16_X_gfx80     : MTBUF_Real_AllAddr_gfx80 <0x08>;
  defm TBUFFER_LOAD_FORMAT_D16_XY_gfx80    : MTBUF_Real_AllAddr_gfx80 <0x09>;
  defm TBUFFER_LOAD_FORMAT_D16_XYZ_gfx80   : MTBUF_Real_AllAddr_gfx80 <0x0a>;
  defm TBUFFER_LOAD_FORMAT_D16_XYZW_gfx80  : MTBUF_Real_AllAddr_gfx80 <0x0b>;
  defm TBUFFER_STORE_FORMAT_D16_X_gfx80    : MTBUF_Real_AllAddr_gfx80 <0x0c>;
  defm TBUFFER_STORE_FORMAT_D16_XY_gfx80   : MTBUF_Real_AllAddr_gfx80 <0x0d>;
  defm TBUFFER_STORE_FORMAT_D16_XYZ_gfx80  : MTBUF_Real_AllAddr_gfx80 <0x0e>;
  defm TBUFFER_STORE_FORMAT_D16_XYZW_gfx80 : MTBUF_Real_AllAddr_gfx80 <0x0f>;
} // End HasUnpackedD16VMem.
let SubtargetPredicate = HasPackedD16VMem in {
  defm TBUFFER_LOAD_FORMAT_D16_X     : MTBUF_Real_AllAddr_vi <0x08>;
  defm TBUFFER_LOAD_FORMAT_D16_XY    : MTBUF_Real_AllAddr_vi <0x09>;
  defm TBUFFER_LOAD_FORMAT_D16_XYZ   : MTBUF_Real_AllAddr_vi <0x0a>;
  defm TBUFFER_LOAD_FORMAT_D16_XYZW  : MTBUF_Real_AllAddr_vi <0x0b>;
  defm TBUFFER_STORE_FORMAT_D16_X    : MTBUF_Real_AllAddr_vi <0x0c>;
  defm TBUFFER_STORE_FORMAT_D16_XY   : MTBUF_Real_AllAddr_vi <0x0d>;
  defm TBUFFER_STORE_FORMAT_D16_XYZ  : MTBUF_Real_AllAddr_vi <0x0e>;
  defm TBUFFER_STORE_FORMAT_D16_XYZW : MTBUF_Real_AllAddr_vi <0x0f>;
} // End HasUnpackedD16VMem.

def MUBUFInfoTable : GenericTable {
  let FilterClass = "MUBUF_Pseudo";
  let CppTypeName = "MUBUFInfo";
  let Fields = ["Opcode", "BaseOpcode", "elements", "has_vaddr", "has_srsrc", "has_soffset"];

  let PrimaryKey = ["Opcode"];
  let PrimaryKeyName = "getMUBUFOpcodeHelper";
}

def getMUBUFInfoFromOpcode : SearchIndex {
  let Table = MUBUFInfoTable;
  let Key = ["Opcode"];
}

def getMUBUFInfoFromBaseOpcodeAndElements : SearchIndex {
  let Table = MUBUFInfoTable;
  let Key = ["BaseOpcode", "elements"];
}

def MTBUFInfoTable : GenericTable {
  let FilterClass = "MTBUF_Pseudo";
  let CppTypeName = "MTBUFInfo";
  let Fields = ["Opcode", "BaseOpcode", "elements", "has_vaddr", "has_srsrc", "has_soffset"];

  let PrimaryKey = ["Opcode"];
  let PrimaryKeyName = "getMTBUFOpcodeHelper";
}

def getMTBUFInfoFromOpcode : SearchIndex {
  let Table = MTBUFInfoTable;
  let Key = ["Opcode"];
}

def getMTBUFInfoFromBaseOpcodeAndElements : SearchIndex {
  let Table = MTBUFInfoTable;
  let Key = ["BaseOpcode", "elements"];
}
