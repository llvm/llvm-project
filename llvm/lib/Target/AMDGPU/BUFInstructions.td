//===-- BUFInstructions.td - Buffer Instruction Defintions ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

def MUBUFAddr32 : ComplexPattern<i64, 9, "SelectMUBUFAddr32">;
def MUBUFAddr64 : ComplexPattern<i64, 7, "SelectMUBUFAddr64">;
def MUBUFAddr64Atomic : ComplexPattern<i64, 5, "SelectMUBUFAddr64">;

def MUBUFScratch : ComplexPattern<i64, 4, "SelectMUBUFScratch">;
def MUBUFOffset : ComplexPattern<i64, 6, "SelectMUBUFOffset">;
def MUBUFOffsetNoGLC : ComplexPattern<i64, 3, "SelectMUBUFOffset">;
def MUBUFOffsetAtomic : ComplexPattern<i64, 4, "SelectMUBUFOffset">;
def MUBUFIntrinsicOffset : ComplexPattern<i32, 2, "SelectMUBUFIntrinsicOffset">;
def MUBUFIntrinsicVOffset : ComplexPattern<i32, 3, "SelectMUBUFIntrinsicVOffset">;

class MubufLoad <SDPatternOperator op> : PatFrag <
  (ops node:$ptr), (op node:$ptr), [{
  auto const AS = cast<MemSDNode>(N)->getAddressSpace();
  return AS == AMDGPUAS::GLOBAL_ADDRESS ||
         AS == AMDGPUAS::CONSTANT_ADDRESS;
}]>;

def mubuf_load          : MubufLoad <load>;
def mubuf_az_extloadi8  : MubufLoad <az_extloadi8>;
def mubuf_sextloadi8    : MubufLoad <sextloadi8>;
def mubuf_az_extloadi16 : MubufLoad <az_extloadi16>;
def mubuf_sextloadi16   : MubufLoad <sextloadi16>;
def mubuf_load_atomic   : MubufLoad <atomic_load>;

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

class MUBUFAddr64Table <bit is_addr64, string suffix = ""> {
  bit IsAddr64 = is_addr64;
  string OpName = NAME # suffix;
}

//===----------------------------------------------------------------------===//
// MTBUF classes
//===----------------------------------------------------------------------===//

class MTBUF_Pseudo <string opName, dag outs, dag ins,
                    string asmOps, list<dag> pattern=[]> :
  InstSI<outs, ins, "", pattern>,
  SIMCInstr<opName, SIEncodingFamily.NONE> {

  let isPseudo = 1;
  let isCodeGenOnly = 1;
  let UseNamedOperandTable = 1;

  string Mnemonic = opName;
  string AsmOperands = asmOps;

  let VM_CNT = 1;
  let EXP_CNT = 1;
  let MTBUF = 1;
  let Uses = [EXEC];

  let hasSideEffects = 0;
  let UseNamedOperandTable = 1;
  let SchedRW = [WriteVMEM];
}

class MTBUF_Real <MTBUF_Pseudo ps> :
  InstSI <ps.OutOperandList, ps.InOperandList, ps.Mnemonic # ps.AsmOperands, []>,
  Enc64 {

  let isPseudo = 0;
  let isCodeGenOnly = 0;

  // copy relevant pseudo op flags
  let SubtargetPredicate = ps.SubtargetPredicate;
  let AsmMatchConverter  = ps.AsmMatchConverter;
  let Constraints        = ps.Constraints;
  let DisableEncoding    = ps.DisableEncoding;
  let TSFlags            = ps.TSFlags;

  bits<8> vdata;
  bits<12> offset;
  bits<1> offen;
  bits<1> idxen;
  bits<1> glc;
  bits<1> addr64;
  bits<4> dfmt;
  bits<3> nfmt;
  bits<8> vaddr;
  bits<7> srsrc;
  bits<1> slc;
  bits<1> tfe;
  bits<8> soffset;

  let Inst{11-0}  = offset;
  let Inst{12}    = offen;
  let Inst{13}    = idxen;
  let Inst{14}    = glc;
  let Inst{22-19} = dfmt;
  let Inst{25-23} = nfmt;
  let Inst{31-26} = 0x3a; //encoding
  let Inst{39-32} = vaddr;
  let Inst{47-40} = vdata;
  let Inst{52-48} = srsrc{6-2};
  let Inst{54}    = slc;
  let Inst{55}    = tfe;
  let Inst{63-56} = soffset;
}

class MTBUF_Load_Pseudo <string opName, RegisterClass regClass> : MTBUF_Pseudo <
  opName, (outs regClass:$dst),
  (ins u16imm:$offset, i1imm:$offen, i1imm:$idxen, i1imm:$glc, i1imm:$addr64,
       i8imm:$dfmt, i8imm:$nfmt, VGPR_32:$vaddr, SReg_128:$srsrc,
       i1imm:$slc, i1imm:$tfe, SCSrc_b32:$soffset),
  " $dst, $offset, $offen, $idxen, $glc, $addr64, $dfmt,"#
  " $nfmt, $vaddr, $srsrc, $slc, $tfe, $soffset"
> {
  let mayLoad = 1;
  let mayStore = 0;
}

class MTBUF_Store_Pseudo <string opName, RegisterClass regClass> : MTBUF_Pseudo <
  opName, (outs),
  (ins regClass:$vdata, u16imm:$offset, i1imm:$offen, i1imm:$idxen, i1imm:$glc,
       i1imm:$addr64, i8imm:$dfmt, i8imm:$nfmt, VGPR_32:$vaddr,
       SReg_128:$srsrc, i1imm:$slc, i1imm:$tfe, SCSrc_b32:$soffset),
  " $vdata, $offset, $offen, $idxen, $glc, $addr64, $dfmt,"#
  " $nfmt, $vaddr, $srsrc, $slc, $tfe, $soffset"
> {
  let mayLoad = 0;
  let mayStore = 1;
}

//===----------------------------------------------------------------------===//
// MUBUF classes
//===----------------------------------------------------------------------===//

class MUBUF_Pseudo <string opName, dag outs, dag ins,
                    string asmOps, list<dag> pattern=[]> :
  InstSI<outs, ins, "", pattern>,
  SIMCInstr<opName, SIEncodingFamily.NONE> {

  let isPseudo = 1;
  let isCodeGenOnly = 1;
  let UseNamedOperandTable = 1;

  string Mnemonic = opName;
  string AsmOperands = asmOps;

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
  bits<1> has_vdata   = 1;
  bits<1> has_vaddr   = 1;
  bits<1> has_glc     = 1;
  bits<1> glc_value   = 0; // the value for glc if no such operand
  bits<1> has_srsrc   = 1;
  bits<1> has_soffset = 1;
  bits<1> has_offset  = 1;
  bits<1> has_slc     = 1;
  bits<1> has_tfe     = 1;
}

class MUBUF_Real <bits<7> op, MUBUF_Pseudo ps> :
  InstSI <ps.OutOperandList, ps.InOperandList, ps.Mnemonic # ps.AsmOperands, []> {

  let isPseudo = 0;
  let isCodeGenOnly = 0;

  // copy relevant pseudo op flags
  let SubtargetPredicate = ps.SubtargetPredicate;
  let AsmMatchConverter  = ps.AsmMatchConverter;
  let Constraints        = ps.Constraints;
  let DisableEncoding    = ps.DisableEncoding;
  let TSFlags            = ps.TSFlags;

  bits<12> offset;
  bits<1>  glc;
  bits<1>  lds = 0;
  bits<8>  vaddr;
  bits<8>  vdata;
  bits<7>  srsrc;
  bits<1>  slc;
  bits<1>  tfe;
  bits<8>  soffset;
}


// For cache invalidation instructions.
class MUBUF_Invalidate <string opName, SDPatternOperator node> :
  MUBUF_Pseudo<opName, (outs), (ins), "", [(node)]> {

  let AsmMatchConverter = "";

  let hasSideEffects = 1;
  let mayStore = 1;

  // Set everything to 0.
  let offen       = 0;
  let idxen       = 0;
  let addr64      = 0;
  let has_vdata   = 0;
  let has_vaddr   = 0;
  let has_glc     = 0;
  let glc_value   = 0;
  let has_srsrc   = 0;
  let has_soffset = 0;
  let has_offset  = 0;
  let has_slc     = 0;
  let has_tfe     = 0;
}

class getMUBUFInsDA<list<RegisterClass> vdataList,
                    list<RegisterClass> vaddrList=[]> {
  RegisterClass vdataClass = !if(!empty(vdataList), ?, !head(vdataList));
  RegisterClass vaddrClass = !if(!empty(vaddrList), ?, !head(vaddrList));
  dag InsNoData = !if(!empty(vaddrList),
    (ins                    SReg_128:$srsrc, SCSrc_b32:$soffset,
         offset:$offset, glc:$glc, slc:$slc, tfe:$tfe),
    (ins vaddrClass:$vaddr, SReg_128:$srsrc, SCSrc_b32:$soffset,
         offset:$offset, glc:$glc, slc:$slc, tfe:$tfe)
  );
  dag InsData = !if(!empty(vaddrList),
    (ins vdataClass:$vdata,                    SReg_128:$srsrc,
         SCSrc_b32:$soffset, offset:$offset, glc:$glc, slc:$slc, tfe:$tfe),
    (ins vdataClass:$vdata, vaddrClass:$vaddr, SReg_128:$srsrc,
         SCSrc_b32:$soffset, offset:$offset, glc:$glc, slc:$slc, tfe:$tfe)
  );
  dag ret = !if(!empty(vdataList), InsNoData, InsData);
}

class getMUBUFIns<int addrKind, list<RegisterClass> vdataList=[]> {
  dag ret =
    !if(!eq(addrKind, BUFAddrKind.Offset), getMUBUFInsDA<vdataList>.ret,
    !if(!eq(addrKind, BUFAddrKind.OffEn),  getMUBUFInsDA<vdataList, [VGPR_32]>.ret,
    !if(!eq(addrKind, BUFAddrKind.IdxEn),  getMUBUFInsDA<vdataList, [VGPR_32]>.ret,
    !if(!eq(addrKind, BUFAddrKind.BothEn), getMUBUFInsDA<vdataList, [VReg_64]>.ret,
    !if(!eq(addrKind, BUFAddrKind.Addr64), getMUBUFInsDA<vdataList, [VReg_64]>.ret,
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
  bits<1> offen  = !if(!eq(addrKind, BUFAddrKind.OffEn), 1,
                   !if(!eq(addrKind, BUFAddrKind.BothEn), 1 , 0));

  bits<1> idxen  = !if(!eq(addrKind, BUFAddrKind.IdxEn), 1,
                   !if(!eq(addrKind, BUFAddrKind.BothEn), 1 , 0));

  bits<1> addr64 = !if(!eq(addrKind, BUFAddrKind.Addr64), 1, 0);

  bits<1> has_vaddr = !if(!eq(addrKind, BUFAddrKind.Offset), 0, 1);
}

class MUBUF_Load_Pseudo <string opName,
                         int addrKind,
                         RegisterClass vdataClass,
                         list<dag> pattern=[],
                         // Workaround bug bz30254
                         int addrKindCopy = addrKind>
  : MUBUF_Pseudo<opName,
                 (outs vdataClass:$vdata),
                 getMUBUFIns<addrKindCopy>.ret,
                 " $vdata, " # getMUBUFAsmOps<addrKindCopy>.ret # "$glc$slc$tfe",
                 pattern>,
    MUBUF_SetupAddr<addrKindCopy> {
  let PseudoInstr = opName # "_" # getAddrName<addrKindCopy>.ret;
  let mayLoad = 1;
  let mayStore = 0;
}

// FIXME: tfe can't be an operand because it requires a separate
// opcode because it needs an N+1 register class dest register.
multiclass MUBUF_Pseudo_Loads<string opName, RegisterClass vdataClass,
                              ValueType load_vt = i32,
                              SDPatternOperator ld = null_frag> {

  def _OFFSET : MUBUF_Load_Pseudo <opName, BUFAddrKind.Offset, vdataClass,
    [(set load_vt:$vdata,
     (ld (MUBUFOffset v4i32:$srsrc, i32:$soffset, i16:$offset, i1:$glc, i1:$slc, i1:$tfe)))]>,
    MUBUFAddr64Table<0>;

  def _ADDR64 : MUBUF_Load_Pseudo <opName, BUFAddrKind.Addr64, vdataClass,
    [(set load_vt:$vdata,
     (ld (MUBUFAddr64 v4i32:$srsrc, i64:$vaddr, i32:$soffset, i16:$offset, i1:$glc, i1:$slc, i1:$tfe)))]>,
    MUBUFAddr64Table<1>;

  def _OFFEN  : MUBUF_Load_Pseudo <opName, BUFAddrKind.OffEn, vdataClass>;
  def _IDXEN  : MUBUF_Load_Pseudo <opName, BUFAddrKind.IdxEn, vdataClass>;
  def _BOTHEN : MUBUF_Load_Pseudo <opName, BUFAddrKind.BothEn, vdataClass>;

  let DisableWQM = 1 in {
    def _OFFSET_exact : MUBUF_Load_Pseudo <opName, BUFAddrKind.Offset, vdataClass>;
    def _OFFEN_exact  : MUBUF_Load_Pseudo <opName, BUFAddrKind.OffEn, vdataClass>;
    def _IDXEN_exact  : MUBUF_Load_Pseudo <opName, BUFAddrKind.IdxEn, vdataClass>;
    def _BOTHEN_exact : MUBUF_Load_Pseudo <opName, BUFAddrKind.BothEn, vdataClass>;
  }
}

class MUBUF_Store_Pseudo <string opName,
                          int addrKind,
                          RegisterClass vdataClass,
                          list<dag> pattern=[],
                          // Workaround bug bz30254
                          int addrKindCopy = addrKind,
                          RegisterClass vdataClassCopy = vdataClass>
  : MUBUF_Pseudo<opName,
                 (outs),
                 getMUBUFIns<addrKindCopy, [vdataClassCopy]>.ret,
                 " $vdata, " # getMUBUFAsmOps<addrKindCopy>.ret # "$glc$slc$tfe",
                 pattern>,
    MUBUF_SetupAddr<addrKindCopy> {
  let PseudoInstr = opName # "_" # getAddrName<addrKindCopy>.ret;
  let mayLoad = 0;
  let mayStore = 1;
}

multiclass MUBUF_Pseudo_Stores<string opName, RegisterClass vdataClass,
                               ValueType store_vt = i32,
                               SDPatternOperator st = null_frag> {

  def _OFFSET : MUBUF_Store_Pseudo <opName, BUFAddrKind.Offset, vdataClass,
    [(st store_vt:$vdata, (MUBUFOffset v4i32:$srsrc, i32:$soffset,
                                       i16:$offset, i1:$glc, i1:$slc, i1:$tfe))]>,
    MUBUFAddr64Table<0>;

  def _ADDR64 : MUBUF_Store_Pseudo <opName, BUFAddrKind.Addr64, vdataClass,
    [(st store_vt:$vdata, (MUBUFAddr64 v4i32:$srsrc, i64:$vaddr, i32:$soffset,
                                       i16:$offset, i1:$glc, i1:$slc, i1:$tfe))]>,
    MUBUFAddr64Table<1>;

  def _OFFEN  : MUBUF_Store_Pseudo <opName, BUFAddrKind.OffEn, vdataClass>;
  def _IDXEN  : MUBUF_Store_Pseudo <opName, BUFAddrKind.IdxEn, vdataClass>;
  def _BOTHEN : MUBUF_Store_Pseudo <opName, BUFAddrKind.BothEn, vdataClass>;

  let DisableWQM = 1 in {
    def _OFFSET_exact : MUBUF_Store_Pseudo <opName, BUFAddrKind.Offset, vdataClass>;
    def _OFFEN_exact  : MUBUF_Store_Pseudo <opName, BUFAddrKind.OffEn, vdataClass>;
    def _IDXEN_exact  : MUBUF_Store_Pseudo <opName, BUFAddrKind.IdxEn, vdataClass>;
    def _BOTHEN_exact : MUBUF_Store_Pseudo <opName, BUFAddrKind.BothEn, vdataClass>;
  }
}


class getMUBUFAtomicInsDA<RegisterClass vdataClass, bit vdata_in,
                          list<RegisterClass> vaddrList=[]> {
  RegisterClass vaddrClass = !if(!empty(vaddrList), ?, !head(vaddrList));
  dag ret = !if(vdata_in,
    !if(!empty(vaddrList),
      (ins vdataClass:$vdata_in,
           SReg_128:$srsrc, SCSrc_b32:$soffset, offset:$offset, slc:$slc),
      (ins vdataClass:$vdata_in, vaddrClass:$vaddr,
           SReg_128:$srsrc, SCSrc_b32:$soffset, offset:$offset, slc:$slc)
    ),
    !if(!empty(vaddrList),
      (ins vdataClass:$vdata,
           SReg_128:$srsrc, SCSrc_b32:$soffset, offset:$offset, slc:$slc),
      (ins vdataClass:$vdata, vaddrClass:$vaddr,
           SReg_128:$srsrc, SCSrc_b32:$soffset, offset:$offset, slc:$slc)
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
  let has_tfe = 0;
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
                        " $vdata, " # getMUBUFAsmOps<addrKindCopy>.ret # "$slc",
                        pattern>,
    AtomicNoRet<opName # "_" # getAddrName<addrKindCopy>.ret, 0> {
  let PseudoInstr = opName # "_" # getAddrName<addrKindCopy>.ret;
  let glc_value = 0;
  let AsmMatchConverter = "cvtMubufAtomic";
}

class MUBUF_AtomicRet_Pseudo<string opName, int addrKind,
                             RegisterClass vdataClass,
                             list<dag> pattern=[],
                             // Workaround bug bz30254
                             int addrKindCopy = addrKind,
                             RegisterClass vdataClassCopy = vdataClass>
  : MUBUF_Atomic_Pseudo<opName, addrKindCopy,
                        (outs vdataClassCopy:$vdata),
                        getMUBUFAtomicIns<addrKindCopy, vdataClassCopy, 1>.ret,
                        " $vdata, " # getMUBUFAsmOps<addrKindCopy>.ret # " glc$slc",
                        pattern>,
    AtomicNoRet<opName # "_" # getAddrName<addrKindCopy>.ret, 1> {
  let PseudoInstr = opName # "_rtn_" # getAddrName<addrKindCopy>.ret;
  let glc_value = 1;
  let Constraints = "$vdata = $vdata_in";
  let DisableEncoding = "$vdata_in";
  let AsmMatchConverter = "cvtMubufAtomicReturn";
}

multiclass MUBUF_Pseudo_Atomics <string opName,
                                 RegisterClass vdataClass,
                                 ValueType vdataType,
                                 SDPatternOperator atomic> {

  def _OFFSET : MUBUF_AtomicNoRet_Pseudo <opName, BUFAddrKind.Offset, vdataClass>,
                MUBUFAddr64Table <0>;
  def _ADDR64 : MUBUF_AtomicNoRet_Pseudo <opName, BUFAddrKind.Addr64, vdataClass>,
                MUBUFAddr64Table <1>;
  def _OFFEN  : MUBUF_AtomicNoRet_Pseudo <opName, BUFAddrKind.OffEn,  vdataClass>;
  def _IDXEN  : MUBUF_AtomicNoRet_Pseudo <opName, BUFAddrKind.IdxEn,  vdataClass>;
  def _BOTHEN : MUBUF_AtomicNoRet_Pseudo <opName, BUFAddrKind.BothEn, vdataClass>;

  def _RTN_OFFSET : MUBUF_AtomicRet_Pseudo <opName, BUFAddrKind.Offset, vdataClass,
    [(set vdataType:$vdata,
     (atomic (MUBUFOffsetAtomic v4i32:$srsrc, i32:$soffset, i16:$offset, i1:$slc),
             vdataType:$vdata_in))]>,
    MUBUFAddr64Table <0, "_RTN">;

  def _RTN_ADDR64 : MUBUF_AtomicRet_Pseudo <opName, BUFAddrKind.Addr64, vdataClass,
    [(set vdataType:$vdata,
     (atomic (MUBUFAddr64Atomic v4i32:$srsrc, i64:$vaddr, i32:$soffset, i16:$offset, i1:$slc),
             vdataType:$vdata_in))]>,
    MUBUFAddr64Table <1, "_RTN">;

  def _RTN_OFFEN  : MUBUF_AtomicRet_Pseudo <opName, BUFAddrKind.OffEn,  vdataClass>;
  def _RTN_IDXEN  : MUBUF_AtomicRet_Pseudo <opName, BUFAddrKind.IdxEn,  vdataClass>;
  def _RTN_BOTHEN : MUBUF_AtomicRet_Pseudo <opName, BUFAddrKind.BothEn, vdataClass>;
}


//===----------------------------------------------------------------------===//
// MUBUF Instructions
//===----------------------------------------------------------------------===//

let SubtargetPredicate = isGCN in {

defm BUFFER_LOAD_FORMAT_X : MUBUF_Pseudo_Loads <
  "buffer_load_format_x", VGPR_32
>;
defm BUFFER_LOAD_FORMAT_XY : MUBUF_Pseudo_Loads <
  "buffer_load_format_xy", VReg_64
>;
defm BUFFER_LOAD_FORMAT_XYZ : MUBUF_Pseudo_Loads <
  "buffer_load_format_xyz", VReg_96
>;
defm BUFFER_LOAD_FORMAT_XYZW : MUBUF_Pseudo_Loads <
  "buffer_load_format_xyzw", VReg_128
>;
defm BUFFER_STORE_FORMAT_X : MUBUF_Pseudo_Stores <
  "buffer_store_format_x", VGPR_32
>;
defm BUFFER_STORE_FORMAT_XY : MUBUF_Pseudo_Stores <
  "buffer_store_format_xy", VReg_64
>;
defm BUFFER_STORE_FORMAT_XYZ : MUBUF_Pseudo_Stores <
  "buffer_store_format_xyz", VReg_96
>;
defm BUFFER_STORE_FORMAT_XYZW : MUBUF_Pseudo_Stores <
  "buffer_store_format_xyzw", VReg_128
>;
defm BUFFER_LOAD_UBYTE : MUBUF_Pseudo_Loads <
  "buffer_load_ubyte", VGPR_32, i32, mubuf_az_extloadi8
>;
defm BUFFER_LOAD_SBYTE : MUBUF_Pseudo_Loads <
  "buffer_load_sbyte", VGPR_32, i32, mubuf_sextloadi8
>;
defm BUFFER_LOAD_USHORT : MUBUF_Pseudo_Loads <
  "buffer_load_ushort", VGPR_32, i32, mubuf_az_extloadi16
>;
defm BUFFER_LOAD_SSHORT : MUBUF_Pseudo_Loads <
  "buffer_load_sshort", VGPR_32, i32, mubuf_sextloadi16
>;
defm BUFFER_LOAD_DWORD : MUBUF_Pseudo_Loads <
  "buffer_load_dword", VGPR_32, i32, mubuf_load
>;
defm BUFFER_LOAD_DWORDX2 : MUBUF_Pseudo_Loads <
  "buffer_load_dwordx2", VReg_64, v2i32, mubuf_load
>;
defm BUFFER_LOAD_DWORDX4 : MUBUF_Pseudo_Loads <
  "buffer_load_dwordx4", VReg_128, v4i32, mubuf_load
>;
defm BUFFER_STORE_BYTE : MUBUF_Pseudo_Stores <
  "buffer_store_byte", VGPR_32, i32, truncstorei8_global
>;
defm BUFFER_STORE_SHORT : MUBUF_Pseudo_Stores <
  "buffer_store_short", VGPR_32, i32, truncstorei16_global
>;
defm BUFFER_STORE_DWORD : MUBUF_Pseudo_Stores <
  "buffer_store_dword", VGPR_32, i32, global_store
>;
defm BUFFER_STORE_DWORDX2 : MUBUF_Pseudo_Stores <
  "buffer_store_dwordx2", VReg_64, v2i32, global_store
>;
defm BUFFER_STORE_DWORDX4 : MUBUF_Pseudo_Stores <
  "buffer_store_dwordx4", VReg_128, v4i32, global_store
>;
defm BUFFER_ATOMIC_SWAP : MUBUF_Pseudo_Atomics <
  "buffer_atomic_swap", VGPR_32, i32, atomic_swap_global
>;
defm BUFFER_ATOMIC_CMPSWAP : MUBUF_Pseudo_Atomics <
  "buffer_atomic_cmpswap", VReg_64, v2i32, null_frag
>;
defm BUFFER_ATOMIC_ADD : MUBUF_Pseudo_Atomics <
  "buffer_atomic_add", VGPR_32, i32, atomic_add_global
>;
defm BUFFER_ATOMIC_SUB : MUBUF_Pseudo_Atomics <
  "buffer_atomic_sub", VGPR_32, i32, atomic_sub_global
>;
defm BUFFER_ATOMIC_SMIN : MUBUF_Pseudo_Atomics <
  "buffer_atomic_smin", VGPR_32, i32, atomic_min_global
>;
defm BUFFER_ATOMIC_UMIN : MUBUF_Pseudo_Atomics <
  "buffer_atomic_umin", VGPR_32, i32, atomic_umin_global
>;
defm BUFFER_ATOMIC_SMAX : MUBUF_Pseudo_Atomics <
  "buffer_atomic_smax", VGPR_32, i32, atomic_max_global
>;
defm BUFFER_ATOMIC_UMAX : MUBUF_Pseudo_Atomics <
  "buffer_atomic_umax", VGPR_32, i32, atomic_umax_global
>;
defm BUFFER_ATOMIC_AND : MUBUF_Pseudo_Atomics <
  "buffer_atomic_and", VGPR_32, i32, atomic_and_global
>;
defm BUFFER_ATOMIC_OR : MUBUF_Pseudo_Atomics <
  "buffer_atomic_or", VGPR_32, i32, atomic_or_global
>;
defm BUFFER_ATOMIC_XOR : MUBUF_Pseudo_Atomics <
  "buffer_atomic_xor", VGPR_32, i32, atomic_xor_global
>;
defm BUFFER_ATOMIC_INC : MUBUF_Pseudo_Atomics <
  "buffer_atomic_inc", VGPR_32, i32, atomic_inc_global
>;
defm BUFFER_ATOMIC_DEC : MUBUF_Pseudo_Atomics <
  "buffer_atomic_dec", VGPR_32, i32, atomic_dec_global
>;
defm BUFFER_ATOMIC_SWAP_X2 : MUBUF_Pseudo_Atomics <
  "buffer_atomic_swap_x2", VReg_64, i64, atomic_swap_global
>;
defm BUFFER_ATOMIC_CMPSWAP_X2 : MUBUF_Pseudo_Atomics <
  "buffer_atomic_cmpswap_x2", VReg_128, v2i64, null_frag
>;
defm BUFFER_ATOMIC_ADD_X2 : MUBUF_Pseudo_Atomics <
  "buffer_atomic_add_x2", VReg_64, i64, atomic_add_global
>;
defm BUFFER_ATOMIC_SUB_X2 : MUBUF_Pseudo_Atomics <
  "buffer_atomic_sub_x2", VReg_64, i64, atomic_sub_global
>;
defm BUFFER_ATOMIC_SMIN_X2 : MUBUF_Pseudo_Atomics <
  "buffer_atomic_smin_x2", VReg_64, i64, atomic_min_global
>;
defm BUFFER_ATOMIC_UMIN_X2 : MUBUF_Pseudo_Atomics <
  "buffer_atomic_umin_x2", VReg_64, i64, atomic_umin_global
>;
defm BUFFER_ATOMIC_SMAX_X2 : MUBUF_Pseudo_Atomics <
  "buffer_atomic_smax_x2", VReg_64, i64, atomic_max_global
>;
defm BUFFER_ATOMIC_UMAX_X2 : MUBUF_Pseudo_Atomics <
  "buffer_atomic_umax_x2", VReg_64, i64, atomic_umax_global
>;
defm BUFFER_ATOMIC_AND_X2 : MUBUF_Pseudo_Atomics <
  "buffer_atomic_and_x2", VReg_64, i64, atomic_and_global
>;
defm BUFFER_ATOMIC_OR_X2 : MUBUF_Pseudo_Atomics <
  "buffer_atomic_or_x2", VReg_64, i64, atomic_or_global
>;
defm BUFFER_ATOMIC_XOR_X2 : MUBUF_Pseudo_Atomics <
  "buffer_atomic_xor_x2", VReg_64, i64, atomic_xor_global
>;
defm BUFFER_ATOMIC_INC_X2 : MUBUF_Pseudo_Atomics <
  "buffer_atomic_inc_x2", VReg_64, i64, atomic_inc_global
>;
defm BUFFER_ATOMIC_DEC_X2 : MUBUF_Pseudo_Atomics <
  "buffer_atomic_dec_x2", VReg_64, i64, atomic_dec_global
>;

let SubtargetPredicate = isSI in { // isn't on CI & VI
/*
defm BUFFER_ATOMIC_RSUB        : MUBUF_Pseudo_Atomics <"buffer_atomic_rsub">;
defm BUFFER_ATOMIC_FCMPSWAP    : MUBUF_Pseudo_Atomics <"buffer_atomic_fcmpswap">;
defm BUFFER_ATOMIC_FMIN        : MUBUF_Pseudo_Atomics <"buffer_atomic_fmin">;
defm BUFFER_ATOMIC_FMAX        : MUBUF_Pseudo_Atomics <"buffer_atomic_fmax">;
defm BUFFER_ATOMIC_RSUB_X2     : MUBUF_Pseudo_Atomics <"buffer_atomic_rsub_x2">;
defm BUFFER_ATOMIC_FCMPSWAP_X2 : MUBUF_Pseudo_Atomics <"buffer_atomic_fcmpswap_x2">;
defm BUFFER_ATOMIC_FMIN_X2     : MUBUF_Pseudo_Atomics <"buffer_atomic_fmin_x2">;
defm BUFFER_ATOMIC_FMAX_X2     : MUBUF_Pseudo_Atomics <"buffer_atomic_fmax_x2">;
*/

def BUFFER_WBINVL1_SC : MUBUF_Invalidate <"buffer_wbinvl1_sc",
                                          int_amdgcn_buffer_wbinvl1_sc>;
}

def BUFFER_WBINVL1 : MUBUF_Invalidate <"buffer_wbinvl1",
                                       int_amdgcn_buffer_wbinvl1>;

//===----------------------------------------------------------------------===//
// MTBUF Instructions
//===----------------------------------------------------------------------===//

//def TBUFFER_LOAD_FORMAT_X    : MTBUF_ <0, "tbuffer_load_format_x", []>;
//def TBUFFER_LOAD_FORMAT_XY   : MTBUF_ <1, "tbuffer_load_format_xy", []>;
//def TBUFFER_LOAD_FORMAT_XYZ  : MTBUF_ <2, "tbuffer_load_format_xyz", []>;
def TBUFFER_LOAD_FORMAT_XYZW  : MTBUF_Load_Pseudo  <"tbuffer_load_format_xyzw", VReg_128>;
def TBUFFER_STORE_FORMAT_X    : MTBUF_Store_Pseudo <"tbuffer_store_format_x", VGPR_32>;
def TBUFFER_STORE_FORMAT_XY   : MTBUF_Store_Pseudo <"tbuffer_store_format_xy", VReg_64>;
def TBUFFER_STORE_FORMAT_XYZ  : MTBUF_Store_Pseudo <"tbuffer_store_format_xyz", VReg_128>;
def TBUFFER_STORE_FORMAT_XYZW : MTBUF_Store_Pseudo <"tbuffer_store_format_xyzw", VReg_128>;

} // End let SubtargetPredicate = isGCN

let SubtargetPredicate = isCIVI in {

//===----------------------------------------------------------------------===//
// Instruction definitions for CI and newer.
//===----------------------------------------------------------------------===//
// Remaining instructions:
// BUFFER_LOAD_DWORDX3
// BUFFER_STORE_DWORDX3

def BUFFER_WBINVL1_VOL : MUBUF_Invalidate <"buffer_wbinvl1_vol",
                                           int_amdgcn_buffer_wbinvl1_vol>;

} // End let SubtargetPredicate = isCIVI

//===----------------------------------------------------------------------===//
// MUBUF Patterns
//===----------------------------------------------------------------------===//

def mubuf_vaddr_offset : PatFrag<
  (ops node:$ptr, node:$offset, node:$imm_offset),
  (add (add node:$ptr, node:$offset), node:$imm_offset)
>;


let Predicates = [isGCN] in {

// int_SI_vs_load_input
def : Pat<
  (SIload_input v4i32:$tlst, imm:$attr_offset, i32:$buf_idx_vgpr),
  (BUFFER_LOAD_FORMAT_XYZW_IDXEN $buf_idx_vgpr, $tlst, 0, imm:$attr_offset, 0, 0, 0)
>;

// Offset in an 32-bit VGPR
def : Pat <
  (SIload_constant v4i32:$sbase, i32:$voff),
  (BUFFER_LOAD_DWORD_OFFEN $voff, $sbase, 0, 0, 0, 0, 0)
>;


//===----------------------------------------------------------------------===//
// buffer_load/store_format patterns
//===----------------------------------------------------------------------===//

multiclass MUBUF_LoadIntrinsicPat<SDPatternOperator name, ValueType vt,
                                  string opcode> {
  def : Pat<
    (vt (name v4i32:$rsrc, 0,
              (MUBUFIntrinsicOffset i32:$soffset, i16:$offset),
              imm:$glc, imm:$slc)),
    (!cast<MUBUF_Pseudo>(opcode # _OFFSET) $rsrc, $soffset, (as_i16imm $offset),
      (as_i1imm $glc), (as_i1imm $slc), 0)
  >;

  def : Pat<
    (vt (name v4i32:$rsrc, i32:$vindex,
              (MUBUFIntrinsicOffset i32:$soffset, i16:$offset),
              imm:$glc, imm:$slc)),
    (!cast<MUBUF_Pseudo>(opcode # _IDXEN) $vindex, $rsrc, $soffset, (as_i16imm $offset),
      (as_i1imm $glc), (as_i1imm $slc), 0)
  >;

  def : Pat<
    (vt (name v4i32:$rsrc, 0,
              (MUBUFIntrinsicVOffset i32:$soffset, i16:$offset, i32:$voffset),
              imm:$glc, imm:$slc)),
    (!cast<MUBUF_Pseudo>(opcode # _OFFEN) $voffset, $rsrc, $soffset, (as_i16imm $offset),
      (as_i1imm $glc), (as_i1imm $slc), 0)
  >;

  def : Pat<
    (vt (name v4i32:$rsrc, i32:$vindex,
              (MUBUFIntrinsicVOffset i32:$soffset, i16:$offset, i32:$voffset),
              imm:$glc, imm:$slc)),
    (!cast<MUBUF_Pseudo>(opcode # _BOTHEN)
      (REG_SEQUENCE VReg_64, $vindex, sub0, $voffset, sub1),
      $rsrc, $soffset, (as_i16imm $offset),
      (as_i1imm $glc), (as_i1imm $slc), 0)
  >;
}

defm : MUBUF_LoadIntrinsicPat<int_amdgcn_buffer_load_format, f32, "BUFFER_LOAD_FORMAT_X">;
defm : MUBUF_LoadIntrinsicPat<int_amdgcn_buffer_load_format, v2f32, "BUFFER_LOAD_FORMAT_XY">;
defm : MUBUF_LoadIntrinsicPat<int_amdgcn_buffer_load_format, v4f32, "BUFFER_LOAD_FORMAT_XYZW">;
defm : MUBUF_LoadIntrinsicPat<int_amdgcn_buffer_load, f32, "BUFFER_LOAD_DWORD">;
defm : MUBUF_LoadIntrinsicPat<int_amdgcn_buffer_load, v2f32, "BUFFER_LOAD_DWORDX2">;
defm : MUBUF_LoadIntrinsicPat<int_amdgcn_buffer_load, v4f32, "BUFFER_LOAD_DWORDX4">;

multiclass MUBUF_StoreIntrinsicPat<SDPatternOperator name, ValueType vt,
                                   string opcode> {
  def : Pat<
    (name vt:$vdata, v4i32:$rsrc, 0,
          (MUBUFIntrinsicOffset i32:$soffset, i16:$offset),
          imm:$glc, imm:$slc),
    (!cast<MUBUF_Pseudo>(opcode # _OFFSET_exact) $vdata, $rsrc, $soffset, (as_i16imm $offset),
                                    (as_i1imm $glc), (as_i1imm $slc), 0)
  >;

  def : Pat<
    (name vt:$vdata, v4i32:$rsrc, i32:$vindex,
          (MUBUFIntrinsicOffset i32:$soffset, i16:$offset),
          imm:$glc, imm:$slc),
    (!cast<MUBUF_Pseudo>(opcode # _IDXEN_exact) $vdata, $vindex, $rsrc, $soffset,
                                   (as_i16imm $offset), (as_i1imm $glc),
                                   (as_i1imm $slc), 0)
  >;

  def : Pat<
    (name vt:$vdata, v4i32:$rsrc, 0,
          (MUBUFIntrinsicVOffset i32:$soffset, i16:$offset, i32:$voffset),
          imm:$glc, imm:$slc),
    (!cast<MUBUF_Pseudo>(opcode # _OFFEN_exact) $vdata, $voffset, $rsrc, $soffset,
                                   (as_i16imm $offset), (as_i1imm $glc),
                                   (as_i1imm $slc), 0)
  >;

  def : Pat<
    (name vt:$vdata, v4i32:$rsrc, i32:$vindex,
          (MUBUFIntrinsicVOffset i32:$soffset, i16:$offset, i32:$voffset),
          imm:$glc, imm:$slc),
    (!cast<MUBUF_Pseudo>(opcode # _BOTHEN_exact)
      $vdata,
      (REG_SEQUENCE VReg_64, $vindex, sub0, $voffset, sub1),
      $rsrc, $soffset, (as_i16imm $offset),
      (as_i1imm $glc), (as_i1imm $slc), 0)
  >;
}

defm : MUBUF_StoreIntrinsicPat<int_amdgcn_buffer_store_format, f32, "BUFFER_STORE_FORMAT_X">;
defm : MUBUF_StoreIntrinsicPat<int_amdgcn_buffer_store_format, v2f32, "BUFFER_STORE_FORMAT_XY">;
defm : MUBUF_StoreIntrinsicPat<int_amdgcn_buffer_store_format, v4f32, "BUFFER_STORE_FORMAT_XYZW">;
defm : MUBUF_StoreIntrinsicPat<int_amdgcn_buffer_store, f32, "BUFFER_STORE_DWORD">;
defm : MUBUF_StoreIntrinsicPat<int_amdgcn_buffer_store, v2f32, "BUFFER_STORE_DWORDX2">;
defm : MUBUF_StoreIntrinsicPat<int_amdgcn_buffer_store, v4f32, "BUFFER_STORE_DWORDX4">;

//===----------------------------------------------------------------------===//
// buffer_atomic patterns
//===----------------------------------------------------------------------===//

multiclass BufferAtomicPatterns<SDPatternOperator name, string opcode> {
  def : Pat<
    (name i32:$vdata_in, v4i32:$rsrc, 0,
          (MUBUFIntrinsicOffset i32:$soffset, i16:$offset),
          imm:$slc),
    (!cast<MUBUF_Pseudo>(opcode # _RTN_OFFSET) $vdata_in, $rsrc, $soffset,
                                        (as_i16imm $offset), (as_i1imm $slc))
  >;

  def : Pat<
    (name i32:$vdata_in, v4i32:$rsrc, i32:$vindex,
          (MUBUFIntrinsicOffset i32:$soffset, i16:$offset),
          imm:$slc),
    (!cast<MUBUF_Pseudo>(opcode # _RTN_IDXEN) $vdata_in, $vindex, $rsrc, $soffset,
                                       (as_i16imm $offset), (as_i1imm $slc))
  >;

  def : Pat<
    (name i32:$vdata_in, v4i32:$rsrc, 0,
          (MUBUFIntrinsicVOffset i32:$soffset, i16:$offset, i32:$voffset),
          imm:$slc),
    (!cast<MUBUF_Pseudo>(opcode # _RTN_OFFEN) $vdata_in, $voffset, $rsrc, $soffset,
                                       (as_i16imm $offset), (as_i1imm $slc))
  >;

  def : Pat<
    (name i32:$vdata_in, v4i32:$rsrc, i32:$vindex,
          (MUBUFIntrinsicVOffset i32:$soffset, i16:$offset, i32:$voffset),
          imm:$slc),
    (!cast<MUBUF_Pseudo>(opcode # _RTN_BOTHEN)
      $vdata_in,
      (REG_SEQUENCE VReg_64, $vindex, sub0, $voffset, sub1),
      $rsrc, $soffset, (as_i16imm $offset), (as_i1imm $slc))
  >;
}

defm : BufferAtomicPatterns<int_amdgcn_buffer_atomic_swap, "BUFFER_ATOMIC_SWAP">;
defm : BufferAtomicPatterns<int_amdgcn_buffer_atomic_add, "BUFFER_ATOMIC_ADD">;
defm : BufferAtomicPatterns<int_amdgcn_buffer_atomic_sub, "BUFFER_ATOMIC_SUB">;
defm : BufferAtomicPatterns<int_amdgcn_buffer_atomic_smin, "BUFFER_ATOMIC_SMIN">;
defm : BufferAtomicPatterns<int_amdgcn_buffer_atomic_umin, "BUFFER_ATOMIC_UMIN">;
defm : BufferAtomicPatterns<int_amdgcn_buffer_atomic_smax, "BUFFER_ATOMIC_SMAX">;
defm : BufferAtomicPatterns<int_amdgcn_buffer_atomic_umax, "BUFFER_ATOMIC_UMAX">;
defm : BufferAtomicPatterns<int_amdgcn_buffer_atomic_and, "BUFFER_ATOMIC_AND">;
defm : BufferAtomicPatterns<int_amdgcn_buffer_atomic_or, "BUFFER_ATOMIC_OR">;
defm : BufferAtomicPatterns<int_amdgcn_buffer_atomic_xor, "BUFFER_ATOMIC_XOR">;

def : Pat<
  (int_amdgcn_buffer_atomic_cmpswap
      i32:$data, i32:$cmp, v4i32:$rsrc, 0,
      (MUBUFIntrinsicOffset i32:$soffset, i16:$offset),
      imm:$slc),
  (EXTRACT_SUBREG
    (BUFFER_ATOMIC_CMPSWAP_RTN_OFFSET
      (REG_SEQUENCE VReg_64, $data, sub0, $cmp, sub1),
      $rsrc, $soffset, (as_i16imm $offset), (as_i1imm $slc)),
    sub0)
>;

def : Pat<
  (int_amdgcn_buffer_atomic_cmpswap
      i32:$data, i32:$cmp, v4i32:$rsrc, i32:$vindex,
      (MUBUFIntrinsicOffset i32:$soffset, i16:$offset),
      imm:$slc),
  (EXTRACT_SUBREG
    (BUFFER_ATOMIC_CMPSWAP_RTN_IDXEN
      (REG_SEQUENCE VReg_64, $data, sub0, $cmp, sub1),
      $vindex, $rsrc, $soffset, (as_i16imm $offset), (as_i1imm $slc)),
    sub0)
>;

def : Pat<
  (int_amdgcn_buffer_atomic_cmpswap
      i32:$data, i32:$cmp, v4i32:$rsrc, 0,
      (MUBUFIntrinsicVOffset i32:$soffset, i16:$offset, i32:$voffset),
      imm:$slc),
  (EXTRACT_SUBREG
    (BUFFER_ATOMIC_CMPSWAP_RTN_OFFEN
      (REG_SEQUENCE VReg_64, $data, sub0, $cmp, sub1),
      $voffset, $rsrc, $soffset, (as_i16imm $offset), (as_i1imm $slc)),
    sub0)
>;

def : Pat<
  (int_amdgcn_buffer_atomic_cmpswap
      i32:$data, i32:$cmp, v4i32:$rsrc, i32:$vindex,
      (MUBUFIntrinsicVOffset i32:$soffset, i16:$offset, i32:$voffset),
      imm:$slc),
  (EXTRACT_SUBREG
    (BUFFER_ATOMIC_CMPSWAP_RTN_BOTHEN
      (REG_SEQUENCE VReg_64, $data, sub0, $cmp, sub1),
      (REG_SEQUENCE VReg_64, $vindex, sub0, $voffset, sub1),
      $rsrc, $soffset, (as_i16imm $offset), (as_i1imm $slc)),
    sub0)
>;


class MUBUFLoad_Pattern <MUBUF_Pseudo Instr_ADDR64, ValueType vt,
                              PatFrag constant_ld> : Pat <
     (vt (constant_ld (MUBUFAddr64 v4i32:$srsrc, i64:$vaddr, i32:$soffset,
                                   i16:$offset, i1:$glc, i1:$slc, i1:$tfe))),
     (Instr_ADDR64 $vaddr, $srsrc, $soffset, $offset, $glc, $slc, $tfe)
  >;

multiclass MUBUFLoad_Atomic_Pattern <MUBUF_Pseudo Instr_ADDR64, MUBUF_Pseudo Instr_OFFSET,
                                     ValueType vt, PatFrag atomic_ld> {
  def : Pat <
     (vt (atomic_ld (MUBUFAddr64 v4i32:$srsrc, i64:$vaddr, i32:$soffset,
                                   i16:$offset, i1:$slc))),
     (Instr_ADDR64 $vaddr, $srsrc, $soffset, $offset, 1, $slc, 0)
  >;

  def : Pat <
    (vt (atomic_ld (MUBUFOffsetNoGLC v4i32:$rsrc, i32:$soffset, i16:$offset))),
    (Instr_OFFSET $rsrc, $soffset, (as_i16imm $offset), 1, 0, 0)
  >;
}

let Predicates = [isSICI] in {
def : MUBUFLoad_Pattern <BUFFER_LOAD_SBYTE_ADDR64, i32, sextloadi8_constant>;
def : MUBUFLoad_Pattern <BUFFER_LOAD_UBYTE_ADDR64, i32, az_extloadi8_constant>;
def : MUBUFLoad_Pattern <BUFFER_LOAD_SSHORT_ADDR64, i32, sextloadi16_constant>;
def : MUBUFLoad_Pattern <BUFFER_LOAD_USHORT_ADDR64, i32, az_extloadi16_constant>;

defm : MUBUFLoad_Atomic_Pattern <BUFFER_LOAD_DWORD_ADDR64, BUFFER_LOAD_DWORD_OFFSET, i32, mubuf_load_atomic>;
defm : MUBUFLoad_Atomic_Pattern <BUFFER_LOAD_DWORDX2_ADDR64, BUFFER_LOAD_DWORDX2_OFFSET, i64, mubuf_load_atomic>;
} // End Predicates = [isSICI]

class MUBUFScratchLoadPat <MUBUF_Pseudo Instr, ValueType vt, PatFrag ld> : Pat <
  (vt (ld (MUBUFScratch v4i32:$srsrc, i32:$vaddr,
                        i32:$soffset, u16imm:$offset))),
  (Instr $vaddr, $srsrc, $soffset, $offset, 0, 0, 0)
>;

def : MUBUFScratchLoadPat <BUFFER_LOAD_SBYTE_OFFEN, i32, sextloadi8_private>;
def : MUBUFScratchLoadPat <BUFFER_LOAD_UBYTE_OFFEN, i32, extloadi8_private>;
def : MUBUFScratchLoadPat <BUFFER_LOAD_SSHORT_OFFEN, i32, sextloadi16_private>;
def : MUBUFScratchLoadPat <BUFFER_LOAD_USHORT_OFFEN, i32, extloadi16_private>;
def : MUBUFScratchLoadPat <BUFFER_LOAD_DWORD_OFFEN, i32, load_private>;
def : MUBUFScratchLoadPat <BUFFER_LOAD_DWORDX2_OFFEN, v2i32, load_private>;
def : MUBUFScratchLoadPat <BUFFER_LOAD_DWORDX4_OFFEN, v4i32, load_private>;

// BUFFER_LOAD_DWORD*, addr64=0
multiclass MUBUF_Load_Dword <ValueType vt,
                             MUBUF_Pseudo offset,
                             MUBUF_Pseudo offen,
                             MUBUF_Pseudo idxen,
                             MUBUF_Pseudo bothen> {

  def : Pat <
    (vt (int_SI_buffer_load_dword v4i32:$rsrc, (i32 imm), i32:$soffset,
                                  imm:$offset, 0, 0, imm:$glc, imm:$slc,
                                  imm:$tfe)),
    (offset $rsrc, $soffset, (as_i16imm $offset), (as_i1imm $glc),
            (as_i1imm $slc), (as_i1imm $tfe))
  >;

  def : Pat <
    (vt (int_SI_buffer_load_dword v4i32:$rsrc, i32:$vaddr, i32:$soffset,
                                  imm:$offset, 1, 0, imm:$glc, imm:$slc,
                                  imm:$tfe)),
    (offen $vaddr, $rsrc, $soffset, (as_i16imm $offset), (as_i1imm $glc), (as_i1imm $slc),
           (as_i1imm $tfe))
  >;

  def : Pat <
    (vt (int_SI_buffer_load_dword v4i32:$rsrc, i32:$vaddr, i32:$soffset,
                                  imm:$offset, 0, 1, imm:$glc, imm:$slc,
                                  imm:$tfe)),
    (idxen $vaddr, $rsrc, $soffset, (as_i16imm $offset), (as_i1imm $glc),
           (as_i1imm $slc), (as_i1imm $tfe))
  >;

  def : Pat <
    (vt (int_SI_buffer_load_dword v4i32:$rsrc, v2i32:$vaddr, i32:$soffset,
                                  imm:$offset, 1, 1, imm:$glc, imm:$slc,
                                  imm:$tfe)),
    (bothen $vaddr, $rsrc, $soffset, (as_i16imm $offset), (as_i1imm $glc), (as_i1imm $slc),
            (as_i1imm $tfe))
  >;
}

defm : MUBUF_Load_Dword <i32, BUFFER_LOAD_DWORD_OFFSET, BUFFER_LOAD_DWORD_OFFEN,
                         BUFFER_LOAD_DWORD_IDXEN, BUFFER_LOAD_DWORD_BOTHEN>;
defm : MUBUF_Load_Dword <v2i32, BUFFER_LOAD_DWORDX2_OFFSET, BUFFER_LOAD_DWORDX2_OFFEN,
                         BUFFER_LOAD_DWORDX2_IDXEN, BUFFER_LOAD_DWORDX2_BOTHEN>;
defm : MUBUF_Load_Dword <v4i32, BUFFER_LOAD_DWORDX4_OFFSET, BUFFER_LOAD_DWORDX4_OFFEN,
                         BUFFER_LOAD_DWORDX4_IDXEN, BUFFER_LOAD_DWORDX4_BOTHEN>;

multiclass MUBUFStore_Atomic_Pattern <MUBUF_Pseudo Instr_ADDR64, MUBUF_Pseudo Instr_OFFSET,
                                      ValueType vt, PatFrag atomic_st> {
  // Store follows atomic op convention so address is forst
  def : Pat <
     (atomic_st (MUBUFAddr64 v4i32:$srsrc, i64:$vaddr, i32:$soffset,
                                   i16:$offset, i1:$slc), vt:$val),
     (Instr_ADDR64 $val, $vaddr, $srsrc, $soffset, $offset, 1, $slc, 0)
  >;

  def : Pat <
    (atomic_st (MUBUFOffsetNoGLC v4i32:$rsrc, i32:$soffset, i16:$offset), vt:$val),
    (Instr_OFFSET $val, $rsrc, $soffset, (as_i16imm $offset), 1, 0, 0)
  >;
}
let Predicates = [isSICI] in {
defm : MUBUFStore_Atomic_Pattern <BUFFER_STORE_DWORD_ADDR64, BUFFER_STORE_DWORD_OFFSET, i32, global_store_atomic>;
defm : MUBUFStore_Atomic_Pattern <BUFFER_STORE_DWORDX2_ADDR64, BUFFER_STORE_DWORDX2_OFFSET, i64, global_store_atomic>;
} // End Predicates = [isSICI]

class MUBUFScratchStorePat <MUBUF_Pseudo Instr, ValueType vt, PatFrag st> : Pat <
  (st vt:$value, (MUBUFScratch v4i32:$srsrc, i32:$vaddr, i32:$soffset,
                               u16imm:$offset)),
  (Instr $value, $vaddr, $srsrc, $soffset, $offset, 0, 0, 0)
>;

def : MUBUFScratchStorePat <BUFFER_STORE_BYTE_OFFEN, i32, truncstorei8_private>;
def : MUBUFScratchStorePat <BUFFER_STORE_SHORT_OFFEN, i32, truncstorei16_private>;
def : MUBUFScratchStorePat <BUFFER_STORE_DWORD_OFFEN, i32, store_private>;
def : MUBUFScratchStorePat <BUFFER_STORE_DWORDX2_OFFEN, v2i32, store_private>;
def : MUBUFScratchStorePat <BUFFER_STORE_DWORDX4_OFFEN, v4i32, store_private>;

//===----------------------------------------------------------------------===//
// MTBUF Patterns
//===----------------------------------------------------------------------===//

// TBUFFER_STORE_FORMAT_*, addr64=0
class MTBUF_StoreResource <ValueType vt, int num_channels, MTBUF_Pseudo opcode> : Pat<
  (SItbuffer_store v4i32:$rsrc, vt:$vdata, num_channels, i32:$vaddr,
                   i32:$soffset, imm:$inst_offset, imm:$dfmt,
                   imm:$nfmt, imm:$offen, imm:$idxen,
                   imm:$glc, imm:$slc, imm:$tfe),
  (opcode
    $vdata, (as_i16imm $inst_offset), (as_i1imm $offen), (as_i1imm $idxen),
    (as_i1imm $glc), 0, (as_i8imm $dfmt), (as_i8imm $nfmt), $vaddr, $rsrc,
    (as_i1imm $slc), (as_i1imm $tfe), $soffset)
>;

def : MTBUF_StoreResource <i32, 1, TBUFFER_STORE_FORMAT_X>;
def : MTBUF_StoreResource <v2i32, 2, TBUFFER_STORE_FORMAT_XY>;
def : MTBUF_StoreResource <v4i32, 3, TBUFFER_STORE_FORMAT_XYZ>;
def : MTBUF_StoreResource <v4i32, 4, TBUFFER_STORE_FORMAT_XYZW>;

} // End let Predicates = [isGCN]

//===----------------------------------------------------------------------===//
// Target instructions, move to the appropriate target TD file
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// SI
//===----------------------------------------------------------------------===//

class MUBUF_Real_si <bits<7> op, MUBUF_Pseudo ps> :
  MUBUF_Real<op, ps>,
  Enc64,
  SIMCInstr<ps.PseudoInstr, SIEncodingFamily.SI> {
  let AssemblerPredicate=isSICI;
  let DecoderNamespace="SICI";

  let Inst{11-0}  = !if(ps.has_offset, offset, ?);
  let Inst{12}    = ps.offen;
  let Inst{13}    = ps.idxen;
  let Inst{14}    = !if(ps.has_glc, glc, ps.glc_value);
  let Inst{15}    = ps.addr64;
  let Inst{16}    = lds;
  let Inst{24-18} = op;
  let Inst{31-26} = 0x38; //encoding
  let Inst{39-32} = !if(ps.has_vaddr, vaddr, ?);
  let Inst{47-40} = !if(ps.has_vdata, vdata, ?);
  let Inst{52-48} = !if(ps.has_srsrc, srsrc{6-2}, ?);
  let Inst{54}    = !if(ps.has_slc, slc, ?);
  let Inst{55}    = !if(ps.has_tfe, tfe, ?);
  let Inst{63-56} = !if(ps.has_soffset, soffset, ?);
}

multiclass MUBUF_Real_AllAddr_si<bits<7> op> {
  def _OFFSET_si : MUBUF_Real_si <op, !cast<MUBUF_Pseudo>(NAME#"_OFFSET")>;
  def _ADDR64_si : MUBUF_Real_si <op, !cast<MUBUF_Pseudo>(NAME#"_ADDR64")>;
  def _OFFEN_si  : MUBUF_Real_si <op, !cast<MUBUF_Pseudo>(NAME#"_OFFEN")>;
  def _IDXEN_si  : MUBUF_Real_si <op, !cast<MUBUF_Pseudo>(NAME#"_IDXEN")>;
  def _BOTHEN_si : MUBUF_Real_si <op, !cast<MUBUF_Pseudo>(NAME#"_BOTHEN")>;
}

multiclass MUBUF_Real_Atomic_si<bits<7> op> : MUBUF_Real_AllAddr_si<op> {
  def _RTN_OFFSET_si : MUBUF_Real_si <op, !cast<MUBUF_Pseudo>(NAME#"_RTN_OFFSET")>;
  def _RTN_ADDR64_si : MUBUF_Real_si <op, !cast<MUBUF_Pseudo>(NAME#"_RTN_ADDR64")>;
  def _RTN_OFFEN_si  : MUBUF_Real_si <op, !cast<MUBUF_Pseudo>(NAME#"_RTN_OFFEN")>;
  def _RTN_IDXEN_si  : MUBUF_Real_si <op, !cast<MUBUF_Pseudo>(NAME#"_RTN_IDXEN")>;
  def _RTN_BOTHEN_si : MUBUF_Real_si <op, !cast<MUBUF_Pseudo>(NAME#"_RTN_BOTHEN")>;
}

defm BUFFER_LOAD_FORMAT_X       : MUBUF_Real_AllAddr_si <0x00>;
defm BUFFER_LOAD_FORMAT_XY      : MUBUF_Real_AllAddr_si <0x01>;
defm BUFFER_LOAD_FORMAT_XYZ     : MUBUF_Real_AllAddr_si <0x02>;
defm BUFFER_LOAD_FORMAT_XYZW    : MUBUF_Real_AllAddr_si <0x03>;
defm BUFFER_STORE_FORMAT_X      : MUBUF_Real_AllAddr_si <0x04>;
defm BUFFER_STORE_FORMAT_XY     : MUBUF_Real_AllAddr_si <0x05>;
defm BUFFER_STORE_FORMAT_XYZ    : MUBUF_Real_AllAddr_si <0x06>;
defm BUFFER_STORE_FORMAT_XYZW   : MUBUF_Real_AllAddr_si <0x07>;
defm BUFFER_LOAD_UBYTE          : MUBUF_Real_AllAddr_si <0x08>;
defm BUFFER_LOAD_SBYTE          : MUBUF_Real_AllAddr_si <0x09>;
defm BUFFER_LOAD_USHORT         : MUBUF_Real_AllAddr_si <0x0a>;
defm BUFFER_LOAD_SSHORT         : MUBUF_Real_AllAddr_si <0x0b>;
defm BUFFER_LOAD_DWORD          : MUBUF_Real_AllAddr_si <0x0c>;
defm BUFFER_LOAD_DWORDX2        : MUBUF_Real_AllAddr_si <0x0d>;
defm BUFFER_LOAD_DWORDX4        : MUBUF_Real_AllAddr_si <0x0e>;
defm BUFFER_STORE_BYTE          : MUBUF_Real_AllAddr_si <0x18>;
defm BUFFER_STORE_SHORT         : MUBUF_Real_AllAddr_si <0x1a>;
defm BUFFER_STORE_DWORD         : MUBUF_Real_AllAddr_si <0x1c>;
defm BUFFER_STORE_DWORDX2       : MUBUF_Real_AllAddr_si <0x1d>;
defm BUFFER_STORE_DWORDX4       : MUBUF_Real_AllAddr_si <0x1e>;

defm BUFFER_ATOMIC_SWAP         : MUBUF_Real_Atomic_si <0x30>;
defm BUFFER_ATOMIC_CMPSWAP      : MUBUF_Real_Atomic_si <0x31>;
defm BUFFER_ATOMIC_ADD          : MUBUF_Real_Atomic_si <0x32>;
defm BUFFER_ATOMIC_SUB          : MUBUF_Real_Atomic_si <0x33>;
//defm BUFFER_ATOMIC_RSUB         : MUBUF_Real_Atomic_si <0x34>;    // isn't on CI & VI
defm BUFFER_ATOMIC_SMIN         : MUBUF_Real_Atomic_si <0x35>;
defm BUFFER_ATOMIC_UMIN         : MUBUF_Real_Atomic_si <0x36>;
defm BUFFER_ATOMIC_SMAX         : MUBUF_Real_Atomic_si <0x37>;
defm BUFFER_ATOMIC_UMAX         : MUBUF_Real_Atomic_si <0x38>;
defm BUFFER_ATOMIC_AND          : MUBUF_Real_Atomic_si <0x39>;
defm BUFFER_ATOMIC_OR           : MUBUF_Real_Atomic_si <0x3a>;
defm BUFFER_ATOMIC_XOR          : MUBUF_Real_Atomic_si <0x3b>;
defm BUFFER_ATOMIC_INC          : MUBUF_Real_Atomic_si <0x3c>;
defm BUFFER_ATOMIC_DEC          : MUBUF_Real_Atomic_si <0x3d>;

//defm BUFFER_ATOMIC_FCMPSWAP     : MUBUF_Real_Atomic_si <0x3e>;    // isn't on VI
//defm BUFFER_ATOMIC_FMIN         : MUBUF_Real_Atomic_si <0x3f>;    // isn't on VI
//defm BUFFER_ATOMIC_FMAX         : MUBUF_Real_Atomic_si <0x40>;    // isn't on VI
defm BUFFER_ATOMIC_SWAP_X2      : MUBUF_Real_Atomic_si <0x50>;
defm BUFFER_ATOMIC_CMPSWAP_X2   : MUBUF_Real_Atomic_si <0x51>;
defm BUFFER_ATOMIC_ADD_X2       : MUBUF_Real_Atomic_si <0x52>;
defm BUFFER_ATOMIC_SUB_X2       : MUBUF_Real_Atomic_si <0x53>;
//defm BUFFER_ATOMIC_RSUB_X2      : MUBUF_Real_Atomic_si <0x54>;    // isn't on CI & VI
defm BUFFER_ATOMIC_SMIN_X2      : MUBUF_Real_Atomic_si <0x55>;
defm BUFFER_ATOMIC_UMIN_X2      : MUBUF_Real_Atomic_si <0x56>;
defm BUFFER_ATOMIC_SMAX_X2      : MUBUF_Real_Atomic_si <0x57>;
defm BUFFER_ATOMIC_UMAX_X2      : MUBUF_Real_Atomic_si <0x58>;
defm BUFFER_ATOMIC_AND_X2       : MUBUF_Real_Atomic_si <0x59>;
defm BUFFER_ATOMIC_OR_X2        : MUBUF_Real_Atomic_si <0x5a>;
defm BUFFER_ATOMIC_XOR_X2       : MUBUF_Real_Atomic_si <0x5b>;
defm BUFFER_ATOMIC_INC_X2       : MUBUF_Real_Atomic_si <0x5c>;
defm BUFFER_ATOMIC_DEC_X2       : MUBUF_Real_Atomic_si <0x5d>;
//defm BUFFER_ATOMIC_FCMPSWAP_X2  : MUBUF_Real_Atomic_si <0x5e">;   // isn't on VI
//defm BUFFER_ATOMIC_FMIN_X2      : MUBUF_Real_Atomic_si <0x5f>;    // isn't on VI
//defm BUFFER_ATOMIC_FMAX_X2      : MUBUF_Real_Atomic_si <0x60>;    // isn't on VI

def BUFFER_WBINVL1_SC_si        : MUBUF_Real_si <0x70, BUFFER_WBINVL1_SC>;
def BUFFER_WBINVL1_si           : MUBUF_Real_si <0x71, BUFFER_WBINVL1>;

class MTBUF_Real_si <bits<3> op, MTBUF_Pseudo ps> :
  MTBUF_Real<ps>,
  SIMCInstr<ps.PseudoInstr, SIEncodingFamily.SI> {
  let AssemblerPredicate=isSICI;
  let DecoderNamespace="SICI";

  bits<1> addr64;
  let Inst{15}    = addr64;
  let Inst{18-16} = op;
}

def TBUFFER_LOAD_FORMAT_XYZW_si  : MTBUF_Real_si <3, TBUFFER_LOAD_FORMAT_XYZW>;
def TBUFFER_STORE_FORMAT_X_si    : MTBUF_Real_si <4, TBUFFER_STORE_FORMAT_X>;
def TBUFFER_STORE_FORMAT_XY_si   : MTBUF_Real_si <5, TBUFFER_STORE_FORMAT_XY>;
def TBUFFER_STORE_FORMAT_XYZ_si  : MTBUF_Real_si <6, TBUFFER_STORE_FORMAT_XYZ>;
def TBUFFER_STORE_FORMAT_XYZW_si : MTBUF_Real_si <7, TBUFFER_STORE_FORMAT_XYZW>;


//===----------------------------------------------------------------------===//
// CI
//===----------------------------------------------------------------------===//

class MUBUF_Real_ci <bits<7> op, MUBUF_Pseudo ps> :
  MUBUF_Real_si<op, ps> {
  let AssemblerPredicate=isCIOnly;
  let DecoderNamespace="CI";
}

def BUFFER_WBINVL1_VOL_ci : MUBUF_Real_ci <0x70, BUFFER_WBINVL1_VOL>;


//===----------------------------------------------------------------------===//
// VI
//===----------------------------------------------------------------------===//

class MUBUF_Real_vi <bits<7> op, MUBUF_Pseudo ps> :
  MUBUF_Real<op, ps>,
  Enc64,
  SIMCInstr<ps.PseudoInstr, SIEncodingFamily.VI> {
  let AssemblerPredicate=isVI;
  let DecoderNamespace="VI";

  let Inst{11-0}  = !if(ps.has_offset, offset, ?);
  let Inst{12}    = ps.offen;
  let Inst{13}    = ps.idxen;
  let Inst{14}    = !if(ps.has_glc, glc, ps.glc_value);
  let Inst{16}    = lds;
  let Inst{17}    = !if(ps.has_slc, slc, ?);
  let Inst{24-18} = op;
  let Inst{31-26} = 0x38; //encoding
  let Inst{39-32} = !if(ps.has_vaddr, vaddr, ?);
  let Inst{47-40} = !if(ps.has_vdata, vdata, ?);
  let Inst{52-48} = !if(ps.has_srsrc, srsrc{6-2}, ?);
  let Inst{55}    = !if(ps.has_tfe, tfe, ?);
  let Inst{63-56} = !if(ps.has_soffset, soffset, ?);
}

multiclass MUBUF_Real_AllAddr_vi<bits<7> op> {
  def _OFFSET_vi : MUBUF_Real_vi <op, !cast<MUBUF_Pseudo>(NAME#"_OFFSET")>;
  def _OFFEN_vi  : MUBUF_Real_vi <op, !cast<MUBUF_Pseudo>(NAME#"_OFFEN")>;
  def _IDXEN_vi  : MUBUF_Real_vi <op, !cast<MUBUF_Pseudo>(NAME#"_IDXEN")>;
  def _BOTHEN_vi : MUBUF_Real_vi <op, !cast<MUBUF_Pseudo>(NAME#"_BOTHEN")>;
}

multiclass MUBUF_Real_Atomic_vi<bits<7> op> :
  MUBUF_Real_AllAddr_vi<op> {
  def _RTN_OFFSET_vi : MUBUF_Real_vi <op, !cast<MUBUF_Pseudo>(NAME#"_RTN_OFFSET")>;
  def _RTN_OFFEN_vi  : MUBUF_Real_vi <op, !cast<MUBUF_Pseudo>(NAME#"_RTN_OFFEN")>;
  def _RTN_IDXEN_vi  : MUBUF_Real_vi <op, !cast<MUBUF_Pseudo>(NAME#"_RTN_IDXEN")>;
  def _RTN_BOTHEN_vi : MUBUF_Real_vi <op, !cast<MUBUF_Pseudo>(NAME#"_RTN_BOTHEN")>;
}

defm BUFFER_LOAD_FORMAT_X       : MUBUF_Real_AllAddr_vi <0x00>;
defm BUFFER_LOAD_FORMAT_XY      : MUBUF_Real_AllAddr_vi <0x01>;
defm BUFFER_LOAD_FORMAT_XYZ     : MUBUF_Real_AllAddr_vi <0x02>;
defm BUFFER_LOAD_FORMAT_XYZW    : MUBUF_Real_AllAddr_vi <0x03>;
defm BUFFER_STORE_FORMAT_X      : MUBUF_Real_AllAddr_vi <0x04>;
defm BUFFER_STORE_FORMAT_XY     : MUBUF_Real_AllAddr_vi <0x05>;
defm BUFFER_STORE_FORMAT_XYZ    : MUBUF_Real_AllAddr_vi <0x06>;
defm BUFFER_STORE_FORMAT_XYZW   : MUBUF_Real_AllAddr_vi <0x07>;
defm BUFFER_LOAD_UBYTE          : MUBUF_Real_AllAddr_vi <0x10>;
defm BUFFER_LOAD_SBYTE          : MUBUF_Real_AllAddr_vi <0x11>;
defm BUFFER_LOAD_USHORT         : MUBUF_Real_AllAddr_vi <0x12>;
defm BUFFER_LOAD_SSHORT         : MUBUF_Real_AllAddr_vi <0x13>;
defm BUFFER_LOAD_DWORD          : MUBUF_Real_AllAddr_vi <0x14>;
defm BUFFER_LOAD_DWORDX2        : MUBUF_Real_AllAddr_vi <0x15>;
defm BUFFER_LOAD_DWORDX4        : MUBUF_Real_AllAddr_vi <0x17>;
defm BUFFER_STORE_BYTE          : MUBUF_Real_AllAddr_vi <0x18>;
defm BUFFER_STORE_SHORT         : MUBUF_Real_AllAddr_vi <0x1a>;
defm BUFFER_STORE_DWORD         : MUBUF_Real_AllAddr_vi <0x1c>;
defm BUFFER_STORE_DWORDX2       : MUBUF_Real_AllAddr_vi <0x1d>;
defm BUFFER_STORE_DWORDX4       : MUBUF_Real_AllAddr_vi <0x1f>;

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

def BUFFER_WBINVL1_vi           : MUBUF_Real_vi <0x3e, BUFFER_WBINVL1>;
def BUFFER_WBINVL1_VOL_vi       : MUBUF_Real_vi <0x3f, BUFFER_WBINVL1_VOL>;

class MTBUF_Real_vi <bits<4> op, MTBUF_Pseudo ps> :
  MTBUF_Real<ps>,
  SIMCInstr<ps.PseudoInstr, SIEncodingFamily.VI> {
  let AssemblerPredicate=isVI;
  let DecoderNamespace="VI";

  let Inst{18-15} = op;
}

def TBUFFER_LOAD_FORMAT_XYZW_vi  : MTBUF_Real_vi <3, TBUFFER_LOAD_FORMAT_XYZW>;
def TBUFFER_STORE_FORMAT_X_vi    : MTBUF_Real_vi <4, TBUFFER_STORE_FORMAT_X>;
def TBUFFER_STORE_FORMAT_XY_vi   : MTBUF_Real_vi <5, TBUFFER_STORE_FORMAT_XY>;
def TBUFFER_STORE_FORMAT_XYZ_vi  : MTBUF_Real_vi <6, TBUFFER_STORE_FORMAT_XYZ>;
def TBUFFER_STORE_FORMAT_XYZW_vi : MTBUF_Real_vi <7, TBUFFER_STORE_FORMAT_XYZW>;

