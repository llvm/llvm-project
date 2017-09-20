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

def MUBUFScratchOffen : ComplexPattern<i64, 4, "SelectMUBUFScratchOffen", [], [SDNPWantParent]>;
def MUBUFScratchOffset : ComplexPattern<i64, 3, "SelectMUBUFScratchOffset", [], [SDNPWantParent], 20>;

def MUBUFOffset : ComplexPattern<i64, 6, "SelectMUBUFOffset">;
def MUBUFOffsetNoGLC : ComplexPattern<i64, 3, "SelectMUBUFOffset">;
def MUBUFOffsetAtomic : ComplexPattern<i64, 4, "SelectMUBUFOffset">;
def MUBUFIntrinsicOffset : ComplexPattern<i32, 2, "SelectMUBUFIntrinsicOffset">;
def MUBUFIntrinsicVOffset : ComplexPattern<i32, 3, "SelectMUBUFIntrinsicVOffset">;

class MubufLoad <SDPatternOperator op> : PatFrag <
  (ops node:$ptr), (op node:$ptr), [{
  auto const AS = cast<MemSDNode>(N)->getAddressSpace();
  return AS == AMDGPUASI.GLOBAL_ADDRESS ||
         AS == AMDGPUASI.CONSTANT_ADDRESS;
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

class MTBUFAddr64Table <bit is_addr64, string suffix = ""> {
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
  let Size = 8;
  let UseNamedOperandTable = 1;

  string Mnemonic = opName;
  string AsmOperands = asmOps;

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
  bits<1> glc_value   = 0; // the value for glc if no such operand
  bits<4> dfmt_value  = 1; // the value for dfmt if no such operand
  bits<3> nfmt_value  = 0; // the value for nfmt if no such operand
  bits<1> has_srsrc   = 1;
  bits<1> has_soffset = 1;
  bits<1> has_offset  = 1;
  bits<1> has_slc     = 1;
  bits<1> has_tfe     = 1;
  bits<1> has_dfmt    = 1;
  bits<1> has_nfmt    = 1;
}

class MTBUF_Real <MTBUF_Pseudo ps> :
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
  bits<4>  dfmt;
  bits<3>  nfmt;
  bits<8>  vaddr;
  bits<8>  vdata;
  bits<7>  srsrc;
  bits<1>  slc;
  bits<1>  tfe;
  bits<8>  soffset;
}

class getMTBUFInsDA<list<RegisterClass> vdataList,
                    list<RegisterClass> vaddrList=[]> {
  RegisterClass vdataClass = !if(!empty(vdataList), ?, !head(vdataList));
  RegisterClass vaddrClass = !if(!empty(vaddrList), ?, !head(vaddrList));
  dag InsNoData = !if(!empty(vaddrList),
    (ins                    SReg_128:$srsrc, SCSrc_b32:$soffset,
         offset:$offset, DFMT:$dfmt, NFMT:$nfmt, GLC:$glc, slc:$slc, tfe:$tfe),
    (ins vaddrClass:$vaddr, SReg_128:$srsrc, SCSrc_b32:$soffset,
         offset:$offset, DFMT:$dfmt, NFMT:$nfmt, GLC:$glc, slc:$slc, tfe:$tfe)
  );
  dag InsData = !if(!empty(vaddrList),
    (ins vdataClass:$vdata,                    SReg_128:$srsrc,
         SCSrc_b32:$soffset, offset:$offset, DFMT:$dfmt, NFMT:$nfmt, GLC:$glc,
         slc:$slc, tfe:$tfe),
    (ins vdataClass:$vdata, vaddrClass:$vaddr, SReg_128:$srsrc,
         SCSrc_b32:$soffset, offset:$offset, DFMT:$dfmt, NFMT:$nfmt, GLC:$glc,
         slc:$slc, tfe:$tfe)
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
    !if(!eq(addrKind, BUFAddrKind.Offset), "off, $srsrc, $dfmt, $nfmt, $soffset",
    !if(!eq(addrKind, BUFAddrKind.OffEn),
            "$vaddr, $srsrc, $dfmt, $nfmt, $soffset offen",
    !if(!eq(addrKind, BUFAddrKind.IdxEn),
            "$vaddr, $srsrc, $dfmt, $nfmt, $soffset idxen",
    !if(!eq(addrKind, BUFAddrKind.BothEn),
            "$vaddr, $srsrc, $dfmt, $nfmt, $soffset idxen offen",
    !if(!eq(addrKind, BUFAddrKind.Addr64),
            "$vaddr, $srsrc, $dfmt, $nfmt, $soffset addr64",
    "")))));
  string ret = Pfx # "$offset";
}

class MTBUF_SetupAddr<int addrKind> {
  bits<1> offen  = !if(!eq(addrKind, BUFAddrKind.OffEn), 1,
                   !if(!eq(addrKind, BUFAddrKind.BothEn), 1 , 0));

  bits<1> idxen  = !if(!eq(addrKind, BUFAddrKind.IdxEn), 1,
                   !if(!eq(addrKind, BUFAddrKind.BothEn), 1 , 0));

  bits<1> addr64 = !if(!eq(addrKind, BUFAddrKind.Addr64), 1, 0);

  bits<1> has_vaddr = !if(!eq(addrKind, BUFAddrKind.Offset), 0, 1);
}

class MTBUF_Load_Pseudo <string opName,
                         int addrKind,
                         RegisterClass vdataClass,
                         list<dag> pattern=[],
                         // Workaround bug bz30254
                         int addrKindCopy = addrKind>
  : MTBUF_Pseudo<opName,
                 (outs vdataClass:$vdata),
                 getMTBUFIns<addrKindCopy>.ret,
                 " $vdata, " # getMTBUFAsmOps<addrKindCopy>.ret # "$glc$slc$tfe",
                 pattern>,
    MTBUF_SetupAddr<addrKindCopy> {
  let PseudoInstr = opName # "_" # getAddrName<addrKindCopy>.ret;
  let mayLoad = 1;
  let mayStore = 0;
}

multiclass MTBUF_Pseudo_Loads<string opName, RegisterClass vdataClass,
                              ValueType load_vt = i32,
                              SDPatternOperator ld = null_frag> {

  def _OFFSET : MTBUF_Load_Pseudo <opName, BUFAddrKind.Offset, vdataClass,
    [(set load_vt:$vdata,
     (ld (MUBUFOffset v4i32:$srsrc, i32:$soffset, i16:$offset, i8:$dfmt,
                      i8:$nfmt, i1:$glc, i1:$slc, i1:$tfe)))]>,
    MTBUFAddr64Table<0>;

  def _ADDR64 : MTBUF_Load_Pseudo <opName, BUFAddrKind.Addr64, vdataClass,
    [(set load_vt:$vdata,
     (ld (MUBUFAddr64 v4i32:$srsrc, i64:$vaddr, i32:$soffset, i16:$offset,
                      i8:$dfmt, i8:$nfmt, i1:$glc, i1:$slc, i1:$tfe)))]>,
    MTBUFAddr64Table<1>;

  def _OFFEN  : MTBUF_Load_Pseudo <opName, BUFAddrKind.OffEn, vdataClass>;
  def _IDXEN  : MTBUF_Load_Pseudo <opName, BUFAddrKind.IdxEn, vdataClass>;
  def _BOTHEN : MTBUF_Load_Pseudo <opName, BUFAddrKind.BothEn, vdataClass>;

  let DisableWQM = 1 in {
    def _OFFSET_exact : MTBUF_Load_Pseudo <opName, BUFAddrKind.Offset, vdataClass>;
    def _OFFEN_exact  : MTBUF_Load_Pseudo <opName, BUFAddrKind.OffEn, vdataClass>;
    def _IDXEN_exact  : MTBUF_Load_Pseudo <opName, BUFAddrKind.IdxEn, vdataClass>;
    def _BOTHEN_exact : MTBUF_Load_Pseudo <opName, BUFAddrKind.BothEn, vdataClass>;
  }
}

class MTBUF_Store_Pseudo <string opName,
                          int addrKind,
                          RegisterClass vdataClass,
                          list<dag> pattern=[],
                          // Workaround bug bz30254
                          int addrKindCopy = addrKind,
                          RegisterClass vdataClassCopy = vdataClass>
  : MTBUF_Pseudo<opName,
                 (outs),
                 getMTBUFIns<addrKindCopy, [vdataClassCopy]>.ret,
                 " $vdata, " # getMTBUFAsmOps<addrKindCopy>.ret # "$glc$slc$tfe",
                 pattern>,
    MTBUF_SetupAddr<addrKindCopy> {
  let PseudoInstr = opName # "_" # getAddrName<addrKindCopy>.ret;
  let mayLoad = 0;
  let mayStore = 1;
}

multiclass MTBUF_Pseudo_Stores<string opName, RegisterClass vdataClass,
                               ValueType store_vt = i32,
                               SDPatternOperator st = null_frag> {

  def _OFFSET : MTBUF_Store_Pseudo <opName, BUFAddrKind.Offset, vdataClass,
    [(st store_vt:$vdata, (MUBUFOffset v4i32:$srsrc, i32:$soffset,
                                       i16:$offset, i8:$dfmt, i8:$nfmt, i1:$glc,
                                       i1:$slc, i1:$tfe))]>,
    MTBUFAddr64Table<0>;

  def _ADDR64 : MTBUF_Store_Pseudo <opName, BUFAddrKind.Addr64, vdataClass,
    [(st store_vt:$vdata, (MUBUFAddr64 v4i32:$srsrc, i64:$vaddr, i32:$soffset,
                                       i16:$offset, i8:$dfmt, i8:$nfmt, i1:$glc,
                                       i1:$slc, i1:$tfe))]>,
    MTBUFAddr64Table<1>;

  def _OFFEN  : MTBUF_Store_Pseudo <opName, BUFAddrKind.OffEn, vdataClass>;
  def _IDXEN  : MTBUF_Store_Pseudo <opName, BUFAddrKind.IdxEn, vdataClass>;
  def _BOTHEN : MTBUF_Store_Pseudo <opName, BUFAddrKind.BothEn, vdataClass>;

  let DisableWQM = 1 in {
    def _OFFSET_exact : MTBUF_Store_Pseudo <opName, BUFAddrKind.Offset, vdataClass>;
    def _OFFEN_exact  : MTBUF_Store_Pseudo <opName, BUFAddrKind.OffEn, vdataClass>;
    def _IDXEN_exact  : MTBUF_Store_Pseudo <opName, BUFAddrKind.IdxEn, vdataClass>;
    def _BOTHEN_exact : MTBUF_Store_Pseudo <opName, BUFAddrKind.BothEn, vdataClass>;
  }
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
  let Size = 8;
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
         offset:$offset, GLC:$glc, slc:$slc, tfe:$tfe),
    (ins vaddrClass:$vaddr, SReg_128:$srsrc, SCSrc_b32:$soffset,
         offset:$offset, GLC:$glc, slc:$slc, tfe:$tfe)
  );
  dag InsData = !if(!empty(vaddrList),
    (ins vdataClass:$vdata,                    SReg_128:$srsrc,
         SCSrc_b32:$soffset, offset:$offset, GLC:$glc, slc:$slc, tfe:$tfe),
    (ins vdataClass:$vdata, vaddrClass:$vaddr, SReg_128:$srsrc,
         SCSrc_b32:$soffset, offset:$offset, GLC:$glc, slc:$slc, tfe:$tfe)
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
                         bit HasTiedDest = 0,
                         list<dag> pattern=[],
                         // Workaround bug bz30254
                         int addrKindCopy = addrKind>
  : MUBUF_Pseudo<opName,
                 (outs vdataClass:$vdata),
                 !con(getMUBUFIns<addrKindCopy>.ret, !if(HasTiedDest, (ins vdataClass:$vdata_in), (ins))),
                 " $vdata, " # getMUBUFAsmOps<addrKindCopy>.ret # "$glc$slc$tfe",
                 pattern>,
    MUBUF_SetupAddr<addrKindCopy> {
  let PseudoInstr = opName # "_" # getAddrName<addrKindCopy>.ret;
  let Constraints = !if(HasTiedDest, "$vdata = $vdata_in", "");
  let mayLoad = 1;
  let mayStore = 0;
  let maybeAtomic = 1;
}

// FIXME: tfe can't be an operand because it requires a separate
// opcode because it needs an N+1 register class dest register.
multiclass MUBUF_Pseudo_Loads<string opName, RegisterClass vdataClass,
                              ValueType load_vt = i32,
                              SDPatternOperator ld = null_frag,
                              bit TiedDest = 0> {

  def _OFFSET : MUBUF_Load_Pseudo <opName, BUFAddrKind.Offset, vdataClass,
    TiedDest,
    [(set load_vt:$vdata,
     (ld (MUBUFOffset v4i32:$srsrc, i32:$soffset, i16:$offset, i1:$glc, i1:$slc, i1:$tfe)))]>,
    MUBUFAddr64Table<0>;

  def _ADDR64 : MUBUF_Load_Pseudo <opName, BUFAddrKind.Addr64, vdataClass,
    TiedDest,
    [(set load_vt:$vdata,
     (ld (MUBUFAddr64 v4i32:$srsrc, i64:$vaddr, i32:$soffset, i16:$offset, i1:$glc, i1:$slc, i1:$tfe)))]>,
    MUBUFAddr64Table<1>;

  def _OFFEN  : MUBUF_Load_Pseudo <opName, BUFAddrKind.OffEn, vdataClass, TiedDest>;
  def _IDXEN  : MUBUF_Load_Pseudo <opName, BUFAddrKind.IdxEn, vdataClass, TiedDest>;
  def _BOTHEN : MUBUF_Load_Pseudo <opName, BUFAddrKind.BothEn, vdataClass, TiedDest>;

  let DisableWQM = 1 in {
    def _OFFSET_exact : MUBUF_Load_Pseudo <opName, BUFAddrKind.Offset, vdataClass, TiedDest>;
    def _OFFEN_exact  : MUBUF_Load_Pseudo <opName, BUFAddrKind.OffEn, vdataClass, TiedDest>;
    def _IDXEN_exact  : MUBUF_Load_Pseudo <opName, BUFAddrKind.IdxEn, vdataClass, TiedDest>;
    def _BOTHEN_exact : MUBUF_Load_Pseudo <opName, BUFAddrKind.BothEn, vdataClass, TiedDest>;
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
  let maybeAtomic = 1;
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
  let maybeAtomic = 1;
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

  def _OFFSET_RTN : MUBUF_AtomicRet_Pseudo <opName, BUFAddrKind.Offset, vdataClass,
    [(set vdataType:$vdata,
     (atomic (MUBUFOffsetAtomic v4i32:$srsrc, i32:$soffset, i16:$offset, i1:$slc),
             vdataType:$vdata_in))]>,
    MUBUFAddr64Table <0, "_RTN">;

  def _ADDR64_RTN : MUBUF_AtomicRet_Pseudo <opName, BUFAddrKind.Addr64, vdataClass,
    [(set vdataType:$vdata,
     (atomic (MUBUFAddr64Atomic v4i32:$srsrc, i64:$vaddr, i32:$soffset, i16:$offset, i1:$slc),
             vdataType:$vdata_in))]>,
    MUBUFAddr64Table <1, "_RTN">;

  def _OFFEN_RTN  : MUBUF_AtomicRet_Pseudo <opName, BUFAddrKind.OffEn,  vdataClass>;
  def _IDXEN_RTN  : MUBUF_AtomicRet_Pseudo <opName, BUFAddrKind.IdxEn,  vdataClass>;
  def _BOTHEN_RTN : MUBUF_AtomicRet_Pseudo <opName, BUFAddrKind.BothEn, vdataClass>;
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
defm BUFFER_LOAD_DWORDX3 : MUBUF_Pseudo_Loads <
  "buffer_load_dwordx3", VReg_96, untyped, mubuf_load
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
  "buffer_store_dword", VGPR_32, i32, store_global
>;
defm BUFFER_STORE_DWORDX2 : MUBUF_Pseudo_Stores <
  "buffer_store_dwordx2", VReg_64, v2i32, store_global
>;
defm BUFFER_STORE_DWORDX3 : MUBUF_Pseudo_Stores <
  "buffer_store_dwordx3", VReg_96, untyped, store_global
>;
defm BUFFER_STORE_DWORDX4 : MUBUF_Pseudo_Stores <
  "buffer_store_dwordx4", VReg_128, v4i32, store_global
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

let SubtargetPredicate = HasD16LoadStore in {

defm BUFFER_LOAD_UBYTE_D16 : MUBUF_Pseudo_Loads <
  "buffer_load_ubyte_d16", VGPR_32, i32
>;

defm BUFFER_LOAD_UBYTE_D16_HI : MUBUF_Pseudo_Loads <
  "buffer_load_ubyte_d16_hi", VGPR_32, i32, null_frag, 1
>;

defm BUFFER_LOAD_SBYTE_D16 : MUBUF_Pseudo_Loads <
  "buffer_load_sbyte_d16", VGPR_32, i32
>;

defm BUFFER_LOAD_SBYTE_D16_HI : MUBUF_Pseudo_Loads <
  "buffer_load_sbyte_d16_hi", VGPR_32, i32, null_frag, 1
>;

defm BUFFER_LOAD_SHORT_D16 : MUBUF_Pseudo_Loads <
  "buffer_load_short_d16", VGPR_32, i32
>;

defm BUFFER_LOAD_SHORT_D16_HI : MUBUF_Pseudo_Loads <
  "buffer_load_short_d16_hi", VGPR_32, i32, null_frag, 1
>;

defm BUFFER_STORE_BYTE_D16_HI : MUBUF_Pseudo_Stores <
  "buffer_store_byte_d16_hi", VGPR_32, i32
>;

defm BUFFER_STORE_SHORT_D16_HI : MUBUF_Pseudo_Stores <
  "buffer_store_short_d16_hi", VGPR_32, i32
>;

} // End HasD16LoadStore

def BUFFER_WBINVL1 : MUBUF_Invalidate <"buffer_wbinvl1",
                                       int_amdgcn_buffer_wbinvl1>;

//===----------------------------------------------------------------------===//
// MTBUF Instructions
//===----------------------------------------------------------------------===//

defm TBUFFER_LOAD_FORMAT_X     : MTBUF_Pseudo_Loads  <"tbuffer_load_format_x",     VGPR_32>;
defm TBUFFER_LOAD_FORMAT_XY    : MTBUF_Pseudo_Loads  <"tbuffer_load_format_xy",    VReg_64>;
defm TBUFFER_LOAD_FORMAT_XYZ   : MTBUF_Pseudo_Loads  <"tbuffer_load_format_xyz",   VReg_128>;
defm TBUFFER_LOAD_FORMAT_XYZW  : MTBUF_Pseudo_Loads  <"tbuffer_load_format_xyzw",  VReg_128>;
defm TBUFFER_STORE_FORMAT_X    : MTBUF_Pseudo_Stores <"tbuffer_store_format_x",    VGPR_32>;
defm TBUFFER_STORE_FORMAT_XY   : MTBUF_Pseudo_Stores <"tbuffer_store_format_xy",   VReg_64>;
defm TBUFFER_STORE_FORMAT_XYZ  : MTBUF_Pseudo_Stores <"tbuffer_store_format_xyz",  VReg_128>;
defm TBUFFER_STORE_FORMAT_XYZW : MTBUF_Pseudo_Stores <"tbuffer_store_format_xyzw", VReg_128>;

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

let Predicates = [isGCN] in {

// Offset in an 32-bit VGPR
def : Pat <
  (SIload_constant v4i32:$sbase, i32:$voff),
  (BUFFER_LOAD_DWORD_OFFEN $voff, $sbase, (i32 0), 0, 0, 0, 0)
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

defm : MUBUF_LoadIntrinsicPat<SIbuffer_load_format, f32, "BUFFER_LOAD_FORMAT_X">;
defm : MUBUF_LoadIntrinsicPat<SIbuffer_load_format, v2f32, "BUFFER_LOAD_FORMAT_XY">;
defm : MUBUF_LoadIntrinsicPat<SIbuffer_load_format, v4f32, "BUFFER_LOAD_FORMAT_XYZW">;
defm : MUBUF_LoadIntrinsicPat<SIbuffer_load, f32, "BUFFER_LOAD_DWORD">;
defm : MUBUF_LoadIntrinsicPat<SIbuffer_load, v2f32, "BUFFER_LOAD_DWORDX2">;
defm : MUBUF_LoadIntrinsicPat<SIbuffer_load, v4f32, "BUFFER_LOAD_DWORDX4">;

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
    (!cast<MUBUF_Pseudo>(opcode # _OFFSET_RTN) $vdata_in, $rsrc, $soffset,
                                        (as_i16imm $offset), (as_i1imm $slc))
  >;

  def : Pat<
    (name i32:$vdata_in, v4i32:$rsrc, i32:$vindex,
          (MUBUFIntrinsicOffset i32:$soffset, i16:$offset),
          imm:$slc),
    (!cast<MUBUF_Pseudo>(opcode # _IDXEN_RTN) $vdata_in, $vindex, $rsrc, $soffset,
                                       (as_i16imm $offset), (as_i1imm $slc))
  >;

  def : Pat<
    (name i32:$vdata_in, v4i32:$rsrc, 0,
          (MUBUFIntrinsicVOffset i32:$soffset, i16:$offset, i32:$voffset),
          imm:$slc),
    (!cast<MUBUF_Pseudo>(opcode # _OFFEN_RTN) $vdata_in, $voffset, $rsrc, $soffset,
                                       (as_i16imm $offset), (as_i1imm $slc))
  >;

  def : Pat<
    (name i32:$vdata_in, v4i32:$rsrc, i32:$vindex,
          (MUBUFIntrinsicVOffset i32:$soffset, i16:$offset, i32:$voffset),
          imm:$slc),
    (!cast<MUBUF_Pseudo>(opcode # _BOTHEN_RTN)
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
    (BUFFER_ATOMIC_CMPSWAP_OFFSET_RTN
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
    (BUFFER_ATOMIC_CMPSWAP_IDXEN_RTN
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
    (BUFFER_ATOMIC_CMPSWAP_OFFEN_RTN
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
    (BUFFER_ATOMIC_CMPSWAP_BOTHEN_RTN
      (REG_SEQUENCE VReg_64, $data, sub0, $cmp, sub1),
      (REG_SEQUENCE VReg_64, $vindex, sub0, $voffset, sub1),
      $rsrc, $soffset, (as_i16imm $offset), (as_i1imm $slc)),
    sub0)
>;


class MUBUFLoad_PatternADDR64 <MUBUF_Pseudo Instr_ADDR64, ValueType vt,
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
     (Instr_ADDR64 $vaddr, $srsrc, $soffset, $offset, 0, $slc, 0)
  >;

  def : Pat <
    (vt (atomic_ld (MUBUFOffsetNoGLC v4i32:$rsrc, i32:$soffset, i16:$offset))),
    (Instr_OFFSET $rsrc, $soffset, (as_i16imm $offset), 0, 0, 0)
  >;
}

let Predicates = [isSICI] in {
def : MUBUFLoad_PatternADDR64 <BUFFER_LOAD_SBYTE_ADDR64, i32, sextloadi8_constant>;
def : MUBUFLoad_PatternADDR64 <BUFFER_LOAD_UBYTE_ADDR64, i32, az_extloadi8_constant>;
def : MUBUFLoad_PatternADDR64 <BUFFER_LOAD_SSHORT_ADDR64, i32, sextloadi16_constant>;
def : MUBUFLoad_PatternADDR64 <BUFFER_LOAD_USHORT_ADDR64, i32, az_extloadi16_constant>;

defm : MUBUFLoad_Atomic_Pattern <BUFFER_LOAD_DWORD_ADDR64, BUFFER_LOAD_DWORD_OFFSET, i32, mubuf_load_atomic>;
defm : MUBUFLoad_Atomic_Pattern <BUFFER_LOAD_DWORDX2_ADDR64, BUFFER_LOAD_DWORDX2_OFFSET, i64, mubuf_load_atomic>;
} // End Predicates = [isSICI]

multiclass MUBUFLoad_Pattern <MUBUF_Pseudo Instr_OFFSET, ValueType vt,
                               PatFrag ld> {

  def : Pat <
    (vt (ld (MUBUFOffset v4i32:$srsrc, i32:$soffset,
                          i16:$offset, i1:$glc, i1:$slc, i1:$tfe))),
    (Instr_OFFSET $srsrc, $soffset, $offset, $glc, $slc, $tfe)
  >;
}

let Predicates = [Has16BitInsts] in {

defm : MUBUFLoad_Pattern <BUFFER_LOAD_SBYTE_OFFSET, i16, sextloadi8_constant>;
defm : MUBUFLoad_Pattern <BUFFER_LOAD_UBYTE_OFFSET, i16, az_extloadi8_constant>;
defm : MUBUFLoad_Pattern <BUFFER_LOAD_SBYTE_OFFSET, i16, mubuf_sextloadi8>;
defm : MUBUFLoad_Pattern <BUFFER_LOAD_UBYTE_OFFSET, i16, mubuf_az_extloadi8>;

defm : MUBUFLoad_Pattern <BUFFER_LOAD_USHORT_OFFSET, i16, mubuf_load>;

} // End Predicates = [Has16BitInsts]

multiclass MUBUFScratchLoadPat <MUBUF_Pseudo InstrOffen,
                                MUBUF_Pseudo InstrOffset,
                                ValueType vt, PatFrag ld> {
  def : Pat <
    (vt (ld (MUBUFScratchOffen v4i32:$srsrc, i32:$vaddr,
                               i32:$soffset, u16imm:$offset))),
    (InstrOffen $vaddr, $srsrc, $soffset, $offset, 0, 0, 0)
  >;

  def : Pat <
    (vt (ld (MUBUFScratchOffset v4i32:$srsrc, i32:$soffset, u16imm:$offset))),
    (InstrOffset $srsrc, $soffset, $offset, 0, 0, 0)
  >;
}

// XXX - Is it possible to have a complex pattern in a PatFrag?
multiclass MUBUFScratchLoadPat_Hi16 <MUBUF_Pseudo InstrOffen,
                                MUBUF_Pseudo InstrOffset,
                                ValueType vt, PatFrag ld> {
  def : Pat <
    (build_vector vt:$lo, (vt (ld (MUBUFScratchOffen v4i32:$srsrc, i32:$vaddr,
                                 i32:$soffset, u16imm:$offset)))),
    (v2i16 (InstrOffen $vaddr, $srsrc, $soffset, $offset, 0, 0, 0, $lo))
  >;

  def : Pat <
    (build_vector f16:$lo, (f16 (bitconvert (vt (ld (MUBUFScratchOffen v4i32:$srsrc, i32:$vaddr,
                               i32:$soffset, u16imm:$offset)))))),
    (v2f16 (InstrOffen $vaddr, $srsrc, $soffset, $offset, 0, 0, 0, $lo))
  >;


  def : Pat <
    (build_vector vt:$lo, (vt (ld (MUBUFScratchOffset v4i32:$srsrc, i32:$soffset, u16imm:$offset)))),
    (v2i16 (InstrOffset $srsrc, $soffset, $offset, 0, 0, 0, $lo))
  >;

  def : Pat <
    (build_vector f16:$lo, (f16 (bitconvert (vt (ld (MUBUFScratchOffset v4i32:$srsrc, i32:$soffset, u16imm:$offset)))))),
    (v2f16 (InstrOffset $srsrc, $soffset, $offset, 0, 0, 0, $lo))
  >;
}

defm : MUBUFScratchLoadPat <BUFFER_LOAD_SBYTE_OFFEN, BUFFER_LOAD_SBYTE_OFFSET, i32, sextloadi8_private>;
defm : MUBUFScratchLoadPat <BUFFER_LOAD_UBYTE_OFFEN, BUFFER_LOAD_UBYTE_OFFSET, i32, az_extloadi8_private>;
defm : MUBUFScratchLoadPat <BUFFER_LOAD_SBYTE_OFFEN, BUFFER_LOAD_SBYTE_OFFSET, i16, sextloadi8_private>;
defm : MUBUFScratchLoadPat <BUFFER_LOAD_UBYTE_OFFEN, BUFFER_LOAD_UBYTE_OFFSET, i16, az_extloadi8_private>;
defm : MUBUFScratchLoadPat <BUFFER_LOAD_SSHORT_OFFEN, BUFFER_LOAD_SSHORT_OFFSET, i32, sextloadi16_private>;
defm : MUBUFScratchLoadPat <BUFFER_LOAD_USHORT_OFFEN, BUFFER_LOAD_USHORT_OFFSET, i32, az_extloadi16_private>;
defm : MUBUFScratchLoadPat <BUFFER_LOAD_USHORT_OFFEN, BUFFER_LOAD_USHORT_OFFSET, i16, load_private>;
defm : MUBUFScratchLoadPat <BUFFER_LOAD_DWORD_OFFEN, BUFFER_LOAD_DWORD_OFFSET, i32, load_private>;
defm : MUBUFScratchLoadPat <BUFFER_LOAD_DWORDX2_OFFEN, BUFFER_LOAD_DWORDX2_OFFSET, v2i32, load_private>;
defm : MUBUFScratchLoadPat <BUFFER_LOAD_DWORDX4_OFFEN, BUFFER_LOAD_DWORDX4_OFFSET, v4i32, load_private>;

let Predicates = [HasD16LoadStore] in {
defm : MUBUFScratchLoadPat_Hi16<BUFFER_LOAD_SHORT_D16_HI_OFFEN, BUFFER_LOAD_SHORT_D16_HI_OFFSET, i16, load_private>;
defm : MUBUFScratchLoadPat_Hi16<BUFFER_LOAD_UBYTE_D16_HI_OFFEN, BUFFER_LOAD_UBYTE_D16_HI_OFFSET, i16, az_extloadi8_private>;
defm : MUBUFScratchLoadPat_Hi16<BUFFER_LOAD_SBYTE_D16_HI_OFFEN, BUFFER_LOAD_SBYTE_D16_HI_OFFSET, i16, sextloadi8_private>;
}

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
     (Instr_ADDR64 $val, $vaddr, $srsrc, $soffset, $offset, 0, $slc, 0)
  >;

  def : Pat <
    (atomic_st (MUBUFOffsetNoGLC v4i32:$rsrc, i32:$soffset, i16:$offset), vt:$val),
    (Instr_OFFSET $val, $rsrc, $soffset, (as_i16imm $offset), 0, 0, 0)
  >;
}
let Predicates = [isSICI] in {
defm : MUBUFStore_Atomic_Pattern <BUFFER_STORE_DWORD_ADDR64, BUFFER_STORE_DWORD_OFFSET, i32, store_atomic_global>;
defm : MUBUFStore_Atomic_Pattern <BUFFER_STORE_DWORDX2_ADDR64, BUFFER_STORE_DWORDX2_OFFSET, i64, store_atomic_global>;
} // End Predicates = [isSICI]


multiclass MUBUFStore_Pattern <MUBUF_Pseudo Instr_OFFSET, ValueType vt,
                               PatFrag st> {

  def : Pat <
    (st vt:$vdata, (MUBUFOffset v4i32:$srsrc, i32:$soffset,
                                      i16:$offset, i1:$glc, i1:$slc, i1:$tfe)),
    (Instr_OFFSET $vdata, $srsrc, $soffset, $offset, $glc, $slc, $tfe)
  >;
}

defm : MUBUFStore_Pattern <BUFFER_STORE_BYTE_OFFSET, i16, truncstorei8_global>;
defm : MUBUFStore_Pattern <BUFFER_STORE_SHORT_OFFSET, i16, store_global>;

multiclass MUBUFScratchStorePat <MUBUF_Pseudo InstrOffen,
                                 MUBUF_Pseudo InstrOffset,
                                 ValueType vt, PatFrag st> {
  def : Pat <
    (st vt:$value, (MUBUFScratchOffen v4i32:$srsrc, i32:$vaddr,
                                      i32:$soffset, u16imm:$offset)),
    (InstrOffen $value, $vaddr, $srsrc, $soffset, $offset, 0, 0, 0)
  >;

  def : Pat <
    (st vt:$value, (MUBUFScratchOffset v4i32:$srsrc, i32:$soffset,
                                       u16imm:$offset)),
    (InstrOffset $value, $srsrc, $soffset, $offset, 0, 0, 0)
  >;
}

defm : MUBUFScratchStorePat <BUFFER_STORE_BYTE_OFFEN, BUFFER_STORE_BYTE_OFFSET, i32, truncstorei8_private>;
defm : MUBUFScratchStorePat <BUFFER_STORE_SHORT_OFFEN, BUFFER_STORE_SHORT_OFFSET, i32, truncstorei16_private>;
defm : MUBUFScratchStorePat <BUFFER_STORE_BYTE_OFFEN, BUFFER_STORE_BYTE_OFFSET, i16, truncstorei8_private>;
defm : MUBUFScratchStorePat <BUFFER_STORE_SHORT_OFFEN, BUFFER_STORE_SHORT_OFFSET, i16, store_private>;
defm : MUBUFScratchStorePat <BUFFER_STORE_DWORD_OFFEN, BUFFER_STORE_DWORD_OFFSET, i32, store_private>;
defm : MUBUFScratchStorePat <BUFFER_STORE_DWORDX2_OFFEN, BUFFER_STORE_DWORDX2_OFFSET, v2i32, store_private>;
defm : MUBUFScratchStorePat <BUFFER_STORE_DWORDX4_OFFEN, BUFFER_STORE_DWORDX4_OFFSET, v4i32, store_private>;


let Predicates = [HasD16LoadStore] in {
 // Hiding the extract high pattern in the PatFrag seems to not
 // automatically increase the complexity.
let AddedComplexity = 1 in {
defm : MUBUFScratchStorePat <BUFFER_STORE_SHORT_D16_HI_OFFEN, BUFFER_STORE_SHORT_D16_HI_OFFSET, i32, store_hi16_private>;
defm : MUBUFScratchStorePat <BUFFER_STORE_BYTE_D16_HI_OFFEN, BUFFER_STORE_BYTE_D16_HI_OFFSET, i32, truncstorei8_hi16_private>;
}
}

//===----------------------------------------------------------------------===//
// MTBUF Patterns
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// tbuffer_load/store_format patterns
//===----------------------------------------------------------------------===//

multiclass MTBUF_LoadIntrinsicPat<SDPatternOperator name, ValueType vt,
                                  string opcode> {
  def : Pat<
    (vt (name v4i32:$rsrc, 0, 0, i32:$soffset, imm:$offset,
              imm:$dfmt, imm:$nfmt, imm:$glc, imm:$slc)),
    (!cast<MTBUF_Pseudo>(opcode # _OFFSET) $rsrc, $soffset, (as_i16imm $offset),
      (as_i8imm $dfmt), (as_i8imm $nfmt), (as_i1imm $glc), (as_i1imm $slc), 0)
  >;

  def : Pat<
    (vt (name v4i32:$rsrc, i32:$vindex, 0, i32:$soffset, imm:$offset,
              imm:$dfmt, imm:$nfmt, imm:$glc, imm:$slc)),
    (!cast<MTBUF_Pseudo>(opcode # _IDXEN) $vindex, $rsrc, $soffset, (as_i16imm $offset),
      (as_i8imm $dfmt), (as_i8imm $nfmt), (as_i1imm $glc), (as_i1imm $slc), 0)
  >;

  def : Pat<
    (vt (name v4i32:$rsrc, 0, i32:$voffset, i32:$soffset, imm:$offset,
              imm:$dfmt, imm:$nfmt, imm:$glc, imm:$slc)),
    (!cast<MTBUF_Pseudo>(opcode # _OFFEN) $voffset, $rsrc, $soffset, (as_i16imm $offset),
      (as_i8imm $dfmt), (as_i8imm $nfmt), (as_i1imm $glc), (as_i1imm $slc), 0)
  >;

  def : Pat<
    (vt (name v4i32:$rsrc, i32:$vindex, i32:$voffset, i32:$soffset, imm:$offset,
              imm:$dfmt, imm:$nfmt, imm:$glc, imm:$slc)),
    (!cast<MTBUF_Pseudo>(opcode # _BOTHEN)
      (REG_SEQUENCE VReg_64, $vindex, sub0, $voffset, sub1),
      $rsrc, $soffset, (as_i16imm $offset),
      (as_i8imm $dfmt), (as_i8imm $nfmt), (as_i1imm $glc), (as_i1imm $slc), 0)
  >;
}

defm : MTBUF_LoadIntrinsicPat<SItbuffer_load, i32,   "TBUFFER_LOAD_FORMAT_X">;
defm : MTBUF_LoadIntrinsicPat<SItbuffer_load, v2i32, "TBUFFER_LOAD_FORMAT_XY">;
defm : MTBUF_LoadIntrinsicPat<SItbuffer_load, v4i32, "TBUFFER_LOAD_FORMAT_XYZW">;
defm : MTBUF_LoadIntrinsicPat<SItbuffer_load, f32,   "TBUFFER_LOAD_FORMAT_X">;
defm : MTBUF_LoadIntrinsicPat<SItbuffer_load, v2f32, "TBUFFER_LOAD_FORMAT_XY">;
defm : MTBUF_LoadIntrinsicPat<SItbuffer_load, v4f32, "TBUFFER_LOAD_FORMAT_XYZW">;

multiclass MTBUF_StoreIntrinsicPat<SDPatternOperator name, ValueType vt,
                                   string opcode> {
  def : Pat<
    (name vt:$vdata, v4i32:$rsrc, 0, 0, i32:$soffset, imm:$offset,
          imm:$dfmt, imm:$nfmt, imm:$glc, imm:$slc),
    (!cast<MTBUF_Pseudo>(opcode # _OFFSET_exact) $vdata, $rsrc, $soffset,
                                (as_i16imm $offset), (as_i8imm $dfmt),
                                (as_i8imm $nfmt), (as_i1imm $glc),
                                (as_i1imm $slc), 0)
  >;

  def : Pat<
    (name vt:$vdata, v4i32:$rsrc, i32:$vindex, 0, i32:$soffset, imm:$offset,
          imm:$dfmt, imm:$nfmt, imm:$glc, imm:$slc),
    (!cast<MTBUF_Pseudo>(opcode # _IDXEN_exact) $vdata, $vindex, $rsrc, $soffset,
                                   (as_i16imm $offset), (as_i8imm $dfmt),
                                   (as_i8imm $nfmt), (as_i1imm $glc),
                                   (as_i1imm $slc), 0)
  >;

  def : Pat<
    (name vt:$vdata, v4i32:$rsrc, 0, i32:$voffset, i32:$soffset, imm:$offset,
          imm:$dfmt, imm:$nfmt, imm:$glc, imm:$slc),
    (!cast<MTBUF_Pseudo>(opcode # _OFFEN_exact) $vdata, $voffset, $rsrc, $soffset,
                                   (as_i16imm $offset), (as_i8imm $dfmt),
                                   (as_i8imm $nfmt), (as_i1imm $glc),
                                   (as_i1imm $slc), 0)
  >;

  def : Pat<
    (name vt:$vdata, v4i32:$rsrc, i32:$vindex, i32:$voffset, i32:$soffset,
          imm:$offset, imm:$dfmt, imm:$nfmt, imm:$glc, imm:$slc),
    (!cast<MTBUF_Pseudo>(opcode # _BOTHEN_exact)
      $vdata,
      (REG_SEQUENCE VReg_64, $vindex, sub0, $voffset, sub1),
      $rsrc, $soffset, (as_i16imm $offset),
      (as_i8imm $dfmt), (as_i8imm $nfmt), (as_i1imm $glc), (as_i1imm $slc), 0)
  >;
}

defm : MTBUF_StoreIntrinsicPat<SItbuffer_store, i32,   "TBUFFER_STORE_FORMAT_X">;
defm : MTBUF_StoreIntrinsicPat<SItbuffer_store, v2i32, "TBUFFER_STORE_FORMAT_XY">;
defm : MTBUF_StoreIntrinsicPat<SItbuffer_store_x3, v4i32, "TBUFFER_STORE_FORMAT_XYZ">;
defm : MTBUF_StoreIntrinsicPat<SItbuffer_store, v4i32, "TBUFFER_STORE_FORMAT_XYZW">;
defm : MTBUF_StoreIntrinsicPat<SItbuffer_store, f32,   "TBUFFER_STORE_FORMAT_X">;
defm : MTBUF_StoreIntrinsicPat<SItbuffer_store, v2f32, "TBUFFER_STORE_FORMAT_XY">;
defm : MTBUF_StoreIntrinsicPat<SItbuffer_store_x3, v4f32, "TBUFFER_STORE_FORMAT_XYZ">;
defm : MTBUF_StoreIntrinsicPat<SItbuffer_store, v4f32, "TBUFFER_STORE_FORMAT_XYZW">;

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
  def _OFFSET_RTN_si : MUBUF_Real_si <op, !cast<MUBUF_Pseudo>(NAME#"_OFFSET_RTN")>;
  def _ADDR64_RTN_si : MUBUF_Real_si <op, !cast<MUBUF_Pseudo>(NAME#"_ADDR64_RTN")>;
  def _OFFEN_RTN_si  : MUBUF_Real_si <op, !cast<MUBUF_Pseudo>(NAME#"_OFFEN_RTN")>;
  def _IDXEN_RTN_si  : MUBUF_Real_si <op, !cast<MUBUF_Pseudo>(NAME#"_IDXEN_RTN")>;
  def _BOTHEN_RTN_si : MUBUF_Real_si <op, !cast<MUBUF_Pseudo>(NAME#"_BOTHEN_RTN")>;
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
defm BUFFER_LOAD_DWORDX3        : MUBUF_Real_AllAddr_si <0x0f>;
defm BUFFER_STORE_BYTE          : MUBUF_Real_AllAddr_si <0x18>;
defm BUFFER_STORE_SHORT         : MUBUF_Real_AllAddr_si <0x1a>;
defm BUFFER_STORE_DWORD         : MUBUF_Real_AllAddr_si <0x1c>;
defm BUFFER_STORE_DWORDX2       : MUBUF_Real_AllAddr_si <0x1d>;
defm BUFFER_STORE_DWORDX4       : MUBUF_Real_AllAddr_si <0x1e>;
defm BUFFER_STORE_DWORDX3       : MUBUF_Real_AllAddr_si <0x1f>;

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
// FIXME: Need to handle hazard for BUFFER_ATOMIC_FCMPSWAP_X2 on CI.
//defm BUFFER_ATOMIC_FCMPSWAP_X2  : MUBUF_Real_Atomic_si <0x5e">;   // isn't on VI
//defm BUFFER_ATOMIC_FMIN_X2      : MUBUF_Real_Atomic_si <0x5f>;    // isn't on VI
//defm BUFFER_ATOMIC_FMAX_X2      : MUBUF_Real_Atomic_si <0x60>;    // isn't on VI

def BUFFER_WBINVL1_SC_si        : MUBUF_Real_si <0x70, BUFFER_WBINVL1_SC>;
def BUFFER_WBINVL1_si           : MUBUF_Real_si <0x71, BUFFER_WBINVL1>;

class MTBUF_Real_si <bits<3> op, MTBUF_Pseudo ps> :
  MTBUF_Real<ps>,
  Enc64,
  SIMCInstr<ps.PseudoInstr, SIEncodingFamily.SI> {
  let AssemblerPredicate=isSICI;
  let DecoderNamespace="SICI";

  let Inst{11-0}  = !if(ps.has_offset, offset, ?);
  let Inst{12}    = ps.offen;
  let Inst{13}    = ps.idxen;
  let Inst{14}    = !if(ps.has_glc, glc, ps.glc_value);
  let Inst{15}    = ps.addr64;
  let Inst{18-16} = op;
  let Inst{22-19} = !if(ps.has_dfmt, dfmt, ps.dfmt_value);
  let Inst{25-23} = !if(ps.has_nfmt, nfmt, ps.nfmt_value);
  let Inst{31-26} = 0x3a; //encoding
  let Inst{39-32} = !if(ps.has_vaddr, vaddr, ?);
  let Inst{47-40} = !if(ps.has_vdata, vdata, ?);
  let Inst{52-48} = !if(ps.has_srsrc, srsrc{6-2}, ?);
  let Inst{54}    = !if(ps.has_slc, slc, ?);
  let Inst{55}    = !if(ps.has_tfe, tfe, ?);
  let Inst{63-56} = !if(ps.has_soffset, soffset, ?);
}

multiclass MTBUF_Real_AllAddr_si<bits<3> op> {
  def _OFFSET_si : MTBUF_Real_si <op, !cast<MTBUF_Pseudo>(NAME#"_OFFSET")>;
  def _ADDR64_si : MTBUF_Real_si <op, !cast<MTBUF_Pseudo>(NAME#"_ADDR64")>;
  def _OFFEN_si  : MTBUF_Real_si <op, !cast<MTBUF_Pseudo>(NAME#"_OFFEN")>;
  def _IDXEN_si  : MTBUF_Real_si <op, !cast<MTBUF_Pseudo>(NAME#"_IDXEN")>;
  def _BOTHEN_si : MTBUF_Real_si <op, !cast<MTBUF_Pseudo>(NAME#"_BOTHEN")>;
}

defm TBUFFER_LOAD_FORMAT_X     : MTBUF_Real_AllAddr_si <0>;
defm TBUFFER_LOAD_FORMAT_XY    : MTBUF_Real_AllAddr_si <1>;
//defm TBUFFER_LOAD_FORMAT_XYZ   : MTBUF_Real_AllAddr_si <2>;
defm TBUFFER_LOAD_FORMAT_XYZW  : MTBUF_Real_AllAddr_si <3>;
defm TBUFFER_STORE_FORMAT_X    : MTBUF_Real_AllAddr_si <4>;
defm TBUFFER_STORE_FORMAT_XY   : MTBUF_Real_AllAddr_si <5>;
defm TBUFFER_STORE_FORMAT_XYZ  : MTBUF_Real_AllAddr_si <6>;
defm TBUFFER_STORE_FORMAT_XYZW : MTBUF_Real_AllAddr_si <7>;

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
  def _OFFSET_RTN_vi : MUBUF_Real_vi <op, !cast<MUBUF_Pseudo>(NAME#"_OFFSET_RTN")>;
  def _OFFEN_RTN_vi  : MUBUF_Real_vi <op, !cast<MUBUF_Pseudo>(NAME#"_OFFEN_RTN")>;
  def _IDXEN_RTN_vi  : MUBUF_Real_vi <op, !cast<MUBUF_Pseudo>(NAME#"_IDXEN_RTN")>;
  def _BOTHEN_RTN_vi : MUBUF_Real_vi <op, !cast<MUBUF_Pseudo>(NAME#"_BOTHEN_RTN")>;
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
defm BUFFER_LOAD_DWORDX3        : MUBUF_Real_AllAddr_vi <0x16>;
defm BUFFER_LOAD_DWORDX4        : MUBUF_Real_AllAddr_vi <0x17>;
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
  Enc64,
  SIMCInstr<ps.PseudoInstr, SIEncodingFamily.VI> {
  let AssemblerPredicate=isVI;
  let DecoderNamespace="VI";

  let Inst{11-0}  = !if(ps.has_offset, offset, ?);
  let Inst{12}    = ps.offen;
  let Inst{13}    = ps.idxen;
  let Inst{14}    = !if(ps.has_glc, glc, ps.glc_value);
  let Inst{18-15} = op;
  let Inst{22-19} = !if(ps.has_dfmt, dfmt, ps.dfmt_value);
  let Inst{25-23} = !if(ps.has_nfmt, nfmt, ps.nfmt_value);
  let Inst{31-26} = 0x3a; //encoding
  let Inst{39-32} = !if(ps.has_vaddr, vaddr, ?);
  let Inst{47-40} = !if(ps.has_vdata, vdata, ?);
  let Inst{52-48} = !if(ps.has_srsrc, srsrc{6-2}, ?);
  let Inst{54}    = !if(ps.has_slc, slc, ?);
  let Inst{55}    = !if(ps.has_tfe, tfe, ?);
  let Inst{63-56} = !if(ps.has_soffset, soffset, ?);
}

multiclass MTBUF_Real_AllAddr_vi<bits<4> op> {
  def _OFFSET_vi : MTBUF_Real_vi <op, !cast<MTBUF_Pseudo>(NAME#"_OFFSET")>;
  def _OFFEN_vi  : MTBUF_Real_vi <op, !cast<MTBUF_Pseudo>(NAME#"_OFFEN")>;
  def _IDXEN_vi  : MTBUF_Real_vi <op, !cast<MTBUF_Pseudo>(NAME#"_IDXEN")>;
  def _BOTHEN_vi : MTBUF_Real_vi <op, !cast<MTBUF_Pseudo>(NAME#"_BOTHEN")>;
}

defm TBUFFER_LOAD_FORMAT_X     : MTBUF_Real_AllAddr_vi <0>;
defm TBUFFER_LOAD_FORMAT_XY    : MTBUF_Real_AllAddr_vi <1>;
//defm TBUFFER_LOAD_FORMAT_XYZ   : MTBUF_Real_AllAddr_vi <2>;
defm TBUFFER_LOAD_FORMAT_XYZW  : MTBUF_Real_AllAddr_vi <3>;
defm TBUFFER_STORE_FORMAT_X    : MTBUF_Real_AllAddr_vi <4>;
defm TBUFFER_STORE_FORMAT_XY   : MTBUF_Real_AllAddr_vi <5>;
defm TBUFFER_STORE_FORMAT_XYZ  : MTBUF_Real_AllAddr_vi <6>;
defm TBUFFER_STORE_FORMAT_XYZW : MTBUF_Real_AllAddr_vi <7>;
