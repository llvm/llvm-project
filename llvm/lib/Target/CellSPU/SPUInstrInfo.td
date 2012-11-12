//==- SPUInstrInfo.td - Describe the Cell SPU Instructions -*- tablegen -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Cell SPU Instructions:
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TODO Items (not urgent today, but would be nice, low priority)
//
// ANDBI, ORBI: SPU constructs a 4-byte constant for these instructions by
// concatenating the byte argument b as "bbbb". Could recognize this bit pattern
// in 16-bit and 32-bit constants and reduce instruction count.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Pseudo instructions:
//===----------------------------------------------------------------------===//

let hasCtrlDep = 1, Defs = [R1], Uses = [R1] in {
  def ADJCALLSTACKDOWN : Pseudo<(outs), (ins u16imm_i32:$amt),
                                "${:comment} ADJCALLSTACKDOWN",
                                [(callseq_start timm:$amt)]>;
  def ADJCALLSTACKUP   : Pseudo<(outs), (ins u16imm_i32:$amt),
                                "${:comment} ADJCALLSTACKUP",
                                [(callseq_end timm:$amt)]>;
  def HBR_LABEL        : Pseudo<(outs), (ins hbrtarget:$targ), 
                                "$targ:\t${:comment}branch hint target",[ ]>;
}

//===----------------------------------------------------------------------===//
// Loads:
// NB: The ordering is actually important, since the instruction selection
// will try each of the instructions in sequence, i.e., the D-form first with
// the 10-bit displacement, then the A-form with the 16 bit displacement, and
// finally the X-form with the register-register.
//===----------------------------------------------------------------------===//

let canFoldAsLoad = 1 in {
  class LoadDFormVec<ValueType vectype>
    : RI10Form<0b00101100, (outs VECREG:$rT), (ins dformaddr:$src),
               "lqd\t$rT, $src",
               LoadStore,
               [(set (vectype VECREG:$rT), (load dform_addr:$src))]>
  { }

  class LoadDForm<RegisterClass rclass>
    : RI10Form<0b00101100, (outs rclass:$rT), (ins dformaddr:$src),
               "lqd\t$rT, $src",
               LoadStore,
               [(set rclass:$rT, (load dform_addr:$src))]>
  { }

  multiclass LoadDForms
  {
    def v16i8: LoadDFormVec<v16i8>;
    def v8i16: LoadDFormVec<v8i16>;
    def v4i32: LoadDFormVec<v4i32>;
    def v2i64: LoadDFormVec<v2i64>;
    def v4f32: LoadDFormVec<v4f32>;
    def v2f64: LoadDFormVec<v2f64>;

    def r128:  LoadDForm<GPRC>;
    def r64:   LoadDForm<R64C>;
    def r32:   LoadDForm<R32C>;
    def f32:   LoadDForm<R32FP>;
    def f64:   LoadDForm<R64FP>;
    def r16:   LoadDForm<R16C>;
    def r8:    LoadDForm<R8C>;
  }

  class LoadAFormVec<ValueType vectype>
    : RI16Form<0b100001100, (outs VECREG:$rT), (ins addr256k:$src),
               "lqa\t$rT, $src",
               LoadStore,
               [(set (vectype VECREG:$rT), (load aform_addr:$src))]>
  { }

  class LoadAForm<RegisterClass rclass>
    : RI16Form<0b100001100, (outs rclass:$rT), (ins addr256k:$src),
               "lqa\t$rT, $src",
               LoadStore,
               [(set rclass:$rT, (load aform_addr:$src))]>
  { }

  multiclass LoadAForms
  {
    def v16i8: LoadAFormVec<v16i8>;
    def v8i16: LoadAFormVec<v8i16>;
    def v4i32: LoadAFormVec<v4i32>;
    def v2i64: LoadAFormVec<v2i64>;
    def v4f32: LoadAFormVec<v4f32>;
    def v2f64: LoadAFormVec<v2f64>;

    def r128:  LoadAForm<GPRC>;
    def r64:   LoadAForm<R64C>;
    def r32:   LoadAForm<R32C>;
    def f32:   LoadAForm<R32FP>;
    def f64:   LoadAForm<R64FP>;
    def r16:   LoadAForm<R16C>;
    def r8:    LoadAForm<R8C>;
  }

  class LoadXFormVec<ValueType vectype>
    : RRForm<0b00100011100, (outs VECREG:$rT), (ins memrr:$src),
             "lqx\t$rT, $src",
             LoadStore,
             [(set (vectype VECREG:$rT), (load xform_addr:$src))]>
  { }

  class LoadXForm<RegisterClass rclass>
    : RRForm<0b00100011100, (outs rclass:$rT), (ins memrr:$src),
             "lqx\t$rT, $src",
             LoadStore,
             [(set rclass:$rT, (load xform_addr:$src))]>
  { }

  multiclass LoadXForms
  {
    def v16i8: LoadXFormVec<v16i8>;
    def v8i16: LoadXFormVec<v8i16>;
    def v4i32: LoadXFormVec<v4i32>;
    def v2i64: LoadXFormVec<v2i64>;
    def v4f32: LoadXFormVec<v4f32>;
    def v2f64: LoadXFormVec<v2f64>;

    def r128:  LoadXForm<GPRC>;
    def r64:   LoadXForm<R64C>;
    def r32:   LoadXForm<R32C>;
    def f32:   LoadXForm<R32FP>;
    def f64:   LoadXForm<R64FP>;
    def r16:   LoadXForm<R16C>;
    def r8:    LoadXForm<R8C>;
  }

  defm LQA : LoadAForms;
  defm LQD : LoadDForms;
  defm LQX : LoadXForms;

/* Load quadword, PC relative: Not much use at this point in time.
   Might be of use later for relocatable code. It's effectively the
   same as LQA, but uses PC-relative addressing.
  def LQR : RI16Form<0b111001100, (outs VECREG:$rT), (ins s16imm:$disp),
                     "lqr\t$rT, $disp", LoadStore,
                     [(set VECREG:$rT, (load iaddr:$disp))]>;
 */
}

//===----------------------------------------------------------------------===//
// Stores:
//===----------------------------------------------------------------------===//
class StoreDFormVec<ValueType vectype>
  : RI10Form<0b00100100, (outs), (ins VECREG:$rT, dformaddr:$src),
             "stqd\t$rT, $src",
             LoadStore,
             [(store (vectype VECREG:$rT), dform_addr:$src)]>
{ }

class StoreDForm<RegisterClass rclass>
  : RI10Form<0b00100100, (outs), (ins rclass:$rT, dformaddr:$src),
             "stqd\t$rT, $src",
             LoadStore,
             [(store rclass:$rT, dform_addr:$src)]>
{ }

multiclass StoreDForms
{
  def v16i8: StoreDFormVec<v16i8>;
  def v8i16: StoreDFormVec<v8i16>;
  def v4i32: StoreDFormVec<v4i32>;
  def v2i64: StoreDFormVec<v2i64>;
  def v4f32: StoreDFormVec<v4f32>;
  def v2f64: StoreDFormVec<v2f64>;

  def r128:  StoreDForm<GPRC>;
  def r64:   StoreDForm<R64C>;
  def r32:   StoreDForm<R32C>;
  def f32:   StoreDForm<R32FP>;
  def f64:   StoreDForm<R64FP>;
  def r16:   StoreDForm<R16C>;
  def r8:    StoreDForm<R8C>;
}

class StoreAFormVec<ValueType vectype>
  : RI16Form<0b0010010, (outs), (ins VECREG:$rT, addr256k:$src),
             "stqa\t$rT, $src",
             LoadStore,
             [(store (vectype VECREG:$rT), aform_addr:$src)]>;

class StoreAForm<RegisterClass rclass>
  : RI16Form<0b001001, (outs), (ins rclass:$rT, addr256k:$src),
             "stqa\t$rT, $src",
             LoadStore,
             [(store rclass:$rT, aform_addr:$src)]>;

multiclass StoreAForms
{
  def v16i8: StoreAFormVec<v16i8>;
  def v8i16: StoreAFormVec<v8i16>;
  def v4i32: StoreAFormVec<v4i32>;
  def v2i64: StoreAFormVec<v2i64>;
  def v4f32: StoreAFormVec<v4f32>;
  def v2f64: StoreAFormVec<v2f64>;

  def r128:  StoreAForm<GPRC>;
  def r64:   StoreAForm<R64C>;
  def r32:   StoreAForm<R32C>;
  def f32:   StoreAForm<R32FP>;
  def f64:   StoreAForm<R64FP>;
  def r16:   StoreAForm<R16C>;
  def r8:    StoreAForm<R8C>;
}

class StoreXFormVec<ValueType vectype>
  : RRForm<0b00100100, (outs), (ins VECREG:$rT, memrr:$src),
           "stqx\t$rT, $src",
           LoadStore,
           [(store (vectype VECREG:$rT), xform_addr:$src)]>
{ }

class StoreXForm<RegisterClass rclass>
  : RRForm<0b00100100, (outs), (ins rclass:$rT, memrr:$src),
           "stqx\t$rT, $src",
           LoadStore,
           [(store rclass:$rT, xform_addr:$src)]>
{ }

multiclass StoreXForms
{
  def v16i8: StoreXFormVec<v16i8>;
  def v8i16: StoreXFormVec<v8i16>;
  def v4i32: StoreXFormVec<v4i32>;
  def v2i64: StoreXFormVec<v2i64>;
  def v4f32: StoreXFormVec<v4f32>;
  def v2f64: StoreXFormVec<v2f64>;

  def r128:  StoreXForm<GPRC>;
  def r64:   StoreXForm<R64C>;
  def r32:   StoreXForm<R32C>;
  def f32:   StoreXForm<R32FP>;
  def f64:   StoreXForm<R64FP>;
  def r16:   StoreXForm<R16C>;
  def r8:    StoreXForm<R8C>;
}

defm STQD : StoreDForms;
defm STQA : StoreAForms;
defm STQX : StoreXForms;

/* Store quadword, PC relative: Not much use at this point in time. Might
   be useful for relocatable code.
def STQR : RI16Form<0b111000100, (outs), (ins VECREG:$rT, s16imm:$disp),
                   "stqr\t$rT, $disp", LoadStore,
                   [(store VECREG:$rT, iaddr:$disp)]>;
*/

//===----------------------------------------------------------------------===//
// Generate Controls for Insertion:
//===----------------------------------------------------------------------===//

def CBD: RI7Form<0b10101111100, (outs VECREG:$rT), (ins shufaddr:$src),
    "cbd\t$rT, $src", ShuffleOp,
    [(set (v16i8 VECREG:$rT), (SPUshufmask dform2_addr:$src))]>;

def CBX: RRForm<0b00101011100, (outs VECREG:$rT), (ins memrr:$src),
    "cbx\t$rT, $src", ShuffleOp,
    [(set (v16i8 VECREG:$rT), (SPUshufmask xform_addr:$src))]>;

def CHD: RI7Form<0b10101111100, (outs VECREG:$rT), (ins shufaddr:$src),
    "chd\t$rT, $src", ShuffleOp,
    [(set (v8i16 VECREG:$rT), (SPUshufmask dform2_addr:$src))]>;

def CHX: RRForm<0b10101011100, (outs VECREG:$rT), (ins memrr:$src),
    "chx\t$rT, $src", ShuffleOp,
    [(set (v8i16 VECREG:$rT), (SPUshufmask xform_addr:$src))]>;

def CWD: RI7Form<0b01101111100, (outs VECREG:$rT), (ins shufaddr:$src),
    "cwd\t$rT, $src", ShuffleOp,
    [(set (v4i32 VECREG:$rT), (SPUshufmask dform2_addr:$src))]>;

def CWX: RRForm<0b01101011100, (outs VECREG:$rT), (ins memrr:$src),
    "cwx\t$rT, $src", ShuffleOp,
    [(set (v4i32 VECREG:$rT), (SPUshufmask xform_addr:$src))]>;

def CWDf32: RI7Form<0b01101111100, (outs VECREG:$rT), (ins shufaddr:$src),
    "cwd\t$rT, $src", ShuffleOp,
    [(set (v4f32 VECREG:$rT), (SPUshufmask dform2_addr:$src))]>;

def CWXf32: RRForm<0b01101011100, (outs VECREG:$rT), (ins memrr:$src),
    "cwx\t$rT, $src", ShuffleOp,
    [(set (v4f32 VECREG:$rT), (SPUshufmask xform_addr:$src))]>;

def CDD: RI7Form<0b11101111100, (outs VECREG:$rT), (ins shufaddr:$src),
    "cdd\t$rT, $src", ShuffleOp,
    [(set (v2i64 VECREG:$rT), (SPUshufmask dform2_addr:$src))]>;

def CDX: RRForm<0b11101011100, (outs VECREG:$rT), (ins memrr:$src),
    "cdx\t$rT, $src", ShuffleOp,
    [(set (v2i64 VECREG:$rT), (SPUshufmask xform_addr:$src))]>;

def CDDf64: RI7Form<0b11101111100, (outs VECREG:$rT), (ins shufaddr:$src),
    "cdd\t$rT, $src", ShuffleOp,
    [(set (v2f64 VECREG:$rT), (SPUshufmask dform2_addr:$src))]>;

def CDXf64: RRForm<0b11101011100, (outs VECREG:$rT), (ins memrr:$src),
    "cdx\t$rT, $src", ShuffleOp,
    [(set (v2f64 VECREG:$rT), (SPUshufmask xform_addr:$src))]>;

//===----------------------------------------------------------------------===//
// Constant formation:
//===----------------------------------------------------------------------===//

def ILHv8i16:
  RI16Form<0b110000010, (outs VECREG:$rT), (ins s16imm:$val),
    "ilh\t$rT, $val", ImmLoad,
    [(set (v8i16 VECREG:$rT), (v8i16 v8i16SExt16Imm:$val))]>;

def ILHr16:
  RI16Form<0b110000010, (outs R16C:$rT), (ins s16imm:$val),
    "ilh\t$rT, $val", ImmLoad,
    [(set R16C:$rT, immSExt16:$val)]>;

// Cell SPU doesn't have a native 8-bit immediate load, but ILH works ("with
// the right constant")
def ILHr8:
  RI16Form<0b110000010, (outs R8C:$rT), (ins s16imm_i8:$val),
    "ilh\t$rT, $val", ImmLoad,
    [(set R8C:$rT, immSExt8:$val)]>;

// IL does sign extension!

class ILInst<dag OOL, dag IOL, list<dag> pattern>:
  RI16Form<0b100000010, OOL, IOL, "il\t$rT, $val",
           ImmLoad, pattern>;

class ILVecInst<ValueType vectype, Operand immtype, PatLeaf xform>:
  ILInst<(outs VECREG:$rT), (ins immtype:$val),
         [(set (vectype VECREG:$rT), (vectype xform:$val))]>;

class ILRegInst<RegisterClass rclass, Operand immtype, PatLeaf xform>:
  ILInst<(outs rclass:$rT), (ins immtype:$val),
         [(set rclass:$rT, xform:$val)]>;

multiclass ImmediateLoad
{
  def v2i64: ILVecInst<v2i64, s16imm_i64, v2i64SExt16Imm>;
  def v4i32: ILVecInst<v4i32, s16imm_i32, v4i32SExt16Imm>;

  // TODO: Need v2f64, v4f32

  def r64: ILRegInst<R64C, s16imm_i64, immSExt16>;
  def r32: ILRegInst<R32C, s16imm_i32, immSExt16>;
  def f32: ILRegInst<R32FP, s16imm_f32, fpimmSExt16>;
  def f64: ILRegInst<R64FP, s16imm_f64, fpimmSExt16>;
}

defm IL : ImmediateLoad;

class ILHUInst<dag OOL, dag IOL, list<dag> pattern>:
  RI16Form<0b010000010, OOL, IOL, "ilhu\t$rT, $val",
           ImmLoad, pattern>;

class ILHUVecInst<ValueType vectype, Operand immtype, PatLeaf xform>:
  ILHUInst<(outs VECREG:$rT), (ins immtype:$val),
           [(set (vectype VECREG:$rT), (vectype xform:$val))]>;

class ILHURegInst<RegisterClass rclass, Operand immtype, PatLeaf xform>:
  ILHUInst<(outs rclass:$rT), (ins immtype:$val),
           [(set rclass:$rT, xform:$val)]>;

multiclass ImmLoadHalfwordUpper
{
  def v2i64: ILHUVecInst<v2i64, u16imm_i64, immILHUvec_i64>;
  def v4i32: ILHUVecInst<v4i32, u16imm_i32, immILHUvec>;

  def r64: ILHURegInst<R64C, u16imm_i64, hi16>;
  def r32: ILHURegInst<R32C, u16imm_i32, hi16>;

  // Loads the high portion of an address
  def hi: ILHURegInst<R32C, symbolHi, hi16>;

  // Used in custom lowering constant SFP loads:
  def f32: ILHURegInst<R32FP, f16imm, hi16_f32>;
}

defm ILHU : ImmLoadHalfwordUpper;

// Immediate load address (can also be used to load 18-bit unsigned constants,
// see the zext 16->32 pattern)

class ILAInst<dag OOL, dag IOL, list<dag> pattern>:
  RI18Form<0b1000010, OOL, IOL, "ila\t$rT, $val",
           LoadNOP, pattern>;

class ILAVecInst<ValueType vectype, Operand immtype, PatLeaf xform>:
  ILAInst<(outs VECREG:$rT), (ins immtype:$val),
          [(set (vectype VECREG:$rT), (vectype xform:$val))]>;

class ILARegInst<RegisterClass rclass, Operand immtype, PatLeaf xform>:
  ILAInst<(outs rclass:$rT), (ins immtype:$val),
          [(set rclass:$rT, xform:$val)]>;

multiclass ImmLoadAddress
{
  def v2i64: ILAVecInst<v2i64, u18imm, v2i64Uns18Imm>;
  def v4i32: ILAVecInst<v4i32, u18imm, v4i32Uns18Imm>;

  def r64: ILARegInst<R64C, u18imm_i64, imm18>;
  def r32: ILARegInst<R32C, u18imm, imm18>;
  def f32: ILARegInst<R32FP, f18imm, fpimm18>;
  def f64: ILARegInst<R64FP, f18imm_f64, fpimm18>;

  def hi: ILARegInst<R32C, symbolHi, imm18>;
  def lo: ILARegInst<R32C, symbolLo, imm18>;

  def lsa: ILAInst<(outs R32C:$rT), (ins symbolLSA:$val),
                   [(set R32C:$rT, imm18:$val)]>;
}

defm ILA : ImmLoadAddress;

// Immediate OR, Halfword Lower: The "other" part of loading large constants
// into 32-bit registers. See the anonymous pattern Pat<(i32 imm:$imm), ...>
// Note that these are really two operand instructions, but they're encoded
// as three operands with the first two arguments tied-to each other.

class IOHLInst<dag OOL, dag IOL, list<dag> pattern>:
  RI16Form<0b100000110, OOL, IOL, "iohl\t$rT, $val",
           ImmLoad, pattern>,
  RegConstraint<"$rS = $rT">,
  NoEncode<"$rS">;

class IOHLVecInst<ValueType vectype, Operand immtype /* , PatLeaf xform */>:
  IOHLInst<(outs VECREG:$rT), (ins VECREG:$rS, immtype:$val),
           [/* no pattern */]>;

class IOHLRegInst<RegisterClass rclass, Operand immtype /* , PatLeaf xform */>:
  IOHLInst<(outs rclass:$rT), (ins rclass:$rS, immtype:$val),
           [/* no pattern */]>;

multiclass ImmOrHalfwordLower
{
  def v2i64: IOHLVecInst<v2i64, u16imm_i64>;
  def v4i32: IOHLVecInst<v4i32, u16imm_i32>;

  def r32: IOHLRegInst<R32C, i32imm>;
  def f32: IOHLRegInst<R32FP, f32imm>;

  def lo: IOHLRegInst<R32C, symbolLo>;
}

defm IOHL: ImmOrHalfwordLower;

// Form select mask for bytes using immediate, used in conjunction with the
// SELB instruction:

class FSMBIVec<ValueType vectype>:
  RI16Form<0b101001100, (outs VECREG:$rT), (ins u16imm:$val),
          "fsmbi\t$rT, $val",
          SelectOp,
          [(set (vectype VECREG:$rT), (SPUselmask (i16 immU16:$val)))]>;

multiclass FormSelectMaskBytesImm
{
  def v16i8: FSMBIVec<v16i8>;
  def v8i16: FSMBIVec<v8i16>;
  def v4i32: FSMBIVec<v4i32>;
  def v2i64: FSMBIVec<v2i64>;
}

defm FSMBI : FormSelectMaskBytesImm;

// fsmb: Form select mask for bytes. N.B. Input operand, $rA, is 16-bits
class FSMBInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm_1<0b01101101100, OOL, IOL, "fsmb\t$rT, $rA", SelectOp,
             pattern>;

class FSMBRegInst<RegisterClass rclass, ValueType vectype>:
    FSMBInst<(outs VECREG:$rT), (ins rclass:$rA),
             [(set (vectype VECREG:$rT), (SPUselmask rclass:$rA))]>;

class FSMBVecInst<ValueType vectype>:
    FSMBInst<(outs VECREG:$rT), (ins VECREG:$rA),
             [(set (vectype VECREG:$rT),
                   (SPUselmask (vectype VECREG:$rA)))]>;

multiclass FormSelectMaskBits {
  def v16i8_r16: FSMBRegInst<R16C, v16i8>;
  def v16i8:     FSMBVecInst<v16i8>;
}

defm FSMB: FormSelectMaskBits;

// fsmh: Form select mask for halfwords. N.B., Input operand, $rA, is
// only 8-bits wide (even though it's input as 16-bits here)

class FSMHInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm_1<0b10101101100, OOL, IOL, "fsmh\t$rT, $rA", SelectOp,
             pattern>;

class FSMHRegInst<RegisterClass rclass, ValueType vectype>:
    FSMHInst<(outs VECREG:$rT), (ins rclass:$rA),
             [(set (vectype VECREG:$rT), (SPUselmask rclass:$rA))]>;

class FSMHVecInst<ValueType vectype>:
    FSMHInst<(outs VECREG:$rT), (ins VECREG:$rA),
             [(set (vectype VECREG:$rT),
                   (SPUselmask (vectype VECREG:$rA)))]>;

multiclass FormSelectMaskHalfword {
  def v8i16_r16: FSMHRegInst<R16C, v8i16>;
  def v8i16:     FSMHVecInst<v8i16>;
}

defm FSMH: FormSelectMaskHalfword;

// fsm: Form select mask for words. Like the other fsm* instructions,
// only the lower 4 bits of $rA are significant.

class FSMInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm_1<0b00101101100, OOL, IOL, "fsm\t$rT, $rA", SelectOp,
             pattern>;

class FSMRegInst<ValueType vectype, RegisterClass rclass>:
    FSMInst<(outs VECREG:$rT), (ins rclass:$rA),
            [(set (vectype VECREG:$rT), (SPUselmask rclass:$rA))]>;

class FSMVecInst<ValueType vectype>:
    FSMInst<(outs VECREG:$rT), (ins VECREG:$rA),
            [(set (vectype VECREG:$rT), (SPUselmask (vectype VECREG:$rA)))]>;

multiclass FormSelectMaskWord {
  def v4i32: FSMVecInst<v4i32>;

  def r32 :  FSMRegInst<v4i32, R32C>;
  def r16 :  FSMRegInst<v4i32, R16C>;
}

defm FSM : FormSelectMaskWord;

// Special case when used for i64 math operations
multiclass FormSelectMaskWord64 {
  def r32 : FSMRegInst<v2i64, R32C>;
  def r16 : FSMRegInst<v2i64, R16C>;
}

defm FSM64 : FormSelectMaskWord64;

//===----------------------------------------------------------------------===//
// Integer and Logical Operations:
//===----------------------------------------------------------------------===//

def AHv8i16:
  RRForm<0b00010011000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
    "ah\t$rT, $rA, $rB", IntegerOp,
    [(set (v8i16 VECREG:$rT), (int_spu_si_ah VECREG:$rA, VECREG:$rB))]>;

def : Pat<(add (v8i16 VECREG:$rA), (v8i16 VECREG:$rB)),
          (AHv8i16 VECREG:$rA, VECREG:$rB)>;

def AHr16:
  RRForm<0b00010011000, (outs R16C:$rT), (ins R16C:$rA, R16C:$rB),
    "ah\t$rT, $rA, $rB", IntegerOp,
    [(set R16C:$rT, (add R16C:$rA, R16C:$rB))]>;

def AHIvec:
    RI10Form<0b10111000, (outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
      "ahi\t$rT, $rA, $val", IntegerOp,
      [(set (v8i16 VECREG:$rT), (add (v8i16 VECREG:$rA),
                                     v8i16SExt10Imm:$val))]>;

def AHIr16:
  RI10Form<0b10111000, (outs R16C:$rT), (ins R16C:$rA, s10imm:$val),
    "ahi\t$rT, $rA, $val", IntegerOp,
    [(set R16C:$rT, (add R16C:$rA, i16ImmSExt10:$val))]>;

// v4i32, i32 add instruction:

class AInst<dag OOL, dag IOL, list<dag> pattern>:
  RRForm<0b00000011000, OOL, IOL,
         "a\t$rT, $rA, $rB", IntegerOp,
         pattern>;

class AVecInst<ValueType vectype>:
  AInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
        [(set (vectype VECREG:$rT), (add (vectype VECREG:$rA),
                                         (vectype VECREG:$rB)))]>;

class ARegInst<RegisterClass rclass>:
  AInst<(outs rclass:$rT), (ins rclass:$rA, rclass:$rB),
        [(set rclass:$rT, (add rclass:$rA, rclass:$rB))]>;
        
multiclass AddInstruction {
  def v4i32: AVecInst<v4i32>;
  def v16i8: AVecInst<v16i8>;
  def r32:   ARegInst<R32C>;
}

defm A : AddInstruction;

class AIInst<dag OOL, dag IOL, list<dag> pattern>:
    RI10Form<0b00111000, OOL, IOL,
             "ai\t$rT, $rA, $val", IntegerOp,
             pattern>;

class AIVecInst<ValueType vectype, PatLeaf immpred>:
    AIInst<(outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
            [(set (vectype VECREG:$rT), (add (vectype VECREG:$rA), immpred:$val))]>;

class AIFPVecInst<ValueType vectype, PatLeaf immpred>:
    AIInst<(outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
            [/* no pattern */]>;

class AIRegInst<RegisterClass rclass, PatLeaf immpred>:
    AIInst<(outs rclass:$rT), (ins rclass:$rA, s10imm_i32:$val),
           [(set rclass:$rT, (add rclass:$rA, immpred:$val))]>;

// This is used to add epsilons to floating point numbers in the f32 fdiv code:
class AIFPInst<RegisterClass rclass, PatLeaf immpred>:
    AIInst<(outs rclass:$rT), (ins rclass:$rA, s10imm_i32:$val),
           [/* no pattern */]>;

multiclass AddImmediate {
  def v4i32: AIVecInst<v4i32, v4i32SExt10Imm>;

  def r32: AIRegInst<R32C, i32ImmSExt10>;

  def v4f32: AIFPVecInst<v4f32, v4i32SExt10Imm>;
  def f32: AIFPInst<R32FP, i32ImmSExt10>;
}

defm AI : AddImmediate;

def SFHvec:
    RRForm<0b00010010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "sfh\t$rT, $rA, $rB", IntegerOp,
      [(set (v8i16 VECREG:$rT), (sub (v8i16 VECREG:$rA),
                                     (v8i16 VECREG:$rB)))]>;

def SFHr16:
    RRForm<0b00010010000, (outs R16C:$rT), (ins R16C:$rA, R16C:$rB),
      "sfh\t$rT, $rA, $rB", IntegerOp,
      [(set R16C:$rT, (sub R16C:$rB, R16C:$rA))]>;

def SFHIvec:
    RI10Form<0b10110000, (outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
      "sfhi\t$rT, $rA, $val", IntegerOp,
      [(set (v8i16 VECREG:$rT), (sub v8i16SExt10Imm:$val,
                                     (v8i16 VECREG:$rA)))]>;

def SFHIr16 : RI10Form<0b10110000, (outs R16C:$rT), (ins R16C:$rA, s10imm:$val),
  "sfhi\t$rT, $rA, $val", IntegerOp,
  [(set R16C:$rT, (sub i16ImmSExt10:$val, R16C:$rA))]>;

def SFvec : RRForm<0b00000010000, (outs VECREG:$rT),
                                  (ins VECREG:$rA, VECREG:$rB),
  "sf\t$rT, $rA, $rB", IntegerOp,
  [(set (v4i32 VECREG:$rT), (sub (v4i32 VECREG:$rB), (v4i32 VECREG:$rA)))]>;


def SFr32 : RRForm<0b00000010000, (outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
  "sf\t$rT, $rA, $rB", IntegerOp,
  [(set R32C:$rT, (sub R32C:$rB, R32C:$rA))]>;

def SFIvec:
    RI10Form<0b00110000, (outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
      "sfi\t$rT, $rA, $val", IntegerOp,
      [(set (v4i32 VECREG:$rT), (sub v4i32SExt10Imm:$val,
                                     (v4i32 VECREG:$rA)))]>;

def SFIr32 : RI10Form<0b00110000, (outs R32C:$rT),
                                  (ins R32C:$rA, s10imm_i32:$val),
  "sfi\t$rT, $rA, $val", IntegerOp,
  [(set R32C:$rT, (sub i32ImmSExt10:$val, R32C:$rA))]>;

// ADDX: only available in vector form, doesn't match a pattern.
class ADDXInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm<0b00000010110, OOL, IOL,
      "addx\t$rT, $rA, $rB",
      IntegerOp, pattern>;

class ADDXVecInst<ValueType vectype>:
    ADDXInst<(outs VECREG:$rT),
             (ins VECREG:$rA, VECREG:$rB, VECREG:$rCarry),
             [/* no pattern */]>,
    RegConstraint<"$rCarry = $rT">,
    NoEncode<"$rCarry">;

class ADDXRegInst<RegisterClass rclass>:
    ADDXInst<(outs rclass:$rT),
             (ins rclass:$rA, rclass:$rB, rclass:$rCarry),
             [/* no pattern */]>,
    RegConstraint<"$rCarry = $rT">,
    NoEncode<"$rCarry">;

multiclass AddExtended {
  def v2i64 : ADDXVecInst<v2i64>;
  def v4i32 : ADDXVecInst<v4i32>;
  def r64 : ADDXRegInst<R64C>;
  def r32 : ADDXRegInst<R32C>;
}

defm ADDX : AddExtended;

// CG: Generate carry for add
class CGInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm<0b01000011000, OOL, IOL,
      "cg\t$rT, $rA, $rB",
      IntegerOp, pattern>;

class CGVecInst<ValueType vectype>:
    CGInst<(outs VECREG:$rT),
           (ins VECREG:$rA, VECREG:$rB),
           [/* no pattern */]>;

class CGRegInst<RegisterClass rclass>:
    CGInst<(outs rclass:$rT),
           (ins rclass:$rA, rclass:$rB),
           [/* no pattern */]>;

multiclass CarryGenerate {
  def v2i64 : CGVecInst<v2i64>;
  def v4i32 : CGVecInst<v4i32>;
  def r64 : CGRegInst<R64C>;
  def r32 : CGRegInst<R32C>;
}

defm CG : CarryGenerate;

// SFX: Subract from, extended. This is used in conjunction with BG to subtract
// with carry (borrow, in this case)
class SFXInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm<0b10000010110, OOL, IOL,
      "sfx\t$rT, $rA, $rB",
      IntegerOp, pattern>;

class SFXVecInst<ValueType vectype>:
    SFXInst<(outs VECREG:$rT),
            (ins VECREG:$rA, VECREG:$rB, VECREG:$rCarry),
             [/* no pattern */]>,
    RegConstraint<"$rCarry = $rT">,
    NoEncode<"$rCarry">;

class SFXRegInst<RegisterClass rclass>:
    SFXInst<(outs rclass:$rT),
            (ins rclass:$rA, rclass:$rB, rclass:$rCarry),
             [/* no pattern */]>,
    RegConstraint<"$rCarry = $rT">,
    NoEncode<"$rCarry">;

multiclass SubtractExtended {
  def v2i64 : SFXVecInst<v2i64>;
  def v4i32 : SFXVecInst<v4i32>;
  def r64 : SFXRegInst<R64C>;
  def r32 : SFXRegInst<R32C>;
}

defm SFX : SubtractExtended;

// BG: only available in vector form, doesn't match a pattern.
class BGInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm<0b01000010000, OOL, IOL,
      "bg\t$rT, $rA, $rB",
      IntegerOp, pattern>;

class BGVecInst<ValueType vectype>:
    BGInst<(outs VECREG:$rT),
           (ins VECREG:$rA, VECREG:$rB),
           [/* no pattern */]>;

class BGRegInst<RegisterClass rclass>:
    BGInst<(outs rclass:$rT),
           (ins rclass:$rA, rclass:$rB),
           [/* no pattern */]>;

multiclass BorrowGenerate {
  def v4i32 : BGVecInst<v4i32>;
  def v2i64 : BGVecInst<v2i64>;
  def r64 : BGRegInst<R64C>;
  def r32 : BGRegInst<R32C>;
}

defm BG : BorrowGenerate;

// BGX: Borrow generate, extended.
def BGXvec:
    RRForm<0b11000010110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB,
                                VECREG:$rCarry),
      "bgx\t$rT, $rA, $rB", IntegerOp,
      []>,
    RegConstraint<"$rCarry = $rT">,
    NoEncode<"$rCarry">;

// Halfword multiply variants:
// N.B: These can be used to build up larger quantities (16x16 -> 32)

def MPYv8i16:
  RRForm<0b00100011110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
    "mpy\t$rT, $rA, $rB", IntegerMulDiv,
    [/* no pattern */]>;

def MPYr16:
  RRForm<0b00100011110, (outs R16C:$rT), (ins R16C:$rA, R16C:$rB),
    "mpy\t$rT, $rA, $rB", IntegerMulDiv,
    [(set R16C:$rT, (mul R16C:$rA, R16C:$rB))]>;

// Unsigned 16-bit multiply:

class MPYUInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm<0b00110011110, OOL, IOL,
      "mpyu\t$rT, $rA, $rB", IntegerMulDiv,
      pattern>;

def MPYUv4i32:
  MPYUInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
           [/* no pattern */]>;

def MPYUr16:
  MPYUInst<(outs R32C:$rT), (ins R16C:$rA, R16C:$rB),
           [(set R32C:$rT, (mul (zext R16C:$rA), (zext R16C:$rB)))]>;

def MPYUr32:
  MPYUInst<(outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
           [/* no pattern */]>;

// mpyi: multiply 16 x s10imm -> 32 result.

class MPYIInst<dag OOL, dag IOL, list<dag> pattern>:
  RI10Form<0b00101110, OOL, IOL,
    "mpyi\t$rT, $rA, $val", IntegerMulDiv,
    pattern>;

def MPYIvec:
  MPYIInst<(outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
           [(set (v8i16 VECREG:$rT),
                 (mul (v8i16 VECREG:$rA), v8i16SExt10Imm:$val))]>;

def MPYIr16:
  MPYIInst<(outs R16C:$rT), (ins R16C:$rA, s10imm:$val),
           [(set R16C:$rT, (mul R16C:$rA, i16ImmSExt10:$val))]>;

// mpyui: same issues as other multiplies, plus, this doesn't match a
// pattern... but may be used during target DAG selection or lowering

class MPYUIInst<dag OOL, dag IOL, list<dag> pattern>:
  RI10Form<0b10101110, OOL, IOL,
           "mpyui\t$rT, $rA, $val", IntegerMulDiv,
           pattern>;
    
def MPYUIvec:
  MPYUIInst<(outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
            []>;

def MPYUIr16:
  MPYUIInst<(outs R16C:$rT), (ins R16C:$rA, s10imm:$val),
            []>;

// mpya: 16 x 16 + 16 -> 32 bit result
class MPYAInst<dag OOL, dag IOL, list<dag> pattern>:
  RRRForm<0b0011, OOL, IOL,
          "mpya\t$rT, $rA, $rB, $rC", IntegerMulDiv,
          pattern>;
          
def MPYAv4i32:
  MPYAInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB, VECREG:$rC),
           [(set (v4i32 VECREG:$rT),
                 (add (v4i32 (bitconvert (mul (v8i16 VECREG:$rA),
                                              (v8i16 VECREG:$rB)))),
                      (v4i32 VECREG:$rC)))]>;

def MPYAr32:
  MPYAInst<(outs R32C:$rT), (ins R16C:$rA, R16C:$rB, R32C:$rC),
           [(set R32C:$rT, (add (sext (mul R16C:$rA, R16C:$rB)),
                                R32C:$rC))]>;
                                
def MPYAr32_sext:
  MPYAInst<(outs R32C:$rT), (ins R16C:$rA, R16C:$rB, R32C:$rC),
           [(set R32C:$rT, (add (mul (sext R16C:$rA), (sext R16C:$rB)),
                                R32C:$rC))]>;

def MPYAr32_sextinreg:
  MPYAInst<(outs R32C:$rT), (ins R32C:$rA, R32C:$rB, R32C:$rC),
           [(set R32C:$rT, (add (mul (sext_inreg R32C:$rA, i16),
                                     (sext_inreg R32C:$rB, i16)),
                                R32C:$rC))]>;

// mpyh: multiply high, used to synthesize 32-bit multiplies
class MPYHInst<dag OOL, dag IOL, list<dag> pattern>:
  RRForm<0b10100011110, OOL, IOL,
         "mpyh\t$rT, $rA, $rB", IntegerMulDiv,
         pattern>;
         
def MPYHv4i32:
    MPYHInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
             [/* no pattern */]>;

def MPYHr32:
    MPYHInst<(outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
             [/* no pattern */]>;

// mpys: multiply high and shift right (returns the top half of
// a 16-bit multiply, sign extended to 32 bits.)

class MPYSInst<dag OOL, dag IOL>:
    RRForm<0b11100011110, OOL, IOL, 
      "mpys\t$rT, $rA, $rB", IntegerMulDiv,
      [/* no pattern */]>;

def MPYSv4i32:
    MPYSInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB)>;
    
def MPYSr16:
    MPYSInst<(outs R32C:$rT), (ins R16C:$rA, R16C:$rB)>;

// mpyhh: multiply high-high (returns the 32-bit result from multiplying
// the top 16 bits of the $rA, $rB)

class MPYHHInst<dag OOL, dag IOL>:
  RRForm<0b01100011110, OOL, IOL,
        "mpyhh\t$rT, $rA, $rB", IntegerMulDiv,
        [/* no pattern */]>;
        
def MPYHHv8i16:
    MPYHHInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB)>;

def MPYHHr32:
    MPYHHInst<(outs R32C:$rT), (ins R32C:$rA, R32C:$rB)>;

// mpyhha: Multiply high-high, add to $rT:

class MPYHHAInst<dag OOL, dag IOL>:
    RRForm<0b01100010110, OOL, IOL,
      "mpyhha\t$rT, $rA, $rB", IntegerMulDiv,
      [/* no pattern */]>;

def MPYHHAvec:
    MPYHHAInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB)>;
    
def MPYHHAr32:
    MPYHHAInst<(outs R32C:$rT), (ins R32C:$rA, R32C:$rB)>;

// mpyhhu: Multiply high-high, unsigned, e.g.:
//
// +-------+-------+   +-------+-------+   +---------+
// |  a0   .  a1   | x |  b0   .  b1   | = | a0 x b0 |
// +-------+-------+   +-------+-------+   +---------+
//
// where a0, b0 are the upper 16 bits of the 32-bit word

class MPYHHUInst<dag OOL, dag IOL>:
    RRForm<0b01110011110, OOL, IOL,
      "mpyhhu\t$rT, $rA, $rB", IntegerMulDiv,
      [/* no pattern */]>;

def MPYHHUv4i32:
    MPYHHUInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB)>;
    
def MPYHHUr32:
    MPYHHUInst<(outs R32C:$rT), (ins R32C:$rA, R32C:$rB)>;

// mpyhhau: Multiply high-high, unsigned

class MPYHHAUInst<dag OOL, dag IOL>:
    RRForm<0b01110010110, OOL, IOL,
      "mpyhhau\t$rT, $rA, $rB", IntegerMulDiv,
      [/* no pattern */]>;

def MPYHHAUvec:
    MPYHHAUInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB)>;
    
def MPYHHAUr32:
    MPYHHAUInst<(outs R32C:$rT), (ins R32C:$rA, R32C:$rB)>;

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// clz: Count leading zeroes
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
class CLZInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm_1<0b10100101010, OOL, IOL, "clz\t$rT, $rA",
             IntegerOp, pattern>;

class CLZRegInst<RegisterClass rclass>:
    CLZInst<(outs rclass:$rT), (ins rclass:$rA),
            [(set rclass:$rT, (ctlz rclass:$rA))]>;

class CLZVecInst<ValueType vectype>:
    CLZInst<(outs VECREG:$rT), (ins VECREG:$rA),
            [(set (vectype VECREG:$rT), (ctlz (vectype VECREG:$rA)))]>;

multiclass CountLeadingZeroes {
  def v4i32 : CLZVecInst<v4i32>;
  def r32   : CLZRegInst<R32C>;
}

defm CLZ : CountLeadingZeroes;

// cntb: Count ones in bytes (aka "population count")
//
// NOTE: This instruction is really a vector instruction, but the custom
// lowering code uses it in unorthodox ways to support CTPOP for other
// data types!

def CNTBv16i8:
    RRForm_1<0b00101101010, (outs VECREG:$rT), (ins VECREG:$rA),
      "cntb\t$rT, $rA", IntegerOp,
      [(set (v16i8 VECREG:$rT), (SPUcntb (v16i8 VECREG:$rA)))]>;

def CNTBv8i16 :
    RRForm_1<0b00101101010, (outs VECREG:$rT), (ins VECREG:$rA),
      "cntb\t$rT, $rA", IntegerOp,
      [(set (v8i16 VECREG:$rT), (SPUcntb (v8i16 VECREG:$rA)))]>;

def CNTBv4i32 :
    RRForm_1<0b00101101010, (outs VECREG:$rT), (ins VECREG:$rA),
      "cntb\t$rT, $rA", IntegerOp,
      [(set (v4i32 VECREG:$rT), (SPUcntb (v4i32 VECREG:$rA)))]>;

// gbb: Gather the low order bits from each byte in $rA into a single 16-bit
// quantity stored into $rT's slot 0, upper 16 bits are zeroed, as are
// slots 1-3.
//
// Note: This instruction "pairs" with the fsmb instruction for all of the
// various types defined here.
//
// Note 2: The "VecInst" and "RegInst" forms refer to the result being either
// a vector or register.

class GBBInst<dag OOL, dag IOL, list<dag> pattern>:
  RRForm_1<0b01001101100, OOL, IOL, "gbb\t$rT, $rA", GatherOp, pattern>;

class GBBRegInst<RegisterClass rclass, ValueType vectype>:
  GBBInst<(outs rclass:$rT), (ins VECREG:$rA),
          [/* no pattern */]>;

class GBBVecInst<ValueType vectype>:
  GBBInst<(outs VECREG:$rT), (ins VECREG:$rA),
          [/* no pattern */]>;

multiclass GatherBitsFromBytes {
  def v16i8_r32: GBBRegInst<R32C, v16i8>;
  def v16i8_r16: GBBRegInst<R16C, v16i8>;
  def v16i8:     GBBVecInst<v16i8>;
}

defm GBB: GatherBitsFromBytes;

// gbh: Gather all low order bits from each halfword in $rA into a single
// 8-bit quantity stored in $rT's slot 0, with the upper bits of $rT set to 0
// and slots 1-3 also set to 0.
//
// See notes for GBBInst, above.

class GBHInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm_1<0b10001101100, OOL, IOL, "gbh\t$rT, $rA", GatherOp,
             pattern>;

class GBHRegInst<RegisterClass rclass, ValueType vectype>:
    GBHInst<(outs rclass:$rT), (ins VECREG:$rA),
            [/* no pattern */]>;

class GBHVecInst<ValueType vectype>:
    GBHInst<(outs VECREG:$rT), (ins VECREG:$rA),
            [/* no pattern */]>;

multiclass GatherBitsHalfword {
  def v8i16_r32: GBHRegInst<R32C, v8i16>;
  def v8i16_r16: GBHRegInst<R16C, v8i16>;
  def v8i16:     GBHVecInst<v8i16>;
}

defm GBH: GatherBitsHalfword;

// gb: Gather all low order bits from each word in $rA into a single
// 4-bit quantity stored in $rT's slot 0, upper bits in $rT set to 0,
// as well as slots 1-3.
//
// See notes for gbb, above.

class GBInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm_1<0b00001101100, OOL, IOL, "gb\t$rT, $rA", GatherOp,
             pattern>;

class GBRegInst<RegisterClass rclass, ValueType vectype>:
    GBInst<(outs rclass:$rT), (ins VECREG:$rA),
           [/* no pattern */]>;

class GBVecInst<ValueType vectype>:
    GBInst<(outs VECREG:$rT), (ins VECREG:$rA),
           [/* no pattern */]>;

multiclass GatherBitsWord {
  def v4i32_r32: GBRegInst<R32C, v4i32>;
  def v4i32_r16: GBRegInst<R16C, v4i32>;
  def v4i32:     GBVecInst<v4i32>;
}

defm GB: GatherBitsWord;

// avgb: average bytes
def AVGB:
    RRForm<0b11001011000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "avgb\t$rT, $rA, $rB", ByteOp,
      []>;

// absdb: absolute difference of bytes
def ABSDB:
    RRForm<0b11001010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "absdb\t$rT, $rA, $rB", ByteOp,
      []>;

// sumb: sum bytes into halfwords
def SUMB:
    RRForm<0b11001010010, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "sumb\t$rT, $rA, $rB", ByteOp,
      []>;

// Sign extension operations:
class XSBHInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm_1<0b01101101010, OOL, IOL,
      "xsbh\t$rDst, $rSrc",
      IntegerOp, pattern>;

class XSBHInRegInst<RegisterClass rclass, list<dag> pattern>:
    XSBHInst<(outs rclass:$rDst), (ins rclass:$rSrc),
             pattern>;

multiclass ExtendByteHalfword {
  def v16i8:     XSBHInst<(outs VECREG:$rDst), (ins VECREG:$rSrc),
                          [
                  /*(set (v8i16 VECREG:$rDst), (sext (v8i16 VECREG:$rSrc)))*/]>;
  def r8:        XSBHInst<(outs R16C:$rDst), (ins R8C:$rSrc),
                          [(set R16C:$rDst, (sext R8C:$rSrc))]>;
  def r16:       XSBHInRegInst<R16C,
                               [(set R16C:$rDst, (sext_inreg R16C:$rSrc, i8))]>;

  // 32-bit form for XSBH: used to sign extend 8-bit quantities to 16-bit
  // quantities to 32-bit quantities via a 32-bit register (see the sext 8->32
  // pattern below). Intentionally doesn't match a pattern because we want the
  // sext 8->32 pattern to do the work for us, namely because we need the extra
  // XSHWr32.
  def r32:   XSBHInRegInst<R32C, [/* no pattern */]>;
  
  // Same as the 32-bit version, but for i64
  def r64:   XSBHInRegInst<R64C, [/* no pattern */]>;
}

defm XSBH : ExtendByteHalfword;

// Sign extend halfwords to words:

class XSHWInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm_1<0b01101101010, OOL, IOL, "xshw\t$rDest, $rSrc",
            IntegerOp, pattern>;

class XSHWVecInst<ValueType in_vectype, ValueType out_vectype>:
    XSHWInst<(outs VECREG:$rDest), (ins VECREG:$rSrc),
             [(set (out_vectype VECREG:$rDest),
                   (sext (in_vectype VECREG:$rSrc)))]>;

class XSHWInRegInst<RegisterClass rclass, list<dag> pattern>:
    XSHWInst<(outs rclass:$rDest), (ins rclass:$rSrc),
             pattern>;
             
class XSHWRegInst<RegisterClass rclass>:
    XSHWInst<(outs rclass:$rDest), (ins R16C:$rSrc),
             [(set rclass:$rDest, (sext R16C:$rSrc))]>;

multiclass ExtendHalfwordWord {
  def v4i32: XSHWVecInst<v8i16, v4i32>;

  def r16:   XSHWRegInst<R32C>;

  def r32:   XSHWInRegInst<R32C,
                          [(set R32C:$rDest, (sext_inreg R32C:$rSrc, i16))]>;
  def r64:   XSHWInRegInst<R64C, [/* no pattern */]>;
}

defm XSHW : ExtendHalfwordWord;

// Sign-extend words to doublewords (32->64 bits)

class XSWDInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm_1<0b01100101010, OOL, IOL, "xswd\t$rDst, $rSrc",
              IntegerOp, pattern>;
      
class XSWDVecInst<ValueType in_vectype, ValueType out_vectype>:
    XSWDInst<(outs VECREG:$rDst), (ins VECREG:$rSrc),
             [/*(set (out_vectype VECREG:$rDst),
                   (sext (out_vectype VECREG:$rSrc)))*/]>;
      
class XSWDRegInst<RegisterClass in_rclass, RegisterClass out_rclass>:
    XSWDInst<(outs out_rclass:$rDst), (ins in_rclass:$rSrc),
             [(set out_rclass:$rDst, (sext in_rclass:$rSrc))]>;
             
multiclass ExtendWordToDoubleWord {
  def v2i64: XSWDVecInst<v4i32, v2i64>;
  def r64:   XSWDRegInst<R32C, R64C>;
  
  def r64_inreg: XSWDInst<(outs R64C:$rDst), (ins R64C:$rSrc),
                          [(set R64C:$rDst, (sext_inreg R64C:$rSrc, i32))]>;
}

defm XSWD : ExtendWordToDoubleWord;

// AND operations

class ANDInst<dag OOL, dag IOL, list<dag> pattern> :
    RRForm<0b10000011000, OOL, IOL, "and\t$rT, $rA, $rB",
           IntegerOp, pattern>;

class ANDVecInst<ValueType vectype>:
    ANDInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
             [(set (vectype VECREG:$rT), (and (vectype VECREG:$rA),
                                              (vectype VECREG:$rB)))]>;

class ANDRegInst<RegisterClass rclass>:
    ANDInst<(outs rclass:$rT), (ins rclass:$rA, rclass:$rB),
             [(set rclass:$rT, (and rclass:$rA, rclass:$rB))]>;

multiclass BitwiseAnd
{
  def v16i8: ANDVecInst<v16i8>;
  def v8i16: ANDVecInst<v8i16>;
  def v4i32: ANDVecInst<v4i32>;
  def v2i64: ANDVecInst<v2i64>;

  def r128:  ANDRegInst<GPRC>;
  def r64:   ANDRegInst<R64C>;
  def r32:   ANDRegInst<R32C>;
  def r16:   ANDRegInst<R16C>;
  def r8:    ANDRegInst<R8C>;

  //===---------------------------------------------
  // Special instructions to perform the fabs instruction
  def fabs32: ANDInst<(outs R32FP:$rT), (ins R32FP:$rA, R32C:$rB),
                      [/* Intentionally does not match a pattern */]>;

  def fabs64: ANDInst<(outs R64FP:$rT), (ins R64FP:$rA, R64C:$rB),
                      [/* Intentionally does not match a pattern */]>;

  def fabsvec: ANDInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
                       [/* Intentionally does not match a pattern */]>;

  //===---------------------------------------------

  // Hacked form of AND to zero-extend 16-bit quantities to 32-bit
  // quantities -- see 16->32 zext pattern.
  //
  // This pattern is somewhat artificial, since it might match some
  // compiler generated pattern but it is unlikely to do so.

  def i16i32: ANDInst<(outs R32C:$rT), (ins R16C:$rA, R32C:$rB),
                      [(set R32C:$rT, (and (zext R16C:$rA), R32C:$rB))]>;
}

defm AND : BitwiseAnd;


def vnot_cell_conv : PatFrag<(ops node:$in),
                             (xor node:$in, (bitconvert (v4i32 immAllOnesV)))>;

// N.B.: vnot_cell_conv is one of those special target selection pattern
// fragments,
// in which we expect there to be a bit_convert on the constant. Bear in mind
// that llvm translates "not <reg>" to "xor <reg>, -1" (or in this case, a
// constant -1 vector.)

class ANDCInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm<0b10000011010, OOL, IOL, "andc\t$rT, $rA, $rB",
           IntegerOp, pattern>;

class ANDCVecInst<ValueType vectype, PatFrag vnot_frag = vnot>:
    ANDCInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
             [(set (vectype VECREG:$rT),
                   (and (vectype VECREG:$rA),
                        (vnot_frag (vectype VECREG:$rB))))]>;

class ANDCRegInst<RegisterClass rclass>:
    ANDCInst<(outs rclass:$rT), (ins rclass:$rA, rclass:$rB),
             [(set rclass:$rT, (and rclass:$rA, (not rclass:$rB)))]>;

multiclass AndComplement
{
  def v16i8: ANDCVecInst<v16i8>;
  def v8i16: ANDCVecInst<v8i16>;
  def v4i32: ANDCVecInst<v4i32>;
  def v2i64: ANDCVecInst<v2i64>;

  def r128: ANDCRegInst<GPRC>;
  def r64:  ANDCRegInst<R64C>;
  def r32:  ANDCRegInst<R32C>;
  def r16:  ANDCRegInst<R16C>;
  def r8:   ANDCRegInst<R8C>;

  // Sometimes, the xor pattern has a bitcast constant:
  def v16i8_conv: ANDCVecInst<v16i8, vnot_cell_conv>;
}

defm ANDC : AndComplement;

class ANDBIInst<dag OOL, dag IOL, list<dag> pattern>:
    RI10Form<0b01101000, OOL, IOL, "andbi\t$rT, $rA, $val",
             ByteOp, pattern>;

multiclass AndByteImm
{
  def v16i8: ANDBIInst<(outs VECREG:$rT), (ins VECREG:$rA, u10imm:$val),
                       [(set (v16i8 VECREG:$rT),
                             (and (v16i8 VECREG:$rA),
                                  (v16i8 v16i8U8Imm:$val)))]>;

  def r8: ANDBIInst<(outs R8C:$rT), (ins R8C:$rA, u10imm_i8:$val),
                    [(set R8C:$rT, (and R8C:$rA, immU8:$val))]>;
}

defm ANDBI : AndByteImm;

class ANDHIInst<dag OOL, dag IOL, list<dag> pattern> :
    RI10Form<0b10101000, OOL, IOL, "andhi\t$rT, $rA, $val",
             ByteOp, pattern>;

multiclass AndHalfwordImm
{
  def v8i16: ANDHIInst<(outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
                       [(set (v8i16 VECREG:$rT),
                             (and (v8i16 VECREG:$rA), v8i16SExt10Imm:$val))]>;

  def r16: ANDHIInst<(outs R16C:$rT), (ins R16C:$rA, u10imm:$val),
                     [(set R16C:$rT, (and R16C:$rA, i16ImmUns10:$val))]>;

  // Zero-extend i8 to i16:
  def i8i16: ANDHIInst<(outs R16C:$rT), (ins R8C:$rA, u10imm:$val),
                      [(set R16C:$rT, (and (zext R8C:$rA), i16ImmUns10:$val))]>;
}

defm ANDHI : AndHalfwordImm;

class ANDIInst<dag OOL, dag IOL, list<dag> pattern> :
    RI10Form<0b00101000, OOL, IOL, "andi\t$rT, $rA, $val",
             IntegerOp, pattern>;

multiclass AndWordImm
{
  def v4i32: ANDIInst<(outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
                      [(set (v4i32 VECREG:$rT),
                            (and (v4i32 VECREG:$rA), v4i32SExt10Imm:$val))]>;

  def r32: ANDIInst<(outs R32C:$rT), (ins R32C:$rA, s10imm_i32:$val),
                    [(set R32C:$rT, (and R32C:$rA, i32ImmSExt10:$val))]>;

  // Hacked form of ANDI to zero-extend i8 quantities to i32. See the zext 8->32
  // pattern below.
  def i8i32: ANDIInst<(outs R32C:$rT), (ins R8C:$rA, s10imm_i32:$val),
                      [(set R32C:$rT,
                            (and (zext R8C:$rA), i32ImmSExt10:$val))]>;

  // Hacked form of ANDI to zero-extend i16 quantities to i32. See the
  // zext 16->32 pattern below.
  //
  // Note that this pattern is somewhat artificial, since it might match
  // something the compiler generates but is unlikely to occur in practice.
  def i16i32: ANDIInst<(outs R32C:$rT), (ins R16C:$rA, s10imm_i32:$val),
                       [(set R32C:$rT,
                             (and (zext R16C:$rA), i32ImmSExt10:$val))]>;
}

defm ANDI : AndWordImm;

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// Bitwise OR group:
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

// Bitwise "or" (N.B.: These are also register-register copy instructions...)
class ORInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm<0b10000010000, OOL, IOL, "or\t$rT, $rA, $rB",
           IntegerOp, pattern>;

class ORVecInst<ValueType vectype>:
    ORInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
           [(set (vectype VECREG:$rT), (or (vectype VECREG:$rA),
                                           (vectype VECREG:$rB)))]>;

class ORRegInst<RegisterClass rclass>:
    ORInst<(outs rclass:$rT), (ins rclass:$rA, rclass:$rB),
           [(set rclass:$rT, (or rclass:$rA, rclass:$rB))]>;


multiclass BitwiseOr
{
  def v16i8: ORVecInst<v16i8>;
  def v8i16: ORVecInst<v8i16>;
  def v4i32: ORVecInst<v4i32>;
  def v2i64: ORVecInst<v2i64>;

  def v4f32: ORInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
                    [(set (v4f32 VECREG:$rT),
                          (v4f32 (bitconvert (or (v4i32 VECREG:$rA),
                                                 (v4i32 VECREG:$rB)))))]>;

  def v2f64: ORInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
                    [(set (v2f64 VECREG:$rT),
                          (v2f64 (bitconvert (or (v2i64 VECREG:$rA),
                                                 (v2i64 VECREG:$rB)))))]>;

  def r128: ORRegInst<GPRC>;
  def r64:  ORRegInst<R64C>;
  def r32:  ORRegInst<R32C>;
  def r16:  ORRegInst<R16C>;
  def r8:   ORRegInst<R8C>;

  // OR instructions used to copy f32 and f64 registers.
  def f32: ORInst<(outs R32FP:$rT), (ins R32FP:$rA, R32FP:$rB),
                  [/* no pattern */]>;

  def f64: ORInst<(outs R64FP:$rT), (ins R64FP:$rA, R64FP:$rB),
                  [/* no pattern */]>;
}

defm OR : BitwiseOr;

//===----------------------------------------------------------------------===//
// SPU::PREFSLOT2VEC and VEC2PREFSLOT re-interpretations of registers
//===----------------------------------------------------------------------===//
def : Pat<(v16i8 (SPUprefslot2vec R8C:$rA)),
          (COPY_TO_REGCLASS R8C:$rA, VECREG)>;

def : Pat<(v8i16 (SPUprefslot2vec R16C:$rA)),
          (COPY_TO_REGCLASS R16C:$rA, VECREG)>;

def : Pat<(v4i32 (SPUprefslot2vec R32C:$rA)),
          (COPY_TO_REGCLASS R32C:$rA, VECREG)>;

def : Pat<(v2i64 (SPUprefslot2vec R64C:$rA)),
          (COPY_TO_REGCLASS R64C:$rA, VECREG)>;

def : Pat<(v4f32 (SPUprefslot2vec R32FP:$rA)),
          (COPY_TO_REGCLASS R32FP:$rA, VECREG)>;

def : Pat<(v2f64 (SPUprefslot2vec R64FP:$rA)),
          (COPY_TO_REGCLASS R64FP:$rA, VECREG)>;
 
def : Pat<(i8 (SPUvec2prefslot (v16i8 VECREG:$rA))),
          (COPY_TO_REGCLASS (v16i8 VECREG:$rA), R8C)>;

def : Pat<(i16 (SPUvec2prefslot (v8i16 VECREG:$rA))),
          (COPY_TO_REGCLASS (v8i16 VECREG:$rA), R16C)>;

def : Pat<(i32 (SPUvec2prefslot (v4i32 VECREG:$rA))),
          (COPY_TO_REGCLASS (v4i32 VECREG:$rA), R32C)>;

def : Pat<(i64 (SPUvec2prefslot (v2i64 VECREG:$rA))),
          (COPY_TO_REGCLASS (v2i64 VECREG:$rA), R64C)>;

def : Pat<(f32 (SPUvec2prefslot (v4f32 VECREG:$rA))),
          (COPY_TO_REGCLASS (v4f32 VECREG:$rA), R32FP)>;

def : Pat<(f64 (SPUvec2prefslot (v2f64 VECREG:$rA))),
          (COPY_TO_REGCLASS (v2f64 VECREG:$rA), R64FP)>;

// Load Register: This is an assembler alias for a bitwise OR of a register
// against itself. It's here because it brings some clarity to assembly
// language output.

let hasCtrlDep = 1 in {
    class LRInst<dag OOL, dag IOL>
              : SPUInstr<OOL, IOL, "lr\t$rT, $rA", IntegerOp> {
      bits<7> RA;
      bits<7> RT;

      let Pattern = [/*no pattern*/];

      let Inst{0-10} = 0b10000010000;   /* It's an OR operation */
      let Inst{11-17} = RA;
      let Inst{18-24} = RA;
      let Inst{25-31} = RT;
    }

    class LRVecInst<ValueType vectype>:
        LRInst<(outs VECREG:$rT), (ins VECREG:$rA)>;

    class LRRegInst<RegisterClass rclass>:
        LRInst<(outs rclass:$rT), (ins rclass:$rA)>;

    multiclass LoadRegister {
      def v2i64: LRVecInst<v2i64>;
      def v2f64: LRVecInst<v2f64>;
      def v4i32: LRVecInst<v4i32>;
      def v4f32: LRVecInst<v4f32>;
      def v8i16: LRVecInst<v8i16>;
      def v16i8: LRVecInst<v16i8>;

      def r128:  LRRegInst<GPRC>;
      def r64:   LRRegInst<R64C>;
      def f64:   LRRegInst<R64FP>;
      def r32:   LRRegInst<R32C>;
      def f32:   LRRegInst<R32FP>;
      def r16:   LRRegInst<R16C>;
      def r8:    LRRegInst<R8C>;
    }

    defm LR: LoadRegister;
}

// ORC: Bitwise "or" with complement (c = a | ~b)

class ORCInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm<0b10010010000, OOL, IOL, "orc\t$rT, $rA, $rB",
           IntegerOp, pattern>;

class ORCVecInst<ValueType vectype>:
    ORCInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
            [(set (vectype VECREG:$rT), (or (vectype VECREG:$rA),
                                            (vnot (vectype VECREG:$rB))))]>;

class ORCRegInst<RegisterClass rclass>:
  ORCInst<(outs rclass:$rT), (ins rclass:$rA, rclass:$rB),
          [(set rclass:$rT, (or rclass:$rA, (not rclass:$rB)))]>;

multiclass BitwiseOrComplement
{
  def v16i8: ORCVecInst<v16i8>;
  def v8i16: ORCVecInst<v8i16>;
  def v4i32: ORCVecInst<v4i32>;
  def v2i64: ORCVecInst<v2i64>;

  def r128:  ORCRegInst<GPRC>;
  def r64:   ORCRegInst<R64C>;
  def r32:   ORCRegInst<R32C>;
  def r16:   ORCRegInst<R16C>;
  def r8:    ORCRegInst<R8C>;
}

defm ORC : BitwiseOrComplement;

// OR byte immediate
class ORBIInst<dag OOL, dag IOL, list<dag> pattern>:
    RI10Form<0b01100000, OOL, IOL, "orbi\t$rT, $rA, $val",
             IntegerOp, pattern>;

class ORBIVecInst<ValueType vectype, PatLeaf immpred>:
    ORBIInst<(outs VECREG:$rT), (ins VECREG:$rA, u10imm:$val),
             [(set (v16i8 VECREG:$rT), (or (vectype VECREG:$rA),
                                           (vectype immpred:$val)))]>;

multiclass BitwiseOrByteImm
{
  def v16i8: ORBIVecInst<v16i8, v16i8U8Imm>;

  def r8: ORBIInst<(outs R8C:$rT), (ins R8C:$rA, u10imm_i8:$val),
                   [(set R8C:$rT, (or R8C:$rA, immU8:$val))]>;
}

defm ORBI : BitwiseOrByteImm;

// OR halfword immediate
class ORHIInst<dag OOL, dag IOL, list<dag> pattern>:
    RI10Form<0b10100000, OOL, IOL, "orhi\t$rT, $rA, $val",
             IntegerOp, pattern>;

class ORHIVecInst<ValueType vectype, PatLeaf immpred>:
    ORHIInst<(outs VECREG:$rT), (ins VECREG:$rA, u10imm:$val),
              [(set (vectype VECREG:$rT), (or (vectype VECREG:$rA),
                                              immpred:$val))]>;

multiclass BitwiseOrHalfwordImm
{
  def v8i16: ORHIVecInst<v8i16, v8i16Uns10Imm>;

  def r16: ORHIInst<(outs R16C:$rT), (ins R16C:$rA, u10imm:$val),
                    [(set R16C:$rT, (or R16C:$rA, i16ImmUns10:$val))]>;

  // Specialized ORHI form used to promote 8-bit registers to 16-bit
  def i8i16: ORHIInst<(outs R16C:$rT), (ins R8C:$rA, s10imm:$val),
                      [(set R16C:$rT, (or (anyext R8C:$rA),
                                          i16ImmSExt10:$val))]>;
}

defm ORHI : BitwiseOrHalfwordImm;

class ORIInst<dag OOL, dag IOL, list<dag> pattern>:
    RI10Form<0b00100000, OOL, IOL, "ori\t$rT, $rA, $val",
             IntegerOp, pattern>;

class ORIVecInst<ValueType vectype, PatLeaf immpred>:
    ORIInst<(outs VECREG:$rT), (ins VECREG:$rA, u10imm:$val),
            [(set (vectype VECREG:$rT), (or (vectype VECREG:$rA),
                                            immpred:$val))]>;

// Bitwise "or" with immediate
multiclass BitwiseOrImm
{
  def v4i32: ORIVecInst<v4i32, v4i32Uns10Imm>;

  def r32: ORIInst<(outs R32C:$rT), (ins R32C:$rA, s10imm_i32:$val),
                   [(set R32C:$rT, (or R32C:$rA, i32ImmSExt10:$val))]>;

  // i16i32: hacked version of the ori instruction to extend 16-bit quantities
  // to 32-bit quantities. used exclusively to match "anyext" conversions (vide
  // infra "anyext 16->32" pattern.)
  def i16i32: ORIInst<(outs R32C:$rT), (ins R16C:$rA, s10imm_i32:$val),
                      [(set R32C:$rT, (or (anyext R16C:$rA),
                                          i32ImmSExt10:$val))]>;

  // i8i32: Hacked version of the ORI instruction to extend 16-bit quantities
  // to 32-bit quantities. Used exclusively to match "anyext" conversions (vide
  // infra "anyext 16->32" pattern.)
  def i8i32: ORIInst<(outs R32C:$rT), (ins R8C:$rA, s10imm_i32:$val),
                     [(set R32C:$rT, (or (anyext R8C:$rA),
                                         i32ImmSExt10:$val))]>;
}

defm ORI : BitwiseOrImm;

// ORX: "or" across the vector: or's $rA's word slots leaving the result in
// $rT[0], slots 1-3 are zeroed.
//
// FIXME: Needs to match an intrinsic pattern.
def ORXv4i32:
    RRForm<0b10010010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "orx\t$rT, $rA, $rB", IntegerOp,
      []>;

// XOR:

class XORInst<dag OOL, dag IOL, list<dag> pattern> :
    RRForm<0b10010010000, OOL, IOL, "xor\t$rT, $rA, $rB",
           IntegerOp, pattern>;

class XORVecInst<ValueType vectype>:
    XORInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
             [(set (vectype VECREG:$rT), (xor (vectype VECREG:$rA),
                                              (vectype VECREG:$rB)))]>;

class XORRegInst<RegisterClass rclass>:
    XORInst<(outs rclass:$rT), (ins rclass:$rA, rclass:$rB),
             [(set rclass:$rT, (xor rclass:$rA, rclass:$rB))]>;

multiclass BitwiseExclusiveOr
{
  def v16i8: XORVecInst<v16i8>;
  def v8i16: XORVecInst<v8i16>;
  def v4i32: XORVecInst<v4i32>;
  def v2i64: XORVecInst<v2i64>;

  def r128:  XORRegInst<GPRC>;
  def r64:   XORRegInst<R64C>;
  def r32:   XORRegInst<R32C>;
  def r16:   XORRegInst<R16C>;
  def r8:    XORRegInst<R8C>;

  // XOR instructions used to negate f32 and f64 quantities.

  def fneg32: XORInst<(outs R32FP:$rT), (ins R32FP:$rA, R32C:$rB),
                     [/* no pattern */]>;

  def fneg64: XORInst<(outs R64FP:$rT), (ins R64FP:$rA, R64C:$rB),
                     [/* no pattern */]>;

  def fnegvec: XORInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
                      [/* no pattern, see fneg{32,64} */]>;
}

defm XOR : BitwiseExclusiveOr;

//==----------------------------------------------------------

class XORBIInst<dag OOL, dag IOL, list<dag> pattern>:
    RI10Form<0b01100000, OOL, IOL, "xorbi\t$rT, $rA, $val",
             IntegerOp, pattern>;

multiclass XorByteImm
{
  def v16i8:
    XORBIInst<(outs VECREG:$rT), (ins VECREG:$rA, u10imm:$val),
              [(set (v16i8 VECREG:$rT), (xor (v16i8 VECREG:$rA), v16i8U8Imm:$val))]>;

  def r8:
    XORBIInst<(outs R8C:$rT), (ins R8C:$rA, u10imm_i8:$val),
              [(set R8C:$rT, (xor R8C:$rA, immU8:$val))]>;
}

defm XORBI : XorByteImm;

def XORHIv8i16:
    RI10Form<0b10100000, (outs VECREG:$rT), (ins VECREG:$rA, u10imm:$val),
      "xorhi\t$rT, $rA, $val", IntegerOp,
      [(set (v8i16 VECREG:$rT), (xor (v8i16 VECREG:$rA),
                                      v8i16SExt10Imm:$val))]>;

def XORHIr16:
    RI10Form<0b10100000, (outs R16C:$rT), (ins R16C:$rA, s10imm:$val),
      "xorhi\t$rT, $rA, $val", IntegerOp,
      [(set R16C:$rT, (xor R16C:$rA, i16ImmSExt10:$val))]>;

def XORIv4i32:
    RI10Form<0b00100000, (outs VECREG:$rT), (ins VECREG:$rA, s10imm_i32:$val),
      "xori\t$rT, $rA, $val", IntegerOp,
      [(set (v4i32 VECREG:$rT), (xor (v4i32 VECREG:$rA),
                                     v4i32SExt10Imm:$val))]>;

def XORIr32:
    RI10Form<0b00100000, (outs R32C:$rT), (ins R32C:$rA, s10imm_i32:$val),
      "xori\t$rT, $rA, $val", IntegerOp,
      [(set R32C:$rT, (xor R32C:$rA, i32ImmSExt10:$val))]>;

// NAND:

class NANDInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm<0b10010011000, OOL, IOL, "nand\t$rT, $rA, $rB",
           IntegerOp, pattern>;

class NANDVecInst<ValueType vectype>:
    NANDInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
             [(set (vectype VECREG:$rT), (vnot (and (vectype VECREG:$rA),
                                                    (vectype VECREG:$rB))))]>;
class NANDRegInst<RegisterClass rclass>:
    NANDInst<(outs rclass:$rT), (ins rclass:$rA, rclass:$rB),
             [(set rclass:$rT, (not (and rclass:$rA, rclass:$rB)))]>;

multiclass BitwiseNand
{
  def v16i8: NANDVecInst<v16i8>;
  def v8i16: NANDVecInst<v8i16>;
  def v4i32: NANDVecInst<v4i32>;
  def v2i64: NANDVecInst<v2i64>;

  def r128:  NANDRegInst<GPRC>;
  def r64:   NANDRegInst<R64C>;
  def r32:   NANDRegInst<R32C>;
  def r16:   NANDRegInst<R16C>;
  def r8:    NANDRegInst<R8C>;
}

defm NAND : BitwiseNand;

// NOR:

class NORInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm<0b10010010000, OOL, IOL, "nor\t$rT, $rA, $rB",
           IntegerOp, pattern>;

class NORVecInst<ValueType vectype>:
    NORInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
            [(set (vectype VECREG:$rT), (vnot (or (vectype VECREG:$rA),
                                                  (vectype VECREG:$rB))))]>;
class NORRegInst<RegisterClass rclass>:
    NORInst<(outs rclass:$rT), (ins rclass:$rA, rclass:$rB),
            [(set rclass:$rT, (not (or rclass:$rA, rclass:$rB)))]>;

multiclass BitwiseNor
{
  def v16i8: NORVecInst<v16i8>;
  def v8i16: NORVecInst<v8i16>;
  def v4i32: NORVecInst<v4i32>;
  def v2i64: NORVecInst<v2i64>;

  def r128:  NORRegInst<GPRC>;
  def r64:   NORRegInst<R64C>;
  def r32:   NORRegInst<R32C>;
  def r16:   NORRegInst<R16C>;
  def r8:    NORRegInst<R8C>;
}

defm NOR : BitwiseNor;

// Select bits:
class SELBInst<dag OOL, dag IOL, list<dag> pattern>:
    RRRForm<0b1000, OOL, IOL, "selb\t$rT, $rA, $rB, $rC",
            IntegerOp, pattern>;

class SELBVecInst<ValueType vectype, PatFrag vnot_frag = vnot>:
  SELBInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB, VECREG:$rC),
           [(set (vectype VECREG:$rT),
                 (or (and (vectype VECREG:$rC), (vectype VECREG:$rB)),
                     (and (vnot_frag (vectype VECREG:$rC)),
                          (vectype VECREG:$rA))))]>;

class SELBVecVCondInst<ValueType vectype>:
  SELBInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB, VECREG:$rC),
           [(set (vectype VECREG:$rT),
                 (select (vectype VECREG:$rC),
                         (vectype VECREG:$rB),
                         (vectype VECREG:$rA)))]>;

class SELBVecCondInst<ValueType vectype>:
  SELBInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB, R32C:$rC),
           [(set (vectype VECREG:$rT),
                 (select R32C:$rC,
                         (vectype VECREG:$rB),
                         (vectype VECREG:$rA)))]>;

class SELBRegInst<RegisterClass rclass>:
  SELBInst<(outs rclass:$rT), (ins rclass:$rA, rclass:$rB, rclass:$rC),
           [(set rclass:$rT,
                 (or (and rclass:$rB, rclass:$rC),
                     (and rclass:$rA, (not rclass:$rC))))]>;

class SELBRegCondInst<RegisterClass rcond, RegisterClass rclass>:
  SELBInst<(outs rclass:$rT), (ins rclass:$rA, rclass:$rB, rcond:$rC),
           [(set rclass:$rT,
                 (select rcond:$rC, rclass:$rB, rclass:$rA))]>;

multiclass SelectBits
{
  def v16i8: SELBVecInst<v16i8>;
  def v8i16: SELBVecInst<v8i16>;
  def v4i32: SELBVecInst<v4i32>;
  def v2i64: SELBVecInst<v2i64, vnot_cell_conv>;

  def r128:  SELBRegInst<GPRC>;
  def r64:   SELBRegInst<R64C>;
  def r32:   SELBRegInst<R32C>;
  def r16:   SELBRegInst<R16C>;
  def r8:    SELBRegInst<R8C>;

  def v16i8_cond: SELBVecCondInst<v16i8>;
  def v8i16_cond: SELBVecCondInst<v8i16>;
  def v4i32_cond: SELBVecCondInst<v4i32>;
  def v2i64_cond: SELBVecCondInst<v2i64>;

  def v16i8_vcond: SELBVecCondInst<v16i8>;
  def v8i16_vcond: SELBVecCondInst<v8i16>;
  def v4i32_vcond: SELBVecCondInst<v4i32>;
  def v2i64_vcond: SELBVecCondInst<v2i64>;

  def v4f32_cond:
        SELBInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB, VECREG:$rC),
                 [(set (v4f32 VECREG:$rT),
                       (select (v4i32 VECREG:$rC),
                               (v4f32 VECREG:$rB),
                               (v4f32 VECREG:$rA)))]>;

  // SELBr64_cond is defined in SPU64InstrInfo.td
  def r32_cond:   SELBRegCondInst<R32C, R32C>;
  def f32_cond:   SELBRegCondInst<R32C, R32FP>;
  def r16_cond:   SELBRegCondInst<R16C, R16C>;
  def r8_cond:    SELBRegCondInst<R8C,  R8C>;
}

defm SELB : SelectBits;

class SPUselbPatVec<ValueType vectype, SPUInstr inst>:
   Pat<(SPUselb (vectype VECREG:$rA), (vectype VECREG:$rB), (vectype VECREG:$rC)),
       (inst VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : SPUselbPatVec<v16i8, SELBv16i8>;
def : SPUselbPatVec<v8i16, SELBv8i16>;
def : SPUselbPatVec<v4i32, SELBv4i32>;
def : SPUselbPatVec<v2i64, SELBv2i64>;

class SPUselbPatReg<RegisterClass rclass, SPUInstr inst>:
   Pat<(SPUselb rclass:$rA, rclass:$rB, rclass:$rC),
       (inst rclass:$rA, rclass:$rB, rclass:$rC)>;

def : SPUselbPatReg<R8C,   SELBr8>;
def : SPUselbPatReg<R16C,  SELBr16>;
def : SPUselbPatReg<R32C,  SELBr32>;
def : SPUselbPatReg<R64C,  SELBr64>;

// EQV: Equivalence (1 for each same bit, otherwise 0)
//
// Note: There are a lot of ways to match this bit operator and these patterns
// attempt to be as exhaustive as possible.

class EQVInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm<0b10010010000, OOL, IOL, "eqv\t$rT, $rA, $rB",
           IntegerOp, pattern>;

class EQVVecInst<ValueType vectype>:
    EQVInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
            [(set (vectype VECREG:$rT),
                  (or (and (vectype VECREG:$rA), (vectype VECREG:$rB)),
                      (and (vnot (vectype VECREG:$rA)),
                           (vnot (vectype VECREG:$rB)))))]>;

class EQVRegInst<RegisterClass rclass>:
    EQVInst<(outs rclass:$rT), (ins rclass:$rA, rclass:$rB),
            [(set rclass:$rT, (or (and rclass:$rA, rclass:$rB),
                                  (and (not rclass:$rA), (not rclass:$rB))))]>;

class EQVVecPattern1<ValueType vectype>:
  EQVInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
          [(set (vectype VECREG:$rT),
                (xor (vectype VECREG:$rA), (vnot (vectype VECREG:$rB))))]>;

class EQVRegPattern1<RegisterClass rclass>:
  EQVInst<(outs rclass:$rT), (ins rclass:$rA, rclass:$rB),
          [(set rclass:$rT, (xor rclass:$rA, (not rclass:$rB)))]>;

class EQVVecPattern2<ValueType vectype>:
  EQVInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
          [(set (vectype VECREG:$rT),
                (or (and (vectype VECREG:$rA), (vectype VECREG:$rB)),
                    (vnot (or (vectype VECREG:$rA), (vectype VECREG:$rB)))))]>;

class EQVRegPattern2<RegisterClass rclass>:
  EQVInst<(outs rclass:$rT), (ins rclass:$rA, rclass:$rB),
          [(set rclass:$rT,
                (or (and rclass:$rA, rclass:$rB),
                    (not (or rclass:$rA, rclass:$rB))))]>;

class EQVVecPattern3<ValueType vectype>:
  EQVInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
          [(set (vectype VECREG:$rT),
                (not (xor (vectype VECREG:$rA), (vectype VECREG:$rB))))]>;

class EQVRegPattern3<RegisterClass rclass>:
  EQVInst<(outs rclass:$rT), (ins rclass:$rA, rclass:$rB),
          [(set rclass:$rT, (not (xor rclass:$rA, rclass:$rB)))]>;

multiclass BitEquivalence
{
  def v16i8: EQVVecInst<v16i8>;
  def v8i16: EQVVecInst<v8i16>;
  def v4i32: EQVVecInst<v4i32>;
  def v2i64: EQVVecInst<v2i64>;

  def v16i8_1: EQVVecPattern1<v16i8>;
  def v8i16_1: EQVVecPattern1<v8i16>;
  def v4i32_1: EQVVecPattern1<v4i32>;
  def v2i64_1: EQVVecPattern1<v2i64>;

  def v16i8_2: EQVVecPattern2<v16i8>;
  def v8i16_2: EQVVecPattern2<v8i16>;
  def v4i32_2: EQVVecPattern2<v4i32>;
  def v2i64_2: EQVVecPattern2<v2i64>;

  def v16i8_3: EQVVecPattern3<v16i8>;
  def v8i16_3: EQVVecPattern3<v8i16>;
  def v4i32_3: EQVVecPattern3<v4i32>;
  def v2i64_3: EQVVecPattern3<v2i64>;

  def r128:  EQVRegInst<GPRC>;
  def r64:   EQVRegInst<R64C>;
  def r32:   EQVRegInst<R32C>;
  def r16:   EQVRegInst<R16C>;
  def r8:    EQVRegInst<R8C>;

  def r128_1: EQVRegPattern1<GPRC>;
  def r64_1:  EQVRegPattern1<R64C>;
  def r32_1:  EQVRegPattern1<R32C>;
  def r16_1:  EQVRegPattern1<R16C>;
  def r8_1:   EQVRegPattern1<R8C>;

  def r128_2: EQVRegPattern2<GPRC>;
  def r64_2:  EQVRegPattern2<R64C>;
  def r32_2:  EQVRegPattern2<R32C>;
  def r16_2:  EQVRegPattern2<R16C>;
  def r8_2:   EQVRegPattern2<R8C>;

  def r128_3: EQVRegPattern3<GPRC>;
  def r64_3:  EQVRegPattern3<R64C>;
  def r32_3:  EQVRegPattern3<R32C>;
  def r16_3:  EQVRegPattern3<R16C>;
  def r8_3:   EQVRegPattern3<R8C>;
}

defm EQV: BitEquivalence;

//===----------------------------------------------------------------------===//
// Vector shuffle...
//===----------------------------------------------------------------------===//
// SPUshuffle is generated in LowerVECTOR_SHUFFLE and gets replaced with SHUFB.
// See the SPUshuffle SDNode operand above, which sets up the DAG pattern
// matcher to emit something when the LowerVECTOR_SHUFFLE generates a node with
// the SPUISD::SHUFB opcode.
//===----------------------------------------------------------------------===//

class SHUFBInst<dag OOL, dag IOL, list<dag> pattern>:
    RRRForm<0b1000, OOL, IOL, "shufb\t$rT, $rA, $rB, $rC",
            ShuffleOp, pattern>;

class SHUFBVecInst<ValueType resultvec, ValueType maskvec>:
    SHUFBInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB, VECREG:$rC),
              [(set (resultvec VECREG:$rT),
                    (SPUshuffle (resultvec VECREG:$rA),
                                (resultvec VECREG:$rB),
                                (maskvec VECREG:$rC)))]>;

class SHUFBGPRCInst:
    SHUFBInst<(outs VECREG:$rT), (ins GPRC:$rA, GPRC:$rB, VECREG:$rC),
              [/* no pattern */]>;

multiclass ShuffleBytes
{
  def v16i8     : SHUFBVecInst<v16i8, v16i8>;
  def v16i8_m32 : SHUFBVecInst<v16i8, v4i32>;
  def v8i16     : SHUFBVecInst<v8i16, v16i8>;
  def v8i16_m32 : SHUFBVecInst<v8i16, v4i32>;
  def v4i32     : SHUFBVecInst<v4i32, v16i8>;
  def v4i32_m32 : SHUFBVecInst<v4i32, v4i32>;
  def v2i64     : SHUFBVecInst<v2i64, v16i8>;
  def v2i64_m32 : SHUFBVecInst<v2i64, v4i32>;

  def v4f32     : SHUFBVecInst<v4f32, v16i8>;
  def v4f32_m32 : SHUFBVecInst<v4f32, v4i32>;

  def v2f64     : SHUFBVecInst<v2f64, v16i8>;
  def v2f64_m32 : SHUFBVecInst<v2f64, v4i32>;

  def gprc      : SHUFBGPRCInst;
}

defm SHUFB : ShuffleBytes;

//===----------------------------------------------------------------------===//
// Shift and rotate group:
//===----------------------------------------------------------------------===//

class SHLHInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm<0b11111010000, OOL, IOL, "shlh\t$rT, $rA, $rB",
           RotShiftVec, pattern>;

class SHLHVecInst<ValueType vectype>:
    SHLHInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
             [(set (vectype VECREG:$rT),
                   (SPUvec_shl (vectype VECREG:$rA), (vectype VECREG:$rB)))]>;

multiclass ShiftLeftHalfword
{
  def v8i16: SHLHVecInst<v8i16>;
  def r16:   SHLHInst<(outs R16C:$rT), (ins R16C:$rA, R16C:$rB),
                      [(set R16C:$rT, (shl R16C:$rA, R16C:$rB))]>;
  def r16_r32: SHLHInst<(outs R16C:$rT), (ins R16C:$rA, R32C:$rB),
                        [(set R16C:$rT, (shl R16C:$rA, R32C:$rB))]>;
}

defm SHLH : ShiftLeftHalfword;

//===----------------------------------------------------------------------===//

class SHLHIInst<dag OOL, dag IOL, list<dag> pattern>:
    RI7Form<0b11111010000, OOL, IOL, "shlhi\t$rT, $rA, $val",
            RotShiftVec, pattern>;

class SHLHIVecInst<ValueType vectype>:
    SHLHIInst<(outs VECREG:$rT), (ins VECREG:$rA, u7imm:$val),
              [(set (vectype VECREG:$rT),
                    (SPUvec_shl (vectype VECREG:$rA), (i16 uimm7:$val)))]>;

multiclass ShiftLeftHalfwordImm
{
  def v8i16: SHLHIVecInst<v8i16>;
  def r16: SHLHIInst<(outs R16C:$rT), (ins R16C:$rA, u7imm:$val),
                     [(set R16C:$rT, (shl R16C:$rA, (i16 uimm7:$val)))]>;
}

defm SHLHI : ShiftLeftHalfwordImm;

def : Pat<(SPUvec_shl (v8i16 VECREG:$rA), (i32 uimm7:$val)),
          (SHLHIv8i16 VECREG:$rA, (TO_IMM16 uimm7:$val))>;

def : Pat<(shl R16C:$rA, (i32 uimm7:$val)),
          (SHLHIr16 R16C:$rA, (TO_IMM16 uimm7:$val))>;

//===----------------------------------------------------------------------===//

class SHLInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm<0b11111010000, OOL, IOL, "shl\t$rT, $rA, $rB",
           RotShiftVec, pattern>;

multiclass ShiftLeftWord
{
  def v4i32:
      SHLInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
              [(set (v4i32 VECREG:$rT),
                    (SPUvec_shl (v4i32 VECREG:$rA), (v4i32 VECREG:$rB)))]>;
  def r32:
      SHLInst<(outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
              [(set R32C:$rT, (shl R32C:$rA, R32C:$rB))]>;
}

defm SHL: ShiftLeftWord;

//===----------------------------------------------------------------------===//

class SHLIInst<dag OOL, dag IOL, list<dag> pattern>:
    RI7Form<0b11111010000, OOL, IOL, "shli\t$rT, $rA, $val",
            RotShiftVec, pattern>;

multiclass ShiftLeftWordImm
{
  def v4i32:
    SHLIInst<(outs VECREG:$rT), (ins VECREG:$rA, u7imm_i32:$val),
             [(set (v4i32 VECREG:$rT),
                   (SPUvec_shl (v4i32 VECREG:$rA), (i32 uimm7:$val)))]>;

  def r32:
    SHLIInst<(outs R32C:$rT), (ins R32C:$rA, u7imm_i32:$val),
             [(set R32C:$rT, (shl R32C:$rA, (i32 uimm7:$val)))]>;
}

defm SHLI : ShiftLeftWordImm;

//===----------------------------------------------------------------------===//
// SHLQBI vec form: Note that this will shift the entire vector (the 128-bit
// register) to the left. Vector form is here to ensure type correctness.
//
// The shift count is in the lowest 3 bits (29-31) of $rB, so only a bit shift
// of 7 bits is actually possible.
//
// Note also that SHLQBI/SHLQBII are used in conjunction with SHLQBY/SHLQBYI
// to shift i64 and i128. SHLQBI is the residual left over after shifting by
// bytes with SHLQBY.

class SHLQBIInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm<0b11011011100, OOL, IOL, "shlqbi\t$rT, $rA, $rB",
           RotShiftQuad, pattern>;

class SHLQBIVecInst<ValueType vectype>:
    SHLQBIInst<(outs VECREG:$rT), (ins VECREG:$rA, R32C:$rB),
               [(set (vectype VECREG:$rT),
                     (SPUshlquad_l_bits (vectype VECREG:$rA), R32C:$rB))]>;

class SHLQBIRegInst<RegisterClass rclass>:
    SHLQBIInst<(outs rclass:$rT), (ins rclass:$rA, R32C:$rB),
               [/* no pattern */]>;

multiclass ShiftLeftQuadByBits
{
  def v16i8: SHLQBIVecInst<v16i8>;
  def v8i16: SHLQBIVecInst<v8i16>;
  def v4i32: SHLQBIVecInst<v4i32>;
  def v4f32: SHLQBIVecInst<v4f32>;
  def v2i64: SHLQBIVecInst<v2i64>;
  def v2f64: SHLQBIVecInst<v2f64>;

  def r128:  SHLQBIRegInst<GPRC>;
}

defm SHLQBI : ShiftLeftQuadByBits;

// See note above on SHLQBI. In this case, the predicate actually does then
// enforcement, whereas with SHLQBI, we have to "take it on faith."
class SHLQBIIInst<dag OOL, dag IOL, list<dag> pattern>:
    RI7Form<0b11011111100, OOL, IOL, "shlqbii\t$rT, $rA, $val",
            RotShiftQuad, pattern>;

class SHLQBIIVecInst<ValueType vectype>:
    SHLQBIIInst<(outs VECREG:$rT), (ins VECREG:$rA, u7imm_i32:$val),
                [(set (vectype VECREG:$rT),
                      (SPUshlquad_l_bits (vectype VECREG:$rA), (i32 bitshift:$val)))]>;

multiclass ShiftLeftQuadByBitsImm
{
  def v16i8 : SHLQBIIVecInst<v16i8>;
  def v8i16 : SHLQBIIVecInst<v8i16>;
  def v4i32 : SHLQBIIVecInst<v4i32>;
  def v4f32 : SHLQBIIVecInst<v4f32>;
  def v2i64 : SHLQBIIVecInst<v2i64>;
  def v2f64 : SHLQBIIVecInst<v2f64>;
}

defm SHLQBII : ShiftLeftQuadByBitsImm;

// SHLQBY, SHLQBYI vector forms: Shift the entire vector to the left by bytes,
// not by bits. See notes above on SHLQBI.

class SHLQBYInst<dag OOL, dag IOL, list<dag> pattern>:
    RI7Form<0b11111011100, OOL, IOL, "shlqby\t$rT, $rA, $rB",
            RotShiftQuad, pattern>;

class SHLQBYVecInst<ValueType vectype>:
    SHLQBYInst<(outs VECREG:$rT), (ins VECREG:$rA, R32C:$rB),
               [(set (vectype VECREG:$rT),
                     (SPUshlquad_l_bytes (vectype VECREG:$rA), R32C:$rB))]>;

multiclass ShiftLeftQuadBytes
{
  def v16i8: SHLQBYVecInst<v16i8>;
  def v8i16: SHLQBYVecInst<v8i16>;
  def v4i32: SHLQBYVecInst<v4i32>;
  def v4f32: SHLQBYVecInst<v4f32>;
  def v2i64: SHLQBYVecInst<v2i64>;
  def v2f64: SHLQBYVecInst<v2f64>;
  def r128: SHLQBYInst<(outs GPRC:$rT), (ins GPRC:$rA, R32C:$rB),
                       [(set GPRC:$rT, (SPUshlquad_l_bytes GPRC:$rA, R32C:$rB))]>;
}

defm SHLQBY: ShiftLeftQuadBytes;

class SHLQBYIInst<dag OOL, dag IOL, list<dag> pattern>:
    RI7Form<0b11111111100, OOL, IOL, "shlqbyi\t$rT, $rA, $val",
            RotShiftQuad, pattern>;

class SHLQBYIVecInst<ValueType vectype>:
    SHLQBYIInst<(outs VECREG:$rT), (ins VECREG:$rA, u7imm_i32:$val),
                [(set (vectype VECREG:$rT),
                      (SPUshlquad_l_bytes (vectype VECREG:$rA), (i32 uimm7:$val)))]>;

multiclass ShiftLeftQuadBytesImm
{
  def v16i8: SHLQBYIVecInst<v16i8>;
  def v8i16: SHLQBYIVecInst<v8i16>;
  def v4i32: SHLQBYIVecInst<v4i32>;
  def v4f32: SHLQBYIVecInst<v4f32>;
  def v2i64: SHLQBYIVecInst<v2i64>;
  def v2f64: SHLQBYIVecInst<v2f64>;
  def r128:  SHLQBYIInst<(outs GPRC:$rT), (ins GPRC:$rA, u7imm_i32:$val),
                         [(set GPRC:$rT,
                               (SPUshlquad_l_bytes GPRC:$rA, (i32 uimm7:$val)))]>;
}

defm SHLQBYI : ShiftLeftQuadBytesImm;

class SHLQBYBIInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm<0b00111001111, OOL, IOL, "shlqbybi\t$rT, $rA, $rB",
           RotShiftQuad, pattern>;

class SHLQBYBIVecInst<ValueType vectype>:
    SHLQBYBIInst<(outs VECREG:$rT), (ins VECREG:$rA, R32C:$rB),
                [/* no pattern */]>;

class SHLQBYBIRegInst<RegisterClass rclass>:
    SHLQBYBIInst<(outs rclass:$rT), (ins rclass:$rA, R32C:$rB),
                 [/* no pattern */]>;

multiclass ShiftLeftQuadBytesBitCount
{
  def v16i8: SHLQBYBIVecInst<v16i8>;
  def v8i16: SHLQBYBIVecInst<v8i16>;
  def v4i32: SHLQBYBIVecInst<v4i32>;
  def v4f32: SHLQBYBIVecInst<v4f32>;
  def v2i64: SHLQBYBIVecInst<v2i64>;
  def v2f64: SHLQBYBIVecInst<v2f64>;

  def r128:  SHLQBYBIRegInst<GPRC>;
}

defm SHLQBYBI : ShiftLeftQuadBytesBitCount;

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// Rotate halfword:
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
class ROTHInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm<0b00111010000, OOL, IOL, "roth\t$rT, $rA, $rB",
           RotShiftVec, pattern>;

class ROTHVecInst<ValueType vectype>:
    ROTHInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
             [(set (vectype VECREG:$rT),
                   (SPUvec_rotl VECREG:$rA, (v8i16 VECREG:$rB)))]>;

class ROTHRegInst<RegisterClass rclass>:
    ROTHInst<(outs rclass:$rT), (ins rclass:$rA, rclass:$rB),
             [(set rclass:$rT, (rotl rclass:$rA, rclass:$rB))]>;

multiclass RotateLeftHalfword
{
  def v8i16: ROTHVecInst<v8i16>;
  def r16: ROTHRegInst<R16C>;
}

defm ROTH: RotateLeftHalfword;

def ROTHr16_r32: ROTHInst<(outs R16C:$rT), (ins R16C:$rA, R32C:$rB),
                          [(set R16C:$rT, (rotl R16C:$rA, R32C:$rB))]>;

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// Rotate halfword, immediate:
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
class ROTHIInst<dag OOL, dag IOL, list<dag> pattern>:
    RI7Form<0b00111110000, OOL, IOL, "rothi\t$rT, $rA, $val",
            RotShiftVec, pattern>;

class ROTHIVecInst<ValueType vectype>:
    ROTHIInst<(outs VECREG:$rT), (ins VECREG:$rA, u7imm:$val),
              [(set (vectype VECREG:$rT),
                    (SPUvec_rotl VECREG:$rA, (i16 uimm7:$val)))]>;

multiclass RotateLeftHalfwordImm
{
  def v8i16: ROTHIVecInst<v8i16>;
  def r16: ROTHIInst<(outs R16C:$rT), (ins R16C:$rA, u7imm:$val),
                     [(set R16C:$rT, (rotl R16C:$rA, (i16 uimm7:$val)))]>;
  def r16_r32: ROTHIInst<(outs R16C:$rT), (ins R16C:$rA, u7imm_i32:$val),
                         [(set R16C:$rT, (rotl R16C:$rA, (i32 uimm7:$val)))]>;
}

defm ROTHI: RotateLeftHalfwordImm;

def : Pat<(SPUvec_rotl (v8i16 VECREG:$rA), (i32 uimm7:$val)),
          (ROTHIv8i16 VECREG:$rA, (TO_IMM16 imm:$val))>;

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// Rotate word:
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

class ROTInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm<0b00011010000, OOL, IOL, "rot\t$rT, $rA, $rB",
           RotShiftVec, pattern>;

class ROTVecInst<ValueType vectype>:
    ROTInst<(outs VECREG:$rT), (ins VECREG:$rA, R32C:$rB),
            [(set (vectype VECREG:$rT),
                  (SPUvec_rotl (vectype VECREG:$rA), R32C:$rB))]>;

class ROTRegInst<RegisterClass rclass>:
    ROTInst<(outs rclass:$rT), (ins rclass:$rA, R32C:$rB),
            [(set rclass:$rT,
                  (rotl rclass:$rA, R32C:$rB))]>;

multiclass RotateLeftWord
{
  def v4i32: ROTVecInst<v4i32>;
  def r32:   ROTRegInst<R32C>;
}

defm ROT: RotateLeftWord;

// The rotate amount is in the same bits whether we've got an 8-bit, 16-bit or
// 32-bit register
def ROTr32_r16_anyext:
    ROTInst<(outs R32C:$rT), (ins R32C:$rA, R16C:$rB),
            [(set R32C:$rT, (rotl R32C:$rA, (i32 (anyext R16C:$rB))))]>;

def : Pat<(rotl R32C:$rA, (i32 (zext R16C:$rB))),
          (ROTr32_r16_anyext R32C:$rA, R16C:$rB)>;

def : Pat<(rotl R32C:$rA, (i32 (sext R16C:$rB))),
          (ROTr32_r16_anyext R32C:$rA, R16C:$rB)>;

def ROTr32_r8_anyext:
    ROTInst<(outs R32C:$rT), (ins R32C:$rA, R8C:$rB),
            [(set R32C:$rT, (rotl R32C:$rA, (i32 (anyext R8C:$rB))))]>;

def : Pat<(rotl R32C:$rA, (i32 (zext R8C:$rB))),
          (ROTr32_r8_anyext R32C:$rA, R8C:$rB)>;

def : Pat<(rotl R32C:$rA, (i32 (sext R8C:$rB))),
          (ROTr32_r8_anyext R32C:$rA, R8C:$rB)>;

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// Rotate word, immediate
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

class ROTIInst<dag OOL, dag IOL, list<dag> pattern>:
    RI7Form<0b00011110000, OOL, IOL, "roti\t$rT, $rA, $val",
            RotShiftVec, pattern>;

class ROTIVecInst<ValueType vectype, Operand optype, ValueType inttype, PatLeaf pred>:
    ROTIInst<(outs VECREG:$rT), (ins VECREG:$rA, optype:$val),
             [(set (vectype VECREG:$rT),
                   (SPUvec_rotl (vectype VECREG:$rA), (inttype pred:$val)))]>;

class ROTIRegInst<RegisterClass rclass, Operand optype, ValueType inttype, PatLeaf pred>:
    ROTIInst<(outs rclass:$rT), (ins rclass:$rA, optype:$val),
             [(set rclass:$rT, (rotl rclass:$rA, (inttype pred:$val)))]>;

multiclass RotateLeftWordImm
{
  def v4i32: ROTIVecInst<v4i32, u7imm_i32, i32, uimm7>;
  def v4i32_i16: ROTIVecInst<v4i32, u7imm, i16, uimm7>;
  def v4i32_i8:  ROTIVecInst<v4i32, u7imm_i8, i8, uimm7>;

  def r32:       ROTIRegInst<R32C, u7imm_i32, i32, uimm7>;
  def r32_i16:   ROTIRegInst<R32C, u7imm, i16, uimm7>;
  def r32_i8:    ROTIRegInst<R32C, u7imm_i8, i8, uimm7>;
}

defm ROTI : RotateLeftWordImm;

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// Rotate quad by byte (count)
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

class ROTQBYInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm<0b00111011100, OOL, IOL, "rotqby\t$rT, $rA, $rB",
           RotShiftQuad, pattern>;

class ROTQBYGenInst<ValueType type, RegisterClass rc>:
    ROTQBYInst<(outs rc:$rT), (ins rc:$rA, R32C:$rB),
               [(set (type rc:$rT),
                     (SPUrotbytes_left (type rc:$rA), R32C:$rB))]>;

class ROTQBYVecInst<ValueType type>:
    ROTQBYGenInst<type, VECREG>;

multiclass RotateQuadLeftByBytes
{
  def v16i8: ROTQBYVecInst<v16i8>;
  def v8i16: ROTQBYVecInst<v8i16>;
  def v4i32: ROTQBYVecInst<v4i32>;
  def v4f32: ROTQBYVecInst<v4f32>;
  def v2i64: ROTQBYVecInst<v2i64>;
  def v2f64: ROTQBYVecInst<v2f64>;
  def i128:  ROTQBYGenInst<i128, GPRC>;
}

defm ROTQBY: RotateQuadLeftByBytes;

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// Rotate quad by byte (count), immediate
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

class ROTQBYIInst<dag OOL, dag IOL, list<dag> pattern>:
    RI7Form<0b00111111100, OOL, IOL, "rotqbyi\t$rT, $rA, $val",
            RotShiftQuad, pattern>;

class ROTQBYIGenInst<ValueType type, RegisterClass rclass>:
    ROTQBYIInst<(outs rclass:$rT), (ins rclass:$rA, u7imm:$val),
                [(set (type rclass:$rT),
                      (SPUrotbytes_left (type rclass:$rA), (i16 uimm7:$val)))]>;

class ROTQBYIVecInst<ValueType vectype>:
    ROTQBYIGenInst<vectype, VECREG>;

multiclass RotateQuadByBytesImm
{
  def v16i8: ROTQBYIVecInst<v16i8>;
  def v8i16: ROTQBYIVecInst<v8i16>;
  def v4i32: ROTQBYIVecInst<v4i32>;
  def v4f32: ROTQBYIVecInst<v4f32>;
  def v2i64: ROTQBYIVecInst<v2i64>;
  def vfi64: ROTQBYIVecInst<v2f64>;
  def i128:  ROTQBYIGenInst<i128, GPRC>;
}

defm ROTQBYI: RotateQuadByBytesImm;

// See ROTQBY note above.
class ROTQBYBIInst<dag OOL, dag IOL, list<dag> pattern>:
    RI7Form<0b00110011100, OOL, IOL,
      "rotqbybi\t$rT, $rA, $shift",
      RotShiftQuad, pattern>;

class ROTQBYBIVecInst<ValueType vectype, RegisterClass rclass>:
    ROTQBYBIInst<(outs VECREG:$rT), (ins VECREG:$rA, rclass:$shift),
      [(set (vectype VECREG:$rT),
            (SPUrotbytes_left_bits (vectype VECREG:$rA), rclass:$shift))]>;

multiclass RotateQuadByBytesByBitshift {
  def v16i8_r32: ROTQBYBIVecInst<v16i8, R32C>;
  def v8i16_r32: ROTQBYBIVecInst<v8i16, R32C>;
  def v4i32_r32: ROTQBYBIVecInst<v4i32, R32C>;
  def v2i64_r32: ROTQBYBIVecInst<v2i64, R32C>;
}

defm ROTQBYBI : RotateQuadByBytesByBitshift;

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// See ROTQBY note above.
//
// Assume that the user of this instruction knows to shift the rotate count
// into bit 29
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

class ROTQBIInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm<0b00011011100, OOL, IOL, "rotqbi\t$rT, $rA, $rB",
           RotShiftQuad, pattern>;

class ROTQBIVecInst<ValueType vectype>:
    ROTQBIInst<(outs VECREG:$rT), (ins VECREG:$rA, R32C:$rB),
               [/* no pattern yet */]>;

class ROTQBIRegInst<RegisterClass rclass>:
    ROTQBIInst<(outs rclass:$rT), (ins rclass:$rA, R32C:$rB),
               [/* no pattern yet */]>;

multiclass RotateQuadByBitCount
{
  def v16i8: ROTQBIVecInst<v16i8>;
  def v8i16: ROTQBIVecInst<v8i16>;
  def v4i32: ROTQBIVecInst<v4i32>;
  def v2i64: ROTQBIVecInst<v2i64>;

  def r128:  ROTQBIRegInst<GPRC>;
  def r64:   ROTQBIRegInst<R64C>;
}

defm ROTQBI: RotateQuadByBitCount;

class ROTQBIIInst<dag OOL, dag IOL, list<dag> pattern>:
    RI7Form<0b00011111100, OOL, IOL, "rotqbii\t$rT, $rA, $val",
            RotShiftQuad, pattern>;

class ROTQBIIVecInst<ValueType vectype, Operand optype, ValueType inttype,
                     PatLeaf pred>:
    ROTQBIIInst<(outs VECREG:$rT), (ins VECREG:$rA, optype:$val),
                [/* no pattern yet */]>;

class ROTQBIIRegInst<RegisterClass rclass, Operand optype, ValueType inttype,
                     PatLeaf pred>:
    ROTQBIIInst<(outs rclass:$rT), (ins rclass:$rA, optype:$val),
                [/* no pattern yet */]>;

multiclass RotateQuadByBitCountImm
{
  def v16i8: ROTQBIIVecInst<v16i8, u7imm_i32, i32, uimm7>;
  def v8i16: ROTQBIIVecInst<v8i16, u7imm_i32, i32, uimm7>;
  def v4i32: ROTQBIIVecInst<v4i32, u7imm_i32, i32, uimm7>;
  def v2i64: ROTQBIIVecInst<v2i64, u7imm_i32, i32, uimm7>;

  def r128:  ROTQBIIRegInst<GPRC, u7imm_i32, i32, uimm7>;
  def r64:   ROTQBIIRegInst<R64C, u7imm_i32, i32, uimm7>;
}

defm ROTQBII : RotateQuadByBitCountImm;

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// ROTHM v8i16 form:
// NOTE(1): No vector rotate is generated by the C/C++ frontend (today),
//          so this only matches a synthetically generated/lowered code
//          fragment.
// NOTE(2): $rB must be negated before the right rotate!
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

class ROTHMInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm<0b10111010000, OOL, IOL, "rothm\t$rT, $rA, $rB",
           RotShiftVec, pattern>;

def ROTHMv8i16:
    ROTHMInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
              [/* see patterns below - $rB must be negated */]>;

def : Pat<(SPUvec_srl (v8i16 VECREG:$rA), (v8i16 VECREG:$rB)),
          (ROTHMv8i16 VECREG:$rA, (SFHIvec VECREG:$rB, 0))>;

// ROTHM r16 form: Rotate 16-bit quantity to right, zero fill at the left
// Note: This instruction doesn't match a pattern because rB must be negated
// for the instruction to work. Thus, the pattern below the instruction!

def ROTHMr16:
    ROTHMInst<(outs R16C:$rT), (ins R16C:$rA, R32C:$rB),
              [/* see patterns below - $rB must be negated! */]>;

def : Pat<(srl R16C:$rA, R32C:$rB),
          (ROTHMr16 R16C:$rA, (SFIr32 R32C:$rB, 0))>;

def : Pat<(srl R16C:$rA, R16C:$rB),
          (ROTHMr16 R16C:$rA,
                    (SFIr32 (XSHWr16 R16C:$rB), 0))>;

def : Pat<(srl R16C:$rA, R8C:$rB),
          (ROTHMr16 R16C:$rA,
                    (SFIr32 (XSHWr16 (XSBHr8 R8C:$rB) ), 0))>;

// ROTHMI v8i16 form: See the comment for ROTHM v8i16. The difference here is
// that the immediate can be complemented, so that the user doesn't have to
// worry about it.

class ROTHMIInst<dag OOL, dag IOL, list<dag> pattern>:
    RI7Form<0b10111110000, OOL, IOL, "rothmi\t$rT, $rA, $val",
            RotShiftVec, pattern>;

def ROTHMIv8i16:
    ROTHMIInst<(outs VECREG:$rT), (ins VECREG:$rA, rothNeg7imm:$val),
               [/* no pattern */]>;

def : Pat<(SPUvec_srl (v8i16 VECREG:$rA), (i32 imm:$val)),
          (ROTHMIv8i16 VECREG:$rA, imm:$val)>;

def: Pat<(SPUvec_srl (v8i16 VECREG:$rA), (i16 imm:$val)),
         (ROTHMIv8i16 VECREG:$rA, (TO_IMM32 imm:$val))>;

def: Pat<(SPUvec_srl (v8i16 VECREG:$rA), (i8 imm:$val)),
         (ROTHMIv8i16 VECREG:$rA, (TO_IMM32 imm:$val))>;

def ROTHMIr16:
    ROTHMIInst<(outs R16C:$rT), (ins R16C:$rA, rothNeg7imm:$val),
               [/* no pattern */]>;

def: Pat<(srl R16C:$rA, (i32 uimm7:$val)),
         (ROTHMIr16 R16C:$rA, uimm7:$val)>;

def: Pat<(srl R16C:$rA, (i16 uimm7:$val)),
         (ROTHMIr16 R16C:$rA, (TO_IMM32 uimm7:$val))>;

def: Pat<(srl R16C:$rA, (i8 uimm7:$val)),
         (ROTHMIr16 R16C:$rA, (TO_IMM32 uimm7:$val))>;

// ROTM v4i32 form: See the ROTHM v8i16 comments.
class ROTMInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm<0b10011010000, OOL, IOL, "rotm\t$rT, $rA, $rB",
           RotShiftVec, pattern>;

def ROTMv4i32:
    ROTMInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
             [/* see patterns below - $rB must be negated */]>;

def : Pat<(SPUvec_srl (v4i32 VECREG:$rA), (v4i32 VECREG:$rB)),
          (ROTMv4i32 VECREG:$rA, (SFIvec VECREG:$rB, 0))>;

def ROTMr32:
    ROTMInst<(outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
             [/* see patterns below - $rB must be negated */]>;

def : Pat<(srl R32C:$rA, R32C:$rB),
          (ROTMr32 R32C:$rA, (SFIr32 R32C:$rB, 0))>;

def : Pat<(srl R32C:$rA, R16C:$rB),
          (ROTMr32 R32C:$rA,
                   (SFIr32 (XSHWr16 R16C:$rB), 0))>;

def : Pat<(srl R32C:$rA, R8C:$rB),
          (ROTMr32 R32C:$rA,
                   (SFIr32 (XSHWr16 (XSBHr8 R8C:$rB)), 0))>;

// ROTMI v4i32 form: See the comment for ROTHM v8i16.
def ROTMIv4i32:
    RI7Form<0b10011110000, (outs VECREG:$rT), (ins VECREG:$rA, rotNeg7imm:$val),
      "rotmi\t$rT, $rA, $val", RotShiftVec,
      [(set (v4i32 VECREG:$rT),
            (SPUvec_srl VECREG:$rA, (i32 uimm7:$val)))]>;

def : Pat<(SPUvec_srl (v4i32 VECREG:$rA), (i16 uimm7:$val)),
          (ROTMIv4i32 VECREG:$rA, (TO_IMM32 uimm7:$val))>;

def : Pat<(SPUvec_srl (v4i32 VECREG:$rA), (i8 uimm7:$val)),
          (ROTMIv4i32 VECREG:$rA, (TO_IMM32 uimm7:$val))>;

// ROTMI r32 form: know how to complement the immediate value.
def ROTMIr32:
    RI7Form<0b10011110000, (outs R32C:$rT), (ins R32C:$rA, rotNeg7imm:$val),
      "rotmi\t$rT, $rA, $val", RotShiftVec,
      [(set R32C:$rT, (srl R32C:$rA, (i32 uimm7:$val)))]>;

def : Pat<(srl R32C:$rA, (i16 imm:$val)),
          (ROTMIr32 R32C:$rA, (TO_IMM32 uimm7:$val))>;

def : Pat<(srl R32C:$rA, (i8 imm:$val)),
          (ROTMIr32 R32C:$rA, (TO_IMM32 uimm7:$val))>;

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// ROTQMBY: This is a vector form merely so that when used in an
// instruction pattern, type checking will succeed. This instruction assumes
// that the user knew to negate $rB.
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

class ROTQMBYInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm<0b10111011100, OOL, IOL, "rotqmby\t$rT, $rA, $rB",
           RotShiftQuad, pattern>;

class ROTQMBYVecInst<ValueType vectype>:
    ROTQMBYInst<(outs VECREG:$rT), (ins VECREG:$rA, R32C:$rB),
                [/* no pattern, $rB must be negated */]>;

class ROTQMBYRegInst<RegisterClass rclass>:
    ROTQMBYInst<(outs rclass:$rT), (ins rclass:$rA, R32C:$rB),
                [/* no pattern */]>;

multiclass RotateQuadBytes
{
  def v16i8: ROTQMBYVecInst<v16i8>;
  def v8i16: ROTQMBYVecInst<v8i16>;
  def v4i32: ROTQMBYVecInst<v4i32>;
  def v2i64: ROTQMBYVecInst<v2i64>;

  def r128: ROTQMBYRegInst<GPRC>;
  def r64:  ROTQMBYRegInst<R64C>;
}

defm ROTQMBY : RotateQuadBytes;

def : Pat<(SPUsrl_bytes GPRC:$rA, R32C:$rB),
          (ROTQMBYr128  GPRC:$rA, 
                        (SFIr32 R32C:$rB, 0))>;

class ROTQMBYIInst<dag OOL, dag IOL, list<dag> pattern>:
    RI7Form<0b10111111100, OOL, IOL, "rotqmbyi\t$rT, $rA, $val",
            RotShiftQuad, pattern>;

class ROTQMBYIVecInst<ValueType vectype>:
    ROTQMBYIInst<(outs VECREG:$rT), (ins VECREG:$rA, rotNeg7imm:$val),
                 [/* no pattern */]>;

class ROTQMBYIRegInst<RegisterClass rclass, Operand optype, ValueType inttype,
                      PatLeaf pred>:
    ROTQMBYIInst<(outs rclass:$rT), (ins rclass:$rA, optype:$val),
                 [/* no pattern */]>;

// 128-bit zero extension form:
class ROTQMBYIZExtInst<RegisterClass rclass, Operand optype, PatLeaf pred>:
    ROTQMBYIInst<(outs GPRC:$rT), (ins rclass:$rA, optype:$val),
                 [/* no pattern */]>;

multiclass RotateQuadBytesImm
{
  def v16i8: ROTQMBYIVecInst<v16i8>;
  def v8i16: ROTQMBYIVecInst<v8i16>;
  def v4i32: ROTQMBYIVecInst<v4i32>;
  def v2i64: ROTQMBYIVecInst<v2i64>;

  def r128:  ROTQMBYIRegInst<GPRC, rotNeg7imm, i32, uimm7>;
  def r64:   ROTQMBYIRegInst<R64C, rotNeg7imm, i32, uimm7>;
  
  def r128_zext_r8:  ROTQMBYIZExtInst<R8C, rotNeg7imm, uimm7>;
  def r128_zext_r16: ROTQMBYIZExtInst<R16C, rotNeg7imm, uimm7>;
  def r128_zext_r32: ROTQMBYIZExtInst<R32C, rotNeg7imm, uimm7>;
  def r128_zext_r64: ROTQMBYIZExtInst<R64C, rotNeg7imm, uimm7>;
}

defm ROTQMBYI : RotateQuadBytesImm;

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// Rotate right and mask by bit count
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

class ROTQMBYBIInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm<0b10110011100, OOL, IOL, "rotqmbybi\t$rT, $rA, $rB",
           RotShiftQuad, pattern>;

class ROTQMBYBIVecInst<ValueType vectype>:
    ROTQMBYBIInst<(outs VECREG:$rT), (ins VECREG:$rA, R32C:$rB),
                  [/* no pattern, */]>;

multiclass RotateMaskQuadByBitCount
{
  def v16i8: ROTQMBYBIVecInst<v16i8>;
  def v8i16: ROTQMBYBIVecInst<v8i16>;
  def v4i32: ROTQMBYBIVecInst<v4i32>;
  def v2i64: ROTQMBYBIVecInst<v2i64>;
  def r128: ROTQMBYBIInst<(outs GPRC:$rT), (ins GPRC:$rA, R32C:$rB),
                           [/*no pattern*/]>;
}

defm ROTQMBYBI: RotateMaskQuadByBitCount;

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// Rotate quad and mask by bits
// Note that the rotate amount has to be negated
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

class ROTQMBIInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm<0b10011011100, OOL, IOL, "rotqmbi\t$rT, $rA, $rB",
           RotShiftQuad, pattern>;

class ROTQMBIVecInst<ValueType vectype>:
    ROTQMBIInst<(outs VECREG:$rT), (ins VECREG:$rA, R32C:$rB),
                [/* no pattern */]>;

class ROTQMBIRegInst<RegisterClass rclass>:
    ROTQMBIInst<(outs rclass:$rT), (ins rclass:$rA, R32C:$rB),
                [/* no pattern */]>;

multiclass RotateMaskQuadByBits
{
  def v16i8: ROTQMBIVecInst<v16i8>;
  def v8i16: ROTQMBIVecInst<v8i16>;
  def v4i32: ROTQMBIVecInst<v4i32>;
  def v2i64: ROTQMBIVecInst<v2i64>;

  def r128:  ROTQMBIRegInst<GPRC>;
  def r64:   ROTQMBIRegInst<R64C>;
}

defm ROTQMBI: RotateMaskQuadByBits;

def : Pat<(srl GPRC:$rA, R32C:$rB),
          (ROTQMBYBIr128 (ROTQMBIr128  GPRC:$rA, 
                                       (SFIr32 R32C:$rB, 0)),
                         (SFIr32 R32C:$rB, 0))>;


//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// Rotate quad and mask by bits, immediate
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

class ROTQMBIIInst<dag OOL, dag IOL, list<dag> pattern>:
    RI7Form<0b10011111100, OOL, IOL, "rotqmbii\t$rT, $rA, $val",
            RotShiftQuad, pattern>;

class ROTQMBIIVecInst<ValueType vectype>:
   ROTQMBIIInst<(outs VECREG:$rT), (ins VECREG:$rA, rotNeg7imm:$val),
                 [/* no pattern */]>;

class ROTQMBIIRegInst<RegisterClass rclass>:
   ROTQMBIIInst<(outs rclass:$rT), (ins rclass:$rA, rotNeg7imm:$val),
                 [/* no pattern */]>;

multiclass RotateMaskQuadByBitsImm
{
  def v16i8: ROTQMBIIVecInst<v16i8>;
  def v8i16: ROTQMBIIVecInst<v8i16>;
  def v4i32: ROTQMBIIVecInst<v4i32>;
  def v2i64: ROTQMBIIVecInst<v2i64>;

  def r128:  ROTQMBIIRegInst<GPRC>;
  def r64:   ROTQMBIIRegInst<R64C>;
}

defm ROTQMBII: RotateMaskQuadByBitsImm;

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

def ROTMAHv8i16:
    RRForm<0b01111010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "rotmah\t$rT, $rA, $rB", RotShiftVec,
      [/* see patterns below - $rB must be negated */]>;

def : Pat<(SPUvec_sra (v8i16 VECREG:$rA), (v8i16 VECREG:$rB)),
          (ROTMAHv8i16 VECREG:$rA, (SFHIvec VECREG:$rB, 0))>;

def ROTMAHr16:
    RRForm<0b01111010000, (outs R16C:$rT), (ins R16C:$rA, R32C:$rB),
      "rotmah\t$rT, $rA, $rB", RotShiftVec,
      [/* see patterns below - $rB must be negated */]>;

def : Pat<(sra R16C:$rA, R32C:$rB),
          (ROTMAHr16 R16C:$rA, (SFIr32 R32C:$rB, 0))>;

def : Pat<(sra R16C:$rA, R16C:$rB),
          (ROTMAHr16 R16C:$rA,
                     (SFIr32 (XSHWr16 R16C:$rB), 0))>;

def : Pat<(sra R16C:$rA, R8C:$rB),
          (ROTMAHr16 R16C:$rA,
                     (SFIr32 (XSHWr16 (XSBHr8 R8C:$rB)), 0))>;

def ROTMAHIv8i16:
    RRForm<0b01111110000, (outs VECREG:$rT), (ins VECREG:$rA, rothNeg7imm:$val),
      "rotmahi\t$rT, $rA, $val", RotShiftVec,
      [(set (v8i16 VECREG:$rT),
            (SPUvec_sra (v8i16 VECREG:$rA), (i32 uimm7:$val)))]>;

def : Pat<(SPUvec_sra (v8i16 VECREG:$rA), (i16 uimm7:$val)),
          (ROTMAHIv8i16 (v8i16 VECREG:$rA), (TO_IMM32 uimm7:$val))>;

def : Pat<(SPUvec_sra (v8i16 VECREG:$rA), (i8 uimm7:$val)),
          (ROTMAHIv8i16 (v8i16 VECREG:$rA), (TO_IMM32 uimm7:$val))>;

def ROTMAHIr16:
    RRForm<0b01111110000, (outs R16C:$rT), (ins R16C:$rA, rothNeg7imm_i16:$val),
      "rotmahi\t$rT, $rA, $val", RotShiftVec,
      [(set R16C:$rT, (sra R16C:$rA, (i16 uimm7:$val)))]>;

def : Pat<(sra R16C:$rA, (i32 imm:$val)),
          (ROTMAHIr16 R16C:$rA, (TO_IMM32 uimm7:$val))>;

def : Pat<(sra R16C:$rA, (i8 imm:$val)),
          (ROTMAHIr16 R16C:$rA, (TO_IMM32 uimm7:$val))>;

def ROTMAv4i32:
    RRForm<0b01011010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "rotma\t$rT, $rA, $rB", RotShiftVec,
      [/* see patterns below - $rB must be negated */]>;

def : Pat<(SPUvec_sra (v4i32 VECREG:$rA), (v4i32 VECREG:$rB)),
          (ROTMAv4i32 VECREG:$rA, (SFIvec (v4i32 VECREG:$rB), 0))>;

def ROTMAr32:
    RRForm<0b01011010000, (outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
      "rotma\t$rT, $rA, $rB", RotShiftVec,
      [/* see patterns below - $rB must be negated */]>;

def : Pat<(sra R32C:$rA, R32C:$rB),
          (ROTMAr32 R32C:$rA, (SFIr32 R32C:$rB, 0))>;

def : Pat<(sra R32C:$rA, R16C:$rB),
          (ROTMAr32 R32C:$rA,
                    (SFIr32 (XSHWr16 R16C:$rB), 0))>;

def : Pat<(sra R32C:$rA, R8C:$rB),
          (ROTMAr32 R32C:$rA,
                    (SFIr32 (XSHWr16 (XSBHr8 R8C:$rB)), 0))>;

class ROTMAIInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm<0b01011110000, OOL, IOL,
      "rotmai\t$rT, $rA, $val",
      RotShiftVec, pattern>;

class ROTMAIVecInst<ValueType vectype, Operand intop, ValueType inttype>:
    ROTMAIInst<(outs VECREG:$rT), (ins VECREG:$rA, intop:$val),
      [(set (vectype VECREG:$rT),
            (SPUvec_sra VECREG:$rA, (inttype uimm7:$val)))]>;

class ROTMAIRegInst<RegisterClass rclass, Operand intop, ValueType inttype>:
    ROTMAIInst<(outs rclass:$rT), (ins rclass:$rA, intop:$val),
      [(set rclass:$rT, (sra rclass:$rA, (inttype uimm7:$val)))]>;

multiclass RotateMaskAlgebraicImm {
  def v2i64_i32 : ROTMAIVecInst<v2i64, rotNeg7imm, i32>;
  def v4i32_i32 : ROTMAIVecInst<v4i32, rotNeg7imm, i32>;
  def r64_i32 : ROTMAIRegInst<R64C, rotNeg7imm, i32>;
  def r32_i32 : ROTMAIRegInst<R32C, rotNeg7imm, i32>;
}

defm ROTMAI : RotateMaskAlgebraicImm;

//===----------------------------------------------------------------------===//
// Branch and conditionals:
//===----------------------------------------------------------------------===//

let isTerminator = 1, isBarrier = 1 in {
  // Halt If Equal (r32 preferred slot only, no vector form)
  def HEQr32:
    RRForm_3<0b00011011110, (outs), (ins R32C:$rA, R32C:$rB),
      "heq\t$rA, $rB", BranchResolv,
      [/* no pattern to match */]>;

  def HEQIr32 :
    RI10Form_2<0b11111110, (outs), (ins R32C:$rA, s10imm:$val),
      "heqi\t$rA, $val", BranchResolv,
      [/* no pattern to match */]>;

  // HGT/HGTI: These instructions use signed arithmetic for the comparison,
  // contrasting with HLGT/HLGTI, which use unsigned comparison:
  def HGTr32:
    RRForm_3<0b00011010010, (outs), (ins R32C:$rA, R32C:$rB),
      "hgt\t$rA, $rB", BranchResolv,
      [/* no pattern to match */]>;

  def HGTIr32:
    RI10Form_2<0b11110010, (outs), (ins R32C:$rA, s10imm:$val),
      "hgti\t$rA, $val", BranchResolv,
      [/* no pattern to match */]>;

  def HLGTr32:
    RRForm_3<0b00011011010, (outs), (ins R32C:$rA, R32C:$rB),
      "hlgt\t$rA, $rB", BranchResolv,
      [/* no pattern to match */]>;

  def HLGTIr32:
    RI10Form_2<0b11111010, (outs), (ins R32C:$rA, s10imm:$val),
      "hlgti\t$rA, $val", BranchResolv,
      [/* no pattern to match */]>;
}

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// Comparison operators for i8, i16 and i32:
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

class CEQBInst<dag OOL, dag IOL, list<dag> pattern> :
  RRForm<0b00001011110, OOL, IOL, "ceqb\t$rT, $rA, $rB",
         ByteOp, pattern>;

multiclass CmpEqualByte
{
  def v16i8 :
    CEQBInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      [(set (v16i8 VECREG:$rT), (seteq (v8i16 VECREG:$rA),
                                       (v8i16 VECREG:$rB)))]>;

  def r8 :
    CEQBInst<(outs R8C:$rT), (ins R8C:$rA, R8C:$rB),
             [(set R8C:$rT, (seteq R8C:$rA, R8C:$rB))]>;
}

class CEQBIInst<dag OOL, dag IOL, list<dag> pattern> :
  RI10Form<0b01111110, OOL, IOL, "ceqbi\t$rT, $rA, $val",
           ByteOp, pattern>;

multiclass CmpEqualByteImm
{
  def v16i8 :
    CEQBIInst<(outs VECREG:$rT), (ins VECREG:$rA, s10imm_i8:$val),
              [(set (v16i8 VECREG:$rT), (seteq (v16i8 VECREG:$rA),
                                               v16i8SExt8Imm:$val))]>;
  def r8:
    CEQBIInst<(outs R8C:$rT), (ins R8C:$rA, s10imm_i8:$val),
             [(set R8C:$rT, (seteq R8C:$rA, immSExt8:$val))]>;
}

class CEQHInst<dag OOL, dag IOL, list<dag> pattern> :
  RRForm<0b00010011110, OOL, IOL, "ceqh\t$rT, $rA, $rB",
         ByteOp, pattern>;

multiclass CmpEqualHalfword
{
  def v8i16 : CEQHInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
                       [(set (v8i16 VECREG:$rT), (seteq (v8i16 VECREG:$rA),
                                                        (v8i16 VECREG:$rB)))]>;

  def r16 : CEQHInst<(outs R16C:$rT), (ins R16C:$rA, R16C:$rB),
                     [(set R16C:$rT, (seteq R16C:$rA, R16C:$rB))]>;
}

class CEQHIInst<dag OOL, dag IOL, list<dag> pattern> :
  RI10Form<0b10111110, OOL, IOL, "ceqhi\t$rT, $rA, $val",
           ByteOp, pattern>;

multiclass CmpEqualHalfwordImm
{
  def v8i16 : CEQHIInst<(outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
                        [(set (v8i16 VECREG:$rT),
                              (seteq (v8i16 VECREG:$rA),
                                     (v8i16 v8i16SExt10Imm:$val)))]>;
  def r16 : CEQHIInst<(outs R16C:$rT), (ins R16C:$rA, s10imm:$val),
                      [(set R16C:$rT, (seteq R16C:$rA, i16ImmSExt10:$val))]>;
}

class CEQInst<dag OOL, dag IOL, list<dag> pattern> :
  RRForm<0b00000011110, OOL, IOL, "ceq\t$rT, $rA, $rB",
         ByteOp, pattern>;

multiclass CmpEqualWord
{
  def v4i32 : CEQInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
                      [(set (v4i32 VECREG:$rT),
                            (seteq (v4i32 VECREG:$rA), (v4i32 VECREG:$rB)))]>;

  def r32 : CEQInst<(outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
                    [(set R32C:$rT, (seteq R32C:$rA, R32C:$rB))]>;
}

class CEQIInst<dag OOL, dag IOL, list<dag> pattern> :
  RI10Form<0b00111110, OOL, IOL, "ceqi\t$rT, $rA, $val",
           ByteOp, pattern>;

multiclass CmpEqualWordImm
{
  def v4i32 : CEQIInst<(outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
                       [(set (v4i32 VECREG:$rT),
                             (seteq (v4i32 VECREG:$rA),
                                    (v4i32 v4i32SExt16Imm:$val)))]>;

  def r32: CEQIInst<(outs R32C:$rT), (ins R32C:$rA, s10imm_i32:$val),
                    [(set R32C:$rT, (seteq R32C:$rA, i32ImmSExt10:$val))]>;
}

class CGTBInst<dag OOL, dag IOL, list<dag> pattern> :
  RRForm<0b00001010010, OOL, IOL, "cgtb\t$rT, $rA, $rB",
         ByteOp, pattern>;

multiclass CmpGtrByte
{
  def v16i8 :
    CGTBInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      [(set (v16i8 VECREG:$rT), (setgt (v8i16 VECREG:$rA),
                                       (v8i16 VECREG:$rB)))]>;

  def r8 :
    CGTBInst<(outs R8C:$rT), (ins R8C:$rA, R8C:$rB),
             [(set R8C:$rT, (setgt R8C:$rA, R8C:$rB))]>;
}

class CGTBIInst<dag OOL, dag IOL, list<dag> pattern> :
  RI10Form<0b01110010, OOL, IOL, "cgtbi\t$rT, $rA, $val",
           ByteOp, pattern>;

multiclass CmpGtrByteImm
{
  def v16i8 :
    CGTBIInst<(outs VECREG:$rT), (ins VECREG:$rA, s10imm_i8:$val),
              [(set (v16i8 VECREG:$rT), (setgt (v16i8 VECREG:$rA),
                                               v16i8SExt8Imm:$val))]>;
  def r8:
    CGTBIInst<(outs R8C:$rT), (ins R8C:$rA, s10imm_i8:$val),
              [(set R8C:$rT, (setgt R8C:$rA, immSExt8:$val))]>;
}

class CGTHInst<dag OOL, dag IOL, list<dag> pattern> :
  RRForm<0b00010010010, OOL, IOL, "cgth\t$rT, $rA, $rB",
         ByteOp, pattern>;

multiclass CmpGtrHalfword
{
  def v8i16 : CGTHInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
                       [(set (v8i16 VECREG:$rT), (setgt (v8i16 VECREG:$rA),
                                                        (v8i16 VECREG:$rB)))]>;

  def r16 : CGTHInst<(outs R16C:$rT), (ins R16C:$rA, R16C:$rB),
                     [(set R16C:$rT, (setgt R16C:$rA, R16C:$rB))]>;
}

class CGTHIInst<dag OOL, dag IOL, list<dag> pattern> :
  RI10Form<0b10110010, OOL, IOL, "cgthi\t$rT, $rA, $val",
           ByteOp, pattern>;

multiclass CmpGtrHalfwordImm
{
  def v8i16 : CGTHIInst<(outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
                        [(set (v8i16 VECREG:$rT),
                              (setgt (v8i16 VECREG:$rA),
                                     (v8i16 v8i16SExt10Imm:$val)))]>;
  def r16 : CGTHIInst<(outs R16C:$rT), (ins R16C:$rA, s10imm:$val),
                      [(set R16C:$rT, (setgt R16C:$rA, i16ImmSExt10:$val))]>;
}

class CGTInst<dag OOL, dag IOL, list<dag> pattern> :
  RRForm<0b00000010010, OOL, IOL, "cgt\t$rT, $rA, $rB",
         ByteOp, pattern>;

multiclass CmpGtrWord
{
  def v4i32 : CGTInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
                      [(set (v4i32 VECREG:$rT),
                            (setgt (v4i32 VECREG:$rA), (v4i32 VECREG:$rB)))]>;

  def r32 : CGTInst<(outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
                    [(set R32C:$rT, (setgt R32C:$rA, R32C:$rB))]>;
}

class CGTIInst<dag OOL, dag IOL, list<dag> pattern> :
  RI10Form<0b00110010, OOL, IOL, "cgti\t$rT, $rA, $val",
           ByteOp, pattern>;

multiclass CmpGtrWordImm
{
  def v4i32 : CGTIInst<(outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
                       [(set (v4i32 VECREG:$rT),
                             (setgt (v4i32 VECREG:$rA),
                                    (v4i32 v4i32SExt16Imm:$val)))]>;

  def r32: CGTIInst<(outs R32C:$rT), (ins R32C:$rA, s10imm_i32:$val),
                    [(set R32C:$rT, (setgt R32C:$rA, i32ImmSExt10:$val))]>;

  // CGTIv4f32, CGTIf32: These are used in the f32 fdiv instruction sequence:
  def v4f32: CGTIInst<(outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
                       [(set (v4i32 VECREG:$rT),
                             (setgt (v4i32 (bitconvert (v4f32 VECREG:$rA))),
                                    (v4i32 v4i32SExt16Imm:$val)))]>;

  def f32:   CGTIInst<(outs R32C:$rT), (ins R32FP:$rA, s10imm_i32:$val),
                      [/* no pattern */]>;
}

class CLGTBInst<dag OOL, dag IOL, list<dag> pattern> :
  RRForm<0b00001011010, OOL, IOL, "clgtb\t$rT, $rA, $rB",
         ByteOp, pattern>;

multiclass CmpLGtrByte
{
  def v16i8 :
    CLGTBInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      [(set (v16i8 VECREG:$rT), (setugt (v8i16 VECREG:$rA),
                                       (v8i16 VECREG:$rB)))]>;

  def r8 :
    CLGTBInst<(outs R8C:$rT), (ins R8C:$rA, R8C:$rB),
             [(set R8C:$rT, (setugt R8C:$rA, R8C:$rB))]>;
}

class CLGTBIInst<dag OOL, dag IOL, list<dag> pattern> :
  RI10Form<0b01111010, OOL, IOL, "clgtbi\t$rT, $rA, $val",
           ByteOp, pattern>;

multiclass CmpLGtrByteImm
{
  def v16i8 :
    CLGTBIInst<(outs VECREG:$rT), (ins VECREG:$rA, s10imm_i8:$val),
              [(set (v16i8 VECREG:$rT), (setugt (v16i8 VECREG:$rA),
                                               v16i8SExt8Imm:$val))]>;
  def r8:
    CLGTBIInst<(outs R8C:$rT), (ins R8C:$rA, s10imm_i8:$val),
             [(set R8C:$rT, (setugt R8C:$rA, immSExt8:$val))]>;
}

class CLGTHInst<dag OOL, dag IOL, list<dag> pattern> :
  RRForm<0b00010011010, OOL, IOL, "clgth\t$rT, $rA, $rB",
         ByteOp, pattern>;

multiclass CmpLGtrHalfword
{
  def v8i16 : CLGTHInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
                       [(set (v8i16 VECREG:$rT), (setugt (v8i16 VECREG:$rA),
                                                        (v8i16 VECREG:$rB)))]>;

  def r16 : CLGTHInst<(outs R16C:$rT), (ins R16C:$rA, R16C:$rB),
                     [(set R16C:$rT, (setugt R16C:$rA, R16C:$rB))]>;
}

class CLGTHIInst<dag OOL, dag IOL, list<dag> pattern> :
  RI10Form<0b10111010, OOL, IOL, "clgthi\t$rT, $rA, $val",
           ByteOp, pattern>;

multiclass CmpLGtrHalfwordImm
{
  def v8i16 : CLGTHIInst<(outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
                         [(set (v8i16 VECREG:$rT),
                               (setugt (v8i16 VECREG:$rA),
                                       (v8i16 v8i16SExt10Imm:$val)))]>;
  def r16 : CLGTHIInst<(outs R16C:$rT), (ins R16C:$rA, s10imm:$val),
                       [(set R16C:$rT, (setugt R16C:$rA, i16ImmSExt10:$val))]>;
}

class CLGTInst<dag OOL, dag IOL, list<dag> pattern> :
  RRForm<0b00000011010, OOL, IOL, "clgt\t$rT, $rA, $rB",
         ByteOp, pattern>;

multiclass CmpLGtrWord
{
  def v4i32 : CLGTInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
                      [(set (v4i32 VECREG:$rT),
                            (setugt (v4i32 VECREG:$rA), (v4i32 VECREG:$rB)))]>;

  def r32 : CLGTInst<(outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
                     [(set R32C:$rT, (setugt R32C:$rA, R32C:$rB))]>;
}

class CLGTIInst<dag OOL, dag IOL, list<dag> pattern> :
  RI10Form<0b00111010, OOL, IOL, "clgti\t$rT, $rA, $val",
           ByteOp, pattern>;

multiclass CmpLGtrWordImm
{
  def v4i32 : CLGTIInst<(outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
                       [(set (v4i32 VECREG:$rT),
                             (setugt (v4i32 VECREG:$rA),
                                    (v4i32 v4i32SExt16Imm:$val)))]>;

  def r32: CLGTIInst<(outs R32C:$rT), (ins R32C:$rA, s10imm_i32:$val),
                     [(set R32C:$rT, (setugt R32C:$rA, i32ImmSExt10:$val))]>;
}

defm CEQB   : CmpEqualByte;
defm CEQBI  : CmpEqualByteImm;
defm CEQH   : CmpEqualHalfword;
defm CEQHI  : CmpEqualHalfwordImm;
defm CEQ    : CmpEqualWord;
defm CEQI   : CmpEqualWordImm;
defm CGTB   : CmpGtrByte;
defm CGTBI  : CmpGtrByteImm;
defm CGTH   : CmpGtrHalfword;
defm CGTHI  : CmpGtrHalfwordImm;
defm CGT    : CmpGtrWord;
defm CGTI   : CmpGtrWordImm;
defm CLGTB  : CmpLGtrByte;
defm CLGTBI : CmpLGtrByteImm;
defm CLGTH  : CmpLGtrHalfword;
defm CLGTHI : CmpLGtrHalfwordImm;
defm CLGT   : CmpLGtrWord;
defm CLGTI  : CmpLGtrWordImm;

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// For SETCC primitives not supported above (setlt, setle, setge, etc.)
// define a pattern to generate the right code, as a binary operator
// (in a manner of speaking.)
//
// Notes:
// 1. This only matches the setcc set of conditionals. Special pattern
//    matching is used for select conditionals.
//
// 2. The "DAG" versions of these classes is almost exclusively used for
//    i64 comparisons. See the tblgen fundamentals documentation for what
//    ".ResultInstrs[0]" means; see TargetSelectionDAG.td and the Pattern
//    class for where ResultInstrs originates.
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

class SETCCNegCondReg<PatFrag cond, RegisterClass rclass, ValueType inttype,
                      SPUInstr xorinst, SPUInstr cmpare>:
  Pat<(cond rclass:$rA, rclass:$rB),
      (xorinst (cmpare rclass:$rA, rclass:$rB), (inttype -1))>;

class SETCCNegCondImm<PatFrag cond, RegisterClass rclass, ValueType inttype,
                      PatLeaf immpred, SPUInstr xorinst, SPUInstr cmpare>:
  Pat<(cond rclass:$rA, (inttype immpred:$imm)),
      (xorinst (cmpare rclass:$rA, (inttype immpred:$imm)), (inttype -1))>;

def : SETCCNegCondReg<setne, R8C, i8, XORBIr8,  CEQBr8>;
def : SETCCNegCondImm<setne, R8C, i8, immSExt8, XORBIr8, CEQBIr8>;

def : SETCCNegCondReg<setne, R16C, i16, XORHIr16,     CEQHr16>;
def : SETCCNegCondImm<setne, R16C, i16, i16ImmSExt10, XORHIr16, CEQHIr16>;

def : SETCCNegCondReg<setne, R32C, i32, XORIr32, CEQr32>;
def : SETCCNegCondImm<setne, R32C, i32, i32ImmSExt10, XORIr32, CEQIr32>;

class SETCCBinOpReg<PatFrag cond, RegisterClass rclass,
                    SPUInstr binop, SPUInstr cmpOp1, SPUInstr cmpOp2>:
    Pat<(cond rclass:$rA, rclass:$rB),
        (binop (cmpOp1 rclass:$rA, rclass:$rB),
               (cmpOp2 rclass:$rA, rclass:$rB))>;

class SETCCBinOpImm<PatFrag cond, RegisterClass rclass, PatLeaf immpred,
                    ValueType immtype,
                    SPUInstr binop, SPUInstr cmpOp1, SPUInstr cmpOp2>:
    Pat<(cond rclass:$rA, (immtype immpred:$imm)),
        (binop (cmpOp1 rclass:$rA, (immtype immpred:$imm)),
               (cmpOp2 rclass:$rA, (immtype immpred:$imm)))>;

def : SETCCBinOpReg<setge, R8C, ORr8, CGTBr8, CEQBr8>;
def : SETCCBinOpImm<setge, R8C, immSExt8, i8, ORr8, CGTBIr8, CEQBIr8>;
def : SETCCBinOpReg<setlt, R8C, NORr8, CGTBr8, CEQBr8>;
def : SETCCBinOpImm<setlt, R8C, immSExt8, i8, NORr8, CGTBIr8, CEQBIr8>;
def : Pat<(setle R8C:$rA, R8C:$rB),
          (XORBIr8 (CGTBr8 R8C:$rA, R8C:$rB), 0xff)>;
def :  Pat<(setle R8C:$rA, immU8:$imm),
           (XORBIr8 (CGTBIr8 R8C:$rA, immU8:$imm), 0xff)>;

def : SETCCBinOpReg<setge, R16C, ORr16, CGTHr16, CEQHr16>;
def : SETCCBinOpImm<setge, R16C, i16ImmSExt10, i16,
                    ORr16, CGTHIr16, CEQHIr16>;
def : SETCCBinOpReg<setlt, R16C, NORr16, CGTHr16, CEQHr16>;
def : SETCCBinOpImm<setlt, R16C, i16ImmSExt10, i16, NORr16, CGTHIr16, CEQHIr16>;
def : Pat<(setle R16C:$rA, R16C:$rB),
          (XORHIr16 (CGTHr16 R16C:$rA, R16C:$rB), 0xffff)>;
def : Pat<(setle R16C:$rA, i16ImmSExt10:$imm),
          (XORHIr16 (CGTHIr16 R16C:$rA, i16ImmSExt10:$imm), 0xffff)>;

def : SETCCBinOpReg<setge, R32C, ORr32, CGTr32, CEQr32>;
def : SETCCBinOpImm<setge, R32C, i32ImmSExt10, i32,
                    ORr32, CGTIr32, CEQIr32>;
def : SETCCBinOpReg<setlt, R32C, NORr32, CGTr32, CEQr32>;
def : SETCCBinOpImm<setlt, R32C, i32ImmSExt10, i32, NORr32, CGTIr32, CEQIr32>;
def : Pat<(setle R32C:$rA, R32C:$rB),
          (XORIr32 (CGTr32 R32C:$rA, R32C:$rB), 0xffffffff)>;
def : Pat<(setle R32C:$rA, i32ImmSExt10:$imm),
          (XORIr32 (CGTIr32 R32C:$rA, i32ImmSExt10:$imm), 0xffffffff)>;

def : SETCCBinOpReg<setuge, R8C, ORr8, CLGTBr8, CEQBr8>;
def : SETCCBinOpImm<setuge, R8C, immSExt8, i8, ORr8, CLGTBIr8, CEQBIr8>;
def : SETCCBinOpReg<setult, R8C, NORr8, CLGTBr8, CEQBr8>;
def : SETCCBinOpImm<setult, R8C, immSExt8, i8, NORr8, CLGTBIr8, CEQBIr8>;
def : Pat<(setule R8C:$rA, R8C:$rB),
          (XORBIr8 (CLGTBr8 R8C:$rA, R8C:$rB), 0xff)>;
def :  Pat<(setule R8C:$rA, immU8:$imm),
           (XORBIr8 (CLGTBIr8 R8C:$rA, immU8:$imm), 0xff)>;

def : SETCCBinOpReg<setuge, R16C, ORr16, CLGTHr16, CEQHr16>;
def : SETCCBinOpImm<setuge, R16C, i16ImmSExt10, i16,
                    ORr16, CLGTHIr16, CEQHIr16>;
def : SETCCBinOpReg<setult, R16C, NORr16, CLGTHr16, CEQHr16>;
def : SETCCBinOpImm<setult, R16C, i16ImmSExt10, i16, NORr16,
                    CLGTHIr16, CEQHIr16>;
def : Pat<(setule R16C:$rA, R16C:$rB),
          (XORHIr16 (CLGTHr16 R16C:$rA, R16C:$rB), 0xffff)>;
def :  Pat<(setule R16C:$rA, i16ImmSExt10:$imm),
           (XORHIr16 (CLGTHIr16 R16C:$rA, i16ImmSExt10:$imm), 0xffff)>;

def : SETCCBinOpReg<setuge, R32C, ORr32, CLGTr32, CEQr32>;
def : SETCCBinOpImm<setuge, R32C, i32ImmSExt10, i32,
                    ORr32, CLGTIr32, CEQIr32>;
def : SETCCBinOpReg<setult, R32C, NORr32, CLGTr32, CEQr32>;
def : SETCCBinOpImm<setult, R32C, i32ImmSExt10, i32, NORr32, CLGTIr32, CEQIr32>;
def : Pat<(setule R32C:$rA, R32C:$rB),
          (XORIr32 (CLGTr32 R32C:$rA, R32C:$rB), 0xffffffff)>;
def : Pat<(setule R32C:$rA, i32ImmSExt10:$imm),
          (XORIr32 (CLGTIr32 R32C:$rA, i32ImmSExt10:$imm), 0xffffffff)>;

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// select conditional patterns:
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

class SELECTNegCondReg<PatFrag cond, RegisterClass rclass, ValueType inttype,
                       SPUInstr selinstr, SPUInstr cmpare>:
  Pat<(select (inttype (cond rclass:$rA, rclass:$rB)),
              rclass:$rTrue, rclass:$rFalse),
      (selinstr rclass:$rTrue, rclass:$rFalse,
                (cmpare rclass:$rA, rclass:$rB))>;

class SELECTNegCondImm<PatFrag cond, RegisterClass rclass, ValueType inttype,
                       PatLeaf immpred, SPUInstr selinstr, SPUInstr cmpare>:
  Pat<(select (inttype (cond rclass:$rA, immpred:$imm)),
              rclass:$rTrue, rclass:$rFalse),
      (selinstr rclass:$rTrue, rclass:$rFalse,
                (cmpare rclass:$rA, immpred:$imm))>;

def : SELECTNegCondReg<setne, R8C, i8, SELBr8, CEQBr8>;
def : SELECTNegCondImm<setne, R8C, i8, immSExt8, SELBr8, CEQBIr8>;
def : SELECTNegCondReg<setle, R8C, i8, SELBr8, CGTBr8>;
def : SELECTNegCondImm<setle, R8C, i8, immSExt8, SELBr8, CGTBr8>;
def : SELECTNegCondReg<setule, R8C, i8, SELBr8, CLGTBr8>;
def : SELECTNegCondImm<setule, R8C, i8, immU8, SELBr8, CLGTBIr8>;

def : SELECTNegCondReg<setne, R16C, i16, SELBr16, CEQHr16>;
def : SELECTNegCondImm<setne, R16C, i16, i16ImmSExt10, SELBr16, CEQHIr16>;
def : SELECTNegCondReg<setle, R16C, i16, SELBr16, CGTHr16>;
def : SELECTNegCondImm<setle, R16C, i16, i16ImmSExt10, SELBr16, CGTHIr16>;
def : SELECTNegCondReg<setule, R16C, i16, SELBr16, CLGTHr16>;
def : SELECTNegCondImm<setule, R16C, i16, i16ImmSExt10, SELBr16, CLGTHIr16>;

def : SELECTNegCondReg<setne, R32C, i32, SELBr32, CEQr32>;
def : SELECTNegCondImm<setne, R32C, i32, i32ImmSExt10, SELBr32, CEQIr32>;
def : SELECTNegCondReg<setle, R32C, i32, SELBr32, CGTr32>;
def : SELECTNegCondImm<setle, R32C, i32, i32ImmSExt10, SELBr32, CGTIr32>;
def : SELECTNegCondReg<setule, R32C, i32, SELBr32, CLGTr32>;
def : SELECTNegCondImm<setule, R32C, i32, i32ImmSExt10, SELBr32, CLGTIr32>;

class SELECTBinOpReg<PatFrag cond, RegisterClass rclass, ValueType inttype,
                     SPUInstr selinstr, SPUInstr binop, SPUInstr cmpOp1,
                     SPUInstr cmpOp2>:
  Pat<(select (inttype (cond rclass:$rA, rclass:$rB)),
              rclass:$rTrue, rclass:$rFalse),
      (selinstr rclass:$rFalse, rclass:$rTrue,
                (binop (cmpOp1 rclass:$rA, rclass:$rB),
                       (cmpOp2 rclass:$rA, rclass:$rB)))>;

class SELECTBinOpImm<PatFrag cond, RegisterClass rclass, PatLeaf immpred,
                     ValueType inttype,
                     SPUInstr selinstr, SPUInstr binop, SPUInstr cmpOp1,
                     SPUInstr cmpOp2>:
    Pat<(select (inttype (cond rclass:$rA, (inttype immpred:$imm))),
                rclass:$rTrue, rclass:$rFalse),
        (selinstr rclass:$rFalse, rclass:$rTrue,
                  (binop (cmpOp1 rclass:$rA, (inttype immpred:$imm)),
                         (cmpOp2 rclass:$rA, (inttype immpred:$imm))))>;

def : SELECTBinOpReg<setge, R8C, i8, SELBr8, ORr8, CGTBr8, CEQBr8>;
def : SELECTBinOpImm<setge, R8C, immSExt8, i8,
                     SELBr8, ORr8, CGTBIr8, CEQBIr8>;

def : SELECTBinOpReg<setge, R16C, i16, SELBr16, ORr16, CGTHr16, CEQHr16>;
def : SELECTBinOpImm<setge, R16C, i16ImmSExt10, i16,
                     SELBr16, ORr16, CGTHIr16, CEQHIr16>;

def : SELECTBinOpReg<setge, R32C, i32, SELBr32, ORr32, CGTr32, CEQr32>;
def : SELECTBinOpImm<setge, R32C, i32ImmSExt10, i32,
                     SELBr32, ORr32, CGTIr32, CEQIr32>;

def : SELECTBinOpReg<setuge, R8C, i8, SELBr8, ORr8, CLGTBr8, CEQBr8>;
def : SELECTBinOpImm<setuge, R8C, immSExt8, i8,
                     SELBr8, ORr8, CLGTBIr8, CEQBIr8>;

def : SELECTBinOpReg<setuge, R16C, i16, SELBr16, ORr16, CLGTHr16, CEQHr16>;
def : SELECTBinOpImm<setuge, R16C, i16ImmUns10, i16,
                     SELBr16, ORr16, CLGTHIr16, CEQHIr16>;

def : SELECTBinOpReg<setuge, R32C, i32, SELBr32, ORr32, CLGTr32, CEQr32>;
def : SELECTBinOpImm<setuge, R32C, i32ImmUns10, i32,
                     SELBr32, ORr32, CLGTIr32, CEQIr32>;

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

let isCall = 1,
  // All calls clobber the non-callee-saved registers:
  Defs = [R0, R1, R2, R3, R4, R5, R6, R7, R8, R9,
          R10,R11,R12,R13,R14,R15,R16,R17,R18,R19,
          R20,R21,R22,R23,R24,R25,R26,R27,R28,R29,
          R30,R31,R32,R33,R34,R35,R36,R37,R38,R39,
          R40,R41,R42,R43,R44,R45,R46,R47,R48,R49,
          R50,R51,R52,R53,R54,R55,R56,R57,R58,R59,
          R60,R61,R62,R63,R64,R65,R66,R67,R68,R69,
          R70,R71,R72,R73,R74,R75,R76,R77,R78,R79],
  // All of these instructions use $lr (aka $0)
  Uses = [R0]  in {
  // Branch relative and set link: Used if we actually know that the target
  // is within [-32768, 32767] bytes of the target
  def BRSL:
    BranchSetLink<0b011001100, (outs), (ins relcalltarget:$func),
      "brsl\t$$lr, $func",
      [(SPUcall (SPUpcrel tglobaladdr:$func, 0))]>;

  // Branch absolute and set link: Used if we actually know that the target
  // is an absolute address
  def BRASL:
    BranchSetLink<0b011001100, (outs), (ins calltarget:$func),
      "brasl\t$$lr, $func",
      [(SPUcall (SPUaform tglobaladdr:$func, 0))]>;

  // Branch indirect and set link if external data. These instructions are not
  // actually generated, matched by an intrinsic:
  def BISLED_00: BISLEDForm<0b11, "bisled\t$$lr, $func", [/* empty pattern */]>;
  def BISLED_E0: BISLEDForm<0b10, "bisled\t$$lr, $func", [/* empty pattern */]>;
  def BISLED_0D: BISLEDForm<0b01, "bisled\t$$lr, $func", [/* empty pattern */]>;
  def BISLED_ED: BISLEDForm<0b00, "bisled\t$$lr, $func", [/* empty pattern */]>;

  // Branch indirect and set link. This is the "X-form" address version of a
  // function call
  def BISL:
    BIForm<0b10010101100, "bisl\t$$lr, $func", [(SPUcall R32C:$func)]>;
}

// Support calls to external symbols:      
def : Pat<(SPUcall (SPUpcrel texternalsym:$func, 0)),
          (BRSL texternalsym:$func)>;
      
def : Pat<(SPUcall (SPUaform texternalsym:$func, 0)),
          (BRASL texternalsym:$func)>;

// Unconditional branches:
let isBranch = 1, isTerminator = 1, hasCtrlDep = 1 in {
  let isBarrier = 1 in {
    def BR :
      UncondBranch<0b001001100, (outs), (ins brtarget:$dest),
        "br\t$dest",
        [(br bb:$dest)]>;

    // Unconditional, absolute address branch
    def BRA:
      UncondBranch<0b001100000, (outs), (ins brtarget:$dest),
        "bra\t$dest",
        [/* no pattern */]>;

    // Indirect branch
    let isIndirectBranch = 1 in {
      def BI:
        BIForm<0b00010101100, "bi\t$func", [(brind R32C:$func)]>;
    }
  }

  // Conditional branches:
  class BRNZInst<dag IOL, list<dag> pattern>:
    RI16Form<0b010000100, (outs), IOL, "brnz\t$rCond,$dest",
             BranchResolv, pattern>;

  class BRNZRegInst<RegisterClass rclass>:
    BRNZInst<(ins rclass:$rCond, brtarget:$dest),
             [(brcond rclass:$rCond, bb:$dest)]>;

  class BRNZVecInst<ValueType vectype>:
    BRNZInst<(ins VECREG:$rCond, brtarget:$dest),
             [(brcond (vectype VECREG:$rCond), bb:$dest)]>;

  multiclass BranchNotZero {
    def v4i32 : BRNZVecInst<v4i32>;
    def r32   : BRNZRegInst<R32C>;
  }

  defm BRNZ : BranchNotZero;

  class BRZInst<dag IOL, list<dag> pattern>:
    RI16Form<0b000000100, (outs), IOL, "brz\t$rT,$dest",
             BranchResolv, pattern>;

  class BRZRegInst<RegisterClass rclass>:
    BRZInst<(ins rclass:$rT, brtarget:$dest), [/* no pattern */]>;

  class BRZVecInst<ValueType vectype>:
    BRZInst<(ins VECREG:$rT, brtarget:$dest), [/* no pattern */]>;

  multiclass BranchZero {
    def v4i32: BRZVecInst<v4i32>;
    def r32:   BRZRegInst<R32C>;
  }

  defm BRZ: BranchZero;

  // Note: LLVM doesn't do branch conditional, indirect. Otherwise these would
  // be useful:
  /*
  class BINZInst<dag IOL, list<dag> pattern>:
   BICondForm<0b10010100100, (outs), IOL, "binz\t$rA, $dest", pattern>;

  class BINZRegInst<RegisterClass rclass>:
    BINZInst<(ins rclass:$rA, brtarget:$dest),
             [(brcond rclass:$rA, R32C:$dest)]>;

  class BINZVecInst<ValueType vectype>:
    BINZInst<(ins VECREG:$rA, R32C:$dest),
             [(brcond (vectype VECREG:$rA), R32C:$dest)]>;

  multiclass BranchNotZeroIndirect {
    def v4i32: BINZVecInst<v4i32>;
    def r32:   BINZRegInst<R32C>;
  }

  defm BINZ: BranchNotZeroIndirect;

  class BIZInst<dag IOL, list<dag> pattern>:
    BICondForm<0b00010100100, (outs), IOL, "biz\t$rA, $func", pattern>;

  class BIZRegInst<RegisterClass rclass>:
    BIZInst<(ins rclass:$rA, R32C:$func), [/* no pattern */]>;

  class BIZVecInst<ValueType vectype>:
    BIZInst<(ins VECREG:$rA, R32C:$func), [/* no pattern */]>;

  multiclass BranchZeroIndirect {
    def v4i32: BIZVecInst<v4i32>;
    def r32:   BIZRegInst<R32C>;
  }

  defm BIZ: BranchZeroIndirect;
  */

  class BRHNZInst<dag IOL, list<dag> pattern>:
    RI16Form<0b011000100, (outs), IOL, "brhnz\t$rCond,$dest", BranchResolv,
             pattern>;

  class BRHNZRegInst<RegisterClass rclass>:
    BRHNZInst<(ins rclass:$rCond, brtarget:$dest),
              [(brcond rclass:$rCond, bb:$dest)]>;

  class BRHNZVecInst<ValueType vectype>:
    BRHNZInst<(ins VECREG:$rCond, brtarget:$dest), [/* no pattern */]>;

  multiclass BranchNotZeroHalfword {
    def v8i16: BRHNZVecInst<v8i16>;
    def r16:   BRHNZRegInst<R16C>;
  }

  defm BRHNZ: BranchNotZeroHalfword;

  class BRHZInst<dag IOL, list<dag> pattern>:
    RI16Form<0b001000100, (outs), IOL, "brhz\t$rT,$dest", BranchResolv,
             pattern>;

  class BRHZRegInst<RegisterClass rclass>:
    BRHZInst<(ins rclass:$rT, brtarget:$dest), [/* no pattern */]>;

  class BRHZVecInst<ValueType vectype>:
    BRHZInst<(ins VECREG:$rT, brtarget:$dest), [/* no pattern */]>;

  multiclass BranchZeroHalfword {
    def v8i16: BRHZVecInst<v8i16>;
    def r16:   BRHZRegInst<R16C>;
  }

  defm BRHZ: BranchZeroHalfword;
}

//===----------------------------------------------------------------------===//
// setcc and brcond patterns:
//===----------------------------------------------------------------------===//

def : Pat<(brcond (i16 (seteq R16C:$rA, 0)), bb:$dest),
          (BRHZr16 R16C:$rA, bb:$dest)>;
def : Pat<(brcond (i16 (setne R16C:$rA, 0)), bb:$dest),
          (BRHNZr16 R16C:$rA, bb:$dest)>;

def : Pat<(brcond (i32 (seteq R32C:$rA, 0)), bb:$dest),
          (BRZr32 R32C:$rA, bb:$dest)>;
def : Pat<(brcond (i32 (setne R32C:$rA, 0)), bb:$dest),
          (BRNZr32 R32C:$rA, bb:$dest)>;

multiclass BranchCondEQ<PatFrag cond, SPUInstr brinst16, SPUInstr brinst32>
{
  def r16imm: Pat<(brcond (i16 (cond R16C:$rA, i16ImmSExt10:$val)), bb:$dest),
                  (brinst16 (CEQHIr16 R16C:$rA, i16ImmSExt10:$val), bb:$dest)>;

  def r16 : Pat<(brcond (i16 (cond R16C:$rA, R16C:$rB)), bb:$dest),
                (brinst16 (CEQHr16 R16C:$rA, R16:$rB), bb:$dest)>;

  def r32imm : Pat<(brcond (i32 (cond R32C:$rA, i32ImmSExt10:$val)), bb:$dest),
                   (brinst32 (CEQIr32 R32C:$rA, i32ImmSExt10:$val), bb:$dest)>;

  def r32 : Pat<(brcond (i32 (cond R32C:$rA, R32C:$rB)), bb:$dest),
                (brinst32 (CEQr32 R32C:$rA, R32C:$rB), bb:$dest)>;
}

defm BRCONDeq : BranchCondEQ<seteq, BRHNZr16, BRNZr32>;
defm BRCONDne : BranchCondEQ<setne, BRHZr16, BRZr32>;

multiclass BranchCondLGT<PatFrag cond, SPUInstr brinst16, SPUInstr brinst32>
{
  def r16imm : Pat<(brcond (i16 (cond R16C:$rA, i16ImmSExt10:$val)), bb:$dest),
                   (brinst16 (CLGTHIr16 R16C:$rA, i16ImmSExt10:$val), bb:$dest)>;

  def r16 : Pat<(brcond (i16 (cond R16C:$rA, R16C:$rB)), bb:$dest),
                (brinst16 (CLGTHr16 R16C:$rA, R16:$rB), bb:$dest)>;

  def r32imm : Pat<(brcond (i32 (cond R32C:$rA, i32ImmSExt10:$val)), bb:$dest),
                   (brinst32 (CLGTIr32 R32C:$rA, i32ImmSExt10:$val), bb:$dest)>;

  def r32 : Pat<(brcond (i32 (cond R32C:$rA, R32C:$rB)), bb:$dest),
                (brinst32 (CLGTr32 R32C:$rA, R32C:$rB), bb:$dest)>;
}

defm BRCONDugt : BranchCondLGT<setugt, BRHNZr16, BRNZr32>;
defm BRCONDule : BranchCondLGT<setule, BRHZr16, BRZr32>;

multiclass BranchCondLGTEQ<PatFrag cond, SPUInstr orinst16, SPUInstr brinst16,
                           SPUInstr orinst32, SPUInstr brinst32>
{
  def r16imm: Pat<(brcond (i16 (cond R16C:$rA, i16ImmSExt10:$val)), bb:$dest),
                  (brinst16 (orinst16 (CLGTHIr16 R16C:$rA, i16ImmSExt10:$val),
                                      (CEQHIr16 R16C:$rA, i16ImmSExt10:$val)),
                            bb:$dest)>;

  def r16: Pat<(brcond (i16 (cond R16C:$rA, R16C:$rB)), bb:$dest),
               (brinst16 (orinst16 (CLGTHr16 R16C:$rA, R16:$rB),
                                   (CEQHr16 R16C:$rA, R16:$rB)),
                         bb:$dest)>;

  def r32imm : Pat<(brcond (i32 (cond R32C:$rA, i32ImmSExt10:$val)), bb:$dest),
                   (brinst32 (orinst32 (CLGTIr32 R32C:$rA, i32ImmSExt10:$val),
                                       (CEQIr32 R32C:$rA, i32ImmSExt10:$val)),
                             bb:$dest)>;

  def r32 : Pat<(brcond (i32 (cond R32C:$rA, R32C:$rB)), bb:$dest),
                (brinst32 (orinst32 (CLGTr32 R32C:$rA, R32C:$rB),
                                    (CEQr32 R32C:$rA, R32C:$rB)),
                          bb:$dest)>;
}

defm BRCONDuge : BranchCondLGTEQ<setuge, ORr16, BRHNZr16, ORr32, BRNZr32>;
defm BRCONDult : BranchCondLGTEQ<setult, ORr16, BRHZr16, ORr32, BRZr32>;

multiclass BranchCondGT<PatFrag cond, SPUInstr brinst16, SPUInstr brinst32>
{
  def r16imm : Pat<(brcond (i16 (cond R16C:$rA, i16ImmSExt10:$val)), bb:$dest),
                   (brinst16 (CGTHIr16 R16C:$rA, i16ImmSExt10:$val), bb:$dest)>;

  def r16 : Pat<(brcond (i16 (cond R16C:$rA, R16C:$rB)), bb:$dest),
                (brinst16 (CGTHr16 R16C:$rA, R16:$rB), bb:$dest)>;

  def r32imm : Pat<(brcond (i32 (cond R32C:$rA, i32ImmSExt10:$val)), bb:$dest),
                   (brinst32 (CGTIr32 R32C:$rA, i32ImmSExt10:$val), bb:$dest)>;

  def r32 : Pat<(brcond (i32 (cond R32C:$rA, R32C:$rB)), bb:$dest),
                (brinst32 (CGTr32 R32C:$rA, R32C:$rB), bb:$dest)>;
}

defm BRCONDgt : BranchCondGT<setgt, BRHNZr16, BRNZr32>;
defm BRCONDle : BranchCondGT<setle, BRHZr16, BRZr32>;

multiclass BranchCondGTEQ<PatFrag cond, SPUInstr orinst16, SPUInstr brinst16,
                          SPUInstr orinst32, SPUInstr brinst32>
{
  def r16imm: Pat<(brcond (i16 (cond R16C:$rA, i16ImmSExt10:$val)), bb:$dest),
                  (brinst16 (orinst16 (CGTHIr16 R16C:$rA, i16ImmSExt10:$val),
                                      (CEQHIr16 R16C:$rA, i16ImmSExt10:$val)),
                            bb:$dest)>;

  def r16: Pat<(brcond (i16 (cond R16C:$rA, R16C:$rB)), bb:$dest),
               (brinst16 (orinst16 (CGTHr16 R16C:$rA, R16:$rB),
                                   (CEQHr16 R16C:$rA, R16:$rB)),
                         bb:$dest)>;

  def r32imm : Pat<(brcond (i32 (cond R32C:$rA, i32ImmSExt10:$val)), bb:$dest),
                   (brinst32 (orinst32 (CGTIr32 R32C:$rA, i32ImmSExt10:$val),
                                       (CEQIr32 R32C:$rA, i32ImmSExt10:$val)),
                             bb:$dest)>;

  def r32 : Pat<(brcond (i32 (cond R32C:$rA, R32C:$rB)), bb:$dest),
                (brinst32 (orinst32 (CGTr32 R32C:$rA, R32C:$rB),
                                    (CEQr32 R32C:$rA, R32C:$rB)),
                          bb:$dest)>;
}

defm BRCONDge : BranchCondGTEQ<setge, ORr16, BRHNZr16, ORr32, BRNZr32>;
defm BRCONDlt : BranchCondGTEQ<setlt, ORr16, BRHZr16, ORr32, BRZr32>;

let isTerminator = 1, isBarrier = 1 in {
  let isReturn = 1 in {
    def RET:
        RETForm<"bi\t$$lr", [(retflag)]>;
  }
}

//===----------------------------------------------------------------------===//
// Single precision floating point instructions
//===----------------------------------------------------------------------===//

class FAInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm<0b01011000100, OOL, IOL, "fa\t$rT, $rA, $rB",
           SPrecFP, pattern>;

class FAVecInst<ValueType vectype>:
    FAInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
             [(set (vectype VECREG:$rT),
                   (fadd (vectype VECREG:$rA), (vectype VECREG:$rB)))]>;

multiclass SFPAdd
{
  def v4f32: FAVecInst<v4f32>;
  def f32:   FAInst<(outs R32FP:$rT), (ins R32FP:$rA, R32FP:$rB),
                    [(set R32FP:$rT, (fadd R32FP:$rA, R32FP:$rB))]>;
}

defm FA : SFPAdd;

class FSInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm<0b01011000100, OOL, IOL, "fs\t$rT, $rA, $rB",
           SPrecFP, pattern>;

class FSVecInst<ValueType vectype>:
    FSInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
           [(set (vectype VECREG:$rT),
                 (fsub (vectype VECREG:$rA), (vectype VECREG:$rB)))]>;

multiclass SFPSub
{
  def v4f32: FSVecInst<v4f32>;
  def f32:   FSInst<(outs R32FP:$rT), (ins R32FP:$rA, R32FP:$rB),
                    [(set R32FP:$rT, (fsub R32FP:$rA, R32FP:$rB))]>;
}

defm FS : SFPSub;

class FMInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm<0b01100011010, OOL, IOL,
      "fm\t$rT, $rA, $rB", SPrecFP,
      pattern>;

class FMVecInst<ValueType type>:
    FMInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
           [(set (type VECREG:$rT),
                 (fmul (type VECREG:$rA), (type VECREG:$rB)))]>;

multiclass SFPMul
{
  def v4f32: FMVecInst<v4f32>;
  def f32:   FMInst<(outs R32FP:$rT), (ins R32FP:$rA, R32FP:$rB),
                     [(set R32FP:$rT, (fmul R32FP:$rA, R32FP:$rB))]>; 
}

defm FM : SFPMul;

// Floating point multiply and add
// e.g. d = c + (a * b)
def FMAv4f32:
    RRRForm<0b0111, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB, VECREG:$rC),
      "fma\t$rT, $rA, $rB, $rC", SPrecFP,
      [(set (v4f32 VECREG:$rT),
            (fadd (v4f32 VECREG:$rC),
                  (fmul (v4f32 VECREG:$rA), (v4f32 VECREG:$rB))))]>;

def FMAf32:
    RRRForm<0b0111, (outs R32FP:$rT), (ins R32FP:$rA, R32FP:$rB, R32FP:$rC),
      "fma\t$rT, $rA, $rB, $rC", SPrecFP,
      [(set R32FP:$rT, (fadd R32FP:$rC, (fmul R32FP:$rA, R32FP:$rB)))]>;

// FP multiply and subtract
// Subtracts value in rC from product
// res = a * b - c
def FMSv4f32 :
    RRRForm<0b0111, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB, VECREG:$rC),
      "fms\t$rT, $rA, $rB, $rC", SPrecFP,
      [(set (v4f32 VECREG:$rT),
            (fsub (fmul (v4f32 VECREG:$rA), (v4f32 VECREG:$rB)),
                  (v4f32 VECREG:$rC)))]>;

def FMSf32 :
    RRRForm<0b0111, (outs R32FP:$rT), (ins R32FP:$rA, R32FP:$rB, R32FP:$rC),
      "fms\t$rT, $rA, $rB, $rC", SPrecFP,
      [(set R32FP:$rT,
            (fsub (fmul R32FP:$rA, R32FP:$rB), R32FP:$rC))]>;

// Floating Negative Mulitply and Subtract
// Subtracts product from value in rC
// res = fneg(fms a b c)
//     = - (a * b - c)
//     = c - a * b
// NOTE: subtraction order
// fsub a b = a - b
// fs a b = b - a?
def FNMSf32 :
    RRRForm<0b1101, (outs R32FP:$rT), (ins R32FP:$rA, R32FP:$rB, R32FP:$rC),
      "fnms\t$rT, $rA, $rB, $rC", SPrecFP,
      [(set R32FP:$rT, (fsub R32FP:$rC, (fmul R32FP:$rA, R32FP:$rB)))]>;

def FNMSv4f32 :
    RRRForm<0b1101, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB, VECREG:$rC),
      "fnms\t$rT, $rA, $rB, $rC", SPrecFP,
      [(set (v4f32 VECREG:$rT),
            (fsub (v4f32 VECREG:$rC),
                  (fmul (v4f32 VECREG:$rA),
                        (v4f32 VECREG:$rB))))]>;




// Floating point reciprocal estimate

class FRESTInst<dag OOL, dag IOL>:
  RRForm_1<0b00110111000, OOL, IOL,
           "frest\t$rT, $rA", SPrecFP,
           [/* no pattern */]>;

def FRESTv4f32 :
    FRESTInst<(outs VECREG:$rT), (ins VECREG:$rA)>;

def FRESTf32 :
    FRESTInst<(outs R32FP:$rT), (ins R32FP:$rA)>;

// Floating point interpolate (used in conjunction with reciprocal estimate)
def FIv4f32 :
    RRForm<0b00101011110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "fi\t$rT, $rA, $rB", SPrecFP,
      [/* no pattern */]>;

def FIf32 :
    RRForm<0b00101011110, (outs R32FP:$rT), (ins R32FP:$rA, R32FP:$rB),
      "fi\t$rT, $rA, $rB", SPrecFP,
      [/* no pattern */]>;

//--------------------------------------------------------------------------
// Basic single precision floating point comparisons:
//
// Note: There is no support on SPU for single precision NaN. Consequently,
// ordered and unordered comparisons are the same.
//--------------------------------------------------------------------------

def FCEQf32 :
    RRForm<0b01000011110, (outs R32C:$rT), (ins R32FP:$rA, R32FP:$rB),
      "fceq\t$rT, $rA, $rB", SPrecFP,
      [(set R32C:$rT, (setueq R32FP:$rA, R32FP:$rB))]>;

def : Pat<(setoeq R32FP:$rA, R32FP:$rB),
          (FCEQf32 R32FP:$rA, R32FP:$rB)>;

def FCMEQf32 :
    RRForm<0b01010011110, (outs R32C:$rT), (ins R32FP:$rA, R32FP:$rB),
      "fcmeq\t$rT, $rA, $rB", SPrecFP,
      [(set R32C:$rT, (setueq (fabs R32FP:$rA), (fabs R32FP:$rB)))]>;

def : Pat<(setoeq (fabs R32FP:$rA), (fabs R32FP:$rB)),
          (FCMEQf32 R32FP:$rA, R32FP:$rB)>;

def FCGTf32 :
    RRForm<0b01000011010, (outs R32C:$rT), (ins R32FP:$rA, R32FP:$rB),
      "fcgt\t$rT, $rA, $rB", SPrecFP,
      [(set R32C:$rT, (setugt R32FP:$rA, R32FP:$rB))]>;

def : Pat<(setogt R32FP:$rA, R32FP:$rB),
          (FCGTf32 R32FP:$rA, R32FP:$rB)>;

def FCMGTf32 :
    RRForm<0b01010011010, (outs R32C:$rT), (ins R32FP:$rA, R32FP:$rB),
      "fcmgt\t$rT, $rA, $rB", SPrecFP,
      [(set R32C:$rT, (setugt (fabs R32FP:$rA), (fabs R32FP:$rB)))]>;

def : Pat<(setogt (fabs R32FP:$rA), (fabs R32FP:$rB)),
          (FCMGTf32 R32FP:$rA, R32FP:$rB)>;

//--------------------------------------------------------------------------
// Single precision floating point comparisons and SETCC equivalents:
//--------------------------------------------------------------------------

def : SETCCNegCondReg<setune, R32FP, i32, XORIr32, FCEQf32>;
def : SETCCNegCondReg<setone, R32FP, i32, XORIr32, FCEQf32>;

def : SETCCBinOpReg<setuge, R32FP, ORr32, FCGTf32, FCEQf32>;
def : SETCCBinOpReg<setoge, R32FP, ORr32, FCGTf32, FCEQf32>;

def : SETCCBinOpReg<setult, R32FP, NORr32, FCGTf32, FCEQf32>;
def : SETCCBinOpReg<setolt, R32FP, NORr32, FCGTf32, FCEQf32>;

def : Pat<(setule R32FP:$rA, R32FP:$rB),
          (XORIr32 (FCGTf32 R32FP:$rA, R32FP:$rB), 0xffffffff)>;
def : Pat<(setole R32FP:$rA, R32FP:$rB),
          (XORIr32 (FCGTf32 R32FP:$rA, R32FP:$rB), 0xffffffff)>;

// FP Status and Control Register Write
// Why isn't rT a don't care in the ISA?
// Should we create a special RRForm_3 for this guy and zero out the rT?
def FSCRWf32 :
    RRForm_1<0b01011101110, (outs R32FP:$rT), (ins R32FP:$rA),
      "fscrwr\t$rA", SPrecFP,
      [/* This instruction requires an intrinsic. Note: rT is unused. */]>;

// FP Status and Control Register Read
def FSCRRf32 :
    RRForm_2<0b01011101110, (outs R32FP:$rT), (ins),
      "fscrrd\t$rT", SPrecFP,
      [/* This instruction requires an intrinsic */]>;

// llvm instruction space
// How do these map onto cell instructions?
// fdiv rA rB
//   frest rC rB        # c = 1/b (both lines)
//   fi rC rB rC
//   fm rD rA rC        # d = a * 1/b
//   fnms rB rD rB rA # b = - (d * b - a) --should == 0 in a perfect world
//   fma rB rB rC rD            # b = b * c + d
//                              = -(d *b -a) * c + d
//                              = a * c - c ( a *b *c - a)

// fcopysign (???)

// Library calls:
// These llvm instructions will actually map to library calls.
// All that's needed, then, is to check that the appropriate library is
// imported and do a brsl to the proper function name.
// frem # fmod(x, y): x - (x/y) * y
// (Note: fmod(double, double), fmodf(float,float)
// fsqrt?
// fsin?
// fcos?
// Unimplemented SPU instruction space
// floating reciprocal absolute square root estimate (frsqest)

// The following are probably just intrinsics
// status and control register write
// status and control register read

//--------------------------------------
// Floating Point Conversions
// Signed conversions:
def CSiFv4f32:
    CVTIntFPForm<0b0101101110, (outs VECREG:$rT), (ins VECREG:$rA),
      "csflt\t$rT, $rA, 0", SPrecFP,
      [(set (v4f32 VECREG:$rT), (sint_to_fp (v4i32 VECREG:$rA)))]>;

// Convert signed integer to floating point
def CSiFf32 :
    CVTIntFPForm<0b0101101110, (outs R32FP:$rT), (ins R32C:$rA),
      "csflt\t$rT, $rA, 0", SPrecFP,
      [(set R32FP:$rT, (sint_to_fp R32C:$rA))]>;

// Convert unsigned into to float
def CUiFv4f32 :
    CVTIntFPForm<0b1101101110, (outs VECREG:$rT), (ins VECREG:$rA),
      "cuflt\t$rT, $rA, 0", SPrecFP,
      [(set (v4f32 VECREG:$rT), (uint_to_fp (v4i32 VECREG:$rA)))]>;

def CUiFf32 :
    CVTIntFPForm<0b1101101110, (outs R32FP:$rT), (ins R32C:$rA),
      "cuflt\t$rT, $rA, 0", SPrecFP,
      [(set R32FP:$rT, (uint_to_fp R32C:$rA))]>;

// Convert float to unsigned int
// Assume that scale = 0

def CFUiv4f32 :
    CVTIntFPForm<0b1101101110, (outs VECREG:$rT), (ins VECREG:$rA),
      "cfltu\t$rT, $rA, 0", SPrecFP,
      [(set (v4i32 VECREG:$rT), (fp_to_uint (v4f32 VECREG:$rA)))]>;

def CFUif32 :
    CVTIntFPForm<0b1101101110, (outs R32C:$rT), (ins R32FP:$rA),
      "cfltu\t$rT, $rA, 0", SPrecFP,
      [(set R32C:$rT, (fp_to_uint R32FP:$rA))]>;

// Convert float to signed int
// Assume that scale = 0

def CFSiv4f32 :
    CVTIntFPForm<0b1101101110, (outs VECREG:$rT), (ins VECREG:$rA),
      "cflts\t$rT, $rA, 0", SPrecFP,
      [(set (v4i32 VECREG:$rT), (fp_to_sint (v4f32 VECREG:$rA)))]>;

def CFSif32 :
    CVTIntFPForm<0b1101101110, (outs R32C:$rT), (ins R32FP:$rA),
      "cflts\t$rT, $rA, 0", SPrecFP,
      [(set R32C:$rT, (fp_to_sint R32FP:$rA))]>;

//===----------------------------------------------------------------------==//
// Single<->Double precision conversions
//===----------------------------------------------------------------------==//

// NOTE: We use "vec" name suffix here to avoid confusion (e.g. input is a
// v4f32, output is v2f64--which goes in the name?)

// Floating point extend single to double
// NOTE: Not sure if passing in v4f32 to FESDvec is correct since it
// operates on two double-word slots (i.e. 1st and 3rd fp numbers
// are ignored).
def FESDvec :
    RRForm_1<0b00011101110, (outs VECREG:$rT), (ins VECREG:$rA),
      "fesd\t$rT, $rA", SPrecFP,
      [/*(set (v2f64 VECREG:$rT), (fextend (v4f32 VECREG:$rA)))*/]>;

def FESDf32 :
    RRForm_1<0b00011101110, (outs R64FP:$rT), (ins R32FP:$rA),
      "fesd\t$rT, $rA", SPrecFP,
      [(set R64FP:$rT, (fextend R32FP:$rA))]>;

// Floating point round double to single
//def FRDSvec :
//    RRForm_1<0b10011101110, (outs VECREG:$rT), (ins VECREG:$rA),
//      "frds\t$rT, $rA,", SPrecFP,
//      [(set (v4f32 R32FP:$rT), (fround (v2f64 R64FP:$rA)))]>;

def FRDSf64 :
    RRForm_1<0b10011101110, (outs R32FP:$rT), (ins R64FP:$rA),
      "frds\t$rT, $rA", SPrecFP,
      [(set R32FP:$rT, (fround R64FP:$rA))]>;

//ToDo include anyextend?

//===----------------------------------------------------------------------==//
// Double precision floating point instructions
//===----------------------------------------------------------------------==//
def FAf64 :
    RRForm<0b00110011010, (outs R64FP:$rT), (ins R64FP:$rA, R64FP:$rB),
      "dfa\t$rT, $rA, $rB", DPrecFP,
      [(set R64FP:$rT, (fadd R64FP:$rA, R64FP:$rB))]>;

def FAv2f64 :
    RRForm<0b00110011010, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "dfa\t$rT, $rA, $rB", DPrecFP,
      [(set (v2f64 VECREG:$rT), (fadd (v2f64 VECREG:$rA), (v2f64 VECREG:$rB)))]>;

def FSf64 :
    RRForm<0b10100011010, (outs R64FP:$rT), (ins R64FP:$rA, R64FP:$rB),
      "dfs\t$rT, $rA, $rB", DPrecFP,
      [(set R64FP:$rT, (fsub R64FP:$rA, R64FP:$rB))]>;

def FSv2f64 :
    RRForm<0b10100011010, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "dfs\t$rT, $rA, $rB", DPrecFP,
      [(set (v2f64 VECREG:$rT),
            (fsub (v2f64 VECREG:$rA), (v2f64 VECREG:$rB)))]>;

def FMf64 :
    RRForm<0b01100011010, (outs R64FP:$rT), (ins R64FP:$rA, R64FP:$rB),
      "dfm\t$rT, $rA, $rB", DPrecFP,
      [(set R64FP:$rT, (fmul R64FP:$rA, R64FP:$rB))]>;

def FMv2f64:
    RRForm<0b00100011010, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "dfm\t$rT, $rA, $rB", DPrecFP,
      [(set (v2f64 VECREG:$rT),
            (fmul (v2f64 VECREG:$rA), (v2f64 VECREG:$rB)))]>;

def FMAf64:
    RRForm<0b00111010110, (outs R64FP:$rT),
                          (ins R64FP:$rA, R64FP:$rB, R64FP:$rC),
      "dfma\t$rT, $rA, $rB", DPrecFP,
      [(set R64FP:$rT, (fadd R64FP:$rC, (fmul R64FP:$rA, R64FP:$rB)))]>,
    RegConstraint<"$rC = $rT">,
    NoEncode<"$rC">;

def FMAv2f64:
    RRForm<0b00111010110, (outs VECREG:$rT),
                          (ins VECREG:$rA, VECREG:$rB, VECREG:$rC),
      "dfma\t$rT, $rA, $rB", DPrecFP,
      [(set (v2f64 VECREG:$rT),
            (fadd (v2f64 VECREG:$rC),
                  (fmul (v2f64 VECREG:$rA), (v2f64 VECREG:$rB))))]>,
    RegConstraint<"$rC = $rT">,
    NoEncode<"$rC">;

def FMSf64 :
    RRForm<0b10111010110, (outs R64FP:$rT),
                          (ins R64FP:$rA, R64FP:$rB, R64FP:$rC),
      "dfms\t$rT, $rA, $rB", DPrecFP,
      [(set R64FP:$rT, (fsub (fmul R64FP:$rA, R64FP:$rB), R64FP:$rC))]>,
    RegConstraint<"$rC = $rT">,
    NoEncode<"$rC">;

def FMSv2f64 :
    RRForm<0b10111010110, (outs VECREG:$rT),
                          (ins VECREG:$rA, VECREG:$rB, VECREG:$rC),
      "dfms\t$rT, $rA, $rB", DPrecFP,
      [(set (v2f64 VECREG:$rT),
            (fsub (fmul (v2f64 VECREG:$rA), (v2f64 VECREG:$rB)),
                  (v2f64 VECREG:$rC)))]>;

// DFNMS: - (a * b - c)
// - (a * b) + c => c - (a * b)

class DFNMSInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm<0b01111010110, OOL, IOL, "dfnms\t$rT, $rA, $rB",
           DPrecFP, pattern>,
    RegConstraint<"$rC = $rT">,
    NoEncode<"$rC">;

class DFNMSVecInst<list<dag> pattern>:
    DFNMSInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB, VECREG:$rC),
              pattern>;

class DFNMSRegInst<list<dag> pattern>:
    DFNMSInst<(outs R64FP:$rT), (ins R64FP:$rA, R64FP:$rB, R64FP:$rC),
             pattern>;

multiclass DFMultiplySubtract
{
  def v2f64 : DFNMSVecInst<[(set (v2f64 VECREG:$rT), 
                                 (fsub (v2f64 VECREG:$rC),
                                       (fmul (v2f64 VECREG:$rA),
                                             (v2f64 VECREG:$rB))))]>;

  def f64 : DFNMSRegInst<[(set R64FP:$rT,
                               (fsub R64FP:$rC,
                                     (fmul R64FP:$rA, R64FP:$rB)))]>;
}

defm DFNMS : DFMultiplySubtract;

// - (a * b + c)
// - (a * b) - c
def FNMAf64 :
    RRForm<0b11111010110, (outs R64FP:$rT),
                          (ins R64FP:$rA, R64FP:$rB, R64FP:$rC),
      "dfnma\t$rT, $rA, $rB", DPrecFP,
      [(set R64FP:$rT, (fneg (fadd R64FP:$rC, (fmul R64FP:$rA, R64FP:$rB))))]>,
    RegConstraint<"$rC = $rT">,
    NoEncode<"$rC">;

def FNMAv2f64 :
    RRForm<0b11111010110, (outs VECREG:$rT),
                          (ins VECREG:$rA, VECREG:$rB, VECREG:$rC),
      "dfnma\t$rT, $rA, $rB", DPrecFP,
      [(set (v2f64 VECREG:$rT),
            (fneg (fadd (v2f64 VECREG:$rC),
                        (fmul (v2f64 VECREG:$rA),
                              (v2f64 VECREG:$rB)))))]>,
    RegConstraint<"$rC = $rT">,
    NoEncode<"$rC">;

//===----------------------------------------------------------------------==//
// Floating point negation and absolute value
//===----------------------------------------------------------------------==//

def : Pat<(fneg (v4f32 VECREG:$rA)),
          (XORfnegvec (v4f32 VECREG:$rA),
                      (v4f32 (ILHUv4i32 0x8000)))>;

def : Pat<(fneg R32FP:$rA),
          (XORfneg32 R32FP:$rA, (ILHUr32 0x8000))>;

// Floating point absolute value
// Note: f64 fabs is custom-selected.

def : Pat<(fabs R32FP:$rA),
          (ANDfabs32 R32FP:$rA, (IOHLr32 (ILHUr32 0x7fff), 0xffff))>;

def : Pat<(fabs (v4f32 VECREG:$rA)),
          (ANDfabsvec (v4f32 VECREG:$rA),
                      (IOHLv4i32 (ILHUv4i32 0x7fff), 0xffff))>;

//===----------------------------------------------------------------------===//
// Hint for branch instructions:
//===----------------------------------------------------------------------===//
def HBRA :
    HBI16Form<0b0001001,(ins hbrtarget:$brinst, brtarget:$btarg), "hbra\t$brinst, $btarg">;

//===----------------------------------------------------------------------===//
// Execution, Load NOP (execute NOPs belong in even pipeline, load NOPs belong
// in the odd pipeline)
//===----------------------------------------------------------------------===//

def ENOP : SPUInstr<(outs), (ins), "nop", ExecNOP> {
  let Pattern = [];

  let Inst{0-10} = 0b10000000010;
  let Inst{11-17} = 0;
  let Inst{18-24} = 0;
  let Inst{25-31} = 0;
}

def LNOP : SPUInstr<(outs), (ins), "lnop", LoadNOP> {
  let Pattern = [];

  let Inst{0-10} = 0b10000000000;
  let Inst{11-17} = 0;
  let Inst{18-24} = 0;
  let Inst{25-31} = 0;
}

//===----------------------------------------------------------------------===//
// Bit conversions (type conversions between vector/packed types)
// NOTE: Promotions are handled using the XS* instructions.
//===----------------------------------------------------------------------===//
def : Pat<(v16i8 (bitconvert (v8i16 VECREG:$src))), (v16i8 VECREG:$src)>;
def : Pat<(v16i8 (bitconvert (v4i32 VECREG:$src))), (v16i8 VECREG:$src)>;
def : Pat<(v16i8 (bitconvert (v2i64 VECREG:$src))), (v16i8 VECREG:$src)>;
def : Pat<(v16i8 (bitconvert (v4f32 VECREG:$src))), (v16i8 VECREG:$src)>;
def : Pat<(v16i8 (bitconvert (v2f64 VECREG:$src))), (v16i8 VECREG:$src)>;

def : Pat<(v8i16 (bitconvert (v16i8 VECREG:$src))), (v8i16 VECREG:$src)>;
def : Pat<(v8i16 (bitconvert (v4i32 VECREG:$src))), (v8i16 VECREG:$src)>;
def : Pat<(v8i16 (bitconvert (v2i64 VECREG:$src))), (v8i16 VECREG:$src)>;
def : Pat<(v8i16 (bitconvert (v4f32 VECREG:$src))), (v8i16 VECREG:$src)>;
def : Pat<(v8i16 (bitconvert (v2f64 VECREG:$src))), (v8i16 VECREG:$src)>;

def : Pat<(v4i32 (bitconvert (v16i8 VECREG:$src))), (v4i32 VECREG:$src)>;
def : Pat<(v4i32 (bitconvert (v8i16 VECREG:$src))), (v4i32 VECREG:$src)>;
def : Pat<(v4i32 (bitconvert (v2i64 VECREG:$src))), (v4i32 VECREG:$src)>;
def : Pat<(v4i32 (bitconvert (v4f32 VECREG:$src))), (v4i32 VECREG:$src)>;
def : Pat<(v4i32 (bitconvert (v2f64 VECREG:$src))), (v4i32 VECREG:$src)>;

def : Pat<(v2i64 (bitconvert (v16i8 VECREG:$src))), (v2i64 VECREG:$src)>;
def : Pat<(v2i64 (bitconvert (v8i16 VECREG:$src))), (v2i64 VECREG:$src)>;
def : Pat<(v2i64 (bitconvert (v4i32 VECREG:$src))), (v2i64 VECREG:$src)>;
def : Pat<(v2i64 (bitconvert (v4f32 VECREG:$src))), (v2i64 VECREG:$src)>;
def : Pat<(v2i64 (bitconvert (v2f64 VECREG:$src))), (v2i64 VECREG:$src)>;

def : Pat<(v4f32 (bitconvert (v16i8 VECREG:$src))), (v4f32 VECREG:$src)>;
def : Pat<(v4f32 (bitconvert (v8i16 VECREG:$src))), (v4f32 VECREG:$src)>;
def : Pat<(v4f32 (bitconvert (v2i64 VECREG:$src))), (v4f32 VECREG:$src)>;
def : Pat<(v4f32 (bitconvert (v4i32 VECREG:$src))), (v4f32 VECREG:$src)>;
def : Pat<(v4f32 (bitconvert (v2f64 VECREG:$src))), (v4f32 VECREG:$src)>;

def : Pat<(v2f64 (bitconvert (v16i8 VECREG:$src))), (v2f64 VECREG:$src)>;
def : Pat<(v2f64 (bitconvert (v8i16 VECREG:$src))), (v2f64 VECREG:$src)>;
def : Pat<(v2f64 (bitconvert (v4i32 VECREG:$src))), (v2f64 VECREG:$src)>;
def : Pat<(v2f64 (bitconvert (v2i64 VECREG:$src))), (v2f64 VECREG:$src)>;
def : Pat<(v2f64 (bitconvert (v4f32 VECREG:$src))), (v2f64 VECREG:$src)>;

def : Pat<(i128 (bitconvert (v16i8 VECREG:$src))),
          (COPY_TO_REGCLASS VECREG:$src, GPRC)>;
def : Pat<(i128 (bitconvert (v8i16 VECREG:$src))),
          (COPY_TO_REGCLASS VECREG:$src, GPRC)>;
def : Pat<(i128 (bitconvert (v4i32 VECREG:$src))),
          (COPY_TO_REGCLASS VECREG:$src, GPRC)>;
def : Pat<(i128 (bitconvert (v2i64 VECREG:$src))),
          (COPY_TO_REGCLASS VECREG:$src, GPRC)>;
def : Pat<(i128 (bitconvert (v4f32 VECREG:$src))),
          (COPY_TO_REGCLASS VECREG:$src, GPRC)>;
def : Pat<(i128 (bitconvert (v2f64 VECREG:$src))),
          (COPY_TO_REGCLASS VECREG:$src, GPRC)>;

def : Pat<(v16i8 (bitconvert (i128 GPRC:$src))),
          (v16i8 (COPY_TO_REGCLASS GPRC:$src, VECREG))>;
def : Pat<(v8i16 (bitconvert (i128 GPRC:$src))),
          (v8i16 (COPY_TO_REGCLASS GPRC:$src, VECREG))>;
def : Pat<(v4i32 (bitconvert (i128 GPRC:$src))),
          (v4i32 (COPY_TO_REGCLASS GPRC:$src, VECREG))>;
def : Pat<(v2i64 (bitconvert (i128 GPRC:$src))),
          (v2i64 (COPY_TO_REGCLASS GPRC:$src, VECREG))>;
def : Pat<(v4f32 (bitconvert (i128 GPRC:$src))),
          (v4f32 (COPY_TO_REGCLASS GPRC:$src, VECREG))>;
def : Pat<(v2f64 (bitconvert (i128 GPRC:$src))),
          (v2f64 (COPY_TO_REGCLASS GPRC:$src, VECREG))>;

def : Pat<(i32 (bitconvert R32FP:$rA)),
          (COPY_TO_REGCLASS R32FP:$rA, R32C)>;

def : Pat<(f32 (bitconvert R32C:$rA)),
          (COPY_TO_REGCLASS R32C:$rA, R32FP)>;

def : Pat<(i64 (bitconvert R64FP:$rA)),
          (COPY_TO_REGCLASS R64FP:$rA, R64C)>;

def : Pat<(f64 (bitconvert R64C:$rA)),
          (COPY_TO_REGCLASS R64C:$rA, R64FP)>;


//===----------------------------------------------------------------------===//
// Instruction patterns:
//===----------------------------------------------------------------------===//

// General 32-bit constants:
def : Pat<(i32 imm:$imm),
          (IOHLr32 (ILHUr32 (HI16 imm:$imm)), (LO16 imm:$imm))>;

// Single precision float constants:
def : Pat<(f32 fpimm:$imm),
          (IOHLf32 (ILHUf32 (HI16_f32 fpimm:$imm)), (LO16_f32 fpimm:$imm))>;

// General constant 32-bit vectors
def : Pat<(v4i32 v4i32Imm:$imm),
          (IOHLv4i32 (v4i32 (ILHUv4i32 (HI16_vec v4i32Imm:$imm))),
                     (LO16_vec v4i32Imm:$imm))>;

// 8-bit constants
def : Pat<(i8 imm:$imm),
          (ILHr8 imm:$imm)>;

//===----------------------------------------------------------------------===//
// Zero/Any/Sign extensions
//===----------------------------------------------------------------------===//

// sext 8->32: Sign extend bytes to words
def : Pat<(sext_inreg R32C:$rSrc, i8),
          (XSHWr32 (XSBHr32 R32C:$rSrc))>;

def : Pat<(i32 (sext R8C:$rSrc)),
          (XSHWr16 (XSBHr8 R8C:$rSrc))>;

// sext 8->64: Sign extend bytes to double word
def : Pat<(sext_inreg R64C:$rSrc, i8),
          (XSWDr64_inreg (XSHWr64 (XSBHr64 R64C:$rSrc)))>;
          
def : Pat<(i64 (sext R8C:$rSrc)),
          (XSWDr64 (XSHWr16 (XSBHr8 R8C:$rSrc)))>;

// zext 8->16: Zero extend bytes to halfwords
def : Pat<(i16 (zext R8C:$rSrc)),
          (ANDHIi8i16 R8C:$rSrc, 0xff)>;

// zext 8->32: Zero extend bytes to words
def : Pat<(i32 (zext R8C:$rSrc)),
          (ANDIi8i32 R8C:$rSrc, 0xff)>;

// zext 8->64: Zero extend bytes to double words
def : Pat<(i64 (zext R8C:$rSrc)),
          (COPY_TO_REGCLASS (SELBv4i32 (ROTQMBYv4i32
                                    (COPY_TO_REGCLASS 
                                       (ANDIi8i32 R8C:$rSrc,0xff), VECREG),
                                    0x4),
                                  (ILv4i32 0x0),
                                  (FSMBIv4i32 0x0f0f)), R64C)>;

// anyext 8->16: Extend 8->16 bits, irrespective of sign, preserves high bits
def : Pat<(i16 (anyext R8C:$rSrc)),
          (ORHIi8i16 R8C:$rSrc, 0)>;

// anyext 8->32: Extend 8->32 bits, irrespective of sign, preserves high bits
def : Pat<(i32 (anyext R8C:$rSrc)),
          (COPY_TO_REGCLASS R8C:$rSrc, R32C)>;

// sext 16->64: Sign extend halfword to double word
def : Pat<(sext_inreg R64C:$rSrc, i16),
          (XSWDr64_inreg (XSHWr64 R64C:$rSrc))>;
          
def : Pat<(sext R16C:$rSrc),
          (XSWDr64 (XSHWr16 R16C:$rSrc))>;

// zext 16->32: Zero extend halfwords to words
def : Pat<(i32 (zext R16C:$rSrc)),
          (ANDi16i32 R16C:$rSrc, (ILAr32 0xffff))>;

def : Pat<(i32 (zext (and R16C:$rSrc, 0xf))),
          (ANDIi16i32 R16C:$rSrc, 0xf)>;

def : Pat<(i32 (zext (and R16C:$rSrc, 0xff))),
          (ANDIi16i32 R16C:$rSrc, 0xff)>;

def : Pat<(i32 (zext (and R16C:$rSrc, 0xfff))),
          (ANDIi16i32 R16C:$rSrc, 0xfff)>;

// anyext 16->32: Extend 16->32 bits, irrespective of sign
def : Pat<(i32 (anyext R16C:$rSrc)),
          (COPY_TO_REGCLASS R16C:$rSrc, R32C)>;

//===----------------------------------------------------------------------===//
// Truncates:
// These truncates are for the SPU's supported types (i8, i16, i32). i64 and
// above are custom lowered.
//===----------------------------------------------------------------------===//

def : Pat<(i8 (trunc GPRC:$src)),
          (COPY_TO_REGCLASS
            (SHUFBgprc GPRC:$src, GPRC:$src,
                       (IOHLv4i32 (ILHUv4i32 0x0f0f), 0x0f0f)), R8C)>;

def : Pat<(i8 (trunc R64C:$src)),
          (COPY_TO_REGCLASS
            (SHUFBv2i64_m32
              (COPY_TO_REGCLASS R64C:$src, VECREG),
              (COPY_TO_REGCLASS R64C:$src, VECREG),
              (IOHLv4i32 (ILHUv4i32 0x0707), 0x0707)), R8C)>;

def : Pat<(i8 (trunc R32C:$src)),
          (COPY_TO_REGCLASS
            (SHUFBv4i32_m32
               (COPY_TO_REGCLASS R32C:$src, VECREG),
               (COPY_TO_REGCLASS R32C:$src, VECREG),
               (IOHLv4i32 (ILHUv4i32 0x0303), 0x0303)), R8C)>;

def : Pat<(i8 (trunc R16C:$src)),
          (COPY_TO_REGCLASS
            (SHUFBv4i32_m32
               (COPY_TO_REGCLASS R16C:$src, VECREG),
               (COPY_TO_REGCLASS R16C:$src, VECREG),
               (IOHLv4i32 (ILHUv4i32 0x0303), 0x0303)), R8C)>;

def : Pat<(i16 (trunc GPRC:$src)),
          (COPY_TO_REGCLASS
            (SHUFBgprc GPRC:$src, GPRC:$src,
                       (IOHLv4i32 (ILHUv4i32 0x0e0f), 0x0e0f)), R16C)>;

def : Pat<(i16 (trunc R64C:$src)),
          (COPY_TO_REGCLASS
            (SHUFBv2i64_m32
              (COPY_TO_REGCLASS R64C:$src, VECREG),
              (COPY_TO_REGCLASS R64C:$src, VECREG),
              (IOHLv4i32 (ILHUv4i32 0x0607), 0x0607)), R16C)>;

def : Pat<(i16 (trunc R32C:$src)),
          (COPY_TO_REGCLASS
            (SHUFBv4i32_m32
               (COPY_TO_REGCLASS R32C:$src, VECREG),
               (COPY_TO_REGCLASS R32C:$src, VECREG),
               (IOHLv4i32 (ILHUv4i32 0x0203), 0x0203)), R16C)>;

def : Pat<(i32 (trunc GPRC:$src)),
          (COPY_TO_REGCLASS
            (SHUFBgprc GPRC:$src, GPRC:$src,
                       (IOHLv4i32 (ILHUv4i32 0x0c0d), 0x0e0f)), R32C)>;

def : Pat<(i32 (trunc R64C:$src)),
          (COPY_TO_REGCLASS
            (SHUFBv2i64_m32
              (COPY_TO_REGCLASS R64C:$src, VECREG),
              (COPY_TO_REGCLASS R64C:$src, VECREG),
              (IOHLv4i32 (ILHUv4i32 0x0405), 0x0607)), R32C)>;

//===----------------------------------------------------------------------===//
// Address generation: SPU, like PPC, has to split addresses into high and
// low parts in order to load them into a register.
//===----------------------------------------------------------------------===//

def : Pat<(SPUaform tglobaladdr:$in, 0),  (ILAlsa tglobaladdr:$in)>;
def : Pat<(SPUaform texternalsym:$in, 0), (ILAlsa texternalsym:$in)>;
def : Pat<(SPUaform tjumptable:$in, 0),   (ILAlsa tjumptable:$in)>;
def : Pat<(SPUaform tconstpool:$in, 0),   (ILAlsa  tconstpool:$in)>;

def : Pat<(SPUindirect (SPUhi tglobaladdr:$in, 0),
                       (SPUlo tglobaladdr:$in, 0)),
          (IOHLlo (ILHUhi tglobaladdr:$in), tglobaladdr:$in)>;

def : Pat<(SPUindirect (SPUhi texternalsym:$in, 0),
                       (SPUlo texternalsym:$in, 0)),
          (IOHLlo (ILHUhi texternalsym:$in), texternalsym:$in)>;

def : Pat<(SPUindirect (SPUhi tjumptable:$in, 0),
                       (SPUlo tjumptable:$in, 0)),
          (IOHLlo (ILHUhi tjumptable:$in), tjumptable:$in)>;

def : Pat<(SPUindirect (SPUhi tconstpool:$in, 0),
                       (SPUlo tconstpool:$in, 0)),
          (IOHLlo (ILHUhi tconstpool:$in), tconstpool:$in)>;

def : Pat<(add (SPUhi tglobaladdr:$in, 0), (SPUlo tglobaladdr:$in, 0)),
          (IOHLlo (ILHUhi tglobaladdr:$in), tglobaladdr:$in)>;

def : Pat<(add (SPUhi texternalsym:$in, 0), (SPUlo texternalsym:$in, 0)),
          (IOHLlo (ILHUhi texternalsym:$in), texternalsym:$in)>;

def : Pat<(add (SPUhi tjumptable:$in, 0), (SPUlo tjumptable:$in, 0)),
          (IOHLlo (ILHUhi tjumptable:$in), tjumptable:$in)>;

def : Pat<(add (SPUhi tconstpool:$in, 0), (SPUlo tconstpool:$in, 0)),
          (IOHLlo (ILHUhi tconstpool:$in), tconstpool:$in)>;

// Intrinsics:
include "CellSDKIntrinsics.td"
// Various math operator instruction sequences
include "SPUMathInstr.td"
// 64-bit "instructions"/support
include "SPU64InstrInfo.td"
// 128-bit "instructions"/support
include "SPU128InstrInfo.td"
