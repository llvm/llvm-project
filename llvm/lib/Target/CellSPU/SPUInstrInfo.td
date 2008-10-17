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
}

//===----------------------------------------------------------------------===//
// DWARF debugging Pseudo Instructions
//===----------------------------------------------------------------------===//

def DWARF_LOC        : Pseudo<(outs), (ins i32imm:$line, i32imm:$col, i32imm:$file),
                              "${:comment} .loc $file, $line, $col",
                      [(dwarf_loc (i32 imm:$line), (i32 imm:$col),
                                  (i32 imm:$file))]>;

//===----------------------------------------------------------------------===//
// Loads:
// NB: The ordering is actually important, since the instruction selection
// will try each of the instructions in sequence, i.e., the D-form first with
// the 10-bit displacement, then the A-form with the 16 bit displacement, and
// finally the X-form with the register-register.
//===----------------------------------------------------------------------===//

let isSimpleLoad = 1 in {
  class LoadDFormVec<ValueType vectype>
    : RI10Form<0b00101100, (outs VECREG:$rT), (ins memri10:$src),
               "lqd\t$rT, $src",
               LoadStore,
               [(set (vectype VECREG:$rT), (load dform_addr:$src))]>
  { }

  class LoadDForm<RegisterClass rclass>
    : RI10Form<0b00101100, (outs rclass:$rT), (ins memri10:$src),
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
  : RI10Form<0b00100100, (outs), (ins VECREG:$rT, memri10:$src),
             "stqd\t$rT, $src",
             LoadStore,
             [(store (vectype VECREG:$rT), dform_addr:$src)]>
{ }

class StoreDForm<RegisterClass rclass>
  : RI10Form<0b00100100, (outs), (ins rclass:$rT, memri10:$src),
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

def CBD :
    RI7Form<0b10101111100, (outs VECREG:$rT), (ins memri7:$src),
      "cbd\t$rT, $src", ShuffleOp,
      [(set (v16i8 VECREG:$rT), (SPUvecinsmask dform2_addr:$src))]>;

def CBX : RRForm<0b00101011100, (outs VECREG:$rT), (ins memrr:$src),
    "cbx\t$rT, $src", ShuffleOp,
    [(set (v16i8 VECREG:$rT), (SPUvecinsmask xform_addr:$src))]>;

def CHD : RI7Form<0b10101111100, (outs VECREG:$rT), (ins memri7:$src),
    "chd\t$rT, $src", ShuffleOp,
    [(set (v8i16 VECREG:$rT), (SPUvecinsmask dform2_addr:$src))]>;

def CHX : RRForm<0b10101011100, (outs VECREG:$rT), (ins memrr:$src),
    "chx\t$rT, $src", ShuffleOp,
    [(set (v8i16 VECREG:$rT), (SPUvecinsmask xform_addr:$src))]>;

def CWD : RI7Form<0b01101111100, (outs VECREG:$rT), (ins memri7:$src),
    "cwd\t$rT, $src", ShuffleOp,
    [(set (v4i32 VECREG:$rT), (SPUvecinsmask dform2_addr:$src))]>;

def CWDf32 : RI7Form<0b01101111100, (outs VECREG:$rT), (ins memri7:$src),
    "cwd\t$rT, $src", ShuffleOp,
    [(set (v4f32 VECREG:$rT), (SPUvecinsmask dform2_addr:$src))]>;

def CWX : RRForm<0b01101011100, (outs VECREG:$rT), (ins memrr:$src),
    "cwx\t$rT, $src", ShuffleOp,
    [(set (v4i32 VECREG:$rT), (SPUvecinsmask xform_addr:$src))]>;

def CWXf32 : RRForm<0b01101011100, (outs VECREG:$rT), (ins memrr:$src),
    "cwx\t$rT, $src", ShuffleOp,
    [(set (v4f32 VECREG:$rT), (SPUvecinsmask xform_addr:$src))]>;

def CDD : RI7Form<0b11101111100, (outs VECREG:$rT), (ins memri7:$src),
    "cdd\t$rT, $src", ShuffleOp,
    [(set (v2i64 VECREG:$rT), (SPUvecinsmask dform2_addr:$src))]>;

def CDDf64 : RI7Form<0b11101111100, (outs VECREG:$rT), (ins memri7:$src),
    "cdd\t$rT, $src", ShuffleOp,
    [(set (v2f64 VECREG:$rT), (SPUvecinsmask dform2_addr:$src))]>;

def CDX : RRForm<0b11101011100, (outs VECREG:$rT), (ins memrr:$src),
    "cdx\t$rT, $src", ShuffleOp,
    [(set (v2i64 VECREG:$rT), (SPUvecinsmask xform_addr:$src))]>;

def CDXf64 : RRForm<0b11101011100, (outs VECREG:$rT), (ins memrr:$src),
    "cdx\t$rT, $src", ShuffleOp,
    [(set (v2f64 VECREG:$rT), (SPUvecinsmask xform_addr:$src))]>;

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

  def lo: ILARegInst<R32C, symbolLo, imm18>;

  def lsa: ILAInst<(outs R32C:$rT), (ins symbolLSA:$val),
                   [/* no pattern */]>;
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
def FSMB:
    RRForm_1<0b01101101100, (outs VECREG:$rT), (ins R16C:$rA),
             "fsmb\t$rT, $rA", SelectOp,
             [(set (v16i8 VECREG:$rT), (SPUselmask R16C:$rA))]>;

// fsmh: Form select mask for halfwords. N.B., Input operand, $rA, is
// only 8-bits wide (even though it's input as 16-bits here)
def FSMH:
    RRForm_1<0b10101101100, (outs VECREG:$rT), (ins R16C:$rA),
      "fsmh\t$rT, $rA", SelectOp,
      [(set (v8i16 VECREG:$rT), (SPUselmask R16C:$rA))]>;

// fsm: Form select mask for words. Like the other fsm* instructions,
// only the lower 4 bits of $rA are significant.
class FSMInst<ValueType vectype, RegisterClass rclass>:
    RRForm_1<0b00101101100, (outs VECREG:$rT), (ins rclass:$rA),
      "fsm\t$rT, $rA",
      SelectOp,
      [(set (vectype VECREG:$rT), (SPUselmask rclass:$rA))]>;

multiclass FormSelectMaskWord {
  def r32 : FSMInst<v4i32, R32C>;
  def r16 : FSMInst<v4i32, R16C>;
}

defm FSM : FormSelectMaskWord;

// Special case when used for i64 math operations
multiclass FormSelectMaskWord64 {
  def r32 : FSMInst<v2i64, R32C>;
  def r16 : FSMInst<v2i64, R16C>;
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
    [(set R16C:$rT, (add R16C:$rA, v8i16SExt10Imm:$val))]>;

def Avec:
  RRForm<0b00000011000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
    "a\t$rT, $rA, $rB", IntegerOp,
    [(set (v4i32 VECREG:$rT), (add (v4i32 VECREG:$rA), (v4i32 VECREG:$rB)))]>;

def : Pat<(add (v16i8 VECREG:$rA), (v16i8 VECREG:$rB)),
          (Avec VECREG:$rA, VECREG:$rB)>;

def Ar32:
  RRForm<0b00000011000, (outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
    "a\t$rT, $rA, $rB", IntegerOp,
    [(set R32C:$rT, (add R32C:$rA, R32C:$rB))]>;

def Ar8:
    RRForm<0b00000011000, (outs R8C:$rT), (ins R8C:$rA, R8C:$rB),
      "a\t$rT, $rA, $rB", IntegerOp,
      [/* no pattern */]>;

def AIvec:
    RI10Form<0b00111000, (outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
      "ai\t$rT, $rA, $val", IntegerOp,
      [(set (v4i32 VECREG:$rT), (add (v4i32 VECREG:$rA),
                                      v4i32SExt10Imm:$val))]>;

def AIr32:
    RI10Form<0b00111000, (outs R32C:$rT), (ins R32C:$rA, s10imm_i32:$val),
      "ai\t$rT, $rA, $val", IntegerOp,
      [(set R32C:$rT, (add R32C:$rA, i32ImmSExt10:$val))]>;

def SFHvec:
    RRForm<0b00010010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "sfh\t$rT, $rA, $rB", IntegerOp,
      [(set (v8i16 VECREG:$rT), (sub (v8i16 VECREG:$rA),
                                     (v8i16 VECREG:$rB)))]>;

def SFHr16:
    RRForm<0b00010010000, (outs R16C:$rT), (ins R16C:$rA, R16C:$rB),
      "sfh\t$rT, $rA, $rB", IntegerOp,
      [(set R16C:$rT, (sub R16C:$rA, R16C:$rB))]>;

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
  [(set (v4i32 VECREG:$rT), (sub (v4i32 VECREG:$rA), (v4i32 VECREG:$rB)))]>;

def SFr32 : RRForm<0b00000010000, (outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
  "sf\t$rT, $rA, $rB", IntegerOp,
  [(set R32C:$rT, (sub R32C:$rA, R32C:$rB))]>;

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
             [(set (vectype VECREG:$rT),
                   (SPUaddx (vectype VECREG:$rA), (vectype VECREG:$rB),
                            (vectype VECREG:$rCarry)))]>,
    RegConstraint<"$rCarry = $rT">,
    NoEncode<"$rCarry">;

class ADDXRegInst<RegisterClass rclass>:
    ADDXInst<(outs rclass:$rT),
             (ins rclass:$rA, rclass:$rB, rclass:$rCarry),
             [(set rclass:$rT,
                   (SPUaddx rclass:$rA, rclass:$rB, rclass:$rCarry))]>,
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
             [(set (vectype VECREG:$rT),
                   (SPUcarry_gen (vectype VECREG:$rA), (vectype VECREG:$rB)))]>;

class CGRegInst<RegisterClass rclass>:
    CGInst<(outs rclass:$rT),
           (ins rclass:$rA, rclass:$rB),
             [(set rclass:$rT,
                   (SPUcarry_gen rclass:$rA, rclass:$rB))]>;

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
             [(set (vectype VECREG:$rT),
                   (SPUsubx (vectype VECREG:$rA), (vectype VECREG:$rB),
                            (vectype VECREG:$rCarry)))]>,
    RegConstraint<"$rCarry = $rT">,
    NoEncode<"$rCarry">;

class SFXRegInst<RegisterClass rclass>:
    SFXInst<(outs rclass:$rT),
            (ins rclass:$rA, rclass:$rB, rclass:$rCarry),
             [(set rclass:$rT,
                   (SPUsubx rclass:$rA, rclass:$rB, rclass:$rCarry))]>,
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
           [(set (vectype VECREG:$rT),
                 (SPUborrow_gen (vectype VECREG:$rA), (vectype VECREG:$rB)))]>;

class BGRegInst<RegisterClass rclass>:
    BGInst<(outs rclass:$rT),
           (ins rclass:$rA, rclass:$rB),
           [(set rclass:$rT,
                 (SPUborrow_gen rclass:$rA, rclass:$rB))]>;

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
    [(set (v8i16 VECREG:$rT), (SPUmpy_v8i16 (v8i16 VECREG:$rA),
                                            (v8i16 VECREG:$rB)))]>;

def MPYr16:
  RRForm<0b00100011110, (outs R16C:$rT), (ins R16C:$rA, R16C:$rB),
    "mpy\t$rT, $rA, $rB", IntegerMulDiv,
    [(set R16C:$rT, (mul R16C:$rA, R16C:$rB))]>;

def MPYUv4i32:
  RRForm<0b00110011110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
    "mpyu\t$rT, $rA, $rB", IntegerMulDiv,
    [(set (v4i32 VECREG:$rT),
          (SPUmpyu_v4i32 (v4i32 VECREG:$rA), (v4i32 VECREG:$rB)))]>;

def MPYUr16:
  RRForm<0b00110011110, (outs R32C:$rT), (ins R16C:$rA, R16C:$rB),
    "mpyu\t$rT, $rA, $rB", IntegerMulDiv,
    [(set R32C:$rT, (mul (zext R16C:$rA),
                         (zext R16C:$rB)))]>;

def MPYUr32:
  RRForm<0b00110011110, (outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
    "mpyu\t$rT, $rA, $rB", IntegerMulDiv,
    [(set R32C:$rT, (SPUmpyu_i32 R32C:$rA, R32C:$rB))]>;

// mpyi: multiply 16 x s10imm -> 32 result (custom lowering for 32 bit result,
// this only produces the lower 16 bits)
def MPYIvec:
  RI10Form<0b00101110, (outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
    "mpyi\t$rT, $rA, $val", IntegerMulDiv,
    [(set (v8i16 VECREG:$rT), (mul (v8i16 VECREG:$rA), v8i16SExt10Imm:$val))]>;

def MPYIr16:
  RI10Form<0b00101110, (outs R16C:$rT), (ins R16C:$rA, s10imm:$val),
    "mpyi\t$rT, $rA, $val", IntegerMulDiv,
    [(set R16C:$rT, (mul R16C:$rA, i16ImmSExt10:$val))]>;

// mpyui: same issues as other multiplies, plus, this doesn't match a
// pattern... but may be used during target DAG selection or lowering
def MPYUIvec:
  RI10Form<0b10101110, (outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
    "mpyui\t$rT, $rA, $val", IntegerMulDiv,
    []>;

def MPYUIr16:
  RI10Form<0b10101110, (outs R16C:$rT), (ins R16C:$rA, s10imm:$val),
    "mpyui\t$rT, $rA, $val", IntegerMulDiv,
    []>;

// mpya: 16 x 16 + 16 -> 32 bit result
def MPYAvec:
  RRRForm<0b0011, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB, VECREG:$rC),
    "mpya\t$rT, $rA, $rB, $rC", IntegerMulDiv,
    [(set (v4i32 VECREG:$rT), (add (v4i32 (bitconvert (mul (v8i16 VECREG:$rA),
                                                           (v8i16 VECREG:$rB)))),
                                   (v4i32 VECREG:$rC)))]>;

def MPYAr32:
  RRRForm<0b0011, (outs R32C:$rT), (ins R16C:$rA, R16C:$rB, R32C:$rC),
    "mpya\t$rT, $rA, $rB, $rC", IntegerMulDiv,
    [(set R32C:$rT, (add (sext (mul R16C:$rA, R16C:$rB)),
                         R32C:$rC))]>;

def : Pat<(add (mul (sext R16C:$rA), (sext R16C:$rB)), R32C:$rC),
          (MPYAr32 R16C:$rA, R16C:$rB, R32C:$rC)>;

def MPYAr32_sextinreg:
  RRRForm<0b0011, (outs R32C:$rT), (ins R32C:$rA, R32C:$rB, R32C:$rC),
    "mpya\t$rT, $rA, $rB, $rC", IntegerMulDiv,
    [(set R32C:$rT, (add (mul (sext_inreg R32C:$rA, i16),
                              (sext_inreg R32C:$rB, i16)),
                         R32C:$rC))]>;

//def MPYAr32:
//  RRRForm<0b0011, (outs R32C:$rT), (ins R16C:$rA, R16C:$rB, R32C:$rC),
//    "mpya\t$rT, $rA, $rB, $rC", IntegerMulDiv,
//    [(set R32C:$rT, (add (sext (mul R16C:$rA, R16C:$rB)),
//                         R32C:$rC))]>;

// mpyh: multiply high, used to synthesize 32-bit multiplies
def MPYHv4i32:
    RRForm<0b10100011110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "mpyh\t$rT, $rA, $rB", IntegerMulDiv,
      [(set (v4i32 VECREG:$rT),
            (SPUmpyh_v4i32 (v4i32 VECREG:$rA), (v4i32 VECREG:$rB)))]>;

def MPYHr32:
    RRForm<0b10100011110, (outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
      "mpyh\t$rT, $rA, $rB", IntegerMulDiv,
      [(set R32C:$rT, (SPUmpyh_i32 R32C:$rA, R32C:$rB))]>;

// mpys: multiply high and shift right (returns the top half of
// a 16-bit multiply, sign extended to 32 bits.)
def MPYSvec:
    RRForm<0b11100011110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "mpys\t$rT, $rA, $rB", IntegerMulDiv,
      []>;

def MPYSr16:
    RRForm<0b11100011110, (outs R32C:$rT), (ins R16C:$rA, R16C:$rB),
      "mpys\t$rT, $rA, $rB", IntegerMulDiv,
      []>;

// mpyhh: multiply high-high (returns the 32-bit result from multiplying
// the top 16 bits of the $rA, $rB)
def MPYHHv8i16:
    RRForm<0b01100011110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "mpyhh\t$rT, $rA, $rB", IntegerMulDiv,
      [(set (v8i16 VECREG:$rT),
            (SPUmpyhh_v8i16 (v8i16 VECREG:$rA), (v8i16 VECREG:$rB)))]>;

def MPYHHr32:
    RRForm<0b01100011110, (outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
      "mpyhh\t$rT, $rA, $rB", IntegerMulDiv,
      []>;

// mpyhha: Multiply high-high, add to $rT:
def MPYHHAvec:
    RRForm<0b01100010110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "mpyhha\t$rT, $rA, $rB", IntegerMulDiv,
      []>;

def MPYHHAr32:
    RRForm<0b01100010110, (outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
      "mpyhha\t$rT, $rA, $rB", IntegerMulDiv,
      []>;

// mpyhhu: Multiply high-high, unsigned
def MPYHHUvec:
    RRForm<0b01110011110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "mpyhhu\t$rT, $rA, $rB", IntegerMulDiv,
      []>;

def MPYHHUr32:
    RRForm<0b01110011110, (outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
      "mpyhhu\t$rT, $rA, $rB", IntegerMulDiv,
      []>;

// mpyhhau: Multiply high-high, unsigned
def MPYHHAUvec:
    RRForm<0b01110010110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "mpyhhau\t$rT, $rA, $rB", IntegerMulDiv,
      []>;

def MPYHHAUr32:
    RRForm<0b01110010110, (outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
      "mpyhhau\t$rT, $rA, $rB", IntegerMulDiv,
      []>;

// clz: Count leading zeroes
def CLZv4i32:
    RRForm_1<0b10100101010, (outs VECREG:$rT), (ins VECREG:$rA),
      "clz\t$rT, $rA", IntegerOp,
      [/* intrinsic */]>;

def CLZr32:
    RRForm_1<0b10100101010, (outs R32C:$rT), (ins R32C:$rA),
      "clz\t$rT, $rA", IntegerOp,
      [(set R32C:$rT, (ctlz R32C:$rA))]>;

// cntb: Count ones in bytes (aka "population count")
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

// gbb: Gather all low order bits from each byte in $rA into a single 16-bit
// quantity stored into $rT
def GBB:
    RRForm_1<0b01001101100, (outs R16C:$rT), (ins VECREG:$rA),
      "gbb\t$rT, $rA", GatherOp,
      []>;

// gbh: Gather all low order bits from each halfword in $rA into a single
// 8-bit quantity stored in $rT
def GBH:
    RRForm_1<0b10001101100, (outs R16C:$rT), (ins VECREG:$rA),
      "gbh\t$rT, $rA", GatherOp,
      []>;

// gb: Gather all low order bits from each word in $rA into a single
// 4-bit quantity stored in $rT
def GB:
    RRForm_1<0b00001101100, (outs R16C:$rT), (ins VECREG:$rA),
      "gb\t$rT, $rA", GatherOp,
      []>;

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

class XSBHVecInst<ValueType vectype>:
    XSBHInst<(outs VECREG:$rDst), (ins VECREG:$rSrc),
      [(set (v8i16 VECREG:$rDst), (sext (vectype VECREG:$rSrc)))]>;

class XSBHRegInst<RegisterClass rclass>:
    XSBHInst<(outs rclass:$rDst), (ins rclass:$rSrc),
      [(set rclass:$rDst, (sext_inreg rclass:$rSrc, i8))]>;

multiclass ExtendByteHalfword {
  def v16i8: XSBHVecInst<v8i16>;
  def r16: XSBHRegInst<R16C>;

  // 32-bit form for XSBH: used to sign extend 8-bit quantities to 16-bit
  // quantities to 32-bit quantities via a 32-bit register (see the sext 8->32
  // pattern below). Intentionally doesn't match a pattern because we want the
  // sext 8->32 pattern to do the work for us, namely because we need the extra
  // XSHWr32.
  def r32: XSBHRegInst<R32C>;
}

defm XSBH : ExtendByteHalfword;

// Sign-extend, but take an 8-bit register to a 16-bit register (not done as
// sext_inreg)
def XSBHr8:
    XSBHInst<(outs R16C:$rDst), (ins R8C:$rSrc),
             [(set R16C:$rDst, (sext R8C:$rSrc))]>;

// Sign extend halfwords to words:
def XSHWvec:
    RRForm_1<0b01101101010, (outs VECREG:$rDest), (ins VECREG:$rSrc),
      "xshw\t$rDest, $rSrc", IntegerOp,
      [(set (v4i32 VECREG:$rDest), (sext (v8i16 VECREG:$rSrc)))]>;

def XSHWr32:
    RRForm_1<0b01101101010, (outs R32C:$rDst), (ins R32C:$rSrc),
      "xshw\t$rDst, $rSrc", IntegerOp,
      [(set R32C:$rDst, (sext_inreg R32C:$rSrc, i16))]>;

def XSHWr16:
    RRForm_1<0b01101101010, (outs R32C:$rDst), (ins R16C:$rSrc),
      "xshw\t$rDst, $rSrc", IntegerOp,
      [(set R32C:$rDst, (sext R16C:$rSrc))]>;

def XSWDvec:
    RRForm_1<0b01100101010, (outs VECREG:$rDst), (ins VECREG:$rSrc),
      "xswd\t$rDst, $rSrc", IntegerOp,
      [(set (v2i64 VECREG:$rDst), (sext (v4i32 VECREG:$rSrc)))]>;

def XSWDr64:
    RRForm_1<0b01100101010, (outs R64C:$rDst), (ins R64C:$rSrc),
      "xswd\t$rDst, $rSrc", IntegerOp,
      [(set R64C:$rDst, (sext_inreg R64C:$rSrc, i32))]>;

def XSWDr32:
    RRForm_1<0b01100101010, (outs R64C:$rDst), (ins R32C:$rSrc),
      "xswd\t$rDst, $rSrc", IntegerOp,
      [(set R64C:$rDst, (SPUsext32_to_64 R32C:$rSrc))]>;

def : Pat<(sext R32C:$inp),
          (XSWDr32 R32C:$inp)>;

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

  def fabs64: ANDInst<(outs R64FP:$rT), (ins R64FP:$rA, VECREG:$rB),
                      [/* Intentionally does not match a pattern */]>;

  // Could use v4i32, but won't for clarity
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

// N.B.: vnot_conv is one of those special target selection pattern fragments,
// in which we expect there to be a bit_convert on the constant. Bear in mind
// that llvm translates "not <reg>" to "xor <reg>, -1" (or in this case, a
// constant -1 vector.)

class ANDCInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm<0b10000011010, OOL, IOL, "andc\t$rT, $rA, $rB",
           IntegerOp, pattern>;

class ANDCVecInst<ValueType vectype>:
    ANDCInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
             [(set (vectype VECREG:$rT), (and (vectype VECREG:$rA),
                                              (vnot (vectype VECREG:$rB))))]>;

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
}

defm ANDC : AndComplement;

class ANDBIInst<dag OOL, dag IOL, list<dag> pattern>:
    RI10Form<0b01101000, OOL, IOL, "andbi\t$rT, $rA, $val",
             IntegerOp, pattern>;

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
             IntegerOp, pattern>;

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

class ORPromoteScalar<RegisterClass rclass>:
    ORInst<(outs VECREG:$rT), (ins rclass:$rA, rclass:$rB),
           [/* no pattern */]>;

class ORExtractElt<RegisterClass rclass>:
    ORInst<(outs rclass:$rT), (ins VECREG:$rA, VECREG:$rB),
           [/* no pattern */]>;

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

  def r64: ORRegInst<R64C>;
  def r32: ORRegInst<R32C>;
  def r16: ORRegInst<R16C>;
  def r8:  ORRegInst<R8C>;

  // OR instructions used to copy f32 and f64 registers.
  def f32: ORInst<(outs R32FP:$rT), (ins R32FP:$rA, R32FP:$rB),
                  [/* no pattern */]>;

  def f64: ORInst<(outs R64FP:$rT), (ins R64FP:$rA, R64FP:$rB),
                  [/* no pattern */]>;

  // scalar->vector promotion:
  def v16i8_i8:  ORPromoteScalar<R8C>;
  def v8i16_i16: ORPromoteScalar<R16C>;
  def v4i32_i32: ORPromoteScalar<R32C>;
  def v2i64_i64: ORPromoteScalar<R64C>;
  def v4f32_f32: ORPromoteScalar<R32FP>;
  def v2f64_f64: ORPromoteScalar<R64FP>;

  // extract element 0:
  def i8_v16i8:  ORExtractElt<R8C>;
  def i16_v8i16: ORExtractElt<R16C>;
  def i32_v4i32: ORExtractElt<R32C>;
  def i64_v2i64: ORExtractElt<R64C>;
  def f32_v4f32: ORExtractElt<R32FP>;
  def f64_v2f64: ORExtractElt<R64FP>;
}

defm OR : BitwiseOr;

// scalar->vector promotion patterns:
def : Pat<(v16i8 (SPUpromote_scalar R8C:$rA)),
          (ORv16i8_i8 R8C:$rA, R8C:$rA)>;

def : Pat<(v8i16 (SPUpromote_scalar R16C:$rA)),
          (ORv8i16_i16 R16C:$rA, R16C:$rA)>;

def : Pat<(v4i32 (SPUpromote_scalar R32C:$rA)),
          (ORv4i32_i32 R32C:$rA, R32C:$rA)>;

def : Pat<(v2i64 (SPUpromote_scalar R64C:$rA)),
          (ORv2i64_i64 R64C:$rA, R64C:$rA)>;

def : Pat<(v4f32 (SPUpromote_scalar R32FP:$rA)),
          (ORv4f32_f32 R32FP:$rA, R32FP:$rA)>;

def : Pat<(v2f64 (SPUpromote_scalar R64FP:$rA)),
          (ORv2f64_f64 R64FP:$rA, R64FP:$rA)>;

// ORi*_v*: Used to extract vector element 0 (the preferred slot)

def : Pat<(SPUextract_elt0 (v16i8 VECREG:$rA)),
          (ORi8_v16i8 VECREG:$rA, VECREG:$rA)>;

def : Pat<(SPUextract_elt0_chained (v16i8 VECREG:$rA)),
          (ORi8_v16i8 VECREG:$rA, VECREG:$rA)>;

def : Pat<(SPUextract_elt0 (v8i16 VECREG:$rA)),
          (ORi16_v8i16 VECREG:$rA, VECREG:$rA)>;

def : Pat<(SPUextract_elt0_chained (v8i16 VECREG:$rA)),
          (ORi16_v8i16 VECREG:$rA, VECREG:$rA)>;

def : Pat<(SPUextract_elt0 (v4i32 VECREG:$rA)),
          (ORi32_v4i32 VECREG:$rA, VECREG:$rA)>;

def : Pat<(SPUextract_elt0_chained (v4i32 VECREG:$rA)),
          (ORi32_v4i32 VECREG:$rA, VECREG:$rA)>;

def : Pat<(SPUextract_elt0 (v2i64 VECREG:$rA)),
          (ORi64_v2i64 VECREG:$rA, VECREG:$rA)>;

def : Pat<(SPUextract_elt0_chained (v2i64 VECREG:$rA)),
          (ORi64_v2i64 VECREG:$rA, VECREG:$rA)>;

def : Pat<(SPUextract_elt0 (v4f32 VECREG:$rA)),
          (ORf32_v4f32 VECREG:$rA, VECREG:$rA)>;

def : Pat<(SPUextract_elt0_chained (v4f32 VECREG:$rA)),
          (ORf32_v4f32 VECREG:$rA, VECREG:$rA)>;

def : Pat<(SPUextract_elt0 (v2f64 VECREG:$rA)),
          (ORf64_v2f64 VECREG:$rA, VECREG:$rA)>;

def : Pat<(SPUextract_elt0_chained (v2f64 VECREG:$rA)),
          (ORf64_v2f64 VECREG:$rA, VECREG:$rA)>;

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

  def r32: ORIInst<(outs R32C:$rT), (ins R32C:$rA, u10imm_i32:$val),
                   [(set R32C:$rT, (or R32C:$rA, i32ImmUns10:$val))]>;

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

  // Special forms for floating point instructions.
  // fneg and fabs require bitwise logical ops to manipulate the sign bit.

  def fneg32: XORInst<(outs R32FP:$rT), (ins R32FP:$rA, R32C:$rB),
                      [/* no pattern */]>;

  def fneg64: XORInst<(outs R64FP:$rT), (ins R64FP:$rA, VECREG:$rB),
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
def NANDv16i8:
    RRForm<0b10010010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "nand\t$rT, $rA, $rB", IntegerOp,
      [(set (v16i8 VECREG:$rT), (vnot (and (v16i8 VECREG:$rA),
                                           (v16i8 VECREG:$rB))))]>;

def NANDv8i16:
    RRForm<0b10010010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "nand\t$rT, $rA, $rB", IntegerOp,
      [(set (v8i16 VECREG:$rT), (vnot (and (v8i16 VECREG:$rA),
                                           (v8i16 VECREG:$rB))))]>;

def NANDv4i32:
    RRForm<0b10010010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "nand\t$rT, $rA, $rB", IntegerOp,
      [(set (v4i32 VECREG:$rT), (vnot (and (v4i32 VECREG:$rA),
                                           (v4i32 VECREG:$rB))))]>;

def NANDr32:
    RRForm<0b10010010000, (outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
      "nand\t$rT, $rA, $rB", IntegerOp,
      [(set R32C:$rT, (not (and R32C:$rA, R32C:$rB)))]>;

def NANDr16:
    RRForm<0b10010010000, (outs R16C:$rT), (ins R16C:$rA, R16C:$rB),
      "nand\t$rT, $rA, $rB", IntegerOp,
      [(set R16C:$rT, (not (and R16C:$rA, R16C:$rB)))]>;

def NANDr8:
    RRForm<0b10010010000, (outs R8C:$rT), (ins R8C:$rA, R8C:$rB),
      "nand\t$rT, $rA, $rB", IntegerOp,
      [(set R8C:$rT, (not (and R8C:$rA, R8C:$rB)))]>;

// NOR:
def NORv16i8:
    RRForm<0b10010010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "nor\t$rT, $rA, $rB", IntegerOp,
      [(set (v16i8 VECREG:$rT), (vnot (or (v16i8 VECREG:$rA),
                                          (v16i8 VECREG:$rB))))]>;

def NORv8i16:
    RRForm<0b10010010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "nor\t$rT, $rA, $rB", IntegerOp,
      [(set (v8i16 VECREG:$rT), (vnot (or (v8i16 VECREG:$rA),
                                          (v8i16 VECREG:$rB))))]>;

def NORv4i32:
    RRForm<0b10010010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "nor\t$rT, $rA, $rB", IntegerOp,
      [(set (v4i32 VECREG:$rT), (vnot (or (v4i32 VECREG:$rA),
                                          (v4i32 VECREG:$rB))))]>;

def NORr32:
    RRForm<0b10010010000, (outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
      "nor\t$rT, $rA, $rB", IntegerOp,
      [(set R32C:$rT, (not (or R32C:$rA, R32C:$rB)))]>;

def NORr16:
    RRForm<0b10010010000, (outs R16C:$rT), (ins R16C:$rA, R16C:$rB),
      "nor\t$rT, $rA, $rB", IntegerOp,
      [(set R16C:$rT, (not (or R16C:$rA, R16C:$rB)))]>;

def NORr8:
    RRForm<0b10010010000, (outs R8C:$rT), (ins R8C:$rA, R8C:$rB),
      "nor\t$rT, $rA, $rB", IntegerOp,
      [(set R8C:$rT, (not (or R8C:$rA, R8C:$rB)))]>;

// Select bits:
class SELBInst<dag OOL, dag IOL, list<dag> pattern>:
    RRRForm<0b1000, OOL, IOL, "selb\t$rT, $rA, $rB, $rC",
            IntegerOp, pattern>;

class SELBVecInst<ValueType vectype>:
  SELBInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB, VECREG:$rC),
           [(set (vectype VECREG:$rT),
                 (or (and (vectype VECREG:$rC), (vectype VECREG:$rB)),
                     (and (vnot (vectype VECREG:$rC)),
                          (vectype VECREG:$rA))))]>;

class SELBRegInst<RegisterClass rclass>:
  SELBInst<(outs rclass:$rT), (ins rclass:$rA, rclass:$rB, rclass:$rC),
           [(set rclass:$rT,
                 (or (and rclass:$rA, rclass:$rC),
                     (and rclass:$rB, (not rclass:$rC))))]>;

multiclass SelectBits
{
  def v16i8: SELBVecInst<v16i8>;
  def v8i16: SELBVecInst<v8i16>;
  def v4i32: SELBVecInst<v4i32>;
  def v2i64: SELBVecInst<v2i64>;

  def r128:  SELBRegInst<GPRC>;
  def r64:   SELBRegInst<R64C>;
  def r32:   SELBRegInst<R32C>;
  def r16:   SELBRegInst<R16C>;
  def r8:    SELBRegInst<R8C>;
}

defm SELB : SelectBits;

class SPUselbPat<ValueType vectype, SPUInstr inst>:
   Pat<(SPUselb (vectype VECREG:$rA), (vectype VECREG:$rB), (vectype VECREG:$rC)),
       (inst VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

def : SPUselbPat<v16i8, SELBv16i8>;
def : SPUselbPat<v8i16, SELBv8i16>;
def : SPUselbPat<v4i32, SELBv4i32>;
def : SPUselbPat<v2i64, SELBv2i64>;

class SelectConditional<RegisterClass rclass, SPUInstr inst>:
    Pat<(select rclass:$rCond, rclass:$rTrue, rclass:$rFalse),
        (inst rclass:$rFalse, rclass:$rTrue, rclass:$rCond)>;

def : SelectConditional<R32C, SELBr32>;
def : SelectConditional<R16C, SELBr16>;
def : SelectConditional<R8C, SELBr8>;

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
            IntegerOp, pattern>;

class SHUFBVecInst<ValueType vectype>:
    SHUFBInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB, VECREG:$rC),
              [(set (vectype VECREG:$rT), (SPUshuffle (vectype VECREG:$rA),
                                                      (vectype VECREG:$rB),
                                                      (vectype VECREG:$rC)))]>;

// It's this pattern that's probably the most useful, since SPUISelLowering
// methods create a v16i8 vector for $rC:
class SHUFBVecPat1<ValueType vectype, ValueType masktype, SPUInstr inst>:
    Pat<(SPUshuffle (vectype VECREG:$rA), (vectype VECREG:$rB),
                    (masktype VECREG:$rC)),
         (inst VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

multiclass ShuffleBytes
{
  def v16i8 : SHUFBVecInst<v16i8>;
  def v8i16 : SHUFBVecInst<v8i16>;
  def v4i32 : SHUFBVecInst<v4i32>;
  def v2i64 : SHUFBVecInst<v2i64>;

  def v4f32 : SHUFBVecInst<v4f32>;
  def v2f64 : SHUFBVecInst<v2f64>;
}

defm SHUFB : ShuffleBytes;

// Shuffle mask is a v16i8 vector
def : SHUFBVecPat1<v8i16, v16i8, SHUFBv16i8>;
def : SHUFBVecPat1<v4i32, v16i8, SHUFBv16i8>;
def : SHUFBVecPat1<v2i64, v16i8, SHUFBv16i8>;
def : SHUFBVecPat1<v4f32, v16i8, SHUFBv16i8>;
def : SHUFBVecPat1<v2f64, v16i8, SHUFBv16i8>;

// Shuffle mask is a v4i32 vector:
def : SHUFBVecPat1<v8i16, v4i32, SHUFBv4i32>;
def : SHUFBVecPat1<v4i32, v4i32, SHUFBv4i32>;
def : SHUFBVecPat1<v2i64, v4i32, SHUFBv4i32>;
def : SHUFBVecPat1<v4f32, v4i32, SHUFBv4i32>;
def : SHUFBVecPat1<v2f64, v4i32, SHUFBv4i32>;

//===----------------------------------------------------------------------===//
// Shift and rotate group:
//===----------------------------------------------------------------------===//

class SHLHInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm<0b11111010000, OOL, IOL, "shlh\t$rT, $rA, $rB",
           RotateShift, pattern>;

class SHLHVecInst<ValueType vectype>:
    SHLHInst<(outs VECREG:$rT), (ins VECREG:$rA, R16C:$rB),
             [(set (vectype VECREG:$rT),
                   (SPUvec_shl (vectype VECREG:$rA), R16C:$rB))]>;

// $rB gets promoted to 32-bit register type when confronted with
// this llvm assembly code:
//
// define i16 @shlh_i16_1(i16 %arg1, i16 %arg2) {
//      %A = shl i16 %arg1, %arg2
//      ret i16 %A
// }

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
            RotateShift, pattern>;

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
          (SHLHIv8i16 VECREG:$rA, uimm7:$val)>;

def : Pat<(shl R16C:$rA, (i32 uimm7:$val)),
          (SHLHIr16 R16C:$rA, uimm7:$val)>;

//===----------------------------------------------------------------------===//

class SHLInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm<0b11111010000, OOL, IOL, "shl\t$rT, $rA, $rB",
           RotateShift, pattern>;

multiclass ShiftLeftWord
{
  def v4i32:
      SHLInst<(outs VECREG:$rT), (ins VECREG:$rA, R16C:$rB),
              [(set (v4i32 VECREG:$rT),
                    (SPUvec_shl (v4i32 VECREG:$rA), R16C:$rB))]>;
  def r32:
      SHLInst<(outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
              [(set R32C:$rT, (shl R32C:$rA, R32C:$rB))]>;
}

defm SHL: ShiftLeftWord;

//===----------------------------------------------------------------------===//

class SHLIInst<dag OOL, dag IOL, list<dag> pattern>:
    RI7Form<0b11111010000, OOL, IOL, "shli\t$rT, $rA, $val",
            RotateShift, pattern>;

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
           RotateShift, pattern>;

class SHLQBIVecInst<ValueType vectype>:
    SHLQBIInst<(outs VECREG:$rT), (ins VECREG:$rA, R32C:$rB),
               [(set (vectype VECREG:$rT),
                     (SPUshlquad_l_bits (vectype VECREG:$rA), R32C:$rB))]>;

multiclass ShiftLeftQuadByBits
{
  def v16i8: SHLQBIVecInst<v16i8>;
  def v8i16: SHLQBIVecInst<v8i16>;
  def v4i32: SHLQBIVecInst<v4i32>;
  def v2i64: SHLQBIVecInst<v2i64>;
}

defm SHLQBI : ShiftLeftQuadByBits;

// See note above on SHLQBI. In this case, the predicate actually does then
// enforcement, whereas with SHLQBI, we have to "take it on faith."
class SHLQBIIInst<dag OOL, dag IOL, list<dag> pattern>:
    RI7Form<0b11011111100, OOL, IOL, "shlqbii\t$rT, $rA, $val",
            RotateShift, pattern>;

class SHLQBIIVecInst<ValueType vectype>:
    SHLQBIIInst<(outs VECREG:$rT), (ins VECREG:$rA, u7imm_i32:$val),
                [(set (vectype VECREG:$rT),
                      (SPUshlquad_l_bits (vectype VECREG:$rA), (i32 bitshift:$val)))]>;

multiclass ShiftLeftQuadByBitsImm
{
  def v16i8 : SHLQBIIVecInst<v16i8>;
  def v8i16 : SHLQBIIVecInst<v8i16>;
  def v4i32 : SHLQBIIVecInst<v4i32>;
  def v2i64 : SHLQBIIVecInst<v2i64>;
}

defm SHLQBII : ShiftLeftQuadByBitsImm;

// SHLQBY, SHLQBYI vector forms: Shift the entire vector to the left by bytes,
// not by bits. See notes above on SHLQBI.

class SHLQBYInst<dag OOL, dag IOL, list<dag> pattern>:
    RI7Form<0b11111011100, OOL, IOL, "shlqbyi\t$rT, $rA, $rB",
            RotateShift, pattern>;

class SHLQBYVecInst<ValueType vectype>:
    SHLQBYInst<(outs VECREG:$rT), (ins VECREG:$rA, R32C:$rB),
               [(set (vectype VECREG:$rT),
                     (SPUshlquad_l_bytes (vectype VECREG:$rA), R32C:$rB))]>;

multiclass ShiftLeftQuadBytes
{
  def v16i8: SHLQBYVecInst<v16i8>;
  def v8i16: SHLQBYVecInst<v8i16>;
  def v4i32: SHLQBYVecInst<v4i32>;
  def v2i64: SHLQBYVecInst<v2i64>;
  def r128: SHLQBYInst<(outs GPRC:$rT), (ins GPRC:$rA, R32C:$rB),
                       [(set GPRC:$rT, (SPUshlquad_l_bytes GPRC:$rA, R32C:$rB))]>;
}

defm SHLQBY: ShiftLeftQuadBytes;

class SHLQBYIInst<dag OOL, dag IOL, list<dag> pattern>:
    RI7Form<0b11111111100, OOL, IOL, "shlqbyi\t$rT, $rA, $val",
            RotateShift, pattern>;

class SHLQBYIVecInst<ValueType vectype>:
    SHLQBYIInst<(outs VECREG:$rT), (ins VECREG:$rA, u7imm_i32:$val),
                [(set (vectype VECREG:$rT),
                      (SPUshlquad_l_bytes (vectype VECREG:$rA), (i32 uimm7:$val)))]>;

multiclass ShiftLeftQuadBytesImm
{
  def v16i8: SHLQBYIVecInst<v16i8>;
  def v8i16: SHLQBYIVecInst<v8i16>;
  def v4i32: SHLQBYIVecInst<v4i32>;
  def v2i64: SHLQBYIVecInst<v2i64>;
  def r128:  SHLQBYIInst<(outs GPRC:$rT), (ins GPRC:$rA, u7imm_i32:$val),
                         [(set GPRC:$rT,
                               (SPUshlquad_l_bytes GPRC:$rA, (i32 uimm7:$val)))]>;
}

defm SHLQBYI : ShiftLeftQuadBytesImm;

// Special form for truncating i64 to i32:
def SHLQBYItrunc64:  SHLQBYIInst<(outs R32C:$rT), (ins R64C:$rA, u7imm_i32:$val),
                                 [/* no pattern, see below */]>;

def : Pat<(trunc R64C:$rSrc),
          (SHLQBYItrunc64 R64C:$rSrc, 4)>;

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// Rotate halfword:
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
class ROTHInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm<0b00111010000, OOL, IOL, "roth\t$rT, $rA, $rB",
           RotateShift, pattern>;

class ROTHVecInst<ValueType vectype>:
    ROTHInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
             [(set (vectype VECREG:$rT),
                   (SPUvec_rotl VECREG:$rA, VECREG:$rB))]>;

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
            RotateShift, pattern>;

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

def : Pat<(SPUvec_rotl VECREG:$rA, (i32 uimm7:$val)),
          (ROTHIv8i16 VECREG:$rA, imm:$val)>;
    
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// Rotate word:
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

class ROTInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm<0b00011010000, OOL, IOL, "rot\t$rT, $rA, $rB",
           RotateShift, pattern>;

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
            RotateShift, pattern>;

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
           RotateShift, pattern>;

class ROTQBYVecInst<ValueType vectype>:
    ROTQBYInst<(outs VECREG:$rT), (ins VECREG:$rA, R32C:$rB),
               [(set (vectype VECREG:$rT),
                     (SPUrotbytes_left (vectype VECREG:$rA), R32C:$rB))]>;

multiclass RotateQuadLeftByBytes
{
  def v16i8: ROTQBYVecInst<v16i8>;
  def v8i16: ROTQBYVecInst<v8i16>;
  def v4i32: ROTQBYVecInst<v4i32>;
  def v2i64: ROTQBYVecInst<v2i64>;
}

defm ROTQBY: RotateQuadLeftByBytes;

def : Pat<(SPUrotbytes_left_chained (v16i8 VECREG:$rA), R32C:$rB),
          (ROTQBYv16i8 VECREG:$rA, R32C:$rB)>;
def : Pat<(SPUrotbytes_left_chained (v8i16 VECREG:$rA), R32C:$rB),
          (ROTQBYv8i16 VECREG:$rA, R32C:$rB)>;
def : Pat<(SPUrotbytes_left_chained (v4i32 VECREG:$rA), R32C:$rB),
          (ROTQBYv4i32 VECREG:$rA, R32C:$rB)>;
def : Pat<(SPUrotbytes_left_chained (v2i64 VECREG:$rA), R32C:$rB),
          (ROTQBYv2i64 VECREG:$rA, R32C:$rB)>;

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// Rotate quad by byte (count), immediate
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

class ROTQBYIInst<dag OOL, dag IOL, list<dag> pattern>:
    RI7Form<0b00111111100, OOL, IOL, "rotqbyi\t$rT, $rA, $val",
            RotateShift, pattern>;

class ROTQBYIVecInst<ValueType vectype>:
    ROTQBYIInst<(outs VECREG:$rT), (ins VECREG:$rA, u7imm:$val),
                [(set (vectype VECREG:$rT),
                      (SPUrotbytes_left (vectype VECREG:$rA), (i16 uimm7:$val)))]>;

multiclass RotateQuadByBytesImm
{
  def v16i8: ROTQBYIVecInst<v16i8>;
  def v8i16: ROTQBYIVecInst<v8i16>;
  def v4i32: ROTQBYIVecInst<v4i32>;
  def v2i64: ROTQBYIVecInst<v2i64>;
}

defm ROTQBYI: RotateQuadByBytesImm;

def : Pat<(SPUrotbytes_left_chained (v16i8 VECREG:$rA), (i16 uimm7:$val)),
          (ROTQBYIv16i8 VECREG:$rA, uimm7:$val)>;
def : Pat<(SPUrotbytes_left_chained (v8i16 VECREG:$rA), (i16 uimm7:$val)),
          (ROTQBYIv8i16 VECREG:$rA, uimm7:$val)>;
def : Pat<(SPUrotbytes_left_chained (v4i32 VECREG:$rA), (i16 uimm7:$val)),
          (ROTQBYIv4i32 VECREG:$rA, uimm7:$val)>;
def : Pat<(SPUrotbytes_left_chained (v2i64 VECREG:$rA), (i16 uimm7:$val)),
          (ROTQBYIv2i64 VECREG:$rA, uimm7:$val)>;

// See ROTQBY note above.
class ROTQBYBIInst<dag OOL, dag IOL, list<dag> pattern>:
    RI7Form<0b00110011100, OOL, IOL,
      "rotqbybi\t$rT, $rA, $shift",
      RotateShift, pattern>;

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
           RotateShift, pattern>;

class ROTQBIVecInst<ValueType vectype>:
    ROTQBIInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
               [/* no pattern yet */]>;

class ROTQBIRegInst<RegisterClass rclass>:
    ROTQBIInst<(outs rclass:$rT), (ins rclass:$rA, rclass:$rB),
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
            RotateShift, pattern>;

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
           RotateShift, pattern>;

def ROTHMv8i16:
    ROTHMInst<(outs VECREG:$rT), (ins VECREG:$rA, R32C:$rB),
              [/* see patterns below - $rB must be negated */]>;

def : Pat<(SPUvec_srl (v8i16 VECREG:$rA), R32C:$rB),
          (ROTHMv8i16 VECREG:$rA, (SFIr32 R32C:$rB, 0))>;

def : Pat<(SPUvec_srl (v8i16 VECREG:$rA), R16C:$rB),
          (ROTHMv8i16 VECREG:$rA,
                      (SFIr32 (XSHWr16 R16C:$rB), 0))>;

def : Pat<(SPUvec_srl (v8i16 VECREG:$rA), R8C:$rB),
          (ROTHMv8i16 VECREG:$rA,
                      (SFIr32 (XSHWr16 (XSBHr8 R8C:$rB) ), 0))>;

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
            RotateShift, pattern>;

def ROTHMIv8i16:
    ROTHMIInst<(outs VECREG:$rT), (ins VECREG:$rA, rothNeg7imm:$val),
               [/* no pattern */]>;

def : Pat<(SPUvec_srl (v8i16 VECREG:$rA), (i32 imm:$val)),
          (ROTHMIv8i16 VECREG:$rA, imm:$val)>;

def: Pat<(SPUvec_srl (v8i16 VECREG:$rA), (i16 imm:$val)),
         (ROTHMIv8i16 VECREG:$rA, imm:$val)>;
 
def: Pat<(SPUvec_srl (v8i16 VECREG:$rA), (i8 imm:$val)),
         (ROTHMIv8i16 VECREG:$rA, imm:$val)>;

def ROTHMIr16:
    ROTHMIInst<(outs R16C:$rT), (ins R16C:$rA, rothNeg7imm:$val),
               [/* no pattern */]>;

def: Pat<(srl R16C:$rA, (i32 uimm7:$val)),
         (ROTHMIr16 R16C:$rA, uimm7:$val)>;

def: Pat<(srl R16C:$rA, (i16 uimm7:$val)),
         (ROTHMIr16 R16C:$rA, uimm7:$val)>;

def: Pat<(srl R16C:$rA, (i8 uimm7:$val)),
         (ROTHMIr16 R16C:$rA, uimm7:$val)>;

// ROTM v4i32 form: See the ROTHM v8i16 comments.
class ROTMInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm<0b10011010000, OOL, IOL, "rotm\t$rT, $rA, $rB",
           RotateShift, pattern>;

def ROTMv4i32:
    ROTMInst<(outs VECREG:$rT), (ins VECREG:$rA, R32C:$rB),
             [/* see patterns below - $rB must be negated */]>;

def : Pat<(SPUvec_srl VECREG:$rA, R32C:$rB),
          (ROTMv4i32 VECREG:$rA, (SFIr32 R32C:$rB, 0))>;

def : Pat<(SPUvec_srl VECREG:$rA, R16C:$rB),
          (ROTMv4i32 VECREG:$rA,
                     (SFIr32 (XSHWr16 R16C:$rB), 0))>;

def : Pat<(SPUvec_srl VECREG:$rA, R8C:$rB),
          (ROTMv4i32 VECREG:$rA,
                     (SFIr32 (XSHWr16 (XSBHr8 R8C:$rB)), 0))>;

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
      "rotmi\t$rT, $rA, $val", RotateShift,
      [(set (v4i32 VECREG:$rT),
            (SPUvec_srl VECREG:$rA, (i32 uimm7:$val)))]>;

def : Pat<(SPUvec_srl VECREG:$rA, (i16 uimm7:$val)),
          (ROTMIv4i32 VECREG:$rA, uimm7:$val)>;
 
def : Pat<(SPUvec_srl VECREG:$rA, (i8 uimm7:$val)),
          (ROTMIv4i32 VECREG:$rA, uimm7:$val)>;

// ROTMI r32 form: know how to complement the immediate value.
def ROTMIr32:
    RI7Form<0b10011110000, (outs R32C:$rT), (ins R32C:$rA, rotNeg7imm:$val),
      "rotmi\t$rT, $rA, $val", RotateShift,
      [(set R32C:$rT, (srl R32C:$rA, (i32 uimm7:$val)))]>;

def : Pat<(srl R32C:$rA, (i16 imm:$val)),
          (ROTMIr32 R32C:$rA, uimm7:$val)>;

def : Pat<(srl R32C:$rA, (i8 imm:$val)),
          (ROTMIr32 R32C:$rA, uimm7:$val)>;

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// ROTQMBYvec: This is a vector form merely so that when used in an
// instruction pattern, type checking will succeed. This instruction assumes
// that the user knew to negate $rB.
//
// Using the SPUrotquad_rz_bytes target-specific DAG node, the patterns
// ensure that $rB is negated.
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

class ROTQMBYInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm<0b10111011100, OOL, IOL, "rotqmby\t$rT, $rA, $rB",
           RotateShift, pattern>;

class ROTQMBYVecInst<ValueType vectype>:
    ROTQMBYInst<(outs VECREG:$rT), (ins VECREG:$rA, R32C:$rB),
                [/* no pattern, $rB must be negated */]>;

class ROTQMBYRegInst<RegisterClass rclass>:
    ROTQMBYInst<(outs rclass:$rT), (ins rclass:$rA, R32C:$rB),
                [(set rclass:$rT,
                      (SPUrotquad_rz_bytes rclass:$rA, R32C:$rB))]>;

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

def : Pat<(SPUrotquad_rz_bytes (v16i8 VECREG:$rA), R32C:$rB),
          (ROTQMBYv16i8 VECREG:$rA, (SFIr32 R32C:$rB, 0))>;
def : Pat<(SPUrotquad_rz_bytes (v8i16 VECREG:$rA), R32C:$rB),
          (ROTQMBYv8i16 VECREG:$rA, (SFIr32 R32C:$rB, 0))>;
def : Pat<(SPUrotquad_rz_bytes (v4i32 VECREG:$rA), R32C:$rB),
          (ROTQMBYv4i32 VECREG:$rA, (SFIr32 R32C:$rB, 0))>;
def : Pat<(SPUrotquad_rz_bytes (v2i64 VECREG:$rA), R32C:$rB),
          (ROTQMBYv2i64 VECREG:$rA, (SFIr32 R32C:$rB, 0))>;
def : Pat<(SPUrotquad_rz_bytes GPRC:$rA, R32C:$rB),
          (ROTQMBYr128 GPRC:$rA, (SFIr32 R32C:$rB, 0))>;
def : Pat<(SPUrotquad_rz_bytes R64C:$rA, R32C:$rB),
          (ROTQMBYr64 R64C:$rA, (SFIr32 R32C:$rB, 0))>;

class ROTQMBYIInst<dag OOL, dag IOL, list<dag> pattern>:
    RI7Form<0b10111111100, OOL, IOL, "rotqmbyi\t$rT, $rA, $val",
            RotateShift, pattern>;

class ROTQMBYIVecInst<ValueType vectype>:
    ROTQMBYIInst<(outs VECREG:$rT), (ins VECREG:$rA, rotNeg7imm:$val),
                 [(set (vectype VECREG:$rT),
                       (SPUrotquad_rz_bytes (vectype VECREG:$rA), (i32 uimm7:$val)))]>;

class ROTQMBYIRegInst<RegisterClass rclass, Operand optype, ValueType inttype, PatLeaf pred>:
    ROTQMBYIInst<(outs rclass:$rT), (ins rclass:$rA, optype:$val),
                 [(set rclass:$rT,
                       (SPUrotquad_rz_bytes rclass:$rA, (inttype pred:$val)))]>;

multiclass RotateQuadBytesImm
{
  def v16i8: ROTQMBYIVecInst<v16i8>;
  def v8i16: ROTQMBYIVecInst<v8i16>;
  def v4i32: ROTQMBYIVecInst<v4i32>;
  def v2i64: ROTQMBYIVecInst<v2i64>;

  def r128:  ROTQMBYIRegInst<GPRC, rotNeg7imm, i32, uimm7>;
  def r64:   ROTQMBYIRegInst<R64C, rotNeg7imm, i32, uimm7>;
}

defm ROTQMBYI : RotateQuadBytesImm;

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// Rotate right and mask by bit count
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

class ROTQMBYBIInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm<0b10110011100, OOL, IOL, "rotqmbybi\t$rT, $rA, $rB",
           RotateShift, pattern>;

class ROTQMBYBIVecInst<ValueType vectype>:
    ROTQMBYBIInst<(outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
                  [/* no pattern, intrinsic? */]>;

multiclass RotateMaskQuadByBitCount
{
  def v16i8: ROTQMBYBIVecInst<v16i8>;
  def v8i16: ROTQMBYBIVecInst<v8i16>;
  def v4i32: ROTQMBYBIVecInst<v4i32>;
  def v2i64: ROTQMBYBIVecInst<v2i64>;
}

defm ROTQMBYBI: RotateMaskQuadByBitCount;

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// Rotate quad and mask by bits
// Note that the rotate amount has to be negated
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

class ROTQMBIInst<dag OOL, dag IOL, list<dag> pattern>:
    RRForm<0b10011011100, OOL, IOL, "rotqmbi\t$rT, $rA, $rB",
           RotateShift, pattern>;

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

def : Pat<(SPUrotquad_rz_bits (v16i8 VECREG:$rA), R32C:$rB),
          (ROTQMBIv16i8 VECREG:$rA, (SFIr32 R32C:$rB, 0))>;
def : Pat<(SPUrotquad_rz_bits (v8i16 VECREG:$rA), R32C:$rB),
          (ROTQMBIv8i16 VECREG:$rA, (SFIr32 R32C:$rB, 0))>;
def : Pat<(SPUrotquad_rz_bits (v4i32 VECREG:$rA), R32C:$rB),
          (ROTQMBIv4i32 VECREG:$rA, (SFIr32 R32C:$rB, 0))>;
def : Pat<(SPUrotquad_rz_bits (v2i64 VECREG:$rA), R32C:$rB),
          (ROTQMBIv2i64 VECREG:$rA, (SFIr32 R32C:$rB, 0))>;
def : Pat<(SPUrotquad_rz_bits GPRC:$rA, R32C:$rB),
          (ROTQMBIr128 GPRC:$rA, (SFIr32 R32C:$rB, 0))>;
def : Pat<(SPUrotquad_rz_bits R64C:$rA, R32C:$rB),
          (ROTQMBIr64 R64C:$rA, (SFIr32 R32C:$rB, 0))>;

//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
// Rotate quad and mask by bits, immediate
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

class ROTQMBIIInst<dag OOL, dag IOL, list<dag> pattern>:
    RI7Form<0b10011111100, OOL, IOL, "rotqmbii\t$rT, $rA, $val",
            RotateShift, pattern>;

class ROTQMBIIVecInst<ValueType vectype>:
   ROTQMBIIInst<(outs VECREG:$rT), (ins VECREG:$rA, rotNeg7imm:$val),
                 [(set (vectype VECREG:$rT),
                       (SPUrotquad_rz_bits (vectype VECREG:$rA), (i32 uimm7:$val)))]>;

class ROTQMBIIRegInst<RegisterClass rclass>:
   ROTQMBIIInst<(outs rclass:$rT), (ins rclass:$rA, rotNeg7imm:$val),
                 [(set rclass:$rT,
                       (SPUrotquad_rz_bits rclass:$rA, (i32 uimm7:$val)))]>;

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
    RRForm<0b01111010000, (outs VECREG:$rT), (ins VECREG:$rA, R32C:$rB),
      "rotmah\t$rT, $rA, $rB", RotateShift,
      [/* see patterns below - $rB must be negated */]>;

def : Pat<(SPUvec_sra VECREG:$rA, R32C:$rB),
          (ROTMAHv8i16 VECREG:$rA, (SFIr32 R32C:$rB, 0))>;

def : Pat<(SPUvec_sra VECREG:$rA, R16C:$rB),
          (ROTMAHv8i16 VECREG:$rA,
                       (SFIr32 (XSHWr16 R16C:$rB), 0))>;

def : Pat<(SPUvec_sra VECREG:$rA, R8C:$rB),
          (ROTMAHv8i16 VECREG:$rA,
                       (SFIr32 (XSHWr16 (XSBHr8 R8C:$rB)), 0))>;

def ROTMAHr16:
    RRForm<0b01111010000, (outs R16C:$rT), (ins R16C:$rA, R32C:$rB),
      "rotmah\t$rT, $rA, $rB", RotateShift,
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
      "rotmahi\t$rT, $rA, $val", RotateShift,
      [(set (v8i16 VECREG:$rT),
            (SPUvec_sra (v8i16 VECREG:$rA), (i32 uimm7:$val)))]>;

def : Pat<(SPUvec_sra (v8i16 VECREG:$rA), (i16 uimm7:$val)),
          (ROTMAHIv8i16 (v8i16 VECREG:$rA), (i32 uimm7:$val))>;

def : Pat<(SPUvec_sra (v8i16 VECREG:$rA), (i8 uimm7:$val)),
          (ROTMAHIv8i16 (v8i16 VECREG:$rA), (i32 uimm7:$val))>;

def ROTMAHIr16:
    RRForm<0b01111110000, (outs R16C:$rT), (ins R16C:$rA, rothNeg7imm_i16:$val),
      "rotmahi\t$rT, $rA, $val", RotateShift,
      [(set R16C:$rT, (sra R16C:$rA, (i16 uimm7:$val)))]>;

def : Pat<(sra R16C:$rA, (i32 imm:$val)),
          (ROTMAHIr16 R16C:$rA, uimm7:$val)>;

def : Pat<(sra R16C:$rA, (i8 imm:$val)),
          (ROTMAHIr16 R16C:$rA, uimm7:$val)>;

def ROTMAv4i32:
    RRForm<0b01011010000, (outs VECREG:$rT), (ins VECREG:$rA, R32C:$rB),
      "rotma\t$rT, $rA, $rB", RotateShift,
      [/* see patterns below - $rB must be negated */]>;

def : Pat<(SPUvec_sra VECREG:$rA, R32C:$rB),
          (ROTMAv4i32 (v4i32 VECREG:$rA), (SFIr32 R32C:$rB, 0))>;

def : Pat<(SPUvec_sra VECREG:$rA, R16C:$rB),
          (ROTMAv4i32 (v4i32 VECREG:$rA),
                      (SFIr32 (XSHWr16 R16C:$rB), 0))>;

def : Pat<(SPUvec_sra VECREG:$rA, R8C:$rB),
          (ROTMAv4i32 (v4i32 VECREG:$rA),
                      (SFIr32 (XSHWr16 (XSBHr8 R8C:$rB)), 0))>;

def ROTMAr32:
    RRForm<0b01011010000, (outs R32C:$rT), (ins R32C:$rA, R32C:$rB),
      "rotma\t$rT, $rA, $rB", RotateShift,
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
      RotateShift, pattern>;

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

//------------------------------------------------------------------------
// Comparison operators:
//------------------------------------------------------------------------

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
// N.B.: This only matches the setcc set of conditionals. Special pattern
// matching is used for select conditionals.
//-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

class SETCCNegCondReg<PatFrag cond, RegisterClass rclass, ValueType inttype,
                      SPUInstr xorinst, SPUInstr cmpare>:
  Pat<(cond rclass:$rA, rclass:$rB),
      (xorinst (cmpare rclass:$rA, rclass:$rB), (inttype -1))>;

class SETCCNegCondImm<PatFrag cond, RegisterClass rclass, ValueType inttype,
                      PatLeaf immpred, SPUInstr xorinst, SPUInstr cmpare>:
  Pat<(cond rclass:$rA, (inttype immpred:$imm)),
      (xorinst (cmpare rclass:$rA, (inttype immpred:$imm)), (inttype -1))>;

def : SETCCNegCondReg<setne, R8C, i8, XORBIr8, CEQBr8>;
def : SETCCNegCondImm<setne, R8C, i8, immSExt8, XORBIr8, CEQBIr8>;

def : SETCCNegCondReg<setne, R16C, i16, XORHIr16, CEQHr16>;
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
              rclass:$rFalse, rclass:$rTrue),
      (selinstr rclass:$rTrue, rclass:$rFalse,
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
    BranchSetLink<0b011001100, (outs), (ins relcalltarget:$func, variable_ops),
      "brsl\t$$lr, $func",
      [(SPUcall (SPUpcrel tglobaladdr:$func, 0))]>;

  // Branch absolute and set link: Used if we actually know that the target
  // is an absolute address
  def BRASL:
    BranchSetLink<0b011001100, (outs), (ins calltarget:$func, variable_ops),
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

// Unconditional branches:
let isBranch = 1, isTerminator = 1, hasCtrlDep = 1, isBarrier = 1 in {
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
  def BI:
    BIForm<0b00010101100, "bi\t$func", [(brind R32C:$func)]>;

  // Various branches:
  def BRNZ:
    RI16Form<0b010000100, (outs), (ins R32C:$rCond, brtarget:$dest),
      "brnz\t$rCond,$dest",
      BranchResolv,
      [(brcond R32C:$rCond, bb:$dest)]>;

  def BRZ:
    RI16Form<0b000000100, (outs), (ins R32C:$rT, brtarget:$dest),
      "brz\t$rT,$dest",
      BranchResolv,
      [/* no pattern */]>;

  def BRHNZ:
    RI16Form<0b011000100, (outs), (ins R16C:$rCond, brtarget:$dest),
      "brhnz\t$rCond,$dest",
      BranchResolv,
      [(brcond R16C:$rCond, bb:$dest)]>;

  def BRHZ:
    RI16Form<0b001000100, (outs), (ins R16C:$rT, brtarget:$dest),
      "brhz\t$rT,$dest",
      BranchResolv,
      [/* no pattern */]>;
  
/*
  def BINZ:
    BICondForm<0b10010100100, "binz\t$rA, $func",
               [(SPUbinz R32C:$rA, R32C:$func)]>;

  def BIZ:
    BICondForm<0b00010100100, "biz\t$rA, $func",
               [(SPUbiz R32C:$rA, R32C:$func)]>;
*/
}

//===----------------------------------------------------------------------===//
// setcc and brcond patterns:
//===----------------------------------------------------------------------===//

def : Pat<(brcond (i16 (seteq R16C:$rA, 0)), bb:$dest), 
          (BRHZ R16C:$rA, bb:$dest)>;
def : Pat<(brcond (i16 (setne R16C:$rA, 0)), bb:$dest), 
          (BRHNZ R16C:$rA, bb:$dest)>;

def : Pat<(brcond (i32 (seteq R32C:$rA, 0)), bb:$dest), 
          (BRZ R32C:$rA, bb:$dest)>;
def : Pat<(brcond (i32 (setne R32C:$rA, 0)), bb:$dest), 
          (BRNZ R32C:$rA, bb:$dest)>;

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

defm BRCONDeq : BranchCondEQ<seteq, BRHZ, BRZ>;
defm BRCONDne : BranchCondEQ<setne, BRHNZ, BRNZ>;

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

defm BRCONDugt : BranchCondLGT<setugt, BRHNZ, BRNZ>;
defm BRCONDule : BranchCondLGT<setule, BRHZ, BRZ>;

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

defm BRCONDuge : BranchCondLGTEQ<setuge, ORr16, BRHNZ, ORr32, BRNZ>;
defm BRCONDult : BranchCondLGTEQ<setult, ORr16, BRHZ, ORr32, BRZ>;

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

defm BRCONDgt : BranchCondGT<setgt, BRHNZ, BRNZ>;
defm BRCONDle : BranchCondGT<setle, BRHZ, BRZ>;

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

defm BRCONDge : BranchCondGTEQ<setge, ORr16, BRHNZ, ORr32, BRNZ>;
defm BRCONDlt : BranchCondGTEQ<setlt, ORr16, BRHZ, ORr32, BRZ>;

let isTerminator = 1, isBarrier = 1 in {
  let isReturn = 1 in {
    def RET:
        RETForm<"bi\t$$lr", [(retflag)]>;
  }
}

//===----------------------------------------------------------------------===//
// Single precision floating point instructions
//===----------------------------------------------------------------------===//

def FAv4f32:
    RRForm<0b00100011010, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "fa\t$rT, $rA, $rB", SPrecFP,
      [(set (v4f32 VECREG:$rT), (fadd (v4f32 VECREG:$rA), (v4f32 VECREG:$rB)))]>;

def FAf32 :
    RRForm<0b00100011010, (outs R32FP:$rT), (ins R32FP:$rA, R32FP:$rB),
      "fa\t$rT, $rA, $rB", SPrecFP,
      [(set R32FP:$rT, (fadd R32FP:$rA, R32FP:$rB))]>;

def FSv4f32:
    RRForm<0b00100011010, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "fs\t$rT, $rA, $rB", SPrecFP,
      [(set (v4f32 VECREG:$rT), (fsub (v4f32 VECREG:$rA), (v4f32 VECREG:$rB)))]>;

def FSf32 :
    RRForm<0b10100011010, (outs R32FP:$rT), (ins R32FP:$rA, R32FP:$rB),
      "fs\t$rT, $rA, $rB", SPrecFP,
      [(set R32FP:$rT, (fsub R32FP:$rA, R32FP:$rB))]>;

// Floating point reciprocal estimate
def FREv4f32 :
    RRForm_1<0b00011101100, (outs VECREG:$rT), (ins VECREG:$rA),
      "frest\t$rT, $rA", SPrecFP,
      [(set (v4f32 VECREG:$rT), (SPUreciprocalEst (v4f32 VECREG:$rA)))]>;

def FREf32 :
    RRForm_1<0b00011101100, (outs R32FP:$rT), (ins R32FP:$rA),
      "frest\t$rT, $rA", SPrecFP,
      [(set R32FP:$rT, (SPUreciprocalEst R32FP:$rA))]>;

// Floating point interpolate (used in conjunction with reciprocal estimate)
def FIv4f32 :
    RRForm<0b00101011110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "fi\t$rT, $rA, $rB", SPrecFP,
      [(set (v4f32 VECREG:$rT), (SPUinterpolate (v4f32 VECREG:$rA),
                                                (v4f32 VECREG:$rB)))]>;

def FIf32 :
    RRForm<0b00101011110, (outs R32FP:$rT), (ins R32FP:$rA, R32FP:$rB),
      "fi\t$rT, $rA, $rB", SPrecFP,
      [(set R32FP:$rT, (SPUinterpolate R32FP:$rA, R32FP:$rB))]>;

// Floating Compare Equal
def FCEQf32 :
    RRForm<0b01000011110, (outs R32C:$rT), (ins R32FP:$rA, R32FP:$rB),
      "fceq\t$rT, $rA, $rB", SPrecFP,
      [(set R32C:$rT, (setoeq R32FP:$rA, R32FP:$rB))]>;

def FCMEQf32 :
    RRForm<0b01010011110, (outs R32C:$rT), (ins R32FP:$rA, R32FP:$rB),
      "fcmeq\t$rT, $rA, $rB", SPrecFP,
      [(set R32C:$rT, (setoeq (fabs R32FP:$rA), (fabs R32FP:$rB)))]>;

def FCGTf32 :
    RRForm<0b01000011010, (outs R32C:$rT), (ins R32FP:$rA, R32FP:$rB),
      "fcgt\t$rT, $rA, $rB", SPrecFP,
      [(set R32C:$rT, (setogt R32FP:$rA, R32FP:$rB))]>;

def FCMGTf32 :
    RRForm<0b01010011010, (outs R32C:$rT), (ins R32FP:$rA, R32FP:$rB),
      "fcmgt\t$rT, $rA, $rB", SPrecFP,
      [(set R32C:$rT, (setogt (fabs R32FP:$rA), (fabs R32FP:$rB)))]>;

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
// Floating point multiply instructions
//--------------------------------------

def FMv4f32:
    RRForm<0b00100011010, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "fm\t$rT, $rA, $rB", SPrecFP,
      [(set (v4f32 VECREG:$rT), (fmul (v4f32 VECREG:$rA),
                                      (v4f32 VECREG:$rB)))]>;

def FMf32 :
    RRForm<0b01100011010, (outs R32FP:$rT), (ins R32FP:$rA, R32FP:$rB),
      "fm\t$rT, $rA, $rB", SPrecFP,
      [(set R32FP:$rT, (fmul R32FP:$rA, R32FP:$rB))]>;

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
      [(set (v2f64 VECREG:$rT), (fextend (v4f32 VECREG:$rA)))]>;

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

// FNMS: - (a * b - c)
// - (a * b) + c => c - (a * b)
def FNMSf64 :
    RRForm<0b01111010110, (outs R64FP:$rT),
                          (ins R64FP:$rA, R64FP:$rB, R64FP:$rC),
      "dfnms\t$rT, $rA, $rB", DPrecFP,
      [(set R64FP:$rT, (fsub R64FP:$rC, (fmul R64FP:$rA, R64FP:$rB)))]>,
    RegConstraint<"$rC = $rT">,
    NoEncode<"$rC">;

def : Pat<(fneg (fsub (fmul R64FP:$rA, R64FP:$rB), R64FP:$rC)),
          (FNMSf64 R64FP:$rA, R64FP:$rB, R64FP:$rC)>;

def FNMSv2f64 :
    RRForm<0b01111010110, (outs VECREG:$rT),
                          (ins VECREG:$rA, VECREG:$rB, VECREG:$rC),
      "dfnms\t$rT, $rA, $rB", DPrecFP,
      [(set (v2f64 VECREG:$rT), 
            (fsub (v2f64 VECREG:$rC), 
                  (fmul (v2f64 VECREG:$rA), 
                        (v2f64 VECREG:$rB))))]>,
    RegConstraint<"$rC = $rT">,
    NoEncode<"$rC">;

def : Pat<(fneg (fsub (fmul (v2f64 VECREG:$rA), (v2f64 VECREG:$rB)),
                (v2f64 VECREG:$rC))),
          (FNMSv2f64 VECREG:$rA, VECREG:$rB, VECREG:$rC)>;

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

def : Pat<(fneg (v2f64 VECREG:$rA)),
          (XORfnegvec (v2f64 VECREG:$rA),
                      (v2f64 (ANDBIv16i8 (FSMBIv16i8 0x8080), 0x80)))>;

def : Pat<(fneg R64FP:$rA),
          (XORfneg64 R64FP:$rA,
                     (ANDBIv16i8 (FSMBIv16i8 0x8080), 0x80))>;

// Floating point absolute value

def : Pat<(fabs R32FP:$rA),
          (ANDfabs32 R32FP:$rA, (IOHLr32 (ILHUr32 0x7fff), 0xffff))>;

def : Pat<(fabs (v4f32 VECREG:$rA)),
          (ANDfabsvec (v4f32 VECREG:$rA),
                      (v4f32 (ANDBIv16i8 (FSMBIv16i8 0xffff), 0x7f)))>;

def : Pat<(fabs R64FP:$rA),
          (ANDfabs64 R64FP:$rA, (ANDBIv16i8 (FSMBIv16i8 0xffff), 0x7f))>;

def : Pat<(fabs (v2f64 VECREG:$rA)),
          (ANDfabsvec (v2f64 VECREG:$rA),
                      (v2f64 (ANDBIv16i8 (FSMBIv16i8 0xffff), 0x7f)))>;

//===----------------------------------------------------------------------===//
// Execution, Load NOP (execute NOPs belong in even pipeline, load NOPs belong
// in the odd pipeline)
//===----------------------------------------------------------------------===//

def ENOP : SPUInstr<(outs), (ins), "enop", ExecNOP> {
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
// NOTE: Promotions are handled using the XS* instructions. Truncation
// is not handled.
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
def : Pat<(v2f64 (bitconvert (v2f64 VECREG:$src))), (v2f64 VECREG:$src)>;

def : Pat<(f32 (bitconvert (i32 R32C:$src))), (f32 R32FP:$src)>;
def : Pat<(f64 (bitconvert (i64 R64C:$src))), (f64 R64FP:$src)>;

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
// Call instruction patterns:
//===----------------------------------------------------------------------===//
// Return void
def : Pat<(ret),
          (RET)>;

//===----------------------------------------------------------------------===//
// Zero/Any/Sign extensions
//===----------------------------------------------------------------------===//

// zext 1->32: Zero extend i1 to i32
def : Pat<(SPUextract_i1_zext R32C:$rSrc),
          (ANDIr32 R32C:$rSrc, 0x1)>;

// sext 8->32: Sign extend bytes to words
def : Pat<(sext_inreg R32C:$rSrc, i8),
          (XSHWr32 (XSBHr32 R32C:$rSrc))>;

def : Pat<(i32 (sext R8C:$rSrc)),
          (XSHWr16 (XSBHr8 R8C:$rSrc))>;

def : Pat<(SPUextract_i8_sext VECREG:$rSrc),
          (XSHWr32 (XSBHr32 (ORi32_v4i32 (v4i32 VECREG:$rSrc),
                            (v4i32 VECREG:$rSrc))))>;

// zext 8->16: Zero extend bytes to halfwords
def : Pat<(i16 (zext R8C:$rSrc)),
          (ANDHIi8i16 R8C:$rSrc, 0xff)>;

// zext 8->32 from preferred slot in load/store
def : Pat<(SPUextract_i8_zext VECREG:$rSrc),
          (ANDIr32 (ORi32_v4i32 (v4i32 VECREG:$rSrc), (v4i32 VECREG:$rSrc)),
                   0xff)>;

// zext 8->32: Zero extend bytes to words
def : Pat<(i32 (zext R8C:$rSrc)),
          (ANDIi8i32 R8C:$rSrc, 0xff)>;

// anyext 8->16: Extend 8->16 bits, irrespective of sign
def : Pat<(i16 (anyext R8C:$rSrc)),
          (ORHIi8i16 R8C:$rSrc, 0)>;

// anyext 8->32: Extend 8->32 bits, irrespective of sign
def : Pat<(i32 (anyext R8C:$rSrc)),
          (ORIi8i32 R8C:$rSrc, 0)>;

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
          (ORIi16i32 R16C:$rSrc, 0)>;

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

def : Pat<(SPUindirect R32C:$sp, i32ImmSExt10:$imm),
          (AIr32 R32C:$sp, i32ImmSExt10:$imm)>;

def : Pat<(SPUindirect R32C:$sp, imm:$imm),
          (Ar32 R32C:$sp,
                (IOHLr32 (ILHUr32 (HI16 imm:$imm)), (LO16 imm:$imm)))>;

def : Pat<(add (SPUhi tglobaladdr:$in, 0), (SPUlo tglobaladdr:$in, 0)),
          (IOHLlo (ILHUhi tglobaladdr:$in), tglobaladdr:$in)>;

def : Pat<(add (SPUhi texternalsym:$in, 0), (SPUlo texternalsym:$in, 0)),
          (IOHLlo (ILHUhi texternalsym:$in), texternalsym:$in)>;

def : Pat<(add (SPUhi tjumptable:$in, 0), (SPUlo tjumptable:$in, 0)),
          (IOHLlo (ILHUhi tjumptable:$in), tjumptable:$in)>;

def : Pat<(add (SPUhi tconstpool:$in, 0), (SPUlo tconstpool:$in, 0)),
          (IOHLlo (ILHUhi tconstpool:$in), tconstpool:$in)>;

// Instrinsics:
include "CellSDKIntrinsics.td"
