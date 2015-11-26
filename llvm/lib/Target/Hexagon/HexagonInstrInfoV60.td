//=- HexagonInstrInfoV60.td - Target Desc. for Hexagon Target -*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the Hexagon V60 instructions in TableGen format.
//
//===----------------------------------------------------------------------===//


// Vector store
let mayStore = 1, validSubTargets = HasV60SubT, hasSideEffects = 0 in
{
  class VSTInst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
                string cstr = "", InstrItinClass itin = CVI_VM_ST,
                IType type = TypeCVI_VM_ST>
  : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, type>, OpcodeHexagon;

}

// Vector load
let Predicates = [HasV60T, UseHVX] in
let mayLoad = 1, validSubTargets = HasV60SubT, hasSideEffects = 0 in
  class V6_LDInst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
                  string cstr = "", InstrItinClass itin = CVI_VM_LD,
                  IType type = TypeCVI_VM_LD>
  : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, type>;

let Predicates = [HasV60T, UseHVX] in
let mayStore = 1, validSubTargets = HasV60SubT, hasSideEffects = 0 in
class V6_STInst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
                string cstr = "", InstrItinClass itin = CVI_VM_ST,
                IType type = TypeCVI_VM_ST>
: InstHexagon<outs, ins, asmstr, pattern, cstr, itin, type>;

//===----------------------------------------------------------------------===//
// Vector loads with base + immediate offset
//===----------------------------------------------------------------------===//
let addrMode = BaseImmOffset, accessSize = Vector64Access in
class T_vload_ai<string asmStr>
  : V6_LDInst <(outs VectorRegs:$dst), (ins IntRegs:$src1, s4_6Imm:$src2),
                asmStr>;

let isCodeGenOnly = 1, addrMode = BaseImmOffset, accessSize = Vector128Access in
class T_vload_ai_128B<string asmStr>
  : V6_LDInst <(outs VectorRegs128B:$dst), (ins IntRegs:$src1, s4_7Imm:$src2),
                asmStr>;

let isCVLoadable = 1, hasNewValue = 1 in {
  def V6_vL32b_ai         : T_vload_ai <"$dst = vmem($src1+#$src2)">,
                            V6_vL32b_ai_enc;
  def V6_vL32b_nt_ai      : T_vload_ai <"$dst = vmem($src1+#$src2):nt">,
                            V6_vL32b_nt_ai_enc;
  // 128B
  def V6_vL32b_ai_128B    : T_vload_ai_128B <"$dst = vmem($src1+#$src2)">,
                            V6_vL32b_ai_128B_enc;
  def V6_vL32b_nt_ai_128B : T_vload_ai_128B <"$dst = vmem($src1+#$src2):nt">,
                            V6_vL32b_nt_ai_128B_enc;
}

let Itinerary = CVI_VM_VP_LDU, Type = TypeCVI_VM_VP_LDU, hasNewValue = 1 in {
  def V6_vL32Ub_ai      : T_vload_ai <"$dst = vmemu($src1+#$src2)">,
                          V6_vL32Ub_ai_enc;
  def V6_vL32Ub_ai_128B : T_vload_ai_128B <"$dst = vmemu($src1+#$src2)">,
                          V6_vL32Ub_ai_128B_enc;
}

let Itinerary = CVI_VM_LD, Type = TypeCVI_VM_LD, isCVLoad = 1,
    hasNewValue = 1 in {
  def V6_vL32b_cur_ai    : T_vload_ai <"$dst.cur = vmem($src1+#$src2)">,
                           V6_vL32b_cur_ai_enc;
  def V6_vL32b_nt_cur_ai : T_vload_ai <"$dst.cur = vmem($src1+#$src2):nt">,
                           V6_vL32b_nt_cur_ai_enc;
  // 128B
  def V6_vL32b_cur_ai_128B    : T_vload_ai_128B
                                <"$dst.cur = vmem($src1+#$src2)">,
                                V6_vL32b_cur_ai_128B_enc;
  def V6_vL32b_nt_cur_ai_128B : T_vload_ai_128B
                                <"$dst.cur = vmem($src1+#$src2):nt">,
                                V6_vL32b_nt_cur_ai_128B_enc;
}


let Itinerary = CVI_VM_TMP_LD, Type = TypeCVI_VM_TMP_LD, hasNewValue = 1 in {
  def V6_vL32b_tmp_ai    : T_vload_ai <"$dst.tmp = vmem($src1+#$src2)">,
                           V6_vL32b_tmp_ai_enc;
  def V6_vL32b_nt_tmp_ai : T_vload_ai <"$dst.tmp = vmem($src1+#$src2):nt">,
                           V6_vL32b_nt_tmp_ai_enc;
  // 128B
  def V6_vL32b_tmp_ai_128B    : T_vload_ai_128B
                                <"$dst.tmp = vmem($src1+#$src2)">,
                                V6_vL32b_tmp_ai_128B_enc;
  def V6_vL32b_nt_tmp_ai_128B : T_vload_ai_128B
                                <"$dst.tmp = vmem($src1+#$src2)">,
                                V6_vL32b_nt_tmp_ai_128B_enc;
}

//===----------------------------------------------------------------------===//
// Vector stores with base + immediate offset - unconditional
//===----------------------------------------------------------------------===//
let addrMode = BaseImmOffset, accessSize = Vector64Access in
class T_vstore_ai <string mnemonic, string baseOp, Operand ImmOp,
                   RegisterClass RC, bit isNT>
  : V6_STInst <(outs), (ins IntRegs:$src1, ImmOp:$src2, RC:$src3),
    mnemonic#"($src1+#$src2)"#!if(isNT, ":nt", "")#" = $src3">, NewValueRel {
  let BaseOpcode = baseOp;
}

let accessSize = Vector64Access in
class T_vstore_ai_64B <string mnemonic, string baseOp, bit isNT = 0>
  : T_vstore_ai <mnemonic, baseOp, s4_6Imm, VectorRegs, isNT>;

let isCodeGenOnly = 1, accessSize = Vector128Access in
class T_vstore_ai_128B <string mnemonic, string baseOp, bit isNT = 0>
  : T_vstore_ai <mnemonic, baseOp#"128B", s4_7Imm, VectorRegs128B, isNT>;

let isNVStorable = 1 in {
  def V6_vS32b_ai         : T_vstore_ai_64B <"vmem", "vS32b_ai">,
                            V6_vS32b_ai_enc;
  def V6_vS32b_ai_128B    : T_vstore_ai_128B <"vmem", "vS32b_ai">,
                            V6_vS32b_ai_128B_enc;
}

let isNVStorable = 1, isNonTemporal = 1 in {
  def V6_vS32b_nt_ai      : T_vstore_ai_64B <"vmem", "vS32b_ai", 1>,
                            V6_vS32b_nt_ai_enc;
  def V6_vS32b_nt_ai_128B : T_vstore_ai_128B <"vmem", "vS32b_ai", 1>,
                            V6_vS32b_nt_ai_128B_enc;
}

let Itinerary = CVI_VM_STU, Type = TypeCVI_VM_STU in {
  def V6_vS32Ub_ai      : T_vstore_ai_64B <"vmemu", "vs32Ub_ai">,
                          V6_vS32Ub_ai_enc;
  def V6_vS32Ub_ai_128B : T_vstore_ai_128B <"vmemu", "vs32Ub_ai">,
                          V6_vS32Ub_ai_128B_enc;
}
//===----------------------------------------------------------------------===//
// Vector stores with base + immediate offset - unconditional new
//===----------------------------------------------------------------------===//
let addrMode = BaseImmOffset, isNewValue = 1, opNewValue = 2, isNVStore = 1,
    Itinerary = CVI_VM_NEW_ST, Type = TypeCVI_VM_NEW_ST in
class T_vstore_new_ai <string baseOp, Operand ImmOp, RegisterClass RC, bit isNT>
  : V6_STInst <(outs ), (ins IntRegs:$src1, ImmOp:$src2, RC:$src3),
    "vmem($src1+#$src2)"#!if(isNT, ":nt", "")#" = $src3.new">, NewValueRel {
  let BaseOpcode = baseOp;
}

let accessSize = Vector64Access in
class T_vstore_new_ai_64B <string baseOp, bit isNT = 0>
  : T_vstore_new_ai <baseOp, s4_6Imm, VectorRegs, isNT>;

let isCodeGenOnly = 1, accessSize = Vector128Access in
class T_vstore_new_ai_128B <string baseOp, bit isNT = 0>
  : T_vstore_new_ai <baseOp#"128B", s4_7Imm, VectorRegs128B, isNT>;

def V6_vS32b_new_ai      : T_vstore_new_ai_64B <"vS32b_ai">, V6_vS32b_new_ai_enc;
def V6_vS32b_new_ai_128B : T_vstore_new_ai_128B <"vS32b_ai">,
                           V6_vS32b_new_ai_128B_enc;

let isNonTemporal = 1 in {
  def V6_vS32b_nt_new_ai      : T_vstore_new_ai_64B<"vS32b_ai", 1>,
                                V6_vS32b_nt_new_ai_enc;
  def V6_vS32b_nt_new_ai_128B : T_vstore_new_ai_128B<"vS32b_ai", 1>,
                                V6_vS32b_nt_new_ai_128B_enc;
}

//===----------------------------------------------------------------------===//
// Vector stores with base + immediate offset - conditional
//===----------------------------------------------------------------------===//
let addrMode = BaseImmOffset, isPredicated = 1 in
class T_vstore_pred_ai <string mnemonic, string baseOp, Operand ImmOp,
                        RegisterClass RC, bit isPredNot = 0, bit isNT = 0>
  : V6_STInst <(outs),
               (ins PredRegs:$src1, IntRegs:$src2, ImmOp:$src3, RC:$src4),
    "if ("#!if(isPredNot, "!", "")#"$src1) "
     #mnemonic#"($src2+#$src3)"#!if(isNT, ":nt", "")#" = $src4">, NewValueRel {
  let isPredicatedFalse = isPredNot;
  let BaseOpcode = baseOp;
}

let accessSize = Vector64Access in
class T_vstore_pred_ai_64B <string mnemonic, string baseOp,
                            bit isPredNot = 0, bit isNT = 0>
  : T_vstore_pred_ai <mnemonic, baseOp, s4_6Imm, VectorRegs, isPredNot, isNT>;

let isCodeGenOnly = 1, accessSize = Vector128Access in
class T_vstore_pred_ai_128B <string mnemonic, string baseOp,
                             bit isPredNot = 0, bit isNT = 0>
  : T_vstore_pred_ai <mnemonic, baseOp#"128B", s4_7Imm, VectorRegs128B,
                      isPredNot, isNT>;

let isNVStorable = 1 in {
  def V6_vS32b_pred_ai     : T_vstore_pred_ai_64B <"vmem", "vS32b_ai">,
                             V6_vS32b_pred_ai_enc;
  def V6_vS32b_npred_ai    : T_vstore_pred_ai_64B <"vmem", "vS32b_ai", 1>,
                             V6_vS32b_npred_ai_enc;
  // 128B
  def V6_vS32b_pred_ai_128B    : T_vstore_pred_ai_128B <"vmem", "vS32b_ai">,
                                 V6_vS32b_pred_ai_128B_enc;
  def V6_vS32b_npred_ai_128B   : T_vstore_pred_ai_128B <"vmem", "vS32b_ai", 1>,
                                 V6_vS32b_npred_ai_128B_enc;
}
let isNVStorable = 1, isNonTemporal = 1 in {
  def V6_vS32b_nt_pred_ai  : T_vstore_pred_ai_64B <"vmem", "vS32b_ai", 0, 1>,
                             V6_vS32b_nt_pred_ai_enc;
  def V6_vS32b_nt_npred_ai : T_vstore_pred_ai_64B <"vmem", "vS32b_ai", 1, 1>,
                             V6_vS32b_nt_npred_ai_enc;
  // 128B
  def V6_vS32b_nt_pred_ai_128B  : T_vstore_pred_ai_128B
                                  <"vmem", "vS32b_ai", 0, 1>,
                                  V6_vS32b_nt_pred_ai_128B_enc;
  def V6_vS32b_nt_npred_ai_128B : T_vstore_pred_ai_128B
                                  <"vmem", "vS32b_ai", 1, 1>,
                                  V6_vS32b_nt_npred_ai_128B_enc;
}

let Itinerary = CVI_VM_STU, Type = TypeCVI_VM_STU in {
  def V6_vS32Ub_pred_ai  : T_vstore_pred_ai_64B <"vmemu", "vS32Ub_ai">,
                           V6_vS32Ub_pred_ai_enc;
  def V6_vS32Ub_npred_ai : T_vstore_pred_ai_64B <"vmemu", "vS32Ub_ai", 1>,
                           V6_vS32Ub_npred_ai_enc;
  // 128B
  def V6_vS32Ub_pred_ai_128B  :T_vstore_pred_ai_128B <"vmemu", "vS32Ub_ai">,
                               V6_vS32Ub_pred_ai_128B_enc;
  def V6_vS32Ub_npred_ai_128B :T_vstore_pred_ai_128B <"vmemu", "vS32Ub_ai", 1>,
                               V6_vS32Ub_npred_ai_128B_enc;
}

//===----------------------------------------------------------------------===//
// Vector stores with base + immediate offset - byte-enabled aligned
//===----------------------------------------------------------------------===//
let addrMode = BaseImmOffset in
class T_vstore_qpred_ai <Operand ImmOp, RegisterClass RC,
                         bit isPredNot = 0, bit isNT = 0>
  : V6_STInst <(outs),
               (ins VecPredRegs:$src1, IntRegs:$src2, ImmOp:$src3, RC:$src4),
    "if ("#!if(isPredNot, "!", "")#"$src1) vmem($src2+#$src3)"
          #!if(isNT, ":nt", "")#" = $src4"> {
  let isPredicatedFalse = isPredNot;
}

let accessSize = Vector64Access in
class T_vstore_qpred_ai_64B <bit isPredNot = 0, bit isNT = 0>
  : T_vstore_qpred_ai <s4_6Imm, VectorRegs, isPredNot, isNT>;

let isCodeGenOnly = 1, accessSize = Vector128Access in
class T_vstore_qpred_ai_128B <bit isPredNot = 0, bit isNT = 0>
  : T_vstore_qpred_ai <s4_7Imm, VectorRegs128B, isPredNot, isNT>;

def V6_vS32b_qpred_ai  : T_vstore_qpred_ai_64B, V6_vS32b_qpred_ai_enc;
def V6_vS32b_nqpred_ai : T_vstore_qpred_ai_64B <1>,
                         V6_vS32b_nqpred_ai_enc;
def V6_vS32b_nt_qpred_ai  : T_vstore_qpred_ai_64B <0, 1>,
                            V6_vS32b_nt_qpred_ai_enc;
def V6_vS32b_nt_nqpred_ai : T_vstore_qpred_ai_64B <1, 1>,
                            V6_vS32b_nt_nqpred_ai_enc;
// 128B
def V6_vS32b_qpred_ai_128B  : T_vstore_qpred_ai_128B, V6_vS32b_qpred_ai_128B_enc;
def V6_vS32b_nqpred_ai_128B : T_vstore_qpred_ai_128B<1>,
                              V6_vS32b_nqpred_ai_128B_enc;
def V6_vS32b_nt_qpred_ai_128B  : T_vstore_qpred_ai_128B<0, 1>,
                                 V6_vS32b_nt_qpred_ai_128B_enc;
def V6_vS32b_nt_nqpred_ai_128B : T_vstore_qpred_ai_128B<1, 1>,
                                 V6_vS32b_nt_nqpred_ai_128B_enc;


//===----------------------------------------------------------------------===//
// Vector stores with base + immediate offset - conditional new
//===----------------------------------------------------------------------===//
let addrMode = BaseImmOffset, isPredicated = 1, isNewValue = 1, opNewValue = 3,
    isNVStore = 1, Type = TypeCVI_VM_NEW_ST, Itinerary = CVI_VM_NEW_ST in
class T_vstore_new_pred_ai <string baseOp, Operand ImmOp, RegisterClass RC,
                            bit isPredNot, bit isNT>
  : V6_STInst <(outs),
               (ins PredRegs:$src1, IntRegs:$src2, ImmOp:$src3, RC:$src4),
    "if("#!if(isPredNot, "!", "")#"$src1) vmem($src2+#$src3)"
         #!if(isNT, ":nt", "")#" = $src4.new">, NewValueRel {
  let isPredicatedFalse = isPredNot;
  let BaseOpcode = baseOp;
}

let accessSize = Vector64Access in
class T_vstore_new_pred_ai_64B <string baseOp, bit isPredNot = 0, bit isNT = 0>
  : T_vstore_new_pred_ai <baseOp, s4_6Imm, VectorRegs, isPredNot, isNT>;

let isCodeGenOnly = 1, accessSize = Vector128Access in
class T_vstore_new_pred_ai_128B <string baseOp, bit isPredNot = 0, bit isNT = 0>
  : T_vstore_new_pred_ai <baseOp#"128B", s4_7Imm, VectorRegs128B,
                          isPredNot, isNT>;


def V6_vS32b_new_pred_ai     : T_vstore_new_pred_ai_64B <"vS32b_ai">,
                               V6_vS32b_new_pred_ai_enc;
def V6_vS32b_new_npred_ai    : T_vstore_new_pred_ai_64B <"vS32b_ai", 1>,
                               V6_vS32b_new_npred_ai_enc;
// 128B
def V6_vS32b_new_pred_ai_128B     : T_vstore_new_pred_ai_128B <"vS32b_ai">,
                                    V6_vS32b_new_pred_ai_128B_enc;
def V6_vS32b_new_npred_ai_128B    : T_vstore_new_pred_ai_128B <"vS32b_ai", 1>,
                                    V6_vS32b_new_npred_ai_128B_enc;
let isNonTemporal = 1 in {
  def V6_vS32b_nt_new_pred_ai  : T_vstore_new_pred_ai_64B <"vS32b_ai", 0, 1>,
                                 V6_vS32b_nt_new_pred_ai_enc;
  def V6_vS32b_nt_new_npred_ai : T_vstore_new_pred_ai_64B <"vS32b_ai", 1, 1>,
                                 V6_vS32b_nt_new_npred_ai_enc;
  // 128B
  def V6_vS32b_nt_new_pred_ai_128B  : T_vstore_new_pred_ai_128B
                                      <"vS32b_ai", 0, 1>,
                                      V6_vS32b_nt_new_pred_ai_128B_enc;
  def V6_vS32b_nt_new_npred_ai_128B : T_vstore_new_pred_ai_128B
                                      <"vS32b_ai", 1, 1>,
                                      V6_vS32b_nt_new_npred_ai_128B_enc;
}

//===----------------------------------------------------------------------===//
// Post increment vector loads with immediate offset.
//===----------------------------------------------------------------------===//
let addrMode = PostInc, hasNewValue = 1 in
class T_vload_pi<string asmStr, Operand ImmOp, RegisterClass RC>
  : V6_LDInst <(outs RC:$dst, IntRegs:$_dst_),
               (ins IntRegs:$src1, ImmOp:$src2), asmStr, [],
    "$src1 = $_dst_">;

let accessSize = Vector64Access in
class T_vload_pi_64B <string asmStr>
  : T_vload_pi <asmStr, s3_6Imm, VectorRegs>;

let isCodeGenOnly = 1, accessSize = Vector128Access in
class T_vload_pi_128B <string asmStr>
  : T_vload_pi <asmStr, s3_7Imm, VectorRegs128B>;

let isCVLoadable = 1 in {
  def V6_vL32b_pi    : T_vload_pi_64B <"$dst = vmem($src1++#$src2)">,
                       V6_vL32b_pi_enc;
  def V6_vL32b_nt_pi : T_vload_pi_64B <"$dst = vmem($src1++#$src2):nt">,
                       V6_vL32b_nt_pi_enc;
  // 128B
  def V6_vL32b_pi_128B    : T_vload_pi_128B <"$dst = vmem($src1++#$src2)">,
                            V6_vL32b_pi_128B_enc;
  def V6_vL32b_nt_pi_128B : T_vload_pi_128B <"$dst = vmem($src1++#$src2):nt">,
                            V6_vL32b_nt_pi_128B_enc;
}

let Itinerary = CVI_VM_VP_LDU, Type = TypeCVI_VM_VP_LDU in {
  def V6_vL32Ub_pi : T_vload_pi_64B <"$dst = vmemu($src1++#$src2)">,
                     V6_vL32Ub_pi_enc;
  // 128B
  def V6_vL32Ub_pi_128B : T_vload_pi_128B <"$dst = vmemu($src1++#$src2)">,
                          V6_vL32Ub_pi_128B_enc;
}

let isCVLoad = 1, Itinerary = CVI_VM_LD, Type = TypeCVI_VM_LD in {
  def V6_vL32b_cur_pi    : T_vload_pi_64B <"$dst.cur = vmem($src1++#$src2)">,
                           V6_vL32b_cur_pi_enc;
  def V6_vL32b_nt_cur_pi : T_vload_pi_64B <"$dst.cur = vmem($src1++#$src2):nt">,
                           V6_vL32b_nt_cur_pi_enc;
  // 128B
  def V6_vL32b_cur_pi_128B    : T_vload_pi_128B
                                <"$dst.cur = vmem($src1++#$src2)">,
                                V6_vL32b_cur_pi_128B_enc;
  def V6_vL32b_nt_cur_pi_128B : T_vload_pi_128B
                                <"$dst.cur = vmem($src1++#$src2):nt">,
                                V6_vL32b_nt_cur_pi_128B_enc;
}

let Itinerary = CVI_VM_TMP_LD, Type = TypeCVI_VM_TMP_LD in {
  def V6_vL32b_tmp_pi    : T_vload_pi_64B <"$dst.tmp = vmem($src1++#$src2)">,
                           V6_vL32b_tmp_pi_enc;
  def V6_vL32b_nt_tmp_pi : T_vload_pi_64B <"$dst.tmp = vmem($src1++#$src2):nt">,
                           V6_vL32b_nt_tmp_pi_enc;
  //128B
  def V6_vL32b_tmp_pi_128B    : T_vload_pi_128B
                                <"$dst.tmp = vmem($src1++#$src2)">,
                                V6_vL32b_tmp_pi_128B_enc;
  def V6_vL32b_nt_tmp_pi_128B : T_vload_pi_128B
                                <"$dst.tmp = vmem($src1++#$src2):nt">,
                                V6_vL32b_nt_tmp_pi_128B_enc;
}

//===----------------------------------------------------------------------===//
// Post increment vector stores with immediate offset.
//===----------------------------------------------------------------------===//
let addrMode = PostInc in
class T_vstore_pi <string mnemonic, string baseOp, Operand ImmOp,
                   RegisterClass RC, bit isNT>
  : V6_STInst <(outs IntRegs:$_dst_),
               (ins IntRegs:$src1, ImmOp:$src2, RC:$src3),
    mnemonic#"($src1++#$src2)"#!if(isNT, ":nt", "")#" = $src3", [],
    "$src1 = $_dst_">, NewValueRel;

let accessSize = Vector64Access in
class T_vstore_pi_64B <string mnemonic, string baseOp, bit isNT = 0>
  : T_vstore_pi <mnemonic, baseOp, s3_6Imm, VectorRegs, isNT>;

let isCodeGenOnly = 1, accessSize = Vector128Access in
class T_vstore_pi_128B <string mnemonic, string baseOp, bit isNT = 0>
  : T_vstore_pi <mnemonic, baseOp, s3_7Imm, VectorRegs128B, isNT>;

let isNVStorable = 1 in {
  def V6_vS32b_pi      : T_vstore_pi_64B <"vmem", "vS32b_pi">, V6_vS32b_pi_enc;
  def V6_vS32b_pi_128B : T_vstore_pi_128B <"vmem", "vS32b_pi">,
                         V6_vS32b_pi_128B_enc;
}

let isNVStorable = 1 , isNonTemporal = 1  in {
  def V6_vS32b_nt_pi      : T_vstore_pi_64B <"vmem", "vS32b_pi", 1>,
                            V6_vS32b_nt_pi_enc;
  def V6_vS32b_nt_pi_128B : T_vstore_pi_128B <"vmem", "vS32b_pi", 1>,
                            V6_vS32b_nt_pi_128B_enc;
}


let Itinerary = CVI_VM_STU, Type = TypeCVI_VM_STU in {
  def V6_vS32Ub_pi      : T_vstore_pi_64B <"vmemu", "vS32Ub_pi">,
                          V6_vS32Ub_pi_enc;
  def V6_vS32Ub_pi_128B : T_vstore_pi_128B <"vmemu", "vS32Ub_pi">,
                          V6_vS32Ub_pi_128B_enc;
}

//===----------------------------------------------------------------------===//
// Post increment unconditional .new vector stores with immediate offset.
//===----------------------------------------------------------------------===//
let addrMode = PostInc, isNVStore = 1 in
let Itinerary = CVI_VM_NEW_ST, Type = TypeCVI_VM_NEW_ST, isNewValue = 1,
    opNewValue = 3, isNVStore = 1 in
class T_vstore_new_pi <string baseOp, Operand ImmOp, RegisterClass RC, bit isNT>
  : V6_STInst <(outs IntRegs:$_dst_),
               (ins IntRegs:$src1, ImmOp:$src2, RC:$src3),
    "vmem($src1++#$src2)"#!if(isNT, ":nt", "")#" = $src3.new", [],
    "$src1 = $_dst_">, NewValueRel {
  let BaseOpcode = baseOp;
}

let accessSize = Vector64Access in
class T_vstore_new_pi_64B <string baseOp, bit isNT = 0>
  : T_vstore_new_pi <baseOp, s3_6Imm, VectorRegs, isNT>;

let isCodeGenOnly = 1, accessSize = Vector128Access in
class T_vstore_new_pi_128B <string baseOp, bit isNT = 0>
  : T_vstore_new_pi <baseOp#"128B", s3_7Imm, VectorRegs128B, isNT>;


def V6_vS32b_new_pi      : T_vstore_new_pi_64B <"vS32b_pi">,
                           V6_vS32b_new_pi_enc;
def V6_vS32b_new_pi_128B : T_vstore_new_pi_128B <"vS32b_pi">,
                           V6_vS32b_new_pi_128B_enc;

let isNonTemporal = 1 in {
  def V6_vS32b_nt_new_pi      : T_vstore_new_pi_64B <"vS32b_pi", 1>,
                                V6_vS32b_nt_new_pi_enc;
  def V6_vS32b_nt_new_pi_128B : T_vstore_new_pi_128B <"vS32b_pi", 1>,
                                V6_vS32b_nt_new_pi_128B_enc;
}

//===----------------------------------------------------------------------===//
// Post increment conditional vector stores with immediate offset
//===----------------------------------------------------------------------===//
let isPredicated = 1, addrMode = PostInc in
class T_vstore_pred_pi <string mnemonic, string baseOp, Operand ImmOp,
                        RegisterClass RC, bit isPredNot, bit isNT>
  : V6_STInst<(outs IntRegs:$_dst_),
             (ins PredRegs:$src1, IntRegs:$src2, ImmOp:$src3, RC:$src4),
    "if ("#!if(isPredNot, "!", "")#"$src1) "#mnemonic#"($src2++#$src3)"
          #!if(isNT, ":nt", "")#" = $src4", [],
    "$src2 = $_dst_">, NewValueRel {
  let isPredicatedFalse = isPredNot;
  let BaseOpcode = baseOp;
}

let accessSize = Vector64Access in
class T_vstore_pred_pi_64B <string mnemonic, string baseOp,
                            bit isPredNot = 0, bit isNT = 0>
  : T_vstore_pred_pi <mnemonic, baseOp, s3_6Imm, VectorRegs, isPredNot, isNT>;

let isCodeGenOnly = 1, accessSize = Vector128Access in
class T_vstore_pred_pi_128B <string mnemonic, string baseOp,
                             bit isPredNot = 0, bit isNT = 0>
  : T_vstore_pred_pi <mnemonic, baseOp#"128B", s3_7Imm, VectorRegs128B,
                      isPredNot, isNT>;

let isNVStorable = 1 in {
  def V6_vS32b_pred_pi     : T_vstore_pred_pi_64B <"vmem", "vS32b_pi">,
                             V6_vS32b_pred_pi_enc;
  def V6_vS32b_npred_pi    : T_vstore_pred_pi_64B <"vmem", "vS32b_pi", 1>,
                             V6_vS32b_npred_pi_enc;
  // 128B
  def V6_vS32b_pred_pi_128B  : T_vstore_pred_pi_128B <"vmem", "vS32b_pi">,
                               V6_vS32b_pred_pi_128B_enc;
  def V6_vS32b_npred_pi_128B : T_vstore_pred_pi_128B <"vmem", "vS32b_pi", 1>,
                               V6_vS32b_npred_pi_128B_enc;
}
let isNVStorable = 1, isNonTemporal = 1 in {
  def V6_vS32b_nt_pred_pi  : T_vstore_pred_pi_64B <"vmem", "vS32b_pi", 0, 1>,
                             V6_vS32b_nt_pred_pi_enc;
  def V6_vS32b_nt_npred_pi : T_vstore_pred_pi_64B <"vmem", "vS32b_pi", 1, 1>,
                             V6_vS32b_nt_npred_pi_enc;
  // 128B
  def V6_vS32b_nt_pred_pi_128B  : T_vstore_pred_pi_128B
                                  <"vmem", "vS32b_pi", 0, 1>,
                                  V6_vS32b_nt_pred_pi_128B_enc;
  def V6_vS32b_nt_npred_pi_128B : T_vstore_pred_pi_128B
                                  <"vmem", "vS32b_pi", 1, 1>,
                                  V6_vS32b_nt_npred_pi_128B_enc;
}

let Itinerary = CVI_VM_STU, Type = TypeCVI_VM_STU in {
  def V6_vS32Ub_pred_pi  : T_vstore_pred_pi_64B <"vmemu", "vS32Ub_pi">,
                           V6_vS32Ub_pred_pi_enc;
  def V6_vS32Ub_npred_pi : T_vstore_pred_pi_64B <"vmemu", "vS32Ub_pi", 1>,
                           V6_vS32Ub_npred_pi_enc;
  // 128B
  def V6_vS32Ub_pred_pi_128B  : T_vstore_pred_pi_128B <"vmemu", "vS32Ub_pi">,
                                V6_vS32Ub_pred_pi_128B_enc;
  def V6_vS32Ub_npred_pi_128B : T_vstore_pred_pi_128B <"vmemu", "vS32Ub_pi", 1>,
                                V6_vS32Ub_npred_pi_128B_enc;
}

//===----------------------------------------------------------------------===//
// Post increment vector stores with immediate offset - byte-enabled aligned
//===----------------------------------------------------------------------===//
let addrMode = PostInc in
class T_vstore_qpred_pi <Operand ImmOp, RegisterClass RC, bit isPredNot = 0,
                         bit isNT = 0>
  : V6_STInst <(outs IntRegs:$_dst_),
               (ins VecPredRegs:$src1, IntRegs:$src2, ImmOp:$src3, RC:$src4),
    "if ("#!if(isPredNot, "!", "")#"$src1) vmem($src2++#$src3)"
          #!if(isNT, ":nt", "")#" = $src4", [],
    "$src2 = $_dst_">;

let accessSize = Vector64Access in
class T_vstore_qpred_pi_64B <bit isPredNot = 0, bit isNT = 0>
  : T_vstore_qpred_pi <s3_6Imm, VectorRegs, isPredNot, isNT>;

let isCodeGenOnly = 1, accessSize = Vector128Access in
class T_vstore_qpred_pi_128B <bit isPredNot = 0, bit isNT = 0>
  : T_vstore_qpred_pi <s3_7Imm, VectorRegs128B, isPredNot, isNT>;

def V6_vS32b_qpred_pi  : T_vstore_qpred_pi_64B, V6_vS32b_qpred_pi_enc;
def V6_vS32b_nqpred_pi : T_vstore_qpred_pi_64B <1>, V6_vS32b_nqpred_pi_enc;
// 128B
def V6_vS32b_qpred_pi_128B  : T_vstore_qpred_pi_128B,
                              V6_vS32b_qpred_pi_128B_enc;
def V6_vS32b_nqpred_pi_128B : T_vstore_qpred_pi_128B<1>,
                              V6_vS32b_nqpred_pi_128B_enc;

let isNonTemporal = 1 in {
  def V6_vS32b_nt_qpred_pi  : T_vstore_qpred_pi_64B <0, 1>,
                              V6_vS32b_nt_qpred_pi_enc;
  def V6_vS32b_nt_nqpred_pi : T_vstore_qpred_pi_64B <1, 1>,
                              V6_vS32b_nt_nqpred_pi_enc;
  // 128B
  def V6_vS32b_nt_qpred_pi_128B  : T_vstore_qpred_pi_128B<0, 1>,
                                   V6_vS32b_nt_qpred_pi_128B_enc;
  def V6_vS32b_nt_nqpred_pi_128B : T_vstore_qpred_pi_128B<1, 1>,
                                   V6_vS32b_nt_nqpred_pi_128B_enc;
}

//===----------------------------------------------------------------------===//
// Post increment conditional .new vector stores with immediate offset
//===----------------------------------------------------------------------===//
let Itinerary = CVI_VM_NEW_ST, Type = TypeCVI_VM_NEW_ST, isPredicated = 1,
    isNewValue = 1, opNewValue = 4, addrMode = PostInc, isNVStore = 1 in
class T_vstore_new_pred_pi <string baseOp, Operand ImmOp, RegisterClass RC,
                            bit isPredNot, bit isNT>
  : V6_STInst <(outs IntRegs:$_dst_),
               (ins PredRegs:$src1, IntRegs:$src2, ImmOp:$src3, RC:$src4),
    "if("#!if(isPredNot, "!", "")#"$src1) vmem($src2++#$src3)"
         #!if(isNT, ":nt", "")#" = $src4.new", [],
    "$src2 = $_dst_"> , NewValueRel {
  let isPredicatedFalse = isPredNot;
  let BaseOpcode = baseOp;
}

let accessSize = Vector64Access in
class T_vstore_new_pred_pi_64B <string baseOp, bit isPredNot = 0, bit isNT = 0>
  : T_vstore_new_pred_pi <baseOp, s3_6Imm, VectorRegs, isPredNot, isNT>;

let isCodeGenOnly = 1, accessSize = Vector128Access in
class T_vstore_new_pred_pi_128B <string baseOp, bit isPredNot = 0, bit isNT = 0>
  : T_vstore_new_pred_pi <baseOp#"128B", s3_7Imm, VectorRegs128B,
                          isPredNot, isNT>;

def V6_vS32b_new_pred_pi     : T_vstore_new_pred_pi_64B <"vS32b_pi">,
                               V6_vS32b_new_pred_pi_enc;
def V6_vS32b_new_npred_pi    : T_vstore_new_pred_pi_64B <"vS32b_pi", 1>,
                               V6_vS32b_new_npred_pi_enc;
// 128B
def V6_vS32b_new_pred_pi_128B    : T_vstore_new_pred_pi_128B <"vS32b_pi">,
                                   V6_vS32b_new_pred_pi_128B_enc;
def V6_vS32b_new_npred_pi_128B   : T_vstore_new_pred_pi_128B <"vS32b_pi", 1>,
                                   V6_vS32b_new_npred_pi_128B_enc;
let isNonTemporal = 1 in {
  def V6_vS32b_nt_new_pred_pi  : T_vstore_new_pred_pi_64B <"vS32b_pi", 0, 1>,
                                 V6_vS32b_nt_new_pred_pi_enc;
  def V6_vS32b_nt_new_npred_pi : T_vstore_new_pred_pi_64B <"vS32b_pi", 1, 1>,
                                 V6_vS32b_nt_new_npred_pi_enc;
  // 128B
  def V6_vS32b_nt_new_pred_pi_128B : T_vstore_new_pred_pi_128B
                                     <"vS32b_pi", 0, 1>,
                                     V6_vS32b_nt_new_pred_pi_128B_enc;
  def V6_vS32b_nt_new_npred_pi_128B : T_vstore_new_pred_pi_128B
                                      <"vS32b_pi", 1, 1>,
                                      V6_vS32b_nt_new_npred_pi_128B_enc;
}

//===----------------------------------------------------------------------===//
// Post increment vector loads with register offset
//===----------------------------------------------------------------------===//
let hasNewValue = 1 in
class T_vload_ppu<string asmStr>
  : V6_LDInst <(outs VectorRegs:$dst, IntRegs:$_dst_),
               (ins IntRegs:$src1, ModRegs:$src2), asmStr, [],
    "$src1 = $_dst_">, NewValueRel;

let isCVLoadable = 1 in {
  def V6_vL32b_ppu    : T_vload_ppu <"$dst = vmem($src1++$src2)">,
                        V6_vL32b_ppu_enc;
  def V6_vL32b_nt_ppu : T_vload_ppu <"$dst = vmem($src1++$src2):nt">,
                        V6_vL32b_nt_ppu_enc;
}

let Itinerary = CVI_VM_VP_LDU, Type = TypeCVI_VM_VP_LDU in
def V6_vL32Ub_ppu : T_vload_ppu <"$dst = vmemu($src1++$src2)">,
                     V6_vL32Ub_ppu_enc;

let isCVLoad = 1, Itinerary = CVI_VM_CUR_LD, Type = TypeCVI_VM_CUR_LD in {
  def V6_vL32b_cur_ppu    : T_vload_ppu <"$dst.cur = vmem($src1++$src2)">,
                             V6_vL32b_cur_ppu_enc;
  def V6_vL32b_nt_cur_ppu : T_vload_ppu <"$dst.cur = vmem($src1++$src2):nt">,
                             V6_vL32b_nt_cur_ppu_enc;
}

let Itinerary = CVI_VM_TMP_LD, Type = TypeCVI_VM_TMP_LD in {
  def V6_vL32b_tmp_ppu    : T_vload_ppu <"$dst.tmp = vmem($src1++$src2)">,
                             V6_vL32b_tmp_ppu_enc;
  def V6_vL32b_nt_tmp_ppu : T_vload_ppu <"$dst.tmp = vmem($src1++$src2):nt">,
                             V6_vL32b_nt_tmp_ppu_enc;
}

//===----------------------------------------------------------------------===//
// Post increment vector stores with register offset
//===----------------------------------------------------------------------===//
class T_vstore_ppu <string mnemonic, bit isNT = 0>
  : V6_STInst <(outs IntRegs:$_dst_),
               (ins IntRegs:$src1, ModRegs:$src2, VectorRegs:$src3),
    mnemonic#"($src1++$src2)"#!if(isNT, ":nt", "")#" = $src3", [],
    "$src1 = $_dst_">, NewValueRel;

let isNVStorable = 1, BaseOpcode = "vS32b_ppu" in {
  def V6_vS32b_ppu    : T_vstore_ppu <"vmem">,
                        V6_vS32b_ppu_enc;
  let isNonTemporal = 1, BaseOpcode = "vS32b_ppu" in
  def V6_vS32b_nt_ppu : T_vstore_ppu <"vmem", 1>,
                        V6_vS32b_nt_ppu_enc;
}

let BaseOpcode = "vS32Ub_ppu", Itinerary = CVI_VM_STU, Type = TypeCVI_VM_STU in
def V6_vS32Ub_ppu   : T_vstore_ppu <"vmemu">, V6_vS32Ub_ppu_enc;

//===----------------------------------------------------------------------===//
// Post increment .new vector stores with register offset
//===----------------------------------------------------------------------===//
let Itinerary = CVI_VM_NEW_ST, Type = TypeCVI_VM_NEW_ST, isNewValue = 1,
    opNewValue = 3, isNVStore = 1 in
class T_vstore_new_ppu <bit isNT = 0>
  : V6_STInst <(outs IntRegs:$_dst_),
               (ins IntRegs:$src1, ModRegs:$src2, VectorRegs:$src3),
    "vmem($src1++$src2)"#!if(isNT, ":nt", "")#" = $src3.new", [],
    "$src1 = $_dst_">, NewValueRel;

let BaseOpcode = "vS32b_ppu" in
def V6_vS32b_new_ppu    : T_vstore_new_ppu, V6_vS32b_new_ppu_enc;

let BaseOpcode = "vS32b_ppu", isNonTemporal = 1 in
def V6_vS32b_nt_new_ppu : T_vstore_new_ppu<1>, V6_vS32b_nt_new_ppu_enc;

//===----------------------------------------------------------------------===//
// Post increment conditional .new vector stores with register offset
//===----------------------------------------------------------------------===//
let isPredicated = 1 in
class T_vstore_pred_ppu <string mnemonic, bit isPredNot = 0, bit isNT = 0>
  : V6_STInst<(outs IntRegs:$_dst_),
           (ins PredRegs:$src1, IntRegs:$src2, ModRegs:$src3, VectorRegs:$src4),
    "if ("#!if(isPredNot, "!", "")#"$src1) "#mnemonic#"($src2++$src3)"
          #!if(isNT, ":nt", "")#" = $src4", [],
    "$src2 = $_dst_">, NewValueRel {
  let isPredicatedFalse = isPredNot;
}

let isNVStorable = 1, BaseOpcode = "vS32b_ppu" in {
  def V6_vS32b_pred_ppu : T_vstore_pred_ppu<"vmem">, V6_vS32b_pred_ppu_enc;
  def V6_vS32b_npred_ppu: T_vstore_pred_ppu<"vmem", 1>, V6_vS32b_npred_ppu_enc;
}

let isNVStorable = 1, BaseOpcode = "vS32b_ppu", isNonTemporal = 1 in {
  def V6_vS32b_nt_pred_ppu  : T_vstore_pred_ppu <"vmem", 0, 1>,
                              V6_vS32b_nt_pred_ppu_enc;
  def V6_vS32b_nt_npred_ppu : T_vstore_pred_ppu <"vmem", 1, 1>,
                              V6_vS32b_nt_npred_ppu_enc;
}

let BaseOpcode = "vS32Ub_ppu", Itinerary = CVI_VM_STU,
    Type = TypeCVI_VM_STU in {
  def V6_vS32Ub_pred_ppu  : T_vstore_pred_ppu <"vmemu">,
                            V6_vS32Ub_pred_ppu_enc;
  def V6_vS32Ub_npred_ppu : T_vstore_pred_ppu <"vmemu", 1>,
                            V6_vS32Ub_npred_ppu_enc;
}

//===----------------------------------------------------------------------===//
// Post increment vector stores with register offset - byte-enabled aligned
//===----------------------------------------------------------------------===//
class T_vstore_qpred_ppu <bit isPredNot = 0, bit isNT = 0>
  : V6_STInst <(outs IntRegs:$_dst_),
        (ins VecPredRegs:$src1, IntRegs:$src2, ModRegs:$src3, VectorRegs:$src4),
    "if ("#!if(isPredNot, "!", "")#"$src1) vmem($src2++$src3)"
          #!if(isNT, ":nt", "")#" = $src4", [],
    "$src2 = $_dst_">, NewValueRel;

def V6_vS32b_qpred_ppu  : T_vstore_qpred_ppu, V6_vS32b_qpred_ppu_enc;
def V6_vS32b_nqpred_ppu : T_vstore_qpred_ppu<1>, V6_vS32b_nqpred_ppu_enc;
def V6_vS32b_nt_qpred_ppu  : T_vstore_qpred_ppu<0, 1>,
                             V6_vS32b_nt_qpred_ppu_enc;
def V6_vS32b_nt_nqpred_ppu : T_vstore_qpred_ppu<1, 1>,
                             V6_vS32b_nt_nqpred_ppu_enc;

//===----------------------------------------------------------------------===//
// Post increment conditional .new vector stores with register offset
//===----------------------------------------------------------------------===//
let Itinerary = CVI_VM_NEW_ST, Type = TypeCVI_VM_NEW_ST, isPredicated = 1,
    isNewValue = 1, opNewValue = 4, isNVStore = 1 in
class T_vstore_new_pred_ppu <bit isPredNot = 0, bit isNT = 0>
  : V6_STInst <(outs IntRegs:$_dst_),
           (ins PredRegs:$src1, IntRegs:$src2, ModRegs:$src3, VectorRegs:$src4),
    "if("#!if(isPredNot, "!", "")#"$src1) vmem($src2++$src3)"
         #!if(isNT, ":nt", "")#" = $src4.new", [],
    "$src2 = $_dst_">, NewValueRel {
  let isPredicatedFalse = isPredNot;
}

let BaseOpcode = "vS32b_ppu" in {
  def V6_vS32b_new_pred_ppu  : T_vstore_new_pred_ppu,
                               V6_vS32b_new_pred_ppu_enc;
  def V6_vS32b_new_npred_ppu : T_vstore_new_pred_ppu<1>,
                               V6_vS32b_new_npred_ppu_enc;
}

let BaseOpcode = "vS32b_ppu", isNonTemporal = 1 in {
def V6_vS32b_nt_new_pred_ppu :  T_vstore_new_pred_ppu<0, 1>,
                                V6_vS32b_nt_new_pred_ppu_enc;
def V6_vS32b_nt_new_npred_ppu : T_vstore_new_pred_ppu<1, 1>,
                                V6_vS32b_nt_new_npred_ppu_enc;
}

let isPseudo = 1, validSubTargets = HasV60SubT in
class STrivv_template<string mnemonic, Operand ImmOp, RegisterClass RC>:
        VSTInst<(outs), (ins IntRegs:$addr, ImmOp:$off, RC:$src),
                #mnemonic#"($addr+#$off) = $src", []>;

def STrivv_indexed: STrivv_template<"vvmem", s4_6Imm, VecDblRegs>,
                    Requires<[HasV60T, UseHVXSgl]>;
def STrivv_indexed_128B: STrivv_template<"vvmem", s4_7Imm, VecDblRegs128B>,
                         Requires<[HasV60T, UseHVXDbl]>;

multiclass STrivv_pats <ValueType VTSgl, ValueType VTDbl> {
  def : Pat<(store (VTSgl VecDblRegs:$src1), IntRegs:$addr),
            (STrivv_indexed IntRegs:$addr, #0, (VTSgl VecDblRegs:$src1))>,
            Requires<[UseHVXSgl]>;

  def : Pat<(store (VTDbl VecDblRegs128B:$src1), IntRegs:$addr),
            (STrivv_indexed_128B IntRegs:$addr, #0,
                                 (VTDbl VecDblRegs128B:$src1))>,
            Requires<[UseHVXDbl]>;
}

defm : STrivv_pats <v128i8, v256i8>;
defm : STrivv_pats <v64i16, v128i16>;
defm : STrivv_pats <v32i32, v64i32>;
defm : STrivv_pats <v16i64, v32i64>;


multiclass vS32b_ai_pats <ValueType VTSgl, ValueType VTDbl> {
  // Aligned stores
  def : Pat<(store (VTSgl VectorRegs:$src1), IntRegs:$addr),
            (V6_vS32b_ai IntRegs:$addr, #0, (VTSgl VectorRegs:$src1))>,
            Requires<[UseHVXSgl]>;

  // 128B Aligned stores
  def : Pat<(store (VTDbl VectorRegs128B:$src1), IntRegs:$addr),
            (V6_vS32b_ai_128B IntRegs:$addr, #0, (VTDbl VectorRegs128B:$src1))>,
            Requires<[UseHVXDbl]>;

  // Fold Add R+IFF into vector store.
  let AddedComplexity = 10 in
  def : Pat<(store (VTSgl VectorRegs:$src1),
                   (add IntRegs:$src2, s4_6ImmPred:$offset)),
            (V6_vS32b_ai IntRegs:$src2, s4_6ImmPred:$offset,
                         (VTSgl VectorRegs:$src1))>,
            Requires<[UseHVXSgl]>;

  // Fold Add R+IFF into vector store 128B.
  let AddedComplexity = 10 in
  def : Pat<(store (VTDbl VectorRegs128B:$src1),
                   (add IntRegs:$src2, s4_7ImmPred:$offset)),
            (V6_vS32b_ai_128B IntRegs:$src2, s4_7ImmPred:$offset,
                              (VTDbl VectorRegs128B:$src1))>,
            Requires<[UseHVXDbl]>;
}

defm : vS32b_ai_pats <v64i8,  v128i8>;
defm : vS32b_ai_pats <v32i16, v64i16>;
defm : vS32b_ai_pats <v16i32, v32i32>;
defm : vS32b_ai_pats <v8i64,  v16i64>;

let isPseudo = 1, validSubTargets = HasV60SubT in
class LDrivv_template<string mnemonic, Operand ImmOp, RegisterClass RC>
  : V6_LDInst <(outs RC:$dst), (ins IntRegs:$addr, ImmOp:$off),
               "$dst="#mnemonic#"($addr+#$off)",
               []>,
               Requires<[HasV60T,UseHVXSgl]>;

def LDrivv_indexed: LDrivv_template<"vvmem", s4_6Imm, VecDblRegs>;
def LDrivv_indexed_128B: LDrivv_template<"vvmem", s4_7Imm, VecDblRegs128B>;

multiclass LDrivv_pats <ValueType VTSgl, ValueType VTDbl> {
  def : Pat < (VTSgl (load IntRegs:$addr)),
              (LDrivv_indexed IntRegs:$addr, #0) >,
              Requires<[UseHVXSgl]>;

  def : Pat < (VTDbl (load IntRegs:$addr)),
              (LDrivv_indexed_128B IntRegs:$addr, #0) >,
              Requires<[UseHVXDbl]>;
}

defm : LDrivv_pats <v128i8, v256i8>;
defm : LDrivv_pats <v64i16, v128i16>;
defm : LDrivv_pats <v32i32, v64i32>;
defm : LDrivv_pats <v16i64, v32i64>;

multiclass vL32b_ai_pats <ValueType VTSgl, ValueType VTDbl> {
  // Aligned loads
  def : Pat < (VTSgl (load IntRegs:$addr)),
              (V6_vL32b_ai IntRegs:$addr, #0) >,
              Requires<[UseHVXSgl]>;

  // 128B Load
  def : Pat < (VTDbl (load IntRegs:$addr)),
              (V6_vL32b_ai_128B IntRegs:$addr, #0) >,
              Requires<[UseHVXDbl]>;

  // Fold Add R+IFF into vector load.
  let AddedComplexity = 10 in
  def : Pat<(VTDbl (load (add IntRegs:$src2, s4_7ImmPred:$offset))),
            (V6_vL32b_ai_128B IntRegs:$src2, s4_7ImmPred:$offset)>,
             Requires<[UseHVXDbl]>;

  let AddedComplexity = 10 in
  def : Pat<(VTSgl (load (add IntRegs:$src2, s4_6ImmPred:$offset))),
            (V6_vL32b_ai IntRegs:$src2, s4_6ImmPred:$offset)>,
            Requires<[UseHVXSgl]>;
}

defm : vL32b_ai_pats <v64i8,  v128i8>;
defm : vL32b_ai_pats <v32i16, v64i16>;
defm : vL32b_ai_pats <v16i32, v32i32>;
defm : vL32b_ai_pats <v8i64,  v16i64>;

// Store vector predicate pseudo.
let isExtendable = 1, opExtendable = 1, isExtentSigned = 1, opExtentBits = 13,
    isCodeGenOnly = 1, isPseudo = 1, mayStore = 1, hasSideEffects = 0 in {
def STriq_pred_V6 : STInst<(outs),
            (ins IntRegs:$base, s32Imm:$offset, VecPredRegs:$src1),
            ".error \"should not emit\" ",
            []>,
            Requires<[HasV60T,UseHVXSgl]>;

def STriq_pred_vec_V6 : STInst<(outs),
            (ins IntRegs:$base, s32Imm:$offset, VectorRegs:$src1),
            ".error \"should not emit\" ",
            []>,
            Requires<[HasV60T,UseHVXSgl]>;

def STriq_pred_V6_128B : STInst<(outs),
            (ins IntRegs:$base, s32Imm:$offset, VecPredRegs128B:$src1),
            ".error \"should not emit\" ",
            []>,
            Requires<[HasV60T,UseHVXDbl]>;

def STriq_pred_vec_V6_128B : STInst<(outs),
            (ins IntRegs:$base, s32Imm:$offset, VectorRegs128B:$src1),
            ".error \"should not emit\" ",
            []>,
            Requires<[HasV60T,UseHVXDbl]>;
}

// Load vector predicate pseudo.
let isExtendable = 1, opExtendable = 2, isExtentSigned = 1, opExtentBits = 13,
    opExtentAlign = 2, isCodeGenOnly = 1, isPseudo = 1, hasSideEffects = 0 in {
def LDriq_pred_V6 : LDInst<(outs VecPredRegs:$dst),
            (ins IntRegs:$base, s32Imm:$offset),
            ".error \"should not emit\" ",
            []>,
            Requires<[HasV60T,UseHVXSgl]>;
def LDriq_pred_vec_V6 : LDInst<(outs VectorRegs:$dst),
            (ins IntRegs:$base, s32Imm:$offset),
            ".error \"should not emit\" ",
            []>,
            Requires<[HasV60T,UseHVXSgl]>;
def LDriq_pred_V6_128B : LDInst<(outs VecPredRegs128B:$dst),
            (ins IntRegs:$base, s32Imm:$offset),
            ".error \"should not emit\" ",
            []>,
            Requires<[HasV60T,UseHVXDbl]>;
def LDriq_pred_vec_V6_128B : LDInst<(outs VectorRegs128B:$dst),
            (ins IntRegs:$base, s32Imm:$offset),
            ".error \"should not emit\" ",
            []>,
            Requires<[HasV60T,UseHVXDbl]>;
}

// Store vector pseudo.
let isExtendable = 1, opExtendable = 1, isExtentSigned = 1, opExtentBits = 13,
    isCodeGenOnly = 1, isPseudo = 1, mayStore = 1, hasSideEffects = 0 in {
def STriv_pseudo_V6 : STInst<(outs),
            (ins IntRegs:$base, s32Imm:$offset, VectorRegs:$src1),
            ".error \"should not emit\" ",
            []>,
            Requires<[HasV60T,UseHVXSgl]>;
def STriv_pseudo_V6_128B : STInst<(outs),
            (ins IntRegs:$base, s32Imm:$offset, VectorRegs128B:$src1),
            ".error \"should not emit\" ",
            []>,
            Requires<[HasV60T,UseHVXDbl]>;
}

let isExtendable = 1, opExtendable = 1, isExtentSigned = 1, opExtentBits = 13,
    isCodeGenOnly = 1, isPseudo = 1, mayStore = 1, hasSideEffects = 0 in {
def STrivv_pseudo_V6 : STInst<(outs),
            (ins IntRegs:$base, s32Imm:$offset, VecDblRegs:$src1),
            ".error \"should not emit\" ",
            []>,
            Requires<[HasV60T,UseHVXSgl]>;
def STrivv_pseudo_V6_128B : STInst<(outs),
            (ins IntRegs:$base, s32Imm:$offset, VecDblRegs128B:$src1),
            ".error \"should not emit\" ",
            []>,
            Requires<[HasV60T,UseHVXDbl]>;
}

// Load vector pseudo.
let isExtendable = 1, opExtendable = 2, isExtentSigned = 1, opExtentBits = 13,
    opExtentAlign = 2, isCodeGenOnly = 1, isPseudo = 1, hasSideEffects = 0 in {
def LDriv_pseudo_V6 : LDInst<(outs VectorRegs:$dst),
            (ins IntRegs:$base, s32Imm:$offset),
            ".error \"should not emit\" ",
            []>,
            Requires<[HasV60T,UseHVXSgl]>;
def LDriv_pseudo_V6_128B : LDInst<(outs VectorRegs128B:$dst),
            (ins IntRegs:$base, s32Imm:$offset),
            ".error \"should not emit\" ",
            []>,
            Requires<[HasV60T,UseHVXDbl]>;
}

let isExtendable = 1, opExtendable = 2, isExtentSigned = 1, opExtentBits = 13,
    opExtentAlign = 2, isCodeGenOnly = 1, isPseudo = 1, hasSideEffects = 0 in {
def LDrivv_pseudo_V6 : LDInst<(outs VecDblRegs:$dst),
            (ins IntRegs:$base, s32Imm:$offset),
            ".error \"should not emit\" ",
            []>,
            Requires<[HasV60T,UseHVXSgl]>;
def LDrivv_pseudo_V6_128B : LDInst<(outs VecDblRegs128B:$dst),
            (ins IntRegs:$base, s32Imm:$offset),
            ".error \"should not emit\" ",
            []>,
            Requires<[HasV60T,UseHVXDbl]>;
}

class VSELInst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
              string cstr = "", InstrItinClass itin = CVI_VA_DV,
              IType type = TypeCVI_VA_DV>
  : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, type>;

let isCodeGenOnly = 1, isPseudo = 1, hasSideEffects = 0 in {
def VSelectPseudo_V6 : VSELInst<(outs VectorRegs:$dst),
            (ins PredRegs:$src1, VectorRegs:$src2, VectorRegs:$src3),
            ".error \"should not emit\" ",
            []>,
            Requires<[HasV60T,UseHVXSgl]>;
def VSelectDblPseudo_V6 : VSELInst<(outs VecDblRegs:$dst),
               (ins PredRegs:$src1, VecDblRegs:$src2, VecDblRegs:$src3),
               ".error \"should not emit\" ",
               []>,
               Requires<[HasV60T,UseHVXSgl]>;
}

def : Pat <(v16i32 (selectcc (i32 IntRegs:$lhs), (i32 IntRegs:$rhs),
                             (v16i32 VectorRegs:$tval),
                             (v16i32 VectorRegs:$fval), SETEQ)),
      (v16i32 (VSelectPseudo_V6 (i32 (C2_cmpeq (i32 IntRegs:$lhs),
                                (i32 IntRegs:$rhs))),
                                (v16i32 VectorRegs:$tval),
                                (v16i32 VectorRegs:$fval)))>;


let hasNewValue = 1 in
class T_vmpy <string asmString, RegisterClass RCout, RegisterClass RCin>
  : CVI_VX_DV_Resource1<(outs RCout:$dst), (ins RCin:$src1, IntRegs:$src2),
    asmString >;

multiclass T_vmpy <string asmString, RegisterClass RCout,
                        RegisterClass RCin> {
  def NAME : T_vmpy <asmString, RCout, RCin>;
  let isCodeGenOnly = 1 in
  def NAME#_128B : T_vmpy <asmString, !cast<RegisterClass>(RCout#"128B"),
                                      !cast<RegisterClass>(RCin#"128B")>;
}

multiclass T_vmpy_VV <string asmString>:
  T_vmpy <asmString, VectorRegs, VectorRegs>;

multiclass T_vmpy_WW <string asmString>:
  T_vmpy <asmString, VecDblRegs, VecDblRegs>;

multiclass T_vmpy_VW <string asmString>:
  T_vmpy <asmString, VectorRegs, VecDblRegs>;

multiclass T_vmpy_WV <string asmString>:
  T_vmpy <asmString, VecDblRegs, VectorRegs>;

defm V6_vtmpyb   :T_vmpy_WW<"$dst.h = vtmpy($src1.b,$src2.b)">, V6_vtmpyb_enc;
defm V6_vtmpybus :T_vmpy_WW<"$dst.h = vtmpy($src1.ub,$src2.b)">, V6_vtmpybus_enc;
defm V6_vdsaduh  :T_vmpy_WW<"$dst.uw = vdsad($src1.uh,$src2.uh)">, V6_vdsaduh_enc;
defm V6_vmpybus  :T_vmpy_WV<"$dst.h = vmpy($src1.ub,$src2.b)">, V6_vmpybus_enc;
defm V6_vmpabus  :T_vmpy_WW<"$dst.h = vmpa($src1.ub,$src2.b)">, V6_vmpabus_enc;
defm V6_vmpahb   :T_vmpy_WW<"$dst.w = vmpa($src1.h,$src2.b)">, V6_vmpahb_enc;
defm V6_vmpyh    :T_vmpy_WV<"$dst.w = vmpy($src1.h,$src2.h)">, V6_vmpyh_enc;
defm V6_vmpyuh   :T_vmpy_WV<"$dst.uw = vmpy($src1.uh,$src2.uh)">, V6_vmpyuh_enc;
defm V6_vmpyiwh  :T_vmpy_VV<"$dst.w = vmpyi($src1.w,$src2.h)">, V6_vmpyiwh_enc;
defm V6_vtmpyhb  :T_vmpy_WW<"$dst.w = vtmpy($src1.h,$src2.b)">, V6_vtmpyhb_enc;
defm V6_vmpyub   :T_vmpy_WV<"$dst.uh = vmpy($src1.ub,$src2.ub)">, V6_vmpyub_enc;

let Itinerary = CVI_VX_LONG, Type = TypeCVI_VX in
defm V6_vmpyihb  :T_vmpy_VV<"$dst.h = vmpyi($src1.h,$src2.b)">, V6_vmpyihb_enc;

defm V6_vdmpybus_dv :
     T_vmpy_WW <"$dst.h = vdmpy($src1.ub,$src2.b)">, V6_vdmpybus_dv_enc;
defm V6_vdmpyhsusat :
     T_vmpy_VV <"$dst.w = vdmpy($src1.h,$src2.uh):sat">, V6_vdmpyhsusat_enc;
defm V6_vdmpyhsuisat :
     T_vmpy_VW <"$dst.w = vdmpy($src1.h,$src2.uh,#1):sat">, V6_vdmpyhsuisat_enc;
defm V6_vdmpyhsat :
     T_vmpy_VV <"$dst.w = vdmpy($src1.h,$src2.h):sat">, V6_vdmpyhsat_enc;
defm V6_vdmpyhisat :
     T_vmpy_VW <"$dst.w = vdmpy($src1.h,$src2.h):sat">, V6_vdmpyhisat_enc;
defm V6_vdmpyhb_dv :
     T_vmpy_WW <"$dst.w = vdmpy($src1.h,$src2.b)">, V6_vdmpyhb_dv_enc;
defm V6_vmpyhss :
     T_vmpy_VV <"$dst.h = vmpy($src1.h,$src2.h):<<1:sat">, V6_vmpyhss_enc;
defm V6_vmpyhsrs :
     T_vmpy_VV <"$dst.h = vmpy($src1.h,$src2.h):<<1:rnd:sat">, V6_vmpyhsrs_enc;

let Itinerary = CVI_VP, Type = TypeCVI_VP in
defm V6_vror : T_vmpy_VV <"$dst = vror($src1,$src2)">, V6_vror_enc;

let Itinerary = CVI_VX, Type = TypeCVI_VX in {
defm V6_vdmpyhb  : T_vmpy_VV<"$dst.w = vdmpy($src1.h,$src2.b)">, V6_vdmpyhb_enc;
defm V6_vrmpybus : T_vmpy_VV<"$dst.w = vrmpy($src1.ub,$src2.b)">, V6_vrmpybus_enc;
defm V6_vdmpybus : T_vmpy_VV<"$dst.h = vdmpy($src1.ub,$src2.b)">, V6_vdmpybus_enc;
defm V6_vmpyiwb  : T_vmpy_VV<"$dst.w = vmpyi($src1.w,$src2.b)">, V6_vmpyiwb_enc;
defm V6_vrmpyub : T_vmpy_VV<"$dst.uw = vrmpy($src1.ub,$src2.ub)">, V6_vrmpyub_enc;
}

let Itinerary = CVI_VS, Type = TypeCVI_VS in {
defm V6_vasrw  : T_vmpy_VV <"$dst.w = vasr($src1.w,$src2)">, V6_vasrw_enc;
defm V6_vasrh  : T_vmpy_VV <"$dst.h = vasr($src1.h,$src2)">, V6_vasrh_enc;
defm V6_vaslw  : T_vmpy_VV <"$dst.w = vasl($src1.w,$src2)">, V6_vaslw_enc;
defm V6_vaslh  : T_vmpy_VV <"$dst.h = vasl($src1.h,$src2)">, V6_vaslh_enc;
defm V6_vlsrw  : T_vmpy_VV <"$dst.uw = vlsr($src1.uw,$src2)">, V6_vlsrw_enc;
defm V6_vlsrh  : T_vmpy_VV <"$dst.uh = vlsr($src1.uh,$src2)">, V6_vlsrh_enc;
}

let hasNewValue = 1 in
class T_HVX_alu <string asmString, InstrItinClass itin,
                 RegisterClass RCout, RegisterClass RCin>
  : CVI_VA_Resource1 <(outs RCout:$dst), (ins RCin:$src1, RCin:$src2),
    asmString >{
  let Itinerary = itin;
  let Type = !cast<IType>("Type"#itin);
}

multiclass T_HVX_alu <string asmString, RegisterClass RCout,
           RegisterClass RCin, InstrItinClass itin> {
  def NAME : T_HVX_alu <asmString, itin, RCout, RCin>;
  let isCodeGenOnly = 1 in
  def NAME#_128B : T_HVX_alu <asmString, itin,
                              !cast<RegisterClass>(RCout#"128B"),
                              !cast<RegisterClass>(RCin#"128B")>;
}

multiclass T_HVX_alu_VV <string asmString>:
  T_HVX_alu <asmString, VectorRegs, VectorRegs, CVI_VA>;

multiclass T_HVX_alu_WW <string asmString>:
  T_HVX_alu <asmString, VecDblRegs, VecDblRegs, CVI_VA_DV>;

multiclass T_HVX_alu_WV <string asmString>:
  T_HVX_alu <asmString, VecDblRegs, VectorRegs, CVI_VX_DV>;


let Itinerary  =  CVI_VX, Type  =  TypeCVI_VX in {
defm V6_vrmpyubv :
     T_HVX_alu_VV <"$dst.uw = vrmpy($src1.ub,$src2.ub)">, V6_vrmpyubv_enc;
defm V6_vrmpybv :
     T_HVX_alu_VV <"$dst.w = vrmpy($src1.b,$src2.b)">, V6_vrmpybv_enc;
defm V6_vrmpybusv :
     T_HVX_alu_VV <"$dst.w = vrmpy($src1.ub,$src2.b)">, V6_vrmpybusv_enc;
defm V6_vabsdiffub :
     T_HVX_alu_VV <"$dst.ub = vabsdiff($src1.ub,$src2.ub)">, V6_vabsdiffub_enc;
defm V6_vabsdiffh :
     T_HVX_alu_VV <"$dst.uh = vabsdiff($src1.h,$src2.h)">, V6_vabsdiffh_enc;
defm V6_vabsdiffuh :
     T_HVX_alu_VV <"$dst.uh = vabsdiff($src1.uh,$src2.uh)">, V6_vabsdiffuh_enc;
defm V6_vabsdiffw :
     T_HVX_alu_VV <"$dst.uw = vabsdiff($src1.w,$src2.w)">, V6_vabsdiffw_enc;
}

let Itinerary = CVI_VX_DV, Type = TypeCVI_VX_DV in {
defm V6_vdmpyhvsat :
     T_HVX_alu_VV <"$dst.w = vdmpy($src1.h,$src2.h):sat">, V6_vdmpyhvsat_enc;
defm V6_vmpyhvsrs :
     T_HVX_alu_VV<"$dst.h = vmpy($src1.h,$src2.h):<<1:rnd:sat">, V6_vmpyhvsrs_enc;
defm V6_vmpyih :
     T_HVX_alu_VV <"$dst.h = vmpyi($src1.h,$src2.h)">, V6_vmpyih_enc;
}

defm V6_vand :
     T_HVX_alu_VV <"$dst = vand($src1,$src2)">, V6_vand_enc;
defm V6_vor :
     T_HVX_alu_VV <"$dst = vor($src1,$src2)">, V6_vor_enc;
defm V6_vxor :
     T_HVX_alu_VV <"$dst = vxor($src1,$src2)">, V6_vxor_enc;
defm V6_vaddw :
     T_HVX_alu_VV <"$dst.w = vadd($src1.w,$src2.w)">, V6_vaddw_enc;
defm V6_vaddubsat :
     T_HVX_alu_VV <"$dst.ub = vadd($src1.ub,$src2.ub):sat">, V6_vaddubsat_enc;
defm V6_vadduhsat :
     T_HVX_alu_VV <"$dst.uh = vadd($src1.uh,$src2.uh):sat">, V6_vadduhsat_enc;
defm V6_vaddhsat :
     T_HVX_alu_VV <"$dst.h = vadd($src1.h,$src2.h):sat">, V6_vaddhsat_enc;
defm V6_vaddwsat :
     T_HVX_alu_VV <"$dst.w = vadd($src1.w,$src2.w):sat">, V6_vaddwsat_enc;
defm V6_vsubb :
     T_HVX_alu_VV <"$dst.b = vsub($src1.b,$src2.b)">, V6_vsubb_enc;
defm V6_vsubh :
     T_HVX_alu_VV <"$dst.h = vsub($src1.h,$src2.h)">, V6_vsubh_enc;
defm V6_vsubw :
     T_HVX_alu_VV <"$dst.w = vsub($src1.w,$src2.w)">, V6_vsubw_enc;
defm V6_vsububsat :
     T_HVX_alu_VV <"$dst.ub = vsub($src1.ub,$src2.ub):sat">, V6_vsububsat_enc;
defm V6_vsubuhsat :
     T_HVX_alu_VV <"$dst.uh = vsub($src1.uh,$src2.uh):sat">, V6_vsubuhsat_enc;
defm V6_vsubhsat :
     T_HVX_alu_VV <"$dst.h = vsub($src1.h,$src2.h):sat">, V6_vsubhsat_enc;
defm V6_vsubwsat :
     T_HVX_alu_VV <"$dst.w = vsub($src1.w,$src2.w):sat">, V6_vsubwsat_enc;
defm V6_vavgub :
     T_HVX_alu_VV <"$dst.ub = vavg($src1.ub,$src2.ub)">, V6_vavgub_enc;
defm V6_vavguh :
     T_HVX_alu_VV <"$dst.uh = vavg($src1.uh,$src2.uh)">, V6_vavguh_enc;
defm V6_vavgh :
     T_HVX_alu_VV <"$dst.h = vavg($src1.h,$src2.h)">, V6_vavgh_enc;
defm V6_vavgw :
     T_HVX_alu_VV <"$dst.w = vavg($src1.w,$src2.w)">, V6_vavgw_enc;
defm V6_vnavgub :
     T_HVX_alu_VV <"$dst.b = vnavg($src1.ub,$src2.ub)">, V6_vnavgub_enc;
defm V6_vnavgh :
     T_HVX_alu_VV <"$dst.h = vnavg($src1.h,$src2.h)">, V6_vnavgh_enc;
defm V6_vnavgw :
     T_HVX_alu_VV <"$dst.w = vnavg($src1.w,$src2.w)">, V6_vnavgw_enc;
defm V6_vavgubrnd :
     T_HVX_alu_VV <"$dst.ub = vavg($src1.ub,$src2.ub):rnd">, V6_vavgubrnd_enc;
defm V6_vavguhrnd :
     T_HVX_alu_VV <"$dst.uh = vavg($src1.uh,$src2.uh):rnd">, V6_vavguhrnd_enc;
defm V6_vavghrnd :
     T_HVX_alu_VV <"$dst.h = vavg($src1.h,$src2.h):rnd">, V6_vavghrnd_enc;
defm V6_vavgwrnd :
     T_HVX_alu_VV <"$dst.w = vavg($src1.w,$src2.w):rnd">, V6_vavgwrnd_enc;

defm V6_vmpybv :
     T_HVX_alu_WV <"$dst.h = vmpy($src1.b,$src2.b)">, V6_vmpybv_enc;
defm V6_vmpyubv :
     T_HVX_alu_WV <"$dst.uh = vmpy($src1.ub,$src2.ub)">, V6_vmpyubv_enc;
defm V6_vmpybusv :
     T_HVX_alu_WV <"$dst.h = vmpy($src1.ub,$src2.b)">, V6_vmpybusv_enc;
defm V6_vmpyhv :
     T_HVX_alu_WV <"$dst.w = vmpy($src1.h,$src2.h)">, V6_vmpyhv_enc;
defm V6_vmpyuhv :
     T_HVX_alu_WV <"$dst.uw = vmpy($src1.uh,$src2.uh)">, V6_vmpyuhv_enc;
defm V6_vmpyhus :
     T_HVX_alu_WV <"$dst.w = vmpy($src1.h,$src2.uh)">, V6_vmpyhus_enc;
defm V6_vaddubh :
     T_HVX_alu_WV <"$dst.h = vadd($src1.ub,$src2.ub)">, V6_vaddubh_enc;
defm V6_vadduhw :
     T_HVX_alu_WV <"$dst.w = vadd($src1.uh,$src2.uh)">, V6_vadduhw_enc;
defm V6_vaddhw :
     T_HVX_alu_WV <"$dst.w = vadd($src1.h,$src2.h)">, V6_vaddhw_enc;
defm V6_vsububh :
     T_HVX_alu_WV <"$dst.h = vsub($src1.ub,$src2.ub)">, V6_vsububh_enc;
defm V6_vsubuhw :
     T_HVX_alu_WV <"$dst.w = vsub($src1.uh,$src2.uh)">, V6_vsubuhw_enc;
defm V6_vsubhw :
     T_HVX_alu_WV <"$dst.w = vsub($src1.h,$src2.h)">, V6_vsubhw_enc;

defm V6_vaddb_dv :
     T_HVX_alu_WW <"$dst.b = vadd($src1.b,$src2.b)">, V6_vaddb_dv_enc;
defm V6_vaddh_dv :
     T_HVX_alu_WW <"$dst.h = vadd($src1.h,$src2.h)">, V6_vaddh_dv_enc;
defm V6_vaddw_dv :
     T_HVX_alu_WW <"$dst.w = vadd($src1.w,$src2.w)">, V6_vaddw_dv_enc;
defm V6_vaddubsat_dv :
     T_HVX_alu_WW <"$dst.ub = vadd($src1.ub,$src2.ub):sat">, V6_vaddubsat_dv_enc;
defm V6_vadduhsat_dv :
     T_HVX_alu_WW <"$dst.uh = vadd($src1.uh,$src2.uh):sat">, V6_vadduhsat_dv_enc;
defm V6_vaddhsat_dv :
     T_HVX_alu_WW <"$dst.h = vadd($src1.h,$src2.h):sat">, V6_vaddhsat_dv_enc;
defm V6_vaddwsat_dv :
     T_HVX_alu_WW <"$dst.w = vadd($src1.w,$src2.w):sat">, V6_vaddwsat_dv_enc;
defm V6_vsubb_dv :
     T_HVX_alu_WW <"$dst.b = vsub($src1.b,$src2.b)">, V6_vsubb_dv_enc;
defm V6_vsubh_dv :
     T_HVX_alu_WW <"$dst.h = vsub($src1.h,$src2.h)">, V6_vsubh_dv_enc;
defm V6_vsubw_dv :
     T_HVX_alu_WW <"$dst.w = vsub($src1.w,$src2.w)">, V6_vsubw_dv_enc;
defm V6_vsububsat_dv :
     T_HVX_alu_WW <"$dst.ub = vsub($src1.ub,$src2.ub):sat">, V6_vsububsat_dv_enc;
defm V6_vsubuhsat_dv :
     T_HVX_alu_WW <"$dst.uh = vsub($src1.uh,$src2.uh):sat">, V6_vsubuhsat_dv_enc;
defm V6_vsubhsat_dv :
     T_HVX_alu_WW <"$dst.h = vsub($src1.h,$src2.h):sat">, V6_vsubhsat_dv_enc;
defm V6_vsubwsat_dv :
     T_HVX_alu_WW <"$dst.w = vsub($src1.w,$src2.w):sat">, V6_vsubwsat_dv_enc;

let Itinerary = CVI_VX_DV_LONG, Type = TypeCVI_VX_DV in {
defm V6_vmpabusv :
     T_HVX_alu_WW <"$dst.h = vmpa($src1.ub,$src2.b)">, V6_vmpabusv_enc;
defm V6_vmpabuuv :
     T_HVX_alu_WW <"$dst.h = vmpa($src1.ub,$src2.ub)">, V6_vmpabuuv_enc;
}

let isAccumulator = 1, hasNewValue = 1 in
class T_HVX_vmpyacc <string asmString, InstrItinClass itin, RegisterClass RCout,
                     RegisterClass RCin1, RegisterClass RCin2>
  : CVI_VA_Resource1 <(outs RCout:$dst),
                      (ins RCout:$_src_, RCin1:$src1, RCin2:$src2), asmString,
                      [], "$dst = $_src_" > {
  let Itinerary = itin;
  let Type = !cast<IType>("Type"#itin);
}

multiclass T_HVX_vmpyacc_both <string asmString, RegisterClass RCout,
           RegisterClass RCin1, RegisterClass RCin2, InstrItinClass itin > {
  def NAME : T_HVX_vmpyacc <asmString, itin, RCout, RCin1, RCin2>;
  let isCodeGenOnly = 1 in
  def NAME#_128B : T_HVX_vmpyacc <asmString, itin,
                   !cast<RegisterClass>(RCout#"128B"),
                   !cast<RegisterClass>(RCin1#"128B"),
                   !cast<RegisterClass>(RCin2#
                   !if(!eq (!cast<string>(RCin2), "IntRegs"), "", "128B"))>;
}

multiclass T_HVX_vmpyacc_VVR <string asmString>:
  T_HVX_vmpyacc_both <asmString, VectorRegs, VectorRegs, IntRegs, CVI_VX>;

multiclass T_HVX_vmpyacc_VWR <string asmString>:
  T_HVX_vmpyacc_both <asmString, VectorRegs, VecDblRegs, IntRegs, CVI_VX_DV>;

multiclass T_HVX_vmpyacc_WVR <string asmString>:
  T_HVX_vmpyacc_both <asmString, VecDblRegs, VectorRegs, IntRegs, CVI_VX_DV>;

multiclass T_HVX_vmpyacc_WWR <string asmString>:
  T_HVX_vmpyacc_both <asmString, VecDblRegs, VecDblRegs, IntRegs, CVI_VX_DV>;

multiclass T_HVX_vmpyacc_VVV <string asmString>:
  T_HVX_vmpyacc_both <asmString, VectorRegs, VectorRegs, VectorRegs, CVI_VX_DV>;

multiclass T_HVX_vmpyacc_WVV <string asmString>:
  T_HVX_vmpyacc_both <asmString, VecDblRegs, VectorRegs, VectorRegs, CVI_VX_DV>;


defm V6_vtmpyb_acc :
     T_HVX_vmpyacc_WWR <"$dst.h += vtmpy($src1.b,$src2.b)">,
     V6_vtmpyb_acc_enc;
defm V6_vtmpybus_acc :
     T_HVX_vmpyacc_WWR <"$dst.h += vtmpy($src1.ub,$src2.b)">,
     V6_vtmpybus_acc_enc;
defm V6_vtmpyhb_acc :
     T_HVX_vmpyacc_WWR <"$dst.w += vtmpy($src1.h,$src2.b)">,
     V6_vtmpyhb_acc_enc;
defm V6_vdmpyhb_acc :
     T_HVX_vmpyacc_VVR <"$dst.w += vdmpy($src1.h,$src2.b)">,
     V6_vdmpyhb_acc_enc;
defm V6_vrmpyub_acc :
     T_HVX_vmpyacc_VVR <"$dst.uw += vrmpy($src1.ub,$src2.ub)">,
     V6_vrmpyub_acc_enc;
defm V6_vrmpybus_acc :
     T_HVX_vmpyacc_VVR <"$dst.w += vrmpy($src1.ub,$src2.b)">,
     V6_vrmpybus_acc_enc;
defm V6_vdmpybus_acc :
     T_HVX_vmpyacc_VVR <"$dst.h += vdmpy($src1.ub,$src2.b)">,
     V6_vdmpybus_acc_enc;
defm V6_vdmpybus_dv_acc :
     T_HVX_vmpyacc_WWR <"$dst.h += vdmpy($src1.ub,$src2.b)">,
     V6_vdmpybus_dv_acc_enc;
defm V6_vdmpyhsuisat_acc :
     T_HVX_vmpyacc_VWR <"$dst.w += vdmpy($src1.h,$src2.uh,#1):sat">,
     V6_vdmpyhsuisat_acc_enc;
defm V6_vdmpyhisat_acc :
     T_HVX_vmpyacc_VWR <"$dst.w += vdmpy($src1.h,$src2.h):sat">,
     V6_vdmpyhisat_acc_enc;
defm V6_vdmpyhb_dv_acc :
     T_HVX_vmpyacc_WWR <"$dst.w += vdmpy($src1.h,$src2.b)">,
     V6_vdmpyhb_dv_acc_enc;
defm V6_vmpybus_acc :
     T_HVX_vmpyacc_WVR <"$dst.h += vmpy($src1.ub,$src2.b)">,
     V6_vmpybus_acc_enc;
defm V6_vmpabus_acc :
     T_HVX_vmpyacc_WWR <"$dst.h += vmpa($src1.ub,$src2.b)">,
     V6_vmpabus_acc_enc;
defm V6_vmpahb_acc :
     T_HVX_vmpyacc_WWR <"$dst.w += vmpa($src1.h,$src2.b)">,
     V6_vmpahb_acc_enc;
defm V6_vmpyhsat_acc :
     T_HVX_vmpyacc_WVR <"$dst.w += vmpy($src1.h,$src2.h):sat">,
     V6_vmpyhsat_acc_enc;
defm V6_vmpyuh_acc :
     T_HVX_vmpyacc_WVR <"$dst.uw += vmpy($src1.uh,$src2.uh)">,
     V6_vmpyuh_acc_enc;
defm V6_vmpyiwb_acc :
     T_HVX_vmpyacc_VVR <"$dst.w += vmpyi($src1.w,$src2.b)">,
     V6_vmpyiwb_acc_enc;
defm V6_vdsaduh_acc :
     T_HVX_vmpyacc_WWR <"$dst.uw += vdsad($src1.uh,$src2.uh)">,
     V6_vdsaduh_acc_enc;
defm V6_vmpyihb_acc :
     T_HVX_vmpyacc_VVR <"$dst.h += vmpyi($src1.h,$src2.b)">,
     V6_vmpyihb_acc_enc;
defm V6_vmpyub_acc :
     T_HVX_vmpyacc_WVR <"$dst.uh += vmpy($src1.ub,$src2.ub)">,
     V6_vmpyub_acc_enc;

let Itinerary = CVI_VX_DV, Type = TypeCVI_VX_DV in {
defm V6_vdmpyhsusat_acc :
     T_HVX_vmpyacc_VVR <"$dst.w += vdmpy($src1.h,$src2.uh):sat">,
     V6_vdmpyhsusat_acc_enc;
defm V6_vdmpyhsat_acc :
     T_HVX_vmpyacc_VVR <"$dst.w += vdmpy($src1.h,$src2.h):sat">,
     V6_vdmpyhsat_acc_enc;
defm V6_vmpyiwh_acc : T_HVX_vmpyacc_VVR
     <"$dst.w += vmpyi($src1.w,$src2.h)">, V6_vmpyiwh_acc_enc;
}

let Itinerary = CVI_VS, Type = TypeCVI_VS in {
defm V6_vaslw_acc :
     T_HVX_vmpyacc_VVR <"$dst.w += vasl($src1.w,$src2)">, V6_vaslw_acc_enc;
defm V6_vasrw_acc :
     T_HVX_vmpyacc_VVR <"$dst.w += vasr($src1.w,$src2)">, V6_vasrw_acc_enc;
}

defm V6_vdmpyhvsat_acc :
     T_HVX_vmpyacc_VVV <"$dst.w += vdmpy($src1.h,$src2.h):sat">,
     V6_vdmpyhvsat_acc_enc;
defm V6_vmpybusv_acc :
     T_HVX_vmpyacc_WVV <"$dst.h += vmpy($src1.ub,$src2.b)">,
     V6_vmpybusv_acc_enc;
defm V6_vmpybv_acc :
     T_HVX_vmpyacc_WVV <"$dst.h += vmpy($src1.b,$src2.b)">, V6_vmpybv_acc_enc;
defm V6_vmpyhus_acc :
     T_HVX_vmpyacc_WVV <"$dst.w += vmpy($src1.h,$src2.uh)">, V6_vmpyhus_acc_enc;
defm V6_vmpyhv_acc :
     T_HVX_vmpyacc_WVV <"$dst.w += vmpy($src1.h,$src2.h)">, V6_vmpyhv_acc_enc;
defm V6_vmpyiewh_acc :
     T_HVX_vmpyacc_VVV <"$dst.w += vmpyie($src1.w,$src2.h)">,
     V6_vmpyiewh_acc_enc;
defm V6_vmpyiewuh_acc :
     T_HVX_vmpyacc_VVV <"$dst.w += vmpyie($src1.w,$src2.uh)">,
     V6_vmpyiewuh_acc_enc;
defm V6_vmpyih_acc :
     T_HVX_vmpyacc_VVV <"$dst.h += vmpyi($src1.h,$src2.h)">, V6_vmpyih_acc_enc;
defm V6_vmpyowh_rnd_sacc :
     T_HVX_vmpyacc_VVV <"$dst.w += vmpyo($src1.w,$src2.h):<<1:rnd:sat:shift">,
     V6_vmpyowh_rnd_sacc_enc;
defm V6_vmpyowh_sacc :
     T_HVX_vmpyacc_VVV <"$dst.w += vmpyo($src1.w,$src2.h):<<1:sat:shift">,
     V6_vmpyowh_sacc_enc;
defm V6_vmpyubv_acc :
     T_HVX_vmpyacc_WVV <"$dst.uh += vmpy($src1.ub,$src2.ub)">,
     V6_vmpyubv_acc_enc;
defm V6_vmpyuhv_acc :
     T_HVX_vmpyacc_WVV <"$dst.uw += vmpy($src1.uh,$src2.uh)">,
     V6_vmpyuhv_acc_enc;
defm V6_vrmpybusv_acc :
     T_HVX_vmpyacc_VVV <"$dst.w += vrmpy($src1.ub,$src2.b)">,
     V6_vrmpybusv_acc_enc;
defm V6_vrmpybv_acc :
     T_HVX_vmpyacc_VVV <"$dst.w += vrmpy($src1.b,$src2.b)">, V6_vrmpybv_acc_enc;
defm V6_vrmpyubv_acc :
     T_HVX_vmpyacc_VVV <"$dst.uw += vrmpy($src1.ub,$src2.ub)">,
     V6_vrmpyubv_acc_enc;


class T_HVX_vcmp <string asmString, RegisterClass RCout, RegisterClass RCin>
  : CVI_VA_Resource1 <(outs RCout:$dst),
                      (ins RCout:$_src_, RCin:$src1, RCin:$src2), asmString,
                      [], "$dst = $_src_" > {
  let Itinerary = CVI_VA;
  let Type = TypeCVI_VA;
}

multiclass T_HVX_vcmp <string asmString> {
  def NAME : T_HVX_vcmp <asmString, VecPredRegs, VectorRegs>;
  let isCodeGenOnly = 1 in
  def NAME#_128B : T_HVX_vcmp <asmString, VecPredRegs128B, VectorRegs128B>;
}

defm V6_veqb_and :
     T_HVX_vcmp <"$dst &= vcmp.eq($src1.b,$src2.b)">, V6_veqb_and_enc;
defm V6_veqh_and :
     T_HVX_vcmp <"$dst &= vcmp.eq($src1.h,$src2.h)">, V6_veqh_and_enc;
defm V6_veqw_and :
     T_HVX_vcmp <"$dst &= vcmp.eq($src1.w,$src2.w)">, V6_veqw_and_enc;
defm V6_vgtb_and :
     T_HVX_vcmp <"$dst &= vcmp.gt($src1.b,$src2.b)">, V6_vgtb_and_enc;
defm V6_vgth_and :
     T_HVX_vcmp <"$dst &= vcmp.gt($src1.h,$src2.h)">, V6_vgth_and_enc;
defm V6_vgtw_and :
     T_HVX_vcmp <"$dst &= vcmp.gt($src1.w,$src2.w)">, V6_vgtw_and_enc;
defm V6_vgtub_and :
     T_HVX_vcmp <"$dst &= vcmp.gt($src1.ub,$src2.ub)">, V6_vgtub_and_enc;
defm V6_vgtuh_and :
     T_HVX_vcmp <"$dst &= vcmp.gt($src1.uh,$src2.uh)">, V6_vgtuh_and_enc;
defm V6_vgtuw_and :
     T_HVX_vcmp <"$dst &= vcmp.gt($src1.uw,$src2.uw)">, V6_vgtuw_and_enc;
defm V6_veqb_or :
     T_HVX_vcmp <"$dst |= vcmp.eq($src1.b,$src2.b)">, V6_veqb_or_enc;
defm V6_veqh_or :
     T_HVX_vcmp <"$dst |= vcmp.eq($src1.h,$src2.h)">, V6_veqh_or_enc;
defm V6_veqw_or :
     T_HVX_vcmp <"$dst |= vcmp.eq($src1.w,$src2.w)">, V6_veqw_or_enc;
defm V6_vgtb_or :
     T_HVX_vcmp <"$dst |= vcmp.gt($src1.b,$src2.b)">, V6_vgtb_or_enc;
defm V6_vgth_or :
     T_HVX_vcmp <"$dst |= vcmp.gt($src1.h,$src2.h)">, V6_vgth_or_enc;
defm V6_vgtw_or :
     T_HVX_vcmp <"$dst |= vcmp.gt($src1.w,$src2.w)">, V6_vgtw_or_enc;
defm V6_vgtub_or :
     T_HVX_vcmp <"$dst |= vcmp.gt($src1.ub,$src2.ub)">, V6_vgtub_or_enc;
defm V6_vgtuh_or :
     T_HVX_vcmp <"$dst |= vcmp.gt($src1.uh,$src2.uh)">, V6_vgtuh_or_enc;
defm V6_vgtuw_or :
     T_HVX_vcmp <"$dst |= vcmp.gt($src1.uw,$src2.uw)">, V6_vgtuw_or_enc;
defm V6_veqb_xor :
     T_HVX_vcmp <"$dst ^= vcmp.eq($src1.b,$src2.b)">, V6_veqb_xor_enc;
defm V6_veqh_xor :
     T_HVX_vcmp <"$dst ^= vcmp.eq($src1.h,$src2.h)">, V6_veqh_xor_enc;
defm V6_veqw_xor :
     T_HVX_vcmp <"$dst ^= vcmp.eq($src1.w,$src2.w)">, V6_veqw_xor_enc;
defm V6_vgtb_xor :
     T_HVX_vcmp <"$dst ^= vcmp.gt($src1.b,$src2.b)">, V6_vgtb_xor_enc;
defm V6_vgth_xor :
     T_HVX_vcmp <"$dst ^= vcmp.gt($src1.h,$src2.h)">, V6_vgth_xor_enc;
defm V6_vgtw_xor :
     T_HVX_vcmp <"$dst ^= vcmp.gt($src1.w,$src2.w)">, V6_vgtw_xor_enc;
defm V6_vgtub_xor :
     T_HVX_vcmp <"$dst ^= vcmp.gt($src1.ub,$src2.ub)">, V6_vgtub_xor_enc;
defm V6_vgtuh_xor :
     T_HVX_vcmp <"$dst ^= vcmp.gt($src1.uh,$src2.uh)">, V6_vgtuh_xor_enc;
defm V6_vgtuw_xor :
     T_HVX_vcmp <"$dst ^= vcmp.gt($src1.uw,$src2.uw)">, V6_vgtuw_xor_enc;

defm V6_vminub :
     T_HVX_alu_VV <"$dst.ub = vmin($src1.ub,$src2.ub)">, V6_vminub_enc;
defm V6_vminuh :
     T_HVX_alu_VV <"$dst.uh = vmin($src1.uh,$src2.uh)">, V6_vminuh_enc;
defm V6_vminh :
     T_HVX_alu_VV <"$dst.h = vmin($src1.h,$src2.h)">, V6_vminh_enc;
defm V6_vminw :
     T_HVX_alu_VV <"$dst.w = vmin($src1.w,$src2.w)">, V6_vminw_enc;
defm V6_vmaxub :
     T_HVX_alu_VV <"$dst.ub = vmax($src1.ub,$src2.ub)">, V6_vmaxub_enc;
defm V6_vmaxuh :
     T_HVX_alu_VV <"$dst.uh = vmax($src1.uh,$src2.uh)">, V6_vmaxuh_enc;
defm V6_vmaxh :
     T_HVX_alu_VV <"$dst.h = vmax($src1.h,$src2.h)">, V6_vmaxh_enc;
defm V6_vmaxw :
     T_HVX_alu_VV <"$dst.w = vmax($src1.w,$src2.w)">, V6_vmaxw_enc;
defm V6_vshuffeb :
     T_HVX_alu_VV <"$dst.b = vshuffe($src1.b,$src2.b)">, V6_vshuffeb_enc;
defm V6_vshuffob :
     T_HVX_alu_VV <"$dst.b = vshuffo($src1.b,$src2.b)">, V6_vshuffob_enc;
defm V6_vshufeh :
     T_HVX_alu_VV <"$dst.h = vshuffe($src1.h,$src2.h)">, V6_vshufeh_enc;
defm V6_vshufoh :
     T_HVX_alu_VV <"$dst.h = vshuffo($src1.h,$src2.h)">, V6_vshufoh_enc;

let Itinerary = CVI_VX_DV, Type = TypeCVI_VX_DV in {
defm V6_vmpyowh_rnd :
     T_HVX_alu_VV <"$dst.w = vmpyo($src1.w,$src2.h):<<1:rnd:sat">,
     V6_vmpyowh_rnd_enc;
defm V6_vmpyiewuh :
     T_HVX_alu_VV <"$dst.w = vmpyie($src1.w,$src2.uh)">, V6_vmpyiewuh_enc;
defm V6_vmpyewuh :
     T_HVX_alu_VV <"$dst.w = vmpye($src1.w,$src2.uh)">, V6_vmpyewuh_enc;
defm V6_vmpyowh :
     T_HVX_alu_VV <"$dst.w = vmpyo($src1.w,$src2.h):<<1:sat">, V6_vmpyowh_enc;
defm V6_vmpyiowh :
     T_HVX_alu_VV <"$dst.w = vmpyio($src1.w,$src2.h)">, V6_vmpyiowh_enc;
}
let Itinerary = CVI_VX, Type = TypeCVI_VX in
defm V6_vmpyieoh :
     T_HVX_alu_VV <"$dst.w = vmpyieo($src1.h,$src2.h)">, V6_vmpyieoh_enc;

let Itinerary = CVI_VA_DV, Type = TypeCVI_VA_DV in {
defm V6_vshufoeh :
     T_HVX_alu_WV <"$dst.h = vshuffoe($src1.h,$src2.h)">, V6_vshufoeh_enc;
defm V6_vshufoeb :
     T_HVX_alu_WV <"$dst.b = vshuffoe($src1.b,$src2.b)">, V6_vshufoeb_enc;
}

let isRegSequence = 1, Itinerary = CVI_VA_DV, Type = TypeCVI_VA_DV in
defm V6_vcombine :
     T_HVX_alu_WV <"$dst = vcombine($src1,$src2)">, V6_vcombine_enc;

let Itinerary = CVI_VINLANESAT, Type = TypeCVI_VINLANESAT in {
defm V6_vsathub :
     T_HVX_alu_VV <"$dst.ub = vsat($src1.h,$src2.h)">, V6_vsathub_enc;
defm V6_vsatwh :
     T_HVX_alu_VV <"$dst.h = vsat($src1.w,$src2.w)">, V6_vsatwh_enc;
}

let Itinerary = CVI_VS, Type = TypeCVI_VS in {
defm V6_vroundwh :
     T_HVX_alu_VV <"$dst.h = vround($src1.w,$src2.w):sat">, V6_vroundwh_enc;
defm V6_vroundwuh :
     T_HVX_alu_VV <"$dst.uh = vround($src1.w,$src2.w):sat">, V6_vroundwuh_enc;
defm V6_vroundhb :
     T_HVX_alu_VV <"$dst.b = vround($src1.h,$src2.h):sat">, V6_vroundhb_enc;
defm V6_vroundhub :
     T_HVX_alu_VV <"$dst.ub = vround($src1.h,$src2.h):sat">, V6_vroundhub_enc;
defm V6_vasrwv :
     T_HVX_alu_VV <"$dst.w = vasr($src1.w,$src2.w)">, V6_vasrwv_enc;
defm V6_vlsrwv :
     T_HVX_alu_VV <"$dst.w = vlsr($src1.w,$src2.w)">, V6_vlsrwv_enc;
defm V6_vlsrhv :
     T_HVX_alu_VV <"$dst.h = vlsr($src1.h,$src2.h)">, V6_vlsrhv_enc;
defm V6_vasrhv :
     T_HVX_alu_VV <"$dst.h = vasr($src1.h,$src2.h)">, V6_vasrhv_enc;
defm V6_vaslwv :
     T_HVX_alu_VV <"$dst.w = vasl($src1.w,$src2.w)">, V6_vaslwv_enc;
defm V6_vaslhv :
     T_HVX_alu_VV <"$dst.h = vasl($src1.h,$src2.h)">, V6_vaslhv_enc;
}

defm V6_vaddb :
     T_HVX_alu_VV <"$dst.b = vadd($src1.b,$src2.b)">, V6_vaddb_enc;
defm V6_vaddh :
     T_HVX_alu_VV <"$dst.h = vadd($src1.h,$src2.h)">, V6_vaddh_enc;

let Itinerary = CVI_VP, Type = TypeCVI_VP in {
defm V6_vdelta :
     T_HVX_alu_VV <"$dst = vdelta($src1,$src2)">, V6_vdelta_enc;
defm V6_vrdelta :
     T_HVX_alu_VV <"$dst = vrdelta($src1,$src2)">, V6_vrdelta_enc;
defm V6_vdealb4w :
     T_HVX_alu_VV <"$dst.b = vdeale($src1.b,$src2.b)">, V6_vdealb4w_enc;
defm V6_vpackeb :
     T_HVX_alu_VV <"$dst.b = vpacke($src1.h,$src2.h)">, V6_vpackeb_enc;
defm V6_vpackeh :
     T_HVX_alu_VV <"$dst.h = vpacke($src1.w,$src2.w)">, V6_vpackeh_enc;
defm V6_vpackhub_sat :
     T_HVX_alu_VV <"$dst.ub = vpack($src1.h,$src2.h):sat">, V6_vpackhub_sat_enc;
defm V6_vpackhb_sat :
     T_HVX_alu_VV <"$dst.b = vpack($src1.h,$src2.h):sat">, V6_vpackhb_sat_enc;
defm V6_vpackwuh_sat :
     T_HVX_alu_VV <"$dst.uh = vpack($src1.w,$src2.w):sat">, V6_vpackwuh_sat_enc;
defm V6_vpackwh_sat :
     T_HVX_alu_VV <"$dst.h = vpack($src1.w,$src2.w):sat">, V6_vpackwh_sat_enc;
defm V6_vpackob :
     T_HVX_alu_VV <"$dst.b = vpacko($src1.h,$src2.h)">, V6_vpackob_enc;
defm V6_vpackoh :
     T_HVX_alu_VV <"$dst.h = vpacko($src1.w,$src2.w)">, V6_vpackoh_enc;
}

let hasNewValue = 1, hasSideEffects = 0 in
class T_HVX_condALU <string asmString, RegisterClass RC1, RegisterClass RC2>
  : CVI_VA_Resource1 <(outs RC2:$dst),
                      (ins RC1:$src1, RC2:$_src_, RC2:$src2), asmString,
                      [], "$dst = $_src_" > {
  let Itinerary = CVI_VA;
  let Type = TypeCVI_VA;
}

multiclass T_HVX_condALU <string asmString> {
  def NAME : T_HVX_condALU <asmString, VecPredRegs, VectorRegs>;
  let isCodeGenOnly = 1 in
  def NAME#_128B : T_HVX_condALU <asmString, VecPredRegs128B, VectorRegs128B>;
}

defm V6_vaddbq  : T_HVX_condALU <"if ($src1) $dst.b += $src2.b">,
                  V6_vaddbq_enc;
defm V6_vaddhq  : T_HVX_condALU <"if ($src1) $dst.h += $src2.h">,
                  V6_vaddhq_enc;
defm V6_vaddwq  : T_HVX_condALU <"if ($src1) $dst.w += $src2.w">,
                  V6_vaddwq_enc;
defm V6_vsubbq  : T_HVX_condALU <"if ($src1) $dst.b -= $src2.b">,
                  V6_vsubbq_enc;
defm V6_vsubhq  : T_HVX_condALU <"if ($src1) $dst.h -= $src2.h">,
                  V6_vsubhq_enc;
defm V6_vsubwq  : T_HVX_condALU <"if ($src1) $dst.w -= $src2.w">,
                  V6_vsubwq_enc;
defm V6_vaddbnq : T_HVX_condALU <"if (!$src1) $dst.b += $src2.b">,
                  V6_vaddbnq_enc;
defm V6_vaddhnq : T_HVX_condALU <"if (!$src1) $dst.h += $src2.h">,
                  V6_vaddhnq_enc;
defm V6_vaddwnq : T_HVX_condALU <"if (!$src1) $dst.w += $src2.w">,
                  V6_vaddwnq_enc;
defm V6_vsubbnq : T_HVX_condALU <"if (!$src1) $dst.b -= $src2.b">,
                  V6_vsubbnq_enc;
defm V6_vsubhnq : T_HVX_condALU <"if (!$src1) $dst.h -= $src2.h">,
                  V6_vsubhnq_enc;
defm V6_vsubwnq : T_HVX_condALU <"if (!$src1) $dst.w -= $src2.w">,
                  V6_vsubwnq_enc;

let hasNewValue = 1 in
class T_HVX_alu_2op <string asmString, InstrItinClass itin,
                 RegisterClass RCout, RegisterClass RCin>
  : CVI_VA_Resource1 <(outs RCout:$dst), (ins RCin:$src1),
    asmString >{
  let Itinerary = itin;
  let Type = !cast<IType>("Type"#itin);
}

multiclass T_HVX_alu_2op <string asmString, RegisterClass RCout,
           RegisterClass RCin, InstrItinClass itin> {
  def NAME : T_HVX_alu_2op <asmString, itin, RCout, RCin>;
  let isCodeGenOnly = 1 in
  def NAME#_128B : T_HVX_alu_2op <asmString, itin,
                              !cast<RegisterClass>(RCout#"128B"),
                              !cast<RegisterClass>(RCin#"128B")>;
}

let hasNewValue = 1 in
multiclass T_HVX_alu_2op_VV <string asmString>:
  T_HVX_alu_2op <asmString, VectorRegs, VectorRegs, CVI_VA>;

multiclass T_HVX_alu_2op_WV <string asmString>:
  T_HVX_alu_2op <asmString, VecDblRegs, VectorRegs, CVI_VA_DV>;


defm V6_vabsh     : T_HVX_alu_2op_VV <"$dst.h = vabs($src1.h)">,
                    V6_vabsh_enc;
defm V6_vabsw     : T_HVX_alu_2op_VV <"$dst.w = vabs($src1.w)">,
                    V6_vabsw_enc;
defm V6_vabsh_sat : T_HVX_alu_2op_VV <"$dst.h = vabs($src1.h):sat">,
                    V6_vabsh_sat_enc;
defm V6_vabsw_sat : T_HVX_alu_2op_VV <"$dst.w = vabs($src1.w):sat">,
                    V6_vabsw_sat_enc;
defm V6_vnot      : T_HVX_alu_2op_VV <"$dst = vnot($src1)">,
                    V6_vnot_enc;
defm V6_vassign   : T_HVX_alu_2op_VV <"$dst = $src1">,
                    V6_vassign_enc;

defm V6_vzb       : T_HVX_alu_2op_WV <"$dst.uh = vzxt($src1.ub)">,
                    V6_vzb_enc;
defm V6_vzh       : T_HVX_alu_2op_WV <"$dst.uw = vzxt($src1.uh)">,
                    V6_vzh_enc;
defm V6_vsb       : T_HVX_alu_2op_WV <"$dst.h = vsxt($src1.b)">,
                    V6_vsb_enc;
defm V6_vsh       : T_HVX_alu_2op_WV <"$dst.w = vsxt($src1.h)">,
                    V6_vsh_enc;

let Itinerary = CVI_VP, Type = TypeCVI_VP in {
defm V6_vdealh    : T_HVX_alu_2op_VV <"$dst.h = vdeal($src1.h)">,
                    V6_vdealh_enc;
defm V6_vdealb    : T_HVX_alu_2op_VV <"$dst.b = vdeal($src1.b)">,
                    V6_vdealb_enc;
defm V6_vshuffh   : T_HVX_alu_2op_VV <"$dst.h = vshuff($src1.h)">,
                    V6_vshuffh_enc;
defm V6_vshuffb   : T_HVX_alu_2op_VV <"$dst.b = vshuff($src1.b)">,
                    V6_vshuffb_enc;
}

let Itinerary = CVI_VP_VS, Type = TypeCVI_VP_VS in {
defm V6_vunpackub : T_HVX_alu_2op_WV <"$dst.uh = vunpack($src1.ub)">,
                    V6_vunpackub_enc;
defm V6_vunpackuh : T_HVX_alu_2op_WV <"$dst.uw = vunpack($src1.uh)">,
                    V6_vunpackuh_enc;
defm V6_vunpackb  : T_HVX_alu_2op_WV <"$dst.h = vunpack($src1.b)">,
                    V6_vunpackb_enc;
defm V6_vunpackh  : T_HVX_alu_2op_WV <"$dst.w = vunpack($src1.h)">,
                    V6_vunpackh_enc;
}

let Itinerary = CVI_VS, Type = TypeCVI_VS in {
defm V6_vcl0w     : T_HVX_alu_2op_VV <"$dst.uw = vcl0($src1.uw)">,
                    V6_vcl0w_enc;
defm V6_vcl0h     : T_HVX_alu_2op_VV <"$dst.uh = vcl0($src1.uh)">,
                    V6_vcl0h_enc;
defm V6_vnormamtw : T_HVX_alu_2op_VV <"$dst.w = vnormamt($src1.w)">,
                    V6_vnormamtw_enc;
defm V6_vnormamth : T_HVX_alu_2op_VV <"$dst.h = vnormamt($src1.h)">,
                    V6_vnormamth_enc;
defm V6_vpopcounth : T_HVX_alu_2op_VV <"$dst.h = vpopcount($src1.h)">,
                     V6_vpopcounth_enc;
}

let isAccumulator = 1, hasNewValue = 1, Itinerary = CVI_VX_DV_LONG,
    Type = TypeCVI_VX_DV in
class T_HVX_vmpyacc2 <string asmString, RegisterClass RC>
  : CVI_VA_Resource1 <(outs RC:$dst),
                      (ins RC:$_src_, RC:$src1, IntRegs:$src2, u1Imm:$src3),
    asmString, [], "$dst = $_src_" > ;


multiclass T_HVX_vmpyacc2 <string asmString> {
  def NAME : T_HVX_vmpyacc2 <asmString, VecDblRegs>;

  let isCodeGenOnly = 1 in
  def NAME#_128B : T_HVX_vmpyacc2 <asmString, VecDblRegs128B>;
}

defm V6_vrmpybusi_acc :
     T_HVX_vmpyacc2<"$dst.w += vrmpy($src1.ub,$src2.b,#$src3)">,
     V6_vrmpybusi_acc_enc;
defm V6_vrsadubi_acc :
     T_HVX_vmpyacc2<"$dst.uw += vrsad($src1.ub,$src2.ub,#$src3)">,
     V6_vrsadubi_acc_enc;
defm V6_vrmpyubi_acc :
     T_HVX_vmpyacc2<"$dst.uw += vrmpy($src1.ub,$src2.ub,#$src3)">,
     V6_vrmpyubi_acc_enc;


let Itinerary = CVI_VX_DV_LONG, Type = TypeCVI_VX_DV, hasNewValue = 1 in
class T_HVX_vmpy2 <string asmString, RegisterClass RC>
  : CVI_VA_Resource1<(outs RC:$dst), (ins RC:$src1, IntRegs:$src2, u1Imm:$src3),
    asmString>;


multiclass T_HVX_vmpy2 <string asmString> {
  def NAME : T_HVX_vmpy2 <asmString, VecDblRegs>;

  let isCodeGenOnly = 1 in
  def NAME#_128B : T_HVX_vmpy2 <asmString, VecDblRegs128B>;
}

defm V6_vrmpybusi :
     T_HVX_vmpy2 <"$dst.w = vrmpy($src1.ub,$src2.b,#$src3)">, V6_vrmpybusi_enc;
defm V6_vrsadubi :
     T_HVX_vmpy2 <"$dst.uw = vrsad($src1.ub,$src2.ub,#$src3)">, V6_vrsadubi_enc;
defm V6_vrmpyubi :
     T_HVX_vmpy2 <"$dst.uw = vrmpy($src1.ub,$src2.ub,#$src3)">, V6_vrmpyubi_enc;


let Itinerary = CVI_VP_VS_LONG_EARLY, Type = TypeCVI_VP_VS,
    hasSideEffects = 0, hasNewValue2 = 1, opNewValue2 = 1 in
class T_HVX_perm <string asmString, RegisterClass RC>
  : CVI_VA_Resource1 <(outs RC:$_dst1_, RC:$_dst2_),
                      (ins RC:$src1, RC:$src2, IntRegs:$src3),
    asmString, [], "$_dst1_ = $src1, $_dst2_ = $src2" >;

multiclass T_HVX_perm <string asmString> {
  def NAME : T_HVX_perm <asmString, VectorRegs>;

  let isCodeGenOnly = 1 in
  def NAME#_128B : T_HVX_perm <asmString, VectorRegs128B>;
}

let hasNewValue = 1, opNewValue = 0, hasNewValue2 = 1, opNewValue2 = 1 in {
  defm V6_vshuff : T_HVX_perm <"vshuff($src1,$src2,$src3)">, V6_vshuff_enc;
  defm V6_vdeal : T_HVX_perm <"vdeal($src1,$src2,$src3)">, V6_vdeal_enc;
}

// Conditional vector move.
let isPredicated = 1, hasSideEffects = 0, hasNewValue = 1, opNewValue = 0 in
class T_HVX_cmov <bit isPredNot, RegisterClass RC>
  : CVI_VA_Resource1 <(outs RC:$dst), (ins PredRegs:$src1, RC:$src2),
    "if ("#!if(isPredNot, "!", "")#"$src1) $dst = $src2"> {
  let isPredicatedFalse = isPredNot;
}

multiclass T_HVX_cmov <bit isPredNot = 0> {
  def NAME : T_HVX_cmov <isPredNot, VectorRegs>;

  let isCodeGenOnly = 1 in
  def NAME#_128B : T_HVX_cmov <isPredNot, VectorRegs128B>;
}

defm V6_vcmov : T_HVX_cmov, V6_vcmov_enc;
defm V6_vncmov : T_HVX_cmov<1>, V6_vncmov_enc;

// Conditional vector combine.
let Itinerary = CVI_VA_DV, Type = TypeCVI_VA_DV, isPredicated = 1,
    hasSideEffects = 0, hasNewValue = 1, opNewValue = 0 in
class T_HVX_ccombine <bit isPredNot, RegisterClass RCout, RegisterClass RCin>
  : CVI_VA_Resource1 < (outs RCout:$dst),
    (ins PredRegs:$src1, RCin:$src2, RCin:$src3),
    "if ("#!if(isPredNot, "!", "")#"$src1) $dst = vcombine($src2,$src3)"> {
  let isPredicatedFalse = isPredNot;
}

multiclass T_HVX_ccombine <bit isPredNot = 0> {
  def NAME : T_HVX_ccombine <isPredNot, VecDblRegs, VectorRegs>;

  let isCodeGenOnly = 1 in
  def NAME#_128B : T_HVX_ccombine <isPredNot, VecDblRegs128B, VectorRegs128B>;
}

defm V6_vccombine : T_HVX_ccombine, V6_vccombine_enc;
defm V6_vnccombine : T_HVX_ccombine<1>, V6_vnccombine_enc;

let hasNewValue = 1 in
class T_HVX_shift <string asmString, RegisterClass RCout, RegisterClass RCin>
  : CVI_VX_DV_Resource1<(outs RCout:$dst),
    (ins RCin:$src1, RCin:$src2, IntRegsLow8:$src3),
    asmString >;

multiclass T_HVX_shift <string asmString, RegisterClass RCout,
                        RegisterClass RCin> {
  def NAME : T_HVX_shift <asmString, RCout, RCin>;
  let isCodeGenOnly = 1 in
  def NAME#_128B : T_HVX_shift <asmString, !cast<RegisterClass>(RCout#"128B"),
                                           !cast<RegisterClass>(RCin#"128B")>;
}

multiclass T_HVX_shift_VV <string asmString>:
  T_HVX_shift <asmString, VectorRegs, VectorRegs>;

multiclass T_HVX_shift_WV <string asmString>:
  T_HVX_shift <asmString, VecDblRegs, VectorRegs>;

let Itinerary = CVI_VP_LONG, Type = TypeCVI_VP in {
defm V6_valignb :
     T_HVX_shift_VV <"$dst = valign($src1,$src2,$src3)">, V6_valignb_enc;
defm V6_vlalignb :
     T_HVX_shift_VV <"$dst = vlalign($src1,$src2,$src3)">, V6_vlalignb_enc;
}

let Itinerary = CVI_VS, Type = TypeCVI_VS in {
defm V6_vasrwh :
     T_HVX_shift_VV <"$dst.h = vasr($src1.w,$src2.w,$src3)">, V6_vasrwh_enc;
defm V6_vasrwhsat :
     T_HVX_shift_VV <"$dst.h = vasr($src1.w,$src2.w,$src3):sat">,
     V6_vasrwhsat_enc;
defm V6_vasrwhrndsat :
     T_HVX_shift_VV <"$dst.h = vasr($src1.w,$src2.w,$src3):rnd:sat">,
     V6_vasrwhrndsat_enc;
defm V6_vasrwuhsat :
     T_HVX_shift_VV <"$dst.uh = vasr($src1.w,$src2.w,$src3):sat">,
     V6_vasrwuhsat_enc;
defm V6_vasrhubsat :
     T_HVX_shift_VV <"$dst.ub = vasr($src1.h,$src2.h,$src3):sat">,
     V6_vasrhubsat_enc;
defm V6_vasrhubrndsat :
     T_HVX_shift_VV <"$dst.ub = vasr($src1.h,$src2.h,$src3):rnd:sat">,
     V6_vasrhubrndsat_enc;
defm V6_vasrhbrndsat :
     T_HVX_shift_VV <"$dst.b = vasr($src1.h,$src2.h,$src3):rnd:sat">,
     V6_vasrhbrndsat_enc;
}

// Assemlber mapped -- alias?
//defm V6_vtran2x2vdd : T_HVX_shift_VV <"">, V6_vtran2x2vdd_enc;
let Itinerary = CVI_VP_VS_LONG, Type = TypeCVI_VP_VS in {
defm V6_vshuffvdd :
     T_HVX_shift_WV <"$dst = vshuff($src1,$src2,$src3)">, V6_vshuffvdd_enc;
defm V6_vdealvdd :
     T_HVX_shift_WV <"$dst = vdeal($src1,$src2,$src3)">, V6_vdealvdd_enc;
}

let hasNewValue = 1, Itinerary = CVI_VP_VS_LONG, Type = TypeCVI_VP_VS in
class T_HVX_unpack <string asmString, RegisterClass RCout, RegisterClass RCin>
  : CVI_VX_DV_Resource1<(outs RCout:$dst), (ins RCout:$_src_, RCin:$src1),
    asmString, [], "$dst = $_src_">;

multiclass T_HVX_unpack <string asmString> {
  def NAME : T_HVX_unpack <asmString, VecDblRegs, VectorRegs>;
  let isCodeGenOnly = 1 in
  def NAME#_128B : T_HVX_unpack <asmString, VecDblRegs128B, VectorRegs128B>;
}

defm V6_vunpackob : T_HVX_unpack <"$dst.h |= vunpacko($src1.b)">, V6_vunpackob_enc;
defm V6_vunpackoh : T_HVX_unpack <"$dst.w |= vunpacko($src1.h)">, V6_vunpackoh_enc;

let Itinerary = CVI_VP_LONG, Type = TypeCVI_VP, hasNewValue = 1,
    hasSideEffects = 0 in
class T_HVX_valign <string asmString, RegisterClass RC>
  : CVI_VA_Resource1<(outs RC:$dst), (ins RC:$src1, RC:$src2, u3Imm:$src3),
    asmString>;

multiclass T_HVX_valign <string asmString> {
  def NAME : T_HVX_valign <asmString, VectorRegs>;

  let isCodeGenOnly = 1 in
  def NAME#_128B : T_HVX_valign <asmString, VectorRegs128B>;
}

defm V6_valignbi :
     T_HVX_valign <"$dst = valign($src1,$src2,#$src3)">, V6_valignbi_enc;
defm V6_vlalignbi :
     T_HVX_valign <"$dst = vlalign($src1,$src2,#$src3)">, V6_vlalignbi_enc;

let Itinerary = CVI_VA_DV, Type = TypeCVI_VA_DV in
class T_HVX_predAlu <string asmString, RegisterClass RC>
  : CVI_VA_Resource1<(outs RC:$dst), (ins RC:$src1, RC:$src2),
    asmString>;

multiclass T_HVX_predAlu <string asmString> {
  def NAME : T_HVX_predAlu <asmString, VecPredRegs>;

  let isCodeGenOnly = 1 in
  def NAME#_128B : T_HVX_predAlu <asmString, VecPredRegs128B>;
}

defm V6_pred_and  : T_HVX_predAlu <"$dst = and($src1,$src2)">, V6_pred_and_enc;
defm V6_pred_or   : T_HVX_predAlu <"$dst = or($src1,$src2)">, V6_pred_or_enc;
defm V6_pred_xor  : T_HVX_predAlu <"$dst = xor($src1,$src2)">, V6_pred_xor_enc;
defm V6_pred_or_n : T_HVX_predAlu <"$dst = or($src1,!$src2)">, V6_pred_or_n_enc;
defm V6_pred_and_n :
     T_HVX_predAlu <"$dst = and($src1,!$src2)">, V6_pred_and_n_enc;

let Itinerary = CVI_VA, Type = TypeCVI_VA in
class T_HVX_prednot <RegisterClass RC>
  : CVI_VA_Resource1<(outs RC:$dst), (ins RC:$src1),
    "$dst = not($src1)">, V6_pred_not_enc;

def V6_pred_not : T_HVX_prednot <VecPredRegs>;
let isCodeGenOnly =  1 in
def V6_pred_not_128B : T_HVX_prednot <VecPredRegs128B>;

let Itinerary = CVI_VA, Type = TypeCVI_VA in
class T_HVX_vcmp2 <string asmString, RegisterClass RCout, RegisterClass RCin>
  : CVI_VA_Resource1 <(outs RCout:$dst), (ins RCin:$src1, RCin:$src2),
    asmString >;

multiclass T_HVX_vcmp2 <string asmString> {
  def NAME : T_HVX_vcmp2 <asmString, VecPredRegs, VectorRegs>;
  let isCodeGenOnly = 1 in
  def NAME#_128B : T_HVX_vcmp2 <asmString, VecPredRegs128B, VectorRegs128B>;
}

defm V6_veqb : T_HVX_vcmp2  <"$dst = vcmp.eq($src1.b,$src2.b)">, V6_veqb_enc;
defm V6_veqh : T_HVX_vcmp2  <"$dst = vcmp.eq($src1.h,$src2.h)">, V6_veqh_enc;
defm V6_veqw : T_HVX_vcmp2  <"$dst = vcmp.eq($src1.w,$src2.w)">, V6_veqw_enc;
defm V6_vgtb : T_HVX_vcmp2  <"$dst = vcmp.gt($src1.b,$src2.b)">, V6_vgtb_enc;
defm V6_vgth : T_HVX_vcmp2  <"$dst = vcmp.gt($src1.h,$src2.h)">, V6_vgth_enc;
defm V6_vgtw : T_HVX_vcmp2  <"$dst = vcmp.gt($src1.w,$src2.w)">, V6_vgtw_enc;
defm V6_vgtub : T_HVX_vcmp2 <"$dst = vcmp.gt($src1.ub,$src2.ub)">, V6_vgtub_enc;
defm V6_vgtuh : T_HVX_vcmp2 <"$dst = vcmp.gt($src1.uh,$src2.uh)">, V6_vgtuh_enc;
defm V6_vgtuw : T_HVX_vcmp2 <"$dst = vcmp.gt($src1.uw,$src2.uw)">, V6_vgtuw_enc;

let isAccumulator = 1, hasNewValue = 1, hasSideEffects = 0 in
class T_V6_vandqrt_acc <RegisterClass RCout, RegisterClass RCin>
  : CVI_VX_Resource_late<(outs RCout:$dst),
    (ins RCout:$_src_, RCin:$src1, IntRegs:$src2),
    "$dst |= vand($src1,$src2)", [], "$dst = $_src_">, V6_vandqrt_acc_enc;

def V6_vandqrt_acc : T_V6_vandqrt_acc <VectorRegs, VecPredRegs>;
let isCodeGenOnly = 1 in
def V6_vandqrt_acc_128B : T_V6_vandqrt_acc <VectorRegs128B, VecPredRegs128B>;

let isAccumulator = 1 in
class T_V6_vandvrt_acc <RegisterClass RCout, RegisterClass RCin>
  : CVI_VX_Resource_late<(outs RCout:$dst),
    (ins RCout:$_src_, RCin:$src1, IntRegs:$src2),
    "$dst |= vand($src1,$src2)", [], "$dst = $_src_">, V6_vandvrt_acc_enc;

def V6_vandvrt_acc : T_V6_vandvrt_acc <VecPredRegs, VectorRegs>;
let isCodeGenOnly = 1 in
def V6_vandvrt_acc_128B : T_V6_vandvrt_acc <VecPredRegs128B, VectorRegs128B>;

let hasNewValue =  1, hasSideEffects = 0 in
class T_V6_vandqrt <RegisterClass RCout, RegisterClass RCin>
  : CVI_VX_Resource_late<(outs RCout:$dst),
    (ins RCin:$src1, IntRegs:$src2),
    "$dst = vand($src1,$src2)" >, V6_vandqrt_enc;

def V6_vandqrt : T_V6_vandqrt <VectorRegs, VecPredRegs>;
let isCodeGenOnly = 1 in
def V6_vandqrt_128B : T_V6_vandqrt <VectorRegs128B, VecPredRegs128B>;

let hasNewValue = 1, hasSideEffects = 0 in
class T_V6_lvsplatw <RegisterClass RC>
  : CVI_VX_Resource_late<(outs RC:$dst), (ins IntRegs:$src1),
    "$dst = vsplat($src1)" >, V6_lvsplatw_enc;

def V6_lvsplatw : T_V6_lvsplatw <VectorRegs>;
let isCodeGenOnly = 1 in
def V6_lvsplatw_128B : T_V6_lvsplatw <VectorRegs128B>;


let hasNewValue = 1 in
class T_V6_vinsertwr <RegisterClass RC>
  : CVI_VX_Resource_late<(outs RC:$dst), (ins RC:$_src_, IntRegs:$src1),
    "$dst.w = vinsert($src1)", [], "$dst = $_src_">,
    V6_vinsertwr_enc;

def V6_vinsertwr : T_V6_vinsertwr <VectorRegs>;
let isCodeGenOnly = 1 in
def V6_vinsertwr_128B : T_V6_vinsertwr <VectorRegs128B>;


let Itinerary = CVI_VP_LONG, Type = TypeCVI_VP in
class T_V6_pred_scalar2 <RegisterClass RC>
  : CVI_VA_Resource1<(outs RC:$dst), (ins IntRegs:$src1),
    "$dst = vsetq($src1)">, V6_pred_scalar2_enc;

def V6_pred_scalar2 : T_V6_pred_scalar2 <VecPredRegs>;
let isCodeGenOnly = 1 in
def V6_pred_scalar2_128B : T_V6_pred_scalar2 <VecPredRegs128B>;

class T_V6_vandvrt <RegisterClass RCout, RegisterClass RCin>
  : CVI_VX_Resource_late<(outs RCout:$dst), (ins RCin:$src1, IntRegs:$src2),
    "$dst = vand($src1,$src2)">, V6_vandvrt_enc;

def V6_vandvrt : T_V6_vandvrt <VecPredRegs, VectorRegs>;
let isCodeGenOnly = 1 in
def V6_vandvrt_128B : T_V6_vandvrt <VecPredRegs128B, VectorRegs128B>;

let validSubTargets = HasV60SubT in
class T_HVX_rol <string asmString, RegisterClass RC, Operand ImmOp >
  : SInst2 <(outs RC:$dst), (ins  RC:$src1, ImmOp:$src2), asmString>;

class T_HVX_rol_R <string asmString>
  : T_HVX_rol <asmString, IntRegs, u5Imm>;
class T_HVX_rol_P <string asmString>
  : T_HVX_rol <asmString, DoubleRegs, u6Imm>;

def S6_rol_i_p : T_HVX_rol_P <"$dst = rol($src1,#$src2)">, S6_rol_i_p_enc;
let hasNewValue = 1, opNewValue = 0 in
def S6_rol_i_r : T_HVX_rol_R <"$dst = rol($src1,#$src2)">, S6_rol_i_r_enc;

let validSubTargets = HasV60SubT in
class T_HVX_rol_acc <string asmString, RegisterClass RC, Operand ImmOp>
  : SInst2 <(outs RC:$dst), (ins RC:$_src_, RC:$src1, ImmOp:$src2),
    asmString, [], "$dst = $_src_" >;

class T_HVX_rol_acc_P <string asmString>
  : T_HVX_rol_acc <asmString, DoubleRegs, u6Imm>;

class T_HVX_rol_acc_R <string asmString>
  : T_HVX_rol_acc <asmString, IntRegs, u5Imm>;

def S6_rol_i_p_nac :
    T_HVX_rol_acc_P <"$dst -= rol($src1,#$src2)">, S6_rol_i_p_nac_enc;
def S6_rol_i_p_acc :
    T_HVX_rol_acc_P <"$dst += rol($src1,#$src2)">, S6_rol_i_p_acc_enc;
def S6_rol_i_p_and :
    T_HVX_rol_acc_P <"$dst &= rol($src1,#$src2)">, S6_rol_i_p_and_enc;
def S6_rol_i_p_or  :
    T_HVX_rol_acc_P <"$dst |= rol($src1,#$src2)">, S6_rol_i_p_or_enc;
def S6_rol_i_p_xacc :
    T_HVX_rol_acc_P<"$dst ^= rol($src1,#$src2)">, S6_rol_i_p_xacc_enc;

let hasNewValue = 1, opNewValue = 0 in {
def S6_rol_i_r_nac :
    T_HVX_rol_acc_R <"$dst -= rol($src1,#$src2)">, S6_rol_i_r_nac_enc;
def S6_rol_i_r_acc :
    T_HVX_rol_acc_R <"$dst += rol($src1,#$src2)">, S6_rol_i_r_acc_enc;
def S6_rol_i_r_and :
    T_HVX_rol_acc_R <"$dst &= rol($src1,#$src2)">, S6_rol_i_r_and_enc;
def S6_rol_i_r_or :
    T_HVX_rol_acc_R <"$dst |= rol($src1,#$src2)">, S6_rol_i_r_or_enc;
def S6_rol_i_r_xacc :
    T_HVX_rol_acc_R <"$dst ^= rol($src1,#$src2)">, S6_rol_i_r_xacc_enc;
}

let isSolo = 1, Itinerary = LD_tc_ld_SLOT0, Type = TypeLD in
class T_V6_extractw <RegisterClass RC>
  : LD1Inst <(outs IntRegs:$dst), (ins RC:$src1, IntRegs:$src2),
    "$dst = vextract($src1,$src2)">, V6_extractw_enc;

def V6_extractw : T_V6_extractw <VectorRegs>;
let isCodeGenOnly = 1 in
def V6_extractw_128B : T_V6_extractw <VectorRegs128B>;

let Itinerary = ST_tc_st_SLOT0, validSubTargets = HasV55SubT  in
class T_sys0op <string asmString>
  : ST1Inst <(outs), (ins), asmString>;

let isSolo = 1, validSubTargets = HasV55SubT in {
def Y5_l2gunlock   : T_sys0op <"l2gunlock">, Y5_l2gunlock_enc;
def Y5_l2gclean    : T_sys0op <"l2gclean">, Y5_l2gclean_enc;
def Y5_l2gcleaninv : T_sys0op <"l2gcleaninv">, Y5_l2gcleaninv_enc;
}

class T_sys1op <string asmString, RegisterClass RC>
  : ST1Inst <(outs), (ins RC:$src1), asmString>;

class T_sys1op_R <string asmString> : T_sys1op <asmString, IntRegs>;
class T_sys1op_P <string asmString> : T_sys1op <asmString, DoubleRegs>;

let isSoloAX = 1, validSubTargets = HasV55SubT in
def Y5_l2unlocka     : T_sys1op_R <"l2unlocka($src1)">, Y5_l2unlocka_enc;

let isSolo = 1, validSubTargets = HasV60SubT in {
def Y6_l2gcleanpa    : T_sys1op_P <"l2gclean($src1)">, Y6_l2gcleanpa_enc;
def Y6_l2gcleaninvpa : T_sys1op_P <"l2gcleaninv($src1)">, Y6_l2gcleaninvpa_enc;
}

let Itinerary = ST_tc_3stall_SLOT0, isPredicateLate = 1, isSoloAX = 1,
    validSubTargets = HasV55SubT in
def Y5_l2locka : ST1Inst <(outs PredRegs:$dst), (ins IntRegs:$src1),
  "$dst = l2locka($src1)">, Y5_l2locka_enc;

// not defined on etc side. why?
// defm S2_cabacencbin : _VV <"Rdd=encbin(Rss,$src2,Pu)">, S2_cabacencbin_enc;

let Defs = [USR_OVF], Itinerary = M_tc_3stall_SLOT23, isPredicateLate = 1,
    hasSideEffects = 0,
validSubTargets = HasV55SubT in
def A5_ACS : MInst2 <(outs DoubleRegs:$dst1, PredRegs:$dst2),
  (ins DoubleRegs:$_src_, DoubleRegs:$src1, DoubleRegs:$src2),
  "$dst1,$dst2 = vacsh($src1,$src2)", [],
  "$dst1 = $_src_" >, Requires<[HasV55T]>, A5_ACS_enc;

let Itinerary = CVI_VA_DV, Type = TypeCVI_VA_DV, hasNewValue = 1,
    hasSideEffects = 0 in
class T_HVX_alu2 <string asmString, RegisterClass RCout, RegisterClass RCin1,
                  RegisterClass RCin2>
  : CVI_VA_Resource1<(outs RCout:$dst),
    (ins RCin1:$src1, RCin2:$src2, RCin2:$src3), asmString>;

multiclass T_HVX_alu2 <string asmString, RegisterClass RC > {
  def NAME : T_HVX_alu2 <asmString, RC, VecPredRegs, VectorRegs>;
  let isCodeGenOnly = 1 in
  def NAME#_128B : T_HVX_alu2 <asmString, !cast<RegisterClass>(RC#"128B"),
                               VecPredRegs128B, VectorRegs128B>;
}

multiclass T_HVX_alu2_V <string asmString> :
  T_HVX_alu2 <asmString, VectorRegs>;

multiclass T_HVX_alu2_W <string asmString> :
  T_HVX_alu2 <asmString, VecDblRegs>;

defm V6_vswap : T_HVX_alu2_W <"$dst = vswap($src1,$src2,$src3)">, V6_vswap_enc;

let Itinerary = CVI_VA, Type = TypeCVI_VA, hasNewValue = 1,
    hasSideEffects = 0 in
defm V6_vmux  : T_HVX_alu2_V <"$dst = vmux($src1,$src2,$src3)">, V6_vmux_enc;

class T_HVX_vlutb <string asmString, RegisterClass RCout, RegisterClass RCin>
  : CVI_VA_Resource1<(outs RCout:$dst),
    (ins RCin:$src1, RCin:$src2, IntRegsLow8:$src3), asmString>;

multiclass T_HVX_vlutb <string asmString, RegisterClass RCout,
                        RegisterClass RCin> {
  def NAME : T_HVX_vlutb <asmString, RCout, RCin>;
  let isCodeGenOnly = 1 in
  def NAME#_128B : T_HVX_vlutb <asmString, !cast<RegisterClass>(RCout#"128B"),
                                           !cast<RegisterClass>(RCin#"128B")>;
}

multiclass T_HVX_vlutb_V <string asmString> :
  T_HVX_vlutb <asmString, VectorRegs, VectorRegs>;

multiclass T_HVX_vlutb_W <string asmString> :
  T_HVX_vlutb <asmString, VecDblRegs, VectorRegs>;

let Itinerary = CVI_VP_VS_LONG, Type = TypeCVI_VP_VS, isAccumulator = 1 in
class T_HVX_vlutb_acc <string asmString, RegisterClass RCout,
                       RegisterClass RCin>
  : CVI_VA_Resource1<(outs RCout:$dst),
    (ins RCout:$_src_, RCin:$src1, RCin:$src2, IntRegsLow8:$src3),
    asmString, [], "$dst = $_src_">;

multiclass T_HVX_vlutb_acc <string asmString, RegisterClass RCout,
                            RegisterClass RCin> {
  def NAME : T_HVX_vlutb_acc <asmString, RCout, RCin>;
  let isCodeGenOnly = 1 in
  def NAME#_128B : T_HVX_vlutb_acc<asmString,
                                   !cast<RegisterClass>(RCout#"128B"),
                                   !cast<RegisterClass>(RCin#"128B")>;
}

multiclass T_HVX_vlutb_acc_V <string asmString> :
  T_HVX_vlutb_acc <asmString, VectorRegs, VectorRegs>;

multiclass T_HVX_vlutb_acc_W <string asmString> :
  T_HVX_vlutb_acc <asmString, VecDblRegs, VectorRegs>;


let Itinerary = CVI_VP_LONG, Type = TypeCVI_VP, hasNewValue = 1 in
defm V6_vlutvvb:
     T_HVX_vlutb_V <"$dst.b = vlut32($src1.b,$src2.b,$src3)">, V6_vlutvvb_enc;

let Itinerary = CVI_VP_VS_LONG, Type = TypeCVI_VP_VS, hasNewValue = 1 in
defm V6_vlutvwh:
     T_HVX_vlutb_W <"$dst.h = vlut16($src1.b,$src2.h,$src3)">, V6_vlutvwh_enc;

let hasNewValue = 1 in {
  defm V6_vlutvvb_oracc:
       T_HVX_vlutb_acc_V <"$dst.b |= vlut32($src1.b,$src2.b,$src3)">,
       V6_vlutvvb_oracc_enc;
  defm V6_vlutvwh_oracc:
       T_HVX_vlutb_acc_W <"$dst.h |= vlut16($src1.b,$src2.h,$src3)">,
       V6_vlutvwh_oracc_enc;
}

// It's a fake instruction and should not be defined?
def S2_cabacencbin
  : SInst2<(outs DoubleRegs:$dst),
          (ins DoubleRegs:$src1, DoubleRegs:$src2, PredRegs:$src3),
    "$dst = encbin($src1,$src2,$src3)">, S2_cabacencbin_enc;

// Vhist instructions
def V6_vhistq
  : CVI_HIST_Resource1 <(outs), (ins VecPredRegs:$src1),
    "vhist($src1)">, V6_vhistq_enc;

def V6_vhist
  : CVI_HIST_Resource1 <(outs), (ins),
    "vhist" >, V6_vhist_enc;
