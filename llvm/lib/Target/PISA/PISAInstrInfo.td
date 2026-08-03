//===-- PISAInstrInfo.td - PISA Instruction defs -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "PISAInstrFormats.td"

// Templates for imm type checks
class uimmCheck<int num, ValueType VT>
  : ImmLeaf<VT, "return isUInt<"#num#">(Imm);">;
class simmCheck<int num, ValueType VT>
  : ImmLeaf<VT, "return isInt<"#num#">(Imm);">;
class immType<bit isSigned, ValueType VT, ValueType extVT = VT> {
  ImmLeaf check = !if(isSigned, simmCheck<VT.Size, extVT>,
                                uimmCheck<VT.Size, extVT>);
}

// 'afn' specified for call
class fastunaryop<SDPatternOperator nodeop> :
    PatFrag<(ops node:$src), (nodeop node:$src), [{ /* empty */ }]> {
  let GISelPredicateCode = [{ return useFastFP(MI); }];
}
// 'arcp' specified for call, used only for division
class fastbinop<SDPatternOperator nodeop> :
    PatFrag<(ops node:$lhs, node:$rhs), (nodeop node:$lhs, node:$rhs), [{ /* empty */ }]> {
  let GISelPredicateCode = [{ return MI.getFlag(MachineInstr::FmArcp); }];
}

// operand handling
class PrintDispatch<string PM, ValueType Ty, string OpType> : OperandWithDefaultOps<Ty, (ops (Ty 0))> {
  let PrintMethod = PM;
  let OperandNamespace = "PISA";
  let OperandType = OpType;
}
def Negate : PrintDispatch<"negateRepr", i32, "OPERAND_NEGATE"> {
  let ParserMatchClass = PISAAsmOperand<"NegateOpnd">;
}

// call target
def FunctionCallTargetAsmOperand : AsmOperandClass {
  let Name = "FunctionCallTargetOpnd";
  let ParserMethod = "parseFunctionCallTargetOpnd";
}
// ... register or globalVar
def FunctionCallTargetOpnd : Operand<i64> {
  let PrintMethod="printFunctionCallTargetOpnd";
  let ParserMatchClass = FunctionCallTargetAsmOperand;
}
// ... register
def IndirectFunctionCallTargetOpnd : PISARegisterOperand<Reg64b> {
  let PrintMethod="printFunctionCallTargetOpnd";
}


// mapping between LLVM and PISA names
class AtomicMemOrder<string MO> {
  string name = !if(!eq(MO,"monotonic"), "relaxed", MO);
}

// ftz modifier
let RecomputePerFunction = 1 in {
  def useFTZ : Predicate<"Subtarget->getTargetLowering()->useFTZ(*MF)">;
  def noFTZ : Predicate<"!Subtarget->getTargetLowering()->useFTZ(*MF)">;
}
class FTZModifier<string ftzName, string ftzOp, Predicate ftzPred> {
  string name = ftzName;
  string op = ftzOp;
  Predicate pred = ftzPred;
}
def noneFTZMod : FTZModifier<"", "", Predicate<"">>;
def omitFTZMod : FTZModifier<"", "", noFTZ>;
def usesFTZMod : FTZModifier<".ftz", "_ftz", useFTZ>;

// saturation modifier
def timm_false : TImmLeaf<i1, [{ return ((uint64_t)Imm) == 0;  }]>;
def timm_true : TImmLeaf<i1, [{ return ((uint64_t)Imm) == (uint64_t)-1; }]>;
class SaturationModifier<string satName, string satOp, TImmLeaf satNode> {
  string name = satName;
  string op = satOp;
  TImmLeaf node = satNode;
}
def NoSatMod   : SaturationModifier<"", "", timm_false>;
def DsatMod    : SaturationModifier<".dsat", "_sat", timm_true>;

// NaN-propagation modifier
class NanPropagateModifier<string nanpName, TImmLeaf nanpNode> {
  string name = nanpName;
  TImmLeaf node = nanpNode;
}
def NoNanpMod : NanPropagateModifier<"",      timm_false>;
def NanpMod   : NanPropagateModifier<".nanp", timm_true>;
defvar AllNanpMods = [NoNanpMod, NanpMod];

// wrapper of TImmLeaf with instruction qualifier name
class NamedTImmLeaf<string _name, ValueType vt, code pred> : TImmLeaf<vt, pred> {
  string name = _name;
}

// ired operation types (PISAIntrinsicUtils.h)
def IRedOp_SUM_timm    : NamedTImmLeaf<".sum",    i8, [{ return Imm == pisa::IRedOp::SUM; }]>;
def IRedOp_SMIN_timm   : NamedTImmLeaf<".smin",   i8, [{ return Imm == pisa::IRedOp::SMIN; }]>;
def IRedOp_SMAX_timm   : NamedTImmLeaf<".smax",   i8, [{ return Imm == pisa::IRedOp::SMAX; }]>;
def IRedOp_UMIN_timm   : NamedTImmLeaf<".umin",   i8, [{ return Imm == pisa::IRedOp::UMIN; }]>;
def IRedOp_UMAX_timm   : NamedTImmLeaf<".umax",   i8, [{ return Imm == pisa::IRedOp::UMAX; }]>;
def IRedOp_AND_timm    : NamedTImmLeaf<".and",    i8, [{ return Imm == pisa::IRedOp::AND; }]>;
def IRedOp_OR_timm     : NamedTImmLeaf<".or",     i8, [{ return Imm == pisa::IRedOp::OR; }]>;
def IRedOp_XOR_timm    : NamedTImmLeaf<".xor",    i8, [{ return Imm == pisa::IRedOp::XOR; }]>;
def IRedOp_ABSMAX_timm : NamedTImmLeaf<".absmax", i8, [{ return Imm == pisa::IRedOp::ABSMAX; }]>;
defvar IRedOpNodes = [IRedOp_SUM_timm, IRedOp_SMIN_timm, IRedOp_SMAX_timm,
                      IRedOp_UMIN_timm, IRedOp_UMAX_timm, IRedOp_AND_timm,
                      IRedOp_OR_timm, IRedOp_XOR_timm, IRedOp_ABSMAX_timm];

// fred operation types (PISAIntrinsicUtils.h)
def FRedOp_MIN_timm    : NamedTImmLeaf<".min",    i8, [{ return Imm == pisa::FRedOp::MIN; }]>;
def FRedOp_MAX_timm    : NamedTImmLeaf<".max",    i8, [{ return Imm == pisa::FRedOp::MAX; }]>;
def FRedOp_ABSMAX_timm : NamedTImmLeaf<".absmax", i8, [{ return Imm == pisa::FRedOp::ABSMAX; }]>;
defvar FRedOpNodes = [FRedOp_MIN_timm, FRedOp_MAX_timm, FRedOp_ABSMAX_timm];


// shfl mode types (PISAIntrinsicUtils.h)
def SHFLMode_UP_timm   : NamedTImmLeaf<".up",  i8, [{ return Imm == pisa::SHFLMode::UP; }]>;
def SHFLMode_DOWN_timm : NamedTImmLeaf<".dn",  i8, [{ return Imm == pisa::SHFLMode::DOWN; }]>;
def SHFLMode_XOR_timm  : NamedTImmLeaf<".xor", i8, [{ return Imm == pisa::SHFLMode::XOR; }]>;
def SHFLMode_IDX_timm  : NamedTImmLeaf<".idx", i8, [{ return Imm == pisa::SHFLMode::IDX; }]>;
defvar SHFLModeNodes = [SHFLMode_UP_timm, SHFLMode_DOWN_timm, SHFLMode_XOR_timm, SHFLMode_IDX_timm];


// memory ordering types (AtomicOrdering.h)
def MemoryOrder_relaxed_timm : NamedTImmLeaf<".relaxed", i8, [{ return Imm == (unsigned)llvm::AtomicOrdering::Monotonic; }]>;
def MemoryOrder_acquire_timm : NamedTImmLeaf<".acquire", i8, [{ return Imm == (unsigned)llvm::AtomicOrdering::Acquire; }]>;
def MemoryOrder_release_timm : NamedTImmLeaf<".release", i8, [{ return Imm == (unsigned)llvm::AtomicOrdering::Release; }]>;
def MemoryOrder_acq_rel_timm : NamedTImmLeaf<".acq_rel", i8, [{ return Imm == (unsigned)llvm::AtomicOrdering::AcquireRelease; }]>;
defvar MemoryOrderNodes = [MemoryOrder_relaxed_timm, MemoryOrder_acquire_timm, MemoryOrder_release_timm, MemoryOrder_acq_rel_timm];

// rounding mode types (FloatingPointMode.h)
def RoundingMode_RZ_timm  : NamedTImmLeaf<".rz",  i8, [{ return Imm == (int)llvm::RoundingMode::TowardZero; }]>;
def RoundingMode_RE_timm  : NamedTImmLeaf<".re",  i8, [{ return Imm == (int)llvm::RoundingMode::NearestTiesToEven; }]>;
def RoundingMode_RU_timm  : NamedTImmLeaf<".ru",  i8, [{ return Imm == (int)llvm::RoundingMode::TowardPositive; }]>;
def RoundingMode_RD_timm  : NamedTImmLeaf<".rd",  i8, [{ return Imm == (int)llvm::RoundingMode::TowardNegative; }]>;
def RoundingMode_RNA_timm : NamedTImmLeaf<".rna", i8, [{ return Imm == (int)llvm::RoundingMode::NearestTiesToAway; }]>;
def RoundingMode_INV_timm : NamedTImmLeaf<"",     i8, [{ return Imm == (int)llvm::RoundingMode::Invalid; }]>;
defvar RoundingModeNodes = [RoundingMode_RZ_timm, RoundingMode_RE_timm, RoundingMode_RU_timm, 
                            RoundingMode_RD_timm, RoundingMode_RNA_timm];
defvar RoundingModeNone  = [RoundingMode_INV_timm];
defvar RoundingModeAllNodes = !listconcat(RoundingModeNodes, RoundingModeNone);


//////
// BEGIN: special register definitions
multiclass SpecialRegPat<Intrinsic intr, Register reg> {
  defvar RetTy = intr.RetTypes[0];
  defvar VT = !if(!isa<LLVMQualPointerType>(RetTy),
                !cast<pisaVT>("_p"#!cast<LLVMQualPointerType>(RetTy).Sig[1]),
                !cast<pisaVT>("_"#RetTy.VT));
  def : Pat<(intr),
      (COPY_TO_REGCLASS (VT.VT reg), VT.RC)>;
}

defm : SpecialRegPat<int_pisa_local_id_x,            SpecialReg_LocalIdX>;
defm : SpecialRegPat<int_pisa_local_id_y,            SpecialReg_LocalIdY>;
defm : SpecialRegPat<int_pisa_local_id_z,            SpecialReg_LocalIdZ>;
defm : SpecialRegPat<int_pisa_local_size_x,          SpecialReg_LocalSizeX>;
defm : SpecialRegPat<int_pisa_local_size_y,          SpecialReg_LocalSizeY>;
defm : SpecialRegPat<int_pisa_local_size_z,          SpecialReg_LocalSizeZ>;
defm : SpecialRegPat<int_pisa_enqueued_local_size_x, SpecialReg_EnqueuedLocalSizeX>;
defm : SpecialRegPat<int_pisa_enqueued_local_size_y, SpecialReg_EnqueuedLocalSizeY>;
defm : SpecialRegPat<int_pisa_enqueued_local_size_z, SpecialReg_EnqueuedLocalSizeZ>;
defm : SpecialRegPat<int_pisa_global_offset_x,       SpecialReg_GlobalOffsetX>;
defm : SpecialRegPat<int_pisa_global_offset_y,       SpecialReg_GlobalOffsetY>;
defm : SpecialRegPat<int_pisa_global_offset_z,       SpecialReg_GlobalOffsetZ>;
defm : SpecialRegPat<int_pisa_global_size_x,         SpecialReg_GlobalSizeX>;
defm : SpecialRegPat<int_pisa_global_size_y,         SpecialReg_GlobalSizeY>;
defm : SpecialRegPat<int_pisa_global_size_z,         SpecialReg_GlobalSizeZ>;
defm : SpecialRegPat<int_pisa_group_id_x,            SpecialReg_GroupIdX>;
defm : SpecialRegPat<int_pisa_group_id_y,            SpecialReg_GroupIdY>;
defm : SpecialRegPat<int_pisa_group_id_z,            SpecialReg_GroupIdZ>;
defm : SpecialRegPat<int_pisa_group_count_x,         SpecialReg_GroupCountX>;
defm : SpecialRegPat<int_pisa_group_count_y,         SpecialReg_GroupCountY>;
defm : SpecialRegPat<int_pisa_group_count_z,         SpecialReg_GroupCountZ>;
defm : SpecialRegPat<int_pisa_subgroup_size,         SpecialReg_SubgroupSize>;
defm : SpecialRegPat<int_pisa_lane_id,               SpecialReg_LaneId>;
defm : SpecialRegPat<int_pisa_activemask,            SpecialReg_ActiveMask>;
defm : SpecialRegPat<int_pisa_work_dim,              SpecialReg_WorkDim>;
// END: special register definitions
//////

//////
// BEGIN: auxiliary class definitions
// generate a table that maps name to instruction
// - used in selection of insert/extract instructions
class Name2InstrEntry<string name, Instruction inst> {
  string Name = name;
  Instruction Inst = inst;
}
def Name2InstrOpTable : GenericTable {
  let FilterClass = "Name2InstrEntry";
  let Fields = ["Name", "Inst"];
}
def lookupName2InstrOpEntry : SearchIndex {
  let Table = Name2InstrOpTable;
  let Key = ["Name"];
}
// Generate a table that maps a register-class name to lifetime.start opcode.
class LifetimeStartEntry<string rcName, Instruction inst> {
  string RegClassName = rcName;
  Instruction Inst = inst;
}
def LifetimeStartTable : GenericTable {
  let FilterClass = "LifetimeStartEntry";
  let Fields = ["RegClassName", "Inst"];
}
def lookupLifetimeStartByRegClass : SearchIndex {
  let Table = LifetimeStartTable;
  let Key = ["RegClassName"];
}
// END: auxiliary class definitions
//////

//////
// BEGIN: unary instruction patterns
class UnaryInst_r<string name, SDPatternOperator node, pisaVT dstVT, pisaVT srcVT,
                  dag extraArgs = (ops)>
  : PISAInst_r<name, dstVT, srcVT> {
  let Pattern = !if(!gt(!size(extraArgs), 0),
    [(set (dstVT.VT dstRC:$dst), !con((node (srcVT.VT srcRC:$src)), extraArgs))],
    [(set (dstVT.VT dstRC:$dst), (node (srcVT.VT srcRC:$src)))]);
}
class UnaryInst_i<string name, SDPatternOperator node, pisaVT dstVT, pisaVT srcVT,
                  dag extraArgs = (ops)>
  : PISAInst_i<name, dstVT, srcVT> {
  let Pattern = !if(!gt(!size(extraArgs), 0),
    [(set (dstVT.VT dstRC:$dst), !con((node (srcVT.VT immVal:$src)), extraArgs))],
    [(set (dstVT.VT dstRC:$dst), (node (srcVT.VT immVal:$src)))]);
}

// unary integer instruction
multiclass UnaryInst<string name, SDPatternOperator node, list<pisaVT> types = [_i16, _i32, _i64]> {
  foreach VT = types in {
    defvar directive = VT.directive;
    defvar opcode = !subst(".", "_", directive);
    def opcode#"_r": UnaryInst_r<name#directive, node, VT, VT>;
    def opcode#"_i": UnaryInst_i<name#directive, node, VT, VT>;
  }
}

// unary float instruction
multiclass UnaryFloatInst<string name, SDPatternOperator node, list<pisaVT> types = [_bf16, _f16, _f32, _f64], bit isFast = 0> {
  foreach ftz = [omitFTZMod, usesFTZMod] in {
    defvar fastName = !if(isFast, ".fast", "");
    foreach VT = types in {
      defvar directive = !interleave([ftz.name, fastName, VT.directive], "");
      defvar opcode = !subst(".", "_", directive);
      def opcode#"_r": UnaryInst_r<name#directive, node, VT, VT>, Requires<[ftz.pred]>;
      def opcode#"_i": UnaryInst_i<name#directive, node, VT, VT>, Requires<[ftz.pred]>;
    }
  }
}

// unary float instruction with rounding mode
multiclass UnaryFloatInstRnd<string name, SDPatternOperator node, list<pisaVT> types = [_bf16, _f16, _f32, _f64]> {
  foreach ftz = [omitFTZMod, usesFTZMod] in {
    foreach VT = types in {
      foreach rndNode = RoundingModeNodes in {
        defvar directive = !interleave([rndNode.name, ftz.name, VT.directive], "");
        defvar opcode = !subst(".", "_", directive);
        def opcode#"_r": UnaryInst_r<name#directive, node, VT, VT, (node rndNode)>, Requires<[ftz.pred]>;
        def opcode#"_i": UnaryInst_i<name#directive, node, VT, VT, (node rndNode)>, Requires<[ftz.pred]>;
      }
    }
  }
}
// END: unary instruction patterns
//////

//////
// BEGIN: binary instruction patterns
class BinaryInst_rr<string name, SDPatternOperator node, pisaVT src0VT, pisaVT src1VT, pisaVT dstVT=src0VT,
                    dag extraArgs = (ops)>
  : PISAInst_rr<name, dstVT, src0VT, src1VT> {
  let Pattern = !if(!gt(!size(extraArgs), 0),
    [(set (dstVT.VT dstRC:$dst), !con((node (src0VT.VT src0RC:$src0), (src1VT.VT src1RC:$src1)), extraArgs))],
    [(set (dstVT.VT dstRC:$dst), (node (src0VT.VT src0RC:$src0), (src1VT.VT src1RC:$src1)))]);
}
class BinaryInst_ri<string name, SDPatternOperator node, pisaVT src0VT, pisaVT src1VT, pisaVT dstVT=src0VT,
                    dag extraArgs = (ops)>
  : PISAInst_ri<name, dstVT, src0VT, src1VT> {
  let Pattern = !if(!gt(!size(extraArgs), 0),
    [(set (dstVT.VT dstRC:$dst), !con((node (src0VT.VT src0RC:$src0), (src1VT.VT immVal1:$src1)), extraArgs))],
    [(set (dstVT.VT dstRC:$dst), (node (src0VT.VT src0RC:$src0), (src1VT.VT immVal1:$src1)))]);
}
class BinaryInst_ir<string name, SDPatternOperator node, pisaVT src0VT, pisaVT src1VT, pisaVT dstVT=src0VT,
                    bit isCommutative = true, dag extraArgs = (ops)>
  : PISAInst_ir<name, dstVT, src0VT, src1VT> {
  // Pattern is suppressed for commutative instructions: the canonicalizer always
  // places immediates on the RHS, so the _ir form would never match. Pass
  // isCommutative = false (e.g. for intrinsics) to emit a pattern.
  let Pattern = !if(isCommutative,
    [],
    !if(!gt(!size(extraArgs), 0),
      [(set (dstVT.VT dstRC:$dst), !con((node (src0VT.VT immVal0:$src0), (src1VT.VT src1RC:$src1)), extraArgs))],
      [(set (dstVT.VT dstRC:$dst), (node (src0VT.VT immVal0:$src0), (src1VT.VT src1RC:$src1)))]));
}
class BinaryInst_ii<string name, SDPatternOperator node, pisaVT src0VT, pisaVT src1VT, pisaVT dstVT=src0VT,
                    dag extraArgs = (ops)>
  : PISAInst_ii<name, dstVT, src0VT, src1VT> {
  let Pattern = !if(!gt(!size(extraArgs), 0),
    [(set (dstVT.VT dstRC:$dst), !con((node (src0VT.VT immVal0:$src0), (src1VT.VT immVal1:$src1)), extraArgs))],
    [(set (dstVT.VT dstRC:$dst), (node (src0VT.VT immVal0:$src0), (src1VT.VT immVal1:$src1)))]);
}

// binary integer instruction with optional saturation modifier
multiclass BinaryInst<string name, SDPatternOperator node, list<pisaVT> types = [_i16, _i32, _i64], bit isCommutative = true, list<SaturationModifier> satMods = [NoSatMod]> {
  foreach sat = satMods in {
    foreach VT = types in {
      defvar directive = !interleave([sat.name, VT.directive], "");
      defvar opcode = !subst(".", "_", directive);
      def opcode#"_rr": BinaryInst_rr<name#directive, node, VT, VT>;
      def opcode#"_ri": BinaryInst_ri<name#directive, node, VT, VT>;
      def opcode#"_ir": BinaryInst_ir<name#directive, node, VT, VT, VT, isCommutative>;
      def opcode#"_ii": BinaryInst_ii<name#directive, node, VT, VT>;
    }
  }
}

// binary float instruction with optional saturation modifier
multiclass BinaryFloatInst<string name, SDPatternOperator node, list<pisaVT> types = [_bf16, _f16, _f32, _f64], bit isCommutative = true, list<SaturationModifier> satMods = [NoSatMod], bit isFast = false> {
  foreach sat = satMods in {
    foreach ftz = [omitFTZMod, usesFTZMod] in {
      defvar fastName = !if(isFast, ".fast", "");
      foreach VT = types in {
        defvar directive = !interleave([ftz.name, fastName, sat.name, VT.directive], "");
        defvar opcode = !subst(".", "_", directive);
        def opcode#"_rr" : BinaryInst_rr<name#directive, node, VT, VT>, Requires<[ftz.pred]>;
        def opcode#"_ri" : BinaryInst_ri<name#directive, node, VT, VT>, Requires<[ftz.pred]>;
        def opcode#"_ir" : BinaryInst_ir<name#directive, node, VT, VT, VT, isCommutative>, Requires<[ftz.pred]>;
        def opcode#"_ii" : BinaryInst_ii<name#directive, node, VT, VT>, Requires<[ftz.pred]>;
      }
    }
  }
}

// binary float instruction with a trailing i1 nanp immediate
multiclass BinaryFloatInstNanp<string name, SDPatternOperator node, list<pisaVT> types = [_bf16, _f16, _f32, _f64], bit isCommutative = true, list<SaturationModifier> satMods = [NoSatMod]> {
  foreach sat = satMods in {
    foreach ftz = [omitFTZMod, usesFTZMod] in {
      foreach nanp = AllNanpMods in {
        foreach VT = types in {
          defvar directive = !interleave([ftz.name, sat.name, nanp.name, VT.directive], "");
          defvar opcode = !subst(".", "_", directive);
          def opcode#"_rr" : BinaryInst_rr<name#directive, node, VT, VT, VT, (node nanp.node)>, Requires<[ftz.pred]>;
          def opcode#"_ri" : BinaryInst_ri<name#directive, node, VT, VT, VT, (node nanp.node)>, Requires<[ftz.pred]>;
          def opcode#"_ir" : BinaryInst_ir<name#directive, node, VT, VT, VT, isCommutative, (node nanp.node)>, Requires<[ftz.pred]>;
          def opcode#"_ii" : BinaryInst_ii<name#directive, node, VT, VT, VT, (node nanp.node)>, Requires<[ftz.pred]>;
        }
      }
    }
  }
}

// binary float instruction with .nanp modifier only
multiclass BinaryFloatInstNanpOnly<string name, SDPatternOperator node, list<pisaVT> types = [_bf16, _f16, _f32, _f64], bit isCommutative = true> {
  foreach ftz = [omitFTZMod, usesFTZMod] in {
    foreach VT = types in {
      defvar directive = !interleave([ftz.name, ".nanp", VT.directive], "");
      defvar opcode = !subst(".", "_", directive);
      def opcode#"_rr" : BinaryInst_rr<name#directive, node, VT, VT>, Requires<[ftz.pred]>;
      def opcode#"_ri" : BinaryInst_ri<name#directive, node, VT, VT>, Requires<[ftz.pred]>;
      def opcode#"_ir" : BinaryInst_ir<name#directive, node, VT, VT, VT, isCommutative>, Requires<[ftz.pred]>;
      def opcode#"_ii" : BinaryInst_ii<name#directive, node, VT, VT>, Requires<[ftz.pred]>;
    }
  }
}

// binary float instruction with rounding mode
multiclass BinaryFloatInstRnd<string name, SDPatternOperator node, list<pisaVT> types = [_bf16, _f16, _f32, _f64], list<NamedTImmLeaf> rndNodes = RoundingModeNodes> {
  foreach ftz = [omitFTZMod, usesFTZMod] in {
    foreach VT = types in {
      foreach rndNode = rndNodes in {
        defvar directive = !interleave([rndNode.name, ftz.name, VT.directive], "");
        defvar opcode = !subst(".", "_", directive);
        def opcode#"_rrii": BinaryInst_rr<name#directive, node, VT, VT, VT, (node rndNode)>, Requires<[ftz.pred]>;
        def opcode#"_riii": BinaryInst_ri<name#directive, node, VT, VT, VT, (node rndNode)>, Requires<[ftz.pred]>;
        def opcode#"_irii": BinaryInst_ir<name#directive, node, VT, VT, VT, false, (node rndNode)>, Requires<[ftz.pred]>;
        def opcode#"_iiii": BinaryInst_ii<name#directive, node, VT, VT, VT, (node rndNode)>, Requires<[ftz.pred]>;
      }
    }
  }
}

// binary float instruction with rounding mode and saturation modifier
multiclass BinaryFloatInstRndSat<string name, SDPatternOperator node, list<pisaVT> types = [_bf16, _f16, _f32, _f64], list<NamedTImmLeaf> rndNodes = RoundingModeAllNodes, list<SaturationModifier> satMods = [NoSatMod, DsatMod], bit isCommutative = true, bit isFullPrecision = false> {
  defvar fullName = !if(isFullPrecision, ".full", "");
  foreach sat = satMods in {
    foreach ftz = [omitFTZMod, usesFTZMod] in {
      foreach VT = types in {
        foreach rndNode = rndNodes in {
          defvar directive = !interleave([fullName, rndNode.name, ftz.name, sat.name, VT.directive], "");
          defvar opcode = !subst(".", "_", directive);
          defvar resultVT = !if(isFullPrecision, _f32, VT);
          def opcode#"_rrii": BinaryInst_rr<name#directive, node, VT, VT, resultVT, (node rndNode, sat.node)>, Requires<[ftz.pred]>;
          def opcode#"_riii": BinaryInst_ri<name#directive, node, VT, VT, resultVT, (node rndNode, sat.node)>, Requires<[ftz.pred]>;
          def opcode#"_irii": BinaryInst_ir<name#directive, node, VT, VT, resultVT, isCommutative, (node rndNode, sat.node)>, Requires<[ftz.pred]>;
          def opcode#"_iiii": BinaryInst_ii<name#directive, node, VT, VT, resultVT, (node rndNode, sat.node)>, Requires<[ftz.pred]>;
        }
      }
    }
  }
}

// END: binary instruction patterns
//////

//////
// BEGIN: ternary instruction patterns
class TernaryInst_rrr<string name, SDPatternOperator node, pisaVT src0VT, pisaVT src2VT,
                      dag extraArgs = (ops)>
  : PISAInst_rrr<name, src2VT, src0VT, src0VT, src2VT> {
  let Pattern = !if(!gt(!size(extraArgs), 0),
    [(set (src0VT.VT dstRC:$dst), !con((node (src0VT.VT src0RC:$src0), (src0VT.VT src1RC:$src1), (src0VT.VT src2RC:$src2)), extraArgs))],
    [(set (src0VT.VT dstRC:$dst), (node (src0VT.VT src0RC:$src0), (src0VT.VT src1RC:$src1), (src0VT.VT src2RC:$src2)))]);
}
class TernaryInst_rri<string name, SDPatternOperator node, pisaVT src0VT, pisaVT src2VT,
                      dag extraArgs = (ops)>
  : PISAInst_rri<name, src2VT, src0VT, src0VT, src2VT> {
  let Pattern = !if(!gt(!size(extraArgs), 0),
    [(set (src0VT.VT dstRC:$dst), !con((node (src0VT.VT src0RC:$src0), (src0VT.VT src1RC:$src1), (src2VT.VT immVal2:$src2)), extraArgs))],
    [(set (src0VT.VT dstRC:$dst), (node (src0VT.VT src0RC:$src0), (src0VT.VT src1RC:$src1), (src2VT.VT immVal2:$src2)))]);
}
class TernaryInst_rir<string name, SDPatternOperator node, pisaVT src0VT, pisaVT src2VT,
                      dag extraArgs = (ops)>
  : PISAInst_rir<name, src2VT, src0VT, src0VT, src2VT> {
  let Pattern = !if(!gt(!size(extraArgs), 0),
    [(set (src0VT.VT dstRC:$dst), !con((node (src0VT.VT src0RC:$src0), (src0VT.VT immVal1:$src1), (src2VT.VT src2RC:$src2)), extraArgs))],
    [(set (src0VT.VT dstRC:$dst), (node (src0VT.VT src0RC:$src0), (src0VT.VT immVal1:$src1), (src2VT.VT src2RC:$src2)))]);
}
class TernaryInst_rii<string name, SDPatternOperator node, pisaVT src0VT, pisaVT src2VT,
                      dag extraArgs = (ops)>
  : PISAInst_rii<name, src2VT, src0VT, src0VT, src2VT> {
  let Pattern = !if(!gt(!size(extraArgs), 0),
    [(set (src0VT.VT dstRC:$dst), !con((node (src0VT.VT src0RC:$src0), (src0VT.VT immVal1:$src1), (src2VT.VT immVal2:$src2)), extraArgs))],
    [(set (src0VT.VT dstRC:$dst), (node (src0VT.VT src0RC:$src0), (src0VT.VT immVal1:$src1), (src2VT.VT immVal2:$src2)))]);
}
class TernaryInst_irr<string name, SDPatternOperator node, pisaVT src0VT, pisaVT src2VT,
                      dag extraArgs = (ops)>
  : PISAInst_irr<name, src2VT, src0VT, src0VT, src2VT> {
  let Pattern = !if(!gt(!size(extraArgs), 0),
    [(set (src0VT.VT dstRC:$dst), !con((node (src0VT.VT immVal0:$src0), (src0VT.VT src1RC:$src1), (src2VT.VT src2RC:$src2)), extraArgs))],
    [(set (src0VT.VT dstRC:$dst), (node (src0VT.VT immVal0:$src0), (src0VT.VT src1RC:$src1), (src2VT.VT src2RC:$src2)))]);
}
class TernaryInst_iri<string name, SDPatternOperator node, pisaVT src0VT, pisaVT src2VT,
                      dag extraArgs = (ops)>
  : PISAInst_iri<name, src2VT, src0VT, src0VT, src2VT> {
  let Pattern = !if(!gt(!size(extraArgs), 0),
    [(set (src0VT.VT dstRC:$dst), !con((node (src0VT.VT immVal0:$src0), (src0VT.VT src1RC:$src1), (src2VT.VT immVal2:$src2)), extraArgs))],
    [(set (src0VT.VT dstRC:$dst), (node (src0VT.VT immVal0:$src0), (src0VT.VT src1RC:$src1), (src2VT.VT immVal2:$src2)))]);
}
class TernaryInst_iir<string name, SDPatternOperator node, pisaVT src0VT, pisaVT src2VT,
                      dag extraArgs = (ops)>
  : PISAInst_iir<name, src2VT, src0VT, src0VT, src2VT> {
  let Pattern = !if(!gt(!size(extraArgs), 0),
    [(set (src0VT.VT dstRC:$dst), !con((node (src0VT.VT immVal0:$src0), (src0VT.VT immVal1:$src1), (src2VT.VT src2RC:$src2)), extraArgs))],
    [(set (src0VT.VT dstRC:$dst), (node (src0VT.VT immVal0:$src0), (src0VT.VT immVal1:$src1), (src2VT.VT src2RC:$src2)))]);
}
class TernaryInst_iii<string name, SDPatternOperator node, pisaVT src0VT, pisaVT src2VT,
                      dag extraArgs = (ops)>
  : PISAInst_iii<name, src2VT, src0VT, src0VT, src2VT> {
  let Pattern = !if(!gt(!size(extraArgs), 0),
    [(set (src0VT.VT dstRC:$dst), !con((node (src0VT.VT immVal0:$src0), (src0VT.VT immVal1:$src1), (src2VT.VT immVal2:$src2)), extraArgs))],
    [(set (src0VT.VT dstRC:$dst), (node (src0VT.VT immVal0:$src0), (src0VT.VT immVal1:$src1), (src2VT.VT immVal2:$src2)))]);
}

// ternary integer instruction with optional saturation modifier
multiclass TernaryInst<string name, SDPatternOperator node, list<pisaVT> types = [_i16, _i32, _i64], list<SaturationModifier> satMods = [NoSatMod]> {
  foreach sat = satMods in {
    foreach VT = types in {
      defvar directive = !interleave([sat.name, VT.directive], "");
      defvar opcode = !subst(".", "_", directive);
      def opcode#"_rrr" : TernaryInst_rrr<name#directive, node, VT, VT>;
      def opcode#"_rri" : TernaryInst_rri<name#directive, node, VT, VT>;
      def opcode#"_rir" : TernaryInst_rir<name#directive, node, VT, VT>;
      def opcode#"_rii" : TernaryInst_rii<name#directive, node, VT, VT>;
      def opcode#"_irr" : TernaryInst_irr<name#directive, node, VT, VT>;
      def opcode#"_iri" : TernaryInst_iri<name#directive, node, VT, VT>;
      def opcode#"_iir" : TernaryInst_iir<name#directive, node, VT, VT>;
      def opcode#"_iii" : TernaryInst_iii<name#directive, node, VT, VT>;
    }
  }
}

// ternary float instruction with optional saturation modifier
multiclass TernaryFloatInst<string name, SDPatternOperator node> {
  foreach sat = [NoSatMod] in {
    foreach ftz = [omitFTZMod, usesFTZMod] in {
      foreach VT = [_bf16, _f16, _f32, _f64] in {
        defvar directive = !interleave([ftz.name, sat.name, VT.directive], "");
        defvar opcode = !subst(".", "_", directive);
        def opcode#"_rrr" : TernaryInst_rrr<name#directive, node, VT, VT>, Requires<[ftz.pred]>;
        def opcode#"_rri" : TernaryInst_rri<name#directive, node, VT, VT>, Requires<[ftz.pred]>;
        def opcode#"_rir" : TernaryInst_rir<name#directive, node, VT, VT>, Requires<[ftz.pred]>;
        def opcode#"_rii" : TernaryInst_rii<name#directive, node, VT, VT>, Requires<[ftz.pred]>;
        def opcode#"_irr" : TernaryInst_irr<name#directive, node, VT, VT>, Requires<[ftz.pred]>;
        def opcode#"_iri" : TernaryInst_iri<name#directive, node, VT, VT>, Requires<[ftz.pred]>;
        def opcode#"_iir" : TernaryInst_iir<name#directive, node, VT, VT>, Requires<[ftz.pred]>;
        def opcode#"_iii" : TernaryInst_iii<name#directive, node, VT, VT>, Requires<[ftz.pred]>;
      }
    }
  }
}

// ternary float instruction with rounding mode and optional saturation modifier
multiclass TernaryFloatInstRndSat<string name, SDPatternOperator node, list<pisaVT> types> {
  foreach sat = [NoSatMod, DsatMod] in {
    foreach ftz = [omitFTZMod, usesFTZMod] in {
      foreach VT = types in {
        foreach rndNode = RoundingModeAllNodes in {
          defvar directive = !interleave([rndNode.name, ftz.name, sat.name, VT.directive], "");
          defvar opcode = !subst(".", "_", directive);
          def opcode#"_rrrii" : TernaryInst_rrr<name#directive, node, VT, VT, (node rndNode, sat.node)>, Requires<[ftz.pred]>;
          def opcode#"_rriii" : TernaryInst_rri<name#directive, node, VT, VT, (node rndNode, sat.node)>, Requires<[ftz.pred]>;
          def opcode#"_ririi" : TernaryInst_rir<name#directive, node, VT, VT, (node rndNode, sat.node)>, Requires<[ftz.pred]>;
          def opcode#"_riiii" : TernaryInst_rii<name#directive, node, VT, VT, (node rndNode, sat.node)>, Requires<[ftz.pred]>;
          def opcode#"_irrii" : TernaryInst_irr<name#directive, node, VT, VT, (node rndNode, sat.node)>, Requires<[ftz.pred]>;
          def opcode#"_iriii" : TernaryInst_iri<name#directive, node, VT, VT, (node rndNode, sat.node)>, Requires<[ftz.pred]>;
          def opcode#"_iirii" : TernaryInst_iir<name#directive, node, VT, VT, (node rndNode, sat.node)>, Requires<[ftz.pred]>;
          def opcode#"_iiiii" : TernaryInst_iii<name#directive, node, VT, VT, (node rndNode, sat.node)>, Requires<[ftz.pred]>;
        }
      }
    }
  }
}

// END: ternary instruction patterns
//////

//////
// BEGIN: convert instruction patterns
// convert instructions from multiple types to a single type
multiclass ConvertMultiInst<string name, SDPatternOperator Node, pisaVT dstTy, list<pisaVT> srcTys> {
  foreach srcTy = srcTys in {
    defvar directive = !interleave([dstTy.directive, srcTy.directive], "");
    defvar opcode = !subst(".", "_", directive);
    def opcode#"_r" : UnaryInst_r<name#directive, Node, dstTy, srcTy>;
    def opcode#"_i" : UnaryInst_i<name#directive, Node, dstTy, srcTy>;
  }
}

// convert float instructions from multiple types to a single type
multiclass ConvertMultiFloatInst<string name, SDPatternOperator node, pisaVT dstTy, list<pisaVT> srcTys> {
  foreach ftz = [omitFTZMod, usesFTZMod] in {
    foreach srcTy = srcTys in {
      defvar directive = !interleave([dstTy.directive, srcTy.directive, ftz.name], "");
      defvar opcode = !subst(".", "_", directive);
      def opcode#"_r": UnaryInst_r<name#directive, node, dstTy, srcTy>, Requires<[ftz.pred]>;
      def opcode#"_i": UnaryInst_i<name#directive, node, dstTy, srcTy>, Requires<[ftz.pred]>;
    }
  }
}

multiclass ConvertMultiFloatInstRnd<string name, SDPatternOperator node, pisaVT dstTy, list<pisaVT> srcTys> {
  foreach ftz = [omitFTZMod, usesFTZMod] in {
    foreach srcTy = srcTys in {
      foreach rndNode = RoundingModeNodes in {
        defvar directive = !interleave([dstTy.directive, srcTy.directive, rndNode.name, ftz.name], "");
        defvar opcode = !subst(".", "_", directive);
        def opcode#"_r": UnaryInst_r<name#directive, node, dstTy, srcTy, (node rndNode)>, Requires<[ftz.pred]>;
        def opcode#"_i": UnaryInst_i<name#directive, node, dstTy, srcTy, (node rndNode)>, Requires<[ftz.pred]>;
      }
    }
  }
}

// convert float instructions with rounding mode and saturation modifier
multiclass ConvertMultiFloatInstRndSat<string name, SDPatternOperator node, pisaVT dstTy, list<pisaVT> srcTys> {
  foreach sat = [NoSatMod, DsatMod] in {
    foreach ftz = [omitFTZMod, usesFTZMod] in {
      foreach srcTy = srcTys in {
        foreach rndNode = RoundingModeAllNodes in {
          defvar directive = !interleave([dstTy.directive, srcTy.directive, rndNode.name, ftz.name, sat.name], "");
          defvar opcode = !subst(".", "_", directive);
          def opcode#"_r": UnaryInst_r<name#directive, node, dstTy, srcTy, (node rndNode, sat.node)>, Requires<[ftz.pred]>;
          def opcode#"_i": UnaryInst_i<name#directive, node, dstTy, srcTy, (node rndNode, sat.node)>, Requires<[ftz.pred]>;
        }
      }
    }
  }
}

// convert instructions from single type to multiple types
multiclass ConvertSingleInst<string name, SDPatternOperator node, list<pisaVT> dstTys, pisaVT srcTy> {
  foreach dstTy = dstTys in {
    defvar directive = !interleave([dstTy.directive, srcTy.directive], "");
    defvar opcode = !subst(".", "_", directive);
    def opcode#"_r" : UnaryInst_r<name#directive, node, dstTy, srcTy>;
    def opcode#"_i" : UnaryInst_i<name#directive, node, dstTy, srcTy>;
  }
}
// END: convert instruction patterns
//////

// function parameters
let isMeta = 1 in {
  multiclass FunctionParameter {
    foreach VT = VTs.CallReturn in {
      def "_"#VT.name : PISAInst<(outs VT.RC:$var), (ins i32imm:$idx), " ; ">;
    }
  }
  defm functionParameter : FunctionParameter<>;
}

//////
// BEGIN: integer arithmetic instruction definitions
// => dp4a
multiclass DP4AInst {
  foreach node = [int_pisa_dp4a_uu, int_pisa_dp4a_us, int_pisa_dp4a_su, int_pisa_dp4a_ss] in {
    defvar type = !subst("int_pisa_dp4a_", "", !cast<string>(node));
    foreach sat = [NoSatMod, DsatMod] in {
      defvar invalid = !and(!eq(type, "uu"), !eq(sat, DsatMod));
      if !not(invalid) then {
        defvar directive = type#sat.name#".32b";
        defvar opcode = !subst(".", "_", directive);
        def opcode#"_rrr": TernaryInst_rrr<NAME#"."#directive, node, _i32, _i32, (node sat.node)>;
        def opcode#"_rri": TernaryInst_rri<NAME#"."#directive, node, _i32, _i32, (node sat.node)>;
        def opcode#"_rir": TernaryInst_rir<NAME#"."#directive, node, _i32, _i32, (node sat.node)>;
        def opcode#"_rii": TernaryInst_rii<NAME#"."#directive, node, _i32, _i32, (node sat.node)>;
        def opcode#"_irr": TernaryInst_irr<NAME#"."#directive, node, _i32, _i32, (node sat.node)>;
        def opcode#"_iri": TernaryInst_iri<NAME#"."#directive, node, _i32, _i32, (node sat.node)>;
        def opcode#"_iir": TernaryInst_iir<NAME#"."#directive, node, _i32, _i32, (node sat.node)>;
        def opcode#"_iii": TernaryInst_iii<NAME#"."#directive, node, _i32, _i32, (node sat.node)>;
      }
    }
  }
}
defm dp4a : DP4AInst;

// => iabs
defm iabs : UnaryInst<"iabs", abs>;


// => iadd
defm iadd: BinaryInst<"iadd", add>;
defm ptradd: BinaryInst<"iadd", ptradd>;

// .. with .dsat
defm iadd: BinaryInst<"iadd", saddsat, satMods=[DsatMod]>;


// => ineg
defm ineg: UnaryInst<"ineg", ineg>;

// => isub
defm isub: BinaryInst<"isub", sub, isCommutative=false>;
// .. with .dsat
defm isub: BinaryInst<"isub", ssubsat, isCommutative=false, satMods=[DsatMod]>;

// => sdiv
defm sdiv_quo: BinaryInst<"sdiv.quo", sdiv, isCommutative=false>;
defm sdiv: BinaryInst<"sdiv", sdiv, isCommutative=false>;
defm sdiv_rem: BinaryInst<"sdiv.rem", srem, isCommutative=false>;

// umad/smad
class MadInst_rrr<string name, pisaVT bitwidth> : PISAInst_rrr<name, bitwidth, bitwidth, bitwidth, bitwidth> {
  let Pattern = [(set dstRC:$dst, (add (mul src0RC:$src0, src1RC:$src1), src2RC:$src2))];
}
class MadInst_rri<string name, pisaVT bitwidth> : PISAInst_rri<name, bitwidth, bitwidth, bitwidth, bitwidth> {
  let Pattern = [(set dstRC:$dst, (add (mul src0RC:$src0, src1RC:$src1), (bitwidth.VT imm:$src2)))];
}
class MadInst_rir<string name, pisaVT bitwidth> : PISAInst_rir<name, bitwidth, bitwidth, bitwidth, bitwidth> {
  let Pattern = [(set dstRC:$dst, (add (mul src0RC:$src0, (bitwidth.VT imm:$src1)), src2RC:$src2))];
}
class MadInst_rii<string name, pisaVT bitwidth> : PISAInst_rii<name, bitwidth, bitwidth, bitwidth, bitwidth> {
  let Pattern = [(set dstRC:$dst, (add (mul src0RC:$src0, (bitwidth.VT imm:$src1)), (bitwidth.VT imm:$src2)))];
}
class MadInst_iii<string name, pisaVT bitwidth> : PISAInst_iii<name, bitwidth, bitwidth, bitwidth, bitwidth> {
  let Pattern = [(set dstRC:$dst, (add (mul (bitwidth.VT imm:$src0), (bitwidth.VT imm:$src1)), (bitwidth.VT imm:$src2)))];
}
class MadFullInst_rrr<string name, pisaVT bitwidth, pisaVT extBitwidth, bit isSigned> :
  PISAInst_rrr<name, extBitwidth, bitwidth, bitwidth, extBitwidth> {
  SDNode extNode = !if(isSigned, sext, zext);
  let Pattern = [(set dstRC:$dst, (add (mul (extNode src0RC:$src0), (extNode src1RC:$src1)), src2RC:$src2))];
}
class MadFullInst_rri<string name, pisaVT bitwidth, pisaVT extBitwidth, bit isSigned> :
  PISAInst_rri<name, extBitwidth, bitwidth, bitwidth, extBitwidth> {
  SDNode extNode = !if(isSigned, sext, zext);
  let Pattern = [(set dstRC:$dst, (add (mul (extNode src0RC:$src0), (extNode src1RC:$src1)), (extBitwidth.VT imm:$src2)))];
}
class MadFullInst_rir<string name, pisaVT bitwidth, pisaVT extBitwidth, bit isSigned> :
  PISAInst_rir<name, extBitwidth, bitwidth, extBitwidth, extBitwidth> {
  ImmLeaf immCheck1 = immType<isSigned, bitwidth.VT, extBitwidth.VT>.check;
  SDNode extNode = !if(isSigned, sext, zext);
  let Pattern = [(set dstRC:$dst, (add (mul (extNode src0RC:$src0), (immCheck1:$src1)), src2RC:$src2))];
}
class MadFullInst_rii<string name, pisaVT bitwidth, pisaVT extBitwidth, bit isSigned> :
  PISAInst_rii<name, extBitwidth, bitwidth, extBitwidth, extBitwidth> {
  ImmLeaf immCheck1 = immType<isSigned, bitwidth.VT, extBitwidth.VT>.check;
  SDNode extNode = !if(isSigned, sext, zext);
  let Pattern = [(set dstRC:$dst, (add (mul (extNode src0RC:$src0), (immCheck1:$src1)), (extBitwidth.VT imm:$src2)))];
}
multiclass MadInst<string name, bit isSigned> {
  foreach VT = [_i16, _i32, _i64] in {
    defvar directive = VT.directive;
    defvar opcode = !subst(".", "_", directive);
    def opcode#"_rrr" : MadInst_rrr<name#directive, VT>;
    def opcode#"_rri" : MadInst_rri<name#directive, VT>;
    def opcode#"_rir" : MadInst_rir<name#directive, VT>;
    def opcode#"_rii" : MadInst_rii<name#directive, VT>;
    def opcode#"_irr" : PISAInst_irr<name#directive, VT, VT, VT, VT>;
    def opcode#"_iri" : PISAInst_iri<name#directive, VT, VT, VT, VT>;
    def opcode#"_iir" : PISAInst_iir<name#directive, VT, VT, VT, VT>;
    def opcode#"_iii" : MadInst_iii<name#directive, VT>;
  }
  foreach VT = [_i16, _i32] in {
    defvar extBitwidth = !if(!eq(VT, _i16), _i32, _i64);
    defvar directive = ".full"#VT.directive;
    defvar opcode = !subst(".", "_", directive);
    def opcode#"_rrr" : MadFullInst_rrr<name#directive, VT, extBitwidth, isSigned>;
    def opcode#"_rir" : MadFullInst_rir<name#directive, VT, extBitwidth, isSigned>;
    def opcode#"_rri" : MadFullInst_rri<name#directive, VT, extBitwidth, isSigned>;
    def opcode#"_rii" : MadFullInst_rii<name#directive, VT, extBitwidth, isSigned>;
    def opcode#"_irr" : PISAInst_irr<name#directive, extBitwidth, extBitwidth, VT, extBitwidth>;
    def opcode#"_iir" : PISAInst_iir<name#directive, extBitwidth, extBitwidth, extBitwidth, extBitwidth>;
    def opcode#"_iri" : PISAInst_iri<name#directive, extBitwidth, extBitwidth, VT, extBitwidth>;
    def opcode#"_iii" : PISAInst_iii<name#directive, extBitwidth, extBitwidth, extBitwidth, extBitwidth>;
  }
}

// => smad
defm smad: MadInst<"smad", 1>;
defm int_smad: TernaryInst<"smad", int_pisa_smad>;

// => umad
defm umad: MadInst<"umad", 0>;

// => smax
defm smax: BinaryInst<"smax", smax>;

// => smin
defm smin: BinaryInst<"smin", smin>;

// => smul/umul
// umul/smul
class MulFullInst_rr<string name, pisaVT bitwidth, pisaVT extBitwidth, bit isSigned>
  : PISAInst2<name, extBitwidth, bitwidth, bitwidth> {
  let InOperandList = (ins src0RC:$src0, src1RC:$src1);
  SDNode extNode = !if(isSigned, sext, zext);
  let Pattern = [(set dstRC:$dst, (mul (extNode src0RC:$src0), (extNode src1RC:$src1)))];
}
class MulFullInst_ri<string name, pisaVT bitwidth, pisaVT extBitwidth, bit isSigned>
  : PISAInst2<name, extBitwidth, bitwidth, bitwidth> {
  ImmLeaf immCheck1 = immType<isSigned, bitwidth.VT, extBitwidth.VT>.check;
  DAGOperand immOpd1 = extBitwidth.ImmOpnd;
  let InOperandList = (ins src0RC:$src0, immOpd1:$src1);
  SDNode extNode = !if(isSigned, sext, zext);
  let Pattern = [(set dstRC:$dst, (mul (extNode src0RC:$src0), (immCheck1:$src1)))];
}
class MulFullInst_ir<string name, pisaVT bitwidth, pisaVT extBitwidth, bit isSigned>
  : PISAInst2<name, extBitwidth, bitwidth, bitwidth> {
  DAGOperand immOpd0 = extBitwidth.ImmOpnd;
  let InOperandList = (ins immOpd0:$src0, src1RC:$src1);
  SDNode extNode = !if(isSigned, sext, zext);
  let Pattern = [];
}
class MulFullInst_ii<string name, pisaVT bitwidth, pisaVT extBitwidth, bit isSigned>
  : PISAInst2<name, extBitwidth, bitwidth, bitwidth> {
  DAGOperand immOpd0 = extBitwidth.ImmOpnd;
  DAGOperand immOpd1 = extBitwidth.ImmOpnd;
  let InOperandList = (ins immOpd0:$src0, immOpd1:$src1);
  SDNode extNode = !if(isSigned, sext, zext);
  let Pattern = [];
}
multiclass MulInst<string name, bit isSigned> {
  defm NAME: BinaryInst<name, mul>;
  foreach VT = [_i16, _i32] in {
    defvar extBitwidth = !if(!eq(VT, _i16), _i32, _i64);
    defvar directive = ".full"#VT.directive;
    defvar opcode = !subst(".", "_", directive);
    def opcode#"_rr" : MulFullInst_rr<name#directive, VT, extBitwidth, isSigned>;
    def opcode#"_ri" : MulFullInst_ri<name#directive, VT, extBitwidth, isSigned>;
    def opcode#"_ir" : MulFullInst_ir<name#directive, VT, extBitwidth, isSigned>;
    def opcode#"_ii" : MulFullInst_ii<name#directive, VT, extBitwidth, isSigned>;
  }
}
defm smul: MulInst<"smul", isSigned=true>;
defm umul: MulInst<"umul", isSigned=false>;

// uaddc/usubb
class CarryOutInst_rr<string name, pisaVT VT> : PISAInst_rr<name, VT, VT, VT> {
  DAGOperand PredRC = _i1.RegOpnd;
  let AsmString = name#" \t$dst, $cout, $src0, $src1;";
  let OutOperandList = (outs dstRC:$dst, PredRC:$cout);
  let InOperandList = (ins src0RC:$src0, src1RC:$src1);
}
class CarryOutInst_ri<string name, pisaVT VT> : PISAInst_ri<name, VT, VT, VT> {
  DAGOperand PredRC = _i1.RegOpnd;
  let AsmString = name#" \t$dst, $cout, $src0, $src1;";
  let OutOperandList = (outs dstRC:$dst, PredRC:$cout);
  let InOperandList = (ins src0RC:$src0, immOpd1:$src1);
}
class CarryOutInst_ir<string name, pisaVT VT> : PISAInst_ir<name, VT, VT, VT> {
  DAGOperand PredRC = _i1.RegOpnd;
  let AsmString = name#" \t$dst, $cout, $src0, $src1;";
  let OutOperandList = (outs dstRC:$dst, PredRC:$cout);
  let InOperandList = (ins immOpd0:$src0, src1RC:$src1);
}
class CarryOutInst_ii<string name, pisaVT VT> : PISAInst_ii<name, VT, VT, VT> {
  DAGOperand PredRC = _i1.RegOpnd;
  let AsmString = name#" \t$dst, $cout, $src0, $src1;";
  let OutOperandList = (outs dstRC:$dst, PredRC:$cout);
  let InOperandList = (ins immOpd0:$src0, immOpd1:$src1);
}
multiclass CarryOutInst<string name> {
  foreach VT = [_i32] in {
    defvar directive = VT.directive;
    defvar opcode = !subst(".", "_", directive);
    def opcode#"_rr" : CarryOutInst_rr<name#directive, VT>;
    def opcode#"_ri" : CarryOutInst_ri<name#directive, VT>;
    def opcode#"_ir" : CarryOutInst_ir<name#directive, VT>;
    def opcode#"_ii" : CarryOutInst_ii<name#directive, VT>;
  }
}
class CarryInInst_rr<string name, pisaVT VT> : PISAInst_rr<name, VT, VT, VT> {
  DAGOperand PredRC = _i1.RegOpnd;
  let AsmString = name#" \t$dst, $src0, $src1, $cin;";
  let OutOperandList = (outs dstRC:$dst);
  let InOperandList = (ins src0RC:$src0, src1RC:$src1, PredRC:$cin);
}
class CarryInInst_ri<string name, pisaVT VT> : PISAInst_ri<name, VT, VT, VT> {
  DAGOperand PredRC = _i1.RegOpnd;
  let AsmString = name#" \t$dst, $src0, $src1, $cin;";
  let OutOperandList = (outs dstRC:$dst);
  let InOperandList = (ins src0RC:$src0, immOpd1:$src1, PredRC:$cin);
}
class CarryInInst_ir<string name, pisaVT VT> : PISAInst_ir<name, VT, VT, VT> {
  DAGOperand PredRC = _i1.RegOpnd;
  let AsmString = name#" \t$dst, $src0, $src1, $cin;";
  let OutOperandList = (outs dstRC:$dst);
  let InOperandList = (ins immOpd0:$src0, src1RC:$src1, PredRC:$cin);
}
class CarryInInst_ii<string name, pisaVT VT> : PISAInst_ii<name, VT, VT, VT> {
  DAGOperand PredRC = _i1.RegOpnd;
  let AsmString = name#" \t$dst, $src0, $src1, $cin;";
  let OutOperandList = (outs dstRC:$dst);
  let InOperandList = (ins immOpd0:$src0, immOpd1:$src1, PredRC:$cin);
}
multiclass CarryInInst<string name> {
  foreach VT = [_i32] in {
    defvar directive = VT.directive;
    defvar opcode = !subst(".", "_", directive);
    def opcode#"_rr" : CarryInInst_rr<name#directive, VT>;
    def opcode#"_ri" : CarryInInst_ri<name#directive, VT>;
    def opcode#"_ir" : CarryInInst_ir<name#directive, VT>;
    def opcode#"_ii" : CarryInInst_ii<name#directive, VT>;
  }
}
class CarryInOutInst_rr<string name, pisaVT VT> : PISAInst_rr<name, VT, VT, VT> {
  DAGOperand PredRC = _i1.RegOpnd;
  let AsmString = name#" \t$dst, $cout, $src0, $src1, $cin;";
  let OutOperandList = (outs dstRC:$dst, PredRC:$cout);
  let InOperandList = (ins src0RC:$src0, src1RC:$src1, PredRC:$cin);
}
class CarryInOutInst_ri<string name, pisaVT VT> : PISAInst_ri<name, VT, VT, VT> {
  DAGOperand PredRC = _i1.RegOpnd;
  let AsmString = name#" \t$dst, $cout, $src0, $src1, $cin;";
  let OutOperandList = (outs dstRC:$dst, PredRC:$cout);
  let InOperandList = (ins src0RC:$src0, immOpd1:$src1, PredRC:$cin);
}
class CarryInOutInst_ir<string name, pisaVT VT> : PISAInst_ir<name, VT, VT, VT> {
  DAGOperand PredRC = _i1.RegOpnd;
  let AsmString = name#" \t$dst, $cout, $src0, $src1, $cin;";
  let OutOperandList = (outs dstRC:$dst, PredRC:$cout);
  let InOperandList = (ins immOpd0:$src0, src1RC:$src1, PredRC:$cin);
}
class CarryInOutInst_ii<string name, pisaVT VT> : PISAInst_ii<name, VT, VT, VT> {
  DAGOperand PredRC = _i1.RegOpnd;
  let AsmString = name#" \t$dst, $cout, $src0, $src1, $cin;";
  let OutOperandList = (outs dstRC:$dst, PredRC:$cout);
  let InOperandList = (ins immOpd0:$src0, immOpd1:$src1, PredRC:$cin);
}
multiclass CarryInOutInst<string name> {
  foreach VT = [_i32] in {
    defvar directive = VT.directive;
    defvar opcode = !subst(".", "_", directive);
    def opcode#"_rr" : CarryInOutInst_rr<name#directive, VT>;
    def opcode#"_ri" : CarryInOutInst_ri<name#directive, VT>;
    def opcode#"_ir" : CarryInOutInst_ir<name#directive, VT>;
    def opcode#"_ii" : CarryInOutInst_ii<name#directive, VT>;
  }
}

// => uaddc
defm uaddc_co    : CarryOutInst<"uaddc.co">;
defm uaddc_ci    : CarryInInst<"uaddc.ci">;
defm uaddc_ci_co : CarryInOutInst<"uaddc.ci.co">;

// => usubb
defm usubb_co    : CarryOutInst<"usubb.co">;
defm usubb_ci    : CarryInInst<"usubb.ci">;
defm usubb_ci_co : CarryInOutInst<"usubb.ci.co">;

// => udiv
defm udiv_quo: BinaryInst<"udiv.quo", udiv, isCommutative=false>;
defm udiv: BinaryInst<"udiv", udiv, isCommutative=false>;
defm udiv_rem: BinaryInst<"udiv.rem", urem, isCommutative=false>;

// => umax
defm umax: BinaryInst<"umax", umax>;

// => umin
defm umin: BinaryInst<"umin", umin>;

// END: integer arithmetic instruction definitions
//////

//////
// BEGIN: floating point instruction definitions
// => fabs
defm fabs: UnaryFloatInst<"fabs", int_pisa_fabs>;


// => fadd
defm fadd: BinaryFloatInst<"fadd", fadd>;
defm fadd: BinaryFloatInstRndSat<"fadd", int_pisa_fadd>;

// => fcos
defm fcos: UnaryFloatInst<"fcos", fcos, types=[_bf16, _f16, _f32]>;

// => fdiv
defm fdiv: BinaryFloatInst<"fdiv", fdiv, types=[_f32, _f64], isCommutative=false>;
defm fdiv: BinaryFloatInst<"fdiv", fastbinop<fdiv>, isCommutative=false, isFast=1>;
defm fdiv: BinaryFloatInstRnd<"fdiv", int_pisa_fdiv_rnd, [_f32, _f64]>;

// => fexp2
defm fexp2: UnaryFloatInst<"fexp2", fexp2, types=[_bf16, _f16, _f32]>;

// => flog2
defm flog2: UnaryFloatInst<"flog2", flog2, types=[_bf16, _f16, _f32]>;

// => fmad
defm fmad: TernaryFloatInst<"fmad", fma>;
defm fmad: TernaryFloatInstRndSat<"fmad", int_pisa_fma, [_bf16, _f16, _f32, _f64]>;


// => fmax
defm fmax: BinaryFloatInst<"fmax", fmaxnum>;
defm fmax: BinaryFloatInstNanp<"fmax", int_pisa_fmax_sat, satMods=[DsatMod]>;
// fmax.nanp (NaN-propagating, no saturation) <= fmaximum, the
// IEEE-754 maximum (any NaN input -> NaN). The plain fmax above (<= fmaxnum) is
// maximumNumber, which suppresses NaN (returns the non-NaN operand).
defm fmax: BinaryFloatInstNanpOnly<"fmax", fmaximum>;

// => fmin
defm fmin: BinaryFloatInst<"fmin", fminnum>;
defm fmin: BinaryFloatInstNanp<"fmin", int_pisa_fmin_sat, satMods=[DsatMod]>;
// fmin.nanp (NaN-propagating, no saturation) <= fminimum, the
// IEEE-754 minimum (any NaN input -> NaN). The plain fmin above (<= fminnum) is
// minimumNumber, which suppresses NaN (returns the non-NaN operand).
defm fmin: BinaryFloatInstNanpOnly<"fmin", fminimum>;

// => fmul
defm fmul: BinaryFloatInst<"fmul", fmul>;
defm fmul: BinaryFloatInstRndSat<"fmul", int_pisa_fmul>;

// => fneg
defm fneg: UnaryFloatInst<"fneg", fneg>;

// => frc
defm frc:  UnaryFloatInst<"frc", int_pisa_frc, [_f32]>;

// => frcp
defm frcp: UnaryFloatInst<"frcp", int_pisa_frcp, types=[_bf16, _f16, _f32]>;

// => frsqrt
defm frsqrt: UnaryFloatInst<"frsqrt", int_pisa_frsqrt, isFast=1>;


// => fsin
defm fsin: UnaryFloatInst<"fsin", fsin, types=[_bf16, _f16, _f32]>;

// => fsqrt
defm fsqrt: UnaryFloatInst<"fsqrt", fsqrt, types=[_f32, _f64]>;
defm fsqrt: UnaryFloatInst<"fsqrt", fastunaryop<fsqrt>, types=[_f32, _f64], isFast=1>;
defm fsqrt: UnaryFloatInst<"fsqrt", fsqrt, types=[_bf16, _f16], isFast=1>;
defm fsqrt: UnaryFloatInstRnd<"fsqrt", int_pisa_fsqrt_rnd, types=[_f32, _f64]>;

// => fsub
defm fsub: BinaryFloatInst<"fsub", fsub, isCommutative=false>;
defm fsub: BinaryFloatInstRndSat<"fsub", int_pisa_fsub, isCommutative=false>;

// => ftanh
defm ftanh: UnaryFloatInst<"ftanh", ftanh, types=[_bf16, _f16, _f32]>;
// END: floating point instruction definitions
//////

//////
// BEGIN: bitwise instruction definitions
// => and
defm and   : BinaryInst<"and", and>;

// => bfi
class Bfi_rrrr<string name, SDPatternOperator node, pisaVT VT>
  : PISAInst4<name, VT, VT, VT, VT, VT> {
  let InOperandList = (ins src0RC:$src0, src1RC:$src1, src2RC:$src2, src3RC:$src3);
  let Pattern = [(set dstRC:$dst, (node src0RC:$src0, src1RC:$src1, src2RC:$src2, src3RC:$src3))];
}
class Bfi_rrri<string name, SDPatternOperator node, pisaVT VT>
  : PISAInst4<name, VT, VT, VT, VT, VT> {
  let InOperandList = (ins src0RC:$src0, src1RC:$src1, src2RC:$src2, VT.ImmOpnd:$src3);
  let Pattern = [(set dstRC:$dst, (node src0RC:$src0, src1RC:$src1, src2RC:$src2, (VT.VT imm:$src3)))];
}
class Bfi_rrir<string name, SDPatternOperator node, pisaVT VT>
  : PISAInst4<name, VT, VT, VT, VT, VT> {
  let InOperandList = (ins src0RC:$src0, src1RC:$src1, VT.ImmOpnd:$src2, src3RC:$src3);
  let Pattern = [(set dstRC:$dst, (node src0RC:$src0, src1RC:$src1, (VT.VT imm:$src2), src3RC:$src3))];
}
class Bfi_rrii<string name, SDPatternOperator node, pisaVT VT>
  : PISAInst4<name, VT, VT, VT, VT, VT> {
  let InOperandList = (ins src0RC:$src0, src1RC:$src1, VT.ImmOpnd:$src2, VT.ImmOpnd:$src3);
  let Pattern = [(set dstRC:$dst, (node src0RC:$src0, src1RC:$src1, (VT.VT imm:$src2), (VT.VT imm:$src3)))];
}
class Bfi_rirr<string name, SDPatternOperator node, pisaVT VT>
  : PISAInst4<name, VT, VT, VT, VT, VT> {
  let InOperandList = (ins src0RC:$src0, VT.ImmOpnd:$src1, src2RC:$src2, src3RC:$src3);
  let Pattern = [(set dstRC:$dst, (node src0RC:$src0, (VT.VT imm:$src1), src2RC:$src2, src3RC:$src3))];
}
class Bfi_riri<string name, SDPatternOperator node, pisaVT VT>
  : PISAInst4<name, VT, VT, VT, VT, VT> {
  let InOperandList = (ins src0RC:$src0, VT.ImmOpnd:$src1, src2RC:$src2, VT.ImmOpnd:$src3);
  let Pattern = [(set dstRC:$dst, (node src0RC:$src0, (VT.VT imm:$src1), src2RC:$src2, (VT.VT imm:$src3)))];
}
class Bfi_riir<string name, SDPatternOperator node, pisaVT VT>
  : PISAInst4<name, VT, VT, VT, VT, VT> {
  let InOperandList = (ins src0RC:$src0, VT.ImmOpnd:$src1, VT.ImmOpnd:$src2, src3RC:$src3);
  let Pattern = [(set dstRC:$dst, (node src0RC:$src0, (VT.VT imm:$src1), (VT.VT imm:$src2), src3RC:$src3))];
}
class Bfi_riii<string name, SDPatternOperator node, pisaVT VT>
  : PISAInst4<name, VT, VT, VT, VT, VT> {
  let InOperandList = (ins src0RC:$src0, VT.ImmOpnd:$src1, VT.ImmOpnd:$src2, VT.ImmOpnd:$src3);
  let Pattern = [(set dstRC:$dst, (node src0RC:$src0, (VT.VT imm:$src1), (VT.VT imm:$src2), (VT.VT imm:$src3)))];
}
class Bfi_irrr<string name, SDPatternOperator node, pisaVT VT>
  : PISAInst4<name, VT, VT, VT, VT, VT> {
  let InOperandList = (ins VT.ImmOpnd:$src0, src1RC:$src1, src2RC:$src2, src3RC:$src3);
  let Pattern = [(set dstRC:$dst, (node (VT.VT imm:$src0), src1RC:$src1, src2RC:$src2, src3RC:$src3))];
}
class Bfi_irri<string name, SDPatternOperator node, pisaVT VT>
  : PISAInst4<name, VT, VT, VT, VT, VT> {
  let InOperandList = (ins VT.ImmOpnd:$src0, src1RC:$src1, src2RC:$src2, VT.ImmOpnd:$src3);
  let Pattern = [(set dstRC:$dst, (node (VT.VT imm:$src0), src1RC:$src1, src2RC:$src2, (VT.VT imm:$src3)))];
}
class Bfi_irir<string name, SDPatternOperator node, pisaVT VT>
  : PISAInst4<name, VT, VT, VT, VT, VT> {
  let InOperandList = (ins VT.ImmOpnd:$src0, src1RC:$src1, VT.ImmOpnd:$src2, src3RC:$src3);
  let Pattern = [(set dstRC:$dst, (node (VT.VT imm:$src0), src1RC:$src1, (VT.VT imm:$src2), src3RC:$src3))];
}
class Bfi_irii<string name, SDPatternOperator node, pisaVT VT>
  : PISAInst4<name, VT, VT, VT, VT, VT> {
  let InOperandList = (ins VT.ImmOpnd:$src0, src1RC:$src1, VT.ImmOpnd:$src2, VT.ImmOpnd:$src3);
  let Pattern = [(set dstRC:$dst, (node (VT.VT imm:$src0), src1RC:$src1, (VT.VT imm:$src2), (VT.VT imm:$src3)))];
}
class Bfi_iirr<string name, SDPatternOperator node, pisaVT VT>
  : PISAInst4<name, VT, VT, VT, VT, VT> {
  let InOperandList = (ins VT.ImmOpnd:$src0, VT.ImmOpnd:$src1, src2RC:$src2, src3RC:$src3);
  let Pattern = [(set dstRC:$dst, (node (VT.VT imm:$src0), (VT.VT imm:$src1), src2RC:$src2, src3RC:$src3))];
}
class Bfi_iiri<string name, SDPatternOperator node, pisaVT VT>
  : PISAInst4<name, VT, VT, VT, VT, VT> {
  let InOperandList = (ins VT.ImmOpnd:$src0, VT.ImmOpnd:$src1, src2RC:$src2, VT.ImmOpnd:$src3);
  let Pattern = [(set dstRC:$dst, (node (VT.VT imm:$src0), (VT.VT imm:$src1), src2RC:$src2, (VT.VT imm:$src3)))];
}
class Bfi_iiir<string name, SDPatternOperator node, pisaVT VT>
  : PISAInst4<name, VT, VT, VT, VT, VT> {
  let InOperandList = (ins VT.ImmOpnd:$src0, VT.ImmOpnd:$src1, VT.ImmOpnd:$src2, src3RC:$src3);
  let Pattern = [(set dstRC:$dst, (node (VT.VT imm:$src0), (VT.VT imm:$src1), (VT.VT imm:$src2), src3RC:$src3))];
}
class Bfi_iiii<string name, SDPatternOperator node, pisaVT VT>
  : PISAInst4<name, VT, VT, VT, VT, VT> {
  let InOperandList = (ins VT.ImmOpnd:$src0, VT.ImmOpnd:$src1, VT.ImmOpnd:$src2, VT.ImmOpnd:$src3);
  let Pattern = [(set dstRC:$dst, (node (VT.VT imm:$src0), (VT.VT imm:$src1), (VT.VT imm:$src2), (VT.VT imm:$src3)))];
}
multiclass BfiInst {
  defvar directive = "bfi.32b";
  def _rrrr: Bfi_rrrr<directive, int_pisa_bfi, _i32>;
  def _rrri: Bfi_rrri<directive, int_pisa_bfi, _i32>;
  def _rrir: Bfi_rrir<directive, int_pisa_bfi, _i32>;
  def _rrii: Bfi_rrii<directive, int_pisa_bfi, _i32>;
  def _rirr: Bfi_rirr<directive, int_pisa_bfi, _i32>;
  def _riri: Bfi_riri<directive, int_pisa_bfi, _i32>;
  def _riir: Bfi_riir<directive, int_pisa_bfi, _i32>;
  def _riii: Bfi_riii<directive, int_pisa_bfi, _i32>;
  def _irrr: Bfi_irrr<directive, int_pisa_bfi, _i32>;
  def _irri: Bfi_irri<directive, int_pisa_bfi, _i32>;
  def _irir: Bfi_irir<directive, int_pisa_bfi, _i32>;
  def _irii: Bfi_irii<directive, int_pisa_bfi, _i32>;
  def _iirr: Bfi_iirr<directive, int_pisa_bfi, _i32>;
  def _iiri: Bfi_iiri<directive, int_pisa_bfi, _i32>;
  def _iiir: Bfi_iiir<directive, int_pisa_bfi, _i32>;
  def _iiii: Bfi_iiii<directive, int_pisa_bfi, _i32>;
}
defm bfi : BfiInst;

// => bfn
class Bfn_irrr<string name, SDPatternOperator node, pisaVT opVT, pisaVT VT>
  : PISAInst4<name, VT, opVT, VT, VT, VT> {
  DAGOperand BfnOpcodeOpnd = PISABfnOpcodeOperand<opVT.VT, "Imm"#opVT.VT.Size#"Opnd">;
  let InOperandList = (ins BfnOpcodeOpnd:$src0, src1RC:$src1, src2RC:$src2, src3RC:$src3);
  let Pattern = [(set dstRC:$dst, (node timm:$src0, src1RC:$src1, src2RC:$src2, src3RC:$src3))];
  let AsmString = name#".${src0}"#VT.directive#" \t$dst, $src1, $src2, $src3;";
}
class Bfn_irri<string name, SDPatternOperator node, pisaVT opVT, pisaVT VT>
  : PISAInst4<name, VT, opVT, VT, VT, VT> {
  DAGOperand BfnOpcodeOpnd = PISABfnOpcodeOperand<opVT.VT, "Imm"#opVT.VT.Size#"Opnd">;
  let InOperandList = (ins BfnOpcodeOpnd:$src0, src1RC:$src1, src2RC:$src2, VT.ImmOpnd:$src3);
  let Pattern = [(set dstRC:$dst, (node timm:$src0, src1RC:$src1, src2RC:$src2, (VT.VT imm:$src3)))];
  let AsmString = name#".${src0}"#VT.directive#" \t$dst, $src1, $src2, $src3;";
}
class Bfn_irir<string name, SDPatternOperator node, pisaVT opVT, pisaVT VT>
  : PISAInst4<name, VT, opVT, VT, VT, VT> {
  DAGOperand BfnOpcodeOpnd = PISABfnOpcodeOperand<opVT.VT, "Imm"#opVT.VT.Size#"Opnd">;
  let InOperandList = (ins BfnOpcodeOpnd:$src0, src1RC:$src1, VT.ImmOpnd:$src2, src3RC:$src3);
  let Pattern = [(set dstRC:$dst, (node timm:$src0, src1RC:$src1, (VT.VT imm:$src2), src3RC:$src3))];
  let AsmString = name#".${src0}"#VT.directive#" \t$dst, $src1, $src2, $src3;";
}
class Bfn_irii<string name, SDPatternOperator node, pisaVT opVT, pisaVT VT>
  : PISAInst4<name, VT, opVT, VT, VT, VT> {
  DAGOperand BfnOpcodeOpnd = PISABfnOpcodeOperand<opVT.VT, "Imm"#opVT.VT.Size#"Opnd">;
  let InOperandList = (ins BfnOpcodeOpnd:$src0, src1RC:$src1, VT.ImmOpnd:$src2, VT.ImmOpnd:$src3);
  let Pattern = [(set dstRC:$dst, (node timm:$src0, src1RC:$src1, (VT.VT imm:$src2), (VT.VT imm:$src3)))];
  let AsmString = name#".${src0}"#VT.directive#" \t$dst, $src1, $src2, $src3;";
}
class Bfn_iirr<string name, SDPatternOperator node, pisaVT opVT, pisaVT VT>
  : PISAInst4<name, VT, opVT, VT, VT, VT> {
  DAGOperand BfnOpcodeOpnd = PISABfnOpcodeOperand<opVT.VT, "Imm"#opVT.VT.Size#"Opnd">;
  let InOperandList = (ins BfnOpcodeOpnd:$src0, VT.ImmOpnd:$src1, src2RC:$src2, src3RC:$src3);
  let Pattern = [(set dstRC:$dst, (node timm:$src0, (VT.VT imm:$src1), src2RC:$src2, src3RC:$src3))];
  let AsmString = name#".${src0}"#VT.directive#" \t$dst, $src1, $src2, $src3;";
}
class Bfn_iiri<string name, SDPatternOperator node, pisaVT opVT, pisaVT VT>
  : PISAInst4<name, VT, opVT, VT, VT, VT> {
  DAGOperand BfnOpcodeOpnd = PISABfnOpcodeOperand<opVT.VT, "Imm"#opVT.VT.Size#"Opnd">;
  let InOperandList = (ins BfnOpcodeOpnd:$src0, VT.ImmOpnd:$src1, src2RC:$src2, VT.ImmOpnd:$src3);
  let Pattern = [(set dstRC:$dst, (node timm:$src0, (VT.VT imm:$src1), src2RC:$src2, (VT.VT imm:$src3)))];
  let AsmString = name#".${src0}"#VT.directive#" \t$dst, $src1, $src2, $src3;";
}
class Bfn_iiir<string name, SDPatternOperator node, pisaVT opVT, pisaVT VT>
  : PISAInst4<name, VT, opVT, VT, VT, VT> {
  DAGOperand BfnOpcodeOpnd = PISABfnOpcodeOperand<opVT.VT, "Imm"#opVT.VT.Size#"Opnd">;
  let InOperandList = (ins BfnOpcodeOpnd:$src0, VT.ImmOpnd:$src1, VT.ImmOpnd:$src2, src3RC:$src3);
  let Pattern = [(set dstRC:$dst, (node timm:$src0, (VT.VT imm:$src1), (VT.VT imm:$src2), src3RC:$src3))];
  let AsmString = name#".${src0}"#VT.directive#" \t$dst, $src1, $src2, $src3;";
}
class Bfn_iiii<string name, SDPatternOperator node, pisaVT opVT, pisaVT VT>
  : PISAInst4<name, VT, opVT, VT, VT, VT> {
  DAGOperand BfnOpcodeOpnd = PISABfnOpcodeOperand<opVT.VT, "Imm"#opVT.VT.Size#"Opnd">;
  let InOperandList = (ins BfnOpcodeOpnd:$src0, VT.ImmOpnd:$src1, VT.ImmOpnd:$src2, VT.ImmOpnd:$src3);
  let Pattern = [(set dstRC:$dst, (node timm:$src0, (VT.VT imm:$src1), (VT.VT imm:$src2), (VT.VT imm:$src3)))];
  let AsmString = name#".${src0}"#VT.directive#" \t$dst, $src1, $src2, $src3;";
}
multiclass BfnInst {
  defvar directive = "bfn";
  def _irrr: Bfn_irrr<directive, int_pisa_bfn, _i8, _i32>;
  def _irri: Bfn_irri<directive, int_pisa_bfn, _i8, _i32>;
  def _irir: Bfn_irir<directive, int_pisa_bfn, _i8, _i32>;
  def _irii: Bfn_irii<directive, int_pisa_bfn, _i8, _i32>;
  def _iirr: Bfn_iirr<directive, int_pisa_bfn, _i8, _i32>;
  def _iiri: Bfn_iiri<directive, int_pisa_bfn, _i8, _i32>;
  def _iiir: Bfn_iiir<directive, int_pisa_bfn, _i8, _i32>;
  def _iiii: Bfn_iiii<directive, int_pisa_bfn, _i8, _i32>;
}
defm bfn : BfnInst;

// => bfrev
defm bfrev : UnaryInst<"bfrev", bitreverse, types=[_i32]>;

// => cbit
defm cbit : UnaryInst<"cbit", ctpop, types=[_i16, _i32]>;

// fbh/fbl
multiclass FirstBitInst<SDPatternOperator node, list<pisaVT> types> {
  foreach VT = types in {
    defvar directive = VT.directive;
    defvar opcode = !subst(".", "_", directive);
    defvar uminOp = !cast<Instruction>(umin#opcode#"_ri");
    defvar bitSize = VT.VT.Size;
    def opcode#"_r" : PISAInst_r<NAME#directive, VT, VT>;
    def : Pat<(node VT.VT:$src0),
              (uminOp (!cast<Instruction>(NAME#opcode#"_r") VT.RC:$src0), bitSize)>;
    def opcode#"_i" : PISAInst_i<NAME#directive, VT, VT>;
    def : Pat<(node (VT.VT imm:$src0)),
              (uminOp (!cast<Instruction>(NAME#opcode#"_i") (VT.VT imm:$src0)), bitSize)>;
  }
}

// => fbh
defm fbh0 : UnaryInst<"fbh", ctlz_zero_poison, types=[_i16, _i32]>;
defm fbh  : FirstBitInst<ctlz, [_i16, _i32]>;

// => fbl
defm fbl0 : UnaryInst<"fbl", cttz_zero_poison, types=[_i16, _i32]>;
defm fbl  : FirstBitInst<cttz, [_i16, _i32]>;

// => not
defm not : UnaryInst<"not", not>;

// => or
defm or : BinaryInst<"or", or>;

// => sbfe
defm sbfe : TernaryInst<"sbfe", int_pisa_sbfe, types=[_i32]>;


// => ubfe
defm ubfe  : TernaryInst<"ubfe", int_pisa_ubfe, types=[_i32]>;

// => xor
defm xor   : BinaryInst<"xor", xor>;

// Temporarily support "bfe.32b" as an alias for "ubfe.32b" until all uses are replaced
def : InstAlias<TernaryInst_rrr<"bfe.32b", int_pisa_ubfe, _i32, _i32>.AsmString, (ubfe_32b_rrr Reg32b:$dst, Reg32b:$src0, Reg32b:$src1, Reg32b:$src2)>;
def : InstAlias<TernaryInst_rri<"bfe.32b", int_pisa_ubfe, _i32, _i32>.AsmString, (ubfe_32b_rri Reg32b:$dst, Reg32b:$src0, Reg32b:$src1, _i32.ImmOpnd:$src2)>;
def : InstAlias<TernaryInst_rir<"bfe.32b", int_pisa_ubfe, _i32, _i32>.AsmString, (ubfe_32b_rir Reg32b:$dst, Reg32b:$src0, _i32.ImmOpnd:$src1, Reg32b:$src2)>;
def : InstAlias<TernaryInst_rii<"bfe.32b", int_pisa_ubfe, _i32, _i32>.AsmString, (ubfe_32b_rii Reg32b:$dst, Reg32b:$src0, _i32.ImmOpnd:$src1, _i32.ImmOpnd:$src2)>;
def : InstAlias<TernaryInst_irr<"bfe.32b", int_pisa_ubfe, _i32, _i32>.AsmString, (ubfe_32b_irr Reg32b:$dst, _i32.ImmOpnd:$src0, Reg32b:$src1, Reg32b:$src2)>;
def : InstAlias<TernaryInst_iri<"bfe.32b", int_pisa_ubfe, _i32, _i32>.AsmString, (ubfe_32b_iri Reg32b:$dst, _i32.ImmOpnd:$src0, Reg32b:$src1, _i32.ImmOpnd:$src2)>;
def : InstAlias<TernaryInst_iir<"bfe.32b", int_pisa_ubfe, _i32, _i32>.AsmString, (ubfe_32b_iir Reg32b:$dst, _i32.ImmOpnd:$src0, _i32.ImmOpnd:$src1, Reg32b:$src2)>;
def : InstAlias<TernaryInst_iii<"bfe.32b", int_pisa_ubfe, _i32, _i32>.AsmString, (ubfe_32b_iii Reg32b:$dst, _i32.ImmOpnd:$src0, _i32.ImmOpnd:$src1, _i32.ImmOpnd:$src2)>;
// END: bitwise instruction definitions
//////

//////
// BEGIN: shift instruction definitions
// asr/shr/shl
multiclass ShiftInst<string name, SDNode node> {
  foreach VT = [_i16, _i32, _i64] in {
    defvar directive = VT.directive;
    defvar opcode = !subst(".", "_", directive);
    def opcode#"_rr": BinaryInst_rr<name#directive, node, VT, _i32>;
    def opcode#"_ri": BinaryInst_ri<name#directive, node, VT, _i32>;
    def opcode#"_ir": BinaryInst_ir<name#directive, node, VT, _i32, VT, false>;
    def opcode#"_ii": BinaryInst_ii<name#directive, node, VT, _i32>;
  }
}

// => asr
defm asr  : ShiftInst<"asr", sra>;

// => shr
defm shr  : ShiftInst<"shr", srl>;

// => shl
defm shl  : ShiftInst<"shl", shl>;

// => shf
defm shf_r : TernaryInst<"shf.r", fshr, types=[_i32]>;
defm shf_l : TernaryInst<"shf.l", fshl, types=[_i32]>;
// END: shift instruction definitions
//////

//////
// BEGIN: data movement and type conversion instructions definitions
// => addrcast
def : GINodeEquiv<G_ADDRSPACE_CAST, addrspacecast>;
class AddrcastToGeneric<string name, AddressSpace Out, AddressSpace In> : _PISAInst<name> {
  let AsmString = name#" \t$dst, $src;";
  let OutOperandList = (outs Out.RC:$dst);
  let InOperandList = (ins In.RC:$src);
  let Pattern = [(set (p0 Out.RC:$dst), (addrspacecast (In.PVT In.RC:$src)))];
}
class AddrcastToGenericFromFI<string name> : _PISAInst<name> {
  let AsmString = name#" \t$dst, $frameindex;";
  let OutOperandList = (outs generic.RC:$dst);
  let InOperandList = (ins FrameIndex:$frameindex);
}
class AddrcastToGenericFromGVar<string name> : _PISAInst<name> {
  let AsmString = name#" \t$dst, $gvar;";
  let OutOperandList = (outs generic.RC:$dst);
  let InOperandList = (ins PISAGlobalVariableOpnd:$gvar);
}
class AddrcastFromGeneric<string name, AddressSpace Out, AddressSpace In>
  : AddrcastToGeneric<name, Out, In> {
  let Pattern = [(set (Out.PVT Out.RC:$dst), (addrspacecast (p0 In.RC:$src)))];
}
class AddrcastToGenericCustomISel<string name, AddressSpace Out, AddressSpace In> : AddrcastToGeneric<name, Out, In> {
  let Pattern = [];
}
multiclass AddrcastInst<string name> {
  defvar Gen = generic;
  // addrspacecast.generic.[private|shared] %dst, @V
  foreach NonGen = [private, shared] in {
    def _gen_#NonGen#_FI:  AddrcastToGenericFromFI<name#"."#Gen#"."#NonGen>;
  }
  // addrspacecast.generic.[global|const] %dst, @G
  foreach NonGen = [global, const] in {
    def _gen_#NonGen#_var:  AddrcastToGenericFromGVar<name#"."#Gen#"."#NonGen>;
  }
  foreach NonGen = [private, shared, global] in {
    def _gen_#NonGen:  AddrcastToGeneric<name#"."#Gen#"."#NonGen, Gen, NonGen>;
    def _#NonGen#_gen: AddrcastFromGeneric<name#"."#NonGen#"."#Gen, NonGen, Gen>;
  }
  foreach NonGen = [const] in {
    def _gen_#NonGen:  AddrcastToGeneric<name#"."#Gen#"."#NonGen, Gen, NonGen>;
  }
}
defm addrcast : AddrcastInst<"addrcast">;

// => addrof
multiclass AddrOf {
    def _32b: PISAInst<(outs _i32.RegOpnd:$var), (ins FrameIndex:$frameindex), "addrof.32b $var, $frameindex;">;
    def _64b: PISAInst<(outs _i64.RegOpnd:$var), (ins PISAGlobalVariableOpnd:$varname), "addrof.64b $var, $varname;">;
    def _64b_func: PISAInst<(outs _i64.RegOpnd:$var), (ins FunctionOpnd:$funcname), "addrof.64b $var, $funcname;">;
}
defm addrof   : AddrOf<>; // MIR for G_FRAME_INDEX

// => extract
class ExtractInst_r<string name, pisaVT dstVT, pisaVT srcVT>
  : PISAInst_r<name, dstVT, srcVT> {
}
multiclass Extract {
  defvar VTs = [_v2i32, _v3i32, _v4i32, _v5i32, _v6i32, _v7i32, _v8i32, _v16i32, _v32i32, _v64i32];
  defvar lengths = [2, 3, 4, 5, 6, 7, 8, 16, 32, 64];
  // For extract, src must be a vector type.
  // For each src vector type...
  foreach i = !range(VTs) in {
    defvar srcVT = VTs[i];
    defvar srcLength = lengths[i];
    // The range of valid extract indices is [0, ..., length - 1] (i.e., for v4, [0, 1, 2, 3])...
    foreach index = !range(srcLength) in {
      // At each valid index, you can always extract a scalar.
      def "_"#index#"_i32_"#srcVT.name#"_r" : ExtractInst_r<"extract."#index#".32b", _i32, srcVT>;
      // For each valid destination vector type, check if it fits in remaining space
      foreach j = !range(VTs) in {
        defvar dstVT = VTs[j];
        defvar dstLength = lengths[j];
        // Destination vector is valid if dstLength <= (srcLength - index)
        if !le(dstLength, !sub(srcLength, index)) then {
          def "_"#index#"_"#dstVT.name#"_"#srcVT.name#"_r" : ExtractInst_r<"extract."#index#dstVT.directive, dstVT, srcVT>;
        }
      }
    }
  }
}
defset list<Instruction> ExtractInsts = {
  defm extract : Extract;
}
foreach EI = ExtractInsts in {
  def : Name2InstrEntry<!cast<string>(EI), EI>;
}

// => extract.dynamic
multiclass ExtractDynamic {
  defvar srcVTs = [_v2i32, _v3i32, _v4i32, _v5i32, _v6i32, _v7i32, _v8i32, _v16i32, _v32i32, _v64i32];
  foreach srcVT = srcVTs in {
    def "_"#srcVT.name : PISAInst_rr<"extract.dynamic"#srcVT.directive, _i32, srcVT, _i32>;
  }
}
defset list<Instruction> ExtractDynamicInsts = {
  defm extract_dynamic : ExtractDynamic;
}
foreach EDI = ExtractDynamicInsts in {
  def : Name2InstrEntry<!cast<string>(EDI), EDI>;
}

// => f2i
multiclass FP2IntInst<string name, SDPatternOperator node, bit isSigned> {
  foreach srcVT = [_bf16, _f16, _f32, _f64] in {
    foreach dstVT = [_i8, _i16, _i32, _i64] in {
      defvar toName = !if(isSigned,".s",".u")#dstVT.VT.Size;
      defvar directive = !interleave([toName, srcVT.directive], "");
      defvar opcode = !subst(".", "_", directive);
      def opcode#"_r" : UnaryInst_r<name#directive, node, dstVT, srcVT>;
      def opcode#"_i" : UnaryInst_i<name#directive, node, dstVT, srcVT>;
    }
  }
}
multiclass FP2IntSatInst<string name, SDPatternOperator node, bit isSigned> {
  foreach srcVT = [_bf16, _f16, _f32, _f64] in {
    foreach dstVT = [_i8, _i16, _i32, _i64] in {
      defvar toName = !if(isSigned,".s",".u")#dstVT.VT.Size;
      defvar directive = !interleave([toName, srcVT.directive], "");
      defvar opcode = !subst(".", "_", directive);
      def opcode#"_r" : UnaryInst_r<name#directive, node, dstVT, srcVT, (node dstVT.VT)>;
      def opcode#"_i" : UnaryInst_i<name#directive, node, dstVT, srcVT, (node dstVT.VT)>;
    }
  }
}
multiclass FP2IntRndInst<string name, SDPatternOperator node, bit isSigned> {
  foreach srcVT = [_bf16, _f16, _f32, _f64] in {
    foreach dstVT = [_i8, _i16, _i32, _i64] in {
      defvar toName = !if(isSigned,".s",".u")#dstVT.VT.Size;
      foreach rndNode = RoundingModeNodes in {
        defvar directive = !interleave([toName, srcVT.directive, rndNode.name], "");
        defvar opcode = !subst(".", "_", directive);
        def opcode#"_r": UnaryInst_r<name#directive, node, dstVT, srcVT, (node rndNode)>;
        def opcode#"_i": UnaryInst_i<name#directive, node, dstVT, srcVT, (node rndNode)>;
      }
    }
  }
}
defm f2i     : FP2IntInst<"f2i", fp_to_sint, isSigned=true>;
defm f2i_sat : FP2IntSatInst<"f2i", fp_to_sint_sat, isSigned=true>;
defm f2i     : FP2IntInst<"f2i", fp_to_uint, isSigned=false>;
defm f2i_sat : FP2IntSatInst<"f2i", fp_to_uint_sat, isSigned=false>;
defm f2i     : FP2IntRndInst<"f2i", int_pisa_fptosi_rnd, isSigned=true>;
defm f2i     : FP2IntRndInst<"f2i", int_pisa_fptoui_rnd, isSigned=false>;

// => fext
defm fext : ConvertSingleInst<"fext", fpextend, [_f32, _f64], _bf16>;
defm fext : ConvertSingleInst<"fext", fpextend, [_f32, _f64], _f16>;
defm fext : ConvertSingleInst<"fext", fpextend, [_f64], _f32>;

// => frnd
def pisa_nearestint : PatFrags<(ops node:$src0), [(frint node:$src0),
                                                  (fnearbyint node:$src0),
                                                  (froundeven node:$src0)]>;
defm frnd_re : UnaryFloatInst<"frnd.re", pisa_nearestint>;
defm frnd_rd : UnaryFloatInst<"frnd.rd", ffloor>;
defm frnd_ru : UnaryFloatInst<"frnd.ru", fceil>;
defm frnd_rz : UnaryFloatInst<"frnd.rz", ftrunc>;
defm frnd_rna: UnaryFloatInst<"frnd.rna", fround>;
defm frnd0   : UnaryFloatInstRnd<"frnd", int_pisa_frnd_rnd>;

// => ftrunc
defm ftrunc : ConvertMultiFloatInst<"ftrunc", fpround, _bf16, [_f32, _f64]>;
defm ftrunc : ConvertMultiFloatInst<"ftrunc", fpround, _f16, [_f32, _f64]>;
defm ftrunc : ConvertMultiFloatInst<"ftrunc", fpround, _f32, [_f64]>;
// .. llvm.fptrunc.round
defm ftrunc : ConvertMultiFloatInstRnd<"ftrunc", fptrunc_round, _bf16, [_f32, _f64]>;
defm ftrunc : ConvertMultiFloatInstRnd<"ftrunc", fptrunc_round, _f16, [_f32, _f64]>;
defm ftrunc : ConvertMultiFloatInstRnd<"ftrunc", fptrunc_round, _f32, [_f64]>;
defm ftrunc0 : ConvertMultiFloatInstRndSat<"ftrunc", int_pisa_ftrunc, _bf16, [_f32, _f64]>;
defm ftrunc0 : ConvertMultiFloatInstRndSat<"ftrunc", int_pisa_ftrunc, _f16, [_f32, _f64]>;
defm ftrunc0 : ConvertMultiFloatInstRndSat<"ftrunc", int_pisa_ftrunc, _f32, [_f64]>;


// => i2f
multiclass Int2FPInst<string name, SDPatternOperator node, bit isSigned> {
  foreach srcVT = [_i8, _i16, _i32, _i64] in {
    defvar fromName = !if(isSigned,".s",".u")#srcVT.VT.Size;
    foreach dstVT = [_bf16, _f16, _f32, _f64] in {
      defvar directive = !interleave([dstVT.directive, fromName], "");
      defvar opcode = !subst(".", "_", directive);
      def opcode#"_r" : UnaryInst_r<name#directive, node, dstVT, srcVT>;
      def opcode#"_i" : UnaryInst_i<name#directive, node, dstVT, srcVT>;
    }
  }
}
multiclass Int2FPInstRndSat<string name, SDPatternOperator node, bit isSigned> {
  foreach sat = [NoSatMod, DsatMod] in {
    foreach srcVT = [_i8, _i16, _i32, _i64]  in {
      defvar fromName = !if(isSigned,".s",".u")#srcVT.VT.Size;
      foreach dstVT = [_bf16, _f16, _f32, _f64] in {
        foreach rndNode = RoundingModeAllNodes in {
          defvar directive = !interleave([dstVT.directive, fromName, rndNode.name, sat.name], "");
          defvar opcode = !subst(".", "_", directive);
          def opcode#"_rii": UnaryInst_r<name#directive, node, dstVT, srcVT, (node rndNode, sat.node)>;
          def opcode#"_iii": UnaryInst_i<name#directive, node, dstVT, srcVT, (node rndNode, sat.node)>;
        }
      }
    }
  }
}
defm i2f : Int2FPInst<"i2f", sint_to_fp, true>;
defm i2f : Int2FPInst<"i2f", uint_to_fp, false>;
defm i2f: Int2FPInstRndSat<"i2f", int_pisa_sitofp, true>;
defm i2f: Int2FPInstRndSat<"i2f", int_pisa_uitofp, false>;

// => insert
class InsertInst_rr<string name, pisaVT dstVT, pisaVT srcVT>
  : PISAInst_rr<name, dstVT, dstVT, srcVT> {
  let AsmString = name#" \t$dst, $src1;";
}
multiclass Insert {
  defvar VTs = [_v2i32, _v3i32, _v4i32, _v5i32, _v6i32, _v7i32, _v8i32, _v16i32, _v32i32, _v64i32];
  defvar lengths = [2, 3, 4, 5, 6, 7, 8, 16, 32, 64];
  // For insert, dst must be a vector type.
  // For each dst vector type...
  foreach i = !range(VTs) in {
    defvar dstVT = VTs[i];
    defvar dstLength = lengths[i];
    // The range of valid insert indices is [0, ..., length - 1] (i.e., for v4, [0, 1, 2, 3])...
    foreach index = !range(dstLength) in {
      // At each valid index, you can always insert a scalar.
      let Constraints = "$src0 = $dst" in {
        def "_"#index#"_"#dstVT.name#"_i32_r" : InsertInst_rr<"insert."#index#".32b", dstVT, _i32>;
        // For each valid source vector type, check if it fits in remaining space
        foreach j = !range(VTs) in {
          defvar srcVT = VTs[j];
          defvar srcLength = lengths[j];
          // Source vector is valid if srcLength <= (dstLength - index)
          if !le(srcLength, !sub(dstLength, index)) then {
            def "_"#index#"_"#dstVT.name#"_"#srcVT.name#"_r" : InsertInst_rr<"insert."#index#srcVT.directive, dstVT, srcVT>;
          }
        }
      }
    }
  }
}
defset list<Instruction> InsertInsts = {
  defm insert : Insert;
}
foreach II = InsertInsts in {
  def : Name2InstrEntry<!cast<string>(II), II>;
}

// => insert.dynamic
multiclass InsertDynamic {
  let Constraints = "$src0 = $dst" in {
      defvar srcVTs = [_v2i32, _v3i32, _v4i32, _v5i32, _v6i32, _v7i32, _v8i32, _v16i32, _v32i32, _v64i32];
      foreach srcVT = srcVTs in {
        let AsmString = "insert.dynamic"#srcVT.directive#" \t$dst, $src1, $src2;" in {
          def "_"#srcVT.name : PISAInst_rrr<"insert.dynamic"#srcVT.directive, srcVT, srcVT, _i32, _i32>;
        }
      }
  }
}
defset list<Instruction> InsertDynamicInsts = {
  defm insert_dynamic : InsertDynamic;
}
foreach IDI = InsertDynamicInsts in {
  def : Name2InstrEntry<!cast<string>(IDI), IDI>;
}

// => isaddr
class IsaddrInst_r<string name, Intrinsic intrinsic, pisaVT dstVT, pisaVT srcVT>
  : UnaryInst_r<name, intrinsic, dstVT, srcVT>;
// FIXME: Need special selection as the input pointer may be from any address space.
multiclass IsaddrInst<string name> {
  def _private: IsaddrInst_r<name#".private", int_pisa_isaddr_private, _i32, _i64>;
  def _shared : IsaddrInst_r<name#".shared",  int_pisa_isaddr_shared,  _i32, _i64>;
  def _global : IsaddrInst_r<name#".global",  int_pisa_isaddr_global,  _i32, _i64>;
}
defm isaddr   : IsaddrInst<"isaddr">;


// => mov
class MovInst_i<string name, pisaVT dstVT, pisaVT srcVT> : PISAInst_i<name, dstVT, srcVT> {
  let Pattern = [(set dstRC:$dst, (srcVT.VT immVal:$src))];
  let isMoveImm = 1;
}
class MovInst_r<string name, pisaVT dstVT, pisaVT srcVT> : PISAInst_r<name, dstVT, srcVT> {
  let isMoveReg = 1;
}
multiclass VectorScalar<string name, pisaIntVT VT1, pisaVectorVT VT2> {
  defvar directive = VT1.directive;
  defvar opcode = !subst(".", "_", directive);
  def VT1.name#_#VT2.name#_r : MovInst_r<name#directive, VT1, VT2>;
  def VT2.name#_#VT1.name#_r : MovInst_r<name#directive, VT2, VT1>;
}
multiclass VectorVector<string name, string bitwidth, pisaVectorVT VT1, pisaVectorVT VT2> {
  def VT1.name#_#VT2.name#_r : MovInst_r<name#"."#bitwidth, VT1, VT2>;
  def VT2.name#_#VT1.name#_r : MovInst_r<name#"."#bitwidth, VT2, VT1>;
}
multiclass Mov {
  foreach VT = VTs.MovImm in {
    defvar directive = "mov."#VT.VT.Size#"b";
    def _#VT.name#_i: MovInst_i<directive, VT, VT>;
  }
  foreach VT = VTs.MovReg in {
    defvar directive = "mov."#VT.VT.Size#"b";
    def _#VT.name#_r : MovInst_r<directive, VT, VT>;
  }

  // 16bit
  defm _: VectorScalar<"mov", _i16, _v2i8>;

  // 32bit
  defm _: VectorScalar<"mov", _i32, _v4i8>;
  defm _: VectorScalar<"mov", _i32, _v2i16>;
  defm _: VectorVector<"mov", "32b", _v2i16, _v4i8>;

  // 64bit
  defm _: VectorScalar<"mov", _i64, _v4i16>;
  defm _: VectorScalar<"mov", _i64, _v2i32>;
  defm _: VectorVector<"mov", "64b", _v2i32, _v4i16>;

  // 128bit
  defm _: VectorScalar<"mov", _i128, _v4i32>;
  defm _: VectorScalar<"mov", _i128, _v2i64>;
  defm _: VectorVector<"mov", "128b", _v2i64, _v4i32>;
}
defm mov      : Mov;

// .. ptr G_CONSTANT imm
def : Pat<(_p0.VT imm:$imm), (mov_i64_i imm:$imm)>;
def : Pat<(_p1.VT imm:$imm), (mov_i64_i imm:$imm)>;
def : Pat<(_p2.VT imm:$imm), (mov_i64_i imm:$imm)>;
def : Pat<(_p3.VT imm:$imm), (mov_i32_i imm:$imm)>;
def : Pat<(_p4.VT imm:$imm), (mov_i32_i imm:$imm)>;


// => sext
defm sext : ConvertMultiInst<"sext", sext, _i64,  [_i8, _i16, _i32]>;
defm sext : ConvertMultiInst<"sext", sext, _i32,  [_i8, _i16]>;
defm sext : ConvertMultiInst<"sext", sext, _i16,  [_i8]>;

// => trunc
defm trunc : ConvertMultiInst<"trunc", trunc, _i8,  [_i16, _i32, _i64]>;
defm trunc : ConvertMultiInst<"trunc", trunc, _i16, [_i32, _i64]>;
defm trunc : ConvertMultiInst<"trunc", trunc, _i32, [_i64]>;

// => zext
defm zext : ConvertMultiInst<"zext", zanyext, _i64,  [_i8, _i16, _i32]>;
defm zext : ConvertMultiInst<"zext", zanyext, _i32,  [_i8, _i16]>;
defm zext : ConvertMultiInst<"zext", zanyext, _i16,  [_i8]>;
// END: data movement and type conversion instructions definitions
//////

//////
// BEGIN: comparison and selection instruction definitions
// ucmp/scmp
class CmpBinaryInstPred_rr<string name, CondCode setcnd, pisaVT srcVT>: PISAInst_rr<name, _i1, srcVT, srcVT> {
  let Pattern = [(set i1:$dst, (setcc srcVT.VT:$src0, srcVT.VT:$src1, (OtherVT setcnd)))];
}
class CmpBinaryInstPred_ri<string name, CondCode setcnd, pisaVT srcVT>: PISAInst_ri<name, _i1, srcVT, srcVT> {
  let Pattern = [(set i1:$dst, (setcc srcVT.VT:$src0, immVal1:$src1, setcnd))];
}
class CmpBinaryInstPred_ir<string name, CondCode setcnd, pisaVT srcVT>: PISAInst_ir<name, _i1, srcVT, srcVT> {
  let Pattern = [(set i1:$dst, (setcc (srcVT.VT immVal0:$src0), srcVT.VT:$src1, setcnd))];
}
class CmpBinaryInstPred_ii<string name, CondCode setcnd, pisaVT srcVT>: PISAInst_ii<name, _i1, srcVT, srcVT> {
  let Pattern = [(set i1:$dst, (setcc (srcVT.VT immVal0:$src0), immVal1:$src1, setcnd))];
}

def selextend : PatFrag<(ops node:$in), (select node:$in, -1, 0)>;
class CmpBinaryInstReg_rr<string name, CondCode setcnd, pisaVT srcVT>: PISAInst_rr<name, _i32, srcVT, srcVT> {
  let Pattern = [(set i32:$dst, (selextend (i1 (setcc srcVT.VT:$src0, srcVT.VT:$src1, setcnd))))];
}
class CmpBinaryInstReg_ri<string name, CondCode setcnd, pisaVT srcVT>: PISAInst_ri<name, _i32, srcVT, srcVT> {
  let Pattern = [(set i32:$dst, (selextend (i1 (setcc srcVT.VT:$src0, immVal1:$src1, setcnd))))];
}
class CmpBinaryInstReg_ir<string name, CondCode setcnd, pisaVT srcVT>: PISAInst_ir<name, _i32, srcVT, srcVT> {
  let Pattern = [(set i32:$dst, (selextend (i1 (setcc (srcVT.VT immVal0:$src0), srcVT.VT:$src1, setcnd))))];
}
class CmpBinaryInstReg_ii<string name, CondCode setcnd, pisaVT srcVT>: PISAInst_ii<name, _i32, srcVT, srcVT> {
  let Pattern = [(set i32:$dst, (selextend (i1 (setcc (srcVT.VT immVal0:$src0), immVal1:$src1, setcnd))))];
}
multiclass CmpInst<string name, list<pisaVT> types, bit isSigned> {
  defvar prefix = !if(isSigned, "", "u");
  foreach cc = ["ne", "eq", "gt", "ge", "lt", "le"] in {
    defvar ccName = !toupper(!if(!or(!eq(cc, "ne"), !eq(cc, "eq")), cc, prefix#cc));
    defvar cond = !cast<CondCode>("SET"#ccName);
    foreach VT = types in {
      defvar directive = "."#cc#VT.directive;
      defvar opcode = !subst(".", "_", directive);

      // Defs for cmp opcodes where dst is a predicate
      // cmp is directly matched, e.g.
      //   %2 = icmp slt i32 %0, %1
      def opcode#"_prr" : CmpBinaryInstPred_rr<name#directive, cond, VT>;
      def opcode#"_pri" : CmpBinaryInstPred_ri<name#directive, cond, VT>;
      def opcode#"_pir" : CmpBinaryInstPred_ir<name#directive, cond, VT>;
      def opcode#"_pii" : CmpBinaryInstPred_ii<name#directive, cond, VT>;

      // Defs for cmp opcodes where dst is a 32b register
      // cmp+sext is matched, e.g.
      //  %2 = icmp slt i32 %0, %1
      //  %3 = sext i1 %2 to i32
      def opcode#"_rrr" : CmpBinaryInstReg_rr<name#directive, cond, VT>;
      def opcode#"_rri" : CmpBinaryInstReg_ri<name#directive, cond, VT>;
      def opcode#"_rir" : CmpBinaryInstReg_ir<name#directive, cond, VT>;
      def opcode#"_rii" : CmpBinaryInstReg_ii<name#directive, cond, VT>;
    }
  }
}

// => ucmp
defm ucmp : CmpInst<"ucmp", [_i16, _i32, _i64], 0>;

// => scmp
defm scmp : CmpInst<"scmp", [_i16, _i32, _i64], 1>;

// => fcmp
multiclass FCmpInst<string name, list<pisaVT> types> {
  foreach cc = ["oge", "ogt", "ole", "olt", "oeq", "une"] in {
    defvar cond = !substr(cc, 1);
    defvar node = !cast<CondCode>("SET"#!toupper(cc));
    foreach ftz = [omitFTZMod, usesFTZMod] in {
      foreach VT = types in {
        defvar directive = !interleave(["."#cond, ftz.name, VT.directive], "");
        defvar opcode = !subst(".", "_", directive);

        // Defs for fcmp opcodes where dst is a predicate
        // fcmp is directly matched, e.g.
        //   %2 = fcmp olt f32 %0, %1
        def opcode#"_prr" : CmpBinaryInstPred_rr<name#directive, node, VT>, Requires<[ftz.pred]>;
        def opcode#"_pri" : CmpBinaryInstPred_ri<name#directive, node, VT>, Requires<[ftz.pred]>;
        def opcode#"_pir" : CmpBinaryInstPred_ir<name#directive, node, VT>, Requires<[ftz.pred]>;
        def opcode#"_pii" : CmpBinaryInstPred_ii<name#directive, node, VT>, Requires<[ftz.pred]>;

        // Defs for fcmp opcodes where dst is a 32b register
        // fcmp+sext is matched, e.g.
        //  %2 = fcmp olt i32 %0, %1
        //  %3 = sext i1 %2 to i32
        def opcode#"_rrr" : CmpBinaryInstReg_rr<name#directive, node, VT>, Requires<[ftz.pred]>;
        def opcode#"_rri" : CmpBinaryInstReg_ri<name#directive, node, VT>, Requires<[ftz.pred]>;
        def opcode#"_rir" : CmpBinaryInstReg_ir<name#directive, node, VT>, Requires<[ftz.pred]>;
        def opcode#"_rii" : CmpBinaryInstReg_ii<name#directive, node, VT>, Requires<[ftz.pred]>;
      }
    }
  }
}
defm fcmp : FCmpInst<"fcmp", [_bf16, _f16, _f32, _f64]>;


// => sel
class SelInst_rrp<string name, pisaVT VT> : PISAInst_rr<name, VT, VT, VT> {
  DAGOperand PredRC = _i1.RegOpnd;
  let AsmString = name#" \t$dst, $src0, $src1, $p;";
  let InOperandList = (ins PredRC:$p, src0RC:$src0, src1RC:$src1);
  let Pattern = [(set VT.VT:$dst, (select i1:$p, VT.VT:$src0, src1RC:$src1))];
}
class SelInst_rip<string name, pisaVT VT> : PISAInst_ri<name, VT, VT, VT> {
  DAGOperand PredRC = _i1.RegOpnd;
  let AsmString = name#" \t$dst, $src0, $src1, $p;";
  let InOperandList = (ins PredRC:$p, src0RC:$src0, immOpd1:$src1);
  let Pattern = [(set VT.VT:$dst, (select i1:$p, VT.VT:$src0, immVal1:$src1))];
}
class SelInst_irp<string name, pisaVT VT> : PISAInst_ir<name, VT, VT, VT> {
  DAGOperand PredRC = _i1.RegOpnd;
  let AsmString = name#" \t$dst, $src0, $src1, $p;";
  let InOperandList = (ins PredRC:$p, immOpd0:$src0, src1RC:$src1);
  let Pattern = [(set VT.VT:$dst, (select i1:$p, immVal0:$src0, VT.VT:$src1))];
}
class SelInst_iip<string name, pisaVT VT> : PISAInst_ii<name, VT, VT, VT> {
  DAGOperand PredRC = _i1.RegOpnd;
  let AsmString = name#" \t$dst, $src0, $src1, $p;";
  let InOperandList = (ins PredRC:$p, immOpd0:$src0, immOpd1:$src1);
  let Pattern = [(set VT.VT:$dst, (select i1:$p, immVal0:$src0, immVal1:$src1))];
}
class SelInst_rrr<string name, pisaVT VT, pisaVT predVT> : PISAInst_rrr<name, VT, VT, VT, predVT> {
  let Pattern = [(set VT.VT:$dst, (pisa_select predVT.VT:$src2, VT.VT:$src0, VT.VT:$src1))];
}
class SelInst_rir<string name, pisaVT VT, pisaVT predVT> : PISAInst_rir<name, VT, VT, VT, predVT> {
  let Pattern = [(set VT.VT:$dst, (pisa_select predVT.VT:$src2, VT.VT:$src0, immVal1:$src1))];
}
class SelInst_irr<string name, pisaVT VT, pisaVT predVT> : PISAInst_irr<name, VT, VT, VT, predVT> {
  let Pattern = [(set VT.VT:$dst, (pisa_select predVT.VT:$src2, immVal0:$src0, VT.VT:$src1))];
}
class SelInst_iir<string name, pisaVT VT, pisaVT predVT> : PISAInst_iir<name, VT, VT, VT, predVT> {
  let Pattern = [(set VT.VT:$dst, (pisa_select predVT.VT:$src2, immVal0:$src0, immVal1:$src1))];
}
multiclass SelInst<string name> {
  foreach VT = [_i16, _i32, _i64, _bf16, _f16, _f32, _f64] in {
    defvar directive = "."#VT.bitwidth;
    defvar opcode = VT.opcode;

    // select on predicate
    def opcode#"_rrp" : SelInst_rrp<name#directive, VT>;
    def opcode#"_irp" : SelInst_irp<name#directive, VT>;
    def opcode#"_rip" : SelInst_rip<name#directive, VT>;
    def opcode#"_iip" : SelInst_iip<name#directive, VT>;
    // select on register
    defvar predVT = !if(!eq(VT.bitwidth,"16b"), _i16, !if(!eq(VT.bitwidth,"32b"), _i32, _i64));
    def opcode#"_rrr"#predVT.bitwidth : SelInst_rrr<name#directive, VT, predVT>;
    def opcode#"_irr"#predVT.bitwidth : SelInst_irr<name#directive, VT, predVT>;
    def opcode#"_rir"#predVT.bitwidth : SelInst_rir<name#directive, VT, predVT>;
    def opcode#"_iir"#predVT.bitwidth : SelInst_iir<name#directive, VT, predVT>;
  }
}
class SelInst_rrnp<string name, pisaVT VT> : PISAInst_rr<name, VT, VT, VT> {
  let AsmString = name#" \t$dst, $src0, $src1, $negate$p;";
  let InOperandList = (ins Negate:$negate, _i1.RegOpnd:$p, src0RC:$src0, src1RC:$src1);
}
class SelInst_rinp<string name, pisaVT VT> : PISAInst_ri<name, VT, VT, VT> {
  let AsmString = name#" \t$dst, $src0, $src1, $negate$p;";
  let InOperandList = (ins Negate:$negate, _i1.RegOpnd:$p, src0RC:$src0, immOpd1:$src1);
}
class SelInst_irnp<string name, pisaVT VT> : PISAInst_ir<name, VT, VT, VT> {
  let AsmString = name#" \t$dst, $src0, $src1, $negate$p;";
  let InOperandList = (ins Negate:$negate, _i1.RegOpnd:$p, immOpd0:$src0, src1RC:$src1);
}
class SelInst_iinp<string name, pisaVT VT> : PISAInst_ii<name, VT, VT, VT> {
  let AsmString = name#" \t$dst, $src0, $src1, $negate$p;";
  let InOperandList = (ins Negate:$negate, _i1.RegOpnd:$p, immOpd0:$src0, immOpd1:$src1);
}
multiclass InvSelInst<string name> {
  foreach VT = [_i16, _i32, _i64, _bf16, _f16, _f32, _f64] in {
    defvar directive = VT.directive;
    defvar opcode = !subst(".", "_", directive);

    // select on negative predicate
    def opcode#"_rrnp" : SelInst_rrnp<name#directive, VT>;
    def opcode#"_irnp" : SelInst_irnp<name#directive, VT>;
    def opcode#"_rinp" : SelInst_rinp<name#directive, VT>;
    def opcode#"_iinp" : SelInst_iinp<name#directive, VT>;
  }
}
defm sel : SelInst<"sel">;
defm sel : InvSelInst<"sel">;

// emit patterns to match the pointer variants for sel and cmp
// e.g.,
// %2:registers(s1) = G_ICMP intpred(eq), %0(p2), %1
// %3:reg64b(p2) = G_SELECT %2(s1), %0, %1

foreach VT = VTs.Ptr in {
  defvar selop = !cast<Instruction>("sel_"#VT.VT.Size#"_rrp");
  def : Pat<(select i1:$p, VT.VT:$src0, VT.VT:$src1),
            (selop Pred:$p, VT.RC:$src0, VT.RC:$src1)>;
  foreach isSigned = [0, 1] in {
    defvar opcode = !if(isSigned, "scmp", "ucmp");
    defvar prefix = !if(isSigned, "", "u");
    foreach cc = ["ne", "eq", "gt", "ge", "lt", "le"] in {
      defvar ccName = !toupper(!if(!or(!eq(cc, "ne"), !eq(cc, "eq")), cc, prefix#cc));
      defvar cond = !cast<CondCode>("SET"#ccName);
      defvar cmpop = !cast<Instruction>(opcode#"_"#cc#VT.opcode#"b_prr");
      def : Pat<(setcc VT.VT:$src0, VT.VT:$src1, cond),
                (cmpop VT.RC:$src0, VT.RC:$src1)>;
    }
  }
}
// END: comparison and selection instruction definitions
//////

//////
// BEGIN: control flow instruction definitions
// => call
let isCall = 1 in {
  // Define function call instructions with a FunctionCallTargetOpnd:
  def functionCall_void: PISAInst<(outs),
            (ins FunctionCallTargetOpnd:$func, variable_ops),
            "call void, $func">;
  foreach VT = VTs.CallReturn in {
    def "functionCall_"#VT.name#"_r": PISAInst<(outs VT.RegOpnd:$ret),
            (ins FunctionCallTargetOpnd:$func, variable_ops),
            "call $ret, $func">;
  }
  // Define function call instructions with a register call target:
  def "indirectFunctionCall_void_r": PISAInst<(outs),
            (ins IndirectFunctionCallTargetOpnd:$func, variable_ops),
            "call void, $func">;
  foreach VT = VTs.CallReturn in {
    def "indirectFunctionCall_"#VT.name#"_r_i64_r": PISAInst<(outs VT.RegOpnd:$ret),
            (ins IndirectFunctionCallTargetOpnd:$func, variable_ops),
            "call $ret, $func">;
  }
}

// => call.cond
let isCall=1, isTerminator=1 in {
  // Define function call instructions with a FunctionCallTargetOpnd:
  def predCall_void: PISAInst<(outs),
            (ins Negate:$negate, _i1.RegOpnd:$p, FunctionCallTargetOpnd:$func, variable_ops),
            "call.cond $negate$p, void, $func">;
  foreach VT = VTs.CallReturn in {
    def "predCall_"#VT.name#"_r": PISAInst<(outs VT.RegOpnd:$ret),
            (ins Negate:$negate, _i1.RegOpnd:$p, FunctionCallTargetOpnd:$func, variable_ops),
            "call.cond $negate$p, $ret, $func">;
  }
  // Define function call instructions with a register call target:
  def "indirectPredCall_void_r": PISAInst<(outs),
            (ins Negate:$negate, _i1.RegOpnd:$p, IndirectFunctionCallTargetOpnd:$func, variable_ops),
            "call.cond $negate$p, void, $func">;
  foreach VT = VTs.CallReturn in {
    def "indirectPredCall_"#VT.name#"_r_i64_r": PISAInst<(outs VT.RegOpnd:$ret),
            (ins Negate:$negate, _i1.RegOpnd:$p, IndirectFunctionCallTargetOpnd:$func, variable_ops),
            "call.cond $negate$p, $ret, $func">;
  }
}


// => goto
def BrTargetOpnd : PISASymbolOperand<OtherVT, "BrTargetOpnd">;
def : GINodeEquiv<G_BRCOND, brcond>;

let isTerminator=1, isBranch=1 in {
  def gotolabel : PISAInst<(outs), (ins BrTargetOpnd:$label),
                           "goto $label;",
                           [(br bb:$label)]> {
    let isBarrier = 1;
    let Defs = [SpecialReg_ActiveMask];
  }
  // FIXME: Need refactoring and adding a complex predicate operand, which is
  // composed of a negate operand and a predicate register.
  def predgoto : PISAInst<(outs), (ins Negate:$negate, _i1.RegOpnd:$cond, BrTargetOpnd:$label),
                          "goto.cond $negate$cond, $label;",
                          [(brcond i1:$cond, bb:$label)]> {
    let Defs = [SpecialReg_ActiveMask];
  }
}

// => return
class SimpleOp<string name, list<dag> pattern = []> : PISAInst<(outs), (ins), name, pattern>;
multiclass RetVal<string name> {
  foreach VT = VTs.CallReturn in {
    def "_"#VT.name#"_r":  PISAInst<(outs), (ins VT.RegOpnd:$value),  name#" $value;">;
  }
  // support return of immediates
  foreach VT = VTs.Integer in {
    def "_"#VT.name#"_i":  PISAInst<(outs), (ins VT.ImmOpnd:$value),  name#" $value;">;
  }
  foreach VT = VTs.Float in {
    def "_"#VT.name#"_i":  PISAInst<(outs), (ins VT.ImmOpnd:$value),  name#" $value;">;
  }
}
let isReturn = 1, isBarrier = 1, isTerminator = 1, isNotDuplicable = 1 in {
  def ret: SimpleOp<"return;">;
  defm retValue: RetVal<"return">;
}

// => return.cond
multiclass PredRet<string name> {
  foreach VT = VTs.CallReturn in {
    def "_"#VT.name#"_r":  PISAInst<(outs), (ins Negate:$negate, _i1.RegOpnd:$cond, VT.RegOpnd:$value),  name#" $negate$cond, $value;">;
  }
}
let isReturn = 1, isBarrier = 1, isTerminator=1, isBranch=1 in {
  defm predRet: PredRet<"return.cond">;
}
// END: control flow instruction definitions
//////

//////
// BEGIN: subgroup communication instruction definitions
// ired/fred
class Red_rrr<string name, SDPatternOperator node, TImmLeaf opNode, pisaVT src0VT, pisaVT src1VT>
  : PISAInst_rrr<name, src0VT, src0VT, src1VT, src0VT> {
  let Pattern = [(set (src0VT.VT dstRC:$dst), (node opNode, (src0VT.VT src0RC:$src0), (src1VT.VT src1RC:$src1), (src0VT.VT src2RC:$src2)))];
  let AsmString = name#" \t$dst, $src0, $src1;";
  let Constraints = "$src2 = $dst";
}
class Red_rir<string name, SDPatternOperator node, TImmLeaf opNode, pisaVT src0VT, pisaVT src1VT>
  : PISAInst_rir<name, src0VT, src0VT, src1VT, src0VT> {
  let Pattern = [(set (src0VT.VT dstRC:$dst), (node opNode, (src0VT.VT src0RC:$src0), (src1VT.VT imm:$src1), (src0VT.VT src2RC:$src2)))];
  let AsmString = name#" \t$dst, $src0, $src1;";
  let Constraints = "$src2 = $dst";
}
class Red_rir_bf<string name, SDPatternOperator node, TImmLeaf opNode, pisaVT src0VT, pisaVT src1VT>
  : PISAInst_rir<name, src0VT, src0VT, src1VT, src0VT> {
  let Pattern = [(set (src0VT.VT dstRC:$dst), (node opNode, (src0VT.VT src0RC:$src0), (src1VT.VT imm:$src1), (src0VT.VT src2RC:$src2)))];
  let AsmString = name#" \t$dst, $src0, $src1;";
  let Constraints = "$src2 = $dst";
}
class Red_rir_bf_nanp<string name, SDPatternOperator node, TImmLeaf opNode, TImmLeaf nanpNode, pisaVT src0VT, pisaVT src1VT>
  : PISAInst_rir<name, src0VT, src0VT, src1VT, src0VT> {
  let Pattern = [(set (src0VT.VT dstRC:$dst), (node opNode, (src0VT.VT src0RC:$src0), (src1VT.VT imm:$src1), (src0VT.VT src2RC:$src2), nanpNode))];
  let AsmString = name#" \t$dst, $src0, $src1;";
  let Constraints = "$src2 = $dst";
}
// fred 5-operand variants - match int_pisa_fred with nanp ImmArg.
class Red_rrr_nanp<string name, SDPatternOperator node, TImmLeaf opNode, TImmLeaf nanpNode, pisaVT src0VT, pisaVT src1VT>
  : PISAInst_rrr<name, src0VT, src0VT, src1VT, src0VT> {
  let Pattern = [(set (src0VT.VT dstRC:$dst), (node opNode, (src0VT.VT src0RC:$src0), (src1VT.VT src1RC:$src1), (src0VT.VT src2RC:$src2), nanpNode))];
  let AsmString = name#" \t$dst, $src0, $src1;";
  let Constraints = "$src2 = $dst";
}
class Red_rir_nanp<string name, SDPatternOperator node, TImmLeaf opNode, TImmLeaf nanpNode, pisaVT src0VT, pisaVT src1VT>
  : PISAInst_rir<name, src0VT, src0VT, src1VT, src0VT> {
  let Pattern = [(set (src0VT.VT dstRC:$dst), (node opNode, (src0VT.VT src0RC:$src0), (src1VT.VT imm:$src1), (src0VT.VT src2RC:$src2), nanpNode))];
  let AsmString = name#" \t$dst, $src0, $src1;";
  let Constraints = "$src2 = $dst";
}

// => fred
multiclass FRedInst<string name> {
  foreach opNode = FRedOpNodes in {
    // scalar bf16/f16/f32: loop over AllNanpMods to emit nanp=0 and nanp=1 variants.
    // Directive: fred.<op>[.nanp].<type>
    foreach nanp = AllNanpMods in {
      foreach type = [_bf16, _f16, _f32] in {
        defvar directive = !interleave([opNode.name, nanp.name, type.directive], "");
        defvar opcode = !subst(".", "_", directive);
        def opcode#"_rrr": Red_rrr_nanp<name#directive, int_pisa_fred, opNode, nanp.node, type, _i32>;
        def opcode#"_rir": Red_rir_nanp<name#directive, int_pisa_fred, opNode, nanp.node, type, _i32>;
      }
    }
  }
}
let isConvergent = 1 in {
  defm fred: FRedInst<"fred">;
}

// => ired
multiclass IRedInst<string name> {
  foreach opNode = IRedOpNodes in {
    // defvar red = IRedOpMode<opNode>.name;
    foreach type = [_i16, _i32] in {
      defvar directive = !interleave([opNode.name, type.directive], "");
      defvar opcode = !subst(".", "_", directive);
      def opcode#"_rrr": Red_rrr<name#directive, int_pisa_ired, opNode, type, _i32>;
      def opcode#"_rir": Red_rir<name#directive, int_pisa_ired, opNode, type, _i32>;
    }
  }
}
let isConvergent = 1 in {
  defm ired: IRedInst<"ired">;
}

// => redfirstidx
let isConvergent = 1 in {
  defm redfirstidx: UnaryInst<"redfirstidx", int_pisa_redfirstidx, types=[_i32]>;
}

// => shfl
class Shuffle_rrrr<string name, TImmLeaf modeNode, TImmLeaf sgNode, pisaVT VT>
  : PISAInst4<name, VT, VT, VT, VT, VT> {
  let InOperandList = (ins src0RC:$src0, src1RC:$src1, src2RC:$src2, src3RC:$src3);
  let Pattern = [(set dstRC:$dst, (int_pisa_shfl modeNode, src0RC:$src0, src1RC:$src1, src2RC:$src2, src3RC:$src3, sgNode))];
  let AsmString = name#" \t$dst, $src0, $src1, $src2;";
  let Constraints = "$src3 = $dst";
}
class Shuffle_rrir<string name, TImmLeaf modeNode, TImmLeaf sgNode, pisaVT VT>
  : PISAInst4<name, VT, VT, VT, VT, VT> {
  let InOperandList = (ins src0RC:$src0, src1RC:$src1, VT.ImmOpnd:$src2, src3RC:$src3);
  let Pattern = [(set dstRC:$dst, (int_pisa_shfl modeNode, src0RC:$src0, src1RC:$src1, (VT.VT imm:$src2), src2RC:$src3, sgNode))];
  let AsmString = name#" \t$dst, $src0, $src1, $src2;";
  let Constraints = "$src3 = $dst";
}
class Shuffle_rirr<string name, TImmLeaf modeNode, TImmLeaf sgNode, pisaVT VT>
  : PISAInst4<name, VT, VT, VT, VT, VT> {
  let InOperandList = (ins src0RC:$src0, VT.ImmOpnd:$src1, src2RC:$src2, src3RC:$src3);
  let Pattern = [(set dstRC:$dst, (int_pisa_shfl modeNode, src0RC:$src0, (VT.VT imm:$src1), src2RC:$src2, src3RC:$src3, sgNode))];
  let AsmString = name#" \t$dst, $src0, $src1, $src2;";
  let Constraints = "$src3 = $dst";
}
class Shuffle_riir<string name, TImmLeaf modeNode, TImmLeaf sgNode, pisaVT VT>
  : PISAInst4<name, VT, VT, VT, VT, VT> {
  let InOperandList = (ins src0RC:$src0, VT.ImmOpnd:$src1, VT.ImmOpnd:$src2, src3RC:$src3);
  let Pattern = [(set dstRC:$dst, (int_pisa_shfl modeNode, src0RC:$src0, (VT.VT imm:$src1), (VT.VT imm:$src2), src3RC:$src3, sgNode))];
  let AsmString = name#" \t$dst, $src0, $src1, $src2;";
  let Constraints = "$src3 = $dst";
}
multiclass ShflInst<string name> {
  foreach modeNode = SHFLModeNodes in {
    foreach sgNode = [timm_false, timm_true] in {
      defvar sg = !if(!eq(!cast<string>(sgNode), !cast<string>(timm_true)), ".sg", "");
      defvar directive = !interleave([modeNode.name, sg, ".32b"], "");
      defvar opcode = !subst(".", "_", directive);
      def opcode#"_rrrr" : Shuffle_rrrr<name#directive, modeNode, sgNode, _i32>;
      def opcode#"_rrir" : Shuffle_rrir<name#directive, modeNode, sgNode, _i32>;
      def opcode#"_rirr" : Shuffle_rirr<name#directive, modeNode, sgNode, _i32>;
      def opcode#"_riir" : Shuffle_riir<name#directive, modeNode, sgNode, _i32>;
    }
  }
}
let isConvergent = 1 in {
  defm shfl: ShflInst<"shfl">;
}
// END: subgroup communication instruction definitions
//////

//////
// BEGIN: memory instruction definitions

// => fatom
defm fatomic_load_add  : binary_atomic_op_fp<atomic_load_fadd>;
defm fatomic_load_sub  : binary_atomic_op_fp<atomic_load_fsub>;
defm fatomic_load_min  : binary_atomic_op_fp<atomic_load_fmin>;
defm fatomic_load_max  : binary_atomic_op_fp<atomic_load_fmax>;

// 2 source operands
multiclass BinaryFloatAtomicInst<string Op> {
  foreach AS = [global, shared, generic] in {
    foreach VT = [_bf16, _f16, _f32, _f64] in {
      // fatom.[shared|generic]* are not allowed for f64
      defvar invalid0 = !and(!or(!eq(AS, shared), !eq(AS, generic)), !eq(VT, _f64));
      // fatom.*.[min|max] are not allowed for f64
      defvar invalid1 = !and(!or(!eq(Op, "min"), !eq(Op, "max")), !eq(VT, _f64));
      defvar invalidOperations = !or(invalid0, invalid1);
      // GlobalISel only checks memory size, not type, when matching MemoryVT.
      // bf16 and f16 are both 16-bit, so we need a GISelPredicateCode to
      // disambiguate them by checking the register type of the result operand.
      defvar needsMemVTCheck = !or(!eq(VT, _bf16), !eq(VT, _f16));
      defvar memVTCheck = !if(!eq(VT, _bf16),
        "MRI.getType(MI.getOperand(0).getReg()).isBFloat16()",
        "MRI.getType(MI.getOperand(0).getReg()).isFloat16()");
      if !not(invalidOperations) then {
        foreach MO = ["monotonic", "release", "acquire", "acq_rel"] in {
          defvar fullName = !strconcat("fatomic_load_", Op, "_"#VT.VT, "_", MO);
          defvar node = !cast<PatFrag>(fullName);
          defvar MOStr = AtomicMemOrder<MO>.name;

          defvar opcode = "_"#VT.name#"_"#AS.name#"_"#MO#"_"#Op;
          defvar directive = "fatom."#AS.name#"."#MOStr;
          if needsMemVTCheck then {
            let Pattern = [(set VT.VT:$dst,
                (BinaryAtomAddrspaceMemVTPat<AS, node, memVTCheck>
                    PISAMem_ri<AS>.Pattern:$addr, VT.VT:$src0))] in {
              def opcode#"_ri" : PISABinaryAtom_ri<directive, Op, AS, VT, node>;
            }
            let Pattern = [(set VT.VT:$dst,
                (BinaryAtomAddrspaceMemVTPat<AS, node, memVTCheck>
                    PISAMem_rr<AS>.Pattern:$addr, VT.VT:$src0))] in {
              def opcode#"_rr" : PISABinaryAtom_rr<directive, Op, AS, VT, node>;
            }
          } else {
            def opcode#"_ri" : PISABinaryAtom_ri<directive, Op, AS, VT, node>;
            def opcode#"_rr" : PISABinaryAtom_rr<directive, Op, AS, VT, node>;
          }
          if !eq(MO, "monotonic") then {
            // .memorder is optional
            defvar opcodeA = "_"#VT.name#"_"#AS.name#"_"#Op;
            defvar directiveA = "fatom."#AS.name;
            if needsMemVTCheck then {
              let Pattern = [(set VT.VT:$dst,
                  (BinaryAtomAddrspaceMemVTPat<AS, node, memVTCheck>
                      PISAMem_ri<AS>.Pattern:$addr, VT.VT:$src0))] in {
                def opcodeA#"_ri" : PISABinaryAtom_ri<directiveA, Op, AS, VT, node>;
              }
              let Pattern = [(set VT.VT:$dst,
                  (BinaryAtomAddrspaceMemVTPat<AS, node, memVTCheck>
                      PISAMem_rr<AS>.Pattern:$addr, VT.VT:$src0))] in {
                def opcodeA#"_rr" : PISABinaryAtom_rr<directiveA, Op, AS, VT, node>;
              }
            } else {
              def opcodeA#"_ri" : PISABinaryAtom_ri<directiveA, Op, AS, VT, node>;
              def opcodeA#"_rr" : PISABinaryAtom_rr<directiveA, Op, AS, VT, node>;
            }
          }
        }
      }
    }
  }
}

defm atomic_load_fadd_impl : BinaryFloatAtomicInst<"add">;
defm atomic_load_fsub_impl : BinaryFloatAtomicInst<"sub">;
defm atomic_load_fmin_impl : BinaryFloatAtomicInst<"min">;
defm atomic_load_fmax_impl : BinaryFloatAtomicInst<"max">;

// fatom.*.cas.* intrinsic
class FAtomCasAddrspacePat<AddressSpace Addrspace, SDPatternOperator pat> :
      PatFrag<(ops node:$ptr, node:$opnd1, node:$opnd2, node:$opnd3), (pat node:$ptr, node:$opnd1, node:$opnd2, node:$opnd3)> {
  let AddressSpaces = [ Addrspace.num ];
  let IsLoad = true;
  let IsStore = true;
}
class FAtomCasInst_rri<string directive, string op, AddressSpace AS, pisaVT VT, SDPatternOperator node, SDPatternOperator moNode>
  : PISATernaryAtom<directive, op, AS, VT, PISAMem_rr<AS>, node> {
  let Pattern = [(set VT.VT:$dst, (FAtomCasAddrspacePat<AS, node> ADDR_rr:$addr, VT.VT:$src0, VT.VT:$src1, moNode))];
}
class FAtomCasInst_rii<string directive, string op, AddressSpace AS, pisaVT VT, SDPatternOperator node, SDPatternOperator moNode>
  : PISATernaryAtom<directive, op, AS, VT, PISAMem_ri<AS>, node> {
  let Pattern = [(set VT.VT:$dst, (FAtomCasAddrspacePat<AS, node> ADDR_ri:$addr, VT.VT:$src0, VT.VT:$src1, moNode))];
}
multiclass FAtomCasInst<string name> {
  defvar Op = "cas";
  foreach AS = [global, shared, generic] in {
    foreach VT = [_bf16, _f16, _f32] in {
      foreach moNode = MemoryOrderNodes in {
        defvar directive = !interleave([AS.directive, moNode.name], "");
        defvar opcode = !subst(".", "_", !interleave([VT.opcode, AS.directive, moNode.name, "_", Op], ""));
        def opcode#"_rri" : FAtomCasInst_rri<name#directive, Op, AS, VT, int_pisa_cas_fatom, moNode>;
        def opcode#"_rii" : FAtomCasInst_rii<name#directive, Op, AS, VT, int_pisa_cas_fatom, moNode>;
        if !eq(moNode.name, ".relaxed") then {
          // .memorder is optional
          defvar directiveA = AS.directive;
          defvar opcodeA = !subst(".", "_", !interleave([VT.opcode, AS.directive, "_", Op], ""));
          def opcodeA#"_rri" : FAtomCasInst_rri<name#directiveA, Op, AS, VT, int_pisa_cas_fatom, moNode>;
          def opcodeA#"_rii" : FAtomCasInst_rii<name#directiveA, Op, AS, VT, int_pisa_cas_fatom, moNode>;
        }
      }
    }
  }
}
defm fatom : FAtomCasInst<"fatom">;

// => iatom
multiclass load_atomic_op_ord {
  def NAME#_monotonic : PatFrag<(ops node:$ptr), (!cast<SDPatternOperator>(NAME) node:$ptr)> {
    let IsAtomic = true;
    let IsAtomicOrderingMonotonic = true;
  }
  def NAME#_acquire : PatFrag<(ops node:$ptr), (!cast<SDPatternOperator>(NAME) node:$ptr)> {
    let IsAtomic = true;
    let IsAtomicOrderingAcquire = true;
  }
  def NAME#_release : PatFrag<(ops node:$ptr), (!cast<SDPatternOperator>(NAME) node:$ptr)> {
    let IsAtomic = true;
    let IsAtomicOrderingRelease = true;
  }
  def NAME#_acq_rel : PatFrag<(ops node:$ptr), (!cast<SDPatternOperator>(NAME) node:$ptr)> {
    let IsAtomic = true;
    let IsAtomicOrderingAcquireRelease = true;
  }
  def NAME#_seq_cst : PatFrag<(ops node:$ptr), (!cast<SDPatternOperator>(NAME) node:$ptr)> {
    let IsAtomic = true;
    let IsAtomicOrderingSequentiallyConsistent = true;
  }
}
class IAtomOpName<string Op> {
  string name = !cond(!eq(Op, "min"): "smin",
                      !eq(Op, "max"): "smax",
                      !eq(Op, "swap"): "xchg",
                      true           : Op);
}

def atomic_swap_i128 : PatFrag<(ops node:$ptr, node:$val), (atomic_swap  node:$ptr, node:$val)> {
    let IsAtomic = true;
    let MemoryVT = i128;
  }
defm atomic_swap_i128 : binary_atomic_op_ord;
def atomic_cmp_swap_i128 : PatFrag<(ops node:$ptr, node:$cmp, node:$val),
                       (atomic_cmp_swap node:$ptr, node:$cmp, node:$val)> {
    let IsAtomic = true;
    let MemoryVT = i128;
  }
defm atomic_cmp_swap_i128 : ternary_atomic_op_ord;

// Define atomic inc/dec as atomic add/sub with imm 1
multiclass atomic_load_incdec<SDNode atomic_op> {
  foreach vt = [i16, i32, i64] in {
    def _#vt : PatFrag<(ops node:$ptr), (atomic_op node:$ptr, 1)> {
      let IsAtomic = true;
      let MemoryVT = vt;
    }
    defm NAME#_#vt : load_atomic_op_ord;
  }
}
defm atomic_load_inc : atomic_load_incdec<atomic_load_add>;
defm atomic_load_dec : atomic_load_incdec<atomic_load_sub>;

// 1 source operand
multiclass UnaryAtomicInst<string Op> {
  foreach AS = [global, shared, generic] in {
    defvar VTs = !if(!eq(AS, global), [_i16, _i32, _i64], [_i16, _i32]);
    foreach VT = VTs in {
      foreach MO = ["monotonic", "release", "acquire", "acq_rel"] in {
        defvar fullName = NAME#"_"#VT.VT#"_"#MO;
        defvar node = !cast<PatFrag>(fullName);
        defvar MOStr = AtomicMemOrder<MO>.name;
        defvar OpStr = IAtomOpName<Op>.name;

        defvar opcode = "_"#VT.name#"_"#AS.name#"_"#MO#"_"#Op;
        defvar directive = "iatom."#AS.name#"."#MOStr;
        def opcode#"_ri" : PISAUnaryAtom_ri<directive, OpStr, AS, VT, node>;
        def opcode#"_rr" : PISAUnaryAtom_rr<directive, OpStr, AS, VT, node>;
        if !eq(MO, "monotonic") then {
          // .memorder is optional
          defvar opcodeA = "_"#VT.name#"_"#AS.name#"_"#Op;
          defvar directiveA = "iatom."#AS.name;
          def opcodeA#"_ri" : PISAUnaryAtom_ri<directiveA, OpStr, AS, VT, node>;
          def opcodeA#"_rr" : PISAUnaryAtom_rr<directiveA, OpStr, AS, VT, node>;
        }
      }
    }
  }
}

// 2 source operands
defm atomic_load_incwrap : binary_atomic_op<atomic_load_uinc_wrap>;
defm atomic_load_decwrap : binary_atomic_op<atomic_load_udec_wrap>;

multiclass BinaryAtomicInst<string Op> {
  foreach AS = [global, shared, generic] in {
    foreach VT = [_i16, _i32, _i64, _i128] in {
      // i128 is only allowed for "swap" op.
      defvar invalid0 = !and(!eq(VT, _i128), !ne(Op, "swap"));
      // iatom.[shared|generic].<non-cas-op> is only allowed for i16, i32
      defvar invalid1 = !and(!or(!eq(AS, shared), !eq(AS, generic)), !eq(VT, _i64));
      // iatom*.[inc|dec]wrap is only allowed for i32, i64
      defvar invalid2 = !and(!or(!eq(Op, "incwrap"), !eq(Op, "decwrap")), !eq(VT, _i16));
      defvar invalidOperations = !or(invalid0, invalid1, invalid2);
      if !not(invalidOperations) then {
        foreach MO = ["monotonic", "release", "acquire", "acq_rel"] in {
          defvar opPrefix = !if(!eq(Op,"swap"), "atomic_", "atomic_load_");
          defvar fullName = !strconcat(opPrefix , Op,      "_"#VT.VT , "_", MO);
          defvar node = !cast<PatFrag>(fullName);
          defvar MOStr = AtomicMemOrder<MO>.name;
          defvar OpStr = IAtomOpName<Op>.name;

          defvar opcode = "_"#VT.name#"_"#AS.name#"_"#MO#"_"#Op;
          defvar directive = "iatom."#AS.name#"."#MOStr;
          def opcode#"_ri" : PISABinaryAtom_ri<directive, OpStr, AS, VT, node>;
          def opcode#"_rr" : PISABinaryAtom_rr<directive, OpStr, AS, VT, node>;
          if !eq(MO, "monotonic") then {
            // .memorder is optional
            defvar opcodeA = "_"#VT.name#"_"#AS.name#"_"#Op;
            defvar directiveA = "iatom."#AS.name;
            def opcodeA#"_ri" : PISABinaryAtom_ri<directiveA, OpStr, AS, VT, node>;
            def opcodeA#"_rr" : PISABinaryAtom_rr<directiveA, OpStr, AS, VT, node>;
          }
        }
      }
    }
  }
}

// 3 source operands
multiclass TernaryAtomicInst<string Op> {
  foreach AS = [global, shared, generic] in {
    foreach VT = [_i16, _i32, _i64, _i128] in {
      foreach MO = ["monotonic", "release", "acquire", "acq_rel"] in {
        defvar fullName = !strconcat("atomic_", Op, "_"#VT.VT , "_", MO);
        defvar node = !cast<SDPatternOperator>(fullName);
        defvar MOStr = AtomicMemOrder<MO>.name;

        defvar opcode = "_"#VT.name#"_"#AS.name#"_"#MO#"_"#Op;
        defvar directive = "iatom."#AS.name#"."#MOStr;
        def opcode#"_ri" : PISATernaryAtom_ri<directive, "cas", AS, VT, node>;
        def opcode#"_rr" : PISATernaryAtom_rr<directive, "cas", AS, VT, node>;
        if !eq(MO, "monotonic") then {
          // .memorder is optional
          defvar opcodeA = "_"#VT.name#"_"#AS.name#"_"#Op;
          defvar directiveA = "iatom."#AS.name;
          def opcodeA#"_ri" : PISATernaryAtom_ri<directiveA, "cas", AS, VT, node>;
          def opcodeA#"_rr" : PISATernaryAtom_rr<directiveA, "cas", AS, VT, node>;
        }
      }
    }
  }
}

defm atomic_load_inc : UnaryAtomicInst<"inc">;
defm atomic_load_dec : UnaryAtomicInst<"dec">;

defm atomic_load_incwrap_impl : BinaryAtomicInst<"incwrap">;
defm atomic_load_decwrap_impl : BinaryAtomicInst<"decwrap">;
defm atomic_load_add_impl : BinaryAtomicInst<"add">;
defm atomic_load_sub_impl : BinaryAtomicInst<"sub">;
defm atomic_load_and_impl : BinaryAtomicInst<"and">;
defm atomic_load_or_impl : BinaryAtomicInst<"or">;
defm atomic_load_xor_impl : BinaryAtomicInst<"xor">;
defm atomic_load_min_impl : BinaryAtomicInst<"min">;
defm atomic_load_max_impl : BinaryAtomicInst<"max">;
defm atomic_load_umin_impl : BinaryAtomicInst<"umin">;
defm atomic_load_umax_impl : BinaryAtomicInst<"umax">;
defm atomic_swap_impl : BinaryAtomicInst<"swap">;

// TODO: llvm's cmpxchg instruction takes 2 memory orderings. Why?
defm atomic_cmp_swap   : TernaryAtomicInst<"cmp_swap">;

// => ld
multiclass load_atomic_op<SDNode atomic_op> {
  foreach vt = VTs.AtomicLoadStore in {
    def _#vt : PatFrag<(ops node:$ptr), (atomic_op node:$ptr)> {
      let IsAtomic = true;
      let MemoryVT = vt.VT;
    }
    defm NAME#_#vt  : load_atomic_op_ord;
  }
}
defm load_atomic : load_atomic_op<atomic_load>;
// atomic loads
multiclass AtomicLoadInst<string name> {
  foreach pVT = VTs.AtomicLoadStore in {
    foreach MO = ["monotonic", "acquire", "seq_cst"] in {
      defvar MOStr = AtomicMemOrder<MO>.name;
      defvar fullName = "load_atomic_"#pVT#"_"#MO;
      defvar node = !cast<PatFrag>(fullName);
      foreach AS = [global, generic, shared] in {
        defvar directive = name#"."#AS.name#"."#MOStr;
        def "_"#AS#"_"#MO#"_"#pVT.name#"_rr" : PISAAtomicLoad_rr<directive, AS, pVT, node>;
        def "_"#AS#"_"#MO#"_"#pVT.name#"_ri" : PISAAtomicLoad_ri<directive, AS, pVT, node>;
      }
      // .addrspace defaults to .generic when not specified
      defvar directive = name#"."#MOStr;
      def "_generic_"#MO#"_"#pVT.name#"_alias_rr" : PISAAtomicLoad_rr<directive, generic, pVT, node>;
      def "_generic_"#MO#"_"#pVT.name#"_alias_ri" : PISAAtomicLoad_ri<directive, generic, pVT, node>;
    }
  }
}
// non-atomic loads
multiclass LoadInst<string name> {
  foreach pVT = VTs.LoadStore in {
    foreach AS = [global, const, generic, private, shared] in {
      defvar directive = name#"."#AS;
      def "_"#AS#"_"#pVT.name#"_ri" : PISALoad_ri<directive, AS, pVT, load>;
      def "_"#AS#"_"#pVT.name#"_rr" : PISALoad_rr<directive, AS, pVT, load>;
      // 'ld.weak' is an alias for 'ld'
      defvar directiveA = name#"."#AS#".weak";
      def "_"#AS#"_weak_"#pVT.name#"_ri" : PISALoad_ri<directiveA, AS, pVT, load>;
      def "_"#AS#"_weak_"#pVT.name#"_rr" : PISALoad_rr<directiveA, AS, pVT, load>;
    }
    // .addrspace defaults to .generic when not specified
    def "_generic_"#pVT.name#"_alias_ri" : PISALoad_ri<name, generic, pVT, load>;
    def "_generic_"#pVT.name#"_alias_rr" : PISALoad_rr<name, generic, pVT, load>;
    // 'ld.weak' is an alias for 'ld'
    def "_generic_weak_"#pVT.name#"_alias_ri" : PISALoad_ri<name#".weak", generic, pVT, load>;
    def "_generic_weak_"#pVT.name#"_alias_rr" : PISALoad_rr<name#".weak", generic, pVT, load>;
  }
}
// kernel param loads (ld.param)
multiclass LoadParamInst {
  foreach VT = VTs.LoadParam in {
    def _#VT.name: PISAInst<(outs VT.RegOpnd:$res),
                            (ins MEM_ii:$addr, variable_ops),
                            "ld.param"#VT.directive#" $res, $addr;">;
    def : Pat<(VT.VT (load MEM_ii:$addr)), (!cast<Instruction>("loadParam_"#VT.name) MEM_ii:$addr)>;
    def _#VT.name#"_ir": PISAInst<(outs VT.RegOpnd:$res),
                            (ins MEM_ir:$addr, variable_ops),
                            "ld.param"#VT.directive#" $res, $addr;">;
    def : Pat<(VT.VT (load MEM_ir:$addr)), (!cast<Instruction>("loadParam_"#VT.name#"_ir") MEM_ir:$addr)>;
  }
}
defm loadParam: LoadParamInst;
defm ld : AtomicLoadInst<"ld">;
defm ld : LoadInst<"ld">;


// => st
multiclass store_atomic_op_ord {
  def NAME#_monotonic : PatFrag<(ops node:$val, node:$ptr), (!cast<SDPatternOperator>(NAME) node:$val, node:$ptr)> {
    let IsAtomic = true;
    let IsAtomicOrderingMonotonic = true;
  }
  def NAME#_acquire : PatFrag<(ops node:$val, node:$ptr), (!cast<SDPatternOperator>(NAME) node:$val, node:$ptr)> {
    let IsAtomic = true;
    let IsAtomicOrderingAcquire = true;
  }
  def NAME#_release : PatFrag<(ops node:$val, node:$ptr), (!cast<SDPatternOperator>(NAME) node:$val, node:$ptr)> {
    let IsAtomic = true;
    let IsAtomicOrderingRelease = true;
  }
  def NAME#_acq_rel : PatFrag<(ops node:$val, node:$ptr), (!cast<SDPatternOperator>(NAME) node:$val, node:$ptr)> {
    let IsAtomic = true;
    let IsAtomicOrderingAcquireRelease = true;
  }
  def NAME#_seq_cst : PatFrag<(ops node:$val, node:$ptr), (!cast<SDPatternOperator>(NAME) node:$val, node:$ptr)> {
    let IsAtomic = true;
    let IsAtomicOrderingSequentiallyConsistent = true;
  }
}
multiclass store_atomic_op<SDNode atomic_op> {
  foreach vt = VTs.AtomicLoadStore in {
    def _#vt : PatFrag<(ops node:$val, node:$ptr), (atomic_op node:$val, node:$ptr)> {
      let IsAtomic = true;
      let MemoryVT = vt.VT;
    }
    defm NAME#_#vt : store_atomic_op_ord;
  }
}
defm store_atomic : store_atomic_op<atomic_store>;
// atomic stores
multiclass AtomicStoreInst<string name> {
  foreach pVT = VTs.AtomicLoadStore in {
    foreach MO = ["monotonic", "release", "seq_cst"] in {
      defvar MOStr = AtomicMemOrder<MO>.name;
      defvar fullName = "store_atomic_"#pVT#"_"#MO;
      defvar node = !cast<PatFrag>(fullName);
      foreach AS = [global, generic, shared] in {
        defvar directive = name#"."#AS.name#"."#MOStr;
        def "_"#AS#"_"#MO#"_"#pVT.name#"_rr" : PISAAtomicStore_rr<directive, AS, pVT, node>;
        def "_"#AS#"_"#MO#"_"#pVT.name#"_ri" : PISAAtomicStore_ri<directive, AS, pVT, node>;
      }
      // .addrspace defaults to .generic when not specified
      defvar directive = name#"."#MOStr;
      def "_generic_"#MO#"_"#pVT.name#"_alias_rr" : PISAAtomicStore_rr<directive, generic, pVT, node>;
      def "_generic_"#MO#"_"#pVT.name#"_alias_ri" : PISAAtomicStore_ri<directive, generic, pVT, node>;
    }
  }
}
// non-atomic stores
multiclass StoreInst<string name> {
  foreach pVT = VTs.LoadStore in {
    foreach AS = [global, generic, private, shared] in {
      defvar directive = name#"."#AS;
      def "_"#AS#"_"#pVT.name#"_ri" : PISAStore_ri<directive, AS, pVT, store>;
      def "_"#AS#"_"#pVT.name#"_rr" : PISAStore_rr<directive, AS, pVT, store>;
      // 'st.weak' is an alias for 'st'
      defvar directiveA = name#"."#AS#".weak";
      def "_"#AS#"_weak_"#pVT.name#"_ri" : PISAStore_ri<directiveA, AS, pVT, store>;
      def "_"#AS#"_weak_"#pVT.name#"_rr" : PISAStore_rr<directiveA, AS, pVT, store>;
    }
    // .addrspace defaults to .generic when not specified
    def "_generic_"#pVT.name#"_alias_ri" : PISAStore_ri<name, generic, pVT, store>;
    def "_generic_"#pVT.name#"_alias_rr" : PISAStore_rr<name, generic, pVT, store>;
    // 'st.weak' is an alias for 'st'
    def "_generic_weak_"#pVT.name#"_alias_ri" : PISAStore_ri<name#".weak", generic, pVT, store>;
    def "_generic_weak_"#pVT.name#"_alias_rr" : PISAStore_rr<name#".weak", generic, pVT, store>;
  }
}
defm st : AtomicStoreInst<"st">;
defm st : StoreInst<"st">;

// END: memory instruction definitions
//////

//////
// BEGIN: synchronization instruction definitions

// => barrier.workgroup
let isConvergent = 1, hasSideEffects = 1 in {
  def barrier_wg : PISAInst<(outs), (ins), "barrier.workgroup;", [(int_pisa_workgroup_barrier)]>;
}

// => fence
multiclass FenceInst<string name> {
  foreach as = [generic, global, shared, default] in {
    foreach scope = [
          workgroup,
          gpu,
          system] in {
      // system scope is only supported for global fence
      if !or(!ne(scope, system), !eq(as, global)) then {
        defvar asName =  !if(!empty(as.name), "", "."#as.name);
        defvar opcode = !if(!empty(as.name), "_"#scope, "_"#as.name#"_"#scope);
        defvar asmstr = name#asName#"${order}."#scope#";";
        def opcode: PISAInst<(outs), (ins FenceMemOrderOpnd:$order), asmstr>;
      }
    }
  }
  defvar directive = ".subgroup";
  defvar opcode = !subst(".", "_", directive);
  def opcode: PISAInst<(outs), (ins), name#directive#";">;
}
defm fence : FenceInst<"fence">;

// END: synchronization instruction definitions
//////



// => lifetime.start
let hasSideEffects = 1, mayLoad = 0, mayStore = 0, isReMaterializable = 0 in {
  multiclass LifetimeStart {
    foreach VT = VTs.LifetimeTypes in {
      def "_"#VT.name#"_r" : PISAInst<(outs VT.RegOpnd:$dst), (ins),
                                      "lifetime.start \t$dst;", []>;
    }
  }
  defm lifetime_start : LifetimeStart;
}
// Register-class to lifetime.start opcode.
foreach VT = VTs.LifetimeTypes in {
  def : LifetimeStartEntry<!cast<string>(VT.RC),
                           !cast<Instruction>("lifetime_start_"#VT.name#"_r")>;
}

