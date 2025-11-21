// RUN: llvm-tblgen -gen-instr-info -I %p/../../include %s -o - | FileCheck -check-prefix=INSTRINFO %s
// RUN: llvm-tblgen -gen-asm-matcher -I %p/../../include %s -o - | FileCheck -check-prefix=ASMMATCHER %s
// RUN: llvm-tblgen -gen-disassembler -I %p/../../include %s -o - | FileCheck -check-prefix=DISASM %s
// RUN: llvm-tblgen -gen-dag-isel -I %p/../../include %s -o - | FileCheck -check-prefix=ISEL-SDAG %s
// RUN: llvm-tblgen -gen-global-isel -I %p/../../include %s -o - | FileCheck -check-prefix=ISEL-GISEL %s

include "llvm/Target/Target.td"

// INSTRINFO:      #ifdef GET_INSTRINFO_ENUM
// INSTRINFO-NEXT: #undef GET_INSTRINFO_ENUM
// INSTRINFO-EMPTY:
// INSTRINFO-NEXT: namespace llvm::MyTarget {
// INSTRINFO-EMPTY:
// INSTRINFO-NEXT: enum {
// INSTRINFO-NEXT:   PHI
// INSTRINFO:      };
// INSTRINFO:      enum RegClassByHwModeUses : uint16_t {
// INSTRINFO-NEXT:   MyPtrRC,
// INSTRINFO-NEXT:   XRegs_EvenIfRequired,
// INSTRINFO-NEXT:   YRegs_EvenIfRequired,
// INSTRINFO-NEXT: };
// INSTRINFO-EMPTY:
// INSTRINFO-NEXT: } // namespace llvm::MyTarget

// INSTRINFO: { MyTarget::XRegsRegClassID, 0, MCOI::OPERAND_REGISTER, 0 },
// INSTRINFO: { MyTarget::XRegs_EvenRegClassID, 0, MCOI::OPERAND_REGISTER, 0 },
// INSTRINFO: { MyTarget::YRegs_EvenIfRequired, 0|(1<<MCOI::LookupRegClassByHwMode), MCOI::OPERAND_REGISTER, 0 },
// INSTRINFO: { MyTarget::XRegs_EvenIfRequired, 0|(1<<MCOI::LookupRegClassByHwMode), MCOI::OPERAND_REGISTER, 0 },
// INSTRINFO: { MyTarget::XRegs_EvenIfRequired, 0|(1<<MCOI::LookupRegClassByHwMode), MCOI::OPERAND_REGISTER, 0 }, { MyTarget::MyPtrRC, 0|(1<<MCOI::LookupRegClassByHwMode), MCOI::OPERAND_REGISTER, 0 },
// INSTRINFO: { MyTarget::YRegs_EvenIfRequired, 0|(1<<MCOI::LookupRegClassByHwMode), MCOI::OPERAND_REGISTER, 0 }, { MyTarget::XRegs_EvenIfRequired, 0|(1<<MCOI::LookupRegClassByHwMode), MCOI::OPERAND_REGISTER, 0 },

// INSTRINFO: extern const int16_t MyTargetRegClassByHwModeTables[4][3] = {
// INSTRINFO-NEXT:  { // DefaultMode
// INSTRINFO-NEXT:    MyTarget::PtrRegs32RegClassID,
// INSTRINFO-NEXT:    MyTarget::XRegsRegClassID,
// INSTRINFO-NEXT:    MyTarget::YRegsRegClassID,
// INSTRINFO-NEXT:  },
// INSTRINFO-NEXT:  { // EvenMode
// INSTRINFO-NEXT:    -1, // Missing mode entry
// INSTRINFO-NEXT:    MyTarget::XRegs_EvenRegClassID,
// INSTRINFO-NEXT:    MyTarget::YRegs_EvenRegClassID,
// INSTRINFO-NEXT:  },
// INSTRINFO-NEXT:  { // OddMode
// INSTRINFO-NEXT:    -1, // Missing mode entry
// INSTRINFO-NEXT:    MyTarget::XRegs_OddRegClassID,
// INSTRINFO-NEXT:    -1, // Missing mode entry
// INSTRINFO-NEXT:  },
// INSTRINFO-NEXT:  { // Ptr64
// INSTRINFO-NEXT:    MyTarget::PtrRegs64RegClassID,
// INSTRINFO-NEXT:    -1, // Missing mode entry
// INSTRINFO-NEXT:    -1, // Missing mode entry
// INSTRINFO-NEXT:  },
// INSTRINFO-NEXT: };

// INSTRINFO: static inline void InitMyTargetMCInstrInfo(
// INSTRINFO-NEXT: II->InitMCInstrInfo(MyTargetDescs.Insts, MyTargetInstrNameIndices, MyTargetInstrNameData, nullptr, nullptr, {{[0-9]+}}, &MyTargetRegClassByHwModeTables[0][0], 3);



// ASMMATCHER: enum MatchClassKind {
// ASMMATCHER:   MCK_LAST_TOKEN = OptionalMatchClass,
// ASMMATCHER-NEXT:   MCK_XRegs_Odd, // register class 'XRegs_Odd'
// ASMMATCHER-NEXT:   MCK_PtrRegs32, // register class 'PtrRegs32'
// ASMMATCHER-NEXT:   MCK_PtrRegs64, // register class 'PtrRegs64'
// ASMMATCHER-NEXT:   MCK_XRegs_Even, // register class 'XRegs_Even'
// ASMMATCHER-NEXT:   MCK_YRegs_Even, // register class 'YRegs_Even'
// ASMMATCHER-NEXT:   MCK_XRegs, // register class 'XRegs'
// ASMMATCHER-NEXT:   MCK_YRegs, // register class 'YRegs'
// ASMMATCHER-NEXT:   MCK_LAST_REGISTER = MCK_YRegs,
// ASMMATCHER-NEXT:   MCK_RegByHwMode_MyPtrRC, // register class by hwmode
// ASMMATCHER-NEXT:   MCK_RegByHwMode_XRegs_EvenIfRequired, // register class by hwmode
// ASMMATCHER-NEXT:   MCK_RegByHwMode_YRegs_EvenIfRequired, // register class by hwmode
// ASMMATCHER-NEXT:   MCK_LAST_REGCLASS_BY_HWMODE = MCK_RegByHwMode_YRegs_EvenIfRequired,
// ASMMATCHER-NEXT:   MCK_Imm, // user defined class 'ImmAsmOperand'

// ASMMATCHER: static unsigned validateOperandClass(MCParsedAsmOperand &GOp, MatchClassKind Kind, const MCSubtargetInfo &STI) {
// ASMMATCHER: if (Operand.isToken() && Kind <= MCK_LAST_TOKEN)

// ASMMATCHER: switch (Kind) {

// ASMMATCHER: if (Operand.isReg() && Kind > MCK_LAST_REGISTER && Kind <= MCK_LAST_REGCLASS_BY_HWMODE) {
// ASMMATCHER-NEXT:    static constexpr MatchClassKind RegClassByHwModeMatchTable[4][3] = {
// ASMMATCHER-NEXT:      { // DefaultMode
// ASMMATCHER-NEXT:        MCK_PtrRegs32,
// ASMMATCHER-NEXT:        MCK_XRegs,
// ASMMATCHER-NEXT:        MCK_YRegs,
// ASMMATCHER-NEXT:      },
// ASMMATCHER-NEXT:      { // EvenMode
// ASMMATCHER-NEXT:        InvalidMatchClass, // Missing mode
// ASMMATCHER-NEXT:        MCK_XRegs_Even,
// ASMMATCHER-NEXT:        MCK_YRegs_Even,
// ASMMATCHER-NEXT:      },
// ASMMATCHER-NEXT:      { // OddMode
// ASMMATCHER-NEXT:        InvalidMatchClass, // Missing mode
// ASMMATCHER-NEXT:        MCK_XRegs_Odd,
// ASMMATCHER-NEXT:        InvalidMatchClass, // Missing mode
// ASMMATCHER-NEXT:      },
// ASMMATCHER-NEXT:      { // Ptr64
// ASMMATCHER-NEXT:        MCK_PtrRegs64,
// ASMMATCHER-NEXT:        InvalidMatchClass, // Missing mode
// ASMMATCHER-NEXT:        InvalidMatchClass, // Missing mode
// ASMMATCHER-NEXT:      },
// ASMMATCHER-NEXT:    };
// ASMMATCHER-EMPTY:
// ASMMATCHER-NEXT:    static_assert(MCK_LAST_REGCLASS_BY_HWMODE - MCK_LAST_REGISTER == 3);
// ASMMATCHER-NEXT:    const unsigned HwMode = STI.getHwMode(MCSubtargetInfo::HwMode_RegInfo);
// ASMMATCHER-NEXT:    Kind = RegClassByHwModeMatchTable[HwMode][Kind - (MCK_LAST_REGISTER + 1)];
// ASMMATCHER-NEXT:  }

// ASMMATCHER: if (Operand.isReg()) {

// ASMMATCHER: static const MatchEntry MatchTable0[] = {
// ASMMATCHER: /* also_my_load_1 */, MyTarget::MY_LOAD, Convert__RegByHwMode_XRegs_EvenIfRequired1_0__RegByHwMode_MyPtrRC1_1, AMFBS_None, { MCK_RegByHwMode_XRegs_EvenIfRequired, MCK_RegByHwMode_MyPtrRC }, },
// ASMMATCHER: /* also_my_load_2 */, MyTarget::MY_LOAD, Convert__RegByHwMode_XRegs_EvenIfRequired1_0__RegByHwMode_MyPtrRC1_1, AMFBS_None, { MCK_RegByHwMode_XRegs_EvenIfRequired, MCK_RegByHwMode_MyPtrRC }, },
// ASMMATCHER: /* always_all */, MyTarget::ALWAYS_ALL, Convert__Reg1_0, AMFBS_None, { MCK_XRegs }, },
// ASMMATCHER: /* always_even */, MyTarget::ALWAYS_EVEN, Convert__Reg1_0, AMFBS_None, { MCK_XRegs_Even }, },
// ASMMATCHER: /* custom_decode */, MyTarget::CUSTOM_DECODE, Convert__RegByHwMode_YRegs_EvenIfRequired1_0, AMFBS_None, { MCK_RegByHwMode_YRegs_EvenIfRequired }, },
// ASMMATCHER: /* even_if_mode */, MyTarget::EVEN_IF_MODE, Convert__RegByHwMode_XRegs_EvenIfRequired1_0, AMFBS_None, { MCK_RegByHwMode_XRegs_EvenIfRequired }, },
// ASMMATCHER: /* my_mov */, MyTarget::MY_MOV, Convert__RegByHwMode_YRegs_EvenIfRequired1_0__RegByHwMode_XRegs_EvenIfRequired1_1, AMFBS_None, { MCK_RegByHwMode_YRegs_EvenIfRequired, MCK_RegByHwMode_XRegs_EvenIfRequired }, },



// DISASM{LITERAL}: [[maybe_unused]]
// DISASM-NEXT: static DecodeStatus DecodeMyPtrRCRegClassByHwMode(MCInst &Inst, unsigned Imm, uint64_t Addr, const MCDisassembler *Decoder) {
// DISASM-NEXT:   switch (Decoder->getSubtargetInfo().getHwMode(MCSubtargetInfo::HwMode_RegInfo)) {
// DISASM-NEXT:   case 0: // DefaultMode
// DISASM-NEXT:     return DecodePtrRegs32RegisterClass(Inst, Imm, Addr, Decoder);
// DISASM-NEXT:   case 3: // Ptr64
// DISASM-NEXT:     return DecodePtrRegs64RegisterClass(Inst, Imm, Addr, Decoder);
// DISASM-NEXT:   default:
// DISASM-NEXT:     llvm_unreachable("no decoder for hwmode");
// DISASM-NEXT:   }
// DISASM-NEXT: }

// DISASM{LITERAL}: [[maybe_unused]]
// DISASM-NEXT: static DecodeStatus DecodeXRegs_EvenIfRequiredRegClassByHwMode(MCInst &Inst, unsigned Imm, uint64_t Addr, const MCDisassembler *Decoder) {
// DISASM-NEXT:  switch (Decoder->getSubtargetInfo().getHwMode(MCSubtargetInfo::HwMode_RegInfo)) {
// DISASM-NEXT:  case 0: // DefaultMode
// DISASM-NEXT:    return DecodeXRegsRegisterClass(Inst, Imm, Addr, Decoder);
// DISASM-NEXT:  case 1: // EvenMode
// DISASM-NEXT:    return DecodeXRegs_EvenRegisterClass(Inst, Imm, Addr, Decoder);
// DISASM-NEXT:  case 2: // OddMode
// DISASM-NEXT:    return DecodeXRegs_OddRegisterClass(Inst, Imm, Addr, Decoder);
// DISASM-NEXT:  default:
// DISASM-NEXT:    llvm_unreachable("no decoder for hwmode");
// DISASM-NEXT:  }
// DISASM-NEXT:}
// DISASM-EMPTY:
// DISASM{LITERAL}: [[maybe_unused]]
// DISASM-NEXT: static DecodeStatus DecodeYRegs_EvenIfRequiredRegClassByHwMode(MCInst &Inst, unsigned Imm, uint64_t Addr, const MCDisassembler *Decoder) {
// DISASM-NEXT:  switch (Decoder->getSubtargetInfo().getHwMode(MCSubtargetInfo::HwMode_RegInfo)) {
// DISASM-NEXT:  case 0: // DefaultMode
// DISASM-NEXT:    return DecodeYRegsRegisterClass(Inst, Imm, Addr, Decoder);
// DISASM-NEXT:  case 1: // EvenMode
// DISASM-NEXT:    return DecodeYRegs_EvenRegisterClass(Inst, Imm, Addr, Decoder);
// DISASM-NEXT:  default:
// DISASM-NEXT:    llvm_unreachable("no decoder for hwmode");
// DISASM-NEXT:  }
// DISASM-NEXT:}

// DISASM: static DecodeStatus decodeToMCInst(
// DISASM:   switch (Idx) {
// DISASM: case 0:
// DISASM: if (!Check(S, DecodeYRegs_EvenIfRequiredRegClassByHwMode(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
// DISASM: if (!Check(S, DecodeXRegs_EvenIfRequiredRegClassByHwMode(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }

// DISASM: case 1:
// DISASM: if (!Check(S, DecodeXRegs_EvenIfRequiredRegClassByHwMode(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }

// DISASM: case 2:
// DISASM: if (!Check(S, DecodeXRegs_EvenRegisterClass(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
// DISASM: case 3:
// DISASM: if (!Check(S, DecodeXRegsRegisterClass(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }

// DISASM: case 4:
// DISASM: if (!Check(S, YEvenIfRequiredCustomDecoder(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }


// ISEL-SDAG: MatcherTable
// ISEL-SDAG: OPC_SwitchOpcode /*2 cases */, {{[0-9]+}}, TARGET_VAL(ISD::STORE),
// ISEL-SDAG: OPC_RecordMemRef,
// ISEL-SDAG: OPC_RecordNode, // #0 = 'st' chained node
// ISEL-SDAG: OPC_RecordChild1, // #1 = $val
// ISEL-SDAG-NEXT: OPC_RecordChild2, // #2 = $src
// ISEL-SDAG-NEXT: OPC_CheckChild2TypeI64,
// ISEL-SDAG-NEXT: OPC_CheckPredicate2,  // Predicate_unindexedstore
// ISEL-SDAG-NEXT: OPC_CheckPredicate3,  // Predicate_store
// ISEL-SDAG: OPC_MorphNodeTo0, TARGET_VAL(MyTarget::MY_STORE), 0|OPFL_Chain|OPFL_MemRefs,

// ISEL-SDAG: /*SwitchOpcode*/ {{[0-9]+}}, TARGET_VAL(ISD::LOAD),
// ISEL-SDAG-NEXT: OPC_RecordMemRef,
// ISEL-SDAG-NEXT: OPC_RecordNode, // #0 = 'ld' chained node
// ISEL-SDAG-NEXT: OPC_RecordChild1, // #1 = $src
// ISEL-SDAG-NEXT: OPC_CheckTypeI64,
// ISEL-SDAG-NEXT: OPC_Scope, {{[0-9]+}}, /*->{{[0-9]+}}*/ // 2 children in Scope
// ISEL-SDAG-NEXT: OPC_CheckChild1TypeI32,
// ISEL-SDAG-NEXT: OPC_CheckPredicate0,  // Predicate_unindexedload
// ISEL-SDAG-NEXT: OPC_CheckPredicate1,  // Predicate_load
// ISEL-SDAG-NEXT: OPC_Scope, {{[0-9]+}}, /*->{{[0-9]+}}*/ // 3 children in Scope
// ISEL-SDAG-NEXT: OPC_CheckPatternPredicate1, // (Subtarget->hasAlignedRegisters())
// ISEL-SDAG-NEXT: OPC_EmitMergeInputChains1_0,
// ISEL-SDAG-NEXT: OPC_MorphNodeTo1, TARGET_VAL(MyTarget::MY_LOAD), 0|OPFL_Chain|OPFL_MemRefs,

// ISEL-SDAG: /*Scope*/
// ISEL-SDAG: OPC_CheckPatternPredicate2, // (Subtarget->hasUnalignedRegisters())
// ISEL-SDAG-NEXT: OPC_EmitMergeInputChains1_0,
// ISEL-SDAG-NEXT: OPC_MorphNodeTo1, TARGET_VAL(MyTarget::MY_LOAD), 0|OPFL_Chain|OPFL_MemRefs,

// ISEL-SDAG: /*Scope*/
// ISEL-SDAG: OPC_CheckPatternPredicate3, // !((Subtarget->hasAlignedRegisters())) && !((Subtarget->hasUnalignedRegisters())) && !((Subtarget->isPtr64()))
// ISEL-SDAG-NEXT: OPC_EmitMergeInputChains1_0,
// ISEL-SDAG-NEXT: OPC_MorphNodeTo1, TARGET_VAL(MyTarget::MY_LOAD), 0|OPFL_Chain|OPFL_MemRefs,

// ISEL-SDAG: /*Scope*/
// ISEL-SDAG-NEXT: OPC_CheckChild1TypeI64,
// ISEL-SDAG-NEXT: OPC_CheckPredicate0,  // Predicate_unindexedload
// ISEL-SDAG-NEXT: OPC_CheckPredicate1,  // Predicate_load
// ISEL-SDAG-NEXT: OPC_CheckPatternPredicate0, // (Subtarget->isPtr64())
// ISEL-SDAG-NEXT: OPC_EmitMergeInputChains1_0,
// ISEL-SDAG-NEXT: OPC_MorphNodeTo1, TARGET_VAL(MyTarget::MY_LOAD), 0|OPFL_Chain|OPFL_MemRefs,





// ISEL-GISEL: GIM_Try, /*On fail goto*//*Label 3*/ GIMT_Encode4(148),
// ISEL-GISEL-NEXT:   GIM_RootCheckType, /*Op*/0, /*Type*/GILLT_s64,
// ISEL-GISEL-NEXT:   GIM_CheckAtomicOrdering, /*MI*/0, /*Order*/(uint8_t)AtomicOrdering::NotAtomic,
// ISEL-GISEL-NEXT:   GIM_CheckMemorySizeEqualToLLT, /*MI*/0, /*MMO*/0, /*OpIdx*/0,
// ISEL-GISEL-NEXT:   GIM_RootCheckRegBankForClass, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::XRegsRegClassID),
// ISEL-GISEL-NEXT:   GIM_Try, /*On fail goto*//*Label 4*/ GIMT_Encode4(101),
// ISEL-GISEL-NEXT:     GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/32,

// FIXME: This should be a direct check for regbank, not have an incorrect class

// ISEL-GISEL-NEXT:     GIM_RootCheckRegBankForClass, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::PtrRegs32RegClassID),
// ISEL-GISEL-NEXT:     GIM_Try, /*On fail goto*//*Label 5*/ GIMT_Encode4(85), // Rule ID 1 //
// ISEL-GISEL-NEXT:       GIM_CheckFeatures, GIMT_Encode2(GIFBS_HwMode1),
// ISEL-GISEL-NEXT:       // (ld:{ *:[i64] } PtrRegOperand:{ *:[i32] }:$src)<<P:Predicate_unindexedload>><<P:Predicate_load>>  =>  (MY_LOAD:{ *:[i64] } ?:{ *:[i32] }:$src)
// ISEL-GISEL-NEXT:       GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::MY_LOAD),
// ISEL-GISEL-NEXT:       GIR_RootConstrainSelectedInstOperands,
// ISEL-GISEL-NEXT:       // GIR_Coverage, 1,
// ISEL-GISEL-NEXT:       GIR_Done,
// ISEL-GISEL-NEXT:     // Label 5: @85
// ISEL-GISEL-NEXT:     GIM_Try, /*On fail goto*//*Label 6*/ GIMT_Encode4(100), // Rule ID 2 //
// ISEL-GISEL-NEXT:       GIM_CheckFeatures, GIMT_Encode2(GIFBS_HwMode2),
// ISEL-GISEL-NEXT:       // (ld:{ *:[i64] } PtrRegOperand:{ *:[i32] }:$src)<<P:Predicate_unindexedload>><<P:Predicate_load>>  =>  (MY_LOAD:{ *:[i64] } ?:{ *:[i32] }:$src)
// ISEL-GISEL-NEXT:       GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::MY_LOAD),
// ISEL-GISEL-NEXT:       GIR_RootConstrainSelectedInstOperands,
// ISEL-GISEL-NEXT:       // GIR_Coverage, 2,
// ISEL-GISEL-NEXT:       GIR_Done,
// ISEL-GISEL-NEXT:     // Label 6: @100
// ISEL-GISEL-NEXT:     GIM_Reject,
// ISEL-GISEL-NEXT:   // Label 4: @101
// ISEL-GISEL-NEXT:   GIM_Try, /*On fail goto*//*Label 7*/ GIMT_Encode4(124), // Rule ID 3 //
// ISEL-GISEL-NEXT:     GIM_CheckFeatures, GIMT_Encode2(GIFBS_HwMode0),
// ISEL-GISEL-NEXT:     // MIs[0] src
// ISEL-GISEL-NEXT:     GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/64,
// ISEL-GISEL-NEXT:     GIM_RootCheckRegBankForClass, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::PtrRegs32RegClassID),
// ISEL-GISEL-NEXT:     // (ld:{ *:[i64] } PtrRegOperand:{ *:[i64] }:$src)<<P:Predicate_unindexedload>><<P:Predicate_load>>  =>  (MY_LOAD:{ *:[i64] } ?:{ *:[i64] }:$src)
// ISEL-GISEL-NEXT:     GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::MY_LOAD),
// ISEL-GISEL-NEXT:     GIR_RootConstrainSelectedInstOperands,
// ISEL-GISEL-NEXT:     // GIR_Coverage, 3,
// ISEL-GISEL-NEXT:     GIR_Done,
// ISEL-GISEL-NEXT:   // Label 7: @124
// ISEL-GISEL-NEXT:   GIM_Try, /*On fail goto*//*Label 8*/ GIMT_Encode4(147), // Rule ID 4 //
// ISEL-GISEL-NEXT:     GIM_CheckFeatures, GIMT_Encode2(GIFBS_HwMode3),
// ISEL-GISEL-NEXT:     // MIs[0] src
// ISEL-GISEL-NEXT:     GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/32,
// ISEL-GISEL-NEXT:     GIM_RootCheckRegBankForClass, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::PtrRegs32RegClassID),
// ISEL-GISEL-NEXT:     // (ld:{ *:[i64] } PtrRegOperand:{ *:[i32] }:$src)<<P:Predicate_unindexedload>><<P:Predicate_load>>  =>  (MY_LOAD:{ *:[i64] } ?:{ *:[i32] }:$src)
// ISEL-GISEL-NEXT:     GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::MY_LOAD),
// ISEL-GISEL-NEXT:     GIR_RootConstrainSelectedInstOperands,
// ISEL-GISEL-NEXT:     // GIR_Coverage, 4,
// ISEL-GISEL-NEXT:     GIR_Done,
// ISEL-GISEL-NEXT:   // Label 8: @147
// ISEL-GISEL-NEXT:   GIM_Reject,
// ISEL-GISEL-NEXT: // Label 3: @148
// ISEL-GISEL-NEXT: GIM_Reject,
// ISEL-GISEL-NEXT: // Label 1: @149
// ISEL-GISEL-NEXT: GIM_Try, /*On fail goto*//*Label 9*/ GIMT_Encode4(186), // Rule ID 0 //
// ISEL-GISEL-NEXT:   GIM_CheckFeatures, GIMT_Encode2(GIFBS_HwMode0),
// ISEL-GISEL-NEXT:   GIM_RootCheckType, /*Op*/0, /*Type*/GILLT_s64,
// ISEL-GISEL-NEXT:   GIM_CheckAtomicOrdering, /*MI*/0, /*Order*/(uint8_t)AtomicOrdering::NotAtomic,
// ISEL-GISEL-NEXT:   GIM_CheckMemorySizeEqualToLLT, /*MI*/0, /*MMO*/0, /*OpIdx*/0,
// ISEL-GISEL-NEXT:   GIM_RootCheckRegBankForClass, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::XRegsRegClassID),
// ISEL-GISEL-NEXT:   // MIs[0] src
// ISEL-GISEL-NEXT:   GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/64,
// ISEL-GISEL-NEXT:   GIM_RootCheckRegBankForClass, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::PtrRegs32RegClassID),
// ISEL-GISEL-NEXT:   // (st XRegs_EvenIfRequired:{ *:[i64] }:$val, MyPtrRC:{ *:[i64] }:$src)<<P:Predicate_unindexedstore>><<P:Predicate_store>>  =>  (MY_STORE ?:{ *:[i64] }:$val, XRegs_EvenIfRequired:{ *:[i64] }:$src)
// ISEL-GISEL-NEXT:   GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::MY_STORE),
// ISEL-GISEL-NEXT:   GIR_RootConstrainSelectedInstOperands,



def HasAlignedRegisters : Predicate<"Subtarget->hasAlignedRegisters()">;
def HasUnalignedRegisters : Predicate<"Subtarget->hasUnalignedRegisters()">;
def IsPtr64 : Predicate<"Subtarget->isPtr64()">;

// FIXME: In reality these should be mutually exclusive and we need
// the cross product of even mode / ptr size. i.e. EvenModePtr64,
// OddMode32 etc. For the purposes of this test where we won't be
// executing the code to compute a mode ID, it's simpler to pretend
// these are orthogonal.
def EvenMode : HwMode<[HasAlignedRegisters]>;
def OddMode : HwMode<[HasUnalignedRegisters]>;
def Ptr64 : HwMode<[IsPtr64]>;

class MyReg<string n>
  : Register<n> {
  let Namespace = "MyTarget";
}

class MyClass<int size, list<ValueType> types, dag registers>
  : RegisterClass<"MyTarget", types, size, registers> {
  let Size = size;
}

def X0 : MyReg<"x0">;
def X1 : MyReg<"x1">;
def X2 : MyReg<"x2">;
def X3 : MyReg<"x3">;
def X4 : MyReg<"x4">;
def X5 : MyReg<"x5">;
def X6 : MyReg<"x6">;

def Y0 : MyReg<"y0">;
def Y1 : MyReg<"y1">;
def Y2 : MyReg<"y2">;
def Y3 : MyReg<"y3">;
def Y4 : MyReg<"y4">;
def Y5 : MyReg<"y5">;
def Y6 : MyReg<"y6">;



def P0_32 : MyReg<"p0">;
def P1_32 : MyReg<"p1">;
def P2_32 : MyReg<"p2">;
def P3_32 : MyReg<"p3">;

def P0_64 : MyReg<"p0_64">;
def P1_64 : MyReg<"p1_64">;
def P2_64 : MyReg<"p2_64">;
def P3_64 : MyReg<"p3_64">;



def XRegs : RegisterClass<"MyTarget", [i64], 64, (add X0, X1, X2, X3, X4, X5, X6)>;
def XRegs_Odd : RegisterClass<"MyTarget", [i64], 64, (add X1, X3, X5)>;
def XRegs_Even : RegisterClass<"MyTarget", [i64], 64, (add X0, X2, X4, X6)>;

def XRegs_EvenIfRequired : RegClassByHwMode<[DefaultMode, EvenMode, OddMode],
                                           [XRegs, XRegs_Even, XRegs_Odd]>;

def YRegs : RegisterClass<"MyTarget", [i64], 64, (add Y0, Y1, Y2, Y3, Y4, Y5, Y6)>;
def YRegs_Even : RegisterClass<"MyTarget", [i64], 64, (add Y0, Y2, Y4, Y6)>;

def YRegs_EvenIfRequired : RegClassByHwMode<[DefaultMode, EvenMode],
                                            [YRegs, YRegs_Even]>;


def PtrRegs32 : RegisterClass<"MyTarget", [i32], 32, (add P0_32, P1_32, P2_32, P3_32)>;
def PtrRegs64 : RegisterClass<"MyTarget", [i64], 64, (add P0_64, P1_64, P2_64, P3_64)>;

def MyPtrRC : RegClassByHwMode<[DefaultMode, Ptr64],
                               [PtrRegs32, PtrRegs64]>;


def PtrRegOperand : RegisterOperand<MyPtrRC>;


def CustomDecodeYEvenIfRequired : RegisterOperand<YRegs_EvenIfRequired> {
  let DecoderMethod = "YEvenIfRequiredCustomDecoder";
}

class TestInstruction : Instruction {
  let Size = 2;
  let Namespace = "MyTarget";
  let hasSideEffects = false;
  let hasExtraSrcRegAllocReq = false;
  let hasExtraDefRegAllocReq = false;

  field bits<16> Inst;
  bits<3> dst;
  bits<3> src;
  bits<3> opcode;

  let Inst{2-0} = dst;
  let Inst{5-3} = src;
  let Inst{7-5} = opcode;
}

def SpecialOperand : RegisterOperand<XRegs_EvenIfRequired>;

def MY_MOV : TestInstruction {
  let OutOperandList = (outs YRegs_EvenIfRequired:$dst);
  let InOperandList = (ins XRegs_EvenIfRequired:$src);
  let AsmString = "my_mov $dst, $src";
  let opcode = 0;
}

def EVEN_IF_MODE : TestInstruction {
  let OutOperandList = (outs);
  let InOperandList = (ins XRegs_EvenIfRequired:$src);
  let AsmString = "even_if_mode $src";
  let opcode = 1;
}

def ALWAYS_EVEN : TestInstruction {
  let OutOperandList = (outs);
  let InOperandList = (ins XRegs_Even:$src);
  let AsmString = "always_even $src";
  let opcode = 2;
}

def ALWAYS_ALL : TestInstruction {
  let OutOperandList = (outs);
  let InOperandList = (ins XRegs:$src);
  let AsmString = "always_all $src";
  let opcode = 3;
}

def CUSTOM_DECODE : TestInstruction {
  let OutOperandList = (outs);
  let InOperandList = (ins CustomDecodeYEvenIfRequired:$src);
  let AsmString = "custom_decode $src";
  let opcode = 4;
}

def MyTargetMov : SDNode<"MyTarget::MY_MOV", SDTUnaryOp, []>;

// Test 2 different cases directly in the instruction
def MY_STORE : TestInstruction {
  let OutOperandList = (outs);
  let InOperandList = (ins XRegs_EvenIfRequired:$src, MyPtrRC:$dst);
  let AsmString = "my_store $src, $dst";
  let opcode = 5;
}

// Test 2 different cases wrapped by RegisterOperand
def MY_LOAD : TestInstruction {
  let OutOperandList = (outs RegisterOperand<XRegs_EvenIfRequired>:$dst);
  let InOperandList = (ins PtrRegOperand:$src);
  let AsmString = "my_load $dst, $src";
  let opcode = 6;
}

// Direct RegClassByHwMode reference
def MY_LOAD_ALIAS1 : InstAlias<"also_my_load_1 $dst, $src",
    (MY_LOAD XRegs_EvenIfRequired:$dst, MyPtrRC:$src)>;

// RegClassByHwMode wrapped in RegisterOperand
def MY_LOAD_ALIAS2 : InstAlias<"also_my_load_2 $dst, $src",
    (MY_LOAD RegisterOperand<XRegs_EvenIfRequired>:$dst, PtrRegOperand:$src)>;

// Direct RegClassByHwMode usage
def : Pat<
  (store XRegs_EvenIfRequired:$val, MyPtrRC:$src),
  (MY_STORE $val, XRegs_EvenIfRequired:$src)
>;

// Wrapped in RegisterOperand
def : Pat<
  (i64 (load PtrRegOperand:$src)),
  (MY_LOAD $src)
>;

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }
