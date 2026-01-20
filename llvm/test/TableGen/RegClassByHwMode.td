// RUN: llvm-tblgen -gen-instr-info   -I %S -I %p/../../include %s -o - | FileCheck -check-prefix=INSTRINFO %s
// RUN: llvm-tblgen -gen-asm-matcher  -I %S -I %p/../../include %s -o - | FileCheck -check-prefix=ASMMATCHER %s
// RUN: llvm-tblgen -gen-disassembler -I %S -I %p/../../include %s -o - | FileCheck -check-prefix=DISASM %s
// RUN: llvm-tblgen -gen-dag-isel     -I %S -I %p/../../include %s -o - | FileCheck -check-prefix=ISEL-SDAG %s
// RUN: llvm-tblgen -gen-global-isel  -I %S -I %p/../../include %s -o - | FileCheck -check-prefix=ISEL-GISEL %s

include "Common/RegClassByHwModeCommon.td"

// INSTRINFO:      #ifdef GET_INSTRINFO_ENUM
// INSTRINFO-NEXT: #undef GET_INSTRINFO_ENUM
// INSTRINFO-EMPTY:
// INSTRINFO-NEXT: namespace llvm::MyTarget {
// INSTRINFO-EMPTY:
// INSTRINFO-NEXT: enum {
// INSTRINFO-NEXT:   PHI
// INSTRINFO: LOAD_STACK_GUARD = [[LOAD_STACK_GUARD_OPCODE:[0-9]+]]
// INSTRINFO:      };
// INSTRINFO:      enum RegClassByHwModeUses : uint16_t {
// INSTRINFO-NEXT:   MyPtrRC,
// INSTRINFO-NEXT:   XRegs_EvenIfRequired,
// INSTRINFO-NEXT:   YRegs_EvenIfRequired,
// INSTRINFO-NEXT: };
// INSTRINFO-EMPTY:
// INSTRINFO-NEXT: } // namespace llvm::MyTarget


// INSTRINFO: { [[LOAD_STACK_GUARD_OPCODE]],	1,	1,	0,	0,	0,	0,	MyTargetOpInfoBase + [[LOAD_STACK_GUARD_OP_INDEX:[0-9]+]],	0,	0|(1ULL<<MCID::Pseudo)|(1ULL<<MCID::MayLoad)|(1ULL<<MCID::Rematerializable)|(1ULL<<MCID::ExtraSrcRegAllocReq)|(1ULL<<MCID::ExtraDefRegAllocReq), 0x0ULL },  // anonymous_

// INSTRINFO: /* [[LOAD_STACK_GUARD_OP_INDEX]] */ { MyTarget::XRegs_EvenIfRequired, 0|(1<<MCOI::LookupRegClassByHwMode), MCOI::OPERAND_REGISTER, 0 },
// INSTRINFO: { MyTarget::XRegs_EvenIfRequired, 0|(1<<MCOI::LookupRegClassByHwMode), MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 },
// INSTRINFO: { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { -1, 0, MCOI::OPERAND_IMMEDIATE, 0 },
// INSTRINFO: { MyTarget::XRegs_EvenIfRequired, 0|(1<<MCOI::LookupRegClassByHwMode), MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 },
// INSTRINFO: { -1, 0, MCOI::OPERAND_UNKNOWN, 0 }, { MyTarget::XRegs_EvenIfRequired, 0|(1<<MCOI::LookupRegClassByHwMode), MCOI::OPERAND_REGISTER, 0 }, { -1, 0, MCOI::OPERAND_UNKNOWN, 0 },

// INSTRINFO: { MyTarget::XRegsRegClassID, 0, MCOI::OPERAND_REGISTER, 0 },
// INSTRINFO: { MyTarget::XRegs_EvenRegClassID, 0, MCOI::OPERAND_REGISTER, 0 },

// INSTRINFO: { MyTarget::YRegs_EvenIfRequired, 0|(1<<MCOI::LookupRegClassByHwMode), MCOI::OPERAND_REGISTER, 0 },

// INSTRINFO: { MyTarget::XRegs_EvenIfRequired, 0|(1<<MCOI::LookupRegClassByHwMode), MCOI::OPERAND_REGISTER, 0 }, { MyTarget::MyPtrRC, 0|(1<<MCOI::LookupRegClassByHwMode), MCOI::OPERAND_REGISTER, 0 },
// INSTRINFO: { MyTarget::YRegs_EvenIfRequired, 0|(1<<MCOI::LookupRegClassByHwMode), MCOI::OPERAND_REGISTER, 0 }, { MyTarget::XRegs_EvenIfRequired, 0|(1<<MCOI::LookupRegClassByHwMode), MCOI::OPERAND_REGISTER, 0 },

// INSTRINFO: extern const int16_t MyTargetRegClassByHwModeTables[4][3] = {
// INSTRINFO-NEXT:  { // DefaultMode
// INSTRINFO-NEXT:    MyTarget::PtrRegs32RegClassID, // MyPtrRC
// INSTRINFO-NEXT:    MyTarget::XRegsRegClassID, // XRegs_EvenIfRequired
// INSTRINFO-NEXT:    MyTarget::YRegsRegClassID, // YRegs_EvenIfRequired
// INSTRINFO-NEXT:  },
// INSTRINFO-NEXT:  { // EvenMode
// INSTRINFO-NEXT:    -1, // Missing mode entry for MyPtrRC
// INSTRINFO-NEXT:    MyTarget::XRegs_EvenRegClassID, // XRegs_EvenIfRequired
// INSTRINFO-NEXT:    MyTarget::YRegs_EvenRegClassID, // YRegs_EvenIfRequired
// INSTRINFO-NEXT:  },
// INSTRINFO-NEXT:  { // OddMode
// INSTRINFO-NEXT:    -1, // Missing mode entry for MyPtrRC
// INSTRINFO-NEXT:    MyTarget::XRegs_OddRegClassID, // XRegs_EvenIfRequired
// INSTRINFO-NEXT:    -1, // Missing mode entry for YRegs_EvenIfRequired
// INSTRINFO-NEXT:  },
// INSTRINFO-NEXT:  { // Ptr64
// INSTRINFO-NEXT:    MyTarget::PtrRegs64RegClassID, // MyPtrRC
// INSTRINFO-NEXT:    -1, // Missing mode entry for XRegs_EvenIfRequired
// INSTRINFO-NEXT:    -1, // Missing mode entry for YRegs_EvenIfRequired
// INSTRINFO-NEXT:  },
// INSTRINFO-NEXT: };

// INSTRINFO: static inline void InitMyTargetMCInstrInfo(
// INSTRINFO-NEXT: II->InitMCInstrInfo(MyTargetDescs.Insts, MyTargetInstrNameIndices, MyTargetInstrNameData, nullptr, nullptr, {{[0-9]+}}, &MyTargetRegClassByHwModeTables[0][0], 3);

// INSTRINFO: #ifdef GET_INSTRINFO_HEADER
// INSTRINFO-NEXT: #undef GET_INSTRINFO_HEADER
// INSTRINFO-EMPTY:
// INSTRINFO-NEXT: namespace llvm {
// INSTRINFO-EMPTY:
// INSTRINFO-NEXT: struct MyTargetGenInstrInfo : public TargetInstrInfo {
// INSTRINFO-NEXT:   explicit MyTargetGenInstrInfo(const TargetSubtargetInfo &STI, const TargetRegisterInfo &TRI, unsigned CFSetupOpcode = ~0u, unsigned CFDestroyOpcode = ~0u, unsigned CatchRetOpcode = ~0u, unsigned ReturnOpcode = ~0u);
// INSTRINFO-NEXT:   ~MyTargetGenInstrInfo() override = default;
// INSTRINFO-NEXT: };
// INSTRINFO-NEXT: extern const int16_t MyTargetRegClassByHwModeTables[4][3];
// INSTRINFO-EMPTY:
// INSTRINFO-NEXT: } // namespace llvm
// INSTRINFO: #endif // GET_INSTRINFO_HEADER


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
// ASMMATCHER-NEXT:        MCK_PtrRegs32, // MyPtrRC
// ASMMATCHER-NEXT:        MCK_XRegs, // XRegs_EvenIfRequired
// ASMMATCHER-NEXT:        MCK_YRegs, // YRegs_EvenIfRequired
// ASMMATCHER-NEXT:      },
// ASMMATCHER-NEXT:      { // EvenMode
// ASMMATCHER-NEXT:        InvalidMatchClass, // Missing mode entry for MyPtrRC
// ASMMATCHER-NEXT:        MCK_XRegs_Even, // XRegs_EvenIfRequired
// ASMMATCHER-NEXT:        MCK_YRegs_Even, // YRegs_EvenIfRequired
// ASMMATCHER-NEXT:      },
// ASMMATCHER-NEXT:      { // OddMode
// ASMMATCHER-NEXT:        InvalidMatchClass, // Missing mode entry for MyPtrRC
// ASMMATCHER-NEXT:        MCK_XRegs_Odd, // XRegs_EvenIfRequired
// ASMMATCHER-NEXT:        InvalidMatchClass, // Missing mode entry for YRegs_EvenIfRequired
// ASMMATCHER-NEXT:      },
// ASMMATCHER-NEXT:      { // Ptr64
// ASMMATCHER-NEXT:        MCK_PtrRegs64, // MyPtrRC
// ASMMATCHER-NEXT:        InvalidMatchClass, // Missing mode entry for XRegs_EvenIfRequired
// ASMMATCHER-NEXT:        InvalidMatchClass, // Missing mode entry for YRegs_EvenIfRequired
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
// ISEL-SDAG-NEXT: OPC_RecordMemRef,
// ISEL-SDAG-NEXT: OPC_RecordNode, // #0 = 'st' chained node
// ISEL-SDAG-NEXT: OPC_RecordChild1, // #1 = $val
// ISEL-SDAG-NEXT: OPC_CheckChild1TypeByHwMode, /*{(*:i64),(m1:i64),(m2:i64)}*/0,
// ISEL-SDAG-NEXT: OPC_RecordChild2, // #2 = $src
// ISEL-SDAG-NEXT: OPC_CheckChild2TypeByHwMode, /*{(*:i32),(m3:i64)}*/1,
// ISEL-SDAG-NEXT: OPC_CheckPredicate0,  // Predicate_unindexedstore
// ISEL-SDAG-NEXT: OPC_CheckPredicate1,  // Predicate_store
// ISEL-SDAG-NEXT: OPC_EmitMergeInputChains1_0,
// ISEL-SDAG-NEXT: OPC_MorphNodeTo0, TARGET_VAL(MyTarget::MY_STORE), 0|OPFL_Chain|OPFL_MemRefs,

// ISEL-SDAG: /*SwitchOpcode*/ {{[0-9]+}}, TARGET_VAL(ISD::LOAD),
// ISEL-SDAG-NEXT: OPC_RecordMemRef,
// ISEL-SDAG-NEXT: OPC_RecordNode, // #0 = 'ld' chained node
// ISEL-SDAG-NEXT: OPC_RecordChild1, // #1 = $src
// ISEL-SDAG-NEXT: OPC_CheckChild1TypeByHwMode, /*{(*:i32),(m3:i64)}*/1,
// ISEL-SDAG-NEXT: OPC_CheckPredicate2,  // Predicate_unindexedload
// ISEL-SDAG-NEXT: OPC_CheckPredicate3,  // Predicate_load
// ISEL-SDAG-NEXT: OPC_CheckTypeI64,
// ISEL-SDAG-NEXT: OPC_EmitMergeInputChains1_0,
// ISEL-SDAG-NEXT: OPC_MorphNodeToByHwMode, TARGET_VAL(MyTarget::MY_LOAD), 0|OPFL_Chain|OPFL_MemRefs,





// ISEL-GISEL: GIM_Try, /*On fail goto*//*Label 3*/ GIMT_Encode4(148),
// ISEL-GISEL-NEXT:   GIM_RootCheckType, /*Op*/0, /*Type*/GILLT_s64,
// ISEL-GISEL-NEXT:   GIM_CheckAtomicOrdering, /*MI*/0, /*Order*/(uint8_t)AtomicOrdering::NotAtomic,
// ISEL-GISEL-NEXT:   GIM_CheckMemorySizeEqualToLLT, /*MI*/0, /*MMO*/0, /*OpIdx*/0,
// ISEL-GISEL-NEXT:   GIM_RootCheckRegBankForClass, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::XRegsRegClassID),
// ISEL-GISEL-NEXT:   GIM_Try, /*On fail goto*//*Label 4*/ GIMT_Encode4(101),
// ISEL-GISEL-NEXT:     GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/32,

// FIXME: This should be a direct check for regbank, not have an incorrect class

// ISEL-GISEL-NEXT:     GIM_RootCheckRegBankForClass, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::PtrRegs32RegClassID),
// ISEL-GISEL-NEXT:     GIM_Try, /*On fail goto*//*Label 5*/ GIMT_Encode4(85), // Rule ID 4 //
// ISEL-GISEL-NEXT:       GIM_CheckFeatures, GIMT_Encode2(GIFBS_HwMode0),
// ISEL-GISEL-NEXT:       // (ld:{ *:[i64] } PtrRegOperand:{ *:[i32] }:$src)<<P:Predicate_unindexedload>><<P:Predicate_load>>  =>  (MY_LOAD:{ *:[i64] } ?:{ *:[i32] }:$src)
// ISEL-GISEL-NEXT:       GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::MY_LOAD),
// ISEL-GISEL-NEXT:       GIR_RootConstrainSelectedInstOperands,
// ISEL-GISEL-NEXT:       // GIR_Coverage, 4,
// ISEL-GISEL-NEXT:       GIR_Done,
// ISEL-GISEL-NEXT:     // Label 5: @85
// ISEL-GISEL-NEXT:     GIM_Try, /*On fail goto*//*Label 6*/ GIMT_Encode4(100), // Rule ID 5 //
// ISEL-GISEL-NEXT:       GIM_CheckFeatures, GIMT_Encode2(GIFBS_HwMode1),
// ISEL-GISEL-NEXT:       // (ld:{ *:[i64] } PtrRegOperand:{ *:[i32] }:$src)<<P:Predicate_unindexedload>><<P:Predicate_load>>  =>  (MY_LOAD:{ *:[i64] } ?:{ *:[i32] }:$src)
// ISEL-GISEL-NEXT:       GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::MY_LOAD),
// ISEL-GISEL-NEXT:       GIR_RootConstrainSelectedInstOperands,
// ISEL-GISEL-NEXT:       // GIR_Coverage, 5,
// ISEL-GISEL-NEXT:       GIR_Done,
// ISEL-GISEL-NEXT:     // Label 6: @100
// ISEL-GISEL-NEXT:     GIM_Reject,
// ISEL-GISEL-NEXT:   // Label 4: @101
// ISEL-GISEL-NEXT:   GIM_Try, /*On fail goto*//*Label 7*/ GIMT_Encode4(124), // Rule ID 6 //
// ISEL-GISEL-NEXT:     GIM_CheckFeatures, GIMT_Encode2(GIFBS_HwMode2),
// ISEL-GISEL-NEXT:     // MIs[0] src
// ISEL-GISEL-NEXT:     GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/64,
// ISEL-GISEL-NEXT:     GIM_RootCheckRegBankForClass, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::PtrRegs32RegClassID),
// ISEL-GISEL-NEXT:     // (ld:{ *:[i64] } PtrRegOperand:{ *:[i64] }:$src)<<P:Predicate_unindexedload>><<P:Predicate_load>>  =>  (MY_LOAD:{ *:[i64] } ?:{ *:[i64] }:$src)
// ISEL-GISEL-NEXT:     GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::MY_LOAD),
// ISEL-GISEL-NEXT:     GIR_RootConstrainSelectedInstOperands,
// ISEL-GISEL-NEXT:     // GIR_Coverage, 6,
// ISEL-GISEL-NEXT:     GIR_Done,
// ISEL-GISEL-NEXT:   // Label 7: @124
// ISEL-GISEL-NEXT:   GIM_Try, /*On fail goto*//*Label 8*/ GIMT_Encode4(147), // Rule ID 7 //
// ISEL-GISEL-NEXT:     GIM_CheckFeatures, GIMT_Encode2(GIFBS_HwMode3),
// ISEL-GISEL-NEXT:     // MIs[0] src
// ISEL-GISEL-NEXT:     GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/32,
// ISEL-GISEL-NEXT:     GIM_RootCheckRegBankForClass, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::PtrRegs32RegClassID),
// ISEL-GISEL-NEXT:     // (ld:{ *:[i64] } PtrRegOperand:{ *:[i32] }:$src)<<P:Predicate_unindexedload>><<P:Predicate_load>>  =>  (MY_LOAD:{ *:[i64] } ?:{ *:[i32] }:$src)
// ISEL-GISEL-NEXT:     GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::MY_LOAD),
// ISEL-GISEL-NEXT:     GIR_RootConstrainSelectedInstOperands,
// ISEL-GISEL-NEXT:     // GIR_Coverage, 7,
// ISEL-GISEL-NEXT:     GIR_Done,
// ISEL-GISEL-NEXT:   // Label 8: @147
// ISEL-GISEL-NEXT:   GIM_Reject,
// ISEL-GISEL-NEXT: // Label 3: @148
// ISEL-GISEL-NEXT: GIM_Reject,
// ISEL-GISEL-NEXT: // Label 1: @149
// ISEL-GISEL-NEXT: GIM_Try, /*On fail goto*//*Label 9*/ GIMT_Encode4(259),
// ISEL-GISEL-NEXT:   GIM_RootCheckType, /*Op*/0, /*Type*/GILLT_s64,
// ISEL-GISEL-NEXT:   GIM_CheckAtomicOrdering, /*MI*/0, /*Order*/(uint8_t)AtomicOrdering::NotAtomic,
// ISEL-GISEL-NEXT:   GIM_CheckMemorySizeEqualToLLT, /*MI*/0, /*MMO*/0, /*OpIdx*/0,
// ISEL-GISEL-NEXT:   GIM_RootCheckRegBankForClass, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::XRegsRegClassID),
// ISEL-GISEL-NEXT:   GIM_Try, /*On fail goto*//*Label 10*/ GIMT_Encode4(212),
// ISEL-GISEL-NEXT:     GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/32,
// ISEL-GISEL-NEXT:     GIM_RootCheckRegBankForClass, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::PtrRegs32RegClassID),
// ISEL-GISEL-NEXT:     GIM_Try, /*On fail goto*//*Label 11*/ GIMT_Encode4(196), // Rule ID 0 //
// ISEL-GISEL-NEXT:       GIM_CheckFeatures, GIMT_Encode2(GIFBS_HwMode0),
// ISEL-GISEL-NEXT:       // (st XRegs_EvenIfRequired:{ *:[i64] }:$val, MyPtrRC:{ *:[i32] }:$src)<<P:Predicate_unindexedstore>><<P:Predicate_store>>  =>  (MY_STORE XRegs_EvenIfRequired:{ *:[i64] }:$val, ?:{ *:[i32] }:$src)
// ISEL-GISEL-NEXT:       GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::MY_STORE),
// ISEL-GISEL-NEXT:       GIR_RootConstrainSelectedInstOperands,
// ISEL-GISEL-NEXT:       // GIR_Coverage, 0,
// ISEL-GISEL-NEXT:       GIR_Done,
// ISEL-GISEL-NEXT:     // Label 11: @196
// ISEL-GISEL-NEXT:     GIM_Try, /*On fail goto*//*Label 12*/ GIMT_Encode4(211), // Rule ID 1 //
// ISEL-GISEL-NEXT:       GIM_CheckFeatures, GIMT_Encode2(GIFBS_HwMode1),
// ISEL-GISEL-NEXT:       // (st XRegs_EvenIfRequired:{ *:[i64] }:$val, MyPtrRC:{ *:[i32] }:$src)<<P:Predicate_unindexedstore>><<P:Predicate_store>>  =>  (MY_STORE XRegs_EvenIfRequired:{ *:[i64] }:$val, ?:{ *:[i32] }:$src)
// ISEL-GISEL-NEXT:       GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::MY_STORE),
// ISEL-GISEL-NEXT:       GIR_RootConstrainSelectedInstOperands,
// ISEL-GISEL-NEXT:       // GIR_Coverage, 1,
// ISEL-GISEL-NEXT:       GIR_Done,
// ISEL-GISEL-NEXT:     // Label 12: @211
// ISEL-GISEL-NEXT:     GIM_Reject,
// ISEL-GISEL-NEXT:   // Label 10: @212
// ISEL-GISEL-NEXT:   GIM_Try, /*On fail goto*//*Label 13*/ GIMT_Encode4(235), // Rule ID 2 //
// ISEL-GISEL-NEXT:     GIM_CheckFeatures, GIMT_Encode2(GIFBS_HwMode2),
// ISEL-GISEL-NEXT:     // MIs[0] src
// ISEL-GISEL-NEXT:     GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/64,
// ISEL-GISEL-NEXT:     GIM_RootCheckRegBankForClass, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::PtrRegs32RegClassID),
// ISEL-GISEL-NEXT:     // (st XRegs_EvenIfRequired:{ *:[i64] }:$val, MyPtrRC:{ *:[i64] }:$src)<<P:Predicate_unindexedstore>><<P:Predicate_store>>  =>  (MY_STORE XRegs_EvenIfRequired:{ *:[i64] }:$val, ?:{ *:[i64] }:$src)
// ISEL-GISEL-NEXT:     GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::MY_STORE),
// ISEL-GISEL-NEXT:     GIR_RootConstrainSelectedInstOperands,
// ISEL-GISEL-NEXT:     // GIR_Coverage, 2,
// ISEL-GISEL-NEXT:     GIR_Done,
// ISEL-GISEL-NEXT:   // Label 13: @235
// ISEL-GISEL-NEXT:   GIM_Try, /*On fail goto*//*Label 14*/ GIMT_Encode4(258), // Rule ID 3 //
// ISEL-GISEL-NEXT:     GIM_CheckFeatures, GIMT_Encode2(GIFBS_HwMode3),
// ISEL-GISEL-NEXT:     // MIs[0] src
// ISEL-GISEL-NEXT:     GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/32,
// ISEL-GISEL-NEXT:     GIM_RootCheckRegBankForClass, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::PtrRegs32RegClassID),
// ISEL-GISEL-NEXT:     // (st XRegs_EvenIfRequired:{ *:[i64] }:$val, MyPtrRC:{ *:[i32] }:$src)<<P:Predicate_unindexedstore>><<P:Predicate_store>>  =>  (MY_STORE XRegs_EvenIfRequired:{ *:[i64] }:$val, ?:{ *:[i32] }:$src)
// ISEL-GISEL-NEXT:     GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::MY_STORE)
// ISEL-GISEL-NEXT:     GIR_RootConstrainSelectedInstOperands,
// ISEL-GISEL-NEXT:     // GIR_Coverage, 3,
// ISEL-GISEL-NEXT:   GIR_Done,
// ISEL-GISEL-NEXT:   // Label 14: @258
// ISEL-GISEL-NEXT:   GIM_Reject,
// ISEL-GISEL-NEXT: // Label 9: @259
// ISEL-GISEL-NEXT: GIM_Reject,
// ISEL-GISEL-NEXT: // Label 2: @260
// ISEL-GISEL-NEXT: GIM_Reject,
// ISEL-GISEL-NEXT: }; // Size: 261 bytes

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

def P0_32 : MyReg<"p0">;
def P1_32 : MyReg<"p1">;
def P2_32 : MyReg<"p2">;
def P3_32 : MyReg<"p3">;

def P0_64 : MyReg<"p0_64">;
def P1_64 : MyReg<"p1_64">;
def P2_64 : MyReg<"p2_64">;
def P3_64 : MyReg<"p3_64">;


def XRegs_Odd : RegisterClass<"MyTarget", [i64], 64, (add X1, X3, X5)>;
def XRegs_Even : RegisterClass<"MyTarget", [i64], 64, (add X0, X2, X4, X6)>;

def XRegs_EvenIfRequired : RegClassByHwMode<[DefaultMode, EvenMode, OddMode],
                                            [XRegs, XRegs_Even, XRegs_Odd]>;

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
  (MY_STORE XRegs_EvenIfRequired:$val, $src)
>;

// Wrapped in RegisterOperand
def : Pat<
  (i64 (load PtrRegOperand:$src)),
  (MY_LOAD $src)
>;

defm : RemapAllTargetPseudoPointerOperands<XRegs_EvenIfRequired>;

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }
