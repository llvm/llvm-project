// RUN: llvm-tblgen %s -gen-global-isel -optimize-match-table=true -I %p/../../../include -I %p/../Common | FileCheck %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

// Check the MatchTable recursively optimizes to maximize group formation.

def InstFourOperands : I<(outs GPR32:$dst), (ins GPR32:$src1, GPR32:$src2, GPR32:$src3, GPR32:$src4), []>;

def S0 : Register<"s0"> { let Namespace = "MyTarget"; }
def S1 : Register<"s1"> { let Namespace = "MyTarget"; }
def Z0 : Register<"s1"> { let Namespace = "MyTarget"; }
def Z1 : Register<"s1"> { let Namespace = "MyTarget"; }

def GPR8_A     : RegisterClass<"MyTarget", [i8], 8, (add Z0)>;
def LOAD8_A    : I<(outs GPR8_A:$dst), (ins GPR8_A:$src), []>;
def LOAD8Imm_A : I<(outs GPR8_A:$dst), (ins GPR8_A:$src), []>;

def GPR8_B     : RegisterClass<"MyTarget", [i8], 8, (add Z1)>;
def LOAD8_B    : I<(outs GPR8_B:$dst), (ins GPR8_B:$src), []>;
def LOAD8Imm_B : I<(outs GPR8_B:$dst), (ins GPR8_B:$src), []>;

def GPR16_A     : RegisterClass<"MyTarget", [i16], 16, (add S0)>;
def LOAD16_A    : I<(outs GPR16_A:$dst), (ins GPR16_A:$src), []>;
def LOAD16Imm_A : I<(outs GPR16_A:$dst), (ins GPR16_A:$src), []>;

def GPR16_B     : RegisterClass<"MyTarget", [i16], 16, (add S0)>;
def LOAD16_B    : I<(outs GPR16_B:$dst), (ins GPR16_B:$src), []>;
def LOAD16Imm_B : I<(outs GPR16_B:$dst), (ins GPR16_B:$src), []>;


def : Pat<(i8 (load GPR8_A:$src)),
          (LOAD8_A GPR8_A:$src)>;
def : Pat<(i8 (load GPR8_B:$src)),
          (LOAD8_B GPR8_B:$src)>;

def : Pat<(i8 (load (add GPR8_A:$src, 10))),
          (LOAD8Imm_A GPR8_A:$src)>;
def : Pat<(i8 (load (add GPR8_B:$src, 10))),
          (LOAD8Imm_B GPR8_B:$src)>;

def : Pat<(i16 (load GPR16_A:$src)),
          (LOAD16_A GPR16_A:$src)>;
def : Pat<(i16 (load GPR16_B:$src)),
          (LOAD16_B GPR16_B:$src)>;

def : Pat<(i16 (load (add GPR16_A:$src, 10))),
          (LOAD16Imm_A GPR16_A:$src)>;
def : Pat<(i16 (load (add GPR16_B:$src, 10))),
          (LOAD16Imm_B GPR16_B:$src)>;


// CHECK:      const uint8_t *MyTargetInstructionSelector::getMatchTable() const {
// CHECK-NEXT:   constexpr static uint8_t MatchTable0[] = {
// CHECK-NEXT:      /*   0 */ GIM_Try, /*On fail goto*//*Label 0*/ GIMT_Encode4(361),
// CHECK-NEXT:      /*   5 */   GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_LOAD),
// CHECK-NEXT:      /*   9 */   GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/GIMT_Encode2(0), GIMT_Encode2(2), /*)*//*default:*//*Label 3*/ GIMT_Encode4(360),
// CHECK-NEXT:      /*  20 */   /*GILLT_s8*//*Label 1*/ GIMT_Encode4(28),
// CHECK-NEXT:      /*  24 */   /*GILLT_s16*//*Label 2*/ GIMT_Encode4(194),
// CHECK-NEXT:      /*  28 */   // Label 1: @28
// CHECK-NEXT:      /*  28 */   GIM_Try, /*On fail goto*//*Label 4*/ GIMT_Encode4(193),
// CHECK-NEXT:      /*  33 */     GIM_CheckAtomicOrdering, /*MI*/0, /*Order*/(uint8_t)AtomicOrdering::NotAtomic,
// CHECK-NEXT:      /*  36 */     GIM_CheckMemorySizeEqualToLLT, /*MI*/0, /*MMO*/0, /*OpIdx*/0,
// CHECK-NEXT:      /*  40 */     GIM_Try, /*On fail goto*//*Label 5*/ GIMT_Encode4(94), // Rule ID 2 //
// CHECK-NEXT:      /*  45 */       GIM_RootCheckRegBankForClass0, /*RC*/GIMT_Encode2(MyTarget::GPR8_ARegClassID),
// CHECK-NEXT:      /*  48 */       // MIs[0] Operand 1
// CHECK-NEXT:      /*  48 */       GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/8,
// CHECK-NEXT:      /*  52 */       GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK-NEXT:      /*  56 */       GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_ADD),
// CHECK-NEXT:      /*  60 */       GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s8,
// CHECK-NEXT:      /*  64 */       GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s8,
// CHECK-NEXT:      /*  68 */       GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPR8_ARegClassID),
// CHECK-NEXT:      /*  73 */       GIM_CheckConstantInt8, /*MI*/1, /*Op*/2, 10,
// CHECK-NEXT:      /*  77 */       GIM_CheckIsSafeToFold, /*NumInsns*/1,
// CHECK-NEXT:      /*  79 */       // (ld:{ *:[i8] } (add:{ *:[i8] } GPR8_A:{ *:[i8] }:$src, 10:{ *:[i8] }))<<P:Predicate_unindexedload>><<P:Predicate_load>>  =>  (LOAD8Imm_A:{ *:[i8] } GPR8_A:{ *:[i8] }:$src)
// CHECK-NEXT:      /*  79 */       GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::LOAD8Imm_A),
// CHECK-NEXT:      /*  82 */       GIR_RootToRootCopy0, // DstI[dst]
// CHECK-NEXT:      /*  83 */       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
// CHECK-NEXT:      /*  87 */       GIR_MergeMemOperands, /*InsnID*/0, /*NumInsns*/2, /*MergeInsnID's*/0, 1,
// CHECK-NEXT:      /*  92 */       GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT:      /*  93 */       // GIR_Coverage, 2,
// CHECK-NEXT:      /*  93 */       GIR_EraseRootFromParent_Done,
// CHECK-NEXT:      /*  94 */     // Label 5: @94
// CHECK-NEXT:      /*  94 */     GIM_Try, /*On fail goto*//*Label 6*/ GIMT_Encode4(148), // Rule ID 3 //
// CHECK-NEXT:      /* 99 */       GIM_RootCheckRegBankForClass0, /*RC*/GIMT_Encode2(MyTarget::GPR8_BRegClassID),
// CHECK-NEXT:      /* 102 */       // MIs[0] Operand 1
// CHECK-NEXT:      /* 102 */       GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/8,
// CHECK-NEXT:      /* 106 */       GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK-NEXT:      /* 110 */       GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_ADD),
// CHECK-NEXT:      /* 114 */       GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s8,
// CHECK-NEXT:      /* 118 */       GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s8,
// CHECK-NEXT:      /* 122 */       GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPR8_BRegClassID),
// CHECK-NEXT:      /* 127 */       GIM_CheckConstantInt8, /*MI*/1, /*Op*/2, 10,
// CHECK-NEXT:      /* 131 */       GIM_CheckIsSafeToFold, /*NumInsns*/1,
// CHECK-NEXT:      /* 133 */       // (ld:{ *:[i8] } (add:{ *:[i8] } GPR8_B:{ *:[i8] }:$src, 10:{ *:[i8] }))<<P:Predicate_unindexedload>><<P:Predicate_load>>  =>  (LOAD8Imm_B:{ *:[i8] } GPR8_B:{ *:[i8] }:$src)
// CHECK-NEXT:      /* 133 */       GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::LOAD8Imm_B),
// CHECK-NEXT:      /* 136 */       GIR_RootToRootCopy0, // DstI[dst]
// CHECK-NEXT:      /* 137 */       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
// CHECK-NEXT:      /* 141 */       GIR_MergeMemOperands, /*InsnID*/0, /*NumInsns*/2, /*MergeInsnID's*/0, 1,
// CHECK-NEXT:      /* 146 */       GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT:      /* 147 */       // GIR_Coverage, 3,
// CHECK-NEXT:      /* 147 */       GIR_EraseRootFromParent_Done,
// CHECK-NEXT:      /* 148 */     // Label 6: @148
// CHECK-NEXT:      /* 148 */     GIM_Try, /*On fail goto*//*Label 7*/ GIMT_Encode4(170), // Rule ID 0 //
// CHECK-NEXT:      /* 153 */       GIM_RootCheckRegBankForClass0, /*RC*/GIMT_Encode2(MyTarget::GPR8_ARegClassID),
// CHECK-NEXT:      /* 156 */       // MIs[0] src
// CHECK-NEXT:      /* 156 */       GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/8,
// CHECK-NEXT:      /* 160 */       GIM_RootCheckRegBankForClass1, /*RC*/GIMT_Encode2(MyTarget::GPR8_ARegClassID),
// CHECK-NEXT:      /* 163 */       // (ld:{ *:[i8] } GPR8_A:{ *:[i8] }:$src)<<P:Predicate_unindexedload>><<P:Predicate_load>>  =>  (LOAD8_A:{ *:[i8] } GPR8_A:{ *:[i8] }:$src)
// CHECK-NEXT:      /* 163 */       GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::LOAD8_A),
// CHECK-NEXT:      /* 168 */       GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT:      /* 169 */       // GIR_Coverage, 0,
// CHECK-NEXT:      /* 169 */       GIR_Done,
// CHECK-NEXT:      /* 170 */     // Label 7: @170
// CHECK-NEXT:      /* 170 */     GIM_Try, /*On fail goto*//*Label 8*/ GIMT_Encode4(192), // Rule ID 1 //
// CHECK-NEXT:      /* 175 */       GIM_RootCheckRegBankForClass0, /*RC*/GIMT_Encode2(MyTarget::GPR8_BRegClassID),
// CHECK-NEXT:      /* 178 */       // MIs[0] src
// CHECK-NEXT:      /* 178 */       GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/8,
// CHECK-NEXT:      /* 182 */       GIM_RootCheckRegBankForClass1, /*RC*/GIMT_Encode2(MyTarget::GPR8_BRegClassID),
// CHECK-NEXT:      /* 185 */       // (ld:{ *:[i8] } GPR8_B:{ *:[i8] }:$src)<<P:Predicate_unindexedload>><<P:Predicate_load>>  =>  (LOAD8_B:{ *:[i8] } GPR8_B:{ *:[i8] }:$src)
// CHECK-NEXT:      /* 185 */       GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::LOAD8_B),
// CHECK-NEXT:      /* 190 */       GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT:      /* 191 */       // GIR_Coverage, 1,
// CHECK-NEXT:      /* 191 */       GIR_Done,
// CHECK-NEXT:      /* 192 */     // Label 8: @192
// CHECK-NEXT:      /* 192 */     GIM_Reject,
// CHECK-NEXT:      /* 193 */   // Label 4: @193
// CHECK-NEXT:      /* 193 */   GIM_Reject,
// CHECK-NEXT:      /* 194 */   // Label 2: @194
// CHECK-NEXT:      /* 194 */   GIM_Try, /*On fail goto*//*Label 9*/ GIMT_Encode4(359),
// CHECK-NEXT:      /* 199 */     GIM_CheckAtomicOrdering, /*MI*/0, /*Order*/(uint8_t)AtomicOrdering::NotAtomic,
// CHECK-NEXT:      /* 202 */     GIM_CheckMemorySizeEqualToLLT, /*MI*/0, /*MMO*/0, /*OpIdx*/0,
// CHECK-NEXT:      /* 206 */     GIM_Try, /*On fail goto*//*Label 10*/ GIMT_Encode4(260), // Rule ID 6 //
// CHECK-NEXT:      /* 211 */       GIM_RootCheckRegBankForClass0, /*RC*/GIMT_Encode2(MyTarget::GPR16_ARegClassID),
// CHECK-NEXT:      /* 214 */       // MIs[0] Operand 1
// CHECK-NEXT:      /* 214 */       GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/16,
// CHECK-NEXT:      /* 218 */       GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK-NEXT:      /* 222 */       GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_ADD),
// CHECK-NEXT:      /* 226 */       GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s16,
// CHECK-NEXT:      /* 230 */       GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s16,
// CHECK-NEXT:      /* 234 */       GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPR16_ARegClassID),
// CHECK-NEXT:      /* 239 */       GIM_CheckConstantInt8, /*MI*/1, /*Op*/2, 10,
// CHECK-NEXT:      /* 243 */       GIM_CheckIsSafeToFold, /*NumInsns*/1,
// CHECK-NEXT:      /* 245 */       // (ld:{ *:[i16] } (add:{ *:[i16] } GPR16_A:{ *:[i16] }:$src, 10:{ *:[i16] }))<<P:Predicate_unindexedload>><<P:Predicate_load>>  =>  (LOAD16Imm_A:{ *:[i16] } GPR16_A:{ *:[i16] }:$src)
// CHECK-NEXT:      /* 245 */       GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::LOAD16Imm_A),
// CHECK-NEXT:      /* 248 */       GIR_RootToRootCopy0, // DstI[dst]
// CHECK-NEXT:      /* 249 */       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
// CHECK-NEXT:      /* 253 */       GIR_MergeMemOperands, /*InsnID*/0, /*NumInsns*/2, /*MergeInsnID's*/0, 1,
// CHECK-NEXT:      /* 258 */       GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT:      /* 259 */       // GIR_Coverage, 6,
// CHECK-NEXT:      /* 259 */       GIR_EraseRootFromParent_Done,
// CHECK-NEXT:      /* 260 */     // Label 10: @260
// CHECK-NEXT:      /* 260 */     GIM_Try, /*On fail goto*//*Label 11*/ GIMT_Encode4(314), // Rule ID 7 //
// CHECK-NEXT:      /* 265 */       GIM_RootCheckRegBankForClass0, /*RC*/GIMT_Encode2(MyTarget::GPR16_BRegClassID),
// CHECK-NEXT:      /* 268 */       // MIs[0] Operand 1
// CHECK-NEXT:      /* 268 */       GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/16,
// CHECK-NEXT:      /* 272 */       GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK-NEXT:      /* 276 */       GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_ADD),
// CHECK-NEXT:      /* 280 */       GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s16,
// CHECK-NEXT:      /* 284 */       GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s16,
// CHECK-NEXT:      /* 288 */       GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPR16_BRegClassID),
// CHECK-NEXT:      /* 293 */       GIM_CheckConstantInt8, /*MI*/1, /*Op*/2, 10,
// CHECK-NEXT:      /* 297 */       GIM_CheckIsSafeToFold, /*NumInsns*/1,
// CHECK-NEXT:      /* 299 */       // (ld:{ *:[i16] } (add:{ *:[i16] } GPR16_B:{ *:[i16] }:$src, 10:{ *:[i16] }))<<P:Predicate_unindexedload>><<P:Predicate_load>>  =>  (LOAD16Imm_B:{ *:[i16] } GPR16_B:{ *:[i16] }:$src)
// CHECK-NEXT:      /* 299 */       GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::LOAD16Imm_B),
// CHECK-NEXT:      /* 302 */       GIR_RootToRootCopy0, // DstI[dst]
// CHECK-NEXT:      /* 303 */       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
// CHECK-NEXT:      /* 307 */       GIR_MergeMemOperands, /*InsnID*/0, /*NumInsns*/2, /*MergeInsnID's*/0, 1,
// CHECK-NEXT:      /* 312 */       GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT:      /* 313 */       // GIR_Coverage, 7,
// CHECK-NEXT:      /* 313 */       GIR_EraseRootFromParent_Done,
// CHECK-NEXT:      /* 314 */     // Label 11: @314
// CHECK-NEXT:      /* 314 */     GIM_Try, /*On fail goto*//*Label 12*/ GIMT_Encode4(336), // Rule ID 4 //
// CHECK-NEXT:      /* 319 */       GIM_RootCheckRegBankForClass0, /*RC*/GIMT_Encode2(MyTarget::GPR16_ARegClassID),
// CHECK-NEXT:      /* 322 */       // MIs[0] src
// CHECK-NEXT:      /* 322 */       GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/16,
// CHECK-NEXT:      /* 326 */       GIM_RootCheckRegBankForClass1, /*RC*/GIMT_Encode2(MyTarget::GPR16_ARegClassID),
// CHECK-NEXT:      /* 329 */       // (ld:{ *:[i16] } GPR16_A:{ *:[i16] }:$src)<<P:Predicate_unindexedload>><<P:Predicate_load>>  =>  (LOAD16_A:{ *:[i16] } GPR16_A:{ *:[i16] }:$src)
// CHECK-NEXT:      /* 329 */       GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::LOAD16_A),
// CHECK-NEXT:      /* 334 */       GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT:      /* 335 */       // GIR_Coverage, 4,
// CHECK-NEXT:      /* 335 */       GIR_Done,
// CHECK-NEXT:      /* 336 */     // Label 12: @336
// CHECK-NEXT:      /* 336 */     GIM_Try, /*On fail goto*//*Label 13*/ GIMT_Encode4(358), // Rule ID 5 //
// CHECK-NEXT:      /* 341 */       GIM_RootCheckRegBankForClass0, /*RC*/GIMT_Encode2(MyTarget::GPR16_BRegClassID),
// CHECK-NEXT:      /* 344 */       // MIs[0] src
// CHECK-NEXT:      /* 344 */       GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/16,
// CHECK-NEXT:      /* 348 */       GIM_RootCheckRegBankForClass1, /*RC*/GIMT_Encode2(MyTarget::GPR16_BRegClassID),
// CHECK-NEXT:      /* 351 */       // (ld:{ *:[i16] } GPR16_B:{ *:[i16] }:$src)<<P:Predicate_unindexedload>><<P:Predicate_load>>  =>  (LOAD16_B:{ *:[i16] } GPR16_B:{ *:[i16] }:$src)
// CHECK-NEXT:      /* 351 */       GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::LOAD16_B),
// CHECK-NEXT:      /* 356 */       GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT:      /* 357 */       // GIR_Coverage, 5,
// CHECK-NEXT:      /* 357 */       GIR_Done,
// CHECK-NEXT:      /* 358 */     // Label 13: @358
// CHECK-NEXT:      /* 358 */     GIM_Reject,
// CHECK-NEXT:      /* 359 */   // Label 9: @359
// CHECK-NEXT:      /* 359 */   GIM_Reject,
// CHECK-NEXT:      /* 360 */   // Label 3: @360
// CHECK-NEXT:      /* 360 */   GIM_Reject,
// CHECK-NEXT:      /* 361 */ // Label 0: @361
// CHECK-NEXT:      /* 361 */ GIM_Reject,
// CHECK-NEXT:      /* 362 */ }; // Size: 362 bytes
// CHECK-NEXT:   return MatchTable0;
// CHECK-NEXT: }
