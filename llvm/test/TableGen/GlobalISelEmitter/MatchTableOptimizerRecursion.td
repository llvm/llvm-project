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
// CHECK-NEXT:      /*   0 */ GIM_Try, /*On fail goto*//*Label 0*/ GIMT_Encode4(377),
// CHECK-NEXT:      /*   5 */   GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_LOAD),
// CHECK-NEXT:      /*   9 */   GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/GIMT_Encode2(0), GIMT_Encode2(2), /*)*//*default:*//*Label 3*/ GIMT_Encode4(376),
// CHECK-NEXT:      /*  20 */   /*GILLT_s8*//*Label 1*/ GIMT_Encode4(28),
// CHECK-NEXT:      /*  24 */   /*GILLT_s16*//*Label 2*/ GIMT_Encode4(202),
// CHECK-NEXT:      /*  28 */   // Label 1: @28
// CHECK-NEXT:      /*  28 */   GIM_Try, /*On fail goto*//*Label 4*/ GIMT_Encode4(201),
// CHECK-NEXT:      /*  33 */     GIM_CheckAtomicOrdering, /*MI*/0, /*Order*/(uint8_t)AtomicOrdering::NotAtomic,
// CHECK-NEXT:      /*  36 */     GIM_CheckMemorySizeEqualToLLT, /*MI*/0, /*MMO*/0, /*OpIdx*/0,
// CHECK-NEXT:      /*  40 */     GIM_Try, /*On fail goto*//*Label 5*/ GIMT_Encode4(96), // Rule ID 2 //
// CHECK-NEXT:      /*  45 */       GIM_RootCheckRegBankForClass, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPR8_ARegClassID),
// CHECK-NEXT:      /*  49 */       // MIs[0] Operand 1
// CHECK-NEXT:      /*  49 */       GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/8,
// CHECK-NEXT:      /*  53 */       GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK-NEXT:      /*  57 */       GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_ADD),
// CHECK-NEXT:      /*  61 */       GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s8,
// CHECK-NEXT:      /*  65 */       GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s8,
// CHECK-NEXT:      /*  69 */       GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPR8_ARegClassID),
// CHECK-NEXT:      /*  74 */       GIM_CheckConstantInt8, /*MI*/1, /*Op*/2, 10,
// CHECK-NEXT:      /*  78 */       GIM_CheckIsSafeToFold, /*NumInsns*/1,
// CHECK-NEXT:      /*  80 */       // (ld:{ *:[i8] } (add:{ *:[i8] } GPR8_A:{ *:[i8] }:$src, 10:{ *:[i8] }))<<P:Predicate_unindexedload>><<P:Predicate_load>>  =>  (LOAD8Imm_A:{ *:[i8] } GPR8_A:{ *:[i8] }:$src)
// CHECK-NEXT:      /*  80 */       GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::LOAD8Imm_A),
// CHECK-NEXT:      /*  83 */       GIR_RootToRootCopy, /*OpIdx*/0, // DstI[dst]
// CHECK-NEXT:      /*  85 */       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
// CHECK-NEXT:      /*  89 */       GIR_MergeMemOperands, /*InsnID*/0, /*NumInsns*/2, /*MergeInsnID's*/0, 1,
// CHECK-NEXT:      /*  94 */       GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT:      /*  95 */       // GIR_Coverage, 2,
// CHECK-NEXT:      /*  95 */       GIR_EraseRootFromParent_Done,
// CHECK-NEXT:      /*  96 */     // Label 5: @96
// CHECK-NEXT:      /*  96 */     GIM_Try, /*On fail goto*//*Label 6*/ GIMT_Encode4(152), // Rule ID 3 //
// CHECK-NEXT:      /* 101 */       GIM_RootCheckRegBankForClass, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPR8_BRegClassID),
// CHECK-NEXT:      /* 105 */       // MIs[0] Operand 1
// CHECK-NEXT:      /* 105 */       GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/8,
// CHECK-NEXT:      /* 109 */       GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK-NEXT:      /* 113 */       GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_ADD),
// CHECK-NEXT:      /* 117 */       GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s8,
// CHECK-NEXT:      /* 121 */       GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s8,
// CHECK-NEXT:      /* 125 */       GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPR8_BRegClassID),
// CHECK-NEXT:      /* 130 */       GIM_CheckConstantInt8, /*MI*/1, /*Op*/2, 10,
// CHECK-NEXT:      /* 134 */       GIM_CheckIsSafeToFold, /*NumInsns*/1,
// CHECK-NEXT:      /* 136 */       // (ld:{ *:[i8] } (add:{ *:[i8] } GPR8_B:{ *:[i8] }:$src, 10:{ *:[i8] }))<<P:Predicate_unindexedload>><<P:Predicate_load>>  =>  (LOAD8Imm_B:{ *:[i8] } GPR8_B:{ *:[i8] }:$src)
// CHECK-NEXT:      /* 136 */       GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::LOAD8Imm_B),
// CHECK-NEXT:      /* 139 */       GIR_RootToRootCopy, /*OpIdx*/0, // DstI[dst]
// CHECK-NEXT:      /* 141 */       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
// CHECK-NEXT:      /* 145 */       GIR_MergeMemOperands, /*InsnID*/0, /*NumInsns*/2, /*MergeInsnID's*/0, 1,
// CHECK-NEXT:      /* 150 */       GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT:      /* 151 */       // GIR_Coverage, 3,
// CHECK-NEXT:      /* 151 */       GIR_EraseRootFromParent_Done,
// CHECK-NEXT:      /* 152 */     // Label 6: @152
// CHECK-NEXT:      /* 152 */     GIM_Try, /*On fail goto*//*Label 7*/ GIMT_Encode4(176), // Rule ID 0 //
// CHECK-NEXT:      /* 157 */       GIM_RootCheckRegBankForClass, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPR8_ARegClassID),
// CHECK-NEXT:      /* 161 */       // MIs[0] src
// CHECK-NEXT:      /* 161 */       GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/8,
// CHECK-NEXT:      /* 165 */       GIM_RootCheckRegBankForClass, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPR8_ARegClassID),
// CHECK-NEXT:      /* 169 */       // (ld:{ *:[i8] } GPR8_A:{ *:[i8] }:$src)<<P:Predicate_unindexedload>><<P:Predicate_load>>  =>  (LOAD8_A:{ *:[i8] } GPR8_A:{ *:[i8] }:$src)
// CHECK-NEXT:      /* 169 */       GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::LOAD8_A),
// CHECK-NEXT:      /* 174 */       GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT:      /* 175 */       // GIR_Coverage, 0,
// CHECK-NEXT:      /* 175 */       GIR_Done,
// CHECK-NEXT:      /* 176 */     // Label 7: @176
// CHECK-NEXT:      /* 176 */     GIM_Try, /*On fail goto*//*Label 8*/ GIMT_Encode4(200), // Rule ID 1 //
// CHECK-NEXT:      /* 181 */       GIM_RootCheckRegBankForClass, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPR8_BRegClassID),
// CHECK-NEXT:      /* 185 */       // MIs[0] src
// CHECK-NEXT:      /* 185 */       GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/8,
// CHECK-NEXT:      /* 189 */       GIM_RootCheckRegBankForClass, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPR8_BRegClassID),
// CHECK-NEXT:      /* 193 */       // (ld:{ *:[i8] } GPR8_B:{ *:[i8] }:$src)<<P:Predicate_unindexedload>><<P:Predicate_load>>  =>  (LOAD8_B:{ *:[i8] } GPR8_B:{ *:[i8] }:$src)
// CHECK-NEXT:      /* 193 */       GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::LOAD8_B),
// CHECK-NEXT:      /* 198 */       GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT:      /* 199 */       // GIR_Coverage, 1,
// CHECK-NEXT:      /* 199 */       GIR_Done,
// CHECK-NEXT:      /* 200 */     // Label 8: @200
// CHECK-NEXT:      /* 200 */     GIM_Reject,
// CHECK-NEXT:      /* 201 */   // Label 4: @201
// CHECK-NEXT:      /* 201 */   GIM_Reject,
// CHECK-NEXT:      /* 202 */   // Label 2: @202
// CHECK-NEXT:      /* 202 */   GIM_Try, /*On fail goto*//*Label 9*/ GIMT_Encode4(375),
// CHECK-NEXT:      /* 207 */     GIM_CheckAtomicOrdering, /*MI*/0, /*Order*/(uint8_t)AtomicOrdering::NotAtomic,
// CHECK-NEXT:      /* 210 */     GIM_CheckMemorySizeEqualToLLT, /*MI*/0, /*MMO*/0, /*OpIdx*/0,
// CHECK-NEXT:      /* 214 */     GIM_Try, /*On fail goto*//*Label 10*/ GIMT_Encode4(270), // Rule ID 6 //
// CHECK-NEXT:      /* 219 */       GIM_RootCheckRegBankForClass, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPR16_ARegClassID),
// CHECK-NEXT:      /* 223 */       // MIs[0] Operand 1
// CHECK-NEXT:      /* 223 */       GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/16,
// CHECK-NEXT:      /* 227 */       GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK-NEXT:      /* 231 */       GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_ADD),
// CHECK-NEXT:      /* 235 */       GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s16,
// CHECK-NEXT:      /* 239 */       GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s16,
// CHECK-NEXT:      /* 243 */       GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPR16_ARegClassID),
// CHECK-NEXT:      /* 248 */       GIM_CheckConstantInt8, /*MI*/1, /*Op*/2, 10,
// CHECK-NEXT:      /* 252 */       GIM_CheckIsSafeToFold, /*NumInsns*/1,
// CHECK-NEXT:      /* 254 */       // (ld:{ *:[i16] } (add:{ *:[i16] } GPR16_A:{ *:[i16] }:$src, 10:{ *:[i16] }))<<P:Predicate_unindexedload>><<P:Predicate_load>>  =>  (LOAD16Imm_A:{ *:[i16] } GPR16_A:{ *:[i16] }:$src)
// CHECK-NEXT:      /* 254 */       GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::LOAD16Imm_A),
// CHECK-NEXT:      /* 257 */       GIR_RootToRootCopy, /*OpIdx*/0, // DstI[dst]
// CHECK-NEXT:      /* 259 */       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
// CHECK-NEXT:      /* 263 */       GIR_MergeMemOperands, /*InsnID*/0, /*NumInsns*/2, /*MergeInsnID's*/0, 1,
// CHECK-NEXT:      /* 268 */       GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT:      /* 269 */       // GIR_Coverage, 6,
// CHECK-NEXT:      /* 269 */       GIR_EraseRootFromParent_Done,
// CHECK-NEXT:      /* 270 */     // Label 10: @270
// CHECK-NEXT:      /* 270 */     GIM_Try, /*On fail goto*//*Label 11*/ GIMT_Encode4(326), // Rule ID 7 //
// CHECK-NEXT:      /* 275 */       GIM_RootCheckRegBankForClass, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPR16_BRegClassID),
// CHECK-NEXT:      /* 279 */       // MIs[0] Operand 1
// CHECK-NEXT:      /* 279 */       GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/16,
// CHECK-NEXT:      /* 283 */       GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK-NEXT:      /* 287 */       GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_ADD),
// CHECK-NEXT:      /* 291 */       GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s16,
// CHECK-NEXT:      /* 295 */       GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s16,
// CHECK-NEXT:      /* 299 */       GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPR16_BRegClassID),
// CHECK-NEXT:      /* 304 */       GIM_CheckConstantInt8, /*MI*/1, /*Op*/2, 10,
// CHECK-NEXT:      /* 308 */       GIM_CheckIsSafeToFold, /*NumInsns*/1,
// CHECK-NEXT:      /* 310 */       // (ld:{ *:[i16] } (add:{ *:[i16] } GPR16_B:{ *:[i16] }:$src, 10:{ *:[i16] }))<<P:Predicate_unindexedload>><<P:Predicate_load>>  =>  (LOAD16Imm_B:{ *:[i16] } GPR16_B:{ *:[i16] }:$src)
// CHECK-NEXT:      /* 310 */       GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::LOAD16Imm_B),
// CHECK-NEXT:      /* 313 */       GIR_RootToRootCopy, /*OpIdx*/0, // DstI[dst]
// CHECK-NEXT:      /* 315 */       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
// CHECK-NEXT:      /* 319 */       GIR_MergeMemOperands, /*InsnID*/0, /*NumInsns*/2, /*MergeInsnID's*/0, 1,
// CHECK-NEXT:      /* 324 */       GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT:      /* 325 */       // GIR_Coverage, 7,
// CHECK-NEXT:      /* 325 */       GIR_EraseRootFromParent_Done,
// CHECK-NEXT:      /* 326 */     // Label 11: @326
// CHECK-NEXT:      /* 326 */     GIM_Try, /*On fail goto*//*Label 12*/ GIMT_Encode4(350), // Rule ID 4 //
// CHECK-NEXT:      /* 331 */       GIM_RootCheckRegBankForClass, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPR16_ARegClassID),
// CHECK-NEXT:      /* 335 */       // MIs[0] src
// CHECK-NEXT:      /* 335 */       GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/16,
// CHECK-NEXT:      /* 339 */       GIM_RootCheckRegBankForClass, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPR16_ARegClassID),
// CHECK-NEXT:      /* 343 */       // (ld:{ *:[i16] } GPR16_A:{ *:[i16] }:$src)<<P:Predicate_unindexedload>><<P:Predicate_load>>  =>  (LOAD16_A:{ *:[i16] } GPR16_A:{ *:[i16] }:$src)
// CHECK-NEXT:      /* 343 */       GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::LOAD16_A),
// CHECK-NEXT:      /* 348 */       GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT:      /* 349 */       // GIR_Coverage, 4,
// CHECK-NEXT:      /* 349 */       GIR_Done,
// CHECK-NEXT:      /* 350 */     // Label 12: @350
// CHECK-NEXT:      /* 350 */     GIM_Try, /*On fail goto*//*Label 13*/ GIMT_Encode4(374), // Rule ID 5 //
// CHECK-NEXT:      /* 355 */       GIM_RootCheckRegBankForClass, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPR16_BRegClassID),
// CHECK-NEXT:      /* 359 */       // MIs[0] src
// CHECK-NEXT:      /* 359 */       GIM_CheckPointerToAny, /*MI*/0, /*Op*/1, /*SizeInBits*/16,
// CHECK-NEXT:      /* 363 */       GIM_RootCheckRegBankForClass, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPR16_BRegClassID),
// CHECK-NEXT:      /* 367 */       // (ld:{ *:[i16] } GPR16_B:{ *:[i16] }:$src)<<P:Predicate_unindexedload>><<P:Predicate_load>>  =>  (LOAD16_B:{ *:[i16] } GPR16_B:{ *:[i16] }:$src)
// CHECK-NEXT:      /* 367 */       GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::LOAD16_B),
// CHECK-NEXT:      /* 372 */       GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT:      /* 373 */       // GIR_Coverage, 5,
// CHECK-NEXT:      /* 373 */       GIR_Done,
// CHECK-NEXT:      /* 374 */     // Label 13: @374
// CHECK-NEXT:      /* 374 */     GIM_Reject,
// CHECK-NEXT:      /* 375 */   // Label 9: @375
// CHECK-NEXT:      /* 375 */   GIM_Reject,
// CHECK-NEXT:      /* 376 */   // Label 3: @376
// CHECK-NEXT:      /* 376 */   GIM_Reject,
// CHECK-NEXT:      /* 377 */ // Label 0: @377
// CHECK-NEXT:      /* 377 */ GIM_Reject,
// CHECK-NEXT:      /* 378 */ }; // Size: 378 bytes
// CHECK-NEXT:   return MatchTable0;
// CHECK-NEXT: }
