; RUN: llc -O0 -aarch64-enable-atomic-cfg-tidy=0 -stop-after=irtranslator -global-isel -verify-machineinstrs %s -o - 2>&1 | FileCheck %s

; This file checks that the translation from llvm IR to generic MachineInstr
; is correct.
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--"

; Tests for add.
; CHECK-LABEL: name: addi64
; CHECK:      [[ARG1:%[0-9]+]]:_(s64) = COPY %x0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s64) = COPY %x1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s64) = G_ADD [[ARG1]], [[ARG2]]
; CHECK-NEXT: %x0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %x0
define i64 @addi64(i64 %arg1, i64 %arg2) {
  %res = add i64 %arg1, %arg2
  ret i64 %res
}

; CHECK-LABEL: name: muli64
; CHECK: [[ARG1:%[0-9]+]]:_(s64) = COPY %x0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s64) = COPY %x1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s64) = G_MUL [[ARG1]], [[ARG2]]
; CHECK-NEXT: %x0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %x0
define i64 @muli64(i64 %arg1, i64 %arg2) {
  %res = mul i64 %arg1, %arg2
  ret i64 %res
}

; Tests for alloca
; CHECK-LABEL: name: allocai64
; CHECK: stack:
; CHECK-NEXT:   - { id: 0, name: ptr1, type: default, offset: 0, size: 8, alignment: 8,
; CHECK-NEXT:       stack-id: 0, callee-saved-register: '', callee-saved-restored: true,
; CHECK-NEXT: di-variable: '', di-expression: '', di-location: '' }
; CHECK-NEXT:   - { id: 1, name: ptr2, type: default, offset: 0, size: 8, alignment: 1,
; CHECK-NEXT:       stack-id: 0, callee-saved-register: '', callee-saved-restored: true,
; CHECK-NEXT:       di-variable: '', di-expression: '', di-location: '' }
; CHECK-NEXT:   - { id: 2, name: ptr3, type: default, offset: 0, size: 128, alignment: 8,
; CHECK-NEXT:       stack-id: 0, callee-saved-register: '', callee-saved-restored: true,
; CHECK-NEXT:       di-variable: '', di-expression: '', di-location: '' }
; CHECK-NEXT:   - { id: 3, name: ptr4, type: default, offset: 0, size: 1, alignment: 8,
; CHECK: %{{[0-9]+}}:_(p0) = G_FRAME_INDEX %stack.0.ptr1
; CHECK: %{{[0-9]+}}:_(p0) = G_FRAME_INDEX %stack.1.ptr2
; CHECK: %{{[0-9]+}}:_(p0) = G_FRAME_INDEX %stack.2.ptr3
; CHECK: %{{[0-9]+}}:_(p0) = G_FRAME_INDEX %stack.3.ptr4
define void @allocai64() {
  %ptr1 = alloca i64
  %ptr2 = alloca i64, align 1
  %ptr3 = alloca i64, i32 16
  %ptr4 = alloca [0 x i64]
  ret void
}

; Tests for br.
; CHECK-LABEL: name: uncondbr
; CHECK: body:
;
; ABI/constant lowering and IR-level entry basic block.
; CHECK: bb.{{[0-9]+}}.{{[a-zA-Z0-9.]+}}:
;
; Make sure we have one successor and only one.
; CHECK-NEXT: successors: %[[BB2:bb.[0-9]+]](0x80000000)
;
; Check that we emit the correct branch.
; CHECK: G_BR %[[BB2]]
;
; Check that end contains the return instruction.
; CHECK: [[END:bb.[0-9]+]].{{[a-zA-Z0-9.]+}}:
; CHECK-NEXT: RET_ReallyLR
;
; CHECK: bb.{{[0-9]+}}.{{[a-zA-Z0-9.]+}}:
; CHECK-NEXT: successors: %[[END]](0x80000000)
; CHECK: G_BR %[[END]]
define void @uncondbr() {
entry:
  br label %bb2
end:
  ret void
bb2:
  br label %end
}

; CHECK-LABEL: name: uncondbr_fallthrough
; CHECK: body:
; CHECK: bb.{{[0-9]+}}.{{[a-zA-Z0-9.]+}}:
; CHECK-NEXT: successors: %[[END:bb.[0-9]+]](0x80000000)
; We don't emit a branch here, as we can fallthrough to the successor.
; CHECK-NOT: G_BR
; CHECK: [[END]].{{[a-zA-Z0-9.]+}}:
; CHECK-NEXT: RET_ReallyLR
define void @uncondbr_fallthrough() {
entry:
  br label %end
end:
  ret void
}

; Tests for conditional br.
; CHECK-LABEL: name: condbr
; CHECK: body:
;
; ABI/constant lowering and IR-level entry basic block.
; CHECK: bb.{{[0-9]+}} (%ir-block.{{[0-9]+}}):
; Make sure we have two successors
; CHECK-NEXT: successors: %[[TRUE:bb.[0-9]+]](0x40000000),
; CHECK:                  %[[FALSE:bb.[0-9]+]](0x40000000)
;
; CHECK: [[ADDR:%.*]]:_(p0) = COPY %x0
;
; Check that we emit the correct branch.
; CHECK: [[TST:%.*]]:_(s1) = G_LOAD [[ADDR]](p0)
; CHECK: G_BRCOND [[TST]](s1), %[[TRUE]]
; CHECK: G_BR %[[FALSE]]
;
; Check that each successor contains the return instruction.
; CHECK: [[TRUE]].{{[a-zA-Z0-9.]+}}:
; CHECK-NEXT: RET_ReallyLR
; CHECK: [[FALSE]].{{[a-zA-Z0-9.]+}}:
; CHECK-NEXT: RET_ReallyLR
define void @condbr(i1* %tstaddr) {
  %tst = load i1, i1* %tstaddr
  br i1 %tst, label %true, label %false
true:
  ret void
false:
  ret void
}

; Tests for switch.
; This gets lowered to a very straightforward sequence of comparisons for now.
; CHECK-LABEL: name: switch
; CHECK: body:
;
; CHECK: bb.{{[a-zA-Z0-9.]+}}:
; CHECK-NEXT: successors: %[[BB_CASE100:bb.[0-9]+]](0x40000000), %[[BB_NOTCASE100_CHECKNEXT:bb.[0-9]+]](0x40000000)
; CHECK: %0:_(s32) = COPY %w0
; CHECK: %[[reg100:[0-9]+]]:_(s32) = G_CONSTANT i32 100
; CHECK: %[[reg200:[0-9]+]]:_(s32) = G_CONSTANT i32 200
; CHECK: %[[reg0:[0-9]+]]:_(s32) = G_CONSTANT i32 0
; CHECK: %[[reg1:[0-9]+]]:_(s32) = G_CONSTANT i32 1
; CHECK: %[[reg2:[0-9]+]]:_(s32) = G_CONSTANT i32 2
; CHECK: %[[regicmp100:[0-9]+]]:_(s1) = G_ICMP intpred(eq), %[[reg100]](s32), %0
; CHECK: G_BRCOND %[[regicmp100]](s1), %[[BB_CASE100]]
; CHECK: G_BR %[[BB_NOTCASE100_CHECKNEXT]]
;
; CHECK: [[BB_NOTCASE100_CHECKNEXT]].{{[a-zA-Z0-9.]+}}:
; CHECK-NEXT: successors: %[[BB_CASE200:bb.[0-9]+]](0x40000000), %[[BB_NOTCASE200_CHECKNEXT:bb.[0-9]+]](0x40000000)
; CHECK: %[[regicmp200:[0-9]+]]:_(s1) = G_ICMP intpred(eq), %[[reg200]](s32), %0
; CHECK: G_BRCOND %[[regicmp200]](s1), %[[BB_CASE200]]
; CHECK: G_BR %[[BB_NOTCASE200_CHECKNEXT]]
;
; CHECK: [[BB_NOTCASE200_CHECKNEXT]].{{[a-zA-Z0-9.]+}}:
; CHECK-NEXT: successors: %[[BB_DEFAULT:bb.[0-9]+]](0x80000000)
; CHECK: G_BR %[[BB_DEFAULT]]
;
; CHECK: [[BB_DEFAULT]].{{[a-zA-Z0-9.]+}}:
; CHECK-NEXT: successors: %[[BB_RET:bb.[0-9]+]](0x80000000)
; CHECK: %[[regretdefault:[0-9]+]]:_(s32) = G_ADD %0, %[[reg0]]
; CHECK: G_BR %[[BB_RET]]
;
; CHECK: [[BB_CASE100]].{{[a-zA-Z0-9.]+}}:
; CHECK-NEXT: successors: %[[BB_RET:bb.[0-9]+]](0x80000000)
; CHECK: %[[regretc100:[0-9]+]]:_(s32) = G_ADD %0, %[[reg1]]
; CHECK: G_BR %[[BB_RET]]
;
; CHECK: [[BB_CASE200]].{{[a-zA-Z0-9.]+}}:
; CHECK-NEXT: successors: %[[BB_RET]](0x80000000)
; CHECK: %[[regretc200:[0-9]+]]:_(s32) = G_ADD %0, %[[reg2]]
;
; CHECK: [[BB_RET]].{{[a-zA-Z0-9.]+}}:
; CHECK-NEXT: %[[regret:[0-9]+]]:_(s32) = G_PHI %[[regretdefault]](s32), %[[BB_DEFAULT]], %[[regretc100]](s32), %[[BB_CASE100]]
; CHECK:  %w0 = COPY %[[regret]](s32)
; CHECK:  RET_ReallyLR implicit %w0
;
define i32 @switch(i32 %argc) {
entry:
  switch i32 %argc, label %default [
    i32 100, label %case100
    i32 200, label %case200
  ]

default:
  %tmp0 = add i32 %argc, 0
  br label %return

case100:
  %tmp1 = add i32 %argc, 1
  br label %return

case200:
  %tmp2 = add i32 %argc, 2
  br label %return

return:
  %res = phi i32 [ %tmp0, %default ], [ %tmp1, %case100 ], [ %tmp2, %case200 ]
  ret i32 %res
}

  ; The switch lowering code changes the CFG, which means that the original
  ; %entry block is no longer a predecessor for the phi instruction. We need to
  ; use the correct lowered MachineBasicBlock instead.
; CHECK-LABEL: name: test_cfg_remap
; CHECK: bb.{{[0-9]+.[a-zA-Z0-9.]+}}:
; CHECK-NEXT: successors: %{{bb.[0-9]+}}(0x40000000), %[[NOTCASE1_BLOCK:bb.[0-9]+]](0x40000000)
; CHECK: [[NOTCASE1_BLOCK]].{{[a-zA-Z0-9.]+}}:
; CHECK-NEXT: successors: %{{bb.[0-9]+}}(0x40000000), %[[NOTCASE57_BLOCK:bb.[0-9]+]](0x40000000)
; CHECK: [[NOTCASE57_BLOCK]].{{[a-zA-Z0-9.]+}}:
; CHECK-NEXT: successors: %[[PHI_BLOCK:bb.[0-9]+]](0x80000000)
; CHECK: G_BR %[[PHI_BLOCK]]
;
; CHECK: [[PHI_BLOCK]].{{[a-zA-Z0-9.]+}}:
; CHECK-NEXT: G_PHI %{{.*}}(s32), %[[NOTCASE57_BLOCK:bb.[0-9]+]], %{{.*}}(s32),
;
define i32 @test_cfg_remap(i32 %in) {
entry:
  switch i32 %in, label %phi.block [i32 1, label %next
                                    i32 57, label %other]

next:
  br label %phi.block

other:
  ret i32 undef

phi.block:
  %res = phi i32 [1, %entry], [42, %next]
  ret i32 %res
}

; CHECK-LABEL: name: test_cfg_remap_multiple_preds
; CHECK: G_PHI [[ENTRY:%.*]](s32), %bb.{{[0-9]+}}, [[ENTRY]](s32), %bb.{{[0-9]+}}
define i32 @test_cfg_remap_multiple_preds(i32 %in) {
entry:
  switch i32 %in, label %odd [i32 1, label %next
                              i32 57, label %other
                              i32 128, label %phi.block
                              i32 256, label %phi.block]
odd:
  unreachable

next:
  br label %phi.block

other:
  ret i32 undef

phi.block:
  %res = phi i32 [1, %entry], [1, %entry], [42, %next]
  ret i32 12
}

; Tests for indirect br.
; CHECK-LABEL: name: indirectbr
; CHECK: body:
;
; ABI/constant lowering and IR-level entry basic block.
; CHECK: bb.{{[0-9]+.[a-zA-Z0-9.]+}}:
; Make sure we have one successor
; CHECK-NEXT: successors: %[[BB_L1:bb.[0-9]+]](0x80000000)
; CHECK-NOT: G_BR
;
; Check basic block L1 has 2 successors: BBL1 and BBL2
; CHECK: [[BB_L1]].{{[a-zA-Z0-9.]+}} (address-taken):
; CHECK-NEXT: successors: %[[BB_L1]](0x40000000),
; CHECK:                  %[[BB_L2:bb.[0-9]+]](0x40000000)
; CHECK: G_BRINDIRECT %{{[0-9]+}}(p0)
;
; Check basic block L2 is the return basic block
; CHECK: [[BB_L2]].{{[a-zA-Z0-9.]+}} (address-taken):
; CHECK-NEXT: RET_ReallyLR

@indirectbr.L = internal unnamed_addr constant [3 x i8*] [i8* blockaddress(@indirectbr, %L1), i8* blockaddress(@indirectbr, %L2), i8* null], align 8

define void @indirectbr() {
entry:
  br label %L1
L1:                                               ; preds = %entry, %L1
  %i = phi i32 [ 0, %entry ], [ %inc, %L1 ]
  %inc = add i32 %i, 1
  %idxprom = zext i32 %i to i64
  %arrayidx = getelementptr inbounds [3 x i8*], [3 x i8*]* @indirectbr.L, i64 0, i64 %idxprom
  %brtarget = load i8*, i8** %arrayidx, align 8
  indirectbr i8* %brtarget, [label %L1, label %L2]
L2:                                               ; preds = %L1
  ret void
}

; Tests for or.
; CHECK-LABEL: name: ori64
; CHECK: [[ARG1:%[0-9]+]]:_(s64) = COPY %x0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s64) = COPY %x1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s64) = G_OR [[ARG1]], [[ARG2]]
; CHECK-NEXT: %x0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %x0
define i64 @ori64(i64 %arg1, i64 %arg2) {
  %res = or i64 %arg1, %arg2
  ret i64 %res
}

; CHECK-LABEL: name: ori32
; CHECK: [[ARG1:%[0-9]+]]:_(s32) = COPY %w0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s32) = COPY %w1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s32) = G_OR [[ARG1]], [[ARG2]]
; CHECK-NEXT: %w0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %w0
define i32 @ori32(i32 %arg1, i32 %arg2) {
  %res = or i32 %arg1, %arg2
  ret i32 %res
}

; Tests for xor.
; CHECK-LABEL: name: xori64
; CHECK: [[ARG1:%[0-9]+]]:_(s64) = COPY %x0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s64) = COPY %x1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s64) = G_XOR [[ARG1]], [[ARG2]]
; CHECK-NEXT: %x0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %x0
define i64 @xori64(i64 %arg1, i64 %arg2) {
  %res = xor i64 %arg1, %arg2
  ret i64 %res
}

; CHECK-LABEL: name: xori32
; CHECK: [[ARG1:%[0-9]+]]:_(s32) = COPY %w0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s32) = COPY %w1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s32) = G_XOR [[ARG1]], [[ARG2]]
; CHECK-NEXT: %w0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %w0
define i32 @xori32(i32 %arg1, i32 %arg2) {
  %res = xor i32 %arg1, %arg2
  ret i32 %res
}

; Tests for and.
; CHECK-LABEL: name: andi64
; CHECK: [[ARG1:%[0-9]+]]:_(s64) = COPY %x0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s64) = COPY %x1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s64) = G_AND [[ARG1]], [[ARG2]]
; CHECK-NEXT: %x0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %x0
define i64 @andi64(i64 %arg1, i64 %arg2) {
  %res = and i64 %arg1, %arg2
  ret i64 %res
}

; CHECK-LABEL: name: andi32
; CHECK: [[ARG1:%[0-9]+]]:_(s32) = COPY %w0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s32) = COPY %w1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s32) = G_AND [[ARG1]], [[ARG2]]
; CHECK-NEXT: %w0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %w0
define i32 @andi32(i32 %arg1, i32 %arg2) {
  %res = and i32 %arg1, %arg2
  ret i32 %res
}

; Tests for sub.
; CHECK-LABEL: name: subi64
; CHECK: [[ARG1:%[0-9]+]]:_(s64) = COPY %x0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s64) = COPY %x1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s64) = G_SUB [[ARG1]], [[ARG2]]
; CHECK-NEXT: %x0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %x0
define i64 @subi64(i64 %arg1, i64 %arg2) {
  %res = sub i64 %arg1, %arg2
  ret i64 %res
}

; CHECK-LABEL: name: subi32
; CHECK: [[ARG1:%[0-9]+]]:_(s32) = COPY %w0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s32) = COPY %w1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s32) = G_SUB [[ARG1]], [[ARG2]]
; CHECK-NEXT: %w0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %w0
define i32 @subi32(i32 %arg1, i32 %arg2) {
  %res = sub i32 %arg1, %arg2
  ret i32 %res
}

; CHECK-LABEL: name: ptrtoint
; CHECK: [[ARG1:%[0-9]+]]:_(p0) = COPY %x0
; CHECK: [[RES:%[0-9]+]]:_(s64) = G_PTRTOINT [[ARG1]]
; CHECK: %x0 = COPY [[RES]]
; CHECK: RET_ReallyLR implicit %x0
define i64 @ptrtoint(i64* %a) {
  %val = ptrtoint i64* %a to i64
  ret i64 %val
}

; CHECK-LABEL: name: inttoptr
; CHECK: [[ARG1:%[0-9]+]]:_(s64) = COPY %x0
; CHECK: [[RES:%[0-9]+]]:_(p0) = G_INTTOPTR [[ARG1]]
; CHECK: %x0 = COPY [[RES]]
; CHECK: RET_ReallyLR implicit %x0
define i64* @inttoptr(i64 %a) {
  %val = inttoptr i64 %a to i64*
  ret i64* %val
}

; CHECK-LABEL: name: trivial_bitcast
; CHECK: [[ARG1:%[0-9]+]]:_(p0) = COPY %x0
; CHECK: %x0 = COPY [[ARG1]]
; CHECK: RET_ReallyLR implicit %x0
define i64* @trivial_bitcast(i8* %a) {
  %val = bitcast i8* %a to i64*
  ret i64* %val
}

; CHECK-LABEL: name: trivial_bitcast_with_copy
; CHECK:     [[A:%[0-9]+]]:_(p0) = COPY %x0
; CHECK:     G_BR %[[CAST:bb\.[0-9]+]]

; CHECK: [[END:bb\.[0-9]+]].{{[a-zA-Z0-9.]+}}:

; CHECK: [[CAST]].{{[a-zA-Z0-9.]+}}:
; CHECK:     {{%[0-9]+}}:_(p0) = COPY [[A]]
; CHECK:     G_BR %[[END]]
define i64* @trivial_bitcast_with_copy(i8* %a) {
  br label %cast

end:
  ret i64* %val

cast:
  %val = bitcast i8* %a to i64*
  br label %end
}

; CHECK-LABEL: name: bitcast
; CHECK: [[ARG1:%[0-9]+]]:_(s64) = COPY %x0
; CHECK: [[RES1:%[0-9]+]]:_(<2 x s32>) = G_BITCAST [[ARG1]]
; CHECK: [[RES2:%[0-9]+]]:_(s64) = G_BITCAST [[RES1]]
; CHECK: %x0 = COPY [[RES2]]
; CHECK: RET_ReallyLR implicit %x0
define i64 @bitcast(i64 %a) {
  %res1 = bitcast i64 %a to <2 x i32>
  %res2 = bitcast <2 x i32> %res1 to i64
  ret i64 %res2
}

; CHECK-LABEL: name: trunc
; CHECK: [[ARG1:%[0-9]+]]:_(s64) = COPY %x0
; CHECK: [[VEC:%[0-9]+]]:_(<4 x s32>) = G_LOAD
; CHECK: [[RES1:%[0-9]+]]:_(s8) = G_TRUNC [[ARG1]]
; CHECK: [[RES2:%[0-9]+]]:_(<4 x s16>) = G_TRUNC [[VEC]]
define void @trunc(i64 %a) {
  %vecptr = alloca <4 x i32>
  %vec = load <4 x i32>, <4 x i32>* %vecptr
  %res1 = trunc i64 %a to i8
  %res2 = trunc <4 x i32> %vec to <4 x i16>
  ret void
}

; CHECK-LABEL: name: load
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY %x0
; CHECK: [[ADDR42:%[0-9]+]]:_(p42) = COPY %x1
; CHECK: [[VAL1:%[0-9]+]]:_(s64) = G_LOAD [[ADDR]](p0) :: (load 8 from %ir.addr, align 16)
; CHECK: [[VAL2:%[0-9]+]]:_(s64) = G_LOAD [[ADDR42]](p42) :: (load 8 from %ir.addr42)
; CHECK: [[SUM2:%.*]]:_(s64) = G_ADD [[VAL1]], [[VAL2]]
; CHECK: [[VAL3:%[0-9]+]]:_(s64) = G_LOAD [[ADDR]](p0) :: (volatile load 8 from %ir.addr)
; CHECK: [[SUM3:%[0-9]+]]:_(s64) = G_ADD [[SUM2]], [[VAL3]]
; CHECK: %x0 = COPY [[SUM3]]
; CHECK: RET_ReallyLR implicit %x0
define i64 @load(i64* %addr, i64 addrspace(42)* %addr42) {
  %val1 = load i64, i64* %addr, align 16

  %val2 = load i64, i64 addrspace(42)* %addr42
  %sum2 = add i64 %val1, %val2

  %val3 = load volatile i64, i64* %addr
  %sum3 = add i64 %sum2, %val3
  ret i64 %sum3
}

; CHECK-LABEL: name: store
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY %x0
; CHECK: [[ADDR42:%[0-9]+]]:_(p42) = COPY %x1
; CHECK: [[VAL1:%[0-9]+]]:_(s64) = COPY %x2
; CHECK: [[VAL2:%[0-9]+]]:_(s64) = COPY %x3
; CHECK: G_STORE [[VAL1]](s64), [[ADDR]](p0) :: (store 8 into %ir.addr, align 16)
; CHECK: G_STORE [[VAL2]](s64), [[ADDR42]](p42) :: (store 8 into %ir.addr42)
; CHECK: G_STORE [[VAL1]](s64), [[ADDR]](p0) :: (volatile store 8 into %ir.addr)
; CHECK: RET_ReallyLR
define void @store(i64* %addr, i64 addrspace(42)* %addr42, i64 %val1, i64 %val2) {
  store i64 %val1, i64* %addr, align 16
  store i64 %val2, i64 addrspace(42)* %addr42
  store volatile i64 %val1, i64* %addr
  %sum = add i64 %val1, %val2
  ret void
}

; CHECK-LABEL: name: intrinsics
; CHECK: [[CUR:%[0-9]+]]:_(s32) = COPY %w0
; CHECK: [[BITS:%[0-9]+]]:_(s32) = COPY %w1
; CHECK: [[CREG:%[0-9]+]]:_(s32) = G_CONSTANT i32 0
; CHECK: [[PTR:%[0-9]+]]:_(p0) = G_INTRINSIC intrinsic(@llvm.returnaddress), [[CREG]]
; CHECK: [[PTR_VEC:%[0-9]+]]:_(p0) = G_FRAME_INDEX %stack.0.ptr.vec
; CHECK: [[VEC:%[0-9]+]]:_(<8 x s8>) = G_LOAD [[PTR_VEC]]
; CHECK: G_INTRINSIC_W_SIDE_EFFECTS intrinsic(@llvm.aarch64.neon.st2), [[VEC]](<8 x s8>), [[VEC]](<8 x s8>), [[PTR]](p0)
; CHECK: RET_ReallyLR
declare i8* @llvm.returnaddress(i32)
declare void @llvm.aarch64.neon.st2.v8i8.p0i8(<8 x i8>, <8 x i8>, i8*)
declare { <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld2.v8i8.p0v8i8(<8 x i8>*)
define void @intrinsics(i32 %cur, i32 %bits) {
  %ptr = call i8* @llvm.returnaddress(i32 0)
  %ptr.vec = alloca <8 x i8>
  %vec = load <8 x i8>, <8 x i8>* %ptr.vec
  call void @llvm.aarch64.neon.st2.v8i8.p0i8(<8 x i8> %vec, <8 x i8> %vec, i8* %ptr)
  ret void
}

; CHECK-LABEL: name: test_phi
; CHECK:     G_BRCOND {{%.*}}, %[[TRUE:bb\.[0-9]+]]
; CHECK:     G_BR %[[FALSE:bb\.[0-9]+]]

; CHECK: [[TRUE]].{{[a-zA-Z0-9.]+}}:
; CHECK:     [[RES1:%[0-9]+]]:_(s32) = G_LOAD

; CHECK: [[FALSE]].{{[a-zA-Z0-9.]+}}:
; CHECK:     [[RES2:%[0-9]+]]:_(s32) = G_LOAD

; CHECK:     [[RES:%[0-9]+]]:_(s32) = G_PHI [[RES1]](s32), %[[TRUE]], [[RES2]](s32), %[[FALSE]]
; CHECK:     %w0 = COPY [[RES]]
define i32 @test_phi(i32* %addr1, i32* %addr2, i1 %tst) {
  br i1 %tst, label %true, label %false

true:
  %res1 = load i32, i32* %addr1
  br label %end

false:
  %res2 = load i32, i32* %addr2
  br label %end

end:
  %res = phi i32 [%res1, %true], [%res2, %false]
  ret i32 %res
}

; CHECK-LABEL: name: unreachable
; CHECK: G_ADD
; CHECK-NEXT: {{^$}}
; CHECK-NEXT: ...
define void @unreachable(i32 %a) {
  %sum = add i32 %a, %a
  unreachable
}

  ; It's important that constants are after argument passing, but before the
  ; rest of the entry block.
; CHECK-LABEL: name: constant_int
; CHECK: [[IN:%[0-9]+]]:_(s32) = COPY %w0
; CHECK: [[ONE:%[0-9]+]]:_(s32) = G_CONSTANT i32 1

; CHECK: bb.{{[0-9]+}}.{{[a-zA-Z0-9.]+}}:
; CHECK: [[SUM1:%[0-9]+]]:_(s32) = G_ADD [[IN]], [[ONE]]
; CHECK: [[SUM2:%[0-9]+]]:_(s32) = G_ADD [[IN]], [[ONE]]
; CHECK: [[RES:%[0-9]+]]:_(s32) = G_ADD [[SUM1]], [[SUM2]]
; CHECK: %w0 = COPY [[RES]]

define i32 @constant_int(i32 %in) {
  br label %next

next:
  %sum1 = add i32 %in, 1
  %sum2 = add i32 %in, 1
  %res = add i32 %sum1, %sum2
  ret i32 %res
}

; CHECK-LABEL: name: constant_int_start
; CHECK: [[TWO:%[0-9]+]]:_(s32) = G_CONSTANT i32 2
; CHECK: [[ANSWER:%[0-9]+]]:_(s32) = G_CONSTANT i32 42
; CHECK: [[RES:%[0-9]+]]:_(s32) = G_ADD [[TWO]], [[ANSWER]]
define i32 @constant_int_start() {
  %res = add i32 2, 42
  ret i32 %res
}

; CHECK-LABEL: name: test_undef
; CHECK: [[UNDEF:%[0-9]+]]:_(s32) = G_IMPLICIT_DEF
; CHECK: %w0 = COPY [[UNDEF]]
define i32 @test_undef() {
  ret i32 undef
}

; CHECK-LABEL: name: test_constant_inttoptr
; CHECK: [[ONE:%[0-9]+]]:_(s64) = G_CONSTANT i64 1
; CHECK: [[PTR:%[0-9]+]]:_(p0) = G_INTTOPTR [[ONE]]
; CHECK: %x0 = COPY [[PTR]]
define i8* @test_constant_inttoptr() {
  ret i8* inttoptr(i64 1 to i8*)
}

  ; This failed purely because the Constant -> VReg map was kept across
  ; functions, so reuse the "i64 1" from above.
; CHECK-LABEL: name: test_reused_constant
; CHECK: [[ONE:%[0-9]+]]:_(s64) = G_CONSTANT i64 1
; CHECK: %x0 = COPY [[ONE]]
define i64 @test_reused_constant() {
  ret i64 1
}

; CHECK-LABEL: name: test_sext
; CHECK: [[IN:%[0-9]+]]:_(s32) = COPY %w0
; CHECK: [[RES:%[0-9]+]]:_(s64) = G_SEXT [[IN]]
; CHECK: %x0 = COPY [[RES]]
define i64 @test_sext(i32 %in) {
  %res = sext i32 %in to i64
  ret i64 %res
}

; CHECK-LABEL: name: test_zext
; CHECK: [[IN:%[0-9]+]]:_(s32) = COPY %w0
; CHECK: [[RES:%[0-9]+]]:_(s64) = G_ZEXT [[IN]]
; CHECK: %x0 = COPY [[RES]]
define i64 @test_zext(i32 %in) {
  %res = zext i32 %in to i64
  ret i64 %res
}

; CHECK-LABEL: name: test_shl
; CHECK: [[ARG1:%[0-9]+]]:_(s32) = COPY %w0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s32) = COPY %w1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s32) = G_SHL [[ARG1]], [[ARG2]]
; CHECK-NEXT: %w0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %w0
define i32 @test_shl(i32 %arg1, i32 %arg2) {
  %res = shl i32 %arg1, %arg2
  ret i32 %res
}


; CHECK-LABEL: name: test_lshr
; CHECK: [[ARG1:%[0-9]+]]:_(s32) = COPY %w0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s32) = COPY %w1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s32) = G_LSHR [[ARG1]], [[ARG2]]
; CHECK-NEXT: %w0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %w0
define i32 @test_lshr(i32 %arg1, i32 %arg2) {
  %res = lshr i32 %arg1, %arg2
  ret i32 %res
}

; CHECK-LABEL: name: test_ashr
; CHECK: [[ARG1:%[0-9]+]]:_(s32) = COPY %w0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s32) = COPY %w1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s32) = G_ASHR [[ARG1]], [[ARG2]]
; CHECK-NEXT: %w0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %w0
define i32 @test_ashr(i32 %arg1, i32 %arg2) {
  %res = ashr i32 %arg1, %arg2
  ret i32 %res
}

; CHECK-LABEL: name: test_sdiv
; CHECK: [[ARG1:%[0-9]+]]:_(s32) = COPY %w0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s32) = COPY %w1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s32) = G_SDIV [[ARG1]], [[ARG2]]
; CHECK-NEXT: %w0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %w0
define i32 @test_sdiv(i32 %arg1, i32 %arg2) {
  %res = sdiv i32 %arg1, %arg2
  ret i32 %res
}

; CHECK-LABEL: name: test_udiv
; CHECK: [[ARG1:%[0-9]+]]:_(s32) = COPY %w0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s32) = COPY %w1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s32) = G_UDIV [[ARG1]], [[ARG2]]
; CHECK-NEXT: %w0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %w0
define i32 @test_udiv(i32 %arg1, i32 %arg2) {
  %res = udiv i32 %arg1, %arg2
  ret i32 %res
}

; CHECK-LABEL: name: test_srem
; CHECK: [[ARG1:%[0-9]+]]:_(s32) = COPY %w0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s32) = COPY %w1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s32) = G_SREM [[ARG1]], [[ARG2]]
; CHECK-NEXT: %w0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %w0
define i32 @test_srem(i32 %arg1, i32 %arg2) {
  %res = srem i32 %arg1, %arg2
  ret i32 %res
}

; CHECK-LABEL: name: test_urem
; CHECK: [[ARG1:%[0-9]+]]:_(s32) = COPY %w0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s32) = COPY %w1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s32) = G_UREM [[ARG1]], [[ARG2]]
; CHECK-NEXT: %w0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %w0
define i32 @test_urem(i32 %arg1, i32 %arg2) {
  %res = urem i32 %arg1, %arg2
  ret i32 %res
}

; CHECK-LABEL: name: test_constant_null
; CHECK: [[NULL:%[0-9]+]]:_(p0) = G_CONSTANT i64 0
; CHECK: %x0 = COPY [[NULL]]
define i8* @test_constant_null() {
  ret i8* null
}

; CHECK-LABEL: name: test_struct_memops
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY %x0
; CHECK: [[VAL:%[0-9]+]]:_(s64) = G_LOAD [[ADDR]](p0) :: (load 8 from  %ir.addr, align 4)
; CHECK: G_STORE [[VAL]](s64), [[ADDR]](p0) :: (store 8 into  %ir.addr, align 4)
define void @test_struct_memops({ i8, i32 }* %addr) {
  %val = load { i8, i32 }, { i8, i32 }* %addr
  store { i8, i32 } %val, { i8, i32 }* %addr
  ret void
}

; CHECK-LABEL: name: test_i1_memops
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY %x0
; CHECK: [[VAL:%[0-9]+]]:_(s1) = G_LOAD [[ADDR]](p0) :: (load 1 from  %ir.addr)
; CHECK: G_STORE [[VAL]](s1), [[ADDR]](p0) :: (store 1 into  %ir.addr)
define void @test_i1_memops(i1* %addr) {
  %val = load i1, i1* %addr
  store i1 %val, i1* %addr
  ret void
}

; CHECK-LABEL: name: int_comparison
; CHECK: [[LHS:%[0-9]+]]:_(s32) = COPY %w0
; CHECK: [[RHS:%[0-9]+]]:_(s32) = COPY %w1
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY %x2
; CHECK: [[TST:%[0-9]+]]:_(s1) = G_ICMP intpred(ne), [[LHS]](s32), [[RHS]]
; CHECK: G_STORE [[TST]](s1), [[ADDR]](p0)
define void @int_comparison(i32 %a, i32 %b, i1* %addr) {
  %res = icmp ne i32 %a, %b
  store i1 %res, i1* %addr
  ret void
}

; CHECK-LABEL: name: ptr_comparison
; CHECK: [[LHS:%[0-9]+]]:_(p0) = COPY %x0
; CHECK: [[RHS:%[0-9]+]]:_(p0) = COPY %x1
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY %x2
; CHECK: [[TST:%[0-9]+]]:_(s1) = G_ICMP intpred(eq), [[LHS]](p0), [[RHS]]
; CHECK: G_STORE [[TST]](s1), [[ADDR]](p0)
define void @ptr_comparison(i8* %a, i8* %b, i1* %addr) {
  %res = icmp eq i8* %a, %b
  store i1 %res, i1* %addr
  ret void
}

; CHECK-LABEL: name: test_fadd
; CHECK: [[ARG1:%[0-9]+]]:_(s32) = COPY %s0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s32) = COPY %s1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s32) = G_FADD [[ARG1]], [[ARG2]]
; CHECK-NEXT: %s0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %s0
define float @test_fadd(float %arg1, float %arg2) {
  %res = fadd float %arg1, %arg2
  ret float %res
}

; CHECK-LABEL: name: test_fsub
; CHECK: [[ARG1:%[0-9]+]]:_(s32) = COPY %s0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s32) = COPY %s1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s32) = G_FSUB [[ARG1]], [[ARG2]]
; CHECK-NEXT: %s0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %s0
define float @test_fsub(float %arg1, float %arg2) {
  %res = fsub float %arg1, %arg2
  ret float %res
}

; CHECK-LABEL: name: test_fmul
; CHECK: [[ARG1:%[0-9]+]]:_(s32) = COPY %s0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s32) = COPY %s1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s32) = G_FMUL [[ARG1]], [[ARG2]]
; CHECK-NEXT: %s0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %s0
define float @test_fmul(float %arg1, float %arg2) {
  %res = fmul float %arg1, %arg2
  ret float %res
}

; CHECK-LABEL: name: test_fdiv
; CHECK: [[ARG1:%[0-9]+]]:_(s32) = COPY %s0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s32) = COPY %s1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s32) = G_FDIV [[ARG1]], [[ARG2]]
; CHECK-NEXT: %s0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %s0
define float @test_fdiv(float %arg1, float %arg2) {
  %res = fdiv float %arg1, %arg2
  ret float %res
}

; CHECK-LABEL: name: test_frem
; CHECK: [[ARG1:%[0-9]+]]:_(s32) = COPY %s0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s32) = COPY %s1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s32) = G_FREM [[ARG1]], [[ARG2]]
; CHECK-NEXT: %s0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %s0
define float @test_frem(float %arg1, float %arg2) {
  %res = frem float %arg1, %arg2
  ret float %res
}

; CHECK-LABEL: name: test_sadd_overflow
; CHECK: [[LHS:%[0-9]+]]:_(s32) = COPY %w0
; CHECK: [[RHS:%[0-9]+]]:_(s32) = COPY %w1
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY %x2
; CHECK: [[VAL:%[0-9]+]]:_(s32), [[OVERFLOW:%[0-9]+]]:_(s1) = G_SADDO [[LHS]], [[RHS]]
; CHECK: [[TMP:%[0-9]+]]:_(s64) = G_IMPLICIT_DEF
; CHECK: [[TMP1:%[0-9]+]]:_(s64) = G_INSERT [[TMP]], [[VAL]](s32), 0
; CHECK: [[RES:%[0-9]+]]:_(s64) = G_INSERT [[TMP1]], [[OVERFLOW]](s1), 32
; CHECK: G_STORE [[RES]](s64), [[ADDR]](p0)
declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32)
define void @test_sadd_overflow(i32 %lhs, i32 %rhs, { i32, i1 }* %addr) {
  %res = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %lhs, i32 %rhs)
  store { i32, i1 } %res, { i32, i1 }* %addr
  ret void
}

; CHECK-LABEL: name: test_uadd_overflow
; CHECK: [[LHS:%[0-9]+]]:_(s32) = COPY %w0
; CHECK: [[RHS:%[0-9]+]]:_(s32) = COPY %w1
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY %x2
; CHECK: [[ZERO:%[0-9]+]]:_(s1) = G_CONSTANT i1 false
; CHECK: [[VAL:%[0-9]+]]:_(s32), [[OVERFLOW:%[0-9]+]]:_(s1) = G_UADDE [[LHS]], [[RHS]], [[ZERO]]
; CHECK: [[TMP:%[0-9]+]]:_(s64) = G_IMPLICIT_DEF
; CHECK: [[TMP1:%[0-9]+]]:_(s64) = G_INSERT [[TMP]], [[VAL]](s32), 0
; CHECK: [[RES:%[0-9]+]]:_(s64) = G_INSERT [[TMP1]], [[OVERFLOW]](s1), 32
; CHECK: G_STORE [[RES]](s64), [[ADDR]](p0)
declare { i32, i1 } @llvm.uadd.with.overflow.i32(i32, i32)
define void @test_uadd_overflow(i32 %lhs, i32 %rhs, { i32, i1 }* %addr) {
  %res = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %lhs, i32 %rhs)
  store { i32, i1 } %res, { i32, i1 }* %addr
  ret void
}

; CHECK-LABEL: name: test_ssub_overflow
; CHECK: [[LHS:%[0-9]+]]:_(s32) = COPY %w0
; CHECK: [[RHS:%[0-9]+]]:_(s32) = COPY %w1
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY %x2
; CHECK: [[VAL:%[0-9]+]]:_(s32), [[OVERFLOW:%[0-9]+]]:_(s1) = G_SSUBO [[LHS]], [[RHS]]
; CHECK: [[TMP:%[0-9]+]]:_(s64) = G_IMPLICIT_DEF
; CHECK: [[TMP1:%[0-9]+]]:_(s64) = G_INSERT [[TMP]], [[VAL]](s32), 0
; CHECK: [[RES:%[0-9]+]]:_(s64) = G_INSERT [[TMP1]], [[OVERFLOW]](s1), 32
; CHECK: G_STORE [[RES]](s64), [[ADDR]](p0)
declare { i32, i1 } @llvm.ssub.with.overflow.i32(i32, i32)
define void @test_ssub_overflow(i32 %lhs, i32 %rhs, { i32, i1 }* %subr) {
  %res = call { i32, i1 } @llvm.ssub.with.overflow.i32(i32 %lhs, i32 %rhs)
  store { i32, i1 } %res, { i32, i1 }* %subr
  ret void
}

; CHECK-LABEL: name: test_usub_overflow
; CHECK: [[LHS:%[0-9]+]]:_(s32) = COPY %w0
; CHECK: [[RHS:%[0-9]+]]:_(s32) = COPY %w1
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY %x2
; CHECK: [[ZERO:%[0-9]+]]:_(s1) = G_CONSTANT i1 false
; CHECK: [[VAL:%[0-9]+]]:_(s32), [[OVERFLOW:%[0-9]+]]:_(s1) = G_USUBE [[LHS]], [[RHS]], [[ZERO]]
; CHECK: [[TMP:%[0-9]+]]:_(s64) = G_IMPLICIT_DEF
; CHECK: [[TMP1:%[0-9]+]]:_(s64) = G_INSERT [[TMP]], [[VAL]](s32), 0
; CHECK: [[RES:%[0-9]+]]:_(s64) = G_INSERT [[TMP1]], [[OVERFLOW]](s1), 32
; CHECK: G_STORE [[RES]](s64), [[ADDR]](p0)
declare { i32, i1 } @llvm.usub.with.overflow.i32(i32, i32)
define void @test_usub_overflow(i32 %lhs, i32 %rhs, { i32, i1 }* %subr) {
  %res = call { i32, i1 } @llvm.usub.with.overflow.i32(i32 %lhs, i32 %rhs)
  store { i32, i1 } %res, { i32, i1 }* %subr
  ret void
}

; CHECK-LABEL: name: test_smul_overflow
; CHECK: [[LHS:%[0-9]+]]:_(s32) = COPY %w0
; CHECK: [[RHS:%[0-9]+]]:_(s32) = COPY %w1
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY %x2
; CHECK: [[VAL:%[0-9]+]]:_(s32), [[OVERFLOW:%[0-9]+]]:_(s1) = G_SMULO [[LHS]], [[RHS]]
; CHECK: [[TMP:%[0-9]+]]:_(s64) = G_IMPLICIT_DEF
; CHECK: [[TMP1:%[0-9]+]]:_(s64) = G_INSERT [[TMP]], [[VAL]](s32), 0
; CHECK: [[RES:%[0-9]+]]:_(s64) = G_INSERT [[TMP1]], [[OVERFLOW]](s1), 32
; CHECK: G_STORE [[RES]](s64), [[ADDR]](p0)
declare { i32, i1 } @llvm.smul.with.overflow.i32(i32, i32)
define void @test_smul_overflow(i32 %lhs, i32 %rhs, { i32, i1 }* %addr) {
  %res = call { i32, i1 } @llvm.smul.with.overflow.i32(i32 %lhs, i32 %rhs)
  store { i32, i1 } %res, { i32, i1 }* %addr
  ret void
}

; CHECK-LABEL: name: test_umul_overflow
; CHECK: [[LHS:%[0-9]+]]:_(s32) = COPY %w0
; CHECK: [[RHS:%[0-9]+]]:_(s32) = COPY %w1
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY %x2
; CHECK: [[VAL:%[0-9]+]]:_(s32), [[OVERFLOW:%[0-9]+]]:_(s1) = G_UMULO [[LHS]], [[RHS]]
; CHECK: [[TMP:%[0-9]+]]:_(s64) = G_IMPLICIT_DEF
; CHECK: [[TMP1:%[0-9]+]]:_(s64) = G_INSERT [[TMP]], [[VAL]](s32), 0
; CHECK: [[RES:%[0-9]+]]:_(s64) = G_INSERT [[TMP1]], [[OVERFLOW]](s1), 32
; CHECK: G_STORE [[RES]](s64), [[ADDR]](p0)
declare { i32, i1 } @llvm.umul.with.overflow.i32(i32, i32)
define void @test_umul_overflow(i32 %lhs, i32 %rhs, { i32, i1 }* %addr) {
  %res = call { i32, i1 } @llvm.umul.with.overflow.i32(i32 %lhs, i32 %rhs)
  store { i32, i1 } %res, { i32, i1 }* %addr
  ret void
}

; CHECK-LABEL: name: test_extractvalue
; CHECK: [[STRUCT:%[0-9]+]]:_(s128) = G_LOAD
; CHECK: [[RES:%[0-9]+]]:_(s32) = G_EXTRACT [[STRUCT]](s128), 64
; CHECK: %w0 = COPY [[RES]]
%struct.nested = type {i8, { i8, i32 }, i32}
define i32 @test_extractvalue(%struct.nested* %addr) {
  %struct = load %struct.nested, %struct.nested* %addr
  %res = extractvalue %struct.nested %struct, 1, 1
  ret i32 %res
}

; CHECK-LABEL: name: test_extractvalue_agg
; CHECK: [[STRUCT:%[0-9]+]]:_(s128) = G_LOAD
; CHECK: [[RES:%[0-9]+]]:_(s64) = G_EXTRACT [[STRUCT]](s128), 32
; CHECK: G_STORE [[RES]]
define void @test_extractvalue_agg(%struct.nested* %addr, {i8, i32}* %addr2) {
  %struct = load %struct.nested, %struct.nested* %addr
  %res = extractvalue %struct.nested %struct, 1
  store {i8, i32} %res, {i8, i32}* %addr2
  ret void
}

; CHECK-LABEL: name: test_insertvalue
; CHECK: [[VAL:%[0-9]+]]:_(s32) = COPY %w1
; CHECK: [[STRUCT:%[0-9]+]]:_(s128) = G_LOAD
; CHECK: [[NEWSTRUCT:%[0-9]+]]:_(s128) = G_INSERT [[STRUCT]], [[VAL]](s32), 64
; CHECK: G_STORE [[NEWSTRUCT]](s128),
define void @test_insertvalue(%struct.nested* %addr, i32 %val) {
  %struct = load %struct.nested, %struct.nested* %addr
  %newstruct = insertvalue %struct.nested %struct, i32 %val, 1, 1
  store %struct.nested %newstruct, %struct.nested* %addr
  ret void
}

define [1 x i64] @test_trivial_insert([1 x i64] %s, i64 %val) {
; CHECK-LABEL: name: test_trivial_insert
; CHECK: [[STRUCT:%[0-9]+]]:_(s64) = COPY %x0
; CHECK: [[VAL:%[0-9]+]]:_(s64) = COPY %x1
; CHECK: [[RES:%[0-9]+]]:_(s64) = COPY [[VAL]](s64)
; CHECK: %x0 = COPY [[RES]]
  %res = insertvalue [1 x i64] %s, i64 %val, 0
  ret [1 x i64] %res
}

define [1 x i8*] @test_trivial_insert_ptr([1 x i8*] %s, i8* %val) {
; CHECK-LABEL: name: test_trivial_insert_ptr
; CHECK: [[STRUCT:%[0-9]+]]:_(s64) = COPY %x0
; CHECK: [[VAL:%[0-9]+]]:_(p0) = COPY %x1
; CHECK: [[RES:%[0-9]+]]:_(s64) = G_PTRTOINT [[VAL]](p0)
; CHECK: %x0 = COPY [[RES]]
  %res = insertvalue [1 x i8*] %s, i8* %val, 0
  ret [1 x i8*] %res
}

; CHECK-LABEL: name: test_insertvalue_agg
; CHECK: [[SMALLSTRUCT:%[0-9]+]]:_(s64) = G_LOAD
; CHECK: [[STRUCT:%[0-9]+]]:_(s128) = G_LOAD
; CHECK: [[RES:%[0-9]+]]:_(s128) = G_INSERT [[STRUCT]], [[SMALLSTRUCT]](s64), 32
; CHECK: G_STORE [[RES]](s128)
define void @test_insertvalue_agg(%struct.nested* %addr, {i8, i32}* %addr2) {
  %smallstruct = load {i8, i32}, {i8, i32}* %addr2
  %struct = load %struct.nested, %struct.nested* %addr
  %res = insertvalue %struct.nested %struct, {i8, i32} %smallstruct, 1
  store %struct.nested %res, %struct.nested* %addr
  ret void
}

; CHECK-LABEL: name: test_select
; CHECK: [[TST_C:%[0-9]+]]:_(s32) = COPY %w0
; CHECK: [[TST:%[0-9]+]]:_(s1) = G_TRUNC [[TST_C]]
; CHECK: [[LHS:%[0-9]+]]:_(s32) = COPY %w1
; CHECK: [[RHS:%[0-9]+]]:_(s32) = COPY %w2
; CHECK: [[RES:%[0-9]+]]:_(s32) = G_SELECT [[TST]](s1), [[LHS]], [[RHS]]
; CHECK: %w0 = COPY [[RES]]
define i32 @test_select(i1 %tst, i32 %lhs, i32 %rhs) {
  %res = select i1 %tst, i32 %lhs, i32 %rhs
  ret i32 %res
}

; CHECK-LABEL: name: test_select_ptr
; CHECK: [[TST_C:%[0-9]+]]:_(s32) = COPY %w0
; CHECK: [[TST:%[0-9]+]]:_(s1) = G_TRUNC [[TST_C]]
; CHECK: [[LHS:%[0-9]+]]:_(p0) = COPY %x1
; CHECK: [[RHS:%[0-9]+]]:_(p0) = COPY %x2
; CHECK: [[RES:%[0-9]+]]:_(p0) = G_SELECT [[TST]](s1), [[LHS]], [[RHS]]
; CHECK: %x0 = COPY [[RES]]
define i8* @test_select_ptr(i1 %tst, i8* %lhs, i8* %rhs) {
  %res = select i1 %tst, i8* %lhs, i8* %rhs
  ret i8* %res
}

; CHECK-LABEL: name: test_select_vec
; CHECK: [[TST_C:%[0-9]+]]:_(s32) = COPY %w0
; CHECK: [[TST:%[0-9]+]]:_(s1) = G_TRUNC [[TST_C]]
; CHECK: [[LHS:%[0-9]+]]:_(<4 x s32>) = COPY %q0
; CHECK: [[RHS:%[0-9]+]]:_(<4 x s32>) = COPY %q1
; CHECK: [[RES:%[0-9]+]]:_(<4 x s32>) = G_SELECT [[TST]](s1), [[LHS]], [[RHS]]
; CHECK: %q0 = COPY [[RES]]
define <4 x i32> @test_select_vec(i1 %tst, <4 x i32> %lhs, <4 x i32> %rhs) {
  %res = select i1 %tst, <4 x i32> %lhs, <4 x i32> %rhs
  ret <4 x i32> %res
}

; CHECK-LABEL: name: test_vselect_vec
; CHECK: [[TST32:%[0-9]+]]:_(<4 x s32>) = COPY %q0
; CHECK: [[LHS:%[0-9]+]]:_(<4 x s32>) = COPY %q1
; CHECK: [[RHS:%[0-9]+]]:_(<4 x s32>) = COPY %q2
; CHECK: [[TST:%[0-9]+]]:_(<4 x s1>) = G_TRUNC [[TST32]](<4 x s32>)
; CHECK: [[RES:%[0-9]+]]:_(<4 x s32>) = G_SELECT [[TST]](<4 x s1>), [[LHS]], [[RHS]]
; CHECK: %q0 = COPY [[RES]]
define <4 x i32> @test_vselect_vec(<4 x i32> %tst32, <4 x i32> %lhs, <4 x i32> %rhs) {
  %tst = trunc <4 x i32> %tst32 to <4 x i1>
  %res = select <4 x i1> %tst, <4 x i32> %lhs, <4 x i32> %rhs
  ret <4 x i32> %res
}

; CHECK-LABEL: name: test_fptosi
; CHECK: [[FPADDR:%[0-9]+]]:_(p0) = COPY %x0
; CHECK: [[FP:%[0-9]+]]:_(s32) = G_LOAD [[FPADDR]](p0)
; CHECK: [[RES:%[0-9]+]]:_(s64) = G_FPTOSI [[FP]](s32)
; CHECK: %x0 = COPY [[RES]]
define i64 @test_fptosi(float* %fp.addr) {
  %fp = load float, float* %fp.addr
  %res = fptosi float %fp to i64
  ret i64 %res
}

; CHECK-LABEL: name: test_fptoui
; CHECK: [[FPADDR:%[0-9]+]]:_(p0) = COPY %x0
; CHECK: [[FP:%[0-9]+]]:_(s32) = G_LOAD [[FPADDR]](p0)
; CHECK: [[RES:%[0-9]+]]:_(s64) = G_FPTOUI [[FP]](s32)
; CHECK: %x0 = COPY [[RES]]
define i64 @test_fptoui(float* %fp.addr) {
  %fp = load float, float* %fp.addr
  %res = fptoui float %fp to i64
  ret i64 %res
}

; CHECK-LABEL: name: test_sitofp
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY %x0
; CHECK: [[IN:%[0-9]+]]:_(s32) = COPY %w1
; CHECK: [[FP:%[0-9]+]]:_(s64) = G_SITOFP [[IN]](s32)
; CHECK: G_STORE [[FP]](s64), [[ADDR]](p0)
define void @test_sitofp(double* %addr, i32 %in) {
  %fp = sitofp i32 %in to double
  store double %fp, double* %addr
  ret void
}

; CHECK-LABEL: name: test_uitofp
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY %x0
; CHECK: [[IN:%[0-9]+]]:_(s32) = COPY %w1
; CHECK: [[FP:%[0-9]+]]:_(s64) = G_UITOFP [[IN]](s32)
; CHECK: G_STORE [[FP]](s64), [[ADDR]](p0)
define void @test_uitofp(double* %addr, i32 %in) {
  %fp = uitofp i32 %in to double
  store double %fp, double* %addr
  ret void
}

; CHECK-LABEL: name: test_fpext
; CHECK: [[IN:%[0-9]+]]:_(s32) = COPY %s0
; CHECK: [[RES:%[0-9]+]]:_(s64) = G_FPEXT [[IN]](s32)
; CHECK: %d0 = COPY [[RES]]
define double @test_fpext(float %in) {
  %res = fpext float %in to double
  ret double %res
}

; CHECK-LABEL: name: test_fptrunc
; CHECK: [[IN:%[0-9]+]]:_(s64) = COPY %d0
; CHECK: [[RES:%[0-9]+]]:_(s32) = G_FPTRUNC [[IN]](s64)
; CHECK: %s0 = COPY [[RES]]
define float @test_fptrunc(double %in) {
  %res = fptrunc double %in to float
  ret float %res
}

; CHECK-LABEL: name: test_constant_float
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY %x0
; CHECK: [[TMP:%[0-9]+]]:_(s32) = G_FCONSTANT float 1.500000e+00
; CHECK: G_STORE [[TMP]](s32), [[ADDR]](p0)
define void @test_constant_float(float* %addr) {
  store float 1.5, float* %addr
  ret void
}

; CHECK-LABEL: name: float_comparison
; CHECK: [[LHSADDR:%[0-9]+]]:_(p0) = COPY %x0
; CHECK: [[RHSADDR:%[0-9]+]]:_(p0) = COPY %x1
; CHECK: [[BOOLADDR:%[0-9]+]]:_(p0) = COPY %x2
; CHECK: [[LHS:%[0-9]+]]:_(s32) = G_LOAD [[LHSADDR]](p0)
; CHECK: [[RHS:%[0-9]+]]:_(s32) = G_LOAD [[RHSADDR]](p0)
; CHECK: [[TST:%[0-9]+]]:_(s1) = G_FCMP floatpred(oge), [[LHS]](s32), [[RHS]]
; CHECK: G_STORE [[TST]](s1), [[BOOLADDR]](p0)
define void @float_comparison(float* %a.addr, float* %b.addr, i1* %bool.addr) {
  %a = load float, float* %a.addr
  %b = load float, float* %b.addr
  %res = fcmp oge float %a, %b
  store i1 %res, i1* %bool.addr
  ret void
}

; CHECK-LABEL: name: trivial_float_comparison
; CHECK: [[ENTRY_R1:%[0-9]+]]:_(s1) = G_CONSTANT i1 false
; CHECK: [[ENTRY_R2:%[0-9]+]]:_(s1) = G_CONSTANT i1 true
; CHECK: [[R1:%[0-9]+]]:_(s1) = COPY [[ENTRY_R1]](s1)
; CHECK: [[R2:%[0-9]+]]:_(s1) = COPY [[ENTRY_R2]](s1)
; CHECK: G_ADD [[R1]], [[R2]]
define i1 @trivial_float_comparison(double %a, double %b) {
  %r1 = fcmp false double %a, %b
  %r2 = fcmp true double %a, %b
  %sum = add i1 %r1, %r2
  ret i1 %sum
}

@var = global i32 0

define i32* @test_global() {
; CHECK-LABEL: name: test_global
; CHECK: [[TMP:%[0-9]+]]:_(p0) = G_GLOBAL_VALUE @var{{$}}
; CHECK: %x0 = COPY [[TMP]](p0)

  ret i32* @var
}

@var1 = addrspace(42) global i32 0
define i32 addrspace(42)* @test_global_addrspace() {
; CHECK-LABEL: name: test_global
; CHECK: [[TMP:%[0-9]+]]:_(p42) = G_GLOBAL_VALUE @var1{{$}}
; CHECK: %x0 = COPY [[TMP]](p42)

  ret i32 addrspace(42)* @var1
}


define void()* @test_global_func() {
; CHECK-LABEL: name: test_global_func
; CHECK: [[TMP:%[0-9]+]]:_(p0) = G_GLOBAL_VALUE @allocai64{{$}}
; CHECK: %x0 = COPY [[TMP]](p0)

  ret void()* @allocai64
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64, i32 %align, i1 %volatile)
define void @test_memcpy(i8* %dst, i8* %src, i64 %size) {
; CHECK-LABEL: name: test_memcpy
; CHECK: [[DST:%[0-9]+]]:_(p0) = COPY %x0
; CHECK: [[SRC:%[0-9]+]]:_(p0) = COPY %x1
; CHECK: [[SIZE:%[0-9]+]]:_(s64) = COPY %x2
; CHECK: %x0 = COPY [[DST]]
; CHECK: %x1 = COPY [[SRC]]
; CHECK: %x2 = COPY [[SIZE]]
; CHECK: BL $memcpy, csr_aarch64_aapcs, implicit-def %lr, implicit %sp, implicit %x0, implicit %x1, implicit %x2
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 %size, i32 1, i1 0)
  ret void
}

declare void @llvm.memmove.p0i8.p0i8.i64(i8*, i8*, i64, i32 %align, i1 %volatile)
define void @test_memmove(i8* %dst, i8* %src, i64 %size) {
; CHECK-LABEL: name: test_memmove
; CHECK: [[DST:%[0-9]+]]:_(p0) = COPY %x0
; CHECK: [[SRC:%[0-9]+]]:_(p0) = COPY %x1
; CHECK: [[SIZE:%[0-9]+]]:_(s64) = COPY %x2
; CHECK: %x0 = COPY [[DST]]
; CHECK: %x1 = COPY [[SRC]]
; CHECK: %x2 = COPY [[SIZE]]
; CHECK: BL $memmove, csr_aarch64_aapcs, implicit-def %lr, implicit %sp, implicit %x0, implicit %x1, implicit %x2
  call void @llvm.memmove.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 %size, i32 1, i1 0)
  ret void
}

declare void @llvm.memset.p0i8.i64(i8*, i8, i64, i32 %align, i1 %volatile)
define void @test_memset(i8* %dst, i8 %val, i64 %size) {
; CHECK-LABEL: name: test_memset
; CHECK: [[DST:%[0-9]+]]:_(p0) = COPY %x0
; CHECK: [[SRC_C:%[0-9]+]]:_(s32) = COPY %w1
; CHECK: [[SRC:%[0-9]+]]:_(s8) = G_TRUNC [[SRC_C]]
; CHECK: [[SIZE:%[0-9]+]]:_(s64) = COPY %x2
; CHECK: %x0 = COPY [[DST]]
; CHECK: [[SRC_TMP:%[0-9]+]]:_(s32) = G_ANYEXT [[SRC]]
; CHECK: %w1 = COPY [[SRC_TMP]]
; CHECK: %x2 = COPY [[SIZE]]
; CHECK: BL $memset, csr_aarch64_aapcs, implicit-def %lr, implicit %sp, implicit %x0, implicit %w1, implicit %x2
  call void @llvm.memset.p0i8.i64(i8* %dst, i8 %val, i64 %size, i32 1, i1 0)
  ret void
}

declare i64 @llvm.objectsize.i64(i8*, i1)
declare i32 @llvm.objectsize.i32(i8*, i1)
define void @test_objectsize(i8* %addr0, i8* %addr1) {
; CHECK-LABEL: name: test_objectsize
; CHECK: [[ADDR0:%[0-9]+]]:_(p0) = COPY %x0
; CHECK: [[ADDR1:%[0-9]+]]:_(p0) = COPY %x1
; CHECK: {{%[0-9]+}}:_(s64) = G_CONSTANT i64 -1
; CHECK: {{%[0-9]+}}:_(s64) = G_CONSTANT i64 0
; CHECK: {{%[0-9]+}}:_(s32) = G_CONSTANT i32 -1
; CHECK: {{%[0-9]+}}:_(s32) = G_CONSTANT i32 0
  %size64.0 = call i64 @llvm.objectsize.i64(i8* %addr0, i1 0)
  %size64.intmin = call i64 @llvm.objectsize.i64(i8* %addr0, i1 1)
  %size32.0 = call i32 @llvm.objectsize.i32(i8* %addr0, i1 0)
  %size32.intmin = call i32 @llvm.objectsize.i32(i8* %addr0, i1 1)
  ret void
}

define void @test_large_const(i128* %addr) {
; CHECK-LABEL: name: test_large_const
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY %x0
; CHECK: [[VAL:%[0-9]+]]:_(s128) = G_CONSTANT i128 42
; CHECK: G_STORE [[VAL]](s128), [[ADDR]](p0)
  store i128 42, i128* %addr
  ret void
}

; When there was no formal argument handling (so the first BB was empty) we used
; to insert the constants at the end of the block, even if they were encountered
; after the block's terminators had been emitted. Also make sure the order is
; correct.
define i8* @test_const_placement() {
; CHECK-LABEL: name: test_const_placement
; CHECK: bb.{{[0-9]+}} (%ir-block.{{[0-9]+}}):
; CHECK:   [[VAL_INT:%[0-9]+]]:_(s32) = G_CONSTANT i32 42
; CHECK:   [[VAL:%[0-9]+]]:_(p0) = G_INTTOPTR [[VAL_INT]](s32)
; CHECK: bb.{{[0-9]+}}.{{[a-zA-Z0-9.]+}}:
  br label %next

next:
  ret i8* inttoptr(i32 42 to i8*)
}

declare void @llvm.va_end(i8*)
define void @test_va_end(i8* %list) {
; CHECK-LABEL: name: test_va_end
; CHECK-NOT: va_end
; CHECK-NOT: INTRINSIC
; CHECK: RET_ReallyLR
  call void @llvm.va_end(i8* %list)
  ret void
}

define void @test_va_arg(i8* %list) {
; CHECK-LABEL: test_va_arg
; CHECK: [[LIST:%[0-9]+]]:_(p0) = COPY %x0
; CHECK: G_VAARG [[LIST]](p0), 8
; CHECK: G_VAARG [[LIST]](p0), 1
; CHECK: G_VAARG [[LIST]](p0), 16

  %v0 = va_arg i8* %list, i64
  %v1 = va_arg i8* %list, i8
  %v2 = va_arg i8* %list, i128
  ret void
}

declare float @llvm.pow.f32(float, float)
define float @test_pow_intrin(float %l, float %r) {
; CHECK-LABEL: name: test_pow_intrin
; CHECK: [[LHS:%[0-9]+]]:_(s32) = COPY %s0
; CHECK: [[RHS:%[0-9]+]]:_(s32) = COPY %s1
; CHECK: [[RES:%[0-9]+]]:_(s32) = G_FPOW [[LHS]], [[RHS]]
; CHECK: %s0 = COPY [[RES]]
  %res = call float @llvm.pow.f32(float %l, float %r)
  ret float %res
}

declare float @llvm.fma.f32(float, float, float)
define float @test_fma_intrin(float %a, float %b, float %c) {
; CHECK-LABEL: name: test_fma_intrin
; CHECK: [[A:%[0-9]+]]:_(s32) = COPY %s0
; CHECK: [[B:%[0-9]+]]:_(s32) = COPY %s1
; CHECK: [[C:%[0-9]+]]:_(s32) = COPY %s2
; CHECK: [[RES:%[0-9]+]]:_(s32) = G_FMA [[A]], [[B]], [[C]]
; CHECK: %s0 = COPY [[RES]]
  %res = call float @llvm.fma.f32(float %a, float %b, float %c)
  ret float %res
}

declare float @llvm.exp.f32(float)
define float @test_exp_intrin(float %a) {
; CHECK-LABEL: name: test_exp_intrin
; CHECK: [[A:%[0-9]+]]:_(s32) = COPY %s0
; CHECK: [[RES:%[0-9]+]]:_(s32) = G_FEXP [[A]]
; CHECK: %s0 = COPY [[RES]]
  %res = call float @llvm.exp.f32(float %a)
  ret float %res
}

declare float @llvm.exp2.f32(float)
define float @test_exp2_intrin(float %a) {
; CHECK-LABEL: name: test_exp2_intrin
; CHECK: [[A:%[0-9]+]]:_(s32) = COPY %s0
; CHECK: [[RES:%[0-9]+]]:_(s32) = G_FEXP2 [[A]]
; CHECK: %s0 = COPY [[RES]]
  %res = call float @llvm.exp2.f32(float %a)
  ret float %res
}

declare float @llvm.log.f32(float)
define float @test_log_intrin(float %a) {
; CHECK-LABEL: name: test_log_intrin
; CHECK: [[A:%[0-9]+]]:_(s32) = COPY %s0
; CHECK: [[RES:%[0-9]+]]:_(s32) = G_FLOG [[A]]
; CHECK: %s0 = COPY [[RES]]
  %res = call float @llvm.log.f32(float %a)
  ret float %res
}

declare float @llvm.log2.f32(float)
define float @test_log2_intrin(float %a) {
; CHECK-LABEL: name: test_log2_intrin
; CHECK: [[A:%[0-9]+]]:_(s32) = COPY %s0
; CHECK: [[RES:%[0-9]+]]:_(s32) = G_FLOG2 [[A]]
; CHECK: %s0 = COPY [[RES]]
  %res = call float @llvm.log2.f32(float %a)
  ret float %res
}
declare void @llvm.lifetime.start.p0i8(i64, i8*)
declare void @llvm.lifetime.end.p0i8(i64, i8*)
define void @test_lifetime_intrin() {
; CHECK-LABEL: name: test_lifetime_intrin
; CHECK: RET_ReallyLR
  %slot = alloca i8, i32 4
  call void @llvm.lifetime.start.p0i8(i64 0, i8* %slot)
  call void @llvm.lifetime.end.p0i8(i64 0, i8* %slot)
  ret void
}

define void @test_load_store_atomics(i8* %addr) {
; CHECK-LABEL: name: test_load_store_atomics
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY %x0
; CHECK: [[V0:%[0-9]+]]:_(s8) = G_LOAD [[ADDR]](p0) :: (load unordered 1 from %ir.addr)
; CHECK: G_STORE [[V0]](s8), [[ADDR]](p0) :: (store monotonic 1 into %ir.addr)
; CHECK: [[V1:%[0-9]+]]:_(s8) = G_LOAD [[ADDR]](p0) :: (load acquire 1 from %ir.addr)
; CHECK: G_STORE [[V1]](s8), [[ADDR]](p0) :: (store release 1 into %ir.addr)
; CHECK: [[V2:%[0-9]+]]:_(s8) = G_LOAD [[ADDR]](p0) :: (load syncscope("singlethread") seq_cst 1 from %ir.addr)
; CHECK: G_STORE [[V2]](s8), [[ADDR]](p0) :: (store syncscope("singlethread") monotonic 1 into %ir.addr)
  %v0 = load atomic i8, i8* %addr unordered, align 1
  store atomic i8 %v0, i8* %addr monotonic, align 1

  %v1 = load atomic i8, i8* %addr acquire, align 1
  store atomic i8 %v1, i8* %addr release, align 1

  %v2 = load atomic i8, i8* %addr syncscope("singlethread") seq_cst, align 1
  store atomic i8 %v2, i8* %addr syncscope("singlethread") monotonic, align 1

  ret void
}

define float @test_fneg_f32(float %x) {
; CHECK-LABEL: name: test_fneg_f32
; CHECK: [[ARG:%[0-9]+]]:_(s32) = COPY %s0
; CHECK: [[RES:%[0-9]+]]:_(s32) = G_FNEG [[ARG]]
; CHECK: %s0 = COPY [[RES]](s32)
  %neg = fsub float -0.000000e+00, %x
  ret float %neg
}

define double @test_fneg_f64(double %x) {
; CHECK-LABEL: name: test_fneg_f64
; CHECK: [[ARG:%[0-9]+]]:_(s64) = COPY %d0
; CHECK: [[RES:%[0-9]+]]:_(s64) = G_FNEG [[ARG]]
; CHECK: %d0 = COPY [[RES]](s64)
  %neg = fsub double -0.000000e+00, %x
  ret double %neg
}

define void @test_trivial_inlineasm() {
; CHECK-LABEL: name: test_trivial_inlineasm
; CHECK: INLINEASM $wibble, 1
; CHECK: INLINEASM $wibble, 0
  call void asm sideeffect "wibble", ""()
  call void asm "wibble", ""()
  ret void
}

define <2 x i32> @test_insertelement(<2 x i32> %vec, i32 %elt, i32 %idx){
; CHECK-LABEL: name: test_insertelement
; CHECK: [[VEC:%[0-9]+]]:_(<2 x s32>) = COPY %d0
; CHECK: [[ELT:%[0-9]+]]:_(s32) = COPY %w0
; CHECK: [[IDX:%[0-9]+]]:_(s32) = COPY %w1
; CHECK: [[RES:%[0-9]+]]:_(<2 x s32>) = G_INSERT_VECTOR_ELT [[VEC]], [[ELT]](s32), [[IDX]](s32)
; CHECK: %d0 = COPY [[RES]](<2 x s32>)
  %res = insertelement <2 x i32> %vec, i32 %elt, i32 %idx
  ret <2 x i32> %res
}

define i32 @test_extractelement(<2 x i32> %vec, i32 %idx) {
; CHECK-LABEL: name: test_extractelement
; CHECK: [[VEC:%[0-9]+]]:_(<2 x s32>) = COPY %d0
; CHECK: [[IDX:%[0-9]+]]:_(s32) = COPY %w0
; CHECK: [[RES:%[0-9]+]]:_(s32) = G_EXTRACT_VECTOR_ELT [[VEC]](<2 x s32>), [[IDX]](s32)
; CHECK: %w0 = COPY [[RES]](s32)
  %res = extractelement <2 x i32> %vec, i32 %idx
  ret i32 %res
}

define i32 @test_singleelementvector(i32 %elt){
; CHECK-LABEL: name: test_singleelementvector
; CHECK: [[ELT:%[0-9]+]]:_(s32) = COPY %w0
; CHECK-NOT: G_INSERT_VECTOR_ELT
; CHECK-NOT: G_EXTRACT_VECTOR_ELT
; CHECK: %w0 = COPY [[ELT]](s32)
  %vec = insertelement <1 x i32> undef, i32 %elt, i32 0
  %res = extractelement <1 x i32> %vec, i32 0
  ret i32 %res
}

define <2 x i32> @test_constantaggzerovector_v2i32() {
; CHECK-LABEL: name: test_constantaggzerovector_v2i32
; CHECK: [[ZERO:%[0-9]+]]:_(s32) = G_CONSTANT i32 0
; CHECK: [[VEC:%[0-9]+]]:_(<2 x s32>) = G_MERGE_VALUES [[ZERO]](s32), [[ZERO]](s32)
; CHECK: %d0 = COPY [[VEC]](<2 x s32>)
  ret <2 x i32> zeroinitializer
}

define <2 x float> @test_constantaggzerovector_v2f32() {
; CHECK-LABEL: name: test_constantaggzerovector_v2f32
; CHECK: [[ZERO:%[0-9]+]]:_(s32) = G_FCONSTANT float 0.000000e+00
; CHECK: [[VEC:%[0-9]+]]:_(<2 x s32>) = G_MERGE_VALUES [[ZERO]](s32), [[ZERO]](s32)
; CHECK: %d0 = COPY [[VEC]](<2 x s32>)
  ret <2 x float> zeroinitializer
}

define i32 @test_constantaggzerovector_v3i32() {
; CHECK-LABEL: name: test_constantaggzerovector_v3i32
; CHECK: [[ZERO:%[0-9]+]]:_(s32) = G_CONSTANT i32 0
; CHECK: [[VEC:%[0-9]+]]:_(<3 x s32>) = G_MERGE_VALUES [[ZERO]](s32), [[ZERO]](s32), [[ZERO]](s32)
; CHECK: G_EXTRACT_VECTOR_ELT [[VEC]](<3 x s32>)
  %elt = extractelement <3 x i32> zeroinitializer, i32 1
  ret i32 %elt
}

define <2 x i32> @test_constantdatavector_v2i32() {
; CHECK-LABEL: name: test_constantdatavector_v2i32
; CHECK: [[C1:%[0-9]+]]:_(s32) = G_CONSTANT i32 1
; CHECK: [[C2:%[0-9]+]]:_(s32) = G_CONSTANT i32 2
; CHECK: [[VEC:%[0-9]+]]:_(<2 x s32>) = G_MERGE_VALUES [[C1]](s32), [[C2]](s32)
; CHECK: %d0 = COPY [[VEC]](<2 x s32>)
  ret <2 x i32> <i32 1, i32 2>
}

define i32 @test_constantdatavector_v3i32() {
; CHECK-LABEL: name: test_constantdatavector_v3i32
; CHECK: [[C1:%[0-9]+]]:_(s32) = G_CONSTANT i32 1
; CHECK: [[C2:%[0-9]+]]:_(s32) = G_CONSTANT i32 2
; CHECK: [[C3:%[0-9]+]]:_(s32) = G_CONSTANT i32 3
; CHECK: [[VEC:%[0-9]+]]:_(<3 x s32>) = G_MERGE_VALUES [[C1]](s32), [[C2]](s32), [[C3]](s32)
; CHECK: G_EXTRACT_VECTOR_ELT [[VEC]](<3 x s32>)
  %elt = extractelement <3 x i32> <i32 1, i32 2, i32 3>, i32 1
  ret i32 %elt
}

define <4 x i32> @test_constantdatavector_v4i32() {
; CHECK-LABEL: name: test_constantdatavector_v4i32
; CHECK: [[C1:%[0-9]+]]:_(s32) = G_CONSTANT i32 1
; CHECK: [[C2:%[0-9]+]]:_(s32) = G_CONSTANT i32 2
; CHECK: [[C3:%[0-9]+]]:_(s32) = G_CONSTANT i32 3
; CHECK: [[C4:%[0-9]+]]:_(s32) = G_CONSTANT i32 4
; CHECK: [[VEC:%[0-9]+]]:_(<4 x s32>) = G_MERGE_VALUES [[C1]](s32), [[C2]](s32), [[C3]](s32), [[C4]](s32)
; CHECK: %q0 = COPY [[VEC]](<4 x s32>)
  ret <4 x i32> <i32 1, i32 2, i32 3, i32 4>
}

define <2 x double> @test_constantdatavector_v2f64() {
; CHECK-LABEL: name: test_constantdatavector_v2f64
; CHECK: [[FC1:%[0-9]+]]:_(s64) = G_FCONSTANT double 1.000000e+00
; CHECK: [[FC2:%[0-9]+]]:_(s64) = G_FCONSTANT double 2.000000e+00
; CHECK: [[VEC:%[0-9]+]]:_(<2 x s64>) = G_MERGE_VALUES [[FC1]](s64), [[FC2]](s64)
; CHECK: %q0 = COPY [[VEC]](<2 x s64>)
  ret <2 x double> <double 1.0, double 2.0>
}

define i32 @test_constantaggzerovector_v1s32(i32 %arg){
; CHECK-LABEL: name: test_constantaggzerovector_v1s32
; CHECK: [[ARG:%[0-9]+]]:_(s32) = COPY %w0
; CHECK: [[C0:%[0-9]+]]:_(s32) = G_CONSTANT i32 0
; CHECK-NOT: G_MERGE_VALUES
; CHECK: G_ADD [[ARG]], [[C0]]
  %vec = insertelement <1 x i32> undef, i32 %arg, i32 0
  %add = add <1 x i32> %vec, zeroinitializer
  %res = extractelement <1 x i32> %add, i32 0
  ret i32 %res
}

define i32 @test_constantdatavector_v1s32(i32 %arg){
; CHECK-LABEL: name: test_constantdatavector_v1s32
; CHECK: [[ARG:%[0-9]+]]:_(s32) = COPY %w0
; CHECK: [[C1:%[0-9]+]]:_(s32) = G_CONSTANT i32 1
; CHECK-NOT: G_MERGE_VALUES
; CHECK: G_ADD [[ARG]], [[C1]]
  %vec = insertelement <1 x i32> undef, i32 %arg, i32 0
  %add = add <1 x i32> %vec, <i32 1>
  %res = extractelement <1 x i32> %add, i32 0
  ret i32 %res
}

declare ghccc float @different_call_conv_target(float %x)
define float @test_different_call_conv_target(float %x) {
; CHECK-LABEL: name: test_different_call_conv
; CHECK: [[X:%[0-9]+]]:_(s32) = COPY %s0
; CHECK: %s8 = COPY [[X]]
; CHECK: BL @different_call_conv_target, csr_aarch64_aapcs, implicit-def %lr, implicit %sp, implicit %s8, implicit-def %s0
  %res = call ghccc float @different_call_conv_target(float %x)
  ret float %res
}

define <2 x i32> @test_shufflevector_s32_v2s32(i32 %arg) {
; CHECK-LABEL: name: test_shufflevector_s32_v2s32
; CHECK: [[ARG:%[0-9]+]]:_(s32) = COPY %w0
; CHECK-DAG: [[UNDEF:%[0-9]+]]:_(s32) = G_IMPLICIT_DEF
; CHECK-DAG: [[C0:%[0-9]+]]:_(s32) = G_CONSTANT i32 0
; CHECK-DAG: [[MASK:%[0-9]+]]:_(<2 x s32>) = G_MERGE_VALUES [[C0]](s32), [[C0]](s32)
; CHECK: [[VEC:%[0-9]+]]:_(<2 x s32>) = G_SHUFFLE_VECTOR [[ARG]](s32), [[UNDEF]], [[MASK]](<2 x s32>)
; CHECK: %d0 = COPY [[VEC]](<2 x s32>)
  %vec = insertelement <1 x i32> undef, i32 %arg, i32 0
  %res = shufflevector <1 x i32> %vec, <1 x i32> undef, <2 x i32> zeroinitializer
  ret <2 x i32> %res
}

define i32 @test_shufflevector_v2s32_s32(<2 x i32> %arg) {
; CHECK-LABEL: name: test_shufflevector_v2s32_s32
; CHECK: [[ARG:%[0-9]+]]:_(<2 x s32>) = COPY %d0
; CHECK-DAG: [[UNDEF:%[0-9]+]]:_(<2 x s32>) = G_IMPLICIT_DEF
; CHECK-DAG: [[C1:%[0-9]+]]:_(s32) = G_CONSTANT i32 1
; CHECK: [[RES:%[0-9]+]]:_(s32) = G_SHUFFLE_VECTOR [[ARG]](<2 x s32>), [[UNDEF]], [[C1]](s32)
; CHECK: %w0 = COPY [[RES]](s32)
  %vec = shufflevector <2 x i32> %arg, <2 x i32> undef, <1 x i32> <i32 1>
  %res = extractelement <1 x i32> %vec, i32 0
  ret i32 %res
}

define <2 x i32> @test_shufflevector_v2s32_v2s32(<2 x i32> %arg) {
; CHECK-LABEL: name: test_shufflevector_v2s32_v2s32
; CHECK: [[ARG:%[0-9]+]]:_(<2 x s32>) = COPY %d0
; CHECK-DAG: [[UNDEF:%[0-9]+]]:_(<2 x s32>) = G_IMPLICIT_DEF
; CHECK-DAG: [[C1:%[0-9]+]]:_(s32) = G_CONSTANT i32 1
; CHECK-DAG: [[C0:%[0-9]+]]:_(s32) = G_CONSTANT i32 0
; CHECK-DAG: [[MASK:%[0-9]+]]:_(<2 x s32>) = G_MERGE_VALUES [[C1]](s32), [[C0]](s32)
; CHECK: [[VEC:%[0-9]+]]:_(<2 x s32>) = G_SHUFFLE_VECTOR [[ARG]](<2 x s32>), [[UNDEF]], [[MASK]](<2 x s32>)
; CHECK: %d0 = COPY [[VEC]](<2 x s32>)
  %res = shufflevector <2 x i32> %arg, <2 x i32> undef, <2 x i32> <i32 1, i32 0>
  ret <2 x i32> %res
}

define i32 @test_shufflevector_v2s32_v3s32(<2 x i32> %arg) {
; CHECK-LABEL: name: test_shufflevector_v2s32_v3s32
; CHECK: [[ARG:%[0-9]+]]:_(<2 x s32>) = COPY %d0
; CHECK-DAG: [[UNDEF:%[0-9]+]]:_(<2 x s32>) = G_IMPLICIT_DEF
; CHECK-DAG: [[C1:%[0-9]+]]:_(s32) = G_CONSTANT i32 1
; CHECK-DAG: [[C0:%[0-9]+]]:_(s32) = G_CONSTANT i32 0
; CHECK-DAG: [[MASK:%[0-9]+]]:_(<3 x s32>) = G_MERGE_VALUES [[C1]](s32), [[C0]](s32), [[C1]](s32)
; CHECK: [[VEC:%[0-9]+]]:_(<3 x s32>) = G_SHUFFLE_VECTOR [[ARG]](<2 x s32>), [[UNDEF]], [[MASK]](<3 x s32>)
; CHECK: G_EXTRACT_VECTOR_ELT [[VEC]](<3 x s32>)
  %vec = shufflevector <2 x i32> %arg, <2 x i32> undef, <3 x i32> <i32 1, i32 0, i32 1>
  %res = extractelement <3 x i32> %vec, i32 0
  ret i32 %res
}

define <4 x i32> @test_shufflevector_v2s32_v4s32(<2 x i32> %arg1, <2 x i32> %arg2) {
; CHECK-LABEL: name: test_shufflevector_v2s32_v4s32
; CHECK: [[ARG1:%[0-9]+]]:_(<2 x s32>) = COPY %d0
; CHECK: [[ARG2:%[0-9]+]]:_(<2 x s32>) = COPY %d1
; CHECK: [[C0:%[0-9]+]]:_(s32) = G_CONSTANT i32 0
; CHECK: [[C1:%[0-9]+]]:_(s32) = G_CONSTANT i32 1
; CHECK: [[C2:%[0-9]+]]:_(s32) = G_CONSTANT i32 2
; CHECK: [[C3:%[0-9]+]]:_(s32) = G_CONSTANT i32 3
; CHECK: [[MASK:%[0-9]+]]:_(<4 x s32>) = G_MERGE_VALUES [[C0]](s32), [[C1]](s32), [[C2]](s32), [[C3]](s32)
; CHECK: [[VEC:%[0-9]+]]:_(<4 x s32>) = G_SHUFFLE_VECTOR [[ARG1]](<2 x s32>), [[ARG2]], [[MASK]](<4 x s32>)
; CHECK: %q0 = COPY [[VEC]](<4 x s32>)
  %res = shufflevector <2 x i32> %arg1, <2 x i32> %arg2, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x i32> %res
}

define <2 x i32> @test_shufflevector_v4s32_v2s32(<4 x i32> %arg) {
; CHECK-LABEL: name: test_shufflevector_v4s32_v2s32
; CHECK: [[ARG:%[0-9]+]]:_(<4 x s32>) = COPY %q0
; CHECK-DAG: [[UNDEF:%[0-9]+]]:_(<4 x s32>) = G_IMPLICIT_DEF
; CHECK-DAG: [[C1:%[0-9]+]]:_(s32) = G_CONSTANT i32 1
; CHECK-DAG: [[C3:%[0-9]+]]:_(s32) = G_CONSTANT i32 3
; CHECK-DAG: [[MASK:%[0-9]+]]:_(<2 x s32>) = G_MERGE_VALUES [[C1]](s32), [[C3]](s32)
; CHECK: [[VEC:%[0-9]+]]:_(<2 x s32>) = G_SHUFFLE_VECTOR [[ARG]](<4 x s32>), [[UNDEF]], [[MASK]](<2 x s32>)
; CHECK: %d0 = COPY [[VEC]](<2 x s32>)
  %res = shufflevector <4 x i32> %arg, <4 x i32> undef, <2 x i32> <i32 1, i32 3>
  ret <2 x i32> %res
}


define <16 x i8> @test_shufflevector_v8s8_v16s8(<8 x i8> %arg1, <8 x i8> %arg2) {
; CHECK-LABEL: name: test_shufflevector_v8s8_v16s8
; CHECK: [[ARG1:%[0-9]+]]:_(<8 x s8>) = COPY %d0
; CHECK: [[ARG2:%[0-9]+]]:_(<8 x s8>) = COPY %d1
; CHECK: [[C0:%[0-9]+]]:_(s32) = G_CONSTANT i32 0
; CHECK: [[C8:%[0-9]+]]:_(s32) = G_CONSTANT i32 8
; CHECK: [[C1:%[0-9]+]]:_(s32) = G_CONSTANT i32 1
; CHECK: [[C9:%[0-9]+]]:_(s32) = G_CONSTANT i32 9
; CHECK: [[C2:%[0-9]+]]:_(s32) = G_CONSTANT i32 2
; CHECK: [[C10:%[0-9]+]]:_(s32) = G_CONSTANT i32 10
; CHECK: [[C3:%[0-9]+]]:_(s32) = G_CONSTANT i32 3
; CHECK: [[C11:%[0-9]+]]:_(s32) = G_CONSTANT i32 11
; CHECK: [[C4:%[0-9]+]]:_(s32) = G_CONSTANT i32 4
; CHECK: [[C12:%[0-9]+]]:_(s32) = G_CONSTANT i32 12
; CHECK: [[C5:%[0-9]+]]:_(s32) = G_CONSTANT i32 5
; CHECK: [[C13:%[0-9]+]]:_(s32) = G_CONSTANT i32 13
; CHECK: [[C6:%[0-9]+]]:_(s32) = G_CONSTANT i32 6
; CHECK: [[C14:%[0-9]+]]:_(s32) = G_CONSTANT i32 14
; CHECK: [[C7:%[0-9]+]]:_(s32) = G_CONSTANT i32 7
; CHECK: [[C15:%[0-9]+]]:_(s32) = G_CONSTANT i32 15
; CHECK: [[MASK:%[0-9]+]]:_(<16 x s32>) = G_MERGE_VALUES [[C0]](s32), [[C8]](s32), [[C1]](s32), [[C9]](s32), [[C2]](s32), [[C10]](s32), [[C3]](s32), [[C11]](s32), [[C4]](s32), [[C12]](s32), [[C5]](s32), [[C13]](s32), [[C6]](s32), [[C14]](s32), [[C7]](s32), [[C15]](s32)
; CHECK: [[VEC:%[0-9]+]]:_(<16 x s8>) = G_SHUFFLE_VECTOR [[ARG1]](<8 x s8>), [[ARG2]], [[MASK]](<16 x s32>)
; CHECK: %q0 = COPY [[VEC]](<16 x s8>)
  %res = shufflevector <8 x i8> %arg1, <8 x i8> %arg2, <16 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11, i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  ret <16 x i8> %res
}

; CHECK-LABEL: test_constant_vector
; CHECK: [[UNDEF:%[0-9]+]]:_(s16) = G_IMPLICIT_DEF
; CHECK: [[F:%[0-9]+]]:_(s16) = G_FCONSTANT half 0xH3C00
; CHECK: [[M:%[0-9]+]]:_(<4 x s16>) = G_MERGE_VALUES [[UNDEF]](s16), [[UNDEF]](s16), [[UNDEF]](s16), [[F]](s16)
; CHECK: %d0 = COPY [[M]](<4 x s16>)
define <4 x half> @test_constant_vector() {
  ret <4 x half> <half undef, half undef, half undef, half 0xH3C00>
}

define i32 @test_target_mem_intrinsic(i32* %addr) {
; CHECK-LABEL: name: test_target_mem_intrinsic
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY %x0
; CHECK: [[VAL:%[0-9]+]]:_(s64) = G_INTRINSIC_W_SIDE_EFFECTS intrinsic(@llvm.aarch64.ldxr), [[ADDR]](p0) :: (volatile load 4 from %ir.addr)
; CHECK: G_TRUNC [[VAL]](s64)
  %val = call i64 @llvm.aarch64.ldxr.p0i32(i32* %addr)
  %trunc = trunc i64 %val to i32
  ret i32 %trunc
}

declare i64 @llvm.aarch64.ldxr.p0i32(i32*) nounwind

%zerosize_type = type {}

define %zerosize_type @test_empty_load_store(%zerosize_type *%ptr, %zerosize_type %in) noinline optnone {
; CHECK-LABEL: name: test_empty_load_store
; CHECK-NOT: G_STORE
; CHECK-NOT: G_LOAD
; CHECK: RET_ReallyLR
entry:
  store %zerosize_type undef, %zerosize_type* undef, align 4
  %val = load %zerosize_type, %zerosize_type* %ptr, align 4
  ret %zerosize_type %in
}
