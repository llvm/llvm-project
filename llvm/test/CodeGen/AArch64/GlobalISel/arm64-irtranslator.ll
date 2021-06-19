; RUN: llc -O0 -aarch64-enable-atomic-cfg-tidy=0 -stop-after=irtranslator -global-isel -verify-machineinstrs %s -o - 2>&1 | FileCheck %s
; RUN: llc -O3 -aarch64-enable-atomic-cfg-tidy=0 -stop-after=irtranslator -global-isel -verify-machineinstrs %s -o - 2>&1 | FileCheck %s --check-prefix=O3

; This file checks that the translation from llvm IR to generic MachineInstr
; is correct.
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--"

; Tests for add.
; CHECK-LABEL: name: addi64
; CHECK:      [[ARG1:%[0-9]+]]:_(s64) = COPY $x0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s64) = COPY $x1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s64) = G_ADD [[ARG1]], [[ARG2]]
; CHECK-NEXT: $x0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit $x0
define i64 @addi64(i64 %arg1, i64 %arg2) {
  %res = add i64 %arg1, %arg2
  ret i64 %res
}

; CHECK-LABEL: name: muli64
; CHECK: [[ARG1:%[0-9]+]]:_(s64) = COPY $x0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s64) = COPY $x1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s64) = G_MUL [[ARG1]], [[ARG2]]
; CHECK-NEXT: $x0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit $x0
define i64 @muli64(i64 %arg1, i64 %arg2) {
  %res = mul i64 %arg1, %arg2
  ret i64 %res
}

; Tests for alloca
; CHECK-LABEL: name: allocai64
; CHECK: stack:
; CHECK-NEXT:   - { id: 0, name: ptr1, type: default, offset: 0, size: 8, alignment: 8,
; CHECK-NEXT:       stack-id: default, callee-saved-register: '', callee-saved-restored: true,
; CHECK-NEXT: debug-info-variable: '', debug-info-expression: '', debug-info-location: '' }
; CHECK-NEXT:   - { id: 1, name: ptr2, type: default, offset: 0, size: 8, alignment: 1,
; CHECK-NEXT:       stack-id: default, callee-saved-register: '', callee-saved-restored: true,
; CHECK-NEXT:       debug-info-variable: '', debug-info-expression: '', debug-info-location: '' }
; CHECK-NEXT:   - { id: 2, name: ptr3, type: default, offset: 0, size: 128, alignment: 8,
; CHECK-NEXT:       stack-id: default, callee-saved-register: '', callee-saved-restored: true,
; CHECK-NEXT:       debug-info-variable: '', debug-info-expression: '', debug-info-location: '' }
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
; CHECK: [[ADDR:%.*]]:_(p0) = COPY $x0
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
; CHECK: [[ARG1:%[0-9]+]]:_(s64) = COPY $x0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s64) = COPY $x1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s64) = G_OR [[ARG1]], [[ARG2]]
; CHECK-NEXT: $x0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit $x0
define i64 @ori64(i64 %arg1, i64 %arg2) {
  %res = or i64 %arg1, %arg2
  ret i64 %res
}

; CHECK-LABEL: name: ori32
; CHECK: [[ARG1:%[0-9]+]]:_(s32) = COPY $w0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s32) = COPY $w1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s32) = G_OR [[ARG1]], [[ARG2]]
; CHECK-NEXT: $w0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit $w0
define i32 @ori32(i32 %arg1, i32 %arg2) {
  %res = or i32 %arg1, %arg2
  ret i32 %res
}

; Tests for xor.
; CHECK-LABEL: name: xori64
; CHECK: [[ARG1:%[0-9]+]]:_(s64) = COPY $x0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s64) = COPY $x1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s64) = G_XOR [[ARG1]], [[ARG2]]
; CHECK-NEXT: $x0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit $x0
define i64 @xori64(i64 %arg1, i64 %arg2) {
  %res = xor i64 %arg1, %arg2
  ret i64 %res
}

; CHECK-LABEL: name: xori32
; CHECK: [[ARG1:%[0-9]+]]:_(s32) = COPY $w0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s32) = COPY $w1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s32) = G_XOR [[ARG1]], [[ARG2]]
; CHECK-NEXT: $w0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit $w0
define i32 @xori32(i32 %arg1, i32 %arg2) {
  %res = xor i32 %arg1, %arg2
  ret i32 %res
}

; Tests for and.
; CHECK-LABEL: name: andi64
; CHECK: [[ARG1:%[0-9]+]]:_(s64) = COPY $x0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s64) = COPY $x1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s64) = G_AND [[ARG1]], [[ARG2]]
; CHECK-NEXT: $x0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit $x0
define i64 @andi64(i64 %arg1, i64 %arg2) {
  %res = and i64 %arg1, %arg2
  ret i64 %res
}

; CHECK-LABEL: name: andi32
; CHECK: [[ARG1:%[0-9]+]]:_(s32) = COPY $w0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s32) = COPY $w1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s32) = G_AND [[ARG1]], [[ARG2]]
; CHECK-NEXT: $w0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit $w0
define i32 @andi32(i32 %arg1, i32 %arg2) {
  %res = and i32 %arg1, %arg2
  ret i32 %res
}

; Tests for sub.
; CHECK-LABEL: name: subi64
; CHECK: [[ARG1:%[0-9]+]]:_(s64) = COPY $x0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s64) = COPY $x1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s64) = G_SUB [[ARG1]], [[ARG2]]
; CHECK-NEXT: $x0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit $x0
define i64 @subi64(i64 %arg1, i64 %arg2) {
  %res = sub i64 %arg1, %arg2
  ret i64 %res
}

; CHECK-LABEL: name: subi32
; CHECK: [[ARG1:%[0-9]+]]:_(s32) = COPY $w0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s32) = COPY $w1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s32) = G_SUB [[ARG1]], [[ARG2]]
; CHECK-NEXT: $w0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit $w0
define i32 @subi32(i32 %arg1, i32 %arg2) {
  %res = sub i32 %arg1, %arg2
  ret i32 %res
}

; CHECK-LABEL: name: ptrtoint
; CHECK: [[ARG1:%[0-9]+]]:_(p0) = COPY $x0
; CHECK: [[RES:%[0-9]+]]:_(s64) = G_PTRTOINT [[ARG1]]
; CHECK: $x0 = COPY [[RES]]
; CHECK: RET_ReallyLR implicit $x0
define i64 @ptrtoint(i64* %a) {
  %val = ptrtoint i64* %a to i64
  ret i64 %val
}

; CHECK-LABEL: name: inttoptr
; CHECK: [[ARG1:%[0-9]+]]:_(s64) = COPY $x0
; CHECK: [[RES:%[0-9]+]]:_(p0) = G_INTTOPTR [[ARG1]]
; CHECK: $x0 = COPY [[RES]]
; CHECK: RET_ReallyLR implicit $x0
define i64* @inttoptr(i64 %a) {
  %val = inttoptr i64 %a to i64*
  ret i64* %val
}

; CHECK-LABEL: name: trivial_bitcast
; CHECK: [[ARG1:%[0-9]+]]:_(p0) = COPY $x0
; CHECK: $x0 = COPY [[ARG1]]
; CHECK: RET_ReallyLR implicit $x0
define i64* @trivial_bitcast(i8* %a) {
  %val = bitcast i8* %a to i64*
  ret i64* %val
}

; CHECK-LABEL: name: trivial_bitcast_with_copy
; CHECK:     [[A:%[0-9]+]]:_(p0) = COPY $x0
; CHECK:     G_BR %[[CAST:bb\.[0-9]+]]

; CHECK: [[END:bb\.[0-9]+]].{{[a-zA-Z0-9.]+}}:
; CHECK:     $x0 = COPY [[A]]

; CHECK: [[CAST]].{{[a-zA-Z0-9.]+}}:
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
; CHECK: [[ARG1:%[0-9]+]]:_(s64) = COPY $x0
; CHECK: [[RES1:%[0-9]+]]:_(<2 x s32>) = G_BITCAST [[ARG1]]
; CHECK: [[RES2:%[0-9]+]]:_(s64) = G_BITCAST [[RES1]]
; CHECK: $x0 = COPY [[RES2]]
; CHECK: RET_ReallyLR implicit $x0
define i64 @bitcast(i64 %a) {
  %res1 = bitcast i64 %a to <2 x i32>
  %res2 = bitcast <2 x i32> %res1 to i64
  ret i64 %res2
}

; CHECK-LABEL: name: addrspacecast
; CHECK: [[ARG1:%[0-9]+]]:_(p1) = COPY $x0
; CHECK: [[RES1:%[0-9]+]]:_(p2) = G_ADDRSPACE_CAST [[ARG1]]
; CHECK: [[RES2:%[0-9]+]]:_(p0) = G_ADDRSPACE_CAST [[RES1]]
; CHECK: $x0 = COPY [[RES2]]
; CHECK: RET_ReallyLR implicit $x0
define i64* @addrspacecast(i32 addrspace(1)* %a) {
  %res1 = addrspacecast i32 addrspace(1)* %a to i64 addrspace(2)*
  %res2 = addrspacecast i64 addrspace(2)* %res1 to i64*
  ret i64* %res2
}

; CHECK-LABEL: name: trunc
; CHECK: [[ARG1:%[0-9]+]]:_(s64) = COPY $x0
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
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY $x0
; CHECK: [[ADDR42:%[0-9]+]]:_(p42) = COPY $x1
; CHECK: [[VAL1:%[0-9]+]]:_(s64) = G_LOAD [[ADDR]](p0) :: (load 8 from %ir.addr, align 16)
; CHECK: [[VAL2:%[0-9]+]]:_(s64) = G_LOAD [[ADDR42]](p42) :: (load 8 from %ir.addr42, addrspace 42)
; CHECK: [[SUM2:%.*]]:_(s64) = G_ADD [[VAL1]], [[VAL2]]
; CHECK: [[VAL3:%[0-9]+]]:_(s64) = G_LOAD [[ADDR]](p0) :: (volatile load 8 from %ir.addr)
; CHECK: [[SUM3:%[0-9]+]]:_(s64) = G_ADD [[SUM2]], [[VAL3]]
; CHECK: [[VAL4:%[0-9]+]]:_(s64) = G_LOAD [[ADDR]](p0) :: (load 8 from %ir.addr, !range !0)
; CHECK: [[SUM4:%[0-9]+]]:_(s64) = G_ADD [[SUM3]], [[VAL4]]
; CHECK: $x0 = COPY [[SUM4]]
; CHECK: RET_ReallyLR implicit $x0
define i64 @load(i64* %addr, i64 addrspace(42)* %addr42) {
  %val1 = load i64, i64* %addr, align 16

  %val2 = load i64, i64 addrspace(42)* %addr42
  %sum2 = add i64 %val1, %val2

  %val3 = load volatile i64, i64* %addr
  %sum3 = add i64 %sum2, %val3

  %val4 = load i64, i64* %addr, !range !0
  %sum4 = add i64 %sum3, %val4
  ret i64 %sum4
}

; CHECK-LABEL: name: store
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY $x0
; CHECK: [[ADDR42:%[0-9]+]]:_(p42) = COPY $x1
; CHECK: [[VAL1:%[0-9]+]]:_(s64) = COPY $x2
; CHECK: [[VAL2:%[0-9]+]]:_(s64) = COPY $x3
; CHECK: G_STORE [[VAL1]](s64), [[ADDR]](p0) :: (store 8 into %ir.addr, align 16)
; CHECK: G_STORE [[VAL2]](s64), [[ADDR42]](p42) :: (store 8 into %ir.addr42, addrspace 42)
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
; CHECK: [[CUR:%[0-9]+]]:_(s32) = COPY $w0
; CHECK: [[BITS:%[0-9]+]]:_(s32) = COPY $w1
; CHECK: [[PTR:%[0-9]+]]:_(p0) = G_INTRINSIC intrinsic(@llvm.returnaddress), 0
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
; CHECK:     $w0 = COPY [[RES]]
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
; CHECK: [[IN:%[0-9]+]]:_(s32) = COPY $w0
; CHECK: [[ONE:%[0-9]+]]:_(s32) = G_CONSTANT i32 1

; CHECK: bb.{{[0-9]+}}.{{[a-zA-Z0-9.]+}}:
; CHECK: [[SUM1:%[0-9]+]]:_(s32) = G_ADD [[IN]], [[ONE]]
; CHECK: [[SUM2:%[0-9]+]]:_(s32) = G_ADD [[IN]], [[ONE]]
; CHECK: [[RES:%[0-9]+]]:_(s32) = G_ADD [[SUM1]], [[SUM2]]
; CHECK: $w0 = COPY [[RES]]

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
; CHECK: [[RES:%[0-9]+]]:_(s32) = G_CONSTANT i32 44
define i32 @constant_int_start() {
  %res = add i32 2, 42
  ret i32 %res
}

; CHECK-LABEL: name: test_undef
; CHECK: [[UNDEF:%[0-9]+]]:_(s32) = G_IMPLICIT_DEF
; CHECK: $w0 = COPY [[UNDEF]]
define i32 @test_undef() {
  ret i32 undef
}

; CHECK-LABEL: name: test_constant_inttoptr
; CHECK: [[ONE:%[0-9]+]]:_(s64) = G_CONSTANT i64 1
; CHECK: [[PTR:%[0-9]+]]:_(p0) = G_INTTOPTR [[ONE]]
; CHECK: $x0 = COPY [[PTR]]
define i8* @test_constant_inttoptr() {
  ret i8* inttoptr(i64 1 to i8*)
}

  ; This failed purely because the Constant -> VReg map was kept across
  ; functions, so reuse the "i64 1" from above.
; CHECK-LABEL: name: test_reused_constant
; CHECK: [[ONE:%[0-9]+]]:_(s64) = G_CONSTANT i64 1
; CHECK: $x0 = COPY [[ONE]]
define i64 @test_reused_constant() {
  ret i64 1
}

; CHECK-LABEL: name: test_sext
; CHECK: [[IN:%[0-9]+]]:_(s32) = COPY $w0
; CHECK: [[RES:%[0-9]+]]:_(s64) = G_SEXT [[IN]]
; CHECK: $x0 = COPY [[RES]]
define i64 @test_sext(i32 %in) {
  %res = sext i32 %in to i64
  ret i64 %res
}

; CHECK-LABEL: name: test_zext
; CHECK: [[IN:%[0-9]+]]:_(s32) = COPY $w0
; CHECK: [[RES:%[0-9]+]]:_(s64) = G_ZEXT [[IN]]
; CHECK: $x0 = COPY [[RES]]
define i64 @test_zext(i32 %in) {
  %res = zext i32 %in to i64
  ret i64 %res
}

; CHECK-LABEL: name: test_shl
; CHECK: [[ARG1:%[0-9]+]]:_(s32) = COPY $w0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s32) = COPY $w1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s32) = G_SHL [[ARG1]], [[ARG2]]
; CHECK-NEXT: $w0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit $w0
define i32 @test_shl(i32 %arg1, i32 %arg2) {
  %res = shl i32 %arg1, %arg2
  ret i32 %res
}


; CHECK-LABEL: name: test_lshr
; CHECK: [[ARG1:%[0-9]+]]:_(s32) = COPY $w0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s32) = COPY $w1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s32) = G_LSHR [[ARG1]], [[ARG2]]
; CHECK-NEXT: $w0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit $w0
define i32 @test_lshr(i32 %arg1, i32 %arg2) {
  %res = lshr i32 %arg1, %arg2
  ret i32 %res
}

; CHECK-LABEL: name: test_ashr
; CHECK: [[ARG1:%[0-9]+]]:_(s32) = COPY $w0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s32) = COPY $w1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s32) = G_ASHR [[ARG1]], [[ARG2]]
; CHECK-NEXT: $w0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit $w0
define i32 @test_ashr(i32 %arg1, i32 %arg2) {
  %res = ashr i32 %arg1, %arg2
  ret i32 %res
}

; CHECK-LABEL: name: test_sdiv
; CHECK: [[ARG1:%[0-9]+]]:_(s32) = COPY $w0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s32) = COPY $w1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s32) = G_SDIV [[ARG1]], [[ARG2]]
; CHECK-NEXT: $w0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit $w0
define i32 @test_sdiv(i32 %arg1, i32 %arg2) {
  %res = sdiv i32 %arg1, %arg2
  ret i32 %res
}

; CHECK-LABEL: name: test_udiv
; CHECK: [[ARG1:%[0-9]+]]:_(s32) = COPY $w0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s32) = COPY $w1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s32) = G_UDIV [[ARG1]], [[ARG2]]
; CHECK-NEXT: $w0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit $w0
define i32 @test_udiv(i32 %arg1, i32 %arg2) {
  %res = udiv i32 %arg1, %arg2
  ret i32 %res
}

; CHECK-LABEL: name: test_srem
; CHECK: [[ARG1:%[0-9]+]]:_(s32) = COPY $w0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s32) = COPY $w1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s32) = G_SREM [[ARG1]], [[ARG2]]
; CHECK-NEXT: $w0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit $w0
define i32 @test_srem(i32 %arg1, i32 %arg2) {
  %res = srem i32 %arg1, %arg2
  ret i32 %res
}

; CHECK-LABEL: name: test_urem
; CHECK: [[ARG1:%[0-9]+]]:_(s32) = COPY $w0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s32) = COPY $w1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s32) = G_UREM [[ARG1]], [[ARG2]]
; CHECK-NEXT: $w0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit $w0
define i32 @test_urem(i32 %arg1, i32 %arg2) {
  %res = urem i32 %arg1, %arg2
  ret i32 %res
}

; CHECK-LABEL: name: test_constant_null
; CHECK: [[NULL:%[0-9]+]]:_(p0) = G_CONSTANT i64 0
; CHECK: $x0 = COPY [[NULL]]
define i8* @test_constant_null() {
  ret i8* null
}

; CHECK-LABEL: name: test_struct_memops
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY $x0
; CHECK: [[VAL1:%[0-9]+]]:_(s8) = G_LOAD %0(p0) :: (load 1 from %ir.addr, align 4)
; CHECK: [[CST1:%[0-9]+]]:_(s64) = G_CONSTANT i64 4
; CHECK: [[GEP1:%[0-9]+]]:_(p0) = G_PTR_ADD [[ADDR]], [[CST1]](s64)
; CHECK: [[VAL2:%[0-9]+]]:_(s32) = G_LOAD [[GEP1]](p0) :: (load 4 from %ir.addr + 4)
; CHECK: G_STORE [[VAL1]](s8), [[ADDR]](p0) :: (store 1 into %ir.addr, align 4)
; CHECK: [[GEP2:%[0-9]+]]:_(p0) = G_PTR_ADD [[ADDR]], [[CST1]](s64)
; CHECK: G_STORE [[VAL2]](s32), [[GEP2]](p0) :: (store 4 into %ir.addr + 4)
define void @test_struct_memops({ i8, i32 }* %addr) {
  %val = load { i8, i32 }, { i8, i32 }* %addr
  store { i8, i32 } %val, { i8, i32 }* %addr
  ret void
}

; CHECK-LABEL: name: test_i1_memops
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY $x0
; CHECK: [[VAL:%[0-9]+]]:_(s1) = G_LOAD [[ADDR]](p0) :: (load 1 from  %ir.addr)
; CHECK: G_STORE [[VAL]](s1), [[ADDR]](p0) :: (store 1 into  %ir.addr)
define void @test_i1_memops(i1* %addr) {
  %val = load i1, i1* %addr
  store i1 %val, i1* %addr
  ret void
}

; CHECK-LABEL: name: int_comparison
; CHECK: [[LHS:%[0-9]+]]:_(s32) = COPY $w0
; CHECK: [[RHS:%[0-9]+]]:_(s32) = COPY $w1
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY $x2
; CHECK: [[TST:%[0-9]+]]:_(s1) = G_ICMP intpred(ne), [[LHS]](s32), [[RHS]]
; CHECK: G_STORE [[TST]](s1), [[ADDR]](p0)
define void @int_comparison(i32 %a, i32 %b, i1* %addr) {
  %res = icmp ne i32 %a, %b
  store i1 %res, i1* %addr
  ret void
}

; CHECK-LABEL: name: ptr_comparison
; CHECK: [[LHS:%[0-9]+]]:_(p0) = COPY $x0
; CHECK: [[RHS:%[0-9]+]]:_(p0) = COPY $x1
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY $x2
; CHECK: [[TST:%[0-9]+]]:_(s1) = G_ICMP intpred(eq), [[LHS]](p0), [[RHS]]
; CHECK: G_STORE [[TST]](s1), [[ADDR]](p0)
define void @ptr_comparison(i8* %a, i8* %b, i1* %addr) {
  %res = icmp eq i8* %a, %b
  store i1 %res, i1* %addr
  ret void
}

; CHECK-LABEL: name: test_fadd
; CHECK: [[ARG1:%[0-9]+]]:_(s32) = COPY $s0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s32) = COPY $s1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s32) = G_FADD [[ARG1]], [[ARG2]]
; CHECK-NEXT: $s0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit $s0
define float @test_fadd(float %arg1, float %arg2) {
  %res = fadd float %arg1, %arg2
  ret float %res
}

; CHECK-LABEL: name: test_fsub
; CHECK: [[ARG1:%[0-9]+]]:_(s32) = COPY $s0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s32) = COPY $s1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s32) = G_FSUB [[ARG1]], [[ARG2]]
; CHECK-NEXT: $s0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit $s0
define float @test_fsub(float %arg1, float %arg2) {
  %res = fsub float %arg1, %arg2
  ret float %res
}

; CHECK-LABEL: name: test_fmul
; CHECK: [[ARG1:%[0-9]+]]:_(s32) = COPY $s0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s32) = COPY $s1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s32) = G_FMUL [[ARG1]], [[ARG2]]
; CHECK-NEXT: $s0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit $s0
define float @test_fmul(float %arg1, float %arg2) {
  %res = fmul float %arg1, %arg2
  ret float %res
}

; CHECK-LABEL: name: test_fdiv
; CHECK: [[ARG1:%[0-9]+]]:_(s32) = COPY $s0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s32) = COPY $s1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s32) = G_FDIV [[ARG1]], [[ARG2]]
; CHECK-NEXT: $s0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit $s0
define float @test_fdiv(float %arg1, float %arg2) {
  %res = fdiv float %arg1, %arg2
  ret float %res
}

; CHECK-LABEL: name: test_frem
; CHECK: [[ARG1:%[0-9]+]]:_(s32) = COPY $s0
; CHECK-NEXT: [[ARG2:%[0-9]+]]:_(s32) = COPY $s1
; CHECK-NEXT: [[RES:%[0-9]+]]:_(s32) = G_FREM [[ARG1]], [[ARG2]]
; CHECK-NEXT: $s0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit $s0
define float @test_frem(float %arg1, float %arg2) {
  %res = frem float %arg1, %arg2
  ret float %res
}

; CHECK-LABEL: name: test_sadd_overflow
; CHECK: [[LHS:%[0-9]+]]:_(s32) = COPY $w0
; CHECK: [[RHS:%[0-9]+]]:_(s32) = COPY $w1
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY $x2
; CHECK: [[VAL:%[0-9]+]]:_(s32), [[OVERFLOW:%[0-9]+]]:_(s1) = G_SADDO [[LHS]], [[RHS]]
; CHECK: G_STORE [[VAL]](s32), [[ADDR]](p0) :: (store 4 into %ir.addr)
; CHECK: [[CST:%[0-9]+]]:_(s64) = G_CONSTANT i64 4
; CHECK: [[GEP:%[0-9]+]]:_(p0) = G_PTR_ADD [[ADDR]], [[CST]](s64)
; CHECK: G_STORE [[OVERFLOW]](s1), [[GEP]](p0) :: (store 1 into %ir.addr + 4, align 4)
declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32)
define void @test_sadd_overflow(i32 %lhs, i32 %rhs, { i32, i1 }* %addr) {
  %res = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %lhs, i32 %rhs)
  store { i32, i1 } %res, { i32, i1 }* %addr
  ret void
}

; CHECK-LABEL: name: test_uadd_overflow
; CHECK: [[LHS:%[0-9]+]]:_(s32) = COPY $w0
; CHECK: [[RHS:%[0-9]+]]:_(s32) = COPY $w1
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY $x2
; CHECK: [[VAL:%[0-9]+]]:_(s32), [[OVERFLOW:%[0-9]+]]:_(s1) = G_UADDO [[LHS]], [[RHS]]
; CHECK: G_STORE [[VAL]](s32), [[ADDR]](p0) :: (store 4 into %ir.addr)
; CHECK: [[CST:%[0-9]+]]:_(s64) = G_CONSTANT i64 4
; CHECK: [[GEP:%[0-9]+]]:_(p0) = G_PTR_ADD [[ADDR]], [[CST]](s64)
; CHECK: G_STORE [[OVERFLOW]](s1), [[GEP]](p0) :: (store 1 into %ir.addr + 4, align 4)
declare { i32, i1 } @llvm.uadd.with.overflow.i32(i32, i32)
define void @test_uadd_overflow(i32 %lhs, i32 %rhs, { i32, i1 }* %addr) {
  %res = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %lhs, i32 %rhs)
  store { i32, i1 } %res, { i32, i1 }* %addr
  ret void
}

; CHECK-LABEL: name: test_ssub_overflow
; CHECK: [[LHS:%[0-9]+]]:_(s32) = COPY $w0
; CHECK: [[RHS:%[0-9]+]]:_(s32) = COPY $w1
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY $x2
; CHECK: [[VAL:%[0-9]+]]:_(s32), [[OVERFLOW:%[0-9]+]]:_(s1) = G_SSUBO [[LHS]], [[RHS]]
; CHECK: G_STORE [[VAL]](s32), [[ADDR]](p0) :: (store 4 into %ir.subr)
; CHECK: [[CST:%[0-9]+]]:_(s64) = G_CONSTANT i64 4
; CHECK: [[GEP:%[0-9]+]]:_(p0) = G_PTR_ADD [[ADDR]], [[CST]](s64)
; CHECK: G_STORE [[OVERFLOW]](s1), [[GEP]](p0) :: (store 1 into %ir.subr + 4, align 4)
declare { i32, i1 } @llvm.ssub.with.overflow.i32(i32, i32)
define void @test_ssub_overflow(i32 %lhs, i32 %rhs, { i32, i1 }* %subr) {
  %res = call { i32, i1 } @llvm.ssub.with.overflow.i32(i32 %lhs, i32 %rhs)
  store { i32, i1 } %res, { i32, i1 }* %subr
  ret void
}

; CHECK-LABEL: name: test_usub_overflow
; CHECK: [[LHS:%[0-9]+]]:_(s32) = COPY $w0
; CHECK: [[RHS:%[0-9]+]]:_(s32) = COPY $w1
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY $x2
; CHECK: [[VAL:%[0-9]+]]:_(s32), [[OVERFLOW:%[0-9]+]]:_(s1) = G_USUBO [[LHS]], [[RHS]]
; CHECK: G_STORE [[VAL]](s32), [[ADDR]](p0) :: (store 4 into %ir.subr)
; CHECK: [[CST:%[0-9]+]]:_(s64) = G_CONSTANT i64 4
; CHECK: [[GEP:%[0-9]+]]:_(p0) = G_PTR_ADD [[ADDR]], [[CST]](s64)
; CHECK: G_STORE [[OVERFLOW]](s1), [[GEP]](p0) :: (store 1 into %ir.subr + 4, align 4)
declare { i32, i1 } @llvm.usub.with.overflow.i32(i32, i32)
define void @test_usub_overflow(i32 %lhs, i32 %rhs, { i32, i1 }* %subr) {
  %res = call { i32, i1 } @llvm.usub.with.overflow.i32(i32 %lhs, i32 %rhs)
  store { i32, i1 } %res, { i32, i1 }* %subr
  ret void
}

; CHECK-LABEL: name: test_smul_overflow
; CHECK: [[LHS:%[0-9]+]]:_(s32) = COPY $w0
; CHECK: [[RHS:%[0-9]+]]:_(s32) = COPY $w1
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY $x2
; CHECK: [[VAL:%[0-9]+]]:_(s32), [[OVERFLOW:%[0-9]+]]:_(s1) = G_SMULO [[LHS]], [[RHS]]
; CHECK: G_STORE [[VAL]](s32), [[ADDR]](p0) :: (store 4 into %ir.addr)
; CHECK: [[CST:%[0-9]+]]:_(s64) = G_CONSTANT i64 4
; CHECK: [[GEP:%[0-9]+]]:_(p0) = G_PTR_ADD [[ADDR]], [[CST]](s64)
; CHECK: G_STORE [[OVERFLOW]](s1), [[GEP]](p0) :: (store 1 into %ir.addr + 4, align 4)
declare { i32, i1 } @llvm.smul.with.overflow.i32(i32, i32)
define void @test_smul_overflow(i32 %lhs, i32 %rhs, { i32, i1 }* %addr) {
  %res = call { i32, i1 } @llvm.smul.with.overflow.i32(i32 %lhs, i32 %rhs)
  store { i32, i1 } %res, { i32, i1 }* %addr
  ret void
}

; CHECK-LABEL: name: test_umul_overflow
; CHECK: [[LHS:%[0-9]+]]:_(s32) = COPY $w0
; CHECK: [[RHS:%[0-9]+]]:_(s32) = COPY $w1
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY $x2
; CHECK: [[VAL:%[0-9]+]]:_(s32), [[OVERFLOW:%[0-9]+]]:_(s1) = G_UMULO [[LHS]], [[RHS]]
; CHECK: G_STORE [[VAL]](s32), [[ADDR]](p0) :: (store 4 into %ir.addr)
; CHECK: [[CST:%[0-9]+]]:_(s64) = G_CONSTANT i64 4
; CHECK: [[GEP:%[0-9]+]]:_(p0) = G_PTR_ADD [[ADDR]], [[CST]](s64)
; CHECK: G_STORE [[OVERFLOW]](s1), [[GEP]](p0) :: (store 1 into %ir.addr + 4, align 4)
declare { i32, i1 } @llvm.umul.with.overflow.i32(i32, i32)
define void @test_umul_overflow(i32 %lhs, i32 %rhs, { i32, i1 }* %addr) {
  %res = call { i32, i1 } @llvm.umul.with.overflow.i32(i32 %lhs, i32 %rhs)
  store { i32, i1 } %res, { i32, i1 }* %addr
  ret void
}

; CHECK-LABEL: name: test_extractvalue
; CHECK: %0:_(p0) = COPY $x0
; CHECK: [[LD1:%[0-9]+]]:_(s8) = G_LOAD %0(p0) :: (load 1 from %ir.addr, align 4)
; CHECK: [[CST1:%[0-9]+]]:_(s64) = G_CONSTANT i64 4
; CHECK: [[GEP1:%[0-9]+]]:_(p0) = G_PTR_ADD %0, [[CST1]](s64)
; CHECK: [[LD2:%[0-9]+]]:_(s8) = G_LOAD [[GEP1]](p0) :: (load 1 from %ir.addr + 4, align 4)
; CHECK: [[CST2:%[0-9]+]]:_(s64) = G_CONSTANT i64 8
; CHECK: [[GEP2:%[0-9]+]]:_(p0) = G_PTR_ADD %0, [[CST2]](s64)
; CHECK: [[LD3:%[0-9]+]]:_(s32) = G_LOAD [[GEP2]](p0) :: (load 4 from %ir.addr + 8)
; CHECK: [[CST3:%[0-9]+]]:_(s64) = G_CONSTANT i64 12
; CHECK: [[GEP3:%[0-9]+]]:_(p0) = G_PTR_ADD %0, [[CST3]](s64)
; CHECK: [[LD4:%[0-9]+]]:_(s32) = G_LOAD [[GEP3]](p0) :: (load 4 from %ir.addr + 12)
; CHECK: $w0 = COPY [[LD3]](s32)
%struct.nested = type {i8, { i8, i32 }, i32}
define i32 @test_extractvalue(%struct.nested* %addr) {
  %struct = load %struct.nested, %struct.nested* %addr
  %res = extractvalue %struct.nested %struct, 1, 1
  ret i32 %res
}

; CHECK-LABEL: name: test_extractvalue_agg
; CHECK: %0:_(p0) = COPY $x0
; CHECK: %1:_(p0) = COPY $x1
; CHECK: [[LD1:%[0-9]+]]:_(s8) = G_LOAD %0(p0) :: (load 1 from %ir.addr, align 4)
; CHECK: [[CST1:%[0-9]+]]:_(s64) = G_CONSTANT i64 4
; CHECK: [[GEP1:%[0-9]+]]:_(p0) = G_PTR_ADD %0, [[CST1]](s64)
; CHECK: [[LD2:%[0-9]+]]:_(s8) = G_LOAD [[GEP1]](p0) :: (load 1 from %ir.addr + 4, align 4)
; CHECK: [[CST2:%[0-9]+]]:_(s64) = G_CONSTANT i64 8
; CHECK: [[GEP2:%[0-9]+]]:_(p0) = G_PTR_ADD %0, [[CST2]](s64)
; CHECK: [[LD3:%[0-9]+]]:_(s32) = G_LOAD [[GEP2]](p0) :: (load 4 from %ir.addr + 8)
; CHECK: [[CST3:%[0-9]+]]:_(s64) = G_CONSTANT i64 12
; CHECK: [[GEP3:%[0-9]+]]:_(p0) = G_PTR_ADD %0, [[CST3]](s64)
; CHECK: [[LD4:%[0-9]+]]:_(s32) = G_LOAD [[GEP3]](p0) :: (load 4 from %ir.addr + 12)
; CHECK: G_STORE [[LD2]](s8), %1(p0) :: (store 1 into %ir.addr2, align 4)
; CHECK: [[GEP4:%[0-9]+]]:_(p0) = G_PTR_ADD %1, [[CST1]](s64)
; CHECK: G_STORE [[LD3]](s32), [[GEP4]](p0) :: (store 4 into %ir.addr2 + 4)
define void @test_extractvalue_agg(%struct.nested* %addr, {i8, i32}* %addr2) {
  %struct = load %struct.nested, %struct.nested* %addr
  %res = extractvalue %struct.nested %struct, 1
  store {i8, i32} %res, {i8, i32}* %addr2
  ret void
}

; CHECK-LABEL: name: test_trivial_extract_ptr
; CHECK: [[STRUCT:%[0-9]+]]:_(p0) = COPY $x0
; CHECK: [[VAL32:%[0-9]+]]:_(s32) = COPY $w1
; CHECK: [[VAL:%[0-9]+]]:_(s8) = G_TRUNC [[VAL32]]
; CHECK: G_STORE [[VAL]](s8), [[STRUCT]](p0)
define void @test_trivial_extract_ptr([1 x i8*] %s, i8 %val) {
  %addr = extractvalue [1 x i8*] %s, 0
  store i8 %val, i8* %addr
  ret void
}

; CHECK-LABEL: name: test_insertvalue
; CHECK: %0:_(p0) = COPY $x0
; CHECK: %1:_(s32) = COPY $w1
; CHECK: [[LD1:%[0-9]+]]:_(s8) = G_LOAD %0(p0) :: (load 1 from %ir.addr, align 4)
; CHECK: [[CST1:%[0-9]+]]:_(s64) = G_CONSTANT i64 4
; CHECK: [[GEP1:%[0-9]+]]:_(p0) = G_PTR_ADD %0, [[CST1]](s64)
; CHECK: [[LD2:%[0-9]+]]:_(s8) = G_LOAD [[GEP1]](p0) :: (load 1 from %ir.addr + 4, align 4)
; CHECK: [[CST2:%[0-9]+]]:_(s64) = G_CONSTANT i64 8
; CHECK: [[GEP2:%[0-9]+]]:_(p0) = G_PTR_ADD %0, [[CST2]](s64)
; CHECK: [[LD3:%[0-9]+]]:_(s32) = G_LOAD [[GEP2]](p0) :: (load 4 from %ir.addr + 8)
; CHECK: [[CST3:%[0-9]+]]:_(s64) = G_CONSTANT i64 12
; CHECK: [[GEP3:%[0-9]+]]:_(p0) = G_PTR_ADD %0, [[CST3]](s64)
; CHECK: [[LD4:%[0-9]+]]:_(s32) = G_LOAD [[GEP3]](p0) :: (load 4 from %ir.addr + 12)
; CHECK: G_STORE [[LD1]](s8), %0(p0) :: (store 1 into %ir.addr, align 4)
; CHECK: [[GEP4:%[0-9]+]]:_(p0) = G_PTR_ADD %0, [[CST1]](s64)
; CHECK: G_STORE [[LD2]](s8), [[GEP4]](p0) :: (store 1 into %ir.addr + 4, align 4)
; CHECK: [[GEP5:%[0-9]+]]:_(p0) = G_PTR_ADD %0, [[CST2]](s64)
; CHECK: G_STORE %1(s32), [[GEP5]](p0) :: (store 4 into %ir.addr + 8)
; CHECK: [[GEP6:%[0-9]+]]:_(p0) = G_PTR_ADD %0, [[CST3]](s64)
; CHECK: G_STORE [[LD4]](s32), [[GEP6]](p0) :: (store 4 into %ir.addr + 12)
define void @test_insertvalue(%struct.nested* %addr, i32 %val) {
  %struct = load %struct.nested, %struct.nested* %addr
  %newstruct = insertvalue %struct.nested %struct, i32 %val, 1, 1
  store %struct.nested %newstruct, %struct.nested* %addr
  ret void
}

define [1 x i64] @test_trivial_insert([1 x i64] %s, i64 %val) {
; CHECK-LABEL: name: test_trivial_insert
; CHECK: [[STRUCT:%[0-9]+]]:_(s64) = COPY $x0
; CHECK: [[VAL:%[0-9]+]]:_(s64) = COPY $x1
; CHECK: $x0 = COPY [[VAL]]
  %res = insertvalue [1 x i64] %s, i64 %val, 0
  ret [1 x i64] %res
}

define [1 x i8*] @test_trivial_insert_ptr([1 x i8*] %s, i8* %val) {
; CHECK-LABEL: name: test_trivial_insert_ptr
; CHECK: [[STRUCT:%[0-9]+]]:_(p0) = COPY $x0
; CHECK: [[VAL:%[0-9]+]]:_(p0) = COPY $x1
; CHECK: $x0 = COPY [[VAL]]
  %res = insertvalue [1 x i8*] %s, i8* %val, 0
  ret [1 x i8*] %res
}

; CHECK-LABEL: name: test_insertvalue_agg
; CHECK: %0:_(p0) = COPY $x0
; CHECK: %1:_(p0) = COPY $x1
; CHECK: [[LD1:%[0-9]+]]:_(s8) = G_LOAD %1(p0) :: (load 1 from %ir.addr2, align 4)
; CHECK: [[CST1:%[0-9]+]]:_(s64) = G_CONSTANT i64 4
; CHECK: [[GEP1:%[0-9]+]]:_(p0) = G_PTR_ADD %1, [[CST1]](s64)
; CHECK: [[LD2:%[0-9]+]]:_(s32) = G_LOAD [[GEP1]](p0) :: (load 4 from %ir.addr2 + 4)
; CHECK: [[LD3:%[0-9]+]]:_(s8) = G_LOAD %0(p0) :: (load 1 from %ir.addr, align 4)
; CHECK: [[GEP2:%[0-9]+]]:_(p0) = G_PTR_ADD %0, [[CST1]](s64)
; CHECK: [[LD4:%[0-9]+]]:_(s8) = G_LOAD [[GEP2]](p0) :: (load 1 from %ir.addr + 4, align 4)
; CHECK: [[CST3:%[0-9]+]]:_(s64) = G_CONSTANT i64 8
; CHECK: [[GEP3:%[0-9]+]]:_(p0) = G_PTR_ADD %0, [[CST3]](s64)
; CHECK: [[LD5:%[0-9]+]]:_(s32) = G_LOAD [[GEP3]](p0) :: (load 4 from %ir.addr + 8)
; CHECK: [[CST4:%[0-9]+]]:_(s64) = G_CONSTANT i64 12
; CHECK: [[GEP4:%[0-9]+]]:_(p0) = G_PTR_ADD %0, [[CST4]](s64)
; CHECK: [[LD6:%[0-9]+]]:_(s32) = G_LOAD [[GEP4]](p0) :: (load 4 from %ir.addr + 12)
; CHECK: G_STORE [[LD3]](s8), %0(p0) :: (store 1 into %ir.addr, align 4)
; CHECK: [[GEP5:%[0-9]+]]:_(p0) = G_PTR_ADD %0, [[CST1]](s64)
; CHECK: G_STORE [[LD1]](s8), [[GEP5]](p0) :: (store 1 into %ir.addr + 4, align 4)
; CHECK: [[GEP6:%[0-9]+]]:_(p0) = G_PTR_ADD %0, [[CST3]](s64)
; CHECK: G_STORE [[LD2]](s32), [[GEP6]](p0) :: (store 4 into %ir.addr + 8)
; CHECK: [[GEP7:%[0-9]+]]:_(p0) = G_PTR_ADD %0, [[CST4]](s64)
; CHECK: G_STORE [[LD6]](s32), [[GEP7]](p0) :: (store 4 into %ir.addr + 12)
define void @test_insertvalue_agg(%struct.nested* %addr, {i8, i32}* %addr2) {
  %smallstruct = load {i8, i32}, {i8, i32}* %addr2
  %struct = load %struct.nested, %struct.nested* %addr
  %res = insertvalue %struct.nested %struct, {i8, i32} %smallstruct, 1
  store %struct.nested %res, %struct.nested* %addr
  ret void
}

; CHECK-LABEL: name: test_select
; CHECK: [[TST_C:%[0-9]+]]:_(s32) = COPY $w0
; CHECK: [[TST:%[0-9]+]]:_(s1) = G_TRUNC [[TST_C]]
; CHECK: [[LHS:%[0-9]+]]:_(s32) = COPY $w1
; CHECK: [[RHS:%[0-9]+]]:_(s32) = COPY $w2
; CHECK: [[RES:%[0-9]+]]:_(s32) = G_SELECT [[TST]](s1), [[LHS]], [[RHS]]
; CHECK: $w0 = COPY [[RES]]
define i32 @test_select(i1 %tst, i32 %lhs, i32 %rhs) {
  %res = select i1 %tst, i32 %lhs, i32 %rhs
  ret i32 %res
}

; CHECK-LABEL: name: test_select_flags
; CHECK:   [[COPY:%[0-9]+]]:_(s32) = COPY $w0
; CHECK:   [[TRUNC:%[0-9]+]]:_(s1) = G_TRUNC [[COPY]](s32)
; CHECK:   [[COPY1:%[0-9]+]]:_(s32) = COPY $s0
; CHECK:   [[COPY2:%[0-9]+]]:_(s32) = COPY $s1
; CHECK:   [[SELECT:%[0-9]+]]:_(s32) = nnan G_SELECT [[TRUNC]](s1), [[COPY1]], [[COPY2]]
define float @test_select_flags(i1 %tst, float %lhs, float %rhs) {
  %res = select nnan i1 %tst, float %lhs, float %rhs
  ret float %res
}

; Don't take the flags from the compare condition
; CHECK-LABEL: name: test_select_cmp_flags
; CHECK:   [[COPY0:%[0-9]+]]:_(s32) = COPY $s0
; CHECK:   [[COPY1:%[0-9]+]]:_(s32) = COPY $s1
; CHECK:   [[COPY2:%[0-9]+]]:_(s32) = COPY $s2
; CHECK:   [[COPY3:%[0-9]+]]:_(s32) = COPY $s3
; CHECK:   [[CMP:%[0-9]+]]:_(s1) = nsz G_FCMP floatpred(oeq), [[COPY0]](s32), [[COPY1]]
; CHECK:   [[SELECT:%[0-9]+]]:_(s32) = G_SELECT [[CMP]](s1), [[COPY2]], [[COPY3]]
define float @test_select_cmp_flags(float %cmp0, float %cmp1, float %lhs, float %rhs) {
  %tst = fcmp nsz oeq float %cmp0, %cmp1
  %res = select i1 %tst, float %lhs, float %rhs
  ret float %res
}

; CHECK-LABEL: name: test_select_ptr
; CHECK: [[TST_C:%[0-9]+]]:_(s32) = COPY $w0
; CHECK: [[TST:%[0-9]+]]:_(s1) = G_TRUNC [[TST_C]]
; CHECK: [[LHS:%[0-9]+]]:_(p0) = COPY $x1
; CHECK: [[RHS:%[0-9]+]]:_(p0) = COPY $x2
; CHECK: [[RES:%[0-9]+]]:_(p0) = G_SELECT [[TST]](s1), [[LHS]], [[RHS]]
; CHECK: $x0 = COPY [[RES]]
define i8* @test_select_ptr(i1 %tst, i8* %lhs, i8* %rhs) {
  %res = select i1 %tst, i8* %lhs, i8* %rhs
  ret i8* %res
}

; CHECK-LABEL: name: test_select_vec
; CHECK: [[TST_C:%[0-9]+]]:_(s32) = COPY $w0
; CHECK: [[TST:%[0-9]+]]:_(s1) = G_TRUNC [[TST_C]]
; CHECK: [[LHS:%[0-9]+]]:_(<4 x s32>) = COPY $q0
; CHECK: [[RHS:%[0-9]+]]:_(<4 x s32>) = COPY $q1
; CHECK: [[RES:%[0-9]+]]:_(<4 x s32>) = G_SELECT [[TST]](s1), [[LHS]], [[RHS]]
; CHECK: $q0 = COPY [[RES]]
define <4 x i32> @test_select_vec(i1 %tst, <4 x i32> %lhs, <4 x i32> %rhs) {
  %res = select i1 %tst, <4 x i32> %lhs, <4 x i32> %rhs
  ret <4 x i32> %res
}

; CHECK-LABEL: name: test_vselect_vec
; CHECK: [[TST32:%[0-9]+]]:_(<4 x s32>) = COPY $q0
; CHECK: [[LHS:%[0-9]+]]:_(<4 x s32>) = COPY $q1
; CHECK: [[RHS:%[0-9]+]]:_(<4 x s32>) = COPY $q2
; CHECK: [[TST:%[0-9]+]]:_(<4 x s1>) = G_TRUNC [[TST32]](<4 x s32>)
; CHECK: [[RES:%[0-9]+]]:_(<4 x s32>) = G_SELECT [[TST]](<4 x s1>), [[LHS]], [[RHS]]
; CHECK: $q0 = COPY [[RES]]
define <4 x i32> @test_vselect_vec(<4 x i32> %tst32, <4 x i32> %lhs, <4 x i32> %rhs) {
  %tst = trunc <4 x i32> %tst32 to <4 x i1>
  %res = select <4 x i1> %tst, <4 x i32> %lhs, <4 x i32> %rhs
  ret <4 x i32> %res
}

; CHECK-LABEL: name: test_fptosi
; CHECK: [[FPADDR:%[0-9]+]]:_(p0) = COPY $x0
; CHECK: [[FP:%[0-9]+]]:_(s32) = G_LOAD [[FPADDR]](p0)
; CHECK: [[RES:%[0-9]+]]:_(s64) = G_FPTOSI [[FP]](s32)
; CHECK: $x0 = COPY [[RES]]
define i64 @test_fptosi(float* %fp.addr) {
  %fp = load float, float* %fp.addr
  %res = fptosi float %fp to i64
  ret i64 %res
}

; CHECK-LABEL: name: test_fptoui
; CHECK: [[FPADDR:%[0-9]+]]:_(p0) = COPY $x0
; CHECK: [[FP:%[0-9]+]]:_(s32) = G_LOAD [[FPADDR]](p0)
; CHECK: [[RES:%[0-9]+]]:_(s64) = G_FPTOUI [[FP]](s32)
; CHECK: $x0 = COPY [[RES]]
define i64 @test_fptoui(float* %fp.addr) {
  %fp = load float, float* %fp.addr
  %res = fptoui float %fp to i64
  ret i64 %res
}

; CHECK-LABEL: name: test_sitofp
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY $x0
; CHECK: [[IN:%[0-9]+]]:_(s32) = COPY $w1
; CHECK: [[FP:%[0-9]+]]:_(s64) = G_SITOFP [[IN]](s32)
; CHECK: G_STORE [[FP]](s64), [[ADDR]](p0)
define void @test_sitofp(double* %addr, i32 %in) {
  %fp = sitofp i32 %in to double
  store double %fp, double* %addr
  ret void
}

; CHECK-LABEL: name: test_uitofp
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY $x0
; CHECK: [[IN:%[0-9]+]]:_(s32) = COPY $w1
; CHECK: [[FP:%[0-9]+]]:_(s64) = G_UITOFP [[IN]](s32)
; CHECK: G_STORE [[FP]](s64), [[ADDR]](p0)
define void @test_uitofp(double* %addr, i32 %in) {
  %fp = uitofp i32 %in to double
  store double %fp, double* %addr
  ret void
}

; CHECK-LABEL: name: test_fpext
; CHECK: [[IN:%[0-9]+]]:_(s32) = COPY $s0
; CHECK: [[RES:%[0-9]+]]:_(s64) = G_FPEXT [[IN]](s32)
; CHECK: $d0 = COPY [[RES]]
define double @test_fpext(float %in) {
  %res = fpext float %in to double
  ret double %res
}

; CHECK-LABEL: name: test_fptrunc
; CHECK: [[IN:%[0-9]+]]:_(s64) = COPY $d0
; CHECK: [[RES:%[0-9]+]]:_(s32) = G_FPTRUNC [[IN]](s64)
; CHECK: $s0 = COPY [[RES]]
define float @test_fptrunc(double %in) {
  %res = fptrunc double %in to float
  ret float %res
}

; CHECK-LABEL: name: test_constant_float
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY $x0
; CHECK: [[TMP:%[0-9]+]]:_(s32) = G_FCONSTANT float 1.500000e+00
; CHECK: G_STORE [[TMP]](s32), [[ADDR]](p0)
define void @test_constant_float(float* %addr) {
  store float 1.5, float* %addr
  ret void
}

; CHECK-LABEL: name: float_comparison
; CHECK: [[LHSADDR:%[0-9]+]]:_(p0) = COPY $x0
; CHECK: [[RHSADDR:%[0-9]+]]:_(p0) = COPY $x1
; CHECK: [[BOOLADDR:%[0-9]+]]:_(p0) = COPY $x2
; CHECK: [[LHS:%[0-9]+]]:_(s32) = G_LOAD [[LHSADDR]](p0)
; CHECK: [[RHS:%[0-9]+]]:_(s32) = G_LOAD [[RHSADDR]](p0)
; CHECK: [[TST:%[0-9]+]]:_(s1) = nnan ninf nsz arcp contract afn reassoc G_FCMP floatpred(oge), [[LHS]](s32), [[RHS]]
; CHECK: G_STORE [[TST]](s1), [[BOOLADDR]](p0)
define void @float_comparison(float* %a.addr, float* %b.addr, i1* %bool.addr) {
  %a = load float, float* %a.addr
  %b = load float, float* %b.addr
  %res = fcmp nnan ninf nsz arcp contract afn reassoc oge float %a, %b
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
; CHECK: $x0 = COPY [[TMP]](p0)

  ret i32* @var
}

@var1 = addrspace(42) global i32 0
define i32 addrspace(42)* @test_global_addrspace() {
; CHECK-LABEL: name: test_global
; CHECK: [[TMP:%[0-9]+]]:_(p42) = G_GLOBAL_VALUE @var1{{$}}
; CHECK: $x0 = COPY [[TMP]](p42)

  ret i32 addrspace(42)* @var1
}


define void()* @test_global_func() {
; CHECK-LABEL: name: test_global_func
; CHECK: [[TMP:%[0-9]+]]:_(p0) = G_GLOBAL_VALUE @allocai64{{$}}
; CHECK: $x0 = COPY [[TMP]](p0)

  ret void()* @allocai64
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64, i1)
define void @test_memcpy(i8* %dst, i8* %src, i64 %size) {
; CHECK-LABEL: name: test_memcpy
; CHECK: [[DST:%[0-9]+]]:_(p0) = COPY $x0
; CHECK: [[SRC:%[0-9]+]]:_(p0) = COPY $x1
; CHECK: [[SIZE:%[0-9]+]]:_(s64) = COPY $x2
; CHECK: G_MEMCPY [[DST]](p0), [[SRC]](p0), [[SIZE]](s64), 0 :: (store 1 into %ir.dst), (load 1 from %ir.src)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 %size, i1 0)
  ret void
}

define void @test_memcpy_tail(i8* %dst, i8* %src, i64 %size) {
; CHECK-LABEL: name: test_memcpy_tail
; CHECK: [[DST:%[0-9]+]]:_(p0) = COPY $x0
; CHECK: [[SRC:%[0-9]+]]:_(p0) = COPY $x1
; CHECK: [[SIZE:%[0-9]+]]:_(s64) = COPY $x2
; CHECK: G_MEMCPY [[DST]](p0), [[SRC]](p0), [[SIZE]](s64), 1 :: (store 1 into %ir.dst), (load 1 from %ir.src)
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 %size, i1 0)
  ret void
}

declare void @llvm.memcpy.p1i8.p1i8.i64(i8 addrspace(1)*, i8 addrspace(1)*, i64, i1)
define void @test_memcpy_nonzero_as(i8 addrspace(1)* %dst, i8 addrspace(1) * %src, i64 %size) {
; CHECK-LABEL: name: test_memcpy_nonzero_as
; CHECK: [[DST:%[0-9]+]]:_(p1) = COPY $x0
; CHECK: [[SRC:%[0-9]+]]:_(p1) = COPY $x1
; CHECK: [[SIZE:%[0-9]+]]:_(s64) = COPY $x2
; CHECK: G_MEMCPY [[DST]](p1), [[SRC]](p1), [[SIZE]](s64), 0 :: (store 1 into %ir.dst, addrspace 1), (load 1 from %ir.src, addrspace 1)
  call void @llvm.memcpy.p1i8.p1i8.i64(i8 addrspace(1)* %dst, i8 addrspace(1)* %src, i64 %size, i1 0)
  ret void
}

declare void @llvm.memmove.p0i8.p0i8.i64(i8*, i8*, i64, i1)
define void @test_memmove(i8* %dst, i8* %src, i64 %size) {
; CHECK-LABEL: name: test_memmove
; CHECK: [[DST:%[0-9]+]]:_(p0) = COPY $x0
; CHECK: [[SRC:%[0-9]+]]:_(p0) = COPY $x1
; CHECK: [[SIZE:%[0-9]+]]:_(s64) = COPY $x2
; CHECK: G_MEMMOVE [[DST]](p0), [[SRC]](p0), [[SIZE]](s64), 0 :: (store 1 into %ir.dst), (load 1 from %ir.src)
  call void @llvm.memmove.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 %size, i1 0)
  ret void
}

declare void @llvm.memset.p0i8.i64(i8*, i8, i64, i1)
define void @test_memset(i8* %dst, i8 %val, i64 %size) {
; CHECK-LABEL: name: test_memset
; CHECK: [[DST:%[0-9]+]]:_(p0) = COPY $x0
; CHECK: [[SRC_C:%[0-9]+]]:_(s32) = COPY $w1
; CHECK: [[SRC:%[0-9]+]]:_(s8) = G_TRUNC [[SRC_C]]
; CHECK: [[SIZE:%[0-9]+]]:_(s64) = COPY $x2
; CHECK: G_MEMSET [[DST]](p0), [[SRC]](s8), [[SIZE]](s64), 0 :: (store 1 into %ir.dst)
  call void @llvm.memset.p0i8.i64(i8* %dst, i8 %val, i64 %size, i1 0)
  ret void
}

define void @test_large_const(i128* %addr) {
; CHECK-LABEL: name: test_large_const
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY $x0
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
; CHECK: [[LIST:%[0-9]+]]:_(p0) = COPY $x0
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
; CHECK: [[LHS:%[0-9]+]]:_(s32) = COPY $s0
; CHECK: [[RHS:%[0-9]+]]:_(s32) = COPY $s1
; CHECK: [[RES:%[0-9]+]]:_(s32) = nnan ninf nsz arcp contract afn reassoc G_FPOW [[LHS]], [[RHS]]
; CHECK: $s0 = COPY [[RES]]
  %res = call nnan ninf nsz arcp contract afn reassoc float @llvm.pow.f32(float %l, float %r)
  ret float %res
}

declare float @llvm.powi.f32.i32(float, i32)
define float @test_powi_intrin(float %l, i32 %r) {
; CHECK-LABEL: name: test_powi_intrin
; CHECK: [[LHS:%[0-9]+]]:_(s32) = COPY $s0
; CHECK: [[RHS:%[0-9]+]]:_(s32) = COPY $w0
; CHECK: [[RES:%[0-9]+]]:_(s32) = nnan ninf nsz arcp contract afn reassoc G_FPOWI [[LHS]], [[RHS]]
; CHECK: $s0 = COPY [[RES]]
  %res = call nnan ninf nsz arcp contract afn reassoc float @llvm.powi.f32.i32(float %l, i32 %r)
  ret float %res
}

declare float @llvm.fma.f32(float, float, float)
define float @test_fma_intrin(float %a, float %b, float %c) {
; CHECK-LABEL: name: test_fma_intrin
; CHECK: [[A:%[0-9]+]]:_(s32) = COPY $s0
; CHECK: [[B:%[0-9]+]]:_(s32) = COPY $s1
; CHECK: [[C:%[0-9]+]]:_(s32) = COPY $s2
; CHECK: [[RES:%[0-9]+]]:_(s32) = nnan ninf nsz arcp contract afn reassoc G_FMA [[A]], [[B]], [[C]]
; CHECK: $s0 = COPY [[RES]]
  %res = call nnan ninf nsz arcp contract afn reassoc float @llvm.fma.f32(float %a, float %b, float %c)
  ret float %res
}

declare float @llvm.exp.f32(float)
define float @test_exp_intrin(float %a) {
; CHECK-LABEL: name: test_exp_intrin
; CHECK: [[A:%[0-9]+]]:_(s32) = COPY $s0
; CHECK: [[RES:%[0-9]+]]:_(s32) = nnan ninf nsz arcp contract afn reassoc G_FEXP [[A]]
; CHECK: $s0 = COPY [[RES]]
  %res = call nnan ninf nsz arcp contract afn reassoc float @llvm.exp.f32(float %a)
  ret float %res
}

declare float @llvm.exp2.f32(float)
define float @test_exp2_intrin(float %a) {
; CHECK-LABEL: name: test_exp2_intrin
; CHECK: [[A:%[0-9]+]]:_(s32) = COPY $s0
; CHECK: [[RES:%[0-9]+]]:_(s32) = nnan ninf nsz arcp contract afn reassoc G_FEXP2 [[A]]
; CHECK: $s0 = COPY [[RES]]
  %res = call nnan ninf nsz arcp contract afn reassoc float @llvm.exp2.f32(float %a)
  ret float %res
}

declare float @llvm.log.f32(float)
define float @test_log_intrin(float %a) {
; CHECK-LABEL: name: test_log_intrin
; CHECK: [[A:%[0-9]+]]:_(s32) = COPY $s0
; CHECK: [[RES:%[0-9]+]]:_(s32) = nnan ninf nsz arcp contract afn reassoc G_FLOG [[A]]
; CHECK: $s0 = COPY [[RES]]
  %res = call nnan ninf nsz arcp contract afn reassoc float @llvm.log.f32(float %a)
  ret float %res
}

declare float @llvm.log2.f32(float)
define float @test_log2_intrin(float %a) {
; CHECK-LABEL: name: test_log2_intrin
; CHECK: [[A:%[0-9]+]]:_(s32) = COPY $s0
; CHECK: [[RES:%[0-9]+]]:_(s32) = G_FLOG2 [[A]]
; CHECK: $s0 = COPY [[RES]]
  %res = call float @llvm.log2.f32(float %a)
  ret float %res
}

declare float @llvm.log10.f32(float)
define float @test_log10_intrin(float %a) {
; CHECK-LABEL: name: test_log10_intrin
; CHECK: [[A:%[0-9]+]]:_(s32) = COPY $s0
; CHECK: [[RES:%[0-9]+]]:_(s32) = nnan ninf nsz arcp contract afn reassoc G_FLOG10 [[A]]
; CHECK: $s0 = COPY [[RES]]
  %res = call nnan ninf nsz arcp contract afn reassoc float @llvm.log10.f32(float %a)
  ret float %res
}

declare float @llvm.fabs.f32(float)
define float @test_fabs_intrin(float %a) {
; CHECK-LABEL: name: test_fabs_intrin
; CHECK: [[A:%[0-9]+]]:_(s32) = COPY $s0
; CHECK: [[RES:%[0-9]+]]:_(s32) = nnan ninf nsz arcp contract afn reassoc G_FABS [[A]]
; CHECK: $s0 = COPY [[RES]]
  %res = call nnan ninf nsz arcp contract afn reassoc float @llvm.fabs.f32(float %a)
  ret float %res
}

declare float @llvm.copysign.f32(float, float)
define float @test_fcopysign_intrin(float %a, float %b) {
; CHECK-LABEL: name: test_fcopysign_intrin
; CHECK: [[A:%[0-9]+]]:_(s32) = COPY $s0
; CHECK: [[B:%[0-9]+]]:_(s32) = COPY $s1
; CHECK: [[RES:%[0-9]+]]:_(s32) = nnan ninf nsz arcp contract afn reassoc G_FCOPYSIGN [[A]], [[B]]
; CHECK: $s0 = COPY [[RES]]

  %res = call nnan ninf nsz arcp contract afn reassoc float @llvm.copysign.f32(float %a, float %b)
  ret float %res
}

declare float @llvm.canonicalize.f32(float)
define float @test_fcanonicalize_intrin(float %a) {
; CHECK-LABEL: name: test_fcanonicalize_intrin
; CHECK: [[A:%[0-9]+]]:_(s32) = COPY $s0
; CHECK: [[RES:%[0-9]+]]:_(s32) = nnan ninf nsz arcp contract afn reassoc G_FCANONICALIZE [[A]]
; CHECK: $s0 = COPY [[RES]]
  %res = call nnan ninf nsz arcp contract afn reassoc float @llvm.canonicalize.f32(float %a)
  ret float %res
}

declare float @llvm.trunc.f32(float)
define float @test_intrinsic_trunc(float %a) {
; CHECK-LABEL: name: test_intrinsic_trunc
; CHECK: [[A:%[0-9]+]]:_(s32) = COPY $s0
; CHECK: [[RES:%[0-9]+]]:_(s32) = G_INTRINSIC_TRUNC [[A]]
; CHECK: $s0 = COPY [[RES]]
  %res = call float @llvm.trunc.f32(float %a)
  ret float %res
}

declare float @llvm.round.f32(float)
define float @test_intrinsic_round(float %a) {
; CHECK-LABEL: name: test_intrinsic_round
; CHECK: [[A:%[0-9]+]]:_(s32) = COPY $s0
; CHECK: [[RES:%[0-9]+]]:_(s32) = G_INTRINSIC_ROUND [[A]]
; CHECK: $s0 = COPY [[RES]]
  %res = call float @llvm.round.f32(float %a)
  ret float %res
}

declare i32 @llvm.lrint.i32.f32(float)
define i32 @test_intrinsic_lrint(float %a) {
; CHECK-LABEL: name: test_intrinsic_lrint
; CHECK: [[A:%[0-9]+]]:_(s32) = COPY $s0
; CHECK: [[RES:%[0-9]+]]:_(s32) = G_INTRINSIC_LRINT [[A]]
; CHECK: $w0 = COPY [[RES]]
  %res = call i32 @llvm.lrint.i32.f32(float %a)
  ret i32 %res
}

declare i32 @llvm.ctlz.i32(i32, i1)
define i32 @test_ctlz_intrinsic_zero_not_undef(i32 %a) {
; CHECK-LABEL: name: test_ctlz_intrinsic_zero_not_undef
; CHECK: [[A:%[0-9]+]]:_(s32) = COPY $w0
; CHECK: [[RES:%[0-9]+]]:_(s32) = G_CTLZ [[A]]
; CHECK: $w0 = COPY [[RES]]
  %res = call i32 @llvm.ctlz.i32(i32 %a, i1 0)
  ret i32 %res
}

declare i32 @llvm.cttz.i32(i32, i1)
define i32 @test_cttz_intrinsic_zero_undef(i32 %a) {
; CHECK-LABEL: name: test_cttz_intrinsic_zero_undef
; CHECK: [[A:%[0-9]+]]:_(s32) = COPY $w0
; CHECK: [[RES:%[0-9]+]]:_(s32) = G_CTTZ_ZERO_UNDEF [[A]]
; CHECK: $w0 = COPY [[RES]]
  %res = call i32 @llvm.cttz.i32(i32 %a, i1 1)
  ret i32 %res
}

declare i32 @llvm.ctpop.i32(i32)
define i32 @test_ctpop_intrinsic(i32 %a) {
; CHECK-LABEL: name: test_ctpop
; CHECK: [[A:%[0-9]+]]:_(s32) = COPY $w0
; CHECK: [[RES:%[0-9]+]]:_(s32) = G_CTPOP [[A]]
; CHECK: $w0 = COPY [[RES]]
  %res = call i32 @llvm.ctpop.i32(i32 %a)
  ret i32 %res
}

declare i32 @llvm.bitreverse.i32(i32)
define i32 @test_bitreverse_intrinsic(i32 %a) {
; CHECK-LABEL: name: test_bitreverse
; CHECK: [[A:%[0-9]+]]:_(s32) = COPY $w0
; CHECK: [[RES:%[0-9]+]]:_(s32) = G_BITREVERSE [[A]]
; CHECK: $w0 = COPY [[RES]]
  %res = call i32 @llvm.bitreverse.i32(i32 %a)
  ret i32 %res
}

declare i32 @llvm.fshl.i32(i32, i32, i32)
define i32 @test_fshl_intrinsic(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: name: test_fshl_intrinsic
; CHECK: [[A:%[0-9]+]]:_(s32) = COPY $w0
; CHECK: [[B:%[0-9]+]]:_(s32) = COPY $w1
; CHECK: [[C:%[0-9]+]]:_(s32) = COPY $w2
; CHECK: [[RES:%[0-9]+]]:_(s32) = G_FSHL [[A]], [[B]], [[C]]
; CHECK: $w0 = COPY [[RES]]
  %res = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %c)
  ret i32 %res
}

declare i32 @llvm.fshr.i32(i32, i32, i32)
define i32 @test_fshr_intrinsic(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: name: test_fshr_intrinsic
; CHECK: [[A:%[0-9]+]]:_(s32) = COPY $w0
; CHECK: [[B:%[0-9]+]]:_(s32) = COPY $w1
; CHECK: [[C:%[0-9]+]]:_(s32) = COPY $w2
; CHECK: [[RES:%[0-9]+]]:_(s32) = G_FSHR [[A]], [[B]], [[C]]
; CHECK: $w0 = COPY [[RES]]
  %res = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %c)
  ret i32 %res
}

declare void @llvm.lifetime.start.p0i8(i64, i8*)
declare void @llvm.lifetime.end.p0i8(i64, i8*)
define void @test_lifetime_intrin() {
; CHECK-LABEL: name: test_lifetime_intrin
; CHECK: RET_ReallyLR
; O3-LABEL: name: test_lifetime_intrin
; O3: {{%[0-9]+}}:_(p0) = G_FRAME_INDEX %stack.0.slot
; O3-NEXT: LIFETIME_START %stack.0.slot
; O3-NEXT: LIFETIME_END %stack.0.slot
; O3-NEXT: RET_ReallyLR
  %slot = alloca i8, i32 4
  call void @llvm.lifetime.start.p0i8(i64 0, i8* %slot)
  call void @llvm.lifetime.end.p0i8(i64 0, i8* %slot)
  ret void
}

define void @test_load_store_atomics(i8* %addr) {
; CHECK-LABEL: name: test_load_store_atomics
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY $x0
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
; CHECK: [[ARG:%[0-9]+]]:_(s32) = COPY $s0
; CHECK: [[RES:%[0-9]+]]:_(s32) = G_FNEG [[ARG]]
; CHECK: $s0 = COPY [[RES]](s32)
  %neg = fneg float %x
  ret float %neg
}

define float @test_fneg_f32_fmf(float %x) {
; CHECK-LABEL: name: test_fneg_f32
; CHECK: [[ARG:%[0-9]+]]:_(s32) = COPY $s0
; CHECK: [[RES:%[0-9]+]]:_(s32) = nnan ninf nsz arcp contract afn reassoc G_FNEG [[ARG]]
; CHECK: $s0 = COPY [[RES]](s32)
  %neg = fneg fast float %x
  ret float %neg
}

define double @test_fneg_f64(double %x) {
; CHECK-LABEL: name: test_fneg_f64
; CHECK: [[ARG:%[0-9]+]]:_(s64) = COPY $d0
; CHECK: [[RES:%[0-9]+]]:_(s64) = G_FNEG [[ARG]]
; CHECK: $d0 = COPY [[RES]](s64)
  %neg = fneg double %x
  ret double %neg
}

define double @test_fneg_f64_fmf(double %x) {
; CHECK-LABEL: name: test_fneg_f64
; CHECK: [[ARG:%[0-9]+]]:_(s64) = COPY $d0
; CHECK: [[RES:%[0-9]+]]:_(s64) = nnan ninf nsz arcp contract afn reassoc G_FNEG [[ARG]]
; CHECK: $d0 = COPY [[RES]](s64)
  %neg = fneg fast double %x
  ret double %neg
}

define void @test_trivial_inlineasm() {
; CHECK-LABEL: name: test_trivial_inlineasm
; CHECK: INLINEASM &wibble, 1
; CHECK: INLINEASM &wibble, 0
  call void asm sideeffect "wibble", ""()
  call void asm "wibble", ""()
  ret void
}

define <2 x i32> @test_insertelement(<2 x i32> %vec, i32 %elt, i32 %idx){
; CHECK-LABEL: name: test_insertelement
; CHECK: [[VEC:%[0-9]+]]:_(<2 x s32>) = COPY $d0
; CHECK: [[ELT:%[0-9]+]]:_(s32) = COPY $w0
; CHECK: [[IDX:%[0-9]+]]:_(s32) = COPY $w1
; CHECK: [[RES:%[0-9]+]]:_(<2 x s32>) = G_INSERT_VECTOR_ELT [[VEC]], [[ELT]](s32), [[IDX]](s32)
; CHECK: $d0 = COPY [[RES]](<2 x s32>)
  %res = insertelement <2 x i32> %vec, i32 %elt, i32 %idx
  ret <2 x i32> %res
}

define i32 @test_extractelement(<2 x i32> %vec, i32 %idx) {
; CHECK-LABEL: name: test_extractelement
; CHECK: [[VEC:%[0-9]+]]:_(<2 x s32>) = COPY $d0
; CHECK: [[IDX:%[0-9]+]]:_(s32) = COPY $w0
; CHECK: [[IDXEXT:%[0-9]+]]:_(s64) = G_SEXT [[IDX]]
; CHECK: [[RES:%[0-9]+]]:_(s32) = G_EXTRACT_VECTOR_ELT [[VEC]](<2 x s32>), [[IDXEXT]](s64)
; CHECK: $w0 = COPY [[RES]](s32)
  %res = extractelement <2 x i32> %vec, i32 %idx
  ret i32 %res
}

define i32 @test_extractelement_const_idx(<2 x i32> %vec) {
; CHECK-LABEL: name: test_extractelement
; CHECK: [[VEC:%[0-9]+]]:_(<2 x s32>) = COPY $d0
; CHECK: [[IDX:%[0-9]+]]:_(s64) = G_CONSTANT i64 1
; CHECK: [[RES:%[0-9]+]]:_(s32) = G_EXTRACT_VECTOR_ELT [[VEC]](<2 x s32>), [[IDX]](s64)
; CHECK: $w0 = COPY [[RES]](s32)
  %res = extractelement <2 x i32> %vec, i32 1
  ret i32 %res
}

define i32 @test_singleelementvector(i32 %elt){
; CHECK-LABEL: name: test_singleelementvector
; CHECK: [[ELT:%[0-9]+]]:_(s32) = COPY $w0
; CHECK-NOT: G_INSERT_VECTOR_ELT
; CHECK-NOT: G_EXTRACT_VECTOR_ELT
; CHECK: $w0 = COPY [[ELT]](s32)
  %vec = insertelement <1 x i32> undef, i32 %elt, i32 0
  %res = extractelement <1 x i32> %vec, i32 0
  ret i32 %res
}

define <2 x i32> @test_constantaggzerovector_v2i32() {
; CHECK-LABEL: name: test_constantaggzerovector_v2i32
; CHECK: [[ZERO:%[0-9]+]]:_(s32) = G_CONSTANT i32 0
; CHECK: [[VEC:%[0-9]+]]:_(<2 x s32>) = G_BUILD_VECTOR [[ZERO]](s32), [[ZERO]](s32)
; CHECK: $d0 = COPY [[VEC]](<2 x s32>)
  ret <2 x i32> zeroinitializer
}

define <2 x float> @test_constantaggzerovector_v2f32() {
; CHECK-LABEL: name: test_constantaggzerovector_v2f32
; CHECK: [[ZERO:%[0-9]+]]:_(s32) = G_FCONSTANT float 0.000000e+00
; CHECK: [[VEC:%[0-9]+]]:_(<2 x s32>) = G_BUILD_VECTOR [[ZERO]](s32), [[ZERO]](s32)
; CHECK: $d0 = COPY [[VEC]](<2 x s32>)
  ret <2 x float> zeroinitializer
}

define i32 @test_constantaggzerovector_v3i32() {
; CHECK-LABEL: name: test_constantaggzerovector_v3i32
; CHECK: [[ZERO:%[0-9]+]]:_(s32) = G_CONSTANT i32 0
; CHECK: [[VEC:%[0-9]+]]:_(<3 x s32>) = G_BUILD_VECTOR [[ZERO]](s32), [[ZERO]](s32), [[ZERO]](s32)
; CHECK: G_EXTRACT_VECTOR_ELT [[VEC]](<3 x s32>)
  %elt = extractelement <3 x i32> zeroinitializer, i32 1
  ret i32 %elt
}

define <2 x i32> @test_constantdatavector_v2i32() {
; CHECK-LABEL: name: test_constantdatavector_v2i32
; CHECK: [[C1:%[0-9]+]]:_(s32) = G_CONSTANT i32 1
; CHECK: [[C2:%[0-9]+]]:_(s32) = G_CONSTANT i32 2
; CHECK: [[VEC:%[0-9]+]]:_(<2 x s32>) = G_BUILD_VECTOR [[C1]](s32), [[C2]](s32)
; CHECK: $d0 = COPY [[VEC]](<2 x s32>)
  ret <2 x i32> <i32 1, i32 2>
}

define i32 @test_constantdatavector_v3i32() {
; CHECK-LABEL: name: test_constantdatavector_v3i32
; CHECK: [[C1:%[0-9]+]]:_(s32) = G_CONSTANT i32 1
; CHECK: [[C2:%[0-9]+]]:_(s32) = G_CONSTANT i32 2
; CHECK: [[C3:%[0-9]+]]:_(s32) = G_CONSTANT i32 3
; CHECK: [[VEC:%[0-9]+]]:_(<3 x s32>) = G_BUILD_VECTOR [[C1]](s32), [[C2]](s32), [[C3]](s32)
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
; CHECK: [[VEC:%[0-9]+]]:_(<4 x s32>) = G_BUILD_VECTOR [[C1]](s32), [[C2]](s32), [[C3]](s32), [[C4]](s32)
; CHECK: $q0 = COPY [[VEC]](<4 x s32>)
  ret <4 x i32> <i32 1, i32 2, i32 3, i32 4>
}

define <2 x double> @test_constantdatavector_v2f64() {
; CHECK-LABEL: name: test_constantdatavector_v2f64
; CHECK: [[FC1:%[0-9]+]]:_(s64) = G_FCONSTANT double 1.000000e+00
; CHECK: [[FC2:%[0-9]+]]:_(s64) = G_FCONSTANT double 2.000000e+00
; CHECK: [[VEC:%[0-9]+]]:_(<2 x s64>) = G_BUILD_VECTOR [[FC1]](s64), [[FC2]](s64)
; CHECK: $q0 = COPY [[VEC]](<2 x s64>)
  ret <2 x double> <double 1.0, double 2.0>
}

define i32 @test_constantaggzerovector_v1s32(i32 %arg){
; CHECK-LABEL: name: test_constantaggzerovector_v1s32
; CHECK: [[ARG:%[0-9]+]]:_(s32) = COPY $w0
; CHECK: [[C0:%[0-9]+]]:_(s32) = G_CONSTANT i32 0
; CHECK-NOT: G_MERGE_VALUES
; CHECK: [[COPY:%[0-9]+]]:_(s32) = COPY [[C0]]
; CHECK-NOT: G_MERGE_VALUES
; CHECK: G_ADD [[ARG]], [[COPY]]
  %vec = insertelement <1 x i32> undef, i32 %arg, i32 0
  %add = add <1 x i32> %vec, zeroinitializer
  %res = extractelement <1 x i32> %add, i32 0
  ret i32 %res
}

define i32 @test_constantdatavector_v1s32(i32 %arg){
; CHECK-LABEL: name: test_constantdatavector_v1s32
; CHECK: [[ARG:%[0-9]+]]:_(s32) = COPY $w0
; CHECK: [[C1:%[0-9]+]]:_(s32) = G_CONSTANT i32 1
; CHECK-NOT: G_MERGE_VALUES
; CHECK: [[COPY:%[0-9]+]]:_(s32) = COPY [[C0]]
; CHECK-NOT: G_MERGE_VALUES
; CHECK: G_ADD [[ARG]], [[COPY]]
  %vec = insertelement <1 x i32> undef, i32 %arg, i32 0
  %add = add <1 x i32> %vec, <i32 1>
  %res = extractelement <1 x i32> %add, i32 0
  ret i32 %res
}

declare ghccc float @different_call_conv_target(float %x)
define float @test_different_call_conv_target(float %x) {
; CHECK-LABEL: name: test_different_call_conv
; CHECK: [[X:%[0-9]+]]:_(s32) = COPY $s0
; CHECK: $s8 = COPY [[X]]
; CHECK: BL @different_call_conv_target, csr_aarch64_noregs, implicit-def $lr, implicit $sp, implicit $s8, implicit-def $s0
  %res = call ghccc float @different_call_conv_target(float %x)
  ret float %res
}

define <2 x i32> @test_shufflevector_s32_v2s32(i32 %arg) {
; CHECK-LABEL: name: test_shufflevector_s32_v2s32
; CHECK: [[ARG:%[0-9]+]]:_(s32) = COPY $w0
; CHECK-DAG: [[UNDEF:%[0-9]+]]:_(s32) = G_IMPLICIT_DEF
; CHECK: [[VEC:%[0-9]+]]:_(<2 x s32>) = G_SHUFFLE_VECTOR [[ARG]](s32), [[UNDEF]], shufflemask(0, 0)
; CHECK: $d0 = COPY [[VEC]](<2 x s32>)
  %vec = insertelement <1 x i32> undef, i32 %arg, i32 0
  %res = shufflevector <1 x i32> %vec, <1 x i32> undef, <2 x i32> zeroinitializer
  ret <2 x i32> %res
}

define i32 @test_shufflevector_v2s32_s32(<2 x i32> %arg) {
; CHECK-LABEL: name: test_shufflevector_v2s32_s32
; CHECK: [[ARG:%[0-9]+]]:_(<2 x s32>) = COPY $d0
; CHECK: [[RES:%[0-9]+]]:_(s32) = G_SHUFFLE_VECTOR [[ARG]](<2 x s32>), [[UNDEF]], shufflemask(1)
; CHECK: $w0 = COPY [[RES]](s32)
  %vec = shufflevector <2 x i32> %arg, <2 x i32> undef, <1 x i32> <i32 1>
  %res = extractelement <1 x i32> %vec, i32 0
  ret i32 %res
}

define <2 x i32> @test_shufflevector_v2s32_v2s32_undef(<2 x i32> %arg) {
; CHECK-LABEL: name: test_shufflevector_v2s32_v2s32_undef
; CHECK: [[ARG:%[0-9]+]]:_(<2 x s32>) = COPY $d0
; CHECK-DAG: [[UNDEF:%[0-9]+]]:_(<2 x s32>) = G_IMPLICIT_DEF
; CHECK: [[VEC:%[0-9]+]]:_(<2 x s32>) = G_SHUFFLE_VECTOR [[ARG]](<2 x s32>), [[UNDEF]], shufflemask(undef, undef)
; CHECK: $d0 = COPY [[VEC]](<2 x s32>)
  %res = shufflevector <2 x i32> %arg, <2 x i32> undef, <2 x i32> undef
  ret <2 x i32> %res
}

define <2 x i32> @test_shufflevector_v2s32_v2s32_undef_0(<2 x i32> %arg) {
; CHECK-LABEL: name: test_shufflevector_v2s32_v2s32_undef_0
; CHECK: [[ARG:%[0-9]+]]:_(<2 x s32>) = COPY $d0
; CHECK-DAG: [[UNDEF:%[0-9]+]]:_(<2 x s32>) = G_IMPLICIT_DEF
; CHECK: [[VEC:%[0-9]+]]:_(<2 x s32>) = G_SHUFFLE_VECTOR [[ARG]](<2 x s32>), [[UNDEF]], shufflemask(undef, 0)
; CHECK: $d0 = COPY [[VEC]](<2 x s32>)
  %res = shufflevector <2 x i32> %arg, <2 x i32> undef, <2 x i32> <i32 undef, i32 0>
  ret <2 x i32> %res
}

define <2 x i32> @test_shufflevector_v2s32_v2s32_0_undef(<2 x i32> %arg) {
; CHECK-LABEL: name: test_shufflevector_v2s32_v2s32_0_undef
; CHECK: [[ARG:%[0-9]+]]:_(<2 x s32>) = COPY $d0
; CHECK-DAG: [[UNDEF:%[0-9]+]]:_(<2 x s32>) = G_IMPLICIT_DEF
; CHECK: [[VEC:%[0-9]+]]:_(<2 x s32>) = G_SHUFFLE_VECTOR [[ARG]](<2 x s32>), [[UNDEF]], shufflemask(0, undef)
; CHECK: $d0 = COPY [[VEC]](<2 x s32>)
  %res = shufflevector <2 x i32> %arg, <2 x i32> undef, <2 x i32> <i32 0, i32 undef>
  ret <2 x i32> %res
}

define i32 @test_shufflevector_v2s32_v3s32(<2 x i32> %arg) {
; CHECK-LABEL: name: test_shufflevector_v2s32_v3s32
; CHECK: [[ARG:%[0-9]+]]:_(<2 x s32>) = COPY $d0
; CHECK-DAG: [[UNDEF:%[0-9]+]]:_(<2 x s32>) = G_IMPLICIT_DEF
; CHECK: [[VEC:%[0-9]+]]:_(<3 x s32>) = G_SHUFFLE_VECTOR [[ARG]](<2 x s32>), [[UNDEF]], shufflemask(1, 0, 1)
; CHECK: G_EXTRACT_VECTOR_ELT [[VEC]](<3 x s32>)
  %vec = shufflevector <2 x i32> %arg, <2 x i32> undef, <3 x i32> <i32 1, i32 0, i32 1>
  %res = extractelement <3 x i32> %vec, i32 0
  ret i32 %res
}

define <4 x i32> @test_shufflevector_v2s32_v4s32(<2 x i32> %arg1, <2 x i32> %arg2) {
; CHECK-LABEL: name: test_shufflevector_v2s32_v4s32
; CHECK: [[ARG1:%[0-9]+]]:_(<2 x s32>) = COPY $d0
; CHECK: [[ARG2:%[0-9]+]]:_(<2 x s32>) = COPY $d1
; CHECK: [[VEC:%[0-9]+]]:_(<4 x s32>) = G_SHUFFLE_VECTOR [[ARG1]](<2 x s32>), [[ARG2]], shufflemask(0, 1, 2, 3)
; CHECK: $q0 = COPY [[VEC]](<4 x s32>)
  %res = shufflevector <2 x i32> %arg1, <2 x i32> %arg2, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x i32> %res
}

define <2 x i32> @test_shufflevector_v4s32_v2s32(<4 x i32> %arg) {
; CHECK-LABEL: name: test_shufflevector_v4s32_v2s32
; CHECK: [[ARG:%[0-9]+]]:_(<4 x s32>) = COPY $q0
; CHECK-DAG: [[UNDEF:%[0-9]+]]:_(<4 x s32>) = G_IMPLICIT_DEF
; CHECK: [[VEC:%[0-9]+]]:_(<2 x s32>) = G_SHUFFLE_VECTOR [[ARG]](<4 x s32>), [[UNDEF]], shufflemask(1, 3)
; CHECK: $d0 = COPY [[VEC]](<2 x s32>)
  %res = shufflevector <4 x i32> %arg, <4 x i32> undef, <2 x i32> <i32 1, i32 3>
  ret <2 x i32> %res
}


define <16 x i8> @test_shufflevector_v8s8_v16s8(<8 x i8> %arg1, <8 x i8> %arg2) {
; CHECK-LABEL: name: test_shufflevector_v8s8_v16s8
; CHECK: [[ARG1:%[0-9]+]]:_(<8 x s8>) = COPY $d0
; CHECK: [[ARG2:%[0-9]+]]:_(<8 x s8>) = COPY $d1
; CHECK: [[VEC:%[0-9]+]]:_(<16 x s8>) = G_SHUFFLE_VECTOR [[ARG1]](<8 x s8>), [[ARG2]], shufflemask(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15)
; CHECK: $q0 = COPY [[VEC]](<16 x s8>)
  %res = shufflevector <8 x i8> %arg1, <8 x i8> %arg2, <16 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11, i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  ret <16 x i8> %res
}

; CHECK-LABEL: test_constant_vector
; CHECK: [[UNDEF:%[0-9]+]]:_(s16) = G_IMPLICIT_DEF
; CHECK: [[F:%[0-9]+]]:_(s16) = G_FCONSTANT half 0xH3C00
; CHECK: [[M:%[0-9]+]]:_(<4 x s16>) = G_BUILD_VECTOR [[UNDEF]](s16), [[UNDEF]](s16), [[UNDEF]](s16), [[F]](s16)
; CHECK: $d0 = COPY [[M]](<4 x s16>)
define <4 x half> @test_constant_vector() {
  ret <4 x half> <half undef, half undef, half undef, half 0xH3C00>
}

define i32 @test_target_mem_intrinsic(i32* %addr) {
; CHECK-LABEL: name: test_target_mem_intrinsic
; CHECK: [[ADDR:%[0-9]+]]:_(p0) = COPY $x0
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


define i64 @test_phi_loop(i32 %n) {
; CHECK-LABEL: name: test_phi_loop
; CHECK: [[ARG1:%[0-9]+]]:_(s32) = COPY $w0
; CHECK: [[CST1:%[0-9]+]]:_(s32) = G_CONSTANT i32 1
; CHECK: [[CST2:%[0-9]+]]:_(s32) = G_CONSTANT i32 0
; CHECK: [[CST3:%[0-9]+]]:_(s64) = G_CONSTANT i64 0
; CHECK: [[CST4:%[0-9]+]]:_(s64) = G_CONSTANT i64 1

; CHECK: [[PN1:%[0-9]+]]:_(s32) = G_PHI [[ARG1]](s32), %bb.1, [[SUB:%[0-9]+]](s32), %bb.2
; CHECK: [[PN2:%[0-9]+]]:_(s64) = G_PHI [[CST3]](s64), %bb.1, [[PN3:%[0-9]+]](s64), %bb.2
; CHECK: [[PN3]]:_(s64) = G_PHI [[CST4]](s64), %bb.1, [[ADD:%[0-9]+]](s64), %bb.2
; CHECK: [[ADD]]:_(s64) = G_ADD [[PN2]], [[PN3]]
; CHECK: [[SUB]]:_(s32) = G_SUB [[PN1]], [[CST1]]
; CHECK: [[CMP:%[0-9]+]]:_(s1) = G_ICMP intpred(sle), [[PN1]](s32), [[CST2]]
; CHECK: G_BRCOND [[CMP]](s1), %bb.3
; CHECK: G_BR %bb.2

; CHECK: $x0 = COPY [[PN2]](s64)
; CHECK: RET_ReallyLR implicit $x0
entry:
  br label %loop

loop:
  %counter = phi i32 [ %n, %entry ], [ %counter.dec, %loop ]
  %elem = phi { i64, i64 } [ { i64 0, i64 1 }, %entry ], [ %updated, %loop ]
  %prev = extractvalue { i64, i64 } %elem, 0
  %curr = extractvalue { i64, i64 } %elem, 1
  %next = add i64 %prev, %curr
  %shifted = insertvalue { i64, i64 } %elem, i64 %curr, 0
  %updated = insertvalue { i64, i64 } %shifted, i64 %next, 1
  %counter.dec = sub i32 %counter, 1
  %cond = icmp sle i32 %counter, 0
  br i1 %cond, label %exit, label %loop

exit:
  %res = extractvalue { i64, i64 } %elem, 0
  ret i64 %res
}

define void @test_phi_diamond({ i8, i16, i32 }* %a.ptr, { i8, i16, i32 }* %b.ptr, i1 %selector, { i8, i16, i32 }* %dst) {
; CHECK-LABEL: name: test_phi_diamond
; CHECK: [[ARG1:%[0-9]+]]:_(p0) = COPY $x0
; CHECK: [[ARG2:%[0-9]+]]:_(p0) = COPY $x1
; CHECK: [[ARG3:%[0-9]+]]:_(s32) = COPY $w2
; CHECK: [[TRUNC:%[0-9]+]]:_(s1) = G_TRUNC [[ARG3]](s32)
; CHECK: [[ARG4:%[0-9]+]]:_(p0) = COPY $x3
; CHECK: G_BRCOND [[TRUNC]](s1), %bb.2
; CHECK: G_BR %bb.3

; CHECK: [[LD1:%[0-9]+]]:_(s8) = G_LOAD [[ARG1]](p0) :: (load 1 from %ir.a.ptr, align 4)
; CHECK: [[CST1:%[0-9]+]]:_(s64) = G_CONSTANT i64 2
; CHECK: [[GEP1:%[0-9]+]]:_(p0) = G_PTR_ADD [[ARG1]], [[CST1]](s64)
; CHECK: [[LD2:%[0-9]+]]:_(s16) = G_LOAD [[GEP1]](p0) :: (load 2 from %ir.a.ptr + 2)
; CHECK: [[CST2:%[0-9]+]]:_(s64) = G_CONSTANT i64 4
; CHECK: [[GEP2:%[0-9]+]]:_(p0) = G_PTR_ADD [[ARG1]], [[CST2]](s64)
; CHECK: [[LD3:%[0-9]+]]:_(s32) = G_LOAD [[GEP2]](p0) :: (load 4 from %ir.a.ptr + 4)
; CHECK: G_BR %bb.4

; CHECK: [[LD4:%[0-9]+]]:_(s8) = G_LOAD [[ARG2]](p0) :: (load 1 from %ir.b.ptr, align 4)
; CHECK: [[CST3:%[0-9]+]]:_(s64) = G_CONSTANT i64 2
; CHECK: [[GEP3:%[0-9]+]]:_(p0) = G_PTR_ADD [[ARG2]], [[CST3]](s64)
; CHECK: [[LD5:%[0-9]+]]:_(s16) = G_LOAD [[GEP3]](p0) :: (load 2 from %ir.b.ptr + 2)
; CHECK: [[CST4:%[0-9]+]]:_(s64) = G_CONSTANT i64 4
; CHECK: [[GEP4:%[0-9]+]]:_(p0) = G_PTR_ADD [[ARG2]], [[CST4]](s64)
; CHECK: [[LD6:%[0-9]+]]:_(s32) = G_LOAD [[GEP4]](p0) :: (load 4 from %ir.b.ptr + 4)

; CHECK: [[PN1:%[0-9]+]]:_(s8) = G_PHI [[LD1]](s8), %bb.2, [[LD4]](s8), %bb.3
; CHECK: [[PN2:%[0-9]+]]:_(s16) = G_PHI [[LD2]](s16), %bb.2, [[LD5]](s16), %bb.3
; CHECK: [[PN3:%[0-9]+]]:_(s32) = G_PHI [[LD3]](s32), %bb.2, [[LD6]](s32), %bb.3
; CHECK: G_STORE [[PN1]](s8), [[ARG4]](p0) :: (store 1 into %ir.dst, align 4)
; CHECK: [[CST5:%[0-9]+]]:_(s64) = G_CONSTANT i64 2
; CHECK: [[GEP5:%[0-9]+]]:_(p0) = G_PTR_ADD [[ARG4]], [[CST5]](s64)
; CHECK: G_STORE [[PN2]](s16), [[GEP5]](p0) :: (store 2 into %ir.dst + 2)
; CHECK: [[CST6:%[0-9]+]]:_(s64) = G_CONSTANT i64 4
; CHECK: [[GEP6:%[0-9]+]]:_(p0) = G_PTR_ADD [[ARG4]], [[CST6]](s64)
; CHECK: G_STORE [[PN3]](s32), [[GEP6]](p0) :: (store 4 into %ir.dst + 4)
; CHECK: RET_ReallyLR

entry:
  br i1 %selector, label %store.a, label %store.b

store.a:
  %a = load { i8, i16, i32 }, { i8, i16, i32 }* %a.ptr
  br label %join

store.b:
  %b = load { i8, i16, i32 }, { i8, i16, i32 }* %b.ptr
  br label %join

join:
  %v = phi { i8, i16, i32 } [ %a, %store.a ], [ %b, %store.b ]
  store { i8, i16, i32 } %v, { i8, i16, i32 }* %dst
  ret void
}

%agg.inner.inner = type {i64, i64}
%agg.inner = type {i16, i8, %agg.inner.inner }
%agg.nested = type {i32, i32, %agg.inner, i32}

define void @test_nested_aggregate_const(%agg.nested *%ptr) {
; CHECK-LABEL: name: test_nested_aggregate_const
; CHECK: [[BASE:%[0-9]+]]:_(p0) = COPY $x0
; CHECK: [[CST1:%[0-9]+]]:_(s32) = G_CONSTANT i32 1
; CHECK: [[CST2:%[0-9]+]]:_(s16) = G_CONSTANT i16 2
; CHECK: [[CST3:%[0-9]+]]:_(s8) = G_CONSTANT i8 3
; CHECK: [[CST4:%[0-9]+]]:_(s64) = G_CONSTANT i64 5
; CHECK: [[CST5:%[0-9]+]]:_(s64) = G_CONSTANT i64 8
; CHECK: [[CST6:%[0-9]+]]:_(s32) = G_CONSTANT i32 13
; CHECK: G_STORE [[CST1]](s32), [[BASE]](p0) :: (store 4 into %ir.ptr, align 8)
; CHECK: [[CST7:%[0-9]+]]:_(s64) = G_CONSTANT i64 4
; CHECK: [[GEP1:%[0-9]+]]:_(p0) = G_PTR_ADD [[BASE]], [[CST7]](s64)
; CHECK: G_STORE [[CST1]](s32), [[GEP1]](p0) :: (store 4 into %ir.ptr + 4)
; CHECK: [[CST8:%[0-9]+]]:_(s64) = G_CONSTANT i64 8
; CHECK: [[GEP2:%[0-9]+]]:_(p0) = G_PTR_ADD [[BASE]], [[CST8]](s64)
; CHECK: G_STORE [[CST2]](s16), [[GEP2]](p0) :: (store 2 into %ir.ptr + 8, align 8)
; CHECK: [[CST9:%[0-9]+]]:_(s64) = G_CONSTANT i64 10
; CHECK: [[GEP3:%[0-9]+]]:_(p0) = G_PTR_ADD [[BASE]], [[CST9]](s64)
; CHECK: G_STORE [[CST3]](s8), [[GEP3]](p0) :: (store 1 into %ir.ptr + 10, align 2)
; CHECK: [[CST10:%[0-9]+]]:_(s64) = G_CONSTANT i64 16
; CHECK: [[GEP4:%[0-9]+]]:_(p0) = G_PTR_ADD [[BASE]], [[CST10]](s64)
; CHECK: G_STORE [[CST4]](s64), [[GEP4]](p0) :: (store 8 into %ir.ptr + 16)
; CHECK: [[CST11:%[0-9]+]]:_(s64) = G_CONSTANT i64 24
; CHECK: [[GEP5:%[0-9]+]]:_(p0) = G_PTR_ADD [[BASE]], [[CST11]](s64)
; CHECK: G_STORE [[CST5]](s64), [[GEP5]](p0) :: (store 8 into %ir.ptr + 24)
; CHECK: [[CST12:%[0-9]+]]:_(s64) = G_CONSTANT i64 32
; CHECK: [[GEP6:%[0-9]+]]:_(p0) = G_PTR_ADD [[BASE]], [[CST12]](s64)
; CHECK: G_STORE [[CST6]](s32), [[GEP6]](p0) :: (store 4 into %ir.ptr + 32, align 8)
  store %agg.nested { i32 1, i32 1, %agg.inner { i16 2, i8 3, %agg.inner.inner {i64 5, i64 8} }, i32 13}, %agg.nested *%ptr
  ret void
}

define i1 @return_i1_zext() {
; AAPCS ABI says that booleans can only be 1 or 0, so we need to zero-extend.
; CHECK-LABEL: name: return_i1_zext
; CHECK: [[CST:%[0-9]+]]:_(s1) = G_CONSTANT i1 true
; CHECK: [[ZEXT:%[0-9]+]]:_(s8) = G_ZEXT [[CST]](s1)
; CHECK: [[ANYEXT:%[0-9]+]]:_(s32) = G_ANYEXT [[ZEXT]](s8)
; CHECK: $w0 = COPY [[ANYEXT]](s32)
; CHECK: RET_ReallyLR implicit $w0
  ret i1 true
}

; Try one cmpxchg
define i32 @test_atomic_cmpxchg_1(i32* %addr) {
; CHECK-LABEL: name: test_atomic_cmpxchg_1
; CHECK:       bb.1.entry:
; CHECK-NEXT:  successors: %bb.{{[^)]+}}
; CHECK-NEXT:  liveins: $x0
; CHECK:         [[ADDR:%[0-9]+]]:_(p0) = COPY $x0
; CHECK-NEXT:    [[OLDVAL:%[0-9]+]]:_(s32) = G_CONSTANT i32 0
; CHECK-NEXT:    [[NEWVAL:%[0-9]+]]:_(s32) = G_CONSTANT i32 1
; CHECK:       bb.2.repeat:
; CHECK-NEXT:    successors: %bb.3({{[^)]+}}), %bb.2({{[^)]+}})
; CHECK:         [[OLDVALRES:%[0-9]+]]:_(s32), [[SUCCESS:%[0-9]+]]:_(s1) = G_ATOMIC_CMPXCHG_WITH_SUCCESS [[ADDR]](p0), [[OLDVAL]], [[NEWVAL]] :: (load store monotonic monotonic 4 on %ir.addr)
; CHECK-NEXT:    G_BRCOND [[SUCCESS]](s1), %bb.3
; CHECK-NEXT:    G_BR %bb.2
; CHECK:       bb.3.done:
entry:
  br label %repeat
repeat:
  %val_success = cmpxchg i32* %addr, i32 0, i32 1 monotonic monotonic
  %value_loaded = extractvalue { i32, i1 } %val_success, 0
  %success = extractvalue { i32, i1 } %val_success, 1
  br i1 %success, label %done, label %repeat
done:
  ret i32 %value_loaded
}

; Try one cmpxchg
define i32 @test_weak_atomic_cmpxchg_1(i32* %addr) {
; CHECK-LABEL: name: test_weak_atomic_cmpxchg_1
; CHECK:       bb.1.entry:
; CHECK-NEXT:  successors: %bb.{{[^)]+}}
; CHECK-NEXT:  liveins: $x0
; CHECK:         [[ADDR:%[0-9]+]]:_(p0) = COPY $x0
; CHECK-NEXT:    [[OLDVAL:%[0-9]+]]:_(s32) = G_CONSTANT i32 0
; CHECK-NEXT:    [[NEWVAL:%[0-9]+]]:_(s32) = G_CONSTANT i32 1
; CHECK:       bb.2.repeat:
; CHECK-NEXT:    successors: %bb.3({{[^)]+}}), %bb.2({{[^)]+}})
; CHECK:         [[OLDVALRES:%[0-9]+]]:_(s32), [[SUCCESS:%[0-9]+]]:_(s1) = G_ATOMIC_CMPXCHG_WITH_SUCCESS [[ADDR]](p0), [[OLDVAL]], [[NEWVAL]] :: (load store monotonic monotonic 4 on %ir.addr)
; CHECK-NEXT:    G_BRCOND [[SUCCESS]](s1), %bb.3
; CHECK-NEXT:    G_BR %bb.2
; CHECK:       bb.3.done:
entry:
  br label %repeat
repeat:
  %val_success = cmpxchg weak i32* %addr, i32 0, i32 1 monotonic monotonic
  %value_loaded = extractvalue { i32, i1 } %val_success, 0
  %success = extractvalue { i32, i1 } %val_success, 1
  br i1 %success, label %done, label %repeat
done:
  ret i32 %value_loaded
}

; Try one cmpxchg with a small type and high atomic ordering.
define i16 @test_atomic_cmpxchg_2(i16* %addr) {
; CHECK-LABEL: name: test_atomic_cmpxchg_2
; CHECK:       bb.1.entry:
; CHECK-NEXT:  successors: %bb.2({{[^)]+}})
; CHECK-NEXT:  liveins: $x0
; CHECK:         [[ADDR:%[0-9]+]]:_(p0) = COPY $x0
; CHECK-NEXT:    [[OLDVAL:%[0-9]+]]:_(s16) = G_CONSTANT i16 0
; CHECK-NEXT:    [[NEWVAL:%[0-9]+]]:_(s16) = G_CONSTANT i16 1
; CHECK:       bb.2.repeat:
; CHECK-NEXT:    successors: %bb.3({{[^)]+}}), %bb.2({{[^)]+}})
; CHECK:         [[OLDVALRES:%[0-9]+]]:_(s16), [[SUCCESS:%[0-9]+]]:_(s1) = G_ATOMIC_CMPXCHG_WITH_SUCCESS [[ADDR]](p0), [[OLDVAL]], [[NEWVAL]] :: (load store seq_cst seq_cst 2 on %ir.addr)
; CHECK-NEXT:    G_BRCOND [[SUCCESS]](s1), %bb.3
; CHECK-NEXT:    G_BR %bb.2
; CHECK:       bb.3.done:
entry:
  br label %repeat
repeat:
  %val_success = cmpxchg i16* %addr, i16 0, i16 1 seq_cst seq_cst
  %value_loaded = extractvalue { i16, i1 } %val_success, 0
  %success = extractvalue { i16, i1 } %val_success, 1
  br i1 %success, label %done, label %repeat
done:
  ret i16 %value_loaded
}

; Try one cmpxchg where the success order and failure order differ.
define i64 @test_atomic_cmpxchg_3(i64* %addr) {
; CHECK-LABEL: name: test_atomic_cmpxchg_3
; CHECK:       bb.1.entry:
; CHECK-NEXT:  successors: %bb.2({{[^)]+}})
; CHECK-NEXT:  liveins: $x0
; CHECK:         [[ADDR:%[0-9]+]]:_(p0) = COPY $x0
; CHECK-NEXT:    [[OLDVAL:%[0-9]+]]:_(s64) = G_CONSTANT i64 0
; CHECK-NEXT:    [[NEWVAL:%[0-9]+]]:_(s64) = G_CONSTANT i64 1
; CHECK:       bb.2.repeat:
; CHECK-NEXT:    successors: %bb.3({{[^)]+}}), %bb.2({{[^)]+}})
; CHECK:         [[OLDVALRES:%[0-9]+]]:_(s64), [[SUCCESS:%[0-9]+]]:_(s1) = G_ATOMIC_CMPXCHG_WITH_SUCCESS [[ADDR]](p0), [[OLDVAL]], [[NEWVAL]] :: (load store seq_cst acquire 8 on %ir.addr)
; CHECK-NEXT:    G_BRCOND [[SUCCESS]](s1), %bb.3
; CHECK-NEXT:    G_BR %bb.2
; CHECK:       bb.3.done:
entry:
  br label %repeat
repeat:
  %val_success = cmpxchg i64* %addr, i64 0, i64 1 seq_cst acquire
  %value_loaded = extractvalue { i64, i1 } %val_success, 0
  %success = extractvalue { i64, i1 } %val_success, 1
  br i1 %success, label %done, label %repeat
done:
  ret i64 %value_loaded
}

; Try a monotonic atomicrmw xchg
; AArch64 will expand some atomicrmw's at the LLVM-IR level so we use a wide type to avoid this.
define i32 @test_atomicrmw_xchg(i256* %addr) {
; CHECK-LABEL: name: test_atomicrmw_xchg
; CHECK:       bb.1 (%ir-block.{{[0-9]+}}):
; CHECK-NEXT:  liveins: $x0
; CHECK:         [[ADDR:%[0-9]+]]:_(p0) = COPY $x0
; CHECK-NEXT:    [[VAL:%[0-9]+]]:_(s256) = G_CONSTANT i256 1
; CHECK-NEXT:    [[OLDVALRES:%[0-9]+]]:_(s256) = G_ATOMICRMW_XCHG [[ADDR]](p0), [[VAL]] :: (load store monotonic 32 on %ir.addr)
; CHECK-NEXT:    [[RES:%[0-9]+]]:_(s32) = G_TRUNC [[OLDVALRES]]
  %oldval = atomicrmw xchg i256* %addr, i256 1 monotonic
  ; FIXME: We currently can't lower 'ret i256' and it's not the purpose of this
  ;        test so work around it by truncating to i32 for now.
  %oldval.trunc = trunc i256 %oldval to i32
  ret i32 %oldval.trunc
}

; Try an acquire atomicrmw add
; AArch64 will expand some atomicrmw's at the LLVM-IR level so we use a wide type to avoid this.
define i32 @test_atomicrmw_add(i256* %addr) {
; CHECK-LABEL: name: test_atomicrmw_add
; CHECK:       bb.1 (%ir-block.{{[0-9]+}}):
; CHECK-NEXT:  liveins: $x0
; CHECK:         [[ADDR:%[0-9]+]]:_(p0) = COPY $x0
; CHECK-NEXT:    [[VAL:%[0-9]+]]:_(s256) = G_CONSTANT i256 1
; CHECK-NEXT:    [[OLDVALRES:%[0-9]+]]:_(s256) = G_ATOMICRMW_ADD [[ADDR]](p0), [[VAL]] :: (load store acquire 32 on %ir.addr)
; CHECK-NEXT:    [[RES:%[0-9]+]]:_(s32) = G_TRUNC [[OLDVALRES]]
  %oldval = atomicrmw add i256* %addr, i256 1 acquire
  ; FIXME: We currently can't lower 'ret i256' and it's not the purpose of this
  ;        test so work around it by truncating to i32 for now.
  %oldval.trunc = trunc i256 %oldval to i32
  ret i32 %oldval.trunc
}

; Try a release atomicrmw sub
; AArch64 will expand some atomicrmw's at the LLVM-IR level so we use a wide type to avoid this.
define i32 @test_atomicrmw_sub(i256* %addr) {
; CHECK-LABEL: name: test_atomicrmw_sub
; CHECK:       bb.1 (%ir-block.{{[0-9]+}}):
; CHECK-NEXT:  liveins: $x0
; CHECK:         [[ADDR:%[0-9]+]]:_(p0) = COPY $x0
; CHECK-NEXT:    [[VAL:%[0-9]+]]:_(s256) = G_CONSTANT i256 1
; CHECK-NEXT:    [[OLDVALRES:%[0-9]+]]:_(s256) = G_ATOMICRMW_SUB [[ADDR]](p0), [[VAL]] :: (load store release 32 on %ir.addr)
; CHECK-NEXT:    [[RES:%[0-9]+]]:_(s32) = G_TRUNC [[OLDVALRES]]
  %oldval = atomicrmw sub i256* %addr, i256 1 release
  ; FIXME: We currently can't lower 'ret i256' and it's not the purpose of this
  ;        test so work around it by truncating to i32 for now.
  %oldval.trunc = trunc i256 %oldval to i32
  ret i32 %oldval.trunc
}

; Try an acq_rel atomicrmw and
; AArch64 will expand some atomicrmw's at the LLVM-IR level so we use a wide type to avoid this.
define i32 @test_atomicrmw_and(i256* %addr) {
; CHECK-LABEL: name: test_atomicrmw_and
; CHECK:       bb.1 (%ir-block.{{[0-9]+}}):
; CHECK-NEXT:  liveins: $x0
; CHECK:         [[ADDR:%[0-9]+]]:_(p0) = COPY $x0
; CHECK-NEXT:    [[VAL:%[0-9]+]]:_(s256) = G_CONSTANT i256 1
; CHECK-NEXT:    [[OLDVALRES:%[0-9]+]]:_(s256) = G_ATOMICRMW_AND [[ADDR]](p0), [[VAL]] :: (load store acq_rel 32 on %ir.addr)
; CHECK-NEXT:    [[RES:%[0-9]+]]:_(s32) = G_TRUNC [[OLDVALRES]]
  %oldval = atomicrmw and i256* %addr, i256 1 acq_rel
  ; FIXME: We currently can't lower 'ret i256' and it's not the purpose of this
  ;        test so work around it by truncating to i32 for now.
  %oldval.trunc = trunc i256 %oldval to i32
  ret i32 %oldval.trunc
}

; Try an seq_cst atomicrmw nand
; AArch64 will expand some atomicrmw's at the LLVM-IR level so we use a wide type to avoid this.
define i32 @test_atomicrmw_nand(i256* %addr) {
; CHECK-LABEL: name: test_atomicrmw_nand
; CHECK:       bb.1 (%ir-block.{{[0-9]+}}):
; CHECK-NEXT:  liveins: $x0
; CHECK:         [[ADDR:%[0-9]+]]:_(p0) = COPY $x0
; CHECK-NEXT:    [[VAL:%[0-9]+]]:_(s256) = G_CONSTANT i256 1
; CHECK-NEXT:    [[OLDVALRES:%[0-9]+]]:_(s256) = G_ATOMICRMW_NAND [[ADDR]](p0), [[VAL]] :: (load store seq_cst 32 on %ir.addr)
; CHECK-NEXT:    [[RES:%[0-9]+]]:_(s32) = G_TRUNC [[OLDVALRES]]
  %oldval = atomicrmw nand i256* %addr, i256 1 seq_cst
  ; FIXME: We currently can't lower 'ret i256' and it's not the purpose of this
  ;        test so work around it by truncating to i32 for now.
  %oldval.trunc = trunc i256 %oldval to i32
  ret i32 %oldval.trunc
}

; Try an seq_cst atomicrmw or
; AArch64 will expand some atomicrmw's at the LLVM-IR level so we use a wide type to avoid this.
define i32 @test_atomicrmw_or(i256* %addr) {
; CHECK-LABEL: name: test_atomicrmw_or
; CHECK:       bb.1 (%ir-block.{{[0-9]+}}):
; CHECK-NEXT:  liveins: $x0
; CHECK:         [[ADDR:%[0-9]+]]:_(p0) = COPY $x0
; CHECK-NEXT:    [[VAL:%[0-9]+]]:_(s256) = G_CONSTANT i256 1
; CHECK-NEXT:    [[OLDVALRES:%[0-9]+]]:_(s256) = G_ATOMICRMW_OR [[ADDR]](p0), [[VAL]] :: (load store seq_cst 32 on %ir.addr)
; CHECK-NEXT:    [[RES:%[0-9]+]]:_(s32) = G_TRUNC [[OLDVALRES]]
  %oldval = atomicrmw or i256* %addr, i256 1 seq_cst
  ; FIXME: We currently can't lower 'ret i256' and it's not the purpose of this
  ;        test so work around it by truncating to i32 for now.
  %oldval.trunc = trunc i256 %oldval to i32
  ret i32 %oldval.trunc
}

; Try an seq_cst atomicrmw xor
; AArch64 will expand some atomicrmw's at the LLVM-IR level so we use a wide type to avoid this.
define i32 @test_atomicrmw_xor(i256* %addr) {
; CHECK-LABEL: name: test_atomicrmw_xor
; CHECK:       bb.1 (%ir-block.{{[0-9]+}}):
; CHECK-NEXT:  liveins: $x0
; CHECK:         [[ADDR:%[0-9]+]]:_(p0) = COPY $x0
; CHECK-NEXT:    [[VAL:%[0-9]+]]:_(s256) = G_CONSTANT i256 1
; CHECK-NEXT:    [[OLDVALRES:%[0-9]+]]:_(s256) = G_ATOMICRMW_XOR [[ADDR]](p0), [[VAL]] :: (load store seq_cst 32 on %ir.addr)
; CHECK-NEXT:    [[RES:%[0-9]+]]:_(s32) = G_TRUNC [[OLDVALRES]]
  %oldval = atomicrmw xor i256* %addr, i256 1 seq_cst
  ; FIXME: We currently can't lower 'ret i256' and it's not the purpose of this
  ;        test so work around it by truncating to i32 for now.
  %oldval.trunc = trunc i256 %oldval to i32
  ret i32 %oldval.trunc
}

; Try an seq_cst atomicrmw min
; AArch64 will expand some atomicrmw's at the LLVM-IR level so we use a wide type to avoid this.
define i32 @test_atomicrmw_min(i256* %addr) {
; CHECK-LABEL: name: test_atomicrmw_min
; CHECK:       bb.1 (%ir-block.{{[0-9]+}}):
; CHECK-NEXT:  liveins: $x0
; CHECK:         [[ADDR:%[0-9]+]]:_(p0) = COPY $x0
; CHECK-NEXT:    [[VAL:%[0-9]+]]:_(s256) = G_CONSTANT i256 1
; CHECK-NEXT:    [[OLDVALRES:%[0-9]+]]:_(s256) = G_ATOMICRMW_MIN [[ADDR]](p0), [[VAL]] :: (load store seq_cst 32 on %ir.addr)
; CHECK-NEXT:    [[RES:%[0-9]+]]:_(s32) = G_TRUNC [[OLDVALRES]]
  %oldval = atomicrmw min i256* %addr, i256 1 seq_cst
  ; FIXME: We currently can't lower 'ret i256' and it's not the purpose of this
  ;        test so work around it by truncating to i32 for now.
  %oldval.trunc = trunc i256 %oldval to i32
  ret i32 %oldval.trunc
}

; Try an seq_cst atomicrmw max
; AArch64 will expand some atomicrmw's at the LLVM-IR level so we use a wide type to avoid this.
define i32 @test_atomicrmw_max(i256* %addr) {
; CHECK-LABEL: name: test_atomicrmw_max
; CHECK:       bb.1 (%ir-block.{{[0-9]+}}):
; CHECK-NEXT:  liveins: $x0
; CHECK:         [[ADDR:%[0-9]+]]:_(p0) = COPY $x0
; CHECK-NEXT:    [[VAL:%[0-9]+]]:_(s256) = G_CONSTANT i256 1
; CHECK-NEXT:    [[OLDVALRES:%[0-9]+]]:_(s256) = G_ATOMICRMW_MAX [[ADDR]](p0), [[VAL]] :: (load store seq_cst 32 on %ir.addr)
; CHECK-NEXT:    [[RES:%[0-9]+]]:_(s32) = G_TRUNC [[OLDVALRES]]
  %oldval = atomicrmw max i256* %addr, i256 1 seq_cst
  ; FIXME: We currently can't lower 'ret i256' and it's not the purpose of this
  ;        test so work around it by truncating to i32 for now.
  %oldval.trunc = trunc i256 %oldval to i32
  ret i32 %oldval.trunc
}

; Try an seq_cst atomicrmw unsigned min
; AArch64 will expand some atomicrmw's at the LLVM-IR level so we use a wide type to avoid this.
define i32 @test_atomicrmw_umin(i256* %addr) {
; CHECK-LABEL: name: test_atomicrmw_umin
; CHECK:       bb.1 (%ir-block.{{[0-9]+}}):
; CHECK-NEXT:  liveins: $x0
; CHECK:         [[ADDR:%[0-9]+]]:_(p0) = COPY $x0
; CHECK-NEXT:    [[VAL:%[0-9]+]]:_(s256) = G_CONSTANT i256 1
; CHECK-NEXT:    [[OLDVALRES:%[0-9]+]]:_(s256) = G_ATOMICRMW_UMIN [[ADDR]](p0), [[VAL]] :: (load store seq_cst 32 on %ir.addr)
; CHECK-NEXT:    [[RES:%[0-9]+]]:_(s32) = G_TRUNC [[OLDVALRES]]
  %oldval = atomicrmw umin i256* %addr, i256 1 seq_cst
  ; FIXME: We currently can't lower 'ret i256' and it's not the purpose of this
  ;        test so work around it by truncating to i32 for now.
  %oldval.trunc = trunc i256 %oldval to i32
  ret i32 %oldval.trunc
}

; Try an seq_cst atomicrmw unsigned max
; AArch64 will expand some atomicrmw's at the LLVM-IR level so we use a wide type to avoid this.
define i32 @test_atomicrmw_umax(i256* %addr) {
; CHECK-LABEL: name: test_atomicrmw_umax
; CHECK:       bb.1 (%ir-block.{{[0-9]+}}):
; CHECK-NEXT:  liveins: $x0
; CHECK:         [[ADDR:%[0-9]+]]:_(p0) = COPY $x0
; CHECK-NEXT:    [[VAL:%[0-9]+]]:_(s256) = G_CONSTANT i256 1
; CHECK-NEXT:    [[OLDVALRES:%[0-9]+]]:_(s256) = G_ATOMICRMW_UMAX [[ADDR]](p0), [[VAL]] :: (load store seq_cst 32 on %ir.addr)
; CHECK-NEXT:    [[RES:%[0-9]+]]:_(s32) = G_TRUNC [[OLDVALRES]]
  %oldval = atomicrmw umax i256* %addr, i256 1 seq_cst
  ; FIXME: We currently can't lower 'ret i256' and it's not the purpose of this
  ;        test so work around it by truncating to i32 for now.
  %oldval.trunc = trunc i256 %oldval to i32
  ret i32 %oldval.trunc
}

@addr = global i8* null

define void @test_blockaddress() {
; CHECK-LABEL: name: test_blockaddress
; CHECK: [[BADDR:%[0-9]+]]:_(p0) = G_BLOCK_ADDR blockaddress(@test_blockaddress, %ir-block.block)
; CHECK: G_STORE [[BADDR]](p0)
  store i8* blockaddress(@test_blockaddress, %block), i8** @addr
  indirectbr i8* blockaddress(@test_blockaddress, %block), [label %block]
block:
  ret void
}

%t = type { i32 }
declare {}* @llvm.invariant.start.p0i8(i64, i8* nocapture) readonly nounwind
declare void @llvm.invariant.end.p0i8({}*, i64, i8* nocapture) nounwind
define void @test_invariant_intrin() {
; CHECK-LABEL: name: test_invariant_intrin
; CHECK: %{{[0-9]+}}:_(s64) = G_IMPLICIT_DEF
; CHECK-NEXT: RET_ReallyLR
  %x = alloca %t
  %y = bitcast %t* %x to i8*
  %inv = call {}* @llvm.invariant.start.p0i8(i64 8, i8* %y)
  call void @llvm.invariant.end.p0i8({}* %inv, i64 8, i8* %y)
  ret void
}

declare float @llvm.ceil.f32(float)
define float @test_ceil_f32(float %x) {
  ; CHECK-LABEL: name:            test_ceil_f32
  ; CHECK: %{{[0-9]+}}:_(s32) = G_FCEIL %{{[0-9]+}}
  %y = call float @llvm.ceil.f32(float %x)
  ret float %y
}

declare double @llvm.ceil.f64(double)
define double @test_ceil_f64(double %x) {
  ; CHECK-LABEL: name:            test_ceil_f64
  ; CHECK: %{{[0-9]+}}:_(s64) = G_FCEIL %{{[0-9]+}}
  %y = call double @llvm.ceil.f64(double %x)
  ret double %y
}

declare <2 x float> @llvm.ceil.v2f32(<2 x float>)
define <2 x float> @test_ceil_v2f32(<2 x float> %x) {
  ; CHECK-LABEL: name:            test_ceil_v2f32
  ; CHECK: %{{[0-9]+}}:_(<2 x s32>) = G_FCEIL %{{[0-9]+}}
  %y = call <2 x float> @llvm.ceil.v2f32(<2 x float> %x)
  ret <2 x float> %y
}

declare <4 x float> @llvm.ceil.v4f32(<4 x float>)
define <4 x float> @test_ceil_v4f32(<4 x float> %x) {
  ; CHECK-LABEL: name:            test_ceil_v4f32
  ; CHECK: %{{[0-9]+}}:_(<4 x s32>) = G_FCEIL %{{[0-9]+}}
  ; SELECT: %{{[0-9]+}}:fpr128 = FRINTPv4f32 %{{[0-9]+}}
  %y = call <4 x float> @llvm.ceil.v4f32(<4 x float> %x)
  ret <4 x float> %y
}

declare <2 x double> @llvm.ceil.v2f64(<2 x double>)
define <2 x double> @test_ceil_v2f64(<2 x double> %x) {
  ; CHECK-LABEL: name:            test_ceil_v2f64
  ; CHECK: %{{[0-9]+}}:_(<2 x s64>) = G_FCEIL %{{[0-9]+}}
  %y = call <2 x double> @llvm.ceil.v2f64(<2 x double> %x)
  ret <2 x double> %y
}

declare float @llvm.cos.f32(float)
define float @test_cos_f32(float %x) {
  ; CHECK-LABEL: name:            test_cos_f32
  ; CHECK: %{{[0-9]+}}:_(s32) = G_FCOS %{{[0-9]+}}
  %y = call float @llvm.cos.f32(float %x)
  ret float %y
}

declare float @llvm.sin.f32(float)
define float @test_sin_f32(float %x) {
  ; CHECK-LABEL: name:            test_sin_f32
  ; CHECK: %{{[0-9]+}}:_(s32) = G_FSIN %{{[0-9]+}}
  %y = call float @llvm.sin.f32(float %x)
  ret float %y
}

declare float @llvm.sqrt.f32(float)
define float @test_sqrt_f32(float %x) {
  ; CHECK-LABEL: name:            test_sqrt_f32
  ; CHECK: %{{[0-9]+}}:_(s32) = G_FSQRT %{{[0-9]+}}
  %y = call float @llvm.sqrt.f32(float %x)
  ret float %y
}

declare float @llvm.floor.f32(float)
define float @test_floor_f32(float %x) {
  ; CHECK-LABEL: name:            test_floor_f32
  ; CHECK: %{{[0-9]+}}:_(s32) = G_FFLOOR %{{[0-9]+}}
  %y = call float @llvm.floor.f32(float %x)
  ret float %y
}

declare float @llvm.nearbyint.f32(float)
define float @test_nearbyint_f32(float %x) {
  ; CHECK-LABEL: name:            test_nearbyint_f32
  ; CHECK: %{{[0-9]+}}:_(s32) = G_FNEARBYINT %{{[0-9]+}}
  %y = call float @llvm.nearbyint.f32(float %x)
  ret float %y
}

; CHECK-LABEL: name: test_llvm.aarch64.neon.ld3.v4i32.p0i32
; CHECK: %1:_(<4 x s32>), %2:_(<4 x s32>), %3:_(<4 x s32>) = G_INTRINSIC_W_SIDE_EFFECTS intrinsic(@llvm.aarch64.neon.ld3), %0(p0) :: (load 48 from %ir.ptr, align 64)
define void @test_llvm.aarch64.neon.ld3.v4i32.p0i32(i32* %ptr) {
  %arst = call { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld3.v4i32.p0i32(i32* %ptr)
  ret void
}

declare { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld3.v4i32.p0i32(i32*) #3

define void @test_i1_arg_zext(void (i1)* %f) {
; CHECK-LABEL: name: test_i1_arg_zext
; CHECK: [[I1:%[0-9]+]]:_(s1) = G_CONSTANT i1 true
; CHECK: [[ZEXT0:%[0-9]+]]:_(s8) = G_ZEXT [[I1]](s1)
; CHECK: [[ZEXT1:%[0-9]+]]:_(s32) = G_ZEXT [[ZEXT0]](s8)
; CHECK: $w0 = COPY [[ZEXT1]](s32)
  call void %f(i1 true)
  ret void
}

declare i8* @llvm.stacksave()
declare void @llvm.stackrestore(i8*)
define void @test_stacksaverestore() {
  ; CHECK-LABEL: name: test_stacksaverestore
  ; CHECK: [[SAVE:%[0-9]+]]:_(p0) = COPY $sp
  ; CHECK-NEXT: $sp = COPY [[SAVE]](p0)
  ; CHECK-NEXT: RET_ReallyLR
  %sp = call i8* @llvm.stacksave()
  call void @llvm.stackrestore(i8* %sp)
  ret void
}

declare float @llvm.rint.f32(float)
define float @test_rint_f32(float %x) {
  ; CHECK-LABEL: name:            test_rint_f32
  ; CHECK: %{{[0-9]+}}:_(s32) = G_FRINT %{{[0-9]+}}
  %y = call float @llvm.rint.f32(float %x)
  ret float %y
}

declare void @llvm.assume(i1)
define void @test_assume(i1 %x) {
  ; CHECK-LABEL: name:            test_assume
  ; CHECK-NOT: llvm.assume
  ; CHECK: RET_ReallyLR
  call void @llvm.assume(i1 %x)
  ret void
}

declare void @llvm.experimental.noalias.scope.decl(metadata)
define void @test.llvm.noalias.scope.decl(i8* %P, i8* %Q) nounwind ssp {
  tail call void @llvm.experimental.noalias.scope.decl(metadata !3)
  ; CHECK-LABEL: name: test.llvm.noalias.scope.decl
  ; CHECK-NOT: llvm.experimental.noalias.scope.decl
  ; CHECK: RET_ReallyLR
  ret void
}

!3 = !{ !4 }
!4 = distinct !{ !4, !5, !"test1: var" }
!5 = distinct !{ !5, !"test1" }


declare void @llvm.sideeffect()
define void @test_sideeffect() {
  ; CHECK-LABEL: name:            test_sideeffect
  ; CHECK-NOT: llvm.sideeffect
  ; CHECK: RET_ReallyLR
  call void @llvm.sideeffect()
  ret void
}

declare void @llvm.var.annotation(i8*, i8*, i8*, i32, i8*)
define void @test_var_annotation(i8*, i8*, i8*, i32) {
  ; CHECK-LABEL: name:            test_var_annotation
  ; CHECK-NOT: llvm.var.annotation
  ; CHECK: RET_ReallyLR
  call void @llvm.var.annotation(i8* %0, i8* %1, i8* %2, i32 %3, i8* null)
  ret void
}

declare i64 @llvm.readcyclecounter()
define i64 @test_readcyclecounter() {
  ; CHECK-LABEL: name:            test_readcyclecounter
  ; CHECK: [[RES:%[0-9]+]]:_(s64) = G_READCYCLECOUNTER{{$}}
  ; CHECK-NEXT: $x0 = COPY [[RES]]
  ; CHECK-NEXT: RET_ReallyLR implicit $x0
  %res = call i64 @llvm.readcyclecounter()
  ret i64 %res
}

define i64 @test_freeze(i64 %a) {
  ; CHECK-LABEL: name:            test_freeze
  ; CHECK: [[COPY:%[0-9]+]]:_(s64) = COPY $x0
  ; CHECK-NEXT: [[RES:%[0-9]+]]:_(s64) = G_FREEZE [[COPY]]
  ; CHECK-NEXT: $x0 = COPY [[RES]]
  ; CHECK-NEXT: RET_ReallyLR implicit $x0
  %res = freeze i64 %a
  ret i64 %res
}

define {i8, i32} @test_freeze_struct({ i8, i32 }* %addr) {
  ; CHECK-LABEL: name:            test_freeze_struct
  ; CHECK: [[COPY:%[0-9]+]]:_(p0) = COPY $x0
  ; CHECK-NEXT: [[LOAD:%[0-9]+]]:_(s8) = G_LOAD [[COPY]](p0)
  ; CHECK-NEXT: [[C:%[0-9]+]]:_(s64) = G_CONSTANT i64 4
  ; CHECK-NEXT: [[PTR_ADD:%[0-9]+]]:_(p0) = G_PTR_ADD [[COPY]], [[C]]
  ; CHECK-NEXT: [[LOAD1:%[0-9]+]]:_(s32) = G_LOAD [[PTR_ADD]](p0)
  ; CHECK-NEXT: [[FREEZE:%[0-9]+]]:_(s8) = G_FREEZE [[LOAD]]
  ; CHECK-NEXT: [[FREEZE1:%[0-9]+]]:_(s32) = G_FREEZE [[LOAD1]]
  ; CHECK-NEXT: [[ANYEXT:%[0-9]+]]:_(s32) = G_ANYEXT [[FREEZE]]
  ; CHECK-NEXT: $w0 = COPY [[ANYEXT]]
  ; CHECK-NEXT: $w1 = COPY [[FREEZE1]]
  ; CHECK-NEXT: RET_ReallyLR implicit $w0, implicit $w1
  %load = load { i8, i32 }, { i8, i32 }* %addr
  %res = freeze {i8, i32} %load
  ret {i8, i32} %res
}

!0 = !{ i64 0, i64 2 }
