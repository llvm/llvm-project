; RUN: llc -O0 -stop-after=irtranslator -global-isel -verify-machineinstrs %s -o - 2>&1 | FileCheck %s

; This file checks that the translation from llvm IR to generic MachineInstr
; is correct.
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-apple-ios"

; Tests for add.
; CHECK-LABEL: name: addi64
; CHECK: [[ARG1:%[0-9]+]](s64) = COPY %x0
; CHECK-NEXT: [[ARG2:%[0-9]+]](s64) = COPY %x1
; CHECK-NEXT: [[RES:%[0-9]+]](s64) = G_ADD [[ARG1]], [[ARG2]]
; CHECK-NEXT: %x0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %x0 
define i64 @addi64(i64 %arg1, i64 %arg2) {
  %res = add i64 %arg1, %arg2
  ret i64 %res
}

; CHECK-LABEL: name: muli64
; CHECK: [[ARG1:%[0-9]+]](s64) = COPY %x0
; CHECK-NEXT: [[ARG2:%[0-9]+]](s64) = COPY %x1
; CHECK-NEXT: [[RES:%[0-9]+]](s64) = G_MUL [[ARG1]], [[ARG2]]
; CHECK-NEXT: %x0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %x0
define i64 @muli64(i64 %arg1, i64 %arg2) {
  %res = mul i64 %arg1, %arg2
  ret i64 %res
}

; Tests for alloca
; CHECK-LABEL: name: allocai64
; CHECK: stack:
; CHECK-NEXT:   - { id: 0, name: ptr1, offset: 0, size: 8, alignment: 8 }
; CHECK-NEXT:   - { id: 1, name: ptr2, offset: 0, size: 8, alignment: 1 }
; CHECK-NEXT:   - { id: 2, name: ptr3, offset: 0, size: 128, alignment: 8 }
; CHECK-NEXT:   - { id: 3, name: ptr4, offset: 0, size: 1, alignment: 8 }
; CHECK: %{{[0-9]+}}(p0) = G_FRAME_INDEX %stack.0.ptr1
; CHECK: %{{[0-9]+}}(p0) = G_FRAME_INDEX %stack.1.ptr2
; CHECK: %{{[0-9]+}}(p0) = G_FRAME_INDEX %stack.2.ptr3
; CHECK: %{{[0-9]+}}(p0) = G_FRAME_INDEX %stack.3.ptr4
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
; Entry basic block.
; CHECK: {{[0-9a-zA-Z._-]+}}:
;
; Make sure we have one successor and only one.
; CHECK-NEXT: successors: %[[END:[0-9a-zA-Z._-]+]]({{0x[a-f0-9]+ / 0x[a-f0-9]+}} = 100.00%)
;
; Check that we emit the correct branch.
; CHECK: G_BR %[[END]]
;
; Check that end contains the return instruction.
; CHECK: [[END]]:
; CHECK-NEXT: RET_ReallyLR
define void @uncondbr() {
  br label %end
end:
  ret void
}

; Tests for conditional br.
; CHECK-LABEL: name: condbr
; CHECK: body:
;
; Entry basic block.
; CHECK: {{[0-9a-zA-Z._-]+}}:
;
; Make sure we have two successors
; CHECK-NEXT: successors: %[[TRUE:[0-9a-zA-Z._-]+]]({{0x[a-f0-9]+ / 0x[a-f0-9]+}} = 50.00%),
; CHECK:                  %[[FALSE:[0-9a-zA-Z._-]+]]({{0x[a-f0-9]+ / 0x[a-f0-9]+}} = 50.00%)
;
; Check that we emit the correct branch.
; CHECK: [[ADDR:%.*]](p0) = COPY %x0
; CHECK: [[TST:%.*]](s1) = G_LOAD [[ADDR]](p0)
; CHECK: G_BRCOND [[TST]](s1), %[[TRUE]]
; CHECK: G_BR %[[FALSE]]
;
; Check that each successor contains the return instruction.
; CHECK: [[TRUE]]:
; CHECK-NEXT: RET_ReallyLR
; CHECK: [[FALSE]]:
; CHECK-NEXT: RET_ReallyLR
define void @condbr(i1* %tstaddr) {
  %tst = load i1, i1* %tstaddr
  br i1 %tst, label %true, label %false
true:
  ret void
false:
  ret void
}

; Tests for or.
; CHECK-LABEL: name: ori64
; CHECK: [[ARG1:%[0-9]+]](s64) = COPY %x0
; CHECK-NEXT: [[ARG2:%[0-9]+]](s64) = COPY %x1
; CHECK-NEXT: [[RES:%[0-9]+]](s64) = G_OR [[ARG1]], [[ARG2]]
; CHECK-NEXT: %x0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %x0
define i64 @ori64(i64 %arg1, i64 %arg2) {
  %res = or i64 %arg1, %arg2
  ret i64 %res
}

; CHECK-LABEL: name: ori32
; CHECK: [[ARG1:%[0-9]+]](s32) = COPY %w0
; CHECK-NEXT: [[ARG2:%[0-9]+]](s32) = COPY %w1
; CHECK-NEXT: [[RES:%[0-9]+]](s32) = G_OR [[ARG1]], [[ARG2]]
; CHECK-NEXT: %w0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %w0
define i32 @ori32(i32 %arg1, i32 %arg2) {
  %res = or i32 %arg1, %arg2
  ret i32 %res
}

; Tests for xor.
; CHECK-LABEL: name: xori64
; CHECK: [[ARG1:%[0-9]+]](s64) = COPY %x0
; CHECK-NEXT: [[ARG2:%[0-9]+]](s64) = COPY %x1
; CHECK-NEXT: [[RES:%[0-9]+]](s64) = G_XOR [[ARG1]], [[ARG2]]
; CHECK-NEXT: %x0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %x0
define i64 @xori64(i64 %arg1, i64 %arg2) {
  %res = xor i64 %arg1, %arg2
  ret i64 %res
}

; CHECK-LABEL: name: xori32
; CHECK: [[ARG1:%[0-9]+]](s32) = COPY %w0
; CHECK-NEXT: [[ARG2:%[0-9]+]](s32) = COPY %w1
; CHECK-NEXT: [[RES:%[0-9]+]](s32) = G_XOR [[ARG1]], [[ARG2]]
; CHECK-NEXT: %w0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %w0
define i32 @xori32(i32 %arg1, i32 %arg2) {
  %res = xor i32 %arg1, %arg2
  ret i32 %res
}

; Tests for and.
; CHECK-LABEL: name: andi64
; CHECK: [[ARG1:%[0-9]+]](s64) = COPY %x0
; CHECK-NEXT: [[ARG2:%[0-9]+]](s64) = COPY %x1
; CHECK-NEXT: [[RES:%[0-9]+]](s64) = G_AND [[ARG1]], [[ARG2]]
; CHECK-NEXT: %x0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %x0
define i64 @andi64(i64 %arg1, i64 %arg2) {
  %res = and i64 %arg1, %arg2
  ret i64 %res
}

; CHECK-LABEL: name: andi32
; CHECK: [[ARG1:%[0-9]+]](s32) = COPY %w0
; CHECK-NEXT: [[ARG2:%[0-9]+]](s32) = COPY %w1
; CHECK-NEXT: [[RES:%[0-9]+]](s32) = G_AND [[ARG1]], [[ARG2]]
; CHECK-NEXT: %w0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %w0
define i32 @andi32(i32 %arg1, i32 %arg2) {
  %res = and i32 %arg1, %arg2
  ret i32 %res
}

; Tests for sub.
; CHECK-LABEL: name: subi64
; CHECK: [[ARG1:%[0-9]+]](s64) = COPY %x0
; CHECK-NEXT: [[ARG2:%[0-9]+]](s64) = COPY %x1
; CHECK-NEXT: [[RES:%[0-9]+]](s64) = G_SUB [[ARG1]], [[ARG2]]
; CHECK-NEXT: %x0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %x0
define i64 @subi64(i64 %arg1, i64 %arg2) {
  %res = sub i64 %arg1, %arg2
  ret i64 %res
}

; CHECK-LABEL: name: subi32
; CHECK: [[ARG1:%[0-9]+]](s32) = COPY %w0
; CHECK-NEXT: [[ARG2:%[0-9]+]](s32) = COPY %w1
; CHECK-NEXT: [[RES:%[0-9]+]](s32) = G_SUB [[ARG1]], [[ARG2]]
; CHECK-NEXT: %w0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %w0
define i32 @subi32(i32 %arg1, i32 %arg2) {
  %res = sub i32 %arg1, %arg2
  ret i32 %res
}

; CHECK-LABEL: name: ptrtoint
; CHECK: [[ARG1:%[0-9]+]](p0) = COPY %x0
; CHECK: [[RES:%[0-9]+]](s64) = G_PTRTOINT [[ARG1]]
; CHECK: %x0 = COPY [[RES]]
; CHECK: RET_ReallyLR implicit %x0
define i64 @ptrtoint(i64* %a) {
  %val = ptrtoint i64* %a to i64
  ret i64 %val
}

; CHECK-LABEL: name: inttoptr
; CHECK: [[ARG1:%[0-9]+]](s64) = COPY %x0
; CHECK: [[RES:%[0-9]+]](p0) = G_INTTOPTR [[ARG1]]
; CHECK: %x0 = COPY [[RES]]
; CHECK: RET_ReallyLR implicit %x0
define i64* @inttoptr(i64 %a) {
  %val = inttoptr i64 %a to i64*
  ret i64* %val
}

; CHECK-LABEL: name: trivial_bitcast
; CHECK: [[ARG1:%[0-9]+]](p0) = COPY %x0
; CHECK: %x0 = COPY [[ARG1]]
; CHECK: RET_ReallyLR implicit %x0
define i64* @trivial_bitcast(i8* %a) {
  %val = bitcast i8* %a to i64*
  ret i64* %val
}

; CHECK-LABEL: name: trivial_bitcast_with_copy
; CHECK:     [[A:%[0-9]+]](p0) = COPY %x0
; CHECK:     G_BR %[[CAST:bb\.[0-9]+]]

; CHECK: [[CAST]]:
; CHECK:     {{%[0-9]+}}(p0) = COPY [[A]]
; CHECK:     G_BR %[[END:bb\.[0-9]+]]

; CHECK: [[END]]:
define i64* @trivial_bitcast_with_copy(i8* %a) {
  br label %cast

end:
  ret i64* %val

cast:
  %val = bitcast i8* %a to i64*
  br label %end
}

; CHECK-LABEL: name: bitcast
; CHECK: [[ARG1:%[0-9]+]](s64) = COPY %x0
; CHECK: [[RES1:%[0-9]+]](<2 x s32>) = G_BITCAST [[ARG1]]
; CHECK: [[RES2:%[0-9]+]](s64) = G_BITCAST [[RES1]]
; CHECK: %x0 = COPY [[RES2]]
; CHECK: RET_ReallyLR implicit %x0
define i64 @bitcast(i64 %a) {
  %res1 = bitcast i64 %a to <2 x i32>
  %res2 = bitcast <2 x i32> %res1 to i64
  ret i64 %res2
}

; CHECK-LABEL: name: trunc
; CHECK: [[ARG1:%[0-9]+]](s64) = COPY %x0
; CHECK: [[VEC:%[0-9]+]](<4 x s32>) = G_LOAD
; CHECK: [[RES1:%[0-9]+]](s8) = G_TRUNC [[ARG1]]
; CHECK: [[RES2:%[0-9]+]](<4 x s16>) = G_TRUNC [[VEC]]
define void @trunc(i64 %a) {
  %vecptr = alloca <4 x i32>
  %vec = load <4 x i32>, <4 x i32>* %vecptr
  %res1 = trunc i64 %a to i8
  %res2 = trunc <4 x i32> %vec to <4 x i16>
  ret void
}

; CHECK-LABEL: name: load
; CHECK: [[ADDR:%[0-9]+]](p0) = COPY %x0
; CHECK: [[ADDR42:%[0-9]+]](p42) = COPY %x1
; CHECK: [[VAL1:%[0-9]+]](s64) = G_LOAD [[ADDR]](p0) :: (load 8 from %ir.addr, align 16)
; CHECK: [[VAL2:%[0-9]+]](s64) = G_LOAD [[ADDR42]](p42) :: (load 8 from %ir.addr42)
; CHECK: [[SUM:%.*]](s64) = G_ADD [[VAL1]], [[VAL2]]
; CHECK: %x0 = COPY [[SUM]]
; CHECK: RET_ReallyLR implicit %x0
define i64 @load(i64* %addr, i64 addrspace(42)* %addr42) {
  %val1 = load i64, i64* %addr, align 16
  %val2 = load i64, i64 addrspace(42)* %addr42
  %sum = add i64 %val1, %val2
  ret i64 %sum
}

; CHECK-LABEL: name: store
; CHECK: [[ADDR:%[0-9]+]](p0) = COPY %x0
; CHECK: [[ADDR42:%[0-9]+]](p42) = COPY %x1
; CHECK: [[VAL1:%[0-9]+]](s64) = COPY %x2
; CHECK: [[VAL2:%[0-9]+]](s64) = COPY %x3
; CHECK: G_STORE [[VAL1]](s64), [[ADDR]](p0) :: (store 8 into %ir.addr, align 16)
; CHECK: G_STORE [[VAL2]](s64), [[ADDR42]](p42) :: (store 8 into %ir.addr42)
; CHECK: RET_ReallyLR
define void @store(i64* %addr, i64 addrspace(42)* %addr42, i64 %val1, i64 %val2) {
  store i64 %val1, i64* %addr, align 16
  store i64 %val2, i64 addrspace(42)* %addr42
  %sum = add i64 %val1, %val2
  ret void
}

; CHECK-LABEL: name: intrinsics
; CHECK: [[CUR:%[0-9]+]](s32) = COPY %w0
; CHECK: [[BITS:%[0-9]+]](s32) = COPY %w1
; CHECK: [[PTR:%[0-9]+]](p0) = G_INTRINSIC intrinsic(@llvm.returnaddress), 0
; CHECK: [[PTR_VEC:%[0-9]+]](p0) = G_FRAME_INDEX %stack.0.ptr.vec
; CHECK: [[VEC:%[0-9]+]](<8 x s8>) = G_LOAD [[PTR_VEC]]
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

; CHECK: [[TRUE]]:
; CHECK:     [[RES1:%[0-9]+]](s32) = G_LOAD

; CHECK: [[FALSE]]:
; CHECK:     [[RES2:%[0-9]+]](s32) = G_LOAD

; CHECK:     [[RES:%[0-9]+]](s32) = PHI [[RES1]](s32), %[[TRUE]], [[RES2]](s32), %[[FALSE]]
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
; CHECK: [[IN:%[0-9]+]](s32) = COPY %w0
; CHECK: [[ONE:%[0-9]+]](s32) = G_CONSTANT 1
; CHECK: G_BR

; CHECK: [[SUM1:%[0-9]+]](s32) = G_ADD [[IN]], [[ONE]]
; CHECK: [[SUM2:%[0-9]+]](s32) = G_ADD [[IN]], [[ONE]]
; CHECK: [[RES:%[0-9]+]](s32) = G_ADD [[SUM1]], [[SUM2]]
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
; CHECK: [[TWO:%[0-9]+]](s32) = G_CONSTANT 2
; CHECK: [[ANSWER:%[0-9]+]](s32) = G_CONSTANT 42
; CHECK: [[RES:%[0-9]+]](s32) = G_ADD [[TWO]], [[ANSWER]]
define i32 @constant_int_start() {
  %res = add i32 2, 42
  ret i32 %res
}

; CHECK-LABEL: name: test_undef
; CHECK: [[UNDEF:%[0-9]+]](s32) = IMPLICIT_DEF
; CHECK: %w0 = COPY [[UNDEF]]
define i32 @test_undef() {
  ret i32 undef
}

; CHECK-LABEL: name: test_constant_inttoptr
; CHECK: [[ONE:%[0-9]+]](s64) = G_CONSTANT 1
; CHECK: [[PTR:%[0-9]+]](p0) = G_INTTOPTR [[ONE]]
; CHECK: %x0 = COPY [[PTR]]
define i8* @test_constant_inttoptr() {
  ret i8* inttoptr(i64 1 to i8*)
}

  ; This failed purely because the Constant -> VReg map was kept across
  ; functions, so reuse the "i64 1" from above.
; CHECK-LABEL: name: test_reused_constant
; CHECK: [[ONE:%[0-9]+]](s64) = G_CONSTANT 1
; CHECK: %x0 = COPY [[ONE]]
define i64 @test_reused_constant() {
  ret i64 1
}

; CHECK-LABEL: name: test_sext
; CHECK: [[IN:%[0-9]+]](s32) = COPY %w0
; CHECK: [[RES:%[0-9]+]](s64) = G_SEXT [[IN]]
; CHECK: %x0 = COPY [[RES]]
define i64 @test_sext(i32 %in) {
  %res = sext i32 %in to i64
  ret i64 %res
}

; CHECK-LABEL: name: test_zext
; CHECK: [[IN:%[0-9]+]](s32) = COPY %w0
; CHECK: [[RES:%[0-9]+]](s64) = G_ZEXT [[IN]]
; CHECK: %x0 = COPY [[RES]]
define i64 @test_zext(i32 %in) {
  %res = zext i32 %in to i64
  ret i64 %res
}

; CHECK-LABEL: name: test_shl
; CHECK: [[ARG1:%[0-9]+]](s32) = COPY %w0
; CHECK-NEXT: [[ARG2:%[0-9]+]](s32) = COPY %w1
; CHECK-NEXT: [[RES:%[0-9]+]](s32) = G_SHL [[ARG1]], [[ARG2]]
; CHECK-NEXT: %w0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %w0
define i32 @test_shl(i32 %arg1, i32 %arg2) {
  %res = shl i32 %arg1, %arg2
  ret i32 %res
}


; CHECK-LABEL: name: test_lshr
; CHECK: [[ARG1:%[0-9]+]](s32) = COPY %w0
; CHECK-NEXT: [[ARG2:%[0-9]+]](s32) = COPY %w1
; CHECK-NEXT: [[RES:%[0-9]+]](s32) = G_LSHR [[ARG1]], [[ARG2]]
; CHECK-NEXT: %w0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %w0
define i32 @test_lshr(i32 %arg1, i32 %arg2) {
  %res = lshr i32 %arg1, %arg2
  ret i32 %res
}

; CHECK-LABEL: name: test_ashr
; CHECK: [[ARG1:%[0-9]+]](s32) = COPY %w0
; CHECK-NEXT: [[ARG2:%[0-9]+]](s32) = COPY %w1
; CHECK-NEXT: [[RES:%[0-9]+]](s32) = G_ASHR [[ARG1]], [[ARG2]]
; CHECK-NEXT: %w0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %w0
define i32 @test_ashr(i32 %arg1, i32 %arg2) {
  %res = ashr i32 %arg1, %arg2
  ret i32 %res
}

; CHECK-LABEL: name: test_sdiv
; CHECK: [[ARG1:%[0-9]+]](s32) = COPY %w0
; CHECK-NEXT: [[ARG2:%[0-9]+]](s32) = COPY %w1
; CHECK-NEXT: [[RES:%[0-9]+]](s32) = G_SDIV [[ARG1]], [[ARG2]]
; CHECK-NEXT: %w0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %w0
define i32 @test_sdiv(i32 %arg1, i32 %arg2) {
  %res = sdiv i32 %arg1, %arg2
  ret i32 %res
}

; CHECK-LABEL: name: test_udiv
; CHECK: [[ARG1:%[0-9]+]](s32) = COPY %w0
; CHECK-NEXT: [[ARG2:%[0-9]+]](s32) = COPY %w1
; CHECK-NEXT: [[RES:%[0-9]+]](s32) = G_UDIV [[ARG1]], [[ARG2]]
; CHECK-NEXT: %w0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %w0
define i32 @test_udiv(i32 %arg1, i32 %arg2) {
  %res = udiv i32 %arg1, %arg2
  ret i32 %res
}

; CHECK-LABEL: name: test_srem
; CHECK: [[ARG1:%[0-9]+]](s32) = COPY %w0
; CHECK-NEXT: [[ARG2:%[0-9]+]](s32) = COPY %w1
; CHECK-NEXT: [[RES:%[0-9]+]](s32) = G_SREM [[ARG1]], [[ARG2]]
; CHECK-NEXT: %w0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %w0
define i32 @test_srem(i32 %arg1, i32 %arg2) {
  %res = srem i32 %arg1, %arg2
  ret i32 %res
}

; CHECK-LABEL: name: test_urem
; CHECK: [[ARG1:%[0-9]+]](s32) = COPY %w0
; CHECK-NEXT: [[ARG2:%[0-9]+]](s32) = COPY %w1
; CHECK-NEXT: [[RES:%[0-9]+]](s32) = G_UREM [[ARG1]], [[ARG2]]
; CHECK-NEXT: %w0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %w0
define i32 @test_urem(i32 %arg1, i32 %arg2) {
  %res = urem i32 %arg1, %arg2
  ret i32 %res
}

; CHECK-LABEL: name: test_constant_null
; CHECK: [[NULL:%[0-9]+]](p0) = G_CONSTANT 0
; CHECK: %x0 = COPY [[NULL]]
define i8* @test_constant_null() {
  ret i8* null
}

; CHECK-LABEL: name: test_struct_memops
; CHECK: [[ADDR:%[0-9]+]](p0) = COPY %x0
; CHECK: [[VAL:%[0-9]+]](s64) = G_LOAD [[ADDR]](p0) :: (load 8 from  %ir.addr, align 4)
; CHECK: G_STORE [[VAL]](s64), [[ADDR]](p0) :: (store 8 into  %ir.addr, align 4)
define void @test_struct_memops({ i8, i32 }* %addr) {
  %val = load { i8, i32 }, { i8, i32 }* %addr
  store { i8, i32 } %val, { i8, i32 }* %addr
  ret void
}

; CHECK-LABEL: name: test_i1_memops
; CHECK: [[ADDR:%[0-9]+]](p0) = COPY %x0
; CHECK: [[VAL:%[0-9]+]](s1) = G_LOAD [[ADDR]](p0) :: (load 1 from  %ir.addr)
; CHECK: G_STORE [[VAL]](s1), [[ADDR]](p0) :: (store 1 into  %ir.addr)
define void @test_i1_memops(i1* %addr) {
  %val = load i1, i1* %addr
  store i1 %val, i1* %addr
  ret void
}

; CHECK-LABEL: name: int_comparison
; CHECK: [[LHS:%[0-9]+]](s32) = COPY %w0
; CHECK: [[RHS:%[0-9]+]](s32) = COPY %w1
; CHECK: [[ADDR:%[0-9]+]](p0) = COPY %x2
; CHECK: [[TST:%[0-9]+]](s1) = G_ICMP intpred(ne), [[LHS]](s32), [[RHS]]
; CHECK: G_STORE [[TST]](s1), [[ADDR]](p0)
define void @int_comparison(i32 %a, i32 %b, i1* %addr) {
  %res = icmp ne i32 %a, %b
  store i1 %res, i1* %addr
  ret void
}

; CHECK-LABEL: name: ptr_comparison
; CHECK: [[LHS:%[0-9]+]](p0) = COPY %x0
; CHECK: [[RHS:%[0-9]+]](p0) = COPY %x1
; CHECK: [[ADDR:%[0-9]+]](p0) = COPY %x2
; CHECK: [[TST:%[0-9]+]](s1) = G_ICMP intpred(eq), [[LHS]](p0), [[RHS]]
; CHECK: G_STORE [[TST]](s1), [[ADDR]](p0)
define void @ptr_comparison(i8* %a, i8* %b, i1* %addr) {
  %res = icmp eq i8* %a, %b
  store i1 %res, i1* %addr
  ret void
}

; CHECK-LABEL: name: test_fadd
; CHECK: [[ARG1:%[0-9]+]](s32) = COPY %s0
; CHECK-NEXT: [[ARG2:%[0-9]+]](s32) = COPY %s1
; CHECK-NEXT: [[RES:%[0-9]+]](s32) = G_FADD [[ARG1]], [[ARG2]]
; CHECK-NEXT: %s0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %s0
define float @test_fadd(float %arg1, float %arg2) {
  %res = fadd float %arg1, %arg2
  ret float %res
}

; CHECK-LABEL: name: test_fsub
; CHECK: [[ARG1:%[0-9]+]](s32) = COPY %s0
; CHECK-NEXT: [[ARG2:%[0-9]+]](s32) = COPY %s1
; CHECK-NEXT: [[RES:%[0-9]+]](s32) = G_FSUB [[ARG1]], [[ARG2]]
; CHECK-NEXT: %s0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %s0
define float @test_fsub(float %arg1, float %arg2) {
  %res = fsub float %arg1, %arg2
  ret float %res
}

; CHECK-LABEL: name: test_fmul
; CHECK: [[ARG1:%[0-9]+]](s32) = COPY %s0
; CHECK-NEXT: [[ARG2:%[0-9]+]](s32) = COPY %s1
; CHECK-NEXT: [[RES:%[0-9]+]](s32) = G_FMUL [[ARG1]], [[ARG2]]
; CHECK-NEXT: %s0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %s0
define float @test_fmul(float %arg1, float %arg2) {
  %res = fmul float %arg1, %arg2
  ret float %res
}

; CHECK-LABEL: name: test_fdiv
; CHECK: [[ARG1:%[0-9]+]](s32) = COPY %s0
; CHECK-NEXT: [[ARG2:%[0-9]+]](s32) = COPY %s1
; CHECK-NEXT: [[RES:%[0-9]+]](s32) = G_FDIV [[ARG1]], [[ARG2]]
; CHECK-NEXT: %s0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %s0
define float @test_fdiv(float %arg1, float %arg2) {
  %res = fdiv float %arg1, %arg2
  ret float %res
}

; CHECK-LABEL: name: test_frem
; CHECK: [[ARG1:%[0-9]+]](s32) = COPY %s0
; CHECK-NEXT: [[ARG2:%[0-9]+]](s32) = COPY %s1
; CHECK-NEXT: [[RES:%[0-9]+]](s32) = G_FREM [[ARG1]], [[ARG2]]
; CHECK-NEXT: %s0 = COPY [[RES]]
; CHECK-NEXT: RET_ReallyLR implicit %s0
define float @test_frem(float %arg1, float %arg2) {
  %res = frem float %arg1, %arg2
  ret float %res
}

; CHECK-LABEL: name: test_sadd_overflow
; CHECK: [[LHS:%[0-9]+]](s32) = COPY %w0
; CHECK: [[RHS:%[0-9]+]](s32) = COPY %w1
; CHECK: [[ADDR:%[0-9]+]](p0) = COPY %x2
; CHECK: [[VAL:%[0-9]+]](s32), [[OVERFLOW:%[0-9]+]](s1) = G_SADDO [[LHS]], [[RHS]]
; CHECK: [[RES:%[0-9]+]](s64) = G_SEQUENCE [[VAL]](s32), 0, [[OVERFLOW]](s1), 32
; CHECK: G_STORE [[RES]](s64), [[ADDR]](p0)
declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32)
define void @test_sadd_overflow(i32 %lhs, i32 %rhs, { i32, i1 }* %addr) {
  %res = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %lhs, i32 %rhs)
  store { i32, i1 } %res, { i32, i1 }* %addr
  ret void
}

; CHECK-LABEL: name: test_uadd_overflow
; CHECK: [[LHS:%[0-9]+]](s32) = COPY %w0
; CHECK: [[RHS:%[0-9]+]](s32) = COPY %w1
; CHECK: [[ADDR:%[0-9]+]](p0) = COPY %x2
; CHECK: [[ZERO:%[0-9]+]](s1) = G_CONSTANT 0
; CHECK: [[VAL:%[0-9]+]](s32), [[OVERFLOW:%[0-9]+]](s1) = G_UADDE [[LHS]], [[RHS]], [[ZERO]]
; CHECK: [[RES:%[0-9]+]](s64) = G_SEQUENCE [[VAL]](s32), 0, [[OVERFLOW]](s1), 32
; CHECK: G_STORE [[RES]](s64), [[ADDR]](p0)
declare { i32, i1 } @llvm.uadd.with.overflow.i32(i32, i32)
define void @test_uadd_overflow(i32 %lhs, i32 %rhs, { i32, i1 }* %addr) {
  %res = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %lhs, i32 %rhs)
  store { i32, i1 } %res, { i32, i1 }* %addr
  ret void
}

; CHECK-LABEL: name: test_ssub_overflow
; CHECK: [[LHS:%[0-9]+]](s32) = COPY %w0
; CHECK: [[RHS:%[0-9]+]](s32) = COPY %w1
; CHECK: [[ADDR:%[0-9]+]](p0) = COPY %x2
; CHECK: [[VAL:%[0-9]+]](s32), [[OVERFLOW:%[0-9]+]](s1) = G_SSUBO [[LHS]], [[RHS]]
; CHECK: [[RES:%[0-9]+]](s64) = G_SEQUENCE [[VAL]](s32), 0, [[OVERFLOW]](s1), 32
; CHECK: G_STORE [[RES]](s64), [[ADDR]](p0)
declare { i32, i1 } @llvm.ssub.with.overflow.i32(i32, i32)
define void @test_ssub_overflow(i32 %lhs, i32 %rhs, { i32, i1 }* %subr) {
  %res = call { i32, i1 } @llvm.ssub.with.overflow.i32(i32 %lhs, i32 %rhs)
  store { i32, i1 } %res, { i32, i1 }* %subr
  ret void
}

; CHECK-LABEL: name: test_usub_overflow
; CHECK: [[LHS:%[0-9]+]](s32) = COPY %w0
; CHECK: [[RHS:%[0-9]+]](s32) = COPY %w1
; CHECK: [[ADDR:%[0-9]+]](p0) = COPY %x2
; CHECK: [[ZERO:%[0-9]+]](s1) = G_CONSTANT 0
; CHECK: [[VAL:%[0-9]+]](s32), [[OVERFLOW:%[0-9]+]](s1) = G_USUBE [[LHS]], [[RHS]], [[ZERO]]
; CHECK: [[RES:%[0-9]+]](s64) = G_SEQUENCE [[VAL]](s32), 0, [[OVERFLOW]](s1), 32
; CHECK: G_STORE [[RES]](s64), [[ADDR]](p0)
declare { i32, i1 } @llvm.usub.with.overflow.i32(i32, i32)
define void @test_usub_overflow(i32 %lhs, i32 %rhs, { i32, i1 }* %subr) {
  %res = call { i32, i1 } @llvm.usub.with.overflow.i32(i32 %lhs, i32 %rhs)
  store { i32, i1 } %res, { i32, i1 }* %subr
  ret void
}

; CHECK-LABEL: name: test_smul_overflow
; CHECK: [[LHS:%[0-9]+]](s32) = COPY %w0
; CHECK: [[RHS:%[0-9]+]](s32) = COPY %w1
; CHECK: [[ADDR:%[0-9]+]](p0) = COPY %x2
; CHECK: [[VAL:%[0-9]+]](s32), [[OVERFLOW:%[0-9]+]](s1) = G_SMULO [[LHS]], [[RHS]]
; CHECK: [[RES:%[0-9]+]](s64) = G_SEQUENCE [[VAL]](s32), 0, [[OVERFLOW]](s1), 32
; CHECK: G_STORE [[RES]](s64), [[ADDR]](p0)
declare { i32, i1 } @llvm.smul.with.overflow.i32(i32, i32)
define void @test_smul_overflow(i32 %lhs, i32 %rhs, { i32, i1 }* %addr) {
  %res = call { i32, i1 } @llvm.smul.with.overflow.i32(i32 %lhs, i32 %rhs)
  store { i32, i1 } %res, { i32, i1 }* %addr
  ret void
}

; CHECK-LABEL: name: test_umul_overflow
; CHECK: [[LHS:%[0-9]+]](s32) = COPY %w0
; CHECK: [[RHS:%[0-9]+]](s32) = COPY %w1
; CHECK: [[ADDR:%[0-9]+]](p0) = COPY %x2
; CHECK: [[VAL:%[0-9]+]](s32), [[OVERFLOW:%[0-9]+]](s1) = G_UMULO [[LHS]], [[RHS]]
; CHECK: [[RES:%[0-9]+]](s64) = G_SEQUENCE [[VAL]](s32), 0, [[OVERFLOW]](s1), 32
; CHECK: G_STORE [[RES]](s64), [[ADDR]](p0)
declare { i32, i1 } @llvm.umul.with.overflow.i32(i32, i32)
define void @test_umul_overflow(i32 %lhs, i32 %rhs, { i32, i1 }* %addr) {
  %res = call { i32, i1 } @llvm.umul.with.overflow.i32(i32 %lhs, i32 %rhs)
  store { i32, i1 } %res, { i32, i1 }* %addr
  ret void
}

; CHECK-LABEL: name: test_extractvalue
; CHECK: [[STRUCT:%[0-9]+]](s128) = G_LOAD
; CHECK: [[RES:%[0-9]+]](s32) = G_EXTRACT [[STRUCT]](s128), 64
; CHECK: %w0 = COPY [[RES]]
%struct.nested = type {i8, { i8, i32 }, i32}
define i32 @test_extractvalue(%struct.nested* %addr) {
  %struct = load %struct.nested, %struct.nested* %addr
  %res = extractvalue %struct.nested %struct, 1, 1
  ret i32 %res
}

; CHECK-LABEL: name: test_extractvalue_agg
; CHECK: [[STRUCT:%[0-9]+]](s128) = G_LOAD
; CHECK: [[RES:%[0-9]+]](s64) = G_EXTRACT [[STRUCT]](s128), 32
; CHECK: G_STORE [[RES]]
define void @test_extractvalue_agg(%struct.nested* %addr, {i8, i32}* %addr2) {
  %struct = load %struct.nested, %struct.nested* %addr
  %res = extractvalue %struct.nested %struct, 1
  store {i8, i32} %res, {i8, i32}* %addr2
  ret void
}

; CHECK-LABEL: name: test_insertvalue
; CHECK: [[VAL:%[0-9]+]](s32) = COPY %w1
; CHECK: [[STRUCT:%[0-9]+]](s128) = G_LOAD
; CHECK: [[NEWSTRUCT:%[0-9]+]](s128) = G_INSERT [[STRUCT]](s128), [[VAL]](s32), 64
; CHECK: G_STORE [[NEWSTRUCT]](s128),
define void @test_insertvalue(%struct.nested* %addr, i32 %val) {
  %struct = load %struct.nested, %struct.nested* %addr
  %newstruct = insertvalue %struct.nested %struct, i32 %val, 1, 1
  store %struct.nested %newstruct, %struct.nested* %addr
  ret void
}

; CHECK-LABEL: name: test_insertvalue_agg
; CHECK: [[SMALLSTRUCT:%[0-9]+]](s64) = G_LOAD
; CHECK: [[STRUCT:%[0-9]+]](s128) = G_LOAD
; CHECK: [[RES:%[0-9]+]](s128) = G_INSERT [[STRUCT]](s128), [[SMALLSTRUCT]](s64), 32
; CHECK: G_STORE [[RES]](s128)
define void @test_insertvalue_agg(%struct.nested* %addr, {i8, i32}* %addr2) {
  %smallstruct = load {i8, i32}, {i8, i32}* %addr2
  %struct = load %struct.nested, %struct.nested* %addr
  %res = insertvalue %struct.nested %struct, {i8, i32} %smallstruct, 1
  store %struct.nested %res, %struct.nested* %addr
  ret void
}

; CHECK-LABEL: name: test_select
; CHECK: [[TST:%[0-9]+]](s1) = COPY %w0
; CHECK: [[LHS:%[0-9]+]](s32) = COPY %w1
; CHECK: [[RHS:%[0-9]+]](s32) = COPY %w2
; CHECK: [[RES:%[0-9]+]](s32) = G_SELECT [[TST]](s1), [[LHS]], [[RHS]]
; CHECK: %w0 = COPY [[RES]]
define i32 @test_select(i1 %tst, i32 %lhs, i32 %rhs) {
  %res = select i1 %tst, i32 %lhs, i32 %rhs
  ret i32 %res
}

; CHECK-LABEL: name: test_fptosi
; CHECK: [[FPADDR:%[0-9]+]](p0) = COPY %x0
; CHECK: [[FP:%[0-9]+]](s32) = G_LOAD [[FPADDR]](p0)
; CHECK: [[RES:%[0-9]+]](s64) = G_FPTOSI [[FP]](s32)
; CHECK: %x0 = COPY [[RES]]
define i64 @test_fptosi(float* %fp.addr) {
  %fp = load float, float* %fp.addr
  %res = fptosi float %fp to i64
  ret i64 %res
}

; CHECK-LABEL: name: test_fptoui
; CHECK: [[FPADDR:%[0-9]+]](p0) = COPY %x0
; CHECK: [[FP:%[0-9]+]](s32) = G_LOAD [[FPADDR]](p0)
; CHECK: [[RES:%[0-9]+]](s64) = G_FPTOUI [[FP]](s32)
; CHECK: %x0 = COPY [[RES]]
define i64 @test_fptoui(float* %fp.addr) {
  %fp = load float, float* %fp.addr
  %res = fptoui float %fp to i64
  ret i64 %res
}

; CHECK-LABEL: name: test_sitofp
; CHECK: [[ADDR:%[0-9]+]](p0) = COPY %x0
; CHECK: [[IN:%[0-9]+]](s32) = COPY %w1
; CHECK: [[FP:%[0-9]+]](s64) = G_SITOFP [[IN]](s32)
; CHECK: G_STORE [[FP]](s64), [[ADDR]](p0)
define void @test_sitofp(double* %addr, i32 %in) {
  %fp = sitofp i32 %in to double
  store double %fp, double* %addr
  ret void
}

; CHECK-LABEL: name: test_uitofp
; CHECK: [[ADDR:%[0-9]+]](p0) = COPY %x0
; CHECK: [[IN:%[0-9]+]](s32) = COPY %w1
; CHECK: [[FP:%[0-9]+]](s64) = G_UITOFP [[IN]](s32)
; CHECK: G_STORE [[FP]](s64), [[ADDR]](p0)
define void @test_uitofp(double* %addr, i32 %in) {
  %fp = uitofp i32 %in to double
  store double %fp, double* %addr
  ret void
}

; CHECK-LABEL: name: test_fpext
; CHECK: [[IN:%[0-9]+]](s32) = COPY %s0
; CHECK: [[RES:%[0-9]+]](s64) = G_FPEXT [[IN]](s32)
; CHECK: %d0 = COPY [[RES]]
define double @test_fpext(float %in) {
  %res = fpext float %in to double
  ret double %res
}

; CHECK-LABEL: name: test_fptrunc
; CHECK: [[IN:%[0-9]+]](s64) = COPY %d0
; CHECK: [[RES:%[0-9]+]](s32) = G_FPTRUNC [[IN]](s64)
; CHECK: %s0 = COPY [[RES]]
define float @test_fptrunc(double %in) {
  %res = fptrunc double %in to float
  ret float %res
}

; CHECK-LABEL: name: test_constant_float
; CHECK: [[ADDR:%[0-9]+]](p0) = COPY %x0
; CHECK: [[TMP:%[0-9]+]](s32) = G_FCONSTANT float 1.500000e+00
; CHECK: G_STORE [[TMP]](s32), [[ADDR]](p0)
define void @test_constant_float(float* %addr) {
  store float 1.5, float* %addr
  ret void
}

; CHECK-LABEL: name: float_comparison
; CHECK: [[LHSADDR:%[0-9]+]](p0) = COPY %x0
; CHECK: [[RHSADDR:%[0-9]+]](p0) = COPY %x1
; CHECK: [[BOOLADDR:%[0-9]+]](p0) = COPY %x2
; CHECK: [[LHS:%[0-9]+]](s32) = G_LOAD [[LHSADDR]](p0)
; CHECK: [[RHS:%[0-9]+]](s32) = G_LOAD [[RHSADDR]](p0)
; CHECK: [[TST:%[0-9]+]](s1) = G_FCMP floatpred(oge), [[LHS]](s32), [[RHS]]
; CHECK: G_STORE [[TST]](s1), [[BOOLADDR]](p0)
define void @float_comparison(float* %a.addr, float* %b.addr, i1* %bool.addr) {
  %a = load float, float* %a.addr
  %b = load float, float* %b.addr
  %res = fcmp oge float %a, %b
  store i1 %res, i1* %bool.addr
  ret void
}

@var = global i32 0

define i32* @test_global() {
; CHECK-LABEL: name: test_global
; CHECK: [[TMP:%[0-9]+]](p0) = G_GLOBAL_VALUE @var{{$}}
; CHECK: %x0 = COPY [[TMP]](p0)

  ret i32* @var
}

@var1 = addrspace(42) global i32 0
define i32 addrspace(42)* @test_global_addrspace() {
; CHECK-LABEL: name: test_global
; CHECK: [[TMP:%[0-9]+]](p42) = G_GLOBAL_VALUE @var1{{$}}
; CHECK: %x0 = COPY [[TMP]](p42)

  ret i32 addrspace(42)* @var1
}


define void()* @test_global_func() {
; CHECK-LABEL: name: test_global_func
; CHECK: [[TMP:%[0-9]+]](p0) = G_GLOBAL_VALUE @allocai64{{$}}
; CHECK: %x0 = COPY [[TMP]](p0)

  ret void()* @allocai64
}
