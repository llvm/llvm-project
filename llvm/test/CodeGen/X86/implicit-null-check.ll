; RUN: llc -verify-machineinstrs -O3 -mtriple=x86_64-apple-macosx -enable-implicit-null-checks < %s | FileCheck %s

define i32 @imp_null_check_load(ptr %x) {
; CHECK-LABEL: imp_null_check_load:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:  Ltmp0:
; CHECK-NEXT:    movl (%rdi), %eax ## on-fault: LBB0_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB0_1: ## %is_null
; CHECK-NEXT:    movl $42, %eax
; CHECK-NEXT:    retq

 entry:
  %c = icmp eq ptr %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %t = load i32, ptr %x
  ret i32 %t
}

; TODO: can make implicit
define i32 @imp_null_check_unordered_load(ptr %x) {
; CHECK-LABEL: imp_null_check_unordered_load:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:  Ltmp1:
; CHECK-NEXT:    movl (%rdi), %eax ## on-fault: LBB1_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB1_1: ## %is_null
; CHECK-NEXT:    movl $42, %eax
; CHECK-NEXT:    retq

 entry:
  %c = icmp eq ptr %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %t = load atomic i32, ptr %x unordered, align 4
  ret i32 %t
}


; TODO: Can be converted into implicit check.
;; Probably could be implicit, but we're conservative for now
define i32 @imp_null_check_seq_cst_load(ptr %x) {
; CHECK-LABEL: imp_null_check_seq_cst_load:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:    testq %rdi, %rdi
; CHECK-NEXT:    je LBB2_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    movl (%rdi), %eax
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB2_1: ## %is_null
; CHECK-NEXT:    movl $42, %eax
; CHECK-NEXT:    retq

 entry:
  %c = icmp eq ptr %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %t = load atomic i32, ptr %x seq_cst, align 4
  ret i32 %t
}

;; Might be memory mapped IO, so can't rely on fault behavior
define i32 @imp_null_check_volatile_load(ptr %x) {
; CHECK-LABEL: imp_null_check_volatile_load:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:    testq %rdi, %rdi
; CHECK-NEXT:    je LBB3_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    movl (%rdi), %eax
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB3_1: ## %is_null
; CHECK-NEXT:    movl $42, %eax
; CHECK-NEXT:    retq

 entry:
  %c = icmp eq ptr %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %t = load volatile i32, ptr %x, align 4
  ret i32 %t
}


define i8 @imp_null_check_load_i8(ptr %x) {
; CHECK-LABEL: imp_null_check_load_i8:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:  Ltmp2:
; CHECK-NEXT:    movb (%rdi), %al ## on-fault: LBB4_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB4_1: ## %is_null
; CHECK-NEXT:    movb $42, %al
; CHECK-NEXT:    retq

 entry:
  %c = icmp eq ptr %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i8 42

 not_null:
  %t = load i8, ptr %x
  ret i8 %t
}

define i256 @imp_null_check_load_i256(ptr %x) {
; CHECK-LABEL: imp_null_check_load_i256:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:    movq %rdi, %rax
; CHECK-NEXT:  Ltmp3:
; CHECK-NEXT:    movaps (%rsi), %xmm0 ## on-fault: LBB5_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    movaps 16(%rsi), %xmm1
; CHECK-NEXT:    movaps %xmm1, 16(%rax)
; CHECK-NEXT:    movaps %xmm0, (%rax)
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB5_1: ## %is_null
; CHECK-NEXT:    xorps %xmm0, %xmm0
; CHECK-NEXT:    movaps %xmm0, 16(%rax)
; CHECK-NEXT:    movq $0, 8(%rax)
; CHECK-NEXT:    movq $42, (%rax)
; CHECK-NEXT:    retq

 entry:
  %c = icmp eq ptr %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i256 42

 not_null:
  %t = load i256, ptr %x
  ret i256 %t
}



define i32 @imp_null_check_gep_load(ptr %x) {
; CHECK-LABEL: imp_null_check_gep_load:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:  Ltmp4:
; CHECK-NEXT:    movl 128(%rdi), %eax ## on-fault: LBB6_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB6_1: ## %is_null
; CHECK-NEXT:    movl $42, %eax
; CHECK-NEXT:    retq

 entry:
  %c = icmp eq ptr %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %x.gep = getelementptr i32, ptr %x, i32 32
  %t = load i32, ptr %x.gep
  ret i32 %t
}

define i32 @imp_null_check_add_result(ptr %x, i32 %p) {
; CHECK-LABEL: imp_null_check_add_result:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:  Ltmp5:
; CHECK-NEXT:    addl (%rdi), %esi ## on-fault: LBB7_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    movl %esi, %eax
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB7_1: ## %is_null
; CHECK-NEXT:    movl $42, %eax
; CHECK-NEXT:    retq

 entry:
  %c = icmp eq ptr %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %t = load i32, ptr %x
  %p1 = add i32 %t, %p
  ret i32 %p1
}

define i32 @imp_null_check_sub_result(ptr %x, i32 %p) {
; CHECK-LABEL: imp_null_check_sub_result:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:  Ltmp6:
; CHECK-NEXT:    movl (%rdi), %eax ## on-fault: LBB8_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    subl %esi, %eax
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB8_1: ## %is_null
; CHECK-NEXT:    movl $42, %eax
; CHECK-NEXT:    retq

 entry:
  %c = icmp eq ptr %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %t = load i32, ptr %x
  %p1 = sub i32 %t, %p
  ret i32 %p1
}

define i32 @imp_null_check_mul_result(ptr %x, i32 %p) {
; CHECK-LABEL: imp_null_check_mul_result:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:  Ltmp7:
; CHECK-NEXT:    imull (%rdi), %esi ## on-fault: LBB9_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    movl %esi, %eax
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB9_1: ## %is_null
; CHECK-NEXT:    movl $42, %eax
; CHECK-NEXT:    retq

 entry:
  %c = icmp eq ptr %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %t = load i32, ptr %x
  %p1 = mul i32 %t, %p
  ret i32 %p1
}

define i32 @imp_null_check_udiv_result(ptr %x, i32 %p) {
; CHECK-LABEL: imp_null_check_udiv_result:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:  Ltmp8:
; CHECK-NEXT:    movl (%rdi), %eax ## on-fault: LBB10_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    xorl %edx, %edx
; CHECK-NEXT:    divl %esi
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB10_1: ## %is_null
; CHECK-NEXT:    movl $42, %eax
; CHECK-NEXT:    retq

 entry:
  %c = icmp eq ptr %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %t = load i32, ptr %x
  %p1 = udiv i32 %t, %p
  ret i32 %p1
}

define i32 @imp_null_check_shl_result(ptr %x, i32 %p) {
; CHECK-LABEL: imp_null_check_shl_result:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:  Ltmp9:
; CHECK-NEXT:    movl (%rdi), %eax ## on-fault: LBB11_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    movl %esi, %ecx
; CHECK-NEXT:    shll %cl, %eax
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB11_1: ## %is_null
; CHECK-NEXT:    movl $42, %eax
; CHECK-NEXT:    retq

 entry:
  %c = icmp eq ptr %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %t = load i32, ptr %x
  %p1 = shl i32 %t, %p
  ret i32 %p1
}

define i32 @imp_null_check_lshr_result(ptr %x, i32 %p) {
; CHECK-LABEL: imp_null_check_lshr_result:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:  Ltmp10:
; CHECK-NEXT:    movl (%rdi), %eax ## on-fault: LBB12_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    movl %esi, %ecx
; CHECK-NEXT:    shrl %cl, %eax
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB12_1: ## %is_null
; CHECK-NEXT:    movl $42, %eax
; CHECK-NEXT:    retq

 entry:
  %c = icmp eq ptr %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %t = load i32, ptr %x
  %p1 = lshr i32 %t, %p
  ret i32 %p1
}




define i32 @imp_null_check_hoist_over_unrelated_load(ptr %x, ptr %y, ptr %z) {
; CHECK-LABEL: imp_null_check_hoist_over_unrelated_load:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:  Ltmp11:
; CHECK-NEXT:    movl (%rdi), %eax ## on-fault: LBB13_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    movl (%rsi), %ecx
; CHECK-NEXT:    movl %ecx, (%rdx)
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB13_1: ## %is_null
; CHECK-NEXT:    movl $42, %eax
; CHECK-NEXT:    retq

 entry:
  %c = icmp eq ptr %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %t0 = load i32, ptr %y
  %t1 = load i32, ptr %x
  store i32 %t0, ptr %z
  ret i32 %t1
}

define i32 @imp_null_check_via_mem_comparision(ptr %x, i32 %val) {
; CHECK-LABEL: imp_null_check_via_mem_comparision:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:  Ltmp12:
; CHECK-NEXT:    cmpl %esi, 4(%rdi) ## on-fault: LBB14_3
; CHECK-NEXT:  ## %bb.1: ## %not_null
; CHECK-NEXT:    jge LBB14_2
; CHECK-NEXT:  ## %bb.4: ## %ret_100
; CHECK-NEXT:    movl $100, %eax
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB14_3: ## %is_null
; CHECK-NEXT:    movl $42, %eax
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB14_2: ## %ret_200
; CHECK-NEXT:    movl $200, %eax
; CHECK-NEXT:    retq

 entry:
  %c = icmp eq ptr %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %x.loc = getelementptr i32, ptr %x, i32 1
  %t = load i32, ptr %x.loc
  %m = icmp slt i32 %t, %val
  br i1 %m, label %ret_100, label %ret_200

 ret_100:
  ret i32 100

 ret_200:
  ret i32 200
}

define i32 @imp_null_check_gep_load_with_use_dep(ptr %x, i32 %a) {
; CHECK-LABEL: imp_null_check_gep_load_with_use_dep:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:    ## kill: def $esi killed $esi def $rsi
; CHECK-NEXT:  Ltmp13:
; CHECK-NEXT:    movl (%rdi), %eax ## on-fault: LBB15_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    addl %edi, %esi
; CHECK-NEXT:    leal 4(%rax,%rsi), %eax
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB15_1: ## %is_null
; CHECK-NEXT:    movl $42, %eax
; CHECK-NEXT:    retq

 entry:
  %c = icmp eq ptr %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %x.loc = getelementptr i32, ptr %x, i32 1
  %y = ptrtoint ptr %x.loc to i32
  %b = add i32 %a, %y
  %t = load i32, ptr %x
  %z = add i32 %t, %b
  ret i32 %z
}

;; TODO: We could handle this case as we can lift the fence into the
;; previous block before the conditional without changing behavior.
define i32 @imp_null_check_load_fence1(ptr %x) {
; CHECK-LABEL: imp_null_check_load_fence1:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:    testq %rdi, %rdi
; CHECK-NEXT:    je LBB16_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    ##MEMBARRIER
; CHECK-NEXT:    movl (%rdi), %eax
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB16_1: ## %is_null
; CHECK-NEXT:    movl $42, %eax
; CHECK-NEXT:    retq

entry:
  %c = icmp eq ptr %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

is_null:
  ret i32 42

not_null:
  fence acquire
  %t = load i32, ptr %x
  ret i32 %t
}

;; TODO: We could handle this case as we can lift the fence into the
;; previous block before the conditional without changing behavior.
define i32 @imp_null_check_load_fence2(ptr %x) {
; CHECK-LABEL: imp_null_check_load_fence2:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:    testq %rdi, %rdi
; CHECK-NEXT:    je LBB17_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    mfence
; CHECK-NEXT:    movl (%rdi), %eax
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB17_1: ## %is_null
; CHECK-NEXT:    movl $42, %eax
; CHECK-NEXT:    retq

entry:
  %c = icmp eq ptr %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

is_null:
  ret i32 42

not_null:
  fence seq_cst
  %t = load i32, ptr %x
  ret i32 %t
}

define void @imp_null_check_store(ptr %x) {
; CHECK-LABEL: imp_null_check_store:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:  Ltmp14:
; CHECK-NEXT:    movl $1, (%rdi) ## on-fault: LBB18_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB18_1: ## %is_null
; CHECK-NEXT:    retq

 entry:
  %c = icmp eq ptr %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret void

 not_null:
  store i32 1, ptr %x
  ret void
}

;; TODO: can be implicit
define void @imp_null_check_unordered_store(ptr %x) {
; CHECK-LABEL: imp_null_check_unordered_store:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:  Ltmp15:
; CHECK-NEXT:    movl $1, (%rdi) ## on-fault: LBB19_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB19_1: ## %is_null
; CHECK-NEXT:    retq

 entry:
  %c = icmp eq ptr %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret void

 not_null:
  store atomic i32 1, ptr %x unordered, align 4
  ret void
}

define i32 @imp_null_check_neg_gep_load(ptr %x) {
; CHECK-LABEL: imp_null_check_neg_gep_load:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:  Ltmp16:
; CHECK-NEXT:    movl -128(%rdi), %eax ## on-fault: LBB20_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB20_1: ## %is_null
; CHECK-NEXT:    movl $42, %eax
; CHECK-NEXT:    retq

 entry:
  %c = icmp eq ptr %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %x.gep = getelementptr i32, ptr %x, i32 -32
  %t = load i32, ptr %x.gep
  ret i32 %t
}

; This redefines the null check reg by doing a zero-extend and a shift on
; itself.
; Converted into implicit null check since both of these operations do not
; change the nullness of %x (i.e. if it is null, it remains null).
define i64 @imp_null_check_load_shift_addr(ptr %x) {
; CHECK-LABEL: imp_null_check_load_shift_addr:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:    shlq $6, %rdi
; CHECK-NEXT:  Ltmp17:
; CHECK-NEXT:    movq 8(%rdi), %rax ## on-fault: LBB21_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB21_1: ## %is_null
; CHECK-NEXT:    movl $42, %eax
; CHECK-NEXT:    retq

  entry:
   %c = icmp eq ptr %x, null
   br i1 %c, label %is_null, label %not_null, !make.implicit !0

  is_null:
   ret i64 42

  not_null:
   %y = ptrtoint ptr %x to i64
   %shry = shl i64 %y, 6
   %y.ptr = inttoptr i64 %shry to ptr
   %x.loc = getelementptr i64, ptr %y.ptr, i64 1
   %t = load i64, ptr %x.loc
   ret i64 %t
}

; Same as imp_null_check_load_shift_addr but shift is by 3 and this is now
; converted into complex addressing.
define i64 @imp_null_check_load_shift_by_3_addr(ptr %x) {
; CHECK-LABEL: imp_null_check_load_shift_by_3_addr:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:  Ltmp18:
; CHECK-NEXT:    movq 8(,%rdi,8), %rax ## on-fault: LBB22_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB22_1: ## %is_null
; CHECK-NEXT:    movl $42, %eax
; CHECK-NEXT:    retq

  entry:
   %c = icmp eq ptr %x, null
   br i1 %c, label %is_null, label %not_null, !make.implicit !0

  is_null:
   ret i64 42

  not_null:
   %y = ptrtoint ptr %x to i64
   %shry = shl i64 %y, 3
   %y.ptr = inttoptr i64 %shry to ptr
   %x.loc = getelementptr i64, ptr %y.ptr, i64 1
   %t = load i64, ptr %x.loc
   ret i64 %t
}

define i64 @imp_null_check_load_shift_add_addr(ptr %x) {
; CHECK-LABEL: imp_null_check_load_shift_add_addr:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:  Ltmp19:
; CHECK-NEXT:    movq 3526(,%rdi,8), %rax ## on-fault: LBB23_1
; CHECK-NEXT:  ## %bb.2: ## %not_null
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB23_1: ## %is_null
; CHECK-NEXT:    movl $42, %eax
; CHECK-NEXT:    retq

  entry:
   %c = icmp eq ptr %x, null
   br i1 %c, label %is_null, label %not_null, !make.implicit !0

  is_null:
   ret i64 42

  not_null:
   %y = ptrtoint ptr %x to i64
   %shry = shl i64 %y, 3
   %shry.add = add i64 %shry, 3518
   %y.ptr = inttoptr i64 %shry.add to ptr
   %x.loc = getelementptr i64, ptr %y.ptr, i64 1
   %t = load i64, ptr %x.loc
   ret i64 %t
}
!0 = !{}
