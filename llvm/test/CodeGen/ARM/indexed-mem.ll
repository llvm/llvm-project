; RUN: llc -mtriple=thumbv8m.main %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-V8M
; RUN: llc -mtriple=armv8 %s -o - | FileCheck %s --check-prefix=CHECK-V8

; CHECK-LABEL: pre_inc_ldr
; CHECK: ldr{{.*}}, [r0, #4]!
define i32* @pre_inc_ldr(i32* %base, i32 %a) {
  %addr = getelementptr i32, i32* %base, i32 1
  %ld = load i32, i32* %addr
  %addr.1 = getelementptr i32, i32* %base, i32 2
  %res = add i32 %ld, %a
  store i32 %res, i32* %addr.1
  ret i32* %addr
}

; CHECK-LABEL: pre_dec_ldr
; CHECK: ldr{{.*}}, [r0, #-4]!
define i32* @pre_dec_ldr(i32* %base, i32 %a) {
  %addr = getelementptr i32, i32* %base, i32 -1
  %ld = load i32, i32* %addr
  %addr.1 = getelementptr i32, i32* %base, i32 2
  %res = add i32 %ld, %a
  store i32 %res, i32* %addr.1
  ret i32* %addr
}

; CHECK-LABEL: post_inc_ldr
; CHECK: ldr{{.*}}, [r0], #4
define i32* @post_inc_ldr(i32* %base, i32* %addr.2, i32 %a) {
  %addr = getelementptr i32, i32* %base, i32 0
  %ld = load i32, i32* %addr
  %addr.1 = getelementptr i32, i32* %base, i32 1
  %res = add i32 %ld, %a
  store i32 %res, i32* %addr.2
  ret i32* %addr.1
}

; CHECK-LABEL: post_dec_ldr
; CHECK: ldr{{.*}}, [r0], #-4
define i32* @post_dec_ldr(i32* %base, i32* %addr.2, i32 %a) {
  %addr = getelementptr i32, i32* %base, i32 0
  %ld = load i32, i32* %addr
  %addr.1 = getelementptr i32, i32* %base, i32 -1
  %res = add i32 %ld, %a
  store i32 %res, i32* %addr.2
  ret i32* %addr.1
}

; CHECK-LABEL: pre_inc_str
; CHECK: str{{.*}}, [r0, #4]!
define i32* @pre_inc_str(i32* %base, i32 %a, i32 %b) {
  %addr.1 = getelementptr i32, i32* %base, i32 1
  %res = add i32 %a, %b
  store i32 %res, i32* %addr.1
  ret i32* %addr.1
}

; CHECK-LABEL: pre_dec_str
; CHECK: str{{.*}}, [r0, #-4]!
define i32* @pre_dec_str(i32* %base, i32 %a, i32 %b) {
  %res = add i32 %a, %b
  %addr.1 = getelementptr i32, i32* %base, i32 -1
  store i32 %res, i32* %addr.1
  ret i32* %addr.1
}

; CHECK-LABEL: post_inc_str
; CHECK: str{{.*}}, [r0], #4
define i32* @post_inc_str(i32* %base, i32 %a, i32 %b) {
  %addr.1 = getelementptr i32, i32* %base, i32 1
  %res = add i32 %a, %b
  store i32 %res, i32* %base
  ret i32* %addr.1
}

; CHECK-LABEL: post_dec_str
; CHECK: str{{.*}}, [r0], #-4
define i32* @post_dec_str(i32* %base, i32 %a, i32 %b) {
  %addr.1 = getelementptr i32, i32* %base, i32 -1
  %res = add i32 %a, %b
  store i32 %res, i32* %base
  ret i32* %addr.1
}

; TODO: Generate ldrd
; CHECK-LABEL: pre_inc_ldrd
; CHECK: ldr{{.*}}, #4]!
define i32* @pre_inc_ldrd(i32* %base) {
  %addr = getelementptr i32, i32* %base, i32 1
  %addr.1 = getelementptr i32, i32* %base, i32 2
  %addr.2 = getelementptr i32, i32* %base, i32 3
  %ld = load i32, i32* %addr
  %ld.1 = load i32, i32* %addr.1
  %res = add i32 %ld, %ld.1
  store i32 %res, i32* %addr.2
  ret i32* %addr
}

; TODO: Generate ldrd
; CHECK-LABEL: pre_dec_ldrd
; CHECK: ldr{{.*}}, #-4]!
define i32* @pre_dec_ldrd(i32* %base) {
  %addr = getelementptr i32, i32* %base, i32 -1
  %addr.1 = getelementptr i32, i32* %base, i32 -2
  %addr.2 = getelementptr i32, i32* %base, i32 -3
  %ld = load i32, i32* %addr
  %ld.1 = load i32, i32* %addr.1
  %res = add i32 %ld, %ld.1
  store i32 %res, i32* %addr.2
  ret i32* %addr
}

; TODO: Generate post inc
; CHECK-LABEL: post_inc_ldrd
; CHECK-V8M: ldrd{{.*}}, [r0]
; CHECK-V8: ldm
; CHECK: add{{.*}}, #8
define i32* @post_inc_ldrd(i32* %base, i32* %addr.3) {
  %addr = getelementptr i32, i32* %base, i32 0
  %ld = load i32, i32* %addr
  %addr.1 = getelementptr i32, i32* %base, i32 1
  %ld.1 = load i32, i32* %addr.1
  %addr.2 = getelementptr i32, i32* %base, i32 2
  %res = add i32 %ld, %ld.1
  store i32 %res, i32* %addr.3
  ret i32* %addr.2
}

; CHECK-LABEL: pre_inc_str_multi
; CHECK: str{{.*}}, #8]!
define i32* @pre_inc_str_multi(i32* %base) {
  %addr = getelementptr i32, i32* %base, i32 0
  %addr.1 = getelementptr i32, i32* %base, i32 1
  %ld = load i32, i32* %addr
  %ld.1 = load i32, i32* %addr.1
  %res = add i32 %ld, %ld.1
  %addr.2 = getelementptr i32, i32* %base, i32 2
  store i32 %res, i32* %addr.2
  ret i32* %addr.2
}

; CHECK-LABEL: pre_dec_str_multi
; CHECK: str{{.*}}, #-4]!
define i32* @pre_dec_str_multi(i32* %base) {
  %addr = getelementptr i32, i32* %base, i32 0
  %addr.1 = getelementptr i32, i32* %base, i32 1
  %ld = load i32, i32* %addr
  %ld.1 = load i32, i32* %addr.1
  %res = add i32 %ld, %ld.1
  %addr.2 = getelementptr i32, i32* %base, i32 -1
  store i32 %res, i32* %addr.2
  ret i32* %addr.2
}

; CHECK-LABEL: illegal_pre_inc_store_1
; CHECK-NOT: str{{.*}} ]!
define i32* @illegal_pre_inc_store_1(i32* %base) {
entry:
  %ptr.to.use = getelementptr i32, i32* %base, i32 2
  %ptr.to.store = ptrtoint i32* %base to i32
  store i32 %ptr.to.store, i32* %ptr.to.use, align 4
  ret i32* %ptr.to.use
}

; TODO: The mov should be unecessary
; CHECK-LABEL: legal_pre_inc_store_needs_copy_1
; CHECK: add{{.*}}, #8
; CHECK-NOT: str{{.*}}]!
; CHECK: mov
define i32* @legal_pre_inc_store_needs_copy_1(i32* %base) {
entry:
  %ptr.to.use = getelementptr i32, i32* %base, i32 2
  %ptr.to.store = ptrtoint i32* %ptr.to.use to i32
  store i32 %ptr.to.store, i32* %ptr.to.use, align 4
  ret i32* %ptr.to.use
}

; CHECK-LABEL: legal_pre_inc_store_needs_copy_2
; CHECK-NOT: mov
; CHECK-NOT: str{{.*}}]!
define i32* @legal_pre_inc_store_needs_copy_2(i32 %base) {
entry:
  %ptr = inttoptr i32 %base to i32*
  %ptr.to.use = getelementptr i32, i32* %ptr, i32 2
  store i32 %base, i32* %ptr.to.use, align 4
  ret i32* %ptr.to.use
}
