; RUN: opt < %s -S -speculative-execution \
; RUN:   -spec-exec-max-speculation-cost 4 -spec-exec-max-not-hoisted 3 \
; RUN:   | FileCheck %s

; CHECK-LABEL: @ifThen_bitcast(
; CHECK: bitcast
; CHECK: br i1 true
define void @ifThen_bitcast() {
  br i1 true, label %a, label %b

a:
  %x = bitcast i32 undef to float
  br label %b

b:
  ret void
}

; CHECK-LABEL: @ifThen_ptrtoint(
; CHECK: ptrtoint
; CHECK: br i1 true
define void @ifThen_ptrtoint() {
  br i1 true, label %a, label %b

a:
  %x = ptrtoint i32* undef to i64
  br label %b

b:
  ret void
}

; CHECK-LABEL: @ifThen_inttoptr(
; CHECK: inttoptr
; CHECK: br i1 true
define void @ifThen_inttoptr() {
  br i1 true, label %a, label %b

a:
  %x = inttoptr i64 undef to i32*
  br label %b

b:
  ret void
}

; CHECK-LABEL: @ifThen_addrspacecast(
; CHECK: addrspacecast
; CHECK: br i1 true
define void @ifThen_addrspacecast() {
  br i1 true, label %a, label %b
a:
  %x = addrspacecast i32* undef to i32 addrspace(1)*
  br label %b

b:
  ret void
}

; CHECK-LABEL: @ifThen_fptoui(
; CHECK: fptoui
; CHECK: br i1 true
define void @ifThen_fptoui() {
  br i1 true, label %a, label %b
a:
  %x = fptoui float undef to i32
  br label %b

b:
  ret void
}

; CHECK-LABEL: @ifThen_fptosi(
; CHECK: fptosi
; CHECK: br i1 true
define void @ifThen_fptosi() {
  br i1 true, label %a, label %b
a:
  %x = fptosi float undef to i32
  br label %b

b:
  ret void
}

; CHECK-LABEL: @ifThen_uitofp(
; CHECK: uitofp
; CHECK: br i1 true
define void @ifThen_uitofp() {
  br i1 true, label %a, label %b
a:
  %x = uitofp i32 undef to float
  br label %b

b:
  ret void
}

; CHECK-LABEL: @ifThen_sitofp(
; CHECK: sitofp
; CHECK: br i1 true
define void @ifThen_sitofp() {
  br i1 true, label %a, label %b
a:
  %x = sitofp i32 undef to float
  br label %b

b:
  ret void
}

; CHECK-LABEL: @ifThen_fpext(
; CHECK: fpext
; CHECK: br i1 true
define void @ifThen_fpext() {
  br i1 true, label %a, label %b
a:
  %x = fpext float undef to double
  br label %b

b:
  ret void
}

; CHECK-LABEL: @ifThen_fptrunc(
; CHECK: fptrunc
; CHECK: br i1 true
define void @ifThen_fptrunc() {
  br i1 true, label %a, label %b
a:
  %x = fptrunc double undef to float
  br label %b

b:
  ret void
}

; CHECK-LABEL: @ifThen_trunc(
; CHECK: trunc
; CHECK: br i1 true
define void @ifThen_trunc() {
  br i1 true, label %a, label %b
a:
  %x = trunc i32 undef to i16
  br label %b

b:
  ret void
}
