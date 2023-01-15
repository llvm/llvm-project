; RUN: opt < %s -S -mtriple=amdgcn-unknown-amdhsa -passes=speculative-execution \
; RUN:   -spec-exec-max-speculation-cost 1 -spec-exec-max-not-hoisted 1 \
; RUN:   | FileCheck %s

; CHECK-LABEL: @ifThen_bitcast(
; CHECK: bitcast
; CHECK: br i1 true
define void @ifThen_bitcast(i32 %y) {
  br i1 true, label %a, label %b

a:
  %x = bitcast i32 %y to float
  br label %b

b:
  ret void
}

; CHECK-LABEL: @ifThen_addrspacecast(
; CHECK: addrspacecast
; CHECK: br i1 true
define void @ifThen_addrspacecast(ptr %y) {
  br i1 true, label %a, label %b
a:
  %x = addrspacecast ptr %y to ptr addrspace(1)
  br label %b

b:
  ret void
}
