; RUN: llc -emit-call-site-info -stop-after=dead-mi-elimination -mtriple=amdgcn-- -mcpu=gfx900 -o - %s | FileCheck %s

; CHECK-LABEL: name:            basic_call
; CHECK: $sgpr30_sgpr31 = SI_CALL {{.*}}, @foo, csr_amdgpu

define i32 @basic_call(i32 %src) #0 {
  %token = call token @llvm.experimental.convergence.entry()
  %result = call i32 @foo(i32 %src) [ "convergencectrl"(token %token) ]
  ret i32 %result
}

declare i32 @foo(i32) #0
declare token @llvm.experimental.convergence.entry()

attributes #0 = { convergent nounwind readnone }
