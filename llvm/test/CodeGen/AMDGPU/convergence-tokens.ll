; RUN: llc --amdgpu-disable-structurizer -stop-after=amdgpu-isel -mtriple=amdgcn-- -mcpu=gfx900 -verify-machineinstrs -o - %s | FileCheck --check-prefixes=CHECK,ISEL %s
; RUN: llc --amdgpu-disable-structurizer -stop-after=dead-mi-elimination -mtriple=amdgcn-- -mcpu=gfx900 -verify-machineinstrs -o - %s | FileCheck --check-prefixes=CHECK,DEADMI %s

; CHECK-LABEL: name:            basic_call
;       CHECK:    [[TOKEN:%[0-9]+]]:sreg_64 = CONVERGENCECTRL_ENTRY
;        ISEL:    {{.*}} SI_CALL_ISEL {{.*}}, @foo, [[TOKEN]], csr_amdgpu, {{.*}}
;      DEADMI:    {{.*}} SI_CALL {{.*}}, @foo, csr_amdgpu, {{.*}}, implicit [[TOKEN]]
define i32 @basic_call(i32 %src) #0 {
  %t = call token @llvm.experimental.convergence.entry()
  %r = call i32 @foo(i32 %src) [ "convergencectrl"(token %t) ]
  ret i32 %r
}

; CHECK-LABEL: name:            basic_intrinsic
;       CHECK:    [[TOKEN:%[0-9]+]]:sreg_64 = CONVERGENCECTRL_ANCHOR
;        ISEL:    CONVERGENCECTRL_GLUE [[TOKEN]]
;  DEADMI-NOT:    CONVERGENCECTRL_GLUE
;       CHECK:    {{.*}} = V_READFIRSTLANE_B32 {{.*}}, implicit [[TOKEN]]
define i32 @basic_intrinsic(i32 %src) #0 {
  %t = call token @llvm.experimental.convergence.anchor()
  %r = call i32 @llvm.amdgcn.readfirstlane(i32 %src) [ "convergencectrl"(token %t) ]
  ret i32 %r
}

; There's nothing to check here. The test is just meant to catch any crashes
; when a convergent call has no token.
define i32 @uncontrolled_call(i32 %src) #0 {
  %r = call i32 @foo(i32 %src)
  ret i32 %r
}

; CHECK-LABEL: name:            basic_branch
;       CHECK:  bb.0.entry:
;       CHECK:    [[TOKEN:%[0-9]+]]:sreg_64 = CONVERGENCECTRL_ANCHOR
;       CHECK:  bb.1.then:
;        ISEL:    CONVERGENCECTRL_GLUE [[TOKEN]]
;  DEADMI-NOT:    CONVERGENCECTRL_GLUE
;       CHECK:    {{.*}} = V_READFIRSTLANE_B32 {{.*}}, implicit [[TOKEN]]
define i32 @basic_branch(i32 %src, i1 %cond) #0 {
entry:
  %t = call token @llvm.experimental.convergence.anchor()
  %x = add i32 %src, 1
  br i1 %cond, label %then, label %else

then:
  %r = call i32 @llvm.amdgcn.readfirstlane(i32 %x) [ "convergencectrl"(token %t) ]
  br label %else

else:
  %p = phi i32 [%r, %then], [%x, %entry]
  ret i32 %p
}

; CHECK-LABEL: name:            basic_loop
;       CHECK:    [[TOKEN:%[0-9]+]]:sreg_64 = CONVERGENCECTRL_ANCHOR
;       CHECK:  bb.1.loop:
;       CHECK:    [[LOOP:%[0-9]+]]:sreg_64 = CONVERGENCECTRL_LOOP [[TOKEN]]
;        ISEL:    CONVERGENCECTRL_GLUE [[LOOP]]
;  DEADMI-NOT:    CONVERGENCECTRL_GLUE
;       CHECK:    {{.*}} = V_READFIRSTLANE_B32 {{.*}}, implicit [[LOOP]]
define i32 @basic_loop(i32 %src, i1 %cond) #0 {
  %t1 = call token @llvm.experimental.convergence.anchor()
  br label %loop

loop:
  %t2 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %t1) ]
  %r = call i32 @llvm.amdgcn.readfirstlane(i32 %src) [ "convergencectrl"(token %t2) ]
  br i1 %cond, label %loop, label %end

end:
  ret i32 %r
}

declare i32 @foo(i32 %x) #0

declare i32 @llvm.amdgcn.readfirstlane(i32) #0

declare token @llvm.experimental.convergence.entry()
declare token @llvm.experimental.convergence.anchor()
declare token @llvm.experimental.convergence.loop()

attributes #0 = { nounwind readnone convergent }
attributes #1 = { nounwind }
