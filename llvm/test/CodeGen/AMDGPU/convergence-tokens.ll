; RUN: llc --amdgpu-disable-structurizer -stop-after=amdgpu-isel -mtriple=amdgcn-- -mcpu=gfx900 -verify-machineinstrs -o - %s | FileCheck --check-prefixes=CHECK,ISEL %s
; RUN: llc --amdgpu-disable-structurizer -stop-after=dead-mi-elimination -mtriple=amdgcn-- -mcpu=gfx900 -verify-machineinstrs -o - %s | FileCheck --check-prefixes=CHECK,DEADMI %s
; RUN: llc --amdgpu-disable-structurizer -global-isel -stop-after=irtranslator -mtriple=amdgcn-- -mcpu=gfx900 -verify-machineinstrs -o - %s | FileCheck %s --check-prefixes=CHECK,GISEL

; CHECK-LABEL: name:            basic_call
;       CHECK:    [[TOKEN:%[0-9]+]]{{[^ ]*}} = CONVERGENCECTRL_ENTRY
;        ISEL:    {{.*}} SI_CALL_ISEL {{.*}}, @foo, [[TOKEN]], csr_amdgpu, {{.*}}
;      DEADMI:    {{.*}} SI_CALL {{.*}}, @foo, csr_amdgpu, {{.*}}, implicit [[TOKEN]]
;       GISEL:    {{.*}} G_SI_CALL {{.*}}, @foo, csr_amdgpu, {{.*}}, implicit [[TOKEN]]
define i32 @basic_call(i32 %src) #0 {
  %t = call token @llvm.experimental.convergence.entry()
  %r = call i32 @foo(i32 %src) [ "convergencectrl"(token %t) ]
  ret i32 %r
}

; CHECK-LABEL: name:            basic_intrinsic
;       CHECK:    [[TOKEN:%[0-9]+]]{{[^ ]*}} = CONVERGENCECTRL_ANCHOR
;        ISEL:    CONVERGENCECTRL_GLUE [[TOKEN]]
;  DEADMI-NOT:    CONVERGENCECTRL_GLUE
;        ISEL:    {{.*}} = V_READFIRSTLANE_B32 {{.*}}, implicit [[TOKEN]]
;       GISEL:    {{.*}} = G_INTRINSIC_CONVERGENT intrinsic(@llvm.amdgcn.readfirstlane){{.*}}, implicit [[TOKEN]]
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
;       CHECK:  bb.[[#]].entry:
;       CHECK:    [[TOKEN:%[0-9]+]]{{[^ ]*}} = CONVERGENCECTRL_ANCHOR
;       CHECK:  bb.[[#]].then:
;        ISEL:    CONVERGENCECTRL_GLUE [[TOKEN]]
;  DEADMI-NOT:    CONVERGENCECTRL_GLUE
;        ISEL:    {{.*}} = V_READFIRSTLANE_B32 {{.*}}, implicit [[TOKEN]]
;       GISEL:    {{.*}} = G_INTRINSIC_CONVERGENT intrinsic(@llvm.amdgcn.readfirstlane){{.*}}, implicit [[TOKEN]]
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
;       CHECK:    [[TOKEN:%[0-9]+]]{{[^ ]*}} = CONVERGENCECTRL_ANCHOR
;       CHECK:  bb.[[#]].loop:
;       CHECK:    [[LOOP:%[0-9]+]]{{[^ ]*}} = CONVERGENCECTRL_LOOP [[TOKEN]]
;        ISEL:    CONVERGENCECTRL_GLUE [[LOOP]]
;  DEADMI-NOT:    CONVERGENCECTRL_GLUE
;        ISEL:    {{.*}} = V_READFIRSTLANE_B32 {{.*}}, implicit [[LOOP]]
;       GISEL:    {{.*}} = G_INTRINSIC_CONVERGENT intrinsic(@llvm.amdgcn.readfirstlane){{.*}}, implicit [[LOOP]]
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

; CHECK-LABEL: name: nested
;       CHECK:    [[ENTRY:%[0-9]+]]{{[^ ]*}} = CONVERGENCECTRL_ENTRY
;       CHECK:    [[ANCHOR:%[0-9]+]]{{[^ ]*}} = CONVERGENCECTRL_ANCHOR
;        ISEL:    {{.*}} = V_READFIRSTLANE_B32 {{.*}}, implicit [[ANCHOR]]
;       GISEL:    {{.*}} = G_INTRINSIC_CONVERGENT intrinsic(@llvm.amdgcn.readfirstlane){{.*}}, implicit [[ANCHOR]]
;        ISEL:    {{.*}} = V_READFIRSTLANE_B32 {{.*}}, implicit [[ENTRY]]
;       GISEL:    {{.*}} = G_INTRINSIC_CONVERGENT intrinsic(@llvm.amdgcn.readfirstlane){{.*}}, implicit [[ENTRY]]
define i32 @nested(i32 %src) #0 {
  %t1 = call token @llvm.experimental.convergence.entry()
  %t2 = call token @llvm.experimental.convergence.anchor()
  %r2 = call i32 @llvm.amdgcn.readfirstlane(i32 %src) [ "convergencectrl"(token %t2) ]
  %r1 = call i32 @llvm.amdgcn.readfirstlane(i32 %src) [ "convergencectrl"(token %t1) ]
  %sum = add i32 %r1, %r2
  ret i32 %sum
}

; COM: FIXME: Tokens on tail-call have not been implemented for SelectionDAG
; COM:        yet; the corresponding checks have been commented out.
;
; CHECK-LABEL: name:            tail_call_void_func_void
;       GISEL:    [[TOKEN:%[0-9]+]]{{[^ ]*}} = CONVERGENCECTRL_ENTRY
; COM:  CHECK:    [[TOKEN:%[0-9]+]]{{[^ ]*}} = CONVERGENCECTRL_ENTRY
; COM:   ISEL:    {{.*}} SI_CALL_ISEL {{.*}}, @external_void_func_void, [[TOKEN]], csr_amdgpu, {{.*}}
; COM: DEADMI:    {{.*}} SI_CALL {{.*}}, @external_void_func_void, csr_amdgpu, {{.*}}, implicit [[TOKEN]]
;       GISEL:    {{.*}} SI_TCRETURN {{.*}}, @external_void_func_void, 0, csr_amdgpu, implicit [[TOKEN]]
define void @tail_call_void_func_void() #0 {
  %t1 = call token @llvm.experimental.convergence.entry()
  tail call void @external_void_func_void() [ "convergencectrl"(token %t1) ]
  ret void
}

declare hidden void @external_void_func_void() #0
declare i32 @foo(i32 %x) #0

declare i32 @llvm.amdgcn.readfirstlane(i32) #0

declare token @llvm.experimental.convergence.entry()
declare token @llvm.experimental.convergence.anchor()
declare token @llvm.experimental.convergence.loop()

attributes #0 = { nounwind readnone convergent }
attributes #1 = { nounwind }
