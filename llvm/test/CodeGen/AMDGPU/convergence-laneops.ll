; RUN: llc -stop-after=amdgpu-isel -mtriple=amdgcn-- -mcpu=gfx1100 -verify-machineinstrs -o - %s | FileCheck --check-prefixes=CHECK,ISEL %s
; RUN: not --crash llc -mtriple=amdgcn--amdhsa -mcpu=1100 -verify-machineinstrs < %s 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s

; FIXME: Merge these tests with existing lane op tests (llvm.amdgcn.readlane.ll, llvm.amdgcn.writelane.ll ...) once the crash is fixed.

; CHECK-LABEL: name:            basic_readfirstlane_i64
;       CHECK:    [[TOKEN:%[0-9]+]]{{[^ ]*}} = CONVERGENCECTRL_ANCHOR
;        ISEL:    CONVERGENCECTRL_GLUE [[TOKEN]]
;        ISEL:    {{.*}} = V_READFIRSTLANE_B32 {{.*}}, implicit [[TOKEN]]
;        ISEL:    CONVERGENCECTRL_GLUE [[TOKEN]]
;        ISEL:    {{.*}} = V_READFIRSTLANE_B32 {{.*}}, implicit [[TOKEN]]
define i64 @basic_readfirstlane_i64(i64 %src, i1 %cond) #0 {
entry:
  %t = call token @llvm.experimental.convergence.anchor()
  %x = add i64 %src, 1
  br i1 %cond, label %then, label %else

then:
; CHECK-ERROR: Cannot mix controlled and uncontrolled convergence in the same function.
; CHECK-ERROR: V_READFIRSTLANE_B32
  %r = call i64 @llvm.amdgcn.readfirstlane.i64(i64 %x) [ "convergencectrl"(token %t) ]
  br label %else

else:
  %p = phi i64 [%r, %then], [%x, %entry]
  ret i64 %p
}

; CHECK-LABEL: name:            basic_readlane_i64
;       CHECK:    [[TOKEN:%[0-9]+]]{{[^ ]*}} = CONVERGENCECTRL_ANCHOR
;        ISEL:    CONVERGENCECTRL_GLUE [[TOKEN]]
;        ISEL:    {{.*}} = V_READLANE_B32 {{.*}}, implicit [[TOKEN]]
;        ISEL:    CONVERGENCECTRL_GLUE [[TOKEN]]
;        ISEL:    {{.*}} = V_READLANE_B32 {{.*}}, implicit [[TOKEN]]
define i64 @basic_readlane_i64(i64 %src, i32 %lane, i1 %cond) #0 {
entry:
  %t = call token @llvm.experimental.convergence.anchor()
  %x = add i64 %src, 1
  br i1 %cond, label %then, label %else

then:
  %r = call i64 @llvm.amdgcn.readlane.i64(i64 %x, i32 %lane) [ "convergencectrl"(token %t) ]
  br label %else

else:
  %p = phi i64 [%r, %then], [%x, %entry]
  ret i64 %p
}

; CHECK-LABEL: name:            basic_writelane_i64
;       CHECK:    [[TOKEN:%[0-9]+]]{{[^ ]*}} = CONVERGENCECTRL_ANCHOR
;        ISEL:    CONVERGENCECTRL_GLUE [[TOKEN]]
;        ISEL:    {{.*}} = V_WRITELANE_B32 {{.*}}, implicit [[TOKEN]]
;        ISEL:    CONVERGENCECTRL_GLUE [[TOKEN]]
;        ISEL:    {{.*}} = V_WRITELANE_B32 {{.*}}, implicit [[TOKEN]]
define i64 @basic_writelane_i64(i64 %src, i1 %cond, i32 %lane, ptr addrspace(1) %out) #0 {
entry:
  %old = load i64, ptr addrspace(1) %out
  %t = call token @llvm.experimental.convergence.anchor()
  %x = add i64 %src, 1
  br i1 %cond, label %then, label %else

then:
  %r = call i64 @llvm.amdgcn.writelane.i64(i64 %x, i32 %lane, i64 %old) [ "convergencectrl"(token %t) ]
  br label %else

else:
  %p = phi i64 [%r, %then], [%x, %entry]
  ret i64 %p
}
