; RUN: llc -mtriple=agx -mcpu=agx-g14c -stop-after=structurizecfg -debug-pass=Executions %s -o - 2>&1 | FileCheck %s --check-prefix=FULL --implicit-check-not="Executing Pass 'Uniformity Analysis'" --implicit-check-not=structurizecfg.uniform
; RUN: llc -mtriple=agx -mcpu=agx-g14c -stop-after=structurizecfg -debug-pass=Executions -structurizecfg-skip-uniform-regions=true %s -o - 2>&1 | FileCheck %s --check-prefix=SKIP --implicit-check-not="Executing Pass 'Uniformity Analysis'"
;
; The legacy region pass must not schedule uniformity when all regions are
; structurized. The library-option override still computes the original
; function's uniformity once: uniform branches get metadata, lane-dependent
; branches do not. Multiple functions also check the analysis is reset.
;
; FULL: define agx_kernel void @uniform_first(
; FULL: define agx_kernel void @divergent(
; FULL: define agx_kernel void @uniform_last(
;
; SKIP: define agx_kernel void @uniform_first(
; SKIP: br i1 %cond, label %then, label %end, !structurizecfg.uniform
; SKIP: define agx_kernel void @divergent(
; SKIP-NOT: !structurizecfg.uniform
; SKIP: define agx_kernel void @uniform_last(
; SKIP: br i1 %cond, label %then, label %end, !structurizecfg.uniform
;
declare i32 @llvm.agx.thread.position.in.grid.x()

define agx_kernel void @uniform_first(ptr addrspace(1) %out, i32 %n) {
entry:
  %cond = icmp eq i32 %n, 0
  br i1 %cond, label %then, label %end
then:
  store i32 42, ptr addrspace(1) %out
  br label %end
end:
  ret void
}

define agx_kernel void @divergent(ptr addrspace(1) %out) {
entry:
  %id = call i32 @llvm.agx.thread.position.in.grid.x()
  %cond = icmp eq i32 %id, 0
  br i1 %cond, label %then, label %end
then:
  store i32 42, ptr addrspace(1) %out
  br label %end
end:
  ret void
}

define agx_kernel void @uniform_last(ptr addrspace(1) %out, i32 %n) {
entry:
  %cond = icmp eq i32 %n, 0
  br i1 %cond, label %then, label %end
then:
  store i32 24, ptr addrspace(1) %out
  br label %end
end:
  ret void
}
