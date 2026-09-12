; RUN: opt -mtriple=amdgpu-- -passes=loop-unroll -S %s | FileCheck %s --check-prefixes=CHECK,DEFAULT
; RUN: opt -mtriple=amdgpu-- -passes=loop-unroll -amdgpu-unroll-runtime-local=false -S %s | FileCheck %s --check-prefixes=CHECK,NOLOCAL

; -amdgpu-unroll-runtime-local gates runtime unrolling per loop, based on
; whether that loop touches local (LDS, addrspace(3)) memory. With the knob
; off, only the LDS loop is suppressed; the global-memory loop is unaffected.
;
;                     knob=true (default)   knob=false
;   %lds.loop            unrolled              NOT unrolled
;   %global.loop         unrolled              unrolled (knob has no effect)

@lds = internal unnamed_addr addrspace(3) global [256 x i32] poison, align 4

; CHECK-LABEL: @two_loops(
define void @two_loops(ptr addrspace(1) %out, i32 %n, i32 %m) {
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %lds.loop, label %global.preheader

; The LDS loop: gated by the knob. Its runtime-unroll epilogue block
; (lds.loop.epil) appears only when the knob is on.
;
; DEFAULT: lds.loop.epil:
;
; NOLOCAL-NOT: lds.loop.epil
lds.loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %lds.loop ]
  %idx = zext i32 %iv to i64
  %ptr = getelementptr inbounds [256 x i32], ptr addrspace(3) @lds, i64 0, i64 %idx
  store i32 %iv, ptr addrspace(3) %ptr, align 4
  %iv.next = add nuw nsw i32 %iv, 1
  %exitcond = icmp eq i32 %iv.next, %n
  br i1 %exitcond, label %global.preheader, label %lds.loop

global.preheader:
  %cmp2 = icmp sgt i32 %m, 0
  br i1 %cmp2, label %global.loop, label %exit

; The global-memory loop: never touches LDS, so the knob must not affect it.
; Its runtime-unroll epilogue block (global.loop.epil) appears under both
; knob settings.
;
; DEFAULT: global.loop.epil:
;
; NOLOCAL: global.loop.epil:
global.loop:
  %jv = phi i32 [ 0, %global.preheader ], [ %jv.next, %global.loop ]
  %jdx = zext i32 %jv to i64
  %gptr = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %jdx
  store i32 %jv, ptr addrspace(1) %gptr, align 4
  %jv.next = add nuw nsw i32 %jv, 1
  %exitcond2 = icmp eq i32 %jv.next, %m
  br i1 %exitcond2, label %exit, label %global.loop

exit:
  ret void
}
