; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 -O3 | FileCheck %s
;
; Check that CodeGenPrepare transforms these functions to use
; uadd.with.overflow / usub.with.overflow intrinsics so that the compare
; instruction is eliminated.

define i32 @uaddo_32(i32 %arg)  {
; CHECK-LABEL: uaddo_32:
; CHECK: alhsik	 %r0, %r2, -1
; CHECK: locrnle %r2, %r0
; CHECK: br      %r14

bb:
  %tmp10 = icmp ne i32 %arg, 0
  %tmp11 = add nsw i32 %arg, -1
  %tmp12 = select i1 %tmp10, i32 %tmp11, i32 %arg
  ret i32 %tmp12
}

define i64 @uaddo_64(i64 %arg)  {
; CHECK-LABEL: uaddo_64:
; CHECK: alghsik  %r0, %r2, -1
; CHECK: locgrnle %r2, %r0
; CHECK: br       %r14
bb:
  %tmp10 = icmp ne i64 %arg, 0
  %tmp11 = add nsw i64 %arg, -1
  %tmp12 = select i1 %tmp10, i64 %tmp11, i64 %arg
  ret i64 %tmp12
}

define i32 @usubo_32(i32 %arg)  {
; CHECK-LABEL: usubo_32:
; CHECK: alhsik %r0, %r2, -1
; CHECK: locrle %r2, %r0
; CHECK: br     %r14
bb:
  %tmp10 = icmp eq i32 %arg, 0
  %tmp11 = sub nsw i32 %arg, 1
  %tmp12 = select i1 %tmp10, i32 %tmp11, i32 %arg
  ret i32 %tmp12
}

define i64 @usubo_64(i64 %arg)  {
; CHECK-LABEL: usubo_64:
; CHECK: alghsik %r0, %r2, -1
; CHECK: locgrle %r2, %r0
; CHECK: br      %r14
bb:
  %tmp10 = icmp eq i64 %arg, 0
  %tmp11 = sub nsw i64 %arg, 1
  %tmp12 = select i1 %tmp10, i64 %tmp11, i64 %arg
  ret i64 %tmp12
}
