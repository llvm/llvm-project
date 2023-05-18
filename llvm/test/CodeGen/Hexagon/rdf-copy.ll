; RUN: llc -march=hexagon -disable-copyprop < %s | FileCheck %s
; Disable MachineCopyPropagation to expose this opportunity to RDF copy.

;
; Check that
;     {
;         r1 = r0
;     }
;     {
;         r0 = memw(r1 + #0)
;     }
; was copy-propagated to
;     {
;         r1 = r0
;         r0 = memw(r0 + #0)
;     }
;
; CHECK-LABEL: LBB0_1
; CHECK: [[DST:r[0-9]+]] = [[SRC:r[0-9]+]]
; CHECK-DAG: memw([[SRC]]
; CHECK-NOT: memw([[DST]]
; CHECK: %if.end

target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a0:0-n16:32"
target triple = "hexagon"

%union.t = type { %struct.t, [64 x i8] }
%struct.t = type { [12 x i8], ptr, double }
%struct.r = type opaque

define ptr @foo(ptr %chain) nounwind readonly #0 {
entry:
  %tobool = icmp eq ptr %chain, null
  br i1 %tobool, label %if.end, label %while.cond.preheader

while.cond.preheader:                             ; preds = %entry
  br label %while.cond

while.cond:                                       ; preds = %while.cond.preheader, %while.cond
  %chain.addr.0 = phi ptr [ %0, %while.cond ], [ %chain, %while.cond.preheader ]
  %0 = load ptr, ptr %chain.addr.0, align 4, !tbaa !0
  %tobool2 = icmp eq ptr %0, null
  br i1 %tobool2, label %if.end.loopexit, label %while.cond

if.end.loopexit:                                  ; preds = %while.cond
  br label %if.end

if.end:                                           ; preds = %if.end.loopexit, %entry
  %chain.addr.1 = phi ptr [ null, %entry ], [ %chain.addr.0, %if.end.loopexit ]
  ret ptr %chain.addr.1
}

attributes #0 = { nounwind "target-features"="-packets" }

!0 = !{!"any pointer", !1}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
