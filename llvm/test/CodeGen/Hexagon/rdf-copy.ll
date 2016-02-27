; RUN: llc -march=hexagon < %s | FileCheck %s
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
; CHECK-LABEL: LBB0_2

target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a0:0-n16:32"
target triple = "hexagon"

%union.t = type { %struct.t, [64 x i8] }
%struct.t = type { [12 x i8], %struct.r*, double }
%struct.r = type opaque

define %union.t* @foo(%union.t* %chain) nounwind readonly {
entry:
  %tobool = icmp eq %union.t* %chain, null
  br i1 %tobool, label %if.end, label %while.cond.preheader

while.cond.preheader:                             ; preds = %entry
  br label %while.cond

while.cond:                                       ; preds = %while.cond.preheader, %while.cond
  %chain.addr.0 = phi %union.t* [ %0, %while.cond ], [ %chain, %while.cond.preheader ]
  %chain1 = bitcast %union.t* %chain.addr.0 to %union.t**
  %0 = load %union.t*, %union.t** %chain1, align 4, !tbaa !0
  %tobool2 = icmp eq %union.t* %0, null
  br i1 %tobool2, label %if.end.loopexit, label %while.cond

if.end.loopexit:                                  ; preds = %while.cond
  br label %if.end

if.end:                                           ; preds = %if.end.loopexit, %entry
  %chain.addr.1 = phi %union.t* [ null, %entry ], [ %chain.addr.0, %if.end.loopexit ]
  ret %union.t* %chain.addr.1
}

!0 = !{!"any pointer", !1}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
