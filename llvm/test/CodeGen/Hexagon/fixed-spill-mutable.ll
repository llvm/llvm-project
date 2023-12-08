; RUN: llc -march=hexagon < %s | FileCheck %s

; The early return is predicated, and the save-restore code is mixed together:
; {
;   p0 = cmp.eq(r0, #0)
;   if (p0.new) r17:16 = memd(r29 + #0)
;   memd(r29+#0) = r17:16
; }
; {
;   if (p0) dealloc_return
; }
; The problem is that the load will execute before the store, clobbering the
; pair r17:16.
;
; Check that the store and the load are not in the same packet.
; CHECK: memd{{.*}} = r17:16
; CHECK: }
; CHECK: r17:16 = memd
; CHECK-LABEL: LBB0_1:

target triple = "hexagon"

%struct.0 = type { i8*, %struct.1*, %struct.2*, %struct.0*, %struct.0* }
%struct.1 = type { [60 x i8], i32, %struct.1* }
%struct.2 = type { i8, i8, i8, i8, %union.anon }
%union.anon = type { %struct.3* }
%struct.3 = type { %struct.3*, %struct.2* }

@var = external hidden unnamed_addr global %struct.0*, align 4

declare void @bar(i8*, i32) local_unnamed_addr #0

define void @foo() local_unnamed_addr #1 {
entry:
  %.pr = load %struct.0*, %struct.0** @var, align 4, !tbaa !1
  %cmp2 = icmp eq %struct.0* %.pr, null
  br i1 %cmp2, label %while.end, label %while.body.preheader

while.body.preheader:                             ; preds = %entry
  br label %while.body

while.body:                                       ; preds = %while.body.preheader, %while.body
  %0 = phi %struct.0* [ %4, %while.body ], [ %.pr, %while.body.preheader ]
  %right = getelementptr inbounds %struct.0, %struct.0* %0, i32 0, i32 4
  %1 = bitcast %struct.0** %right to i32*
  %2 = load i32, i32* %1, align 4, !tbaa !5
  %3 = bitcast %struct.0* %0 to i8*
  tail call void @bar(i8* %3, i32 20) #1
  store i32 %2, i32* bitcast (%struct.0** @var to i32*), align 4, !tbaa !1
  %4 = inttoptr i32 %2 to %struct.0*
  %cmp = icmp eq i32 %2, 0
  br i1 %cmp, label %while.end.loopexit, label %while.body

while.end.loopexit:                               ; preds = %while.body
  br label %while.end

while.end:                                        ; preds = %while.end.loopexit, %entry
  ret void
}

attributes #0 = { optsize }
attributes #1 = { nounwind optsize }

!1 = !{!2, !2, i64 0}
!2 = !{!"any pointer", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!6, !2, i64 16}
!6 = !{!"0", !2, i64 0, !2, i64 4, !2, i64 8, !2, i64 12, !2, i64 16}
