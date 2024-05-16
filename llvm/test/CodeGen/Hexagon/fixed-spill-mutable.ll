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

%struct.0 = type { ptr, ptr, ptr, ptr, ptr }
%struct.1 = type { [60 x i8], i32, ptr }
%struct.2 = type { i8, i8, i8, i8, %union.anon }
%union.anon = type { ptr }
%struct.3 = type { ptr, ptr }

@var = external hidden unnamed_addr global ptr, align 4

declare void @bar(ptr, i32) local_unnamed_addr #0

define void @foo() local_unnamed_addr #1 {
entry:
  %.pr = load ptr, ptr @var, align 4, !tbaa !1
  %cmp2 = icmp eq ptr %.pr, null
  br i1 %cmp2, label %while.end, label %while.body.preheader

while.body.preheader:                             ; preds = %entry
  br label %while.body

while.body:                                       ; preds = %while.body.preheader, %while.body
  %0 = phi ptr [ %4, %while.body ], [ %.pr, %while.body.preheader ]
  %right = getelementptr inbounds %struct.0, ptr %0, i32 0, i32 4
  %1 = bitcast ptr %right to ptr
  %2 = load i32, ptr %1, align 4, !tbaa !5
  %3 = bitcast ptr %0 to ptr
  tail call void @bar(ptr %3, i32 20) #1
  store i32 %2, ptr @var, align 4, !tbaa !1
  %4 = inttoptr i32 %2 to ptr
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
