; RUN: llc -mtriple=hexagon < %s
; RUN: llc -mtriple=hexagon -disable-hcp < %s
; REQUIRES: asserts

target datalayout = "e-m:e-p:32:32-i1:32-i64:64-a:0-v32:32-n16:32"
target triple = "hexagon-unknown--elf"

%struct.item = type { i32, ptr, ptr, i32, i8, i8, i16, i32, i8, i16, i32 }

declare ptr @foo(ptr, ptr, i32) #1

; Function Attrs: nounwind
define i32 @bar(ptr %ptr, ptr %buf, i32 %c, ptr %d, i32 %e) #1 {
entry:
  br i1 undef, label %return, label %if.end

if.end:                                           ; preds = %entry
  br i1 undef, label %while.cond13.preheader, label %if.end3

if.end3:                                          ; preds = %if.end
  br label %while.cond13.preheader

while.cond13.preheader:                           ; preds = %if.end3, %if.end
  br i1 undef, label %while.body20, label %return

while.body20:                                     ; preds = %if.end38, %while.cond13.preheader
  %addr.0100 = phi i32 [ undef, %if.end38 ], [ %c, %while.cond13.preheader ]
  %cond = select i1 undef, i32 %addr.0100, i32 undef
  br i1 undef, label %while.body20.if.end38_crit_edge, label %if.then32

while.body20.if.end38_crit_edge:                  ; preds = %while.body20
  %conv39.pre = and i32 %cond, 65535
  br label %if.end38

if.then32:                                        ; preds = %while.body20
  %conv33 = and i32 %cond, 65535
  %.pre = load ptr, ptr %ptr, align 4, !tbaa !1
  br label %if.end38

if.end38:                                         ; preds = %if.then32, %while.body20.if.end38_crit_edge
  %conv39.pre-phi = phi i32 [ %conv39.pre, %while.body20.if.end38_crit_edge ], [ %conv33, %if.then32 ]
  %0 = phi ptr [ undef, %while.body20.if.end38_crit_edge ], [ %.pre, %if.then32 ]
  %add = add i32 %conv39.pre-phi, 0
  %call52 = tail call ptr @foo(ptr %0, ptr %d, i32 %e) #1
  br i1 undef, label %while.body20, label %return

return:                                           ; preds = %if.end38, %while.cond13.preheader, %entry
  %retval.0 = phi i32 [ 0, %entry ], [ 0, %while.cond13.preheader ], [ %add, %if.end38 ]
  ret i32 %retval.0
}


attributes #0 = { nounwind readonly }
attributes #1 = { nounwind }

!1 = !{!2, !2, i64 0}
!2 = !{!"any pointer", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
