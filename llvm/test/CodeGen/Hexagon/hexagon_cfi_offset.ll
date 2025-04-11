; RUN: llc -mtriple=hexagon -O2 < %s | FileCheck %s
; Check the values of cfi offsets emitted.
; CHECK: .cfi_def_cfa r30, 8
; CHECK: .cfi_offset r31, -4
; CHECK: .cfi_offset r30, -8
; CHECK: .cfi_offset r17, -12
; CHECK: .cfi_offset r16, -16

%s.0 = type { ptr, ptr }
%s.1 = type { ptr, ptr, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, %s.2 }
%s.2 = type { i64, ptr, i32, i32, [12 x i8] }
%s.3 = type { ptr, i32, ptr, i32 }

; Function Attrs: noreturn
define void @f0(ptr %a0, ptr %a1, ptr %a2) #0 {
b0:
  %v0 = getelementptr inbounds i8, ptr %a0, i32 -80
  store ptr %a1, ptr %v0, align 16, !tbaa !0
  %v2 = getelementptr inbounds i8, ptr %a0, i32 -76
  store ptr %a2, ptr %v2, align 4, !tbaa !9
  %v4 = tail call ptr @f1(ptr null) #3
  %v5 = getelementptr inbounds i8, ptr %a0, i32 -72
  store ptr %v4, ptr %v5, align 8, !tbaa !10
  %v7 = tail call ptr @f1(ptr %v4) #3
  %v8 = tail call ptr @f2(ptr null) #3
  %v9 = getelementptr inbounds i8, ptr %a0, i32 -68
  store ptr %v8, ptr %v9, align 4, !tbaa !11
  %v11 = tail call ptr @f2(ptr %v8) #3
  %v12 = getelementptr inbounds i8, ptr %a0, i32 -64
  store ptr null, ptr %v12, align 16, !tbaa !12
  %v14 = getelementptr inbounds i8, ptr %a0, i32 -60
  store i32 0, ptr %v14, align 4, !tbaa !13
  %v16 = getelementptr inbounds i8, ptr %a0, i32 -32
  store i64 4921953907261516544, ptr %v16, align 16, !tbaa !14
  %v19 = getelementptr inbounds i8, ptr %a0, i32 -24
  store ptr @f3, ptr %v19, align 8, !tbaa !15
  %v21 = tail call ptr @f4() #3
  %v22 = getelementptr inbounds %s.3, ptr %v21, i32 0, i32 1
  %v23 = load i32, ptr %v22, align 4, !tbaa !16
  %v24 = add i32 %v23, 1
  store i32 %v24, ptr %v22, align 4, !tbaa !16
  %v25 = tail call zeroext i8 @f5(ptr %v16) #4
  %v26 = tail call ptr @f6(ptr %v16) #3
  tail call void @f7() #5
  unreachable
}

; Function Attrs: nounwind
declare ptr @f1(ptr) #1

; Function Attrs: nounwind
declare ptr @f2(ptr) #1

define internal void @f3(i8 zeroext %a0, ptr %a1) #2 {
b0:
  %v1 = load i64, ptr %a1, align 16, !tbaa !18
  %v2 = icmp eq i64 %v1, 4921953907261516544
  br i1 %v2, label %b1, label %b4

b1:                                               ; preds = %b0
  %v3 = getelementptr inbounds %s.2, ptr %a1, i32 1
  %v5 = getelementptr inbounds %s.2, ptr %a1, i32 -2, i32 3
  %v6 = getelementptr inbounds i32, ptr %v5, i32 1
  %v8 = load ptr, ptr %v6, align 4, !tbaa !9
  %v9 = icmp eq ptr %v8, null
  br i1 %v9, label %b3, label %b2

b2:                                               ; preds = %b1
  tail call void %v8(ptr %v3) #4
  br label %b3

b3:                                               ; preds = %b2, %b1
  tail call void @f8(ptr %v3) #3
  br label %b4

b4:                                               ; preds = %b3, %b0
  ret void
}

; Function Attrs: nounwind
declare ptr @f4() #1

declare zeroext i8 @f5(ptr) #2

; Function Attrs: nounwind
declare ptr @f6(ptr) #1

; Function Attrs: noreturn
declare void @f7() #0

; Function Attrs: nounwind
declare void @f8(ptr) #1

attributes #0 = { noreturn "target-cpu"="hexagonv60" }
attributes #1 = { nounwind "target-cpu"="hexagonv60" }
attributes #2 = { "target-cpu"="hexagonv60" }
attributes #3 = { nobuiltin nounwind }
attributes #4 = { nobuiltin }
attributes #5 = { nobuiltin noreturn }

!0 = !{!1, !2, i64 0}
!1 = !{!"_ZTS15__cxa_exception", !2, i64 0, !2, i64 4, !2, i64 8, !2, i64 12, !2, i64 16, !5, i64 20, !5, i64 24, !2, i64 28, !2, i64 32, !2, i64 36, !2, i64 40, !6, i64 48}
!2 = !{!"any pointer", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!"int", !3, i64 0}
!6 = !{!"_ZTS17_Unwind_Exception", !7, i64 0, !2, i64 8, !8, i64 12, !8, i64 16}
!7 = !{!"long long", !3, i64 0}
!8 = !{!"long", !3, i64 0}
!9 = !{!1, !2, i64 4}
!10 = !{!1, !2, i64 8}
!11 = !{!1, !2, i64 12}
!12 = !{!1, !2, i64 16}
!13 = !{!1, !5, i64 20}
!14 = !{!1, !7, i64 48}
!15 = !{!1, !2, i64 56}
!16 = !{!17, !5, i64 4}
!17 = !{!"_ZTS16__cxa_eh_globals", !2, i64 0, !5, i64 4, !2, i64 8, !5, i64 12}
!18 = !{!6, !7, i64 0}
