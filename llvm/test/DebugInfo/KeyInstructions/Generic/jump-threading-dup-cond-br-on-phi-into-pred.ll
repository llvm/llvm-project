; RUN: opt %s --passes=jump-threading -S -o - -S | FileCheck %s

;;        +-> T1 -+
;;        |       v      +-> T2
;; Entry -+       Merge -+
;;        |       ^      +-> F2
;;        +-> F1 -+
;;
;; Duplicate Merge into T1 then fold Merge into its only pred F1 (taking its name).
;;
;;        +-> T1 -----> T2
;;        |     \       ^
;;        |      \     /
;;        |       \   /
;; Entry -+        +----+
;;        |         /   v 
;;        +--> Merge -> F2
;;
;; Check the duplicated (into T1) instructions' atoms are remapped.

; CHECK: T1:
; CHECK-NEXT: %v1 = call i32 @f1()
; CHECK-NEXT: %cond3 = icmp eq i32 %v1, 412
; CHECK-NEXT: %C1 = add i32 %v1, 1, !dbg [[G3R2:!.*]]
; CHECK-NEXT: store i32 %C1, ptr %p, align 4, !dbg [[G3R1:!.*]]

; CHECK: Merge:
; CHECK-NEXT: %v2 = call i32 @f2()
; CHECK-NEXT: store i32 1, ptr %p, align 4, !dbg [[G1R1:!.*]]
; CHECK-NEXT: %C = add i32 %v2, 1, !dbg [[G2R2:!.*]]
; CHECK-NEXT: store i32 %C, ptr %p, align 4, !dbg [[G2R1:!.*]]

; CHECK: [[G3R2]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 2)
; CHECK: [[G3R1]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 1)
; CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
; CHECK: [[G2R2]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 2)
; CHECK: [[G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)

define i32 @test5(i1 %cond, i1 %cond2, ptr %p) !dbg !5 {
  br i1 %cond, label %T1, label %F1

T1:                                               ; preds = %0
  %v1 = call i32 @f1()
  %cond3 = icmp eq i32 %v1, 412
  br label %Merge

F1:                                               ; preds = %0
  %v2 = call i32 @f2()
  store i32 1, ptr %p, align 4, !dbg !8
  br label %Merge

Merge:                                            ; preds = %F1, %T1
  %A = phi i1 [ %cond3, %T1 ], [ %cond2, %F1 ]
  %B = phi i32 [ %v1, %T1 ], [ %v2, %F1 ]
  %C = add i32 %B, 1, !dbg !9
  store i32 %C, ptr %p, align 4, !dbg !10
  br i1 %A, label %T2, label %F2

T2:                                               ; preds = %Merge
  call void @f3()
  ret i32 %B

F2:                                               ; preds = %Merge
  ret i32 %B
}

declare i32 @f1()

declare i32 @f2()

declare void @f3()

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.ll", directory: "/")
!2 = !{i32 12}
!3 = !{i32 0}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "test5", linkageName: "test5", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !DILocation(line: 1, column: 1, scope: !5, atomGroup: 1, atomRank: 1)
!9 = !DILocation(line: 2, column: 1, scope: !5, atomGroup: 2, atomRank: 2)
!10 = !DILocation(line: 2, column: 1, scope: !5, atomGroup: 2, atomRank: 1)
