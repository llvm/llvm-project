; RUN: opt -S -passes=jump-threading,verify %s | FileCheck %s

;; Modified from llvm/test/Transforms/JumpThreading/thread-two-bbs.ll
;;
;; JumpThreading duplicates bb.cond2 to thread through bb.file to bb.file,
;; bb.f2 or exit.
;;
;; Check the duplicated instructions get remapped atom groups.

; CHECK: bb.cond2:
; CHECK-NEXT: call void @f1()
; CHECK-NEXT: %tobool1 = icmp eq i32 %cond2, 0, !dbg [[G1R2:!.*]]
; CHECK-NEXT: br i1 %tobool1, label %bb.file, label %exit, !dbg [[G1R1:!.*]]

; CHECK: bb.cond2.thread:
; CHECK-NEXT: %tobool12 = icmp eq i32 %cond2, 0, !dbg [[G3R2:!.*]]
; CHECK-NEXT: br i1 %tobool12, label %bb.f2, label %exit, !dbg [[G3R1:!.*]]

;; After the transform %ptr is null through bb.cond2 and @a through
;; bb.cond2.thread. Thread bb.cond2.thread->bb.f2 through bb.file.
;; Check the duplicated store gets a remapped atom group too.

; CHECK: bb.file:
; CHECK-NEXT: %ptr3 = phi ptr [ null, %bb.cond2 ]
; CHECK-NEXT: store ptr %ptr3, ptr %p, align 4, !dbg [[G2R1:!.*]]

; CHECK: bb.f2:
; CHECK-NEXT: store ptr @a, ptr %p, align 4, !dbg [[G4R1:!.*]]

; CHECK: distinct !DISubprogram(name: "foo", {{.*}}keyInstructions: true)
; CHECK: [[G1R2]] = !DILocation(line: 1, column: 1, scope: ![[#]], atomGroup: 1, atomRank: 2)
; CHECK: [[G1R1]] = !DILocation(line: 1, column: 1, scope: ![[#]], atomGroup: 1, atomRank: 1)
; CHECK: [[G3R2]] = !DILocation(line: 1, column: 1, scope: ![[#]], atomGroup: 3, atomRank: 2)
; CHECK: [[G3R1]] = !DILocation(line: 1, column: 1, scope: ![[#]], atomGroup: 3, atomRank: 1)
; CHECK: [[G2R1]] = !DILocation(line: 2, column: 1, scope: ![[#]], atomGroup: 2, atomRank: 1)
; CHECK: [[G4R1]] = !DILocation(line: 2, column: 1, scope: ![[#]], atomGroup: 4, atomRank: 1)

@a = global i32 0, align 4

define void @foo(i32 %cond1, i32 %cond2, ptr %p) !dbg !5 {
entry:
  %tobool = icmp eq i32 %cond1, 0
  br i1 %tobool, label %bb.cond2, label %bb.f1

bb.f1:                                            ; preds = %entry
  call void @f1()
  br label %bb.cond2

bb.cond2:                                         ; preds = %bb.f1, %entry
  %ptr = phi ptr [ null, %bb.f1 ], [ @a, %entry ]
  %tobool1 = icmp eq i32 %cond2, 0, !dbg !9
  br i1 %tobool1, label %bb.file, label %exit, !dbg !10

bb.file:                                          ; preds = %bb.cond2
  store ptr %ptr, ptr %p, align 4, !dbg !11
  %cmp = icmp eq ptr %ptr, null
  br i1 %cmp, label %exit, label %bb.f2

bb.f2:                                            ; preds = %bb.file
  call void @f2()
  br label %exit

exit:                                             ; preds = %bb.f2, %bb.file, %bb.cond2
  ret void
}

declare void @f1()

declare void @f2()

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "<stdin>", directory: "/")
!2 = !{i32 16}
!3 = !{i32 0}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, keyInstructions: true)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!9 = !DILocation(line: 1, column: 1, scope: !5, atomGroup: 1, atomRank: 2)
!10 = !DILocation(line: 1, column: 1, scope: !5, atomGroup: 1, atomRank: 1)
!11 = !DILocation(line: 2, column: 1, scope: !5, atomGroup: 2, atomRank: 1)
