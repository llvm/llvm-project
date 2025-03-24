; RUN: opt %s -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S \
; RUN: | FileCheck %s

;; Generated using:
;;   opt -passes=debugify --debugify-atoms --debugify-level=locations \
;;      llvm/test/Transforms/SimplifyCFG/debug-info-thread-phi.ll
;; With unused/untested metadata nodes removed.

;; Check the duplicated store gets distinct atom info in each branch.

; CHECK-LABEL: @bar(
; CHECK: if.then:
; CHECK:   store i32 1{{.*}}, !dbg [[DBG1:!.*]]
; CHECK: if.end.1.critedge:
; CHECK:   store i32 1{{.*}}, !dbg [[DBG2:!.*]]
; CHECK: [[DBG1]] = !DILocation(line: 1{{.*}}, atomGroup: 1
; CHECK: [[DBG2]] = !DILocation(line: 1{{.*}}, atomGroup: 2

define void @bar(i32 %aa) !dbg !5 {
entry:
  %aa.addr = alloca i32, align 4
  %bb = alloca i32, align 4
  store i32 %aa, ptr %aa.addr, align 4
  store i32 0, ptr %bb, align 4
  %tobool = icmp ne i32 %aa, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  store i32 1, ptr %bb, align 4, !dbg !8
  br i1 %tobool, label %if.then.1, label %if.end.1

if.then.1:                                        ; preds = %if.end
  call void @foo()
  br label %if.end.1

if.end.1:                                         ; preds = %if.then.1, %if.end
  store i32 2, ptr %bb, align 4
  br label %for.end

for.end:                                          ; preds = %if.end.1
  ret void
}

declare void @foo()

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "llvm/test/Transforms/SimplifyCFG/debug-info-thread-phi.ll", directory: "/")
!2 = !{i32 15}
!3 = !{i32 0}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "bar", linkageName: "bar", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !DILocation(line: 1, column: 1, scope: !5, atomGroup: 1, atomRank: 1)
