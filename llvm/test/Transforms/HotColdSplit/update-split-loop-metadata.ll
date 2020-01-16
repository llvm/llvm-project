; RUN: opt -hotcoldsplit -hotcoldsplit-threshold=0 -S < %s | FileCheck %s

; Check that llvm.loop metadata extracted by CodeExtractor is updated so that
; the debug locations it contains have the right scope.

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

; CHECK-LABEL: define {{.*}}@basic.cold.1
; CHECK: br i1 {{.*}}, !llvm.loop [[LOOP_MD:![0-9]+]]

; The scope for these debug locations should be @basic.cold.1, not @basic.
; CHECK: [[SCOPE:![0-9]+]] = distinct !DISubprogram(name: "basic.cold.1"
; CHECK: [[LOOP_MD]] = distinct !{[[LOOP_MD]], [[LINE:![0-9]+]], [[LINE]]}
; CHECK: [[LINE]] = !DILocation(line: 1, column: 1, scope: [[SCOPE]])

define void @basic(i32* %p, i32 %k) !dbg !6 {
entry:
  %cmp3 = icmp slt i32 0, %k
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.05 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %p.addr.04 = phi i32* [ %p, %for.body.lr.ph ], [ %incdec.ptr, %for.body ]
  %incdec.ptr = getelementptr inbounds i32, i32* %p.addr.04, i32 1
  store i32 %i.05, i32* %p.addr.04, align 4
  %inc = add nsw i32 %i.05, 1
  call void @sink()
  %cmp = icmp slt i32 %inc, %k
  br i1 %cmp, label %for.body, label %for.cond.for.end_crit_edge, !llvm.loop !10

for.cond.for.end_crit_edge:                       ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry
  ret void
}

declare void @sink() cold

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!3, !4}
!llvm.module.flags = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "<stdin>", directory: "/")
!2 = !{}
!3 = !{i32 22}
!4 = !{i32 12}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "basic", linkageName: "basic", scope: null, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!7 = !DISubroutineType(types: !2)
!8 = !{}
!9 = !DILocation(line: 1, column: 1, scope: !6)
!10 = distinct !{!10, !9, !9}
