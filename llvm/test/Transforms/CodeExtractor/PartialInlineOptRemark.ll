; RUN: opt -S -partial-inliner -pass-remarks=partial-inlining  -disable-output < %s 2>&1 | FileCheck %s
; RUN: opt -S -passes=partial-inliner  -pass-remarks=partial-inlining -disable-output < %s 2>&1 | FileCheck %s
; RUN: opt -S -partial-inliner -pass-remarks=partial-inlining  -disable-output -max-partial-inlining=1 < %s 2>&1 | FileCheck %s
; RUN: opt -S -passes=partial-inliner  -pass-remarks=partial-inlining -disable-output -max-partial-inlining=1 < %s 2>&1 | FileCheck %s

; RUN: opt -S -partial-inliner -pass-remarks=partial-inlining  -disable-partial-inlining < %s 2>&1 | FileCheck --check-prefix=LIMIT %s
; RUN: opt -S -passes=partial-inliner  -pass-remarks=partial-inlining  --disable-partial-inlining < %s 2>&1 | FileCheck  --check-prefix=LIMIT %s
; RUN: opt -S -partial-inliner -pass-remarks=partial-inlining   -max-partial-inlining=0 < %s 2>&1 | FileCheck --check-prefix=LIMIT  %s
; RUN: opt -S -passes=partial-inliner  -pass-remarks=partial-inlining  -max-partial-inlining=0 < %s 2>&1 | FileCheck --check-prefix=LIMIT  %s

define i32 @bar(i32 %arg) local_unnamed_addr #0 !dbg !5 {
bb:
  %tmp = icmp slt i32 %arg, 0, !dbg !7
  br i1 %tmp, label %bb1, label %bb2, !dbg !8

bb1:                                              ; preds = %bb
  tail call void (...) @foo() #0, !dbg !9
  tail call void (...) @foo() #0, !dbg !10
  tail call void (...) @foo() #0, !dbg !11
  tail call void (...) @foo() #0, !dbg !12
  tail call void (...) @foo() #0, !dbg !13
  tail call void (...) @foo() #0, !dbg !14
  tail call void (...) @foo() #0, !dbg !15
  tail call void (...) @foo() #0, !dbg !16
  tail call void (...) @foo() #0, !dbg !17
  br label %bb2, !dbg !18

bb2:                                              ; preds = %bb1, %bb
  %tmp3 = phi i32 [ 0, %bb1 ], [ 1, %bb ]
  ret i32 %tmp3, !dbg !19
}

; Function Attrs: nounwind
declare void @foo(...) local_unnamed_addr #0

; Function Attrs: nounwind
define i32 @dummy_caller(i32 %arg) local_unnamed_addr #0 !dbg !20 {
bb:
; CHECK:remark{{.*}}bar partially inlined into dummy_caller
; LIMIT-NOT:remark{{.*}}bar partially inlined into dummy_caller
  %tmp = tail call i32 @bar(i32 %arg), !dbg !21
  ret i32 %tmp, !dbg !22
}

attributes #0 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}
!llvm.ident = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, enums: !2)
!1 = !DIFile(filename: "t.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{!"clang "}
!5 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 3, type: !6, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !2)
!6 = !DISubroutineType(types: !2)
!7 = !DILocation(line: 4, column: 14, scope: !5)
!8 = !DILocation(line: 4, column: 6, scope: !5)
!9 = !DILocation(line: 5, column: 5, scope: !5)
!10 = !DILocation(line: 6, column: 5, scope: !5)
!11 = !DILocation(line: 7, column: 5, scope: !5)
!12 = !DILocation(line: 8, column: 5, scope: !5)
!13 = !DILocation(line: 9, column: 5, scope: !5)
!14 = !DILocation(line: 10, column: 5, scope: !5)
!15 = !DILocation(line: 11, column: 5, scope: !5)
!16 = !DILocation(line: 12, column: 5, scope: !5)
!17 = !DILocation(line: 13, column: 5, scope: !5)
!18 = !DILocation(line: 14, column: 5, scope: !5)
!19 = !DILocation(line: 17, column: 1, scope: !5)
!20 = distinct !DISubprogram(name: "dummy_caller", scope: !1, file: !1, line: 19, type: !6, isLocal: false, isDefinition: true, scopeLine: 19, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !2)
!21 = !DILocation(line: 21, column: 11, scope: !20)
!22 = !DILocation(line: 21, column: 4, scope: !20)
