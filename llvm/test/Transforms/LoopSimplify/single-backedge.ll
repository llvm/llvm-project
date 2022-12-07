; The loop canonicalization pass should guarantee that there is one backedge
; for all loops.  This allows the -passes=indvars pass to recognize the %IV
; induction variable in this testcase.

; RUN: opt < %s -passes=indvars -S | FileCheck %s
; CHECK: Loop.backedge:
; CHECK-NOT: br
; CHECK: br label %Loop, !dbg [[BACKEDGE_LOC:![0-9]+]]

; CHECK: [[BACKEDGE_LOC]] = !DILocation(line: 101, column: 1, scope: !{{.*}})

define i32 @test(i1 %C) {
; <label>:0
  br label %Loop, !dbg !6
Loop: ; preds = %BE2, %BE1, %0
  %IV = phi i32 [ 1, %0 ], [ %IV2, %BE1 ], [ %IV2, %BE2 ] ; <i32> [#uses=2]
  store i32 %IV, i32* null, !dbg !7
  %IV2 = add i32 %IV, 2, !dbg !8 ; <i32> [#uses=2]
  br i1 %C, label %BE1, label %BE2, !dbg !9
BE1:  ; preds = %Loop
  br label %Loop, !dbg !10
BE2:    ; preds = %n br label %Loop
  br label %Loop, !dbg !11
}

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!12}
!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}

!2 = !{}
!3 = !DISubroutineType(types: !2)
!4 = !DIFile(filename: "atomic.cpp", directory: "/tmp")
!5 = distinct !DISubprogram(name: "test", scope: !4, file: !4, line: 99, type: !3, isLocal: false, isDefinition: true, scopeLine: 100, flags: DIFlagPrototyped, isOptimized: false, unit: !12, retainedNodes: !2)
!6 = !DILocation(line: 100, column: 1, scope: !5)
!7 = !DILocation(line: 101, column: 1, scope: !5)
!8 = !DILocation(line: 102, column: 1, scope: !5)
!9 = !DILocation(line: 103, column: 1, scope: !5)
!10 = !DILocation(line: 104, column: 1, scope: !5)
!11 = !DILocation(line: 105, column: 1, scope: !5)
!12 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang",
                             file: !4,
                             isOptimized: true, flags: "-O2",
                             splitDebugFilename: "abc.debug", emissionKind: 2)
