;; This test verifies 'auto' hotness threshold when profile file is provided.
;;
;; new PM
; RUN: rm -f %t.yaml %t.hot.yaml
; RUN: opt %s --passes='sample-profile,cgscc(inline)' \
; RUN: --sample-profile-file=%S/Inputs/remarks-hotness.prof \
; RUN: -S --pass-remarks-filter=inline --pass-remarks-output=%t.yaml \
; RUN: -pass-remarks-with-hotness --disable-output
; RUN: FileCheck %s -check-prefix=YAML-PASS < %t.yaml
; RUN: FileCheck %s -check-prefix=YAML-MISS < %t.yaml

;; test 'auto' threshold
; RUN: opt %s --passes='sample-profile,cgscc(inline)' \
; RUN: --sample-profile-file=%S/Inputs/remarks-hotness.prof \
; RUN: -S --pass-remarks-filter=inline --pass-remarks-output=%t.hot.yaml \
; RUN: --pass-remarks-with-hotness --pass-remarks-hotness-threshold=auto --disable-output
; RUN: FileCheck %s -check-prefix=YAML-PASS < %t.hot.yaml
; RUN: not FileCheck %s -check-prefix=YAML-MISS < %t.hot.yaml

; RUN: opt %s --passes='sample-profile,cgscc(inline)' \
; RUN: --sample-profile-file=%S/Inputs/remarks-hotness.prof \
; RUN: -S --pass-remarks=inline --pass-remarks-missed=inline --pass-remarks-analysis=inline \
; RUN: --pass-remarks-with-hotness --pass-remarks-hotness-threshold=auto --disable-output 2>&1 | FileCheck %s -check-prefix=CHECK-RPASS

; YAML-PASS:      --- !Passed
; YAML-PASS-NEXT: Pass:            inline
; YAML-PASS-NEXT: Name:            Inlined
; YAML-PASS-NEXT: DebugLoc:        { File: remarks-hotness.cpp, Line: 10, Column: 10 }
; YAML-PASS-NEXT: Function:        _Z7caller1v
; YAML-PASS-NEXT: Hotness:         401

; YAML-MISS:      --- !Missed
; YAML-MISS-NEXT: Pass:            inline
; YAML-MISS-NEXT: Name:            NeverInline
; YAML-MISS-NEXT: DebugLoc:        { File: remarks-hotness.cpp, Line: 14, Column: 10 }
; YAML-MISS-NEXT: Function:        _Z7caller2v
; YAML-MISS-NEXT: Hotness:         2

; CHECK-RPASS: '_Z7callee1v' inlined into '_Z7caller1v' with (cost=-30, threshold=4500) at callsite _Z7caller1v:1:10; (hotness: 401)
; CHECK-RPASS-NOT: '_Z7callee2v' not inlined into '_Z7caller2v' because it should never be inlined (cost=never): noinline function attribute (hotness: 2)

; ModuleID = 'remarks-hotness.cpp'
source_filename = "remarks-hotness.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: use-sample-profile
define dso_local i32 @_Z7callee1v() #0 !dbg !7 {
  ret i32 1, !dbg !11
}

; Function Attrs: noinline nounwind uwtable use-sample-profile
define dso_local i32 @_Z7callee2v() #1 !dbg !12 {
  ret i32 2, !dbg !13
}

; Function Attrs: use-sample-profile
define dso_local i32 @_Z7caller1v() #0 !dbg !14 {
  %1 = call i32 @_Z7callee1v(), !dbg !15
  ret i32 %1, !dbg !16
}

; Function Attrs: use-sample-profile
define dso_local i32 @_Z7caller2v() #0 !dbg !17 {
  %1 = call i32 @_Z7callee2v(), !dbg !18
  ret i32 %1, !dbg !19
}

attributes #0 = { "use-sample-profile" }
attributes #1 = { noinline nounwind uwtable "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "remarks-hotness.cpp", directory: ".")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 11.0.0"}
!7 = distinct !DISubprogram(name: "callee1", linkageName: "_Z7callee1v", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DILocation(line: 2, column: 3, scope: !7)
!12 = distinct !DISubprogram(name: "callee2", linkageName: "_Z7callee2v", scope: !1, file: !1, line: 5, type: !8, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!13 = !DILocation(line: 6, column: 3, scope: !12)
!14 = distinct !DISubprogram(name: "caller1", linkageName: "_Z7caller1v", scope: !1, file: !1, line: 9, type: !8, scopeLine: 9, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!15 = !DILocation(line: 10, column: 10, scope: !14)
!16 = !DILocation(line: 10, column: 3, scope: !14)
!17 = distinct !DISubprogram(name: "caller2", linkageName: "_Z7caller2v", scope: !1, file: !1, line: 13, type: !8, scopeLine: 13, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!18 = !DILocation(line: 14, column: 10, scope: !17)
!19 = !DILocation(line: 14, column: 3, scope: !17)
