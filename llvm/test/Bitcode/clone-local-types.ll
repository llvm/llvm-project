; RUN: llvm-as < %s -o %t0
; RUN: llvm-dis %t0 -o - | FileCheck %s

; Ensure that function-local types with the same ODR identifier belonging
; to different subprograms are not deduplicated when a module is being loaded.

; CHECK: [[CU:![0-9]+]] = distinct !DICompileUnit
; CHECK: [[BAR:![0-9]+]] = distinct !DISubprogram(name: "bar", {{.*}}, retainedNodes: [[RN_BAR:![0-9]+]])
; CHECK: [[RN_BAR]] = !{[[T2:![0-9]+]]}
; CHECK: [[T2]] = !DICompositeType(tag: DW_TAG_class_type, {{.*}}, identifier: "local_type")
; CHECK: [[FOO:![0-9]+]] = distinct !DISubprogram(name: "foo", {{.*}}, retainedNodes: [[RN_FOO:![0-9]+]])
; CHECK: [[RN_FOO]] = !{[[T1:![0-9]+]]}
; CHECK: [[T1]] = !DICompositeType(tag: DW_TAG_class_type, {{.*}}, identifier: "local_type")

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @foo()

define void @bar(ptr %this) !dbg !10 {
  call void @foo(), !dbg !17
  ret void, !dbg !13
}

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7}
!llvm.dbg.cu = !{!8}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 11, i32 4]}
!1 = !{i32 7, !"Dwarf Version", i32 2}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 7, !"openmp", i32 50}
!5 = !{i32 7, !"openmp-device", i32 50}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{i32 7, !"frame-pointer", i32 2}
!8 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !9, producer: "clang version 14.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!9 = !DIFile(filename: "tmp.cpp", directory: "/tmp/")
!10 = distinct !DISubprogram(name: "bar", scope: !9, file: !9, line: 68, type: !12, scopeLine: 68, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !8, retainedNodes: !16)
!11 = distinct !DISubprogram(name: "foo", scope: !9, file: !9, line: 68, type: !12, scopeLine: 68, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !8, retainedNodes: !16)
!12 = !DISubroutineType(types: !15)
!13 = !DILocation(line: 69, column: 18, scope: !10)
!14 = !DICompositeType(tag: DW_TAG_class_type, scope: !11, file: !9, line: 210, size: 8, flags: DIFlagTypePassByValue, elements: !15, identifier: "local_type")
!15 = !{}
!16 = !{!14}
!17 = !DILocation(line: 69, column: 18, scope: !11, inlinedAt: !18)
!18 = !DILocation(line: 4, column: 1, scope: !10)
