; Test finding types by CompilerContext.
; REQUIRES: aarch64
; RUN: llc %s -filetype=obj -o %t.o
; RUN: lldb-test symbols %t.o -find=type --mangled-name=UniqueDifferentName | FileCheck %s 
;
; NORESULTS: Found 0 types
; CHECK: Found 1 types:
; CHECK: struct DifferentName {
; CHECK-NEXT:     int i;
; CHECK-NEXT: }

source_filename = "t.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-unknown-linux-gnu"

%struct.SameName = type { i32 }
%struct.DifferentName = type { i32 }

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !10 {
entry:
  %retval = alloca i32, align 4
  %s = alloca %struct.SameName, align 4
  %d = alloca %struct.DifferentName, align 4
  store i32 0, ptr %retval, align 4
    #dbg_declare(ptr %s, !16, !DIExpression(), !20)
    #dbg_declare(ptr %d, !21, !DIExpression(), !25)
  ret i32 0, !dbg !26
}

attributes #0 = { noinline  optnone  }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "t.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 1}
!9 = !{!""}
!10 = distinct !DISubprogram(name: "main", scope: !11, file: !11, line: 9, type: !12, scopeLine: 9, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !15)
!11 = !DIFile(filename: "t.c", directory: "")
!12 = !DISubroutineType(types: !13)
!13 = !{!14}
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !{}
!16 = !DILocalVariable(name: "s", scope: !10, file: !11, line: 10, type: !17)
!17 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "SameName", file: !11, line: 1, size: 32, elements: !18, runtimeLang: DW_LANG_Swift, identifier: "SameName")
!18 = !{!19}
!19 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !17, file: !11, line: 2, baseType: !14, size: 32)
!20 = !DILocation(line: 10, column: 19, scope: !10)
!21 = !DILocalVariable(name: "d", scope: !10, file: !11, line: 11, type: !22)
!22 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "DifferentName", file: !11, line: 5, size: 32, elements: !23, runtimeLang: DW_LANG_Swift, identifier: "UniqueDifferentName")
!23 = !{!24}
!24 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !22, file: !11, line: 6, baseType: !14, size: 32)
!25 = !DILocation(line: 11, column: 24, scope: !10)
!26 = !DILocation(line: 12, column: 3, scope: !10)
