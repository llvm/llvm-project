; RUN: opt %s -passes='sroa,verify' -S -o - | FileCheck %s
; RUN: opt --try-experimental-debuginfo-iterators %s -passes='sroa,verify' -S -o - | FileCheck %s

; ModuleID = 'test.c'
; Test that SROA updates the debug info correctly if an alloca was rewritten but
; not partitioned into multiple allocas.
;
; CHECK: #dbg_value(float %s.coerce, ![[VAR:[0-9]+]], !DIExpression(),
; CHECK: ![[VAR]] = !DILocalVariable(name: "s",{{.*}} line: 3,

;
; struct S { float f; };
;
; float foo(struct S s) {
;   return s.f;
; }
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

%struct.S = type { float }

; Function Attrs: nounwind ssp uwtable
define float @foo(float %s.coerce) #0 !dbg !4 {
entry:
  %s = alloca %struct.S, align 4
  store float %s.coerce, ptr %s, align 1
  call void @llvm.dbg.declare(metadata ptr %s, metadata !16, metadata !17), !dbg !18
  %0 = load float, ptr %s, align 4, !dbg !19
  ret float %0, !dbg !19
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind ssp uwtable }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12, !13, !14}
!llvm.ident = !{!15}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.6.0 ", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "test.c", directory: "/Volumes/Data/llvm/_build.ninja.debug")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", line: 3, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 3, file: !1, scope: !5, type: !6, retainedNodes: !2)
!5 = !DIFile(filename: "test.c", directory: "/Volumes/Data/llvm/_build.ninja.debug")
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !9}
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "float", size: 32, align: 32, encoding: DW_ATE_float)
!9 = !DICompositeType(tag: DW_TAG_structure_type, name: "S", line: 1, size: 32, align: 32, file: !1, elements: !10)
!10 = !{!11}
!11 = !DIDerivedType(tag: DW_TAG_member, name: "f", line: 1, size: 32, align: 32, file: !1, scope: !9, baseType: !8)
!12 = !{i32 2, !"Dwarf Version", i32 2}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"PIC Level", i32 2}
!15 = !{!"clang version 3.6.0 "}
!16 = !DILocalVariable(name: "s", line: 3, arg: 1, scope: !4, file: !5, type: !9)
!17 = !DIExpression()
!18 = !DILocation(line: 3, column: 20, scope: !4)
!19 = !DILocation(line: 4, column: 2, scope: !4)
