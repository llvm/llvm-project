; RUN: opt -verify -disable-output <%s 2>&1 | FileCheck %s
;
; This test creates an invalid vector by defining multiple elements for the
; vector's DICompositeType definition.  A vector should only have one element
; in its DICompositeType 'elements' array.
;
; CHECK: invalid vector

@f.foo = private unnamed_addr constant <6 x float> zeroinitializer, align 32

define void @f() {
  %1 = alloca <6 x float>, align 32
  call void @llvm.dbg.declare(metadata <6 x float>* %1, metadata !10, metadata !DIExpression()), !dbg !18
  ret void
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "/dbg/info")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!7 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 3, type: !8, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DILocalVariable(name: "foo", scope: !7, file: !1, line: 4, type: !12)
!12 = !DICompositeType(tag: DW_TAG_array_type, baseType: !13, size: 256, flags: DIFlagVector, elements: !14)
!13 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!14 = !{!15, !19}
!15 = !DISubrange(count: 6)
!18 = !DILocation(line: 4, column: 48, scope: !7)
!19 = !DISubrange(count: 42)
