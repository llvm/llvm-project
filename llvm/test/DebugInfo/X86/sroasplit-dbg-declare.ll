; RUN: opt -S -passes='sroa' -o - %s | FileCheck %s
; RUN: opt --try-experimental-debuginfo-iterators -S -passes='sroa' -o - %s | FileCheck %s

; SROA should split the alloca in two new ones, each with its own dbg.declare.
; The original alloca and dbg.declare should be removed.

define void @f1() {
entry:
  %0 = alloca [9 x i32]
  call void @llvm.dbg.declare(metadata ptr %0, metadata !11, metadata !DIExpression()), !dbg !17
  call void @llvm.memset.p0.i64(ptr align 16 %0, i8 0, i64 36, i1 true)
  store volatile i32 1, ptr %0
  ret void
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1) #0

attributes #0 = { argmemonly nounwind }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "foo.c", directory: "/bar")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 6.0.0"}
!7 = distinct !DISubprogram(name: "f1", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !10)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !{!11}
!11 = !DILocalVariable(name: "b", scope: !7, file: !1, line: 3, type: !12)
!12 = !DICompositeType(tag: DW_TAG_array_type, baseType: !13, size: 288, elements: !15)
!13 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !14)
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !{!16}
!16 = !DISubrange(count: 9)
!17 = !DILocation(line: 3, column: 18, scope: !7)

; CHECK-NOT:  = alloca [9 x i32]
; CHECK-NOT:  call void @llvm.dbg.declare(metadata ptr

; CHECK:      %[[VAR1:.*]] = alloca i32
; CHECK-NEXT: %[[VAR2:.*]] = alloca [8 x i32]
; CHECK-NEXT: call void @llvm.dbg.declare(metadata ptr %[[VAR1]]
; CHECK-NEXT: call void @llvm.dbg.declare(metadata ptr %[[VAR2]]

; CHECK-NOT:  = alloca [9 x i32]
; CHECK-NOT:  call void @llvm.dbg.declare(metadata ptr

