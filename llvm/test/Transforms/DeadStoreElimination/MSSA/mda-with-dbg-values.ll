; RUN: opt -S -dse -dse-memoryssa-scanlimit=2 < %s | FileCheck %s
; RUN: opt -S -strip-debug -dse -dse-memoryssa-scanlimit=2 < %s | FileCheck %s

; Test case to check that DSE gets the same result even if we have a dbg value
; between the memcpy.

; This test case is less relevant for the MemorySSA backed version of DSE, as
; debug values are not modeled in MemorySSA and are skipped regardless of the
; exploration limit.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@g = common global [1 x i8] zeroinitializer, align 1, !dbg !0

; Function Attrs: noinline nounwind uwtable
define void @foo() #0 !dbg !14 {
entry:
  %i = alloca i8, align 1
  store i8 1, i8* %i, align 1, !dbg !19
  call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !17, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !17, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !17, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !17, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !17, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !17, metadata !DIExpression()), !dbg !18
  %0 = bitcast [1 x i8]* @g to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %i, i8* %0, i64 1, i1 false), !dbg !20
  br label %bb2

bb2:                                              ; preds = %0
  ret void, !dbg !21
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #2

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { argmemonly nounwind }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10, !11, !12}
!llvm.ident = !{!13}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g", scope: !2, file: !3, line: 3, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 6.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "foo.c", directory: "/bar")
!4 = !{}
!5 = !{!0}
!6 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 8, elements: !8)
!7 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!8 = !{!9}
!9 = !DISubrange(count: 1)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{!"clang version 6.0.0"}
!14 = distinct !DISubprogram(name: "foo", scope: !3, file: !3, line: 5, type: !15, isLocal: false, isDefinition: true, scopeLine: 6, isOptimized: false, unit: !2, retainedNodes: !4)
!15 = !DISubroutineType(types: !16)
!16 = !{null}
!17 = !DILocalVariable(name: "i", scope: !14, file: !3, line: 7, type: !7)
!18 = !DILocation(line: 7, column: 10, scope: !14)
!19 = !DILocation(line: 8, column: 7, scope: !14)
!20 = !DILocation(line: 9, column: 5, scope: !14)
!21 = !DILocation(line: 10, column: 1, scope: !14)

; Check that the both the store and memcpy are removed because they both access
; an alloca that is not read.
; CHECK-LABEL: foo
; CHECK-NOT:   store i8
; CHECK-NOT:   call void @llvm.memcpy
; CHECK:       ret void
