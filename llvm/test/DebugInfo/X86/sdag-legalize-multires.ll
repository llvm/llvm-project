; RUN: llc -O0 %s -stop-after=livedebugvars -o - | FileCheck %s
; This is a hand-crafted example modified after some Swift compiler output.
; Test that SelectionDAG legalization of DAG nodes with two results
; transfers debug info correctly.

source_filename = "/tmp/sincos.ll"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.13.0"

declare float @llvm.cos.f32(float)
declare float @llvm.sin.f32(float)
declare void @llvm.dbg.value(metadata, metadata, metadata)
declare swiftcc void @g(float, float)

; CHECK: ![[RCOS:.*]] = !DILocalVariable(name: "rcos"
; CHECK: ![[RSIN:.*]] = !DILocalVariable(name: "rsin"

define hidden swiftcc void @f() #0 !dbg !8 {
entry:
  ; CHECK: CALL64pcrel32 &__sincosf_stret
  %0 = call float @llvm.cos.f32(float 1.500000e+00), !dbg !13
  ; CHECK: $xmm1 = MOVAPSrr $xmm0
  call void @llvm.dbg.value(metadata float %0, metadata !15, metadata !DIExpression()), !dbg !13
  ; CHECK: DBG_VALUE {{.*}}$xmm1, {{.*}}, ![[RSIN]], !DIExpression(),
  %1 = call float @llvm.sin.f32(float 1.500000e+00), !dbg !13
  call void @llvm.dbg.value(metadata float %1, metadata !11, metadata !DIExpression()), !dbg !13
  ; CHECK: DBG_VALUE {{.*}}$xmm0, {{.*}}, ![[RCOS]], !DIExpression(),
  call void @g(float %0, float %1), !dbg !13
  ret void, !dbg !13
}

attributes #0 = { noinline nounwind optnone ssp uwtable "target-features"="+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "sincos.ll", directory: "/")
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!8 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 2, type: !9, isLocal: false, isDefinition: true, scopeLine: 2, isOptimized: false, unit: !0)
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = !DILocalVariable(name: "rsin", scope: !8, file: !1, line: 3, type: !12)
!12 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!13 = !DILocation(line: 3, scope: !8)
!15 = !DILocalVariable(name: "rcos", scope: !8, file: !1, line: 4, type: !12)
