; RUN: llc -march=hexagon < %s | FileCheck %s
; Verify store/load for -g prologue

; CHECK: allocframe
; CHECK: memw([[MEM:.*]]) = r{{[0-9]+}}
; CHECK: r{{[0-9]+}} = memw([[MEM]])

; Function Attrs: nounwind
define i32 @f0(i32 %a0) #0 !dbg !5 {
b0:
  %v0 = alloca i32, align 4
  %v1 = alloca i32, align 4
  %v2 = alloca i32, align 4
  store i32 %a0, i32* %v1, align 4
  call void @llvm.dbg.declare(metadata i32* %v1, metadata !9, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.declare(metadata i32* %v2, metadata !11, metadata !DIExpression()), !dbg !12
  %v3 = load i32, i32* %v1, align 4, !dbg !13
  %v4 = icmp sgt i32 %v3, 1, !dbg !15
  br i1 %v4, label %b1, label %b2, !dbg !16

b1:                                               ; preds = %b0
  %v5 = load i32, i32* %v1, align 4, !dbg !17
  %v6 = load i32, i32* %v1, align 4, !dbg !18
  %v7 = sub nsw i32 %v6, 1, !dbg !19
  %v8 = call i32 @f0(i32 %v7), !dbg !20
  %v9 = mul nsw i32 %v5, %v8, !dbg !21
  store i32 %v9, i32* %v0, align 4, !dbg !22
  br label %b3, !dbg !22

b2:                                               ; preds = %b0
  %v10 = load i32, i32* %v1, align 4, !dbg !23
  store i32 %v10, i32* %v0, align 4, !dbg !24
  br label %b3, !dbg !24

b3:                                               ; preds = %b2, %b1
  %v11 = load i32, i32* %v0, align 4, !dbg !25
  ret i32 %v11, !dbg !25
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "Clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "/tmp/test.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "factorial", scope: !1, file: !1, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !8}
!8 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !DILocalVariable(name: "value", arg: 1, scope: !5, file: !1, line: 1, type: !8)
!10 = !DILocation(line: 1, column: 20, scope: !5)
!11 = !DILocalVariable(name: "local_var", scope: !5, file: !1, line: 2, type: !8)
!12 = !DILocation(line: 2, column: 7, scope: !5)
!13 = !DILocation(line: 3, column: 7, scope: !14)
!14 = distinct !DILexicalBlock(scope: !5, file: !1, line: 3, column: 7)
!15 = !DILocation(line: 3, column: 13, scope: !14)
!16 = !DILocation(line: 3, column: 7, scope: !5)
!17 = !DILocation(line: 4, column: 12, scope: !14)
!18 = !DILocation(line: 4, column: 28, scope: !14)
!19 = !DILocation(line: 4, column: 33, scope: !14)
!20 = !DILocation(line: 4, column: 18, scope: !14)
!21 = !DILocation(line: 4, column: 17, scope: !14)
!22 = !DILocation(line: 4, column: 5, scope: !14)
!23 = !DILocation(line: 5, column: 15, scope: !14)
!24 = !DILocation(line: 5, column: 8, scope: !14)
!25 = !DILocation(line: 6, column: 1, scope: !5)
