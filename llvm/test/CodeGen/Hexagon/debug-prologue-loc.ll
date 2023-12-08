; RUN: llc -O2 -march=hexagon < %s | FileCheck %s

; Broken after r326208.
; XFAIL: *
; CHECK: allocframe{{.*}}
; CHECK-NEXT: }
; CHECK-NEXT:{{.*}}tmp{{[0-9]+}}:
; CHECK-NEXT: .loc {{[0-9 ]+}} prologue_end

; Function Attrs: nounwind
define i32 @f0(i32 %a0, i32 %a1) #0 !dbg !5 {
b0:
  %v0 = alloca i32, align 4
  %v1 = alloca i32, align 4
  %v2 = alloca i32*, align 4
  store i32 %a0, i32* %v0, align 4
  call void @llvm.dbg.declare(metadata i32* %v0, metadata !9, metadata !DIExpression()), !dbg !10
  store i32 %a1, i32* %v1, align 4
  call void @llvm.dbg.declare(metadata i32* %v1, metadata !11, metadata !DIExpression()), !dbg !12
  call void @llvm.dbg.declare(metadata i32** %v2, metadata !13, metadata !DIExpression()), !dbg !15
  store i32* %v1, i32** %v2, align 4, !dbg !15
  %v3 = load i32, i32* %v0, align 4, !dbg !16
  %v4 = load i32*, i32** %v2, align 4, !dbg !17
  %v5 = call i32 @f1(i32* %v4), !dbg !18
  %v6 = add nsw i32 %v3, %v5, !dbg !19
  ret i32 %v6, !dbg !20
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind
define i32 @f1(i32* %a0) #0 !dbg !21 {
b0:
  %v0 = alloca i32*, align 4
  store i32* %a0, i32** %v0, align 4
  call void @llvm.dbg.declare(metadata i32** %v0, metadata !24, metadata !DIExpression()), !dbg !25
  ret i32 0, !dbg !26
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "Clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "/tmp/test.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !8, !8}
!8 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !DILocalVariable(name: "a", arg: 1, scope: !5, file: !1, line: 1, type: !8)
!10 = !DILocation(line: 1, column: 13, scope: !5)
!11 = !DILocalVariable(name: "b", arg: 2, scope: !5, file: !1, line: 1, type: !8)
!12 = !DILocation(line: 1, column: 20, scope: !5)
!13 = !DILocalVariable(name: "ptr", scope: !5, file: !1, line: 2, type: !14)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 32, align: 32)
!15 = !DILocation(line: 2, column: 8, scope: !5)
!16 = !DILocation(line: 3, column: 10, scope: !5)
!17 = !DILocation(line: 3, column: 16, scope: !5)
!18 = !DILocation(line: 3, column: 12, scope: !5)
!19 = !DILocation(line: 3, column: 11, scope: !5)
!20 = !DILocation(line: 3, column: 3, scope: !5)
!21 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 7, type: !22, isLocal: false, isDefinition: true, scopeLine: 7, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!22 = !DISubroutineType(types: !23)
!23 = !{!8, !14}
!24 = !DILocalVariable(name: "var", arg: 1, scope: !21, file: !1, line: 7, type: !14)
!25 = !DILocation(line: 7, column: 14, scope: !21)
!26 = !DILocation(line: 8, column: 3, scope: !21)
