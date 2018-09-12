; RUN: llc -filetype=obj -mtriple=riscv32 -mattr=+relax %s -o - \
; RUN:     | llvm-readobj -r | FileCheck -check-prefix=RELAX %s
;
; RELAX: .rela.debug_line {
; RELAX: R_RISCV_ADD16
; RELAX: R_RISCV_SUB16
source_filename = "line.c"

; Function Attrs: noinline nounwind optnone
define i32 @init() !dbg !7 {
entry:
  ret i32 0, !dbg !11
}

; Function Attrs: noinline nounwind optnone
define i32 @foo(i32 signext %value) !dbg !12 {
entry:
  %value.addr = alloca i32, align 4
  store i32 %value, i32* %value.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %value.addr, metadata !15, metadata !DIExpression()), !dbg !16
  %0 = load i32, i32* %value.addr, align 4, !dbg !17
  ret i32 %0, !dbg !18
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata)

; Function Attrs: noinline nounwind optnone
define i32 @bar() !dbg !19 {
entry:
  %result = alloca i32, align 4
  %v = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %result, metadata !20, metadata !DIExpression()), !dbg !21
  call void @llvm.dbg.declare(metadata i32* %v, metadata !22, metadata !DIExpression()), !dbg !23
  %call = call i32 @init(), !dbg !24
  store i32 %call, i32* %v, align 4, !dbg !23
  %0 = load i32, i32* %v, align 4, !dbg !25
  %call1 = call i32 @foo(i32 signext %0), !dbg !26
  store i32 %call1, i32* %result, align 4, !dbg !27
  %1 = load i32, i32* %result, align 4, !dbg !28
  ret i32 %1, !dbg !29
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "line.c", directory: "./")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!7 = distinct !DISubprogram(name: "init", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 2, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DILocation(line: 3, column: 3, scope: !7)
!12 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 6, type: !13, isLocal: false, isDefinition: true, scopeLine: 7, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!13 = !DISubroutineType(types: !14)
!14 = !{!10, !10}
!15 = !DILocalVariable(name: "value", arg: 1, scope: !12, file: !1, line: 6, type: !10)
!16 = !DILocation(line: 6, column: 13, scope: !12)
!17 = !DILocation(line: 8, column: 10, scope: !12)
!18 = !DILocation(line: 8, column: 3, scope: !12)
!19 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 11, type: !8, isLocal: false, isDefinition: true, scopeLine: 12, isOptimized: false, unit: !0, retainedNodes: !2)
!20 = !DILocalVariable(name: "result", scope: !19, file: !1, line: 13, type: !10)
!21 = !DILocation(line: 13, column: 7, scope: !19)
!22 = !DILocalVariable(name: "v", scope: !19, file: !1, line: 14, type: !10)
!23 = !DILocation(line: 14, column: 7, scope: !19)
!24 = !DILocation(line: 14, column: 11, scope: !19)
!25 = !DILocation(line: 16, column: 16, scope: !19)
!26 = !DILocation(line: 16, column: 12, scope: !19)
!27 = !DILocation(line: 16, column: 10, scope: !19)
!28 = !DILocation(line: 18, column: 10, scope: !19)
!29 = !DILocation(line: 18, column: 3, scope: !19)
