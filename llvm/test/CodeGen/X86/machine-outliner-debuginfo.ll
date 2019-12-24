; RUN: llc -enable-machine-outliner -mtriple=x86_64-apple-darwin < %s | FileCheck %s

@x = global i32 0, align 4, !dbg !0

define i32 @main() #0 !dbg !11 {
  ; CHECK-LABEL: _main:
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  ; There is a debug value in the middle of this section, make sure debug values are ignored.
  ; CHECK: callq
  ; CHECK-SAME: OUTLINED_FUNCTION_0
  store i32 1, i32* %2, align 4
  store i32 2, i32* %3, align 4
  store i32 3, i32* %4, align 4
  call void @llvm.dbg.value(metadata i32 10, i64 0, metadata !15, metadata !16), !dbg !17
  store i32 4, i32* %5, align 4
  store i32 0, i32* @x, align 4, !dbg !24
  call void asm sideeffect "", "~{memory},~{dirflag},~{fpsr},~{flags}"()
  ; This is the same sequence of instructions without a debug value. It should be outlined
  ; in the same way.
  ; CHECK: callq
  ; CHECK-SAME: OUTLINED_FUNCTION_0
  store i32 1, i32* %2, align 4
  store i32 2, i32* %3, align 4
  store i32 3, i32* %4, align 4
  store i32 4, i32* %5, align 4
  store i32 1, i32* @x, align 4, !dbg !14
  ret i32 0, !dbg !25
}

; CHECK: OUTLINED_FUNCTION_0:
; CHECK-NOT:  .loc  {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{^(is_stmt)}}
; CHECK-NOT:  ##DEBUG_VALUE: main:{{[a-z]}} <- {{[0-9]+}}
; CHECK:      movl  $1, -{{[0-9]+}}(%rbp)
; CHECK-NEXT: movl  $2, -{{[0-9]+}}(%rbp)
; CHECK-NEXT: movl  $3, -{{[0-9]+}}(%rbp)
; CHECK-NEXT: movl  $4, -{{[0-9]+}}(%rbp)
; CHECK-NEXT: retq

declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { noredzone nounwind ssp uwtable "frame-pointer"="all" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{!10}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "x", scope: !2, file: !3, line: 2, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 5.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "debug-test.c", directory: "dir")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"PIC Level", i32 2}
!10 = !{!"clang version 5.0.0"}
!11 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 4, type: !12, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: false, unit: !2, retainedNodes: !4)
!12 = !DISubroutineType(types: !13)
!13 = !{!6}
!14 = !DILocation(line: 7, column: 4, scope: !11)
!15 = !DILocalVariable(name: "a", scope: !11, file: !3, line: 5, type: !6)
!16 = !DIExpression()
!17 = !DILocation(line: 5, column: 6, scope: !11)
!18 = !DILocalVariable(name: "b", scope: !11, file: !3, line: 5, type: !6)
!19 = !DILocation(line: 5, column: 9, scope: !11)
!20 = !DILocalVariable(name: "c", scope: !11, file: !3, line: 5, type: !6)
!21 = !DILocation(line: 5, column: 12, scope: !11)
!22 = !DILocalVariable(name: "d", scope: !11, file: !3, line: 5, type: !6)
!23 = !DILocation(line: 5, column: 15, scope: !11)
!24 = !DILocation(line: 14, column: 4, scope: !11)
!25 = !DILocation(line: 21, column: 2, scope: !11)
