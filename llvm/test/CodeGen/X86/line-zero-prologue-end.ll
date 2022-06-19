; RUN: llc -filetype=asm -mtriple=x86_64-apple-macosx12.0.0 -O0 %s -o - | FileCheck %s
; CHECK: Lfunc_begin0:
; CHECK-NEXT: .file{{.+}}
; CHECK-NEXT: .loc 1 2 0 ## test/test.c:2:0{{$}}
; CHECK-NEXT: .cfi_startproc
; CHECK-NEXT: ## %bb.{{[0-9]+}}:
; CHECK-NEXT: .loc 1 0 5 {{is_stmt [0-9]+}} ## test/test.c:0:5{{$}}
@x = common global i32 0, align 4
define void @test() #0 !dbg !9 {
  store i32 1, i32* @x, align 4, !dbg !12
  ret void, !dbg !14
}
!llvm.module.flags = !{!0,!2,!4}
!llvm.dbg.cu = !{!5}
!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 0]}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 7, !"PIC Level", i32 2}
!5 = distinct !DICompileUnit(language: DW_LANG_C99, file: !6, producer: "Apple clang version 13.0.0 (clang-1300.0.29.3)", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !7, nameTableKind: None, sysroot: "/Library/Developer/CommandLineTools/SDKs/MacOSX12.0.sdk", sdk: "MacOSX12.0.sdk")
!6 = !DIFile(filename: "/Users/shubham/Development/test/test.c", directory: "/Users/shubham/Development/deltaTest")
!7 = !{}
!9 = distinct !DISubprogram(name: "test", scope: !10, file: !10, line: 2, type: !11, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!10 = !DIFile(filename: "test/test.c", directory: "/Users/shubham/Development")
!11 = !DISubroutineType(types: !7)
!12 = !DILocation(line: 0, column: 5, scope: !9)
!14 = !DILocation(line: 3, column: 1, scope: !9)