; RUN: llc < %s -mtriple=i686-pc-linux | FileCheck %s


; Function Attrs: optsize
declare void @foo(i32, i32) #0
declare x86_stdcallcc void @stdfoo(i32, i32) #0

; CHECK-LABEL: test1:
; CHECK: subl $8, %esp
; CHECK: .cfi_adjust_cfa_offset 8
; CHECK: pushl $2
; CHECK: .cfi_adjust_cfa_offset 4
; CHECK: pushl $1
; CHECK: .cfi_adjust_cfa_offset 4
; CHECK: calll foo
; CHECK: addl $16, %esp
; CHECK: .cfi_adjust_cfa_offset -16
; CHECK: subl $8, %esp
; CHECK: .cfi_adjust_cfa_offset 8
; CHECK: pushl $4
; CHECK: .cfi_adjust_cfa_offset 4
; CHECK: pushl $3
; CHECK: .cfi_adjust_cfa_offset 4
; CHECK: calll stdfoo
; CHECK: .cfi_adjust_cfa_offset -8
; CHECK: addl $8, %esp
; CHECK: .cfi_adjust_cfa_offset -8
define void @test1() #0 !dbg !4 {
entry:
  tail call void @foo(i32 1, i32 2) #1, !dbg !10
  tail call x86_stdcallcc void @stdfoo(i32 3, i32 4) #1, !dbg !11
  ret void, !dbg !12
}

attributes #0 = { nounwind optsize }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.8.0 (trunk 250289)", isOptimized: true, runtimeVersion: 0, emissionKind: 1, enums: !2, subprograms: !3)
!1 = !DIFile(filename: "foo.c", directory: "foo")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "test1", scope: !1, file: !1, line: 3, type: !5, isLocal: false, isDefinition: true, scopeLine: 3, isOptimized: true, variables: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{!"clang version 3.8.0 (trunk 250289)"}
!10 = !DILocation(line: 4, column: 3, scope: !4)
!11 = !DILocation(line: 5, column: 3, scope: !4)
!12 = !DILocation(line: 6, column: 1, scope: !4)
