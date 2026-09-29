; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64-nvidia-cuda | %ptxas-verify %}


define i32 @foo(i32 %a, i32 %b) !dbg !3 {

; CHECK:     .loc    [[FILE:[0-9]+]] 26 0          // extra-lineinfo.cu:26:0
; CHECK-NOT: .loc    [[FILE]]        26 0          // extra-lineinfo.cu:26:0
; CHECK:     .file   [[FILE]] "/test/directory/extra-lineinfo.cu"

  %add = add i32 %b, %a, !dbg !6
  ret i32 %add, !dbg !6
}

!llvm.dbg.cu = !{!0}
!nvvm.annotations = !{}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: DebugDirectivesOnly)
!1 = !DIFile(filename: "extra-lineinfo.cu", directory: "/test/directory/")
!2 = !{i32 1, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "kernel", linkageName: "foo", scope: !1, file: !1, line: 123, type: !4, scopeLine: 26, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{}
!6 = !DILocation(line: 40, column: 22, scope: !31)
!31 = distinct !DILexicalBlock(scope: !3, file: !1, line: 3, column: 17)
