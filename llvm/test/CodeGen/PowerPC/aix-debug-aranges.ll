; RUN: llc -filetype=obj -function-sections -generate-arange-section < %s | \
; RUN: llvm-objdump -dr - | FileCheck %s

; Make sure that enabling debug_arange does not corrupt branches.

target triple = "powerpc64-ibm-aix"

define i64 @fn1() {
; CHECK-LABEL: <.fn1>:
; CHECK: bl {{.*}} <.fn2>
; CHECK-NEXT: R_RBR .fn2
  %1 = call i64 @fn2()
  ret i64 %1
}

define i64 @fn2() !dbg !4 {
  ret i64 0
}

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !2, producer: "clang LLVM (rustc version 1.95.0-dev)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !3, globals: !3, splitDebugInlining: false, nameTableKind: None)
!2 = !DIFile(filename: "foo", directory: "")
!3 = !{}
!4 = distinct !DISubprogram(name: "fn2", file: !2, line: 277, type: !5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1, templateParams: !3, retainedNodes: !3)
!5 = !DISubroutineType(types: !3)
