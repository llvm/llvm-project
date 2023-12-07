; RUN: not mlir-translate %s -import-llvm -split-input-file -error-diagnostics-only 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: not mlir-translate %s -import-llvm -split-input-file 2>&1 | FileCheck %s --check-prefix=DEFAULT
; RUN: not mlir-translate %s -import-llvm -split-input-file -emit-expensive-warnings 2>&1 | FileCheck %s --check-prefix=EXPENSIVE

; ERROR-NOT: warning:
; DEFAULT:   warning:
; EXPENSIVE: warning:
define void @warning(i64 %n, ptr %A) {
entry:
  br label %end, !llvm.loop !0
end:
  ret void
}

!0 = distinct !{!0, !1, !2}
!1 = !{!"llvm.loop.disable_nonforced"}
!2 = !{!"llvm.loop.typo"}

; // -----

; ERROR:     error:
; DEFAULT:   error:
; EXPENSIVE: error:
define i32 @error(ptr %dst) {
  indirectbr ptr %dst, [label %bb1, label %bb2]
bb1:
  ret i32 0
bb2:
  ret i32 1
}

; // -----

declare void @llvm.dbg.value(metadata, metadata, metadata)

; ERROR-NOT:   warning:
; DEFAULT-NOT: warning:
; EXPENSIVE:   warning:
define void @dropped_instruction(i64 %arg1) {
  call void @llvm.dbg.value(metadata i64 %arg1, metadata !3, metadata !DIExpression(DW_OP_plus_uconst, 42, DW_OP_stack_value)), !dbg !5
  ret void
}

!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C, file: !2)
!2 = !DIFile(filename: "import-failure.ll", directory: "/")
!3 = !DILocalVariable(scope: !4, name: "arg1", file: !2, line: 1, arg: 1, align: 64);
!4 = distinct !DISubprogram(name: "intrinsic", scope: !2, file: !2, spFlags: DISPFlagDefinition, unit: !1)
!5 = !DILocation(line: 1, column: 2, scope: !4)
