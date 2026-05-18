; Verify that DISubprogram rejects a non-DISubroutineType value for its `type:`
; field.  Missing/null `type:` is now rejected at parse time (see
; llvm/test/Assembler/disubprogram-type-required.ll).
; See https://github.com/llvm/llvm-project/issues/186557.

; RUN: llvm-as -disable-output < %s 2>&1 | FileCheck %s

; CHECK: invalid subroutine type
; CHECK-NEXT: ![[INVALID:[0-9]+]] = distinct !DISubprogram(name: "invalid_type"
; CHECK: warning: ignoring invalid debug info
; CHECK-NOT: valid_void_type

define void @invalid_type() !dbg !4 {
  ret void, !dbg !5
}

define void @valid_void_type() !dbg !6 {
  ret void, !dbg !7
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", emissionKind: FullDebug)
!1 = !DIFile(filename: "x.c", directory: "/")
!2 = !{null}
!3 = !DISubroutineType(types: !2)
!4 = distinct !DISubprogram(name: "invalid_type", scope: !1, file: !1, line: 1, type: !1, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!5 = !DILocation(line: 1, column: 1, scope: !4)
!6 = distinct !DISubprogram(name: "valid_void_type", scope: !1, file: !1, line: 2, type: !3, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0)
!7 = !DILocation(line: 2, column: 1, scope: !6)
!8 = !{i32 1, !"Debug Info Version", i32 3}
