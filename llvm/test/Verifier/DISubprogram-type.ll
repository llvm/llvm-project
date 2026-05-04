; Verify that DISubprogram requires a non-null DISubroutineType for its
; `type:` field. See https://github.com/llvm/llvm-project/issues/186557.

; RUN: llvm-as -disable-output < %s 2>&1 | FileCheck %s

; CHECK: DISubprogram requires a non-null type
; CHECK-NEXT: ![[MISSING:[0-9]+]] = distinct !DISubprogram(name: "missing_type"
; CHECK: DISubprogram requires a non-null type
; CHECK-NEXT: ![[NULL:[0-9]+]] = distinct !DISubprogram(name: "null_type"
; CHECK: invalid subroutine type
; CHECK-NEXT: ![[INVALID:[0-9]+]] = distinct !DISubprogram(name: "invalid_type"
; CHECK-NEXT: !{{[0-9]+}} = !DIFile(filename: "x.c", directory: "/")
; CHECK: warning: ignoring invalid debug info
; CHECK-NOT: valid_void_type

define void @missing_type() !dbg !4 {
  ret void, !dbg !5
}

define void @null_type() !dbg !7 {
  ret void, !dbg !8
}

define void @invalid_type() !dbg !9 {
  ret void, !dbg !10
}

define void @valid_void_type() !dbg !11 {
  ret void, !dbg !12
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", emissionKind: FullDebug)
!1 = !DIFile(filename: "x.c", directory: "/")
!2 = !{null}
!3 = !DISubroutineType(types: !2)
!4 = distinct !DISubprogram(name: "missing_type", scope: !1, file: !1, line: 1, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!5 = !DILocation(line: 1, column: 1, scope: !4)
!7 = distinct !DISubprogram(name: "null_type", scope: !1, file: !1, line: 2, type: null, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0)
!8 = !DILocation(line: 2, column: 1, scope: !7)
!9 = distinct !DISubprogram(name: "invalid_type", scope: !1, file: !1, line: 3, type: !1, scopeLine: 3, spFlags: DISPFlagDefinition, unit: !0)
!10 = !DILocation(line: 3, column: 1, scope: !9)
!11 = distinct !DISubprogram(name: "valid_void_type", scope: !1, file: !1, line: 4, type: !3, scopeLine: 4, spFlags: DISPFlagDefinition, unit: !0)
!12 = !DILocation(line: 4, column: 1, scope: !11)
!13 = !{i32 1, !"Debug Info Version", i32 3}
