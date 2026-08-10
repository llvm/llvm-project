; RUN: llvm-as -disable-output < %s 2>&1 | FileCheck %s

; The lengths for None and MD5 are wrong; SHA1 has a non-hex digit.
; CHECK: invalid checksum{{$}}
; CHECK: invalid checksum length
; CHECK: warning: ignoring invalid debug info in <stdin>

@t1 = global i32 1, align 4, !dbg !0
@t2 = global i32 0, align 4, !dbg !6

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!11, !12, !13}
!llvm.ident = !{!14}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "t1", scope: !2, file: !10, line: 1, type: !9, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 322159)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "t.c", directory: "/scratch")
!4 = !{}
!5 = !{!0, !6}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "t2", scope: !2, file: !8, line: 1, type: !9, isLocal: false, isDefinition: true)
!8 = !DIFile(filename: "./t2.h", directory: "/scratch", checksumkind: CSK_MD5, checksum: "2222")
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DIFile(filename: "./t1.h", directory: "/scratch", checksumkind: CSK_SHA1, checksum: "123456789012345678901234567890123456789.")
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 1, !"wchar_size", i32 4}
!14 = !{!"clang version 7.0.0 (trunk 322159)"}
