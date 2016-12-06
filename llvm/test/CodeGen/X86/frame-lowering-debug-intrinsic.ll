; Test ensuring debug intrinsics do not affect generated function prologue.
;
; RUN: llc -O1 -mtriple=x86_64-unknown-unknown -o - %s | FileCheck %s


define i64 @noDebug(i64 %a) {
  %call = call i64 @fn(i64 %a, i64 0)
  ret i64 %call
}

; CHECK-LABEL: noDebug
; CHECK: popq %rcx
; CHECK: ret


define i64 @withDebug(i64 %a) !dbg !4 {
  %call = call i64 @fn(i64 %a, i64 0)
  tail call void @llvm.dbg.value(metadata i64 %call, i64 0, metadata !5, metadata !6), !dbg !7
  ret i64 %call
}

; CHECK-LABEL: withDebug
; CHECK: popq %rcx
; CHECK: ret


declare i64 @fn(i64, i64)

declare void @llvm.dbg.value(metadata, i64, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2,!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 4.0.0")
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "withDebug", unit: !0)
!5 = !DILocalVariable(name: "w", scope: !4)
!6 = !DIExpression()
!7 = !DILocation(line: 210, column: 12, scope: !4)
