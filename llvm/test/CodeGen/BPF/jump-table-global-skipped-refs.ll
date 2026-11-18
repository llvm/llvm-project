; RUN: not llc -mtriple=bpfel -mcpu=v4 -filetype=obj -o /dev/null %s 2>&1 | FileCheck %s

; FIXME: These tables are only lowered to jump tables, so they are not
; emitted. They are still referenced by the initializer of another
; global or by debug info, which then fails to assemble.

; CHECK-DAG: error: Undefined temporary symbol .Ltbl_ref{{$}}
; CHECK-DAG: error: Undefined temporary symbol .Ltbl_dbg{{$}}

@tbl_ref = private constant [2 x ptr] [ptr blockaddress(@use_ref, %a), ptr blockaddress(@use_ref, %b)]
@tbl_dbg = private constant [2 x ptr] [ptr blockaddress(@use_dbg, %a), ptr blockaddress(@use_dbg, %b)], !dbg !0
@ref = global ptr @tbl_ref

define i32 @use_ref(i64 %i) {
entry:
  %gep = getelementptr [2 x ptr], ptr @tbl_ref, i64 0, i64 %i
  %tgt = load ptr, ptr %gep
  indirectbr ptr %tgt, [label %a, label %b]

a:
  ret i32 1

b:
  ret i32 2
}

define i32 @use_dbg(i64 %i) {
entry:
  %gep = getelementptr [2 x ptr], ptr @tbl_dbg, i64 0, i64 %i
  %tgt = load ptr, ptr %gep
  indirectbr ptr %tgt, [label %a, label %b]

a:
  ret i32 1

b:
  ret i32 2
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!6, !7}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "tbl_dbg", scope: !2, file: !3, line: 1, type: !5, isLocal: true, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4)
!3 = !DIFile(filename: "t.c", directory: "/")
!4 = !{!0}
!5 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!6 = !{i32 7, !"Dwarf Version", i32 5}
!7 = !{i32 2, !"Debug Info Version", i32 3}
