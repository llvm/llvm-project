; RUN: opt %s -debugify -early-cse -S | FileCheck %s
define i32 @foo(i64 %nose, i32 %more) {
; CHECK-LABEL: @foo(
; CHECK: call void @llvm.dbg.value(metadata i64 %nose, metadata [[V1:![0-9]+]], metadata !DIExpression(DW_OP_LLVM_convert, 64, DW_ATE_unsigned, DW_OP_LLVM_convert, 32, DW_ATE_unsigned
; CHECK: call void @llvm.dbg.value(metadata i64 %nose.shift, metadata [[V2:![0-9]+]]
; CHECK: call void @llvm.dbg.value(metadata i64 %nose.shift, metadata [[V3:![0-9]+]], metadata !DIExpression(DW_OP_LLVM_convert, 64, DW_ATE_unsigned, DW_OP_LLVM_convert, 32, DW_ATE_unsigned

entry:
  %nose.trunc = trunc i64 %nose to i32
  %nose.shift = lshr i64 %nose, 32
  %nose.trunc.2 = trunc i64 %nose.shift to i32
  %add = add nsw i32 %more, 1
  ret i32 %add
}

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 2}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{!"clang version 9.0.0 "}

; CHECK: [[V1]] = !DILocalVariable(
; CHECK: [[V2]] = !DILocalVariable(
; CHECK: [[V3]] = !DILocalVariable(
