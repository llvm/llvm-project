; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

; ptxas has no special meaning for '$' character, so it should be used
; without parens.

@"$str" = private addrspace(1) constant [4 x i8] c"str\00"

declare void @str2(ptr %str)
define void @str1() {
entry:
;; CHECK: mov.u64 %rd{{[0-9]+}}, $str;
  tail call void @str2(ptr addrspacecast (ptr addrspace(1) @"$str" to ptr))
  ret void
}
