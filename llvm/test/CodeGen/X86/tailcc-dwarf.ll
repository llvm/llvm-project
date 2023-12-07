; RUN: llc -mtriple=x86_64-unknown-linux-gnu -O0 --frame-pointer=non-leaf %s -o - | FileCheck %s

%block = type { %blockheader, [0 x i64*] }
%blockheader = type { i64 }

define void @scanStackRoots(i32) {
  ret void
}

define i32 @main(i32 %argc, i8** %argv) {
entry:
  %0 = call tailcc %block* @apply_rule_6870(%block* null, %block* null)
  ret i32 0
}

define internal tailcc %block* @apply_rule_6870(%block* %0, %block* %1) {
entry:
  %2 = tail call tailcc %block* @sender12(%block* %0, %block* %1)
  ret %block* null
}

define internal tailcc %block* @sender12(%block* %0, %block* %1) {
; CHECK-LABEL: sender12:
; CHECK: .cfi_startproc
; CHECK: subq $8160, %rsp
; CHECK: pushq %rbp
; CHECK: .cfi_def_cfa_offset 8176
; CHECK: .cfi_offset %rbp, -8176
entry:
  %a = alloca [1024 x i32]
  %b = load [1024 x i32], [1024 x i32]* %a
  call void @scanStackRoots(i32 1)
  %2 = tail call tailcc %block* @apply_rule_6300(%block* %0, %block* %1, [1024 x i32] %b)
  ret %block* %2
}

define internal tailcc %block* @apply_rule_6300(%block* %0, %block* %1, [1024 x i32] %2) {
entry:
  %3 = tail call tailcc %block* @sender4(%block* %0, %block* %1)
  ret %block* %3
}

define internal tailcc %block* @sender4(%block* %0, %block* %1) {
entry:
  call void @scanStackRoots(i32 2)
  ret %block* null
}
