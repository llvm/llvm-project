; RUN: llc -mtriple=x86_64-unknown-linux-gnu -O0 --frame-pointer=non-leaf %s -o - | FileCheck %s

%block = type { %blockheader, [0 x ptr] }
%blockheader = type { i64 }

define void @scanStackRoots(i32) {
  ret void
}

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %0 = call tailcc ptr @apply_rule_6870(ptr null, ptr null)
  ret i32 0
}

define internal tailcc ptr @apply_rule_6870(ptr %0, ptr %1) {
entry:
  %2 = tail call tailcc ptr @sender12(ptr %0, ptr %1)
  ret ptr null
}

define internal tailcc ptr @sender12(ptr %0, ptr %1) {
; CHECK-LABEL: sender12:
; CHECK: .cfi_startproc
; CHECK: subq $8160, %rsp
; CHECK: pushq %rbp
; CHECK: .cfi_def_cfa_offset 8176
; CHECK: .cfi_offset %rbp, -8176
entry:
  %a = alloca [1024 x i32]
  %b = load [1024 x i32], ptr %a
  call void @scanStackRoots(i32 1)
  %2 = tail call tailcc ptr @apply_rule_6300(ptr %0, ptr %1, [1024 x i32] %b)
  ret ptr %2
}

define internal tailcc ptr @apply_rule_6300(ptr %0, ptr %1, [1024 x i32] %2) {
entry:
  %3 = tail call tailcc ptr @sender4(ptr %0, ptr %1)
  ret ptr %3
}

define internal tailcc ptr @sender4(ptr %0, ptr %1) {
entry:
  call void @scanStackRoots(i32 2)
  ret ptr null
}
