; RUN: not llc --mtriple=loongarch32 2>&1 < %s | FileCheck %s
; RUN: not llc --mtriple=loongarch64 2>&1 < %s | FileCheck %s

define i32 @non_exit_r32(i32 %a) nounwind {
; CHECK: error: couldn't allocate input reg for constraint '{$r32}'
  %1 = tail call i32 asm "addi.w $0, $1, 1", "=r,{$r32}"(i32 %a)
  ret i32 %1
}

define i32 @non_exit_foo(i32 %a) nounwind {
; CHECK: error: couldn't allocate input reg for constraint '{$foo}'
  %1 = tail call i32 asm "addi.w $0, $1, 1", "=r,{$foo}"(i32 %a)
  ret i32 %1
}
