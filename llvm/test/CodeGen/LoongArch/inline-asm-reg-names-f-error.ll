; RUN: not llc --mtriple=loongarch32 --mattr=+f,+d 2>&1 < %s | FileCheck %s
; RUN: not llc --mtriple=loongarch64 --mattr=+f,+d 2>&1 < %s | FileCheck %s

define double @non_exit_f32(double %a) nounwind {
; CHECK: error: couldn't allocate input reg for constraint '{$f32}'
  %1 = tail call double asm "fabs.d $0, $1", "=f,{$f32}"(double %a)
  ret double %1
}

define double @non_exit_foo(double %a) nounwind {
; CHECK: error: couldn't allocate input reg for constraint '{$foo}'
  %1 = tail call double asm "fabs.d $0, $1", "=f,{$foo}"(double %a)
  ret double %1
}
