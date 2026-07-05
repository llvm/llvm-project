; RUN: llc -O2 -mtriple=x86_64-- -stop-after=finalize-isel < %s | FileCheck %s

define i1 @fold_test(ptr %x, i64 %l) {
entry:
  %0 = load i64, ptr %x, align 8
  %and = and i64 %0, %l
  %tobool = icmp ne i64 %and, 0
  ret i1 %tobool

  ; Folding the load+and+icmp instructions into a TEST64mr instruction
  ; should preserve memory operands.
  ; CHECK: TEST64mr {{.*}} :: (load (s64) from {{%.*}})
}

