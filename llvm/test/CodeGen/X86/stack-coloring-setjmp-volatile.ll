; RUN: llc -mtriple=x86_64-linux -no-stack-coloring=false -debug-only=stack-coloring < %s -o /dev/null 2>&1 | FileCheck %s
; REQUIRES: asserts

; Test that volatile stack slots accessed after setjmp are not merged.
; Volatile variables must retain their values across longjmp, so their
; stack slots cannot be reused even if their lifetimes don't overlap.

declare i32 @setjmp(ptr) returns_twice

; CHECK-LABEL: volatile_after_setjmp
; CHECK: Volatile slots after setjmp : { {{.*}}1{{.*}} }
; CHECK: Merge 0 slots

define void @volatile_after_setjmp(ptr %jump_buffer) {
entry:
  %foo = alloca i32, align 4
  %bar = alloca [100 x i32], align 4

  call void @llvm.lifetime.start.p0(i64 4, ptr %foo)

  %setjmp_result = call i32 @setjmp(ptr %jump_buffer)
  %cmp = icmp eq i32 %setjmp_result, 0
  br i1 %cmp, label %after_setjmp, label %continue

after_setjmp:
  store volatile i32 100, ptr %foo, align 4
  br label %exit

continue:
  call void @llvm.lifetime.end.p0(i64 4, ptr %foo)
  call void @llvm.lifetime.start.p0(i64 400, ptr %bar)
  call void @llvm.lifetime.end.p0(i64 400, ptr %bar)
  br label %exit

exit:
  ret void
}

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)
