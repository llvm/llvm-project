; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: compare value and new value type do not match
define void @cmpxchg(ptr %p, i32 %a, i64 %b) {
    %val_success = cmpxchg ptr %p, i32 %a, i64 %b acq_rel monotonic
    ret void
}
