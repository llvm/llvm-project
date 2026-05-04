; REQUIRES: asserts
; RUN: not --crash opt -mtriple=aarch64 -passes=load-store-vectorizer \
; RUN:   -disable-output %s 2>&1 | FileCheck %s

define i32 @load_cycle(ptr %x) {
; CHECK: Unexpected cycle while re-ordering instructions
entry:
  %gep.x.1 = getelementptr inbounds [2 x i32], ptr %x, i32 0, i32 1
  %load.x.1 = load i32, ptr %gep.x.1
  %rem = urem i32 %load.x.1, 1
  %gep.x.2 = getelementptr inbounds [2 x i32], ptr %x, i32 %rem, i32 0
  %load.x.2 = load i32, ptr %gep.x.2
  %ret = add i32 %load.x.2, %load.x.1
  ret i32 %ret
}
