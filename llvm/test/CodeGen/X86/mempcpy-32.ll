;  RUN: llc < %s -mtriple=i686-unknown-linux -O2 | FileCheck %s

; This tests the i686 lowering of mempcpy.
; Also see mempcpy.ll

@G = common global ptr null, align 8

; CHECK-LABEL: RET_MEMPCPY:
; CHECK: movl [[REG:%e[a-z0-9]+]], {{.*}}G
; CHECK: calll {{.*}}memcpy
; CHECK: movl [[REG]], %eax
;
define ptr @RET_MEMPCPY(ptr %DST, ptr %SRC, i32 %N) {
  %add.ptr = getelementptr inbounds i8, ptr %DST, i32 %N
  store ptr %add.ptr, ptr @G, align 8
  %call = tail call ptr @mempcpy(ptr %DST, ptr %SRC, i32 %N)
  ret ptr %call
}

declare ptr @mempcpy(ptr, ptr, i32)
