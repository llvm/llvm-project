;  RUN: llc < %s -mtriple=i686-unknown-linux -O2 | FileCheck %s

; This tests the i686 lowering of mempcpy.
; Also see mempcpy.ll

@G = common global i8* null, align 8

; CHECK-LABEL: RET_MEMPCPY:
; CHECK: movl [[REG:%e[a-z0-9]+]], {{.*}}G
; CHECK: calll {{.*}}memcpy
; CHECK: movl [[REG]], %eax
;
define i8* @RET_MEMPCPY(i8* %DST, i8* %SRC, i32 %N) {
  %add.ptr = getelementptr inbounds i8, i8* %DST, i32 %N
  store i8* %add.ptr, i8** @G, align 8
  %call = tail call i8* @mempcpy(i8* %DST, i8* %SRC, i32 %N)
  ret i8* %call
}

declare i8* @mempcpy(i8*, i8*, i32)
