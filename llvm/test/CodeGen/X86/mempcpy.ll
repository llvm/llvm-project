;  RUN: llc < %s -mtriple=x86_64-unknown-linux -O2 | FileCheck %s
;  RUN: llc < %s -mtriple=i686-unknown-linux -O2 | FileCheck %s

; This test checks that:
; (1)  mempcpy is lowered as memcpy, and 
; (2)  its return value is DST+N i.e. the dst pointer adjusted by the copy size.
; To keep the testing of (2) independent of the exact instructions used to
; adjust the dst pointer, DST+N is explicitly computed and stored to a global
; variable G before the mempcpy call. This instance of DST+N causes the repeat
; DST+N done in the context of the return value of mempcpy to be redundant, and
; the first instance to be reused as the return value. This allows the check for
; (2) to be expressed as verifying that the MOV to store DST+N to G and
; the MOV to copy DST+N to %rax use the same source register.
@G = common global i8* null, align 8

; CHECK-LABEL: RET_MEMPCPY:
; CHECK: mov{{.*}} [[REG:%[er][a-z0-9]+]], {{.*}}G
; CHECK: call{{.*}} {{.*}}memcpy
; CHECK: mov{{.*}} [[REG]], %{{[er]}}ax
;
define i8* @RET_MEMPCPY(i8* %DST, i8* %SRC, i64 %N) {
  %add.ptr = getelementptr inbounds i8, i8* %DST, i64 %N
  store i8* %add.ptr, i8** @G, align 8
  %call = tail call i8* @mempcpy(i8* %DST, i8* %SRC, i64 %N) 
  ret i8* %call
}

declare i8* @mempcpy(i8*, i8*, i64)
