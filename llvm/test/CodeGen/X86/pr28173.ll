; RUN: llc -mattr=+avx512f < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Note that the kmovs should really *not* appear in the output, this is an
; artifact of the current poor lowering. This is tracked by PR28175.

define i64 @foo64(i1 zeroext %i, i32 %j) #0 {
; CHECK-LABEL: foo64:
; CHECK:       # BB#0:
; CHECK-NEXT:    # kill
; CHECK-NEXT:    orq $-2, %rdi
; CHECK-NEXT:    movq %rdi, %rax
; CHECK-NEXT:    retq
  br label %bb

bb:
  %z = zext i1 %i to i64
  %v = or i64 %z, -2
  br label %end

end:
  ret i64 %v
}

define i16 @foo16(i1 zeroext %i, i32 %j) #0 {
; CHECK-LABEL: foo16:
; CHECK:       # BB#0:
; CHECK-NEXT:    orl $65534, %edi # imm = 0xFFFE
; CHECK-NEXT:    movl %edi, %eax
; CHECK-NEXT:    retq
  br label %bb

bb:
  %z = zext i1 %i to i16
  %v = or i16 %z, -2
  br label %end

end:
  ret i16 %v
}
