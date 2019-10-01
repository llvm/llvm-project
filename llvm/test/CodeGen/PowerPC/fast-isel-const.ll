; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 | FileCheck %s --check-prefix=ELF64

define zeroext i1 @testi1(i8 %in) nounwind {
entry:
  %c = icmp eq i8 %in, 5
  br i1 %c, label %true, label %false

; ELF64-LABEL: @testi1

true:
  br label %end

; ELF64-NOT: li {{[0-9]+}}, -1
; ELF64: li {{[0-9]+}}, 1

false:
  br label %end

; ELF64: li {{[0-9]+}}, 0

end:
  %r = phi i1 [ 0, %false], [ 1, %true ]
  ret i1 %r

; ELF64: blr
}

