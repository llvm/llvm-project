; RUN: llc -mtriple=arm64-apple-ios < %s | FileCheck %s
; RUN: llc -mtriple=arm64-linux-gnu -relocation-model=pic < %s | FileCheck %s --check-prefix=CHECK-LINUX
; <rdar://problem/11392109>

define hidden void @t(ptr %addr) optsize ssp {
entry:
  store i64 zext (i32 ptrtoint (ptr @x to i32) to i64), ptr %addr, align 8
; CHECK:             adrp    x{{[0-9]+}}, _x@GOTPAGE
; CHECK:        ldr     x{{[0-9]+}}, [x{{[0-9]+}}, _x@GOTPAGEOFF]
; CHECK-NEXT:        and     x{{[0-9]+}}, x{{[0-9]+}}, #0xffffffff
; CHECK-NEXT:        str     x{{[0-9]+}}, [x{{[0-9]+}}]
  ret void
}

declare i64 @x(i32) optsize

; Worth checking the Linux code is sensible too: only way to access
; the GOT is via a 64-bit load. Just loading wN is unacceptable
; (there's no ELF relocation to do that).

; CHECK-LINUX: adrp {{x[0-9]+}}, :got:x
; CHECK-LINUX: ldr {{x[0-9]+}}, [{{x[0-9]+}}, :got_lo12:x]
