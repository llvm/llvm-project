; RUN: llc < %s -mtriple=arm64ec-pc-windows-msvc -O0 -o - | FileCheck %s

;; The x87 `long double` is an 80-bit value that occupies a 16-byte slot with
;; 16-byte alignment, and the backend softens it to a 16-byte integer image.
;; The frontend converts between that image and the value with `trunc` and
;; `bitcast` pairs, so a bitcast has to produce and consume the whole slot
;; rather than just the 80 value bits; otherwise the image is an i80 and the
;; operations on it no longer agree on their type.
;;
;; The helpers are the ARM64EC ones, which are named apart from the exact x87
;; soft-float helpers of libgcc, and are `#`-prefixed at the object level.

target triple = "arm64ec-pc-windows-msvc"

define x86_fp80 @bitcast_select(i128 %bits, i1 %c) {
entry:
  %trunc = trunc i128 %bits to i80
  %value = bitcast i80 %trunc to x86_fp80
  %neg = fneg x86_fp80 %value
  %sel = select i1 %c, x86_fp80 %value, x86_fp80 %neg
  ret x86_fp80 %sel
}

; CHECK-LABEL: "#bitcast_select":
;; The sign is flipped with an exclusive or of the image, with no helper call.
; CHECK:       eor
; CHECK-NOT:   bl
; CHECK:       ret

define x86_fp80 @bitcast_arithmetic(i128 %a, i128 %b) {
entry:
  %at = trunc i128 %a to i80
  %x = bitcast i80 %at to x86_fp80
  %bt = trunc i128 %b to i80
  %y = bitcast i80 %bt to x86_fp80
  %sum = fadd x86_fp80 %x, %y
  ret x86_fp80 %sum
}

; CHECK-LABEL: "#bitcast_arithmetic":
; CHECK:       bl      "#__arm64ec_addxf3"
