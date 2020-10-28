; Test that the scale (-heapprof-mapping-scale) and granularity (-heapprof-mapping-granularity) command-line options work as expected
;
; RUN: opt < %s -heapprof -heapprof-module -heapprof-mapping-granularity 32 -S | FileCheck --check-prefix=CHECK-GRAN %s
; RUN: opt < %s -heapprof -heapprof-module -heapprof-mapping-scale 1 -S | FileCheck --check-prefix=CHECK-SCALE %s
; RUN: opt < %s -heapprof -heapprof-module -heapprof-mapping-granularity 16 -heapprof-mapping-scale 0 -S | FileCheck --check-prefix=CHECK-BOTH %s
target triple = "x86_64-unknown-linux-gnu"

define i32 @read(i32* %a) {
entry:
  %tmp1 = load i32, i32* %a, align 4
  ret i32 %tmp1
}
; CHECK-GRAN-LABEL: @read
; CHECK-GRAN-NOT:     ret
; CHECK-GRAN:         and {{.*}} -32
; CHECK-GRAN-NEXT:    lshr {{.*}} 3
; CHECK-GRAN:         ret

; CHECK-SCALE-LABEL: @read
; CHECK-SCALE-NOT:     ret
; CHECK-SCALE:         and {{.*}} -64
; CHECK-SCALE-NEXT:    lshr {{.*}} 1
; CHECK-SCALE:         ret

; CHECK-BOTH-LABEL: @read
; CHECK-BOTH-NOT:     ret
; CHECK-BOTH:         and {{.*}} -16
; CHECK-BOTH-NEXT:    lshr {{.*}} 0
; CHECK-BOTH:         ret
