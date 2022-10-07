; Test that the scale (-asan-mapping-scale) and offset (-asan-mapping-offset) command-line options work as expected
;
; RUN: opt < %s -passes=asan -asan-mapping-offset 0xdeadbeef -S | FileCheck --check-prefix=CHECK-OFFSET %s
; RUN: opt < %s -passes=asan -asan-mapping-scale 1 -S | FileCheck --check-prefix=CHECK-SCALE %s
; RUN: opt < %s -passes=asan -asan-mapping-offset 0xc0ffee -asan-mapping-scale 0 -S | FileCheck --check-prefix=CHECK-BOTH %s
target triple = "x86_64-unknown-linux-gnu"

define i32 @read_offset(i32* %a) sanitize_address {
entry:
  %tmp1 = load i32, i32* %a, align 4
  ret i32 %tmp1
}
; CHECK-OFFSET-LABEL: @read_offset
; CHECK-OFFSET-NOT: ret
; CHECK-OFFSET: lshr {{.*}} 3
; CHECK-OFFSET-NEXT: add{{.*}}3735928559
; CHECK-OFFSET: ret

define i32 @read_scale(i32* %a) sanitize_address {
entry:
  %tmp1 = load i32, i32* %a, align 4
  ret i32 %tmp1
}
; CHECK-SCALE-LABEL: @read_scale
; CHECK-SCALE-NOT: ret
; CHECK-SCALE: lshr {{.*}} 1
; CHECK-SCALE-NEXT: add{{.*}}
; CHECK-SCALE: ret

define i32 @read_both(i32* %a) sanitize_address {
entry:
  %tmp1 = load i32, i32* %a, align 4
  ret i32 %tmp1
}
; CHECK-BOTH-LABEL: @read_both
; CHECK-BOTH-NOT: ret
; CHECK-BOTH: lshr {{.*}} 0
; CHECK-BOTH-NEXT: add{{.*}}12648430
; CHECK-BOTH: ret
