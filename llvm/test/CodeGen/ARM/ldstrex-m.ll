; RUN: llc < %s -mtriple=thumbv7m-none-eabi -mcpu=cortex-m4 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-V7
; RUN: llc < %s -mtriple=thumbv8m.main-none-eabi | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-V8
; RUN: llc < %s -mtriple=thumbv8m.base-none-eabi | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-V8

; CHECK-LABEL: f0:
; CHECK-NOT: ldrexd
define i64 @f0(ptr %p) nounwind readonly {
entry:
  %0 = load atomic i64, ptr %p seq_cst, align 8
  ret i64 %0
}

; CHECK-LABEL: f1:
; CHECK-NOT: strexd
define void @f1(ptr %p) nounwind readonly {
entry:
  store atomic i64 0, ptr %p seq_cst, align 8
  ret void
}

; CHECK-LABEL: f2:
; CHECK-NOT: ldrexd
; CHECK-NOT: strexd
define i64 @f2(ptr %p) nounwind readonly {
entry:
  %0 = atomicrmw add ptr %p, i64 1 seq_cst
  ret i64 %0
}

; CHECK-LABEL: f3:
; CHECK-V7: ldr
; CHECK-V8: lda
define i32 @f3(ptr %p) nounwind readonly {
entry:
  %0 = load atomic i32, ptr %p seq_cst, align 4
  ret i32 %0
}

; CHECK-LABEL: f4:
; CHECK-V7: ldrb
; CHECK-V8: ldab
define i8 @f4(ptr %p) nounwind readonly {
entry:
  %0 = load atomic i8, ptr %p seq_cst, align 4
  ret i8 %0
}

; CHECK-LABEL: f5:
; CHECK-V7: str
; CHECK-V8: stl
define void @f5(ptr %p) nounwind readonly {
entry:
  store atomic i32 0, ptr %p seq_cst, align 4
  ret void
}

; CHECK-LABEL: f6:
; CHECK-V7: ldrex
; CHECK-V7: strex
; CHECK-V8: ldaex
; CHECK-V8: stlex
define i32 @f6(ptr %p) nounwind readonly {
entry:
  %0 = atomicrmw add ptr %p, i32 1 seq_cst
  ret i32 %0
}
