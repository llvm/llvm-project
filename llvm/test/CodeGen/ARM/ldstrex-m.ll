; RUN: llc < %s -mtriple=thumbv7m-none-eabi -mcpu=cortex-m4 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-V7
; RUN: llc < %s -mtriple=thumbv8m.main-none-eabi | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-V8
; RUN: llc < %s -mtriple=thumbv8m.base-none-eabi | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-V8

; CHECK-LABEL: f0:
; CHECK-NOT: ldrexd
define i64 @f0(i64* %p) nounwind readonly {
entry:
  %0 = load atomic i64, i64* %p seq_cst, align 8
  ret i64 %0
}

; CHECK-LABEL: f1:
; CHECK-NOT: strexd
define void @f1(i64* %p) nounwind readonly {
entry:
  store atomic i64 0, i64* %p seq_cst, align 8
  ret void
}

; CHECK-LABEL: f2:
; CHECK-NOT: ldrexd
; CHECK-NOT: strexd
define i64 @f2(i64* %p) nounwind readonly {
entry:
  %0 = atomicrmw add i64* %p, i64 1 seq_cst
  ret i64 %0
}

; CHECK-LABEL: f3:
; CHECK-V7: ldr
; CHECK-V8: lda
define i32 @f3(i32* %p) nounwind readonly {
entry:
  %0 = load atomic i32, i32* %p seq_cst, align 4
  ret i32 %0
}

; CHECK-LABEL: f4:
; CHECK-V7: ldrb
; CHECK-V8: ldab
define i8 @f4(i8* %p) nounwind readonly {
entry:
  %0 = load atomic i8, i8* %p seq_cst, align 4
  ret i8 %0
}

; CHECK-LABEL: f5:
; CHECK-V7: str
; CHECK-V8: stl
define void @f5(i32* %p) nounwind readonly {
entry:
  store atomic i32 0, i32* %p seq_cst, align 4
  ret void
}

; CHECK-LABEL: f6:
; CHECK-V7: ldrex
; CHECK-V7: strex
; CHECK-V8: ldaex
; CHECK-V8: stlex
define i32 @f6(i32* %p) nounwind readonly {
entry:
  %0 = atomicrmw add i32* %p, i32 1 seq_cst
  ret i32 %0
}
