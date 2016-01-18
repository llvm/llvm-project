; RUN: llc < %s -mtriple=thumbv7m-none-eabi -mcpu=cortex-m4 | FileCheck %s

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
; CHECK: ldr
define i32 @f3(i32* %p) nounwind readonly {
entry:
  %0 = load atomic i32, i32* %p seq_cst, align 4
  ret i32 %0
}

; CHECK-LABEL: f4:
; CHECK: ldrb
define i8 @f4(i8* %p) nounwind readonly {
entry:
  %0 = load atomic i8, i8* %p seq_cst, align 4
  ret i8 %0
}

; CHECK-LABEL: f5:
; CHECK: str
define void @f5(i32* %p) nounwind readonly {
entry:
  store atomic i32 0, i32* %p seq_cst, align 4
  ret void
}

; CHECK-LABEL: f6:
; CHECK: ldrex
; CHECK: strex
define i32 @f6(i32* %p) nounwind readonly {
entry:
  %0 = atomicrmw add i32* %p, i32 1 seq_cst
  ret i32 %0
}
