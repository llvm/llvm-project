; RUN: opt -mergefunc -S < %s | FileCheck %s

;; Make sure that two different allocas are not treated as equal.

target datalayout = "e-m:w-p:32:32-i64:64-f80:32-n8:16:32-S32"

%kv1 = type { i32, i32 }
%kv2 = type { i8 }
%kv3 = type { i64, i64 }

; Size difference.

; CHECK-LABEL: define void @size1
; CHECK-NOT: call void @
define void @size1(i8 *%f) {
  %v = alloca %kv1, align 8
  %f_2 = bitcast i8* %f to void (%kv1 *)*
  call void %f_2(%kv1 * %v)
  call void %f_2(%kv1 * %v)
  call void %f_2(%kv1 * %v)
  call void %f_2(%kv1 * %v)
  ret void
}

; CHECK-LABEL: define void @size2
; CHECK-NOT: call void @
define void @size2(i8 *%f) {
  %v = alloca %kv2, align 8
  %f_2 = bitcast i8* %f to void (%kv2 *)*
  call void %f_2(%kv2 * %v)
  call void %f_2(%kv2 * %v)
  call void %f_2(%kv2 * %v)
  call void %f_2(%kv2 * %v)
  ret void
}

; Alignment difference.

; CHECK-LABEL: define void @align1
; CHECK-NOT: call void @
define void @align1(i8 *%f) {
  %v = alloca %kv3, align 8
  %f_2 = bitcast i8* %f to void (%kv3 *)*
  call void %f_2(%kv3 * %v)
  call void %f_2(%kv3 * %v)
  call void %f_2(%kv3 * %v)
  call void %f_2(%kv3 * %v)
  ret void
}

; CHECK-LABEL: define void @align2
; CHECK-NOT: call void @
define void @align2(i8 *%f) {
  %v = alloca %kv3, align 16
  %f_2 = bitcast i8* %f to void (%kv3 *)*
  call void %f_2(%kv3 * %v)
  call void %f_2(%kv3 * %v)
  call void %f_2(%kv3 * %v)
  call void %f_2(%kv3 * %v)
  ret void
}
