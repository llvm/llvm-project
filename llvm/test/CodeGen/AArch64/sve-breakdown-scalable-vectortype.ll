; RUN: llc -mtriple=aarch64--linux-gnu -mattr=+sve -asm-verbose=0 < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

; Test that scalable vectors that are a multiple of the legal vector size
; can be properly broken down into part vectors.

declare void @bar()

;
; Vectors twice the size
;

define <vscale x 32 x i8> @wide_32i8(i1 %b, <vscale x 16 x i8> %legal, <vscale x 32 x i8> %illegal) nounwind {
; CHECK-LABEL: wide_32i8
; CHECK:      mov     z0.d, z1.d
; CHECK-NEXT: mov     z1.d, z2.d
; CHECK-NEXT: ret
  br i1 %b, label %L1, label %L2
L1:
  call void @bar()
  ret <vscale x 32 x i8> undef
L2:
  ret <vscale x 32 x i8> %illegal
}

define <vscale x 16 x i16> @wide_16i16(i1 %b, <vscale x 16 x i8> %legal, <vscale x 16 x i16> %illegal) nounwind {
; CHECK-LABEL: wide_16i16
; CHECK:      mov     z0.d, z1.d
; CHECK-NEXT: mov     z1.d, z2.d
; CHECK-NEXT: ret
  br i1 %b, label %L1, label %L2
L1:
  call void @bar()
  ret <vscale x 16 x i16> undef
L2:
  ret <vscale x 16 x i16> %illegal
}

define <vscale x 8 x i32> @wide_8i32(i1 %b, <vscale x 16 x i8> %legal, <vscale x 8 x i32> %illegal) nounwind {
; CHECK-LABEL: wide_8i32
; CHECK:      mov     z0.d, z1.d
; CHECK-NEXT: mov     z1.d, z2.d
; CHECK-NEXT: ret
  br i1 %b, label %L1, label %L2
L1:
  call void @bar()
  ret <vscale x 8 x i32> undef
L2:
  ret <vscale x 8 x i32> %illegal
}

define <vscale x 4 x i64> @wide_4i64(i1 %b, <vscale x 16 x i8> %legal, <vscale x 4 x i64> %illegal) nounwind {
; CHECK-LABEL: wide_4i64
; CHECK:      mov     z0.d, z1.d
; CHECK-NEXT: mov     z1.d, z2.d
; CHECK-NEXT: ret
  br i1 %b, label %L1, label %L2
L1:
  call void @bar()
  ret <vscale x 4 x i64> undef
L2:
  ret <vscale x 4 x i64> %illegal
}

define <vscale x 16 x half> @wide_16f16(i1 %b, <vscale x 16 x i8> %legal, <vscale x 16 x half> %illegal) nounwind {
; CHECK-LABEL: wide_16f16
; CHECK:      mov     z0.d, z1.d
; CHECK-NEXT: mov     z1.d, z2.d
; CHECK-NEXT: ret
  br i1 %b, label %L1, label %L2
L1:
  call void @bar()
  ret <vscale x 16 x half> undef
L2:
  ret <vscale x 16 x half> %illegal
}

define <vscale x 8 x float> @wide_8f32(i1 %b, <vscale x 16 x i8> %legal, <vscale x 8 x float> %illegal) nounwind {
; CHECK-LABEL: wide_8f32
; CHECK:      mov     z0.d, z1.d
; CHECK-NEXT: mov     z1.d, z2.d
; CHECK-NEXT: ret
  br i1 %b, label %L1, label %L2
L1:
  call void @bar()
  ret <vscale x 8 x float> undef
L2:
  ret <vscale x 8 x float> %illegal
}

define <vscale x 4 x double> @wide_4f64(i1 %b, <vscale x 16 x i8> %legal, <vscale x 4 x double> %illegal) nounwind {
; CHECK-LABEL: wide_4f64
; CHECK:      mov     z0.d, z1.d
; CHECK-NEXT: mov     z1.d, z2.d
; CHECK-NEXT: ret
  br i1 %b, label %L1, label %L2
L1:
  call void @bar()
  ret <vscale x 4 x double> undef
L2:
  ret <vscale x 4 x double> %illegal
}

;
; Vectors three times the size
;

define <vscale x 48 x i8> @wide_48i8(i1 %b, <vscale x 16 x i8> %legal, <vscale x 48 x i8> %illegal) nounwind {
; CHECK-LABEL: wide_48i8
; CHECK:      mov     z0.d, z1.d
; CHECK-NEXT: mov     z1.d, z2.d
; CHECK-NEXT: mov     z2.d, z3.d
; CHECK-NEXT: ret
  br i1 %b, label %L1, label %L2
L1:
  call void @bar()
  ret <vscale x 48 x i8> undef
L2:
  ret <vscale x 48 x i8> %illegal
}

define <vscale x 24 x i16> @wide_24i16(i1 %b, <vscale x 16 x i8> %legal, <vscale x 24 x i16> %illegal) nounwind {
; CHECK-LABEL: wide_24i16
; CHECK:      mov     z0.d, z1.d
; CHECK-NEXT: mov     z1.d, z2.d
; CHECK-NEXT: mov     z2.d, z3.d
; CHECK-NEXT: ret
  br i1 %b, label %L1, label %L2
L1:
  call void @bar()
  ret <vscale x 24 x i16> undef
L2:
  ret <vscale x 24 x i16> %illegal
}

define <vscale x 12 x i32> @wide_12i32(i1 %b, <vscale x 16 x i8> %legal, <vscale x 12 x i32> %illegal) nounwind {
; CHECK-LABEL: wide_12i32
; CHECK:      mov     z0.d, z1.d
; CHECK-NEXT: mov     z1.d, z2.d
; CHECK-NEXT: mov     z2.d, z3.d
; CHECK-NEXT: ret
  br i1 %b, label %L1, label %L2
L1:
  call void @bar()
  ret <vscale x 12 x i32> undef
L2:
  ret <vscale x 12 x i32> %illegal
}

define <vscale x 6 x i64> @wide_6i64(i1 %b, <vscale x 16 x i8> %legal, <vscale x 6 x i64> %illegal) nounwind {
; CHECK-LABEL: wide_6i64
; CHECK:      mov     z0.d, z1.d
; CHECK-NEXT: mov     z1.d, z2.d
; CHECK-NEXT: mov     z2.d, z3.d
; CHECK-NEXT: ret
  br i1 %b, label %L1, label %L2
L1:
  call void @bar()
  ret <vscale x 6 x i64> undef
L2:
  ret <vscale x 6 x i64> %illegal
}

define <vscale x 24 x half> @wide_24f16(i1 %b, <vscale x 16 x i8> %legal, <vscale x 24 x half> %illegal) nounwind {
; CHECK-LABEL: wide_24f16
; CHECK:      mov     z0.d, z1.d
; CHECK-NEXT: mov     z1.d, z2.d
; CHECK-NEXT: mov     z2.d, z3.d
; CHECK-NEXT: ret
  br i1 %b, label %L1, label %L2
L1:
  call void @bar()
  ret <vscale x 24 x half> undef
L2:
  ret <vscale x 24 x half> %illegal
}

define <vscale x 12 x float> @wide_12f32(i1 %b, <vscale x 16 x i8> %legal, <vscale x 12 x float> %illegal) nounwind {
; CHECK-LABEL: wide_12f32
; CHECK:      mov     z0.d, z1.d
; CHECK-NEXT: mov     z1.d, z2.d
; CHECK-NEXT: mov     z2.d, z3.d
; CHECK-NEXT: ret
  br i1 %b, label %L1, label %L2
L1:
  call void @bar()
  ret <vscale x 12 x float> undef
L2:
  ret <vscale x 12 x float> %illegal
}

define <vscale x 6 x double> @wide_6f64(i1 %b, <vscale x 16 x i8> %legal, <vscale x 6 x double> %illegal) nounwind {
; CHECK-LABEL: wide_6f64
; CHECK:      mov     z0.d, z1.d
; CHECK-NEXT: mov     z1.d, z2.d
; CHECK-NEXT: mov     z2.d, z3.d
; CHECK-NEXT: ret
  br i1 %b, label %L1, label %L2
L1:
  call void @bar()
  ret <vscale x 6 x double> undef
L2:
  ret <vscale x 6 x double> %illegal
}

;
; Vectors four times the size
;

define <vscale x 64 x i8> @wide_64i8(i1 %b, <vscale x 16 x i8> %legal, <vscale x 64 x i8> %illegal) nounwind {
; CHECK-LABEL: wide_64i8
; CHECK:      mov     z0.d, z1.d
; CHECK-NEXT: mov     z1.d, z2.d
; CHECK-NEXT: mov     z2.d, z3.d
; CHECK-NEXT: mov     z3.d, z4.d
; CHECK-NEXT: ret
  br i1 %b, label %L1, label %L2
L1:
  call void @bar()
  ret <vscale x 64 x i8> undef
L2:
  ret <vscale x 64 x i8> %illegal
}

define <vscale x 32 x i16> @wide_32i16(i1 %b, <vscale x 16 x i8> %legal, <vscale x 32 x i16> %illegal) nounwind {
; CHECK-LABEL: wide_32i16
; CHECK:      mov     z0.d, z1.d
; CHECK-NEXT: mov     z1.d, z2.d
; CHECK-NEXT: mov     z2.d, z3.d
; CHECK-NEXT: mov     z3.d, z4.d
; CHECK-NEXT: ret
  br i1 %b, label %L1, label %L2
L1:
  call void @bar()
  ret <vscale x 32 x i16> undef
L2:
  ret <vscale x 32 x i16> %illegal
}

define <vscale x 16 x i32> @wide_16i32(i1 %b, <vscale x 16 x i8> %legal, <vscale x 16 x i32> %illegal) nounwind {
; CHECK-LABEL: wide_16i32
; CHECK:      mov     z0.d, z1.d
; CHECK-NEXT: mov     z1.d, z2.d
; CHECK-NEXT: mov     z2.d, z3.d
; CHECK-NEXT: mov     z3.d, z4.d
; CHECK-NEXT: ret
  br i1 %b, label %L1, label %L2
L1:
  call void @bar()
  ret <vscale x 16 x i32> undef
L2:
  ret <vscale x 16 x i32> %illegal
}

define <vscale x 8 x i64> @wide_8i64(i1 %b, <vscale x 16 x i8> %legal, <vscale x 8 x i64> %illegal) nounwind {
; CHECK-LABEL: wide_8i64
; CHECK:      mov     z0.d, z1.d
; CHECK-NEXT: mov     z1.d, z2.d
; CHECK-NEXT: mov     z2.d, z3.d
; CHECK-NEXT: mov     z3.d, z4.d
; CHECK-NEXT: ret
  br i1 %b, label %L1, label %L2
L1:
  call void @bar()
  ret <vscale x 8 x i64> undef
L2:
  ret <vscale x 8 x i64> %illegal
}

define <vscale x 32 x half> @wide_32f16(i1 %b, <vscale x 16 x i8> %legal, <vscale x 32 x half> %illegal) nounwind {
; CHECK-LABEL: wide_32f16
; CHECK:      mov     z0.d, z1.d
; CHECK-NEXT: mov     z1.d, z2.d
; CHECK-NEXT: mov     z2.d, z3.d
; CHECK-NEXT: mov     z3.d, z4.d
; CHECK-NEXT: ret
  br i1 %b, label %L1, label %L2
L1:
  call void @bar()
  ret <vscale x 32 x half> undef
L2:
  ret <vscale x 32 x half> %illegal
}

define <vscale x 16 x float> @wide_16f32(i1 %b, <vscale x 16 x i8> %legal, <vscale x 16 x float> %illegal) nounwind {
; CHECK-LABEL: wide_16f32
; CHECK:      mov     z0.d, z1.d
; CHECK-NEXT: mov     z1.d, z2.d
; CHECK-NEXT: mov     z2.d, z3.d
; CHECK-NEXT: mov     z3.d, z4.d
; CHECK-NEXT: ret
  br i1 %b, label %L1, label %L2
L1:
  call void @bar()
  ret <vscale x 16 x float> undef
L2:
  ret <vscale x 16 x float> %illegal
}

define <vscale x 8 x double> @wide_8f64(i1 %b, <vscale x 16 x i8> %legal, <vscale x 8 x double> %illegal) nounwind {
; CHECK-LABEL: wide_8f64
; CHECK:      mov     z0.d, z1.d
; CHECK-NEXT: mov     z1.d, z2.d
; CHECK-NEXT: mov     z2.d, z3.d
; CHECK-NEXT: mov     z3.d, z4.d
; CHECK-NEXT: ret
  br i1 %b, label %L1, label %L2
L1:
  call void @bar()
  ret <vscale x 8 x double> undef
L2:
  ret <vscale x 8 x double> %illegal
}
