; RUN: llc -march=hexagon < %s | FileCheck %s

; Check that a store with a proper addressing mode is selected for various
; cases of storing an immediate value.


@var_i8 = global [10 x i8] zeroinitializer, align 8

; CHECK-LABEL: store_imm_i8:
; CHECK: memb(r0+#0) = #-1
define void @store_imm_i8(i8* %p) nounwind {
  store i8 255, i8* %p, align 4
  ret void
}

; CHECK-LABEL: store_rr_i8:
; CHECK: [[RV:r[0-9]+]] = #255
; CHECK: memb(r0+r1<<#0) = [[RV]]
define void @store_rr_i8(i8* %p, i32 %x) nounwind {
  %t0 = getelementptr i8, i8* %p, i32 %x
  store i8 255, i8* %t0, align 4
  ret void
}

; CHECK-LABEL: store_io_i8:
; CHECK: [[RV:r[0-9]+]] = #255
; CHECK: memb(r0+##var_i8) = [[RV]]
define void @store_io_i8(i32 %x) nounwind {
  %t0 = getelementptr [10 x i8], [10 x i8]* @var_i8, i32 0, i32 %x
  store i8 255, i8* %t0, align 4
  ret void
}

; CHECK-LABEL: store_ur_i8:
; CHECK: [[RV:r[0-9]+]] = #255
; CHECK: memb(r0<<#2+##var_i8) = [[RV]]
define void @store_ur_i8(i32 %x) nounwind {
  %t0 = shl i32 %x, 2
  %t1 = getelementptr [10 x i8], [10 x i8]* @var_i8, i32 0, i32 %t0
  store i8 255, i8* %t1, align 4
  ret void
}

@var_i16 = global [10 x i16] zeroinitializer, align 8

; CHECK-LABEL: store_imm_i16:
; CHECK: memh(r0+#0) = #-1
define void @store_imm_i16(i16* %p) nounwind {
  store i16 65535, i16* %p, align 4
  ret void
}

; CHECK-LABEL: store_rr_i16:
; CHECK: [[RV:r[0-9]+]] = ##65535
; CHECK: memh(r0+r1<<#1) = [[RV]]
define void @store_rr_i16(i16* %p, i32 %x) nounwind {
  %t0 = getelementptr i16, i16* %p, i32 %x
  store i16 65535, i16* %t0, align 4
  ret void
}

; CHECK-LABEL: store_ur_i16:
; CHECK: [[RV:r[0-9]+]] = ##65535
; CHECK: memh(r0<<#1+##var_i16) = [[RV]]
define void @store_ur_i16(i32 %x) nounwind {
  %t0 = getelementptr [10 x i16], [10 x i16]* @var_i16, i32 0, i32 %x
  store i16 65535, i16* %t0, align 4
  ret void
}

@var_i32 = global [10 x i32] zeroinitializer, align 8

; CHECK-LABEL: store_imm_i32:
; CHECK: memw(r0+#0) = #-1
define void @store_imm_i32(i32* %p) nounwind {
  store i32 4294967295, i32* %p, align 4
  ret void
}

; CHECK-LABEL: store_rr_i32:
; CHECK: [[RV:r[0-9]+]] = #-1
; CHECK: memw(r0+r1<<#2) = [[RV]]
define void @store_rr_i32(i32* %p, i32 %x) nounwind {
  %t0 = getelementptr i32, i32* %p, i32 %x
  store i32 4294967295, i32* %t0, align 4
  ret void
}

; CHECK-LABEL: store_ur_i32:
; CHECK: [[RV:r[0-9]+]] = #-1
; CHECK: memw(r0<<#2+##var_i32) = [[RV]]
define void @store_ur_i32(i32 %x) nounwind {
  %t0 = getelementptr [10 x i32], [10 x i32]* @var_i32, i32 0, i32 %x
  store i32 4294967295, i32* %t0, align 4
  ret void
}

