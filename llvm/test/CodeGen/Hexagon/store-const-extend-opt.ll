; RUN: llc -march=hexagon -O3 -hexagon-small-data-threshold=0 < %s | FileCheck %s
; This test checks the case if there are more than 2 uses of a constan address, move the
; value in to a register and replace all instances of constant with the register.
; The GenMemAbsolute pass generates a absolute-set instruction if there are more
; than 2 uses of this register.

; CHECK: storetrunci32_3
; CHECK-NOT: memw(##441652) = r{{[0-9]+}}
; CHECK-NOT: memw(r{{[0-9]+}}+#0) = r{{[0-9]+}}
; CHECK:memw(r[[REG:[0-9]+]]=##441652) = r{{[0-9]+}}
; CHECK-NOT: memw(##441652) = r{{[0-9]+}}
; CHECK:memw(r[[REG]]+#0) = r{{[0-9]+}}
; CHECK-NOT: memw(##441652) = r{{[0-9]+}}
; CHECK:memw(r[[REG]]+#0) = r{{[0-9]+}}
; CHECK-NOT: memw(##441652) = r{{[0-9]+}}

define void @storetrunci32_3(i64 %descr_addr, i32 %rpm_or_sys, i32 %kkr) #0 {
entry:
  %conv = trunc i64 %descr_addr to i32
  store volatile i32 %conv, ptr inttoptr (i32 441652 to ptr), align 4
  store volatile i32 %rpm_or_sys, ptr inttoptr (i32 441652 to ptr), align 4
  store volatile i32 %kkr, ptr inttoptr (i32 441652 to ptr), align 4
  ret void
}

; CHECK: storetrunci32_2
; CHECK-NOT: r{{[0-9]+}} = ##441652
; CHECK: memw(##441652) = r{{[0-9]+}}
; CHECK: memw(##441652) = r{{[0-9]+}}

define void @storetrunci32_2(i64 %descr_addr, i32 %rpm_or_sys) #0 {
entry:
  %conv = trunc i64 %descr_addr to i32
  store volatile i32 %conv, ptr inttoptr (i32 441652 to ptr), align 4
  store volatile i32 %rpm_or_sys, ptr inttoptr (i32 441652 to ptr), align 4
  ret void
}

; CHECK: storetrunci32_abs_global_3
; CHECK-NOT: memw(##globalInt) = r{{[0-9]+}}
; CHECK-NOT: memw(r{{[0-9]+}}+#0) = r{{[0-9]+}}
; CHECK:memw(r[[REG:[0-9]+]]=##globalInt) = r{{[0-9]+}}
; CHECK-NOT: memw(##globalInt) = r{{[0-9]+}}
; CHECK:memw(r[[REG]]+#0) = r{{[0-9]+}}
; CHECK-NOT: memw(##globalInt) = r{{[0-9]+}}
; CHECK:memw(r[[REG]]+#0) = r{{[0-9]+}}
; CHECK-NOT: memw(##globalInt) = r{{[0-9]+}}

@globalInt = external global i32, align 8
define void @storetrunci32_abs_global_3(i64 %descr_addr, i32 %rpm_or_sys, i32 %kkr) #0 {
entry:
  %conv = trunc i64 %descr_addr to i32
  store volatile i32 %conv, ptr @globalInt, align 4
  store volatile i32 %rpm_or_sys, ptr @globalInt, align 4
  store volatile i32 %kkr, ptr @globalInt, align 4
  ret void
}

; CHECK: storetrunci32_abs_global_2
; CHECK-NOT:r[[REG:[0-9]+]] = ##globalInt
; CHECK:memw(##globalInt) = r{{[0-9]+}}
; CHECK:memw(##globalInt) = r{{[0-9]+}}

define void @storetrunci32_abs_global_2(i64 %descr_addr, i32 %rpm_or_sys) #0 {
entry:
  %conv = trunc i64 %descr_addr to i32
  store volatile i32 %conv, ptr @globalInt, align 4
  store volatile i32 %rpm_or_sys, ptr @globalInt, align 4
  ret void
}

attributes #0 = { nounwind }
