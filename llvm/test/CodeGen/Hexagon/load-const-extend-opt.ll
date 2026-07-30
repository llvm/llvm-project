; RUN: llc -mtriple=hexagon -O3 -hexagon-small-data-threshold=0 < %s | FileCheck %s
; This test checks the case if there are more than 2 uses of a constan address, move the
; value in to a register and replace all instances of constant with the register.
; The GenMemAbsolute pass generates a absolute-set instruction if there are more
; than 2 uses of this register.

; CHECK: loadi32_3
; CHECK-NOT: r{{[0-9]+}} = memw(##441652)
; CHECK-NOT: r{{[0-9]+}} = memw(r{{[0-9]+}}+#0)
; CHECK:r{{[0-9]+}} = memw(r[[REG:[0-9]+]]=##441652)
; CHECK-NOT: r{{[0-9]+}} = {emw(##441652)
; CHECK:r{{[0-9]+}} = memw(r[[REG]]+#0)
; CHECK-NOT: r{{[0-9]+}} = memw(##441652)
; CHECK:r{{[0-9]+}} = memw(r[[REG]]+#0)
; CHECK-NOT: r{{[0-9]+}} = memw(##441652)

define void @loadi32_3() #0 {
entry:
  %0 = load volatile i32, ptr inttoptr (i32 441652 to ptr), align 4
  %1 = load volatile i32, ptr inttoptr (i32 441652 to ptr), align 4
  %2 = load volatile i32, ptr inttoptr (i32 441652 to ptr), align 4
  ret void
}

; CHECK: loadi32_2
; CHECK-NOT: r{{[0-9]+}} = ##441652
; CHECK: r{{[0-9]+}} = memw(##441652)
; CHECK: r{{[0-9]+}} = memw(##441652)

define void @loadi32_2() #0 {
entry:
  %0 = load volatile i32, ptr inttoptr (i32 441652 to ptr), align 4
  %1 = load volatile i32, ptr inttoptr (i32 441652 to ptr), align 4
  ret void
}

; CHECK: loadi32_abs_global_3
; CHECK-NOT: r{{[0-9]+}} = memw(##globalInt)
; CHECK-NOT: r{{[0-9]+}} = memw(r{{[0-9]+}}+#0)
; CHECK:r{{[0-9]+}} = memw(r[[REG:[0-9]+]]=##globalInt)
; CHECK-NOT: r{{[0-9]+}} = memw(##globalInt)
; CHECK:r{{[0-9]+}} = memw(r[[REG]]+#0)
; CHECK-NOT: r{{[0-9]+}} = memw(##globalInt)
; CHECK:r{{[0-9]+}} = memw(r[[REG]]+#0)
; CHECK-NOT: r{{[0-9]+}} = memw(##globalInt)

@globalInt = external global i32, align 8
define void @loadi32_abs_global_3() #0 {
entry:
  %0 = load volatile i32, ptr @globalInt, align 4
  %1 = load volatile i32, ptr @globalInt, align 4
  %2 = load volatile i32, ptr @globalInt, align 4
  ret void
}

; CHECK: loadi32_abs_global_2
; CHECK-NOT:r[[REG:[0-9]+]] = ##globalInt
; CHECK:r{{[0-9]+}} = memw(##globalInt)
; CHECK:r{{[0-9]+}} = memw(##globalInt)

define void @loadi32_abs_global_2() #0 {
entry:
  %0 = load volatile i32, ptr @globalInt, align 4
  %1 = load volatile i32, ptr @globalInt, align 4
  ret void
}

attributes #0 = { nounwind }
