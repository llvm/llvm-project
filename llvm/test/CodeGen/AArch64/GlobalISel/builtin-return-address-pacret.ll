;; RUN: llc -mtriple aarch64               -global-isel -O0 %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-NOP
;; RUN: llc -mtriple aarch64 -mattr=+v8.3a -global-isel -O0 %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-V83
declare void @g0() #1
declare void @g1(i8*) #1
declare void @g2(i32, i8*) #1

declare i8* @llvm.returnaddress(i32 immarg) #2

define i8* @f0() #0 {
entry:
  %0 = call i8* @llvm.returnaddress(i32 0)
  call void @g1(i8* %0)
  %1 = call i8* @llvm.returnaddress(i32 1)
  call void @g2(i32 1, i8* %1)
  %2 = call i8* @llvm.returnaddress(i32 2)
  ret i8* %2
}
;; CHECK-LABEL:    f0:
;; CHECK-NOT:      {{(mov|ldr)}} x30
;; CHECK-NOP:      hint #7
;; CHECK-V83:      xpaci x30
;; CHECK:          bl g1
;; CHECK:          ldr x[[T0:[0-9]+]], [x29]
;; CHECK-NOP-NEXT: ldr x30, [x[[T0]], #8]
;; CHECK-NOP-NEXT: hint #7
;; CHECK-V83-NEXT: ldr x[[T0]], [x[[T0]], #8]
;; CHECK-V83-NEXT: xpaci x[[T0]]
;; CHECK:          bl g2
;; CHECK:          ldr x[[T1:[0-9]+]], [x29]
;; CHECK-NEXT:     ldr x[[T1]], [x[[T1]]]
;; CHECK-NOP-NEXT: ldr x30, [x[[T1]], #8]
;; CHECK-NOP-NEXT: hint #7
;; CHECK-NOP-NEXT: mov x0, x30
;; CHECK-V83-NEXT: ldr x[[T1]], [x[[T1]], #8]
;; CHECK-V83-NEXT: xpaci x[[T1]]
;; CHECK-V83-NEXT: mov x0, x[[T1]]

define i8* @f1() #0 {
entry:
  %0 = call i8* @llvm.returnaddress(i32 1)
  call void @g1(i8* %0)
  %1 = call i8* @llvm.returnaddress(i32 2)
  call void @g2(i32 1, i8* %1)
  %2 = call i8* @llvm.returnaddress(i32 0)
  ret i8* %2
}
;; CHECK-LABEL:    f1:
;; CHECK-DAG:      ldr x[[T0:[0-9]+]], [x29]
;; CHECK-NOP-DAG:  str x30, [sp, #[[OFF:[0-9]+]]
;; CHECK-NOP:      ldr x30, [x[[T0]], #8]
;; CHECK-NOP-NEXT: hint #7
;; CHECK-V83:      ldr x[[T0]], [x[[T0]], #8]
;; CHECK-V83-NEXT: xpaci x[[T0]]
;; CHECK-V83:      str x30, [sp, #[[OFF:[0-9]+]]
;; CHECK:          bl g1
;; CHECK:          ldr x[[T1:[0-9]+]], [x29]
;; CHECK-NEXT:     ldr x[[T1]], [x[[T1]]]
;; CHECK-NOP-NEXT: ldr x30, [x[[T1]], #8]
;; CHECK-NOP-NEXT: hint #7
;; CHECK-V83-NEXT: ldr x[[T1]], [x[[T1]], #8]
;; CHECK-V83-NEXT: xpaci x[[T1]]
;; CHECK:          bl g2
;; CHECK:          ldr x[[T2:[0-9]+]], [sp, #[[OFF]]]
;; CHECK-NOP-NEXT: mov x30, x[[T2]]
;; CHECK-NOP-NEXT: hint #7
;; CHECK-NOP-NEXT: mov x0, x30
;; CHECK-V83-NEXT: xpaci x[[T2]]
;; CHECK-V83-NEXT: mov x0, x[[T2]]
;; CHECK-NOT:      x0
;; CHECK:          ret

define i8* @f2() #0 {
entry:
  call void bitcast (void ()* @g0 to void ()*)()
  %0 = call i8* @llvm.returnaddress(i32 0)
  ret i8* %0
}
;; CHECK-LABEL:    f2
;; CHECK:          bl g0
;; CHECK:          ldr x[[T0:[0-9]+]], [sp,
;; CHECK-NOP-NEXT: mov x30, x[[T2]]
;; CHECK-NOP-NEXT: hint #7
;; CHECK-NOP-NEXT: mov x0, x30
;; CHECK-V83-NEXT: xpaci x[[T2]]
;; CHECK-V83-NEXT: mov x0, x[[T2]]
;; CHECK-NOT:      x0
;; CHECK:          ret

define i8* @f3() #0 {
entry:
  %0 = call i8* @llvm.returnaddress(i32 0)
  ret i8* %0
}
;; CHECK-LABEL:    f3:
;; CHECK:          str x30, [sp,
;; CHECK-NOP-NEXT: hint #7
;; CHECK-V83-NEXT: xpaci x30
;; CHECK-NEXT:     mov x0, x30
;; CHECK-NOT:      x0
;; CHECK:          ret
attributes #0 = { nounwind }
attributes #1 = { nounwind }
attributes #2 = { nounwind readnone }
