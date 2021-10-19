;; RUN: llc -mtriple aarch64               -global-isel -O0 %s -o - | FileCheck -enable-var-scope %s --check-prefixes=CHECK,CHECK-NOP
;; RUN: llc -mtriple aarch64 -mattr=+v8.3a -global-isel -O0 %s -o - | FileCheck -enable-var-scope %s --check-prefixes=CHECK,CHECK-V83
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
;; CHECK-V83:      mov [[COPY_X30:x[0-9]+]], x30
;; CHECK-V83:      xpaci [[COPY_X30]]
;; CHECK:          bl g1
;; CHECK:          ldr x[[T0:[0-9]+]], [x29]
;; CHECK-NOP-NEXT: ldr x30, [x[[T0]], #8]
;; CHECK-NOP-NEXT: hint #7
;; CHECK-V83-NEXT: ldr x[[LD0:[0-9]+]], [x[[T0]], #8]
;; CHECK-V83-NEXT: xpaci x[[LD0]]
;; CHECK:          bl g2
;; CHECK:          ldr x[[T1:[0-9]+]], [x29]
;; CHECK-NEXT:     ldr x[[T1]], [x[[T1]]]
;; CHECK-NOP-NEXT: ldr x30, [x[[T1]], #8]
;; CHECK-NOP-NEXT: hint #7
;; CHECK-NOP-NEXT: mov x0, x30
;; CHECK-V83-NEXT: ldr x0, [x[[T1]], #8]
;; CHECK-V83-NEXT: xpaci x0

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
;; CHECK-V83-DAG:  str x30, [sp, #[[OFF:[0-9]+]]
;; CHECK-V83:      ldr x[[T1:[0-9]+]], [x[[T0]], #8]
;; CHECK-V83-NEXT: xpaci x[[T1]]

;; CHECK:          bl g1
;; CHECK:          ldr x[[T2:[0-9]+]], [x29]
;; CHECK-NEXT:     ldr x[[T2]], [x[[T2]]]
;; CHECK-NOP-NEXT: ldr x30, [x[[T2]], #8]
;; CHECK-NOP-NEXT: hint #7
;; CHECK-V83-NEXT: ldr x[[T3:[0-9]+]], [x[[T2]], #8]
;; CHECK-V83-NEXT: xpaci x[[T3]]
;; CHECK:          bl g2

;; CHECK-NOP:      ldr x30, [sp, #[[OFF]]]
;; CHECK-NOP-NEXT: hint #7
;; CHECK-NOP-NEXT: mov x0, x30

;; CHECK-V83:      ldr x0, [sp, #[[OFF]]]
;; CHECK-V83-NEXT: xpaci x0
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
;; CHECK-NOP:      ldr x30, [sp,
;; CHECK-NOP-NEXT: hint #7
;; CHECK-NOP-NEXT: mov x0, x30

;; CHECK-V83:      ldr x0, [sp,
;; CHECK-V83-NEXT: xpaci x0
;; CHECK-NOT:      x0
;; CHECK:          ret

define i8* @f3() #0 {
entry:
  %0 = call i8* @llvm.returnaddress(i32 0)
  ret i8* %0
}
;; CHECK-LABEL:    f3:
;; CHECK-NOP:      str x30, [sp,
;; CHECK-NOP-NEXT: hint #7
;; CHECK-NOP-NEXT: mov x0, x30

;; CHECK-V83:      mov x0, x30
;; CHECK-V83-NEXT: xpaci x0
;; CHECK-NOT:      x0
;; CHECK:          ret
attributes #0 = { nounwind }
attributes #1 = { nounwind }
attributes #2 = { nounwind readnone }
