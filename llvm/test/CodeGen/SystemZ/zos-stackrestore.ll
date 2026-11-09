; RUN: llc < %s -mtriple=s390x-ibm-zos | FileCheck %s
; RUN: llc < %s -mtriple=s390x-ibm-zos -filetype=obj -o /dev/null
;
; @@ALCAXP moves the frame header (save area at SP + 2048) down with the stack
; pointer. A stack restore must copy the current header to the restored stack
; pointer, and no signal may be delivered meanwhile. This is done in a stub
; routine (stub entry marker, no PPA1), which is emitted once per module at the
; start of the code section and gets the new stack pointer in R1.

declare void @use(ptr)

; The stub copies the header from the highest word down and sets R4.
; CHECK:      * XPLINK Stub Entry Padding
; CHECK-NEXT:  DC XL8'0000000000000000'
; CHECK-NEXT: * XPLINK Stub Entry Marker
; CHECK-NEXT: * Eyecatcher 0x00C300C500C500
; CHECK-NEXT:  DC XL7'00C300C500C500'
; CHECK-NEXT: * Mark Type C'4'
; CHECK-NEXT:  DC XL1'F4'
; CHECK-NEXT: [[STUB:L#stackrestore[0-9]+]] DS 0H
; CHECK-NEXT:  lg 0,2168(4)
; CHECK-NEXT:  stg 0,2168(1)
; CHECK-NEXT:  lg 0,2160(4)
; CHECK-NEXT:  stg 0,2160(1)
; CHECK:       lg 0,2056(4)
; CHECK-NEXT:  stg 0,2056(1)
; CHECK-NEXT:  lg 0,2048(4)
; CHECK-NEXT:  stg 0,2048(1)
; CHECK-NEXT:  lgr 4,1
; CHECK-NEXT:  b 2(7)
; CHECK-NOT:  Stub Entry Marker

; The stack restore calls the stub with the saved stack pointer in R1.
; CHECK-LABEL: f DS 0H
; CHECK:       nill 1,65504
; CHECK-NEXT:  lgr [[SP:[0-9]+]],4
; CHECK:       basr 7,6
; CHECK:       la 1,2240(4)
; CHECK:       basr 7,6
; CHECK:       lgr 1,[[SP]]
; CHECK-NEXT:  larl 6,[[STUB]]
; CHECK-NEXT:  basr 7,6
; CHECK-NEXT:  bcr 0,0
; CHECK:       basr 7,6
; CHECK:       lmg 4,11,2048(4)
define void @f(i64 %n) {
  %sp = call ptr @llvm.stacksave.p0()
  %p = alloca i8, i64 %n, align 8
  call void @use(ptr %p)
  call void @llvm.stackrestore.p0(ptr %sp)
  call void @use(ptr null)
  ret void
}

; A second stack restore uses the same stub.
; CHECK-LABEL: g DS 0H
; CHECK:       larl 6,[[STUB]]
define void @g(i64 %n) {
  %sp = call ptr @llvm.stacksave.p0()
  %p = alloca i8, i64 %n, align 8
  call void @use(ptr %p)
  call void @llvm.stackrestore.p0(ptr %sp)
  ret void
}
