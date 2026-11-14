; RUN: llc < %s -mtriple=s390x-ibm-zos | FileCheck %s
; RUN: llc < %s -mtriple=s390x-ibm-zos -filetype=obj -o /dev/null
;
; @@ALCAXP moves the frame header (save area at SP + 2048) down with the stack
; pointer. A stack restore must copy the current header to the restored stack
; pointer, and no signal may be delivered meanwhile. This is done in a stub
; routine (stub entry marker, no PPA1), which is emitted once per module after
; the last function and gets the new stack pointer in R1.

declare void @use(ptr)

; CHECK-NOT:  Stub Entry Marker

; The stack restore calls the stub with the saved stack pointer in R1.
; CHECK-LABEL: f DS 0H
; CHECK:       nill 1,65504
; CHECK-NEXT:  lgr [[SP:[0-9]+]],4
; CHECK:       basr 7,6
; CHECK:       la 1,2240(4)
; CHECK:       basr 7,6
; CHECK:       lgr 1,[[SP]]
; CHECK-NEXT:  larl 3,[[STUB:L#stackrestore[0-9]+]]
; CHECK-NEXT:  basr 3,3
; CHECK-NEXT:  bcr 0,7
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
; CHECK:       larl 3,[[STUB]]
define void @g(i64 %n) {
  %sp = call ptr @llvm.stacksave.p0()
  %p = alloca i8, i64 %n, align 8
  call void @use(ptr %p)
  call void @llvm.stackrestore.p0(ptr %sp)
  ret void
}

; The stub copies the header in blocks of 32 bytes from the highest block down
; and sets R4. It is called with BASR 3,3 and returns with B 2(3).
; CHECK:      * XPLINK Stub Entry Padding
; CHECK-NEXT:  DC XL8'0000000000000000'
; CHECK-NEXT: * XPLINK Stub Entry Marker
; CHECK-NEXT: * Eyecatcher 0x00C300C500C500
; CHECK-NEXT:  DC XL7'00C300C500C500'
; CHECK-NEXT: * Mark Type C'4'
; CHECK-NEXT:  DC XL1'F4'
; CHECK-NEXT: [[STUB]] DS 0H
; CHECK-NEXT:  mvc 2144(32,1),2144(4)
; CHECK-NEXT:  mvc 2112(32,1),2112(4)
; CHECK-NEXT:  mvc 2080(32,1),2080(4)
; CHECK-NEXT:  mvc 2048(32,1),2048(4)
; CHECK-NEXT:  lgr 4,1
; CHECK-NEXT:  b 2(3)
; CHECK-NOT:  Stub Entry Marker
