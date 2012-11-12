; RUN: llc < %s -mcpu=generic -march=x86 | FileCheck %s

;; Simple case
define i32 @test1(i8 %x) nounwind readnone {
  %A = and i8 %x, -32
  %B = zext i8 %A to i32
  ret i32 %B
}
; CHECK: test1
; CHECK: movzbl
; CHECK-NEXT: andl {{.*}}224

;; Multiple uses of %x but easily extensible. 
define i32 @test2(i8 %x) nounwind readnone {
  %A = and i8 %x, -32
  %B = zext i8 %A to i32
  %C = or i8 %x, 63
  %D = zext i8 %C to i32
  %E = add i32 %B, %D
  ret i32 %E
}
; CHECK: test2
; CHECK: movzbl
; CHECK: orl $63
; CHECK: andl $224

declare void @use(i32, i8)

;; Multiple uses of %x where we shouldn't extend the load.
define void @test3(i8 %x) nounwind readnone {
  %A = and i8 %x, -32
  %B = zext i8 %A to i32
  call void @use(i32 %B, i8 %x)
  ret void
}
; CHECK: test3
; CHECK: movzbl {{[0-9]+}}(%esp), [[REGISTER:%e[a-z]{2}]]
; CHECK-NEXT: movl [[REGISTER]], 4(%esp)
; CHECK-NEXT: andl $224, [[REGISTER]]
; CHECK-NEXT: movl [[REGISTER]], (%esp)
; CHECK-NEXT: call{{.*}}use
