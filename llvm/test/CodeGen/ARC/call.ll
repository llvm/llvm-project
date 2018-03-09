; RUN: llc -march=arc < %s | FileCheck %s


declare i32 @goo1(i32) nounwind

; CHECK-LABEL: call1
; CHECK: bl @goo1
define i32 @call1(i32 %a) nounwind {
entry:
  %x = call i32 @goo1(i32 %a)
  ret i32 %x
}

declare i32 @goo2(i32, i32, i32, i32, i32, i32, i32, i32) nounwind

; CHECK-LABEL: call2
; CHECK-DAG: mov %r0, 0
; CHECK-DAG: mov %r1, 1
; CHECK-DAG: mov %r2, 2
; CHECK-DAG: mov %r3, 3
; CHECK-DAG: mov %r4, 4
; CHECK-DAG: mov %r5, 5
; CHECK-DAG: mov %r6, 6
; CHECK-DAG: mov %r7, 7
; CHECK: bl @goo2
define i32 @call2() nounwind {
entry:
  %x = call i32 @goo2(i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7)
  ret i32 %x
}

declare i32 @goo3(i64, i32, i64) nounwind
; call goo3(0xEEEEEEEE77777777, 0x55555555, 0xAAAAAAAA33333333)
; 0xEEEEEEEE == -286331154
; 0x77777777 == 2004318071
; 0x55555555 == 1431655765
; 0xAAAAAAAA == -1431655766
; 0x33333333 == 858993459
; CHECK-LABEL: call3
; CHECK-DAG: mov %r0, 2004318071
; CHECK-DAG: mov %r1, -286331154
; CHECK-DAG: mov %r2, 1431655765
; CHECK-DAG: mov %r3, 858993459
; CHECK-DAG: mov %r4, -1431655766
; CHECK: bl @goo3
define i32 @call3() nounwind {
entry:
  %x = call i32 @goo3(i64 17216961133457930103,
                      i32 1431655765,
                      i64 12297829380468716339)
  ret i32 %x
}

declare i64 @goo4()

; 64-bit values are returned in r0r1
; CHECK-LABEL: call4
; CHECK: bl @goo4
; CHECK: lsr %r0, %r1, 16
define i32 @call4() nounwind {
  %x = call i64 @goo4()
  %v1 = lshr i64 %x, 48
  %v = trunc i64 %v1 to i32
  ret i32 %v
}

; 0x0000ffff00ff00ff=281470698455295
; returned as r0=0x00ff00ff=16711935, r1=0x0000ffff=65535
; CHECK-LABEL: ret1
; CHECK-DAG: mov %r1, 65535
; CHECK-DAG: mov %r0, 16711935
define i64 @ret1() nounwind {
  ret i64 281470698455295
}

@funcptr = external global i32 (i32)*, align 4
; Indirect calls use JL
; CHECK-LABEL: call_indirect
; CHECK-DAG: ld %r[[REG:[0-9]+]], [@funcptr]
; CHECK-DAG: mov %r0, 12
; CHECK:     jl [%r[[REG]]]
define i32 @call_indirect(i32 %x) nounwind {
  %f = load i32 (i32)*, i32 (i32)** @funcptr, align 4
  %call = call i32 %f(i32 12)
  %add = add nsw i32 %call, %x
  ret i32 %add
}

