; RUN: llc %s -o - -mtriple=thumbv8m.base | FileCheck %s

declare i32 @g(...)

declare i32 @h0(i32, i32, i32, i32)
define hidden i32 @f0() {
  %1 = tail call i32 bitcast (i32 (...)* @g to i32 ()*)()
  %2 = tail call i32 @h0(i32 %1, i32 1, i32 2, i32 3)
  ret i32 %2
; CHECK-LABEL: f0
; CHECK:      ldr     [[POP:r[4567]]], [sp, #4]
; CHECK-NEXT: mov     lr, [[POP]]
; CHECK-NEXT: pop     {{.*}}[[POP]]
; CHECK-NEXT: add     sp, #4
; CHECK-NEXT: b       h0
}

declare i32 @h1(i32)
define hidden i32 @f1() {
  %1 = tail call i32 bitcast (i32 (...)* @g to i32 ()*)()
  %2 = tail call i32 @h1(i32 %1)
  ret i32 %2
; CHECK-LABEL: f1
; CHECK: pop     {r7}
; CHECK: pop     {r1}
; CHECK: mov     lr, r1
; CHECK: b       h1
}

declare i32 @h2(i32, i32, i32, i32, i32)
define hidden i32 @f2(i32, i32, i32, i32, i32) {
  %6 = tail call i32 bitcast (i32 (...)* @g to i32 ()*)()
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %10, label %8

  %9 = tail call i32 @h2(i32 %6, i32 %1, i32 %2, i32 %3, i32 %4)
  br label %10

  %11 = phi i32 [ %9, %8 ], [ -1, %5 ]
  ret i32 %11
; CHECK-LABEL: f2
; CHECK:      ldr     [[POP:r[4567]]], [sp, #12]
; CHECK-NEXT: mov     lr, [[POP]]
; CHECK-NEXT: pop     {{.*}}[[POP]]
; CHECK-NEXT: add     sp, #4
; CHECK-NEXT: b       h2
}

; Make sure that tail calls to function pointers that require r0-r3 for argument
; passing do not break the compiler.
@fnptr = global i32 (i32, i32, i32, i32)* null
define i32 @test3() {
; CHECK-LABEL: test3:
; CHECK: blx {{r[0-9]+}}
  %1 = load i32 (i32, i32, i32, i32)*, i32 (i32, i32, i32, i32)** @fnptr
  %2 = tail call i32 %1(i32 1, i32 2, i32 3, i32 4)
  ret i32 %2
}

@fnptr2 = global i32 (i32, i32, i64)* null
define i32 @test4() {
; CHECK-LABEL: test4:
; CHECK: blx {{r[0-9]+}}
  %1 = load i32 (i32, i32, i64)*, i32 (i32, i32, i64)** @fnptr2
  %2 = tail call i32 %1(i32 1, i32 2, i64 3)
  ret i32 %2
}

; Check that tail calls to function pointers where not all of r0-r3 are used for
; parameter passing are tail-call optimized.
; test5: params in r0, r1. r2 & r3 are free.
@fnptr3 = global i32 (i32, i32)* null
define i32 @test5() {
; CHECK-LABEL: test5:
; CHECK: ldr [[REG:r[0-9]+]]
; CHECK: bx [[REG]]
; CHECK-NOT: blx [[REG]]
  %1 = load i32 (i32, i32)*, i32 (i32, i32)** @fnptr3
  %2 = tail call i32 %1(i32 1, i32 2)
  ret i32 %2
}

; test6: params in r0 and r2-r3. r1 is free.
@fnptr4 = global i32 (i32, i64)* null
define i32 @test6() {
; CHECK-LABEL: test6:
; CHECK: ldr [[REG:r[0-9]+]]
; CHECK: bx [[REG]]
; CHECK-NOT: blx [[REG]]
  %1 = load i32 (i32, i64)*, i32 (i32, i64)** @fnptr4
  %2 = tail call i32 %1(i32 1, i64 2)
  ret i32 %2
}

; Check that tail calls to functions other than function pointers are
; tail-call optimized.
define i32 @test7() {
; CHECK-LABEL: test7:
; CHECK: b bar
; CHECK-NOT: bl bar
  %tail = tail call i32 @bar(i32 1, i32 2, i32 3, i32 4)
  ret i32 %tail
}

declare i32 @bar(i32, i32, i32, i32)

; Regression test for failure to load indirect branch target (class tcGPR) from
; a stack slot.
%struct.S = type { i32 }

define void @test8(i32 (i32, i32, i32)* nocapture %fn, i32 %x) local_unnamed_addr {
entry:
  %call = tail call %struct.S* bitcast (%struct.S* (...)* @test8_u to %struct.S* ()*)()
  %a = getelementptr inbounds %struct.S, %struct.S* %call, i32 0, i32 0
  %0 = load i32, i32* %a, align 4
  %call1 = tail call i32 @test8_h(i32 0)
  %call2 = tail call i32 @test8_g(i32 %0, i32 %call1, i32 0)
  store i32 %x, i32* %a, align 4
  %call4 = tail call i32 %fn(i32 1, i32 2, i32 3)
  ret void
}

declare %struct.S* @test8_u(...)

declare i32 @test8_g(i32, i32, i32)

declare i32 @test8_h(i32)
; CHECK: str r0, [sp] @ 4-byte Spill
; CHECK: ldr r3, [sp] @ 4-byte Reload
; CHECK: bx r3
