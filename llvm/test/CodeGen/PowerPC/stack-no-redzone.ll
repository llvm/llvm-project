; Test that accesses of the stack remain within the range defined by R1,
; i.e. that loads and stores only access the allocated stack. This does not
; have to be the case when red zone is present.

; Make sure that there is no red zone, i.e. ppc32 and SVR4 ABI.
; RUN: llc -mtriple=powerpc--freebsd-elf < %s | FileCheck %s

; There are two ways that the stack pointer can be adjusted in the prologue:
; - by adding an immediate value:
;     stwu r1, -imm(r1)
; - by adding another register:
;     stwux r1, rx, r1
;
; The restoring of the stack pointer can be done:
; - by adding an immediate value to it:
;     addi r1, r1, imm
; - by copying the value from another register:
;     mr r1, rx


; Nothing (no special features).
;
; CHECK-LABEL: test_n:
; CHECK-NOT: stw {{[0-9]+}}, -{{[0-9]+}}(1)
; CHECK: stwu 1, -[[SIZE:[0-9]+]](1)
; CHECK: addi 1, 1, [[SIZE]]
; CHECK-NOT: lwz {{[0-9]+}}, -{{[0-9]+}}(1)
define i32 @test_n() local_unnamed_addr #0 {
entry:
  %t0 = tail call i32 @bar0() #0
  ret i32 %t0
}

; Aligned object on the stack.
;
; CHECK-LABEL: test_a:
; CHECK-NOT: stw {{[0-9]+}}, -{{[0-9]+}}(1)
; CHECK: stwux 1, 1, {{[0-9]+}}
; CHECK: mr 1, {{[0-9]+}}
; CHECK-NOT: lwz {{[0-9]+}}, -{{[0-9]+}}(1)

define i32 @test_a() local_unnamed_addr #0 {
entry:
  %t0 = alloca i32, align 128
  %t1 = tail call i32 @bar1(ptr %t0) #0
  ret i32 %t1
}

; Dynamic allocation on the stack.
;
; CHECK-LABEL: test_d:
; CHECK-NOT: stw {{[0-9]+}}, -{{[0-9]+}}(1)
; CHECK: stwu 1, -[[SIZE:[0-9]+]](1)
; CHECK: mr 1, {{[0-9]+}}
; CHECK-NOT: lwz {{[0-9]+}}, -{{[0-9]+}}(1)
define i32 @test_d(i32 %p0) local_unnamed_addr #0 {
  %t0 = alloca i32, i32 %p0, align 4
  %t1 = tail call i32 @bar1(ptr %t0) #0
  ret i32 %t1
}

; Large stack (exceeds size of D-field).
; CHECK-LABEL: test_s:
; CHECK-NOT: stw {{[0-9]+}}, -{{[0-9]+}}(1)
; CHECK: stwux 1, 1, {{[0-9]+}}
; CHECK: mr 1, {{[0-9]+}}
; CHECK-NOT: lwz {{[0-9]+}}, -{{[0-9]+}}(1)
define i32 @test_s(i32 %p0) local_unnamed_addr #0 {
entry:
  %t0 = alloca [16384 x i32]
  %t2 = tail call i32 @bar1(ptr %t0) #0
  ret i32 %t2
}

; Combinations.

; CHECK-LABEL: test_ad:
; CHECK-NOT: stw {{[0-9]+}}, -{{[0-9]+}}(1)
; CHECK: stwux 1, 1, {{[0-9]+}}
; CHECK: mr 1, {{[0-9]+}}
; CHECK-NOT: lwz {{[0-9]+}}, -{{[0-9]+}}(1)
define i32 @test_ad(i32 %p0) local_unnamed_addr #0 {
  %t0 = alloca i32, align 128
  %t1 = alloca i32, i32 %p0, align 4
  %t2 = tail call i32 @bar1(ptr %t0) #0
  %t3 = tail call i32 @bar1(ptr %t1) #0
  %t4 = add i32 %t2, %t3
  ret i32 %t4
}

; CHECK-LABEL: test_as:
; CHECK-NOT: stw {{[0-9]+}}, -{{[0-9]+}}(1)
; CHECK: stwux 1, 1, {{[0-9]+}}
; CHECK: mr 1, {{[0-9]+}}
; CHECK-NOT: lwz {{[0-9]+}}, -{{[0-9]+}}(1)
define i32 @test_as() local_unnamed_addr #0 {
  %t0 = alloca i32, align 128
  %t1 = alloca [16384 x i32]
  %t2 = tail call i32 @bar1(ptr %t0) #0
  %t4 = tail call i32 @bar1(ptr %t1) #0
  %t5 = add i32 %t2, %t4
  ret i32 %t5
}

; CHECK-LABEL: test_ds:
; CHECK-NOT: stw {{[0-9]+}}, -{{[0-9]+}}(1)
; CHECK: stwux 1, 1, {{[0-9]+}}
; CHECK: mr 1, {{[0-9]+}}
; CHECK-NOT: lwz {{[0-9]+}}, -{{[0-9]+}}(1)
define i32 @test_ds(i32 %p0) local_unnamed_addr #0 {
  %t0 = alloca i32, i32 %p0, align 4
  %t1 = alloca [16384 x i32]
  %t2 = tail call i32 @bar1(ptr %t0) #0
  %t4 = tail call i32 @bar1(ptr %t1) #0
  %t5 = add i32 %t2, %t4
  ret i32 %t5
}

; CHECK-LABEL: test_ads:
; CHECK-NOT: stw {{[0-9]+}}, -{{[0-9]+}}(1)
; CHECK: stwux 1, 1, {{[0-9]+}}
; CHECK: mr 1, {{[0-9]+}}
; CHECK-NOT: lwz {{[0-9]+}}, -{{[0-9]+}}(1)
define i32 @test_ads(i32 %p0) local_unnamed_addr #0 {
  %t0 = alloca i32, align 128
  %t1 = alloca i32, i32 %p0, align 4
  %t2 = alloca [16384 x i32]

  %t3 = tail call i32 @bar1(ptr %t0) #0
  %t4 = tail call i32 @bar1(ptr %t1) #0
  %t5 = add i32 %t3, %t4

  %t7 = tail call i32 @bar1(ptr %t2) #0
  %t8 = add i32 %t5, %t7
  ret i32 %t7
}


declare i32 @bar0(...) local_unnamed_addr #0
declare i32 @bar1(...) local_unnamed_addr #0

attributes #0 = { nounwind }
