; RUN: opt < %s -passes=constmerge -S | FileCheck %s

; Don't merge constants with specified sections.

@T1G1 = internal constant i32 1, section "foo"
@T1G2 = internal constant i32 1, section "bar"
@T1G3 = internal constant i32 1, section "bar"

; CHECK: @T1G1
; CHECK: @T1G2
; CHECK: @T1G3

define void @test1(ptr %P1, ptr %P2, ptr %P3) {
        store ptr @T1G1, ptr %P1
        store ptr @T1G2, ptr %P2
        store ptr @T1G3, ptr %P3
        ret void
}

@T2a = internal constant i32 224
@T2b = internal addrspace(30) constant i32 224

; CHECK: @T2a
; CHECK: @T2b

define void @test2(ptr %P1, ptr %P2) {
        store ptr @T2a, ptr %P1
        store ptr addrspace(30)  @T2b, ptr %P2
        ret void
}

; PR8144 - Don't merge globals marked attribute(used)
; CHECK: @T3A = 
; CHECK: @T3B = 

@T3A = internal constant i32 0
@T3B = internal constant i32 0
@llvm.used = appending global [2 x ptr] [ptr @T3A, ptr @T3B], section
"llvm.metadata"

define void @test3() {
  call void asm sideeffect "T3A, T3B",""() ; invisible use of T3A and T3B
  ret void
}

; Don't merge constants with !type annotations.

@T4A1 = internal constant i32 2, !type !0
@T4A2 = internal unnamed_addr constant i32 2, !type !1

@T4B1 = internal constant i32 3, !type !0
@T4B2 = internal unnamed_addr constant i32 3, !type !0

@T4C1 = internal constant i32 4, !type !0
@T4C2 = unnamed_addr constant i32 4

@T4D1 = unnamed_addr constant i32 5, !type !0
@T4D2 = internal constant i32 5

!0 = !{i64 0, !"typeinfo name for A"}
!1 = !{i64 0, !"typeinfo name for B"}

; CHECK: @T4A1
; CHECK: @T4A2
; CHECK: @T4B1
; CHECK: @T4B2
; CHECK: @T4C1
; CHECK: @T4C2
; CHECK: @T4D1
; CHECK: @T4D2

define void @test4(ptr %P1, ptr %P2, ptr %P3, ptr %P4, ptr %P5, ptr %P6, ptr %P7, ptr %P8) {
        store ptr @T4A1, ptr %P1
        store ptr @T4A2, ptr %P2
        store ptr @T4B1, ptr %P3
        store ptr @T4B2, ptr %P4
        store ptr @T4C1, ptr %P5
        store ptr @T4C2, ptr %P6
        store ptr @T4D1, ptr %P7
        store ptr @T4D2, ptr %P8
        ret void
}

; CHECK: @T5tls
; CHECK: @T5ua

@T5tls = private thread_local constant i32 555
@T5ua = private unnamed_addr constant i32 555

define void @test5(ptr %P1, ptr %P2) {
        store ptr @T5tls, ptr %P1
        store ptr @T5ua, ptr %P2
        ret void
}
