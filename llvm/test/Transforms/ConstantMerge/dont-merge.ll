; RUN: opt < %s -constmerge -S | FileCheck %s

; Don't merge constants with specified sections.

@T1G1 = internal constant i32 1, section "foo"
@T1G2 = internal constant i32 1, section "bar"
@T1G3 = internal constant i32 1, section "bar"

; CHECK: @T1G1
; CHECK: @T1G2
; CHECK: @T1G3

define void @test1(i32** %P1, i32** %P2, i32** %P3) {
        store i32* @T1G1, i32** %P1
        store i32* @T1G2, i32** %P2
        store i32* @T1G3, i32** %P3
        ret void
}

@T2a = internal constant i32 224
@T2b = internal addrspace(30) constant i32 224

; CHECK: @T2a
; CHECK: @T2b

define void @test2(i32** %P1, i32 addrspace(30)** %P2) {
        store i32* @T2a, i32** %P1
        store i32 addrspace(30)*  @T2b, i32 addrspace(30)** %P2
        ret void
}

; PR8144 - Don't merge globals marked attribute(used)
; CHECK: @T3A = 
; CHECK: @T3B = 

@T3A = internal constant i32 0
@T3B = internal constant i32 0
@llvm.used = appending global [2 x i32*] [i32* @T3A, i32* @T3B], section
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

define void @test4(i32** %P1, i32** %P2, i32** %P3, i32** %P4, i32** %P5, i32** %P6, i32** %P7, i32** %P8) {
        store i32* @T4A1, i32** %P1
        store i32* @T4A2, i32** %P2
        store i32* @T4B1, i32** %P3
        store i32* @T4B2, i32** %P4
        store i32* @T4C1, i32** %P5
        store i32* @T4C2, i32** %P6
        store i32* @T4D1, i32** %P7
        store i32* @T4D2, i32** %P8
        ret void
}

; CHECK: @T5tls
; CHECK: @T5ua

@T5tls = private thread_local constant i32 555
@T5ua = private unnamed_addr constant i32 555

define void @test5(i32** %P1, i32** %P2) {
        store i32* @T5tls, i32** %P1
        store i32* @T5ua, i32** %P2
        ret void
}
