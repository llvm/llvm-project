; Tests to make sure elimination of casts is working correctly
; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "p:32:32"

; This shouldn't convert to getelementptr because the relationship
; between the arithmetic and the layout of allocated memory is
; entirely unknown.
; CHECK: @test1
; CHECK: ptrtoint
; CHECK: add
; CHECK: inttoptr
define i8* @test1(i8* %t) {
        %tmpc = ptrtoint i8* %t to i32          ; <i32> [#uses=1]
        %tmpa = add i32 %tmpc, 32               ; <i32> [#uses=1]
        %tv = inttoptr i32 %tmpa to i8*         ; <i8*> [#uses=1]
        ret i8* %tv
}

; These casts should be folded away.
; CHECK: @test2
; CHECK: icmp eq i8* %a, %b
define i1 @test2(i8* %a, i8* %b) {
        %tmpa = ptrtoint i8* %a to i32          ; <i32> [#uses=1]
        %tmpb = ptrtoint i8* %b to i32          ; <i32> [#uses=1]
        %r = icmp eq i32 %tmpa, %tmpb           ; <i1> [#uses=1]
        ret i1 %r
}

; These casts should also be folded away.
; CHECK: @test3
; CHECK: icmp eq i8* %a, @global
@global = global i8 0
define i1 @test3(i8* %a) {
        %tmpa = ptrtoint i8* %a to i32
        %r = icmp eq i32 %tmpa, ptrtoint (i8* @global to i32)
        ret i1 %r
}

define i1 @test4(i32 %A) {
  %B = inttoptr i32 %A to i8*
  %C = icmp eq i8* %B, null
  ret i1 %C
; CHECK: @test4
; CHECK-NEXT: %C = icmp eq i32 %A, 0
; CHECK-NEXT: ret i1 %C 
}


; Pulling the cast out of the load allows us to eliminate the load, and then 
; the whole array.

        %op = type { float }
        %unop = type { i32 }
@Array = internal constant [1 x %op* (%op*)*] [ %op* (%op*)* @foo ]             ; <[1 x %op* (%op*)*]*> [#uses=1]

declare %op* @foo(%op* %X)

define %unop* @test5(%op* %O) {
        %tmp = load %unop* (%op*)** bitcast ([1 x %op* (%op*)*]* @Array to %unop* (%op*)**); <%unop* (%op*)*> [#uses=1]
        %tmp.2 = call %unop* %tmp( %op* %O )            ; <%unop*> [#uses=1]
        ret %unop* %tmp.2
; CHECK: @test5
; CHECK: call %op* @foo(%op* %O)
}



; InstCombine can not 'load (cast P)' -> cast (load P)' if the cast changes
; the address space.

define i8 @test6(i8 addrspace(1)* %source) {                                                                                        
entry: 
  %arrayidx223 = bitcast i8 addrspace(1)* %source to i8*
  %tmp4 = load i8* %arrayidx223
  ret i8 %tmp4
; CHECK: @test6
; CHECK: load i8* %arrayidx223
} 
