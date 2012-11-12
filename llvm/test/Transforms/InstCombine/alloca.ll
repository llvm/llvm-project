target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

; RUN: opt < %s -instcombine -S | FileCheck %s
; END.

declare void @use(...)

@int = global i32 zeroinitializer

; Zero byte allocas should be merged if they can't be deleted.
; CHECK: @test
; CHECK: alloca
; CHECK-NOT: alloca
define void @test() {
        %X = alloca [0 x i32]           ; <[0 x i32]*> [#uses=1]
        call void (...)* @use( [0 x i32]* %X )
        %Y = alloca i32, i32 0          ; <i32*> [#uses=1]
        call void (...)* @use( i32* %Y )
        %Z = alloca {  }                ; <{  }*> [#uses=1]
        call void (...)* @use( {  }* %Z )
        %size = load i32* @int
        %A = alloca {{}}, i32 %size
        call void (...)* @use( {{}}* %A )
        ret void
}

; Zero byte allocas should be deleted.
; CHECK: @test2
; CHECK-NOT: alloca
define void @test2() {
        %A = alloca i32         ; <i32*> [#uses=1]
        store i32 123, i32* %A
        ret void
}

; Zero byte allocas should be deleted.
; CHECK: @test3
; CHECK-NOT: alloca
define void @test3() {
        %A = alloca { i32 }             ; <{ i32 }*> [#uses=1]
        %B = getelementptr { i32 }* %A, i32 0, i32 0            ; <i32*> [#uses=1]
        store i32 123, i32* %B
        ret void
}

; CHECK: @test4
; CHECK: = zext i32 %n to i64
; CHECK: %A = alloca i32, i64 %
define i32* @test4(i32 %n) {
  %A = alloca i32, i32 %n
  ret i32* %A
}

; Allocas which are only used by GEPs, bitcasts, and stores (transitively)
; should be deleted.
define void @test5() {
; CHECK: @test5
; CHECK-NOT: alloca
; CHECK-NOT: store
; CHECK: ret

entry:
  %a = alloca { i32 }
  %b = alloca i32*
  %a.1 = getelementptr { i32 }* %a, i32 0, i32 0
  store i32 123, i32* %a.1
  store i32* %a.1, i32** %b
  %b.1 = bitcast i32** %b to i32*
  store i32 123, i32* %b.1
  %a.2 = getelementptr { i32 }* %a, i32 0, i32 0
  store atomic i32 2, i32* %a.2 unordered, align 4
  %a.3 = getelementptr { i32 }* %a, i32 0, i32 0
  store atomic i32 3, i32* %a.3 release, align 4
  %a.4 = getelementptr { i32 }* %a, i32 0, i32 0
  store atomic i32 4, i32* %a.4 seq_cst, align 4
  ret void
}

declare void @f(i32* %p)

; Check that we don't delete allocas in some erroneous cases.
define void @test6() {
; CHECK: @test6
; CHECK-NOT: ret
; CHECK: alloca
; CHECK-NEXT: alloca
; CHECK: ret

entry:
  %a = alloca { i32 }
  %b = alloca i32
  %a.1 = getelementptr { i32 }* %a, i32 0, i32 0
  store volatile i32 123, i32* %a.1
  tail call void @f(i32* %b)
  ret void
}
