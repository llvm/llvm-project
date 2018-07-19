; RUN: opt < %s -simplifycfg -S | FileCheck %s
; PR2967

target datalayout =
"e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32"
target triple = "i386-pc-linux-gnu"

define void @test1(i32 %x) nounwind {
entry:
        %0 = icmp eq i32 %x, 0          ; <i1> [#uses=1]
        br i1 %0, label %bb, label %return

bb:             ; preds = %entry
        %1 = load volatile i32, i32* null
        unreachable
        
        br label %return
return:         ; preds = %entry
        ret void
; CHECK-LABEL: @test1(
; CHECK: load volatile
}

; rdar://7958343
define void @test2() nounwind {
entry:
        store i32 4,i32* null
        ret void
        
; CHECK-LABEL: @test2(
; CHECK: call void @llvm.trap
; CHECK: unreachable
}

; PR7369
define void @test3() nounwind {
entry:
        store volatile i32 4, i32* null
        ret void

; CHECK-LABEL: @test3(
; CHECK: store volatile i32 4, i32* null
; CHECK: ret
}

; Check store before unreachable.
define void @test4(i1 %C, i32* %P) {
; CHECK-LABEL: @test4(
; CHECK: entry:
; CHECK-NEXT: br i1 %C
entry:
  br i1 %C, label %T, label %F
T:
  store volatile i32 0, i32* %P
  unreachable
F:
  ret void
}

; Check cmpxchg before unreachable.
define void @test5(i1 %C, i32* %P) {
; CHECK-LABEL: @test5(
; CHECK: entry:
; CHECK-NEXT: br i1 %C
entry:
  br i1 %C, label %T, label %F
T:
  cmpxchg volatile i32* %P, i32 0, i32 1 seq_cst seq_cst
  unreachable
F:
  ret void
}

; Check atomicrmw before unreachable.
define void @test6(i1 %C, i32* %P) {
; CHECK-LABEL: @test6(
; CHECK: entry:
; CHECK-NEXT: br i1 %C
entry:
  br i1 %C, label %T, label %F
T:
  atomicrmw volatile xchg i32* %P, i32 0 seq_cst
  unreachable
F:
  ret void
}

