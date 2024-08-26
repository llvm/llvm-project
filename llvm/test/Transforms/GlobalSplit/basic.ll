; RUN: opt -S -passes=globalsplit %s | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @vtt = constant [3 x ptr] [ptr @global.0, ptr getelementptr (i8, ptr @global.0, i64 8), ptr @global.1]
@vtt = constant [3 x ptr] [
  ptr getelementptr inrange(0, 16) ({ [2 x ptr], [1 x ptr] }, ptr @global, i32 0, i32 0, i32 0),
  ptr getelementptr inrange(-8, 8) ({ [2 x ptr], [1 x ptr] }, ptr @global, i32 0, i32 0, i32 1),
  ptr getelementptr inrange(0, 8) ({ [2 x ptr], [1 x ptr] }, ptr @global, i32 0, i32 1, i32 0)
]

; CHECK-NOT: @global =
; CHECK: @global.0 = private constant [2 x ptr] [ptr @f1, ptr @f2], !type [[T1:![0-9]+]], !type [[T2:![0-9]+]], !type [[T3:![0-9]+]], !vcall_visibility [[VIS:![0-9]+]]{{$}}
; CHECK: @global.1 = private constant [1 x ptr] [ptr @f3], !type [[T4:![0-9]+]], !type [[T5:![0-9]+]], !vcall_visibility [[VIS]]{{$}}
; CHECK-NOT: @global =
@global = internal constant { [2 x ptr], [1 x ptr] } {
  [2 x ptr] [ptr @f1, ptr @f2],
  [1 x ptr] [ptr @f3]
}, !type !0, !type !1, !type !2, !type !3, !type !4, !vcall_visibility !5

; CHECK: define ptr @f1()
define ptr @f1() {
  ; CHECK-NEXT: ret ptr @global.0
  ret ptr getelementptr inrange(0, 16) ({ [2 x ptr], [1 x ptr] }, ptr @global, i32 0, i32 0, i32 0)
}

; CHECK: define ptr @f2()
define ptr @f2() {
  ; CHECK-NEXT: ret ptr getelementptr (i8, ptr @global.0, i64 8)
  ret ptr getelementptr inrange(-8, 8) ({ [2 x ptr], [1 x ptr] }, ptr @global, i32 0, i32 0, i32 1)
}

; CHECK: define ptr @f3()
define ptr @f3() {
  ; CHECK-NEXT: ret ptr getelementptr (i8, ptr @global.0, i64 16)
  ret ptr getelementptr inrange(-16, 0) ({ [2 x ptr], [1 x ptr] }, ptr @global, i32 0, i32 0, i32 2)
}

; CHECK: define ptr @f4()
define ptr @f4() {
  ; CHECK-NEXT: ret ptr @global.1
  ret ptr getelementptr inrange(0, 8) ({ [2 x ptr], [1 x ptr] }, ptr @global, i32 0, i32 1, i32 0)
}

define void @foo() {
  %p = call i1 @llvm.type.test(ptr null, metadata !"")
  ret void
}

declare i1 @llvm.type.test(ptr, metadata) nounwind readnone

; CHECK: [[T1]] = !{i32 0, !"foo"}
; CHECK: [[T2]] = !{i32 15, !"bar"}
; CHECK: [[T3]] = !{i32 16, !"a"}
; CHECK: [[VIS]] = !{i64 2}
; CHECK: [[T4]] = !{i32 1, !"b"}
; CHECK: [[T5]] = !{i32 8, !"c"}
!0 = !{i32 0, !"foo"}
!1 = !{i32 15, !"bar"}
!2 = !{i32 16, !"a"}
!3 = !{i32 17, !"b"}
!4 = !{i32 24, !"c"}
!5 = !{i64 2}
