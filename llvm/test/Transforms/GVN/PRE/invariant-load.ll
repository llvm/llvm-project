; Test if the !invariant.load metadata is maintained by GVN.
; RUN: opt -basicaa -gvn -S < %s | FileCheck %s

define i32 @test1(i32* nocapture %p, i8* nocapture %q) {
; CHECK-LABEL: test1
; CHECK: %x = load i32, i32* %p, align 4, !invariant.load !0
; CHECK-NOT: %y = load
entry:
  %x = load i32, i32* %p, align 4, !invariant.load !0
  %conv = trunc i32 %x to i8
  store i8 %conv, i8* %q, align 1
  %y = load i32, i32* %p, align 4, !invariant.load !0
  %add = add i32 %y, 1
  ret i32 %add
}

define i32 @test2(i32* nocapture %p, i8* nocapture %q) {
; CHECK-LABEL: test2
; CHECK-NOT: !invariant.load
; CHECK-NOT: %y = load
entry:
  %x = load i32, i32* %p, align 4
  %conv = trunc i32 %x to i8
  store i8 %conv, i8* %q, align 1
  %y = load i32, i32* %p, align 4, !invariant.load !0
  %add = add i32 %y, 1
  ret i32 %add
}

; With the invariant.load metadata, what would otherwise
; be a case for PRE becomes a full redundancy.
define i32 @test3(i1 %cnd, i32* %p, i32* %q) {
; CHECK-LABEL: test3
; CHECK-NOT: load
entry:
  %v1 = load i32, i32* %p
  br i1 %cnd, label %bb1, label %bb2

bb1:
  store i32 5, i32* %q
  br label %bb2

bb2:
  %v2 = load i32, i32* %p, !invariant.load !0
  %res = sub i32 %v1, %v2
  ret i32 %res
}

; This test is here to document a case which doesn't optimize
; as well as it could.  
define i32 @test4(i1 %cnd, i32* %p, i32* %q) {
; CHECK-LABEL: test4
; %v2 is redundant, but GVN currently doesn't catch that
entry:
  %v1 = load i32, i32* %p, !invariant.load !0
  br i1 %cnd, label %bb1, label %bb2

bb1:
  store i32 5, i32* %q
  br label %bb2

bb2:
  %v2 = load i32, i32* %p
  %res = sub i32 %v1, %v2
  ret i32 %res
}

; Checks that we return the mustalias store as a def
; so that it contributes to value forwarding.  Note
; that we could and should remove the store too.
define i32 @test5(i1 %cnd, i32* %p) {
; CHECK-LABEL: test5
; CHECK-LABEL: entry:
; CHECK-NEXT: store i32 5, i32* %p
; CHECK-NEXT: ret i32 5
entry:
  %v1 = load i32, i32* %p, !invariant.load !0
  store i32 5, i32* %p ;; must alias store, want to exploit
  %v2 = load i32, i32* %p, !invariant.load !0
  ret i32 %v2
}


declare void @foo()

; Clobbering (mayalias) stores, even in function calls, can be ignored
define i32 @test6(i1 %cnd, i32* %p) {
; CHECK-LABEL: test6
; CHECK-LABEL: entry:
; CHECK-NEXT: @foo
; CHECK-NEXT: ret i32 0
entry:
  %v1 = load i32, i32* %p, !invariant.load !0
  call void @foo()
  %v2 = load i32, i32* %p, !invariant.load !0
  %res = sub i32 %v1, %v2
  ret i32 %res
}

declare noalias i32* @bar(...) 

; Same as previous, but a function with a noalias result (since they're handled
; differently in MDA)
define i32 @test7(i1 %cnd, i32* %p) {
; CHECK-LABEL: test7
; CHECK-LABEL: entry:
; CHECK-NEXT: @bar
; CHECK-NEXT: ret i32 0
entry:
  %v1 = load i32, i32* %p, !invariant.load !0
  call i32* (...) @bar(i32* %p)
  %v2 = load i32, i32* %p, !invariant.load !0
  %res = sub i32 %v1, %v2
  ret i32 %res
}

define i32 @test8(i1 %cnd, i32* %p) {
; CHECK-LABEL: test8
; CHECK: @bar
; CHECK: load i32, i32* %p2, !invariant.load
; CHECK: br label %merge
entry:
  %v1 = load i32, i32* %p, !invariant.load !0
  br i1 %cnd, label %taken, label %merge
taken:
  %p2 = call i32* (...) @bar(i32* %p)
  br label %merge
merge:
  %p3 = phi i32* [%p, %entry], [%p2, %taken]
  %v2 = load i32, i32* %p3, !invariant.load !0
  %res = sub i32 %v1, %v2
  ret i32 %res
}

!0 = !{ }

