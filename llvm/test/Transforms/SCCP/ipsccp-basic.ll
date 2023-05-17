; RUN: opt < %s -passes=ipsccp -S | FileCheck %s
; RUN: opt < %s -enable-debugify -passes=ipsccp -debugify-quiet -disable-output

;;======================== test1

define internal i32 @test1a(i32 %A) {
	%X = add i32 1, 2
	ret i32 %A
}
; CHECK-LABEL: define internal i32 @test1a(
; CHECK: ret i32 undef

define i32 @test1b() {
	%X = call i32 @test1a( i32 17 )
	ret i32 %X

; CHECK-LABEL: define i32 @test1b(
; CHECK: ret i32 17
}



;;======================== test2

define internal i32 @test2a(i32 %A) {
	%C = icmp eq i32 %A, 0	
	br i1 %C, label %T, label %F
T:
	%B = call i32 @test2a( i32 0 )
	ret i32 0
F:
	%C.upgrd.1 = call i32 @test2a(i32 1)
	ret i32 %C.upgrd.1
}
; CHECK-LABEL: define internal i32 @test2a(
; CHECK-NEXT: br label %T
; CHECK: ret i32 undef


define i32 @test2b() {
	%X = call i32 @test2a(i32 0)
	ret i32 %X
}
; CHECK-LABEL: define i32 @test2b(
; CHECK-NEXT: %X = call i32 @test2a(i32 0)
; CHECK-NEXT: ret i32 0


;;======================== test3

@G = internal global i32 undef

define void @test3a() {
	%X = load i32, ptr @G
	store i32 %X, ptr @G
	ret void
}
; CHECK-LABEL: define void @test3a(
; CHECK-NEXT:    %X = load i32, ptr @G
; CHECK-NEXT:    store i32 %X, ptr @G
; CHECK-NEXT:   ret void


define i32 @test3b() {
	%V = load i32, ptr @G
	%C = icmp eq i32 %V, 17
	br i1 %C, label %T, label %F
T:
	store i32 17, ptr @G
	ret i32 %V
F:	
	store i32 123, ptr @G
	ret i32 0
}
; CHECK-LABEL: define i32 @test3b(
; CHECK-NEXT:    %V = load i32, ptr @G
; CHECK-NEXT:    %C = icmp eq i32 %V, 17
; CHECK-NEXT:    br i1 %C, label %T, label %F

; CHECK-LABEL: T:
; CHECK-NEXT:   store i32 17, ptr @G
; CHECK-NEXT:   ret i32 17

; CHECK-LABEL: F:
; CHECK-NEXT:    store i32 123, ptr @G
; CHECK-NEXT:    ret i32 0

;;======================== test4

define internal {i64,i64} @test4a() {
  %a = insertvalue {i64,i64} undef, i64 4, 1
  %b = insertvalue {i64,i64} %a, i64 5, 0
  ret {i64,i64} %b
}

; CHECK-LABEL: define internal { i64, i64 } @test4a(
; CHECK-NEXT:   ret { i64, i64 } undef
; CHECK-NEXT: }

define i64 @test4b() personality ptr @__gxx_personality_v0 {
  %a = invoke {i64,i64} @test4a()
          to label %A unwind label %B
A:
  %b = extractvalue {i64,i64} %a, 0
  %c = call i64 @test4c(i64 %b)
  ret i64 %c
B:
  %val = landingpad { ptr, i32 }
           catch ptr null
  ret i64 0
}
; CHECK: define i64 @test4b()
; CHECK:   %c = call i64 @test4c(i64 5)
; CHECK-NEXT:  ret i64 5


define internal i64 @test4c(i64 %a) {
  ret i64 %a
}
; CHECK-LABEL: define internal i64 @test4c(
; CHECK: ret i64 undef



;;======================== test5

; PR4313
define internal {i64,i64} @test5a() {
  %a = insertvalue {i64,i64} undef, i64 4, 1
  %b = insertvalue {i64,i64} %a, i64 5, 0
  ret {i64,i64} %b
}

define i64 @test5b() personality ptr @__gxx_personality_v0 {
  %a = invoke {i64,i64} @test5a()
          to label %A unwind label %B
A:
  %c = call i64 @test5c({i64,i64} %a)
  ret i64 %c
B:
  %val = landingpad { ptr, i32 }
           catch ptr null
  ret i64 0
}

; CHECK: define i64 @test5b()
; CHECK:     A:
; CHECK-NEXT:  %c = call i64 @test5c({ i64, i64 } { i64 5, i64 4 })
; CHECK-NEXT:  ret i64 5

define internal i64 @test5c({i64,i64} %a) {
  %b = extractvalue {i64,i64} %a, 0
  ret i64 %b
}


;;======================== test6

define i64 @test6a() {
  ret i64 0
}

define i64 @test6b() {
  %a = call i64 @test6a()
  ret i64 %a
}
; CHECK-LABEL: define i64 @test6b(
; CHECK: ret i64 0

;;======================== test7


%T = type {i32,i32}

define internal %T @test7a(i32 %A) {
  %X = add i32 1, %A
  %mrv0 = insertvalue %T undef, i32 %X, 0
  %mrv1 = insertvalue %T %mrv0, i32 %A, 1
  ret %T %mrv1
; CHECK-LABEL: @test7a(
; CHECK-NEXT: ret %T undef
}

define i32 @test7b() {
	%X = call %T @test7a(i32 17)
        %Y = extractvalue %T %X, 0
	%Z = add i32 %Y, %Y
	ret i32 %Z
; CHECK-LABEL: define i32 @test7b(
; CHECK-NEXT: call %T @test7a(i32 17)
; CHECK-NEXT: ret i32 36
}

;;======================== test8


define internal {} @test8a(i32 %A, ptr %P) {
  store i32 %A, ptr %P
  ret {} {}
; CHECK-LABEL: @test8a(
; CHECK-NEXT: store i32 5, 
; CHECK-NEXT: ret 
}

define void @test8b(ptr %P) {
    %X = call {} @test8a(i32 5, ptr %P)
    ret void
; CHECK-LABEL: define void @test8b(
; CHECK-NEXT: call {} @test8a
; CHECK-NEXT: ret void
}

;;======================== test9

@test9g = internal global {  } zeroinitializer

define void @test9() {
entry:
        %local_foo = alloca {  }
        load {  }, ptr @test9g
        store {  } %0, ptr %local_foo
        ret void
}

; CHECK-LABEL: define void @test9(
; CHECK-NEXT: entry:
; CHECK-NEXT: %local_foo = alloca {}
; CHECK-NEXT:  store {} zeroinitializer, ptr %local_foo
; CHECK-NEXT: ret void

declare i32 @__gxx_personality_v0(...)

;;======================== test10

define i32 @test10a() nounwind {
entry:
  %call = call i32 @test10b(i32 undef)
  ret i32 %call

; CHECK-LABEL: define i32 @test10a(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %call = call i32 @test10b(i32 undef)
; CHECK-NEXT:    ret i32 %call
}

define internal i32 @test10b(i32 %x) nounwind {
entry:
  %r = and i32 %x, 1
  ret i32 %r
; CHECK-LABEL: define internal i32 @test10b(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %r = and i32 undef, 1
; CHECK-NEXT:    ret i32 %r
}

;;======================== test11

define i64 @test11a() {
  %xor = xor i64 undef, undef
  ret i64 %xor
; CHECK-LABEL: define i64 @test11a
; CHECK-NEXT:    %xor = xor i64 undef, undef
; CHECK-NEXT:    ret i64 %xor
}

define i64 @test11b() {
  %call1 = call i64 @test11a()
  %call2 = call i64 @llvm.ctpop.i64(i64 %call1)
  ret i64 %call2
; CHECK-LABEL: define i64 @test11b
; CHECK-NEXT:   [[call1:%.*]] = call i64 @test11a()
; CHECK-NEXT:    [[call2:%.*]] = call i64 @llvm.ctpop.i64(i64 [[call1]])
; CHECK-NEXT: ret i64 [[call2]]
}

declare i64 @llvm.ctpop.i64(i64)

;;======================== test12
;; Ensure that a struct as an arg to a potentially constant-foldable
;; function does not crash SCCP (for now it'll just ignores it)

define i1 @test12() {
  %c = call i1 @llvm.is.constant.sl_i32i32s({i32, i32} {i32 -1, i32 32})
  ret i1 %c
; CHECK-LABEL: define i1 @test12
; CHECK: ret i1 %c
}

declare i1 @llvm.is.constant.sl_i32i32s({i32, i32} %a)
