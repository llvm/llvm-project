; RUN: opt < %s -functionattrs -S | FileCheck %s
@g = global i32* null		; <i32**> [#uses=1]

; CHECK: define i32* @c1(i32* %q)
define i32* @c1(i32* %q) {
	ret i32* %q
}

; CHECK: define void @c2(i32* %q)
define void @c2(i32* %q) {
	store i32* %q, i32** @g
	ret void
}

; CHECK: define void @c3(i32* %q)
define void @c3(i32* %q) {
	call void @c2(i32* %q)
	ret void
}

; CHECK: define i1 @c4(i32* %q, i32 %bitno)
define i1 @c4(i32* %q, i32 %bitno) {
	%tmp = ptrtoint i32* %q to i32
	%tmp2 = lshr i32 %tmp, %bitno
	%bit = trunc i32 %tmp2 to i1
	br i1 %bit, label %l1, label %l0
l0:
	ret i1 0 ; escaping value not caught by def-use chaining.
l1:
	ret i1 1 ; escaping value not caught by def-use chaining.
}

@lookup_table = global [2 x i1] [ i1 0, i1 1 ]

; CHECK: define i1 @c5(i32* %q, i32 %bitno)
define i1 @c5(i32* %q, i32 %bitno) {
	%tmp = ptrtoint i32* %q to i32
	%tmp2 = lshr i32 %tmp, %bitno
	%bit = and i32 %tmp2, 1
        ; subtle escape mechanism follows
	%lookup = getelementptr [2 x i1]* @lookup_table, i32 0, i32 %bit
	%val = load i1* %lookup
	ret i1 %val
}

declare void @throw_if_bit_set(i8*, i8) readonly

; CHECK: define i1 @c6(i8* %q, i8 %bit)
define i1 @c6(i8* %q, i8 %bit) {
	invoke void @throw_if_bit_set(i8* %q, i8 %bit)
		to label %ret0 unwind label %ret1
ret0:
	ret i1 0
ret1:
        %exn = landingpad {i8*, i32} personality i32 (...)* @__gxx_personality_v0
                 cleanup
	ret i1 1
}

declare i32 @__gxx_personality_v0(...)

define i1* @lookup_bit(i32* %q, i32 %bitno) readnone nounwind {
	%tmp = ptrtoint i32* %q to i32
	%tmp2 = lshr i32 %tmp, %bitno
	%bit = and i32 %tmp2, 1
	%lookup = getelementptr [2 x i1]* @lookup_table, i32 0, i32 %bit
	ret i1* %lookup
}

; CHECK: define i1 @c7(i32* %q, i32 %bitno)
define i1 @c7(i32* %q, i32 %bitno) {
	%ptr = call i1* @lookup_bit(i32* %q, i32 %bitno)
	%val = load i1* %ptr
	ret i1 %val
}


; CHECK: define i32 @nc1(i32* %q, i32* nocapture %p, i1 %b)
define i32 @nc1(i32* %q, i32* %p, i1 %b) {
e:
	br label %l
l:
	%x = phi i32* [ %p, %e ]
	%y = phi i32* [ %q, %e ]
	%tmp = bitcast i32* %x to i32*		; <i32*> [#uses=2]
	%tmp2 = select i1 %b, i32* %tmp, i32* %y
	%val = load i32* %tmp2		; <i32> [#uses=1]
	store i32 0, i32* %tmp
	store i32* %y, i32** @g
	ret i32 %val
}

; CHECK: define void @nc2(i32* nocapture %p, i32* %q)
define void @nc2(i32* %p, i32* %q) {
	%1 = call i32 @nc1(i32* %q, i32* %p, i1 0)		; <i32> [#uses=0]
	ret void
}

; CHECK: define void @nc3(void ()* nocapture %p)
define void @nc3(void ()* %p) {
	call void %p()
	ret void
}

declare void @external(i8*) readonly nounwind
; CHECK: define void @nc4(i8* nocapture %p)
define void @nc4(i8* %p) {
	call void @external(i8* %p)
	ret void
}

; CHECK: define void @nc5(void (i8*)* nocapture %f, i8* nocapture %p)
define void @nc5(void (i8*)* %f, i8* %p) {
	call void %f(i8* %p) readonly nounwind
	call void %f(i8* nocapture %p)
	ret void
}

; CHECK: define void @test1_1(i8* nocapture %x1_1, i8* %y1_1)
define void @test1_1(i8* %x1_1, i8* %y1_1) {
  call i8* @test1_2(i8* %x1_1, i8* %y1_1)
  store i32* null, i32** @g
  ret void
}

; CHECK: define i8* @test1_2(i8* nocapture %x1_2, i8* %y1_2)
define i8* @test1_2(i8* %x1_2, i8* %y1_2) {
  call void @test1_1(i8* %x1_2, i8* %y1_2)
  store i32* null, i32** @g
  ret i8* %y1_2
}

; CHECK: define void @test2(i8* nocapture %x2)
define void @test2(i8* %x2) {
  call void @test2(i8* %x2)
  store i32* null, i32** @g
  ret void
}

; CHECK: define void @test3(i8* nocapture %x3, i8* nocapture %y3, i8* nocapture %z3)
define void @test3(i8* %x3, i8* %y3, i8* %z3) {
  call void @test3(i8* %z3, i8* %y3, i8* %x3)
  store i32* null, i32** @g
  ret void
}

; CHECK: define void @test4_1(i8* %x4_1)
define void @test4_1(i8* %x4_1) {
  call i8* @test4_2(i8* %x4_1, i8* %x4_1, i8* %x4_1)
  store i32* null, i32** @g
  ret void
}

; CHECK: define i8* @test4_2(i8* nocapture %x4_2, i8* %y4_2, i8* nocapture %z4_2)
define i8* @test4_2(i8* %x4_2, i8* %y4_2, i8* %z4_2) {
  call void @test4_1(i8* null)
  store i32* null, i32** @g
  ret i8* %y4_2
}

declare i8* @test5_1(i8* %x5_1)

; CHECK: define void @test5_2(i8* %x5_2)
define void @test5_2(i8* %x5_2) {
  call i8* @test5_1(i8* %x5_2)
  store i32* null, i32** @g
  ret void
}

declare void @test6_1(i8* %x6_1, i8* nocapture %y6_1, ...)

; CHECK: define void @test6_2(i8* %x6_2, i8* nocapture %y6_2, i8* %z6_2)
define void @test6_2(i8* %x6_2, i8* %y6_2, i8* %z6_2) {
  call void (i8*, i8*, ...)* @test6_1(i8* %x6_2, i8* %y6_2, i8* %z6_2)
  store i32* null, i32** @g
  ret void
}

