; RUN: opt -function-attrs -S < %s | FileCheck %s --check-prefixes=FNATTR
; RUN: opt -passes=function-attrs -S < %s | FileCheck %s --check-prefixes=FNATTR

@g = global ptr null		; <ptr> [#uses=1]

; FNATTR: define ptr @c1(ptr readnone returned %q)
define ptr @c1(ptr %q) {
	ret ptr %q
}

; FNATTR: define void @c2(ptr %q)
; It would also be acceptable to mark %q as readnone. Update @c3 too.
define void @c2(ptr %q) {
	store ptr %q, ptr @g
	ret void
}

; FNATTR: define void @c3(ptr %q)
define void @c3(ptr %q) {
	call void @c2(ptr %q)
	ret void
}

; FNATTR: define i1 @c4(ptr %q, i32 %bitno)
define i1 @c4(ptr %q, i32 %bitno) {
	%tmp = ptrtoint ptr %q to i32
	%tmp2 = lshr i32 %tmp, %bitno
	%bit = trunc i32 %tmp2 to i1
	br i1 %bit, label %l1, label %l0
l0:
	ret i1 0 ; escaping value not caught by def-use chaining.
l1:
	ret i1 1 ; escaping value not caught by def-use chaining.
}

; c4b is c4 but without the escaping part
; FNATTR: define i1 @c4b(ptr %q, i32 %bitno)
define i1 @c4b(ptr %q, i32 %bitno) {
	%tmp = ptrtoint ptr %q to i32
	%tmp2 = lshr i32 %tmp, %bitno
	%bit = trunc i32 %tmp2 to i1
	br i1 %bit, label %l1, label %l0
l0:
	ret i1 0 ; not escaping!
l1:
	ret i1 0 ; not escaping!
}

@lookup_table = global [2 x i1] [ i1 0, i1 1 ]

; FNATTR: define i1 @c5(ptr %q, i32 %bitno)
define i1 @c5(ptr %q, i32 %bitno) {
	%tmp = ptrtoint ptr %q to i32
	%tmp2 = lshr i32 %tmp, %bitno
	%bit = and i32 %tmp2, 1
        ; subtle escape mechanism follows
	%lookup = getelementptr [2 x i1], ptr @lookup_table, i32 0, i32 %bit
	%val = load i1, ptr %lookup
	ret i1 %val
}

declare void @throw_if_bit_set(ptr, i8) readonly

; FNATTR: define i1 @c6(ptr readonly %q, i8 %bit)
define i1 @c6(ptr %q, i8 %bit) personality ptr @__gxx_personality_v0 {
	invoke void @throw_if_bit_set(ptr %q, i8 %bit)
		to label %ret0 unwind label %ret1
ret0:
	ret i1 0
ret1:
        %exn = landingpad {ptr, i32}
                 cleanup
	ret i1 1
}

declare i32 @__gxx_personality_v0(...)

define ptr @lookup_bit(ptr %q, i32 %bitno) readnone nounwind {
	%tmp = ptrtoint ptr %q to i32
	%tmp2 = lshr i32 %tmp, %bitno
	%bit = and i32 %tmp2, 1
	%lookup = getelementptr [2 x i1], ptr @lookup_table, i32 0, i32 %bit
	ret ptr %lookup
}

; FNATTR: define i1 @c7(ptr readonly %q, i32 %bitno)
define i1 @c7(ptr %q, i32 %bitno) {
	%ptr = call ptr @lookup_bit(ptr %q, i32 %bitno)
	%val = load i1, ptr %ptr
	ret i1 %val
}


; FNATTR: define i32 @nc1(ptr %q, ptr nocapture %p, i1 %b)
define i32 @nc1(ptr %q, ptr %p, i1 %b) {
e:
	br label %l
l:
	%x = phi ptr [ %p, %e ]
	%y = phi ptr [ %q, %e ]
	%tmp2 = select i1 %b, ptr %x, ptr %y
	%val = load i32, ptr %tmp2		; <i32> [#uses=1]
	store i32 0, ptr %x
	store ptr %y, ptr @g
	ret i32 %val
}

; FNATTR: define i32 @nc1_addrspace(ptr %q, ptr addrspace(1) nocapture %p, i1 %b)
define i32 @nc1_addrspace(ptr %q, ptr addrspace(1) %p, i1 %b) {
e:
	br label %l
l:
	%x = phi ptr addrspace(1) [ %p, %e ]
	%y = phi ptr [ %q, %e ]
	%tmp = addrspacecast ptr addrspace(1) %x to ptr		; <ptr> [#uses=2]
	%tmp2 = select i1 %b, ptr %tmp, ptr %y
	%val = load i32, ptr %tmp2		; <i32> [#uses=1]
	store i32 0, ptr %tmp
	store ptr %y, ptr @g
	ret i32 %val
}

; FNATTR: define void @nc2(ptr nocapture %p, ptr %q)
define void @nc2(ptr %p, ptr %q) {
	%1 = call i32 @nc1(ptr %q, ptr %p, i1 0)		; <i32> [#uses=0]
	ret void
}


; FNATTR: define void @nc3(ptr nocapture readonly %p)
define void @nc3(ptr %p) {
	call void %p()
	ret void
}

declare void @external(ptr) readonly nounwind
; FNATTR: define void @nc4(ptr nocapture readonly %p)
define void @nc4(ptr %p) {
	call void @external(ptr %p)
	ret void
}

; FNATTR: define void @nc5(ptr nocapture readonly %f, ptr nocapture %p)
define void @nc5(ptr %f, ptr %p) {
	call void %f(ptr %p) readonly nounwind
	call void %f(ptr nocapture %p)
	ret void
}

; FNATTR:     define void @test1_1(ptr nocapture readnone %x1_1, ptr %y1_1, i1 %c)
; It would be acceptable to add readnone to %y1_1 and %y1_2.
define void @test1_1(ptr %x1_1, ptr %y1_1, i1 %c) {
  call ptr @test1_2(ptr %x1_1, ptr %y1_1, i1 %c)
  store ptr null, ptr @g
  ret void
}

; FNATTR: define ptr @test1_2(ptr nocapture readnone %x1_2, ptr returned %y1_2, i1 %c)
define ptr @test1_2(ptr %x1_2, ptr %y1_2, i1 %c) {
  br i1 %c, label %t, label %f
t:
  call void @test1_1(ptr %x1_2, ptr %y1_2, i1 %c)
  store ptr null, ptr @g
  br label %f
f:
  ret ptr %y1_2
}

; FNATTR: define void @test2(ptr nocapture readnone %x2)
define void @test2(ptr %x2) {
  call void @test2(ptr %x2)
  store ptr null, ptr @g
  ret void
}

; FNATTR: define void @test3(ptr nocapture readnone %x3, ptr nocapture readnone %y3, ptr nocapture readnone %z3)
define void @test3(ptr %x3, ptr %y3, ptr %z3) {
  call void @test3(ptr %z3, ptr %y3, ptr %x3)
  store ptr null, ptr @g
  ret void
}

; FNATTR: define void @test4_1(ptr %x4_1, i1 %c)
define void @test4_1(ptr %x4_1, i1 %c) {
  call ptr @test4_2(ptr %x4_1, ptr %x4_1, ptr %x4_1, i1 %c)
  store ptr null, ptr @g
  ret void
}

; FNATTR: define ptr @test4_2(ptr nocapture readnone %x4_2, ptr readnone returned %y4_2, ptr nocapture readnone %z4_2, i1 %c)
define ptr @test4_2(ptr %x4_2, ptr %y4_2, ptr %z4_2, i1 %c) {
  br i1 %c, label %t, label %f
t:
  call void @test4_1(ptr null, i1 %c)
  store ptr null, ptr @g
  br label %f
f:
  ret ptr %y4_2
}

declare ptr @test5_1(ptr %x5_1)

; FNATTR: define void @test5_2(ptr %x5_2)
define void @test5_2(ptr %x5_2) {
  call ptr @test5_1(ptr %x5_2)
  store ptr null, ptr @g
  ret void
}

declare void @test6_1(ptr %x6_1, ptr nocapture %y6_1, ...)

; FNATTR: define void @test6_2(ptr %x6_2, ptr nocapture %y6_2, ptr %z6_2)
define void @test6_2(ptr %x6_2, ptr %y6_2, ptr %z6_2) {
  call void (ptr, ptr, ...) @test6_1(ptr %x6_2, ptr %y6_2, ptr %z6_2)
  store ptr null, ptr @g
  ret void
}

; FNATTR: define void @test_cmpxchg(ptr nocapture %p)
define void @test_cmpxchg(ptr %p) {
  cmpxchg ptr %p, i32 0, i32 1 acquire monotonic
  ret void
}

; FNATTR: define void @test_cmpxchg_ptr(ptr nocapture %p, ptr %q)
define void @test_cmpxchg_ptr(ptr %p, ptr %q) {
  cmpxchg ptr %p, ptr null, ptr %q acquire monotonic
  ret void
}

; FNATTR: define void @test_atomicrmw(ptr nocapture %p)
define void @test_atomicrmw(ptr %p) {
  atomicrmw add ptr %p, i32 1 seq_cst
  ret void
}

; FNATTR: define void @test_volatile(ptr %x)
define void @test_volatile(ptr %x) {
entry:
  %gep = getelementptr i32, ptr %x, i64 1
  store volatile i32 0, ptr %gep, align 4
  ret void
}

; FNATTR: nocaptureLaunder(ptr nocapture %p)
define void @nocaptureLaunder(ptr %p) {
entry:
  %b = call ptr @llvm.launder.invariant.group.p0(ptr %p)
  store i8 42, ptr %b
  ret void
}

@g2 = global ptr null
; FNATTR: define void @captureLaunder(ptr %p)
define void @captureLaunder(ptr %p) {
  %b = call ptr @llvm.launder.invariant.group.p0(ptr %p)
  store ptr %b, ptr @g2
  ret void
}

; FNATTR: @nocaptureStrip(ptr nocapture writeonly %p)
define void @nocaptureStrip(ptr %p) {
entry:
  %b = call ptr @llvm.strip.invariant.group.p0(ptr %p)
  store i8 42, ptr %b
  ret void
}

@g3 = global ptr null
; FNATTR: define void @captureStrip(ptr %p)
define void @captureStrip(ptr %p) {
  %b = call ptr @llvm.strip.invariant.group.p0(ptr %p)
  store ptr %b, ptr @g3
  ret void
}

; FNATTR: define i1 @captureICmp(ptr readnone %x)
define i1 @captureICmp(ptr %x) {
  %1 = icmp eq ptr %x, null
  ret i1 %1
}

; FNATTR: define i1 @captureICmpRev(ptr readnone %x)
define i1 @captureICmpRev(ptr %x) {
  %1 = icmp eq ptr null, %x
  ret i1 %1
}

; FNATTR: define i1 @nocaptureInboundsGEPICmp(ptr nocapture readnone %x)
define i1 @nocaptureInboundsGEPICmp(ptr %x) {
  %1 = getelementptr inbounds i32, ptr %x, i32 5
  %2 = icmp eq ptr %1, null
  ret i1 %2
}

; FNATTR: define i1 @nocaptureInboundsGEPICmpRev(ptr nocapture readnone %x)
define i1 @nocaptureInboundsGEPICmpRev(ptr %x) {
  %1 = getelementptr inbounds i32, ptr %x, i32 5
  %2 = icmp eq ptr null, %1
  ret i1 %2
}

; FNATTR: define i1 @nocaptureDereferenceableOrNullICmp(ptr nocapture readnone dereferenceable_or_null(4) %x)
define i1 @nocaptureDereferenceableOrNullICmp(ptr dereferenceable_or_null(4) %x) {
  %1 = icmp eq ptr %x, null
  ret i1 %1
}

; FNATTR: define i1 @captureDereferenceableOrNullICmp(ptr readnone dereferenceable_or_null(4) %x)
define i1 @captureDereferenceableOrNullICmp(ptr dereferenceable_or_null(4) %x) null_pointer_is_valid {
  %1 = icmp eq ptr %x, null
  ret i1 %1
}

declare void @capture(ptr)

; FNATTR: define void @nocapture_fptr(ptr nocapture readonly %f, ptr %p)
define void @nocapture_fptr(ptr %f, ptr %p) {
  %res = call ptr %f(ptr %p)
  call void @capture(ptr %res)
  ret void
}

; FNATTR: define void @recurse_fptr(ptr nocapture readonly %f, ptr %p)
define void @recurse_fptr(ptr %f, ptr %p) {
  %res = call ptr %f(ptr %p)
  store i8 0, ptr %res
  ret void
}

; FNATTR: define void @readnone_indirec(ptr nocapture readonly %f, ptr readnone %p)
define void @readnone_indirec(ptr %f, ptr %p) {
  call void %f(ptr %p) readnone
  ret void
}


declare ptr @llvm.launder.invariant.group.p0(ptr)
declare ptr @llvm.strip.invariant.group.p0(ptr)
