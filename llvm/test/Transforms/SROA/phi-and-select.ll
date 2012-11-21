; RUN: opt < %s -sroa -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n8:16:32:64"

define i32 @test1() {
; CHECK: @test1
entry:
	%a = alloca [2 x i32]
; CHECK-NOT: alloca

  %a0 = getelementptr [2 x i32]* %a, i64 0, i32 0
  %a1 = getelementptr [2 x i32]* %a, i64 0, i32 1
	store i32 0, i32* %a0
	store i32 1, i32* %a1
	%v0 = load i32* %a0
	%v1 = load i32* %a1
; CHECK-NOT: store
; CHECK-NOT: load

	%cond = icmp sle i32 %v0, %v1
	br i1 %cond, label %then, label %exit

then:
	br label %exit

exit:
	%phi = phi i32* [ %a1, %then ], [ %a0, %entry ]
; CHECK: phi i32 [ 1, %{{.*}} ], [ 0, %{{.*}} ]

	%result = load i32* %phi
	ret i32 %result
}

define i32 @test2() {
; CHECK: @test2
entry:
	%a = alloca [2 x i32]
; CHECK-NOT: alloca

  %a0 = getelementptr [2 x i32]* %a, i64 0, i32 0
  %a1 = getelementptr [2 x i32]* %a, i64 0, i32 1
	store i32 0, i32* %a0
	store i32 1, i32* %a1
	%v0 = load i32* %a0
	%v1 = load i32* %a1
; CHECK-NOT: store
; CHECK-NOT: load

	%cond = icmp sle i32 %v0, %v1
	%select = select i1 %cond, i32* %a1, i32* %a0
; CHECK: select i1 %{{.*}}, i32 1, i32 0

	%result = load i32* %select
	ret i32 %result
}

define i32 @test3(i32 %x) {
; CHECK: @test3
entry:
	%a = alloca [2 x i32]
; CHECK-NOT: alloca

  ; Note that we build redundant GEPs here to ensure that having different GEPs
  ; into the same alloca partation continues to work with PHI speculation. This
  ; was the underlying cause of PR13926.
  %a0 = getelementptr [2 x i32]* %a, i64 0, i32 0
  %a0b = getelementptr [2 x i32]* %a, i64 0, i32 0
  %a1 = getelementptr [2 x i32]* %a, i64 0, i32 1
  %a1b = getelementptr [2 x i32]* %a, i64 0, i32 1
	store i32 0, i32* %a0
	store i32 1, i32* %a1
; CHECK-NOT: store

  switch i32 %x, label %bb0 [ i32 1, label %bb1
                              i32 2, label %bb2
                              i32 3, label %bb3
                              i32 4, label %bb4
                              i32 5, label %bb5
                              i32 6, label %bb6
                              i32 7, label %bb7 ]

bb0:
	br label %exit
bb1:
	br label %exit
bb2:
	br label %exit
bb3:
	br label %exit
bb4:
	br label %exit
bb5:
	br label %exit
bb6:
	br label %exit
bb7:
	br label %exit

exit:
	%phi = phi i32* [ %a1, %bb0 ], [ %a0, %bb1 ], [ %a0, %bb2 ], [ %a1, %bb3 ],
                  [ %a1b, %bb4 ], [ %a0b, %bb5 ], [ %a0b, %bb6 ], [ %a1b, %bb7 ]
; CHECK: phi i32 [ 1, %{{.*}} ], [ 0, %{{.*}} ], [ 0, %{{.*}} ], [ 1, %{{.*}} ], [ 1, %{{.*}} ], [ 0, %{{.*}} ], [ 0, %{{.*}} ], [ 1, %{{.*}} ]

	%result = load i32* %phi
	ret i32 %result
}

define i32 @test4() {
; CHECK: @test4
entry:
	%a = alloca [2 x i32]
; CHECK-NOT: alloca

  %a0 = getelementptr [2 x i32]* %a, i64 0, i32 0
  %a1 = getelementptr [2 x i32]* %a, i64 0, i32 1
	store i32 0, i32* %a0
	store i32 1, i32* %a1
	%v0 = load i32* %a0
	%v1 = load i32* %a1
; CHECK-NOT: store
; CHECK-NOT: load

	%cond = icmp sle i32 %v0, %v1
	%select = select i1 %cond, i32* %a0, i32* %a0
; CHECK-NOT: select

	%result = load i32* %select
	ret i32 %result
; CHECK: ret i32 0
}

define i32 @test5(i32* %b) {
; CHECK: @test5
entry:
	%a = alloca [2 x i32]
; CHECK-NOT: alloca

  %a1 = getelementptr [2 x i32]* %a, i64 0, i32 1
	store i32 1, i32* %a1
; CHECK-NOT: store

	%select = select i1 true, i32* %a1, i32* %b
; CHECK-NOT: select

	%result = load i32* %select
; CHECK-NOT: load

	ret i32 %result
; CHECK: ret i32 1
}

declare void @f(i32*, i32*)

define i32 @test6(i32* %b) {
; CHECK: @test6
entry:
	%a = alloca [2 x i32]
  %c = alloca i32
; CHECK-NOT: alloca

  %a1 = getelementptr [2 x i32]* %a, i64 0, i32 1
	store i32 1, i32* %a1

	%select = select i1 true, i32* %a1, i32* %b
	%select2 = select i1 false, i32* %a1, i32* %b
  %select3 = select i1 false, i32* %c, i32* %b
; CHECK: %[[select2:.*]] = select i1 false, i32* undef, i32* %b
; CHECK: %[[select3:.*]] = select i1 false, i32* undef, i32* %b

  ; Note, this would potentially escape the alloca pointer except for the
  ; constant folding of the select.
  call void @f(i32* %select2, i32* %select3)
; CHECK: call void @f(i32* %[[select2]], i32* %[[select3]])


	%result = load i32* %select
; CHECK-NOT: load

  %dead = load i32* %c

	ret i32 %result
; CHECK: ret i32 1
}

define i32 @test7() {
; CHECK: @test7
; CHECK-NOT: alloca

entry:
  %X = alloca i32
  br i1 undef, label %good, label %bad

good:
  %Y1 = getelementptr i32* %X, i64 0
  store i32 0, i32* %Y1
  br label %exit

bad:
  %Y2 = getelementptr i32* %X, i64 1
  store i32 0, i32* %Y2
  br label %exit

exit:
	%P = phi i32* [ %Y1, %good ], [ %Y2, %bad ]
; CHECK: %[[phi:.*]] = phi i32 [ 0, %good ],
  %Z2 = load i32* %P
  ret i32 %Z2
; CHECK: ret i32 %[[phi]]
}

define i32 @test8(i32 %b, i32* %ptr) {
; Ensure that we rewrite allocas to the used type when that use is hidden by
; a PHI that can be speculated.
; CHECK: @test8
; CHECK-NOT: alloca
; CHECK-NOT: load
; CHECK: %[[value:.*]] = load i32* %ptr
; CHECK-NOT: load
; CHECK: %[[result:.*]] = phi i32 [ undef, %else ], [ %[[value]], %then ]
; CHECK-NEXT: ret i32 %[[result]]

entry:
  %f = alloca float
  %test = icmp ne i32 %b, 0
  br i1 %test, label %then, label %else

then:
  br label %exit

else:
  %bitcast = bitcast float* %f to i32*
  br label %exit

exit:
  %phi = phi i32* [ %bitcast, %else ], [ %ptr, %then ]
  %loaded = load i32* %phi, align 4
  ret i32 %loaded
}

define i32 @test9(i32 %b, i32* %ptr) {
; Same as @test8 but for a select rather than a PHI node.
; CHECK: @test9
; CHECK-NOT: alloca
; CHECK-NOT: load
; CHECK: %[[value:.*]] = load i32* %ptr
; CHECK-NOT: load
; CHECK: %[[result:.*]] = select i1 %{{.*}}, i32 undef, i32 %[[value]]
; CHECK-NEXT: ret i32 %[[result]]

entry:
  %f = alloca float
  store i32 0, i32* %ptr
  %test = icmp ne i32 %b, 0
  %bitcast = bitcast float* %f to i32*
  %select = select i1 %test, i32* %bitcast, i32* %ptr
  %loaded = load i32* %select, align 4
  ret i32 %loaded
}

define float @test10(i32 %b, float* %ptr) {
; Don't try to promote allocas which are not elligible for it even after
; rewriting due to the necessity of inserting bitcasts when speculating a PHI
; node.
; CHECK: @test10
; CHECK: %[[alloca:.*]] = alloca
; CHECK: %[[argvalue:.*]] = load float* %ptr
; CHECK: %[[cast:.*]] = bitcast double* %[[alloca]] to float*
; CHECK: %[[allocavalue:.*]] = load float* %[[cast]]
; CHECK: %[[result:.*]] = phi float [ %[[allocavalue]], %else ], [ %[[argvalue]], %then ]
; CHECK-NEXT: ret float %[[result]]

entry:
  %f = alloca double
  store double 0.0, double* %f
  %test = icmp ne i32 %b, 0
  br i1 %test, label %then, label %else

then:
  br label %exit

else:
  %bitcast = bitcast double* %f to float*
  br label %exit

exit:
  %phi = phi float* [ %bitcast, %else ], [ %ptr, %then ]
  %loaded = load float* %phi, align 4
  ret float %loaded
}

define float @test11(i32 %b, float* %ptr) {
; Same as @test10 but for a select rather than a PHI node.
; CHECK: @test11
; CHECK: %[[alloca:.*]] = alloca
; CHECK: %[[cast:.*]] = bitcast double* %[[alloca]] to float*
; CHECK: %[[allocavalue:.*]] = load float* %[[cast]]
; CHECK: %[[argvalue:.*]] = load float* %ptr
; CHECK: %[[result:.*]] = select i1 %{{.*}}, float %[[allocavalue]], float %[[argvalue]]
; CHECK-NEXT: ret float %[[result]]

entry:
  %f = alloca double
  store double 0.0, double* %f
  store float 0.0, float* %ptr
  %test = icmp ne i32 %b, 0
  %bitcast = bitcast double* %f to float*
  %select = select i1 %test, float* %bitcast, float* %ptr
  %loaded = load float* %select, align 4
  ret float %loaded
}

define i32 @test12(i32 %x, i32* %p) {
; Ensure we don't crash or fail to nuke dead selects of allocas if no load is
; never found.
; CHECK: @test12
; CHECK-NOT: alloca
; CHECK-NOT: select
; CHECK: ret i32 %x

entry:
  %a = alloca i32
  store i32 %x, i32* %a
  %dead = select i1 undef, i32* %a, i32* %p
  %load = load i32* %a
  ret i32 %load
}

define i32 @test13(i32 %x, i32* %p) {
; Ensure we don't crash or fail to nuke dead phis of allocas if no load is ever
; found.
; CHECK: @test13
; CHECK-NOT: alloca
; CHECK-NOT: phi
; CHECK: ret i32 %x

entry:
  %a = alloca i32
  store i32 %x, i32* %a
  br label %loop

loop:
  %phi = phi i32* [ %p, %entry ], [ %a, %loop ]
  br i1 undef, label %loop, label %exit

exit:
  %load = load i32* %a
  ret i32 %load
}

define i32 @PR13905() {
; Check a pattern where we have a chain of dead phi nodes to ensure they are
; deleted and promotion can proceed.
; CHECK: @PR13905
; CHECK-NOT: alloca i32
; CHECK: ret i32 undef

entry:
  %h = alloca i32
  store i32 0, i32* %h
  br i1 undef, label %loop1, label %exit

loop1:
  %phi1 = phi i32* [ null, %entry ], [ %h, %loop1 ], [ %h, %loop2 ]
  br i1 undef, label %loop1, label %loop2

loop2:
  br i1 undef, label %loop1, label %exit

exit:
  %phi2 = phi i32* [ %phi1, %loop2 ], [ null, %entry ]
  ret i32 undef
}

define i32 @PR13906() {
; Another pattern which can lead to crashes due to failing to clear out dead
; PHI nodes or select nodes. This triggers subtly differently from the above
; cases because the PHI node is (recursively) alive, but the select is dead.
; CHECK: @PR13906
; CHECK-NOT: alloca

entry:
  %c = alloca i32
  store i32 0, i32* %c
  br label %for.cond

for.cond:
  %d.0 = phi i32* [ undef, %entry ], [ %c, %if.then ], [ %d.0, %for.cond ]
  br i1 undef, label %if.then, label %for.cond

if.then:
  %tmpcast.d.0 = select i1 undef, i32* %c, i32* %d.0
  br label %for.cond
}

define i64 @PR14132(i1 %flag) {
; CHECK: @PR14132
; Here we form a PHI-node by promoting the pointer alloca first, and then in
; order to promote the other two allocas, we speculate the load of the
; now-phi-node-pointer. In doing so we end up loading a 64-bit value from an i8
; alloca, which is completely bogus. However, we were asserting on trying to
; rewrite it. Now it is replaced with undef. Eventually we may replace it with
; unrechable and even the CFG will go away here.
entry:
  %a = alloca i64
  %b = alloca i8
  %ptr = alloca i64*
; CHECK-NOT: alloca

  %ptr.cast = bitcast i64** %ptr to i8**
  store i64 0, i64* %a
  store i8 1, i8* %b
  store i64* %a, i64** %ptr
  br i1 %flag, label %if.then, label %if.end

if.then:
  store i8* %b, i8** %ptr.cast
  br label %if.end

if.end:
  %tmp = load i64** %ptr
  %result = load i64* %tmp
; CHECK-NOT: store
; CHECK-NOT: load
; CHECK: %[[result:.*]] = phi i64 [ undef, %if.then ], [ 0, %entry ]

  ret i64 %result
; CHECK-NEXT: ret i64 %[[result]]
}
