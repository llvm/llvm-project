; RUN: opt -instcombine -instcombine-infinite-loop-threshold=3 -S < %s | FileCheck %s

%struct1 = type { %struct2*, i32, i32, i32 }
%struct2 = type { i32, i32 }
%struct3 = type { i32, %struct4, %struct4 }
%struct4 = type { %struct2, %struct2 }

define i32 @test1(%struct1* %dm, i1 %tmp4, i64 %tmp9, i64 %tmp19) {
bb:
  %tmp = getelementptr inbounds %struct1, %struct1* %dm, i64 0, i32 0
  %tmp1 = load %struct2*, %struct2** %tmp, align 8
  br i1 %tmp4, label %bb1, label %bb2

bb1:
  %tmp10 = getelementptr inbounds %struct2, %struct2* %tmp1, i64 %tmp9
  %tmp11 = getelementptr inbounds %struct2, %struct2* %tmp10, i64 0, i32 0
  store i32 0, i32* %tmp11, align 4
  br label %bb3

bb2:
  %tmp20 = getelementptr inbounds %struct2, %struct2* %tmp1, i64 %tmp19
  %tmp21 = getelementptr inbounds %struct2, %struct2* %tmp20, i64 0, i32 0
  store i32 0, i32* %tmp21, align 4
  br label %bb3

bb3:
  %phi = phi %struct2* [ %tmp10, %bb1 ], [ %tmp20, %bb2 ]
  %tmp24 = getelementptr inbounds %struct2, %struct2* %phi, i64 0, i32 1
  %tmp25 = load i32, i32* %tmp24, align 4
  ret i32 %tmp25

; CHECK-LABEL: @test1(
; CHECK: getelementptr inbounds %struct2, %struct2* %tmp1, i64 %tmp9, i32 0
; CHECK: getelementptr inbounds %struct2, %struct2* %tmp1, i64 %tmp19, i32 0
; CHECK: %[[PHI:[0-9A-Za-z]+]] = phi i64 [ %tmp9, %bb1 ], [ %tmp19, %bb2 ]
; CHECK: getelementptr inbounds %struct2, %struct2* %tmp1, i64 %[[PHI]], i32 1

}

define i32 @test2(%struct1* %dm, i1 %tmp4, i64 %tmp9, i64 %tmp19) {
bb:
  %tmp = getelementptr inbounds %struct1, %struct1* %dm, i64 0, i32 0
  %tmp1 = load %struct2*, %struct2** %tmp, align 8
  %tmp10 = getelementptr inbounds %struct2, %struct2* %tmp1, i64 %tmp9
  %tmp11 = getelementptr inbounds %struct2, %struct2* %tmp10, i64 0, i32 0
  store i32 0, i32* %tmp11, align 4
  %tmp20 = getelementptr inbounds %struct2, %struct2* %tmp1, i64 %tmp19
  %tmp21 = getelementptr inbounds %struct2, %struct2* %tmp20, i64 0, i32 0
  store i32 0, i32* %tmp21, align 4
  %tmp24 = getelementptr inbounds %struct2, %struct2* %tmp10, i64 0, i32 1
  %tmp25 = load i32, i32* %tmp24, align 4
  ret i32 %tmp25

; CHECK-LABEL: @test2(
; CHECK: getelementptr inbounds %struct2, %struct2* %tmp1, i64 %tmp9, i32 0
; CHECK: getelementptr inbounds %struct2, %struct2* %tmp1, i64 %tmp19, i32 0
; CHECK: getelementptr inbounds %struct2, %struct2* %tmp1, i64 %tmp9, i32 1
}

; Check that instcombine doesn't insert GEPs before landingpad.

define i32 @test3(%struct3* %dm, i1 %tmp4, i64 %tmp9, i64 %tmp19, i64 %tmp20, i64 %tmp21) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
bb:
  %tmp = getelementptr inbounds %struct3, %struct3* %dm, i64 0
  br i1 %tmp4, label %bb1, label %bb2

bb1:
  %tmp1 = getelementptr inbounds %struct3, %struct3* %tmp, i64 %tmp19, i32 1
  %tmp11 = getelementptr inbounds %struct4, %struct4* %tmp1, i64 0, i32 0, i32 0
  store i32 0, i32* %tmp11, align 4
  br label %bb3

bb2:
  %tmp2 = getelementptr inbounds %struct3, %struct3* %tmp, i64 %tmp20, i32 1
  %tmp12 = getelementptr inbounds %struct4, %struct4* %tmp2, i64 0, i32 0, i32 1
  store i32 0, i32* %tmp12, align 4
  br label %bb3

bb3:
  %phi = phi %struct4* [ %tmp1, %bb1 ], [ %tmp2, %bb2 ]
  %tmp22 = invoke i32 @foo1(i32 11) to label %bb4 unwind label %bb5

bb4:
  ret i32 0

bb5:
  %tmp27 = landingpad { i8*, i32 } catch i8* bitcast (i8** @_ZTIi to i8*)
  %tmp34 = getelementptr inbounds %struct4, %struct4* %phi, i64 %tmp21, i32 1
  %tmp35 = getelementptr inbounds %struct2, %struct2* %tmp34, i64 0, i32 1
  %tmp25 = load i32, i32* %tmp35, align 4
  ret i32 %tmp25

; CHECK-LABEL: @test3(
; CHECK: bb5:
; CHECK-NEXT: {{.*}}landingpad { i8*, i32 }
}

@_ZTIi = external constant i8*
declare i32 @__gxx_personality_v0(...)
declare i32 @foo1(i32)


; Check that instcombine doesn't fold GEPs into themselves through a loop
; back-edge.

define i8* @test4(i32 %value, i8* %buffer) {
entry:
  %incptr = getelementptr inbounds i8, i8* %buffer, i64 1
  %cmp = icmp ugt i32 %value, 127
  br i1 %cmp, label %loop.header, label %exit

loop.header:
  br label %loop.body

loop.body:
  %loopptr = phi i8* [ %incptr, %loop.header ], [ %incptr2, %loop.body ]
  %newval = phi i32 [ %value, %loop.header ], [ %shr, %loop.body ]
  %shr = lshr i32 %newval, 7
  %incptr2 = getelementptr inbounds i8, i8* %loopptr, i64 1
  %cmp2 = icmp ugt i32 %shr, 127
  br i1 %cmp2, label %loop.body, label %loop.exit

loop.exit:
  %exitptr = phi i8* [ %incptr2, %loop.body ]
  br label %exit

exit:
  %ptr2 = phi i8* [ %exitptr, %loop.exit ], [ %incptr, %entry ]
  %incptr3 = getelementptr inbounds i8, i8* %ptr2, i64 1
  ret i8* %incptr3

; CHECK-LABEL: @test4(
; CHECK: loop.body:
; CHECK: getelementptr{{.*}}i64 1
; CHECK: exit:
}

@.str.4 = external unnamed_addr constant [100 x i8], align 1

; Instcombine shouldn't add new PHI nodes while folding GEPs if that will leave
; old PHI nodes behind as this is not clearly beneficial.
; CHECK-LABEL: @test5(
define void @test5(i16 *%idx, i8 **%in) #0 {
entry:
  %0 = load i8*, i8** %in
  %incdec.ptr = getelementptr inbounds i8, i8* %0, i32 1
  %1 = load i8, i8* %incdec.ptr, align 1
  %cmp23 = icmp eq i8 %1, 54
  br i1 %cmp23, label %while.cond, label %if.then.25

if.then.25:
  call void @g(i8* getelementptr inbounds ([100 x i8], [100 x i8]* @.str.4, i32 0, i32 0))
  br label %while.cond

while.cond:
; CHECK-LABEL: while.cond
; CHECK-NOT: phi i8* [ %0, %entry ], [ %Ptr, %while.body ], [ %0, %if.then.25 ]
  %Ptr = phi i8* [ %incdec.ptr, %entry ], [ %incdec.ptr32, %while.body], [%incdec.ptr, %if.then.25 ]
  %2 = load i8, i8* %Ptr
  %and = and i8 %2, 64
  %lnot = icmp eq i8 %and, 0
  br i1 %lnot, label %while.body, label %while.cond.33

while.body:
  %incdec.ptr32 = getelementptr inbounds i8, i8* %Ptr, i32 1
  br label %while.cond

while.cond.33:
  %incdec.ptr34 = getelementptr inbounds i8, i8* %Ptr, i32 1
  br label %while.cond.57

while.cond.57:
  %3 = load i8, i8* %incdec.ptr34, align 1
  %conv59 = zext i8 %3 to i32
  %arrayidx61 = getelementptr inbounds i16, i16* %idx, i32 %conv59
  %4 = load i16, i16* %arrayidx61, align 2
  %and63 = and i16 %4, 2048
  %tobool64 = icmp eq i16 %and63, 0
  br i1 %tobool64, label %while.cond.73, label %while.cond.57

while.cond.73:
  br label %while.cond.73

}

declare void @g(i8*)
