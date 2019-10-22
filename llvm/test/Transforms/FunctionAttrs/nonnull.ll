; RUN: opt -S -functionattrs -enable-nonnull-arg-prop %s | FileCheck %s --check-prefixes=BOTH,FNATTR
; RUN: opt -S -passes=function-attrs -enable-nonnull-arg-prop %s | FileCheck %s --check-prefixes=BOTH,FNATTR
; RUN: opt -attributor --attributor-disable=false -attributor-max-iterations-verify -attributor-max-iterations=8 -S < %s | FileCheck %s --check-prefixes=BOTH,ATTRIBUTOR

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare nonnull i8* @ret_nonnull()

; Return a pointer trivially nonnull (call return attribute)
define i8* @test1() {
; BOTH: define nonnull i8* @test1
  %ret = call i8* @ret_nonnull()
  ret i8* %ret
}

; Return a pointer trivially nonnull (argument attribute)
define i8* @test2(i8* nonnull %p) {
; BOTH: define nonnull i8* @test2
  ret i8* %p
}

; Given an SCC where one of the functions can not be marked nonnull,
; can we still mark the other one which is trivially nonnull
define i8* @scc_binder(i1 %c) {
; FNATTR: define i8* @scc_binder
; ATTRIBUTOR: define noalias i8* @scc_binder
  br i1 %c, label %rec, label %end
rec:
  call i8* @test3(i1 %c)
  br label %end
end:
  ret i8* null
}

define i8* @test3(i1 %c) {
; BOTH: define nonnull i8* @test3
  call i8* @scc_binder(i1 %c)
  %ret = call i8* @ret_nonnull()
  ret i8* %ret
}

; Given a mutual recursive set of functions, we can mark them
; nonnull if neither can ever return null.  (In this case, they
; just never return period.)
define i8* @test4_helper() {
; FNATTR: define noalias nonnull i8* @test4_helper
; ATTRIBUTOR: define noalias nonnull align 536870912 dereferenceable(4294967295) i8* @test4_helper
  %ret = call i8* @test4()
  ret i8* %ret
}

define i8* @test4() {
; FNATTR: define noalias nonnull i8* @test4
; ATTRIBUTOR: define noalias nonnull align 536870912 dereferenceable(4294967295) i8* @test4
  %ret = call i8* @test4_helper()
  ret i8* %ret
}

; Given a mutual recursive set of functions which *can* return null
; make sure we haven't marked them as nonnull.
define i8* @test5_helper(i1 %c) {
; FNATTR: define noalias i8* @test5_helper
; ATTRIBUTOR: define noalias i8* @test5_helper
  br i1 %c, label %rec, label %end
rec:
  %ret = call i8* @test5(i1 %c)
  br label %end
end:
  ret i8* null
}

define i8* @test5(i1 %c) {
; FNATTR: define noalias i8* @test5
; ATTRIBUTOR: define noalias i8* @test5
  %ret = call i8* @test5_helper(i1 %c)
  ret i8* %ret
}

; Local analysis, but going through a self recursive phi
define i8* @test6() {
entry:
; BOTH: define nonnull i8* @test6
  %ret = call i8* @ret_nonnull()
  br label %loop
loop:
  %phi = phi i8* [%ret, %entry], [%phi, %loop]
  br i1 undef, label %loop, label %exit
exit:
  ret i8* %phi
}

; BOTH: define i8* @test7
define i8* @test7(i8* %a) {
  %b = getelementptr inbounds i8, i8* %a, i64 0
  ret i8* %b
}

; BOTH: define nonnull i8* @test8
define i8* @test8(i8* %a) {
  %b = getelementptr inbounds i8, i8* %a, i64 1
  ret i8* %b
}

; BOTH: define i8* @test9
define i8* @test9(i8* %a, i64 %n) {
  %b = getelementptr inbounds i8, i8* %a, i64 %n
  ret i8* %b
}

declare void @llvm.assume(i1)
; FNATTR: define i8* @test10
; FIXME: missing nonnull
; ATTRIBUTOR: define i8* @test10
define i8* @test10(i8* %a, i64 %n) {
  %cmp = icmp ne i64 %n, 0
  call void @llvm.assume(i1 %cmp)
  %b = getelementptr inbounds i8, i8* %a, i64 %n
  ret i8* %b
}

; TEST 11
; char* test11(char *p) {
;   return p? p: nonnull();
; }
; FNATTR: define i8* @test11
; FIXME: missing nonnull
; ATTRIBUTOR: define i8* @test11
define i8* @test11(i8*) local_unnamed_addr {
  %2 = icmp eq i8* %0, null
  br i1 %2, label %3, label %5

; <label>:3:                                      ; preds = %1
  %4 = tail call i8* @ret_nonnull()
  br label %5

; <label>:5:                                      ; preds = %3, %1
  %6 = phi i8* [ %4, %3 ], [ %0, %1 ]
  ret i8* %6
}

; TEST 12
; Simple CallSite Test
declare void @test12_helper(i8*)
define void @test12(i8* nonnull %a) {
; ATTRIBUTOR: define void @test12(i8* nonnull %a)
; ATTRIBUTOR-NEXT: tail call void @test12_helper(i8* nonnull %a)
  tail call void @test12_helper(i8* %a)
  ret void
}

; TEST 13
; Simple Argument Tests
declare i8* @unknown()
define void @test13_helper() {
  %nonnullptr = tail call i8* @ret_nonnull()
  %maybenullptr = tail call i8* @unknown()
  tail call void @test13(i8* %nonnullptr, i8* %nonnullptr, i8* %maybenullptr)
  tail call void @test13(i8* %nonnullptr, i8* %maybenullptr, i8* %nonnullptr)
  ret void
}
define internal void @test13(i8* %a, i8* %b, i8* %c) {
; ATTRIBUTOR: define internal void @test13(i8* nocapture nonnull readnone %a, i8* nocapture readnone %b, i8* nocapture readnone %c) 
  ret void
}

declare nonnull i8* @nonnull()

; TEST 14
; Complex propagation
; Argument of f1, f2, f3 can be marked with nonnull.

; * Argument
; 1. In f1:bb6, %arg can be marked with nonnull because of the comparison in bb1
; 2. Because f2 is internal function, f2(i32* %arg) -> @f2(i32* nonnull %arg)
; 3. In f1:bb4 %tmp5 is nonnull and f3 is internal function. 
;    Then, f3(i32* %arg) -> @f3(i32* nonnull %arg)
; 4. We get nonnull in whole f1 call sites so f1(i32* %arg) -> @f1(i32* nonnull %arg)


define internal i32* @f1(i32* %arg) {
; FIXME: missing nonnull It should be nonnull @f1(i32* nonnull readonly %arg)
; ATTRIBUTOR: define internal nonnull i32* @f1(i32* readonly %arg)

bb:
  %tmp = icmp eq i32* %arg, null
  br i1 %tmp, label %bb9, label %bb1

bb1:                                              ; preds = %bb
  %tmp2 = load i32, i32* %arg, align 4
  %tmp3 = icmp eq i32 %tmp2, 0
  br i1 %tmp3, label %bb6, label %bb4

bb4:                                              ; preds = %bb1
  %tmp5 = getelementptr inbounds i32, i32* %arg, i64 1
; ATTRIBUTOR: %tmp5b = tail call i32* @f3(i32* nonnull %tmp5)
  %tmp5b = tail call i32* @f3(i32* %tmp5)
  br label %bb9

bb6:                                              ; preds = %bb1
; FIXME: missing nonnull. It should be @f2(i32* nonnull %arg)
; ATTRIBUTOR: %tmp7 = tail call nonnull i32* @f2(i32* readonly %arg)
  %tmp7 = tail call i32* @f2(i32* %arg)
  ret i32* %tmp7

bb9:                                              ; preds = %bb4, %bb
  %tmp10 = phi i32* [ %tmp5, %bb4 ], [ inttoptr (i64 4 to i32*), %bb ]
  ret i32* %tmp10
}

define internal i32* @f2(i32* %arg) {
; FIXME: missing nonnull. It should be nonnull @f2(i32* nonnull %arg) 
; ATTRIBUTOR: define internal nonnull i32* @f2(i32* readonly %arg)
bb:

; FIXME: missing nonnull. It should be @f1(i32* nonnull readonly %arg) 
; ATTRIBUTOR:   %tmp = tail call nonnull i32* @f1(i32* readonly %arg)
  %tmp = tail call i32* @f1(i32* %arg)
  ret i32* %tmp
}

define dso_local noalias i32* @f3(i32* %arg) {
; FIXME: missing nonnull. It should be nonnull @f3(i32* nonnull readonly %arg) 
; ATTRIBUTOR: define dso_local noalias i32* @f3(i32* nocapture readonly %arg)
bb:
; FIXME: missing nonnull. It should be @f1(i32* nonnull readonly %arg) 
; ATTRIBUTOR:   %tmp = call i32* @f1(i32* readonly %arg)
  %tmp = call i32* @f1(i32* %arg)
  ret i32* null
}

; TEST 15
define void @f15(i8* %arg) {
; ATTRIBUTOR:   tail call void @use1(i8* nonnull dereferenceable(4) %arg)

  tail call void @use1(i8* dereferenceable(4) %arg)
  ret void
}

declare void @fun0() #1
declare void @fun1(i8*) #1
declare void @fun2(i8*, i8*) #1
declare void @fun3(i8*, i8*, i8*) #1
; TEST 16 simple path test
; if(..)
;   fun2(nonnull %a, nonnull %b)
; else
;   fun2(nonnull %a, %b)
; We can say that %a is nonnull but %b is not.
define void @f16(i8* %a, i8 * %b, i8 %c) {
; FIXME: missing nonnull on %a
; ATTRIBUTOR: define void @f16(i8* %a, i8* %b, i8 %c)
  %cmp = icmp eq i8 %c, 0
  br i1 %cmp, label %if.then, label %if.else
if.then:
  tail call void @fun2(i8* nonnull %a, i8* nonnull %b)
  ret void
if.else:
  tail call void @fun2(i8* nonnull %a, i8* %b)
  ret void
}
; TEST 17 explore child BB test
; if(..)
;    ... (willreturn & nounwind)
; else
;    ... (willreturn & nounwind)
; fun1(nonnull %a)
; We can say that %a is nonnull
define void @f17(i8* %a, i8 %c) {
; FIXME: missing nonnull on %a
; ATTRIBUTOR: define void @f17(i8* %a, i8 %c)
  %cmp = icmp eq i8 %c, 0
  br i1 %cmp, label %if.then, label %if.else
if.then:
  tail call void @fun0()
  br label %cont
if.else:
  tail call void @fun0()
  br label %cont
cont:
  tail call void @fun1(i8* nonnull %a)
  ret void
}
; TEST 18 More complex test
; if(..)
;    ... (willreturn & nounwind)
; else
;    ... (willreturn & nounwind)
; if(..)
;    ... (willreturn & nounwind)
; else
;    ... (willreturn & nounwind)
; fun1(nonnull %a)

define void @f18(i8* %a, i8* %b, i8 %c) {
; FIXME: missing nonnull on %a
; ATTRIBUTOR: define void @f18(i8* %a, i8* %b, i8 %c)
  %cmp1 = icmp eq i8 %c, 0
  br i1 %cmp1, label %if.then, label %if.else
if.then:
  tail call void @fun0()
  br label %cont
if.else:
  tail call void @fun0()
  br label %cont
cont:
  %cmp2 = icmp eq i8 %c, 1
  br i1 %cmp2, label %cont.then, label %cont.else
cont.then:
  tail call void @fun1(i8* nonnull %b)
  br label %cont2
cont.else:
  tail call void @fun0()
  br label %cont2
cont2:
  tail call void @fun1(i8* nonnull %a)
  ret void
}

; TEST 19: Loop

define void @f19(i8* %a, i8* %b, i8 %c) {
; FIXME: missing nonnull on %b
; ATTRIBUTOR: define void @f19(i8* %a, i8* %b, i8 %c)
  br label %loop.header
loop.header:
  %cmp2 = icmp eq i8 %c, 0
  br i1 %cmp2, label %loop.body, label %loop.exit
loop.body:
  tail call void @fun1(i8* nonnull %b)
  tail call void @fun1(i8* nonnull %a)
  br label %loop.header
loop.exit:
  tail call void @fun1(i8* nonnull %b)
  ret void
}

; Test propagation of nonnull callsite args back to caller.

declare void @use1(i8* %x)
declare void @use2(i8* %x, i8* %y);
declare void @use3(i8* %x, i8* %y, i8* %z);

declare void @use1nonnull(i8* nonnull %x);
declare void @use2nonnull(i8* nonnull %x, i8* nonnull %y);
declare void @use3nonnull(i8* nonnull %x, i8* nonnull %y, i8* nonnull %z);

declare i8 @use1safecall(i8* %x) readonly nounwind ; readonly+nounwind guarantees that execution continues to successor

; Can't extend non-null to parent for any argument because the 2nd call is not guaranteed to execute.

define void @parent1(i8* %a, i8* %b, i8* %c) {
; BOTH-LABEL: @parent1(i8* %a, i8* %b, i8* %c)
; BOTH-NEXT:    call void @use3(i8* %c, i8* %a, i8* %b)
; FNATTR-NEXT:    call void @use3nonnull(i8* %b, i8* %c, i8* %a)
; ATTRIBUTOR-NEXT:    call void @use3nonnull(i8* nonnull %b, i8* nonnull %c, i8* nonnull %a)
; BOTH-NEXT:    ret void
  call void @use3(i8* %c, i8* %a, i8* %b)
  call void @use3nonnull(i8* %b, i8* %c, i8* %a)
  ret void
}

; Extend non-null to parent for all arguments.

define void @parent2(i8* %a, i8* %b, i8* %c) {
; FNATTR-LABEL: @parent2(i8* nonnull %a, i8* nonnull %b, i8* nonnull %c)
; FNATTR-NEXT:    call void @use3nonnull(i8* %b, i8* %c, i8* %a)
; FNATTR-NEXT:    call void @use3(i8* %c, i8* %a, i8* %b)

; ATTRIBUTOR-LABEL: @parent2(i8* nonnull %a, i8* nonnull %b, i8* nonnull %c)
; ATTRIBUTOR-NEXT:    call void @use3nonnull(i8* nonnull %b, i8* nonnull %c, i8* nonnull %a)
; ATTRIBUTOR-NEXT:    call void @use3(i8* nonnull %c, i8* nonnull %a, i8* nonnull %b)

; BOTH-NEXT:    ret void
  call void @use3nonnull(i8* %b, i8* %c, i8* %a)
  call void @use3(i8* %c, i8* %a, i8* %b)
  ret void
}

; Extend non-null to parent for 1st argument.

define void @parent3(i8* %a, i8* %b, i8* %c) {
; FNATTR-LABEL: @parent3(i8* nonnull %a, i8* %b, i8* %c)
; FNATTR-NEXT:    call void @use1nonnull(i8* %a)
; FNATTR-NEXT:    call void @use3(i8* %c, i8* %b, i8* %a)

; ATTRIBUTOR-LABEL: @parent3(i8* nonnull %a, i8* %b, i8* %c)
; ATTRIBUTOR-NEXT:    call void @use1nonnull(i8* nonnull %a)
; ATTRIBUTOR-NEXT:    call void @use3(i8* %c, i8* %b, i8* nonnull %a)

; BOTH-NEXT:  ret void

  call void @use1nonnull(i8* %a)
  call void @use3(i8* %c, i8* %b, i8* %a)
  ret void
}

; Extend non-null to parent for last 2 arguments.

define void @parent4(i8* %a, i8* %b, i8* %c) {
; CHECK-LABEL: @parent4(i8* %a, i8* nonnull %b, i8* nonnull %c)
; CHECK-NEXT:    call void @use2nonnull(i8* %c, i8* %b)
; CHECK-NEXT:    call void @use2(i8* %a, i8* %c)
; CHECK-NEXT:    call void @use1(i8* %b)

; ATTRIBUTOR-LABEL: @parent4(i8* %a, i8* nonnull %b, i8* nonnull %c)
; ATTRIBUTOR-NEXT:    call void @use2nonnull(i8* nonnull %c, i8* nonnull %b)
; ATTRIBUTOR-NEXT:    call void @use2(i8* %a, i8* nonnull %c)
; ATTRIBUTOR-NEXT:    call void @use1(i8* nonnull %b)

; BOTH: ret void

  call void @use2nonnull(i8* %c, i8* %b)
  call void @use2(i8* %a, i8* %c)
  call void @use1(i8* %b)
  ret void
}

; The callsite must execute in order for the attribute to transfer to the parent.
; It appears benign to extend non-null to the parent in this case, but we can't do that
; because it would incorrectly propagate the wrong information to its callers.

define void @parent5(i8* %a, i1 %a_is_notnull) {
; BOTH: @parent5(i8* %a, i1 %a_is_notnull)
; BOTH-NEXT:    br i1 %a_is_notnull, label %t, label %f
; BOTH:       t:
; FNATTR-NEXT:    call void @use1nonnull(i8* %a)
; ATTRIBUTOR-NEXT:    call void @use1nonnull(i8* nonnull %a)
; BOTH-NEXT:    ret void
; BOTH:       f:
; BOTH-NEXT:    ret void

  br i1 %a_is_notnull, label %t, label %f
t:
  call void @use1nonnull(i8* %a)
  ret void
f:
  ret void
}

; The callsite must execute in order for the attribute to transfer to the parent.
; The volatile load can't trap, so we can guarantee that we'll get to the call.

define i8 @parent6(i8* %a, i8* %b) {
; FNATTR-LABEL: @parent6(i8* nonnull %a, i8* %b)
; ATTRIBUTOR-LABEL: @parent6(i8* nonnull %a, i8* %b)
; BOTH-NEXT:    [[C:%.*]] = load volatile i8, i8* %b
; FNATTR-NEXT:    call void @use1nonnull(i8* %a)
; ATTRIBUTOR-NEXT:    call void @use1nonnull(i8* nonnull %a)
; BOTH-NEXT:    ret i8 [[C]]

  %c = load volatile i8, i8* %b
  call void @use1nonnull(i8* %a)
  ret i8 %c
}

; The nonnull callsite is guaranteed to execute, so the argument must be nonnull throughout the parent.

define i8 @parent7(i8* %a) {
; FNATTR-LABEL: @parent7(i8* nonnull %a)
; FNATTR-NEXT:    [[RET:%.*]] = call i8 @use1safecall(i8* %a)
; FNATTR-NEXT:    call void @use1nonnull(i8* %a)


; ATTRIBUTOR-LABEL: @parent7(i8* nonnull %a)
; ATTRIBUTOR-NEXT:    [[RET:%.*]] = call i8 @use1safecall(i8* nonnull %a)
; ATTRIBUTOR-NEXT:    call void @use1nonnull(i8* nonnull %a)

; BOTH-NEXT: ret i8 [[RET]]

  %ret = call i8 @use1safecall(i8* %a)
  call void @use1nonnull(i8* %a)
  ret i8 %ret
}

; Make sure that an invoke works similarly to a call.

declare i32 @esfp(...)

define i1 @parent8(i8* %a, i8* %bogus1, i8* %b) personality i8* bitcast (i32 (...)* @esfp to i8*){
; BOTH-LABEL: @parent8(i8* nonnull %a, i8* nocapture readnone %bogus1, i8* nonnull %b) 
; BOTH-NEXT:  entry:
; FNATTR-NEXT:    invoke void @use2nonnull(i8* %a, i8* %b)
; ATTRIBUTOR-NEXT:    invoke void @use2nonnull(i8* nonnull %a, i8* nonnull %b)
; BOTH-NEXT:    to label %cont unwind label %exc
; BOTH:       cont:
; BOTH-NEXT:    [[NULL_CHECK:%.*]] = icmp eq i8* %b, null
; BOTH-NEXT:    ret i1 [[NULL_CHECK]]
; BOTH:       exc:
; BOTH-NEXT:    [[LP:%.*]] = landingpad { i8*, i32 }
; BOTH-NEXT:    filter [0 x i8*] zeroinitializer
; BOTH-NEXT:    unreachable

entry:
  invoke void @use2nonnull(i8* %a, i8* %b)
  to label %cont unwind label %exc

cont:
  %null_check = icmp eq i8* %b, null
  ret i1 %null_check

exc:
  %lp = landingpad { i8*, i32 }
  filter [0 x i8*] zeroinitializer
  unreachable
}

; BOTH: define nonnull i32* @gep1(
define i32* @gep1(i32* %p) {
  %q = getelementptr inbounds i32, i32* %p, i32 1
  ret i32* %q
}

define i32* @gep1_no_null_opt(i32* %p) #0 {
; Should't be able to derive nonnull based on gep.
; BOTH: define i32* @gep1_no_null_opt(
  %q = getelementptr inbounds i32, i32* %p, i32 1
  ret i32* %q
}

; BOTH: define i32 addrspace(3)* @gep2(
define i32 addrspace(3)* @gep2(i32 addrspace(3)* %p) {
  %q = getelementptr inbounds i32, i32 addrspace(3)* %p, i32 1
  ret i32 addrspace(3)* %q
}

; FNATTR:     define i32 addrspace(3)* @as(i32 addrspace(3)* readnone returned dereferenceable(4) %p)
; FIXME: We should propagate dereferenceable here but *not* nonnull
; ATTRIBUTOR: define dereferenceable_or_null(4) i32 addrspace(3)* @as(i32 addrspace(3)* readnone returned dereferenceable(4) dereferenceable_or_null(4) %p)
define i32 addrspace(3)* @as(i32 addrspace(3)* dereferenceable(4) %p) {
  ret i32 addrspace(3)* %p
}

; BOTH: define internal nonnull i32* @g2()
define internal i32* @g2() {
  ret i32* inttoptr (i64 4 to i32*)
}

define  i32* @g1() {
 %c = call i32* @g2()
  ret i32* %c
}

declare void @use_i32_ptr(i32*) readnone nounwind
; ATTRIBUTOR: define internal void @called_by_weak(i32* nocapture nonnull readnone %a)
define internal void @called_by_weak(i32* %a) {
  call void @use_i32_ptr(i32* %a)
  ret void
}

; Check we do not annotate the function interface of this weak function.
; ATTRIBUTOR: define weak_odr void @weak_caller(i32* nonnull %a)
define weak_odr void @weak_caller(i32* nonnull %a) {
  call void @called_by_weak(i32* %a)
  ret void
}

; Expect nonnull
; ATTRIBUTOR: define internal void @control(i32* nocapture nonnull readnone align 16 dereferenceable(8) %a)
define internal void @control(i32* dereferenceable(4) %a) {
  call void @use_i32_ptr(i32* %a)
  ret void
}
; Avoid nonnull as we do not touch naked functions
; ATTRIBUTOR: define internal void @naked(i32* dereferenceable(4) %a)
define internal void @naked(i32* dereferenceable(4) %a) naked {
  call void @use_i32_ptr(i32* %a)
  ret void
}
; Avoid nonnull as we do not touch optnone
; ATTRIBUTOR: define internal void @optnone(i32* dereferenceable(4) %a)
define internal void @optnone(i32* dereferenceable(4) %a) optnone noinline {
  call void @use_i32_ptr(i32* %a)
  ret void
}
define void @make_live(i32* nonnull dereferenceable(8) %a) {
  call void @naked(i32* nonnull dereferenceable(8) align 16 %a)
  call void @control(i32* nonnull dereferenceable(8) align 16 %a)
  call void @optnone(i32* nonnull dereferenceable(8) align 16 %a)
  ret void
}

attributes #0 = { "null-pointer-is-valid"="true" }
attributes #1 = { nounwind willreturn}
