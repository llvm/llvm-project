; RUN: opt < %s -passes=tailcallelim -verify-dom-info -S | FileCheck %s

declare void @noarg()
declare void @use(ptr)
declare void @use_nocapture(ptr nocapture)
declare void @use2_nocapture(ptr nocapture, ptr nocapture)

; Trivial case. Mark @noarg with tail call.
define void @test0() {
; CHECK: tail call void @noarg()
	call void @noarg()
	ret void
}

; Make sure that we do not do TRE if pointer to local stack
; escapes through function call.
define i32 @test1() {
; CHECK: i32 @test1()
; CHECK-NEXT: alloca
	%A = alloca i32		; <ptr> [#uses=2]
	store i32 5, ptr %A
	call void @use(ptr %A)
; CHECK: call i32 @test1
	%X = call i32 @test1()		; <i32> [#uses=1]
	ret i32 %X
}

; This function contains intervening instructions which should be moved out of the way
define i32 @test2(i32 %X) {
; CHECK: i32 @test2
; CHECK-NOT: call
; CHECK: ret i32
entry:
	%tmp.1 = icmp eq i32 %X, 0		; <i1> [#uses=1]
	br i1 %tmp.1, label %then.0, label %endif.0
then.0:		; preds = %entry
	%tmp.4 = add i32 %X, 1		; <i32> [#uses=1]
	ret i32 %tmp.4
endif.0:		; preds = %entry
	%tmp.10 = add i32 %X, -1		; <i32> [#uses=1]
	%tmp.8 = call i32 @test2(i32 %tmp.10)		; <i32> [#uses=1]
	%DUMMY = add i32 %X, 1		; <i32> [#uses=0]
	ret i32 %tmp.8
}

; Though this case seems to be fairly unlikely to occur in the wild, someone
; plunked it into the demo script, so maybe they care about it.
define i32 @test3(i32 %c) {
; CHECK: i32 @test3
; CHECK: tailrecurse:
; CHECK: %ret.tr = phi i32 [ poison, %entry ], [ %current.ret.tr, %else ]
; CHECK: %ret.known.tr = phi i1 [ false, %entry ], [ true, %else ]
; CHECK: else:
; CHECK-NOT: call
; CHECK: %current.ret.tr = select i1 %ret.known.tr, i32 %ret.tr, i32 0
; CHECK-NOT: ret
; CHECK: return:
; CHECK: %current.ret.tr1 = select i1 %ret.known.tr, i32 %ret.tr, i32 0
; CHECK: ret i32 %current.ret.tr1
entry:
	%tmp.1 = icmp eq i32 %c, 0		; <i1> [#uses=1]
	br i1 %tmp.1, label %return, label %else
else:		; preds = %entry
	%tmp.5 = add i32 %c, -1		; <i32> [#uses=1]
	%tmp.3 = call i32 @test3(i32 %tmp.5)		; <i32> [#uses=0]
	ret i32 0
return:		; preds = %entry
	ret i32 0
}

; Make sure that a nocapture pointer does not stop adding a tail call marker to
; an unrelated call and additionally that we do not mark the nocapture call with
; a tail call.
;
; rdar://14324281
define void @test4() {
; CHECK: void @test4
; CHECK-NOT: tail call void @use_nocapture
; CHECK: tail call void @noarg()
; CHECK: ret void
  %a = alloca i32
  call void @use_nocapture(ptr %a)
  call void @noarg()
  ret void
}

; Make sure that we do not perform TRE even with a nocapture use. This is due to
; bad codegen caused by PR962.
;
; rdar://14324281.
define ptr @test5(ptr nocapture %A, i1 %cond) {
; CHECK: ptr @test5
; CHECK-NOT: tailrecurse:
; CHECK: ret ptr null
  %B = alloca i32
  br i1 %cond, label %cond_true, label %cond_false
cond_true:
  call ptr @test5(ptr %B, i1 false)
  ret ptr null
cond_false:
  call void @use2_nocapture(ptr %A, ptr %B)
  call void @noarg()
  ret ptr null
}

; PR14143: Make sure that we do not mark functions with nocapture allocas with tail.
;
; rdar://14324281.
define void @test6(ptr %a, ptr %b) {
; CHECK-LABEL: @test6(
; CHECK-NOT: tail call
; CHECK: ret void
  %c = alloca [100 x i8], align 16
  call void @use2_nocapture(ptr %b, ptr %c)
  ret void
}

; PR14143: Make sure that we do not mark functions with nocapture allocas with tail.
;
; rdar://14324281
define void @test7(ptr %a, ptr %b) nounwind uwtable {
entry:
; CHECK-LABEL: @test7(
; CHECK-NOT: tail call
; CHECK: ret void
  %c = alloca [100 x i8], align 16
  call void @use2_nocapture(ptr %c, ptr %a)
  call void @use2_nocapture(ptr %b, ptr %c)
  ret void
}

; If we have a mix of escaping captured/non-captured allocas, ensure that we do
; not do anything including marking callsites with the tail call marker.
;
; rdar://14324281.
define ptr @test8(ptr nocapture %A, i1 %cond) {
; CHECK: ptr @test8
; CHECK-NOT: tailrecurse:
; CHECK-NOT: tail call
; CHECK: ret ptr null
  %B = alloca i32
  %B2 = alloca i32
  br i1 %cond, label %cond_true, label %cond_false
cond_true:
  call void @use(ptr %B2)
  call ptr @test8(ptr %B, i1 false)
  ret ptr null
cond_false:
  call void @use2_nocapture(ptr %A, ptr %B)
  call void @noarg()
  ret ptr null
}

; Don't tail call if a byval arg is captured.
define void @test9(ptr byval(i32) %a) {
; CHECK-LABEL: define void @test9(
; CHECK: {{^ *}}call void @use(
  call void @use(ptr %a)
  ret void
}

%struct.X = type { ptr }

declare void @ctor(ptr)
define void @test10(ptr noalias sret(%struct.X) %agg.result, i1 zeroext %b) {
; CHECK-LABEL: @test10
entry:
  %x = alloca %struct.X, align 8
  br i1 %b, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @ctor(ptr %agg.result)
; CHECK: tail call void @ctor
  br label %return

if.end:
  call void @ctor(ptr %x)
; CHECK: call void @ctor
  br label %return

return:
  ret void
}

declare void @test11_helper1(ptr nocapture, ptr)
declare void @test11_helper2(ptr)
define void @test11() {
; CHECK-LABEL: @test11
; CHECK-NOT: tail
  %a = alloca ptr
  %b = alloca i8
  call void @test11_helper1(ptr %a, ptr %b)  ; a = &b
  %c = load ptr, ptr %a
  call void @test11_helper2(ptr %c)
; CHECK: call void @test11_helper2
  ret void
}

; PR25928
define void @test12() {
entry:
; CHECK-LABEL: @test12
; CHECK: {{^ *}} call void undef(ptr undef) [ "foo"(ptr %e) ]
  %e = alloca i8
  call void undef(ptr undef) [ "foo"(ptr %e) ]
  unreachable
}

%struct.foo = type { [10 x i32] }

; If an alloca is passed byval it is not a use of the alloca or an escape
; point, and both calls below can be marked tail.
define void @test13() {
; CHECK-LABEL: @test13
; CHECK: tail call void @bar(ptr byval(%struct.foo) %f)
; CHECK: tail call void @bar(ptr byval(%struct.foo) null)
entry:
  %f = alloca %struct.foo
  call void @bar(ptr byval(%struct.foo) %f)
  call void @bar(ptr byval(%struct.foo) null)
  ret void
}

; A call which passes a byval parameter using byval can be marked tail.
define void @test14(ptr byval(%struct.foo) %f) {
; CHECK-LABEL: @test14
; CHECK: tail call void @bar
entry:
  call void @bar(ptr byval(%struct.foo) %f)
  ret void
}

; If a byval parameter is copied into an alloca and passed byval the call can
; be marked tail.
define void @test15(ptr byval(%struct.foo) %f) {
; CHECK-LABEL: @test15
; CHECK: tail call void @bar
entry:
  %agg.tmp = alloca %struct.foo
  call void @llvm.memcpy.p0.p0.i64(ptr %agg.tmp, ptr %f, i64 40, i1 false)
  call void @bar(ptr byval(%struct.foo) %agg.tmp)
  ret void
}

declare void @bar(ptr byval(%struct.foo))
declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1)
