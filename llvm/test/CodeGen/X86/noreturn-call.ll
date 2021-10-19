; RUN: llc < %s -mtriple=i686-pc-win32 | FileCheck %s

define void @test1(i32 %c) {
; CHECK-LABEL: test1:
entry:
  %0 = alloca i8, i32 %c
  %tobool = icmp eq i32 %c, 0
  br i1 %tobool, label %if.end, label %if.then

if.end:
  call void @g(i8* %0)
  ret void

if.then:
  call void @crash(i8* %0)
  unreachable
; CHECK: calll _crash
; There is no need to adjust the stack after the call, since
; the function is noreturn and that code will therefore never run.
; CHECK-NOT: add
; CHECK-NOT: pop
}

define void @test2(i32 %c) {
; CHECK-LABEL: test2:
entry:
  %0 = alloca i8, i32 %c
  %tobool = icmp eq i32 %c, 0
  br i1 %tobool, label %if.end, label %if.then

if.end:
  call void @g(i8* %0)
  ret void

if.then:
  call void @crash2(i8* %0)
  unreachable
; CHECK: calll _crash2
; Even though _crash2 is not marked noreturn, it is in practice because
; of the "unreachable" right after it. This happens e.g. when falling off
; a non-void function after a call.
; CHECK-NOT: add
; CHECK-NOT: pop
}

declare void @crash(i8*) noreturn
declare void @crash2(i8*)
declare void @g(i8*)

%struct.ByVal = type { [10 x i32] }

define dso_local i32 @pr43155() {
entry:
  %agg.tmp = alloca %struct.ByVal, align 4
  %agg.tmp5 = alloca %struct.ByVal, align 4
  %agg.tmp6 = alloca %struct.ByVal, align 4
  %call = tail call i32 @cond()
  %tobool = icmp eq i32 %call, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  tail call x86_stdcallcc void @stdcall_abort(i32 12, i32 2)
  unreachable

if.end:                                           ; preds = %entry
  %call1 = tail call i32 @cond()
  %tobool2 = icmp eq i32 %call1, 0
  br i1 %tobool2, label %if.end4, label %if.then3

if.then3:                                         ; preds = %if.end
  tail call x86_stdcallcc void @stdcall_abort(i32 15, i32 2)
  unreachable

if.end4:                                          ; preds = %if.end
  call void @getbyval(%struct.ByVal* nonnull sret(%struct.ByVal) %agg.tmp)
  call void @make_push_unprofitable(%struct.ByVal* nonnull byval(%struct.ByVal) align 4 %agg.tmp)
  call void @getbyval(%struct.ByVal* nonnull sret(%struct.ByVal) %agg.tmp5)
  call void @make_push_unprofitable(%struct.ByVal* nonnull byval(%struct.ByVal) align 4 %agg.tmp5)
  call void @getbyval(%struct.ByVal* nonnull sret(%struct.ByVal) %agg.tmp6)
  call void @make_push_unprofitable(%struct.ByVal* nonnull byval(%struct.ByVal) align 4 %agg.tmp6)
  ret i32 0
}

;   Check that there are no stack adjustments after stdcall_abort.
; CHECK-LABEL: pr43155:
;   The main function body contents are not important.
; CHECK: retl
; CHECK:  # %if.then
; CHECK: calll _stdcall_abort@8
; CHECK-NOT: sub
; CHECK-NOT: add
; CHECK:  # %if.then3
; CHECK: calll _stdcall_abort@8
; CHECK-NOT: sub
; CHECK-NOT: add
; CHECK: # -- End function

declare dso_local i32 @cond()

declare dso_local x86_stdcallcc void @stdcall_abort(i32, i32) noreturn

declare dso_local void @make_push_unprofitable(%struct.ByVal* byval(%struct.ByVal) align 4)

declare dso_local void @getbyval(%struct.ByVal* sret(%struct.ByVal))
