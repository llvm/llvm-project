; RUN: opt < %s -pgo-instr-gen -S | FileCheck %s --check-prefix=GEN
; RUN: opt < %s -passes=pgo-instr-gen -S | FileCheck %s --check-prefix=GEN
; RUN: opt < %s -passes=pgo-instr-gen,instrprof -S | FileCheck %s --check-prefix=LOWER

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$foo3 = comdat any

@bar = external global void ()*, align 8
; GEN: @__profn_foo = private constant [3 x i8] c"foo"

define void @foo() {
entry:
; GEN: entry:
; GEN-NEXT: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_foo, i32 0, i32 0), i64 12884901887, i32 1, i32 0)
  %tmp = load void ()*, void ()** @bar, align 8
; GEN: [[ICALL_TARGET:%[0-9]+]] = ptrtoint void ()* %tmp to i64
; GEN-NEXT: call void @llvm.instrprof.value.profile(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_foo, i32 0, i32 0), i64 12884901887, i64 [[ICALL_TARGET]], i32 0, i32 0)
  call void %tmp()
  ret void
}

@bar2 = global void ()* null, align 8
@_ZTIi = external constant i8*

define i32 @foo2(i32 %arg, i8** nocapture readnone %arg1) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
bb:
  %tmp2 = load void ()*, void ()** @bar2, align 8
  invoke void %tmp2()
          to label %bb10 unwind label %bb2
; GEN: [[ICALL_TARGET2:%[0-9]+]] = ptrtoint void ()* %tmp2 to i64
; GEN-NEXT: call void @llvm.instrprof.value.profile(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @__profn_foo2, i32 0, i32 0), i64 38432627612, i64 [[ICALL_TARGET2]], i32 0, i32 0)

bb2:                                              ; preds = %bb
  %tmp3 = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @_ZTIi to i8*)
  %tmp4 = extractvalue { i8*, i32 } %tmp3, 1
  %tmp5 = tail call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*))
  %tmp6 = icmp eq i32 %tmp4, %tmp5
  br i1 %tmp6, label %bb7, label %bb11

bb7:                                              ; preds = %bb2
  %tmp8 = extractvalue { i8*, i32 } %tmp3, 0
  %tmp9 = tail call i8* @__cxa_begin_catch(i8* %tmp8)
  tail call void @__cxa_end_catch()
  br label %bb10

bb10:                                             ; preds = %bb7, %bb
  ret i32 0

bb11:                                             ; preds = %bb2
  resume { i8*, i32 } %tmp3
}

; Test that comdat function's address is recorded.
; LOWER: @__profd_foo3 = linkonce_odr{{.*}}@foo3
; Function Attrs: nounwind uwtable
define linkonce_odr i32 @foo3()  comdat  {
  ret i32 1
}

declare i32 @__gxx_personality_v0(...)

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(i8*) #0

declare i8* @__cxa_begin_catch(i8*)

declare void @__cxa_end_catch()

