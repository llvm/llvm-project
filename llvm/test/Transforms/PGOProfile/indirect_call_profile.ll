; RUN: opt < %s -pgo-instr-gen -S | FileCheck %s --check-prefix=GEN
; RUN: opt < %s -passes=pgo-instr-gen -S | FileCheck %s --check-prefix=GEN
; RUN: opt < %s -passes=pgo-instr-gen,instrprof -vp-static-alloc=true -S | FileCheck %s --check-prefix=LOWER
; RUN: opt < %s -passes=pgo-instr-gen,instrprof -vp-static-alloc=false -S | FileCheck %s --check-prefix=LOWER

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$novp_inline = comdat any
$vp_inline = comdat any

@bar = external global void ()*, align 8

; GEN: @__profn_novp_inline = linkonce_odr hidden constant [11 x i8] c"novp_inline"
; GEN: @__profn_foo = private constant [3 x i8] c"foo"
; GEN: @__profn_vp_inline = linkonce_odr hidden constant [9 x i8] c"vp_inline"

;; Test that a linkonce function's address is recorded.
;; We allow a linkonce profd to be private if the function does not use value profiling.
; LOWER:      @__profd_novp_inline.[[HASH:[0-9]+]] = private global {{.*}} @__profc_novp_inline.[[HASH]]
; LOWER-SAME:   i8* bitcast (void ()* @novp_inline to i8*)
; LOWER:      @__profd_foo = private {{.*}} @__profc_foo

;; __profd_vp_inline.[[#]] is referenced by code and may be referenced by other
;; text sections due to inlining. It can't be local because a linker error would
;; occur if a prevailing text section references the non-prevailing local symbol.
; LOWER:      @__profd_vp_inline.[[FOO_HASH:[0-9]+]] = linkonce_odr hidden {{.*}} @__profc_vp_inline.[[FOO_HASH]]
; LOWER-SAME:   i8* bitcast (void ()* @vp_inline to i8*)

define linkonce_odr void @novp_inline() comdat {
  ret void
}

define void @foo() {
entry:
; GEN: @foo()
; GEN: entry:
; GEN-NEXT: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_foo, i32 0, i32 0), i64 [[#FOO_HASH:]], i32 1, i32 0)
  %tmp = load void ()*, void ()** @bar, align 8
; GEN: [[ICALL_TARGET:%[0-9]+]] = ptrtoint void ()* %tmp to i64
; GEN-NEXT: call void @llvm.instrprof.value.profile(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_foo, i32 0, i32 0), i64 [[#FOO_HASH]], i64 [[ICALL_TARGET]], i32 0, i32 0)
; LOWER: call void @__llvm_profile_instrument_target(i64 %1, i8* bitcast ({ i64, i64, i64, i8*, i8*, i32, [2 x i16] }* @__profd_foo to i8*), i32 0)
  call void %tmp()
  ret void
}

define linkonce_odr void @vp_inline() comdat {
entry:
; GEN: @vp_inline()
; GEN: entry:
; GEN-NEXT: call void @llvm.instrprof.increment(i8* getelementptr inbounds ([9 x i8], [9 x i8]* @__profn_vp_inline, i32 0, i32 0), i64 [[#FOO_HASH:]], i32 1, i32 0)
  %tmp = load void ()*, void ()** @bar, align 8
; GEN: [[ICALL_TARGET:%[0-9]+]] = ptrtoint void ()* %tmp to i64
; GEN-NEXT: call void @llvm.instrprof.value.profile(i8* getelementptr inbounds ([9 x i8], [9 x i8]* @__profn_vp_inline, i32 0, i32 0), i64 [[#FOO_HASH]], i64 [[ICALL_TARGET]], i32 0, i32 0)
; LOWER: call void @__llvm_profile_instrument_target(i64 %1, i8* bitcast ({ i64, i64, i64, i8*, i8*, i32, [2 x i16] }* @__profd_vp_inline.[[#]] to i8*), i32 0)
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
; GEN-NEXT: call void @llvm.instrprof.value.profile(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @__profn_foo2, i32 0, i32 0), i64 [[FOO2_HASH:[0-9]+]], i64 [[ICALL_TARGET2]], i32 0, i32 0)

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

declare i32 @__gxx_personality_v0(...)

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(i8*) #0

declare i8* @__cxa_begin_catch(i8*)

declare void @__cxa_end_catch()

