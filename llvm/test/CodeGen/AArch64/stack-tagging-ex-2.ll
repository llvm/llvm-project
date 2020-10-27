; clang -target aarch64-eabi -O2 -march=armv8.5-a+memtag -fsanitize=memtag -S -emit-llvm test.cc
; void bar() {
;   throw 42;
; }

; void foo() {
;   int A0;
;   __asm volatile("" : : "r"(&A0));

;   try {
;     bar();
;   } catch (int exc) {
;   }

;   throw 15532;
; }

; int main() {
;   try {
;     foo();
;   } catch (int exc) {
;   }

;   return 0;
; }

; RUN: opt -S -aarch64-stack-tagging %s -o - | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-unknown-eabi"

@_ZTIi = external dso_local constant i8*

; Function Attrs: noreturn sanitize_memtag
define dso_local void @_Z3barv() local_unnamed_addr #0 {
entry:
  %exception = tail call i8* @__cxa_allocate_exception(i64 4) #4
  %0 = bitcast i8* %exception to i32*
  store i32 42, i32* %0, align 16, !tbaa !2
  tail call void @__cxa_throw(i8* %exception, i8* bitcast (i8** @_ZTIi to i8*), i8* null) #5
  unreachable
}

declare dso_local i8* @__cxa_allocate_exception(i64) local_unnamed_addr

declare dso_local void @__cxa_throw(i8*, i8*, i8*) local_unnamed_addr

; Function Attrs: noreturn sanitize_memtag
define dso_local void @_Z3foov() local_unnamed_addr #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %A0 = alloca i32, align 4
  %0 = bitcast i32* %A0 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %0) #4
  call void asm sideeffect "", "r"(i32* nonnull %A0) #4, !srcloc !6
  invoke void @_Z3barv()
          to label %try.cont unwind label %lpad

lpad:                                             ; preds = %entry
  %1 = landingpad { i8*, i32 }
          cleanup
          catch i8* bitcast (i8** @_ZTIi to i8*)
  %2 = extractvalue { i8*, i32 } %1, 1
  %3 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*)) #4
  %matches = icmp eq i32 %2, %3
  br i1 %matches, label %catch, label %ehcleanup

catch:                                            ; preds = %lpad
  %4 = extractvalue { i8*, i32 } %1, 0
  %5 = call i8* @__cxa_begin_catch(i8* %4) #4
  call void @__cxa_end_catch() #4
  br label %try.cont

try.cont:                                         ; preds = %entry, %catch
  %exception = call i8* @__cxa_allocate_exception(i64 4) #4
  %6 = bitcast i8* %exception to i32*
  store i32 15532, i32* %6, align 16, !tbaa !2
  call void @__cxa_throw(i8* %exception, i8* bitcast (i8** @_ZTIi to i8*), i8* null) #5
  unreachable

ehcleanup:                                        ; preds = %lpad
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %0) #4
  resume { i8*, i32 } %1
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

declare dso_local i32 @__gxx_personality_v0(...)

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(i8*) #2

declare dso_local i8* @__cxa_begin_catch(i8*) local_unnamed_addr

declare dso_local void @__cxa_end_catch() local_unnamed_addr

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: norecurse sanitize_memtag
define dso_local i32 @main() local_unnamed_addr #3 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
; CHECK-LABEL: entry:
  %A0.i = alloca i32, align 4
  %0 = bitcast i32* %A0.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %0) #4
  call void asm sideeffect "", "r"(i32* nonnull %A0.i) #4, !srcloc !6
; CHECK: call void @llvm.aarch64.settag(i8* %1, i64 16)
; CHECK-NEXT: call void asm sideeffect
  %exception.i6 = call i8* @__cxa_allocate_exception(i64 4) #4
  %1 = bitcast i8* %exception.i6 to i32*
  store i32 42, i32* %1, align 16, !tbaa !2
  invoke void @__cxa_throw(i8* %exception.i6, i8* bitcast (i8** @_ZTIi to i8*), i8* null) #5
          to label %.noexc7 unwind label %lpad.i

.noexc7:                                          ; preds = %entry
  unreachable

lpad.i:                                           ; preds = %entry
  %2 = landingpad { i8*, i32 }
          cleanup
          catch i8* bitcast (i8** @_ZTIi to i8*)
  %3 = extractvalue { i8*, i32 } %2, 1
  %4 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*)) #4
  %matches.i = icmp eq i32 %3, %4
  br i1 %matches.i, label %catch.i, label %ehcleanup.i

catch.i:                                          ; preds = %lpad.i
  %5 = extractvalue { i8*, i32 } %2, 0
  %6 = call i8* @__cxa_begin_catch(i8* %5) #4
  call void @__cxa_end_catch() #4
  %exception.i = call i8* @__cxa_allocate_exception(i64 4) #4
  %7 = bitcast i8* %exception.i to i32*
  store i32 15532, i32* %7, align 16, !tbaa !2
  invoke void @__cxa_throw(i8* %exception.i, i8* bitcast (i8** @_ZTIi to i8*), i8* null) #5
          to label %.noexc unwind label %lpad

.noexc:                                           ; preds = %catch.i
  unreachable

ehcleanup.i:                                      ; preds = %lpad.i
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %0) #4
  br label %lpad.body

lpad:                                             ; preds = %catch.i
  %8 = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @_ZTIi to i8*)
  %.pre = extractvalue { i8*, i32 } %8, 1
  br label %lpad.body

lpad.body:                                        ; preds = %ehcleanup.i, %lpad
  %.pre-phi = phi i32 [ %3, %ehcleanup.i ], [ %.pre, %lpad ]
  %eh.lpad-body = phi { i8*, i32 } [ %2, %ehcleanup.i ], [ %8, %lpad ]
  %matches = icmp eq i32 %.pre-phi, %4
  br i1 %matches, label %catch, label %eh.resume

catch:                                            ; preds = %lpad.body
  %9 = extractvalue { i8*, i32 } %eh.lpad-body, 0
  %10 = call i8* @__cxa_begin_catch(i8* %9) #4
  call void @__cxa_end_catch() #4
  ret i32 0

eh.resume:                                        ; preds = %lpad.body
  resume { i8*, i32 } %eh.lpad-body
}

attributes #0 = { noreturn sanitize_memtag "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+mte,+neon,+v8.5a" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }
attributes #2 = { nounwind readnone }
attributes #3 = { norecurse sanitize_memtag "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+mte,+neon,+v8.5a" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }
attributes #5 = { noreturn }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 10.0.0 (https://github.com/llvm/llvm-project.git c38188c5fe41751fda095edde1a878b2a051ae58)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
!6 = !{i32 70}
