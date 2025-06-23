; clang -target aarch64-eabi -O2 -fsanitize=memtag -S -emit-llvm test.cc
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
target triple = "aarch64"

@_ZTIi = external dso_local constant ptr

; Function Attrs: noreturn sanitize_memtag
define dso_local void @_Z3barv() local_unnamed_addr #0 {
entry:
  %exception = tail call ptr @__cxa_allocate_exception(i64 4) #4
  store i32 42, ptr %exception, align 16, !tbaa !2
  tail call void @__cxa_throw(ptr %exception, ptr @_ZTIi, ptr null) #5
  unreachable
}

declare dso_local ptr @__cxa_allocate_exception(i64) local_unnamed_addr

declare dso_local void @__cxa_throw(ptr, ptr, ptr) local_unnamed_addr

; Function Attrs: noreturn sanitize_memtag
define dso_local void @_Z3foov() local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
entry:
  %A0 = alloca i32, align 4
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %A0) #4
  call void asm sideeffect "", "r"(ptr nonnull %A0) #4, !srcloc !6
  invoke void @_Z3barv()
          to label %try.cont unwind label %lpad

lpad:                                             ; preds = %entry
  %0 = landingpad { ptr, i32 }
          cleanup
          catch ptr @_ZTIi
  %1 = extractvalue { ptr, i32 } %0, 1
  %2 = call i32 @llvm.eh.typeid.for(ptr @_ZTIi) #4
  %matches = icmp eq i32 %1, %2
  br i1 %matches, label %catch, label %ehcleanup

catch:                                            ; preds = %lpad
  %3 = extractvalue { ptr, i32 } %0, 0
  %4 = call ptr @__cxa_begin_catch(ptr %3) #4
  call void @__cxa_end_catch() #4
  br label %try.cont

try.cont:                                         ; preds = %entry, %catch
  %exception = call ptr @__cxa_allocate_exception(i64 4) #4
  store i32 15532, ptr %exception, align 16, !tbaa !2
  call void @__cxa_throw(ptr %exception, ptr @_ZTIi, ptr null) #5
  unreachable

ehcleanup:                                        ; preds = %lpad
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %A0) #4
  resume { ptr, i32 } %0
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

declare dso_local i32 @__gxx_personality_v0(...)

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(ptr) #2

declare dso_local ptr @__cxa_begin_catch(ptr) local_unnamed_addr

declare dso_local void @__cxa_end_catch() local_unnamed_addr

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: norecurse sanitize_memtag
define dso_local i32 @main() local_unnamed_addr #3 personality ptr @__gxx_personality_v0 {
entry:
; CHECK-LABEL: entry:
  %A0.i = alloca i32, align 4
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %A0.i) #4
  call void asm sideeffect "", "r"(ptr nonnull %A0.i) #4, !srcloc !6
; CHECK: call void @llvm.aarch64.settag(ptr %A0.i.tag, i64 16)
; CHECK-NEXT: call void asm sideeffect
  %exception.i6 = call ptr @__cxa_allocate_exception(i64 4) #4
  store i32 42, ptr %exception.i6, align 16, !tbaa !2
  invoke void @__cxa_throw(ptr %exception.i6, ptr @_ZTIi, ptr null) #5
          to label %.noexc7 unwind label %lpad.i

.noexc7:                                          ; preds = %entry
  unreachable

lpad.i:                                           ; preds = %entry
  %0 = landingpad { ptr, i32 }
          cleanup
          catch ptr @_ZTIi
  %1 = extractvalue { ptr, i32 } %0, 1
  %2 = call i32 @llvm.eh.typeid.for(ptr @_ZTIi) #4
  %matches.i = icmp eq i32 %1, %2
  br i1 %matches.i, label %catch.i, label %ehcleanup.i

catch.i:                                          ; preds = %lpad.i
  %3 = extractvalue { ptr, i32 } %0, 0
  %4 = call ptr @__cxa_begin_catch(ptr %3) #4
  call void @__cxa_end_catch() #4
  %exception.i = call ptr @__cxa_allocate_exception(i64 4) #4
  store i32 15532, ptr %exception.i, align 16, !tbaa !2
  invoke void @__cxa_throw(ptr %exception.i, ptr @_ZTIi, ptr null) #5
          to label %.noexc unwind label %lpad

.noexc:                                           ; preds = %catch.i
  unreachable

ehcleanup.i:                                      ; preds = %lpad.i
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %A0.i) #4
  br label %lpad.body

lpad:                                             ; preds = %catch.i
  %5 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %.pre = extractvalue { ptr, i32 } %5, 1
  br label %lpad.body

lpad.body:                                        ; preds = %ehcleanup.i, %lpad
  %.pre-phi = phi i32 [ %1, %ehcleanup.i ], [ %.pre, %lpad ]
  %eh.lpad-body = phi { ptr, i32 } [ %0, %ehcleanup.i ], [ %5, %lpad ]
  %matches = icmp eq i32 %.pre-phi, %2
  br i1 %matches, label %catch, label %eh.resume

catch:                                            ; preds = %lpad.body
  %6 = extractvalue { ptr, i32 } %eh.lpad-body, 0
  %7 = call ptr @__cxa_begin_catch(ptr %6) #4
  call void @__cxa_end_catch() #4
  ret i32 0

eh.resume:                                        ; preds = %lpad.body
  resume { ptr, i32 } %eh.lpad-body
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
