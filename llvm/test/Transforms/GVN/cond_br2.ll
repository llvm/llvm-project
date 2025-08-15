; RUN: opt -passes=gvn -S < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

%"class.llvm::SmallVector" = type { %"class.llvm::SmallVectorImpl", [1 x %"union.llvm::SmallVectorBase::U"] }
%"class.llvm::SmallVectorImpl" = type { %"class.llvm::SmallVectorTemplateBase" }
%"class.llvm::SmallVectorTemplateBase" = type { %"class.llvm::SmallVectorTemplateCommon" }
%"class.llvm::SmallVectorTemplateCommon" = type { %"class.llvm::SmallVectorBase" }
%"class.llvm::SmallVectorBase" = type { ptr, ptr, ptr, %"union.llvm::SmallVectorBase::U" }
%"union.llvm::SmallVectorBase::U" = type { x86_fp80 }

; Function Attrs: ssp uwtable
define void @_Z4testv() #0 personality ptr @__gxx_personality_v0 {
; CHECK: @_Z4testv()
; CHECK: invoke.cont:
; CHECK: br i1 true, label %new.notnull.i11, label %if.end.i14
; CHECK: Retry.i10:

entry:
  %sv = alloca %"class.llvm::SmallVector", align 16
  call void @llvm.lifetime.start.p0(ptr %sv) #1
  %FirstEl.i.i.i.i.i.i = getelementptr inbounds %"class.llvm::SmallVector", ptr %sv, i64 0, i32 0, i32 0, i32 0, i32 0, i32 3
  store ptr %FirstEl.i.i.i.i.i.i, ptr %sv, align 16, !tbaa !4
  %EndX.i.i.i.i.i.i = getelementptr inbounds %"class.llvm::SmallVector", ptr %sv, i64 0, i32 0, i32 0, i32 0, i32 0, i32 1
  store ptr %FirstEl.i.i.i.i.i.i, ptr %EndX.i.i.i.i.i.i, align 8, !tbaa !4
  %CapacityX.i.i.i.i.i.i = getelementptr inbounds %"class.llvm::SmallVector", ptr %sv, i64 0, i32 0, i32 0, i32 0, i32 0, i32 2
  %add.ptr.i.i.i.i2.i.i = getelementptr inbounds %"union.llvm::SmallVectorBase::U", ptr %FirstEl.i.i.i.i.i.i, i64 2
  store ptr %add.ptr.i.i.i.i2.i.i, ptr %CapacityX.i.i.i.i.i.i, align 16, !tbaa !4
  %EndX.i = getelementptr inbounds %"class.llvm::SmallVector", ptr %sv, i64 0, i32 0, i32 0, i32 0, i32 0, i32 1
  %0 = load ptr, ptr %EndX.i, align 8, !tbaa !4
  %CapacityX.i = getelementptr inbounds %"class.llvm::SmallVector", ptr %sv, i64 0, i32 0, i32 0, i32 0, i32 0, i32 2
  %cmp.i = icmp ult ptr %0, %add.ptr.i.i.i.i2.i.i
  br i1 %cmp.i, label %Retry.i, label %if.end.i

Retry.i:                                          ; preds = %.noexc, %entry
  %1 = phi ptr [ %0, %entry ], [ %.pre.i, %.noexc ]
  %new.isnull.i = icmp eq ptr %1, null
  br i1 %new.isnull.i, label %invoke.cont, label %new.notnull.i

new.notnull.i:                                    ; preds = %Retry.i
  store i32 1, ptr %1, align 4, !tbaa !5
  br label %invoke.cont

if.end.i:                                         ; preds = %entry
  invoke void @_ZN4llvm15SmallVectorBase8grow_podEmm(ptr %sv, i64 0, i64 4)
          to label %.noexc unwind label %lpad

.noexc:                                           ; preds = %if.end.i
  %.pre.i = load ptr, ptr %EndX.i, align 8, !tbaa !4
  br label %Retry.i

invoke.cont:                                      ; preds = %new.notnull.i, %Retry.i
  %add.ptr.i = getelementptr inbounds i8, ptr %1, i64 4
  store ptr %add.ptr.i, ptr %EndX.i, align 8, !tbaa !4
  %2 = load ptr, ptr %CapacityX.i, align 16, !tbaa !4
  %cmp.i8 = icmp ult ptr %add.ptr.i, %2
  br i1 %cmp.i8, label %new.notnull.i11, label %if.end.i14

Retry.i10:                                        ; preds = %if.end.i14
  %.pre.i13 = load ptr, ptr %EndX.i, align 8, !tbaa !4
  %new.isnull.i9 = icmp eq ptr %.pre.i13, null
  br i1 %new.isnull.i9, label %invoke.cont2, label %new.notnull.i11

new.notnull.i11:                                  ; preds = %invoke.cont, %Retry.i10
  %3 = phi ptr [ %.pre.i13, %Retry.i10 ], [ %add.ptr.i, %invoke.cont ]
  store i32 2, ptr %3, align 4, !tbaa !5
  br label %invoke.cont2

if.end.i14:                                       ; preds = %invoke.cont
  invoke void @_ZN4llvm15SmallVectorBase8grow_podEmm(ptr %sv, i64 0, i64 4)
          to label %Retry.i10 unwind label %lpad

invoke.cont2:                                     ; preds = %new.notnull.i11, %Retry.i10
  %4 = phi ptr [ null, %Retry.i10 ], [ %3, %new.notnull.i11 ]
  %add.ptr.i12 = getelementptr inbounds i8, ptr %4, i64 4
  store ptr %add.ptr.i12, ptr %EndX.i, align 8, !tbaa !4
  invoke void @_Z1gRN4llvm11SmallVectorIiLj8EEE(ptr %sv)
          to label %invoke.cont3 unwind label %lpad

invoke.cont3:                                     ; preds = %invoke.cont2
  %5 = load ptr, ptr %sv, align 16, !tbaa !4
  %cmp.i.i.i.i19 = icmp eq ptr %5, %FirstEl.i.i.i.i.i.i
  br i1 %cmp.i.i.i.i19, label %_ZN4llvm11SmallVectorIiLj8EED1Ev.exit21, label %if.then.i.i.i20

if.then.i.i.i20:                                  ; preds = %invoke.cont3
  call void @free(ptr %5) #1
  br label %_ZN4llvm11SmallVectorIiLj8EED1Ev.exit21

_ZN4llvm11SmallVectorIiLj8EED1Ev.exit21:          ; preds = %invoke.cont3, %if.then.i.i.i20
  call void @llvm.lifetime.end.p0(ptr %sv) #1
  ret void

lpad:                                             ; preds = %if.end.i14, %if.end.i, %invoke.cont2
  %6 = landingpad { ptr, i32 }
          cleanup
  %7 = load ptr, ptr %sv, align 16, !tbaa !4
  %cmp.i.i.i.i = icmp eq ptr %7, %FirstEl.i.i.i.i.i.i
  br i1 %cmp.i.i.i.i, label %eh.resume, label %if.then.i.i.i

if.then.i.i.i:                                    ; preds = %lpad
  call void @free(ptr %7) #1
  br label %eh.resume

eh.resume:                                        ; preds = %if.then.i.i.i, %lpad
  resume { ptr, i32 } %6
}

; Function Attrs: nounwind
declare void @llvm.lifetime.start.p0(ptr nocapture) #1

declare i32 @__gxx_personality_v0(...)

declare void @_Z1gRN4llvm11SmallVectorIiLj8EEE(ptr) #2

; Function Attrs: nounwind
declare void @llvm.lifetime.end.p0(ptr nocapture) #1

declare void @_ZN4llvm15SmallVectorBase8grow_podEmm(ptr, i64, i64) #2

; Function Attrs: nounwind
declare void @free(ptr nocapture) #3

attributes #0 = { ssp uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }
attributes #2 = { "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!0 = !{!"any pointer", !1}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
!3 = !{!"int", !1}
!4 = !{!0, !0, i64 0}
!5 = !{!3, !3, i64 0}
