; XFAIL: *
; RUN: opt -basic-aa -newgvn -S < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

%"class.llvm::SmallVector" = type { %"class.llvm::SmallVectorImpl", [1 x %"union.llvm::SmallVectorBase::U"] }
%"class.llvm::SmallVectorImpl" = type { %"class.llvm::SmallVectorTemplateBase" }
%"class.llvm::SmallVectorTemplateBase" = type { %"class.llvm::SmallVectorTemplateCommon" }
%"class.llvm::SmallVectorTemplateCommon" = type { %"class.llvm::SmallVectorBase" }
%"class.llvm::SmallVectorBase" = type { i8*, i8*, i8*, %"union.llvm::SmallVectorBase::U" }
%"union.llvm::SmallVectorBase::U" = type { x86_fp80 }

; Function Attrs: ssp uwtable
define void @_Z4testv() #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; CHECK: @_Z4testv()
; CHECK: invoke.cont:
; CHECK: br i1 true, label %new.notnull.i11, label %if.end.i14
; CHECK: Retry.i10:

entry:
  %sv = alloca %"class.llvm::SmallVector", align 16
  %0 = bitcast %"class.llvm::SmallVector"* %sv to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* %0) #1
  %BeginX.i.i.i.i.i.i = getelementptr inbounds %"class.llvm::SmallVector", %"class.llvm::SmallVector"* %sv, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %FirstEl.i.i.i.i.i.i = getelementptr inbounds %"class.llvm::SmallVector", %"class.llvm::SmallVector"* %sv, i64 0, i32 0, i32 0, i32 0, i32 0, i32 3
  %1 = bitcast %"union.llvm::SmallVectorBase::U"* %FirstEl.i.i.i.i.i.i to i8*
  store i8* %1, i8** %BeginX.i.i.i.i.i.i, align 16, !tbaa !4
  %EndX.i.i.i.i.i.i = getelementptr inbounds %"class.llvm::SmallVector", %"class.llvm::SmallVector"* %sv, i64 0, i32 0, i32 0, i32 0, i32 0, i32 1
  store i8* %1, i8** %EndX.i.i.i.i.i.i, align 8, !tbaa !4
  %CapacityX.i.i.i.i.i.i = getelementptr inbounds %"class.llvm::SmallVector", %"class.llvm::SmallVector"* %sv, i64 0, i32 0, i32 0, i32 0, i32 0, i32 2
  %add.ptr.i.i.i.i2.i.i = getelementptr inbounds %"union.llvm::SmallVectorBase::U", %"union.llvm::SmallVectorBase::U"* %FirstEl.i.i.i.i.i.i, i64 2
  %add.ptr.i.i.i.i.i.i = bitcast %"union.llvm::SmallVectorBase::U"* %add.ptr.i.i.i.i2.i.i to i8*
  store i8* %add.ptr.i.i.i.i.i.i, i8** %CapacityX.i.i.i.i.i.i, align 16, !tbaa !4
  %EndX.i = getelementptr inbounds %"class.llvm::SmallVector", %"class.llvm::SmallVector"* %sv, i64 0, i32 0, i32 0, i32 0, i32 0, i32 1
  %2 = load i8*, i8** %EndX.i, align 8, !tbaa !4
  %CapacityX.i = getelementptr inbounds %"class.llvm::SmallVector", %"class.llvm::SmallVector"* %sv, i64 0, i32 0, i32 0, i32 0, i32 0, i32 2
  %cmp.i = icmp ult i8* %2, %add.ptr.i.i.i.i.i.i
  br i1 %cmp.i, label %Retry.i, label %if.end.i

Retry.i:                                          ; preds = %.noexc, %entry
  %3 = phi i8* [ %2, %entry ], [ %.pre.i, %.noexc ]
  %new.isnull.i = icmp eq i8* %3, null
  br i1 %new.isnull.i, label %invoke.cont, label %new.notnull.i

new.notnull.i:                                    ; preds = %Retry.i
  %4 = bitcast i8* %3 to i32*
  store i32 1, i32* %4, align 4, !tbaa !5
  br label %invoke.cont

if.end.i:                                         ; preds = %entry
  %5 = getelementptr inbounds %"class.llvm::SmallVector", %"class.llvm::SmallVector"* %sv, i64 0, i32 0, i32 0, i32 0, i32 0
  invoke void @_ZN4llvm15SmallVectorBase8grow_podEmm(%"class.llvm::SmallVectorBase"* %5, i64 0, i64 4)
          to label %.noexc unwind label %lpad

.noexc:                                           ; preds = %if.end.i
  %.pre.i = load i8*, i8** %EndX.i, align 8, !tbaa !4
  br label %Retry.i

invoke.cont:                                      ; preds = %new.notnull.i, %Retry.i
  %add.ptr.i = getelementptr inbounds i8, i8* %3, i64 4
  store i8* %add.ptr.i, i8** %EndX.i, align 8, !tbaa !4
  %6 = load i8*, i8** %CapacityX.i, align 16, !tbaa !4
  %cmp.i8 = icmp ult i8* %add.ptr.i, %6
  br i1 %cmp.i8, label %new.notnull.i11, label %if.end.i14

Retry.i10:                                        ; preds = %if.end.i14
  %.pre.i13 = load i8*, i8** %EndX.i, align 8, !tbaa !4
  %new.isnull.i9 = icmp eq i8* %.pre.i13, null
  br i1 %new.isnull.i9, label %invoke.cont2, label %new.notnull.i11

new.notnull.i11:                                  ; preds = %invoke.cont, %Retry.i10
  %7 = phi i8* [ %.pre.i13, %Retry.i10 ], [ %add.ptr.i, %invoke.cont ]
  %8 = bitcast i8* %7 to i32*
  store i32 2, i32* %8, align 4, !tbaa !5
  br label %invoke.cont2

if.end.i14:                                       ; preds = %invoke.cont
  %9 = getelementptr inbounds %"class.llvm::SmallVector", %"class.llvm::SmallVector"* %sv, i64 0, i32 0, i32 0, i32 0, i32 0
  invoke void @_ZN4llvm15SmallVectorBase8grow_podEmm(%"class.llvm::SmallVectorBase"* %9, i64 0, i64 4)
          to label %Retry.i10 unwind label %lpad

invoke.cont2:                                     ; preds = %new.notnull.i11, %Retry.i10
  %10 = phi i8* [ null, %Retry.i10 ], [ %7, %new.notnull.i11 ]
  %add.ptr.i12 = getelementptr inbounds i8, i8* %10, i64 4
  store i8* %add.ptr.i12, i8** %EndX.i, align 8, !tbaa !4
  invoke void @_Z1gRN4llvm11SmallVectorIiLj8EEE(%"class.llvm::SmallVector"* %sv)
          to label %invoke.cont3 unwind label %lpad

invoke.cont3:                                     ; preds = %invoke.cont2
  %11 = load i8*, i8** %BeginX.i.i.i.i.i.i, align 16, !tbaa !4
  %cmp.i.i.i.i19 = icmp eq i8* %11, %1
  br i1 %cmp.i.i.i.i19, label %_ZN4llvm11SmallVectorIiLj8EED1Ev.exit21, label %if.then.i.i.i20

if.then.i.i.i20:                                  ; preds = %invoke.cont3
  call void @free(i8* %11) #1
  br label %_ZN4llvm11SmallVectorIiLj8EED1Ev.exit21

_ZN4llvm11SmallVectorIiLj8EED1Ev.exit21:          ; preds = %invoke.cont3, %if.then.i.i.i20
  call void @llvm.lifetime.end.p0i8(i64 64, i8* %0) #1
  ret void

lpad:                                             ; preds = %if.end.i14, %if.end.i, %invoke.cont2
  %12 = landingpad { i8*, i32 }
          cleanup
  %13 = load i8*, i8** %BeginX.i.i.i.i.i.i, align 16, !tbaa !4
  %cmp.i.i.i.i = icmp eq i8* %13, %1
  br i1 %cmp.i.i.i.i, label %eh.resume, label %if.then.i.i.i

if.then.i.i.i:                                    ; preds = %lpad
  call void @free(i8* %13) #1
  br label %eh.resume

eh.resume:                                        ; preds = %if.then.i.i.i, %lpad
  resume { i8*, i32 } %12
}

; Function Attrs: nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

declare i32 @__gxx_personality_v0(...)

declare void @_Z1gRN4llvm11SmallVectorIiLj8EEE(%"class.llvm::SmallVector"*) #2

; Function Attrs: nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

declare void @_ZN4llvm15SmallVectorBase8grow_podEmm(%"class.llvm::SmallVectorBase"*, i64, i64) #2

; Function Attrs: nounwind
declare void @free(i8* nocapture) #3

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
