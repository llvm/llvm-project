; RUN: opt -S -Os < %s | FileCheck %s
; RUN: opt -S -aa-pipeline=basic-aa -passes='default<Os>' < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"


; Simple devirt testcase, requires iteration between inliner and GVN.
;  rdar://6295824
define i32 @foo(ptr noalias %p, ptr noalias %q) nounwind ssp {
entry:
  store ptr @bar, ptr %p
  store i64 0, ptr %q
  %tmp3 = load ptr, ptr %p                        ; <ptr> [#uses=1]
  %call = call i32 %tmp3()                        ; <i32> [#uses=1]
  %X = add i32 %call, 4
  ret i32 %X
  
; CHECK-LABEL: @foo(
; CHECK-NEXT: entry:
; CHECK-NEXT: store
; CHECK-NEXT: store
; CHECK-NEXT: ret i32 11
}

define internal i32 @bar() nounwind ssp {
entry:
  ret i32 7
}


;; More complex devirt case, from PR6724
; CHECK: @_Z1gv()
; CHECK-NEXT: entry:
; CHECK-NEXT: ret i32 7

%0 = type { ptr, ptr }
%1 = type { ptr, ptr, i32, i32, ptr, i64, ptr, i64 }
%2 = type { ptr, ptr, ptr }
%struct.A = type { ptr }
%struct.B = type { ptr }
%struct.C = type { [16 x i8] }
%struct.D = type { [16 x i8] }

@_ZTV1D = linkonce_odr constant [6 x ptr] [ptr null, ptr @_ZTI1D, ptr @_ZN1D1fEv, ptr inttoptr (i64 -8 to ptr), ptr @_ZTI1D, ptr @_ZThn8_N1D1fEv] ; <ptr> [#uses=2]
@_ZTVN10__cxxabiv120__si_class_type_infoE = external global ptr ; <ptr> [#uses=1]
@_ZTS1D = linkonce_odr constant [3 x i8] c"1D\00"     ; <ptr> [#uses=1]
@_ZTVN10__cxxabiv121__vmi_class_type_infoE = external global ptr ; <ptr> [#uses=1]
@_ZTS1C = linkonce_odr constant [3 x i8] c"1C\00"     ; <ptr> [#uses=1]
@_ZTVN10__cxxabiv117__class_type_infoE = external global ptr ; <ptr> [#uses=1]
@_ZTS1A = linkonce_odr constant [3 x i8] c"1A\00"     ; <ptr> [#uses=1]
@_ZTI1A = linkonce_odr constant %0 { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS1A } ; <ptr> [#uses=1]
@_ZTS1B = linkonce_odr constant [3 x i8] c"1B\00"     ; <ptr> [#uses=1]
@_ZTI1B = linkonce_odr constant %0 { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS1B } ; <ptr> [#uses=1]
@_ZTI1C = linkonce_odr constant %1 { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv121__vmi_class_type_infoE, i64 2), ptr @_ZTS1C, i32 0, i32 2, ptr @_ZTI1A, i64 2, ptr @_ZTI1B, i64 2050 } ; <ptr> [#uses=1]
@_ZTI1D = linkonce_odr constant %2 { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2), ptr @_ZTS1D, ptr @_ZTI1C } ; <ptr> [#uses=1]
@_ZTV1C = linkonce_odr constant [6 x ptr] [ptr null, ptr @_ZTI1C, ptr @_ZN1C1fEv, ptr inttoptr (i64 -8 to ptr), ptr @_ZTI1C, ptr @_ZThn8_N1C1fEv] ; <ptr> [#uses=2]
@_ZTV1B = linkonce_odr constant [3 x ptr] [ptr null, ptr @_ZTI1B, ptr @_ZN1B1fEv] ; <ptr> [#uses=1]
@_ZTV1A = linkonce_odr constant [3 x ptr] [ptr null, ptr @_ZTI1A, ptr @_ZN1A1fEv] ; <ptr> [#uses=1]

define i32 @_Z1gv() ssp {
entry:
  %d = alloca %struct.C, align 8                  ; <ptr> [#uses=2]
  call void @_ZN1DC1Ev(ptr %d)
  %call = call i32 @_Z1fP1D(ptr %d)        ; <i32> [#uses=1]
  %X = add i32 %call, 3
  ret i32 %X
}

define linkonce_odr void @_ZN1DC1Ev(ptr %this) inlinehint ssp align 2 {
entry:
  call void @_ZN1DC2Ev(ptr %this)
  ret void
}

define internal i32 @_Z1fP1D(ptr %d) ssp {
entry:
  %0 = icmp eq ptr %d, null                ; <i1> [#uses=1]
  br i1 %0, label %cast.end, label %cast.notnull

cast.notnull:                                     ; preds = %entry
  %add.ptr = getelementptr i8, ptr %d, i64 8          ; <ptr> [#uses=1]
  br label %cast.end

cast.end:                                         ; preds = %entry, %cast.notnull
  %1 = phi ptr [ %add.ptr, %cast.notnull ], [ null, %entry ] ; <ptr> [#uses=2]
  %2 = load ptr, ptr %1                ; <ptr> [#uses=1]
  %vfn = getelementptr inbounds ptr, ptr %2, i64 0 ; <ptr> [#uses=1]
  %3 = load ptr, ptr %vfn               ; <ptr> [#uses=1]
  %call = call i32 %3(ptr %1)              ; <i32> [#uses=1]
  ret i32 %call
}

define linkonce_odr i32 @_ZN1D1fEv(ptr %this) ssp align 2 {
entry:
  ret i32 4
}

define linkonce_odr i32 @_ZThn8_N1D1fEv(ptr %this) ssp {
entry:
  %0 = getelementptr inbounds i8, ptr %this, i64 -8      ; <ptr> [#uses=1]
  %call = call i32 @_ZN1D1fEv(ptr %0)      ; <i32> [#uses=1]
  ret i32 %call
}

define linkonce_odr void @_ZN1DC2Ev(ptr %this) inlinehint ssp align 2 {
entry:
  call void @_ZN1CC2Ev(ptr %this)
  %0 = getelementptr inbounds i8, ptr %this, i64 0       ; <ptr> [#uses=1]
  store ptr getelementptr inbounds ([6 x ptr], ptr @_ZTV1D, i64 0, i64 2), ptr %0
  %1 = getelementptr inbounds i8, ptr %this, i64 8       ; <ptr> [#uses=1]
  store ptr getelementptr inbounds ([6 x ptr], ptr @_ZTV1D, i64 0, i64 5), ptr %1
  ret void
}

define linkonce_odr void @_ZN1CC2Ev(ptr %this) inlinehint ssp align 2 {
entry:
  call void @_ZN1AC2Ev(ptr %this)
  %0 = getelementptr inbounds i8, ptr %this, i64 8       ; <ptr> [#uses=1]
  call void @_ZN1BC2Ev(ptr %0)
  %1 = getelementptr inbounds i8, ptr %this, i64 0       ; <ptr> [#uses=1]
  store ptr getelementptr inbounds ([6 x ptr], ptr @_ZTV1C, i64 0, i64 2), ptr %1
  %2 = getelementptr inbounds i8, ptr %this, i64 8       ; <ptr> [#uses=1]
  store ptr getelementptr inbounds ([6 x ptr], ptr @_ZTV1C, i64 0, i64 5), ptr %2
  ret void
}

define linkonce_odr i32 @_ZN1C1fEv(ptr %this) ssp align 2 {
entry:
  ret i32 3
}

define linkonce_odr i32 @_ZThn8_N1C1fEv(ptr %this) {
entry:
  %0 = getelementptr inbounds i8, ptr %this, i64 -8      ; <ptr> [#uses=1]
  %call = call i32 @_ZN1C1fEv(ptr %0)      ; <i32> [#uses=1]
  ret i32 %call
}

define linkonce_odr void @_ZN1AC2Ev(ptr %this) inlinehint ssp align 2 {
entry:
  %0 = getelementptr inbounds i8, ptr %this, i64 0       ; <ptr> [#uses=1]
  store ptr getelementptr inbounds ([3 x ptr], ptr @_ZTV1A, i64 0, i64 2), ptr %0
  ret void
}

define linkonce_odr void @_ZN1BC2Ev(ptr %this) inlinehint ssp align 2 {
entry:
  %0 = getelementptr inbounds i8, ptr %this, i64 0       ; <ptr> [#uses=1]
  store ptr getelementptr inbounds ([3 x ptr], ptr @_ZTV1B, i64 0, i64 2), ptr %0
  ret void
}

define linkonce_odr i32 @_ZN1B1fEv(ptr %this) ssp align 2 {
entry:
  ret i32 2
}

define linkonce_odr i32 @_ZN1A1fEv(ptr %this) ssp align 2 {
entry:
  ret i32 1
}
