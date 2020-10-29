; RUN: opt -S -Os < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"


; Simple devirt testcase, requires iteration between inliner and GVN.
;  rdar://6295824
define i32 @foo(i32 ()** noalias %p, i64* noalias %q) nounwind ssp {
entry:
  store i32 ()* @bar, i32 ()** %p
  store i64 0, i64* %q
  %tmp3 = load i32 ()*, i32 ()** %p                        ; <i32 ()*> [#uses=1]
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

%0 = type { i8*, i8* }
%1 = type { i8*, i8*, i32, i32, i8*, i64, i8*, i64 }
%2 = type { i8*, i8*, i8* }
%struct.A = type { i8** }
%struct.B = type { i8** }
%struct.C = type { [16 x i8] }
%struct.D = type { [16 x i8] }

@_ZTV1D = linkonce_odr constant [6 x i8*] [i8* null, i8* bitcast (%2* @_ZTI1D to i8*), i8* bitcast (i32 (%struct.C*)* @_ZN1D1fEv to i8*), i8* inttoptr (i64 -8 to i8*), i8* bitcast (%2* @_ZTI1D to i8*), i8* bitcast (i32 (%struct.C*)* @_ZThn8_N1D1fEv to i8*)] ; <[6 x i8*]*> [#uses=2]
@_ZTVN10__cxxabiv120__si_class_type_infoE = external global i8* ; <i8**> [#uses=1]
@_ZTS1D = linkonce_odr constant [3 x i8] c"1D\00"     ; <[3 x i8]*> [#uses=1]
@_ZTVN10__cxxabiv121__vmi_class_type_infoE = external global i8* ; <i8**> [#uses=1]
@_ZTS1C = linkonce_odr constant [3 x i8] c"1C\00"     ; <[3 x i8]*> [#uses=1]
@_ZTVN10__cxxabiv117__class_type_infoE = external global i8* ; <i8**> [#uses=1]
@_ZTS1A = linkonce_odr constant [3 x i8] c"1A\00"     ; <[3 x i8]*> [#uses=1]
@_ZTI1A = linkonce_odr constant %0 { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv117__class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @_ZTS1A, i32 0, i32 0) } ; <%0*> [#uses=1]
@_ZTS1B = linkonce_odr constant [3 x i8] c"1B\00"     ; <[3 x i8]*> [#uses=1]
@_ZTI1B = linkonce_odr constant %0 { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv117__class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @_ZTS1B, i32 0, i32 0) } ; <%0*> [#uses=1]
@_ZTI1C = linkonce_odr constant %1 { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv121__vmi_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @_ZTS1C, i32 0, i32 0), i32 0, i32 2, i8* bitcast (%0* @_ZTI1A to i8*), i64 2, i8* bitcast (%0* @_ZTI1B to i8*), i64 2050 } ; <%1*> [#uses=1]
@_ZTI1D = linkonce_odr constant %2 { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @_ZTS1D, i32 0, i32 0), i8* bitcast (%1* @_ZTI1C to i8*) } ; <%2*> [#uses=1]
@_ZTV1C = linkonce_odr constant [6 x i8*] [i8* null, i8* bitcast (%1* @_ZTI1C to i8*), i8* bitcast (i32 (%struct.C*)* @_ZN1C1fEv to i8*), i8* inttoptr (i64 -8 to i8*), i8* bitcast (%1* @_ZTI1C to i8*), i8* bitcast (i32 (%struct.C*)* @_ZThn8_N1C1fEv to i8*)] ; <[6 x i8*]*> [#uses=2]
@_ZTV1B = linkonce_odr constant [3 x i8*] [i8* null, i8* bitcast (%0* @_ZTI1B to i8*), i8* bitcast (i32 (%struct.A*)* @_ZN1B1fEv to i8*)] ; <[3 x i8*]*> [#uses=1]
@_ZTV1A = linkonce_odr constant [3 x i8*] [i8* null, i8* bitcast (%0* @_ZTI1A to i8*), i8* bitcast (i32 (%struct.A*)* @_ZN1A1fEv to i8*)] ; <[3 x i8*]*> [#uses=1]

define i32 @_Z1gv() ssp {
entry:
  %d = alloca %struct.C, align 8                  ; <%struct.C*> [#uses=2]
  call void @_ZN1DC1Ev(%struct.C* %d)
  %call = call i32 @_Z1fP1D(%struct.C* %d)        ; <i32> [#uses=1]
  %X = add i32 %call, 3
  ret i32 %X
}

define linkonce_odr void @_ZN1DC1Ev(%struct.C* %this) inlinehint ssp align 2 {
entry:
  call void @_ZN1DC2Ev(%struct.C* %this)
  ret void
}

define internal i32 @_Z1fP1D(%struct.C* %d) ssp {
entry:
  %0 = icmp eq %struct.C* %d, null                ; <i1> [#uses=1]
  br i1 %0, label %cast.end, label %cast.notnull

cast.notnull:                                     ; preds = %entry
  %1 = bitcast %struct.C* %d to i8*               ; <i8*> [#uses=1]
  %add.ptr = getelementptr i8, i8* %1, i64 8          ; <i8*> [#uses=1]
  %2 = bitcast i8* %add.ptr to %struct.A*         ; <%struct.A*> [#uses=1]
  br label %cast.end

cast.end:                                         ; preds = %entry, %cast.notnull
  %3 = phi %struct.A* [ %2, %cast.notnull ], [ null, %entry ] ; <%struct.A*> [#uses=2]
  %4 = bitcast %struct.A* %3 to i32 (%struct.A*)*** ; <i32 (%struct.A*)***> [#uses=1]
  %5 = load i32 (%struct.A*)**, i32 (%struct.A*)*** %4                ; <i32 (%struct.A*)**> [#uses=1]
  %vfn = getelementptr inbounds i32 (%struct.A*)*, i32 (%struct.A*)** %5, i64 0 ; <i32 (%struct.A*)**> [#uses=1]
  %6 = load i32 (%struct.A*)*, i32 (%struct.A*)** %vfn               ; <i32 (%struct.A*)*> [#uses=1]
  %call = call i32 %6(%struct.A* %3)              ; <i32> [#uses=1]
  ret i32 %call
}

define linkonce_odr i32 @_ZN1D1fEv(%struct.C* %this) ssp align 2 {
entry:
  ret i32 4
}

define linkonce_odr i32 @_ZThn8_N1D1fEv(%struct.C* %this) {
entry:
  %0 = bitcast %struct.C* %this to i8*            ; <i8*> [#uses=1]
  %1 = getelementptr inbounds i8, i8* %0, i64 -8      ; <i8*> [#uses=1]
  %2 = bitcast i8* %1 to %struct.C*               ; <%struct.C*> [#uses=1]
  %call = call i32 @_ZN1D1fEv(%struct.C* %2)      ; <i32> [#uses=1]
  ret i32 %call
}

define linkonce_odr void @_ZN1DC2Ev(%struct.C* %this) inlinehint ssp align 2 {
entry:
  call void @_ZN1CC2Ev(%struct.C* %this)
  %0 = bitcast %struct.C* %this to i8*            ; <i8*> [#uses=1]
  %1 = getelementptr inbounds i8, i8* %0, i64 0       ; <i8*> [#uses=1]
  %2 = bitcast i8* %1 to i8***                    ; <i8***> [#uses=1]
  store i8** getelementptr inbounds ([6 x i8*], [6 x i8*]* @_ZTV1D, i64 0, i64 2), i8*** %2
  %3 = bitcast %struct.C* %this to i8*            ; <i8*> [#uses=1]
  %4 = getelementptr inbounds i8, i8* %3, i64 8       ; <i8*> [#uses=1]
  %5 = bitcast i8* %4 to i8***                    ; <i8***> [#uses=1]
  store i8** getelementptr inbounds ([6 x i8*], [6 x i8*]* @_ZTV1D, i64 0, i64 5), i8*** %5
  ret void
}

define linkonce_odr void @_ZN1CC2Ev(%struct.C* %this) inlinehint ssp align 2 {
entry:
  %0 = bitcast %struct.C* %this to %struct.A*     ; <%struct.A*> [#uses=1]
  call void @_ZN1AC2Ev(%struct.A* %0)
  %1 = bitcast %struct.C* %this to i8*            ; <i8*> [#uses=1]
  %2 = getelementptr inbounds i8, i8* %1, i64 8       ; <i8*> [#uses=1]
  %3 = bitcast i8* %2 to %struct.A*               ; <%struct.A*> [#uses=1]
  call void @_ZN1BC2Ev(%struct.A* %3)
  %4 = bitcast %struct.C* %this to i8*            ; <i8*> [#uses=1]
  %5 = getelementptr inbounds i8, i8* %4, i64 0       ; <i8*> [#uses=1]
  %6 = bitcast i8* %5 to i8***                    ; <i8***> [#uses=1]
  store i8** getelementptr inbounds ([6 x i8*], [6 x i8*]* @_ZTV1C, i64 0, i64 2), i8*** %6
  %7 = bitcast %struct.C* %this to i8*            ; <i8*> [#uses=1]
  %8 = getelementptr inbounds i8, i8* %7, i64 8       ; <i8*> [#uses=1]
  %9 = bitcast i8* %8 to i8***                    ; <i8***> [#uses=1]
  store i8** getelementptr inbounds ([6 x i8*], [6 x i8*]* @_ZTV1C, i64 0, i64 5), i8*** %9
  ret void
}

define linkonce_odr i32 @_ZN1C1fEv(%struct.C* %this) ssp align 2 {
entry:
  ret i32 3
}

define linkonce_odr i32 @_ZThn8_N1C1fEv(%struct.C* %this) {
entry:
  %0 = bitcast %struct.C* %this to i8*            ; <i8*> [#uses=1]
  %1 = getelementptr inbounds i8, i8* %0, i64 -8      ; <i8*> [#uses=1]
  %2 = bitcast i8* %1 to %struct.C*               ; <%struct.C*> [#uses=1]
  %call = call i32 @_ZN1C1fEv(%struct.C* %2)      ; <i32> [#uses=1]
  ret i32 %call
}

define linkonce_odr void @_ZN1AC2Ev(%struct.A* %this) inlinehint ssp align 2 {
entry:
  %0 = bitcast %struct.A* %this to i8*            ; <i8*> [#uses=1]
  %1 = getelementptr inbounds i8, i8* %0, i64 0       ; <i8*> [#uses=1]
  %2 = bitcast i8* %1 to i8***                    ; <i8***> [#uses=1]
  store i8** getelementptr inbounds ([3 x i8*], [3 x i8*]* @_ZTV1A, i64 0, i64 2), i8*** %2
  ret void
}

define linkonce_odr void @_ZN1BC2Ev(%struct.A* %this) inlinehint ssp align 2 {
entry:
  %0 = bitcast %struct.A* %this to i8*            ; <i8*> [#uses=1]
  %1 = getelementptr inbounds i8, i8* %0, i64 0       ; <i8*> [#uses=1]
  %2 = bitcast i8* %1 to i8***                    ; <i8***> [#uses=1]
  store i8** getelementptr inbounds ([3 x i8*], [3 x i8*]* @_ZTV1B, i64 0, i64 2), i8*** %2
  ret void
}

define linkonce_odr i32 @_ZN1B1fEv(%struct.A* %this) ssp align 2 {
entry:
  ret i32 2
}

define linkonce_odr i32 @_ZN1A1fEv(%struct.A* %this) ssp align 2 {
entry:
  ret i32 1
}
