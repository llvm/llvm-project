; RUN: rm -rf %t && mkdir %t && cd %t

; Generate unsplit module with summary for ThinLTO index-based WPD.
; RUN: opt -thinlto-bc -o summary.o %s

; RUN: llvm-dis -o - summary.o

;; TODO: Implement the fix for WPD in regular or hybrid LTO, and add test coverage.

; Index based WPD
; For `_ZTI7Derived`, the 'llvm-lto2' resolution arguments specifies  `VisibleOutsideSummary` as false
; and `ExportDynamic` as false. The callsite inside @_ZN4Base8dispatchEv
; got devirtualized.
; RUN: llvm-lto2 run summary.o -save-temps -pass-remarks=. \
; RUN:   -o tmp \
; RUN:   --whole-program-visibility-enabled-in-lto=true \
; RUN:   --validate-all-vtables-have-type-infos=true \
; RUN:   --all-vtables-have-type-infos=true \
; RUN:   -r=summary.o,__cxa_pure_virtual, \
; RUN:   -r=summary.o,_ZN8DerivedNC2Ev,x \
; RUN:   -r=summary.o,_ZN4Base8dispatchEv,px \
; RUN:   -r=summary.o,_ZN7DerivedC2Ev, \
; RUN:   -r=summary.o,_ZN8DerivedN5printEv,px \
; RUN:   -r=summary.o,_ZTS4Base, \
; RUN:   -r=summary.o,_ZTV8DerivedN,p \
; RUN:   -r=summary.o,_ZTI8DerivedN,p \
; RUN:   -r=summary.o,_ZTI4Base, \
; RUN:   -r=summary.o,_ZTS8DerivedN,p \
; RUN:   -r=summary.o,_ZTI7Derived, \
; RUN:   -r=summary.o,_ZTV4Base 2>&1 | FileCheck --allow-empty %s --check-prefix=REMARK

; REMARK: single-impl: devirtualized a call to _ZN8DerivedN5printEv 

; Index based WPD
; For `_ZTI7Derived`, the 'llvm-lto2' resolution arguments specifies  `VisibleOutsideSummary` as false
; and `ExportDynamic` as true. The callsite inside @_ZN4Base8dispatchEv won't
; get devirtualized.
; RUN: llvm-lto2  run summary.o -save-temps -pass-remarks=. \
; RUN:   -o tmp \
; RUN:   --whole-program-visibility-enabled-in-lto=true \
; RUN:   --validate-all-vtables-have-type-infos=true \
; RUN:   --all-vtables-have-type-infos=true \
; RUN:   -r=summary.o,__cxa_pure_virtual, \
; RUN:   -r=summary.o,_ZN8DerivedNC2Ev,x \
; RUN:   -r=summary.o,_ZN4Base8dispatchEv,px \
; RUN:   -r=summary.o,_ZN7DerivedC2Ev, \
; RUN:   -r=summary.o,_ZN8DerivedN5printEv,px \
; RUN:   -r=summary.o,_ZTS4Base, \
; RUN:   -r=summary.o,_ZTV8DerivedN,p \
; RUN:   -r=summary.o,_ZTI8DerivedN,p \
; RUN:   -r=summary.o,_ZTI4Base, \
; RUN:   -r=summary.o,_ZTS8DerivedN,p \
; RUN:   -r=summary.o,_ZTI7Derived,d \
; RUN:   -r=summary.o,_ZTV4Base 2>&1 | FileCheck %s --allow-empty --implicit-check-not='single-impl: devirtualized a call to'

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@_ZTV8DerivedN = linkonce_odr hidden constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI8DerivedN, ptr @_ZN8DerivedN5printEv] }, !type !0, !type !1, !type !2, !type !3, !type !4, !type !5, !vcall_visibility !6
@_ZTI8DerivedN = linkonce_odr hidden constant { ptr, ptr, ptr } { ptr null, ptr @_ZTS8DerivedN, ptr @_ZTI7Derived }
@_ZTS8DerivedN = linkonce_odr hidden constant [10 x i8] c"8DerivedN\00", align 1
@_ZTI7Derived = external constant ptr
@_ZTV4Base = linkonce_odr hidden constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI4Base, ptr @__cxa_pure_virtual] }, !type !0, !type !1, !vcall_visibility !6
@_ZTI4Base = linkonce_odr hidden constant { ptr, ptr } { ptr null, ptr @_ZTS4Base }
@_ZTS4Base = linkonce_odr hidden constant [6 x i8] c"4Base\00", align 1

@llvm.used = appending global [1 x ptr] [ptr @_ZN8DerivedNC2Ev], section "llvm.metadata"

define hidden void @_ZN4Base8dispatchEv(ptr %this) {
entry:
  %this.addr = alloca ptr
  store ptr %this, ptr %this.addr
  %this1 = load ptr, ptr %this.addr
  %vtable = load ptr, ptr %this1
  %0 = call i1 @llvm.type.test(ptr %vtable, metadata !"_ZTS7Derived")
  call void @llvm.assume(i1 %0)
  %vfn = getelementptr inbounds ptr, ptr %vtable, i64 0
  %1 = load ptr, ptr %vfn
  call void %1(ptr %this1)
  ret void
}

define linkonce_odr hidden void @_ZN8DerivedNC2Ev(ptr %this) #0 {
entry:
  %this.addr = alloca ptr
  store ptr %this, ptr %this.addr
  %this1 = load ptr, ptr %this.addr
  call void @_ZN7DerivedC2Ev(ptr %this1)
  store ptr getelementptr inbounds inrange(-16, 8) ({ [3 x ptr] }, ptr @_ZTV8DerivedN, i32 0, i32 0, i32 2), ptr %this1
  ret void
}

define linkonce_odr hidden void @_ZN8DerivedN5printEv(ptr %this) #0 {
entry:
  ret void
}

attributes #0 = { noinline optnone }

declare void @__cxa_pure_virtual()
declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)
declare void @_ZN7DerivedC2Ev(ptr)

!0 = !{i64 16, !"_ZTS4Base"}
!1 = !{i64 16, !"_ZTSM4BaseFvvE.virtual"}
!2 = !{i64 16, !"_ZTS7Derived"}
!3 = !{i64 16, !"_ZTSM7DerivedFvvE.virtual"}
!4 = !{i64 16, !"_ZTS8DerivedN"}
!5 = !{i64 16, !"_ZTSM8DerivedNFvvE.virtual"}
!6 = !{i64 0}
