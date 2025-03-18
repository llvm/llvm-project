; RUN: rm -rf %t && mkdir %t && cd %t

; Tests that devirtualization is suppressed on a class when its compatible
; class could be referenced from dynamic linker that is not visible to the
; linker.

; Generate unsplit module with summary for ThinLTO index-based WPD.
; RUN: opt -thinlto-bc -o summary.o %s

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
; RUN:   -r=summary.o,_ZTV8DerivedN,p \
; RUN:   -r=summary.o,_ZTI8DerivedN,p \
; RUN:   -r=summary.o,_ZTS8DerivedN,p \
; RUN:   -r=summary.o,_ZTI7Derived, \
; RUN:   2>&1 | FileCheck --allow-empty %s --check-prefix=REMARK


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
; RUN:   -r=summary.o,_ZTV8DerivedN,p \
; RUN:   -r=summary.o,_ZTI8DerivedN,p \
; RUN:   -r=summary.o,_ZTS8DerivedN,p \
; RUN:   -r=summary.o,_ZTI7Derived,d \
; RUN:   2>&1 | FileCheck %s --allow-empty --implicit-check-not='single-impl: devirtualized a call to'


; Hybrid LTO WPD
; RUN: opt  --thinlto-bc --thinlto-split-lto-unit -o hybrid.o %s
; RUN: llvm-lto2 run hybrid.o -save-temps -pass-remarks=. \
; RUN:   -o hybrid \
; RUN:   --whole-program-visibility-enabled-in-lto=true \
; RUN:   --validate-all-vtables-have-type-infos=true \
; RUN:   --all-vtables-have-type-infos=true \
; RUN:   -r=hybrid.o,__cxa_pure_virtual, \
; RUN:   -r=hybrid.o,_ZN8DerivedNC2Ev,x \
; RUN:   -r=hybrid.o,_ZN4Base8dispatchEv,px \
; RUN:   -r=hybrid.o,_ZN7DerivedC2Ev, \
; RUN:   -r=hybrid.o,_ZN8DerivedN5printEv,px \
; RUN:   -r=hybrid.o,_ZTV8DerivedN,p \
; RUN:   -r=hybrid.o,_ZTI8DerivedN,p \
; RUN:   -r=hybrid.o,_ZTS8DerivedN,p \
; RUN:   -r=hybrid.o,_ZTI7Derived, \
; RUN:   -r=hybrid.o,_ZN8DerivedN5printEv,px \
; RUN:   -r=hybrid.o,_ZTV8DerivedN,p \
; RUN:   -r=hybrid.o,_ZTI8DerivedN,p \
; RUN:   2>&1 | FileCheck --allow-empty %s --check-prefix=REMARK

; Hybrid LTO WPD
; RUN: llvm-lto2 run hybrid.o -save-temps -pass-remarks=. \
; RUN:   -o hybrid \
; RUN:   --whole-program-visibility-enabled-in-lto=true \
; RUN:   --validate-all-vtables-have-type-infos=true \
; RUN:   --all-vtables-have-type-infos=true \
; RUN:   -r=hybrid.o,__cxa_pure_virtual, \
; RUN:   -r=hybrid.o,_ZN8DerivedNC2Ev,x \
; RUN:   -r=hybrid.o,_ZN4Base8dispatchEv,px \
; RUN:   -r=hybrid.o,_ZN7DerivedC2Ev, \
; RUN:   -r=hybrid.o,_ZN8DerivedN5printEv,px \
; RUN:   -r=hybrid.o,_ZTV8DerivedN,p \
; RUN:   -r=hybrid.o,_ZTI8DerivedN,p \
; RUN:   -r=hybrid.o,_ZTS8DerivedN,p \
; RUN:   -r=hybrid.o,_ZTI7Derived,d \
; RUN:   -r=hybrid.o,_ZN8DerivedN5printEv,px \
; RUN:   -r=hybrid.o,_ZTV8DerivedN,p \
; RUN:   -r=hybrid.o,_ZTI8DerivedN,p \
; RUN:   2>&1 | FileCheck --allow-empty %s --implicit-check-not='single-impl: devirtualized a call to'


; In regular LTO, global resolutions (as expected) show symbols are visible
; outside summary (when they come from regular LTO module without summaries).
; In the setting of this test case (equivalent of `-Wl,--lto-whole-program-visibility -Wl,--lto-validate-all-vtables-have-type-infos` in lld),
; devirtualization will be suppressed even if the compatible class is not
; referenced from shared libraries. So regular LTO test coverage is not meaningful.

; REMARK: single-impl: devirtualized a call to _ZN8DerivedN5printEv

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@_ZTV8DerivedN = linkonce_odr hidden constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI8DerivedN, ptr @_ZN8DerivedN5printEv] }, !type !0, !type !1, !type !2, !type !3, !vcall_visibility !6
@_ZTI8DerivedN = linkonce_odr hidden constant { ptr, ptr, ptr } { ptr null, ptr @_ZTS8DerivedN, ptr @_ZTI7Derived }
@_ZTS8DerivedN = linkonce_odr hidden constant [10 x i8] c"8DerivedN\00", align 1
@_ZTI7Derived = external constant ptr

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

!0 = !{i64 16, !"_ZTS7Derived"}
!1 = !{i64 16, !"_ZTSM7DerivedFvvE.virtual"}
!2 = !{i64 16, !"_ZTS8DerivedN"}
!3 = !{i64 16, !"_ZTSM8DerivedNFvvE.virtual"}
;!4 = !{i64 16, !"_ZTS8DerivedN"}
;!5 = !{i64 16, !"_ZTSM8DerivedNFvvE.virtual"}
!6 = !{i64 0}
