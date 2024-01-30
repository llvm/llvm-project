; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec -mtriple powerpc64-ibm-aix-xcoff \
; RUN:     --code-model=small -mattr=+aix-func-use-tls-local-dynamic < %s | FileCheck %s --check-prefixes=SMALL64,SMALL
; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec -mtriple powerpc64-ibm-aix-xcoff \
; RUN:     --code-model=large -mattr=+aix-func-use-tls-local-dynamic < %s | FileCheck %s --check-prefixes=LARGE64,LARGE
; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff \
; RUN:     --code-model=small -mattr=+aix-func-use-tls-local-dynamic < %s | FileCheck %s --check-prefixes=SMALL32,SMALL
; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff \
; RUN:     --code-model=large -mattr=+aix-func-use-tls-local-dynamic < %s | FileCheck %s --check-prefixes=LARGE32,LARGE

@TIInit = internal thread_local(initialexec) global i32 42, align 4
@TIUninit = internal thread_local(initialexec) global i32 0, align 4

define i32 @loadTIInit() {
; SMALL-LABEL:  loadTIInit:
; SMALL64:      ld [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL:        bla .__tls_get_mod[PR]
; SMALL64:      ld [[OffsetR:[0-9]+]], [[TIInitL:L..C[0-9]+]](2)
; SMALL32:      lwz [[OffsetR:[0-9]+]], [[TIInitL:L..C[0-9]+]](2)
; SMALL:        lwzx [[TIInitValR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
;
; LARGE-LABEL:  loadTIInit:
; LARGE64:      addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE:        addis [[OffsetHR:[0-9]+]], [[TIInitL:L..C[0-9]+]]@u(2)
; LARGE32:      addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE:        bla .__tls_get_mod[PR]
; LARGE64:      ld [[OffsetR:[0-9]+]], [[TIInitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE32:      lwz [[OffsetR:[0-9]+]], [[TIInitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE:        lwzx [[TIInitValR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TIInit)
  %1 = load i32, ptr %0, align 4
  ret i32 %1
}

define void @storeTIInit(i32 noundef signext %i) {
; SMALL-LABEL:  storeTIInit:
; SMALL64:      ld [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL:        bla .__tls_get_mod[PR]
; SMALL64:      ld [[OffsetR:[0-9]+]], [[TIInitL:L..C[0-9]+]](2)
; SMALL32:      lwz [[OffsetR:[0-9]+]], [[TIInitL:L..C[0-9]+]](2)
; SMALL:        stwx [[TIInitValR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
;
; LARGE-LABEL:  storeTIInit:
; LARGE64:      addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE:        addis [[OffsetHR:[0-9]+]], [[TIInitL:L..C[0-9]+]]@u(2)
; LARGE32:      addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE:        bla .__tls_get_mod[PR]
; LARGE64:      ld [[OffsetR:[0-9]+]], [[TIInitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE32:      lwz [[OffsetR:[0-9]+]], [[TIInitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE:        stwx [[TIInitValR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TIInit)
  store i32 %i, ptr %0, align 4
  ret void
}

define i32 @loadTIUninit() {
; SMALL-LABEL:  loadTIUninit:
; SMALL64:      ld [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL:        bla .__tls_get_mod[PR]
; SMALL64:      ld [[OffsetR:[0-9]+]], [[TIUninitL:L..C[0-9]+]](2)
; SMALL32:      lwz [[OffsetR:[0-9]+]], [[TIUninitL:L..C[0-9]+]](2)
; SMALL:        lwzx [[TIUninitValR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
;
; LARGE-LABEL:  loadTIUninit:
; LARGE64:      addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE:        addis [[OffsetHR:[0-9]+]], [[TIUninitL:L..C[0-9]+]]@u(2)
; LARGE32:      addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE:        bla .__tls_get_mod[PR]
; LARGE64:      ld [[OffsetR:[0-9]+]], [[TIUninitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE32:      lwz [[OffsetR:[0-9]+]], [[TIUninitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE:        lwzx [[TIUninitValR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TIUninit)
  %1 = load i32, ptr %0, align 4
  ret i32 %1
}

define void @storeTIUninit(i32 noundef signext %i) {
; SMALL-LABEL:  storeTIUninit:
; SMALL64:      ld [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL:        bla .__tls_get_mod[PR]
; SMALL64:      ld [[OffsetR:[0-9]+]], [[TIUninitL:L..C[0-9]+]](2)
; SMALL32:      lwz [[OffsetR:[0-9]+]], [[TIUninitL:L..C[0-9]+]](2)
; SMALL:        stwx [[TIUninitValR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
;
; LARGE-LABEL:  storeTIUninit:
; LARGE64:      addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE:        addis [[OffsetHR:[0-9]+]], [[TIUninitL:L..C[0-9]+]]@u(2)
; LARGE32:      addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE:        bla .__tls_get_mod[PR]
; LARGE64:      ld [[OffsetR:[0-9]+]], [[TIUninitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE32:      lwz [[OffsetR:[0-9]+]], [[TIUninitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE:        stwx [[TIUninitValR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TIUninit)
  store i32 %i, ptr %0, align 4
  ret void
}

; SMALL:          .extern .__tls_get_mod[PR]
; LARGE:          .extern .__tls_get_mod[PR]
; SMALL-NOT:      .extern _Renamed..5f24__TLSML[TC]
; LARGE-NOT:      .extern _Renamed..5f24__TLSML[TC]

; SMALL:        [[ModuleHandleL]]:
; SMALL-NEXT:   .tc _Renamed..5f24__TLSML[TC],_Renamed..5f24__TLSML[TC]@ml
; SMALL-NEXT:   .rename _Renamed..5f24__TLSML[TC],"_$TLSML"
; SMALL:        [[TIInitL]]:
; SMALL-NEXT:   .tc _Renamed..5f24__TLSLD.TIInit[TC],TIInit[TL]@ld
; SMALL-NEXT:   .rename _Renamed..5f24__TLSLD.TIInit[TC],"TIInit"
; SMALL:        [[TIUninitL]]:
; SMALL-NEXT:   .tc _Renamed..5f24__TLSLD.TIUninit[TC],TIUninit[UL]@ld
; SMALL-NEXT:   .rename _Renamed..5f24__TLSLD.TIUninit[TC],"TIUninit"

; LARGE64:        [[ModuleHandleL]]:
; LARGE64-NEXT:   .tc _Renamed..5f24__TLSML[TC],_Renamed..5f24__TLSML[TC]@ml
; LARGE64-NEXT:   .rename _Renamed..5f24__TLSML[TC],"_$TLSML"
; LARGE64:        [[TIInitL]]:
; LARGE64-NEXT:   .tc _Renamed..5f24__TLSLD.TIInit[TE],TIInit[TL]@ld
; LARGE64-NEXT:   .rename _Renamed..5f24__TLSLD.TIInit[TE],"TIInit"
; LARGE64:        [[TIUninitL]]:
; LARGE64-NEXT:   .tc _Renamed..5f24__TLSLD.TIUninit[TE],TIUninit[UL]@ld
; LARGE64-NEXT:   .rename _Renamed..5f24__TLSLD.TIUninit[TE],"TIUninit"

; LARGE32:        [[TIInitL]]:
; LARGE32-NEXT:   .tc _Renamed..5f24__TLSLD.TIInit[TE],TIInit[TL]@ld
; LARGE32-NEXT:   .rename _Renamed..5f24__TLSLD.TIInit[TE],"TIInit"
; LARGE32:        [[ModuleHandleL]]:
; LARGE32-NEXT:   .tc _Renamed..5f24__TLSML[TC],_Renamed..5f24__TLSML[TC]@ml
; LARGE32-NEXT:   .rename _Renamed..5f24__TLSML[TC],"_$TLSML"
; LARGE32:        [[TIUninitL]]:
; LARGE32-NEXT:   .tc _Renamed..5f24__TLSLD.TIUninit[TE],TIUninit[UL]@ld
; LARGE32-NEXT:   .rename _Renamed..5f24__TLSLD.TIUninit[TE],"TIUninit"

declare nonnull ptr @llvm.threadlocal.address.p0(ptr nonnull)
