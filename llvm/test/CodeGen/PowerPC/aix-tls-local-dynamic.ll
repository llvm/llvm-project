; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec -mtriple powerpc64-ibm-aix-xcoff \
; RUN:     --code-model=small < %s | FileCheck %s --check-prefixes=SMALL64,SMALL
; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec -mtriple powerpc64-ibm-aix-xcoff \
; RUN:     --code-model=large < %s | FileCheck %s --check-prefixes=LARGE64,LARGE
; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff \
; RUN:     --code-model=small < %s | FileCheck %s --check-prefixes=SMALL32,SMALL
; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff \
; RUN:     --code-model=large < %s | FileCheck %s --check-prefixes=LARGE32,LARGE

@TGInit = thread_local(localdynamic) global i32 42, align 4
@TGUninit = thread_local(localdynamic) global i32 0, align 4
@TIInit = internal thread_local(localdynamic) global i32 42, align 4
@TIUninit = internal thread_local(localdynamic) global i32 0, align 4
@TWInit = weak thread_local(localdynamic) global i32 42, align 4
@TWUninit = weak thread_local(localdynamic) global i32 0, align 4

define i32 @loadTGInit() {
; SMALL-LABEL:  loadTGInit:
; SMALL64:      ld [[OffsetR:[0-9]+]], [[TGInitL:L..C[0-9]+]](2)
; SMALL32:      lwz [[OffsetR:[0-9]+]], [[TGInitL:L..C[0-9]+]](2)
; SMALL64:      ld [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL:        bla .__tls_get_mod[PR]
; SMALL:        add [[TGInitAddrR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
; SMALL:        lwz [[TGInitValR:[0-9]+]], 0([[TGInitAddrR]])
;
; LARGE-LABEL:  loadTGInit:
; LARGE:        addis [[OffsetHR:[0-9]+]], [[TGInitL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[OffsetR:[0-9]+]], [[TGInitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE32:      lwz [[OffsetR:[0-9]+]], [[TGInitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE:        addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE:        bla .__tls_get_mod[PR]
; LARGE:        add [[TGInitAddrR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
; LARGE:        lwz [[TGInitValR:[0-9]+]], 0([[TGInitAddrR]])
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TGInit)
  %1 = load i32, ptr %0, align 4
  ret i32 %1
}

define void @storeTGInit(i32 noundef signext %i) {
; SMALL-LABEL:  storeTGInit:
; SMALL64:      ld [[OffsetR:[0-9]+]], [[TGInitL:L..C[0-9]+]](2)
; SMALL32:      lwz [[OffsetR:[0-9]+]], [[TGInitL:L..C[0-9]+]](2)
; SMALL64:      ld [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL:        bla .__tls_get_mod[PR]
; SMALL:        add [[TGInitAddrR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
; SMALL:        stw [[TGInitValR:[0-9]+]], 0([[TGInitAddrR]])
;
; LARGE-LABEL:  storeTGInit:
; LARGE:        addis [[OffsetHR:[0-9]+]], [[TGInitL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[OffsetR:[0-9]+]], [[TGInitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE32:      lwz [[OffsetR:[0-9]+]], [[TGInitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE:        addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE:        bla .__tls_get_mod[PR]
; LARGE:        add [[TGInitAddrR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
; LARGE:        stw [[TGInitValR:[0-9]+]], 0([[TGInitAddrR]])
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TGInit)
  store i32 %i, ptr %0, align 4
  ret void
}

define i32 @loadTGUninit() {
; SMALL-LABEL:  loadTGUninit:
; SMALL64:      ld [[OffsetR:[0-9]+]], [[TGUninitL:L..C[0-9]+]](2)
; SMALL32:      lwz [[OffsetR:[0-9]+]], [[TGUninitL:L..C[0-9]+]](2)
; SMALL64:      ld [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL:        bla .__tls_get_mod[PR]
; SMALL:        add [[TGUninitAddrR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
; SMALL:        lwz [[TGUninitValR:[0-9]+]], 0([[TGUninitAddrR]])
;
; LARGE-LABEL:  loadTGUninit:
; LARGE:        addis [[OffsetHR:[0-9]+]], [[TGUninitL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[OffsetR:[0-9]+]], [[TGUninitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE32:      lwz [[OffsetR:[0-9]+]], [[TGUninitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE:        addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE:        bla .__tls_get_mod[PR]
; LARGE:        add [[TGUninitAddrR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
; LARGE:        lwz [[TGUninitValR:[0-9]+]], 0([[TGUninitAddrR]])
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TGUninit)
  %1 = load i32, ptr %0, align 4
  ret i32 %1
}

define void @storeTGUninit(i32 noundef signext %i) {
; SMALL-LABEL:  storeTGUninit:
; SMALL64:      ld [[OffsetR:[0-9]+]], [[TGUninitL:L..C[0-9]+]](2)
; SMALL32:      lwz [[OffsetR:[0-9]+]], [[TGUninitL:L..C[0-9]+]](2)
; SMALL64:      ld [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL:        bla .__tls_get_mod[PR]
; SMALL:        add [[TGUninitAddrR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
; SMALL:        stw [[TGUninitValR:[0-9]+]], 0([[TGUninitAddrR]])
;
; LARGE-LABEL:  storeTGUninit:
; LARGE:        addis [[OffsetHR:[0-9]+]], [[TGUninitL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[OffsetR:[0-9]+]], [[TGUninitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE32:      lwz [[OffsetR:[0-9]+]], [[TGUninitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE:        addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE:        bla .__tls_get_mod[PR]
; LARGE:        add [[TGUninitAddrR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
; LARGE:        stw [[TGUninitValR:[0-9]+]], 0([[TGUninitAddrR]])
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TGUninit)
  store i32 %i, ptr %0, align 4
  ret void
}

define i32 @loadTIInit() {
; SMALL-LABEL:  loadTIInit:
; SMALL64:      ld [[OffsetR:[0-9]+]], [[TIInitL:L..C[0-9]+]](2)
; SMALL32:      lwz [[OffsetR:[0-9]+]], [[TIInitL:L..C[0-9]+]](2)
; SMALL64:      ld [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL:        bla .__tls_get_mod[PR]
; SMALL:        add [[TIInitAddrR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
; SMALL:        lwz [[TIInitValR:[0-9]+]], 0([[TIInitAddrR]])
;
; LARGE-LABEL:  loadTIInit:
; LARGE:        addis [[OffsetHR:[0-9]+]], [[TIInitL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[OffsetR:[0-9]+]], [[TIInitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE32:      lwz [[OffsetR:[0-9]+]], [[TIInitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE:        addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE:        bla .__tls_get_mod[PR]
; LARGE:        add [[TIInitAddrR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
; LARGE:        lwz [[TIInitValR:[0-9]+]], 0([[TIInitAddrR]])
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TIInit)
  %1 = load i32, ptr %0, align 4
  ret i32 %1
}

define void @storeTIInit(i32 noundef signext %i) {
; SMALL-LABEL:  storeTIInit:
; SMALL64:      ld [[OffsetR:[0-9]+]], [[TIInitL:L..C[0-9]+]](2)
; SMALL32:      lwz [[OffsetR:[0-9]+]], [[TIInitL:L..C[0-9]+]](2)
; SMALL64:      ld [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL:        bla .__tls_get_mod[PR]
; SMALL:        add [[TIInitAddrR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
; SMALL:        stw [[TIInitValR:[0-9]+]], 0([[TIInitAddrR]])
;
; LARGE-LABEL:  storeTIInit:
; LARGE:        addis [[OffsetHR:[0-9]+]], [[TIInitL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[OffsetR:[0-9]+]], [[TIInitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE32:      lwz [[OffsetR:[0-9]+]], [[TIInitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE:        addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE:        bla .__tls_get_mod[PR]
; LARGE:        add [[TIInitAddrR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
; LARGE:        stw [[TIInitValR:[0-9]+]], 0([[TIInitAddrR]])
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TIInit)
  store i32 %i, ptr %0, align 4
  ret void
}

define i32 @loadTIUninit() {
; SMALL-LABEL:  loadTIUninit:
; SMALL64:      ld [[OffsetR:[0-9]+]], [[TIUninitL:L..C[0-9]+]](2)
; SMALL32:      lwz [[OffsetR:[0-9]+]], [[TIUninitL:L..C[0-9]+]](2)
; SMALL64:      ld [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL:        bla .__tls_get_mod[PR]
; SMALL:        add [[TIUninitAddrR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
; SMALL:        lwz [[TIUninitValR:[0-9]+]], 0([[TIUninitAddrR]])
;
; LARGE-LABEL:  loadTIUninit:
; LARGE:        addis [[OffsetHR:[0-9]+]], [[TIUninitL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[OffsetR:[0-9]+]], [[TIUninitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE32:      lwz [[OffsetR:[0-9]+]], [[TIUninitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE:        addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE:        bla .__tls_get_mod[PR]
; LARGE:        add [[TIUninitAddrR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
; LARGE:        lwz [[TIUninitValR:[0-9]+]], 0([[TIUninitAddrR]])
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TIUninit)
  %1 = load i32, ptr %0, align 4
  ret i32 %1
}

define void @storeTIUninit(i32 noundef signext %i) {
; SMALL-LABEL:  storeTIUninit:
; SMALL64:      ld [[OffsetR:[0-9]+]], [[TIUninitL:L..C[0-9]+]](2)
; SMALL32:      lwz [[OffsetR:[0-9]+]], [[TIUninitL:L..C[0-9]+]](2)
; SMALL64:      ld [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL:        bla .__tls_get_mod[PR]
; SMALL:        add [[TIUninitAddrR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
; SMALL:        stw [[TIUninitValR:[0-9]+]], 0([[TIUninitAddrR]])
;
; LARGE-LABEL:  storeTIUninit:
; LARGE:        addis [[OffsetHR:[0-9]+]], [[TIUninitL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[OffsetR:[0-9]+]], [[TIUninitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE32:      lwz [[OffsetR:[0-9]+]], [[TIUninitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE:        addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE:        bla .__tls_get_mod[PR]
; LARGE:        add [[TIUninitAddrR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
; LARGE:        stw [[TIUninitValR:[0-9]+]], 0([[TIUninitAddrR]])
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TIUninit)
  store i32 %i, ptr %0, align 4
  ret void
}

define i32 @loadTWInit() {
; SMALL-LABEL:  loadTWInit:
; SMALL64:      ld [[OffsetR:[0-9]+]], [[TWInitL:L..C[0-9]+]](2)
; SMALL32:      lwz [[OffsetR:[0-9]+]], [[TWInitL:L..C[0-9]+]](2)
; SMALL64:      ld [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL:        bla .__tls_get_mod[PR]
; SMALL:        add [[TWInitAddrR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
; SMALL:        lwz [[TWInitValR:[0-9]+]], 0([[TWInitAddrR]])
;
; LARGE-LABEL:  loadTWInit:
; LARGE:        addis [[OffsetHR:[0-9]+]], [[TWInitL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[OffsetR:[0-9]+]], [[TWInitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE32:      lwz [[OffsetR:[0-9]+]], [[TWInitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE:        addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE:        bla .__tls_get_mod[PR]
; LARGE:        add [[TWInitAddrR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
; LARGE:        lwz [[TWInitValR:[0-9]+]], 0([[TWInitAddrR]])
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TWInit)
  %1 = load i32, ptr %0, align 4
  ret i32 %1
}

define void @storeTWInit(i32 noundef signext %i) {
; SMALL-LABEL:  storeTWInit:
; SMALL64:      ld [[OffsetR:[0-9]+]], [[TWInitL:L..C[0-9]+]](2)
; SMALL32:      lwz [[OffsetR:[0-9]+]], [[TWInitL:L..C[0-9]+]](2)
; SMALL64:      ld [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL:        bla .__tls_get_mod[PR]
; SMALL:        add [[TWInitAddrR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
; SMALL:        stw [[TWInitValR:[0-9]+]], 0([[TWInitAddrR]])
;
; LARGE-LABEL:  storeTWInit:
; LARGE:        addis [[OffsetHR:[0-9]+]], [[TWInitL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[OffsetR:[0-9]+]], [[TWInitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE32:      lwz [[OffsetR:[0-9]+]], [[TWInitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE:        addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE:        bla .__tls_get_mod[PR]
; LARGE:        add [[TWInitAddrR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
; LARGE:        stw [[TWInitValR:[0-9]+]], 0([[TWInitAddrR]])
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TWInit)
  store i32 %i, ptr %0, align 4
  ret void
}

define i32 @loadTWUninit() {
; SMALL-LABEL:  loadTWUninit:
; SMALL64:      ld [[OffsetR:[0-9]+]], [[TWUninitL:L..C[0-9]+]](2)
; SMALL32:      lwz [[OffsetR:[0-9]+]], [[TWUninitL:L..C[0-9]+]](2)
; SMALL64:      ld [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL:        bla .__tls_get_mod[PR]
; SMALL:        add [[TWUninitAddrR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
; SMALL:        lwz [[TWUninitValR:[0-9]+]], 0([[TWUninitAddrR]])
;
; LARGE-LABEL:  loadTWUninit:
; LARGE:        addis [[OffsetHR:[0-9]+]], [[TWUninitL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[OffsetR:[0-9]+]], [[TWUninitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE32:      lwz [[OffsetR:[0-9]+]], [[TWUninitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE:        addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE:        bla .__tls_get_mod[PR]
; LARGE:        add [[TWUninitAddrR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
; LARGE:        lwz [[TWUninitValR:[0-9]+]], 0([[TWUninitAddrR]])
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TWUninit)
  %1 = load i32, ptr %0, align 4
  ret i32 %1
}

define void @storeTWUninit(i32 noundef signext %i) {
; SMALL-LABEL:  storeTWUninit:
; SMALL64:      ld [[OffsetR:[0-9]+]], [[TWUninitL:L..C[0-9]+]](2)
; SMALL32:      lwz [[OffsetR:[0-9]+]], [[TWUninitL:L..C[0-9]+]](2)
; SMALL64:      ld [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL:        bla .__tls_get_mod[PR]
; SMALL:        add [[TWUninitAddrR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
; SMALL:        stw [[TWUninitValR:[0-9]+]], 0([[TWUninitAddrR]])
;
; LARGE-LABEL:  storeTWUninit:
; LARGE:        addis [[OffsetHR:[0-9]+]], [[TWUninitL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[OffsetR:[0-9]+]], [[TWUninitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE32:      lwz [[OffsetR:[0-9]+]], [[TWUninitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE:        addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE:        bla .__tls_get_mod[PR]
; LARGE:        add [[TWUninitAddrR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
; LARGE:        stw [[TWUninitValR:[0-9]+]], 0([[TWUninitAddrR]])
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TWUninit)
  store i32 %i, ptr %0, align 4
  ret void
}

; SMALL:          .extern .__tls_get_mod[PR]
; LARGE:          .extern .__tls_get_mod[PR]

; SMALL:        [[TGInitL]]:
; SMALL-NEXT:   .tc TGInit[TC],TGInit[TL]@ld
; SMALL:        [[ModuleHandleL]]:
; SMALL-NEXT:   .tc _Renamed..5f24__TLSML[TC],_Renamed..5f24__TLSML[TC]@ml
; SMALL-NEXT:   .rename _Renamed..5f24__TLSML[TC],"_$TLSML"
; SMALL:        [[TGUninitL]]:
; SMALL-NEXT:   .tc TGUninit[TC],TGUninit[TL]@ld
; SMALL:        [[TIInitL]]:
; SMALL-NEXT:   .tc TIInit[TC],TIInit[TL]@ld
; SMALL:        [[TIUninitL]]:
; SMALL-NEXT:   .tc TIUninit[TC],TIUninit[UL]@ld
; SMALL:        [[TWInitL]]:
; SMALL-NEXT:   .tc TWInit[TC],TWInit[TL]@ld
; SMALL:        [[TWUninitL]]:
; SMALL-NEXT:   .tc TWUninit[TC],TWUninit[TL]@ld

; LARGE:        [[TGInitL]]:
; LARGE-NEXT:   .tc TGInit[TE],TGInit[TL]@ld
; LARGE:        [[ModuleHandleL]]:
; LARGE-NEXT:   .tc _Renamed..5f24__TLSML[TC],_Renamed..5f24__TLSML[TC]@ml
; LARGE-NEXT:   .rename _Renamed..5f24__TLSML[TC],"_$TLSML"
; LARGE:        [[TGUninitL]]:
; LARGE-NEXT:   .tc TGUninit[TE],TGUninit[TL]@ld
; LARGE:        [[TIInitL]]:
; LARGE-NEXT:   .tc TIInit[TE],TIInit[TL]@ld
; LARGE:        [[TIUninitL]]:
; LARGE-NEXT:   .tc TIUninit[TE],TIUninit[UL]@ld
; LARGE:        [[TWInitL]]:
; LARGE-NEXT:   .tc TWInit[TE],TWInit[TL]@ld
; LARGE:        [[TWUninitL]]:
; LARGE-NEXT:   .tc TWUninit[TE],TWUninit[TL]@ld

declare nonnull ptr @llvm.threadlocal.address.p0(ptr nonnull)
