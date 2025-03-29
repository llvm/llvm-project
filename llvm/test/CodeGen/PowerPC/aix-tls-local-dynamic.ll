; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec -mtriple powerpc64-ibm-aix-xcoff \
; RUN:     --code-model=small < %s | FileCheck %s --check-prefixes=SMALL64,SMALL
; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec -mtriple powerpc64-ibm-aix-xcoff \
; RUN:     --code-model=large < %s | FileCheck %s --check-prefixes=LARGE64,LARGE
; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff \
; RUN:     --code-model=small < %s | FileCheck %s --check-prefixes=SMALL32,SMALL
; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff \
; RUN:     --code-model=large < %s | FileCheck %s --check-prefixes=LARGE32,LARGE
; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec -mtriple powerpc64-ibm-aix-xcoff \
; RUN:     --code-model=small -O0 < %s | FileCheck %s --check-prefixes=WITHDUP
; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec -mtriple powerpc64-ibm-aix-xcoff \
; RUN:     --code-model=small -O1 < %s | FileCheck %s --check-prefixes=NODUP

@TGInit = thread_local(localdynamic) global i32 42, align 4
@TGUninit = thread_local(localdynamic) global i32 0, align 4
@TIInit = internal thread_local(localdynamic) global i32 42, align 4
@TIUninit = internal thread_local(localdynamic) global i32 0, align 4
@TWInit = weak thread_local(localdynamic) global i32 42, align 4
@TWUninit = weak thread_local(localdynamic) global i32 0, align 4
@x = thread_local(localdynamic) global i32 42, align 4
@y = thread_local(localdynamic) global i32 42, align 4

define i32 @loadTGInit() {
; SMALL-LABEL:  loadTGInit:
; SMALL64:      ld [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL:        bla .__tls_get_mod[PR]
; SMALL64:      ld [[OffsetR:[0-9]+]], [[TGInitL:L..C[0-9]+]](2)
; SMALL32:      lwz [[OffsetR:[0-9]+]], [[TGInitL:L..C[0-9]+]](2)
; SMALL:        lwzx [[TGInitValR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
;
; LARGE-LABEL:  loadTGInit:
; LARGE64:      addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE:        addis [[OffsetHR:[0-9]+]], [[TGInitL:L..C[0-9]+]]@u(2)
; LARGE32:      addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE:        bla .__tls_get_mod[PR]
; LARGE64:      ld [[OffsetR:[0-9]+]], [[TGInitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE32:      lwz [[OffsetR:[0-9]+]], [[TGInitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE:        lwzx [[TGInitValR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TGInit)
  %1 = load i32, ptr %0, align 4
  ret i32 %1
}

define void @storeTGInit(i32 noundef signext %i) {
; SMALL-LABEL:  storeTGInit:
; SMALL64:      ld [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL:        bla .__tls_get_mod[PR]
; SMALL64:      ld [[OffsetR:[0-9]+]], [[TGInitL:L..C[0-9]+]](2)
; SMALL32:      lwz [[OffsetR:[0-9]+]], [[TGInitL:L..C[0-9]+]](2)
; SMALL:        stwx [[TGInitValR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
;
; LARGE-LABEL:  storeTGInit:
; LARGE64:      addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE:        addis [[OffsetHR:[0-9]+]], [[TGInitL:L..C[0-9]+]]@u(2)
; LARGE32:      addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE:        bla .__tls_get_mod[PR]
; LARGE64:      ld [[OffsetR:[0-9]+]], [[TGInitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE32:      lwz [[OffsetR:[0-9]+]], [[TGInitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE:        stwx [[TGInitValR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TGInit)
  store i32 %i, ptr %0, align 4
  ret void
}

define i32 @loadTGUninit() {
; SMALL-LABEL:  loadTGUninit:
; SMALL64:      ld [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL:        bla .__tls_get_mod[PR]
; SMALL64:      ld [[OffsetR:[0-9]+]], [[TGUninitL:L..C[0-9]+]](2)
; SMALL32:      lwz [[OffsetR:[0-9]+]], [[TGUninitL:L..C[0-9]+]](2)
; SMALL:        lwzx [[TGInitValR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
;
; LARGE-LABEL:  loadTGUninit:
; LARGE64:      addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE:        addis [[OffsetHR:[0-9]+]], [[TGUninitL:L..C[0-9]+]]@u(2)
; LARGE32:      addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE:        bla .__tls_get_mod[PR]
; LARGE64:      ld [[OffsetR:[0-9]+]], [[TGUninitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE32:      lwz [[OffsetR:[0-9]+]], [[TGUninitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE:        lwzx [[TGUninitValR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TGUninit)
  %1 = load i32, ptr %0, align 4
  ret i32 %1
}

define void @storeTGUninit(i32 noundef signext %i) {
; SMALL-LABEL:  storeTGUninit:
; SMALL64:      ld [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL:        bla .__tls_get_mod[PR]
; SMALL64:      ld [[OffsetR:[0-9]+]], [[TGUninitL:L..C[0-9]+]](2)
; SMALL32:      lwz [[OffsetR:[0-9]+]], [[TGUninitL:L..C[0-9]+]](2)
; SMALL:        stwx [[TGUninitValR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
;
; LARGE-LABEL:  storeTGUninit:
; LARGE64:      addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE:        addis [[OffsetHR:[0-9]+]], [[TGUninitL:L..C[0-9]+]]@u(2)
; LARGE32:      addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE:        bla .__tls_get_mod[PR]
; LARGE64:      ld [[OffsetR:[0-9]+]], [[TGUninitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE32:      lwz [[OffsetR:[0-9]+]], [[TGUninitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE:        stwx [[TGUninitValR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TGUninit)
  store i32 %i, ptr %0, align 4
  ret void
}

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

define i32 @loadTWInit() {
; SMALL-LABEL:  loadTWInit:
; SMALL64:      ld [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL:        bla .__tls_get_mod[PR]
; SMALL64:      ld [[OffsetR:[0-9]+]], [[TWInitL:L..C[0-9]+]](2)
; SMALL32:      lwz [[OffsetR:[0-9]+]], [[TWInitL:L..C[0-9]+]](2)
; SMALL:        lwzx [[TWInitValR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
;
; LARGE-LABEL:  loadTWInit:
; LARGE64:      addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE:        addis [[OffsetHR:[0-9]+]], [[TWInitL:L..C[0-9]+]]@u(2)
; LARGE32:      addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE:        bla .__tls_get_mod[PR]
; LARGE64:      ld [[OffsetR:[0-9]+]], [[TWInitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE32:      lwz [[OffsetR:[0-9]+]], [[TWInitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE:        lwzx [[TWInitValR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TWInit)
  %1 = load i32, ptr %0, align 4
  ret i32 %1
}

define void @storeTWInit(i32 noundef signext %i) {
; SMALL-LABEL:  storeTWInit:
; SMALL64:      ld [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL:        bla .__tls_get_mod[PR]
; SMALL64:      ld [[OffsetR:[0-9]+]], [[TWInitL:L..C[0-9]+]](2)
; SMALL32:      lwz [[OffsetR:[0-9]+]], [[TWInitL:L..C[0-9]+]](2)
; SMALL:        stwx [[TWInitValR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
;
; LARGE-LABEL:  storeTWInit:
; LARGE64:      addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE:        addis [[OffsetHR:[0-9]+]], [[TWInitL:L..C[0-9]+]]@u(2)
; LARGE32:      addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE:        bla .__tls_get_mod[PR]
; LARGE64:      ld [[OffsetR:[0-9]+]], [[TWInitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE32:      lwz [[OffsetR:[0-9]+]], [[TWInitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE:        stwx [[TWInitValR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TWInit)
  store i32 %i, ptr %0, align 4
  ret void
}

define i32 @loadTWUninit() {
; SMALL-LABEL:  loadTWUninit:
; SMALL64:      ld [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL:        bla .__tls_get_mod[PR]
; SMALL64:      ld [[OffsetR:[0-9]+]], [[TWUninitL:L..C[0-9]+]](2)
; SMALL32:      lwz [[OffsetR:[0-9]+]], [[TWUninitL:L..C[0-9]+]](2)
; SMALL:        lwzx [[TWUninitValR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
;
; LARGE-LABEL:  loadTWUninit:
; LARGE64:      addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE:        addis [[OffsetHR:[0-9]+]], [[TWUninitL:L..C[0-9]+]]@u(2)
; LARGE32:      addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE:        bla .__tls_get_mod[PR]
; LARGE64:      ld [[OffsetR:[0-9]+]], [[TWUninitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE32:      lwz [[OffsetR:[0-9]+]], [[TWUninitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE:        lwzx [[TWUninitValR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TWUninit)
  %1 = load i32, ptr %0, align 4
  ret i32 %1
}

define void @storeTWUninit(i32 noundef signext %i) {
; SMALL-LABEL:  storeTWUninit:
; SMALL64:      ld [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL:        bla .__tls_get_mod[PR]
; SMALL64:      ld [[OffsetR:[0-9]+]], [[TWUninitL:L..C[0-9]+]](2)
; SMALL32:      lwz [[OffsetR:[0-9]+]], [[TWUninitL:L..C[0-9]+]](2)
; SMALL:        stwx [[TWUninitValR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
;
; LARGE-LABEL:  storeTWUninit:
; LARGE64:      addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE:        addis [[OffsetHR:[0-9]+]], [[TWUninitL:L..C[0-9]+]]@u(2)
; LARGE32:      addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE:        bla .__tls_get_mod[PR]
; LARGE64:      ld [[OffsetR:[0-9]+]], [[TWUninitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE32:      lwz [[OffsetR:[0-9]+]], [[TWUninitL:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE:        stwx [[TWUninitValR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TWUninit)
  store i32 %i, ptr %0, align 4
  ret void
}

define i32 @DedupTlsGetMod() #0 {
; WITHDUP-LABEL:  DedupTlsGetMod:
; WITHDUP:        ld [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; WITHDUP-NEXT:   bla .__tls_get_mod[PR]
; WITHDUP-NEXT:   ld [[OffsetXR:[0-9]+]], [[X:L..C[0-9]+]](2)
; WITHDUP:        ld [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; WITHDUP-NEXT:   bla .__tls_get_mod[PR]
; WITHDUP:        ld [[OffsetYR:[0-9]+]], [[Y:L..C[0-9]+]](2)
; WITHDUP-LABEL:  L..DedupTlsGetMod0:
;
; NODUP-LABEL:  DedupTlsGetMod:
; NODUP:        ld [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; NODUP-NEXT:   bla .__tls_get_mod[PR]
; NODUP-NEXT:   ld [[OffsetXR:[0-9]+]], [[X:L..C[0-9]+]](2)
; NODUP-NEXT:   ld [[OffsetYR:[0-9]+]], [[Y:L..C[0-9]+]](2)
; NODUP-NEXT:   lwzx [[XValR:[0-9]+]], [[ModuleHandleR]], [[OffsetXR]]
; NODUP-NEXT:   lwzx [[YValR:[0-9]+]], [[ModuleHandleR]], [[OffsetYR]]
; NODUP-LABEL:  L..DedupTlsGetMod0:
entry:
  %retval = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  %0 = call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @x)
  %1 = load i32, ptr %0, align 4
  %2 = call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @y)
  %3 = load i32, ptr %2, align 4
  %add = add nsw i32 %1, %3
  ret i32 %add
}

; SMALL:          .extern .__tls_get_mod[PR]
; LARGE:          .extern .__tls_get_mod[PR]
; SMALL-NOT:      .extern _Renamed..5f24__TLSML[TC]
; LARGE-NOT:      .extern _Renamed..5f24__TLSML[TC]

; SMALL:        [[ModuleHandleL]]:
; SMALL-NEXT:   .tc _Renamed..5f24__TLSML[TC],_Renamed..5f24__TLSML[TC]@ml
; SMALL-NEXT:   .rename _Renamed..5f24__TLSML[TC],"_$TLSML"
; SMALL:        [[TGInitL]]:
; SMALL-NEXT:   .tc TGInit[TC],TGInit[TL]@ld
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

; LARGE64:        [[ModuleHandleL]]:
; LARGE64-NEXT:   .tc _Renamed..5f24__TLSML[TC],_Renamed..5f24__TLSML[TC]@ml
; LARGE64-NEXT:   .rename _Renamed..5f24__TLSML[TC],"_$TLSML"
; LARGE64:        [[TGInitL]]:
; LARGE64-NEXT:   .tc TGInit[TE],TGInit[TL]@ld
;
; LARGE32:        [[TGInitL]]:
; LARGE32-NEXT:   .tc TGInit[TE],TGInit[TL]@ld
; LARGE32:        [[ModuleHandleL]]:
; LARGE32-NEXT:   .tc _Renamed..5f24__TLSML[TC],_Renamed..5f24__TLSML[TC]@ml
; LARGE32-NEXT:   .rename _Renamed..5f24__TLSML[TC],"_$TLSML"
;
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
