; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec -mtriple powerpc64-ibm-aix-xcoff \
; RUN:     --code-model=small < %s | FileCheck %s --check-prefixes=SMALL64,SMALL
; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec -mtriple powerpc64-ibm-aix-xcoff \
; RUN:     --code-model=large < %s | FileCheck %s --check-prefixes=LARGE64,LARGE
; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff \
; RUN:     --code-model=small < %s | FileCheck %s --check-prefixes=SMALL32,SMALL
; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff \
; RUN:     --code-model=large < %s | FileCheck %s --check-prefixes=LARGE32,LARGE

@TIInitIE = internal thread_local(initialexec) global i32 42, align 4
@TIUninitIE = internal thread_local(initialexec) global i32 0, align 4
@TIInitLD = internal thread_local(localdynamic) global i32 42, align 4
@TIUninitLD = internal thread_local(localdynamic) global i32 0, align 4

define i32 @loadTIInitIE_USE_TLS_LD() #0 {
; SMALL-LABEL:  loadTIInitIE_USE_TLS_LD:
; SMALL64:      ld [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL:        bla .__tls_get_mod[PR]
; SMALL64:      ld [[OffsetR:[0-9]+]], [[TIInitIE_USE_LD_L:L..C[0-9]+]](2)
; SMALL32:      lwz [[OffsetR:[0-9]+]], [[TIInitIE_USE_LD_L:L..C[0-9]+]](2)
; SMALL:        lwzx [[TIInitValR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
;
; LARGE-LABEL:  loadTIInitIE_USE_TLS_LD:
; LARGE64:      addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE:        addis [[OffsetHR:[0-9]+]], [[TIInitIE_USE_LD_L:L..C[0-9]+]]@u(2)
; LARGE32:      addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE:        bla .__tls_get_mod[PR]
; LARGE64:      ld [[OffsetR:[0-9]+]], [[TIInitIE_USE_LD_L:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE32:      lwz [[OffsetR:[0-9]+]], [[TIInitIE_USE_LD_L:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE:        lwzx [[TIInitValR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TIInitIE)
  %1 = load i32, ptr %0, align 4
  ret i32 %1
}

define void @storeTIInitIE_USE_TLS_LD(i32 noundef signext %i) #0 {
; SMALL-LABEL:  storeTIInitIE_USE_TLS_LD:
; SMALL64:      ld [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL:        bla .__tls_get_mod[PR]
; SMALL64:      ld [[OffsetR:[0-9]+]], [[TIInitIE_USE_LD_L:L..C[0-9]+]](2)
; SMALL32:      lwz [[OffsetR:[0-9]+]], [[TIInitIE_USE_LD_L:L..C[0-9]+]](2)
; SMALL:        stwx [[TIInitValR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
;
; LARGE-LABEL:  storeTIInitIE_USE_TLS_LD:
; LARGE64:      addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE:        addis [[OffsetHR:[0-9]+]], [[TIInitIE_USE_LD_L:L..C[0-9]+]]@u(2)
; LARGE32:      addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE:        bla .__tls_get_mod[PR]
; LARGE64:      ld [[OffsetR:[0-9]+]], [[TIInitIE_USE_LD_L:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE32:      lwz [[OffsetR:[0-9]+]], [[TIInitIE_USE_LD_L:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE:        stwx [[TIInitValR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TIInitIE)
  store i32 %i, ptr %0, align 4
  ret void
}

define i32 @loadTIUninitIE_USE_TLS_LD() #0 {
; SMALL-LABEL:  loadTIUninitIE_USE_TLS_LD:
; SMALL64:      ld [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL:        bla .__tls_get_mod[PR]
; SMALL64:      ld [[OffsetR:[0-9]+]], [[TIUninitIE_USE_LD_L:L..C[0-9]+]](2)
; SMALL32:      lwz [[OffsetR:[0-9]+]], [[TIUninitIE_USE_LD_L:L..C[0-9]+]](2)
; SMALL:        lwzx [[TIUninitValR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
;
; LARGE-LABEL:  loadTIUninitIE_USE_TLS_LD:
; LARGE64:      addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE:        addis [[OffsetHR:[0-9]+]], [[TIUninitIE_USE_LD_L:L..C[0-9]+]]@u(2)
; LARGE32:      addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE:        bla .__tls_get_mod[PR]
; LARGE64:      ld [[OffsetR:[0-9]+]], [[TIUninitIE_USE_LD_L:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE32:      lwz [[OffsetR:[0-9]+]], [[TIUninitIE_USE_LD_L:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE:        lwzx [[TIUninitValR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TIUninitIE)
  %1 = load i32, ptr %0, align 4
  ret i32 %1
}

define void @storeTIUninitIE_USE_TLS_LD(i32 noundef signext %i) #0 {
; SMALL-LABEL:  storeTIUninitIE_USE_TLS_LD:
; SMALL64:      ld [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL:        bla .__tls_get_mod[PR]
; SMALL64:      ld [[OffsetR:[0-9]+]], [[TIUninitIE_USE_LD_L:L..C[0-9]+]](2)
; SMALL32:      lwz [[OffsetR:[0-9]+]], [[TIUninitIE_USE_LD_L:L..C[0-9]+]](2)
; SMALL:        stwx [[TIUninitValR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
;
; LARGE-LABEL:  storeTIUninitIE_USE_TLS_LD:
; LARGE64:      addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE:        addis [[OffsetHR:[0-9]+]], [[TIUninitIE_USE_LD_L:L..C[0-9]+]]@u(2)
; LARGE32:      addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE:        bla .__tls_get_mod[PR]
; LARGE64:      ld [[OffsetR:[0-9]+]], [[TIUninitIE_USE_LD_L:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE32:      lwz [[OffsetR:[0-9]+]], [[TIUninitIE_USE_LD_L:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE:        stwx [[TIUninitValR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TIUninitIE)
  store i32 %i, ptr %0, align 4
  ret void
}

define i32 @loadTIInitIE() {
; SMALL64-LABEL: loadTIInitIE:
; SMALL64:       # %bb.0: # %entry
; SMALL64-NEXT:    ld 3, L..C3(2) # target-flags(ppc-tprel) @TIInitIE
; SMALL64-NEXT:    lwzx 3, 13, 3
; SMALL64-NEXT:    blr
;
; LARGE64-LABEL: loadTIInitIE:
; LARGE64:       # %bb.0: # %entry
; LARGE64-NEXT:    addis 3, L..C3@u(2)
; LARGE64-NEXT:    ld 3, L..C3@l(3)
; LARGE64-NEXT:    lwzx 3, 13, 3
; LARGE64-NEXT:    blr
;
; SMALL32-LABEL: loadTIInitIE:
; SMALL32:       # %bb.0: # %entry
; SMALL32-NEXT:    mflr 0
; SMALL32-NEXT:    stwu 1, -32(1)
; SMALL32-NEXT:    lwz 4, L..C3(2) # target-flags(ppc-tprel) @TIInitIE
; SMALL32-NEXT:    stw 0, 40(1)
; SMALL32-NEXT:    bla .__get_tpointer[PR]
; SMALL32-NEXT:    lwzx 3, 3, 4
; SMALL32-NEXT:    addi 1, 1, 32
; SMALL32-NEXT:    lwz 0, 8(1)
; SMALL32-NEXT:    mtlr 0
; SMALL32-NEXT:    blr
;
; LARGE32-LABEL: loadTIInitIE:
; LARGE32:       # %bb.0: # %entry
; LARGE32-NEXT:    mflr 0
; LARGE32-NEXT:    stwu 1, -32(1)
; LARGE32-NEXT:    stw 0, 40(1)
; LARGE32-NEXT:    addis 3, L..C3@u(2)
; LARGE32-NEXT:    lwz 4, L..C3@l(3)
; LARGE32-NEXT:    bla .__get_tpointer[PR]
; LARGE32-NEXT:    lwzx 3, 3, 4
; LARGE32-NEXT:    addi 1, 1, 32
; LARGE32-NEXT:    lwz 0, 8(1)
; LARGE32-NEXT:    mtlr 0
; LARGE32-NEXT:    blr
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TIInitIE)
  %1 = load i32, ptr %0, align 4
  ret i32 %1
}

define void @storeTIInitIE(i32 noundef signext %i) {
; SMALL64-LABEL: storeTIInitIE:
; SMALL64:       # %bb.0: # %entry
; SMALL64-NEXT:    ld 4, L..C3(2) # target-flags(ppc-tprel) @TIInitIE
; SMALL64-NEXT:    stwx 3, 13, 4
; SMALL64-NEXT:    blr
;
; LARGE64-LABEL: storeTIInitIE:
; LARGE64:       # %bb.0: # %entry
; LARGE64-NEXT:    addis 4, L..C3@u(2)
; LARGE64-NEXT:    ld 4, L..C3@l(4)
; LARGE64-NEXT:    stwx 3, 13, 4
; LARGE64-NEXT:    blr
;
; SMALL32-LABEL: storeTIInitIE:
; SMALL32:       # %bb.0: # %entry
; SMALL32-NEXT:    mflr 0
; SMALL32-NEXT:    stwu 1, -32(1)
; SMALL32-NEXT:    lwz 5, L..C3(2) # target-flags(ppc-tprel) @TIInitIE
; SMALL32-NEXT:    mr 4, 3
; SMALL32-NEXT:    bla .__get_tpointer[PR]
; SMALL32-NEXT:    stw 0, 40(1)
; SMALL32-NEXT:    stwx 4, 3, 5
; SMALL32-NEXT:    addi 1, 1, 32
; SMALL32-NEXT:    lwz 0, 8(1)
; SMALL32-NEXT:    mtlr 0
; SMALL32-NEXT:    blr
;
; LARGE32-LABEL: storeTIInitIE:
; LARGE32:       # %bb.0: # %entry
; LARGE32-NEXT:    mflr 0
; LARGE32-NEXT:    stwu 1, -32(1)
; LARGE32-NEXT:    stw 0, 40(1)
; LARGE32-NEXT:    mr 4, 3
; LARGE32-NEXT:    addis 3, L..C3@u(2)
; LARGE32-NEXT:    lwz 5, L..C3@l(3)
; LARGE32-NEXT:    bla .__get_tpointer[PR]
; LARGE32-NEXT:    stwx 4, 3, 5
; LARGE32-NEXT:    addi 1, 1, 32
; LARGE32-NEXT:    lwz 0, 8(1)
; LARGE32-NEXT:    mtlr 0
; LARGE32-NEXT:    blr
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TIInitIE)
  store i32 %i, ptr %0, align 4
  ret void
}

define i32 @loadTIUninitIE() {
; SMALL64-LABEL: loadTIUninitIE:
; SMALL64:       # %bb.0: # %entry
; SMALL64-NEXT:    ld 3, L..C4(2) # target-flags(ppc-tprel) @TIUninitIE
; SMALL64-NEXT:    lwzx 3, 13, 3
; SMALL64-NEXT:    blr
;
; LARGE64-LABEL: loadTIUninitIE:
; LARGE64:       # %bb.0: # %entry
; LARGE64-NEXT:    addis 3, L..C4@u(2)
; LARGE64-NEXT:    ld 3, L..C4@l(3)
; LARGE64-NEXT:    lwzx 3, 13, 3
; LARGE64-NEXT:    blr
;
; SMALL32-LABEL: loadTIUninitIE:
; SMALL32:       # %bb.0: # %entry
; SMALL32-NEXT:    mflr 0
; SMALL32-NEXT:    stwu 1, -32(1)
; SMALL32-NEXT:    lwz 4, L..C4(2) # target-flags(ppc-tprel) @TIUninitIE
; SMALL32-NEXT:    stw 0, 40(1)
; SMALL32-NEXT:    bla .__get_tpointer[PR]
; SMALL32-NEXT:    lwzx 3, 3, 4
; SMALL32-NEXT:    addi 1, 1, 32
; SMALL32-NEXT:    lwz 0, 8(1)
; SMALL32-NEXT:    mtlr 0
; SMALL32-NEXT:    blr
;
; LARGE32-LABEL: loadTIUninitIE:
; LARGE32:       # %bb.0: # %entry
; LARGE32-NEXT:    mflr 0
; LARGE32-NEXT:    stwu 1, -32(1)
; LARGE32-NEXT:    stw 0, 40(1)
; LARGE32-NEXT:    addis 3, L..C4@u(2)
; LARGE32-NEXT:    lwz 4, L..C4@l(3)
; LARGE32-NEXT:    bla .__get_tpointer[PR]
; LARGE32-NEXT:    lwzx 3, 3, 4
; LARGE32-NEXT:    addi 1, 1, 32
; LARGE32-NEXT:    lwz 0, 8(1)
; LARGE32-NEXT:    mtlr 0
; LARGE32-NEXT:    blr
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TIUninitIE)
  %1 = load i32, ptr %0, align 4
  ret i32 %1
}

define void @storeTIUninitIE(i32 noundef signext %i) {
; SMALL64-LABEL: storeTIUninitIE:
; SMALL64:       # %bb.0: # %entry
; SMALL64-NEXT:    ld 4, L..C4(2) # target-flags(ppc-tprel) @TIUninitIE
; SMALL64-NEXT:    stwx 3, 13, 4
; SMALL64-NEXT:    blr
;
; LARGE64-LABEL: storeTIUninitIE:
; LARGE64:       # %bb.0: # %entry
; LARGE64-NEXT:    addis 4, L..C4@u(2)
; LARGE64-NEXT:    ld 4, L..C4@l(4)
; LARGE64-NEXT:    stwx 3, 13, 4
; LARGE64-NEXT:    blr
;
; SMALL32-LABEL: storeTIUninitIE:
; SMALL32:       # %bb.0: # %entry
; SMALL32-NEXT:    mflr 0
; SMALL32-NEXT:    stwu 1, -32(1)
; SMALL32-NEXT:    lwz 5, L..C4(2) # target-flags(ppc-tprel) @TIUninitIE
; SMALL32-NEXT:    mr 4, 3
; SMALL32-NEXT:    bla .__get_tpointer[PR]
; SMALL32-NEXT:    stw 0, 40(1)
; SMALL32-NEXT:    stwx 4, 3, 5
; SMALL32-NEXT:    addi 1, 1, 32
; SMALL32-NEXT:    lwz 0, 8(1)
; SMALL32-NEXT:    mtlr 0
; SMALL32-NEXT:    blr
;
; LARGE32-LABEL: storeTIUninitIE:
; LARGE32:       # %bb.0: # %entry
; LARGE32-NEXT:    mflr 0
; LARGE32-NEXT:    stwu 1, -32(1)
; LARGE32-NEXT:    stw 0, 40(1)
; LARGE32-NEXT:    mr 4, 3
; LARGE32-NEXT:    addis 3, L..C4@u(2)
; LARGE32-NEXT:    lwz 5, L..C4@l(3)
; LARGE32-NEXT:    bla .__get_tpointer[PR]
; LARGE32-NEXT:    stwx 4, 3, 5
; LARGE32-NEXT:    addi 1, 1, 32
; LARGE32-NEXT:    lwz 0, 8(1)
; LARGE32-NEXT:    mtlr 0
; LARGE32-NEXT:    blr
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TIUninitIE)
  store i32 %i, ptr %0, align 4
  ret void
}

define i32 @loadTIInitLD_USE_TLS_IE() #1 {
; SMALL64-LABEL: loadTIInitLD_USE_TLS_IE:
; SMALL64:       # %bb.0: # %entry
; SMALL64-NEXT:    ld 3, L..C5(2) # target-flags(ppc-tprel) @TIInitLD
; SMALL64-NEXT:    lwzx 3, 13, 3
; SMALL64-NEXT:    blr
;
; LARGE64-LABEL: loadTIInitLD_USE_TLS_IE:
; LARGE64:       # %bb.0: # %entry
; LARGE64-NEXT:    addis 3, L..C5@u(2)
; LARGE64-NEXT:    ld 3, L..C5@l(3)
; LARGE64-NEXT:    lwzx 3, 13, 3
; LARGE64-NEXT:    blr
;
; SMALL32-LABEL: loadTIInitLD_USE_TLS_IE:
; SMALL32:       # %bb.0: # %entry
; SMALL32-NEXT:    mflr 0
; SMALL32-NEXT:    stwu 1, -32(1)
; SMALL32-NEXT:    lwz 4, L..C5(2) # target-flags(ppc-tprel) @TIInitLD
; SMALL32-NEXT:    stw 0, 40(1)
; SMALL32-NEXT:    bla .__get_tpointer[PR]
; SMALL32-NEXT:    lwzx 3, 3, 4
; SMALL32-NEXT:    addi 1, 1, 32
; SMALL32-NEXT:    lwz 0, 8(1)
; SMALL32-NEXT:    mtlr 0
; SMALL32-NEXT:    blr
;
; LARGE32-LABEL: loadTIInitLD_USE_TLS_IE:
; LARGE32:       # %bb.0: # %entry
; LARGE32-NEXT:    mflr 0
; LARGE32-NEXT:    stwu 1, -32(1)
; LARGE32-NEXT:    stw 0, 40(1)
; LARGE32-NEXT:    addis 3, L..C5@u(2)
; LARGE32-NEXT:    lwz 4, L..C5@l(3)
; LARGE32-NEXT:    bla .__get_tpointer[PR]
; LARGE32-NEXT:    lwzx 3, 3, 4
; LARGE32-NEXT:    addi 1, 1, 32
; LARGE32-NEXT:    lwz 0, 8(1)
; LARGE32-NEXT:    mtlr 0
; LARGE32-NEXT:    blr
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TIInitLD)
  %1 = load i32, ptr %0, align 4
  ret i32 %1
}

define void @storeTIInitLD_USE_TLS_IE(i32 noundef signext %i) #1 {
; SMALL64-LABEL: storeTIInitLD_USE_TLS_IE:
; SMALL64:       # %bb.0: # %entry
; SMALL64-NEXT:    ld 4, L..C5(2) # target-flags(ppc-tprel) @TIInitLD
; SMALL64-NEXT:    stwx 3, 13, 4
; SMALL64-NEXT:    blr
;
; LARGE64-LABEL: storeTIInitLD_USE_TLS_IE:
; LARGE64:       # %bb.0: # %entry
; LARGE64-NEXT:    addis 4, L..C5@u(2)
; LARGE64-NEXT:    ld 4, L..C5@l(4)
; LARGE64-NEXT:    stwx 3, 13, 4
; LARGE64-NEXT:    blr
;
; SMALL32-LABEL: storeTIInitLD_USE_TLS_IE:
; SMALL32:       # %bb.0: # %entry
; SMALL32-NEXT:    mflr 0
; SMALL32-NEXT:    stwu 1, -32(1)
; SMALL32-NEXT:    lwz 5, L..C5(2) # target-flags(ppc-tprel) @TIInitLD
; SMALL32-NEXT:    mr 4, 3
; SMALL32-NEXT:    bla .__get_tpointer[PR]
; SMALL32-NEXT:    stw 0, 40(1)
; SMALL32-NEXT:    stwx 4, 3, 5
; SMALL32-NEXT:    addi 1, 1, 32
; SMALL32-NEXT:    lwz 0, 8(1)
; SMALL32-NEXT:    mtlr 0
; SMALL32-NEXT:    blr
;
; LARGE32-LABEL: storeTIInitLD_USE_TLS_IE:
; LARGE32:       # %bb.0: # %entry
; LARGE32-NEXT:    mflr 0
; LARGE32-NEXT:    stwu 1, -32(1)
; LARGE32-NEXT:    stw 0, 40(1)
; LARGE32-NEXT:    mr 4, 3
; LARGE32-NEXT:    addis 3, L..C5@u(2)
; LARGE32-NEXT:    lwz 5, L..C5@l(3)
; LARGE32-NEXT:    bla .__get_tpointer[PR]
; LARGE32-NEXT:    stwx 4, 3, 5
; LARGE32-NEXT:    addi 1, 1, 32
; LARGE32-NEXT:    lwz 0, 8(1)
; LARGE32-NEXT:    mtlr 0
; LARGE32-NEXT:    blr
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TIInitLD)
  store i32 %i, ptr %0, align 4
  ret void
}

define i32 @loadTIUninitLD_USE_TLS_IE() #1 {
; SMALL64-LABEL: loadTIUninitLD_USE_TLS_IE:
; SMALL64:       # %bb.0: # %entry
; SMALL64-NEXT:    ld 3, L..C6(2) # target-flags(ppc-tprel) @TIUninitLD
; SMALL64-NEXT:    lwzx 3, 13, 3
; SMALL64-NEXT:    blr
;
; LARGE64-LABEL: loadTIUninitLD_USE_TLS_IE:
; LARGE64:       # %bb.0: # %entry
; LARGE64-NEXT:    addis 3, L..C6@u(2)
; LARGE64-NEXT:    ld 3, L..C6@l(3)
; LARGE64-NEXT:    lwzx 3, 13, 3
; LARGE64-NEXT:    blr
;
; SMALL32-LABEL: loadTIUninitLD_USE_TLS_IE:
; SMALL32:       # %bb.0: # %entry
; SMALL32-NEXT:    mflr 0
; SMALL32-NEXT:    stwu 1, -32(1)
; SMALL32-NEXT:    lwz 4, L..C6(2) # target-flags(ppc-tprel) @TIUninitLD
; SMALL32-NEXT:    stw 0, 40(1)
; SMALL32-NEXT:    bla .__get_tpointer[PR]
; SMALL32-NEXT:    lwzx 3, 3, 4
; SMALL32-NEXT:    addi 1, 1, 32
; SMALL32-NEXT:    lwz 0, 8(1)
; SMALL32-NEXT:    mtlr 0
; SMALL32-NEXT:    blr
;
; LARGE32-LABEL: loadTIUninitLD_USE_TLS_IE:
; LARGE32:       # %bb.0: # %entry
; LARGE32-NEXT:    mflr 0
; LARGE32-NEXT:    stwu 1, -32(1)
; LARGE32-NEXT:    stw 0, 40(1)
; LARGE32-NEXT:    addis 3, L..C6@u(2)
; LARGE32-NEXT:    lwz 4, L..C6@l(3)
; LARGE32-NEXT:    bla .__get_tpointer[PR]
; LARGE32-NEXT:    lwzx 3, 3, 4
; LARGE32-NEXT:    addi 1, 1, 32
; LARGE32-NEXT:    lwz 0, 8(1)
; LARGE32-NEXT:    mtlr 0
; LARGE32-NEXT:    blr
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TIUninitLD)
  %1 = load i32, ptr %0, align 4
  ret i32 %1
}

define void @storeTIUninitLD_USE_TLS_IE(i32 noundef signext %i) #1 {
; SMALL64-LABEL: storeTIUninitLD_USE_TLS_IE:
; SMALL64:       # %bb.0: # %entry
; SMALL64-NEXT:    ld 4, L..C6(2) # target-flags(ppc-tprel) @TIUninitLD
; SMALL64-NEXT:    stwx 3, 13, 4
; SMALL64-NEXT:    blr
;
; LARGE64-LABEL: storeTIUninitLD_USE_TLS_IE:
; LARGE64:       # %bb.0: # %entry
; LARGE64-NEXT:    addis 4, L..C6@u(2)
; LARGE64-NEXT:    ld 4, L..C6@l(4)
; LARGE64-NEXT:    stwx 3, 13, 4
; LARGE64-NEXT:    blr
;
; SMALL32-LABEL: storeTIUninitLD_USE_TLS_IE:
; SMALL32:       # %bb.0: # %entry
; SMALL32-NEXT:    mflr 0
; SMALL32-NEXT:    stwu 1, -32(1)
; SMALL32-NEXT:    lwz 5, L..C6(2) # target-flags(ppc-tprel) @TIUninitLD
; SMALL32-NEXT:    mr 4, 3
; SMALL32-NEXT:    bla .__get_tpointer[PR]
; SMALL32-NEXT:    stw 0, 40(1)
; SMALL32-NEXT:    stwx 4, 3, 5
; SMALL32-NEXT:    addi 1, 1, 32
; SMALL32-NEXT:    lwz 0, 8(1)
; SMALL32-NEXT:    mtlr 0
; SMALL32-NEXT:    blr
;
; LARGE32-LABEL: storeTIUninitLD_USE_TLS_IE:
; LARGE32:       # %bb.0: # %entry
; LARGE32-NEXT:    mflr 0
; LARGE32-NEXT:    stwu 1, -32(1)
; LARGE32-NEXT:    stw 0, 40(1)
; LARGE32-NEXT:    mr 4, 3
; LARGE32-NEXT:    addis 3, L..C6@u(2)
; LARGE32-NEXT:    lwz 5, L..C6@l(3)
; LARGE32-NEXT:    bla .__get_tpointer[PR]
; LARGE32-NEXT:    stwx 4, 3, 5
; LARGE32-NEXT:    addi 1, 1, 32
; LARGE32-NEXT:    lwz 0, 8(1)
; LARGE32-NEXT:    mtlr 0
; LARGE32-NEXT:    blr
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TIUninitLD)
  store i32 %i, ptr %0, align 4
  ret void
}

define i32 @loadTIInitLD() {
; SMALL-LABEL:  loadTIInitLD:
; SMALL64:      ld [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL:        bla .__tls_get_mod[PR]
; SMALL64:      ld [[OffsetR:[0-9]+]], [[TIInitLD_USE_IE_L:L..C[0-9]+]](2)
; SMALL32:      lwz [[OffsetR:[0-9]+]], [[TIInitLD_USE_IE_L:L..C[0-9]+]](2)
; SMALL:        lwzx [[TIInitValR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
;
; LARGE-LABEL:  loadTIInitLD:
; LARGE64:      addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE:        addis [[OffsetHR:[0-9]+]], [[TIInitLD_USE_IE_L:L..C[0-9]+]]@u(2)
; LARGE32:      addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE:        bla .__tls_get_mod[PR]
; LARGE64:      ld [[OffsetR:[0-9]+]], [[TIInitLD_USE_IE_L:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE32:      lwz [[OffsetR:[0-9]+]], [[TIInitLD_USE_IE_L:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE:        lwzx [[TIInitValR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TIInitLD)
  %1 = load i32, ptr %0, align 4
  ret i32 %1
}

define void @storeTIInitLD(i32 noundef signext %i) {
; SMALL-LABEL:  storeTIInitLD:
; SMALL64:      ld [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL:        bla .__tls_get_mod[PR]
; SMALL64:      ld [[OffsetR:[0-9]+]], [[TIInitLD_USE_IE_L:L..C[0-9]+]](2)
; SMALL32:      lwz [[OffsetR:[0-9]+]], [[TIInitLD_USE_IE_L:L..C[0-9]+]](2)
; SMALL:        stwx [[TIInitValR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
;
; LARGE-LABEL:  storeTIInitLD:
; LARGE64:      addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE:        addis [[OffsetHR:[0-9]+]], [[TIInitLD_USE_IE_L:L..C[0-9]+]]@u(2)
; LARGE32:      addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE:        bla .__tls_get_mod[PR]
; LARGE64:      ld [[OffsetR:[0-9]+]], [[TIInitLD_USE_IE_L:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE32:      lwz [[OffsetR:[0-9]+]], [[TIInitLD_USE_IE_L:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE:        stwx [[TIInitValR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TIInitLD)
  store i32 %i, ptr %0, align 4
  ret void
}

define i32 @loadTIUninitLD() {
; SMALL-LABEL:  loadTIUninitLD:
; SMALL64:      ld [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL:        bla .__tls_get_mod[PR]
; SMALL64:      ld [[OffsetR:[0-9]+]], [[TIUninitLD_USE_IE_L:L..C[0-9]+]](2)
; SMALL32:      lwz [[OffsetR:[0-9]+]], [[TIUninitLD_USE_IE_L:L..C[0-9]+]](2)
; SMALL:        lwzx [[TIUninitValR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
;
; LARGE-LABEL:  loadTIUninitLD:
; LARGE64:      addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE:        addis [[OffsetHR:[0-9]+]], [[TIUninitLD_USE_IE_L:L..C[0-9]+]]@u(2)
; LARGE32:      addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE:        bla .__tls_get_mod[PR]
; LARGE64:      ld [[OffsetR:[0-9]+]], [[TIUninitLD_USE_IE_L:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE32:      lwz [[OffsetR:[0-9]+]], [[TIUninitLD_USE_IE_L:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE:        lwzx [[TIUninitValR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TIUninitLD)
  %1 = load i32, ptr %0, align 4
  ret i32 %1
}

define void @storeTIUninitLD(i32 noundef signext %i) {
; SMALL-LABEL:  storeTIUninitLD:
; SMALL64:      ld [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL:L..C[0-9]+]](2)
; SMALL:        bla .__tls_get_mod[PR]
; SMALL64:      ld [[OffsetR:[0-9]+]], [[TIUninitLD_USE_IE_L:L..C[0-9]+]](2)
; SMALL32:      lwz [[OffsetR:[0-9]+]], [[TIUninitLD_USE_IE_L:L..C[0-9]+]](2)
; SMALL:        stwx [[TIUninitValR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
;
; LARGE-LABEL:  storeTIUninitLD:
; LARGE64:      addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE:        addis [[OffsetHR:[0-9]+]], [[TIUninitLD_USE_IE_L:L..C[0-9]+]]@u(2)
; LARGE32:      addis [[ModuleHandleHR:[0-9]+]], [[ModuleHandleL:L..C[0-9]+]]@u(2)
; LARGE64:      ld [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE32:      lwz [[ModuleHandleR:3]], [[ModuleHandleL]]@l([[ModuleHandleHR]])
; LARGE:        bla .__tls_get_mod[PR]
; LARGE64:      ld [[OffsetR:[0-9]+]], [[TIUninitLD_USE_IE_L:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE32:      lwz [[OffsetR:[0-9]+]], [[TIUninitLD_USE_IE_L:L..C[0-9]+]]@l([[OffsetHR]])
; LARGE:        stwx [[TIUninitValR:[0-9]+]], [[ModuleHandleR]], [[OffsetR]]
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TIUninitLD)
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
; SMALL:        [[TIInitIE_USE_LD_L]]:
; SMALL-NEXT:   .tc _Renamed..5f24__TLSLD.TIInitIE[TC],TIInitIE[TL]@ld
; SMALL-NEXT:   .rename _Renamed..5f24__TLSLD.TIInitIE[TC],"TIInitIE"
; SMALL:        [[TIUninitIE_USE_LD_L]]:
; SMALL-NEXT:   .tc _Renamed..5f24__TLSLD.TIUninitIE[TC],TIUninitIE[UL]@ld
; SMALL-NEXT:   .rename _Renamed..5f24__TLSLD.TIUninitIE[TC],"TIUninitIE"
; SMALL:        L..C3:
; SMALL-NEXT:   .tc TIInitIE[TC],TIInitIE[TL]@ie
; SMALL:        L..C4:
; SMALL-NEXT:   .tc TIUninitIE[TC],TIUninitIE[UL]@ie
; SMALL:        L..C5:
; SMALL-NEXT:   .tc TIInitLD[TC],TIInitLD[TL]@ie
; SMALL:        L..C6:
; SMALL-NEXT:   .tc TIUninitLD[TC],TIUninitLD[UL]@ie
; SMALL:        [[TIInitLD_USE_IE_L]]:
; SMALL-NEXT:   .tc _Renamed..5f24__TLSLD.TIInitLD[TC],TIInitLD[TL]@ld
; SMALL-NEXT:   .rename _Renamed..5f24__TLSLD.TIInitLD[TC],"TIInitLD"
; SMALL:        [[TIUninitLD_USE_IE_L]]:
; SMALL-NEXT:   .tc _Renamed..5f24__TLSLD.TIUninitLD[TC],TIUninitLD[UL]@ld
; SMALL-NEXT:   .rename _Renamed..5f24__TLSLD.TIUninitLD[TC],"TIUninitLD"

; LARGE64:        [[ModuleHandleL]]:
; LARGE64-NEXT:   .tc _Renamed..5f24__TLSML[TC],_Renamed..5f24__TLSML[TC]@ml
; LARGE64-NEXT:   .rename _Renamed..5f24__TLSML[TC],"_$TLSML"
; LARGE64:        [[TIInitIE_USE_LD_L]]:
; LARGE64-NEXT:   .tc _Renamed..5f24__TLSLD.TIInitIE[TE],TIInitIE[TL]@ld
; LARGE64-NEXT:   .rename _Renamed..5f24__TLSLD.TIInitIE[TE],"TIInitIE"
; LARGE64:        [[TIUninitIE_USE_LD_L]]:
; LARGE64-NEXT:   .tc _Renamed..5f24__TLSLD.TIUninitIE[TE],TIUninitIE[UL]@ld
; LARGE64-NEXT:   .rename _Renamed..5f24__TLSLD.TIUninitIE[TE],"TIUninitIE"

; LARGE32:        [[TIInitIE_USE_LD_L]]:
; LARGE32-NEXT:   .tc _Renamed..5f24__TLSLD.TIInitIE[TE],TIInitIE[TL]@ld
; LARGE32-NEXT:   .rename _Renamed..5f24__TLSLD.TIInitIE[TE],"TIInitIE"
; LARGE32:        [[ModuleHandleL]]:
; LARGE32-NEXT:   .tc _Renamed..5f24__TLSML[TC],_Renamed..5f24__TLSML[TC]@ml
; LARGE32-NEXT:   .rename _Renamed..5f24__TLSML[TC],"_$TLSML"
; LARGE32:        [[TIUninitIE_USE_LD_L]]:
; LARGE32-NEXT:   .tc _Renamed..5f24__TLSLD.TIUninitIE[TE],TIUninitIE[UL]@ld
; LARGE32-NEXT:   .rename _Renamed..5f24__TLSLD.TIUninitIE[TE],"TIUninitIE"

; LARGE:        L..C3:
; LARGE-NEXT:   .tc TIInitIE[TE],TIInitIE[TL]@ie
; LARGE:        L..C4:
; LARGE-NEXT:   .tc TIUninitIE[TE],TIUninitIE[UL]@ie
; LARGE:        L..C5:
; LARGE-NEXT:   .tc TIInitLD[TE],TIInitLD[TL]@ie
; LARGE:        L..C6:
; LARGE-NEXT:   .tc TIUninitLD[TE],TIUninitLD[UL]@ie
; LARGE:        [[TIInitLD_USE_IE_L]]:
; LARGE-NEXT:   .tc _Renamed..5f24__TLSLD.TIInitLD[TE],TIInitLD[TL]@ld
; LARGE-NEXT:   .rename _Renamed..5f24__TLSLD.TIInitLD[TE],"TIInitLD"
; LARGE:        [[TIUninitLD_USE_IE_L]]:
; LARGE-NEXT:   .tc _Renamed..5f24__TLSLD.TIUninitLD[TE],TIUninitLD[UL]@ld
; LARGE-NEXT:   .rename _Renamed..5f24__TLSLD.TIUninitLD[TE],"TIUninitLD"

declare nonnull ptr @llvm.threadlocal.address.p0(ptr nonnull)

attributes #0 = { "target-features"="+aix-func-use-tls-local-dynamic" }

attributes #1 = { "target-features"="+aix-func-use-tls-initial-exec" }
