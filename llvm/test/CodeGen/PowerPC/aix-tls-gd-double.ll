; RUN: llc  -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec \
; RUN:      -mtriple powerpc-ibm-aix-xcoff < %s | FileCheck %s \
; RUN:      --check-prefix=SMALL32
; RUN: llc  -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec \
; RUN:      -mtriple powerpc-ibm-aix-xcoff --code-model=large < %s \
; RUN:      | FileCheck %s --check-prefix=LARGE32
; RUN: llc  -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec \
; RUN:      -mtriple powerpc64-ibm-aix-xcoff < %s | FileCheck %s \
; RUN:      --check-prefix=SMALL64
; RUN: llc  -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec \
; RUN:      -mtriple powerpc64-ibm-aix-xcoff --code-model=large < %s \
; RUN:      | FileCheck %s --check-prefix=LARGE64

@TGInit = thread_local global double 1.000000e+00, align 8
@TWInit = weak thread_local global double 1.000000e+00, align 8
@GInit = global double 1.000000e+00, align 8
@TGUninit = thread_local global double 0.000000e+00, align 8
@TIInit = internal thread_local global double 1.000000e+00, align 8

; Function Attrs: nofree norecurse nounwind willreturn writeonly
define void @storesTGUninit(double %Val) #0 {
; SMALL32-LABEL: storesTGUninit:
; SMALL32:       # %bb.0: # %entry
; SMALL32-NEXT:    mflr 0
; SMALL32-NEXT:    stw 0, 8(1)
; SMALL32-NEXT:    stwu 1, -32(1)
; SMALL32-NEXT:    lwz 3, L..C0(2)
; SMALL32-NEXT:    lwz 4, L..C1(2)
; SMALL32-NEXT:    bla .__tls_get_addr[PR]
; SMALL32-NEXT:    stfd 1, 0(3)
; SMALL32-NEXT:    addi 1, 1, 32
; SMALL32-NEXT:    lwz 0, 8(1)
; SMALL32-NEXT:    mtlr 0
; SMALL32-NEXT:    blr
;
; LARGE32-LABEL: storesTGUninit:
; LARGE32:       # %bb.0: # %entry
; LARGE32-NEXT:    mflr 0
; LARGE32-NEXT:    stw 0, 8(1)
; LARGE32-NEXT:    stwu 1, -32(1)
; LARGE32-NEXT:    addis 3, L..C0@u(2)
; LARGE32-NEXT:    addis 4, L..C1@u(2)
; LARGE32-NEXT:    lwz 3, L..C0@l(3)
; LARGE32-NEXT:    lwz 4, L..C1@l(4)
; LARGE32-NEXT:    bla .__tls_get_addr[PR]
; LARGE32-NEXT:    stfd 1, 0(3)
; LARGE32-NEXT:    addi 1, 1, 32
; LARGE32-NEXT:    lwz 0, 8(1)
; LARGE32-NEXT:    mtlr 0
; LARGE32-NEXT:    blr
;
; SMALL64-LABEL: storesTGUninit:
; SMALL64:       # %bb.0: # %entry
; SMALL64-NEXT:    mflr 0
; SMALL64-NEXT:    std 0, 16(1)
; SMALL64-NEXT:    stdu 1, -48(1)
; SMALL64-NEXT:    ld 3, L..C0(2)
; SMALL64-NEXT:    ld 4, L..C1(2)
; SMALL64-NEXT:    bla .__tls_get_addr[PR]
; SMALL64-NEXT:    stfd 1, 0(3)
; SMALL64-NEXT:    addi 1, 1, 48
; SMALL64-NEXT:    ld 0, 16(1)
; SMALL64-NEXT:    mtlr 0
; SMALL64-NEXT:    blr
;
; LARGE64-LABEL: storesTGUninit:
; LARGE64:       # %bb.0: # %entry
; LARGE64-NEXT:    mflr 0
; LARGE64-NEXT:    std 0, 16(1)
; LARGE64-NEXT:    stdu 1, -48(1)
; LARGE64-NEXT:    addis 3, L..C0@u(2)
; LARGE64-NEXT:    addis 4, L..C1@u(2)
; LARGE64-NEXT:    ld 3, L..C0@l(3)
; LARGE64-NEXT:    ld 4, L..C1@l(4)
; LARGE64-NEXT:    bla .__tls_get_addr[PR]
; LARGE64-NEXT:    stfd 1, 0(3)
; LARGE64-NEXT:    addi 1, 1, 48
; LARGE64-NEXT:    ld 0, 16(1)
; LARGE64-NEXT:    mtlr 0
; LARGE64-NEXT:    blr
entry:
  store double %Val, ptr @TGUninit, align 8
  ret void
}

; Function Attrs: nofree norecurse nounwind willreturn writeonly
define void @storesTGInit(double %Val) #0 {
; SMALL32-LABEL: storesTGInit:
; SMALL32:       # %bb.0: # %entry
; SMALL32-NEXT:    mflr 0
; SMALL32-NEXT:    stw 0, 8(1)
; SMALL32-NEXT:    stwu 1, -32(1)
; SMALL32-NEXT:    lwz 3, L..C2(2)
; SMALL32-NEXT:    lwz 4, L..C3(2)
; SMALL32-NEXT:    bla .__tls_get_addr[PR]
; SMALL32-NEXT:    stfd 1, 0(3)
; SMALL32-NEXT:    addi 1, 1, 32
; SMALL32-NEXT:    lwz 0, 8(1)
; SMALL32-NEXT:    mtlr 0
; SMALL32-NEXT:    blr
;
; LARGE32-LABEL: storesTGInit:
; LARGE32:       # %bb.0: # %entry
; LARGE32-NEXT:    mflr 0
; LARGE32-NEXT:    stw 0, 8(1)
; LARGE32-NEXT:    stwu 1, -32(1)
; LARGE32-NEXT:    addis 3, L..C2@u(2)
; LARGE32-NEXT:    addis 4, L..C3@u(2)
; LARGE32-NEXT:    lwz 3, L..C2@l(3)
; LARGE32-NEXT:    lwz 4, L..C3@l(4)
; LARGE32-NEXT:    bla .__tls_get_addr[PR]
; LARGE32-NEXT:    stfd 1, 0(3)
; LARGE32-NEXT:    addi 1, 1, 32
; LARGE32-NEXT:    lwz 0, 8(1)
; LARGE32-NEXT:    mtlr 0
; LARGE32-NEXT:    blr
;
; SMALL64-LABEL: storesTGInit:
; SMALL64:       # %bb.0: # %entry
; SMALL64-NEXT:    mflr 0
; SMALL64-NEXT:    std 0, 16(1)
; SMALL64-NEXT:    stdu 1, -48(1)
; SMALL64-NEXT:    ld 3, L..C2(2)
; SMALL64-NEXT:    ld 4, L..C3(2)
; SMALL64-NEXT:    bla .__tls_get_addr[PR]
; SMALL64-NEXT:    stfd 1, 0(3)
; SMALL64-NEXT:    addi 1, 1, 48
; SMALL64-NEXT:    ld 0, 16(1)
; SMALL64-NEXT:    mtlr 0
; SMALL64-NEXT:    blr
;
; LARGE64-LABEL: storesTGInit:
; LARGE64:       # %bb.0: # %entry
; LARGE64-NEXT:    mflr 0
; LARGE64-NEXT:    std 0, 16(1)
; LARGE64-NEXT:    stdu 1, -48(1)
; LARGE64-NEXT:    addis 3, L..C2@u(2)
; LARGE64-NEXT:    addis 4, L..C3@u(2)
; LARGE64-NEXT:    ld 3, L..C2@l(3)
; LARGE64-NEXT:    ld 4, L..C3@l(4)
; LARGE64-NEXT:    bla .__tls_get_addr[PR]
; LARGE64-NEXT:    stfd 1, 0(3)
; LARGE64-NEXT:    addi 1, 1, 48
; LARGE64-NEXT:    ld 0, 16(1)
; LARGE64-NEXT:    mtlr 0
; LARGE64-NEXT:    blr
entry:
  store double %Val, ptr @TGInit, align 8
  ret void
}

; Function Attrs: nofree norecurse nounwind willreturn writeonly
define void @storesTIInit(double %Val) #0 {
; SMALL32-LABEL: storesTIInit:
; SMALL32:       # %bb.0: # %entry
; SMALL32-NEXT:    mflr 0
; SMALL32-NEXT:    stw 0, 8(1)
; SMALL32-NEXT:    stwu 1, -32(1)
; SMALL32-NEXT:    lwz 3, L..C4(2)
; SMALL32-NEXT:    lwz 4, L..C5(2)
; SMALL32-NEXT:    bla .__tls_get_addr[PR]
; SMALL32-NEXT:    stfd 1, 0(3)
; SMALL32-NEXT:    addi 1, 1, 32
; SMALL32-NEXT:    lwz 0, 8(1)
; SMALL32-NEXT:    mtlr 0
; SMALL32-NEXT:    blr
;
; LARGE32-LABEL: storesTIInit:
; LARGE32:       # %bb.0: # %entry
; LARGE32-NEXT:    mflr 0
; LARGE32-NEXT:    stw 0, 8(1)
; LARGE32-NEXT:    stwu 1, -32(1)
; LARGE32-NEXT:    addis 3, L..C4@u(2)
; LARGE32-NEXT:    addis 4, L..C5@u(2)
; LARGE32-NEXT:    lwz 3, L..C4@l(3)
; LARGE32-NEXT:    lwz 4, L..C5@l(4)
; LARGE32-NEXT:    bla .__tls_get_addr[PR]
; LARGE32-NEXT:    stfd 1, 0(3)
; LARGE32-NEXT:    addi 1, 1, 32
; LARGE32-NEXT:    lwz 0, 8(1)
; LARGE32-NEXT:    mtlr 0
; LARGE32-NEXT:    blr
;
; SMALL64-LABEL: storesTIInit:
; SMALL64:       # %bb.0: # %entry
; SMALL64-NEXT:    mflr 0
; SMALL64-NEXT:    std 0, 16(1)
; SMALL64-NEXT:    stdu 1, -48(1)
; SMALL64-NEXT:    ld 3, L..C4(2)
; SMALL64-NEXT:    ld 4, L..C5(2)
; SMALL64-NEXT:    bla .__tls_get_addr[PR]
; SMALL64-NEXT:    stfd 1, 0(3)
; SMALL64-NEXT:    addi 1, 1, 48
; SMALL64-NEXT:    ld 0, 16(1)
; SMALL64-NEXT:    mtlr 0
; SMALL64-NEXT:    blr
;
; LARGE64-LABEL: storesTIInit:
; LARGE64:       # %bb.0: # %entry
; LARGE64-NEXT:    mflr 0
; LARGE64-NEXT:    std 0, 16(1)
; LARGE64-NEXT:    stdu 1, -48(1)
; LARGE64-NEXT:    addis 3, L..C4@u(2)
; LARGE64-NEXT:    addis 4, L..C5@u(2)
; LARGE64-NEXT:    ld 3, L..C4@l(3)
; LARGE64-NEXT:    ld 4, L..C5@l(4)
; LARGE64-NEXT:    bla .__tls_get_addr[PR]
; LARGE64-NEXT:    stfd 1, 0(3)
; LARGE64-NEXT:    addi 1, 1, 48
; LARGE64-NEXT:    ld 0, 16(1)
; LARGE64-NEXT:    mtlr 0
; LARGE64-NEXT:    blr
entry:
  store double %Val, ptr @TIInit, align 8
  ret void
}

; Function Attrs: nofree norecurse nounwind willreturn writeonly
define void @storesTWInit(double %Val) #0 {
; SMALL32-LABEL: storesTWInit:
; SMALL32:       # %bb.0: # %entry
; SMALL32-NEXT:    mflr 0
; SMALL32-NEXT:    stw 0, 8(1)
; SMALL32-NEXT:    stwu 1, -32(1)
; SMALL32-NEXT:    lwz 3, L..C6(2)
; SMALL32-NEXT:    lwz 4, L..C7(2)
; SMALL32-NEXT:    bla .__tls_get_addr[PR]
; SMALL32-NEXT:    stfd 1, 0(3)
; SMALL32-NEXT:    addi 1, 1, 32
; SMALL32-NEXT:    lwz 0, 8(1)
; SMALL32-NEXT:    mtlr 0
; SMALL32-NEXT:    blr
;
; LARGE32-LABEL: storesTWInit:
; LARGE32:       # %bb.0: # %entry
; LARGE32-NEXT:    mflr 0
; LARGE32-NEXT:    stw 0, 8(1)
; LARGE32-NEXT:    stwu 1, -32(1)
; LARGE32-NEXT:    addis 3, L..C6@u(2)
; LARGE32-NEXT:    addis 4, L..C7@u(2)
; LARGE32-NEXT:    lwz 3, L..C6@l(3)
; LARGE32-NEXT:    lwz 4, L..C7@l(4)
; LARGE32-NEXT:    bla .__tls_get_addr[PR]
; LARGE32-NEXT:    stfd 1, 0(3)
; LARGE32-NEXT:    addi 1, 1, 32
; LARGE32-NEXT:    lwz 0, 8(1)
; LARGE32-NEXT:    mtlr 0
; LARGE32-NEXT:    blr
;
; SMALL64-LABEL: storesTWInit:
; SMALL64:       # %bb.0: # %entry
; SMALL64-NEXT:    mflr 0
; SMALL64-NEXT:    std 0, 16(1)
; SMALL64-NEXT:    stdu 1, -48(1)
; SMALL64-NEXT:    ld 3, L..C6(2)
; SMALL64-NEXT:    ld 4, L..C7(2)
; SMALL64-NEXT:    bla .__tls_get_addr[PR]
; SMALL64-NEXT:    stfd 1, 0(3)
; SMALL64-NEXT:    addi 1, 1, 48
; SMALL64-NEXT:    ld 0, 16(1)
; SMALL64-NEXT:    mtlr 0
; SMALL64-NEXT:    blr
;
; LARGE64-LABEL: storesTWInit:
; LARGE64:       # %bb.0: # %entry
; LARGE64-NEXT:    mflr 0
; LARGE64-NEXT:    std 0, 16(1)
; LARGE64-NEXT:    stdu 1, -48(1)
; LARGE64-NEXT:    addis 3, L..C6@u(2)
; LARGE64-NEXT:    addis 4, L..C7@u(2)
; LARGE64-NEXT:    ld 3, L..C6@l(3)
; LARGE64-NEXT:    ld 4, L..C7@l(4)
; LARGE64-NEXT:    bla .__tls_get_addr[PR]
; LARGE64-NEXT:    stfd 1, 0(3)
; LARGE64-NEXT:    addi 1, 1, 48
; LARGE64-NEXT:    ld 0, 16(1)
; LARGE64-NEXT:    mtlr 0
; LARGE64-NEXT:    blr
entry:
  store double %Val, ptr @TWInit, align 8
  ret void
}

; Function Attrs: norecurse nounwind readonly willreturn
define double @loadsTGUninit() #1 {
; SMALL32-LABEL: loadsTGUninit:
; SMALL32:       # %bb.0: # %entry
; SMALL32-NEXT:    mflr 0
; SMALL32-NEXT:    stw 0, 8(1)
; SMALL32-NEXT:    stwu 1, -32(1)
; SMALL32-NEXT:    lwz 3, L..C0(2)
; SMALL32-NEXT:    lwz 4, L..C1(2)
; SMALL32-NEXT:    bla .__tls_get_addr[PR]
; SMALL32-NEXT:    lwz 4, L..C8(2)
; SMALL32-NEXT:    lfd 0, 0(3)
; SMALL32-NEXT:    lfd 1, 0(4)
; SMALL32-NEXT:    fadd 1, 0, 1
; SMALL32-NEXT:    addi 1, 1, 32
; SMALL32-NEXT:    lwz 0, 8(1)
; SMALL32-NEXT:    mtlr 0
; SMALL32-NEXT:    blr
;
; LARGE32-LABEL: loadsTGUninit:
; LARGE32:       # %bb.0: # %entry
; LARGE32-NEXT:    mflr 0
; LARGE32-NEXT:    stw 0, 8(1)
; LARGE32-NEXT:    stwu 1, -32(1)
; LARGE32-NEXT:    addis 3, L..C0@u(2)
; LARGE32-NEXT:    addis 4, L..C1@u(2)
; LARGE32-NEXT:    lwz 3, L..C0@l(3)
; LARGE32-NEXT:    lwz 4, L..C1@l(4)
; LARGE32-NEXT:    bla .__tls_get_addr[PR]
; LARGE32-NEXT:    lfd 0, 0(3)
; LARGE32-NEXT:    addis 3, L..C8@u(2)
; LARGE32-NEXT:    lwz 3, L..C8@l(3)
; LARGE32-NEXT:    lfd 1, 0(3)
; LARGE32-NEXT:    fadd 1, 0, 1
; LARGE32-NEXT:    addi 1, 1, 32
; LARGE32-NEXT:    lwz 0, 8(1)
; LARGE32-NEXT:    mtlr 0
; LARGE32-NEXT:    blr
;
; SMALL64-LABEL: loadsTGUninit:
; SMALL64:       # %bb.0: # %entry
; SMALL64-NEXT:    mflr 0
; SMALL64-NEXT:    std 0, 16(1)
; SMALL64-NEXT:    stdu 1, -48(1)
; SMALL64-NEXT:    ld 3, L..C0(2)
; SMALL64-NEXT:    ld 4, L..C1(2)
; SMALL64-NEXT:    bla .__tls_get_addr[PR]
; SMALL64-NEXT:    ld 4, L..C8(2)
; SMALL64-NEXT:    lfd 0, 0(3)
; SMALL64-NEXT:    lfd 1, 0(4)
; SMALL64-NEXT:    fadd 1, 0, 1
; SMALL64-NEXT:    addi 1, 1, 48
; SMALL64-NEXT:    ld 0, 16(1)
; SMALL64-NEXT:    mtlr 0
; SMALL64-NEXT:    blr
;
; LARGE64-LABEL: loadsTGUninit:
; LARGE64:       # %bb.0: # %entry
; LARGE64-NEXT:    mflr 0
; LARGE64-NEXT:    std 0, 16(1)
; LARGE64-NEXT:    stdu 1, -48(1)
; LARGE64-NEXT:    addis 3, L..C0@u(2)
; LARGE64-NEXT:    addis 4, L..C1@u(2)
; LARGE64-NEXT:    ld 3, L..C0@l(3)
; LARGE64-NEXT:    ld 4, L..C1@l(4)
; LARGE64-NEXT:    bla .__tls_get_addr[PR]
; LARGE64-NEXT:    addis 4, L..C8@u(2)
; LARGE64-NEXT:    lfd 0, 0(3)
; LARGE64-NEXT:    ld 3, L..C8@l(4)
; LARGE64-NEXT:    lfd 1, 0(3)
; LARGE64-NEXT:    fadd 1, 0, 1
; LARGE64-NEXT:    addi 1, 1, 48
; LARGE64-NEXT:    ld 0, 16(1)
; LARGE64-NEXT:    mtlr 0
; LARGE64-NEXT:    blr
entry:
  %0 = load double, ptr @TGUninit, align 8
  %1 = load double, ptr @GInit, align 8
  %add = fadd double %0, %1
  ret double %add
}

; Function Attrs: norecurse nounwind readonly willreturn
define double @loadsTGInit() #1 {
; SMALL32-LABEL: loadsTGInit:
; SMALL32:       # %bb.0: # %entry
; SMALL32-NEXT:    mflr 0
; SMALL32-NEXT:    stw 0, 8(1)
; SMALL32-NEXT:    stwu 1, -32(1)
; SMALL32-NEXT:    lwz 3, L..C2(2)
; SMALL32-NEXT:    lwz 4, L..C3(2)
; SMALL32-NEXT:    bla .__tls_get_addr[PR]
; SMALL32-NEXT:    lwz 4, L..C8(2)
; SMALL32-NEXT:    lfd 0, 0(3)
; SMALL32-NEXT:    lfd 1, 0(4)
; SMALL32-NEXT:    fadd 1, 0, 1
; SMALL32-NEXT:    addi 1, 1, 32
; SMALL32-NEXT:    lwz 0, 8(1)
; SMALL32-NEXT:    mtlr 0
; SMALL32-NEXT:    blr
;
; LARGE32-LABEL: loadsTGInit:
; LARGE32:       # %bb.0: # %entry
; LARGE32-NEXT:    mflr 0
; LARGE32-NEXT:    stw 0, 8(1)
; LARGE32-NEXT:    stwu 1, -32(1)
; LARGE32-NEXT:    addis 3, L..C2@u(2)
; LARGE32-NEXT:    addis 4, L..C3@u(2)
; LARGE32-NEXT:    lwz 3, L..C2@l(3)
; LARGE32-NEXT:    lwz 4, L..C3@l(4)
; LARGE32-NEXT:    bla .__tls_get_addr[PR]
; LARGE32-NEXT:    lfd 0, 0(3)
; LARGE32-NEXT:    addis 3, L..C8@u(2)
; LARGE32-NEXT:    lwz 3, L..C8@l(3)
; LARGE32-NEXT:    lfd 1, 0(3)
; LARGE32-NEXT:    fadd 1, 0, 1
; LARGE32-NEXT:    addi 1, 1, 32
; LARGE32-NEXT:    lwz 0, 8(1)
; LARGE32-NEXT:    mtlr 0
; LARGE32-NEXT:    blr
;
; SMALL64-LABEL: loadsTGInit:
; SMALL64:       # %bb.0: # %entry
; SMALL64-NEXT:    mflr 0
; SMALL64-NEXT:    std 0, 16(1)
; SMALL64-NEXT:    stdu 1, -48(1)
; SMALL64-NEXT:    ld 3, L..C2(2)
; SMALL64-NEXT:    ld 4, L..C3(2)
; SMALL64-NEXT:    bla .__tls_get_addr[PR]
; SMALL64-NEXT:    ld 4, L..C8(2)
; SMALL64-NEXT:    lfd 0, 0(3)
; SMALL64-NEXT:    lfd 1, 0(4)
; SMALL64-NEXT:    fadd 1, 0, 1
; SMALL64-NEXT:    addi 1, 1, 48
; SMALL64-NEXT:    ld 0, 16(1)
; SMALL64-NEXT:    mtlr 0
; SMALL64-NEXT:    blr
;
; LARGE64-LABEL: loadsTGInit:
; LARGE64:       # %bb.0: # %entry
; LARGE64-NEXT:    mflr 0
; LARGE64-NEXT:    std 0, 16(1)
; LARGE64-NEXT:    stdu 1, -48(1)
; LARGE64-NEXT:    addis 3, L..C2@u(2)
; LARGE64-NEXT:    addis 4, L..C3@u(2)
; LARGE64-NEXT:    ld 3, L..C2@l(3)
; LARGE64-NEXT:    ld 4, L..C3@l(4)
; LARGE64-NEXT:    bla .__tls_get_addr[PR]
; LARGE64-NEXT:    addis 4, L..C8@u(2)
; LARGE64-NEXT:    lfd 0, 0(3)
; LARGE64-NEXT:    ld 3, L..C8@l(4)
; LARGE64-NEXT:    lfd 1, 0(3)
; LARGE64-NEXT:    fadd 1, 0, 1
; LARGE64-NEXT:    addi 1, 1, 48
; LARGE64-NEXT:    ld 0, 16(1)
; LARGE64-NEXT:    mtlr 0
; LARGE64-NEXT:    blr
entry:
  %0 = load double, ptr @TGInit, align 8
  %1 = load double, ptr @GInit, align 8
  %add = fadd double %0, %1
  ret double %add
}

; Function Attrs: norecurse nounwind readonly willreturn
define double @loadsTIInit() #1 {
; SMALL32-LABEL: loadsTIInit:
; SMALL32:       # %bb.0: # %entry
; SMALL32-NEXT:    mflr 0
; SMALL32-NEXT:    stw 0, 8(1)
; SMALL32-NEXT:    stwu 1, -32(1)
; SMALL32-NEXT:    lwz 3, L..C4(2)
; SMALL32-NEXT:    lwz 4, L..C5(2)
; SMALL32-NEXT:    bla .__tls_get_addr[PR]
; SMALL32-NEXT:    lwz 4, L..C8(2)
; SMALL32-NEXT:    lfd 0, 0(3)
; SMALL32-NEXT:    lfd 1, 0(4)
; SMALL32-NEXT:    fadd 1, 0, 1
; SMALL32-NEXT:    addi 1, 1, 32
; SMALL32-NEXT:    lwz 0, 8(1)
; SMALL32-NEXT:    mtlr 0
; SMALL32-NEXT:    blr
;
; LARGE32-LABEL: loadsTIInit:
; LARGE32:       # %bb.0: # %entry
; LARGE32-NEXT:    mflr 0
; LARGE32-NEXT:    stw 0, 8(1)
; LARGE32-NEXT:    stwu 1, -32(1)
; LARGE32-NEXT:    addis 3, L..C4@u(2)
; LARGE32-NEXT:    addis 4, L..C5@u(2)
; LARGE32-NEXT:    lwz 3, L..C4@l(3)
; LARGE32-NEXT:    lwz 4, L..C5@l(4)
; LARGE32-NEXT:    bla .__tls_get_addr[PR]
; LARGE32-NEXT:    lfd 0, 0(3)
; LARGE32-NEXT:    addis 3, L..C8@u(2)
; LARGE32-NEXT:    lwz 3, L..C8@l(3)
; LARGE32-NEXT:    lfd 1, 0(3)
; LARGE32-NEXT:    fadd 1, 0, 1
; LARGE32-NEXT:    addi 1, 1, 32
; LARGE32-NEXT:    lwz 0, 8(1)
; LARGE32-NEXT:    mtlr 0
; LARGE32-NEXT:    blr
;
; SMALL64-LABEL: loadsTIInit:
; SMALL64:       # %bb.0: # %entry
; SMALL64-NEXT:    mflr 0
; SMALL64-NEXT:    std 0, 16(1)
; SMALL64-NEXT:    stdu 1, -48(1)
; SMALL64-NEXT:    ld 3, L..C4(2)
; SMALL64-NEXT:    ld 4, L..C5(2)
; SMALL64-NEXT:    bla .__tls_get_addr[PR]
; SMALL64-NEXT:    ld 4, L..C8(2)
; SMALL64-NEXT:    lfd 0, 0(3)
; SMALL64-NEXT:    lfd 1, 0(4)
; SMALL64-NEXT:    fadd 1, 0, 1
; SMALL64-NEXT:    addi 1, 1, 48
; SMALL64-NEXT:    ld 0, 16(1)
; SMALL64-NEXT:    mtlr 0
; SMALL64-NEXT:    blr
;
; LARGE64-LABEL: loadsTIInit:
; LARGE64:       # %bb.0: # %entry
; LARGE64-NEXT:    mflr 0
; LARGE64-NEXT:    std 0, 16(1)
; LARGE64-NEXT:    stdu 1, -48(1)
; LARGE64-NEXT:    addis 3, L..C4@u(2)
; LARGE64-NEXT:    addis 4, L..C5@u(2)
; LARGE64-NEXT:    ld 3, L..C4@l(3)
; LARGE64-NEXT:    ld 4, L..C5@l(4)
; LARGE64-NEXT:    bla .__tls_get_addr[PR]
; LARGE64-NEXT:    addis 4, L..C8@u(2)
; LARGE64-NEXT:    lfd 0, 0(3)
; LARGE64-NEXT:    ld 3, L..C8@l(4)
; LARGE64-NEXT:    lfd 1, 0(3)
; LARGE64-NEXT:    fadd 1, 0, 1
; LARGE64-NEXT:    addi 1, 1, 48
; LARGE64-NEXT:    ld 0, 16(1)
; LARGE64-NEXT:    mtlr 0
; LARGE64-NEXT:    blr
entry:
  %0 = load double, ptr @TIInit, align 8
  %1 = load double, ptr @GInit, align 8
  %add = fadd double %0, %1
  ret double %add
}

; Function Attrs: norecurse nounwind readonly willreturn
define double @loadsTWInit() #1 {
; SMALL32-LABEL: loadsTWInit:
; SMALL32:       # %bb.0: # %entry
; SMALL32-NEXT:    mflr 0
; SMALL32-NEXT:    stw 0, 8(1)
; SMALL32-NEXT:    stwu 1, -32(1)
; SMALL32-NEXT:    lwz 3, L..C6(2)
; SMALL32-NEXT:    lwz 4, L..C7(2)
; SMALL32-NEXT:    bla .__tls_get_addr[PR]
; SMALL32-NEXT:    lwz 4, L..C8(2)
; SMALL32-NEXT:    lfd 0, 0(3)
; SMALL32-NEXT:    lfd 1, 0(4)
; SMALL32-NEXT:    fadd 1, 0, 1
; SMALL32-NEXT:    addi 1, 1, 32
; SMALL32-NEXT:    lwz 0, 8(1)
; SMALL32-NEXT:    mtlr 0
; SMALL32-NEXT:    blr
;
; LARGE32-LABEL: loadsTWInit:
; LARGE32:       # %bb.0: # %entry
; LARGE32-NEXT:    mflr 0
; LARGE32-NEXT:    stw 0, 8(1)
; LARGE32-NEXT:    stwu 1, -32(1)
; LARGE32-NEXT:    addis 3, L..C6@u(2)
; LARGE32-NEXT:    addis 4, L..C7@u(2)
; LARGE32-NEXT:    lwz 3, L..C6@l(3)
; LARGE32-NEXT:    lwz 4, L..C7@l(4)
; LARGE32-NEXT:    bla .__tls_get_addr[PR]
; LARGE32-NEXT:    lfd 0, 0(3)
; LARGE32-NEXT:    addis 3, L..C8@u(2)
; LARGE32-NEXT:    lwz 3, L..C8@l(3)
; LARGE32-NEXT:    lfd 1, 0(3)
; LARGE32-NEXT:    fadd 1, 0, 1
; LARGE32-NEXT:    addi 1, 1, 32
; LARGE32-NEXT:    lwz 0, 8(1)
; LARGE32-NEXT:    mtlr 0
; LARGE32-NEXT:    blr
;
; SMALL64-LABEL: loadsTWInit:
; SMALL64:       # %bb.0: # %entry
; SMALL64-NEXT:    mflr 0
; SMALL64-NEXT:    std 0, 16(1)
; SMALL64-NEXT:    stdu 1, -48(1)
; SMALL64-NEXT:    ld 3, L..C6(2)
; SMALL64-NEXT:    ld 4, L..C7(2)
; SMALL64-NEXT:    bla .__tls_get_addr[PR]
; SMALL64-NEXT:    ld 4, L..C8(2)
; SMALL64-NEXT:    lfd 0, 0(3)
; SMALL64-NEXT:    lfd 1, 0(4)
; SMALL64-NEXT:    fadd 1, 0, 1
; SMALL64-NEXT:    addi 1, 1, 48
; SMALL64-NEXT:    ld 0, 16(1)
; SMALL64-NEXT:    mtlr 0
; SMALL64-NEXT:    blr
;
; LARGE64-LABEL: loadsTWInit:
; LARGE64:       # %bb.0: # %entry
; LARGE64-NEXT:    mflr 0
; LARGE64-NEXT:    std 0, 16(1)
; LARGE64-NEXT:    stdu 1, -48(1)
; LARGE64-NEXT:    addis 3, L..C6@u(2)
; LARGE64-NEXT:    addis 4, L..C7@u(2)
; LARGE64-NEXT:    ld 3, L..C6@l(3)
; LARGE64-NEXT:    ld 4, L..C7@l(4)
; LARGE64-NEXT:    bla .__tls_get_addr[PR]
; LARGE64-NEXT:    addis 4, L..C8@u(2)
; LARGE64-NEXT:    lfd 0, 0(3)
; LARGE64-NEXT:    ld 3, L..C8@l(4)
; LARGE64-NEXT:    lfd 1, 0(3)
; LARGE64-NEXT:    fadd 1, 0, 1
; LARGE64-NEXT:    addi 1, 1, 48
; LARGE64-NEXT:    ld 0, 16(1)
; LARGE64-NEXT:    mtlr 0
; LARGE64-NEXT:    blr
entry:
  %0 = load double, ptr @TWInit, align 8
  %1 = load double, ptr @GInit, align 8
  %add = fadd double %0, %1
  ret double %add
}

; External symbol reference checks for .__tls_get_addr

; SMALL32: .extern .__tls_get_addr[PR]
; SMALL64: .extern .__tls_get_addr[PR]
; LARGE32: .extern .__tls_get_addr[PR]
; LARGE64: .extern .__tls_get_addr[PR]

; TOC entry checks

; SMALL32-LABEL:  .toc
; SMALL32-LABEL:  L..C0:
; SMALL32-NEXT:   .tc .TGUninit[TC],TGUninit[TL]@m
; SMALL32-LABEL:  L..C1:
; SMALL32-NEXT:   .tc TGUninit[TC],TGUninit[TL]@gd
; SMALL32-LABEL:  L..C2:
; SMALL32-NEXT:   .tc .TGInit[TC],TGInit[TL]@m
; SMALL32-LABEL:  L..C3:
; SMALL32-NEXT:   .tc TGInit[TC],TGInit[TL]@gd
; SMALL32-LABEL:  L..C4:
; SMALL32-NEXT:   .tc .TIInit[TC],TIInit[TL]@m
; SMALL32-LABEL:  L..C5:
; SMALL32-NEXT:   .tc TIInit[TC],TIInit[TL]@gd
; SMALL32-LABEL:  L..C6:
; SMALL32-NEXT:   .tc .TWInit[TC],TWInit[TL]@m
; SMALL32-LABEL:  L..C7:
; SMALL32-NEXT:   .tc TWInit[TC],TWInit[TL]@gd
; SMALL32-LABEL:  L..C8:
; SMALL32-NEXT:   .tc GInit[TC],GInit[RW]

; LARGE32-LABEL:  .toc
; LARGE32-LABEL:  L..C0:
; LARGE32-NEXT:   .tc .TGUninit[TE],TGUninit[TL]@m
; LARGE32-LABEL:  L..C1:
; LARGE32-NEXT:   .tc TGUninit[TE],TGUninit[TL]@gd
; LARGE32-LABEL:  L..C2:
; LARGE32-NEXT:   .tc .TGInit[TE],TGInit[TL]@m
; LARGE32-LABEL:  L..C3:
; LARGE32-NEXT:   .tc TGInit[TE],TGInit[TL]@gd
; LARGE32-LABEL:  L..C4:
; LARGE32-NEXT:   .tc .TIInit[TE],TIInit[TL]@m
; LARGE32-LABEL:  L..C5:
; LARGE32-NEXT:   .tc TIInit[TE],TIInit[TL]@gd
; LARGE32-LABEL:  L..C6:
; LARGE32-NEXT:   .tc .TWInit[TE],TWInit[TL]@m
; LARGE32-LABEL:  L..C7:
; LARGE32-NEXT:   .tc TWInit[TE],TWInit[TL]@gd
; LARGE32-LABEL:  L..C8:
; LARGE32-NEXT:   .tc GInit[TE],GInit[RW]

; SMALL64-LABEL:  .toc
; SMALL64-LABEL:  L..C0:
; SMALL64-NEXT:  .tc .TGUninit[TC],TGUninit[TL]@m
; SMALL64-LABEL:  L..C1:
; SMALL64-NEXT:  .tc TGUninit[TC],TGUninit[TL]@gd
; SMALL64-LABEL:  L..C2:
; SMALL64-NEXT:  .tc .TGInit[TC],TGInit[TL]@m
; SMALL64-LABEL:  L..C3:
; SMALL64-NEXT:  .tc TGInit[TC],TGInit[TL]@gd
; SMALL64-LABEL:  L..C4:
; SMALL64-NEXT:  .tc .TIInit[TC],TIInit[TL]@m
; SMALL64-LABEL:  L..C5:
; SMALL64-NEXT:  .tc TIInit[TC],TIInit[TL]@gd
; SMALL64-LABEL:  L..C6:
; SMALL64-NEXT:  .tc .TWInit[TC],TWInit[TL]@m
; SMALL64-LABEL:  L..C7:
; SMALL64-NEXT:  .tc TWInit[TC],TWInit[TL]@gd
; SMALL64-LABEL:  L..C8:
; SMALL64-NEXT:  .tc GInit[TC],GInit[RW]

; LARGE64-LABEL:  .toc
; LARGE64-LABEL:  L..C0:
; LARGE64-NEXT:  .tc .TGUninit[TE],TGUninit[TL]@m
; LARGE64-LABEL:  L..C1:
; LARGE64-NEXT:  .tc TGUninit[TE],TGUninit[TL]@gd
; LARGE64-LABEL:  L..C2:
; LARGE64-NEXT:  .tc .TGInit[TE],TGInit[TL]@m
; LARGE64-LABEL:  L..C3:
; LARGE64-NEXT:  .tc TGInit[TE],TGInit[TL]@gd
; LARGE64-LABEL:  L..C4:
; LARGE64-NEXT:  .tc .TIInit[TE],TIInit[TL]@m
; LARGE64-LABEL:  L..C5:
; LARGE64-NEXT:  .tc TIInit[TE],TIInit[TL]@gd
; LARGE64-LABEL:  L..C6:
; LARGE64-NEXT:  .tc .TWInit[TE],TWInit[TL]@m
; LARGE64-LABEL:  L..C7:
; LARGE64-NEXT:  .tc TWInit[TE],TWInit[TL]@gd
; LARGE64-LABEL:  L..C8:
; LARGE64-NEXT:  .tc GInit[TE],GInit[RW]


attributes #0 = { nofree norecurse nounwind willreturn writeonly "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pwr4" "target-features"="-altivec,-bpermd,-crypto,-direct-move,-extdiv,-float128,-htm,-mma,-paired-vector-memops,-power10-vector,-power8-vector,-power9-vector,-rop-protection,-spe,-vsx" }
attributes #1 = { norecurse nounwind readonly willreturn "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pwr4" "target-features"="-altivec,-bpermd,-crypto,-direct-move,-extdiv,-float128,-htm,-mma,-paired-vector-memops,-power10-vector,-power8-vector,-power9-vector,-rop-protection,-spe,-vsx" }
