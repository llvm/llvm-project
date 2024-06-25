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

@TGInit = thread_local global i32 1, align 4
@GInit = global i32 1, align 4
@TGUninit = thread_local global i32 0, align 4
@TIUninit = internal thread_local global i32 0, align 4
@TWUninit = weak thread_local global i32 0, align 4

; Function Attrs: nofree norecurse nounwind willreturn writeonly
define void @storesTGUninit(i32 %Val) #0 {
; SMALL32-LABEL: storesTGUninit:
; SMALL32:       # %bb.0: # %entry
; SMALL32-NEXT:    mflr 0
; SMALL32-NEXT:    stwu 1, -32(1)
; SMALL32-NEXT:    mr 6, 3
; SMALL32-NEXT:    lwz 3, L..C0(2)
; SMALL32-NEXT:    lwz 4, L..C1(2)
; SMALL32-NEXT:    stw 0, 40(1)
; SMALL32-NEXT:    bla .__tls_get_addr[PR]
; SMALL32-NEXT:    stw 6, 0(3)
; SMALL32-NEXT:    addi 1, 1, 32
; SMALL32-NEXT:    lwz 0, 8(1)
; SMALL32-NEXT:    mtlr 0
; SMALL32-NEXT:    blr
;
; LARGE32-LABEL: storesTGUninit:
; LARGE32:       # %bb.0: # %entry
; LARGE32-NEXT:    mflr 0
; LARGE32-NEXT:    stwu 1, -32(1)
; LARGE32-NEXT:    stw 0, 40(1)
; LARGE32-NEXT:    mr 6, 3
; LARGE32-NEXT:    addis 3, L..C0@u(2)
; LARGE32-NEXT:    addis 4, L..C1@u(2)
; LARGE32-NEXT:    lwz 3, L..C0@l(3)
; LARGE32-NEXT:    lwz 4, L..C1@l(4)
; LARGE32-NEXT:    bla .__tls_get_addr[PR]
; LARGE32-NEXT:    stw 6, 0(3)
; LARGE32-NEXT:    addi 1, 1, 32
; LARGE32-NEXT:    lwz 0, 8(1)
; LARGE32-NEXT:    mtlr 0
; LARGE32-NEXT:    blr
;
; SMALL64-LABEL: storesTGUninit:
; SMALL64:       # %bb.0: # %entry
; SMALL64-NEXT:    mflr 0
; SMALL64-NEXT:    stdu 1, -48(1)
; SMALL64-NEXT:    mr 6, 3
; SMALL64-NEXT:    ld 3, L..C0(2)
; SMALL64-NEXT:    ld 4, L..C1(2)
; SMALL64-NEXT:    std 0, 64(1)
; SMALL64-NEXT:    bla .__tls_get_addr[PR]
; SMALL64-NEXT:    stw 6, 0(3)
; SMALL64-NEXT:    addi 1, 1, 48
; SMALL64-NEXT:    ld 0, 16(1)
; SMALL64-NEXT:    mtlr 0
; SMALL64-NEXT:    blr
;
; LARGE64-LABEL: storesTGUninit:
; LARGE64:       # %bb.0: # %entry
; LARGE64-NEXT:    mflr 0
; LARGE64-NEXT:    stdu 1, -48(1)
; LARGE64-NEXT:    mr 6, 3
; LARGE64-NEXT:    addis 3, L..C0@u(2)
; LARGE64-NEXT:    addis 4, L..C1@u(2)
; LARGE64-NEXT:    std 0, 64(1)
; LARGE64-NEXT:    ld 3, L..C0@l(3)
; LARGE64-NEXT:    ld 4, L..C1@l(4)
; LARGE64-NEXT:    bla .__tls_get_addr[PR]
; LARGE64-NEXT:    stw 6, 0(3)
; LARGE64-NEXT:    addi 1, 1, 48
; LARGE64-NEXT:    ld 0, 16(1)
; LARGE64-NEXT:    mtlr 0
; LARGE64-NEXT:    blr
entry:
  store i32 %Val, ptr @TGUninit, align 4
  ret void
}

; Function Attrs: nofree norecurse nounwind willreturn writeonly
define void @storesTGInit(i32 %Val) #0 {
; SMALL32-LABEL: storesTGInit:
; SMALL32:       # %bb.0: # %entry
; SMALL32-NEXT:    mflr 0
; SMALL32-NEXT:    stwu 1, -32(1)
; SMALL32-NEXT:    mr 6, 3
; SMALL32-NEXT:    lwz 3, L..C2(2)
; SMALL32-NEXT:    lwz 4, L..C3(2)
; SMALL32-NEXT:    stw 0, 40(1)
; SMALL32-NEXT:    bla .__tls_get_addr[PR]
; SMALL32-NEXT:    stw 6, 0(3)
; SMALL32-NEXT:    addi 1, 1, 32
; SMALL32-NEXT:    lwz 0, 8(1)
; SMALL32-NEXT:    mtlr 0
; SMALL32-NEXT:    blr
;
; LARGE32-LABEL: storesTGInit:
; LARGE32:       # %bb.0: # %entry
; LARGE32-NEXT:    mflr 0
; LARGE32-NEXT:    stwu 1, -32(1)
; LARGE32-NEXT:    stw 0, 40(1)
; LARGE32-NEXT:    mr 6, 3
; LARGE32-NEXT:    addis 3, L..C2@u(2)
; LARGE32-NEXT:    addis 4, L..C3@u(2)
; LARGE32-NEXT:    lwz 3, L..C2@l(3)
; LARGE32-NEXT:    lwz 4, L..C3@l(4)
; LARGE32-NEXT:    bla .__tls_get_addr[PR]
; LARGE32-NEXT:    stw 6, 0(3)
; LARGE32-NEXT:    addi 1, 1, 32
; LARGE32-NEXT:    lwz 0, 8(1)
; LARGE32-NEXT:    mtlr 0
; LARGE32-NEXT:    blr
;
; SMALL64-LABEL: storesTGInit:
; SMALL64:       # %bb.0: # %entry
; SMALL64-NEXT:    mflr 0
; SMALL64-NEXT:    stdu 1, -48(1)
; SMALL64-NEXT:    mr 6, 3
; SMALL64-NEXT:    ld 3, L..C2(2)
; SMALL64-NEXT:    ld 4, L..C3(2)
; SMALL64-NEXT:    std 0, 64(1)
; SMALL64-NEXT:    bla .__tls_get_addr[PR]
; SMALL64-NEXT:    stw 6, 0(3)
; SMALL64-NEXT:    addi 1, 1, 48
; SMALL64-NEXT:    ld 0, 16(1)
; SMALL64-NEXT:    mtlr 0
; SMALL64-NEXT:    blr
;
; LARGE64-LABEL: storesTGInit:
; LARGE64:       # %bb.0: # %entry
; LARGE64-NEXT:    mflr 0
; LARGE64-NEXT:    stdu 1, -48(1)
; LARGE64-NEXT:    mr 6, 3
; LARGE64-NEXT:    addis 3, L..C2@u(2)
; LARGE64-NEXT:    addis 4, L..C3@u(2)
; LARGE64-NEXT:    std 0, 64(1)
; LARGE64-NEXT:    ld 3, L..C2@l(3)
; LARGE64-NEXT:    ld 4, L..C3@l(4)
; LARGE64-NEXT:    bla .__tls_get_addr[PR]
; LARGE64-NEXT:    stw 6, 0(3)
; LARGE64-NEXT:    addi 1, 1, 48
; LARGE64-NEXT:    ld 0, 16(1)
; LARGE64-NEXT:    mtlr 0
; LARGE64-NEXT:    blr
entry:
  store i32 %Val, ptr @TGInit, align 4
  ret void
}

; Function Attrs: nofree norecurse nounwind willreturn writeonly
define void @storesTIUninit(i32 %Val) #0 {
; SMALL32-LABEL: storesTIUninit:
; SMALL32:       # %bb.0: # %entry
; SMALL32-NEXT:    mflr 0
; SMALL32-NEXT:    stwu 1, -32(1)
; SMALL32-NEXT:    mr 6, 3
; SMALL32-NEXT:    lwz 3, L..C4(2)
; SMALL32-NEXT:    stw 0, 40(1)
; SMALL32-NEXT:    bla .__tls_get_mod[PR]
; SMALL32-NEXT:    lwz 4, L..C5(2)
; SMALL32-NEXT:    stwx 6, 3, 4
; SMALL32-NEXT:    addi 1, 1, 32
; SMALL32-NEXT:    lwz 0, 8(1)
; SMALL32-NEXT:    mtlr 0
; SMALL32-NEXT:    blr
;
; LARGE32-LABEL: storesTIUninit:
; LARGE32:       # %bb.0: # %entry
; LARGE32-NEXT:    mflr 0
; LARGE32-NEXT:    stwu 1, -32(1)
; LARGE32-NEXT:    stw 0, 40(1)
; LARGE32-NEXT:    mr 6, 3
; LARGE32-NEXT:    addis 7, L..C4@u(2)
; LARGE32-NEXT:    addis 3, L..C5@u(2)
; LARGE32-NEXT:    lwz 3, L..C5@l(3)
; LARGE32-NEXT:    bla .__tls_get_mod[PR]
; LARGE32-NEXT:    lwz 4, L..C4@l(7)
; LARGE32-NEXT:    stwx 6, 3, 4
; LARGE32-NEXT:    addi 1, 1, 32
; LARGE32-NEXT:    lwz 0, 8(1)
; LARGE32-NEXT:    mtlr 0
; LARGE32-NEXT:    blr
;
; SMALL64-LABEL: storesTIUninit:
; SMALL64:       # %bb.0: # %entry
; SMALL64-NEXT:    mflr 0
; SMALL64-NEXT:    stdu 1, -48(1)
; SMALL64-NEXT:    mr 6, 3
; SMALL64-NEXT:    ld 3, L..C4(2)
; SMALL64-NEXT:    std 0, 64(1)
; SMALL64-NEXT:    bla .__tls_get_mod[PR]
; SMALL64-NEXT:    ld 4, L..C5(2)
; SMALL64-NEXT:    stwx 6, 3, 4
; SMALL64-NEXT:    addi 1, 1, 48
; SMALL64-NEXT:    ld 0, 16(1)
; SMALL64-NEXT:    mtlr 0
; SMALL64-NEXT:    blr
;
; LARGE64-LABEL: storesTIUninit:
; LARGE64:       # %bb.0: # %entry
; LARGE64-NEXT:    mflr 0
; LARGE64-NEXT:    stdu 1, -48(1)
; LARGE64-NEXT:    mr 6, 3
; LARGE64-NEXT:    addis 3, L..C4@u(2)
; LARGE64-NEXT:    std 0, 64(1)
; LARGE64-NEXT:    addis 7, L..C5@u(2)
; LARGE64-NEXT:    ld 3, L..C4@l(3)
; LARGE64-NEXT:    bla .__tls_get_mod[PR]
; LARGE64-NEXT:    ld 4, L..C5@l(7)
; LARGE64-NEXT:    stwx 6, 3, 4
; LARGE64-NEXT:    addi 1, 1, 48
; LARGE64-NEXT:    ld 0, 16(1)
; LARGE64-NEXT:    mtlr 0
; LARGE64-NEXT:    blr
entry:
  store i32 %Val, ptr @TIUninit, align 4
  ret void
}

; Function Attrs: nofree norecurse nounwind willreturn writeonly
define void @storesTWUninit(i32 %Val) #0 {
; SMALL32-LABEL: storesTWUninit:
; SMALL32:       # %bb.0: # %entry
; SMALL32-NEXT:    mflr 0
; SMALL32-NEXT:    stwu 1, -32(1)
; SMALL32-NEXT:    mr 6, 3
; SMALL32-NEXT:    lwz 3, L..C6(2)
; SMALL32-NEXT:    lwz 4, L..C7(2)
; SMALL32-NEXT:    stw 0, 40(1)
; SMALL32-NEXT:    bla .__tls_get_addr[PR]
; SMALL32-NEXT:    stw 6, 0(3)
; SMALL32-NEXT:    addi 1, 1, 32
; SMALL32-NEXT:    lwz 0, 8(1)
; SMALL32-NEXT:    mtlr 0
; SMALL32-NEXT:    blr
;
; LARGE32-LABEL: storesTWUninit:
; LARGE32:       # %bb.0: # %entry
; LARGE32-NEXT:    mflr 0
; LARGE32-NEXT:    stwu 1, -32(1)
; LARGE32-NEXT:    stw 0, 40(1)
; LARGE32-NEXT:    mr 6, 3
; LARGE32-NEXT:    addis 3, L..C6@u(2)
; LARGE32-NEXT:    addis 4, L..C7@u(2)
; LARGE32-NEXT:    lwz 3, L..C6@l(3)
; LARGE32-NEXT:    lwz 4, L..C7@l(4)
; LARGE32-NEXT:    bla .__tls_get_addr[PR]
; LARGE32-NEXT:    stw 6, 0(3)
; LARGE32-NEXT:    addi 1, 1, 32
; LARGE32-NEXT:    lwz 0, 8(1)
; LARGE32-NEXT:    mtlr 0
; LARGE32-NEXT:    blr
;
; SMALL64-LABEL: storesTWUninit:
; SMALL64:       # %bb.0: # %entry
; SMALL64-NEXT:    mflr 0
; SMALL64-NEXT:    stdu 1, -48(1)
; SMALL64-NEXT:    mr 6, 3
; SMALL64-NEXT:    ld 3, L..C6(2)
; SMALL64-NEXT:    ld 4, L..C7(2)
; SMALL64-NEXT:    std 0, 64(1)
; SMALL64-NEXT:    bla .__tls_get_addr[PR]
; SMALL64-NEXT:    stw 6, 0(3)
; SMALL64-NEXT:    addi 1, 1, 48
; SMALL64-NEXT:    ld 0, 16(1)
; SMALL64-NEXT:    mtlr 0
; SMALL64-NEXT:    blr
;
; LARGE64-LABEL: storesTWUninit:
; LARGE64:       # %bb.0: # %entry
; LARGE64-NEXT:    mflr 0
; LARGE64-NEXT:    stdu 1, -48(1)
; LARGE64-NEXT:    mr 6, 3
; LARGE64-NEXT:    addis 3, L..C6@u(2)
; LARGE64-NEXT:    addis 4, L..C7@u(2)
; LARGE64-NEXT:    std 0, 64(1)
; LARGE64-NEXT:    ld 3, L..C6@l(3)
; LARGE64-NEXT:    ld 4, L..C7@l(4)
; LARGE64-NEXT:    bla .__tls_get_addr[PR]
; LARGE64-NEXT:    stw 6, 0(3)
; LARGE64-NEXT:    addi 1, 1, 48
; LARGE64-NEXT:    ld 0, 16(1)
; LARGE64-NEXT:    mtlr 0
; LARGE64-NEXT:    blr
entry:
  store i32 %Val, ptr @TWUninit, align 4
  ret void
}

; Function Attrs: norecurse nounwind readonly willreturn
define i32 @loadsTGUninit() #1 {
; SMALL32-LABEL: loadsTGUninit:
; SMALL32:       # %bb.0: # %entry
; SMALL32-NEXT:    mflr 0
; SMALL32-NEXT:    stwu 1, -32(1)
; SMALL32-NEXT:    lwz 3, L..C0(2)
; SMALL32-NEXT:    lwz 4, L..C1(2)
; SMALL32-NEXT:    stw 0, 40(1)
; SMALL32-NEXT:    bla .__tls_get_addr[PR]
; SMALL32-NEXT:    lwz 4, L..C8(2)
; SMALL32-NEXT:    lwz 3, 0(3)
; SMALL32-NEXT:    lwz 4, 0(4)
; SMALL32-NEXT:    add 3, 4, 3
; SMALL32-NEXT:    addi 1, 1, 32
; SMALL32-NEXT:    lwz 0, 8(1)
; SMALL32-NEXT:    mtlr 0
; SMALL32-NEXT:    blr
;
; LARGE32-LABEL: loadsTGUninit:
; LARGE32:       # %bb.0: # %entry
; LARGE32-NEXT:    mflr 0
; LARGE32-NEXT:    stwu 1, -32(1)
; LARGE32-NEXT:    stw 0, 40(1)
; LARGE32-NEXT:    addis 3, L..C0@u(2)
; LARGE32-NEXT:    addis 4, L..C1@u(2)
; LARGE32-NEXT:    lwz 3, L..C0@l(3)
; LARGE32-NEXT:    lwz 4, L..C1@l(4)
; LARGE32-NEXT:    bla .__tls_get_addr[PR]
; LARGE32-NEXT:    lwz 3, 0(3)
; LARGE32-NEXT:    addis 4, L..C8@u(2)
; LARGE32-NEXT:    lwz 4, L..C8@l(4)
; LARGE32-NEXT:    lwz 4, 0(4)
; LARGE32-NEXT:    add 3, 4, 3
; LARGE32-NEXT:    addi 1, 1, 32
; LARGE32-NEXT:    lwz 0, 8(1)
; LARGE32-NEXT:    mtlr 0
; LARGE32-NEXT:    blr
;
; SMALL64-LABEL: loadsTGUninit:
; SMALL64:       # %bb.0: # %entry
; SMALL64-NEXT:    mflr 0
; SMALL64-NEXT:    stdu 1, -48(1)
; SMALL64-NEXT:    ld 3, L..C0(2)
; SMALL64-NEXT:    ld 4, L..C1(2)
; SMALL64-NEXT:    std 0, 64(1)
; SMALL64-NEXT:    bla .__tls_get_addr[PR]
; SMALL64-NEXT:    ld 4, L..C8(2)
; SMALL64-NEXT:    lwz 3, 0(3)
; SMALL64-NEXT:    lwz 4, 0(4)
; SMALL64-NEXT:    add 3, 4, 3
; SMALL64-NEXT:    addi 1, 1, 48
; SMALL64-NEXT:    ld 0, 16(1)
; SMALL64-NEXT:    mtlr 0
; SMALL64-NEXT:    blr
;
; LARGE64-LABEL: loadsTGUninit:
; LARGE64:       # %bb.0: # %entry
; LARGE64-NEXT:    mflr 0
; LARGE64-NEXT:    stdu 1, -48(1)
; LARGE64-NEXT:    addis 3, L..C0@u(2)
; LARGE64-NEXT:    addis 4, L..C1@u(2)
; LARGE64-NEXT:    std 0, 64(1)
; LARGE64-NEXT:    ld 3, L..C0@l(3)
; LARGE64-NEXT:    ld 4, L..C1@l(4)
; LARGE64-NEXT:    bla .__tls_get_addr[PR]
; LARGE64-NEXT:    addis 4, L..C8@u(2)
; LARGE64-NEXT:    lwz 3, 0(3)
; LARGE64-NEXT:    ld 4, L..C8@l(4)
; LARGE64-NEXT:    lwz 4, 0(4)
; LARGE64-NEXT:    add 3, 4, 3
; LARGE64-NEXT:    addi 1, 1, 48
; LARGE64-NEXT:    ld 0, 16(1)
; LARGE64-NEXT:    mtlr 0
; LARGE64-NEXT:    blr
entry:
  %0 = load i32, ptr @TGUninit, align 4
  %1 = load i32, ptr @GInit, align 4
  %add = add nsw i32 %1, %0
  ret i32 %add
}

; Function Attrs: norecurse nounwind readonly willreturn
define i32 @loadsTGInit() #1 {
; SMALL32-LABEL: loadsTGInit:
; SMALL32:       # %bb.0: # %entry
; SMALL32-NEXT:    mflr 0
; SMALL32-NEXT:    stwu 1, -32(1)
; SMALL32-NEXT:    lwz 3, L..C2(2)
; SMALL32-NEXT:    lwz 4, L..C3(2)
; SMALL32-NEXT:    stw 0, 40(1)
; SMALL32-NEXT:    bla .__tls_get_addr[PR]
; SMALL32-NEXT:    lwz 4, L..C8(2)
; SMALL32-NEXT:    lwz 3, 0(3)
; SMALL32-NEXT:    lwz 4, 0(4)
; SMALL32-NEXT:    add 3, 4, 3
; SMALL32-NEXT:    addi 1, 1, 32
; SMALL32-NEXT:    lwz 0, 8(1)
; SMALL32-NEXT:    mtlr 0
; SMALL32-NEXT:    blr
;
; LARGE32-LABEL: loadsTGInit:
; LARGE32:       # %bb.0: # %entry
; LARGE32-NEXT:    mflr 0
; LARGE32-NEXT:    stwu 1, -32(1)
; LARGE32-NEXT:    stw 0, 40(1)
; LARGE32-NEXT:    addis 3, L..C2@u(2)
; LARGE32-NEXT:    addis 4, L..C3@u(2)
; LARGE32-NEXT:    lwz 3, L..C2@l(3)
; LARGE32-NEXT:    lwz 4, L..C3@l(4)
; LARGE32-NEXT:    bla .__tls_get_addr[PR]
; LARGE32-NEXT:    lwz 3, 0(3)
; LARGE32-NEXT:    addis 4, L..C8@u(2)
; LARGE32-NEXT:    lwz 4, L..C8@l(4)
; LARGE32-NEXT:    lwz 4, 0(4)
; LARGE32-NEXT:    add 3, 4, 3
; LARGE32-NEXT:    addi 1, 1, 32
; LARGE32-NEXT:    lwz 0, 8(1)
; LARGE32-NEXT:    mtlr 0
; LARGE32-NEXT:    blr
;
; SMALL64-LABEL: loadsTGInit:
; SMALL64:       # %bb.0: # %entry
; SMALL64-NEXT:    mflr 0
; SMALL64-NEXT:    stdu 1, -48(1)
; SMALL64-NEXT:    ld 3, L..C2(2)
; SMALL64-NEXT:    ld 4, L..C3(2)
; SMALL64-NEXT:    std 0, 64(1)
; SMALL64-NEXT:    bla .__tls_get_addr[PR]
; SMALL64-NEXT:    ld 4, L..C8(2)
; SMALL64-NEXT:    lwz 3, 0(3)
; SMALL64-NEXT:    lwz 4, 0(4)
; SMALL64-NEXT:    add 3, 4, 3
; SMALL64-NEXT:    addi 1, 1, 48
; SMALL64-NEXT:    ld 0, 16(1)
; SMALL64-NEXT:    mtlr 0
; SMALL64-NEXT:    blr
;
; LARGE64-LABEL: loadsTGInit:
; LARGE64:       # %bb.0: # %entry
; LARGE64-NEXT:    mflr 0
; LARGE64-NEXT:    stdu 1, -48(1)
; LARGE64-NEXT:    addis 3, L..C2@u(2)
; LARGE64-NEXT:    addis 4, L..C3@u(2)
; LARGE64-NEXT:    std 0, 64(1)
; LARGE64-NEXT:    ld 3, L..C2@l(3)
; LARGE64-NEXT:    ld 4, L..C3@l(4)
; LARGE64-NEXT:    bla .__tls_get_addr[PR]
; LARGE64-NEXT:    addis 4, L..C8@u(2)
; LARGE64-NEXT:    lwz 3, 0(3)
; LARGE64-NEXT:    ld 4, L..C8@l(4)
; LARGE64-NEXT:    lwz 4, 0(4)
; LARGE64-NEXT:    add 3, 4, 3
; LARGE64-NEXT:    addi 1, 1, 48
; LARGE64-NEXT:    ld 0, 16(1)
; LARGE64-NEXT:    mtlr 0
; LARGE64-NEXT:    blr
entry:
  %0 = load i32, ptr @TGInit, align 4
  %1 = load i32, ptr @GInit, align 4
  %add = add nsw i32 %1, %0
  ret i32 %add
}

; Function Attrs: norecurse nounwind readonly willreturn
define i32 @loadsTIUninit() #1 {
; SMALL32-LABEL: loadsTIUninit:
; SMALL32:       # %bb.0: # %entry
; SMALL32-NEXT:    mflr 0
; SMALL32-NEXT:    stwu 1, -32(1)
; SMALL32-NEXT:    lwz 3, L..C4(2)
; SMALL32-NEXT:    stw 0, 40(1)
; SMALL32-NEXT:    bla .__tls_get_mod[PR]
; SMALL32-NEXT:    lwz 4, L..C5(2)
; SMALL32-NEXT:    lwzx 3, 3, 4
; SMALL32-NEXT:    lwz 4, L..C8(2)
; SMALL32-NEXT:    lwz 4, 0(4)
; SMALL32-NEXT:    add 3, 4, 3
; SMALL32-NEXT:    addi 1, 1, 32
; SMALL32-NEXT:    lwz 0, 8(1)
; SMALL32-NEXT:    mtlr 0
; SMALL32-NEXT:    blr
;
; LARGE32-LABEL: loadsTIUninit:
; LARGE32:       # %bb.0: # %entry
; LARGE32-NEXT:    mflr 0
; LARGE32-NEXT:    stwu 1, -32(1)
; LARGE32-NEXT:    stw 0, 40(1)
; LARGE32-NEXT:    addis 6, L..C4@u(2)
; LARGE32-NEXT:    addis 3, L..C5@u(2)
; LARGE32-NEXT:    lwz 3, L..C5@l(3)
; LARGE32-NEXT:    bla .__tls_get_mod[PR]
; LARGE32-NEXT:    lwz 4, L..C4@l(6)
; LARGE32-NEXT:    lwzx 3, 3, 4
; LARGE32-NEXT:    addis 4, L..C8@u(2)
; LARGE32-NEXT:    lwz 4, L..C8@l(4)
; LARGE32-NEXT:    lwz 4, 0(4)
; LARGE32-NEXT:    add 3, 4, 3
; LARGE32-NEXT:    addi 1, 1, 32
; LARGE32-NEXT:    lwz 0, 8(1)
; LARGE32-NEXT:    mtlr 0
; LARGE32-NEXT:    blr
;
; SMALL64-LABEL: loadsTIUninit:
; SMALL64:       # %bb.0: # %entry
; SMALL64-NEXT:    mflr 0
; SMALL64-NEXT:    stdu 1, -48(1)
; SMALL64-NEXT:    ld 3, L..C4(2)
; SMALL64-NEXT:    std 0, 64(1)
; SMALL64-NEXT:    bla .__tls_get_mod[PR]
; SMALL64-NEXT:    ld 4, L..C5(2)
; SMALL64-NEXT:    lwzx 3, 3, 4
; SMALL64-NEXT:    ld 4, L..C8(2)
; SMALL64-NEXT:    lwz 4, 0(4)
; SMALL64-NEXT:    add 3, 4, 3
; SMALL64-NEXT:    addi 1, 1, 48
; SMALL64-NEXT:    ld 0, 16(1)
; SMALL64-NEXT:    mtlr 0
; SMALL64-NEXT:    blr
;
; LARGE64-LABEL: loadsTIUninit:
; LARGE64:       # %bb.0: # %entry
; LARGE64-NEXT:    mflr 0
; LARGE64-NEXT:    stdu 1, -48(1)
; LARGE64-NEXT:    addis 3, L..C4@u(2)
; LARGE64-NEXT:    std 0, 64(1)
; LARGE64-NEXT:    addis 6, L..C5@u(2)
; LARGE64-NEXT:    ld 3, L..C4@l(3)
; LARGE64-NEXT:    bla .__tls_get_mod[PR]
; LARGE64-NEXT:    ld 4, L..C5@l(6)
; LARGE64-NEXT:    addis 5, L..C8@u(2)
; LARGE64-NEXT:    lwzx 3, 3, 4
; LARGE64-NEXT:    ld 4, L..C8@l(5)
; LARGE64-NEXT:    lwz 4, 0(4)
; LARGE64-NEXT:    add 3, 4, 3
; LARGE64-NEXT:    addi 1, 1, 48
; LARGE64-NEXT:    ld 0, 16(1)
; LARGE64-NEXT:    mtlr 0
; LARGE64-NEXT:    blr
entry:
  %0 = load i32, ptr @TIUninit, align 4
  %1 = load i32, ptr @GInit, align 4
  %add = add nsw i32 %1, %0
  ret i32 %add
}

; Function Attrs: norecurse nounwind readonly willreturn
define i32 @loadsTWUninit() #1 {
; SMALL32-LABEL: loadsTWUninit:
; SMALL32:       # %bb.0: # %entry
; SMALL32-NEXT:    mflr 0
; SMALL32-NEXT:    stwu 1, -32(1)
; SMALL32-NEXT:    lwz 3, L..C6(2)
; SMALL32-NEXT:    lwz 4, L..C7(2)
; SMALL32-NEXT:    stw 0, 40(1)
; SMALL32-NEXT:    bla .__tls_get_addr[PR]
; SMALL32-NEXT:    lwz 4, L..C8(2)
; SMALL32-NEXT:    lwz 3, 0(3)
; SMALL32-NEXT:    lwz 4, 0(4)
; SMALL32-NEXT:    add 3, 4, 3
; SMALL32-NEXT:    addi 1, 1, 32
; SMALL32-NEXT:    lwz 0, 8(1)
; SMALL32-NEXT:    mtlr 0
; SMALL32-NEXT:    blr
;
; LARGE32-LABEL: loadsTWUninit:
; LARGE32:       # %bb.0: # %entry
; LARGE32-NEXT:    mflr 0
; LARGE32-NEXT:    stwu 1, -32(1)
; LARGE32-NEXT:    stw 0, 40(1)
; LARGE32-NEXT:    addis 3, L..C6@u(2)
; LARGE32-NEXT:    addis 4, L..C7@u(2)
; LARGE32-NEXT:    lwz 3, L..C6@l(3)
; LARGE32-NEXT:    lwz 4, L..C7@l(4)
; LARGE32-NEXT:    bla .__tls_get_addr[PR]
; LARGE32-NEXT:    lwz 3, 0(3)
; LARGE32-NEXT:    addis 4, L..C8@u(2)
; LARGE32-NEXT:    lwz 4, L..C8@l(4)
; LARGE32-NEXT:    lwz 4, 0(4)
; LARGE32-NEXT:    add 3, 4, 3
; LARGE32-NEXT:    addi 1, 1, 32
; LARGE32-NEXT:    lwz 0, 8(1)
; LARGE32-NEXT:    mtlr 0
; LARGE32-NEXT:    blr
;
; SMALL64-LABEL: loadsTWUninit:
; SMALL64:       # %bb.0: # %entry
; SMALL64-NEXT:    mflr 0
; SMALL64-NEXT:    stdu 1, -48(1)
; SMALL64-NEXT:    ld 3, L..C6(2)
; SMALL64-NEXT:    ld 4, L..C7(2)
; SMALL64-NEXT:    std 0, 64(1)
; SMALL64-NEXT:    bla .__tls_get_addr[PR]
; SMALL64-NEXT:    ld 4, L..C8(2)
; SMALL64-NEXT:    lwz 3, 0(3)
; SMALL64-NEXT:    lwz 4, 0(4)
; SMALL64-NEXT:    add 3, 4, 3
; SMALL64-NEXT:    addi 1, 1, 48
; SMALL64-NEXT:    ld 0, 16(1)
; SMALL64-NEXT:    mtlr 0
; SMALL64-NEXT:    blr
;
; LARGE64-LABEL: loadsTWUninit:
; LARGE64:       # %bb.0: # %entry
; LARGE64-NEXT:    mflr 0
; LARGE64-NEXT:    stdu 1, -48(1)
; LARGE64-NEXT:    addis 3, L..C6@u(2)
; LARGE64-NEXT:    addis 4, L..C7@u(2)
; LARGE64-NEXT:    std 0, 64(1)
; LARGE64-NEXT:    ld 3, L..C6@l(3)
; LARGE64-NEXT:    ld 4, L..C7@l(4)
; LARGE64-NEXT:    bla .__tls_get_addr[PR]
; LARGE64-NEXT:    addis 4, L..C8@u(2)
; LARGE64-NEXT:    lwz 3, 0(3)
; LARGE64-NEXT:    ld 4, L..C8@l(4)
; LARGE64-NEXT:    lwz 4, 0(4)
; LARGE64-NEXT:    add 3, 4, 3
; LARGE64-NEXT:    addi 1, 1, 48
; LARGE64-NEXT:    ld 0, 16(1)
; LARGE64-NEXT:    mtlr 0
; LARGE64-NEXT:    blr
entry:
  %0 = load i32, ptr @TWUninit, align 4
  %1 = load i32, ptr @GInit, align 4
  %add = add nsw i32 %1, %0
  ret i32 %add
}

; External symbol reference checks for .__tls_get_addr/.__tls_get_mod

; SMALL32: .extern .__tls_get_addr[PR]
; SMALL32: .extern .__tls_get_mod[PR]
; SMALL64: .extern .__tls_get_addr[PR]
; SMALL64: .extern .__tls_get_mod[PR]
; LARGE32: .extern .__tls_get_addr[PR]
; LARGE32: .extern .__tls_get_mod[PR]
; LARGE64: .extern .__tls_get_addr[PR]
; LARGE64: .extern .__tls_get_mod[PR]

; TOC entry checks

; SMALL32-LABEL: .toc
; SMALL32-LABEL: L..C0:
; SMALL32-NEXT:	 .tc .TGUninit[TC],TGUninit[TL]@m
; SMALL32-LABEL: L..C1:
; SMALL32-NEXT:	 .tc TGUninit[TC],TGUninit[TL]@gd
; SMALL32-LABEL: L..C2:
; SMALL32-NEXT:	 .tc .TGInit[TC],TGInit[TL]@m
; SMALL32-LABEL: L..C3:
; SMALL32-NEXT:	 .tc TGInit[TC],TGInit[TL]@gd
; SMALL32-LABEL: L..C4:
; SMALL32-NEXT:	 .tc _Renamed..5f24__TLSML[TC],_Renamed..5f24__TLSML[TC]@ml
; SMALL32-NEXT:	 .rename _Renamed..5f24__TLSML[TC],"_$TLSML"
; SMALL32-LABEL: L..C5:
; SMALL32-NEXT:	 .tc TIUninit[TC],TIUninit[UL]@ld
; SMALL32-LABEL: L..C6:
; SMALL32-NEXT:	 .tc .TWUninit[TC],TWUninit[TL]@m
; SMALL32-LABEL: L..C7:
; SMALL32-NEXT:	 .tc TWUninit[TC],TWUninit[TL]@gd
; SMALL32-LABEL: L..C8:
; SMALL32-NEXT:	 .tc GInit[TC],GInit[RW]

; LARGE32-LABEL: .toc
; LARGE32-LABEL: L..C0:
; LARGE32-NEXT:  .tc .TGUninit[TE],TGUninit[TL]@m
; LARGE32-LABEL: L..C1:
; LARGE32-NEXT:  .tc TGUninit[TE],TGUninit[TL]@gd
; LARGE32-LABEL: L..C2:
; LARGE32-NEXT:  .tc .TGInit[TE],TGInit[TL]@m
; LARGE32-LABEL: L..C3:
; LARGE32-NEXT:  .tc TGInit[TE],TGInit[TL]@gd
; LARGE32-LABEL: L..C4:
; LARGE32-NEXT:  .tc TIUninit[TE],TIUninit[UL]@ld
; LARGE32-LABEL: L..C5:
; LARGE32-NEXT:  .tc _Renamed..5f24__TLSML[TC],_Renamed..5f24__TLSML[TC]@ml
; LARGE32-NEXT:  .rename _Renamed..5f24__TLSML[TC],"_$TLSML"
; LARGE32-LABEL: L..C6:
; LARGE32-NEXT:  .tc .TWUninit[TE],TWUninit[TL]@m
; LARGE32-LABEL: L..C7:
; LARGE32-NEXT:  .tc TWUninit[TE],TWUninit[TL]@gd
; LARGE32-LABEL: L..C8:
; LARGE32-NEXT:  .tc GInit[TE],GInit[RW]

; SMALL64-LABEL:  .toc
; SMALL64-LABEL:  L..C0:
; SMALL64-NEXT:   .tc .TGUninit[TC],TGUninit[TL]@m
; SMALL64-LABEL:  L..C1:
; SMALL64-NEXT:   .tc TGUninit[TC],TGUninit[TL]@gd
; SMALL64-LABEL:  L..C2:
; SMALL64-NEXT:   .tc .TGInit[TC],TGInit[TL]@m
; SMALL64-LABEL:  L..C3:
; SMALL64-NEXT:   .tc TGInit[TC],TGInit[TL]@gd
; SMALL64-LABEL:  L..C4:
; SMALL64-NEXT:   .tc _Renamed..5f24__TLSML[TC],_Renamed..5f24__TLSML[TC]@ml
; SMALL64-NEXT:   .rename _Renamed..5f24__TLSML[TC],"_$TLSML"
; SMALL64-LABEL:  L..C5:
; SMALL64-NEXT:   .tc TIUninit[TC],TIUninit[UL]@ld
; SMALL64-LABEL:  L..C6:
; SMALL64-NEXT:   .tc .TWUninit[TC],TWUninit[TL]@m
; SMALL64-LABEL:  L..C7:
; SMALL64-NEXT:   .tc TWUninit[TC],TWUninit[TL]@gd
; SMALL64-LABEL:  L..C8:
; SMALL64-NEXT:   .tc GInit[TC],GInit[RW]

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
; LARGE64-NEXT:  .tc _Renamed..5f24__TLSML[TC],_Renamed..5f24__TLSML[TC]@ml
; LARGE64-NEXT:  .rename _Renamed..5f24__TLSML[TC],"_$TLSML"
; LARGE64-LABEL:  L..C5:
; LARGE64-NEXT:  .tc TIUninit[TE],TIUninit[UL]@ld
; LARGE64-LABEL:  L..C6:
; LARGE64-NEXT:  .tc .TWUninit[TE],TWUninit[TL]@m
; LARGE64-LABEL:  L..C7:
; LARGE64-NEXT:  .tc TWUninit[TE],TWUninit[TL]@gd
; LARGE64-LABEL:  L..C8:
; LARGE64-NEXT:  .tc GInit[TE],GInit[RW]

attributes #0 = { nofree norecurse nounwind willreturn writeonly "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pwr4" "target-features"="-altivec,-bpermd,-crypto,-direct-move,-extdiv,-float128,-htm,-mma,-paired-vector-memops,-power10-vector,-power8-vector,-power9-vector,-spe,-vsx" }
attributes #1 = { norecurse nounwind readonly willreturn "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pwr4" "target-features"="-altivec,-bpermd,-crypto,-direct-move,-extdiv,-float128,-htm,-mma,-paired-vector-memops,-power10-vector,-power8-vector,-power9-vector,-spe,-vsx" }
