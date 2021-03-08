; RUN: llc  -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec \
; RUN:      -mtriple powerpc-ibm-aix-xcoff < %s | FileCheck %s \
; RUN:      --check-prefix=SMALL32
; RUN: llc  -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec \
; RUN:      -mtriple powerpc-ibm-aix-xcoff --code-model=large < %s \
; RUN:      | FileCheck %s --check-prefix=LARGE32

@TGInit = thread_local global i64 1, align 8
@TWInit = weak thread_local global i64 1, align 8
@GInit = global i64 1, align 8
@TIUninit = internal thread_local global i64 0, align 8
@TIInit = internal thread_local global i64 1, align 8

; Function Attrs: nofree norecurse nounwind willreturn writeonly
define void @storesTGInit(i64 %Val) #0 {
; SMALL32-LABEL: storesTGInit:
; SMALL32:       # %bb.0: # %entry
; SMALL32-NEXT:    mflr 0
; SMALL32-NEXT:    stw 0, 8(1)
; SMALL32-NEXT:    stwu 1, -32(1)
; SMALL32-NEXT:    mr 6, 4
; SMALL32-NEXT:    mr 7, 3
; SMALL32-NEXT:    lwz 3, L..C0(2)
; SMALL32-NEXT:    lwz 4, L..C1(2)
; SMALL32-NEXT:    bla .__tls_get_addr
; SMALL32-NEXT:    stw 6, 4(3)
; SMALL32-NEXT:    stw 7, 0(3)
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
; LARGE32-NEXT:    mr 6, 4
; LARGE32-NEXT:    mr 7, 3
; LARGE32-NEXT:    addis 3, L..C0@u(2)
; LARGE32-NEXT:    addis 4, L..C1@u(2)
; LARGE32-NEXT:    lwz 3, L..C0@l(3)
; LARGE32-NEXT:    lwz 4, L..C1@l(4)
; LARGE32-NEXT:    bla .__tls_get_addr
; LARGE32-NEXT:    stw 6, 4(3)
; LARGE32-NEXT:    stw 7, 0(3)
; LARGE32-NEXT:    addi 1, 1, 32
; LARGE32-NEXT:    lwz 0, 8(1)
; LARGE32-NEXT:    mtlr 0
; LARGE32-NEXT:    blr
entry:
  store i64 %Val, i64* @TGInit, align 8
  ret void
}

; Function Attrs: nofree norecurse nounwind willreturn writeonly
define void @storesTIUninit(i64 %Val) #0 {
; SMALL32-LABEL: storesTIUninit:
; SMALL32:       # %bb.0: # %entry
; SMALL32-NEXT:    mflr 0
; SMALL32-NEXT:    stw 0, 8(1)
; SMALL32-NEXT:    stwu 1, -32(1)
; SMALL32-NEXT:    mr 6, 4
; SMALL32-NEXT:    mr 7, 3
; SMALL32-NEXT:    lwz 3, L..C2(2)
; SMALL32-NEXT:    lwz 4, L..C3(2)
; SMALL32-NEXT:    bla .__tls_get_addr
; SMALL32-NEXT:    stw 6, 4(3)
; SMALL32-NEXT:    stw 7, 0(3)
; SMALL32-NEXT:    addi 1, 1, 32
; SMALL32-NEXT:    lwz 0, 8(1)
; SMALL32-NEXT:    mtlr 0
; SMALL32-NEXT:    blr
;
; LARGE32-LABEL: storesTIUninit:
; LARGE32:       # %bb.0: # %entry
; LARGE32-NEXT:    mflr 0
; LARGE32-NEXT:    stw 0, 8(1)
; LARGE32-NEXT:    stwu 1, -32(1)
; LARGE32-NEXT:    mr 6, 4
; LARGE32-NEXT:    mr 7, 3
; LARGE32-NEXT:    addis 3, L..C2@u(2)
; LARGE32-NEXT:    addis 4, L..C3@u(2)
; LARGE32-NEXT:    lwz 3, L..C2@l(3)
; LARGE32-NEXT:    lwz 4, L..C3@l(4)
; LARGE32-NEXT:    bla .__tls_get_addr
; LARGE32-NEXT:    stw 6, 4(3)
; LARGE32-NEXT:    stw 7, 0(3)
; LARGE32-NEXT:    addi 1, 1, 32
; LARGE32-NEXT:    lwz 0, 8(1)
; LARGE32-NEXT:    mtlr 0
; LARGE32-NEXT:    blr
entry:
  store i64 %Val, i64* @TIUninit, align 8
  ret void
}

; Function Attrs: nofree norecurse nounwind willreturn writeonly
define void @storesTIInit(i64 %Val) #0 {
; SMALL32-LABEL: storesTIInit:
; SMALL32:       # %bb.0: # %entry
; SMALL32-NEXT:    mflr 0
; SMALL32-NEXT:    stw 0, 8(1)
; SMALL32-NEXT:    stwu 1, -32(1)
; SMALL32-NEXT:    mr 6, 4
; SMALL32-NEXT:    mr 7, 3
; SMALL32-NEXT:    lwz 3, L..C4(2)
; SMALL32-NEXT:    lwz 4, L..C5(2)
; SMALL32-NEXT:    bla .__tls_get_addr
; SMALL32-NEXT:    stw 6, 4(3)
; SMALL32-NEXT:    stw 7, 0(3)
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
; LARGE32-NEXT:    mr 6, 4
; LARGE32-NEXT:    mr 7, 3
; LARGE32-NEXT:    addis 3, L..C4@u(2)
; LARGE32-NEXT:    addis 4, L..C5@u(2)
; LARGE32-NEXT:    lwz 3, L..C4@l(3)
; LARGE32-NEXT:    lwz 4, L..C5@l(4)
; LARGE32-NEXT:    bla .__tls_get_addr
; LARGE32-NEXT:    stw 6, 4(3)
; LARGE32-NEXT:    stw 7, 0(3)
; LARGE32-NEXT:    addi 1, 1, 32
; LARGE32-NEXT:    lwz 0, 8(1)
; LARGE32-NEXT:    mtlr 0
; LARGE32-NEXT:    blr
entry:
  store i64 %Val, i64* @TIInit, align 8
  ret void
}

; Function Attrs: nofree norecurse nounwind willreturn writeonly
define void @storesTWInit(i64 %Val) #0 {
; SMALL32-LABEL: storesTWInit:
; SMALL32:       # %bb.0: # %entry
; SMALL32-NEXT:    mflr 0
; SMALL32-NEXT:    stw 0, 8(1)
; SMALL32-NEXT:    stwu 1, -32(1)
; SMALL32-NEXT:    mr 6, 4
; SMALL32-NEXT:    mr 7, 3
; SMALL32-NEXT:    lwz 3, L..C6(2)
; SMALL32-NEXT:    lwz 4, L..C7(2)
; SMALL32-NEXT:    bla .__tls_get_addr
; SMALL32-NEXT:    stw 6, 4(3)
; SMALL32-NEXT:    stw 7, 0(3)
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
; LARGE32-NEXT:    mr 6, 4
; LARGE32-NEXT:    mr 7, 3
; LARGE32-NEXT:    addis 3, L..C6@u(2)
; LARGE32-NEXT:    addis 4, L..C7@u(2)
; LARGE32-NEXT:    lwz 3, L..C6@l(3)
; LARGE32-NEXT:    lwz 4, L..C7@l(4)
; LARGE32-NEXT:    bla .__tls_get_addr
; LARGE32-NEXT:    stw 6, 4(3)
; LARGE32-NEXT:    stw 7, 0(3)
; LARGE32-NEXT:    addi 1, 1, 32
; LARGE32-NEXT:    lwz 0, 8(1)
; LARGE32-NEXT:    mtlr 0
; LARGE32-NEXT:    blr
entry:
  store i64 %Val, i64* @TWInit, align 8
  ret void
}

; Function Attrs: norecurse nounwind readonly willreturn
define i64 @loadsTGInit() #1 {
; SMALL32-LABEL: loadsTGInit:
; SMALL32:       # %bb.0: # %entry
; SMALL32-NEXT:    mflr 0
; SMALL32-NEXT:    stw 0, 8(1)
; SMALL32-NEXT:    stwu 1, -32(1)
; SMALL32-NEXT:    lwz 3, L..C0(2)
; SMALL32-NEXT:    lwz 4, L..C1(2)
; SMALL32-NEXT:    bla .__tls_get_addr
; SMALL32-NEXT:    lwz 4, L..C8(2)
; SMALL32-NEXT:    lwz 5, 4(3)
; SMALL32-NEXT:    lwz 6, 4(4)
; SMALL32-NEXT:    lwz 3, 0(3)
; SMALL32-NEXT:    lwz 7, 0(4)
; SMALL32-NEXT:    addc 4, 6, 5
; SMALL32-NEXT:    adde 3, 7, 3
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
; LARGE32-NEXT:    addis 3, L..C0@u(2)
; LARGE32-NEXT:    addis 4, L..C1@u(2)
; LARGE32-NEXT:    lwz 3, L..C0@l(3)
; LARGE32-NEXT:    lwz 4, L..C1@l(4)
; LARGE32-NEXT:    bla .__tls_get_addr
; LARGE32-NEXT:    lwz 4, 4(3)
; LARGE32-NEXT:    lwz 3, 0(3)
; LARGE32-NEXT:    addis 5, L..C8@u(2)
; LARGE32-NEXT:    lwz 5, L..C8@l(5)
; LARGE32-NEXT:    lwz 6, 4(5)
; LARGE32-NEXT:    lwz 5, 0(5)
; LARGE32-NEXT:    addc 4, 6, 4
; LARGE32-NEXT:    adde 3, 5, 3
; LARGE32-NEXT:    addi 1, 1, 32
; LARGE32-NEXT:    lwz 0, 8(1)
; LARGE32-NEXT:    mtlr 0
; LARGE32-NEXT:    blr
entry:
  %0 = load i64, i64* @TGInit, align 8
  %1 = load i64, i64* @GInit, align 8
  %add = add nsw i64 %1, %0
  ret i64 %add
}

; Function Attrs: norecurse nounwind readonly willreturn
define i64 @loadsTIUninit() #1 {
; SMALL32-LABEL: loadsTIUninit:
; SMALL32:       # %bb.0: # %entry
; SMALL32-NEXT:    mflr 0
; SMALL32-NEXT:    stw 0, 8(1)
; SMALL32-NEXT:    stwu 1, -32(1)
; SMALL32-NEXT:    lwz 3, L..C2(2)
; SMALL32-NEXT:    lwz 4, L..C3(2)
; SMALL32-NEXT:    bla .__tls_get_addr
; SMALL32-NEXT:    lwz 4, L..C8(2)
; SMALL32-NEXT:    lwz 5, 4(3)
; SMALL32-NEXT:    lwz 6, 4(4)
; SMALL32-NEXT:    lwz 3, 0(3)
; SMALL32-NEXT:    lwz 7, 0(4)
; SMALL32-NEXT:    addc 4, 6, 5
; SMALL32-NEXT:    adde 3, 7, 3
; SMALL32-NEXT:    addi 1, 1, 32
; SMALL32-NEXT:    lwz 0, 8(1)
; SMALL32-NEXT:    mtlr 0
; SMALL32-NEXT:    blr
;
; LARGE32-LABEL: loadsTIUninit:
; LARGE32:       # %bb.0: # %entry
; LARGE32-NEXT:    mflr 0
; LARGE32-NEXT:    stw 0, 8(1)
; LARGE32-NEXT:    stwu 1, -32(1)
; LARGE32-NEXT:    addis 3, L..C2@u(2)
; LARGE32-NEXT:    addis 4, L..C3@u(2)
; LARGE32-NEXT:    lwz 3, L..C2@l(3)
; LARGE32-NEXT:    lwz 4, L..C3@l(4)
; LARGE32-NEXT:    bla .__tls_get_addr
; LARGE32-NEXT:    lwz 4, 4(3)
; LARGE32-NEXT:    lwz 3, 0(3)
; LARGE32-NEXT:    addis 5, L..C8@u(2)
; LARGE32-NEXT:    lwz 5, L..C8@l(5)
; LARGE32-NEXT:    lwz 6, 4(5)
; LARGE32-NEXT:    lwz 5, 0(5)
; LARGE32-NEXT:    addc 4, 6, 4
; LARGE32-NEXT:    adde 3, 5, 3
; LARGE32-NEXT:    addi 1, 1, 32
; LARGE32-NEXT:    lwz 0, 8(1)
; LARGE32-NEXT:    mtlr 0
; LARGE32-NEXT:    blr
entry:
  %0 = load i64, i64* @TIUninit, align 8
  %1 = load i64, i64* @GInit, align 8
  %add = add nsw i64 %1, %0
  ret i64 %add
}

; Function Attrs: norecurse nounwind readonly willreturn
define i64 @loadsTIInit() #1 {
; SMALL32-LABEL: loadsTIInit:
; SMALL32:       # %bb.0: # %entry
; SMALL32-NEXT:    mflr 0
; SMALL32-NEXT:    stw 0, 8(1)
; SMALL32-NEXT:    stwu 1, -32(1)
; SMALL32-NEXT:    lwz 3, L..C4(2)
; SMALL32-NEXT:    lwz 4, L..C5(2)
; SMALL32-NEXT:    bla .__tls_get_addr
; SMALL32-NEXT:    lwz 4, L..C8(2)
; SMALL32-NEXT:    lwz 5, 4(3)
; SMALL32-NEXT:    lwz 6, 4(4)
; SMALL32-NEXT:    lwz 3, 0(3)
; SMALL32-NEXT:    lwz 7, 0(4)
; SMALL32-NEXT:    addc 4, 6, 5
; SMALL32-NEXT:    adde 3, 7, 3
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
; LARGE32-NEXT:    bla .__tls_get_addr
; LARGE32-NEXT:    lwz 4, 4(3)
; LARGE32-NEXT:    lwz 3, 0(3)
; LARGE32-NEXT:    addis 5, L..C8@u(2)
; LARGE32-NEXT:    lwz 5, L..C8@l(5)
; LARGE32-NEXT:    lwz 6, 4(5)
; LARGE32-NEXT:    lwz 5, 0(5)
; LARGE32-NEXT:    addc 4, 6, 4
; LARGE32-NEXT:    adde 3, 5, 3
; LARGE32-NEXT:    addi 1, 1, 32
; LARGE32-NEXT:    lwz 0, 8(1)
; LARGE32-NEXT:    mtlr 0
; LARGE32-NEXT:    blr
entry:
  %0 = load i64, i64* @TIInit, align 8
  %1 = load i64, i64* @GInit, align 8
  %add = add nsw i64 %1, %0
  ret i64 %add
}

; Function Attrs: norecurse nounwind readonly willreturn
define i64 @loadsTWInit() #1 {
; SMALL32-LABEL: loadsTWInit:
; SMALL32:       # %bb.0: # %entry
; SMALL32-NEXT:    mflr 0
; SMALL32-NEXT:    stw 0, 8(1)
; SMALL32-NEXT:    stwu 1, -32(1)
; SMALL32-NEXT:    lwz 3, L..C6(2)
; SMALL32-NEXT:    lwz 4, L..C7(2)
; SMALL32-NEXT:    bla .__tls_get_addr
; SMALL32-NEXT:    lwz 4, L..C8(2)
; SMALL32-NEXT:    lwz 5, 4(3)
; SMALL32-NEXT:    lwz 6, 4(4)
; SMALL32-NEXT:    lwz 3, 0(3)
; SMALL32-NEXT:    lwz 7, 0(4)
; SMALL32-NEXT:    addc 4, 6, 5
; SMALL32-NEXT:    adde 3, 7, 3
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
; LARGE32-NEXT:    bla .__tls_get_addr
; LARGE32-NEXT:    lwz 4, 4(3)
; LARGE32-NEXT:    lwz 3, 0(3)
; LARGE32-NEXT:    addis 5, L..C8@u(2)
; LARGE32-NEXT:    lwz 5, L..C8@l(5)
; LARGE32-NEXT:    lwz 6, 4(5)
; LARGE32-NEXT:    lwz 5, 0(5)
; LARGE32-NEXT:    addc 4, 6, 4
; LARGE32-NEXT:    adde 3, 5, 3
; LARGE32-NEXT:    addi 1, 1, 32
; LARGE32-NEXT:    lwz 0, 8(1)
; LARGE32-NEXT:    mtlr 0
; LARGE32-NEXT:    blr
entry:
  %0 = load i64, i64* @TWInit, align 8
  %1 = load i64, i64* @GInit, align 8
  %add = add nsw i64 %1, %0
  ret i64 %add
}

; TOC entry checks

; SMALL32-LABEL:	.toc
; SMALL32-LABEL:  L..C0:
; SMALL32-NEXT:  .tc .TGInit[TC],TGInit[TL]@m
; SMALL32-LABEL:  L..C1:
; SMALL32-NEXT:  .tc TGInit[TC],TGInit[TL]
; SMALL32-LABEL:  L..C2:
; SMALL32-NEXT:  .tc .TIUninit[TC],TIUninit[UL]@m
; SMALL32-LABEL:  L..C3:
; SMALL32-NEXT:  .tc TIUninit[TC],TIUninit[UL]
; SMALL32-LABEL:  L..C4:
; SMALL32-NEXT:  .tc .TIInit[TC],TIInit[TL]@m
; SMALL32-LABEL:  L..C5:
; SMALL32-NEXT:  .tc TIInit[TC],TIInit[TL]
; SMALL32-LABEL:  L..C6:
; SMALL32-NEXT:  .tc .TWInit[TC],TWInit[TL]@m
; SMALL32-LABEL:  L..C7:
; SMALL32-NEXT:  .tc TWInit[TC],TWInit[TL]
; SMALL32-LABEL:  L..C8:
; SMALL32-NEXT:  .tc GInit[TC],GInit[RW]

; LARGE32-LABEL:  .toc
; LARGE32-LABEL:  L..C0:
; LARGE32-NEXT:  .tc .TGInit[TE],TGInit[TL]@m
; LARGE32-LABEL:  L..C1:
; LARGE32-NEXT:  .tc TGInit[TE],TGInit[TL]
; LARGE32-LABEL:  L..C2:
; LARGE32-NEXT:  .tc .TIUninit[TE],TIUninit[UL]@m
; LARGE32-LABEL:  L..C3:
; LARGE32-NEXT:  .tc TIUninit[TE],TIUninit[UL]
; LARGE32-LABEL:  L..C4:
; LARGE32-NEXT:  .tc .TIInit[TE],TIInit[TL]@m
; LARGE32-LABEL:  L..C5:
; LARGE32-NEXT:  .tc TIInit[TE],TIInit[TL]
; LARGE32-LABEL:  L..C6:
; LARGE32-NEXT:  .tc .TWInit[TE],TWInit[TL]@m
; LARGE32-LABEL:  L..C7:
; LARGE32-NEXT:  .tc TWInit[TE],TWInit[TL]
; LARGE32-LABEL:  L..C8:
; LARGE32-NEXT:  .tc GInit[TE],GInit[RW]
attributes #0 = { nofree norecurse nounwind willreturn writeonly "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pwr4" "target-features"="-altivec,-bpermd,-crypto,-direct-move,-extdiv,-float128,-htm,-mma,-paired-vector-memops,-power10-vector,-power8-vector,-power9-vector,-rop-protection,-spe,-vsx" }
attributes #1 = { norecurse nounwind readonly willreturn "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pwr4" "target-features"="-altivec,-bpermd,-crypto,-direct-move,-extdiv,-float128,-htm,-mma,-paired-vector-memops,-power10-vector,-power8-vector,-power9-vector,-rop-protection,-spe,-vsx" }
