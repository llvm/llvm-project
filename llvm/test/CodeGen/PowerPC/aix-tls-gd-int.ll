; RUN: llc  -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec \
; RUN:      -mtriple powerpc-ibm-aix-xcoff < %s | FileCheck %s \
; RUN:      --check-prefix=SMALL32
; RUN: llc  -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec \
; RUN:      -mtriple powerpc-ibm-aix-xcoff --code-model=large < %s \
; RUN:      | FileCheck %s --check-prefix=LARGE32

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
; SMALL32-NEXT:    stw 0, 8(1)
; SMALL32-NEXT:    stwu 1, -32(1)
; SMALL32-NEXT:    mr 6, 3
; SMALL32-NEXT:    lwz 3, L..C0(2)
; SMALL32-NEXT:    lwz 4, L..C1(2)
; SMALL32-NEXT:    bla .__tls_get_addr
; SMALL32-NEXT:    stw 6, 0(3)
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
; LARGE32-NEXT:    mr 6, 3
; LARGE32-NEXT:    addis 3, L..C0@u(2)
; LARGE32-NEXT:    addis 4, L..C1@u(2)
; LARGE32-NEXT:    lwz 3, L..C0@l(3)
; LARGE32-NEXT:    lwz 4, L..C1@l(4)
; LARGE32-NEXT:    bla .__tls_get_addr
; LARGE32-NEXT:    stw 6, 0(3)
; LARGE32-NEXT:    addi 1, 1, 32
; LARGE32-NEXT:    lwz 0, 8(1)
; LARGE32-NEXT:    mtlr 0
; LARGE32-NEXT:    blr
entry:
  store i32 %Val, i32* @TGUninit, align 4
  ret void
}

; Function Attrs: nofree norecurse nounwind willreturn writeonly
define void @storesTGInit(i32 %Val) #0 {
; SMALL32-LABEL: storesTGInit:
; SMALL32:       # %bb.0: # %entry
; SMALL32-NEXT:    mflr 0
; SMALL32-NEXT:    stw 0, 8(1)
; SMALL32-NEXT:    stwu 1, -32(1)
; SMALL32-NEXT:    mr 6, 3
; SMALL32-NEXT:    lwz 3, L..C2(2)
; SMALL32-NEXT:    lwz 4, L..C3(2)
; SMALL32-NEXT:    bla .__tls_get_addr
; SMALL32-NEXT:    stw 6, 0(3)
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
; LARGE32-NEXT:    mr 6, 3
; LARGE32-NEXT:    addis 3, L..C2@u(2)
; LARGE32-NEXT:    addis 4, L..C3@u(2)
; LARGE32-NEXT:    lwz 3, L..C2@l(3)
; LARGE32-NEXT:    lwz 4, L..C3@l(4)
; LARGE32-NEXT:    bla .__tls_get_addr
; LARGE32-NEXT:    stw 6, 0(3)
; LARGE32-NEXT:    addi 1, 1, 32
; LARGE32-NEXT:    lwz 0, 8(1)
; LARGE32-NEXT:    mtlr 0
; LARGE32-NEXT:    blr
entry:
  store i32 %Val, i32* @TGInit, align 4
  ret void
}

; Function Attrs: nofree norecurse nounwind willreturn writeonly
define void @storesTIUninit(i32 %Val) #0 {
; SMALL32-LABEL: storesTIUninit:
; SMALL32:       # %bb.0: # %entry
; SMALL32-NEXT:    mflr 0
; SMALL32-NEXT:    stw 0, 8(1)
; SMALL32-NEXT:    stwu 1, -32(1)
; SMALL32-NEXT:    mr 6, 3
; SMALL32-NEXT:    lwz 3, L..C4(2)
; SMALL32-NEXT:    lwz 4, L..C5(2)
; SMALL32-NEXT:    bla .__tls_get_addr
; SMALL32-NEXT:    stw 6, 0(3)
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
; LARGE32-NEXT:    mr 6, 3
; LARGE32-NEXT:    addis 3, L..C4@u(2)
; LARGE32-NEXT:    addis 4, L..C5@u(2)
; LARGE32-NEXT:    lwz 3, L..C4@l(3)
; LARGE32-NEXT:    lwz 4, L..C5@l(4)
; LARGE32-NEXT:    bla .__tls_get_addr
; LARGE32-NEXT:    stw 6, 0(3)
; LARGE32-NEXT:    addi 1, 1, 32
; LARGE32-NEXT:    lwz 0, 8(1)
; LARGE32-NEXT:    mtlr 0
; LARGE32-NEXT:    blr
entry:
  store i32 %Val, i32* @TIUninit, align 4
  ret void
}

; Function Attrs: nofree norecurse nounwind willreturn writeonly
define void @storesTWUninit(i32 %Val) #0 {
; SMALL32-LABEL: storesTWUninit:
; SMALL32:       # %bb.0: # %entry
; SMALL32-NEXT:    mflr 0
; SMALL32-NEXT:    stw 0, 8(1)
; SMALL32-NEXT:    stwu 1, -32(1)
; SMALL32-NEXT:    mr 6, 3
; SMALL32-NEXT:    lwz 3, L..C6(2)
; SMALL32-NEXT:    lwz 4, L..C7(2)
; SMALL32-NEXT:    bla .__tls_get_addr
; SMALL32-NEXT:    stw 6, 0(3)
; SMALL32-NEXT:    addi 1, 1, 32
; SMALL32-NEXT:    lwz 0, 8(1)
; SMALL32-NEXT:    mtlr 0
; SMALL32-NEXT:    blr
;
; LARGE32-LABEL: storesTWUninit:
; LARGE32:       # %bb.0: # %entry
; LARGE32-NEXT:    mflr 0
; LARGE32-NEXT:    stw 0, 8(1)
; LARGE32-NEXT:    stwu 1, -32(1)
; LARGE32-NEXT:    mr 6, 3
; LARGE32-NEXT:    addis 3, L..C6@u(2)
; LARGE32-NEXT:    addis 4, L..C7@u(2)
; LARGE32-NEXT:    lwz 3, L..C6@l(3)
; LARGE32-NEXT:    lwz 4, L..C7@l(4)
; LARGE32-NEXT:    bla .__tls_get_addr
; LARGE32-NEXT:    stw 6, 0(3)
; LARGE32-NEXT:    addi 1, 1, 32
; LARGE32-NEXT:    lwz 0, 8(1)
; LARGE32-NEXT:    mtlr 0
; LARGE32-NEXT:    blr
entry:
  store i32 %Val, i32* @TWUninit, align 4
  ret void
}

; Function Attrs: norecurse nounwind readonly willreturn
define i32 @loadsTGUninit() #1 {
; SMALL32-LABEL: loadsTGUninit:
; SMALL32:       # %bb.0: # %entry
; SMALL32-NEXT:    mflr 0
; SMALL32-NEXT:    stw 0, 8(1)
; SMALL32-NEXT:    stwu 1, -32(1)
; SMALL32-NEXT:    lwz 3, L..C0(2)
; SMALL32-NEXT:    lwz 4, L..C1(2)
; SMALL32-NEXT:    bla .__tls_get_addr
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
; LARGE32-NEXT:    stw 0, 8(1)
; LARGE32-NEXT:    stwu 1, -32(1)
; LARGE32-NEXT:    addis 3, L..C0@u(2)
; LARGE32-NEXT:    addis 4, L..C1@u(2)
; LARGE32-NEXT:    lwz 3, L..C0@l(3)
; LARGE32-NEXT:    lwz 4, L..C1@l(4)
; LARGE32-NEXT:    bla .__tls_get_addr
; LARGE32-NEXT:    lwz 3, 0(3)
; LARGE32-NEXT:    addis 4, L..C8@u(2)
; LARGE32-NEXT:    lwz 4, L..C8@l(4)
; LARGE32-NEXT:    lwz 4, 0(4)
; LARGE32-NEXT:    add 3, 4, 3
; LARGE32-NEXT:    addi 1, 1, 32
; LARGE32-NEXT:    lwz 0, 8(1)
; LARGE32-NEXT:    mtlr 0
; LARGE32-NEXT:    blr
entry:
  %0 = load i32, i32* @TGUninit, align 4
  %1 = load i32, i32* @GInit, align 4
  %add = add nsw i32 %1, %0
  ret i32 %add
}

; Function Attrs: norecurse nounwind readonly willreturn
define i32 @loadsTGInit() #1 {
; SMALL32-LABEL: loadsTGInit:
; SMALL32:       # %bb.0: # %entry
; SMALL32-NEXT:    mflr 0
; SMALL32-NEXT:    stw 0, 8(1)
; SMALL32-NEXT:    stwu 1, -32(1)
; SMALL32-NEXT:    lwz 3, L..C2(2)
; SMALL32-NEXT:    lwz 4, L..C3(2)
; SMALL32-NEXT:    bla .__tls_get_addr
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
; LARGE32-NEXT:    stw 0, 8(1)
; LARGE32-NEXT:    stwu 1, -32(1)
; LARGE32-NEXT:    addis 3, L..C2@u(2)
; LARGE32-NEXT:    addis 4, L..C3@u(2)
; LARGE32-NEXT:    lwz 3, L..C2@l(3)
; LARGE32-NEXT:    lwz 4, L..C3@l(4)
; LARGE32-NEXT:    bla .__tls_get_addr
; LARGE32-NEXT:    lwz 3, 0(3)
; LARGE32-NEXT:    addis 4, L..C8@u(2)
; LARGE32-NEXT:    lwz 4, L..C8@l(4)
; LARGE32-NEXT:    lwz 4, 0(4)
; LARGE32-NEXT:    add 3, 4, 3
; LARGE32-NEXT:    addi 1, 1, 32
; LARGE32-NEXT:    lwz 0, 8(1)
; LARGE32-NEXT:    mtlr 0
; LARGE32-NEXT:    blr
entry:
  %0 = load i32, i32* @TGInit, align 4
  %1 = load i32, i32* @GInit, align 4
  %add = add nsw i32 %1, %0
  ret i32 %add
}

; Function Attrs: norecurse nounwind readonly willreturn
define i32 @loadsTIUninit() #1 {
; SMALL32-LABEL: loadsTIUninit:
; SMALL32:       # %bb.0: # %entry
; SMALL32-NEXT:    mflr 0
; SMALL32-NEXT:    stw 0, 8(1)
; SMALL32-NEXT:    stwu 1, -32(1)
; SMALL32-NEXT:    lwz 3, L..C4(2)
; SMALL32-NEXT:    lwz 4, L..C5(2)
; SMALL32-NEXT:    bla .__tls_get_addr
; SMALL32-NEXT:    lwz 4, L..C8(2)
; SMALL32-NEXT:    lwz 3, 0(3)
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
; LARGE32-NEXT:    stw 0, 8(1)
; LARGE32-NEXT:    stwu 1, -32(1)
; LARGE32-NEXT:    addis 3, L..C4@u(2)
; LARGE32-NEXT:    addis 4, L..C5@u(2)
; LARGE32-NEXT:    lwz 3, L..C4@l(3)
; LARGE32-NEXT:    lwz 4, L..C5@l(4)
; LARGE32-NEXT:    bla .__tls_get_addr
; LARGE32-NEXT:    lwz 3, 0(3)
; LARGE32-NEXT:    addis 4, L..C8@u(2)
; LARGE32-NEXT:    lwz 4, L..C8@l(4)
; LARGE32-NEXT:    lwz 4, 0(4)
; LARGE32-NEXT:    add 3, 4, 3
; LARGE32-NEXT:    addi 1, 1, 32
; LARGE32-NEXT:    lwz 0, 8(1)
; LARGE32-NEXT:    mtlr 0
; LARGE32-NEXT:    blr
entry:
  %0 = load i32, i32* @TIUninit, align 4
  %1 = load i32, i32* @GInit, align 4
  %add = add nsw i32 %1, %0
  ret i32 %add
}

; Function Attrs: norecurse nounwind readonly willreturn
define i32 @loadsTWUninit() #1 {
; SMALL32-LABEL: loadsTWUninit:
; SMALL32:       # %bb.0: # %entry
; SMALL32-NEXT:    mflr 0
; SMALL32-NEXT:    stw 0, 8(1)
; SMALL32-NEXT:    stwu 1, -32(1)
; SMALL32-NEXT:    lwz 3, L..C6(2)
; SMALL32-NEXT:    lwz 4, L..C7(2)
; SMALL32-NEXT:    bla .__tls_get_addr
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
; LARGE32-NEXT:    stw 0, 8(1)
; LARGE32-NEXT:    stwu 1, -32(1)
; LARGE32-NEXT:    addis 3, L..C6@u(2)
; LARGE32-NEXT:    addis 4, L..C7@u(2)
; LARGE32-NEXT:    lwz 3, L..C6@l(3)
; LARGE32-NEXT:    lwz 4, L..C7@l(4)
; LARGE32-NEXT:    bla .__tls_get_addr
; LARGE32-NEXT:    lwz 3, 0(3)
; LARGE32-NEXT:    addis 4, L..C8@u(2)
; LARGE32-NEXT:    lwz 4, L..C8@l(4)
; LARGE32-NEXT:    lwz 4, 0(4)
; LARGE32-NEXT:    add 3, 4, 3
; LARGE32-NEXT:    addi 1, 1, 32
; LARGE32-NEXT:    lwz 0, 8(1)
; LARGE32-NEXT:    mtlr 0
; LARGE32-NEXT:    blr
entry:
  %0 = load i32, i32* @TWUninit, align 4
  %1 = load i32, i32* @GInit, align 4
  %add = add nsw i32 %1, %0
  ret i32 %add
}

; TOC entry checks

; SMALL32-LABEL: .toc
; SMALL32-LABEL: L..C0:
; SMALL32-NEXT:	 .tc .TGUninit[TC],TGUninit[TL]@m
; SMALL32-LABEL: L..C1:
; SMALL32-NEXT:	 .tc TGUninit[TC],TGUninit[TL]
; SMALL32-LABEL: L..C2:
; SMALL32-NEXT:	 .tc .TGInit[TC],TGInit[TL]@m
; SMALL32-LABEL: L..C3:
; SMALL32-NEXT:	 .tc TGInit[TC],TGInit[TL]
; SMALL32-LABEL: L..C4:
; SMALL32-NEXT:	 .tc .TIUninit[TC],TIUninit[UL]@m
; SMALL32-LABEL: L..C5:
; SMALL32-NEXT:	 .tc TIUninit[TC],TIUninit[UL]
; SMALL32-LABEL: L..C6:
; SMALL32-NEXT:	 .tc .TWUninit[TC],TWUninit[TL]@m
; SMALL32-LABEL: L..C7:
; SMALL32-NEXT:	 .tc TWUninit[TC],TWUninit[TL]
; SMALL32-LABEL: L..C8:
; SMALL32-NEXT:	 .tc GInit[TC],GInit[RW]

; LARGE32-LABEL: .toc
; LARGE32-LABEL: L..C0:
; LARGE32-NEXT:  .tc .TGUninit[TE],TGUninit[TL]@m
; LARGE32-LABEL: L..C1:
; LARGE32-NEXT:  .tc TGUninit[TE],TGUninit[TL]
; LARGE32-LABEL: L..C2:
; LARGE32-NEXT:  .tc .TGInit[TE],TGInit[TL]@m
; LARGE32-LABEL: L..C3:
; LARGE32-NEXT:  .tc TGInit[TE],TGInit[TL]
; LARGE32-LABEL: L..C4:
; LARGE32-NEXT:  .tc .TIUninit[TE],TIUninit[UL]@m
; LARGE32-LABEL: L..C5:
; LARGE32-NEXT:  .tc TIUninit[TE],TIUninit[UL]
; LARGE32-LABEL: L..C6:
; LARGE32-NEXT:  .tc .TWUninit[TE],TWUninit[TL]@m
; LARGE32-LABEL: L..C7:
; LARGE32-NEXT:  .tc TWUninit[TE],TWUninit[TL]
; LARGE32-LABEL: L..C8:
; LARGE32-NEXT:  .tc GInit[TE],GInit[RW]

attributes #0 = { nofree norecurse nounwind willreturn writeonly "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pwr4" "target-features"="-altivec,-bpermd,-crypto,-direct-move,-extdiv,-float128,-htm,-mma,-paired-vector-memops,-power10-vector,-power8-vector,-power9-vector,-rop-protection,-spe,-vsx" }
attributes #1 = { norecurse nounwind readonly willreturn "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="pwr4" "target-features"="-altivec,-bpermd,-crypto,-direct-move,-extdiv,-float128,-htm,-mma,-paired-vector-memops,-power10-vector,-power8-vector,-power9-vector,-rop-protection,-spe,-vsx" }
