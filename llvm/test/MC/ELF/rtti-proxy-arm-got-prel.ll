; REQUIRES: arm-registered-target

;; Verify that the generated assembly omits the RTTI proxy and uses the correct
;; relocations and addends.
; RUN: llc %s -mtriple=armv7 -o - | FileCheck %s

;; Verify that the generated object uses the correct relocations and addends.
; RUN: llc %s -filetype=obj -mtriple=thumbv7 -o %t.o
; RUN: llvm-readelf --relocs --hex-dump=.rodata %t.o | FileCheck --check-prefix=OBJ %s

@vtable = dso_local unnamed_addr constant i32 sub (i32 ptrtoint (ptr @rtti.proxy to i32), i32 ptrtoint (ptr @vtable to i32)), align 4

@vtable_with_offset = dso_local unnamed_addr constant [2 x i32] [i32 0, i32 sub (i32 ptrtoint (ptr @rtti.proxy to i32), i32 ptrtoint (ptr @vtable_with_offset to i32))], align 4

@vtable_with_negative_offset = dso_local unnamed_addr constant [2 x i32] [
  i32 sub (
    i32 ptrtoint (ptr @rtti.proxy to i32),
    i32 ptrtoint (ptr getelementptr inbounds ([2 x i32], ptr @vtable_with_negative_offset, i32 0, i32 1) to i32)
  ),
  i32 0
], align 4

@rtti = external global i8, align 4
@rtti.proxy = linkonce_odr hidden unnamed_addr constant ptr @rtti

; CHECK-NOT: rtti.proxy

; CHECK-LABEL: vtable:
; CHECK-NEXT:    .long   rtti(GOT_PREL)+0{{$}}

; CHECK-LABEL: vtable_with_offset:
; CHECK-NEXT:    .long   0
; CHECK-NEXT:    .long   rtti(GOT_PREL)+4{{$}}

; CHECK-LABEL: vtable_with_negative_offset:
; CHECK-NEXT:    .long   rtti(GOT_PREL)-4{{$}}
; CHECK-NEXT:    .long   0

; OBJ-LABEL: Relocation section '.rel.rodata' at offset [[#%#x,]] contains 3 entries:
; OBJ:       {{^}}00000000 [[#]] R_ARM_GOT_PREL [[#]] rtti{{$}}
; OBJ-NEXT:  {{^}}00000008 [[#]] R_ARM_GOT_PREL [[#]] rtti{{$}}
; OBJ-NEXT:  {{^}}0000000c [[#]] R_ARM_GOT_PREL [[#]] rtti{{$}}

; OBJ-LABEL: Hex dump of section '.rodata':
; OBJ-NEXT:  0x00000000 00000000 00000000 04000000 fcffffff
