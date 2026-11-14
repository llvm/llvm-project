; Test naked functions on z/OS. A naked function is an XPLINK stub: it has a
; stub entry marker (mark type C'4') instead of an entry point marker, and no
; PPA1. R5 contains the ADA on entry.
;
; RUN: llc -mtriple s390x-ibm-zos < %s | FileCheck %s
; RUN: llc -mtriple s390x-ibm-zos -filetype=obj < %s -o /dev/null

@g = external global i64
declare void @ext()

define i32 @before(i32 signext %arg) {
; CHECK:      * @before
; CHECK-NEXT: * XPLINK Routine Layout Entry
; CHECK-NEXT: L#EPM_before_0 DS 0H
; CHECK:       DC XL1'F1'
entry:
  ret i32 %arg
}

; The stub marker is 8 bytes, padded to 16 bytes like the entry point marker.
define void @stub() naked {
; CHECK:      * @stub
; CHECK-NEXT: * XPLINK Stub Entry Padding
; CHECK-NEXT:  DC 8XL1'00'
; CHECK-NEXT: * XPLINK Stub Entry Marker
; CHECK-NEXT: * Eyecatcher 0x00C300C500C500
; CHECK-NEXT:  DC XL7'00C300C500C500'
; CHECK-NEXT: * Mark Type C'4'
; CHECK-NEXT:  DC XL1'F4'
; CHECK-NEXT:  ENTRY stub
; CHECK:      stub DS 0H
; CHECK:      * %bb.0:
; CHECK-NEXT:  lg 6,8(5)
; CHECK-NEXT:  lg 5,0(5)
; CHECK-NEXT:  basr 7,6
; CHECK-NEXT:  bcr 0,0
; CHECK-NEXT: L#func_end1 DS 0H
entry:
  call void @ext()
  unreachable
}

; Globals are addressed through R5 on entry.
define void @stub_global() naked {
; CHECK:      * @stub_global
; CHECK:       DC XL1'F4'
; CHECK:      stub_global DS 0H
; CHECK:      * %bb.0:
; CHECK-NEXT:  lg 1,{{[0-9]+}}(5)
; CHECK-NEXT:  mvghi 0(1),1
; CHECK-NEXT: L#func_end2 DS 0H
entry:
  store volatile i64 1, ptr @g
  unreachable
}

; Same but using inline asm.
define void @stub_global_asm() naked {
; CHECK:      * @stub_global_asm
; CHECK:       DC XL1'F4'
; CHECK:      stub_global_asm DS 0H
; CHECK:      * %bb.0:
; CHECK-NEXT:  lg 1,{{[0-9]+}}(5)
; CHECK-NEXT:  lg 1,0(1)
; CHECK-NEXT: *APP
; CHECK-NEXT:  lgr 3,1
; CHECK-NEXT:  b 2(7,0)
; CHECK-NEXT: *NO_APP
; CHECK-NEXT: L#func_end3 DS 0H
entry:
  %1 = load i64, ptr @g, align 8
  call void asm sideeffect " lgr 3,$0\0A b 2(7)", "a"(i64 %1)
  unreachable
}

; A function size or alignment hint does not change the stub layout.
define void @stub_minsize() naked minsize optsize {
; CHECK:      * @stub_minsize
; CHECK-NEXT: * XPLINK Stub Entry Padding
; CHECK-NEXT:  DC 8XL1'00'
; CHECK-NEXT: * XPLINK Stub Entry Marker
; CHECK:       DC XL1'F4'
entry:
  call void @ext()
  unreachable
}

; A naked function has no frame pointer or PPA1 related state.
define void @stub_fp() naked "frame-pointer"="all" {
; CHECK:      * @stub_fp
; CHECK:       DC XL1'F4'
entry:
  call void @ext()
  unreachable
}

; The neighbouring function after the stubs still gets its own marker and
; PPA1.
define i32 @after(i32 signext %arg) {
; CHECK:      * @after
; CHECK-NEXT: * XPLINK Routine Layout Entry
; CHECK-NEXT: L#EPM_after_0 DS 0H
; CHECK:       DC XL1'F1'
entry:
  ret i32 %arg
}

; Only the normal functions have PPA1 blocks, in order.
; CHECK:      L#PPA1_before_0 DS 0H
; CHECK:      L#PPA1_after_0 DS 0H
; CHECK-NOT:  PPA1_stub
; CHECK-NOT:  EPM_stub
