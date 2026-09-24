; Verify that ILDN part is created when debug info is on.

; RUN: llc %s --filetype=obj -o %t.dxbc
; RUN: obj2yaml %t.dxbc >%t.yaml
; RUN: llvm-objcopy --dump-section=DXIL=%t0.bc %t.dxbc
; RUN: %md5sum %t0.bc >%t0.bc.md5
; RUN: cat %t.yaml %t0.bc.md5 | FileCheck %s

; Verify that ILDN part is not created when debug info is off.
; RUN: opt -strip-debug < %s | llc --filetype=obj | obj2yaml | \
; RUN:   FileCheck --implicit-check-not=ILDN --check-prefix=NODEBUG %s

; CHECK:       - Name:            ILDN
; CHECK-NEXT:    Size:            44
; CHECK-NEXT:    DebugName:
; CHECK-NEXT:      Flags:           0
; CHECK-NEXT:      NameLength:      36
; CHECK-NEXT:      DebugName:       [[MD5:[0-9a-f]+]].pdb
; CHECK:       ...
; CHECK-NEXT:  [[MD5]]

; NODEBUG:     - Name: DXIL

target triple = "dxilv1.3-pc-shadermodel6.3-library"

define float @_Z3fooff(float %a, float %b) {
entry1:
  %add = fadd float %a, %b
  ret float %add
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "dx-source-metadata.hlsl", directory: "C:\\")
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
