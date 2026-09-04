; RUN: llc %s --filetype=obj -o - | obj2yaml | FileCheck %s

; CHECK:        - Name:            VERS
; CHECK:          CompilerVersion:
; CHECK-NEXT:       Major:           {{[0-9]+}}
; CHECK-NEXT:       Minor:           {{[0-9]+}}
; CHECK-NEXT:       IsDebugBuild:    {{true|false}}
; CHECK-NEXT:       IsValidated:     false
; CHECK-NEXT:       CommitCount:     {{[0-9]+}}
; CHECK-NEXT:       ContentSizeInBytes: {{[0-9]+}}
; CHECK-NEXT:       CommitSha:       {{.*}}
; CHECK-NEXT:       CustomVersionString: {{.*}}

target triple = "dxilv1.3-pc-shadermodel6.3-library"

define float @_Z3fooff(float %a, float %b) {
entry:
  %add = fadd float %a, %b
  ret float %add
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "dx-source-metadata.hlsl", directory: "C:\\")
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
