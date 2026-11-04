target triple = "dxilv1.3-pc-shadermodel6.3-library"

define float @_Z3fooff(float %a, float %b) {
entry:
  %add = fadd float %a, %b
  ret float %add
}

!llvm.dbg.cu = !{!4}
!llvm.module.flags = !{!6, !7}

!dx.source.args = !{!0}
!dx.source.contents = !{!1, !2, !3}

!0 = !{!"-g", !"-Tlib_6_3", !"-DUSER_DEF0=42", !"-DUSER_DEF1=43", !"C:\\\\dx-source-metadata.hlsl"}
!1 = !{!"C:\\dx-source-metadata.hlsl", !"#include \22a.hlsl\22\0A#include \22b.hlsl\22\0A\0Afloat foo(float a, float b) {\0A  return a + b;\0A}\0A"}
!2 = !{!"C:\\a.hlsl", !"#include \22b.hlsl\22\0A"}
!3 = !{!"C:\\b.hlsl", !"#include <c.hlsl>\0A"}
!4 = distinct !DICompileUnit(language: DW_LANG_C99, file: !5, emissionKind: FullDebug)
!5 = !DIFile(filename: "dx-source-metadata.hlsl", directory: "C:\\")
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
