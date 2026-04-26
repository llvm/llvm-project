target triple = "dxilv1.3-pc-shadermodel6.3-library"

define float @_Z3fooff(float %a, float %b) {
entry:
  %add = fadd float %a, %b
  ret float %add
}

!dx.source.args = !{!0}
!dx.source.contents = !{!1, !2, !3}

!0 = !{!"-g", !"-Tlib_6_3", !"-DUSER_DEF0=42", !"-DUSER_DEF1=43", !"C:\\\\dx-source-metadata.hlsl"}
!1 = !{!"C:\\dx-source-metadata.hlsl", !"#include \22a.hlsl\22\0A#include \22b.hlsl\22\0A\0Afloat foo(float a, float b) {\0A  return a + b;\0A}\0A"}
!2 = !{!"C:\\a.hlsl", !"#include \22b.hlsl\22\0A"}
!3 = !{!"C:\\b.hlsl", !"#include <c.hlsl>\0A"}
