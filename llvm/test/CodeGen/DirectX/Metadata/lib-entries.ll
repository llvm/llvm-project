; RUN: opt -S  -S -dxil-translate-metadata %s 2>&1 | FileCheck %s
target triple = "dxil-pc-shadermodel6.8-library"


; CHECK: !dx.shaderModel = !{![[SM:[0-9]+]]}
; CHECK: !dx.version = !{![[DXVER:[0-9]+]]}
; CHECK: !dx.entryPoints = !{![[LIB:[0-9]+]], ![[AS:[0-9]+]], ![[MS:[0-9]+]], ![[CS:[0-9]+]]}

; CHECK: ![[SM]] = !{!"lib", i32 6, i32 8}
; CHECK: ![[DXVER]] = !{i32 1, i32 8}
; CHECK: ![[LIB]] = !{null, !"", null, null, null}
; CHECK: ![[AS]] = !{ptr @entry_as, !"entry_as", null, null, ![[AS_SF:[0-9]*]]}
; CHECK: ![[AS_SF]] =  !{i32 8, i32 14}
; CHECK: ![[MS]] = !{ptr @entry_ms, !"entry_ms", null, null, ![[MS_SF:[0-9]*]]}
; CHECK: ![[MS_SF]] =  !{i32 8, i32 13}
; CHECK: ![[CS]] = !{ptr @entry_cs, !"entry_cs", null, null, ![[CS_SF:[0-9]*]]}
; CHECK: ![[CS_SF]] =  !{i32 8, i32 5, i32 4, ![[CS_NT:[0-9]*]]}
; CHECK: !{i32 1, i32 2, i32 1}

define void @entry_as() #0 {
entry:
  ret void
}

define i32 @entry_ms(i32 %a) #1 {
entry:
  ret i32 %a
}

define float @entry_cs(float %f) #3 {
entry:
  ret float %f
}

attributes #0 = { noinline nounwind "hlsl.shader"="amplification" }
attributes #1 = { noinline nounwind "hlsl.shader"="mesh" }
attributes #3 = { noinline nounwind "hlsl.numthreads"="1,2,1" "hlsl.shader"="compute" }
