; RUN: split-file %s %t
; RUN: opt -S --dxil-translate-metadata %t/only.ll | FileCheck %t/only.ll
; RUN: opt -S --dxil-translate-metadata %t/min.ll | FileCheck %t/min.ll
; RUN: opt -S --dxil-translate-metadata %t/max.ll | FileCheck %t/max.ll
; RUN: opt -S --dxil-translate-metadata %t/pref.ll | FileCheck %t/pref.ll

; RUN: llc --filetype=obj %t/only.ll -o - | obj2yaml | FileCheck %t/only.ll --check-prefix=OBJ
; RUN: llc --filetype=obj %t/min.ll -o - | obj2yaml | FileCheck %t/min.ll --check-prefix=OBJ
; RUN: llc --filetype=obj %t/max.ll -o - | obj2yaml | FileCheck %t/max.ll --check-prefix=OBJ
; RUN: llc --filetype=obj %t/pref.ll -o - | obj2yaml | FileCheck %t/pref.ll --check-prefix=OBJ

; Test that wave size/range metadata is correctly generated with the correct tag

;--- only.ll

; CHECK: !dx.entryPoints = !{![[#ENTRY:]]}
; CHECK: ![[#ENTRY]] = !{ptr @main, !"main", null, null, ![[#PROPS:]]}
; CHECK: ![[#PROPS]] = !{{{.*}}i32 11, ![[#WAVE_SIZE:]]{{.*}}}
; CHECK: ![[#WAVE_SIZE]] = !{i32 16}

; OBJ: - Name:    PSV0
; OBJ:   PSVInfo:
; OBJ:     MinimumWaveLaneCount: 16
; OBJ:     MaximumWaveLaneCount: 16

target triple = "dxil-unknown-shadermodel6.6-compute"

define void @main() #0 {
entry:
  ret void
}

attributes #0 = { "hlsl.wavesize"="16,0,0" "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

;--- min.ll

; CHECK: !dx.entryPoints = !{![[#ENTRY:]]}
; CHECK: ![[#ENTRY]] = !{ptr @main, !"main", null, null, ![[#PROPS:]]}
; CHECK: ![[#PROPS]] = !{{{.*}}i32 23, ![[#WAVE_SIZE:]]{{.*}}}
; CHECK: ![[#WAVE_SIZE]] = !{i32 16, i32 0, i32 0}

; OBJ: - Name:    PSV0
; OBJ:   PSVInfo:
; OBJ:     MinimumWaveLaneCount: 16
; OBJ:     MaximumWaveLaneCount: 16

target triple = "dxil-unknown-shadermodel6.8-compute"

define void @main() #0 {
entry:
  ret void
}

attributes #0 = { "hlsl.wavesize"="16,0,0" "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

;--- max.ll

; CHECK: !dx.entryPoints = !{![[#ENTRY:]]}
; CHECK: ![[#ENTRY]] = !{ptr @main, !"main", null, null, ![[#PROPS:]]}
; CHECK: ![[#PROPS]] = !{{{.*}}i32 23, ![[#WAVE_SIZE:]]{{.*}}}
; CHECK: ![[#WAVE_SIZE]] = !{i32 16, i32 32, i32 0}

; OBJ: - Name:    PSV0
; OBJ:   PSVInfo:
; OBJ:     MinimumWaveLaneCount: 16
; OBJ:     MaximumWaveLaneCount: 32

target triple = "dxil-unknown-shadermodel6.8-compute"

define void @main() #0 {
entry:
  ret void
}

attributes #0 = { "hlsl.wavesize"="16,32,0" "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

;--- pref.ll

; CHECK: !dx.entryPoints = !{![[#ENTRY:]]}
; CHECK: ![[#ENTRY]] = !{ptr @main, !"main", null, null, ![[#PROPS:]]}
; CHECK: ![[#PROPS]] = !{{{.*}}i32 23, ![[#WAVE_SIZE:]]{{.*}}}
; CHECK: ![[#WAVE_SIZE]] = !{i32 16, i32 64, i32 32}

; OBJ: - Name:    PSV0
; OBJ:   PSVInfo:
; OBJ:     MinimumWaveLaneCount: 16
; OBJ:     MaximumWaveLaneCount: 64

target triple = "dxil-unknown-shadermodel6.8-compute"

define void @main() #0 {
entry:
  ret void
}

attributes #0 = { "hlsl.wavesize"="16,64,32" "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
