; RUN: llc %s --filetype=asm -o - | FileCheck %s
; RUN: llc %s --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=DXC
target triple = "dxil-unknown-shadermodel6.5-library"

; CHECK: @dx.hash = private constant [20 x i8] c"\01\00\00\00{{.*}}", section "HASH", align 4

define i32 @add(i32 %a, i32 %b) {
  %sum = add i32 %a, %b
  ret i32 %sum
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "Some Compiler", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "hlsl.hlsl", directory: "/some-path")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 2}
!4 = !{i32 2, !"Debug Info Version", i32 3}

; DXC: - Name:            HASH
; DXC:   Size:            20
; DXC:   Hash:
; DXC:     IncludesSource:  true
; DXC:     Digest:          [ 
