; RUN: opt %s -dxil-embed -dxil-globals -S -o - | FileCheck %s

; RUN: llc %s --filetype=obj -o - | obj2yaml -o %t0.yaml
; RUN: llc -dx-Zss %s --filetype=obj -o - | obj2yaml -o %t1.yaml
;; Put the YAML files together to compare matched hashes.
; RUN: cat %t0.yaml %t1.yaml | FileCheck %s --check-prefix=YAML

target triple = "dxil-unknown-shadermodel6.5-library"

; CHECK: @dx.hash = private constant [20 x i8] c"\00\00\00\00{{.*}}", section "HASH", align 4

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

;; Check that the default DXContainer has a hash that doesn't include the source
; YAML: --- !dxcontainer
; YAML: - Name:            HASH
; YAML:   Size:            20
; YAML:   Hash:
; YAML:     IncludesSource:  false
; YAML:     Digest:          [ [[HASH:.+]]
; YAML: ...

;; Check that the -Zss DXContainer has a hash that includes the source and
;; is not the same as the hash from the other one.
; YAML: --- !dxcontainer
; YAML: - Name:            HASH
; YAML:   Size:            20
; YAML:   Hash:
; YAML:     IncludesSource:  true
; YAML:     Digest:          [
; YAML-NOT: [[HASH]]
