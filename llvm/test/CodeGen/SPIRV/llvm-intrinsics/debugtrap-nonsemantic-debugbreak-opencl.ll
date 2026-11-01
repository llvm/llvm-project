; RUN: llc -O0 -mtriple=spirv64v1.6-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,CORE
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s --check-prefixes=CHECK,EXT
; Default OpenCL version (1.4) without the extension falls back to OpNop.
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=NOEXT

; CORE-NOT:  OpExtension "SPV_KHR_non_semantic_info"
; EXT:       OpExtension "SPV_KHR_non_semantic_info"
; CHECK:     %[[#Set:]] = OpExtInstImport "NonSemantic.DebugBreak"
; CHECK:     OpExtInst %[[#]] %[[#Set]] DebugBreak
; CHECK:     OpReturn

; NOEXT-NOT: OpExtInstImport "NonSemantic.DebugBreak"
; NOEXT:     OpNop
; NOEXT:     OpReturn

define void @test_kernel() {
  call void @llvm.debugtrap()
  ret void
}

declare void @llvm.debugtrap()
