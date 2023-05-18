; Check that sanitizers invalidate GlobalsAA

; Msan and Dfsan use globals for origin tracking and TLS for parameters.
; RUN: opt < %s -S -passes='require<globals-aa>,module(msan)' -debug-pass-manager 2>&1 | FileCheck %s
; RUN: opt < %s -S -passes='require<globals-aa>,module(dfsan)' -debug-pass-manager 2>&1 | FileCheck %s

; Some types of coverage use globals.
; RUN: opt < %s -S -passes='require<globals-aa>,module(sancov-module)' -sanitizer-coverage-level=2 -debug-pass-manager 2>&1 | FileCheck %s

; Uses TLS for tags.
; RUN: opt < %s -S -passes='require<globals-aa>,module(hwasan)' -debug-pass-manager 2>&1 | FileCheck %s

; Modifies globals.
; RUN: opt < %s -S -passes='require<globals-aa>,module(asan)' -debug-pass-manager 2>&1 | FileCheck %s

; CHECK: Running analysis: GlobalsAA on [module]
; CHECK: Running pass: {{.*}}Sanitizer{{.*}}Pass on [module]
; CHECK: Invalidating analysis: GlobalsAA on [module]

target triple = "x86_64-unknown-linux"

define i32 @test(ptr readonly %a) local_unnamed_addr sanitize_address sanitize_hwaddress {
entry:
  %0 = load i32, ptr %a, align 4
  ret i32 %0
}
