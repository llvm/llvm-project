; ptxas and nvlink do not support COMDAT (see Triple::supportsCOMDAT): A COMDAT
; group set in LLVM IR is thus simply dropped when lowered, leaving only the
; linkage upgrade from weak to external that LLVM bundles with COMDAT.  If these
; variables took that upgrade, linking more than one translation unit together
; would then trigger an nvlink "multiple definition" error on the resulting
; plain, non-weak duplicate symbol.

; RUN: opt < %s -passes='thinlto-pre-link<O2>' --cs-profilegen-file=file \
; RUN:     -cspgo-kind=cspgo-instr-gen-pipeline -S | \
; RUN:   FileCheck -implicit-check-not=comdat %s

target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; CHECK: @__llvm_profile_filename = weak hidden local_unnamed_addr constant [5 x i8] c"file\00"
; CHECK: @__llvm_profile_raw_version = weak protected constant i64

define i32 @test_nvptx(i32 %i) {
entry:
  ret i32 %i
}
