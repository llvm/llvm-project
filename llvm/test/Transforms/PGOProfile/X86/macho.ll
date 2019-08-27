; RUN: opt < %s -pgo-instr-gen -instrprof -S | llc | FileCheck %s --check-prefix=MACHO-DIRECTIVE

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

; MACHO-DIRECTIVE: .weak_definition        ___llvm_profile_raw_version
define i32 @test_macho(i32 %i) {
entry:
  ret i32 %i
}
