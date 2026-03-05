; RUN: llvm-as < %s | llvm-dis > %t.orig
; RUN: llvm-as < %s | llvm-c-test --echo > %t.echo
; RUN: diff -w %t.orig %t.echo
;
; Test extended type coverage for echo command.

source_filename = "types_extended.ll"

; Named struct with explicit body (ensure set_body path is covered)
%NamedStruct = type { i32, i64, ptr }

; Opaque struct (should not set body)
%OpaqueStruct = type opaque

; Test bfloat type
define bfloat @test_bfloat(bfloat %x) {
  %1 = fadd bfloat %x, %x
  ret bfloat %1
}

; Test named struct usage
define %NamedStruct @test_named_struct(%NamedStruct %s) {
  %1 = extractvalue %NamedStruct %s, 0
  %2 = extractvalue %NamedStruct %s, 1
  %3 = extractvalue %NamedStruct %s, 2
  %4 = insertvalue %NamedStruct %s, i32 %1, 0
  %5 = insertvalue %NamedStruct %4, i64 %2, 1
  %6 = insertvalue %NamedStruct %5, ptr %3, 2
  ret %NamedStruct %6
}

; Test opaque struct usage (just as pointer)
define ptr @test_opaque_struct(ptr %p) {
  ret ptr %p
}

; Test metadata type usage in function signature (via llvm intrinsics)
declare void @llvm.dbg.value(metadata, metadata, metadata)
