; RUN: llvm-as < %s | llvm-dis > %t.orig
; RUN: llvm-as < %s | llvm-c-test --echo > %t.echo
; RUN: diff -w %t.orig %t.echo
;
source_filename = "/test/Bindings/basic_blocks.ll"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

define ptr @test_block_address_01() {
entry:
  br label %block_0
block_0:
  ret ptr blockaddress(@test_block_address_01, %block_0)
}

define ptr @test_block_address_02() {
entry:
  ret ptr blockaddress(@test_block_address_01, %block_0)
}

define ptr @test_block_address_03() {
entry:
  br label %block_1
block_1:
  ret ptr blockaddress(@test_block_address_04, %block_2)
}

define ptr @test_block_address_04() {
entry:
  br label %block_2
block_2:
  ret ptr blockaddress(@test_block_address_03, %block_1)
}

define ptr @test_block_address_unnamed_01(i32 %0) {
1:
  br label %2
2:
  br label %block_0
block_0:
  ret ptr blockaddress(@test_block_address_unnamed_01, %2)
}

define void @test_block_address_global_01() {
    br label %foo
foo:
    ret void
}

@block_addr_global = global ptr blockaddress(@test_block_address_global_01, %foo)
