; RUN: split-file %s %t
; RUN: opt -passes='default<O3>' -S %t/main.ll | FileCheck %s --check-prefix=DEFAULT
; RUN: opt -passes='lto-pre-link<O3>' -S %t/main.ll | FileCheck %s --check-prefix=PRELINK
; RUN: llvm-link %t/main.ll %t/foo.ll -S | opt -passes='lto<O3>' -S | FileCheck %s --check-prefix=LTO

; DEFAULT-LABEL: define {{.*}} @test(
; DEFAULT: ret i1 false

; PRELINK-LABEL: define {{.*}} @test(
; PRELINK: call i1 @llvm.is.constant.i32
; PRELINK: ret i1

; LTO-LABEL: define {{.*}} @test(
; LTO: ret i1 true

;--- main.ll
@foo = external global i32

define i1 @test() {
  %v = load i32, ptr @foo
  %c = call i1 @llvm.is.constant.i32(i32 %v)
  ret i1 %c
}

;--- foo.ll
t@foo = constant i32 42
