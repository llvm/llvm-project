; RUN: rm -rf %t && split-file %s %t
; RUN: %lli -jit-kind=mcjit -force-interpreter=true %t/elf.ll | FileCheck %s
; RUN: %lli -jit-kind=mcjit -force-interpreter=true %t/macho.ll | FileCheck %s

; The interpreter must strip the '\01' mangling escape, along with the global
; prefix it leaves in the name, when looking up an external function.

; CHECK: hello 42

;--- elf.ll
target datalayout = "m:e"

@fmt = private constant [10 x i8] c"hello %d\0A\00"

declare i32 @"\01printf"(ptr, ...)

define i32 @main() {
  %r = call i32 (ptr, ...) @"\01printf"(ptr @fmt, i32 42)
  ret i32 0
}

;--- macho.ll
target datalayout = "m:o"

@fmt = private constant [10 x i8] c"hello %d\0A\00"

declare i32 @"\01_printf"(ptr, ...)

define i32 @main() {
  %r = call i32 (ptr, ...) @"\01_printf"(ptr @fmt, i32 42)
  ret i32 0
}
