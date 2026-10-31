; RUN: rm -rf %t && split-file %s %t
; RUN: %lli -force-interpreter=true %t/elf.ll | FileCheck %s
; RUN: %lli -force-interpreter=true %t/macho.ll | FileCheck %s
; RUN: not --crash %lli -force-interpreter=true %t/macho-no-prefix.ll 2>&1 \
; RUN:   | FileCheck %s --check-prefix=NO-PREFIX

; The interpreter must strip the '\01' mangling escape, along with the global
; prefix it leaves in the name, when looking up an external function. A name
; without the prefix has no C name, so it isn't found.

; CHECK: hello 42

; NO-PREFIX: Tried to execute an unknown external function

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

;--- macho-no-prefix.ll
target datalayout = "m:o"

@fmt = private constant [10 x i8] c"hello %d\0A\00"

declare i32 @"\01printf"(ptr, ...)

define i32 @main() {
  %r = call i32 (ptr, ...) @"\01printf"(ptr @fmt, i32 42)
  ret i32 0
}
