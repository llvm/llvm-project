; RUN: llvm-link -S -o %t %s %p/Inputs/comdat-rm-dst.ll
; RUN: FileCheck %s < %t
; RUN: FileCheck --check-prefix=RM %s < %t

target datalayout = "e-m:w-p:32:32-i64:64-f80:32-n8:16:32-S32"
target triple = "i686-pc-windows-msvc"

$foo = comdat largest
@foo = global i32 42, comdat
; CHECK-DAG: @foo = global i64 43, comdat

; RM-NOT: @alias =
@alias = alias i32, ptr @foo

; We should arguably reject an out of comdat reference to int_alias. Given that
; the verifier accepts it, test that we at least produce an output that passes
; the verifier.
; CHECK-DAG: @int_alias = external global i32
@int_alias = internal alias i32, ptr @foo
@bar = global ptr @int_alias

@func_alias = alias void (), ptr @func
@zed = global ptr @func_alias
; CHECK-DAG: @zed = global ptr @func_alias
; CHECK-DAG: declare void @func_alias()

; RM-NOT: @func()
define void @func() comdat($foo) {
  ret void
}

; RM-NOT: var
@var = global i32 42, comdat($foo)
