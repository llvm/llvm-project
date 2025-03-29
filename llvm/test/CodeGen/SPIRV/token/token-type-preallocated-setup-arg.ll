; Example of token usage is from https://llvm.org/docs/LangRef.html (Preallocated Operand Bundles)

; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %s -o - 2>&1 | FileCheck %s

; CHECK: A token is encountered but SPIR-V without extensions does not support token type

%foo = type { i64, i32 }

define dso_local spir_func void @test() {
entry:
  %tok = call token @llvm.call.preallocated.setup(i32 1)
  %a = call ptr @llvm.call.preallocated.arg(token %tok, i32 0) preallocated(%foo)
  ret void
}

declare token @llvm.call.preallocated.setup(i32 %num_args)
declare ptr @llvm.call.preallocated.arg(token %setup_token, i32 %arg_index)
