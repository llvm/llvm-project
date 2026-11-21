; RUN: not llc -mtriple=x86_64-unknown-linux-gnu -filetype=null %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=i686-unknown-linux-gnu -filetype=null %s 2>&1 | FileCheck %s

; The X86 floating-point environment is 256 bits. Other widths are rejected
; with an error instead of reading or writing past the memory operand.
; https://github.com/llvm/llvm-project/issues/228830

; CHECK: error: unsupported type for get_fpenv_mem: the X86 floating-point environment is 256 bits
define i64 @get_fpenv_i64() nounwind {
  %env = call i64 @llvm.get.fpenv.i64()
  ret i64 %env
}

; CHECK: error: unsupported type for set_fpenv_mem: the X86 floating-point environment is 256 bits
define void @set_fpenv_i128(i128 %env) nounwind {
  call void @llvm.set.fpenv.i128(i128 %env)
  ret void
}

; CHECK: error: unsupported type for get_fpenv_mem: the X86 floating-point environment is 256 bits
define void @get_fpenv_i512(ptr %p) nounwind {
  %env = call i512 @llvm.get.fpenv.i512()
  store i512 %env, ptr %p
  ret void
}
