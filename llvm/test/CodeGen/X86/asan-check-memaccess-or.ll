; XFAIL: *
; RUN: llc < %s

target triple = "x86_64-pc-win"

define void @load1(ptr nocapture readonly %x) {
  call void @llvm.asan.check.memaccess(ptr %x, i32 0)
  ret void
}

declare void @llvm.asan.check.memaccess(ptr, i32 immarg)
