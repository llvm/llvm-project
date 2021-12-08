; XFAIL: *
; RUN: llc < %s

target triple = "x86_64-pc-win"

define void @load1(i8* nocapture readonly %x) {
  call void @llvm.asan.check.memaccess(i8* %x, i32 0)
  ret void
}

declare void @llvm.asan.check.memaccess(i8*, i32 immarg)
