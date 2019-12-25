; RUN: opt -ipsccp -S %s | FileCheck %s
; RUN: opt -ipconstprop -S %s | FileCheck %s

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc19.0.24215"

define i32 @dipsy(i32, i32) local_unnamed_addr #0 {
BasicBlock0:
  call void asm "\0D\0Apushl %ebp\0D\0Amovl 8(%esp),%eax\0D\0Amovl 12(%esp), %ebp\0D\0Acalll *%eax\0D\0Apopl %ebp\0D\0Aretl\0D\0A", ""()
  ret i32 0
}

define void @tinkywinky(i32, i32, i32) local_unnamed_addr #0 {
BasicBlock1:
  call void asm "\0D\0A    movl 12(%esp), %ebp\0D\0A    movl 4(%esp), %eax\0D\0A    movl 8(%esp), %esp\0D\0A    jmpl *%eax\0D\0A", ""()
  ret void
}

define void @patatino(i32, i32, i32) local_unnamed_addr #1 {
bb:
  %3 = tail call i32 @dipsy(i32 %0, i32 %1) #0
; Check that we don't accidentally propagate zero.
; CHECK: @tinkywinky(i32 %3, i32 %2, i32 %1) #0
  tail call void @tinkywinky(i32 %3, i32 %2, i32 %1) #0
  ret void
}

attributes #0 = { naked }
attributes #1 = { "frame-pointer"="all" }
