; RUN: not --crash opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not --crash opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not --crash opt -passes=safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not --crash opt -passes=safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o /dev/null 2>&1 | FileCheck %s

define void @foo()  nounwind uwtable safestack gc "shadow-stack" {
  %1 = alloca ptr, align 4
  call void @bar(ptr %1)
  ; CHECK: LLVM ERROR: gcroot intrinsic not compatible with safestack attribute
  call void @llvm.gcroot(ptr %1, ptr null)
  ret void
}

declare void @bar(ptr)
declare void @llvm.gcroot(ptr, ptr)
