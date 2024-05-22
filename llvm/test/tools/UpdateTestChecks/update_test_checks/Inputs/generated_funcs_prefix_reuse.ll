; RUN: opt -passes=hotcoldsplit -hotcoldsplit-threshold=0 -S < %s | FileCheck %s --check-prefix=REUSE
; RUN: opt -passes=hotcoldsplit -hotcoldsplit-threshold=0 -S < %s | FileCheck %s --check-prefix=REUSE
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

@G = external dso_local global i32, align 4

define void @foo(i32) {
  %2 = icmp eq i32 %0, 0
  tail call void @_Z10sideeffectv()
  br i1 %2, label %sink, label %exit

sink:
  tail call void @_Z10sideeffectv()
  call void @llvm.trap()
  unreachable

exit:
  ret void
}

define void @bar(i32) {
  %2 = icmp eq i32 %0, 0
  tail call void @_Z10sideeffectv()
  br i1 %2, label %sink, label %exit

sink:
  tail call void @_Z10sideeffectv()
  call void @llvm.trap()
  unreachable

exit:
  ret void
}

declare void @llvm.trap() noreturn cold memory(inaccessiblemem: write)
declare void @_Z10sideeffectv()
