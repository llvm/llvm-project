; RUN: opt -passes=inline -S < %s | FileCheck %s
; RUN: opt -passes='cgscc(inline)' -S < %s | FileCheck %s
; RUN: opt -passes='module-inline' -S < %s | FileCheck %s

; PR23216: We can't inline functions using llvm.localescape.

declare void @llvm.localescape(...)
declare ptr @llvm.frameaddress(i32)
declare ptr @llvm.localrecover(ptr, ptr, i32)

define internal void @foo(ptr %fp) {
  %a.i8 = call ptr @llvm.localrecover(ptr @bar, ptr %fp, i32 0)
  store i32 42, ptr %a.i8
  ret void
}

define internal i32 @bar() {
entry:
  %a = alloca i32
  call void (...) @llvm.localescape(ptr %a)
  %fp = call ptr @llvm.frameaddress(i32 0)
  tail call void @foo(ptr %fp)
  %r = load i32, ptr %a
  ret i32 %r
}

; We even bail when someone marks it alwaysinline.
define internal i32 @bar_alwaysinline() alwaysinline {
entry:
  %a = alloca i32
  call void (...) @llvm.localescape(ptr %a)
  tail call void @foo(ptr null)
  ret i32 0
}

define i32 @bazz() {
entry:
  %r = tail call i32 @bar()
  %r1 = tail call i32 @bar_alwaysinline()
  ret i32 %r
}

; CHECK: define i32 @bazz()
; CHECK: call i32 @bar()
; CHECK: call i32 @bar_alwaysinline()
