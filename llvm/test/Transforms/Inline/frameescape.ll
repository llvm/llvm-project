; RUN: opt -inline -S < %s | FileCheck %s
; RUN: opt -passes='cgscc(inline)' -S < %s | FileCheck %s
; RUN: opt -passes='module-inline' -S < %s | FileCheck %s

; PR23216: We can't inline functions using llvm.localescape.

declare void @llvm.localescape(...)
declare i8* @llvm.frameaddress(i32)
declare i8* @llvm.localrecover(i8*, i8*, i32)

define internal void @foo(i8* %fp) {
  %a.i8 = call i8* @llvm.localrecover(i8* bitcast (i32 ()* @bar to i8*), i8* %fp, i32 0)
  %a = bitcast i8* %a.i8 to i32*
  store i32 42, i32* %a
  ret void
}

define internal i32 @bar() {
entry:
  %a = alloca i32
  call void (...) @llvm.localescape(i32* %a)
  %fp = call i8* @llvm.frameaddress(i32 0)
  tail call void @foo(i8* %fp)
  %r = load i32, i32* %a
  ret i32 %r
}

; We even bail when someone marks it alwaysinline.
define internal i32 @bar_alwaysinline() alwaysinline {
entry:
  %a = alloca i32
  call void (...) @llvm.localescape(i32* %a)
  tail call void @foo(i8* null)
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
