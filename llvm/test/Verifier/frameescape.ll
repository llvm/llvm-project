; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

declare void @llvm.localescape(...)
declare i8* @llvm.localrecover(i8*, i8*, i32)

define internal void @f() {
  %a = alloca i8
  call void (...) @llvm.localescape(i8* %a)
  call void (...) @llvm.localescape(i8* %a)
  ret void
}
; CHECK: multiple calls to llvm.localescape in one function

define internal void @g() {
entry:
  %a = alloca i8
  br label %not_entry
not_entry:
  call void (...) @llvm.localescape(i8* %a)
  ret void
}
; CHECK: llvm.localescape used outside of entry block

define internal void @h() {
  call i8* @llvm.localrecover(i8* null, i8* null, i32 0)
  ret void
}
; CHECK: llvm.localrecover first argument must be function defined in this module

@global = constant i8 0

declare void @declaration()

define internal void @i() {
  call i8* @llvm.localrecover(i8* @global, i8* null, i32 0)
  ret void
}
; CHECK: llvm.localrecover first argument must be function defined in this module

define internal void @j() {
  call i8* @llvm.localrecover(i8* bitcast(void()* @declaration to i8*), i8* null, i32 0)
  ret void
}
; CHECK: llvm.localrecover first argument must be function defined in this module

define internal void @k(i32 %n) {
  call i8* @llvm.localrecover(i8* bitcast(void()* @f to i8*), i8* null, i32 %n)
  ret void
}

; CHECK: immarg operand has non-immediate parameter
; CHECK-NEXT: i32 %n
; CHECK-NEXT: %1 = call i8* @llvm.localrecover(i8* bitcast (void ()* @f to i8*), i8* null, i32 %n)

define internal void @l(i8* %b) {
  %a = alloca i8
  call void (...) @llvm.localescape(i8* %a, i8* %b)
  ret void
}
; CHECK: llvm.localescape only accepts static allocas

define internal void @m() {
  %a = alloca i8
  call void (...) @llvm.localescape(i8* %a)
  ret void
}

define internal void @n(i8* %fp) {
  call i8* @llvm.localrecover(i8* bitcast(void ()* @m to i8*), i8* %fp, i32 1)
  ret void
}
; CHECK: all indices passed to llvm.localrecover must be less than the number of arguments passed to llvm.localescape in the parent function
