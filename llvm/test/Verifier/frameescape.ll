; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

declare void @llvm.localescape(...)
declare ptr @llvm.localrecover(ptr, ptr, i32)

define internal void @f() {
  %a = alloca i8
  call void (...) @llvm.localescape(ptr %a)
  call void (...) @llvm.localescape(ptr %a)
  ret void
}
; CHECK: multiple calls to llvm.localescape in one function

define internal void @g() {
entry:
  %a = alloca i8
  br label %not_entry
not_entry:
  call void (...) @llvm.localescape(ptr %a)
  ret void
}
; CHECK: llvm.localescape used outside of entry block

define internal void @h() {
  call ptr @llvm.localrecover(ptr null, ptr null, i32 0)
  ret void
}
; CHECK: llvm.localrecover first argument must be function defined in this module

@global = constant i8 0

declare void @declaration()

define internal void @i() {
  call ptr @llvm.localrecover(ptr @global, ptr null, i32 0)
  ret void
}
; CHECK: llvm.localrecover first argument must be function defined in this module

define internal void @j() {
  call ptr @llvm.localrecover(ptr @declaration, ptr null, i32 0)
  ret void
}
; CHECK: llvm.localrecover first argument must be function defined in this module

define internal void @k(i32 %n) {
  call ptr @llvm.localrecover(ptr @f, ptr null, i32 %n)
  ret void
}

; CHECK: immarg operand has non-immediate parameter
; CHECK-NEXT: i32 %n
; CHECK-NEXT: %1 = call ptr @llvm.localrecover(ptr @f, ptr null, i32 %n)

define internal void @l(ptr %b) {
  %a = alloca i8
  call void (...) @llvm.localescape(ptr %a, ptr %b)
  ret void
}
; CHECK: llvm.localescape only accepts static allocas

define internal void @m() {
  %a = alloca i8
  call void (...) @llvm.localescape(ptr %a)
  ret void
}

define internal void @n(ptr %fp) {
  call ptr @llvm.localrecover(ptr @m, ptr %fp, i32 1)
  ret void
}
; CHECK: all indices passed to llvm.localrecover must be less than the number of arguments passed to llvm.localescape in the parent function
