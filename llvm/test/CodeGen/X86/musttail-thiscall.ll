; RUN: llc -verify-machineinstrs -mtriple=i686-- < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=i686-- -O0 < %s | FileCheck %s

; CHECK-LABEL: t1:
; CHECK: jmp {{_?}}t1_callee
define x86_thiscallcc void @t1(ptr %this) {
  %adj = getelementptr i8, ptr %this, i32 4
  musttail call x86_thiscallcc void @t1_callee(ptr %adj)
  ret void
}
declare x86_thiscallcc void @t1_callee(ptr %this)

; CHECK-LABEL: t2:
; CHECK: jmp {{_?}}t2_callee
define x86_thiscallcc i32 @t2(ptr %this, i32 %a) {
  %adj = getelementptr i8, ptr %this, i32 4
  %rv = musttail call x86_thiscallcc i32 @t2_callee(ptr %adj, i32 %a)
  ret i32 %rv
}
declare x86_thiscallcc i32 @t2_callee(ptr %this, i32 %a)

; CHECK-LABEL: t3:
; CHECK: jmp {{_?}}t3_callee
define x86_thiscallcc ptr @t3(ptr %this, ptr inalloca(<{ ptr, i32 }>) %args) {
  %adj = getelementptr i8, ptr %this, i32 4
  %a_ptr = getelementptr <{ ptr, i32 }>, ptr %args, i32 0, i32 1
  store i32 0, ptr %a_ptr
  %rv = musttail call x86_thiscallcc ptr @t3_callee(ptr %adj, ptr inalloca(<{ ptr, i32 }>) %args)
  ret ptr %rv
}
declare x86_thiscallcc ptr @t3_callee(ptr %this, ptr inalloca(<{ ptr, i32 }>) %args);

; CHECK-LABEL: t4:
; CHECK: jmp {{_?}}t4_callee
define x86_thiscallcc ptr @t4(ptr %this, ptr preallocated(<{ ptr, i32 }>) %args) {
  %adj = getelementptr i8, ptr %this, i32 4
  %a_ptr = getelementptr <{ ptr, i32 }>, ptr %args, i32 0, i32 1
  store i32 0, ptr %a_ptr
  %rv = musttail call x86_thiscallcc ptr @t4_callee(ptr %adj, ptr preallocated(<{ ptr, i32 }>) %args)
  ret ptr %rv
}
declare x86_thiscallcc ptr @t4_callee(ptr %this, ptr preallocated(<{ ptr, i32 }>) %args);
