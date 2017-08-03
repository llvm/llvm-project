; RUN: llc -march=x86 < %s | FileCheck %s
; RUN: llc -march=x86 -O0 < %s | FileCheck %s

; CHECK-LABEL: t1:
; CHECK: jmp {{_?}}t1_callee
define x86_thiscallcc void @t1(i8* %this) {
  %adj = getelementptr i8, i8* %this, i32 4
  musttail call x86_thiscallcc void @t1_callee(i8* %adj)
  ret void
}
declare x86_thiscallcc void @t1_callee(i8* %this)

; CHECK-LABEL: t2:
; CHECK: jmp {{_?}}t2_callee
define x86_thiscallcc i32 @t2(i8* %this, i32 %a) {
  %adj = getelementptr i8, i8* %this, i32 4
  %rv = musttail call x86_thiscallcc i32 @t2_callee(i8* %adj, i32 %a)
  ret i32 %rv
}
declare x86_thiscallcc i32 @t2_callee(i8* %this, i32 %a)

; CHECK-LABEL: t3:
; CHECK: jmp {{_?}}t3_callee
define x86_thiscallcc i8* @t3(i8* %this, <{ i8*, i32 }>* inalloca %args) {
  %adj = getelementptr i8, i8* %this, i32 4
  %a_ptr = getelementptr <{ i8*, i32 }>, <{ i8*, i32 }>* %args, i32 0, i32 1
  store i32 0, i32* %a_ptr
  %rv = musttail call x86_thiscallcc i8* @t3_callee(i8* %adj, <{ i8*, i32 }>* inalloca %args)
  ret i8* %rv
}
declare x86_thiscallcc i8* @t3_callee(i8* %this, <{ i8*, i32 }>* inalloca %args);
