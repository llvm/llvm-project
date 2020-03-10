; RUN: llc  -mtriple=x86_64-unknown-unknown --exception-model=sjlj --print-after=sjljehprepare < %s |& FileCheck --check-prefix=CHECK-X86 %s
; RUN: (llc  -mtriple=ve-unknown-unknown --exception-model=sjlj  --print-after=sjljehprepare < %s || true) |& FileCheck --check-prefix=CHECK-VE %s

@SomeGlobal = external dso_local global i8

define dso_local i32 @foo(i32 %arg) local_unnamed_addr personality i8* bitcast (i32 (...)* @__gxx_personality_sj0 to i8*) {
; CHECK-VE: *** IR Dump After SJLJ Exception Handling preparation ***
; CHECK-VE-NEXT: define dso_local i32 @foo(i32 %arg) local_unnamed_addr personality i8* bitcast (i32 (...)* @__gxx_personality_sj0 to i8*) {
; CHECK-VE-NEXT: entry:
; CHECK-VE-NEXT:   %fn_context = alloca { i8*, i64, [4 x i64], i8*, i8*, [5 x i8*] }, align 8
; CHECK-VE-NEXT:   %arg.tmp = select i1 true, i32 %arg, i32 undef
; CHECK-VE-NEXT:   %pers_fn_gep = getelementptr { i8*, i64, [4 x i64], i8*, i8*, [5 x i8*] }, { i8*, i64, [4 x i64], i8*, i8*, [5 x i8*] }* %fn_context, i32 0, i32 3
; CHECK-X86: *** IR Dump After SJLJ Exception Handling preparation ***
; CHECK-X86-NEXT: define dso_local i32 @foo(i32 %arg) local_unnamed_addr personality i8* bitcast (i32 (...)* @__gxx_personality_sj0 to i8*) {
; CHECK-X86-NEXT: entry:
; CHECK-X86-NEXT:   %fn_context = alloca { i8*, i32, [4 x i32], i8*, i8*, [5 x i8*] }, align 8
; CHECK-X86-NEXT:   %arg.tmp = select i1 true, i32 %arg, i32 undef
; CHECK-X86-NEXT:   %pers_fn_gep = getelementptr { i8*, i32, [4 x i32], i8*, i8*, [5 x i8*] }, { i8*, i32, [4 x i32], i8*, i8*, [5 x i8*] }* %fn_context, i32 0, i32 3
entry:
  invoke void @errorbar() to label %exit unwind label %handle

handle:
  %error = landingpad { i8*, i32 } catch i8* @SomeGlobal
  ret i32 1

exit:
  ret i32 0
}

declare dso_local void @errorbar() local_unnamed_addr

declare dso_local i32 @__gxx_personality_sj0(...)
