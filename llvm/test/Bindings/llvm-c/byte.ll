; RUN: llvm-as < %s | llvm-dis > %t.orig
; RUN: llvm-as < %s | llvm-c-test --echo > %t.echo
; RUN: diff -w %t.orig %t.echo

define void @foo(b8 %a, b16 %b, b64 %c) {
  %1 = alloca b16, align 2
  store b16 %b, ptr %1, align 2
  %2 = load b16, ptr %1, align 2
  %3 = bitcast b16 %2 to <2 x b8>
  ret void
}
