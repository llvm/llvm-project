; RUN: llvm-as < %s | llvm-dis > %t.orig
; RUN: llvm-as < %s | llvm-c-test --echo --opaque-pointers > %t.echo
; RUN: diff -w %t.orig %t.echo

define i32 @main() {
  %1 = alloca i32, align 4
  %2 = cmpxchg ptr %1, i32 2, i32 3 seq_cst acquire
  %3 = extractvalue { i32, i1 } %2, 0
  ret i32 %3
}
