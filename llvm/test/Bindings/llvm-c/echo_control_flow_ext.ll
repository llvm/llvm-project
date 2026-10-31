; RUN: llvm-as < %s | llvm-dis > %t.orig
; RUN: llvm-as < %s | llvm-c-test --echo > %t.echo
; RUN: diff -w %t.orig %t.echo
;
; Extended echo coverage for control-flow forms that require blockaddress
; constants and successor reconstruction.

define i32 @echo_switch(i32 %x) {
entry:
  switch i32 %x, label %sw.default [
    i32 0, label %sw.case0
    i32 1, label %sw.case1
  ]

sw.case0:
  ret i32 10

sw.case1:
  ret i32 11

sw.default:
  ret i32 12
}

define i32 @echo_indirectbr(i1 %cond) {
entry:
  %dest = select i1 %cond, ptr blockaddress(@echo_indirectbr, %bb1), ptr blockaddress(@echo_indirectbr, %bb2)
  indirectbr ptr %dest, [label %bb1, label %bb2]

bb1:
  ret i32 1

bb2:
  ret i32 2
}
