; RUN: opt %s -load=%llvmshlibdir/libBye%shlibext -goodbye -wave-goodbye \
; RUN:   -disable-output 2>&1 | FileCheck %s
; REQUIRES: plugins
; CHECK: Bye

@junk = global i32 0

define i32* @somefunk() {
  ret i32* @junk
}

