; RUN: opt %s %loadbye -goodbye -wave-goodbye -disable-output 2>&1 | FileCheck %s
; REQUIRES: plugins, examples
; CHECK: Bye

@junk = global i32 0

define i32* @somefunk() {
  ret i32* @junk
}

