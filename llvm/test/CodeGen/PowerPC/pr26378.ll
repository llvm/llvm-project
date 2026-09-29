; RUN: llc -verify-machineinstrs -compile-twice -filetype obj \
; RUN:   -mtriple=powerpc64le-unknown-unknown -mcpu=pwr8 < %s
@foo = common global i32 0, align 4
define ptr @blah() #0 {
  ret ptr @foo
}  
