; RUN: llc < %s -mtriple=nvptx -mcpu=sm_20 2>&1

; Check that llc doesn't die when given an empty global ctor / dtor.
@llvm.global_ctors = appending global [0 x { i32, ptr, ptr }] []
@llvm.global_dtors = appending global [0 x { i32, ptr, ptr }] []
