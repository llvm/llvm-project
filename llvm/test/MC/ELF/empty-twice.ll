; Check that there is no persistent state in the ELF emitter that crashes us
; when we try to reuse the pass manager
; RUN: llc -compile-twice -filetype=obj %s -o -

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"
target triple = "i386-pc-linux-gnu"
