; RUN: llc -mtriple i386-pc-win32 < %s | FileCheck %s

%struct.foo = type { i32, i32 }

@"\01?thingy@@3Ufoo@@B" = global %struct.foo zeroinitializer, align 4
; CHECK: .bss

@thingy_linkonce = linkonce_odr global %struct.foo zeroinitializer, align 4
; CHECK: .section .bss,"bw",discard,_thingy_linkonce
