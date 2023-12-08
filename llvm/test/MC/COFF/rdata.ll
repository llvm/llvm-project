; RUN: llc -mtriple i386-pc-win32 < %s | FileCheck %s

%struct.foo = type { i32, i32 }

@"\01?thingy@@3Ufoo@@B" = constant %struct.foo zeroinitializer, align 4
; CHECK: .section        .rdata
