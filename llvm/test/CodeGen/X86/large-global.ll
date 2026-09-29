; RUN: llc < %s -mtriple=x86_64-apple-macosx | FileCheck %s
; rdar://11729134

; EmitZerofill was incorrectly expecting a 32-bit "size" so 26214400000
; was printed as 444596224

%struct.X = type { [25000 x i8] }

@gArray = global [1048576 x %struct.X] zeroinitializer, align 16

; CHECK: .zerofill __DATA,__common,_gArray,26214400000,4
