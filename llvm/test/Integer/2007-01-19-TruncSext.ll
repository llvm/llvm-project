; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll
; RUN: llvm-as < %s | lli -jit-kind=mcjit --force-interpreter=true | FileCheck %s
; CHECK: -255

@ARRAY   = global [ 20 x i17 ] zeroinitializer
@FORMAT  = constant [ 4 x i8 ] c"%d\0A\00"

declare i32 @printf(i8* %format, ...)

define void @multiply(i32 %index, i32 %X, i32 %Y) {
  %Z = mul i32 %X, %Y
  %P = getelementptr [20 x i17], [20 x i17]* @ARRAY, i32 0, i32 %index
  %Result = trunc i32 %Z to i17
  store i17 %Result, i17* %P
  ret void
}

define i32 @main(i32 %argc, i8** %argv) {
  %i = bitcast i32 0 to i32
  call void @multiply(i32 %i, i32 -1, i32 255) 
  %P = getelementptr [20 x i17], [20 x i17]* @ARRAY, i32 0, i32 0
  %X = load i17, i17* %P
  %result = sext i17 %X to i32
  %fmt = getelementptr [4 x i8], [4 x i8]* @FORMAT, i32 0, i32 0
  call i32 (i8*,...) @printf(i8* %fmt, i32 %result)
  ret i32 0
}

