; The intrinsic lowering pass was lowering intrinsics like llvm.memcpy to 
; explicitly specified prototypes, inserting a new function if the old one
; didn't exist.  This caused there to be two external memcpy functions in 
; this testcase for example, which caused the CBE to mangle one, screwing
; everything up.  :(  Test that this does not happen anymore.
;
; RUN: llvm-as < %s | llc -march=c | not grep _memcpy

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32)

declare float* @memcpy(i32*, i32, i32)

define i32 @test(i8* %A, i8* %B, i32* %C) {
        call float* @memcpy( i32* %C, i32 4, i32 17 )           ; <float*>:1 [#uses=0]
        call void @llvm.memcpy.i32( i8* %A, i8* %B, i32 123, i32 14 )
        ret i32 7
}

