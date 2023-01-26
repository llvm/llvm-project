; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

; Demonstrate all of the variable argument handling intrinsic functions plus 
; the va_arg instruction.

declare void @llvm.va_start(ptr)

declare void @llvm.va_copy(ptr, ptr)

declare void @llvm.va_end(ptr)

define i32 @test(i32 %X, ...) {
        %ap = alloca ptr                ; <ptr> [#uses=4]
        call void @llvm.va_start( ptr %ap )
        %tmp = va_arg ptr %ap, i32             ; <i32> [#uses=1]
        %aq = alloca ptr                ; <ptr> [#uses=2]
        call void @llvm.va_copy( ptr %aq, ptr %ap )
        call void @llvm.va_end( ptr %aq )
        call void @llvm.va_end( ptr %ap )
        ret i32 %tmp
}

