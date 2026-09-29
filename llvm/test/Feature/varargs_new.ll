; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

; Demonstrate all of the variable argument handling intrinsic functions plus 
; the va_arg instruction.

declare void @llvm.va_start(ptr)

declare void @llvm.va_copy(ptr, ptr)

declare void @llvm.va_end(ptr)

define i32 @test(i32 %X, ...) {
        ; Allocate two va_list items.  On this target, va_list is of type sbyte*
        %ap = alloca ptr                ; <ptr> [#uses=4]
        %aq = alloca ptr                ; <ptr> [#uses=2]

        ; Initialize variable argument processing
        call void @llvm.va_start( ptr %ap )

        ; Read a single integer argument
        %tmp = va_arg ptr %ap, i32             ; <i32> [#uses=1]

        ; Demonstrate usage of llvm.va_copy and llvm_va_end
        %apv = load ptr, ptr %ap            ; <ptr> [#uses=1]
        call void @llvm.va_copy( ptr %aq, ptr %apv )
        call void @llvm.va_end( ptr %aq )

        ; Stop processing of arguments.
        call void @llvm.va_end( ptr %ap )
        ret i32 %tmp
}
