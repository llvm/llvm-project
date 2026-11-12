; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll


declare i32 @printf(ptr, ...)   ;; Prototype for: int __builtin_printf(const char*, ...)

define i32 @testvarar() {
        call i32 (ptr, ...) @printf( ptr null, i32 12, i8 42 )         ; <i32>:1 [#uses=1]
        ret i32 %1
}

