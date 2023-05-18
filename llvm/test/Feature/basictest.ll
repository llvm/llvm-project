; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

; Test "stripped" format where nothing is symbolic... this is how the bytecode
; format looks anyways (except for negative vs positive offsets)...
;

define void @void(i32, i32) {
        add i32 0, 0            ; <i32>:3 [#uses=2]
        sub i32 0, 4            ; <i32>:4 [#uses=2]
        br label %5

; <label>:5             ; preds = %5, %2
        add i32 %0, %1          ; <i32>:6 [#uses=2]
        sub i32 %6, %4          ; <i32>:7 [#uses=1]
        icmp sle i32 %7, %3             ; <i1>:8 [#uses=1]
        br i1 %8, label %9, label %5

; <label>:9             ; preds = %5
        add i32 %0, %1          ; <i32>:10 [#uses=0]
        sub i32 %6, %4          ; <i32>:11 [#uses=1]
        icmp sle i32 %11, %3            ; <i1>:12 [#uses=0]
        ret void
}

; This function always returns zero
define i32 @zarro() {
Startup:
        ret i32 0
}
