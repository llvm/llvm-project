; RUN: rm -f %t.ll
; RUN: cat %s | sed -e 's/zeroext/signext/' -e 's/noundef/noalias/' -e 's/nounwind/nosync/' > %t.ll
; RUN: not llvm-diff %s %t.ll 2>&1 | FileCheck %s
; CHECK:in function return:
; CHECK:  in block %entry:
; CHECK:    >   %1 = call signext i32 @foo(ptr %0)
; CHECK:    >   ret i32 %1
; CHECK:    <   %1 = call zeroext i32 @foo(ptr %0)
; CHECK:    <   ret i32 %1
; CHECK:in function param:
; CHECK:  in block %entry:
; CHECK:    >   %1 = call i32 @foo(ptr noalias %0)
; CHECK:    >   ret i32 %1
; CHECK:    <   %1 = call i32 @foo(ptr noundef %0)
; CHECK:    <   ret i32 %1
; CHECK:in function function:
; CHECK:  in block %entry:
; CHECK:    >   %1 = call i32 @foo(ptr %0) #0
; CHECK:    >   ret i32 %1
; CHECK:    <   %1 = call i32 @foo(ptr %0) #0
; CHECK:    <   ret i32 %1
; CHECK:in function all_possible:
; CHECK:  in block %entry:
; CHECK:    >   %1 = call signext i32 @foo(ptr noalias %0) #0
; CHECK:    >   ret i32 %1
; CHECK:    <   %1 = call zeroext i32 @foo(ptr noundef %0) #0
; CHECK:    <   ret i32 %1

declare i32 @foo(ptr)

define i32 @return(ptr %0) {
entry:
    %1 = call zeroext i32 @foo(ptr %0)
    ret i32 %1
}

define i32 @param(ptr %0) {
entry:
    %1 = call i32 @foo(ptr noundef %0)
    ret i32 %1
}

define i32 @function(ptr %0) {
entry:
    %1 = call i32 @foo(ptr %0) nounwind
    ret i32 %1
}

define i32 @all_possible(ptr %0) {
entry:
    %1 = call zeroext i32 @foo(ptr noundef %0) nounwind
    ret i32 %1
}
