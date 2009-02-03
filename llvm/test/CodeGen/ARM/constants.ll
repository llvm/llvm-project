; RUN: llvm-as < %s | llc -march=arm | \
; RUN:   grep {mov r0, #0} | count 1
; RUN: llvm-as < %s | llc -march=arm | \
; RUN:   grep {mov r0, #255$} | count 1
; RUN: llvm-as < %s | llc -march=arm | \
; RUN:   grep {mov r0.*256} | count 1
; RUN: llvm-as < %s | llc -march=arm | grep {orr.*256} | count 1
; RUN: llvm-as < %s | llc -march=arm | grep {mov r0, .*-1073741761} | count 1
; RUN: llvm-as < %s | llc -march=arm | grep {mov r0, .*1008} | count 1
; RUN: llvm-as < %s | llc -march=arm | grep {cmp r0, #1, 16} | count 1

define i32 @f1() {
        ret i32 0
}

define i32 @f2() {
        ret i32 255
}

define i32 @f3() {
        ret i32 256
}

define i32 @f4() {
        ret i32 257
}

define i32 @f5() {
        ret i32 -1073741761
}

define i32 @f6() {
        ret i32 1008
}

define void @f7(i32 %a) {
        %b = icmp ugt i32 %a, 65536             ; <i1> [#uses=1]
        br i1 %b, label %r, label %r

r:              ; preds = %0, %0
        ret void
}
