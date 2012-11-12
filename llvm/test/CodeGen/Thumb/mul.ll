; RUN: llc < %s -march=thumb | grep mul | count 3
; RUN: llc < %s -march=thumb | grep lsl | count 1

define i32 @f1(i32 %u) {
    %tmp = mul i32 %u, %u
    ret i32 %tmp
}

define i32 @f2(i32 %u, i32 %v) {
    %tmp = mul i32 %u, %v
    ret i32 %tmp
}

define i32 @f3(i32 %u) {
    %tmp = mul i32 %u, 5
    ret i32 %tmp
}

define i32 @f4(i32 %u) {
    %tmp = mul i32 %u, 4
    ret i32 %tmp
}
