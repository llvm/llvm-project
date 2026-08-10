; RUN: not llc < %s -mtriple=bpfel 2>&1 | FileCheck %s

define i64 @vla(i64 %num) {
; CHECK: unsupported dynamic stack allocation
    %vla = alloca i32, i64 %num
    %ret = ptrtoint ptr %vla to i64
    ret i64 %ret
}
