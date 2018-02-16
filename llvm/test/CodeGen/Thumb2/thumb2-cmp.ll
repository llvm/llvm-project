; RUN: llc -mtriple=thumb-eabi -mcpu=arm1156t2-s -mattr=+thumb2 %s -o - | FileCheck %s

; These tests would be improved by 'movs r0, #0' being rematerialized below the
; test as 'mov.w r0, #0'.

; 0x000000bb = 187
define i32 @f1(i32 %a) {
; CHECK-LABEL: f1:
; CHECK: cmp {{.*}}, #187
    %tmp = icmp ne i32 %a, 187
    %ret = select i1 %tmp, i32 42, i32 24
    ret i32 %ret
}

; 0x00aa00aa = 11141290
define i32 @f2(i32 %a) {
; CHECK-LABEL: f2:
; CHECK: cmp.w {{.*}}, #11141290
    %tmp = icmp eq i32 %a, 11141290 
    %ret = select i1 %tmp, i32 42, i32 24
    ret i32 %ret
}

; 0xcc00cc00 = 3422604288
define i32 @f3(i32 %a) {
; CHECK-LABEL: f3:
; CHECK: cmp.w {{.*}}, #-872363008
    %tmp = icmp ne i32 %a, 3422604288
    %ret = select i1 %tmp, i32 42, i32 24
    ret i32 %ret
}

; 0xdddddddd = 3722304989
define i32 @f4(i32 %a) {
; CHECK-LABEL: f4:
; CHECK: cmp.w {{.*}}, #-572662307
    %tmp = icmp ne i32 %a, 3722304989
    %ret = select i1 %tmp, i32 42, i32 24
    ret i32 %ret
}

; 0x00110000 = 1114112
define i32 @f5(i32 %a) {
; CHECK-LABEL: f5:
; CHECK: cmp.w {{.*}}, #1114112
    %tmp = icmp eq i32 %a, 1114112
    %ret = select i1 %tmp, i32 42, i32 24
    ret i32 %ret
}

; Check that we don't do an invalid (a > b) --> !(a < b + 1) transform.
;
; CHECK-LABEL: f6:
; CHECK-NOT: cmp.w {{.*}}, #-2147483648
; CHECK: bx lr
define i32 @f6(i32 %a) {
    %tmp = icmp sgt i32 %a, 2147483647
    br i1 %tmp, label %true, label %false
true:
    ret i32 2
false:
    ret i32 0
}
