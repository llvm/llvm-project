; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s

; These tests would be improved by 'movs r0, #0' being rematerialized below the
; test as 'mov.w r0, #0'.

; 0x000000bb = 187
define i1 @f1(i32 %a) {
; CHECK: f1:
; CHECK: cmp {{.*}}, #187
    %tmp = icmp ne i32 %a, 187
    ret i1 %tmp
}

; 0x00aa00aa = 11141290
define i1 @f2(i32 %a) {
; CHECK: f2:
; CHECK: cmp.w {{.*}}, #11141290
    %tmp = icmp eq i32 %a, 11141290 
    ret i1 %tmp
}

; 0xcc00cc00 = 3422604288
define i1 @f3(i32 %a) {
; CHECK: f3:
; CHECK: cmp.w {{.*}}, #-872363008
    %tmp = icmp ne i32 %a, 3422604288
    ret i1 %tmp
}

; 0xdddddddd = 3722304989
define i1 @f4(i32 %a) {
; CHECK: f4:
; CHECK: cmp.w {{.*}}, #-572662307
    %tmp = icmp ne i32 %a, 3722304989
    ret i1 %tmp
}

; 0x00110000 = 1114112
define i1 @f5(i32 %a) {
; CHECK: f5:
; CHECK: cmp.w {{.*}}, #1114112
    %tmp = icmp eq i32 %a, 1114112
    ret i1 %tmp
}

; Check that we don't do an invalid (a > b) --> !(a < b + 1) transform.
;
; CHECK: f6:
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
