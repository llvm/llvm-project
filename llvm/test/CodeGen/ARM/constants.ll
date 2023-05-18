; RUN: llc < %s -mtriple=armv4t-unknown-linux-gnueabi -disable-cgp-branch-opts -verify-machineinstrs | FileCheck %s

define i32 @f1() {
; CHECK: f1
; CHECK: mov r0, #0
        ret i32 0
}

define i32 @f2() {
; CHECK: f2
; CHECK: mov r0, #255
        ret i32 255
}

define i32 @f3() {
; CHECK: f3
; CHECK: mov r0, #256
        ret i32 256
}

define i32 @f4() {
; CHECK: f4
; CHECK: orr{{.*}}#256
        ret i32 257
}

define i32 @f5() {
; CHECK: f5
; CHECK: mov r0, #-1073741761
        ret i32 -1073741761
}

define i32 @f6() {
; CHECK: f6
; CHECK: mov r0, #1008
        ret i32 1008
}

define void @f7(i32 %a) {
; CHECK: f7
; CHECK: cmp r0, #65536
        %b = icmp ugt i32 %a, 65536
        br i1 %b, label %r, label %r
r:
        ret void
}

define i32 @f8() nounwind {
; Check that constant propagation through (i32)-1 => (float)Nan => (i32)-1
; gives expected result
; CHECK: f8
; CHECK: mvn r0, #0
        %tmp0 = bitcast i32 -1 to float
        %tmp1 = bitcast float %tmp0 to i32
        ret i32 %tmp1
}

%t1 = type { <3 x float>, <3 x float> }

@const1 = global %t1 { <3 x float> zeroinitializer,
                       <3 x float> <float 1.000000e+00,
                                    float 2.000000e+00,
                                    float 3.000000e+00> }, align 16
; CHECK: const1
; CHECK: .zero 16
; CHECK: float 1
; CHECK: float 2
; CHECK: float 3
; CHECK: .zero 4
