; RUN: llc < %s -march=nvptx -mcpu=sm_20 -O1 | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 -O1 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx -mcpu=sm_20 -O1 | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 -O1 | %ptxas-verify %}

define i32 @test1(i32 %n, i32 %m) {
;
; CHECK: ld.param.u32   %[[N:r[0-9]+]], [test1_param_0];
; CHECK: ld.param.u32   %[[M:r[0-9]+]], [test1_param_1];
; CHECK: mad.lo.s32     %[[MAD:r[0-9]+]], %[[M]], %[[N]], %[[M]];
; CHECK: st.param.b32   [func_retval0+0], %[[MAD]];
;
  %add = add i32 %n, 1
  %mul = mul i32 %add, %m
  ret i32 %mul
}

define i32 @test1_rev(i32 %n, i32 %m) {
;
; CHECK: ld.param.u32   %[[N:r[0-9]+]], [test1_rev_param_0];
; CHECK: ld.param.u32   %[[M:r[0-9]+]], [test1_rev_param_1];
; CHECK: mad.lo.s32     %[[MAD:r[0-9]+]], %[[M]], %[[N]], %[[M]];
; CHECK: st.param.b32   [func_retval0+0], %[[MAD]];
;
  %add = add i32 %n, 1
  %mul = mul i32 %m, %add
  ret i32 %mul
}

; Transpose (mul (select)) if it can then be folded to mad
define i32 @test2(i32 %n, i32 %m, i32 %s) {
;
; CHECK: ld.param.u32   %[[N:r[0-9]+]], [test2_param_0];
; CHECK: ld.param.u32   %[[M:r[0-9]+]], [test2_param_1];
; CHECK: ld.param.u32   %[[S:r[0-9]+]], [test2_param_2];
; CHECK: setp.lt.s32    %[[COND:p[0-9]+]], %[[S]], 1;
; CHECK: mad.lo.s32     %[[MAD:r[0-9]+]], %[[M]], %[[N]], %[[M]];
; CHECK: selp.b32       %[[SEL:r[0-9]+]], %[[M]], %[[MAD]], %[[COND]];
; CHECK: st.param.b32   [func_retval0+0], %[[SEL]];
;
  %add = add i32 %n, 1
  %cond = icmp slt i32 %s, 1
  %sel = select i1 %cond, i32 1, i32 %add
  %mul = mul i32 %sel, %m
  ret i32 %mul
}

;; Transpose (mul (select)) if it can then be folded to mad
define i32 @test2_rev1(i32 %n, i32 %m, i32 %s) {
;
; CHECK: ld.param.u32   %[[N:r[0-9]+]], [test2_rev1_param_0];
; CHECK: ld.param.u32   %[[M:r[0-9]+]], [test2_rev1_param_1];
; CHECK: ld.param.u32   %[[S:r[0-9]+]], [test2_rev1_param_2];
; CHECK: setp.lt.s32    %[[COND:p[0-9]+]], %[[S]], 1;
; CHECK: mad.lo.s32     %[[MAD:r[0-9]+]], %[[M]], %[[N]], %[[M]];
; CHECK: selp.b32       %[[SEL:r[0-9]+]], %[[MAD]], %[[M]], %[[COND]];
; CHECK: st.param.b32   [func_retval0+0], %[[SEL]];
;
  %add = add i32 %n, 1
  %cond = icmp slt i32 %s, 1
  %sel = select i1 %cond, i32 %add, i32 1
  %mul = mul i32 %sel, %m
  ret i32 %mul
}

;; Transpose (mul (select)) if it can then be folded to mad
define i32 @test2_rev2(i32 %n, i32 %m, i32 %s) {
;
; CHECK: ld.param.u32   %[[N:r[0-9]+]], [test2_rev2_param_0];
; CHECK: ld.param.u32   %[[M:r[0-9]+]], [test2_rev2_param_1];
; CHECK: ld.param.u32   %[[S:r[0-9]+]], [test2_rev2_param_2];
; CHECK: setp.lt.s32    %[[COND:p[0-9]+]], %[[S]], 1;
; CHECK: mad.lo.s32     %[[MAD:r[0-9]+]], %[[M]], %[[N]], %[[M]];
; CHECK: selp.b32       %[[SEL:r[0-9]+]], %[[MAD]], %[[M]], %[[COND]];
; CHECK: st.param.b32   [func_retval0+0], %[[SEL]];
;
  %add = add i32 %n, 1
  %cond = icmp slt i32 %s, 1
  %sel = select i1 %cond, i32 %add, i32 1
  %mul = mul i32  %m, %sel
  ret i32 %mul
}

;; Leave (mul (select)) intact if it transposing is not profitable
define i32 @test3(i32 %n, i32 %m, i32 %s) {
;
; CHECK: ld.param.u32   %[[N:r[0-9]+]], [test3_param_0];
; CHECK: add.s32        %[[ADD:r[0-9]+]], %[[N]], 3;
; CHECK: ld.param.u32   %[[M:r[0-9]+]], [test3_param_1];
; CHECK: ld.param.u32   %[[S:r[0-9]+]], [test3_param_2];
; CHECK: setp.lt.s32    %[[COND:p[0-9]+]], %[[S]], 1;
; CHECK: selp.b32       %[[SEL:r[0-9]+]], 1, %[[ADD]], %[[COND]];
; CHECK: mul.lo.s32     %[[MUL:r[0-9]+]], %[[SEL]], %[[M]];
; CHECK: st.param.b32   [func_retval0+0], %[[MUL]];
;
  %add = add i32 %n, 3
  %cond = icmp slt i32 %s, 1
  %sel = select i1 %cond, i32 1, i32 %add
  %mul = mul i32 %sel, %m
  ret i32 %mul
}
