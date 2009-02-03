; PR1075
; RUN: llvm-as < %s | llc -mtriple=x86_64-apple-darwin | \
; RUN:   %prcontext {mulss	LCPI1_3} 1 | grep mulss | count 1

define float @foo(float %x) {
    %tmp1 = mul float %x, 3.000000e+00
    %tmp3 = mul float %x, 5.000000e+00
    %tmp5 = mul float %x, 7.000000e+00
    %tmp7 = mul float %x, 1.100000e+01
    %tmp10 = add float %tmp1, %tmp3
    %tmp12 = add float %tmp10, %tmp5
    %tmp14 = add float %tmp12, %tmp7
    ret float %tmp14
}
