; RUN: opt < %s -passes=instcombine -S | not grep 1431655764

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"

define i8* @foo([100 x {i8,i8,i8}]* %x) {
entry:
        %p = bitcast [100 x {i8,i8,i8}]* %x to i8*
        %q = getelementptr i8, i8* %p, i32 -4
        ret i8* %q
}
