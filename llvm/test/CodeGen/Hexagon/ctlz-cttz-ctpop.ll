; RUN: llc -march=hexagon -mcpu=hexagonv4 < %s | FileCheck %s

; CHECK-DAG: ct0({{r[0-9]*:[0-9]*}})
; CHECK-DAG: cl0({{r[0-9]*:[0-9]*}})
; CHECK-DAG: ct0({{r[0-9]*}})
; CHECK-DAG: cl0({{r[0-9]*}})
; CHECK-DAG: r{{[0-9]+}} += lsr(r{{[0-9]+}},#4)

define i32 @foo(i64 %a, i32 %b) nounwind  {
entry:
        %tmp0 = tail call i64 @llvm.ctlz.i64( i64 %a, i1 true )
        %tmp1 = tail call i64 @llvm.cttz.i64( i64 %a, i1 true )
        %tmp2 = tail call i32 @llvm.ctlz.i32( i32 %b, i1 true )
        %tmp3 = tail call i32 @llvm.cttz.i32( i32 %b, i1 true )
        %tmp4 = tail call i64 @llvm.ctpop.i64( i64 %a )
        %tmp5 = tail call i32 @llvm.ctpop.i32( i32 %b )


        %tmp6 = trunc i64 %tmp0 to i32
        %tmp7 = trunc i64 %tmp1 to i32
        %tmp8 = trunc i64 %tmp4 to i32
        %tmp9 = add i32 %tmp6, %tmp7
        %tmp10 = add i32 %tmp9, %tmp8
        %tmp11 = add i32 %tmp10, %tmp2
        %tmp12 = add i32 %tmp11, %tmp3
        %tmp13 = add i32 %tmp12, %tmp5

        ret i32 %tmp13
}

declare i64 @llvm.ctlz.i64(i64, i1) nounwind readnone
declare i32 @llvm.ctlz.i32(i32, i1) nounwind readnone
declare i64 @llvm.cttz.i64(i64, i1) nounwind readnone
declare i32 @llvm.cttz.i32(i32, i1) nounwind readnone
declare i64 @llvm.ctpop.i64(i64) nounwind readnone
declare i32 @llvm.ctpop.i32(i32) nounwind readnone
