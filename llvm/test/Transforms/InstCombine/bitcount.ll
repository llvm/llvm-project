; Tests to make sure bit counts of constants are folded
; RUN: opt < %s -instcombine -S | grep "ret i32 19"
; RUN: opt < %s -instcombine -S | \
; RUN:   grep -v declare | not grep llvm.ct

declare i31 @llvm.ctpop.i31(i31 %val) 
declare i32 @llvm.cttz.i32(i32 %val, i1) 
declare i33 @llvm.ctlz.i33(i33 %val, i1) 

define i32 @test(i32 %A) {
  %c1 = call i31 @llvm.ctpop.i31(i31 12415124)
  %c2 = call i32 @llvm.cttz.i32(i32 87359874, i1 true)
  %c3 = call i33 @llvm.ctlz.i33(i33 87359874, i1 true)
  %t1 = zext i31 %c1 to i32
  %t3 = trunc i33 %c3 to i32
  %r1 = add i32 %t1, %c2
  %r2 = add i32 %r1, %t3
  ret i32 %r2
}
