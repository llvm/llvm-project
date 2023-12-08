; RUN: llc -march=hexagon -mcpu=hexagonv66 < %s | FileCheck %s

; CHECK-LABEL: @test1
; CHECK: r0 -= mpyi(r1,r2)
define i32 @test1(i32 %rx, i32 %rs, i32 %rt) local_unnamed_addr #0 {
entry:
  %v0 = tail call i32 @llvm.hexagon.M2.mnaci(i32 %rx, i32 %rs, i32 %rt)
  ret i32 %v0
}

declare i32 @llvm.hexagon.M2.mnaci(i32, i32, i32) #1

; CHECK-LABEL: @test2
; CHECK: r1:0 = dfadd(r1:0,r3:2)
define double @test2(double %rss, double %rtt) local_unnamed_addr #0 {
entry:
  %v0 = tail call double @llvm.hexagon.F2.dfadd(double %rss, double %rtt)
  ret double %v0
}

declare double @llvm.hexagon.F2.dfadd(double, double) #1

; CHECK-LABEL: @test3
; CHECK: r1:0 = dfsub(r1:0,r3:2)
define double @test3(double %rss, double %rtt) local_unnamed_addr #0 {
entry:
  %v0 = tail call double @llvm.hexagon.F2.dfsub(double %rss, double %rtt)
  ret double %v0
}

declare double @llvm.hexagon.F2.dfsub(double, double) #1

; CHECK-LABEL: @test4
; CHECK: r0 = mask(#1,#2)
define i32 @test4() local_unnamed_addr #0 {
entry:
  %v0 = tail call i32 @llvm.hexagon.S2.mask(i32 1, i32 2)
  ret i32 %v0
}

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.S2.mask(i32, i32) #1

attributes #0 = { nounwind readnone "target-cpu"="hexagonv66" "target-features"="-hvx,-long-calls" }
attributes #1 = { nounwind readnone }
