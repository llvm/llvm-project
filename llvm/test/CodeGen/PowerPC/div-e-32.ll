; RUN: llc -verify-machineinstrs -mtriple=powerpc-unknown-linux-gnu -mcpu=pwr7 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc-unknown-linux-gnu -mcpu=pwr8 < %s | FileCheck %s

; Function Attrs: nounwind
define signext i32 @test1() #0 {
entry:
  %0 = call i32 @llvm.ppc.divwe(i32 32, i32 16)
  ret i32 %0
; CHECK: divwe 3, 4, 3
}

; Function Attrs: nounwind readnone
declare i32 @llvm.ppc.divwe(i32, i32) #1

; Function Attrs: nounwind
define signext i32 @test2() #0 {
entry:
  %0 = call i32 @llvm.ppc.divweu(i32 32, i32 16)
  ret i32 %0
; CHECK: divweu 3, 4, 3
}

; Function Attrs: nounwind readnone
declare i32 @llvm.ppc.divweu(i32, i32) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.7.0 (trunk 231831) (llvm/trunk 231828:231843M)"}
