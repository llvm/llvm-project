; RUN: llc -filetype=obj -march=hexagon %s -o - | llvm-objdump -r - | FileCheck %s

; CHECK: RELOCATION RECORDS FOR [.rela.text]:
; CHECK: 00000000 R_HEX_B22_PCREL printf
; CHECK: 00000004 R_HEX_32_6_X .rodata.str1.1
; CHECK: 00000008 R_HEX_6_X .rodata.str1.1

target triple = "hexagon-unknown--elf"

@.str = private unnamed_addr constant [10 x i8] c"cxfir.log\00", align 1

declare i32 @printf(i8*, ...) #1

; Function Attrs: nounwind
define i32 @main() #0 {
entry:
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str, i32 0, i32 0))
  ret i32 0
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

