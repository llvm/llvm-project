; RUN: llc -march=mipsel -relocation-model=pic -O0 -fast-isel-abort=3 -mcpu=mips32r2 \
; RUN:     < %s | FileCheck %s
; RUN: llc -march=mipsel -relocation-model=pic -O0 -fast-isel-abort=3 -mcpu=mips32 \
; RUN:     < %s | FileCheck %s

@.str = private unnamed_addr constant [6 x i8] c"hello\00", align 1
@s = common global i8* null, align 4

; Function Attrs: nounwind
define void @foo() #0 {
entry:
  store i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str, i32 0, i32 0), i8** @s, align 4
  ret void
; CHECK:        .ent    foo
; CHECK:        lw      $[[REG1:[0-9]+]], %got($.str)(${{[0-9]+}})
; CHECK:        addiu   ${{[0-9]+}}, $[[REG1]], %lo($.str)

}

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

