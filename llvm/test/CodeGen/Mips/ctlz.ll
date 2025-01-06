; RUN: llc -mtriple=mipsel-linux-gnu -mcpu=mips32 -mattr=+mips16 -relocation-model=static < %s | FileCheck %s -check-prefix=static

@x = global i32 28912, align 4
@y = common global i32 0, align 4


; Function Attrs: nounwind
define i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  store i32 0, ptr %retval
  %0 = load i32, ptr @x, align 4
  %1 = call i32 @llvm.ctlz.i32(i32 %0, i1 true)
  store i32 %1, ptr @y, align 4
  ret i32 0
}

; static: .end main

; Function Attrs: nounwind readnone
declare i32 @llvm.ctlz.i32(i32, i1) #1



attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="true" }
attributes #1 = { nounwind readnone }

