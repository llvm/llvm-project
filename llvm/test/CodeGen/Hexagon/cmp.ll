; RUN: llc -march=hexagon --filetype=obj < %s  -o - | llvm-objdump --no-print-imm-hex -d - | FileCheck %s

; Function Attrs: nounwind
define i32 @cmpeq(i32 %i) #0 {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  %0 = load i32, i32* %i.addr, align 4
  %1 = call i32 @llvm.hexagon.C2.cmpeq(i32 %0, i32 1)
  ret i32 %1
}
; CHECK: { p{{[0-3]}} = cmp.eq(r{{[0-9]}},r{{[0-9]}})

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.C2.cmpeq(i32, i32) #1

; Function Attrs: nounwind
define i32 @cmpgt(i32 %i) #0 {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  %0 = load i32, i32* %i.addr, align 4
  %1 = call i32 @llvm.hexagon.C2.cmpgt(i32 %0, i32 2)
  ret i32 %1
}
; CHECK: { p{{[0-3]}} = cmp.gt(r{{[0-9]}},r{{[0-9]}})

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.C2.cmpgt(i32, i32) #1

; Function Attrs: nounwind
define i32 @cmpgtu(i32 %i) #0 {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  %0 = load i32, i32* %i.addr, align 4
  %1 = call i32 @llvm.hexagon.C2.cmpgtu(i32 %0, i32 3)
  ret i32 %1
}
; CHECK: { p{{[0-3]}} = cmp.gtu(r{{[0-9]}},r{{[0-9]}})

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.C2.cmpgtu(i32, i32) #1

; Function Attrs: nounwind
define i32 @cmplt(i32 %i) #0 {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  %0 = load i32, i32* %i.addr, align 4
  %1 = call i32 @llvm.hexagon.C2.cmplt(i32 %0, i32 4)
  ret i32 %1
}
; CHECK: { p{{[0-3]}} = cmp.gt(r{{[0-9]}},r{{[0-9]}})

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.C2.cmplt(i32, i32) #1

; Function Attrs: nounwind
define i32 @cmpltu(i32 %i) #0 {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  %0 = load i32, i32* %i.addr, align 4
  %1 = call i32 @llvm.hexagon.C2.cmpltu(i32 %0, i32 5)
  ret i32 %1
}
; CHECK: { p{{[0-3]}} = cmp.gtu(r{{[0-9]}},r{{[0-9]}})

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.C2.cmpltu(i32, i32) #1

; Function Attrs: nounwind
define i32 @cmpeqi(i32 %i) #0 {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  %0 = load i32, i32* %i.addr, align 4
  %1 = call i32 @llvm.hexagon.C2.cmpeqi(i32 %0, i32 10)
  ret i32 %1
}
; CHECK: { p{{[0-3]}} = cmp.eq(r{{[0-9]}},#10)

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.C2.cmpeqi(i32, i32) #1

; Function Attrs: nounwind
define i32 @cmpgti(i32 %i) #0 {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  %0 = load i32, i32* %i.addr, align 4
  %1 = call i32 @llvm.hexagon.C2.cmpgti(i32 %0, i32 20)
  ret i32 %1
}
; CHECK: { p{{[0-3]}} = cmp.gt(r{{[0-9]}},#20)

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.C2.cmpgti(i32, i32) #1

; Function Attrs: nounwind
define i32 @cmpgtui(i32 %i) #0 {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  %0 = load i32, i32* %i.addr, align 4
  %1 = call i32 @llvm.hexagon.C2.cmpgtui(i32 %0, i32 40)
  ret i32 %1
}
; CHECK: { p{{[0-3]}} = cmp.gtu(r{{[0-9]}},#40)

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.C2.cmpgtui(i32, i32) #1

; Function Attrs: nounwind
define i32 @cmpgei(i32 %i) #0 {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  %0 = load i32, i32* %i.addr, align 4
  %1 = call i32 @llvm.hexagon.C2.cmpgei(i32 %0, i32 3)
  ret i32 %1
}
; CHECK: { p{{[0-3]}} = cmp.gt(r{{[0-9]}},#2)

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.C2.cmpgei(i32, i32) #1

; Function Attrs: nounwind
define i32 @cmpgeu(i32 %i) #0 {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  %0 = load i32, i32* %i.addr, align 4
  %1 = call i32 @llvm.hexagon.C2.cmpgeui(i32 %0, i32 3)
  ret i32 %1
}
; CHECK: { p{{[0-3]}} = cmp.gtu(r{{[0-9]}},#2)

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.C2.cmpgeui(i32, i32) #1

; Function Attrs: nounwind
define i32 @cmpgeu0(i32 %i) #0 {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  %0 = load i32, i32* %i.addr, align 4
  %1 = call i32 @llvm.hexagon.C2.cmpgeui(i32 %0, i32 0)
  ret i32 %1
}
; CHECK: { p{{[0-3]}} = cmp.eq(r{{[0-9]}},r{{[0-9]}})


attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.ident = !{!0}

!0 = !{!"Clang 3.1"}

