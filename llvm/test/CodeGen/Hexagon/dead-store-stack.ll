; RUN: llc -O2 -march=hexagon -mcpu=hexagonv62< %s | FileCheck %s
; CHECK: ParseFunc:
; CHECK: r[[ARG0:[0-9]+]] = memuh(r[[ARG1:[0-9]+]]+#[[OFFSET:[0-9]+]])
; CHECK: memw(r[[ARG1]]+#[[OFFSET]]) = r[[ARG0]]

@.str.3 = external unnamed_addr constant [8 x i8], align 1
; Function Attrs: nounwind
define void @ParseFunc() local_unnamed_addr #0 {
entry:
  %dataVar = alloca i32, align 4
  %0 = load i32, i32* %dataVar, align 4
  %and = and i32 %0, 65535
  store i32 %and, i32* %dataVar, align 4
  %.pr = load i32, i32* %dataVar, align 4
  switch i32 %.pr, label %sw.epilog [
    i32 4, label %sw.bb
    i32 5, label %sw.bb
    i32 1, label %sw.bb39
    i32 2, label %sw.bb40
    i32 3, label %sw.bb41
    i32 6, label %sw.bb42
    i32 7, label %sw.bb43
    i32 13, label %sw.bb44
    i32 0, label %sw.bb44
    i32 14, label %sw.bb45
    i32 15, label %sw.bb46
  ]

sw.bb:
  %cmp1.i = icmp eq i32 %.pr, 4
  br label %land.rhs.i

land.rhs.i:
  br label %ParseFuncNext.exit.i

ParseFuncNext.exit.i:
  br i1 %cmp1.i, label %if.then.i, label %if.else10.i

if.then.i:
  call void (i8*, i32, i8*, ...) @snprintf(i8* undef, i32 undef, i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.3, i32 0, i32 0), i32 undef) #2
  br label %if.end27.i

if.else10.i:
  unreachable

if.end27.i:
  br label %land.rhs.i

sw.bb39:
  unreachable

sw.bb40:
  unreachable

sw.bb41:
  unreachable

sw.bb42:
  %1 = load i32, i32* undef, align 4
  %shr.i = lshr i32 %1, 16
  br label %while.cond.i.i

while.cond.i.i:
  %2 = load i8, i8* undef, align 1
  switch i8 %2, label %if.then4.i [
    i8 48, label %land.end.i.i
    i8 120, label %land.end.i.i
    i8 37, label %do.body.i.i
  ]

land.end.i.i:
  unreachable

do.body.i.i:
  switch i8 undef, label %if.then4.i [
    i8 117, label %if.end40.i.i
    i8 120, label %if.end40.i.i
    i8 88, label %if.end40.i.i
    i8 100, label %if.end40.i.i
    i8 105, label %if.end40.i.i
  ]

if.end40.i.i:
  %trunc.i = trunc i32 %shr.i to i16
  br label %land.rhs.i126

if.then4.i:
  unreachable

land.rhs.i126:
  switch i16 %trunc.i, label %sw.epilog.i [
    i16 1, label %sw.bb.i
    i16 2, label %sw.bb12.i
    i16 4, label %sw.bb16.i
  ]

sw.bb.i:
  unreachable

sw.bb12.i:
  unreachable

sw.bb16.i:
  unreachable

sw.epilog.i:
  call void (i8*, i32, i8*, ...) @snprintf(i8* undef, i32 undef, i8* nonnull undef, i32 undef) #2
  br label %land.rhs.i126

sw.bb43:
  unreachable

sw.bb44:
  unreachable

sw.bb45:
  unreachable

sw.bb46:
  unreachable

sw.epilog:
  ret void
}

; Function Attrs: nounwind
declare void @snprintf(i8* nocapture, i32, i8* nocapture readonly, ...) local_unnamed_addr #1

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv62" "target-features"="+hvx,+hvx-length64b" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv62" "target-features"="+hvx,+hvx-length64b" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

