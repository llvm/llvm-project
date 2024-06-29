; RUN: llc -march=msp430 < %s
%VeryLarge = type { i8, i32, i32, i32, i32, i32, i32, i32, i32, i32 }

; intentionally cause a spill
define void @inc(ptr byval(%VeryLarge) align 1 %s) {
entry:
  %p0 = getelementptr inbounds %VeryLarge, ptr %s, i32 0, i32 0
  %0 = load i8, ptr %p0
  %p1 = getelementptr inbounds %VeryLarge, ptr %s, i32 0, i32 1
  %1 = load i32, ptr %p1
  %p2 = getelementptr inbounds %VeryLarge, ptr %s, i32 0, i32 2
  %2 = load i32, ptr %p2
  %p3 = getelementptr inbounds %VeryLarge, ptr %s, i32 0, i32 3
  %3 = load i32, ptr %p3
  %p4 = getelementptr inbounds %VeryLarge, ptr %s, i32 0, i32 4
  %4 = load i32, ptr %p4
  %p5 = getelementptr inbounds %VeryLarge, ptr %s, i32 0, i32 5
  %5 = load i32, ptr %p5
  %p6 = getelementptr inbounds %VeryLarge, ptr %s, i32 0, i32 6
  %6 = load i32, ptr %p6
  %p7 = getelementptr inbounds %VeryLarge, ptr %s, i32 0, i32 7
  %7 = load i32, ptr %p7
  %add = add i8 %0, 1
  store i8 %add, ptr %p0
  %add2 = add i32 %1, 2
  store i32 %add2, ptr %p1
  %add3 = add i32 %2, 3
  store i32 %add3, ptr %p2
  %add4 = add i32 %3, 4
  store i32 %add4, ptr %p3
  %add5 = add i32 %4, 5
  store i32 %add5, ptr %p4
  %add6 = add i32 %5, 6
  store i32 %add6, ptr %p5
  %add7 = add i32 %6, 7
  store i32 %add7, ptr %p6
  %add8 = add i32 %7, 8
  store i32 %add8, ptr %p7
  ret void
}
