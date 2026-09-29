; RUN: llc < %s -mtriple=avr -mattr=avr6 | FileCheck %s

%str_slice = type { ptr, i16 }
%Machine = type { i16, [0 x i8], i16, [0 x i8], [16 x i8], [0 x i8] }

; CHECK-LABEL: step
define void @step(ptr) {
 ret void
}

; CHECK-LABEL: main
define void @main() {
start:
  %machine = alloca %Machine, align 8
  %v0 = bitcast ptr %machine to ptr
  %v1 = getelementptr inbounds %Machine, ptr %machine, i16 0, i32 2
  %v2 = load i16, ptr %v1, align 2
  br label %bb2.i5

bb2.i5:
  %v18 = load volatile i8, ptr inttoptr (i16 77 to ptr), align 1
  %v19 = icmp sgt i8 %v18, -1
  br i1 %v19, label %bb2.i5, label %bb.exit6

bb.exit6:
  %v20 = load volatile i8, ptr inttoptr (i16 78 to ptr), align 2
  br label %bb7

bb7:
  call void @step(ptr %machine)
  br label %bb7
}

