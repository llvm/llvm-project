; RUN: llc < %s -march=avr -mattr=avr6 | FileCheck %s

%str_slice = type { i8*, i16 }
%Machine = type { i16, [0 x i8], i16, [0 x i8], [16 x i8], [0 x i8] }

; CHECK-LABEL: step
define void @step(%Machine*) {
 ret void
}

; CHECK-LABEL: main
define void @main() {
start:
  %machine = alloca %Machine, align 8
  %v0 = bitcast %Machine* %machine to i8*
  %v1 = getelementptr inbounds %Machine, %Machine* %machine, i16 0, i32 2
  %v2 = load i16, i16* %v1, align 2
  br label %bb2.i5

bb2.i5:
  %v18 = load volatile i8, i8* inttoptr (i16 77 to i8*), align 1
  %v19 = icmp sgt i8 %v18, -1
  br i1 %v19, label %bb2.i5, label %bb.exit6

bb.exit6:
  %v20 = load volatile i8, i8* inttoptr (i16 78 to i8*), align 2
  br label %bb7

bb7:
  call void @step(%Machine* %machine)
  br label %bb7
}

