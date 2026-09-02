; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

declare i8 @func()

define void @test(ptr %x) {
entry:
  ; CHECK: Ranges are only for loads, calls and invokes!
  store i8 0, ptr %x, align 1, !range !{i8 0, i8 1}

  ; CHECK: It should have at least one range!
  load i8, ptr %x, align 1, !range !{}

  ; CHECK: Unfinished range!
  load i8, ptr %x, align 1, !range !{i8 0}

  ; CHECK: The lower limit must be an integer!
  load i8, ptr %x, align 1, !range !{double 0.0, i8 0}

  ; CHECK: The upper limit must be an integer!
  load i8, ptr %x, align 1, !range !{i8 0, double 0.0}

  ; CHECK: Range pair types must match!
  load i8, ptr %x, align 1, !range !{i32 0, i8 0}

  ; CHECK: Range pair types must match!
  load i8, ptr %x, align 1, !range !{i8 0, i32 0}

  ; CHECK: Range types must match instruction type!
  load i8, ptr %x, align 1, !range !{i32 0, i32 0}

  ; CHECK: Range must not be empty!
  load i8, ptr %x, align 1, !range !{i8 0, i8 0}

  ; CHECK: Intervals are overlapping
  load i8, ptr %x, align 1, !range !{i8 0, i8 2, i8 1, i8 3}

  ; CHECK: Intervals are overlapping
  load i8, ptr %x, align 1, !range !{i8 1, i8 3, i8 5, i8 2}

  ; CHECK: Intervals are overlapping
  load i8, ptr %x, align 1, !range !{i8 10, i8 1, i8 12, i8 13}

  ; CHECK: Intervals are overlapping
  load i8, ptr %x, align 1, !range !{i8 1, i8 3, i8 4, i8 5, i8 6, i8 2}

  ; CHECK: Intervals are contiguous
  load i8, ptr %x, align 1, !range !{i8 0, i8 2, i8 2, i8 3}

  ; CHECK: Intervals are contiguous
  load i8, ptr %x, align 1, !range !{i8 1, i8 3, i8 5, i8 1}

  ; CHECK: Intervals are contiguous
  load i8, ptr %x, align 1, !range !{i8 1, i8 3, i8 4, i8 5, i8 6, i8 1}

  ; CHECK: Intervals are not in order
  load i8, ptr %x, align 1, !range !{i8 1, i8 2, i8 -1, i8 0}

  ; CHECK: It should have at least one range!
  call i8 @func(), !range !{}

  ; CHECK: Range types must match instruction type!
  load <2 x i8>, ptr %x, !range !{i16 0, i16 10}

  ; CHECK: The upper and lower limits cannot be the same value{{$}}
  load i32, ptr %x, !range !{i32 123, i32 123}

  ret void
}
