; RUN: llc -mtriple thumbv7-windows-gnu -filetype asm -o - %s

define i32 @divoverflow32(i32 %a, i32 %b) {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = load i32, i32* %1, align 4
  %4 = load i32, i32* %2, align 4
  %5 = sub nsw i32 0, %4
  %6 = sdiv i32 -2147483647, %3
  %7 = icmp sgt i32 %5, %6
  br i1 %7, label %8, label %9
  call void (...) @abort_impl32()
  unreachable
  %10 = load i32, i32* %1, align 4
  %11 = load i32, i32* %2, align 4
  %12 = mul nsw i32 %10, %11
  ret i32 %12
}

declare void @abort_impl32(...)

define i64 @divoverflow64(i64 %a, i64 %b) {
  %1 = alloca i64, align 8
  %2 = alloca i64, align 8
  %3 = load i64, i64* %1, align 8
  %4 = load i64, i64* %2, align 8
  %5 = sub nsw i64 0, %4
  %6 = sdiv i64 -9223372036854775808, %3
  %7 = icmp sgt i64 %5, %6
  br i1 %7, label %8, label %9
  call void (...) @abort_impl64()
  unreachable
  %10 = load i64, i64* %1, align 8
  %11 = load i64, i64* %2, align 8
  %12 = mul nsw i64 %10, %11
  ret i64 %12
}

declare void @abort_impl64(...)
