; RUN: llc < %s -mcpu=cortex-a9 -mattr=+neon,+neonfp -relocation-model=pic

target triple = "armv6-none-linux-gnueabi"

define void @sample_test(ptr %.T0348, ptr nocapture %sourceA, ptr nocapture %destValues) {
L.entry:
  %0 = call i32 (...) @get_index(ptr %.T0348, i32 0)
  %1 = mul i32 %0, 6
  %2 = getelementptr i8, ptr %destValues, i32 %1
  %3 = load <3 x i16>, ptr %2, align 1
  %4 = getelementptr i8, ptr %sourceA, i32 %1
  %5 = load <3 x i16>, ptr %4, align 1
  %6 = or <3 x i16> %5, %3
  store <3 x i16> %6, ptr %2, align 1
  ret void
}

declare i32 @get_index(...)
