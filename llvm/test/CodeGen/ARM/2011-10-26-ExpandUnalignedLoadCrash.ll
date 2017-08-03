; RUN: llc < %s -march=arm -mcpu=cortex-a9 -mattr=+neon,+neonfp -relocation-model=pic

target triple = "armv6-none-linux-gnueabi"

define void @sample_test(i8* %.T0348, i16* nocapture %sourceA, i16* nocapture %destValues) {
L.entry:
  %0 = call i32 (...) @get_index(i8* %.T0348, i32 0)
  %1 = bitcast i16* %destValues to i8*
  %2 = mul i32 %0, 6
  %3 = getelementptr i8, i8* %1, i32 %2
  %4 = bitcast i8* %3 to <3 x i16>*
  %5 = load <3 x i16>, <3 x i16>* %4, align 1
  %6 = bitcast i16* %sourceA to i8*
  %7 = getelementptr i8, i8* %6, i32 %2
  %8 = bitcast i8* %7 to <3 x i16>*
  %9 = load <3 x i16>, <3 x i16>* %8, align 1
  %10 = or <3 x i16> %9, %5
  store <3 x i16> %10, <3 x i16>* %4, align 1
  ret void
}

declare i32 @get_index(...)
