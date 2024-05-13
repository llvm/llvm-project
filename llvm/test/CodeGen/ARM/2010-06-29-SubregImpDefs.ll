; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o /dev/null

@.str271 = external constant [21 x i8], align 4   ; <ptr> [#uses=1]
@llvm.used = appending global [1 x ptr] [ptr @main], section "llvm.metadata" ; <ptr> [#uses=0]

define i32 @main(i32 %argc, ptr %argv) nounwind {
entry:
  %0 = shufflevector <2 x i64> undef, <2 x i64> zeroinitializer, <2 x i32> <i32 1, i32 2> ; <<2 x i64>> [#uses=1]
  store <2 x i64> %0, ptr undef, align 16
  %val4723 = load <8 x i16>, ptr undef                ; <<8 x i16>> [#uses=1]
  call void @PrintShortX(ptr @.str271, <8 x i16> %val4723, i32 0) nounwind
  ret i32 undef
}

declare void @PrintShortX(ptr, <8 x i16>, i32) nounwind
