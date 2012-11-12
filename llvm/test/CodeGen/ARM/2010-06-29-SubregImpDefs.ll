; RUN: llc < %s -march=arm -mattr=+neon

@.str271 = external constant [21 x i8], align 4   ; <[21 x i8]*> [#uses=1]
@llvm.used = appending global [1 x i8*] [i8* bitcast (i32 (i32, i8**)* @main to i8*)], section "llvm.metadata" ; <[1 x i8*]*> [#uses=0]

define i32 @main(i32 %argc, i8** %argv) nounwind {
entry:
  %0 = shufflevector <2 x i64> undef, <2 x i64> zeroinitializer, <2 x i32> <i32 1, i32 2> ; <<2 x i64>> [#uses=1]
  store <2 x i64> %0, <2 x i64>* undef, align 16
  %val4723 = load <8 x i16>* undef                ; <<8 x i16>> [#uses=1]
  call void @PrintShortX(i8* getelementptr inbounds ([21 x i8]* @.str271, i32 0, i32 0), <8 x i16> %val4723, i32 0) nounwind
  ret i32 undef
}

declare void @PrintShortX(i8*, <8 x i16>, i32) nounwind
