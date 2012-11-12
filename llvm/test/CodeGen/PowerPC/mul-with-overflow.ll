; RUN: llc < %s -march=ppc32

declare {i32, i1} @llvm.umul.with.overflow.i32(i32 %a, i32 %b)
define zeroext i1 @a(i32 %x)  nounwind {
  %res = call {i32, i1} @llvm.umul.with.overflow.i32(i32 %x, i32 3)
  %obil = extractvalue {i32, i1} %res, 1
  ret i1 %obil
}

declare {i32, i1} @llvm.smul.with.overflow.i32(i32 %a, i32 %b)
define zeroext i1 @b(i32 %x)  nounwind {
  %res = call {i32, i1} @llvm.smul.with.overflow.i32(i32 %x, i32 3)
  %obil = extractvalue {i32, i1} %res, 1
  ret i1 %obil
}
