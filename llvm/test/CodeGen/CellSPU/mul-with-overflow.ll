; RUN: llc < %s -march=cellspu

declare {i16, i1} @llvm.smul.with.overflow.i16(i16 %a, i16 %b)
define zeroext i1 @a(i16 %x)  nounwind {
  %res = call {i16, i1} @llvm.smul.with.overflow.i16(i16 %x, i16 3)
  %obil = extractvalue {i16, i1} %res, 1
  ret i1 %obil
}

declare {i16, i1} @llvm.umul.with.overflow.i16(i16 %a, i16 %b)
define zeroext i1 @b(i16 %x)  nounwind {
  %res = call {i16, i1} @llvm.umul.with.overflow.i16(i16 %x, i16 3)
  %obil = extractvalue {i16, i1} %res, 1
  ret i1 %obil
}
