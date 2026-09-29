; RUN: llc < %s
; Verify codegen's don't crash on overflow intrinsics.

;; SADD

define zeroext i8 @sadd_i8(i8 signext %a, i8 signext %b) nounwind ssp {
entry:
  %sadd = tail call { i8, i1 } @llvm.sadd.with.overflow.i8(i8 %a, i8 %b)
  %cmp = extractvalue { i8, i1 } %sadd, 1
  %sadd.result = extractvalue { i8, i1 } %sadd, 0
  %X = select i1 %cmp, i8 %sadd.result, i8 42
  ret i8 %X
}

declare { i8, i1 } @llvm.sadd.with.overflow.i8(i8, i8) nounwind readnone

define zeroext i16 @sadd_i16(i16 signext %a, i16 signext %b) nounwind ssp {
entry:
  %sadd = tail call { i16, i1 } @llvm.sadd.with.overflow.i16(i16 %a, i16 %b)
  %cmp = extractvalue { i16, i1 } %sadd, 1
  %sadd.result = extractvalue { i16, i1 } %sadd, 0
  %X = select i1 %cmp, i16 %sadd.result, i16 42
  ret i16 %X
}

declare { i16, i1 } @llvm.sadd.with.overflow.i16(i16, i16) nounwind readnone

define zeroext i32 @sadd_i32(i32 signext %a, i32 signext %b) nounwind ssp {
entry:
  %sadd = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %a, i32 %b)
  %cmp = extractvalue { i32, i1 } %sadd, 1
  %sadd.result = extractvalue { i32, i1 } %sadd, 0
  %X = select i1 %cmp, i32 %sadd.result, i32 42
  ret i32 %X
}

declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32) nounwind readnone


;; UADD

define zeroext i8 @uadd_i8(i8 signext %a, i8 signext %b) nounwind ssp {
entry:
  %uadd = tail call { i8, i1 } @llvm.uadd.with.overflow.i8(i8 %a, i8 %b)
  %cmp = extractvalue { i8, i1 } %uadd, 1
  %uadd.result = extractvalue { i8, i1 } %uadd, 0
  %X = select i1 %cmp, i8 %uadd.result, i8 42
  ret i8 %X
}

declare { i8, i1 } @llvm.uadd.with.overflow.i8(i8, i8) nounwind readnone

define zeroext i16 @uadd_i16(i16 signext %a, i16 signext %b) nounwind ssp {
entry:
  %uadd = tail call { i16, i1 } @llvm.uadd.with.overflow.i16(i16 %a, i16 %b)
  %cmp = extractvalue { i16, i1 } %uadd, 1
  %uadd.result = extractvalue { i16, i1 } %uadd, 0
  %X = select i1 %cmp, i16 %uadd.result, i16 42
  ret i16 %X
}

declare { i16, i1 } @llvm.uadd.with.overflow.i16(i16, i16) nounwind readnone

define zeroext i32 @uadd_i32(i32 signext %a, i32 signext %b) nounwind ssp {
entry:
  %uadd = tail call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %a, i32 %b)
  %cmp = extractvalue { i32, i1 } %uadd, 1
  %uadd.result = extractvalue { i32, i1 } %uadd, 0
  %X = select i1 %cmp, i32 %uadd.result, i32 42
  ret i32 %X
}

declare { i32, i1 } @llvm.uadd.with.overflow.i32(i32, i32) nounwind readnone



;; ssub

define zeroext i8 @ssub_i8(i8 signext %a, i8 signext %b) nounwind ssp {
entry:
  %ssub = tail call { i8, i1 } @llvm.ssub.with.overflow.i8(i8 %a, i8 %b)
  %cmp = extractvalue { i8, i1 } %ssub, 1
  %ssub.result = extractvalue { i8, i1 } %ssub, 0
  %X = select i1 %cmp, i8 %ssub.result, i8 42
  ret i8 %X
}

declare { i8, i1 } @llvm.ssub.with.overflow.i8(i8, i8) nounwind readnone

define zeroext i16 @ssub_i16(i16 signext %a, i16 signext %b) nounwind ssp {
entry:
  %ssub = tail call { i16, i1 } @llvm.ssub.with.overflow.i16(i16 %a, i16 %b)
  %cmp = extractvalue { i16, i1 } %ssub, 1
  %ssub.result = extractvalue { i16, i1 } %ssub, 0
  %X = select i1 %cmp, i16 %ssub.result, i16 42
  ret i16 %X
}

declare { i16, i1 } @llvm.ssub.with.overflow.i16(i16, i16) nounwind readnone

define zeroext i32 @ssub_i32(i32 signext %a, i32 signext %b) nounwind ssp {
entry:
  %ssub = tail call { i32, i1 } @llvm.ssub.with.overflow.i32(i32 %a, i32 %b)
  %cmp = extractvalue { i32, i1 } %ssub, 1
  %ssub.result = extractvalue { i32, i1 } %ssub, 0
  %X = select i1 %cmp, i32 %ssub.result, i32 42
  ret i32 %X
}

declare { i32, i1 } @llvm.ssub.with.overflow.i32(i32, i32) nounwind readnone


;; usub

define zeroext i8 @usub_i8(i8 signext %a, i8 signext %b) nounwind ssp {
entry:
  %usub = tail call { i8, i1 } @llvm.usub.with.overflow.i8(i8 %a, i8 %b)
  %cmp = extractvalue { i8, i1 } %usub, 1
  %usub.result = extractvalue { i8, i1 } %usub, 0
  %X = select i1 %cmp, i8 %usub.result, i8 42
  ret i8 %X
}

declare { i8, i1 } @llvm.usub.with.overflow.i8(i8, i8) nounwind readnone

define zeroext i16 @usub_i16(i16 signext %a, i16 signext %b) nounwind ssp {
entry:
  %usub = tail call { i16, i1 } @llvm.usub.with.overflow.i16(i16 %a, i16 %b)
  %cmp = extractvalue { i16, i1 } %usub, 1
  %usub.result = extractvalue { i16, i1 } %usub, 0
  %X = select i1 %cmp, i16 %usub.result, i16 42
  ret i16 %X
}

declare { i16, i1 } @llvm.usub.with.overflow.i16(i16, i16) nounwind readnone

define zeroext i32 @usub_i32(i32 signext %a, i32 signext %b) nounwind ssp {
entry:
  %usub = tail call { i32, i1 } @llvm.usub.with.overflow.i32(i32 %a, i32 %b)
  %cmp = extractvalue { i32, i1 } %usub, 1
  %usub.result = extractvalue { i32, i1 } %usub, 0
  %X = select i1 %cmp, i32 %usub.result, i32 42
  ret i32 %X
}

declare { i32, i1 } @llvm.usub.with.overflow.i32(i32, i32) nounwind readnone



;; smul

define zeroext i8 @smul_i8(i8 signext %a, i8 signext %b) nounwind ssp {
entry:
  %smul = tail call { i8, i1 } @llvm.smul.with.overflow.i8(i8 %a, i8 %b)
  %cmp = extractvalue { i8, i1 } %smul, 1
  %smul.result = extractvalue { i8, i1 } %smul, 0
  %X = select i1 %cmp, i8 %smul.result, i8 42
  ret i8 %X
}

declare { i8, i1 } @llvm.smul.with.overflow.i8(i8, i8) nounwind readnone

define zeroext i16 @smul_i16(i16 signext %a, i16 signext %b) nounwind ssp {
entry:
  %smul = tail call { i16, i1 } @llvm.smul.with.overflow.i16(i16 %a, i16 %b)
  %cmp = extractvalue { i16, i1 } %smul, 1
  %smul.result = extractvalue { i16, i1 } %smul, 0
  %X = select i1 %cmp, i16 %smul.result, i16 42
  ret i16 %X
}

declare { i16, i1 } @llvm.smul.with.overflow.i16(i16, i16) nounwind readnone

define zeroext i32 @smul_i32(i32 signext %a, i32 signext %b) nounwind ssp {
entry:
  %smul = tail call { i32, i1 } @llvm.smul.with.overflow.i32(i32 %a, i32 %b)
  %cmp = extractvalue { i32, i1 } %smul, 1
  %smul.result = extractvalue { i32, i1 } %smul, 0
  %X = select i1 %cmp, i32 %smul.result, i32 42
  ret i32 %X
}

declare { i32, i1 } @llvm.smul.with.overflow.i32(i32, i32) nounwind readnone


;; umul

define zeroext i8 @umul_i8(i8 signext %a, i8 signext %b) nounwind ssp {
entry:
  %umul = tail call { i8, i1 } @llvm.umul.with.overflow.i8(i8 %a, i8 %b)
  %cmp = extractvalue { i8, i1 } %umul, 1
  %umul.result = extractvalue { i8, i1 } %umul, 0
  %X = select i1 %cmp, i8 %umul.result, i8 42
  ret i8 %X
}

declare { i8, i1 } @llvm.umul.with.overflow.i8(i8, i8) nounwind readnone

define zeroext i16 @umul_i16(i16 signext %a, i16 signext %b) nounwind ssp {
entry:
  %umul = tail call { i16, i1 } @llvm.umul.with.overflow.i16(i16 %a, i16 %b)
  %cmp = extractvalue { i16, i1 } %umul, 1
  %umul.result = extractvalue { i16, i1 } %umul, 0
  %X = select i1 %cmp, i16 %umul.result, i16 42
  ret i16 %X
}

declare { i16, i1 } @llvm.umul.with.overflow.i16(i16, i16) nounwind readnone

define zeroext i32 @umul_i32(i32 signext %a, i32 signext %b) nounwind ssp {
entry:
  %umul = tail call { i32, i1 } @llvm.umul.with.overflow.i32(i32 %a, i32 %b)
  %cmp = extractvalue { i32, i1 } %umul, 1
  %umul.result = extractvalue { i32, i1 } %umul, 0
  %X = select i1 %cmp, i32 %umul.result, i32 42
  ret i32 %X
}

declare { i32, i1 } @llvm.umul.with.overflow.i32(i32, i32) nounwind readnone

