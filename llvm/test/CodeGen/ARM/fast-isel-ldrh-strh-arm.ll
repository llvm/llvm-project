; RUN: llc < %s -O0 -fast-isel-abort -relocation-model=dynamic-no-pic -mtriple=armv7-apple-ios | FileCheck %s --check-prefix=ARM
; rdar://10418009

define zeroext i16 @t1(i16* nocapture %a) nounwind uwtable readonly ssp {
entry:
; ARM: t1
  %add.ptr = getelementptr inbounds i16* %a, i64 -8
  %0 = load i16* %add.ptr, align 2
; ARM: ldrh r0, [r0, #-16]
  ret i16 %0
}

define zeroext i16 @t2(i16* nocapture %a) nounwind uwtable readonly ssp {
entry:
; ARM: t2
  %add.ptr = getelementptr inbounds i16* %a, i64 -16
  %0 = load i16* %add.ptr, align 2
; ARM: ldrh r0, [r0, #-32]
  ret i16 %0
}

define zeroext i16 @t3(i16* nocapture %a) nounwind uwtable readonly ssp {
entry:
; ARM: t3
  %add.ptr = getelementptr inbounds i16* %a, i64 -127
  %0 = load i16* %add.ptr, align 2
; ARM: ldrh r0, [r0, #-254]
  ret i16 %0
}

define zeroext i16 @t4(i16* nocapture %a) nounwind uwtable readonly ssp {
entry:
; ARM: t4
  %add.ptr = getelementptr inbounds i16* %a, i64 -128
  %0 = load i16* %add.ptr, align 2
; ARM: mvn r{{[1-9]}}, #255
; ARM: add r0, r0, r{{[1-9]}}
; ARM: ldrh r0, [r0]
  ret i16 %0
}

define zeroext i16 @t5(i16* nocapture %a) nounwind uwtable readonly ssp {
entry:
; ARM: t5
  %add.ptr = getelementptr inbounds i16* %a, i64 8
  %0 = load i16* %add.ptr, align 2
; ARM: ldrh r0, [r0, #16]
  ret i16 %0
}

define zeroext i16 @t6(i16* nocapture %a) nounwind uwtable readonly ssp {
entry:
; ARM: t6
  %add.ptr = getelementptr inbounds i16* %a, i64 16
  %0 = load i16* %add.ptr, align 2
; ARM: ldrh r0, [r0, #32]
  ret i16 %0
}

define zeroext i16 @t7(i16* nocapture %a) nounwind uwtable readonly ssp {
entry:
; ARM: t7
  %add.ptr = getelementptr inbounds i16* %a, i64 127
  %0 = load i16* %add.ptr, align 2
; ARM: ldrh r0, [r0, #254]
  ret i16 %0
}

define zeroext i16 @t8(i16* nocapture %a) nounwind uwtable readonly ssp {
entry:
; ARM: t8
  %add.ptr = getelementptr inbounds i16* %a, i64 128
  %0 = load i16* %add.ptr, align 2
; ARM: add r0, r0, #256
; ARM: ldrh r0, [r0]
  ret i16 %0
}

define void @t9(i16* nocapture %a) nounwind uwtable ssp {
entry:
; ARM: t9
  %add.ptr = getelementptr inbounds i16* %a, i64 -8
  store i16 0, i16* %add.ptr, align 2
; ARM: strh	r1, [r0, #-16]
  ret void
}

; mvn r1, #255
; strh r2, [r0, r1]
define void @t10(i16* nocapture %a) nounwind uwtable ssp {
entry:
; ARM: t10
  %add.ptr = getelementptr inbounds i16* %a, i64 -128
  store i16 0, i16* %add.ptr, align 2
; ARM: mvn r{{[1-9]}}, #255
; ARM: add r0, r0, r{{[1-9]}}
; ARM: strh r{{[1-9]}}, [r0]
  ret void
}

define void @t11(i16* nocapture %a) nounwind uwtable ssp {
entry:
; ARM: t11
  %add.ptr = getelementptr inbounds i16* %a, i64 8
  store i16 0, i16* %add.ptr, align 2
; ARM strh r{{[1-9]}}, [r0, #16]
  ret void
}

; mov r1, #256
; strh r2, [r0, r1]
define void @t12(i16* nocapture %a) nounwind uwtable ssp {
entry:
; ARM: t12
  %add.ptr = getelementptr inbounds i16* %a, i64 128
  store i16 0, i16* %add.ptr, align 2
; ARM: add r0, r0, #256
; ARM: strh r{{[1-9]}}, [r0]
  ret void
}

define signext i8 @t13(i8* nocapture %a) nounwind uwtable readonly ssp {
entry:
; ARM: t13
  %add.ptr = getelementptr inbounds i8* %a, i64 -8
  %0 = load i8* %add.ptr, align 2
; ARM: ldrsb r0, [r0, #-8]
  ret i8 %0
}

define signext i8 @t14(i8* nocapture %a) nounwind uwtable readonly ssp {
entry:
; ARM: t14
  %add.ptr = getelementptr inbounds i8* %a, i64 -255
  %0 = load i8* %add.ptr, align 2
; ARM: ldrsb r0, [r0, #-255]
  ret i8 %0
}

define signext i8 @t15(i8* nocapture %a) nounwind uwtable readonly ssp {
entry:
; ARM: t15
  %add.ptr = getelementptr inbounds i8* %a, i64 -256
  %0 = load i8* %add.ptr, align 2
; ARM: mvn r{{[1-9]}}, #255
; ARM: add r0, r0, r{{[1-9]}}
; ARM: ldrsb r0, [r0]
  ret i8 %0
}
