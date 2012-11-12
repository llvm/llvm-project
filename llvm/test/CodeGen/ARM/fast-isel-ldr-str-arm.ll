; RUN: llc < %s -O0 -fast-isel-abort -relocation-model=dynamic-no-pic -mtriple=thumbv7-apple-darwin | FileCheck %s --check-prefix=ARM

define i32 @t1(i32* nocapture %ptr) nounwind readonly {
entry:
; ARM: t1
  %add.ptr = getelementptr inbounds i32* %ptr, i32 1
  %0 = load i32* %add.ptr, align 4
; ARM: ldr r{{[0-9]}}, [r0, #4]
  ret i32 %0
}

define i32 @t2(i32* nocapture %ptr) nounwind readonly {
entry:
; ARM: t2
  %add.ptr = getelementptr inbounds i32* %ptr, i32 63
  %0 = load i32* %add.ptr, align 4
; ARM: ldr.w r{{[0-9]}}, [r0, #252]
  ret i32 %0
}

define zeroext i16 @t3(i16* nocapture %ptr) nounwind readonly {
entry:
; ARM: t3
  %add.ptr = getelementptr inbounds i16* %ptr, i16 1
  %0 = load i16* %add.ptr, align 4
; ARM: ldrh r{{[0-9]}}, [r0, #2]
  ret i16 %0
}

define zeroext i16 @t4(i16* nocapture %ptr) nounwind readonly {
entry:
; ARM: t4
  %add.ptr = getelementptr inbounds i16* %ptr, i16 63
  %0 = load i16* %add.ptr, align 4
; ARM: ldrh.w r{{[0-9]}}, [r0, #126]
  ret i16 %0
}

define zeroext i8 @t5(i8* nocapture %ptr) nounwind readonly {
entry:
; ARM: t5
  %add.ptr = getelementptr inbounds i8* %ptr, i8 1
  %0 = load i8* %add.ptr, align 4
; ARM: ldrb r{{[0-9]}}, [r0, #1]
  ret i8 %0
}

define zeroext i8 @t6(i8* nocapture %ptr) nounwind readonly {
entry:
; ARM: t6
  %add.ptr = getelementptr inbounds i8* %ptr, i8 63
  %0 = load i8* %add.ptr, align 4
; ARM: ldrb.w r{{[0-9]}}, [r0, #63]
  ret i8 %0
}
