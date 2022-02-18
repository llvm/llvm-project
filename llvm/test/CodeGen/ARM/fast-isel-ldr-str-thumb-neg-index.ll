; RUN: llc < %s -O0 -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=thumbv7-apple-ios -verify-machineinstrs | FileCheck %s --check-prefix=THUMB

define i32 @t1(i32* nocapture %ptr) nounwind readonly {
entry:
; THUMB-LABEL: t1:
  %add.ptr = getelementptr inbounds i32, i32* %ptr, i32 -1
  %0 = load i32, i32* %add.ptr, align 4
; THUMB: ldr r{{[0-9]}}, [r0, #-4]
  ret i32 %0
}

define i32 @t2(i32* nocapture %ptr) nounwind readonly {
entry:
; THUMB-LABEL: t2:
  %add.ptr = getelementptr inbounds i32, i32* %ptr, i32 -63
  %0 = load i32, i32* %add.ptr, align 4
; THUMB: ldr r{{[0-9]}}, [r0, #-252]
  ret i32 %0
}

define i32 @t3(i32* nocapture %ptr) nounwind readonly {
entry:
; THUMB-LABEL: t3:
  %add.ptr = getelementptr inbounds i32, i32* %ptr, i32 -64
  %0 = load i32, i32* %add.ptr, align 4
; THUMB: ldr r{{[0-9]}}, [r0]
  ret i32 %0
}

define zeroext i16 @t4(i16* nocapture %ptr) nounwind readonly {
entry:
; THUMB-LABEL: t4:
  %add.ptr = getelementptr inbounds i16, i16* %ptr, i32 -1
  %0 = load i16, i16* %add.ptr, align 2
; THUMB: ldrh r{{[0-9]}}, [r0, #-2]
  ret i16 %0
}

define zeroext i16 @t5(i16* nocapture %ptr) nounwind readonly {
entry:
; THUMB-LABEL: t5:
  %add.ptr = getelementptr inbounds i16, i16* %ptr, i32 -127
  %0 = load i16, i16* %add.ptr, align 2
; THUMB: ldrh r{{[0-9]}}, [r0, #-254]
  ret i16 %0
}

define zeroext i16 @t6(i16* nocapture %ptr) nounwind readonly {
entry:
; THUMB-LABEL: t6:
  %add.ptr = getelementptr inbounds i16, i16* %ptr, i32 -128
  %0 = load i16, i16* %add.ptr, align 2
; THUMB: ldrh r{{[0-9]}}, [r0]
  ret i16 %0
}

define zeroext i8 @t7(i8* nocapture %ptr) nounwind readonly {
entry:
; THUMB-LABEL: t7:
  %add.ptr = getelementptr inbounds i8, i8* %ptr, i32 -1
  %0 = load i8, i8* %add.ptr, align 1
; THUMB: ldrb r{{[0-9]}}, [r0, #-1]
  ret i8 %0
}

define zeroext i8 @t8(i8* nocapture %ptr) nounwind readonly {
entry:
; THUMB-LABEL: t8:
  %add.ptr = getelementptr inbounds i8, i8* %ptr, i32 -255
  %0 = load i8, i8* %add.ptr, align 1
; THUMB: ldrb r{{[0-9]}}, [r0, #-255]
  ret i8 %0
}

define zeroext i8 @t9(i8* nocapture %ptr) nounwind readonly {
entry:
; THUMB-LABEL: t9:
  %add.ptr = getelementptr inbounds i8, i8* %ptr, i32 -256
  %0 = load i8, i8* %add.ptr, align 1
; THUMB: ldrb r{{[0-9]}}, [r0]
  ret i8 %0
}

define void @t10(i32* nocapture %ptr) nounwind {
entry:
; THUMB-LABEL: t10:
  %add.ptr = getelementptr inbounds i32, i32* %ptr, i32 -1
  store i32 0, i32* %add.ptr, align 4
; THUMB: mov [[REG:r[0-9]+]], r0
; THUMB: str r{{[0-9]}}, [[[REG]], #-4]
  ret void
}

define void @t11(i32* nocapture %ptr) nounwind {
entry:
; THUMB-LABEL: t11:
  %add.ptr = getelementptr inbounds i32, i32* %ptr, i32 -63
  store i32 0, i32* %add.ptr, align 4
; THUMB: mov [[REG:r[0-9]+]], r0
; THUMB: str r{{[0-9]}}, [[[REG]], #-252]
  ret void
}

define void @t12(i32* nocapture %ptr) nounwind {
entry:
; THUMB-LABEL: t12:
  %add.ptr = getelementptr inbounds i32, i32* %ptr, i32 -64
  store i32 0, i32* %add.ptr, align 4
; THUMB: mov [[PTR:r[0-9]+]], r0
; THUMB: movs [[VAL:r[0-9]+]], #0
; THUMB: movw [[REG:r[0-9]+]], #65280
; THUMB: movt [[REG]], #65535
; THUMB: add [[PTR]], [[REG]]
; THUMB: str [[VAL]], [[[PTR]]]
  ret void
}

define void @t13(i16* nocapture %ptr) nounwind {
entry:
; THUMB-LABEL: t13:
  %add.ptr = getelementptr inbounds i16, i16* %ptr, i32 -1
  store i16 0, i16* %add.ptr, align 2
; THUMB: mov [[REG:r[0-9]+]], r0
; THUMB: strh r{{[0-9]}}, [[[REG]], #-2]
  ret void
}

define void @t14(i16* nocapture %ptr) nounwind {
entry:
; THUMB-LABEL: t14:
  %add.ptr = getelementptr inbounds i16, i16* %ptr, i32 -127
  store i16 0, i16* %add.ptr, align 2
; THUMB: mov [[REG:r[0-9]+]], r0
; THUMB: strh r{{[0-9]}}, [[[REG]], #-254]
  ret void
}

define void @t15(i16* nocapture %ptr) nounwind {
entry:
; THUMB-LABEL: t15:
  %add.ptr = getelementptr inbounds i16, i16* %ptr, i32 -128
  store i16 0, i16* %add.ptr, align 2
; THUMB: mov [[PTR:r[0-9]+]], r0
; THUMB: movs [[VAL:r[0-9]+]], #0
; THUMB: movw [[REG:r[0-9]+]], #65280
; THUMB: movt [[REG]], #65535
; THUMB: add [[PTR]], [[REG]]
; THUMB: strh [[VAL]], [[[PTR]]]
  ret void
}

define void @t16(i8* nocapture %ptr) nounwind {
entry:
; THUMB-LABEL: t16:
  %add.ptr = getelementptr inbounds i8, i8* %ptr, i32 -1
  store i8 0, i8* %add.ptr, align 1
; THUMB: mov [[REG:r[0-9]+]], r0
; THUMB: strb r{{[0-9]}}, [[[REG]], #-1]
  ret void
}

define void @t17(i8* nocapture %ptr) nounwind {
entry:
; THUMB-LABEL: t17:
  %add.ptr = getelementptr inbounds i8, i8* %ptr, i32 -255
  store i8 0, i8* %add.ptr, align 1
; THUMB: mov [[REG:r[0-9]+]], r0
; THUMB: strb r{{[0-9]}}, [[[REG]], #-255]
  ret void
}

define void @t18(i8* nocapture %ptr) nounwind {
entry:
; THUMB-LABEL: t18:
  %add.ptr = getelementptr inbounds i8, i8* %ptr, i32 -256
  store i8 0, i8* %add.ptr, align 1
; THUMB: mov [[PTR:r[0-9]+]], r0
; THUMB: movs [[VAL]], #0
; THUMB: movw [[REG:r[0-9]+]], #65280
; THUMB: movt [[REG]], #65535
; THUMB: add [[PTR]], [[REG]]
; THUMB: strb [[VAL]], [[[PTR]]]
  ret void
}
