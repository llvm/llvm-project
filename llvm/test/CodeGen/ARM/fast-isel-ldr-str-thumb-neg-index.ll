; RUN: llc < %s -O0 -fast-isel-abort -relocation-model=dynamic-no-pic -mtriple=thumbv7-apple-ios | FileCheck %s --check-prefix=THUMB

define i32 @t1(i32* nocapture %ptr) nounwind readonly {
entry:
; THUMB: t1
  %add.ptr = getelementptr inbounds i32* %ptr, i32 -1
  %0 = load i32* %add.ptr, align 4, !tbaa !0
; THUMB: ldr r{{[0-9]}}, [r0, #-4]
  ret i32 %0
}

define i32 @t2(i32* nocapture %ptr) nounwind readonly {
entry:
; THUMB: t2
  %add.ptr = getelementptr inbounds i32* %ptr, i32 -63
  %0 = load i32* %add.ptr, align 4, !tbaa !0
; THUMB: ldr r{{[0-9]}}, [r0, #-252]
  ret i32 %0
}

define i32 @t3(i32* nocapture %ptr) nounwind readonly {
entry:
; THUMB: t3
  %add.ptr = getelementptr inbounds i32* %ptr, i32 -64
  %0 = load i32* %add.ptr, align 4, !tbaa !0
; THUMB: ldr r{{[0-9]}}, [r0]
  ret i32 %0
}

define zeroext i16 @t4(i16* nocapture %ptr) nounwind readonly {
entry:
; THUMB: t4
  %add.ptr = getelementptr inbounds i16* %ptr, i32 -1
  %0 = load i16* %add.ptr, align 2, !tbaa !3
; THUMB: ldrh r{{[0-9]}}, [r0, #-2]
  ret i16 %0
}

define zeroext i16 @t5(i16* nocapture %ptr) nounwind readonly {
entry:
; THUMB: t5
  %add.ptr = getelementptr inbounds i16* %ptr, i32 -127
  %0 = load i16* %add.ptr, align 2, !tbaa !3
; THUMB: ldrh r{{[0-9]}}, [r0, #-254]
  ret i16 %0
}

define zeroext i16 @t6(i16* nocapture %ptr) nounwind readonly {
entry:
; THUMB: t6
  %add.ptr = getelementptr inbounds i16* %ptr, i32 -128
  %0 = load i16* %add.ptr, align 2, !tbaa !3
; THUMB: ldrh r{{[0-9]}}, [r0]
  ret i16 %0
}

define zeroext i8 @t7(i8* nocapture %ptr) nounwind readonly {
entry:
; THUMB: t7
  %add.ptr = getelementptr inbounds i8* %ptr, i32 -1
  %0 = load i8* %add.ptr, align 1, !tbaa !1
; THUMB: ldrb r{{[0-9]}}, [r0, #-1]
  ret i8 %0
}

define zeroext i8 @t8(i8* nocapture %ptr) nounwind readonly {
entry:
; THUMB: t8
  %add.ptr = getelementptr inbounds i8* %ptr, i32 -255
  %0 = load i8* %add.ptr, align 1, !tbaa !1
; THUMB: ldrb r{{[0-9]}}, [r0, #-255]
  ret i8 %0
}

define zeroext i8 @t9(i8* nocapture %ptr) nounwind readonly {
entry:
; THUMB: t9
  %add.ptr = getelementptr inbounds i8* %ptr, i32 -256
  %0 = load i8* %add.ptr, align 1, !tbaa !1
; THUMB: ldrb r{{[0-9]}}, [r0]
  ret i8 %0
}

define void @t10(i32* nocapture %ptr) nounwind {
entry:
; THUMB: t10
  %add.ptr = getelementptr inbounds i32* %ptr, i32 -1
  store i32 0, i32* %add.ptr, align 4, !tbaa !0
; THUMB: str r{{[0-9]}}, [r0, #-4]
  ret void
}

define void @t11(i32* nocapture %ptr) nounwind {
entry:
; THUMB: t11
  %add.ptr = getelementptr inbounds i32* %ptr, i32 -63
  store i32 0, i32* %add.ptr, align 4, !tbaa !0
; THUMB: str r{{[0-9]}}, [r0, #-252]
  ret void
}

define void @t12(i32* nocapture %ptr) nounwind {
entry:
; THUMB: t12
  %add.ptr = getelementptr inbounds i32* %ptr, i32 -64
  store i32 0, i32* %add.ptr, align 4, !tbaa !0
; THUMB: str r{{[0-9]}}, [r0]
  ret void
}

define void @t13(i16* nocapture %ptr) nounwind {
entry:
; THUMB: t13
  %add.ptr = getelementptr inbounds i16* %ptr, i32 -1
  store i16 0, i16* %add.ptr, align 2, !tbaa !3
; THUMB: strh r{{[0-9]}}, [r0, #-2]
  ret void
}

define void @t14(i16* nocapture %ptr) nounwind {
entry:
; THUMB: t14
  %add.ptr = getelementptr inbounds i16* %ptr, i32 -127
  store i16 0, i16* %add.ptr, align 2, !tbaa !3
; THUMB: strh r{{[0-9]}}, [r0, #-254]
  ret void
}

define void @t15(i16* nocapture %ptr) nounwind {
entry:
; THUMB: t15
  %add.ptr = getelementptr inbounds i16* %ptr, i32 -128
  store i16 0, i16* %add.ptr, align 2, !tbaa !3
; THUMB: strh r{{[0-9]}}, [r0]
  ret void
}

define void @t16(i8* nocapture %ptr) nounwind {
entry:
; THUMB: t16
  %add.ptr = getelementptr inbounds i8* %ptr, i32 -1
  store i8 0, i8* %add.ptr, align 1, !tbaa !1
; THUMB: strb r{{[0-9]}}, [r0, #-1]
  ret void
}

define void @t17(i8* nocapture %ptr) nounwind {
entry:
; THUMB: t17
  %add.ptr = getelementptr inbounds i8* %ptr, i32 -255
  store i8 0, i8* %add.ptr, align 1, !tbaa !1
; THUMB: strb r{{[0-9]}}, [r0, #-255]
  ret void
}

define void @t18(i8* nocapture %ptr) nounwind {
entry:
; THUMB: t18
  %add.ptr = getelementptr inbounds i8* %ptr, i32 -256
  store i8 0, i8* %add.ptr, align 1, !tbaa !1
; THUMB: strb r{{[0-9]}}, [r0]
  ret void
}

!0 = metadata !{metadata !"int", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA", null}
!3 = metadata !{metadata !"short", metadata !1}
