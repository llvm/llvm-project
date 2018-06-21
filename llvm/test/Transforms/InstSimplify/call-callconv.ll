; RUN: opt < %s -instcombine -S | FileCheck %s
; Verify that the non-default calling conv doesn't prevent the libcall simplification

@.str = private unnamed_addr constant [4 x i8] c"abc\00", align 1

define arm_aapcscc i32 @_abs(i32 %i) nounwind readnone {
; CHECK: _abs
  %call = tail call arm_aapcscc i32 @abs(i32 %i) nounwind readnone
  ret i32 %call
; CHECK: %[[ISPOS:.*]] = icmp sgt i32 %i, -1
; CHECK: %[[NEG:.*]] = sub i32 0, %i
; CHECK: %[[RET:.*]] = select i1 %[[ISPOS]], i32 %i, i32 %[[NEG]]
; CHECK: ret i32 %[[RET]]
}

declare arm_aapcscc i32 @abs(i32) nounwind readnone

define arm_aapcscc i32 @_labs(i32 %i) nounwind readnone {
; CHECK: _labs
  %call = tail call arm_aapcscc i32 @labs(i32 %i) nounwind readnone
  ret i32 %call
; CHECK: %[[ISPOS:.*]] = icmp sgt i32 %i, -1
; CHECK: %[[NEG:.*]] = sub i32 0, %i
; CHECK: %[[RET:.*]] = select i1 %[[ISPOS]], i32 %i, i32 %[[NEG]]
; CHECK: ret i32 %[[RET]]
}

declare arm_aapcscc i32 @labs(i32) nounwind readnone

define arm_aapcscc i32 @_strlen1() {
; CHECK: _strlen1
  %call = tail call arm_aapcscc i32 @strlen(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i32 0, i32 0))
  ret i32 %call
; CHECK: ret i32 3
}

declare arm_aapcscc i32 @strlen(i8*)

define arm_aapcscc zeroext i1 @_strlen2(i8* %str) {
; CHECK: _strlen2
  %call = tail call arm_aapcscc i32 @strlen(i8* %str)
  %cmp = icmp ne i32 %call, 0
  ret i1 %cmp

; CHECK: %[[STRLENFIRST:.*]] = load i8, i8* %str
; CHECK: %[[CMP:.*]] = icmp ne i8 %[[STRLENFIRST]], 0
; CHECK: ret i1 %[[CMP]]
}
