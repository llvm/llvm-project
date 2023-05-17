; RUN: llc < %s -mtriple=arm-linux-gnueabi
; PR4528

; Inline asm is allowed to contain operands "=&r", "0".

%struct.device_dma_parameters = type { i32, i32 }
%struct.iovec = type { ptr, i32 }

define i32 @generic_segment_checks(ptr nocapture %iov, ptr nocapture %nr_segs, ptr nocapture %count, i32 %access_flags) nounwind optsize {
entry:
  br label %bb8

bb:                                               ; preds = %bb8
  br i1 undef, label %bb10, label %bb2

bb2:                                              ; preds = %bb
  %asmtmp = tail call %struct.device_dma_parameters asm "adds $1, $2, $3; sbcccs $1, $1, $0; movcc $0, #0", "=&r,=&r,r,Ir,0,~{cc}"(ptr undef, i32 undef, i32 0) nounwind; <%struct.device_dma_parameters> [#uses=1]
  %asmresult = extractvalue %struct.device_dma_parameters %asmtmp, 0; <i32> [#uses=1]
  %0 = icmp eq i32 %asmresult, 0                  ; <i1> [#uses=1]
  br i1 %0, label %bb7, label %bb4

bb4:                                              ; preds = %bb2
  br i1 undef, label %bb10, label %bb9

bb7:                                              ; preds = %bb2
  %1 = add i32 %2, 1                              ; <i32> [#uses=1]
  br label %bb8

bb8:                                              ; preds = %bb7, %entry
  %2 = phi i32 [ 0, %entry ], [ %1, %bb7 ]        ; <i32> [#uses=3]
  %scevgep22 = getelementptr %struct.iovec, ptr %iov, i32 %2, i32 0; <ptr> [#uses=0]
  %3 = load i32, ptr %nr_segs, align 4                ; <i32> [#uses=1]
  %4 = icmp ult i32 %2, %3                        ; <i1> [#uses=1]
  br i1 %4, label %bb, label %bb9

bb9:                                              ; preds = %bb8, %bb4
  store i32 undef, ptr %count, align 4
  ret i32 0

bb10:                                             ; preds = %bb4, %bb
  ret i32 0
}
