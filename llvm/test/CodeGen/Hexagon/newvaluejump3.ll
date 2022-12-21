; RUN: llc -march=hexagon -filetype=obj -o /dev/null < %s
; REQUIRES: asserts

; This crashed in the MC code emitter, because a new-value branch was created
; with IMPLICIT_DEF as the producer.

target triple = "hexagon"

%type.0 = type { %type.1, [64 x i8] }
%type.1 = type { [12 x i8], ptr, double }
%type.2 = type { i16, i16, [1 x %type.3] }
%type.3 = type { i32 }
%type.4 = type { ptr, i32 }

define hidden fastcc ptr @fred(ptr nocapture readonly %a0, ptr readonly %a1) unnamed_addr #2 {
b2:
  %v3 = load i8, ptr %a1, align 1
  br i1 undef, label %b4, label %b24

b4:                                               ; preds = %b2
  switch i8 %v3, label %b13 [
    i8 25, label %b5
    i8 26, label %b6
    i8 28, label %b8
    i8 27, label %b9
    i8 43, label %b11
    i8 110, label %b12
  ]

b5:                                               ; preds = %b4
  unreachable

b6:                                               ; preds = %b4
  %v7 = getelementptr inbounds i8, ptr %a1, i32 2
  br label %b16

b8:                                               ; preds = %b4
  br label %b16

b9:                                               ; preds = %b4
  %v10 = tail call fastcc ptr @fred(ptr undef, ptr undef)
  br label %b24

b11:                                              ; preds = %b4
  unreachable

b12:                                              ; preds = %b4
  unreachable

b13:                                              ; preds = %b4
  br label %b14

b14:                                              ; preds = %b13
  br i1 undef, label %b15, label %b16

b15:                                              ; preds = %b14
  unreachable

b16:                                              ; preds = %b20, %b14, %b8, %b6
  %v17 = phi ptr [ %v21, %b20 ], [ undef, %b14 ], [ undef, %b8 ], [ %v7, %b6 ]
  %v18 = phi i32 [ 0, %b20 ], [ undef, %b14 ], [ 0, %b8 ], [ 8, %b6 ]
  %v19 = icmp sgt i32 %v18, 0
  br i1 %v19, label %b20, label %b24

b20:                                              ; preds = %b16
  %v21 = getelementptr inbounds i8, ptr %v17, i32 1
  %v22 = load i8, ptr %v17, align 1
  %v23 = icmp eq i8 %v22, undef
  br i1 %v23, label %b16, label %b24

b24:                                              ; preds = %b20, %b16, %b9, %b2
  %v25 = phi ptr [ null, %b2 ], [ null, %b9 ], [ %v17, %b16 ], [ null, %b20 ]
  ret ptr %v25
}

attributes #0 = { argmemonly nounwind }
attributes #1 = { nounwind readonly "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b,-long-calls" }
attributes #2 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b,-long-calls" }

