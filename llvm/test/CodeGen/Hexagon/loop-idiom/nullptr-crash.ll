; RUN: opt -hexagon-loop-idiom -mtriple hexagon-unknown-elf < %s
; REQUIRES: asserts

target triple = "hexagon"

; Function Attrs: nounwind
define void @fred(i8 zeroext %L) #0 {
entry:
  br i1 undef, label %if.end53, label %while.body37

while.body37:                                     ; preds = %while.body37, %entry
  %i.121 = phi i32 [ %inc46, %while.body37 ], [ 0, %entry ]
  %shl = shl i32 1, %i.121
  %and39 = and i32 %shl, undef
  %tobool40 = icmp eq i32 %and39, 0
  %inc46 = add nuw nsw i32 %i.121, 1
  %storemerge = select i1 %tobool40, i8 %L, i8 0
  br i1 undef, label %while.body37, label %if.end53

if.end53:                                         ; preds = %while.body37, %entry
  ret void
}

attributes #0 = { nounwind }
