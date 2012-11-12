; RUN: llc < %s -march=arm

define void @"java.lang.String::getChars"([84 x i8]* %method, i32 %base_pc, [788 x i8]* %thread) {
  %1 = sub i32 undef, 48                          ; <i32> [#uses=1]
  br i1 undef, label %stack_overflow, label %no_overflow

stack_overflow:                                   ; preds = %0
  unreachable

no_overflow:                                      ; preds = %0
  %frame = inttoptr i32 %1 to [17 x i32]*         ; <[17 x i32]*> [#uses=4]
  %2 = load i32* null                             ; <i32> [#uses=2]
  %3 = getelementptr inbounds [17 x i32]* %frame, i32 0, i32 14 ; <i32*> [#uses=1]
  %4 = load i32* %3                               ; <i32> [#uses=2]
  %5 = load [8 x i8]** undef                      ; <[8 x i8]*> [#uses=2]
  br i1 undef, label %bci_13, label %bci_4

bci_13:                                           ; preds = %no_overflow
  br i1 undef, label %bci_30, label %bci_21

bci_30:                                           ; preds = %bci_13
  %6 = icmp sle i32 %2, %4                        ; <i1> [#uses=1]
  br i1 %6, label %bci_46, label %bci_35

bci_46:                                           ; preds = %bci_30
  store [84 x i8]* %method, [84 x i8]** undef
  br i1 false, label %no_exception, label %exception

exception:                                        ; preds = %bci_46
  ret void

no_exception:                                     ; preds = %bci_46
  ret void

bci_35:                                           ; preds = %bci_30
  %7 = getelementptr inbounds [17 x i32]* %frame, i32 0, i32 15 ; <i32*> [#uses=1]
  store i32 %2, i32* %7
  %8 = getelementptr inbounds [17 x i32]* %frame, i32 0, i32 14 ; <i32*> [#uses=1]
  store i32 %4, i32* %8
  %9 = getelementptr inbounds [17 x i32]* %frame, i32 0, i32 13 ; <i32*> [#uses=1]
  %10 = bitcast i32* %9 to [8 x i8]**             ; <[8 x i8]**> [#uses=1]
  store [8 x i8]* %5, [8 x i8]** %10
  call void inttoptr (i32 13839116 to void ([788 x i8]*, i32)*)([788 x i8]* %thread, i32 7)
  ret void

bci_21:                                           ; preds = %bci_13
  ret void

bci_4:                                            ; preds = %no_overflow
  store [8 x i8]* %5, [8 x i8]** undef
  store i32 undef, i32* undef
  call void inttoptr (i32 13839116 to void ([788 x i8]*, i32)*)([788 x i8]* %thread, i32 7)
  ret void
}
