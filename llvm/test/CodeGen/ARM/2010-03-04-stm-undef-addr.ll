; RUN: llc -mtriple=arm-eabi %s -o /dev/null

define void @"java.lang.String::getChars"(ptr %method, i32 %base_pc, ptr %thread) {
  %1 = sub i32 undef, 48                          ; <i32> [#uses=1]
  br i1 undef, label %stack_overflow, label %no_overflow

stack_overflow:                                   ; preds = %0
  unreachable

no_overflow:                                      ; preds = %0
  %frame = inttoptr i32 %1 to ptr         ; <ptr> [#uses=4]
  %2 = load i32, ptr null                             ; <i32> [#uses=2]
  %3 = getelementptr inbounds [17 x i32], ptr %frame, i32 0, i32 14 ; <ptr> [#uses=1]
  %4 = load i32, ptr %3                               ; <i32> [#uses=2]
  %5 = load ptr, ptr undef                      ; <ptr> [#uses=2]
  br i1 undef, label %bci_13, label %bci_4

bci_13:                                           ; preds = %no_overflow
  br i1 undef, label %bci_30, label %bci_21

bci_30:                                           ; preds = %bci_13
  %6 = icmp sle i32 %2, %4                        ; <i1> [#uses=1]
  br i1 %6, label %bci_46, label %bci_35

bci_46:                                           ; preds = %bci_30
  store ptr %method, ptr undef
  br i1 false, label %no_exception, label %exception

exception:                                        ; preds = %bci_46
  ret void

no_exception:                                     ; preds = %bci_46
  ret void

bci_35:                                           ; preds = %bci_30
  %7 = getelementptr inbounds [17 x i32], ptr %frame, i32 0, i32 15 ; <ptr> [#uses=1]
  store i32 %2, ptr %7
  %8 = getelementptr inbounds [17 x i32], ptr %frame, i32 0, i32 14 ; <ptr> [#uses=1]
  store i32 %4, ptr %8
  %9 = getelementptr inbounds [17 x i32], ptr %frame, i32 0, i32 13 ; <ptr> [#uses=1]
  store ptr %5, ptr %9
  call void inttoptr (i32 13839116 to ptr)(ptr %thread, i32 7)
  ret void

bci_21:                                           ; preds = %bci_13
  ret void

bci_4:                                            ; preds = %no_overflow
  store ptr %5, ptr undef
  store i32 undef, ptr undef
  call void inttoptr (i32 13839116 to ptr)(ptr %thread, i32 7)
  ret void
}
