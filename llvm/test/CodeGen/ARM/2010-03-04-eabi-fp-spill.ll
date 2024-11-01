; RUN: llc < %s -mtriple=arm-unknown-linux-gnueabi

define void @"java.lang.String::getChars"(ptr %method, i32 %base_pc, ptr %thread) {
  %1 = load i32, ptr undef                            ; <i32> [#uses=1]
  %2 = sub i32 %1, 48                             ; <i32> [#uses=1]
  br i1 undef, label %stack_overflow, label %no_overflow

stack_overflow:                                   ; preds = %0
  unreachable

no_overflow:                                      ; preds = %0
  %frame = inttoptr i32 %2 to ptr         ; <ptr> [#uses=4]
  %3 = load i32, ptr undef                            ; <i32> [#uses=1]
  %4 = load i32, ptr null                             ; <i32> [#uses=1]
  %5 = getelementptr inbounds [17 x i32], ptr %frame, i32 0, i32 13 ; <ptr> [#uses=1]
  %6 = load ptr, ptr %5                         ; <ptr> [#uses=1]
  %7 = getelementptr inbounds [17 x i32], ptr %frame, i32 0, i32 12 ; <ptr> [#uses=1]
  %8 = load i32, ptr %7                               ; <i32> [#uses=1]
  br i1 undef, label %bci_13, label %bci_4

bci_13:                                           ; preds = %no_overflow
  br i1 undef, label %bci_30, label %bci_21

bci_30:                                           ; preds = %bci_13
  br i1 undef, label %bci_46, label %bci_35

bci_46:                                           ; preds = %bci_30
  %9 = sub i32 %4, %3                            ; <i32> [#uses=1]
  %10 = load ptr, ptr null                      ; <ptr> [#uses=1]
  %base_pc7 = load i32, ptr undef                       ; <i32> [#uses=2]
  %11 = add i32 %base_pc7, 0                      ; <i32> [#uses=1]
  %12 = inttoptr i32 %11 to ptr ; <ptr> [#uses=1]
  %entry_point = load ptr, ptr %12 ; <ptr> [#uses=1]
  %13 = getelementptr inbounds [17 x i32], ptr %frame, i32 0, i32 1 ; <ptr> [#uses=1]
  %14 = ptrtoint ptr %13 to i32                  ; <i32> [#uses=1]
  store i32 %14, ptr undef
  %15 = getelementptr inbounds [17 x i32], ptr %frame, i32 0, i32 2 ; <ptr> [#uses=1]
  store i32 %8, ptr %15
  store i32 %9, ptr undef
  store ptr %method, ptr undef
  %16 = add i32 %base_pc, 20                      ; <i32> [#uses=1]
  store i32 %16, ptr undef
  store ptr %6, ptr undef
  call void %entry_point(ptr %10, i32 %base_pc7, ptr %thread)
  br i1 undef, label %no_exception, label %exception

exception:                                        ; preds = %bci_46
  ret void

no_exception:                                     ; preds = %bci_46
  ret void

bci_35:                                           ; preds = %bci_30
  ret void

bci_21:                                           ; preds = %bci_13
  ret void

bci_4:                                            ; preds = %no_overflow
  ret void
}
