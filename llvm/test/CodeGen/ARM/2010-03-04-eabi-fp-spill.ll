; RUN: llc < %s -mtriple=arm-unknown-linux-gnueabi

define void @"java.lang.String::getChars"([84 x i8]* %method, i32 %base_pc, [788 x i8]* %thread) {
  %1 = load i32* undef                            ; <i32> [#uses=1]
  %2 = sub i32 %1, 48                             ; <i32> [#uses=1]
  br i1 undef, label %stack_overflow, label %no_overflow

stack_overflow:                                   ; preds = %0
  unreachable

no_overflow:                                      ; preds = %0
  %frame = inttoptr i32 %2 to [17 x i32]*         ; <[17 x i32]*> [#uses=4]
  %3 = load i32* undef                            ; <i32> [#uses=1]
  %4 = load i32* null                             ; <i32> [#uses=1]
  %5 = getelementptr inbounds [17 x i32]* %frame, i32 0, i32 13 ; <i32*> [#uses=1]
  %6 = bitcast i32* %5 to [8 x i8]**              ; <[8 x i8]**> [#uses=1]
  %7 = load [8 x i8]** %6                         ; <[8 x i8]*> [#uses=1]
  %8 = getelementptr inbounds [17 x i32]* %frame, i32 0, i32 12 ; <i32*> [#uses=1]
  %9 = load i32* %8                               ; <i32> [#uses=1]
  br i1 undef, label %bci_13, label %bci_4

bci_13:                                           ; preds = %no_overflow
  br i1 undef, label %bci_30, label %bci_21

bci_30:                                           ; preds = %bci_13
  br i1 undef, label %bci_46, label %bci_35

bci_46:                                           ; preds = %bci_30
  %10 = sub i32 %4, %3                            ; <i32> [#uses=1]
  %11 = load [8 x i8]** null                      ; <[8 x i8]*> [#uses=1]
  %callee = bitcast [8 x i8]* %11 to [84 x i8]*   ; <[84 x i8]*> [#uses=1]
  %12 = bitcast i8* undef to i32*                 ; <i32*> [#uses=1]
  %base_pc7 = load i32* %12                       ; <i32> [#uses=2]
  %13 = add i32 %base_pc7, 0                      ; <i32> [#uses=1]
  %14 = inttoptr i32 %13 to void ([84 x i8]*, i32, [788 x i8]*)** ; <void ([84 x i8]*, i32, [788 x i8]*)**> [#uses=1]
  %entry_point = load void ([84 x i8]*, i32, [788 x i8]*)** %14 ; <void ([84 x i8]*, i32, [788 x i8]*)*> [#uses=1]
  %15 = getelementptr inbounds [17 x i32]* %frame, i32 0, i32 1 ; <i32*> [#uses=1]
  %16 = ptrtoint i32* %15 to i32                  ; <i32> [#uses=1]
  %stack_pointer_addr9 = bitcast i8* undef to i32* ; <i32*> [#uses=1]
  store i32 %16, i32* %stack_pointer_addr9
  %17 = getelementptr inbounds [17 x i32]* %frame, i32 0, i32 2 ; <i32*> [#uses=1]
  store i32 %9, i32* %17
  store i32 %10, i32* undef
  store [84 x i8]* %method, [84 x i8]** undef
  %18 = add i32 %base_pc, 20                      ; <i32> [#uses=1]
  store i32 %18, i32* undef
  store [8 x i8]* %7, [8 x i8]** undef
  call void %entry_point([84 x i8]* %callee, i32 %base_pc7, [788 x i8]* %thread)
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
