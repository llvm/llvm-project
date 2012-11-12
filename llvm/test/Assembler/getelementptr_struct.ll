; RUN: not llvm-as < %s >/dev/null 2> %t
; RUN: FileCheck %s < %t
; Test the case of a incorrect indices type into struct

; CHECK: invalid getelementptr indices

%RT = type { i8 , [10 x [20 x i32]], i8  }
%ST = type { i32, double, %RT }

define i32* @foo(%ST* %s) {
entry:
  %reg = getelementptr %ST* %s, i32 1, i64 2, i32 1, i32 5, i32 13
  ret i32* %reg
}

