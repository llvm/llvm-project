; RUN: opt < %s -constprop -S -o - | FileCheck %s

; Function Attrs: minsize norecurse nounwind optsize readnone
define void @foo1() #0 {
entry:
  ret void
}

; Function Attrs: minsize norecurse nounwind optsize readnone
define void @foo2() align 4 {
entry:
  ret void
}

; Function Attrs: minsize nounwind optsize
define i32 @main() local_unnamed_addr #1 {
entry:
; CHECK: ptrtoint
  %call = tail call i32 bitcast (i32 (...)* @process to i32 (i32)*)(i32 and (i32 ptrtoint (void ()* @foo1 to i32), i32 2)) #3
; CHECK-NEXT: ptrtoint
  %call2 = tail call i32 bitcast (i32 (...)* @process to i32 (i32)*)(i32 and (i32 ptrtoint (void ()* @foo2 to i32), i32 2)) #3
  ret i32 0
}

; Function Attrs: minsize optsize
declare i32 @process(...) local_unnamed_addr #2

