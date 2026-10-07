; RUN: llc -mtriple=x86_64-unknown-linux-gnu -O2 -stop-after=branch-folder -o - %s | FileCheck %s

define void @true_likely(i1 noundef zeroext %0) {

  ;CHECK: name: true_likely
  ;CHECK: successors: %bb.1(0x7fef9fcb)

  br i1 %0, label %2, label %3, !prof !6

2:
  tail call void @func_true()
  br label %4

3:
  tail call void @func_false()
  br label %4

4:
  ret void
}

define void @false_likely(i1 noundef zeroext %0) {

  ;CHECK: name: false_likely
  ;CHECK: successors: %bb.1(0x7fef9fcb)

  br i1 %0, label %2, label %3, !prof !7

2:
  tail call void @func_true()
  br label %4

3:
  tail call void @func_false()
  br label %4

4:
  ret void
}

!6 = !{!"branch_weights", i32 2000, i32 1}
!7 = !{!"branch_weights", i32 1, i32 2000}


declare dso_local void @func_true()
declare dso_local void @func_false()
