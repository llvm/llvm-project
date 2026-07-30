; Ensure that llvm-reduce doesn't try to remove the first BB of a
; function when the second BB has multiple predecessors, since that
; results in invalid IR. This issue was fixed by:
; https://reviews.llvm.org/D131026

; RUN: llvm-reduce --delta-passes=basic-blocks --test %python --test-arg %p/Inputs/remove-bbs.py -abort-on-invalid-reduction %s -o %t

define void @f(ptr %x0) {
uninteresting:
  %x2 = alloca ptr, i32 0, align 8
  %x3 = alloca ptr, i32 0, align 8
  br label %interesting1

; this block has 2 predecessors and can't become the entry block
interesting1:
  %x5 = icmp ne ptr %x0, null
  br i1 %x5, label %interesting2, label %interesting1

interesting2:
  store ptr null, ptr null, align 8
  br label %interesting3

interesting3:
  ret void
}
