; Check that verification doesn't fail when reducing a function with
; unreachable blocks.
;
; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=unreachable-basic-blocks --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck -check-prefix=UNREACHABLE %s < %t

; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=basic-blocks --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck -check-prefix=REACHABLE %s < %t

; CHECK-INTERESTINGNESS: test0
; CHECK-INTERESTINGNESS: test1

; UNREACHABLE: define void @test0() {
; UNREACHABLE-NEXT: entry:
; UNREACHABLE-NEXT:   br label %exit

; UNREACHABLE-NOT: unreachable
; UNREACHABLE: exit:
; UNREACHABLE-NEXT: ret void


; basic-blocks cannot deal with unreachable blocks, leave it behind
; REACHABLE: define void @test0() {
; REACHABLE: entry:
; REACHABLE: unreachable:
; REACHABLE: exit:

define void @test0() {
entry:
  br label %exit

unreachable:                                        ; No predecessors!
  br label %exit

exit:
  ret void
}

; UNREACHABLE: define void @test1() {
; UNREACHABLE-NEXT: entry:
; UNREACHABLE-NEXT:   br label %exit

; REACHABLE: define void @test1() {
; REACHABLE: entry:
; REACHABLE: unreachable:
; REACHABLE: exit:
define void @test1() {
entry:
  br label %exit

unreachable:
  br label %unreachable

exit:
  ret void
}
