; Tests that check our handling of volatile instructions encountered
; when scanning for dependencies
; RUN: opt -basicaa -gvn -S < %s | FileCheck %s

; Check that we can bypass a volatile load when searching
; for dependencies of a non-volatile load
define i32 @test1(i32* nocapture %p, i32* nocapture %q) {
; CHECK-LABEL: test1
; CHECK:      %0 = load volatile i32, i32* %q
; CHECK-NEXT: ret i32 0
entry:
  %x = load i32, i32* %p
  load volatile i32, i32* %q
  %y = load i32, i32* %p
  %add = sub i32 %y, %x
  ret i32 %add
}

; We can not value forward if the query instruction is 
; volatile, this would be (in effect) removing the volatile load
define i32 @test2(i32* nocapture %p, i32* nocapture %q) {
; CHECK-LABEL: test2
; CHECK:      %x = load i32, i32* %p
; CHECK-NEXT: %y = load volatile i32, i32* %p
; CHECK-NEXT: %add = sub i32 %y, %x
entry:
  %x = load i32, i32* %p
  %y = load volatile i32, i32* %p
  %add = sub i32 %y, %x
  ret i32 %add
}

; If the query instruction is itself volatile, we *cannot*
; reorder it even if p and q are noalias
define i32 @test3(i32* noalias nocapture %p, i32* noalias nocapture %q) {
; CHECK-LABEL: test3
; CHECK:      %x = load i32, i32* %p
; CHECK-NEXT: %0 = load volatile i32, i32* %q
; CHECK-NEXT: %y = load volatile i32, i32* %p
entry:
  %x = load i32, i32* %p
  load volatile i32, i32* %q
  %y = load volatile i32, i32* %p
  %add = sub i32 %y, %x
  ret i32 %add
}

; If an encountered instruction is both volatile and ordered, 
; we need to use the strictest ordering of either.  In this 
; case, the ordering prevents forwarding.
define i32 @test4(i32* noalias nocapture %p, i32* noalias nocapture %q) {
; CHECK-LABEL: test4
; CHECK:      %x = load i32, i32* %p
; CHECK-NEXT: %0 = load atomic volatile i32, i32* %q seq_cst 
; CHECK-NEXT: %y = load atomic i32, i32* %p seq_cst
entry:
  %x = load i32, i32* %p
  load atomic volatile i32, i32* %q seq_cst, align 4
  %y = load atomic i32, i32* %p seq_cst, align 4
  %add = sub i32 %y, %x
  ret i32 %add
}

; Value forwarding from a volatile load is perfectly legal
define i32 @test5(i32* nocapture %p, i32* nocapture %q) {
; CHECK-LABEL: test5
; CHECK:      %x = load volatile i32, i32* %p
; CHECK-NEXT: ret i32 0
entry:
  %x = load volatile i32, i32* %p
  %y = load i32, i32* %p
  %add = sub i32 %y, %x
  ret i32 %add
}

; Does cross block redundancy elimination work with volatiles?
define i32 @test6(i32* noalias nocapture %p, i32* noalias nocapture %q) {
; CHECK-LABEL: test6
; CHECK:      %y1 = load i32, i32* %p
; CHECK-LABEL: header
; CHECK:      %x = load volatile i32, i32* %q
; CHECK-NEXT: %add = sub i32 %y1, %x
entry:
  %y1 = load i32, i32* %p
  call void @use(i32 %y1)
  br label %header
header:
  %x = load volatile i32, i32* %q
  %y = load i32, i32* %p
  %add = sub i32 %y, %x
  %cnd = icmp eq i32 %add, 0
  br i1 %cnd, label %exit, label %header
exit:
  ret i32 %add
}

; Does cross block PRE work with volatiles?
define i32 @test7(i1 %c, i32* noalias nocapture %p, i32* noalias nocapture %q) {
; CHECK-LABEL: test7
; CHECK-LABEL: entry.header_crit_edge:
; CHECK:       %y.pre = load i32, i32* %p
; CHECK-LABEL: skip:
; CHECK:       %y1 = load i32, i32* %p
; CHECK-LABEL: header:
; CHECK:      %y = phi i32
; CHECK-NEXT: %x = load volatile i32, i32* %q
; CHECK-NEXT: %add = sub i32 %y, %x
entry:
  br i1 %c, label %header, label %skip
skip:
  %y1 = load i32, i32* %p
  call void @use(i32 %y1)
  br label %header
header:
  %x = load volatile i32, i32* %q
  %y = load i32, i32* %p
  %add = sub i32 %y, %x
  %cnd = icmp eq i32 %add, 0
  br i1 %cnd, label %exit, label %header
exit:
  ret i32 %add
}

; Another volatile PRE case - two paths through a loop
; load in preheader, one path read only, one not
define i32 @test8(i1 %b, i1 %c, i32* noalias %p, i32* noalias %q) {
; CHECK-LABEL: test8
; CHECK-LABEL: entry
; CHECK:       %y1 = load i32, i32* %p
; CHECK-LABEL: header:
; CHECK:      %y = phi i32
; CHECK-NEXT: %x = load volatile i32, i32* %q
; CHECK-NOT:  load
; CHECK-LABEL: skip.header_crit_edge:
; CHECK:       %y.pre = load i32, i32* %p
entry:
  %y1 = load i32, i32* %p
  call void @use(i32 %y1)
  br label %header
header:
  %x = load volatile i32, i32* %q
  %y = load i32, i32* %p
  call void @use(i32 %y)
  br i1 %b, label %skip, label %header
skip:
  ; escaping the arguments is explicitly required since we marked 
  ; them noalias
  call void @clobber(i32* %p, i32* %q)
  br i1 %c, label %header, label %exit
exit:
  %add = sub i32 %y, %x
  ret i32 %add
}

define i32 @test9(i32* %V) {
entry:
  %load = load volatile i32, i32* %V, !range !0
  ret i32 %load
}
; CHECK-LABEL: test9
; CHECK: load volatile
; CHECK: ret i32 0

declare void @use(i32) readonly
declare void @clobber(i32* %p, i32* %q)

!0 = !{ i32 0, i32 1 }
