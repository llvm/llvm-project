; Check the inliner doesn't inline a function with different sign return address schemes.
; RUN: opt < %s -passes=inline -S | FileCheck %s

define internal void @foo_all() #0 {
  ret void
}

define internal void @foo_nonleaf() #1 {
  ret void
}

define internal void @foo_none() {
  ret void
}

define internal void @foo_lr() #3 {
  ret void
}

define internal void @foo_bkey() #4 {
  ret void
}

define dso_local void @bar_all() #0 {
; CHECK-LABEL: bar_all
; CHECK-NOT:     call void @foo_all()
; CHECK-NEXT:    call void @foo_nonleaf()
; CHECK-NEXT:    call void @foo_none()
; CHECK-NEXT:    call void @foo_lr()
; CHECK-NEXT:    call void @foo_bkey()
  call void @foo_all()
  call void @foo_nonleaf()
  call void @foo_none()
  call void @foo_lr()
  call void @foo_bkey()
  ret void
}

define dso_local void @bar_nonleaf() #1 {
; CHECK-LABEL: bar_nonleaf
; CHECK-NEXT:    call void @foo_all()
; CHECK-NOT:     call void @foo_nonleaf()
; CHECK-NEXT:    call void @foo_none()
; CHECK-NEXT:    call void @foo_lr()
; CHECK-NEXT:    call void @foo_bkey()
  call void @foo_all()
  call void @foo_nonleaf()
  call void @foo_none()
  call void @foo_lr()
  call void @foo_bkey()
  ret void
}

define dso_local void @bar_none()  {
; CHECK-LABEL: bar_none
; CHECK-NEXT:    call void @foo_all()
; CHECK-NEXT:    call void @foo_nonleaf()
; CHECK-NOT:     call void @foo_none()
; CHECK-NEXT:    call void @foo_lr()
; CHECK-NEXT:    call void @foo_bkey()
  call void @foo_all()
  call void @foo_nonleaf()
  call void @foo_none()
  call void @foo_lr()
  call void @foo_bkey()
  ret void
}

define dso_local void @bar_lr() #3 {
; CHECK-LABEL: bar_lr
; CHECK-NEXT:    call void @foo_all()
; CHECK-NEXT:    call void @foo_nonleaf()
; CHECK-NEXT:    call void @foo_none()
; CHECK-NOT:     call void @foo_lr()
; CHECK-NEXT:    call void @foo_bkey()
  call void @foo_all()
  call void @foo_nonleaf()
  call void @foo_none()
  call void @foo_lr()
  call void @foo_bkey()
  ret void
}

define dso_local void @bar_bkey() #4 {
; CHECK-LABEL: bar_bkey
; CHECK-NEXT:    call void @foo_all()
; CHECK-NEXT:    call void @foo_nonleaf()
; CHECK-NEXT:    call void @foo_none()
; CHECK-NEXT:    call void @foo_lr()
; CHECK-NOT:     call void @foo_bkey()
  call void @foo_all()
  call void @foo_nonleaf()
  call void @foo_none()
  call void @foo_lr()
  call void @foo_bkey()
  ret void
}


attributes #0 = { "sign-return-address"="all" }
attributes #1 = { "sign-return-address"="non-leaf" }
attributes #3 = { "branch-protection-pauth-lr" "sign-return-address"="non-leaf" }
attributes #4 = { "branch-protection-pauth-lr" "sign-return-address"="non-leaf" "sign-return-address-key"="b_key" }
