; RUN: opt < %s -basicaa -functionattrs -S | FileCheck %s

; CHECK: define i32 @leaf() #0
define i32 @leaf() {
  ret i32 1
}

; CHECK: define i32 @self_rec() #1
define i32 @self_rec() {
  %a = call i32 @self_rec()
  ret i32 4
}

; CHECK: define i32 @indirect_rec() #1
define i32 @indirect_rec() {
  %a = call i32 @indirect_rec2()
  ret i32 %a
}
; CHECK: define i32 @indirect_rec2() #1
define i32 @indirect_rec2() {
  %a = call i32 @indirect_rec()
  ret i32 %a
}

; CHECK: define i32 @extern() #1
define i32 @extern() {
  %a = call i32 @k()
  ret i32 %a
}
declare i32 @k() readnone

; CHECK: define internal i32 @called_by_norecurse() #0
define internal i32 @called_by_norecurse() {
  %a = call i32 @k()
  ret i32 %a
}
define void @m() norecurse {
  %a = call i32 @called_by_norecurse()
  ret void
}

; CHECK: define internal i32 @called_by_norecurse_indirectly() #0
define internal i32 @called_by_norecurse_indirectly() {
  %a = call i32 @k()
  ret i32 %a
}
define internal void @o() {
  %a = call i32 @called_by_norecurse_indirectly()
  ret void
}
define void @p() norecurse {
  call void @o()
  ret void
}

; CHECK: attributes #0 = { norecurse readnone }
; CHECK: attributes #1 = { readnone }
