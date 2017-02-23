; Make sure we don't crash when referencing an unnamed global.
; RUN: opt %s -module-summary-analysis -S

@0 = external global [1 x { i64 }]

define internal void @tinkywinky() {
  call void @patatino(i64 ptrtoint ([1 x { i64 }]* @0 to i64), i64 4)
  ret void
}
declare void @patatino(i64, i64)
