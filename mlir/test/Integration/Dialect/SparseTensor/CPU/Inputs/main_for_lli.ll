; Dummy wrapper required by lli, which does not support void functions (i.e.
; it fails if non-zero code is returned)
define i32 @entry_lli() {
  call void @entry()
  ret i32 0
}

declare void @entry()
