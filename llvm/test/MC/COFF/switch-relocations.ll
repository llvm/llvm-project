; The purpose of this test is to see if the COFF object writer can properly
; relax the fixups that are created for jump tables on x86-64. See PR7960.

; This test case was reduced from Lua/lapi.c.

; This test has yet to be converted to assembly becase llvm-mc cannot read
; x86-64 COFF code yet.

; RUN: llc -filetype=obj -mtriple i686-pc-win32 %s -o %t
; RUN: llc -filetype=obj -mtriple x86_64-pc-win32 %s -o %t

define void @lua_gc(i32 %what) nounwind {
entry:
  switch i32 %what, label %sw.epilog [
    i32 0, label %sw.bb
    i32 1, label %sw.bb
    i32 2, label %sw.bb
    i32 3, label %sw.bb14
    i32 4, label %sw.bb18
    i32 6, label %sw.bb57
  ]

sw.bb:                                            ; preds = %entry, %entry, %entry
  ret void

sw.bb14:                                          ; preds = %entry
  ret void

sw.bb18:                                          ; preds = %entry
  ret void

sw.bb57:                                          ; preds = %entry
  ret void

sw.epilog:                                        ; preds = %entry
  ret void
}
