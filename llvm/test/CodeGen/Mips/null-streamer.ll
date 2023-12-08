; Test the null streamer with a terget streamer.
; RUN: llc -O0 -filetype=null -mtriple=mips-linux < %s

define i32 @main()  {
entry:
  ret i32 0
}
