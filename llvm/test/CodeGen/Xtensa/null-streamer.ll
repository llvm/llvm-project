; Test the null streamer with a target streamer.
; RUN: llc -O0 -filetype=null -mtriple=xtensa < %s

define i32 @main()  {
entry:
  ret i32 0
}
