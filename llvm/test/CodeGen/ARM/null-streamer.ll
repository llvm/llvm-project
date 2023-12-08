; Test the null streamer with a terget streamer.
; RUN: llc -O0 -filetype=null -mtriple=arm-linux < %s

define i32 @main()  {
entry:
  ret i32 0
}

module asm ".fnstart"
