; RUN: llc < %s -mtriple=xcore
; PR3898

define i32 @vector_param(<2 x double> %x) nounwind {
       ret i32 1
}
