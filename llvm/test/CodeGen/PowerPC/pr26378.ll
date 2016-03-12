; RUN: llc -compile-twice -filetype obj \
; RUN:   -mtriple=powerpc64le-unknown-unknown -mcpu=pwr8 < %s
@foo = common global i32 0, align 4
define i8* @blah() #0 {
  ret i8* bitcast (i32* @foo to i8*)
}  
