; RUN: opt < %s -instcombine -S | grep "16" | count 1

define i8* @bork(i8** %qux) {
  %tmp275 = load i8*, i8** %qux, align 1
  %tmp275276 = ptrtoint i8* %tmp275 to i32
  %tmp277 = add i32 %tmp275276, 16
  %tmp277278 = inttoptr i32 %tmp277 to i8*
  ret i8* %tmp277278
}
