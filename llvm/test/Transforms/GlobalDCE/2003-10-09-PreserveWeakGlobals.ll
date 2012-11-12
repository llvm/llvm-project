; Weak variables should be preserved by global DCE!

; RUN: opt < %s -globaldce -S | grep @A


@A = weak global i32 54
