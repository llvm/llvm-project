; RUN: opt < %s -passes=globaldce

@A = internal alias void (), ptr @F
define internal void @F() { ret void }
