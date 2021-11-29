; RUN: opt < %s -passes=globaldce

@A = internal alias void (), void ()* @F
define internal void @F() { ret void }
