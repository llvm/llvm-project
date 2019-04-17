; RUN: opt < %s -globaldce

@A = internal alias void (), void ()* @F
define internal void @F() { ret void }
