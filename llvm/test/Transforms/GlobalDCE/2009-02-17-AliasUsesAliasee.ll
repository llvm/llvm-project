; RUN: opt < %s -globaldce

@A = alias internal void ()* @F
define internal void @F() { ret void }
