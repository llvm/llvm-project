; RUN: opt < %s -passes=instcombine -disable-output
; PR1304

define i64 @bork(<1 x i64> %vec) {
  %tmp = extractelement <1 x i64> %vec, i32 0
  ret i64 %tmp
}
