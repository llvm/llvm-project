; Test that opaque constants are not creating an infinite DAGCombine loop
; RUN: llc < %s
; XFAIL: r600

@a = common global i32* null, align 8
@c = common global i32 0, align 4
@b = common global i32* null, align 8

; Function Attrs: nounwind ssp uwtable
define void @fn() {
  store i32* inttoptr (i64 68719476735 to i32*), i32** @a, align 8
  %1 = load i32, i32* @c, align 4
  %2 = sext i32 %1 to i64
  %3 = lshr i64 %2, 12
  %4 = and i64 %3, 68719476735
  %5 = getelementptr inbounds i32, i32* null, i64 %4
  store i32* %5, i32** @b, align 8
  ret void
}
