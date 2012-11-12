; RUN: opt < %s -instcombine -S | not grep call.*bswap

define i1 @test1(i16 %tmp2) {
        %tmp10 = call i16 @llvm.bswap.i16( i16 %tmp2 )
        %tmp = icmp eq i16 %tmp10, 1
        ret i1 %tmp
}

define i1 @test2(i32 %tmp) {
        %tmp34 = tail call i32 @llvm.bswap.i32( i32 %tmp )
        %tmp.upgrd.1 = icmp eq i32 %tmp34, 1
        ret i1 %tmp.upgrd.1
}

declare i32 @llvm.bswap.i32(i32)

define i1 @test3(i64 %tmp) {
        %tmp34 = tail call i64 @llvm.bswap.i64( i64 %tmp )
        %tmp.upgrd.2 = icmp eq i64 %tmp34, 1
        ret i1 %tmp.upgrd.2
}

declare i64 @llvm.bswap.i64(i64)

declare i16 @llvm.bswap.i16(i16)

; rdar://5992453
; A & 255
define i32 @test4(i32 %a) nounwind  {
entry:
	%tmp2 = tail call i32 @llvm.bswap.i32( i32 %a )	
	%tmp4 = lshr i32 %tmp2, 24
	ret i32 %tmp4
}

; A
define i32 @test5(i32 %a) nounwind  {
entry:
	%tmp2 = tail call i32 @llvm.bswap.i32( i32 %a )
	%tmp4 = tail call i32 @llvm.bswap.i32( i32 %tmp2 )
	ret i32 %tmp4
}

; a >> 24
define i32 @test6(i32 %a) nounwind  {
entry:
	%tmp2 = tail call i32 @llvm.bswap.i32( i32 %a )	
	%tmp4 = and i32 %tmp2, 255
	ret i32 %tmp4
}

; PR5284
define i16 @test7(i32 %A) {
  %B = tail call i32 @llvm.bswap.i32(i32 %A) nounwind 
  %C = trunc i32 %B to i16
  %D = tail call i16 @llvm.bswap.i16(i16 %C) nounwind
  ret i16 %D
}

define i16 @test8(i64 %A) {
  %B = tail call i64 @llvm.bswap.i64(i64 %A) nounwind 
  %C = trunc i64 %B to i16
  %D = tail call i16 @llvm.bswap.i16(i16 %C) nounwind
  ret i16 %D
}

; Misc: Fold bswap(undef) to undef.
define i64 @foo() {
  %a = call i64 @llvm.bswap.i64(i64 undef)
  ret i64 %a
}
