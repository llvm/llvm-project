; RUN: llc -verify-machineinstrs < %s -mtriple=ppc64--

define i16 @test(i8* %d1, i16* %d2) {
 %tmp237 = call i16 asm "lhbrx $0, $2, $1", "=r,r,bO,m"( i8* %d1, i32 0, i16* %d2 )
 ret i16 %tmp237
}
