; RUN: not llc -mtriple=i386-unknown-linux-gnu -o - %s 2>&1 | FileCheck %s

; CHECK: error: couldn't allocate input reg for constraint '{esp}'
define dso_local i64 @test_esp(i64 %in) local_unnamed_addr nounwind {
entry:
  %0 = tail call i64 asm sideeffect "mov $1, $0", "=r,{esp},~{dirflag},~{fpsr},~{flags}"(i64 81985529216486895)
  %conv = trunc i64 %0 to i32
  %add = add nsw i32 %conv, 3
  %conv1 = sext i32 %add to i64
  ret i64 %conv1
}

