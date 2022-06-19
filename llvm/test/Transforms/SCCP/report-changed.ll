; RUN: opt -passes='require<no-op-module>,ipsccp' -disable-output < %s 2>&1 -debug-pass-manager | FileCheck %s

; CHECK: Invalidating {{.*}} NoOpModuleAnalysis

define i16 @main() {
entry:
  %call10 = call i16 @test_local_fp3(i16 undef)
  ret i16 0
}

declare i16 (i16, i16)* @add_fp2()

define internal i16 (i16, i16)* ()* @add_fp3() {
entry:
  ret i16 (i16, i16)* ()* @add_fp2
}

define internal i16 @test_local_fp3(i16 %tnr) {
entry:
  %tnr.addr = alloca i16, align 1
  %call10 = call i16 @apply_fp3_local(i16 (i16, i16)* ()* ()* @add_fp3, i16 181, i16 16384)
  %0 = load i16, i16* %tnr.addr, align 1
  ret i16 %0
}

define internal i16 @apply_fp3_local(i16 (i16, i16)* ()* ()* %fp, i16 %p1, i16 %p2) {
entry:
  %fp.addr = alloca i16 (i16, i16)* ()* ()*, align 1
  store i16 (i16, i16)* ()* ()* %fp, i16 (i16, i16)* ()* ()** %fp.addr, align 1
  %0 = load i16 (i16, i16)* ()* ()*, i16 (i16, i16)* ()* ()** %fp.addr, align 1
  %call = call i16 (i16, i16)* ()* %0()
  %call2 = call i16 undef(i16 undef, i16 undef)
  ret i16 %call2
}
