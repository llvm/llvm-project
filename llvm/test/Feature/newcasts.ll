; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

define void @"NewCasts" (i16 %x) {
  %a = zext i16 %x to i32
  %b = sext i16 %x to i32
  %c = trunc i16 %x to i8
  %d = uitofp i16 %x to float
  %e = sitofp i16 %x to double
  %f = fptoui float %d to i16
  %g = fptosi double %e to i16
  %i = fpext float %d to double
  %j = fptrunc double %i to float
  %k = bitcast i32 %a to float
  %l = inttoptr i16 %x to i32*
  %m = ptrtoint i32* %l to i64
  %n = insertelement <4 x i32> undef, i32 %a, i32 0
  %o = sitofp <4 x i32> %n to <4 x float>
  %p = uitofp <4 x i32> %n to <4 x float>
  %q = fptosi <4 x float> %p to <4 x i32>
  %r = fptoui <4 x float> %p to <4 x i32>
  ret void
}


define i16 @"ZExtConst" () {
  ret i16 trunc ( i32 zext ( i16 42 to i32) to i16 )
}

define i16 @"SExtConst" () {
  ret i16 trunc (i32 sext (i16 42 to i32) to i16 )
}
