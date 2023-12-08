; RUN: llc < %s -fast-isel -fast-isel-abort=1 -verify-machineinstrs -mtriple=i686-- -mattr=sse2 -no-integrated-as
; RUN: llc < %s -fast-isel -fast-isel-abort=1 -verify-machineinstrs -mtriple=x86_64-apple-darwin10 -no-integrated-as

; This tests very minimal fast-isel functionality.

define ptr @foo(ptr %p, ptr %q, ptr %z) nounwind {
entry:
  %r = load i32, ptr %p
  %s = load i32, ptr %q
  %y = load ptr, ptr %z
  br label %fast

fast:
  %t0 = add i32 %r, %s
  %t1 = mul i32 %t0, %s
  %t2 = sub i32 %t1, %s
  %t3 = and i32 %t2, %s
  %t4 = xor i32 %t3, 3
  %t5 = xor i32 %t4, %s
  %t6 = add i32 %t5, 2
  %t7 = getelementptr i32, ptr %y, i32 1
  %t8 = getelementptr i32, ptr %t7, i32 %t6
  call void asm sideeffect "hello world", ""()
  br label %exit

exit:
  ret ptr %t8
}

define void @bar(ptr %p, ptr %q) nounwind {
entry:
  %r = load double, ptr %p
  %s = load double, ptr %q
  br label %fast

fast:
  %t0 = fadd double %r, %s
  %t1 = fmul double %t0, %s
  %t2 = fsub double %t1, %s
  %t3 = fadd double %t2, 707.0
  br label %exit

exit:
  store double %t3, ptr %q
  ret void
}

define i32 @cast() nounwind {
entry:
	%tmp2 = bitcast i32 0 to i32
	ret i32 %tmp2
}

define void @ptrtoint_i1(ptr %p, ptr %q) nounwind {
  %t = ptrtoint ptr %p to i1
  store i1 %t, ptr %q
  ret void
}
define ptr @inttoptr_i1(i1 %p) nounwind {
  %t = inttoptr i1 %p to ptr
  ret ptr %t
}
define i32 @ptrtoint_i32(ptr %p) nounwind {
  %t = ptrtoint ptr %p to i32
  ret i32 %t
}
define ptr @inttoptr_i32(i32 %p) nounwind {
  %t = inttoptr i32 %p to ptr
  ret ptr %t
}

define void @trunc_i32_i8(i32 %x, ptr %p) nounwind  {
	%tmp1 = trunc i32 %x to i8
	store i8 %tmp1, ptr %p
	ret void
}

define void @trunc_i16_i8(i16 signext %x, ptr %p) nounwind  {
	%tmp1 = trunc i16 %x to i8
	store i8 %tmp1, ptr %p
	ret void
}

define void @shl_i8(i8 %a, i8 %c, ptr %p) nounwind {
  %tmp = shl i8 %a, %c
  store i8 %tmp, ptr %p
  ret void
}

define void @mul_i8(i8 %a, ptr %p) nounwind {
  %tmp = mul i8 %a, 17
  store i8 %tmp, ptr %p
  ret void
}

define void @load_store_i1(ptr %p, ptr %q) nounwind {
  %t = load i1, ptr %p
  store i1 %t, ptr %q
  ret void
}

define void @freeze_i32(i32 %x) {
  %t = freeze i32 %x
  ret void
}

@crash_test1x = external global <2 x i32>, align 8

define void @crash_test1() nounwind ssp {
  %tmp = load <2 x i32>, ptr @crash_test1x, align 8
  %neg = xor <2 x i32> %tmp, <i32 -1, i32 -1>
  ret void
}

declare void @llvm.lifetime.start.p0(i64, ptr nocapture) nounwind

define ptr @life() nounwind {
  %a1 = alloca ptr, align 8
  call void @llvm.lifetime.start.p0(i64 -1, ptr %a1) nounwind      
  %a3 = load ptr, ptr %a1, align 8
  ret ptr %a3
}

declare void @llvm.donothing() readnone

; CHECK: donada
define void @donada() nounwind {
entry:
  call void @llvm.donothing()
  ret void
}
