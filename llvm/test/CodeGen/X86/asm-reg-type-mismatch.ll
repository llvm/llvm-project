; RUN: llc < %s -mcpu=core2 | FileCheck %s
; PR2715

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"
	%struct.XPTTypeDescriptorPrefix = type { i8 }
	%struct.nsISupports = type { i32 (...)** }
	%struct.nsXPTCMiniVariant = type { %"struct.nsXPTCMiniVariant::._39" }
	%"struct.nsXPTCMiniVariant::._39" = type { i64 }
	%struct.nsXPTCVariant = type { %struct.nsXPTCMiniVariant, i8*, %struct.nsXPTType, i8 }
	%struct.nsXPTType = type { %struct.XPTTypeDescriptorPrefix }

define i32 @test1(%struct.nsISupports* %that, i32 %methodIndex, i32 %paramCount, %struct.nsXPTCVariant* %params) nounwind {
entry:
	call void asm sideeffect "", "{xmm0},{xmm1},{xmm2},{xmm3},{xmm4},{xmm5},{xmm6},{xmm7},~{dirflag},~{fpsr},~{flags}"( double undef, double undef, double undef, double 1.0, double undef, double 0.0, double undef, double 0.0 ) nounwind
	ret i32 0
	; CHECK: test1
	; CHECK-NOT: movap
	; CHECK: xorps
	; CHECK: xorps
	; CHECK-NOT: movap
}

define i64 @test2() nounwind {
entry:
  %0 = tail call i64 asm sideeffect "movq $1, $0", "={xmm7},*m,~{dirflag},~{fpsr},~{flags}"(i64* null) nounwind
  ret i64 %0
  ; CHECK: test2
	; CHECK: movq {{.*}}, %xmm7
	; CHECK: movq %xmm7, %rax
}
