; RUN: llc < %s -relocation-model=static -mcpu=yonah | FileCheck %s

; The double argument is at 4(esp) which is 16-byte aligned, but we
; are required to read in extra bytes of memory in order to fold the
; load. Bad Things may happen when reading/processing undefined bytes,
; so don't fold the load.
; PR22371 / http://reviews.llvm.org/D7474

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin8"
@G = external global double

define void @test(ptr byval({ double, double })  %z, ptr %P) nounwind {
entry:
	%tmp3 = load double, ptr @G, align 16		; <double> [#uses=1]
	%tmp4 = tail call double @fabs( double %tmp3 ) readnone	; <double> [#uses=1]
        store volatile double %tmp4, ptr %P
	%tmp = getelementptr { double, double }, ptr %z, i32 0, i32 0		; <ptr> [#uses=1]
	%tmp1 = load volatile double, ptr %tmp, align 8		; <double> [#uses=1]
	%tmp2 = tail call double @fabs( double %tmp1 ) readnone	; <double> [#uses=1]
	%tmp6 = fadd double %tmp4, %tmp2		; <double> [#uses=1]
	store volatile double %tmp6, ptr %P, align 8
	ret void

; CHECK-LABEL: test:
; CHECK:       movsd	{{.*}}G, %xmm{{.*}}
; CHECK:       andpd	%xmm{{.*}}, %xmm{{.*}}
; CHECK:       movsd	4(%esp), %xmm{{.*}}
; CHECK:       andpd	%xmm{{.*}}, %xmm{{.*}}


}

define void @test2() alignstack(16) nounwind {
entry:
; CHECK-LABEL: test2:
; CHECK: andl{{.*}}$-16, %esp
    ret void
}

; Use a call to force a spill.
define <2 x double> @test3(<2 x double> %x, <2 x double> %y) alignstack(32) nounwind {
entry:
; CHECK-LABEL: test3:
; CHECK: andl{{.*}}$-32, %esp
    call void @test2()
    %A = fmul <2 x double> %x, %y
    ret <2 x double> %A
}

declare double @fabs(double)

; The pointer is already known aligned, so and x,-16 is eliminable.
define i32 @test4() nounwind {
entry:
  %buffer = alloca [2048 x i8], align 16
  %0 = ptrtoint ptr %buffer to i32
  %and = and i32 %0, -16
  ret i32 %and
; CHECK-LABEL: test4:
; CHECK-NOT: and
; CHECK: ret
}

%struct.sixteen = type { [16 x i8] }

; Accessing stack parameters shouldn't assume stack alignment. Here we should
; emit two 8-byte loads, followed by two 8-byte stores.
define x86_stdcallcc void @test5(ptr byval(%struct.sixteen) nocapture readonly align 4 %s) #0 {
  %d.sroa.0 = alloca [16 x i8], align 1
  call void @llvm.lifetime.start.p0(i64 16, ptr %d.sroa.0)
  call void @llvm.memcpy.p0.p0.i32(ptr %d.sroa.0, ptr %s, i32 16, i1 true)
  call void @llvm.lifetime.end.p0(i64 16, ptr %d.sroa.0)
  ret void
; CHECK-LABEL: test5:
; CHECK: and
; CHECK: movsd
; CHECK-NEXT: movsd
; CHECK-NEXT: movsd
; CHECK-NEXT: movsd
}

declare void @llvm.lifetime.start.p0(i64, ptr nocapture) argmemonly nounwind

declare void @llvm.memcpy.p0.p0.i32(ptr nocapture, ptr nocapture readonly, i32, i1) argmemonly nounwind

declare void @llvm.lifetime.end.p0(i64, ptr nocapture) argmemonly nounwind

attributes #0 = { nounwind alignstack=16 "frame-pointer"="all" }
