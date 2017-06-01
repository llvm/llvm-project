; RUN: llc < %s -relocation-model=static -mcpu=yonah | FileCheck %s

; The double argument is at 4(esp) which is 16-byte aligned, but we
; are required to read in extra bytes of memory in order to fold the
; load. Bad Things may happen when reading/processing undefined bytes,
; so don't fold the load.
; PR22371 / http://reviews.llvm.org/D7474

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin8"
@G = external global double

define void @test({ double, double }* byval  %z, double* %P) nounwind {
entry:
	%tmp3 = load double, double* @G, align 16		; <double> [#uses=1]
	%tmp4 = tail call double @fabs( double %tmp3 ) readnone	; <double> [#uses=1]
        store volatile double %tmp4, double* %P
	%tmp = getelementptr { double, double }, { double, double }* %z, i32 0, i32 0		; <double*> [#uses=1]
	%tmp1 = load volatile double, double* %tmp, align 8		; <double> [#uses=1]
	%tmp2 = tail call double @fabs( double %tmp1 ) readnone	; <double> [#uses=1]
	%tmp6 = fadd double %tmp4, %tmp2		; <double> [#uses=1]
	store volatile double %tmp6, double* %P, align 8
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
  %0 = ptrtoint [2048 x i8]* %buffer to i32
  %and = and i32 %0, -16
  ret i32 %and
; CHECK-LABEL: test4:
; CHECK-NOT: and
; CHECK: ret
}

%struct.sixteen = type { [16 x i8] }

; Accessing stack parameters shouldn't assume stack alignment. Here we should
; emit two 8-byte loads, followed by two 8-byte stores.
define x86_stdcallcc void @test5(%struct.sixteen* byval nocapture readonly align 4 %s) #0 {
  %d.sroa.0 = alloca [16 x i8], align 1
  %1 = getelementptr inbounds [16 x i8], [16 x i8]* %d.sroa.0, i32 0, i32 0
  call void @llvm.lifetime.start(i64 16, i8* %1)
  %2 = getelementptr inbounds %struct.sixteen, %struct.sixteen* %s, i32 0, i32 0, i32 0
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %1, i8* %2, i32 16, i32 1, i1 true)
  call void @llvm.lifetime.end(i64 16, i8* %1)
  ret void
; CHECK-LABEL: test5:
; CHECK: and
; CHECK: movsd
; CHECK-NEXT: movsd
; CHECK-NEXT: movsd
; CHECK-NEXT: movsd
}

declare void @llvm.lifetime.start(i64, i8* nocapture) argmemonly nounwind

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture readonly, i32, i32, i1) argmemonly nounwind

declare void @llvm.lifetime.end(i64, i8* nocapture) argmemonly nounwind

attributes #0 = { nounwind alignstack=16 "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" }
