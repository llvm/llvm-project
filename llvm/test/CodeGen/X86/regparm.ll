; RUN: llc %s -mtriple=i386-pc-linux -o - | FileCheck %s 
; RUN: llc %s -mtriple=i386-pc-win32 -o - | FileCheck -check-prefix=WIN %s
; RUN: llc %s -mtriple=i386-pc-linux -fast-isel -o - | FileCheck -check-prefix=FAST %s 
; RUN: llc %s -mtriple=i386-pc-win32 -fast-isel -o - | FileCheck -check-prefix=FASTWIN %s



target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386-unknown-linux-gnu"

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture writeonly, i8* nocapture readonly, i32, i1) #1

define void @use_memset(i8* inreg nocapture %dest, i8 inreg %c, i32 inreg %n) local_unnamed_addr #0 {
entry:
;CHECK-LABEL: @use_memset
;CHECK-NOT: push
;CHECK: jmp	memset
;CHECK-NOT: retl
;WIN-LABEL: @use_memset
;WIN-NOT: push
;WIN: jmp	_memset
;WIN-NOT: retl
;FAST-LABEL: @use_memset
;FAST:	subl	$12, %esp
;FAST-NEXT: 	movzbl	%dl, %edx
;FAST-NEXT:     calll	memset
;FAST-NEXT:	addl	$12, %esp
;FASTWIN-LABEL: @use_memset
;FASTWIN: 	movzbl	%dl, %edx
;FASTWIN-NEXT:     calll	_memset
;FASTWIN-NEXT:     retl
  tail call void @llvm.memset.p0i8.i32(i8* %dest, i8 %c, i32 %n, i1 false)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i32(i8* nocapture writeonly, i8, i32, i1) #1


attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"NumRegisterParameters", i32 3}
!1 = !{!"clang version 4.0.0 (trunk 288025) (llvm/trunk 288033)"}
