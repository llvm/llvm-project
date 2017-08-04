; RUN: llc -relocation-model=static -verify-machineinstrs < %s -mtriple=ppc64-- -tailcallopt | grep TC_RETURNd8
; RUN: llc -relocation-model=static -verify-machineinstrs -mtriple=ppc64-- < %s | FileCheck %s
define fastcc i32 @tailcallee(i32 %a1, i32 %a2, i32 %a3, i32 %a4) {
entry:
	ret i32 %a3
}

define fastcc i32 @tailcaller(i32 %in1, i32 %in2) {
entry:
	%tmp11 = tail call fastcc i32 @tailcallee( i32 %in1, i32 %in2, i32 %in1, i32 %in2 )
	ret i32 %tmp11
; CHECK-LABEL: tailcaller
; CHECK-NOT: stdu
}
