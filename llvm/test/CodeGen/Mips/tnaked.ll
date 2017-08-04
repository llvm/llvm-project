; RUN: llc -march=mipsel < %s -verify-machineinstrs | FileCheck %s


define void @tnaked() #0 {
entry:
  ret void
}

; CHECK: 	.ent	tnaked
; CHECK-LABEL:          tnaked:
; CHECK-NOT:	.frame	{{.*}}
; CHECK-NOT:     .mask 	{{.*}}
; CHECK-NOT:	.fmask	{{.*}}
; CHECK-NOT:	 addiu	$sp, $sp, -8

define void @tnonaked() #1 {
entry:
  ret void
}

; CHECK: 	.ent	tnonaked
; CHECK-LABEL:         tnonaked:
; CHECK:	.frame	$fp,8,$ra
; CHECK:        .mask 	0x40000000,-4
; CHECK:	.fmask	0x00000000,0
; CHECK: 	addiu	$sp, $sp, -8

attributes #0 = { naked noinline nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
