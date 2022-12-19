; RUN: llc -mtriple=mipsel-linux-gnu -march=mipsel -mattr=mips16 -mattr=+soft-float -mips16-hard-float -relocation-model=static     < %s | FileCheck %s -check-prefix=ci

@i = global i32 0, align 4
@j = common global i32 0, align 4
@k = common global i32 0, align 4

; Function Attrs: nounwind optsize
define i32 @x0() #0 {
entry:
  %0 = load i32, ptr @i, align 4, !tbaa !1
  %cmp = icmp eq i32 %0, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  tail call void asm sideeffect ".space 1000", ""() #1, !srcloc !5
  br label %if.end

if.else:                                          ; preds = %entry
  tail call void asm sideeffect ".space 1004", ""() #1, !srcloc !6
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %storemerge = phi i32 [ 1, %if.else ], [ 0, %if.then ]
  store i32 %storemerge, ptr @i, align 4, !tbaa !1
  ret i32 0
}

; ci:	.ent	x0
; ci: 	beqz	$3, $BB0_2
; ci: $BB0_2:
; ci:	.end	x0

; Function Attrs: nounwind optsize
define i32 @x1() #0 {
entry:
  %0 = load i32, ptr @i, align 4, !tbaa !1
  %cmp = icmp eq i32 %0, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  tail call void asm sideeffect ".space 1000000", ""() #1, !srcloc !7
  br label %if.end

if.else:                                          ; preds = %entry
  tail call void asm sideeffect ".space 1000004", ""() #1, !srcloc !8
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %storemerge = phi i32 [ 1, %if.else ], [ 0, %if.then ]
  store i32 %storemerge, ptr @i, align 4, !tbaa !1
  ret i32 0
}

; ci:	.ent	x1
; ci:	bnez	$3, $BB1_1  # 16 bit inst
; ci:	jal	$BB1_2	# branch
; ci:	nop
; ci: $BB1_1:
; ci:	.end	x1

; Function Attrs: nounwind optsize
define i32 @y0() #0 {
entry:
  %0 = load i32, ptr @i, align 4, !tbaa !1
  %cmp = icmp eq i32 %0, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  store i32 10, ptr @j, align 4, !tbaa !1
  tail call void asm sideeffect ".space 1000", ""() #1, !srcloc !9
  br label %if.end

if.else:                                          ; preds = %entry
  store i32 55, ptr @j, align 4, !tbaa !1
  tail call void asm sideeffect ".space 1004", ""() #1, !srcloc !10
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret i32 0
}

; ci:	.ent	y0
; ci:	beqz	$2, $BB2_2
; ci:	.end	y0

; Function Attrs: nounwind optsize
define i32 @y1() #0 {
entry:
  %0 = load i32, ptr @i, align 4, !tbaa !1
  %cmp = icmp eq i32 %0, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  store i32 10, ptr @j, align 4, !tbaa !1
  tail call void asm sideeffect ".space 1000000", ""() #1, !srcloc !11
  br label %if.end

if.else:                                          ; preds = %entry
  store i32 55, ptr @j, align 4, !tbaa !1
  tail call void asm sideeffect ".space 1000004", ""() #1, !srcloc !12
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret i32 0
}

; ci:	.ent	y1
; ci:	bnez	$2, $BB3_1  # 16 bit inst
; ci:	jal	$BB3_2	# branch
; ci:	nop
; ci: $BB3_1:
; ci:	.end	y1

; Function Attrs: nounwind optsize
define void @z0() #0 {
entry:
  %0 = load i32, ptr @i, align 4, !tbaa !1
  %1 = load i32, ptr @j, align 4, !tbaa !1
  %cmp = icmp eq i32 %0, %1
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  store i32 1, ptr @k, align 4, !tbaa !1
  tail call void asm sideeffect ".space 10000", ""() #1, !srcloc !13
  br label %if.end

if.else:                                          ; preds = %entry
  tail call void asm sideeffect ".space 10004", ""() #1, !srcloc !14
  store i32 2, ptr @k, align 4, !tbaa !1
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

; ci:	.ent	z0
; ci:	btnez	$BB4_2
; ci:	.end	z0

; Function Attrs: nounwind optsize
define void @z1() #0 {
entry:
  %0 = load i32, ptr @i, align 4, !tbaa !1
  %1 = load i32, ptr @j, align 4, !tbaa !1
  %cmp = icmp eq i32 %0, %1
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  store i32 1, ptr @k, align 4, !tbaa !1
  tail call void asm sideeffect ".space 10000000", ""() #1, !srcloc !15
  br label %if.end

if.else:                                          ; preds = %entry
  tail call void asm sideeffect ".space 10000004", ""() #1, !srcloc !16
  store i32 2, ptr @k, align 4, !tbaa !1
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

; ci:	.ent	z1
; ci:	bteqz	$BB5_1  # 16 bit inst
; ci:	jal	$BB5_2	# branch
; ci:	nop
; ci: $BB5_1:
; ci:	.end	z1

; Function Attrs: nounwind optsize
define void @z3() #0 {
entry:
  %0 = load i32, ptr @i, align 4, !tbaa !1
  %1 = load i32, ptr @j, align 4, !tbaa !1
  %cmp1 = icmp sgt i32 %0, %1
  br i1 %cmp1, label %if.then, label %if.end

if.then:                                          ; preds = %entry, %if.then
  tail call void asm sideeffect ".space 10000", ""() #1, !srcloc !17
  %2 = load i32, ptr @i, align 4, !tbaa !1
  %3 = load i32, ptr @j, align 4, !tbaa !1
  %cmp = icmp sgt i32 %2, %3
  br i1 %cmp, label %if.then, label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

; ci:	.ent	z3
; ci:	bteqz	$BB6_2
; ci:	.end	z3

; Function Attrs: nounwind optsize
define void @z4() #0 {
entry:
  %0 = load i32, ptr @i, align 4, !tbaa !1
  %1 = load i32, ptr @j, align 4, !tbaa !1
  %cmp1 = icmp sgt i32 %0, %1
  br i1 %cmp1, label %if.then, label %if.end

if.then:                                          ; preds = %entry, %if.then
  tail call void asm sideeffect ".space 10000000", ""() #1, !srcloc !18
  %2 = load i32, ptr @i, align 4, !tbaa !1
  %3 = load i32, ptr @j, align 4, !tbaa !1
  %cmp = icmp sgt i32 %2, %3
  br i1 %cmp, label %if.then, label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

; ci:	.ent	z4
; ci:	btnez	$BB7_1  # 16 bit inst
; ci:	jal	$BB7_2	# branch
; ci:	nop
; ci:	.p2align	2
; ci: $BB7_1:
; ci:	.end	z4

attributes #0 = { nounwind optsize "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }


!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{i32 57}
!6 = !{i32 107}
!7 = !{i32 188}
!8 = !{i32 241}
!9 = !{i32 338}
!10 = !{i32 391}
!11 = !{i32 477}
!12 = !{i32 533}
!13 = !{i32 621}
!14 = !{i32 663}
!15 = !{i32 747}
!16 = !{i32 792}
!17 = !{i32 867}
!18 = !{i32 953}
