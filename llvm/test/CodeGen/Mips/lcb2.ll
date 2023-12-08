; RUN: llc -mtriple=mipsel-linux-gnu -march=mipsel -mattr=mips16 -mattr=+soft-float -mips16-hard-float -relocation-model=static -mips16-constant-islands=true   < %s | FileCheck %s -check-prefix=lcb

; RUN: llc -mtriple=mipsel-linux-gnu -march=mipsel -mattr=mips16 -mattr=+soft-float -mips16-hard-float -relocation-model=static -mips16-constant-islands=true   < %s | FileCheck %s -check-prefix=lcbn

@i = global i32 0, align 4
@j = common global i32 0, align 4
@k = common global i32 0, align 4

; Function Attrs: nounwind optsize
define i32 @bnez() #0 {
entry:
  %0 = load i32, ptr @i, align 4, !tbaa !1
  %cmp = icmp eq i32 %0, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void asm sideeffect ".space 10000", ""() #1, !srcloc !5
  store i32 0, ptr @i, align 4, !tbaa !1
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret i32 0
}
; lcb: 	.ent	bnez
; lcbn:	.ent	bnez
; lcb:	bnez	${{[0-9]+}}, $BB{{[0-9]+}}_{{[0-9]+}}
; lcbn-NOT: bnez	${{[0-9]+}}, $BB{{[0-9]+}}_{{[0-9]+}}  # 16 bit inst
; lcb: 	.end	bnez
; lcbn:	.end	bnez

; Function Attrs: nounwind optsize
define i32 @beqz() #0 {
entry:
  %0 = load i32, ptr @i, align 4, !tbaa !1
  %cmp = icmp eq i32 %0, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  store i32 10, ptr @j, align 4, !tbaa !1
  tail call void asm sideeffect ".space 10000", ""() #1, !srcloc !6
  br label %if.end

if.else:                                          ; preds = %entry
  store i32 55, ptr @j, align 4, !tbaa !1
  tail call void asm sideeffect ".space 10000", ""() #1, !srcloc !7
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret i32 0
}

; lcb: 	.ent	beqz
; lcbn:	.ent	beqz
; lcb:	beqz	${{[0-9]+}}, $BB{{[0-9]+}}_{{[0-9]+}}
; lcbn-NOT: beqz	${{[0-9]+}}, $BB{{[0-9]+}}_{{[0-9]+}}  # 16 bit inst
; lcb: 	.end	beqz
; lcbn:	.end	beqz


; Function Attrs: nounwind optsize
define void @bteqz() #0 {
entry:
  %0 = load i32, ptr @i, align 4, !tbaa !1
  %1 = load i32, ptr @j, align 4, !tbaa !1
  %cmp = icmp eq i32 %0, %1
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  store i32 1, ptr @k, align 4, !tbaa !1
  tail call void asm sideeffect ".space 1000", ""() #1, !srcloc !8
  br label %if.end

if.else:                                          ; preds = %entry
  tail call void asm sideeffect ".space 1000", ""() #1, !srcloc !9
  store i32 2, ptr @k, align 4, !tbaa !1
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

; lcb: 	.ent	bteqz
; lcbn:	.ent	bteqz
; lcb:	btnez	$BB{{[0-9]+}}_{{[0-9]+}}
; lcbn-NOT: btnez	$BB{{[0-9]+}}_{{[0-9]+}} # 16 bit inst
; lcb: 	.end	bteqz
; lcbn:	.end	bteqz


; Function Attrs: nounwind optsize
define void @btz() #0 {
entry:
  %0 = load i32, ptr @i, align 4, !tbaa !1
  %1 = load i32, ptr @j, align 4, !tbaa !1
  %cmp1 = icmp sgt i32 %0, %1
  br i1 %cmp1, label %if.then, label %if.end

if.then:                                          ; preds = %entry, %if.then
  tail call void asm sideeffect ".space 60000", ""() #1, !srcloc !10
  %2 = load i32, ptr @i, align 4, !tbaa !1
  %3 = load i32, ptr @j, align 4, !tbaa !1
  %cmp = icmp sgt i32 %2, %3
  br i1 %cmp, label %if.then, label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

; lcb: 	.ent	btz
; lcbn:	.ent	btz
; lcb:	bteqz	$BB{{[0-9]+}}_{{[0-9]+}}
; lcbn-NOT: bteqz	$BB{{[0-9]+}}_{{[0-9]+}} # 16 bit inst
; lcb:	btnez	$BB{{[0-9]+}}_{{[0-9]+}}
; lcbn-NOT: btnez	$BB{{[0-9]+}}_{{[0-9]+}} # 16 bit inst
; lcb: 	.end	btz
; lcbn:	.end	btz

attributes #0 = { nounwind optsize "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.5 (gitosis@dmz-portal.mips.com:clang.git ed197d08c90d82e1119774e10920e6f7a841c8ec) (gitosis@dmz-portal.mips.com:llvm.git b9235a363fa2dddb26ac01cbaed58efbc9eff392)"}
!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{i32 59}
!6 = !{i32 156}
!7 = !{i32 210}
!8 = !{i32 299}
!9 = !{i32 340}
!10 = !{i32 412}
