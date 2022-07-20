; RUN: llc < %s -mtriple=i386-apple-darwin | FileCheck %s
; Make sure xorl operands are 32-bit registers.

	%struct.tm = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8* }
	%struct.wxDateTime = type { %struct.wxLongLong }
	%"struct.wxDateTime::TimeZone" = type { i32 }
	%struct.wxLongLong = type { i64 }
	%struct.wxString = type { %struct.wxStringBase }
	%struct.wxStringBase = type { i32* }
@.str = external constant [27 x i32]		; <[27 x i32]*> [#uses=1]
@.str4 = external constant [14 x i32]		; <[14 x i32]*> [#uses=1]
@_ZZNK10wxDateTime5GetTmERKNS_8TimeZoneEE12__FUNCTION__ = external constant [6 x i8]		; <[6 x i8]*> [#uses=1]
@.str33 = external constant [29 x i32]		; <[29 x i32]*> [#uses=1]
@.str89 = external constant [5 x i32]		; <[5 x i32]*> [#uses=1]

define void @_ZNK10wxDateTime6FormatEPKwRKNS_8TimeZoneE(%struct.wxString* noalias sret(%struct.wxString)  %agg.result, %struct.wxDateTime* %this, i32* %format, %"struct.wxDateTime::TimeZone"* %tz, i1 %foo) personality i32 (...)* @__gxx_personality_v0 {
; CHECK-LABEL: _ZNK10wxDateTime6FormatEPKwRKNS_8TimeZoneE:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:    pushl %ebp
; CHECK-NEXT:    .cfi_def_cfa_offset 8
; CHECK-NEXT:    pushl %ebx
; CHECK-NEXT:    .cfi_def_cfa_offset 12
; CHECK-NEXT:    pushl %edi
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; CHECK-NEXT:    pushl %esi
; CHECK-NEXT:    .cfi_def_cfa_offset 20
; CHECK-NEXT:    subl $28, %esp
; CHECK-NEXT:    .cfi_def_cfa_offset 48
; CHECK-NEXT:    .cfi_offset %esi, -20
; CHECK-NEXT:    .cfi_offset %edi, -16
; CHECK-NEXT:    .cfi_offset %ebx, -12
; CHECK-NEXT:    .cfi_offset %ebp, -8
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %esi
; CHECK-NEXT:    movzbl {{[0-9]+}}(%esp), %ebx
; CHECK-NEXT:    testb $1, %bl
; CHECK-NEXT:    je LBB0_25
; CHECK-NEXT:  ## %bb.1: ## %bb116.i
; CHECK-NEXT:    je LBB0_25
; CHECK-NEXT:  ## %bb.2: ## %bb52.i.i
; CHECK-NEXT:    testb $1, %bl
; CHECK-NEXT:    je LBB0_25
; CHECK-NEXT:  ## %bb.3: ## %bb142.i
; CHECK-NEXT:    je LBB0_25
; CHECK-NEXT:  ## %bb.4:
; CHECK-NEXT:    movl L_.str89$non_lazy_ptr, %edi
; CHECK-NEXT:    movb $1, %bh
; CHECK-NEXT:    movl L_.str$non_lazy_ptr, %ebp
; CHECK-NEXT:    jmp LBB0_5
; CHECK-NEXT:  LBB0_21: ## %bb7806
; CHECK-NEXT:    ## in Loop: Header=BB0_5 Depth=1
; CHECK-NEXT:  Ltmp16:
; CHECK-NEXT:    movl $0, {{[0-9]+}}(%esp)
; CHECK-NEXT:    movl $1, {{[0-9]+}}(%esp)
; CHECK-NEXT:    movl $0, (%esp)
; CHECK-NEXT:    calll __ZN12wxStringBase6appendEmw
; CHECK-NEXT:  Ltmp17:
; CHECK-NEXT:  LBB0_5: ## %bb3261
; CHECK-NEXT:    ## =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    cmpl $37, 0
; CHECK-NEXT:    jne LBB0_25
; CHECK-NEXT:  ## %bb.6: ## %bb3306
; CHECK-NEXT:    ## in Loop: Header=BB0_5 Depth=1
; CHECK-NEXT:  Ltmp0:
; CHECK-NEXT:    movl %edi, {{[0-9]+}}(%esp)
; CHECK-NEXT:    movl $0, (%esp)
; CHECK-NEXT:    calll __ZN12wxStringBaseaSEPKw
; CHECK-NEXT:  Ltmp1:
; CHECK-NEXT:  ## %bb.7: ## %bb3314
; CHECK-NEXT:    ## in Loop: Header=BB0_5 Depth=1
; CHECK-NEXT:    movl 0, %eax
; CHECK-NEXT:    cmpl $121, %eax
; CHECK-NEXT:    ja LBB0_25
; CHECK-NEXT:  ## %bb.8: ## %bb3314
; CHECK-NEXT:    ## in Loop: Header=BB0_5 Depth=1
; CHECK-NEXT:    jmpl *LJTI0_0(,%eax,4)
; CHECK-NEXT:  LBB0_10: ## %bb5809
; CHECK-NEXT:    ## in Loop: Header=BB0_5 Depth=1
; CHECK-NEXT:    xorl %eax, %eax
; CHECK-NEXT:    testb %al, %al
; CHECK-NEXT:    jne LBB0_25
; CHECK-NEXT:  ## %bb.11: ## %bb5809
; CHECK-NEXT:    ## in Loop: Header=BB0_5 Depth=1
; CHECK-NEXT:    testb %bh, %bh
; CHECK-NEXT:    je LBB0_25
; CHECK-NEXT:  ## %bb.12: ## %bb91.i8504
; CHECK-NEXT:    ## in Loop: Header=BB0_5 Depth=1
; CHECK-NEXT:    testb $1, %bl
; CHECK-NEXT:    je LBB0_14
; CHECK-NEXT:  ## %bb.13: ## %bb155.i8541
; CHECK-NEXT:    ## in Loop: Header=BB0_5 Depth=1
; CHECK-NEXT:  Ltmp4:
; CHECK-NEXT:    movl $0, {{[0-9]+}}(%esp)
; CHECK-NEXT:    movl $0, (%esp)
; CHECK-NEXT:    calll _gmtime_r
; CHECK-NEXT:  Ltmp5:
; CHECK-NEXT:  LBB0_14: ## %bb182.i8560
; CHECK-NEXT:    ## in Loop: Header=BB0_5 Depth=1
; CHECK-NEXT:    testb $1, %bl
; CHECK-NEXT:    je LBB0_15
; CHECK-NEXT:  ## %bb.16: ## %bb278.i8617
; CHECK-NEXT:    ## in Loop: Header=BB0_5 Depth=1
; CHECK-NEXT:    je LBB0_18
; CHECK-NEXT:  ## %bb.17: ## %bb440.i8663
; CHECK-NEXT:    ## in Loop: Header=BB0_5 Depth=1
; CHECK-NEXT:  Ltmp6:
; CHECK-NEXT:    movl L_.str4$non_lazy_ptr, %eax
; CHECK-NEXT:    movl %eax, {{[0-9]+}}(%esp)
; CHECK-NEXT:    movl L_.str33$non_lazy_ptr, %eax
; CHECK-NEXT:    movl %eax, {{[0-9]+}}(%esp)
; CHECK-NEXT:    movl L__ZZNK10wxDateTime5GetTmERKNS_8TimeZoneEE12__FUNCTION__$non_lazy_ptr, %eax
; CHECK-NEXT:    movl %eax, {{[0-9]+}}(%esp)
; CHECK-NEXT:    movl %ebp, (%esp)
; CHECK-NEXT:    movl $1717, {{[0-9]+}}(%esp) ## imm = 0x6B5
; CHECK-NEXT:    calll __Z10wxOnAssertPKwiPKcS0_S0_
; CHECK-NEXT:  Ltmp7:
; CHECK-NEXT:    jmp LBB0_18
; CHECK-NEXT:  LBB0_15: ## %bb187.i8591
; CHECK-NEXT:    ## in Loop: Header=BB0_5 Depth=1
; CHECK-NEXT:    jne LBB0_25
; CHECK-NEXT:  LBB0_18: ## %invcont5814
; CHECK-NEXT:    ## in Loop: Header=BB0_5 Depth=1
; CHECK-NEXT:  Ltmp8:
; CHECK-NEXT:    movl $0, {{[0-9]+}}(%esp)
; CHECK-NEXT:    movl $0, {{[0-9]+}}(%esp)
; CHECK-NEXT:    movl $0, (%esp)
; CHECK-NEXT:    calll __ZN8wxString6FormatEPKwz
; CHECK-NEXT:    subl $4, %esp
; CHECK-NEXT:  Ltmp9:
; CHECK-NEXT:  ## %bb.19: ## %invcont5831
; CHECK-NEXT:    ## in Loop: Header=BB0_5 Depth=1
; CHECK-NEXT:  Ltmp10:
; CHECK-NEXT:    movl $0, {{[0-9]+}}(%esp)
; CHECK-NEXT:    movl $0, {{[0-9]+}}(%esp)
; CHECK-NEXT:    movl $0, {{[0-9]+}}(%esp)
; CHECK-NEXT:    movl $0, (%esp)
; CHECK-NEXT:    calll __ZN12wxStringBase10ConcatSelfEmPKwm
; CHECK-NEXT:  Ltmp11:
; CHECK-NEXT:    jmp LBB0_5
; CHECK-NEXT:  LBB0_9: ## %bb5657
; CHECK-NEXT:  Ltmp13:
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %eax
; CHECK-NEXT:    movl %eax, {{[0-9]+}}(%esp)
; CHECK-NEXT:    movl {{[0-9]+}}(%esp), %eax
; CHECK-NEXT:    movl %eax, (%esp)
; CHECK-NEXT:    calll __ZNK10wxDateTime12GetDayOfYearERKNS_8TimeZoneE
; CHECK-NEXT:  Ltmp14:
; CHECK-NEXT:    jmp LBB0_25
; CHECK-NEXT:  LBB0_20: ## %bb5968
; CHECK-NEXT:  Ltmp2:
; CHECK-NEXT:    movl $0, {{[0-9]+}}(%esp)
; CHECK-NEXT:    movl $0, {{[0-9]+}}(%esp)
; CHECK-NEXT:    movl $0, (%esp)
; CHECK-NEXT:    calll __ZN8wxString6FormatEPKwz
; CHECK-NEXT:    subl $4, %esp
; CHECK-NEXT:  Ltmp3:
; CHECK-NEXT:  LBB0_25: ## %bb115.critedge.i
; CHECK-NEXT:    movl %esi, %eax
; CHECK-NEXT:    addl $28, %esp
; CHECK-NEXT:    popl %esi
; CHECK-NEXT:    popl %edi
; CHECK-NEXT:    popl %ebx
; CHECK-NEXT:    popl %ebp
; CHECK-NEXT:    retl $4
; CHECK-NEXT:  LBB0_23: ## %lpad.loopexit.split-lp
; CHECK-NEXT:  Ltmp15:
; CHECK-NEXT:    jmp LBB0_25
; CHECK-NEXT:  LBB0_24: ## %lpad8185
; CHECK-NEXT:  Ltmp12:
; CHECK-NEXT:    jmp LBB0_25
; CHECK-NEXT:  LBB0_22: ## %lpad.loopexit
; CHECK-NEXT:  Ltmp18:
; CHECK-NEXT:    jmp LBB0_25
; CHECK-NEXT:  Lfunc_end0:
entry:
	br i1 %foo, label %bb116.i, label %bb115.critedge.i
bb115.critedge.i:		; preds = %entry
	ret void
bb116.i:		; preds = %entry
	br i1 %foo, label %bb52.i.i, label %bb3118
bb3118:		; preds = %bb116.i
	ret void
bb52.i.i:		; preds = %bb116.i
	br i1 %foo, label %bb142.i, label %bb115.critedge.i.i
bb115.critedge.i.i:		; preds = %bb52.i.i
	ret void
bb142.i:		; preds = %bb52.i.i
	br i1 %foo, label %bb161.i, label %bb182.i
bb161.i:		; preds = %bb142.i
	br label %bb3261
bb182.i:		; preds = %bb142.i
	ret void
bb3261:		; preds = %bb7834, %bb161.i
	%tmp3263 = load i32, i32* null, align 4		; <i32> [#uses=1]
	%tmp3264 = icmp eq i32 %tmp3263, 37		; <i1> [#uses=1]
	br i1 %tmp3264, label %bb3306, label %bb3267
bb3267:		; preds = %bb3261
	ret void
bb3306:		; preds = %bb3261
	%tmp3310 = invoke %struct.wxStringBase* @_ZN12wxStringBaseaSEPKw( %struct.wxStringBase* null, i32* getelementptr ([5 x i32], [5 x i32]* @.str89, i32 0, i32 0) )
			to label %bb3314 unwind label %lpad		; <%struct.wxStringBase*> [#uses=0]
bb3314:		; preds = %bb3306
	%tmp3316 = load i32, i32* null, align 4		; <i32> [#uses=1]
	switch i32 %tmp3316, label %bb7595 [
		 i32 0, label %bb7819
		 i32 37, label %bb7806
		 i32 66, label %bb3477
		 i32 72, label %bb5334
		 i32 73, label %bb5484
		 i32 77, label %bb6118
		 i32 83, label %bb6406
		 i32 85, label %bb6556
		 i32 87, label %bb6708
		 i32 89, label %bb7308
		 i32 98, label %bb3477
		 i32 99, label %bb3626
		 i32 100, label %bb5184
		 i32 106, label %bb5657
		 i32 108, label %bb5809
		 i32 109, label %bb5968
		 i32 119, label %bb6860
		 i32 120, label %bb3626
		 i32 121, label %bb7158
	]
bb3477:		; preds = %bb3314, %bb3314
	ret void
bb3626:		; preds = %bb3314, %bb3314
	ret void
bb5184:		; preds = %bb3314
	ret void
bb5334:		; preds = %bb3314
	ret void
bb5484:		; preds = %bb3314
	ret void
bb5657:		; preds = %bb3314
	%tmp5661 = invoke zeroext i16 @_ZNK10wxDateTime12GetDayOfYearERKNS_8TimeZoneE( %struct.wxDateTime* %this, %"struct.wxDateTime::TimeZone"* %tz )
			to label %invcont5660 unwind label %lpad		; <i16> [#uses=0]
invcont5660:		; preds = %bb5657
	ret void
bb5809:		; preds = %bb3314
	%tmp61.i.i8486 = icmp sgt i64 0, -1		; <i1> [#uses=1]
	%tmp95.i.i8490 = icmp slt i64 0, 2147483647000		; <i1> [#uses=1]
	%bothcond9308 = and i1 %tmp61.i.i8486, %tmp95.i.i8490		; <i1> [#uses=1]
	br i1 %bothcond9308, label %bb91.i8504, label %bb115.critedge.i.i8492
bb115.critedge.i.i8492:		; preds = %bb5809
	ret void
bb91.i8504:		; preds = %bb5809
	br i1 %foo, label %bb155.i8541, label %bb182.i8560
bb155.i8541:		; preds = %bb91.i8504
	%tmp156.i85398700 = invoke %struct.tm* @gmtime_r( i32* null, %struct.tm* null )
			to label %bb182.i8560 unwind label %lpad		; <%struct.tm*> [#uses=1]
bb182.i8560:		; preds = %bb155.i8541, %bb91.i8504
	%tm48.0.i8558 = phi %struct.tm* [ null, %bb91.i8504 ], [ %tmp156.i85398700, %bb155.i8541 ]		; <%struct.tm*> [#uses=0]
	br i1 %foo, label %bb278.i8617, label %bb187.i8591
bb187.i8591:		; preds = %bb182.i8560
	%tmp245.i8588 = srem i64 0, 86400000		; <i64> [#uses=1]
	br i1 %foo, label %bb264.i8592, label %bb265.i8606
bb264.i8592:		; preds = %bb187.i8591
	ret void
bb265.i8606:		; preds = %bb187.i8591
	%tmp268269.i8593 = trunc i64 %tmp245.i8588 to i32		; <i32> [#uses=1]
	%tmp273.i8594 = srem i32 %tmp268269.i8593, 1000		; <i32> [#uses=1]
	%tmp273274.i8595 = trunc i32 %tmp273.i8594 to i16		; <i16> [#uses=1]
	br label %invcont5814
bb278.i8617:		; preds = %bb182.i8560
	%timeOnly50.0.i8622 = add i32 0, 0		; <i32> [#uses=1]
	br i1 %foo, label %bb440.i8663, label %bb448.i8694
bb440.i8663:		; preds = %bb278.i8617
	invoke void @_Z10wxOnAssertPKwiPKcS0_S0_( i32* getelementptr ([27 x i32], [27 x i32]* @.str, i32 0, i32 0), i32 1717, i8* getelementptr ([6 x i8], [6 x i8]* @_ZZNK10wxDateTime5GetTmERKNS_8TimeZoneEE12__FUNCTION__, i32 0, i32 0), i32* getelementptr ([29 x i32], [29 x i32]* @.str33, i32 0, i32 0), i32* getelementptr ([14 x i32], [14 x i32]* @.str4, i32 0, i32 0) )
			to label %bb448.i8694 unwind label %lpad
bb448.i8694:		; preds = %bb440.i8663, %bb278.i8617
	%tmp477.i8669 = srem i32 %timeOnly50.0.i8622, 1000		; <i32> [#uses=1]
	%tmp477478.i8670 = trunc i32 %tmp477.i8669 to i16		; <i16> [#uses=1]
	br label %invcont5814
invcont5814:		; preds = %bb448.i8694, %bb265.i8606
	%tmp812.0.0 = phi i16 [ %tmp477478.i8670, %bb448.i8694 ], [ %tmp273274.i8595, %bb265.i8606 ]		; <i16> [#uses=1]
	%tmp58165817 = zext i16 %tmp812.0.0 to i32		; <i32> [#uses=1]
	invoke void (%struct.wxString*, i32*, ...) @_ZN8wxString6FormatEPKwz(%struct.wxString* noalias sret(%struct.wxString)  null, i32* null, i32 %tmp58165817 )
			to label %invcont5831 unwind label %lpad
invcont5831:		; preds = %invcont5814
	%tmp5862 = invoke zeroext  i8 @_ZN12wxStringBase10ConcatSelfEmPKwm(%struct.wxStringBase* null, i32 0, i32* null, i32 0 )
			to label %bb7834 unwind label %lpad8185		; <i8> [#uses=0]
bb5968:		; preds = %bb3314
	invoke void (%struct.wxString*, i32*, ...) @_ZN8wxString6FormatEPKwz(%struct.wxString* noalias sret(%struct.wxString)  null, i32* null, i32 0 )
			to label %invcont5981 unwind label %lpad
invcont5981:		; preds = %bb5968
	ret void
bb6118:		; preds = %bb3314
	ret void
bb6406:		; preds = %bb3314
	ret void
bb6556:		; preds = %bb3314
	ret void
bb6708:		; preds = %bb3314
	ret void
bb6860:		; preds = %bb3314
	ret void
bb7158:		; preds = %bb3314
	ret void
bb7308:		; preds = %bb3314
	ret void
bb7595:		; preds = %bb3314
	ret void
bb7806:		; preds = %bb3314
	%tmp7814 = invoke %struct.wxStringBase* @_ZN12wxStringBase6appendEmw( %struct.wxStringBase* null, i32 1, i32 0 )
			to label %bb7834 unwind label %lpad		; <%struct.wxStringBase*> [#uses=0]
bb7819:		; preds = %bb3314
	ret void
bb7834:		; preds = %bb7806, %invcont5831
	br label %bb3261
lpad:		; preds = %bb7806, %bb5968, %invcont5814, %bb440.i8663, %bb155.i8541, %bb5657, %bb3306
        %exn = landingpad {i8*, i32}
                 cleanup
	ret void
lpad8185:		; preds = %invcont5831
        %exn8185 = landingpad {i8*, i32}
                 cleanup
	ret void
}

declare void @_Z10wxOnAssertPKwiPKcS0_S0_(i32*, i32, i8*, i32*, i32*)

declare zeroext  i8 @_ZN12wxStringBase10ConcatSelfEmPKwm(%struct.wxStringBase*, i32, i32*, i32)

declare %struct.tm* @gmtime_r(i32*, %struct.tm*)

declare zeroext  i16 @_ZNK10wxDateTime12GetDayOfYearERKNS_8TimeZoneE(%struct.wxDateTime*, %"struct.wxDateTime::TimeZone"*)

declare %struct.wxStringBase* @_ZN12wxStringBase6appendEmw(%struct.wxStringBase*, i32, i32)

declare %struct.wxStringBase* @_ZN12wxStringBaseaSEPKw(%struct.wxStringBase*, i32*)

declare void @_ZN8wxString6FormatEPKwz(%struct.wxString* noalias sret(%struct.wxString) , i32*, ...)

declare i32 @__gxx_personality_v0(...)
