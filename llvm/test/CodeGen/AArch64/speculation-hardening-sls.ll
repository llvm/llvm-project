; RUN: llc -mattr=harden-sls-retbr,harden-sls-blr -verify-machineinstrs -mtriple=aarch64-none-linux-gnu < %s | FileCheck %s --check-prefixes=CHECK,HARDEN,ISBDSB,ISBDSBDAGISEL
; RUN: llc -mattr=harden-sls-retbr,harden-sls-blr -mattr=+sb -verify-machineinstrs -mtriple=aarch64-none-linux-gnu < %s | FileCheck %s --check-prefixes=CHECK,HARDEN,SB,SBDAGISEL
; RUN: llc -verify-machineinstrs -mtriple=aarch64-none-linux-gnu < %s | FileCheck %s --check-prefixes=CHECK,NOHARDEN
; RUN: llc -global-isel -global-isel-abort=0 -mattr=harden-sls-retbr,harden-sls-blr -verify-machineinstrs -mtriple=aarch64-none-linux-gnu < %s | FileCheck %s --check-prefixes=CHECK,HARDEN,ISBDSB
; RUN: llc -global-isel -global-isel-abort=0 -mattr=harden-sls-retbr,harden-sls-blr -mattr=+sb -verify-machineinstrs -mtriple=aarch64-none-linux-gnu < %s | FileCheck %s --check-prefixes=CHECK,HARDEN,SB

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @double_return(i32 %a, i32 %b) local_unnamed_addr {
entry:
  %cmp = icmp sgt i32 %a, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %div = sdiv i32 %a, %b
  ret i32 %div

if.else:                                          ; preds = %entry
  %div1 = sdiv i32 %b, %a
  ret i32 %div1
; CHECK-LABEL: double_return:
; CHECK:       {{ret$}}
; ISBDSB-NEXT: dsb sy
; ISBDSB-NEXT: isb
; SB-NEXT:     {{ sb$}}
; CHECK:       {{ret$}}
; ISBDSB-NEXT: dsb sy
; ISBDSB-NEXT: isb
; SB-NEXT:     {{ sb$}}
; CHECK-NEXT: .Lfunc_end
}

@__const.indirect_branch.ptr = private unnamed_addr constant [2 x i8*] [i8* blockaddress(@indirect_branch, %return), i8* blockaddress(@indirect_branch, %l2)], align 8

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @indirect_branch(i32 %a, i32 %b, i32 %i) {
; CHECK-LABEL: indirect_branch:
entry:
  %idxprom = sext i32 %i to i64
  %arrayidx = getelementptr inbounds [2 x i8*], [2 x i8*]* @__const.indirect_branch.ptr, i64 0, i64 %idxprom
  %0 = load i8*, i8** %arrayidx, align 8
  indirectbr i8* %0, [label %return, label %l2]
; CHECK:       br x
; ISBDSB-NEXT: dsb sy
; ISBDSB-NEXT: isb
; SB-NEXT:     {{ sb$}}

l2:                                               ; preds = %entry
  br label %return
; CHECK:       {{ret$}}
; ISBDSB-NEXT: dsb sy
; ISBDSB-NEXT: isb
; SB-NEXT:     {{ sb$}}

return:                                           ; preds = %entry, %l2
  %retval.0 = phi i32 [ 1, %l2 ], [ 0, %entry ]
  ret i32 %retval.0
; CHECK:       {{ret$}}
; ISBDSB-NEXT: dsb sy
; ISBDSB-NEXT: isb
; SB-NEXT:     {{ sb$}}
; CHECK-NEXT: .Lfunc_end
}

; Check that RETAA and RETAB instructions are also protected as expected.
define dso_local i32 @ret_aa(i32 returned %a) local_unnamed_addr "target-features"="+neon,+v8.3a" "sign-return-address"="all" "sign-return-address-key"="a_key" {
entry:
; CHECK-LABEL: ret_aa:
; CHECK:       {{ retaa$}}
; ISBDSB-NEXT: dsb sy
; ISBDSB-NEXT: isb
; SB-NEXT:     {{ sb$}}
; CHECK-NEXT: .Lfunc_end
	  ret i32 %a
}

define dso_local i32 @ret_ab(i32 returned %a) local_unnamed_addr "target-features"="+neon,+v8.3a" "sign-return-address"="all" "sign-return-address-key"="b_key" {
entry:
; CHECK-LABEL: ret_ab:
; CHECK:       {{ retab$}}
; ISBDSB-NEXT: dsb sy
; ISBDSB-NEXT: isb
; SB-NEXT:     {{ sb$}}
; CHECK-NEXT: .Lfunc_end
	  ret i32 %a
}

define i32 @asmgoto() {
entry:
; CHECK-LABEL: asmgoto:
  callbr void asm sideeffect "B $0", "X"(i8* blockaddress(@asmgoto, %d))
            to label %asm.fallthrough [label %d]
     ; The asm goto above produces a direct branch:
; CHECK:           //APP
; CHECK-NEXT:      {{^[ \t]+b }}
; CHECK-NEXT:      //NO_APP
     ; For direct branches, no mitigation is needed.
; ISDDSB-NOT: dsb sy
; SB-NOT:     {{ sb$}}

asm.fallthrough:               ; preds = %entry
  ret i32 0
; CHECK:       {{ret$}}
; ISBDSB-NEXT: dsb sy
; ISBDSB-NEXT: isb
; SB-NEXT:     {{ sb$}}

d:                             ; preds = %asm.fallthrough, %entry
  ret i32 1
; CHECK:       {{ret$}}
; ISBDSB-NEXT: dsb sy
; ISBDSB-NEXT: isb
; SB-NEXT:     {{ sb$}}
; CHECK-NEXT: .Lfunc_end
}

define dso_local i32 @indirect_call(
i32 (...)* nocapture %f1, i32 (...)* nocapture %f2) {
entry:
; CHECK-LABEL: indirect_call:
  %callee.knr.cast = bitcast i32 (...)* %f1 to i32 ()*
  %call = tail call i32 %callee.knr.cast()
; HARDEN: bl {{__llvm_slsblr_thunk_x[0-9]+$}}
  %callee.knr.cast1 = bitcast i32 (...)* %f2 to i32 ()*
  %call2 = tail call i32 %callee.knr.cast1()
; HARDEN: bl {{__llvm_slsblr_thunk_x[0-9]+$}}
  %add = add nsw i32 %call2, %call
  ret i32 %add
; CHECK: .Lfunc_end
}

; verify calling through a function pointer.
@a = dso_local local_unnamed_addr global i32 (...)* null, align 8
@b = dso_local local_unnamed_addr global i32 0, align 4
define dso_local void @indirect_call_global() local_unnamed_addr {
; CHECK-LABEL: indirect_call_global:
entry:
  %0 = load i32 ()*, i32 ()** bitcast (i32 (...)** @a to i32 ()**), align 8
  %call = tail call i32 %0()  nounwind
; HARDEN: bl {{__llvm_slsblr_thunk_x[0-9]+$}}
  store i32 %call, i32* @b, align 4
  ret void
; CHECK: .Lfunc_end
}

; Verify that neither x16 nor x17 are used when the BLR mitigation is enabled,
; as a linker is allowed to clobber x16 or x17 on calls, which would break the
; correct execution of the code sequence produced by the mitigation.
; The below test carefully increases register pressure to persuade code
; generation to produce a BLR x16. Yes, that is a bit fragile.
define i64 @check_x16(i64 (i8*, i64, i64, i64, i64, i64, i64, i64)** nocapture readonly %fp, i64 (i8*, i64, i64, i64, i64, i64, i64, i64)** nocapture readonly %fp2) "target-features"="+neon,+reserve-x10,+reserve-x11,+reserve-x12,+reserve-x13,+reserve-x14,+reserve-x15,+reserve-x18,+reserve-x20,+reserve-x21,+reserve-x22,+reserve-x23,+reserve-x24,+reserve-x25,+reserve-x26,+reserve-x27,+reserve-x28,+reserve-x30,+reserve-x9" {
entry:
; CHECK-LABEL: check_x16:
  %0 = load i64 (i8*, i64, i64, i64, i64, i64, i64, i64)*, i64 (i8*, i64, i64, i64, i64, i64, i64, i64)** %fp, align 8
  %1 = bitcast i64 (i8*, i64, i64, i64, i64, i64, i64, i64)** %fp2 to i8**
  %2 = load i8*, i8** %1, align 8
  %call = call i64 %0(i8* %2, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0)
  %3 = load i64 (i8*, i64, i64, i64, i64, i64, i64, i64)*, i64 (i8*, i64, i64, i64, i64, i64, i64, i64)** %fp2, align 8
  %4 = bitcast i64 (i8*, i64, i64, i64, i64, i64, i64, i64)** %fp to i8**
  %5 = load i8*, i8** %4, align 8;, !tbaa !2
  %call1 = call i64 %3(i8* %5, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0)
; NOHARDEN:   blr x16
; ISBDSB-NOT: bl __llvm_slsblr_thunk_x16
; SB-NOT:     bl __llvm_slsblr_thunk_x16
; CHECK
  %add = add nsw i64 %call1, %call
  ret i64 %add
; CHECK: .Lfunc_end
}

; Verify that the transformation works correctly for x29 when it is not
; reserved to be used as a frame pointer.
; Since this is sensitive to register allocation choices, only check this with
; DAGIsel to avoid too much accidental breaking of this test that is a bit
; brittle.
define i64 @check_x29(i64 (i8*, i8*, i64, i64, i64, i64, i64, i64)** nocapture readonly %fp,
                      i64 (i8*, i8*, i64, i64, i64, i64, i64, i64)** nocapture readonly %fp2,
                      i64 (i8*, i8*, i64, i64, i64, i64, i64, i64)** nocapture readonly %fp3)
"target-features"="+neon,+reserve-x10,+reserve-x11,+reserve-x12,+reserve-x13,+reserve-x14,+reserve-x15,+reserve-x18,+reserve-x20,+reserve-x21,+reserve-x22,+reserve-x23,+reserve-x24,+reserve-x25,+reserve-x26,+reserve-x27,+reserve-x28,+reserve-x9"
"frame-pointer"="none"
{
entry:
; CHECK-LABEL: check_x29:
  %0 = load i64 (i8*, i8*, i64, i64, i64, i64, i64, i64)*, i64 (i8*, i8*, i64, i64, i64, i64, i64, i64)** %fp, align 8
  %1 = bitcast i64 (i8*, i8*, i64, i64, i64, i64, i64, i64)** %fp2 to i8**
  %2 = load i8*, i8** %1, align 8
  %3 = load i64 (i8*, i8*, i64, i64, i64, i64, i64, i64)*, i64 (i8*, i8*, i64, i64, i64, i64, i64, i64)** %fp2, align 8
  %4 = bitcast i64 (i8*, i8*, i64, i64, i64, i64, i64, i64)** %fp3 to i8**
  %5 = load i8*, i8** %4, align 8
  %6 = load i64 (i8*, i8*, i64, i64, i64, i64, i64, i64)*, i64 (i8*, i8*, i64, i64, i64, i64, i64, i64)** %fp3, align 8
  %7 = bitcast i64 (i8*, i8*, i64, i64, i64, i64, i64, i64)** %fp to i8**
  %8 = load i8*, i8** %7, align 8
  %call = call i64 %0(i8* %2, i8* %5, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0)
  %call1 = call i64 %3(i8* %2, i8* %5, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0)
; NOHARDEN:      blr x29
; ISBDSBDAGISEL: bl __llvm_slsblr_thunk_x29
; SBDAGISEL:     bl __llvm_slsblr_thunk_x29
; CHECK
  %call2 = call i64 %6(i8* %2, i8* %8, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0)
  %add = add nsw i64 %call1, %call
  %add1 = add nsw i64 %call2, %add
  ret i64 %add1
; CHECK: .Lfunc_end
}

; HARDEN-label: __llvm_slsblr_thunk_x0:
; HARDEN:    br x0
; ISBDSB-NEXT: dsb sy
; ISBDSB-NEXT: isb
; SB-NEXT:     dsb sy
; SB-NEXT:     isb
; HARDEN-NEXT: .Lfunc_end
; HARDEN-label: __llvm_slsblr_thunk_x19:
; HARDEN:    br x19
; ISBDSB-NEXT: dsb sy
; ISBDSB-NEXT: isb
; SB-NEXT:     dsb sy
; SB-NEXT:     isb
; HARDEN-NEXT: .Lfunc_end
