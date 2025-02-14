; RUN: llc -mattr=harden-sls-retbr,harden-sls-blr -verify-machineinstrs -mtriple=aarch64-none-linux-gnu < %s | FileCheck %s --check-prefixes=CHECK,HARDEN,HARDEN-COMDAT,ISBDSB,ISBDSBDAGISEL
; RUN: llc -mattr=harden-sls-retbr,harden-sls-blr,harden-sls-nocomdat -verify-machineinstrs -mtriple=aarch64-none-linux-gnu < %s | FileCheck %s --check-prefixes=CHECK,HARDEN,HARDEN-COMDAT-OFF,ISBDSB,ISBDSBDAGISEL
; RUN: llc -mattr=harden-sls-retbr,harden-sls-blr -mattr=+sb -verify-machineinstrs -mtriple=aarch64-none-linux-gnu < %s | FileCheck %s --check-prefixes=CHECK,HARDEN,SB,SBDAGISEL
; RUN: llc -global-isel -global-isel-abort=0 -mattr=harden-sls-retbr,harden-sls-blr -verify-machineinstrs -mtriple=aarch64-none-linux-gnu < %s | FileCheck %s --check-prefixes=CHECK,HARDEN,HARDEN-COMDAT,ISBDSB
; RUN: llc -verify-machineinstrs -mtriple=aarch64-none-linux-gnu < %s | FileCheck %s --check-prefixes=CHECK,NOHARDEN
; RUN: llc -global-isel -global-isel-abort=0 -mattr=harden-sls-retbr,harden-sls-blr,harden-sls-nocomdat -verify-machineinstrs -mtriple=aarch64-none-linux-gnu < %s | FileCheck %s --check-prefixes=CHECK,HARDEN,HARDEN-COMDAT-OFF,ISBDSB
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

@__const.indirect_branch.ptr = private unnamed_addr constant [2 x ptr] [ptr blockaddress(@indirect_branch, %return), ptr blockaddress(@indirect_branch, %l2)], align 8

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @indirect_branch(i32 %a, i32 %b, i32 %i) {
; CHECK-LABEL: indirect_branch:
entry:
  %idxprom = sext i32 %i to i64
  %arrayidx = getelementptr inbounds [2 x ptr], ptr @__const.indirect_branch.ptr, i64 0, i64 %idxprom
  %0 = load ptr, ptr %arrayidx, align 8
  indirectbr ptr %0, [label %return, label %l2]
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
  callbr void asm sideeffect "B $0", "!i"()
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
ptr nocapture %f1, ptr nocapture %f2) {
entry:
; CHECK-LABEL: indirect_call:
  %call = tail call i32 %f1()
; HARDEN: bl {{__llvm_slsblr_thunk_x[0-9]+$}}
  %call2 = tail call i32 %f2()
; HARDEN: bl {{__llvm_slsblr_thunk_x[0-9]+$}}
  %add = add nsw i32 %call2, %call
  ret i32 %add
; CHECK: .Lfunc_end
}

; verify calling through a function pointer.
@a = dso_local local_unnamed_addr global ptr null, align 8
@b = dso_local local_unnamed_addr global i32 0, align 4
define dso_local void @indirect_call_global() local_unnamed_addr {
; CHECK-LABEL: indirect_call_global:
entry:
  %0 = load ptr, ptr @a, align 8
  %call = tail call i32 %0()  nounwind
; HARDEN: bl {{__llvm_slsblr_thunk_x[0-9]+$}}
  store i32 %call, ptr @b, align 4
  ret void
; CHECK: .Lfunc_end
}

; Verify that neither x16 nor x17 are used when the BLR mitigation is enabled,
; as a linker is allowed to clobber x16 or x17 on calls, which would break the
; correct execution of the code sequence produced by the mitigation. The below
; test attempts to force *%f into x16 using inline assembly.
define i64 @check_x16(ptr nocapture readonly %fp, ptr nocapture readonly %fp2) "target-features"="+neon,+reserve-x10,+reserve-x11,+reserve-x12,+reserve-x13,+reserve-x14,+reserve-x15,+reserve-x18,+reserve-x20,+reserve-x21,+reserve-x22,+reserve-x23,+reserve-x24,+reserve-x25,+reserve-x26,+reserve-x27,+reserve-x28,+reserve-x30,+reserve-x9" {
entry:
; CHECK-LABEL: check_x16:
  %f = load ptr, ptr %fp, align 8
  %x16_f = tail call ptr asm "add $0, $1, #0", "={x16},{x16}"(ptr %f) nounwind
  %call1 = call i64 %x16_f()
; NOHARDEN:   blr x16
; ISBDSB-NOT: bl __llvm_slsblr_thunk_x16
; SB-NOT:     bl __llvm_slsblr_thunk_x16
; CHECK
  ret i64 %call1
; CHECK: .Lfunc_end
}

; Verify that the transformation works correctly for x29 when it is not
; reserved to be used as a frame pointer.
; Since this is sensitive to register allocation choices, only check this with
; DAGIsel to avoid too much accidental breaking of this test that is a bit
; brittle.
define i64 @check_x29(ptr nocapture readonly %fp,
                      ptr nocapture readonly %fp2,
                      ptr nocapture readonly %fp3)
"target-features"="+neon,+reserve-x10,+reserve-x11,+reserve-x12,+reserve-x13,+reserve-x14,+reserve-x15,+reserve-x18,+reserve-x20,+reserve-x21,+reserve-x22,+reserve-x23,+reserve-x24,+reserve-x25,+reserve-x26,+reserve-x27,+reserve-x28,+reserve-x9"
"frame-pointer"="none"
{
entry:
; CHECK-LABEL: check_x29:
  %0 = load ptr, ptr %fp, align 8
  %1 = load ptr, ptr %fp2, align 8
  %2 = load ptr, ptr %fp2, align 8
  %3 = load ptr, ptr %fp3, align 8
  %4 = load ptr, ptr %fp3, align 8
  %5 = load ptr, ptr %fp, align 8
  %call = call i64 %0(ptr %1, ptr %3, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0)
  %call1 = call i64 %2(ptr %1, ptr %3, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0)
; NOHARDEN:      blr x29
; ISBDSBDAGISEL: bl __llvm_slsblr_thunk_x29
; SBDAGISEL:     bl __llvm_slsblr_thunk_x29
; CHECK
  %call2 = call i64 %4(ptr %1, ptr %5, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0)
  %add = add nsw i64 %call1, %call
  %add1 = add nsw i64 %call2, %add
  ret i64 %add1
; CHECK: .Lfunc_end
}

; HARDEN-LABEL: __llvm_slsblr_thunk_x0:
; HARDEN:    mov x16, x0
; HARDEN:    br x16
; ISBDSB-NEXT: dsb sy
; ISBDSB-NEXT: isb
; SB-NEXT:     dsb sy
; SB-NEXT:     isb
; HARDEN-NEXT: .Lfunc_end
; HARDEN-COMDAT:  .section .text.__llvm_slsblr_thunk_x19
; HARDEN-COMDAT:  .hidden __llvm_slsblr_thunk_x19
; HARDEN-COMDAT:  .weak __llvm_slsblr_thunk_x19
; HARDEN-COMDAT: .type  __llvm_slsblr_thunk_x19,@function
; HARDEN-COMDAT-OFF-NOT:  .section .text.__llvm_slsblr_thunk_x19
; HARDEN-COMDAT-OFF-NOT:  .hidden __llvm_slsblr_thunk_x19
; HARDEN-COMDAT-OFF-NOT:  .weak __llvm_slsblr_thunk_x19
; HARDEN-COMDAT-OFF:      .type __llvm_slsblr_thunk_x19,@function
; HARDEN-LABEL: __llvm_slsblr_thunk_x19:
; HARDEN:    mov x16, x19
; HARDEN:    br x16
; ISBDSB-NEXT: dsb sy
; ISBDSB-NEXT: isb
; SB-NEXT:     dsb sy
; SB-NEXT:     isb
; HARDEN-NEXT: .Lfunc_end
