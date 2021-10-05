; RUN: llc -mattr=harden-sls-retbr -mattr=harden-sls-blr -verify-machineinstrs -mtriple=armv8-linux-gnueabi < %s | FileCheck %s --check-prefixes=CHECK,ARM,HARDEN,HARDEN-COMDAT,ISBDSB -dump-input-context=100
; RUN: llc -mattr=harden-sls-retbr -mattr=harden-sls-blr -verify-machineinstrs -mtriple=thumbv8-linux-gnueabi < %s | FileCheck %s --check-prefixes=CHECK,THUMB,HARDENTHUMB,HARDEN,HARDEN-COMDAT,ISBDSB -dump-input-context=100
; RUN: llc -mattr=harden-sls-retbr -mattr=harden-sls-blr -mattr=+sb -verify-machineinstrs -mtriple=armv8-linux-gnueabi < %s | FileCheck %s --check-prefixes=CHECK,ARM,HARDEN,HARDEN-COMDAT,SB -dump-input-context=100
; RUN: llc -mattr=harden-sls-retbr -mattr=harden-sls-blr -mattr=+sb -verify-machineinstrs -mtriple=thumbv8-linux-gnueabi < %s | FileCheck %s --check-prefixes=CHECK,THUMB,HARDENTHUMB,HARDEN,HARDEN-COMDAT,SB -dump-input-context=100
; RUN: llc -mattr=harden-sls-retbr -mattr=harden-sls-blr -mattr=harden-sls-nocomdat -verify-machineinstrs -mtriple=armv8-linux-gnueabi < %s | FileCheck %s --check-prefixes=CHECK,ARM,HARDEN,HARDEN-COMDAT-OFF,ISBDSB -dump-input-context=100
; RUN: llc -mattr=harden-sls-retbr -mattr=harden-sls-blr -mattr=harden-sls-nocomdat -verify-machineinstrs -mtriple=thumbv8-linux-gnueabi < %s | FileCheck %s --check-prefixes=CHECK,THUMB,HARDENTHUMB,HARDEN,HARDEN-COMDAT-OFF,ISBDSB -dump-input-context=100
; RUN: llc -mattr=harden-sls-retbr -mattr=harden-sls-blr -mattr=harden-sls-nocomdat -mattr=+sb -verify-machineinstrs -mtriple=armv8-linux-gnueabi < %s | FileCheck %s --check-prefixes=CHECK,ARM,HARDEN,HARDEN-COMDAT-OFF,SB -dump-input-context=100
; RUN: llc -mattr=harden-sls-retbr -mattr=harden-sls-blr -mattr=harden-sls-nocomdat -mattr=+sb -verify-machineinstrs -mtriple=thumbv8-linux-gnueabi < %s | FileCheck %s --check-prefixes=CHECK,THUMB,HARDENTHUMB,HARDEN,HARDEN-COMDAT-OFF,SB -dump-input-context=100
; RUN: llc -verify-machineinstrs -mtriple=armv8-linux-gnueabi < %s | FileCheck %s --check-prefixes=CHECK,ARM,NOHARDENARM -dump-input-context=100
; RUN: llc -verify-machineinstrs -mtriple=thumbv8-linux-gnueabi < %s | FileCheck %s --check-prefixes=CHECK,THUMB,NOHARDENTHUMB
; RUN: llc -global-isel -global-isel-abort=0 -mattr=harden-sls-retbr -mattr=harden-sls-blr -verify-machineinstrs -mtriple=armv8-linux-gnueabi < %s | FileCheck %s --check-prefixes=CHECK,ARM,HARDEN,HARDEN-COMDAT,ISBDSB
; RUN: llc -global-isel -global-isel-abort=0 -mattr=harden-sls-retbr -mattr=harden-sls-blr -verify-machineinstrs -mtriple=thumbv8-linux-gnueabi < %s | FileCheck %s --check-prefixes=CHECK,THUMB,HARDENTHUMB,HARDEN,HARDEN-COMDAT,ISBDSB
; RUN: llc -global-isel -global-isel-abort=0 -mattr=harden-sls-retbr -mattr=harden-sls-nocomdat -mattr=harden-sls-blr -verify-machineinstrs -mtriple=armv8-linux-gnueabi < %s | FileCheck %s --check-prefixes=CHECK,ARM,HARDEN,HARDEN-COMDAT-OFF,ISBDSB
; RUN: llc -global-isel -global-isel-abort=0 -mattr=harden-sls-retbr -mattr=harden-sls-nocomdat -mattr=harden-sls-blr -verify-machineinstrs -mtriple=thumbv8-linux-gnueabi < %s | FileCheck %s --check-prefixes=CHECK,THUMB,HARDENTHUMB,HARDEN,HARDEN-COMDAT-OFF,ISBDSB
; RUN: llc -global-isel -global-isel-abort=0 -mattr=harden-sls-retbr -mattr=harden-sls-blr -mattr=+sb -verify-machineinstrs -mtriple=armv8-linux-gnueabi < %s | FileCheck %s --check-prefixes=CHECK,ARM,HARDEN,SB
; RUN: llc -global-isel -global-isel-abort=0 -mattr=harden-sls-retbr -mattr=harden-sls-blr -mattr=+sb -verify-machineinstrs -mtriple=thumbv8-linux-gnueabi < %s | FileCheck %s --check-prefixes=CHECK,THUMB,HARDENTHUMB,HARDEN,SB
; RUN: llc -fast-isel -mattr=harden-sls-retbr -mattr=harden-sls-blr -verify-machineinstrs -mtriple=armv8-linux-gnueabi < %s | FileCheck %s --check-prefixes=CHECK,ARM,HARDEN,HARDEN-COMDAT,ISBDSB
; RUN: llc -fast-isel -mattr=harden-sls-retbr -mattr=harden-sls-blr -verify-machineinstrs -mtriple=thumbv8-linux-gnueabi < %s | FileCheck %s --check-prefixes=CHECK,THUMB,HARDENTHUMB,HARDEN,HARDEN-COMDAT,ISBDSB
; RUN: llc -fast-isel -mattr=harden-sls-retbr -mattr=harden-sls-blr -mattr=harden-sls-nocomdat -verify-machineinstrs -mtriple=armv8-linux-gnueabi < %s | FileCheck %s --check-prefixes=CHECK,ARM,HARDEN,HARDEN-COMDAT-OFF,ISBDSB
; RUN: llc -fast-isel -mattr=harden-sls-retbr -mattr=harden-sls-blr -mattr=harden-sls-nocomdat -verify-machineinstrs -mtriple=thumbv8-linux-gnueabi < %s | FileCheck %s --check-prefixes=CHECK,THUMB,HARDENTHUMB,HARDEN,HARDEN-COMDAT-OFF,ISBDSB
; RUN: llc -fast-isel -mattr=harden-sls-retbr -mattr=harden-sls-blr -mattr=+sb -verify-machineinstrs -mtriple=armv8-linux-gnueabi < %s | FileCheck %s --check-prefixes=CHECK,ARM,HARDEN,SB
; RUN: llc -fast-isel -mattr=harden-sls-retbr -mattr=harden-sls-blr -mattr=+sb -verify-machineinstrs -mtriple=thumbv8-linux-gnueabi < %s | FileCheck %s --check-prefixes=CHECK,THUMB,HARDENTHUMB,HARDEN,SB

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @double_return(i32 %a, i32 %b) local_unnamed_addr {
entry:
  %cmp = icmp sgt i32 %a, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  ; Make a very easy, very likely to predicate return (BX LR), to test that
  ; it will not get predicated when sls-hardening is enabled.
  %mul = mul i32 %b, %a
  ret i32 %mul
; CHECK-LABEL: double_return:
; HARDEN:          {{bx lr$}}
; NOHARDENARM:     {{bxgt lr$}}
; NOHARDENTHUMB:   {{bx lr$}}
; ISBDSB-NEXT: dsb sy
; ISBDSB-NEXT: isb
; SB-NEXT:     {{ sb$}}

if.else:                                          ; preds = %entry
  %div3 = sdiv i32 %a, %b
  %div2 = sdiv i32 %a, %div3
  %div1 = sdiv i32 %a, %div2
  ret i32 %div1

; CHECK:       {{bx lr$}}
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
; ARM:       bx r0
; THUMB:     mov pc, r0
; ISBDSB-NEXT: dsb sy
; ISBDSB-NEXT: isb
; SB-NEXT:     {{ sb$}}

l2:                                               ; preds = %entry
  br label %return
; CHECK:       {{bx lr$}}
; ISBDSB-NEXT: dsb sy
; ISBDSB-NEXT: isb
; SB-NEXT:     {{ sb$}}

return:                                           ; preds = %entry, %l2
  %retval.0 = phi i32 [ 1, %l2 ], [ 0, %entry ]
  ret i32 %retval.0
; CHECK:       {{bx lr$}}
; ISBDSB-NEXT: dsb sy
; ISBDSB-NEXT: isb
; SB-NEXT:     {{ sb$}}
; CHECK-NEXT: .Lfunc_end
}

define i32 @asmgoto() {
entry:
; CHECK-LABEL: asmgoto:
  callbr void asm sideeffect "B $0", "X"(i8* blockaddress(@asmgoto, %d))
            to label %asm.fallthrough [label %d]
     ; The asm goto above produces a direct branch:
; CHECK:           @APP
; CHECK-NEXT:      {{^[ \t]+b }}
; CHECK-NEXT:      @NO_APP
     ; For direct branches, no mitigation is needed.
; ISDDSB-NOT: dsb sy
; SB-NOT:     {{ sb$}}

asm.fallthrough:               ; preds = %entry
  ret i32 0
; CHECK:       {{bx lr$}}
; ISBDSB-NEXT: dsb sy
; ISBDSB-NEXT: isb
; SB-NEXT:     {{ sb$}}

d:                             ; preds = %asm.fallthrough, %entry
  ret i32 1
; CHECK:       {{bx lr$}}
; ISBDSB-NEXT: dsb sy
; ISBDSB-NEXT: isb
; SB-NEXT:     {{ sb$}}
; CHECK-NEXT: .Lfunc_end
}

; Check that indirect branches produced through switch jump tables are also
; hardened:
define dso_local i32 @jumptable(i32 %a, i32 %b) {
; CHECK-LABEL: jumptable:
entry:
  switch i32 %b, label %sw.epilog [
    i32 0, label %sw.bb
    i32 1, label %sw.bb1
    i32 3, label %sw.bb3
    i32 4, label %sw.bb5
  ]
; ARM:             ldr pc, [{{r[0-9]}}, {{r[0-9]}}, lsl #2]
; NOHARDENTHUMB:   tbb [pc, {{r[0-9]}}]
; HARDENTHUMB:     mov pc, {{r[0-9]}}
; ISBDSB-NEXT:     dsb sy
; ISBDSB-NEXT:     isb
; SB-NEXT:         {{ sb$}}


sw.bb:                                            ; preds = %entry
  %add = shl nsw i32 %a, 1
  br label %sw.bb1

sw.bb1:                                           ; preds = %entry, %sw.bb
  %a.addr.0 = phi i32 [ %a, %entry ], [ %add, %sw.bb ]
  %add2 = shl nsw i32 %a.addr.0, 1
  br label %sw.bb3

sw.bb3:                                           ; preds = %entry, %sw.bb1
  %a.addr.1 = phi i32 [ %a, %entry ], [ %add2, %sw.bb1 ]
  %add4 = shl nsw i32 %a.addr.1, 1
  br label %sw.bb5

sw.bb5:                                           ; preds = %entry, %sw.bb3
  %a.addr.2 = phi i32 [ %a, %entry ], [ %add4, %sw.bb3 ]
  %add6 = shl nsw i32 %a.addr.2, 1
  br label %sw.epilog

sw.epilog:                                        ; preds = %sw.bb5, %entry
  %a.addr.3 = phi i32 [ %a, %entry ], [ %add6, %sw.bb5 ]
  ret i32 %a.addr.3
; CHECK:       {{bx lr$}}
; ISBDSB-NEXT: dsb sy
; ISBDSB-NEXT: isb
; SB-NEXT:     {{ sb$}}
}

define dso_local i32 @indirect_call(
i32 (...)* nocapture %f1, i32 (...)* nocapture %f2) {
entry:
; CHECK-LABEL: indirect_call:
  %callee.knr.cast = bitcast i32 (...)* %f1 to i32 ()*
  %call = tail call i32 %callee.knr.cast()
; HARDENARM: bl {{__llvm_slsblr_thunk_arm_r[0-9]+$}}
; HARDENTHUMB: bl {{__llvm_slsblr_thunk_thumb_r[0-9]+$}}
  %callee.knr.cast1 = bitcast i32 (...)* %f2 to i32 ()*
  %call2 = tail call i32 %callee.knr.cast1()
; HARDENARM: bl {{__llvm_slsblr_thunk_arm_r[0-9]+$}}
; HARDENTHUMB: bl {{__llvm_slsblr_thunk_thumb_r[0-9]+$}}
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
; HARDENARM: bl {{__llvm_slsblr_thunk_arm_r[0-9]+$}}
; HARDENTHUMB: bl {{__llvm_slsblr_thunk_thumb_r[0-9]+$}}
  store i32 %call, i32* @b, align 4
  ret void
; CHECK: .Lfunc_end
}

; Verify that neither r12 nor lr are used as registers in indirect call
; instructions when the sls-hardening-blr mitigation is enabled, as
; (a) a linker is allowed to clobber r12 on calls, and
; (b) the hardening transformation isn't correct if lr is the register holding
;     the address of the function called.
define i32 @check_r12(i32 ()** %fp) {
entry:
; CHECK-LABEL: check_r12:
  %f = load i32 ()*, i32 ()** %fp, align 4
  ; Force f to be moved into r12
  %r12_f = tail call i32 ()* asm "add $0, $1, #0", "={r12},{r12}"(i32 ()* %f) nounwind
  %call = call i32 %r12_f()
; NOHARDENARM:     blx r12
; NOHARDENTHUMB:   blx r12
; HARDEN-NOT: bl {{__llvm_slsblr_thunk_(arm|thumb)_r12}}
  ret i32 %call
; CHECK: .Lfunc_end
}

define i32 @check_lr(i32 ()** %fp) {
entry:
; CHECK-LABEL: check_lr:
  %f = load i32 ()*, i32 ()** %fp, align 4
  ; Force f to be moved into lr
  %lr_f = tail call i32 ()* asm "add $0, $1, #0", "={lr},{lr}"(i32 ()* %f) nounwind
  %call = call i32 %lr_f()
; NOHARDENARM:     blx lr
; NOHARDENTHUMB:   blx lr
; HARDEN-NOT: bl {{__llvm_slsblr_thunk_(arm|thumb)_lr}}
  ret i32 %call
; CHECK: .Lfunc_end
}

; Verify that even when sls-harden-blr is enabled, "blx r12" is still an
; instruction that is accepted by the inline assembler
define void @verify_inline_asm_blx_r12(void ()* %g) {
entry:
; CHECK-LABEL: verify_inline_asm_blx_r12:
  %0 = bitcast void ()* %g to i8*
  tail call void asm sideeffect "blx $0", "{r12}"(i8* %0) nounwind
; CHECK: blx r12
  ret void
; CHECK:       {{bx lr$}}
; ISBDSB-NEXT: dsb sy
; ISBDSB-NEXT: isb
; SB-NEXT:     {{ sb$}}
; CHECK: .Lfunc_end
}

; HARDEN-COMDAT:  .section {{.text.__llvm_slsblr_thunk_(arm|thumb)_r5}}
; HARDEN-COMDAT:  .hidden {{__llvm_slsblr_thunk_(arm|thumb)_r5}}
; HARDEN-COMDAT:  .weak {{__llvm_slsblr_thunk_(arm|thumb)_r5}}
; HARDEN-COMDAT:  .type {{__llvm_slsblr_thunk_(arm|thumb)_r5}},%function
; HARDEN-COMDAT-OFF-NOT:  .section {{.text.__llvm_slsblr_thunk_(arm|thumb)_r5}}
; HARDEN-COMDAT-OFF-NOT:  .hidden {{__llvm_slsblr_thunk_(arm|thumb)_r5}}
; HARDEN-COMDAT-OFF-NOT:  .weak {{__llvm_slsblr_thunk_(arm|thumb)_r5}}
; HARDEN-COMDAT-OFF:      .type {{__llvm_slsblr_thunk_(arm|thumb)_r5}},%function
; HARDEN-label: {{__llvm_slsblr_thunk_(arm|thumb)_r5}}:
; HARDEN:    bx r5
; ISBDSB-NEXT: dsb sy
; ISBDSB-NEXT: isb
; SB-NEXT:     dsb sy
; SB-NEXT:     isb
; HARDEN-NEXT: .Lfunc_end


