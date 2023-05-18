; RUN: llc < %s -O2 -mtriple=i386-pc-linux-gnu -relocation-model=pic | FileCheck %s
; PR9237: Assertion in VirtRegRewriter.cpp, ResurrectConfirmedKill
;         `KillOps[*SR] == KillOp && "invalid subreg kill flags"'

%t = type { i32 }

define i32 @foo(ptr %s) nounwind {
entry:
  br label %if.then735

if.then735:
  %call747 = call i32 undef(ptr %s, ptr null, ptr undef, i32 128, ptr undef, i32 516) nounwind
  br i1 undef, label %if.then751, label %if.then758

if.then751:
  unreachable

if.then758:
  %add761 = add i32 %call747, 4
  %add763 = add i32 %add761, %call747
  %add.ptr768 = getelementptr inbounds [516 x i8], ptr null, i32 0, i32 %add761
  br i1 undef, label %cond.false783, label %cond.true771

cond.true771:
  %call782 = call ptr @__memmove_chk(ptr %add.ptr768, ptr undef, i32 %call747, i32 undef)
  br label %cond.end791

; CHECK: calll __memmove_chk
cond.false783:
  %call.i1035 = call ptr @__memmove_chk(ptr %add.ptr768, ptr undef, i32 %call747, i32 undef) nounwind
  br label %cond.end791

cond.end791:
  %conv801 = trunc i32 %call747 to i8
  %add.ptr822.sum = add i32 %call747, 3
  %arrayidx833 = getelementptr inbounds [516 x i8], ptr null, i32 0, i32 %add.ptr822.sum
  store i8 %conv801, ptr %arrayidx833, align 1
  %cmp841 = icmp eq ptr undef, null
  br i1 %cmp841, label %if.end849, label %if.then843

if.then843:
  unreachable

if.end849:
  %call921 = call i32 undef(ptr %s, ptr undef, ptr undef, i32 %add763) nounwind
  unreachable

}

declare ptr @__memmove_chk(ptr, ptr, i32, i32) nounwind
