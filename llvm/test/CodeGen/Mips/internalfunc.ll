; RUN: llc < %s -march=mipsel -relocation-model=pic | FileCheck %s

@caller.sf1 = internal unnamed_addr global ptr null, align 4
@gf1 = external global ptr
@.str = private unnamed_addr constant [3 x i8] c"f2\00"

define i32 @main(i32 %argc, ptr nocapture %argv) nounwind {
entry:
; CHECK: lw $[[R0:[0-9]+]], %got(f2)
; CHECK: addiu $25, $[[R0]], %lo(f2)
  tail call fastcc void @f2()
  ret i32 0
}

define void @caller(i32 %a0, i32 %a1) nounwind {
entry:
; CHECK: lw  $[[R1:[0-9]+]], %got(caller.sf1)
; CHECK: lw  $25, %lo(caller.sf1)($[[R1]])
  %tobool = icmp eq i32 %a1, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %tmp1 = load ptr, ptr @caller.sf1, align 4
  tail call void (...) %tmp1() nounwind
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
; CHECK: lw  $[[R2:[0-9]+]], %got(sf2)
; CHECK: addiu ${{[0-9]+}}, $[[R2]], %lo(sf2)
; CHECK: lw  $[[R3:[0-9]+]], %got(caller.sf1)
; CHECK: sw  ${{[0-9]+}}, %lo(caller.sf1)($[[R3]])
  %tobool3 = icmp ne i32 %a0, 0
  %tmp4 = load ptr, ptr @gf1, align 4
  %cond = select i1 %tobool3, ptr %tmp4, ptr @sf2
  store ptr %cond, ptr @caller.sf1, align 4
  ret void
}

define internal void @sf2() nounwind {
entry:
  %call = tail call i32 (ptr, ...) @printf(ptr @.str) nounwind
  ret void
}

declare i32 @printf(ptr nocapture, ...) nounwind

define internal fastcc void @f2() nounwind noinline {
entry:
  %call = tail call i32 (ptr, ...) @printf(ptr @.str) nounwind
  ret void
}

