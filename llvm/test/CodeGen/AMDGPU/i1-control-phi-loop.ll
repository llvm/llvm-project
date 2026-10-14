; RUN: llc -mtriple=amdgpu12.01-amd-amdhsa < %s | FileCheck %s
; CHECK-LABEL: CSMain:
; CHECK: s_or_b32 exec_lo, exec_lo, {{s[0-9]+}}
; CHECK-NOT: s_and_not1_b32 {{s[0-9]+}}, {{s[0-9]+}}, exec_lo
; CHECK-NOT: s_and_b32 {{s[0-9]+}}, {{s[0-9]+}}, exec_lo
; CHECK-NOT: s_or_b32 {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
; CHECK: s_or_b32 exec_lo, exec_lo, {{s[0-9]+}}

define amdgpu_kernel void @CSMain(ptr addrspace(1) %buf) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %nraw = load i32, ptr addrspace(1) %buf, align 4
  %n = and i32 %nraw, 15
  %base.ptr = getelementptr i32, ptr addrspace(1) %buf, i32 %tid
  %base = load i32, ptr addrspace(1) %base.ptr, align 4
  %go = icmp ne i32 %n, 0
  br i1 %go, label %loop, label %exit

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %latch ]
  %acc = phi i32 [ 0, %entry ], [ %acc.next, %latch ]
  %live = phi i1 [ true, %entry ], [ %live.next, %latch ]
  %side = phi i1 [ false, %entry ], [ %side.next, %latch ]
  br i1 %live, label %if.live, label %latch

if.live:
  %idx = add i32 %base, %i
  %element.ptr = getelementptr i32, ptr addrspace(1) %buf, i32 %idx
  %x = load i32, ptr addrspace(1) %element.ptr, align 4
  %odd = trunc i32 %x to i1
  br i1 %odd, label %then.hit, label %else.kill

then.hit:
  %acc.hit = add i32 %acc, %x
  br label %latch

else.kill:
  br label %latch

latch:
  %acc.next = phi i32 [ %acc, %loop ], [ %acc.hit, %then.hit ], [ %acc, %else.kill ]
  %live.next = phi i1 [ %live, %loop ], [ true, %then.hit ], [ false, %else.kill ]
  %side.next = phi i1 [ %side, %loop ], [ true, %then.hit ], [ %side, %else.kill ]
  %i.next = add i32 %i, 1
  %cont = icmp ult i32 %i.next, %n
  br i1 %cont, label %loop, label %exit

exit:
  %acc.end = phi i32 [ 0, %entry ], [ %acc.next, %latch ]
  %side.end = phi i1 [ false, %entry ], [ %side.next, %latch ]
  %side.z = zext i1 %side.end to i32
  %outv = add i32 %acc.end, %side.z
  store i32 %outv, ptr addrspace(1) %base.ptr, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
