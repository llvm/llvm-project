; RUN: sed -e 's/SLHATTR/speculative_load_hardening/' %s | llc -verify-machineinstrs -mtriple=aarch64-none-linux-gnu | FileCheck %s --check-prefixes=CHECK,SLH --dump-input-on-failure
; RUN: sed -e 's/SLHATTR//' %s | llc -verify-machineinstrs -mtriple=aarch64-none-linux-gnu | FileCheck %s --check-prefixes=CHECK,NOSLH --dump-input-on-failure

declare i64 @g(i64, i64) local_unnamed_addr
define i64 @f_using_reserved_reg_x16(i64 %a, i64 %b) local_unnamed_addr SLHATTR {
; CHECK-LABEL: f_using_reserved_reg_x16
; SLH: dsb sy
; SLH: isb
; NOSLH-NOT: dsb sy
; NOSLH-NOT: isb
entry:
  %cmp = icmp ugt i64 %a, %b
  br i1 %cmp, label %if.then, label %cleanup

; CHECK: b.ls
; SLH: dsb sy
; SLH: isb
; NOSLH-NOT: dsb sy
; NOSLH-NOT: isb
if.then:
  %0 = tail call i64 asm "autia1716", "={x17},{x16},0"(i64 %b, i64 %a)
; CHECK: bl g
; SLH: dsb sy
; SLH: isb
; NOSLH-NOT: dsb sy
; NOSLH-NOT: isb
; CHECK: ret
  %call = tail call i64 @g(i64 %a, i64 %b) #3
  %add = add i64 %call, %0
  br label %cleanup

cleanup:
; SLH: dsb sy
; SLH: isb
; NOSLH-NOT: dsb sy
; NOSLH-NOT: isb
; SLH: ret
  %retval.0 = phi i64 [ %add, %if.then ], [ %b, %entry ]
  ret i64 %retval.0
}

define i32 @f_clobbered_reg_w16(i32 %a, i32 %b) local_unnamed_addr SLHATTR {
; CHECK-LABEL: f_clobbered_reg_w16
entry:
; SLH: dsb sy
; SLH: isb
; NOSLH-NOT: dsb sy
; NOSLH-NOT: isb
  %cmp = icmp sgt i32 %a, %b
  br i1 %cmp, label %if.then, label %if.end
; CHECK: b.le

if.then:
; SLH: dsb sy
; SLH: isb
; NOSLH-NOT: dsb sy
; NOSLH-NOT: isb
; CHECK: mov w16, w0
  tail call void asm sideeffect "mov w16, ${0:w}", "r,~{w16}"(i32 %a)
  br label %if.end
; SLH: ret

if.end:
  %add = add nsw i32 %b, %a
  ret i32 %add
; SLH: dsb sy
; SLH: isb
; NOSLH-NOT: dsb sy
; NOSLH-NOT: isb
; SLH: ret
}
