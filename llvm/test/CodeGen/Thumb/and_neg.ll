; RUN: llc < %s -mtriple=thumbv7-linux-gnueabi -verify-machineinstrs
; Just shouldn't crash, PR28348

%C = type { ptr }

define void @repro(ptr %this, i32 %a) {
  %a_align1 = and i32 %a, -4096
  %a_and = and i32 %a, 4095
  %a_align2 = or i32 %a_and, 4096

  call void @use(i32 %a_align1)

  %addptr = getelementptr inbounds i8, ptr null, i32 %a_align2
  store ptr %addptr, ptr %this, align 4

  ret void
}

declare void @use(i32)
