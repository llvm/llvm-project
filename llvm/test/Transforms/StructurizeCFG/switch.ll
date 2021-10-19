; RUN: opt -S -structurizecfg %s -o - -enable-new-pm=0 | FileCheck %s

; The structurizecfg pass cannot handle switch instructions, so we need to
; make sure the lower switch pass is always run before structurizecfg.

; CHECK-LABEL: @switch
define void @switch(i32 addrspace(1)* %out, i32 %cond) nounwind {
entry:
; CHECK: icmp
  switch i32 %cond, label %done [ i32 0, label %zero]

; CHECK: zero:
zero:
; CHECK: store i32 7, i32 addrspace(1)* %out
  store i32 7, i32 addrspace(1)* %out
; CHECK: br label %done
  br label %done

; CHECK: done:
done:
; CHECK: ret void
  ret void
}
