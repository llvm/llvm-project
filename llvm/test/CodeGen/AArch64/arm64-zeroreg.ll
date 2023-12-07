; RUN: llc -o - %s | FileCheck %s
target triple = "aarch64--"

declare void @begin()
declare void @end()

; Test that we use the zero register before regalloc and do not unnecessarily
; clobber a register with the SUBS (cmp) instruction.
; CHECK-LABEL: func:
define void @func(ptr %addr) {
  ; We should not see any spills or reloads between begin and end
  ; CHECK: bl begin
  ; CHECK-NOT: str{{.*}}sp
  ; CHECK-NOT: Folded Spill
  ; CHECK-NOT: ldr{{.*}}sp
  ; CHECK-NOT: Folded Reload
  call void @begin()
  %v0 = load volatile i64, ptr %addr  
  %v1 = load volatile i64, ptr %addr  
  %v2 = load volatile i64, ptr %addr  
  %v3 = load volatile i64, ptr %addr  
  %v4 = load volatile i64, ptr %addr  
  %v5 = load volatile i64, ptr %addr  
  %v6 = load volatile i64, ptr %addr  
  %v7 = load volatile i64, ptr %addr  
  %v8 = load volatile i64, ptr %addr  
  %v9 = load volatile i64, ptr %addr  
  %v10 = load volatile i64, ptr %addr  
  %v11 = load volatile i64, ptr %addr  
  %v12 = load volatile i64, ptr %addr  
  %v13 = load volatile i64, ptr %addr  
  %v14 = load volatile i64, ptr %addr  
  %v15 = load volatile i64, ptr %addr  
  %v16 = load volatile i64, ptr %addr  
  %v17 = load volatile i64, ptr %addr  
  %v18 = load volatile i64, ptr %addr  
  %v19 = load volatile i64, ptr %addr  
  %v20 = load volatile i64, ptr %addr
  %v21 = load volatile i64, ptr %addr
  %v22 = load volatile i64, ptr %addr
  %v23 = load volatile i64, ptr %addr
  %v24 = load volatile i64, ptr %addr
  %v25 = load volatile i64, ptr %addr
  %v26 = load volatile i64, ptr %addr
  %v27 = load volatile i64, ptr %addr
  %v28 = load volatile i64, ptr %addr
  %v29 = load volatile i64, ptr %addr

  %c = icmp eq i64 %v0, %v1
  br i1 %c, label %if.then, label %if.end

if.then:
  store volatile i64 %v2, ptr %addr
  br label %if.end

if.end:
  store volatile i64 %v0, ptr %addr
  store volatile i64 %v1, ptr %addr
  store volatile i64 %v2, ptr %addr
  store volatile i64 %v3, ptr %addr
  store volatile i64 %v4, ptr %addr
  store volatile i64 %v5, ptr %addr
  store volatile i64 %v6, ptr %addr
  store volatile i64 %v7, ptr %addr
  store volatile i64 %v8, ptr %addr
  store volatile i64 %v9, ptr %addr
  store volatile i64 %v10, ptr %addr
  store volatile i64 %v11, ptr %addr
  store volatile i64 %v12, ptr %addr
  store volatile i64 %v13, ptr %addr
  store volatile i64 %v14, ptr %addr
  store volatile i64 %v15, ptr %addr
  store volatile i64 %v16, ptr %addr
  store volatile i64 %v17, ptr %addr
  store volatile i64 %v18, ptr %addr
  store volatile i64 %v19, ptr %addr
  store volatile i64 %v20, ptr %addr
  store volatile i64 %v21, ptr %addr
  store volatile i64 %v22, ptr %addr
  store volatile i64 %v23, ptr %addr
  store volatile i64 %v24, ptr %addr
  store volatile i64 %v25, ptr %addr
  store volatile i64 %v26, ptr %addr
  store volatile i64 %v27, ptr %addr
  store volatile i64 %v28, ptr %addr
  store volatile i64 %v29, ptr %addr
  ; CHECK: bl end
  call void @end()

  ret void
}
