; RUN: llc -o - %s | FileCheck %s
target triple = "aarch64--"

declare void @begin()
declare void @end()

; Test that we use the zero register before regalloc and do not unnecessarily
; clobber a register with the SUBS (cmp) instruction.
; CHECK-LABEL: func:
define void @func(i64* %addr) {
  ; We should not see any spills or reloads between begin and end
  ; CHECK: bl begin
  ; CHECK-NOT: str{{.*}}sp
  ; CHECK-NOT: Folded Spill
  ; CHECK-NOT: ldr{{.*}}sp
  ; CHECK-NOT: Folded Reload
  call void @begin()
  %v0 = load volatile i64, i64* %addr  
  %v1 = load volatile i64, i64* %addr  
  %v2 = load volatile i64, i64* %addr  
  %v3 = load volatile i64, i64* %addr  
  %v4 = load volatile i64, i64* %addr  
  %v5 = load volatile i64, i64* %addr  
  %v6 = load volatile i64, i64* %addr  
  %v7 = load volatile i64, i64* %addr  
  %v8 = load volatile i64, i64* %addr  
  %v9 = load volatile i64, i64* %addr  
  %v10 = load volatile i64, i64* %addr  
  %v11 = load volatile i64, i64* %addr  
  %v12 = load volatile i64, i64* %addr  
  %v13 = load volatile i64, i64* %addr  
  %v14 = load volatile i64, i64* %addr  
  %v15 = load volatile i64, i64* %addr  
  %v16 = load volatile i64, i64* %addr  
  %v17 = load volatile i64, i64* %addr  
  %v18 = load volatile i64, i64* %addr  
  %v19 = load volatile i64, i64* %addr  
  %v20 = load volatile i64, i64* %addr
  %v21 = load volatile i64, i64* %addr
  %v22 = load volatile i64, i64* %addr
  %v23 = load volatile i64, i64* %addr
  %v24 = load volatile i64, i64* %addr
  %v25 = load volatile i64, i64* %addr
  %v26 = load volatile i64, i64* %addr
  %v27 = load volatile i64, i64* %addr
  %v28 = load volatile i64, i64* %addr
  %v29 = load volatile i64, i64* %addr

  %c = icmp eq i64 %v0, %v1
  br i1 %c, label %if.then, label %if.end

if.then:
  store volatile i64 %v2, i64* %addr
  br label %if.end

if.end:
  store volatile i64 %v0, i64* %addr
  store volatile i64 %v1, i64* %addr
  store volatile i64 %v2, i64* %addr
  store volatile i64 %v3, i64* %addr
  store volatile i64 %v4, i64* %addr
  store volatile i64 %v5, i64* %addr
  store volatile i64 %v6, i64* %addr
  store volatile i64 %v7, i64* %addr
  store volatile i64 %v8, i64* %addr
  store volatile i64 %v9, i64* %addr
  store volatile i64 %v10, i64* %addr
  store volatile i64 %v11, i64* %addr
  store volatile i64 %v12, i64* %addr
  store volatile i64 %v13, i64* %addr
  store volatile i64 %v14, i64* %addr
  store volatile i64 %v15, i64* %addr
  store volatile i64 %v16, i64* %addr
  store volatile i64 %v17, i64* %addr
  store volatile i64 %v18, i64* %addr
  store volatile i64 %v19, i64* %addr
  store volatile i64 %v20, i64* %addr
  store volatile i64 %v21, i64* %addr
  store volatile i64 %v22, i64* %addr
  store volatile i64 %v23, i64* %addr
  store volatile i64 %v24, i64* %addr
  store volatile i64 %v25, i64* %addr
  store volatile i64 %v26, i64* %addr
  store volatile i64 %v27, i64* %addr
  store volatile i64 %v28, i64* %addr
  store volatile i64 %v29, i64* %addr
  ; CHECK: bl end
  call void @end()

  ret void
}
