; XFAIL: *
; RUN: opt -S -basicaa -newgvn < %s | FileCheck %s

@a = external constant i32
; We can value forward across the fence since we can (semantically) 
; reorder the following load before the fence.
define i32 @test(i32* %addr.i) {
; CHECK-LABEL: @test
; CHECK: store
; CHECK: fence
; CHECK-NOT: load
; CHECK: ret
  store i32 5, i32* %addr.i, align 4
  fence release
  %a = load i32, i32* %addr.i, align 4
  ret i32 %a
}

; Same as above
define i32 @test2(i32* %addr.i) {
; CHECK-LABEL: @test2
; CHECK-NEXT: fence
; CHECK-NOT: load
; CHECK: ret
  %a = load i32, i32* %addr.i, align 4
  fence release
  %a2 = load i32, i32* %addr.i, align 4
  %res = sub i32 %a, %a2
  ret i32 %res
}

; We can not value forward across an acquire barrier since we might
; be syncronizing with another thread storing to the same variable
; followed by a release fence.  This is not so much enforcing an
; ordering property (though it is that too), but a liveness 
; property.  We expect to eventually see the value of store by
; another thread when spinning on that location.  
define i32 @test3(i32* noalias %addr.i, i32* noalias %otheraddr) {
; CHECK-LABEL: @test3
; CHECK: load
; CHECK: fence
; CHECK: load
; CHECK: ret i32 %res
  ; the following code is intented to model the unrolling of
  ; two iterations in a spin loop of the form:
  ;   do { fence acquire: tmp = *%addr.i; ) while (!tmp);
  ; It's hopefully clear that allowing PRE to turn this into:
  ;   if (!*%addr.i) while(true) {} would be unfortunate
  fence acquire
  %a = load i32, i32* %addr.i, align 4
  fence acquire
  %a2 = load i32, i32* %addr.i, align 4
  %res = sub i32 %a, %a2
  ret i32 %res
}

; We can forward the value forward the load
; across both the fences, because the load is from
; a constant memory location.
define i32 @test4(i32* %addr) {
; CHECK-LABEL: @test4
; CHECK-NOT: load
; CHECK: fence release
; CHECK: store
; CHECK: fence seq_cst
; CHECK: ret i32 0
  %var = load i32, i32* @a
  fence release
  store i32 42, i32* %addr, align 8
  fence seq_cst
  %var2 = load i32, i32* @a
  %var3 = sub i32 %var, %var2
  ret i32 %var3
}

; Another example of why forwarding across an acquire fence is problematic
; can be seen in a normal locking operation.  Say we had:
; *p = 5; unlock(l); lock(l); use(p);
; forwarding the store to p would be invalid.  A reasonable implementation
; of unlock and lock might be:
; unlock() { atomicrmw sub %l, 1 unordered; fence release }
; lock() { 
;   do {
;     %res = cmpxchg %p, 0, 1, monotonic monotonic
;   } while(!%res.success)
;   fence acquire;
; }
; Given we chose to forward across the release fence, we clearly can't forward
; across the acquire fence as well.

