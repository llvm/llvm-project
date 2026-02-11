; RUN: llc -march=amdgcn -mcpu=gfx900  < %s | FileCheck %s
; RUN: llc -march=amdgcn -mcpu=gfx942  < %s | FileCheck %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 < %s | FileCheck %s

; Loop body exceeds MaxAsyncMarkers on first iteration
; Preloop: 5 markers
; Loop body: 18 markers

; CHECK-LABEL: test_loop_exceeds_max_first_iteration:
; CHECK: ; wait_asyncmark(3)
; CHECK-NEXT: s_waitcnt vmcnt(3)

define void @test_loop_exceeds_max_first_iteration(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 %n, ptr addrspace(1) %out) {
entry:
  ; Preloop: 5 async LDS DMA operations
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()

  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()

  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()

  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()

  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()

  br label %loop_header

loop_header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop_body ]
  %i.next = add i32 %i, 1
  %cmp = icmp slt i32 %i, %n
  br i1 %cmp, label %loop_body, label %exit

loop_body:
  ; Loop body with 18 async operations
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()

  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()

  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()

  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()

  br label %loop_header

exit:
  call void @llvm.amdgcn.wait.asyncmark(i16 3)
  %lds_val = load i32, ptr addrspace(3) %lds
  store i32 %lds_val, ptr addrspace(1) %out
  ret void
}

; Loop body does not exceed MaxAsyncMarkers on first iteration
; Preloop: 5 markers
; Loop body: 5 markers

; CHECK-LABEL: test_loop_needs_more_iterations:
; CHECK: ; wait_asyncmark(3)
; CHECK-NEXT: s_waitcnt vmcnt(3)

define void @test_loop_needs_more_iterations(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 %n, ptr addrspace(1) %out) {
entry:
  ; Preloop: 5 async LDS DMA operations
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()

  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()

  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()

  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()

  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()

  br label %loop_header

loop_header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop_body ]
  %i.next = add i32 %i, 1
  %cmp = icmp slt i32 %i, %n
  br i1 %cmp, label %loop_body, label %exit

loop_body:
  ; Loop body with 5 async operations
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()

  br label %loop_header

exit:
  call void @llvm.amdgcn.wait.asyncmark(i16 3)
  %lds_val = load i32, ptr addrspace(3) %lds
  store i32 %lds_val, ptr addrspace(1) %out
  ret void
}

; Merge exceeds MaxAsyncMarkers

; CHECK-LABEL: max_when_merged:
; CHECK: ; wait_asyncmark(17)
; CHECK-NEXT: s_waitcnt vmcnt(15)

define void @max_when_merged(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 %n, ptr addrspace(1) %out) {
entry:
  %cmp = icmp slt i32 0, %n
  br i1 %cmp, label %then, label %else

then:
  ; 5 async operations
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()

  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()

  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()

  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()

  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()

  br label %endif

else:
  ; 18 async operations
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()

  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()

  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()

  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()

  br label %endif

endif:
  call void @llvm.amdgcn.wait.asyncmark(i16 17)
  %lds_val = load i32, ptr addrspace(3) %lds
  store i32 %lds_val, ptr addrspace(1) %out
  ret void
}

; Straightline exceeds MaxAsyncMarkers

; CHECK-LABEL: no_max_in_straightline:
; CHECK: ; wait_asyncmark(17)
; CHECK-NEXT: s_waitcnt vmcnt(17)

define void @no_max_in_straightline(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 %n, ptr addrspace(1) %out) {
  ; 18 async operations
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()

  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()

  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()

  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %in, ptr addrspace(3) %lds, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()

  call void @llvm.amdgcn.wait.asyncmark(i16 17)
  %lds_val = load i32, ptr addrspace(3) %lds
  store i32 %lds_val, ptr addrspace(1) %out
  ret void
}
