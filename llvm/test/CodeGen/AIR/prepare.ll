; AIRPrepare performs the final pre-serialization IR normalizations:
;   - struct-typed phis are split into one scalar phi per element
;   - i1 globals/GEPs/loads/stores are legalized to i8 (AIR has no i1 memory)
;   - device-pointer atomic intrinsic call sites get a ptrtoint+inttoptr
;     typed-pointer transition (so the writer can pick the intrinsic-expected
;     pointee type, not the upstream GEP's float)
;   - pointer phis with an undef incoming are converted to i64 phis with
;     ptrtoint / inttoptr bridges
; RUN: llc -mtriple=air %s -o - | FileCheck %s

declare void @air.atomic.global.add.explicit.i32(ptr addrspace(1), i32, i32, i32, i32)

@i1_arr = internal addrspace(3) global [8 x i1] undef
; CHECK: @i1_arr = internal addrspace(3) global [8 x i8] undef

; CHECK-LABEL: define ptr addrspace(1) @undef_ptr_phi
; CHECK: phi i64
; CHECK: inttoptr i64
define ptr addrspace(1) @undef_ptr_phi(i1 %c, ptr addrspace(1) %p) {
entry:
  br i1 %c, label %then, label %else
then:
  br label %merge
else:
  br label %merge
merge:
  %r = phi ptr addrspace(1) [ %p, %then ], [ undef, %else ]
  ret ptr addrspace(1) %r
}

; CHECK-LABEL: define i1 @i1_array_global
; CHECK: getelementptr [8 x i8], ptr addrspace(3) @i1_arr
; CHECK: %[[L:.*]] = load i8, ptr addrspace(3)
; CHECK: trunc i8 %[[L]] to i1
; CHECK: zext i1 %{{.*}} to i8
; CHECK: store i8
define i1 @i1_array_global(i32 %idx, i1 %v) {
entry:
  %g = getelementptr [8 x i1], ptr addrspace(3) @i1_arr, i32 0, i32 %idx
  %old = load i1, ptr addrspace(3) %g
  store i1 %v, ptr addrspace(3) %g
  ret i1 %old
}

; CHECK-LABEL: define ptr addrspace(1) @struct_phi
; CHECK-NOT: phi { ptr addrspace(1) }
; CHECK: phi i64
; CHECK: inttoptr i64
define ptr addrspace(1) @struct_phi(i1 %c, ptr addrspace(1) %a, ptr addrspace(1) %b) {
entry:
  %ia = insertvalue { ptr addrspace(1) } undef, ptr addrspace(1) %a, 0
  %ib = insertvalue { ptr addrspace(1) } undef, ptr addrspace(1) %b, 0
  br i1 %c, label %then, label %else
then:
  br label %merge
else:
  br label %merge
merge:
  %s = phi { ptr addrspace(1) } [ %ia, %then ], [ %ib, %else ]
  %p = extractvalue { ptr addrspace(1) } %s, 0
  ret ptr addrspace(1) %p
}

; CHECK-LABEL: define void @i1_gep
; CHECK: getelementptr inbounds i8, ptr addrspace(1) %p
define void @i1_gep(ptr addrspace(1) %p, i32 %idx) {
entry:
  %g = getelementptr inbounds i1, ptr addrspace(1) %p, i32 %idx
  store i8 1, ptr addrspace(1) %g
  ret void
}

; CHECK-LABEL: define void @atomic_fixup
; CHECK: getelementptr inbounds float, ptr addrspace(1) %buf
; CHECK-NEXT: ptrtoint ptr addrspace(1) %{{.*}} to i64
; CHECK-NEXT: inttoptr i64 %{{.*}} to ptr addrspace(1)
; CHECK-NEXT: call void @air.atomic.global.add.explicit.i32
define void @atomic_fixup(ptr addrspace(1) %buf, i32 %i) {
entry:
  %gep = getelementptr inbounds float, ptr addrspace(1) %buf, i32 %i
  call void @air.atomic.global.add.explicit.i32(ptr addrspace(1) %gep, i32 1, i32 0, i32 0, i32 0)
  ret void
}
