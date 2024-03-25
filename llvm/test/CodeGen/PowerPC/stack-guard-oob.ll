; RUN: llc -mtriple=powerpc64le -O0 < %s | FileCheck %s
; RUN: llc -mtriple=powerpc64-ibm-aix-xcoff -O0 < %s | FileCheck %s --check-prefix=AIX
; RUN: llc -mtriple=powerpc-ibm-aix-xcoff -O0 < %s | FileCheck %s --check-prefix=AIX

; CHECK-LABEL: in_bounds:
; CHECK-NOT: __stack_chk_guard
; AIX-NOT: __ssp_canary_word
define i32 @in_bounds() #0 {
  %var = alloca i32, align 4
  store i32 0, ptr %var, align 4
  %ret = load i32, ptr %var, align 4
  ret i32 %ret
}

; CHECK-LABEL: constant_out_of_bounds:
; CHECK: __stack_chk_guard
; AIX: __ssp_canary_word
define i32 @constant_out_of_bounds() #0 {
  %var = alloca i32, align 4
  store i32 0, ptr %var, align 4
  %gep = getelementptr inbounds i32, ptr %var, i32 1
  %ret = load i32, ptr %gep, align 4
  ret i32 %ret
}

; CHECK-LABEL: nonconstant_out_of_bounds:
; CHECK: __stack_chk_guard
; AIX: __ssp_canary_word
define i32 @nonconstant_out_of_bounds(i32 %n) #0 {
  %var = alloca i32, align 4
  store i32 0, ptr %var, align 4
  %gep = getelementptr inbounds i32, ptr %var, i32 %n
  %ret = load i32, ptr %gep, align 4
  ret i32 %ret
}

; CHECK-LABEL: phi_before_gep_in_bounds:
; CHECK-NOT: __stack_chk_guard
; AIX-NOT: __ssp_canary_word
define i32 @phi_before_gep_in_bounds(i32 %k) #0 {
entry:
  %var1 = alloca i32, align 4
  %var2 = alloca i32, align 4
  store i32 0, ptr %var1, align 4
  store i32 0, ptr %var2, align 4
  %cmp = icmp ne i32 %k, 0
  br i1 %cmp, label %if, label %then

if:
  br label %then

then:
  %ptr = phi ptr [ %var1, %entry ], [ %var2, %if ]
  %ret = load i32, ptr %ptr, align 4
  ret i32 %ret
}

; CHECK-LABEL: phi_before_gep_constant_out_of_bounds:
; CHECK: __stack_chk_guard
; AIX: __ssp_canary_word
define i32 @phi_before_gep_constant_out_of_bounds(i32 %k) #0 {
entry:
  %var1 = alloca i32, align 4
  %var2 = alloca i32, align 4
  store i32 0, ptr %var1, align 4
  store i32 0, ptr %var2, align 4
  %cmp = icmp ne i32 %k, 0
  br i1 %cmp, label %if, label %then

if:
  br label %then

then:
  %ptr = phi ptr [ %var1, %entry ], [ %var2, %if ]
  %gep = getelementptr inbounds i32, ptr %ptr, i32 1
  %ret = load i32, ptr %gep, align 4
  ret i32 %ret
}

; CHECK-LABEL: phi_before_gep_nonconstant_out_of_bounds:
; CHECK: __stack_chk_guard
; AIX: __ssp_canary_word
define i32 @phi_before_gep_nonconstant_out_of_bounds(i32 %k, i32 %n) #0 {
entry:
  %var1 = alloca i32, align 4
  %var2 = alloca i32, align 4
  store i32 0, ptr %var1, align 4
  store i32 0, ptr %var2, align 4
  %cmp = icmp ne i32 %k, 0
  br i1 %cmp, label %if, label %then

if:
  br label %then

then:
  %ptr = phi ptr [ %var1, %entry ], [ %var2, %if ]
  %gep = getelementptr inbounds i32, ptr %ptr, i32 %n
  %ret = load i32, ptr %gep, align 4
  ret i32 %ret
}

; CHECK-LABEL: phi_after_gep_in_bounds:
; CHECK-NOT: __stack_chk_guard
; AIX-NOT: __ssp_canary_word
define i32 @phi_after_gep_in_bounds(i32 %k) #0 {
entry:
  %var1 = alloca i32, align 4
  %var2 = alloca i32, align 4
  store i32 0, ptr %var1, align 4
  store i32 0, ptr %var2, align 4
  %cmp = icmp ne i32 %k, 0
  br i1 %cmp, label %if, label %else

if:
  br label %then

else:
  br label %then

then:
  %ptr = phi ptr [ %var1, %if ], [ %var2, %else ]
  %ret = load i32, ptr %ptr, align 4
  ret i32 %ret
}

; CHECK-LABEL: phi_after_gep_constant_out_of_bounds_a:
; CHECK: __stack_chk_guard
; AIX: __ssp_canary_word
define i32 @phi_after_gep_constant_out_of_bounds_a(i32 %k) #0 {
entry:
  %var1 = alloca i32, align 4
  %var2 = alloca i32, align 4
  store i32 0, ptr %var1, align 4
  store i32 0, ptr %var2, align 4
  %cmp = icmp ne i32 %k, 0
  br i1 %cmp, label %if, label %else

if:
  br label %then

else:
  %gep2 = getelementptr inbounds i32, ptr %var2, i32 1
  br label %then

then:
  %ptr = phi ptr [ %var1, %if ], [ %gep2, %else ]
  %ret = load i32, ptr %ptr, align 4
  ret i32 %ret
}

; CHECK-LABEL: phi_after_gep_constant_out_of_bounds_b:
; CHECK: __stack_chk_guard
; AIX: __ssp_canary_word
define i32 @phi_after_gep_constant_out_of_bounds_b(i32 %k) #0 {
entry:
  %var1 = alloca i32, align 4
  %var2 = alloca i32, align 4
  store i32 0, ptr %var1, align 4
  store i32 0, ptr %var2, align 4
  %cmp = icmp ne i32 %k, 0
  br i1 %cmp, label %if, label %else

if:
  %gep1 = getelementptr inbounds i32, ptr %var1, i32 1
  br label %then

else:
  br label %then

then:
  %ptr = phi ptr [ %gep1, %if ], [ %var2, %else ]
  %ret = load i32, ptr %ptr, align 4
  ret i32 %ret
}

; CHECK-LABEL: phi_different_types_a:
; CHECK: __stack_chk_guard
; AIX: __ssp_canary_word
define i64 @phi_different_types_a(i32 %k) #0 {
entry:
  %var1 = alloca i64, align 4
  %var2 = alloca i32, align 4
  store i64 0, ptr %var1, align 4
  store i32 0, ptr %var2, align 4
  %cmp = icmp ne i32 %k, 0
  br i1 %cmp, label %if, label %then

if:
  br label %then

then:
  %ptr = phi ptr [ %var1, %entry ], [ %var2, %if ]
  %ret = load i64, ptr %ptr, align 4
  ret i64 %ret
}

; CHECK-LABEL: phi_different_types_b:
; CHECK: __stack_chk_guard
; AIX: __ssp_canary_word
define i64 @phi_different_types_b(i32 %k) #0 {
entry:
  %var1 = alloca i32, align 4
  %var2 = alloca i64, align 4
  store i32 0, ptr %var1, align 4
  store i64 0, ptr %var2, align 4
  %cmp = icmp ne i32 %k, 0
  br i1 %cmp, label %if, label %then

if:
  br label %then

then:
  %ptr = phi ptr [ %var2, %entry ], [ %var1, %if ]
  %ret = load i64, ptr %ptr, align 4
  ret i64 %ret
}

; CHECK-LABEL: phi_after_gep_nonconstant_out_of_bounds_a:
; CHECK: __stack_chk_guard
; AIX: __ssp_canary_word
define i32 @phi_after_gep_nonconstant_out_of_bounds_a(i32 %k, i32 %n) #0 {
entry:
  %var1 = alloca i32, align 4
  %var2 = alloca i32, align 4
  store i32 0, ptr %var1, align 4
  store i32 0, ptr %var2, align 4
  %cmp = icmp ne i32 %k, 0
  br i1 %cmp, label %if, label %else

if:
  br label %then

else:
  %gep2 = getelementptr inbounds i32, ptr %var2, i32 %n
  br label %then

then:
  %ptr = phi ptr [ %var1, %if ], [ %gep2, %else ]
  %ret = load i32, ptr %ptr, align 4
  ret i32 %ret
}

; CHECK-LABEL: phi_after_gep_nonconstant_out_of_bounds_b:
; CHECK: __stack_chk_guard
; AIX: __ssp_canary_word
define i32 @phi_after_gep_nonconstant_out_of_bounds_b(i32 %k, i32 %n) #0 {
entry:
  %var1 = alloca i32, align 4
  %var2 = alloca i32, align 4
  store i32 0, ptr %var1, align 4
  store i32 0, ptr %var2, align 4
  %cmp = icmp ne i32 %k, 0
  br i1 %cmp, label %if, label %else

if:
  %gep1 = getelementptr inbounds i32, ptr %var1, i32 %n
  br label %then

else:
  br label %then

then:
  %ptr = phi ptr [ %gep1, %if ], [ %var2, %else ]
  %ret = load i32, ptr %ptr, align 4
  ret i32 %ret
}

%struct.outer = type { %struct.inner, %struct.inner }
%struct.inner = type { i32, i32 }

; CHECK-LABEL: struct_in_bounds:
; CHECK-NOT: __stack_chk_guard
; AIX-NOT: __ssp_canary_word
define void @struct_in_bounds() #0 {
  %var = alloca %struct.outer, align 4
  %outergep = getelementptr inbounds %struct.outer, ptr %var, i32 0, i32 1
  %innergep = getelementptr inbounds %struct.inner, ptr %outergep, i32 0, i32 1
  store i32 0, ptr %innergep, align 4
  ret void
}

; CHECK-LABEL: struct_constant_out_of_bounds_a:
; CHECK: __stack_chk_guard
; AIX: __ssp_canary_word
define void @struct_constant_out_of_bounds_a() #0 {
  %var = alloca %struct.outer, align 4
  %outergep = getelementptr inbounds %struct.outer, ptr %var, i32 1, i32 0
  store i32 0, ptr %outergep, align 4
  ret void
}

; CHECK-LABEL: struct_constant_out_of_bounds_b:
; Here the offset is out-of-bounds of the addressed struct.inner member, but
; still within bounds of the outer struct so no stack guard is needed.
; CHECK-NOT: __stack_chk_guard
; AIX-NOT: __ssp_canary_word
define void @struct_constant_out_of_bounds_b() #0 {
  %var = alloca %struct.outer, align 4
  %innergep = getelementptr inbounds %struct.inner, ptr %var, i32 1, i32 0
  store i32 0, ptr %innergep, align 4
  ret void
}

; CHECK-LABEL: struct_constant_out_of_bounds_c:
; Here we are out-of-bounds of both the inner and outer struct.
; CHECK: __stack_chk_guard
; AIX: __ssp_canary_word
define void @struct_constant_out_of_bounds_c() #0 {
  %var = alloca %struct.outer, align 4
  %outergep = getelementptr inbounds %struct.outer, ptr %var, i32 0, i32 1
  %innergep = getelementptr inbounds %struct.inner, ptr %outergep, i32 1, i32 0
  store i32 0, ptr %innergep, align 4
  ret void
}

; CHECK-LABEL: struct_nonconstant_out_of_bounds_a:
; CHECK: __stack_chk_guard
; AIX: __ssp_canary_word
define void @struct_nonconstant_out_of_bounds_a(i32 %n) #0 {
  %var = alloca %struct.outer, align 4
  %outergep = getelementptr inbounds %struct.outer, ptr %var, i32 %n, i32 0
  store i32 0, ptr %outergep, align 4
  ret void
}

; CHECK-LABEL: struct_nonconstant_out_of_bounds_b:
; CHECK: __stack_chk_guard
; AIX: __ssp_canary_word
define void @struct_nonconstant_out_of_bounds_b(i32 %n) #0 {
  %var = alloca %struct.outer, align 4
  %innergep = getelementptr inbounds %struct.inner, ptr %var, i32 %n, i32 0
  store i32 0, ptr %innergep, align 4
  ret void
}

; CHECK-LABEL: bitcast_smaller_load
; CHECK-NOT: __stack_chk_guard
; AIX-NOT: __ssp_canary_word
define i32 @bitcast_smaller_load() #0 {
  %var = alloca i64, align 4
  store i64 0, ptr %var, align 4
  %ret = load i32, ptr %var, align 4
  ret i32 %ret
}

; CHECK-LABEL: bitcast_same_size_load
; CHECK-NOT: __stack_chk_guard
; AIX-NOT: __ssp_canary_word
define i32 @bitcast_same_size_load() #0 {
  %var = alloca i64, align 4
  store i64 0, ptr %var, align 4
  %gep = getelementptr inbounds %struct.inner, ptr %var, i32 0, i32 1
  %ret = load i32, ptr %gep, align 4
  ret i32 %ret
}

; CHECK-LABEL: bitcast_larger_load
; CHECK: __stack_chk_guard
; AIX: __ssp_canary_word
define i64 @bitcast_larger_load() #0 {
  %var = alloca i32, align 4
  store i32 0, ptr %var, align 4
  %ret = load i64, ptr %var, align 4
  ret i64 %ret
}

; CHECK-LABEL: bitcast_larger_store
; CHECK: __stack_chk_guard
; AIX: __ssp_canary_word
define i32 @bitcast_larger_store() #0 {
  %var = alloca i32, align 4
  store i64 0, ptr %var, align 4
  %ret = load i32, ptr %var, align 4
  ret i32 %ret
}

; CHECK-LABEL: bitcast_larger_cmpxchg
; CHECK: __stack_chk_guard
; AIX: __ssp_canary_word
define i64 @bitcast_larger_cmpxchg(i64 %desired, i64 %new) #0 {
  %var = alloca i32, align 4
  %pair = cmpxchg ptr %var, i64 %desired, i64 %new seq_cst monotonic
  %ret = extractvalue { i64, i1 } %pair, 0
  ret i64 %ret
}

; CHECK-LABEL: bitcast_larger_atomic_rmw
; CHECK: __stack_chk_guard
; AIX: __ssp_canary_word
define i64 @bitcast_larger_atomic_rmw() #0 {
  %var = alloca i32, align 4
  %ret = atomicrmw add ptr %var, i64 1 monotonic
  ret i64 %ret
}

%struct.packed = type <{ i16, i32 }>

; CHECK-LABEL: bitcast_overlap
; CHECK: __stack_chk_guard
; AIX: __ssp_canary_word
define i32 @bitcast_overlap() #0 {
  %var = alloca i32, align 4
  %gep = getelementptr inbounds %struct.packed, ptr %var, i32 0, i32 1
  %ret = load i32, ptr %gep, align 2
  ret i32 %ret
}

%struct.multi_dimensional = type { [10 x [10 x i32]], i32 }

; CHECK-LABEL: multi_dimensional_array
; CHECK: __stack_chk_guard
; AIX: __ssp_canary_word
define i32 @multi_dimensional_array() #0 {
  %var = alloca %struct.multi_dimensional, align 4
  %gep2 = getelementptr inbounds [10 x [10 x i32]], ptr %var, i32 0, i32 10
  %gep3 = getelementptr inbounds [10 x i32], ptr %gep2, i32 0, i32 5
  %ret = load i32, ptr %gep3, align 4
  ret i32 %ret
}

attributes #0 = { sspstrong }

!llvm.module.flags = !{!0}
!0 = !{i32 7, !"direct-access-external-data", i32 1}
