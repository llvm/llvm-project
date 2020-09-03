; RUN: llc -mtriple=i686 -O0 < %s | FileCheck %s
; RUN: llc -mtriple=x86_64 -O0 < %s | FileCheck %s

; CHECK-LABEL: in_bounds:
; CHECK-NOT: __stack_chk_guard
define i32 @in_bounds() #0 {
  %var = alloca i32, align 4
  store i32 0, i32* %var, align 4
  %gep = getelementptr inbounds i32, i32* %var, i32 0
  %ret = load i32, i32* %gep, align 4
  ret i32 %ret
}

; CHECK-LABEL: constant_out_of_bounds:
; CHECK: __stack_chk_guard
define i32 @constant_out_of_bounds() #0 {
  %var = alloca i32, align 4
  store i32 0, i32* %var, align 4
  %gep = getelementptr inbounds i32, i32* %var, i32 1
  %ret = load i32, i32* %gep, align 4
  ret i32 %ret
}

; CHECK-LABEL: nonconstant_out_of_bounds:
; CHECK: __stack_chk_guard
define i32 @nonconstant_out_of_bounds(i32 %n) #0 {
  %var = alloca i32, align 4
  store i32 0, i32* %var, align 4
  %gep = getelementptr inbounds i32, i32* %var, i32 %n
  %ret = load i32, i32* %gep, align 4
  ret i32 %ret
}

; CHECK-LABEL: phi_before_gep_in_bounds:
; CHECK-NOT: __stack_chk_guard
define i32 @phi_before_gep_in_bounds(i32 %k) #0 {
entry:
  %var1 = alloca i32, align 4
  %var2 = alloca i32, align 4
  store i32 0, i32* %var1, align 4
  store i32 0, i32* %var2, align 4
  %cmp = icmp ne i32 %k, 0
  br i1 %cmp, label %if, label %then

if:
  br label %then

then:
  %ptr = phi i32* [ %var1, %entry ], [ %var2, %if ]
  %gep = getelementptr inbounds i32, i32* %ptr, i32 0
  %ret = load i32, i32* %gep, align 4
  ret i32 %ret
}

; CHECK-LABEL: phi_before_gep_constant_out_of_bounds:
; CHECK: __stack_chk_guard
define i32 @phi_before_gep_constant_out_of_bounds(i32 %k) #0 {
entry:
  %var1 = alloca i32, align 4
  %var2 = alloca i32, align 4
  store i32 0, i32* %var1, align 4
  store i32 0, i32* %var2, align 4
  %cmp = icmp ne i32 %k, 0
  br i1 %cmp, label %if, label %then

if:
  br label %then

then:
  %ptr = phi i32* [ %var1, %entry ], [ %var2, %if ]
  %gep = getelementptr inbounds i32, i32* %ptr, i32 1
  %ret = load i32, i32* %gep, align 4
  ret i32 %ret
}

; CHECK-LABEL: phi_before_gep_nonconstant_out_of_bounds:
; CHECK: __stack_chk_guard
define i32 @phi_before_gep_nonconstant_out_of_bounds(i32 %k, i32 %n) #0 {
entry:
  %var1 = alloca i32, align 4
  %var2 = alloca i32, align 4
  store i32 0, i32* %var1, align 4
  store i32 0, i32* %var2, align 4
  %cmp = icmp ne i32 %k, 0
  br i1 %cmp, label %if, label %then

if:
  br label %then

then:
  %ptr = phi i32* [ %var1, %entry ], [ %var2, %if ]
  %gep = getelementptr inbounds i32, i32* %ptr, i32 %n
  %ret = load i32, i32* %gep, align 4
  ret i32 %ret
}

; CHECK-LABEL: phi_after_gep_in_bounds:
; CHECK-NOT: __stack_chk_guard
define i32 @phi_after_gep_in_bounds(i32 %k) #0 {
entry:
  %var1 = alloca i32, align 4
  %var2 = alloca i32, align 4
  store i32 0, i32* %var1, align 4
  store i32 0, i32* %var2, align 4
  %cmp = icmp ne i32 %k, 0
  br i1 %cmp, label %if, label %else

if:
  %gep1 = getelementptr inbounds i32, i32* %var1, i32 0
  br label %then

else:
  %gep2 = getelementptr inbounds i32, i32* %var2, i32 0
  br label %then

then:
  %ptr = phi i32* [ %gep1, %if ], [ %gep2, %else ]
  %ret = load i32, i32* %ptr, align 4
  ret i32 %ret
}

; CHECK-LABEL: phi_after_gep_constant_out_of_bounds_a:
; CHECK: __stack_chk_guard
define i32 @phi_after_gep_constant_out_of_bounds_a(i32 %k) #0 {
entry:
  %var1 = alloca i32, align 4
  %var2 = alloca i32, align 4
  store i32 0, i32* %var1, align 4
  store i32 0, i32* %var2, align 4
  %cmp = icmp ne i32 %k, 0
  br i1 %cmp, label %if, label %else

if:
  %gep1 = getelementptr inbounds i32, i32* %var1, i32 0
  br label %then

else:
  %gep2 = getelementptr inbounds i32, i32* %var2, i32 1
  br label %then

then:
  %ptr = phi i32* [ %gep1, %if ], [ %gep2, %else ]
  %ret = load i32, i32* %ptr, align 4
  ret i32 %ret
}

; CHECK-LABEL: phi_after_gep_constant_out_of_bounds_b:
; CHECK: __stack_chk_guard
define i32 @phi_after_gep_constant_out_of_bounds_b(i32 %k) #0 {
entry:
  %var1 = alloca i32, align 4
  %var2 = alloca i32, align 4
  store i32 0, i32* %var1, align 4
  store i32 0, i32* %var2, align 4
  %cmp = icmp ne i32 %k, 0
  br i1 %cmp, label %if, label %else

if:
  %gep1 = getelementptr inbounds i32, i32* %var1, i32 1
  br label %then

else:
  %gep2 = getelementptr inbounds i32, i32* %var2, i32 0
  br label %then

then:
  %ptr = phi i32* [ %gep1, %if ], [ %gep2, %else ]
  %ret = load i32, i32* %ptr, align 4
  ret i32 %ret
}

; CHECK-LABEL: phi_different_types_a:
; CHECK: __stack_chk_guard
define i64 @phi_different_types_a(i32 %k) #0 {
entry:
  %var1 = alloca i64, align 4
  %var2 = alloca i32, align 4
  store i64 0, i64* %var1, align 4
  store i32 0, i32* %var2, align 4
  %cmp = icmp ne i32 %k, 0
  br i1 %cmp, label %if, label %then

if:
  %bitcast = bitcast i32* %var2 to i64*
  br label %then

then:
  %ptr = phi i64* [ %var1, %entry ], [ %bitcast, %if ]
  %ret = load i64, i64* %ptr, align 4
  ret i64 %ret
}

; CHECK-LABEL: phi_different_types_b:
; CHECK: __stack_chk_guard
define i64 @phi_different_types_b(i32 %k) #0 {
entry:
  %var1 = alloca i32, align 4
  %var2 = alloca i64, align 4
  store i32 0, i32* %var1, align 4
  store i64 0, i64* %var2, align 4
  %cmp = icmp ne i32 %k, 0
  br i1 %cmp, label %if, label %then

if:
  %bitcast = bitcast i32* %var1 to i64*
  br label %then

then:
  %ptr = phi i64* [ %var2, %entry ], [ %bitcast, %if ]
  %ret = load i64, i64* %ptr, align 4
  ret i64 %ret
}

; CHECK-LABEL: phi_after_gep_nonconstant_out_of_bounds_a:
; CHECK: __stack_chk_guard
define i32 @phi_after_gep_nonconstant_out_of_bounds_a(i32 %k, i32 %n) #0 {
entry:
  %var1 = alloca i32, align 4
  %var2 = alloca i32, align 4
  store i32 0, i32* %var1, align 4
  store i32 0, i32* %var2, align 4
  %cmp = icmp ne i32 %k, 0
  br i1 %cmp, label %if, label %else

if:
  %gep1 = getelementptr inbounds i32, i32* %var1, i32 0
  br label %then

else:
  %gep2 = getelementptr inbounds i32, i32* %var2, i32 %n
  br label %then

then:
  %ptr = phi i32* [ %gep1, %if ], [ %gep2, %else ]
  %ret = load i32, i32* %ptr, align 4
  ret i32 %ret
}

; CHECK-LABEL: phi_after_gep_nonconstant_out_of_bounds_b:
; CHECK: __stack_chk_guard
define i32 @phi_after_gep_nonconstant_out_of_bounds_b(i32 %k, i32 %n) #0 {
entry:
  %var1 = alloca i32, align 4
  %var2 = alloca i32, align 4
  store i32 0, i32* %var1, align 4
  store i32 0, i32* %var2, align 4
  %cmp = icmp ne i32 %k, 0
  br i1 %cmp, label %if, label %else

if:
  %gep1 = getelementptr inbounds i32, i32* %var1, i32 %n
  br label %then

else:
  %gep2 = getelementptr inbounds i32, i32* %var2, i32 0
  br label %then

then:
  %ptr = phi i32* [ %gep1, %if ], [ %gep2, %else ]
  %ret = load i32, i32* %ptr, align 4
  ret i32 %ret
}

%struct.outer = type { %struct.inner, %struct.inner }
%struct.inner = type { i32, i32 }

; CHECK-LABEL: struct_in_bounds:
; CHECK-NOT: __stack_chk_guard
define void @struct_in_bounds() #0 {
  %var = alloca %struct.outer, align 4
  %outergep = getelementptr inbounds %struct.outer, %struct.outer* %var, i32 0, i32 1
  %innergep = getelementptr inbounds %struct.inner, %struct.inner* %outergep, i32 0, i32 1
  store i32 0, i32* %innergep, align 4
  ret void
}

; CHECK-LABEL: struct_constant_out_of_bounds_a:
; CHECK: __stack_chk_guard
define void @struct_constant_out_of_bounds_a() #0 {
  %var = alloca %struct.outer, align 4
  %outergep = getelementptr inbounds %struct.outer, %struct.outer* %var, i32 1, i32 0
  %innergep = getelementptr inbounds %struct.inner, %struct.inner* %outergep, i32 0, i32 0
  store i32 0, i32* %innergep, align 4
  ret void
}

; CHECK-LABEL: struct_constant_out_of_bounds_b:
; Here the offset is out-of-bounds of the addressed struct.inner member, but
; still within bounds of the outer struct so no stack guard is needed.
; CHECK-NOT: __stack_chk_guard
define void @struct_constant_out_of_bounds_b() #0 {
  %var = alloca %struct.outer, align 4
  %outergep = getelementptr inbounds %struct.outer, %struct.outer* %var, i32 0, i32 0
  %innergep = getelementptr inbounds %struct.inner, %struct.inner* %outergep, i32 1, i32 0
  store i32 0, i32* %innergep, align 4
  ret void
}

; CHECK-LABEL: struct_constant_out_of_bounds_c:
; Here we are out-of-bounds of both the inner and outer struct.
; CHECK: __stack_chk_guard
define void @struct_constant_out_of_bounds_c() #0 {
  %var = alloca %struct.outer, align 4
  %outergep = getelementptr inbounds %struct.outer, %struct.outer* %var, i32 0, i32 1
  %innergep = getelementptr inbounds %struct.inner, %struct.inner* %outergep, i32 1, i32 0
  store i32 0, i32* %innergep, align 4
  ret void
}

; CHECK-LABEL: struct_nonconstant_out_of_bounds_a:
; CHECK: __stack_chk_guard
define void @struct_nonconstant_out_of_bounds_a(i32 %n) #0 {
  %var = alloca %struct.outer, align 4
  %outergep = getelementptr inbounds %struct.outer, %struct.outer* %var, i32 %n, i32 0
  %innergep = getelementptr inbounds %struct.inner, %struct.inner* %outergep, i32 0, i32 0
  store i32 0, i32* %innergep, align 4
  ret void
}

; CHECK-LABEL: struct_nonconstant_out_of_bounds_b:
; CHECK: __stack_chk_guard
define void @struct_nonconstant_out_of_bounds_b(i32 %n) #0 {
  %var = alloca %struct.outer, align 4
  %outergep = getelementptr inbounds %struct.outer, %struct.outer* %var, i32 0, i32 0
  %innergep = getelementptr inbounds %struct.inner, %struct.inner* %outergep, i32 %n, i32 0
  store i32 0, i32* %innergep, align 4
  ret void
}

; CHECK-LABEL: bitcast_smaller_load
; CHECK-NOT: __stack_chk_guard
define i32 @bitcast_smaller_load() #0 {
  %var = alloca i64, align 4
  store i64 0, i64* %var, align 4
  %bitcast = bitcast i64* %var to i32*
  %ret = load i32, i32* %bitcast, align 4
  ret i32 %ret
}

; CHECK-LABEL: bitcast_same_size_load
; CHECK-NOT: __stack_chk_guard
define i32 @bitcast_same_size_load() #0 {
  %var = alloca i64, align 4
  store i64 0, i64* %var, align 4
  %bitcast = bitcast i64* %var to %struct.inner*
  %gep = getelementptr inbounds %struct.inner, %struct.inner* %bitcast, i32 0, i32 1
  %ret = load i32, i32* %gep, align 4
  ret i32 %ret
}

; CHECK-LABEL: bitcast_larger_load
; CHECK: __stack_chk_guard
define i64 @bitcast_larger_load() #0 {
  %var = alloca i32, align 4
  store i32 0, i32* %var, align 4
  %bitcast = bitcast i32* %var to i64*
  %ret = load i64, i64* %bitcast, align 4
  ret i64 %ret
}

; CHECK-LABEL: bitcast_larger_store
; CHECK: __stack_chk_guard
define i32 @bitcast_larger_store() #0 {
  %var = alloca i32, align 4
  %bitcast = bitcast i32* %var to i64*
  store i64 0, i64* %bitcast, align 4
  %ret = load i32, i32* %var, align 4
  ret i32 %ret
}

; CHECK-LABEL: bitcast_larger_cmpxchg
; CHECK: __stack_chk_guard
define i64 @bitcast_larger_cmpxchg(i64 %desired, i64 %new) #0 {
  %var = alloca i32, align 4
  %bitcast = bitcast i32* %var to i64*
  %pair = cmpxchg i64* %bitcast, i64 %desired, i64 %new seq_cst monotonic
  %ret = extractvalue { i64, i1 } %pair, 0
  ret i64 %ret
}

; CHECK-LABEL: bitcast_larger_atomic_rmw
; CHECK: __stack_chk_guard
define i64 @bitcast_larger_atomic_rmw() #0 {
  %var = alloca i32, align 4
  %bitcast = bitcast i32* %var to i64*
  %ret = atomicrmw add i64* %bitcast, i64 1 monotonic
  ret i64 %ret
}

%struct.packed = type <{ i16, i32 }>

; CHECK-LABEL: bitcast_overlap
; CHECK: __stack_chk_guard
define i32 @bitcast_overlap() #0 {
  %var = alloca i32, align 4
  %bitcast = bitcast i32* %var to %struct.packed*
  %gep = getelementptr inbounds %struct.packed, %struct.packed* %bitcast, i32 0, i32 1
  %ret = load i32, i32* %gep, align 2
  ret i32 %ret
}

%struct.multi_dimensional = type { [10 x [10 x i32]], i32 }

; CHECK-LABEL: multi_dimensional_array
; CHECK: __stack_chk_guard
define i32 @multi_dimensional_array() #0 {
  %var = alloca %struct.multi_dimensional, align 4
  %gep1 = getelementptr inbounds %struct.multi_dimensional, %struct.multi_dimensional* %var, i32 0, i32 0
  %gep2 = getelementptr inbounds [10 x [10 x i32]], [10 x [10 x i32]]* %gep1, i32 0, i32 10
  %gep3 = getelementptr inbounds [10 x i32], [10 x i32]* %gep2, i32 0, i32 5
  %ret = load i32, i32* %gep3, align 4
  ret i32 %ret
}

attributes #0 = { sspstrong }
