; RUN: llubi --verbose < %s 2>&1 | FileCheck %s

define range(i32 0, 2) i32 @add_with_range(i32 range(i32 0, 2) %x) {
  %add = add i32 %x, 1
  ret i32 %add
}

define range(i32 0, 2) <4 x i32> @add_with_range_vec(<4 x i32> range(i32 0, 2) %x) {
  %add = add <4 x i32> %x, splat(i32 1)
  ret <4 x i32> %add
}

define nofpclass(nan) half @identity_nofpclass(half nofpclass(inf) %x) {
  ret half %x
}

define nofpclass(nan) <4 x half> @identity_nofpclass_vec(<4 x half> nofpclass(inf) %x) {
  ret <4 x half> %x
}

define nofpclass(nan) {<2 x half>, <2 x half>} @identity_nofpclass_agg({<2 x half>, <2 x half>} nofpclass(inf) %x) {
  ret {<2 x half>, <2 x half>} %x
}

define nonnull ptr @gep_nonnull(ptr nonnull %p) {
  %gep = getelementptr i8, ptr %p, i32 -1
  ret ptr %gep
}

define ptr @gep(ptr %p) {
  %gep = getelementptr i8, ptr %p, i32 -1
  ret ptr %gep
}

define align 16 ptr @gep_align(ptr align 8 %p) {
  %gep = getelementptr i8, ptr %p, i32 8
  ret ptr %gep
}

define align 16 <4 x ptr> @gep_align_vec(<4 x ptr> align 8 %p) {
  %gep = getelementptr i8, <4 x ptr> %p, i32 8
  ret <4 x ptr> %gep
}

define noundef i32 @identity_noundef(i32 noundef %x) {
  ret i32 %x
}

define noundef {i32, <2 x i32>, [2 x i32]} @identity_noundef_agg({i32, <2 x i32>, [2 x i32]} noundef %x) {
  ret {i32, <2 x i32>, [2 x i32]} %x
}

define noundef dereferenceable(4) ptr @identity_dereferenceable(ptr noundef dereferenceable(4) %p) {
  ret ptr %p
}

define noundef dereferenceable(1) ptr @identity_dereferenceable_single_byte(ptr noundef dereferenceable(1) %p) {
  ret ptr %p
}

define noundef dereferenceable_or_null(1) ptr @identity_dereferenceable_or_null(ptr noundef dereferenceable_or_null(1) %p) {
  ret ptr %p
}

define void @main() {
  %range_valid = call i32 @add_with_range(i32 0)
  %range_poison_input = call i32 @add_with_range(i32 poison)
  %range_invalid_input = call i32 @add_with_range(i32 3)
  %range_invalid_output = call i32 @add_with_range(i32 1)
  %range_vec = call <4 x i32> @add_with_range_vec(<4 x i32> <i32 0, i32 poison, i32 3, i32 1>)
  %range_intrinsic_valid = call i32 @llvm.ctpop.i32(i32 range(i32 1, 255) 15)
  %range_intrinsic_invalid_input = call i32 @llvm.ctpop.i32(i32 range(i32 1, 255) 1500)
  %range_intrinsic_invalid_output = call range(i32 1, 32) i32 @llvm.ctpop.i32(i32 0)
  %range_intrinsic_vec = call range(i32 1, 32) <4 x i32> @llvm.ctpop.v4i32(<4 x i32> range(i32 1, 255) <i32 15, i32 1500, i32 0, i32 poison>)
  
  %nofpclass_valid = call half @identity_nofpclass(half 1.0)
  %nofpclass_poison_input = call half @identity_nofpclass(half poison)
  %nofpclass_invalid_input = call half @identity_nofpclass(half 0xH7C00)
  %nofpclass_invalid_output = call half @identity_nofpclass(half 0xH7E00)
  %nofpclass_vec = call <4 x half> @identity_nofpclass_vec(<4 x half> <half 1.0, half poison, half 0xH7C00, half 0xH7E00>)
  %nofpclass_callsite_invalid_input = call half @identity_nofpclass(half nofpclass(norm) 1.0)
  %nofpclass_callsite_invalid_output = call nofpclass(norm) half @identity_nofpclass(half 1.0)
  %nofpclass_agg = call {<2 x half>, <2 x half>} @identity_nofpclass_agg({<2 x half>, <2 x half>} {<2 x half> <half 1.0, half poison>, <2 x half> <half 0xH7C00, half 0xH7E00>})

  %alloc = alloca i32
  %ptr_one = getelementptr i8, ptr null, i32 1
  %nonnull_valid = call ptr @gep_nonnull(ptr %alloc)
  %nonnull_invalid_input = call ptr @gep_nonnull(ptr null)
  %nonnull_invalid_output = call ptr @gep_nonnull(ptr %ptr_one)
  %nonnull_callsite_valid = call nonnull ptr @gep(ptr nonnull %alloc)
  %nonnull_callsite_invalid_input = call ptr @gep(ptr nonnull null)
  %nonnull_callsite_invalid_output = call nonnull ptr @gep(ptr %ptr_one)

  %align_valid = call ptr @gep_align(ptr %alloc)
  %align_invalid_input = call ptr @gep_align(ptr %ptr_one)
  %align_invalid_output = call ptr @gep_align(ptr null)
  %ptr_vec_1 = insertelement <4 x ptr> poison, ptr %alloc, i32 0
  %ptr_vec_2 = insertelement <4 x ptr> %ptr_vec_1, ptr %ptr_one, i32 1
  %ptr_vec_3 = insertelement <4 x ptr> %ptr_vec_2, ptr null, i32 2
  %align_vec = call <4 x ptr> @gep_align_vec(<4 x ptr> %ptr_vec_3)
  %align_valid_mixed = call align 1 ptr @gep_align(ptr align 4 %alloc)
  %align_invalid_mixed1 = call ptr @gep_align(ptr align 1024 %alloc)
  %align_invalid_mixed2 = call align 1024 ptr @gep_align(ptr %alloc)

  %noundef_valid = call noundef i32 @identity_noundef(i32 noundef 1)
  %noundef_valid_agg = call noundef {i32, <2 x i32>, [2 x i32]} @identity_noundef_agg({i32, <2 x i32>, [2 x i32]} noundef zeroinitializer)
  
  %deref_valid = call ptr @identity_dereferenceable(ptr %alloc)
  %deref_valid_mixed = call dereferenceable(1) ptr @identity_dereferenceable(ptr dereferenceable(1) %alloc)
  %gep = getelementptr i8, ptr %alloc, i32 3
  %deref_valid_middle = call ptr @identity_dereferenceable_single_byte(ptr %gep)
  %deref_or_null_valid1 = call ptr @identity_dereferenceable_or_null(ptr %alloc)
  %deref_or_null_valid2 = call ptr @identity_dereferenceable_or_null(ptr null)
  %deref_or_null_mixed = call dereferenceable_or_null(1) dereferenceable(1) ptr @identity_dereferenceable_or_null(ptr dereferenceable_or_null(1) dereferenceable(1) %alloc)
  ret void
}
