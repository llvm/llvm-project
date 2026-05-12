; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-enable-object-linking -filetype=obj < %s | llvm-readobj -r - | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-enable-object-linking -filetype=asm < %s | FileCheck %s --check-prefix=ASM --implicit-check-not=.amdgpu_num_agpr

; Test ABI register-size type ID generation for various function types.
; The type ID encodes each parameter/return by bit width: v=void, i=<=32-bit,
; l=33-64-bit, and >64-bit types widen to ceil(bits/32) x "i". Types with the
; same register footprint share an encoding (e.g. float(float) and i32(i32)
; both produce "ii"). Coverage here spans scalars, vectors (whose size is the
; total element bit width), pointers across address spaces (AS 1 is 64-bit on
; amdhsa; AS 3 / AS 5 are 32-bit), and small integer types (i1/i8/i16) that
; are not natively passed as function arguments but ABI-promoted to i32 slots
; -- they still encode as "i", matching an i32 parameter.
;
; Cross-TU coverage: an address-taken declaration (defined in another TU) still
; gets an .amdgpu_info scope with .amdgpu_typeid, but no per-function resource
; counts since its body isn't available here.
declare void @extern_decl(i32)

; Zero-sized aggregate types are valid IR function argument/return types. They
; consume no ABI registers, so arguments are omitted from the type ID and returns
; use the same no-result marker as void.
declare void @empty_struct_arg({})
declare void @empty_array_i32_arg([0 x i8], i32)
declare void @empty_array_i64_i32_arg([0 x i8], i64, i32)
declare {} @empty_struct_ret()

define void @void_void() {
  ret void
}

define i32 @i32_i32(i32 %x) {
  ret i32 %x
}

define void @void_ptr_i32(ptr %p, i32 %x) {
  ret void
}

define i64 @i64_i64_i64(i64 %a, i64 %b) {
  ret i64 %a
}

define float @float_float(float %x) {
  ret float %x
}

; Address-space pointer widths: AS 1 (global) is 64-bit -> "l"; AS 3 (LDS) and
; AS 5 (private) are 32-bit -> "i".
define void @ptr_addrspaces(ptr addrspace(1) %g, ptr addrspace(3) %l, ptr addrspace(5) %p) {
  ret void
}

; Vector types: encoded by total bit width. <2 x i32> = 64 bits -> "l";
; <4 x i32>/<4 x float>/<2 x i64> = 128 bits -> "iiii" each.
define <4 x i32> @vectors(<2 x i32> %a, <4 x float> %b, <2 x i64> %c) {
  ret <4 x i32> zeroinitializer
}

; Small integer types (i1/i8/i16) are ABI-promoted to i32 register slots on
; AMDGPU. They all collapse to "i" under the bit-width scheme, matching an i32
; parameter, so callers declared as void(i32, ...) remain compatible with
; callees taking void(i8, ...). signext/zeroext attributes describe the
; promotion mode and do not affect the encoding.
define void @promoted_small_ints(i8 signext %a, i16 zeroext %b, i1 %c) {
  ret void
}

; Wider non-vector scalars: double is "l" (64 bits); i128 widens to 4 x "i"
; (ceil(128/32)).
define double @wide_scalars(double %a, i128 %b) {
  ret double %a
}

%Struct16 = type { i32, i32, i32, i32 }

; byval / byref struct pointer parameters encode as a single pointer register
; slot, the same as a plain pointer in the same address space. byval describes
; a caller-side stack copy and byref describes a pointer handed through
; unchanged; neither changes the callee's register footprint (one pointer),
; so both collapse to "i" (for 32-bit AS) or "l" (for 64-bit AS). Compare
; with the plain-pointer encodings in @ptr_addrspaces.
define void @byval_struct_private(ptr addrspace(5) byval(%Struct16) %p) {
  ret void
}

define void @byref_struct_constant(ptr addrspace(4) byref(%Struct16) %p) {
  ret void
}

; Indirect-call type IDs are derived from the call instruction's FunctionType
; using the same rules, so they match the .amdgpu_typeid of an ABI-compatible
; address-taken callee. Duplicate signatures within one function are
; deduplicated (the second void() call below shares "v" with the first and
; yields only one .amdgpu_indirect_call entry; likewise a plain
; ptr addrspace(5) call and a ptr addrspace(5) byval(...) call both encode as
; "vi" and collapse to one entry). Zero-sized arguments consume no ABI registers,
; so the empty-array/i64/i32 call encodes as "vli".
define void @icaller(ptr %f_void, ptr %f_ptrs, ptr %f_vec, ptr %f_small, ptr %f_wide, ptr %f_dup, ptr %f_priv, ptr %f_priv_byval, ptr %f_const, ptr %f_const_byref, ptr %f_empty_struct, ptr %f_empty_array_i32, ptr %f_empty_array_i64_i32, ptr %f_empty_ret) {
  call void %f_void()
  call void %f_ptrs(ptr addrspace(1) null, ptr addrspace(3) null, ptr addrspace(5) null)
  %v = call <4 x i32> %f_vec(<2 x i32> zeroinitializer, <4 x float> zeroinitializer, <2 x i64> zeroinitializer)
  call void %f_small(i8 signext 0, i16 zeroext 0, i1 false)
  %d = call double %f_wide(double 0.0, i128 0)
  call void %f_dup()
  call void %f_priv(ptr addrspace(5) null)
  call void %f_priv_byval(ptr addrspace(5) byval(%Struct16) null)
  call void %f_const(ptr addrspace(4) null)
  call void %f_const_byref(ptr addrspace(4) byref(%Struct16) null)
  call void %f_empty_struct({} zeroinitializer)
  call void %f_empty_array_i32([0 x i8] zeroinitializer, i32 0)
  call void %f_empty_array_i64_i32([0 x i8] zeroinitializer, i64 0, i32 0)
  %empty_ret = call {} %f_empty_ret()
  ret void
}

; Take the address of each function so they appear as resource nodes.
define void @taker() {
  %p0 = alloca ptr, addrspace(5)
  store volatile ptr @void_void, ptr addrspace(5) %p0
  store volatile ptr @i32_i32, ptr addrspace(5) %p0
  store volatile ptr @void_ptr_i32, ptr addrspace(5) %p0
  store volatile ptr @i64_i64_i64, ptr addrspace(5) %p0
  store volatile ptr @float_float, ptr addrspace(5) %p0
  store volatile ptr @ptr_addrspaces, ptr addrspace(5) %p0
  store volatile ptr @vectors, ptr addrspace(5) %p0
  store volatile ptr @promoted_small_ints, ptr addrspace(5) %p0
  store volatile ptr @wide_scalars, ptr addrspace(5) %p0
  store volatile ptr @byval_struct_private, ptr addrspace(5) %p0
  store volatile ptr @byref_struct_constant, ptr addrspace(5) %p0
  store volatile ptr @extern_decl, ptr addrspace(5) %p0
  store volatile ptr @empty_struct_arg, ptr addrspace(5) %p0
  store volatile ptr @empty_array_i32_arg, ptr addrspace(5) %p0
  store volatile ptr @empty_array_i64_i32_arg, ptr addrspace(5) %p0
  store volatile ptr @empty_struct_ret, ptr addrspace(5) %p0
  ret void
}

define amdgpu_kernel void @kern() {
  call void @taker()
  call void @icaller(ptr @void_void, ptr @ptr_addrspaces, ptr @vectors,
                     ptr @promoted_small_ints, ptr @wide_scalars,
                     ptr @void_void,
                     ptr @byval_struct_private, ptr @byval_struct_private,
                     ptr @byref_struct_constant, ptr @byref_struct_constant,
                     ptr @empty_struct_arg, ptr @empty_array_i32_arg,
                     ptr @empty_array_i64_i32_arg, ptr @empty_struct_ret)
  ret void
}

; CHECK-DAG: R_AMDGPU_ABS64 void_void
; CHECK-DAG: R_AMDGPU_ABS64 i32_i32
; CHECK-DAG: R_AMDGPU_ABS64 void_ptr_i32
; CHECK-DAG: R_AMDGPU_ABS64 i64_i64_i64
; CHECK-DAG: R_AMDGPU_ABS64 float_float
; CHECK-DAG: R_AMDGPU_ABS64 ptr_addrspaces
; CHECK-DAG: R_AMDGPU_ABS64 vectors
; CHECK-DAG: R_AMDGPU_ABS64 promoted_small_ints
; CHECK-DAG: R_AMDGPU_ABS64 wide_scalars
; CHECK-DAG: R_AMDGPU_ABS64 byval_struct_private
; CHECK-DAG: R_AMDGPU_ABS64 byref_struct_constant
; CHECK-DAG: R_AMDGPU_ABS64 extern_decl
; CHECK-DAG: R_AMDGPU_ABS64 empty_struct_arg
; CHECK-DAG: R_AMDGPU_ABS64 empty_array_i32_arg
; CHECK-DAG: R_AMDGPU_ABS64 empty_array_i64_i32_arg
; CHECK-DAG: R_AMDGPU_ABS64 empty_struct_ret
; CHECK-DAG: R_AMDGPU_ABS64 icaller
; CHECK-DAG: R_AMDGPU_ABS64 taker
; CHECK-DAG: R_AMDGPU_ABS64 kern

; ASM-DAG:    .amdgpu_info void_void
; ASM-DAG:      .amdgpu_flags 0
; ASM-DAG:      .amdgpu_typeid "v"
; ASM-DAG:    .end_amdgpu_info
; ASM-DAG:    .amdgpu_info i32_i32
; ASM-DAG:      .amdgpu_flags 0
; ASM-DAG:      .amdgpu_typeid "ii"
; ASM-DAG:    .end_amdgpu_info
; ASM-DAG:    .amdgpu_info void_ptr_i32
; ASM-DAG:      .amdgpu_flags 0
; ASM-DAG:      .amdgpu_typeid "vli"
; ASM-DAG:    .end_amdgpu_info
; ASM-DAG:    .amdgpu_info i64_i64_i64
; ASM-DAG:      .amdgpu_flags 0
; ASM-DAG:      .amdgpu_typeid "lll"
; ASM-DAG:    .end_amdgpu_info
; ASM-DAG:    .amdgpu_info float_float
; ASM-DAG:      .amdgpu_flags 0
; ASM-DAG:      .amdgpu_typeid "ii"
; ASM-DAG:    .end_amdgpu_info
; ASM-DAG:    .amdgpu_info ptr_addrspaces
; ASM-DAG:      .amdgpu_flags 0
; ASM-DAG:      .amdgpu_typeid "vlii"
; ASM-DAG:    .end_amdgpu_info
; ASM-DAG:    .amdgpu_info vectors
; ASM-DAG:      .amdgpu_flags 0
; ASM-DAG:      .amdgpu_typeid "iiiiliiiiiiii"
; ASM-DAG:    .end_amdgpu_info
; ASM-DAG:    .amdgpu_info promoted_small_ints
; ASM-DAG:      .amdgpu_flags 0
; ASM-DAG:      .amdgpu_typeid "viii"
; ASM-DAG:    .end_amdgpu_info
; ASM-DAG:    .amdgpu_info wide_scalars
; ASM-DAG:      .amdgpu_flags 0
; ASM-DAG:      .amdgpu_typeid "lliiii"
; ASM-DAG:    .end_amdgpu_info
; ASM-DAG:    .amdgpu_info byval_struct_private
; ASM-DAG:      .amdgpu_flags 0
; ASM-DAG:      .amdgpu_typeid "vi"
; ASM-DAG:    .end_amdgpu_info
; ASM-DAG:    .amdgpu_info byref_struct_constant
; ASM-DAG:      .amdgpu_flags 0
; ASM-DAG:      .amdgpu_typeid "vl"
; ASM-DAG:    .end_amdgpu_info
; COM: Address-taken declaration: only the type-ID appears in its scope, with
; COM: no per-function resource counts (the body lives in another TU).
; ASM-DAG:    .amdgpu_info extern_decl
; ASM-DAG:      .amdgpu_typeid "vi"
; ASM-DAG:    .end_amdgpu_info
; ASM-DAG:    .amdgpu_info empty_struct_arg
; ASM-DAG:      .amdgpu_typeid "v"
; ASM-DAG:    .end_amdgpu_info
; ASM-DAG:    .amdgpu_info empty_array_i32_arg
; ASM-DAG:      .amdgpu_typeid "vi"
; ASM-DAG:    .end_amdgpu_info
; ASM-DAG:    .amdgpu_info empty_array_i64_i32_arg
; ASM-DAG:      .amdgpu_typeid "vli"
; ASM-DAG:    .end_amdgpu_info
; ASM-DAG:    .amdgpu_info empty_struct_ret
; ASM-DAG:      .amdgpu_typeid "v"
; ASM-DAG:    .end_amdgpu_info
; COM: @icaller's indirect call type IDs mirror the @void_void / @ptr_addrspaces /
; COM: @vectors / @promoted_small_ints / @wide_scalars .amdgpu_typeid encodings
; COM: above, proving ABI compatibility. The duplicate void() indirect call is
; COM: deduplicated, so "v" appears only once. The byval/byref pairs dedupe
; COM: with their plain-pointer counterparts: "vi" (AS 5) and "vl" (AS 4) each
; COM: appear once despite two call sites apiece.
; ASM-DAG:    .amdgpu_info icaller
; ASM-DAG:      .amdgpu_flags 1
; ASM-DAG:      .amdgpu_indirect_call "v"
; ASM-DAG:      .amdgpu_indirect_call "vlii"
; ASM-DAG:      .amdgpu_indirect_call "iiiiliiiiiiii"
; ASM-DAG:      .amdgpu_indirect_call "viii"
; ASM-DAG:      .amdgpu_indirect_call "lliiii"
; ASM-DAG:      .amdgpu_indirect_call "vi"
; ASM-DAG:      .amdgpu_indirect_call "vl"
; ASM-DAG:      .amdgpu_indirect_call "vli"
; ASM-DAG:    .end_amdgpu_info
; ASM-DAG:    .amdgpu_info taker
; ASM-DAG:      .amdgpu_flags 0
; ASM-DAG:      .amdgpu_num_vgpr {{[0-9]+}}
; ASM-DAG:    .end_amdgpu_info
; COM: The kernel scope is present but carries no type IDs of its own (kernels
; COM: aren't indirect-call targets). Direct-call edges from the kernel body are
; COM: exercised separately in the callgraph test.
; ASM-DAG:    .amdgpu_info kern
; ASM-DAG:      .amdgpu_flags {{[0-9]+}}
; ASM-DAG:    .end_amdgpu_info
