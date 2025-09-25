; RUN: opt -passes='print<escape-analysis>' -disable-output %s 2>&1 | FileCheck %s

; NOTE:
; - The printer emits:
;   "EscapeAnalysis for function: <func>"
;   "<alloc-name> escapes: yes|no" per allocation site (alloca/malloc-like).
;   "EA: none" if no allocations in the function.
; - Names are taken from SSA. We avoid relying on "unnamed#N" in tests.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

@G = global ptr null
@GPtr = dso_local global ptr null, align 8
@GPtrPtr = dso_local global ptr null, align 8
@GPtrPtrPtr = dso_local global ptr null, align 8
@GAlias = alias ptr, ptr @GPtr

%S = type { ptr, ptr }
@GS = dso_local global %S zeroinitializer, align 8
@GArr = dso_local global [2 x %S] zeroinitializer, align 8

declare noalias ptr @malloc(i64)
declare noalias ptr @external(ptr)

; ============================================================================ ;
; Basics and locals
; ============================================================================ ;

; No allocations
define void @no_allocs() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'no_allocs':
; CHECK-NEXT: none
  ret void
}

; Using pointer in icmp -> no escape
define void @icmp_no_escape() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'icmp_no_escape':
; CHECK: a escapes: no
  %a = alloca i8, align 1
  %cmp = icmp eq ptr %a, null
  ret void
}

; Passthrough via phi/select-like -> no escape
define void @passthrough_phi_no_escape(i1 %c) {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'passthrough_phi_no_escape':
; CHECK: a escapes: no
entry:
  %a = alloca i8, align 1
  br i1 %c, label %t, label %f
t:
  br label %merge
f:
  br label %merge
merge:
  %p = phi ptr [ %a, %t ], [ %a, %f ]
  ret void
}

; Safe store to local memory -> no escape.
define void @store_to_local_ok() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'store_to_local_ok':
; CHECK: a escapes: no
; CHECK: p escapes: no
  %a = alloca i8, align 1
  %p = alloca ptr, align 8
  store ptr %a, ptr %p
  ret void
}

; Chain through local pointer (double indirection) remains local:
; %x is stored in %p, %p in %pp -> no escape.
define void @double_ptr_local_ok() sanitize_thread {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'double_ptr_local_ok':
; CHECK:  x escapes: no
; CHECK:  p escapes: no
; CHECK:  pp escapes: no
  %x  = alloca i32, align 4
  %p  = alloca ptr, align 8
  %pp = alloca ptr, align 8
  store ptr %x, ptr %p
  store ptr %p, ptr %pp
  store i32 1, ptr %x
  %lv = load i32, ptr %x
  ret void
}

; ============================================================================ ;
; Returns and heap allocations
; ============================================================================ ;

; Returning alloca pointer -> escape
define ptr @return_alloca_escape() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'return_alloca_escape':
; CHECK: a escapes: yes
  %a = alloca i8, align 1
  ret ptr %a
}

; Malloc-like allocations
define ptr @malloc_local_no_escape() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'malloc_local_no_escape':
; CHECK: m1 escapes: no
; CHECK: m2 escapes: yes
  %m1 = call ptr @malloc(i64 16)
  %m2 = call ptr @malloc(i64 32)
  ret ptr %m2
}

; Escape of malloc calls
define dso_local void @malloc_escape() #0 {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'malloc_escape':
; CHECK:   p escapes: no
; CHECK:   call escapes: yes
; CHECK:   call1 escapes: yes
entry:
  %p = alloca ptr, align 8
  %call = call noalias ptr @malloc(i64 noundef 4) #2
  store ptr %call, ptr @GPtr, align 8
  %call1 = call noalias ptr @malloc(i64 noundef 4) #2
  store ptr %call1, ptr %p, align 8
  %0 = load ptr, ptr %p, align 8
  store ptr %0, ptr @GPtr, align 8
  ret void
}

; Store to malloc'ed array element, which escapes -> local escapes
define dso_local void @escape_through_malloc() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'escape_through_malloc':
; CHECK:   a escapes: yes
; CHECK:   x escapes: yes
; CHECK:   b escapes: no
; CHECK:   y escapes: yes
; CHECK:   call escapes: yes
; CHECK:   call1 escapes: yes
entry:
  %a = alloca ptr, align 8
  %x = alloca i32, align 4
  %b = alloca ptr, align 8
  %y = alloca i32, align 4
  %call = call noalias ptr @malloc(i64 noundef 80)
  store ptr %call, ptr %a, align 8
  store ptr %a, ptr @GPtrPtrPtr, align 8
  %0 = load ptr, ptr %a, align 8
  %arrayidx = getelementptr inbounds ptr, ptr %0, i64 5
  store ptr %x, ptr %arrayidx, align 8
  %call1 = call noalias ptr @malloc(i64 noundef 80)
  store ptr %call1, ptr %b, align 8
  %1 = load ptr, ptr %b, align 8
  store ptr %1, ptr @GPtrPtr, align 8
  %2 = load ptr, ptr %b, align 8
  %arrayidx2 = getelementptr inbounds ptr, ptr %2, i64 5
  store ptr %y, ptr %arrayidx2, align 8
  ret void
}

; Store to malloc'ed array element, which does not escape -> no escape
define dso_local void @no_escape_through_malloc() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'no_escape_through_malloc':
; CHECK:   a escapes: no
; CHECK:   x escapes: no
; CHECK:   b escapes: no
; CHECK:   y escapes: no
; CHECK:   call escapes: no
; CHECK:   call1 escapes: no
entry:
  %a = alloca ptr, align 8
  %x = alloca i32, align 4
  %b = alloca ptr, align 8
  %y = alloca i32, align 4
  %call = call noalias ptr @malloc(i64 noundef 80)
  store ptr %call, ptr %a, align 8
  %0 = load ptr, ptr %a, align 8
  %arrayidx = getelementptr inbounds ptr, ptr %0, i64 5
  store ptr %x, ptr %arrayidx, align 8
  %call1 = call noalias ptr @malloc(i64 noundef 80)
  store ptr %call1, ptr %b, align 8
  %1 = load ptr, ptr %b, align 8
  %arrayidx2 = getelementptr inbounds ptr, ptr %1, i64 5
  store ptr %y, ptr %arrayidx2, align 8
  ret void
}

; ============================================================================ ;
; Globals, arguments, and mixed destinations
; ============================================================================ ;

; Store to global, global alias and global structure field -> escape
define void @store_to_global_escape() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'store_to_global_escape':
; CHECK: a escapes: yes
; CHECK: b escapes: yes
; CHECK: c escapes: yes
  %a = alloca i8, align 1
  %b = alloca i8, align 1
  %c = alloca i8, align 1
  store ptr %a, ptr @G
  store ptr %b, ptr @GAlias
  %f0 = getelementptr inbounds %S, ptr @GS, i64 0, i32 0
  store ptr %c, ptr %f0, align 8
  ret void
}

; Store to argument -> escape
define void @store_to_arg_escape(ptr %out) {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'store_to_arg_escape':
; CHECK: a escapes: yes
  %a = alloca i8, align 1
  store ptr %a, ptr %out
  ret void
}

; Store to the pointer returned by an external function -> escape
define void @store_to_unknown_ret_escape() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'store_to_unknown_ret_escape':
; CHECK: a escapes: yes
; CHECK: b escapes: yes
  %a = alloca i8, align 1
  %b = alloca i8, align 1
  %p = call ptr @external(ptr %b)
  store ptr %a, ptr %p
  ret void
}

; Cyclic dependency between local allocas, one stored to global -> both escape
define void @cycle_allocas_escape() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'cycle_allocas_escape':
; CHECK: a escapes: yes
; CHECK: b escapes: yes
  %a = alloca ptr, align 8
  %b = alloca ptr, align 8
  store ptr %a, ptr %b
  store ptr %b, ptr %a
  store ptr %a, ptr @G
  ret void
}

; Destination via phi mixing local and global -> escape
define void @phi_mixed_dest_escape(i1 %c) {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'phi_mixed_dest_escape':
; CHECK: a escapes: yes
; CHECK: p escapes: no
  %a = alloca i8, align 1
  %p = alloca ptr, align 8
  br i1 %c, label %t, label %f
t:
  br label %m
f:
  br label %m
m:
  %dst = phi ptr [ %p, %t ], [ @GPtr, %f ]
  store ptr %a, ptr %dst, align 8
  ret void
}

; Store to local pointer and then store from global pointer -> no escape
define dso_local void @store_ptr_store_from_global_no_escape() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'store_ptr_store_from_global_no_escape':
; CHECK: x escapes: no
; CHECK: p escapes: no
  %x = alloca i32, align 4
  %p = alloca ptr, align 8
  store ptr %x, ptr %p, align 8
  %1 = load ptr, ptr @GPtr, align 8
  store ptr %1, ptr %p, align 8
  ret void
}

declare void @varargs_func(ptr, ...)

; Passing pointer to varargs -> escape
define void @varargs_escape() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'varargs_escape':
; CHECK: a escapes: yes
  %a = alloca i32, align 4
  call void (ptr, ...) @varargs_func(ptr null, ptr %a)
  ret void
}

; ============================================================================ ;
; Loaded destination patterns
; ============================================================================ ;

; Store through pointer loaded from argument (LiveOnEntry) -> escape
define void @store_through_loaded_arg_escape(ptr %out) {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'store_through_loaded_arg_escape':
; CHECK: a escapes: yes
  %a = alloca i8, align 1
  %l = load ptr, ptr %out, align 8
  store ptr %a, ptr %l, align 8
  ret void
}

; Multiple stores to same location - last one escapes
define void @multiple_stores_last_escapes(i1 %c) {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'multiple_stores_last_escapes':
; CHECK: x1 escapes: no
; CHECK: x2 escapes: yes
; CHECK: p escapes: no
  %x1 = alloca i32, align 4
  %x2 = alloca i32, align 4
  %p = alloca ptr, align 8

  store ptr %x1, ptr %p, align 8
  store ptr %x2, ptr %p, align 8

  %loaded = load ptr, ptr %p, align 8
  store ptr %loaded, ptr @GPtr, align 8
  ret void
}

; Loaded destination with MemoryPhi (two stores) -> no escape
define void @loaded_dest_memphi_local_ok(i1 %c) {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'loaded_dest_memphi_local_ok':
; CHECK: x escapes: no
; CHECK: p escapes: no
; CHECK: s1 escapes: no
; CHECK: s2 escapes: no
entry:
  %x = alloca i32, align 4
  %p = alloca ptr, align 8
  %s1 = alloca ptr, align 8
  %s2 = alloca ptr, align 8
  br i1 %c, label %t, label %f
t:
  store ptr %s1, ptr %p, align 8
  br label %m
f:
  store ptr %s2, ptr %p, align 8
  br label %m
m:
  %l = load ptr, ptr %p, align 8
  store ptr %x, ptr %l, align 8
  ret void
}

; ============================================================================ ;
; Arrays and GEP
; ============================================================================ ;

; Store to local array element via GEP -> no escape
define void @store_to_gep_local_ok() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'store_to_gep_local_ok':
; CHECK: a escapes: no
; CHECK: arr escapes: no
  %a = alloca i8, align 1
  %arr = alloca [2 x ptr], align 8
  %elem = getelementptr inbounds [2 x ptr], ptr %arr, i64 0, i64 1
  store ptr %a, ptr %elem, align 8
  ret void
}

; Escape through heap array element: store pointer to local into malloc'ed array
; element, then read back and store to global -> local escapes; the malloc escapes.
define dso_local void @escape_through_heap_array_element() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'escape_through_heap_array_element':
; CHECK:  x escapes: yes
; CHECK:  p escapes: no
; CHECK:  call escapes: yes
entry:
  %x = alloca i32, align 4
  %p = alloca ptr, align 8
  %call = call noalias ptr @malloc(i64 noundef 800) #2
  store ptr %call, ptr %p, align 8
  %0 = load ptr, ptr %p, align 8
  %arrayidx = getelementptr inbounds ptr, ptr %0, i64 33
  store ptr %x, ptr %arrayidx, align 8
  %1 = load ptr, ptr %p, align 8
  %arrayidx1 = getelementptr inbounds ptr, ptr %1, i64 11
  %2 = load ptr, ptr %arrayidx1, align 8
  store ptr %2, ptr @GPtr, align 8
  ret void
}

; Escape through stack array element read from another index: local escapes,
; array remains local (element copied out to global).
define dso_local void @escape_through_array_element() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'escape_through_array_element':
; CHECK:   x escapes: yes
; CHECK:   p escapes: no
entry:
  %x = alloca i32, align 4
  %p = alloca [100 x ptr], align 16
  %arrayidx = getelementptr inbounds [100 x ptr], ptr %p, i64 0, i64 33
  store ptr %x, ptr %arrayidx, align 8
  %arrayidx1 = getelementptr inbounds [100 x ptr], ptr %p, i64 0, i64 0
  %0 = load ptr, ptr %arrayidx1, align 16
  store ptr %0, ptr @GPtr, align 8
  ret void
}

; Whole array (stack): leak address of an element itself -> the array escapes
define dso_local void @escape_whole_array() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'escape_whole_array':
; CHECK:   a1 escapes: yes
entry:
  %a1 = alloca [100 x i32], align 16
  %arrayidx = getelementptr inbounds [100 x i32], ptr %a1, i64 0, i64 33
  store ptr %arrayidx, ptr @GPtr, align 8
  ret void
}

; Whole array (heap): leak address of an element; also keep a local alloca with
; the malloc pointer to ensure both are reported.
define dso_local void @escape_whole_array_heap() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'escape_whole_array_heap':
; CHECK:   a2 escapes: yes
; CHECK:   call escapes: yes
entry:
  %a2 = alloca ptr, align 8
  %call = call noalias ptr @malloc(i64 noundef 400) #2
  store ptr %call, ptr %a2, align 8
  store ptr %a2, ptr @GPtrPtr, align 8
  ret void
}

; Struct (stack): leak address of a field -> the struct itself escapes
define void @struct_stack_self_escape_via_field_addr() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'struct_stack_self_escape_via_field_addr':
; CHECK: s escapes: yes
  %s = alloca %S, align 8
  %f0 = getelementptr inbounds %S, ptr %s, i64 0, i32 0
  store ptr %f0, ptr @GPtr, align 8
  ret void
}

; Struct (heap-esque via malloc): leak address of a field -> the heap object escapes
define void @struct_heap_self_escape_via_field_addr() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'struct_heap_self_escape_via_field_addr':
; CHECK: m escapes: yes
  %m = call ptr @malloc(i64 16)
  %f0 = getelementptr inbounds %S, ptr %m, i64 0, i32 0
  store ptr %f0, ptr @GPtr, align 8
  ret void
}

; Two-dimensional heap array pointer-contained structures:
; store local into an element, then array escapes -> local escapes; mallocs escape.
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'two_dimensional_array_escape':
; CHECK:   x escapes: yes
; CHECK:   arr escapes: no
; CHECK:   i escapes: no
; CHECK:   call escapes: yes
; CHECK:   call1 escapes: yes
define dso_local void @two_dimensional_array_escape() {
entry:
  %x = alloca i32, align 4
  %arr = alloca ptr, align 8
  %i = alloca i32, align 4
  %call = call noalias ptr @malloc(i64 noundef 80)
  store ptr %call, ptr %arr, align 8
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %0, 10
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %call1 = call noalias ptr @malloc(i64 noundef 160)
  %1 = load ptr, ptr %arr, align 8
  %2 = load i32, ptr %i, align 4
  %idxprom = sext i32 %2 to i64
  %arrayidx = getelementptr inbounds ptr, ptr %1, i64 %idxprom
  store ptr %call1, ptr %arrayidx, align 8
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %3 = load i32, ptr %i, align 4
  %inc = add nsw i32 %3, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %4 = load ptr, ptr %arr, align 8
  %arrayidx2 = getelementptr inbounds ptr, ptr %4, i64 5
  %5 = load ptr, ptr %arrayidx2, align 8
  %arrayidx3 = getelementptr inbounds %S, ptr %5, i64 5
  %f1 = getelementptr inbounds nuw %S, ptr %arrayidx3, i32 0, i32 0
  store ptr %x, ptr %f1, align 8
  %6 = load ptr, ptr %arr, align 8
  %arrayidx4 = getelementptr inbounds ptr, ptr %6, i64 0
  %7 = load ptr, ptr %arrayidx4, align 8
  %arrayidx5 = getelementptr inbounds %S, ptr %7, i64 0
  store ptr %arrayidx5, ptr @GPtr, align 8
  ret void
}

; ============================================================================ ;
; Structs and struct fields
; ============================================================================ ;

; Store into field of a local struct -> no escape
define void @struct_field_local_ok() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'struct_field_local_ok':
; CHECK: x escapes: no
; CHECK: s escapes: no
  %x = alloca i8, align 1
  %s = alloca %S, align 8
  %f0 = getelementptr inbounds %S, ptr %s, i64 0, i32 0
  store ptr %x, ptr %f0, align 8
  ret void
}

; Loaded-dest via field of a local struct -> no escape
define void @loaded_dest_struct_local_ok() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'loaded_dest_struct_local_ok':
; CHECK: x escapes: no
; CHECK: q escapes: no
; CHECK: s escapes: no
  %x = alloca i8, align 1
  %q = alloca ptr, align 8
  %s = alloca %S, align 8
  %f1 = getelementptr inbounds %S, ptr %s, i64 0, i32 1
  store ptr %q, ptr %f1, align 8
  %l = load ptr, ptr %f1, align 8
  store ptr %x, ptr %l, align 8
  ret void
}

; Local struct holds pointer to local; then the pointer is stored to global
; -> local escapes, struct remains local.
define void @struct_field_local_escape_via_global() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'struct_field_local_escape_via_global':
; CHECK: x escapes: yes
; CHECK: s escapes: no
  %x = alloca i8, align 1
  %s = alloca %S, align 8
  %f0 = getelementptr inbounds %S, ptr %s, i64 0, i32 0
  store ptr %x, ptr %f0, align 8
  %loaded = load ptr, ptr %f0, align 8
  store ptr %loaded, ptr @GPtr, align 8
  ret void
}

; Global struct stores the address of a local slot ->
; x escapes via escaping slot; q escapes
define void @loaded_dest_struct_global_escape() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'loaded_dest_struct_global_escape':
; CHECK: x escapes: yes
; CHECK: q escapes: yes
  %x = alloca i8, align 1
  %q = alloca ptr, align 8
  store ptr %q, ptr getelementptr inbounds (%S, ptr @GS, i32 0, i32 1), align 8
  %l = load ptr, ptr %q, align 8
  store ptr %x, ptr %l, align 8
  ret void
}

; Select between local and global struct as container -> escape
define void @select_struct_dest_mixed_escape(i1 %c) {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'select_struct_dest_mixed_escape':
; CHECK: x escapes: yes
; CHECK: s escapes: no
  %x = alloca i8, align 1
  %s = alloca %S, align 8
  %dst = select i1 %c, ptr %s, ptr @GS
  %f0 = getelementptr inbounds %S, ptr %dst, i64 0, i32 0
  store ptr %x, ptr %f0, align 8
  ret void
}

; Return a struct containing a pointer to a local -> escape
define %S @return_struct_containing_ptr_escape() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'return_struct_containing_ptr_escape':
; CHECK: x escapes: yes
  %x = alloca i8, align 1
  %u = insertvalue %S poison, ptr %x, 0
  %u2 = insertvalue %S %u, ptr null, 1
  ret %S %u2
}

; Deep hierarchy of nested structures
%struct.Parent = type { i32, %struct.Child }
%struct.Child = type { i32, i32, ptr }
%struct.GrandParent = type { i32, %struct.Parent }
define dso_local void @escape_through_nested_struct() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'escape_through_nested_struct':
; CHECK:  sp escapes: yes
; CHECK:  sg escapes: yes
; CHECK:  x escapes: yes
entry:
  %sp = alloca %struct.Parent, align 8
  %sg = alloca %struct.GrandParent, align 8
  %x = alloca i32, align 4
  %s = getelementptr inbounds nuw %struct.Parent, ptr %sp, i32 0, i32 1
  %a = getelementptr inbounds nuw %struct.Child, ptr %s, i32 0, i32 0
  store ptr %a, ptr @GPtr, align 8
  %sp1 = getelementptr inbounds nuw %struct.GrandParent, ptr %sg, i32 0, i32 1
  %s2 = getelementptr inbounds nuw %struct.Parent, ptr %sp1, i32 0, i32 1
  %b = getelementptr inbounds nuw %struct.Child, ptr %s2, i32 0, i32 1
  store ptr %b, ptr @GPtr, align 8
  %sp3 = getelementptr inbounds nuw %struct.GrandParent, ptr %sg, i32 0, i32 1
  %s4 = getelementptr inbounds nuw %struct.Parent, ptr %sp3, i32 0, i32 1
  %p = getelementptr inbounds nuw %struct.Child, ptr %s4, i32 0, i32 2
  store ptr %x, ptr %p, align 8
  ret void
}

; ============================================================================ ;
; Atomics and volatile
; ============================================================================ ;

; Atomic store of pointer -> treated as escape
define void @atomic_store_escape() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'atomic_store_escape':
; CHECK: a escapes: yes
; CHECK: p escapes: no
  %a = alloca i8, align 1
  %p = alloca ptr, align 8
  store atomic ptr %a, ptr %p seq_cst, align 8
  ret void
}

; Volatile store of pointer -> treated as escape
define void @volatile_store_escape() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'volatile_store_escape':
; CHECK: a escapes: yes
; CHECK: p escapes: no
  %a = alloca i8, align 1
  %p = alloca ptr, align 8
  store volatile ptr %a, ptr %p
  ret void
}

; ============================================================================ ;
; Casts
; ============================================================================ ;

; PtrToInt cast -> escape
define void @worklist_limit_bailout() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'worklist_limit_bailout':
; CHECK: a escapes: yes
  %a = alloca i8, align 1
  %c1 = icmp ne ptr %a, null
  %c2 = icmp eq ptr %a, null
  %sel = select i1 %c1, ptr %a, ptr %a
  %use = ptrtoint ptr %sel to i64
  ret void
}

; ============================================================================ ;
; Escape through double pointers
; ============================================================================ ;

; Store pp to global triple pointer -> all three allocations escape
define dso_local void @esc_thorugh_double_ptr1() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'esc_thorugh_double_ptr1':
; CHECK:  x escapes: yes
; CHECK:  p escapes: yes
; CHECK:  pp escapes: yes
entry:
  %x = alloca i32, align 4
  %p = alloca ptr, align 8
  %pp = alloca ptr, align 8
  store ptr %x, ptr %p, align 8
  store ptr %p, ptr %pp, align 8
  store ptr %pp, ptr @GPtrPtrPtr, align 8
  ret void
}

; Store p (loaded from pp) to global double pointer
; -> x and p escape, pp stays local
define dso_local void @esc_thorugh_double_ptr2() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'esc_thorugh_double_ptr2':
; CHECK:  x escapes: yes
; CHECK:  p escapes: yes
; CHECK:  pp escapes: no
entry:
  %x = alloca i32, align 4
  %p = alloca ptr, align 8
  %pp = alloca ptr, align 8
  store ptr %x, ptr %p, align 8
  store ptr %p, ptr %pp, align 8
  %0 = load ptr, ptr %pp, align 8
  store ptr %0, ptr @GPtrPtr, align 8
  ret void
}

; ============================================================================ ;
; Loops
; ============================================================================ ;

; Store pointer in a loop -> escape detection through loop iterations
define void @store_in_loop_escape(i32 %n) {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'store_in_loop_escape':
; CHECK: x escapes: yes
; CHECK: arr escapes: no
entry:
  %x = alloca i32, align 4
  %arr = alloca [10 x ptr], align 8
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %gep = getelementptr inbounds [10 x ptr], ptr %arr, i64 0, i32 %i
  store ptr %x, ptr %gep, align 8
  %next = add i32 %i, 1
  %cond = icmp slt i32 %next, %n
  br i1 %cond, label %loop, label %exit

exit:
  %load_gep = getelementptr inbounds [10 x ptr], ptr %arr, i64 0, i32 5
  %loaded = load ptr, ptr %load_gep, align 8
  store ptr %loaded, ptr @GPtr, align 8
  ret void
}

; ============================================================================ ;
; Calls with nocapture arguments (e.g. memory intrinsics)
; ============================================================================ ;

declare void @memintr_like_func(ptr noalias readonly captures(none))
declare void @memintr_like_func_writeonly(ptr noalias readonly captures(none)) writeonly
declare void @memintr_like_func_arg_writeonly(ptr noalias writeonly captures(none))
declare void @memintr_like_func_readonly(ptr noalias readonly captures(none)) readonly

; Object stored into array, which is passed to a nocapture argument -> escape
define dso_local void @pass_nocapture_arg() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'pass_nocapture_arg':
; Case 0: store followed by nocapture argument, but writeonly -> no escape
; CHECK:  x01 escapes: no
; CHECK:  arr01 escapes: no
; CHECK:  x02 escapes: no
; CHECK:  arr02 escapes: no
; Case 1: stored pointer is read by a nocapture-like call -> escape
; CHECK:  x1 escapes: yes
; CHECK:  arr1 escapes: no
; Case 2: another independent case -> escape
; CHECK:  x2 escapes: yes
; CHECK:  arr2 escapes: no
; Overwrite case below: first store is overwritten before the call -> x3 does not escape, x4 escapes
; CHECK:  x3 escapes: no
; CHECK:  x4 escapes: yes
; CHECK:  arr3 escapes: no
entry:
  ; Case 0-1: store followed by nocapture argument, but the function is writeonly -> no escape
  %x01 = alloca i32, align 4
  %arr01 = alloca [10 x ptr], align 16
  %arrayidx0 = getelementptr inbounds [10 x ptr], ptr %arr01, i64 0, i64 5
  store ptr %x01, ptr %arrayidx0, align 8
  %arraydecay0 = getelementptr inbounds [10 x ptr], ptr %arr01, i64 0, i64 0
  call void @memintr_like_func_writeonly(ptr align 16 %arraydecay0)

  ; Case 0-2: store followed by nocapture argument,
  ; but the function argument is writeonly -> no escape
  %x02 = alloca i32, align 4
  %arr02 = alloca [10 x ptr], align 16
  %arrayidx02 = getelementptr inbounds [10 x ptr], ptr %arr02, i64 0, i64 5
  store ptr %x02, ptr %arrayidx02, align 8
  %arraydecay02 = getelementptr inbounds [10 x ptr], ptr %arr02, i64 0, i64 0
  call void @memintr_like_func_arg_writeonly(ptr align 16 %arraydecay02)

  ; Case 1: stored pointer is read by a nocapture-like call -> escape
  %x1 = alloca i32, align 4
  %arr1 = alloca [10 x ptr], align 16
  %arrayidx1 = getelementptr inbounds [10 x ptr], ptr %arr1, i64 0, i64 5
  store ptr %x1, ptr %arrayidx1, align 8
  %arraydecay1 = getelementptr inbounds [10 x ptr], ptr %arr1, i64 0, i64 0
  call void @memintr_like_func(ptr align 16 %arraydecay1)

  ; Case 2: another independent case -> escape
  %x2 = alloca i32, align 4
  %arr2 = alloca [10 x ptr], align 16
  %arrayidx2 = getelementptr inbounds [10 x ptr], ptr %arr2, i64 0, i64 5
  store ptr %x2, ptr %arrayidx2, align 8
  %arraydecay2 = getelementptr inbounds [10 x ptr], ptr %arr2, i64 0, i64 0
  call void @memintr_like_func_readonly(ptr align 16 %arraydecay2)

  ; Case 3: Overwrite before the nocapture call:
  ; first store (%x3) is overwritten by second store (%x4) before the call.
  ; Expected: x3 does not escape; x4 escapes; the array itself does not escape.
  %x3 = alloca i32, align 4
  %x4 = alloca i32, align 4
  %arr3 = alloca [10 x ptr], align 16
  %slot3 = getelementptr inbounds [10 x ptr], ptr %arr3, i64 0, i64 5
  store ptr %x3, ptr %slot3, align 8
  store ptr %x4, ptr %slot3, align 8
  %decay3 = getelementptr inbounds [10 x ptr], ptr %arr3, i64 0, i64 0
  call void @memintr_like_func(ptr align 16 %decay3)

  ret void
}

; Calls with nocapture arguments (heap): passing a heap pointer itself to a nocapture
; consumer does NOT make the heap allocation escape; however, objects stored inside
; that heap memory may escape. Includes overwrite cases for both heap pointers and x's.
define dso_local void @pass_nocapture_arg_heap() {
; CHECK-LABEL: Printing analysis 'Escape Analysis' for function 'pass_nocapture_arg_heap':
; Case 1: Store local x into heap memory, then nocapture reads -> x escapes
; CHECK:  x1 escapes: yes
; CHECK:  arr1 escapes: no
; CHECK:  call1 escapes: no
; Case 2: Readonly variant -> x still escapes
; CHECK:  x2 escapes: yes
; CHECK:  arr2 escapes: no
; CHECK:  call2 escapes: no
; Case 3: Overwrite scenario for x on heap:
; CHECK:  x3 escapes: no
; CHECK:  x4 escapes: yes
; CHECK:  arr3 escapes: no
; CHECK:  call3 escapes: no
entry:
  ; Case 1: Store local x into heap memory, then nocapture reads -> x escapes
  %x1 = alloca i32, align 4
  %arr1 = alloca ptr, align 8
  %call1 = call noalias ptr @malloc(i64 noundef 80)
  store ptr %call1, ptr %arr1, align 8
  %hp1 = load ptr, ptr %arr1, align 8
  %slot1 = getelementptr inbounds ptr, ptr %hp1, i64 5
  store ptr %x1, ptr %slot1, align 8
  %base1 = load ptr, ptr %arr1, align 8
  call void @memintr_like_func(ptr align 8 %base1)

  ; Case 2: Readonly variant -> x still escapes
  %x2 = alloca i32, align 4
  %arr2 = alloca ptr, align 8
  %call2 = call noalias ptr @malloc(i64 noundef 80)
  store ptr %call2, ptr %arr2, align 8
  %hp2 = load ptr, ptr %arr2, align 8
  %slot2 = getelementptr inbounds ptr, ptr %hp2, i64 5
  store ptr %x2, ptr %slot2, align 8
  %base2 = load ptr, ptr %arr2, align 8
  call void @memintr_like_func_readonly(ptr align 8 %base2)

  ; Case 3: Overwrite scenario for x on heap:
  %x3 = alloca i32, align 4
  %x4 = alloca i32, align 4
  %arr3 = alloca ptr, align 8
  %call3 = call noalias ptr @malloc(i64 noundef 80)
  store ptr %call3, ptr %arr3, align 8
  %hp3 = load ptr, ptr %arr3, align 8
  %slot3 = getelementptr inbounds ptr, ptr %hp3, i64 5
  store ptr %x3, ptr %slot3, align 8
  store ptr %x4, ptr %slot3, align 8
  %base3 = load ptr, ptr %arr3, align 8
  call void @memintr_like_func(ptr align 8 %base3)

  ret void
}

