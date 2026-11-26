; RUN: not llubi --verbose --entry-function=alloca_load < %s 2>&1 | FileCheck %s --check-prefix=ALLOCA-LOAD
; RUN: not llubi --verbose --entry-function=alloca_store < %s 2>&1 | FileCheck %s --check-prefix=ALLOCA-STORE
; RUN: not llubi --verbose --entry-function=musttail_alloca_load < %s 2>&1 | FileCheck %s --check-prefix=MUSTTAIL-LOAD
; RUN: not llubi --verbose --entry-function=byval_passed_as_ptr < %s 2>&1 | FileCheck %s --check-prefix=BYVAL-PTR
; RUN: not llubi --verbose --entry-function=global_escape_transitive < %s 2>&1 | FileCheck %s --check-prefix=TRANSITIVE
; RUN: not llubi --verbose --entry-function=dereferenceable_arg < %s 2>&1 | FileCheck %s --check-prefix=DEREF
; RUN: not llubi --verbose --entry-function=intrinsic_memcpy < %s 2>&1 | FileCheck %s --check-prefix=MEMCPY
; RUN: not llubi --verbose --entry-function=intrinsic_lifetime_start < %s 2>&1 | FileCheck %s --check-prefix=LIFETIME-START
; RUN: not llubi --verbose --entry-function=intrinsic_lifetime_end < %s 2>&1 | FileCheck %s --check-prefix=LIFETIME-END

@g = global i32 1
@gp = global ptr null

define i32 @read(ptr %p) {
  %v = load i32, ptr %p
  ret i32 %v
}

define void @write(ptr %p) {
  store i32 0, ptr %p
  ret void
}

define i32 @alloca_load() {
; ALLOCA-LOAD: Entering function: alloca_load
; ALLOCA-LOAD-NEXT:   %p = alloca i32, align 4 => ptr 0x1C [p]
; ALLOCA-LOAD-NEXT:   store i32 0, ptr %p, align 4
; ALLOCA-LOAD-NEXT: Entering function: read
; ALLOCA-LOAD-NEXT:   ptr %p = ptr 0x1C [p (inaccessible)]
; ALLOCA-LOAD-NEXT: Stacktrace:
; ALLOCA-LOAD-NEXT: #0   %v = load i32, ptr %p, align 4 at @read <stdin>:{{[0-9]+}}
; ALLOCA-LOAD-NEXT: #1   %v = tail call i32 @read(ptr %p) at @alloca_load <stdin>:{{[0-9]+}}
; ALLOCA-LOAD-NEXT: Immediate UB detected: Try to access an inaccessible memory object: ptr 0x1C [p (inaccessible)].
; ALLOCA-LOAD-NEXT: error: Execution of function 'alloca_load' failed.
  %p = alloca i32
  store i32 0, ptr %p
  %v = tail call i32 @read(ptr %p)
  ret i32 %v
}

define void @alloca_store() {
; ALLOCA-STORE: Entering function: alloca_store
; ALLOCA-STORE-NEXT:   %p = alloca i32, align 4 => ptr 0x1C [p]
; ALLOCA-STORE-NEXT: Entering function: write
; ALLOCA-STORE-NEXT:   ptr %p = ptr 0x1C [p (inaccessible)]
; ALLOCA-STORE-NEXT: Stacktrace:
; ALLOCA-STORE-NEXT: #0   store i32 0, ptr %p, align 4 at @write <stdin>:{{[0-9]+}}
; ALLOCA-STORE-NEXT: #1   tail call void @write(ptr %p) at @alloca_store <stdin>:{{[0-9]+}}
; ALLOCA-STORE-NEXT: Immediate UB detected: Try to access an inaccessible memory object: ptr 0x1C [p (inaccessible)].
; ALLOCA-STORE-NEXT: error: Execution of function 'alloca_store' failed.
  %p = alloca i32
  tail call void @write(ptr %p)
  ret void
}

define i32 @musttail_read(ptr %p) {
  %v = load i32, ptr %p
  ret i32 %v
}

define i32 @musttail_alloca_load(ptr %unused) {
; MUSTTAIL-LOAD: Entering function: musttail_alloca_load
; MUSTTAIL-LOAD-NEXT:   ptr %unused = ptr 0x0 [nullary]
; MUSTTAIL-LOAD-NEXT:   %p = alloca i32, align 4 => ptr 0x1C [p]
; MUSTTAIL-LOAD-NEXT:   store i32 0, ptr %p, align 4
; MUSTTAIL-LOAD-NEXT: Entering function: musttail_read
; MUSTTAIL-LOAD-NEXT:   ptr %p = ptr 0x1C [p (inaccessible)]
; MUSTTAIL-LOAD-NEXT: Stacktrace:
; MUSTTAIL-LOAD-NEXT: #0   %v = load i32, ptr %p, align 4 at @musttail_read <stdin>:{{[0-9]+}}
; MUSTTAIL-LOAD-NEXT: #1   %v = musttail call i32 @musttail_read(ptr %p) at @musttail_alloca_load <stdin>:{{[0-9]+}}
; MUSTTAIL-LOAD-NEXT: Immediate UB detected: Try to access an inaccessible memory object: ptr 0x1C [p (inaccessible)].
; MUSTTAIL-LOAD-NEXT: error: Execution of function 'musttail_alloca_load' failed.
  %p = alloca i32
  store i32 0, ptr %p
  %v = musttail call i32 @musttail_read(ptr %p)
  ret i32 %v
}

define i32 @outer_byval(ptr byval(i32) %x) {
  %v = tail call i32 @read(ptr %x)
  ret i32 %v
}

define i32 @byval_passed_as_ptr() {
; BYVAL-PTR: Entering function: byval_passed_as_ptr
; BYVAL-PTR-NEXT: Entering function: outer_byval
; BYVAL-PTR-NEXT:   ptr %x = ptr 0x1C [x]
; BYVAL-PTR-NEXT: Entering function: read
; BYVAL-PTR-NEXT:   ptr %p = ptr 0x1C [x (inaccessible)]
; BYVAL-PTR-NEXT: Stacktrace:
; BYVAL-PTR-NEXT: #0   %v = load i32, ptr %p, align 4 at @read <stdin>:{{[0-9]+}}
; BYVAL-PTR-NEXT: #1   %v = tail call i32 @read(ptr %x) at @outer_byval <stdin>:{{[0-9]+}}
; BYVAL-PTR-NEXT: #2   %v = call i32 @outer_byval(ptr byval(i32) @g) at @byval_passed_as_ptr <stdin>:{{[0-9]+}}
; BYVAL-PTR-NEXT: Immediate UB detected: Try to access an inaccessible memory object: ptr 0x1C [x (inaccessible)].
; BYVAL-PTR-NEXT: error: Execution of function 'byval_passed_as_ptr' failed.
  %v = call i32 @outer_byval(ptr byval(i32) @g)
  ret i32 %v
}

define i32 @read_global() {
  %p = load ptr, ptr @gp
  %v = load i32, ptr %p
  ret i32 %v
}

define i32 @mid() {
  %v = call i32 @read_global()
  ret i32 %v
}

define i32 @global_escape_transitive() {
; TRANSITIVE: Entering function: global_escape_transitive
; TRANSITIVE-NEXT:   %p = alloca i32, align 4 => ptr 0x1C [p]
; TRANSITIVE-NEXT:   store i32 0, ptr %p, align 4
; TRANSITIVE-NEXT:   store ptr %p, ptr @gp, align 8
; TRANSITIVE-NEXT: Entering function: mid
; TRANSITIVE-NEXT: Entering function: read_global
; TRANSITIVE-NEXT:   %p = load ptr, ptr @gp, align 8 => ptr 0x1C [p (inaccessible)]
; TRANSITIVE-NEXT: Stacktrace:
; TRANSITIVE-NEXT: #0   %v = load i32, ptr %p, align 4 at @read_global <stdin>:{{[0-9]+}}
; TRANSITIVE-NEXT: #1   %v = call i32 @read_global() at @mid <stdin>:{{[0-9]+}}
; TRANSITIVE-NEXT: #2   %v = tail call i32 @mid() at @global_escape_transitive <stdin>:{{[0-9]+}}
; TRANSITIVE-NEXT: Immediate UB detected: Try to access an inaccessible memory object: ptr 0x1C [p (inaccessible)].
; TRANSITIVE-NEXT: error: Execution of function 'global_escape_transitive' failed.
  %p = alloca i32
  store i32 0, ptr %p
  store ptr %p, ptr @gp
  %v = tail call i32 @mid()
  ret i32 %v
}

define void @noop_deref(ptr dereferenceable(4) %p) {
  ret void
}

define void @dereferenceable_arg() {
; DEREF: Entering function: dereferenceable_arg
; DEREF-NEXT:   %p = alloca i32, align 4 => ptr 0x1C [p]
; DEREF-NEXT: Stacktrace:
; DEREF-NEXT: #0   tail call void @noop_deref(ptr dereferenceable(4) %p) at @dereferenceable_arg <stdin>:{{[0-9]+}}
; DEREF-NEXT: Immediate UB detected: The value ptr 0x1C [p (inaccessible)] violates dereferenceable(4) attribute.
; DEREF-NEXT: error: Execution of function 'dereferenceable_arg' failed.
  %p = alloca i32
  tail call void @noop_deref(ptr dereferenceable(4) %p)
  ret void
}

define void @intrinsic_memcpy() {
; MEMCPY: Entering function: intrinsic_memcpy
; MEMCPY-NEXT:   %p = alloca i32, align 4 => ptr 0x1C [p]
; MEMCPY-NEXT:   store i32 0, ptr %p, align 4
; MEMCPY-NEXT: Stacktrace:
; MEMCPY-NEXT: #0   tail call void @llvm.memcpy.p0.p0.i64(ptr @g, ptr %p, i64 4, i1 false) at @intrinsic_memcpy <stdin>:{{[0-9]+}}
; MEMCPY-NEXT: Immediate UB detected: Try to access an inaccessible memory object: ptr 0x1C [p (inaccessible)].
; MEMCPY-NEXT: error: Execution of function 'intrinsic_memcpy' failed.
  %p = alloca i32
  store i32 0, ptr %p
  tail call void @llvm.memcpy.p0.p0.i64(ptr @g, ptr %p, i64 4, i1 false)
  ret void
}

define void @intrinsic_lifetime_start() {
; LIFETIME-START: Entering function: intrinsic_lifetime_start
; LIFETIME-START-NEXT:   %p = alloca i32, align 4 => ptr 0x1C [p (dead)]
; LIFETIME-START-NEXT: Stacktrace:
; LIFETIME-START-NEXT: #0   tail call void @llvm.lifetime.start.p0(ptr %p) at @intrinsic_lifetime_start <stdin>:{{[0-9]+}}
; LIFETIME-START-NEXT: Immediate UB detected: Try to access an inaccessible memory object: ptr 0x1C [p (inaccessible)].
; LIFETIME-START-NEXT: error: Execution of function 'intrinsic_lifetime_start' failed.
  %p = alloca i32
  tail call void @llvm.lifetime.start.p0(ptr %p)
  ret void
}

define void @intrinsic_lifetime_end() {
; LIFETIME-END: Entering function: intrinsic_lifetime_end
; LIFETIME-END-NEXT:   %p = alloca i32, align 4 => ptr 0x1C [p]
; LIFETIME-END-NEXT: Stacktrace:
; LIFETIME-END-NEXT: #0   tail call void @llvm.lifetime.end.p0(ptr %p) at @intrinsic_lifetime_end <stdin>:{{[0-9]+}}
; LIFETIME-END-NEXT: Immediate UB detected: Try to access an inaccessible memory object: ptr 0x1C [p (inaccessible)].
; LIFETIME-END-NEXT: error: Execution of function 'intrinsic_lifetime_end' failed.
  %p = alloca i32
  tail call void @llvm.lifetime.end.p0(ptr %p)
  ret void
}
