; RUN: llvm-diff %s %s

; An initializer that has a GEP instruction in it won't match itself in
; llvm-diff unless the a deep comparison is done on the initializer.

@gv1 = external dso_local global [28 x i16], align 16
@gv2 = private unnamed_addr constant [2 x ptr] [ptr @gv1, ptr poison], align 16

define void @foo() {
  %1 = getelementptr [2 x ptr], ptr @gv2, i64 0, i64 undef
  ret void
}

; A named structure may be renamed when the right module is read. This is due
; to the LLParser being different between the left and right modules, and the
; context renaming one.

%struct.ty1 = type { i16, i16 }

@gv3 = internal global [1 x %struct.ty1] [%struct.ty1 { i16 928, i16 0 }], align 16

define void @bar() {
  %1 = getelementptr [1 x %struct.ty1], ptr @gv3, i64 0, i64 undef
  ret void
}

; An initializer may reference the variable it's initializing via bitcast /
; GEP. Check that it doesn't cause an infinite loop.

%struct.mutex = type { %struct.list_head }
%struct.list_head = type { ptr, ptr }

@vmx_l1d_flush_mutex = internal global %struct.mutex { %struct.list_head { ptr getelementptr (i8, ptr @vmx_l1d_flush_mutex, i64 16), ptr getelementptr (i8, ptr @vmx_l1d_flush_mutex, i64 16) } }, align 8

define internal i32 @qux() {
  call void undef(ptr @vmx_l1d_flush_mutex)
  ret i32 undef
}

; An initializer could use itself as part of the initialization.

@kvm_debugfs_entries = internal global %struct.list_head { ptr @kvm_debugfs_entries, ptr @kvm_debugfs_entries }, align 8

define i64 @mux() {
  %1 = load ptr, ptr @kvm_debugfs_entries, align 8
  ret i64 undef
}
