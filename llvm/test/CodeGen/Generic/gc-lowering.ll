; RUN: opt -S -passes='require<collector-metadata>,function(gc-lowering)' < %s | FileCheck %s

declare ptr @llvm_gc_allocate(i32)
declare void @llvm_gc_initialize(i32)

declare void @llvm.gcroot(ptr, ptr)
declare void @llvm.gcwrite(ptr, ptr, ptr)

define i32 @main() gc "shadow-stack" {
entry:
	%A = alloca ptr
	%B = alloca ptr

	call void @llvm_gc_initialize(i32 1048576)  ; Start with 1MB heap

        ;; ptr A;
	call void @llvm.gcroot(ptr %A, ptr null)

        ;; A = gcalloc(10);
	%Aptr = call ptr @llvm_gc_allocate(i32 10)
	store ptr %Aptr, ptr %A

        ;; ptr B;
	call void @llvm.gcroot(ptr %B, ptr null)

	;; B = gcalloc(4);
	%B.upgrd.1 = call ptr @llvm_gc_allocate(i32 8)
	store ptr %B.upgrd.1, ptr %B

	;; *B = A;
	%B.1 = load ptr, ptr %B
	%A.1 = load ptr, ptr %A
	call void @llvm.gcwrite(ptr %A.1, ptr %B.upgrd.1, ptr %B.1)
	; CHECK: 			%A.1 = load ptr, ptr %A, align 8
	; CHECK-NEXT: store ptr %A.1, ptr %B.1, align 8

  ret i32 0
}

define void @no_gc() {
	ret void
}
