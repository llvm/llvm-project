; RUN: llc -mtriple=x86_64 < %s

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
	
	br label %AllocLoop

AllocLoop:
	%i = phi i32 [ 0, %entry ], [ %indvar.next, %AllocLoop ]
        ;; Allocated mem: allocated memory is immediately dead.
	call ptr @llvm_gc_allocate(i32 100)
	
	%indvar.next = add i32 %i, 1
	%exitcond = icmp eq i32 %indvar.next, 10000000
	br i1 %exitcond, label %Exit, label %AllocLoop

Exit:
	ret i32 0
}

declare void @__main()
