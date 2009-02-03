; RUN: llvm-as < %s | llc | grep caml.*__frametable
; RUN: llvm-as < %s | llc -march=x86 | grep {movl	.0}

%struct.obj = type { i8*, %struct.obj* }

define %struct.obj* @fun(%struct.obj* %head) gc "ocaml" {
entry:
	%gcroot.0 = alloca i8*
	%gcroot.1 = alloca i8*
	
	call void @llvm.gcroot(i8** %gcroot.0, i8* null)
	call void @llvm.gcroot(i8** %gcroot.1, i8* null)
	
	%local.0 = bitcast i8** %gcroot.0 to %struct.obj**
	%local.1 = bitcast i8** %gcroot.1 to %struct.obj**

	store %struct.obj* %head, %struct.obj** %local.0
	br label %bb.loop
bb.loop:
	%t0 = load %struct.obj** %local.0
	%t1 = getelementptr %struct.obj* %t0, i32 0, i32 1
	%t2 = bitcast %struct.obj* %t0 to i8*
	%t3 = bitcast %struct.obj** %t1 to i8**
	%t4 = call i8* @llvm.gcread(i8* %t2, i8** %t3)
	%t5 = bitcast i8* %t4 to %struct.obj*
	%t6 = icmp eq %struct.obj* %t5, null
	br i1 %t6, label %bb.loop, label %bb.end
bb.end:
	%t7 = malloc %struct.obj
	store %struct.obj* %t7, %struct.obj** %local.1
	%t8 = bitcast %struct.obj* %t7 to i8*
	%t9 = load %struct.obj** %local.0
	%t10 = getelementptr %struct.obj* %t9, i32 0, i32 1
	%t11 = bitcast %struct.obj* %t9 to i8*
	%t12 = bitcast %struct.obj** %t10 to i8**
	call void @llvm.gcwrite(i8* %t8, i8* %t11, i8** %t12)
	ret %struct.obj* %t7
}

declare void @llvm.gcroot(i8** %value, i8* %tag)
declare void @llvm.gcwrite(i8* %value, i8* %obj, i8** %field)
declare i8* @llvm.gcread(i8* %obj, i8** %field)
