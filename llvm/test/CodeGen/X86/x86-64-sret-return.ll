; RUN: llc -mtriple=x86_64-apple-darwin8 < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-pc-linux < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-pc-linux-gnux32 < %s | FileCheck -check-prefix=X32ABI %s

%struct.foo = type { [4 x i64] }

; CHECK-LABEL: bar:
; CHECK: movq %rdi, %rax

; For the x32 ABI, pointers are 32-bit but passed in zero-extended to 64-bit
; so either 32-bit or 64-bit instructions may be used.
; X32ABI-LABEL: bar:
; X32ABI: mov{{l|q}} %{{r|e}}di, %{{r|e}}ax

define void @bar(ptr noalias sret(%struct.foo)  %agg.result, ptr %d) nounwind  {
entry:
	%d_addr = alloca ptr		; <ptr> [#uses=2]
	%memtmp = alloca %struct.foo, align 8		; <ptr> [#uses=1]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store ptr %d, ptr %d_addr
	%tmp = load ptr, ptr %d_addr, align 8		; <ptr> [#uses=1]
	%tmp1 = getelementptr %struct.foo, ptr %agg.result, i32 0, i32 0		; <ptr> [#uses=4]
	%tmp2 = getelementptr %struct.foo, ptr %tmp, i32 0, i32 0		; <ptr> [#uses=4]
	%tmp3 = getelementptr [4 x i64], ptr %tmp1, i32 0, i32 0		; <ptr> [#uses=1]
	%tmp4 = getelementptr [4 x i64], ptr %tmp2, i32 0, i32 0		; <ptr> [#uses=1]
	%tmp5 = load i64, ptr %tmp4, align 8		; <i64> [#uses=1]
	store i64 %tmp5, ptr %tmp3, align 8
	%tmp6 = getelementptr [4 x i64], ptr %tmp1, i32 0, i32 1		; <ptr> [#uses=1]
	%tmp7 = getelementptr [4 x i64], ptr %tmp2, i32 0, i32 1		; <ptr> [#uses=1]
	%tmp8 = load i64, ptr %tmp7, align 8		; <i64> [#uses=1]
	store i64 %tmp8, ptr %tmp6, align 8
	%tmp9 = getelementptr [4 x i64], ptr %tmp1, i32 0, i32 2		; <ptr> [#uses=1]
	%tmp10 = getelementptr [4 x i64], ptr %tmp2, i32 0, i32 2		; <ptr> [#uses=1]
	%tmp11 = load i64, ptr %tmp10, align 8		; <i64> [#uses=1]
	store i64 %tmp11, ptr %tmp9, align 8
	%tmp12 = getelementptr [4 x i64], ptr %tmp1, i32 0, i32 3		; <ptr> [#uses=1]
	%tmp13 = getelementptr [4 x i64], ptr %tmp2, i32 0, i32 3		; <ptr> [#uses=1]
	%tmp14 = load i64, ptr %tmp13, align 8		; <i64> [#uses=1]
	store i64 %tmp14, ptr %tmp12, align 8
	%tmp15 = getelementptr %struct.foo, ptr %memtmp, i32 0, i32 0		; <ptr> [#uses=4]
	%tmp16 = getelementptr %struct.foo, ptr %agg.result, i32 0, i32 0		; <ptr> [#uses=4]
	%tmp17 = getelementptr [4 x i64], ptr %tmp15, i32 0, i32 0		; <ptr> [#uses=1]
	%tmp18 = getelementptr [4 x i64], ptr %tmp16, i32 0, i32 0		; <ptr> [#uses=1]
	%tmp19 = load i64, ptr %tmp18, align 8		; <i64> [#uses=1]
	store i64 %tmp19, ptr %tmp17, align 8
	%tmp20 = getelementptr [4 x i64], ptr %tmp15, i32 0, i32 1		; <ptr> [#uses=1]
	%tmp21 = getelementptr [4 x i64], ptr %tmp16, i32 0, i32 1		; <ptr> [#uses=1]
	%tmp22 = load i64, ptr %tmp21, align 8		; <i64> [#uses=1]
	store i64 %tmp22, ptr %tmp20, align 8
	%tmp23 = getelementptr [4 x i64], ptr %tmp15, i32 0, i32 2		; <ptr> [#uses=1]
	%tmp24 = getelementptr [4 x i64], ptr %tmp16, i32 0, i32 2		; <ptr> [#uses=1]
	%tmp25 = load i64, ptr %tmp24, align 8		; <i64> [#uses=1]
	store i64 %tmp25, ptr %tmp23, align 8
	%tmp26 = getelementptr [4 x i64], ptr %tmp15, i32 0, i32 3		; <ptr> [#uses=1]
	%tmp27 = getelementptr [4 x i64], ptr %tmp16, i32 0, i32 3		; <ptr> [#uses=1]
	%tmp28 = load i64, ptr %tmp27, align 8		; <i64> [#uses=1]
	store i64 %tmp28, ptr %tmp26, align 8
	br label %return

return:		; preds = %entry
	ret void
}

; CHECK-LABEL: foo:
; CHECK: movq %rdi, %rax

; For the x32 ABI, pointers are 32-bit but passed in zero-extended to 64-bit
; so either 32-bit or 64-bit instructions may be used.
; X32ABI-LABEL: foo:
; X32ABI: mov{{l|q}} %{{r|e}}di, %{{r|e}}ax

define void @foo(ptr noalias nocapture sret({ i64 }) %agg.result) nounwind {
  store { i64 } { i64 0 }, ptr %agg.result
  ret void
}
