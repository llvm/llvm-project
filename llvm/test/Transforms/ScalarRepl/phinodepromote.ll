; RUN: opt < %s -simplifycfg -instcombine -mem2reg -S | not grep alloca
;
; This tests to see if mem2reg can promote alloca instructions whose addresses
; are used by PHI nodes that are immediately loaded.  The LLVM C++ front-end
; often generates code that looks like this (when it codegen's ?: exprs as
; lvalues), so handling this simple extension is quite useful.
;
; This testcase is what the following program looks like when it reaches
; instcombine:
;
; template<typename T>
; const T& max(const T& a1, const T& a2) { return a1 < a2 ? a1 : a2; }
; int main() { return max(0, 1); }
;
; This test checks to make sure the combination of instcombine and mem2reg
; perform the transformation.

define i32 @main() {
entry:
	%mem_tmp.0 = alloca i32		; <i32*> [#uses=3]
	%mem_tmp.1 = alloca i32		; <i32*> [#uses=3]
	store i32 0, i32* %mem_tmp.0
	store i32 1, i32* %mem_tmp.1
	%tmp.1.i = load i32* %mem_tmp.1		; <i32> [#uses=1]
	%tmp.3.i = load i32* %mem_tmp.0		; <i32> [#uses=1]
	%tmp.4.i = icmp sle i32 %tmp.1.i, %tmp.3.i		; <i1> [#uses=1]
	br i1 %tmp.4.i, label %cond_true.i, label %cond_continue.i
cond_true.i:		; preds = %entry
	br label %cond_continue.i
cond_continue.i:		; preds = %cond_true.i, %entry
	%mem_tmp.i.0 = phi i32* [ %mem_tmp.1, %cond_true.i ], [ %mem_tmp.0, %entry ]		; <i32*> [#uses=1]
	%tmp.3 = load i32* %mem_tmp.i.0		; <i32> [#uses=1]
	ret i32 %tmp.3
}
