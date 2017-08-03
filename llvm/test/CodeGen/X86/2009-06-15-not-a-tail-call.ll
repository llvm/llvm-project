; RUN: llc < %s -march=x86 -tailcallopt | not grep TAILCALL 

; Bug 4396. This tail call can NOT be optimized.

declare fastcc i8* @_D3gcx2GC12mallocNoSyncMFmkZPv() nounwind

define fastcc i8* @_D3gcx2GC12callocNoSyncMFmkZPv() nounwind {
entry:
	%tmp6 = tail call fastcc i8* @_D3gcx2GC12mallocNoSyncMFmkZPv()		; <i8*> [#uses=2]
	%tmp9 = tail call i8* @memset(i8* %tmp6, i32 0, i64 2)		; <i8*> [#uses=0]
	ret i8* %tmp6
}

declare i8* @memset(i8*, i32, i64)
