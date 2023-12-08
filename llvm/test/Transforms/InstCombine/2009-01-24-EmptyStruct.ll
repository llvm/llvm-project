; RUN: opt < %s -passes=instcombine
; PR3381
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"
	%struct.atomic_t = type { i32 }
	%struct.inode = type { i32, %struct.mutex }
	%struct.list_head = type { ptr, ptr }
	%struct.lock_class_key = type {  }
	%struct.mutex = type { %struct.atomic_t, %struct.rwlock_t, %struct.list_head }
	%struct.rwlock_t = type { %struct.lock_class_key }

define void @handle_event(ptr %bar) nounwind {
entry:
	%0 = getelementptr %struct.inode, ptr %bar, i64 -1, i32 1, i32 1		; <ptr> [#uses=1]
	store i32 1, ptr %0, align 4
	ret void
}
