; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32--

	%struct..0objc_object = type { ptr }
	%struct.NSArray = type { %struct..0objc_object }
	%struct.NSMutableArray = type { %struct.NSArray }
	%struct.PFTPersistentSymbols = type { %struct..0objc_object, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i8, %struct.pthread_mutex_t, ptr, %struct.pthread_rwlock_t }
	%struct.VMUMachTaskContainer = type { %struct..0objc_object, i32, i32 }
	%struct.VMUSymbolicator = type { %struct..0objc_object, ptr, ptr, ptr, ptr, i8 }
	%struct.__CFDictionary = type opaque
	%struct.__builtin_CFString = type { ptr, i32, ptr, i32 }
	%struct.objc_class = type opaque
	%struct.objc_selector = type opaque
	%struct.pthread_mutex_t = type { i32, [40 x i8] }
	%struct.pthread_rwlock_t = type { i32, [124 x i8] }
@0 = external constant %struct.__builtin_CFString		; <ptr>:0 [#uses=1]

define void @"-[PFTPersistentSymbols saveSymbolWithName:address:path:lineNumber:flags:owner:]"(ptr %self, ptr %_cmd, ptr %name, i64 %address, ptr %path, i32 %lineNumber, i64 %flags, ptr %owner) nounwind  {
entry:
	br i1 false, label %bb12, label %bb21
bb12:		; preds = %entry
	%tmp17 = tail call signext i8 inttoptr (i64 4294901504 to ptr)( ptr null, ptr null, ptr @0 )  nounwind 		; <i8> [#uses=0]
	br i1 false, label %bb25, label %bb21
bb21:		; preds = %bb12, %entry
	%tmp24 = or i64 %flags, 4		; <i64> [#uses=1]
	br label %bb25
bb25:		; preds = %bb21, %bb12
	%flags_addr.0 = phi i64 [ %tmp24, %bb21 ], [ %flags, %bb12 ]		; <i64> [#uses=1]
	%tmp3233 = trunc i64 %flags_addr.0 to i32		; <i32> [#uses=0]
	ret void
}
