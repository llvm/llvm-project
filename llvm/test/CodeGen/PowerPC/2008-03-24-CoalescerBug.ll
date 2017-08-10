; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32--

	%struct..0objc_object = type { %struct.objc_class* }
	%struct.NSArray = type { %struct..0objc_object }
	%struct.NSMutableArray = type { %struct.NSArray }
	%struct.PFTPersistentSymbols = type { %struct..0objc_object, %struct.VMUSymbolicator*, %struct.NSMutableArray*, %struct.__CFDictionary*, %struct.__CFDictionary*, %struct.__CFDictionary*, %struct.__CFDictionary*, %struct.NSMutableArray*, i8, %struct.pthread_mutex_t, %struct.NSMutableArray*, %struct.pthread_rwlock_t }
	%struct.VMUMachTaskContainer = type { %struct..0objc_object, i32, i32 }
	%struct.VMUSymbolicator = type { %struct..0objc_object, %struct.NSMutableArray*, %struct.NSArray*, %struct.NSArray*, %struct.VMUMachTaskContainer*, i8 }
	%struct.__CFDictionary = type opaque
	%struct.__builtin_CFString = type { i32*, i32, i8*, i32 }
	%struct.objc_class = type opaque
	%struct.objc_selector = type opaque
	%struct.pthread_mutex_t = type { i32, [40 x i8] }
	%struct.pthread_rwlock_t = type { i32, [124 x i8] }
@0 = external constant %struct.__builtin_CFString		; <%struct.__builtin_CFString*>:0 [#uses=1]

define void @"-[PFTPersistentSymbols saveSymbolWithName:address:path:lineNumber:flags:owner:]"(%struct.PFTPersistentSymbols* %self, %struct.objc_selector* %_cmd, %struct.NSArray* %name, i64 %address, %struct.NSArray* %path, i32 %lineNumber, i64 %flags, %struct..0objc_object* %owner) nounwind  {
entry:
	br i1 false, label %bb12, label %bb21
bb12:		; preds = %entry
	%tmp17 = tail call signext i8 inttoptr (i64 4294901504 to i8 (%struct..0objc_object*, %struct.objc_selector*, %struct.NSArray*)*)( %struct..0objc_object* null, %struct.objc_selector* null, %struct.NSArray* bitcast (%struct.__builtin_CFString* @0 to %struct.NSArray*) )  nounwind 		; <i8> [#uses=0]
	br i1 false, label %bb25, label %bb21
bb21:		; preds = %bb12, %entry
	%tmp24 = or i64 %flags, 4		; <i64> [#uses=1]
	br label %bb25
bb25:		; preds = %bb21, %bb12
	%flags_addr.0 = phi i64 [ %tmp24, %bb21 ], [ %flags, %bb12 ]		; <i64> [#uses=1]
	%tmp3233 = trunc i64 %flags_addr.0 to i32		; <i32> [#uses=0]
	ret void
}
