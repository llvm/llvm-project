; PR905
; RUN: llvm-as < %s | opt -instcombine -disable-output
; END.
	
%RPYTHON_EXCEPTION = type { %RPYTHON_EXCEPTION_VTABLE* }
%RPYTHON_EXCEPTION_VTABLE = type { %RPYTHON_EXCEPTION_VTABLE*, i32, i32, %RPyOpaque_RuntimeTypeInfo*, %arraytype_Char*, %functiontype_12* }
%RPyOpaque_RuntimeTypeInfo = type opaque*
%arraytype_Char = type { i32, [0 x i8] }
%fixarray_array1019 = type [1019 x i8*]
%functiontype_12 = type %RPYTHON_EXCEPTION* ()
%functiontype_14 = type void (%structtype_pypy.rpython.memory.gc.MarkSweepGC*)
%structtype_AddressLinkedListChunk = type { %structtype_AddressLinkedListChunk*, i32, %fixarray_array1019 }
%structtype_exceptions.Exception = type { %RPYTHON_EXCEPTION }
%structtype_gc_pool = type {  }
%structtype_gc_pool_node = type { %structtype_header*, %structtype_gc_pool_node* }
%structtype_header = type { i32, %structtype_header* }
%structtype_pypy.rpython.memory.gc.MarkSweepGC = type { %structtype_exceptions.Exception, i32, i32, i1, %structtype_gc_pool*, i32, %structtype_header*, %structtype_header*, %structtype_gc_pool_node*, double, double }

define fastcc void @pypy_MarkSweepGC.collect() {
block0:
	%v1221 = load %structtype_AddressLinkedListChunk** null		; <%structtype_AddressLinkedListChunk*> [#uses=1]
	%v1222 = icmp ne %structtype_AddressLinkedListChunk* %v1221, null		; <i1> [#uses=1]
	br i1 %v1222, label %block79, label %block4

block4:		; preds = %block0
	ret void

block22:		; preds = %block79
	ret void

block67:		; preds = %block79
	%v1459 = load %structtype_gc_pool** null		; <%structtype_gc_pool*> [#uses=1]
	%v1460 = bitcast %structtype_gc_pool* %v1459 to i8*		; <i8*> [#uses=1]
	%tmp_873 = ptrtoint i8* %v1460 to i32		; <i32> [#uses=1]
	%tmp_874 = sub i32 %tmp_873, 0		; <i32> [#uses=1]
	%v1461 = inttoptr i32 %tmp_874 to i8*		; <i8*> [#uses=1]
	%v1462 = bitcast i8* %v1461 to %structtype_header*		; <%structtype_header*> [#uses=1]
	%tmp_876 = getelementptr %structtype_header* %v1462, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 0, i32* %tmp_876
	ret void

block79:		; preds = %block0
	%v1291 = load %structtype_gc_pool** null		; <%structtype_gc_pool*> [#uses=1]
	%v1292 = icmp ne %structtype_gc_pool* %v1291, null		; <i1> [#uses=1]
	br i1 %v1292, label %block67, label %block22
}
