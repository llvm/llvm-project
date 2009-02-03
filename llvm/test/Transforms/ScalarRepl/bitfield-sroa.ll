; RUN: llvm-as < %s | opt -scalarrepl | llvm-dis | not grep alloca        
; rdar://6532315
%t = type { { i32, i16, i8, i8 } }

define i8 @foo(i64 %A) {
        %ALL = alloca %t, align 8 
        %tmp59172 = bitcast %t* %ALL to i64*
        store i64 %A, i64* %tmp59172, align 8
        %C = getelementptr %t* %ALL, i32 0, i32 0, i32 1             
        %D = bitcast i16* %C to i32*    
        %E = load i32* %D, align 4     
        %F = bitcast %t* %ALL to i8* 
        %G = load i8* %F, align 8 
	ret i8 %G
}

