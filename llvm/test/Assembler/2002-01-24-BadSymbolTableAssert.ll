; RUN: llvm-as %s -o /dev/null -f

; This testcase failed due to a bad assertion in SymbolTable.cpp, removed in
; the 1.20 revision. Basically the symbol table assumed that if there was an
; abstract type in the symbol table, [in this case for the entry %foo of type
; void(opaque)* ], that there should have also been named types by now.  This
; was obviously not the case here, and this is valid.  Assertion disabled.
	
%bb = type i32

declare void @foo(i32)
